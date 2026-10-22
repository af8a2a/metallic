#include "RHITest.h"
#include "TestComputeProgram.h"
#include "TestResourceParameters.h"
#include "harness/RayQueryFixture.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamGPUCLASPool.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <json.hpp>

namespace metallic::tests {
namespace {
using namespace render;
constexpr uint32_t kSmallPageCount = 64;

template<typename T>
T checked(Result<T> result)
{
    if (!result) { throw std::runtime_error(resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

void require(bool condition, const char* reason)
{
    if (!condition) { throw std::runtime_error(reason); }
}

std::unique_ptr<Buffer> makeBuffer(Device& device, uint64_t bytes, MemoryLocation location = MemoryLocation::Device)
{
    return checked(device.createBuffer({.size = bytes,
        .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
        .memoryLocation = location}));
}

template<typename T>
void upload(Buffer& buffer, const T& value)
{
    void* data = buffer.map();
    require(data != nullptr, "Upload map failed");
    std::memcpy(data, &value, sizeof(value));
    buffer.flush(); buffer.unmap();
}

template<typename T>
T read(Buffer& buffer)
{
    buffer.invalidate();
    const void* data = buffer.map();
    require(data != nullptr, "Readback map failed");
    T value;
    std::memcpy(&value, data, sizeof(value)); buffer.unmap();
    return value;
}

// Independent single-triangle pages plus a larger grid page. The small
// pages have identical encoded size, making holes and exact-fit budgets precise.
void writeFixture(const std::filesystem::path& directory)
{
    nlohmann::json document{{"asset", {{"version", "2.0"}}}, {"scene", 0},
        {"scenes", nlohmann::json::array({{{"nodes", nlohmann::json::array()}}})}};
    document["nodes"] = nlohmann::json::array();
    document["meshes"] = nlohmann::json::array();
    document["bufferViews"] = nlohmann::json::array();
    document["accessors"] = nlohmann::json::array();
    std::ofstream binary(directory / "mesh.bin", std::ios::binary);
    uint32_t offset = 0;
    auto append = [&](const auto& values, uint32_t componentType, const char* type, uint32_t count,
                      nlohmann::json bounds = {}) {
        const uint32_t bytes = uint32_t(values.size() * sizeof(values[0]));
        const uint32_t accessor = uint32_t(document["accessors"].size());
        document["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", offset}, {"byteLength", bytes}});
        nlohmann::json entry{{"bufferView", accessor}, {"componentType", componentType}, {"type", type}, {"count", count}};
        if (!bounds.is_null()) { entry.update(bounds); }
        document["accessors"].push_back(std::move(entry));
        binary.write(reinterpret_cast<const char*>(values.data()), bytes);
        offset += bytes;
        return accessor;
    };
    const auto positions = append(bench::kRayVertices, 5126, "VEC3", 3,
        {{"min", {0, 0, 2}}, {"max", {1, 1, 2}}});
    const auto normals = append(std::array<float, 9>{0, 0, 1, 0, 0, 1, 0, 0, 1}, 5126, "VEC3", 3);
    const auto indices = append(std::array<uint32_t, 3>{0, 1, 2}, 5125, "SCALAR", 3);
    for (uint32_t mesh = 0; mesh < kSmallPageCount; ++mesh) {
        document["nodes"].push_back({{"mesh", mesh}});
        document["scenes"][0]["nodes"].push_back(mesh);
        document["meshes"].push_back({{"primitives", nlohmann::json::array({{
            {"attributes", {{"POSITION", positions}, {"NORMAL", normals}}}, {"indices", indices}}})}});
    }
    std::vector<float> grid, gridNormals;
    std::vector<uint32_t> gridIndices;
    for (uint32_t y = 0; y < 16; ++y) {
        for (uint32_t x = 0; x < 16; ++x) {
            grid.insert(grid.end(), {float(x), float(y), 2.f});
            gridNormals.insert(gridNormals.end(), {0.f, 0.f, 1.f});
            if (x < 15 && y < 15) {
                const uint32_t a = y * 16 + x;
                gridIndices.insert(gridIndices.end(), {a, a + 1, a + 16, a + 1, a + 17, a + 16});
            }
        }
    }
    const auto gridPosition = append(grid, 5126, "VEC3", 256, {{"min", {0, 0, 2}}, {"max", {15, 15, 2}}});
    const auto gridNormal = append(gridNormals, 5126, "VEC3", 256);
    const auto gridIndex = append(gridIndices, 5125, "SCALAR", uint32_t(gridIndices.size()));
    document["nodes"].push_back({{"mesh", kSmallPageCount}});
    document["scenes"][0]["nodes"].push_back(kSmallPageCount);
    document["meshes"].push_back({{"primitives", nlohmann::json::array({{
        {"attributes", {{"POSITION", gridPosition}, {"NORMAL", gridNormal}}}, {"indices", gridIndex}}})}});
    document["buffers"] = nlohmann::json::array({{{"uri", "mesh.bin"}, {"byteLength", offset}}});
    std::ofstream(directory / "scene.gltf") << document.dump(2);
}

struct Slot {
    explicit Slot(uint32_t index) : frame(index) {}
    RenderFrameContext frame;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    std::unique_ptr<Buffer> table, addresses, rays;
};

class GPUCLASAllocatorTest final : public RHITest {
public:
    GPUCLASAllocatorTest()
    {
        type = RHITestType::Rendering;
        name = "clas_gpu_allocator_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::atomic_uint validationErrors = 0;
        auto created = createDevice({.applicationName = "GPU CLAS allocator lifecycle",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
            .enableClusterAccelerationStructure = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}});
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("CLAS/ray query unavailable")
                : RHITestResult::fail(resultToString(created));
        }
        auto device = std::move(*created);
        std::string phase = "fixture";
        try {
            const auto directory = std::filesystem::absolute(context.outputDirectory / "GPUCLASAllocator");
            std::filesystem::create_directories(directory);
            writeFixture(directory);
            scene::Scene source;
            require(source.load(directory / "scene.gltf"), "Fixture import failed");
            std::string log;
            const auto assetPath = directory / "scene.meshstream.bin";
            require(scene::buildMeshletStreamAsset({.scene = &source, .sourcePath = directory / "scene.gltf",
                .outputPath = assetPath, .compressionMode = scene::MeshletStreamPayloadCompression::ByteRle}, log), log.c_str());
            scene::MeshletStreamAsset asset;
            require(asset.open(assetPath, log), log.c_str());
            std::array<uint32_t, kSmallPageCount> smallPages;
            smallPages.fill(UINT32_MAX);
            uint32_t largePage = UINT32_MAX;
            std::vector<MeshletStreamCLASPagePlan> plans(asset.pageCount());
            std::vector<uint64_t> geometryOffsets(asset.pageCount());
            std::vector<uint8_t> geometry;
            for (uint32_t page = 0; page < asset.pageCount(); ++page) {
                std::vector<uint8_t> decoded;
                std::span<const uint8_t> payload;
                require(scene::decodeMeshletStreamPayloadForDevice(asset.pages()[page], asset.pagePayload(page),
                    decoded, payload, log), log.c_str());
                require(buildMeshletStreamClasPagePlan(asset.pages()[page], payload, page,
                    page * asset.maxPageClusters(), plans[page], log), log.c_str());
                geometry.resize((geometry.size() + 255) / 256 * 256);
                geometryOffsets[page] = geometry.size();
                geometry.insert(geometry.end(), payload.begin(), payload.end());
                const uint32_t primitive = asset.pages()[page].primitiveIndex;
                if (primitive < kSmallPageCount && plans[page].clusters.size() == 1 && plans[page].clusters[0].triangleCount == 1) {
                    smallPages[primitive] = page;
                }
                if (primitive == kSmallPageCount && plans[page].clusters.size() >= 2 &&
                    (largePage == UINT32_MAX || plans[page].clusters.size() < plans[largePage].clusters.size())) { largePage = page; }
            }
            require(std::all_of(smallPages.begin(), smallPages.end(), [](uint32_t page) { return page != UINT32_MAX; }) &&
                largePage != UINT32_MAX, "Need independent single-cluster pages and a multi-cluster page");
            auto pageBuffer = makeBuffer(*device, geometry.size(), MemoryLocation::HostUpload);
            void* geometryMap = pageBuffer->map();
            require(geometryMap != nullptr, "Geometry map failed");
            std::memcpy(geometryMap, geometry.data(), geometry.size()); pageBuffer->flush(); pageBuffer->unmap();
            auto request = [&](uint32_t page) {
                return MeshletStreamCLASPageBuild{.pageIndex = page, .deviceOffsetBytes = geometryOffsets[page], .plan = &plans[page]};
            };

            const auto properties = checked(device->queryClusterAccelerationStructureProperties());
            const auto blasSizes = checked(device->queryClusterAccelerationStructureBottomLevelBuildSizes({
                .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1}));
            auto blas = makeBuffer(*device, blasSizes.accelerationStructureSize + properties.bottomLevelStorageAlignment);
            const uint64_t blasOffset = (0 - blas->deviceAddress()) & (properties.bottomLevelStorageAlignment - 1);
            auto blasAddress = makeBuffer(*device, 8, MemoryLocation::HostUpload);
            upload(*blasAddress, blas->deviceAddress() + blasOffset);
            auto blasInfo = makeBuffer(*device, sizeof(ClusterAccelerationStructureBottomLevelBuildInfo), MemoryLocation::HostUpload);
            const auto tlasSizes = checked(device->queryRayTracingAccelerationStructureBuildSizes({
                .type = RayTracingAccelerationStructureType::TopLevel, .instanceCount = 1}));
            auto tlas = checked(device->createRayTracingAccelerationStructure({
                .type = RayTracingAccelerationStructureType::TopLevel, .size = tlasSizes.accelerationStructureSize}));
            const auto rtProperties = checked(device->queryRayTracingAccelerationStructureProperties());
            auto scratch = makeBuffer(*device, std::max(blasSizes.buildScratchSize, tlasSizes.buildScratchSize) +
                std::max(properties.scratchAlignment, rtProperties.scratchAlignment));
            auto instances = makeBuffer(*device, sizeof(RayTracingGPUInstance), MemoryLocation::HostUpload);
            RayTracingGPUInstance instance;
            instance.customIndexAndMask = 37 | (1u << 24);
            instance.shaderBindingTableRecordOffsetAndFlags = uint32_t(RayTracingInstanceFlags::TriangleFacingCullDisable) << 24;
            instance.accelerationStructureReference = blas->deviceAddress() + blasOffset;
            upload(*instances, instance);
            const char* capabilities[]{"spvRayQueryKHR"};
            const auto shader = checked(compileSlangShaderToSpirv({.moduleName = "UnifiedTopLevelProbe",
                .entryPointName = "unifiedTopLevelMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .capabilities = capabilities}, log));
            const ComputeResourceBindingDesc layout[]{
                {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), ComputeResourceBindingKind::AccelerationStructure},
                {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output)}};
            ComputeProgram probe;
            checked(probe.initialize(*device, {.spirv = shader.spirv, .bindings = layout,
                .resourceParameterSize = sizeof(UnifiedTopLevelProbeResources)}, log));

            auto* queue = device->getQueue(QueueType::Graphics);
            QueueSubmissionTracker tracker;
            checked(tracker.initialize(*device, *queue));
            auto pool = std::make_unique<MeshletStreamGPUCLASPool>();
            checked(pool->initialize(*device, {.asset = &asset, .maxStorageBytes = 1ull << 20,
                .maxBuildClusters = asset.maxPageClusters() * (kSmallPageCount + 1), .queuedFrameCount = 2,
                .persistentPages = {&smallPages[0], 1}}, log));
            std::array<Slot, 2> slots{Slot{0}, Slot{1}};
            for (auto& slot : slots) {
                slot.pool = checked(device->createCommandPool(*queue));
                slot.commands = checked(slot.pool->createCommandBuffer());
                slot.table = makeBuffer(*device, uint64_t(asset.pageCount()) * sizeof(MeshletStreamCLASPageEntry), MemoryLocation::HostReadback);
                slot.addresses = makeBuffer(*device, pool->clusterAddressBuffer()->desc().size, MemoryLocation::HostReadback);
                slot.rays = makeBuffer(*device, sizeof(bench::RayObservations), MemoryLocation::HostReadback);
            }
            struct Drain {
                Queue& queue; std::array<Slot, 2>& slots;
                ~Drain() { for (auto& slot : slots) { slot.frame.cancel(); } (void)queue.waitIdle(); }
            } drain{*queue, slots};
            uint64_t frameIndex = 0;
            nlohmann::json evidence = nlohmann::json::array();
            auto copy = [&](CommandBuffer& commands, Buffer& sourceBuffer, Buffer& destination) {
                BufferBarrierDesc barrier{.buffer = &sourceBuffer,
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                checked(commands.synchronize({.buffers = {&barrier, 1}}));
                checked(commands.copyBuffer(checked(sourceBuffer.slice()), checked(destination.slice({0, sourceBuffer.desc().size}))));
                std::swap(barrier.before, barrier.after);
                checked(commands.synchronize({.buffers = {&barrier, 1}}));
            };
            auto begin = [&](Slot& slot, std::span<const MeshletStreamCLASPageBuild> requests, bool buildRayScene, bool trace) {
                checked(slot.frame.begin(++frameIndex)); checked(slot.pool->reset());
                checked(slot.commands->begin(slot.frame.submissionContext()));
                checked(pool->cmdBuildPages(*slot.commands, *pageBuffer, requests, log));
                require(pool->publicationRevision() == pool->stats().publicationRevision,
                    "Recorded publication revision differs from pool statistics");
                if (buildRayScene) {
                    // This is the address of the GPU table slot, never a CPU-read CLAS address.
                    const auto offset = pool->pageClasAddressOffset(smallPages[0]);
                    require(offset != kInvalidMeshletStreamCLASAddressOffset, "Recorded page has no address-table slot");
                    upload(*blasInfo, ClusterAccelerationStructureBottomLevelBuildInfo{.clusterReferencesCount = 1,
                        .clusterReferencesAddress = pool->clusterAddressBuffer()->deviceAddress() + uint64_t(offset) * 8});
                    checked(slot.commands->retainResource(blas->retainAllocation()));
                    checked(slot.commands->buildClusterAccelerationStructureBottomLevels({
                        .destinationMode = ClusterAccelerationStructureDestinationMode::Explicit,
                        .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1,
                        .buildInfoBuffer = checked(blasInfo->slice()), .destinationAddressBuffer = checked(blasAddress->slice()),
                        .scratchBuffer = checked(scratch->slice())}));
                    checked(slot.commands->buildRayTracingAccelerationStructure({.destination = tlas.get(),
                        .instanceBuffer = checked(instances->slice()), .instanceCount = 1,
                        .scratchBuffer = checked(scratch->slice())}));
                }
                if (trace) {
                    const ComputeDispatchBinding bindings[]{
                        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), .accelerationStructure = tlas.get()},
                        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output), .buffer = slot.rays.get()}};
                    checked(probe.dispatch({.commandBuffer = slot.commands.get(), .bindings = bindings}));
                }
                copy(*slot.commands, *pool->pageTableBuffer(), *slot.table);
                copy(*slot.commands, *pool->clusterAddressBuffer(), *slot.addresses);
                checked(slot.commands->end());
            };
            auto submit = [&](Slot& slot, bool wait = true) {
                CommandBuffer* submitted[]{slot.commands.get()};
                checked(tracker.submit({.commandBuffers = submitted}, slot.frame));
                if (wait) { checked(slot.frame.wait(5'000'000'000ull)); }
            };
            auto entry = [&](Slot& slot, uint32_t page) {
                slot.table->invalidate();
                const auto* mapped = static_cast<const MeshletStreamCLASPageEntry*>(slot.table->map());
                require(mapped != nullptr, "Page table readback map failed");
                const auto value = mapped[page]; slot.table->unmap(); return value;
            };
            auto address = [&](Slot& slot, uint32_t page) {
                const auto state = entry(slot, page);
                if (meshletStreamClasPageState(state) == MeshletStreamCLASPageState::Empty) { return uint64_t(0); }
                slot.addresses->invalidate();
                const auto* mapped = static_cast<const uint64_t*>(slot.addresses->map());
                require(mapped != nullptr, "Address readback map failed");
                const auto value = mapped[meshletStreamClasPageAddressOffset(state)]; slot.addresses->unmap(); return value;
            };
            auto observe = [&](Slot& slot, bool trace) {
                const auto state = entry(slot, smallPages[0]);
                nlohmann::json sample{{"frame", frameIndex}, {"phase", phase},
                    {"state", uint32_t(meshletStreamClasPageState(state))}, {"generation", state.publicationGeneration},
                    {"address", address(slot, smallPages[0])}, {"builtPages", pool->stats().totalBuiltPageCount},
                    {"usedBytes", pool->stats().usedStorageBytes}};
                if (trace) { sample["rays"] = bench::rayOracle(read<bench::RayObservations>(*slot.rays)); }
                evidence.push_back(std::move(sample));
            };
            auto checkSinglePageStats = [&](uint32_t active, uint32_t retiring, uint64_t bytes, uint64_t encoded) {
                const auto stats = pool->stats();
                require(stats.builtPageCount == active && stats.builtClusterCount == active &&
                    stats.retiringPageCount == retiring && stats.retiringClusterCount == retiring,
                    "Single-page active/retiring statistics disagree with residency");
                require(stats.usedStorageBytes == bytes && stats.encodedStorageBytes == encoded &&
                    stats.worstCaseStorageBytes == uint64_t(active + retiring) * stats.clusterStrideBytes &&
                    stats.retiringStorageBytes == (retiring ? bytes : 0),
                    "Single-page byte statistics disagree with residency");
                require(stats.persistentUsedBytes == bytes && stats.transientUsedBytes == 0,
                    "Single-page persistent/transient statistics disagree with residency");
            };
            const auto first = request(smallPages[0]);
            phase = "cancel initial Build Allocate Move Publish";
            pool->beginFrame(); begin(slots[0], {&first, 1}, true, true);
            checkSinglePageStats(0, 0, 0, 0);
            slots[0].frame.cancel();
            pool->beginFrame();
            require(!pool->pageBuildPending(smallPages[0]) && !pool->pageHasClas(smallPages[0]), "Cancelled first build became resident");
            require(pool->publicationRevision() == pool->stats().publicationRevision,
                "Cancelled publication revision differs from pool statistics");
            require(pool->stats().totalBuiltPageCount == 0 && pool->stats().usedStorageBytes == 0,
                "Cancelled first build consumed GPU allocation/accounting");
            checkSinglePageStats(0, 0, 0, 0);
            phase = "first submission includes ray query";
            begin(slots[0], {&first, 1}, true, true); submit(slots[0]); observe(slots[0], true);
            require(meshletStreamClasPageState(entry(slots[0], smallPages[0])) == MeshletStreamCLASPageState::Active,
                "First submission did not publish the page");
            const uint64_t firstAddress = address(slots[0], smallPages[0]);
            require(firstAddress != 0, "First submission did not emit a CLAS address");
            pool->beginFrame();
            const auto firstStats = pool->stats();
            require(firstStats.totalBuiltPageCount == 1, "First successful submission must build exactly one page");
            const uint64_t smallBytes = pool->pageStorageBytes(smallPages[0]);
            require(smallBytes > 0, "Completed page has no encoded allocation size");
            const uint64_t smallEncodedBytes = firstStats.encodedStorageBytes;
            require(smallEncodedBytes > 0 && smallEncodedBytes <= smallBytes, "Invalid encoded CLAS size statistics");
            checkSinglePageStats(1, 0, smallBytes, smallEncodedBytes);
            phase = "32 stable frames";
            for (uint32_t frame = 0; frame < 32; ++frame) {
                pool->beginFrame(); begin(slots[frame % 2], {&first, 1}, false, true); submit(slots[frame % 2]);
                observe(slots[frame % 2], true);
                require(address(slots[frame % 2], smallPages[0]) == firstAddress, "Stable page moved");
                require(pool->stats().totalBuiltPageCount == 1, "Stable frame rebuilt a CLAS page");
                checkSinglePageStats(1, 0, smallBytes, smallEncodedBytes);
            }
            phase = "cancelled use of published CLAS";
            pool->beginFrame(); begin(slots[0], {}, false, true); slots[0].frame.cancel(); pool->beginFrame();
            begin(slots[0], {&first, 1}, false, true); submit(slots[0]); observe(slots[0], true);
            require(address(slots[0], smallPages[0]) == firstAddress && pool->stats().totalBuiltPageCount == 1,
                "Cancelled use invalidated a previously submitted CLAS");
            phase = "retire and revive before free";
            pool->retirePages({&smallPages[0], 1}); pool->beginFrame();
            checkSinglePageStats(0, 1, smallBytes, smallEncodedBytes);
            begin(slots[0], {&first, 1}, false, true); submit(slots[0]); observe(slots[0], true);
            checkSinglePageStats(1, 0, smallBytes, smallEncodedBytes);
            require(address(slots[0], smallPages[0]) == firstAddress && pool->stats().totalBuiltPageCount == 1,
                "Revive changed a resident CLAS address or rebuilt it");
            phase = "old BLAS retained by unsubmitted reader";
            pool->beginFrame(); begin(slots[1], {}, false, true);
            pool->retirePages({&smallPages[0], 1});
            for (uint32_t frame = 0; frame < 4; ++frame) { pool->beginFrame(); }
            require(pool->pageHasClas(smallPages[0]) && pool->stats().usedStorageBytes >= smallBytes,
                "Retirement freed a CLAS referenced by an unsubmitted old BLAS reader");
            checkSinglePageStats(0, 1, smallBytes, smallEncodedBytes);
            submit(slots[1], false);
            // Record the next frame before explicitly waiting for the old reader.
            // Completion may already have happened; both orderings are legitimate.
            pool->beginFrame(); begin(slots[0], {}, false, false); submit(slots[0]);
            checked(slots[1].frame.wait(5'000'000'000ull)); observe(slots[1], true);
            phase = "retirement completes";
            for (uint32_t frame = 0; frame < 4; ++frame) {
                pool->beginFrame(); begin(slots[0], {}, false, false); submit(slots[0]);
            }
            require(meshletStreamClasPageState(entry(slots[0], smallPages[0])) == MeshletStreamCLASPageState::Empty,
                "Retired GPU page remained published");
            pool->beginFrame();
            require(pool->stats().usedStorageBytes == 0, "Retired GPU page leaked allocator bytes");
            checkSinglePageStats(0, 0, 0, 0);
            phase = "rebuild after retirement";
            begin(slots[0], {&first, 1}, true, true); submit(slots[0]); observe(slots[0], true); pool->beginFrame();
            require(pool->stats().totalBuiltPageCount == 2, "Retired page failed to rebuild exactly once");
            checkSinglePageStats(1, 0, smallBytes, smallEncodedBytes);
            require(pool->publicationRevision() == pool->stats().publicationRevision,
                "Completed rebuild revision differs from pool statistics");

            phase = "request latency includes an unsubmitted free";
            const auto beforeQueuedFree = pool->stats();
            pool->retirePages({&smallPages[0], 1});
            for (uint32_t frame = 0; frame < 4; ++frame) { pool->beginFrame(); }
            begin(slots[0], {}, false, false);
            require(pool->pageBuildPending(smallPages[0]) && !pool->pageHasClas(smallPages[0]),
                "Retired page must wait for its recorded free to complete");
            checkSinglePageStats(0, 1, smallBytes, smallEncodedBytes);
            // CPU requests survive cancellation of their GPU recording. Keep
            // the free unsubmitted while requesting the same page twice, so
            // the eventual replacement must retain the earliest request frame.
            pool->beginFrame();
            begin(slots[1], {&first, 1}, false, false); slots[1].frame.cancel();
            pool->beginFrame();
            begin(slots[1], {&first, 1}, false, false); slots[1].frame.cancel();
            pool->beginFrame();
            require(pool->stats().totalBuiltPageCount == beforeQueuedFree.totalBuiltPageCount,
                "A request rebuilt its page before the old free completed");
            submit(slots[0]);
            pool->beginFrame();
            begin(slots[0], {&first, 1}, true, true); submit(slots[0]); observe(slots[0], true);
            require(meshletStreamClasPageState(entry(slots[0], smallPages[0])) == MeshletStreamCLASPageState::Active,
                "Request waiting for an old free failed to publish its replacement");
            pool->beginFrame();
            const auto afterQueuedFree = pool->stats();
            checkSinglePageStats(1, 0, smallBytes, smallEncodedBytes);
            constexpr uint64_t kQueuedFreeRequestLatencyFrames = 3;
            require(afterQueuedFree.totalBuiltPageCount == beforeQueuedFree.totalBuiltPageCount + 1 &&
                afterQueuedFree.readyLatencySamples == beforeQueuedFree.readyLatencySamples + 1,
                "Waiting request must produce exactly one replacement and latency sample");
            require(afterQueuedFree.readyLatencyFrameTotal ==
                    beforeQueuedFree.readyLatencyFrameTotal + kQueuedFreeRequestLatencyFrames &&
                afterQueuedFree.readyLatencyHistogram[kQueuedFreeRequestLatencyFrames] ==
                    beforeQueuedFree.readyLatencyHistogram[kQueuedFreeRequestLatencyFrames] + 1,
                "Replacement latency omitted frames waiting for the old free");

            phase = "measure fixture allocation sizes after completed GPU publication";
            const auto large = request(largePage);
            begin(slots[0], {&large, 1}, false, false); submit(slots[0]); pool->beginFrame();
            const uint64_t largeBytes = pool->pageStorageBytes(largePage);
            require(largeBytes > smallBytes, "Large fixture page must need more than one small-page hole");
            const auto mixedStats = pool->stats();
            require(mixedStats.usedStorageBytes == smallBytes + largeBytes &&
                mixedStats.persistentUsedBytes == smallBytes && mixedStats.transientUsedBytes == largeBytes &&
                mixedStats.builtPageCount == 2 && mixedStats.builtClusterCount == 1 + plans[largePage].clusters.size() &&
                mixedStats.worstCaseStorageBytes == uint64_t(mixedStats.builtClusterCount) * mixedStats.clusterStrideBytes,
                "Mixed persistent/transient multi-cluster page statistics disagree with page totals");
            // Only the test derives an adversarial capacity from completed
            // observations. Production allocation already happened on the GPU.
            const uint32_t fillCount = uint32_t(2 * ((largeBytes + smallBytes - 1) / smallBytes) + 1);
            require(fillCount < kSmallPageCount, "Fixture needs more small pages to produce fragmented free space");
            for (auto& slot : slots) { checked(slot.frame.reset()); checked(slot.pool->reset()); }
            const uint64_t tightBudget = fillCount * smallBytes;
            pool = std::make_unique<MeshletStreamGPUCLASPool>();
            checked(pool->initialize(*device, {.asset = &asset, .maxStorageBytes = tightBudget,
                .maxBuildClusters = asset.maxPageClusters() * (kSmallPageCount + 1), .queuedFrameCount = 2}, log));
            std::vector<MeshletStreamCLASPageBuild> fill;
            for (uint32_t index = 0; index < fillCount; ++index) { fill.push_back(request(smallPages[index])); }
            phase = "exact-fit full arena";
            pool->beginFrame(); begin(slots[0], fill, false, false); submit(slots[0]); pool->beginFrame();
            std::vector<uint64_t> originalAddresses;
            for (uint32_t index = 0; index < fillCount; ++index) {
                const auto value = address(slots[0], smallPages[index]);
                require(value != 0, "Exact-fit arena rejected a fitting page");
                require(std::find(originalAddresses.begin(), originalAddresses.end(), value) == originalAddresses.end(),
                    "Two pages overlap in the allocator");
                originalAddresses.push_back(value);
            }
            require(pool->stats().usedStorageBytes == tightBudget, "Exact-fit allocation accounting differs from page totals");
            require(pool->stats().persistentUsedBytes == 0 && pool->stats().transientUsedBytes == tightBudget &&
                pool->stats().builtPageCount == fillCount && pool->stats().builtClusterCount == fillCount,
                "Exact-fit transient page statistics disagree with page totals");
            const auto overflow = request(smallPages[fillCount]);
            phase = "full arena whole-page rejection";
            begin(slots[0], {&overflow, 1}, false, false); submit(slots[0]); pool->beginFrame();
            require(meshletStreamClasPageState(entry(slots[0], overflow.pageIndex)) == MeshletStreamCLASPageState::Empty &&
                pool->stats().usedStorageBytes == tightBudget, "Full arena partially published a rejected page");
            const uint64_t rejectedBefore = pool->stats().totalRejectedPageCount;
            require(rejectedBefore != 0, "Full arena rejection was not counted");

            std::vector<uint32_t> holes, survivors;
            std::vector<std::pair<uint64_t, uint32_t>> physicalOrder;
            for (uint32_t index = 0; index < fillCount; ++index) {
                physicalOrder.emplace_back(originalAddresses[index], smallPages[index]);
            }
            std::sort(physicalOrder.begin(), physicalOrder.end());
            for (uint32_t index = 0; index < fillCount; ++index) {
                (index % 2 ? survivors : holes).push_back(physicalOrder[index].second);
            }
            phase = "cancel free then retry";
            pool->retirePages(holes);
            for (uint32_t frame = 0; frame < 4; ++frame) { pool->beginFrame(); }
            begin(slots[0], {}, false, false); slots[0].frame.cancel();
            pool->beginFrame();
            const auto cancelledFreeStats = pool->stats();
            require(cancelledFreeStats.usedStorageBytes == tightBudget &&
                cancelledFreeStats.transientUsedBytes == tightBudget &&
                cancelledFreeStats.builtPageCount == survivors.size() &&
                cancelledFreeStats.builtClusterCount == survivors.size() &&
                cancelledFreeStats.retiringPageCount == holes.size() &&
                cancelledFreeStats.retiringClusterCount == holes.size() &&
                cancelledFreeStats.retiringStorageBytes == holes.size() * smallBytes,
                "Cancelled free changed live or retiring page statistics");
            for (uint32_t frame = 0; frame < 4; ++frame) {
                pool->beginFrame(); begin(slots[0], {}, false, false); submit(slots[0]);
            }
            pool->beginFrame();
            const uint64_t fragmentedUsed = survivors.size() * smallBytes;
            require(pool->stats().usedStorageBytes == fragmentedUsed && tightBudget - fragmentedUsed >= largeBytes,
                "Fragmentation fixture did not free enough total bytes");
            // Select holes in physical order, independently of workgroup order.
            std::vector<uint64_t> liveAddresses;
            for (uint32_t page : survivors) {
                const auto value = address(slots[0], page);
                const auto index = size_t(std::find(smallPages.begin(), smallPages.end(), page) - smallPages.begin());
                require(value == originalAddresses[index], "Retiring neighbours moved a live page");
                liveAddresses.push_back(value);
            }
            std::sort(originalAddresses.begin(), originalAddresses.end());
            std::sort(liveAddresses.begin(), liveAddresses.end());
            uint64_t largestHole = liveAddresses.front() - originalAddresses.front();
            for (size_t index = 1; index < liveAddresses.size(); ++index) {
                largestHole = std::max(largestHole, liveAddresses[index] - liveAddresses[index - 1] - smallBytes);
            }
            largestHole = std::max(largestHole,
                originalAddresses.front() + tightBudget - liveAddresses.back() - smallBytes);
            require(largestHole < largeBytes, "Allocation order did not produce the intended fragmented arena");
            phase = "fragmented arena whole-page rejection";
            begin(slots[0], {&large, 1}, false, false); submit(slots[0]); pool->beginFrame();
            require(meshletStreamClasPageState(entry(slots[0], largePage)) == MeshletStreamCLASPageState::Empty &&
                pool->stats().usedStorageBytes == fragmentedUsed && pool->stats().totalRejectedPageCount > rejectedBefore,
                "Fragmented arena published a partial multi-cluster page or changed accounting");
            phase = "coalesce neighbours and retry rejected page";
            pool->retirePages(survivors);
            for (uint32_t frame = 0; frame < 4; ++frame) {
                pool->beginFrame(); begin(slots[0], {}, false, false); submit(slots[0]);
            }
            pool->beginFrame();
            require(pool->stats().usedStorageBytes == 0, "Cancelled-free retry leaked or double-freed allocator bytes");
            checkSinglePageStats(0, 0, 0, 0);
            begin(slots[0], {&large, 1}, false, false); submit(slots[0]); pool->beginFrame();
            require(meshletStreamClasPageState(entry(slots[0], largePage)) == MeshletStreamCLASPageState::Active &&
                pool->stats().usedStorageBytes == largeBytes, "Adjacent holes did not coalesce for multi-cluster retry");

            std::ofstream(directory / "Frames.json") << evidence.dump(2);
            std::ofstream(directory / "Summary.json") << nlohmann::json{{"stableFrames", 32},
                {"firstPublicationSubmission", 1}, {"firstRayHitSubmission", 1},
                {"cpuSizeReadbackBeforeFirstRay", false}, {"firstAddress", firstAddress},
                {"freeQueuedRequestLatencyFrames", kQueuedFreeRequestLatencyFrames},
                {"smallPageBytes", smallBytes}, {"largePageBytes", largeBytes}, {"tightBudgetBytes", tightBudget},
                {"fragmentedFreeBytes", tightBudget - fragmentedUsed}, {"largestHoleBytes", largestHole},
                {"validationErrors", validationErrors.load()}}.dump(2);
            require(validationErrors.load() == 0, "GPU allocator regression produced Vulkan validation errors");
            return RHITestResult::pass("First-submission GPU allocation/move/publication and analytic rays; 32 stable frames; cancellation; retire/revive; old BLAS reader; full/fragmented whole-page rejection and coalescing");
        } catch (const std::exception& error) {
            return RHITestResult::fail(phase + ": " + error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(GPUCLASAllocatorTest);
} // namespace
} // namespace metallic::tests
