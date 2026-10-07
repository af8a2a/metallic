#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "RHITest.h"
#include "Runtime/Render/Streamer/MeshletStreamCompactCLASPool.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Scene/Scene.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
// Instrument only this test's privately owned device, restoring dispatch before destruction.
class ScopedPropertyQueryCounter {
public:
    explicit ScopedPropertyQueryCounter(render::Device& device)
        : table_(*const_cast<VolkInstanceTable*>(render::vulkan::nativeDevice(device).instanceFunctions))
    {
        original_ = table_.vkGetPhysicalDeviceProperties2;
        count_ = 0;
        table_.vkGetPhysicalDeviceProperties2 = countQuery;
    }
    ~ScopedPropertyQueryCounter() { table_.vkGetPhysicalDeviceProperties2 = original_; }
    ScopedPropertyQueryCounter(const ScopedPropertyQueryCounter&) = delete;
    ScopedPropertyQueryCounter& operator=(const ScopedPropertyQueryCounter&) = delete;
    uint32_t count() const { return count_; }
private:
    static VKAPI_ATTR void VKAPI_CALL countQuery(VkPhysicalDevice device, VkPhysicalDeviceProperties2* properties)
    {
        ++count_;
        original_(device, properties);
    }
    VolkInstanceTable& table_;
    inline static thread_local PFN_vkGetPhysicalDeviceProperties2 original_ = nullptr;
    inline static thread_local uint32_t count_ = 0;
};

class CLASSizeMoveTest final : public RHITest {
  public:
    CLASSizeMoveTest()
    {
        type = RHITestType::Resource;
        name = "clas_actual_sizes_and_move";
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "CLAS size/move regression",
                                    .enableValidation = context.enableValidation,
                                    .enableClusterAccelerationStructure = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(result, Error::Unsupported)) {
            return RHITestResult::skip("CLAS unavailable");
        }
        const auto require = [](bool ok, const char* text) {
            if (!ok) {
                throw std::runtime_error(text);
            }
        };
        try {
            require(bool(result), "Device creation failed");
            const auto slice = [&](Buffer* buffer, BufferRange range = {}) {
                auto result = buffer->slice(range);
                require(bool(result), "Buffer slice failed");
                return *result;
            };
            ScopedPropertyQueryCounter propertyQueries(*device);
            auto* queue = device->getQueue(QueueType::Graphics);
            constexpr auto usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                                   BufferUsageBits::AccelerationStructureBuildInput |
                                   BufferUsageBits::AccelerationStructureStorage;
            auto buffer = [&](uint64_t bytes, MemoryLocation location) {
                std::unique_ptr<Buffer> out;
                require(bool(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = location}).transform([&](auto rhiValue) { out = std::move(rhiValue); })),
                        "Buffer allocation failed");
                return out;
            };
            ClusterAccelerationStructureProperties properties;
            require(bool(device->queryClusterAccelerationStructureProperties().transform([&](auto rhiValue) { properties = std::move(rhiValue); })),
                    "CLAS properties unavailable");
            ClusterAccelerationStructureBuildSizes worst, moveSizes;
            require(bool(device->queryClusterAccelerationStructureTriangleBuildSizes({.maxClusterTriangleCount = 128,
                                                                                      .maxClusterVertexCount = 128,
                                                                                      .maxTotalTriangleCount = 128,
                                                                                      .maxTotalVertexCount = 128}).transform([&](auto rhiValue) { worst = std::move(rhiValue); })),
                    "Build sizes failed");
            const uint64_t stride = worst.accelerationStructureSize;
            require(bool(device->queryClusterAccelerationStructureMoveSizes(2, 2 * stride).transform([&](auto rhiValue) { moveSizes = std::move(rhiValue); })),
                    "Move sizes failed");
            auto temp = buffer(2 * stride, MemoryLocation::Device);
            auto scratch =
                buffer(std::max(worst.buildScratchSize * 2, moveSizes.updateScratchSize) + properties.scratchAlignment,
                       MemoryLocation::Device);
            const uint64_t scratchOffset =
                (properties.scratchAlignment - scratch->deviceAddress() % properties.scratchAlignment) %
                properties.scratchAlignment;
            auto vertices = buffer(36, MemoryLocation::HostUpload), indices = buffer(3, MemoryLocation::HostUpload);
            const float xyz[] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
            const uint8_t ix[] = {0, 1, 2};
            std::memcpy(vertices->map(), xyz, sizeof(xyz));
            vertices->flush();
            vertices->unmap();
            std::memcpy(indices->map(), ix, sizeof(ix));
            indices->flush();
            indices->unmap();
            auto infos = buffer(properties.triangleBuildInfoSize * 2 + 64, MemoryLocation::HostUpload);
            auto destinations = buffer(16, MemoryLocation::HostUpload),
                 sources = buffer(16, MemoryLocation::HostUpload);
            auto sizes = buffer(16, MemoryLocation::HostReadback);
            std::unique_ptr<CommandPool> commandsPool;
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Fence> fence;
            require(bool(device->createCommandPool(*queue).transform([&](auto rhiValue) { commandsPool = std::move(rhiValue); })) &&
                        bool(commandsPool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); })) && bool(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); })),
                    "Command resources failed");
            auto submit = [&]() {
                require(bool(commands->end()), "End failed");
                CommandBuffer* list[] = {commands.get()};
                require(bool(queue->submit(
                            {.commandBuffers = {list, 1}, .signalFence = fence.get()})) &&
                            bool(fence->wait(5000000000ull)),
                        "Submission failed");
            };
            require(bool(commands->begin()), "Begin failed");
            ClusterAccelerationStructureTriangleBuildInfo input[2];
            for (uint32_t i = 0; i < 2; ++i) {
                input[i] = {.clusterId = i,
                            .triangleCount = 1,
                            .vertexCount = 3,
                            .indexFormat = ClusterAccelerationStructureIndexFormat::Uint8,
                            .indexBufferStride = 1,
                            .vertexBufferStride = 12,
                            .indexBuffer = slice(indices.get()),
                            .vertexBuffer = slice(vertices.get()),
                            .destinationBuffer = slice(temp.get(), {i * stride, stride})};
            }
            const ClusterAccelerationStructureTriangleBuildDesc build{
                .clusters = {input, 2},
                .maxClusterTriangleCount = 128,
                .maxClusterVertexCount = 128,
                .scratchBuffer = slice(scratch.get(), {scratchOffset}),
                .buildInfoBuffer = slice(infos.get(), {64, properties.triangleBuildInfoSize * 2}),
                .destinationAddressBuffer = slice(destinations.get()),
                .destinationSizeBuffer = slice(sizes.get(), {8, 8}),
            };
            auto invalidBuild = build;
            invalidBuild.buildInfoBuffer = slice(infos.get(), {64, properties.triangleBuildInfoSize * 2 - 1});
            require(hasError(commands->buildClusterAccelerationStructureTriangles(invalidBuild), Error::InvalidArgument),
                "Undersized build info slice accepted");
            invalidBuild = build;
            invalidBuild.scratchBuffer = slice(scratch.get(), {scratchOffset, 1});
            require(hasError(commands->buildClusterAccelerationStructureTriangles(invalidBuild), Error::InvalidArgument),
                "Undersized build scratch slice accepted");
            const auto destinationSlice = input[0].destinationBuffer;
            input[0].destinationBuffer = slice(temp.get(), {0, 1});
            require(hasError(commands->buildClusterAccelerationStructureTriangles(build), Error::InvalidArgument),
                "Undersized CLAS destination slice accepted");
            input[0].destinationBuffer = destinationSlice;
            require(bool(commands->buildClusterAccelerationStructureTriangles(build)), "Build with size output failed");
            submit();
            sizes->invalidate();
            uint32_t actual[2];
            std::memcpy(actual, static_cast<const uint8_t*>(sizes->map()) + 8, sizeof(actual));
            sizes->unmap();
            require(actual[0] > 0 && actual[0] < stride && actual[0] % properties.clusterStorageAlignment == 0 &&
                        actual[0] == actual[1],
                    "Invalid actual sizes");
            auto compact = buffer(uint64_t(actual[0]) + actual[1], MemoryLocation::Device);
            require(bool(commandsPool->reset()) && bool(fence->reset()) && bool(commands->begin()),
                    "Move begin failed");
            ClusterAccelerationStructureMoveInfo moves[] = {
                {.sourceBuffer = slice(temp.get(), {0, actual[0]}), .destinationBuffer = slice(compact.get(), {0, actual[0]})},
                {.sourceBuffer = slice(temp.get(), {stride, actual[1]}), .destinationBuffer = slice(compact.get(), {actual[0], actual[1]})}};
            auto move = ClusterAccelerationStructureMoveDesc{
                .objects = {moves, 2},
                .sourceAddressBuffer = slice(sources.get()),
                .destinationAddressBuffer = slice(destinations.get()),
                .scratchBuffer = slice(scratch.get(), {scratchOffset}),
            };
            moves[1].destinationBuffer = slice(compact.get(), {actual[0] + 1, actual[1] - 1});
            require(hasError(commands->moveClusterAccelerationStructures(move), Error::InvalidArgument),
                    "Unaligned move accepted");
            moves[1].destinationBuffer = slice(compact.get(), {0, actual[1]});
            require(hasError(commands->moveClusterAccelerationStructures(move), Error::InvalidArgument),
                    "Overlapping destinations accepted");
            moves[1].destinationBuffer = slice(compact.get(), {actual[0], actual[1]});
            ClusterAccelerationStructureBuildSizes exactMoveSizes;
            require(bool(device->queryClusterAccelerationStructureMoveSizes(2, uint64_t(actual[0]) + actual[1]).transform([&](auto rhiValue) { exactMoveSizes = std::move(rhiValue); })),
                    "Exact move sizes failed");
            if (exactMoveSizes.updateScratchSize > 1) {
                auto invalidMove = move;
                invalidMove.scratchBuffer = slice(scratch.get(), {scratchOffset, exactMoveSizes.updateScratchSize - 1});
                require(hasError(commands->moveClusterAccelerationStructures(invalidMove), Error::InvalidArgument),
                        "Undersized move scratch accepted");
            }
            require(bool(commands->moveClusterAccelerationStructures(move)), "Compact move failed");
            submit();
            // A second relocation reads the newly packed objects, exercising the
            // driver's internal fixups rather than treating the AS as raw bytes.
            require(bool(commandsPool->reset()) && bool(fence->reset()) && bool(commands->begin()),
                    "Second move begin failed");
            for (uint32_t i = 0; i < 2; ++i) {
                moves[i] = {.sourceBuffer = slice(compact.get(), {uint64_t(i) * actual[0], actual[i]}),
                            .destinationBuffer = slice(temp.get(), {i * stride, actual[i]})};
            }
            auto retainedSources = buffer(32, MemoryLocation::HostUpload);
            auto retainedDestinations = buffer(32, MemoryLocation::HostUpload);
            std::weak_ptr<void> sourceTableLifetime = retainedSources->retainAllocation();
            std::weak_ptr<void> destinationTableLifetime = retainedDestinations->retainAllocation();
            move.sourceAddressBuffer = slice(retainedSources.get(), {16, 16});
            move.destinationAddressBuffer = slice(retainedDestinations.get(), {16, 16});
            require(bool(commands->moveClusterAccelerationStructures(move)), "Relocating compact CLAS failed");
            move.sourceAddressBuffer = {}; move.destinationAddressBuffer = {};
            retainedSources.reset(); retainedDestinations.reset();
            require(!sourceTableLifetime.expired() && !destinationTableLifetime.expired(), "Move did not retain sliced address tables");
            submit();
            require(bool(commandsPool->reset()) && bool(commands->begin()), "Retirement reset failed");
            require(sourceTableLifetime.expired() && destinationTableLifetime.expired(), "Move address tables leaked after retirement");
            require(bool(commands->end()), "Retirement end failed");
            std::filesystem::create_directories(context.outputDirectory);
            require(propertyQueries.count() == 0, "CLAS runtime re-queried fixed physical-device properties");
            std::ofstream(context.outputDirectory / "CLASSizeMove.txt")
                << "worstCaseBytes=" << stride * 2 << " actualBytes=" << actual[0] + actual[1]
                << " runtimePropertyQueries=" << propertyQueries.count()
                << " moveScratchBytes=" << exactMoveSizes.updateScratchSize << '\n';
            return RHITestResult::pass(
                "Actual GPU sizes, compact relocation, second relocation and invalid-range rejection");
        } catch (const std::exception& error) {
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(CLASSizeMoveTest);
class CompactCLASLifecycleTest final : public RHITest {
  public:
    CompactCLASLifecycleTest()
    {
        type = RHITestType::Resource;
        name = "clas_compact_lifecycle";
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::filesystem::create_directories(context.outputDirectory);
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Compact CLAS lifecycle",
                                    .enableValidation = context.enableValidation,
                                    .enableClusterAccelerationStructure = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(result, Error::Unsupported)) {
            return RHITestResult::skip("CLAS unavailable");
        }
        if (!result) {
            return RHITestResult::fail("Device failed");
        }
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        scene::Scene loadedScene;
        if (!loadedScene.load(scenePath)) {
            return RHITestResult::fail("failed to load Stanford Bunny scene: " + loadedScene.lastLoadResult().error);
        }
        const std::filesystem::path streamAssetPath =
            context.outputDirectory / "meshlet_stream_clas_pool.meshstream.bin";
        std::string log;
        if (!scene::buildMeshletStreamAsset(
                scene::MeshletStreamAssetBuildDesc{
                    .scene = &loadedScene,
                    .sourcePath = scenePath,
                    .outputPath = streamAssetPath,
                    .compressionMode = scene::MeshletStreamPayloadCompression::ByteRle,
                },
                log)) {
            return RHITestResult::fail("buildMeshletStreamAsset failed: " + log);
        }

        scene::MeshletStreamAsset asset;
        if (!asset.open(streamAssetPath, log)) {
            return RHITestResult::fail("MeshletStreamAsset::open failed: " + log);
        }
        uint32_t pageIndex = UINT32_MAX;
        for (const scene::MeshletStreamGroupInfo& group : asset.groups()) {
            if (group.maxQuadricError == scene::kMeshletStreamTerminalGroupError) {
                pageIndex = group.pageIndex;
                break;
            }
        }
        if (pageIndex == UINT32_MAX) {
            return RHITestResult::fail("streamasset has no fallback page for CLAS pool build");
        }

        std::vector<uint8_t> decodedStorage;
        std::span<const uint8_t> decodedPayload;
        if (!scene::decodeMeshletStreamPayloadForDevice(asset.pages()[pageIndex], asset.pagePayload(pageIndex),
                                                        decodedStorage, decodedPayload, log)) {
            return RHITestResult::fail("streamasset fallback page decode failed: " + log);
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = device->createBuffer(render::BufferDesc{
                .size = decodedPayload.size(),
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::ShaderDeviceAddress |
                         render::BufferUsageBits::AccelerationStructureBuildInput,
                .memoryLocation = render::MemoryLocation::HostUpload,
            }).transform([&](auto rhiValue) { pageBuffer = std::move(rhiValue); });
        if (!result || pageBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer(stream CLAS page) returned ") + toString(result));
        }
        void* mapped = pageBuffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("stream CLAS page buffer did not map");
        }
        std::memcpy(mapped, decodedPayload.data(), decodedPayload.size());
        pageBuffer->flush({0, decodedPayload.size()});
        pageBuffer->unmap();

        const auto require = [](bool ok, const std::string& message) {
            if (!ok) {
                throw std::runtime_error(message);
            }
        };
        try {
            MeshletStreamCompactCLASPool pool;
            require(bool(pool.initialize(*device,
                                         {.asset = &asset,
                                          .maxStorageBytes = 1024 * 1024,
                                          .maxBuildClusters = asset.maxPageClusters(),
                                          .queuedFrameCount = 2,
                                          .growStorageBytes = 4096},
                                         log)),
                    log);
            require(pool.stats().storageBytes == 0 && pool.stats().storageChunkCount == 0 &&
                        pool.stats().storageBudgetBytes == 1024 * 1024,
                    "Compact pool allocated its budget at initialization");
            MeshletStreamCLASPagePlan plan;
            require(buildMeshletStreamClasPagePlan(asset.pages()[pageIndex], decodedPayload, pageIndex,
                                                   pageIndex * asset.maxPageClusters(), plan, log),
                    log);
            const MeshletStreamCLASPageBuild build{.pageIndex = pageIndex, .deviceOffsetBytes = 0, .plan = &plan};
            auto* queue = device->getQueue(QueueType::Graphics);
            QueueSubmissionTracker tracker;
            RenderFrameContext frame;
            std::unique_ptr<CommandPool> commandPool;
            std::unique_ptr<CommandBuffer> cmd;
            require(bool(tracker.initialize(*device, *queue)) && bool(device->createCommandPool(*queue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); })) &&
                        bool(commandPool->createCommandBuffer().transform([&](auto rhiValue) { cmd = std::move(rhiValue); })),
                    "Commands failed");
            uint64_t frameId = 0;
            std::unique_ptr<Buffer> publicationReadback;
            require(bool(device->createBuffer({.size = 4, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { publicationReadback = std::move(rhiValue); })), "Publication readback failed");
            uint32_t gpuPageEntry = 0;
            std::span<const MeshletStreamCLASPageBuild> activeRequests(&build, 1);
            auto record = [&](bool request, bool cancel = false) {
                require(bool(frame.begin(++frameId)) && bool(commandPool->reset()) && bool(cmd->begin(frame.submissionContext())),
                        "Begin failed");
                require(bool(pool.cmdBuildPages(
                            *cmd, *pageBuffer,
                            request ? activeRequests : std::span<const MeshletStreamCLASPageBuild>{}, log)),
                        log);
                BufferBarrierDesc tableBarrier{
                    .buffer = pool.pageTableBuffer(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                };
                if (auto commandResult = cmd->synchronize({.buffers = {&tableBarrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
                {
                    auto sourceSlice = pool.pageTableBuffer()->slice({uint64_t(pageIndex) * sizeof(MeshletStreamCLASPageEntry), 4});
                    if (!sourceSlice) { throw std::runtime_error(std::string("source slice failed: ") + metallic::render::resultToString(sourceSlice)); }
                    auto destinationSlice = publicationReadback.get()->slice({0, 4});
                    if (!destinationSlice) { throw std::runtime_error(std::string("destination slice failed: ") + metallic::render::resultToString(destinationSlice)); }
                    if (auto commandResult = cmd->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { throw std::runtime_error(std::string("copyBuffer failed: ") + metallic::render::resultToString(commandResult)); }
                }
                std::swap(tableBarrier.before, tableBarrier.after);
                if (auto commandResult = cmd->synchronize({.buffers = {&tableBarrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
                require(bool(cmd->end()), "End failed");
                if (cancel) {
                    frame.cancel();
                    return;
                }
                CommandBuffer* list[] = {cmd.get()};
                require(bool(tracker.submit({.commandBuffers = {list, 1}}, frame)) &&
                            bool(frame.wait(5000000000ull)),
                        "Tracked submit failed");
                publicationReadback->invalidate();
                const auto* mappedEntry = static_cast<const uint32_t*>(publicationReadback->map());
                require(mappedEntry != nullptr, "Publication map failed");
                gpuPageEntry = *mappedEntry;
                publicationReadback->unmap();
            };
            pool.beginFrame();
            record(true);
            require(pool.pageBuildPending(pageIndex) && !pool.pageHasClas(pageIndex) &&
                        pool.clusterAddress(pageIndex, 0) == 0,
                    "Build published an address before relocation");
            pool.retirePages(std::span(&pageIndex, 1));
            pool.beginFrame();
            record(true); // Revive during completed build / size collection.
            require(!pool.pageHasClas(pageIndex) && pool.stats().frameMovedPageCount == 1,
                    "CPU move ownership must await completion collection");
            require((gpuPageEntry >> kMeshletStreamCLASPageStateShift) == uint32_t(MeshletStreamCLASPageState::Active),
                    "GPU traversal cannot use CLAS in the MOVE submission");
            pool.beginFrame();
            require(pool.pageHasClas(pageIndex) && !pool.pageBuildPending(pageIndex), "Move did not publish");
            const auto stats = pool.stats();
            require(stats.encodedStorageBytes > 0 && stats.usedStorageBytes < stats.worstCaseStorageBytes &&
                        stats.usedStorageBytes >= stats.encodedStorageBytes,
                    "Pool did not allocate actual sizes");
            {
                // Nsight can consume all device-local headroom after the pool
                // was initialized. Reviving a resident CLAS must still publish
                // its address/table entry using pure host staging.
                const auto budget = device->memoryBudget();
                const bool separateHostHeap = std::any_of(budget.heaps.begin(), budget.heaps.end(),
                    [](const auto& heap) { return !heap.deviceLocal; });
                if (separateHostHeap) {
                    struct RestoreBudget {
                        Device& device;
                        MemoryBudgetPolicy policy;
                        ~RestoreBudget() { device.setMemoryBudgetPolicy(policy); }
                    } restore{*device, budget.policy};
                    auto pressure = budget.policy;
                    pressure.enabled = true;
                    pressure.deviceLocalHeapLimitBytes = 1;
                    device->setMemoryBudgetPolicy(pressure);
                    pool.retirePages(std::span(&pageIndex, 1));
                    pool.beginFrame();
                    record(true);
                    require(pool.pageHasClas(pageIndex) &&
                        (gpuPageEntry >> kMeshletStreamCLASPageStateShift) ==
                            uint32_t(MeshletStreamCLASPageState::Active) &&
                        device->memoryBudget().deniedAllocations == budget.deniedAllocations,
                        "CLAS publication must progress without device-local allocation headroom");
                }
            }
            require(stats.storageBytes >= stats.usedStorageBytes && stats.storageBytes < stats.storageBudgetBytes &&
                        stats.storageChunkCount == 1, "Physical backing did not grow on demand");
            const uint64_t address = pool.clusterAddress(pageIndex, 0);
            const auto* backing = pool.pageStorageBuffer(pageIndex);
            require(backing && address >= backing->deviceAddress() &&
                        address + pool.pageStorageBytes(pageIndex) <= backing->deviceAddress() + backing->desc().size,
                    "Published address is outside its physical chunk");
            pool.retirePages(std::span(&pageIndex, 1));
            pool.beginFrame();
            record(true);
            require(pool.clusterAddress(pageIndex, 0) == address && pool.stats().totalBuiltPageCount == 1,
                    "Retiring reload rebuilt CLAS");
            pool.retirePages(std::span(&pageIndex, 1));
            pool.beginFrame();
            require(pool.stats().usedStorageBytes > 0 && pool.stats().retiringPageCount == 1 &&
                        pool.clusterAddress(pageIndex, 0) == address,
                    "Stale retirement expired a revived page before its new deadline");
            pool.beginFrame();
            require(pool.stats().usedStorageBytes == 0 && pool.stats().retiringPageCount == 0 &&
                        !pool.pageHasClas(pageIndex) && pool.stats().storageBytes == 0 && pool.stats().storageChunkCount == 0,
                    "Retirement leaked physical storage or double-counted a stale entry");
            require(device->memoryBudget().domains[size_t(MemoryBudgetDomain::CLAS)].allocationBytes == 0,
                    "Completed command wrapper retained retired CLAS physical memory");
            record(true, true);
            pool.beginFrame();
            require(!pool.pageBuildPending(pageIndex) && pool.stats().trackedPageCount == 0,
                    "Cancelled build leaked pending page");
            record(true);
            pool.beginFrame();
            record(false, true);
            pool.beginFrame(); // Cancel the move, retain its completed source.
            require(pool.stats().usedStorageBytes == 0 && pool.pageBuildPending(pageIndex),
                    "Cancelled move leaked allocation");
            record(false);
            pool.beginFrame();
            require(pool.pageHasClas(pageIndex), "Cancelled move could not retry");
            pool.retirePages(std::span(&pageIndex, 1));
            pool.beginFrame();
            pool.beginFrame();
            record(true);
            pool.retirePages(std::span(&pageIndex, 1));
            pool.beginFrame();
            record(false);
            pool.beginFrame();
            require(pool.stats().usedStorageBytes == 0 && pool.stats().trackedPageCount == 0,
                    "Abandoned build leaked storage");
            require(pool.stats().storageBytes == 0, "Abandoned build kept physical backing");

            // Two different pages must occupy distinct physical chunks, while the
            // GPU address of one survives growth and reclamation of the other.
            uint32_t secondPage = 0;
            while (secondPage < asset.pageCount() &&
                   (secondPage == pageIndex || !asset.pages()[secondPage].clusterCount)) { ++secondPage; }
            require(secondPage < asset.pageCount(), "Need two pages for physical growth regression");
            std::vector<uint8_t> secondStorage;
            std::span<const uint8_t> secondPayload;
            require(scene::decodeMeshletStreamPayloadForDevice(asset.pages()[secondPage], asset.pagePayload(secondPage),
                secondStorage, secondPayload, log), log);
            MeshletStreamCLASPagePlan secondPlan;
            require(buildMeshletStreamClasPagePlan(asset.pages()[secondPage], secondPayload, secondPage,
                secondPage * asset.maxPageClusters(), secondPlan, log), log);
            const uint64_t secondOffset = (decodedPayload.size() + 255) / 256 * 256;
            require(bool(device->createBuffer({.size = secondOffset + secondPayload.size(),
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                         BufferUsageBits::AccelerationStructureBuildInput,
                .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto value) { pageBuffer = std::move(value); })),
                "Two-page geometry allocation failed");
            auto* geometry = static_cast<uint8_t*>(pageBuffer->map());
            require(geometry != nullptr, "Two-page geometry map failed");
            std::memcpy(geometry, decodedPayload.data(), decodedPayload.size());
            std::memcpy(geometry + secondOffset, secondPayload.data(), secondPayload.size());
            pageBuffer->flush();
            pageBuffer->unmap();
            const MeshletStreamCLASPageBuild twoBuilds[] = {build,
                {.pageIndex = secondPage, .deviceOffsetBytes = secondOffset, .plan = &secondPlan}};
            activeRequests = twoBuilds;
            for (uint32_t i = 0; i < 8 && (!pool.pageHasClas(pageIndex) || !pool.pageHasClas(secondPage)); ++i) {
                record(true);
                pool.beginFrame();
            }
            require(pool.pageHasClas(pageIndex) && pool.pageHasClas(secondPage) && pool.stats().storageChunkCount == 2,
                "Demand growth did not build two pages in separate chunks");
            require(pool.pageStorageBuffer(pageIndex) != pool.pageStorageBuffer(secondPage), "Pages share a chunk unexpectedly");
            const uint64_t secondAddress = pool.clusterAddress(secondPage, 0);
            const uint64_t bothBytes = pool.stats().storageBytes;
            require(bothBytes <= pool.stats().storageBudgetBytes, "Physical growth exceeded budget");
            require(bool(frame.begin(++frameId)) && bool(commandPool->reset()) && bool(cmd->begin(frame.submissionContext())), "Begin pending frame");
            require(bool(pool.cmdBuildPages(*cmd, *pageBuffer, {}, log)) && bool(cmd->end()), "Record pending frame");
            pool.retirePages(std::span(&pageIndex, 1));
            for (uint32_t i = 0; i < 4; ++i) { pool.beginFrame(); }
            require(pool.pageHasClas(pageIndex) && pool.stats().storageBytes == bothBytes,
                "Retirement freed storage referenced by an unsubmitted frame");
            frame.cancel();
            pool.beginFrame();
            require(!pool.pageHasClas(pageIndex) && pool.stats().storageChunkCount == 1 &&
                pool.stats().storageBytes < bothBytes && pool.clusterAddress(secondPage, 0) == secondAddress,
                "Empty chunk reclaim changed a live address or retained cancelled work");
            pool.retirePages(std::span(&secondPage, 1));
            pool.beginFrame();
            pool.beginFrame();
            require(pool.stats().storageBytes == 0 && pool.stats().usedStorageBytes == 0, "Final chunk leaked");

            require(pool.stats().totalStorageGrowthCount >= 2 && pool.stats().totalStorageReleasedBytes >= bothBytes,
                "Growth/release counters did not observe physical lifecycle");
            {
                MeshletStreamCompactCLASPool warm;
                require(bool(warm.initialize(*device, {.asset = &asset, .maxStorageBytes = 16384,
                    .maxBuildClusters = asset.maxPageClusters(), .queuedFrameCount = 2,
                    .startStorageBytes = 4096, .growStorageBytes = 8192, .emptyChunkRetentionFrames = 3}, log)), log);
                require(warm.stats().storageBytes == 4096 && warm.stats().startStorageBytes == 4096 &&
                    warm.stats().growStorageBytes == 8192 && warm.stats().storageBudgetBytes == 16384,
                    "Start/grow/max were not independent");
                warm.beginFrame(); warm.beginFrame();
                require(warm.stats().storageBytes == 4096 && warm.stats().emptyStorageBytes == 4096,
                    "Empty retention released warm backing too early");
                warm.beginFrame();
                require(warm.stats().storageBytes == 0 && warm.stats().totalStorageReleasedBytes == 4096 &&
                    warm.stats().frameStorageReleaseCount == 1,
                    "Unused start allocation did not return after retention");
            }
            {
                MeshletStreamCompactCLASPool invalid;
                require(!invalid.initialize(*device, {.asset = &asset, .maxStorageBytes = 4096,
                    .startStorageBytes = 8192}, log), "Start greater than max was accepted");
                require(!invalid.initialize(*device, {.asset = &asset, .growStorageBytes = 0}, log),
                    "Zero growth was accepted");
            }
            {
                // A large page needs one contiguous segment. Retained empty
                // small chunks may be returned early under the max budget.
                MeshletStreamCompactCLASPool pressure;
                const uint64_t capacity = stats.usedStorageBytes * 2;
                require(bool(pressure.initialize(*device, {.asset = &asset, .maxStorageBytes = capacity,
                    .maxBuildClusters = asset.maxPageClusters(), .queuedFrameCount = 2,
                    .startStorageBytes = capacity, .growStorageBytes = 256, .emptyChunkRetentionFrames = 60}, log)), log);
                for (uint32_t i = 0; i < 5 && !pressure.pageHasClas(pageIndex); ++i) {
                    pressure.beginFrame();
                    require(bool(frame.begin(++frameId)) && bool(commandPool->reset()) && bool(cmd->begin(frame.submissionContext())), "Pressure begin");
                    require(bool(pressure.cmdBuildPages(*cmd, *pageBuffer, std::span(&build, 1), log)) && bool(cmd->end()), log);
                    CommandBuffer* list[] = {cmd.get()};
                    require(bool(tracker.submit({.commandBuffers = {list, 1}}, frame)) && bool(frame.wait(5000000000ull)), "Pressure submit");
                }
                require(pressure.pageHasClas(pageIndex) && pressure.stats().totalStorageReleasedBytes >= capacity &&
                    pressure.stats().storageBytes <= capacity, "Empty retention blocked a page that fits the max budget");
            }
            {
                MeshletStreamCompactCLASPool segregated;
                require(bool(segregated.initialize(*device, {.asset = &asset, .maxStorageBytes = 2 * 1024 * 1024,
                    .maxBuildClusters = asset.maxPageClusters(), .queuedFrameCount = 2,
                    .growStorageBytes = 64 * 1024 * 1024, .persistentGrowStorageBytes = 512 * 1024,
                    .persistentPages = std::span(&pageIndex, 1)}, log)), log);
                auto tick = [&](std::span<const MeshletStreamCLASPageBuild> requests) {
                    segregated.beginFrame();
                    require(bool(frame.begin(++frameId)) && bool(commandPool->reset()) && bool(cmd->begin(frame.submissionContext())), "Segregated begin");
                    require(bool(segregated.cmdBuildPages(*cmd, *pageBuffer, requests, log)) && bool(cmd->end()), log);
                    CommandBuffer* list[] = {cmd.get()};
                    require(bool(tracker.submit({.commandBuffers = {list, 1}}, frame)) && bool(frame.wait(5000000000ull)), "Segregated submit");
                };
                for (uint32_t i = 0; i < 8 && (!segregated.pageHasClas(pageIndex) || !segregated.pageHasClas(secondPage)); ++i) {
                    tick(twoBuilds);
                }
                const auto separated = segregated.stats();
                require(segregated.pageHasClas(pageIndex) && segregated.pageHasClas(secondPage) &&
                    separated.usedStorageBytes < 1024 * 1024 && separated.storageChunkCount == 2 &&
                    separated.persistentStorageBytes == 512 * 1024 && separated.persistentGrowStorageBytes == 512 * 1024 && separated.transientStorageBytes == 1024 * 1024 &&
                    separated.persistentUsedBytes + separated.transientUsedBytes == separated.usedStorageBytes &&
                    segregated.pageStorageBuffer(pageIndex) != segregated.pageStorageBuffer(secondPage),
                    "Root and transient pages mixed despite fitting in one block");
                const uint64_t rootAddress = segregated.clusterAddress(pageIndex, 0);
                segregated.retirePages(std::span(&secondPage, 1));
                for (uint32_t i = 0; i < 4; ++i) { tick(std::span(&build, 1)); }
                require(segregated.stats().transientStorageBytes == 0 && segregated.stats().storageBytes == 512 * 1024 &&
                    segregated.stats().totalStorageReleasedBytes == 1024 * 1024 &&
                    segregated.clusterAddress(pageIndex, 0) == rootAddress,
                    "Transient block did not return independently of the stable root");
            }
            MeshletStreamCompactCLASPool constrained;
            require(bool(constrained.initialize(*device, {.asset = &asset, .maxStorageBytes = 256,
                .maxBuildClusters = asset.maxPageClusters(), .queuedFrameCount = 2, .growStorageBytes = 256}, log)), log);
            for (uint32_t i = 0; i < 3; ++i) {
                constrained.beginFrame();
                require(bool(frame.begin(++frameId)) && bool(commandPool->reset()) && bool(cmd->begin(frame.submissionContext())), "Budget test begin");
                require(bool(constrained.cmdBuildPages(*cmd, *pageBuffer, std::span(&build, 1), log)) && bool(cmd->end()), log);
                CommandBuffer* list[] = {cmd.get()};
                require(bool(tracker.submit({.commandBuffers = {list, 1}}, frame)) && bool(frame.wait(5000000000ull)), "Budget test submit");
            }
            require(constrained.pageBuildPending(pageIndex) && constrained.stats().totalRejectedPageCount > 0 &&
                constrained.stats().storageBytes == 0 && constrained.clusterAddress(pageIndex, 0) == 0,
                "Budget rejection must retain sized source without allocating/publishing storage");
            std::ofstream(context.outputDirectory / "CompactCLASLifecycle.txt")
                << "worstCaseBytes=" << stats.worstCaseStorageBytes << " allocatedBytes=" << stats.usedStorageBytes
                << " encodedBytes=" << stats.encodedStorageBytes << '\n';
            return RHITestResult::pass("Actual allocation, ordered GPU publication, revival, retirement, build/move "
                                       "cancellation and abandoned build");
        } catch (const std::exception& error) {
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(CompactCLASLifecycleTest);
class MiniZorahCLASInFlightTest final : public RHITest {
  public:
    MiniZorahCLASInFlightTest()
    {
        type = RHITestType::Rendering;
        name = "minizorah_clas_in_flight";
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        if (!std::getenv("METALLIC_TEST_MINIZORAH")) { return RHITestResult::skip("Set METALLIC_TEST_MINIZORAH=1"); }
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream trace(context.outputDirectory / "MiniZorahCLASInFlight.jsonl");
        DeviceDesc desc;
        desc.applicationName = "MiniZorah CLAS in-flight regression";
        desc.enableValidation = context.enableValidation;
        desc.enableBindlessDescriptorHeap = true;
        desc.enableShaderObject = true;
        desc.enableMeshShader = true;
        desc.enableTaskShader = true;
        desc.enableTaskShaderSubgroupBallot = true;
        desc.enableGeometryShader = true;
        desc.enableSubgroupSizeControl = true;
        desc.enableComputeFullSubgroups = true;
        desc.preferredTaskSubgroupSize = 32;
        desc.enableAsyncCompute = true;
        desc.enableStreamline = std::getenv("METALLIC_TEST_CLAS_SCENE_SWITCH") != nullptr;
        desc.enableRayTracingAccelerationStructure = true;
        desc.enableRayQuery = true;
        desc.enableClusterAccelerationStructure = true;
        desc.enableAftermath = true;
        std::unique_ptr<Device> device;
        auto result = createDevice(desc).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("CLAS unavailable"); }
        if (!result) { return RHITestResult::fail(toString(result)); }
        std::string log;
        RenderSampleLoadResult sample;
        if (!loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        auto& graph = sample.graph;
        if (std::getenv("METALLIC_TEST_CLAS_LEGACY")) { graph.findNode("GPUDriven")->properties["compactClas"] = false; }
        RenderView view;
        const auto original = graph.viewProperties().at("camera");
        if (!view.setCameraProperties(original)) { return RHITestResult::fail("Camera failed"); }
        scene::Scene runtimeScene;
        HistoryResourceManager history;
        if (!(result = history.initialize(*device))) { return RHITestResult::fail(toString(result)); }
        RenderGraphExecutor executor;
        executor.bindRenderView(&view);
        executor.bindRuntimeScene(&runtimeScene);
        if (std::getenv("METALLIC_TEST_CLAS_SCENE_SWITCH")) {
            RenderSampleLoadResult previous;
            if (!loadBuiltInRenderSample("realtime-lighting", previous, log) ||
                !setRenderSampleScenePath(previous, "Asset/Sponza/glTF/Sponza.gltf", log)) { return RHITestResult::fail(log); }
            if (!runtimeScene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / previous.desc.scenePath)) {
                return RHITestResult::fail("Sponza load failed");
            }
            SceneAccelerationStructureBuilder staticRtas;
            if (!(result = staticRtas.build(*device, *device->getQueue(QueueType::Compute), runtimeScene, log))) {
                return RHITestResult::fail(log);
            }
            (void)view.setCameraProperties(previous.graph.viewProperties().at("camera"));
            if (!(result = executor.compile(*device, previous.graph, 1564, 708, log))) { return RHITestResult::fail(log); }
            for (uint32_t f = 0; f < 60; ++f) {
                result = executor.execute({.graphicsQueue = device->getQueue(QueueType::Graphics),
                    .computeQueue = device->getQueue(QueueType::Compute), .historyResources = &history});
                if (!result) { return RHITestResult::fail(toString(result)); }
            }
            staticRtas.clear();
            runtimeScene.clear();
            history.invalidateAll();
            (void)view.setCameraProperties(original);
        }
        result = executor.compile(*device, graph, 1564, 708, log);
        if (!result) { return RHITestResult::fail(log); }
        uint32_t overlappingFrames = 0;
        for (uint32_t f = 0; f < 1200; ++f) {
            auto camera = original;
            const float angle = f < 180 ? 0.f : std::sin(float(f - 180) * .021f) * 2.7f;
            const float x = original["center"][0].get<float>() - original["eye"][0].get<float>();
            const float z = original["center"][2].get<float>() - original["eye"][2].get<float>();
            camera["center"][0] = original["eye"][0].get<float>() + std::cos(angle) * x + std::sin(angle) * z;
            camera["center"][2] = original["eye"][2].get<float>() - std::sin(angle) * x + std::cos(angle) * z;
            (void)view.setCameraProperties(camera);
            if (executor.lastSubmittedCompletion().valid() && !executor.lastSubmittedCompletion().isComplete()) { ++overlappingFrames; }
            // Same execution path as the editor: only the reused frame slot is
            // waited; do not drain the graph after every frame like PreviewRenderer.
            result = executor.execute({.graphicsQueue = device->getQueue(QueueType::Graphics),
                .computeQueue = device->getQueue(QueueType::Compute), .historyResources = &history});
            if (!result) { return RHITestResult::fail("Frame " + std::to_string(f) + ": " + toString(result)); }
            const auto& stats = executor.executionStats();
            const auto& stream = stats.streaming.at(0);
            if (stream.clasUsedBytes > stream.clasAllocatedBytes || stream.clasAllocatedBytes > stream.clasCapacityBytes) {
                return RHITestResult::fail("CLAS physical storage accounting exceeded budget");
            }
            trace << nlohmann::json{{"frame", f}, {"overlappingFrames", overlappingFrames}, {"clasBytes", stream.clasUsedBytes},
                {"clasAllocatedBytes", stream.clasAllocatedBytes}, {"clasStorageChunks", stream.clasStorageChunks},
                {"clasBuilt", stream.clasBuiltClusters}, {"clasMoved", stream.clasMovedClusters}, {"clasPending", stream.clasPendingPages}}.dump() << std::endl;
        }
        result = executor.waitForSubmittedWork();
        if (!result) { return RHITestResult::fail(toString(result)); }
        if (overlappingFrames < 100) { return RHITestResult::fail("Did not exercise frames in flight"); }
        return RHITestResult::pass("1200 frames, overlap observed " + std::to_string(overlappingFrames));
    }
};
METALLIC_REGISTER_RHI_TEST(MiniZorahCLASInFlightTest);
} // namespace
} // namespace metallic::tests
