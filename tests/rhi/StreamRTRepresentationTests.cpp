#include "RHITest.h"
#include "TestComputeProgram.h"
#include "TestResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Profiling/RenderGraphProfile.h"
#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <json.hpp>

namespace metallic::tests {
namespace {
using namespace render;

constexpr uint32_t kInstanceCount = 8;
struct ProbeObservation {
    uint32_t directInstance, reflectedInstance;
    float directDistance, reflectedDistance;
};
struct NativeInstance {
    float transform[3][4];
    uint32_t customIndexMask, flags;
    uint64_t address;
};
static_assert(sizeof(NativeInstance) == 64);
struct Snapshot {
    MeshletStreamGPUBLASHeader header;
    std::array<NativeInstance, kInstanceCount> instances;
    std::array<MeshletStreamGPUInstanceBLAS, kInstanceCount> allocations;
    std::array<uint32_t, 16> cutCache;
    std::array<ProbeObservation, kInstanceCount * 256> rays;
    std::vector<MeshletStreamGPURTRepresentationGroup> groups;
};

class StreamRTRepresentationTest final : public RHITest {
public:
    StreamRTRepresentationTest()
    {
        type = RHITestType::Rendering;
        name = "stream_rt_representation_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto require = [](bool condition, const std::string& message) {
            if (!condition) { throw std::runtime_error(message); }
        };
        const auto checked = [](const Result<>& result, const char* operation) {
            if (!result) { throw std::runtime_error(std::string(operation) + ": " + resultToString(result)); }
        };
        std::atomic_uint validationErrors = 0;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "RT representation lifecycle",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true, .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true, .enableClusterAccelerationStructure = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}}).transform([&](auto value) { device = std::move(value); });
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("CLAS unavailable")
                : RHITestResult::fail("Device creation failed");
        }
        std::string phase = "fixture setup";
        MeshletStreamGPUBLASHeader lastHeader;
        try {
            const auto original = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
            const auto directory = std::filesystem::absolute(context.outputDirectory / "RTRepresentation");
            std::filesystem::create_directories(directory);
            nlohmann::json document;
            { std::ifstream input(original); input >> document; }
            // One primitive with enough groups to cross a 64-lane compaction
            // tile. The second copy is behind the first, so the authored direct
            // and secondary rays still have the same front surface oracle.
            scene::Scene sourceScene;
            require(sourceScene.load(original), sourceScene.lastLoadResult().error);
            const auto& mesh = sourceScene.renderPrimitives()[0];
            require(mesh.positions.size() == mesh.normals.size(), "Bunny fixture needs authored normals");
            const uint32_t vertices = static_cast<uint32_t>(mesh.positions.size());
            const uint32_t indices = static_cast<uint32_t>(mesh.indices.size());
            const uint32_t attributeBytes = vertices * 2 * 3 * sizeof(float);
            const uint32_t indexBytes = indices * 2 * sizeof(uint32_t);
            const auto binaryPath = directory / "multi-tile.bin";
            {
                std::ofstream output(binaryPath, std::ios::binary);
                for (uint32_t copy = 0; copy < 2; ++copy) {
                    for (const auto& position : mesh.positions) {
                        const float value[]{position.x, position.y, position.z - float(copy) * .18f};
                        output.write(reinterpret_cast<const char*>(value), sizeof(value));
                    }
                }
                for (uint32_t copy = 0; copy < 2; ++copy) {
                    for (const auto& normal : mesh.normals) {
                        const float value[]{normal.x, normal.y, normal.z};
                        output.write(reinterpret_cast<const char*>(value), sizeof(value));
                    }
                }
                for (uint32_t copy = 0; copy < 2; ++copy) {
                    for (const uint32_t index : mesh.indices) {
                        const uint32_t value = index + copy * vertices;
                        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
                    }
                }
            }
            document["buffers"] = nlohmann::json::array({{{"uri", "multi-tile.bin"}, {"byteLength", attributeBytes * 2 + indexBytes}}});
            document["bufferViews"] = nlohmann::json::array({
                {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", attributeBytes}},
                {{"buffer", 0}, {"byteOffset", attributeBytes}, {"byteLength", attributeBytes}},
                {{"buffer", 0}, {"byteOffset", attributeBytes * 2}, {"byteLength", indexBytes}}});
            document["accessors"] = nlohmann::json::array({
                {{"bufferView", 0}, {"componentType", 5126}, {"count", vertices * 2}, {"type", "VEC3"},
                    {"min", {mesh.localBounds.min.x, mesh.localBounds.min.y, mesh.localBounds.min.z - .18f}},
                    {"max", {mesh.localBounds.max.x, mesh.localBounds.max.y, mesh.localBounds.max.z}}},
                {{"bufferView", 1}, {"componentType", 5126}, {"count", vertices * 2}, {"type", "VEC3"}},
                {{"bufferView", 2}, {"componentType", 5125}, {"count", indices * 2}, {"type", "SCALAR"}}});
            document["meshes"][0]["primitives"][0]["attributes"] = {{"POSITION", 0}, {"NORMAL", 1}};
            document["meshes"][0]["primitives"][0]["indices"] = 2;
            document["nodes"] = nlohmann::json::array();
            document["scenes"] = nlohmann::json::array({{{"nodes", nlohmann::json::array()}}});
            document["scene"] = 0;
            for (uint32_t i = 0; i < kInstanceCount; ++i) {
                // Distinct nonuniform transforms test that BLAS sharing never
                // replaces the TLAS identity or transform with its representative.
                document["nodes"].push_back({{"mesh", 0}, {"name", "copy" + std::to_string(i)},
                    {"translation", {float(i) * .3f, 0.f, 0.f}},
                    {"scale", {i % 2 ? .9f : 1.f, 1.f, i % 2 ? 1.1f : 1.f}}});
                document["scenes"][0]["nodes"].push_back(i);
            }
            const auto source = directory / "scene.gltf";
            { std::ofstream output(source); output << document.dump(2); }
            std::string log;
            const auto assetPath = directory / "shared-offline-multitile.meshstream.bin";
            // Offline cooking deduplicates authored mesh primitives before
            // expanding nodes. The resident Scene importer intentionally emits
            // a RenderPrimitive per node and would not exercise BLAS sharing.
            require(scene::buildMeshletStreamAssetOffline({.sourcePath = source,
                .outputPath = assetPath, .compressionMode = scene::MeshletStreamPayloadCompression::ByteRle}, log), log);
            scene::MeshletStreamAsset asset;
            require(asset.open(assetPath, log), log);
            require(asset.primitiveCount() == 1 && asset.instanceCount() == kInstanceCount &&
                asset.primitives()[0].lodLevelCount >= 3, "Fixture must share one multi-level primitive: primitives=" +
                    std::to_string(asset.primitiveCount()) + ", instances=" + std::to_string(asset.instanceCount()) +
                    ", levels=" + std::to_string(asset.lodLevelCount()));
            require(asset.groupCount() > 64 && asset.groupCount() % 64 != 0,
                "Fixture must exercise multiple group tiles and a partial final tile");

            MeshletStreamRuntime runtime;
            runtime.setDebugReadbackEnabled(true);
            MeshletStreamRuntimeDesc runtimeDesc{.sourcePath = source, .streamAssetPath = assetPath,
                .maxResidentBytes = 32ull << 20, .maxResidentPages = 1024,
                .maxPageUploadsPerFrame = 256, .maxGpuPageRequests = 4096, .maxGpuPageUnloadRequests = 1024,
                .maxActiveGroups = 16384, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 32768,
                .pageLoadConcurrency = 0, .maxPageLoadsInFlight = 256, .queuedFrameCount = 2,
                .enableClusterRtx = true, .enableClas = true, .compactClas = true,
                .maxClasBytes = 64ull << 20, .maxClasBuildClusters = 8192,
                .maxBlasClusterReferences = 8192, .maxBlasBytes = 32ull << 20,
                .enableBlasGPUAllocator = true, .maxBlasBuilds = 3,
                .maxFallbackBlasBytes = 4ull << 20, .prefetchPages = false};
            {
                MeshletStreamRuntime rejected;
                auto limited = runtimeDesc;
                limited.maxBlasStagingBytes = 0;
                const auto result = rejected.initialize(*device, limited, log);
                require(hasError(result, Error::OutOfMemory) && !rejected.ready() &&
                    log.find("maxBlasStagingBytes") != std::string::npos,
                    "Insufficient implicit build staging silently changed allocator or RT quality");
            }
            require(bool(runtime.initialize(*device, runtimeDesc, log)), log);
            const auto clusterProperties = device->queryClusterAccelerationStructureProperties();
            require(bool(clusterProperties), "BLAS alignment query failed");

            ShaderCompileResult compiled;
            const char* capability = "spvRayQueryKHR";
            require(bool(compileSlangShaderToSpirv({.moduleName = "StreamRTRepresentationProbe",
                .entryPointName = "streamRTRepresentationProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .capabilities = {&capability, 1}}, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); })), compiled.diagnostics);
            const ComputeResourceBindingDesc layout[] = {
                {METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, scene), ComputeResourceBindingKind::AccelerationStructure},
                {METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, output)}};
            ComputeProgram probe;
            require(bool(probe.initialize(*device, {.spirv = compiled.spirv, .bindings = layout,
                .resourceParameterSize = sizeof(UnifiedTopLevelProbeResources)}, log)), log);
            auto* queue = device->getQueue(QueueType::Graphics);
            QueueSubmissionTracker tracker;
            require(bool(tracker.initialize(*device, *queue)), "Submission tracker");
            std::unique_ptr<Streamer> streamer;
            StreamerDesc streamerDesc;
            streamerDesc.constantBufferSize = 4096;
            require(bool(createStreamer(*device, streamerDesc).transform([&](auto value) { streamer = std::move(value); })), "Streamer");
            struct Slot {
                explicit Slot(uint32_t index) : frame(index) {}
                RenderFrameContext frame;
                std::unique_ptr<CommandPool> pool;
                std::unique_ptr<CommandBuffer> commands;
                std::unique_ptr<Buffer> readback, rays;
                bool pending = false, traced = false, hasCutCache = false;
                uint64_t representationBytes = 0;
            };
            std::array<Slot, 2> slots{Slot{0}, Slot{1}};
            struct Drain {
                Queue& queue;
                std::array<Slot, 2>& slots;
                ~Drain()
                {
                    for (auto& slot : slots) { slot.frame.cancel(); }
                    (void)queue.waitIdle();
                }
            } drain{*queue, slots};
            constexpr uint64_t kReadbackBytes = sizeof(MeshletStreamGPUBLASHeader) +
                kInstanceCount * (sizeof(NativeInstance) + sizeof(MeshletStreamGPUInstanceBLAS)) + sizeof(Snapshot::cutCache);
            const uint64_t representationReadbackBytes = uint64_t(asset.groupCount()) * 3 * sizeof(MeshletStreamGPURTRepresentationGroup);
            for (auto& slot : slots) {
                require(bool(device->createCommandPool(*queue).transform([&](auto value) { slot.pool = std::move(value); })) &&
                    bool(slot.pool->createCommandBuffer().transform([&](auto value) { slot.commands = std::move(value); })) &&
                    bool(device->createBuffer({.size = kReadbackBytes + representationReadbackBytes, .usage = BufferUsageBits::TransferDestination,
                        .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { slot.readback = std::move(value); })) &&
                    bool(device->createBuffer({.size = sizeof(Snapshot::rays), .structureStride = sizeof(ProbeObservation),
                        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
                        .transform([&](auto value) { slot.rays = std::move(value); })), "Slot resources");
            }
            MeshletStreamFrameDesc view{.width = 512, .height = 192, .selectedLodLevel = 0, .enableGpuLodSelection = false};
            view.camera = {.eye = {1.f, .11f, 2.f}, .center = {1.f, .11f, 0.f}, .znear = .001f, .zfar = 10.f};
            uint64_t frameId = 0;
            nlohmann::json frames = nlohmann::json::array();
            phase = "candidate finest";
            // Prove the GPU output is a cut of the authored DAG independently
            // of residency selection: a selected cluster excludes every selected
            // descendant; an unselected cluster delegates to a complete child.
            const auto checkLegalCuts = [&](const Snapshot& snapshot) {
                uint32_t previousRepresentation = 0;
                for (const auto& entry : snapshot.groups) {
                    require(entry.representationIndex < asset.primitiveCount() * 3 &&
                        entry.representationIndex >= previousRepresentation,
                        "Parallel compaction reordered or corrupted representation ownership");
                    previousRepresentation = entry.representationIndex;
                }
                for (uint32_t representation = 0; representation < asset.primitiveCount() * 3; ++representation) {
                    std::vector<uint32_t> masks(asset.groupCount());
                    bool nonempty = false;
                    uint32_t previousGroup = 0;
                    for (const auto& entry : snapshot.groups) {
                        if (entry.representationIndex != representation) { continue; }
                        require(entry.pageIndex < asset.pageCount(), "Representation emitted an invalid page");
                        const uint32_t group = asset.pages()[entry.pageIndex].lodGroupIndex;
                        const auto& authored = asset.groups()[group];
                        require(authored.pageIndex == entry.pageIndex && authored.primitiveIndex == representation / 3 &&
                            authored.clusterCount == entry.clusterCount, "Representation group identity differs from its authored page");
                        const uint32_t validMask = entry.clusterCount == 32 ? UINT32_MAX : (1u << entry.clusterCount) - 1u;
                        require(entry.clusterMask != 0 && (entry.clusterMask & ~validMask) == 0,
                            "Representation contains an empty or out-of-range cluster mask");
                        require(masks[group] == 0 && (!nonempty || group > previousGroup),
                            "Parallel compaction duplicated or reordered representation groups");
                        masks[group] = entry.clusterMask;
                        nonempty = true;
                        previousGroup = group;
                    }
                    if (!nonempty) { continue; }
                    std::vector<int8_t> contains(asset.groupCount(), -1);
                    std::function<bool(uint32_t)> hasSelection = [&](uint32_t group) {
                        if (contains[group] >= 0) { return contains[group] != 0; }
                        bool selected = masks[group] != 0;
                        const auto& authored = asset.groups()[group];
                        for (uint32_t cluster = 0; cluster < authored.clusterCount; ++cluster) {
                            const uint32_t child = asset.refinedGroups()[authored.clusterRefinedOffset + cluster];
                            if (child != scene::kMeshletStreamInvalidGroupIndex) { selected |= hasSelection(child); }
                        }
                        contains[group] = selected ? 1 : 0;
                        return selected;
                    };
                    std::vector<bool> visited(asset.groupCount());
                    std::function<void(uint32_t)> visit = [&](uint32_t group) {
                        if (visited[group]) { return; }
                        visited[group] = true;
                        const auto& authored = asset.groups()[group];
                        for (uint32_t cluster = 0; cluster < authored.clusterCount; ++cluster) {
                            const uint32_t child = asset.refinedGroups()[authored.clusterRefinedOffset + cluster];
                            if ((masks[group] & (1u << cluster)) != 0) {
                                require(child == scene::kMeshletStreamInvalidGroupIndex || !hasSelection(child),
                                    "Representation overlaps a selected ancestor with its descendants");
                            } else {
                                require(child != scene::kMeshletStreamInvalidGroupIndex,
                                    "Representation omitted a leaf cluster (partial cut)");
                                visit(child);
                            }
                        }
                    };
                    for (uint32_t root : asset.primitiveTerminalGroups(representation / 3)) { visit(root); }
                    for (uint32_t group = 0; group < masks.size(); ++group) {
                        require(masks[group] == 0 || visited[group], "Representation selected an unreachable descendant");
                    }
                }
            };
            const auto complete = [&](Slot& slot) {
                require(slot.pending, "Wait requires a submitted slot");
                checked(slot.frame.wait(10'000'000'000ull), "Frame wait");
                slot.pending = false;
                Snapshot snapshot{};
                slot.readback->invalidate();
                const auto* data = static_cast<const uint8_t*>(slot.readback->map());
                require(data != nullptr, "Readback map");
                std::memcpy(&snapshot.header, data, sizeof(snapshot.header));
                lastHeader = snapshot.header;
                std::memcpy(snapshot.instances.data(), data + sizeof(snapshot.header), sizeof(snapshot.instances));
                std::memcpy(snapshot.allocations.data(), data + sizeof(snapshot.header) + sizeof(snapshot.instances), sizeof(snapshot.allocations));
                if (slot.hasCutCache) {
                    std::memcpy(snapshot.cutCache.data(), data + kReadbackBytes - sizeof(snapshot.cutCache), sizeof(snapshot.cutCache));
                }
                require(uint64_t(snapshot.header.representationGroupCount) * sizeof(MeshletStreamGPURTRepresentationGroup) <= slot.representationBytes,
                    "Representation count exceeded the output allocation");
                snapshot.groups.resize(snapshot.header.representationGroupCount);
                if (!snapshot.groups.empty()) {
                    std::memcpy(snapshot.groups.data(), data + kReadbackBytes,
                        snapshot.groups.size() * sizeof(MeshletStreamGPURTRepresentationGroup));
                }
                slot.readback->unmap();
                checkLegalCuts(snapshot);
                if (slot.traced) {
                    slot.rays->invalidate(); const void* rays = slot.rays->map(); require(rays != nullptr, "Ray readback");
                    std::memcpy(snapshot.rays.data(), rays, sizeof(snapshot.rays)); slot.rays->unmap();
                }
                frames.push_back({{"phase", phase}, {"builds", snapshot.header.blasBuildCount},
                    {"representations", snapshot.header.representationCount},
                    {"sharing", snapshot.header.sharingInstanceCount},
                    {"merged", snapshot.header.mergedInstanceCount},
                    {"cached", snapshot.header.cachedInstanceCount},
                    {"groups", snapshot.header.representationGroupCount},
                    {"groupOverflow", snapshot.header.representationGroupOverflow},
                    {"references", snapshot.header.liveClusterReferences},
                    {"storageBytes", snapshot.header.arenaUsedBytes},
                    {"cutCacheReuseBits", snapshot.cutCache[15]},
                    {"mergedPageTableSerial", snapshot.cutCache[0]},
                    {"mergedClasRevision", snapshot.cutCache[1]},
                    {"semanticChanges", snapshot.header.semanticChangeCount}});
                return snapshot;
            };
            const auto record = [&](Slot& slot, bool cancel = false, bool trace = true) {
                require(!slot.pending, "A pending slot cannot be rerecorded");
                runtime.prepareMaintenance();
                checked(slot.frame.begin(++frameId), "Frame begin");
                checked(slot.pool->reset(), "Command pool reset");
                checked(slot.commands->begin(slot.frame.submissionContext()), "Command buffer begin");
                checked(streamer->beginFrame(slot.frame), "Streamer begin");
                checked(runtime.cmdBeginFrame(*slot.commands, *streamer, view), "Stream runtime begin");
                checked(streamer->copyStreamedData(*slot.commands), "Copy streamed data");
                checked(runtime.cmdPreTraversal(*slot.commands, view), "Stream pre-traversal");
                checked(runtime.cmdPostTraversal(*slot.commands), "Stream post-traversal");
                checked(runtime.cmdEndFrame(*slot.commands), "Stream end frame");
                std::vector<DebugResourceBinding> bindings;
                runtime.appendDebugBindings(bindings, "test.");
                uint64_t offset = 0;
                slot.hasCutCache = false;
                for (const auto& [name, bytes] : {
                    std::pair{"test.blasHeader", uint64_t(sizeof(MeshletStreamGPUBLASHeader))},
                    std::pair{"test.tlasInstances", uint64_t(kInstanceCount * sizeof(NativeInstance))},
                    std::pair{"test.instanceBlas", uint64_t(kInstanceCount * sizeof(MeshletStreamGPUInstanceBLAS))},
                    std::pair{"test.rtRepresentationCache", uint64_t(sizeof(Snapshot::cutCache))}}) {
                    const auto found = std::find_if(bindings.begin(), bindings.end(), [&](const auto& entry) { return entry.id == name; });
                    if (std::string_view(name) == "test.rtRepresentationCache") {
                        slot.hasCutCache = found != bindings.end();
                        if (!slot.hasCutCache) { continue; } // Legacy runtime has no representation cache.
                    }
                    require(found != bindings.end(), std::string("Missing debug binding ") + name);
                    BufferBarrierDesc barrier{.buffer = found->buffer,
                        .before = resourceSyncScope(found->state, PipelineStageBits::AllCommands),
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    require(bool(slot.commands->synchronize({.buffers = {&barrier, 1}})), "Readback barrier");
                    auto src = found->buffer->slice({0, bytes}); auto dst = slot.readback->slice({offset, bytes});
                    require(bool(src) && bool(dst) && bool(slot.commands->copyBuffer(*src, *dst)), "Readback copy");
                    std::swap(barrier.before, barrier.after);
                    require(bool(slot.commands->synchronize({.buffers = {&barrier, 1}})), "Restore barrier");
                    offset += bytes;
                }
                slot.representationBytes = 0;
                const auto representationGroups = std::find_if(bindings.begin(), bindings.end(), [](const auto& entry) {
                    return entry.id == "test.rtRepresentationGroups";
                });
                if (representationGroups != bindings.end()) {
                    slot.representationBytes = representationGroups->buffer->desc().size;
                    require(slot.representationBytes <= representationReadbackBytes, "Representation readback capacity");
                    BufferBarrierDesc barrier{.buffer = representationGroups->buffer,
                        .before = resourceSyncScope(representationGroups->state, PipelineStageBits::AllCommands),
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    checked(slot.commands->synchronize({.buffers = {&barrier, 1}}), "Representation readback barrier");
                    auto src = representationGroups->buffer->slice({0, slot.representationBytes});
                    auto dst = slot.readback->slice({kReadbackBytes, slot.representationBytes});
                    require(bool(src) && bool(dst), "Representation readback slices");
                    checked(slot.commands->copyBuffer(*src, *dst), "Representation readback copy");
                    std::swap(barrier.before, barrier.after);
                    checked(slot.commands->synchronize({.buffers = {&barrier, 1}}), "Representation restore barrier");
                }
                slot.traced = trace && runtime.tlasReady();
                if (slot.traced) {
                    const ComputeDispatchBinding resources[] = {
                        {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, scene),
                            .accelerationStructure = runtime.accelerationStructure()},
                        {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, output), .buffer = slot.rays.get()}};
                    checked(probe.dispatch({.commandBuffer = slot.commands.get(), .bindings = resources,
                        .groupCountX = kInstanceCount * 4}), "Ray query probe");
                }
                checked(slot.commands->end(), "Command buffer end");
                if (cancel) { slot.frame.cancel(); streamer->endFrame(); return; }
                CommandBuffer* commands[] = {slot.commands.get()};
                checked(tracker.submit({.commandBuffers = commands}, slot.frame), "Tracker submit");
                slot.pending = true;
                streamer->endFrame();
            };
            const auto checkIdentity = [&](const Snapshot& snapshot) {
                std::array<uint32_t, kInstanceCount> hits{};
                for (uint32_t i = 0; i < kInstanceCount; ++i) {
                    const auto& instance = snapshot.instances[i];
                    require((instance.customIndexMask & 0xffffffu) == i && (instance.customIndexMask >> 24) == 255,
                        "Shared representation changed TLAS identity or visibility");
                    require(instance.address != 0 && std::abs(instance.transform[0][3] - float(i) * .3f) < 1e-6f &&
                        std::abs(instance.transform[0][0] - (i % 2 ? .9f : 1.f)) < 1e-6f &&
                        std::abs(instance.transform[2][2] - (i % 2 ? 1.1f : 1.f)) < 1e-6f,
                        "Shared representation changed instance transform");
                    for (uint32_t pixel = 0; pixel < 256; ++pixel) {
                        const auto& ray = snapshot.rays[i * 256 + pixel];
                        require(ray.directInstance == ray.reflectedInstance, "Reflected ray changed hit identity");
                        if (ray.directInstance == UINT32_MAX) { continue; }
                        require(ray.directInstance == i, "Ray hit the representative instance instead of the authored instance");
                        require(std::isfinite(ray.directDistance) && std::isfinite(ray.reflectedDistance) &&
                            std::abs(ray.directDistance - ray.reflectedDistance - .1f) < 1e-4f,
                            "Secondary ray hit distance disagrees with mirror-plane geometry");
                        ++hits[i];
                    }
                    require(hits[i] >= 20, "Shared BLAS lost visible geometry or secondary-ray reachability");
                }
            };

            Snapshot stable{};
            uint32_t built = 0;
            for (uint32_t i = 0; i < 100; ++i) {
                record(slots[i % 2]); stable = complete(slots[i % 2]); built += stable.header.blasBuildCount;
                require(stable.header.blasBuildCount <= 3, "Build count exceeded the unique primitive representation bound");
            }
            require(built > 0 && runtime.sceneReady() && runtime.tlasReady(), "Representations never built");
            const auto convergedGroups = stable.groups;
            uint32_t stableAllocatorSamples = 0;
            for (uint32_t i = 0; i < 32; ++i) {
                record(slots[i % 2]); stable = complete(slots[i % 2]);
                require(stable.header.blasBuildCount == 0 && stable.header.arenaRepack == 0 && stable.header.semanticChangeCount == 0,
                    "Stable representation rebuilt, repacked or invalidated shading");
                const uint32_t demandedModes = (stable.header.mergedInstanceCount ? 1u : 0u) |
                    (stable.header.sharingInstanceCount ? 2u : 0u) | (stable.header.cachedInstanceCount ? 4u : 0u);
                require(demandedModes != 0 && (stable.cutCache[15] & demandedModes) == demandedModes,
                    "Stable representation regenerated its cut instead of hitting the GPU cache");
                require(stable.groups.size() == convergedGroups.size() &&
                    std::memcmp(stable.groups.data(), convergedGroups.data(), stable.groups.size() * sizeof(MeshletStreamGPURTRepresentationGroup)) == 0,
                    "Parallel representation output is not deterministic across stable frames");
                checkIdentity(stable);
                const auto statistics = runtime.profilingStats();
                require(statistics.blasGPUAllocatorEnabled, "Candidate runtime did not enable the GPU BLAS allocator");
                if (statistics.blasFeedbackAvailable && statistics.blasEncodedBytes != 0) {
                    ++stableAllocatorSamples;
                    require(statistics.blasBuildCount == 0 && statistics.blasRelocatedCount == 0 &&
                        statistics.blasRelocatedBytes == 0 && statistics.blasArenaRepack == 0,
                        "Completed stationary feedback reported a rebuild, Move or repack");
                    require(statistics.blasAllocatedBytes >= statistics.blasEncodedBytes &&
                        uint64_t(statistics.blasAllocatedBytes) - statistics.blasEncodedBytes <
                            uint64_t(asset.primitiveCount()) * kMeshletStreamRTRepresentationsPerPrimitive *
                            clusterProperties->bottomLevelStorageAlignment,
                        "Runtime actual-size allocation retained bucket-sized internal waste");
                    require(uint64_t(statistics.blasAllocatedBytes) + statistics.blasFreeBytes == statistics.blasPersistentBytes &&
                        statistics.blasLargestFreeBytes <= statistics.blasFreeBytes &&
                        statistics.blasStagingBytes != 0 && statistics.blasStagingBytes <= runtimeDesc.maxBlasStagingBytes,
                        "Runtime persistent/free/staging accounting violates allocator budgets");
                }
            }
            require(stableAllocatorSamples != 0, "No completed actual-size allocator feedback during 32 stationary frames");
            std::set<uint64_t> addresses;
            for (const auto& instance : stable.instances) { addresses.insert(instance.address); }
            require(addresses.size() == 1, "Identical finest cuts did not share one primitive representation");
            require(stable.header.liveClusterReferences > 0 && stable.header.requestedInstances <= 3,
                "BLAS budget was charged per instance instead of per representation");
            const Snapshot finestCandidate = stable;
            uint32_t sharingFrames = stable.header.sharingInstanceCount != 0;
            uint32_t mergedFrames = stable.header.mergedInstanceCount != 0;
            uint32_t sharedCutCacheHits = 0;

            phase = "candidate discrete cache";
            const uint32_t cacheLevel = asset.primitives()[0].lodLevelCount > 3
                ? asset.primitives()[0].lodLevelCount - 3 : 0;
            view.selectedLodLevel = cacheLevel;
            for (uint32_t i = 0; i < 16; ++i) {
                record(slots[i % 2]); stable = complete(slots[i % 2]);
            }
            require(stable.header.cachedInstanceCount == kInstanceCount && stable.header.blasBuildCount == 0,
                "Fully resident discrete level failed to converge to the persistent cache");
            require((stable.cutCache[15] & 4u) != 0, "Stable discrete representation missed its cut cache");
            checkIdentity(stable);
            const Snapshot discreteCandidate = stable;
            view.selectedLodLevel = 0;
            record(slots[0]); complete(slots[0]);
            record(slots[1]); complete(slots[1]);
            view.selectedLodLevel = cacheLevel;
            record(slots[0]); stable = complete(slots[0]);
            require(stable.header.cachedInstanceCount == kInstanceCount && stable.header.blasBuildCount == 0,
                "A fully resident discrete BLAS was discarded while its consumers selected a finer representation");
            checkIdentity(stable);
            view.selectedLodLevel = 0;
            record(slots[0]); complete(slots[0]);
            record(slots[1]); complete(slots[1]);

            phase = "candidate view-dependent sharing and merging";
            view.enableGpuLodSelection = true;
            view.selectedLodLevel = kMeshletStreamNoDebugLODOverride;
            view.camera.eye = {-.016f, .11f, .25f};
            view.camera.center = {-.016f, .11f, 0.f};
            for (const float error : {.25f, .5f, 1.f, 2.f, 4.f, 8.f, 16.f, 32.f}) {
                view.lodPixelError = error;
                for (uint32_t i = 0; i < 4; ++i) {
                    record(slots[i % 2]); stable = complete(slots[i % 2]);
                    sharingFrames += stable.header.sharingInstanceCount != 0;
                    mergedFrames += stable.header.mergedInstanceCount != 0;
                    sharedCutCacheHits += (stable.cutCache[15] & 2u) != 0;
                    require(stable.header.blasBuildCount <= 3 && stable.header.requestedInstances <= 3,
                        "View-dependent selection charged duplicate per-instance representations");
                    checkIdentity(stable);
                }
            }
            require(sharingFrames != 0 && mergedFrames != 0, "Fixture did not exercise both Sharing and Merging");
            require(sharedCutCacheHits != 0, "Repeated view-dependent sharing never hit its exact-mask cut cache");
            view.enableGpuLodSelection = false;
            view.selectedLodLevel = 0;
            view.camera.eye = {1.f, .11f, 2.f};
            view.camera.center = {1.f, .11f, 0.f};
            view.lodPixelError = 1.5f;
            for (uint32_t i = 0; i < 4; ++i) { record(slots[i % 2]); complete(slots[i % 2]); }

            phase = "candidate cancellation and in-flight dependencies";
            // A canceled coarse selection must not replace the last submitted
            // representation. Both slots remain independent until GPU completion.
            view.selectedLodLevel = 2;
            record(slots[0], true);
            view.selectedLodLevel = 0;
            record(slots[0]);
            record(slots[1]);
            const auto first = complete(slots[0]);
            const auto second = complete(slots[1]);
            require(first.header.blasBuildCount == 0 && second.header.blasBuildCount == 0 &&
                first.header.semanticChangeCount == 0 && second.header.semanticChangeCount == 0,
                "Canceled selection or a later in-flight frame invalidated the retained representation");
            checkIdentity(first); checkIdentity(second);

            // Retry the exact selection whose recording was cancelled. Restoring
            // fine LOD alone would not catch a CPU cache key committed before its
            // new GPU cut was submitted.
            view.selectedLodLevel = cacheLevel;
            record(slots[0], true);
            record(slots[0]);
            const auto retriedDiscrete = complete(slots[0]);
            checkIdentity(retriedDiscrete);
            require(retriedDiscrete.header.cachedInstanceCount == kInstanceCount &&
                retriedDiscrete.groups.size() == discreteCandidate.groups.size() &&
                std::memcmp(retriedDiscrete.groups.data(), discreteCandidate.groups.data(),
                    retriedDiscrete.groups.size() * sizeof(MeshletStreamGPURTRepresentationGroup)) == 0,
                "Cancelled cut-cache selection changed the retried discrete representation");
            for (uint32_t ray = 0; ray < retriedDiscrete.rays.size(); ++ray) {
                const auto& actual = retriedDiscrete.rays[ray];
                const auto& expected = discreteCandidate.rays[ray];
                require(actual.directInstance == expected.directInstance &&
                    actual.reflectedInstance == expected.reflectedInstance &&
                    std::abs(actual.directDistance - expected.directDistance) <= 2e-5f &&
                    std::abs(actual.reflectedDistance - expected.reflectedDistance) <= 2e-5f,
                    "Cancelled cut-cache retry changed discrete ray coverage or geometry");
            }
            view.selectedLodLevel = 0;
            for (uint32_t i = 0; i < 2; ++i) { record(slots[i]); complete(slots[i]); }

            // Retire a non-root dependency while the old TLAS is submitted. Its
            // storage must survive that submission and the next frame must stop
            // publishing the retired CLAS through a cached representation.
            const auto resident = runtime.residency().residentPages();
            const auto detail = std::find_if(resident.begin(), resident.end(), [&](uint32_t page) {
                return runtime.residency().pageState(page) == MeshletStreamPageResidencyState::Resident &&
                    asset.pages()[page].lodLevel == 0;
            });
            require(detail != resident.end(), "No detail dependency to retire");
            record(slots[0]);
            runtime.clasPool()->retirePages(std::span(&*detail, 1));
            // Cancellation must not publish the new dependency epochs. Retrying
            // the same invalidated cut must miss, then converge to cache hits.
            record(slots[1], true);
            record(slots[1]);
            const auto oldVersion = complete(slots[0]);
            checkIdentity(oldVersion);
            for (uint32_t ray = 0; ray < oldVersion.rays.size(); ++ray) {
                const auto& retained = oldVersion.rays[ray];
                const auto& original = finestCandidate.rays[ray];
                require(retained.directInstance == original.directInstance &&
                    retained.reflectedInstance == original.reflectedInstance &&
                    std::abs(retained.directDistance - original.directDistance) <= 2e-5f &&
                    std::abs(retained.reflectedDistance - original.reflectedDistance) <= 2e-5f,
                    "Retiring a dependency changed the geometry of an already submitted version");
            }
            const auto retired = complete(slots[1]);
            require(retired.header.blasBuildCount != 0 || retired.header.semanticChangeCount != 0,
                "Retiring a dependency left the old representation published unchanged");
            require(retired.header.mergedInstanceCount != 0 && (retired.cutCache[15] & 1u) == 0,
                "Cancelled dependency change published a cut-cache hit before its GPU work ran");
            checkIdentity(retired);
            for (uint32_t i = 0; i < 16; ++i) {
                record(slots[i % 2]); stable = complete(slots[i % 2]);
            }
            require(stable.header.blasBuildCount == 0, "Residency change failed to converge");
            require((stable.cutCache[15] & 1u) != 0, "Retried dependency change never converged to a merged cut-cache hit");
            checkIdentity(stable);

            // Correctness oracle uses the unchanged per-instance path with a
            // generous budget. It is not a performance comparison: only the
            // submitted finest geometry and the authored rays are held equal.
            phase = "legacy finest oracle";
            runtime.reset();
            runtimeDesc.enableBlasRepresentations = false;
            runtimeDesc.enableBlasGPUAllocator = false;
            runtimeDesc.maxBlasBuilds = kInstanceCount;
            runtimeDesc.maxBlasClusterReferences = 64 * 1024;
            runtimeDesc.maxBlasBytes = 128ull << 20;
            require(bool(runtime.initialize(*device, runtimeDesc, log)), log);
            for (uint32_t i = 0; i < 100; ++i) {
                record(slots[i % 2]); stable = complete(slots[i % 2]);
            }
            checkIdentity(stable);
            require(stable.header.blasBuildCount == 0 && stable.header.reusedInstances == kInstanceCount,
                "Legacy oracle did not converge to eight per-instance BLAS");
            require(!runtime.profilingStats().blasGPUAllocatorEnabled, "Legacy ray oracle used the candidate allocator");
            const Snapshot finestLegacy = stable;
            float maximumDistanceError = 0;
            for (uint32_t ray = 0; ray < finestCandidate.rays.size(); ++ray) {
                const auto& candidate = finestCandidate.rays[ray];
                const auto& baseline = stable.rays[ray];
                require(candidate.directInstance == baseline.directInstance &&
                    candidate.reflectedInstance == baseline.reflectedInstance,
                    "Shared finest representation changed legacy direct/secondary hit coverage or identity");
                const float error = std::max(std::abs(candidate.directDistance - baseline.directDistance),
                    std::abs(candidate.reflectedDistance - baseline.reflectedDistance));
                maximumDistanceError = std::max(maximumDistanceError, error);
                require(error <= 2e-5f, "Shared finest representation changed legacy ray hit distance");
            }

            // An exact one-cut reference budget and one build slot must serve
            // all eight instances. Populate the discrete cache first, then
            // demand fine geometry so idle-cache eviction is exercised too.
            phase = "candidate one-representation budget";
            runtime.reset();
            runtimeDesc.enableBlasRepresentations = true;
            runtimeDesc.enableBlasGPUAllocator = true;
            runtimeDesc.maxBlasBuilds = 1;
            runtimeDesc.maxBlasClusterReferences = finestLegacy.allocations[0].selectedClusterCount;
            runtimeDesc.maxBlasBytes = 32ull << 20;
            require(runtimeDesc.maxBlasClusterReferences > 0, "Legacy oracle did not expose the finest reference count");
            require(bool(runtime.initialize(*device, runtimeDesc, log)), log);
            for (uint32_t i = 0; i < 100; ++i) { record(slots[i % 2]); stable = complete(slots[i % 2]); }
            require(stable.header.blasBuildCount == 0 && stable.header.requestedInstances == 1 &&
                stable.header.referenceBudgetRejected == 0 && stable.header.buildBudgetRejected == 0,
                "Exact one-representation budget rejected duplicate instances");
            checkIdentity(stable);
            view.selectedLodLevel = cacheLevel;
            for (uint32_t i = 0; i < 16; ++i) { record(slots[i % 2]); stable = complete(slots[i % 2]); }
            require(stable.header.cachedInstanceCount == kInstanceCount && stable.header.blasBuildCount == 0,
                "Tight budget failed to cache the demanded discrete level");
            view.selectedLodLevel = 0;
            for (uint32_t i = 0; i < 16; ++i) {
                record(slots[i % 2]); stable = complete(slots[i % 2]);
                require(stable.header.referenceBudgetRejected == 0 && stable.header.buildBudgetRejected == 0 &&
                    stable.header.blasBuildCount <= 1, "Idle cache blocked the demanded representation");
            }
            require(stable.header.blasBuildCount == 0 && stable.header.arenaRepack == 0,
                "Tight representation budget caused a persistent rebuild/repack loop");
            checkIdentity(stable);

            view.selectedLodLevel = asset.primitives()[0].lodLevelCount - 1;
            record(slots[0]); complete(slots[0]);
            record(slots[1]); const auto terminalOracle = complete(slots[1]);
            phase = "candidate whole-cut overflow";
            runtime.reset();
            runtimeDesc.maxBlasClusterReferences = 1;
            view.selectedLodLevel = 0;
            require(bool(runtime.initialize(*device, runtimeDesc, log)), log);
            for (uint32_t i = 0; i < 100; ++i) { record(slots[i % 2]); stable = complete(slots[i % 2]); }
            for (uint32_t i = 0; i < 16; ++i) {
                record(slots[i % 2]); stable = complete(slots[i % 2]);
                require(stable.header.representationGroupOverflow != 0 && stable.groups.empty() &&
                    stable.header.blasBuildCount == 0 && stable.header.arenaRepack == 0 &&
                    stable.header.semanticChangeCount == 0,
                    "An oversized primitive published a partial cut or failed to converge to complete fallback");
                for (uint32_t ray = 0; ray < stable.rays.size(); ++ray) {
                    const auto& actual = stable.rays[ray];
                    const auto& expected = terminalOracle.rays[ray];
                    require(actual.directInstance == expected.directInstance &&
                        actual.reflectedInstance == expected.reflectedInstance &&
                        std::abs(actual.directDistance - expected.directDistance) <= 2e-5f &&
                        std::abs(actual.reflectedDistance - expected.reflectedDistance) <= 2e-5f,
                        "Representation overflow changed complete terminal-fallback ray coverage");
                }
            }
            require(validationErrors.load() == 0, "Vulkan validation errors during representation lifetime regression");
            { std::ofstream report(directory / "Frames.json"); report << frames.dump(2); }
            { std::ofstream report(directory / "RayComparison.json"); report << nlohmann::json({
                {"pairs", finestCandidate.rays.size()}, {"rayCount", finestCandidate.rays.size() * 2},
                {"maximumDistanceError", maximumDistanceError}, {"tolerance", 2e-5f},
                {"sharingFrames", sharingFrames}, {"mergedFrames", mergedFrames}, {"sharedCutCacheHits", sharedCutCacheHits},
                {"candidateBuildBudget", 3}, {"legacyBuildBudget", kInstanceCount},
                {"candidateAllocator", "gpu"}, {"legacyAllocator", "legacy"},
                {"stableAllocatorSamples", stableAllocatorSamples}, {"stagingCapFailureRejected", true}}).dump(2); }
            for (const auto& [name, snapshot] : {std::pair{"CandidateRays.bin", &finestCandidate},
                    std::pair{"LegacyRays.bin", &finestLegacy}}) {
                std::ofstream output(directory / name, std::ios::binary);
                output.write(reinterpret_cast<const char*>(snapshot->rays.data()), sizeof(snapshot->rays));
            }
            return RHITestResult::pass("Eight instances; Sharing/Merging/Discrete Caching; 32 stable frames; unique budgets; cancellation; in-flight CLAS retirement; 4096 direct/secondary rays against legacy hit coverage, identity and distance");
        } catch (const std::exception& error) {
            return RHITestResult::fail(phase + ": " + error.what() + "; builds=" + std::to_string(lastHeader.blasBuildCount) +
                ", sharing=" + std::to_string(lastHeader.sharingInstanceCount) + ", merged=" + std::to_string(lastHeader.mergedInstanceCount) +
                ", cached=" + std::to_string(lastHeader.cachedInstanceCount) + ", refs=" + std::to_string(lastHeader.liveClusterReferences) +
                ", groups=" + std::to_string(lastHeader.representationGroupCount) + ", groupOverflow=" + std::to_string(lastHeader.representationGroupOverflow));
        }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamRTRepresentationTest);
} // namespace
} // namespace metallic::tests
