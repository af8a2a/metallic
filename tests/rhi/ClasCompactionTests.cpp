#include "RhiTest.h"
#include "Runtime/Render/Streamer/MeshletStreamCompactClasPool.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Scene/Scene.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderView.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
class ClasSizeMoveTest final : public RhiTest {
  public:
    ClasSizeMoveTest()
    {
        type = RhiTestType::Resource;
        name = "clas_actual_sizes_and_move";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "CLAS size/move regression",
                                    .enableValidation = context.enableValidation,
                                    .enableClusterAccelerationStructure = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(result, Error::Unsupported)) {
            return RhiTestResult::skip("CLAS unavailable");
        }
        const auto require = [](bool ok, const char* text) {
            if (!ok) {
                throw std::runtime_error(text);
            }
        };
        try {
            require(bool(result), "Device creation failed");
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
            auto infos = buffer(properties.triangleBuildInfoSize * 2, MemoryLocation::HostUpload);
            auto destinations = buffer(16, MemoryLocation::HostUpload),
                 sources = buffer(16, MemoryLocation::HostUpload);
            auto sizes = buffer(8, MemoryLocation::HostReadback);
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
                            {.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()})) &&
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
                            .indexBuffer = indices.get(),
                            .vertexBuffer = vertices.get(),
                            .destinationBuffer = temp.get(),
                            .destinationBufferOffset = i * stride,
                            .destinationSize = stride};
            }
            require(bool(commands->buildClusterAccelerationStructureTriangles(
                        {.clusters = input,
                         .clusterCount = 2,
                         .maxClusterTriangleCount = 128,
                         .maxClusterVertexCount = 128,
                         .scratchBuffer = scratch.get(),
                         .scratchBufferOffset = scratchOffset,
                         .buildInfoBuffer = infos.get(),
                         .destinationAddressBuffer = destinations.get(),
                         .destinationSizeBuffer = sizes.get()})),
                    "Build with size output failed");
            submit();
            sizes->invalidate();
            uint32_t actual[2];
            std::memcpy(actual, sizes->map(), sizeof(actual));
            sizes->unmap();
            require(actual[0] > 0 && actual[0] < stride && actual[0] % properties.clusterStorageAlignment == 0 &&
                        actual[0] == actual[1],
                    "Invalid actual sizes");
            auto compact = buffer(uint64_t(actual[0]) + actual[1], MemoryLocation::Device);
            require(bool(commandsPool->reset()) && bool(fence->reset()) && bool(commands->begin()),
                    "Move begin failed");
            ClusterAccelerationStructureMoveInfo moves[] = {{.sourceBuffer = temp.get(),
                                                             .sourceOffset = 0,
                                                             .destinationBuffer = compact.get(),
                                                             .destinationOffset = 0,
                                                             .size = actual[0]},
                                                            {.sourceBuffer = temp.get(),
                                                             .sourceOffset = stride,
                                                             .destinationBuffer = compact.get(),
                                                             .destinationOffset = actual[0],
                                                             .size = actual[1]}};
            auto move = ClusterAccelerationStructureMoveDesc{.objects = moves,
                                                             .objectCount = 2,
                                                             .sourceAddressBuffer = sources.get(),
                                                             .destinationAddressBuffer = destinations.get(),
                                                             .scratchBuffer = scratch.get(),
                                                             .scratchBufferOffset = scratchOffset};
            moves[1].destinationOffset += 1;
            require(hasError(commands->moveClusterAccelerationStructures(move), Error::InvalidArgument),
                    "Unaligned move accepted");
            moves[1].destinationOffset = 0;
            require(hasError(commands->moveClusterAccelerationStructures(move), Error::InvalidArgument),
                    "Overlapping destinations accepted");
            moves[1].destinationOffset = actual[0];
            ClusterAccelerationStructureBuildSizes exactMoveSizes;
            require(bool(device->queryClusterAccelerationStructureMoveSizes(2, uint64_t(actual[0]) + actual[1]).transform([&](auto rhiValue) { exactMoveSizes = std::move(rhiValue); })),
                    "Exact move sizes failed");
            if (exactMoveSizes.updateScratchSize > 1) {
                auto smallScratch = buffer(exactMoveSizes.updateScratchSize - 1, MemoryLocation::Device);
                auto invalidMove = move;
                invalidMove.scratchBuffer = smallScratch.get();
                invalidMove.scratchBufferOffset = (properties.scratchAlignment -
                    smallScratch->deviceAddress() % properties.scratchAlignment) % properties.scratchAlignment;
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
                moves[i] = {.sourceBuffer = compact.get(),
                            .sourceOffset = uint64_t(i) * actual[0],
                            .destinationBuffer = temp.get(),
                            .destinationOffset = i * stride,
                            .size = actual[i]};
            }
            require(bool(commands->moveClusterAccelerationStructures(move)), "Relocating compact CLAS failed");
            submit();
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream(context.outputDirectory / "ClasSizeMove.txt")
                << "worstCaseBytes=" << stride * 2 << " actualBytes=" << actual[0] + actual[1]
                << " moveScratchBytes=" << exactMoveSizes.updateScratchSize << '\n';
            return RhiTestResult::pass(
                "Actual GPU sizes, compact relocation, second relocation and invalid-range rejection");
        } catch (const std::exception& error) {
            return RhiTestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(ClasSizeMoveTest);
class CompactClasLifecycleTest final : public RhiTest {
  public:
    CompactClasLifecycleTest()
    {
        type = RhiTestType::Resource;
        name = "clas_compact_lifecycle";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::filesystem::create_directories(context.outputDirectory);
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Compact CLAS lifecycle",
                                    .enableValidation = context.enableValidation,
                                    .enableClusterAccelerationStructure = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(result, Error::Unsupported)) {
            return RhiTestResult::skip("CLAS unavailable");
        }
        if (!result) {
            return RhiTestResult::fail("Device failed");
        }
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        scene::Scene loadedScene;
        if (!loadedScene.load(scenePath)) {
            return RhiTestResult::fail("failed to load Stanford Bunny scene: " + loadedScene.lastLoadResult().error);
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
            return RhiTestResult::fail("buildMeshletStreamAsset failed: " + log);
        }

        scene::MeshletStreamAsset asset;
        if (!asset.open(streamAssetPath, log)) {
            return RhiTestResult::fail("MeshletStreamAsset::open failed: " + log);
        }
        uint32_t pageIndex = UINT32_MAX;
        for (const scene::MeshletStreamGroupInfo& group : asset.groups()) {
            if (group.maxQuadricError == scene::kMeshletStreamTerminalGroupError) {
                pageIndex = group.pageIndex;
                break;
            }
        }
        if (pageIndex == UINT32_MAX) {
            return RhiTestResult::fail("streamasset has no fallback page for CLAS pool build");
        }

        std::vector<uint8_t> decodedStorage;
        std::span<const uint8_t> decodedPayload;
        if (!scene::decodeMeshletStreamPayloadForDevice(asset.pages()[pageIndex], asset.pagePayload(pageIndex),
                                                        decodedStorage, decodedPayload, log)) {
            return RhiTestResult::fail("streamasset fallback page decode failed: " + log);
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = device->createBuffer(render::BufferDesc{
                .size = decodedPayload.size(),
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::ShaderDeviceAddress |
                         render::BufferUsageBits::AccelerationStructureBuildInput,
                .memoryLocation = render::MemoryLocation::HostUpload,
            }).transform([&](auto rhiValue) { pageBuffer = std::move(rhiValue); });
        if (!result || pageBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(stream CLAS page) returned ") + toString(result));
        }
        void* mapped = pageBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("stream CLAS page buffer did not map");
        }
        std::memcpy(mapped, decodedPayload.data(), decodedPayload.size());
        pageBuffer->flush(0, decodedPayload.size());
        pageBuffer->unmap();

        const auto require = [](bool ok, const std::string& message) {
            if (!ok) {
                throw std::runtime_error(message);
            }
        };
        try {
            MeshletStreamCompactClasPool pool;
            require(bool(pool.initialize(*device,
                                         {.asset = &asset,
                                          .maxStorageBytes = 1024 * 1024,
                                          .maxBuildClusters = asset.maxPageClusters(),
                                          .queuedFrameCount = 2},
                                         log)),
                    log);
            MeshletStreamClasPagePlan plan;
            require(buildMeshletStreamClasPagePlan(asset.pages()[pageIndex], decodedPayload, pageIndex,
                                                   pageIndex * asset.maxPageClusters(), plan, log),
                    log);
            const MeshletStreamClasPageBuild build{.pageIndex = pageIndex, .deviceOffsetBytes = 0, .plan = &plan};
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
            auto record = [&](bool request, bool cancel = false) {
                require(bool(frame.begin(++frameId)) && bool(commandPool->reset()) && bool(cmd->begin(&frame)),
                        "Begin failed");
                require(bool(pool.cmdBuildPages(
                            *cmd, *pageBuffer,
                            request ? std::span(&build, 1) : std::span<const MeshletStreamClasPageBuild>{}, log)),
                        log);
                BufferBarrierDesc tableBarrier{.buffer = pool.pageTableBuffer(), .before = ResourceState::General,
                    .after = ResourceState::TransferSource};
                cmd->barrier({.buffers = &tableBarrier, .bufferCount = 1});
                cmd->copyBuffer({.source = pool.pageTableBuffer(), .destination = publicationReadback.get(),
                    .sourceOffset = uint64_t(pageIndex) * 4u, .size = 4});
                std::swap(tableBarrier.before, tableBarrier.after);
                cmd->barrier({.buffers = &tableBarrier, .bufferCount = 1});
                require(bool(cmd->end()), "End failed");
                if (cancel) {
                    frame.cancel();
                    return;
                }
                CommandBuffer* list[] = {cmd.get()};
                require(bool(tracker.submit({.commandBuffers = list, .commandBufferCount = 1}, frame)) &&
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
            require((gpuPageEntry >> kMeshletStreamClasPageStateShift) == uint32_t(MeshletStreamClasPageState::Active),
                    "GPU traversal cannot use CLAS in the MOVE submission");
            pool.beginFrame();
            require(pool.pageHasClas(pageIndex) && !pool.pageBuildPending(pageIndex), "Move did not publish");
            const auto stats = pool.stats();
            require(stats.encodedStorageBytes > 0 && stats.usedStorageBytes < stats.worstCaseStorageBytes &&
                        stats.usedStorageBytes >= stats.encodedStorageBytes,
                    "Pool did not allocate actual sizes");
            const uint64_t address = pool.clusterAddress(pageIndex, 0);
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
                        !pool.pageHasClas(pageIndex), "Retirement leaked storage or double-counted a stale entry");
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
            std::ofstream(context.outputDirectory / "CompactClasLifecycle.txt")
                << "worstCaseBytes=" << stats.worstCaseStorageBytes << " allocatedBytes=" << stats.usedStorageBytes
                << " encodedBytes=" << stats.encodedStorageBytes << '\n';
            return RhiTestResult::pass("Actual allocation, ordered GPU publication, revival, retirement, build/move "
                                       "cancellation and abandoned build");
        } catch (const std::exception& error) {
            return RhiTestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(CompactClasLifecycleTest);
class MiniZorahClasInFlightTest final : public RhiTest {
  public:
    MiniZorahClasInFlightTest()
    {
        type = RhiTestType::Rendering;
        name = "minizorah_clas_in_flight";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        if (!std::getenv("METALLIC_TEST_MINIZORAH")) { return RhiTestResult::skip("Set METALLIC_TEST_MINIZORAH=1"); }
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream trace(context.outputDirectory / "MiniZorahClasInFlight.jsonl");
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
        desc.enablePushDescriptor = true;
        desc.enableRayQuery = true;
        desc.enableClusterAccelerationStructure = true;
        desc.enableAftermath = true;
        std::unique_ptr<Device> device;
        auto result = createDevice(desc).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(result, Error::Unsupported)) { return RhiTestResult::skip("CLAS unavailable"); }
        if (!result) { return RhiTestResult::fail(toString(result)); }
        std::string log;
        RenderSampleLoadResult sample;
        if (!loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log)) { return RhiTestResult::fail(log); }
        auto& graph = sample.graph;
        if (std::getenv("METALLIC_TEST_CLAS_LEGACY")) { graph.findNode("GPUDriven")->properties["compactClas"] = false; }
        RenderView view;
        const auto original = graph.viewProperties().at("camera");
        if (!view.setCameraProperties(original)) { return RhiTestResult::fail("Camera failed"); }
        scene::Scene runtimeScene;
        HistoryResourceManager history;
        if (!(result = history.initialize(*device))) { return RhiTestResult::fail(toString(result)); }
        RenderGraphExecutor executor;
        executor.bindRenderView(&view);
        executor.bindRuntimeScene(&runtimeScene);
        if (std::getenv("METALLIC_TEST_CLAS_SCENE_SWITCH")) {
            RenderSampleLoadResult previous;
            if (!loadBuiltInRenderSample("realtime-lighting", previous, log) ||
                !setRenderSampleScenePath(previous, "Asset/Sponza/glTF/Sponza.gltf", log)) { return RhiTestResult::fail(log); }
            if (!runtimeScene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / previous.desc.scenePath)) {
                return RhiTestResult::fail("Sponza load failed");
            }
            SceneAccelerationStructureBuilder staticRtas;
            if (!(result = staticRtas.build(*device, *device->getQueue(QueueType::Compute), runtimeScene, log))) {
                return RhiTestResult::fail(log);
            }
            (void)view.setCameraProperties(previous.graph.viewProperties().at("camera"));
            if (!(result = executor.compile(*device, previous.graph, 1564, 708, log))) { return RhiTestResult::fail(log); }
            for (uint32_t f = 0; f < 60; ++f) {
                result = executor.execute({.graphicsQueue = device->getQueue(QueueType::Graphics),
                    .computeQueue = device->getQueue(QueueType::Compute), .historyResources = &history});
                if (!result) { return RhiTestResult::fail(toString(result)); }
            }
            staticRtas.clear();
            runtimeScene.clear();
            history.invalidateAll();
            (void)view.setCameraProperties(original);
        }
        result = executor.compile(*device, graph, 1564, 708, log);
        if (!result) { return RhiTestResult::fail(log); }
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
            if (!result) { return RhiTestResult::fail("Frame " + std::to_string(f) + ": " + toString(result)); }
            const auto& stats = executor.executionStats();
            const auto& stream = stats.streaming.at(0);
            trace << nlohmann::json{{"frame", f}, {"overlappingFrames", overlappingFrames}, {"clasBytes", stream.clasUsedBytes},
                {"clasBuilt", stream.clasBuiltClusters}, {"clasMoved", stream.clasMovedClusters}, {"clasPending", stream.clasPendingPages}}.dump() << std::endl;
        }
        result = executor.waitForSubmittedWork();
        if (!result) { return RhiTestResult::fail(toString(result)); }
        if (overlappingFrames < 100) { return RhiTestResult::fail("Did not exercise frames in flight"); }
        return RhiTestResult::pass("1200 frames, overlap observed " + std::to_string(overlappingFrames));
    }
};
METALLIC_REGISTER_RHI_TEST(MiniZorahClasInFlightTest);
} // namespace
} // namespace metallic::tests
