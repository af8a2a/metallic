#include "RhiTest.h"

#include "Runtime/Render/MeshletStreamClas.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructureExtensions.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace metallic::tests {
namespace {

class SceneAccelerationStructureBuildTest : public RhiTest {
public:
    SceneAccelerationStructureBuildTest()
    {
        type = RhiTestType::Resource;
        name = "scene_acceleration_structure_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Scene Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enableRayTracingAccelerationStructure = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().rayTracingAccelerationStructure) {
            return RhiTestResult::skip("ray tracing acceleration structure capability is unavailable");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("scene acceleration structure test device has no graphics queue");
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RhiTestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::SceneAccelerationStructureBuilder builder;
        std::string log;
        result = builder.beginBuild(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SceneAccelerationStructureBuilder::beginBuild returned ") +
                toString(result) +
                ": " +
                log);
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!builder.pollBuild() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        if (!builder.valid()) {
            return RhiTestResult::fail("SceneAccelerationStructureBuilder asynchronous build did not produce a valid TLAS");
        }

        const render::SceneAccelerationStructureStats& stats = builder.stats();
        if (stats.blasCount == 0 || stats.instanceCount == 0 || stats.triangleCount == 0) {
            return RhiTestResult::fail("SceneAccelerationStructureBuilder produced empty RTAS stats");
        }

        const render::SceneAccelerationStructureStats statsBeforeUpdate = stats;
        const int32_t movedNodeIndex = loadedScene.renderNodes().front().nodeIndex;
        if (movedNodeIndex < 0 || static_cast<size_t>(movedNodeIndex) >= loadedScene.nodes().size()) {
            return RhiTestResult::fail("SceneAccelerationStructureBuilder test scene has no editable instance owner");
        }
        float4x4 movedLocal = loadedScene.nodes()[static_cast<size_t>(movedNodeIndex)].localMatrix;
        movedLocal.a03 += 2.0f;
        if (!loadedScene.setNodeLocalMatrix(movedNodeIndex, movedLocal)) {
            return RhiTestResult::fail("failed to move the RTAS test instance");
        }
        result = builder.updateInstanceTransforms(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SceneAccelerationStructureBuilder::updateInstanceTransforms returned ") +
                toString(result) + ": " + log);
        }
        const render::SceneAccelerationStructureStats& statsAfterUpdate = builder.stats();
        if (statsAfterUpdate.blasCount != statsBeforeUpdate.blasCount ||
            statsAfterUpdate.instanceCount != statsBeforeUpdate.instanceCount ||
            statsAfterUpdate.triangleCount != statsBeforeUpdate.triangleCount) {
            return RhiTestResult::fail("TLAS refit changed BLAS, instance, or triangle counts");
        }

        bool visibilityChanged = false;
        for (const scene::RenderNode& renderNode : loadedScene.renderNodes()) {
            if (renderNode.object != scene::kNullSceneEntity) {
                visibilityChanged =
                    loadedScene.setObjectVisible(renderNode.object, false) || visibilityChanged;
            }
        }
        if (!visibilityChanged ||
            std::any_of(
                loadedScene.renderNodes().begin(),
                loadedScene.renderNodes().end(),
                [](const scene::RenderNode& node) { return node.visible; })) {
            return RhiTestResult::fail("failed to hide every RTAS test instance");
        }

        result = builder.build(*device, *graphicsQueue, loadedScene, log);
        if (!result || !builder.valid()) {
            return RhiTestResult::fail(
                std::string("SceneAccelerationStructureBuilder empty-scene build returned ") +
                toString(result) + ": " + log);
        }
        const render::SceneAccelerationStructureStats emptyStats = builder.stats();
        if (emptyStats.blasCount == 0 ||
            emptyStats.instanceCount != 0 ||
            emptyStats.triangleCount == 0) {
            return RhiTestResult::fail("empty TLAS did not preserve geometry with zero visible instances");
        }

        float4x4 hiddenMovedLocal =
            loadedScene.nodes()[static_cast<size_t>(movedNodeIndex)].localMatrix;
        hiddenMovedLocal.a13 += 1.0f;
        if (!loadedScene.setNodeLocalMatrix(movedNodeIndex, hiddenMovedLocal)) {
            return RhiTestResult::fail("failed to move a hidden RTAS test instance");
        }
        result = builder.updateInstanceTransforms(*device, *graphicsQueue, loadedScene, log);
        if (!result || builder.stats().instanceCount != 0) {
            return RhiTestResult::fail(
                std::string("empty TLAS transform sync returned ") +
                toString(result) + ": " + log);
        }

        {
            render::ScenePathTraceResources resources;
            const render::RenderGraphProperties properties{
                {"path", scenePath.string()},
            };
            result = resources.beginPrepareAsync(
                *device,
                *graphicsQueue,
                properties,
                loadedScene,
                log);
            bool resourcesComplete = false;
            scene::SceneLoadProgress progress;
            const auto resourceDeadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (result && !resourcesComplete &&
                   std::chrono::steady_clock::now() < resourceDeadline) {
                result = resources.pumpPrepareAsync(
                    10.0,
                    resourcesComplete,
                    progress,
                    log);
                if (result && !resourcesComplete) {
                    std::this_thread::yield();
                }
            }
            if (!result || !resourcesComplete || !resources.valid() ||
                resources.accelerationStructure().stats().instanceCount != 0 ||
                resources.instanceBuffer() == nullptr ||
                resources.instanceBuffer()->desc().size == 0) {
                return RhiTestResult::fail(
                    std::string("ScenePathTraceResources empty-scene preparation failed: ") + log);
            }

            const uint64_t resourcesRevision = resources.revision();
            hiddenMovedLocal.a23 += 1.0f;
            if (!loadedScene.setNodeLocalMatrix(movedNodeIndex, hiddenMovedLocal)) {
                return RhiTestResult::fail("failed to move a hidden path-trace instance");
            }
            result = resources.syncRuntimeScene(&loadedScene, log);
            if (!result || !resources.valid() ||
                resources.revision() <= resourcesRevision ||
                resources.accelerationStructure().stats().instanceCount != 0) {
                return RhiTestResult::fail(
                    std::string("ScenePathTraceResources empty-scene sync failed: ") + log);
            }

            std::vector<scene::SceneEntity> renderObjects;
            renderObjects.reserve(loadedScene.renderNodes().size());
            for (const scene::RenderNode& renderNode : loadedScene.renderNodes()) {
                if (renderNode.object != scene::kNullSceneEntity) {
                    renderObjects.push_back(renderNode.object);
                }
            }
            bool visibilityRestored = false;
            for (scene::SceneEntity object : renderObjects) {
                visibilityRestored =
                    loadedScene.setObjectVisible(object, true) || visibilityRestored;
            }
            if (!visibilityRestored ||
                std::none_of(
                    loadedScene.renderNodes().begin(),
                    loadedScene.renderNodes().end(),
                    [](const scene::RenderNode& node) { return node.visible; })) {
                return RhiTestResult::fail("failed to restore path-trace instance visibility");
            }

            const uint64_t topologyRevision = resources.revision();
            result = resources.syncRuntimeScene(&loadedScene, log);
            if (!result || !resources.valid() ||
                resources.revision() <= topologyRevision ||
                resources.accelerationStructure().stats().instanceCount == 0) {
                return RhiTestResult::fail(
                    std::string("ScenePathTraceResources topology rebuild failed: ") + log);
            }

            resources.clear();
        }

        result = device->waitIdle();
        if (!result) {
            return RhiTestResult::fail("failed to wait for empty-scene resource retirement");
        }

        return RhiTestResult::pass(log);
    }
};

class SceneClusterAccelerationStructureBuildTest : public RhiTest {
public:
    SceneClusterAccelerationStructureBuildTest()
    {
        type = RhiTestType::Resource;
        name = "scene_cluster_acceleration_structure_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Scene Cluster Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enableClusterAccelerationStructure = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().clusterAccelerationStructure) {
            return RhiTestResult::skip("cluster acceleration structure capability is unavailable");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("scene cluster RTAS test device has no graphics queue");
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RhiTestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::SceneClusterAccelerationStructureBuilder builder;
        std::string log;
        result = builder.build(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("SceneClusterAccelerationStructureBuilder::build returned ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RhiTestResult::fail(
                std::string("SceneClusterAccelerationStructureBuilder::build returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (!builder.valid()) {
            return RhiTestResult::fail("SceneClusterAccelerationStructureBuilder did not produce a valid TLAS");
        }

        const render::SceneClusterAccelerationStructureStats& stats = builder.stats();
        if (stats.clasCount == 0 ||
            stats.clusterBlasCount == 0 ||
            stats.instanceCount == 0 ||
            stats.clusterTriangleCount == 0 ||
            stats.accelerationStructureBytes == 0) {
            return RhiTestResult::fail("SceneClusterAccelerationStructureBuilder produced empty cluster RTAS stats");
        }

        return RhiTestResult::pass(log);
    }
};

class ScenePartitionedAccelerationStructureBuildTest : public RhiTest {
public:
    ScenePartitionedAccelerationStructureBuildTest()
    {
        type = RhiTestType::Resource;
        name = "scene_partitioned_acceleration_structure_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Scene Partitioned Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enablePartitionedAccelerationStructure = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().partitionedAccelerationStructure) {
            return RhiTestResult::skip("partitioned acceleration structure capability is unavailable");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail(
                "scene partitioned acceleration structure test device has no graphics queue");
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RhiTestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::ScenePartitionedAccelerationStructureBuilder builder;
        std::string log;
        result = builder.build(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("ScenePartitionedAccelerationStructureBuilder::build returned ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RhiTestResult::fail(
                std::string("ScenePartitionedAccelerationStructureBuilder::build returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (!builder.valid()) {
            return RhiTestResult::fail("ScenePartitionedAccelerationStructureBuilder did not produce a valid PTLAS");
        }

        const render::ScenePartitionedAccelerationStructureStats& stats = builder.stats();
        if (stats.blasCount == 0 ||
            stats.instanceCount == 0 ||
            stats.partitionCount == 0 ||
            stats.triangleCount == 0 ||
            stats.accelerationStructureBytes == 0 ||
            stats.operationBytes == 0) {
            return RhiTestResult::fail("ScenePartitionedAccelerationStructureBuilder produced empty PTLAS stats");
        }

        return RhiTestResult::pass(log);
    }
};

class MeshletStreamClasPoolBuildTest : public RhiTest {
public:
    MeshletStreamClasPoolBuildTest()
    {
        type = RhiTestType::Resource;
        name = "meshlet_stream_clas_pool_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Meshlet Stream CLAS Pool Test",
                .enableValidation = context.enableValidation,
                .enableClusterAccelerationStructure = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().clusterAccelerationStructure) {
            return RhiTestResult::skip("cluster acceleration structure capability is unavailable");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("stream CLAS pool test device has no graphics queue");
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
        if (!scene::decodeMeshletStreamPayloadForDevice(
                asset.pages()[pageIndex],
                asset.pagePayload(pageIndex),
                decodedStorage,
                decodedPayload,
                log)) {
            return RhiTestResult::fail("streamasset fallback page decode failed: " + log);
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = device->createBuffer(
            render::BufferDesc{
                .size = decodedPayload.size(),
                .usage = render::BufferUsageBits::Storage |
                    render::BufferUsageBits::ShaderDeviceAddress |
                    render::BufferUsageBits::AccelerationStructureBuildInput,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            pageBuffer);
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

        render::MeshletStreamClasPool pool;
        result = pool.initialize(
            *device,
            render::MeshletStreamClasPoolDesc{
                .asset = &asset,
                .maxStorageBytes = 64ull * 1024ull * 1024ull,
                .maxBuildClusters = asset.maxPageClusters(),
                .queuedFrameCount = 2,
            },
            log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip("MeshletStreamClasPool::initialize returned Unsupported: " + log);
            }
            return RhiTestResult::fail(
                std::string("MeshletStreamClasPool::initialize returned ") + toString(result) + ": " + log);
        }
        if (pool.stats().trackedPageCount != 0) {
            return RhiTestResult::fail("stream CLAS pool eagerly tracked empty scene pages");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }
        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        pool.beginFrame();
        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        const render::MeshletStreamClasPageBuild pageBuild{
            .pageIndex = pageIndex,
            .deviceOffsetBytes = 0,
        };
        result = pool.cmdBuildPages(*commandBuffer, *pageBuffer, std::span(&pageBuild, 1), log);
        if (!result) {
            return RhiTestResult::fail(
                std::string("MeshletStreamClasPool::cmdBuildPages returned ") + toString(result) + ": " + log);
        }
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }
        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = fence.get(),
        });
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }
        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        const render::MeshletStreamClasPoolStats builtStats = pool.stats();
        if (!pool.pageHasClas(pageIndex) ||
            pool.pageClasAddressOffset(pageIndex) == UINT32_MAX ||
            pool.clusterAddress(pageIndex, 0) == 0 ||
            pool.clusterAddressBuffer() == nullptr ||
            pool.pageTableBuffer() == nullptr ||
            pool.pageTableBuffer()->desc().size !=
                static_cast<uint64_t>(asset.pageCount()) *
                    sizeof(render::MeshletStreamClasPageEntry) ||
            builtStats.builtPageCount != 1 ||
            builtStats.trackedPageCount != 1 ||
            builtStats.builtClusterCount != asset.pages()[pageIndex].clusterCount ||
            builtStats.frameBuiltPageCount != 1 ||
            builtStats.usedStorageBytes == 0 ||
            builtStats.usedStorageBytes > builtStats.storageBytes) {
            return RhiTestResult::fail("stream CLAS pool did not retain the built fallback page");
        }
        render::MeshletStreamClasPageEntry gpuPageEntry;
        mapped = pool.pageTableBuffer()->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("stream CLAS page table did not map");
        }
        std::memcpy(
            &gpuPageEntry,
            static_cast<uint8_t*>(mapped) +
                static_cast<uint64_t>(pageIndex) * sizeof(gpuPageEntry),
            sizeof(gpuPageEntry));
        pool.pageTableBuffer()->unmap();
        if (render::meshletStreamClasPageAddressOffset(gpuPageEntry) !=
                pool.pageClasAddressOffset(pageIndex) ||
            render::meshletStreamClasPageState(gpuPageEntry) !=
                render::MeshletStreamClasPageState::Active) {
            return RhiTestResult::fail("stream CLAS GPU page table did not expose the built page");
        }

        pool.retirePages(std::span(&pageIndex, 1));
        const render::MeshletStreamClasPoolStats retiringStats = pool.stats();
        if (!pool.pageHasClas(pageIndex) ||
            retiringStats.builtPageCount != 0 ||
            retiringStats.trackedPageCount != 1 ||
            retiringStats.retiringPageCount != 1) {
            return RhiTestResult::fail("stream CLAS pool did not defer retired page storage");
        }
        mapped = pool.pageTableBuffer()->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("stream CLAS page table did not map after retirement");
        }
        std::memcpy(
            &gpuPageEntry,
            static_cast<uint8_t*>(mapped) +
                static_cast<uint64_t>(pageIndex) * sizeof(gpuPageEntry),
            sizeof(gpuPageEntry));
        pool.pageTableBuffer()->unmap();
        if (render::meshletStreamClasPageState(gpuPageEntry) !=
            render::MeshletStreamClasPageState::Retiring) {
            return RhiTestResult::fail("stream CLAS GPU page table did not hide the retired page");
        }
        pool.beginFrame();
        if (!pool.pageHasClas(pageIndex)) {
            return RhiTestResult::fail("stream CLAS pool released a retired page before the queued-frame delay");
        }
        pool.beginFrame();
        if (pool.pageHasClas(pageIndex) ||
            pool.stats().retiringPageCount != 0 ||
            pool.stats().trackedPageCount != 0 ||
            pool.stats().usedStorageBytes != 0) {
            return RhiTestResult::fail("stream CLAS pool did not release retired storage after the queued-frame delay");
        }
        mapped = pool.pageTableBuffer()->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("stream CLAS page table did not map after release");
        }
        std::memcpy(
            &gpuPageEntry,
            static_cast<uint8_t*>(mapped) +
                static_cast<uint64_t>(pageIndex) * sizeof(gpuPageEntry),
            sizeof(gpuPageEntry));
        pool.pageTableBuffer()->unmap();
        if (render::meshletStreamClasPageAddressOffset(gpuPageEntry) !=
                render::kInvalidMeshletStreamClasAddressOffset ||
            render::meshletStreamClasPageState(gpuPageEntry) !=
                render::MeshletStreamClasPageState::Empty) {
            return RhiTestResult::fail("stream CLAS GPU page table did not clear the released page");
        }

        return RhiTestResult::pass("Built and retired persistent stream CLAS page storage");
    }
};

METALLIC_REGISTER_RHI_TEST(SceneAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(SceneClusterAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(ScenePartitionedAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamClasPoolBuildTest);

} // namespace
} // namespace metallic::tests
