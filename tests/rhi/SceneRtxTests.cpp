#include "RhiTest.h"

#include "Runtime/Render/GAPI/Vulkan/VulkanMeshletStreamClas.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::tests {
namespace {

class SceneRtxAccelerationStructureBuildTest : public RhiTest {
public:
    SceneRtxAccelerationStructureBuildTest()
    {
        type = RhiTestType::Resource;
        name = "scene_rtx_acceleration_structure_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Scene RTX Test",
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
            return RhiTestResult::fail("scene RTX test device has no graphics queue");
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RhiTestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::vulkan::SceneRtxBuilder builder;
        std::string log;
        result = builder.build(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SceneRtxBuilder::build returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (!builder.valid()) {
            return RhiTestResult::fail("SceneRtxBuilder did not produce a valid TLAS");
        }

        const render::vulkan::SceneRtxStats& stats = builder.stats();
        if (stats.blasCount == 0 || stats.instanceCount == 0 || stats.triangleCount == 0) {
            return RhiTestResult::fail("SceneRtxBuilder produced empty RTX stats");
        }

        const render::vulkan::SceneRtxStats statsBeforeUpdate = stats;
        const int32_t movedNodeIndex = loadedScene.renderNodes().front().nodeIndex;
        if (movedNodeIndex < 0 || static_cast<size_t>(movedNodeIndex) >= loadedScene.nodes().size()) {
            return RhiTestResult::fail("SceneRtxBuilder test scene has no editable instance owner");
        }
        float4x4 movedLocal = loadedScene.nodes()[static_cast<size_t>(movedNodeIndex)].localMatrix;
        movedLocal.a03 += 2.0f;
        if (!loadedScene.setNodeLocalMatrix(movedNodeIndex, movedLocal)) {
            return RhiTestResult::fail("failed to move the RTX test instance");
        }
        result = builder.updateInstanceTransforms(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SceneRtxBuilder::updateInstanceTransforms returned ") +
                toString(result) + ": " + log);
        }
        const render::vulkan::SceneRtxStats& statsAfterUpdate = builder.stats();
        if (statsAfterUpdate.blasCount != statsBeforeUpdate.blasCount ||
            statsAfterUpdate.instanceCount != statsBeforeUpdate.instanceCount ||
            statsAfterUpdate.triangleCount != statsBeforeUpdate.triangleCount) {
            return RhiTestResult::fail("TLAS refit changed BLAS, instance, or triangle counts");
        }

        return RhiTestResult::pass(log);
    }
};

class SceneClusterRtxAccelerationStructureBuildTest : public RhiTest {
public:
    SceneClusterRtxAccelerationStructureBuildTest()
    {
        type = RhiTestType::Resource;
        name = "scene_cluster_rtx_acceleration_structure_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Scene Cluster RTX Test",
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
            return RhiTestResult::fail("scene cluster RTX test device has no graphics queue");
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RhiTestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::vulkan::SceneClusterRtxBuilder builder;
        std::string log;
        result = builder.build(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("SceneClusterRtxBuilder::build returned ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RhiTestResult::fail(
                std::string("SceneClusterRtxBuilder::build returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (!builder.valid()) {
            return RhiTestResult::fail("SceneClusterRtxBuilder did not produce a valid TLAS");
        }

        const render::vulkan::SceneClusterRtxStats& stats = builder.stats();
        if (stats.clasCount == 0 ||
            stats.clusterBlasCount == 0 ||
            stats.instanceCount == 0 ||
            stats.clusterTriangleCount == 0 ||
            stats.accelerationStructureBytes == 0) {
            return RhiTestResult::fail("SceneClusterRtxBuilder produced empty cluster RTX stats");
        }

        return RhiTestResult::pass(log);
    }
};

class ScenePartitionedRtxAccelerationStructureBuildTest : public RhiTest {
public:
    ScenePartitionedRtxAccelerationStructureBuildTest()
    {
        type = RhiTestType::Resource;
        name = "scene_partitioned_rtx_acceleration_structure_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Scene Partitioned RTX Test",
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
            return RhiTestResult::fail("scene partitioned RTX test device has no graphics queue");
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RhiTestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::vulkan::ScenePartitionedRtxBuilder builder;
        std::string log;
        result = builder.build(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("ScenePartitionedRtxBuilder::build returned ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RhiTestResult::fail(
                std::string("ScenePartitionedRtxBuilder::build returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (!builder.valid()) {
            return RhiTestResult::fail("ScenePartitionedRtxBuilder did not produce a valid PTLAS");
        }

        const render::vulkan::ScenePartitionedRtxStats& stats = builder.stats();
        if (stats.blasCount == 0 ||
            stats.instanceCount == 0 ||
            stats.partitionCount == 0 ||
            stats.triangleCount == 0 ||
            stats.accelerationStructureBytes == 0 ||
            stats.operationBytes == 0) {
            return RhiTestResult::fail("ScenePartitionedRtxBuilder produced empty PTLAS stats");
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

        render::vulkan::MeshletStreamClasPool pool;
        result = pool.initialize(
            *device,
            render::vulkan::MeshletStreamClasPoolDesc{
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
        const render::vulkan::MeshletStreamClasPageBuild pageBuild{
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

        const render::vulkan::MeshletStreamClasPoolStats builtStats = pool.stats();
        if (!pool.pageHasClas(pageIndex) ||
            pool.pageClasAddressOffset(pageIndex) == UINT32_MAX ||
            pool.clusterAddress(pageIndex, 0) == 0 ||
            pool.clusterAddressBuffer() == nullptr ||
            pool.pageTableBuffer() == nullptr ||
            pool.pageTableBuffer()->desc().size !=
                static_cast<uint64_t>(asset.pageCount()) *
                    sizeof(render::vulkan::MeshletStreamClasPageEntry) ||
            builtStats.builtPageCount != 1 ||
            builtStats.trackedPageCount != 1 ||
            builtStats.builtClusterCount != asset.pages()[pageIndex].clusterCount ||
            builtStats.frameBuiltPageCount != 1 ||
            builtStats.usedStorageBytes == 0 ||
            builtStats.usedStorageBytes > builtStats.storageBytes) {
            return RhiTestResult::fail("stream CLAS pool did not retain the built fallback page");
        }
        render::vulkan::MeshletStreamClasPageEntry gpuPageEntry;
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
        if (render::vulkan::meshletStreamClasPageAddressOffset(gpuPageEntry) !=
                pool.pageClasAddressOffset(pageIndex) ||
            render::vulkan::meshletStreamClasPageState(gpuPageEntry) !=
                render::vulkan::MeshletStreamClasPageState::Active) {
            return RhiTestResult::fail("stream CLAS GPU page table did not expose the built page");
        }

        pool.retirePages(std::span(&pageIndex, 1));
        const render::vulkan::MeshletStreamClasPoolStats retiringStats = pool.stats();
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
        if (render::vulkan::meshletStreamClasPageState(gpuPageEntry) !=
            render::vulkan::MeshletStreamClasPageState::Retiring) {
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
        if (render::vulkan::meshletStreamClasPageAddressOffset(gpuPageEntry) !=
                render::vulkan::kInvalidMeshletStreamClasAddressOffset ||
            render::vulkan::meshletStreamClasPageState(gpuPageEntry) !=
                render::vulkan::MeshletStreamClasPageState::Empty) {
            return RhiTestResult::fail("stream CLAS GPU page table did not clear the released page");
        }

        return RhiTestResult::pass("Built and retired persistent stream CLAS page storage");
    }
};

METALLIC_REGISTER_RHI_TEST(SceneRtxAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(SceneClusterRtxAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(ScenePartitionedRtxAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamClasPoolBuildTest);

} // namespace
} // namespace metallic::tests
