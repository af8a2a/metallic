#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "RHITest.h"

#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace metallic::tests {
namespace {

class RayTracingAccelerationStructureBarrierTest : public RHITest {
public:
    RayTracingAccelerationStructureBarrierTest()
    {
        type = RHITestType::Resource;
        name = "ray_tracing_acceleration_structure_barriers";
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto deviceResult = render::createDevice(render::DeviceDesc{
            .applicationName = "Metallic AS Barrier Test",
            .enableValidation = context.enableValidation,
            .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true,
            .enableAsyncCompute = true,
        });
        if (!deviceResult) {
            return render::hasError(deviceResult, render::Error::Unsupported)
                ? RHITestResult::skip("ray tracing acceleration structures unavailable")
                : RHITestResult::fail(std::string("createDevice returned ") + render::resultToString(deviceResult));
        }
        auto device = std::move(*deviceResult);
        auto accelerationStructureResult = device->createRayTracingAccelerationStructure({
            .type = render::RayTracingAccelerationStructureType::TopLevel,
            .size = 4096,
        });
        if (!accelerationStructureResult) {
            return RHITestResult::fail("failed to create AS barrier resource");
        }
        auto accelerationStructure = std::move(*accelerationStructureResult);
        // The compute-queue recording/submission below validates actual queue support.
        if (!accelerationStructure->memoryInfo().allocationId) {
            return RHITestResult::fail("AS allocation metadata is invalid");
        }
        auto* queue = device->getQueue(render::QueueType::Compute);
        if (!queue) { queue = device->getQueue(render::QueueType::Graphics); }
        if (!queue) { return RHITestResult::fail("no AS build queue is available"); }
        auto poolResult = device->createCommandPool(*queue);
        if (!poolResult) { return RHITestResult::fail("failed to create AS barrier command pool"); }
        auto pool = std::move(*poolResult);
        auto commandsResult = pool->createCommandBuffer();
        if (!commandsResult) { return RHITestResult::fail("failed to create AS barrier commands"); }
        auto commands = std::move(*commandsResult);
        if (!commands->begin()) { return RHITestResult::fail("failed to begin AS barrier commands"); }
        const render::AccelerationStructureBarrierDesc invalidBarrier;
        if (!render::hasError(commands->synchronize({.accelerationStructures = std::span(&invalidBarrier, 1)}),
                render::Error::InvalidArgument) || commands->synchronizationStats().calls != 0) {
            return RHITestResult::fail("invalid AS barriers must be rejected before recording");
        }
        const render::AccelerationStructureBarrierDesc barriers[]{
            {.accelerationStructure = accelerationStructure.get(),
                .before = {render::PipelineStageBits::AccelerationStructureBuild, render::AccessBits::AccelerationStructureWrite},
                .after = {render::PipelineStageBits::ComputeShader, render::AccessBits::AccelerationStructureRead}},
            {.accelerationStructure = accelerationStructure.get(),
                .before = {render::PipelineStageBits::AccelerationStructureBuild, render::AccessBits::AccelerationStructureRead},
                .after = {render::PipelineStageBits::ComputeShader, render::AccessBits::AccelerationStructureRead}},
        };
        if (!commands->synchronize({.accelerationStructures = barriers})) {
            return RHITestResult::fail("compute AS build-to-read barriers were rejected");
        }
        const auto stats = commands->synchronizationStats();
        if (stats.calls != 1 || stats.memoryBarriers != 1 || stats.coalescedResources != 2 || stats.imageTransitions != 0) {
            return RHITestResult::fail("AS dependencies were not coalesced into a memory barrier");
        }
        const std::weak_ptr<void> allocation = accelerationStructure->retainAllocation();
        accelerationStructure.reset();
        if (allocation.expired()) { return RHITestResult::fail("AS barrier commands did not retain the allocation"); }
        if (!commands->end()) { return RHITestResult::fail("failed to end AS barrier commands"); }
        commands.reset();
        pool.reset();
        if (!allocation.expired()) { return RHITestResult::fail("cancelled AS barrier recording leaked the allocation"); }
        return RHITestResult::pass("Validated compute AS barriers, allocation metadata, coalescing and retention");
    }
};

class SceneNonGeometryTransformSyncTest final : public RHITest {
public:
    SceneNonGeometryTransformSyncTest()
    {
        type = RHITestType::Resource;
        name = "scene_path_trace_non_geometry_transform_sync";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({
            .applicationName = "Non-geometry transform sync test",
            .enableValidation = context.enableValidation,
            .enableRayTracingAccelerationStructure = true,
        }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            return render::hasError(result, render::Error::Unsupported)
                ? RHITestResult::skip("ray tracing acceleration structures unavailable")
                : RHITestResult::fail(std::string("device creation failed: ") + toString(result));
        }
        auto* queue = device->getQueue(render::QueueType::Graphics);
        auto* accelerationQueue = device->getQueue(render::QueueType::Compute);
        if (queue == nullptr) { return RHITestResult::fail("graphics queue unavailable"); }
        if (accelerationQueue == nullptr) { accelerationQueue = queue; }

        const auto directory = context.outputDirectory / "non-geometry-transform";
        std::filesystem::create_directories(directory);
        const auto path = directory / "scene.gltf";
        {
            const float triangle[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
            std::ofstream binary(directory / "triangle.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(triangle), sizeof(triangle));
            std::ofstream gltf(path);
            gltf << R"json({
                "asset":{"version":"2.0"},"scene":0,
                "scenes":[{"nodes":[0,1,2]}],
                "nodes":[
                    {"name":"Light only","extensions":{"KHR_lights_punctual":{"light":0}}},
                    {"name":"Camera only","camera":0},
                    {"name":"Light with geometry child","children":[3],"extensions":{"KHR_lights_punctual":{"light":0}}},
                    {"name":"Geometry","mesh":0}
                ],
                "cameras":[{"type":"perspective","perspective":{"yfov":0.8,"znear":0.1}}],
                "extensionsUsed":["KHR_lights_punctual"],
                "extensions":{"KHR_lights_punctual":{"lights":[{"type":"point","intensity":10}]}},
                "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
                "buffers":[{"uri":"triangle.bin","byteLength":36}],
                "bufferViews":[{"buffer":0,"byteLength":36}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}]
            })json";
        }
        scene::Scene scene;
        if (!scene.load(path)) { return RHITestResult::fail(scene.lastLoadResult().error); }
        render::ScenePathTraceResources resources;
        std::string log;
        result = resources.beginPrepareAsync(*device, *queue, {{"path", path.string()}}, scene, log);
        bool complete = false;
        scene::SceneLoadProgress progress;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (result && !complete && std::chrono::steady_clock::now() < deadline) {
            result = resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { complete = std::move(value); });
            if (!complete) { std::this_thread::yield(); }
        }
        if (!result || !complete || !resources.valid()) {
            return RHITestResult::fail("resource preparation failed: " + log);
        }
        const uint64_t revision = resources.revision();
        auto* const materials = resources.materialBuffer();
        // Simulate consecutive drag updates through the same API as the gizmo.
        for (int step = 0; step < 16; ++step) {
            for (int node : {0, 1}) {
                auto moved = scene.nodes()[node].worldMatrix;
                moved.a03 += 0.1f;
                if (!scene.setObjectWorldMatrix(scene.objectForNode(node).entity(), moved)) {
                    return RHITestResult::fail("non-geometry node edit failed");
                }
                result = resources.syncRuntimeScene(&scene, log);
                if (!result || resources.revision() != revision ||
                    resources.materialBuffer() != materials) {
                    return RHITestResult::fail("light/camera drag rebuilt geometry resources: " + log);
                }
            }
        }
        const float oldMinX = resources.bounds().min.x;
        auto parent = scene.nodes()[2].localMatrix;
        parent.a03 += 3.0f;
        if (!scene.setNodeLocalMatrix(2, parent)) {
            return RHITestResult::fail("light parent edit failed");
        }
        result = resources.syncRuntimeScene(&scene, log);
        if (!result || resources.revision() <= revision || resources.materialBuffer() != materials ||
            std::abs(resources.bounds().min.x - oldMinX - 3.0f) > 1e-5f) {
            return RHITestResult::fail("light parent failed to update its geometry child: " + log);
        }
        return RHITestResult::pass("32 light/camera drag steps reused geometry; geometry-parent move updated streamed instance metadata");
    }
};

METALLIC_REGISTER_RHI_TEST(SceneNonGeometryTransformSyncTest);

class MeshletStreamCLASPoolBuildTest : public RHITest {
public:
    MeshletStreamCLASPoolBuildTest()
    {
        type = RHITestType::Resource;
        name = "meshlet_stream_clas_pool_build";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic Meshlet Stream CLAS Pool Test",
                .enableValidation = context.enableValidation,
                .enableClusterAccelerationStructure = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().clusterAccelerationStructure) {
            return RHITestResult::skip("cluster acceleration structure capability is unavailable");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RHITestResult::fail("stream CLAS pool test device has no graphics queue");
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
        if (!scene::decodeMeshletStreamPayloadForDevice(
                asset.pages()[pageIndex],
                asset.pagePayload(pageIndex),
                decodedStorage,
                decodedPayload,
                log)) {
            return RHITestResult::fail("streamasset fallback page decode failed: " + log);
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = device->createBuffer(render::BufferDesc{
                .size = decodedPayload.size(),
                .usage = render::BufferUsageBits::Storage |
                    render::BufferUsageBits::ShaderDeviceAddress |
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

        render::MeshletStreamCLASPool pool;
        result = pool.initialize(
            *device,
            render::MeshletStreamCLASPoolDesc{
                .asset = &asset,
                .maxStorageBytes = 64ull * 1024ull * 1024ull,
                .maxBuildClusters = asset.maxPageClusters(),
                .queuedFrameCount = 2,
            },
            log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip("MeshletStreamCLASPool::initialize returned Unsupported: " + log);
            }
            return RHITestResult::fail(
                std::string("MeshletStreamCLASPool::initialize returned ") + toString(result) + ": " + log);
        }
        if (pool.stats().trackedPageCount != 0) {
            return RHITestResult::fail("stream CLAS pool eagerly tracked empty scene pages");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }
        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
        }

        pool.beginFrame();
        result = commandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::MeshletStreamCLASPagePlan uploadPlan;
        if (!render::buildMeshletStreamClasPagePlan(asset.pages()[pageIndex], decodedPayload, pageIndex,
                pageIndex * asset.maxPageClusters(), uploadPlan, log)) {
            return RHITestResult::fail("Upload CLAS plan: " + log);
        }
        const render::MeshletStreamCLASPageBuild pageBuild{
            .pageIndex = pageIndex,
            .deviceOffsetBytes = 0,
            .plan = &uploadPlan,
        };
        result = pool.cmdBuildPages(*commandBuffer, *pageBuffer, std::span(&pageBuild, 1), log);
        if (!result) {
            return RHITestResult::fail(
                std::string("MeshletStreamCLASPool::cmdBuildPages returned ") + toString(result) + ": " + log);
        }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }
        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = {commandBuffers, 1},
            .signalFence = fence.get(),
        });
        if (!result) {
            return RHITestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }
        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        const render::MeshletStreamCLASPoolStats builtStats = pool.stats();
        if (!pool.pageHasClas(pageIndex) ||
            pool.pageClasAddressOffset(pageIndex) == UINT32_MAX ||
            pool.clusterAddress(pageIndex, 0) == 0 ||
            pool.clusterAddressBuffer() == nullptr ||
            pool.pageTableBuffer() == nullptr ||
            pool.pageTableBuffer()->desc().size !=
                static_cast<uint64_t>(asset.pageCount()) *
                    sizeof(render::MeshletStreamCLASPageEntry) ||
            builtStats.builtPageCount != 1 ||
            builtStats.trackedPageCount != 1 ||
            builtStats.builtClusterCount != asset.pages()[pageIndex].clusterCount ||
            builtStats.frameBuiltPageCount != 1 ||
            builtStats.usedStorageBytes == 0 ||
            builtStats.usedStorageBytes > builtStats.storageBytes) {
            return RHITestResult::fail("stream CLAS pool did not retain the built fallback page");
        }
        render::MeshletStreamCLASPageEntry gpuPageEntry;
        mapped = pool.pageTableBuffer()->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("stream CLAS page table did not map");
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
                render::MeshletStreamCLASPageState::Active) {
            return RHITestResult::fail("stream CLAS GPU page table did not expose the built page");
        }

        pool.retirePages(std::span(&pageIndex, 1));
        const render::MeshletStreamCLASPoolStats retiringStats = pool.stats();
        if (!pool.pageHasClas(pageIndex) ||
            retiringStats.builtPageCount != 0 ||
            retiringStats.trackedPageCount != 1 ||
            retiringStats.retiringPageCount != 1) {
            return RHITestResult::fail("stream CLAS pool did not defer retired page storage");
        }
        mapped = pool.pageTableBuffer()->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("stream CLAS page table did not map after retirement");
        }
        std::memcpy(
            &gpuPageEntry,
            static_cast<uint8_t*>(mapped) +
                static_cast<uint64_t>(pageIndex) * sizeof(gpuPageEntry),
            sizeof(gpuPageEntry));
        pool.pageTableBuffer()->unmap();
        if (render::meshletStreamClasPageState(gpuPageEntry) !=
            render::MeshletStreamCLASPageState::Retiring) {
            return RHITestResult::fail("stream CLAS GPU page table did not hide the retired page");
        }
        const auto retainedAddress = pool.clusterAddress(pageIndex, 0);
        if (!commandPool->reset() || !commandBuffer->begin()) { return RHITestResult::fail("Cannot reset reactivation commands"); }
        result = pool.cmdBuildPages(*commandBuffer, *pageBuffer, std::span(&pageBuild, 1), log);
        if (!commandBuffer->end()) { return RHITestResult::fail("Cannot end reactivation commands"); }
        if (!result || pool.stats().retiringPageCount != 0 || pool.stats().builtPageCount != 1 ||
            pool.stats().totalBuiltPageCount != 1 || pool.clusterAddress(pageIndex, 0) != retainedAddress) {
            return RHITestResult::fail("Retiring CLAS was rebuilt instead of reactivated");
        }
        pool.retirePages(std::span(&pageIndex, 1));
        pool.beginFrame();
        if (!pool.pageHasClas(pageIndex)) {
            return RHITestResult::fail("stream CLAS pool released a retired page before the queued-frame delay");
        }
        pool.beginFrame();
        if (pool.pageHasClas(pageIndex) ||
            pool.stats().retiringPageCount != 0 ||
            pool.stats().trackedPageCount != 0 ||
            pool.stats().usedStorageBytes != 0) {
            return RHITestResult::fail("stream CLAS pool did not release retired storage after the queued-frame delay");
        }
        mapped = pool.pageTableBuffer()->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("stream CLAS page table did not map after release");
        }
        std::memcpy(
            &gpuPageEntry,
            static_cast<uint8_t*>(mapped) +
                static_cast<uint64_t>(pageIndex) * sizeof(gpuPageEntry),
            sizeof(gpuPageEntry));
        pool.pageTableBuffer()->unmap();
        if (render::meshletStreamClasPageAddressOffset(gpuPageEntry) !=
                render::kInvalidMeshletStreamCLASAddressOffset ||
            render::meshletStreamClasPageState(gpuPageEntry) !=
                render::MeshletStreamCLASPageState::Empty) {
            return RHITestResult::fail("stream CLAS GPU page table did not clear the released page");
        }

        return RHITestResult::pass("Built and retired persistent stream CLAS page storage");
    }
};

METALLIC_REGISTER_RHI_TEST(RayTracingAccelerationStructureBarrierTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamCLASPoolBuildTest);

} // namespace
} // namespace metallic::tests
