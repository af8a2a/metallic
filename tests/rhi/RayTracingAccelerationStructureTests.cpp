#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "RHITest.h"

#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructureExtensions.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
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

class SceneAccelerationStructureBuildTest : public RHITest {
public:
    explicit SceneAccelerationStructureBuildTest(bool partitioned = false) : partitioned_(partitioned)
    {
        type = RHITestType::Resource;
        name = partitioned ? "scene_partitioned_acceleration_structure_build" : "scene_acceleration_structure_build";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic Scene Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enableRayTracingAccelerationStructure = true,
                .enablePartitionedAccelerationStructure = partitioned_,
                .enableAsyncCompute = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().rayTracingAccelerationStructure) {
            return RHITestResult::skip("ray tracing acceleration structure capability is unavailable");
        }

        if (partitioned_ && !device->capabilities().partitionedAccelerationStructure) {
            return RHITestResult::skip("partitioned acceleration structures unavailable");
        }
        const render::SceneAccelerationStructureBuildOptions options{
            .topLevelBackend = partitioned_ ? render::RayTracingTopLevelBackend::Partitioned : render::RayTracingTopLevelBackend::Standard};
        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RHITestResult::fail("scene acceleration structure test device has no graphics queue");
        }
        render::Queue* accelerationQueue = device->getQueue(render::QueueType::Compute);
        if (accelerationQueue == nullptr) {
            accelerationQueue = graphicsQueue;
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RHITestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::SceneAccelerationStructureBuilder builder;
        std::string log;
        if (!partitioned_) {
            const auto unsupported = builder.beginBuild(*device, *accelerationQueue, loadedScene, log,
                {.topLevelBackend = render::RayTracingTopLevelBackend::Partitioned});
            if (!render::hasError(unsupported, render::Error::Unsupported) || builder.valid() ||
                builder.buildState() != render::SceneAccelerationStructureBuildState::Idle) {
                return RHITestResult::fail("disabled PTLAS must fail before starting BLAS/OMM work");
            }
        }
        result = builder.beginBuild(*device, *accelerationQueue, loadedScene, log, options);
        if (!result) {
            return RHITestResult::fail(
                std::string("SceneAccelerationStructureBuilder::beginBuild returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (builder.buildState() != render::SceneAccelerationStructureBuildState::Building ||
            builder.valid()) {
            return RHITestResult::fail(
                "SceneAccelerationStructureBuilder did not enter its asynchronous build state");
        }

        const auto clearDeadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        bool clearProbeComplete = false;
        while (!clearProbeComplete && builder.stats().compactedBlasBytes == 0 &&
               std::chrono::steady_clock::now() < clearDeadline) {
            result = builder.pollBuild(log).transform([&](auto value) { clearProbeComplete = std::move(value); });
            if (!result) {
                return RHITestResult::fail(
                    std::string("SceneAccelerationStructureBuilder clear probe returned ") +
                    toString(result) +
                    ": " +
                    log);
            }
            if (!clearProbeComplete && builder.stats().compactedBlasBytes == 0) {
                std::this_thread::yield();
            }
        }
        if (clearProbeComplete || builder.stats().compactedBlasBytes == 0 ||
            builder.buildState() != render::SceneAccelerationStructureBuildState::Building) {
            return RHITestResult::fail(
                "SceneAccelerationStructureBuilder did not expose an in-flight compaction phase");
        }

        const render::SceneAccelerationStructureStats inFlightStats = builder.stats();
        if (inFlightStats.originalBlasBytes == 0 ||
            inFlightStats.compactedBlasBytes == 0 ||
            inFlightStats.compactedBlasBytes > inFlightStats.originalBlasBytes ||
            inFlightStats.compactionSavedBytes !=
                inFlightStats.originalBlasBytes - inFlightStats.compactedBlasBytes ||
            inFlightStats.accelerationStructureBytes <= inFlightStats.compactedBlasBytes ||
            inFlightStats.peakAccelerationStructureBytes <
                inFlightStats.accelerationStructureBytes +
                    inFlightStats.compactionSavedBytes) {
            return RHITestResult::fail(
                "SceneAccelerationStructureBuilder exposed inconsistent in-flight compaction statistics");
        }

        builder.clear();
        if (builder.buildState() != render::SceneAccelerationStructureBuildState::Idle ||
            builder.valid() || builder.accelerationStructure() != nullptr ||
            builder.stats().blasCount != 0 ||
            builder.stats().accelerationStructureBytes != 0 ||
            builder.stats().originalBlasBytes != 0 ||
            builder.stats().compactedBlasBytes != 0 ||
            builder.stats().compactionSavedBytes != 0 ||
            builder.stats().peakAccelerationStructureBytes != 0) {
            return RHITestResult::fail(
                "SceneAccelerationStructureBuilder clear did not retire an in-flight build");
        }

        result = builder.beginBuild(*device, *accelerationQueue, loadedScene, log, options);
        if (!result) {
            return RHITestResult::fail(
                std::string("SceneAccelerationStructureBuilder::beginBuild after clear returned ") +
                toString(result) +
                ": " +
                log);
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        bool buildComplete = false;
        bool observedIncompletePoll = false;
        while (!buildComplete && std::chrono::steady_clock::now() < deadline) {
            result = builder.pollBuild(log).transform([&](auto value) { buildComplete = std::move(value); });
            if (!result) {
                return RHITestResult::fail(
                    std::string("SceneAccelerationStructureBuilder::pollBuild returned ") +
                    toString(result) +
                    ": " +
                    log);
            }
            if (!buildComplete) {
                observedIncompletePoll = true;
                if (builder.buildState() !=
                    render::SceneAccelerationStructureBuildState::Building) {
                    return RHITestResult::fail(
                        "SceneAccelerationStructureBuilder left Building before compaction completed");
                }
                std::this_thread::yield();
            }
        }
        if (!buildComplete || !observedIncompletePoll ||
            builder.buildState() != render::SceneAccelerationStructureBuildState::Ready ||
            !builder.valid()) {
            return RHITestResult::fail(
                "SceneAccelerationStructureBuilder asynchronous multi-stage build did not produce a valid TLAS");
        }

        if (builder.accelerationStructure()->desc().topLevelBackend != options.topLevelBackend ||
            (partitioned_ && (builder.stats().partitionCount == 0 || builder.stats().operationBytes == 0))) {
            return RHITestResult::fail("incorrect scene top-level strategy");
        }
        const render::SceneAccelerationStructureStats& stats = builder.stats();
        if (stats.blasCount == 0 || stats.instanceCount == 0 || stats.triangleCount == 0 ||
            stats.originalBlasBytes == 0 || stats.compactedBlasBytes == 0 ||
            stats.compactedBlasBytes > stats.originalBlasBytes ||
            stats.compactionSavedBytes !=
                stats.originalBlasBytes - stats.compactedBlasBytes ||
            stats.accelerationStructureBytes <= stats.compactedBlasBytes ||
            stats.peakAccelerationStructureBytes <
                stats.accelerationStructureBytes + stats.compactionSavedBytes) {
            return RHITestResult::fail(
                "SceneAccelerationStructureBuilder produced inconsistent compaction statistics");
        }

        const render::SceneAccelerationStructureStats statsBeforeUpdate = stats;
        const int32_t movedNodeIndex = loadedScene.renderNodes().front().nodeIndex;
        if (movedNodeIndex < 0 || static_cast<size_t>(movedNodeIndex) >= loadedScene.nodes().size()) {
            return RHITestResult::fail("SceneAccelerationStructureBuilder test scene has no editable instance owner");
        }
        float4x4 movedLocal = loadedScene.nodes()[static_cast<size_t>(movedNodeIndex)].localMatrix;
        movedLocal.a03 += 2.0f;
        if (!loadedScene.setNodeLocalMatrix(movedNodeIndex, movedLocal)) {
            return RHITestResult::fail("failed to move the RTAS test instance");
        }
        result = builder.updateInstanceTransforms(*device, *accelerationQueue, loadedScene, log);
        if (!result) {
            return RHITestResult::fail(
                std::string("SceneAccelerationStructureBuilder::updateInstanceTransforms returned ") +
                toString(result) + ": " + log);
        }
        const render::SceneAccelerationStructureStats& statsAfterUpdate = builder.stats();
        if (statsAfterUpdate.blasCount != statsBeforeUpdate.blasCount ||
            statsAfterUpdate.instanceCount != statsBeforeUpdate.instanceCount ||
            statsAfterUpdate.triangleCount != statsBeforeUpdate.triangleCount ||
            statsAfterUpdate.originalBlasBytes != statsBeforeUpdate.originalBlasBytes ||
            statsAfterUpdate.compactedBlasBytes != statsBeforeUpdate.compactedBlasBytes ||
            statsAfterUpdate.compactionSavedBytes != statsBeforeUpdate.compactionSavedBytes ||
            statsAfterUpdate.accelerationStructureBytes !=
                statsBeforeUpdate.accelerationStructureBytes ||
            statsAfterUpdate.peakAccelerationStructureBytes !=
                statsBeforeUpdate.peakAccelerationStructureBytes) {
            return RHITestResult::fail(
                "TLAS refit changed BLAS topology or compaction statistics");
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
            return RHITestResult::fail("failed to hide every RTAS test instance");
        }

        result = builder.build(*device, *accelerationQueue, loadedScene, log, options);
        if (!result || !builder.valid()) {
            return RHITestResult::fail(
                std::string("SceneAccelerationStructureBuilder empty-scene build returned ") +
                toString(result) + ": " + log);
        }
        const render::SceneAccelerationStructureStats emptyStats = builder.stats();
        if (emptyStats.blasCount == 0 ||
            emptyStats.instanceCount != 0 ||
            emptyStats.triangleCount == 0) {
            return RHITestResult::fail("empty TLAS did not preserve geometry with zero visible instances");
        }

        float4x4 hiddenMovedLocal =
            loadedScene.nodes()[static_cast<size_t>(movedNodeIndex)].localMatrix;
        hiddenMovedLocal.a13 += 1.0f;
        if (!loadedScene.setNodeLocalMatrix(movedNodeIndex, hiddenMovedLocal)) {
            return RHITestResult::fail("failed to move a hidden RTAS test instance");
        }
        result = builder.updateInstanceTransforms(*device, *accelerationQueue, loadedScene, log);
        if (!result || builder.stats().instanceCount != 0) {
            return RHITestResult::fail(
                std::string("empty TLAS transform sync returned ") +
                toString(result) + ": " + log);
        }

        builder.clear();
        if (builder.buildState() != render::SceneAccelerationStructureBuildState::Idle ||
            builder.valid() || builder.accelerationStructure() != nullptr ||
            builder.stats().blasCount != 0 ||
            builder.stats().accelerationStructureBytes != 0 ||
            builder.stats().originalBlasBytes != 0 ||
            builder.stats().compactedBlasBytes != 0 ||
            builder.stats().compactionSavedBytes != 0 ||
            builder.stats().peakAccelerationStructureBytes != 0) {
            return RHITestResult::fail(
                "SceneAccelerationStructureBuilder clear did not reset a completed compact build");
        }

        {
            render::ScenePathTraceResources resources;
            const render::RenderGraphProperties properties{
                {"path", scenePath.string()},
                {"topLevelBackend", partitioned_ ? "partitioned" : "standard"},
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
                result = resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { resourcesComplete = std::move(value); });
                if (result && !resourcesComplete) {
                    std::this_thread::yield();
                }
            }
            const render::SceneAccelerationStructureStats& resourceStats =
                resources.accelerationStructure().stats();
            if (!result || !resourcesComplete || !resources.valid() ||
                resourceStats.instanceCount != 0 ||
                resourceStats.originalBlasBytes == 0 ||
                resourceStats.compactedBlasBytes == 0 ||
                resourceStats.compactedBlasBytes > resourceStats.originalBlasBytes ||
                resourceStats.compactionSavedBytes !=
                    resourceStats.originalBlasBytes - resourceStats.compactedBlasBytes ||
                resources.instanceBuffer() == nullptr ||
                resources.instanceBuffer()->desc().size == 0) {
                return RHITestResult::fail(
                    std::string("ScenePathTraceResources empty-scene preparation failed: ") + log);
            }

            const uint64_t resourcesRevision = resources.revision();
            hiddenMovedLocal.a23 += 1.0f;
            if (!loadedScene.setNodeLocalMatrix(movedNodeIndex, hiddenMovedLocal)) {
                return RHITestResult::fail("failed to move a hidden path-trace instance");
            }
            result = resources.syncRuntimeScene(&loadedScene, log);
            if (!result || !resources.valid() ||
                resources.revision() <= resourcesRevision ||
                resources.accelerationStructure().stats().instanceCount != 0) {
                return RHITestResult::fail(
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
                return RHITestResult::fail("failed to restore path-trace instance visibility");
            }

            const uint64_t topologyRevision = resources.revision();
            result = resources.syncRuntimeScene(&loadedScene, log);
            const render::SceneAccelerationStructureStats& rebuiltResourceStats =
                resources.accelerationStructure().stats();
            if (!result || !resources.valid() ||
                resources.revision() <= topologyRevision ||
                rebuiltResourceStats.instanceCount == 0 ||
                rebuiltResourceStats.originalBlasBytes == 0 ||
                rebuiltResourceStats.compactedBlasBytes == 0 ||
                rebuiltResourceStats.compactedBlasBytes >
                    rebuiltResourceStats.originalBlasBytes ||
                rebuiltResourceStats.compactionSavedBytes !=
                    rebuiltResourceStats.originalBlasBytes -
                        rebuiltResourceStats.compactedBlasBytes) {
                return RHITestResult::fail(
                    std::string("ScenePathTraceResources topology rebuild failed: ") + log);
            }

            resources.clear();
        }

        result = device->waitIdle();
        if (!result) {
            return RHITestResult::fail("failed to wait for empty-scene resource retirement");
        }

        return RHITestResult::pass(log);
    }
private:
    bool partitioned_ = false;
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
        auto* const instances = resources.instanceBuffer();
        auto* const vertices = resources.shadingVertexBuffer();
        auto* const tlas = resources.accelerationStructure().accelerationStructure();
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
                    resources.instanceBuffer() != instances || resources.shadingVertexBuffer() != vertices ||
                    resources.accelerationStructure().accelerationStructure() != tlas) {
                    return RHITestResult::fail("light/camera drag rebuilt geometry resources: " + log);
                }
                result = resources.accelerationStructure().updateInstanceTransforms(
                    *device, *accelerationQueue, scene, log);
                if (!result || !log.empty()) {
                    return RHITestResult::fail("light/camera drag submitted a redundant TLAS refit: " + log);
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
        if (!result || resources.revision() <= revision || resources.shadingVertexBuffer() != vertices ||
            std::abs(resources.bounds().min.x - oldMinX - 3.0f) > 1e-5f ||
            log.find("Updated scene acceleration-structure instance transforms") == std::string::npos) {
            return RHITestResult::fail("light parent failed to update its geometry child: " + log);
        }
        return RHITestResult::pass("32 light/camera drag steps reused geometry; geometry-parent move updated TLAS");
    }
};

METALLIC_REGISTER_RHI_TEST(SceneNonGeometryTransformSyncTest);

class SceneClusterAccelerationStructureBuildTest : public RHITest {
public:
    SceneClusterAccelerationStructureBuildTest()
    {
        type = RHITestType::Resource;
        name = "scene_cluster_acceleration_structure_build";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic Scene Cluster Acceleration Structure Test",
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
            return RHITestResult::fail("scene cluster RTAS test device has no graphics queue");
        }
        render::Queue* accelerationQueue = device->getQueue(render::QueueType::Compute);
        if (accelerationQueue == nullptr) {
            accelerationQueue = graphicsQueue;
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RHITestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::SceneClusterAccelerationStructureBuilder builder;
        std::string log;
        result = builder.build(*device, *accelerationQueue, loadedScene, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("SceneClusterAccelerationStructureBuilder::build returned ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RHITestResult::fail(
                std::string("SceneClusterAccelerationStructureBuilder::build returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (!builder.valid()) {
            return RHITestResult::fail("SceneClusterAccelerationStructureBuilder did not produce a valid TLAS");
        }

        const render::SceneClusterAccelerationStructureStats& stats = builder.stats();
        if (stats.clasCount == 0 ||
            stats.clusterBlasCount == 0 ||
            stats.instanceCount == 0 ||
            stats.clusterTriangleCount == 0 ||
            stats.accelerationStructureBytes == 0) {
            return RHITestResult::fail("SceneClusterAccelerationStructureBuilder produced empty cluster RTAS stats");
        }

        return RHITestResult::pass(log);
    }
};

class ScenePartitionedAccelerationStructureBuildTest final : public SceneAccelerationStructureBuildTest {
public:
    ScenePartitionedAccelerationStructureBuildTest() : SceneAccelerationStructureBuildTest(true) {}
};

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

METALLIC_REGISTER_RHI_TEST(SceneAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(RayTracingAccelerationStructureBarrierTest);
METALLIC_REGISTER_RHI_TEST(SceneClusterAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(ScenePartitionedAccelerationStructureBuildTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamCLASPoolBuildTest);

} // namespace
} // namespace metallic::tests
