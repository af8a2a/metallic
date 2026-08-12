#include "RhiTest.h"

#include "Runtime/Render/Subsystem/BuiltinRenderSubsystems.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace metallic::tests {
namespace {

scene::RenderPrimitive makeTrianglePrimitive(float firstVertexX = 0.0f)
{
    scene::RenderPrimitive primitive;
    primitive.name = "triangle";
    primitive.meshIndex = 7;
    primitive.primitiveIndex = 3;
    primitive.materialIndex = 0;
    primitive.mode = 4;
    primitive.vertexCount = 3;
    primitive.indexCount = 3;
    primitive.triangleCount = 1;
    primitive.positions = {
        float3(firstVertexX, 0.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
    };
    primitive.indices = {0, 1, 2};
    primitive.localBounds.min = float3(firstVertexX, 0.0f, 0.0f);
    primitive.localBounds.max = float3(1.0f, 1.0f, 0.0f);
    primitive.localBounds.valid = true;
    return primitive;
}

scene::RenderPrimitive makeCanonicalRasterPrimitive()
{
    scene::RenderPrimitive primitive = makeTrianglePrimitive();
    primitive.normals = {
        float3(0.0f, 0.0f, 1.0f),
        float3(0.0f, 0.0f, 1.0f),
        float3(0.0f, 0.0f, 1.0f),
    };
    primitive.tangents = {
        float4(1.0f, 0.0f, 0.0f, 1.0f),
        float4(1.0f, 0.0f, 0.0f, 1.0f),
        float4(1.0f, 0.0f, 0.0f, 1.0f),
    };
    primitive.texcoords0 = {
        float2(0.0f, 0.0f),
        float2(1.0f, 0.0f),
        float2(0.0f, 1.0f),
    };
    primitive.hasAuthoredNormals = true;
    primitive.hasAuthoredTangents = true;

    const auto cluster = [](uint32_t lodLevel,
                            int32_t lodGroupIndex,
                            uint32_t vertexOffset,
                            uint32_t triangleOffset) {
        scene::MeshletCluster result;
        result.vertexOffset = vertexOffset;
        result.vertexCount = 3;
        result.triangleOffset = triangleOffset;
        result.triangleCount = 1;
        result.lodLevel = lodLevel;
        result.lodGroupIndex = lodGroupIndex;
        result.lodError = static_cast<float>(lodLevel) * 0.25f;
        result.boundingSphereCenter = float3(0.5f, 0.5f, 0.0f);
        result.boundingSphereRadius = std::sqrt(0.5f);
        result.coneApex = float3(0.0f, 0.0f, 0.0f);
        result.coneAxis = float3(0.0f, 0.0f, 1.0f);
        result.coneCutoff = 0.75f;
        return result;
    };

    primitive.meshletClusters = {cluster(0, 0, 0, 0)};
    primitive.meshletVertices = {0, 1, 2};
    primitive.meshletTriangles = {0, 1, 2};
    primitive.meshletLodLevels = {
        scene::MeshletLodLevel{
            .groupOffset = 0,
            .groupCount = 1,
            .clusterOffset = 0,
            .clusterCount = 1,
        },
        scene::MeshletLodLevel{
            .groupOffset = 1,
            .groupCount = 1,
            .clusterOffset = 1,
            .clusterCount = 1,
        },
    };
    primitive.meshletLodGroups.resize(2);
    primitive.meshletLodGroups[0].clusterOffset = 0;
    primitive.meshletLodGroups[0].clusterCount = 1;
    primitive.meshletLodGroups[0].lodLevel = 0;
    primitive.meshletLodGroups[1].clusterOffset = 1;
    primitive.meshletLodGroups[1].clusterCount = 1;
    primitive.meshletLodGroups[1].lodLevel = 1;
    primitive.meshletLodClusters = {
        cluster(0, 0, 0, 0),
        cluster(1, 1, 3, 3),
    };
    primitive.meshletLodVertices = {0, 1, 2, 0, 1, 2};
    primitive.meshletLodTriangles = {0, 1, 2, 0, 1, 2};
    return primitive;
}

float4x4 translationMatrix(const float3& translation)
{
    float4x4 matrix;
    matrix.SetupByTranslation(translation);
    return matrix;
}

bool sameMatrix(const float4x4& lhs, const float4x4& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(float4x4)) == 0;
}

render::GPUSceneSourceView makeSourceView(
    const std::vector<scene::RenderPrimitive>& primitives,
    const std::vector<scene::RenderNode>& nodes,
    const std::vector<scene::RenderMaterial>& materials,
    uint64_t transformRevision,
    uint64_t visibilityRevision,
    uint64_t contentRevision)
{
    return render::GPUSceneSourceView{
        .renderPrimitives = primitives,
        .renderNodes = nodes,
        .materials = materials,
        .lifetimeRevision = 1,
        .structuralRevision = 1,
        .contentRevision = contentRevision,
        .transformRevision = transformRevision,
        .visibilityRevision = visibilityRevision,
        .externalRevision = contentRevision,
    };
}

class GPUSceneCpuCoreTest final : public RhiTest {
public:
    GPUSceneCpuCoreTest()
    {
        type = RhiTestType::Resource;
        name = "gpu_scene_cpu_core";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::vector<scene::RenderPrimitive> primitives{
            makeTrianglePrimitive(),
            makeTrianglePrimitive(),
            makeTrianglePrimitive(0.25f),
        };

        std::vector<scene::RenderMaterial> materials(5);
        materials[0].name = "opaque single";
        materials[1].name = "opaque double";
        materials[1].doubleSided = true;
        materials[2].name = "masked single";
        materials[2].alphaMode = "MASK";
        materials[3].name = "masked double";
        materials[3].alphaMode = "MASK";
        materials[3].doubleSided = true;
        materials[4].name = "blend";
        materials[4].alphaMode = "BLEND";
        materials[4].doubleSided = true;

        const scene::SceneEntity objectA = static_cast<scene::SceneEntity>(1);
        const scene::SceneEntity objectB = static_cast<scene::SceneEntity>(2);
        std::vector<scene::RenderNode> nodes(6);
        for (uint32_t index = 0; index < 5; ++index) {
            nodes[index].object = index < 2 ? objectA : objectB;
            nodes[index].nodeIndex = static_cast<int32_t>(index);
            nodes[index].renderPrimitiveIndex = static_cast<int32_t>(index < 3 ? index : 0);
            nodes[index].materialIndex = static_cast<int32_t>(index);
            nodes[index].transformRevision = 1;
        }
        nodes[4].visible = false;
        nodes[5].renderPrimitiveIndex = scene::kInvalidSceneIndex;

        render::GPUScene gpuScene;
        gpuScene.setDefaultFrameSlotCount(2);
        std::string log;
        render::Result result = gpuScene.rebuild(
            makeSourceView(primitives, nodes, materials, 1, 1, 1),
            log);
        if (!result) {
            return RhiTestResult::fail("GPUScene rebuild failed: " + log);
        }

        const render::GPUSceneStats& stats = gpuScene.stats();
        if (stats.geometryCount != 2 || stats.materialCount != 5 ||
            stats.instanceCount != 5 || stats.deduplicatedGeometryCount != 1 ||
            stats.geometryPayloadConflictCount != 1 || stats.skippedRenderNodeCount != 1) {
            return RhiTestResult::fail("GPUScene build/deduplication stats are incorrect");
        }
        for (uint32_t count : stats.bucketInstanceCounts) {
            if (count != 1) {
                return RhiTestResult::fail("GPUScene did not classify one instance into every draw bucket");
            }
        }

        const render::GPUSceneGeometryId geometry0 = gpuScene.geometryForRenderPrimitive(0);
        const render::GPUSceneGeometryId geometry1 = gpuScene.geometryForRenderPrimitive(1);
        const render::GPUSceneGeometryId geometry2 = gpuScene.geometryForRenderPrimitive(2);
        const render::GPUSceneMaterialId material0 = gpuScene.materialForSourceMaterial(0);
        const render::GPUSceneInstanceId instance0 = gpuScene.instanceForRenderNode(0);
        if (!geometry0 || geometry0 != geometry1 || geometry0 == geometry2 ||
            !material0 || !instance0 || gpuScene.instanceForRenderNode(5) ||
            gpuScene.instancesForObject(objectA).size() != 2) {
            return RhiTestResult::fail("GPUScene source mappings are incorrect");
        }
        const uint32_t initialDrawSetGeneration = gpuScene.drawSet().generation;
        const uint64_t initialDrawSetRevision = gpuScene.drawSet().revision;
        const render::GPUSceneInstanceRecord* initialInstance = gpuScene.instance(instance0);
        if (initialDrawSetGeneration == 0 || initialDrawSetRevision == 0 ||
            stats.drawSetGeneration != initialDrawSetGeneration ||
            initialInstance == nullptr ||
            !sameMatrix(initialInstance->worldMatrix, initialInstance->previousWorldMatrix) ||
            std::abs(initialInstance->localBoundingSphere.x - 0.5f) > 0.00001f ||
            std::abs(initialInstance->localBoundingSphere.y - 0.5f) > 0.00001f ||
            std::abs(initialInstance->localBoundingSphere.z) > 0.00001f ||
            std::abs(initialInstance->localBoundingSphere.w - std::sqrt(0.5f)) > 0.00001f) {
            return RhiTestResult::fail("GPUScene generation, transform history, or local bounds sphere is incorrect");
        }

        const render::GPUSceneViewId view = gpuScene.createView();
        if (!view || !gpuScene.prepareView(view, 0, [](const render::GPUSceneInstanceRecord& instance) {
                return instance.sourceNodeIndex < 3;
            })) {
            return RhiTestResult::fail("GPUScene failed to create or prepare a View");
        }
        const render::GPUSceneVisibleDrawSet* visible = gpuScene.visibleDrawSet(view, 0);
        if (visible == nullptr || visible->instances.size() != 3 ||
            visible->stats.sourceInstanceCount != 5 || visible->stats.prepareCount != 1) {
            return RhiTestResult::fail("GPUScene View predicate or VisibleDrawSet stats are incorrect");
        }
        if (!gpuScene.prepareView(view, 1) ||
            gpuScene.visibleDrawSet(view, 1)->instances.size() != 4) {
            return RhiTestResult::fail("GPUScene visibility filtering is incorrect");
        }

        const render::GPUSceneViewId secondView = gpuScene.createView();
        const render::GPUSceneViewPrepareInfo initialViewInfo{
            .width = 640,
            .height = 480,
        };
        if (!secondView ||
            !gpuScene.prepareView(view, 0, initialViewInfo) ||
            !gpuScene.prepareView(secondView, 0, initialViewInfo)) {
            return RhiTestResult::fail("GPUScene failed to prepare two isolated View histories");
        }
        const uint64_t initialHzbEpoch =
            gpuScene.visibleDrawSet(view, 0)->stats.hzbHistoryEpoch;
        if (!gpuScene.markViewHzbValid(view, 0) ||
            !gpuScene.prepareView(view, 0, initialViewInfo) ||
            !gpuScene.visibleDrawSet(view, 0)->stats.hzbValid ||
            gpuScene.visibleDrawSet(secondView, 0)->stats.hzbValid) {
            return RhiTestResult::fail("GPUScene View HZB validity leaked between Views");
        }
        const render::GPUSceneViewPrepareInfo resizedViewInfo{
            .width = 800,
            .height = 600,
        };
        if (!gpuScene.prepareView(view, 0, resizedViewInfo) ||
            gpuScene.visibleDrawSet(view, 0)->stats.hzbValid ||
            gpuScene.visibleDrawSet(view, 0)->stats.hzbHistoryEpoch <= initialHzbEpoch) {
            return RhiTestResult::fail("GPUScene resize did not invalidate the View HZB history");
        }
        if (!gpuScene.markViewHzbValid(view, 0) ||
            !gpuScene.prepareView(
                view,
                0,
                render::GPUSceneViewPrepareInfo{
                    .width = 800,
                    .height = 600,
                    .freezeCullingCamera = true,
                }) ||
            gpuScene.visibleDrawSet(view, 0)->stats.hzbValid ||
            !gpuScene.markViewHzbValid(view, 0) ||
            !gpuScene.prepareView(
                view,
                0,
                render::GPUSceneViewPrepareInfo{
                    .width = 800,
                    .height = 600,
                    .cameraCut = true,
                    .freezeCullingCamera = true,
                }) ||
            gpuScene.visibleDrawSet(view, 0)->stats.hzbValid) {
            return RhiTestResult::fail(
                "GPUScene freeze toggle or camera cut did not invalidate HZB history");
        }
        if (!gpuScene.destroyView(secondView)) {
            return RhiTestResult::fail("GPUScene failed to destroy the second isolated View");
        }

        const float4x4 initialWorldMatrix = nodes[0].worldMatrix;
        const float4x4 updatedWorldMatrix = translationMatrix(float3(4.0f, 5.0f, 6.0f));
        nodes[0].worldMatrix = updatedWorldMatrix;
        nodes[0].transformRevision = 2;
        nodes[4].visible = true;
        const render::GPUSceneSyncResult syncResult = gpuScene.sync(
            makeSourceView(primitives, nodes, materials, 2, 2, 1));
        if (syncResult != render::GPUSceneSyncResult::Updated ||
            gpuScene.drawSet().generation != initialDrawSetGeneration ||
            gpuScene.drawSet().revision == initialDrawSetRevision ||
            gpuScene.instanceForRenderNode(0) != instance0 ||
            gpuScene.instance(instance0) == nullptr ||
            gpuScene.instance(instance0)->transformRevision != 2 ||
            !sameMatrix(gpuScene.instance(instance0)->worldMatrix, updatedWorldMatrix) ||
            !sameMatrix(gpuScene.instance(instance0)->previousWorldMatrix, initialWorldMatrix) ||
            gpuScene.visibleDrawSet(view, 0) != nullptr ||
            gpuScene.visibleDrawSet(view, 1) != nullptr) {
            return RhiTestResult::fail("GPUScene incremental transform/visibility sync is incorrect");
        }
        const uint64_t updatedDrawSetRevision = gpuScene.drawSet().revision;
        if (gpuScene.sync(makeSourceView(primitives, nodes, materials, 2, 2, 1)) !=
                render::GPUSceneSyncResult::HistoryUpdated ||
            gpuScene.drawSet().generation != initialDrawSetGeneration ||
            gpuScene.drawSet().revision == updatedDrawSetRevision ||
            !sameMatrix(
                gpuScene.instance(instance0)->previousWorldMatrix,
                updatedWorldMatrix)) {
            return RhiTestResult::fail("GPUScene previous transform did not converge on the next frame");
        }
        const uint64_t convergedDrawSetRevision = gpuScene.drawSet().revision;
        if (gpuScene.sync(makeSourceView(primitives, nodes, materials, 2, 2, 1)) !=
                render::GPUSceneSyncResult::Unchanged ||
            gpuScene.drawSet().revision != convergedDrawSetRevision) {
            return RhiTestResult::fail("GPUScene transform history did not settle after convergence");
        }
        if (!gpuScene.prepareView(view, 0) ||
            gpuScene.visibleDrawSet(view, 0)->instances.size() != 5) {
            return RhiTestResult::fail("GPUScene did not refresh VisibleDrawSet after incremental sync");
        }

        std::unique_ptr<render::Buffer> gpuViewBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = 256,
                .structureStride = sizeof(uint32_t),
                .usage = render::BufferUsageBits::Storage |
                    render::BufferUsageBits::Indirect,
            },
            gpuViewBuffer);
        if (!result || gpuViewBuffer == nullptr) {
            return RhiTestResult::fail("GPUScene buffer-view test buffer creation failed");
        }
        const uint32_t currentGeneration = gpuScene.drawSet().generation;
        const uint64_t currentRevision = gpuScene.drawSet().revision;
        auto makeVisibleGpuResources = [&]() {
            render::GPUSceneVisibleGpuResources resources;
            resources.instanceVisibilityStates = render::GPUSceneBufferView{
                .buffer = gpuViewBuffer.get(),
                .offset = 0,
                .size = 64,
                .structureStride = sizeof(uint32_t),
            };
            for (uint32_t phaseIndex = 0;
                 phaseIndex < render::kGPUSceneCullPhaseCount;
                 ++phaseIndex) {
                render::GPUSceneCullPhaseGpuView& phase = resources.phases[phaseIndex];
                phase.visibleMeshletIds = render::GPUSceneBufferView{
                    .buffer = gpuViewBuffer.get(),
                    .offset = 0,
                    .size = 64,
                    .structureStride = sizeof(uint32_t),
                };
                for (uint32_t bucket = 0;
                     bucket < render::kGPUSceneRasterDrawBucketCount;
                     ++bucket) {
                    const uint64_t bucketOffset = 64u +
                        static_cast<uint64_t>(phaseIndex) * 64u +
                        static_cast<uint64_t>(bucket) * 16u;
                    phase.buckets[bucket].indirectArguments = render::GPUSceneBufferView{
                        .buffer = gpuViewBuffer.get(),
                        .offset = bucketOffset,
                        .size = 12,
                        .structureStride = sizeof(uint32_t),
                    };
                    phase.buckets[bucket].overflow = render::GPUSceneBufferView{
                        .buffer = gpuViewBuffer.get(),
                        .offset = bucketOffset + 12u,
                        .size = sizeof(uint32_t),
                        .structureStride = sizeof(uint32_t),
                    };
                    phase.buckets[bucket].visibleMeshletCapacity = 16;
                }
            }
            resources.hzb.history[0] = render::GPUSceneBufferView{
                .buffer = gpuViewBuffer.get(),
                .offset = 192,
                .size = 32,
                .structureStride = sizeof(float),
            };
            resources.hzb.history[1] = render::GPUSceneBufferView{
                .buffer = gpuViewBuffer.get(),
                .offset = 224,
                .size = 32,
                .structureStride = sizeof(float),
            };
            resources.hzb.width = 4;
            resources.hzb.height = 4;
            resources.hzb.mipCount = 3;
            const render::GPUSceneVisibleDrawSet* prepared =
                gpuScene.visibleDrawSet(view, 0);
            resources.hzb.historyEpoch = prepared != nullptr
                ? prepared->stats.hzbHistoryEpoch
                : 0;
            resources.hzb.valid = prepared != nullptr && prepared->stats.hzbValid;
            return resources;
        };
        if (!gpuScene.setVisibleGpuResources(view, 0, makeVisibleGpuResources()) ||
            !gpuScene.visibleDrawSet(view, 0)->gpu.validFor(
                currentGeneration,
                currentRevision)) {
            return RhiTestResult::fail("GPUScene rejected valid visible GPU buffer views");
        }
        render::GPUSceneVisibleGpuResources staleVisible = makeVisibleGpuResources();
        staleVisible.sourceDrawSetGeneration = currentGeneration;
        staleVisible.sourceDrawSetRevision = currentRevision;
        staleVisible.instanceVisibilityStates.generation = currentGeneration + 1;
        staleVisible.instanceVisibilityStates.revision = currentRevision;
        if (gpuScene.setVisibleGpuResources(view, 0, std::move(staleVisible))) {
            return RhiTestResult::fail("GPUScene accepted a stale visible GPU buffer view");
        }
        render::GPUSceneVisibleGpuResources outOfRangeVisible = makeVisibleGpuResources();
        outOfRangeVisible.instanceVisibilityStates.offset = 252;
        outOfRangeVisible.instanceVisibilityStates.size = 8;
        if (gpuScene.setVisibleGpuResources(view, 0, std::move(outOfRangeVisible))) {
            return RhiTestResult::fail("GPUScene accepted an out-of-range visible GPU buffer view");
        }
        render::GPUSceneVisibleGpuResources invalidStrideVisible = makeVisibleGpuResources();
        invalidStrideVisible.instanceVisibilityStates.structureStride = 3;
        if (gpuScene.setVisibleGpuResources(view, 0, std::move(invalidStrideVisible))) {
            return RhiTestResult::fail("GPUScene accepted an invalid visible GPU buffer stride");
        }

        render::GPUSceneGlobalBufferViews globalViews;
        globalViews.geometries = render::GPUSceneBufferView{
            .buffer = gpuViewBuffer.get(),
            .offset = 0,
            .size = 16,
            .structureStride = sizeof(uint32_t),
        };
        globalViews.materials = render::GPUSceneBufferView{
            .buffer = gpuViewBuffer.get(),
            .offset = 16,
            .size = 16,
            .structureStride = sizeof(uint32_t),
        };
        globalViews.instances = render::GPUSceneBufferView{
            .buffer = gpuViewBuffer.get(),
            .offset = 32,
            .size = 16,
            .structureStride = sizeof(uint32_t),
        };
        globalViews.drawKeys = render::GPUSceneBufferView{
            .buffer = gpuViewBuffer.get(),
            .offset = 48,
            .size = 16,
            .structureStride = sizeof(uint32_t),
        };
        if (!gpuScene.setGlobalBufferViews(globalViews) ||
            !gpuScene.globalBufferViews().validFor(currentGeneration, currentRevision)) {
            return RhiTestResult::fail("GPUScene rejected valid global GPU buffer views");
        }
        render::GPUSceneGlobalBufferViews staleGlobal = gpuScene.globalBufferViews();
        staleGlobal.geometries.generation = currentGeneration + 1;
        if (gpuScene.setGlobalBufferViews(std::move(staleGlobal)) ||
            !gpuScene.globalBufferViews().validFor(currentGeneration, currentRevision)) {
            return RhiTestResult::fail("GPUScene accepted a stale global GPU buffer view");
        }
        render::GPUSceneGlobalBufferViews outOfRangeGlobal = gpuScene.globalBufferViews();
        outOfRangeGlobal.geometries.offset = 252;
        outOfRangeGlobal.geometries.size = 8;
        if (gpuScene.setGlobalBufferViews(std::move(outOfRangeGlobal)) ||
            !gpuScene.globalBufferViews().validFor(currentGeneration, currentRevision)) {
            return RhiTestResult::fail("GPUScene accepted an out-of-range global GPU buffer view");
        }

        materials[0].alphaMode = "MASK";
        if (gpuScene.sync(makeSourceView(primitives, nodes, materials, 2, 2, 2)) !=
            render::GPUSceneSyncResult::RebuildRequired) {
            return RhiTestResult::fail("GPUScene did not request rebuild after a material change");
        }
        result = gpuScene.rebuild(makeSourceView(primitives, nodes, materials, 2, 2, 2), log);
        if (!result || gpuScene.geometry(geometry0) != nullptr ||
            gpuScene.material(material0) != nullptr || gpuScene.instance(instance0) != nullptr ||
            gpuScene.drawSet().generation == initialDrawSetGeneration ||
            gpuScene.stats().drawSetGeneration != gpuScene.drawSet().generation) {
            return RhiTestResult::fail("GPUScene generational source IDs survived a full rebuild");
        }
        if (!gpuScene.prepareView(view, 0)) {
            return RhiTestResult::fail("GPUScene View did not survive a source rebuild");
        }

        if (!gpuScene.destroyView(view) || gpuScene.prepareView(view, 0)) {
            return RhiTestResult::fail("GPUScene accepted a destroyed View ID");
        }
        const render::GPUSceneViewId replacementView = gpuScene.createView();
        if (replacementView.index != view.index || replacementView.generation == view.generation) {
            return RhiTestResult::fail("GPUScene did not increment the reused View generation");
        }

        std::vector<scene::RenderPrimitive> invalidPrimitives{
            makeTrianglePrimitive(),
            makeTrianglePrimitive(),
        };
        invalidPrimitives[0].indexCount = 4;
        invalidPrimitives[1].indices[2] = 9;
        std::vector<scene::RenderNode> invalidNodes(2);
        for (uint32_t index = 0; index < invalidNodes.size(); ++index) {
            invalidNodes[index].object = objectA;
            invalidNodes[index].nodeIndex = static_cast<int32_t>(index);
            invalidNodes[index].renderPrimitiveIndex = static_cast<int32_t>(index);
            invalidNodes[index].materialIndex = 0;
        }
        std::vector<scene::RenderMaterial> invalidMaterials(1);
        render::GPUScene invalidGpuScene;
        std::string invalidLog;
        result = invalidGpuScene.rebuild(
            makeSourceView(
                invalidPrimitives,
                invalidNodes,
                invalidMaterials,
                1,
                1,
                1),
            invalidLog);
        const std::span<const render::GPUSceneInvalidPrimitiveDiagnostic> diagnostics =
            invalidGpuScene.invalidPrimitiveDiagnostics();
        if (!result || invalidGpuScene.stats().invalidPrimitiveCount != 2 ||
            invalidGpuScene.stats().invalidIndexCountPrimitiveCount != 1 ||
            invalidGpuScene.stats().outOfRangeIndexPrimitiveCount != 1 ||
            invalidGpuScene.stats().skippedRenderNodeCount != 2 ||
            invalidGpuScene.stats().instanceCount != 0 || diagnostics.size() != 2 ||
            diagnostics[0].reason !=
                render::GPUSceneInvalidPrimitiveReason::IndexCountNotMultipleOfThree ||
            diagnostics[1].reason != render::GPUSceneInvalidPrimitiveReason::IndexOutOfRange ||
            diagnostics[1].indexOffset != 2 || diagnostics[1].vertexIndex != 9 ||
            invalidLog.find("invalid triangle primitive") == std::string::npos) {
            return RhiTestResult::fail("GPUScene invalid triangle diagnostics are incorrect");
        }

        const uint32_t generationBeforeClear = gpuScene.drawSet().generation;
        gpuScene.clearSource();
        if (gpuScene.drawSet().generation == generationBeforeClear ||
            gpuScene.drawSet().generation == 0 ||
            gpuScene.stats().drawSetGeneration != gpuScene.drawSet().generation ||
            gpuScene.stats().instanceCount != 0) {
            return RhiTestResult::fail("GPUScene clearSource did not advance DrawSet generation");
        }

        render::RenderSubsystemHost host;
        if (!render::registerBuiltInRenderSubsystems(host, log) ||
            !host.isRegistered(render::GPUSceneSubsystem::kSubsystemId)) {
            return RhiTestResult::fail("GPUSceneSubsystem was not registered as a built-in subsystem: " + log);
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(GPUSceneCpuCoreTest);

class GPUSceneCompositeFlatSourceTest final : public RhiTest {
public:
    GPUSceneCompositeFlatSourceTest()
    {
        type = RhiTestType::Resource;
        name = "gpu_scene_composite_flat_source";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        std::vector<scene::RenderPrimitive> primitives{
            makeTrianglePrimitive(),
            makeTrianglePrimitive(0.25f),
        };
        primitives[0].name = "source A triangle";
        primitives[0].meshIndex = 0;
        primitives[0].primitiveIndex = 0;
        primitives[0].materialIndex = 0;
        primitives[1].name = "source B triangle";
        primitives[1].meshIndex = 0;
        primitives[1].primitiveIndex = 0;
        primitives[1].materialIndex = 1;

        std::vector<scene::RenderMaterial> materials(2);
        materials[0].name = "source A material";
        materials[1].name = "source B material";
        materials[1].baseColorFactor = float4(0.25f, 0.5f, 0.75f, 1.0f);

        const scene::SceneEntity objectA = static_cast<scene::SceneEntity>(101);
        const scene::SceneEntity objectB = static_cast<scene::SceneEntity>(202);
        std::vector<scene::RenderNode> nodes(2);
        nodes[0].object = objectA;
        nodes[0].nodeIndex = 0;
        nodes[0].renderPrimitiveIndex = 0;
        nodes[0].materialIndex = 0;
        nodes[0].worldMatrix = translationMatrix(float3(1.0f, 0.0f, 0.0f));
        nodes[0].transformRevision = 1;
        nodes[1].object = objectB;
        nodes[1].nodeIndex = 1;
        nodes[1].renderPrimitiveIndex = 1;
        nodes[1].materialIndex = 1;
        nodes[1].worldMatrix = translationMatrix(float3(-2.0f, 0.0f, 0.0f));
        nodes[1].transformRevision = 1;

        render::GPUScene gpuScene;
        std::string log;
        render::Result result = gpuScene.rebuild(
            makeSourceView(primitives, nodes, materials, 1, 1, 1),
            log);
        if (!result) {
            return RhiTestResult::fail(
                "GPUScene failed to build a flattened composite source: " + log);
        }

        const render::GPUSceneGeometryId geometryA =
            gpuScene.geometryForRenderPrimitive(0);
        const render::GPUSceneGeometryId geometryB =
            gpuScene.geometryForRenderPrimitive(1);
        const render::GPUSceneMaterialId materialA =
            gpuScene.materialForSourceMaterial(0);
        const render::GPUSceneInstanceId instanceA =
            gpuScene.instanceForRenderNode(0);
        const render::GPUSceneInstanceId instanceB =
            gpuScene.instanceForRenderNode(1);
        const scene::RenderPrimitive* geometrySourceA =
            gpuScene.geometrySourcePrimitive(geometryA);
        const scene::RenderPrimitive* geometrySourceB =
            gpuScene.geometrySourcePrimitive(geometryB);
        if (gpuScene.stats().geometryCount != 2 ||
            gpuScene.stats().materialCount != 2 ||
            gpuScene.stats().instanceCount != 2 ||
            gpuScene.stats().geometryPayloadConflictCount != 1 ||
            !geometryA || !geometryB || geometryA == geometryB ||
            !materialA || !instanceA || !instanceB ||
            geometrySourceA == nullptr || geometrySourceB == nullptr ||
            geometrySourceA->meshIndex != 0 || geometrySourceB->meshIndex != 0 ||
            geometrySourceA->primitiveIndex != 0 ||
            geometrySourceB->primitiveIndex != 0 ||
            geometrySourceA->positions[0].x == geometrySourceB->positions[0].x ||
            gpuScene.instancesForObject(objectA).size() != 1 ||
            gpuScene.instancesForObject(objectB).size() != 1 ||
            gpuScene.instance(instanceA) == nullptr ||
            gpuScene.instance(instanceA)->sourceObject != objectA ||
            gpuScene.instance(instanceB) == nullptr ||
            gpuScene.instance(instanceB)->sourceObject != objectB) {
            return RhiTestResult::fail(
                "GPUScene flattened composite identity or duplicate local primitive handling is incorrect");
        }

        const uint32_t initialGeneration = gpuScene.drawSet().generation;
        const uint64_t initialRevision = gpuScene.drawSet().revision;
        const float4x4 previousWorldB = nodes[1].worldMatrix;
        const float4x4 updatedWorldB =
            translationMatrix(float3(-2.0f, 3.0f, 4.0f));
        nodes[1].worldMatrix = updatedWorldB;
        nodes[1].transformRevision = 2;
        if (gpuScene.sync(makeSourceView(primitives, nodes, materials, 2, 1, 1)) !=
                render::GPUSceneSyncResult::Updated ||
            gpuScene.drawSet().generation != initialGeneration ||
            gpuScene.drawSet().revision == initialRevision ||
            gpuScene.instanceForRenderNode(0) != instanceA ||
            gpuScene.instanceForRenderNode(1) != instanceB ||
            gpuScene.instance(instanceB) == nullptr ||
            !sameMatrix(gpuScene.instance(instanceB)->worldMatrix, updatedWorldB) ||
            !sameMatrix(gpuScene.instance(instanceB)->previousWorldMatrix, previousWorldB)) {
            return RhiTestResult::fail(
                "GPUScene composite transform sync did not preserve IDs and transform history");
        }

        const uint64_t updatedRevision = gpuScene.drawSet().revision;
        if (gpuScene.sync(makeSourceView(primitives, nodes, materials, 2, 1, 1)) !=
                render::GPUSceneSyncResult::HistoryUpdated ||
            gpuScene.drawSet().generation != initialGeneration ||
            gpuScene.drawSet().revision == updatedRevision ||
            gpuScene.instance(instanceB) == nullptr ||
            !sameMatrix(
                gpuScene.instance(instanceB)->previousWorldMatrix,
                updatedWorldB)) {
            return RhiTestResult::fail(
                "GPUScene composite transform history did not converge on the next sync");
        }

        const uint64_t convergedRevision = gpuScene.drawSet().revision;
        if (gpuScene.sync(makeSourceView(primitives, nodes, materials, 2, 1, 1)) !=
                render::GPUSceneSyncResult::Unchanged ||
            gpuScene.drawSet().generation != initialGeneration ||
            gpuScene.drawSet().revision != convergedRevision) {
            return RhiTestResult::fail(
                "GPUScene composite transform history did not settle after convergence");
        }

        nodes[0].visible = false;
        if (gpuScene.sync(makeSourceView(primitives, nodes, materials, 2, 2, 1)) !=
                render::GPUSceneSyncResult::Updated ||
            gpuScene.drawSet().generation != initialGeneration ||
            gpuScene.instanceForRenderNode(0) != instanceA ||
            gpuScene.instance(instanceA) == nullptr ||
            gpuScene.instance(instanceA)->visible ||
            gpuScene.instance(instanceB) == nullptr ||
            !gpuScene.instance(instanceB)->visible) {
            return RhiTestResult::fail(
                "GPUScene composite visibility update did not remain incremental");
        }
        const render::GPUSceneViewId view = gpuScene.createView();
        if (!view || !gpuScene.prepareView(view, 0) ||
            gpuScene.visibleDrawSet(view, 0) == nullptr ||
            gpuScene.visibleDrawSet(view, 0)->instances.size() != 1 ||
            gpuScene.visibleDrawSet(view, 0)->instances.front() != instanceB) {
            return RhiTestResult::fail(
                "GPUScene composite visibility did not refresh the visible DrawSet");
        }

        primitives.push_back(makeTrianglePrimitive(-0.25f));
        primitives.back().name = "source C triangle";
        primitives.back().meshIndex = 0;
        primitives.back().primitiveIndex = 0;
        primitives.back().materialIndex = 2;
        materials.emplace_back();
        materials.back().name = "source C material";
        const scene::SceneEntity objectC = static_cast<scene::SceneEntity>(303);
        scene::RenderNode nodeC;
        nodeC.object = objectC;
        nodeC.nodeIndex = 2;
        nodeC.renderPrimitiveIndex = 2;
        nodeC.materialIndex = 2;
        nodeC.worldMatrix = translationMatrix(float3(0.0f, 5.0f, 0.0f));
        nodeC.transformRevision = 2;
        nodes.push_back(nodeC);
        render::GPUSceneSourceView expandedSource =
            makeSourceView(primitives, nodes, materials, 2, 2, 2);
        expandedSource.structuralRevision = 2;
        if (gpuScene.sync(expandedSource) !=
            render::GPUSceneSyncResult::RebuildRequired) {
            return RhiTestResult::fail(
                "GPUScene did not request a rebuild after adding a composite source");
        }

        result = gpuScene.rebuild(expandedSource, log);
        const render::GPUSceneInstanceId instanceC =
            gpuScene.instanceForRenderNode(2);
        if (!result || gpuScene.drawSet().generation == initialGeneration ||
            gpuScene.geometry(geometryA) != nullptr ||
            gpuScene.material(materialA) != nullptr ||
            gpuScene.instance(instanceA) != nullptr ||
            gpuScene.stats().geometryCount != 3 ||
            gpuScene.stats().materialCount != 3 ||
            gpuScene.stats().instanceCount != 3 ||
            !instanceC || gpuScene.instance(instanceC) == nullptr ||
            gpuScene.instance(instanceC)->sourceObject != objectC ||
            gpuScene.instancesForObject(objectC).size() != 1 ||
            !gpuScene.prepareView(view, 0)) {
            return RhiTestResult::fail(
                "GPUScene composite topology rebuild did not replace generations and mappings");
        }

        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(GPUSceneCompositeFlatSourceTest);

class GPUSceneSourceOverrideLeaseTest final : public RhiTest {
public:
    GPUSceneSourceOverrideLeaseTest()
    {
        type = RhiTestType::Resource;
        name = "gpu_scene_source_override_lease";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::GPUSceneSubsystem subsystem;
        scene::Scene sceneA;
        scene::Scene sceneB;
        std::string log;
        render::GPUSceneSourceOverrideToken first;
        render::GPUSceneSourceOverrideToken second;
        render::GPUSceneSourceOverrideToken conflict;

        if (!subsystem.acquireSourceOverride(&sceneA, first, log) || !first ||
            subsystem.sourceOverride() != &sceneA) {
            return RhiTestResult::fail("GPUScene failed to acquire the first source override lease: " + log);
        }
        if (!subsystem.acquireSourceOverride(&sceneA, second, log) || !second ||
            second == first) {
            return RhiTestResult::fail("GPUScene failed to share a source override lease for the same Scene: " + log);
        }
        if (subsystem.acquireSourceOverride(&sceneB, conflict, log) || conflict ||
            subsystem.sourceOverride() != &sceneA) {
            return RhiTestResult::fail("GPUScene accepted concurrent source override leases for different Scenes");
        }

        subsystem.setSourceOverride(&sceneB);
        if (subsystem.sourceOverride() != &sceneA) {
            return RhiTestResult::fail("Legacy source override replaced an active lease");
        }
        if (!subsystem.releaseSourceOverride(first) ||
            subsystem.sourceOverride() != &sceneA) {
            return RhiTestResult::fail("Releasing one shared lease cleared another consumer's lease");
        }
        if (subsystem.releaseSourceOverride(first)) {
            return RhiTestResult::fail("GPUScene accepted a stale source override lease token");
        }
        if (!subsystem.releaseSourceOverride(second) ||
            subsystem.sourceOverride() != &sceneB) {
            return RhiTestResult::fail("GPUScene did not restore the compatible legacy override after the last lease");
        }
        if (!subsystem.clearSourceOverride(&sceneB) || subsystem.sourceOverride() != nullptr) {
            return RhiTestResult::fail("GPUScene failed to clear the legacy source override after lease release");
        }
        if (subsystem.releaseSourceOverride({}) ||
            subsystem.releaseSourceOverride(second)) {
            return RhiTestResult::fail("GPUScene accepted an invalid or stale source override lease token");
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(GPUSceneSourceOverrideLeaseTest);

class GPUSceneGpuResourcesTest final : public RhiTest {
public:
    GPUSceneGpuResourcesTest()
    {
        type = RhiTestType::Command;
        name = "gpu_scene_global_gpu_resources";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic GPUScene GPU Resources Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            },
            device);
        if (!result) {
            return render::hasError(result, render::Error::Unsupported)
                ? RhiTestResult::skip(std::string("createDevice returned ") + toString(result))
                : RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("GPUScene binding test requires bindless buffers");
        }
        render::Queue* queue = device->getQueue(render::QueueType::Graphics);
        if (queue == nullptr) {
            return RhiTestResult::skip("GPUScene GPU resources test requires a graphics queue");
        }

        std::string log;
        render::RenderSubsystemHost host;
        if (!render::registerBuiltInRenderSubsystems(host, log)) {
            return RhiTestResult::fail("GPUScene built-in registration failed: " + log);
        }
        result = host.initialize(*device, 3, log);
        if (!result || !host.activate(render::GPUSceneSubsystem::kSubsystemId, log)) {
            return RhiTestResult::fail("GPUScene subsystem activation failed: " + log);
        }
        auto* subsystem = host.get<render::GPUSceneSubsystem>();
        if (subsystem == nullptr) {
            return RhiTestResult::fail("GPUScene subsystem lookup failed after activation");
        }

        std::unique_ptr<render::BindlessHeap> bindlessHeap;
        result = device->createBindlessHeap(
            render::BindlessHeapDesc{.maxBuffers = 64},
            bindlessHeap);
        if (!result || bindlessHeap == nullptr) {
            return RhiTestResult::fail(
                std::string("createBindlessHeap returned ") + toString(result));
        }

        std::vector<scene::RenderPrimitive> primitives{
            makeCanonicalRasterPrimitive(),
            makeTrianglePrimitive(0.25f),
        };
        std::vector<scene::RenderMaterial> materials(2);
        materials[0].name = "GPUScene GPU material";
        materials[0].baseColorTexture.textureIndex = 7;
        materials[0].baseColorTexture.texCoord = 1;
        materials[0].baseColorTexture.uvTransform = {
            2.0f, 0.0f, 0.25f,
            0.0f, 3.0f, 0.5f,
        };
        materials[0].metallicRoughnessTexture.textureIndex = 8;
        materials[0].normalTexture.textureIndex = 9;
        materials[0].occlusionTexture.textureIndex = 10;
        materials[0].emissiveTexture.textureIndex = 11;
        materials[0].transmissionTexture.textureIndex = 12;
        materials[0].thicknessTexture.textureIndex = 13;
        materials[0].diffuseTransmissionTexture.textureIndex = 14;
        materials[0].diffuseTransmissionColorTexture.textureIndex = 15;
        materials[1].name = "GPUScene retained blend material";
        materials[1].alphaMode = "BLEND";
        std::vector<scene::RenderNode> nodes(4);
        for (uint32_t nodeIndex = 0; nodeIndex < 3; ++nodeIndex) {
            nodes[nodeIndex].object = static_cast<scene::SceneEntity>(nodeIndex + 1);
            nodes[nodeIndex].nodeIndex = static_cast<int32_t>(nodeIndex);
            nodes[nodeIndex].renderPrimitiveIndex = nodeIndex < 2 ? 0 : 1;
            nodes[nodeIndex].materialIndex = nodeIndex == 1 ? 1 : 0;
            nodes[nodeIndex].transformRevision = 1;
        }
        nodes[3].object = static_cast<scene::SceneEntity>(4);
        nodes[3].nodeIndex = 3;
        nodes[3].renderPrimitiveIndex = scene::kInvalidSceneIndex;

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*queue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(
                std::string("createCommandPool returned ") + toString(result));
        }

        auto submitFrame = [&]<typename Check>(
                               uint64_t frameIndex,
                               uint32_t frameSlot,
                               Check&& check) -> RhiTestResult {
            log.clear();
            render::Result frameResult = host.beginFrame(frameIndex, frameSlot, nullptr, log);
            if (!frameResult) {
                return RhiTestResult::fail("GPUScene beginFrame failed: " + log);
            }
            std::unique_ptr<render::CommandBuffer> commandBuffer;
            frameResult = commandPool->createCommandBuffer(commandBuffer);
            if (!frameResult || commandBuffer == nullptr || !commandBuffer->begin()) {
                host.endFrame();
                return RhiTestResult::fail("GPUScene command-buffer creation/begin failed");
            }
            constexpr std::array<render::RenderSubsystemId, 1> kRequired{
                render::GPUSceneSubsystem::kSubsystemId,
            };
            frameResult = host.recordPreGraph(
                *commandBuffer,
                nullptr,
                kRequired,
                log);
            if (!frameResult) {
                host.endFrame();
                return RhiTestResult::fail("GPUScene recordPreGraph failed: " + log);
            }
            RhiTestResult checkResult = [&]() {
                if constexpr (requires { check(*commandBuffer); }) {
                    return check(*commandBuffer);
                } else {
                    return check();
                }
            }();
            if (!checkResult.passed) {
                host.endFrame();
                return checkResult;
            }
            frameResult = host.recordPostGraph(
                *commandBuffer,
                nullptr,
                kRequired,
                log);
            if (!frameResult || !commandBuffer->end()) {
                host.endFrame();
                return RhiTestResult::fail("GPUScene command recording failed: " + log);
            }
            std::unique_ptr<render::Fence> fence;
            frameResult = device->createFence(false, fence);
            if (!frameResult || fence == nullptr) {
                host.endFrame();
                return RhiTestResult::fail("GPUScene fence creation failed");
            }
            render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
            frameResult = queue->submit(render::QueueSubmitDesc{
                .commandBuffers = commandBuffers,
                .commandBufferCount = 1,
                .signalFence = fence.get(),
            });
            if (frameResult) {
                frameResult = fence->wait(5'000'000'000ull);
            }
            host.endFrame();
            return frameResult
                ? RhiTestResult::pass()
                : RhiTestResult::fail(
                    std::string("GPUScene submit/wait returned ") + toString(frameResult));
        };

        // A newly activated subsystem with no World/Scene must remain a valid
        // no-op until a non-zero DrawSet generation exists.
        RhiTestResult emptyFrameResult = submitFrame(0, 0, [&]() {
            if (subsystem->globalBufferViews().validFor(0, 0) ||
                subsystem->gpuUploadStats().fullUploadCount != 0) {
                return RhiTestResult::fail("GPUScene uploaded invalid generation-zero resources");
            }
            return RhiTestResult::pass();
        });
        if (!emptyFrameResult.passed) {
            return emptyFrameResult;
        }
        result = subsystem->scene().rebuild(
            makeSourceView(primitives, nodes, materials, 1, 1, 1),
            log);
        if (!result) {
            return RhiTestResult::fail("GPUScene test DrawSet rebuild failed: " + log);
        }
        const render::GPUSceneGeometryId canonicalGeometry =
            subsystem->scene().geometryForRenderPrimitive(0);
        const scene::RenderPrimitive* canonicalSource =
            subsystem->scene().geometrySourcePrimitive(canonicalGeometry);
        if (!canonicalGeometry || canonicalSource == nullptr ||
            canonicalSource->meshletClusters.size() != 1 ||
            canonicalSource->meshletLodLevels.size() != 2) {
            return RhiTestResult::fail(
                "GPUScene did not retain generational canonical geometry backing");
        }
        if (subsystem->instances().size() != 3 ||
            subsystem->instanceForRenderNode(3) ||
            subsystem->stats().skippedRenderNodeCount != 1) {
            return RhiTestResult::fail(
                "GPUScene did not skip the invalid RenderNode without creating an instance hole");
        }
        for (uint32_t instanceIndex = 0;
             instanceIndex < subsystem->instances().size();
             ++instanceIndex) {
            if (subsystem->instances()[instanceIndex].id.index != instanceIndex) {
                return RhiTestResult::fail(
                    "GPUScene global instance IDs are not dense after skipped nodes");
            }
        }

        render::GPUSceneConsumerBindings bindings;
        render::Buffer* firstGeometry = nullptr;
        render::Buffer* firstMaterial = nullptr;
        render::Buffer* firstInstance = nullptr;
        render::Buffer* firstDrawKeys = nullptr;
        std::unique_ptr<render::Buffer> canonicalReadback;
        constexpr uint64_t kGeometryReadbackOffset = 0;
        constexpr uint64_t kMeshletReadbackOffset =
            kGeometryReadbackOffset + 2u * sizeof(render::GPUSceneGpuGeometryRecord);
        constexpr uint64_t kMeshletDrawReadbackOffset =
            kMeshletReadbackOffset + 3u * sizeof(render::GPUSceneGpuMeshletRecord);
        constexpr uint64_t kMaterialReadbackOffset =
            kMeshletDrawReadbackOffset +
            6u * sizeof(render::GPUSceneGpuMeshletDrawRecord);
        constexpr uint64_t kMeshletVertexReadbackOffset = kMaterialReadbackOffset;
        constexpr uint64_t kOpenPbrMaterialReadbackOffset =
            kMeshletVertexReadbackOffset + 9u * sizeof(uint32_t);
        constexpr uint64_t kDescriptorRemapReadbackOffset =
            kOpenPbrMaterialReadbackOffset +
            2u * sizeof(render::GPUSceneGpuMaterialRecord);
        constexpr uint64_t kTriangleWordReadbackOffset =
            kDescriptorRemapReadbackOffset +
            2u * render::kGPUSceneMaterialTextureSlotCount *
                sizeof(render::GPUSceneGpuDescriptorRemapRecord);
        constexpr uint64_t kCanonicalReadbackSize =
            kTriangleWordReadbackOffset + 3u * sizeof(uint32_t);
        uint32_t firstGeneration = 0;
        uint64_t firstRevision = 0;
        RhiTestResult frameResult = submitFrame(1, 1, [&](render::CommandBuffer& commandBuffer) {
            const render::GPUSceneGlobalBufferViews& views = subsystem->globalBufferViews();
            firstGeneration = subsystem->drawSet().generation;
            firstRevision = subsystem->drawSet().revision;
            if (!views.validFor(firstGeneration, firstRevision) ||
                !views.geometries.valid() || !views.materials.valid() ||
                !views.instances.valid() || !views.drawKeys.valid() ||
                !views.drawInstanceIds.valid() || !views.vertices.valid() ||
                !views.indices.valid() || !views.meshlets.valid() ||
                !views.meshletDraws.valid() || !views.meshletVertices.valid() ||
                !views.meshletTriangleWords.valid() ||
                !views.descriptorRemap.valid()) {
                return RhiTestResult::fail("GPUScene did not publish all mandatory global buffer views");
            }
            const std::array mandatory{
                views.geometries.buffer,
                views.materials.buffer,
                views.instances.buffer,
                views.drawKeys.buffer,
                views.drawInstanceIds.buffer,
                views.vertices.buffer,
                views.indices.buffer,
                views.meshlets.buffer,
                views.meshletDraws.buffer,
                views.meshletVertices.buffer,
                views.meshletTriangleWords.buffer,
                views.descriptorRemap.buffer,
            };
            for (render::Buffer* buffer : mandatory) {
                if (buffer == nullptr ||
                    buffer->desc().memoryLocation != render::MemoryLocation::Device ||
                    !render::hasFlag(buffer->desc().usage, render::BufferUsageBits::Storage) ||
                    !render::hasFlag(
                        buffer->desc().usage,
                        render::BufferUsageBits::TransferDestination)) {
                    return RhiTestResult::fail("GPUScene global buffer is not device-local Storage|TransferDestination");
                }
            }
            if (views.geometries.size != 2u * sizeof(render::GPUSceneGpuGeometryRecord) ||
                views.vertices.size != 6u * sizeof(render::GPUSceneGpuVertexRecord) ||
                views.indices.size != 6u * sizeof(uint32_t) ||
                views.meshlets.size != 3u * sizeof(render::GPUSceneGpuMeshletRecord) ||
                views.meshletDraws.size !=
                    6u * sizeof(render::GPUSceneGpuMeshletDrawRecord) ||
                views.meshletVertices.size != 9u * sizeof(uint32_t) ||
                // Three local u8 triangle triplets pack into three uint words.
                views.meshletTriangleWords.size != 3u * sizeof(uint32_t) ||
                views.materials.size !=
                    2u * sizeof(render::GPUSceneGpuMaterialRecord) ||
                views.descriptorRemap.size !=
                    2u * render::kGPUSceneMaterialTextureSlotCount *
                        sizeof(render::GPUSceneGpuDescriptorRemapRecord)) {
                return RhiTestResult::fail(
                    "GPUScene canonical raster payload was duplicated or has an invalid ABI size");
            }
            const render::GPUSceneRasterDrawLayout& rasterLayout =
                subsystem->rasterDrawLayout();
            if (!rasterLayout.validFor(firstGeneration, firstRevision) ||
                rasterLayout.baseRange != render::GPUSceneRasterDrawRange{0, 2} ||
                rasterLayout.lodRanges.size() != 2 ||
                rasterLayout.lodRanges[0] != render::GPUSceneRasterDrawRange{2, 2} ||
                rasterLayout.lodRanges[1] != render::GPUSceneRasterDrawRange{4, 2} ||
                rasterLayout.maxRangeCount != 2) {
                return RhiTestResult::fail(
                    "GPUScene base/LOD meshlet draw ranges are not stable or omitted BLEND");
            }
            firstGeometry = views.geometries.buffer;
            firstMaterial = views.materials.buffer;
            firstInstance = views.instances.buffer;
            firstDrawKeys = views.drawKeys.buffer;
            log.clear();
            render::Result bindingResult =
                subsystem->createBindings(*bindlessHeap, bindings, log);
            if (!bindingResult || !bindings.validFor(views) ||
                !bindings[render::GPUSceneGlobalBufferKind::Geometries].valid() ||
                !bindings[render::GPUSceneGlobalBufferKind::Instances].valid() ||
                !bindings[render::GPUSceneGlobalBufferKind::MeshletDraws].valid() ||
                !bindings[render::GPUSceneGlobalBufferKind::DescriptorRemap].valid()) {
                return RhiTestResult::fail("GPUScene consumer binding creation failed: " + log);
            }
            if (subsystem->gpuUploadStats().fullUploadCount != 1 ||
                subsystem->gpuUploadStats().instanceUploadCount != 0) {
                return RhiTestResult::fail("GPUScene full-upload statistics are incorrect");
            }

            render::Result readbackResult = device->createBuffer(
                render::BufferDesc{
                    .size = kCanonicalReadbackSize,
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback,
                    .queueAccess = render::QueueAccessBits::Graphics,
                },
                canonicalReadback);
            if (!readbackResult || canonicalReadback == nullptr) {
                return RhiTestResult::fail(
                    "GPUScene canonical payload readback allocation failed");
            }

            const std::array sourceViews{
                &views.geometries,
                &views.meshlets,
                &views.meshletDraws,
                &views.meshletVertices,
                &views.materials,
                &views.descriptorRemap,
                &views.meshletTriangleWords,
            };
            std::array<render::BufferBarrierDesc, sourceViews.size()> toCopy{};
            std::array<render::BufferBarrierDesc, sourceViews.size()> toRead{};
            for (size_t index = 0; index < sourceViews.size(); ++index) {
                toCopy[index] = render::BufferBarrierDesc{
                    .buffer = sourceViews[index]->buffer,
                    .before = render::ResourceState::ShaderRead,
                    .after = render::ResourceState::TransferSource,
                    .offset = 0,
                    .size = sourceViews[index]->size,
                };
                toRead[index] = render::BufferBarrierDesc{
                    .buffer = sourceViews[index]->buffer,
                    .before = render::ResourceState::TransferSource,
                    .after = render::ResourceState::ShaderRead,
                    .offset = 0,
                    .size = sourceViews[index]->size,
                };
            }
            render::BufferBarrierDesc readbackDestination{
                .buffer = canonicalReadback.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::TransferDestination,
                .offset = 0,
                .size = kCanonicalReadbackSize,
            };
            commandBuffer.barrier(render::BarrierDesc{
                .buffers = &readbackDestination,
                .bufferCount = 1,
            });
            commandBuffer.barrier(render::BarrierDesc{
                .buffers = toCopy.data(),
                .bufferCount = static_cast<uint32_t>(toCopy.size()),
            });
            const std::array destinationOffsets{
                kGeometryReadbackOffset,
                kMeshletReadbackOffset,
                kMeshletDrawReadbackOffset,
                kMeshletVertexReadbackOffset,
                kOpenPbrMaterialReadbackOffset,
                kDescriptorRemapReadbackOffset,
                kTriangleWordReadbackOffset,
            };
            for (size_t index = 0; index < sourceViews.size(); ++index) {
                commandBuffer.copyBuffer(render::BufferCopyDesc{
                    .source = sourceViews[index]->buffer,
                    .destination = canonicalReadback.get(),
                    .sourceOffset = 0,
                    .destinationOffset = destinationOffsets[index],
                    .size = sourceViews[index]->size,
                });
            }
            commandBuffer.barrier(render::BarrierDesc{
                .buffers = toRead.data(),
                .bufferCount = static_cast<uint32_t>(toRead.size()),
            });
            return RhiTestResult::pass();
        });
        if (!frameResult.passed) {
            return frameResult;
        }

        canonicalReadback->invalidate(0, kCanonicalReadbackSize);
        const void* canonicalMapped = canonicalReadback->map();
        if (canonicalMapped == nullptr) {
            return RhiTestResult::fail("GPUScene canonical payload readback did not map");
        }
        const auto* canonicalBytes = static_cast<const uint8_t*>(canonicalMapped);
        std::array<render::GPUSceneGpuGeometryRecord, 2> geometryRecords;
        std::array<render::GPUSceneGpuMeshletRecord, 3> meshletRecords;
        std::array<render::GPUSceneGpuMeshletDrawRecord, 6> meshletDrawRecords;
        std::array<uint32_t, 9> meshletVertexRecords;
        std::array<render::GPUSceneGpuMaterialRecord, 2> materialRecords;
        std::array<
            render::GPUSceneGpuDescriptorRemapRecord,
            2u * render::kGPUSceneMaterialTextureSlotCount>
            descriptorRemapRecords;
        std::array<uint32_t, 3> triangleWordRecords;
        std::memcpy(
            geometryRecords.data(),
            canonicalBytes + kGeometryReadbackOffset,
            sizeof(geometryRecords));
        std::memcpy(
            meshletRecords.data(),
            canonicalBytes + kMeshletReadbackOffset,
            sizeof(meshletRecords));
        std::memcpy(
            meshletDrawRecords.data(),
            canonicalBytes + kMeshletDrawReadbackOffset,
            sizeof(meshletDrawRecords));
        std::memcpy(
            meshletVertexRecords.data(),
            canonicalBytes + kMeshletVertexReadbackOffset,
            sizeof(meshletVertexRecords));
        std::memcpy(
            materialRecords.data(),
            canonicalBytes + kOpenPbrMaterialReadbackOffset,
            sizeof(materialRecords));
        std::memcpy(
            descriptorRemapRecords.data(),
            canonicalBytes + kDescriptorRemapReadbackOffset,
            sizeof(descriptorRemapRecords));
        std::memcpy(
            triangleWordRecords.data(),
            canonicalBytes + kTriangleWordReadbackOffset,
            sizeof(triangleWordRecords));
        canonicalReadback->unmap();

        if (geometryRecords[0].payload != std::array<uint32_t, 4>{0, 0, 0, 1} ||
            geometryRecords[0].counts[3] != 2 ||
            geometryRecords[1].payload[0] != 3 ||
            geometryRecords[1].payload[2] != 3 ||
            geometryRecords[1].payload[3] != 0) {
            return RhiTestResult::fail(
                "GPUScene geometry records do not expose the canonical vertex/base-meshlet ranges");
        }
        for (uint32_t meshletIndex = 0; meshletIndex < meshletRecords.size(); ++meshletIndex) {
            const render::GPUSceneGpuMeshletRecord& meshlet =
                meshletRecords[meshletIndex];
            if (meshlet.ranges != std::array<uint32_t, 4>{
                    meshletIndex * 3u,
                    3,
                    meshletIndex,
                    1,
                } ||
                meshlet.lod[0] != (meshletIndex == 2 ? 1u : 0u) ||
                meshlet.boundingSphere[3] <= 0.0f ||
                meshlet.coneAxisLodError[2] != 1.0f) {
                return RhiTestResult::fail(
                    "GPUScene canonical base/LOD meshlet metadata is incorrect");
            }
        }
        constexpr std::array<uint32_t, 9> kExpectedLocalMeshletVertices{
            0, 1, 2,
            0, 1, 2,
            0, 1, 2,
        };
        if (meshletVertexRecords != kExpectedLocalMeshletVertices ||
            triangleWordRecords != std::array<uint32_t, 3>{
                0x00020100u,
                0x00020100u,
                0x00020100u,
            }) {
            return RhiTestResult::fail(
                "GPUScene meshlet vertices are not geometry-local or triangle bytes are not packed");
        }
        for (uint32_t rangeIndex = 0; rangeIndex < 3; ++rangeIndex) {
            const uint32_t firstDraw = rangeIndex * 2u;
            const uint32_t expectedMeshlet = rangeIndex;
            const render::GPUSceneGpuMeshletDrawRecord& opaque =
                meshletDrawRecords[firstDraw];
            const render::GPUSceneGpuMeshletDrawRecord& blend =
                meshletDrawRecords[firstDraw + 1u];
            if (opaque.meshletIndex != expectedMeshlet || opaque.instanceIndex != 0 ||
                opaque.geometryIndex != 0 ||
                opaque.drawBucket != static_cast<uint32_t>(
                    render::GPUSceneDrawBucket::OpaqueSingleSided) ||
                blend.meshletIndex != expectedMeshlet || blend.instanceIndex != 1 ||
                blend.geometryIndex != 0 ||
                blend.drawBucket != static_cast<uint32_t>(render::GPUSceneDrawBucket::Blend)) {
                return RhiTestResult::fail(
                    "GPUScene MeshletDraws do not use dense global instance IDs or retain BLEND");
            }
        }
        if (materialRecords[0].baseColorTexture.textureIndex != 0 ||
            materialRecords[0].baseColorTexture.texCoord != 1 ||
            materialRecords[0].baseColorTexture.transform0 !=
                std::array<float, 4>{2.0f, 0.0f, 0.25f, 0.0f} ||
            materialRecords[0].baseColorTexture.transform1 !=
                std::array<float, 4>{0.0f, 3.0f, 0.5f, 0.0f} ||
            materialRecords[0].diffuseTransmissionColorTexture.textureIndex != 8 ||
            materialRecords[0].identity[0] != 0 ||
            materialRecords[1].baseColorTexture.textureIndex != 9 ||
            materialRecords[1].identity[0] != 1) {
            return RhiTestResult::fail(
                "GPUScene OpenPBR material ABI or texture-remap indices are incorrect");
        }
        for (uint32_t textureSlot = 0;
             textureSlot < render::kGPUSceneMaterialTextureSlotCount;
             ++textureSlot) {
            const render::GPUSceneGpuDescriptorRemapRecord& remap =
                descriptorRemapRecords[textureSlot];
            if (remap.logicalTextureId != static_cast<int32_t>(7 + textureSlot) ||
                remap.descriptorIndex != std::numeric_limits<uint32_t>::max() ||
                remap.materialIndex != 0 || remap.textureSlot != textureSlot) {
                return RhiTestResult::fail(
                    "GPUScene descriptor remap lost a logical material texture reference");
            }
        }
        if (descriptorRemapRecords[9].logicalTextureId != scene::kInvalidSceneIndex ||
            descriptorRemapRecords[9].descriptorIndex !=
                std::numeric_limits<uint32_t>::max() ||
            descriptorRemapRecords[9].materialIndex != 1 ||
            descriptorRemapRecords[9].textureSlot != 0) {
            return RhiTestResult::fail(
                "GPUScene invalid logical textures were written back as consumer descriptors");
        }

        nodes[0].worldMatrix = translationMatrix(float3(2.0f, 3.0f, 4.0f));
        nodes[0].transformRevision = 2;
        if (subsystem->scene().sync(
                makeSourceView(primitives, nodes, materials, 2, 1, 1)) !=
            render::GPUSceneSyncResult::Updated) {
            return RhiTestResult::fail("GPUScene incremental test did not update the CPU instance");
        }
        frameResult = submitFrame(2, 2, [&]() {
            const render::GPUSceneGlobalBufferViews& views = subsystem->globalBufferViews();
            if (!views.validFor(subsystem->drawSet().generation, subsystem->drawSet().revision) ||
                views.geometries.buffer != firstGeometry ||
                views.materials.buffer != firstMaterial ||
                views.instances.buffer != firstInstance ||
                views.drawKeys.buffer != firstDrawKeys) {
                return RhiTestResult::fail("GPUScene incremental sync recreated a global device buffer");
            }
            if (!subsystem->rasterDrawLayout().validFor(
                    subsystem->drawSet().generation,
                    subsystem->drawSet().revision) ||
                subsystem->rasterDrawLayout().baseRange.count != 2) {
                return RhiTestResult::fail(
                    "GPUScene incremental sync invalidated its generation-scoped raster layout");
            }
            if (!bindings.validFor(views)) {
                return RhiTestResult::fail(
                    "GPUScene invalidated generation-scoped bindings after an in-place instance update");
            }
            if (subsystem->gpuUploadStats().fullUploadCount != 1 ||
                subsystem->gpuUploadStats().instanceUploadCount != 1) {
                return RhiTestResult::fail("GPUScene instance-only upload statistics are incorrect");
            }
            return RhiTestResult::pass();
        });
        if (!frameResult.passed) {
            return frameResult;
        }

        if (subsystem->scene().sync(
                makeSourceView(primitives, nodes, materials, 2, 1, 1)) !=
            render::GPUSceneSyncResult::HistoryUpdated) {
            return RhiTestResult::fail("GPUScene incremental test did not advance transform history");
        }
        frameResult = submitFrame(3, 0, [&]() {
            const render::GPUSceneGlobalBufferViews& views = subsystem->globalBufferViews();
            if (views.instances.buffer != firstInstance ||
                subsystem->gpuUploadStats().instanceUploadCount != 2) {
                return RhiTestResult::fail("GPUScene history-only upload did not reuse the instance buffer");
            }
            if (!bindings.validFor(views)) {
                return RhiTestResult::fail(
                    "GPUScene invalidated generation-scoped bindings after a history-only update");
            }
            return RhiTestResult::pass();
        });
        if (!frameResult.passed) {
            return frameResult;
        }

        materials[0].alphaMode = "MASK";
        result = subsystem->scene().rebuild(
            makeSourceView(primitives, nodes, materials, 2, 1, 2),
            log);
        if (!result || subsystem->drawSet().generation == firstGeneration) {
            return RhiTestResult::fail("GPUScene full rebuild did not advance generation");
        }
        frameResult = submitFrame(4, 1, [&]() {
            const render::GPUSceneGlobalBufferViews& views = subsystem->globalBufferViews();
            if (!views.validFor(subsystem->drawSet().generation, subsystem->drawSet().revision) ||
                views.geometries.buffer == firstGeometry ||
                views.materials.buffer == firstMaterial ||
                views.instances.buffer == firstInstance ||
                views.drawKeys.buffer == firstDrawKeys ||
                bindings.validFor(views)) {
                return RhiTestResult::fail("GPUScene full rebuild did not replace resources and stale bindings");
            }
            if (subsystem->gpuUploadStats().fullUploadCount != 2) {
                return RhiTestResult::fail("GPUScene rebuild upload statistics are incorrect");
            }
            subsystem->releaseBindings(*bindlessHeap, bindings);
            log.clear();
            if (!subsystem->createBindings(*bindlessHeap, bindings, log) ||
                !bindings.validFor(views)) {
                return RhiTestResult::fail("GPUScene rebuild binding creation failed: " + log);
            }
            return RhiTestResult::pass();
        });
        if (!frameResult.passed) {
            return frameResult;
        }

        subsystem->releaseBindings(*bindlessHeap, bindings);
        host.shutdown();
        result = device->waitIdle();
        return result
            ? RhiTestResult::pass()
            : RhiTestResult::fail(std::string("GPUScene device waitIdle returned ") + toString(result));
    }
};

METALLIC_REGISTER_RHI_TEST(GPUSceneGpuResourcesTest);

class GPUSceneViewGpuResourcesTest final : public RhiTest {
public:
    GPUSceneViewGpuResourcesTest()
    {
        type = RhiTestType::Command;
        name = "gpu_scene_view_gpu_resources";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic GPUScene View GPU Resources Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            },
            device);
        if (!result) {
            return render::hasError(result, render::Error::Unsupported)
                ? RhiTestResult::skip(std::string("createDevice returned ") + toString(result))
                : RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        render::Queue* queue = device->getQueue(render::QueueType::Graphics);
        if (queue == nullptr) {
            return RhiTestResult::skip(
                "GPUScene View resource test requires a graphics queue");
        }

        std::string log;
        render::RenderSubsystemHost host;
        if (!render::registerBuiltInRenderSubsystems(host, log)) {
            return RhiTestResult::fail(
                "GPUScene View resource registration failed: " + log);
        }
        result = host.initialize(*device, 3, log);
        if (!result || !host.activate(render::GPUSceneSubsystem::kSubsystemId, log)) {
            return RhiTestResult::fail(
                "GPUScene View resource subsystem activation failed: " + log);
        }
        auto* subsystem = host.get<render::GPUSceneSubsystem>();
        if (subsystem == nullptr) {
            return RhiTestResult::fail(
                "GPUScene View resource subsystem lookup failed");
        }
        result = host.beginFrame(0, 0, nullptr, log);
        if (!result) {
            return RhiTestResult::fail(
                "GPUScene View resource warm-up beginFrame failed: " + log);
        }
        host.endFrame();

        std::vector<scene::RenderPrimitive> primitives{makeTrianglePrimitive()};
        std::vector<scene::RenderMaterial> materials(1);
        std::vector<scene::RenderNode> nodes(1);
        nodes[0].object = static_cast<scene::SceneEntity>(1);
        nodes[0].nodeIndex = 0;
        nodes[0].renderPrimitiveIndex = 0;
        nodes[0].materialIndex = 0;
        nodes[0].transformRevision = 1;
        result = subsystem->scene().rebuild(
            makeSourceView(primitives, nodes, materials, 1, 1, 1),
            log);
        if (!result) {
            return RhiTestResult::fail(
                "GPUScene View resource DrawSet rebuild failed: " + log);
        }

        const render::GPUSceneViewDesc viewDesc{
            .frameSlotCount = 3,
            .instanceCapacity = 8,
            .visibleMeshletCapacity = {11, 13, 17, 19},
            .hzbWidth = 8,
            .hzbHeight = 4,
            .hzbMipCount = 4,
            .hzbElementCount = 43,
        };
        render::GPUSceneViewId firstView;
        render::GPUSceneViewId secondView;
        result = subsystem->createView(viewDesc, firstView, log);
        if (result) {
            result = subsystem->createView(viewDesc, secondView, log);
        }
        if (!result || !firstView || !secondView) {
            return RhiTestResult::fail(
                "GPUScene failed to allocate two GPU-backed Views: " + log);
        }

        std::array<render::GPUSceneViewGpuResourcesView, 3> firstSlots;
        for (uint32_t frameSlot = 0; frameSlot < firstSlots.size(); ++frameSlot) {
            if (!subsystem->viewGpuResources(
                    firstView,
                    frameSlot,
                    firstSlots[frameSlot])) {
                return RhiTestResult::fail(
                    "GPUScene did not expose all three View frame-slot bundles");
            }
        }
        render::GPUSceneViewGpuResourcesView secondSlot;
        if (!subsystem->viewGpuResources(secondView, 0, secondSlot)) {
            return RhiTestResult::fail(
                "GPUScene did not expose the second View GPU bundle");
        }
        constexpr std::array<uint32_t, 4> kExpectedOffsets{0, 11, 24, 41};
        for (uint32_t frameSlot = 0; frameSlot < firstSlots.size(); ++frameSlot) {
            const render::GPUSceneViewGpuResourcesView& slot = firstSlots[frameSlot];
            if (slot.instanceVisibilityStates.buffer == nullptr ||
                slot.instanceVisibilityStates.view == nullptr ||
                slot.instanceVisibilityStates.size != 8u * sizeof(uint32_t) ||
                slot.visibleInstanceIds.buffer == nullptr ||
                slot.visibleInstanceIds.size != 8u * sizeof(uint32_t) ||
                slot.visibleInstanceCounter.buffer == nullptr ||
                slot.visibleInstanceCounter.size != sizeof(uint32_t) ||
                slot.hzbHistory[0].buffer != firstSlots[0].hzbHistory[0].buffer ||
                slot.hzbHistory[1].buffer != firstSlots[0].hzbHistory[1].buffer ||
                slot.hzbHistory[0].size != 43u * sizeof(float)) {
                return RhiTestResult::fail(
                    "GPUScene View resource sizes or per-View HZB sharing are incorrect");
            }
            for (uint32_t phaseIndex = 0;
                 phaseIndex < render::kGPUSceneCullPhaseCount;
                 ++phaseIndex) {
                const render::GPUSceneCullPhaseGpuView& phase =
                    slot.phases[phaseIndex];
                if (phase.visibleMeshletIds.buffer == nullptr ||
                    phase.visibleMeshletIds.size != 60u * sizeof(uint32_t)) {
                    return RhiTestResult::fail(
                        "GPUScene visible-meshlet worklist size is incorrect");
                }
                for (uint32_t bucketIndex = 0;
                     bucketIndex < render::kGPUSceneRasterDrawBucketCount;
                     ++bucketIndex) {
                    const render::GPUSceneBucketGpuView& bucket =
                        phase.buckets[bucketIndex];
                    if (bucket.visibleMeshletOffset != kExpectedOffsets[bucketIndex] ||
                        bucket.visibleMeshletCapacity !=
                            viewDesc.visibleMeshletCapacity[bucketIndex] ||
                        bucket.indirectArguments.offset != bucketIndex * 16u ||
                        bucket.indirectArguments.size != 12u ||
                        bucket.overflow.offset != bucketIndex * 16u + 12u ||
                        bucket.overflow.size != 4u ||
                        bucket.indirectArguments.buffer != bucket.overflow.buffer ||
                        !render::hasFlag(
                            bucket.indirectArguments.buffer->desc().usage,
                            render::BufferUsageBits::Indirect)) {
                        return RhiTestResult::fail(
                            "GPUScene bucket worklist or 16-byte indirect layout is incorrect");
                    }
                }
            }
        }
        for (uint32_t lhs = 0; lhs < firstSlots.size(); ++lhs) {
            for (uint32_t rhs = lhs + 1; rhs < firstSlots.size(); ++rhs) {
                if (firstSlots[lhs].instanceVisibilityStates.buffer ==
                        firstSlots[rhs].instanceVisibilityStates.buffer ||
                    firstSlots[lhs].visibleInstanceIds.buffer ==
                        firstSlots[rhs].visibleInstanceIds.buffer ||
                    firstSlots[lhs].visibleInstanceCounter.buffer ==
                        firstSlots[rhs].visibleInstanceCounter.buffer ||
                    firstSlots[lhs].phases[0].visibleMeshletIds.buffer ==
                        firstSlots[rhs].phases[0].visibleMeshletIds.buffer ||
                    firstSlots[lhs].phases[1].buckets[0].indirectArguments.buffer ==
                        firstSlots[rhs].phases[1].buckets[0].indirectArguments.buffer) {
                    return RhiTestResult::fail(
                        "GPUScene frame-slot GPU resources alias each other");
                }
            }
        }
        if (firstSlots[0].instanceVisibilityStates.buffer ==
                secondSlot.instanceVisibilityStates.buffer ||
            firstSlots[0].hzbHistory[0].buffer == secondSlot.hzbHistory[0].buffer ||
            firstSlots[0].allocationId == secondSlot.allocationId) {
            return RhiTestResult::fail(
                "GPUScene resources alias between two independent Views");
        }

        const render::GPUSceneViewPrepareInfo prepareInfo{
            .width = viewDesc.hzbWidth,
            .height = viewDesc.hzbHeight,
        };
        for (uint32_t frameSlot = 0; frameSlot < firstSlots.size(); ++frameSlot) {
            if (!subsystem->prepareView(firstView, frameSlot, prepareInfo)) {
                return RhiTestResult::fail(
                    "GPUScene failed to prepare a GPU-backed View frame slot");
            }
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*queue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(
                "GPUScene View resource command-pool creation failed");
        }
        result = host.beginFrame(1, 0, nullptr, log);
        if (!result) {
            return RhiTestResult::fail(
                "GPUScene View resource beginFrame failed: " + log);
        }
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr || !commandBuffer->begin()) {
            host.endFrame();
            return RhiTestResult::fail(
                "GPUScene View resource command-buffer begin failed");
        }

        for (uint32_t frameSlot = 0; frameSlot < firstSlots.size(); ++frameSlot) {
            log.clear();
            result = subsystem->recordInitialize(
                *commandBuffer,
                firstView,
                frameSlot,
                log);
            if (!result) {
                host.endFrame();
                return RhiTestResult::fail(
                    "GPUScene View resource initialization failed: " + log);
            }
            // The second call must be a state-tracked no-op, not another
            // Undefined transition.
            result = subsystem->recordInitialize(
                *commandBuffer,
                firstView,
                frameSlot,
                log);
            render::GPUSceneViewGpuResourcesView initialized;
            if (!result || !subsystem->viewGpuResources(
                    firstView,
                    frameSlot,
                    initialized) ||
                !initialized.frameSlotInitialized ||
                !initialized.hzbInitialized) {
                host.endFrame();
                return RhiTestResult::fail(
                    "GPUScene did not track initialized slot/HZB resources");
            }
            result = subsystem->publishViewGpuResources(
                firstView,
                frameSlot,
                frameSlot & 1u,
                log);
            const render::GPUSceneVisibleDrawSet* visible =
                subsystem->visibleDrawSet(firstView, frameSlot);
            if (!result || visible == nullptr ||
                !visible->gpu.validFor(
                    subsystem->drawSet().generation,
                    subsystem->drawSet().revision) ||
                visible->gpu.visibleInstanceIds.buffer !=
                    initialized.visibleInstanceIds.buffer ||
                visible->gpu.visibleInstanceCounter.buffer !=
                    initialized.visibleInstanceCounter.buffer ||
                visible->gpu.hzb.writeIndex != (frameSlot & 1u)) {
                host.endFrame();
                return RhiTestResult::fail(
                    "GPUScene failed to publish its owned View GPU resources: " + log);
            }
        }

        render::Buffer* retiredInstance =
            firstSlots[0].instanceVisibilityStates.buffer;
        render::Buffer* retiredHzb = firstSlots[0].hzbHistory[0].buffer;
        const uint64_t retiredAllocationId = firstSlots[0].allocationId;
        render::GPUSceneViewDesc grownDesc = viewDesc;
        grownDesc.instanceCapacity = 16;
        grownDesc.visibleMeshletCapacity[2] = 25;
        grownDesc.hzbWidth = 16;
        grownDesc.hzbHeight = 8;
        grownDesc.hzbMipCount = 5;
        grownDesc.hzbElementCount = 171;
        result = subsystem->ensureViewGpuResources(firstView, grownDesc, log);
        render::GPUSceneViewGpuResourcesView grown;
        const render::GPUSceneVisibleDrawSet* invalidatedVisible =
            subsystem->visibleDrawSet(firstView, 0);
        if (!result || !subsystem->viewGpuResources(firstView, 0, grown) ||
            grown.allocationId == retiredAllocationId ||
            grown.instanceVisibilityStates.buffer == retiredInstance ||
            grown.hzbHistory[0].buffer == retiredHzb ||
            grown.instanceVisibilityStates.size != 16u * sizeof(uint32_t) ||
            grown.phases[0].buckets[2].visibleMeshletCapacity != 25 ||
            grown.hzbHistory[0].size != 171u * sizeof(float) ||
            invalidatedVisible == nullptr || invalidatedVisible->gpu.sourceView.valid() ||
            invalidatedVisible->stats.hzbValid) {
            host.endFrame();
            return RhiTestResult::fail(
                "GPUScene View resize/growth did not replace and invalidate the stale bundle: " +
                log);
        }
        // These non-owning pointers remain live because ensure retired the old
        // shared allocation into the active host frame slot.
        if (retiredInstance->desc().size != 8u * sizeof(uint32_t) ||
            retiredHzb->desc().size != 43u * sizeof(float)) {
            host.endFrame();
            return RhiTestResult::fail(
                "GPUScene destroyed a resized View bundle before deferred retirement");
        }

        result = subsystem->recordInitialize(*commandBuffer, firstView, 0, log);
        if (!result || !subsystem->prepareView(
                firstView,
                0,
                render::GPUSceneViewPrepareInfo{
                    .width = grownDesc.hzbWidth,
                    .height = grownDesc.hzbHeight,
                }) ||
            !subsystem->publishViewGpuResources(firstView, 0, 1, log)) {
            host.endFrame();
            return RhiTestResult::fail(
                "GPUScene failed to initialize/publish resized View resources: " + log);
        }

        render::Buffer* destroyedInstance = grown.instanceVisibilityStates.buffer;
        const uint64_t destroyedInstanceSize = destroyedInstance->desc().size;
        if (!subsystem->destroyView(firstView) ||
            !subsystem->destroyView(secondView) ||
            subsystem->viewGpuResources(firstView, 0, grown) ||
            subsystem->visibleDrawSet(firstView, 0) != nullptr ||
            destroyedInstance->desc().size != destroyedInstanceSize) {
            host.endFrame();
            return RhiTestResult::fail(
                "GPUScene destroyView did not invalidate and defer-retire its GPU bundle");
        }

        if (!commandBuffer->end()) {
            host.endFrame();
            return RhiTestResult::fail(
                "GPUScene View resource command-buffer end failed");
        }
        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            host.endFrame();
            return RhiTestResult::fail(
                "GPUScene View resource fence creation failed");
        }
        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = queue->submit(render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = fence.get(),
        });
        if (result) {
            result = fence->wait(5'000'000'000ull);
        }
        host.endFrame();
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUScene View resource submit/wait returned ") +
                toString(result));
        }

        host.shutdown();
        result = device->waitIdle();
        return result
            ? RhiTestResult::pass()
            : RhiTestResult::fail(
                std::string("GPUScene View resource waitIdle returned ") +
                toString(result));
    }
};

METALLIC_REGISTER_RHI_TEST(GPUSceneViewGpuResourcesTest);

} // namespace
} // namespace metallic::tests
