#include "Runtime/Render/Debug/RenderDebug.h"

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/MeshletStreamRuntime.h"

namespace metallic::render {
using debug::DebugFieldDesc;
using debug::DebugTypeDesc;
using debug::DebugValue;

std::unordered_map<std::string, DebugTypeDesc> renderDebugLayouts()
{
    std::unordered_map<std::string, DebugTypeDesc> result;
    auto add = [&](DebugTypeDesc type) { result.emplace(type.name, std::move(type)); };
    add({"u32", 4, {{"value", "u32", 0}}});
    add({"i32", 4, {{"value", "i32", 0}}});
    add({"u64", 8, {{"value", "u64", 0}}});
    add({"f32", 4, {{"value", "f32", 0}}});
    add({"RGBA8", 4, {{"r", "u8", 0}, {"g", "u8", 1}, {"b", "u8", 2}, {"a", "u8", 3}}});
    add({"BGRA8", 4, {{"b", "u8", 0}, {"g", "u8", 1}, {"r", "u8", 2}, {"a", "u8", 3}}});
    add({"RGBA16F", 8, {{"r", "f16", 0}, {"g", "f16", 2}, {"b", "f16", 4}, {"a", "f16", 6}}});
    add({"RGBA32F", 16, {{"r", "f32", 0}, {"g", "f32", 4}, {"b", "f32", 8}, {"a", "f32", 12}}});
#define DEBUG_FIELD(T, F) DebugFieldDesc{#F, "u32", static_cast<uint32_t>(offsetof(T, F))}
    static_assert(sizeof(VisibleClusterRecord) == 16 && offsetof(VisibleClusterRecord, flags) == 12);
    add({"VisibleClusterRecord", sizeof(VisibleClusterRecord), {
        DEBUG_FIELD(VisibleClusterRecord, clusterIndex), DEBUG_FIELD(VisibleClusterRecord, instanceIndex),
        DEBUG_FIELD(VisibleClusterRecord, dataIndex), DEBUG_FIELD(VisibleClusterRecord, flags),
        {"source", "u32", offsetof(VisibleClusterRecord, flags), 1, 28, 2, 1, {{"0", "Resident"}, {"1", "StreamPage"}}},
        {"drawBucket", "u32", offsetof(VisibleClusterRecord, flags), 1, 0, 4}}});
    static_assert(sizeof(StreamRequestBufferHeader) == 64);
    add({"StreamRequestBufferHeader", sizeof(StreamRequestBufferHeader), {
        DEBUG_FIELD(StreamRequestBufferHeader, maxLoadRequests), DEBUG_FIELD(StreamRequestBufferHeader, maxUnloadRequests),
        DEBUG_FIELD(StreamRequestBufferHeader, loadCounter), DEBUG_FIELD(StreamRequestBufferHeader, unloadCounter),
        DEBUG_FIELD(StreamRequestBufferHeader, frameIndex), DEBUG_FIELD(StreamRequestBufferHeader, loadOverflowCounter),
        DEBUG_FIELD(StreamRequestBufferHeader, unloadOverflowCounter), DEBUG_FIELD(StreamRequestBufferHeader, invalidPageCounter),
        DEBUG_FIELD(StreamRequestBufferHeader, lastLoadOverflowFrame), DEBUG_FIELD(StreamRequestBufferHeader, lastUnloadOverflowFrame),
        DEBUG_FIELD(StreamRequestBufferHeader, lastInvalidPageFrame)}});
    static_assert(sizeof(StreamPageTableEntry) == 8);
    add({"StreamPageTableEntry", sizeof(StreamPageTableEntry), {
        DEBUG_FIELD(StreamPageTableEntry, deviceOffsetAndState), DEBUG_FIELD(StreamPageTableEntry, lastRequestFrame),
        {"state", "u32", offsetof(StreamPageTableEntry, deviceOffsetAndState), 1, 0, 3, 1,
            {{"0", "Unloaded"}, {"1", "PendingUpload"}, {"2", "Resident"}, {"3", "LockedFallback"}, {"4", "PendingUnload"}}},
        {"deviceOffset", "u32", offsetof(StreamPageTableEntry, deviceOffsetAndState), 1, 3, 29, 8}}});
    static_assert(sizeof(MeshletStreamGpuActiveHeader) == 32);
    add({"MeshletStreamGpuActiveHeader", sizeof(MeshletStreamGpuActiveHeader), {
        DEBUG_FIELD(MeshletStreamGpuActiveHeader, activeGroupCount), DEBUG_FIELD(MeshletStreamGpuActiveHeader, activeGroupCapacity),
        DEBUG_FIELD(MeshletStreamGpuActiveHeader, maxActiveGroupClusters), DEBUG_FIELD(MeshletStreamGpuActiveHeader, overflowCount),
        DEBUG_FIELD(MeshletStreamGpuActiveHeader, frameIndex)}});
    static_assert(sizeof(MeshletStreamGpuActiveGroup) == 112 && offsetof(MeshletStreamGpuActiveGroup, world0) == 48);
    add({"MeshletStreamGpuActiveGroup", sizeof(MeshletStreamGpuActiveGroup), {
        DEBUG_FIELD(MeshletStreamGpuActiveGroup, pageDeviceOffsetBytes), DEBUG_FIELD(MeshletStreamGpuActiveGroup, pageIndex),
        DEBUG_FIELD(MeshletStreamGpuActiveGroup, clusterCount), DEBUG_FIELD(MeshletStreamGpuActiveGroup, primitiveIndex),
        DEBUG_FIELD(MeshletStreamGpuActiveGroup, lodLevel), DEBUG_FIELD(MeshletStreamGpuActiveGroup, materialIndex),
        DEBUG_FIELD(MeshletStreamGpuActiveGroup, clusterSelectionMask), DEBUG_FIELD(MeshletStreamGpuActiveGroup, flags),
        DEBUG_FIELD(MeshletStreamGpuActiveGroup, instanceIndex), DEBUG_FIELD(MeshletStreamGpuActiveGroup, gpuSceneInstanceIndex),
        {"world", "f32", offsetof(MeshletStreamGpuActiveGroup, world0), 16}}});
    static_assert(sizeof(GPUSceneGpuGeometryRecord) == 96 && offsetof(GPUSceneGpuGeometryRecord, payload) == 32);
    add({"GPUSceneGpuGeometryRecord", sizeof(GPUSceneGpuGeometryRecord), {
        {"source", "u32", offsetof(GPUSceneGpuGeometryRecord, source), 4},
        {"counts", "u32", offsetof(GPUSceneGpuGeometryRecord, counts), 4},
        {"payload", "u32", offsetof(GPUSceneGpuGeometryRecord, payload), 4},
        {"meshletPayload", "u32", offsetof(GPUSceneGpuGeometryRecord, meshletPayload), 4},
        {"bounds", "f32", offsetof(GPUSceneGpuGeometryRecord, localBoundingSphere), 4},
        {"identity", "u32", offsetof(GPUSceneGpuGeometryRecord, identity), 4}}});
    add({"GPUSceneGpuInstanceRecord", sizeof(GPUSceneGpuInstanceRecord), {
        {"world", "f32", offsetof(GPUSceneGpuInstanceRecord, worldMatrix), 16},
        {"previousWorld", "f32", offsetof(GPUSceneGpuInstanceRecord, previousWorldMatrix), 16},
        {"bounds", "f32", offsetof(GPUSceneGpuInstanceRecord, localBoundingSphere), 4},
        {"identity", "u32", offsetof(GPUSceneGpuInstanceRecord, identity), 4}}});
    add({"GPUSceneGpuMeshletRecord", sizeof(GPUSceneGpuMeshletRecord), {
        {"ranges", "u32", offsetof(GPUSceneGpuMeshletRecord, ranges), 4},
        {"lod", "u32", offsetof(GPUSceneGpuMeshletRecord, lod), 4},
        {"bounds", "f32", offsetof(GPUSceneGpuMeshletRecord, boundingSphere), 4}}});
#undef DEBUG_FIELD
    return result;
}

void gpuDrivenDebugCheckpoint(RenderGraphExecutionContext& context, std::string_view checkpoint,
    GPUSceneSubsystem* gpuScene, GPUSceneViewId view, uint32_t frameSlot, MeshletStreamRuntime* streaming, uint32_t phase, uint64_t streamVisibleRecordBase)
{
    if (!context.debugEnabled()) { return; }
    std::vector<DebugResourceBinding> bindings;
    DebugValue values = DebugValue::object();
    if (gpuScene) {
        const auto& globals = gpuScene->globalBufferViews();
        const auto global = [&](std::string name, const GPUSceneBufferView& buffer, std::string layout) {
            if (!buffer.validFor(globals.drawSetGeneration, globals.drawSetRevision)) { return; }
            bindings.push_back({.id = "gpuScene." + context.passName() + "." + name,
                .buffer = buffer.buffer, .state = ResourceState::ShaderRead, .offset = buffer.offset, .size = buffer.size,
                .layout = std::move(layout), .allocation = buffer.generation,
                .metadata = {{"drawSetGeneration", buffer.generation}, {"drawSetRevision", buffer.revision}, {"owner", "GPUScene global upload"}}});
        };
        global("instances", globals.instances, "GPUSceneGpuInstanceRecord");
        global("geometries", globals.geometries, "GPUSceneGpuGeometryRecord");
        global("meshlets", globals.meshlets, "GPUSceneGpuMeshletRecord");
        global("meshletDraws", globals.meshletDraws, "VisibleClusterRecord");
        global("drawInstanceIds", globals.drawInstanceIds, "u32");
        GPUSceneViewGpuResourcesView resources;
        if (phase < kGPUSceneCullPhaseCount && gpuScene->viewGpuResources(view, frameSlot, resources) && resources.frameSlotInitialized) {
            const auto add = [&](std::string name, const GPUSceneBufferView& buffer, std::string layout = "u32") {
                if (!buffer.buffer) { return; }
                bindings.push_back({.id = "gpuScene." + context.passName() + "." + name,
                    .buffer = buffer.buffer, .state = ResourceState::General, .offset = buffer.offset, .size = buffer.size,
                    .layout = std::move(layout), .allocation = resources.allocationId,
                    .metadata = {{"view", view.index}, {"viewGeneration", view.generation}, {"frameSlot", frameSlot},
                        {"drawSetGeneration", buffer.generation}, {"drawSetRevision", buffer.revision}, {"phase", phase},
                        {"validity", "Capacity is not live count; counter/list describe the last instance-cull dispatch at this boundary"}}});
            };
            add("instanceVisibility", resources.instanceVisibilityStates);
            add("visibleInstanceIds", resources.visibleInstanceIds);
            add("visibleInstanceCounter", resources.visibleInstanceCounter);
            const size_t unproducedBegin = bindings.size();
            add("visibleMeshletIds", resources.phases[phase].visibleMeshletIds);
            // Each bucket publishes explicit packed worklist and indirect offsets.
            DebugValue buckets = DebugValue::array();
            for (size_t i = 0; i < resources.phases[phase].buckets.size(); ++i) {
                const auto& bucket = resources.phases[phase].buckets[i];
                add("bucket" + std::to_string(i) + ".indirect", bucket.indirectArguments);
                add("bucket" + std::to_string(i) + ".overflow", bucket.overflow);
                buckets.push_back({{"index", i}, {"offset", bucket.visibleMeshletOffset}, {"capacity", bucket.visibleMeshletCapacity}});
            }
            // These two passes use recordInstanceCull, not recordCull. The view
            // allocates meshlet worklists but these paths do not populate them.
            for (size_t i = unproducedBegin; i < bindings.size(); ++i) {
                bindings[i].metadata["captureSupported"] = false;
                bindings[i].metadata["reason"] = "This pass uses instance culling; meshlet worklists and bucket indirect arguments are not produced";
            }
            values["gpuSceneView"] = {{"id", view.index}, {"generation", view.generation}, {"allocation", resources.allocationId},
                {"frameSlot", frameSlot}, {"phase", phase}, {"buckets", buckets}};
        }
    }
    if (streaming && streaming->ready()) {
        streaming->appendDebugBindings(bindings, "streaming." + context.passName() + ".");
        for (auto& binding : bindings) {
            if (binding.layout == "VisibleClusterRecord" && binding.id.starts_with("streaming.")) {
                binding.metadata["captureSupported"] = checkpoint == "AfterPass" && phase < kGPUSceneCullPhaseCount;
                binding.metadata["visibleRecordBase"] = streamVisibleRecordBase;
                binding.metadata["reason"] = "VisibleClusterRecord is written during drawing; capture at AfterPass and validate sparse slots using visibility pixels";
                binding.metadata["validity"] = "Sparse storage; capacity is not a live record count";
            }
        }
        values["streaming"] = {{"instances", DebugValue::array({streaming->debugSnapshot()})}};
        values["streaming"]["instances"][0]["pass"] = context.passName();
    }
    context.debugCheckpoint(checkpoint, bindings, values);
}

} // namespace metallic::render
