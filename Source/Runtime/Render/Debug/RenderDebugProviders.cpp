#include "Runtime/Render/Debug/RenderDebug.h"

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

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
    add({"MeshletLODSelectionHeader", 16, {{"count", "u32", 0}, {"capacity", "u32", 4},
        {"candidateCount", "u32", 8}, {"overflow", "u32", 12}}});
    add({"MeshletLODSelection", sizeof(MeshletLODSelection), {
        DEBUG_FIELD(MeshletLODSelection, instanceIndex), DEBUG_FIELD(MeshletLODSelection, clusterIndex),
        DEBUG_FIELD(MeshletLODSelection, recordIndex), DEBUG_FIELD(MeshletLODSelection, geometryIndex)}});
    add({"MeshletLODGroupRecord", sizeof(MeshletLODGroupRecord), {
        {"sphere", "f32", offsetof(MeshletLODGroupRecord, sphere), 4},
        {"error", "f32", offsetof(MeshletLODGroupRecord, error)},
        DEBUG_FIELD(MeshletLODGroupRecord, level), DEBUG_FIELD(MeshletLODGroupRecord, flags)}});
    add({"CompactStreamVisibleRecord", sizeof(CompactStreamVisibleRecord), {
        {"packed", "u32", 0}, {"clusterIndex", "u32", 0, 1, 0, 5},
        {"groupIndexPlusOne", "u32", 0, 1, 5, 25}, {"rasterFlags", "u32", 0, 1, 30, 2}}});
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
        DEBUG_FIELD(StreamRequestBufferHeader, lastInvalidPageFrame),
        DEBUG_FIELD(StreamRequestBufferHeader, loadPriorityOffset), DEBUG_FIELD(StreamRequestBufferHeader, priorityTableOffset),
        DEBUG_FIELD(StreamRequestBufferHeader, prefetchRequestLimit), DEBUG_FIELD(StreamRequestBufferHeader, prefetchRequestCounter),
        DEBUG_FIELD(StreamRequestBufferHeader, prefetchDroppedCounter)}});
    static_assert(sizeof(StreamPageTableEntry) == 8);
    add({"StreamPageTableEntry", sizeof(StreamPageTableEntry), {
        DEBUG_FIELD(StreamPageTableEntry, deviceOffsetAndState), DEBUG_FIELD(StreamPageTableEntry, lastRequestFrame),
        {"state", "u32", offsetof(StreamPageTableEntry, deviceOffsetAndState), 1, 0, 3, 1,
            {{"0", "Unloaded"}, {"1", "PendingUpload"}, {"2", "Resident"}, {"3", "LockedFallback"}, {"4", "PendingUnload"}}},
        {"deviceOffset", "u32", offsetof(StreamPageTableEntry, deviceOffsetAndState), 1, 3, 29, 8}}});
    static_assert(sizeof(MeshletStreamGPUActiveHeader) == 32);
    add({"MeshletStreamGPUActiveHeader", sizeof(MeshletStreamGPUActiveHeader), {
        DEBUG_FIELD(MeshletStreamGPUActiveHeader, activeGroupCount), DEBUG_FIELD(MeshletStreamGPUActiveHeader, activeGroupCapacity),
        DEBUG_FIELD(MeshletStreamGPUActiveHeader, maxActiveGroupClusters), DEBUG_FIELD(MeshletStreamGPUActiveHeader, overflowCount),
        DEBUG_FIELD(MeshletStreamGPUActiveHeader, frameIndex),
        {"terminalFallback", "u32", offsetof(MeshletStreamGPUActiveHeader, padding0)},
        {"invalidCapacity", "u32", offsetof(MeshletStreamGPUActiveHeader, padding1)}}});
    static_assert(sizeof(MeshletStreamGPUActiveGroup) == 112 && offsetof(MeshletStreamGPUActiveGroup, world0) == 48);
    add({"MeshletStreamGPUActiveGroup", sizeof(MeshletStreamGPUActiveGroup), {
        DEBUG_FIELD(MeshletStreamGPUActiveGroup, pageDeviceOffsetBytes), DEBUG_FIELD(MeshletStreamGPUActiveGroup, pageIndex),
        DEBUG_FIELD(MeshletStreamGPUActiveGroup, clusterCount), DEBUG_FIELD(MeshletStreamGPUActiveGroup, primitiveIndex),
        DEBUG_FIELD(MeshletStreamGPUActiveGroup, lodLevel), DEBUG_FIELD(MeshletStreamGPUActiveGroup, materialIndex),
        DEBUG_FIELD(MeshletStreamGPUActiveGroup, clusterSelectionMask), DEBUG_FIELD(MeshletStreamGPUActiveGroup, flags),
        DEBUG_FIELD(MeshletStreamGPUActiveGroup, instanceIndex), DEBUG_FIELD(MeshletStreamGPUActiveGroup, gpuSceneInstanceIndex),
        {"world", "f32", offsetof(MeshletStreamGPUActiveGroup, world0), 16}}});
    static_assert(sizeof(GPUSceneGPUGeometryRecord) == 96 && offsetof(GPUSceneGPUGeometryRecord, payload) == 32);
    add({"GPUSceneGPUGeometryRecord", sizeof(GPUSceneGPUGeometryRecord), {
        {"source", "u32", offsetof(GPUSceneGPUGeometryRecord, source), 4},
        {"counts", "u32", offsetof(GPUSceneGPUGeometryRecord, counts), 4},
        {"payload", "u32", offsetof(GPUSceneGPUGeometryRecord, payload), 4},
        {"meshletPayload", "u32", offsetof(GPUSceneGPUGeometryRecord, meshletPayload), 4},
        {"bounds", "f32", offsetof(GPUSceneGPUGeometryRecord, localBoundingSphere), 4},
        {"identity", "u32", offsetof(GPUSceneGPUGeometryRecord, identity), 4}}});
    add({"GPUSceneGPUInstanceRecord", sizeof(GPUSceneGPUInstanceRecord), {
        {"world", "f32", offsetof(GPUSceneGPUInstanceRecord, worldMatrix), 16},
        {"previousWorld", "f32", offsetof(GPUSceneGPUInstanceRecord, previousWorldMatrix), 16},
        {"bounds", "f32", offsetof(GPUSceneGPUInstanceRecord, localBoundingSphere), 4},
        {"identity", "u32", offsetof(GPUSceneGPUInstanceRecord, identity), 4}}});
    add({"GPUSceneGPUMeshletRecord", sizeof(GPUSceneGPUMeshletRecord), {
        {"ranges", "u32", offsetof(GPUSceneGPUMeshletRecord, ranges), 4},
        {"lod", "u32", offsetof(GPUSceneGPUMeshletRecord, lod), 4},
        {"bounds", "f32", offsetof(GPUSceneGPUMeshletRecord, boundingSphere), 4}}});
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
        global("instances", globals.instances, "GPUSceneGPUInstanceRecord");
        global("geometries", globals.geometries, "GPUSceneGPUGeometryRecord");
        global("meshlets", globals.meshlets, "GPUSceneGPUMeshletRecord");
        global("lodGroups", globals.lodGroups, "MeshletLODGroupRecord");
        global("meshletDraws", globals.meshletDraws, "VisibleClusterRecord");
        global("drawInstanceIds", globals.drawInstanceIds, "u32");
        GPUSceneViewGPUResourcesView resources;
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
            if (binding.layout == "CompactStreamVisibleRecord" && binding.id.starts_with("streaming.")) {
                binding.metadata["captureSupported"] = checkpoint == "AfterPass" && phase < kGPUSceneCullPhaseCount;
                binding.metadata["visibleRecordBase"] = streamVisibleRecordBase;
                binding.metadata["reason"] = "CompactStreamVisibleRecord is written during drawing; capture at AfterPass and validate sparse slots using visibility pixels";
                binding.metadata["validity"] = "Sparse storage; capacity is not a live record count";
            }
        }
        values["streaming"] = {{"instances", DebugValue::array({streaming->debugSnapshot(
            context.properties().value("debugStreamingPages", true))})}};
        values["streaming"]["instances"][0]["pass"] = context.passName();
    }
    context.debugCheckpoint(checkpoint, bindings, values);
}

} // namespace metallic::render
