#include "Runtime/Render/ResidentMeshletLod.h"
#include "Runtime/Render/SlangCompiler.h"
#include <algorithm>

namespace metallic::render {
namespace {

struct LodPush {
    // The RHI prepends two descriptor-base words. Align the following float4
    // camera fields to 16 bytes in the complete Vulkan push-constant block.
    uint32_t padding[2]{};
    MeshletLodView view;
    uint32_t clusters, records, instances, groups;
    uint32_t output, arguments, offset, count;
    uint32_t capacity, instanceCount, groupCount, manualLevel;
    uint32_t scratch, padding2;
};
static_assert(sizeof(LodPush) == 112);

} // namespace

Result ResidentMeshletLod::initialize(Device& device, uint32_t capacity, std::string& log)
{
    if (capacity == 0 || !visibilityRecordCapacityFitsId(capacity)) { return makeError(Error::InvalidArgument); }
    capacity_ = capacity;
    Result result = device.createBuffer({.size = (uint64_t(capacity) + 1u) * 16u, .structureStride = 16,
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}, selections_);
    if (result) {
        result = device.createBuffer({.size = 24, .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::Indirect | BufferUsageBits::TransferSource,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}, arguments_);
    }
    if (!result) { return result; }
    result = device.createBuffer({.size = (uint64_t(capacity) + (capacity + 63u) / 64u) * 4u,
        .structureStride = 4, .usage = BufferUsageBits::Storage}, scratch_);
    if (!result) { return result; }
    const char* entries[] = {"residentLodResetMain", "residentLodSelectMain", "residentLodArgumentsMain", "residentLodScatterMain"};
    for (size_t i = 0; i < shaders_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/GPUDriven/ResidentMeshletLod",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
        if (!result) { log += shader.diagnostics; return result; }
        result = device.createShaderModule({.code = shader.spirv.data(), .byteSize = shader.spirv.size() * 4u,
            .debugName = entries[i]}, shaders_[i]);
        if (result) {
            result = device.createComputePipeline({.computeShader = shaders_[i].get(),
                .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(LodPush)}, pipelines_[i]);
        }
        if (!result) { return result; }
    }
    return {};
}

Result ResidentMeshletLod::record(CommandBuffer& commands, BindlessHeap& heap,
    const GPUSceneConsumerBindings& bindings, const MeshletLodView& view,
    GPUSceneRasterDrawRange candidates, uint32_t instanceCount, uint32_t groupCount,
    BindlessHandle output, BindlessHandle arguments, BindlessHandle scratch, uint32_t manualLevel)
{
    // Provision for every input record. Selection can never overflow, even when
    // the camera crosses the near plane and requests the finest entire scene.
    if (candidates.count > capacity_) { return makeError(Error::InvalidArgument); }
    const LodPush push{{}, view,
        bindings[GPUSceneGlobalBufferKind::Meshlets].index,
        bindings[GPUSceneGlobalBufferKind::MeshletDraws].index,
        bindings[GPUSceneGlobalBufferKind::Instances].index,
        bindings[GPUSceneGlobalBufferKind::LodGroups].index,
        output.index, arguments.index, candidates.offset, candidates.count,
        capacity_, instanceCount, groupCount, manualLevel, scratch.index, 0u};
    commands.beginDebugLabel({.name = "Resident adaptive meshlet LOD"});
    BufferBarrierDesc barriers[] = {
        {.buffer = selections_.get(), .before = initialized_ ? ResourceState::ShaderRead : ResourceState::Undefined,
            .after = ResourceState::General},
        {.buffer = arguments_.get(), .before = initialized_ ? ResourceState::IndirectArgument : ResourceState::Undefined,
            .after = ResourceState::General},
        {.buffer = scratch_.get(), .before = initialized_ ? ResourceState::General : ResourceState::Undefined,
            .after = ResourceState::General}};
    commands.barrier({.buffers = barriers, .bufferCount = 3});
    commands.bindBindlessHeap(heap);
    for (uint32_t stage = 0; stage < 4; ++stage) {
        commands.bindComputePipeline(*pipelines_[stage]);
        commands.pushBindlessData(&push, sizeof(push));
        uint32_t groups = (stage == 1 || stage == 3) ? (candidates.count + 63u) / 64u : 1u;
        if (groups != 0) { commands.dispatch(std::min(groups, 65535u), (groups + 65534u) / 65535u); }
        barriers[0].before = ResourceState::General;
        commands.barrier({.buffers = barriers, .bufferCount = 1});
        barriers[2].before = ResourceState::General;
        commands.barrier({.buffers = &barriers[2], .bufferCount = 1});
    }
    barriers[0].after = ResourceState::ShaderRead;
    barriers[1].before = ResourceState::General;
    barriers[1].after = ResourceState::IndirectArgument;
    commands.barrier({.buffers = barriers, .bufferCount = 2});
    commands.endDebugLabel();
    initialized_ = true;
    return {};
}

} // namespace metallic::render
