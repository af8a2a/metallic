#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/ResidentMeshletLod.h"
#include "Runtime/Render/SlangCompiler.h"
#include <algorithm>

namespace metallic::render {
namespace {

struct LodPush {
    // Camera data starts at push byte zero, matching Slang float4 alignment.
    MeshletLodView view;
    uint32_t clusters, records, instances, groups;
    uint32_t output, arguments, offset, count;
    uint32_t capacity, instanceCount, groupCount, manualLevel;
    uint32_t scratch, padding2;
};
static_assert(offsetof(LodPush, view) == 0);
static_assert(offsetof(LodPush, clusters) == 48);
static_assert(sizeof(LodPush) == 104);

} // namespace

Result<> ResidentMeshletLod::initialize(Device& device, uint32_t capacity, std::string& log)
{
    if (capacity == 0 || !visibilityRecordCapacityFitsId(capacity)) { return makeError(Error::InvalidArgument); }
    capacity_ = capacity;
    Result<> result = device.createBuffer({.size = (uint64_t(capacity) + 1u) * 16u, .structureStride = 16,
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { selections_ = std::move(rhiValue); });
    if (result) {
        result = device.createBuffer({.size = 36, .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::Indirect | BufferUsageBits::TransferSource,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { arguments_ = std::move(rhiValue); });
    }
    if (!result) { return result; }
    result = device.createBuffer({.size = (uint64_t(capacity) + (capacity + 63u) / 64u) * 4u,
        .structureStride = 4, .usage = BufferUsageBits::Storage}).transform([&](auto rhiValue) { scratch_ = std::move(rhiValue); });
    if (!result) { return result; }
    const char* entries[] = {"residentLodResetMain", "residentLodSelectMain", "residentLodArgumentsMain", "residentLodScatterMain"};
    for (size_t i = 0; i < shaders_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/GPUDriven/ResidentMeshletLod",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log += shader.diagnostics; return result; }
        result = device.createShaderModule({
            .spirv = shader.spirv,
            .debugName = entries[i],
        }).transform([&](auto rhiValue) { shaders_[i] = std::move(rhiValue); });
        if (result) {
            result = device.createComputePipeline({
                .computeShader = {shaders_[i].get()},
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(LodPush),
            }).transform([&](auto rhiValue) { pipelines_[i] = std::move(rhiValue); });
        }
        if (!result) { return result; }
    }
    return {};
}

Result<> ResidentMeshletLod::record(CommandBuffer& commands, ResourceRegistry& registry,
    const GPUSceneConsumerBindings& bindings, const MeshletLodView& view,
    GPUSceneRasterDrawRange candidates, uint32_t instanceCount, uint32_t groupCount,
    ResourceLease output, ResourceLease arguments, ResourceLease scratch, uint32_t manualLevel)
{
    // Provision for every input record. Selection can never overflow, even when
    // the camera crosses the near plane and requests the finest entire scene.
    if (candidates.count > capacity_) { return makeError(Error::InvalidArgument); }
    for (const auto& lease : bindings.buffers) {
        if (!lease.valid()) { continue; }
        auto result = registry.retain(commands, lease);
        if (!result) { return result; }
    }
    for (const auto& lease : {output, arguments, scratch}) {
        auto result = registry.retain(commands, lease);
        if (!result) { return result; }
    }
    auto result = registry.bind(commands);
    if (!result) { return result; }
    const LodPush push{view,
        bindings[GPUSceneGlobalBufferKind::Meshlets].shaderIndex(),
        bindings[GPUSceneGlobalBufferKind::MeshletDraws].shaderIndex(),
        bindings[GPUSceneGlobalBufferKind::Instances].shaderIndex(),
        bindings[GPUSceneGlobalBufferKind::LodGroups].shaderIndex(),
        output.shaderIndex(), arguments.shaderIndex(), candidates.offset, candidates.count,
        capacity_, instanceCount, groupCount, manualLevel, scratch.shaderIndex(), 0u};
    using namespace detail;
    using Access = RenderGraphResourceAccess;
    constexpr auto compute = RenderGraphPassKind::Compute;
    constexpr auto consumer = RenderGraphPassKind::Unsafe;
    Buffer* buffers[] = {selections_.get(), arguments_.get(), scratch_.get()};
    std::array<GraphAccessResource, 3> resources;
    std::array<GraphAccessBinding, 3> resourcesBound;
    for (size_t i = 0; i < resources.size(); ++i) {
        resources[i] = {.type = RenderGraphResourceType::Buffer,
            .state = initialized_ ? ResourceState::General : ResourceState::Undefined,
            .scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
        auto slice = buffers[i]->slice();
        if (!slice) { return makeError(slice.error()); }
        resourcesBound[i].buffer = std::move(*slice);
    }
    const GraphAccessPass phases[] = {
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(2, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute),
            declaredGraphAccess(1, Access::BufferStorageWrite, compute),
            declaredGraphAccess(2, Access::BufferStorageReadWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute),
            declaredGraphAccess(2, Access::BufferStorageRead, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferShaderRead, consumer),
            declaredGraphAccess(1, Access::BufferIndirectRead, consumer)}},
    };
    auto plan = buildGraphAccessPlan(resources, phases);
    if (!plan) { return makeError(plan.error()); }
    commands.beginDebugLabel({.name = "Resident adaptive meshlet LOD"});
    for (uint32_t stage = 0; stage < 4; ++stage) {
        result = recordGraphAccessBarriers(commands, plan->passes[stage], resourcesBound);
        if (!result) { return result; }
        if (auto commandResult = commands.bindExecution(pipelines_[stage]->execution()); !commandResult) { return commandResult; }
        commands.pushBindlessData(&push, sizeof(push));
        uint32_t groups = (stage == 1 || stage == 3) ? (candidates.count + 63u) / 64u : 1u;
        if (groups != 0) { commands.dispatch(std::min(groups, 65535u), (groups + 65534u) / 65535u); }
    }
    result = recordGraphAccessBarriers(commands, plan->passes.back(), resourcesBound);
    if (!result) { return result; }
    commands.endDebugLabel();
    initialized_ = true;
    return {};
}

} // namespace metallic::render
