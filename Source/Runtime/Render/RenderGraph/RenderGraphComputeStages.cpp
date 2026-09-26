#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"

#include <algorithm>
#include <unordered_map>

namespace metallic::render {
namespace {

bool scopeContains(SyncScope allowed, SyncScope requested)
{
    const auto stages = static_cast<uint64_t>(allowed.stages);
    const auto requiredStages = static_cast<uint64_t>(requested.stages);
    const auto access = static_cast<uint64_t>(allowed.access);
    const auto requiredAccess = static_cast<uint64_t>(requested.access);
    return ((stages & static_cast<uint64_t>(PipelineStageBits::AllCommands)) != 0 ||
        (stages & requiredStages) == requiredStages) && (access & requiredAccess) == requiredAccess;
}

void mergeScope(SyncScope& destination, SyncScope source)
{
    destination.stages = destination.stages | source.stages;
    destination.access = destination.access | source.access;
}

bool validTextureState(const Texture& texture, ResourceState state)
{
    TextureUsageBits required;
    switch (state) {
    case ResourceState::Undefined:
    case ResourceState::General:
        return true;
    case ResourceState::ShaderRead: required = TextureUsageBits::Sampled; break;
    case ResourceState::ColorAttachment: required = TextureUsageBits::ColorAttachment; break;
    case ResourceState::DepthStencilAttachment: required = TextureUsageBits::DepthStencilAttachment; break;
    case ResourceState::TransferSource: required = TextureUsageBits::TransferSource; break;
    case ResourceState::TransferDestination: required = TextureUsageBits::TransferDestination; break;
    default: return false;
    }
    return (uint64_t(texture.desc().usage) & uint64_t(required)) == uint64_t(required);
}

} // namespace

Result<> RenderGraphExecutionContext::executeComputeStages(
    std::span<const RenderGraphComputeStage> stages, std::span<const RenderGraphBufferImport> imports)
{
    return executeStagesImpl(stages, imports, {}, true);
}

Result<> RenderGraphExecutionContext::executeStages(std::span<const RenderGraphStage> stages,
    std::span<const RenderGraphBufferImport> buffers, std::span<const RenderGraphTextureImport> textures)
{
    return executeStagesImpl(stages, buffers, textures, false);
}

Result<> RenderGraphExecutionContext::executeStagesImpl(std::span<const RenderGraphStage> stages,
    std::span<const RenderGraphBufferImport> imports, std::span<const RenderGraphTextureImport> textures,
    bool computeOnly)
{
    using namespace detail;
    if (computeStagesExecuted_ || stages.empty() || !commandBuffer().recording()) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<GraphAccessResource> resources;
    std::vector<GraphAccessBinding> resolved;
    std::vector<SyncScope> permissions;
    std::vector<bool> internalLayouts;
    std::vector<ResourceState> finalStates;
    std::vector<std::shared_ptr<void>> textureOwners;
    std::unordered_map<std::string, size_t> names;
    std::unordered_map<const void*, size_t> bufferIdentities;
    std::unordered_map<const void*, size_t> textureIdentities;

    // Bindings may be per-worker snapshots. Normalize by retained allocation,
    // not the address of the snapshot, view descriptor or public Buffer wrapper.
    for (const auto& binding : bindings_) {
        if (!binding.resource) { continue; }
        const auto& resource = *binding.resource;
        auto bound = bindGraphAccessResource(resource);
        if (!bound) { return makeError(bound.error()); }
        std::shared_ptr<void> textureOwner;
        const void* identity = nullptr;
        if (resource.type == RenderGraphResourceType::Buffer) {
            identity = bound->buffer.allocationIdentity();
            if (bound->buffer.deviceIdentity() != commandBuffer().deviceIdentity()) {
                return makeError(Error::InvalidArgument);
            }
        } else {
            if (!resource.view || resource.view->deviceIdentity() != commandBuffer().deviceIdentity()) {
                return makeError(Error::InvalidArgument);
            }
            textureOwner = resource.view->retainTexture();
            identity = textureOwner.get();
            // Graph-owned images have a retained identity. Borrowed images need
            // an explicit import/completion contract and are outside this API.
            if (!identity) { return makeError(Error::InvalidArgument); }
        }
        auto& identities = resource.type == RenderGraphResourceType::Buffer ? bufferIdentities : textureIdentities;
        const auto [entry, inserted] = identities.try_emplace(identity, resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = resource.type, .state = resource.state, .boundarySynchronized = true});
            resolved.push_back(std::move(*bound));
            permissions.push_back(binding.scope);
            internalLayouts.push_back(binding.internalLayouts);
            finalStates.push_back(resource.state);
            textureOwners.push_back(std::move(textureOwner));
        } else {
            if (resources[index].state != resource.state) { return makeError(Error::InvalidArgument); }
            mergeScope(permissions[index], binding.scope);
            internalLayouts[index] = internalLayouts[index] || binding.internalLayouts;
        }
        const auto qualified = (binding.visibility == RenderGraphFieldVisibility::Input ? "input." : "output.") + binding.fieldName;
        if (!names.emplace(qualified, index).second) { return makeError(Error::InvalidArgument); }
        const auto [shortName, unique] = names.emplace(binding.fieldName, index);
        if (!unique) { shortName->second = SIZE_MAX; }
    }

    const size_t graphResourceCount = resources.size();
    for (const auto& imported : imports) {
        const auto scope = scopeForGraphAccess(imported.access, RenderGraphPassKind::Compute);
        const auto state = stateForAccess(imported.access);
        const auto requiredUsage = bufferUsageForAccess(imported.access);
        if (imported.name.empty() || imported.access == RenderGraphResourceAccess::None ||
            !imported.buffer.valid() || imported.buffer.size() == 0 ||
            imported.buffer.deviceIdentity() != commandBuffer().deviceIdentity() ||
            !accessMatchesResourceType(imported.access, RenderGraphResourceType::Buffer) ||
            (computeOnly && scope.stages != PipelineStageBits::ComputeShader) ||
            (static_cast<uint64_t>(imported.buffer.allocationDesc().usage) & static_cast<uint64_t>(requiredUsage)) !=
                static_cast<uint64_t>(requiredUsage)) {
            return makeError(Error::InvalidArgument);
        }
        const auto [entry, inserted] = bufferIdentities.try_emplace(imported.buffer.allocationIdentity(), resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = RenderGraphResourceType::Buffer, .state = state,
                .scope = computeOnly ? scope : SyncScope{PipelineStageBits::AllCommands,
                    AccessBits::MemoryRead | AccessBits::MemoryWrite}});
            resolved.push_back({.buffer = imported.buffer});
            permissions.push_back(scope);
            internalLayouts.push_back(false);
            finalStates.push_back(ResourceState::Undefined);
            textureOwners.push_back({});
        } else if (index < graphResourceCount || resources[index].state != state ||
            permissions[index].stages != scope.stages || permissions[index].access != scope.access) {
            // A private import cannot grant hidden writes to a reflected field.
            // Aliases of a private allocation must agree on its boundary contract.
            return makeError(Error::InvalidArgument);
        }
        if (!names.emplace(imported.name, index).second) { return makeError(Error::InvalidArgument); }
    }

    for (const auto& imported : textures) {
        if (computeOnly || imported.name.empty() || !imported.texture ||
            imported.texture->deviceIdentity() != commandBuffer().deviceIdentity() ||
            !validTextureState(*imported.texture, imported.initialState) ||
            !validTextureState(*imported.texture, imported.finalState)) {
            return makeError(Error::InvalidArgument);
        }
        auto owner = imported.texture->retainAllocation();
        if (!owner || (imported.view && imported.view->retainTexture() != owner)) {
            return makeError(Error::InvalidArgument);
        }
        const auto [entry, inserted] = textureIdentities.try_emplace(owner.get(), resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = RenderGraphResourceType::Texture2D, .state = imported.initialState,
                .scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}});
            resolved.push_back({.texture = imported.texture, .mipCount = imported.texture->desc().mipCount,
                .layerCount = imported.texture->desc().layerCount});
            permissions.push_back({});
            internalLayouts.push_back(true);
            finalStates.push_back(imported.finalState);
            textureOwners.push_back(std::move(owner));
        } else if (index < graphResourceCount || resources[index].state != imported.initialState ||
            finalStates[index] != imported.finalState) {
            return makeError(Error::InvalidArgument);
        }
        if (!names.emplace(imported.name, index).second) { return makeError(Error::InvalidArgument); }
    }

    std::vector<GraphAccessPass> accesses;
    accesses.reserve(stages.size());
    std::vector<bool> used(resources.size(), false);
    std::vector<ResourceState> lastStates;
    lastStates.reserve(resources.size());
    for (const auto& resource : resources) { lastStates.push_back(resource.state); }
    for (const auto& stage : stages) {
        if (stage.name.empty() || !stage.record || (computeOnly && stage.kind != RenderGraphPassKind::Compute) ||
            (stage.allowParallelCompute && (computeOnly || stage.kind != RenderGraphPassKind::Unsafe))) {
            return makeError(Error::InvalidArgument);
        }
        auto& access = accesses.emplace_back();
        for (const auto& use : stage.uses) {
            const auto found = names.find(std::string(use.resource));
            if (found == names.end() || found->second == SIZE_MAX) { return makeError(Error::InvalidArgument); }
            const size_t index = found->second;
            const auto scope = scopeForGraphAccess(use.access, stage.kind);
            const auto state = stateForAccess(use.access);
            const bool graph = index < graphResourceCount;
            if (use.access == RenderGraphResourceAccess::None || !accessMatchesResourceType(use.access, resources[index].type) ||
                (computeOnly && scope.stages != PipelineStageBits::ComputeShader) ||
                ((graph || computeOnly) && !scopeContains(permissions[index], scope)) ||
                (resources[index].type == RenderGraphResourceType::Texture2D && resources[index].state != state &&
                    (computeOnly || !internalLayouts[index]))) {
                return makeError(Error::InvalidArgument);
            }
            if (resources[index].type == RenderGraphResourceType::Buffer) {
                const auto required = static_cast<uint64_t>(bufferUsageForAccess(use.access));
                if ((static_cast<uint64_t>(resolved[index].buffer.allocationDesc().usage) & required) != required) {
                    return makeError(Error::InvalidArgument);
                }
            } else {
                const auto required = static_cast<uint64_t>(textureUsageForAccess(use.access));
                if ((static_cast<uint64_t>(resolved[index].texture->desc().usage) & required) != required) {
                    return makeError(Error::InvalidArgument);
                }
            }
            used[index] = true;
            lastStates[index] = state;
            access.uses.push_back({index, state, scope, accessWrites(use.access)});
        }
    }
    if (!computeOnly) {
        auto& outgoing = accesses.emplace_back();
        for (size_t index = 0; index < resources.size(); ++index) {
            const bool graph = index < graphResourceCount;
            if (resources[index].type != RenderGraphResourceType::Texture2D ||
                (graph && !used[index]) || finalStates[index] == ResourceState::Undefined ||
                lastStates[index] == finalStates[index]) { continue; }
            // Restore only changed image layouts. The outer plan already sees
            // every internal access scope and synchronizes the next consumer;
            // a same-layout exit barrier would repeat that RAW dependency.
            const auto scope = graph ? permissions[index] :
                SyncScope{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
            outgoing.uses.push_back({index, finalStates[index], scope, false});
            used[index] = true;
        }
    }
    auto plan = buildGraphAccessPlan(resources, accesses);
    if (!plan) { return makeError(plan.error()); }

    // Retain the complete sequence before its first callback, including resources
    // whose first access needs no barrier. Scope-local BufferSlices are not enough
    // to keep GPU allocations alive after this recording function returns.
    for (size_t index = 0; index < resources.size(); ++index) {
        if (!used[index]) { continue; }
        auto owner = resources[index].type == RenderGraphResourceType::Buffer
            ? resolved[index].buffer.retainAllocation() : textureOwners[index];
        auto result = commandBuffer().retainResource(std::move(owner));
        if (!result) { return result; }
    }

    computeStagesExecuted_ = true;
    computeStagesActive_ = true;
    struct ActiveScope {
        bool& active;
        ~ActiveScope() { active = false; }
    } activeScope{computeStagesActive_};
    for (size_t index = 0; index < stages.size(); ++index) {
        auto result = recordGraphAccessBarriers(commandBuffer(), plan->passes[index], resolved);
        if (!result) { return result; }
        auto profile = profileScope(stages[index].name);
        stagesAllowParallel_ = stages[index].allowParallelCompute;
        result = stages[index].record(commandBuffer());
        stagesAllowParallel_ = false;
        if (!result) { return result; }
        if (stages[index].allowParallelCompute) {
            // A fork may replace the command buffer. Keep the entire opaque
            // operation's resources through its final join, including accesses
            // for which RAW visibility reuse omitted a native texture barrier.
            for (const auto& use : plan->passes[index].uses) {
                const size_t resource = use.resource;
                auto owner = resources[resource].type == RenderGraphResourceType::Buffer
                    ? resolved[resource].buffer.retainAllocation() : textureOwners[resource];
                result = commandBuffer().retainResource(std::move(owner));
                if (!result) { return result; }
            }
        }
    }
    if (!computeOnly) {
        auto result = recordGraphAccessBarriers(commandBuffer(), plan->passes.back(), resolved);
        if (!result) { return result; }
    }
    // Local phases never publish state into the immutable enclosing pass plan.
    return {};
}

} // namespace metallic::render
