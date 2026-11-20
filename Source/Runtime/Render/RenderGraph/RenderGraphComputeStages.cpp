#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"

#include <algorithm>
#include <array>
#include <memory_resource>
#include <unordered_map>

namespace metallic::render {
namespace {

struct StageResourceIdentity
{
    const void* allocation;
    RenderGraphResourceType type;
    bool operator==(const StageResourceIdentity&) const = default;
};

struct StageResourceIdentityHash
{
    size_t operator()(const StageResourceIdentity& identity) const
    {
        return std::hash<const void*>{}(identity.allocation) ^ static_cast<size_t>(identity.type);
    }
};

struct StageResourceInfo
{
    std::shared_ptr<void> owner;
    SyncScope permissions;
    ResourceState finalState = ResourceState::Undefined;
    ResourceState lastState = ResourceState::Undefined;
    bool internalLayouts = false;
    bool used = false;
};

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
    return executeStagesImpl(stages, imports, {}, {}, true);
}

Result<> RenderGraphExecutionContext::executeStages(std::span<const RenderGraphStage> stages,
    std::span<const RenderGraphBufferImport> buffers, std::span<const RenderGraphTextureImport> textures,
    std::span<const RenderGraphAccelerationStructureImport> accelerationStructures)
{
    return executeStagesImpl(stages, buffers, textures, accelerationStructures, false);
}

Result<> RenderGraphExecutionContext::executeStagesImpl(std::span<const RenderGraphStage> stages,
    std::span<const RenderGraphBufferImport> imports, std::span<const RenderGraphTextureImport> textures,
    std::span<const RenderGraphAccelerationStructureImport> accelerationStructures, bool computeOnly)
{
    using namespace detail;
    if (computeStagesExecuted_ || stages.empty() || !commandBuffer().recording()) {
        return makeError(Error::InvalidArgument);
    }

    // These tables are only needed during CPU recording. Larger sequences
    // can grow through the arena's upstream allocator.
    std::array<std::byte, 8 * 1024> storage;
    std::pmr::monotonic_buffer_resource scratch(storage.data(), storage.size());
    std::pmr::vector<GraphAccessResource> resources(&scratch);
    std::pmr::vector<GraphAccessBinding> resolved(&scratch);
    std::pmr::vector<StageResourceInfo> resourceInfo(&scratch);
    std::pmr::vector<std::pmr::string> qualifiedNames(&scratch);
    std::pmr::unordered_map<std::string_view, size_t> names(&scratch);
    std::pmr::unordered_map<StageResourceIdentity, size_t, StageResourceIdentityHash> identities(&scratch);
    const size_t importCount = imports.size() + textures.size() + accelerationStructures.size();
    const size_t resourceCount = bindings_.size() + importCount;
    resources.reserve(resourceCount);
    resolved.reserve(resourceCount);
    resourceInfo.reserve(resourceCount);
    qualifiedNames.reserve(bindings_.size());
    names.reserve(2 * bindings_.size() + importCount);
    identities.reserve(resourceCount);
    const auto device = commandBuffer().deviceIdentity();

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
            if (bound->buffer.deviceIdentity() != device) {
                return makeError(Error::InvalidArgument);
            }
        } else if (resource.type == RenderGraphResourceType::AccelerationStructure) {
            if (resource.accelerationStructure) {
                if (resource.accelerationStructure->deviceIdentity() != device) {
                    return makeError(Error::InvalidArgument);
                }
                textureOwner = resource.accelerationStructure->retainAllocation();
            }
            identity = textureOwner ? textureOwner.get() : binding.resource;
        } else {
            if (!resource.view || resource.view->deviceIdentity() != device) {
                return makeError(Error::InvalidArgument);
            }
            textureOwner = resource.view->retainTexture();
            identity = textureOwner.get();
            // Graph-owned images have a retained identity. Borrowed images need
            // an explicit import/completion contract and are outside this API.
            if (!identity) { return makeError(Error::InvalidArgument); }
        }
        const auto [entry, inserted] = identities.try_emplace(StageResourceIdentity{identity, resource.type}, resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = resource.type, .state = resource.state, .boundarySynchronized = true});
            resolved.push_back(std::move(*bound));
            resourceInfo.push_back({.owner = std::move(textureOwner), .permissions = binding.scope,
                .finalState = resource.state, .lastState = resource.state, .internalLayouts = binding.internalLayouts});
        } else {
            if (resources[index].state != resource.state) { return makeError(Error::InvalidArgument); }
            auto& info = resourceInfo[index];
            mergeScope(info.permissions, binding.scope);
            info.internalLayouts = info.internalLayouts || binding.internalLayouts;
        }
        auto& qualified = qualifiedNames.emplace_back(binding.visibility == RenderGraphFieldVisibility::Input ? "input." : "output.");
        qualified += binding.fieldName;
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
            imported.buffer.deviceIdentity() != device ||
            !accessMatchesResourceType(imported.access, RenderGraphResourceType::Buffer) ||
            (computeOnly && scope.stages != PipelineStageBits::ComputeShader) ||
            (static_cast<uint64_t>(imported.buffer.allocationDesc().usage) & static_cast<uint64_t>(requiredUsage)) !=
                static_cast<uint64_t>(requiredUsage)) {
            return makeError(Error::InvalidArgument);
        }
        const auto [entry, inserted] = identities.try_emplace(
            StageResourceIdentity{imported.buffer.allocationIdentity(), RenderGraphResourceType::Buffer}, resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = RenderGraphResourceType::Buffer, .state = state,
                .scope = computeOnly ? scope : SyncScope{PipelineStageBits::AllCommands,
                    AccessBits::MemoryRead | AccessBits::MemoryWrite}});
            resolved.push_back({.buffer = imported.buffer});
            resourceInfo.push_back({.permissions = scope, .lastState = state});
        } else if (index < graphResourceCount || resources[index].state != state ||
            resourceInfo[index].permissions.stages != scope.stages || resourceInfo[index].permissions.access != scope.access) {
            // A private import cannot grant hidden writes to a reflected field.
            // Aliases of a private allocation must agree on its boundary contract.
            return makeError(Error::InvalidArgument);
        }
        if (!names.emplace(imported.name, index).second) { return makeError(Error::InvalidArgument); }
    }

    for (const auto& imported : textures) {
        if (computeOnly || imported.name.empty() || !imported.texture ||
            imported.texture->deviceIdentity() != device ||
            !validTextureState(*imported.texture, imported.initialState) ||
            !validTextureState(*imported.texture, imported.finalState)) {
            return makeError(Error::InvalidArgument);
        }
        auto owner = imported.texture->retainAllocation();
        if (!owner || (imported.view && imported.view->retainTexture() != owner)) {
            return makeError(Error::InvalidArgument);
        }
        const auto [entry, inserted] = identities.try_emplace(
            StageResourceIdentity{owner.get(), RenderGraphResourceType::Texture2D}, resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = RenderGraphResourceType::Texture2D, .state = imported.initialState,
                .scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}});
            resolved.push_back({.texture = imported.texture, .mipCount = imported.texture->desc().mipCount,
                .layerCount = imported.texture->desc().layerCount});
            resourceInfo.push_back({.owner = std::move(owner), .finalState = imported.finalState,
                .lastState = imported.initialState, .internalLayouts = true});
        } else if (index < graphResourceCount || resources[index].state != imported.initialState ||
            resourceInfo[index].finalState != imported.finalState) {
            return makeError(Error::InvalidArgument);
        }
        if (!names.emplace(imported.name, index).second) { return makeError(Error::InvalidArgument); }
    }

    for (const auto& imported : accelerationStructures) {
        if (computeOnly || imported.name.empty() || !imported.accelerationStructure ||
            !imported.accelerationStructure->valid() ||
            imported.accelerationStructure->deviceIdentity() != device ||
            !accessMatchesResourceType(imported.access, RenderGraphResourceType::AccelerationStructure) ||
            imported.access == RenderGraphResourceAccess::None) {
            return makeError(Error::InvalidArgument);
        }
        auto owner = imported.accelerationStructure->retainAllocation();
        if (!owner) { return makeError(Error::InvalidArgument); }
        const auto scope = scopeForGraphAccess(imported.access, RenderGraphPassKind::Compute);
        const auto [entry, inserted] = identities.try_emplace(
            StageResourceIdentity{owner.get(), RenderGraphResourceType::AccelerationStructure}, resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = RenderGraphResourceType::AccelerationStructure,
                .state = stateForAccess(imported.access),
                .scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}});
            resolved.push_back({.accelerationStructure = imported.accelerationStructure});
            resourceInfo.push_back({.owner = std::move(owner), .permissions = scope});
        } else if (index < graphResourceCount || resourceInfo[index].permissions.stages != scope.stages ||
            resourceInfo[index].permissions.access != scope.access) {
            return makeError(Error::InvalidArgument);
        }
        if (!names.emplace(imported.name, index).second) { return makeError(Error::InvalidArgument); }
    }

    std::pmr::vector<GraphAccessPass> accesses(&scratch);
    accesses.reserve(stages.size() + size_t(!computeOnly));
    for (const auto& stage : stages) {
        if (stage.name.empty() || !stage.record || (computeOnly && stage.kind != RenderGraphPassKind::Compute) ||
            (stage.allowParallelCompute && (computeOnly || stage.kind != RenderGraphPassKind::Unsafe))) {
            return makeError(Error::InvalidArgument);
        }
        auto& access = accesses.emplace_back();
        access.uses.reserve(stage.uses.size());
        for (const auto& use : stage.uses) {
            const auto found = names.find(use.resource);
            if (found == names.end() || found->second == SIZE_MAX) { return makeError(Error::InvalidArgument); }
            const size_t index = found->second;
            const auto declared = declaredGraphAccess(index, use.access, stage.kind);
            auto& info = resourceInfo[index];
            const bool graph = index < graphResourceCount;
            if (use.access == RenderGraphResourceAccess::None || !accessMatchesResourceType(use.access, resources[index].type) ||
                (computeOnly && declared.scope.stages != PipelineStageBits::ComputeShader) ||
                ((graph || computeOnly) && !scopeContains(info.permissions, declared.scope)) ||
                (resources[index].type == RenderGraphResourceType::Texture2D && resources[index].state != declared.state &&
                    (computeOnly || !info.internalLayouts))) {
                return makeError(Error::InvalidArgument);
            }
            if (resources[index].type == RenderGraphResourceType::Buffer) {
                const auto required = static_cast<uint64_t>(bufferUsageForAccess(use.access));
                if ((static_cast<uint64_t>(resolved[index].buffer.allocationDesc().usage) & required) != required) {
                    return makeError(Error::InvalidArgument);
                }
            } else if (resources[index].type == RenderGraphResourceType::Texture2D) {
                const auto required = static_cast<uint64_t>(textureUsageForAccess(use.access));
                if ((static_cast<uint64_t>(resolved[index].texture->desc().usage) & required) != required) {
                    return makeError(Error::InvalidArgument);
                }
            }
            info.used = true;
            info.lastState = declared.state;
            access.uses.push_back(declared);
        }
    }
    if (!computeOnly) {
        auto& outgoing = accesses.emplace_back();
        for (size_t index = 0; index < resources.size(); ++index) {
            const bool graph = index < graphResourceCount;
            auto& info = resourceInfo[index];
            if (resources[index].type != RenderGraphResourceType::Texture2D ||
                (graph && !info.used) || info.finalState == ResourceState::Undefined ||
                info.lastState == info.finalState) { continue; }
            // Restore only changed image layouts. The outer plan already sees
            // every internal access scope and synchronizes the next consumer;
            // a same-layout exit barrier would repeat that RAW dependency.
            const auto scope = graph ? info.permissions :
                SyncScope{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
            outgoing.uses.push_back({index, info.finalState, scope, false});
            info.used = true;
        }
    }
    auto plan = buildGraphAccessPlan(resources, accesses, GraphAccessPlanMode::InternalStages);
    if (!plan) { return makeError(plan.error()); }

    if (executionCapture_ && capturedImports_) {
        std::vector<uint64_t> ids;
        ids.reserve(resources.size());
        for (size_t index = 0; index < resources.size(); ++index) {
            RenderGraphExecutionResourceSnapshot resource;
            resource.type = resources[index].type;
            resource.privateResource = index >= graphResourceCount;
            if (resource.type == RenderGraphResourceType::Buffer) {
                resource.memory = resolved[index].buffer.memoryInfo();
                resource.bufferDesc = resolved[index].buffer.allocationDesc();
            } else if (resource.type == RenderGraphResourceType::AccelerationStructure) {
                const auto* accelerationStructure = resolved[index].accelerationStructureResource
                    ? resolved[index].accelerationStructureResource->accelerationStructure : resolved[index].accelerationStructure;
                if (accelerationStructure) {
                    resource.memory = accelerationStructure->memoryInfo();
                    resource.accelerationStructureDesc = accelerationStructure->desc();
                }
                resource.id = resolved[index].accelerationStructureResource
                    ? resolved[index].accelerationStructureResource->logicalResourceId : resource.memory.allocationId;
            } else {
                resource.memory = resolved[index].texture->memoryInfo();
                resource.textureDesc = resolved[index].texture->desc();
            }
            if (resource.type != RenderGraphResourceType::AccelerationStructure) { resource.id = resource.memory.allocationId; }
            ids.push_back(resource.id);
            if (resource.privateResource && resourceInfo[index].used) {
                for (const auto& [name, namedIndex] : names) {
                    if (namedIndex == index) {
                        resource.aliases.push_back(passName_ + ".");
                        resource.aliases.back().append(name);
                    }
                }
                std::sort(resource.aliases.begin(), resource.aliases.end());
                if (!resource.aliases.empty()) { resource.name = resource.aliases.front(); }
                capturedImports_->push_back(std::move(resource));
            }
        }
        executionCapture_->stages.reserve(plan->passes.size());
        for (size_t index = 0; index < plan->passes.size(); ++index) {
            const bool restore = index == stages.size();
            auto& captured = executionCapture_->stages.emplace_back();
            captured.name = restore ? "Restore boundary" : std::string(stages[index].name);
            captured.kind = restore ? RenderGraphPassKind::Unsafe : stages[index].kind;
            captured.allowParallelCompute = !restore && stages[index].allowParallelCompute;
            captured.restoreBoundary = restore;
            captureGraphAccessBoundary(plan->passes[index], ids, captured.uses, captured.barriers);
            if (!restore) {
                for (const auto& declared : stages[index].uses) {
                    const auto id = ids[names.at(declared.resource)];
                    for (auto& use : captured.uses) {
                        if (use.resourceId == id) { captureGraphDeclaredAccess(use, declared.access); }
                    }
                }
            }
        }
    }

    // Retain the complete sequence before its first callback, including resources
    // whose first access needs no barrier. Scope-local BufferSlices are not enough
    // to keep GPU allocations alive after this recording function returns.
    for (size_t index = 0; index < resources.size(); ++index) {
        if (!resourceInfo[index].used) { continue; }
        auto owner = resources[index].type == RenderGraphResourceType::Buffer
            ? resolved[index].buffer.retainAllocation() : resourceInfo[index].owner;
        if (!owner && resources[index].type == RenderGraphResourceType::AccelerationStructure) { continue; }
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
        const auto before = executionCapture_ ? commandBuffer().synchronizationStats() : SynchronizationStats{};
        auto result = recordGraphAccessBarriers(commandBuffer(), plan->passes[index], resolved);
        if (!result) { return result; }
        if (executionCapture_) {
            executionCapture_->stages[index].synchronization = synchronizationDelta(before, commandBuffer().synchronizationStats());
        }
        auto profile = profileScope(stages[index].name);
        stagesAllowParallel_ = stages[index].allowParallelCompute;
        result = stages[index].record(commandBuffer());
        stagesAllowParallel_ = false;
        if (!result) { return result; }
        if (executionCapture_) { executionCapture_->stages[index].recorded = true; }
        if (stages[index].allowParallelCompute) {
            // A fork may replace the command buffer. Keep the entire opaque
            // operation's resources through its final join, including accesses
            // for which RAW visibility reuse omitted a native texture barrier.
            for (const auto& use : plan->passes[index].uses) {
                const size_t resource = use.resource;
                auto owner = resources[resource].type == RenderGraphResourceType::Buffer
                    ? resolved[resource].buffer.retainAllocation() : resourceInfo[resource].owner;
                if (!owner && resources[resource].type == RenderGraphResourceType::AccelerationStructure) { continue; }
                result = commandBuffer().retainResource(std::move(owner));
                if (!result) { return result; }
            }
        }
    }
    if (!computeOnly) {
        const auto before = executionCapture_ ? commandBuffer().synchronizationStats() : SynchronizationStats{};
        auto result = recordGraphAccessBarriers(commandBuffer(), plan->passes.back(), resolved);
        if (!result) { return result; }
        if (executionCapture_) {
            auto& captured = executionCapture_->stages.back();
            captured.synchronization = synchronizationDelta(before, commandBuffer().synchronizationStats());
            captured.recorded = true;
        }
    }
    // Local phases never publish state into the immutable enclosing pass plan.
    return {};
}

} // namespace metallic::render
