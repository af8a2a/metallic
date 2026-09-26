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

} // namespace

Result<> RenderGraphExecutionContext::executeComputeStages(
    std::span<const RenderGraphComputeStage> stages, std::span<const RenderGraphBufferImport> imports)
{
    using namespace detail;
    if (computeStagesExecuted_ || stages.empty() || !commandBuffer().recording()) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<GraphAccessResource> resources;
    std::vector<GraphAccessBinding> resolved;
    std::vector<SyncScope> permissions;
    std::vector<std::shared_ptr<void>> textureOwners;
    std::unordered_map<std::string_view, size_t> names;
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
            textureOwners.push_back(std::move(textureOwner));
        } else {
            if (resources[index].state != resource.state) { return makeError(Error::InvalidArgument); }
            mergeScope(permissions[index], binding.scope);
        }
        if (!names.emplace(binding.fieldName, index).second) { return makeError(Error::InvalidArgument); }
    }

    const size_t graphResourceCount = resources.size();
    for (const auto& imported : imports) {
        const auto scope = scopeForGraphAccess(imported.access, RenderGraphPassKind::Compute);
        const auto state = stateForAccess(imported.access);
        const auto requiredUsage = bufferUsageForAccess(imported.access);
        if (imported.name.empty() || !imported.buffer.valid() || imported.buffer.size() == 0 ||
            imported.buffer.deviceIdentity() != commandBuffer().deviceIdentity() ||
            !accessMatchesResourceType(imported.access, RenderGraphResourceType::Buffer) ||
            scope.stages != PipelineStageBits::ComputeShader ||
            (static_cast<uint64_t>(imported.buffer.allocationDesc().usage) & static_cast<uint64_t>(requiredUsage)) !=
                static_cast<uint64_t>(requiredUsage)) {
            return makeError(Error::InvalidArgument);
        }
        const auto [entry, inserted] = bufferIdentities.try_emplace(imported.buffer.allocationIdentity(), resources.size());
        const size_t index = entry->second;
        if (inserted) {
            resources.push_back({.type = RenderGraphResourceType::Buffer, .state = state, .scope = scope});
            resolved.push_back({.buffer = imported.buffer});
            permissions.push_back(scope);
            textureOwners.push_back({});
        } else if (index < graphResourceCount || resources[index].state != state ||
            permissions[index].stages != scope.stages || permissions[index].access != scope.access) {
            // A private import cannot grant hidden writes to a reflected field.
            // Aliases of a private allocation must agree on its boundary contract.
            return makeError(Error::InvalidArgument);
        }
        if (!names.emplace(imported.name, index).second) { return makeError(Error::InvalidArgument); }
    }

    std::vector<GraphAccessPass> accesses;
    accesses.reserve(stages.size());
    std::vector<bool> used(resources.size(), false);
    for (const auto& stage : stages) {
        if (stage.name.empty() || !stage.record) { return makeError(Error::InvalidArgument); }
        auto& access = accesses.emplace_back();
        for (const auto& use : stage.uses) {
            const auto found = names.find(use.resource);
            if (found == names.end()) { return makeError(Error::InvalidArgument); }
            const size_t index = found->second;
            const auto scope = scopeForGraphAccess(use.access, RenderGraphPassKind::Compute);
            const auto state = stateForAccess(use.access);
            if (!accessMatchesResourceType(use.access, resources[index].type) ||
                scope.stages != PipelineStageBits::ComputeShader || !scopeContains(permissions[index], scope) ||
                (resources[index].type == RenderGraphResourceType::Texture2D && resources[index].state != state)) {
                return makeError(Error::InvalidArgument);
            }
            used[index] = true;
            access.uses.push_back({index, state, scope, accessWrites(use.access)});
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
        result = stages[index].record(commandBuffer());
        if (!result) { return result; }
    }
    // Local phases never publish state into the immutable enclosing pass plan.
    return {};
}

} // namespace metallic::render
