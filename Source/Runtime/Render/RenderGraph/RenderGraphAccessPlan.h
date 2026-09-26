#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"

#include <cstddef>
#include <span>
#include <vector>

namespace metallic::render::detail {

// Resource indices refer to canonical graph allocations, including every input
// alias. Queue indices identify actual queues, not individual Queue wrappers.
struct GraphAccessResource {
    RenderGraphResourceType type;
    ResourceState state = ResourceState::Undefined;
    SyncScope scope;
    // The parent access boundary already made prior work visible to every
    // internal stage. Keep its layout without inventing a second producer.
    bool boundarySynchronized = false;
};

struct GraphAccessUse {
    size_t resource = 0;
    ResourceState state = ResourceState::Undefined;
    SyncScope scope;
    bool writes = false;
};

struct GraphAccessPass {
    uint32_t queue = 0;
    std::vector<GraphAccessUse> uses;
};

struct GraphAccessBarrier {
    size_t resource = 0;
    ResourceState before = ResourceState::Undefined;
    ResourceState after = ResourceState::Undefined;
    SyncScope beforeScope;
    SyncScope afterScope;
    bool executionOnly = false;
};

struct GraphAccessPassPlan {
    std::vector<size_t> predecessors;
    std::vector<GraphAccessBarrier> barriers;
    std::vector<GraphAccessUse> uses;
};

struct GraphAccessPlan {
    std::vector<GraphAccessPassPlan> passes;
};

struct GraphAccessBinding {
    // Owned allocation required; borrowed images need an external lifetime
    // contract and cannot be encoded through this planner yet.
    Texture* texture = nullptr;
    uint32_t mipCount = 1;
    uint32_t layerCount = 1;
    BufferSlice buffer;
};

Result<GraphAccessBinding> bindGraphAccessResource(const RenderGraphResource& resource);
// Shared by whole graph passes and internal stages. Texture and buffer uses retain
// their allocation through submission even when their first access needs no barrier.
Result<> recordGraphAccessBarriers(CommandBuffer& commands, const GraphAccessPassPlan& pass,
    std::span<const GraphAccessBinding> bindings);

// Pure CPU planning. A defined initial state represents prior external/frame
// work; the caller must provide its queue waits before executing the plan.
// Passes must preserve input order on each actual queue: visibility established
// by an earlier boundary can cover later reads, including later submissions.
// Coverage is local to this invocation; failed/cancelled plans cannot seed it.
Result<GraphAccessPlan> buildGraphAccessPlan(
    std::span<const GraphAccessResource> resources,
    std::span<const GraphAccessPass> passes);

SyncScope scopeForGraphAccess(RenderGraphResourceAccess access, RenderGraphPassKind kind);

// Diagnostic values only; resource IDs are allocation generations.
void captureGraphAccessBoundary(const GraphAccessPassPlan& pass, std::span<const uint64_t> resourceIds,
    std::vector<RenderGraphExecutionUseSnapshot>& uses,
    std::vector<RenderGraphExecutionBarrierSnapshot>& barriers);
void captureGraphDeclaredAccess(RenderGraphExecutionUseSnapshot& use, RenderGraphResourceAccess access);
SynchronizationStats synchronizationDelta(SynchronizationStats before, SynchronizationStats after);

} // namespace metallic::render::detail
