#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

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

// Pure CPU planning. A defined initial state represents prior external/frame
// work; the caller must provide its queue waits before executing the plan.
Result<GraphAccessPlan> buildGraphAccessPlan(
    std::span<const GraphAccessResource> resources,
    std::span<const GraphAccessPass> passes);

SyncScope scopeForGraphAccess(RenderGraphResourceAccess access, RenderGraphPassKind kind);

} // namespace metallic::render::detail
