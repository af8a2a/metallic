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
    Texture* texture = nullptr;
    uint32_t mipCount = 1;
    uint32_t layerCount = 1;
    BufferSlice buffer;
};

Result<GraphAccessBinding> bindGraphAccessResource(const RenderGraphResource& resource);
// Shared by whole graph passes and internal stages. Buffer slices retain their
// allocation through submission even when their first access needs no barrier.
Result<> recordGraphAccessBarriers(CommandBuffer& commands, const GraphAccessPassPlan& pass,
    std::span<const GraphAccessBinding> bindings);

// Pure CPU planning. A defined initial state represents prior external/frame
// work; the caller must provide its queue waits before executing the plan.
Result<GraphAccessPlan> buildGraphAccessPlan(
    std::span<const GraphAccessResource> resources,
    std::span<const GraphAccessPass> passes);

SyncScope scopeForGraphAccess(RenderGraphResourceAccess access, RenderGraphPassKind kind);

} // namespace metallic::render::detail
