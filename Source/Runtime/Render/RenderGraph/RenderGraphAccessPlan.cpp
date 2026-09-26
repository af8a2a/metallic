#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace metallic::render::detail {
namespace {

constexpr size_t kInitialAccess = std::numeric_limits<size_t>::max();

struct AccessEvent {
    size_t pass = kInitialAccess;
    uint32_t queue = 0;
    SyncScope scope;
};

struct VisibleScope {
    uint32_t queue = 0;
    SyncScope scope;
};

struct MemoryProducer {
    AccessEvent event;
    std::vector<VisibleScope> visible;
};

struct ResourceFrontier {
    ResourceState state = ResourceState::Undefined;
    std::optional<MemoryProducer> writer;
    std::optional<MemoryProducer> layout;
    std::vector<AccessEvent> readers;
};

bool scopeCovers(SyncScope available, SyncScope requested)
{
    // Match explicit bits conservatively; aggregate stage/access aliases need
    // not be expanded to eliminate repeated uses of the same declaration.
    const auto stages = static_cast<uint64_t>(requested.stages);
    const auto access = static_cast<uint64_t>(requested.access);
    return (static_cast<uint64_t>(available.stages) & stages) == stages &&
        (static_cast<uint64_t>(available.access) & access) == access;
}

bool visibleTo(const MemoryProducer& producer, uint32_t queue, SyncScope scope)
{
    return std::any_of(producer.visible.begin(), producer.visible.end(), [&](const VisibleScope& visible) {
        return visible.queue == queue && scopeCovers(visible.scope, scope);
    });
}

void rememberVisibility(MemoryProducer& producer, uint32_t queue, SyncScope scope)
{
    if (visibleTo(producer, queue, scope)) { return; }
    // Keep stage/access pairs together: independently OR-ing them would invent
    // visibility for combinations no barrier actually made visible.
    std::erase_if(producer.visible, [&](const VisibleScope& visible) {
        return visible.queue == queue && scopeCovers(scope, visible.scope);
    });
    producer.visible.push_back({queue, scope});
}

void mergeScope(SyncScope& destination, SyncScope source)
{
    destination.stages = destination.stages | source.stages;
    destination.access = destination.access | source.access;
}

bool mergeUse(GraphAccessUse& destination, const GraphAccessUse& source, RenderGraphResourceType type)
{
    if (destination.state != source.state) {
        // Buffers have no image layout. Both shader declarations can share one
        // binding while retaining their separate read/write scope information.
        const auto shaderState = [](ResourceState state) {
            return state == ResourceState::ShaderRead || state == ResourceState::General;
        };
        if (type != RenderGraphResourceType::Buffer ||
            !shaderState(destination.state) || !shaderState(source.state)) {
            return false;
        }
        destination.state = ResourceState::General;
    }
    mergeScope(destination.scope, source.scope);
    destination.writes |= source.writes;
    return true;
}

} // namespace

SyncScope scopeForGraphAccess(RenderGraphResourceAccess access, RenderGraphPassKind kind)
{
    const PipelineStageBits shaderStages = kind == RenderGraphPassKind::Compute
        ? PipelineStageBits::ComputeShader
        : kind == RenderGraphPassKind::Raster
            ? PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader
            : PipelineStageBits::AllCommands;
    switch (access) {
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureSampleReadGeneral:
    case RenderGraphResourceAccess::BufferShaderRead:
        return {shaderStages, AccessBits::ShaderRead};
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::BufferStorageRead:
        return {shaderStages, kind == RenderGraphPassKind::Unsafe
            ? AccessBits::ShaderRead | AccessBits::MemoryRead : AccessBits::ShaderRead};
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::BufferStorageWrite:
        return {shaderStages, kind == RenderGraphPassKind::Unsafe
            ? AccessBits::ShaderWrite | AccessBits::MemoryWrite : AccessBits::ShaderWrite};
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
        // Opaque SDK operations may also copy, fill or consume indirect data.
        // Keep explicit shader bits for permission subsets and conservatively
        // cover those hidden operations at the declared General boundary.
        return {shaderStages, kind == RenderGraphPassKind::Unsafe
            ? AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::MemoryRead | AccessBits::MemoryWrite
            : AccessBits::ShaderRead | AccessBits::ShaderWrite};
    case RenderGraphResourceAccess::BufferConstantRead:
        return {shaderStages, AccessBits::UniformRead};
    case RenderGraphResourceAccess::BufferIndirectRead:
        return {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead};
    case RenderGraphResourceAccess::TextureColorWrite:
        return {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite};
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
        return {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite};
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::BufferTransferRead:
        return {PipelineStageBits::Transfer, AccessBits::TransferRead};
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::BufferTransferWrite:
        return {PipelineStageBits::Transfer, AccessBits::TransferWrite};
    case RenderGraphResourceAccess::None:
        return {PipelineStageBits::AllCommands, AccessBits::None};
    }
    return {};
}

void captureGraphAccessBoundary(const GraphAccessPassPlan& pass, std::span<const uint64_t> resourceIds,
    std::vector<RenderGraphExecutionUseSnapshot>& uses,
    std::vector<RenderGraphExecutionBarrierSnapshot>& barriers)
{
    for (const auto& use : pass.uses) {
        uses.push_back({.resourceId = resourceIds[use.resource], .state = use.state,
            .scope = use.scope, .exclusive = use.writes});
    }
    for (const auto& barrier : pass.barriers) {
        barriers.push_back({resourceIds[barrier.resource], barrier.before, barrier.after,
            barrier.beforeScope, barrier.afterScope, barrier.executionOnly});
    }
}

void captureGraphDeclaredAccess(RenderGraphExecutionUseSnapshot& use, RenderGraphResourceAccess access)
{
    // Declarations carry data access. Planner scopes additionally include image
    // layout writes and conservative restore boundaries; those are not data RW.
    const auto scope = scopeForGraphAccess(access, RenderGraphPassKind::Compute);
    constexpr auto reads = AccessBits::ShaderRead | AccessBits::UniformRead | AccessBits::TransferRead |
        AccessBits::ColorRead | AccessBits::DepthStencilRead | AccessBits::IndirectRead;
    constexpr auto writes = AccessBits::ShaderWrite | AccessBits::TransferWrite | AccessBits::ColorWrite |
        AccessBits::DepthStencilWrite;
    use.reads |= (uint64_t(scope.access) & uint64_t(reads)) != 0;
    use.writes |= (uint64_t(scope.access) & uint64_t(writes)) != 0;
}

SynchronizationStats synchronizationDelta(SynchronizationStats before, SynchronizationStats after)
{
    return {after.calls - before.calls, after.memoryBarriers - before.memoryBarriers,
        after.imageTransitions - before.imageTransitions, after.coalescedResources - before.coalescedResources};
}

Result<GraphAccessBinding> bindGraphAccessResource(const RenderGraphResource& resource)
{
    if (resource.type == RenderGraphResourceType::Texture2D) {
        if (!resource.texture || resource.buffer || !resource.desc.mipCount || !resource.desc.layerCount) {
            return makeError(Error::InvalidArgument);
        }
        return GraphAccessBinding{.texture = resource.texture,
            .mipCount = resource.desc.mipCount, .layerCount = resource.desc.layerCount};
    }
    if (resource.type != RenderGraphResourceType::Buffer || !resource.buffer || resource.texture ||
        !resource.bufferDesc.size) {
        return makeError(Error::InvalidArgument);
    }
    auto slice = resource.buffer->slice(0, resource.bufferDesc.size);
    if (!slice) { return makeError(slice.error()); }
    return GraphAccessBinding{.buffer = std::move(*slice)};
}

Result<> recordGraphAccessBarriers(CommandBuffer& commands, const GraphAccessPassPlan& pass,
    std::span<const GraphAccessBinding> bindings)
{
    // Validate the complete boundary before recording any barrier or retaining
    // allocations. Each binding represents exactly one native resource kind.
    const auto validBinding = [&](size_t resource) {
        if (resource >= bindings.size()) { return false; }
        const auto& binding = bindings[resource];
        if (binding.texture) {
            return !binding.buffer.valid() && binding.mipCount != 0 && binding.layerCount != 0 &&
                binding.texture->deviceIdentity() == commands.deviceIdentity() &&
                binding.texture->retainAllocation() &&
                binding.mipCount <= binding.texture->desc().mipCount &&
                binding.layerCount <= binding.texture->desc().layerCount;
        }
        return binding.buffer.valid() && binding.buffer.size() != 0 &&
            binding.buffer.deviceIdentity() == commands.deviceIdentity();
    };
    for (const auto& use : pass.uses) {
        if (!validBinding(use.resource)) { return makeError(Error::InvalidArgument); }
    }
    for (const auto& barrier : pass.barriers) {
        if (!validBinding(barrier.resource)) { return makeError(Error::InvalidArgument); }
    }

    std::vector<TextureBarrierDesc> textures;
    std::vector<MemoryBarrierDesc> memory;
    for (const auto& barrier : pass.barriers) {
        const auto& binding = bindings[barrier.resource];
        if (binding.texture && !barrier.executionOnly) {
            textures.push_back({.texture = binding.texture, .before = barrier.before, .after = barrier.after,
                .baseMip = 0, .mipCount = binding.mipCount, .baseLayer = 0, .layerCount = binding.layerCount,
                .beforeScope = barrier.beforeScope, .afterScope = barrier.afterScope});
        } else {
            // Buffers have no layout. Explicit execution-only dependencies also
            // need this path because their empty access masks are intentional.
            memory.push_back({barrier.beforeScope, barrier.afterScope});
        }
    }
    for (const auto& use : pass.uses) {
        const auto& binding = bindings[use.resource];
        if (binding.buffer.valid()) {
            auto retained = commands.retainResource(binding.buffer.retainAllocation());
            if (!retained) { return retained; }
        } else {
            auto retained = commands.retainResource(binding.texture->retainAllocation());
            if (!retained) { return retained; }
        }
    }
    return commands.synchronize({.textures = textures.data(), .textureCount = uint32_t(textures.size()),
        .memory = memory.data(), .memoryCount = uint32_t(memory.size())});
}

Result<GraphAccessPlan> buildGraphAccessPlan(
    std::span<const GraphAccessResource> resources,
    std::span<const GraphAccessPass> passes)
{
    GraphAccessPlan result;
    result.passes.resize(passes.size());
    std::vector<ResourceFrontier> frontiers(resources.size());
    for (size_t index = 0; index < resources.size(); ++index) {
        const auto& resource = resources[index];
        auto& frontier = frontiers[index];
        frontier.state = resource.state;
        if (resource.state != ResourceState::Undefined && !resource.boundarySynchronized) {
            SyncScope scope = resource.scope;
            if (scope.stages == PipelineStageBits::None) {
                scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
            }
            frontier.writer = MemoryProducer{.event = {.scope = scope}};
        }
    }

    for (size_t passIndex = 0; passIndex < passes.size(); ++passIndex) {
        const auto& pass = passes[passIndex];
        auto& planned = result.passes[passIndex];
        // A pass is one access boundary. Field order must not invent barriers
        // between aliases that will actually be used by the same GPU commands.
        for (const auto& use : pass.uses) {
            if (use.resource >= resources.size() || use.scope.stages == PipelineStageBits::None) {
                return makeError(Error::InvalidArgument);
            }
            const auto existing = std::find_if(planned.uses.begin(), planned.uses.end(),
                [&](const auto& value) { return value.resource == use.resource; });
            if (existing == planned.uses.end()) {
                planned.uses.push_back(use);
            } else if (!mergeUse(*existing, use, resources[use.resource].type)) {
                return makeError(Error::InvalidArgument);
            }
        }

        for (const auto& use : planned.uses) {
            auto& frontier = frontiers[use.resource];
            const bool texture = resources[use.resource].type == RenderGraphResourceType::Texture2D;
            const bool layoutChange = texture && frontier.state != use.state;
            if (layoutChange && frontier.writer) {
                // Do not carry data visibility across an image transition.
                // The new layout anchor will track its own transition writes.
                frontier.writer->visible.clear();
            }
            SyncScope before;
            bool localMemoryProducer = false;
            const auto consume = [&](const AccessEvent& event, bool memoryProducer, bool alreadyVisible = false) {
                if (event.pass != kInitialAccess &&
                    std::find(planned.predecessors.begin(), planned.predecessors.end(), event.pass) == planned.predecessors.end()) {
                    planned.predecessors.push_back(event.pass);
                }
                // Semaphores provide availability and visibility for remote
                // producers. Their stage masks may be illegal on this queue.
                if (!alreadyVisible && (event.pass == kInitialAccess || event.queue == pass.queue)) {
                    mergeScope(before, event.scope);
                    localMemoryProducer |= memoryProducer;
                }
            };

            const auto consumeProducer = [&](MemoryProducer& producer) {
                // The executor preserves actual-queue submission order. A
                // barrier's destination scope also covers later commands on
                // that queue, including commands in later submissions.
                const bool covered = !use.writes && !layoutChange && visibleTo(producer, pass.queue, use.scope);
                consume(producer.event, true, covered);
                if (producer.event.pass == kInitialAccess || producer.event.queue == pass.queue) {
                    // An uncovered local producer necessarily emits a memory
                    // dependency below. Dependencies remain even when covered.
                    rememberVisibility(producer, pass.queue, use.scope);
                }
            };
            if (frontier.writer) { consumeProducer(*frontier.writer); }
            if (frontier.layout) { consumeProducer(*frontier.layout); }
            if (use.writes || layoutChange) {
                for (const auto& reader : frontier.readers) { consume(reader, false); }
            }

            if (layoutChange || before.stages != PipelineStageBits::None) {
                const bool executionOnly = !layoutChange && !localMemoryProducer;
                if (before.stages == PipelineStageBits::None) {
                    // Explicit nonempty scope prevents RHI state inference from
                    // introducing unsupported stages of a remote source queue.
                    before = {PipelineStageBits::TopOfPipe, AccessBits::None};
                }
                SyncScope after = use.scope;
                if (executionOnly) {
                    before.access = AccessBits::None;
                    after.access = AccessBits::None;
                }
                planned.barriers.push_back({.resource = use.resource, .before = frontier.state, .after = use.state,
                    .beforeScope = before, .afterScope = after, .executionOnly = executionOnly});
            }

            const AccessEvent current{.pass = passIndex, .queue = pass.queue, .scope = use.scope};
            if (layoutChange) {
                // Layout transitions also write memory. Keep their dependency
                // chain separate from the data writer: later readers on another
                // stage must see both, even after an earlier read consumed them.
                frontier.layout = MemoryProducer{.event = current};
                frontier.layout->event.scope.access = AccessBits::MemoryWrite;
                // The transition barrier already makes its own writes visible
                // to this destination scope and later matching queue accesses.
                rememberVisibility(*frontier.layout, pass.queue, use.scope);
            }
            if (use.writes) {
                frontier.writer = MemoryProducer{.event = current};
                frontier.readers.clear();
            } else {
                frontier.readers.push_back(current);
            }
            frontier.state = use.state;
        }
        std::sort(planned.predecessors.begin(), planned.predecessors.end());
    }
    return result;
}

} // namespace metallic::render::detail
