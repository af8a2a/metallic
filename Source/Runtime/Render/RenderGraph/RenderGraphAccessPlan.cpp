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

struct ResourceFrontier {
    ResourceState state = ResourceState::Undefined;
    std::optional<AccessEvent> writer;
    std::optional<AccessEvent> layout;
    std::vector<AccessEvent> readers;
};

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
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageRead:
        return {shaderStages, AccessBits::ShaderRead};
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::BufferStorageWrite:
        return {shaderStages, AccessBits::ShaderWrite};
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
        return {shaderStages, AccessBits::ShaderRead | AccessBits::ShaderWrite};
    case RenderGraphResourceAccess::BufferConstantRead:
        return {shaderStages, AccessBits::UniformRead};
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
        if (resource.state != ResourceState::Undefined) {
            SyncScope scope = resource.scope;
            if (scope.stages == PipelineStageBits::None) {
                scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
            }
            frontier.writer = AccessEvent{.scope = scope};
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
            SyncScope before;
            bool localMemoryProducer = false;
            const auto consume = [&](const AccessEvent& event, bool memoryProducer) {
                if (event.pass != kInitialAccess &&
                    std::find(planned.predecessors.begin(), planned.predecessors.end(), event.pass) == planned.predecessors.end()) {
                    planned.predecessors.push_back(event.pass);
                }
                // Semaphores provide availability and visibility for remote
                // producers. Their stage masks may be illegal on this queue.
                if (event.pass == kInitialAccess || event.queue == pass.queue) {
                    mergeScope(before, event.scope);
                    localMemoryProducer |= memoryProducer;
                }
            };

            if (frontier.writer) { consume(*frontier.writer, true); }
            if (frontier.layout) { consume(*frontier.layout, true); }
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
                frontier.layout = current;
                frontier.layout->scope.access = AccessBits::MemoryWrite;
            }
            if (use.writes) {
                frontier.writer = current;
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
