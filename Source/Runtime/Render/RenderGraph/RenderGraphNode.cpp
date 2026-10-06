#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metallic::render::detail {

bool isOutputMarked(const RenderGraph& graph, std::string_view fullName)
{
    for (const RenderGraphOutput& output : graph.outputs()) {
        if (makeRenderGraphFieldName(output.passName, output.fieldName) == fullName) {
            return true;
        }
    }
    return false;
}
TextureUsageBits addTextureUsage(TextureUsageBits usage, TextureUsageBits flag)
{
    return usage | flag;
}

BufferUsageBits addBufferUsage(BufferUsageBits usage, BufferUsageBits flag)
{
    return usage | flag;
}

bool isTextureField(const RenderGraphField& field)
{
    return field.resourceType == RenderGraphResourceType::Texture2D;
}

bool isBufferField(const RenderGraphField& field)
{
    return field.resourceType == RenderGraphResourceType::Buffer;
}

bool isBindlessField(const RenderGraphField& field)
{
    return field.bindlessAccess != RenderGraphBindlessAccess::None;
}

bool isBindlessSampledImageField(const RenderGraphField& field)
{
    return field.bindlessAccess == RenderGraphBindlessAccess::SampledImage;
}

bool isBindlessBufferField(const RenderGraphField& field)
{
    return field.bindlessAccess == RenderGraphBindlessAccess::Buffer;
}

bool accessWrites(RenderGraphResourceAccess access)
{
    switch (access) {
    case RenderGraphResourceAccess::TextureColorWrite:
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferStorageWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferWrite:
    case RenderGraphResourceAccess::AccelerationStructureBuildWrite:
    case RenderGraphResourceAccess::AccelerationStructureBuildReadWrite:
    case RenderGraphResourceAccess::BufferAccelerationStructureScratchReadWrite:
        return true;
    case RenderGraphResourceAccess::None:
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureSampleReadGeneral:
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageRead:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferConstantRead:
    case RenderGraphResourceAccess::BufferIndirectRead:
    case RenderGraphResourceAccess::AccelerationStructureBuildRead:
    case RenderGraphResourceAccess::AccelerationStructureShaderRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureBuildRead:
        return false;
    }
    return false;
}

bool fieldChangesLayout(const RenderGraphField& field)
{
    return isTextureField(field) && std::any_of(field.internalAccesses.begin(), field.internalAccesses.end(),
        [&](const auto& internal) { return stateForAccess(internal.access) != stateForAccess(field.access); });
}

SyncScope fieldAccessScope(const RenderGraphField& field, RenderGraphPassKind kind)
{
    auto scope = scopeForGraphAccess(field.access, kind);
    for (const auto& internal : field.internalAccesses) {
        const auto extra = scopeForGraphAccess(internal.access, internal.kind);
        scope.stages = scope.stages | extra.stages;
        scope.access = scope.access | extra.access;
    }
    // Transitions inside a pass also write image memory, even if every shader
    // access is a read. Export that hazard and serialize competing consumers.
    if (fieldChangesLayout(field)) {
        scope.stages = scope.stages | PipelineStageBits::AllCommands;
        scope.access = scope.access | AccessBits::MemoryWrite;
    }
    return scope;
}

bool fieldAccessWrites(const RenderGraphField& field)
{
    return accessWrites(field.access) || fieldChangesLayout(field) ||
        std::any_of(field.internalAccesses.begin(), field.internalAccesses.end(),
            [](const auto& internal) { return accessWrites(internal.access); });
}

ResourceState stateForAccess(RenderGraphResourceAccess access)
{
    switch (access) {
    case RenderGraphResourceAccess::BufferIndirectRead:
        return ResourceState::IndirectArgument;
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferConstantRead:
        return ResourceState::ShaderRead;
    case RenderGraphResourceAccess::TextureColorWrite:
        return ResourceState::ColorAttachment;
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
        return ResourceState::DepthStencilAttachment;
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::BufferTransferRead:
        return ResourceState::TransferSource;
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::BufferTransferWrite:
        return ResourceState::TransferDestination;
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferStorageRead:
    case RenderGraphResourceAccess::BufferStorageWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
        return ResourceState::General;
    case RenderGraphResourceAccess::TextureSampleReadGeneral:
    case RenderGraphResourceAccess::AccelerationStructureBuildRead:
    case RenderGraphResourceAccess::AccelerationStructureBuildWrite:
    case RenderGraphResourceAccess::AccelerationStructureBuildReadWrite:
    case RenderGraphResourceAccess::AccelerationStructureShaderRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureBuildRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureScratchReadWrite:
        return ResourceState::General;
    case RenderGraphResourceAccess::None:
        return ResourceState::Undefined;
    }
    return ResourceState::Undefined;
}

TextureUsageBits textureUsageForAccess(RenderGraphResourceAccess access)
{
    switch (access) {
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureSampleReadGeneral:
        return TextureUsageBits::Sampled;
    case RenderGraphResourceAccess::TextureColorWrite:
        return TextureUsageBits::ColorAttachment;
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
        return TextureUsageBits::DepthStencilAttachment;
    case RenderGraphResourceAccess::TextureTransferRead:
        return TextureUsageBits::TransferSource;
    case RenderGraphResourceAccess::TextureTransferWrite:
        return TextureUsageBits::TransferDestination;
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
        return TextureUsageBits::Storage;
    case RenderGraphResourceAccess::None:
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageRead:
    case RenderGraphResourceAccess::BufferStorageWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferTransferWrite:
    case RenderGraphResourceAccess::BufferConstantRead:
    case RenderGraphResourceAccess::BufferIndirectRead:
    case RenderGraphResourceAccess::AccelerationStructureBuildRead:
    case RenderGraphResourceAccess::AccelerationStructureBuildWrite:
    case RenderGraphResourceAccess::AccelerationStructureBuildReadWrite:
    case RenderGraphResourceAccess::AccelerationStructureShaderRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureBuildRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureScratchReadWrite:
        return TextureUsageBits::None;
    }
    return TextureUsageBits::None;
}

BufferUsageBits bufferUsageForAccess(RenderGraphResourceAccess access)
{
    switch (access) {
    case RenderGraphResourceAccess::BufferAccelerationStructureBuildRead:
        return BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress;
    case RenderGraphResourceAccess::BufferAccelerationStructureScratchReadWrite:
        return BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress;
    case RenderGraphResourceAccess::BufferIndirectRead:
        return BufferUsageBits::Indirect;
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageRead:
    case RenderGraphResourceAccess::BufferStorageWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
        return BufferUsageBits::Storage;
    case RenderGraphResourceAccess::BufferTransferRead:
        return BufferUsageBits::TransferSource;
    case RenderGraphResourceAccess::BufferTransferWrite:
        return BufferUsageBits::TransferDestination;
    case RenderGraphResourceAccess::BufferConstantRead:
        return BufferUsageBits::Constant;
    case RenderGraphResourceAccess::None:
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureColorWrite:
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::TextureSampleReadGeneral:
    case RenderGraphResourceAccess::AccelerationStructureBuildRead:
    case RenderGraphResourceAccess::AccelerationStructureBuildWrite:
    case RenderGraphResourceAccess::AccelerationStructureBuildReadWrite:
    case RenderGraphResourceAccess::AccelerationStructureShaderRead:
        return BufferUsageBits::None;
    }
    return BufferUsageBits::None;
}

BufferViewType bufferViewTypeForField(const RenderGraphField& field)
{
    switch (field.access) {
    case RenderGraphResourceAccess::BufferConstantRead:
        return BufferViewType::Constant;
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageRead:
        return field.structureStride == 0 ? BufferViewType::Raw : BufferViewType::Structured;
    case RenderGraphResourceAccess::BufferStorageWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
        return field.structureStride == 0 ? BufferViewType::ReadWriteRaw : BufferViewType::ReadWriteStructured;
    case RenderGraphResourceAccess::None:
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureColorWrite:
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferTransferWrite:
    case RenderGraphResourceAccess::BufferIndirectRead:
    case RenderGraphResourceAccess::TextureSampleReadGeneral:
    case RenderGraphResourceAccess::AccelerationStructureBuildRead:
    case RenderGraphResourceAccess::AccelerationStructureBuildWrite:
    case RenderGraphResourceAccess::AccelerationStructureBuildReadWrite:
    case RenderGraphResourceAccess::AccelerationStructureShaderRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureBuildRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureScratchReadWrite:
        return field.bufferViewType;
    }
    return field.bufferViewType;
}

bool accessMatchesResourceType(RenderGraphResourceAccess access, RenderGraphResourceType resourceType)
{
    switch (access) {
    case RenderGraphResourceAccess::AccelerationStructureBuildRead:
    case RenderGraphResourceAccess::AccelerationStructureBuildWrite:
    case RenderGraphResourceAccess::AccelerationStructureBuildReadWrite:
    case RenderGraphResourceAccess::AccelerationStructureShaderRead:
        return resourceType == RenderGraphResourceType::AccelerationStructure;
    case RenderGraphResourceAccess::None:
        return true;
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureColorWrite:
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::TextureStorageRead:
    case RenderGraphResourceAccess::TextureStorageWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::TextureSampleReadGeneral:
        return resourceType == RenderGraphResourceType::Texture2D;
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageRead:
    case RenderGraphResourceAccess::BufferStorageWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferTransferWrite:
    case RenderGraphResourceAccess::BufferConstantRead:
    case RenderGraphResourceAccess::BufferIndirectRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureBuildRead:
    case RenderGraphResourceAccess::BufferAccelerationStructureScratchReadWrite:
        return resourceType == RenderGraphResourceType::Buffer;
    }
    return false;
}

TextureUsageBits textureUsageForField(const RenderGraphField& field)
{
    TextureUsageBits usage = textureUsageForAccess(field.access);
    for (const auto& internal : field.internalAccesses) {
        usage = addTextureUsage(usage, textureUsageForAccess(internal.access));
    }
    if (field.usage != TextureUsageBits::None) {
        usage = addTextureUsage(usage, field.usage);
    }
    if (isBindlessSampledImageField(field)) {
        usage = addTextureUsage(usage, TextureUsageBits::Sampled);
    }
    return usage;
}

BufferUsageBits bufferUsageForField(const RenderGraphField& field)
{
    BufferUsageBits usage = bufferUsageForAccess(field.access);
    for (const auto& internal : field.internalAccesses) {
        usage = addBufferUsage(usage, bufferUsageForAccess(internal.access));
    }
    if (field.bufferUsage != BufferUsageBits::None) {
        usage = addBufferUsage(usage, field.bufferUsage);
    }
    if (isBindlessBufferField(field)) {
        usage = addBufferUsage(usage, BufferUsageBits::Storage);
    }
    return usage;
}

void applyAccessDefaults(RenderGraphField& field)
{
    field.state = stateForAccess(field.access);
    if (field.resourceType == RenderGraphResourceType::Texture2D) {
        field.usage = textureUsageForAccess(field.access);
        return;
    }

    field.usage = TextureUsageBits::None;
    field.bufferUsage = bufferUsageForAccess(field.access);
    field.bufferViewType = bufferViewTypeForField(field);
}

RenderGraphResourceAccess explicitAccessForState(RenderGraphResourceType type, ResourceState state)
{
    if (type == RenderGraphResourceType::AccelerationStructure) {
        return state == ResourceState::Undefined ? RenderGraphResourceAccess::None :
            state == ResourceState::ShaderRead ? RenderGraphResourceAccess::AccelerationStructureShaderRead :
            RenderGraphResourceAccess::AccelerationStructureBuildReadWrite;
    }
    if (type == RenderGraphResourceType::Texture2D) {
        switch (state) {
        case ResourceState::ShaderRead:
            return RenderGraphResourceAccess::TextureSampleRead;
        case ResourceState::ColorAttachment:
            return RenderGraphResourceAccess::TextureColorWrite;
        case ResourceState::DepthStencilAttachment:
            return RenderGraphResourceAccess::TextureDepthStencilWrite;
        case ResourceState::TransferSource:
            return RenderGraphResourceAccess::TextureTransferRead;
        case ResourceState::TransferDestination:
            return RenderGraphResourceAccess::TextureTransferWrite;
        case ResourceState::General:
            return RenderGraphResourceAccess::TextureStorageReadWrite;
        case ResourceState::Undefined:
        case ResourceState::Present:
            return RenderGraphResourceAccess::None;
        }
    }

    switch (state) {
    case ResourceState::ShaderRead:
        return RenderGraphResourceAccess::BufferShaderRead;
    case ResourceState::TransferSource:
        return RenderGraphResourceAccess::BufferTransferRead;
    case ResourceState::TransferDestination:
        return RenderGraphResourceAccess::BufferTransferWrite;
    case ResourceState::General:
        return RenderGraphResourceAccess::BufferStorageReadWrite;
    case ResourceState::Undefined:
    case ResourceState::Present:
    case ResourceState::ColorAttachment:
    case ResourceState::DepthStencilAttachment:
        return RenderGraphResourceAccess::None;
    }
    return RenderGraphResourceAccess::None;
}

Format resolveFormat(Format format, Format defaultFormat)
{
    return format == Format::Unknown ? defaultFormat : format;
}

std::string resultMessage(std::string_view label, const Result<>& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

std::string passProfileMarkerName(const std::string& name, const std::string& type)
{
    std::string marker("RenderGraphPass: ");
    marker += name;
    marker += " (";
    marker += type;
    marker += ")";
    return marker;
}

ColorValue debugLabelColorFromArgb(uint32_t argb)
{
    constexpr float kInv255 = 1.0f / 255.0f;
    return ColorValue{
        static_cast<float>((argb >> 16u) & 0xffu) * kInv255,
        static_cast<float>((argb >> 8u) & 0xffu) * kInv255,
        static_cast<float>(argb & 0xffu) * kInv255,
        static_cast<float>((argb >> 24u) & 0xffu) * kInv255,
    };
}

const char* queueTypeName(QueueType type)
{
    switch (type) {
    case QueueType::Graphics:
        return "Graphics";
    case QueueType::Compute:
        return "Compute";
    case QueueType::Copy:
        return "Copy";
    }

    return "Unknown";
}

Queue* queueForSubmitDesc(const RenderGraphSubmitDesc& desc, QueueType type)
{
    switch (type) {
    case QueueType::Graphics:
        return desc.graphicsQueue;
    case QueueType::Compute:
        return desc.computeQueue;
    case QueueType::Copy:
        return desc.copyQueue;
    }

    return nullptr;
}

bool nodeNameExists(const std::vector<RenderGraphNode>& nodes, std::string_view name, uint32_t ignoreId)
{
    return std::any_of(
        nodes.begin(),
        nodes.end(),
        [name, ignoreId](const RenderGraphNode& node) {
            return node.id != ignoreId && node.name == name;
        });
}

const RenderGraphNode* findNodeByName(const std::vector<RenderGraphNode>& nodes, std::string_view name)
{
    const auto iter = std::find_if(
        nodes.begin(),
        nodes.end(),
        [name](const RenderGraphNode& node) {
            return node.name == name;
        });
    return iter == nodes.end() ? nullptr : &(*iter);
}

std::string validationPrefix(std::string_view issue)
{
    std::string message("RenderGraph validation failed: ");
    message += issue;
    return message;
}

bool validateAcyclic(
    const std::vector<RenderGraphNode>& nodes,
    const std::vector<RenderGraphEdge>& edges,
    std::string& log)
{
    std::unordered_map<std::string, uint32_t> indegree;
    std::unordered_map<std::string, std::vector<std::string>> outgoing;

    for (const RenderGraphNode& node : nodes) {
        indegree.emplace(node.name, 0);
    }

    for (const RenderGraphEdge& edge : edges) {
        if (indegree.find(edge.srcPass) == indegree.end() || indegree.find(edge.dstPass) == indegree.end()) {
            continue;
        }
        outgoing[edge.srcPass].push_back(edge.dstPass);
        ++indegree[edge.dstPass];
    }

    std::queue<std::string> ready;
    for (const auto& [name, degree] : indegree) {
        if (degree == 0) {
            ready.push(name);
        }
    }

    size_t visited = 0;
    while (!ready.empty()) {
        std::string current = ready.front();
        ready.pop();
        ++visited;

        for (const std::string& next : outgoing[current]) {
            auto iter = indegree.find(next);
            if (iter == indegree.end()) {
                continue;
            }
            if (--iter->second == 0) {
                ready.push(next);
            }
        }
    }

    if (visited != nodes.size()) {
        log = validationPrefix("cycle detected");
        return false;
    }
    return true;
}

bool buildActiveGraph(const RenderGraph& graph, ActiveGraph& activeGraph, std::string& log)
{
    static const std::vector<std::string> kNoExtraOutputs;
    return buildActiveGraph(graph, kNoExtraOutputs, activeGraph, log);
}

bool buildActiveGraph(
    const RenderGraph& graph,
    const std::vector<std::string>& extraOutputs,
    ActiveGraph& activeGraph,
    std::string& log,
    const ActiveGraphSchedulingTraitsResolver& schedulingTraits)
{
    activeGraph = {};
    std::unordered_map<std::string, std::vector<std::string>> incoming;
    for (const RenderGraphEdge& edge : graph.edges()) {
        incoming[edge.dstPass].push_back(edge.srcPass);
    }

    std::function<void(const std::string&)> visitInputs = [&](const std::string& passName) {
        if (!activeGraph.activePasses.insert(passName).second) {
            return;
        }
        for (const std::string& srcPass : incoming[passName]) {
            visitInputs(srcPass);
        }
    };

    for (const RenderGraphOutput& output : graph.outputs()) {
        visitInputs(output.passName);
    }
    const std::string presentationOutput = graph.presentationOutputName();
    if (!presentationOutput.empty()) {
        std::string passName;
        std::string fieldName;
        splitRenderGraphFieldName(presentationOutput, passName, fieldName);
        visitInputs(passName);
    }
    for (const std::string& output : extraOutputs) {
        std::string passName;
        std::string fieldName;
        if (!splitRenderGraphFieldName(output, passName, fieldName)) {
            log = validationPrefix(std::string("invalid extra output '") + output + "'");
            return false;
        }
        visitInputs(passName);
    }

    std::unordered_map<std::string, uint32_t> indegree;
    std::unordered_map<std::string, std::vector<std::string>> outgoing;
    for (const std::string& passName : activeGraph.activePasses) {
        indegree.emplace(passName, 0);
    }
    for (const RenderGraphEdge& edge : graph.edges()) {
        if (!activeGraph.activePasses.contains(edge.srcPass) ||
            !activeGraph.activePasses.contains(edge.dstPass)) {
            continue;
        }
        outgoing[edge.srcPass].push_back(edge.dstPass);
        ++indegree[edge.dstPass];
    }

    std::unordered_map<std::string, const RenderGraphNode*> nodes;
    std::unordered_map<std::string, ActiveGraphSchedulingTraits> traits;
    for (const auto& node : graph.nodes()) {
        if (!activeGraph.activePasses.contains(node.name)) { continue; }
        nodes.emplace(node.name, &node);
        traits.emplace(node.name, schedulingTraits ? schedulingTraits(node) : ActiveGraphSchedulingTraits{});
    }
    if (nodes.size() != activeGraph.activePasses.size()) {
        log = validationPrefix("active pass is missing");
        return false;
    }
    const auto stableLess = [&](const std::string& left, const std::string& right) {
        const uint32_t leftId = nodes.at(left)->id;
        const uint32_t rightId = nodes.at(right)->id;
        return leftId != rightId ? leftId < rightId : left < right;
    };
    const auto schedule = [&](const std::vector<std::string>* opaqueOrder) {
        auto remaining = indegree;
        std::set<std::string, decltype(stableLess)> ready(stableLess);
        for (const auto& [name, degree] : remaining) {
            if (degree == 0) { ready.insert(name); }
        }
        std::vector<std::string> order;
        order.reserve(nodes.size());
        QueueType currentQueue = QueueType::Graphics;
        size_t nextOpaque = 0;
        while (!ready.empty()) {
            const auto allowed = [&](const std::string& name) {
                return opaqueOrder == nullptr || !traits.at(name).opaque ||
                    (nextOpaque < opaqueOrder->size() && name == (*opaqueOrder)[nextOpaque]);
            };
            auto next = ready.end();
            if (opaqueOrder != nullptr) {
                next = std::find_if(ready.begin(), ready.end(), [&](const std::string& name) {
                    return allowed(name) && traits.at(name).queue == currentQueue;
                });
            }
            if (next == ready.end()) { next = std::find_if(ready.begin(), ready.end(), allowed); }
            if (next == ready.end()) { break; }
            const std::string current = *next;
            ready.erase(next);
            order.push_back(current);
            if (opaqueOrder != nullptr) {
                currentQueue = traits.at(current).queue;
                if (traits.at(current).opaque) { ++nextOpaque; }
            }
            for (const std::string& successor : outgoing[current]) {
                if (--remaining.at(successor) == 0) { ready.insert(successor); }
            }
        }
        return order;
    };
    // A legal, stable topology defines opaque relative order. Sorting opaque
    // nodes directly by ID could conflict with a reversed authored dependency.
    const auto baseline = schedule(nullptr);
    if (baseline.size() != nodes.size()) {
        log = validationPrefix("cycle detected in active graph");
        return false;
    }
    std::vector<std::string> opaqueOrder;
    for (const auto& name : baseline) {
        if (traits.at(name).opaque) { opaqueOrder.push_back(name); }
    }
    // Newly ready work competes immediately with independent roots. Prefer the
    // current logical queue, while every data edge and opaque order stays valid.
    activeGraph.executionOrder = schedule(&opaqueOrder);
    if (activeGraph.executionOrder.size() != nodes.size()) {
        log = validationPrefix("unable to preserve active graph ordering");
        return false;
    }
    return true;
}

} // namespace metallic::render::detail
