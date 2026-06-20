#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"

#include <algorithm>
#include <functional>
#include <queue>
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
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferWrite:
        return true;
    case RenderGraphResourceAccess::None:
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferConstantRead:
        return false;
    }
    return false;
}

ResourceState stateForAccess(RenderGraphResourceAccess access)
{
    switch (access) {
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
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
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
        return TextureUsageBits::Sampled;
    case RenderGraphResourceAccess::TextureColorWrite:
        return TextureUsageBits::ColorAttachment;
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
        return TextureUsageBits::DepthStencilAttachment;
    case RenderGraphResourceAccess::TextureTransferRead:
        return TextureUsageBits::TransferSource;
    case RenderGraphResourceAccess::TextureTransferWrite:
        return TextureUsageBits::TransferDestination;
    case RenderGraphResourceAccess::TextureStorageReadWrite:
        return TextureUsageBits::Storage;
    case RenderGraphResourceAccess::None:
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferTransferWrite:
    case RenderGraphResourceAccess::BufferConstantRead:
        return TextureUsageBits::None;
    }
    return TextureUsageBits::None;
}

BufferUsageBits bufferUsageForAccess(RenderGraphResourceAccess access)
{
    switch (access) {
    case RenderGraphResourceAccess::BufferShaderRead:
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
    case RenderGraphResourceAccess::TextureStorageReadWrite:
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
        return field.structureStride == 0 ? BufferViewType::Raw : BufferViewType::Structured;
    case RenderGraphResourceAccess::BufferStorageReadWrite:
        return field.structureStride == 0 ? BufferViewType::ReadWriteRaw : BufferViewType::ReadWriteStructured;
    case RenderGraphResourceAccess::None:
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureColorWrite:
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferTransferWrite:
        return field.bufferViewType;
    }
    return field.bufferViewType;
}

bool accessMatchesResourceType(RenderGraphResourceAccess access, RenderGraphResourceType resourceType)
{
    switch (access) {
    case RenderGraphResourceAccess::None:
        return true;
    case RenderGraphResourceAccess::TextureSampleRead:
    case RenderGraphResourceAccess::TextureColorWrite:
    case RenderGraphResourceAccess::TextureDepthStencilWrite:
    case RenderGraphResourceAccess::TextureTransferRead:
    case RenderGraphResourceAccess::TextureTransferWrite:
    case RenderGraphResourceAccess::TextureStorageReadWrite:
        return resourceType == RenderGraphResourceType::Texture2D;
    case RenderGraphResourceAccess::BufferShaderRead:
    case RenderGraphResourceAccess::BufferStorageReadWrite:
    case RenderGraphResourceAccess::BufferTransferRead:
    case RenderGraphResourceAccess::BufferTransferWrite:
    case RenderGraphResourceAccess::BufferConstantRead:
        return resourceType == RenderGraphResourceType::Buffer;
    }
    return false;
}

TextureUsageBits textureUsageForField(const RenderGraphField& field)
{
    TextureUsageBits usage = textureUsageForAccess(field.access);
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

std::string resultMessage(std::string_view label, const Result& result)
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
    std::string& log)
{
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

    std::queue<std::string> ready;
    for (const auto& [name, degree] : indegree) {
        if (degree == 0) {
            ready.push(name);
        }
    }

    while (!ready.empty()) {
        std::string current = ready.front();
        ready.pop();
        activeGraph.executionOrder.push_back(current);

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

    if (activeGraph.executionOrder.size() != activeGraph.activePasses.size()) {
        log = validationPrefix("cycle detected in active graph");
        return false;
    }
    return true;
}

} // namespace metallic::render::detail
