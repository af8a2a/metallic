#include "Runtime/Render/RenderGraph/render_graph.h"
#include "Runtime/Render/history_resources.h"
#include "Runtime/Render/Profiling/nsight_events.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metallic::render {
namespace {

struct RenderGraphPassRegistryEntry {
    std::string description;
    RenderGraphPassFactory factory;
};

std::unordered_map<std::string, RenderGraphPassRegistryEntry>& passRegistry()
{
    static std::unordered_map<std::string, RenderGraphPassRegistryEntry> registry;
    return registry;
}

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

bool nodeNameExists(const std::vector<RenderGraphNode>& nodes, std::string_view name, uint32_t ignoreId = 0)
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

struct ActiveGraph {
    std::unordered_set<std::string> activePasses;
    std::vector<std::string> executionOrder;
};

bool buildActiveGraph(const RenderGraph& graph, ActiveGraph& activeGraph, std::string& log)
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

} // namespace

RenderGraphField& RenderGraphField::texture2D(uint32_t newWidth, uint32_t newHeight)
{
    resourceType = RenderGraphResourceType::Texture2D;
    width = newWidth;
    height = newHeight;
    if (!accessMatchesResourceType(access, resourceType) || access == RenderGraphResourceAccess::None) {
        access = visibility == RenderGraphFieldVisibility::Input
            ? RenderGraphResourceAccess::TextureSampleRead
            : RenderGraphResourceAccess::TextureColorWrite;
    }
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::buffer(uint64_t newSize, uint32_t newStructureStride)
{
    resourceType = RenderGraphResourceType::Buffer;
    size = newSize;
    structureStride = newStructureStride;
    if (!accessMatchesResourceType(access, resourceType) || access == RenderGraphResourceAccess::None) {
        access = visibility == RenderGraphFieldVisibility::Input
            ? RenderGraphResourceAccess::BufferShaderRead
            : RenderGraphResourceAccess::BufferStorageReadWrite;
    }
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::setOptional(bool value)
{
    optional = value;
    return *this;
}

RenderGraphField& RenderGraphField::sampledRead()
{
    resourceType = RenderGraphResourceType::Texture2D;
    access = RenderGraphResourceAccess::TextureSampleRead;
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::colorWrite()
{
    resourceType = RenderGraphResourceType::Texture2D;
    access = RenderGraphResourceAccess::TextureColorWrite;
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::depthStencilWrite()
{
    resourceType = RenderGraphResourceType::Texture2D;
    access = RenderGraphResourceAccess::TextureDepthStencilWrite;
    applyAccessDefaults(*this);
    format = Format::D32Sfloat;
    return *this;
}

RenderGraphField& RenderGraphField::storageReadWrite()
{
    access = resourceType == RenderGraphResourceType::Buffer
        ? RenderGraphResourceAccess::BufferStorageReadWrite
        : RenderGraphResourceAccess::TextureStorageReadWrite;
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::transferRead()
{
    access = resourceType == RenderGraphResourceType::Buffer
        ? RenderGraphResourceAccess::BufferTransferRead
        : RenderGraphResourceAccess::TextureTransferRead;
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::transferWrite()
{
    access = resourceType == RenderGraphResourceType::Buffer
        ? RenderGraphResourceAccess::BufferTransferWrite
        : RenderGraphResourceAccess::TextureTransferWrite;
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::shaderRead()
{
    resourceType = RenderGraphResourceType::Buffer;
    access = RenderGraphResourceAccess::BufferShaderRead;
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::constantRead()
{
    resourceType = RenderGraphResourceType::Buffer;
    access = RenderGraphResourceAccess::BufferConstantRead;
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::bindlessSampledImage()
{
    resourceType = RenderGraphResourceType::Texture2D;
    bindlessAccess = RenderGraphBindlessAccess::SampledImage;
    if (!accessMatchesResourceType(access, resourceType) || access == RenderGraphResourceAccess::None) {
        access = RenderGraphResourceAccess::TextureSampleRead;
    }
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::bindlessBuffer()
{
    resourceType = RenderGraphResourceType::Buffer;
    bindlessAccess = RenderGraphBindlessAccess::Buffer;
    if (!accessMatchesResourceType(access, resourceType) || access == RenderGraphResourceAccess::None) {
        access = visibility == RenderGraphFieldVisibility::Input
            ? RenderGraphResourceAccess::BufferShaderRead
            : RenderGraphResourceAccess::BufferStorageReadWrite;
    }
    applyAccessDefaults(*this);
    return *this;
}

RenderGraphField& RenderGraphField::hostReadback()
{
    memoryLocation = MemoryLocation::HostReadback;
    return *this;
}

RenderGraphField& RenderPassReflection::addInput(std::string name, std::string description)
{
    return addTextureInput(std::move(name), std::move(description));
}

RenderGraphField& RenderPassReflection::addBindlessSampledInput(std::string name, std::string description)
{
    return addTextureInput(std::move(name), std::move(description)).bindlessSampledImage().sampledRead();
}

RenderGraphField& RenderPassReflection::addOutput(std::string name, std::string description)
{
    return addTextureOutput(std::move(name), std::move(description));
}

RenderGraphField& RenderPassReflection::addTextureInput(std::string name, std::string description)
{
    fields_.push_back(RenderGraphField{
        .name = std::move(name),
        .description = std::move(description),
        .visibility = RenderGraphFieldVisibility::Input,
        .resourceType = RenderGraphResourceType::Texture2D,
        .access = RenderGraphResourceAccess::TextureSampleRead,
        .bindlessAccess = RenderGraphBindlessAccess::None,
        .format = Format::Rgba8Unorm,
        .usage = TextureUsageBits::Sampled,
        .bufferUsage = BufferUsageBits::None,
        .bufferViewType = BufferViewType::Raw,
        .state = ResourceState::ShaderRead,
    });
    return fields_.back();
}

RenderGraphField& RenderPassReflection::addTextureOutput(std::string name, std::string description)
{
    fields_.push_back(RenderGraphField{
        .name = std::move(name),
        .description = std::move(description),
        .visibility = RenderGraphFieldVisibility::Output,
        .resourceType = RenderGraphResourceType::Texture2D,
        .access = RenderGraphResourceAccess::TextureColorWrite,
        .bindlessAccess = RenderGraphBindlessAccess::None,
        .format = Format::Rgba8Unorm,
        .usage = TextureUsageBits::ColorAttachment,
        .bufferUsage = BufferUsageBits::None,
        .bufferViewType = BufferViewType::Raw,
        .state = ResourceState::ColorAttachment,
    });
    return fields_.back();
}

RenderGraphField& RenderPassReflection::addBufferInput(std::string name, std::string description)
{
    fields_.push_back(RenderGraphField{
        .name = std::move(name),
        .description = std::move(description),
        .visibility = RenderGraphFieldVisibility::Input,
        .resourceType = RenderGraphResourceType::Buffer,
        .access = RenderGraphResourceAccess::BufferShaderRead,
        .bindlessAccess = RenderGraphBindlessAccess::None,
        .format = Format::Unknown,
        .usage = TextureUsageBits::None,
        .bufferUsage = BufferUsageBits::Storage,
        .bufferViewType = BufferViewType::Raw,
        .state = ResourceState::ShaderRead,
    });
    return fields_.back();
}

RenderGraphField& RenderPassReflection::addBufferOutput(std::string name, std::string description)
{
    fields_.push_back(RenderGraphField{
        .name = std::move(name),
        .description = std::move(description),
        .visibility = RenderGraphFieldVisibility::Output,
        .resourceType = RenderGraphResourceType::Buffer,
        .access = RenderGraphResourceAccess::BufferStorageReadWrite,
        .bindlessAccess = RenderGraphBindlessAccess::None,
        .format = Format::Unknown,
        .usage = TextureUsageBits::None,
        .bufferUsage = BufferUsageBits::Storage,
        .bufferViewType = BufferViewType::ReadWriteRaw,
        .state = ResourceState::General,
    });
    return fields_.back();
}

const RenderGraphField* RenderPassReflection::findField(
    std::string_view name,
    RenderGraphFieldVisibility visibility) const
{
    const auto iter = std::find_if(
        fields_.begin(),
        fields_.end(),
        [name, visibility](const RenderGraphField& field) {
            return field.visibility == visibility && field.name == name;
        });
    return iter == fields_.end() ? nullptr : &(*iter);
}

const char* renderGraphPassKindName(RenderGraphPassKind kind)
{
    switch (kind) {
    case RenderGraphPassKind::Raster:
        return "Raster";
    case RenderGraphPassKind::Compute:
        return "Compute";
    case RenderGraphPassKind::Unsafe:
        return "Unsafe";
    }

    return "Unknown";
}

RenderGraphPassKind RenderGraphPass::kind() const
{
    return RenderGraphPassKind::Unsafe;
}

QueueType RenderGraphPass::queueType() const
{
    switch (kind()) {
    case RenderGraphPassKind::Compute:
        return QueueType::Compute;
    case RenderGraphPassKind::Raster:
    case RenderGraphPassKind::Unsafe:
        return QueueType::Graphics;
    }

    return QueueType::Graphics;
}

Result RenderGraphPass::compile(const RenderGraphCompileContext&, std::string&)
{
    return {};
}

RenderGraphPassKind RasterPass::kind() const
{
    return RenderGraphPassKind::Raster;
}

QueueType RasterPass::queueType() const
{
    return QueueType::Graphics;
}

RenderGraphPassKind ComputePass::kind() const
{
    return RenderGraphPassKind::Compute;
}

QueueType ComputePass::queueType() const
{
    return QueueType::Compute;
}

RenderGraphPassKind UnsafePass::kind() const
{
    return RenderGraphPassKind::Unsafe;
}

QueueType UnsafePass::queueType() const
{
    return QueueType::Graphics;
}

TextureHandle::TextureHandle(RenderGraphResource* resource)
    : resource_(resource)
{
}

bool TextureHandle::valid() const
{
    return resource_ != nullptr &&
        resource_->type == RenderGraphResourceType::Texture2D &&
        resource_->texture != nullptr &&
        resource_->view != nullptr;
}

Texture* TextureHandle::texture() const
{
    return valid() ? resource_->texture : nullptr;
}

TextureView* TextureHandle::view() const
{
    return valid() ? resource_->view : nullptr;
}

const TextureDesc& TextureHandle::desc() const
{
    static const TextureDesc kEmptyDesc;
    return valid() ? resource_->desc : kEmptyDesc;
}

const BindlessHandle& TextureHandle::bindlessHandle() const
{
    static const BindlessHandle kEmptyHandle;
    return resource_ != nullptr ? resource_->bindlessHandle : kEmptyHandle;
}

BufferHandle::BufferHandle(RenderGraphResource* resource)
    : resource_(resource)
{
}

bool BufferHandle::valid() const
{
    return resource_ != nullptr &&
        resource_->type == RenderGraphResourceType::Buffer &&
        resource_->buffer != nullptr;
}

Buffer* BufferHandle::buffer() const
{
    return valid() ? resource_->buffer : nullptr;
}

BufferView* BufferHandle::view() const
{
    return valid() ? resource_->bufferView : nullptr;
}

const BufferDesc& BufferHandle::desc() const
{
    static const BufferDesc kEmptyDesc;
    return valid() ? resource_->bufferDesc : kEmptyDesc;
}

const BufferViewDesc& BufferHandle::viewDesc() const
{
    static const BufferViewDesc kEmptyDesc;
    return valid() ? resource_->bufferViewDesc : kEmptyDesc;
}

const BindlessHandle& BufferHandle::bindlessHandle() const
{
    static const BindlessHandle kEmptyHandle;
    return resource_ != nullptr ? resource_->bindlessHandle : kEmptyHandle;
}

RenderGraphExecutionContext::RenderGraphExecutionContext(
    CommandBuffer& commandBuffer,
    uint32_t width,
    uint32_t height,
    std::string passName,
    const RenderGraphProperties& properties,
    std::vector<Binding> bindings,
    HistoryResourceManager* historyResources)
    : commandBuffer_(commandBuffer)
    , width_(width)
    , height_(height)
    , passName_(std::move(passName))
    , properties_(properties)
    , bindings_(std::move(bindings))
    , historyResources_(historyResources)
{
}

RenderGraphResource* RenderGraphExecutionContext::resource(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.fieldName == fieldName;
        });
    return iter == bindings_.end() ? nullptr : iter->resource;
}

RenderGraphResource* RenderGraphExecutionContext::input(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.visibility == RenderGraphFieldVisibility::Input &&
                binding.fieldName == fieldName;
        });
    return iter == bindings_.end() ? nullptr : iter->resource;
}

RenderGraphResource* RenderGraphExecutionContext::output(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.visibility == RenderGraphFieldVisibility::Output &&
                binding.fieldName == fieldName;
        });
    return iter == bindings_.end() ? nullptr : iter->resource;
}

TextureHandle RenderGraphExecutionContext::texture(std::string_view fieldName) const
{
    RenderGraphResource* found = resource(fieldName);
    return found != nullptr && found->type == RenderGraphResourceType::Texture2D
        ? TextureHandle(found)
        : TextureHandle();
}

TextureHandle RenderGraphExecutionContext::inputTexture(std::string_view fieldName) const
{
    RenderGraphResource* found = input(fieldName);
    return found != nullptr && found->type == RenderGraphResourceType::Texture2D
        ? TextureHandle(found)
        : TextureHandle();
}

TextureHandle RenderGraphExecutionContext::outputTexture(std::string_view fieldName) const
{
    RenderGraphResource* found = output(fieldName);
    return found != nullptr && found->type == RenderGraphResourceType::Texture2D
        ? TextureHandle(found)
        : TextureHandle();
}

BufferHandle RenderGraphExecutionContext::buffer(std::string_view fieldName) const
{
    RenderGraphResource* found = resource(fieldName);
    return found != nullptr && found->type == RenderGraphResourceType::Buffer
        ? BufferHandle(found)
        : BufferHandle();
}

BufferHandle RenderGraphExecutionContext::inputBuffer(std::string_view fieldName) const
{
    RenderGraphResource* found = input(fieldName);
    return found != nullptr && found->type == RenderGraphResourceType::Buffer
        ? BufferHandle(found)
        : BufferHandle();
}

BufferHandle RenderGraphExecutionContext::outputBuffer(std::string_view fieldName) const
{
    RenderGraphResource* found = output(fieldName);
    return found != nullptr && found->type == RenderGraphResourceType::Buffer
        ? BufferHandle(found)
        : BufferHandle();
}

const BindlessHandle* RenderGraphExecutionContext::bindlessResource(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.fieldName == fieldName;
        });
    if (iter == bindings_.end() ||
        iter->bindlessAccess == RenderGraphBindlessAccess::None ||
        !iter->bindlessHandle.valid()) {
        return nullptr;
    }
    return &iter->bindlessHandle;
}

const BindlessHandle* RenderGraphExecutionContext::bindlessInput(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.visibility == RenderGraphFieldVisibility::Input &&
                binding.fieldName == fieldName;
        });
    if (iter == bindings_.end() ||
        iter->bindlessAccess == RenderGraphBindlessAccess::None ||
        !iter->bindlessHandle.valid()) {
        return nullptr;
    }
    return &iter->bindlessHandle;
}

bool registerRenderGraphPassType(
    std::string type,
    std::string description,
    RenderGraphPassFactory factory)
{
    if (type.empty() || !factory) {
        return false;
    }
    passRegistry()[std::move(type)] = RenderGraphPassRegistryEntry{
        .description = std::move(description),
        .factory = std::move(factory),
    };
    return true;
}

std::unique_ptr<RenderGraphPass> createRenderGraphPass(std::string_view type)
{
    registerBuiltInRenderGraphPasses();
    const auto iter = passRegistry().find(std::string(type));
    if (iter == passRegistry().end() || !iter->second.factory) {
        return {};
    }
    return iter->second.factory();
}

std::vector<RenderGraphPassInfo> listRenderGraphPassTypes()
{
    registerBuiltInRenderGraphPasses();
    std::vector<RenderGraphPassInfo> passTypes;
    passTypes.reserve(passRegistry().size());
    for (const auto& [type, entry] : passRegistry()) {
        std::unique_ptr<RenderGraphPass> pass = entry.factory ? entry.factory() : nullptr;
        passTypes.push_back(RenderGraphPassInfo{
            .type = type,
            .description = entry.description,
            .kind = pass != nullptr ? pass->kind() : RenderGraphPassKind::Unsafe,
            .queueType = pass != nullptr ? pass->queueType() : QueueType::Graphics,
        });
    }
    std::sort(
        passTypes.begin(),
        passTypes.end(),
        [](const RenderGraphPassInfo& lhs, const RenderGraphPassInfo& rhs) {
            return lhs.type < rhs.type;
        });
    return passTypes;
}

RenderGraph::RenderGraph()
{
    registerBuiltInRenderGraphPasses();
}

void RenderGraph::setName(std::string name)
{
    if (name.empty()) {
        name = "RenderGraph";
    }
    if (name_ != name) {
        name_ = std::move(name);
        markDirty();
    }
}

const RenderGraphNode* RenderGraph::findNode(std::string_view name) const
{
    return findNodeByName(nodes_, name);
}

RenderGraphNode* RenderGraph::findNode(std::string_view name)
{
    return const_cast<RenderGraphNode*>(static_cast<const RenderGraph*>(this)->findNode(name));
}

const RenderGraphNode* RenderGraph::findNode(uint32_t id) const
{
    const auto iter = std::find_if(
        nodes_.begin(),
        nodes_.end(),
        [id](const RenderGraphNode& node) {
            return node.id == id;
        });
    return iter == nodes_.end() ? nullptr : &(*iter);
}

RenderGraphNode* RenderGraph::findNode(uint32_t id)
{
    return const_cast<RenderGraphNode*>(static_cast<const RenderGraph*>(this)->findNode(id));
}

const RenderGraphEdge* RenderGraph::findEdge(uint32_t id) const
{
    const auto iter = std::find_if(
        edges_.begin(),
        edges_.end(),
        [id](const RenderGraphEdge& edge) {
            return edge.id == id;
        });
    return iter == edges_.end() ? nullptr : &(*iter);
}

RenderGraphNode* RenderGraph::addNode(
    std::string type,
    std::string name,
    RenderGraphProperties properties,
    float uiX,
    float uiY)
{
    if (type.empty() || name.empty() || nodeNameExists(nodes_, name)) {
        return nullptr;
    }
    nodes_.push_back(RenderGraphNode{
        .id = nextNodeId_++,
        .name = std::move(name),
        .type = std::move(type),
        .properties = std::move(properties),
        .uiX = uiX,
        .uiY = uiY,
    });
    markDirty();
    return &nodes_.back();
}

bool RenderGraph::removeNode(uint32_t id)
{
    const RenderGraphNode* node = findNode(id);
    if (node == nullptr) {
        return false;
    }
    const std::string nodeName = node->name;
    nodes_.erase(
        std::remove_if(
            nodes_.begin(),
            nodes_.end(),
            [id](const RenderGraphNode& candidate) { return candidate.id == id; }),
        nodes_.end());
    edges_.erase(
        std::remove_if(
            edges_.begin(),
            edges_.end(),
            [&nodeName](const RenderGraphEdge& edge) {
                return edge.srcPass == nodeName || edge.dstPass == nodeName;
            }),
        edges_.end());
    outputs_.erase(
        std::remove_if(
            outputs_.begin(),
            outputs_.end(),
            [&nodeName](const RenderGraphOutput& output) {
                return output.passName == nodeName;
            }),
        outputs_.end());
    markDirty();
    return true;
}

bool RenderGraph::renameNode(uint32_t id, std::string newName)
{
    if (newName.empty() || nodeNameExists(nodes_, newName, id)) {
        return false;
    }
    RenderGraphNode* node = findNode(id);
    if (node == nullptr || node->name == newName) {
        return node != nullptr;
    }
    const std::string oldName = node->name;
    node->name = std::move(newName);
    for (RenderGraphEdge& edge : edges_) {
        if (edge.srcPass == oldName) {
            edge.srcPass = node->name;
        }
        if (edge.dstPass == oldName) {
            edge.dstPass = node->name;
        }
    }
    for (RenderGraphOutput& output : outputs_) {
        if (output.passName == oldName) {
            output.passName = node->name;
        }
    }
    markDirty();
    return true;
}

bool RenderGraph::setNodeProperties(uint32_t id, RenderGraphProperties properties)
{
    RenderGraphNode* node = findNode(id);
    if (node == nullptr) {
        return false;
    }
    node->properties = std::move(properties);
    markDirty();
    return true;
}

bool RenderGraph::setNodePosition(uint32_t id, float uiX, float uiY)
{
    RenderGraphNode* node = findNode(id);
    if (node == nullptr) {
        return false;
    }
    if (node->uiX == uiX && node->uiY == uiY) {
        return true;
    }
    node->uiX = uiX;
    node->uiY = uiY;
    return true;
}

RenderGraphEdge* RenderGraph::addEdge(std::string src, std::string dst)
{
    std::string srcPass;
    std::string srcField;
    std::string dstPass;
    std::string dstField;
    if (!splitRenderGraphFieldName(src, srcPass, srcField) ||
        !splitRenderGraphFieldName(dst, dstPass, dstField)) {
        return nullptr;
    }

    const auto exists = std::any_of(
        edges_.begin(),
        edges_.end(),
        [&](const RenderGraphEdge& edge) {
            return edge.srcPass == srcPass &&
                edge.srcField == srcField &&
                edge.dstPass == dstPass &&
                edge.dstField == dstField;
        });
    if (exists) {
        return nullptr;
    }

    edges_.push_back(RenderGraphEdge{
        .id = nextEdgeId_++,
        .srcPass = std::move(srcPass),
        .srcField = std::move(srcField),
        .dstPass = std::move(dstPass),
        .dstField = std::move(dstField),
    });
    markDirty();
    return &edges_.back();
}

bool RenderGraph::removeEdge(uint32_t id)
{
    const auto oldSize = edges_.size();
    edges_.erase(
        std::remove_if(
            edges_.begin(),
            edges_.end(),
            [id](const RenderGraphEdge& edge) { return edge.id == id; }),
        edges_.end());
    if (edges_.size() == oldSize) {
        return false;
    }
    markDirty();
    return true;
}

bool RenderGraph::markOutput(std::string output)
{
    std::string passName;
    std::string fieldName;
    if (!splitRenderGraphFieldName(output, passName, fieldName)) {
        return false;
    }
    const auto exists = std::any_of(
        outputs_.begin(),
        outputs_.end(),
        [&](const RenderGraphOutput& candidate) {
            return candidate.passName == passName && candidate.fieldName == fieldName;
        });
    if (exists) {
        return true;
    }
    outputs_.push_back(RenderGraphOutput{
        .passName = std::move(passName),
        .fieldName = std::move(fieldName),
    });
    markDirty();
    return true;
}

bool RenderGraph::unmarkOutput(std::string output)
{
    std::string passName;
    std::string fieldName;
    if (!splitRenderGraphFieldName(output, passName, fieldName)) {
        return false;
    }
    const auto oldSize = outputs_.size();
    outputs_.erase(
        std::remove_if(
            outputs_.begin(),
            outputs_.end(),
            [&](const RenderGraphOutput& candidate) {
                return candidate.passName == passName && candidate.fieldName == fieldName;
            }),
        outputs_.end());
    if (outputs_.size() == oldSize) {
        return false;
    }
    markDirty();
    return true;
}

void RenderGraph::clearOutputs()
{
    if (!outputs_.empty()) {
        outputs_.clear();
        markDirty();
    }
}

bool RenderGraph::validate(std::string& log) const
{
    registerBuiltInRenderGraphPasses();
    log.clear();

    if (nodes_.empty()) {
        log = validationPrefix("graph has no nodes");
        return false;
    }
    if (outputs_.empty()) {
        log = validationPrefix("graph has no marked output");
        return false;
    }

    std::unordered_set<uint32_t> ids;
    std::unordered_set<std::string> names;
    std::unordered_map<std::string, RenderPassReflection> reflections;
    const RenderGraphCompileContext reflectContext{};

    for (const RenderGraphNode& node : nodes_) {
        if (node.id == 0 || !ids.insert(node.id).second) {
            log = validationPrefix("duplicate node id");
            return false;
        }
        if (node.name.empty() || !names.insert(node.name).second) {
            log = validationPrefix("duplicate or empty node name");
            return false;
        }
        std::unique_ptr<RenderGraphPass> pass = createRenderGraphPass(node.type);
        if (pass == nullptr) {
            log = validationPrefix(std::string("unknown pass type '") + node.type + "'");
            return false;
        }
        pass->setProperties(node.properties);
        RenderPassReflection reflection = pass->reflect(reflectContext);
        for (const RenderGraphField& field : reflection.fields()) {
            if (!accessMatchesResourceType(field.access, field.resourceType)) {
                log = validationPrefix(
                    std::string("field access does not match resource type '") +
                    makeRenderGraphFieldName(node.name, field.name) +
                    "'");
                return false;
            }
            if (field.bindlessAccess == RenderGraphBindlessAccess::SampledImage &&
                field.resourceType != RenderGraphResourceType::Texture2D) {
                log = validationPrefix(
                    std::string("sampled image bindless access requires a texture field '") +
                    makeRenderGraphFieldName(node.name, field.name) +
                    "'");
                return false;
            }
            if (field.bindlessAccess == RenderGraphBindlessAccess::Buffer &&
                field.resourceType != RenderGraphResourceType::Buffer) {
                log = validationPrefix(
                    std::string("buffer bindless access requires a buffer field '") +
                    makeRenderGraphFieldName(node.name, field.name) +
                    "'");
                return false;
            }
            if (field.visibility == RenderGraphFieldVisibility::Output &&
                field.resourceType == RenderGraphResourceType::Buffer &&
                field.size == 0) {
                log = validationPrefix(
                    std::string("buffer output has zero size '") +
                    makeRenderGraphFieldName(node.name, field.name) +
                    "'");
                return false;
            }
        }
        reflections.emplace(node.name, std::move(reflection));
    }

    for (const RenderGraphOutput& output : outputs_) {
        const auto iter = reflections.find(output.passName);
        if (iter == reflections.end() ||
            iter->second.findField(output.fieldName, RenderGraphFieldVisibility::Output) == nullptr) {
            log = validationPrefix(
                std::string("invalid output '") +
                makeRenderGraphFieldName(output.passName, output.fieldName) +
                "'");
            return false;
        }
    }

    for (const RenderGraphEdge& edge : edges_) {
        const auto src = reflections.find(edge.srcPass);
        const auto dst = reflections.find(edge.dstPass);
        const RenderGraphField* srcField = src == reflections.end()
            ? nullptr
            : src->second.findField(edge.srcField, RenderGraphFieldVisibility::Output);
        if (srcField == nullptr) {
            log = validationPrefix(
                std::string("invalid edge source '") +
                makeRenderGraphFieldName(edge.srcPass, edge.srcField) +
                "'");
            return false;
        }
        const RenderGraphField* dstField = dst == reflections.end()
            ? nullptr
            : dst->second.findField(edge.dstField, RenderGraphFieldVisibility::Input);
        if (dstField == nullptr) {
            log = validationPrefix(
                std::string("invalid edge destination '") +
                makeRenderGraphFieldName(edge.dstPass, edge.dstField) +
                "'");
            return false;
        }
        if (srcField->resourceType != dstField->resourceType) {
            log = validationPrefix(
                std::string("edge resource type mismatch '") +
                makeRenderGraphFieldName(edge.srcPass, edge.srcField) +
                "' -> '" +
                makeRenderGraphFieldName(edge.dstPass, edge.dstField) +
                "'");
            return false;
        }
    }

    for (const auto& [passName, reflection] : reflections) {
        for (const RenderGraphField& field : reflection.fields()) {
            if (field.visibility != RenderGraphFieldVisibility::Input || field.optional) {
                continue;
            }
            const bool connected = std::any_of(
                edges_.begin(),
                edges_.end(),
                [&](const RenderGraphEdge& edge) {
                    return edge.dstPass == passName && edge.dstField == field.name;
                });
            if (!connected) {
                log = validationPrefix(
                    std::string("required input is not connected '") +
                    makeRenderGraphFieldName(passName, field.name) +
                    "'");
                return false;
            }
        }
    }

    if (!validateAcyclic(nodes_, edges_, log)) {
        return false;
    }

    log = "RenderGraph is valid";
    return true;
}

void RenderGraph::clear()
{
    name_ = "RenderGraph";
    nodes_.clear();
    edges_.clear();
    outputs_.clear();
    nextNodeId_ = 1;
    nextEdgeId_ = 1;
    markDirty();
}

std::string RenderGraph::firstOutputName() const
{
    if (outputs_.empty()) {
        return {};
    }
    return makeRenderGraphFieldName(outputs_.front().passName, outputs_.front().fieldName);
}

RenderGraph RenderGraph::createDefaultTriangleGraph()
{
    RenderGraph graph;
    graph.setName("DefaultTriangle");
    graph.addNode("TriangleRasterPass", "Triangle", RenderGraphProperties::object(), 40.0f, 80.0f);
    graph.markOutput("Triangle.color");
    graph.clearDirty();
    return graph;
}

RenderGraph RenderGraph::createDefaultBunnyGraph()
{
    RenderGraph graph;
    graph.setName("StanfordBunnyWireframe");
    graph.addNode(
        "BunnyWireframePass",
        "Bunny",
        RenderGraphProperties{
            {"path", "Asset/StandfordBunny/scene.gltf"},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 60.0f},
                {"znear", 0.1f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {-0.0168404f, 0.110154f, 0.22f}},
                {"center", {-0.0168404f, 0.110154f, -0.00153695f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        },
        40.0f,
        80.0f);
    graph.markOutput("Bunny.color");
    graph.clearDirty();
    return graph;
}

bool splitRenderGraphFieldName(
    std::string_view fullName,
    std::string& outPassName,
    std::string& outFieldName)
{
    const size_t separator = fullName.find('.');
    if (separator == std::string_view::npos ||
        separator == 0 ||
        separator + 1 >= fullName.size()) {
        return false;
    }
    outPassName = std::string(fullName.substr(0, separator));
    outFieldName = std::string(fullName.substr(separator + 1));
    return true;
}

std::string makeRenderGraphFieldName(std::string_view passName, std::string_view fieldName)
{
    std::string fullName(passName);
    fullName += '.';
    fullName += fieldName;
    return fullName;
}

struct RenderGraphExecutor::Impl {
    struct ResourceSlot {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> textureView;
        std::unique_ptr<Buffer> buffer;
        std::unique_ptr<BufferView> bufferView;
        RenderGraphResource resource;
    };

    struct CompiledNode {
        uint32_t id = 0;
        std::string name;
        std::string type;
        RenderGraphPassKind kind = RenderGraphPassKind::Unsafe;
        QueueType queueType = QueueType::Graphics;
        RenderGraphProperties properties = RenderGraphProperties::object();
        std::unique_ptr<RenderGraphPass> pass;
        RenderPassReflection reflection;
    };

    struct QueueCommandContext {
        Queue* queue = nullptr;
        std::unique_ptr<CommandPool> commandPool;
        bool resetForCurrentSubmit = false;
    };

    struct SubmissionSegment {
        QueueType queueType = QueueType::Graphics;
        Queue* queue = nullptr;
        CommandBuffer* commandBuffer = nullptr;
    };

    Device* device = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    Format defaultFormat = Format::Rgba8Unorm;
    HistoryResourceManager* historyResources = nullptr;
    std::vector<CompiledNode> executionList;
    std::unordered_map<std::string, ResourceSlot> resources;
    std::unordered_map<std::string, std::string> inputAliases;
    std::unique_ptr<BindlessHeap> bindlessHeap;
    std::array<QueueCommandContext, 3> queueCommandContexts;
    std::vector<std::unique_ptr<CommandBuffer>> submittedCommandBuffers;
    std::vector<std::unique_ptr<Semaphore>> submittedSemaphores;
    std::vector<std::unique_ptr<Fence>> submittedFences;
    bool hasSubmittedWork = false;
    bool isCompiled = false;

    RenderGraphResource* resource(std::string_view fullName)
    {
        auto iter = resources.find(std::string(fullName));
        return iter == resources.end() ? nullptr : &iter->second.resource;
    }

    const RenderGraphResource* resource(std::string_view fullName) const
    {
        auto iter = resources.find(std::string(fullName));
        return iter == resources.end() ? nullptr : &iter->second.resource;
    }

    const CompiledNode* compiledNode(std::string_view name) const
    {
        const auto iter = std::find_if(
            executionList.begin(),
            executionList.end(),
            [name](const CompiledNode& node) {
                return node.name == name;
            });
        return iter == executionList.end() ? nullptr : &(*iter);
    }

    const RenderGraphField* reflectedField(
        std::string_view passName,
        std::string_view fieldName,
        RenderGraphFieldVisibility visibility) const
    {
        const CompiledNode* node = compiledNode(passName);
        if (node == nullptr) {
            return nullptr;
        }
        return node->reflection.findField(fieldName, visibility);
    }

    static bool usesBindlessResource(const CompiledNode& node)
    {
        return std::any_of(
            node.reflection.fields().begin(),
            node.reflection.fields().end(),
            [](const RenderGraphField& field) {
                return isBindlessField(field);
            });
    }

    static size_t queueContextIndex(QueueType type)
    {
        switch (type) {
        case QueueType::Graphics:
            return 0;
        case QueueType::Compute:
            return 1;
        case QueueType::Copy:
            return 2;
        }

        return 0;
    }

    QueueCommandContext& queueCommandContext(QueueType type)
    {
        return queueCommandContexts[queueContextIndex(type)];
    }

    Result waitForSubmittedWork(uint64_t timeoutNanoseconds)
    {
        if (!hasSubmittedWork) {
            return {};
        }

        for (const std::unique_ptr<Fence>& fence : submittedFences) {
            if (fence == nullptr) {
                continue;
            }
            Result result = fence->wait(timeoutNanoseconds);
            if (!result) {
                return result;
            }
        }

        hasSubmittedWork = false;
        submittedCommandBuffers.clear();
        submittedSemaphores.clear();
        submittedFences.clear();
        return {};
    }

    Result prepareCommandPool(QueueType type, Queue& queue, CommandPool*& outCommandPool)
    {
        outCommandPool = nullptr;
        QueueCommandContext& context = queueCommandContext(type);
        if (context.queue != &queue || context.commandPool == nullptr) {
            context.commandPool.reset();
            Result result = device->createCommandPool(queue, context.commandPool);
            if (!result || context.commandPool == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }
            context.queue = &queue;
        }

        if (!context.resetForCurrentSubmit) {
            Result result = context.commandPool->reset();
            if (!result) {
                return result;
            }
            context.resetForCurrentSubmit = true;
        }

        outCommandPool = context.commandPool.get();
        return {};
    }

    bool hasCrossQueueResourceEdges(std::string& log) const
    {
        for (const auto& [inputName, outputName] : inputAliases) {
            std::string inputPass;
            std::string inputField;
            std::string outputPass;
            std::string outputField;
            if (!splitRenderGraphFieldName(inputName, inputPass, inputField) ||
                !splitRenderGraphFieldName(outputName, outputPass, outputField)) {
                continue;
            }

            const CompiledNode* inputNode = compiledNode(inputPass);
            const CompiledNode* outputNode = compiledNode(outputPass);
            if (inputNode == nullptr || outputNode == nullptr) {
                continue;
            }

            if (inputNode->queueType != outputNode->queueType) {
                log = std::string("RenderGraph multi-queue submission does not yet support "
                    "cross-queue resource edges: ") +
                    outputName +
                    " (" +
                    queueTypeName(outputNode->queueType) +
                    ") -> " +
                    inputName +
                    " (" +
                    queueTypeName(inputNode->queueType) +
                    ")";
                return true;
            }
        }

        return false;
    }

    Result transition(
        CommandBuffer& commandBuffer,
        RenderGraphResource& resource,
        ResourceState state,
        RenderGraphResourceAccess access)
    {
        const bool needsSameStateStorageBarrier =
            resource.state == state &&
            state == ResourceState::General &&
            (accessWrites(resource.lastAccess) || accessWrites(access));
        if (resource.state == state && !needsSameStateStorageBarrier) {
            resource.lastAccess = access;
            return {};
        }

        if (resource.type == RenderGraphResourceType::Texture2D) {
            if (resource.texture == nullptr) {
                return {};
            }
            TextureBarrierDesc barrier{
                .texture = resource.texture,
                .before = resource.state,
                .after = state,
                .baseMip = 0,
                .mipCount = resource.desc.mipCount,
                .baseLayer = 0,
                .layerCount = resource.desc.layerCount,
            };
            commandBuffer.barrier(BarrierDesc{
                .textures = &barrier,
                .textureCount = 1,
            });
        } else {
            if (resource.buffer == nullptr) {
                return {};
            }
            BufferBarrierDesc barrier{
                .buffer = resource.buffer,
                .before = resource.state,
                .after = state,
                .offset = 0,
                .size = resource.bufferDesc.size,
            };
            commandBuffer.barrier(BarrierDesc{
                .buffers = &barrier,
                .bufferCount = 1,
            });
        }

        resource.state = state;
        resource.lastAccess = access;
        return {};
    }

    Result executeNode(CommandBuffer& commandBuffer, CompiledNode& node)
    {
        std::vector<RenderGraphExecutionContext::Binding> bindings;

        for (const RenderGraphField& field : node.reflection.fields()) {
            const std::string localName = field.name;
            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            RenderGraphResource* resource = nullptr;

            if (field.visibility == RenderGraphFieldVisibility::Output) {
                resource = this->resource(fullName);
                if (resource != nullptr) {
                    Result result = transition(
                        commandBuffer,
                        *resource,
                        stateForAccess(field.access),
                        field.access);
                    if (!result) {
                        return result;
                    }
                }
            } else {
                const auto alias = inputAliases.find(fullName);
                if (alias != inputAliases.end()) {
                    resource = this->resource(alias->second);
                    if (resource != nullptr) {
                        Result result = transition(
                            commandBuffer,
                            *resource,
                            stateForAccess(field.access),
                            field.access);
                        if (!result) {
                            return result;
                        }
                    }
                }
            }

            bindings.push_back(RenderGraphExecutionContext::Binding{
                .fieldName = localName,
                .resource = resource,
                .visibility = field.visibility,
                .bindlessAccess = field.bindlessAccess,
                .bindlessHandle = resource != nullptr
                    ? resource->bindlessHandle
                    : BindlessHandle{},
                .sampledImageBindlessHandle = resource != nullptr
                    ? resource->sampledImageBindlessHandle
                    : BindlessHandle{},
            });
        }

        if (bindlessHeap != nullptr && usesBindlessResource(node)) {
            commandBuffer.bindBindlessHeap(*bindlessHeap);
        }

        RenderGraphExecutionContext context(
            commandBuffer,
            width,
            height,
            node.name,
            node.properties,
            std::move(bindings),
            historyResources);
        const std::string markerName = passProfileMarkerName(node.name, node.type);
        const uint32_t markerColor = profiling::nsightColorFromName(node.type);
        const profiling::NsightProfileRange passMarker(
            markerName.c_str(),
            markerColor,
            node.id);
        commandBuffer.beginDebugLabel(DebugLabelDesc{
            .name = markerName.c_str(),
            .color = debugLabelColorFromArgb(markerColor),
        });
        Result result = node.pass->execute(context);
        commandBuffer.endDebugLabel();
        return result;
    }
};

RenderGraphExecutor::RenderGraphExecutor()
    : impl_(std::make_unique<Impl>())
{
}

RenderGraphExecutor::~RenderGraphExecutor()
{
    if (impl_ != nullptr) {
        (void)impl_->waitForSubmittedWork(UINT64_MAX);
    }
}
RenderGraphExecutor::RenderGraphExecutor(RenderGraphExecutor&&) noexcept = default;
RenderGraphExecutor& RenderGraphExecutor::operator=(RenderGraphExecutor&&) noexcept = default;

Result RenderGraphExecutor::compile(
    Device& device,
    const RenderGraph& graph,
    uint32_t width,
    uint32_t height,
    std::string& log)
{
    if (width == 0 || height == 0) {
        log = validationPrefix("invalid default dimensions");
        return makeError(Error::InvalidArgument);
    }

    std::string validationLog;
    if (!graph.validate(validationLog)) {
        log = validationLog;
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }

    ActiveGraph activeGraph;
    if (!buildActiveGraph(graph, activeGraph, log)) {
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }

    Result pendingResult = impl_->waitForSubmittedWork(UINT64_MAX);
    if (!pendingResult) {
        log = resultMessage("RenderGraph waitForSubmittedWork", pendingResult);
        impl_->isCompiled = false;
        return pendingResult;
    }

    if (impl_->device != nullptr && impl_->device != &device) {
        for (Impl::QueueCommandContext& queueContext : impl_->queueCommandContexts) {
            queueContext.queue = nullptr;
            queueContext.commandPool.reset();
            queueContext.resetForCurrentSubmit = false;
        }
    }

    impl_->device = &device;
    impl_->width = width;
    impl_->height = height;
    impl_->executionList.clear();
    impl_->resources.clear();
    impl_->inputAliases.clear();
    impl_->bindlessHeap.reset();
    impl_->isCompiled = false;

    const RenderGraphCompileContext compileContext{
        .device = &device,
        .graphicsQueue = device.getQueue(QueueType::Graphics),
        .width = width,
        .height = height,
        .defaultFormat = impl_->defaultFormat,
    };

    for (const std::string& passName : activeGraph.executionOrder) {
        const RenderGraphNode* node = graph.findNode(passName);
        if (node == nullptr) {
            log = validationPrefix(std::string("active pass is missing '") + passName + "'");
            return makeError(Error::InvalidArgument);
        }

        std::unique_ptr<RenderGraphPass> pass = createRenderGraphPass(node->type);
        if (pass == nullptr) {
            log = validationPrefix(std::string("unknown pass type '") + node->type + "'");
            return makeError(Error::InvalidArgument);
        }
        pass->setProperties(node->properties);
        const RenderGraphPassKind kind = pass->kind();
        const QueueType queueType = pass->queueType();
        RenderPassReflection reflection = pass->reflect(compileContext);
        impl_->executionList.push_back(Impl::CompiledNode{
            .id = node->id,
            .name = node->name,
            .type = node->type,
            .kind = kind,
            .queueType = queueType,
            .properties = node->properties,
            .pass = std::move(pass),
            .reflection = std::move(reflection),
        });
    }

    for (const RenderGraphEdge& edge : graph.edges()) {
        if (!activeGraph.activePasses.contains(edge.srcPass) ||
            !activeGraph.activePasses.contains(edge.dstPass)) {
            continue;
        }
        impl_->inputAliases.emplace(
            makeRenderGraphFieldName(edge.dstPass, edge.dstField),
            makeRenderGraphFieldName(edge.srcPass, edge.srcField));
    }

    std::vector<std::string> bindlessSampledImageResources;
    std::vector<std::string> bindlessBufferResources;
    std::unordered_set<std::string> bindlessSampledImageResourceSet;
    std::unordered_set<std::string> bindlessBufferResourceSet;
    for (const Impl::CompiledNode& node : impl_->executionList) {
        for (const RenderGraphField& field : node.reflection.fields()) {
            if (!isBindlessField(field)) {
                continue;
            }

            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            std::string resourceName = fullName;
            if (field.visibility == RenderGraphFieldVisibility::Input) {
                const auto alias = impl_->inputAliases.find(fullName);
                if (alias == impl_->inputAliases.end()) {
                    continue;
                }
                resourceName = alias->second;
            }

            if (isBindlessSampledImageField(field) &&
                bindlessSampledImageResourceSet.insert(resourceName).second) {
                bindlessSampledImageResources.push_back(std::move(resourceName));
                continue;
            }
            if (isBindlessBufferField(field) &&
                bindlessBufferResourceSet.insert(resourceName).second) {
                bindlessBufferResources.push_back(std::move(resourceName));
            }
        }
    }

    if ((!bindlessSampledImageResources.empty() || !bindlessBufferResources.empty()) &&
        !device.capabilities().bindlessDescriptorHeap) {
        log = "RenderGraph compile failed: bindless resources require "
            "DeviceCapabilities::bindlessDescriptorHeap";
        return makeError(Error::Unsupported);
    }

    for (Impl::CompiledNode& node : impl_->executionList) {
        Result result = node.pass->compile(compileContext, log);
        if (!result) {
            impl_->isCompiled = false;
            return result;
        }
    }

    for (const Impl::CompiledNode& node : impl_->executionList) {
        for (const RenderGraphField& field : node.reflection.fields()) {
            if (field.visibility != RenderGraphFieldVisibility::Output) {
                continue;
            }

            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            Impl::ResourceSlot slot;

            if (field.resourceType == RenderGraphResourceType::Texture2D) {
                TextureUsageBits usage = textureUsageForField(field);
                if (usage == TextureUsageBits::None) {
                    usage = TextureUsageBits::ColorAttachment;
                }
                if (isOutputMarked(graph, fullName)) {
                    usage = addTextureUsage(usage, TextureUsageBits::TransferSource);
                    usage = addTextureUsage(usage, TextureUsageBits::Sampled);
                }
                for (const RenderGraphEdge& edge : graph.edges()) {
                    if (edge.srcPass != node.name ||
                        edge.srcField != field.name ||
                        !activeGraph.activePasses.contains(edge.dstPass)) {
                        continue;
                    }

                    const RenderGraphField* dstField = impl_->reflectedField(
                        edge.dstPass,
                        edge.dstField,
                        RenderGraphFieldVisibility::Input);
                    if (dstField != nullptr) {
                        usage = addTextureUsage(usage, textureUsageForField(*dstField));
                    }
                }

                TextureDesc desc{
                    .type = TextureType::Texture2D,
                    .usage = usage,
                    .format = resolveFormat(field.format, impl_->defaultFormat),
                    .width = field.width == 0 ? width : field.width,
                    .height = field.height == 0 ? height : field.height,
                    .depth = 1,
                    .mipCount = 1,
                    .layerCount = 1,
                    .memoryLocation = MemoryLocation::Device,
                };

                Result result = device.createTexture(desc, slot.texture);
                if (!result || slot.texture == nullptr) {
                    log += resultMessage(std::string("createTexture(") + fullName + ")", result);
                    log += '\n';
                    return result ? makeError(Error::Failure) : result;
                }
                result = device.createTextureView(
                    *slot.texture,
                    TextureViewDesc{
                        .format = desc.format,
                        .baseMip = 0,
                        .mipCount = 1,
                        .baseLayer = 0,
                        .layerCount = 1,
                    },
                    slot.textureView);
                if (!result || slot.textureView == nullptr) {
                    log += resultMessage(std::string("createTextureView(") + fullName + ")", result);
                    log += '\n';
                    return result ? makeError(Error::Failure) : result;
                }
                slot.resource = RenderGraphResource{
                    .type = RenderGraphResourceType::Texture2D,
                    .texture = slot.texture.get(),
                    .view = slot.textureView.get(),
                    .desc = desc,
                    .state = ResourceState::Undefined,
                };
            } else {
                BufferUsageBits usage = bufferUsageForField(field);
                if (usage == BufferUsageBits::None) {
                    usage = BufferUsageBits::Storage;
                }
                BufferViewType viewType = bufferViewTypeForField(field);
                for (const RenderGraphEdge& edge : graph.edges()) {
                    if (edge.srcPass != node.name ||
                        edge.srcField != field.name ||
                        !activeGraph.activePasses.contains(edge.dstPass)) {
                        continue;
                    }

                    const RenderGraphField* dstField = impl_->reflectedField(
                        edge.dstPass,
                        edge.dstField,
                        RenderGraphFieldVisibility::Input);
                    if (dstField == nullptr) {
                        continue;
                    }
                    usage = addBufferUsage(usage, bufferUsageForField(*dstField));
                    if (dstField->access == RenderGraphResourceAccess::BufferStorageReadWrite) {
                        viewType = dstField->structureStride == 0
                            ? BufferViewType::ReadWriteRaw
                            : BufferViewType::ReadWriteStructured;
                    }
                }

                const bool markedBufferOutput = isOutputMarked(graph, fullName);
                BufferDesc desc{
                    .size = field.size,
                    .structureStride = field.structureStride,
                    .usage = usage,
                    .memoryLocation = markedBufferOutput
                        ? MemoryLocation::HostReadback
                        : field.memoryLocation,
                };

                Result result = device.createBuffer(desc, slot.buffer);
                if (!result || slot.buffer == nullptr) {
                    log += resultMessage(std::string("createBuffer(") + fullName + ")", result);
                    log += '\n';
                    return result ? makeError(Error::Failure) : result;
                }

                BufferViewDesc viewDesc{
                    .type = viewType,
                    .offset = 0,
                    .size = desc.size,
                    .structureStride = desc.structureStride,
                };
                const bool needsBindlessBuffer = bindlessBufferResourceSet.contains(fullName);
                if (needsBindlessBuffer) {
                    result = device.createBufferView(*slot.buffer, viewDesc, slot.bufferView);
                    if (!result || slot.bufferView == nullptr) {
                        log += resultMessage(std::string("createBufferView(") + fullName + ")", result);
                        log += '\n';
                        return result ? makeError(Error::Failure) : result;
                    }
                    viewDesc = slot.bufferView->desc();
                }

                slot.resource = RenderGraphResource{
                    .type = RenderGraphResourceType::Buffer,
                    .buffer = slot.buffer.get(),
                    .bufferView = slot.bufferView.get(),
                    .bufferDesc = desc,
                    .bufferViewDesc = viewDesc,
                    .state = ResourceState::Undefined,
                };
            }

            impl_->resources.emplace(fullName, std::move(slot));
        }
    }

    if (!bindlessSampledImageResources.empty() || !bindlessBufferResources.empty()) {
        Result result = device.createBindlessHeap(
            BindlessHeapDesc{
                .maxSampledImages = static_cast<uint32_t>(bindlessSampledImageResources.size()),
                .maxBuffers = static_cast<uint32_t>(bindlessBufferResources.size()),
            },
            impl_->bindlessHeap);
        if (!result || impl_->bindlessHeap == nullptr) {
            log += resultMessage("createBindlessHeap(RenderGraph)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        for (const std::string& fullName : bindlessSampledImageResources) {
            RenderGraphResource* resource = impl_->resource(fullName);
            if (resource == nullptr || resource->view == nullptr) {
                log = validationPrefix(std::string("bindless sampled image resource is missing '") + fullName + "'");
                return makeError(Error::InvalidArgument);
            }

            BindlessHandle handle;
            result = impl_->bindlessHeap->allocateSampledImage(handle);
            if (!result) {
                log += resultMessage(std::string("allocateSampledImage(") + fullName + ")", result);
                log += '\n';
                return result;
            }

            result = impl_->bindlessHeap->writeSampledImage(
                handle,
                *resource->view,
                ResourceState::ShaderRead);
            if (!result) {
                log += resultMessage(std::string("writeSampledImage(") + fullName + ")", result);
                log += '\n';
                return result;
            }
            resource->bindlessHandle = handle;
            resource->sampledImageBindlessHandle = handle;
        }

        for (const std::string& fullName : bindlessBufferResources) {
            RenderGraphResource* resource = impl_->resource(fullName);
            if (resource == nullptr || resource->bufferView == nullptr) {
                log = validationPrefix(std::string("bindless buffer resource is missing '") + fullName + "'");
                return makeError(Error::InvalidArgument);
            }

            BindlessHandle handle;
            result = impl_->bindlessHeap->allocateBuffer(handle);
            if (!result) {
                log += resultMessage(std::string("allocateBuffer(") + fullName + ")", result);
                log += '\n';
                return result;
            }

            result = impl_->bindlessHeap->writeBufferView(handle, *resource->bufferView);
            if (!result) {
                log += resultMessage(std::string("writeBufferView(") + fullName + ")", result);
                log += '\n';
                return result;
            }
            resource->bindlessHandle = handle;
        }
    }

    impl_->isCompiled = true;
    log = "RenderGraph compiled";
    return {};
}

Result RenderGraphExecutor::execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources)
{
    if (!impl_->isCompiled) {
        return makeError(Error::InvalidArgument);
    }

    impl_->historyResources = historyResources;
    for (Impl::CompiledNode& node : impl_->executionList) {
        Result result = impl_->executeNode(commandBuffer, node);
        if (!result) {
            impl_->historyResources = nullptr;
            return result;
        }
    }

    impl_->historyResources = nullptr;
    return {};
}

Result RenderGraphExecutor::execute(const RenderGraphSubmitDesc& desc)
{
    if (!impl_->isCompiled || impl_->device == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    impl_->historyResources = nullptr;
    Result result = impl_->waitForSubmittedWork(UINT64_MAX);
    if (!result) {
        return result;
    }
    impl_->submittedCommandBuffers.clear();
    impl_->submittedSemaphores.clear();
    impl_->submittedFences.clear();

    std::string crossQueueLog;
    if (impl_->hasCrossQueueResourceEdges(crossQueueLog)) {
        return makeError(Error::Unsupported);
    }

    for (Impl::QueueCommandContext& queueContext : impl_->queueCommandContexts) {
        queueContext.resetForCurrentSubmit = false;
    }

    std::vector<Impl::SubmissionSegment> segments;
    CommandBuffer* currentCommandBuffer = nullptr;
    QueueType currentQueueType = QueueType::Graphics;
    bool hasCurrentSegment = false;

    auto endCurrentSegment = [&]() -> Result {
        if (currentCommandBuffer == nullptr) {
            return {};
        }
        Result endResult = currentCommandBuffer->end();
        if (!endResult) {
            return endResult;
        }
        currentCommandBuffer = nullptr;
        hasCurrentSegment = false;
        return {};
    };

    auto beginSegment = [&](QueueType queueType, Queue& queue) -> Result {
        CommandPool* commandPool = nullptr;
        Result prepareResult = impl_->prepareCommandPool(queueType, queue, commandPool);
        if (!prepareResult) {
            return prepareResult;
        }
        if (commandPool == nullptr) {
            return makeError(Error::Failure);
        }

        std::unique_ptr<CommandBuffer> commandBuffer;
        Result createResult = commandPool->createCommandBuffer(commandBuffer);
        if (!createResult || commandBuffer == nullptr) {
            return createResult ? makeError(Error::Failure) : createResult;
        }
        Result beginResult = commandBuffer->begin();
        if (!beginResult) {
            return beginResult;
        }

        currentCommandBuffer = commandBuffer.get();
        currentQueueType = queueType;
        hasCurrentSegment = true;
        segments.push_back(Impl::SubmissionSegment{
            .queueType = queueType,
            .queue = &queue,
            .commandBuffer = currentCommandBuffer,
        });
        impl_->submittedCommandBuffers.push_back(std::move(commandBuffer));
        return {};
    };

    for (Impl::CompiledNode& node : impl_->executionList) {
        Queue* queue = queueForSubmitDesc(desc, node.queueType);
        if (queue == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!hasCurrentSegment || node.queueType != currentQueueType) {
            result = endCurrentSegment();
            if (!result) {
                return result;
            }
            result = beginSegment(node.queueType, *queue);
            if (!result) {
                return result;
            }
        }

        result = impl_->executeNode(*currentCommandBuffer, node);
        if (!result) {
            return result;
        }
    }

    result = endCurrentSegment();
    if (!result) {
        return result;
    }

    if (segments.empty()) {
        return makeError(Error::InvalidArgument);
    }

    if (segments.size() > 1) {
        impl_->submittedSemaphores.reserve(segments.size() - 1);
        for (size_t index = 0; index + 1 < segments.size(); ++index) {
            std::unique_ptr<Semaphore> semaphore;
            result = impl_->device->createSemaphore(semaphore);
            if (!result || semaphore == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }
            impl_->submittedSemaphores.push_back(std::move(semaphore));
        }
    }

    impl_->submittedFences.reserve(segments.size());
    for (size_t index = 0; index < segments.size(); ++index) {
        std::unique_ptr<Fence> fence;
        result = impl_->device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return result ? makeError(Error::Failure) : result;
        }
        impl_->submittedFences.push_back(std::move(fence));
    }

    for (size_t index = 0; index < segments.size(); ++index) {
        Impl::SubmissionSegment& segment = segments[index];
        CommandBuffer* commandBuffers[] = {segment.commandBuffer};

        SemaphoreSubmitDesc waitSemaphore{};
        const bool waitsOnPrevious = index > 0;
        if (waitsOnPrevious) {
            waitSemaphore = SemaphoreSubmitDesc{
                .semaphore = impl_->submittedSemaphores[index - 1].get(),
                .stages = PipelineStageBits::AllCommands,
            };
        }

        SemaphoreSubmitDesc signalSemaphore{};
        const bool signalsNext = index + 1 < segments.size();
        if (signalsNext) {
            signalSemaphore = SemaphoreSubmitDesc{
                .semaphore = impl_->submittedSemaphores[index].get(),
                .stages = PipelineStageBits::AllCommands,
            };
        }

        result = segment.queue->submit(QueueSubmitDesc{
            .waitSemaphores = waitsOnPrevious ? &waitSemaphore : nullptr,
            .waitSemaphoreCount = waitsOnPrevious ? 1u : 0u,
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalSemaphores = signalsNext ? &signalSemaphore : nullptr,
            .signalSemaphoreCount = signalsNext ? 1u : 0u,
            .signalFence = impl_->submittedFences[index].get(),
        });
        if (!result) {
            impl_->submittedFences.resize(index);
            impl_->hasSubmittedWork = index > 0;
            if (!impl_->hasSubmittedWork) {
                impl_->submittedCommandBuffers.clear();
                impl_->submittedSemaphores.clear();
                impl_->submittedFences.clear();
            }
            return result;
        }
        impl_->hasSubmittedWork = true;
    }

    return {};
}

Result RenderGraphExecutor::waitForSubmittedWork(uint64_t timeoutNanoseconds)
{
    return impl_->waitForSubmittedWork(timeoutNanoseconds);
}

bool RenderGraphExecutor::syncProperties(const RenderGraph& graph)
{
    if (!impl_->isCompiled) {
        return false;
    }

    bool synced = false;
    for (Impl::CompiledNode& compiledNode : impl_->executionList) {
        const RenderGraphNode* graphNode = graph.findNode(compiledNode.id);
        if (graphNode == nullptr ||
            graphNode->name != compiledNode.name ||
            graphNode->type != compiledNode.type) {
            return false;
        }

        if (compiledNode.properties != graphNode->properties) {
            compiledNode.properties = graphNode->properties;
            compiledNode.pass->setProperties(compiledNode.properties);
            synced = true;
        }
    }
    return synced;
}

Result RenderGraphExecutor::transitionOutput(
    CommandBuffer& commandBuffer,
    std::string_view fullName,
    ResourceState state)
{
    RenderGraphResource* resource = outputResource(fullName);
    if (resource == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return impl_->transition(
        commandBuffer,
        *resource,
        state,
        explicitAccessForState(resource->type, state));
}

RenderGraphResource* RenderGraphExecutor::outputResource(std::string_view fullName)
{
    return impl_->resource(fullName);
}

const RenderGraphResource* RenderGraphExecutor::outputResource(std::string_view fullName) const
{
    return impl_->resource(fullName);
}

bool RenderGraphExecutor::compiled() const
{
    return impl_->isCompiled;
}

uint32_t RenderGraphExecutor::width() const
{
    return impl_->width;
}

uint32_t RenderGraphExecutor::height() const
{
    return impl_->height;
}

struct RenderGraphPreviewRenderer::Impl {
    std::unique_ptr<Device> device;
    Queue* graphicsQueue = nullptr;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
    std::unique_ptr<Buffer> readbackBuffer;
    RenderGraphExecutor executor;
    HistoryResourceManager historyResources;
    std::vector<uint32_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t readbackWidth = 0;
    uint32_t readbackHeight = 0;
    uint64_t historyFrameIndex = 0;
    std::string lastLog;

    Result ensureReadback(uint32_t newWidth, uint32_t newHeight)
    {
        if (device == nullptr || newWidth == 0 || newHeight == 0) {
            return makeError(Error::InvalidArgument);
        }
        if (readbackBuffer != nullptr && readbackWidth == newWidth && readbackHeight == newHeight) {
            return {};
        }
        readbackBuffer.reset();
        const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
        Result result = device->createBuffer(
            BufferDesc{
                .size = byteSize,
                .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result) {
            return result;
        }
        readbackWidth = newWidth;
        readbackHeight = newHeight;
        pixels.resize(static_cast<size_t>(newWidth) * static_cast<size_t>(newHeight));
        return {};
    }
};

RenderGraphPreviewRenderer::RenderGraphPreviewRenderer()
    : impl_(std::make_unique<Impl>())
{
}

RenderGraphPreviewRenderer::~RenderGraphPreviewRenderer() = default;
RenderGraphPreviewRenderer::RenderGraphPreviewRenderer(RenderGraphPreviewRenderer&&) noexcept = default;
RenderGraphPreviewRenderer& RenderGraphPreviewRenderer::operator=(RenderGraphPreviewRenderer&&) noexcept = default;

Result RenderGraphPreviewRenderer::initialize(bool enableValidation, bool enableRayQuery)
{
    Result result = createDevice(
        DeviceDesc{
            .applicationName = "Metallic RenderGraph Preview",
            .enableValidation = enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true,
            .enableRayTracingAccelerationStructure = enableRayQuery,
            .enableRayQuery = enableRayQuery,
        },
        impl_->device);
    if (!result) {
        return result;
    }

    impl_->graphicsQueue = impl_->device->getQueue(QueueType::Graphics);
    if (impl_->graphicsQueue == nullptr) {
        return makeError(Error::Unsupported);
    }

    result = impl_->device->createCommandPool(*impl_->graphicsQueue, impl_->commandPool);
    if (!result) {
        return result;
    }
    result = impl_->commandPool->createCommandBuffer(impl_->commandBuffer);
    if (!result) {
        return result;
    }
    result = impl_->historyResources.initialize(*impl_->device);
    if (!result) {
        return result;
    }
    return impl_->device->createFence(true, impl_->fence);
}

Result RenderGraphPreviewRenderer::render(RenderGraph& graph, uint32_t newWidth, uint32_t newHeight)
{
    if (impl_->device == nullptr ||
        impl_->graphicsQueue == nullptr ||
        impl_->commandPool == nullptr ||
        impl_->commandBuffer == nullptr ||
        impl_->fence == nullptr ||
        newWidth == 0 ||
        newHeight == 0) {
        return makeError(Error::InvalidArgument);
    }

    Result result = impl_->fence->wait();
    if (!result) {
        return result;
    }

    const bool needsCompile =
        graph.dirty() ||
        !impl_->executor.compiled() ||
        impl_->executor.width() != newWidth ||
        impl_->executor.height() != newHeight;
    if (needsCompile) {
        result = impl_->device->waitIdle();
        if (!result) {
            return result;
        }
        impl_->historyResources.invalidateAll();
        impl_->historyFrameIndex = 0;
        result = impl_->executor.compile(
            *impl_->device,
            graph,
            newWidth,
            newHeight,
            impl_->lastLog);
        if (!result) {
            return result;
        }
        graph.clearDirty();
    } else {
        impl_->executor.syncProperties(graph);
    }

    result = impl_->ensureReadback(newWidth, newHeight);
    if (!result) {
        return result;
    }

    result = impl_->fence->reset();
    if (!result) {
        return result;
    }
    result = impl_->commandPool->reset();
    if (!result) {
        return result;
    }
    result = impl_->commandBuffer->begin();
    if (!result) {
        return result;
    }

    impl_->historyResources.beginFrame(impl_->historyFrameIndex++);
    result = impl_->executor.execute(*impl_->commandBuffer, &impl_->historyResources);
    if (!result) {
        return result;
    }

    const std::string outputName = graph.firstOutputName();
    RenderGraphResource* output = impl_->executor.outputResource(outputName);
    if (output == nullptr || impl_->readbackBuffer == nullptr) {
        impl_->lastLog = std::string("RenderGraph preview output resource is missing '") + outputName + "'";
        return makeError(Error::InvalidArgument);
    }
    if (output->type != RenderGraphResourceType::Texture2D || output->texture == nullptr) {
        impl_->lastLog = std::string("RenderGraph preview output is not a Texture2D '") + outputName + "'";
        return makeError(Error::InvalidArgument);
    }
    result = impl_->executor.transitionOutput(
        *impl_->commandBuffer,
        outputName,
        ResourceState::TransferSource);
    if (!result) {
        return result;
    }
    impl_->commandBuffer->copyTextureToBuffer(TextureBufferCopyDesc{
        .texture = output->texture,
        .buffer = impl_->readbackBuffer.get(),
        .width = newWidth,
        .height = newHeight,
        .depth = 1,
        .mipLevel = 0,
        .baseLayer = 0,
    });

    result = impl_->commandBuffer->end();
    if (!result) {
        return result;
    }

    CommandBuffer* commandBuffers[] = {impl_->commandBuffer.get()};
    result = impl_->graphicsQueue->submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = impl_->fence.get(),
    });
    if (!result) {
        return result;
    }
    result = impl_->fence->wait();
    if (!result) {
        return result;
    }

    impl_->readbackBuffer->invalidate();
    void* mapped = impl_->readbackBuffer->map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
    std::memcpy(impl_->pixels.data(), mapped, static_cast<size_t>(byteSize));
    impl_->readbackBuffer->unmap();

    impl_->width = newWidth;
    impl_->height = newHeight;
    return {};
}

const std::vector<uint32_t>& RenderGraphPreviewRenderer::pixels() const
{
    return impl_->pixels;
}

uint32_t RenderGraphPreviewRenderer::width() const
{
    return impl_->width;
}

uint32_t RenderGraphPreviewRenderer::height() const
{
    return impl_->height;
}

const std::string& RenderGraphPreviewRenderer::lastLog() const
{
    return impl_->lastLog;
}

std::string serializeRenderGraphToString(const RenderGraph& graph)
{
    nlohmann::json root;
    root["version"] = 1;
    root["name"] = graph.name();
    root["nodes"] = nlohmann::json::array();
    root["edges"] = nlohmann::json::array();
    root["outputs"] = nlohmann::json::array();

    for (const RenderGraphNode& node : graph.nodes()) {
        root["nodes"].push_back({
            {"id", node.id},
            {"name", node.name},
            {"type", node.type},
            {"properties", node.properties},
            {"position", {{"x", node.uiX}, {"y", node.uiY}}},
        });
    }

    for (const RenderGraphEdge& edge : graph.edges()) {
        root["edges"].push_back({
            {"id", edge.id},
            {"src", makeRenderGraphFieldName(edge.srcPass, edge.srcField)},
            {"dst", makeRenderGraphFieldName(edge.dstPass, edge.dstField)},
        });
    }

    for (const RenderGraphOutput& output : graph.outputs()) {
        root["outputs"].push_back(makeRenderGraphFieldName(output.passName, output.fieldName));
    }

    return root.dump(4);
}

bool deserializeRenderGraphFromString(
    const std::string& text,
    RenderGraph& outGraph,
    std::string& outMessage)
{
    try {
        nlohmann::json root = nlohmann::json::parse(text);
        if (!root.is_object() || root.value("version", 0) != 1) {
            outMessage = "Unsupported RenderGraph JSON version";
            return false;
        }

        RenderGraph graph;
        graph.clear();
        graph.name_ = root.value("name", "RenderGraph");
        graph.nodes_.clear();
        graph.edges_.clear();
        graph.outputs_.clear();

        uint32_t maxNodeId = 0;
        uint32_t maxEdgeId = 0;
        for (const nlohmann::json& nodeJson : root.at("nodes")) {
            RenderGraphNode node;
            node.id = nodeJson.value("id", 0u);
            node.name = nodeJson.value("name", "");
            node.type = nodeJson.value("type", "");
            node.properties = nodeJson.value("properties", RenderGraphProperties::object());
            if (nodeJson.contains("position")) {
                node.uiX = nodeJson["position"].value("x", 0.0f);
                node.uiY = nodeJson["position"].value("y", 0.0f);
            }
            maxNodeId = std::max(maxNodeId, node.id);
            graph.nodes_.push_back(std::move(node));
        }

        for (const nlohmann::json& edgeJson : root.value("edges", nlohmann::json::array())) {
            RenderGraphEdge edge;
            edge.id = edgeJson.value("id", 0u);
            const std::string src = edgeJson.value("src", "");
            const std::string dst = edgeJson.value("dst", "");
            if (!splitRenderGraphFieldName(src, edge.srcPass, edge.srcField) ||
                !splitRenderGraphFieldName(dst, edge.dstPass, edge.dstField)) {
                outMessage = "Invalid edge endpoint in RenderGraph JSON";
                return false;
            }
            maxEdgeId = std::max(maxEdgeId, edge.id);
            graph.edges_.push_back(std::move(edge));
        }

        for (const nlohmann::json& outputJson : root.value("outputs", nlohmann::json::array())) {
            std::string passName;
            std::string fieldName;
            if (!splitRenderGraphFieldName(outputJson.get<std::string>(), passName, fieldName)) {
                outMessage = "Invalid graph output in RenderGraph JSON";
                return false;
            }
            graph.outputs_.push_back(RenderGraphOutput{
                .passName = std::move(passName),
                .fieldName = std::move(fieldName),
            });
        }

        graph.nextNodeId_ = maxNodeId + 1;
        graph.nextEdgeId_ = maxEdgeId + 1;
        graph.markDirty();

        std::string validationLog;
        if (!graph.validate(validationLog)) {
            outMessage = validationLog;
            return false;
        }

        outGraph = std::move(graph);
        outMessage = "Loaded RenderGraph";
        return true;
    } catch (const std::exception& exception) {
        outMessage = exception.what();
        return false;
    }
}

bool saveRenderGraphToFile(
    const RenderGraph& graph,
    const std::filesystem::path& path,
    std::string& outMessage)
{
    std::error_code error;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            outMessage = error.message();
            return false;
        }
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        outMessage = "Failed to open RenderGraph file for writing";
        return false;
    }
    file << serializeRenderGraphToString(graph);
    outMessage = std::string("Saved RenderGraph to ") + path.string();
    return true;
}

bool loadRenderGraphFromFile(
    const std::filesystem::path& path,
    RenderGraph& outGraph,
    std::string& outMessage)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        outMessage = "Failed to open RenderGraph file";
        return false;
    }
    std::ostringstream stream;
    stream << file.rdbuf();
    return deserializeRenderGraphFromString(stream.str(), outGraph, outMessage);
}

} // namespace metallic::render
