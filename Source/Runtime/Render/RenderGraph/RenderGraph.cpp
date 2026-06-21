#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"

#include <algorithm>
#include <exception>
#include <fstream>
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


} // namespace

using namespace detail;
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

std::vector<RenderGraphRuntimeSetting> RenderGraphPass::runtimeSettings() const
{
    return {};
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

bool RenderGraph::setNodeRuntimeProperties(uint32_t id, RenderGraphProperties properties)
{
    RenderGraphNode* node = findNode(id);
    if (node == nullptr) {
        return false;
    }
    node->runtimeProperties = std::move(properties);
    return true;
}

bool RenderGraph::setNodeRuntimeProperty(uint32_t id, std::string key, RenderGraphProperties value)
{
    RenderGraphNode* node = findNode(id);
    if (node == nullptr || key.empty()) {
        return false;
    }
    if (!node->runtimeProperties.is_object()) {
        node->runtimeProperties = RenderGraphProperties::object();
    }

    RenderGraphProperties* object = &node->runtimeProperties;
    size_t begin = 0;
    while (begin < key.size()) {
        const size_t dot = key.find('.', begin);
        const std::string part = key.substr(begin, dot == std::string::npos ? std::string::npos : dot - begin);
        if (part.empty()) {
            return false;
        }
        if (dot == std::string::npos) {
            (*object)[part] = std::move(value);
            return true;
        }

        RenderGraphProperties& child = (*object)[part];
        if (!child.is_object()) {
            child = RenderGraphProperties::object();
        }
        object = &child;
        begin = dot + 1;
    }
    return false;
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
    std::unordered_set<uint32_t> edgeIds;
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
        if (edge.id == 0 || !edgeIds.insert(edge.id).second) {
            log = validationPrefix("duplicate edge id");
            return false;
        }
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

        const nlohmann::json edgesJson = root.value("edges", nlohmann::json::array());
        for (const nlohmann::json& edgeJson : edgesJson) {
            maxEdgeId = std::max(maxEdgeId, edgeJson.value("id", 0u));
        }
        uint32_t generatedEdgeId = maxEdgeId + 1u;
        std::unordered_set<uint32_t> usedEdgeIds;
        for (const nlohmann::json& edgeJson : edgesJson) {
            RenderGraphEdge edge;
            const uint32_t requestedEdgeId = edgeJson.value("id", 0u);
            if (requestedEdgeId != 0u && usedEdgeIds.insert(requestedEdgeId).second) {
                edge.id = requestedEdgeId;
            } else {
                while (generatedEdgeId == 0u || usedEdgeIds.contains(generatedEdgeId)) {
                    ++generatedEdgeId;
                }
                edge.id = generatedEdgeId++;
                usedEdgeIds.insert(edge.id);
            }
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
