#pragma once

#include "Runtime/Render/GAPI/rhi.h"

#include "json.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metallic::render {

class HistoryResourceManager;

using RenderGraphProperties = nlohmann::json;

enum class RenderGraphFieldVisibility : uint8_t {
    Input,
    Output,
};

enum class RenderGraphResourceType : uint8_t {
    Texture2D,
    Buffer,
};

enum class RenderGraphResourceAccess : uint8_t {
    None,
    TextureSampleRead,
    TextureColorWrite,
    TextureDepthStencilWrite,
    TextureTransferRead,
    TextureTransferWrite,
    TextureStorageReadWrite,
    BufferShaderRead,
    BufferStorageReadWrite,
    BufferTransferRead,
    BufferTransferWrite,
    BufferConstantRead,
};

enum class RenderGraphBindlessAccess : uint8_t {
    None,
    SampledImage,
    Buffer,
};

enum class RenderGraphPassKind : uint8_t {
    Raster,
    Compute,
    Unsafe,
};

struct RenderGraphField {
    std::string name;
    std::string description;
    RenderGraphFieldVisibility visibility = RenderGraphFieldVisibility::Output;
    RenderGraphResourceType resourceType = RenderGraphResourceType::Texture2D;
    RenderGraphResourceAccess access = RenderGraphResourceAccess::TextureColorWrite;
    RenderGraphBindlessAccess bindlessAccess = RenderGraphBindlessAccess::None;
    Format format = Format::Rgba8Unorm;
    TextureUsageBits usage = TextureUsageBits::ColorAttachment;
    BufferUsageBits bufferUsage = BufferUsageBits::None;
    BufferViewType bufferViewType = BufferViewType::Raw;
    ResourceState state = ResourceState::ColorAttachment;
    bool optional = false;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t size = 0;
    uint32_t structureStride = 0;
    MemoryLocation memoryLocation = MemoryLocation::Device;

    RenderGraphField& texture2D(uint32_t newWidth = 0, uint32_t newHeight = 0);
    RenderGraphField& buffer(uint64_t newSize, uint32_t newStructureStride = 0);
    RenderGraphField& setOptional(bool value = true);
    RenderGraphField& sampledRead();
    RenderGraphField& colorWrite();
    RenderGraphField& depthStencilWrite();
    RenderGraphField& storageReadWrite();
    RenderGraphField& transferRead();
    RenderGraphField& transferWrite();
    RenderGraphField& shaderRead();
    RenderGraphField& constantRead();
    RenderGraphField& bindlessSampledImage();
    RenderGraphField& bindlessBuffer();
    RenderGraphField& hostReadback();
};

class RenderPassReflection {
public:
    RenderGraphField& addInput(std::string name, std::string description = {});
    RenderGraphField& addBindlessSampledInput(std::string name, std::string description = {});
    RenderGraphField& addOutput(std::string name, std::string description = {});
    RenderGraphField& addTextureInput(std::string name, std::string description = {});
    RenderGraphField& addTextureOutput(std::string name, std::string description = {});
    RenderGraphField& addBufferInput(std::string name, std::string description = {});
    RenderGraphField& addBufferOutput(std::string name, std::string description = {});

    const RenderGraphField* findField(
        std::string_view name,
        RenderGraphFieldVisibility visibility) const;
    const std::vector<RenderGraphField>& fields() const { return fields_; }

private:
    std::vector<RenderGraphField> fields_;
};

struct RenderGraphCompileContext {
    Device* device = nullptr;
    Queue* graphicsQueue = nullptr;
    uint32_t width = 1;
    uint32_t height = 1;
    Format defaultFormat = Format::Rgba8Unorm;
};

struct RenderGraphResource {
    RenderGraphResourceType type = RenderGraphResourceType::Texture2D;
    Texture* texture = nullptr;
    TextureView* view = nullptr;
    TextureDesc desc;
    Buffer* buffer = nullptr;
    BufferView* bufferView = nullptr;
    BufferDesc bufferDesc;
    BufferViewDesc bufferViewDesc;
    ResourceState state = ResourceState::Undefined;
    RenderGraphResourceAccess lastAccess = RenderGraphResourceAccess::None;
    BindlessHandle bindlessHandle;
    BindlessHandle sampledImageBindlessHandle;
};

class TextureHandle {
public:
    TextureHandle() = default;

    bool valid() const;
    Texture* texture() const;
    TextureView* view() const;
    const TextureDesc& desc() const;
    const BindlessHandle& bindlessHandle() const;

private:
    explicit TextureHandle(RenderGraphResource* resource);

    RenderGraphResource* resource_ = nullptr;

    friend class RenderGraphExecutionContext;
};

class BufferHandle {
public:
    BufferHandle() = default;

    bool valid() const;
    Buffer* buffer() const;
    BufferView* view() const;
    const BufferDesc& desc() const;
    const BufferViewDesc& viewDesc() const;
    const BindlessHandle& bindlessHandle() const;

private:
    explicit BufferHandle(RenderGraphResource* resource);

    RenderGraphResource* resource_ = nullptr;

    friend class RenderGraphExecutionContext;
};

class RenderGraphExecutionContext {
public:
    CommandBuffer& commandBuffer() const { return commandBuffer_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    const std::string& passName() const { return passName_; }
    const RenderGraphProperties& properties() const { return properties_; }
    HistoryResourceManager* historyResources() const { return historyResources_; }

    RenderGraphResource* resource(std::string_view fieldName) const;
    RenderGraphResource* input(std::string_view fieldName) const;
    RenderGraphResource* output(std::string_view fieldName) const;
    TextureHandle texture(std::string_view fieldName) const;
    TextureHandle inputTexture(std::string_view fieldName) const;
    TextureHandle outputTexture(std::string_view fieldName) const;
    BufferHandle buffer(std::string_view fieldName) const;
    BufferHandle inputBuffer(std::string_view fieldName) const;
    BufferHandle outputBuffer(std::string_view fieldName) const;
    const BindlessHandle* bindlessResource(std::string_view fieldName) const;
    const BindlessHandle* bindlessInput(std::string_view fieldName) const;

private:
    struct Binding {
        std::string fieldName;
        RenderGraphResource* resource = nullptr;
        RenderGraphFieldVisibility visibility = RenderGraphFieldVisibility::Output;
        RenderGraphBindlessAccess bindlessAccess = RenderGraphBindlessAccess::None;
        BindlessHandle bindlessHandle;
        BindlessHandle sampledImageBindlessHandle;
    };

    RenderGraphExecutionContext(
        CommandBuffer& commandBuffer,
        uint32_t width,
        uint32_t height,
        std::string passName,
        const RenderGraphProperties& properties,
        std::vector<Binding> bindings,
        HistoryResourceManager* historyResources);

    CommandBuffer& commandBuffer_;
    uint32_t width_ = 1;
    uint32_t height_ = 1;
    std::string passName_;
    const RenderGraphProperties& properties_;
    std::vector<Binding> bindings_;
    HistoryResourceManager* historyResources_ = nullptr;

    friend class RenderGraphExecutor;
};

class RenderGraphPass {
public:
    virtual ~RenderGraphPass() = default;

    virtual RenderPassReflection reflect(const RenderGraphCompileContext& context) const = 0;
    virtual RenderGraphPassKind kind() const;
    virtual QueueType queueType() const;
    virtual Result compile(const RenderGraphCompileContext& context, std::string& log);
    virtual Result execute(RenderGraphExecutionContext& context) = 0;

    void setProperties(RenderGraphProperties properties) { properties_ = std::move(properties); }
    const RenderGraphProperties& properties() const { return properties_; }

private:
    RenderGraphProperties properties_ = RenderGraphProperties::object();
};

class RasterPass : public RenderGraphPass {
public:
    RenderGraphPassKind kind() const override;
    QueueType queueType() const override;
};

class ComputePass : public RenderGraphPass {
public:
    RenderGraphPassKind kind() const override;
    QueueType queueType() const override;
};

// Unsafe passes may record mixed graphics, compute, or transfer commands and are
// kept on the graphics queue until the graph can prove finer-grained hazards.
class UnsafePass : public RenderGraphPass {
public:
    RenderGraphPassKind kind() const override;
    QueueType queueType() const override;
};

using RenderGraphPassFactory = std::function<std::unique_ptr<RenderGraphPass>()>;

struct RenderGraphPassInfo {
    std::string type;
    std::string description;
    RenderGraphPassKind kind = RenderGraphPassKind::Unsafe;
    QueueType queueType = QueueType::Graphics;
};

const char* renderGraphPassKindName(RenderGraphPassKind kind);
bool registerRenderGraphPassType(
    std::string type,
    std::string description,
    RenderGraphPassFactory factory);
void registerBuiltInRenderGraphPasses();
std::unique_ptr<RenderGraphPass> createRenderGraphPass(std::string_view type);
std::vector<RenderGraphPassInfo> listRenderGraphPassTypes();

struct RenderGraphNode {
    uint32_t id = 0;
    std::string name;
    std::string type;
    RenderGraphProperties properties = RenderGraphProperties::object();
    float uiX = 0.0f;
    float uiY = 0.0f;
};

struct RenderGraphEdge {
    uint32_t id = 0;
    std::string srcPass;
    std::string srcField;
    std::string dstPass;
    std::string dstField;
};

struct RenderGraphOutput {
    std::string passName;
    std::string fieldName;
};

class RenderGraph {
public:
    RenderGraph();

    const std::string& name() const { return name_; }
    void setName(std::string name);

    const std::vector<RenderGraphNode>& nodes() const { return nodes_; }
    const std::vector<RenderGraphEdge>& edges() const { return edges_; }
    const std::vector<RenderGraphOutput>& outputs() const { return outputs_; }

    const RenderGraphNode* findNode(std::string_view name) const;
    RenderGraphNode* findNode(std::string_view name);
    const RenderGraphNode* findNode(uint32_t id) const;
    RenderGraphNode* findNode(uint32_t id);
    const RenderGraphEdge* findEdge(uint32_t id) const;

    RenderGraphNode* addNode(
        std::string type,
        std::string name,
        RenderGraphProperties properties = RenderGraphProperties::object(),
        float uiX = 0.0f,
        float uiY = 0.0f);
    bool removeNode(uint32_t id);
    bool renameNode(uint32_t id, std::string newName);
    bool setNodeProperties(uint32_t id, RenderGraphProperties properties);
    bool setNodePosition(uint32_t id, float uiX, float uiY);

    RenderGraphEdge* addEdge(std::string src, std::string dst);
    bool removeEdge(uint32_t id);
    bool markOutput(std::string output);
    bool unmarkOutput(std::string output);
    void clearOutputs();

    bool validate(std::string& log) const;

    bool dirty() const { return dirty_; }
    void clearDirty() { dirty_ = false; }
    void markDirty() { dirty_ = true; }
    void clear();

    std::string firstOutputName() const;

    static RenderGraph createDefaultTriangleGraph();
    static RenderGraph createDefaultBunnyGraph();

private:
    std::string name_ = "RenderGraph";
    std::vector<RenderGraphNode> nodes_;
    std::vector<RenderGraphEdge> edges_;
    std::vector<RenderGraphOutput> outputs_;
    uint32_t nextNodeId_ = 1;
    uint32_t nextEdgeId_ = 1;
    bool dirty_ = true;

    friend bool deserializeRenderGraphFromString(
        const std::string& text,
        RenderGraph& outGraph,
        std::string& outMessage);
};

bool splitRenderGraphFieldName(
    std::string_view fullName,
    std::string& outPassName,
    std::string& outFieldName);
std::string makeRenderGraphFieldName(std::string_view passName, std::string_view fieldName);

struct RenderGraphSubmitDesc {
    Queue* graphicsQueue = nullptr;
    Queue* computeQueue = nullptr;
    Queue* copyQueue = nullptr;
};

class RenderGraphExecutor {
public:
    RenderGraphExecutor();
    ~RenderGraphExecutor();

    RenderGraphExecutor(RenderGraphExecutor&&) noexcept;
    RenderGraphExecutor& operator=(RenderGraphExecutor&&) noexcept;

    RenderGraphExecutor(const RenderGraphExecutor&) = delete;
    RenderGraphExecutor& operator=(const RenderGraphExecutor&) = delete;

    Result compile(
        Device& device,
        const RenderGraph& graph,
        uint32_t width,
        uint32_t height,
        std::string& log);
    Result execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources = nullptr);
    Result execute(const RenderGraphSubmitDesc& desc);
    Result waitForSubmittedWork(uint64_t timeoutNanoseconds = UINT64_MAX);
    bool syncProperties(const RenderGraph& graph);
    Result transitionOutput(
        CommandBuffer& commandBuffer,
        std::string_view fullName,
        ResourceState state);

    RenderGraphResource* outputResource(std::string_view fullName);
    const RenderGraphResource* outputResource(std::string_view fullName) const;
    bool compiled() const;
    uint32_t width() const;
    uint32_t height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RenderGraphPreviewRenderer {
public:
    RenderGraphPreviewRenderer();
    ~RenderGraphPreviewRenderer();

    RenderGraphPreviewRenderer(RenderGraphPreviewRenderer&&) noexcept;
    RenderGraphPreviewRenderer& operator=(RenderGraphPreviewRenderer&&) noexcept;

    RenderGraphPreviewRenderer(const RenderGraphPreviewRenderer&) = delete;
    RenderGraphPreviewRenderer& operator=(const RenderGraphPreviewRenderer&) = delete;

    Result initialize(bool enableValidation = false, bool enableRayQuery = false);
    Result render(RenderGraph& graph, uint32_t width, uint32_t height);
    const std::vector<uint32_t>& pixels() const;
    uint32_t width() const;
    uint32_t height() const;
    const std::string& lastLog() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string serializeRenderGraphToString(const RenderGraph& graph);
bool deserializeRenderGraphFromString(
    const std::string& text,
    RenderGraph& outGraph,
    std::string& outMessage);
bool saveRenderGraphToFile(
    const RenderGraph& graph,
    const std::filesystem::path& path,
    std::string& outMessage);
bool loadRenderGraphFromFile(
    const std::filesystem::path& path,
    RenderGraph& outGraph,
    std::string& outMessage);

} // namespace metallic::render
