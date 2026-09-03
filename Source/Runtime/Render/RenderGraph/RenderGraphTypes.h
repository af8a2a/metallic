#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include "json.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metallic::scene {
class Scene;
}

namespace metallic::render {

class HistoryResourceManager;
class SceneResourceManager;

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

enum class RenderGraphRuntimeSettingType : uint8_t {
    Bool,
    Int,
    Float,
    Float3,
    Color4,
    Enum,
    ActionCounter,
};

struct RenderGraphRuntimeSettingOption {
    std::string label;
    RenderGraphProperties value;
};

struct RenderGraphRuntimeSetting {
    std::string key;
    std::string label;
    RenderGraphRuntimeSettingType type = RenderGraphRuntimeSettingType::Bool;
    RenderGraphProperties defaultValue;
    RenderGraphProperties minValue;
    RenderGraphProperties maxValue;
    std::vector<RenderGraphRuntimeSettingOption> options;
    bool invalidateHistory = false;
    bool rebuildGraph = false;
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

    bool operator==(const RenderGraphField&) const = default;

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

    bool operator==(const RenderPassReflection&) const = default;

private:
    std::vector<RenderGraphField> fields_;
};

struct RenderGraphCompileContext {
    Device* device = nullptr;
    Queue* graphicsQueue = nullptr;
    const scene::Scene* runtimeScene = nullptr;
    SceneResourceManager* sceneResourceManager = nullptr;
    RenderWorld* renderWorld = nullptr;
    RenderSubsystemHost* subsystemHost = nullptr;
    uint32_t width = 1;
    uint32_t height = 1;
    Format defaultFormat = Format::Rgba8Unorm;

    RenderWorld* world() const { return renderWorld; }
    RenderSubsystemHost* subsystems() const { return subsystemHost; }

    template <typename T>
    T* subsystem() const
    {
        return subsystemHost != nullptr ? subsystemHost->get<T>() : nullptr;
    }
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
    uint64_t frameIndex() const { return frameIndex_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    const std::string& passName() const { return passName_; }
    const RenderGraphProperties& properties() const { return properties_; }
    HistoryResourceManager* historyResources() const { return historyResources_; }
    Streamer* streamer() const { return streamer_; }
    const scene::Scene* runtimeScene() const { return runtimeScene_; }
    RenderWorld* world() const { return world_; }
    RenderSubsystemHost* subsystems() const { return subsystems_; }

    template <typename T>
    T* subsystem() const
    {
        return subsystems_ != nullptr ? subsystems_->get<T>() : nullptr;
    }

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
        uint64_t frameIndex,
        uint32_t width,
        uint32_t height,
        std::string passName,
        const RenderGraphProperties& properties,
        std::vector<Binding> bindings,
        HistoryResourceManager* historyResources,
        Streamer* streamer,
        const scene::Scene* runtimeScene,
        RenderWorld* world,
        RenderSubsystemHost* subsystems);

    CommandBuffer& commandBuffer_;
    uint64_t frameIndex_ = 0;
    uint32_t width_ = 1;
    uint32_t height_ = 1;
    std::string passName_;
    const RenderGraphProperties& properties_;
    std::vector<Binding> bindings_;
    HistoryResourceManager* historyResources_ = nullptr;
    Streamer* streamer_ = nullptr;
    const scene::Scene* runtimeScene_ = nullptr;
    RenderWorld* world_ = nullptr;
    RenderSubsystemHost* subsystems_ = nullptr;

    friend class RenderGraphExecutor;
};

class RenderGraphPass {
public:
    virtual ~RenderGraphPass() = default;

    virtual RenderPassReflection reflect(const RenderGraphCompileContext& context) const = 0;
    virtual RenderGraphPassKind kind() const;
    virtual QueueType queueType() const;
    virtual std::span<const RenderSubsystemId> requiredSubsystems() const;
    virtual std::vector<RenderGraphRuntimeSetting> runtimeSettings() const;
    virtual Result prepare(const RenderGraphCompileContext& context, std::string& log);
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


} // namespace metallic::render
