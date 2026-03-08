#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;

enum class RhiBackendType {
    Metal,
    Vulkan,
};

enum class RhiFormat {
    Undefined,
    R8Unorm,
    R16Float,
    R32Float,
    R32Uint,
    RG8Unorm,
    RG16Float,
    RG32Float,
    RGBA8Unorm,
    BGRA8Unorm,
    RGBA16Float,
    RGBA32Float,
    D32Float,
    D16Unorm,
};

enum class RhiVertexFormat {
    Float2,
    Float3,
    Float4,
};

enum class RhiTextureUsage : uint32_t {
    None = 0,
    RenderTarget = 1u << 0,
    ShaderRead = 1u << 1,
    ShaderWrite = 1u << 2,
};

enum class RhiTextureStorageMode {
    Private,
    Shared,
};

enum class RhiLoadAction {
    DontCare,
    Load,
    Clear,
};

enum class RhiStoreAction {
    DontCare,
    Store,
};

constexpr RhiTextureUsage operator|(RhiTextureUsage lhs, RhiTextureUsage rhs) {
    return static_cast<RhiTextureUsage>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr RhiTextureUsage operator&(RhiTextureUsage lhs, RhiTextureUsage rhs) {
    return static_cast<RhiTextureUsage>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

constexpr RhiTextureUsage& operator|=(RhiTextureUsage& lhs, RhiTextureUsage rhs) {
    lhs = lhs | rhs;
    return lhs;
}

struct RhiClearColor {
    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    double alpha = 1.0;

    constexpr RhiClearColor() = default;
    constexpr RhiClearColor(double r, double g, double b, double a)
        : red(r), green(g), blue(b), alpha(a) {}
};

struct RhiTextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    RhiFormat format = RhiFormat::BGRA8Unorm;
    RhiTextureUsage usage = RhiTextureUsage::RenderTarget;
    RhiTextureStorageMode storageMode = RhiTextureStorageMode::Private;

    static RhiTextureDesc renderTarget(uint32_t w, uint32_t h, RhiFormat fmt) {
        return {w, h, fmt, RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead, RhiTextureStorageMode::Private};
    }

    static RhiTextureDesc depthTarget(uint32_t w, uint32_t h, RhiFormat fmt = RhiFormat::D32Float) {
        return {w, h, fmt, RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead, RhiTextureStorageMode::Private};
    }

    static RhiTextureDesc storageTexture(uint32_t w, uint32_t h, RhiFormat fmt) {
        return {w, h, fmt, RhiTextureUsage::ShaderRead | RhiTextureUsage::ShaderWrite, RhiTextureStorageMode::Private};
    }
};

class RhiTexture {
public:
    virtual ~RhiTexture() = default;
    virtual void* nativeHandle() const = 0;
};

struct RhiColorAttachmentDesc {
    RhiTexture* texture = nullptr;
    RhiLoadAction loadAction = RhiLoadAction::Clear;
    RhiStoreAction storeAction = RhiStoreAction::Store;
    RhiClearColor clearColor{};
};

struct RhiDepthAttachmentDesc {
    RhiTexture* texture = nullptr;
    RhiLoadAction loadAction = RhiLoadAction::Clear;
    RhiStoreAction storeAction = RhiStoreAction::DontCare;
    double clearDepth = 1.0;
    bool bound = false;
};

struct RhiRenderPassDesc {
    const char* label = nullptr;
    std::array<RhiColorAttachmentDesc, 8> colorAttachments{};
    uint32_t colorAttachmentCount = 0;
    RhiDepthAttachmentDesc depthAttachment{};
};

struct RhiComputePassDesc {
    const char* label = nullptr;
};

struct RhiBlitPassDesc {
    const char* label = nullptr;
};

class RhiRenderCommandEncoder {
public:
    virtual ~RhiRenderCommandEncoder() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiComputeCommandEncoder {
public:
    virtual ~RhiComputeCommandEncoder() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiBlitCommandEncoder {
public:
    virtual ~RhiBlitCommandEncoder() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiCommandBuffer {
public:
    virtual ~RhiCommandBuffer() = default;
    virtual std::unique_ptr<RhiRenderCommandEncoder> beginRenderPass(const RhiRenderPassDesc& desc) = 0;
    virtual std::unique_ptr<RhiComputeCommandEncoder> beginComputePass(const RhiComputePassDesc& desc) = 0;
    virtual std::unique_ptr<RhiBlitCommandEncoder> beginBlitPass(const RhiBlitPassDesc& desc) = 0;
};

class RhiFrameGraphBackend {
public:
    virtual ~RhiFrameGraphBackend() = default;
    virtual std::unique_ptr<RhiTexture> createTexture(const RhiTextureDesc& desc) = 0;
};

struct RhiFeatures {
    bool dynamicRendering = false;
    bool meshShaders = false;
    bool rayTracing = false;
    bool validation = false;
};

struct RhiDeviceInfo {
    std::string adapterName;
    std::string driverName;
    uint32_t apiVersion = 0;
};

struct RhiCreateInfo {
    GLFWwindow* window = nullptr;
    uint32_t width = 1280;
    uint32_t height = 720;
    const char* applicationName = "Metallic";
    bool enableValidation = true;
    bool requireVulkan14 = true;
};

struct RhiBufferDesc {
    size_t size = 0;
    const void* initialData = nullptr;
    bool hostVisible = true;
    const char* debugName = nullptr;
};

struct RhiShaderModuleDesc {
    std::vector<uint32_t> spirv;
    const char* debugName = nullptr;
};

struct RhiVertexBindingDesc {
    uint32_t binding = 0;
    uint32_t stride = 0;
};

struct RhiVertexAttributeDesc {
    uint32_t location = 0;
    uint32_t binding = 0;
    RhiVertexFormat format = RhiVertexFormat::Float3;
    uint32_t offset = 0;
};

class RhiShaderModule {
public:
    virtual ~RhiShaderModule() = default;
};

class RhiBuffer {
public:
    virtual ~RhiBuffer() = default;
    virtual size_t size() const = 0;
};

class RhiGraphicsPipeline {
public:
    virtual ~RhiGraphicsPipeline() = default;
};

struct RhiGraphicsPipelineDesc {
    const RhiShaderModule* shaderModule = nullptr;
    const char* vertexEntry = "vertexMain";
    const char* fragmentEntry = "fragmentMain";
    const char* meshEntry = nullptr;
    const char* taskEntry = nullptr;
    std::vector<RhiVertexBindingDesc> bindings;
    std::vector<RhiVertexAttributeDesc> attributes;
    RhiFormat colorFormat = RhiFormat::BGRA8Unorm;
    RhiFormat depthFormat = RhiFormat::Undefined;
    bool enableDepth = false;
    bool enableMeshShaders = false;
};

struct RhiNativeHandles {
    void* instance = nullptr;
    void* physicalDevice = nullptr;
    void* device = nullptr;
    void* queue = nullptr;
    void* descriptorPool = nullptr;
    uint32_t graphicsQueueFamily = 0;
    uint32_t swapchainImageCount = 0;
    uint32_t colorFormat = 0;
    uint32_t apiVersion = 0;
};

struct RhiRenderTargetInfo {
    float clearColor[4] = {0.08f, 0.09f, 0.12f, 1.0f};
    bool clear = true;
};

class RhiCommandContext {
public:
    virtual ~RhiCommandContext() = default;
    virtual void beginRendering(const RhiRenderTargetInfo& targetInfo) = 0;
    virtual void endRendering() = 0;
    virtual void setViewport(float width, float height) = 0;
    virtual void setScissor(uint32_t width, uint32_t height) = 0;
    virtual void bindGraphicsPipeline(const RhiGraphicsPipeline& pipeline) = 0;
    virtual void bindVertexBuffer(const RhiBuffer& buffer, uint64_t offset = 0) = 0;
    virtual void draw(uint32_t vertexCount,
                      uint32_t instanceCount = 1,
                      uint32_t firstVertex = 0,
                      uint32_t firstInstance = 0) = 0;
    virtual void* nativeCommandBuffer() const = 0;
};

class RhiContext {
public:
    virtual ~RhiContext() = default;
    virtual RhiBackendType backendType() const = 0;
    virtual const RhiFeatures& features() const = 0;
    virtual const RhiDeviceInfo& deviceInfo() const = 0;
    virtual const RhiNativeHandles& nativeHandles() const = 0;
    virtual bool beginFrame() = 0;
    virtual void endFrame() = 0;
    virtual void resize(uint32_t width, uint32_t height) = 0;
    virtual void waitIdle() = 0;
    virtual RhiCommandContext& commandContext() = 0;
    virtual uint32_t drawableWidth() const = 0;
    virtual uint32_t drawableHeight() const = 0;
    virtual RhiFormat colorFormat() const = 0;
    virtual std::unique_ptr<RhiShaderModule> createShaderModule(const RhiShaderModuleDesc& desc) = 0;
    virtual std::unique_ptr<RhiBuffer> createVertexBuffer(const RhiBufferDesc& desc) = 0;
    virtual std::unique_ptr<RhiGraphicsPipeline> createGraphicsPipeline(const RhiGraphicsPipelineDesc& desc) = 0;
};

std::unique_ptr<RhiContext> createRhiContext(RhiBackendType backend,
                                             const RhiCreateInfo& createInfo,
                                             std::string& errorMessage);

