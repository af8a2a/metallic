#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rhi_interop.h"

struct GLFWwindow;
struct RhiBufferDesc;

class RhiBuffer;
class RhiGraphicsPipeline;
class RhiComputePipeline;
class RhiSampler;
class RhiDepthStencilState;
class RhiAccelerationStructure;
class RhiDevice;
class RhiCommandQueue;
class RhiNativeCommandBuffer;
class RhiShaderLibrary;
class RhiVertexDescriptor;
class RhiContext;

enum class RhiBackendType {
    Metal,
    Vulkan,
};

enum class RhiFormat {
    Undefined = 0,
    R8Unorm = 1,
    R16Float = 2,
    R32Float = 3,
    R32Uint = 4,
    RG8Unorm = 5,
    RG16Float = 6,
    RG32Float = 7,
    RGBA8Unorm = 8,
    BGRA8Unorm = 9,
    RGBA16Float = 10,
    RGBA32Float = 11,
    D32Float = 12,
    D16Unorm = 13,
    RGBA8Srgb = 14,
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

enum class RhiSamplerFilterMode {
    Nearest,
    Linear,
};

enum class RhiSamplerMipFilterMode {
    None,
    Linear,
};

enum class RhiSamplerAddressMode {
    Repeat,
    ClampToEdge,
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

enum class RhiWinding {
    Clockwise,
    CounterClockwise,
};

enum class RhiCullMode {
    None,
    Front,
    Back,
};

enum class RhiPrimitiveType {
    Triangle,
};

enum class RhiIndexType {
    UInt16,
    UInt32,
};

enum class RhiBarrierScope : uint32_t {
    None = 0,
    Buffers = 1u << 0,
    Textures = 1u << 1,
    RenderTargets = 1u << 2,
};

// Hints to the backend which physical queue a pass should execute on.
// The backend uses this to route work to dedicated compute/transfer queues if available,
// falling back to the graphics queue otherwise.
enum class RhiQueueHint {
    Auto,         // Backend decides based on pass type
    Graphics,     // Requires graphics queue (rasterisation, mesh shaders, RT)
    AsyncCompute, // Prefer dedicated async compute queue; falls back to graphics
    Transfer,     // Prefer dedicated transfer queue; falls back to graphics
};

enum class RhiResourceUsage : uint32_t {
    Read = 1u << 0,
    Write = 1u << 1,
};

constexpr RhiBarrierScope operator|(RhiBarrierScope lhs, RhiBarrierScope rhs) {
    return static_cast<RhiBarrierScope>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr RhiResourceUsage operator|(RhiResourceUsage lhs, RhiResourceUsage rhs) {
    return static_cast<RhiResourceUsage>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

struct RhiOrigin3D {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
};

struct RhiSize3D {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
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
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
};

class RhiSampler {
public:
    virtual ~RhiSampler() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiDepthStencilState {
public:
    virtual ~RhiDepthStencilState() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiAccelerationStructure {
public:
    virtual ~RhiAccelerationStructure() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiComputePipeline {
public:
    virtual ~RhiComputePipeline() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiDevice {
public:
    virtual ~RhiDevice() = default;
    virtual RhiContext* ownerContext() const { return nullptr; }
    virtual void* nativeHandle() const = 0;
};

class RhiCommandQueue {
public:
    virtual ~RhiCommandQueue() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiNativeCommandBuffer {
public:
    virtual ~RhiNativeCommandBuffer() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiShaderLibrary {
public:
    virtual ~RhiShaderLibrary() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiVertexDescriptor {
public:
    virtual ~RhiVertexDescriptor() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiDeviceHandle final : public RhiDevice {
public:
    constexpr RhiDeviceHandle() = default;
    explicit constexpr RhiDeviceHandle(void* native, RhiContext* ownerContext = nullptr)
        : m_native(native)
        , m_ownerContext(ownerContext) {}

    void setNativeHandle(void* native, RhiContext* ownerContext = nullptr) {
        m_native = native;
        m_ownerContext = ownerContext;
    }

    RhiContext* ownerContext() const override { return m_ownerContext; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
    RhiContext* m_ownerContext = nullptr;
};

class RhiCommandQueueHandle final : public RhiCommandQueue {
public:
    constexpr RhiCommandQueueHandle() = default;
    explicit constexpr RhiCommandQueueHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
};

class RhiNativeCommandBufferHandle final : public RhiNativeCommandBuffer {
public:
    constexpr RhiNativeCommandBufferHandle() = default;
    explicit constexpr RhiNativeCommandBufferHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
};

class RhiShaderLibraryHandle final : public RhiShaderLibrary {
public:
    constexpr RhiShaderLibraryHandle() = default;
    explicit constexpr RhiShaderLibraryHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
};

class RhiVertexDescriptorHandle final : public RhiVertexDescriptor {
public:
    constexpr RhiVertexDescriptorHandle() = default;
    explicit constexpr RhiVertexDescriptorHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
};

class RhiTextureHandle final : public RhiTexture {
public:
    constexpr RhiTextureHandle() = default;
    constexpr RhiTextureHandle(void* native, uint32_t w = 0, uint32_t h = 0)
        : m_native(native), m_width(w), m_height(h) {}

    void setNativeHandle(void* native, uint32_t w = 0, uint32_t h = 0) {
        m_native = native;
        m_width = w;
        m_height = h;
    }

    void* nativeHandle() const override { return m_native; }
    uint32_t width() const override { return m_width; }
    uint32_t height() const override { return m_height; }

private:
    void* m_native = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
};

class RhiSamplerHandle final : public RhiSampler {
public:
    constexpr RhiSamplerHandle() = default;
    explicit constexpr RhiSamplerHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
};

class RhiDepthStencilStateHandle final : public RhiDepthStencilState {
public:
    constexpr RhiDepthStencilStateHandle() = default;
    explicit constexpr RhiDepthStencilStateHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
};

class RhiAccelerationStructureHandle final : public RhiAccelerationStructure {
public:
    constexpr RhiAccelerationStructureHandle() = default;
    explicit constexpr RhiAccelerationStructureHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
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
    virtual void setViewport(float width, float height, bool flipY = true) = 0;
    virtual void setDepthStencilState(const RhiDepthStencilState* state) = 0;
    virtual void setFrontFacingWinding(RhiWinding winding) = 0;
    virtual void setCullMode(RhiCullMode cullMode) = 0;
    virtual void setRenderPipeline(const RhiGraphicsPipeline& pipeline) = 0;
    virtual void setVertexBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) = 0;
    virtual void setFragmentBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) = 0;
    virtual void setMeshBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) = 0;
    virtual void setVertexBytes(const void* data, size_t size, uint32_t index) = 0;
    virtual void setFragmentBytes(const void* data, size_t size, uint32_t index) = 0;
    virtual void setMeshBytes(const void* data, size_t size, uint32_t index) = 0;
    virtual void setPushConstants(const void* data, size_t size) = 0;
    virtual void setFragmentTexture(const RhiTexture* texture, uint32_t index) = 0;
    virtual void setFragmentTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) = 0;
    virtual void setMeshTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) = 0;
    virtual void setFragmentSampler(const RhiSampler* sampler, uint32_t index) = 0;
    virtual void setMeshSampler(const RhiSampler* sampler, uint32_t index) = 0;
    virtual void drawPrimitives(RhiPrimitiveType primitiveType, uint32_t vertexStart, uint32_t vertexCount) = 0;
    virtual void drawIndexedPrimitives(RhiPrimitiveType primitiveType,
                                       uint32_t indexCount,
                                       RhiIndexType indexType,
                                       const RhiBuffer& indexBuffer,
                                       uint64_t indexBufferOffset) = 0;
    virtual void drawMeshThreadgroups(RhiSize3D threadgroupsPerGrid,
                                      RhiSize3D threadsPerObjectThreadgroup,
                                      RhiSize3D threadsPerMeshThreadgroup) = 0;
    virtual void drawMeshThreadgroupsIndirect(const RhiBuffer& indirectBuffer,
                                              uint64_t indirectBufferOffset,
                                              RhiSize3D threadsPerObjectThreadgroup,
                                              RhiSize3D threadsPerMeshThreadgroup) = 0;
    virtual void renderImGuiDrawData() = 0;
};

class RhiComputeCommandEncoder {
public:
    virtual ~RhiComputeCommandEncoder() = default;
    virtual void* nativeHandle() const = 0;
    virtual void setComputePipeline(const RhiComputePipeline& pipeline) = 0;
    virtual void setBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) = 0;
    virtual void setBytes(const void* data, size_t size, uint32_t index) = 0;
    virtual void setPushConstants(const void* data, size_t size) = 0;
    virtual void setTexture(const RhiTexture* texture, uint32_t index) = 0;
    virtual void setStorageTexture(const RhiTexture* texture, uint32_t index) = 0;
    virtual void setTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) = 0;
    virtual void setSampler(const RhiSampler* sampler, uint32_t index) = 0;
    virtual void setAccelerationStructure(const RhiAccelerationStructure* accelerationStructure, uint32_t index) = 0;
    virtual void useResource(const RhiBuffer& resource, RhiResourceUsage usage) = 0;
    virtual void useResource(const RhiAccelerationStructure& resource, RhiResourceUsage usage) = 0;
    virtual void memoryBarrier(RhiBarrierScope scope) = 0;
    virtual void dispatchThreadgroups(RhiSize3D threadgroupsPerGrid, RhiSize3D threadsPerThreadgroup) = 0;
};

class RhiBlitCommandEncoder {
public:
    virtual ~RhiBlitCommandEncoder() = default;
    virtual void* nativeHandle() const = 0;
    virtual void copyTexture(const RhiTexture& source,
                             uint32_t sourceSlice,
                             uint32_t sourceLevel,
                             RhiOrigin3D sourceOrigin,
                             RhiSize3D sourceSize,
                             const RhiTexture& destination,
                             uint32_t destinationSlice,
                             uint32_t destinationLevel,
                             RhiOrigin3D destinationOrigin) = 0;
};

class RhiCommandBuffer {
public:
    virtual ~RhiCommandBuffer() = default;
    virtual std::unique_ptr<RhiRenderCommandEncoder> beginRenderPass(const RhiRenderPassDesc& desc) = 0;
    virtual std::unique_ptr<RhiComputeCommandEncoder> beginComputePass(const RhiComputePassDesc& desc) = 0;
    virtual std::unique_ptr<RhiBlitCommandEncoder> beginBlitPass(const RhiBlitPassDesc& desc) = 0;

    // Prepare a texture for sampling in the next pass.
    // Backend implementations handle any necessary state transitions (e.g. Vulkan layout transitions).
    virtual void prepareTextureForSampling(const RhiTexture* /*texture*/) {}

    // Flush all accumulated pending barriers as a single batched pipeline barrier.
    // No-op on backends that insert barriers eagerly (e.g. Metal).
    virtual void flushBarriers() {}

    // Hint to the backend which queue subsequent work should target.
    // Called by FrameGraph before each pass. No-op on Metal.
    virtual void setNextPassQueueHint(RhiQueueHint /*hint*/) {}
};

class RhiFrameGraphBackend {
public:
    virtual ~RhiFrameGraphBackend() = default;
    virtual std::unique_ptr<RhiTexture> createTexture(const RhiTextureDesc& desc) = 0;
    virtual std::unique_ptr<RhiBuffer> createBuffer(const RhiBufferDesc& desc) = 0;
};

struct RhiFeatures {
    bool dynamicRendering = false;
    bool bufferDeviceAddress = false;
    bool meshShaders = false;
    bool rayTracing = false;
    bool validation = false;
    bool synchronization2 = false;
    bool shaderDrawParameters = false;
    bool taskShaders = false;
    bool descriptorIndexing = false;
    bool timelineSemaphore = false;
    bool externalHostMemory = false;
};

struct RhiLimits {
    // Push constants
    uint32_t maxPushConstantSize = 128;

    // Uniform buffers
    uint64_t minUniformBufferOffsetAlignment = 256;
    uint32_t maxUniformBufferRange = 65536;

    // Storage buffers
    uint32_t maxStorageBufferRange = 0;

    // Compute
    uint32_t maxComputeWorkGroupCountX = 65535;
    uint32_t maxComputeWorkGroupCountY = 65535;
    uint32_t maxComputeWorkGroupCountZ = 65535;
    uint32_t maxComputeWorkGroupInvocations = 1024;
    uint32_t maxComputeWorkGroupSizeX = 1024;
    uint32_t maxComputeWorkGroupSizeY = 1024;
    uint32_t maxComputeWorkGroupSizeZ = 64;

    // Mesh shaders
    uint32_t maxMeshOutputVertices = 0;
    uint32_t maxMeshOutputPrimitives = 0;
    uint32_t maxMeshWorkGroupInvocations = 0;
    uint32_t maxTaskWorkGroupInvocations = 0;

    // Descriptors
    uint32_t maxBoundDescriptorSets = 4;
    uint32_t maxPerStageDescriptorSamplers = 16;
    uint32_t maxPerStageDescriptorUniformBuffers = 12;
    uint32_t maxPerStageDescriptorStorageBuffers = 8;
    uint32_t maxPerStageDescriptorSampledImages = 16;
    uint32_t maxPerStageDescriptorStorageImages = 4;
    uint32_t maxDescriptorSetSamplers = 0;
    uint32_t maxDescriptorSetUniformBuffers = 0;
    uint32_t maxDescriptorSetStorageBuffers = 0;
    uint32_t maxDescriptorSetSampledImages = 0;
    uint32_t maxDescriptorSetStorageImages = 0;

    // Memory
    uint64_t nonCoherentAtomSize = 256;

    // Textures / framebuffer
    uint32_t maxImageDimension2D = 4096;
    uint32_t maxFramebufferWidth = 4096;
    uint32_t maxFramebufferHeight = 4096;
    uint32_t maxColorAttachments = 4;

    // Timing
    float timestampPeriod = 0.0f;
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

    // Extra Vulkan extensions requested by external integrations (e.g. Streamline)
    std::vector<const char*> extraInstanceExtensions;
    std::vector<const char*> extraDeviceExtensions;
    bool enableTimelineSemaphore = false;

    // Optional Vulkan proxy lookup used by integrations such as Streamline manual hooking.
    void* vkGetDeviceProcAddrProxy = nullptr;
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

struct RhiSamplerDesc {
    RhiSamplerFilterMode minFilter = RhiSamplerFilterMode::Linear;
    RhiSamplerFilterMode magFilter = RhiSamplerFilterMode::Linear;
    RhiSamplerMipFilterMode mipFilter = RhiSamplerMipFilterMode::None;
    RhiSamplerAddressMode addressModeS = RhiSamplerAddressMode::Repeat;
    RhiSamplerAddressMode addressModeT = RhiSamplerAddressMode::Repeat;
    RhiSamplerAddressMode addressModeR = RhiSamplerAddressMode::Repeat;
};

class RhiShaderModule {
public:
    virtual ~RhiShaderModule() = default;
};

class RhiBuffer {
public:
    virtual ~RhiBuffer() = default;
    virtual size_t size() const = 0;
    virtual void* nativeHandle() const = 0;
    virtual void* mappedData() { return nullptr; }
};

class RhiBufferHandle final : public RhiBuffer {
public:
    constexpr RhiBufferHandle() = default;
    constexpr RhiBufferHandle(void* native, size_t byteSize = 0)
        : m_native(native), m_size(byteSize) {}

    void setNativeHandle(void* native, size_t byteSize = 0) {
        m_native = native;
        m_size = byteSize;
    }

    size_t size() const override { return m_size; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
    size_t m_size = 0;
};

class RhiGraphicsPipeline {
public:
    virtual ~RhiGraphicsPipeline() = default;
    virtual void* nativeHandle() const = 0;
};

class RhiGraphicsPipelineHandle final : public RhiGraphicsPipeline {
public:
    constexpr RhiGraphicsPipelineHandle() = default;
    explicit constexpr RhiGraphicsPipelineHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
};

class RhiComputePipelineHandle final : public RhiComputePipeline {
public:
    constexpr RhiComputePipelineHandle() = default;
    explicit constexpr RhiComputePipelineHandle(void* native)
        : m_native(native) {}

    void setNativeHandle(void* native) { m_native = native; }
    void* nativeHandle() const override { return m_native; }

private:
    void* m_native = nullptr;
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

struct RhiRenderPipelineSourceDesc {
    const char* vertexEntry = nullptr;
    const char* meshEntry = nullptr;
    const char* fragmentEntry = nullptr;
    RhiFormat colorFormat = RhiFormat::BGRA8Unorm;
    RhiFormat depthFormat = RhiFormat::Undefined;
    const RhiVertexDescriptor* vertexDescriptor = nullptr;
};

struct RhiShaderLibrarySourceDesc {
    uint32_t languageVersion = 0;
};

struct RhiNativeHandles {
    void* instance = nullptr;
    void* physicalDevice = nullptr;
    void* device = nullptr;
    void* queue = nullptr;
    void* descriptorPool = nullptr;
    void* allocator = nullptr;  // VmaAllocator for Vulkan, unused for Metal
    uint32_t graphicsQueueFamily = 0;
    uint32_t computeQueueFamily = UINT32_MAX;   // UINT32_MAX = unavailable
    uint32_t transferQueueFamily = UINT32_MAX;  // UINT32_MAX = unavailable
    uint32_t swapchainImageCount = 0;
    uint32_t colorFormat = 0;
    uint32_t apiVersion = 0;
    // Multi-queue handles (null if queue not available)
    void* computeQueue = nullptr;                 // VkQueue (dedicated compute, if present)
    void* transferQueue = nullptr;                // VkQueue (dedicated transfer, if present)
    void* computeTimelineSemaphore = nullptr;     // VkSemaphore (timeline, for async compute sync)
    void* transferTimelineSemaphore = nullptr;    // VkSemaphore (timeline, for async transfer sync)
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
    virtual const RhiLimits& limits() const = 0;
    virtual const RhiDeviceInfo& deviceInfo() const = 0;
    virtual const RhiNativeHandles& nativeHandles() const = 0;
    virtual const IRhiInteropProvider* interopProvider() const { return nullptr; }
    virtual bool beginFrame() = 0;
    virtual void endFrame() = 0;
    virtual void resize(uint32_t width, uint32_t height) = 0;
    virtual void waitIdle() = 0;
    virtual RhiCommandContext& commandContext() = 0;
    virtual uint32_t drawableWidth() const = 0;
    virtual uint32_t drawableHeight() const = 0;
    virtual RhiFormat colorFormat() const = 0;
    virtual RhiBufferHandle createSharedBuffer(const void* initialData,
                                               size_t size,
                                               const char* debugName = nullptr) = 0;
    virtual RhiTextureHandle createTexture2D(uint32_t width,
                                             uint32_t height,
                                             RhiFormat format,
                                             bool mipmapped,
                                             uint32_t mipLevelCount,
                                             RhiTextureStorageMode storageMode,
                                             RhiTextureUsage usage) = 0;
    virtual RhiTextureHandle createTexture3D(uint32_t width,
                                             uint32_t height,
                                             uint32_t depth,
                                             RhiFormat format,
                                             RhiTextureStorageMode storageMode,
                                             RhiTextureUsage usage) = 0;
    virtual RhiSamplerHandle createSampler(const RhiSamplerDesc& desc) = 0;
    virtual RhiDepthStencilStateHandle createDepthStencilState(bool depthWriteEnabled,
                                                               bool reversedZ) = 0;
    virtual RhiShaderLibraryHandle createShaderLibraryFromSource(const std::string& source,
                                                                 const RhiShaderLibrarySourceDesc& desc,
                                                                 std::string& errorMessage) = 0;
    virtual RhiComputePipelineHandle createComputePipelineFromLibrary(const RhiShaderLibrary& library,
                                                                      const char* entryPoint,
                                                                      std::string& errorMessage) = 0;
    virtual RhiGraphicsPipelineHandle createRenderPipelineFromSource(const std::string& source,
                                                                     const RhiRenderPipelineSourceDesc& desc,
                                                                     std::string& errorMessage) = 0;
    virtual RhiComputePipelineHandle createComputePipelineFromSource(const std::string& source,
                                                                     const char* entryPoint,
                                                                     std::string& errorMessage) = 0;
    virtual std::unique_ptr<RhiShaderModule> createShaderModule(const RhiShaderModuleDesc& desc) = 0;
    virtual std::unique_ptr<RhiBuffer> createVertexBuffer(const RhiBufferDesc& desc) = 0;
    virtual std::unique_ptr<RhiGraphicsPipeline> createGraphicsPipeline(const RhiGraphicsPipelineDesc& desc) = 0;
};

std::unique_ptr<RhiContext> createRhiContext(RhiBackendType backend,
                                             const RhiCreateInfo& createInfo,
                                             std::string& errorMessage);
