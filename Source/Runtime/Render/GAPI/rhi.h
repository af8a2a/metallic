#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

namespace metallic::render {

enum class Error : int8_t {
    Failure = 1,
    InvalidArgument,
    OutOfMemory,
    Unsupported,
    OutOfDate,
    DeviceLost,
};

using Result = std::expected<void, Error>;

[[nodiscard]] inline Result makeError(Error error)
{
    return std::unexpected(error);
}

[[nodiscard]] inline bool hasError(const Result& result, Error error)
{
    return !result.has_value() && result.error() == error;
}

[[nodiscard]] constexpr const char* errorToString(Error error)
{
    switch (error) {
    case Error::Failure:
        return "Failure";
    case Error::InvalidArgument:
        return "InvalidArgument";
    case Error::OutOfMemory:
        return "OutOfMemory";
    case Error::Unsupported:
        return "Unsupported";
    case Error::OutOfDate:
        return "OutOfDate";
    case Error::DeviceLost:
        return "DeviceLost";
    }

    return "Unknown";
}

[[nodiscard]] inline const char* resultToString(const Result& result)
{
    return result.has_value() ? "Success" : errorToString(result.error());
}

enum class WindowSystem : uint8_t {
    Sdl3,
};

struct WindowHandle {
    WindowSystem system = WindowSystem::Sdl3;
    void* nativeWindow = nullptr;
};

enum class Format : uint16_t {
    Unknown,
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    Rg8Unorm,
    Rg8Snorm,
    Rg8Uint,
    Rg8Sint,
    Bgra8Unorm,
    Bgra8Srgb,
    Rgba8Unorm,
    Rgba8Snorm,
    Rgba8Srgb,
    Rgba8Uint,
    Rgba8Sint,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16Sfloat,
    Rg16Unorm,
    Rg16Snorm,
    Rg16Uint,
    Rg16Sint,
    Rg16Sfloat,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgba16Uint,
    Rgba16Sint,
    Rgba16Sfloat,
    R32Uint,
    R32Sint,
    R32Sfloat,
    Rg32Uint,
    Rg32Sint,
    Rg32Sfloat,
    Rgb32Uint,
    Rgb32Sint,
    Rgb32Sfloat,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Sfloat,
    A2B10G10R10UnormPack32,
    A2R10G10B10UintPack32,
    B10G11R11UfloatPack32,
    E5B9G9R9UfloatPack32,
    D32Sfloat,
};

enum class QueueType : uint8_t {
    Graphics,
    Compute,
    Copy,
};

enum class MemoryLocation : uint8_t {
    Device,
    HostUpload,
    HostReadback,
};

enum class ResourceState : uint8_t {
    Undefined,
    Present,
    ColorAttachment,
    DepthStencilAttachment,
    ShaderRead,
    TransferSource,
    TransferDestination,
    General,
};

enum class PipelineStageBits : uint64_t {
    None = 0,
    TopOfPipe = 1ull << 0,
    DrawIndirect = 1ull << 1,
    VertexShader = 1ull << 2,
    FragmentShader = 1ull << 3,
    ComputeShader = 1ull << 4,
    ColorAttachment = 1ull << 5,
    Transfer = 1ull << 6,
    BottomOfPipe = 1ull << 7,
    AllCommands = 1ull << 8,
};

enum class BufferUsageBits : uint32_t {
    None = 0,
    Vertex = 1u << 0,
    Index = 1u << 1,
    Constant = 1u << 2,
    Storage = 1u << 3,
    TransferSource = 1u << 4,
    TransferDestination = 1u << 5,
    ShaderDeviceAddress = 1u << 6,
    AccelerationStructureBuildInput = 1u << 7,
    AccelerationStructureStorage = 1u << 8,
};

enum class TextureUsageBits : uint32_t {
    None = 0,
    Sampled = 1u << 0,
    Storage = 1u << 1,
    ColorAttachment = 1u << 2,
    DepthStencilAttachment = 1u << 3,
    TransferSource = 1u << 4,
    TransferDestination = 1u << 5,
    Present = 1u << 6,
};

enum class TextureType : uint8_t {
    Texture1D,
    Texture2D,
    Texture3D,
};

enum class LoadOp : uint8_t {
    Load,
    Clear,
    DontCare,
};

enum class StoreOp : uint8_t {
    Store,
    DontCare,
};

enum class CompareOp : uint8_t {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
};

enum class PrimitiveTopology : uint8_t {
    TriangleList,
};

constexpr PipelineStageBits operator|(PipelineStageBits lhs, PipelineStageBits rhs)
{
    return static_cast<PipelineStageBits>(
        static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

constexpr BufferUsageBits operator|(BufferUsageBits lhs, BufferUsageBits rhs)
{
    return static_cast<BufferUsageBits>(
        static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr TextureUsageBits operator|(TextureUsageBits lhs, TextureUsageBits rhs)
{
    return static_cast<TextureUsageBits>(
        static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr bool hasFlag(BufferUsageBits value, BufferUsageBits flag)
{
    return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

constexpr bool hasFlag(TextureUsageBits value, TextureUsageBits flag)
{
    return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

struct ColorValue {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;
};

struct Rect {
    int32_t x = 0;
    int32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct DebugLabelDesc {
    const char* name = nullptr;
    ColorValue color{0.35f, 0.55f, 1.0f, 1.0f};
};

struct DeviceDesc {
    const char* applicationName = "Metallic";
    bool enableValidation = false;
    bool enableBindlessDescriptorHeap = false;
    bool enableShaderObject = false;
    bool enableMeshShader = false;
    bool enableRayTracingAccelerationStructure = false;
    bool enableRayQuery = false;
    bool enablePushDescriptor = false;
    bool enableClusterAccelerationStructure = false;
    bool enableStreamline = false;
    bool enableAftermath = false;
};

struct DeviceCapabilities {
    bool bindlessDescriptorHeap = false;
    bool shaderObject = false;
    bool meshShader = false;
    bool rayTracingAccelerationStructure = false;
    bool rayQuery = false;
    bool pushDescriptor = false;
    bool clusterAccelerationStructure = false;
    bool streamline = false;
    bool streamlineDlssRr = false;
    bool aftermath = false;
    uint32_t maxBindlessSamplers = 0;
    uint32_t maxBindlessSampledImages = 0;
    uint32_t maxBindlessBuffers = 0;
};

struct BufferDesc {
    uint64_t size = 0;
    uint32_t structureStride = 0;
    BufferUsageBits usage = BufferUsageBits::None;
    MemoryLocation memoryLocation = MemoryLocation::Device;
};

enum class BufferViewType : uint8_t {
    Constant,
    Structured,
    Raw,
    ReadWriteStructured,
    ReadWriteRaw,
};

struct BufferViewDesc {
    BufferViewType type = BufferViewType::Raw;
    uint64_t offset = 0;
    uint64_t size = UINT64_MAX;
    uint32_t structureStride = 0;
};

struct TextureDesc {
    TextureType type = TextureType::Texture2D;
    TextureUsageBits usage = TextureUsageBits::None;
    Format format = Format::Unknown;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
    uint32_t mipCount = 1;
    uint32_t layerCount = 1;
    MemoryLocation memoryLocation = MemoryLocation::Device;
};

struct TextureViewDesc {
    Format format = Format::Unknown;
    uint32_t baseMip = 0;
    uint32_t mipCount = 1;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;
};

struct SwapchainDesc {
    WindowHandle window;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t imageCount = 3;
    uint32_t framesInFlight = 2;
    Format format = Format::Bgra8Srgb;
    bool vsync = true;
};

struct TextureBarrierDesc {
    class Texture* texture = nullptr;
    ResourceState before = ResourceState::Undefined;
    ResourceState after = ResourceState::Undefined;
    uint32_t baseMip = 0;
    uint32_t mipCount = 1;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;
};

struct BufferBarrierDesc {
    class Buffer* buffer = nullptr;
    ResourceState before = ResourceState::Undefined;
    ResourceState after = ResourceState::Undefined;
    uint64_t offset = 0;
    uint64_t size = UINT64_MAX;
};

struct BarrierDesc {
    const TextureBarrierDesc* textures = nullptr;
    uint32_t textureCount = 0;
    const BufferBarrierDesc* buffers = nullptr;
    uint32_t bufferCount = 0;
};

struct RenderingAttachmentDesc {
    class TextureView* view = nullptr;
    ResourceState state = ResourceState::ColorAttachment;
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    ColorValue clearColor;
    float clearDepth = 1.0f;
    uint32_t clearStencil = 0;
};

struct RenderingDesc {
    Rect renderArea;
    const RenderingAttachmentDesc* colorAttachments = nullptr;
    uint32_t colorAttachmentCount = 0;
    const RenderingAttachmentDesc* depthStencilAttachment = nullptr;
};

struct SemaphoreSubmitDesc {
    class Semaphore* semaphore = nullptr;
    PipelineStageBits stages = PipelineStageBits::AllCommands;
};

struct QueueSubmitDesc {
    const SemaphoreSubmitDesc* waitSemaphores = nullptr;
    uint32_t waitSemaphoreCount = 0;
    class CommandBuffer* const* commandBuffers = nullptr;
    uint32_t commandBufferCount = 0;
    const SemaphoreSubmitDesc* signalSemaphores = nullptr;
    uint32_t signalSemaphoreCount = 0;
    class Fence* signalFence = nullptr;
};

struct Viewport {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float minDepth = 0.0f;
    float maxDepth = 1.0f;
};

struct DepthStencilState {
    bool depthTestEnable = false;
    bool depthWriteEnable = false;
    CompareOp depthCompareOp = CompareOp::LessEqual;
};

struct ShaderModuleDesc {
    const uint32_t* code = nullptr;
    uint64_t byteSize = 0;
};

struct GraphicsPipelineDesc {
    class ShaderModule* vertexShader = nullptr;
    class ShaderModule* meshShader = nullptr;
    class ShaderModule* fragmentShader = nullptr;
    const char* vertexEntryPoint = "main";
    const char* meshEntryPoint = "main";
    const char* fragmentEntryPoint = "main";
    Format colorFormat = Format::Unknown;
    Format depthStencilFormat = Format::Unknown;
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    DepthStencilState depthStencil;
    bool usesBindlessHeap = false;
};

struct ComputePipelineDesc {
    class ShaderModule* computeShader = nullptr;
    const char* computeEntryPoint = "main";
    bool usesBindlessHeap = false;
    uint32_t bindlessUserPushDataSize = 0;
};

struct GraphicsShaderObjectProgramDesc {
    const uint32_t* vertexCode = nullptr;
    uint64_t vertexByteSize = 0;
    const char* vertexEntryPoint = "main";
    const uint32_t* fragmentCode = nullptr;
    uint64_t fragmentByteSize = 0;
    const char* fragmentEntryPoint = "main";
    bool usesBindlessHeap = false;
    uint32_t bindlessUserPushDataSize = 0;
};

struct TextureBufferCopyDesc {
    class Texture* texture = nullptr;
    class Buffer* buffer = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mipLevel = 0;
    uint32_t baseLayer = 0;
};

struct BufferTextureCopyDesc {
    class Buffer* buffer = nullptr;
    class Texture* texture = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mipLevel = 0;
    uint32_t baseLayer = 0;
};

struct TextureCopyDesc {
    class Texture* source = nullptr;
    class Texture* destination = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t sourceMipLevel = 0;
    uint32_t sourceBaseLayer = 0;
    uint32_t destinationMipLevel = 0;
    uint32_t destinationBaseLayer = 0;
};

struct BindlessHeapDesc {
    uint32_t maxSamplers = 0;
    uint32_t maxSampledImages = 0;
    uint32_t maxBuffers = 0;
};

enum class BindlessHandleKind : uint8_t {
    Invalid,
    Sampler,
    SampledImage,
    Buffer,
};

struct BindlessHandle {
    BindlessHandleKind kind = BindlessHandleKind::Invalid;
    uint32_t index = UINT32_MAX;
    uint32_t shaderIndex = UINT32_MAX;

    bool valid() const { return kind != BindlessHandleKind::Invalid && index != UINT32_MAX; }
};

namespace detail {
struct DeviceImpl;
struct QueueImpl;
struct SwapchainImpl;
struct CommandPoolImpl;
struct CommandBufferImpl;
struct FenceImpl;
struct SemaphoreImpl;
struct BufferImpl;
struct BufferViewImpl;
struct TextureImpl;
struct TextureViewImpl;
struct ShaderModuleImpl;
struct GraphicsPipelineImpl;
struct ComputePipelineImpl;
struct GraphicsShaderObjectProgramImpl;
struct BindlessHeapImpl;
struct TrianglePreviewRendererImpl;
struct VulkanNativeAccess;
} // namespace detail

class Queue {
public:
    Queue() = default;
    ~Queue();
    Queue(Queue&&) noexcept;
    Queue& operator=(Queue&&) noexcept;

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    Result submit(const QueueSubmitDesc& desc);
    Result waitIdle();
    QueueType type() const;

private:
    explicit Queue(std::unique_ptr<detail::QueueImpl> impl);

    std::unique_ptr<detail::QueueImpl> impl_;

    friend class Device;
    friend class Swapchain;
    friend class CommandPool;
    friend struct detail::DeviceImpl;
    friend struct detail::VulkanNativeAccess;
};

class Fence {
public:
    Fence() = default;
    ~Fence();
    Fence(Fence&&) noexcept;
    Fence& operator=(Fence&&) noexcept;

    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;

    Result wait(uint64_t timeoutNanoseconds = UINT64_MAX);
    Result reset();
    bool isSignaled() const;

private:
    explicit Fence(std::unique_ptr<detail::FenceImpl> impl);

    std::unique_ptr<detail::FenceImpl> impl_;

    friend class Device;
    friend class Queue;
    friend struct detail::DeviceImpl;
};

class Semaphore {
public:
    Semaphore() = default;
    ~Semaphore();
    Semaphore(Semaphore&&) noexcept;
    Semaphore& operator=(Semaphore&&) noexcept;

    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;

private:
    explicit Semaphore(std::unique_ptr<detail::SemaphoreImpl> impl);

    std::unique_ptr<detail::SemaphoreImpl> impl_;

    friend class Device;
    friend class Queue;
    friend class Swapchain;
    friend struct detail::DeviceImpl;
    friend struct detail::VulkanNativeAccess;
};

class Buffer {
public:
    Buffer() = default;
    ~Buffer();
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    const BufferDesc& desc() const;
    void* map();
    void unmap();
    void flush(uint64_t offset = 0, uint64_t size = UINT64_MAX);
    void invalidate(uint64_t offset = 0, uint64_t size = UINT64_MAX);

private:
    explicit Buffer(std::unique_ptr<detail::BufferImpl> impl);

    std::unique_ptr<detail::BufferImpl> impl_;

    friend class Device;
    friend class CommandBuffer;
    friend class BufferView;
    friend class BindlessHeap;
    friend struct detail::DeviceImpl;
    friend struct detail::VulkanNativeAccess;
};

class BufferView {
public:
    BufferView() = default;
    ~BufferView();
    BufferView(BufferView&&) noexcept;
    BufferView& operator=(BufferView&&) noexcept;

    BufferView(const BufferView&) = delete;
    BufferView& operator=(const BufferView&) = delete;

    const BufferViewDesc& desc() const;

private:
    explicit BufferView(std::unique_ptr<detail::BufferViewImpl> impl);

    std::unique_ptr<detail::BufferViewImpl> impl_;

    friend class Device;
    friend class BindlessHeap;
    friend struct detail::DeviceImpl;
    friend struct detail::VulkanNativeAccess;
};

class Texture {
public:
    Texture() = default;
    ~Texture();
    Texture(Texture&&) noexcept;
    Texture& operator=(Texture&&) noexcept;

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    const TextureDesc& desc() const;

private:
    explicit Texture(std::unique_ptr<detail::TextureImpl> impl);

    std::unique_ptr<detail::TextureImpl> impl_;

    friend class Device;
    friend class Swapchain;
    friend class CommandBuffer;
    friend class TextureView;
    friend class BindlessHeap;
    friend struct detail::DeviceImpl;
    friend struct detail::SwapchainImpl;
    friend struct detail::VulkanNativeAccess;
};

class TextureView {
public:
    TextureView() = default;
    ~TextureView();
    TextureView(TextureView&&) noexcept;
    TextureView& operator=(TextureView&&) noexcept;

    TextureView(const TextureView&) = delete;
    TextureView& operator=(const TextureView&) = delete;

private:
    explicit TextureView(std::unique_ptr<detail::TextureViewImpl> impl);

    std::unique_ptr<detail::TextureViewImpl> impl_;

    friend class Device;
    friend class CommandBuffer;
    friend class BindlessHeap;
    friend struct detail::DeviceImpl;
    friend struct detail::VulkanNativeAccess;
};

class ShaderModule {
public:
    ShaderModule() = default;
    ~ShaderModule();
    ShaderModule(ShaderModule&&) noexcept;
    ShaderModule& operator=(ShaderModule&&) noexcept;

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

private:
    explicit ShaderModule(std::unique_ptr<detail::ShaderModuleImpl> impl);

    std::unique_ptr<detail::ShaderModuleImpl> impl_;

    friend class Device;
    friend struct detail::DeviceImpl;
};

class GraphicsPipeline {
public:
    GraphicsPipeline() = default;
    ~GraphicsPipeline();
    GraphicsPipeline(GraphicsPipeline&&) noexcept;
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept;

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

private:
    explicit GraphicsPipeline(std::unique_ptr<detail::GraphicsPipelineImpl> impl);

    std::unique_ptr<detail::GraphicsPipelineImpl> impl_;

    friend class Device;
    friend class CommandBuffer;
    friend struct detail::DeviceImpl;
};

class ComputePipeline {
public:
    ComputePipeline() = default;
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&&) noexcept;
    ComputePipeline& operator=(ComputePipeline&&) noexcept;

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

private:
    explicit ComputePipeline(std::unique_ptr<detail::ComputePipelineImpl> impl);

    std::unique_ptr<detail::ComputePipelineImpl> impl_;

    friend class Device;
    friend class CommandBuffer;
    friend struct detail::DeviceImpl;
};

class GraphicsShaderObjectProgram {
public:
    GraphicsShaderObjectProgram() = default;
    ~GraphicsShaderObjectProgram();
    GraphicsShaderObjectProgram(GraphicsShaderObjectProgram&&) noexcept;
    GraphicsShaderObjectProgram& operator=(GraphicsShaderObjectProgram&&) noexcept;

    GraphicsShaderObjectProgram(const GraphicsShaderObjectProgram&) = delete;
    GraphicsShaderObjectProgram& operator=(const GraphicsShaderObjectProgram&) = delete;

private:
    explicit GraphicsShaderObjectProgram(std::unique_ptr<detail::GraphicsShaderObjectProgramImpl> impl);

    std::unique_ptr<detail::GraphicsShaderObjectProgramImpl> impl_;

    friend class Device;
    friend class CommandBuffer;
    friend struct detail::DeviceImpl;
};

class BindlessHeap {
public:
    BindlessHeap() = default;
    ~BindlessHeap();
    BindlessHeap(BindlessHeap&&) noexcept;
    BindlessHeap& operator=(BindlessHeap&&) noexcept;

    BindlessHeap(const BindlessHeap&) = delete;
    BindlessHeap& operator=(const BindlessHeap&) = delete;

    const BindlessHeapDesc& desc() const;
    uint32_t imageShaderIndexBase() const;
    uint32_t bufferShaderIndexBase() const;

    Result allocateSampledImage(BindlessHandle& outHandle);
    Result allocateBuffer(BindlessHandle& outHandle);
    void release(BindlessHandle handle);
    Result writeSampledImage(BindlessHandle handle, TextureView& view, ResourceState state = ResourceState::ShaderRead);
    Result writeBufferView(BindlessHandle handle, BufferView& view);
    Result writeConstantBuffer(BindlessHandle handle, Buffer& buffer);
    Result writeStorageBuffer(BindlessHandle handle, Buffer& buffer);

private:
    explicit BindlessHeap(std::unique_ptr<detail::BindlessHeapImpl> impl);

    std::unique_ptr<detail::BindlessHeapImpl> impl_;

    friend class Device;
    friend class CommandBuffer;
    friend struct detail::DeviceImpl;
};

class CommandBuffer {
public:
    CommandBuffer() = default;
    ~CommandBuffer();
    CommandBuffer(CommandBuffer&&) noexcept;
    CommandBuffer& operator=(CommandBuffer&&) noexcept;

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    Result begin();
    Result end();
    void beginDebugLabel(const DebugLabelDesc& desc);
    void endDebugLabel();
    void barrier(const BarrierDesc& desc);
    void copyTexture(const TextureCopyDesc& desc);
    void copyTextureToBuffer(const TextureBufferCopyDesc& desc);
    void copyBufferToTexture(const BufferTextureCopyDesc& desc);
    void beginRendering(const RenderingDesc& desc);
    void clearColorAttachment(uint32_t attachmentIndex, const ColorValue& color, const Rect& rect);
    void endRendering();
    void setViewport(const Viewport& viewport);
    void setScissor(const Rect& scissor);
    void setDepthStencilState(const DepthStencilState& state);
    void bindGraphicsPipeline(GraphicsPipeline& pipeline);
    void bindComputePipeline(ComputePipeline& pipeline);
    void setGraphicsShaderObjectState();
    void bindGraphicsShaderObjectProgram(GraphicsShaderObjectProgram& program);
    void bindBindlessHeap(BindlessHeap& heap);
    void pushBindlessData(const void* data, uint32_t byteSize);
    void draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t firstInstance = 0);
    void drawMeshTasks(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    void dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

private:
    explicit CommandBuffer(std::unique_ptr<detail::CommandBufferImpl> impl);

    std::unique_ptr<detail::CommandBufferImpl> impl_;

    friend class CommandPool;
    friend class Queue;
    friend struct detail::CommandPoolImpl;
    friend struct detail::VulkanNativeAccess;
};

class CommandPool {
public:
    CommandPool() = default;
    ~CommandPool();
    CommandPool(CommandPool&&) noexcept;
    CommandPool& operator=(CommandPool&&) noexcept;

    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;

    Result reset();
    Result createCommandBuffer(std::unique_ptr<CommandBuffer>& outCommandBuffer);

private:
    explicit CommandPool(std::unique_ptr<detail::CommandPoolImpl> impl);

    std::unique_ptr<detail::CommandPoolImpl> impl_;

    friend class Device;
    friend struct detail::DeviceImpl;
    friend struct detail::VulkanNativeAccess;
};

class Swapchain {
public:
    Swapchain() = default;
    ~Swapchain();
    Swapchain(Swapchain&&) noexcept;
    Swapchain& operator=(Swapchain&&) noexcept;

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    uint32_t imageCount() const;
    uint32_t width() const;
    uint32_t height() const;
    Format format() const;
    Texture* texture(uint32_t imageIndex);
    Result acquireNextImage(Semaphore& semaphore, uint32_t& imageIndex);
    Result present(Queue& queue, uint32_t imageIndex, Semaphore& waitSemaphore);

private:
    explicit Swapchain(std::unique_ptr<detail::SwapchainImpl> impl);

    std::unique_ptr<detail::SwapchainImpl> impl_;

    friend class Device;
    friend struct detail::DeviceImpl;
    friend struct detail::VulkanNativeAccess;
};

class Device {
public:
    Device() = default;
    ~Device();
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    const DeviceCapabilities& capabilities() const;
    Queue* getQueue(QueueType type, uint32_t index = 0);
    Result waitIdle();
    Result createSwapchain(const SwapchainDesc& desc, std::unique_ptr<Swapchain>& outSwapchain);
    Result createCommandPool(Queue& queue, std::unique_ptr<CommandPool>& outCommandPool);
    Result createFence(bool signaled, std::unique_ptr<Fence>& outFence);
    Result createSemaphore(std::unique_ptr<Semaphore>& outSemaphore);
    Result createBuffer(const BufferDesc& desc, std::unique_ptr<Buffer>& outBuffer);
    Result createBufferView(Buffer& buffer, const BufferViewDesc& desc, std::unique_ptr<BufferView>& outBufferView);
    Result createTexture(const TextureDesc& desc, std::unique_ptr<Texture>& outTexture);
    Result createTextureView(Texture& texture, const TextureViewDesc& desc, std::unique_ptr<TextureView>& outTextureView);
    Result createShaderModule(const ShaderModuleDesc& desc, std::unique_ptr<ShaderModule>& outShaderModule);
    Result createGraphicsPipeline(const GraphicsPipelineDesc& desc, std::unique_ptr<GraphicsPipeline>& outGraphicsPipeline);
    Result createComputePipeline(const ComputePipelineDesc& desc, std::unique_ptr<ComputePipeline>& outComputePipeline);
    Result createGraphicsShaderObjectProgram(
        const GraphicsShaderObjectProgramDesc& desc,
        std::unique_ptr<GraphicsShaderObjectProgram>& outProgram);
    Result createBindlessHeap(const BindlessHeapDesc& desc, std::unique_ptr<BindlessHeap>& outBindlessHeap);

private:
    explicit Device(std::unique_ptr<detail::DeviceImpl> impl);

    std::unique_ptr<detail::DeviceImpl> impl_;

    friend Result createDevice(const DeviceDesc& desc, std::unique_ptr<Device>& outDevice);
    friend struct detail::VulkanNativeAccess;
};

Result createDevice(const DeviceDesc& desc, std::unique_ptr<Device>& outDevice);
int runRhiSmokeTest(bool enableValidation);
int runRhiTrianglePreviewTest(bool enableValidation);
int runRhiBindlessDescriptorHeapSmokeTest(bool enableValidation);

class TrianglePreviewRenderer {
public:
    TrianglePreviewRenderer();
    ~TrianglePreviewRenderer();

    TrianglePreviewRenderer(TrianglePreviewRenderer&&) noexcept;
    TrianglePreviewRenderer& operator=(TrianglePreviewRenderer&&) noexcept;

    TrianglePreviewRenderer(const TrianglePreviewRenderer&) = delete;
    TrianglePreviewRenderer& operator=(const TrianglePreviewRenderer&) = delete;

    Result initialize(bool enableValidation = false);
    Result render(uint32_t width, uint32_t height);
    const std::vector<uint32_t>& pixels() const;
    uint32_t width() const;
    uint32_t height() const;

private:
    std::unique_ptr<detail::TrianglePreviewRendererImpl> impl_;
};

} // namespace metallic::render
