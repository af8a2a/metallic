#pragma once

#include <cstdint>
#include <memory>

namespace metallic::render {

enum class Result : int8_t {
    Success = 0,
    Failure,
    InvalidArgument,
    OutOfMemory,
    Unsupported,
    OutOfDate,
    DeviceLost,
};

enum class WindowSystem : uint8_t {
    Sdl3,
};

struct WindowHandle {
    WindowSystem system = WindowSystem::Sdl3;
    void* nativeWindow = nullptr;
};

enum class Format : uint16_t {
    Unknown,
    Bgra8Unorm,
    Bgra8Srgb,
    Rgba8Unorm,
    Rgba8Srgb,
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

struct DeviceDesc {
    const char* applicationName = "Metallic";
    bool enableValidation = false;
};

struct BufferDesc {
    uint64_t size = 0;
    uint32_t structureStride = 0;
    BufferUsageBits usage = BufferUsageBits::None;
    MemoryLocation memoryLocation = MemoryLocation::Device;
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

struct BarrierDesc {
    const TextureBarrierDesc* textures = nullptr;
    uint32_t textureCount = 0;
};

struct RenderingAttachmentDesc {
    class TextureView* view = nullptr;
    ResourceState state = ResourceState::ColorAttachment;
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    ColorValue clearColor;
};

struct RenderingDesc {
    Rect renderArea;
    const RenderingAttachmentDesc* colorAttachments = nullptr;
    uint32_t colorAttachmentCount = 0;
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

namespace detail {
struct DeviceImpl;
struct QueueImpl;
struct SwapchainImpl;
struct CommandPoolImpl;
struct CommandBufferImpl;
struct FenceImpl;
struct SemaphoreImpl;
struct BufferImpl;
struct TextureImpl;
struct TextureViewImpl;
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

private:
    explicit Buffer(std::unique_ptr<detail::BufferImpl> impl);

    std::unique_ptr<detail::BufferImpl> impl_;

    friend class Device;
    friend struct detail::DeviceImpl;
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
    friend struct detail::DeviceImpl;
    friend struct detail::SwapchainImpl;
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
    void barrier(const BarrierDesc& desc);
    void beginRendering(const RenderingDesc& desc);
    void clearColorAttachment(uint32_t attachmentIndex, const ColorValue& color, const Rect& rect);
    void endRendering();

private:
    explicit CommandBuffer(std::unique_ptr<detail::CommandBufferImpl> impl);

    std::unique_ptr<detail::CommandBufferImpl> impl_;

    friend class CommandPool;
    friend class Queue;
    friend struct detail::CommandPoolImpl;
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
};

class Device {
public:
    Device() = default;
    ~Device();
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Queue* getQueue(QueueType type, uint32_t index = 0);
    Result waitIdle();
    Result createSwapchain(const SwapchainDesc& desc, std::unique_ptr<Swapchain>& outSwapchain);
    Result createCommandPool(Queue& queue, std::unique_ptr<CommandPool>& outCommandPool);
    Result createFence(bool signaled, std::unique_ptr<Fence>& outFence);
    Result createSemaphore(std::unique_ptr<Semaphore>& outSemaphore);
    Result createBuffer(const BufferDesc& desc, std::unique_ptr<Buffer>& outBuffer);
    Result createTexture(const TextureDesc& desc, std::unique_ptr<Texture>& outTexture);
    Result createTextureView(Texture& texture, const TextureViewDesc& desc, std::unique_ptr<TextureView>& outTextureView);

private:
    explicit Device(std::unique_ptr<detail::DeviceImpl> impl);

    std::unique_ptr<detail::DeviceImpl> impl_;

    friend Result createDevice(const DeviceDesc& desc, std::unique_ptr<Device>& outDevice);
};

Result createDevice(const DeviceDesc& desc, std::unique_ptr<Device>& outDevice);
int runRhiSmokeTest(bool enableValidation);

} // namespace metallic::render
