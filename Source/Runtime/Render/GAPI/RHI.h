#pragma once

#include "RHIHandle.h"

#include <functional>
#include <array>
#include <any>

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>
#include <span>

namespace metallic::render {

enum class Error : int8_t {
    Failure = 1,
    InvalidArgument,
    OutOfMemory,
    Unsupported,
    OutOfDate,
    DeviceLost,
};

// Operations return their value on success and Error on failure. Use Result<>
// for commands with no result value; resource factories return owning pointers.
template<typename T = void>
using Result = std::expected<T, Error>;

[[nodiscard]] inline std::unexpected<Error> makeError(Error error)
{
    return std::unexpected(error);
}

template<typename T>
[[nodiscard]] inline bool hasError(const Result<T>& result, Error error)
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

template<typename T>
[[nodiscard]] inline const char* resultToString(const Result<T>& result)
{
    return result.has_value() ? "Success" : errorToString(result.error());
}

enum class WindowSystem : uint8_t {
    SDL3,
};

struct WindowHandle {
    WindowSystem system = WindowSystem::SDL3;
    void* nativeWindow = nullptr;
};

enum class Format : uint16_t {
    Unknown,
    R8Unorm,
    R8Snorm,
    R8Uint,
    R8Sint,
    RG8Unorm,
    RG8Snorm,
    RG8Uint,
    RG8Sint,
    BGRA8Unorm,
    BGRA8sRGB,
    RGBA8Unorm,
    RGBA8Snorm,
    RGBA8sRGB,
    RGBA8Uint,
    RGBA8Sint,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16Sfloat,
    RG16Unorm,
    RG16Snorm,
    RG16Uint,
    RG16Sint,
    RG16Sfloat,
    RGBA16Unorm,
    RGBA16Snorm,
    RGBA16Uint,
    RGBA16Sint,
    RGBA16Sfloat,
    R32Uint,
    R32Sint,
    R32Sfloat,
    RG32Uint,
    RG32Sint,
    RG32Sfloat,
    RGB32Uint,
    RGB32Sint,
    RGB32Sfloat,
    RGBA32Uint,
    RGBA32Sint,
    RGBA32Sfloat,
    A2B10G10R10UnormPack32,
    A2R10G10B10UintPack32,
    B10G11R11UfloatPack32,
    E5B9G9R9UfloatPack32,
    D32Sfloat,
    BGRA4Unorm,
    BC4Unorm,
    BC5Unorm,
    BC7Unorm,
    BC7sRGB,
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
    DepthStencil = 1ull << 9,
    PreRasterization = 1ull << 10,
    AccelerationStructureBuild = 1ull << 11,
    RayTracingShader = 1ull << 12,
    MemoryDecompression = 1ull << 13,
    Host = 1ull << 14,
    // Matrix layout/type conversion; use MemoryRead/MemoryWrite access scopes.
    CooperativeVectorConversion = 1ull << 15,
};

// Access semantics are independent of image layout policy.
enum class AccessBits : uint64_t {
    None = 0,
    ShaderRead = 1ull << 0, ShaderWrite = 1ull << 1, UniformRead = 1ull << 2,
    IndirectRead = 1ull << 3, TransferRead = 1ull << 4, TransferWrite = 1ull << 5,
    ColorRead = 1ull << 6, ColorWrite = 1ull << 7,
    DepthStencilRead = 1ull << 8, DepthStencilWrite = 1ull << 9,
    AccelerationStructureRead = 1ull << 10, AccelerationStructureWrite = 1ull << 11,
    DecompressionRead = 1ull << 12, DecompressionWrite = 1ull << 13,
    DescriptorRead = 1ull << 14, HostRead = 1ull << 15, HostWrite = 1ull << 16,
    MemoryRead = 1ull << 17, MemoryWrite = 1ull << 18,
};
constexpr AccessBits operator|(AccessBits lhs, AccessBits rhs)
{
    return static_cast<AccessBits>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}
// Exact execution/access scope. None means empty, never inferred from a layout.
// Nonempty stages with AccessBits::None express an execution-only dependency.
struct SyncScope {
    PipelineStageBits stages = PipelineStageBits::None;
    AccessBits access = AccessBits::None;
};
struct MemoryBarrierDesc {
    SyncScope before;
    SyncScope after;
};
struct SynchronizationStats {
    uint64_t calls = 0;
    uint64_t memoryBarriers = 0;
    uint64_t imageTransitions = 0;
    uint64_t coalescedResources = 0;
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
    Indirect = 1u << 9,
    MemoryDecompression = 1u << 10,
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

enum class QueueAccessBits : uint8_t {
    None = 0,
    Graphics = 1u << 0,
    Compute = 1u << 1,
    Copy = 1u << 2,
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

enum class CullMode : uint8_t {
    None,
    Front,
    Back,
};

enum class FrontFace : uint8_t {
    CounterClockwise,
    Clockwise,
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

constexpr QueueAccessBits operator|(QueueAccessBits lhs, QueueAccessBits rhs)
{
    return static_cast<QueueAccessBits>(
        static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr bool hasFlag(BufferUsageBits value, BufferUsageBits flag)
{
    return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

constexpr bool hasFlag(TextureUsageBits value, TextureUsageBits flag)
{
    return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

constexpr bool hasFlag(QueueAccessBits value, QueueAccessBits flag)
{
    return (static_cast<uint8_t>(value) & static_cast<uint8_t>(flag)) != 0;
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

struct TimestampQueryPoolDesc {
    uint32_t queryCount = 0;
};

struct TimestampQueryResult {
    uint64_t value = 0;
    bool available = false;
};

// A simultaneous device/host clock sample. Host time is QPC on Windows and
// CLOCK_MONOTONIC_RAW on Linux, converted to nanoseconds (not wall-clock time).
struct GPUClockCalibration {
    uint64_t gpuTimestamp = 0;
    uint64_t cpuNanoseconds = 0;
    uint64_t maxDeviationNanoseconds = 0;
};

struct RayTracingAccelerationStructureCompactionQueryPoolDesc {
    uint32_t queryCount = 0;
};

// Backend-independent diagnostic values; numeric encodings are not native API bits.
enum class ValidationSeverity : uint8_t {
    Unknown, Verbose, Info, Warning, Error,
};

enum class ValidationCategory : uint8_t {
    None = 0,
    General = 1 << 0,
    Validation = 1 << 1,
    Performance = 1 << 2,
    ResourceBinding = 1 << 3,
    Unknown = 1 << 4,
};

constexpr ValidationCategory operator|(ValidationCategory lhs, ValidationCategory rhs)
{
    return static_cast<ValidationCategory>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr bool hasFlag(ValidationCategory value, ValidationCategory flags)
{
    return (static_cast<uint8_t>(value) & static_cast<uint8_t>(flags)) != 0;
}

enum class ValidationObjectType : uint8_t {
    Unknown, Instance, Adapter, Device, Queue, CommandBuffer, Semaphore, Fence,
    DeviceMemory, Buffer, BufferView, Texture, TextureView, Sampler, ShaderModule,
    Pipeline, PipelineCache, PipelineLayout, DescriptorHeap, QueryPool, CommandPool,
    Surface, Swapchain, AccelerationStructure, Micromap,
};

struct ValidationObject {
    uint64_t handle = 0;
    ValidationObjectType type = ValidationObjectType::Unknown;
    const char* name = nullptr;
};

struct ValidationMessage {
    ValidationSeverity severity = ValidationSeverity::Unknown;
    ValidationCategory type = ValidationCategory::None;
    int32_t messageId = 0;
    const char* messageIdName = nullptr;
    const char* message = nullptr;
    std::span<const ValidationObject> objects;
};

struct ValidationSink {
    // Called from arbitrary validation threads. Data is borrowed for the call.
    void (*callback)(void*, const ValidationMessage&) noexcept = nullptr;
    void* context = nullptr;
};

enum class MemoryBudgetDomain : uint8_t {
    Other, Geometry, CLAS, CLASScratch, RayTracing, MaterialTextures, FrameResources, Upload, Count
};
struct MemoryBudgetPolicy {
    bool enabled = false;
    uint64_t safetyBytes = 256ull * 1024 * 1024;
    uint64_t graphReserveBytes = 256ull * 1024 * 1024;
    uint64_t externalFeatureReserveBytes = 512ull * 1024 * 1024;
    // Planning allowance for driver/OS overhead of each dedicated material
    // image, in addition to its Vulkan memory requirement. Not a heap guarantee.
    uint64_t materialImageOverheadBytes = 64ull * 1024;
    // Optional per-device-local-heap ceiling, also useful for reproducible pressure tests.
    uint64_t deviceLocalHeapLimitBytes = 0;
};
struct MemoryHeapBudget {
    uint64_t sizeBytes = 0, budgetBytes = 0, usageBytes = 0;
    uint64_t blockBytes = 0, allocationBytes = 0;
    uint32_t blockCount = 0, allocationCount = 0;
    bool deviceLocal = false;
};
struct MemoryDomainBudget {
    uint64_t allocationBytes = 0, peakAllocationBytes = 0;
    uint64_t deviceLocalBytes = 0, allocationCount = 0;
};
struct DeviceMemoryBudget {
    std::vector<MemoryHeapBudget> heaps;
    std::array<MemoryDomainBudget, size_t(MemoryBudgetDomain::Count)> domains{};
    MemoryBudgetPolicy policy;
    uint32_t primaryDeviceLocalHeap = UINT32_MAX;
    bool driverBudget = false;
    uint64_t reservedBytes = 0, deniedAllocations = 0;
    uint64_t availableBytes = 0;
};
namespace detail { struct MemoryBudgetState; }
// A promise for not-yet-created resources. Release before spending it; live
// allocations are already included in heap usage. Safe to outlive the device.
class MemoryBudgetReservation {
public:
    MemoryBudgetReservation() = default;
    ~MemoryBudgetReservation();
    MemoryBudgetReservation(MemoryBudgetReservation&&) noexcept;
    MemoryBudgetReservation& operator=(MemoryBudgetReservation&&) noexcept;
    MemoryBudgetReservation(const MemoryBudgetReservation&) = delete;
    MemoryBudgetReservation& operator=(const MemoryBudgetReservation&) = delete;
    void reset();
    explicit operator bool() const { return bytes_ != 0; }
private:
    std::shared_ptr<detail::MemoryBudgetState> state_;
    uint64_t bytes_ = 0;
    friend class Device;
};

struct DeviceDesc {
    const char* applicationName = "Metallic";
    bool enableValidation = false;
    // Adds synchronization checks and requires validation plus layer settings support.
    bool enableSynchronizationValidation = false;
    bool enableBindlessDescriptorHeap = false;
    // Required for every device, including tests and pipeline-only workloads.
    // createDevice rejects false to prevent cross-feature pipeline-cache reuse.
    bool enableShaderObject = true;
    bool enableMeshShader = false;
    bool enableTaskShader = false;
    bool enableTaskShaderSubgroupBallot = false;
    bool enableGeometryShader = false;
    bool enableSubgroupSizeControl = false;
    bool enableComputeFullSubgroups = false;
    // Soft device-selection preference; pipelines opt in separately.
    uint32_t preferredTaskSubgroupSize = 0;
    bool enableRayTracingAccelerationStructure = false;
    bool enableRayQuery = false;
    // Optional optimization, enabled only when acceleration structures and the
    // device's position-fetch feature are available. False forces the fallback.
    bool enableRayTracingPositionFetch = true;
    // Optional KHR OMM optimization. Unsupported devices keep shader alpha tests.
    bool enableOpacityMicromap = true;
    bool enableClusterAccelerationStructure = false;
    bool enablePartitionedAccelerationStructure = false;
    ValidationSink validationSink;
    // Optional separate compute queue; legacy callers keep their universal queue.
    bool enableAsyncCompute = false;
    // Optional: unsupported devices keep ordinary command recording.
    bool enableDeviceGeneratedCommands = true;
    MemoryBudgetPolicy memoryBudget;
    // Optional backend-owned configuration value. Copies own independent options;
    // each backend validates the payload type before initialization. Empty uses defaults.
    std::any backendExtensions;
};

struct DeviceCapabilities {
    bool memoryDecompression = false;
    bool deviceGeneratedCommands = false;
    bool dynamicGeneratedPipelineLayout = false;
    bool independentCopyQueue = false;
    bool independentComputeQueue = false;
    bool bindlessDescriptorHeap = false;
    bool shaderObject = false;
    bool meshShader = false;
    bool taskShader = false;
    bool geometryShader = false;
    bool subgroupSizeControl = false;
    bool computeFullSubgroups = false;
    bool computeSubgroupBallotArithmetic = false;
    bool computeSubgroupShuffle = false;
    bool taskShaderSubgroupBallot = false;
    bool taskShaderSubgroupSizeControl = false;
    uint32_t subgroupSize = 0;
    uint32_t minSubgroupSize = 0;
    uint32_t maxSubgroupSize = 0;
    uint32_t maxComputeWorkgroupSubgroups = 0;
    bool rayTracingAccelerationStructure = false;
    bool rayQuery = false;
    bool rayTracingPositionFetch = false;
    bool opacityMicromap = false;
    bool clusterAccelerationStructure = false;
    bool partitionedAccelerationStructure = false;
    bool shaderBufferInt64Atomics = false;
    uint32_t subPixelPrecisionBits = 0;
    bool shaderIntegerDotProduct = false;
    bool shaderImageGatherExtended = false;
    bool cooperativeVector = false;
    bool timestampQueries = false;
    double timestampPeriodNanoseconds = 0.0;
    uint64_t bufferCopyOffsetAlignment = 1;
    uint64_t textureUploadBufferOffsetAlignment = 1;
    uint64_t textureUploadRowPitchAlignment = 1;
    uint64_t textureUploadSlicePitchAlignment = 1;
    uint64_t constantBufferOffsetAlignment = 1;
    uint32_t maxBindlessSamplers = 0;
    uint32_t maxBindlessSampledImages = 0;
    uint32_t maxBindlessBuffers = 0;
};

struct BufferDesc {
    uint64_t size = 0;
    uint32_t structureStride = 0;
    BufferUsageBits usage = BufferUsageBits::None;
    MemoryLocation memoryLocation = MemoryLocation::Device;
    QueueAccessBits queueAccess = QueueAccessBits::Graphics;
    MemoryBudgetDomain memoryDomain = MemoryBudgetDomain::Other;
};

// Native requirements for the descriptor used by createAliasedBuffers().
struct BufferAllocationRequirements {
    uint64_t sizeBytes = 0;
    uint64_t alignmentBytes = 0;
    uint32_t memoryTypeBits = 0;
    bool requiresDedicatedAllocation = false;
};

enum class BufferViewType : uint8_t {
    Constant,
    Structured,
    Raw,
    ReadWriteStructured,
    ReadWriteRaw,
};

// Byte range relative to a buffer or slice. UINT64_MAX selects the remainder.
struct BufferRange {
    uint64_t offset = 0;
    uint64_t size = UINT64_MAX;
    bool operator==(const BufferRange&) const = default;

    [[nodiscard]] Result<BufferRange> resolve(uint64_t totalSize) const
    {
        if (offset > totalSize || (size != UINT64_MAX && size > totalSize - offset)) {
            return makeError(Error::InvalidArgument);
        }
        return BufferRange{offset, size == UINT64_MAX ? totalSize - offset : size};
    }
};

namespace detail {
struct BufferImpl;
struct BufferAddressCommandAccess;
}
struct ResourceMemoryInfo;

// CPU range with allocation provenance. It owns the native allocation, not the
// movable Buffer wrapper; no constructor accepts an arbitrary GPU address.
// Device must outlive all slices and GPU work. Ownership does not imply synchronization.
class BufferSlice {
public:
    bool valid() const { return allocation_ != nullptr; }
    uint64_t offset() const { return offset_; }
    uint64_t size() const { return size_; }
    const BufferDesc& allocationDesc() const;
    // Describes the complete backing allocation; offset()/size() describe this slice.
    ResourceMemoryInfo memoryInfo() const;
    const void* allocationIdentity() const { return allocation_.get(); }
    const void* deviceIdentity() const;
    uint64_t deviceAddress() const;
    std::shared_ptr<void> retainAllocation() const;
    // UINT64_MAX takes the remainder; failure returns an error. Empty CPU slices are valid.
    // Preserve allocation ownership when composing ranges (also used by AS scratch alignment).
    [[nodiscard]] Result<BufferSlice> subslice(BufferRange range = {}) const;
    // Requires a nonempty addressed range, all usage bits, and absolute alignment.
    Result<> validate(const void* device, BufferUsageBits usage, uint64_t alignment = 1,
        uint64_t minimumSize = 1) const;
private:
    std::shared_ptr<detail::BufferImpl> allocation_;
    uint64_t offset_ = 0;
    uint64_t size_ = 0;
    friend class Buffer;
    friend class BufferView;
    friend class BindlessHeap;
    friend struct detail::BufferAddressCommandAccess;
};

struct TextureSubresourceRange {
    uint32_t baseMip = 0;
    uint32_t mipCount = 1;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;

    [[nodiscard]] bool valid(uint32_t totalMips, uint32_t totalLayers) const
    {
        return mipCount != 0 && layerCount != 0 && baseMip < totalMips && baseLayer < totalLayers &&
            mipCount <= totalMips - baseMip && layerCount <= totalLayers - baseLayer;
    }
};

struct BufferViewDesc {
    BufferViewType type = BufferViewType::Raw;
    BufferRange range;
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
    QueueAccessBits queueAccess = QueueAccessBits::Graphics;
    MemoryBudgetDomain memoryDomain = MemoryBudgetDomain::Other;
};

// Native requirements for the same descriptor used by createAliasedTextures().
// They describe an alias-capable image, not the ordinary dedicated image path.
struct TextureAllocationRequirements {
    uint64_t sizeBytes = 0;
    uint64_t alignmentBytes = 0;
    uint32_t memoryTypeBits = 0;
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation = false;
};

struct TextureViewDesc {
    Format format = Format::Unknown;
    TextureSubresourceRange range;
    // Values mirror the portable component selection, independent of Vulkan.
    enum class Component : uint8_t { Identity, Zero, One, R, G, B, A };
    std::array<Component, 4> swizzle{};
};

enum class DisplayOutputMode : uint8_t {
    SDR_sRGB,
    HDR_scRGB,
    HDR10_PQ,
    // Source compatibility for existing integrations.
    SDR = SDR_sRGB,
    HDRscRGB = HDR_scRGB,
};

constexpr bool isHDROutput(DisplayOutputMode mode)
{
    return mode == DisplayOutputMode::HDR_scRGB || mode == DisplayOutputMode::HDR10_PQ;
}

constexpr const char* displayOutputName(DisplayOutputMode mode)
{
    switch (mode) {
    case DisplayOutputMode::SDR_sRGB: return "SDR_sRGB";
    case DisplayOutputMode::HDR_scRGB: return "HDR_scRGB";
    case DisplayOutputMode::HDR10_PQ: return "HDR10_PQ";
    default: return "Unknown";
    }
}

struct SwapchainDesc {
    WindowHandle window;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t imageCount = 3;
    uint32_t framesInFlight = 2;
    Format format = Format::BGRA8sRGB;
    bool vsync = true;
    DisplayOutputMode outputMode = DisplayOutputMode::SDR;
    bool allowSdrFallback = true;
    float peakNits = 1000.0f;
};

enum class TextureLayout : uint8_t {
    Undefined,
    Present,
    ColorAttachment,
    DepthStencilAttachment,
    ShaderRead,
    TransferSource,
    TransferDestination,
    General,
};

struct TextureBarrierDesc {
    class Texture* texture = nullptr;
    TextureLayout oldLayout = TextureLayout::Undefined;
    TextureLayout newLayout = TextureLayout::Undefined;
    SyncScope before;
    SyncScope after;
    TextureSubresourceRange range;
};

struct BufferBarrierDesc {
    class Buffer* buffer = nullptr;
    SyncScope before;
    SyncScope after;
    BufferRange range;
};

struct AccelerationStructureBarrierDesc {
    class RayTracingAccelerationStructure* accelerationStructure = nullptr;
    SyncScope before;
    SyncScope after;
};

struct ClusterAccelerationStructureProperties {
    uint64_t clusterStorageAlignment = 0;
    uint64_t bottomLevelStorageAlignment = 0;
    uint64_t scratchAlignment = 0;
    uint64_t triangleBuildInfoSize = 0;
    uint64_t bottomLevelBuildInfoSize = 0;
};

struct ClusterAccelerationStructureBuildSizes {
    uint64_t accelerationStructureSize = 0;
    uint64_t updateScratchSize = 0;
    uint64_t buildScratchSize = 0;
};

struct ClusterAccelerationStructureTriangleBuildSizesDesc {
    uint32_t maxClusterTriangleCount = 0;
    uint32_t maxClusterVertexCount = 0;
    uint32_t maxClusterUniqueGeometryCount = 1;
    uint32_t maxGeometryIndexValue = 0;
    uint32_t minPositionTruncateBitCount = 0;
    uint32_t maxTotalTriangleCount = 0;
    uint32_t maxTotalVertexCount = 0;
    Format vertexFormat = Format::RGB32Sfloat;
    uint32_t maxAccelerationStructureCount = 1;
};

enum class ClusterAccelerationStructureIndexFormat : uint8_t {
    Uint8,
    Uint16,
    Uint32,
};

struct ClusterAccelerationStructureTriangleBuildInfo {
    uint32_t clusterId = 0;
    uint32_t triangleCount = 0;
    uint32_t vertexCount = 0;
    uint32_t positionTruncateBitCount = 0;
    uint32_t geometryIndex = 0;
    ClusterAccelerationStructureIndexFormat indexFormat =
        ClusterAccelerationStructureIndexFormat::Uint8;
    uint16_t indexBufferStride = 1;
    uint16_t vertexBufferStride = 0;
    BufferSlice indexBuffer;
    BufferSlice vertexBuffer;
    BufferSlice destinationBuffer;
    bool opaque = true;
};

struct ClusterAccelerationStructureTriangleBuildDesc {
    std::span<const ClusterAccelerationStructureTriangleBuildInfo> clusters;
    uint32_t maxClusterTriangleCount = 0;
    uint32_t maxClusterVertexCount = 0;
    uint32_t maxClusterUniqueGeometryCount = 1;
    uint32_t maxGeometryIndexValue = 0;
    uint32_t minPositionTruncateBitCount = 0;
    Format vertexFormat = Format::RGB32Sfloat;
    BufferSlice scratchBuffer;
    BufferSlice buildInfoBuffer;
    BufferSlice destinationAddressBuffer;
    // Optional GPU output: one uint32_t actual encoded size per cluster.
    BufferSlice destinationSizeBuffer;
};

// The source slice size is the encoded object size; destination must cover it.
struct ClusterAccelerationStructureMoveInfo {
    BufferSlice sourceBuffer;
    BufferSlice destinationBuffer;
};

// Non-overlapping copies of triangle CLAS, with driver relocation of their contents.
struct ClusterAccelerationStructureMoveDesc {
    std::span<const ClusterAccelerationStructureMoveInfo> objects;
    BufferSlice sourceAddressBuffer;
    BufferSlice destinationAddressBuffer;
    BufferSlice scratchBuffer;
};

enum class RayTracingAccelerationStructureType : uint8_t {
    BottomLevel,
    TopLevel,
    OpacityMicromap,
};

enum class RayTracingAccelerationStructureBuildMode : uint8_t {
    Build,
    Update,
};

enum class RayTracingAccelerationStructureBuildFlags : uint8_t {
    None = 0,
    PreferFastTrace = 1u << 0,
    PreferFastBuild = 1u << 1,
    AllowUpdate = 1u << 2,
    AllowCompaction = 1u << 3,
    AllowDataAccess = 1u << 4,
};

constexpr RayTracingAccelerationStructureBuildFlags operator|(
    RayTracingAccelerationStructureBuildFlags lhs,
    RayTracingAccelerationStructureBuildFlags rhs)
{
    return static_cast<RayTracingAccelerationStructureBuildFlags>(
        static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr bool hasFlag(
    RayTracingAccelerationStructureBuildFlags value,
    RayTracingAccelerationStructureBuildFlags flag)
{
    return (static_cast<uint8_t>(value) & static_cast<uint8_t>(flag)) != 0;
}

enum class RayTracingGeometryFlags : uint8_t {
    None = 0,
    Opaque = 1u << 0,
    NoDuplicateAnyHitInvocation = 1u << 1,
};

constexpr RayTracingGeometryFlags operator|(
    RayTracingGeometryFlags lhs,
    RayTracingGeometryFlags rhs)
{
    return static_cast<RayTracingGeometryFlags>(
        static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr bool hasFlag(RayTracingGeometryFlags value, RayTracingGeometryFlags flag)
{
    return (static_cast<uint8_t>(value) & static_cast<uint8_t>(flag)) != 0;
}

enum class RayTracingInstanceFlags : uint8_t {
    None = 0,
    TriangleFacingCullDisable = 1u << 0,
    TriangleFrontCounterClockwise = 1u << 1,
    ForceOpaque = 1u << 2,
    ForceNonOpaque = 1u << 3,
};

constexpr RayTracingInstanceFlags operator|(
    RayTracingInstanceFlags lhs,
    RayTracingInstanceFlags rhs)
{
    return static_cast<RayTracingInstanceFlags>(
        static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr bool hasFlag(RayTracingInstanceFlags value, RayTracingInstanceFlags flag)
{
    return (static_cast<uint8_t>(value) & static_cast<uint8_t>(flag)) != 0;
}

enum class RayTracingIndexType : uint8_t {
    None,
    Uint16,
    Uint32,
};

struct RayTracingAccelerationStructureProperties {
    uint64_t scratchAlignment = 1;
    uint64_t instanceBufferAlignment = 16;
    uint64_t instanceRecordSize = 0;
    uint32_t maxOpacity2StateSubdivisionLevel = 0;
    uint32_t maxOpacity4StateSubdivisionLevel = 0;
    uint64_t maxMicromapTriangles = 0;
};

enum class OpacityMicromapFormat : uint16_t {
    TwoState = 1,
    FourState = 2,
};

// Packed device input; one record per original triangle, in BLAS triangle order.
struct OpacityMicromapTriangle {
    uint32_t dataOffset = 0;
    uint16_t subdivisionLevel = 0;
    OpacityMicromapFormat format = OpacityMicromapFormat::FourState;
};
static_assert(sizeof(OpacityMicromapTriangle) == 8);

struct OpacityMicromapUsage {
    uint32_t count = 0;
    uint32_t subdivisionLevel = 0;
    OpacityMicromapFormat format = OpacityMicromapFormat::FourState;
};

struct OpacityMicromapBuildInput {
    std::span<const OpacityMicromapUsage> usages;
    BufferSlice dataBuffer;
    BufferSlice triangleBuffer;
    uint64_t triangleStride = sizeof(OpacityMicromapTriangle);
};

struct RayTracingTriangleGeometryDesc {
    BufferSlice vertexBuffer;
    uint64_t vertexStride = 0;
    Format vertexFormat = Format::RGB32Sfloat;
    uint32_t vertexCount = 0;
    BufferSlice indexBuffer;
    RayTracingIndexType indexType = RayTracingIndexType::Uint32;
    uint32_t primitiveCount = 0;
    RayTracingGeometryFlags flags = RayTracingGeometryFlags::Opaque;
    // One micromap triangle per geometry triangle. Keep alive with the BLAS.
    class RayTracingAccelerationStructure* opacityMicromap = nullptr;
    // Histogram for these geometry triangles; required by the EXT OMM backend.
    // Only read during size queries and command recording.
    std::span<const OpacityMicromapUsage> opacityMicromapUsages;
};

struct RayTracingAccelerationStructureBuildInputs {
    RayTracingAccelerationStructureType type =
        RayTracingAccelerationStructureType::BottomLevel;
    RayTracingAccelerationStructureBuildFlags flags =
        RayTracingAccelerationStructureBuildFlags::PreferFastTrace;
    std::span<const RayTracingTriangleGeometryDesc> geometries;
    uint32_t instanceCount = 0;
    const OpacityMicromapBuildInput* micromap = nullptr;
};

struct RayTracingAccelerationStructureBuildSizes {
    uint64_t accelerationStructureSize = 0;
    uint64_t buildScratchSize = 0;
    uint64_t updateScratchSize = 0;
};

// Both backends expose a traceable TopLevel resource and the same shader address ABI.
enum class RayTracingTopLevelBackend : uint8_t {
    Standard,
    Partitioned,
};

struct RayTracingAccelerationStructureDesc {
    RayTracingAccelerationStructureType type =
        RayTracingAccelerationStructureType::BottomLevel;
    RayTracingAccelerationStructureBuildFlags buildFlags =
        RayTracingAccelerationStructureBuildFlags::PreferFastTrace;
    uint64_t size = 0;
    // Partitioned resources use the PartitionedAccelerationStructureDesc overload
    // of createRayTracingAccelerationStructure; this creation descriptor is Standard-only.
    // BottomLevel and OpacityMicromap resources always use Standard.
    RayTracingTopLevelBackend topLevelBackend = RayTracingTopLevelBackend::Standard;
};

struct RayTracingInstanceDesc {
    float transform[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
    class RayTracingAccelerationStructure* bottomLevel = nullptr;
    uint32_t customIndex = 0;
    uint32_t shaderBindingTableRecordOffset = 0;
    uint8_t mask = 0xff;
    RayTracingInstanceFlags flags = RayTracingInstanceFlags::TriangleFacingCullDisable;
};

// GPU-writable instance record consumed by top-level acceleration-structure
// builds. The packed fields use 24 low bits for the index/record offset and
// 8 high bits for the mask/flags respectively.
struct RayTracingGPUInstance {
    float transform[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
    uint32_t customIndexAndMask = 0;
    uint32_t shaderBindingTableRecordOffsetAndFlags = 0;
    uint64_t accelerationStructureReference = 0;
};

static_assert(sizeof(RayTracingGPUInstance) == 64);

// Build commands retain all supplied slice allocations through submission completion.
// Scratch alignment is applied within the supplied slice, consuming its leading padding.
struct RayTracingAccelerationStructureBuildDesc {
    class RayTracingAccelerationStructure* destination = nullptr;
    class RayTracingAccelerationStructure* source = nullptr;
    RayTracingAccelerationStructureBuildMode mode =
        RayTracingAccelerationStructureBuildMode::Build;
    std::span<const RayTracingTriangleGeometryDesc> geometries;
    BufferSlice instanceBuffer;
    uint32_t instanceCount = 0;
    BufferSlice scratchBuffer;
    const OpacityMicromapBuildInput* micromap = nullptr;
    // RenderGraph declares the AS write and synchronizes subsequent consumers.
    // Standalone builds retain the legacy post-build dependency by default.
    bool graphManagedSynchronization = false;
};

struct ClusterAccelerationStructureBottomLevelBuildSizesDesc {
    RayTracingAccelerationStructureBuildFlags flags =
        RayTracingAccelerationStructureBuildFlags::PreferFastTrace;
    uint32_t maxClusterCountPerAccelerationStructure = 0;
    uint32_t maxTotalClusterCount = 0;
    uint32_t maxAccelerationStructureCount = 1;
};

// GPU-writable input record for one cluster-based bottom-level acceleration
// structure. clusterReferencesAddress points to an array of CLAS addresses.
struct ClusterAccelerationStructureBottomLevelBuildInfo {
    uint32_t clusterReferencesCount = 0;
    uint32_t clusterReferencesStride = sizeof(uint64_t);
    uint64_t clusterReferencesAddress = 0;
};

static_assert(sizeof(ClusterAccelerationStructureBottomLevelBuildInfo) == 16);

enum class ClusterAccelerationStructureDestinationMode : uint8_t {
    Implicit,
    Explicit,
};

struct ClusterAccelerationStructureBottomLevelBuildDesc {
    RayTracingAccelerationStructureBuildFlags flags =
        RayTracingAccelerationStructureBuildFlags::PreferFastTrace;
    ClusterAccelerationStructureDestinationMode destinationMode =
        ClusterAccelerationStructureDestinationMode::Implicit;
    uint32_t maxClusterCountPerAccelerationStructure = 0;
    uint32_t maxTotalClusterCount = 0;
    uint32_t maxAccelerationStructureCount = 1;
    BufferSlice buildInfoBuffer;
    uint64_t buildInfoStride = sizeof(ClusterAccelerationStructureBottomLevelBuildInfo);
    BufferSlice buildInfoCountBuffer;
    BufferSlice destinationStorageBuffer;
    BufferSlice destinationAddressBuffer;
    uint64_t destinationAddressStride = sizeof(uint64_t);
    BufferSlice destinationSizeBuffer;
    uint64_t destinationSizeStride = sizeof(uint32_t);
    BufferSlice scratchBuffer;
};

struct PartitionedAccelerationStructureBuildInputs {
    RayTracingAccelerationStructureBuildFlags flags =
        RayTracingAccelerationStructureBuildFlags::PreferFastTrace;
    uint32_t instanceCount = 0;
    uint32_t partitionCount = 1;
    uint32_t maxInstancePerPartitionCount = 0;
    uint32_t maxInstanceInGlobalPartitionCount = 0;
    uint32_t maxOperationCount = 1;
    bool allowInstanceUpdate = false;
    bool allowPartitionTranslation = false;
};

struct PartitionedAccelerationStructureBuildSizes {
    uint64_t accelerationStructureSize = 0;
    uint64_t updateScratchSize = 0;
    uint64_t buildScratchSize = 0;
    uint64_t operationInfoSize = 0;
    uint64_t operationCountSize = 0;
    uint64_t instanceWriteInfoSize = 0;
    uint64_t instanceUpdateInfoSize = 0;
    uint64_t partitionWriteInfoSize = 0;
};

struct PartitionedAccelerationStructureDesc {
    // Selects the Partitioned TopLevel backend at creation. Keep its capacity
    // and operation-storage requirements separate from the Standard descriptor.
    PartitionedAccelerationStructureBuildInputs inputs;
    PartitionedAccelerationStructureBuildSizes sizes;
};

struct PartitionedAccelerationStructureInstanceDesc {
    float transform[3][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
    class RayTracingAccelerationStructure* bottomLevel = nullptr;
    uint32_t instanceIndex = 0;
    uint32_t partitionIndex = 0;
    uint32_t customIndex = 0;
    uint32_t shaderBindingTableRecordOffset = 0;
    uint8_t mask = 0xff;
    RayTracingInstanceFlags flags = RayTracingInstanceFlags::TriangleFacingCullDisable;
};

struct PartitionedAccelerationStructureBuildDesc {
    class RayTracingAccelerationStructure* destination = nullptr;
    BufferSlice instanceBuffer;
    uint32_t instanceCount = 0;
    BufferSlice scratchBuffer;
    bool graphManagedSynchronization = false;
};

struct BarrierDesc {
    // Resources remain explicit for layout transitions and graph dependency tracking.
    std::span<const TextureBarrierDesc> textures;
    std::span<const BufferBarrierDesc> buffers;
    std::span<const MemoryBarrierDesc> memory;
    std::span<const AccelerationStructureBarrierDesc> accelerationStructures;
};

struct SemaphoreDesc {
    uint64_t initialValue = 0;
};

struct RenderingAttachmentDesc {
    class TextureView* view = nullptr;
    TextureLayout layout = TextureLayout::ColorAttachment;
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    ColorValue clearColor;
    float clearDepth = 1.0f;
    uint32_t clearStencil = 0;
};

struct RenderingDesc {
    Rect renderArea;
    std::span<const RenderingAttachmentDesc> colorAttachments;
    const RenderingAttachmentDesc* depthStencilAttachment = nullptr;
};

struct SemaphoreSubmitDesc {
    class Semaphore* semaphore = nullptr;
    uint64_t value = 0;
    PipelineStageBits stages = PipelineStageBits::AllCommands;
};

struct SwapchainSemaphoreSubmitDesc {
    class SwapchainSemaphore* semaphore = nullptr;
    PipelineStageBits stages = PipelineStageBits::AllCommands;
};

struct QueueSubmitDesc {
    std::span<const SemaphoreSubmitDesc> waitSemaphores;
    std::span<const SwapchainSemaphoreSubmitDesc> waitSwapchainSemaphores;
    std::span<class CommandBuffer* const> commandBuffers;
    std::span<const SemaphoreSubmitDesc> signalSemaphores;
    std::span<const SwapchainSemaphoreSubmitDesc> signalSwapchainSemaphores;
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

struct RasterizationState {
    CullMode cullMode = CullMode::None;
    FrontFace frontFace = FrontFace::CounterClockwise;
};

// SPIR-V is supplied in words and copied during creation. The input may then be released.
struct ShaderModuleDesc {
    std::span<const uint32_t> spirv;
    const char* debugName = nullptr;
};

enum class PipelineCacheLoadStatus : uint8_t {
    NotFound,
    Loaded,
    Invalid,
    Incompatible,
};

struct PipelineCacheDesc {
    // Persistent caches use the backend-neutral .pso container. A null or empty
    // path creates an in-memory cache.
    const char* filePath = nullptr;
    bool saveOnDestroy = true;
};

struct PipelineCacheStats {
    PipelineCacheLoadStatus loadStatus = PipelineCacheLoadStatus::NotFound;
    uint64_t storedPsoCount = 0;
    uint64_t sessionPsoCount = 0;
    uint64_t hitCount = 0;
    uint64_t missCount = 0;
    uint64_t backendDataSize = 0;
    // Revisions advance only for successful, previously unseen PSO identities.
    // A save commits its snapshot; concurrent newer revisions remain pending.
    uint64_t dirtyRevision = 0;
    uint64_t persistedRevision = 0;
    uint64_t saveCount = 0;
    uint64_t saveFailureCount = 0;
    uint64_t lastExtractTimeNanoseconds = 0;
    uint64_t lastWriteTimeNanoseconds = 0;
    uint64_t lastSaveTimeNanoseconds = 0;
    bool saveInProgress = false;
};

// Stages borrow their module only for creation; executables own backend state.
struct ShaderStageDesc {
    class ShaderModule* module = nullptr;
    const char* entryPoint = "main";
};

struct GraphicsPipelineDesc {
    ShaderStageDesc vertexShader;
    ShaderStageDesc taskShader;
    ShaderStageDesc meshShader;
    ShaderStageDesc fragmentShader;
    uint32_t taskRequiredSubgroupSize = 0;
    // Full-subgroup mode is accepted only with a fixed required subgroup size.
    bool taskRequireFullSubgroups = false;
    static constexpr uint32_t kMaxColorAttachments = 8;
    // Only the first colorAttachmentCount entries participate in creation and hashing.
    std::array<Format, kMaxColorAttachments> colorFormats{};
    uint32_t colorAttachmentCount = 0;
    Format depthStencilFormat = Format::Unknown;
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    RasterizationState rasterization;
    DepthStencilState depthStencil;
    bool usesBindlessHeap = false;
    class PipelineCache* pipelineCache = nullptr;
    // Required for membership in a DGC indirect execution set.
    bool indirectBindable = false;
};

struct ComputePipelineDesc {
    ShaderStageDesc computeShader;
    bool usesBindlessHeap = false;
    uint32_t bindlessUserPushDataSize = 0;
    class PipelineCache* pipelineCache = nullptr;
    // Required for membership in a DGC indirect execution set.
    bool indirectBindable = false;
};

struct GraphicsShaderObjectProgramDesc {
    ShaderStageDesc vertexShader;
    ShaderStageDesc fragmentShader;
    bool usesBindlessHeap = false;
    uint32_t bindlessUserPushDataSize = 0;
    // Required for membership in a DGC indirect execution set.
    bool indirectBindable = false;
    // Optional backend binary persistence. Registry supplies its default
    // directory; direct RHI callers opt in explicitly. Borrowed for creation.
    const char* binaryCacheDirectory = nullptr;
};

struct ShaderObjectCacheStats {
    PipelineCacheLoadStatus loadStatus = PipelineCacheLoadStatus::NotFound;
    uint64_t programHash = 0;
    uint64_t binaryDataSize = 0;
    // Time inside vkCreateShadersEXT only, including any failed binary attempt.
    uint64_t creationTimeNanoseconds = 0;
    // True only after the driver successfully creates both stages from BINARY.
    bool binaryCacheHit = false;
    bool persisted = false;
    bool driverRejected = false;
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
    uint32_t layerCount = 1;
};

// Slice sizes are the compressed input size and exact decoded output size.
struct BufferDecompressionDesc {
    BufferSlice source;
    BufferSlice destination;
};

struct BindlessHeapDesc {
    uint32_t maxSamplers = 0;
    uint32_t maxSampledImages = 0;
    uint32_t maxStorageImages = 0;
    uint32_t maxBuffers = 0;
};

enum class BindlessHandleKind : uint8_t {
    Invalid,
    Sampler,
    SampledImage,
    StorageImage,
    Buffer,
    AccelerationStructure,
};

struct BindlessHandle {
    BindlessHandleKind kind = BindlessHandleKind::Invalid;
    // Allocator-local slot; only use for descriptor writes and release.
    uint32_t index = UINT32_MAX;
    // Final typed descriptor index in the owning heap, ready for shader parameters.
    // This is not a byte offset or an index that can be shared across heaps.
    // AS parameters currently use the separate address resolver ABI.
    uint32_t shaderIndex = UINT32_MAX;

    bool valid() const { return kind != BindlessHandleKind::Invalid && index != UINT32_MAX; }
};

enum class SamplerFilter : uint8_t {
    Nearest,
    Linear,
};

enum class SamplerAddressMode : uint8_t {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
};

struct SamplerDesc {
    SamplerFilter minFilter = SamplerFilter::Linear;
    SamplerFilter magFilter = SamplerFilter::Linear;
    SamplerFilter mipFilter = SamplerFilter::Nearest;
    SamplerAddressMode addressU = SamplerAddressMode::ClampToEdge;
    SamplerAddressMode addressV = SamplerAddressMode::ClampToEdge;
    SamplerAddressMode addressW = SamplerAddressMode::ClampToEdge;
    float minLod = 0.0f;
    float maxLod = 1000.0f;
};

namespace detail {
struct DeviceImpl;
struct QueueImpl;
struct SwapchainImpl;
struct CommandPoolImpl;
struct CommandBufferImpl;
struct CommandSubmissionState;
struct FenceImpl;
struct TimestampQueryPoolImpl;
struct RayTracingAccelerationStructureCompactionQueryPoolImpl;
struct SemaphoreImpl;
struct SwapchainSemaphoreImpl;
struct BufferImpl;
struct BufferAddressCommandAccess;
struct BufferViewImpl;
struct RayTracingAccelerationStructureImpl;
struct TextureImpl;
struct TextureViewImpl;
struct ShaderModuleImpl;
struct PipelineCacheImpl;
struct GraphicsPipelineImpl;
struct ComputePipelineImpl;
struct GraphicsShaderObjectProgramImpl;
struct BindlessHeapImpl;
struct VulkanNativeAccess;
struct ShaderBindingMappingDesc;
} // namespace detail

class Queue {
    METALLIC_RHI_HANDLE(Queue, unique_ptr,
        friend class Device;
        friend class Swapchain;
        friend class CommandPool;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    Result<> submit(const QueueSubmitDesc& desc);
    // For owners that seal recordings and attach GPU completion tracking.
    Result<> submitTracked(const QueueSubmitDesc& desc);
    Result<> waitIdle();
    QueueType type() const;
    bool sameQueue(const Queue& other) const;
    uint32_t timestampValidBits() const;
    // Optional and non-blocking: Unsupported leaves ordinary timestamp timing usable.
    [[nodiscard]] Result<GPUClockCalibration> calibrateTimestamps() const;

private:
    Result<> submitImpl(const QueueSubmitDesc& desc, bool tracked);
};

class Fence {
    METALLIC_RHI_HANDLE(Fence, unique_ptr,
        friend class Device;
        friend class Queue;
        friend struct detail::DeviceImpl;
    )

    Result<> wait(uint64_t timeoutNanoseconds = UINT64_MAX);
    Result<> reset();
    bool isSignaled() const;
};

class Semaphore {
    METALLIC_RHI_HANDLE(Semaphore, unique_ptr,
        friend class Device;
        friend class Queue;
        friend class Swapchain;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    Result<> wait(uint64_t value, uint64_t timeoutNanoseconds = UINT64_MAX);
    // Host-to-GPU timeline synchronization; also used to gate in-flight lifetime tests.
    Result<> signal(uint64_t value);
    uint64_t currentValue() const;
};

class SwapchainSemaphore {
    METALLIC_RHI_HANDLE(SwapchainSemaphore, unique_ptr,
        friend class Device;
        friend class Queue;
        friend class Swapchain;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

};

// Value-only allocation diagnostics: taking a snapshot does not retain GPU memory.
// allocationId identifies one resource allocation generation for this process.
// backingAllocationId identifies the physical allocation owner. Distinct alias
// images have distinct allocationIds and share one backingAllocationId.
// memoryBlockId is an opaque native-memory token, comparable only within one
// device's live snapshot; a freed block's token can later be reused. Equal blocks
// with disjoint ranges are suballocations, not memory aliases. Borrowed images
// have an allocationId but unknown backing (known == false).
struct ResourceMemoryInfo {
    uint64_t allocationId = 0;
    uint64_t backingAllocationId = 0;
    uint64_t backingSizeBytes = 0;
    uint64_t memoryBlockId = 0;
    uint64_t offsetBytes = 0;
    uint64_t sizeBytes = 0;
    uint32_t memoryTypeIndex = UINT32_MAX;
    uint32_t heapIndex = UINT32_MAX;
    bool known = false;
};

struct BufferTextureRegion {
    class Texture* texture = nullptr;
    BufferSlice buffer;
    uint32_t bufferRowPitch = 0;
    uint32_t bufferSlicePitch = 0;
    int32_t textureOffsetX = 0;
    int32_t textureOffsetY = 0;
    int32_t textureOffsetZ = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mipLevel = 0;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;
};

class Buffer {
    METALLIC_RHI_HANDLE(Buffer, shared_ptr,
        friend class Device;
        friend class CommandBuffer;
        friend class BufferView;
        friend class BindlessHeap;
        friend struct detail::DeviceImpl;
        friend struct detail::BufferAddressCommandAccess;
        friend struct detail::VulkanNativeAccess;
    )

    const BufferDesc& desc() const;
    ResourceMemoryInfo memoryInfo() const;
    uint64_t deviceAddress() const;
    [[nodiscard]] Result<BufferSlice> slice(BufferRange range = {}) const;
    // Retains this allocation, not the movable public wrapper. Device must outlive it.
    std::shared_ptr<void> retainAllocation() const;
    const void* deviceIdentity() const;
    // Independent host writes must not share a non-coherent flush atom.
    uint64_t hostWriteAlignment() const;
    void* map();
    void unmap();
    void flush(BufferRange range = {});
    void invalidate(BufferRange range = {});
};

class BufferView {
    METALLIC_RHI_HANDLE(BufferView, unique_ptr,
        friend class Device;
        friend class BindlessHeap;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    const BufferViewDesc& desc() const;
    BufferSlice slice() const;
};

class TimestampQueryPool {
    METALLIC_RHI_HANDLE(TimestampQueryPool, unique_ptr,
        friend class Device;
        friend class CommandBuffer;
    )

    const TimestampQueryPoolDesc& desc() const;
    // All submitted work using this range must be complete. Externally synchronize
    // host resets and result reads for the same range.
    Result<> reset(uint32_t firstQuery, uint32_t queryCount);
    Result<> readResults(
        uint32_t firstQuery,
        std::span<TimestampQueryResult> outResults) const;
    double durationMilliseconds(uint64_t beginTimestamp, uint64_t endTimestamp) const;
};

class RayTracingAccelerationStructureCompactionQueryPool {
    METALLIC_RHI_HANDLE(RayTracingAccelerationStructureCompactionQueryPool, unique_ptr,
        friend class Device;
        friend class CommandBuffer;
    )

    const RayTracingAccelerationStructureCompactionQueryPoolDesc& desc() const;
    Result<> readResults(
        uint32_t firstQuery,
        std::span<uint64_t> outCompactedSizes) const;
};

class RayTracingAccelerationStructure {
    METALLIC_RHI_HANDLE(RayTracingAccelerationStructure, shared_ptr,
        friend class Device;
        friend class CommandBuffer;
        friend class BindlessHeap;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    const RayTracingAccelerationStructureDesc& desc() const;
    ResourceMemoryInfo memoryInfo() const;
    bool valid() const;
    uint64_t deviceAddress() const;
    std::shared_ptr<void> retainAllocation() const;
    const void* deviceIdentity() const;
};

class Texture {
    METALLIC_RHI_HANDLE(Texture, shared_ptr,
        friend class Device;
        friend class Swapchain;
        friend class CommandBuffer;
        friend class TextureView;
        friend class BindlessHeap;
        friend struct detail::DeviceImpl;
        friend struct detail::SwapchainImpl;
        friend struct detail::VulkanNativeAccess;
    )

    const TextureDesc& desc() const;
    uint64_t allocationSize() const;
    ResourceMemoryInfo memoryInfo() const;
    // Owns the image allocation; borrowed swapchain images return empty.
    std::shared_ptr<void> retainAllocation() const;
    const void* deviceIdentity() const;
};

class TextureView {
    METALLIC_RHI_HANDLE(TextureView, shared_ptr,
        friend class Device;
        friend class CommandBuffer;
        friend class BindlessHeap;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    const TextureViewDesc& desc() const;
    // Semantic shader views are cheap. Materialize only for attachments/interop.
    // Owns the image allocation, not this view. Borrowed swapchain images return empty.
    std::shared_ptr<void> retainTexture() const;
    const void* deviceIdentity() const;
};

class ShaderModule {
    METALLIC_RHI_HANDLE(ShaderModule, unique_ptr,
        friend class Device;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    // Length-prefixed device SPIR-V identity used by pipeline caches.
    uint64_t contentHash() const;
    // Raw input SPIR-V FNV-1a, before backend rewriting, for diagnostic file matching.
    uint64_t inputSpirvHash() const;
};

class PipelineCache {
    METALLIC_RHI_HANDLE(PipelineCache, unique_ptr,
        friend class Device;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    PipelineCacheStats stats() const;
    // Persists pending cache changes. This is a no-op when no new PSO hash was
    // recorded since loading or the previous save.
    Result<> save();
};

struct RasterExecutionState {
    RasterizationState rasterization;
    DepthStencilState depthStencil;
    uint32_t colorAttachmentCount = 1;
};
enum class ExecutionKind : uint8_t { Compute, Raster };

// Immutable snapshot of already-created executable code. Copies keep native
// objects alive across reload/retirement; no lookup or PSO creation at bind time.
class PreparedExecution {
public:
    bool valid() const { return compute_ || graphics_ || shaders_; }
    ExecutionKind kind() const { return compute_ ? ExecutionKind::Compute : ExecutionKind::Raster; }
    const void* deviceIdentity() const;
private:
    std::shared_ptr<detail::ComputePipelineImpl> compute_;
    std::shared_ptr<detail::GraphicsPipelineImpl> graphics_;
    std::shared_ptr<detail::GraphicsShaderObjectProgramImpl> shaders_;
    RasterExecutionState raster_;
    friend class ComputePipeline;
    friend class GraphicsPipeline;
    friend class GraphicsShaderObjectProgram;
    friend class CommandBuffer;
};

class GraphicsPipeline {
    METALLIC_RHI_HANDLE(GraphicsPipeline, shared_ptr,
        friend class Device;
        friend class CommandBuffer;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    // pipelineCacheHit reports an exact PSO hash-table hit. The backend may
    // still perform implementation-defined validation when creating the PSO.
    uint64_t psoHash() const;
    bool pipelineCacheHit() const;

    PreparedExecution execution() const;
};

class ComputePipeline {
    METALLIC_RHI_HANDLE(ComputePipeline, shared_ptr,
        friend class Device;
        friend class CommandBuffer;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    // pipelineCacheHit reports an exact PSO hash-table hit. The backend may
    // still perform implementation-defined validation when creating the PSO.
    uint64_t psoHash() const;
    bool pipelineCacheHit() const;

    PreparedExecution execution() const;
};

class GraphicsShaderObjectProgram {
    METALLIC_RHI_HANDLE(GraphicsShaderObjectProgram, shared_ptr,
        friend class Device;
        friend class CommandBuffer;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    ShaderObjectCacheStats cacheStats() const;
    const char* binaryCacheFilePath() const;
    PreparedExecution execution(const RasterExecutionState& state = {}) const;
};

class BindlessHeap {
    METALLIC_RHI_HANDLE(BindlessHeap, unique_ptr,
        friend class Device;
        friend class CommandBuffer;
        friend struct detail::DeviceImpl;
    )

    const BindlessHeapDesc& desc() const;

    [[nodiscard]] Result<BindlessHandle> allocate(BindlessHandleKind kind);
    void release(BindlessHandle handle);
    Result<> writeSampler(BindlessHandle handle, const SamplerDesc& sampler);
    Result<> writeSampledImage(BindlessHandle handle, TextureView& view, TextureLayout layout = TextureLayout::ShaderRead);
    Result<> writeStorageImage(BindlessHandle handle, TextureView& view);
    Result<> writeBufferView(BindlessHandle handle, BufferView& view);
    Result<> writeConstantBuffer(BindlessHandle handle, Buffer& buffer);
    // Writes the complete backing allocation, even when only a slice owner remains.
    Result<> writeStorageBuffer(BindlessHandle handle, const BufferSlice& buffer);
    Result<> writeAccelerationStructure(
        BindlessHandle handle,
        RayTracingAccelerationStructure& accelerationStructure);

private:
    struct BindlessSamplerWrite {
        BindlessHandle handle;
        SamplerDesc sampler;
    };

    struct BindlessImageWrite {
        BindlessHandle handle;
        TextureView* view = nullptr;
        TextureLayout layout = TextureLayout::ShaderRead;
    };
    Result<> writeSamplers(std::span<const BindlessSamplerWrite> writes);
    Result<> writeImages(std::span<const BindlessImageWrite> writes);
};

class SubmissionTransaction;
class CommandSubmissionContext;


class CommandBuffer {
    METALLIC_RHI_HANDLE(CommandBuffer, unique_ptr,
        friend class CommandPool;
        friend class Queue;
        friend struct detail::CommandPoolImpl;
        friend struct detail::VulkanNativeAccess;
    )

    Result<> begin(std::shared_ptr<CommandSubmissionContext> context = {});
    const std::shared_ptr<CommandSubmissionContext>& submissionContext() const { return submissionContext_; }
    const std::shared_ptr<detail::CommandSubmissionState>& submissionState() const { return submission_; }
    bool recording() const { return recording_; }
    QueueAccessBits queueCapabilities() const;
    const void* deviceIdentity() const;
    // Queue::submit merges these waits and the command buffer retains their
    // timeline lifetimes until its next recording. Call while recording.
    Result<> addDependency(std::span<const SemaphoreSubmitDesc> waits, std::shared_ptr<const void> lifetime = {});
    Result<> addSubmissionTransaction(std::shared_ptr<SubmissionTransaction> transaction);
    // Local while recording. Queue acceptance transfers ownership to the
    // submission context; standalone callers retain until command reset.
    Result<> retainResource(std::shared_ptr<void> resource);
    Result<> end();
    void beginDebugLabel(const DebugLabelDesc& desc);
    void endDebugLabel();
    // Reset on a graphics/compute queue, including pools written by another queue.
    // The caller must order that queue after the reset before writing timestamps.
    Result<> resetTimestampQueries(
        TimestampQueryPool& queryPool,
        uint32_t firstQuery,
        uint32_t queryCount);
    Result<> writeTimestamp(
        TimestampQueryPool& queryPool,
        uint32_t queryIndex,
        PipelineStageBits stage);
    Result<> resetRayTracingAccelerationStructureCompactionQueries(
        RayTracingAccelerationStructureCompactionQueryPool& queryPool,
        uint32_t firstQuery,
        uint32_t queryCount);
    Result<> writeRayTracingAccelerationStructureCompactedSize(
        RayTracingAccelerationStructureCompactionQueryPool& queryPool,
        uint32_t queryIndex,
        RayTracingAccelerationStructure& accelerationStructure);
    // One dependency boundary; compatible resource barriers are coalesced.
    [[nodiscard]] Result<> synchronize(const BarrierDesc& desc);
    SynchronizationStats synchronizationStats() const;
    void hostWriteBarrier();
    [[nodiscard]] Result<> copyBuffer(const BufferSlice& source, const BufferSlice& destination);
    Result<> decompressBuffers(std::span<const BufferDecompressionDesc> regions);
    Result<> validateDecompressionBuffers(std::span<const BufferDecompressionDesc> regions) const;
    [[nodiscard]] Result<> copyTexture(const TextureCopyDesc& desc);
    [[nodiscard]] Result<> copyTextureToBuffer(const BufferTextureRegion& desc);
    [[nodiscard]] Result<> copyBufferToTexture(const BufferTextureRegion& desc);
    [[nodiscard]] Result<> clearColorTexture(Texture& texture, TextureLayout layout, const ColorValue& color = {});
    Result<> beginRendering(const RenderingDesc& desc);
    // Native SDK consumers retain the view itself as well as its image.
    Result<> useNativeTextureView(TextureView& view);
    void clearColorAttachment(uint32_t attachmentIndex, const ColorValue& color, const Rect& rect);
    void endRendering();
    [[nodiscard]] Result<> setViewport(const Viewport& viewport);
    void setScissor(const Rect& scissor);
    [[nodiscard]] Result<> bindExecution(const PreparedExecution& execution);
    [[nodiscard]] Result<> bindExecution(const PreparedExecution& execution, const void* pushData, uint32_t byteSize);
    [[nodiscard]] Result<> bindBindlessHeap(BindlessHeap& heap);
    // Upload the caller's shader parameter ABI at byte zero, without a heap header.
    [[nodiscard]] Result<> pushBindlessData(const void* data, uint32_t byteSize);
    // Record compute-only instrumentation, restoring the compute pipeline,
    // descriptor heap and shared push data before returning. No rendering scope.
    Result<> recordIsolatedCompute(const std::function<Result<>()>& record);
    // Direct draw/dispatch calls require active recording and a compatible queue first.
    // A zero vertex/instance count or group dimension then succeeds without recording work,
    // even when the optional mesh feature is unavailable. Non-empty unsupported work returns Unsupported.
    // This no-op rule does not apply to zero-sized copy regions or AS build descriptions.
    [[nodiscard]] Result<> draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t firstInstance = 0);
    // Direct mesh dispatch for CPU-known counts; distinct from GPU-generated indirect draws.
    [[nodiscard]] Result<> drawMeshTasks(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    [[nodiscard]] Result<> drawMeshTasksIndirect(const BufferSlice& arguments);
    [[nodiscard]] Result<> dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    // Three GPU-written uint32 group counts; offset is 4-byte aligned.
    Result<> dispatchIndirect(const BufferSlice& arguments);
    Result<> buildClusterAccelerationStructureTriangles(
        const ClusterAccelerationStructureTriangleBuildDesc& desc);
    Result<> moveClusterAccelerationStructures(const ClusterAccelerationStructureMoveDesc& desc);
    Result<> buildClusterAccelerationStructureBottomLevels(
        const ClusterAccelerationStructureBottomLevelBuildDesc& desc);
    Result<> buildPartitionedAccelerationStructure(
        const PartitionedAccelerationStructureBuildDesc& desc);
    Result<> buildRayTracingAccelerationStructure(
        const RayTracingAccelerationStructureBuildDesc& desc);
    Result<> compactRayTracingAccelerationStructure(
        RayTracingAccelerationStructure& source,
        RayTracingAccelerationStructure& destination);

private:
    enum class BufferTextureCopyDirection { ToBuffer, ToTexture };
    Result<> copyBufferTexture(const BufferTextureRegion& region, BufferTextureCopyDirection direction);
    void setGraphicsShaderObjectState();
    void setDepthStencilState(const DepthStencilState& state);
    Result<> bindExecutionImpl(const PreparedExecution& execution, const void* data, uint32_t byteSize, bool replaceData);

    std::shared_ptr<CommandSubmissionContext> submissionContext_;
    std::shared_ptr<detail::CommandSubmissionState> submission_;
    std::vector<SemaphoreSubmitDesc> dependencyWaits_;
    std::vector<std::shared_ptr<const void>> dependencyLifetimes_;
    bool recording_ = false;
};

class CommandPool {
    METALLIC_RHI_HANDLE(CommandPool, unique_ptr,
        friend class Device;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    Result<> reset();
    [[nodiscard]] Result<std::unique_ptr<CommandBuffer>> createCommandBuffer();
};

class Swapchain {
    METALLIC_RHI_HANDLE(Swapchain, unique_ptr,
        friend class Device;
        friend struct detail::DeviceImpl;
        friend struct detail::VulkanNativeAccess;
    )

    uint32_t imageCount() const;
    uint32_t width() const;
    uint32_t height() const;
    Format format() const;
    // The negotiated mode, including any SDR fallback.
    DisplayOutputMode outputMode() const;
    Texture* texture(uint32_t imageIndex);
    [[nodiscard]] Result<uint32_t> acquireNextImage(SwapchainSemaphore& semaphore);
    Result<> present(Queue& queue, uint32_t imageIndex, SwapchainSemaphore& waitSemaphore);
};


class Device {
    METALLIC_RHI_HANDLE(Device, unique_ptr,
        friend Result<std::unique_ptr<Device>> createDevice(const DeviceDesc& desc);
        friend class CommandBuffer;
        friend struct detail::VulkanNativeAccess;
    )

    const DeviceCapabilities& capabilities() const;
    const void* identity() const;
    // Owner-defined per-device state. Factory runs once under a lock and must
    // not recursively request shared state. Released before native teardown;
    // external references and their resources must not outlive Device.
    [[nodiscard]] Result<std::shared_ptr<void>> sharedState(
        const void* key, const std::function<Result<std::shared_ptr<void>>()>& factory);
    DeviceMemoryBudget memoryBudget() const;
    void setMemoryBudgetPolicy(const MemoryBudgetPolicy& policy);
    [[nodiscard]] Result<MemoryBudgetReservation> reserveMemoryBudget(uint64_t bytes);
    void logMemoryBudget(const char* phase) const;
    Queue* getQueue(QueueType type, uint32_t index = 0);
    Result<> waitIdle();
    [[nodiscard]] Result<std::unique_ptr<Swapchain>> createSwapchain(const SwapchainDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<CommandPool>> createCommandPool(Queue& queue);
    [[nodiscard]] Result<std::unique_ptr<Fence>> createFence(bool signaled);
    [[nodiscard]] Result<std::unique_ptr<TimestampQueryPool>> createTimestampQueryPool(Queue& queue,
        const TimestampQueryPoolDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<RayTracingAccelerationStructureCompactionQueryPool>> createRayTracingAccelerationStructureCompactionQueryPool(const RayTracingAccelerationStructureCompactionQueryPoolDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<Semaphore>> createSemaphore(const SemaphoreDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<Semaphore>> createSemaphore();
    [[nodiscard]] Result<std::unique_ptr<SwapchainSemaphore>> createSwapchainSemaphore();
    [[nodiscard]] Result<std::unique_ptr<Buffer>> createBuffer(const BufferDesc& desc);
    [[nodiscard]] Result<uint64_t> bufferAllocationSize(const BufferDesc& desc);
    [[nodiscard]] Result<BufferAllocationRequirements> bufferAliasAllocationRequirements(const BufferDesc& desc);
    // Creates independent Device buffers at offset zero in one shared allocation,
    // accounted once in their common memory domain. The caller must order alias
    // uses and initialize every buffer after each handoff. Only Other and
    // FrameResources domains are supported; host buffers and acceleration-
    // structure/decompression usages are unsupported. Device must
    // outlive buffers, views, and retained commands. Failure returns no partial group.
    [[nodiscard]] Result<std::vector<std::unique_ptr<Buffer>>> createAliasedBuffers(std::span<const BufferDesc> descriptions);
    [[nodiscard]] Result<RayTracingAccelerationStructureProperties> queryRayTracingAccelerationStructureProperties() const;
    [[nodiscard]] Result<RayTracingAccelerationStructureBuildSizes> queryRayTracingAccelerationStructureBuildSizes(const RayTracingAccelerationStructureBuildInputs& inputs) const;
    [[nodiscard]] Result<std::unique_ptr<RayTracingAccelerationStructure>> createRayTracingAccelerationStructure(const RayTracingAccelerationStructureDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<RayTracingAccelerationStructure>> createRayTracingAccelerationStructure(const PartitionedAccelerationStructureDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<Buffer>> createRayTracingInstanceBuffer(std::span<const RayTracingInstanceDesc> instances);
    Result<> writeRayTracingInstances(
        Buffer& buffer,
        std::span<const RayTracingInstanceDesc> instances);
    [[nodiscard]] Result<std::unique_ptr<BufferView>> createBufferView(Buffer& buffer, const BufferViewDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<Texture>> createTexture(const TextureDesc& desc);
    [[nodiscard]] Result<uint64_t> textureAllocationSize(const TextureDesc& desc);
    [[nodiscard]] Result<TextureAllocationRequirements> textureAliasAllocationRequirements(const TextureDesc& desc);
    // Creates independent Device Texture2D color images at offset zero in one
    // shared allocation, accounted once as FrameResources. The caller must order
    // alias uses and initialize each image after every handoff. Device must outlive
    // the images, views, and retained commands. Failure returns no partial group.
    [[nodiscard]] Result<std::vector<std::unique_ptr<Texture>>> createAliasedTextures(std::span<const TextureDesc> descriptions);
    [[nodiscard]] Result<std::unique_ptr<TextureView>> createTextureView(Texture& texture, const TextureViewDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<ShaderModule>> createShaderModule(const ShaderModuleDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<PipelineCache>> createPipelineCache(const PipelineCacheDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<GraphicsPipeline>> createGraphicsPipeline(const GraphicsPipelineDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<ComputePipeline>> createComputePipeline(const ComputePipelineDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<GraphicsShaderObjectProgram>> createGraphicsShaderObjectProgram(const GraphicsShaderObjectProgramDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<BindlessHeap>> createBindlessHeap(const BindlessHeapDesc& desc);
    [[nodiscard]] Result<ClusterAccelerationStructureProperties> queryClusterAccelerationStructureProperties() const;
    [[nodiscard]] Result<ClusterAccelerationStructureBuildSizes> queryClusterAccelerationStructureTriangleBuildSizes(const ClusterAccelerationStructureTriangleBuildSizesDesc& desc) const;
    // MOVE_OBJECTS reports its required scratch bytes in updateScratchSize.
    [[nodiscard]] Result<ClusterAccelerationStructureBuildSizes> queryClusterAccelerationStructureMoveSizes(uint32_t maxCount, uint64_t maxBytes) const;
    [[nodiscard]] Result<ClusterAccelerationStructureBuildSizes> queryClusterAccelerationStructureBottomLevelBuildSizes(const ClusterAccelerationStructureBottomLevelBuildSizesDesc& desc) const;
    [[nodiscard]] Result<PartitionedAccelerationStructureBuildSizes> queryPartitionedAccelerationStructureBuildSizes(const PartitionedAccelerationStructureBuildInputs& inputs) const;
    [[nodiscard]] Result<std::unique_ptr<Buffer>> createPartitionedAccelerationStructureInstanceBuffer(std::span<const PartitionedAccelerationStructureInstanceDesc> instances);

private:
    bool validShaderStage(const ShaderStageDesc& stage) const;
    Result<std::unique_ptr<ComputePipeline>> createComputePipelineImpl(const ComputePipelineDesc& desc,
        std::span<const detail::ShaderBindingMappingDesc> mappings);
    static Result<std::unique_ptr<Buffer>> createBuffer(detail::DeviceImpl* implementation, const BufferDesc& desc);
};

[[nodiscard]] Result<std::unique_ptr<Device>> createDevice(const DeviceDesc& desc);
} // namespace metallic::render
