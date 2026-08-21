#include "Runtime/Render/GAPI/Vulkan/VulkanNrcWrapper.h"

#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <spdlog/spdlog.h>

#include <array>
#include <mutex>

namespace metallic::render::vulkan {

#if METALLIC_HAS_NRC

namespace {

Result resultFromNrc(nrc::Status status)
{
    switch (status) {
    case nrc::Status::OK:
        return {};
    case nrc::Status::WrongParameter:
    case nrc::Status::MemoryNotProvided:
        return makeError(Error::InvalidArgument);
    case nrc::Status::OutOfMemory:
    case nrc::Status::AllocationFailed:
        return makeError(Error::OutOfMemory);
    case nrc::Status::UnsupportedDriver:
    case nrc::Status::UnsupportedHardware:
        return makeError(Error::Unsupported);
    case nrc::Status::SDKVersionMismatch:
    case nrc::Status::AlreadyInitialized:
    case nrc::Status::SDKNotInitialized:
    case nrc::Status::InternalError:
    case nrc::Status::ErrorParsingJSON:
        return makeError(Error::Failure);
    }
    return makeError(Error::Failure);
}

std::string nrcResultMessage(std::string_view label, nrc::Status status)
{
    std::string message(label);
    message += " returned ";
    message += std::to_string(static_cast<uint32_t>(status));
    return message;
}

void nrcLoggerCallback(const char* message, nrc::LogLevel logLevel)
{
    if (message == nullptr) {
        return;
    }
    switch (logLevel) {
    case nrc::LogLevel::Debug:
        spdlog::debug("[NRC] {}", message);
        break;
    case nrc::LogLevel::Info:
        spdlog::info("[NRC] {}", message);
        break;
    case nrc::LogLevel::Warning:
        spdlog::warn("[NRC] {}", message);
        break;
    case nrc::LogLevel::Error:
    default:
        spdlog::error("[NRC] {}", message);
        break;
    }
}

void nrcMemoryLoggerCallback(nrc::MemoryEventType eventType, size_t size, const char* bufferName)
{
    const char* name = bufferName != nullptr ? bufferName : "?";
    switch (eventType) {
    case nrc::MemoryEventType::Allocation:
        spdlog::debug("[NRC] allocated {} bytes ({})", size, name);
        break;
    case nrc::MemoryEventType::Deallocation:
        spdlog::debug("[NRC] deallocated {} bytes ({})", size, name);
        break;
    case nrc::MemoryEventType::MemoryStats:
        spdlog::debug("[NRC] {} bytes currently allocated", size);
        break;
    }
}

// The NRC library is initialized once per process; contexts are per instance.
struct NrcLibraryRefCount {
    std::mutex mutex;
    uint32_t count = 0;
};

NrcLibraryRefCount& nrcLibraryRefCount()
{
    static NrcLibraryRefCount refCount;
    return refCount;
}

} // namespace

NrcIntegration::NrcIntegration() = default;

NrcIntegration::~NrcIntegration()
{
    clear();
}

NrcIntegration::NrcIntegration(NrcIntegration&&) noexcept = default;
NrcIntegration& NrcIntegration::operator=(NrcIntegration&&) noexcept = default;

Result NrcIntegration::initialize(Device& device, std::string& log)
{
    if (valid()) {
        return {};
    }

    const NativeDevice nativeDeviceInfo = nativeDevice(device);
    if (nativeDeviceInfo.device == VK_NULL_HANDLE ||
        nativeDeviceInfo.physicalDevice == VK_NULL_HANDLE ||
        nativeDeviceInfo.instance == VK_NULL_HANDLE) {
        log = "NRC requires a live Vulkan device, physical device and instance";
        return makeError(Error::Failure);
    }

    {
        std::lock_guard<std::mutex> lock(nrcLibraryRefCount().mutex);
        if (nrcLibraryRefCount().count == 0) {
            nrc::GlobalSettings globalSettings;
            globalSettings.loggerFn = &nrcLoggerCallback;
            globalSettings.memoryLoggerFn = &nrcMemoryLoggerCallback;
            // Metallic allocates and binds the NRC buffers through the RHI.
            globalSettings.enableGPUMemoryAllocation = false;
            // Debug buffers enable the extra resolve debug modes.
            globalSettings.enableDebugBuffers = true;
            globalSettings.maxNumFramesInFlight = 4;
            globalSettings.depsDirectoryPath = nullptr;

            const nrc::Status initStatus = nrc::vulkan::Initialize(globalSettings);
            if (initStatus != nrc::Status::OK) {
                log = nrcResultMessage("nrc::vulkan::Initialize", initStatus);
                return resultFromNrc(initStatus);
            }
        }
        ++nrcLibraryRefCount().count;
    }

    nrc::vulkan::Context* context = nullptr;
    const nrc::Status createStatus = nrc::vulkan::Context::Create(
        nativeDeviceInfo.device,
        nativeDeviceInfo.physicalDevice,
        nativeDeviceInfo.instance,
        context);
    if (createStatus != nrc::Status::OK || context == nullptr) {
        log = nrcResultMessage("nrc::vulkan::Context::Create", createStatus);
        clear();
        return resultFromNrc(createStatus);
    }
    context_ = context;
    return {};
}

void NrcIntegration::clear()
{
    if (context_ != nullptr) {
        nrc::vulkan::Context::Destroy(*context_);
        context_ = nullptr;
    }
    for (std::unique_ptr<Buffer>& buffer : buffers_) {
        buffer.reset();
    }
    nativeBuffers_ = nrc::vulkan::Buffers {};

    {
        std::lock_guard<std::mutex> lock(nrcLibraryRefCount().mutex);
        if (nrcLibraryRefCount().count > 0) {
            --nrcLibraryRefCount().count;
            if (nrcLibraryRefCount().count == 0) {
                nrc::vulkan::Shutdown();
            }
        }
    }
}

bool NrcIntegration::valid() const
{
    return context_ != nullptr;
}

Result NrcIntegration::configure(const nrc::ContextSettings& settings, Device& device, std::string& log)
{
    if (!valid()) {
        log = "NRC integration is not initialized";
        return makeError(Error::Failure);
    }

    nrc::BuffersAllocationInfo allocationInfo;
    const nrc::Status allocationStatus =
        nrc::vulkan::Context::GetBuffersAllocationInfo(settings, allocationInfo);
    if (allocationStatus != nrc::Status::OK) {
        log = nrcResultMessage("nrc::vulkan::Context::GetBuffersAllocationInfo", allocationStatus);
        return resultFromNrc(allocationStatus);
    }

    for (uint32_t index = 0; index < kBufferCount; ++index) {
        const nrc::BufferIdx bufferIdx = static_cast<nrc::BufferIdx>(index);
        const nrc::AllocationInfo& info = allocationInfo[bufferIdx];
        buffers_[index].reset();
        if (info.elementCount == 0 || info.elementSize == 0) {
            continue;
        }

        BufferDesc desc{
            .size = static_cast<uint64_t>(info.elementCount) * static_cast<uint64_t>(info.elementSize),
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                BufferUsageBits::TransferDestination | BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::Device,
        };
        std::unique_ptr<Buffer> buffer;
        const Result result = device.createBuffer(desc, buffer);
        if (!result || buffer == nullptr) {
            const char* debugName = info.debugName != nullptr ? info.debugName : "?";
            log += "createBuffer(NRC ";
            log += debugName;
            log += ") returned ";
            log += result ? "null buffer" : resultToString(result);
            log += '\n';
            for (uint32_t cleanup = 0; cleanup < kBufferCount; ++cleanup) {
                buffers_[cleanup].reset();
            }
            return result ? makeError(Error::Failure) : result;
        }
        buffers_[index] = std::move(buffer);
    }

    for (uint32_t index = 0; index < kBufferCount; ++index) {
        const nrc::BufferIdx bufferIdx = static_cast<nrc::BufferIdx>(index);
        nrc::vulkan::BufferInfo& bufferInfo = nativeBuffers_[bufferIdx];
        bufferInfo = nrc::vulkan::BufferInfo {};
        if (buffers_[index] == nullptr) {
            continue;
        }
        const NativeBuffer nativeBufferInfo = nativeBuffer(*buffers_[index]);
        bufferInfo.resource = nativeBufferInfo.buffer;
        bufferInfo.allocatedSize = buffers_[index]->desc().size;
        bufferInfo.allocatedOffset = 0;
        bufferInfo.memory = VK_NULL_HANDLE;
        bufferInfo.deviceAddress = nativeBufferInfo.address;
    }

    const nrc::Status configureStatus = context_->Configure(settings, &nativeBuffers_);
    if (configureStatus != nrc::Status::OK) {
        log = nrcResultMessage("nrc::vulkan::Context::Configure", configureStatus);
        return resultFromNrc(configureStatus);
    }
    contextSettings_ = settings;
    return {};
}

Result NrcIntegration::beginFrame(CommandBuffer& commandBuffer, const nrc::FrameSettings& frameSettings)
{
    if (!valid()) {
        return makeError(Error::Failure);
    }
    const nrc::Status status = context_->BeginFrame(nativeCommandBuffer(commandBuffer), frameSettings);
    return resultFromNrc(status);
}

Result NrcIntegration::populateShaderConstants(::NrcConstants& outConstants) const
{
    if (!valid()) {
        return makeError(Error::Failure);
    }
    const nrc::Status status = context_->PopulateShaderConstants(outConstants);
    return resultFromNrc(status);
}

Result NrcIntegration::queryAndTrain(CommandBuffer& commandBuffer, float* trainingLoss)
{
    if (!valid()) {
        return makeError(Error::Failure);
    }
    const nrc::Status status = context_->QueryAndTrain(nativeCommandBuffer(commandBuffer), trainingLoss);
    return resultFromNrc(status);
}

Result NrcIntegration::resolve(CommandBuffer& commandBuffer, TextureView& outputView)
{
    if (!valid()) {
        return makeError(Error::Failure);
    }
    const nrc::Status status = context_->Resolve(nativeCommandBuffer(commandBuffer), nativeImageView(outputView));
    return resultFromNrc(status);
}

Result NrcIntegration::endFrame(Queue& queue)
{
    if (!valid()) {
        return makeError(Error::Failure);
    }
    const NativeQueue nativeQueueInfo = nativeQueue(queue);
    const nrc::Status status = context_->EndFrame(nativeQueueInfo.queue);
    return resultFromNrc(status);
}

#endif // METALLIC_HAS_NRC

} // namespace metallic::render::vulkan
