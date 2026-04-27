#include "metal_runtime.h"

#include "glfw_metal_bridge.h"
#include "metal_resource_utils.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kDefaultUploadPageSize = 16ull * 1024ull * 1024ull;
constexpr size_t kUploadAlignment = 256;
constexpr NS::UInteger kMaxArgumentTableBuffers = 31;
constexpr NS::UInteger kMaxArgumentTableTextures = 128;
constexpr NS::UInteger kMaxArgumentTableSamplers = 16;

struct UploadPage {
    MTL::Buffer* buffer = nullptr;
    size_t offset = 0;
    size_t size = 0;
};

struct CommandBufferState {
    MTL4::CommandAllocator* allocator = nullptr;
    MTL4::ArgumentTable* vertexTable = nullptr;
    MTL4::ArgumentTable* fragmentTable = nullptr;
    MTL4::ArgumentTable* meshTable = nullptr;
    MTL4::ArgumentTable* computeTable = nullptr;
    std::vector<UploadPage> uploadPages;
    bool ended = false;
    bool committed = false;
    uint64_t frameSerial = 0;
};

std::mutex g_commandBufferMutex;
std::unordered_map<void*, CommandBufferState> g_commandBufferStates;

MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

MTL4::CommandQueue* metalCommandQueue(void* handle) {
    return static_cast<MTL4::CommandQueue*>(handle);
}

MTL::SharedEvent* metalSharedEvent(void* handle) {
    return static_cast<MTL::SharedEvent*>(handle);
}

MTL::ResidencySet* metalResidencySet(void* handle) {
    return static_cast<MTL::ResidencySet*>(handle);
}

CA::MetalLayer* metalLayer(void* handle) {
    return static_cast<CA::MetalLayer*>(handle);
}

CA::MetalDrawable* metalDrawable(void* handle) {
    return static_cast<CA::MetalDrawable*>(handle);
}

MTL4::CommandBuffer* metalCommandBuffer(void* handle) {
    return static_cast<MTL4::CommandBuffer*>(handle);
}

size_t alignUp(size_t value, size_t alignment) {
    const size_t mask = alignment - 1;
    return (value + mask) & ~mask;
}

std::string metalErrorMessage(NS::Error* error, const char* fallback) {
    if (error && error->localizedDescription()) {
        return error->localizedDescription()->utf8String();
    }
    return fallback ? fallback : "Unknown Metal error";
}

MTL4::ArgumentTable* createArgumentTable(MTL::Device* device,
                                         const char* label,
                                         std::string& errorMessage) {
    auto* desc = MTL4::ArgumentTableDescriptor::alloc()->init();
    desc->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
    desc->setMaxBufferBindCount(kMaxArgumentTableBuffers);
    desc->setMaxTextureBindCount(kMaxArgumentTableTextures);
    desc->setMaxSamplerStateBindCount(kMaxArgumentTableSamplers);
    desc->setInitializeBindings(true);
    desc->setSupportAttributeStrides(true);

    NS::Error* error = nullptr;
    auto* table = device->newArgumentTable(desc, &error);
    desc->release();
    if (!table) {
        errorMessage = metalErrorMessage(error, "Failed to create Metal4 argument table");
    }
    return table;
}

bool createUploadPage(MTL::Device* device,
                      size_t requestedSize,
                      UploadPage& outPage) {
    const size_t pageSize = alignUp(std::max(kDefaultUploadPageSize, requestedSize), kUploadAlignment);
    auto* buffer = device->newBuffer(pageSize, MTL::ResourceStorageModeShared);
    if (!buffer) {
        return false;
    }

    buffer->setLabel(NS::String::string("Metal4 Upload Ring Page", NS::UTF8StringEncoding));
    metalTrackAllocation(device, buffer);
    outPage = { buffer, 0, pageSize };
    return true;
}

void releaseCommandBufferState(MTL4::CommandBuffer* commandBuffer, CommandBufferState& state) {
    for (auto& page : state.uploadPages) {
        if (page.buffer) {
            metalUntrackAllocation(page.buffer);
            page.buffer->release();
        }
    }
    state.uploadPages.clear();

    if (state.computeTable) state.computeTable->release();
    if (state.meshTable) state.meshTable->release();
    if (state.fragmentTable) state.fragmentTable->release();
    if (state.vertexTable) state.vertexTable->release();
    if (state.allocator) state.allocator->release();
    if (commandBuffer) commandBuffer->release();
}

} // namespace

bool createMetalRuntime(void* glfwWindowHandle,
                        MetalRuntimeContext& runtime,
                        std::string& errorMessage) {
    runtime = {};

    auto* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        errorMessage = "Metal is not supported on this device";
        return false;
    }

    if (!device->supportsFamily(MTL::GPUFamilyMetal4)) {
        errorMessage = "Metal 4 is not supported on this device";
        device->release();
        return false;
    }

    auto* commandQueue = device->newMTL4CommandQueue();
    if (!commandQueue) {
        errorMessage = "Failed to create Metal4 command queue";
        device->release();
        return false;
    }

    auto* sharedEvent = device->newSharedEvent();
    if (!sharedEvent) {
        errorMessage = "Failed to create Metal4 shared event";
        commandQueue->release();
        device->release();
        return false;
    }
    sharedEvent->setSignaledValue(0);

    auto* residencyDesc = MTL::ResidencySetDescriptor::alloc()->init();
    residencyDesc->setLabel(NS::String::string("Metallic Internal Residency Set", NS::UTF8StringEncoding));
    residencyDesc->setInitialCapacity(512);
    NS::Error* residencyError = nullptr;
    auto* residencySet = device->newResidencySet(residencyDesc, &residencyError);
    residencyDesc->release();
    if (!residencySet) {
        errorMessage = metalErrorMessage(residencyError, "Failed to create Metal4 residency set");
        sharedEvent->release();
        commandQueue->release();
        device->release();
        return false;
    }

    commandQueue->addResidencySet(residencySet);
    metalRegisterResidencySet(device, residencySet);

    auto* layer = metalLayer(attachMetalLayerToGLFWWindow(glfwWindowHandle));
    if (!layer) {
        errorMessage = "Failed to attach Metal layer to GLFW window";
        metalUnregisterResidencySet(device);
        residencySet->release();
        sharedEvent->release();
        commandQueue->release();
        device->release();
        return false;
    }

    layer->setDevice(device);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer->setFramebufferOnly(false);

    runtime.device = device;
    runtime.commandQueue = commandQueue;
    runtime.layer = layer;
    runtime.sharedEvent = sharedEvent;
    runtime.residencySet = residencySet;
    runtime.nextFrameSerial = 0;
    runtime.tracyContext = nullptr;
    return true;
}

void destroyMetalRuntime(MetalRuntimeContext& runtime) {
    if (runtime.sharedEvent && runtime.nextFrameSerial > 0) {
        metalRuntimeWaitForFrame(runtime, runtime.nextFrameSerial);
    }

    if (runtime.tracyContext) {
        tracyMetalDestroy(runtime.tracyContext);
        runtime.tracyContext = nullptr;
    }
    if (runtime.commandQueue && runtime.residencySet) {
        metalCommandQueue(runtime.commandQueue)->removeResidencySet(metalResidencySet(runtime.residencySet));
    }
    if (runtime.device) {
        metalUnregisterResidencySet(runtime.device);
    }
    if (runtime.residencySet) {
        metalResidencySet(runtime.residencySet)->release();
        runtime.residencySet = nullptr;
    }
    if (runtime.sharedEvent) {
        metalSharedEvent(runtime.sharedEvent)->release();
        runtime.sharedEvent = nullptr;
    }
    if (runtime.commandQueue) {
        metalCommandQueue(runtime.commandQueue)->release();
        runtime.commandQueue = nullptr;
    }
    if (runtime.device) {
        metalDevice(runtime.device)->release();
        runtime.device = nullptr;
    }
    runtime.layer = nullptr;
    runtime.nextFrameSerial = 0;
}

void* metalRuntimeCreateAutoreleasePool() {
    return NS::AutoreleasePool::alloc()->init();
}

void metalRuntimeDestroyAutoreleasePool(void* poolHandle) {
    if (poolHandle) {
        static_cast<NS::AutoreleasePool*>(poolHandle)->release();
    }
}

std::string metalRuntimeDeviceName(const MetalRuntimeContext& runtime) {
    auto* device = metalDevice(runtime.device);
    if (!device || !device->name()) {
        return {};
    }
    return device->name()->utf8String();
}

void metalRuntimeCollectGpuTimestamps(const MetalRuntimeContext& runtime) {
    (void)runtime;
}

void metalRuntimeSetDrawableSize(MetalRuntimeContext& runtime, uint32_t width, uint32_t height) {
    if (!runtime.layer) return;
    metalLayer(runtime.layer)->setDrawableSize(CGSizeMake(width, height));
}

void* metalRuntimeNextDrawable(MetalRuntimeContext& runtime) {
    if (!runtime.layer) return nullptr;
    return metalLayer(runtime.layer)->nextDrawable();
}

void* metalRuntimeDrawableTexture(void* drawableHandle) {
    auto* drawable = metalDrawable(drawableHandle);
    return drawable ? drawable->texture() : nullptr;
}

void* metalRuntimeCreateCommandBuffer(MetalRuntimeContext& runtime) {
    auto* device = metalDevice(runtime.device);
    if (!device || !runtime.commandQueue) {
        return nullptr;
    }

    auto* allocator = device->newCommandAllocator();
    auto* commandBuffer = device->newCommandBuffer();
    if (!allocator || !commandBuffer) {
        if (allocator) allocator->release();
        if (commandBuffer) commandBuffer->release();
        return nullptr;
    }
    allocator->reset();

    std::string errorMessage;
    CommandBufferState state;
    state.allocator = allocator;
    state.vertexTable = createArgumentTable(device, "Metal4 Vertex Arguments", errorMessage);
    state.fragmentTable = createArgumentTable(device, "Metal4 Fragment Arguments", errorMessage);
    state.meshTable = createArgumentTable(device, "Metal4 Mesh Arguments", errorMessage);
    state.computeTable = createArgumentTable(device, "Metal4 Compute Arguments", errorMessage);
    UploadPage page;
    const bool uploadCreated = createUploadPage(device, kDefaultUploadPageSize, page);
    if (!state.vertexTable || !state.fragmentTable || !state.meshTable || !state.computeTable || !uploadCreated) {
        if (uploadCreated) {
            metalUntrackAllocation(page.buffer);
            page.buffer->release();
        }
        releaseCommandBufferState(commandBuffer, state);
        return nullptr;
    }
    state.uploadPages.push_back(page);

    commandBuffer->beginCommandBuffer(allocator);
    if (runtime.residencySet) {
        commandBuffer->useResidencySet(metalResidencySet(runtime.residencySet));
    }

    {
        std::lock_guard<std::mutex> lock(g_commandBufferMutex);
        g_commandBufferStates[commandBuffer] = std::move(state);
    }

    return commandBuffer->autorelease();
}

uint64_t metalRuntimeCommitAndPresent(MetalRuntimeContext& runtime,
                                      void* commandBufferHandle,
                                      void* drawableHandle) {
    auto* commandQueue = metalCommandQueue(runtime.commandQueue);
    auto* commandBuffer = metalCommandBuffer(commandBufferHandle);
    auto* drawable = metalDrawable(drawableHandle);
    auto* sharedEvent = metalSharedEvent(runtime.sharedEvent);
    if (!commandQueue || !commandBuffer || !sharedEvent) {
        return 0;
    }

    {
        std::lock_guard<std::mutex> lock(g_commandBufferMutex);
        auto it = g_commandBufferStates.find(commandBuffer);
        if (it != g_commandBufferStates.end() && !it->second.ended) {
            commandBuffer->endCommandBuffer();
            it->second.ended = true;
            it->second.committed = true;
        }
    }

    const MTL4::CommandBuffer* buffers[] = { commandBuffer };
    commandQueue->commit(buffers, 1);
    if (drawable) {
        commandQueue->wait(drawable);
        commandQueue->signalDrawable(drawable);
        drawable->present();
    }

    const uint64_t frameSerial = ++runtime.nextFrameSerial;
    commandQueue->signalEvent(sharedEvent, frameSerial);
    {
        std::lock_guard<std::mutex> lock(g_commandBufferMutex);
        auto it = g_commandBufferStates.find(commandBuffer);
        if (it != g_commandBufferStates.end()) {
            it->second.frameSerial = frameSerial;
        }
    }
    return frameSerial;
}

bool metalRuntimeIsFrameComplete(const MetalRuntimeContext& runtime, uint64_t frameSerial) {
    auto* sharedEvent = metalSharedEvent(runtime.sharedEvent);
    return !sharedEvent || frameSerial == 0 || sharedEvent->signaledValue() >= frameSerial;
}

void metalRuntimeWaitForFrame(const MetalRuntimeContext& runtime, uint64_t frameSerial) {
    auto* sharedEvent = metalSharedEvent(runtime.sharedEvent);
    if (sharedEvent && frameSerial != 0) {
        sharedEvent->waitUntilSignaledValue(frameSerial, std::numeric_limits<uint64_t>::max());
    }
}

void metalRuntimeReleaseCommandBuffer(void* commandBufferHandle) {
    auto* commandBuffer = metalCommandBuffer(commandBufferHandle);
    if (!commandBuffer) {
        return;
    }

    CommandBufferState state;
    bool hasState = false;
    {
        std::lock_guard<std::mutex> lock(g_commandBufferMutex);
        auto it = g_commandBufferStates.find(commandBuffer);
        if (it != g_commandBufferStates.end()) {
            state = std::move(it->second);
            g_commandBufferStates.erase(it);
            hasState = true;
        }
    }

    if (hasState) {
        releaseCommandBufferState(commandBuffer, state);
    } else {
        commandBuffer->release();
    }
}

void* metalRuntimeArgumentTable(void* commandBufferHandle, MetalArgumentTableSlot slot) {
    auto* commandBuffer = metalCommandBuffer(commandBufferHandle);
    if (!commandBuffer) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_commandBufferMutex);
    const auto it = g_commandBufferStates.find(commandBuffer);
    if (it == g_commandBufferStates.end()) {
        return nullptr;
    }

    switch (slot) {
    case MetalArgumentTableSlot::Vertex: return it->second.vertexTable;
    case MetalArgumentTableSlot::Fragment: return it->second.fragmentTable;
    case MetalArgumentTableSlot::Mesh: return it->second.meshTable;
    case MetalArgumentTableSlot::Compute: return it->second.computeTable;
    default: return nullptr;
    }
}

bool metalRuntimeUploadBytes(void* commandBufferHandle,
                             const void* data,
                             size_t size,
                             size_t alignment,
                             MetalUploadAllocation& outAllocation) {
    outAllocation = {};
    auto* commandBuffer = metalCommandBuffer(commandBufferHandle);
    if (!commandBuffer || !data || size == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_commandBufferMutex);
    auto it = g_commandBufferStates.find(commandBuffer);
    if (it == g_commandBufferStates.end()) {
        return false;
    }

    auto* device = commandBuffer->device();
    auto& state = it->second;
    const size_t effectiveAlignment = std::max(alignment, kUploadAlignment);
    for (auto& page : state.uploadPages) {
        const size_t alignedOffset = alignUp(page.offset, effectiveAlignment);
        if (alignedOffset + size <= page.size) {
            auto* cpu = static_cast<uint8_t*>(page.buffer->contents()) + alignedOffset;
            std::memcpy(cpu, data, size);
            page.offset = alignedOffset + size;
            outAllocation = {
                page.buffer,
                cpu,
                static_cast<uint64_t>(alignedOffset),
                page.buffer->gpuAddress() + alignedOffset,
                size,
            };
            return true;
        }
    }

    UploadPage newPage;
    if (!createUploadPage(device, size + effectiveAlignment, newPage)) {
        return false;
    }
    const size_t alignedOffset = alignUp(newPage.offset, effectiveAlignment);
    auto* cpu = static_cast<uint8_t*>(newPage.buffer->contents()) + alignedOffset;
    std::memcpy(cpu, data, size);
    newPage.offset = alignedOffset + size;
    outAllocation = {
        newPage.buffer,
        cpu,
        static_cast<uint64_t>(alignedOffset),
        newPage.buffer->gpuAddress() + alignedOffset,
        size,
    };
    state.uploadPages.push_back(newPage);
    return true;
}
