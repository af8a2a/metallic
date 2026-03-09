#include "metal_runtime.h"

#include "glfw_metal_bridge.h"

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace {

MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

MTL::CommandQueue* metalCommandQueue(void* handle) {
    return static_cast<MTL::CommandQueue*>(handle);
}

CA::MetalLayer* metalLayer(void* handle) {
    return static_cast<CA::MetalLayer*>(handle);
}

CA::MetalDrawable* metalDrawable(void* handle) {
    return static_cast<CA::MetalDrawable*>(handle);
}

MTL::CommandBuffer* metalCommandBuffer(void* handle) {
    return static_cast<MTL::CommandBuffer*>(handle);
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

    auto* commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        errorMessage = "Failed to create Metal command queue";
        device->release();
        return false;
    }

    auto* layer = metalLayer(attachMetalLayerToGLFWWindow(glfwWindowHandle));
    if (!layer) {
        errorMessage = "Failed to attach Metal layer to GLFW window";
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
    runtime.tracyContext = tracyMetalCreate(device);
    return true;
}

void destroyMetalRuntime(MetalRuntimeContext& runtime) {
    if (runtime.tracyContext) {
        tracyMetalDestroy(runtime.tracyContext);
        runtime.tracyContext = nullptr;
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
}

std::string metalRuntimeDeviceName(const MetalRuntimeContext& runtime) {
    auto* device = metalDevice(runtime.device);
    if (!device || !device->name()) {
        return {};
    }
    return device->name()->utf8String();
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

void* metalRuntimeCreateCommandBuffer(const MetalRuntimeContext& runtime) {
    if (!runtime.commandQueue) return nullptr;
    return metalCommandQueue(runtime.commandQueue)->commandBuffer();
}

void metalRuntimePresentDrawable(void* commandBufferHandle, void* drawableHandle) {
    auto* commandBuffer = metalCommandBuffer(commandBufferHandle);
    auto* drawable = metalDrawable(drawableHandle);
    if (commandBuffer && drawable) {
        commandBuffer->presentDrawable(drawable);
    }
}

void metalRuntimeCommit(void* commandBufferHandle) {
    auto* commandBuffer = metalCommandBuffer(commandBufferHandle);
    if (commandBuffer) {
        commandBuffer->commit();
    }
}
