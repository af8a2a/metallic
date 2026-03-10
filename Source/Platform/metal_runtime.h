#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tracy_metal.h"

struct MetalRuntimeContext {
    void* device = nullptr;
    void* commandQueue = nullptr;
    void* layer = nullptr;
    TracyMetalCtxHandle tracyContext = nullptr;
};

bool createMetalRuntime(void* glfwWindowHandle,
                        MetalRuntimeContext& runtime,
                        std::string& errorMessage);
void destroyMetalRuntime(MetalRuntimeContext& runtime);

void* metalRuntimeCreateAutoreleasePool();
void metalRuntimeDestroyAutoreleasePool(void* poolHandle);

std::string metalRuntimeDeviceName(const MetalRuntimeContext& runtime);
void metalRuntimeCollectGpuTimestamps(const MetalRuntimeContext& runtime);
void metalRuntimeSetDrawableSize(MetalRuntimeContext& runtime, uint32_t width, uint32_t height);
void* metalRuntimeNextDrawable(MetalRuntimeContext& runtime);
void* metalRuntimeDrawableTexture(void* drawableHandle);
void* metalRuntimeCreateCommandBuffer(const MetalRuntimeContext& runtime);
void metalRuntimePresentDrawable(void* commandBufferHandle, void* drawableHandle);
void metalRuntimeCommit(void* commandBufferHandle);
