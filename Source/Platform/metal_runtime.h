#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tracy_metal.h"

struct MetalRuntimeContext {
    void* device = nullptr;
    void* commandQueue = nullptr;
    void* layer = nullptr;
    void* sharedEvent = nullptr;
    void* residencySet = nullptr;
    uint64_t nextFrameSerial = 0;
    TracyMetalCtxHandle tracyContext = nullptr;
};

enum class MetalArgumentTableSlot : uint32_t {
    Vertex,
    Fragment,
    Mesh,
    Compute,
};

struct MetalUploadAllocation {
    void* buffer = nullptr;
    void* cpuAddress = nullptr;
    uint64_t offset = 0;
    uint64_t gpuAddress = 0;
    size_t size = 0;
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
void* metalRuntimeCreateCommandBuffer(MetalRuntimeContext& runtime);
uint64_t metalRuntimeCommitAndPresent(MetalRuntimeContext& runtime,
                                      void* commandBufferHandle,
                                      void* drawableHandle);
bool metalRuntimeIsFrameComplete(const MetalRuntimeContext& runtime, uint64_t frameSerial);
void metalRuntimeWaitForFrame(const MetalRuntimeContext& runtime, uint64_t frameSerial);
void metalRuntimeReleaseCommandBuffer(void* commandBufferHandle);
void* metalRuntimeArgumentTable(void* commandBufferHandle, MetalArgumentTableSlot slot);
bool metalRuntimeUploadBytes(void* commandBufferHandle,
                             const void* data,
                             size_t size,
                             size_t alignment,
                             MetalUploadAllocation& outAllocation);
