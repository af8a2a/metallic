#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "rhi_backend.h"

struct GLFWwindow;

class RhiWindowRuntime {
public:
    virtual ~RhiWindowRuntime() = default;

    virtual RhiBackendType backendType() const = 0;
    virtual const RhiDevice& device() const = 0;
    virtual const RhiCommandQueue& commandQueue() const = 0;
    virtual const RhiNativeCommandBuffer& currentCommandBuffer() const = 0;
    virtual RhiTexture& currentBackbufferTexture() = 0;
    virtual const RhiTexture& currentBackbufferTexture() const = 0;
    virtual std::string deviceName() const = 0;

    virtual bool beginFrame(uint32_t width, uint32_t height) = 0;
    virtual void collectGpuTimestamps() = 0;
    virtual void present() = 0;

    virtual std::unique_ptr<RhiFrameGraphBackend> createFrameGraphBackend() const = 0;
    virtual std::unique_ptr<RhiCommandBuffer> createCommandBuffer() const = 0;

    virtual void initImGui() = 0;
    virtual void beginImGuiFrame(const RhiTexture* depthTexture) = 0;
    virtual void shutdownImGui() = 0;
};

std::unique_ptr<RhiWindowRuntime> createRhiWindowRuntime(RhiBackendType backend,
                                                         GLFWwindow* window,
                                                         std::string& errorMessage);
