#include "rhi_window_runtime.h"

#ifdef __APPLE__

#include "imgui_metal_bridge.h"
#include "metal_frame_graph.h"
#include "metal_runtime.h"

namespace {

class MetalWindowRuntime final : public RhiWindowRuntime {
public:
    explicit MetalWindowRuntime(MetalRuntimeContext runtime)
        : m_runtime(runtime),
          m_device(runtime.device),
          m_commandQueue(runtime.commandQueue) {}

    ~MetalWindowRuntime() override {
        shutdownImGui();
        cleanupFrameState();
        destroyMetalRuntime(m_runtime);
    }

    RhiBackendType backendType() const override { return RhiBackendType::Metal; }
    const RhiDevice& device() const override { return m_device; }
    const RhiCommandQueue& commandQueue() const override { return m_commandQueue; }
    const RhiNativeCommandBuffer& currentCommandBuffer() const override { return m_commandBuffer; }
    RhiTexture& currentBackbufferTexture() override { return m_backbufferTexture; }
    const RhiTexture& currentBackbufferTexture() const override { return m_backbufferTexture; }
    std::string deviceName() const override { return metalRuntimeDeviceName(m_runtime); }

    bool beginFrame(uint32_t width, uint32_t height) override {
        cleanupFrameState();

        m_autoreleasePool = metalRuntimeCreateAutoreleasePool();
        metalRuntimeSetDrawableSize(m_runtime, width, height);

        m_drawable = metalRuntimeNextDrawable(m_runtime);
        if (!m_drawable) {
            cleanupFrameState();
            return false;
        }

        void* drawableTexture = metalRuntimeDrawableTexture(m_drawable);
        if (!drawableTexture) {
            cleanupFrameState();
            return false;
        }

        m_commandBuffer.setNativeHandle(metalRuntimeCreateCommandBuffer(m_runtime));
        if (!m_commandBuffer.nativeHandle()) {
            cleanupFrameState();
            return false;
        }

        m_backbufferTexture.setNativeHandle(drawableTexture, width, height);
        return true;
    }

    void collectGpuTimestamps() override {
        metalRuntimeCollectGpuTimestamps(m_runtime);
    }

    void present() override {
        if (m_commandBuffer.nativeHandle() && m_drawable) {
            metalRuntimePresentDrawable(m_commandBuffer.nativeHandle(), m_drawable);
            metalRuntimeCommit(m_commandBuffer.nativeHandle());
        }
        cleanupFrameState();
    }

    std::unique_ptr<RhiFrameGraphBackend> createFrameGraphBackend() const override {
        return std::make_unique<MetalFrameGraphBackend>(m_device.nativeHandle());
    }

    std::unique_ptr<RhiCommandBuffer> createCommandBuffer() const override {
        return std::make_unique<MetalCommandBuffer>(m_commandBuffer.nativeHandle(),
                                                    m_runtime.tracyContext);
    }

    void initImGui() override {
        if (m_imguiInitialized) {
            return;
        }
        imguiInit(m_device.nativeHandle());
        m_imguiInitialized = true;
    }

    void beginImGuiFrame(const RhiTexture* depthTexture) override {
        imguiNewFrameForTargets(m_backbufferTexture.nativeHandle(),
                                depthTexture ? depthTexture->nativeHandle() : nullptr);
    }

    void shutdownImGui() override {
        if (!m_imguiInitialized) {
            return;
        }
        imguiShutdown();
        m_imguiInitialized = false;
    }

private:
    void cleanupFrameState() {
        m_commandBuffer.setNativeHandle(nullptr);
        m_backbufferTexture.setNativeHandle(nullptr);
        m_drawable = nullptr;
        if (m_autoreleasePool) {
            metalRuntimeDestroyAutoreleasePool(m_autoreleasePool);
            m_autoreleasePool = nullptr;
        }
    }

    MetalRuntimeContext m_runtime{};
    RhiDeviceHandle m_device;
    RhiCommandQueueHandle m_commandQueue;
    RhiNativeCommandBufferHandle m_commandBuffer;
    RhiTextureHandle m_backbufferTexture;
    void* m_autoreleasePool = nullptr;
    void* m_drawable = nullptr;
    bool m_imguiInitialized = false;
};

} // namespace

#endif

std::unique_ptr<RhiWindowRuntime> createRhiWindowRuntime(RhiBackendType backend,
                                                         GLFWwindow* window,
                                                         std::string& errorMessage) {
#ifdef __APPLE__
    if (backend == RhiBackendType::Metal) {
        MetalRuntimeContext runtime;
        if (!createMetalRuntime(window, runtime, errorMessage)) {
            return {};
        }
        return std::make_unique<MetalWindowRuntime>(runtime);
    }
#endif

    errorMessage = "Requested window runtime backend is not available in this build.";
    return {};
}
