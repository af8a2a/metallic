#include "rhi_window_runtime.h"

#ifdef __APPLE__

#include "imgui_metal_bridge.h"
#include "metal_frame_graph.h"
#include "metal_resource_utils.h"
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
        releaseRetiredFrames(true);
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
        releaseRetiredFrames(false);
        cleanupFrameState();

        m_autoreleasePool = metalRuntimeCreateAutoreleasePool();
        metalRuntimeSetDrawableSize(m_runtime, width, height);

        m_drawable = metalRetainHandle(metalRuntimeNextDrawable(m_runtime));
        if (!m_drawable) {
            cleanupFrameState();
            return false;
        }

        void* drawableTexture = metalRuntimeDrawableTexture(m_drawable);
        if (!drawableTexture) {
            cleanupFrameState();
            return false;
        }

        m_commandBuffer.setNativeHandle(metalRetainHandle(metalRuntimeCreateCommandBuffer(m_runtime)));
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
            retireFrameState();
            return;
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
    struct RetiredFrameState {
        void* commandBuffer = nullptr;
        void* drawable = nullptr;
    };

    static bool isCommandBufferComplete(void* commandBufferHandle) {
        auto* commandBuffer = static_cast<MTL::CommandBuffer*>(commandBufferHandle);
        if (!commandBuffer) {
            return true;
        }

        const MTL::CommandBufferStatus status = commandBuffer->status();
        return status == MTL::CommandBufferStatusCompleted ||
               status == MTL::CommandBufferStatusError;
    }

    static void waitForCommandBuffer(void* commandBufferHandle) {
        auto* commandBuffer = static_cast<MTL::CommandBuffer*>(commandBufferHandle);
        if (commandBuffer) {
            commandBuffer->waitUntilCompleted();
        }
    }

    void releaseRetiredFrame(RetiredFrameState& frame) {
        if (frame.drawable) {
            metalReleaseHandle(frame.drawable);
            frame.drawable = nullptr;
        }
        if (frame.commandBuffer) {
            metalReleaseHandle(frame.commandBuffer);
            frame.commandBuffer = nullptr;
        }
    }

    void retireFrameState() {
        m_retiredFrames.push_back({
            m_commandBuffer.nativeHandle(),
            m_drawable,
        });
        m_commandBuffer.setNativeHandle(nullptr);
        m_backbufferTexture.setNativeHandle(nullptr);
        m_drawable = nullptr;
        if (m_autoreleasePool) {
            metalRuntimeDestroyAutoreleasePool(m_autoreleasePool);
            m_autoreleasePool = nullptr;
        }
    }

    void releaseRetiredFrames(bool waitForCompletion) {
        size_t releaseCount = 0;
        while (releaseCount < m_retiredFrames.size()) {
            auto& frame = m_retiredFrames[releaseCount];
            if (waitForCompletion) {
                waitForCommandBuffer(frame.commandBuffer);
            } else if (!isCommandBufferComplete(frame.commandBuffer)) {
                break;
            }

            releaseRetiredFrame(frame);
            ++releaseCount;
        }

        if (releaseCount > 0) {
            m_retiredFrames.erase(m_retiredFrames.begin(),
                                  m_retiredFrames.begin() + static_cast<std::ptrdiff_t>(releaseCount));
        }
    }

    void cleanupFrameState() {
        if (m_drawable) {
            metalReleaseHandle(m_drawable);
            m_drawable = nullptr;
        }
        if (m_commandBuffer.nativeHandle()) {
            metalReleaseHandle(m_commandBuffer.nativeHandle());
        }
        m_commandBuffer.setNativeHandle(nullptr);
        m_backbufferTexture.setNativeHandle(nullptr);
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
    std::vector<RetiredFrameState> m_retiredFrames;
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
