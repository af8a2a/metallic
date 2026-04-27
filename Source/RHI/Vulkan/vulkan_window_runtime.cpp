#include "rhi_window_runtime.h"

#ifdef _WIN32

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui_impl_vulkan.h"
#include "vulkan_backend.h"
#include "vulkan_descriptor_buffer.h"
#include "vulkan_descriptor_manager.h"
#include "vulkan_frame_graph.h"
#include "vulkan_resource_handles.h"
#include "vulkan_resource_state_tracker.h"
#include "Helpers/vulkan_upload_service.h"

#include "rhi_resource_utils.h"

#include <algorithm>
#include <memory>
#include <spdlog/spdlog.h>
#include <type_traits>

namespace {

void checkImGuiVkResult(VkResult result) {
    if (result == VK_SUCCESS) {
        return;
    }
    spdlog::error("ImGui Vulkan error: {}", static_cast<int>(result));
}

template <typename VkHandle>
VkHandle nativeToVkHandle(void* handle) {
    if constexpr (std::is_pointer_v<VkHandle>) {
        return reinterpret_cast<VkHandle>(handle);
    } else {
        return static_cast<VkHandle>(reinterpret_cast<uint64_t>(handle));
    }
}

class VulkanWindowRuntime final : public RhiWindowRuntime {
public:
    explicit VulkanWindowRuntime(std::unique_ptr<RhiContext> context)
        : m_context(std::move(context)) {
        const RhiNativeHandles& native = m_context->nativeHandles();
        m_device.setNativeHandle(native.device, m_context.get());
        m_commandQueue.setNativeHandle(native.queue);
        initDescriptorBackend();
        initUploadServices();
    }

    ~VulkanWindowRuntime() override {
        shutdownImGui();
        if (m_context) {
            m_context->waitIdle();
        }
        vulkanSetUploadService(nullptr);
        m_readbackService.destroy();
        m_uploadService.destroy();
        m_descriptorBufferManager.destroy();
        m_descriptorManager.destroy();
    }

    RhiBackendType backendType() const override { return RhiBackendType::Vulkan; }
    const RhiDevice& device() const override { return m_device; }
    const RhiCommandQueue& commandQueue() const override { return m_commandQueue; }
    const RhiNativeCommandBuffer& currentCommandBuffer() const override { return m_commandBuffer; }
    RhiTexture& currentBackbufferTexture() override { return m_backbufferTexture; }
    const RhiTexture& currentBackbufferTexture() const override { return m_backbufferTexture; }

    std::string deviceName() const override {
        return m_context ? m_context->deviceInfo().adapterName : std::string{};
    }

    bool beginFrame(uint32_t width, uint32_t height) override {
        if (!m_context) {
            return false;
        }

        if (width != m_context->drawableWidth() || height != m_context->drawableHeight()) {
            m_context->resize(width, height);
        }

        if (!m_context->beginFrame()) {
            if (vulkanIsDeviceLost(*m_context)) {
                spdlog::critical("Vulkan device lost: {}", vulkanDeviceLostMessage(*m_context));
            }
            return false;
        }

        const RhiNativeHandles& native = m_context->nativeHandles();
        m_commandBuffer.setNativeHandle(getVulkanCurrentCommandBuffer(*m_context));

        VkImage backbufferImage = getVulkanCurrentBackbufferImage(*m_context);
        VkImageView backbufferImageView = getVulkanCurrentBackbufferImageView(*m_context);
        VkExtent2D backbufferExtent = getVulkanCurrentBackbufferExtent(*m_context);
        m_backbufferTexture.set(backbufferImage,
                                backbufferImageView,
                                backbufferExtent.width,
                                backbufferExtent.height,
                                static_cast<VkFormat>(native.colorFormat),
                                RhiTextureUsage::RenderTarget);

        if (m_descriptorBackend) {
            m_descriptorBackend->resetFrame();
        }

        m_uploadService.beginFrame(m_uploadFrameIndex);
        m_readbackService.beginFrame(m_uploadFrameIndex);
        ++m_uploadFrameIndex;

        m_resourceStateTracker.clear();
        if (backbufferImage != VK_NULL_HANDLE) {
            m_resourceStateTracker.setLayout(backbufferImage,
                                             getVulkanCurrentBackbufferLayout(*m_context));
        }

        VkCommandBuffer nativeCommandBuffer = getVulkanCurrentCommandBuffer(*m_context);
        if (m_uploadService.hasPendingUploads()) {
            m_uploadService.recordPendingUploads(nativeCommandBuffer);
        }
        m_readbackService.recordPendingReadbacks(nativeCommandBuffer);

        m_activeCommandBuffer = nullptr;
        return true;
    }

    void collectGpuTimestamps() override {}

    void present() override {
        if (!m_context) {
            return;
        }

        if (m_activeCommandBuffer && m_activeCommandBuffer->hadAsyncComputeWork()) {
            vulkanScheduleAsyncComputeSubmit(*m_context);
        }
        m_activeCommandBuffer = nullptr;

        m_context->endFrame();
        if (vulkanIsDeviceLost(*m_context)) {
            spdlog::critical("Vulkan submit/present reported device loss: {}",
                             vulkanDeviceLostMessage(*m_context));
        }
    }

    std::unique_ptr<RhiFrameGraphBackend> createFrameGraphBackend() const override {
        auto backend = std::make_unique<VulkanFrameGraphBackend>(
            getVulkanDevice(*m_context),
            getVulkanPhysicalDevice(*m_context),
            getVulkanAllocator(*m_context));
        backend->setTransientPool(&getVulkanTransientPool(*m_context));
        return backend;
    }

    std::unique_ptr<RhiCommandBuffer> createCommandBuffer() const override {
        auto commandBuffer = std::make_unique<VulkanCommandBuffer>(
            getVulkanCurrentCommandBuffer(*m_context),
            getVulkanDevice(*m_context),
            m_descriptorBackend,
            const_cast<VulkanResourceStateTracker*>(&m_resourceStateTracker),
            getVulkanGpuProfiler(*m_context),
            getVulkanCurrentComputeCommandBuffer(*m_context));
        m_activeCommandBuffer = commandBuffer.get();
        return commandBuffer;
    }

    const IRhiInteropProvider* interopProvider() const override {
        return m_context ? m_context->interopProvider() : nullptr;
    }

    void initImGui() override {
        if (m_imguiInitialized || !m_context) {
            return;
        }

        const RhiNativeHandles& native = m_context->nativeHandles();
        const uint32_t swapchainImageCount = std::max(2u, native.swapchainImageCount);
        const VkFormat colorFormat = static_cast<VkFormat>(native.colorFormat);
        const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

        VkPipelineRenderingCreateInfoKHR renderingInfo{
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;
        renderingInfo.depthAttachmentFormat = depthFormat;

        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.ApiVersion = native.apiVersion;
        initInfo.Instance = nativeToVkHandle<VkInstance>(native.instance);
        initInfo.PhysicalDevice = nativeToVkHandle<VkPhysicalDevice>(native.physicalDevice);
        initInfo.Device = nativeToVkHandle<VkDevice>(native.device);
        initInfo.QueueFamily = native.graphicsQueueFamily;
        initInfo.Queue = nativeToVkHandle<VkQueue>(native.queue);
        initInfo.DescriptorPool = nativeToVkHandle<VkDescriptorPool>(native.descriptorPool);
        initInfo.MinImageCount = swapchainImageCount;
        initInfo.ImageCount = swapchainImageCount;
        initInfo.UseDynamicRendering = true;
        initInfo.CheckVkResultFn = checkImGuiVkResult;
        initInfo.PipelineInfoMain.Subpass = 0;
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;

        if (!ImGui_ImplVulkan_Init(&initInfo)) {
            spdlog::error("Failed to initialize Dear ImGui for Vulkan");
            return;
        }

        m_imguiInitialized = true;
    }

    void beginImGuiFrame(const RhiTexture* /*depthTexture*/) override {
        if (m_imguiInitialized) {
            ImGui_ImplVulkan_NewFrame();
        }
    }

    void shutdownImGui() override {
        if (!m_imguiInitialized) {
            return;
        }
        if (m_context) {
            m_context->waitIdle();
        }
        ImGui_ImplVulkan_Shutdown();
        m_imguiInitialized = false;
    }

private:
    void initDescriptorBackend() {
        const RhiNativeHandles& native = m_context->nativeHandles();
        const RhiLimits& limits = m_context->limits();
        VkDevice device = nativeToVkHandle<VkDevice>(native.device);
        VkPhysicalDevice physicalDevice = nativeToVkHandle<VkPhysicalDevice>(native.physicalDevice);
        VmaAllocator allocator = getVulkanAllocator(*m_context);

        if (m_context->features().descriptorBuffer) {
            m_descriptorBufferManager.init(device,
                                           physicalDevice,
                                           allocator,
                                           getVulkanDescriptorBufferProperties(*m_context),
                                           limits.minUniformBufferOffsetAlignment,
                                           limits.nonCoherentAtomSize,
                                           limits.maxUniformBufferRange);
            m_descriptorBackend = &m_descriptorBufferManager;
            spdlog::info("Vulkan: using VK_EXT_descriptor_buffer path");
            return;
        }

        m_descriptorManager.init(device,
                                 physicalDevice,
                                 allocator,
                                 limits.minUniformBufferOffsetAlignment,
                                 limits.nonCoherentAtomSize,
                                 limits.maxUniformBufferRange);
        m_descriptorBackend = &m_descriptorManager;
    }

    void initUploadServices() {
        const RhiNativeHandles& native = m_context->nativeHandles();
        VkDevice device = nativeToVkHandle<VkDevice>(native.device);
        VkQueue graphicsQueue = nativeToVkHandle<VkQueue>(native.queue);
        VkQueue transferQueue = nativeToVkHandle<VkQueue>(native.transferQueue);
        VkSemaphore transferTimeline =
            nativeToVkHandle<VkSemaphore>(native.transferTimelineSemaphore);

        m_uploadService.init(device,
                             getVulkanAllocator(*m_context),
                             graphicsQueue,
                             native.graphicsQueueFamily,
                             transferQueue,
                             native.transferQueueFamily,
                             transferTimeline,
                             &getVulkanUploadRing(*m_context));
        vulkanSetUploadService(&m_uploadService);

        m_readbackService.init(device, &getVulkanReadbackHeap(*m_context), 2u);
    }

    std::unique_ptr<RhiContext> m_context;
    RhiDeviceHandle m_device;
    RhiCommandQueueHandle m_commandQueue;
    RhiNativeCommandBufferHandle m_commandBuffer;
    VulkanImportedTexture m_backbufferTexture;
    VulkanDescriptorManager m_descriptorManager;
    VulkanDescriptorBufferManager m_descriptorBufferManager;
    IVulkanDescriptorBackend* m_descriptorBackend = nullptr;
    mutable VulkanResourceStateTracker m_resourceStateTracker;
    VulkanUploadService m_uploadService;
    VulkanReadbackService m_readbackService;
    uint32_t m_uploadFrameIndex = 0;
    mutable VulkanCommandBuffer* m_activeCommandBuffer = nullptr;
    bool m_imguiInitialized = false;
};

} // namespace

std::unique_ptr<RhiWindowRuntime> createRhiWindowRuntime(RhiBackendType backend,
                                                         GLFWwindow* window,
                                                         std::string& errorMessage) {
    if (backend != RhiBackendType::Vulkan) {
        errorMessage = "Requested window runtime backend is not available in this build.";
        return {};
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

    RhiCreateInfo createInfo{};
    createInfo.window = window;
    createInfo.width = static_cast<uint32_t>(std::max(framebufferWidth, 1));
    createInfo.height = static_cast<uint32_t>(std::max(framebufferHeight, 1));
    createInfo.applicationName = "Metallic";
    createInfo.requireVulkan14 = true;
    createInfo.enableTimelineSemaphore = true;

    auto context = createRhiContext(RhiBackendType::Vulkan, createInfo, errorMessage);
    if (!context) {
        return {};
    }

    return std::make_unique<VulkanWindowRuntime>(std::move(context));
}

RhiBackendType defaultRhiWindowRuntimeBackend() {
    return RhiBackendType::Vulkan;
}

#endif
