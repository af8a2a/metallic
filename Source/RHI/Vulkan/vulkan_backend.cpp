#include "vulkan_backend.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#ifndef VK_API_VERSION_1_4
#define VK_API_VERSION_1_4 VK_MAKE_API_VERSION(0, 1, 4, 0)
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMaxFramesInFlight = 2;
const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

VkFormat toVkFormat(RhiFormat format) {
    switch (format) {
    case RhiFormat::BGRA8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
    case RhiFormat::RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
    case RhiFormat::D32Float: return VK_FORMAT_D32_SFLOAT;
    case RhiFormat::Undefined:
    default: return VK_FORMAT_UNDEFINED;
    }
}

RhiFormat fromVkFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB: return RhiFormat::BGRA8Unorm;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB: return RhiFormat::RGBA8Unorm;
    case VK_FORMAT_D32_SFLOAT: return RhiFormat::D32Float;
    default: return RhiFormat::Undefined;
    }
}

VkFormat toVkVertexFormat(RhiVertexFormat format) {
    switch (format) {
    case RhiVertexFormat::Float2: return VK_FORMAT_R32G32_SFLOAT;
    case RhiVertexFormat::Float3: return VK_FORMAT_R32G32B32_SFLOAT;
    case RhiVertexFormat::Float4: return VK_FORMAT_R32G32B32A32_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

std::string vkVersionString(uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (VkResult=" + std::to_string(result) + ")");
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
        if (presentSupport == VK_TRUE) {
            indices.present = i;
        }

        if (indices.complete()) {
            break;
        }
    }

    return indices;
}

SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    if (formatCount > 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    if (presentModeCount > 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& format : availableFormats) {
        if ((format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_SRGB) &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return availableFormats.front();
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& presentMode : availablePresentModes) {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return presentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                            uint32_t requestedWidth,
                            uint32_t requestedHeight) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }

    VkExtent2D actualExtent = {requestedWidth, requestedHeight};
    actualExtent.width = std::clamp(actualExtent.width,
                                    capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height,
                                     capabilities.minImageExtent.height,
                                     capabilities.maxImageExtent.height);
    return actualExtent;
}

bool hasExtension(std::span<const VkExtensionProperties> extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

bool hasLayer(std::span<const VkLayerProperties> layers, const char* name) {
    return std::any_of(layers.begin(), layers.end(), [name](const VkLayerProperties& layer) {
        return std::strcmp(layer.layerName, name) == 0;
    });
}

class VulkanShaderModule final : public RhiShaderModule {
public:
    VulkanShaderModule(VkDevice device, VkShaderModule module)
        : m_device(device), m_module(module) {}

    ~VulkanShaderModule() override {
        if (m_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, m_module, nullptr);
        }
    }

    VkShaderModule handle() const { return m_module; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

class VulkanBuffer final : public RhiBuffer {
public:
    VulkanBuffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize size)
        : m_device(device), m_buffer(buffer), m_memory(memory), m_size(static_cast<size_t>(size)) {}

    ~VulkanBuffer() override {
        if (m_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_device, m_buffer, nullptr);
        }
        if (m_memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_memory, nullptr);
        }
    }

    size_t size() const override { return m_size; }
    VkBuffer handle() const { return m_buffer; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    size_t m_size = 0;
};

class VulkanGraphicsPipeline final : public RhiGraphicsPipeline {
public:
    VulkanGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkPipeline pipeline)
        : m_device(device), m_layout(layout), m_pipeline(pipeline) {}

    ~VulkanGraphicsPipeline() override {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        }
        if (m_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_layout, nullptr);
        }
    }

    VkPipeline pipeline() const { return m_pipeline; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

class VulkanContext final : public RhiContext {
public:
    explicit VulkanContext(const RhiCreateInfo& createInfo)
        : m_commandContext(*this) {
        if (!createInfo.window) {
            throw std::runtime_error("Vulkan RHI requires a valid GLFW window.");
        }

        m_requestedWidth = createInfo.width;
        m_requestedHeight = createInfo.height;
        createInstance(createInfo);
        createSurface(createInfo.window);
        pickPhysicalDevice(createInfo.requireVulkan14);
        createLogicalDevice(createInfo.enableValidation);
        createCommandObjects();
        createDescriptorPool();
        recreateSwapchain();
        populateNativeHandles();
    }

    ~VulkanContext() override {
        waitIdle();
        cleanupSwapchain();

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        }
        if (m_commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        }

        for (auto& frame : m_frames) {
            if (frame.imageAvailable != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
            }
            if (frame.renderFinished != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_device, frame.renderFinished, nullptr);
            }
            if (frame.inFlight != VK_NULL_HANDLE) {
                vkDestroyFence(m_device, frame.inFlight, nullptr);
            }
        }

        if (m_device != VK_NULL_HANDLE) {
            vkDestroyDevice(m_device, nullptr);
        }
        if (m_surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        }
        if (m_instance != VK_NULL_HANDLE) {
            vkDestroyInstance(m_instance, nullptr);
        }
    }

    RhiBackendType backendType() const override { return RhiBackendType::Vulkan; }
    const RhiFeatures& features() const override { return m_features; }
    const RhiDeviceInfo& deviceInfo() const override { return m_deviceInfo; }
    const RhiNativeHandles& nativeHandles() const override { return m_nativeHandles; }

    bool beginFrame() override {
        if (m_pendingResize) {
            recreateSwapchain();
            m_pendingResize = false;
        }

        FrameResources& frame = m_frames[m_frameIndex];
        checkVk(vkWaitForFences(m_device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX),
                "Failed to wait for in-flight frame fence");

        VkResult acquireResult = vkAcquireNextImageKHR(
            m_device,
            m_swapchain,
            UINT64_MAX,
            frame.imageAvailable,
            VK_NULL_HANDLE,
            &m_imageIndex);

        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return false;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire Vulkan swapchain image.");
        }

        if (m_imagesInFlight[m_imageIndex] != VK_NULL_HANDLE) {
            checkVk(vkWaitForFences(m_device, 1, &m_imagesInFlight[m_imageIndex], VK_TRUE, UINT64_MAX),
                    "Failed to wait for a busy swapchain image fence");
        }
        m_imagesInFlight[m_imageIndex] = frame.inFlight;

        checkVk(vkResetFences(m_device, 1, &frame.inFlight), "Failed to reset frame fence");
        checkVk(vkResetCommandPool(m_device, m_commandPool, 0), "Failed to reset command pool");

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo), "Failed to begin command buffer");

        m_insideRendering = false;
        return true;
    }

    void endFrame() override {
        FrameResources& frame = m_frames[m_frameIndex];
        transitionCurrentImage(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                               VK_ACCESS_2_NONE);

        checkVk(vkEndCommandBuffer(frame.commandBuffer), "Failed to end command buffer");

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &frame.imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &frame.commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &frame.renderFinished;

        checkVk(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, frame.inFlight), "Failed to submit graphics queue");

        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &frame.renderFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_swapchain;
        presentInfo.pImageIndices = &m_imageIndex;

        const VkResult presentResult = vkQueuePresentKHR(m_presentQueue, &presentInfo);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || m_pendingResize) {
            recreateSwapchain();
            m_pendingResize = false;
        } else if (presentResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to present Vulkan swapchain image.");
        }

        m_frameIndex = (m_frameIndex + 1) % kMaxFramesInFlight;
    }

    void resize(uint32_t width, uint32_t height) override {
        if (width > 0 && height > 0) {
            m_requestedWidth = width;
            m_requestedHeight = height;
            m_pendingResize = true;
        }
    }

    void waitIdle() override {
        if (m_device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_device);
        }
    }

    RhiCommandContext& commandContext() override { return m_commandContext; }
    uint32_t drawableWidth() const override { return m_swapchainExtent.width; }
    uint32_t drawableHeight() const override { return m_swapchainExtent.height; }
    RhiFormat colorFormat() const override { return fromVkFormat(m_swapchainFormat.format); }

    std::unique_ptr<RhiShaderModule> createShaderModule(const RhiShaderModuleDesc& desc) override {
        if (desc.spirv.empty()) {
            throw std::runtime_error("Cannot create Vulkan shader module from an empty SPIR-V blob.");
        }

        VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        createInfo.codeSize = desc.spirv.size() * sizeof(uint32_t);
        createInfo.pCode = desc.spirv.data();

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        checkVk(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule),
                "Failed to create Vulkan shader module");
        return std::make_unique<VulkanShaderModule>(m_device, shaderModule);
    }

    std::unique_ptr<RhiBuffer> createVertexBuffer(const RhiBufferDesc& desc) override {
        if (desc.size == 0) {
            throw std::runtime_error("Cannot create a zero-sized Vulkan buffer.");
        }

        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = desc.size;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        checkVk(vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer), "Failed to create Vulkan vertex buffer");

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, buffer, &memoryRequirements);

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(
            memoryRequirements.memoryTypeBits,
            desc.hostVisible ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkDeviceMemory memory = VK_NULL_HANDLE;
        checkVk(vkAllocateMemory(m_device, &allocInfo, nullptr, &memory), "Failed to allocate Vulkan buffer memory");
        checkVk(vkBindBufferMemory(m_device, buffer, memory, 0), "Failed to bind Vulkan buffer memory");

        if (desc.initialData) {
            if (!desc.hostVisible) {
                throw std::runtime_error("Device-local uploads are not implemented yet for Vulkan vertex buffers.");
            }

            void* mappedMemory = nullptr;
            checkVk(vkMapMemory(m_device, memory, 0, desc.size, 0, &mappedMemory), "Failed to map Vulkan buffer memory");
            std::memcpy(mappedMemory, desc.initialData, desc.size);
            vkUnmapMemory(m_device, memory);
        }

        return std::make_unique<VulkanBuffer>(m_device, buffer, memory, bufferInfo.size);
    }

    std::unique_ptr<RhiGraphicsPipeline> createGraphicsPipeline(const RhiGraphicsPipelineDesc& desc) override {
        const auto* shaderModule = dynamic_cast<const VulkanShaderModule*>(desc.shaderModule);
        if (!shaderModule) {
            throw std::runtime_error("Vulkan graphics pipeline requires a Vulkan shader module.");
        }

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        auto pushStage = [&](VkShaderStageFlagBits stage, const char* entryPoint) {
            VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stageInfo.stage = stage;
            stageInfo.module = shaderModule->handle();
            stageInfo.pName = entryPoint;
            shaderStages.push_back(stageInfo);
        };

        if (desc.enableMeshShaders) {
            if (!m_features.meshShaders) {
                throw std::runtime_error("The active Vulkan device does not support mesh shaders.");
            }
            if (desc.taskEntry && desc.taskEntry[0] != '\0') {
                pushStage(VK_SHADER_STAGE_TASK_BIT_EXT, desc.taskEntry);
            }
            pushStage(VK_SHADER_STAGE_MESH_BIT_EXT, desc.meshEntry ? desc.meshEntry : "meshMain");
        } else {
            pushStage(VK_SHADER_STAGE_VERTEX_BIT, desc.vertexEntry ? desc.vertexEntry : "vertexMain");
        }

        if (desc.fragmentEntry && desc.fragmentEntry[0] != '\0') {
            pushStage(VK_SHADER_STAGE_FRAGMENT_BIT, desc.fragmentEntry);
        }

        std::vector<VkVertexInputBindingDescription> bindings;
        bindings.reserve(desc.bindings.size());
        for (const auto& binding : desc.bindings) {
            bindings.push_back({binding.binding, binding.stride, VK_VERTEX_INPUT_RATE_VERTEX});
        }

        std::vector<VkVertexInputAttributeDescription> attributes;
        attributes.reserve(desc.attributes.size());
        for (const auto& attribute : desc.attributes) {
            attributes.push_back({attribute.location, attribute.binding, toVkVertexFormat(attribute.format), attribute.offset});
        }

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
        vertexInputInfo.pVertexBindingDescriptions = bindings.data();
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = desc.enableDepth ? VK_TRUE : VK_FALSE;
        depthStencil.depthWriteEnable = desc.enableDepth ? VK_TRUE : VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout layout = VK_NULL_HANDLE;
        checkVk(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &layout),
                "Failed to create Vulkan pipeline layout");

        VkFormat colorFormat = toVkFormat(desc.colorFormat);
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = desc.enableMeshShaders ? nullptr : &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = desc.enableMeshShaders ? nullptr : &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = layout;
        pipelineInfo.renderPass = VK_NULL_HANDLE;
        pipelineInfo.subpass = 0;
        pipelineInfo.pDepthStencilState = desc.enableDepth ? &depthStencil : nullptr;

        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult pipelineResult = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        if (pipelineResult != VK_SUCCESS) {
            vkDestroyPipelineLayout(m_device, layout, nullptr);
            throw std::runtime_error("Failed to create Vulkan graphics pipeline.");
        }

        return std::make_unique<VulkanGraphicsPipeline>(m_device, layout, pipeline);
    }

private:
    struct FrameResources {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
    };

    class CommandContext final : public RhiCommandContext {
    public:
        explicit CommandContext(VulkanContext& parent)
            : m_parent(parent) {}

        void beginRendering(const RhiRenderTargetInfo& targetInfo) override {
            if (m_parent.m_insideRendering) {
                throw std::runtime_error("Vulkan rendering has already begun for this frame.");
            }

            VkImageLayout currentLayout = m_parent.m_swapchainImageLayouts[m_parent.m_imageIndex];
            m_parent.transitionCurrentImage(currentLayout,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                            VK_ACCESS_2_NONE,
                                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

            VkClearValue clearValue{};
            clearValue.color = {{
                targetInfo.clearColor[0],
                targetInfo.clearColor[1],
                targetInfo.clearColor[2],
                targetInfo.clearColor[3],
            }};

            VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colorAttachment.imageView = m_parent.m_swapchainImageViews[m_parent.m_imageIndex];
            colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttachment.loadOp = targetInfo.clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.clearValue = clearValue;

            VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
            renderingInfo.renderArea.offset = {0, 0};
            renderingInfo.renderArea.extent = m_parent.m_swapchainExtent;
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &colorAttachment;

            vkCmdBeginRendering(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer, &renderingInfo);
            m_parent.m_insideRendering = true;
        }

        void endRendering() override {
            if (!m_parent.m_insideRendering) {
                return;
            }

            vkCmdEndRendering(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer);
            m_parent.m_insideRendering = false;
        }

        void setViewport(float width, float height) override {
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = width;
            viewport.height = height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer, 0, 1, &viewport);
        }

        void setScissor(uint32_t width, uint32_t height) override {
            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = {width, height};
            vkCmdSetScissor(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer, 0, 1, &scissor);
        }

        void bindGraphicsPipeline(const RhiGraphicsPipeline& pipeline) override {
            const auto& vkPipeline = dynamic_cast<const VulkanGraphicsPipeline&>(pipeline);
            vkCmdBindPipeline(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer,
                              VK_PIPELINE_BIND_POINT_GRAPHICS,
                              vkPipeline.pipeline());
        }

        void bindVertexBuffer(const RhiBuffer& buffer, uint64_t offset) override {
            const auto& vkBuffer = dynamic_cast<const VulkanBuffer&>(buffer);
            VkBuffer vertexBuffer = vkBuffer.handle();
            VkDeviceSize vertexOffset = offset;
            vkCmdBindVertexBuffers(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer,
                                   0, 1, &vertexBuffer, &vertexOffset);
        }

        void draw(uint32_t vertexCount,
                  uint32_t instanceCount,
                  uint32_t firstVertex,
                  uint32_t firstInstance) override {
            vkCmdDraw(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer,
                      vertexCount, instanceCount, firstVertex, firstInstance);
        }

        void* nativeCommandBuffer() const override {
            return m_parent.m_frames[m_parent.m_frameIndex].commandBuffer;
        }

    private:
        VulkanContext& m_parent;
    };

    void createInstance(const RhiCreateInfo& createInfo) {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions) {
            throw std::runtime_error("GLFW did not provide Vulkan instance extensions.");
        }

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        uint32_t availableLayerCount = 0;
        vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(availableLayerCount);
        vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data());

        std::vector<const char*> layers;
        const bool validationAvailable = hasLayer(availableLayers, kValidationLayerName);
        if (createInfo.enableValidation && validationAvailable) {
            layers.push_back(kValidationLayerName);
            m_features.validation = true;
        }

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = createInfo.applicationName;
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "Metallic";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_4;

        VkInstanceCreateInfo createInfoVk{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfoVk.pApplicationInfo = &appInfo;
        createInfoVk.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfoVk.ppEnabledExtensionNames = extensions.data();
        createInfoVk.enabledLayerCount = static_cast<uint32_t>(layers.size());
        createInfoVk.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

        checkVk(vkCreateInstance(&createInfoVk, nullptr, &m_instance), "Failed to create Vulkan instance");
    }

    void createSurface(GLFWwindow* window) {
        checkVk(glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface), "Failed to create Vulkan surface");
    }

    void pickPhysicalDevice(bool requireVulkan14) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("No Vulkan physical devices were found.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

        for (VkPhysicalDevice device : devices) {
            if (isDeviceSuitable(device, requireVulkan14)) {
                m_physicalDevice = device;
                vkGetPhysicalDeviceProperties(device, &m_physicalDeviceProperties);
                m_queueFamilies = findQueueFamilies(device, m_surface);
                m_deviceInfo.adapterName = m_physicalDeviceProperties.deviceName;
                m_deviceInfo.driverName = vkVersionString(m_physicalDeviceProperties.driverVersion);
                m_deviceInfo.apiVersion = m_physicalDeviceProperties.apiVersion;
                return;
            }
        }

        throw std::runtime_error("No suitable Vulkan 1.4 device was found for Metallic.");
    }

    bool isDeviceSuitable(VkPhysicalDevice device, bool requireVulkan14) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if (requireVulkan14 && VK_API_VERSION_MAJOR(properties.apiVersion) == 1 && VK_API_VERSION_MINOR(properties.apiVersion) < 4) {
            return false;
        }
        if (requireVulkan14 && VK_API_VERSION_MAJOR(properties.apiVersion) < 1) {
            return false;
        }

        QueueFamilyIndices indices = findQueueFamilies(device, m_surface);
        if (!indices.complete()) {
            return false;
        }

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

        if (!hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            return false;
        }

        const bool meshShaderAvailable = hasExtension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
        const bool dynamicRenderingAvailable = hasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) ||
                                               properties.apiVersion >= VK_API_VERSION_1_3;
        const bool sync2Available = hasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) ||
                                    properties.apiVersion >= VK_API_VERSION_1_3;
        if (!dynamicRenderingAvailable || !sync2Available) {
            return false;
        }

        SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device, m_surface);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
            return false;
        }

        VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceSynchronization2Features sync2Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &dynamicRenderingFeatures;
        dynamicRenderingFeatures.pNext = &sync2Features;
        sync2Features.pNext = &meshShaderFeatures;
        vkGetPhysicalDeviceFeatures2(device, &features2);

        if (dynamicRenderingFeatures.dynamicRendering != VK_TRUE || sync2Features.synchronization2 != VK_TRUE) {
            return false;
        }

        m_features.dynamicRendering = true;
        m_features.meshShaders = meshShaderAvailable && meshShaderFeatures.meshShader == VK_TRUE;
        return true;
    }

    void createLogicalDevice(bool enableValidation) {
        float queuePriority = 1.0f;
        std::set<uint32_t> uniqueQueueFamilies = {m_queueFamilies.graphics.value(), m_queueFamilies.present.value()};

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t family : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueCreateInfo.queueFamilyIndex = family;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        if (m_physicalDeviceProperties.apiVersion < VK_API_VERSION_1_3) {
            deviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        }
        if (m_features.meshShaders) {
            deviceExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        }

        VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        meshShaderFeatures.meshShader = m_features.meshShaders ? VK_TRUE : VK_FALSE;
        meshShaderFeatures.taskShader = m_features.meshShaders ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceSynchronization2Features sync2Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        sync2Features.synchronization2 = VK_TRUE;
        sync2Features.pNext = m_features.meshShaders ? &meshShaderFeatures : nullptr;

        VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
        dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
        dynamicRenderingFeatures.pNext = &sync2Features;

        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &dynamicRenderingFeatures;

        std::vector<const char*> layers;
        if (enableValidation && m_features.validation) {
            layers.push_back(kValidationLayerName);
        }

        VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
        createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        createInfo.pNext = &features2;

        checkVk(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device),
                "Failed to create Vulkan logical device");

        vkGetDeviceQueue(m_device, m_queueFamilies.graphics.value(), 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, m_queueFamilies.present.value(), 0, &m_presentQueue);
    }

    void createCommandObjects() {
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = m_queueFamilies.graphics.value();
        checkVk(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool),
                "Failed to create Vulkan command pool");

        VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = kMaxFramesInFlight;

        std::array<VkCommandBuffer, kMaxFramesInFlight> commandBuffers{};
        checkVk(vkAllocateCommandBuffers(m_device, &allocInfo, commandBuffers.data()),
                "Failed to allocate Vulkan command buffers");

        for (size_t i = 0; i < kMaxFramesInFlight; ++i) {
            m_frames[i].commandBuffer = commandBuffers[i];

            VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            checkVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_frames[i].imageAvailable),
                    "Failed to create Vulkan image-available semaphore");
            checkVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_frames[i].renderFinished),
                    "Failed to create Vulkan render-finished semaphore");
            checkVk(vkCreateFence(m_device, &fenceInfo, nullptr, &m_frames[i].inFlight),
                    "Failed to create Vulkan fence");
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 11> poolSizes = {{
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1024},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1024},
        }};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1024 * static_cast<uint32_t>(poolSizes.size());
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        checkVk(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool),
                "Failed to create Vulkan descriptor pool");
    }

    void recreateSwapchain() {
        if (m_requestedWidth == 0 || m_requestedHeight == 0) {
            return;
        }

        waitIdle();
        cleanupSwapchain();

        const SwapchainSupportDetails support = querySwapchainSupport(m_physicalDevice, m_surface);
        if (support.formats.empty() || support.presentModes.empty()) {
            throw std::runtime_error("Swapchain support disappeared while recreating the Vulkan swapchain.");
        }

        m_swapchainFormat = chooseSwapSurfaceFormat(support.formats);
        const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
        m_swapchainExtent = chooseSwapExtent(support.capabilities, m_requestedWidth, m_requestedHeight);

        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
            imageCount = support.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        createInfo.surface = m_surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = m_swapchainFormat.format;
        createInfo.imageColorSpace = m_swapchainFormat.colorSpace;
        createInfo.imageExtent = m_swapchainExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        const uint32_t queueFamilyIndices[] = {m_queueFamilies.graphics.value(), m_queueFamilies.present.value()};
        if (m_queueFamilies.graphics != m_queueFamilies.present) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        checkVk(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain),
                "Failed to create Vulkan swapchain");

        uint32_t createdImageCount = 0;
        vkGetSwapchainImagesKHR(m_device, m_swapchain, &createdImageCount, nullptr);
        m_swapchainImages.resize(createdImageCount);
        vkGetSwapchainImagesKHR(m_device, m_swapchain, &createdImageCount, m_swapchainImages.data());

        m_swapchainImageViews.resize(createdImageCount);
        m_swapchainImageLayouts.assign(createdImageCount, VK_IMAGE_LAYOUT_UNDEFINED);
        m_imagesInFlight.assign(createdImageCount, VK_NULL_HANDLE);

        for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = m_swapchainImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_swapchainFormat.format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            checkVk(vkCreateImageView(m_device, &viewInfo, nullptr, &m_swapchainImageViews[i]),
                    "Failed to create Vulkan swapchain image view");
        }

        populateNativeHandles();
    }

    void cleanupSwapchain() {
        for (VkImageView imageView : m_swapchainImageViews) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(m_device, imageView, nullptr);
            }
        }
        m_swapchainImageViews.clear();
        m_swapchainImages.clear();
        m_swapchainImageLayouts.clear();
        m_imagesInFlight.clear();

        if (m_swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
            m_swapchain = VK_NULL_HANDLE;
        }
    }

    void populateNativeHandles() {
        m_nativeHandles.instance = m_instance;
        m_nativeHandles.physicalDevice = m_physicalDevice;
        m_nativeHandles.device = m_device;
        m_nativeHandles.queue = m_graphicsQueue;
        m_nativeHandles.descriptorPool = m_descriptorPool;
        m_nativeHandles.graphicsQueueFamily = m_queueFamilies.graphics.value_or(0);
        m_nativeHandles.swapchainImageCount = static_cast<uint32_t>(m_swapchainImages.size());
        m_nativeHandles.colorFormat = static_cast<uint32_t>(m_swapchainFormat.format);
        m_nativeHandles.apiVersion = m_physicalDeviceProperties.apiVersion;
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            const bool typeMatches = (typeFilter & (1u << i)) != 0;
            const bool propertyMatches = (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;
            if (typeMatches && propertyMatches) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find a suitable Vulkan memory type.");
    }

    void transitionCurrentImage(VkImageLayout oldLayout,
                                VkImageLayout newLayout,
                                VkPipelineStageFlags2 srcStage,
                                VkPipelineStageFlags2 dstStage,
                                VkAccessFlags2 srcAccess,
                                VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_swapchainImages[m_imageIndex];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependencyInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(m_frames[m_frameIndex].commandBuffer, &dependencyInfo);
        m_swapchainImageLayouts[m_imageIndex] = newLayout;
    }

    VkInstance m_instance = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties m_physicalDeviceProperties{};
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;
    QueueFamilyIndices m_queueFamilies;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::array<FrameResources, kMaxFramesInFlight> m_frames{};

    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkSurfaceFormatKHR m_swapchainFormat{};
    VkExtent2D m_swapchainExtent{};
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkImageLayout> m_swapchainImageLayouts;
    std::vector<VkFence> m_imagesInFlight;

    uint32_t m_frameIndex = 0;
    uint32_t m_imageIndex = 0;
    uint32_t m_requestedWidth = 0;
    uint32_t m_requestedHeight = 0;
    bool m_pendingResize = false;
    bool m_insideRendering = false;

    RhiFeatures m_features{};
    RhiDeviceInfo m_deviceInfo{};
    RhiNativeHandles m_nativeHandles{};
    CommandContext m_commandContext;
};

} // namespace

std::unique_ptr<RhiContext> createVulkanContext(const RhiCreateInfo& createInfo,
                                                std::string& errorMessage) {
    try {
        return std::make_unique<VulkanContext>(createInfo);
    } catch (const std::exception& e) {
        errorMessage = e.what();
        return {};
    }
}
