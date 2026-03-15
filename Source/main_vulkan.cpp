#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include "frame_graph.h"
#include "render_pass.h"
#include "rhi_backend.h"
#include "rhi_resource_utils.h"
#include "rhi_shader_utils.h"
#include "slang_compiler.h"
#include "vulkan_backend.h"
#include "vulkan_frame_graph.h"

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

#include <array>
#include <cstddef>
#include <string>

namespace {

struct Vertex {
    float position[3];
    float color[3];
};

struct AppState {
    bool framebufferResized = false;
};

void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state && width > 0 && height > 0) {
        state->framebufferResized = true;
    }
}

} // namespace

int main() {
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Metallic - Vulkan RenderGraph", nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return 1;
    }

    AppState appState{};
    glfwSetWindowUserPointer(window, &appState);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    RhiCreateInfo createInfo;
    createInfo.window = window;
    createInfo.width = static_cast<uint32_t>(width);
    createInfo.height = static_cast<uint32_t>(height);
    createInfo.applicationName = "Metallic";
    createInfo.enableValidation = true;
    createInfo.requireVulkan14 = true;

    std::string backendError;
    auto rhi = createRhiContext(RhiBackendType::Vulkan, createInfo, backendError);
    if (!rhi) {
        spdlog::error("Failed to create Vulkan backend: {}", backendError);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const auto spirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                  "Shaders/Vertex/triangle",
                                                  PROJECT_SOURCE_DIR);
    if (spirv.empty()) {
        spdlog::error("Failed to compile SPIR-V for triangle shader");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const std::array<Vertex, 3> vertices = {{
        {{0.0f, -0.6f, 0.0f}, {1.0f, 0.3f, 0.2f}},
        {{0.6f, 0.6f, 0.0f}, {0.2f, 0.9f, 0.4f}},
        {{-0.6f, 0.6f, 0.0f}, {0.2f, 0.5f, 1.0f}},
    }};

    const RhiNativeHandles& native = rhi->nativeHandles();
    VkDevice vkDevice = static_cast<VkDevice>(native.device);
    VkPhysicalDevice vkPhysicalDevice = static_cast<VkPhysicalDevice>(native.physicalDevice);
    VmaAllocator vmaAllocator = getVulkanAllocator(*rhi);
    RhiDeviceHandle deviceHandle(native.device);

    vulkanSetResourceContext(vkDevice,
                             vkPhysicalDevice,
                             vmaAllocator,
                             static_cast<VkQueue>(native.queue),
                             native.graphicsQueueFamily);
    vulkanSetShaderContext(vkDevice);

    RhiBufferHandle vertexBuffer =
        rhiCreateSharedBuffer(deviceHandle, vertices.data(), sizeof(vertices), "TriangleVB");
    if (!vertexBuffer.nativeHandle()) {
        spdlog::error("Failed to create shared vertex buffer for RenderGraph test");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    RhiVertexDescriptorHandle vertexDescriptor = rhiCreateVertexDescriptor();
    rhiVertexDescriptorSetAttribute(vertexDescriptor,
                                    0,
                                    RhiVertexFormat::Float3,
                                    static_cast<uint32_t>(offsetof(Vertex, position)),
                                    0);
    rhiVertexDescriptorSetAttribute(vertexDescriptor,
                                    1,
                                    RhiVertexFormat::Float3,
                                    static_cast<uint32_t>(offsetof(Vertex, color)),
                                    0);
    rhiVertexDescriptorSetLayout(vertexDescriptor, 0, sizeof(Vertex));

    std::string shaderSource(reinterpret_cast<const char*>(spirv.data()),
                             spirv.size() * sizeof(uint32_t));
    RhiRenderPipelineSourceDesc pipelineDesc;
    pipelineDesc.vertexEntry = "vertexMain";
    pipelineDesc.fragmentEntry = "fragmentMain";
    pipelineDesc.colorFormat = rhi->colorFormat();
    pipelineDesc.vertexDescriptor = &vertexDescriptor;

    std::string pipelineError;
    RhiGraphicsPipelineHandle pipeline =
        rhiCreateRenderPipelineFromSource(deviceHandle, shaderSource, pipelineDesc, pipelineError);
    if (!pipeline.nativeHandle()) {
        spdlog::error("Failed to create graphics pipeline for RenderGraph test: {}", pipelineError);
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    spdlog::info("Created triangle graphics pipeline");

    vulkanLoadMeshShaderFunctions(vkDevice);

    VulkanFrameGraphBackend frameGraphBackend(vkDevice, vkPhysicalDevice, vmaAllocator);
    VulkanImageLayoutTracker imageTracker;
    VulkanImportedTexture backbufferTexture;

    FrameGraph frameGraph;
    FGResource backbufferRes = frameGraph.import("backbuffer", &backbufferTexture);

    struct TrianglePassData {
        FGResource target;
    };

    const RhiGraphicsPipeline* trianglePipeline = &pipeline;
    RhiBuffer* triangleBuffer = &vertexBuffer;
    frameGraph.addRenderPass<TrianglePassData>(
        "Triangle Pass",
        [backbufferRes](FGBuilder& builder, TrianglePassData& data) {
            data.target = backbufferRes;
            builder.setColorAttachment(0,
                                       data.target,
                                       RhiLoadAction::Clear,
                                       RhiStoreAction::Store,
                                       RhiClearColor(0.08, 0.09, 0.12, 1.0));
            builder.setSideEffect();
        },
        [trianglePipeline, triangleBuffer](const TrianglePassData&, RhiRenderCommandEncoder& encoder) {
            encoder.setRenderPipeline(*trianglePipeline);
            encoder.setVertexBuffer(triangleBuffer, 0, 0);
            encoder.drawPrimitives(RhiPrimitiveType::Triangle, 0, 3);
        });
    frameGraph.compile();

    while (!glfwWindowShouldClose(window)) {
        ZoneScopedN("VulkanRenderGraphFrame");

        glfwPollEvents();
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            glfwWaitEvents();
            continue;
        }

        if (appState.framebufferResized) {
            rhi->resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            appState.framebufferResized = false;
        }

        if (!rhi->beginFrame()) {
            continue;
        }

        VkImage backbufferImage = getVulkanCurrentBackbufferImage(*rhi);
        VkImageView backbufferImageView = getVulkanCurrentBackbufferImageView(*rhi);
        VkExtent2D backbufferExtent = getVulkanCurrentBackbufferExtent(*rhi);
        backbufferTexture.set(backbufferImage,
                              backbufferImageView,
                              backbufferExtent.width,
                              backbufferExtent.height,
                              static_cast<VkFormat>(native.colorFormat),
                              RhiTextureUsage::RenderTarget);

        imageTracker.clear();
        if (backbufferImage != VK_NULL_HANDLE) {
            imageTracker.setLayout(backbufferImage, getVulkanCurrentBackbufferLayout(*rhi));
        }

        frameGraph.resetTransients();
        VulkanCommandBuffer commandBuffer(getVulkanCurrentCommandBuffer(*rhi),
                                          vkDevice,
                                          nullptr,
                                          &imageTracker);
        frameGraph.execute(commandBuffer, frameGraphBackend);

        rhi->endFrame();
        FrameMark;
    }

    rhi->waitIdle();
    rhiReleaseHandle(pipeline);
    rhiReleaseHandle(vertexDescriptor);
    rhiReleaseHandle(vertexBuffer);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
