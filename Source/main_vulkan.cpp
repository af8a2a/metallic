#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <string>

#include "frame_context.h"
#include "frame_graph.h"
#include "pipeline_asset.h"
#include "pipeline_builder.h"
#include "render_pass.h"
#include "rhi_backend.h"
#include "rhi_resource_utils.h"
#include "rhi_shader_utils.h"
#include "slang_compiler.h"
#include "vulkan_backend.h"
#include "vulkan_frame_graph.h"

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>

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

PipelineAsset makePresentPipelineAsset() {
    PipelineAsset asset;
    asset.name = "VulkanPresentPipeline";
    asset.resources.push_back({"sceneColor", "texture", "RGBA16Float", "screen"});
    asset.passes.push_back({
        "Output",
        "OutputPass",
        {"sceneColor"},
        {"$backbuffer"},
        true,
        true,
        {},
    });
    return asset;
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

    const auto triangleSpirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                          "Shaders/Vertex/triangle",
                                                          PROJECT_SOURCE_DIR);
    if (triangleSpirv.empty()) {
        spdlog::error("Failed to compile SPIR-V for triangle shader");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const auto passthroughSpirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                             "Shaders/Post/passthrough",
                                                             PROJECT_SOURCE_DIR);
    if (passthroughSpirv.empty()) {
        spdlog::error("Failed to compile SPIR-V for passthrough shader");
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
    vulkanLoadMeshShaderFunctions(vkDevice);

    RhiBufferHandle vertexBuffer =
        rhiCreateSharedBuffer(deviceHandle, vertices.data(), sizeof(vertices), "TriangleVB");
    if (!vertexBuffer.nativeHandle()) {
        spdlog::error("Failed to create shared vertex buffer for Vulkan RenderGraph test");
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

    std::string triangleShaderSource(reinterpret_cast<const char*>(triangleSpirv.data()),
                                     triangleSpirv.size() * sizeof(uint32_t));
    RhiRenderPipelineSourceDesc trianglePipelineDesc;
    trianglePipelineDesc.vertexEntry = "vertexMain";
    trianglePipelineDesc.fragmentEntry = "fragmentMain";
    trianglePipelineDesc.colorFormat = RhiFormat::RGBA16Float;
    trianglePipelineDesc.vertexDescriptor = &vertexDescriptor;

    std::string trianglePipelineError;
    RhiGraphicsPipelineHandle trianglePipeline =
        rhiCreateRenderPipelineFromSource(deviceHandle,
                                          triangleShaderSource,
                                          trianglePipelineDesc,
                                          trianglePipelineError);
    if (!trianglePipeline.nativeHandle()) {
        spdlog::error("Failed to create triangle pipeline: {}", trianglePipelineError);
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::string passthroughShaderSource(reinterpret_cast<const char*>(passthroughSpirv.data()),
                                        passthroughSpirv.size() * sizeof(uint32_t));
    RhiRenderPipelineSourceDesc passthroughPipelineDesc;
    passthroughPipelineDesc.vertexEntry = "vertexMain";
    passthroughPipelineDesc.fragmentEntry = "fragmentMain";
    passthroughPipelineDesc.colorFormat = rhi->colorFormat();

    std::string passthroughPipelineError;
    RhiGraphicsPipelineHandle passthroughPipeline =
        rhiCreateRenderPipelineFromSource(deviceHandle,
                                          passthroughShaderSource,
                                          passthroughPipelineDesc,
                                          passthroughPipelineError);
    if (!passthroughPipeline.nativeHandle()) {
        spdlog::error("Failed to create passthrough pipeline: {}", passthroughPipelineError);
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.addressModeS = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeT = RhiSamplerAddressMode::ClampToEdge;
    RhiSamplerHandle linearSampler = rhiCreateSampler(deviceHandle, samplerDesc);
    if (!linearSampler.nativeHandle()) {
        spdlog::error("Failed to create Vulkan sampler");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    VulkanFrameGraphBackend frameGraphBackend(vkDevice, vkPhysicalDevice, vmaAllocator);
    VulkanDescriptorManager descriptorManager;
    descriptorManager.init(vkDevice, vmaAllocator);
    VulkanImageLayoutTracker imageTracker;
    VulkanImportedTexture backbufferTexture;

    RhiTextureHandle sceneColorTexture;
    auto recreateSceneColorTexture = [&](uint32_t targetWidth, uint32_t targetHeight) {
        rhiReleaseHandle(sceneColorTexture);
        sceneColorTexture = rhiCreateTexture2D(deviceHandle,
                                               targetWidth,
                                               targetHeight,
                                               RhiFormat::RGBA16Float,
                                               false,
                                               1,
                                               RhiTextureStorageMode::Private,
                                               RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead);
        return sceneColorTexture.nativeHandle() != nullptr;
    };

    if (!recreateSceneColorTexture(createInfo.width, createInfo.height)) {
        spdlog::error("Failed to create offscreen scene color texture");
        rhi->waitIdle();
        descriptorManager.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    struct TrianglePassData {
        FGResource colorTarget;
    };

    FrameGraph sceneGraph;
    FGResource sceneColorRes = sceneGraph.import("sceneColor", &sceneColorTexture);
    const RhiGraphicsPipeline* trianglePipelinePtr = &trianglePipeline;
    RhiBuffer* triangleBufferPtr = &vertexBuffer;
    sceneGraph.addRenderPass<TrianglePassData>(
        "Triangle Pass",
        [sceneColorRes](FGBuilder& builder, TrianglePassData& data) {
            data.colorTarget = sceneColorRes;
            builder.setColorAttachment(0,
                                       data.colorTarget,
                                       RhiLoadAction::Clear,
                                       RhiStoreAction::Store,
                                       RhiClearColor(0.08, 0.09, 0.12, 1.0));
        },
        [trianglePipelinePtr, triangleBufferPtr](const TrianglePassData&, RhiRenderCommandEncoder& encoder) {
            encoder.setRenderPipeline(*trianglePipelinePtr);
            encoder.setFrontFacingWinding(RhiWinding::CounterClockwise);
            encoder.setCullMode(RhiCullMode::None);
            encoder.setVertexBuffer(triangleBufferPtr, 0, 0);
            encoder.drawPrimitives(RhiPrimitiveType::Triangle, 0, 3);
        });
    sceneGraph.compile();

    LoadedMesh emptyMesh;
    MeshletData emptyMeshlets;
    LoadedMaterials emptyMaterials;
    SceneGraph emptyScene;
    RaytracedShadowResources emptyShadows;
    RenderContext renderContext{
        emptyMesh,
        emptyMeshlets,
        emptyMaterials,
        emptyScene,
        emptyShadows,
        {},
        {},
        {},
        1.0,
    };

    PipelineRuntimeContext runtimeContext;
    runtimeContext.renderPipelinesRhi["OutputPass"] = passthroughPipeline;
    runtimeContext.samplersRhi["tonemap"] = linearSampler;
    runtimeContext.importedTexturesRhi["sceneColor"] = sceneColorTexture;
    runtimeContext.backbufferRhi = &backbufferTexture;
    runtimeContext.resourceFactory = &frameGraphBackend;

    const PipelineAsset presentAsset = makePresentPipelineAsset();
    PipelineBuilder presentBuilder(renderContext);
    auto rebuildPresentBuilder = [&](int targetWidth, int targetHeight) {
        runtimeContext.importedTexturesRhi["sceneColor"] = sceneColorTexture;
        if (!presentBuilder.build(presentAsset, runtimeContext, targetWidth, targetHeight)) {
            spdlog::error("Failed to build Vulkan present pipeline: {}", presentBuilder.lastError());
            return false;
        }
        presentBuilder.compile();
        return true;
    };

    if (!rebuildPresentBuilder(width, height)) {
        rhi->waitIdle();
        descriptorManager.destroy();
        rhiReleaseHandle(linearSampler);
        rhiReleaseHandle(passthroughPipeline);
        rhiReleaseHandle(trianglePipeline);
        rhiReleaseHandle(sceneColorTexture);
        rhiReleaseHandle(vertexDescriptor);
        rhiReleaseHandle(vertexBuffer);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    VkImageLayout sceneColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    FrameContext frameContext;

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
            if (!recreateSceneColorTexture(static_cast<uint32_t>(width), static_cast<uint32_t>(height))) {
                spdlog::error("Failed to recreate offscreen scene color texture");
                break;
            }
            if (!rebuildPresentBuilder(width, height)) {
                break;
            }
            sceneColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
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

        descriptorManager.resetFrame();
        imageTracker.clear();
        if (backbufferImage != VK_NULL_HANDLE) {
            imageTracker.setLayout(backbufferImage, getVulkanCurrentBackbufferLayout(*rhi));
        }
        if (sceneColorTexture.nativeHandle()) {
            imageTracker.setLayout(getVulkanImage(&sceneColorTexture), sceneColorLayout);
        }

        VulkanCommandBuffer commandBuffer(getVulkanCurrentCommandBuffer(*rhi),
                                          vkDevice,
                                          &descriptorManager,
                                          &imageTracker);

        sceneGraph.execute(commandBuffer, frameGraphBackend);

        RhiNativeCommandBufferHandle nativeCommandBuffer(getVulkanCurrentCommandBuffer(*rhi));
        frameContext.width = width;
        frameContext.height = height;
        frameContext.commandBuffer = &nativeCommandBuffer;

        runtimeContext.importedTexturesRhi["sceneColor"] = sceneColorTexture;
        runtimeContext.backbufferRhi = &backbufferTexture;
        presentBuilder.updateFrame(&backbufferTexture, &frameContext);
        presentBuilder.execute(commandBuffer, frameGraphBackend);

        sceneColorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        rhi->endFrame();
        FrameMark;
    }

    rhi->waitIdle();
    descriptorManager.destroy();
    rhiReleaseHandle(linearSampler);
    rhiReleaseHandle(passthroughPipeline);
    rhiReleaseHandle(trianglePipeline);
    rhiReleaseHandle(sceneColorTexture);
    rhiReleaseHandle(vertexDescriptor);
    rhiReleaseHandle(vertexBuffer);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
