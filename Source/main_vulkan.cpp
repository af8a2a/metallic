#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include "rhi_backend.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "slang_compiler.h"

#include <spdlog/spdlog.h>

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

void checkVkResult(VkResult err) {
    if (err == VK_SUCCESS) {
        return;
    }
    spdlog::error("ImGui Vulkan error: {}", static_cast<int>(err));
}

RhiBackendType parseBackend(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend=metal") {
            return RhiBackendType::Metal;
        }
        if (arg == "--backend=vulkan") {
            return RhiBackendType::Vulkan;
        }
    }
    return RhiBackendType::Vulkan;
}

} // namespace

int main(int argc, char** argv) {
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Metallic - Vulkan", nullptr, nullptr);
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

    std::string backendError;
    RhiCreateInfo createInfo;
    createInfo.window = window;
    createInfo.width = static_cast<uint32_t>(width);
    createInfo.height = static_cast<uint32_t>(height);
    createInfo.applicationName = "Metallic";
    createInfo.enableValidation = true;
    createInfo.requireVulkan14 = true;

    auto backend = parseBackend(argc, argv);
    auto rhi = createRhiContext(backend, createInfo, backendError);
    if (!rhi) {
        spdlog::error("Failed to create RHI backend: {}", backendError);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const auto spirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                  "Shaders/Vertex/triangle",
                                                  PROJECT_SOURCE_DIR);
    if (spirv.empty()) {
        spdlog::error("Failed to compile SPIR-V for triangle shader");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    auto shaderModule = rhi->createShaderModule({spirv, "Triangle"});

    const std::array<Vertex, 3> vertices = {{
        {{0.0f, -0.6f, 0.0f}, {1.0f, 0.3f, 0.2f}},
        {{0.6f, 0.6f, 0.0f}, {0.2f, 0.9f, 0.4f}},
        {{-0.6f, 0.6f, 0.0f}, {0.2f, 0.5f, 1.0f}},
    }};

    auto vertexBuffer = rhi->createVertexBuffer({sizeof(vertices), vertices.data(), true, "TriangleVB"});

    RhiGraphicsPipelineDesc pipelineDesc;
    pipelineDesc.shaderModule = shaderModule.get();
    pipelineDesc.vertexEntry = "vertexMain";
    pipelineDesc.fragmentEntry = "fragmentMain";
    pipelineDesc.colorFormat = rhi->colorFormat();
    pipelineDesc.bindings = {{0, sizeof(Vertex)}};
    pipelineDesc.attributes = {
        {0, 0, RhiVertexFormat::Float3, static_cast<uint32_t>(offsetof(Vertex, position))},
        {1, 0, RhiVertexFormat::Float3, static_cast<uint32_t>(offsetof(Vertex, color))},
    };

    auto pipeline = rhi->createGraphicsPipeline(pipelineDesc);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);

    const RhiNativeHandles& native = rhi->nativeHandles();
    VkFormat colorAttachmentFormat = static_cast<VkFormat>(native.colorFormat);
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = native.apiVersion;
    initInfo.Instance = static_cast<VkInstance>(native.instance);
    initInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(native.physicalDevice);
    initInfo.Device = static_cast<VkDevice>(native.device);
    initInfo.QueueFamily = native.graphicsQueueFamily;
    initInfo.Queue = static_cast<VkQueue>(native.queue);
    initInfo.DescriptorPool = static_cast<VkDescriptorPool>(native.descriptorPool);
    initInfo.MinImageCount = native.swapchainImageCount;
    initInfo.ImageCount = native.swapchainImageCount;
    initInfo.UseDynamicRendering = true;
    initInfo.CheckVkResultFn = checkVkResult;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorAttachmentFormat;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        spdlog::error("Failed to initialize ImGui Vulkan backend");
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    while (!glfwWindowShouldClose(window)) {
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

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("RHI Status");
        ImGui::Text("Backend: Vulkan 1.4");
        ImGui::Text("Adapter: %s", rhi->deviceInfo().adapterName.c_str());
        ImGui::Text("Driver: %s", rhi->deviceInfo().driverName.c_str());
        ImGui::Text("API: %u.%u.%u",
                    VK_API_VERSION_MAJOR(rhi->deviceInfo().apiVersion),
                    VK_API_VERSION_MINOR(rhi->deviceInfo().apiVersion),
                    VK_API_VERSION_PATCH(rhi->deviceInfo().apiVersion));
        ImGui::Separator();
        ImGui::Text("Dynamic Rendering: %s", rhi->features().dynamicRendering ? "Yes" : "No");
        ImGui::Text("Mesh Shaders: %s", rhi->features().meshShaders ? "Yes" : "No");
        ImGui::Text("Ray Tracing: %s", rhi->features().rayTracing ? "Yes" : "No (Metal-only currently)");
        ImGui::Separator();
        ImGui::TextWrapped("This Windows path boots the new RHI/Vulkan backend and draws a Slang-compiled triangle."
                           " The existing render graph and pass stack remain on the Metal path for now.");
        ImGui::End();

        ImGui::Render();

        auto& cmd = rhi->commandContext();
        RhiRenderTargetInfo targetInfo{};
        targetInfo.clearColor[0] = 0.08f;
        targetInfo.clearColor[1] = 0.09f;
        targetInfo.clearColor[2] = 0.12f;
        targetInfo.clearColor[3] = 1.0f;
        targetInfo.clear = true;
        cmd.beginRendering(targetInfo);
        cmd.setViewport(static_cast<float>(rhi->drawableWidth()), static_cast<float>(rhi->drawableHeight()));
        cmd.setScissor(rhi->drawableWidth(), rhi->drawableHeight());
        cmd.bindGraphicsPipeline(*pipeline);
        cmd.bindVertexBuffer(*vertexBuffer);
        cmd.draw(3);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), static_cast<VkCommandBuffer>(cmd.nativeCommandBuffer()));
        cmd.endRendering();

        rhi->endFrame();
    }

    rhi->waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
