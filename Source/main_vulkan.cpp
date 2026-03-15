#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "camera.h"
#include "frame_context.h"
#include "frame_graph.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "pipeline_asset.h"
#include "pipeline_builder.h"
#include "render_pass.h"
#include "rhi_backend.h"
#include "rhi_resource_utils.h"
#include "rhi_shader_utils.h"
#include "shader_manager.h"
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

void checkImGuiVkResult(VkResult result) {
    if (result == VK_SUCCESS) {
        return;
    }
    spdlog::error("ImGui Vulkan error: {}", static_cast<int>(result));
}

struct AtmosphereTextures {
    RhiTextureHandle transmittance;
    RhiTextureHandle scattering;
    RhiTextureHandle irradiance;

    bool isValid() const {
        return transmittance.nativeHandle() &&
               scattering.nativeHandle() &&
               irradiance.nativeHandle();
    }

    void release() {
        rhiReleaseHandle(transmittance);
        rhiReleaseHandle(scattering);
        rhiReleaseHandle(irradiance);
    }
};

void releaseMeshBuffers(LoadedMesh& mesh) {
    rhiReleaseHandle(mesh.positionBuffer);
    rhiReleaseHandle(mesh.normalBuffer);
    rhiReleaseHandle(mesh.uvBuffer);
    rhiReleaseHandle(mesh.indexBuffer);
}

bool createPreviewScene(const RhiDevice& device, LoadedMesh& outMesh, SceneGraph& outScene) {
    static constexpr std::array<float, 24 * 3> kPositions = {
        -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    };
    static constexpr std::array<float, 24 * 3> kNormals = {
         0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
         0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,
         0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
         0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,
         1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
    };
    static constexpr std::array<uint32_t, 36> kIndices = {
         0,  1,  2,  2,  3,  0,
         4,  5,  6,  6,  7,  4,
         8,  9, 10, 10, 11,  8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16,
        20, 21, 22, 22, 23, 20,
    };

    outMesh.positionBuffer =
        rhiCreateSharedBuffer(device, kPositions.data(), sizeof(kPositions), "Preview Positions");
    outMesh.normalBuffer =
        rhiCreateSharedBuffer(device, kNormals.data(), sizeof(kNormals), "Preview Normals");
    outMesh.indexBuffer =
        rhiCreateSharedBuffer(device, kIndices.data(), sizeof(kIndices), "Preview Indices");

    if (!outMesh.positionBuffer.nativeHandle() ||
        !outMesh.normalBuffer.nativeHandle() ||
        !outMesh.indexBuffer.nativeHandle()) {
        rhiReleaseHandle(outMesh.positionBuffer);
        rhiReleaseHandle(outMesh.normalBuffer);
        rhiReleaseHandle(outMesh.indexBuffer);
        return false;
    }

    outMesh.vertexCount = 24;
    outMesh.indexCount = static_cast<uint32_t>(kIndices.size());
    outMesh.bboxMin[0] = -1.0f;
    outMesh.bboxMin[1] = -1.0f;
    outMesh.bboxMin[2] = -1.0f;
    outMesh.bboxMax[0] = 1.0f;
    outMesh.bboxMax[1] = 1.0f;
    outMesh.bboxMax[2] = 1.0f;

    outScene.nodes.clear();
    outScene.rootNodes.clear();
    outScene.selectedNode = -1;
    outScene.sunLightNode = -1;

    outScene.nodes.emplace_back();
    SceneNode& node = outScene.nodes.back();
    node.name = "PreviewCube";
    node.id = 0;
    node.parent = -1;
    node.meshIndex = 0;
    node.indexStart = 0;
    node.indexCount = outMesh.indexCount;
    node.transform.localMatrix = float4x4::Identity();
    node.transform.worldMatrix = float4x4::Identity();
    node.transform.dirty = false;
    outScene.rootNodes.push_back(0);

    return true;
}

void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state && width > 0 && height > 0) {
        state->framebufferResized = true;
    }
}

std::vector<float> loadFloatData(const std::string& path, size_t expectedCount) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::warn("Atmosphere: missing texture data {}", path);
        return {};
    }

    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    if (size == 0 || size % sizeof(float) != 0) {
        spdlog::warn("Atmosphere: invalid data size {} ({} bytes)", path, size);
        return {};
    }

    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!file) {
        spdlog::warn("Atmosphere: failed to read {}", path);
        return {};
    }

    if (expectedCount > 0 && data.size() != expectedCount) {
        spdlog::warn("Atmosphere: unexpected element count in {} ({} vs {})",
                     path,
                     data.size(),
                     expectedCount);
    }

    return data;
}

bool loadPipelineAssetChecked(const std::string& path,
                              const char* label,
                              PipelineAsset& outAsset) {
    PipelineAsset loaded = PipelineAsset::load(path);
    if (loaded.name.empty()) {
        spdlog::error("Failed to load {} pipeline from '{}'", label, path);
        return false;
    }

    std::string validationError;
    if (!loaded.validate(validationError)) {
        spdlog::error("Invalid {} pipeline '{}': {}", label, path, validationError);
        return false;
    }

    outAsset = std::move(loaded);
    spdlog::info("Loaded {} pipeline: {} ({} passes, {} resources)",
                 label,
                 outAsset.name,
                 outAsset.passes.size(),
                 outAsset.resources.size());
    return true;
}

void disablePreviewAutoExposure(PipelineAsset& asset) {
    asset.resources.erase(
        std::remove_if(asset.resources.begin(), asset.resources.end(),
                       [](const ResourceDecl& resource) {
                           return resource.name == "exposureLut";
                       }),
        asset.resources.end());

    asset.passes.erase(
        std::remove_if(asset.passes.begin(), asset.passes.end(),
                       [](const PassDecl& pass) {
                           return pass.type == "AutoExposurePass";
                       }),
        asset.passes.end());

    for (auto& pass : asset.passes) {
        if (pass.type != "TonemapPass") {
            continue;
        }

        pass.inputs.erase(
            std::remove(pass.inputs.begin(), pass.inputs.end(), "exposureLut"),
            pass.inputs.end());
        pass.config["autoExposure"] = false;
    }
}

bool loadAtmosphereTextures(const RhiDevice& device,
                            const char* projectRoot,
                            AtmosphereTextures& outTextures) {
    constexpr uint32_t kTransmittanceWidth = 256;
    constexpr uint32_t kTransmittanceHeight = 64;
    constexpr uint32_t kScatteringWidth = 256;
    constexpr uint32_t kScatteringHeight = 128;
    constexpr uint32_t kScatteringDepth = 32;
    constexpr uint32_t kIrradianceWidth = 64;
    constexpr uint32_t kIrradianceHeight = 16;

    const std::string basePath = std::string(projectRoot) + "/Asset/Atmosphere/";
    const auto transmittance = loadFloatData(basePath + "transmittance.dat",
                                             static_cast<size_t>(kTransmittanceWidth) *
                                                 kTransmittanceHeight * 4);
    const auto scattering = loadFloatData(basePath + "scattering.dat",
                                          static_cast<size_t>(kScatteringWidth) *
                                              kScatteringHeight * kScatteringDepth * 4);
    const auto irradiance = loadFloatData(basePath + "irradiance.dat",
                                          static_cast<size_t>(kIrradianceWidth) *
                                              kIrradianceHeight * 4);

    if (transmittance.empty() || scattering.empty() || irradiance.empty()) {
        return false;
    }

    outTextures.release();
    outTextures.transmittance = rhiCreateTexture2D(device,
                                                   kTransmittanceWidth,
                                                   kTransmittanceHeight,
                                                   RhiFormat::RGBA32Float,
                                                   false,
                                                   1,
                                                   RhiTextureStorageMode::Shared,
                                                   RhiTextureUsage::ShaderRead);
    outTextures.scattering = rhiCreateTexture3D(device,
                                                kScatteringWidth,
                                                kScatteringHeight,
                                                kScatteringDepth,
                                                RhiFormat::RGBA32Float,
                                                RhiTextureStorageMode::Shared,
                                                RhiTextureUsage::ShaderRead);
    outTextures.irradiance = rhiCreateTexture2D(device,
                                                kIrradianceWidth,
                                                kIrradianceHeight,
                                                RhiFormat::RGBA32Float,
                                                false,
                                                1,
                                                RhiTextureStorageMode::Shared,
                                                RhiTextureUsage::ShaderRead);

    if (!outTextures.isValid()) {
        outTextures.release();
        return false;
    }

    rhiUploadTexture2D(outTextures.transmittance,
                       kTransmittanceWidth,
                       kTransmittanceHeight,
                       transmittance.data(),
                       static_cast<size_t>(kTransmittanceWidth) * 4 * sizeof(float));
    rhiUploadTexture3D(outTextures.scattering,
                       kScatteringWidth,
                       kScatteringHeight,
                       kScatteringDepth,
                       scattering.data(),
                       static_cast<size_t>(kScatteringWidth) * 4 * sizeof(float),
                       static_cast<size_t>(kScatteringWidth) * kScatteringHeight * 4 * sizeof(float));
    rhiUploadTexture2D(outTextures.irradiance,
                       kIrradianceWidth,
                       kIrradianceHeight,
                       irradiance.data(),
                       static_cast<size_t>(kIrradianceWidth) * 4 * sizeof(float));
    return true;
}

PipelineAsset makeSceneColorPostPipelineAsset() {
    PipelineAsset asset;
    asset.name = "VulkanPostPipeline";
    asset.resources.push_back({"sceneColor", "texture", "RGBA16Float", "screen"});
    asset.passes.push_back({
        "Tonemap",
        "TonemapPass",
        {"sceneColor"},
        {"$backbuffer"},
        true,
        false,
        {
            {"method", "Clip"},
            {"exposure", 1.0},
            {"contrast", 1.0},
            {"brightness", 1.0},
            {"saturation", 1.0},
            {"vignette", 0.0},
            {"dither", false},
            {"autoExposure", false},
        },
    });
    asset.passes.push_back({
        "ImGui Overlay",
        "ImGuiOverlayPass",
        {},
        {"$backbuffer"},
        true,
        false,
        {},
    });
    return asset;
}

float4 orbitCameraWorldPosition(const OrbitCamera& camera) {
    const float cosAzimuth = std::cos(camera.azimuth);
    const float sinAzimuth = std::sin(camera.azimuth);
    const float cosElevation = std::cos(camera.elevation);
    const float sinElevation = std::sin(camera.elevation);

    return float4(camera.target.x + camera.distance * cosElevation * sinAzimuth,
                  camera.target.y + camera.distance * sinElevation,
                  camera.target.z + camera.distance * cosElevation * cosAzimuth,
                  1.0f);
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
    const uint32_t swapchainImageCount = std::max(2u, native.swapchainImageCount);
    const VkFormat swapchainFormat = static_cast<VkFormat>(native.colorFormat);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui_ImplGlfw_InitForOther(window, true);

    VkPipelineRenderingCreateInfoKHR imguiRenderingInfo{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    imguiRenderingInfo.colorAttachmentCount = 1;
    imguiRenderingInfo.pColorAttachmentFormats = &swapchainFormat;

    ImGui_ImplVulkan_InitInfo imguiInitInfo{};
    imguiInitInfo.ApiVersion = native.apiVersion;
    imguiInitInfo.Instance = static_cast<VkInstance>(native.instance);
    imguiInitInfo.PhysicalDevice = vkPhysicalDevice;
    imguiInitInfo.Device = vkDevice;
    imguiInitInfo.QueueFamily = native.graphicsQueueFamily;
    imguiInitInfo.Queue = static_cast<VkQueue>(native.queue);
    imguiInitInfo.DescriptorPool = static_cast<VkDescriptorPool>(native.descriptorPool);
    imguiInitInfo.MinImageCount = swapchainImageCount;
    imguiInitInfo.ImageCount = swapchainImageCount;
    imguiInitInfo.UseDynamicRendering = true;
    imguiInitInfo.CheckVkResultFn = checkImGuiVkResult;
    imguiInitInfo.PipelineInfoMain.Subpass = 0;
    imguiInitInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    imguiInitInfo.PipelineInfoMain.PipelineRenderingCreateInfo = imguiRenderingInfo;

    if (!ImGui_ImplVulkan_Init(&imguiInitInfo)) {
        spdlog::error("Failed to initialize Dear ImGui for Vulkan");
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    vulkanSetResourceContext(vkDevice,
                             vkPhysicalDevice,
                             vmaAllocator,
                             static_cast<VkQueue>(native.queue),
                             native.graphicsQueueFamily);
    vulkanSetShaderContext(vkDevice);
    vulkanLoadMeshShaderFunctions(vkDevice);

    ShaderManager shaderManager(deviceHandle,
                                PROJECT_SOURCE_DIR,
                                false,
                                false,
                                ShaderManagerProfile::preview());
    if (!shaderManager.buildAll()) {
        spdlog::error("Failed to build Vulkan preview shader set");
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    PipelineRuntimeContext& runtimeContext = shaderManager.runtimeContext();
    auto hasRenderPipeline = [&](const char* name) {
        auto it = runtimeContext.renderPipelinesRhi.find(name);
        return it != runtimeContext.renderPipelinesRhi.end() && it->second.nativeHandle() != nullptr;
    };
    auto hasComputePipeline = [&](const char* name) {
        auto it = runtimeContext.computePipelinesRhi.find(name);
        return it != runtimeContext.computePipelinesRhi.end() && it->second.nativeHandle() != nullptr;
    };
    auto importRuntimeTexture = [&](const char* name, const RhiTexture& texture) {
        shaderManager.importTexture(name, texture);
    };

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

    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.addressModeS = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeT = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeR = RhiSamplerAddressMode::ClampToEdge;
    RhiSamplerHandle linearSampler = rhiCreateSampler(deviceHandle, samplerDesc);
    if (!linearSampler.nativeHandle()) {
        spdlog::error("Failed to create Vulkan sampler");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    shaderManager.importSampler("atmosphere", linearSampler);

    AtmosphereTextures atmosphereTextures;
    if (!loadAtmosphereTextures(deviceHandle, PROJECT_SOURCE_DIR, atmosphereTextures)) {
        spdlog::warn("Failed to load atmosphere textures; SkyPass will clear to black");
    } else {
        importRuntimeTexture("transmittance", atmosphereTextures.transmittance);
        importRuntimeTexture("scattering", atmosphereTextures.scattering);
        importRuntimeTexture("irradiance", atmosphereTextures.irradiance);
    }

    LoadedMesh previewMesh;
    SceneGraph previewScene;
    if (!createPreviewScene(deviceHandle, previewMesh, previewScene)) {
        spdlog::warn("Failed to create preview mesh; falling back to triangle path");
    }

    const std::string previewPipelinePath = std::string(PROJECT_SOURCE_DIR) + "/Pipelines/vulkan_preview.json";
    PipelineAsset previewPipelineBaseAsset;
    PipelineAsset previewPipelineAsset;
    const bool previewPipelineBaseLoaded =
        loadPipelineAssetChecked(previewPipelinePath, "Vulkan preview", previewPipelineBaseAsset);
    bool previewPipelineAssetLoaded = false;
    bool previewAutoExposureAvailable = false;
    bool usePreviewRenderGraph = false;
    bool atmosphereSkyAvailable = false;

    auto refreshPreviewPipelineState = [&]() {
        previewPipelineAssetLoaded = previewPipelineBaseLoaded;
        previewPipelineAsset = previewPipelineBaseLoaded ? previewPipelineBaseAsset : PipelineAsset{};
        previewAutoExposureAvailable =
            hasComputePipeline("HistogramPass") && hasComputePipeline("AutoExposurePass");
        if (previewPipelineAssetLoaded && !previewAutoExposureAvailable) {
            disablePreviewAutoExposure(previewPipelineAsset);
            std::string validationError;
            previewPipelineAssetLoaded = previewPipelineAsset.validate(validationError);
            if (!previewPipelineAssetLoaded) {
                spdlog::error("Invalid downgraded Vulkan preview pipeline: {}", validationError);
            } else {
                spdlog::warn("Vulkan preview auto-exposure unavailable; falling back to manual exposure");
            }
        }

        usePreviewRenderGraph =
            previewPipelineAssetLoaded &&
            hasRenderPipeline("ForwardPass") &&
            hasRenderPipeline("TonemapPass") &&
            previewMesh.positionBuffer.nativeHandle() &&
            previewMesh.normalBuffer.nativeHandle() &&
            previewMesh.indexBuffer.nativeHandle();
        atmosphereSkyAvailable = hasRenderPipeline("SkyPass") && atmosphereTextures.isValid();
    };

    auto logPreviewMode = [&]() {
        if (usePreviewRenderGraph) {
            if (previewAutoExposureAvailable) {
                spdlog::info("Vulkan RenderGraph mode: SkyPass -> ForwardPass -> AutoExposurePass -> TonemapPass");
            } else {
                spdlog::info("Vulkan RenderGraph mode: SkyPass -> ForwardPass -> TonemapPass");
            }
        } else {
            spdlog::info("Vulkan RenderGraph mode: Triangle -> TonemapPass fallback");
        }
    };

    refreshPreviewPipelineState();
    logPreviewMode();

    VulkanFrameGraphBackend frameGraphBackend(vkDevice, vkPhysicalDevice, vmaAllocator);
    VulkanDescriptorManager descriptorManager;
    descriptorManager.init(vkDevice, vmaAllocator);
    VulkanImageLayoutTracker imageTracker;
    VulkanImportedTexture backbufferTexture;

    RhiTextureHandle sceneColorTexture;
    auto recreateSceneColorTexture = [&](uint32_t targetWidth, uint32_t targetHeight) {
        rhiReleaseHandle(sceneColorTexture);
        runtimeContext.importedTexturesRhi.erase("sceneColor");
        sceneColorTexture = rhiCreateTexture2D(deviceHandle,
                                               targetWidth,
                                               targetHeight,
                                               RhiFormat::RGBA16Float,
                                               false,
                                               1,
                                               RhiTextureStorageMode::Private,
                                               RhiTextureUsage::RenderTarget | RhiTextureUsage::ShaderRead);
        if (!sceneColorTexture.nativeHandle()) {
            return false;
        }
        importRuntimeTexture("sceneColor", sceneColorTexture);
        return true;
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
    TrianglePassData& trianglePassData = sceneGraph.addRenderPass<TrianglePassData>(
        "Triangle Pass",
        [sceneColorRes](FGBuilder& builder, TrianglePassData& data) {
            data.colorTarget = builder.setColorAttachment(0,
                                                          sceneColorRes,
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
    sceneGraph.exportResource(trianglePassData.colorTarget);
    sceneGraph.compile();

    MeshletData emptyMeshlets;
    LoadedMaterials previewMaterials;
    RaytracedShadowResources emptyShadows;
    const double depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;
    RhiDepthStencilStateHandle depthState =
        rhiCreateDepthStencilState(deviceHandle, true, ML_DEPTH_REVERSED);
    RenderContext renderContext{
        previewMesh,
        emptyMeshlets,
        previewMaterials,
        previewScene,
        emptyShadows,
        depthState,
        {},
        {},
        depthClearValue,
    };

    runtimeContext.backbufferRhi = &backbufferTexture;
    runtimeContext.resourceFactory = &frameGraphBackend;

    const PipelineAsset sceneColorPostAsset = makeSceneColorPostPipelineAsset();
    PipelineBuilder postBuilder(renderContext);
    auto rebuildPostBuilder = [&](int targetWidth, int targetHeight) {
        const PipelineAsset& activePostAsset =
            usePreviewRenderGraph ? previewPipelineAsset : sceneColorPostAsset;
        if (!postBuilder.build(activePostAsset, runtimeContext, targetWidth, targetHeight)) {
            spdlog::error("Failed to build Vulkan post pipeline: {}", postBuilder.lastError());
            return false;
        }
        postBuilder.compile();
        return true;
    };

    auto cleanupRuntimeResources = [&]() {
        rhi->waitIdle();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        postBuilder.frameGraph().reset();
        sceneGraph.reset();
        descriptorManager.destroy();
        atmosphereTextures.release();
        releaseMeshBuffers(previewMesh);
        rhiReleaseHandle(depthState);
        rhiReleaseHandle(linearSampler);
        rhiReleaseHandle(trianglePipeline);
        rhiReleaseHandle(sceneColorTexture);
        rhiReleaseHandle(vertexDescriptor);
        rhiReleaseHandle(vertexBuffer);
    };

    if (!rebuildPostBuilder(width, height)) {
        cleanupRuntimeResources();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    VkImageLayout sceneColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    FrameContext frameContext;
    OrbitCamera previewCamera;
    previewCamera.initFromBounds(previewMesh.bboxMin, previewMesh.bboxMax);
    previewCamera.distance *= 0.8f;
    previewCamera.azimuth = 0.55f;
    previewCamera.elevation = 0.35f;
    previewCamera.nearZ = 0.1f;
    previewCamera.farZ = 100.0f;
    const float3 sunDirection = normalize(float3(0.35f, 0.85f, 0.25f));
    const std::vector<uint32_t> previewVisibleIndexNodes = {0};
    bool showGraphDebug = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool reloadKeyDown = false;
    bool shaderReloadRequested = false;
    bool postBuilderNeedsRebuild = false;

    auto rebuildActivePipeline = [&](int targetWidth, int targetHeight) {
        if (!usePreviewRenderGraph &&
            (!sceneColorTexture.nativeHandle() ||
             sceneColorTexture.width() != static_cast<uint32_t>(targetWidth) ||
             sceneColorTexture.height() != static_cast<uint32_t>(targetHeight))) {
            if (!recreateSceneColorTexture(static_cast<uint32_t>(targetWidth),
                                           static_cast<uint32_t>(targetHeight))) {
                spdlog::error("Failed to recreate offscreen scene color texture");
                return false;
            }
        }
        sceneColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!rebuildPostBuilder(targetWidth, targetHeight)) {
            return false;
        }
        return true;
    };

    while (!glfwWindowShouldClose(window)) {
        ZoneScopedN("VulkanRenderGraphFrame");

        glfwPollEvents();
        const bool f5Down = glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
        if (f5Down && !reloadKeyDown) {
            shaderReloadRequested = true;
        }
        reloadKeyDown = f5Down;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            glfwWaitEvents();
            continue;
        }

        if (appState.framebufferResized) {
            rhi->resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            ImGui_ImplVulkan_SetMinImageCount(std::max(2u, rhi->nativeHandles().swapchainImageCount));
            postBuilderNeedsRebuild = true;
            appState.framebufferResized = false;
        }

        if (shaderReloadRequested) {
            shaderReloadRequested = false;
            spdlog::info("Reloading Vulkan preview shaders...");
            const bool previousPreviewRenderGraph = usePreviewRenderGraph;
            const bool previousAutoExposure = previewAutoExposureAvailable;
            const bool previousAtmosphereSky = atmosphereSkyAvailable;
            auto [reloaded, failed] = shaderManager.reloadAll();
            refreshPreviewPipelineState();

            if (failed == 0) {
                spdlog::info("All {} Vulkan preview shaders reloaded successfully", reloaded);
            } else {
                spdlog::warn("{} Vulkan preview shaders reloaded, {} failed (keeping old pipelines)",
                             reloaded,
                             failed);
            }

            if (previousPreviewRenderGraph != usePreviewRenderGraph ||
                previousAutoExposure != previewAutoExposureAvailable ||
                previousAtmosphereSky != atmosphereSkyAvailable) {
                logPreviewMode();
            }
            postBuilderNeedsRebuild = true;
        }

        if (postBuilderNeedsRebuild || postBuilder.needsRebuild(width, height)) {
            if (!rebuildActivePipeline(width, height)) {
                break;
            }
            postBuilderNeedsRebuild = false;
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
        if (!usePreviewRenderGraph && sceneColorTexture.nativeHandle()) {
            imageTracker.setLayout(getVulkanImage(&sceneColorTexture), sceneColorLayout);
        }

        VulkanCommandBuffer commandBuffer(getVulkanCurrentCommandBuffer(*rhi),
                                          vkDevice,
                                          &descriptorManager,
                                          &imageTracker);

        if (!usePreviewRenderGraph) {
            sceneGraph.execute(commandBuffer, frameGraphBackend);
        }

        RhiNativeCommandBufferHandle nativeCommandBuffer(getVulkanCurrentCommandBuffer(*rhi));
        frameContext.width = width;
        frameContext.height = height;
        frameContext.view = previewCamera.viewMatrix();
        frameContext.proj =
            previewCamera.projectionMatrix(static_cast<float>(width) / static_cast<float>(std::max(height, 1)));
        frameContext.cameraWorldPos = orbitCameraWorldPosition(previewCamera);
        frameContext.prevView = frameContext.view;
        frameContext.prevProj = frameContext.proj;
        frameContext.worldLightDir = float4(sunDirection.x, sunDirection.y, sunDirection.z, 0.0f);
        frameContext.viewLightDir = frameContext.view * frameContext.worldLightDir;
        frameContext.lightColorIntensity = float4(1.0f, 0.98f, 0.95f, 2.5f);
        frameContext.visibleIndexNodes = previewVisibleIndexNodes;
        frameContext.renderMode = 0;
        frameContext.commandBuffer = &nativeCommandBuffer;
        frameContext.cameraFarZ = previewCamera.farZ;
        frameContext.depthClearValue = depthClearValue;
        frameContext.enableAtmosphereSky = atmosphereSkyAvailable;

        postBuilder.updateFrame(&backbufferTexture, &frameContext);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Vulkan Preview");
        ImGui::Text("Resolution: %d x %d", width, height);
        ImGui::TextUnformatted(usePreviewRenderGraph ? "Pipeline: Preview" : "Pipeline: Triangle Fallback");
        ImGui::TextUnformatted(previewAutoExposureAvailable ? "Exposure: Auto" : "Exposure: Manual");
        if (ImGui::Button("Reload Shaders (F5)")) {
            shaderReloadRequested = true;
        }
        ImGui::Checkbox("FrameGraph Debug", &showGraphDebug);
        ImGui::Checkbox("Render Pass UI", &showRenderPassUI);
        ImGui::Checkbox("ImGui Demo", &showImGuiDemo);
        ImGui::End();

        FrameGraph& activeFg = postBuilder.frameGraph();
        if (showGraphDebug) {
            activeFg.debugImGui();
        }
        if (showRenderPassUI) {
            activeFg.renderPassUI();
        }
        if (showImGuiDemo) {
            ImGui::ShowDemoWindow(&showImGuiDemo);
        }
        ImGui::Render();

        postBuilder.execute(commandBuffer, frameGraphBackend);

        if (!usePreviewRenderGraph) {
            sceneColorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        rhi->endFrame();
        FrameMark;
    }

    cleanupRuntimeResources();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
