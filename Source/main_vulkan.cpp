#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "camera.h"
#include "frame_context.h"
#include "frame_graph.h"
#include "input.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "pipeline_asset.h"
#include "pipeline_builder.h"
#include "render_pass.h"
#include "render_uniforms.h"
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
    InputState input;
    bool framebufferResized = false;
};

static_assert(std::is_standard_layout_v<AppState>);
static_assert(offsetof(AppState, input) == 0);

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

void releaseMeshletBuffers(MeshletData& meshlets) {
    rhiReleaseHandle(meshlets.meshletBuffer);
    rhiReleaseHandle(meshlets.meshletVertices);
    rhiReleaseHandle(meshlets.meshletTriangles);
    rhiReleaseHandle(meshlets.boundsBuffer);
    rhiReleaseHandle(meshlets.materialIDs);
}

void releaseMaterialResources(LoadedMaterials& materials) {
    for (auto& texture : materials.textures) {
        rhiReleaseHandle(texture);
    }
    materials.textures.clear();
    materials.textureViews.clear();
    rhiReleaseHandle(materials.materialBuffer);
    rhiReleaseHandle(materials.sampler);
    materials.materialCount = 0;
}

bool loadSponzaScene(const RhiDevice& device,
                     const RhiCommandQueue& commandQueue,
                     const char* projectRoot,
                     LoadedMesh& outMesh,
                     MeshletData& outMeshlets,
                     LoadedMaterials& outMaterials,
                     SceneGraph& outScene) {
    const std::string gltfPath = std::string(projectRoot) + "/Asset/Sponza/glTF/Sponza.gltf";

    releaseMaterialResources(outMaterials);
    releaseMeshletBuffers(outMeshlets);
    releaseMeshBuffers(outMesh);
    outMesh = LoadedMesh{};
    outMeshlets = MeshletData{};
    outMaterials = LoadedMaterials{};
    outScene = SceneGraph{};

    if (!loadGLTFMesh(device, gltfPath, outMesh)) {
        spdlog::error("Failed to load Vulkan scene mesh: {}", gltfPath);
        return false;
    }

    if (!buildMeshlets(device, outMesh, outMeshlets)) {
        spdlog::error("Failed to build Vulkan meshlets: {}", gltfPath);
        releaseMeshBuffers(outMesh);
        outMesh = LoadedMesh{};
        return false;
    }

    if (!loadGLTFMaterials(device, commandQueue, gltfPath, outMaterials)) {
        spdlog::error("Failed to load Vulkan scene materials: {}", gltfPath);
        releaseMeshletBuffers(outMeshlets);
        releaseMeshBuffers(outMesh);
        outMeshlets = MeshletData{};
        outMesh = LoadedMesh{};
        return false;
    }

    if (!outScene.buildFromGLTF(gltfPath, outMesh, outMeshlets)) {
        spdlog::error("Failed to build Vulkan scene graph: {}", gltfPath);
        releaseMaterialResources(outMaterials);
        releaseMeshletBuffers(outMeshlets);
        releaseMeshBuffers(outMesh);
        outMaterials = LoadedMaterials{};
        outMeshlets = MeshletData{};
        outMesh = LoadedMesh{};
        return false;
    }

    outScene.updateTransforms();
    spdlog::info("Loaded Vulkan scene: {}", gltfPath);
    return true;
}

bool createSceneFallbackTextures(const RhiDevice& device,
                                 RhiTextureHandle& outShadowDummy,
                                 RhiTextureHandle& outSkyFallback) {
    rhiReleaseHandle(outShadowDummy);
    rhiReleaseHandle(outSkyFallback);

    outShadowDummy = rhiCreateTexture2D(device,
                                        1,
                                        1,
                                        RhiFormat::R8Unorm,
                                        false,
                                        1,
                                        RhiTextureStorageMode::Shared,
                                        RhiTextureUsage::ShaderRead);
    outSkyFallback = rhiCreateTexture2D(device,
                                        1,
                                        1,
                                        RhiFormat::BGRA8Unorm,
                                        false,
                                        1,
                                        RhiTextureStorageMode::Shared,
                                        RhiTextureUsage::ShaderRead);
    if (!outShadowDummy.nativeHandle() || !outSkyFallback.nativeHandle()) {
        rhiReleaseHandle(outShadowDummy);
        rhiReleaseHandle(outSkyFallback);
        return false;
    }

    const uint8_t shadowValue = 0xFF;
    const uint8_t skyValue[4] = {77, 51, 26, 255};
    rhiUploadTexture2D(outShadowDummy, 1, 1, &shadowValue, 1);
    rhiUploadTexture2D(outSkyFallback, 1, 1, skyValue, sizeof(skyValue));
    return true;
}

void eraseResourceByName(PipelineAsset& asset, const char* resourceName) {
    asset.resources.erase(
        std::remove_if(asset.resources.begin(),
                       asset.resources.end(),
                       [resourceName](const ResourceDecl& resource) {
                           return resource.name == resourceName;
                       }),
        asset.resources.end());
}

void erasePassByType(PipelineAsset& asset, const char* passType) {
    asset.passes.erase(
        std::remove_if(asset.passes.begin(),
                       asset.passes.end(),
                       [passType](const PassDecl& pass) {
                           return pass.type == passType;
                       }),
        asset.passes.end());
}

void replacePassInput(PassDecl& pass, const char* oldName, const char* newName) {
    for (auto& input : pass.inputs) {
        if (input == oldName) {
            input = newName;
        }
    }
}

void disablePipelineAutoExposure(PipelineAsset& asset) {
    eraseResourceByName(asset, "exposureLut");
    erasePassByType(asset, "AutoExposurePass");

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

void disableVisibilityRayTracing(PipelineAsset& asset) {
    eraseResourceByName(asset, "shadowMap");
    erasePassByType(asset, "ShadowRayPass");

    for (auto& pass : asset.passes) {
        if (pass.type != "DeferredLightingPass") {
            continue;
        }

        pass.inputs.erase(
            std::remove(pass.inputs.begin(), pass.inputs.end(), "shadowMap"),
            pass.inputs.end());
    }
}

void disableVisibilityTAA(PipelineAsset& asset) {
    eraseResourceByName(asset, "taaOutput");
    erasePassByType(asset, "TAAPass");

    for (auto& pass : asset.passes) {
        if (pass.type == "AutoExposurePass" || pass.type == "TonemapPass") {
            replacePassInput(pass, "taaOutput", "lightingOutput");
        }
    }
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

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Metallic - Vulkan Sponza", nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return 1;
    }

    OrbitCamera previewCamera;
    AppState appState{};
    appState.input.camera = &previewCamera;
    glfwSetWindowUserPointer(window, &appState);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    setupInputCallbacks(window, &appState.input);

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
    VkDevice vkDevice = nativeToVkHandle<VkDevice>(native.device);
    VkPhysicalDevice vkPhysicalDevice = nativeToVkHandle<VkPhysicalDevice>(native.physicalDevice);
    VmaAllocator vmaAllocator = getVulkanAllocator(*rhi);
    RhiDeviceHandle deviceHandle(native.device);
    RhiCommandQueueHandle queueHandle(native.queue);
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
    imguiInitInfo.Instance = nativeToVkHandle<VkInstance>(native.instance);
    imguiInitInfo.PhysicalDevice = vkPhysicalDevice;
    imguiInitInfo.Device = vkDevice;
    imguiInitInfo.QueueFamily = native.graphicsQueueFamily;
    imguiInitInfo.Queue = nativeToVkHandle<VkQueue>(native.queue);
    imguiInitInfo.DescriptorPool = nativeToVkHandle<VkDescriptorPool>(native.descriptorPool);
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
                             nativeToVkHandle<VkQueue>(native.queue),
                             native.graphicsQueueFamily);
    vulkanSetShaderContext(vkDevice);
    vulkanLoadMeshShaderFunctions(vkDevice);

    const RhiFeatures& features = rhi->features();
    ShaderManager shaderManager(deviceHandle,
                                PROJECT_SOURCE_DIR,
                                features.meshShaders,
                                features.meshShaders,
                                ShaderManagerProfile::vulkanVisibility());
    if (!shaderManager.buildAll()) {
        spdlog::error("Failed to build Vulkan visibility shader set");
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
    MeshletData previewMeshlets;
    LoadedMaterials previewMaterials;
    SceneGraph previewScene;
    RhiTextureHandle shadowDummyTexture;
    RhiTextureHandle skyFallbackTexture;
    bool previewSceneReady = loadSponzaScene(deviceHandle,
                                             queueHandle,
                                             PROJECT_SOURCE_DIR,
                                             previewMesh,
                                             previewMeshlets,
                                             previewMaterials,
                                             previewScene);
    if (previewSceneReady) {
        previewSceneReady =
            createSceneFallbackTextures(deviceHandle, shadowDummyTexture, skyFallbackTexture);
    }
    if (!previewSceneReady) {
        spdlog::warn("Failed to load Vulkan Sponza scene; falling back to triangle path");
        releaseMaterialResources(previewMaterials);
        releaseMeshletBuffers(previewMeshlets);
        releaseMeshBuffers(previewMesh);
        rhiReleaseHandle(shadowDummyTexture);
        rhiReleaseHandle(skyFallbackTexture);
    }

    const std::string visibilityPipelinePath =
        std::string(PROJECT_SOURCE_DIR) + "/Pipelines/visibility_buffer.json";
    PipelineAsset visibilityPipelineBaseAsset;
    PipelineAsset visibilityPipelineAsset;
    bool visibilityPipelineBaseLoaded =
        loadPipelineAssetChecked(visibilityPipelinePath, "Vulkan visibility", visibilityPipelineBaseAsset);
    bool visibilityPipelineAssetLoaded = false;
    bool visibilityAutoExposureAvailable = false;
    bool visibilityTaaAvailable = false;
    bool visibilityGpuCullingAvailable = false;
    bool useVisibilityRenderGraph = false;
    bool atmosphereSkyAvailable = false;

    auto refreshVisibilityPipelineState = [&]() {
        visibilityPipelineAssetLoaded = visibilityPipelineBaseLoaded;
        visibilityPipelineAsset =
            visibilityPipelineBaseLoaded ? visibilityPipelineBaseAsset : PipelineAsset{};
        visibilityAutoExposureAvailable =
            hasComputePipeline("HistogramPass") && hasComputePipeline("AutoExposurePass");
        visibilityTaaAvailable = hasComputePipeline("TAAPass");
        visibilityGpuCullingAvailable =
            hasComputePipeline("MeshletCullPass") &&
            hasComputePipeline("BuildIndirectPass") &&
            hasRenderPipeline("VisibilityIndirectPass");

        auto validateVisibilityAsset = [&](const char* label) {
            std::string validationError;
            if (!visibilityPipelineAsset.validate(validationError)) {
                spdlog::error("{}: {}", label, validationError);
                visibilityPipelineAssetLoaded = false;
                return false;
            }
            return true;
        };

        if (visibilityPipelineAssetLoaded) {
            disableVisibilityRayTracing(visibilityPipelineAsset);
            if (validateVisibilityAsset("Invalid Vulkan visibility pipeline without ShadowRayPass")) {
                spdlog::info("Vulkan visibility pipeline uses dummy shadow input until ray tracing backend is ready");
            }
        }

        if (visibilityPipelineAssetLoaded && !visibilityTaaAvailable) {
            disableVisibilityTAA(visibilityPipelineAsset);
            if (validateVisibilityAsset("Invalid Vulkan visibility pipeline without TAA")) {
                spdlog::warn("Vulkan visibility TAA unavailable; tonemap now reads lightingOutput directly");
            }
        }

        if (visibilityPipelineAssetLoaded && !visibilityAutoExposureAvailable) {
            disablePipelineAutoExposure(visibilityPipelineAsset);
            if (validateVisibilityAsset("Invalid Vulkan visibility pipeline without AutoExposure")) {
                spdlog::warn("Vulkan visibility auto-exposure unavailable; falling back to manual exposure");
            }
        }

        useVisibilityRenderGraph =
            visibilityPipelineAssetLoaded &&
            previewSceneReady &&
            hasRenderPipeline("VisibilityPass") &&
            hasComputePipeline("DeferredLightingPass") &&
            hasRenderPipeline("TonemapPass") &&
            previewMesh.positionBuffer.nativeHandle() &&
            previewMesh.normalBuffer.nativeHandle() &&
            previewMesh.uvBuffer.nativeHandle() &&
            previewMesh.indexBuffer.nativeHandle() &&
            previewMeshlets.meshletBuffer.nativeHandle() &&
            previewMeshlets.meshletVertices.nativeHandle() &&
            previewMeshlets.meshletTriangles.nativeHandle() &&
            previewMeshlets.boundsBuffer.nativeHandle() &&
            previewMeshlets.materialIDs.nativeHandle() &&
            previewMaterials.materialBuffer.nativeHandle() &&
            previewMaterials.sampler.nativeHandle() &&
            shadowDummyTexture.nativeHandle() &&
            skyFallbackTexture.nativeHandle();
        atmosphereSkyAvailable = hasRenderPipeline("SkyPass") && atmosphereTextures.isValid();
    };

    auto logVisibilityMode = [&]() {
        if (useVisibilityRenderGraph) {
            std::string mode = "Vulkan RenderGraph mode: MeshletCullPass -> VisibilityPass -> SkyPass -> DeferredLightingPass";
            if (visibilityTaaAvailable) {
                mode += " -> TAAPass";
            }
            if (visibilityAutoExposureAvailable) {
                mode += " -> AutoExposurePass";
            }
            mode += " -> TonemapPass";
            spdlog::info("{}", mode);
            if (visibilityGpuCullingAvailable) {
                spdlog::info("Vulkan visibility dispatch: GPU-driven indirect meshlet path");
            } else {
                spdlog::info("Vulkan visibility dispatch: CPU meshlet path");
            }
        } else {
            spdlog::info("Vulkan RenderGraph mode: Triangle -> TonemapPass fallback");
        }
    };

    refreshVisibilityPipelineState();
    logVisibilityMode();

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
        atmosphereTextures.release();
        rhiReleaseHandle(shadowDummyTexture);
        rhiReleaseHandle(skyFallbackTexture);
        releaseMaterialResources(previewMaterials);
        releaseMeshletBuffers(previewMeshlets);
        releaseMeshBuffers(previewMesh);
        rhiReleaseHandle(linearSampler);
        rhiReleaseHandle(trianglePipeline);
        rhiReleaseHandle(vertexDescriptor);
        rhiReleaseHandle(vertexBuffer);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
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

    RaytracedShadowResources emptyShadows;
    const double depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;
    RhiDepthStencilStateHandle depthState =
        rhiCreateDepthStencilState(deviceHandle, true, ML_DEPTH_REVERSED);
    RenderContext renderContext{
        previewMesh,
        previewMeshlets,
        previewMaterials,
        previewScene,
        emptyShadows,
        depthState,
        shadowDummyTexture,
        skyFallbackTexture,
        depthClearValue,
    };

    runtimeContext.backbufferRhi = &backbufferTexture;
    runtimeContext.resourceFactory = &frameGraphBackend;

    const PipelineAsset sceneColorPostAsset = makeSceneColorPostPipelineAsset();
    PipelineBuilder postBuilder(renderContext);
    auto rebuildPostBuilder = [&](int targetWidth, int targetHeight) {
        const PipelineAsset& activePostAsset =
            useVisibilityRenderGraph ? visibilityPipelineAsset : sceneColorPostAsset;
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
        rhiReleaseHandle(shadowDummyTexture);
        rhiReleaseHandle(skyFallbackTexture);
        releaseMaterialResources(previewMaterials);
        releaseMeshletBuffers(previewMeshlets);
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
    previewCamera.initFromBounds(previewMesh.bboxMin, previewMesh.bboxMax);
    previewCamera.distance *= 0.8f;
    previewCamera.azimuth = 0.55f;
    previewCamera.elevation = 0.35f;
    if (!previewSceneReady) {
        previewCamera.nearZ = 0.1f;
        previewCamera.farZ = 100.0f;
    }
    DirectionalLight sunLight;
    if (previewSceneReady) {
        sunLight = previewScene.getSunDirectionalLight();
    } else {
        sunLight.direction = normalize(float3(0.35f, 0.85f, 0.25f));
        sunLight.color = float3(1.0f, 0.98f, 0.95f);
        sunLight.intensity = 2.5f;
    }
    const float3 sunDirection = normalize(sunLight.direction);
    std::vector<uint32_t> previewVisibleMeshletNodes;
    std::vector<uint32_t> previewVisibleIndexNodes;
    if (previewSceneReady) {
        previewVisibleMeshletNodes.reserve(previewScene.nodes.size());
        previewVisibleIndexNodes.reserve(previewScene.nodes.size());
        for (const auto& node : previewScene.nodes) {
            if (!previewScene.isNodeVisible(node.id)) {
                continue;
            }
            if (node.meshletCount > 0) {
                previewVisibleMeshletNodes.push_back(node.id);
            }
            if (node.indexCount > 0) {
                previewVisibleIndexNodes.push_back(node.id);
            }
        }
    } else {
        previewVisibleIndexNodes = {0};
    }
    bool showGraphDebug = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool reloadKeyDown = false;
    bool pipelineReloadKeyDown = false;
    bool shaderReloadRequested = false;
    bool pipelineReloadRequested = false;
    bool postBuilderNeedsRebuild = false;
    double lastFrameTime = glfwGetTime();
    float4x4 prevView = float4x4::Identity();
    float4x4 prevProj = float4x4::Identity();
    bool hasPrevMatrices = false;
    uint32_t frameIndex = 0;

    auto rebuildActivePipeline = [&](int targetWidth, int targetHeight) {
        rhi->waitIdle();
        if (!useVisibilityRenderGraph &&
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
        const bool f6Down = glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS;
        if (f6Down && !pipelineReloadKeyDown) {
            pipelineReloadRequested = true;
        }
        pipelineReloadKeyDown = f6Down;
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
            rhi->waitIdle();
            spdlog::info("Reloading Vulkan visibility shaders...");
            const bool previousVisibilityRenderGraph = useVisibilityRenderGraph;
            const bool previousAutoExposure = visibilityAutoExposureAvailable;
            const bool previousTaa = visibilityTaaAvailable;
            const bool previousAtmosphereSky = atmosphereSkyAvailable;
            auto [reloaded, failed] = shaderManager.reloadAll();
            refreshVisibilityPipelineState();

            if (failed == 0) {
                spdlog::info("All {} Vulkan visibility shaders reloaded successfully", reloaded);
            } else {
                spdlog::warn("{} Vulkan visibility shaders reloaded, {} failed (keeping old pipelines)",
                             reloaded,
                             failed);
            }

            if (previousVisibilityRenderGraph != useVisibilityRenderGraph ||
                previousAutoExposure != visibilityAutoExposureAvailable ||
                previousTaa != visibilityTaaAvailable ||
                previousAtmosphereSky != atmosphereSkyAvailable) {
                logVisibilityMode();
            }
            postBuilderNeedsRebuild = true;
        }

        if (pipelineReloadRequested) {
            pipelineReloadRequested = false;
            spdlog::info("Reloading Vulkan visibility pipeline asset...");

            PipelineAsset reloadedVisibilityAsset;
            if (loadPipelineAssetChecked(visibilityPipelinePath,
                                         "Vulkan visibility",
                                         reloadedVisibilityAsset)) {
                const bool previousVisibilityRenderGraph = useVisibilityRenderGraph;
                const bool previousAutoExposure = visibilityAutoExposureAvailable;
                const bool previousTaa = visibilityTaaAvailable;
                const bool previousAtmosphereSky = atmosphereSkyAvailable;
                visibilityPipelineBaseAsset = std::move(reloadedVisibilityAsset);
                visibilityPipelineBaseLoaded = true;
                refreshVisibilityPipelineState();

                if (previousVisibilityRenderGraph != useVisibilityRenderGraph ||
                    previousAutoExposure != visibilityAutoExposureAvailable ||
                    previousTaa != visibilityTaaAvailable ||
                    previousAtmosphereSky != atmosphereSkyAvailable) {
                    logVisibilityMode();
                }
                postBuilderNeedsRebuild = true;
            } else if (visibilityPipelineBaseLoaded) {
                spdlog::warn("Keeping previous Vulkan visibility pipeline: {}",
                             visibilityPipelineBaseAsset.name);
            }
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
        if (!useVisibilityRenderGraph && sceneColorTexture.nativeHandle()) {
            imageTracker.setLayout(getVulkanImage(&sceneColorTexture), sceneColorLayout);
        }

        VulkanCommandBuffer commandBuffer(getVulkanCurrentCommandBuffer(*rhi),
                                          vkDevice,
                                          &descriptorManager,
                                          &imageTracker);

        if (!useVisibilityRenderGraph) {
            sceneGraph.execute(commandBuffer, frameGraphBackend);
        }

        RhiNativeCommandBufferHandle nativeCommandBuffer(getVulkanCurrentCommandBuffer(*rhi));
        const float aspect =
            static_cast<float>(width) / static_cast<float>(std::max(height, 1));
        const float4x4 view = previewCamera.viewMatrix();
        const float4x4 unjitteredProj = previewCamera.projectionMatrix(aspect);
        float4x4 proj = unjitteredProj;
        float2 jitterOffset = float2(0.0f, 0.0f);
        const bool enableVisibilityTAA = useVisibilityRenderGraph && visibilityTaaAvailable;
        if (enableVisibilityTAA) {
            jitterOffset = OrbitCamera::haltonJitter(frameIndex);
            proj = OrbitCamera::jitteredProjectionMatrix(previewCamera.fovY,
                                                         aspect,
                                                         previewCamera.nearZ,
                                                         previewCamera.farZ,
                                                         jitterOffset.x,
                                                         jitterOffset.y,
                                                         static_cast<uint32_t>(width),
                                                         static_cast<uint32_t>(height));
        }

        std::unique_ptr<RhiBuffer> instanceTransformBuffer;
        uint32_t visibilityInstanceCount = 0;
        if (useVisibilityRenderGraph && !previewVisibleMeshletNodes.empty()) {
            visibilityInstanceCount = static_cast<uint32_t>(previewVisibleMeshletNodes.size());
            std::vector<SceneInstanceTransform> instanceTransforms;
            instanceTransforms.reserve(visibilityInstanceCount);
            const float4x4 prevViewForMotion = hasPrevMatrices ? prevView : view;
            const float4x4 prevProjForMotion = hasPrevMatrices ? prevProj : unjitteredProj;
            for (uint32_t instanceId = 0; instanceId < visibilityInstanceCount; ++instanceId) {
                const SceneNode& node = previewScene.nodes[previewVisibleMeshletNodes[instanceId]];
                const float4x4 nodeModelView = view * node.transform.worldMatrix;
                const float4x4 nodeMvp = proj * nodeModelView;
                const float4x4 prevNodeModelView = prevViewForMotion * node.transform.worldMatrix;
                const float4x4 prevNodeMvp = prevProjForMotion * prevNodeModelView;

                SceneInstanceTransform transform{};
                transform.mvp = transpose(nodeMvp);
                transform.modelView = transpose(nodeModelView);
                transform.prevMvp = transpose(prevNodeMvp);
                instanceTransforms.push_back(transform);
            }

            RhiBufferDesc instanceBufferDesc;
            instanceBufferDesc.size = instanceTransforms.size() * sizeof(SceneInstanceTransform);
            instanceBufferDesc.initialData = instanceTransforms.data();
            instanceBufferDesc.hostVisible = true;
            instanceBufferDesc.debugName = "Vulkan Visibility Instance Transforms";
            instanceTransformBuffer = frameGraphBackend.createBuffer(instanceBufferDesc);
            if (!instanceTransformBuffer) {
                visibilityInstanceCount = 0;
            }
        }

        frameContext = FrameContext{};
        frameContext.width = width;
        frameContext.height = height;
        frameContext.view = view;
        frameContext.proj = proj;
        frameContext.cameraWorldPos = orbitCameraWorldPosition(previewCamera);
        frameContext.prevView = hasPrevMatrices ? prevView : view;
        frameContext.prevProj = hasPrevMatrices ? prevProj : unjitteredProj;
        frameContext.jitterOffset = jitterOffset;
        frameContext.frameIndex = frameIndex;
        frameContext.enableTAA = enableVisibilityTAA;
        frameContext.worldLightDir = float4(sunDirection.x, sunDirection.y, sunDirection.z, 0.0f);
        frameContext.viewLightDir = view * frameContext.worldLightDir;
        frameContext.lightColorIntensity =
            float4(sunLight.color.x, sunLight.color.y, sunLight.color.z, sunLight.intensity);
        frameContext.meshletCount = previewMeshlets.meshletCount;
        frameContext.materialCount = previewMaterials.materialCount;
        frameContext.textureCount = static_cast<uint32_t>(previewMaterials.textures.size());
        frameContext.visibleMeshletNodes = previewVisibleMeshletNodes;
        frameContext.visibleIndexNodes = previewVisibleIndexNodes;
        frameContext.visibilityInstanceCount = visibilityInstanceCount;
        frameContext.instanceTransformBufferRhi = instanceTransformBuffer.get();
        frameContext.commandBuffer = &nativeCommandBuffer;
        frameContext.depthClearValue = depthClearValue;
        frameContext.cameraFarZ = previewCamera.farZ;
        {
            const double now = glfwGetTime();
            frameContext.deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;
        }
        frameContext.enableRTShadows = false;
        frameContext.enableAtmosphereSky = atmosphereSkyAvailable;
        frameContext.gpuDrivenCulling = useVisibilityRenderGraph && visibilityGpuCullingAvailable;
        frameContext.renderMode = useVisibilityRenderGraph ? 2 : 0;

        postBuilder.updateFrame(&backbufferTexture, &frameContext);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Vulkan Sponza");
        ImGui::Text("Resolution: %d x %d", width, height);
        ImGui::TextUnformatted(useVisibilityRenderGraph ? "Pipeline: Visibility Buffer" : "Pipeline: Triangle Fallback");
        ImGui::TextUnformatted(visibilityTaaAvailable ? "TAA: Enabled" : "TAA: Disabled");
        ImGui::TextUnformatted(visibilityAutoExposureAvailable ? "Exposure: Auto" : "Exposure: Manual");
        ImGui::TextUnformatted(visibilityGpuCullingAvailable ? "Visibility Dispatch: GPU" : "Visibility Dispatch: CPU");
        if (ImGui::Button("Reload Shaders (F5)")) {
            shaderReloadRequested = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload Pipeline (F6)")) {
            pipelineReloadRequested = true;
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

        if (!useVisibilityRenderGraph) {
            sceneColorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        prevView = view;
        prevProj = unjitteredProj;
        hasPrevMatrices = true;
        frameIndex++;

        rhi->endFrame();
        FrameMark;
    }

    cleanupRuntimeResources();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
