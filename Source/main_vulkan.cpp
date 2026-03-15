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
        true,
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

    const auto tonemapSpirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                         "Shaders/Post/tonemap",
                                                         PROJECT_SOURCE_DIR);
    if (tonemapSpirv.empty()) {
        spdlog::error("Failed to compile SPIR-V for tonemap shader");
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const auto forwardSpirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                         "Shaders/Vertex/bunny",
                                                         PROJECT_SOURCE_DIR);
    const auto skySpirv = compileSlangGraphicsBinary(RhiBackendType::Vulkan,
                                                     "Shaders/Atmosphere/sky",
                                                     PROJECT_SOURCE_DIR);

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

    std::string tonemapShaderSource(reinterpret_cast<const char*>(tonemapSpirv.data()),
                                    tonemapSpirv.size() * sizeof(uint32_t));
    RhiRenderPipelineSourceDesc tonemapPipelineDesc;
    tonemapPipelineDesc.vertexEntry = "vertexMain";
    tonemapPipelineDesc.fragmentEntry = "fragmentMain";
    tonemapPipelineDesc.colorFormat = rhi->colorFormat();

    std::string tonemapPipelineError;
    RhiGraphicsPipelineHandle tonemapPipeline =
        rhiCreateRenderPipelineFromSource(deviceHandle,
                                          tonemapShaderSource,
                                          tonemapPipelineDesc,
                                          tonemapPipelineError);
    if (!tonemapPipeline.nativeHandle()) {
        spdlog::error("Failed to create tonemap pipeline: {}", tonemapPipelineError);
        rhi->waitIdle();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    RhiVertexDescriptorHandle forwardVertexDescriptor = rhiCreateVertexDescriptor();
    rhiVertexDescriptorSetAttribute(forwardVertexDescriptor, 0, RhiVertexFormat::Float3, 0, 1);
    rhiVertexDescriptorSetAttribute(forwardVertexDescriptor, 1, RhiVertexFormat::Float3, 0, 2);
    rhiVertexDescriptorSetLayout(forwardVertexDescriptor, 1, sizeof(float) * 3);
    rhiVertexDescriptorSetLayout(forwardVertexDescriptor, 2, sizeof(float) * 3);

    RhiGraphicsPipelineHandle forwardPipeline;
    if (!forwardSpirv.empty()) {
        std::string forwardShaderSource(reinterpret_cast<const char*>(forwardSpirv.data()),
                                        forwardSpirv.size() * sizeof(uint32_t));
        RhiRenderPipelineSourceDesc forwardPipelineDesc;
        forwardPipelineDesc.vertexEntry = "vertexMain";
        forwardPipelineDesc.fragmentEntry = "fragmentMain";
        forwardPipelineDesc.colorFormat = RhiFormat::RGBA16Float;
        forwardPipelineDesc.depthFormat = RhiFormat::D32Float;
        forwardPipelineDesc.vertexDescriptor = &forwardVertexDescriptor;

        std::string forwardPipelineError;
        forwardPipeline = rhiCreateRenderPipelineFromSource(deviceHandle,
                                                            forwardShaderSource,
                                                            forwardPipelineDesc,
                                                            forwardPipelineError);
        if (!forwardPipeline.nativeHandle()) {
            spdlog::warn("Failed to create forward pipeline: {}", forwardPipelineError);
        }
    } else {
        spdlog::warn("Failed to compile SPIR-V for forward preview shader; falling back to triangle path");
    }

    RhiGraphicsPipelineHandle skyPipeline;
    if (!skySpirv.empty()) {
        std::string skyShaderSource(reinterpret_cast<const char*>(skySpirv.data()),
                                    skySpirv.size() * sizeof(uint32_t));
        RhiRenderPipelineSourceDesc skyPipelineDesc;
        skyPipelineDesc.vertexEntry = "vertexMain";
        skyPipelineDesc.fragmentEntry = "fragmentMain";
        skyPipelineDesc.colorFormat = RhiFormat::RGBA16Float;

        std::string skyPipelineError;
        skyPipeline = rhiCreateRenderPipelineFromSource(deviceHandle,
                                                        skyShaderSource,
                                                        skyPipelineDesc,
                                                        skyPipelineError);
        if (!skyPipeline.nativeHandle()) {
            spdlog::warn("Failed to create sky pipeline: {}", skyPipelineError);
        }
    } else {
        spdlog::warn("Failed to compile SPIR-V for sky shader; falling back to sceneColor present path");
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

    AtmosphereTextures atmosphereTextures;
    if (!loadAtmosphereTextures(deviceHandle, PROJECT_SOURCE_DIR, atmosphereTextures)) {
        spdlog::warn("Failed to load atmosphere textures; SkyPass will clear to black");
    }

    LoadedMesh previewMesh;
    SceneGraph previewScene;
    if (!createPreviewScene(deviceHandle, previewMesh, previewScene)) {
        spdlog::warn("Failed to create preview mesh; falling back to triangle path");
    }

    PipelineAsset previewPipelineAsset;
    const bool previewPipelineAssetLoaded =
        loadPipelineAssetChecked(std::string(PROJECT_SOURCE_DIR) + "/Pipelines/vulkan_preview.json",
                                 "Vulkan preview",
                                 previewPipelineAsset);
    const bool usePreviewRenderGraph =
        previewPipelineAssetLoaded &&
        forwardPipeline.nativeHandle() &&
        previewMesh.positionBuffer.nativeHandle() &&
        previewMesh.normalBuffer.nativeHandle() &&
        previewMesh.indexBuffer.nativeHandle();
    if (usePreviewRenderGraph) {
        spdlog::info("Vulkan RenderGraph mode: SkyPass -> ForwardPass -> TonemapPass");
    } else {
        spdlog::info("Vulkan RenderGraph mode: Triangle -> TonemapPass fallback");
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

    PipelineRuntimeContext runtimeContext;
    if (forwardPipeline.nativeHandle()) {
        runtimeContext.renderPipelinesRhi["ForwardPass"] = forwardPipeline;
    }
    runtimeContext.renderPipelinesRhi["TonemapPass"] = tonemapPipeline;
    if (skyPipeline.nativeHandle()) {
        runtimeContext.renderPipelinesRhi["SkyPass"] = skyPipeline;
    }
    runtimeContext.samplersRhi["tonemap"] = linearSampler;
    runtimeContext.samplersRhi["atmosphere"] = linearSampler;
    runtimeContext.importedTexturesRhi["sceneColor"] = sceneColorTexture;
    if (atmosphereTextures.isValid()) {
        runtimeContext.importedTexturesRhi["transmittance"] = atmosphereTextures.transmittance;
        runtimeContext.importedTexturesRhi["scattering"] = atmosphereTextures.scattering;
        runtimeContext.importedTexturesRhi["irradiance"] = atmosphereTextures.irradiance;
    }
    runtimeContext.backbufferRhi = &backbufferTexture;
    runtimeContext.resourceFactory = &frameGraphBackend;

    const PipelineAsset sceneColorPostAsset = makeSceneColorPostPipelineAsset();
    PipelineBuilder postBuilder(renderContext);
    auto rebuildPostBuilder = [&](int targetWidth, int targetHeight) {
        runtimeContext.importedTexturesRhi["sceneColor"] = sceneColorTexture;
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
        postBuilder.frameGraph().reset();
        sceneGraph.reset();
        descriptorManager.destroy();
        atmosphereTextures.release();
        releaseMeshBuffers(previewMesh);
        rhiReleaseHandle(depthState);
        rhiReleaseHandle(linearSampler);
        rhiReleaseHandle(skyPipeline);
        rhiReleaseHandle(forwardPipeline);
        rhiReleaseHandle(tonemapPipeline);
        rhiReleaseHandle(trianglePipeline);
        rhiReleaseHandle(sceneColorTexture);
        rhiReleaseHandle(forwardVertexDescriptor);
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
    const bool atmosphereSkyAvailable = skyPipeline.nativeHandle() && atmosphereTextures.isValid();
    const std::vector<uint32_t> previewVisibleIndexNodes = {0};

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
            if (!usePreviewRenderGraph) {
                if (!recreateSceneColorTexture(static_cast<uint32_t>(width), static_cast<uint32_t>(height))) {
                    spdlog::error("Failed to recreate offscreen scene color texture");
                    break;
                }
                sceneColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            }
            if (!rebuildPostBuilder(width, height)) {
                break;
            }
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

        runtimeContext.importedTexturesRhi["sceneColor"] = sceneColorTexture;
        runtimeContext.backbufferRhi = &backbufferTexture;
        postBuilder.updateFrame(&backbufferTexture, &frameContext);
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
