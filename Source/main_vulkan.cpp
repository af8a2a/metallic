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
#include "bindless_scene_constants.h"
#include "raytraced_shadows.h"
#include "rhi_resource_utils.h"
#include "rhi_shader_utils.h"
#include "shader_manager.h"
#include "slang_compiler.h"
#include "visibility_constants.h"
#include "cluster_lod_builder.h"
#include "vulkan_backend.h"
#include "vulkan_frame_graph.h"
#include "streamline_context.h"

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

static constexpr int kRenderResolutionBaseWidth = 1920;
static constexpr int kRenderResolutionBaseHeight = 1080;
static constexpr float kMinRenderScale = 0.25f;
static constexpr float kMaxRenderScale = 2.0f;

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
                     SceneGraph& outScene,
                     ClusterLODData& outLOD) {
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

    if (!outScene.normalizeSingleRootScale(device, outMesh, outMeshlets)) {
        spdlog::error("Failed to normalize Vulkan scene root scale: {}", gltfPath);
        releaseMaterialResources(outMaterials);
        releaseMeshletBuffers(outMeshlets);
        releaseMeshBuffers(outMesh);
        outMaterials = LoadedMaterials{};
        outMeshlets = MeshletData{};
        outMesh = LoadedMesh{};
        outScene = SceneGraph{};
        return false;
    }

    outScene.updateTransforms();

    // Build cluster LOD hierarchy
    outLOD = ClusterLODData{};
    if (!buildClusterLOD(device, outMesh, outMeshlets, outLOD)) {
        spdlog::warn("Cluster LOD build failed; continuing without LOD hierarchy");
    }

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

void eraseResourceById(PipelineAsset& asset, const std::string& resourceId) {
    asset.removeEdgesForResource(resourceId);
    asset.resources.erase(
        std::remove_if(asset.resources.begin(),
                       asset.resources.end(),
                       [&resourceId](const ResourceDecl& resource) {
                           return resource.id == resourceId;
                       }),
        asset.resources.end());
}

void erasePassByType(PipelineAsset& asset, const char* passType) {
    std::vector<std::string> passIds;
    for (const auto& pass : asset.passes) {
        if (pass.type == passType) {
            passIds.push_back(pass.id);
        }
    }

    for (const auto& passId : passIds) {
        asset.removeEdgesForPass(passId);
    }

    asset.passes.erase(
        std::remove_if(asset.passes.begin(),
                       asset.passes.end(),
                       [passType](const PassDecl& pass) {
                           return pass.type == passType;
                       }),
        asset.passes.end());
}

ResourceDecl* findFirstResourceByKind(PipelineAsset& asset, const char* resourceKind) {
    for (auto& resource : asset.resources) {
        if (resource.kind == resourceKind) {
            return &resource;
        }
    }
    return nullptr;
}

const ResourceDecl* findFirstResourceByKind(const PipelineAsset& asset, const char* resourceKind) {
    for (const auto& resource : asset.resources) {
        if (resource.kind == resourceKind) {
            return &resource;
        }
    }
    return nullptr;
}

EdgeDecl* findPassSlotBinding(PipelineAsset& asset,
                              const PassDecl* pass,
                              const char* direction,
                              const char* slotKey) {
    if (!pass) {
        return nullptr;
    }
    return asset.findEdge(pass->id, direction, slotKey);
}

const EdgeDecl* findPassSlotBinding(const PipelineAsset& asset,
                                    const PassDecl* pass,
                                    const char* direction,
                                    const char* slotKey) {
    if (!pass) {
        return nullptr;
    }
    return asset.findEdge(pass->id, direction, slotKey);
}

const ResourceDecl* findPassBoundResource(const PipelineAsset& asset,
                                          const PassDecl* pass,
                                          const char* direction,
                                          const char* slotKey) {
    const EdgeDecl* edge = findPassSlotBinding(asset, pass, direction, slotKey);
    return edge ? asset.findResourceById(edge->resourceId) : nullptr;
}

void ensurePassSlotBinding(PipelineAsset& asset,
                           const PassDecl& pass,
                           const char* direction,
                           const char* slotKey,
                           const std::string& resourceId) {
    if (EdgeDecl* edge = asset.findEdge(pass.id, direction, slotKey)) {
        edge->resourceId = resourceId;
        return;
    }

    asset.edges.push_back({
        generatePipelineAssetGuid(),
        pass.id,
        slotKey,
        direction,
        resourceId,
    });
}

void erasePassSlotBinding(PipelineAsset& asset,
                          const PassDecl& pass,
                          const char* direction,
                          const char* slotKey) {
    asset.edges.erase(
        std::remove_if(asset.edges.begin(),
                       asset.edges.end(),
                       [&](const EdgeDecl& edge) {
                           return edge.passId == pass.id &&
                                  edge.direction == direction &&
                                  edge.slotKey == slotKey;
                       }),
        asset.edges.end());
}

bool passProducesResource(const PipelineAsset& asset,
                          const PassDecl& pass,
                          const std::string& resourceId) {
    if (resourceId.empty()) {
        return false;
    }

    for (const auto& edge : asset.edges) {
        if (edge.passId == pass.id &&
            edge.direction == "output" &&
            edge.resourceId == resourceId) {
            return true;
        }
    }
    return false;
}

PassDecl* findFirstEnabledPassByType(PipelineAsset& asset, const char* passType) {
    for (auto& pass : asset.passes) {
        if (pass.enabled && pass.type == passType) {
            return &pass;
        }
    }
    return nullptr;
}

void normalizePipelineFinalDisplayOutput(PipelineAsset& asset) {
    const ResourceDecl* backbuffer = findFirstResourceByKind(asset, "backbuffer");
    if (!backbuffer) {
        return;
    }

    if (PassDecl* outputPass = findFirstEnabledPassByType(asset, "OutputPass")) {
        ensurePassSlotBinding(asset, *outputPass, "output", "target", backbuffer->id);
    } else if (PassDecl* tonemapPass = findFirstEnabledPassByType(asset, "TonemapPass")) {
        ensurePassSlotBinding(asset, *tonemapPass, "output", "output", backbuffer->id);
    }

    if (PassDecl* imguiPass = findFirstEnabledPassByType(asset, "ImGuiOverlayPass")) {
        ensurePassSlotBinding(asset, *imguiPass, "output", "target", backbuffer->id);
    }
}

void disablePipelineAutoExposure(PipelineAsset& asset) {
    std::vector<std::string> producedResourceIds;
    for (const auto& pass : asset.passes) {
        if (pass.type != "AutoExposurePass") {
            continue;
        }
        if (const EdgeDecl* edge = findPassSlotBinding(asset, &pass, "output", "exposureLut")) {
            producedResourceIds.push_back(edge->resourceId);
        }
    }

    erasePassByType(asset, "AutoExposurePass");

    for (auto& pass : asset.passes) {
        if (pass.type != "TonemapPass") {
            continue;
        }

        erasePassSlotBinding(asset, pass, "input", "exposureLut");
        pass.config["autoExposure"] = false;
    }

    for (const auto& resourceId : producedResourceIds) {
        eraseResourceById(asset, resourceId);
    }
}

void disableVisibilityRayTracing(PipelineAsset& asset) {
    std::vector<std::string> producedResourceIds;
    for (const auto& pass : asset.passes) {
        if (pass.type != "ShadowRayPass") {
            continue;
        }
        if (const EdgeDecl* edge = findPassSlotBinding(asset, &pass, "output", "shadowMap")) {
            producedResourceIds.push_back(edge->resourceId);
        }
    }

    erasePassByType(asset, "ShadowRayPass");

    for (auto& pass : asset.passes) {
        if (pass.type != "DeferredLightingPass") {
            continue;
        }

        erasePassSlotBinding(asset, pass, "input", "shadowMap");
    }

    for (const auto& resourceId : producedResourceIds) {
        eraseResourceById(asset, resourceId);
    }
}

enum class VisibilityUpscalerMode {
    None,
    TAA,
    DLSS,
};

struct VisibilityUpscalerSelection {
    VisibilityUpscalerMode activeMode = VisibilityUpscalerMode::None;
    bool hasTaaPass = false;
    bool hasDlssPass = false;
    std::string activePostSource;
    std::string diagnostic;
};

const PassDecl* findFirstEnabledPassByType(const PipelineAsset& asset, const char* passType) {
    for (const auto& pass : asset.passes) {
        if (pass.enabled && pass.type == passType) {
            return &pass;
        }
    }
    return nullptr;
}

const PassDecl* findEnabledProducerForResource(const PipelineAsset& asset, const std::string& resourceName) {
    if (resourceName.empty()) {
        return nullptr;
    }
    for (const auto& pass : asset.passes) {
        if (!pass.enabled) {
            continue;
        }

        if (passProducesResource(asset, pass, resourceName)) {
            return &pass;
        }
    }
    return nullptr;
}

VisibilityUpscalerSelection analyzeVisibilityUpscalerSelection(const PipelineAsset& asset) {
    VisibilityUpscalerSelection selection;

    const PassDecl* tonemapPass = findFirstEnabledPassByType(asset, "TonemapPass");
    const PassDecl* outputPass = findFirstEnabledPassByType(asset, "OutputPass");
    const PassDecl* autoExposurePass = findFirstEnabledPassByType(asset, "AutoExposurePass");

    selection.hasTaaPass = findFirstEnabledPassByType(asset, "TAAPass") != nullptr;
    selection.hasDlssPass = findFirstEnabledPassByType(asset, "StreamlineDlssPass") != nullptr;

    const ResourceDecl* tonemapSource =
        findPassBoundResource(asset, tonemapPass, "input", "source");
    const ResourceDecl* outputSource =
        findPassBoundResource(asset, outputPass, "input", "source");
    const ResourceDecl* autoExposureSource =
        findPassBoundResource(asset, autoExposurePass, "input", "source");

    const ResourceDecl* activePostSource = tonemapSource ? tonemapSource : outputSource;
    selection.activePostSource = activePostSource ? activePostSource->name : std::string{};

    const PassDecl* activePostProducer =
        activePostSource ? findEnabledProducerForResource(asset, activePostSource->id) : nullptr;
    const PassDecl* autoExposureProducer =
        autoExposureSource ? findEnabledProducerForResource(asset, autoExposureSource->id) : nullptr;

    if (activePostProducer) {
        if (activePostProducer->type == "TAAPass") {
            selection.activeMode = VisibilityUpscalerMode::TAA;
        } else if (activePostProducer->type == "StreamlineDlssPass") {
            selection.activeMode = VisibilityUpscalerMode::DLSS;
        }
    }

    if (selection.activeMode != VisibilityUpscalerMode::None) {
        return selection;
    }

    if (tonemapPass && !tonemapSource) {
        if (autoExposureProducer && autoExposureProducer->type == "StreamlineDlssPass") {
            selection.diagnostic =
                "TonemapPass.source is unbound. AutoExposurePass already reads "
                "StreamlineDlssPass.dlssOutput; connect TonemapPass.source to the same output "
                "and keep TonemapPass.exposureLut as the second input.";
        } else if (autoExposureProducer && autoExposureProducer->type == "TAAPass") {
            selection.diagnostic =
                "TonemapPass.source is unbound. AutoExposurePass already reads "
                "TAAPass.taaOutput; connect TonemapPass.source to the same output and keep "
                "TonemapPass.exposureLut as the second input.";
        } else {
            selection.diagnostic =
                "TonemapPass.source is unbound. Connect it to DeferredLightingPass.lightingOutput, "
                "TAAPass.taaOutput, or StreamlineDlssPass.dlssOutput.";
        }
        return selection;
    }

    if (!tonemapPass && outputPass && !outputSource) {
        selection.diagnostic =
            "OutputPass.source is unbound. Connect it to DeferredLightingPass.lightingOutput, "
            "TAAPass.taaOutput, or StreamlineDlssPass.dlssOutput.";
        return selection;
    }

    if (selection.hasDlssPass) {
        if (autoExposureProducer && autoExposureProducer->type == "StreamlineDlssPass") {
            selection.diagnostic =
                "StreamlineDlssPass exists and AutoExposurePass.source already reads "
                "StreamlineDlssPass.dlssOutput, but the active post chain does not.";
        } else if (!selection.activePostSource.empty()) {
            selection.diagnostic =
                "StreamlineDlssPass exists, but the active post chain still uses '" +
                selection.activePostSource + "'.";
        } else {
            selection.diagnostic =
                "StreamlineDlssPass is present, but TonemapPass.source/OutputPass.source is not "
                "bound to StreamlineDlssPass.dlssOutput.";
        }
        return selection;
    }

    if (selection.hasTaaPass) {
        if (autoExposureProducer && autoExposureProducer->type == "TAAPass") {
            selection.diagnostic =
                "TAAPass exists and AutoExposurePass.source already reads TAAPass.taaOutput, "
                "but the active post chain does not.";
        } else if (!selection.activePostSource.empty()) {
            selection.diagnostic =
                "TAAPass exists, but the active post chain still uses '" +
                selection.activePostSource + "'.";
        } else {
            selection.diagnostic =
                "TAAPass is present, but TonemapPass.source/OutputPass.source is not bound to "
                "TAAPass.taaOutput.";
        }
    }

    return selection;
}

VisibilityUpscalerMode detectVisibilityUpscalerMode(const PipelineAsset& asset) {
    return analyzeVisibilityUpscalerSelection(asset).activeMode;
}

const char* visibilityUpscalerModeName(VisibilityUpscalerMode mode) {
    switch (mode) {
    case VisibilityUpscalerMode::TAA:
        return "TAA";
    case VisibilityUpscalerMode::DLSS:
        return "DLSS";
    default:
        return "None";
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
    asset.schemaVersion = kPipelineAssetSchemaVersion;
    asset.name = "VulkanPostPipeline";

    ResourceDecl sceneColor;
    sceneColor.id = generatePipelineAssetGuid();
    sceneColor.name = "Scene Color";
    sceneColor.kind = "imported";
    sceneColor.type = "texture";
    sceneColor.format = "RGBA16Float";
    sceneColor.size = "screen";
    sceneColor.importKey = "sceneColor";

    ResourceDecl tonemapOutput;
    tonemapOutput.id = generatePipelineAssetGuid();
    tonemapOutput.name = "Tonemap Output";
    tonemapOutput.kind = "transient";
    tonemapOutput.type = "texture";
    tonemapOutput.format = "RGBA8Srgb";
    tonemapOutput.size = "screen";

    ResourceDecl backbuffer;
    backbuffer.id = generatePipelineAssetGuid();
    backbuffer.name = "Backbuffer";
    backbuffer.kind = "backbuffer";
    backbuffer.type = "texture";
    backbuffer.format = "BGRA8Unorm";
    backbuffer.size = "screen";

    asset.resources.push_back(sceneColor);
    asset.resources.push_back(tonemapOutput);
    asset.resources.push_back(backbuffer);

    PassDecl tonemapPass;
    tonemapPass.id = generatePipelineAssetGuid();
    tonemapPass.name = "Tonemap";
    tonemapPass.type = "TonemapPass";
    tonemapPass.enabled = true;
    tonemapPass.sideEffect = false;
    tonemapPass.config = {
        {"method", "Clip"},
        {"exposure", 1.0},
        {"contrast", 1.0},
        {"brightness", 1.0},
        {"saturation", 1.0},
        {"vignette", 0.0},
        {"dither", false},
        {"autoExposure", false},
    };

    PassDecl outputPass;
    outputPass.id = generatePipelineAssetGuid();
    outputPass.name = "Output";
    outputPass.type = "OutputPass";
    outputPass.enabled = true;
    outputPass.sideEffect = false;
    outputPass.config = nlohmann::json::object();

    PassDecl imguiOverlayPass;
    imguiOverlayPass.id = generatePipelineAssetGuid();
    imguiOverlayPass.name = "ImGui Overlay";
    imguiOverlayPass.type = "ImGuiOverlayPass";
    imguiOverlayPass.enabled = true;
    imguiOverlayPass.sideEffect = false;
    imguiOverlayPass.config = nlohmann::json::object();

    asset.passes.push_back(tonemapPass);
    asset.passes.push_back(outputPass);
    asset.passes.push_back(imguiOverlayPass);

    asset.edges.push_back({generatePipelineAssetGuid(), tonemapPass.id, "source", "input", sceneColor.id});
    asset.edges.push_back({generatePipelineAssetGuid(), tonemapPass.id, "output", "output", tonemapOutput.id});
    asset.edges.push_back({generatePipelineAssetGuid(), outputPass.id, "source", "input", tonemapOutput.id});
    asset.edges.push_back({generatePipelineAssetGuid(), outputPass.id, "target", "output", backbuffer.id});
    asset.edges.push_back({generatePipelineAssetGuid(), imguiOverlayPass.id, "target", "output", backbuffer.id});

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

float3 quaternionToEulerDeg(const float4& q) {
    const float sinRoll = 2.0f * (q.w * q.x + q.y * q.z);
    const float cosRoll = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    const float roll = std::atan2(sinRoll, cosRoll);

    const float sinPitch = 2.0f * (q.w * q.y - q.z * q.x);
    const float pitch = (std::fabs(sinPitch) >= 1.0f)
        ? std::copysign(OrbitCamera::kPi * 0.5f, sinPitch)
        : std::asin(sinPitch);

    const float sinYaw = 2.0f * (q.w * q.z + q.x * q.y);
    const float cosYaw = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    const float yaw = std::atan2(sinYaw, cosYaw);

    const float toDegrees = 180.0f / OrbitCamera::kPi;
    return float3(roll * toDegrees, pitch * toDegrees, yaw * toDegrees);
}

float4 eulerDegToQuaternion(const float3& eulerDeg) {
    const float toRadians = OrbitCamera::kPi / 180.0f;
    const float rx = eulerDeg.x * toRadians * 0.5f;
    const float ry = eulerDeg.y * toRadians * 0.5f;
    const float rz = eulerDeg.z * toRadians * 0.5f;

    const float cx = std::cos(rx);
    const float sx = std::sin(rx);
    const float cy = std::cos(ry);
    const float sy = std::sin(ry);
    const float cz = std::cos(rz);
    const float sz = std::sin(rz);

    return float4(sx * cy * cz - cx * sy * sz,
                  cx * sy * cz + sx * cy * sz,
                  cx * cy * sz - sx * sy * cz,
                  cx * cy * cz + sx * sy * sz);
}

void drawSceneGraphNodeTree(SceneGraph& scene, uint32_t nodeIndex) {
    SceneNode& node = scene.nodes[nodeIndex];

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (node.children.empty()) {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }
    if (node.parent < 0) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (scene.selectedNode == static_cast<int32_t>(nodeIndex)) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    if (!node.visible) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
    }

    const bool open =
        ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<uintptr_t>(nodeIndex)),
                         flags,
                         "%s",
                         node.name.c_str());

    if (!node.visible) {
        ImGui::PopStyleColor();
    }

    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        scene.selectedNode = static_cast<int32_t>(nodeIndex);
    }

    if (!open) {
        return;
    }

    for (uint32_t childIndex : node.children) {
        drawSceneGraphNodeTree(scene, childIndex);
    }
    ImGui::TreePop();
}

void drawSceneGraphProperties(SceneGraph& scene) {
    if (scene.selectedNode < 0 || scene.selectedNode >= static_cast<int32_t>(scene.nodes.size())) {
        ImGui::TextDisabled("No node selected");
        return;
    }

    SceneNode& node = scene.nodes[scene.selectedNode];
    TransformComponent& transform = node.transform;

    ImGui::Text("Name: %s", node.name.c_str());
    ImGui::Text("ID: %u", node.id);
    ImGui::Separator();

    bool transformChanged = false;

    ImGui::Checkbox("Visible", &node.visible);

    ImGui::Separator();
    ImGui::Text("Transform");

    float translation[3] = {transform.translation.x, transform.translation.y, transform.translation.z};
    if (ImGui::DragFloat3("Translation", translation, 0.01f)) {
        transform.translation = float3(translation[0], translation[1], translation[2]);
        transform.useLocalMatrix = false;
        transformChanged = true;
    }

    const float3 euler = quaternionToEulerDeg(transform.rotation);
    float rotation[3] = {euler.x, euler.y, euler.z};
    if (ImGui::DragFloat3("Rotation", rotation, 0.1f)) {
        transform.rotation = eulerDegToQuaternion(float3(rotation[0], rotation[1], rotation[2]));
        transform.useLocalMatrix = false;
        transformChanged = true;
    }

    float scale[3] = {transform.scale.x, transform.scale.y, transform.scale.z};
    if (ImGui::DragFloat3("Scale", scale, 0.01f, 0.001f, 100.0f)) {
        transform.scale = float3(scale[0], scale[1], scale[2]);
        transform.useLocalMatrix = false;
        transformChanged = true;
    }

    if (transformChanged) {
        scene.markDirty(static_cast<uint32_t>(scene.selectedNode));
    }

    if (node.hasLight && node.light.type == LightType::Directional) {
        ImGui::Separator();
        ImGui::Text("Light");
        ImGui::Text("Type: Directional");

        float direction[3] = {
            node.light.directional.direction.x,
            node.light.directional.direction.y,
            node.light.directional.direction.z,
        };
        if (ImGui::DragFloat3("Direction", direction, 0.01f, -1.0f, 1.0f)) {
            const float3 dir(direction[0], direction[1], direction[2]);
            const float dirLength = length(dir);
            if (dirLength > 1e-6f) {
                node.light.directional.direction = dir / dirLength;
            }
        }

        float color[3] = {
            node.light.directional.color.x,
            node.light.directional.color.y,
            node.light.directional.color.z,
        };
        if (ImGui::ColorEdit3("Color", color)) {
            node.light.directional.color = float3(color[0], color[1], color[2]);
        }

        ImGui::DragFloat("Intensity", &node.light.directional.intensity, 0.01f, 0.0f, 100.0f);

        if (scene.sunLightNode == scene.selectedNode) {
            ImGui::TextDisabled("Scene Sun Source");
        }
    }

    ImGui::Separator();
    ImGui::Text("Mesh Info");
    if (node.meshIndex >= 0) {
        ImGui::Text("Mesh Index: %d", node.meshIndex);
        ImGui::Text("Primitive Group Start: %u", node.primitiveGroupStart);
        ImGui::Text("Primitive Group Count: %u", node.primitiveGroupCount);
        ImGui::Text("Meshlet Start: %u", node.meshletStart);
        ImGui::Text("Meshlet Count: %u", node.meshletCount);
        ImGui::Text("Index Start: %u", node.indexStart);
        ImGui::Text("Index Count: %u", node.indexCount);
    } else {
        ImGui::TextDisabled("No mesh (transform node)");
    }
}

void drawSceneGraphUI(SceneGraph& scene) {
    ImGui::SetNextWindowSize(ImVec2(500.0f, 400.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Scene Graph")) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTable("SceneGraphLayout",
                          2,
                          ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_BordersInnerV |
                              ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Tree", ImGuiTableColumnFlags_WidthStretch, 0.45f);
        ImGui::TableSetupColumn("Properties", ImGuiTableColumnFlags_WidthStretch, 0.55f);

        ImGui::TableNextColumn();
        ImGui::BeginChild("TreePanel", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        for (uint32_t rootIndex : scene.rootNodes) {
            drawSceneGraphNodeTree(scene, rootIndex);
        }
        ImGui::EndChild();

        ImGui::TableNextColumn();
        ImGui::BeginChild("PropertyPanel", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        drawSceneGraphProperties(scene);
        ImGui::EndChild();

        ImGui::EndTable();
    }

    ImGui::End();
}

ImGuiID beginDockspace(bool& showSceneGraphWindow,
                       bool& showGraphDebug,
                       bool& showRenderPassUI,
                       bool& showImGuiDemo) {
    const ImGuiID dockspaceId =
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene Graph", nullptr, &showSceneGraphWindow);
            ImGui::MenuItem("FrameGraph", nullptr, &showGraphDebug);
            ImGui::MenuItem("Render Passes", nullptr, &showRenderPassUI);
            ImGui::MenuItem("ImGui Demo", nullptr, &showImGuiDemo);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    return dockspaceId;
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

    // --- Streamline / DLSS initialization (before RHI so SL can intercept device creation) ---
    StreamlineContext streamlineCtx;
    bool dlssAvailable = false;
    DlssPreset dlssPreset = DlssPreset::Balanced;
    uint32_t dlssRenderWidth = 0;
    uint32_t dlssRenderHeight = 0;
    float visibilityRenderScale = 1.0f;
    PipelineUiControls pipelineUiControls{};

#ifdef METALLIC_HAS_STREAMLINE
    if (streamlineCtx.init(PROJECT_SOURCE_DIR)) {
        createInfo.vkGetDeviceProcAddrProxy = streamlineCtx.vulkanDeviceProcAddrProxy();
        StreamlineVulkanRequirements slReqs;
        if (streamlineCtx.queryVulkanRequirements(slReqs)) {
            for (auto ext : slReqs.instanceExtensions) {
                createInfo.extraInstanceExtensions.push_back(ext);
            }
            for (auto ext : slReqs.deviceExtensions) {
                createInfo.extraDeviceExtensions.push_back(ext);
            }
            createInfo.enableTimelineSemaphore = slReqs.needsTimelineSemaphore;
        }
    }
#endif

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
    RhiDeviceHandle deviceHandle(native.device, rhi.get());
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

    const RhiFeatures& features = rhi->features();
    ShaderManagerProfile shaderProfile = ShaderManagerProfile::vulkanVisibility();
    ShaderManager shaderManager(deviceHandle,
                                PROJECT_SOURCE_DIR,
                                features.meshShaders,
                                features.meshShaders,
                                shaderProfile);
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
    ClusterLODData previewLOD;
    RhiTextureHandle shadowDummyTexture;
    RhiTextureHandle skyFallbackTexture;
    bool previewSceneReady = loadSponzaScene(deviceHandle,
                                             queueHandle,
                                             PROJECT_SOURCE_DIR,
                                             previewMesh,
                                             previewMeshlets,
                                             previewMaterials,
                                             previewScene,
                                             previewLOD);
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

    RaytracedShadowResources shadowResources;
    bool rtShadowsAvailable = false;
    bool enableRTShadows = true;
    if (previewSceneReady && rhi->features().rayTracing) {
        if (buildAccelerationStructures(deviceHandle,
                                        queueHandle,
                                        previewMesh,
                                        previewScene,
                                        shadowResources) &&
            createShadowPipeline(deviceHandle, shadowResources, PROJECT_SOURCE_DIR)) {
            rtShadowsAvailable = true;
            spdlog::info("Vulkan raytraced shadows enabled");
        } else {
            spdlog::warn("Failed to initialize Vulkan raytraced shadows; continuing with fallback lighting");
            shadowResources.release();
        }
    } else if (!rhi->features().rayTracing) {
        spdlog::info("Vulkan ray tracing not supported on this device; shadow pass stays disabled");
    }

    // --- Streamline: set Vulkan device and query DLSS availability ---
#ifdef METALLIC_HAS_STREAMLINE
    if (streamlineCtx.isInitialized()) {
        VkInstance vkInstance = nativeToVkHandle<VkInstance>(native.instance);
        if (streamlineCtx.setVulkanDevice(vkInstance, vkPhysicalDevice, vkDevice,
                                           nativeToVkHandle<VkQueue>(native.queue),
                                            native.graphicsQueueFamily, 0)) {
            vulkanSetStreamlineHookedCommandsEnabled(true);
            dlssAvailable = streamlineCtx.isDlssAvailable();
        }
    }
#endif

    const std::string visibilityPipelinePath =
        std::string(PROJECT_SOURCE_DIR) + "/Pipelines/visibilitybuffer.json";
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
    VisibilityUpscalerMode visibilityUpscalerMode = VisibilityUpscalerMode::None;
    VisibilityUpscalerSelection visibilityUpscalerSelection;
    bool dlssStateDirty = true;

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
            normalizePipelineFinalDisplayOutput(visibilityPipelineAsset);
        }

        if (visibilityPipelineAssetLoaded && !validateVisibilityAsset("Invalid Vulkan visibility pipeline")) {
            visibilityUpscalerMode = VisibilityUpscalerMode::None;
            dlssStateDirty = true;
        }

        if (visibilityPipelineAssetLoaded && !visibilityAutoExposureAvailable) {
            disablePipelineAutoExposure(visibilityPipelineAsset);
            if (validateVisibilityAsset("Invalid Vulkan visibility pipeline without AutoExposure")) {
                spdlog::warn("Vulkan visibility auto-exposure unavailable; falling back to manual exposure");
            }
        }

        visibilityUpscalerSelection = visibilityPipelineAssetLoaded
            ? analyzeVisibilityUpscalerSelection(visibilityPipelineAsset)
            : VisibilityUpscalerSelection{};
        visibilityUpscalerMode = visibilityUpscalerSelection.activeMode;
        if (visibilityUpscalerMode == VisibilityUpscalerMode::TAA && !visibilityTaaAvailable) {
            spdlog::warn("Vulkan visibility pipeline selects TAAPass, but the TAA shader is unavailable; the pass will forward its source input");
        }
        if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS && !dlssAvailable) {
            spdlog::warn("Vulkan visibility pipeline selects StreamlineDlssPass, but DLSS is unavailable; the pass will forward its source input");
        }
        dlssStateDirty = true;

        useVisibilityRenderGraph =
            visibilityPipelineAssetLoaded &&
            previewSceneReady &&
            hasRenderPipeline("VisibilityPass") &&
            hasComputePipeline("DeferredLightingPass") &&
            (hasRenderPipeline("TonemapPass") || hasRenderPipeline("OutputPass")) &&
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
            if (visibilityUpscalerMode == VisibilityUpscalerMode::TAA) {
                mode += " -> TAAPass";
            } else if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) {
                mode += " -> StreamlineDlssPass";
            }
            if (findFirstEnabledPassByType(visibilityPipelineAsset, "AutoExposurePass")) {
                mode += " -> AutoExposurePass";
            }
            if (findFirstEnabledPassByType(visibilityPipelineAsset, "TonemapPass")) {
                mode += " -> TonemapPass";
            }
            if (findFirstEnabledPassByType(visibilityPipelineAsset, "OutputPass")) {
                mode += " -> OutputPass";
            }
            spdlog::info("{}", mode);
            if (visibilityGpuCullingAvailable) {
                spdlog::info("Vulkan visibility dispatch: GPU-driven indirect meshlet path");
            } else {
                spdlog::info("Vulkan visibility dispatch: CPU meshlet path");
            }
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS &&
                runtimeContext.upscaler && runtimeContext.upscaler->isEnabled()) {
                spdlog::info("Vulkan render resolution: {} x {} -> {} x {} (DLSS)",
                             runtimeContext.renderWidth,
                             runtimeContext.renderHeight,
                             runtimeContext.displayWidth,
                             runtimeContext.displayHeight);
            } else {
                spdlog::info("Vulkan render resolution: {} x {} (base {} x {}, scale {:.2f}) -> {} x {}",
                             runtimeContext.renderWidth,
                             runtimeContext.renderHeight,
                             kRenderResolutionBaseWidth,
                             kRenderResolutionBaseHeight,
                             visibilityRenderScale,
                             runtimeContext.displayWidth,
                             runtimeContext.displayHeight);
            }
        } else {
            spdlog::info("Vulkan RenderGraph mode: Triangle -> TonemapPass -> OutputPass fallback");
        }
    };

    auto syncVisibilityUpscalerState = [&](int displayWidth, int displayHeight) {
        visibilityRenderScale = std::clamp(visibilityRenderScale, kMinRenderScale, kMaxRenderScale);

        int renderWidth = std::max(1, static_cast<int>(std::lround(
            static_cast<float>(kRenderResolutionBaseWidth) * visibilityRenderScale)));
        int renderHeight = std::max(1, static_cast<int>(std::lround(
            static_cast<float>(kRenderResolutionBaseHeight) * visibilityRenderScale)));
        bool enableDlssEvaluation = false;

#ifdef METALLIC_HAS_STREAMLINE
        if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS &&
            dlssAvailable &&
            dlssPreset != DlssPreset::Off) {
            uint32_t optimalWidth = static_cast<uint32_t>(displayWidth);
            uint32_t optimalHeight = static_cast<uint32_t>(displayHeight);
            if (streamlineCtx.getOptimalRenderSize(dlssPreset,
                                                   static_cast<uint32_t>(displayWidth),
                                                   static_cast<uint32_t>(displayHeight),
                                                   optimalWidth,
                                                   optimalHeight)) {
                renderWidth = static_cast<int>(optimalWidth);
                renderHeight = static_cast<int>(optimalHeight);
                enableDlssEvaluation = true;
                if (dlssStateDirty ||
                    runtimeContext.displayWidth != displayWidth ||
                    runtimeContext.displayHeight != displayHeight) {
                    if (!streamlineCtx.setDlssOptions(dlssPreset,
                                                      static_cast<uint32_t>(displayWidth),
                                                      static_cast<uint32_t>(displayHeight))) {
                        spdlog::warn("Failed to configure DLSS; StreamlineDlssPass will run in pass-through mode");
                        renderWidth = displayWidth;
                        renderHeight = displayHeight;
                        enableDlssEvaluation = false;
                    }
                    dlssStateDirty = false;
                }
            } else if (dlssStateDirty) {
                spdlog::warn("Failed to query optimal DLSS render size; StreamlineDlssPass will run in pass-through mode");
                dlssStateDirty = false;
            }
        } else {
            dlssStateDirty = true;
        }
#endif

        runtimeContext.upscaler = streamlineCtx.isInitialized() ? static_cast<IUpscalerIntegration*>(&streamlineCtx) : nullptr;
        runtimeContext.displayWidth = displayWidth;
        runtimeContext.displayHeight = displayHeight;
        runtimeContext.renderWidth = renderWidth;
        runtimeContext.renderHeight = renderHeight;
        dlssRenderWidth = static_cast<uint32_t>(renderWidth);
        dlssRenderHeight = static_cast<uint32_t>(renderHeight);
    };

    refreshVisibilityPipelineState();
    syncVisibilityUpscalerState(width, height);
    logVisibilityMode();

    VulkanFrameGraphBackend frameGraphBackend(vkDevice, vkPhysicalDevice, vmaAllocator);
    VulkanDescriptorManager descriptorManager;
    descriptorManager.init(vkDevice, vmaAllocator);
    if (previewSceneReady) {
        if (!previewMaterials.textureViews.empty()) {
            descriptorManager.updateBindlessSampledTextures(previewMaterials.textureViews.data(),
                                                            0,
                                                            static_cast<uint32_t>(previewMaterials.textureViews.size()));
        }
        descriptorManager.updateBindlessSampler(METALLIC_BINDLESS_SCENE_SAMPLER_INDEX,
                                                &previewMaterials.sampler);
        runtimeContext.useBindlessSceneTextures = true;
    }
    VulkanImageLayoutTracker imageTracker;
    streamlineCtx.setImageLayoutTracker(&imageTracker);
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
        shadowResources.release();
        rhiReleaseHandle(shadowDummyTexture);
        rhiReleaseHandle(skyFallbackTexture);
        releaseMaterialResources(previewMaterials);
        releaseMeshletBuffers(previewMeshlets);
        releaseMeshBuffers(previewMesh);
        rhiReleaseHandle(previewLOD.meshletBuffer);
        rhiReleaseHandle(previewLOD.meshletVerticesBuffer);
        rhiReleaseHandle(previewLOD.meshletTrianglesBuffer);
        rhiReleaseHandle(previewLOD.boundsBuffer);
        rhiReleaseHandle(previewLOD.materialIDsBuffer);
        rhiReleaseHandle(previewLOD.groupBuffer);
        rhiReleaseHandle(previewLOD.nodeBuffer);
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

    const double depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;
    RhiDepthStencilStateHandle depthState =
        rhiCreateDepthStencilState(deviceHandle, true, ML_DEPTH_REVERSED);
    RenderContext renderContext{
        previewMesh,
        previewMeshlets,
        previewMaterials,
        previewScene,
        shadowResources,
        depthState,
        shadowDummyTexture,
        skyFallbackTexture,
        depthClearValue,
    };

    runtimeContext.backbufferRhi = &backbufferTexture;
    runtimeContext.resourceFactory = &frameGraphBackend;
    runtimeContext.upscaler = streamlineCtx.isInitialized() ? static_cast<IUpscalerIntegration*>(&streamlineCtx) : nullptr;
    runtimeContext.uiControls = &pipelineUiControls;

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
        streamlineCtx.shutdown();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        postBuilder.frameGraph().reset();
        sceneGraph.reset();
        descriptorManager.destroy();
        atmosphereTextures.release();
        shadowResources.release();
        rhiReleaseHandle(shadowDummyTexture);
        rhiReleaseHandle(skyFallbackTexture);
        releaseMaterialResources(previewMaterials);
        releaseMeshletBuffers(previewMeshlets);
        releaseMeshBuffers(previewMesh);
        // Release cluster LOD buffers
        rhiReleaseHandle(previewLOD.meshletBuffer);
        rhiReleaseHandle(previewLOD.meshletVerticesBuffer);
        rhiReleaseHandle(previewLOD.meshletTrianglesBuffer);
        rhiReleaseHandle(previewLOD.boundsBuffer);
        rhiReleaseHandle(previewLOD.materialIDsBuffer);
        rhiReleaseHandle(previewLOD.groupBuffer);
        rhiReleaseHandle(previewLOD.nodeBuffer);
        rhiReleaseHandle(depthState);
        rhiReleaseHandle(linearSampler);
        rhiReleaseHandle(trianglePipeline);
        rhiReleaseHandle(sceneColorTexture);
        rhiReleaseHandle(vertexDescriptor);
        rhiReleaseHandle(vertexBuffer);
    };

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
    std::vector<uint32_t> previewVisibleMeshletNodes;
    std::vector<uint32_t> previewVisibleIndexNodes;
    auto refreshPreviewSceneState = [&]() {
        if (!previewSceneReady) {
            return;
        }

        previewScene.updateTransforms();
        sunLight = previewScene.getSunDirectionalLight();

        previewVisibleMeshletNodes.clear();
        previewVisibleIndexNodes.clear();
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
    };
    if (previewSceneReady) {
        previewVisibleMeshletNodes.reserve(previewScene.nodes.size());
        previewVisibleIndexNodes.reserve(previewScene.nodes.size());
        refreshPreviewSceneState();
    } else {
        previewVisibleIndexNodes = {0};
    }
    bool showSceneGraphWindow = true;
    bool showGraphDebug = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool reloadKeyDown = false;
    bool pipelineReloadKeyDown = false;
    bool shaderReloadRequested = false;
    bool pipelineReloadRequested = false;
    bool postBuilderNeedsRebuild = false;
    bool visibilityHistoryResetRequested = false;
    double lastFrameTime = glfwGetTime();
    float4x4 prevView = float4x4::Identity();
    float4x4 prevProj = float4x4::Identity();
    float4x4 prevCullView = float4x4::Identity();
    float4x4 prevCullProj = float4x4::Identity();
    float4 prevCameraWorldPos = float4(0.0f, 0.0f, 0.0f, 1.0f);
    bool hasPrevMatrices = false;
    uint32_t frameIndex = 0;

    auto applyDlssPresetChange = [&](DlssPreset newPreset) {
        if (newPreset == dlssPreset) {
            return;
        }
        dlssPreset = newPreset;
        dlssStateDirty = true;
        visibilityHistoryResetRequested = true;
        hasPrevMatrices = false;
        streamlineCtx.resetHistory();
        postBuilderNeedsRebuild = true;
    };

    auto refreshPipelineUiControls = [&]() {
        pipelineUiControls.enableRTShadows = &enableRTShadows;
        pipelineUiControls.rtShadowsAvailable = rtShadowsAvailable;
        pipelineUiControls.useVisibilityRenderGraph = useVisibilityRenderGraph;
        pipelineUiControls.hasDlssPass = visibilityUpscalerSelection.hasDlssPass;
        pipelineUiControls.dlssAvailable = dlssAvailable;
        pipelineUiControls.dlssEnabled = runtimeContext.upscaler && runtimeContext.upscaler->isEnabled();
        pipelineUiControls.dlssIsActiveUpscaler =
            visibilityUpscalerMode == VisibilityUpscalerMode::DLSS;
        pipelineUiControls.currentPreset = dlssPreset;
        pipelineUiControls.dlssRenderWidth = dlssRenderWidth;
        pipelineUiControls.dlssRenderHeight = dlssRenderHeight;
        pipelineUiControls.displayWidth = width;
        pipelineUiControls.displayHeight = height;
        pipelineUiControls.dlssDiagnostic = visibilityUpscalerSelection.diagnostic;
        pipelineUiControls.onDlssPresetChanged = applyDlssPresetChange;
        pipelineUiControls.onResetDlssHistory = [&]() {
            streamlineCtx.resetHistory();
        };
        runtimeContext.uiControls = &pipelineUiControls;
    };

    // Ring buffer to keep instance transform buffers alive for 2 frames (kMaxFramesInFlight).
    // Without this, the GPU may still be reading a buffer that the CPU has already freed.
    static constexpr uint32_t kBufferRingSize = 2;
    std::unique_ptr<RhiBuffer> instanceTransformRing[kBufferRingSize];

    auto rebuildActivePipeline = [&](int targetWidth, int targetHeight) {
        rhi->waitIdle();
        syncVisibilityUpscalerState(targetWidth, targetHeight);
        const int buildWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : targetWidth;
        const int buildHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : targetHeight;
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
        if (!rebuildPostBuilder(buildWidth, buildHeight)) {
            return false;
        }
        return true;
    };

    if (!rebuildActivePipeline(width, height)) {
        cleanupRuntimeResources();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

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
            dlssStateDirty = true;
            visibilityHistoryResetRequested = true;
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) {
                streamlineCtx.resetHistory();
            }
        }

        if (shaderReloadRequested) {
            shaderReloadRequested = false;
            rhi->waitIdle();
            spdlog::info("Reloading Vulkan visibility shaders...");
            const bool previousVisibilityRenderGraph = useVisibilityRenderGraph;
            const bool previousAutoExposure = visibilityAutoExposureAvailable;
            const bool previousTaa = visibilityTaaAvailable;
            const bool previousAtmosphereSky = atmosphereSkyAvailable;
            const VisibilityUpscalerMode previousUpscalerMode = visibilityUpscalerMode;
            auto [reloaded, failed] = shaderManager.reloadAll();
            refreshVisibilityPipelineState();

            if (rtShadowsAvailable) {
                if (reloadShadowPipeline(deviceHandle, shadowResources, PROJECT_SOURCE_DIR)) {
                    reloaded++;
                    spdlog::info("Reloaded Vulkan RT shadow shader");
                } else {
                    failed++;
                    spdlog::warn("Failed to reload Vulkan RT shadow shader; keeping previous pipeline");
                }
            }

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
                previousAtmosphereSky != atmosphereSkyAvailable ||
                previousUpscalerMode != visibilityUpscalerMode) {
                logVisibilityMode();
            }
            postBuilderNeedsRebuild = true;
            dlssStateDirty = true;
            visibilityHistoryResetRequested = true;
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) streamlineCtx.resetHistory();
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
                const VisibilityUpscalerMode previousUpscalerMode = visibilityUpscalerMode;
                visibilityPipelineBaseAsset = std::move(reloadedVisibilityAsset);
                visibilityPipelineBaseLoaded = true;
                refreshVisibilityPipelineState();

                if (previousVisibilityRenderGraph != useVisibilityRenderGraph ||
                    previousAutoExposure != visibilityAutoExposureAvailable ||
                    previousTaa != visibilityTaaAvailable ||
                    previousAtmosphereSky != atmosphereSkyAvailable ||
                    previousUpscalerMode != visibilityUpscalerMode) {
                    logVisibilityMode();
                }
                postBuilderNeedsRebuild = true;
            } else if (visibilityPipelineBaseLoaded) {
                spdlog::warn("Keeping previous Vulkan visibility pipeline: {}",
                             visibilityPipelineBaseAsset.name);
            }
            dlssStateDirty = true;
            visibilityHistoryResetRequested = true;
            if (visibilityUpscalerMode == VisibilityUpscalerMode::DLSS) streamlineCtx.resetHistory();
        }

        syncVisibilityUpscalerState(width, height);
        refreshPipelineUiControls();
        const int activeBuildWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : width;
        const int activeBuildHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : height;
        if (postBuilderNeedsRebuild || postBuilder.needsRebuild(activeBuildWidth, activeBuildHeight)) {
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

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const ImGuiID dockspaceId =
            beginDockspace(showSceneGraphWindow, showGraphDebug, showRenderPassUI, showImGuiDemo);

        ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
        ImGui::Begin("Vulkan Sponza");
        const bool autoExposureEnabled =
            useVisibilityRenderGraph &&
            findFirstEnabledPassByType(visibilityPipelineAsset, "AutoExposurePass") != nullptr;
        ImGui::Text("Resolution: %d x %d", width, height);
        ImGui::TextUnformatted(useVisibilityRenderGraph ? "Pipeline: Visibility Buffer" : "Pipeline: Triangle Fallback");
        ImGui::Text("Upscaler: %s", visibilityUpscalerModeName(visibilityUpscalerMode));
        ImGui::TextUnformatted(autoExposureEnabled ? "Exposure: Auto" : "Exposure: Manual");
        ImGui::TextUnformatted(visibilityGpuCullingAvailable ? "Visibility Dispatch: GPU" : "Visibility Dispatch: CPU");
        if (useVisibilityRenderGraph) {
            ImGui::Text("Render Base: %d x %d", kRenderResolutionBaseWidth, kRenderResolutionBaseHeight);
            const bool allowManualRenderScale = visibilityUpscalerMode != VisibilityUpscalerMode::DLSS;
            if (!allowManualRenderScale) {
                ImGui::BeginDisabled();
            }
            float requestedRenderScale = visibilityRenderScale;
            if (ImGui::SliderFloat("Render Scale",
                                   &requestedRenderScale,
                                   kMinRenderScale,
                                   kMaxRenderScale,
                                   "%.2f")) {
                requestedRenderScale = std::clamp(requestedRenderScale, kMinRenderScale, kMaxRenderScale);
                if (std::abs(requestedRenderScale - visibilityRenderScale) > 0.0001f) {
                    visibilityRenderScale = requestedRenderScale;
                    postBuilderNeedsRebuild = true;
                    visibilityHistoryResetRequested = true;
                    hasPrevMatrices = false;
                }
            }
            if (!allowManualRenderScale) {
                ImGui::EndDisabled();
                ImGui::TextUnformatted("Render Scale is disabled while DLSS is the active upscaler.");
            }
            ImGui::Text("Render Resolution: %d x %d",
                        runtimeContext.renderWidth,
                        runtimeContext.renderHeight);
        }
        if (ImGui::Button("Reload Shaders (F5)")) {
            shaderReloadRequested = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload Pipeline (F6)")) {
            pipelineReloadRequested = true;
        }
        ImGui::Checkbox("FrameGraph Debug", &showGraphDebug);
        ImGui::Checkbox("Render Pass UI", &showRenderPassUI);
        ImGui::Checkbox("Scene Graph", &showSceneGraphWindow);
        ImGui::Checkbox("ImGui Demo", &showImGuiDemo);

        // LOD stats panel
        if (previewSceneReady && previewLOD.lodLevelCount > 0) {
            drawClusterLODStats(previewLOD);
        }

        ImGui::End();

        if (previewSceneReady && showSceneGraphWindow) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            drawSceneGraphUI(previewScene);
        }

        refreshPreviewSceneState();

        const int renderWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : width;
        const int renderHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : height;
        const float aspect =
            static_cast<float>(width) / static_cast<float>(std::max(height, 1));
        const float3 sunDirection = normalize(sunLight.direction);
        const float4x4 view = previewCamera.viewMatrix();
        const float4x4 unjitteredProj = previewCamera.projectionMatrix(aspect);
        const float4 cameraWorldPos = orbitCameraWorldPosition(previewCamera);
        const float3 cameraWorldPos3(cameraWorldPos.x, cameraWorldPos.y, cameraWorldPos.z);
        const float3 cameraForward = normalize(previewCamera.target - cameraWorldPos3);
        const float3 worldUp(0.0f, 1.0f, 0.0f);
        const float3 cameraRight = normalize(cross(cameraForward, worldUp));
        const float3 cameraUp = cross(cameraRight, cameraForward);
        float4x4 proj = unjitteredProj;
        float2 jitterOffset = float2(0.0f, 0.0f);
        const bool enableVisibilityTAA =
            useVisibilityRenderGraph &&
            visibilityUpscalerMode == VisibilityUpscalerMode::TAA &&
            visibilityTaaAvailable;
        const bool enableVisibilityDlss =
            useVisibilityRenderGraph &&
            visibilityUpscalerMode == VisibilityUpscalerMode::DLSS &&
            runtimeContext.upscaler && runtimeContext.upscaler->isEnabled();
        const bool needsJitter = enableVisibilityTAA || enableVisibilityDlss;
        if (needsJitter) {
            jitterOffset = OrbitCamera::haltonJitter(frameIndex);
            proj = OrbitCamera::jitteredProjectionMatrix(previewCamera.fovY,
                                                         aspect,
                                                         previewCamera.nearZ,
                                                         previewCamera.farZ,
                                                         jitterOffset.x,
                                                         jitterOffset.y,
                                                         static_cast<uint32_t>(renderWidth),
                                                         static_cast<uint32_t>(renderHeight));
        }

        auto& instanceTransformBuffer = instanceTransformRing[frameIndex % kBufferRingSize];
        instanceTransformBuffer.reset();
        uint32_t visibilityInstanceCount = 0;
        if (useVisibilityRenderGraph && !previewVisibleMeshletNodes.empty()) {
            visibilityInstanceCount = static_cast<uint32_t>(
                std::min<size_t>(previewVisibleMeshletNodes.size(),
                                 static_cast<size_t>(kVisibilityInstanceMask + 1u)));
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

        RhiNativeCommandBufferHandle nativeCommandBuffer(getVulkanCurrentCommandBuffer(*rhi));
        frameContext = FrameContext{};
        frameContext.width = renderWidth;
        frameContext.height = renderHeight;
        frameContext.view = view;
        frameContext.proj = proj;
        frameContext.unjitteredProj = unjitteredProj;
        frameContext.cameraWorldPos = cameraWorldPos;
        frameContext.cameraRight = float4(cameraRight.x, cameraRight.y, cameraRight.z, 0.0f);
        frameContext.cameraUp = float4(cameraUp.x, cameraUp.y, cameraUp.z, 0.0f);
        frameContext.cameraForward = float4(cameraForward.x, cameraForward.y, cameraForward.z, 0.0f);
        frameContext.cameraNearZ = previewCamera.nearZ;
        frameContext.cameraFovY = previewCamera.fovY;
        frameContext.prevView = hasPrevMatrices ? prevView : view;
        frameContext.prevProj = hasPrevMatrices ? prevProj : unjitteredProj;
        frameContext.prevCullView = hasPrevMatrices ? prevCullView : view;
        frameContext.prevCullProj = hasPrevMatrices ? prevCullProj : proj;
        frameContext.prevCameraWorldPos =
            hasPrevMatrices ? prevCameraWorldPos : frameContext.cameraWorldPos;
        frameContext.jitterOffset = jitterOffset;
        frameContext.frameIndex = frameIndex;
        frameContext.enableTAA = enableVisibilityTAA;
        frameContext.displayWidth = width;
        frameContext.displayHeight = height;
        frameContext.renderWidth = renderWidth;
        frameContext.renderHeight = renderHeight;
        frameContext.historyReset = visibilityHistoryResetRequested;
        frameContext.worldLightDir = float4(sunDirection.x, sunDirection.y, sunDirection.z, 0.0f);
        frameContext.viewLightDir = view * frameContext.worldLightDir;
        frameContext.lightColorIntensity =
            float4(sunLight.color.x, sunLight.color.y, sunLight.color.z, sunLight.intensity);
        frameContext.meshletCount = previewMeshlets.meshletCount;
        frameContext.materialCount = previewMaterials.materialCount;
        frameContext.textureCount = static_cast<uint32_t>(previewMaterials.textures.size());
        frameContext.visibleMeshletNodes = previewVisibleMeshletNodes;
        if (useVisibilityRenderGraph &&
            frameContext.visibleMeshletNodes.size() > static_cast<size_t>(visibilityInstanceCount)) {
            frameContext.visibleMeshletNodes.resize(visibilityInstanceCount);
        }
        frameContext.visibleIndexNodes = previewVisibleIndexNodes;
        frameContext.visibilityInstanceCount = visibilityInstanceCount;
        frameContext.instanceTransformBufferRhi = instanceTransformBuffer.get();
        frameContext.depthClearValue = depthClearValue;
        frameContext.cameraFarZ = previewCamera.farZ;
        {
            const double now = glfwGetTime();
            frameContext.deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;
        }
        frameContext.enableRTShadows =
            useVisibilityRenderGraph && rtShadowsAvailable && enableRTShadows;
        frameContext.enableAtmosphereSky = atmosphereSkyAvailable;
        frameContext.gpuDrivenCulling = useVisibilityRenderGraph && visibilityGpuCullingAvailable;
        frameContext.renderMode = useVisibilityRenderGraph ? 2 : 0;

        postBuilder.updateFrame(&backbufferTexture, &frameContext);

        FrameGraph& activeFg = postBuilder.frameGraph();
        if (showGraphDebug) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            activeFg.debugImGui();
        }
        if (showRenderPassUI) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            activeFg.renderPassUI();
        }
        if (showImGuiDemo) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            ImGui::ShowDemoWindow(&showImGuiDemo);
        }
        ImGui::Render();

        if (!useVisibilityRenderGraph) {
            sceneGraph.execute(commandBuffer, frameGraphBackend);
        }

        if (useVisibilityRenderGraph && rtShadowsAvailable) {
            updateTLAS(nativeCommandBuffer, previewScene, shadowResources);
        }

        postBuilder.execute(commandBuffer, frameGraphBackend);
        visibilityHistoryResetRequested = false;

        if (!useVisibilityRenderGraph) {
            sceneColorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        prevView = view;
        prevProj = unjitteredProj;
        prevCullView = view;
        prevCullProj = proj;
        prevCameraWorldPos = frameContext.cameraWorldPos;
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
