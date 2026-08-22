#pragma once

#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructureExtensions.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Scene/Scene.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render::builtin_pass {

inline constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
inline constexpr const char* kTriangleShaderModuleName = "Triangle";
inline constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
inline constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";
inline constexpr const char* kImageSampleShaderModuleName = "ImageSample";
inline constexpr const char* kImageSampleVertexEntryPoint = "imageSampleVertexMain";
inline constexpr const char* kImageSampleFragmentEntryPoint = "imageSampleFragmentMain";
inline constexpr const char* kBunnyWireframeShaderModuleName = "BunnyWireframe";
inline constexpr const char* kBunnyWireframeVertexEntryPoint = "bunnyWireframeVertexMain";
inline constexpr const char* kBunnyWireframeFragmentEntryPoint = "bunnyWireframeFragmentMain";
inline constexpr const char* kMaterialShaderObjectShaderModuleName = "MaterialShaderObject";
inline constexpr const char* kMaterialShaderObjectVertexEntryPoint = "materialShaderObjectVertexMain";
inline constexpr const char* kMaterialShaderObjectFragmentEntryPoint = "materialShaderObjectFragmentMain";
inline constexpr const char* kMaterialShaderObjectAlternateFragmentEntryPoint =
    "materialShaderObjectAlternateFragmentMain";
inline constexpr const char* kSceneRayQueryVisualizationShaderModuleName = "SceneRayQueryVisualize";
inline constexpr const char* kSceneRayQueryVisualizationEntryPoint = "sceneRayQueryVisualizeMain";
inline constexpr const char* kGPUDrivenPreviewShaderModuleName = "GPUDrivenPreview";
inline constexpr const char* kGPUDrivenDeferredShaderModuleName = "GPUDrivenDeferred";
inline constexpr const char* kGPUDrivenPreviewMeshEntryPoint = "gpuDrivenPreviewMeshMain";
inline constexpr const char* kGPUDrivenPreviewFragmentEntryPoint = "gpuDrivenPreviewFragmentMain";
inline constexpr const char* kGPUDrivenPreviewMaskedFragmentEntryPoint =
    "gpuDrivenPreviewMaskedFragmentMain";
inline constexpr const char* kGPUDrivenPreviewResetEntryPoint = "gpuDrivenPreviewResetMain";
inline constexpr const char* kGPUDrivenPreviewInstanceCullEntryPoint = "gpuDrivenPreviewInstanceCullMain";
inline constexpr const char* kGPUDrivenPreviewCompactEntryPoint = "gpuDrivenPreviewCompactMain";
inline constexpr const char* kGPUDrivenPreviewHzbEntryPoint = "gpuDrivenPreviewHzbMain";
inline constexpr const char* kGPUDrivenPreviewDeferredEntryPoint = "gpuDrivenPreviewDeferredMain";
inline constexpr const char* kGPUDrivenPreviewCompositeVertexEntryPoint =
    "gpuDrivenPreviewCompositeVertexMain";
inline constexpr const char* kGPUDrivenPreviewCompositeFragmentEntryPoint =
    "gpuDrivenPreviewCompositeFragmentMain";
inline constexpr const char* kGPUDrivenStreamAssetShaderModuleName = "GPUDrivenStreamAsset";
inline constexpr const char* kGPUDrivenStreamAssetMeshEntryPoint = "gpuDrivenStreamAssetMeshMain";
inline constexpr const char* kGPUDrivenStreamAssetFragmentEntryPoint = "gpuDrivenStreamAssetFragmentMain";
inline constexpr const char* kGPUDrivenStreamAssetUpdateEntryPoint = "gpuDrivenStreamAssetApplyUpdatesMain";
inline constexpr const char* kSceneMaterialVisualizationShaderModuleName = "SceneMaterialVisualize";
inline constexpr const char* kSceneMaterialVisualizationEntryPoint = "sceneMaterialVisualizeMain";
inline constexpr const char* kScenePathTraceShaderModuleName = "ScenePathTrace";
inline constexpr const char* kScenePathTraceEntryPoint = "scenePathTraceMain";
inline constexpr const char* kSceneRtxdiShaderModuleName = "SceneRtxdi";
inline constexpr const char* kSceneRtxdiEntryPoint = "sceneRtxdiMain";
inline constexpr const char* kRtxdiConfidenceShaderModuleName = "RtxdiConfidence";
inline constexpr const char* kRtxdiConfidenceEntryPoint = "rtxdiConfidenceMain";
inline constexpr const char* kRtxdiCompositeShaderModuleName = "RtxdiComposite";
inline constexpr const char* kRtxdiCompositeEntryPoint = "rtxdiCompositeMain";
inline constexpr const char* kScenePathTraceGuidesShaderModuleName = "ScenePathTraceGuides";
inline constexpr const char* kScenePathTraceGuidesEntryPoint = "scenePathTraceGuidesMain";
inline constexpr const char* kSceneSharcMaintenanceShaderModuleName = "SceneSharcMaintenance";
inline constexpr const char* kScenePathTraceTonemapShaderModuleName = "ScenePathTraceTonemap";
inline constexpr const char* kScenePathTraceTonemapEntryPointName = "scenePathTraceTonemapMain";
inline constexpr const char* kOpenPBRRayQueryPathTraceShaderModuleName = "OpenPBRRayQueryPathTrace";
inline constexpr const char* kOpenPBRRayQueryPathTraceEntryPoint = "openPbrRayQueryPathTraceMain";
inline constexpr const char* kOpenPBRRayQueryPathTraceGuidesShaderModuleName = "OpenPBRRayQueryPathTraceGuides";
inline constexpr const char* kOpenPBRRayQueryPathTraceGuidesEntryPoint = "openPbrRayQueryPathTraceGuidesMain";
inline constexpr const char* kRenderGraphBufferShaderModuleName = "RenderGraphBuffer";
inline constexpr const char* kRenderGraphBufferWriteEntryPoint = "renderGraphBufferWriteMain";
inline constexpr const char* kRenderGraphBufferCopyEntryPoint = "renderGraphBufferCopyMain";
inline constexpr const char* kDefaultImageSamplePath = PROJECT_SOURCE_DIR "/Asset/statue-1275469_1280.jpg";
inline constexpr const char* kDefaultBunnyScenePath = PROJECT_SOURCE_DIR "/Asset/StandfordBunny/scene.gltf";
inline constexpr const char* kDefaultMaterialScenePath = PROJECT_SOURCE_DIR "/Asset/StandfordBunny/scene.gltf";
inline constexpr const char* kDefaultGPUDrivenScenePath =
    PROJECT_SOURCE_DIR "/Asset/SuperSponza/NewSponza_Main_glTF_003.gltf";
inline constexpr uint64_t kRenderGraphBufferByteSize = 16;
inline constexpr int32_t kGltfTriangleListMode = 4;
inline constexpr uint32_t kRayQueryVisualizationGranularityInstance = 0;
inline constexpr uint32_t kRayQueryVisualizationGranularityPrimitive = 1;
inline constexpr uint32_t kRayQueryVisualizationGranularityClusterId = 2;
inline constexpr uint32_t kGPUDrivenPreviewModeMeshlet = 0;
inline constexpr uint32_t kGPUDrivenPreviewModePrimitive = 1;
inline constexpr uint32_t kGPUDrivenPreviewModeLod = 2;
inline constexpr uint32_t kGPUDrivenPreviewModeShaded = 3;
inline constexpr uint32_t kGPUDrivenPreviewModeBaseColor = 4;
inline constexpr uint32_t kGPUDrivenPreviewMeshletTriangleChunkSize = 64;
inline constexpr uint32_t kGPUDrivenPreviewMeshletChunkCount = 2;
inline constexpr uint32_t kGPUDrivenPreviewDrawBucketCount = 4;
inline constexpr uint32_t kGPUDrivenPreviewIndirectArgumentUintCount = 4;
inline constexpr uint32_t kGPUDrivenPreviewMeshletDoubleSided = 1u << 0u;
inline constexpr uint32_t kGPUDrivenPreviewMeshletAlphaMasked = 1u << 1u;
inline constexpr uint32_t kGPUDrivenPreviewMeshletAlphaBlend = 1u << 2u;
inline constexpr uint32_t kGPUDrivenPreviewInstanceVisible = 1u << 0u;
inline constexpr uint32_t kGPUDrivenPreviewCullInstanceFrustum = 1u << 0u;
inline constexpr uint32_t kGPUDrivenPreviewCullInstanceHzb = 1u << 1u;
inline constexpr uint32_t kGPUDrivenPreviewCullMeshletFrustum = 1u << 2u;
inline constexpr uint32_t kGPUDrivenPreviewCullMeshletNormalCone = 1u << 3u;
inline constexpr uint32_t kGPUDrivenStreamAssetDebugPage = 0;
inline constexpr uint32_t kGPUDrivenStreamAssetDebugLod = 1;
inline constexpr uint32_t kGPUDrivenStreamAssetDebugPrimitive = 2;
inline constexpr uint32_t kSceneMaterialVisualizationModeMaterial = 0;
inline constexpr uint32_t kSceneMaterialVisualizationModeBaseColor = 1;
inline constexpr uint32_t kSceneMaterialVisualizationModeNormal = 2;
inline constexpr uint32_t kSceneMaterialVisualizationModeRoughness = 3;
inline constexpr uint32_t kSceneMaterialVisualizationModeMetallic = 4;
inline constexpr uint32_t kSceneMaterialVisualizationModeAo = 5;
inline constexpr uint32_t kSceneMaterialVisualizationModeGeometryNormal = 6;
inline constexpr uint32_t kSceneMaterialVisualizationModeVertexNormal = 7;
inline constexpr uint32_t kSceneMaterialVisualizationModeNormalTexture = 8;
inline constexpr uint32_t kSceneMaterialVisualizationModeTangent = 9;
inline constexpr uint32_t kSceneMaterialVisualizationModeBitangent = 10;
inline constexpr uint32_t kSceneMaterialVisualizationModeNrdNormalRoughness = 11;
inline constexpr uint32_t kSceneMaterialVisualizationModeNormalDeviation = 12;
inline constexpr uint32_t kDefaultPathTraceMaxDepth = 3;
inline constexpr uint32_t kDefaultPathTraceSamples = 2;
inline constexpr uint32_t kMaxPathTraceMaxDepth = 32;
inline constexpr uint32_t kMaxPathTraceSamples = 16;
inline constexpr uint32_t kDefaultRtxdiLightCount = 256;
inline constexpr uint32_t kDefaultRtxdiInitialSamples = 8;
inline constexpr uint32_t kDefaultRtxdiSpatialSamples = 1;
inline constexpr uint32_t kMaxRtxdiLightCount = 4096;
inline constexpr uint32_t kMaxRtxdiInitialSamples = 32;
inline constexpr uint32_t kMaxRtxdiSpatialSamples = 16;
inline constexpr uint32_t kDefaultReGIRGridSize = 12;
inline constexpr uint32_t kDefaultReGIRLightsPerCell = 64;
inline constexpr uint32_t kDefaultReGIRBuildSamples = 8;
inline constexpr uint32_t kMaxReGIRGridSize = 24;
inline constexpr uint32_t kMaxReGIRLightsPerCell = 128;
inline constexpr uint32_t kMaxReGIRBuildSamples = 32;
inline constexpr uint32_t kRtxdiVisualizationShaded = 0;
inline constexpr uint32_t kRtxdiVisualizationLightId = 1;
inline constexpr uint32_t kRtxdiVisualizationHistory = 2;
inline constexpr uint32_t kRtxdiVisualizationReGIRCells = 3;
inline constexpr uint32_t kRtxdiBehaviorTemporalReuse = 1u << 0u;
inline constexpr uint32_t kRtxdiBehaviorSpatialReuse = 1u << 1u;
inline constexpr uint32_t kRtxdiBehaviorAnimateLights = 1u << 2u;
inline constexpr uint32_t kRtxdiBehaviorInitialVisibility = 1u << 3u;
inline constexpr uint32_t kRtxdiBehaviorLocalLightImportance = 1u << 4u;
inline constexpr uint32_t kRtxdiBehaviorEnvironmentEnabled = 1u << 5u;
inline constexpr uint32_t kRtxdiBehaviorEnvironmentVisible = 1u << 6u;
inline constexpr uint32_t kRtxdiBehaviorEnvironmentImportance = 1u << 7u;
inline constexpr uint32_t kRtxdiBehaviorVisualizationShift = 8u;
inline constexpr uint32_t kRtxdiBehaviorReGIR = 1u << 10u;
inline constexpr uint32_t kScenePathTraceEnvironmentModeProcedural = 0;
inline constexpr uint32_t kScenePathTraceEnvironmentModeMap = 1;
inline constexpr uint32_t kScenePathTraceEnvironmentModeDisabled = 2;
inline constexpr uint32_t kScenePathTraceDebugViewFinal = 0;
inline constexpr uint32_t kScenePathTraceDebugViewGeometryNormal = 1;
inline constexpr uint32_t kScenePathTraceDebugViewShadingNormal = 2;
inline constexpr uint32_t kScenePathTraceDebugViewMappedNormal = 3;
inline constexpr uint32_t kScenePathTraceDebugViewTangent = 4;
inline constexpr uint32_t kScenePathTraceDebugViewBitangent = 5;
inline constexpr uint32_t kScenePathTraceDebugViewTangentHandedness = 6;
inline constexpr uint32_t kScenePathTraceDebugViewTexcoord = 7;
inline constexpr uint32_t kScenePathTraceDebugViewFrontFace = 8;
inline constexpr uint32_t kScenePathTraceDebugViewMaterial = 9;
inline constexpr uint32_t kScenePathTraceDebugViewInstance = 10;
inline constexpr uint32_t kScenePathTraceDebugViewTriangle = 11;
inline constexpr uint32_t kScenePathTraceDebugViewBaseColor = 12;
inline constexpr uint32_t kScenePathTraceDebugViewNormalTexture = 13;
inline constexpr uint32_t kScenePathTraceDebugViewShadowTransmittance = 14;
inline constexpr uint32_t kScenePathTraceDebugViewShadingSide = 15;
inline constexpr uint32_t kScenePathTraceDebugDisableNormalMap = 1u << 0u;
inline constexpr uint32_t kScenePathTraceDebugForceGeometryNormal = 1u << 1u;
inline constexpr uint32_t kScenePathTraceDebugDisableMaterialTextures = 1u << 2u;
inline constexpr uint32_t kScenePathTraceDebugDisableDirectLighting = 1u << 3u;
inline constexpr uint32_t kScenePathTraceDebugUseOpaqueShadows = 1u << 4u;
inline constexpr uint32_t kScenePathTraceDebugDisableTransmission = 1u << 5u;
inline constexpr uint32_t kScenePathTraceDebugDisableShadows = 1u << 6u;
inline constexpr uint32_t kScenePathTraceDebugDisableVolumeAttenuation = 1u << 7u;
// Radiance cache modes (RTXGI SHaRC / NVIDIA NRC reference integrations).
inline constexpr uint32_t kScenePathTraceCacheModeOff = 0;
inline constexpr uint32_t kScenePathTraceCacheModeSharc = 1;
inline constexpr uint32_t kScenePathTraceCacheModeNrc = 2;
// Extra descriptor bindings used by the radiance-cache permutations of
// ScenePathTrace.slang. Must match the [[vk::binding]] annotations there.
inline constexpr uint32_t kScenePathTraceCacheParamsBinding = 20;
inline constexpr uint32_t kScenePathTraceSharcHashEntriesBinding = 21;
inline constexpr uint32_t kScenePathTraceSharcAccumulationBinding = 22;
inline constexpr uint32_t kScenePathTraceSharcResolvedBinding = 23;
inline constexpr uint32_t kScenePathTraceNrcQueryPathInfoBinding = 24;
inline constexpr uint32_t kScenePathTraceNrcTrainingPathInfoBinding = 25;
inline constexpr uint32_t kScenePathTraceNrcTrainingPathVerticesBinding = 26;
inline constexpr uint32_t kScenePathTraceNrcQueryRadianceParamsBinding = 27;
inline constexpr uint32_t kScenePathTraceNrcCountersBinding = 28;
inline constexpr uint32_t kNrdDenoiserModeReblur = 0;
inline constexpr uint32_t kNrdDenoiserModeRelax = 1;
inline constexpr uint32_t kNrdDenoiserModeReference = 2;
inline constexpr const char* kScenePathTraceHistoryPrefix = "ScenePathTracePass.";
inline constexpr bool kDefaultReversedZ = true;

inline RenderGraphRuntimeSetting runtimeBoolSetting(
    std::string key,
    std::string label,
    bool defaultValue,
    bool invalidateHistory = false)
{
    return RenderGraphRuntimeSetting{
        .key = std::move(key),
        .label = std::move(label),
        .type = RenderGraphRuntimeSettingType::Bool,
        .defaultValue = defaultValue,
        .invalidateHistory = invalidateHistory,
    };
}

inline RenderGraphRuntimeSetting runtimeIntSetting(
    std::string key,
    std::string label,
    int32_t defaultValue,
    int32_t minValue,
    int32_t maxValue,
    bool invalidateHistory = false)
{
    return RenderGraphRuntimeSetting{
        .key = std::move(key),
        .label = std::move(label),
        .type = RenderGraphRuntimeSettingType::Int,
        .defaultValue = defaultValue,
        .minValue = minValue,
        .maxValue = maxValue,
        .invalidateHistory = invalidateHistory,
    };
}

inline RenderGraphRuntimeSetting runtimeFloatSetting(
    std::string key,
    std::string label,
    float defaultValue,
    float minValue,
    float maxValue,
    bool invalidateHistory = false)
{
    return RenderGraphRuntimeSetting{
        .key = std::move(key),
        .label = std::move(label),
        .type = RenderGraphRuntimeSettingType::Float,
        .defaultValue = defaultValue,
        .minValue = minValue,
        .maxValue = maxValue,
        .invalidateHistory = invalidateHistory,
    };
}

inline RenderGraphRuntimeSetting runtimeFloat3Setting(
    std::string key,
    std::string label,
    const std::array<float, 3>& defaultValue,
    bool invalidateHistory = false)
{
    return RenderGraphRuntimeSetting{
        .key = std::move(key),
        .label = std::move(label),
        .type = RenderGraphRuntimeSettingType::Float3,
        .defaultValue = RenderGraphProperties::array({defaultValue[0], defaultValue[1], defaultValue[2]}),
        .invalidateHistory = invalidateHistory,
    };
}

inline RenderGraphRuntimeSetting runtimeColor4Setting(
    std::string key,
    std::string label,
    const std::array<float, 4>& defaultValue,
    bool invalidateHistory = false)
{
    return RenderGraphRuntimeSetting{
        .key = std::move(key),
        .label = std::move(label),
        .type = RenderGraphRuntimeSettingType::Color4,
        .defaultValue = RenderGraphProperties::array({defaultValue[0], defaultValue[1], defaultValue[2], defaultValue[3]}),
        .minValue = RenderGraphProperties::array({0.0f, 0.0f, 0.0f, 0.0f}),
        .maxValue = RenderGraphProperties::array({1.0f, 1.0f, 1.0f, 1.0f}),
        .invalidateHistory = invalidateHistory,
    };
}

inline RenderGraphRuntimeSetting runtimeEnumSetting(
    std::string key,
    std::string label,
    std::string defaultValue,
    std::initializer_list<std::pair<std::string_view, std::string_view>> options,
    bool invalidateHistory = false)
{
    RenderGraphRuntimeSetting setting{
        .key = std::move(key),
        .label = std::move(label),
        .type = RenderGraphRuntimeSettingType::Enum,
        .defaultValue = std::move(defaultValue),
        .invalidateHistory = invalidateHistory,
    };
    setting.options.reserve(options.size());
    for (const auto& [optionLabel, optionValue] : options) {
        setting.options.push_back(RenderGraphRuntimeSettingOption{
            .label = std::string(optionLabel),
            .value = std::string(optionValue),
        });
    }
    return setting;
}

inline RenderGraphRuntimeSetting runtimeActionCounterSetting(
    std::string key,
    std::string label,
    bool invalidateHistory = false)
{
    return RenderGraphRuntimeSetting{
        .key = std::move(key),
        .label = std::move(label),
        .type = RenderGraphRuntimeSettingType::ActionCounter,
        .defaultValue = 0,
        .minValue = 0,
        .maxValue = std::numeric_limits<int32_t>::max(),
        .invalidateHistory = invalidateHistory,
    };
}

inline void appendCameraRuntimeSettings(
    std::vector<RenderGraphRuntimeSetting>& settings,
    const std::array<float, 3>& eye,
    const std::array<float, 3>& center,
    float fovDegrees,
    bool invalidateHistory = false)
{
    settings.push_back(runtimeFloat3Setting("camera.eye", "Eye", eye, invalidateHistory));
    settings.push_back(runtimeFloat3Setting("camera.center", "Center", center, invalidateHistory));
    settings.push_back(runtimeFloat3Setting("camera.up", "Up", {0.0f, 1.0f, 0.0f}, invalidateHistory));
    settings.push_back(runtimeFloatSetting("camera.fovDegrees", "FOV", fovDegrees, 1.0f, 179.0f, invalidateHistory));
}

struct RenderGraphBufferUserPush {
    uint32_t inputBuffer = 0;
    uint32_t outputBuffer = 0;
    uint32_t passIndex = 0;
    uint32_t padding = 0;
};

struct SceneGpuTransform {
    float world[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
};

inline std::vector<SceneGpuTransform> buildSceneGpuTransforms(const scene::Scene& loadedScene)
{
    std::vector<SceneGpuTransform> transforms(loadedScene.renderNodes().size());
    for (size_t index = 0; index < loadedScene.renderNodes().size(); ++index) {
        std::memcpy(
            transforms[index].world,
            loadedScene.renderNodes()[index].worldMatrix.a,
            sizeof(transforms[index].world));
    }
    return transforms;
}

struct BunnyWireframeGpuPosition {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct BunnyWireframeGpuParams {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    float clearColor[4] = {};
    float wireColor[4] = {};
    float settings[4] = {};
};

struct BunnyWireframeUserPush {
    uint32_t paramsBuffer = 0;
    uint32_t positionBuffer = 0;
    uint32_t transformBuffer = 0;
    uint32_t padding = 0;
};

struct MaterialShaderObjectGpuPosition {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct MaterialShaderObjectGpuMaterial {
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

struct MaterialShaderObjectGpuParams {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
};

struct MaterialShaderObjectUserPush {
    uint32_t positionBuffer = 0;
    uint32_t materialIndexBuffer = 0;
    uint32_t materialBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t vertexOffset = 0;
    uint32_t materialVariant = 0;
    uint32_t transformBuffer = 0;
    uint32_t padding = 0;
};

struct MaterialShaderObjectBatch {
    uint32_t materialIndex = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
};

struct SceneRayQueryVisualizationPush {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    uint32_t mode = kRayQueryVisualizationGranularityInstance;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t padding = 0;
};

struct GPUDrivenPreviewGpuVertex {
    float position[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float normal[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    float tangent[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float texcoord[4] = {};
};

struct GPUDrivenPreviewGpuTextureInfo {
    uint32_t textureIndex = UINT32_MAX;
    uint32_t texCoord = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    float transform0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float transform1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
};

struct GPUDrivenPreviewGpuMaterial {
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float emissive[4] = {};
    float params[4] = {1.0f, 1.0f, 0.5f, 0.0f};
    float textureParams[4] = {1.0f, 1.0f, 0.0f, 0.0f};
    float glassParams[4] = {0.0f, 1.5f, 0.0f, 0.0f};
    float attenuationColor[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    float diffuseTransmission[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    GPUDrivenPreviewGpuTextureInfo baseColorTexture;
    GPUDrivenPreviewGpuTextureInfo metallicRoughnessTexture;
    GPUDrivenPreviewGpuTextureInfo normalTexture;
    GPUDrivenPreviewGpuTextureInfo occlusionTexture;
    GPUDrivenPreviewGpuTextureInfo emissiveTexture;
    GPUDrivenPreviewGpuTextureInfo transmissionTexture;
    GPUDrivenPreviewGpuTextureInfo thicknessTexture;
    GPUDrivenPreviewGpuTextureInfo diffuseTransmissionTexture;
    GPUDrivenPreviewGpuTextureInfo diffuseTransmissionColorTexture;
    uint32_t identity[4] = {};
};

struct GPUDrivenPreviewGpuMeshlet {
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleOffset = 0;
    uint32_t triangleCount = 0;
    uint32_t lodLevel = 0;
    uint32_t lodGroupIndex = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    float boundingSphere[4] = {};
    float coneApexCutoff[4] = {};
    float coneAxis[4] = {};
};

// Per-instance draw metadata. Geometry payload, including meshlet bounds and
// topology offsets, lives in GPUDrivenPreviewGpuMeshlet and is uploaded once.
struct GPUDrivenPreviewGpuMeshletDraw {
    uint32_t geometryMeshletIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t transformIndex = 0;
    uint32_t instanceIndex = 0;
    uint32_t flags = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct GPUDrivenPreviewGeometrySource {
    const scene::RenderPrimitive* primitive = nullptr;
    uint32_t renderPrimitiveIndex = 0;
};

struct GPUDrivenPreviewGeometryDedupPlan {
    std::vector<uint32_t> geometryIndices;
    uint32_t geometryCount = 0;
    uint32_t conflictingPayloadCount = 0;
};

inline uint64_t gpuDrivenPreviewGeometryKey(
    const scene::RenderPrimitive& primitive,
    uint32_t renderPrimitiveIndex)
{
    if (primitive.meshIndex >= 0 && primitive.primitiveIndex >= 0) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(primitive.meshIndex)) << 32u) |
            static_cast<uint32_t>(primitive.primitiveIndex);
    }
    return (1ull << 63u) | renderPrimitiveIndex;
}

template <typename T>
inline bool gpuDrivenPreviewEqualVectorBytes(
    const std::vector<T>& left,
    const std::vector<T>& right)
{
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0);
}

inline bool gpuDrivenPreviewMatchingGeometryPayload(
    const scene::RenderPrimitive& left,
    const scene::RenderPrimitive& right)
{
    return left.mode == right.mode &&
        left.vertexCount == right.vertexCount &&
        left.indexCount == right.indexCount &&
        left.triangleCount == right.triangleCount &&
        left.localBounds.valid == right.localBounds.valid &&
        std::memcmp(&left.localBounds.min, &right.localBounds.min, sizeof(left.localBounds.min)) == 0 &&
        std::memcmp(&left.localBounds.max, &right.localBounds.max, sizeof(left.localBounds.max)) == 0 &&
        left.hasAuthoredNormals == right.hasAuthoredNormals &&
        left.hasAuthoredTangents == right.hasAuthoredTangents &&
        gpuDrivenPreviewEqualVectorBytes(left.positions, right.positions) &&
        gpuDrivenPreviewEqualVectorBytes(left.normals, right.normals) &&
        gpuDrivenPreviewEqualVectorBytes(left.tangents, right.tangents) &&
        gpuDrivenPreviewEqualVectorBytes(left.texcoords0, right.texcoords0) &&
        gpuDrivenPreviewEqualVectorBytes(left.indices, right.indices) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletClusters, right.meshletClusters) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletVertices, right.meshletVertices) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletTriangles, right.meshletTriangles) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletLodLevels, right.meshletLodLevels) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletLodGroups, right.meshletLodGroups) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletLodClusters, right.meshletLodClusters) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletLodVertices, right.meshletLodVertices) &&
        gpuDrivenPreviewEqualVectorBytes(left.meshletLodTriangles, right.meshletLodTriangles);
}

inline GPUDrivenPreviewGeometryDedupPlan buildGPUDrivenPreviewGeometryDedupPlan(
    std::span<const GPUDrivenPreviewGeometrySource> sources)
{
    struct Candidate {
        const scene::RenderPrimitive* primitive = nullptr;
        uint32_t geometryIndex = 0;
    };

    GPUDrivenPreviewGeometryDedupPlan result;
    result.geometryIndices.resize(sources.size(), UINT32_MAX);
    std::unordered_map<uint64_t, std::vector<Candidate>> candidatesByKey;
    for (size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex) {
        const GPUDrivenPreviewGeometrySource& source = sources[sourceIndex];
        if (source.primitive == nullptr) {
            continue;
        }
        const uint64_t key = gpuDrivenPreviewGeometryKey(
            *source.primitive,
            source.renderPrimitiveIndex);
        std::vector<Candidate>& candidates = candidatesByKey[key];
        for (const Candidate& candidate : candidates) {
            if (gpuDrivenPreviewMatchingGeometryPayload(*candidate.primitive, *source.primitive)) {
                result.geometryIndices[sourceIndex] = candidate.geometryIndex;
                break;
            }
        }
        if (result.geometryIndices[sourceIndex] != UINT32_MAX) {
            continue;
        }
        if (!candidates.empty()) {
            ++result.conflictingPayloadCount;
        }
        const uint32_t geometryIndex = result.geometryCount++;
        candidates.push_back(Candidate{
            .primitive = source.primitive,
            .geometryIndex = geometryIndex,
        });
        result.geometryIndices[sourceIndex] = geometryIndex;
    }
    return result;
}

struct GPUDrivenPreviewGpuInstance {
    float boundingSphere[4] = {};
    uint32_t transformIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t flags = kGPUDrivenPreviewInstanceVisible;
    uint32_t padding1 = 0;
};

struct GPUDrivenPreviewGpuParams {
    // Culling camera. This matches the render camera unless culling is frozen.
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    float clearColor[4] = {};
    float previousEye[4] = {};
    float previousCenter[4] = {};
    float previousUpProjection[4] = {};
    float previousViewport[4] = {};
    float previousClipOrtho[4] = {};
    // Viewport camera used only to project the surviving meshlets.
    float renderEye[4] = {};
    float renderCenter[4] = {};
    float renderUpProjection[4] = {};
    float renderViewport[4] = {};
    float renderClipOrtho[4] = {};
    uint32_t mode = kGPUDrivenPreviewModeMeshlet;
    uint32_t meshletOffset = 0;
    uint32_t meshletCount = 0;
    uint32_t selectedLodLevel = 0;
    uint32_t instanceCount = 0;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t hzbMipCount = 1;
    uint32_t frameIndex = 0;
    uint32_t hzbValid = 0;
    uint32_t cullingFlags =
        kGPUDrivenPreviewCullInstanceFrustum |
        kGPUDrivenPreviewCullInstanceHzb |
        kGPUDrivenPreviewCullMeshletFrustum |
        kGPUDrivenPreviewCullMeshletNormalCone;
    uint32_t materialTextureCount = 0;
    float environmentIntensity = 1.0f;
    float environmentRotationRadians = 0.0f;
    uint32_t environmentMode = 2;
    uint32_t environmentVisible = 1;
    uint32_t materialCount = 1;
    uint32_t visibleMeshletCapacity = 0;
    uint32_t shadingPadding1 = 0;
    uint32_t shadingPadding2 = 0;
};

struct GPUDrivenPreviewUserPush {
    uint32_t positionBuffer = 0;
    uint32_t meshletBuffer = 0;
    uint32_t meshletDrawBuffer = 0;
    uint32_t meshletVertexBuffer = 0;
    uint32_t meshletTriangleBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t transformBuffer = 0;
    uint32_t instanceBuffer = 0;
    uint32_t instanceVisibilityBuffer = 0;
    uint32_t visibleInstanceIdsBuffer = 0;
    uint32_t visibleInstanceCounterBuffer = 0;
    uint32_t visibleMeshletBuffer0 = 0;
    uint32_t visibleMeshletBuffer1 = 0;
    uint32_t indirectBuffer0 = 0;
    uint32_t indirectBuffer1 = 0;
    uint32_t hzbBuffer0 = 0;
    uint32_t hzbBuffer1 = 0;
    uint32_t deferredColorBuffer = 0;
    uint32_t depthImage = 0;
    uint32_t visibilityImage = 0;
    uint32_t passIndex = 0;
    uint32_t mipLevel = 0;
    uint32_t projectWithCullingCamera = 0;
    uint32_t materialBuffer = 0;
    uint32_t materialTextureRemapBuffer = 0;
    uint32_t environmentImage = 0;
    uint32_t environmentSHBuffer = 0;
    uint32_t streamDeferredBindingsBuffer = std::numeric_limits<uint32_t>::max();
    uint32_t residentRecordCapacity = 0;
    uint32_t streamOwnerMaskBuffer = std::numeric_limits<uint32_t>::max();
};

static_assert(sizeof(GPUDrivenPreviewGpuVertex) == 64);
static_assert(sizeof(GPUDrivenPreviewGpuTextureInfo) == 48);
static_assert(sizeof(GPUDrivenPreviewGpuMaterial) == sizeof(GPUSceneGpuMaterialRecord));
static_assert(offsetof(GPUDrivenPreviewGpuMaterial, identity) ==
    offsetof(GPUSceneGpuMaterialRecord, identity));
static_assert(sizeof(GPUDrivenPreviewGpuMeshlet) == 80);
static_assert(sizeof(GPUDrivenPreviewGpuMeshletDraw) == 32);
static_assert(sizeof(GPUDrivenPreviewGpuInstance) == 32);
static_assert(sizeof(GPUDrivenPreviewGpuVertex) == sizeof(GPUSceneGpuVertexRecord));
static_assert(sizeof(GPUDrivenPreviewGpuMeshlet) == sizeof(GPUSceneGpuMeshletRecord));
static_assert(sizeof(GPUSceneGpuMeshletDrawRecord) == 16);
static_assert(sizeof(GPUSceneGpuInstanceRecord) == 160);
static_assert(sizeof(GPUSceneGpuGeometryRecord) == 96);
static_assert(sizeof(GPUDrivenPreviewGpuParams) == 336);
static_assert(sizeof(GPUDrivenPreviewUserPush) == 120);

struct SceneMaterialVisualizationPush {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t mode = kSceneMaterialVisualizationModeMaterial;
    uint32_t materialTextureCount = 0;
    float bitangentFlip = 1.0f;
    uint32_t ntcTextureSetCount = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

struct ScenePathTracePush {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    float previousEye[4] = {};
    float previousCenter[4] = {};
    float previousUpProjection[4] = {};
    float previousViewport[4] = {};
    float previousClipOrtho[4] = {};
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t maxDepth = kDefaultPathTraceMaxDepth;
    uint32_t samples = kDefaultPathTraceSamples;
    uint32_t accumulationFrame = 0;
    uint32_t hasHistory = 0;
    uint32_t enableAccumulation = 1;
    uint32_t previousCameraValid = 0;
    uint32_t materialTextureCount = 0;
    float bitangentFlip = 1.0f;
    float environmentIntensity = 1.0f;
    float environmentRotationRadians = 0.0f;
    uint32_t environmentMode = kScenePathTraceEnvironmentModeProcedural;
    uint32_t environmentVisible = 1;
    // When set, the shader writes linear HDR and a follow-up pass tonemaps
    // (used by the NRC query permutation whose resolve adds radiance first).
    uint32_t outputLinear = 0;
    uint32_t cacheMode = kScenePathTraceCacheModeOff;
    uint32_t ntcTextureSetCount = 0;
    uint32_t debugView = kScenePathTraceDebugViewFinal;
    uint32_t debugFlags = 0;
};

// Per-frame parameters for the radiance-cache permutations of
// ScenePathTrace.slang (binding kScenePathTraceCacheParamsBinding). Layout
// must match struct ScenePathTraceCacheParams in the shader byte for byte;
// nrc mirrors ::NrcConstants from the NRC SDK headers.
struct ScenePathTraceCacheParams {
    float sharcCameraPosition[4] = {};
    float sharcCameraPositionPrev[4] = {};
    float sharcSceneScale = 1.0f;
    uint32_t sharcEntriesNum = 0;
    uint32_t sharcAccumulationFrameNum = 128;
    uint32_t sharcStaleFrameNumMax = 64;
    uint32_t frameIndex = 0;
    uint32_t cacheMode = kScenePathTraceCacheModeOff;
    uint32_t sharcUpdateStride = 5;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t trainingWidth = 1;
    uint32_t trainingHeight = 1;
    // ::NrcConstants (96 bytes)
    uint32_t nrcFrameDimensions[2] = {};
    uint32_t nrcTrainingDimensions[2] = {};
    float nrcScenePosScale[3] = {};
    uint32_t nrcSamplesPerPixel = 1;
    float nrcScenePosBias[3] = {};
    uint32_t nrcMaxPathVertices = 8;
    uint32_t nrcLearnIrradiance = 0;
    uint32_t nrcRadianceCacheDirect = 0;
    float nrcRadianceUnpackMultiplier = 1.0f;
    int32_t nrcResolveMode = 0;
    uint32_t nrcEnableTerminationHeuristic = 1;
    uint32_t nrcSkipDeltaVertices = 0;
    float nrcTerminationHeuristicThreshold = 0.1f;
    float nrcTrainingTerminationHeuristicThreshold = 0.1f;
    float nrcProportionUnbiased = 0.0625f;
    uint32_t nrcPad0 = 0;
    uint32_t nrcPad1 = 0;
    uint32_t nrcPad2 = 0;
};

static_assert(sizeof(ScenePathTraceCacheParams) == 172);
static_assert(offsetof(ScenePathTraceCacheParams, nrcFrameDimensions) == 76);
static_assert(sizeof(ScenePathTracePush) == 236);

struct SceneRtxdiPush {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    float previousEye[4] = {};
    float previousCenter[4] = {};
    float previousUpProjection[4] = {};
    float previousViewport[4] = {};
    float previousClipOrtho[4] = {};
    float sceneCenterRadius[4] = {};
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t frameIndex = 0;
    uint32_t hasHistory = 0;
    uint32_t lightCount = kDefaultRtxdiLightCount;
    uint32_t initialSampleCount = kDefaultRtxdiInitialSamples;
    uint32_t spatialSampleCount = kDefaultRtxdiSpatialSamples;
    uint32_t maxHistoryLength = 20;
    uint32_t behaviorFlags =
        kRtxdiBehaviorTemporalReuse |
        kRtxdiBehaviorSpatialReuse |
        kRtxdiBehaviorAnimateLights |
        kRtxdiBehaviorInitialVisibility |
        kRtxdiBehaviorLocalLightImportance |
        kRtxdiBehaviorEnvironmentVisible |
        kRtxdiBehaviorEnvironmentImportance;
    float environmentIntensity = 1.0f;
    float environmentRotationRadians = 0.0f;
    uint32_t environmentSampleCount = 0;
    uint32_t materialTextureCount = 0;
    float bitangentFlip = 1.0f;
    float lightIntensity = 12.0f;
    float exposure = 1.0f;
    float normalThreshold = 0.6f;
    float depthThreshold = 0.08f;
    uint32_t ntcTextureSetCount = 0;
    uint32_t padding2 = 0;
};

static_assert(sizeof(SceneRtxdiPush) == 256);

struct RtxdiConfidencePush {
    uint32_t mode = 0;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t gradientWidth = 1;
    uint32_t gradientHeight = 1;
    uint32_t hasHistory = 0;
    uint32_t filterStep = 1;
    uint32_t padding0 = 0;
    float darknessBias = 0.000244140625f;
    float sensitivity = 8.0f;
    float blendFactor = 1.0f;
    float padding1 = 0.0f;
};

static_assert(sizeof(RtxdiConfidencePush) == 48);

struct RtxdiCompositePush {
    uint32_t width = 1;
    uint32_t height = 1;
    float exposure = 1.0f;
    uint32_t padding = 0;
};

static_assert(sizeof(RtxdiCompositePush) == 16);

inline std::string resultMessage(std::string_view label, const Result& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

inline bool boolProperty(const RenderGraphProperties* properties, const char* key, bool fallback)
{
    if (properties == nullptr) {
        return fallback;
    }
    auto iter = properties->find(key);
    return iter != properties->end() && iter->is_boolean() ? iter->get<bool>() : fallback;
}

inline bool cameraUsesReversedZ(const RenderGraphProperties* camera)
{
    return boolProperty(camera, "reversedZ", kDefaultReversedZ);
}

inline float depthClearValue(bool reversedZ)
{
    return reversedZ ? 0.0f : 1.0f;
}

inline CompareOp depthCompareOp(bool reversedZ)
{
    return reversedZ ? CompareOp::GreaterEqual : CompareOp::LessEqual;
}

inline Result createSlangShaderModule(
    Device& device,
    const char* moduleName,
    const char* entryPointName,
    std::unique_ptr<ShaderModule>& outShaderModule,
    std::string& log)
{
    ShaderCompileResult compileResult;
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        },
        compileResult);
    if (!result) {
        log += "compileSlangShaderToSpirv(";
        log += moduleName;
        log += ".";
        log += entryPointName;
        log += ") returned ";
        log += resultToString(result);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        log += '\n';
        return result;
    }

    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
        },
        outShaderModule);
    if (!result) {
        log += resultMessage("createShaderModule", result);
        log += '\n';
    }
    return result;
}

inline Result compileSlangShader(
    const char* moduleName,
    const char* entryPointName,
    ShaderCompileResult& outCompileResult,
    std::string& log)
{
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        },
        outCompileResult);
    if (!result) {
        log += "compileSlangShaderToSpirv(";
        log += moduleName;
        log += ".";
        log += entryPointName;
        log += ") returned ";
        log += resultToString(result);
        if (!outCompileResult.diagnostics.empty()) {
            log += ": ";
            log += outCompileResult.diagnostics;
        }
        log += '\n';
    }
    return result;
}

} // namespace metallic::render::builtin_pass
