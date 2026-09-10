/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NrdPlan.h"

#include "Shaders/Libraries/Denoising/NRD/RELAX_Config.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_AntiFirefly.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_Atrous.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_AtrousSmem.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_ClassifyTiles.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_Copy.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_HistoryClamping.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_HistoryFix.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_HitDistReconstruction.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_PrePass.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_SplitScreen.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_TemporalAccumulation.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/RELAX_Validation.resources.hlsli"

// Permutations
#define RELAX_HITDIST_RECONSTRUCTION_PERMUTATION_NUM 2
#define RELAX_PREPASS_PERMUTATION_NUM 2
#define RELAX_TEMPORAL_ACCUMULATION_PERMUTATION_NUM 4
#define RELAX_ATROUS_PERMUTATION_NUM 2 // * RELAX_ATROUS_BINDING_VARIANT_NUM

// Other
#define RELAX_DUMMY AsUint(ResourceType::IN_VIEWZ)
#define RELAX_NO_PERMUTATIONS 1
#define RELAX_ATROUS_BINDING_VARIANT_NUM 5

constexpr uint32_t RELAX_MAX_ATROUS_PASS_NUM = 8;

#define RELAX_ADD_VALIDATION_DISPATCH                                                                                  \
    NRD_PASS("Validation");                                                                                            \
    {                                                                                                                  \
        readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));                                                        \
        readTexture(AsUint(ResourceType::IN_VIEWZ));                                                                   \
        readTexture(AsUint(ResourceType::IN_MV));                                                                      \
        readTexture(AsUint(Transient::HISTORY_LENGTH));                                                                \
        writeTexture(AsUint(ResourceType::OUT_VALIDATION));                                                            \
        std::array<ShaderDefine, 0> defines = {};                                                                      \
        NRD_STAGE_SIZED(RELAX_Validation, defines, IGNORE_RS);                                                         \
    }

inline float3 RELAX_GetFrustumForward(const float4x4& viewToWorld, const float4& frustum)
{
    float4 frustumForwardView = float4(0.5f, 0.5f, 1.0f, 0.0f) * float4(frustum.z, frustum.w, 1.0f, 0.0f) +
                                float4(frustum.x, frustum.y, 0.0f, 0.0f);
    float3 frustumForwardWorld = (viewToWorld * frustumForwardView).xyz;

    // Vector is not normalized for non-symmetric projections, it has to have .z = 1.0 to correctly reconstruct world
    // position in shaders
    return frustumForwardWorld;
}

void metallic::render::denoising::NrdPlan::writeRelaxConstants(const RelaxSettings& settings, void* data)
{
    struct SharedConstants {
        RELAX_SHARED_CONSTANTS
    };

    NRD_DECLARE_DIMS;

    float tanHalfFov = 1.0f / viewToClip_.a00;
    float aspect = viewToClip_.a00 / viewToClip_.a11;
    float3 frustumRight = worldToView_.Row(0).xyz * tanHalfFov;
    float3 frustumUp = worldToView_.Row(1).xyz * tanHalfFov * aspect;
    float3 frustumForward = RELAX_GetFrustumForward(viewToWorld_, frustum_);

    float prevTanHalfFov = 1.0f / viewToClipPrev_.a00;
    float prevAspect = viewToClipPrev_.a00 / viewToClipPrev_.a11;
    float3 prevFrustumRight = worldToViewPrev_.Row(0).xyz * prevTanHalfFov;
    float3 prevFrustumUp = worldToViewPrev_.Row(1).xyz * prevTanHalfFov * prevAspect;
    float3 prevFrustumForward = RELAX_GetFrustumForward(viewToWorldPrev_, frustumPrev_);

    float maxDiffuseLuminanceRelativeDifference = -log(saturate(settings.diffuseMinLuminanceWeight));
    float maxSpecularLuminanceRelativeDifference = -log(saturate(settings.specularMinLuminanceWeight));
    float disocclusionThresholdBonus = (1.0f + jitterDelta_) / float(rectH);
    bool isHistoryReset = commonSettings_.accumulationMode != AccumulationMode::CONTINUE;

    // Checkerboard logic
    uint32_t specCheckerboard = 2;
    uint32_t diffCheckerboard = 2;
    switch (settings.checkerboardMode) {
    case CheckerboardMode::BLACK:
        diffCheckerboard = 0;
        specCheckerboard = 1;
        break;
    case CheckerboardMode::WHITE:
        diffCheckerboard = 1;
        specCheckerboard = 0;
        break;
    default:
        break;
    }

    SharedConstants* consts = (SharedConstants*)data;
    consts->gWorldToClip = worldToClip_;
    consts->gWorldToClipPrev = worldToClipPrev_;
    consts->gWorldToViewPrev = worldToViewPrev_;
    consts->gWorldPrevToWorld = worldPrevToWorld_;
    consts->gRotatorPre = rotatorPre_;
    consts->gFrustumRight = float4(frustumRight, 0.0f);
    consts->gFrustumUp = float4(frustumUp, 0.0f);
    consts->gFrustumForward = float4(frustumForward, 0.0f);
    consts->gPrevFrustumRight = float4(prevFrustumRight, 0.0f);
    consts->gPrevFrustumUp = float4(prevFrustumUp, 0.0f);
    consts->gPrevFrustumForward = float4(prevFrustumForward, 0.0f);
    consts->gCameraDelta = float4(cameraDelta_, 0.0f);
    consts->gMvScale =
        float4(commonSettings_.motionVectorScale[0], commonSettings_.motionVectorScale[1],
               commonSettings_.motionVectorScale[2], commonSettings_.isMotionVectorInWorldSpace ? 1.0f : 0.0f);
    consts->gJitter = float2(commonSettings_.cameraJitter[0], commonSettings_.cameraJitter[1]);
    consts->gResolutionScale = float2(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    consts->gRectOffset = float2(float(commonSettings_.rectOrigin[0]) / float(resourceW),
                                 float(commonSettings_.rectOrigin[1]) / float(resourceH));
    consts->gResourceSizeInv = float2(1.0f / resourceW, 1.0f / resourceH);
    consts->gResourceSize = float2(resourceW, resourceH);
    consts->gRectSizeInv = float2(1.0f / rectW, 1.0f / rectH);
    consts->gRectSizePrev = float2(float(rectWprev), float(rectHprev));
    consts->gResourceSizeInvPrev = float2(1.0f / resourceWprev, 1.0f / resourceHprev);
    consts->gPrintfAt = uint2(commonSettings_.printfAt[0], commonSettings_.printfAt[1]);
    consts->gRectOrigin = uint2(commonSettings_.rectOrigin[0], commonSettings_.rectOrigin[1]);
    consts->gRectSize = int2(rectW, rectH);
    consts->gSpecMaxAccumulatedFrameNum =
        isHistoryReset ? 0.0f : (float)min(settings.specularMaxAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gSpecMaxFastAccumulatedFrameNum =
        isHistoryReset ? 0.0f : (float)min(settings.specularMaxFastAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gDiffMaxAccumulatedFrameNum =
        isHistoryReset ? 0.0f : (float)min(settings.diffuseMaxAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gDiffMaxFastAccumulatedFrameNum =
        isHistoryReset ? 0.0f : (float)min(settings.diffuseMaxFastAccumulatedFrameNum, RELAX_MAX_HISTORY_FRAME_NUM);
    consts->gDisocclusionThreshold = commonSettings_.disocclusionThreshold + disocclusionThresholdBonus;
    consts->gDisocclusionThresholdAlternate =
        commonSettings_.disocclusionThresholdAlternate + disocclusionThresholdBonus;
    consts->gCameraAttachedReflectionMaterialID = commonSettings_.cameraAttachedReflectionMaterialID;
    consts->gStrandMaterialID = commonSettings_.strandMaterialID;
    consts->gStrandThickness = commonSettings_.strandThickness;
    consts->gRoughnessFraction = settings.roughnessFraction;
    consts->gSpecVarianceBoost = settings.specularVarianceBoost;
    consts->gSplitScreen = commonSettings_.splitScreen;
    consts->gDiffBlurRadius = settings.diffusePrepassBlurRadius;
    consts->gSpecBlurRadius = settings.specularPrepassBlurRadius;
    consts->gDepthThreshold = settings.depthThreshold;
    consts->gLobeAngleFraction = settings.lobeAngleFraction;
    consts->gSpecLobeAngleSlack = radians(settings.specularLobeAngleSlack);
    consts->gHistoryFixEdgeStoppingNormalPower = settings.historyFixEdgeStoppingNormalPower;
    consts->gRoughnessEdgeStoppingRelaxation = settings.roughnessEdgeStoppingRelaxation;
    consts->gNormalEdgeStoppingRelaxation = settings.normalEdgeStoppingRelaxation;
    consts->gFastHistoryClampingSigmaScale = settings.fastHistoryClampingSigmaScale;
    consts->gHistoryAccelerationAmount = settings.antilagSettings.accelerationAmount;
    consts->gHistoryResetTemporalSigmaScale = settings.antilagSettings.temporalSigmaScale;
    consts->gHistoryResetSpatialSigmaScale = settings.antilagSettings.spatialSigmaScale;
    consts->gHistoryResetAmount = settings.antilagSettings.resetAmount;
    consts->gDenoisingRange = commonSettings_.denoisingRange;
    consts->gSpecPhiLuminance = settings.specularPhiLuminance;
    consts->gDiffPhiLuminance = settings.diffusePhiLuminance;
    consts->gDiffMaxLuminanceRelativeDifference = maxDiffuseLuminanceRelativeDifference;
    consts->gSpecMaxLuminanceRelativeDifference = maxSpecularLuminanceRelativeDifference;
    consts->gLuminanceEdgeStoppingRelaxation = settings.roughnessEdgeStoppingRelaxation;
    consts->gConfidenceDrivenRelaxationMultiplier = settings.confidenceDrivenRelaxationMultiplier;
    consts->gConfidenceDrivenLuminanceEdgeStoppingRelaxation = settings.confidenceDrivenLuminanceEdgeStoppingRelaxation;
    consts->gConfidenceDrivenNormalEdgeStoppingRelaxation = settings.confidenceDrivenNormalEdgeStoppingRelaxation;
    consts->gDebug = commonSettings_.debug;
    consts->gOrthoMode = orthoMode_;
    consts->gUnproject = 1.0f / (0.5f * rectH * projectY_);
    consts->gFramerateScale = clamp(16.66f / timeDelta_, 0.25f, 4.0f); // TODO: use frameRateScale_?
    consts->gCheckerboardResolveAccumSpeed = checkerboardResolveAccumSpeed_;
    consts->gJitterDelta = jitterDelta_;
    consts->gHistoryFixFrameNum = settings.historyFixFrameNum + 1.0f;
    consts->gHistoryFixBasePixelStride = (float)settings.historyFixBasePixelStride;
    consts->gHistoryFixAlternatePixelStride = (float)settings.historyFixAlternatePixelStride;
    consts->gHistoryFixAlternatePixelStrideMaterialID = commonSettings_.historyFixAlternatePixelStrideMaterialID;
    consts->gHistoryThreshold = (float)settings.spatialVarianceEstimationHistoryThreshold;
    consts->gViewZScale = commonSettings_.viewZScale;
    consts->gMinHitDistanceWeight =
        settings.minHitDistanceWeight *
        2.0f; // TODO: 2 to match REBLUR units and make Pre passes identical (matches old default)
    consts->gDiffMinMaterial = settings.minMaterialForDiffuse;
    consts->gSpecMinMaterial = settings.minMaterialForSpecular;
    consts->gRoughnessEdgeStoppingEnabled = settings.enableRoughnessEdgeStopping ? 1 : 0;
    consts->gFrameIndex = commonSettings_.frameIndex;
    consts->gDiffCheckerboard = diffCheckerboard;
    consts->gSpecCheckerboard = specCheckerboard;
    consts->gHasHistoryConfidence = commonSettings_.isHistoryConfidenceAvailable ? 1 : 0;
    consts->gHasDisocclusionThresholdMix = commonSettings_.isDisocclusionThresholdMixAvailable ? 1 : 0;
    consts->gResetHistory = isHistoryReset ? 1 : 0;
}

void metallic::render::denoising::NrdPlan::updateRelax(const DenoiserData& denoiserData)
{
    enum class Dispatch {
        CLASSIFY_TILES,
        HITDIST_RECONSTRUCTION = CLASSIFY_TILES + RELAX_NO_PERMUTATIONS,
        PREPASS = HITDIST_RECONSTRUCTION + RELAX_HITDIST_RECONSTRUCTION_PERMUTATION_NUM,
        TEMPORAL_ACCUMULATION = PREPASS + RELAX_PREPASS_PERMUTATION_NUM,
        HISTORY_FIX = TEMPORAL_ACCUMULATION + RELAX_TEMPORAL_ACCUMULATION_PERMUTATION_NUM,
        HISTORY_CLAMPING = HISTORY_FIX + RELAX_NO_PERMUTATIONS,
        COPY = HISTORY_CLAMPING + RELAX_NO_PERMUTATIONS,
        ANTI_FIREFLY = COPY + RELAX_NO_PERMUTATIONS,
        ATROUS = ANTI_FIREFLY + RELAX_NO_PERMUTATIONS,
        SPLIT_SCREEN = ATROUS + RELAX_ATROUS_PERMUTATION_NUM * RELAX_ATROUS_BINDING_VARIANT_NUM,
        VALIDATION = SPLIT_SCREEN + RELAX_NO_PERMUTATIONS,
    };

    NRD_DECLARE_DIMS;

    const RelaxSettings& settings = denoiserData.settings.relax;
    bool enableHitDistanceReconstruction =
        settings.hitDistanceReconstructionMode != HitDistanceReconstructionMode::OFF &&
        settings.checkerboardMode == CheckerboardMode::OFF;
    uint32_t iterationNum = clamp(settings.atrousIterationNum, 2u, RELAX_MAX_ATROUS_PASS_NUM);

    // SPLIT_SCREEN (passthrough)
    if (commonSettings_.splitScreen >= 1.0f) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        writeRelaxConstants(settings, consts);

        return;
    }

    { // CLASSIFY_TILES
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::CLASSIFY_TILES));
        writeRelaxConstants(settings, consts);
    }

    // HITDIST_RECONSTRUCTION
    if (enableHitDistanceReconstruction) {
        bool is5x5 = settings.hitDistanceReconstructionMode == HitDistanceReconstructionMode::AREA_5X5;
        uint32_t passIndex = AsUint(Dispatch::HITDIST_RECONSTRUCTION) + (is5x5 ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeRelaxConstants(settings, consts);
    }

    { // PREPASS
        uint32_t passIndex = AsUint(Dispatch::PREPASS) + (enableHitDistanceReconstruction ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeRelaxConstants(settings, consts);
    }

    { // TEMPORAL_ACCUMULATION
        uint32_t passIndex = AsUint(Dispatch::TEMPORAL_ACCUMULATION) +
                             (commonSettings_.isDisocclusionThresholdMixAvailable ? 2 : 0) +
                             (commonSettings_.isHistoryConfidenceAvailable ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeRelaxConstants(settings, consts);
    }

    { // HISTORY_FIX
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::HISTORY_FIX));
        writeRelaxConstants(settings, consts);
    }

    { // HISTORY_CLAMPING
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::HISTORY_CLAMPING));
        writeRelaxConstants(settings, consts);
    }

    if (settings.enableAntiFirefly) {
        { // COPY
            void* consts = emitDispatch(denoiserData, AsUint(Dispatch::COPY));
            writeRelaxConstants(settings, consts);
        }

        { // ANTI_FIREFLY
            void* consts = emitDispatch(denoiserData, AsUint(Dispatch::ANTI_FIREFLY));
            writeRelaxConstants(settings, consts);
        }
    }

    // A-TROUS
    for (uint32_t i = 0; i < iterationNum; i++) {
        uint32_t passIndex = AsUint(Dispatch::ATROUS) +
                             (commonSettings_.isHistoryConfidenceAvailable ? RELAX_ATROUS_BINDING_VARIANT_NUM : 0);
        if (i != 0)
            passIndex += 2 - (i & 0x1);
        if (i == iterationNum - 1)
            passIndex += 2;

        RELAX_AtrousConstants* consts = (RELAX_AtrousConstants*)emitDispatch(
            denoiserData, AsUint(passIndex)); // TODO: same as "RELAX_AtrousSmemConstants"
        writeRelaxConstants(settings, consts);
        consts->gStepSize = 1 << i;                          // TODO: push constant
        consts->gIsLastPass = i == iterationNum - 1 ? 1 : 0; // TODO: push constant
    }

    // SPLIT_SCREEN
    if (commonSettings_.splitScreen > 0.0f) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        writeRelaxConstants(settings, consts);
    }

    // VALIDATION
    if (commonSettings_.enableValidation) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::VALIDATION));
        writeRelaxConstants(settings, consts);
    }
}

/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME RELAX_DiffuseSpecular

void metallic::render::denoising::NrdPlan::addRelax(DenoiserData& denoiserData)
{
    denoiserData.settings.relax = RelaxSettings();

    enum class Permanent {
        SPEC_ILLUM_PREV = PERMANENT_POOL_START,
        DIFF_ILLUM_PREV,
        SPEC_ILLUM_RESPONSIVE_PREV,
        DIFF_ILLUM_RESPONSIVE_PREV,
        REFLECTION_HIT_T_CURR,
        REFLECTION_HIT_T_PREV,
        HISTORY_LENGTH_PREV,
        NORMAL_ROUGHNESS_PREV,
        MATERIAL_ID_PREV,
        VIEWZ_PREV,
    };

    addPermanent({Format::RGBA16_SFLOAT, 1});
    addPermanent({Format::RGBA16_SFLOAT, 1});
    addPermanent({Format::RGBA16_SFLOAT, 1});
    addPermanent({Format::RGBA16_SFLOAT, 1});
    addPermanent({Format::R16_SFLOAT, 1});
    addPermanent({Format::R16_SFLOAT, 1});
    addPermanent({Format::R8_UNORM, 1});
    addPermanent({Format::RGBA8_UNORM, 1});
    addPermanent({Format::R8_UNORM, 1});
    addPermanent({Format::R32_SFLOAT, 1});

    enum class Transient {
        SPEC_ILLUM_PING = TRANSIENT_POOL_START,
        SPEC_ILLUM_PONG,
        DIFF_ILLUM_PING,
        DIFF_ILLUM_PONG,
        SPEC_REPROJECTION_CONFIDENCE,
        TILES,
        HISTORY_LENGTH
    };

    addTransient({Format::RGBA16_SFLOAT, 1});
    addTransient({Format::RGBA16_SFLOAT, 1});
    addTransient({Format::RGBA16_SFLOAT, 1});
    addTransient({Format::RGBA16_SFLOAT, 1});
    addTransient({Format::R8_UNORM, 1});
    addTransient({Format::R8_UNORM, 16});
    addTransient({Format::R8_UNORM, 1});

    std::array<ShaderDefine, 2> commonDefines = {{
        {"NRD_SIGNAL", NRD_DIFFUSE_SPECULAR},
        {"NRD_MODE", NRD_RADIANCE},
    }};

    NRD_PASS("Classify tiles");
    {
        // Inputs
        readTexture(AsUint(ResourceType::IN_VIEWZ));

        // Outputs
        writeTexture(AsUint(Transient::TILES));

        // Shaders
        std::array<ShaderDefine, 0> defines = {};
        NRD_STAGE(RELAX_ClassifyTiles, defines);
    }

    for (int i = 0; i < RELAX_HITDIST_RECONSTRUCTION_PERMUTATION_NUM; i++) {
        bool is5x5 = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Hit distance reconstruction");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(ResourceType::IN_VIEWZ));
            readTexture(AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));
            readTexture(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));

            // Outputs
            writeTexture(AsUint(Transient::SPEC_ILLUM_PING));
            writeTexture(AsUint(Transient::DIFF_ILLUM_PING));

            // Shaders
            std::array<ShaderDefine, 3> defines = {{
                commonDefines[0],
                commonDefines[1],
                {"MODE_5X5", is5x5 ? "1" : "0"},
            }};
            NRD_STAGE(RELAX_HitDistReconstruction, defines);
        }
    }

    for (int i = 0; i < RELAX_PREPASS_PERMUTATION_NUM; i++) {
        bool isAfterReconstruction = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Pre-pass");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(ResourceType::IN_VIEWZ));
            readTexture(isAfterReconstruction ? AsUint(Transient::SPEC_ILLUM_PING)
                                              : AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));
            readTexture(isAfterReconstruction ? AsUint(Transient::DIFF_ILLUM_PING)
                                              : AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));

            // Outputs
            writeTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            writeTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

            // Shaders
            NRD_STAGE(RELAX_PrePass, commonDefines);
        }
    }

    for (int i = 0; i < RELAX_TEMPORAL_ACCUMULATION_PERMUTATION_NUM; i++) {
        bool hasDisocclusionThresholdMix = (((i >> 1) & 0x1) != 0);
        bool hasConfidenceInputs = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Temporal accumulation");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_MV));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(ResourceType::IN_VIEWZ));
            readTexture(hasDisocclusionThresholdMix ? AsUint(ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX)
                                                    : RELAX_DUMMY);
            readTexture(AsUint(Permanent::NORMAL_ROUGHNESS_PREV));
            readTexture(AsUint(Permanent::VIEWZ_PREV));
            readTexture(AsUint(Permanent::HISTORY_LENGTH_PREV));
            readTexture(AsUint(Permanent::MATERIAL_ID_PREV));
            readTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            readTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
            readTexture(AsUint(Permanent::SPEC_ILLUM_RESPONSIVE_PREV));
            readTexture(AsUint(Permanent::DIFF_ILLUM_RESPONSIVE_PREV));
            readTexture(AsUint(Permanent::SPEC_ILLUM_PREV));
            readTexture(AsUint(Permanent::DIFF_ILLUM_PREV));
            readTexture(AsUint(Permanent::REFLECTION_HIT_T_PREV), AsUint(Permanent::REFLECTION_HIT_T_CURR));
            readTexture(hasConfidenceInputs ? AsUint(ResourceType::IN_SPEC_CONFIDENCE) : RELAX_DUMMY);
            readTexture(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : RELAX_DUMMY);

            // Outputs
            writeTexture(AsUint(Transient::HISTORY_LENGTH));
            writeTexture(AsUint(Transient::SPEC_ILLUM_PING));
            writeTexture(AsUint(Transient::DIFF_ILLUM_PING));
            writeTexture(AsUint(Transient::SPEC_ILLUM_PONG));
            writeTexture(AsUint(Transient::DIFF_ILLUM_PONG));
            writeTexture(AsUint(Permanent::REFLECTION_HIT_T_CURR), AsUint(Permanent::REFLECTION_HIT_T_PREV));
            writeTexture(AsUint(Transient::SPEC_REPROJECTION_CONFIDENCE));

            // Shaders
            NRD_STAGE(RELAX_TemporalAccumulation, commonDefines);
        }
    }

    NRD_PASS("History fix");
    {
        // Inputs
        readTexture(AsUint(Transient::TILES));
        readTexture(AsUint(Transient::HISTORY_LENGTH));
        readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(Transient::SPEC_ILLUM_PING)); // Normal history
        readTexture(AsUint(Transient::DIFF_ILLUM_PING));

        // Outputs
        writeTexture(AsUint(Transient::SPEC_ILLUM_PONG)); // Responsive history
        writeTexture(AsUint(Transient::DIFF_ILLUM_PONG));

        // Shaders
        NRD_STAGE(RELAX_HistoryFix, commonDefines);
    }

    NRD_PASS("History clamping");
    {
        // Inputs
        readTexture(AsUint(Transient::TILES));
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(Transient::HISTORY_LENGTH));
        readTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST)); // Noisy input with preblur applied
        readTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
        readTexture(AsUint(Transient::SPEC_ILLUM_PING)); // Normal history
        readTexture(AsUint(Transient::DIFF_ILLUM_PING));
        readTexture(AsUint(Transient::SPEC_ILLUM_PONG)); // Responsive history
        readTexture(AsUint(Transient::DIFF_ILLUM_PONG));

        // Outputs
        writeTexture(AsUint(Permanent::HISTORY_LENGTH_PREV));
        writeTexture(AsUint(Permanent::SPEC_ILLUM_PREV));
        writeTexture(AsUint(Permanent::DIFF_ILLUM_PREV));
        writeTexture(AsUint(Permanent::SPEC_ILLUM_RESPONSIVE_PREV));
        writeTexture(AsUint(Permanent::DIFF_ILLUM_RESPONSIVE_PREV));

        // Shaders
        NRD_STAGE(RELAX_HistoryClamping, commonDefines);
    }

    NRD_PASS("Copy");
    {
        // Inputs
        readTexture(AsUint(Permanent::SPEC_ILLUM_PREV));
        readTexture(AsUint(Permanent::DIFF_ILLUM_PREV));

        // Outputs
        writeTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
        writeTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

        // Shaders
        NRD_STAGE(RELAX_Copy, commonDefines);
    }

    NRD_PASS("Anti-firefly");
    {
        // Inputs
        readTexture(AsUint(Transient::TILES));
        readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
        readTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));

        // Outputs
        writeTexture(AsUint(Permanent::SPEC_ILLUM_PREV));
        writeTexture(AsUint(Permanent::DIFF_ILLUM_PREV));

        // Shaders
        NRD_STAGE(RELAX_AntiFirefly, commonDefines);
    }

    for (int i = 0; i < RELAX_ATROUS_PERMUTATION_NUM; i++) {
        bool hasConfidenceInputs = (((i >> 0) & 0x1) != 0);

        for (int j = 0; j < RELAX_ATROUS_BINDING_VARIANT_NUM; j++) {
            bool isSmem = j == 0;
            bool isEven = j % 2 == 0;
            bool isLast = j > 2;

            if (isSmem)
                NRD_PASS("A-trous (SMEM)");
            else
                NRD_PASS("A-trous");

            {
                // Inputs
                readTexture(AsUint(Transient::TILES));
                readTexture(AsUint(Transient::HISTORY_LENGTH));
                readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
                readTexture(AsUint(ResourceType::IN_VIEWZ));

                if (isSmem) {
                    readTexture(AsUint(Permanent::SPEC_ILLUM_PREV));
                    readTexture(AsUint(Permanent::DIFF_ILLUM_PREV));
                } else {
                    readTexture(isEven ? AsUint(Transient::SPEC_ILLUM_PONG) : AsUint(Transient::SPEC_ILLUM_PING));
                    readTexture(isEven ? AsUint(Transient::DIFF_ILLUM_PONG) : AsUint(Transient::DIFF_ILLUM_PING));
                }

                readTexture(AsUint(Transient::SPEC_REPROJECTION_CONFIDENCE));
                readTexture(hasConfidenceInputs ? AsUint(ResourceType::IN_SPEC_CONFIDENCE) : RELAX_DUMMY);
                readTexture(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : RELAX_DUMMY);

                // Outputs
                if (isLast) {
                    writeTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
                    writeTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
                } else {
                    writeTexture(isEven ? AsUint(Transient::SPEC_ILLUM_PING) : AsUint(Transient::SPEC_ILLUM_PONG));
                    writeTexture(isEven ? AsUint(Transient::DIFF_ILLUM_PING) : AsUint(Transient::DIFF_ILLUM_PONG));
                }

                if (isSmem) {
                    writeTexture(AsUint(Permanent::NORMAL_ROUGHNESS_PREV));
                    writeTexture(AsUint(Permanent::MATERIAL_ID_PREV));
                    writeTexture(AsUint(Permanent::VIEWZ_PREV));
                }

                // Shaders
                if (isSmem)
                    NRD_STAGE(RELAX_AtrousSmem, commonDefines);
                else
                    NRD_STAGE(RELAX_Atrous, commonDefines);
            }
        }
    }

    NRD_PASS("Split screen");
    {
        // Inputs
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
        readTexture(AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));

        // Outputs
        writeTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
        writeTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));

        // Shaders
        NRD_STAGE(RELAX_SplitScreen, commonDefines);
    }

    RELAX_ADD_VALIDATION_DISPATCH;
}

#undef DENOISER_NAME
