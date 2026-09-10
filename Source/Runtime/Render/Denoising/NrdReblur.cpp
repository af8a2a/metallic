/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NrdPlan.h"

#include "Shaders/Libraries/Denoising/NRD/REBLUR_Config.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_Blur.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_ClassifyTiles.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_HistoryFix.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_HitDistReconstruction.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_PostBlur.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_PrePass.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_SplitScreen.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_TemporalAccumulation.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_TemporalStabilization.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REBLUR_Validation.resources.hlsli"

// Permutations
#define REBLUR_HITDIST_RECONSTRUCTION_PERMUTATION_NUM 4
#define REBLUR_PREPASS_PERMUTATION_NUM 2
#define REBLUR_TEMPORAL_ACCUMULATION_PERMUTATION_NUM 8
#define REBLUR_POST_BLUR_PERMUTATION_NUM 2
#define REBLUR_TEMPORAL_STABILIZATION_PERMUTATION_NUM 2

// Formats
#define REBLUR_FORMAT Format::RGBA16_SFLOAT           // .xyz - color, .w - normalized hit distance
#define REBLUR_FORMAT_FAST_HISTORY Format::R16_SFLOAT // .x - luminance
#define REBLUR_FORMAT_PREV_VIEWZ Format::R32_SFLOAT
#define REBLUR_FORMAT_PREV_INTERNAL_DATA Format::R16_UINT

#define REBLUR_FORMAT_TILES Format::R8_UNORM

#if (NRD_NORMAL_ENCODING == 0)
#define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS Format::RGBA8_UNORM
#elif (NRD_NORMAL_ENCODING == 1)
#define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS Format::RGBA8_SNORM
#elif (NRD_NORMAL_ENCODING == 2)
#define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS Format::R10_G10_B10_A2_UNORM
#elif (NRD_NORMAL_ENCODING == 3)
#define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS Format::RGBA16_UNORM
#elif (NRD_NORMAL_ENCODING == 4)
#define REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS Format::RGBA16_SFLOAT
#else
#error "'NRDConfig.h' not included"
#endif

#define REBLUR_FORMAT_HITDIST_FOR_TRACKING Format::R16_SFLOAT

// Other
#define REBLUR_DUMMY AsUint(ResourceType::IN_VIEWZ)
#define REBLUR_NO_PERMUTATIONS 1

#define REBLUR_ADD_VALIDATION_DISPATCH(data2, diff, spec)                                                              \
    NRD_PASS("Validation");                                                                                            \
    {                                                                                                                  \
        readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));                                                        \
        readTexture(AsUint(ResourceType::IN_VIEWZ));                                                                   \
        readTexture(AsUint(ResourceType::IN_MV));                                                                      \
        readTexture(AsUint(Transient::DATA1));                                                                         \
        readTexture(AsUint(data2));                                                                                    \
        readTexture(AsUint(diff));                                                                                     \
        readTexture(AsUint(spec));                                                                                     \
        writeTexture(AsUint(ResourceType::OUT_VALIDATION));                                                            \
        std::array<ShaderDefine, 0> defines = {};                                                                      \
        NRD_STAGE_SIZED(REBLUR_Validation, defines, IGNORE_RS);                                                        \
    }

void metallic::render::denoising::NrdPlan::updateReblur(const DenoiserData& denoiserData)
{
    enum class Dispatch {
        CLASSIFY_TILES,
        HITDIST_RECONSTRUCTION = CLASSIFY_TILES + REBLUR_NO_PERMUTATIONS,
        PREPASS = HITDIST_RECONSTRUCTION + REBLUR_HITDIST_RECONSTRUCTION_PERMUTATION_NUM,
        TEMPORAL_ACCUMULATION = PREPASS + REBLUR_PREPASS_PERMUTATION_NUM,
        HISTORY_FIX = TEMPORAL_ACCUMULATION + REBLUR_TEMPORAL_ACCUMULATION_PERMUTATION_NUM,
        BLUR = HISTORY_FIX + REBLUR_NO_PERMUTATIONS,
        POST_BLUR = BLUR + REBLUR_NO_PERMUTATIONS,
        TEMPORAL_STABILIZATION = POST_BLUR + REBLUR_POST_BLUR_PERMUTATION_NUM,
        SPLIT_SCREEN = TEMPORAL_STABILIZATION + REBLUR_TEMPORAL_STABILIZATION_PERMUTATION_NUM,
        VALIDATION = SPLIT_SCREEN + REBLUR_NO_PERMUTATIONS,
    };

    NRD_DECLARE_DIMS;

    const ReblurSettings& settings = denoiserData.settings.reblur;

    bool enableHitDistanceReconstruction =
        settings.hitDistanceReconstructionMode != HitDistanceReconstructionMode::OFF &&
        settings.checkerboardMode == CheckerboardMode::OFF;
    bool skipTemporalStabilization = settings.maxStabilizedFrameNum == 0;
    bool skipPrePass = (settings.diffusePrepassBlurRadius == 0.0f) && (settings.specularPrepassBlurRadius == 0.0f) &&
                       settings.checkerboardMode == CheckerboardMode::OFF;

    // SPLIT_SCREEN (passthrough)
    if (commonSettings_.splitScreen >= 1.0f) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        writeReblurConstants(settings, consts);

        return;
    }

    { // CLASSIFY_TILES
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::CLASSIFY_TILES));
        writeReblurConstants(settings, consts);
    }

    // HITDIST_RECONSTRUCTION
    if (enableHitDistanceReconstruction) {
        uint32_t passIndex =
            AsUint(Dispatch::HITDIST_RECONSTRUCTION) +
            (settings.hitDistanceReconstructionMode == HitDistanceReconstructionMode::AREA_5X5 ? 2 : 0) +
            (!skipPrePass ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeReblurConstants(settings, consts);
    }

    // PREPASS
    if (!skipPrePass) {
        uint32_t passIndex = AsUint(Dispatch::PREPASS) + (enableHitDistanceReconstruction ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeReblurConstants(settings, consts);
    }

    { // TEMPORAL_ACCUMULATION
        uint32_t passIndex = AsUint(Dispatch::TEMPORAL_ACCUMULATION) +
                             (commonSettings_.isDisocclusionThresholdMixAvailable ? 4 : 0) +
                             (commonSettings_.isHistoryConfidenceAvailable ? 2 : 0) +
                             ((!skipPrePass || enableHitDistanceReconstruction) ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeReblurConstants(settings, consts);
    }

    { // HISTORY_FIX
        uint32_t passIndex = AsUint(Dispatch::HISTORY_FIX);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeReblurConstants(settings, consts);
    }

    { // BLUR
        uint32_t passIndex = AsUint(Dispatch::BLUR);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeReblurConstants(settings, consts);
    }

    { // POST_BLUR
        uint32_t passIndex = AsUint(Dispatch::POST_BLUR) + (skipTemporalStabilization ? 0 : 1);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeReblurConstants(settings, consts);
    }

    // TEMPORAL_STABILIZATION
    if (!skipTemporalStabilization) {
        uint32_t passIndex =
            AsUint(Dispatch::TEMPORAL_STABILIZATION) + (commonSettings_.isBaseColorMetalnessAvailable ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeReblurConstants(settings, consts);
    }

    // SPLIT_SCREEN
    if (commonSettings_.splitScreen > 0.0f) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        writeReblurConstants(settings, consts);
    }

    // VALIDATION
    if (commonSettings_.enableValidation) {
        REBLUR_ValidationConstants* consts =
            (REBLUR_ValidationConstants*)emitDispatch(denoiserData, AsUint(Dispatch::VALIDATION));
        writeReblurConstants(settings, consts);
        consts->gHasDiffuse = 1;  // TODO: push constant
        consts->gHasSpecular = 1; // TODO: push constant
    }
}

void metallic::render::denoising::NrdPlan::writeReblurConstants(const ReblurSettings& settings, void* data)
{
    struct SharedConstants {
        REBLUR_SHARED_CONSTANTS
    };

    NRD_DECLARE_DIMS;

    bool isRectChanged = rectW != rectWprev || rectH != rectHprev;
    bool isHistoryReset = commonSettings_.accumulationMode != AccumulationMode::CONTINUE;
    float unproject = 1.0f / (0.5f * rectH * projectY_);
    float worstResolutionScale = min(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    float maxBlurRadius = settings.maxBlurRadius * worstResolutionScale;
    float diffusePrepassBlurRadius = settings.diffusePrepassBlurRadius * worstResolutionScale;
    float specularPrepassBlurRadius = settings.specularPrepassBlurRadius * worstResolutionScale;
    float disocclusionThresholdBonus = (1.0f + jitterDelta_) / float(rectH);
    float stabilizationStrength = settings.maxStabilizedFrameNum / (1.0f + settings.maxStabilizedFrameNum);
    uint32_t maxAccumulatedFrameNum = min(settings.maxAccumulatedFrameNum, REBLUR_MAX_HISTORY_FRAME_NUM);

    uint32_t diffCheckerboard = 2;
    uint32_t specCheckerboard = 2;
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
    consts->gViewToClip = viewToClip_;
    consts->gViewToWorld = viewToWorld_;
    consts->gWorldToViewPrev = worldToViewPrev_;
    consts->gWorldToClipPrev = worldToClipPrev_;
    consts->gWorldPrevToWorld = worldPrevToWorld_;
    consts->gRotatorPre = rotatorPre_;
    consts->gRotator = rotator_;
    consts->gRotatorPost = rotatorPost_;
    consts->gFrustum = frustum_;
    consts->gFrustumPrev = frustumPrev_;
    consts->gCameraDelta = cameraDelta_.xmm;
    consts->gHitDistParams = float4(settings.hitDistanceParameters.A, settings.hitDistanceParameters.B,
                                    settings.hitDistanceParameters.C, settings.hitDistanceParameters.D);
    consts->gViewVectorWorld = viewDirection_.xmm;
    consts->gViewVectorWorldPrev = viewDirectionPrev_.xmm;
    consts->gMvScale =
        float4(commonSettings_.motionVectorScale[0], commonSettings_.motionVectorScale[1],
               commonSettings_.motionVectorScale[2], commonSettings_.isMotionVectorInWorldSpace ? 1.0f : 0.0f);
    consts->gAntilagParams =
        float2(settings.antilagSettings.luminanceSigmaScale, settings.antilagSettings.luminanceSensitivity);
    consts->gResourceSize = float2(float(resourceW), float(resourceH));
    consts->gResourceSizeInv = float2(1.0f / float(resourceW), 1.0f / float(resourceH));
    consts->gResourceSizeInvPrev = float2(1.0f / float(resourceWprev), 1.0f / float(resourceHprev));
    consts->gRectSize = float2(float(rectW), float(rectH));
    consts->gRectSizeInv = float2(1.0f / float(rectW), 1.0f / float(rectH));
    consts->gRectSizePrev = float2(float(rectWprev), float(rectHprev));
    consts->gResolutionScale = float2(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    consts->gResolutionScalePrev =
        float2(float(rectWprev) / float(resourceWprev), float(rectHprev) / float(resourceHprev));
    consts->gRectOffset = float2(float(commonSettings_.rectOrigin[0]) / float(resourceW),
                                 float(commonSettings_.rectOrigin[1]) / float(resourceH));
    consts->gSpecProbabilityThresholdsForMvModification = float2(
        commonSettings_.isBaseColorMetalnessAvailable ? settings.specularProbabilityThresholdsForMvModification[0]
                                                      : 2.0f,
        commonSettings_.isBaseColorMetalnessAvailable ? settings.specularProbabilityThresholdsForMvModification[1]
                                                      : 3.0f);
    consts->gJitter = float2(commonSettings_.cameraJitter[0], commonSettings_.cameraJitter[1]);
    consts->gPrintfAt = uint2(commonSettings_.printfAt[0], commonSettings_.printfAt[1]);
    consts->gRectOrigin = uint2(commonSettings_.rectOrigin[0], commonSettings_.rectOrigin[1]);
    consts->gRectSizeMinusOne = int2(rectW - 1, rectH - 1);
    consts->gDisocclusionThreshold = commonSettings_.disocclusionThreshold + disocclusionThresholdBonus;
    consts->gDisocclusionThresholdAlternate =
        commonSettings_.disocclusionThresholdAlternate + disocclusionThresholdBonus;
    consts->gCameraAttachedReflectionMaterialID = commonSettings_.cameraAttachedReflectionMaterialID;
    consts->gStrandMaterialID = commonSettings_.strandMaterialID;
    consts->gStrandThickness = commonSettings_.strandThickness;
    consts->gStabilizationStrength = isHistoryReset ? 0.0f : stabilizationStrength;
    consts->gDebug = commonSettings_.debug;
    consts->gOrthoMode = orthoMode_;
    consts->gUnproject = unproject;
    consts->gDenoisingRange = commonSettings_.denoisingRange;
    consts->gPlaneDistSensitivity = settings.planeDistanceSensitivity;
    consts->gFramerateScale = frameRateScale_;
    consts->gMaxBlurRadius = max(maxBlurRadius, settings.minBlurRadius);
    consts->gMinBlurRadius = settings.minBlurRadius;
    consts->gDiffPrepassBlurRadius = diffusePrepassBlurRadius;
    consts->gSpecPrepassBlurRadius = specularPrepassBlurRadius;
    consts->gMaxAccumulatedFrameNum = isHistoryReset ? 0 : float(maxAccumulatedFrameNum);
    consts->gMaxFastAccumulatedFrameNum = isHistoryReset ? 0 : float(settings.maxFastAccumulatedFrameNum);
    consts->gAntiFirefly = settings.enableAntiFirefly ? 1.0f : 0.0f;
    consts->gLobeAngleFraction =
        settings.lobeAngleFraction * settings.lobeAngleFraction; // TODO: GetSpecularLobeTanHalfAngle has been fixed,
                                                                 // but we want to use existing settings
    consts->gRoughnessFraction = settings.roughnessFraction;
    consts->gHistoryFixFrameNum = (float)settings.historyFixFrameNum;
    consts->gHistoryFixBasePixelStride = (float)settings.historyFixBasePixelStride;
    consts->gHistoryFixAlternatePixelStride = (float)settings.historyFixAlternatePixelStride;
    consts->gHistoryFixAlternatePixelStrideMaterialID = commonSettings_.historyFixAlternatePixelStrideMaterialID;
    consts->gFastHistoryClampingSigmaScale =
        lerp(3.0f, settings.fastHistoryClampingSigmaScale, saturate(max(maxBlurRadius, settings.minBlurRadius) / 2.0f));
    consts->gMinRectDimMulUnproject = (float)min(rectW, rectH) * unproject;
    consts->gUsePrepassNotOnlyForSpecularMotionEstimation =
        settings.usePrepassOnlyForSpecularMotionEstimation ? 0.0f : 1.0f;
    consts->gSplitScreen = commonSettings_.splitScreen;
    consts->gSplitScreenPrev = splitScreenPrev_;
    consts->gCheckerboardResolveAccumSpeed = checkerboardResolveAccumSpeed_;
    consts->gViewZScale = commonSettings_.viewZScale;
    consts->gFireflySuppressorMinRelativeScale = settings.fireflySuppressorMinRelativeScale;
    consts->gMinHitDistanceWeight = settings.minHitDistanceWeight;
    consts->gDiffMinMaterial = settings.minMaterialForDiffuse;
    consts->gSpecMinMaterial = settings.minMaterialForSpecular;
    consts->gResponsiveAccumulationInvRoughnessThreshold =
        1.0f / max(settings.responsiveAccumulationSettings.roughnessThreshold, 1e-3f);
    consts->gResponsiveAccumulationMinAccumulatedFrameNum =
        settings.responsiveAccumulationSettings.minAccumulatedFrameNum;
    consts->gHasHistoryConfidence = commonSettings_.isHistoryConfidenceAvailable;
    consts->gHasDisocclusionThresholdMix = commonSettings_.isDisocclusionThresholdMixAvailable;
    consts->gDiffCheckerboard = diffCheckerboard;
    consts->gSpecCheckerboard = specCheckerboard;
    consts->gFrameIndex = commonSettings_.frameIndex;
    consts->gIsRectChanged = isRectChanged ? 1 : 0;
    consts->gResetHistory = isHistoryReset ? 1 : 0;
    consts->gReturnHistoryLengthInsteadOfOcclusion = settings.returnHistoryLengthInsteadOfOcclusion ? 1 : 0;
}

/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME REBLUR_DiffuseSpecular
#define DIFF_TEMP1 AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST)
#define DIFF_TEMP2 AsUint(Transient::DIFF_TMP2)
#define SPEC_TEMP1 AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST)
#define SPEC_TEMP2 AsUint(Transient::SPEC_TMP2)

void metallic::render::denoising::NrdPlan::addReblur(DenoiserData& denoiserData)
{
    denoiserData.settings.reblur = ReblurSettings();

    enum class Permanent {
        PREV_VIEWZ = PERMANENT_POOL_START,
        PREV_NORMAL_ROUGHNESS,
        PREV_INTERNAL_DATA,
        DIFF_HISTORY,
        DIFF_FAST_HISTORY,
        DIFF_HISTORY_STABILIZED_PING,
        DIFF_HISTORY_STABILIZED_PONG,
        SPEC_HISTORY,
        SPEC_FAST_HISTORY,
        SPEC_HISTORY_STABILIZED_PING,
        SPEC_HISTORY_STABILIZED_PONG,
        SPEC_HITDIST_FOR_TRACKING_PING,
        SPEC_HITDIST_FOR_TRACKING_PONG,
    };

    addPermanent({REBLUR_FORMAT_PREV_VIEWZ, 1});
    addPermanent({REBLUR_FORMAT_PREV_NORMAL_ROUGHNESS, 1});
    addPermanent({REBLUR_FORMAT_PREV_INTERNAL_DATA, 1});
    addPermanent({REBLUR_FORMAT, 1});
    addPermanent({REBLUR_FORMAT_FAST_HISTORY, 1});
    addPermanent({Format::R16_SFLOAT, 1});
    addPermanent({Format::R16_SFLOAT, 1});
    addPermanent({REBLUR_FORMAT, 1});
    addPermanent({REBLUR_FORMAT_FAST_HISTORY, 1});
    addPermanent({Format::R16_SFLOAT, 1});
    addPermanent({Format::R16_SFLOAT, 1});
    addPermanent({REBLUR_FORMAT_HITDIST_FOR_TRACKING, 1});
    addPermanent({REBLUR_FORMAT_HITDIST_FOR_TRACKING, 1});

    enum class Transient {
        DATA1 = TRANSIENT_POOL_START,
        DATA2,
        SPEC_HITDIST_FOR_TRACKING,
        DIFF_TMP2,
        DIFF_FAST_HISTORY,
        SPEC_TMP2,
        SPEC_FAST_HISTORY,
        TILES,
    };

    addTransient({Format::RG8_UNORM, 1});
    addTransient({Format::R32_UINT, 1});
    addTransient({REBLUR_FORMAT_HITDIST_FOR_TRACKING, 1});
    addTransient({REBLUR_FORMAT, 1});
    addTransient({REBLUR_FORMAT_FAST_HISTORY, 1});
    addTransient({REBLUR_FORMAT, 1});
    addTransient({REBLUR_FORMAT_FAST_HISTORY, 1});
    addTransient({REBLUR_FORMAT_TILES, 16});

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
        NRD_STAGE(REBLUR_ClassifyTiles, defines);
    }

    for (int i = 0; i < REBLUR_HITDIST_RECONSTRUCTION_PERMUTATION_NUM; i++) {
        bool is5x5 = (((i >> 1) & 0x1) != 0);
        bool isPrepassEnabled = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Hit distance reconstruction");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(ResourceType::IN_VIEWZ));
            readTexture(AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            readTexture(AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));

            // Outputs
            writeTexture(isPrepassEnabled ? DIFF_TEMP2 : DIFF_TEMP1);
            writeTexture(isPrepassEnabled ? SPEC_TEMP2 : SPEC_TEMP1);

            // Shaders
            std::array<ShaderDefine, 3> defines = {{
                commonDefines[0],
                commonDefines[1],
                {"MODE_5X5", is5x5 ? "1" : "0"},
            }};
            NRD_STAGE(REBLUR_HitDistReconstruction, defines);
        }
    }

    for (int i = 0; i < REBLUR_PREPASS_PERMUTATION_NUM; i++) {
        bool isAfterReconstruction = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Pre-pass");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(ResourceType::IN_VIEWZ));
            readTexture(isAfterReconstruction ? DIFF_TEMP2 : AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            readTexture(isAfterReconstruction ? SPEC_TEMP2 : AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));

            // Outputs
            writeTexture(DIFF_TEMP1);
            writeTexture(SPEC_TEMP1);
            writeTexture(AsUint(Transient::SPEC_HITDIST_FOR_TRACKING));

            // Shaders
            NRD_STAGE(REBLUR_PrePass, commonDefines);
        }
    }

    for (int i = 0; i < REBLUR_TEMPORAL_ACCUMULATION_PERMUTATION_NUM; i++) {
        bool hasDisocclusionThresholdMix = (((i >> 2) & 0x1) != 0);
        bool hasConfidenceInputs = (((i >> 1) & 0x1) != 0);
        bool isAfterPrepass = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Temporal accumulation");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(ResourceType::IN_VIEWZ));
            readTexture(AsUint(ResourceType::IN_MV));
            readTexture(AsUint(Permanent::PREV_VIEWZ));
            readTexture(AsUint(Permanent::PREV_NORMAL_ROUGHNESS));
            readTexture(AsUint(Permanent::PREV_INTERNAL_DATA));
            readTexture(hasDisocclusionThresholdMix ? AsUint(ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX)
                                                    : REBLUR_DUMMY);
            readTexture(hasConfidenceInputs ? AsUint(ResourceType::IN_DIFF_CONFIDENCE) : REBLUR_DUMMY);
            readTexture(hasConfidenceInputs ? AsUint(ResourceType::IN_SPEC_CONFIDENCE) : REBLUR_DUMMY);
            readTexture(isAfterPrepass ? DIFF_TEMP1 : AsUint(ResourceType::IN_DIFF_RADIANCE_HITDIST));
            readTexture(isAfterPrepass ? SPEC_TEMP1 : AsUint(ResourceType::IN_SPEC_RADIANCE_HITDIST));
            readTexture(AsUint(Permanent::DIFF_HISTORY));
            readTexture(AsUint(Permanent::SPEC_HISTORY));
            readTexture(AsUint(Permanent::DIFF_FAST_HISTORY));
            readTexture(AsUint(Permanent::SPEC_FAST_HISTORY));
            readTexture(AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PING),
                        AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PONG));
            readTexture(AsUint(Transient::SPEC_HITDIST_FOR_TRACKING));

            // Outputs
            writeTexture(AsUint(Transient::DATA1));
            writeTexture(DIFF_TEMP2);
            writeTexture(SPEC_TEMP2);
            writeTexture(AsUint(Transient::DIFF_FAST_HISTORY));
            writeTexture(AsUint(Transient::SPEC_FAST_HISTORY));
            writeTexture(AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PONG),
                         AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PING));
            writeTexture(AsUint(Transient::DATA2));

            // Shaders
            NRD_STAGE(REBLUR_TemporalAccumulation, commonDefines);
        }
    }

    NRD_PASS("History fix");
    {
        // Inputs
        readTexture(AsUint(Transient::TILES));
        readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        readTexture(AsUint(Transient::DATA1));
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(DIFF_TEMP2);
        readTexture(SPEC_TEMP2);
        readTexture(AsUint(Transient::DIFF_FAST_HISTORY));
        readTexture(AsUint(Transient::SPEC_FAST_HISTORY));
        readTexture(AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PONG),
                    AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PING));

        // Outputs
        writeTexture(DIFF_TEMP1);
        writeTexture(SPEC_TEMP1);
        writeTexture(AsUint(Permanent::DIFF_FAST_HISTORY));
        writeTexture(AsUint(Permanent::SPEC_FAST_HISTORY));

        // Shaders
        NRD_STAGE(REBLUR_HistoryFix, commonDefines);
    }

    NRD_PASS("Blur");
    {
        // Inputs
        readTexture(AsUint(Transient::TILES));
        readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(Transient::DATA1));
        readTexture(DIFF_TEMP1);
        readTexture(SPEC_TEMP1);

        // Outputs
        writeTexture(AsUint(Permanent::PREV_VIEWZ));
        writeTexture(DIFF_TEMP2);
        writeTexture(SPEC_TEMP2);

        // Shaders
        NRD_STAGE(REBLUR_Blur, commonDefines);
    }

    for (int i = 0; i < REBLUR_POST_BLUR_PERMUTATION_NUM; i++) {
        bool isTemporalStabilization = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Post-blur");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(Transient::DATA1));
            readTexture(AsUint(Permanent::PREV_VIEWZ));
            readTexture(DIFF_TEMP2);
            readTexture(SPEC_TEMP2);

            // Outputs
            writeTexture(AsUint(Permanent::PREV_NORMAL_ROUGHNESS));
            writeTexture(AsUint(Permanent::DIFF_HISTORY));
            writeTexture(AsUint(Permanent::SPEC_HISTORY));

            if (!isTemporalStabilization) {
                writeTexture(AsUint(Permanent::PREV_INTERNAL_DATA));
                writeTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
                writeTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            }

            // Shaders
            std::array<ShaderDefine, 3> defines = {{
                commonDefines[0],
                commonDefines[1],
                {"TEMPORAL_STABILIZATION", isTemporalStabilization ? "1" : "0"},
            }};
            NRD_STAGE(REBLUR_PostBlur, defines);
        }
    }

    for (int i = 0; i < REBLUR_TEMPORAL_STABILIZATION_PERMUTATION_NUM; i++) {
        bool hasRf0AndMetalness = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Temporal stabilization");
        {
            // Inputs
            readTexture(AsUint(Transient::TILES));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(Permanent::PREV_VIEWZ));
            readTexture(AsUint(Transient::DATA1));
            readTexture(AsUint(Transient::DATA2));
            readTexture(hasRf0AndMetalness ? AsUint(ResourceType::IN_BASECOLOR_METALNESS) : REBLUR_DUMMY);
            readTexture(AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PONG),
                        AsUint(Permanent::SPEC_HITDIST_FOR_TRACKING_PING));
            readTexture(AsUint(Permanent::DIFF_HISTORY));
            readTexture(AsUint(Permanent::SPEC_HISTORY));
            readTexture(AsUint(Permanent::DIFF_HISTORY_STABILIZED_PING),
                        AsUint(Permanent::DIFF_HISTORY_STABILIZED_PONG));
            readTexture(AsUint(Permanent::SPEC_HISTORY_STABILIZED_PING),
                        AsUint(Permanent::SPEC_HISTORY_STABILIZED_PONG));

            // Outputs
            writeTexture(AsUint(ResourceType::IN_MV));
            writeTexture(AsUint(Permanent::PREV_INTERNAL_DATA));
            writeTexture(AsUint(ResourceType::OUT_DIFF_RADIANCE_HITDIST));
            writeTexture(AsUint(ResourceType::OUT_SPEC_RADIANCE_HITDIST));
            writeTexture(AsUint(Permanent::DIFF_HISTORY_STABILIZED_PONG),
                         AsUint(Permanent::DIFF_HISTORY_STABILIZED_PING));
            writeTexture(AsUint(Permanent::SPEC_HISTORY_STABILIZED_PONG),
                         AsUint(Permanent::SPEC_HISTORY_STABILIZED_PING));

            // Shaders
            NRD_STAGE(REBLUR_TemporalStabilization, commonDefines);
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
        NRD_STAGE(REBLUR_SplitScreen, commonDefines);
    }

    REBLUR_ADD_VALIDATION_DISPATCH(Transient::DATA2, ResourceType::IN_DIFF_RADIANCE_HITDIST,
                                   ResourceType::IN_SPEC_RADIANCE_HITDIST);
}

#undef DENOISER_NAME
#undef DIFF_TEMP1
#undef SPEC_TEMP1
#undef DIFF_TEMP2
#undef SPEC_TEMP2
