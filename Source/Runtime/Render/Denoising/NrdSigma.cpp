/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NrdPlan.h"

#include "Shaders/Libraries/Denoising/NRD/SIGMA_Config.hlsli"
#include "Shaders/Libraries/Denoising/NRD/SIGMA_Blur.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/SIGMA_ClassifyTiles.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/SIGMA_Copy.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/SIGMA_SmoothTiles.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/SIGMA_SplitScreen.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/SIGMA_TemporalStabilization.resources.hlsli"

// Permutations
#define SIGMA_POST_BLUR_PERMUTATION_NUM 2
#define SIGMA_NO_PERMUTATIONS           1

void metallic::render::denoising::NrdPlan::updateSigma(const DenoiserData& denoiserData)
{
    enum class Dispatch {
        CLASSIFY_TILES,
        SMOOTH_TILES = CLASSIFY_TILES + SIGMA_NO_PERMUTATIONS,
        COPY = SMOOTH_TILES + SIGMA_NO_PERMUTATIONS,
        BLUR = COPY + SIGMA_NO_PERMUTATIONS,
        POST_BLUR = BLUR + SIGMA_NO_PERMUTATIONS,
        TEMPORAL_STABILIZATION = POST_BLUR + SIGMA_POST_BLUR_PERMUTATION_NUM,
        SPLIT_SCREEN = TEMPORAL_STABILIZATION + SIGMA_NO_PERMUTATIONS,
    };

    const SigmaSettings& settings = denoiserData.settings.sigma;

    // SPLIT_SCREEN (passthrough)
    if (commonSettings_.splitScreen >= 1.0f) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        writeSigmaConstants(settings, consts);

        return;
    }

    { // CLASSIFY_TILES
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::CLASSIFY_TILES));
        writeSigmaConstants(settings, consts);
    }

    { // SMOOTH_TILES
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::SMOOTH_TILES));
        writeSigmaConstants(settings, consts);
    }

    // COPY
    if (settings.maxStabilizedFrameNum) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::COPY));
        writeSigmaConstants(settings, consts);
    }

    { // BLUR
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::BLUR));
        writeSigmaConstants(settings, consts);
    }

    { // POST_BLUR
        uint32_t passIndex = AsUint(Dispatch::POST_BLUR) + (settings.maxStabilizedFrameNum ? 1 : 0);
        void* consts = emitDispatch(denoiserData, passIndex);
        writeSigmaConstants(settings, consts);
    }

    // TEMPORAL_STABILIZATION
    if (settings.maxStabilizedFrameNum) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::TEMPORAL_STABILIZATION));
        writeSigmaConstants(settings, consts);
    }

    // SPLIT_SCREEN
    if (commonSettings_.splitScreen > 0.0f) {
        void* consts = emitDispatch(denoiserData, AsUint(Dispatch::SPLIT_SCREEN));
        writeSigmaConstants(settings, consts);
    }
}

void metallic::render::denoising::NrdPlan::writeSigmaConstants(const SigmaSettings& settings, void* data)
{
    struct SharedConstants {
        SIGMA_SHARED_CONSTANTS
    };

    NRD_DECLARE_DIMS;

    float unproject = 1.0f / (0.5f * rectH * projectY_);
    uint16_t tilesW = DivideUp(rectW, 16);
    uint16_t tilesH = DivideUp(rectH, 16);

    bool isRectChanged = rectW != rectWprev || rectH != rectHprev;
    uint32_t frameNum = min(settings.maxStabilizedFrameNum, SIGMA_MAX_HISTORY_FRAME_NUM);
    float3 lightDirectionView = Rotate(worldToView_, float3(settings.lightDirection[0], settings.lightDirection[1], settings.lightDirection[2]));
    float stabilizationStrength = frameNum / (1.0f + frameNum);

    SharedConstants* consts = (SharedConstants*)data;
    consts->gWorldToView = worldToView_;
    consts->gViewToClip = viewToClip_;
    consts->gWorldToClipPrev = worldToClipPrev_;
    consts->gWorldToViewPrev = worldToViewPrev_;
    consts->gRotator = rotator_;
    consts->gRotatorPost = rotatorPost_;
    consts->gViewVectorWorld = viewDirection_.xmm;
    consts->gLightDirectionView = float4(lightDirectionView.x, lightDirectionView.y, lightDirectionView.z, 0.0f);
    consts->gFrustum = frustum_;
    consts->gFrustumPrev = frustumPrev_;
    consts->gCameraDelta = cameraDelta_.xmm;
    consts->gMvScale = float4(commonSettings_.motionVectorScale[0], commonSettings_.motionVectorScale[1], commonSettings_.motionVectorScale[2], commonSettings_.isMotionVectorInWorldSpace ? 1.0f : 0.0f);
    consts->gResourceSizeInv = float2(1.0f / float(resourceW), 1.0f / float(resourceH));
    consts->gResourceSizeInvPrev = float2(1.0f / float(resourceWprev), 1.0f / float(resourceHprev));
    consts->gRectSize = float2(float(rectW), float(rectH));
    consts->gRectSizeInv = float2(1.0f / float(rectW), 1.0f / float(rectH));
    consts->gRectSizePrev = float2(float(rectWprev), float(rectHprev));
    consts->gResolutionScale = float2(float(rectW) / float(resourceW), float(rectH) / float(resourceH));
    consts->gRectOffset = float2(float(commonSettings_.rectOrigin[0]) / float(resourceW), float(commonSettings_.rectOrigin[1]) / float(resourceH));
    consts->gPrintfAt = uint2(commonSettings_.printfAt[0], commonSettings_.printfAt[1]);
    consts->gRectOrigin = uint2(commonSettings_.rectOrigin[0], commonSettings_.rectOrigin[1]);
    consts->gRectSizeMinusOne = int2(rectW - 1, rectH - 1);
    consts->gTilesSizeMinusOne = int2(tilesW - 1, tilesH - 1);
    consts->gOrthoMode = orthoMode_;
    consts->gUnproject = unproject;
    consts->gDenoisingRange = commonSettings_.denoisingRange;
    consts->gPlaneDistSensitivity = settings.planeDistanceSensitivity;
    consts->gStabilizationStrength = commonSettings_.accumulationMode == AccumulationMode::CONTINUE ? stabilizationStrength : 0.0f;
    consts->gDebug = commonSettings_.debug;
    consts->gSplitScreen = commonSettings_.splitScreen;
    consts->gViewZScale = commonSettings_.viewZScale;
    consts->gMinRectDimMulUnproject = (float)min(rectW, rectH) * unproject;
    consts->gFrameIndex = commonSettings_.frameIndex;
    consts->gIsRectChanged = isRectChanged ? 1 : 0;
}


/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define DENOISER_NAME SIGMA_Shadow

void metallic::render::denoising::NrdPlan::addSigma(DenoiserData& denoiserData)
{
    denoiserData.settings.sigma = SigmaSettings();

    enum class Permanent {
        HISTORY_LENGTH = PERMANENT_POOL_START,
    };

    addPermanent({Format::R32_UINT, 1});

    enum class Transient {
        DATA_1 = TRANSIENT_POOL_START,
        DATA_2,
        TEMP_1,
        TEMP_2,
        HISTORY,
        HISTORY_LENGTH,
        TILES,
        SMOOTHED_TILES,
    };

    addTransient({Format::R16_SFLOAT, 1});
    addTransient({Format::R16_SFLOAT, 1});
    addTransient({Format::R8_UNORM, 1});
    addTransient({Format::R8_UNORM, 1});
    addTransient({Format::R8_UNORM, 1});
    addTransient({Format::R32_UINT, 1});
    addTransient({Format::RGBA8_UNORM, 16});
    addTransient({Format::RG8_UNORM, 16});

    std::array<ShaderDefine, 1> commonDefines = {{
        {"TRANSLUCENCY", "0"},
    }};

    NRD_PASS("Classify tiles");
    {
        // Inputs
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(ResourceType::IN_PENUMBRA));

        // Outputs
        writeTexture(AsUint(Transient::TILES));

        // Shaders
        NRD_STAGE(SIGMA_ClassifyTiles, commonDefines);
    }

    NRD_PASS("Smooth tiles");
    {
        // Inputs
        readTexture(AsUint(Transient::TILES));

        // Outputs
        writeTexture(AsUint(Transient::SMOOTHED_TILES));

        // Shaders
        std::array<ShaderDefine, 0> defines = {};
        NRD_STAGE_SIZED(SIGMA_SmoothTiles, defines, 16);
    }

    NRD_PASS("Copy");
    {
        // Inputs
        readTexture(AsUint(Transient::SMOOTHED_TILES));
        readTexture(AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));
        readTexture(AsUint(Permanent::HISTORY_LENGTH));

        // Outputs
        writeTexture(AsUint(Transient::HISTORY));
        writeTexture(AsUint(Transient::HISTORY_LENGTH));

        // Shaders
        std::array<ShaderDefine, 0> defines = {};
        NRD_STAGE_SIZED(SIGMA_Copy, defines, USE_PREV_DIMS);
    }

    NRD_PASS("Blur");
    {
        // Inputs
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
        readTexture(AsUint(ResourceType::IN_PENUMBRA));
        readTexture(AsUint(Transient::SMOOTHED_TILES));

        // Outputs
        writeTexture(AsUint(Transient::DATA_1));
        writeTexture(AsUint(Transient::TEMP_1));

        // Shaders
        std::array<ShaderDefine, 2> defines = {{
            commonDefines[0],
            {"FIRST_PASS", "1"},
        }};
        NRD_STAGE(SIGMA_Blur, defines);
    }

    for (int i = 0; i < SIGMA_POST_BLUR_PERMUTATION_NUM; i++) {
        bool isStabilizationEnabled = (((i >> 0) & 0x1) != 0);

        NRD_PASS("Post-blur");
        {
            // Inputs
            readTexture(AsUint(ResourceType::IN_VIEWZ));
            readTexture(AsUint(ResourceType::IN_NORMAL_ROUGHNESS));
            readTexture(AsUint(Transient::DATA_1));
            readTexture(AsUint(Transient::SMOOTHED_TILES));
            readTexture(AsUint(Transient::TEMP_1));

            // Outputs
            writeTexture(AsUint(Transient::DATA_2));
            writeTexture(isStabilizationEnabled ? AsUint(Transient::TEMP_2) : AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));

            // Shaders
            std::array<ShaderDefine, 2> defines = {{
                commonDefines[0],
                {"FIRST_PASS", "0"},
            }};
            NRD_STAGE(SIGMA_Blur, defines);
        }
    }

    NRD_PASS("Temporal stabilization");
    {
        // Inputs
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(ResourceType::IN_MV));
        readTexture(AsUint(Transient::DATA_2));
        readTexture(AsUint(Transient::TEMP_2));
        readTexture(AsUint(Transient::HISTORY));
        readTexture(AsUint(Transient::HISTORY_LENGTH));
        readTexture(AsUint(Transient::SMOOTHED_TILES));

        // Outputs
        writeTexture(AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));
        writeTexture(AsUint(Permanent::HISTORY_LENGTH));

        // Shaders
        NRD_STAGE(SIGMA_TemporalStabilization, commonDefines);
    }

    NRD_PASS("Split screen");
    {
        // Inputs
        readTexture(AsUint(ResourceType::IN_VIEWZ));
        readTexture(AsUint(ResourceType::IN_PENUMBRA));

        // Outputs
        writeTexture(AsUint(ResourceType::OUT_SHADOW_TRANSLUCENCY));

        // Shaders
        NRD_STAGE(SIGMA_SplitScreen, commonDefines);
    }
}

#undef DENOISER_NAME
