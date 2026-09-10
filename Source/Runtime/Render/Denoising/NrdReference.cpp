/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NrdPlan.h"
/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shaders/Libraries/Denoising/NRD/REFERENCE_Copy.resources.hlsli"
#include "Shaders/Libraries/Denoising/NRD/REFERENCE_TemporalAccumulation.resources.hlsli"

#define DENOISER_NAME Reference

void metallic::render::denoising::NrdPlan::addReference(DenoiserData& denoiserData)
{
    denoiserData.settings.reference = ReferenceSettings();

    enum class Permanent {
        HISTORY = PERMANENT_POOL_START,
    };

    addPermanent({Format::RGBA32_SFLOAT, 1});

    std::array<ShaderDefine, 0> commonDefines = {};

    NRD_PASS("Temporal accumulation");
    {
        // Inputs
        readTexture(AsUint(ResourceType::IN_SIGNAL));

        // Outputs
        writeTexture(AsUint(Permanent::HISTORY));

        // Shaders
        NRD_STAGE(REFERENCE_TemporalAccumulation, commonDefines);
    }

    NRD_PASS("Copy");
    {
        // Inputs
        readTexture(AsUint(Permanent::HISTORY));

        // Outputs
        writeTexture(AsUint(ResourceType::OUT_SIGNAL));

        // Shaders
        NRD_STAGE(REFERENCE_Copy, commonDefines);
    }
}

#undef DENOISER_NAME

void metallic::render::denoising::NrdPlan::updateReference(const DenoiserData& denoiserData)
{
    enum class Dispatch {
        ACCUMULATE,
        COPY,
    };

    const ReferenceSettings& settings = denoiserData.settings.reference;

    if (worldToClip_ != worldToClipPrev_ || commonSettings_.accumulationMode != AccumulationMode::CONTINUE ||
        commonSettings_.rectSize[0] != commonSettings_.rectSizePrev[0] ||
        commonSettings_.rectSize[1] != commonSettings_.rectSizePrev[1])
        accumulatedFrames_[denoiserData.index] = 0;
    else {
        uint32_t maxAccumulatedFRameNum = min(settings.maxAccumulatedFrameNum, REFERENCE_MAX_HISTORY_FRAME_NUM);
        accumulatedFrames_[denoiserData.index] =
            min(accumulatedFrames_[denoiserData.index] + 1, maxAccumulatedFRameNum);
    }

    NRD_DECLARE_DIMS;

    { // ACCUMULATE
        REFERENCE_TemporalAccumulationConstants* consts =
            (REFERENCE_TemporalAccumulationConstants*)emitDispatch(denoiserData, AsUint(Dispatch::ACCUMULATE));
        consts->gAccumSpeed = 1.0f / (1.0f + accumulatedFrames_[denoiserData.index]);
        consts->gDebug = commonSettings_.debug;
    }

    { // COPY
        REFERENCE_CopyConstants* consts = (REFERENCE_CopyConstants*)emitDispatch(denoiserData, AsUint(Dispatch::COPY));
        consts->gRectSizeInv = float2(1.0f / float(rectW), 1.0f / float(rectH));
        consts->gSplitScreen = commonSettings_.splitScreen;
    }
}
