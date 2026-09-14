/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( REBLUR_TemporalAccumulationConstants )
    REBLUR_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, REBLUR_TILE_TYPE, gIn_Tiles, 0 )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, 1 )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, 2 )
    NRD_INPUT( Texture2D, float3, gIn_Mv, 3 )
    NRD_INPUT( Texture2D, float, gPrev_ViewZ, 4 )
    NRD_INPUT( Texture2D, float4, gPrev_Normal_Roughness, 5 )
    NRD_INPUT( Texture2D, uint, gPrev_InternalData, 6 )
    NRD_INPUT( Texture2D, float, gIn_DisocclusionThresholdMix, 7 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, 8 )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, 9 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, 10 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, 11 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Diff, 12 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Spec, 13 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_DiffFast, 14 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_SpecFast, 15 )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDistForTracking, 16 )
        #if( NRD_MODE != OCCLUSION )
            NRD_INPUT( Texture2D, float, gIn_SpecHitDistForTracking, 17 )
        #endif
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, 18 )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, 19 )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_DiffSh, 20 )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_SpecSh, 21 )
        #endif
    #elif( NRD_DIFF )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, 8 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, 9 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Diff, 10 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_DiffFast, 11 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, 12 )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_DiffSh, 13 )
        #endif
    #else
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, 8 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, 9 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gHistory_Spec, 10 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gHistory_SpecFast, 11 )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDistForTracking, 12 )
        #if( NRD_MODE != OCCLUSION )
            NRD_INPUT( Texture2D, float, gIn_SpecHitDistForTracking, 13 )
        #endif
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, 14 )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gHistory_SpecSh, 15 )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    NRD_OUTPUT( RWTexture2D, REBLUR_DATA1_TYPE, gOut_Data1, 0 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, 1 )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, 2 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_DiffFast, 3 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_SpecFast, 4 )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDistForTracking, 5 )
        #if( NRD_MODE != OCCLUSION )
            NRD_OUTPUT( RWTexture2D, uint, gOut_Data2, 6 )
        #endif
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, 7 )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, 8 )
        #endif
    #elif( NRD_DIFF )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, 1 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_DiffFast, 2 )
        #if( NRD_MODE != OCCLUSION )
            NRD_OUTPUT( RWTexture2D, uint, gOut_Data2, 3 )
        #endif
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, 4 )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, 1 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_SpecFast, 2 )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDistForTracking, 3 )
        #if( NRD_MODE != OCCLUSION )
            NRD_OUTPUT( RWTexture2D, uint, gOut_Data2, 4 )
        #endif
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, 5 )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define REBLUR_TemporalAccumulationGroupX 8
#define REBLUR_TemporalAccumulationGroupY 16

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X REBLUR_TemporalAccumulationGroupX
#define GROUP_Y REBLUR_TemporalAccumulationGroupY
