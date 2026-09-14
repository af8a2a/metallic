/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( REBLUR_HistoryFixConstants )
    REBLUR_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, REBLUR_TILE_TYPE, gIn_Tiles, 0 )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, 1 )
    NRD_INPUT( Texture2D, REBLUR_DATA1_TYPE, gIn_Data1, 2 )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, 3 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, 4 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, 5 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gIn_DiffFast, 6 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gIn_SpecFast, 7 )
        #if( NRD_MODE != OCCLUSION )
            NRD_INPUT( Texture2D, float, gIn_SpecHitDistForTracking, 8 )
        #endif
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, 9 )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, 10 )
        #endif
    #elif( NRD_DIFF )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, 4 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gIn_DiffFast, 5 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, 6 )
        #endif
    #else
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, 4 )
        NRD_INPUT( Texture2D, REBLUR_FAST_TYPE, gIn_SpecFast, 5 )
        #if( NRD_MODE != OCCLUSION )
            NRD_INPUT( Texture2D, float, gIn_SpecHitDistForTracking, 6 )
        #endif
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, 7 )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    #if( NRD_DIFF && NRD_SPEC )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, 0 )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, 1 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_DiffFast, 2 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_SpecFast, 3 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, 4 )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, 5 )
        #endif
    #elif( NRD_DIFF )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, 0 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_DiffFast, 1 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, 2 )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, 0 )
        NRD_OUTPUT( RWTexture2D, REBLUR_FAST_TYPE, gOut_SpecFast, 1 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, 2 )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define REBLUR_HistoryFixGroupX 8
#define REBLUR_HistoryFixGroupY 16

#define NRD_USE_BORDER_2

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X REBLUR_HistoryFixGroupX
#define GROUP_Y REBLUR_HistoryFixGroupY
