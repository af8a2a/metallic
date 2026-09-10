/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( REBLUR_BlurConstants )
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
    NRD_INPUT( Texture2D, REBLUR_DATA1_TYPE, gIn_Data1, 3 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, 4 )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, 5 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, 6 )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, 7 )
        #endif
    #elif( NRD_DIFF )
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Diff, 4 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_DiffSh, 5 )
        #endif
    #else
        NRD_INPUT( Texture2D, REBLUR_TYPE, gIn_Spec, 4 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, REBLUR_SH_TYPE, gIn_SpecSh, 5 )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    NRD_OUTPUT( RWTexture2D, float, gOut_ViewZ, 0 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, 1 )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, 2 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, 3 )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, 4 )
        #endif
    #elif( NRD_DIFF )
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Diff, 1 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_DiffSh, 2 )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, REBLUR_TYPE, gOut_Spec, 1 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, REBLUR_SH_TYPE, gOut_SpecSh, 2 )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define REBLUR_BlurGroupX 8
#define REBLUR_BlurGroupY 16

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X REBLUR_BlurGroupX
#define GROUP_Y REBLUR_BlurGroupY
