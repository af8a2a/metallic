/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( RELAX_HistoryFixConstants )
    RELAX_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, float, gIn_Tiles, 0 )
    NRD_INPUT( Texture2D, float,  gIn_HistoryLength, 1 )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, 2 )
    NRD_INPUT( Texture2D, float,  gIn_ViewZ, 3 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_INPUT( Texture2D, float4, gIn_Spec, 4 )
        NRD_INPUT( Texture2D, float4, gIn_Diff, 5 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 6 )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 7 )
        #endif
    #elif( NRD_DIFF )
        NRD_INPUT( Texture2D, float4, gIn_Diff, 4 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 5 )
        #endif
    #else
        NRD_INPUT( Texture2D, float4, gIn_Spec, 4 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 5 )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    #if( NRD_DIFF && NRD_SPEC )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, 0 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, 1 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 2 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffSh, 3 )
        #endif
    #elif( NRD_DIFF )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, 0 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffSh, 1 )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, 0 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 1 )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define RELAX_HistoryFixGroupX 8
#define RELAX_HistoryFixGroupY 8

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X RELAX_HistoryFixGroupX
#define GROUP_Y RELAX_HistoryFixGroupY
