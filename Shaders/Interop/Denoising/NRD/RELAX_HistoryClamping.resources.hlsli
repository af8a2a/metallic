/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( RELAX_HistoryClampingConstants )
    RELAX_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, float, gIn_Tiles, 0 )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, 1 )
    NRD_INPUT( Texture2D, float, gIn_HistoryLength, 2 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_INPUT( Texture2D, float4, gIn_SpecNoisy, 3 )
        NRD_INPUT( Texture2D, float4, gIn_DiffNoisy, 4 )
        NRD_INPUT( Texture2D, float4, gIn_Spec, 5 )
        NRD_INPUT( Texture2D, float4, gIn_Diff, 6 )
        NRD_INPUT( Texture2D, float4, gIn_SpecFast, 7 )
        NRD_INPUT( Texture2D, float4, gIn_DiffFast, 8 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 9 )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 10 )
            NRD_INPUT( Texture2D, float4, gIn_SpecShFast, 11 )
            NRD_INPUT( Texture2D, float4, gIn_DiffShFast, 12 )
        #endif
    #elif( NRD_DIFF )
        NRD_INPUT( Texture2D, float4, gIn_DiffNoisy, 3 )
        NRD_INPUT( Texture2D, float4, gIn_Diff, 4 )
        NRD_INPUT( Texture2D, float4, gIn_DiffFast, 5 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 6 )
            NRD_INPUT( Texture2D, float4, gIn_DiffShFast, 7 )
        #endif
    #else
        NRD_INPUT( Texture2D, float4, gIn_SpecNoisy, 3 )
        NRD_INPUT( Texture2D, float4, gIn_Spec, 4 )
        NRD_INPUT( Texture2D, float4, gIn_SpecFast, 5 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 6 )
            NRD_INPUT( Texture2D, float4, gIn_SpecShFast, 7 )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    NRD_OUTPUT( RWTexture2D, float, gOut_HistoryLength, 0 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, 1 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, 2 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_SpecFast, 3 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_DiffFast, 4 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 5 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffSh, 6 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecShFast, 7 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffShFast, 8 )
        #endif
    #elif( NRD_DIFF )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff, 1 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_DiffFast, 2 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffSh, 3 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffShFast, 4 )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec, 1 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_SpecFast, 2 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 3 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecShFast, 4 )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define RELAX_HistoryClampingGroupX 8
#define RELAX_HistoryClampingGroupY 8

#define NRD_USE_BORDER_2

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X RELAX_HistoryClampingGroupX
#define GROUP_Y RELAX_HistoryClampingGroupY

