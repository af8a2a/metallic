/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( RELAX_AtrousSmemConstants )
    RELAX_SHARED_CONSTANTS
    NRD_CONSTANT( uint, gStepSize )
    NRD_CONSTANT( uint, gIsLastPass )
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, float, gIn_Tiles, 0 )
    NRD_INPUT( Texture2D, float, gIn_HistoryLength, 1 )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, 2 )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, 3 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_INPUT( Texture2D, float4, gIn_Spec_Variance, 4 )
        NRD_INPUT( Texture2D, float4, gIn_Diff_Variance, 5 )
        NRD_INPUT( Texture2D, float, gIn_SpecReprojectionConfidence, 6 )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, 7 )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, 8 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 9 )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 10 )
        #endif
    #elif( NRD_DIFF )
        NRD_INPUT( Texture2D, float4, gIn_Diff_Variance, 4 )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, 5 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 6 )
        #endif
    #else
        NRD_INPUT( Texture2D, float4, gIn_Spec_Variance, 4 )
        NRD_INPUT( Texture2D, float, gIn_SpecReprojectionConfidence, 5 )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, 6 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 7 )
        #endif
    #endif
NRD_INPUTS_END

NRD_OUTPUTS_START
    #if( NRD_DIFF && NRD_SPEC )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec_Variance, 0 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff_Variance, 1 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_NormalRoughness, 2 )
        NRD_OUTPUT( RWTexture2D, float, gOut_MaterialID, 3 )
        NRD_OUTPUT( RWTexture2D, float, gOut_ViewZ, 4 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 5 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffSh, 6 )
        #endif
    #elif( NRD_DIFF )
        NRD_OUTPUT( RWTexture2D, float4, gOut_Diff_Variance, 0 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_NormalRoughness, 1 )
        NRD_OUTPUT( RWTexture2D, float, gOut_MaterialID, 2 )
        NRD_OUTPUT( RWTexture2D, float, gOut_ViewZ, 3 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffSh, 4 )
        #endif
    #else
        NRD_OUTPUT( RWTexture2D, float4, gOut_Spec_Variance, 0 )
        NRD_OUTPUT( RWTexture2D, float4, gOut_NormalRoughness, 1 )
        NRD_OUTPUT( RWTexture2D, float, gOut_MaterialID, 2 )
        NRD_OUTPUT( RWTexture2D, float, gOut_ViewZ, 3 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 4 )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define RELAX_AtrousSmemGroupX 8
#define RELAX_AtrousSmemGroupY 8

#define NRD_USE_BORDER_2

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X RELAX_AtrousSmemGroupX
#define GROUP_Y RELAX_AtrousSmemGroupY
