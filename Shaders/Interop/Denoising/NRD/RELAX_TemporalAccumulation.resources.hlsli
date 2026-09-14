/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( RELAX_TemporalAccumulationConstants )
    RELAX_SHARED_CONSTANTS
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, float, gIn_Tiles, 0 )
    NRD_INPUT( Texture2D, float3, gIn_Mv, 1 )
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, 2 )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, 3 )
    NRD_INPUT( Texture2D, float, gIn_DisocclusionThresholdMix, 4 )
    NRD_INPUT( Texture2D, float4, gPrev_Normal_Roughness, 5 )
    NRD_INPUT( Texture2D, float, gPrev_ViewZ, 6 )
    NRD_INPUT( Texture2D, float, gPrev_HistoryLength, 7 )
    NRD_INPUT( Texture2D, float, gPrev_MateriallID, 8 )
    #if( NRD_DIFF && NRD_SPEC )
        NRD_INPUT( Texture2D, float4, gIn_Spec, 9 )
        NRD_INPUT( Texture2D, float4, gIn_Diff, 10 )
        NRD_INPUT( Texture2D, float4, gHistory_SpecFast, 11 )
        NRD_INPUT( Texture2D, float4, gHistory_DiffFast, 12 )
        NRD_INPUT( Texture2D, float4, gHistory_Spec, 13 )
        NRD_INPUT( Texture2D, float4, gHistory_Diff, 14 )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDist, 15 )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, 16 )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, 17 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 18 )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 19 )
            NRD_INPUT( Texture2D, float4, gHistory_SpecShFast, 20 )
            NRD_INPUT( Texture2D, float4, gHistory_DiffShFast, 21 )
            NRD_INPUT( Texture2D, float4, gHistory_SpecSh, 22 )
            NRD_INPUT( Texture2D, float4, gHistory_DiffSh, 23 )
        #endif
    #elif( NRD_DIFF )
        NRD_INPUT( Texture2D, float4, gIn_Diff, 9 )
        NRD_INPUT( Texture2D, float4, gHistory_DiffFast, 10 )
        NRD_INPUT( Texture2D, float4, gHistory_Diff, 11 )
        NRD_INPUT( Texture2D, float, gIn_DiffConfidence, 12 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_DiffSh, 13 )
            NRD_INPUT( Texture2D, float4, gHistory_DiffShFast, 14 )
            NRD_INPUT( Texture2D, float4, gHistory_DiffSh, 15 )
        #endif
    #else
        NRD_INPUT( Texture2D, float4, gIn_Spec, 9 )
        NRD_INPUT( Texture2D, float4, gHistory_SpecFast, 10 )
        NRD_INPUT( Texture2D, float4, gHistory_Spec, 11 )
        NRD_INPUT( Texture2D, float, gPrev_SpecHitDist, 12 )
        NRD_INPUT( Texture2D, float, gIn_SpecConfidence, 13 )
        #if( NRD_MODE == SH )
            NRD_INPUT( Texture2D, float4, gIn_SpecSh, 14 )
            NRD_INPUT( Texture2D, float4, gHistory_SpecShFast, 15 )
            NRD_INPUT( Texture2D, float4, gHistory_SpecSh, 16 )
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
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDist, 5 )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecReprojectionConfidence, 6 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 7 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffSh, 8 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecShFast, 9 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_DiffShFast, 10 )
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
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecHitDist, 3 )
        NRD_OUTPUT( RWTexture2D, float, gOut_SpecReprojectionConfidence, 4 )
        #if( NRD_MODE == SH )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecSh, 5 )
            NRD_OUTPUT( RWTexture2D, float4, gOut_SpecShFast, 6 )
        #endif
    #endif
NRD_OUTPUTS_END

// Macro magic
#define RELAX_TemporalAccumulationGroupX 8
#define RELAX_TemporalAccumulationGroupY 16

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X RELAX_TemporalAccumulationGroupX
#define GROUP_Y RELAX_TemporalAccumulationGroupY
