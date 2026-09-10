/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

NRD_CONSTANTS_START( REBLUR_ValidationConstants )
    REBLUR_SHARED_CONSTANTS
    NRD_CONSTANT( uint, gHasDiffuse )
    NRD_CONSTANT( uint, gHasSpecular )
NRD_CONSTANTS_END

NRD_SAMPLERS_START
    NRD_SAMPLER( SamplerState, gNearestClamp, 0 )
    NRD_SAMPLER( SamplerState, gLinearClamp, 1 )
NRD_SAMPLERS_END

NRD_INPUTS_START
    NRD_INPUT( Texture2D, float4, gIn_Normal_Roughness, 0 )
    NRD_INPUT( Texture2D, float, gIn_ViewZ, 1 )
    NRD_INPUT( Texture2D, float3, gIn_Mv, 2 )
    NRD_INPUT( Texture2D, REBLUR_DATA1_TYPE, gIn_Data1, 3 )
    NRD_INPUT( Texture2D, uint, gIn_Data2, 4 )
    NRD_INPUT( Texture2D, float4, gIn_Diff, 5 )
    NRD_INPUT( Texture2D, float4, gIn_Spec, 6 )
NRD_INPUTS_END

NRD_OUTPUTS_START
    NRD_OUTPUT( RWTexture2D, float4, gOut_Validation, 0 )
NRD_OUTPUTS_END

// Macro magic
#define REBLUR_ValidationGroupX 8
#define REBLUR_ValidationGroupY 16

// Redirection
#undef GROUP_X
#undef GROUP_Y
#define GROUP_X REBLUR_ValidationGroupX
#define GROUP_Y REBLUR_ValidationGroupY
