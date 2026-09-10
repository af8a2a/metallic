#include "Bindless.hlsli"
/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "NRD.hlsli"
#include "ml.hlsli"

#include "REFERENCE_TemporalAccumulation.bindings.hlsli"

#include "Common.hlsli"

[numthreads( GROUP_X, GROUP_Y, 1 )]
[shader("compute")] NRD_EXPORT void NRD_CS_MAIN( NRD_CS_MAIN_ARGS )
{
    NRD_CTA_ORDER_DEFAULT;
    uint width, height;
    gInOut_History.GetDimensions(width, height);
    if (any(pixelPos >= uint2(width, height))) return;

    float4 input = gIn_Input[ pixelPos ];
    float4 history = gInOut_History[ pixelPos ];
    float4 result = lerp( history, input, gAccumSpeed );

    gInOut_History[ pixelPos ] = result;
}
