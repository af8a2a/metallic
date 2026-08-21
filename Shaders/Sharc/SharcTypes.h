/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * This software contains source code provided by NVIDIA Corporation.
 * Ported from RTXGI v2 (Libraries/Sharc/include/SharcTypes.h) for Metallic's
 * Slang/SPIR-V path tracer. The resolved-cache radiance is stored as four
 * packed fp16 halves inside a uint2 (via f32tof16/f16tof32) instead of native
 * float16_t4 storage, keeping the identical 16-byte entry layout without
 * requiring native 16-bit storage features.
 */

#ifndef SHARC_TYPES_H
#define SHARC_TYPES_H

struct SharcAccumulationData
{
    uint4 data;
};

// radianceData packs float4(radiance.rgb, sampleNum) as four fp16 halves:
//   x = pack(radiance.r, radiance.g), y = pack(radiance.b, sampleNum)
struct SharcPackedData
{
    uint2 radianceData;
    uint sampleData;
    uint sampleDataExt;
};

#endif // SHARC_TYPES_H
