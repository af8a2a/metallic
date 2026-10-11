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

implementing SHARC;

namespace Metallic.SHARC
{
public struct HashGridParameters
{
    public float3 cameraPosition;      // world-space camera position; distance to camera defines voxel level/size
    public float logarithmBase;        // controls how quickly grid levels change with distance (SHaRC typically uses 2.0f)
    public float sceneScale;           // world-space scale factor controlling voxel size
    public float levelBias;            // biases level selection: can add more near-camera levels or clamp minimum voxel size (start with 0)
};

public struct SHARCAccumulationData
{
    public uint4 data;
};

// radianceData packs float4(radiance.rgb, sampleNum) as four fp16 halves:
//   x = pack(radiance.r, radiance.g), y = pack(radiance.b, sampleNum)
public struct SHARCPackedData
{
    public uint2 radianceData;
    public uint sampleData;
    public uint sampleDataExt;
};

} // namespace Metallic.SHARC
