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
public typealias HashGridKey = uint64_t;
public typealias HashGridIndex = uint;

// Fixed renderer configuration. Pass-local SHARC_UPDATE/SHARC_QUERY defines do
// not configure this module; runtime tuning is passed through SHARCParameters.
public static const uint kHashGridHashMapBucketSize = 16;
public static const uint64_t kHashGridInvalidHashKey = 0;
public static const uint kHashGridInvalidCacheIndex = 0xFFFFFFFFu;
public static const uint kHashGridKeyBitNum = 64;
public static const uint kHashGridPositionBitNum = 17;
public static const uint kHashGridLevelBitNum = 9;
public static const uint kHashGridNormalBitNum = 3;
public static const uint kHashGridPositionBitMask = (1u << kHashGridPositionBitNum) - 1;
public static const uint kHashGridLevelBitMask = (1u << kHashGridLevelBitNum) - 1;
public static const uint kHashGridNormalBitMask = (1u << kHashGridNormalBitNum) - 1;
public static const uint kHashGridLevelBitOffset = kHashGridPositionBitNum * 3;
public static const uint kHashGridNormalBitOffset = kHashGridLevelBitOffset + kHashGridLevelBitNum;
public static const float kHashGridPositionBias = 1e-4f;
public static const float kHashGridNormalBias = 1e-3f;
public static const uint kHashGridLimitEmptySlots = 2;
public static const uint kSHARCVersionMajor = 1;
public static const uint kSHARCVersionMinor = 8;
public static const uint kSHARCVersionBuild = 3;
public static const uint kSHARCVersionRevision = 0;
public static const uint kSHARCAccumulatedFrameNumBitOffset = 0;
public static const uint kSHARCAccumulatedFrameNumBitNum = 16;
public static const uint kSHARCAccumulatedFrameNumBitMask = ((1 << kSHARCAccumulatedFrameNumBitNum) - 1);
public static const uint kSHARCStaleFrameNumBitOffset = 16;
public static const uint kSHARCStaleFrameNumBitNum = 16;
public static const uint kSHARCStaleFrameNumBitMask = ((1 << kSHARCStaleFrameNumBitNum) - 1);
public static const float kSHARCGridLogarithmBase = 2.0f;
public static const uint kSHARCAccumulatedFrameNumMin = 1;
public static const uint kSHARCAccumulatedFrameNumMax = 1024;
public static const uint kSHARCStaleFrameNumMax = 1024;
public static const uint kSHARCSampleNumThreshold = 0;
public static const uint kSHARCLinearProbeWindowSize = 8;
public static const uint kSHARCResamplingDepthMin = 1;
public static const uint kSHARCStaleFrameNumMin = 8;
public static const float kSHARCGridLevelBias = 0;
// Update resampling retains two vertices for radiance back-propagation.
public static const uint kSHARCPropagationDepth = 2;

} // namespace Metallic.SHARC
