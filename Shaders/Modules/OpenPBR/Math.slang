#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_math.h. Modified by Metallic, 2026.
namespace Metallic.OpenPBR
{
/***************************************************************************
 * Copyright 2026 Adobe
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

// Preserve the Adobe default math operations. No fast-math approximation
// or renderer-global macro override is used by the native module.

// clang-format off

public float openPBRFastRcpSqrt(const float x) { return (1.0f / sqrt(x)); }

public float openPBRFastSqrt(const float x)     { return sqrt(x); }

public float3  openPBRFastSqrt(const float3 v)      { return float3(sqrt(v.x), sqrt(v.y), sqrt(v.z)); }

public float2  openPBRFastNormalize(const float2 v) { return normalize(v); }

public float3  openPBRFastNormalize(const float3 v) { return normalize(v); }
// clang-format on

public float openPBRSquare(const float x) { return x * x; }

public float3 openPBRSquare(const float3 v) { return float3(v.x * v.x, v.y * v.y, v.z * v.z); }

public float openPBRFourthPower(const float x)
{
    const float xx = x * x;
    return xx * xx;
}

public float openPBRFifthPower(const float x)
{
    const float xx = x * x;
    return xx * xx * x;
}

public float openPBRSixthPower(const float x)
{
    const float xx = x * x;
    return xx * xx * xx;
}

public float openPBRMin3(const float3 v) { return min(min(v.x, v.y), v.z); }

public float openPBRMax3(const float3 v) { return max(max(v.x, v.y), v.z); }

public float openPBRGetCosThetaLocal(const float3 direction_local) { return direction_local.z; }

public float openPBRSignNonzero(const float x) { return (x >= 0.0f) ? 1.0f : -1.0f; }

public float2 openPBRSignNonzero(const float2 v) { return float2(openPBRSignNonzero(v.x), openPBRSignNonzero(v.y)); }

public float openPBRSafeDivide(const float numerator, const float denominator, const float fallback)
{
    return (denominator != 0.0f) ? (numerator / denominator) : fallback;
}

public float3 openPBRSafeDivide(const float3 numerator, const float3 denominator, const float3 fallback)
{
    return float3(openPBRSafeDivide(numerator[0], denominator[0], fallback[0]),
                  openPBRSafeDivide(numerator[1], denominator[1], fallback[1]),
                  openPBRSafeDivide(numerator[2], denominator[2], fallback[2]));
}

public bool openPBRIsNormalized(const float3 normal, const float tolerance)
{
    // We use <= to ensure that the test could still be passed even if the tolerance were set to zero.
    return abs(length(normal) - 1.0f) <= tolerance;
}

public bool openPBRIsNormalized(const float3 normal) { return openPBRIsNormalized(normal, 1.0e-6f); }

public float openPBRSum(const float3 v) { return v.x + v.y + v.z; }

public float openPBRSafePow(const float a, const float b) { return a > 0 ? pow(a, b) : 0; }

public float3 openPBRSafePow(const float3 a, const float3 b)
{
    return float3(openPBRSafePow(a.x, b.x), openPBRSafePow(a.y, b.y), openPBRSafePow(a.z, b.z));
}

// Assuming a z-up local space, returns whether two directions are in the same hemisphere.

public bool openPBRAreInSameHemisphereLocal(const float3 direction_local, const float3 other_direction_local)
{
    return direction_local.z * other_direction_local.z > 0.0f;
}

public float openPBRFourthRoot(const float x) { return sqrt(sqrt(x)); }

public float openPBRAverage(const float3 v) { return openPBRSum(v) * (1.0f / 3); }

public float openPBRCube(const float x) { return x * x * x; }

public float openPBRThreeHalvesPower(const float x) { return sqrt(openPBRCube(x)); }
} // namespace Metallic.OpenPBR
