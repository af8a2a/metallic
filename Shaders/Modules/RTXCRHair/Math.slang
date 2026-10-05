#language slang 2026
implementing RTXCRHair;

// Native Slang port; modified by Metallic, 2026. Upstream provenance in Upstream.json.
namespace Metallic.RTXCR
{
/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/


// PIs
public static const float kPi = (3.141592653589f);

public static const float kTwoPi = (2.0f * kPi);

public static const float kFourPi = (4.0f * kPi);

public static const float kOneOverPi = (1.0f / kPi);

public static const float kOneOverTwoPi = (1.0f / kTwoPi);

public static const float kPiOverEight = (0.626657069f);

public static const float3 kWhiteColor = float3(1.0f, 1.0f, 1.0f);

// Safe sqrt for x between [0, 1]
public float sqrt01(float x)
{
    return max(sqrt(saturate(x)), 1e-7);
}

// Safe sqrt for x
public float sqrt0(float x)
{
    return sqrt(max(x, 1e-7));
}

public float3 sqrt0(float3 x)
{
    return sqrt(max(x, 1e-7));
}

public float atan2safe(float x, float y)
{
    return abs(x) + abs(y) < 1e-7 ? 0 : atan2(x, y);
}

public float i0(float x)
{
    float val = 0.f;
    float x2i = 1.f;
    float ifact = 1.f;
    uint i4 = 1;

    [unroll]
    for (uint i = 0; i < 10; i++)
    {
        if (i > 1)
            ifact *= i;
        val += x2i / (ifact * ifact * i4);
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}

public float logI0(float x)
{
    if (x > 12)
    {
        return x + 0.5f * (-log(kTwoPi) + log(1.f / x) + 0.125f / x);
    }
    else
    {
        return log(i0(x));
    }
}

public float phiFunction(int p, float gammaI, float gammaT)
{
    return 2.f * p * gammaT - 2.f * gammaI + p * kPi;
}

public float logistic(float x, float s)
{
    x = abs(x);
    float tmp = exp(-x / s);
    return tmp / (s * (1.f + tmp) * (1.f + tmp));
}

public float logisticCDF(float x, float s)
{
    return 1.f / (1.f + exp(-x / s));
}

public float trimmedLogistic(float x, float s, float a, float b)
{
    return logistic(x, s) / (logisticCDF(b, s) - logisticCDF(a, s));
}

public float sampleTrimmedLogistic(float u, float s, float a, float b)
{
    float k = logisticCDF(b, s) - logisticCDF(a, s);
    float x = -s * log(1.f / (u * k + logisticCDF(a, s)) - 1.f);
    return clamp(x, a, b);
}

// 1D Gaussian distribution normalized over [-inf,inf]
public float gaussian1D(const float x, const float stddev)
{
    return exp(-x * x / (2.0f * stddev * stddev)) / (stddev * sqrt(2.0f * kPi));
}

public float phiR(const float h)
{
    return -2.0 * asin(h);
}

public float phiTT(const float h, const float a) // a = 1.0 / eta_prime
{
    return kPi - 2.0 * asin(h) + 2.0 * asin(h * a);
}

public float phiTRT(const float h, const float a) // a = 1.0 / eta_prime
{
    return -2.0 * asin(h) + 4.0 * asin(h * a);
}

// sample from normal distribution (Box-Muller transform)
public float randomGaussian1D(const float xi1, const float xi2)
{
    return sqrt(2.0f) * cos(2.0f * kPi * xi1) * sqrt0(-log(1 - xi2));
}

public float2 polarToCartesian(float r, float theta)
{
    return r * float2(cos(theta), sin(theta));
}

public void createCoordinateSystemFromZ(bool rightHand, float3 zAxis, out float3 xAxis, out float3 yAxis)
{
    float yz = -zAxis.y * zAxis.z;
    yAxis = normalize(abs(zAxis.z) > 0.9999 ? float3(-zAxis.x * zAxis.y, 1.f - zAxis.y * zAxis.y, yz) :
                                              float3(-zAxis.x * zAxis.z, yz, 1.f - zAxis.z * zAxis.z));
    xAxis = rightHand ? cross(yAxis, zAxis) : cross(zAxis, yAxis);
}

// Spherical to Cartesian in the basis x, y, z
// z is up
public float3 sphericalDirection(float sinTheta, float cosTheta, float phi, float3 x, float3 y, float3 z)
{
    return sinTheta * cos(phi) * x + sinTheta * sin(phi) * y + cosTheta * z;
}

public float3 calculateDiskSamplePosition(
    in const float rand,
    in const float r,
    in float3 centerPos,
    in float3 tangent,
    in float3 biTangent)
{
    // Sample Disk
    const float theta = rand * kTwoPi;
    const float2 diskSample = polarToCartesian(r, theta);

    return centerPos + tangent * diskSample.xxx + biTangent * diskSample.yyy;
}

// Samples a direction within a hemisphere oriented along +Z axis with a cosine-weighted distribution
public float3 sampleHemisphere(float2 u, out float pdf)
{
    const float a = sqrt(u.x);
    const float b = kTwoPi * u.y;

    const float3 result = float3(
        a * cos(b),
        a * sin(b),
        sqrt(1.0f - u.x));

    pdf = result.z * kOneOverPi;

    return result;
}
} // namespace Metallic.RTXCR
