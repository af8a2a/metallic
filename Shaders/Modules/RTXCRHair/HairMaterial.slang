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



public typealias HairLobeType = uint;
public static const uint kHairLobeTypeR = (0);
public static const uint kHairLobeTypeTT = (1);
public static const uint kHairLobeTypeTRT = (2);
public static const uint kHairMaxScatteringEvents = (3);

public typealias HairAbsorptionModel = uint;
public static const uint kHairAbsorptionModelColor = (0);
public static const uint kHairAbsorptionModelPhysics = (1);
public static const uint kHairAbsorptionModelNormalized = (2);

/************************************************
    Hair Surface
************************************************/

public struct HairInteractionSurface
{
    public float3 incidentRayDirection;
    public float3 shadingNormal;
    public float3 tangent;
};

public HairInteractionSurface createHairInteractionSurface(
    const float3 incidentRayDirection,
    const float3 normalWorld,
    const float3 tangentWorld)
{
    const float3 biTangentWorld = cross(normalWorld, tangentWorld);
    const float3x3 hairTangentBasis = float3x3(tangentWorld, biTangentWorld, normalWorld); // TBN

    const float3 incidentRayDirectionTangentSpace = mul(hairTangentBasis, incidentRayDirection);
    HairInteractionSurface hairInteractionSurface;
    hairInteractionSurface.incidentRayDirection = incidentRayDirectionTangentSpace;
    hairInteractionSurface.shadingNormal = float3(0.0f, 0.0f, 1.0f);
    hairInteractionSurface.tangent = float3(0.0f, 1.0f, 0.0f);
    return hairInteractionSurface;
}

/************************************************
    Hair Material
************************************************/

public struct HairMaterialData
{
    public float3 baseColor;
    public float  longitudinalRoughness; // beta_m

    public float  azimuthalRoughness;    // beta_n
    public float  ior;
    public float  eta;
    public uint   fresnelApproximation;

    public uint   absorptionModel;
    public float  melanin;
    public float  melaninRedness;
    public float  cuticleAngleInDegrees; // alpha
};

/************************************************
    Hair Interaction - Chiang BSDF
************************************************/

public struct HairMaterialInteraction
{
    public float  h;
    public float  gammaI;
    public float3 absorptionCoefficient;

    public float  ior;
    public float  eta;
    public uint   fresnelApproximation;

    public float  logisticDistributionScalar; // s

    public float v[kHairMaxScatteringEvents + 1];

    public float sin2kAlpha[kHairMaxScatteringEvents];
    public float cos2kAlpha[kHairMaxScatteringEvents];
};

// Compute Longitudinal Roughness Variance
public void computeRoughnessVariance(const float betaM, inout HairMaterialInteraction hairMaterialInteraction)
{
    float tmp = 0.726f * betaM + 0.812f * betaM * betaM + 3.7f * pow(betaM, 20.f);
    hairMaterialInteraction.v[0] = max(tmp * tmp, 1e-7f);
    hairMaterialInteraction.v[1] = 0.25f * hairMaterialInteraction.v[0];
    hairMaterialInteraction.v[2] = 4 * hairMaterialInteraction.v[0];
    [unroll]
    for (uint p = 3; p <= kHairMaxScatteringEvents; ++p)
    {
        hairMaterialInteraction.v[p] = hairMaterialInteraction.v[2];
    }
}

// Compute azimuthally offset h
public float calculateAzimuthallyDistance(const HairInteractionSurface hairInteractionSurface)
{
    // Project wi to the (B, N) plane
    float3 wiProj = normalize(hairInteractionSurface.incidentRayDirection -
        dot(hairInteractionSurface.incidentRayDirection, hairInteractionSurface.tangent) * hairInteractionSurface.tangent);
    // Calculate the vector that perpendicular with projected wi on (B, N) plane
    float3 wiProjPerpendicular = cross(wiProj, hairInteractionSurface.tangent);
    // h = sin(Gamma) = cos(pi/2 - Gamma) = dot(N, Wi_Proj_Prependicular)
    return dot(hairInteractionSurface.shadingNormal, wiProjPerpendicular);
}

// Mapping from color to absorption coefficient.
public float3 absorptionCoefficientFromColor(const float3 color, const float betaN)
{
    const float tmp = 5.969f - 0.215f * betaN + 2.532f * betaN * betaN - 10.73f * pow(betaN, 3.0f) + 5.574f * pow(betaN, 4.0f) + 0.245f * pow(betaN, 5.0f);
    const float3 sqrtAbsorptionCoefficient = log(max(color, 1e-4f)) / tmp;
    return sqrtAbsorptionCoefficient * sqrtAbsorptionCoefficient;
}

// Mapping from hair melanin to absorption coefficient
public float3 computeAbsorptionFromMelanin(float eumelanin, float pheomelanin)
{
    return max(eumelanin * float3(0.506f, 0.841f, 1.653f) + pheomelanin * float3(0.343f, 0.733f, 1.924f), float3(0.0f, 0.0f, 0.0f));
}

public float3 absorptionCoefficientFromMelanin(const float melanin_concentration, const float melanin_redness)
{
    float melanin_concentration_value = melanin_concentration;
    float melanin_gamma = 2.4f;
    float melanin = melanin_concentration_value * melanin_concentration_value * melanin_gamma;
    float eumelanin = melanin * (1.0f - melanin_redness);
    float pheomelanin = melanin * melanin_redness;
    return computeAbsorptionFromMelanin(eumelanin, pheomelanin);
}

public float3 absorptionCoefficientFromMelaninNormalized(const float melanin, const float melaninRedness)
{
    const float melaninQty = -log(max(1.0f - melanin, 0.0001f));
    const float eumelanin = melaninQty * (1.0f - melaninRedness);
    const float pheomelanin = melaninQty * melaninRedness;
    // Adjusted sigma coefficient for range [0, 1]
    const float3 eumelaninSigmaA = float3(0.506f, 0.841f, 1.653f);
    const float3 pheomelaninSigmaA = float3(0.343f, 0.733f, 1.924f);
    return eumelanin.rrr * eumelaninSigmaA + pheomelanin.rrr * pheomelaninSigmaA;
}

public float3 computeAbsorptionCoefficient(const HairMaterialData hairMaterialData)
{
    switch (hairMaterialData.absorptionModel)
    {
        case kHairAbsorptionModelColor:
            return absorptionCoefficientFromColor(hairMaterialData.baseColor, hairMaterialData.azimuthalRoughness);
        case kHairAbsorptionModelPhysics:
            return absorptionCoefficientFromMelanin(hairMaterialData.melanin, hairMaterialData.melaninRedness);
        case kHairAbsorptionModelNormalized:
            return absorptionCoefficientFromMelaninNormalized(hairMaterialData.melanin, hairMaterialData.melaninRedness);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

// Compute azimuthal logistic scale factor
public float computeLogisticDistributionScalar(const float betaN)
{
    return max(kPiOverEight * (0.265f * betaN + 1.194f * betaN * betaN + 5.372f * pow(betaN, 22.0f)), 1e-7f);
}

// Compute the scales that caused by the angle between hair cuticle and hair surface
//    /    /    /  <-- Hair Cuticles
//   /    /    /
//  /____/____/____   <-- Hair Surface
//
public void computeHairCuticleScales(const float cuticleAngleInDegrees, inout HairMaterialInteraction hairMaterialInteraction)
{
    hairMaterialInteraction.sin2kAlpha[0] = sin(cuticleAngleInDegrees / 180.0f * kPi);
    hairMaterialInteraction.cos2kAlpha[0] = sqrt(saturate(1.f - hairMaterialInteraction.sin2kAlpha[0] * hairMaterialInteraction.sin2kAlpha[0]));
    [unroll]
    for (uint i = 1; i < 3; i++)
    {
        // sin(2*Theta) = 2 * sin(Theta) * cos(Theta)
        hairMaterialInteraction.sin2kAlpha[i] =
            2 * hairMaterialInteraction.cos2kAlpha[i - 1] * hairMaterialInteraction.sin2kAlpha[i - 1];
        // cos(2*Theta) = (cos(Theta))^2 - (sin(Theta))^2
        hairMaterialInteraction.cos2kAlpha[i] =
            hairMaterialInteraction.cos2kAlpha[i - 1] * hairMaterialInteraction.cos2kAlpha[i - 1] -
            hairMaterialInteraction.sin2kAlpha[i - 1] * hairMaterialInteraction.sin2kAlpha[i - 1];
    }
}

public HairMaterialInteraction createHairMaterialInteraction(
    const HairMaterialData hairMaterialData,
    const HairInteractionSurface hairInteractionSurface)
{
    HairMaterialInteraction hairMaterialInteraction;
    hairMaterialInteraction.h = calculateAzimuthallyDistance(hairInteractionSurface);
    hairMaterialInteraction.gammaI = asin(clamp(hairMaterialInteraction.h, -1.0f, 1.0f));
    hairMaterialInteraction.absorptionCoefficient = computeAbsorptionCoefficient(hairMaterialData);
    hairMaterialInteraction.fresnelApproximation = hairMaterialData.fresnelApproximation;
    hairMaterialInteraction.ior = hairMaterialData.ior;
    hairMaterialInteraction.eta = hairMaterialData.eta;
    hairMaterialInteraction.logisticDistributionScalar = computeLogisticDistributionScalar(hairMaterialData.azimuthalRoughness);
    // Compute hairMaterialInteraction.v
    computeRoughnessVariance(hairMaterialData.longitudinalRoughness, hairMaterialInteraction);
    // Compute Hair Scales
    computeHairCuticleScales(hairMaterialData.cuticleAngleInDegrees, hairMaterialInteraction);
    return hairMaterialInteraction;
}

/************************************************
    Hair Interaction - Separate Chiang BSDF
************************************************/

public struct HairMaterialSeparateChiangData
{
    public HairMaterialData base;

    public float longitudinalRoughnessTT;
    public float longitudinalRoughnessTRT;
    public float azimuthalRoughnessTT;
    public float azimuthalRoughnessTRT;
};

public struct HairMaterialSeparateChiangInteraction
{
    public float h;
    public float gammaI;
    public float3 absorptionCoefficient;

    public float ior;
    public float eta;
    public uint fresnelApproximation;

    public float logisticDistributionScalar[kHairMaxScatteringEvents + 1]; // s

    public float v[kHairMaxScatteringEvents + 1];

    public float sin2kAlpha[kHairMaxScatteringEvents];
    public float cos2kAlpha[kHairMaxScatteringEvents];
};

public float computeRoughnessVarianceSeparateChiang(const float betaM)
{
    const float tmp = 0.726f * betaM + 0.812f * betaM * betaM + 3.7f * pow(betaM, 20.f);
    return max(tmp * tmp, 1e-7f);
}

public void computeHairCuticleScalesSeparateChiang(const float cuticleAngleInDegrees, inout HairMaterialSeparateChiangInteraction hairMaterialSeparateChiangInteraction)
{
    hairMaterialSeparateChiangInteraction.sin2kAlpha[0] = sin(cuticleAngleInDegrees / 180.0f * kPi);
    hairMaterialSeparateChiangInteraction.cos2kAlpha[0] = sqrt(saturate(1.f - hairMaterialSeparateChiangInteraction.sin2kAlpha[0] * hairMaterialSeparateChiangInteraction.sin2kAlpha[0]));
    [unroll]
    for (uint i = 1; i < 3; i++)
    {
        // sin(2*Theta) = 2 * sin(Theta) * cos(Theta)
        hairMaterialSeparateChiangInteraction.sin2kAlpha[i] =
            2 * hairMaterialSeparateChiangInteraction.cos2kAlpha[i - 1] * hairMaterialSeparateChiangInteraction.sin2kAlpha[i - 1];
        // cos(2*Theta) = (cos(Theta))^2 - (sin(Theta))^2
        hairMaterialSeparateChiangInteraction.cos2kAlpha[i] =
            hairMaterialSeparateChiangInteraction.cos2kAlpha[i - 1] * hairMaterialSeparateChiangInteraction.cos2kAlpha[i - 1] -
            hairMaterialSeparateChiangInteraction.sin2kAlpha[i - 1] * hairMaterialSeparateChiangInteraction.sin2kAlpha[i - 1];
    }
}

public HairMaterialSeparateChiangInteraction createHairMaterialSeparateChiangInteraction(
    const HairMaterialSeparateChiangData hairMaterialSeparateChiangData,
    const HairInteractionSurface hairInteractionSurface)
{
    HairMaterialSeparateChiangInteraction hairMaterialSeparateChiangInteraction;
    hairMaterialSeparateChiangInteraction.h = calculateAzimuthallyDistance(hairInteractionSurface);
    hairMaterialSeparateChiangInteraction.gammaI = asin(clamp(hairMaterialSeparateChiangInteraction.h, -1.0f, 1.0f));
    hairMaterialSeparateChiangInteraction.absorptionCoefficient = computeAbsorptionCoefficient(hairMaterialSeparateChiangData.base);
    hairMaterialSeparateChiangInteraction.fresnelApproximation = hairMaterialSeparateChiangData.base.fresnelApproximation;
    hairMaterialSeparateChiangInteraction.ior = hairMaterialSeparateChiangData.base.ior;
    hairMaterialSeparateChiangInteraction.eta = hairMaterialSeparateChiangData.base.eta;
    hairMaterialSeparateChiangInteraction.logisticDistributionScalar[0] = computeLogisticDistributionScalar(hairMaterialSeparateChiangData.base.azimuthalRoughness);
    hairMaterialSeparateChiangInteraction.logisticDistributionScalar[1] = computeLogisticDistributionScalar(hairMaterialSeparateChiangData.azimuthalRoughnessTT);
    hairMaterialSeparateChiangInteraction.logisticDistributionScalar[2] = computeLogisticDistributionScalar(hairMaterialSeparateChiangData.azimuthalRoughnessTRT);
    hairMaterialSeparateChiangInteraction.logisticDistributionScalar[3] = hairMaterialSeparateChiangInteraction.logisticDistributionScalar[2];
    // Compute hairMaterialInteraction.v
    hairMaterialSeparateChiangInteraction.v[0] = computeRoughnessVarianceSeparateChiang(hairMaterialSeparateChiangData.base.longitudinalRoughness);
    hairMaterialSeparateChiangInteraction.v[1] = computeRoughnessVarianceSeparateChiang(hairMaterialSeparateChiangData.longitudinalRoughnessTT);
    hairMaterialSeparateChiangInteraction.v[2] = computeRoughnessVarianceSeparateChiang(hairMaterialSeparateChiangData.longitudinalRoughnessTRT);
    hairMaterialSeparateChiangInteraction.v[3] = hairMaterialSeparateChiangInteraction.v[2];
    // Compute Hair Scales
    computeHairCuticleScalesSeparateChiang(hairMaterialSeparateChiangData.base.cuticleAngleInDegrees, hairMaterialSeparateChiangInteraction);
    return hairMaterialSeparateChiangInteraction;
}

/************************************************
    Hair Interaction - Farfield BSDF
************************************************/

public struct HairMaterialInteractionBCSDF
{
    public float3 diffuseReflectionTint;
    public float diffuseReflectionWeight;

    public float roughness;
    public float3 absorptionCoefficient;

    public float ior;
    public float cuticleAngle;
};

public HairMaterialInteractionBCSDF createHairMaterialInteractionBCSDF(
    const HairMaterialData hairMaterialData,
    const float3                 diffuseRefelctionTint,
    const float                  diffuseReflectionWeight,
    const float                  roughness)
{
    HairMaterialInteractionBCSDF hairMaterialInteractionBcsdf;
    hairMaterialInteractionBcsdf.diffuseReflectionTint = diffuseRefelctionTint;
    hairMaterialInteractionBcsdf.diffuseReflectionWeight = diffuseReflectionWeight;
    hairMaterialInteractionBcsdf.roughness = roughness;
    hairMaterialInteractionBcsdf.absorptionCoefficient = computeAbsorptionCoefficient(hairMaterialData);
    hairMaterialInteractionBcsdf.ior = hairMaterialData.ior;
    hairMaterialInteractionBcsdf.cuticleAngle = radians(hairMaterialData.cuticleAngleInDegrees);
    return hairMaterialInteractionBcsdf;
}
} // namespace Metallic.RTXCR
