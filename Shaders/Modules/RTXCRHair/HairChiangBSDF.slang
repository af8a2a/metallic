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

// Chiang16 Hair model
// Reference Paper: https://benedikt-bitterli.me/pchfm/
// Reference Article: https://www.pbrt.org/hair.pdf




/************************************************
    BSDF Evaluation
************************************************/

public float3 hairChiangBSDFEval(in const HairMaterialInteraction hairMaterialInteraction,
                                in const float3 wi,
                                in const float3 wo)
{
    const float sinThetaO = wo.x;
    const float cosThetaO = sqrt01(1.0f - sinThetaO * sinThetaO);
    const float phiO = atan2safe(wo.z, wo.y);

    const float sinThetaI = wi.x;
    const float cosThetaI = sqrt01(1.0f - sinThetaI * sinThetaI);
    const float phiI = atan2safe(wi.z, wi.y);

    // Compute refracted ray.
    const float sinThetaT = sinThetaO / hairMaterialInteraction.ior;
    const float cosThetaT = sqrt01(1.0f - sinThetaT * sinThetaT);

    const float etap = sqrt0(hairMaterialInteraction.ior * hairMaterialInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
    const float sinGammaT = hairMaterialInteraction.h / etap;
    const float cosGammaT = sqrt01(1.0f - sinGammaT * sinGammaT);
    const float gammaT = asin(clamp(sinGammaT, -1.0f, 1.0f));

    // Compute the transmittance T of a single path through the cylinder
    const float tmp = -2.0f * cosGammaT / cosThetaT;
    const float3 T = exp(hairMaterialInteraction.absorptionCoefficient * tmp);

    // Evaluate hair BCSDF for each lobe
    const float phi = phiI - phiO;
    float3 ap[kHairMaxScatteringEvents + 1];
    aP(hairMaterialInteraction, cosThetaO, T, ap);
    float3 result = 0.0f;

    [unroll]
    for (uint p = 0; p < kHairMaxScatteringEvents; ++p)
    {
        float sinThetaOp, cosThetaOp;
        if (p == 0)
        {
            sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialInteraction.sin2kAlpha[1];
            cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialInteraction.sin2kAlpha[1];
        }
        else if (p == 1)
        {
            sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialInteraction.sin2kAlpha[0];
            cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialInteraction.sin2kAlpha[0];
        }
        else if (p == 2)
        {
            sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialInteraction.sin2kAlpha[2];
            cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialInteraction.sin2kAlpha[2];
        }
        else
        {
            sinThetaOp = sinThetaO;
            cosThetaOp = cosThetaO;
        }

        cosThetaOp = abs(cosThetaOp);
        result += mP(cosThetaOp, cosThetaI, sinThetaOp, sinThetaI, hairMaterialInteraction.v[p]) *
                  ap[p] *
                  nP(phi, p, hairMaterialInteraction.logisticDistributionScalar, hairMaterialInteraction.gammaI, gammaT);
    }

    // Compute contribution of remaining terms after kHairMaxScatteringEvents
    result += mP(cosThetaO, cosThetaI, sinThetaO, sinThetaI, hairMaterialInteraction.v[kHairMaxScatteringEvents]) *
              ap[kHairMaxScatteringEvents] *
              kOneOverTwoPi;

    // We omit this computation in BSDF, because the cosThetaI_N will be cancelled out when evaluate scattered radiance anyway
    // const float cosThetaI_N = wi.z; // The angle between wi and normal, which is (0, 0, 1) on local space
    // result = abs(cosThetaI_N) > 0.0f ? result / abs(cosThetaI_N) : 0.0f;

    return max(result, 0.0f);
}

public float3 hairChiangBSDFEval(in const HairMaterialData hairMaterialData,
                                in const HairInteractionSurface hairInteractionSurface,
                                in const float3 wi)
{
    const float3 tangentWorld = hairInteractionSurface.tangent;
    const float3 biTangentWorld = cross(hairInteractionSurface.shadingNormal, tangentWorld);
    const float3x3 hairTangentBasis = float3x3(tangentWorld, biTangentWorld, hairInteractionSurface.shadingNormal); // TBN

    const float3 lightVectorLocal = mul(hairTangentBasis, wi);
    const float3 viewVectorLocal = mul(hairTangentBasis, hairInteractionSurface.incidentRayDirection);

    const HairMaterialInteraction hairMaterialInteraction = createHairMaterialInteraction(hairMaterialData, hairInteractionSurface);
    return hairChiangBSDFEval(hairMaterialInteraction, lightVectorLocal, viewVectorLocal);
}

/************************************************
    Importance Sampling
************************************************/

public bool sampleChiangBSDF(in const HairMaterialInteraction hairMaterialInteraction,
                            in const float3 wo,
                            in float2 u[2],
                            out float3 wi,
                            out float pdf,
                            out float3 weight,
                            out HairLobeType lobeType)
{
    const float sinThetaO = wo.x;
    const float cosThetaO = sqrt01(1.0f - sinThetaO * sinThetaO);
    const float phiO = atan2safe(wo.z, wo.y);
    
    // Determine which term p to sample for hair scattering.
    float apPdf[kHairMaxScatteringEvents + 1];
    computeApPdf(hairMaterialInteraction, cosThetaO, apPdf);

    uint p = 0;
    float vp = hairMaterialInteraction.v[0];
    {
        [unroll]
        for (uint i = 0; i < kHairMaxScatteringEvents; ++i)
        {
            if (u[0].x >= apPdf[i])
            {
                u[0].x -= apPdf[i];
                p = i + 1;
                vp = hairMaterialInteraction.v[i + 1];
            }
            else
            {
                break;
            }
        }
    }

    float sinThetaOp = sinThetaO;
    float cosThetaOp = cosThetaO;
    if (p == 0)
    {
        sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialInteraction.sin2kAlpha[1];
        cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialInteraction.sin2kAlpha[1];

        lobeType = kHairLobeTypeR;
    }
    else if (p == 1)
    {
        sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialInteraction.sin2kAlpha[0];
        cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialInteraction.sin2kAlpha[0];

        lobeType = kHairLobeTypeTT;
    }
    else if (p == 2)
    {
        sinThetaOp = sinThetaO * hairMaterialInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialInteraction.sin2kAlpha[2];
        cosThetaOp = cosThetaO * hairMaterialInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialInteraction.sin2kAlpha[2];

        lobeType = kHairLobeTypeTRT;
    }
    else
    {
        lobeType = kHairLobeTypeTRT;
    }

    // Sample Mp to compute thetaO
    u[1].x = max(u[1].x, 1e-5f);
    const float cosTheta = 1.0f + vp * log(u[1].x + (1.0f - u[1].x) * exp(-2.0f / vp));
    const float sinTheta = sqrt01(1.0f - cosTheta * cosTheta);
    const float cosPhi = cos(u[1].y * kTwoPi);
    const float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
    const float cosThetaI = sqrt01(1.0f - sinThetaI * sinThetaI);

    // Sample Np to compute dphi
    const float etap = sqrt0(hairMaterialInteraction.ior * hairMaterialInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
    const float sinGammaT = hairMaterialInteraction.h / etap;
    const float gammaT = asin(clamp(sinGammaT, -1.0f, 1.0f));
    float dphi;
    if (p < kHairMaxScatteringEvents)
    {
        dphi = phiFunction(p, hairMaterialInteraction.gammaI, gammaT) +
               sampleTrimmedLogistic(u[0].y, hairMaterialInteraction.logisticDistributionScalar, -kPi, kPi);
    }
    else
    {
        dphi = u[0].y * kTwoPi;
    }

    const float phiI = phiO + dphi;
    wi = float3(sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI));

    pdf = 0.0f;
    [unroll]
    for (uint i = 0; i < kHairMaxScatteringEvents; ++i)
    {
        float sinThetaIp, cosThetaIp;
        if (i == 0)
        {
            sinThetaIp = sinThetaI * hairMaterialInteraction.cos2kAlpha[1] - cosThetaI * hairMaterialInteraction.sin2kAlpha[1];
            cosThetaIp = cosThetaI * hairMaterialInteraction.cos2kAlpha[1] + sinThetaI * hairMaterialInteraction.sin2kAlpha[1];
        }
        else if (i == 1)
        {
            sinThetaIp = sinThetaI * hairMaterialInteraction.cos2kAlpha[0] + cosThetaI * hairMaterialInteraction.sin2kAlpha[0];
            cosThetaIp = cosThetaI * hairMaterialInteraction.cos2kAlpha[0] - sinThetaI * hairMaterialInteraction.sin2kAlpha[0];
        }
        else if (i == 2)
        {
            sinThetaIp = sinThetaI * hairMaterialInteraction.cos2kAlpha[2] + cosThetaI * hairMaterialInteraction.sin2kAlpha[2];
            cosThetaIp = cosThetaI * hairMaterialInteraction.cos2kAlpha[2] - sinThetaI * hairMaterialInteraction.sin2kAlpha[2];
        }
        else
        {
            sinThetaIp = sinThetaI;
            cosThetaIp = cosThetaI;
        }

        cosThetaIp = abs(cosThetaIp);
        pdf += mP(cosThetaIp, cosThetaO, sinThetaIp, sinThetaO, hairMaterialInteraction.v[i]) *
               apPdf[i] *
               nP(dphi, i, hairMaterialInteraction.logisticDistributionScalar, hairMaterialInteraction.gammaI, gammaT);
    }
    pdf += mP(cosThetaI, cosThetaO, sinThetaI, sinThetaO, hairMaterialInteraction.v[kHairMaxScatteringEvents]) *
           apPdf[kHairMaxScatteringEvents] *
           kOneOverTwoPi;

    if (pdf > 1e-3f)
    {
        weight = hairChiangBSDFEval(hairMaterialInteraction, wi, wo);
    }
    else
    {
        pdf = 0.0f;
        weight = float3(0.0f, 0.0f, 0.0f);
    }

    return (pdf > 0.0f);
}

public bool sampleChiangBSDF(in const HairMaterialData hairMaterialData,
                            in const HairInteractionSurface hairInteractionSurface,
                            in float2 u[2],
                            out float3 wi,
                            out float pdf,
                            out float3 weight,
                            out HairLobeType lobeType)
{
    const float3 tangentWorld = hairInteractionSurface.tangent;
    const float3 biTangentWorld = cross(hairInteractionSurface.shadingNormal, tangentWorld);
    const float3x3 hairTangentBasis = float3x3(tangentWorld, biTangentWorld, hairInteractionSurface.shadingNormal); // TBN

    const float3 viewVectorLocal = mul(hairTangentBasis, hairInteractionSurface.incidentRayDirection);

    const HairMaterialInteraction hairMaterialInteraction = createHairMaterialInteraction(hairMaterialData, hairInteractionSurface);
    if (sampleChiangBSDF(hairMaterialInteraction, viewVectorLocal, u, wi, pdf, weight, lobeType))
    {
        wi = mul(transpose(hairTangentBasis), wi);
        return true;
    }
    else
    {
        return false;
    }
}

/************************************************
    Separate Chiang
************************************************/

public float3 hairSeparateChiangBSDFEval(in const HairMaterialSeparateChiangInteraction hairMaterialSeparateChiangInteraction,
                                        in const float3 wi,
                                        in const float3 wo)
{
    const float sinThetaO = wo.x;
    const float cosThetaO = sqrt01(1.0f - sinThetaO * sinThetaO);
    const float phiO = atan2safe(wo.z, wo.y);

    const float sinThetaI = wi.x;
    const float cosThetaI = sqrt01(1.0f - sinThetaI * sinThetaI);
    const float phiI = atan2safe(wi.z, wi.y);

    // Compute refracted ray.
    const float sinThetaT = sinThetaO / hairMaterialSeparateChiangInteraction.ior;
    const float cosThetaT = sqrt01(1.0f - sinThetaT * sinThetaT);

    const float etap = sqrt0(hairMaterialSeparateChiangInteraction.ior * hairMaterialSeparateChiangInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
    const float sinGammaT = hairMaterialSeparateChiangInteraction.h / etap;
    const float cosGammaT = sqrt01(1.0f - sinGammaT * sinGammaT);
    const float gammaT = asin(clamp(sinGammaT, -1.0f, 1.0f));

    // Compute the transmittance T of a single path through the cylinder
    const float tmp = -2.0f * cosGammaT / cosThetaT;
    const float3 T = exp(hairMaterialSeparateChiangInteraction.absorptionCoefficient * tmp);

    // Evaluate hair BCSDF for each lobe
    const float phi = phiI - phiO;
    float3 ap[kHairMaxScatteringEvents + 1];
    apSeparateChiang(hairMaterialSeparateChiangInteraction, cosThetaO, T, ap);
    float3 result = 0.0f;

    [unroll]
    for (uint p = 0; p < kHairMaxScatteringEvents; ++p)
    {
        float sinThetaOp, cosThetaOp;
        if (p == 0)
        {
            sinThetaOp = sinThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[1];
            cosThetaOp = cosThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[1];
        }
        else if (p == 1)
        {
            sinThetaOp = sinThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[0];
            cosThetaOp = cosThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[0];
        }
        else if (p == 2)
        {
            sinThetaOp = sinThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[2];
            cosThetaOp = cosThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[2];
        }
        else
        {
            sinThetaOp = sinThetaO;
            cosThetaOp = cosThetaO;
        }

        cosThetaOp = abs(cosThetaOp);
        result += mP(cosThetaOp, cosThetaI, sinThetaOp, sinThetaI, hairMaterialSeparateChiangInteraction.v[p]) *
                  ap[p] *
                  nP(phi, p, hairMaterialSeparateChiangInteraction.logisticDistributionScalar[p], hairMaterialSeparateChiangInteraction.gammaI, gammaT);
    }

    // Compute contribution of remaining terms after kHairMaxScatteringEvents
    result += mP(cosThetaO, cosThetaI, sinThetaO, sinThetaI, hairMaterialSeparateChiangInteraction.v[kHairMaxScatteringEvents]) *
              ap[kHairMaxScatteringEvents] *
              kOneOverTwoPi;

    // We omit this computation in BSDF, because the cosThetaI_N will be cancelled out when evaluate scattered radiance anyway
    // const float cosThetaI_N = wi.z; // The angle between wi and normal, which is (0, 0, 1) on local space
    // result = abs(cosThetaI_N) > 0.0f ? result / abs(cosThetaI_N) : 0.0f;

    return max(result, 0.0f);
}

public float3 hairSeparateChiangBSDFEval(in const HairMaterialSeparateChiangData hairMaterialSeparateChiangData,
                                        in const HairInteractionSurface hairInteractionSurface,
                                        in const float3 wi)
{
    const float3 tangentWorld = hairInteractionSurface.tangent;
    const float3 biTangentWorld = cross(hairInteractionSurface.shadingNormal, tangentWorld);
    const float3x3 hairTangentBasis = float3x3(tangentWorld, biTangentWorld, hairInteractionSurface.shadingNormal); // TBN

    const float3 lightVectorLocal = mul(hairTangentBasis, wi);
    const float3 viewVectorLocal = mul(hairTangentBasis, hairInteractionSurface.incidentRayDirection);

    const HairMaterialSeparateChiangInteraction hairMaterialSeparateChiangInteraction =
      createHairMaterialSeparateChiangInteraction(hairMaterialSeparateChiangData, hairInteractionSurface);
    return hairSeparateChiangBSDFEval(hairMaterialSeparateChiangInteraction, lightVectorLocal, viewVectorLocal);
}

public bool sampleSeparateChiangBSDF(in const HairMaterialSeparateChiangInteraction hairMaterialSeparateChiangInteraction,
                                    in const float3 wo,
                                    in float2 u[2],
                                    out float3 wi,
                                    out float pdf,
                                    out float3 weight,
                                    out HairLobeType lobeType)
{
    const float sinThetaO = wo.x;
    const float cosThetaO = sqrt01(1.0f - sinThetaO * sinThetaO);
    const float phiO = atan2safe(wo.z, wo.y);

    // Determine which term p to sample for hair scattering.
    float apPdf[kHairMaxScatteringEvents + 1];
    computeSeparateChiangApPdf(hairMaterialSeparateChiangInteraction, cosThetaO, apPdf);

    uint p = 0;
    float vp = hairMaterialSeparateChiangInteraction.v[0];
    {
        [unroll]
        for (uint i = 0; i < kHairMaxScatteringEvents; ++i)
        {
            if (u[0].x >= apPdf[i])
            {
                u[0].x -= apPdf[i];
                p = i + 1;
                vp = hairMaterialSeparateChiangInteraction.v[i + 1];
            }
            else
            {
                break;
            }
        }
    }

    float sinThetaOp = sinThetaO;
    float cosThetaOp = cosThetaO;
    if (p == 0)
    {
        sinThetaOp = sinThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[1] - cosThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[1];
        cosThetaOp = cosThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[1] + sinThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[1];

        lobeType = kHairLobeTypeR;
    }
    else if (p == 1)
    {
        sinThetaOp = sinThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[0] + cosThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[0];
        cosThetaOp = cosThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[0] - sinThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[0];

        lobeType = kHairLobeTypeTT;
    }
    else if (p == 2)
    {
        sinThetaOp = sinThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[2] + cosThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[2];
        cosThetaOp = cosThetaO * hairMaterialSeparateChiangInteraction.cos2kAlpha[2] - sinThetaO * hairMaterialSeparateChiangInteraction.sin2kAlpha[2];

        lobeType = kHairLobeTypeTRT;
    }
    else
    {
        lobeType = kHairLobeTypeTRT;
    }

    // Sample Mp to compute thetaO
    u[1].x = max(u[1].x, 1e-5f);
    const float cosTheta = 1.0f + vp * log(u[1].x + (1.0f - u[1].x) * exp(-2.0f / vp));
    const float sinTheta = sqrt01(1.0f - cosTheta * cosTheta);
    const float cosPhi = cos(u[1].y * kTwoPi);
    const float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
    const float cosThetaI = sqrt01(1.0f - sinThetaI * sinThetaI);

    // Sample Np to compute dphi
    const float etap = sqrt0(hairMaterialSeparateChiangInteraction.ior * hairMaterialSeparateChiangInteraction.ior - sinThetaO * sinThetaO) / cosThetaO;
    const float sinGammaT = hairMaterialSeparateChiangInteraction.h / etap;
    const float gammaT = asin(clamp(sinGammaT, -1.0f, 1.0f));
    float dphi;
    if (p < kHairMaxScatteringEvents)
    {
        dphi = phiFunction(p, hairMaterialSeparateChiangInteraction.gammaI, gammaT) +
               sampleTrimmedLogistic(u[0].y, hairMaterialSeparateChiangInteraction.logisticDistributionScalar[p], -kPi, kPi);
    }
    else
    {
        dphi = u[0].y * kTwoPi;
    }

    const float phiI = phiO + dphi;
    wi = float3(sinThetaI, cosThetaI * cos(phiI), cosThetaI * sin(phiI));

    pdf = 0.0f;
    [unroll]
    for (uint i = 0; i < kHairMaxScatteringEvents; ++i)
    {
        float sinThetaIp, cosThetaIp;
        if (i == 0)
        {
            sinThetaIp = sinThetaI * hairMaterialSeparateChiangInteraction.cos2kAlpha[1] - cosThetaI * hairMaterialSeparateChiangInteraction.sin2kAlpha[1];
            cosThetaIp = cosThetaI * hairMaterialSeparateChiangInteraction.cos2kAlpha[1] + sinThetaI * hairMaterialSeparateChiangInteraction.sin2kAlpha[1];
        }
        else if (i == 1)
        {
            sinThetaIp = sinThetaI * hairMaterialSeparateChiangInteraction.cos2kAlpha[0] + cosThetaI * hairMaterialSeparateChiangInteraction.sin2kAlpha[0];
            cosThetaIp = cosThetaI * hairMaterialSeparateChiangInteraction.cos2kAlpha[0] - sinThetaI * hairMaterialSeparateChiangInteraction.sin2kAlpha[0];
        }
        else if (i == 2)
        {
            sinThetaIp = sinThetaI * hairMaterialSeparateChiangInteraction.cos2kAlpha[2] + cosThetaI * hairMaterialSeparateChiangInteraction.sin2kAlpha[2];
            cosThetaIp = cosThetaI * hairMaterialSeparateChiangInteraction.cos2kAlpha[2] - sinThetaI * hairMaterialSeparateChiangInteraction.sin2kAlpha[2];
        }
        else
        {
            sinThetaIp = sinThetaI;
            cosThetaIp = cosThetaI;
        }

        cosThetaIp = abs(cosThetaIp);
        pdf += mP(cosThetaIp, cosThetaO, sinThetaIp, sinThetaO, hairMaterialSeparateChiangInteraction.v[i]) *
               apPdf[i] *
               nP(dphi, i, hairMaterialSeparateChiangInteraction.logisticDistributionScalar[p], hairMaterialSeparateChiangInteraction.gammaI, gammaT);
    }
    pdf += mP(cosThetaI, cosThetaO, sinThetaI, sinThetaO, hairMaterialSeparateChiangInteraction.v[kHairMaxScatteringEvents]) *
           apPdf[kHairMaxScatteringEvents] *
           kOneOverTwoPi;

    if (pdf > 1e-3f)
    {
        weight = hairSeparateChiangBSDFEval(hairMaterialSeparateChiangInteraction, wi, wo);
    }
    else
    {
        pdf = 0.0f;
        weight = float3(0.0f, 0.0f, 0.0f);
    }

    return (pdf > 0.0f);
}

public bool sampleSeparateChiangBSDF(in const HairMaterialSeparateChiangData hairMaterialSeparateChiangData,
                                    in const HairInteractionSurface hairInteractionSurface,
                                    in float2 u[2],
                                    out float3 wi,
                                    out float pdf,
                                    out float3 weight,
                                    out HairLobeType lobeType)
{
    const float3 tangentWorld = hairInteractionSurface.tangent;
    const float3 biTangentWorld = cross(hairInteractionSurface.shadingNormal, tangentWorld);
    const float3x3 hairTangentBasis = float3x3(tangentWorld, biTangentWorld, hairInteractionSurface.shadingNormal); // TBN

    const float3 viewVectorLocal = mul(hairTangentBasis, hairInteractionSurface.incidentRayDirection);

    const HairMaterialSeparateChiangInteraction hairMaterialSeparateChiangInteraction =
        createHairMaterialSeparateChiangInteraction(hairMaterialSeparateChiangData, hairInteractionSurface);
    if (sampleSeparateChiangBSDF(hairMaterialSeparateChiangInteraction, viewVectorLocal, u, wi, pdf, weight, lobeType))
    {
        wi = mul(transpose(hairTangentBasis), wi);
        return true;
    }
    else
    {
        return false;
    }
}
} // namespace Metallic.RTXCR
