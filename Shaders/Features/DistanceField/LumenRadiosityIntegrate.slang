module LumenRadiosityIntegrate;

import ShaderCore;
import ParameterRoot;
import LumenRadiosity;

using Metallic;
using Metallic.Lumen;

struct RadiositySH
{
    float4 red;
    float4 green;
    float4 blue;
};

uint2 packRadiositySH(float4 sh)
{
    return uint2(packHalf2x16(sh.xy), packHalf2x16(sh.zw));
}

float4 unpackRadiositySH(uint2 sh)
{
    return float4(unpackHalf2x16ToFloat(sh.x), unpackHalf2x16ToFloat(sh.y));
}

// UE 5.7.4 SHCommon.ush, SHBasisFunction for the two-band basis.
float4 radiositySHBasis(float3 direction)
{
    float4 result;
    result.x = 0.282095f;
    result.y = -0.488603f * direction.y;
    result.z = 0.488603f * direction.z;
    result.w = -0.488603f * direction.x;
    return result;
}

// UE 5.7.4 SHCommon.ush, CalcDiffuseTransferSH.
float4 radiosityDiffuseTransferSH(float3 normal, float exponent)
{
    float4 result = radiositySHBasis(normal);
    float l0 = 2.0f * SH::kPi / (1.0f + 1.0f * exponent);
    float l1 = 2.0f * SH::kPi / (2.0f + 1.0f * exponent);
    result.x *= l0;
    result.yzw *= l1;
    return result;
}

// UE 5.7.4 LumenRadiosity.usf, LumenRadiosityConvertToSH. Each dispatched
// texel corresponds to one probe of the fixed physical card atlas.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadiosityConvertToSHMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadiosityParams params = getParameters<LumenRadiosityParams>();
    uint2 probeAtlasCoord;
    if (!getRadiosityDispatchTexel(params, thread, kRadiosityProbeSpacing, probeAtlasCoord)) {
        return;
    }

    uint2 probeTexel = probeAtlasCoord * kRadiosityProbeSpacing + radiosityProbeJitter(params.temporalIndex);
    RadiosityTexel texel = radiosityGetTexel(params, probeTexel);
    RadiositySH irradianceSH = {};
    float numValidSamples = 0.0f;
    if (texel.valid && params.traceValid != 0u) {
        Texture2D<float4> radianceAtlas = resolveUniform(params.filteredRadiance);
        for (uint traceY = 0u; traceY < kRadiosityRayResolution; ++traceY) {
            for (uint traceX = 0u; traceX < kRadiosityRayResolution; ++traceX) {
                uint2 traceTexelCoord = uint2(traceX, traceY);
                uint2 traceAtlasCoord = probeAtlasCoord * kRadiosityRayResolution + traceTexelCoord;
                float3 traceRadiance = radianceAtlas.Load(int3(traceAtlasCoord, 0)).xyz;
                float4 ray = radiosityRay(texel, traceTexelCoord); // Direction and PDF.
                float4 basis = radiositySHBasis(ray.xyz);
                float3 radianceOverPDF = traceRadiance / ray.w;
                irradianceSH.red += basis * radianceOverPDF.r;
                irradianceSH.green += basis * radianceOverPDF.g;
                irradianceSH.blue += basis * radianceOverPDF.b;
                numValidSamples += 1.0f;
            }
        }
    }
    if (numValidSamples > 0.0f) {
        float invNumValidSamples = 1.0f / numValidSamples;
        irradianceSH.red *= invNumValidSamples;
        irradianceSH.green *= invNumValidSamples;
        irradianceSH.blue *= invNumValidSamples;
    }

    uint probeAddress = probeAtlasCoord.x + probeAtlasCoord.y * (params.atlasSize / kRadiosityProbeSpacing);
    resolveUniform(params.outputSHRed)[probeAddress] = packRadiositySH(irradianceSH.red);
    resolveUniform(params.outputSHGreen)[probeAddress] = packRadiositySH(irradianceSH.green);
    resolveUniform(params.outputSHBlue)[probeAddress] = packRadiositySH(irradianceSH.blue);
}

// UE 5.7.4 LumenRadiosity.usf, SampleRadiositySH. The fixed atlas has one
// physical tile per card and the default software probe occlusion is disabled.
void sampleRadiositySH(LumenRadiosityParams params, RadiosityTexel texel, uint2 cardAtlasOrigin,
    uint2 probeCoordInCard, float interpolationWeight, inout RadiositySH irradianceSH, inout float weightSum)
{
    uint2 probesPerCard = params.tileResolution / kRadiosityProbeSpacing;
    if (all(probeCoordInCard < probesPerCard)) {
        uint2 coordInCard = probeCoordInCard * kRadiosityProbeSpacing + radiosityProbeJitter(params.temporalIndex);
        uint2 probeAtlasCoord = (cardAtlasOrigin + coordInCard) / kRadiosityProbeSpacing;
        if (all(coordInCard < params.tileResolution)) {
            RadiosityTexel neighborProbe = radiosityGetTexel(params, cardAtlasOrigin + coordInCard);
            if (neighborProbe.valid) {
                float planeWeight = radiosityPlaneWeight(texel.worldPosition, texel.worldNormal, neighborProbe.worldPosition);
                float weight = min(interpolationWeight, planeWeight);
                if (weight > 0.0f) {
                    uint probeAddress = probeAtlasCoord.x + probeAtlasCoord.y * (params.atlasSize / kRadiosityProbeSpacing);
                    irradianceSH.red += unpackRadiositySH(resolveUniform(params.probeSHRed)[probeAddress]) * weight;
                    irradianceSH.green += unpackRadiositySH(resolveUniform(params.probeSHGreen)[probeAddress]) * weight;
                    irradianceSH.blue += unpackRadiositySH(resolveUniform(params.probeSHBlue)[probeAddress]) * weight;
                    weightSum += weight;
                }
            }
        }
    }
}

// UE 5.7.4 SurfaceCache/LumenSurfaceCache.ush CombineFinalLighting, with the
// default diffuse color boost and neutral view override.
float3 combineRadiosityLighting(float3 encodedAlbedo, float3 emissive, float3 directLighting, float3 indirectLighting)
{
    float3 albedo = encodedAlbedo * encodedAlbedo;
    float3 diffuseLambert = albedo * (1.0f / SH::kPi);
    float3 finalLighting = (directLighting + indirectLighting) * diffuseLambert + emissive;
    bool3 finiteLighting = (asuint(finalLighting) & 0x7f800000u) != 0x7f800000u;
    return max(select(!finiteLighting, 0.0f, finalLighting), 0.0f);
}

float3 loadAndCombineRadiosityLighting(LumenRadiosityParams params, uint2 atlasCoord, float3 texelRadiance)
{
    const int3 address = int3(atlasCoord, 0);
    float3 albedo = resolveUniform(params.cardAlbedo).Load(address).xyz;
    float3 emissive = resolveUniform(params.cardEmissive).Load(address).xyz;
    float3 directLighting = resolveUniform(params.cardDirectLighting).Load(address).xyz;
    float3 finalLighting = combineRadiosityLighting(albedo, emissive, directLighting, texelRadiance);
    return radiosityQuantizeForRenderTarget(finalLighting, int3(atlasCoord, params.viewFrameIndex % 8u + 2u));
}

#ifndef METALLIC_LUMEN_PERSISTENT_CARDS
// UE combines direct and retained indirect before tracing updated card probes.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadiosityRefreshDenseLightingMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadiosityParams params = getParameters<LumenRadiosityParams>();
    uint2 atlasCoord = thread.xy;
    if (any(atlasCoord >= params.atlasSize)) { return; }

    const int3 address = int3(atlasCoord, 0);
    float3 finalLighting = 0.0f;
    if (resolveUniform(params.scene.cardDepth).Load(address) < 1.0f) {
        const float3 history = resolveUniform(params.previousIndirect).Load(address).xyz;
        finalLighting = loadAndCombineRadiosityLighting(params, atlasCoord, history);
    }
    resolveUniform(params.currentLighting)[atlasCoord] = float4(finalLighting, 0.0f);
}
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadiosityIntegrateMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadiosityParams params = getParameters<LumenRadiosityParams>();
    uint2 atlasCoord;
    if (!getRadiosityDispatchTexel(params, thread, 1u, atlasCoord)) {
        return;
    }

    RadiosityTexel texel = radiosityGetTexel(params, atlasCoord);
    float3 texelRadiance = 0.0f;
    float weightSum = 0.0f;
    float newNumFramesAccumulated = 0.0f;
    float3 finalLighting = 0.0f;
    if (texel.valid) {
        if (params.traceValid != 0u) {
            // UE 5.7.4 LumenRadiosity.usf, LumenRadiosityIntegrateCS.
            uint2 coordInCard = atlasCoord % params.tileResolution;
            uint2 cardAtlasOrigin = atlasCoord - coordInCard;
            uint2 probeFullResCoord = uint2(max(float2(coordInCard) - float2(radiosityProbeJitter(params.temporalIndex)), 0.0f));
            uint2 probeCoord00 = probeFullResCoord / kRadiosityProbeSpacing;
            uint2 probeCoord10 = probeCoord00 + uint2(1u, 0u);
            uint2 probeCoord01 = probeCoord00 + uint2(0u, 1u);
            uint2 probeCoord11 = probeCoord00 + uint2(1u, 1u);

            const float bilinearExpand = 1.0f;
            float2 bilinearWeights = (float2(probeFullResCoord) - float2(probeCoord00 * kRadiosityProbeSpacing) +
                bilinearExpand) / (float(kRadiosityProbeSpacing) + 2.0f * bilinearExpand);
            float4 weights = float4(
                (1.0f - bilinearWeights.x) * (1.0f - bilinearWeights.y),
                bilinearWeights.x * (1.0f - bilinearWeights.y),
                (1.0f - bilinearWeights.x) * bilinearWeights.y,
                bilinearWeights.x * bilinearWeights.y);

            RadiositySH irradianceSHWeighted = {};
            sampleRadiositySH(params, texel, cardAtlasOrigin, probeCoord00, weights.x, irradianceSHWeighted, weightSum);
            sampleRadiositySH(params, texel, cardAtlasOrigin, probeCoord10, weights.y, irradianceSHWeighted, weightSum);
            sampleRadiositySH(params, texel, cardAtlasOrigin, probeCoord01, weights.z, irradianceSHWeighted, weightSum);
            sampleRadiositySH(params, texel, cardAtlasOrigin, probeCoord11, weights.w, irradianceSHWeighted, weightSum);

            float4 diffuseTransferSH = radiosityDiffuseTransferSH(texel.worldNormal, 1.0f);
            texelRadiance = max(float3(0.0f), float3(dot(irradianceSHWeighted.red, diffuseTransferSH),
                dot(irradianceSHWeighted.green, diffuseTransferSH), dot(irradianceSHWeighted.blue, diffuseTransferSH)));
            if (weightSum > 0.0f) {
                texelRadiance /= weightSum;
            }

#ifdef METALLIC_LUMEN_PERSISTENT_CARDS
            float4 history = resolveUniform(params.outputIndirect)[atlasCoord];
#else
            float4 history = 0.0f;
            if (params.historyValid != 0u) {
                history = resolveUniform(params.previousIndirect).Load(int3(atlasCoord, 0));
            }
#endif
            float numFramesAccumulated = history.w;
            newNumFramesAccumulated = min(numFramesAccumulated + 1.0f, float(kRadiosityMaxFrames));
            float alpha = 1.0f / (1.0f + numFramesAccumulated);
            texelRadiance = lerp(history.xyz, texelRadiance, alpha);
            texelRadiance = radiosityQuantizeForRenderTarget(texelRadiance, int3(atlasCoord, params.temporalIndex + 2u));

            // Retain the intermediate atlas's PF_FloatRGBA precision before
            // CombineLumenSceneLightingCS reads it at the texel center.
            texelRadiance = float3(unpackHalf2x16ToFloat(packHalf2x16(texelRadiance.xy)),
                unpackHalf2x16ToFloat(packHalf2x16(float2(texelRadiance.z, 0.0f))).x);
        }

        finalLighting = loadAndCombineRadiosityLighting(params, atlasCoord, texelRadiance);
    }

    // Alpha holds the exact integer represented by UE's separate UNORM8
    // accumulation count. RGB holds the incoming irradiance before albedo.
    float4 indirect = float4(texelRadiance, newNumFramesAccumulated);
    resolveUniform(params.outputIndirect)[atlasCoord] = indirect;
    resolveUniform(params.finalLighting)[atlasCoord] = float4(finalLighting, 0.0f);
#ifndef METALLIC_LUMEN_PERSISTENT_CARDS
    resolveUniform(params.currentIndirect)[atlasCoord] = indirect;
    resolveUniform(params.currentLighting)[atlasCoord] = float4(finalLighting, 0.0f);
#endif
}

#ifdef METALLIC_LUMEN_PERSISTENT_CARDS
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadiosityRefreshCardsMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadiosityParams params = getParameters<LumenRadiosityParams>();
    uint2 atlasCoord;
    bool reset;
    bool active;
    if (!getRadiosityRefreshTexel(params, thread, atlasCoord, reset, active)) { return; }

    RadiosityImage indirect = resolveUniform(params.outputIndirect);
    if (reset) {
        indirect[atlasCoord] = 0.0f;
        resolveUniform(params.outputTrace)[atlasCoord] = 0.0f;
        resolveUniform(params.outputFiltered)[atlasCoord] = 0.0f;
        if (all(atlasCoord % kRadiosityProbeSpacing == 0u)) {
            const uint2 probeAtlasCoord = atlasCoord / kRadiosityProbeSpacing;
            const uint probeAddress = probeAtlasCoord.x + probeAtlasCoord.y * (params.atlasSize / kRadiosityProbeSpacing);
            resolveUniform(params.outputSHRed)[probeAddress] = 0u;
            resolveUniform(params.outputSHGreen)[probeAddress] = 0u;
            resolveUniform(params.outputSHBlue)[probeAddress] = 0u;
        }
    }

    float3 finalLighting = 0.0f;
    if (active && resolveUniform(params.scene.cardDepth).Load(int3(atlasCoord, 0)) < 1.0f) {
        const float3 texelRadiance = reset ? float3(0.0f) : indirect[atlasCoord].xyz;
        finalLighting = loadAndCombineRadiosityLighting(params, atlasCoord, texelRadiance);
    }
    resolveUniform(params.finalLighting)[atlasCoord] = float4(finalLighting, 0.0f);
}
#endif
