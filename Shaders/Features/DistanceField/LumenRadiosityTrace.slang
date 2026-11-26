module LumenRadiosityTrace;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceField;
import LumenGlobalRay;
import LumenRadiosity;
import LumenSky;

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

// UE 5.7.4 LumenRadiosity.cpp:91–112,946–949. The reference distances are in centimeters.
static const float kRadiosityMinTraceDistance = 0.10f;
static const float kRadiositySurfaceBias = 0.05f;

// UE 5.7.4 LumenRadiosity.usf:82, opaque Global SDF and unblurred skylight on miss.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenTraceRadiosityMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadiosityParams params = getParameters<LumenRadiosityParams>();
    uint2 atlasCoord;
    if (!getRadiosityDispatchTexel(params, thread, 1u, atlasCoord)) {
        return;
    }

    uint2 probeCoord = atlasCoord / kRadiosityRayResolution;
    uint2 traceTexelCoord = atlasCoord % kRadiosityRayResolution;
    uint2 probeAtlasTexel = probeCoord * kRadiosityProbeSpacing + radiosityProbeJitter(params.temporalIndex);

    float3 radiance = 0.0f;
    uint traceClass = 0u;
    if (params.traceValid != 0u) {
        RadiosityTexel texel = radiosityGetTexel(params, probeAtlasTexel);
        if (texel.valid) {
            float3 worldPosition = texel.worldPosition;
            float3 worldNormal = texel.worldNormal;
            float2 probeTexelCenter = radiosityProbeTexelCenter(params, texel);
            float3 worldRayDirection = radiosityRay(texel, traceTexelCoord, probeTexelCenter).xyz;
            worldPosition += worldNormal * kRadiositySurfaceBias;
            float3 samplePosition = worldPosition + kRadiositySurfaceBias * worldRayDirection;
            LumenGlobalTraceRay ray = {float4(samplePosition, kRadiosityMinTraceDistance),
                float4(worldRayDirection, params.maxTraceDistance)};
            LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
            LumenGlobalTraceResult traceResult = traceLumenGlobalRay(scene, ray);
            traceClass = traceResult.state.x;

            if (traceClass == kGlobalDistanceFieldHit) {
                radiance = traceResult.radiance.xyz;
            } else {
                radiance = sampleLumenSky(params.sky, worldRayDirection);
            }
            float maxLighting = max(radiance.x, max(radiance.y, radiance.z));
            if (maxLighting > params.maxRayIntensity) {
                radiance *= params.maxRayIntensity / maxLighting;
            }
        }
    }

    RadiosityImage outputTrace = resolveUniform(params.outputTrace);
    outputTrace[atlasCoord] = float4(radiance, float(traceClass));
}

// UE 5.7.4 LumenRadiosity.usf:184. Each Metallic card occupies a single mapped tile.
void sampleTraceRadianceAtlas(LumenRadiosityParams params, Texture2D<float4> traceRadiance,
    RadiosityTexel gatherProbeTexel, uint2 cardProbeOrigin, int2 probeCoordInCard,
    uint2 traceTexelCoord, inout float3 radiance, inout float weightSum)
{
    uint probesPerCardEdge = params.tileResolution / kRadiosityProbeSpacing;
    if (all(probeCoordInCard >= 0) && all(probeCoordInCard < int(probesPerCardEdge))) {
        uint2 probeCoord = cardProbeOrigin + uint2(probeCoordInCard);
        uint2 probeAtlasTexel = probeCoord * kRadiosityProbeSpacing + radiosityProbeJitter(params.temporalIndex);
        RadiosityTexel probeTexel = radiosityGetTexel(params, probeAtlasTexel);
        if (probeTexel.valid) {
            float weight = min(1.0f,
                radiosityPlaneWeight(gatherProbeTexel.worldPosition, gatherProbeTexel.worldNormal, probeTexel.worldPosition));
            if (weight > 0.0f) {
                uint2 atlasTexel = probeCoord * kRadiosityRayResolution + traceTexelCoord;
                radiance += traceRadiance.Load(int3(atlasTexel, 0)).xyz * weight;
                weightSum += weight;
            }
        }
    }
}

// UE 5.7.4 LumenRadiosity.usf:250, kernel size one with plane weighting and no probe occlusion.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenFilterRadiosityMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadiosityParams params = getParameters<LumenRadiosityParams>();
    uint2 atlasCoord;
    if (!getRadiosityDispatchTexel(params, thread, 1u, atlasCoord)) {
        return;
    }

    RadiosityImage outputFiltered = resolveUniform(params.outputFiltered);
    if (params.traceValid == 0u) {
        outputFiltered[atlasCoord] = float4(0.0f);
        return;
    }

    Texture2D<float4> traceRadiance = resolveUniform(params.traceRadiance);
    float4 trace = traceRadiance.Load(int3(atlasCoord, 0));
    float3 radiance = trace.xyz;
    uint2 probeCoord = atlasCoord / kRadiosityRayResolution;
    uint2 traceTexelCoord = atlasCoord % kRadiosityRayResolution;
    uint2 probeAtlasTexel = probeCoord * kRadiosityProbeSpacing + radiosityProbeJitter(params.temporalIndex);
    RadiosityTexel texel = radiosityGetTexel(params, probeAtlasTexel);
    if (texel.valid) {
        const float centerWeight = 2.0f;
        radiance *= centerWeight;
        float totalWeight = centerWeight;

        uint probesPerCardEdge = params.tileResolution / kRadiosityProbeSpacing;
        uint2 cardProbeOrigin = (probeCoord / probesPerCardEdge) * probesPerCardEdge;
        int2 probeCoordInCard = int2(probeCoord - cardProbeOrigin);
        const int2 neighborOffsets[4] = {int2(0, 1), int2(1, 0), int2(0, -1), int2(-1, 0)};
        [unroll]
        for (uint i = 0u; i < 4u; ++i) {
            sampleTraceRadianceAtlas(params, traceRadiance, texel, cardProbeOrigin, probeCoordInCard + neighborOffsets[i],
                traceTexelCoord, radiance, totalWeight);
        }

        // The reference rendering commandlet's scalar BlueNoise input is BlackDummy.
        radiance = radiosityQuantizeForRenderTarget(radiance / totalWeight, 0.0f);
    }

    outputFiltered[atlasCoord] = float4(radiance, trace.w);
}
