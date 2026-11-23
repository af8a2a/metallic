module LumenRadiosity;

import ShaderCore;
import LumenGlobalRay;
import LumenProbeMath;
import LumenSurfaceCache;

using Metallic;

namespace Metallic.Lumen
{
public static const uint kRadiosityProbeSpacing = 4u;
public static const uint kRadiosityRayResolution = 4u;
public static const uint kRadiosityMaxFrames = 4u;
public static const uint kRadiosityDispatchCardsPerRow = 256u;
public static const uint kRadiosityScheduledCount = 0u;
public static const uint kRadiosityRefreshCount = 1u;

// The last _Texture argument is the SPIR-V Rgba16f image format (2). The
// persistent atlas reads its own texel through the typed UAV before overwriting
// it, which avoids requiring Vulkan's formatless storage image read feature.
public typealias RadiosityImage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;

public struct LumenRadiosityParams
{
    public LumenGlobalTraceBindings scene;
    public ResourceHandle<Texture2D<float4>> cardNormal;
    public ResourceHandle<Texture2D<float4>> cardAlbedo;
    public ResourceHandle<Texture2D<float4>> cardEmissive;
    public ResourceHandle<Texture2D<float4>> cardDirectLighting;
    public ResourceHandle<Texture2D<float4>> previousIndirect;
    public ResourceHandle<Texture2D<float4>> traceRadiance;
    public ResourceHandle<Texture2D<float4>> filteredRadiance;
    public ResourceHandle<StructuredBuffer<uint2>> probeSHRed;
    public ResourceHandle<StructuredBuffer<uint2>> probeSHGreen;
    public ResourceHandle<StructuredBuffer<uint2>> probeSHBlue;
    public ResourceHandle<RadiosityImage> outputTrace;
    public ResourceHandle<RadiosityImage> outputFiltered;
    public ResourceHandle<RWStructuredBuffer<uint2>> outputSHRed;
    public ResourceHandle<RWStructuredBuffer<uint2>> outputSHGreen;
    public ResourceHandle<RWStructuredBuffer<uint2>> outputSHBlue;
    public ResourceHandle<RadiosityImage> outputIndirect;
    public ResourceHandle<RadiosityImage> finalLighting;
    public ResourceHandle<RadiosityImage> currentIndirect;
    public ResourceHandle<RadiosityImage> currentLighting;
    public uint atlasSize;
    public uint tileResolution;
    public uint clipmapCount;
    public uint historyValid;
    public uint temporalIndex;
    public float maxTraceDistance;
    public uint viewFrameIndex;
#ifdef METALLIC_LUMEN_PERSISTENT_CARDS
    public ResourceHandle<StructuredBuffer<uint2>> scheduledCards;
    public ResourceHandle<StructuredBuffer<uint2>> refreshCards;
    public ResourceHandle<StructuredBuffer<uint>> updateState;
#endif
};

public bool getRadiosityDispatchTexel(inout LumenRadiosityParams params, uint3 thread, uint stride,
    out uint2 atlasCoord)
{
#ifdef METALLIC_LUMEN_PERSISTENT_CARDS
    const uint edge = params.tileResolution / stride;
    const uint width = (edge + 7u) / 8u * 8u;
    const uint listIndex = thread.x / width + thread.z * kRadiosityDispatchCardsPerRow;
    const uint2 localCoord = uint2(thread.x % width, thread.y);
    if (any(localCoord >= edge) ||
        listIndex >= resolveUniform(params.updateState)[kRadiosityScheduledCount]) {
        atlasCoord = 0u;
        return false;
    }
    const uint2 scheduled = resolveUniform(params.scheduledCards)[listIndex];
    const uint tilesPerRow = params.atlasSize / params.tileResolution;
    atlasCoord = uint2(scheduled.x % tilesPerRow, scheduled.x / tilesPerRow) * edge + localCoord;
    params.temporalIndex = scheduled.y;
#else
    atlasCoord = thread.xy;
    if (any(atlasCoord >= params.atlasSize / stride)) { return false; }
#endif
    return true;
}

#ifdef METALLIC_LUMEN_PERSISTENT_CARDS
public bool getRadiosityRefreshTexel(LumenRadiosityParams params, uint3 thread, out uint2 atlasCoord,
    out bool reset, out bool active)
{
    const uint edge = params.tileResolution;
    const uint width = (edge + 7u) / 8u * 8u;
    const uint listIndex = thread.x / width + thread.z * kRadiosityDispatchCardsPerRow;
    const uint2 localCoord = uint2(thread.x % width, thread.y);
    if (any(localCoord >= edge) ||
        listIndex >= resolveUniform(params.updateState)[kRadiosityRefreshCount]) {
        atlasCoord = 0u;
        reset = false;
        active = false;
        return false;
    }
    const uint2 selected = resolveUniform(params.refreshCards)[listIndex];
    const uint tilesPerRow = params.atlasSize / edge;
    atlasCoord = uint2(selected.x % tilesPerRow, selected.x / tilesPerRow) * edge + localCoord;
    reset = (selected.y & 1u) != 0u;
    active = (selected.y & 2u) != 0u;
    return true;
}
#endif

public struct RadiosityTexel
{
    public bool valid;
    public float3 worldPosition;
    public float3 worldNormal;
};

// UE 5.7.4 LumenRadiosity.ush:145, full-resolution indirect atlas.
public RadiosityTexel radiosityGetTexel(LumenRadiosityParams params, uint2 atlasTexel)
{
    RadiosityTexel texel = {false, float3(0.0f), float3(0.0f)};
    Texture2D<float> depthAtlas = resolveUniform(params.scene.cardDepth);
    float depth = depthAtlas.Load(int3(atlasTexel, 0));
    if (depth < 1.0f) {
        uint2 tile = atlasTexel / params.tileResolution;
        uint cardIndex = tile.x + tile.y * (params.atlasSize / params.tileResolution);
        StructuredBuffer<SurfaceCacheCard> cards = resolveUniform(params.scene.cards);
        SurfaceCacheCard card = cards[cardIndex];
        Texture2D<float4> normalAtlas = resolveUniform(params.cardNormal);
        float2 cardUV = (float2(atlasTexel - tile * params.tileResolution) + 0.5f) / float(params.tileResolution);
        texel.valid = true;
        texel.worldPosition = surfaceCacheWorldPosition(card, cardUV, depth);
        texel.worldNormal = decodeSurfaceCacheNormal(card, normalAtlas.Load(int3(atlasTexel, 0)).xy);
    }
    return texel;
}

// UE 5.7.4 LumenRadiosity.ush:90. Each resident card updates its own temporal phase.
public uint2 radiosityProbeJitter(uint temporalIndex)
{
    return uint2(lumenHammersley16(temporalIndex % kRadiosityMaxFrames, kRadiosityMaxFrames,
        uint2(0x4ae4, 0x9bdb)) * kRadiosityProbeSpacing);
}

// UE 5.7.4 MonteCarlo.ush:27.
float3x3 radiosityTangentBasisFrisvad(float3 tangentZ)
{
    float3 tangentX;
    float3 tangentY;
    if (tangentZ.z < -0.9999999f) {
        tangentX = float3(0, -1, 0);
        tangentY = float3(-1, 0, 0);
    } else {
        float a = 1.0f / (1.0f + tangentZ.z);
        float b = -tangentZ.x * tangentZ.y * a;
        tangentX = float3(1.0f - tangentZ.x * tangentZ.x * a, b, -tangentZ.x);
        tangentY = float3(b, 1.0f - tangentZ.y * tangentZ.y * a, -tangentZ.y);
    }
    return float3x3(tangentX, tangentY, tangentZ);
}

// UE 5.7.4 LumenRadiosity.ush:236,283 and MonteCarlo.ush:231, default uniform hemisphere branch.
public float4 radiosityRay(RadiosityTexel texel, uint2 traceTexelCoord)
{
    // GetBlueNoiseDummyParameters supplies BlackDummy for rendering commandlets.
    const float2 probeTexelCenter = 0.0f;
    float2 probeUV = (float2(traceTexelCoord) + probeTexelCenter) / float(kRadiosityRayResolution);
    float phi = 2.0f * kLumenProbePi * probeUV.x;
    float cosTheta = probeUV.y;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float3 localRayDirection = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    float3 worldRayDirection = mul(localRayDirection, radiosityTangentBasisFrisvad(texel.worldNormal));
    float pdf = 1.0f / (2.0f * kLumenProbePi);
    return float4(worldRayDirection, pdf);
}

// UE 5.7.4 LumenRadiosity.usf:175. Convert the reference 0.01 cm denominator bias to meters.
public float radiosityPlaneWeight(float3 worldPosition, float3 worldNormal, float3 probeWorldPosition)
{
    float4 texelPlane = float4(worldNormal, dot(probeWorldPosition, worldNormal));
    float planeDistance = abs(dot(float4(worldPosition, -1), texelPlane));
    float relativeDistance = max(planeDistance / (length(probeWorldPosition - worldPosition) + 0.0001f), 0.1f);
    float depthWeight = exp2(-100.0f * (relativeDistance * relativeDistance));
    return depthWeight > 0.01f ? 1.0f : 0.0f;
}

// UE 5.7.4 Quantization.ush / Quantization.cpp, PF_FloatRGBA with ten mantissa bits.
public float3 radiosityQuantizeForRenderTarget(float3 color, float jitter)
{
    float3 error = color * (1.0f / 1024.0f);
    error = asfloat(asuint(error) & ~0x007FFFFFu);
    return color + error * jitter;
}

public float3 radiosityQuantizeForRenderTarget(float3 color, int3 coord)
{
    uint random = lumenRand3DPCG16(coord).x;
    float jitter = float(random) * (1.0f / 65536.0f);
    return radiosityQuantizeForRenderTarget(color, jitter);
}
} // namespace Metallic.Lumen
