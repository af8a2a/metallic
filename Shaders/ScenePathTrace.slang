#ifndef STF_SHADER_STAGE
#define STF_SHADER_STAGE 6
#endif
#ifndef STF_SHADER_MODEL_MAJOR
#define STF_SHADER_MODEL_MAJOR 6
#endif
#ifndef STF_SHADER_MODEL_MINOR
#define STF_SHADER_MODEL_MINOR 6
#endif
#ifndef STF_ALLOW_WAVE_READ
#define STF_ALLOW_WAVE_READ 0
#endif
#include "../External/Rtxtf/STFSamplerState.hlsli"
#include "NeuralTextureSampling.slang"

#ifndef METALLIC_HAS_RTXCR
#define METALLIC_HAS_RTXCR 0
#endif

#if METALLIC_HAS_RTXCR
#include <rtxcr/HairChiangBSDF.hlsli>
#endif

// ---------------------------------------------------------------------------
// Radiance cache integration (reference: RTXGI v2 SDK)
//
// The same entry point is compiled into several permutations:
//   (no defines)      plain path tracing
//   SHARC_UPDATE=1    sparse SHaRC update pass (populates the hash-grid cache)
//   SHARC_QUERY=1     full-resolution pass with SHaRC early termination
//   NRC_UPDATE=1      NRC training pass at reduced resolution
//   NRC_QUERY=1       full-resolution pass creating NRC queries; outputs
//                     linear HDR that the NRC resolve adds onto before the
//                     separate tonemap pass
// ---------------------------------------------------------------------------
#if defined(SHARC_UPDATE) || defined(SHARC_QUERY)
#define METALLIC_PT_SHARC 1
#else
#define METALLIC_PT_SHARC 0
#endif

#if defined(NRC_UPDATE) || defined(NRC_QUERY)
#define METALLIC_PT_NRC 1
#else
#define METALLIC_PT_NRC 0
#endif

#if METALLIC_PT_SHARC
#include "Sharc/SharcCommon.h"
#endif

#include "Nrc/Nrc.hlsli"

// Cache mode pushed through the per-frame parameter buffer.
#define METALLIC_CACHE_MODE_OFF 0
#define METALLIC_CACHE_MODE_SHARC 1
#define METALLIC_CACHE_MODE_NRC 2

// Per-frame radiance-cache parameters (binding 20). Layout must match
// ScenePathTraceCacheParams in BuiltinPassCommon.h byte for byte.
struct ScenePathTraceCacheParams
{
    float4 sharcCameraPosition;
    float4 sharcCameraPositionPrev;
    float sharcSceneScale;
    uint sharcEntriesNum;
    uint sharcAccumulationFrameNum;
    uint sharcStaleFrameNumMax;
    uint frameIndex;
    uint cacheMode;
    uint sharcUpdateStride;
    uint width;
    uint height;
    uint trainingWidth;
    uint trainingHeight;
    NrcConstants nrc;
};

[[vk::binding(20, 0)]]
StructuredBuffer<ScenePathTraceCacheParams> gCacheParams;

#if METALLIC_PT_SHARC
[[vk::binding(21, 0)]]
RWStructuredBuffer<uint64_t> gSharcHashEntries;

[[vk::binding(22, 0)]]
RWStructuredBuffer<SharcAccumulationData> gSharcAccumulation;

[[vk::binding(23, 0)]]
RWStructuredBuffer<SharcPackedData> gSharcResolved;
#endif

#if METALLIC_PT_NRC
[[vk::binding(24, 0)]]
RWStructuredBuffer<NrcPackedQueryPathInfo> gNrcQueryPathInfo;

[[vk::binding(25, 0)]]
RWStructuredBuffer<NrcPackedTrainingPathInfo> gNrcTrainingPathInfo;

[[vk::binding(26, 0)]]
RWStructuredBuffer<NrcPackedPathVertex> gNrcTrainingPathVertices;

[[vk::binding(27, 0)]]
RWStructuredBuffer<NrcRadianceParams> gNrcQueryRadianceParams;

[[vk::binding(28, 0)]]
RWStructuredBuffer<uint> gNrcCounters;
#endif


[[vk::binding(0, 0)]]
RaytracingAccelerationStructure gScene;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gOutput;

struct PathTraceVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 texcoord;
};

struct PathTracePrimitive
{
    uint firstVertex;
    uint vertexCount;
    uint firstIndex;
    uint indexCount;
    uint flags;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct PathTraceInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint flags;
    float rayConeLodConstant;
};

struct PathTraceTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint ntcTextureSetIndex;
    uint ntcChannelMapping;
    float4 transform0;
    float4 transform1;
};

struct PathTraceMaterial
{
    float4 baseColor;
    float4 emissive;
    float4 params;
    float4 textureParams;
    float4 glassParams;
    float4 attenuationColor;
    float4 diffuseTransmission;
    float4 rtxcrHairBaseColor;
    float4 rtxcrHairParams0;
    float4 rtxcrHairParams1;
    float4 rtxcrHairDiffuseTint;
    PathTraceTextureInfo baseColorTexture;
    PathTraceTextureInfo metallicRoughnessTexture;
    PathTraceTextureInfo normalTexture;
    PathTraceTextureInfo occlusionTexture;
    PathTraceTextureInfo emissiveTexture;
    PathTraceTextureInfo transmissionTexture;
    PathTraceTextureInfo thicknessTexture;
    PathTraceTextureInfo diffuseTransmissionTexture;
    PathTraceTextureInfo diffuseTransmissionColorTexture;
};

[[vk::binding(2, 0)]]
StructuredBuffer<PathTraceVertex> gVertices;

[[vk::binding(3, 0)]]
StructuredBuffer<uint> gIndices;

[[vk::binding(4, 0)]]
StructuredBuffer<PathTracePrimitive> gPrimitives;

[[vk::binding(5, 0)]]
StructuredBuffer<PathTraceInstance> gInstances;

[[vk::binding(6, 0)]]
StructuredBuffer<PathTraceMaterial> gMaterials;

[[vk::binding(7, 0)]]
RWTexture2D<float4> gHistoryCurrent;

[[vk::binding(8, 0)]]
RWTexture2D<float4> gHistoryPrevious;

[[vk::binding(9, 0)]]
Texture2D<float4> gMaterialTextures[256];

[[vk::binding(10, 0)]]
Texture2D<float4> gEnvironmentMap;

[[vk::binding(13, 0)]]
Texture2D<float> gEnvironmentImportancePdf;


static const uint kMaxMaterialTextures = 256u;
static const uint kInvalidMaterialTextureIndex = 0xffffffffu;
static const uint kPrimitiveHasAuthoredTangents = 1u;
static const uint kEnvironmentModeProcedural = 0u;
static const uint kEnvironmentModeMap = 1u;
static const uint kEnvironmentModeDisabled = 2u;

struct ScenePathTracePush
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 previousEye;
    float4 previousCenter;
    float4 previousUpProjection;
    float4 previousViewport;
    float4 previousClipOrtho;
    uint width;
    uint height;
    uint maxDepth;
    uint samples;
    uint accumulationFrame;
    uint hasHistory;
    uint enableAccumulation;
    uint previousCameraValid;
    uint materialTextureCount;
    float bitangentFlip;
    float environmentIntensity;
    float environmentRotationRadians;
    uint environmentMode;
    uint environmentVisible;
    uint outputLinear;      // when set, write linear HDR (tonemap happens in a follow-up pass; used by NRC query mode)
    uint cacheMode;         // METALLIC_CACHE_MODE_*
    uint ntcTextureSetCount;
    uint debugView;
    uint debugFlags;
};

struct Ray
{
    float3 origin;
    float3 direction;
};

struct RayCone
{
    float width;
    float spreadAngle;
};

struct HitInfo
{
    float t;
    float3 position;
    float3 normal;
    float3 geometryNormal;
    float4 tangent;
    float3 bitangent;
    float2 texcoord;
    uint materialIndex;
    uint instanceIndex;
    uint primitiveIndex;
    uint triangleIndex;
    float rayConeLodConstant;
    bool frontFace;
};

static const float kPi = 3.14159265358979323846;
static const float3 kEnvironmentKeyLightDirection = float3(0.35, 0.88, 0.22);
static const float3 kEnvironmentKeyLightRadiance = float3(5.2, 4.8, 4.2);

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float3 safeTriangleNormal(float3 edge1, float3 edge2, float3 fallback)
{
    float3 unnormalizedNormal = cross(edge1, edge2);
    float normalLengthSquared = dot(unnormalizedNormal, unnormalizedNormal);
    float edgeLengthProduct = dot(edge1, edge1) * dot(edge2, edge2);
    if (edgeLengthProduct <= 0.0 ||
        normalLengthSquared <= edgeLengthProduct * 1e-12) {
        return fallback;
    }
    return unnormalizedNormal * rsqrt(normalLengthSquared);
}

uint hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float rand(inout uint state)
{
    state = hash(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

bool isRtxcrHairMaterial(PathTraceMaterial material)
{
    return material.rtxcrHairBaseColor.w > 0.5;
}

#if METALLIC_HAS_RTXCR
RTXCR_HairMaterialData makeRtxcrHairMaterial(PathTraceMaterial material)
{
    RTXCR_HairMaterialData hair;
    hair.baseColor = saturate(material.rtxcrHairBaseColor.rgb);
    hair.longitudinalRoughness = clamp(material.rtxcrHairParams0.x, 0.02, 1.0);
    hair.azimuthalRoughness = clamp(material.rtxcrHairParams0.y, 0.02, 1.0);
    hair.ior = clamp(material.rtxcrHairParams0.z, 1.01, 3.0);
    hair.eta = 1.0 / hair.ior;
    hair.fresnelApproximation = 1u;
    hair.absorptionModel = RTXCR_HairAbsorptionModel_Normalized;
    hair.melanin = saturate(material.rtxcrHairParams1.x);
    hair.melaninRedness = saturate(material.rtxcrHairParams1.y);
    hair.cuticleAngleInDegrees = clamp(material.rtxcrHairParams0.w, -10.0, 10.0);
    return hair;
}

RTXCR_HairInteractionSurface makeRtxcrHairSurface(
    float3 stableNormal,
    float3 tangent,
    float3 viewDirection)
{
    RTXCR_HairInteractionSurface surface;
    surface.shadingNormal = safeNormalize(stableNormal, float3(0.0, 0.0, 1.0));
    float3 tangentFallback = abs(surface.shadingNormal.y) < 0.99
        ? cross(float3(0.0, 1.0, 0.0), surface.shadingNormal)
        : cross(float3(1.0, 0.0, 0.0), surface.shadingNormal);
    surface.tangent = safeNormalize(
        tangent - surface.shadingNormal * dot(surface.shadingNormal, tangent),
        tangentFallback);
    surface.incidentRayDirection = safeNormalize(viewDirection, -surface.shadingNormal);
    return surface;
}
#endif

float4 rand4(inout uint state)
{
    return float4(rand(state), rand(state), rand(state), rand(state));
}

float safeLog2(float value)
{
    return log2(max(value, 0.0000001));
}

RayCone makeRayCone(float width, float spreadAngle)
{
    RayCone rayCone;
    rayCone.width = width;
    rayCone.spreadAngle = spreadAngle;
    return rayCone;
}

RayCone propagateRayCone(RayCone rayCone, float distance)
{
    rayCone.width = abs(rayCone.width) + abs(rayCone.spreadAngle) * max(distance, 0.0);
    return rayCone;
}

float rayConeSpreadAngleExpansionByScatterPdf(float bsdfScatterPdf)
{
    float inversePdfSolidAngle = (1.0 / max(bsdfScatterPdf, 0.000001)) / (2.0 * kPi);
    return 0.3 * 2.0 * acos(max(-1.0, 1.0 - inversePdfSolidAngle));
}

RayCone expandRayConeAfterScatter(RayCone rayCone, float bsdfScatterPdf)
{
    rayCone.spreadAngle = min(
        rayCone.spreadAngle + rayConeSpreadAngleExpansionByScatterPdf(bsdfScatterPdf),
        2.0 * kPi);
    return rayCone;
}

float3 rayDirectionFromPixel(float2 pixel, ScenePathTracePush push)
{
    float aspect = max(push.viewport.x, 0.001);
    float2 uv = pixel / float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(push.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));

    if (push.upProjection.w > 0.5) {
        return forward;
    }

    float fovRadians = clamp(push.viewport.w, 0.017453292, 3.12413936);
    float halfHeight = tan(fovRadians * 0.5);
    return safeNormalize(
        forward + right * (ndc.x * aspect * halfHeight) + up * (ndc.y * halfHeight),
        forward);
}

float3 rayOriginFromPixel(float2 pixel, ScenePathTracePush push)
{
    if (push.upProjection.w <= 0.5) {
        return push.eye.xyz;
    }

    float aspect = max(push.viewport.x, 0.001);
    float2 uv = pixel / float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(push.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float halfHeight = max(push.clipOrtho.z * 0.5, 0.0001);
    return push.eye.xyz + right * (ndc.x * halfHeight * aspect) + up * (ndc.y * halfHeight);
}

RayCone primaryRayConeFromPixel(float2 pixel, ScenePathTracePush push)
{
    if (push.upProjection.w > 0.5) {
        float aspect = max(push.viewport.x, 0.001);
        float halfHeight = max(push.clipOrtho.z * 0.5, 0.0001);
        float pixelWidth = halfHeight * 2.0 * aspect / float(max(push.width, 1u));
        float pixelHeight = halfHeight * 2.0 / float(max(push.height, 1u));
        return makeRayCone(max(pixelWidth, pixelHeight), 0.0);
    }

    float3 centerDirection = rayDirectionFromPixel(pixel, push);
    float3 xDirection = rayDirectionFromPixel(pixel + float2(1.0, 0.0), push);
    float3 yDirection = rayDirectionFromPixel(pixel + float2(0.0, 1.0), push);
    float spreadAngle = max(
        acos(clamp(dot(centerDirection, xDirection), -1.0, 1.0)),
        acos(clamp(dot(centerDirection, yDirection), -1.0, 1.0))) * 2.0;
    return makeRayCone(0.0, spreadAngle);
}

float3 offsetRay(float3 position, float3 normal)
{
    return position + normal * 0.001;
}

float3 cameraForwardFrom(float4 eye, float4 center)
{
    return safeNormalize(center.xyz - eye.xyz, float3(0.0, 0.0, -1.0));
}

float3 cameraRightFrom(float3 forward, float3 cameraUp)
{
    return safeNormalize(cross(forward, safeNormalize(cameraUp, float3(0.0, 1.0, 0.0))), float3(1.0, 0.0, 0.0));
}

float3 cameraUpFrom(float3 forward, float3 right)
{
    return safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
}

bool projectWorldToUv(
    float3 worldPosition,
    float4 eye,
    float4 center,
    float4 upProjection,
    float4 viewport,
    float4 clipOrtho,
    out float2 uv)
{
    float3 forward = cameraForwardFrom(eye, center);
    float3 right = cameraRightFrom(forward, upProjection.xyz);
    float3 up = cameraUpFrom(forward, right);
    float3 view = worldPosition - eye.xyz;
    float aspect = max(viewport.x, 0.001);
    float2 ndc;
    if (upProjection.w > 0.5) {
        float halfHeight = max(clipOrtho.z * 0.5, 0.0001);
        ndc = float2(
            dot(view, right) / max(halfHeight * aspect, 0.0001),
            dot(view, up) / halfHeight);
    } else {
        float viewZ = dot(view, forward);
        if (viewZ <= max(clipOrtho.x, 0.0001)) {
            uv = float2(0.0, 0.0);
            return false;
        }
        float fovRadians = clamp(viewport.w, 0.017453292, 3.12413936);
        float halfHeight = max(tan(fovRadians * 0.5), 0.0001);
        ndc = float2(
            dot(view, right) / max(viewZ * aspect * halfHeight, 0.0001),
            dot(view, up) / max(viewZ * halfHeight, 0.0001));
    }
    uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return true;
}

float cameraLinearDepth(float3 worldPosition, ScenePathTracePush push)
{
    float3 forward = cameraForwardFrom(push.eye, push.center);
    return max(dot(worldPosition - push.eye.xyz, forward), 0.0);
}

float2 motionVectorToPrevious(float3 worldPosition, float2 currentSamplePixel, ScenePathTracePush push)
{
    if (push.previousCameraValid == 0u) {
        return float2(0.0, 0.0);
    }

    float2 previousUv;
    if (!projectWorldToUv(
        worldPosition,
        push.previousEye,
        push.previousCenter,
        push.previousUpProjection,
        push.previousViewport,
        push.previousClipOrtho,
        previousUv)) {
        return float2(0.0, 0.0);
    }
    float2 currentUv = currentSamplePixel / float2(max(push.width, 1u), max(push.height, 1u));
    return previousUv - currentUv;
}

float3 approxSpecularIntegralGGX(float3 specularReflectance, float alpha, float cosTheta)
{
    cosTheta = abs(cosTheta);

    float4 x = float4(1.0, cosTheta, cosTheta * cosTheta, cosTheta * cosTheta * cosTheta);
    float4 y = float4(1.0, alpha, alpha * alpha, alpha * alpha * alpha);
    float2x2 m1 = float2x2(
        0.995367, -1.38839,
        -0.24751, 1.97442);
    float3x3 m2 = float3x3(
        1.0, 2.68132, 52.366,
        16.0932, -3.98452, 59.3013,
        -5.18731, 255.259, 2544.07);
    float2x2 m3 = float2x2(
        -0.0564526, 3.82901,
        16.91, -11.0303);
    float3x3 m4 = float3x3(
        1.0, 4.11118, -1.37886,
        19.3254, -28.9947, 16.9514,
        0.545386, 96.0994, -79.4492);

    float bias = dot(mul(m1, x.xy), y.xy) / max(dot(mul(m2, x.xyw), y.xyw), 0.000001);
    float scale = dot(mul(m3, x.xy), y.xy) / max(dot(mul(m4, x.xzw), y.xyw), 0.000001);
    bias *= saturate(dot(specularReflectance, float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)) * 50.0);
    return specularReflectance * max(scale, 0.0) + max(bias, 0.0);
}

float3 environmentLocalDirectionToWorld(float3 direction, float rotationRadians)
{
    float sine = sin(rotationRadians);
    float cosine = cos(rotationRadians);
    return float3(
        direction.x * cosine + direction.z * sine,
        direction.y,
        -direction.x * sine + direction.z * cosine);
}

float3 environmentKeyLightDirection(ScenePathTracePush push)
{
    return safeNormalize(
        environmentLocalDirectionToWorld(kEnvironmentKeyLightDirection, push.environmentRotationRadians),
        float3(0.0, 1.0, 0.0));
}

float3 environmentKeyLightRadiance(ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled) {
        return float3(0.0, 0.0, 0.0);
    }
    return kEnvironmentKeyLightRadiance * max(push.environmentIntensity, 0.0);
}

float3 environmentLocalDirectionFromUv(float2 uv)
{
    float phi = (uv.x - 0.5) * (2.0 * kPi);
    float theta = uv.y * kPi;
    float sinTheta = sin(theta);
    return float3(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
}

float environmentTexelSolidAngle(uint y, uint width, uint height)
{
    float centerTheta = (float(y) + 0.5) * (kPi / float(max(height, 1u)));
    return max((2.0 * kPi / float(max(width, 1u))) * (kPi / float(max(height, 1u))) * sin(centerTheta), 0.0000001);
}

float misPowerHeuristic(float pdfA, float pdfB)
{
    float a = max(pdfA, 0.0);
    float b = max(pdfB, 0.0);
    a *= a;
    b *= b;
    float sum = a + b;
    return sum > 0.0 ? a / sum : 1.0;
}

struct EnvironmentLightSample
{
    float3 direction;
    float3 radiance;
    float pdfSolidAngle;
};

float loadEnvironmentImportanceWeight(uint2 position, uint mipLevel)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    gEnvironmentImportancePdf.GetDimensions(mipLevel, width, height, mipCount);
    return all(position < uint2(width, height))
        ? max(gEnvironmentImportancePdf.Load(int3(position, mipLevel)), 0.0)
        : 0.0;
}

bool sampleEnvironmentImportanceMipmap(
    inout uint seed,
    out uint2 position,
    out float probability)
{
    position = uint2(0u);
    probability = 1.0;
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    gEnvironmentImportancePdf.GetDimensions(0u, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        probability = 0.0;
        return false;
    }
    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
        float4 weights = float4(
            loadEnvironmentImportanceWeight(position + uint2(0u, 0u), uint(mipLevel)),
            loadEnvironmentImportanceWeight(position + uint2(0u, 1u), uint(mipLevel)),
            loadEnvironmentImportanceWeight(position + uint2(1u, 0u), uint(mipLevel)),
            loadEnvironmentImportanceWeight(position + uint2(1u, 1u), uint(mipLevel)));
        float weightSum = weights.x + weights.y + weights.z + weights.w;
        if (weightSum <= 0.0) {
            probability = 0.0;
            return false;
        }
        weights /= weightSum;
        float randomValue = rand(seed);
        if (randomValue < weights.x) {
            probability *= weights.x;
        } else if (randomValue < weights.x + weights.y) {
            position += uint2(0u, 1u);
            probability *= weights.y;
        } else if (randomValue < weights.x + weights.y + weights.z) {
            position += uint2(1u, 0u);
            probability *= weights.z;
        } else {
            position += uint2(1u, 1u);
            probability *= weights.w;
        }
    }
    return probability > 0.0;
}

float environmentImportanceTexelProbability(uint2 position)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    gEnvironmentImportancePdf.GetDimensions(0u, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u || any(position >= uint2(width, height))) {
        return 0.0;
    }
    float averageWeight = loadEnvironmentImportanceWeight(uint2(0u), mipCount - 1u);
    float totalWeight = averageWeight * float(width * height);
    return totalWeight > 0.0
        ? loadEnvironmentImportanceWeight(position, 0u) / totalWeight
        : 0.0;
}

bool sampleEnvironmentLight(ScenePathTracePush push, inout uint seed, out EnvironmentLightSample sample)
{
    sample.direction = float3(0.0, 1.0, 0.0);
    sample.radiance = float3(0.0, 0.0, 0.0);
    sample.pdfSolidAngle = 1.0;

    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) {
        return false;
    }

    if (push.environmentMode != kEnvironmentModeMap) {
        sample.direction = environmentKeyLightDirection(push);
        sample.radiance = environmentKeyLightRadiance(push);
        return max(sample.radiance.x, max(sample.radiance.y, sample.radiance.z)) > 0.0;
    }

    uint width = 1u;
    uint height = 1u;
    gEnvironmentMap.GetDimensions(width, height);
    uint textureTexelCount = width * height;
    if (width == 0u || height == 0u || textureTexelCount == 0u) {
        return false;
    }

    uint2 texel = uint2(0u);
    float texelProbability = 0.0;
    if (!sampleEnvironmentImportanceMipmap(seed, texel, texelProbability) ||
        texel.x >= width || texel.y >= height) {
        uint texelIndex = min(uint(rand(seed) * float(textureTexelCount)), textureTexelCount - 1u);
        texel = uint2(texelIndex % width, texelIndex / width);
        texelProbability = 1.0 / float(textureTexelCount);
    }

    float2 uv = (float2(texel) + float2(rand(seed), rand(seed))) / float2(width, height);
    float texelSolidAngle = environmentTexelSolidAngle(texel.y, width, height);

    float3 localDirection = environmentLocalDirectionFromUv(uv);
    sample.direction = safeNormalize(
        environmentLocalDirectionToWorld(localDirection, push.environmentRotationRadians),
        float3(0.0, 1.0, 0.0));
    sample.radiance = max(gEnvironmentMap.Load(int3(int2(texel), 0)).rgb, float3(0.0, 0.0, 0.0)) *
        max(push.environmentIntensity, 0.0);
    sample.pdfSolidAngle = texelProbability / texelSolidAngle;
    return max(sample.radiance.x, max(sample.radiance.y, sample.radiance.z)) > 0.0 && sample.pdfSolidAngle > 0.0;
}

float3 proceduralSkyRadiance(float3 direction, float rotationRadians)
{
    float t = saturate(direction.y * 0.5 + 0.5);
    float3 horizon = float3(0.62, 0.70, 0.78);
    float3 zenith = float3(0.12, 0.24, 0.48);
    float3 sunDirection = safeNormalize(
        environmentLocalDirectionToWorld(kEnvironmentKeyLightDirection, rotationRadians),
        float3(0.0, 1.0, 0.0));
    float sun = pow(saturate(dot(direction, sunDirection)), 256.0);
    return lerp(horizon, zenith, t) * 0.45 + sun * float3(7.0, 6.4, 5.3);
}

float2 environmentUvFromDirection(float3 direction, float rotationRadians)
{
    float phi = atan2(direction.z, direction.x) + rotationRadians;
    float u = frac(phi / (2.0 * kPi) + 0.5);
    float v = acos(clamp(direction.y, -1.0, 1.0)) / kPi;
    return float2(u, v);
}

float environmentPdfSolidAngle(float3 direction, ScenePathTracePush push)
{
    if (push.environmentMode != kEnvironmentModeMap || push.environmentIntensity <= 0.0) {
        return 0.0;
    }

    uint width = 1u;
    uint height = 1u;
    gEnvironmentMap.GetDimensions(width, height);
    uint textureTexelCount = width * height;
    if (width == 0u || height == 0u || textureTexelCount == 0u) {
        return 0.0;
    }

    float2 uv = environmentUvFromDirection(direction, push.environmentRotationRadians);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1u, height - 1u));
    float texelProbability = environmentImportanceTexelProbability(texel);
    if (texelProbability <= 0.0) {
        texelProbability = 1.0 / float(textureTexelCount);
    }
    return texelProbability / environmentTexelSolidAngle(texel.y, width, height);
}

float environmentLightMisWeight(ScenePathTracePush push, float environmentPdf, float bsdfPdf)
{
    return push.environmentMode == kEnvironmentModeMap
        ? misPowerHeuristic(environmentPdf, bsdfPdf)
        : 1.0;
}

float environmentBsdfMisWeight(ScenePathTracePush push, float bsdfPdf, float3 direction, bool sampledDelta)
{
    if (push.environmentMode != kEnvironmentModeMap || sampledDelta) {
        return 1.0;
    }
    return misPowerHeuristic(bsdfPdf, environmentPdfSolidAngle(direction, push));
}

float3 environmentMapRadiance(float3 direction, ScenePathTracePush push)
{
    uint width = 1u;
    uint height = 1u;
    gEnvironmentMap.GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return float3(0.0, 0.0, 0.0);
    }

    float2 uv = environmentUvFromDirection(direction, push.environmentRotationRadians);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1u, height - 1u));
    return max(gEnvironmentMap.Load(int3(texel, 0)).rgb, float3(0.0, 0.0, 0.0));
}

float3 environmentRadiance(float3 direction, ScenePathTracePush push, bool primaryMiss)
{
    if (push.environmentMode == kEnvironmentModeDisabled) {
        return float3(0.0, 0.0, 0.0);
    }
    if (primaryMiss && push.environmentVisible == 0u) {
        return float3(0.0, 0.0, 0.0);
    }

    float3 radiance = push.environmentMode == kEnvironmentModeMap
        ? environmentMapRadiance(direction, push)
        : proceduralSkyRadiance(direction, push.environmentRotationRadians);
    return radiance * max(push.environmentIntensity, 0.0);
}

float3 sampleCosineHemisphere(float3 normal, inout uint seed)
{
    float r0 = rand(seed);
    float r1 = rand(seed);
    float phi = 2.0 * kPi * r0;
    float cosTheta = sqrt(1.0 - r1);
    float sinTheta = sqrt(r1);

    float3 tangent = abs(normal.z) < 0.999
        ? safeNormalize(cross(float3(0.0, 0.0, 1.0), normal), float3(1.0, 0.0, 0.0))
        : safeNormalize(cross(float3(0.0, 1.0, 0.0), normal), float3(1.0, 0.0, 0.0));
    float3 bitangent = cross(normal, tangent);
    return safeNormalize(
        tangent * (cos(phi) * sinTheta) + bitangent * (sin(phi) * sinTheta) + normal * cosTheta,
        normal);
}

bool hasMaterialTexture(PathTraceTextureInfo textureInfo, ScenePathTracePush push)
{
    return textureInfo.textureIndex != kInvalidMaterialTextureIndex &&
        textureInfo.textureIndex < push.materialTextureCount &&
        textureInfo.textureIndex < kMaxMaterialTextures;
}

bool hasNeuralMaterialTexture(PathTraceTextureInfo textureInfo, ScenePathTracePush push)
{
    return metallicHasNeuralTexture(textureInfo.ntcTextureSetIndex, push.ntcTextureSetCount);
}

float2 transformMaterialUv(PathTraceTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

float textureTransformLodBias(PathTraceTextureInfo textureInfo)
{
    float determinant =
        textureInfo.transform0.x * textureInfo.transform1.y -
        textureInfo.transform0.y * textureInfo.transform1.x;
    return 0.5 * safeLog2(abs(determinant));
}

float rayConeTextureLod(
    PathTraceTextureInfo textureInfo,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    uint width,
    uint height,
    uint mipCount)
{
    float textureDimensionLod = 0.5 * safeLog2(float(max(width, 1u)) * float(max(height, 1u)));
    float normalTerm = sqrt(max(abs(dot(rayDirection, hit.geometryNormal)), 0.000001));
    float footprintLod = hit.rayConeLodConstant + safeLog2(max(abs(rayCone.width), 0.000001) / normalTerm);
    float mipLevel = footprintLod + textureDimensionLod + textureTransformLodBias(textureInfo);
    return clamp(mipLevel, 0.0, float(max(mipCount, 1u) - 1u));
}

STF_SamplerState makeMaterialStfSampler(ScenePathTracePush push, inout uint seed)
{
    STF_SamplerState samplerState = STF_SamplerState::Create(rand4(seed));
    samplerState.SetFrameIndex(push.accumulationFrame);
    samplerState.SetFilterType(STF_FILTER_TYPE_LINEAR);
    samplerState.SetMagMethod(STF_MAGNIFICATION_METHOD_NONE);
    samplerState.SetAnisoMethod(STF_ANISO_LOD_METHOD_NONE);
    samplerState.SetAddressingModes(uint3(STF_ADDRESS_MODE_WRAP, STF_ADDRESS_MODE_WRAP, STF_ADDRESS_MODE_WRAP));
    samplerState.SetReseedOnSample(false);
    return samplerState;
}

float4 loadMaterialTextureMip0(
    PathTraceTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push)
{
    if (hasNeuralMaterialTexture(textureInfo, push)) {
        uint2 dimensions;
        uint mipCount;
        if (!metallicNeuralTextureDimensions(
                textureInfo.ntcTextureSetIndex,
                push.ntcTextureSetCount,
                0u,
                dimensions,
                mipCount)) {
            return fallback;
        }
        float2 wrappedUv = frac(transformMaterialUv(textureInfo, uv));
        int2 texel = int2(min(
            uint2(wrappedUv * float2(dimensions)),
            dimensions - uint2(1u, 1u)));
        return metallicSampleNeuralTexture(
            textureInfo.ntcTextureSetIndex,
            push.ntcTextureSetCount,
            textureInfo.ntcChannelMapping,
            texel,
            0,
            fallback);
    }
    if (!hasMaterialTexture(textureInfo, push)) {
        return fallback;
    }

    uint width = 1u;
    uint height = 1u;
    uint textureIndex = textureInfo.textureIndex;
    gMaterialTextures[textureIndex].GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return fallback;
    }

    float2 wrappedUv = frac(transformMaterialUv(textureInfo, uv));
    uint2 texel = min(uint2(wrappedUv * float2(width, height)), uint2(width - 1u, height - 1u));
    return gMaterialTextures[textureIndex].Load(int3(texel, 0));
}

float4 loadMaterialTexture(
    PathTraceTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    inout uint seed)
{
    if (hasNeuralMaterialTexture(textureInfo, push)) {
        uint2 baseDimensions;
        uint mipCount;
        if (!metallicNeuralTextureDimensions(
                textureInfo.ntcTextureSetIndex,
                push.ntcTextureSetCount,
                0u,
                baseDimensions,
                mipCount)) {
            return fallback;
        }

        float mipLevel = rayConeTextureLod(
            textureInfo,
            rayCone,
            hit,
            rayDirection,
            baseDimensions.x,
            baseDimensions.y,
            mipCount);
        STF_SamplerState samplerState = makeMaterialStfSampler(push, seed);
        float3 samplePosition = samplerState.Texture2DGetSamplePosLevel(
            baseDimensions.x,
            baseDimensions.y,
            mipCount,
            transformMaterialUv(textureInfo, uv),
            mipLevel);
        uint sampleMip = min(uint(max(samplePosition.z, 0.0)), mipCount - 1u);
        uint2 mipDimensions;
        uint ignoredMipCount;
        metallicNeuralTextureDimensions(
            textureInfo.ntcTextureSetIndex,
            push.ntcTextureSetCount,
            sampleMip,
            mipDimensions,
            ignoredMipCount);
        bool border;
        samplePosition.xy = STF_ApplyAddressingMode2D(
            samplePosition.xy,
            mipDimensions,
            uint2(STF_ADDRESS_MODE_WRAP, STF_ADDRESS_MODE_WRAP),
            border);
        int2 texel = int2(floor(samplePosition.xy * float2(mipDimensions)));
        return metallicSampleNeuralTexture(
            textureInfo.ntcTextureSetIndex,
            push.ntcTextureSetCount,
            textureInfo.ntcChannelMapping,
            texel,
            int(sampleMip),
            fallback);
    }
    if (!hasMaterialTexture(textureInfo, push)) {
        return fallback;
    }

    uint width = 1u;
    uint height = 1u;
    uint mipCount = 1u;
    uint textureIndex = textureInfo.textureIndex;
    gMaterialTextures[textureIndex].GetDimensions(0, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        return fallback;
    }

    float mipLevel = rayConeTextureLod(textureInfo, rayCone, hit, rayDirection, width, height, mipCount);
    STF_SamplerState samplerState = makeMaterialStfSampler(push, seed);
    return samplerState.Texture2DLoadLevel(
        gMaterialTextures[textureIndex],
        transformMaterialUv(textureInfo, uv),
        mipLevel);
}

float4 loadMaterialTextureForMaterial(
    PathTraceTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    bool deterministicMip0,
    inout uint seed)
{
    if (deterministicMip0) {
        return loadMaterialTextureMip0(textureInfo, uv, fallback, push);
    }
    return loadMaterialTexture(textureInfo, uv, fallback, push, rayCone, hit, rayDirection, seed);
}

bool materialNeedsDeterministicTextureMip0(PathTraceMaterial material)
{
    // Specular transmission magnifies stochastic/material-unaware mips into hard surface bands.
    return material.glassParams.x > 0.01;
}

float2 triangleTexcoord(uint instanceIndex, uint triangleIndex, float2 bary)
{
    PathTraceInstance instance = gInstances[instanceIndex];
    PathTracePrimitive primitive = gPrimitives[instance.primitiveIndex];
    uint indexOffset = primitive.firstIndex + triangleIndex * 3u;
    uint i0 = primitive.firstVertex + gIndices[indexOffset + 0u];
    uint i1 = primitive.firstVertex + gIndices[indexOffset + 1u];
    uint i2 = primitive.firstVertex + gIndices[indexOffset + 2u];
    float3 weights = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    return gVertices[i0].texcoord.xy * weights.x +
        gVertices[i1].texcoord.xy * weights.y +
        gVertices[i2].texcoord.xy * weights.z;
}

bool alphaMaskAcceptsHit(uint instanceIndex, uint triangleIndex, float2 bary, ScenePathTracePush push)
{
    PathTraceInstance instance = gInstances[instanceIndex];
    PathTraceMaterial material = gMaterials[instance.materialIndex];
    if (abs(material.textureParams.w - 1.0) > 0.25) {
        return true;
    }

    float2 uv = triangleTexcoord(instanceIndex, triangleIndex, bary);
    float alpha = material.baseColor.a * loadMaterialTextureMip0(
        material.baseColorTexture,
        uv,
        float4(1.0, 1.0, 1.0, 1.0),
        push).a;
    return saturate(alpha) >= saturate(material.params.z);
}

void computeTriangleTangentSpace(
    float3 p0,
    float3 p1,
    float3 p2,
    float2 uv0,
    float2 uv1,
    float2 uv2,
    float3 normal,
    out float3 tangent,
    out float3 bitangent)
{
    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float2 deltaUv1 = uv1 - uv0;
    float2 deltaUv2 = uv2 - uv0;
    float determinant = deltaUv1.x * deltaUv2.y - deltaUv1.y * deltaUv2.x;

    if (abs(determinant) <= 0.0000001) {
        float3 referenceAxis = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
        tangent = safeNormalize(cross(referenceAxis, normal), float3(1.0, 0.0, 0.0));
        bitangent = safeNormalize(cross(normal, tangent), float3(0.0, 1.0, 0.0));
        return;
    }

    float invDeterminant = 1.0 / determinant;
    tangent = (edge1 * deltaUv2.y - edge2 * deltaUv1.y) * invDeterminant;
    bitangent = (edge2 * deltaUv1.x - edge1 * deltaUv2.x) * invDeterminant;
}

float3 transformObjectNormalToWorld(float3 normal, float3x4 worldToObject)
{
    return mul(normal, worldToObject).xyz;
}

float3 faceForwardShadingNormal(float3 normal, float3 geometryNormal, float3 rayDirection)
{
    float3 facingGeometryNormal = dot(geometryNormal, -rayDirection) < 0.0
        ? -geometryNormal
        : geometryNormal;
    if (dot(normal, facingGeometryNormal) < 0.0) {
        normal = -normal;
    }
    return safeNormalize(normal, facingGeometryNormal);
}

bool traceShadow(float3 origin, float3 direction, float maxDistance, ScenePathTracePush push)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = max(maxDistance, 0.001);

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(gScene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alphaMaskAcceptsHit(
                query.CandidateInstanceIndex(),
                query.CandidatePrimitiveIndex(),
                query.CandidateTriangleBarycentrics(),
                push)) {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

bool traceClosest(Ray ray, ScenePathTracePush push, out HitInfo hit)
{
    hit = HitInfo();
    hit.t = 0.0;
    hit.position = ray.origin;
    hit.normal = float3(0.0, 1.0, 0.0);
    hit.geometryNormal = hit.normal;
    hit.tangent = float4(1.0, 0.0, 0.0, 1.0);
    hit.bitangent = float3(0.0, 0.0, 1.0);
    hit.texcoord = float2(0.0, 0.0);
    hit.materialIndex = 0u;
    hit.instanceIndex = 0u;
    hit.primitiveIndex = 0u;
    hit.triangleIndex = 0u;
    hit.rayConeLodConstant = 0.0;
    hit.frontFace = true;

    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    rayDesc.TMin = max(push.clipOrtho.x, 0.0001);
    rayDesc.TMax = max(push.clipOrtho.y, rayDesc.TMin + 0.0001);

    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(gScene, RAY_FLAG_NONE, 0xff, rayDesc);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alphaMaskAcceptsHit(
                query.CandidateInstanceIndex(),
                query.CandidatePrimitiveIndex(),
                query.CandidateTriangleBarycentrics(),
                push)) {
            query.CommitNonOpaqueTriangleHit();
        }
    }

    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return false;
    }

    uint instanceIndex = query.CommittedInstanceIndex();
    PathTraceInstance instance = gInstances[instanceIndex];
    PathTracePrimitive primitive = gPrimitives[instance.primitiveIndex];
    uint triangleIndex = query.CommittedPrimitiveIndex();
    uint indexOffset = primitive.firstIndex + triangleIndex * 3u;

    uint i0 = primitive.firstVertex + gIndices[indexOffset + 0u];
    uint i1 = primitive.firstVertex + gIndices[indexOffset + 1u];
    uint i2 = primitive.firstVertex + gIndices[indexOffset + 2u];
    PathTraceVertex v0 = gVertices[i0];
    PathTraceVertex v1 = gVertices[i1];
    PathTraceVertex v2 = gVertices[i2];

    float2 bary = query.CommittedTriangleBarycentrics();
    float3 weights = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

    float3 p0 = v0.position.xyz;
    float3 p1 = v1.position.xyz;
    float3 p2 = v2.position.xyz;
    float2 uv0 = v0.texcoord.xy;
    float2 uv1 = v1.texcoord.xy;
    float2 uv2 = v2.texcoord.xy;
    float3 localPosition = p0 * weights.x + p1 * weights.y + p2 * weights.z;
    float3 localNormal = v0.normal.xyz * weights.x + v1.normal.xyz * weights.y + v2.normal.xyz * weights.z;
    float3 localNormalFallback = safeNormalize(localNormal, float3(0.0, 1.0, 0.0));
    float3 localGeometryNormal = safeTriangleNormal(
        p1 - p0,
        p2 - p0,
        localNormalFallback);
    localNormal = dot(localNormal, localNormal) > 0.000001 ? safeNormalize(localNormal, localGeometryNormal) : localGeometryNormal;
    float4 localTangent = v0.tangent * weights.x + v1.tangent * weights.y + v2.tangent * weights.z;
    float3 localTangentDirection = dot(localTangent.xyz, localTangent.xyz) > 0.000001
        ? safeNormalize(localTangent.xyz, float3(1.0, 0.0, 0.0))
        : float3(1.0, 0.0, 0.0);
    float tangentSign = localTangent.w >= 0.0 ? 1.0 : -1.0;
    hit.texcoord = uv0 * weights.x + uv1 * weights.y + uv2 * weights.z;

    float3 localTriangleTangent;
    float3 localTriangleBitangent;
    computeTriangleTangentSpace(p0, p1, p2, uv0, uv1, uv2, localNormal, localTriangleTangent, localTriangleBitangent);

    float4x3 objectToWorld = query.CommittedObjectToWorld4x3();
    float3x4 worldToObject = query.CommittedWorldToObject3x4();
    hit.position = float3(mul(float4(localPosition, 1.0), objectToWorld));
    hit.geometryNormal = safeNormalize(transformObjectNormalToWorld(localGeometryNormal, worldToObject), localGeometryNormal);
    hit.normal = safeNormalize(transformObjectNormalToWorld(localNormal, worldToObject), hit.geometryNormal);
    hit.frontFace = dot(hit.geometryNormal, -ray.direction) >= 0.0;

    float3 referenceAxis = abs(hit.normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 fallbackTangent = safeNormalize(cross(referenceAxis, hit.normal), float3(1.0, 0.0, 0.0));
    if ((primitive.flags & kPrimitiveHasAuthoredTangents) != 0u) {
        float3 worldTangent = safeNormalize(mul(localTangentDirection, worldToObject).xyz, fallbackTangent);
        worldTangent = safeNormalize(worldTangent - hit.normal * dot(hit.normal, worldTangent), fallbackTangent);
        hit.tangent = float4(worldTangent, tangentSign);
        hit.bitangent = safeNormalize(cross(hit.normal, worldTangent) * tangentSign, cross(hit.normal, fallbackTangent));
    } else {
        float3 worldTangent = safeNormalize(mul(localTriangleTangent, worldToObject).xyz, fallbackTangent);
        worldTangent = safeNormalize(worldTangent - hit.normal * dot(hit.normal, worldTangent), fallbackTangent);
        float3 worldBitangent = safeNormalize(mul(localTriangleBitangent, worldToObject).xyz, cross(hit.normal, worldTangent));
        worldBitangent = safeNormalize(worldBitangent - hit.normal * dot(hit.normal, worldBitangent), cross(hit.normal, worldTangent));
        tangentSign = dot(cross(hit.normal, worldTangent), worldBitangent) < 0.0 ? -1.0 : 1.0;
        hit.tangent = float4(worldTangent, tangentSign);
        hit.bitangent = worldBitangent;
    }

    hit.tangent.w *= push.bitangentFlip;
    hit.bitangent *= push.bitangentFlip;

    hit.t = query.CommittedRayT();
    hit.materialIndex = instance.materialIndex;
    hit.instanceIndex = instanceIndex;
    hit.primitiveIndex = instance.primitiveIndex;
    hit.triangleIndex = triangleIndex;
    hit.rayConeLodConstant = instance.rayConeLodConstant;
    return true;
}

float luminanceMax(float3 value)
{
    return max(value.x, max(value.y, value.z));
}

float fresnelDielectric(float ior, float cosTheta)
{
    float f0 = (ior - 1.0) / (ior + 1.0);
    f0 *= f0;
    float x = 1.0 - saturate(cosTheta);
    return saturate(f0 + (1.0 - f0) * x * x * x * x * x);
}

bool refractDirection(float3 incident, float3 normal, float eta, out float3 transmitted)
{
    float cosTheta = saturate(dot(-incident, normal));
    float sin2Theta = max(0.0, 1.0 - cosTheta * cosTheta);
    float sin2Transmitted = eta * eta * sin2Theta;
    if (sin2Transmitted >= 1.0) {
        transmitted = float3(0.0, 0.0, 0.0);
        return false;
    }

    float cosTransmitted = sqrt(max(0.0, 1.0 - sin2Transmitted));
    transmitted = safeNormalize(eta * incident + (eta * cosTheta - cosTransmitted) * normal, incident);
    return true;
}

float3 powColor(float3 value, float exponent)
{
    float3 clamped = max(value, float3(0.001, 0.001, 0.001));
    return float3(pow(clamped.x, exponent), pow(clamped.y, exponent), pow(clamped.z, exponent));
}

float3 volumeTransmittance(float3 baseColor, float transmission, float thickness, float attenuationDistance, float3 attenuationColor, float distance)
{
    float exponent = 0.0;
    if (attenuationDistance > 0.0001) {
        exponent = max(distance, 0.0) / attenuationDistance;
    }

    float3 attenuation = powColor(max(attenuationColor, float3(0.001, 0.001, 0.001)), exponent);
    float3 baseTint = lerp(float3(1.0, 1.0, 1.0), powColor(baseColor, 0.5), transmission);
    return saturate(attenuation * baseTint);
}

float3 srgbToLinear(float3 color)
{
    return float3(
        color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4));
}

float fallbackBsdfPdf(
    float3 incidentDirection,
    float3 normal,
    float3 sampledDirection,
    float metallic,
    float roughness,
    float diffuseTransmission)
{
    float specularChance = saturate(0.08 + metallic * 0.72 + (1.0 - roughness) * 0.18);
    float transmissionChance = saturate(diffuseTransmission * (1.0 - metallic));
    float frontCos = saturate(dot(normal, sampledDirection));
    float backCos = saturate(dot(-normal, sampledDirection));
    float diffusePdf = frontCos / kPi;
    float transmissionPdf = backCos / kPi;

    float3 reflectedDirection = safeNormalize(reflect(incidentDirection, normal), normal);
    float specularPower = lerp(96.0, 8.0, roughness);
    float specularPdf = ((specularPower + 1.0) / (2.0 * kPi)) *
        pow(saturate(dot(reflectedDirection, sampledDirection)), specularPower);
    return specularChance * specularPdf +
        (1.0 - specularChance) *
            ((1.0 - transmissionChance) * diffusePdf + transmissionChance * transmissionPdf);
}


float3 applyNormalMap(
    float3 normal,
    float3 geometryNormal,
    float4 tangent,
    float3 bitangent,
    float4 normalSample,
    float normalScale)
{
    normal = safeNormalize(normal, geometryNormal);
    float3 referenceAxis = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 fallbackTangent = safeNormalize(cross(referenceAxis, normal), float3(1.0, 0.0, 0.0));
    float3 tangentDirection = tangent.xyz - normal * dot(normal, tangent.xyz);
    tangentDirection = safeNormalize(tangentDirection, fallbackTangent);
    float tangentSign = tangent.w >= 0.0 ? 1.0 : -1.0;
    float3 bitangentFallback = cross(normal, tangentDirection) * tangentSign;
    float3 bitangentDirection = safeNormalize(bitangent - normal * dot(normal, bitangent), bitangentFallback);

    float3 tangentNormal = float3(normalSample.xy * 2.0 - 1.0, normalSample.z * 2.0 - 1.0);
    tangentNormal.xy *= max(normalScale, 0.0);
    tangentNormal = safeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));
    return safeNormalize(
        tangentDirection * tangentNormal.x +
        bitangentDirection * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

// Per-path radiance-cache state. Members exist only in permutations that use
// them; the plain permutation carries just the glossy cone tracker.
struct PathTraceCacheContext
{
#if METALLIC_PT_SHARC
    SharcParameters sharc;
    SharcState sharcState;
#endif
#if METALLIC_PT_NRC
    NrcContext nrc;
    NrcPathState nrcPathState;
#endif
    float materialRoughnessPrev;
};

float3 pathTrace(Ray ray, RayCone rayCone, ScenePathTracePush push, inout PathTraceCacheContext cache, inout uint seed)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float previousBsdfPdf = 0.0;
    bool previousScatterDelta = true;
    uint maxDepth = clamp(push.maxDepth, 1u, 32u);
    cache.materialRoughnessPrev = 1.0;

#if METALLIC_PT_NRC
    NrcSetSampleIndex(cache.nrc, 0u);
#endif

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        if (!traceClosest(ray, push, hit)) {
            float misWeight = depth == 0u
                ? 1.0
                : environmentBsdfMisWeight(push, previousBsdfPdf, ray.direction, previousScatterDelta);
            float3 missRadiance = environmentRadiance(ray.direction, push, depth == 0u) * misWeight;
            radiance += throughput * missRadiance;
#if METALLIC_PT_SHARC
            SharcUpdateMiss(cache.sharc, cache.sharcState, missRadiance);
#endif
#if METALLIC_PT_NRC
            NrcUpdateOnMiss(cache.nrcPathState);
#endif
            break;
        }

#if METALLIC_PT_SHARC && SHARC_UPDATE
        // SHaRC accumulates each path segment independently: drop the signal
        // gathered for the previous segment before evaluating this hit.
        radiance = float3(0.0, 0.0, 0.0);
        throughput = float3(1.0, 1.0, 1.0);
#endif

        rayCone = propagateRayCone(rayCone, hit.t);
        PathTraceMaterial material = gMaterials[hit.materialIndex];
        bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
        float4 baseColorSample = loadMaterialTextureForMaterial(
            material.baseColorTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 metallicRoughnessSample = loadMaterialTextureForMaterial(
            material.metallicRoughnessTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 emissiveSample = loadMaterialTextureForMaterial(
            material.emissiveTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 normalSample = loadMaterialTextureForMaterial(
            material.normalTexture,
            hit.texcoord,
            float4(0.5, 0.5, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 occlusionSample = loadMaterialTextureForMaterial(
            material.occlusionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 transmissionSample = loadMaterialTextureForMaterial(
            material.transmissionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 thicknessSample = loadMaterialTextureForMaterial(
            material.thicknessTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 diffuseTransmissionSample = loadMaterialTextureForMaterial(
            material.diffuseTransmissionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 diffuseTransmissionColorSample = loadMaterialTextureForMaterial(
            material.diffuseTransmissionColorTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
        float3 stableShadingNormal = hit.normal;
        hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);
        float occlusion = lerp(1.0, saturate(occlusionSample.r), saturate(material.textureParams.y));

        float3 baseColor = saturate(material.baseColor.rgb * srgbToLinear(baseColorSample.rgb));
        float metallic = saturate(material.params.x * metallicRoughnessSample.b);
        float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.02, 1.0);
        float transmission = saturate(material.glassParams.x * transmissionSample.r);
        float ior = clamp(material.glassParams.y, 1.0, 3.0);
        float thickness = max(material.glassParams.z * thicknessSample.g, 0.0);
        float attenuationDistance = max(material.glassParams.w, 0.0);
        float3 attenuationColor = max(saturate(material.attenuationColor.rgb), float3(0.001, 0.001, 0.001));
        float diffuseTransmission = saturate(material.diffuseTransmission.a * diffuseTransmissionSample.a);
        float3 diffuseTransmissionColor = max(
            saturate(material.diffuseTransmission.rgb * srgbToLinear(diffuseTransmissionColorSample.rgb)),
            float3(0.001, 0.001, 0.001));
        float dielectricF0 = (ior - 1.0) / (ior + 1.0);
        dielectricF0 *= dielectricF0;
        float3 f0 = lerp(float3(dielectricF0, dielectricF0, dielectricF0), baseColor, metallic);

        float3 emissive = max(material.emissive.rgb * srgbToLinear(emissiveSample.rgb), float3(0.0, 0.0, 0.0));
        radiance += throughput * emissive;

#if METALLIC_HAS_RTXCR
        if (isRtxcrHairMaterial(material)) {
            RTXCR_HairMaterialData hairMaterial = makeRtxcrHairMaterial(material);
            RTXCR_HairInteractionSurface hairSurface = makeRtxcrHairSurface(
                stableShadingNormal,
                hit.tangent.xyz,
                -ray.direction);

            EnvironmentLightSample hairLightSample;
            if (sampleEnvironmentLight(push, seed, hairLightSample)) {
                float approximateHairPdf = fallbackBsdfPdf(
                    ray.direction,
                    hit.normal,
                    hairLightSample.direction,
                    0.0,
                    hairMaterial.longitudinalRoughness,
                    0.5);
                float lightMisWeight = environmentLightMisWeight(
                    push,
                    hairLightSample.pdfSolidAngle,
                    approximateHairPdf);
                float3 lightRadianceOverPdf =
                    hairLightSample.radiance * lightMisWeight /
                    max(hairLightSample.pdfSolidAngle, 0.000001);
                float3 shadowOffsetNormal = dot(hairLightSample.direction, hit.geometryNormal) >= 0.0
                    ? hit.geometryNormal
                    : -hit.geometryNormal;
                bool occluded = traceShadow(
                    offsetRay(hit.position, shadowOffsetNormal),
                    hairLightSample.direction,
                    10000.0,
                    push);
                if (!occluded) {
                    float3 hairBsdf = RTXCR_HairChiangBsdfEval(
                        hairMaterial,
                        hairSurface,
                        hairLightSample.direction);
                    radiance += throughput * max(hairBsdf, 0.0) * lightRadianceOverPdf;
                }
            }

            float2 hairRandom[2] = {
                float2(rand(seed), rand(seed)),
                float2(rand(seed), rand(seed)),
            };
            float3 hairDirection;
            float hairPdf;
            float3 hairWeight;
            RTXCR_HairLobeType hairLobe;
            if (!RTXCR_SampleChiangBsdf(
                    hairMaterial,
                    hairSurface,
                    hairRandom,
                    hairDirection,
                    hairPdf,
                    hairWeight,
                    hairLobe)) {
                break;
            }

            ray.direction = safeNormalize(hairDirection, reflect(ray.direction, hit.normal));
            throughput *= max(hairWeight, 0.0) / max(hairPdf, 0.000001);
            previousBsdfPdf = max(hairPdf, 0.000001);
            previousScatterDelta = false;
            rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf);
            float3 offsetNormal = dot(ray.direction, hit.geometryNormal) >= 0.0
                ? hit.geometryNormal
                : -hit.geometryNormal;
            ray.origin = offsetRay(hit.position, offsetNormal);

            if (depth >= 2u) {
                float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
                if (rand(seed) > continueProbability) {
                    break;
                }
                throughput /= max(continueProbability, 0.001);
            }
            continue;
        }
#endif

        if (transmission > 0.01) {
            float eta = hit.frontFace ? 1.0 / max(ior, 1.001) : max(ior, 1.001);
            float3 refractedDirection = float3(0.0, 0.0, 0.0);
            bool canRefract = refractDirection(ray.direction, hit.normal, eta, refractedDirection);
            float reflectProbability = canRefract
                ? fresnelDielectric(ior, saturate(dot(-ray.direction, hit.normal)))
                : 1.0;
            if (!canRefract || rand(seed) < reflectProbability) {
                ray.direction = safeNormalize(reflect(ray.direction, hit.normal), hit.normal);
            } else {
                ray.direction = safeNormalize(refractedDirection, ray.direction);
                throughput *= volumeTransmittance(
                    baseColor,
                    transmission,
                    thickness,
                    attenuationDistance,
                    attenuationColor,
                    max(thickness, 0.001)) * transmission;
            }
            previousBsdfPdf = 1.0;
            previousScatterDelta = true;
            ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);
            continue;
        }

        // ------------------------------------------------------------------
        // Radiance caches (RTXGI SHaRC / NVIDIA NRC reference integration).
        // Delta-lobe segments (ideal transmission, hair) skip the caches,
        // mirroring the reference treatment of skipped delta vertices.
        // ------------------------------------------------------------------
#if METALLIC_PT_NRC
        NrcSurfaceAttributes nrcSurface = (NrcSurfaceAttributes)0;
        nrcSurface.encodedPosition = NrcEncodePosition(hit.position, cache.nrc.constants);
        nrcSurface.roughness = roughness;
        nrcSurface.specularF0 = f0;
        nrcSurface.diffuseReflectance = baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission);
        nrcSurface.shadingNormal = hit.normal;
        nrcSurface.viewVector = -ray.direction;
        nrcSurface.isDeltaLobe = transmission > 0.01;
        NrcProgressState nrcProgressState = NrcUpdateOnHit(
            cache.nrc,
            cache.nrcPathState,
            nrcSurface,
            hit.t,
            depth,
            throughput,
            radiance);
        if (nrcProgressState == NrcProgressState::TerminateImmediately) {
            break;
        }
#endif

#if METALLIC_PT_SHARC
        SharcHitData sharcHitData;
        sharcHitData.positionWorld = hit.position;
        sharcHitData.normalWorld = hit.geometryNormal;

#if SHARC_QUERY
        {
            uint gridLevel = HashGridGetLevel(hit.position, cache.sharc.hashGridParameters);
            float voxelSize = HashGridGetVoxelSize(gridLevel, cache.sharc.hashGridParameters);
            bool isValidHit = hit.t > voxelSize * sqrt(3.0);

            float roughnessPrev = min(cache.materialRoughnessPrev, 0.99);
            float alpha = roughnessPrev * roughnessPrev;
            float footprint = hit.t * sqrt(max(0.5 * alpha * alpha / max(1.0 - alpha * alpha, 1e-6), 0.0));
            isValidHit &= footprint > voxelSize;

            float3 sharcRadiance;
            if (isValidHit && SharcGetCachedRadiance(cache.sharc, sharcHitData, sharcRadiance)) {
                radiance += sharcRadiance * throughput;
                break; // Terminate the path once we've looked up into the cache
            }
        }
#endif
#endif

        EnvironmentLightSample lightSample;
        if (sampleEnvironmentLight(push, seed, lightSample)) {
            float3 lightDirection = lightSample.direction;
            float lightMisWeight = environmentLightMisWeight(
                push,
                lightSample.pdfSolidAngle,
                fallbackBsdfPdf(ray.direction, hit.normal, lightDirection, metallic, roughness, diffuseTransmission));
            float3 lightRadianceOverPdf =
                lightSample.radiance * lightMisWeight / max(lightSample.pdfSolidAngle, 0.000001);
            float nDotL = saturate(dot(hit.normal, lightDirection));
            if (nDotL > 0.0) {
                bool occluded = traceShadow(offsetRay(hit.position, hit.geometryNormal), lightDirection, 10000.0, push);
                if (!occluded) {
                    float3 viewDirection = -ray.direction;
                    float3 halfVector = safeNormalize(lightDirection + viewDirection, hit.normal);
                    float nDotH = saturate(dot(hit.normal, halfVector));
                    float specularPower = lerp(96.0, 8.0, roughness);
                    float3 diffuse = baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission) * (nDotL / kPi);
                    float3 specular = f0 * pow(nDotH, specularPower) * nDotL;
                    radiance += throughput * (diffuse + specular) * lightRadianceOverPdf;
                }
            }
            if (diffuseTransmission > 0.01) {
                float backNDotL = saturate(dot(-hit.normal, lightDirection));
                if (backNDotL > 0.0) {
                    float3 transmittedDiffuse =
                        baseColor *
                        diffuseTransmissionColor *
                        diffuseTransmission *
                        volumeTransmittance(
                            baseColor,
                            transmission,
                            thickness,
                            attenuationDistance,
                            attenuationColor,
                            max(thickness, 0.01)) *
                        (backNDotL / kPi);
                    radiance += throughput * transmittedDiffuse * lightRadianceOverPdf;
                }
            }
        }

#if METALLIC_PT_NRC
        // The NRC query was created at this vertex; direct lighting and
        // emissive terms have been added, so the path can terminate here.
        if (nrcProgressState == NrcProgressState::TerminateAfterDirectLighting) {
            break;
        }
#endif

#if METALLIC_PT_SHARC && SHARC_UPDATE
        // Feed the signal accumulated for this segment (emissive + direct
        // lighting) into the cache. Returning false means the path must stop.
        if (!SharcUpdateHit(cache.sharc, cache.sharcState, sharcHitData, radiance, rand(seed))) {
            break;
        }
#endif

        float3 diffuseDirection = sampleCosineHemisphere(hit.normal, seed);
        float3 incidentDirection = ray.direction;
        float3 reflectedDirection = reflect(incidentDirection, hit.normal);
        float specularChance = saturate(0.08 + metallic * 0.72 + (1.0 - roughness) * 0.18);
        bool sampledSpecularLobe = false;
        if (rand(seed) < specularChance) {
            sampledSpecularLobe = true;
            float glossMix = roughness * roughness;
            ray.direction = safeNormalize(lerp(reflectedDirection, diffuseDirection, glossMix), reflectedDirection);
            throughput *= lerp(f0, baseColor, metallic * 0.35);
        } else if (rand(seed) < diffuseTransmission * (1.0 - metallic)) {
            ray.direction = sampleCosineHemisphere(-hit.normal, seed);
            throughput *=
                baseColor *
                diffuseTransmissionColor *
                diffuseTransmission *
                volumeTransmittance(
                    baseColor,
                    transmission,
                    thickness,
                    attenuationDistance,
                    attenuationColor,
                    max(thickness, 0.01));
        } else {
            ray.direction = diffuseDirection;
            throughput *= baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission);
        }
        previousBsdfPdf = max(
            fallbackBsdfPdf(incidentDirection, hit.normal, ray.direction, metallic, roughness, diffuseTransmission),
            0.000001);
        previousScatterDelta = false;
        rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf);
        ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);

#if METALLIC_PT_SHARC
        // Track the effective cone spread of the path so far for the glossy
        // footprint test performed on the next hit.
        cache.materialRoughnessPrev += sampledSpecularLobe ? roughness : 1.0;
#endif
#if METALLIC_PT_NRC
        NrcSetBrdfPdf(cache.nrcPathState, previousBsdfPdf);
#endif
#if METALLIC_PT_SHARC && SHARC_UPDATE
        // Hand the segment throughput to the cache; each segment is
        // accumulated independently afterwards.
        SharcSetThroughput(cache.sharcState, throughput);
#endif

        bool allowRussianRoulette = true;
#if METALLIC_PT_NRC
        allowRussianRoulette = NrcCanUseRussianRoulette(cache.nrcPathState);
#endif
        if (allowRussianRoulette && depth >= 2u) {
            float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
            if (rand(seed) > continueProbability) {
                break;
            }
            throughput /= max(continueProbability, 0.001);
        }
    }

#if METALLIC_PT_NRC
    NrcWriteFinalPathInfo(cache.nrc, cache.nrcPathState, throughput, radiance);
#endif

    return radiance;
}

PathTraceCacheContext makeCacheContext()
{
    PathTraceCacheContext cache;
    ScenePathTraceCacheParams params = gCacheParams[0];
#if METALLIC_PT_SHARC
    cache.sharc.hashGridParameters.cameraPosition = params.sharcCameraPosition.xyz;
    cache.sharc.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    cache.sharc.hashGridParameters.sceneScale = max(params.sharcSceneScale, 0.0001);
    cache.sharc.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

    cache.sharc.hashGridData.capacity = params.sharcEntriesNum;
    cache.sharc.hashGridData.hashEntriesBuffer = gSharcHashEntries;
    cache.sharc.accumulationBuffer = gSharcAccumulation;
    cache.sharc.resolvedBuffer = gSharcResolved;
    cache.sharc.radianceScale = 1.0e3;
    SharcInit(cache.sharcState);
#endif
#if METALLIC_PT_NRC
    NrcBuffers nrcBuffers;
    nrcBuffers.queryPathInfo = gNrcQueryPathInfo;
    nrcBuffers.trainingPathInfo = gNrcTrainingPathInfo;
    nrcBuffers.trainingPathVertices = gNrcTrainingPathVertices;
    nrcBuffers.queryRadianceParams = gNrcQueryRadianceParams;
    nrcBuffers.countersData = gNrcCounters;
    cache.nrc = NrcCreateContext(params.nrc, nrcBuffers, uint2(0, 0));
#endif
    cache.materialRoughnessPrev = 1.0;
    return cache;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                        uniform ScenePathTracePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    uint samples = clamp(push.samples, 1u, 16u);
    uint2 outputPixel = pixel;
    uint seed = 0u;

#if SHARC_UPDATE
    // Sparse update: one random sample per stride x stride block of pixels,
    // with coverage distributed over time through the jitter and frame index.
    ScenePathTraceCacheParams cacheParams = gCacheParams[0];
    uint stride = max(cacheParams.sharcUpdateStride, 1u);
    if (pixel.x >= (push.width + stride - 1u) / stride || pixel.y >= (push.height + stride - 1u) / stride) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 updateJitter = float2(rand(seed), rand(seed));
    float2 samplePixel = (float2(pixel) + updateJitter) * float(stride);
    if (samplePixel.x >= float(push.width) || samplePixel.y >= float(push.height)) {
        return;
    }
    samples = 1u;
#elif NRC_UPDATE
    // NRC training pass: dispatch covers the training resolution; each thread
    // traces one path at a jittered position in the corresponding full-res area.
    ScenePathTraceCacheParams cacheParams = gCacheParams[0];
    if (pixel.x >= cacheParams.trainingWidth || pixel.y >= cacheParams.trainingHeight) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(cacheParams.trainingWidth, 1u) +
        (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 updateJitter = float2(rand(seed), rand(seed));
    float2 samplePixel = (float2(pixel) + updateJitter) *
        float2(push.width, push.height) /
        float2(max(cacheParams.trainingWidth, 1u), max(cacheParams.trainingHeight, 1u));
    samplePixel = min(samplePixel, float2(push.width, push.height) - float2(0.001, 0.001));
    samples = 1u;
#elif NRC_QUERY
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }
    ScenePathTraceCacheParams cacheParams = gCacheParams[0];
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 samplePixel = float2(pixel) + float2(0.5, 0.5);
    samples = 1u;
#else
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.accumulationFrame + 1u) * 0x9e3779b9u);
    float2 samplePixel = float2(pixel);
#endif

    PathTraceCacheContext cache = makeCacheContext();

    float3 color = float3(0.0, 0.0, 0.0);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
#if SHARC_UPDATE || NRC_UPDATE || NRC_QUERY
        float2 jitter = float2(0.0, 0.0);
#else
        float2 jitter = samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed));
#endif
#if !(SHARC_UPDATE || NRC_UPDATE || NRC_QUERY)
        samplePixel = float2(outputPixel) + jitter;
#endif

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
#if METALLIC_PT_NRC
        cache.nrc.pixelIndex = outputPixel;
        NrcSetSampleIndex(cache.nrc, sampleIndex);
        cache.nrcPathState = NrcCreatePathState(cache.nrc.constants, rand(seed));
#endif
        color += pathTrace(ray, rayCone, push, cache, seed);
    }
#if SHARC_UPDATE || NRC_UPDATE
    // Cache update passes produce no image output.
    return;
#endif
    color /= float(samples);

#if NRC_QUERY
    // Linear HDR output: the NRC resolve adds the predicted query radiance on
    // top of this image before the tonemap pass runs.
    if (push.enableAccumulation != 0u) {
        if (push.hasHistory != 0u) {
            float accumulationWeight = 1.0 / float(push.accumulationFrame + 1u);
            float3 previousColor = gHistoryPrevious[pixel].rgb;
            color = lerp(previousColor, color, accumulationWeight);
        }
        gHistoryCurrent[pixel] = float4(color, 1.0);
    } else {
        gHistoryCurrent[pixel] = float4(color, 1.0);
    }
    return;
#endif

    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
    color = saturate(color);

    if (push.enableAccumulation != 0u) {
        if (push.hasHistory != 0u) {
            float accumulationWeight = 1.0 / float(push.accumulationFrame + 1u);
            float3 previousColor = gHistoryPrevious[pixel].rgb;
            color = lerp(previousColor, color, accumulationWeight);
        }
        gHistoryCurrent[pixel] = float4(color, 1.0);
    }

    gOutput[pixel] = float4(color, 1.0);
}
