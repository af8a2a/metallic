#ifndef METALLIC_SCENE_SURFACE
#define METALLIC_SCENE_SURFACE

#include "../../Interop/MaterialTextureDecode.hlsli"
#include "../../Interop/AlphaCoverage.hlsli"
import Core;
using Metallic;
import Material;
using Metallic.Material;
import MaterialProgram;
#if METALLIC_CUSTOM_MATERIALS
#include "MaterialValueDispatch.hlsli"
#endif
import Lighting;
using Metallic.Lighting;

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
#include "../../../External/Rtxtf/STFSamplerState.hlsli"
import NeuralTextures;
using Metallic.Interop.NeuralTextures;

#ifndef SCENE_RAYQUERY_ENABLE_POSITION_FETCH
#define SCENE_RAYQUERY_ENABLE_POSITION_FETCH 0
#endif

import RTXCRHair;


#define gOutput resolveUniform(getResourceParameters<SceneResourceParameters>().output)

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

#define gIndices resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().indices)

#define gPrimitives resolveBuffer<StructuredBuffer<PathTracePrimitive>>(getResourceParameters<SceneResourceParameters>().primitives)

#define gInstances resolveBuffer<StructuredBuffer<PathTraceInstance>>(getResourceParameters<SceneResourceParameters>().instances)

#define gMaterials resolveBuffer<StructuredBuffer<PathTraceMaterial>>(getResourceParameters<SceneResourceParameters>().materials)

#define gHistoryCurrent resolveUniform(getResourceParameters<SceneResourceParameters>().accumulationCurrent)

#define gHistoryPrevious resolveUniform(getResourceParameters<SceneResourceParameters>().accumulationPrevious)

#define gMaterialTextures(index) resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<SceneResourceParameters>().materialTextures.load(index)))

#define gEnvironmentMap resolveUniform(getResourceParameters<SceneResourceParameters>().environmentMap)

#define gEnvironmentImportancePdf resolveUniform(getResourceParameters<SceneResourceParameters>().environmentPdf)


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
    uint outputLinear;      // when set, write linear HDR for the downstream composite
    uint cacheMode;         // METALLIC_CACHE_MODE_*
    uint ntcTextureSetCount;
    uint debugView;
    uint debugFlags;
    float jitterOffsetX;
    float jitterOffsetY;
    uint sampleFrame;
    uint temporalJitter;
    uint deferredSettings; // material shading class
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

// Both ray hits and VBuffer reconstruction already produce HitInfo. Preserve
// the authored TBN; UV1 is not present in this legacy vertex/hit representation.
SurfaceMaterialContext sceneSurfaceMaterialContext(HitInfo hit, TextureFootprint footprint)
{
    SurfaceMaterialContext context = (SurfaceMaterialContext)0;
    context.positionWS = hit.position;
    context.geometryNormalWS = hit.geometryNormal;
    context.shadingNormalWS = hit.normal;
    context.tangentWS = hit.tangent.xyz;
    context.bitangentWS = hit.bitangent;
    context.texCoord0 = hit.texcoord;
    context.texCoordMask = 1u;
    context.frontFace = hit.frontFace ? 1u : 0u;
    context.footprint = footprint;
    return context;
}

PathTraceMaterial loadSceneMaterial(uint materialIndex)
{
    LegacyMaterialProgram program;
    return program.loadParameters(gMaterials, legacyMaterialInstanceRef(materialIndex));
}

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

#include "../../Interop/RTXCRMaterialAdapter.hlsli"

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
    // One internal-render pixel, matching the reference ray-cone diameter.
    // Analytic projection avoids acos precision steps at small pixel angles.
    float2 cone = primaryTextureRayCone(uint2(push.width, push.height), push.viewport.x,
        push.viewport.w, push.clipOrtho.z, push.upProjection.w > 0.5);
    return makeRayCone(cone.x, cone.y);
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

// SDK guide textures use standard depth (near 0, far 1), independently of the
// renderer's reversed-Z attachments and projection.
float cameraHardwareDepth(float linearDepth, ScenePathTracePush push)
{
    float zNear = max(push.clipOrtho.x, 0.0001);
    float zFar = max(push.clipOrtho.y, zNear + 0.0001);
    float viewDepth = clamp(linearDepth, zNear, zFar);
    if (push.upProjection.w > 0.5) {
        return saturate((viewDepth - zNear) / (zFar - zNear));
    }
    return saturate(
        zFar / (zFar - zNear) -
        (zNear * zFar) / ((zFar - zNear) * viewDepth));
}

float2 motionVectorToPrevious(float3 worldPosition, float2 currentPixelCenter, ScenePathTracePush push)
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
    // The hit came from the jittered primary ray. Subtract that same sample
    // position so a stationary camera produces zero motion, as required by
    // Streamline's motionVectorsJittered = false contract.
    float2 samplePixel = currentPixelCenter + float2(push.jitterOffsetX, push.jitterOffsetY);
    float2 currentUv = samplePixel / float2(max(push.width, 1u), max(push.height, 1u));
    return previousUv - currentUv;
}

float2 skyMotionVectorToPrevious(float3 direction, float2 currentPixelCenter, ScenePathTracePush push)
{
    // A sky direction is a point at infinity (w = 0). Project it relative to
    // the previous eye so camera translation cannot create sky parallax.
    if (push.upProjection.w > 0.5) {
        return float2(0.0, 0.0);
    }
    return motionVectorToPrevious(
        push.previousEye.xyz + direction * max(push.previousClipOrtho.y, 1.0),
        currentPixelCenter,
        push);
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
    return Metallic.WorkingColor::fromLinearRec709(kEnvironmentKeyLightRadiance) * max(push.environmentIntensity, 0.0);
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
    // Smooth transmission can produce finite PDFs whose squares overflow.
    // A common power-of-two multiplier cancels from the heuristic. Keep it
    // normal: dividing by a near-FLT_MAX PDF can flush its reciprocal to zero
    // on the GPU. The largest scaled normal PDF is in [1, 4).
    float maxPdf = max(a, b);
    if (maxPdf == 0.0) { return 1.0; }
    uint exponent = (asuint(maxPdf) >> 23u) & 0xffu;
    float scale = asfloat((254u - min(exponent, 253u)) << 23u);
    a *= scale;
    b *= scale;
    a *= a;
    b *= b;
    return a / (a + b);
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
    return proceduralEnvironmentLocalRadiance(environmentLocalDirectionToWorld(direction, -rotationRadians));
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

float3 environmentRadiance(float3 direction, ScenePathTracePush push, bool primaryMiss, float3 primaryWorldOrigin)
{
    if (push.environmentMode == kEnvironmentModeDisabled) {
        return float3(0.0, 0.0, 0.0);
    }
    if (primaryMiss && push.environmentVisible == 0u) {
        return float3(0.0, 0.0, 0.0);
    }
    // The proposal/capture excludes celestial disks. Only the primary visible
    // background queries the physical sky with disks, avoiding direct-light MIS duplication.
    let physicalResources = getResourceParameters<SceneResourceParameters>().environment;
    if (primaryMiss && physicalEnvironmentEnabled(physicalResources)) {
        return physicalVisibleEnvironment(physicalResources, direction, primaryWorldOrigin) * max(push.environmentIntensity, 0.0);
    }

    float3 radiance = push.environmentMode == kEnvironmentModeMap
        ? environmentMapRadiance(direction, push)
        : proceduralSkyRadiance(direction, push.environmentRotationRadians);
    return radiance * max(push.environmentIntensity, 0.0);
}

float3 environmentRadiance(float3 direction, ScenePathTracePush push, bool primaryMiss)
{
    return environmentRadiance(direction, push, primaryMiss, push.eye.xyz);
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
        textureInfo.textureIndex < push.materialTextureCount;
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
    uint mipCount,
    TextureFootprint footprint = (TextureFootprint)0)
{
    float textureDimensionLod = 0.5 * safeLog2(float(max(width, 1u)) * float(max(height, 1u)));
    SurfaceMaterialContext context = sceneSurfaceMaterialContext(hit, (TextureFootprint)0);
    float normalTerm = sqrt(max(abs(dot(rayDirection, context.geometryNormalWS)), 0.000001));
    context.footprint.isotropicLod0 = hit.rayConeLodConstant + safeLog2(max(abs(rayCone.width), 0.000001) / normalTerm);
    context.footprint.flags = kTextureFootprintIsotropic0;
    float normalizedLod = (footprint.flags & kTextureFootprintIsotropic0) != 0u
        ? footprint.isotropicLod0 : context.footprint.isotropicLod0;
    float mipLevel = normalizedLod + textureDimensionLod + textureTransformLodBias(textureInfo);
    return clamp(mipLevel, 0.0, float(max(mipCount, 1u) - 1u));
}

SurfaceMaterialContext sceneSurfaceMaterialContextForRay(HitInfo hit, RayCone rayCone, float3 rayDirection)
{
    SurfaceMaterialContext context = sceneSurfaceMaterialContext(hit, (TextureFootprint)0);
    float normalTerm = sqrt(max(abs(dot(rayDirection, context.geometryNormalWS)), 0.000001));
    context.footprint.isotropicLod0 = hit.rayConeLodConstant + safeLog2(max(abs(rayCone.width), 0.000001) / normalTerm);
    context.footprint.flags = kTextureFootprintIsotropic0;
    return context;
}

import RayMaterialExecution;
RayMaterialWork sceneRayMaterialWork(HitInfo hit, RayCone cone, float3 direction, uint seed, uint bounce)
{
    RayMaterialWork work = (RayMaterialWork)0;
    work.surface = sceneSurfaceMaterialContextForRay(hit, cone, direction);
    work.woWS = safeNormalize(-direction, hit.normal);
    work.instance = legacyMaterialInstanceRef(hit.materialIndex);
    work.seed = seed;
    work.bounce = bounce;
    return work;
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

float materialTextureSamplingFloor(uint textureIndex)
{
#if METALLIC_DEFERRED_LIGHT_GRID
    RWStructuredBuffer<uint> metadata = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().textureFeedback);
    uint count, stride;
    metadata.GetDimensions(count, stride);
    uint offset = textureIndex * 8u;
    // A frozen/full feedback queue disables source-dimension demand, but still
    // supplies the sampling floor associated with this frame's resident tail.
    if (offset + 6u < count) { return max(asfloat(metadata[offset + 6u]), 0.0); }
#endif
    return 0.0;
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
    gMaterialTextures(textureIndex).GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return fallback;
    }

    float2 wrappedUv = frac(transformMaterialUv(textureInfo, uv));
#if METALLIC_REALTIME_DEFERRED
    // Glass explicitly requests deterministic finest-available sampling in
    // both comparison modes, including the streaming publication floor.
    return decodeMaterialTextureSample(sampleResidentMaterialTexture(gMaterialTextures(textureIndex),
        resolveUniform(getResourceParameters<SceneResourceParameters>().materialSampler), wrappedUv, 0.0, materialTextureSamplingFloor(textureIndex)),
        textureInfo.transform0.w);
#else
    uint2 texel = min(uint2(wrappedUv * float2(width, height)), uint2(width - 1u, height - 1u));
    return decodeMaterialTextureSample(gMaterialTextures(textureIndex).Load(int3(texel, 0)), textureInfo.transform0.w);
#endif
}

float loadMaterialAlphaCoverage(PathTraceTextureInfo textureInfo, float2 uv, ScenePathTracePush push)
{
    if (hasNeuralMaterialTexture(textureInfo, push)) {
        return loadMaterialTextureMip0(textureInfo, uv, float4(1.0), push).a;
    }
    if (!hasMaterialTexture(textureInfo, push)) { return 1.0; }
    return sampleAlphaCoverage(gMaterialTextures(textureInfo.textureIndex), transformMaterialUv(textureInfo, uv));
}

float4 loadMaterialTexture(
    PathTraceTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    inout uint seed,
    TextureFootprint footprint = (TextureFootprint)0)
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
            mipCount, footprint);
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
    gMaterialTextures(textureIndex).GetDimensions(0, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        return fallback;
    }

    float mipLevel = rayConeTextureLod(textureInfo, rayCone, hit, rayDirection, width, height, mipCount, footprint);
    float mipFloor = materialTextureSamplingFloor(textureIndex);
#if METALLIC_REALTIME_DEFERRED
    if ((push.debugFlags & 256u) == 0u) {
        // Preserve the subsequent lighting/feedback random dimensions when
        // comparing filtering modes; only the texture estimator changes.
        rand4(seed);
        return decodeMaterialTextureSample(sampleResidentMaterialTexture(gMaterialTextures(textureIndex),
            resolveUniform(getResourceParameters<SceneResourceParameters>().materialSampler), transformMaterialUv(textureInfo, uv), mipLevel, mipFloor),
            textureInfo.transform0.w);
    }
#endif
    STF_SamplerState samplerState = makeMaterialStfSampler(push, seed);
    return decodeMaterialTextureSample(samplerState.Texture2DLoadLevel(
        gMaterialTextures(textureIndex),
        transformMaterialUv(textureInfo, uv),
        max(mipLevel, mipFloor)), textureInfo.transform0.w);
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
    inout uint seed,
    TextureFootprint footprint = (TextureFootprint)0)
{
#if METALLIC_DEFERRED_LIGHT_GRID
    // Demand uses the source dimensions, never the currently coarse resident tail.
    // The same UV transform and ray-cone footprint drive sampling and residency.
    if (hasMaterialTexture(textureInfo, push) && (seed & 7u) == 0u) {
        RWStructuredBuffer<uint> feedback = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().textureFeedback);
        uint count, stride;
        feedback.GetDimensions(count, stride);
        uint offset = textureInfo.textureIndex * 8u;
        if (offset + 7u < count && feedback[offset] != 0u) {
            float wanted = rayConeTextureLod(textureInfo, rayCone, hit, rayDirection,
                feedback[offset], feedback[offset + 1u], feedback[offset + 2u], footprint);
            uint ignored;
            InterlockedMin(feedback[offset + 4u], uint(max(floor(wanted), 0.0)), ignored);
            InterlockedAdd(feedback[offset + 5u], 1u, ignored);
        }
    }
#endif
    if (deterministicMip0) {
        return loadMaterialTextureMip0(textureInfo, uv, fallback, push);
    }
    return loadMaterialTexture(textureInfo, uv, fallback, push, rayCone, hit, rayDirection, seed, footprint);
}

#include "MaterialValueTextures.slang"

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
    return resolveBuffer<StructuredBuffer<SceneShadingVertex>>(getResourceParameters<SceneResourceParameters>().vertices)[i0].texcoord.xy * weights.x +
        resolveBuffer<StructuredBuffer<SceneShadingVertex>>(getResourceParameters<SceneResourceParameters>().vertices)[i1].texcoord.xy * weights.y +
        resolveBuffer<StructuredBuffer<SceneShadingVertex>>(getResourceParameters<SceneResourceParameters>().vertices)[i2].texcoord.xy * weights.z;
}

CoverageResult evaluateSceneCoverage(uint materialIndex, PathTraceMaterial material, float2 uv, ScenePathTracePush push)
{
    CoverageContext context;
    context.uv = uv; context.baseAlpha = 1.0;
    context.cutoff = material.params.z; context.alphaMode = material.textureParams.w;
    if (context.alphaMode < 0.5) { return coverageFromOpacity(context, 1.0); }
    if (material.attenuationColor.w > 0.5) {
        CoverageProgram program;
        program.inputs = resolveUniform(getResourceParameters<SceneResourceParameters>().materialValues);
        MaterialInstanceRef instance = coverageInstanceRef(materialIndex);
        if (program.usesBaseAlpha(instance)) {
            context.baseAlpha = material.baseColor.a * loadMaterialAlphaCoverage(material.baseColorTexture, uv, push);
        }
        return program.evaluateCoverage(context, instance);
    }
    context.baseAlpha = material.baseColor.a * loadMaterialAlphaCoverage(material.baseColorTexture, uv, push);
    LegacyCoverageEvaluator legacy;
    return legacy.evaluateCoverage(context, coverageInstanceRef(materialIndex));
}

bool alphaMaskAcceptsHit(uint instanceIndex, uint triangleIndex, float2 bary, ScenePathTracePush push)
{
    PathTraceInstance instance = gInstances[instanceIndex];
    PathTraceMaterial material = loadSceneMaterial(instance.materialIndex);
    bool blend = abs(material.textureParams.w - 2.0) < 0.25;
    if (!blend && abs(material.textureParams.w - 1.0) > 0.25) {
        return true;
    }

    float2 uv = triangleTexcoord(instanceIndex, triangleIndex, bary);
    return evaluateSceneCoverage(instance.materialIndex, material, uv, push).accepted;
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


#endif
