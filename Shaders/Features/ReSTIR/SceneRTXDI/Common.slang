#language slang 2026
implementing SceneRTXDI;

import Core;
using Metallic;

struct RTXDIPrimitive
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

struct RTXDIInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint flags;
    float padding;
};

static const uint kInvalidMaterialTextureIndex = 0xffffffffu;
static const uint kPrimitiveHasAuthoredTangents = 1u;
static const uint kVisualizationShaded = 0u;
static const uint kVisualizationLightId = 1u;
static const uint kVisualizationHistory = 2u;
static const uint kVisualizationReGIRCells = 3u;
static const uint kEnvironmentSampleBit = 0x80000000u;
static const uint kEnvironmentSampleMask = 0x7fffffffu;
static const uint kBehaviorTemporalReuse = 1u << 0u;
static const uint kBehaviorSpatialReuse = 1u << 1u;
static const uint kBehaviorAnimateLights = 1u << 2u;
static const uint kBehaviorInitialVisibility = 1u << 3u;
static const uint kBehaviorLocalLightImportance = 1u << 4u;
static const uint kBehaviorEnvironmentEnabled = 1u << 5u;
static const uint kBehaviorEnvironmentVisible = 1u << 6u;
static const uint kBehaviorEnvironmentImportance = 1u << 7u;
static const uint kBehaviorVisualizationShift = 8u;
static const uint kBehaviorVisualizationMask = 3u;
static const uint kBehaviorReGIR = 1u << 10u;
static const float kPi = 3.14159265358979323846;

struct SceneRTXDIPush
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
    float4 sceneCenterRadius;
    uint width;
    uint height;
    uint frameIndex;
    uint hasHistory;
    uint lightCount;
    uint initialSampleCount;
    uint spatialSampleCount;
    uint maxHistoryLength;
    uint behaviorFlags;
    float environmentIntensity;
    float environmentRotationRadians;
    uint environmentSampleCount;
    uint materialTextureCount;
    float bitangentFlip;
    float lightIntensity;
    float exposure;
    float normalThreshold;
    float depthThreshold;
    uint outputLinear;
};

bool behaviorEnabled(SceneRTXDIPush push, uint flag)
{
    return (push.behaviorFlags & flag) != 0u;
}

uint visualizationMode(SceneRTXDIPush push)
{
    return (push.behaviorFlags >> kBehaviorVisualizationShift) & kBehaviorVisualizationMask;
}


struct Surface
{
    float3 position;
    float3 normal;
    float3 geometryNormal;
    float3 viewDirection;
    float3 baseColor;
    float3 diffuseAlbedo;
    float3 specularF0;
    float3 emissive;
    float metalness;
    float roughness;
};

struct LightEvaluation
{
    float3 contribution;
    float3 diffuseRadiance;
    float3 specularRadiance;
    float3 direction;
    float distance;
    float target;
};

struct DIReservoir
{
    uint lightIndex;
    float weightSum;
    float selectedTarget;
    uint M;
};

LightEvaluation emptyLightEvaluation()
{
    LightEvaluation evaluation;
    evaluation.contribution = float3(0.0, 0.0, 0.0);
    evaluation.diffuseRadiance = float3(0.0, 0.0, 0.0);
    evaluation.specularRadiance = float3(0.0, 0.0, 0.0);
    evaluation.direction = float3(0.0, 1.0, 0.0);
    evaluation.distance = 0.0;
    evaluation.target = 0.0;
    return evaluation;
}

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
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

float hashToFloat(uint value)
{
    return float(hash(value) & 0x00ffffffu) / 16777216.0;
}

float randomFloat(inout uint state)
{
    state = hash(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float luminance(float3 color)
{
    return Metallic.WorkingColor::luminance(color);
}

float3 srgbToLinear(float3 color)
{
    return float3(
        color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4));
}

float3 hashColor(uint value)
{
    float3 color = float3(
        hashToFloat(value * 3u + 0u),
        hashToFloat(value * 3u + 1u),
        hashToFloat(value * 3u + 2u));
    return 0.2 + color * 0.8;
}
