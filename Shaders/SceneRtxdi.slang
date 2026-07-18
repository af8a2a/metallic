// ReSTIR DI implementation for Metallic. The integration follows the fused
// spatiotemporal flow demonstrated by RTXDI while using Metallic-native data.

#define NRD_HEADER_ONLY
#include "NRD.hlsli"

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure gScene;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gOutput;

struct RtxdiVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 texcoord;
};

struct RtxdiPrimitive
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

struct RtxdiInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint flags;
    float padding;
};

struct RtxdiTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint padding0;
    uint padding1;
    float4 transform0;
    float4 transform1;
};

struct RtxdiMaterial
{
    float4 baseColor;
    float4 emissive;
    float4 params;
    float4 textureParams;
    float4 glassParams;
    float4 attenuationColor;
    float4 diffuseTransmission;
    RtxdiTextureInfo baseColorTexture;
    RtxdiTextureInfo metallicRoughnessTexture;
    RtxdiTextureInfo normalTexture;
    RtxdiTextureInfo occlusionTexture;
    RtxdiTextureInfo emissiveTexture;
    RtxdiTextureInfo transmissionTexture;
    RtxdiTextureInfo thicknessTexture;
    RtxdiTextureInfo diffuseTransmissionTexture;
    RtxdiTextureInfo diffuseTransmissionColorTexture;
};

[[vk::binding(2, 0)]]
StructuredBuffer<RtxdiVertex> gVertices;

[[vk::binding(3, 0)]]
StructuredBuffer<uint> gIndices;

[[vk::binding(4, 0)]]
StructuredBuffer<RtxdiPrimitive> gPrimitives;

[[vk::binding(5, 0)]]
StructuredBuffer<RtxdiInstance> gInstances;

[[vk::binding(6, 0)]]
StructuredBuffer<RtxdiMaterial> gMaterials;

[[vk::binding(7, 0)]]
Texture2D<float4> gMaterialTextures[256];

[[vk::binding(8, 0)]]
RWTexture2D<uint4> gReservoirCurrent;

[[vk::binding(9, 0)]]
RWTexture2D<uint4> gReservoirPrevious;

[[vk::binding(10, 0)]]
RWTexture2D<float4> gPositionCurrent;

[[vk::binding(11, 0)]]
RWTexture2D<float4> gPositionPrevious;

[[vk::binding(12, 0)]]
RWTexture2D<float4> gNormalCurrent;

[[vk::binding(13, 0)]]
RWTexture2D<float4> gNormalPrevious;

[[vk::binding(14, 0)]]
RWTexture2D<float4> gNoisyDiffuse;

[[vk::binding(15, 0)]]
RWTexture2D<float4> gNoisySpecular;

[[vk::binding(16, 0)]]
RWTexture2D<float4> gNrdNormalRoughness;

[[vk::binding(17, 0)]]
RWTexture2D<float4> gNrdMotionVectors;

[[vk::binding(18, 0)]]
RWTexture2D<float> gNrdViewZ;

[[vk::binding(19, 0)]]
RWTexture2D<float4> gBaseColorMetalness;

[[vk::binding(20, 0)]]
RWTexture2D<float4> gEmissive;

[[vk::binding(21, 0)]]
Texture2D<float4> gEnvironmentMap;

[[vk::binding(22, 0)]]
Texture2D<float> gLocalLightPdf;

[[vk::binding(23, 0)]]
Texture2D<float> gEnvironmentPdf;

[[vk::binding(24, 0)]]
StructuredBuffer<uint4> gReGIRData;

static const uint kMaxMaterialTextures = 256u;
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
static const uint kReGIRHeaderRecordCount = 3u;
static const uint kReGIRMagic = 0x52454749u;
static const float kPi = 3.14159265358979323846;
static const float kGoldenAngle = 2.39996322972865332;

struct SceneRtxdiPush
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
    uint padding1;
    uint padding2;
};

bool behaviorEnabled(SceneRtxdiPush push, uint flag)
{
    return (push.behaviorFlags & flag) != 0u;
}

uint visualizationMode(SceneRtxdiPush push)
{
    return (push.behaviorFlags >> kBehaviorVisualizationShift) & kBehaviorVisualizationMask;
}

struct Ray
{
    float3 origin;
    float3 direction;
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
};

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

struct AnalyticLight
{
    float3 position;
    float3 radiance;
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
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

uint zCurveToLinearIndex(uint2 position)
{
    uint index = 0u;
    for (uint bit = 0u; bit < 16u; ++bit) {
        index |= ((position.x >> bit) & 1u) << (bit * 2u);
        index |= ((position.y >> bit) & 1u) << (bit * 2u + 1u);
    }
    return index;
}

float loadImportanceWeight(Texture2D<float> pdfTexture, uint2 position, uint mipLevel)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    pdfTexture.GetDimensions(mipLevel, width, height, mipCount);
    if (position.x >= width || position.y >= height) {
        return 0.0;
    }
    return max(pdfTexture.Load(int3(position, mipLevel)), 0.0);
}

bool sampleImportanceMipmap(
    Texture2D<float> pdfTexture,
    inout uint seed,
    out uint2 position,
    out float pdf)
{
    position = uint2(0u, 0u);
    pdf = 1.0;
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    pdfTexture.GetDimensions(0u, width, height, mipCount);
    uint2 pdfTextureSize = uint2(width, height);
    if (pdfTextureSize.x == 0u || pdfTextureSize.y == 0u || mipCount == 0u) {
        pdf = 0.0;
        return false;
    }
    if (mipCount == 1u) {
        return loadImportanceWeight(pdfTexture, position, 0u) > 0.0;
    }

    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
        float4 weights = float4(
            loadImportanceWeight(pdfTexture, position + uint2(0u, 0u), uint(mipLevel)),
            loadImportanceWeight(pdfTexture, position + uint2(0u, 1u), uint(mipLevel)),
            loadImportanceWeight(pdfTexture, position + uint2(1u, 0u), uint(mipLevel)),
            loadImportanceWeight(pdfTexture, position + uint2(1u, 1u), uint(mipLevel)));
        float weightSum = weights.x + weights.y + weights.z + weights.w;
        if (weightSum <= 0.0) {
            pdf = 0.0;
            return false;
        }
        weights /= weightSum;
        float randomValue = randomFloat(seed);
        if (randomValue < weights.x) {
            pdf *= weights.x;
        } else if (randomValue < weights.x + weights.y) {
            position += uint2(0u, 1u);
            pdf *= weights.y;
        } else if (randomValue < weights.x + weights.y + weights.z) {
            position += uint2(1u, 0u);
            pdf *= weights.z;
        } else {
            position += uint2(1u, 1u);
            pdf *= weights.w;
        }
    }
    return pdf > 0.0 && all(position < pdfTextureSize);
}

struct ReGIRSelectionContext
{
    uint lightSlotBase;
    uint lightsPerCell;
    uint lightSlotCount;
    uint valid;
};

ReGIRSelectionContext initializeReGIRSelectionContext(
    SceneRtxdiPush push,
    Surface surface,
    inout uint seed)
{
    ReGIRSelectionContext context;
    context.lightSlotBase = 0u;
    context.lightsPerCell = 0u;
    context.lightSlotCount = 0u;
    context.valid = 0u;
    if (!behaviorEnabled(push, kBehaviorReGIR)) {
        return context;
    }

    uint4 grid = gReGIRData[0];
    uint4 centerAndCellSize = gReGIRData[1];
    uint4 metadata = gReGIRData[2];
    if (metadata.x != kReGIRMagic ||
        any(grid.xyz == uint3(0u, 0u, 0u)) ||
        grid.w == 0u ||
        metadata.y == 0u) {
        return context;
    }

    float3 gridCenter = asfloat(centerAndCellSize.xyz);
    float cellSize = asfloat(centerAndCellSize.w);
    float samplingJitter = max(asfloat(metadata.z), 0.0);
    if (!(cellSize > 0.0) || !isfinite(cellSize) || !isfinite(samplingJitter)) {
        return context;
    }

    float3 cellJitter = float3(
        randomFloat(seed),
        randomFloat(seed),
        randomFloat(seed));
    cellJitter = (cellJitter - 0.5) * (samplingJitter * cellSize * 2.0);
    float3 samplingPosition = surface.position + cellJitter;
    float3 gridOrigin = gridCenter - float3(grid.xyz) * (cellSize * 0.5);
    int3 cellPosition = int3(floor((samplingPosition - gridOrigin) / cellSize));
    if (any(cellPosition < int3(0, 0, 0)) || any(cellPosition >= int3(grid.xyz))) {
        return context;
    }

    uint cellIndex = uint(cellPosition.x) +
        (uint(cellPosition.y) + uint(cellPosition.z) * grid.y) * grid.x;
    uint lightSlotBase = cellIndex * grid.w;
    if (lightSlotBase >= metadata.y || grid.w > metadata.y - lightSlotBase) {
        return context;
    }

    context.lightSlotBase = lightSlotBase;
    context.lightsPerCell = grid.w;
    context.lightSlotCount = metadata.y;
    context.valid = 1u;
    return context;
}

bool sampleLocalLightFromReGIR(
    SceneRtxdiPush push,
    ReGIRSelectionContext context,
    inout uint seed,
    out uint lightIndex,
    out float sourcePdf)
{
    lightIndex = 0u;
    sourcePdf = 0.0;
    if (context.valid == 0u || context.lightsPerCell == 0u) {
        return false;
    }

    uint lightInCell = min(
        uint(randomFloat(seed) * float(context.lightsPerCell)),
        context.lightsPerCell - 1u);
    uint lightSlot = context.lightSlotBase + lightInCell;
    if (lightSlot >= context.lightSlotCount) {
        return false;
    }
    uint4 lightData = gReGIRData[kReGIRHeaderRecordCount + lightSlot];
    float inverseSourcePdf = asfloat(lightData.y);
    if (lightData.x >= push.lightCount ||
        !(inverseSourcePdf > 0.0) ||
        !isfinite(inverseSourcePdf)) {
        return false;
    }

    lightIndex = lightData.x;
    sourcePdf = 1.0 / inverseSourcePdf;
    return sourcePdf > 0.0 && isfinite(sourcePdf);
}

float3 visualizeReGIRCell(float3 worldPosition)
{
    uint4 grid = gReGIRData[0];
    uint4 centerAndCellSize = gReGIRData[1];
    uint4 metadata = gReGIRData[2];
    float cellSize = asfloat(centerAndCellSize.w);
    if (metadata.x != kReGIRMagic ||
        any(grid.xyz == uint3(0u, 0u, 0u)) ||
        !(cellSize > 0.0)) {
        return float3(0.0, 0.0, 0.0);
    }

    float3 gridCenter = asfloat(centerAndCellSize.xyz);
    float3 gridOrigin = gridCenter - float3(grid.xyz) * (cellSize * 0.5);
    float3 cellCoordinates = (worldPosition - gridOrigin) / cellSize;
    int3 cellPosition = int3(floor(cellCoordinates));
    if (any(cellPosition < int3(0, 0, 0)) || any(cellPosition >= int3(grid.xyz))) {
        return float3(0.0, 0.0, 0.0);
    }

    uint cellIndex = uint(cellPosition.x) +
        (uint(cellPosition.y) + uint(cellPosition.z) * grid.y) * grid.x;
    float3 cellColor = hashColor(cellIndex + 0x7f4a7c15u);
    float3 centerDistance = abs(frac(cellCoordinates) - 0.5) * 2.0;
    return cellColor * lerp(0.35, 1.0, max(centerDistance.x, max(centerDistance.y, centerDistance.z)));
}

bool sampleLocalLight(
    SceneRtxdiPush push,
    ReGIRSelectionContext reGIRContext,
    inout uint seed,
    out uint lightIndex,
    out float sourcePdf)
{
    lightIndex = 0u;
    sourcePdf = 0.0;
    if (push.lightCount == 0u) {
        return false;
    }
    if (sampleLocalLightFromReGIR(push, reGIRContext, seed, lightIndex, sourcePdf)) {
        return true;
    }
    if (!behaviorEnabled(push, kBehaviorLocalLightImportance)) {
        lightIndex = min(uint(randomFloat(seed) * float(push.lightCount)), push.lightCount - 1u);
        sourcePdf = 1.0 / float(push.lightCount);
        return true;
    }

    uint2 position;
    if (sampleImportanceMipmap(
            gLocalLightPdf,
            seed,
            position,
            sourcePdf)) {
        lightIndex = zCurveToLinearIndex(position);
        if (lightIndex < push.lightCount && sourcePdf > 0.0) {
            return true;
        }
    }

    lightIndex = min(uint(randomFloat(seed) * float(push.lightCount)), push.lightCount - 1u);
    sourcePdf = 1.0 / float(push.lightCount);
    return true;
}

bool sampleEnvironmentTexel(
    SceneRtxdiPush push,
    inout uint seed,
    out uint2 texel,
    out float texelPdf)
{
    texel = uint2(0u, 0u);
    texelPdf = 0.0;
    uint width = 0u;
    uint height = 0u;
    gEnvironmentMap.GetDimensions(width, height);
    if (!behaviorEnabled(push, kBehaviorEnvironmentEnabled) || width == 0u || height == 0u) {
        return false;
    }

    uint environmentTexelCount = width * height;
    if (!behaviorEnabled(push, kBehaviorEnvironmentImportance)) {
        uint texelIndex = min(
            uint(randomFloat(seed) * float(max(environmentTexelCount, 1u))),
            max(environmentTexelCount, 1u) - 1u);
        texel = uint2(texelIndex % width, texelIndex / width);
        texelPdf = 1.0 / float(max(environmentTexelCount, 1u));
        return texel.y < height;
    }

    if (sampleImportanceMipmap(
            gEnvironmentPdf,
            seed,
            texel,
            texelPdf) &&
        texel.x < width &&
        texel.y < height &&
        texelPdf > 0.0) {
        return true;
    }

    uint texelIndex = min(
        uint(randomFloat(seed) * float(max(environmentTexelCount, 1u))),
        max(environmentTexelCount, 1u) - 1u);
    texel = uint2(texelIndex % width, texelIndex / width);
    texelPdf = 1.0 / float(max(environmentTexelCount, 1u));
    return texel.y < height;
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

float3 rayDirectionFromPixel(float2 pixel, SceneRtxdiPush push)
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
    float halfHeight = tan(clamp(push.viewport.w, 0.017453292, 3.12413936) * 0.5);
    return safeNormalize(
        forward + right * (ndc.x * aspect * halfHeight) + up * (ndc.y * halfHeight),
        forward);
}

float3 rayOriginFromPixel(float2 pixel, SceneRtxdiPush push)
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

bool projectWorldToPreviousUv(float3 worldPosition, SceneRtxdiPush push, out float2 uv)
{
    float3 forward = safeNormalize(
        push.previousCenter.xyz - push.previousEye.xyz,
        float3(0.0, 0.0, -1.0));
    float3 right = safeNormalize(
        cross(forward, safeNormalize(push.previousUpProjection.xyz, float3(0.0, 1.0, 0.0))),
        float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 view = worldPosition - push.previousEye.xyz;
    float aspect = max(push.previousViewport.x, 0.001);
    float2 ndc;
    if (push.previousUpProjection.w > 0.5) {
        float halfHeight = max(push.previousClipOrtho.z * 0.5, 0.0001);
        ndc = float2(
            dot(view, right) / max(halfHeight * aspect, 0.0001),
            dot(view, up) / halfHeight);
    } else {
        float viewZ = dot(view, forward);
        if (viewZ <= max(push.previousClipOrtho.x, 0.0001)) {
            uv = 0.0;
            return false;
        }
        float halfHeight = max(tan(push.previousViewport.w * 0.5), 0.0001);
        ndc = float2(
            dot(view, right) / max(viewZ * aspect * halfHeight, 0.0001),
            dot(view, up) / max(viewZ * halfHeight, 0.0001));
    }
    uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return all(uv >= 0.0) && all(uv < 1.0);
}

float cameraLinearDepth(float3 worldPosition, SceneRtxdiPush push)
{
    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    return max(dot(worldPosition - push.eye.xyz, forward), 0.0);
}

float2 motionVectorToPrevious(float3 worldPosition, float2 currentSamplePixel, SceneRtxdiPush push)
{
    if (push.hasHistory == 0u) {
        return float2(0.0, 0.0);
    }
    float2 previousUv;
    if (!projectWorldToPreviousUv(worldPosition, push, previousUv)) {
        return float2(0.0, 0.0);
    }
    float2 currentUv = currentSamplePixel / float2(max(push.width, 1u), max(push.height, 1u));
    return previousUv - currentUv;
}

bool hasMaterialTexture(RtxdiTextureInfo textureInfo, SceneRtxdiPush push)
{
    return textureInfo.textureIndex != kInvalidMaterialTextureIndex &&
        textureInfo.textureIndex < push.materialTextureCount &&
        textureInfo.textureIndex < kMaxMaterialTextures;
}

float2 transformMaterialUv(RtxdiTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

float4 loadMaterialTexture(
    RtxdiTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    SceneRtxdiPush push)
{
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

float2 triangleTexcoord(uint instanceIndex, uint triangleIndex, float2 bary)
{
    RtxdiInstance instance = gInstances[instanceIndex];
    RtxdiPrimitive primitive = gPrimitives[instance.primitiveIndex];
    uint indexOffset = primitive.firstIndex + triangleIndex * 3u;
    uint i0 = primitive.firstVertex + gIndices[indexOffset + 0u];
    uint i1 = primitive.firstVertex + gIndices[indexOffset + 1u];
    uint i2 = primitive.firstVertex + gIndices[indexOffset + 2u];
    float3 weights = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    return gVertices[i0].texcoord.xy * weights.x +
        gVertices[i1].texcoord.xy * weights.y +
        gVertices[i2].texcoord.xy * weights.z;
}

bool alphaMaskAcceptsHit(uint instanceIndex, uint triangleIndex, float2 bary, SceneRtxdiPush push)
{
    RtxdiInstance instance = gInstances[instanceIndex];
    RtxdiMaterial material = gMaterials[instance.materialIndex];
    if (abs(material.textureParams.w - 1.0) > 0.25) {
        return true;
    }
    float alpha = material.baseColor.a * loadMaterialTexture(
        material.baseColorTexture,
        triangleTexcoord(instanceIndex, triangleIndex, bary),
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
        float3 referenceAxis = abs(normal.z) < 0.999
            ? float3(0.0, 0.0, 1.0)
            : float3(0.0, 1.0, 0.0);
        tangent = safeNormalize(cross(referenceAxis, normal), float3(1.0, 0.0, 0.0));
        bitangent = safeNormalize(cross(normal, tangent), float3(0.0, 1.0, 0.0));
        return;
    }
    float inverseDeterminant = 1.0 / determinant;
    tangent = (edge1 * deltaUv2.y - edge2 * deltaUv1.y) * inverseDeterminant;
    bitangent = (edge2 * deltaUv1.x - edge1 * deltaUv2.x) * inverseDeterminant;
}

float3 transformObjectNormalToWorld(float3 normal, float3x4 worldToObject)
{
    return mul(normal, worldToObject).xyz;
}

bool traceClosest(Ray ray, SceneRtxdiPush push, out HitInfo hit)
{
    hit = HitInfo();
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
    RtxdiInstance instance = gInstances[instanceIndex];
    RtxdiPrimitive primitive = gPrimitives[instance.primitiveIndex];
    uint triangleIndex = query.CommittedPrimitiveIndex();
    uint indexOffset = primitive.firstIndex + triangleIndex * 3u;
    uint i0 = primitive.firstVertex + gIndices[indexOffset + 0u];
    uint i1 = primitive.firstVertex + gIndices[indexOffset + 1u];
    uint i2 = primitive.firstVertex + gIndices[indexOffset + 2u];
    RtxdiVertex v0 = gVertices[i0];
    RtxdiVertex v1 = gVertices[i1];
    RtxdiVertex v2 = gVertices[i2];
    float2 bary = query.CommittedTriangleBarycentrics();
    float3 weights = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    float3 p0 = v0.position.xyz;
    float3 p1 = v1.position.xyz;
    float3 p2 = v2.position.xyz;
    float2 uv0 = v0.texcoord.xy;
    float2 uv1 = v1.texcoord.xy;
    float2 uv2 = v2.texcoord.xy;
    float3 localPosition = p0 * weights.x + p1 * weights.y + p2 * weights.z;
    float3 localGeometryNormal = safeNormalize(cross(p1 - p0, p2 - p0), float3(0.0, 1.0, 0.0));
    float3 localNormal = v0.normal.xyz * weights.x + v1.normal.xyz * weights.y + v2.normal.xyz * weights.z;
    localNormal = safeNormalize(localNormal, localGeometryNormal);
    float4 localTangent = v0.tangent * weights.x + v1.tangent * weights.y + v2.tangent * weights.z;
    float3 localTangentDirection = safeNormalize(localTangent.xyz, float3(1.0, 0.0, 0.0));
    float tangentSign = localTangent.w >= 0.0 ? 1.0 : -1.0;
    hit.texcoord = uv0 * weights.x + uv1 * weights.y + uv2 * weights.z;

    float3 localTriangleTangent;
    float3 localTriangleBitangent;
    computeTriangleTangentSpace(
        p0,
        p1,
        p2,
        uv0,
        uv1,
        uv2,
        localNormal,
        localTriangleTangent,
        localTriangleBitangent);

    float4x3 objectToWorld = query.CommittedObjectToWorld4x3();
    float3x4 worldToObject = query.CommittedWorldToObject3x4();
    hit.position = float3(mul(float4(localPosition, 1.0), objectToWorld));
    hit.geometryNormal = safeNormalize(
        transformObjectNormalToWorld(localGeometryNormal, worldToObject),
        localGeometryNormal);
    hit.normal = safeNormalize(transformObjectNormalToWorld(localNormal, worldToObject), hit.geometryNormal);

    float3 referenceAxis = abs(hit.normal.z) < 0.999
        ? float3(0.0, 0.0, 1.0)
        : float3(0.0, 1.0, 0.0);
    float3 fallbackTangent = safeNormalize(cross(referenceAxis, hit.normal), float3(1.0, 0.0, 0.0));
    if ((primitive.flags & kPrimitiveHasAuthoredTangents) != 0u) {
        float3 worldTangent = safeNormalize(mul(localTangentDirection, worldToObject).xyz, fallbackTangent);
        worldTangent = safeNormalize(worldTangent - hit.normal * dot(hit.normal, worldTangent), fallbackTangent);
        hit.tangent = float4(worldTangent, tangentSign);
        hit.bitangent = safeNormalize(
            cross(hit.normal, worldTangent) * tangentSign,
            cross(hit.normal, fallbackTangent));
    } else {
        float3 worldTangent = safeNormalize(mul(localTriangleTangent, worldToObject).xyz, fallbackTangent);
        worldTangent = safeNormalize(worldTangent - hit.normal * dot(hit.normal, worldTangent), fallbackTangent);
        float3 worldBitangent = safeNormalize(
            mul(localTriangleBitangent, worldToObject).xyz,
            cross(hit.normal, worldTangent));
        worldBitangent = safeNormalize(
            worldBitangent - hit.normal * dot(hit.normal, worldBitangent),
            cross(hit.normal, worldTangent));
        tangentSign = dot(cross(hit.normal, worldTangent), worldBitangent) < 0.0 ? -1.0 : 1.0;
        hit.tangent = float4(worldTangent, tangentSign);
        hit.bitangent = worldBitangent;
    }
    hit.tangent.w *= push.bitangentFlip;
    hit.bitangent *= push.bitangentFlip;
    hit.t = query.CommittedRayT();
    hit.materialIndex = instance.materialIndex;
    return true;
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
    float3 referenceAxis = abs(normal.z) < 0.999
        ? float3(0.0, 0.0, 1.0)
        : float3(0.0, 1.0, 0.0);
    float3 fallbackTangent = safeNormalize(cross(referenceAxis, normal), float3(1.0, 0.0, 0.0));
    float3 tangentDirection = safeNormalize(
        tangent.xyz - normal * dot(normal, tangent.xyz),
        fallbackTangent);
    float tangentSign = tangent.w >= 0.0 ? 1.0 : -1.0;
    float3 bitangentDirection = safeNormalize(
        bitangent - normal * dot(normal, bitangent),
        cross(normal, tangentDirection) * tangentSign);
    float3 tangentNormal = float3(normalSample.xy * 2.0 - 1.0, normalSample.z * 2.0 - 1.0);
    tangentNormal.xy *= max(normalScale, 0.0);
    tangentNormal = safeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));
    return safeNormalize(
        tangentDirection * tangentNormal.x +
        bitangentDirection * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
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

bool traceShadow(float3 origin, float3 direction, float maxDistance, SceneRtxdiPush push)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = max(maxDistance - 0.002, 0.001);
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

Surface makeSurface(HitInfo hit, Ray ray, SceneRtxdiPush push)
{
    RtxdiMaterial material = gMaterials[hit.materialIndex];
    float4 baseColorSample = loadMaterialTexture(
        material.baseColorTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push);
    float4 metallicRoughnessSample = loadMaterialTexture(
        material.metallicRoughnessTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push);
    float4 normalSample = loadMaterialTexture(
        material.normalTexture,
        hit.texcoord,
        float4(0.5, 0.5, 1.0, 1.0),
        push);
    float4 emissiveSample = loadMaterialTexture(
        material.emissiveTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push);

    Surface surface;
    surface.position = hit.position;
    surface.geometryNormal = hit.geometryNormal;
    surface.normal = applyNormalMap(
        hit.normal,
        hit.geometryNormal,
        hit.tangent,
        hit.bitangent,
        normalSample,
        material.textureParams.x);
    surface.normal = faceForwardShadingNormal(surface.normal, surface.geometryNormal, ray.direction);
    surface.viewDirection = -ray.direction;
    surface.baseColor = saturate(material.baseColor.rgb * srgbToLinear(baseColorSample.rgb));
    surface.metalness = saturate(material.params.x * metallicRoughnessSample.b);
    surface.roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.03, 1.0);
    surface.diffuseAlbedo = surface.baseColor * (1.0 - surface.metalness);
    surface.specularF0 = lerp(float3(0.04, 0.04, 0.04), surface.baseColor, surface.metalness);
    surface.emissive = max(material.emissive.rgb * srgbToLinear(emissiveSample.rgb), 0.0);
    return surface;
}

AnalyticLight getAnalyticLight(uint lightIndex, SceneRtxdiPush push)
{
    float radius = max(push.sceneCenterRadius.w, 0.01);
    uint ringCount = 16u;
    uint ringIndex = lightIndex % ringCount;
    uint layerIndex = lightIndex / ringCount;
    uint layerCount = max((push.lightCount + ringCount - 1u) / ringCount, 1u);
    float layer = (float(layerIndex) + 0.5) / float(layerCount);
    float animation = behaviorEnabled(push, kBehaviorAnimateLights)
        ? float(push.frameIndex) * 0.0125
        : 0.0;
    float angle = float(lightIndex) * kGoldenAngle + animation * (0.55 + hashToFloat(lightIndex + 19u));
    float radialJitter = 0.55 + hashToFloat(lightIndex * 11u + 7u) * 0.5;
    float radialDistance = radius * radialJitter;
    float vertical = radius * lerp(-0.2, 1.4, layer);
    float wobble = behaviorEnabled(push, kBehaviorAnimateLights)
        ? sin(animation * (0.7 + hashToFloat(lightIndex + 31u)) + float(ringIndex)) * radius * 0.08
        : 0.0;

    AnalyticLight light;
    light.position = push.sceneCenterRadius.xyz + float3(
        cos(angle) * radialDistance,
        vertical + wobble,
        sin(angle) * radialDistance);
    float3 color = hashColor(lightIndex + 1u);
    color *= 0.65 + hashToFloat(lightIndex * 17u + 3u) * 0.7;
    // Keep total emitted power stable while the demo varies the light count.
    light.radiance = color * push.lightIntensity * radius * radius /
        float(max(push.lightCount, 1u));
    return light;
}

LightEvaluation evaluateLight(uint lightIndex, Surface surface, SceneRtxdiPush push)
{
    AnalyticLight light = getAnalyticLight(lightIndex, push);
    float3 toLight = light.position - surface.position;
    float distanceSquared = max(dot(toLight, toLight), 0.000001);
    float distance = sqrt(distanceSquared);
    float3 lightDirection = toLight / distance;
    float nDotL = saturate(dot(surface.normal, lightDirection));
    float3 halfVector = safeNormalize(lightDirection + surface.viewDirection, surface.normal);
    float nDotH = saturate(dot(surface.normal, halfVector));
    float specularPower = lerp(128.0, 4.0, surface.roughness);
    float3 incidentRadiance = light.radiance / distanceSquared;

    LightEvaluation evaluation;
    evaluation.diffuseRadiance = incidentRadiance * (nDotL / kPi);
    evaluation.specularRadiance = incidentRadiance * nDotL *
        ((specularPower + 2.0) / (2.0 * kPi)) *
        pow(nDotH, specularPower);
    evaluation.contribution = evaluation.diffuseRadiance * surface.diffuseAlbedo +
        evaluation.specularRadiance * surface.specularF0;
    evaluation.direction = lightDirection;
    evaluation.distance = distance;
    evaluation.target = max(luminance(evaluation.contribution), 0.0);
    return evaluation;
}

bool isEnvironmentSample(uint lightIndex)
{
    return (lightIndex & kEnvironmentSampleBit) != 0u;
}

uint environmentTexelIndex(uint lightIndex)
{
    return lightIndex & kEnvironmentSampleMask;
}

uint encodeEnvironmentSample(uint texelIndex)
{
    return kEnvironmentSampleBit | (texelIndex & kEnvironmentSampleMask);
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

float3 environmentLocalDirectionFromUv(float2 uv)
{
    float phi = (uv.x - 0.5) * (2.0 * kPi);
    float theta = uv.y * kPi;
    float sinTheta = sin(theta);
    return float3(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
}

float2 environmentUvFromDirection(float3 direction, float rotationRadians)
{
    float sine = sin(-rotationRadians);
    float cosine = cos(-rotationRadians);
    float3 localDirection = float3(
        direction.x * cosine + direction.z * sine,
        direction.y,
        -direction.x * sine + direction.z * cosine);
    float phi = atan2(localDirection.z, localDirection.x);
    float u = frac(phi / (2.0 * kPi) + 0.5);
    float v = acos(clamp(localDirection.y, -1.0, 1.0)) / kPi;
    return float2(u, v);
}

float environmentTexelSolidAngle(uint y, uint width, uint height)
{
    float theta = (float(y) + 0.5) * (kPi / float(max(height, 1u)));
    return max(
        (2.0 * kPi / float(max(width, 1u))) *
            (kPi / float(max(height, 1u))) *
            sin(theta),
        0.0000001);
}

float3 environmentRadiance(float3 direction, SceneRtxdiPush push)
{
    if (!behaviorEnabled(push, kBehaviorEnvironmentEnabled) || push.environmentIntensity <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    uint width = 0u;
    uint height = 0u;
    gEnvironmentMap.GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return float3(0.0, 0.0, 0.0);
    }
    float2 uv = environmentUvFromDirection(direction, push.environmentRotationRadians);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1u, height - 1u));
    return max(gEnvironmentMap.Load(int3(texel, 0)).rgb, float3(0.0, 0.0, 0.0)) *
        push.environmentIntensity;
}

LightEvaluation evaluateEnvironmentSample(uint encodedSample, Surface surface, SceneRtxdiPush push)
{
    LightEvaluation evaluation = emptyLightEvaluation();
    evaluation.distance = max(push.clipOrtho.y, push.sceneCenterRadius.w * 16.0);

    uint width = 0u;
    uint height = 0u;
    gEnvironmentMap.GetDimensions(width, height);
    uint texelIndex = environmentTexelIndex(encodedSample);
    if (!behaviorEnabled(push, kBehaviorEnvironmentEnabled) || width == 0u || height == 0u ||
        texelIndex >= width * height) {
        return evaluation;
    }

    uint2 texel = uint2(texelIndex % width, texelIndex / width);
    if (texel.y >= height) {
        return evaluation;
    }
    float2 uv = (float2(texel) + 0.5) / float2(width, height);
    evaluation.direction = safeNormalize(
        environmentLocalDirectionToWorld(
            environmentLocalDirectionFromUv(uv),
            push.environmentRotationRadians),
        float3(0.0, 1.0, 0.0));
    float nDotL = saturate(dot(surface.normal, evaluation.direction));
    float3 halfVector = safeNormalize(evaluation.direction + surface.viewDirection, surface.normal);
    float nDotH = saturate(dot(surface.normal, halfVector));
    float specularPower = lerp(128.0, 4.0, surface.roughness);
    float3 incidentRadiance = max(
        gEnvironmentMap.Load(int3(texel, 0)).rgb,
        float3(0.0, 0.0, 0.0)) * push.environmentIntensity;
    evaluation.diffuseRadiance = incidentRadiance * (nDotL / kPi);
    evaluation.specularRadiance = incidentRadiance * nDotL *
        ((specularPower + 2.0) / (2.0 * kPi)) *
        pow(nDotH, specularPower);
    evaluation.contribution = evaluation.diffuseRadiance * surface.diffuseAlbedo +
        evaluation.specularRadiance * surface.specularF0;
    evaluation.target = max(luminance(evaluation.contribution), 0.0);
    return evaluation;
}

LightEvaluation evaluateLightSample(uint lightIndex, Surface surface, SceneRtxdiPush push)
{
    return isEnvironmentSample(lightIndex)
        ? evaluateEnvironmentSample(lightIndex, surface, push)
        : evaluateLight(lightIndex, surface, push);
}

DIReservoir emptyReservoir()
{
    DIReservoir reservoir;
    reservoir.lightIndex = 0xffffffffu;
    reservoir.weightSum = 0.0;
    reservoir.selectedTarget = 0.0;
    reservoir.M = 0u;
    return reservoir;
}

bool reservoirValid(DIReservoir reservoir, SceneRtxdiPush push)
{
    uint environmentWidth = 0u;
    uint environmentHeight = 0u;
    gEnvironmentMap.GetDimensions(environmentWidth, environmentHeight);
    bool sampleIndexValid = isEnvironmentSample(reservoir.lightIndex)
        ? behaviorEnabled(push, kBehaviorEnvironmentEnabled) &&
            environmentTexelIndex(reservoir.lightIndex) < environmentWidth * environmentHeight
        : reservoir.lightIndex < push.lightCount;
    return sampleIndexValid &&
        reservoir.M > 0u &&
        reservoir.weightSum > 0.0 &&
        reservoir.selectedTarget > 0.0;
}

void updateReservoir(
    inout DIReservoir reservoir,
    uint lightIndex,
    float candidateWeight,
    float candidateTarget,
    uint candidateM,
    float randomValue)
{
    if (candidateM == 0u) {
        return;
    }
    reservoir.M += candidateM;
    if (candidateWeight <= 0.0 || candidateTarget <= 0.0) {
        return;
    }
    reservoir.weightSum += candidateWeight;
    if (randomValue * reservoir.weightSum < candidateWeight) {
        reservoir.lightIndex = lightIndex;
        reservoir.selectedTarget = candidateTarget;
    }
}

void clampReservoirHistory(inout DIReservoir reservoir, uint maximumM)
{
    if (reservoir.M > maximumM && maximumM > 0u) {
        float scale = float(maximumM) / float(reservoir.M);
        reservoir.weightSum *= scale;
        reservoir.M = maximumM;
    }
}

void combineReservoir(
    inout DIReservoir reservoir,
    DIReservoir candidate,
    float currentTarget,
    uint maximumM,
    float randomValue,
    SceneRtxdiPush push)
{
    if (!reservoirValid(candidate, push) || currentTarget <= 0.0) {
        return;
    }
    uint candidateM = min(candidate.M, maximumM);
    float historyScale = float(candidateM) / float(max(candidate.M, 1u));
    float candidateWeight = currentTarget *
        candidate.weightSum /
        max(candidate.selectedTarget, 0.000001) *
        historyScale;
    updateReservoir(
        reservoir,
        candidate.lightIndex,
        candidateWeight,
        currentTarget,
        candidateM,
        randomValue);
    clampReservoirHistory(reservoir, maximumM);
}

uint4 packReservoir(DIReservoir reservoir)
{
    return uint4(
        reservoir.lightIndex,
        asuint(reservoir.weightSum),
        asuint(reservoir.selectedTarget),
        reservoir.M);
}

DIReservoir unpackReservoir(uint4 packed)
{
    DIReservoir reservoir;
    reservoir.lightIndex = packed.x;
    reservoir.weightSum = asfloat(packed.y);
    reservoir.selectedTarget = asfloat(packed.z);
    reservoir.M = packed.w;
    return reservoir;
}

bool previousSurfaceMatches(
    int2 pixel,
    Surface surface,
    float positionThreshold,
    SceneRtxdiPush push)
{
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= int(push.width) || pixel.y >= int(push.height)) {
        return false;
    }
    float4 previousPosition = gPositionPrevious[pixel];
    float4 previousNormal = gNormalPrevious[pixel];
    if (previousPosition.w < 0.5 || previousNormal.w < 0.5) {
        return false;
    }
    return distance(previousPosition.xyz, surface.position) <= positionThreshold &&
        dot(safeNormalize(previousNormal.xyz, surface.normal), surface.normal) >= push.normalThreshold;
}

float3 toneMap(float3 color, float exposure)
{
    color = 1.0 - exp(-max(color, 0.0) * exposure);
    return pow(saturate(color), 1.0 / 2.2);
}

float3 skyColor(float3 direction)
{
    float sky = saturate(direction.y * 0.5 + 0.5);
    return lerp(float3(0.025, 0.03, 0.04), float3(0.12, 0.18, 0.28), sky);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sceneRtxdiMain(uint3 dispatchThreadId : SV_DispatchThreadID, uniform SceneRtxdiPush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    uint seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.frameIndex + 1u) * 0x9e3779b9u);
    float2 samplePixel = float2(pixel) + float2(0.5, 0.5);
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    HitInfo hit;
    if (!traceClosest(ray, push, hit)) {
        float3 sky = behaviorEnabled(push, kBehaviorEnvironmentEnabled)
            ? (behaviorEnabled(push, kBehaviorEnvironmentVisible)
                ? environmentRadiance(ray.direction, push)
                : float3(0.0, 0.0, 0.0))
            : skyColor(ray.direction);
        gReservoirCurrent[pixel] = packReservoir(emptyReservoir());
        gPositionCurrent[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gNormalCurrent[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gNoisyDiffuse[pixel] = RELAX_FrontEnd_PackRadianceAndHitDist(float3(0.0, 0.0, 0.0), 0.0, true);
        gNoisySpecular[pixel] = RELAX_FrontEnd_PackRadianceAndHitDist(float3(0.0, 0.0, 0.0), 0.0, true);
        gNrdNormalRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(
            float3(0.0, 1.0, 0.0),
            1.0,
            0.0);
        gNrdMotionVectors[pixel] = float4(
            motionVectorToPrevious(
                ray.origin + ray.direction * max(push.clipOrtho.y, 1.0),
                samplePixel,
                push),
            0.0,
            0.0);
        gNrdViewZ[pixel] = max(push.clipOrtho.y, 1.0);
        gBaseColorMetalness[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gEmissive[pixel] = float4(sky, 1.0);
        gOutput[pixel] = float4(toneMap(sky, push.exposure), 1.0);
        return;
    }

    Surface surface = makeSurface(hit, ray, push);
    gPositionCurrent[pixel] = float4(surface.position, 1.0);
    gNormalCurrent[pixel] = float4(surface.normal, 1.0);
    gNrdNormalRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(
        surface.normal,
        surface.roughness,
        0.0);
    gNrdMotionVectors[pixel] = float4(
        motionVectorToPrevious(surface.position, samplePixel, push),
        0.0,
        0.0);
    gNrdViewZ[pixel] = cameraLinearDepth(surface.position, push);
    gBaseColorMetalness[pixel] = float4(surface.baseColor, surface.metalness);
    float3 emissiveAndAmbient = surface.emissive + surface.diffuseAlbedo * 0.012;
    gEmissive[pixel] = float4(emissiveAndAmbient, 1.0);

    DIReservoir reservoir = emptyReservoir();
    ReGIRSelectionContext reGIRContext = initializeReGIRSelectionContext(
        push,
        surface,
        seed);
    uint localSampleCount = max(push.initialSampleCount, 1u);
    uint environmentSampleCount = behaviorEnabled(push, kBehaviorEnvironmentEnabled)
        ? push.environmentSampleCount
        : 0u;
    uint initialSampleCount = localSampleCount + environmentSampleCount;
    float localTechniqueProbability = float(localSampleCount) / float(max(initialSampleCount, 1u));
    float environmentTechniqueProbability = float(environmentSampleCount) / float(max(initialSampleCount, 1u));
    for (uint sampleIndex = 0u; sampleIndex < localSampleCount; ++sampleIndex) {
        uint lightIndex;
        float sourcePdf;
        bool sampled = sampleLocalLight(push, reGIRContext, seed, lightIndex, sourcePdf);
        LightEvaluation evaluation = sampled
            ? evaluateLight(lightIndex, surface, push)
            : emptyLightEvaluation();
        if (behaviorEnabled(push, kBehaviorInitialVisibility) && evaluation.target > 0.0) {
            float3 offsetNormal = dot(evaluation.direction, surface.geometryNormal) >= 0.0
                ? surface.geometryNormal
                : -surface.geometryNormal;
            if (traceShadow(
                    surface.position + offsetNormal * 0.001,
                    evaluation.direction,
                    evaluation.distance,
                    push)) {
                evaluation.target = 0.0;
            }
        }
        updateReservoir(
            reservoir,
            lightIndex,
            evaluation.target / max(sourcePdf * localTechniqueProbability, 0.000001),
            evaluation.target,
            1u,
            randomFloat(seed));
    }

    for (uint sampleIndex = 0u; sampleIndex < environmentSampleCount; ++sampleIndex) {
        uint2 texel;
        float texelPdf;
        bool sampled = sampleEnvironmentTexel(push, seed, texel, texelPdf);
        uint environmentWidth = 1u;
        uint environmentHeight = 1u;
        gEnvironmentMap.GetDimensions(environmentWidth, environmentHeight);
        uint texelIndex = texel.y * environmentWidth + texel.x;
        uint lightIndex = encodeEnvironmentSample(texelIndex);
        LightEvaluation evaluation = sampled
            ? evaluateEnvironmentSample(lightIndex, surface, push)
            : emptyLightEvaluation();
        if (behaviorEnabled(push, kBehaviorInitialVisibility) && evaluation.target > 0.0) {
            float3 offsetNormal = dot(evaluation.direction, surface.geometryNormal) >= 0.0
                ? surface.geometryNormal
                : -surface.geometryNormal;
            if (traceShadow(
                    surface.position + offsetNormal * 0.001,
                    evaluation.direction,
                    evaluation.distance,
                    push)) {
                evaluation.target = 0.0;
            }
        }
        float sourcePdfSolidAngle = texelPdf /
            environmentTexelSolidAngle(texel.y, environmentWidth, environmentHeight);
        updateReservoir(
            reservoir,
            lightIndex,
            evaluation.target /
                max(sourcePdfSolidAngle * environmentTechniqueProbability, 0.000001),
            evaluation.target,
            1u,
            randomFloat(seed));
    }

    uint maximumM = max(initialSampleCount * max(push.maxHistoryLength, 1u), initialSampleCount);
    float positionThreshold = max(push.sceneCenterRadius.w * push.depthThreshold, 0.0001);
    if (push.hasHistory != 0u && behaviorEnabled(push, kBehaviorTemporalReuse)) {
        float2 previousUv;
        if (projectWorldToPreviousUv(surface.position, push, previousUv)) {
            int2 previousPixel = int2(previousUv * float2(push.width, push.height));
            if (previousSurfaceMatches(previousPixel, surface, positionThreshold, push)) {
                DIReservoir previousReservoir = unpackReservoir(gReservoirPrevious[previousPixel]);
                if (reservoirValid(previousReservoir, push)) {
                    float currentTarget = evaluateLightSample(previousReservoir.lightIndex, surface, push).target;
                    combineReservoir(
                        reservoir,
                        previousReservoir,
                        currentTarget,
                        maximumM,
                        randomFloat(seed),
                        push);
                }
            }
        }
    }

    if (push.hasHistory != 0u && behaviorEnabled(push, kBehaviorSpatialReuse)) {
        uint spatialSampleCount = min(push.spatialSampleCount, 16u);
        for (uint sampleIndex = 0u; sampleIndex < spatialSampleCount; ++sampleIndex) {
            float angle = randomFloat(seed) * 2.0 * kPi;
            float sampleRadius = sqrt(randomFloat(seed)) * 24.0 + 1.0;
            int2 offset = int2(round(float2(cos(angle), sin(angle)) * sampleRadius));
            int2 neighborPixel = int2(pixel) + offset;
            float spatialPositionThreshold = positionThreshold * (1.0 + sampleRadius / 24.0);
            if (!previousSurfaceMatches(neighborPixel, surface, spatialPositionThreshold, push)) {
                continue;
            }
            DIReservoir neighborReservoir = unpackReservoir(gReservoirPrevious[neighborPixel]);
            if (!reservoirValid(neighborReservoir, push)) {
                continue;
            }
            float currentTarget = evaluateLightSample(neighborReservoir.lightIndex, surface, push).target;
            combineReservoir(
                reservoir,
                neighborReservoir,
                currentTarget,
                maximumM,
                randomFloat(seed),
                push);
        }
    }

    clampReservoirHistory(reservoir, maximumM);
    gReservoirCurrent[pixel] = packReservoir(reservoir);

    float3 noisyDiffuse = float3(0.0, 0.0, 0.0);
    float3 noisySpecular = float3(0.0, 0.0, 0.0);
    float lightDistance = 0.0;
    float3 color = emissiveAndAmbient;
    if (reservoirValid(reservoir, push)) {
        LightEvaluation selected = evaluateLightSample(reservoir.lightIndex, surface, push);
        float reservoirWeight = reservoir.weightSum /
            max(float(reservoir.M) * reservoir.selectedTarget, 0.000001);
        float3 offsetNormal = dot(selected.direction, surface.geometryNormal) >= 0.0
            ? surface.geometryNormal
            : -surface.geometryNormal;
        bool occluded = traceShadow(
            surface.position + offsetNormal * 0.001,
            selected.direction,
            selected.distance,
            push);
        if (!occluded) {
            noisyDiffuse = selected.diffuseRadiance * reservoirWeight;
            noisySpecular = selected.specularRadiance * reservoirWeight;
            lightDistance = selected.distance;
            color += selected.contribution * reservoirWeight;
        }
    }
    gNoisyDiffuse[pixel] = RELAX_FrontEnd_PackRadianceAndHitDist(
        noisyDiffuse,
        lightDistance,
        true);
    gNoisySpecular[pixel] = RELAX_FrontEnd_PackRadianceAndHitDist(
        noisySpecular,
        lightDistance,
        true);

    if (visualizationMode(push) == kVisualizationLightId) {
        float3 selectedColor = reservoirValid(reservoir, push)
            ? hashColor(reservoir.lightIndex)
            : float3(0.0, 0.0, 0.0);
        gOutput[pixel] = float4(selectedColor, 1.0);
        return;
    }
    if (visualizationMode(push) == kVisualizationHistory) {
        float history = saturate(float(reservoir.M) / float(max(maximumM, 1u)));
        float3 heat = lerp(float3(0.02, 0.05, 0.2), float3(1.0, 0.25, 0.02), history);
        gOutput[pixel] = float4(heat, 1.0);
        return;
    }
    if (visualizationMode(push) == kVisualizationReGIRCells) {
        gOutput[pixel] = float4(visualizeReGIRCell(surface.position), 1.0);
        return;
    }

    gOutput[pixel] = float4(toneMap(color, push.exposure), 1.0);
}
