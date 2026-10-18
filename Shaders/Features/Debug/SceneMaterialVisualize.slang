#language slang 2026
module SceneMaterialVisualize;
import MaterialTextureDecode;
import AlphaCoverage;
import Material;
import OpenPBRMaterialParameters;
using Metallic.Material;

import Core;
using Metallic;


#define gScene resolveAccelerationStructure(getResourceParameters<SceneResourceParameters>().scene)

#define gOutput resolveUniform(getResourceParameters<SceneResourceParameters>().output)

struct MaterialVisualizePrimitive
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

struct MaterialVisualizeInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint flags;
    uint padding;
};

#define gMaterials resolveBuffer<StructuredBuffer<SceneMaterialData>>(getResourceParameters<SceneResourceParameters>().materials)

#define gMaterialTextures(index) resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<SceneResourceParameters>().materialTextures.load(index)))


static const uint kInvalidMaterialTextureIndex = 0xffffffffu;
static const uint kModeMaterial = 0u;
static const uint kModeBaseColor = 1u;
static const uint kModeNormal = 2u;
static const uint kModeRoughness = 3u;
static const uint kModeMetallic = 4u;
static const uint kModeAo = 5u;
static const uint kModeGeometryNormal = 6u;
static const uint kModeVertexNormal = 7u;
static const uint kModeNormalTexture = 8u;
static const uint kModeTangent = 9u;
static const uint kModeBitangent = 10u;
static const uint kModeNRDNormalRoughness = 11u;
static const uint kModeNormalDeviation = 12u;
static const uint kPrimitiveHasAuthoredTangents = 1u;

struct SceneMaterialVisualizationPush
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    uint width;
    uint height;
    uint mode;
    uint materialTextureCount;
    float bitangentFlip;
};

import ScenePathTypes;

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float3 hashColor(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;

    float3 color = float3(
        float(value & 255u),
        float((value >> 8) & 255u),
        float((value >> 16) & 255u)) / 255.0;
    return lerp(float3(0.16, 0.20, 0.26), color, 0.82);
}

float3 packNrdNormalRoughnessForDisplay(float3 normal, float roughness)
{
    float3 n = safeNormalize(normal, float3(0.0, 1.0, 0.0));
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 0.000001);

    float3 packed;
    packed.y = n.y * 0.5 + 0.5;
    packed.x = n.x * 0.5 + packed.y;
    packed.y -= n.x * 0.5;

    roughness = max(saturate(roughness), 1.5 / 512.0);
    float signedRoughness = n.z < 0.0 ? -roughness : roughness;
    packed.z = signedRoughness * 0.5 + 0.5;
    return saturate(packed);
}

float3 srgbToLinear(float3 color)
{
    return float3(
        color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4));
}

float3 rayDirectionFromPixel(float2 pixel, SceneMaterialVisualizationPush push)
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

float3 rayOriginFromPixel(float2 pixel, SceneMaterialVisualizationPush push)
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

bool hasMaterialTexture(MaterialTextureInfo textureInfo, SceneMaterialVisualizationPush push)
{
    return textureInfo.textureIndex != kInvalidMaterialTextureIndex &&
        textureInfo.textureIndex < push.materialTextureCount;
}

float2 transformMaterialUv(MaterialTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

import StreamTextureSampling;

StructuredBuffer<TextureSamplingMetadata> sceneStreamTextureMetadata()
{
    return resolveBuffer<StructuredBuffer<TextureSamplingMetadata>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneTextureSamplingMetadataOffset));
}
RWStructuredBuffer<uint> sceneStreamTextureDemandMip()
{
    return resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneTextureDemandMipOffset));
}
RWStructuredBuffer<uint> sceneStreamTextureDemandHits()
{
    return resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneTextureDemandHitsOffset));
}

float4 loadMaterialTexture(
    MaterialTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    SceneMaterialVisualizationPush push)
{
    if (!hasMaterialTexture(textureInfo, push)) {
        return fallback;
    }

    uint width = 1u;
    uint height = 1u;
    uint textureIndex = textureInfo.textureIndex;
    requestStreamTextureMip0(sceneStreamTextureMetadata(), sceneStreamTextureDemandMip(), sceneStreamTextureDemandHits(), textureIndex);
    uint mip = uint(ceil(streamTextureSamplingFloor(sceneStreamTextureMetadata(), textureIndex)));
    uint mipCount;
    gMaterialTextures(textureIndex).GetDimensions(mip, width, height, mipCount);
    if (width == 0u || height == 0u) {
        return fallback;
    }

    float2 wrappedUv = frac(transformMaterialUv(textureInfo, uv));
    uint2 texel = min(uint2(wrappedUv * float2(width, height)), uint2(width - 1u, height - 1u));
    return decodeMaterialTextureSample(gMaterialTextures(textureIndex).Load(int3(texel, mip)), textureInfo.transform0.w);
}



bool alphaMaskAcceptsHit(HitInfo hit, SceneMaterialVisualizationPush push)
{
    OpenPBRMaterialParameters material = sceneOpenPBRParameters(gMaterials[hit.materialIndex]);
    bool blend = abs(material.textureParams.w - 2.0) < 0.25;
    if (!blend && abs(material.textureParams.w - 1.0) > 0.25) {
        return true;
    }

    CoverageContext context;
    context.uv = hit.texcoord;
    context.baseAlpha = material.baseColor.a; context.cutoff = material.params.z; context.alphaMode = material.textureParams.w;
    CoverageProgram program;
    // Initialize the resource before branching: a buffer-valued phi with an
    // undefined legacy arm requires VariablePointersStorageBuffer in SPIR-V.
    program.inputs = resolveUniform(getResourceParameters<SceneResourceParameters>().materialValues);
    bool custom = material.attenuationColor.w > 0.5;
    bool needsAlpha = true;
    if (custom) {
        needsAlpha = program.usesBaseAlpha(coverageInstanceRef(hit.materialIndex));
    }
    if (needsAlpha) {
        if (hasMaterialTexture(material.baseColorTexture, push)) {
            requestStreamTextureMip0(sceneStreamTextureMetadata(), sceneStreamTextureDemandMip(), sceneStreamTextureDemandHits(), material.baseColorTexture.textureIndex);
            context.baseAlpha *= sampleAlphaCoverage(gMaterialTextures(material.baseColorTexture.textureIndex), transformMaterialUv(material.baseColorTexture, context.uv), uint(ceil(streamTextureSamplingFloor(sceneStreamTextureMetadata(), material.baseColorTexture.textureIndex))));
        }
    }
    if (custom) { return program.evaluateCoverage(context, coverageInstanceRef(hit.materialIndex)).accepted; }
    return coverageFromOpacity(context, context.baseAlpha).accepted;
}

float3 transformObjectNormalToWorld(float3 normal, float3x4 worldToObject)
{
    return mul(normal, worldToObject).xyz;
}

import StreamRayGeometry;
using Metallic.StreamGeometry;

bool traceClosest(Ray ray, SceneMaterialVisualizationPush push, out HitInfo hit)
{
    hit = (HitInfo)0;
    RayDesc desc;
    desc.Origin = ray.origin; desc.Direction = ray.direction;
    desc.TMin = max(push.clipOrtho.x, 0.0001);
    desc.TMax = max(push.clipOrtho.y, desc.TMin + 0.0001);
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(gScene, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, desc);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { continue; }
        HitInfo candidate;
        if (streamRaySurface(query.CandidateInstanceID(), uint(query.CandidateClusterID()),
                query.CandidatePrimitiveIndex(), query.CandidateTriangleBarycentrics(), ray,
                query.CandidateTriangleRayT(), push.bitangentFlip, candidate) && alphaMaskAcceptsHit(candidate, push)) {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT &&
        streamRaySurface(query.CommittedInstanceID(), uint(query.CommittedClusterID()),
            query.CommittedPrimitiveIndex(), query.CommittedTriangleBarycentrics(), ray,
            query.CommittedRayT(), push.bitangentFlip, hit);
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

float3 visualizeHit(HitInfo hit, SceneMaterialVisualizationPush push)
{
    OpenPBRMaterialParameters material = sceneOpenPBRParameters(gMaterials[hit.materialIndex]);
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
    float4 occlusionSample = loadMaterialTexture(
        material.occlusionTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push);

    if (push.mode == kModeBaseColor) {
        return saturate(Metallic.WorkingColor::toLinearRec709(materialTextureModulate(material.baseColor.rgb, baseColorSample.rgb, material.baseColorTexture.transform0.w)));
    }
    if (push.mode == kModeNormal) {
        float3 normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
        return saturate(normal * 0.5 + 0.5);
    }
    if (push.mode == kModeGeometryNormal) {
        return saturate(hit.geometryNormal * 0.5 + 0.5);
    }
    if (push.mode == kModeVertexNormal) {
        return saturate(hit.normal * 0.5 + 0.5);
    }
    if (push.mode == kModeNormalTexture) {
        return saturate(normalSample.rgb);
    }
    if (push.mode == kModeTangent) {
        return saturate(safeNormalize(hit.tangent.xyz, float3(1.0, 0.0, 0.0)) * 0.5 + 0.5);
    }
    if (push.mode == kModeBitangent) {
        return saturate(safeNormalize(hit.bitangent, float3(0.0, 1.0, 0.0)) * 0.5 + 0.5);
    }
    if (push.mode == kModeNRDNormalRoughness) {
        float3 normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
        float roughness = saturate(material.params.y * metallicRoughnessSample.g);
        return packNrdNormalRoughnessForDisplay(normal, roughness);
    }
    if (push.mode == kModeNormalDeviation) {
        float normalDot = abs(dot(safeNormalize(hit.normal, hit.geometryNormal), safeNormalize(hit.geometryNormal, hit.normal)));
        float deviation = saturate(acos(clamp(normalDot, -1.0, 1.0)) / 0.7853981633974483);
        return lerp(float3(0.02, 0.14, 0.03), float3(1.0, 0.08, 0.02), deviation);
    }
    if (push.mode == kModeRoughness) {
        float roughness = saturate(material.params.y * metallicRoughnessSample.g);
        return float3(roughness, roughness, roughness);
    }
    if (push.mode == kModeMetallic) {
        float metallic = saturate(material.params.x * metallicRoughnessSample.b);
        return float3(metallic, metallic, metallic);
    }
    if (push.mode == kModeAo) {
        float ao = lerp(1.0, saturate(occlusionSample.r), saturate(material.textureParams.y));
        return float3(ao, ao, ao);
    }
    return hashColor(hit.materialIndex + 1u);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sceneMaterialVisualizeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    SceneMaterialVisualizationPush push = getConstants<SceneMaterialVisualizationPush>();
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float2 pixelCenter = float2(pixel) + 0.5;
    Ray ray;
    ray.origin = rayOriginFromPixel(pixelCenter, push);
    ray.direction = rayDirectionFromPixel(pixelCenter, push);

    HitInfo hit;
    float3 color = float3(0.010, 0.012, 0.016);
    if (traceClosest(ray, push, hit)) {
        color = visualizeHit(hit, push);
    }

    gOutput[pixel] = float4(color, 1.0);
}
