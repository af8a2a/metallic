#language slang 2026
implementing SceneRTXDI;

import Core;
import MaterialTextureDecode;
import AlphaCoverage;
import Material;
import OpenPBRMaterialParameters;
import StreamTextureSampling;
import ScenePathTypes;
import StreamRayGeometry;
using Metallic;
using Metallic.Material;
using Metallic.StreamGeometry;

#define gScene resolveAccelerationStructure(getResourceParameters<SceneResourceParameters>().scene)
#define gMaterials resolveBuffer<StructuredBuffer<SceneMaterialData>>(getResourceParameters<SceneResourceParameters>().materials)
#define gMaterialTextures(index) resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<SceneResourceParameters>().materialTextures.load(index)))

bool hasMaterialTexture(MaterialTextureInfo textureInfo, SceneRTXDIPush push)
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
    SceneRTXDIPush push)
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



bool alphaMaskAcceptsHit(HitInfo hit, SceneRTXDIPush push)
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
    // Keep the buffer out of branch phis (no VariablePointersStorageBuffer).
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


bool traceClosest(Ray ray, SceneRTXDIPush push, out HitInfo hit, bool occlusionOnly = false)
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
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }
    // Coverage is already accepted; shadow callers need no committed surface.
    if (occlusionOnly) { return true; }
    return streamRaySurface(query.CommittedInstanceID(), uint(query.CommittedClusterID()),
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

bool traceShadow(float3 origin, float3 direction, float maxDistance, SceneRTXDIPush push)
{
    Ray ray; ray.origin = origin; ray.direction = direction;
    push.clipOrtho.x = 0.001;
    push.clipOrtho.y = max(maxDistance, 0.001);
    HitInfo hit;
    return traceClosest(ray, push, hit, true);
}


Surface makeSurface(HitInfo hit, Ray ray, SceneRTXDIPush push)
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
    surface.baseColor = saturate(materialTextureModulate(material.baseColor.rgb, baseColorSample.rgb, material.baseColorTexture.transform0.w));
    surface.metalness = saturate(material.params.x * metallicRoughnessSample.b);
    surface.roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.03, 1.0);
    surface.diffuseAlbedo = surface.baseColor * (1.0 - surface.metalness);
    surface.specularF0 = lerp(float3(0.04, 0.04, 0.04), surface.baseColor, surface.metalness);
    surface.emissive = max(materialTextureModulate(material.emissive.rgb, emissiveSample.rgb, material.emissiveTexture.transform0.w), 0.0);
    return surface;
}
