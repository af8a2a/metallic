#language slang 2026
module SceneStandardGuides;
import Core;
import MaterialTextureDecode;
import RTXCRMaterialAdapter;
import RayMaterialExecution;
import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import Lighting;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.SceneSurface;
using Metallic.ScenePath;
import RTXCRFiber;
namespace Metallic.SceneStandardGuides
{
// Standard renderer guide evaluation from an authored hit. Traversal remains
// with the caller; the compute probe exercises this same production function.

public struct DLSSRRGuideData
{
    public float3 albedo;
    public float3 specularAlbedo;
    public float3 normal;
    public float roughness;
    public float2 motionVector;
    public float linearDepth;
    public float specularHitDistance;
};

public DLSSRRGuideData makeDefaultDlssRrGuideData(ScenePathTracePush push)
{
    DLSSRRGuideData data;
    data.albedo = float3(0.0, 0.0, 0.0);
    data.specularAlbedo = float3(0.0, 0.0, 0.0);
    data.normal = float3(0.0, 1.0, 0.0);
    data.roughness = 1.0;
    data.motionVector = float2(0.0, 0.0);
    data.linearDepth = max(push.clipOrtho.y, 0.0);
    data.specularHitDistance = 0.0;
    return data;
}

public DLSSRRGuideData evaluateStandardDlssRrGuidesAtHit(
    Ray ray,
    RayCone rayCone,
    float2 samplePixel,
    ScenePathTracePush push,
    inout uint seed,
    HitInfo hit)
{
    DLSSRRGuideData data = makeDefaultDlssRrGuideData(push);
    rayCone = propagateRayCone(rayCone, hit.t);
    SceneMaterialData sceneMaterial = loadSceneMaterial(hit.materialIndex);
    if (materialDomain(sceneMaterial) == kMaterialDomainFiber) {
        let guides = fiberGuideParameters(sceneFiberParameters(sceneMaterial));
        data.albedo = guides.albedo;
        data.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);
        float nDotV = saturate(dot(data.normal, -ray.direction));
        data.specularAlbedo = saturate(approxSpecularIntegralGGX(guides.specularReflectance,
            guides.roughness * guides.roughness, nDotV));
        data.roughness = guides.roughness;
        data.motionVector = motionVectorToPrevious(hit.position, samplePixel, push);
        data.linearDepth = cameraLinearDepth(hit.position, push);
        return data;
    }
    OpenPBRMaterialParameters material = sceneOpenPBRParameters(sceneMaterial);
    bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
    TextureFootprint footprint = (TextureFootprint)0;
    if (!deterministicGlassTextures) {
        footprint = resolveRayConeTextureFootprint(hit, rayCone, ray.direction, footprint);
    }
    float4 baseColorSample = loadMaterialTextureForMaterialResolved(
        material.baseColorTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        deterministicGlassTextures,
        seed, footprint);
    float4 metallicRoughnessSample = loadMaterialTextureForMaterialResolved(
        material.metallicRoughnessTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        deterministicGlassTextures,
        seed, footprint);
    float4 normalSample = loadMaterialTextureForMaterialResolved(
        material.normalTexture,
        hit.texcoord,
        float4(0.5, 0.5, 1.0, 1.0),
        push,
        deterministicGlassTextures,
        seed, footprint);

    hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
    hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);

    float3 baseColor = saturate(materialTextureModulate(material.baseColor.rgb, baseColorSample.rgb, material.baseColorTexture.transform0.w));
    float metallic = saturate(material.params.x * metallicRoughnessSample.b);
    float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.02, 1.0);
    float ior = clamp(material.glassParams.y, 1.0, 3.0);
    float dielectricF0 = (ior - 1.0) / (ior + 1.0);
    dielectricF0 *= dielectricF0;

    data.albedo = baseColor * (1.0 - metallic);
    data.normal = Metallic.ScenePath::safeNormalize(hit.normal, hit.geometryNormal);
    float nDotV = saturate(dot(data.normal, -ray.direction));
    float3 specularReflectance = lerp(float3(dielectricF0, dielectricF0, dielectricF0), baseColor, metallic);
    data.specularAlbedo = saturate(approxSpecularIntegralGGX(specularReflectance, roughness * roughness, nDotV));
    data.roughness = roughness;
    data.motionVector = motionVectorToPrevious(hit.position, samplePixel, push);
    data.linearDepth = cameraLinearDepth(hit.position, push);
    return data;
}
}
