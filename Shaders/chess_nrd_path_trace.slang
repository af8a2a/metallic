#include "NRD.hlsli"

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure gScene;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gNoisyDiffuse;

[[vk::binding(2, 0)]]
RWTexture2D<float4> gNoisySpecular;

[[vk::binding(3, 0)]]
RWTexture2D<float4> gNormalRoughness;

[[vk::binding(4, 0)]]
RWTexture2D<float4> gMotionVectors;

[[vk::binding(5, 0)]]
RWTexture2D<float> gViewZ;

[[vk::binding(6, 0)]]
RWTexture2D<float4> gDirectLighting;

[[vk::binding(7, 0)]]
RWTexture2D<float4> gBaseColorMetalness;

[[vk::binding(8, 0)]]
RWTexture2D<float4> gNoisyColor;

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
};

struct PathTraceInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint flags;
    uint padding;
};

struct PathTraceTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint padding0;
    uint padding1;
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

[[vk::binding(9, 0)]]
StructuredBuffer<PathTraceVertex> gVertices;

[[vk::binding(10, 0)]]
StructuredBuffer<uint> gIndices;

[[vk::binding(11, 0)]]
StructuredBuffer<PathTracePrimitive> gPrimitives;

[[vk::binding(12, 0)]]
StructuredBuffer<PathTraceInstance> gInstances;

[[vk::binding(13, 0)]]
StructuredBuffer<PathTraceMaterial> gMaterials;

[[vk::binding(14, 0)]]
Texture2D<float4> gMaterialTextures[128];

struct ChessNrdPush
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    uint width;
    uint height;
    uint maxDepth;
    uint samples;
    uint frameIndex;
    uint denoiserMode;
    uint padding0;
    uint padding1;
};

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
    float3 tangent;
    float bitangentSign;
    float2 texcoord;
    uint materialIndex;
    bool frontFace;
};

struct MaterialSample
{
    float3 baseColor;
    float metallic;
    float roughness;
    float3 emissive;
    float occlusion;
    float3 normal;
    float transmission;
    float ior;
    float thickness;
    float attenuationDistance;
    float3 attenuationColor;
    float diffuseTransmission;
    float3 diffuseTransmissionColor;
};

static const uint kNrdReblur = 0u;
static const uint kNrdRelax = 1u;
static const uint kNrdReference = 2u;
static const uint kInvalidTextureIndex = 0xffffffffu;
static const uint kMaxMaterialTextures = 128u;
static const float kPi = 3.14159265358979323846;
static const float4 kDiffHitDistParams = float4(3.0, 0.1, 20.0, -25.0);
static const float4 kSpecHitDistParams = float4(3.0, 0.1, 20.0, -25.0);

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

float rand(inout uint state)
{
    state = hash(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

float3 rayDirectionFromPixel(float2 pixel, ChessNrdPush push)
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
    return safeNormalize(forward + right * (ndc.x * aspect * halfHeight) + up * (ndc.y * halfHeight), forward);
}

float3 rayOriginFromPixel(float2 pixel, ChessNrdPush push)
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

float3 offsetRay(float3 position, float3 normal)
{
    return position + normal * 0.001;
}

float3 sunDirection()
{
    return safeNormalize(float3(-0.35, 0.82, -0.28), float3(0.0, 1.0, 0.0));
}

float3 skyRadiance(float3 direction)
{
    float t = saturate(direction.y * 0.5 + 0.5);
    float3 horizon = float3(0.70, 0.76, 0.82);
    float3 zenith = float3(0.14, 0.23, 0.42);
    float sun = pow(saturate(dot(direction, sunDirection())), 512.0);
    return lerp(horizon, zenith, t) * 0.55 + sun * float3(8.0, 7.1, 5.8);
}

float3 srgbToLinear(float3 value)
{
    return pow(max(value, float3(0.0, 0.0, 0.0)), 2.2);
}

bool hasMaterialTexture(PathTraceTextureInfo textureInfo)
{
    return textureInfo.textureIndex != kInvalidTextureIndex &&
        textureInfo.textureIndex < kMaxMaterialTextures;
}

float2 transformTexcoord(PathTraceTextureInfo textureInfo, float2 texcoord)
{
    float3 uv = float3(texcoord, 1.0);
    return float2(dot(textureInfo.transform0.xyz, uv), dot(textureInfo.transform1.xyz, uv));
}

uint wrapTexel(int value, uint size)
{
    int signedSize = int(max(size, 1u));
    int wrapped = value % signedSize;
    return uint(wrapped < 0 ? wrapped + signedSize : wrapped);
}

float4 loadMaterialTexture(PathTraceTextureInfo textureInfo, float2 texcoord, bool srgb)
{
    if (!hasMaterialTexture(textureInfo)) {
        return float4(1.0, 1.0, 1.0, 1.0);
    }

    Texture2D<float4> texture = gMaterialTextures[NonUniformResourceIndex(textureInfo.textureIndex)];
    uint width = 1u;
    uint height = 1u;
    texture.GetDimensions(width, height);

    float2 uv = frac(transformTexcoord(textureInfo, texcoord));
    float2 imageCoord = uv * float2(width, height) - 0.5;
    int2 baseCoord = int2(floor(imageCoord));
    float2 blend = frac(imageCoord);

    uint2 p00 = uint2(wrapTexel(baseCoord.x, width), wrapTexel(baseCoord.y, height));
    uint2 p10 = uint2(wrapTexel(baseCoord.x + 1, width), p00.y);
    uint2 p01 = uint2(p00.x, wrapTexel(baseCoord.y + 1, height));
    uint2 p11 = uint2(p10.x, p01.y);

    float4 c00 = texture.Load(int3(p00, 0));
    float4 c10 = texture.Load(int3(p10, 0));
    float4 c01 = texture.Load(int3(p01, 0));
    float4 c11 = texture.Load(int3(p11, 0));
    float4 value = lerp(lerp(c00, c10, blend.x), lerp(c01, c11, blend.x), blend.y);
    if (srgb) {
        value.rgb = srgbToLinear(value.rgb);
    }
    return value;
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
    return safeNormalize(tangent * (cos(phi) * sinTheta) + bitangent * (sin(phi) * sinTheta) + normal * cosTheta, normal);
}

float3 sampleSubsurfaceDiffuseDirection(float3 normal, inout uint seed)
{
    return sampleCosineHemisphere(-normal, seed);
}

bool traceShadow(float3 origin, float3 direction, float maxDistance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = max(maxDistance, 0.001);

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(gScene, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    while (query.Proceed()) {
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

bool traceClosestRange(Ray ray, float minDistance, float maxDistance, out HitInfo hit)
{
    hit = HitInfo();
    hit.t = 0.0;
    hit.position = ray.origin;
    hit.normal = float3(0.0, 1.0, 0.0);
    hit.geometryNormal = hit.normal;
    hit.tangent = float3(1.0, 0.0, 0.0);
    hit.bitangentSign = 1.0;
    hit.texcoord = float2(0.0, 0.0);
    hit.materialIndex = 0u;
    hit.frontFace = true;

    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    rayDesc.TMin = max(minDistance, 0.0001);
    rayDesc.TMax = max(maxDistance, rayDesc.TMin + 0.0001);

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(gScene, RAY_FLAG_FORCE_OPAQUE, 0xff, rayDesc);
    while (query.Proceed()) {
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
    float3 localPosition = p0 * weights.x + p1 * weights.y + p2 * weights.z;
    float3 localGeometryNormal = safeNormalize(cross(p1 - p0, p2 - p0), float3(0.0, 1.0, 0.0));
    float3 localNormal = v0.normal.xyz * weights.x + v1.normal.xyz * weights.y + v2.normal.xyz * weights.z;
    localNormal = dot(localNormal, localNormal) > 0.000001 ? safeNormalize(localNormal, localGeometryNormal) : localGeometryNormal;
    float4 localTangent4 = v0.tangent * weights.x + v1.tangent * weights.y + v2.tangent * weights.z;
    float3 localTangent = dot(localTangent4.xyz, localTangent4.xyz) > 0.000001
        ? safeNormalize(localTangent4.xyz, float3(1.0, 0.0, 0.0))
        : float3(1.0, 0.0, 0.0);
    float2 localTexcoord = v0.texcoord.xy * weights.x + v1.texcoord.xy * weights.y + v2.texcoord.xy * weights.z;

    float4x3 objectToWorld = query.CommittedObjectToWorld4x3();
    float4x3 worldToObject = query.CommittedWorldToObject4x3();
    hit.position = float3(mul(float4(localPosition, 1.0), objectToWorld));
    hit.geometryNormal = safeNormalize(float3(mul(worldToObject, localGeometryNormal).xyz), localGeometryNormal);
    hit.normal = safeNormalize(float3(mul(worldToObject, localNormal).xyz), hit.geometryNormal);
    hit.tangent = safeNormalize(float3(mul(worldToObject, localTangent).xyz), float3(1.0, 0.0, 0.0));
    hit.texcoord = localTexcoord;

    float3 view = -ray.direction;
    hit.frontFace = dot(hit.geometryNormal, view) >= 0.0;
    if (!hit.frontFace) {
        hit.geometryNormal = -hit.geometryNormal;
    }
    if (dot(hit.normal, hit.geometryNormal) < 0.0) {
        hit.normal = -hit.normal;
    }
    hit.tangent = safeNormalize(hit.tangent - hit.normal * dot(hit.normal, hit.tangent), safeNormalize(cross(float3(0.0, 1.0, 0.0), hit.normal), float3(1.0, 0.0, 0.0)));
    hit.bitangentSign = localTangent4.w >= 0.0 ? 1.0 : -1.0;

    hit.t = query.CommittedRayT();
    hit.materialIndex = instance.materialIndex;
    return true;
}

bool traceClosest(Ray ray, ChessNrdPush push, out HitInfo hit)
{
    return traceClosestRange(ray, push.clipOrtho.x, push.clipOrtho.y, hit);
}

bool traceClosestSecondary(Ray ray, ChessNrdPush push, out HitInfo hit)
{
    return traceClosestRange(ray, 0.001, push.clipOrtho.y, hit);
}

MaterialSample sampleMaterial(HitInfo hit)
{
    PathTraceMaterial source = gMaterials[hit.materialIndex];
    MaterialSample material;
    material.baseColor = saturate(source.baseColor.rgb);
    material.metallic = saturate(source.params.x);
    material.roughness = clamp(source.params.y, 0.03, 0.95);
    material.emissive = max(source.emissive.rgb, float3(0.0, 0.0, 0.0));
    material.occlusion = 1.0;
    material.normal = hit.normal;
    material.transmission = saturate(source.glassParams.x);
    material.ior = clamp(source.glassParams.y, 1.0, 3.0);
    material.thickness = max(source.glassParams.z, 0.0);
    material.attenuationDistance = max(source.glassParams.w, 0.0);
    material.attenuationColor = saturate(source.attenuationColor.rgb);
    material.diffuseTransmissionColor = saturate(source.diffuseTransmission.rgb);
    material.diffuseTransmission = saturate(source.diffuseTransmission.a);

    if (hasMaterialTexture(source.baseColorTexture)) {
        float4 baseColor = loadMaterialTexture(source.baseColorTexture, hit.texcoord, true);
        material.baseColor *= saturate(baseColor.rgb);
    }

    if (hasMaterialTexture(source.metallicRoughnessTexture)) {
        float4 metallicRoughness = loadMaterialTexture(source.metallicRoughnessTexture, hit.texcoord, false);
        material.roughness *= metallicRoughness.g;
        material.metallic *= metallicRoughness.b;
    }

    if (hasMaterialTexture(source.occlusionTexture)) {
        float occlusion = loadMaterialTexture(source.occlusionTexture, hit.texcoord, false).r;
        material.occlusion = 1.0 + source.textureParams.y * (occlusion - 1.0);
    }

    if (hasMaterialTexture(source.emissiveTexture)) {
        material.emissive *= loadMaterialTexture(source.emissiveTexture, hit.texcoord, true).rgb;
    }

    if (hasMaterialTexture(source.transmissionTexture)) {
        material.transmission *= loadMaterialTexture(source.transmissionTexture, hit.texcoord, false).r;
    }

    if (hasMaterialTexture(source.thicknessTexture)) {
        material.thickness *= loadMaterialTexture(source.thicknessTexture, hit.texcoord, false).g;
    }

    if (hasMaterialTexture(source.diffuseTransmissionTexture)) {
        material.diffuseTransmission *= loadMaterialTexture(
            source.diffuseTransmissionTexture,
            hit.texcoord,
            false).a;
    }

    if (hasMaterialTexture(source.diffuseTransmissionColorTexture)) {
        material.diffuseTransmissionColor *= loadMaterialTexture(
            source.diffuseTransmissionColorTexture,
            hit.texcoord,
            true).rgb;
    }

    if (hasMaterialTexture(source.normalTexture)) {
        float3 tangentNormal = loadMaterialTexture(source.normalTexture, hit.texcoord, false).xyz * 2.0 - 1.0;
        tangentNormal.xy *= max(source.textureParams.x, 0.0);
        tangentNormal = safeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));
        float3 fallbackBitangent = safeNormalize(cross(hit.geometryNormal, hit.tangent), float3(0.0, 0.0, 1.0));
        float3 bitangent = safeNormalize(cross(hit.normal, hit.tangent) * hit.bitangentSign, fallbackBitangent);
        material.normal = safeNormalize(
            hit.tangent * tangentNormal.x + bitangent * tangentNormal.y + hit.normal * tangentNormal.z,
            hit.normal);
        if (dot(material.normal, hit.geometryNormal) < 0.0) {
            material.normal = hit.normal;
        }
    }

    material.baseColor = saturate(material.baseColor);
    material.metallic = saturate(material.metallic);
    material.roughness = clamp(material.roughness, 0.03, 0.95);
    material.occlusion = saturate(material.occlusion);
    material.transmission = saturate(material.transmission);
    material.ior = clamp(material.ior, 1.0, 3.0);
    material.thickness = max(material.thickness, 0.0);
    material.attenuationDistance = max(material.attenuationDistance, 0.0);
    material.attenuationColor = max(material.attenuationColor, float3(0.001, 0.001, 0.001));
    material.diffuseTransmission = saturate(material.diffuseTransmission);
    material.diffuseTransmissionColor = max(material.diffuseTransmissionColor, float3(0.001, 0.001, 0.001));
    return material;
}

float3 fresnelF0(MaterialSample material)
{
    float f0 = (material.ior - 1.0) / (material.ior + 1.0);
    f0 *= f0;
    return lerp(float3(f0, f0, f0), material.baseColor, material.metallic);
}

float luminanceMax(float3 value)
{
    return max(value.x, max(value.y, value.z));
}

bool isTransmissive(MaterialSample material)
{
    return material.transmission > 0.01;
}

bool isDiffuseTransmissive(MaterialSample material)
{
    return material.diffuseTransmission > 0.01;
}

float fresnelDielectric(MaterialSample material, float cosTheta)
{
    float f0 = (material.ior - 1.0) / (material.ior + 1.0);
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

float3 volumeTransmittance(MaterialSample material, float distance)
{
    float exponent = 0.0;
    if (material.attenuationDistance > 0.0001) {
        exponent = max(distance, 0.0) / material.attenuationDistance;
    } else if (material.thickness > 0.0001) {
        exponent = saturate(max(distance, material.thickness) * 2.0);
    }

    float3 attenuation = powColor(material.attenuationColor, exponent);
    float3 baseTint = lerp(float3(1.0, 1.0, 1.0), max(material.baseColor, float3(0.001, 0.001, 0.001)), material.transmission * 0.35);
    return saturate(attenuation * baseTint);
}

float3 offsetNormalForDirection(float3 direction, float3 geometryNormal)
{
    return dot(direction, geometryNormal) > 0.0 ? geometryNormal : -geometryNormal;
}

bool sampleGlassEvent(
    Ray ray,
    HitInfo hit,
    MaterialSample material,
    inout uint seed,
    bool forceTransmission,
    out float3 direction,
    out float3 weight)
{
    float eta = hit.frontFace ? 1.0 / max(material.ior, 1.001) : max(material.ior, 1.001);
    float3 refractedDirection = float3(0.0, 0.0, 0.0);
    bool canRefract = refractDirection(ray.direction, material.normal, eta, refractedDirection);
    float cosTheta = saturate(dot(-ray.direction, material.normal));
    float reflectProbability = canRefract ? fresnelDielectric(material, cosTheta) : 1.0;

    if (!forceTransmission && rand(seed) < reflectProbability) {
        direction = safeNormalize(reflect(ray.direction, material.normal), material.normal);
        weight = float3(1.0, 1.0, 1.0);
        return true;
    }

    if (canRefract) {
        direction = safeNormalize(refractedDirection, ray.direction);
        weight = volumeTransmittance(material, max(material.thickness, 0.001)) * material.transmission;
        return true;
    }

    direction = safeNormalize(reflect(ray.direction, material.normal), material.normal);
    weight = float3(1.0, 1.0, 1.0);
    return true;
}

float3 traceVisibility(float3 origin, float3 direction, float maxDistance, ChessNrdPush push)
{
    float3 visibility = float3(1.0, 1.0, 1.0);
    Ray ray;
    ray.origin = origin;
    ray.direction = direction;
    float remainingDistance = max(maxDistance, 0.001);

    for (uint depth = 0u; depth < 8u; ++depth) {
        HitInfo hit;
        if (!traceClosestSecondary(ray, push, hit) || hit.t >= remainingDistance) {
            return visibility;
        }

        MaterialSample material = sampleMaterial(hit);
        if (!isTransmissive(material)) {
            return float3(0.0, 0.0, 0.0);
        }

        visibility *= volumeTransmittance(material, max(material.thickness, hit.t * 0.1)) * material.transmission;
        if (luminanceMax(visibility) < 0.01) {
            return float3(0.0, 0.0, 0.0);
        }

        remainingDistance -= hit.t;
        ray.origin = offsetRay(hit.position, offsetNormalForDirection(direction, hit.geometryNormal));
    }

    return visibility;
}

float3 evaluateDirect(HitInfo hit, MaterialSample material, float3 viewDirection, ChessNrdPush push)
{
    float3 lightDirection = sunDirection();
    float nDotL = saturate(dot(material.normal, lightDirection));
    float3 direct = float3(0.0, 0.0, 0.0);
    if (nDotL > 0.0) {
        float3 visibility = traceVisibility(
            offsetRay(hit.position, offsetNormalForDirection(lightDirection, hit.geometryNormal)),
            lightDirection,
            10000.0,
            push);
        if (luminanceMax(visibility) > 0.0) {
            float3 halfVector = safeNormalize(lightDirection + viewDirection, material.normal);
            float nDotH = saturate(dot(material.normal, halfVector));
            float specularPower = lerp(128.0, 10.0, material.roughness);
            float diffuseReflectance = (1.0 - material.metallic) * (1.0 - material.transmission) * (1.0 - material.diffuseTransmission);
            float3 diffuse = material.baseColor * diffuseReflectance * (nDotL / kPi);
            float3 specular = fresnelF0(material) * pow(nDotH, specularPower) * nDotL;
            direct += (diffuse + specular) * float3(7.2, 6.5, 5.3) * visibility;
        }
    }
    if (isDiffuseTransmissive(material)) {
        float backNDotL = saturate(dot(-material.normal, lightDirection));
        if (backNDotL > 0.0) {
            float3 visibility = traceVisibility(
                offsetRay(hit.position, offsetNormalForDirection(lightDirection, hit.geometryNormal)),
                lightDirection,
                10000.0,
                push);
            float3 transmittance = volumeTransmittance(material, max(material.thickness, 0.01));
            float3 transmittedDiffuse =
                material.baseColor *
                material.diffuseTransmissionColor *
                material.diffuseTransmission *
                transmittance *
                (backNDotL / kPi);
            direct += transmittedDiffuse * float3(7.2, 6.5, 5.3) * visibility;
        }
    }
    float ambient = (0.10 + 0.08 * saturate(material.normal.y)) * material.occlusion * (1.0 - material.transmission * 0.75);
    direct += material.baseColor * ambient;
    direct += material.baseColor * material.diffuseTransmissionColor * material.diffuseTransmission * material.occlusion * 0.08;
    return direct + material.emissive;
}

float3 traceBsdfPath(Ray ray, ChessNrdPush push, inout uint seed, out float firstHitDistance)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    firstHitDistance = 500000.0;
    uint maxDepth = clamp(push.maxDepth, 1u, 12u);

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        if (!traceClosestSecondary(ray, push, hit)) {
            radiance += throughput * skyRadiance(ray.direction);
            break;
        }

        if (depth == 0u) {
            firstHitDistance = max(hit.t, 0.0);
        }

        MaterialSample material = sampleMaterial(hit);
        float3 viewDirection = -ray.direction;
        radiance += throughput * evaluateDirect(hit, material, viewDirection, push);

        float3 nextDirection = material.normal;
        float3 nextWeight = float3(1.0, 1.0, 1.0);
        if (isTransmissive(material)) {
            sampleGlassEvent(ray, hit, material, seed, false, nextDirection, nextWeight);
        } else {
            float3 diffuseDirection = sampleCosineHemisphere(material.normal, seed);
            float3 reflectedDirection = safeNormalize(reflect(ray.direction, material.normal), material.normal);
            float specularChance = saturate(0.08 + material.metallic * 0.72 + (1.0 - material.roughness) * 0.18);
            if (rand(seed) < specularChance) {
                float glossMix = material.roughness * material.roughness;
                nextDirection = safeNormalize(lerp(reflectedDirection, diffuseDirection, glossMix), reflectedDirection);
                nextWeight = lerp(fresnelF0(material), material.baseColor, material.metallic * 0.35);
            } else if (rand(seed) < material.diffuseTransmission * (1.0 - material.metallic)) {
                nextDirection = sampleSubsurfaceDiffuseDirection(material.normal, seed);
                nextWeight =
                    material.baseColor *
                    material.diffuseTransmissionColor *
                    material.diffuseTransmission *
                    volumeTransmittance(material, max(material.thickness, 0.01)) *
                    material.occlusion;
            } else {
                nextDirection = diffuseDirection;
                nextWeight =
                    material.baseColor *
                    (1.0 - material.metallic) *
                    (1.0 - material.diffuseTransmission) *
                    material.occlusion;
            }
        }

        throughput *= max(nextWeight, float3(0.0, 0.0, 0.0));
        if (luminanceMax(throughput) < 0.001) {
            break;
        }

        ray.origin = offsetRay(hit.position, offsetNormalForDirection(nextDirection, hit.geometryNormal));
        ray.direction = nextDirection;

        if (depth >= 2u) {
            float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
            if (rand(seed) > continueProbability) {
                break;
            }
            throughput /= max(continueProbability, 0.001);
        }
    }

    return radiance;
}

float3 tonemap(float3 color)
{
    color = color / (color + float3(1.0, 1.0, 1.0));
    return pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void chessNrdPathTraceMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                           uniform ChessNrdPush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    uint seed = hash(pixel.x + pixel.y * max(push.width, 1u) + (push.frameIndex + 1u) * 0x9e3779b9u);
    float2 jitter = push.samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed));
    Ray primaryRay;
    primaryRay.origin = rayOriginFromPixel(float2(pixel) + jitter, push);
    primaryRay.direction = rayDirectionFromPixel(float2(pixel) + jitter, push);

    HitInfo primaryHit;
    MaterialSample material;
    Ray primarySurfaceRay = primaryRay;
    float3 primaryThroughput = float3(1.0, 1.0, 1.0);
    float3 primaryPrefix = float3(0.0, 0.0, 0.0);
    float viewZ = 0.0;
    bool resolvedPrimary = false;

    for (uint psrDepth = 0u; psrDepth < 5u; ++psrDepth) {
        HitInfo candidateHit;
        bool foundCandidate = false;
        if (psrDepth == 0u) {
            foundCandidate = traceClosest(primarySurfaceRay, push, candidateHit);
        } else {
            foundCandidate = traceClosestSecondary(primarySurfaceRay, push, candidateHit);
        }
        if (!foundCandidate) {
            float3 sky = primaryPrefix + primaryThroughput * skyRadiance(primarySurfaceRay.direction);
            gNoisyDiffuse[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gNoisySpecular[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gNormalRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(float3(0.0, 1.0, 0.0), 1.0, 0.0);
            gMotionVectors[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gViewZ[pixel] = push.clipOrtho.y + 1.0;
            gDirectLighting[pixel] = float4(sky, 0.0);
            gBaseColorMetalness[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gNoisyColor[pixel] = float4(tonemap(sky), 1.0);
            return;
        }

        MaterialSample candidateMaterial = sampleMaterial(candidateHit);
        viewZ += max(candidateHit.t, 0.0);
        if (!isTransmissive(candidateMaterial) || psrDepth == 4u) {
            primaryHit = candidateHit;
            material = candidateMaterial;
            resolvedPrimary = true;
            break;
        }

        primaryPrefix += primaryThroughput * evaluateDirect(candidateHit, candidateMaterial, -primarySurfaceRay.direction, push) * 0.25;

        float3 nextDirection = candidateMaterial.normal;
        float3 nextWeight = float3(1.0, 1.0, 1.0);
        sampleGlassEvent(primarySurfaceRay, candidateHit, candidateMaterial, seed, true, nextDirection, nextWeight);
        primaryThroughput *= max(nextWeight, float3(0.0, 0.0, 0.0));
        if (luminanceMax(primaryThroughput) < 0.001) {
            float3 sky = primaryPrefix;
            gNoisyDiffuse[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gNoisySpecular[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gNormalRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(float3(0.0, 1.0, 0.0), 1.0, 0.0);
            gMotionVectors[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gViewZ[pixel] = push.clipOrtho.y + 1.0;
            gDirectLighting[pixel] = float4(sky, 0.0);
            gBaseColorMetalness[pixel] = float4(0.0, 0.0, 0.0, 0.0);
            gNoisyColor[pixel] = float4(tonemap(sky), 1.0);
            return;
        }

        primarySurfaceRay.origin = offsetRay(candidateHit.position, offsetNormalForDirection(nextDirection, candidateHit.geometryNormal));
        primarySurfaceRay.direction = nextDirection;
    }

    if (!resolvedPrimary) {
        float3 sky = primaryPrefix;
        gNoisyDiffuse[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gNoisySpecular[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gNormalRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(float3(0.0, 1.0, 0.0), 1.0, 0.0);
        gMotionVectors[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gViewZ[pixel] = push.clipOrtho.y + 1.0;
        gDirectLighting[pixel] = float4(sky, 0.0);
        gBaseColorMetalness[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gNoisyColor[pixel] = float4(tonemap(sky), 1.0);
        return;
    }

    float3 viewDirection = -primarySurfaceRay.direction;
    viewZ = max(viewZ, 0.0001);
    uint samples = clamp(push.samples, 1u, 16u);

    float3 direct = primaryPrefix + primaryThroughput * evaluateDirect(primaryHit, material, viewDirection, push);
    float3 diffuseAccum = float3(0.0, 0.0, 0.0);
    float3 specularAccum = float3(0.0, 0.0, 0.0);
    float diffuseHitDistance = 0.0;
    float specularHitDistance = 0.0;

    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        bool sampleDiffuseTransmission =
            isDiffuseTransmissive(material) &&
            rand(seed) < material.diffuseTransmission * (1.0 - material.metallic);
        float3 diffuseDirection = sampleDiffuseTransmission
            ? sampleSubsurfaceDiffuseDirection(material.normal, seed)
            : sampleCosineHemisphere(material.normal, seed);
        Ray diffuseRay;
        diffuseRay.origin = offsetRay(primaryHit.position, offsetNormalForDirection(diffuseDirection, primaryHit.geometryNormal));
        diffuseRay.direction = diffuseDirection;
        float diffHit = 0.0;
        float3 diffRadiance = traceBsdfPath(diffuseRay, push, seed, diffHit);
        float3 diffuseFactor = material.baseColor * (1.0 - material.metallic) * (1.0 - material.transmission) * material.occlusion;
        if (sampleDiffuseTransmission) {
            diffuseFactor *=
                material.diffuseTransmissionColor *
                material.diffuseTransmission *
                volumeTransmittance(material, max(material.thickness, 0.01));
        } else {
            diffuseFactor *= (1.0 - material.diffuseTransmission);
        }
        diffuseAccum += primaryThroughput * diffRadiance * diffuseFactor;
        diffuseHitDistance += diffHit;

        float3 reflectedDirection = reflect(primarySurfaceRay.direction, material.normal);
        float3 specularDirection = safeNormalize(
            lerp(reflectedDirection, diffuseDirection, material.roughness * material.roughness),
            reflectedDirection);
        Ray specularRay;
        specularRay.origin = offsetRay(primaryHit.position, offsetNormalForDirection(specularDirection, primaryHit.geometryNormal));
        specularRay.direction = specularDirection;
        float specHit = 0.0;
        float3 specRadiance = traceBsdfPath(specularRay, push, seed, specHit);
        float3 specFactor = lerp(fresnelF0(material), material.baseColor, material.metallic * 0.25);
        specularAccum += primaryThroughput * specRadiance * specFactor;
        specularHitDistance += specHit;
    }

    diffuseAccum /= float(samples);
    specularAccum /= float(samples);
    diffuseHitDistance /= float(samples);
    specularHitDistance /= float(samples);

    float3 diffDemodulate = material.baseColor * 0.99 + 0.01;
    float3 specDemodulate = fresnelF0(material) * 0.99 + 0.01;
    float3 diffuseRadiance = diffuseAccum / diffDemodulate;
    float3 specularRadiance = specularAccum / specDemodulate;

    float4 packedDiffuse;
    float4 packedSpecular;
    if (push.denoiserMode == kNrdReblur) {
        float diffNormHit = REBLUR_FrontEnd_GetNormHitDist(diffuseHitDistance, viewZ, kDiffHitDistParams, 1.0);
        float specNormHit = REBLUR_FrontEnd_GetNormHitDist(specularHitDistance, viewZ, kSpecHitDistParams, material.roughness);
        packedDiffuse = REBLUR_FrontEnd_PackRadianceAndNormHitDist(diffuseRadiance, diffNormHit, true);
        packedSpecular = REBLUR_FrontEnd_PackRadianceAndNormHitDist(specularRadiance, specNormHit, true);
    } else if (push.denoiserMode == kNrdRelax) {
        packedDiffuse = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseRadiance, diffuseHitDistance, true);
        packedSpecular = RELAX_FrontEnd_PackRadianceAndHitDist(specularRadiance, specularHitDistance, true);
    } else {
        packedDiffuse = float4(diffuseRadiance, diffuseHitDistance);
        packedSpecular = float4(specularRadiance, specularHitDistance);
    }

    gNoisyDiffuse[pixel] = packedDiffuse;
    gNoisySpecular[pixel] = packedSpecular;
    gNormalRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(material.normal, material.roughness, 0.0);
    gMotionVectors[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    gViewZ[pixel] = viewZ;
    gDirectLighting[pixel] = float4(direct, 1.0);
    gBaseColorMetalness[pixel] = float4(material.baseColor, material.metallic);
    gNoisyColor[pixel] = float4(tonemap(direct + diffuseAccum + specularAccum), 1.0);
}
