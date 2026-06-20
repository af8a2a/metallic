[[vk::binding(0, 0)]]
RaytracingAccelerationStructure gScene;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gOutput;

struct MaterialVisualizeVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 texcoord;
};

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

struct MaterialVisualizeTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint padding0;
    uint padding1;
    float4 transform0;
    float4 transform1;
};

struct MaterialVisualizeMaterial
{
    float4 baseColor;
    float4 emissive;
    float4 params;
    float4 textureParams;
    float4 glassParams;
    float4 attenuationColor;
    float4 diffuseTransmission;
    MaterialVisualizeTextureInfo baseColorTexture;
    MaterialVisualizeTextureInfo metallicRoughnessTexture;
    MaterialVisualizeTextureInfo normalTexture;
    MaterialVisualizeTextureInfo occlusionTexture;
    MaterialVisualizeTextureInfo emissiveTexture;
    MaterialVisualizeTextureInfo transmissionTexture;
    MaterialVisualizeTextureInfo thicknessTexture;
    MaterialVisualizeTextureInfo diffuseTransmissionTexture;
    MaterialVisualizeTextureInfo diffuseTransmissionColorTexture;
};

[[vk::binding(2, 0)]]
StructuredBuffer<MaterialVisualizeVertex> gVertices;

[[vk::binding(3, 0)]]
StructuredBuffer<uint> gIndices;

[[vk::binding(4, 0)]]
StructuredBuffer<MaterialVisualizePrimitive> gPrimitives;

[[vk::binding(5, 0)]]
StructuredBuffer<MaterialVisualizeInstance> gInstances;

[[vk::binding(6, 0)]]
StructuredBuffer<MaterialVisualizeMaterial> gMaterials;

[[vk::binding(7, 0)]]
Texture2D<float4> gMaterialTextures[256];

static const uint kMaxMaterialTextures = 256u;
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
static const uint kModeNrdNormalRoughness = 11u;
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
    uint padding0;
    uint padding1;
    uint padding2;
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
    float4 tangent;
    float3 bitangent;
    float2 texcoord;
    uint materialIndex;
    bool frontFace;
};

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

bool hasMaterialTexture(MaterialVisualizeTextureInfo textureInfo, SceneMaterialVisualizationPush push)
{
    return textureInfo.textureIndex != kInvalidMaterialTextureIndex &&
        textureInfo.textureIndex < push.materialTextureCount &&
        textureInfo.textureIndex < kMaxMaterialTextures;
}

float2 transformMaterialUv(MaterialVisualizeTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

float4 loadMaterialTexture(
    MaterialVisualizeTextureInfo textureInfo,
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
    MaterialVisualizeInstance instance = gInstances[instanceIndex];
    MaterialVisualizePrimitive primitive = gPrimitives[instance.primitiveIndex];
    uint indexOffset = primitive.firstIndex + triangleIndex * 3u;
    uint i0 = primitive.firstVertex + gIndices[indexOffset + 0u];
    uint i1 = primitive.firstVertex + gIndices[indexOffset + 1u];
    uint i2 = primitive.firstVertex + gIndices[indexOffset + 2u];
    float3 weights = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    return gVertices[i0].texcoord.xy * weights.x +
        gVertices[i1].texcoord.xy * weights.y +
        gVertices[i2].texcoord.xy * weights.z;
}

bool alphaMaskAcceptsHit(uint instanceIndex, uint triangleIndex, float2 bary, SceneMaterialVisualizationPush push)
{
    MaterialVisualizeInstance instance = gInstances[instanceIndex];
    MaterialVisualizeMaterial material = gMaterials[instance.materialIndex];
    if (abs(material.textureParams.w - 1.0) > 0.25) {
        return true;
    }

    float2 uv = triangleTexcoord(instanceIndex, triangleIndex, bary);
    float alpha = material.baseColor.a * loadMaterialTexture(
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

bool traceClosest(Ray ray, SceneMaterialVisualizationPush push, out HitInfo hit)
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
    MaterialVisualizeInstance instance = gInstances[instanceIndex];
    MaterialVisualizePrimitive primitive = gPrimitives[instance.primitiveIndex];
    uint triangleIndex = query.CommittedPrimitiveIndex();
    uint indexOffset = primitive.firstIndex + triangleIndex * 3u;

    uint i0 = primitive.firstVertex + gIndices[indexOffset + 0u];
    uint i1 = primitive.firstVertex + gIndices[indexOffset + 1u];
    uint i2 = primitive.firstVertex + gIndices[indexOffset + 2u];
    MaterialVisualizeVertex v0 = gVertices[i0];
    MaterialVisualizeVertex v1 = gVertices[i1];
    MaterialVisualizeVertex v2 = gVertices[i2];

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
    MaterialVisualizeMaterial material = gMaterials[hit.materialIndex];
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
        return saturate(material.baseColor.rgb * srgbToLinear(baseColorSample.rgb));
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
    if (push.mode == kModeNrdNormalRoughness) {
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
void sceneMaterialVisualizeMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                                uniform SceneMaterialVisualizationPush push)
{
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