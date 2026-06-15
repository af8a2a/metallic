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

struct PathTraceMaterial
{
    float4 baseColor;
    float4 emissive;
    float4 params;
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
    float2 texcoord;
    uint materialIndex;
};

struct MaterialSample
{
    float3 baseColor;
    float metallic;
    float roughness;
    float3 emissive;
};

static const uint kNrdReblur = 0u;
static const uint kNrdRelax = 1u;
static const uint kNrdReference = 2u;
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

bool traceClosest(Ray ray, ChessNrdPush push, out HitInfo hit)
{
    hit = HitInfo();
    hit.t = 0.0;
    hit.position = ray.origin;
    hit.normal = float3(0.0, 1.0, 0.0);
    hit.geometryNormal = hit.normal;
    hit.texcoord = float2(0.0, 0.0);
    hit.materialIndex = 0u;

    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    rayDesc.TMin = max(push.clipOrtho.x, 0.0001);
    rayDesc.TMax = max(push.clipOrtho.y, rayDesc.TMin + 0.0001);

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
    float2 localTexcoord = v0.texcoord.xy * weights.x + v1.texcoord.xy * weights.y + v2.texcoord.xy * weights.z;

    float4x3 objectToWorld = query.CommittedObjectToWorld4x3();
    float4x3 worldToObject = query.CommittedWorldToObject4x3();
    hit.position = float3(mul(float4(localPosition, 1.0), objectToWorld));
    hit.geometryNormal = safeNormalize(float3(mul(worldToObject, localGeometryNormal).xyz), localGeometryNormal);
    hit.normal = safeNormalize(float3(mul(worldToObject, localNormal).xyz), hit.geometryNormal);
    hit.texcoord = localTexcoord;

    float3 view = -ray.direction;
    if (dot(hit.geometryNormal, view) < 0.0) {
        hit.geometryNormal = -hit.geometryNormal;
    }
    if (dot(hit.normal, hit.geometryNormal) < 0.0) {
        hit.normal = -hit.normal;
    }

    hit.t = query.CommittedRayT();
    hit.materialIndex = instance.materialIndex;
    return true;
}

MaterialSample sampleMaterial(HitInfo hit)
{
    PathTraceMaterial source = gMaterials[hit.materialIndex];
    MaterialSample material;
    material.baseColor = saturate(source.baseColor.rgb);
    material.metallic = saturate(source.params.x);
    material.roughness = clamp(source.params.y, 0.03, 0.95);
    material.emissive = max(source.emissive.rgb, float3(0.0, 0.0, 0.0));

    uint kind = uint(round(source.params.w));
    if (kind == 1u) {
        float squareSize = 0.0625;
        float2 board = (hit.position.xz + float2(0.255, 0.255)) / squareSize;
        uint square = (uint(floor(board.x)) + uint(floor(board.y))) & 1u;
        float edge = smoothstep(0.025, 0.08, min(frac(board.x), min(frac(board.y), min(1.0 - frac(board.x), 1.0 - frac(board.y)))));
        float3 lightSquare = float3(0.86, 0.78, 0.58);
        float3 darkSquare = float3(0.18, 0.24, 0.22);
        material.baseColor = lerp(lerp(lightSquare, darkSquare, float(square)), float3(0.08, 0.08, 0.075), 1.0 - edge);
        material.metallic = 0.0;
        material.roughness = 0.42;
    } else if (kind == 2u) {
        material.baseColor = max(material.baseColor, float3(0.92, 0.86, 0.68));
        material.metallic = max(material.metallic, 0.05);
        material.roughness = min(material.roughness, 0.34);
    } else if (kind == 3u) {
        material.baseColor = min(max(material.baseColor, float3(0.09, 0.21, 0.19)), float3(0.22, 0.42, 0.36));
        material.metallic = max(material.metallic, 0.08);
        material.roughness = min(material.roughness, 0.36);
    }
    return material;
}

float3 fresnelF0(MaterialSample material)
{
    return lerp(float3(0.04, 0.04, 0.04), material.baseColor, material.metallic);
}

float3 evaluateDirect(HitInfo hit, MaterialSample material, float3 viewDirection)
{
    float3 lightDirection = sunDirection();
    float nDotL = saturate(dot(hit.normal, lightDirection));
    float3 direct = float3(0.0, 0.0, 0.0);
    if (nDotL > 0.0) {
        bool occluded = traceShadow(offsetRay(hit.position, hit.geometryNormal), lightDirection, 10000.0);
        if (!occluded) {
            float3 halfVector = safeNormalize(lightDirection + viewDirection, hit.normal);
            float nDotH = saturate(dot(hit.normal, halfVector));
            float specularPower = lerp(128.0, 10.0, material.roughness);
            float3 diffuse = material.baseColor * (1.0 - material.metallic) * (nDotL / kPi);
            float3 specular = fresnelF0(material) * pow(nDotH, specularPower) * nDotL;
            direct += (diffuse + specular) * float3(7.2, 6.5, 5.3);
        }
    }
    float ambient = 0.10 + 0.08 * saturate(hit.normal.y);
    direct += material.baseColor * ambient;
    return direct + material.emissive;
}

float3 bounceRadiance(Ray ray, ChessNrdPush push, inout uint seed, out float hitDistance)
{
    HitInfo hit;
    if (!traceClosest(ray, push, hit)) {
        hitDistance = 500000.0;
        return skyRadiance(ray.direction);
    }

    MaterialSample material = sampleMaterial(hit);
    float3 viewDirection = -ray.direction;
    hitDistance = max(hit.t, 0.0);
    return evaluateDirect(hit, material, viewDirection) + material.emissive;
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
    if (!traceClosest(primaryRay, push, primaryHit)) {
        float3 sky = skyRadiance(primaryRay.direction);
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

    MaterialSample material = sampleMaterial(primaryHit);
    float3 viewDirection = -primaryRay.direction;
    float viewZ = max(primaryHit.t, 0.0001);
    uint samples = clamp(push.samples, 1u, 16u);

    float3 direct = evaluateDirect(primaryHit, material, viewDirection);
    float3 diffuseAccum = float3(0.0, 0.0, 0.0);
    float3 specularAccum = float3(0.0, 0.0, 0.0);
    float diffuseHitDistance = 0.0;
    float specularHitDistance = 0.0;

    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float3 diffuseDirection = sampleCosineHemisphere(primaryHit.normal, seed);
        Ray diffuseRay;
        diffuseRay.origin = offsetRay(primaryHit.position, primaryHit.geometryNormal);
        diffuseRay.direction = diffuseDirection;
        float diffHit = 0.0;
        float3 diffRadiance = bounceRadiance(diffuseRay, push, seed, diffHit);
        diffuseAccum += diffRadiance * material.baseColor * (1.0 - material.metallic);
        diffuseHitDistance += diffHit;

        float3 reflectedDirection = reflect(primaryRay.direction, primaryHit.normal);
        float3 specularDirection = safeNormalize(lerp(reflectedDirection, diffuseDirection, material.roughness * material.roughness), reflectedDirection);
        Ray specularRay;
        specularRay.origin = offsetRay(primaryHit.position, dot(specularDirection, primaryHit.geometryNormal) > 0.0 ? primaryHit.geometryNormal : -primaryHit.geometryNormal);
        specularRay.direction = specularDirection;
        float specHit = 0.0;
        float3 specRadiance = bounceRadiance(specularRay, push, seed, specHit);
        float3 specFactor = lerp(fresnelF0(material), material.baseColor, material.metallic * 0.25);
        specularAccum += specRadiance * specFactor;
        specularHitDistance += specHit;
    }

    diffuseAccum /= float(samples);
    specularAccum /= float(samples);
    diffuseHitDistance /= float(samples);
    specularHitDistance /= float(samples);

    float3 diffDemodulate = material.baseColor * 0.99 + 0.01;
    float3 specDemodulate = lerp(float3(0.04, 0.04, 0.04), material.baseColor, material.metallic) * 0.99 + 0.01;
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
    gNormalRoughness[pixel] = NRD_FrontEnd_PackNormalAndRoughness(primaryHit.normal, material.roughness, 0.0);
    gMotionVectors[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    gViewZ[pixel] = viewZ;
    gDirectLighting[pixel] = float4(direct, 1.0);
    gBaseColorMetalness[pixel] = float4(material.baseColor, material.metallic);
    gNoisyColor[pixel] = float4(tonemap(direct + diffuseAccum + specularAccum), 1.0);
}
