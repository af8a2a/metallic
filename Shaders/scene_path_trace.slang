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
    PathTraceTextureInfo baseColorTexture;
    PathTraceTextureInfo metallicRoughnessTexture;
    PathTraceTextureInfo normalTexture;
    PathTraceTextureInfo occlusionTexture;
    PathTraceTextureInfo emissiveTexture;
    PathTraceTextureInfo transmissionTexture;
    PathTraceTextureInfo thicknessTexture;
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

struct ScenePathTracePush
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
    uint accumulationFrame;
    uint hasHistory;
    uint enableAccumulation;
    uint padding;
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
    uint materialIndex;
    bool frontFace;
};

static const float kPi = 3.14159265358979323846;

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

float3 offsetRay(float3 position, float3 normal)
{
    return position + normal * 0.001;
}

float3 skyRadiance(float3 direction)
{
    float t = saturate(direction.y * 0.5 + 0.5);
    float3 horizon = float3(0.62, 0.70, 0.78);
    float3 zenith = float3(0.12, 0.24, 0.48);
    float sun = pow(saturate(dot(direction, safeNormalize(float3(0.35, 0.88, 0.22), float3(0.0, 1.0, 0.0)))), 256.0);
    return lerp(horizon, zenith, t) * 0.45 + sun * float3(7.0, 6.4, 5.3);
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

bool traceClosest(Ray ray, ScenePathTracePush push, out HitInfo hit)
{
    hit = HitInfo();
    hit.t = 0.0;
    hit.position = ray.origin;
    hit.normal = float3(0.0, 1.0, 0.0);
    hit.geometryNormal = hit.normal;
    hit.materialIndex = 0u;
    hit.frontFace = true;

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

    float4x3 objectToWorld = query.CommittedObjectToWorld4x3();
    float4x3 worldToObject = query.CommittedWorldToObject4x3();
    hit.position = float3(mul(float4(localPosition, 1.0), objectToWorld));
    hit.geometryNormal = safeNormalize(float3(mul(worldToObject, localGeometryNormal).xyz), localGeometryNormal);
    hit.normal = safeNormalize(float3(mul(worldToObject, localNormal).xyz), hit.geometryNormal);

    float3 view = -ray.direction;
    hit.frontFace = dot(hit.geometryNormal, view) >= 0.0;
    if (!hit.frontFace) {
        hit.geometryNormal = -hit.geometryNormal;
    }
    if (dot(hit.normal, hit.geometryNormal) < 0.0) {
        hit.normal = -hit.normal;
    }

    hit.t = query.CommittedRayT();
    hit.materialIndex = instance.materialIndex;
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
    } else if (thickness > 0.0001) {
        exponent = saturate(max(distance, thickness) * 2.0);
    }

    float3 attenuation = powColor(max(attenuationColor, float3(0.001, 0.001, 0.001)), exponent);
    float3 baseTint = lerp(float3(1.0, 1.0, 1.0), max(baseColor, float3(0.001, 0.001, 0.001)), transmission * 0.35);
    return saturate(attenuation * baseTint);
}

float3 pathTrace(Ray ray, ScenePathTracePush push, inout uint seed)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    uint maxDepth = clamp(push.maxDepth, 1u, 12u);

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        if (!traceClosest(ray, push, hit)) {
            radiance += throughput * skyRadiance(ray.direction);
            break;
        }

        PathTraceMaterial material = gMaterials[hit.materialIndex];
        float3 baseColor = saturate(material.baseColor.rgb);
        float metallic = saturate(material.params.x);
        float roughness = clamp(material.params.y, 0.02, 1.0);
        float transmission = saturate(material.glassParams.x);
        float ior = clamp(material.glassParams.y, 1.0, 3.0);
        float thickness = max(material.glassParams.z, 0.0);
        float attenuationDistance = max(material.glassParams.w, 0.0);
        float3 attenuationColor = max(saturate(material.attenuationColor.rgb), float3(0.001, 0.001, 0.001));
        float dielectricF0 = (ior - 1.0) / (ior + 1.0);
        dielectricF0 *= dielectricF0;
        float3 f0 = lerp(float3(dielectricF0, dielectricF0, dielectricF0), baseColor, metallic);

        radiance += throughput * max(material.emissive.rgb, float3(0.0, 0.0, 0.0));

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
            ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);
            continue;
        }

        float3 sunDirection = safeNormalize(float3(0.35, 0.88, 0.22), float3(0.0, 1.0, 0.0));
        float nDotL = saturate(dot(hit.normal, sunDirection));
        if (nDotL > 0.0) {
            bool occluded = traceShadow(offsetRay(hit.position, hit.geometryNormal), sunDirection, 10000.0);
            if (!occluded) {
                float3 viewDirection = -ray.direction;
                float3 halfVector = safeNormalize(sunDirection + viewDirection, hit.normal);
                float nDotH = saturate(dot(hit.normal, halfVector));
                float specularPower = lerp(96.0, 8.0, roughness);
                float3 diffuse = baseColor * (1.0 - metallic) * (nDotL / kPi);
                float3 specular = f0 * pow(nDotH, specularPower) * nDotL;
                radiance += throughput * (diffuse + specular) * float3(5.2, 4.8, 4.2);
            }
        }

        float3 diffuseDirection = sampleCosineHemisphere(hit.normal, seed);
        float3 reflectedDirection = reflect(ray.direction, hit.normal);
        float specularChance = saturate(0.08 + metallic * 0.72 + (1.0 - roughness) * 0.18);
        if (rand(seed) < specularChance) {
            float glossMix = roughness * roughness;
            ray.direction = safeNormalize(lerp(reflectedDirection, diffuseDirection, glossMix), reflectedDirection);
            throughput *= lerp(f0, baseColor, metallic * 0.35);
        } else {
            ray.direction = diffuseDirection;
            throughput *= baseColor * (1.0 - metallic);
        }
        ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);

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

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                        uniform ScenePathTracePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    uint samples = clamp(push.samples, 1u, 16u);
    uint seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.accumulationFrame + 1u) * 0x9e3779b9u);
    float3 color = float3(0.0, 0.0, 0.0);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float2 jitter = samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed));
        float2 samplePixel = float2(pixel) + jitter;

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        color += pathTrace(ray, push, seed);
    }
    color /= float(samples);

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
