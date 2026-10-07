import Atmosphere;
#include "SceneSurface.slang"
#include "SceneFiberPath.slang"

// ---------------------------------------------------------------------------
// Radiance cache integration (reference: RTXGI v2 SDK)
//
// The same entry point is compiled into several permutations:
//   (no defines)      plain path tracing
//   SHARC_UPDATE=1    sparse SHaRC update pass (populates the hash-grid cache)
//   SHARC_QUERY=1     full-resolution pass with SHaRC early termination
//   NRC_UPDATE=1      NRC training pass at reduced resolution
//   NRC_QUERY=1       full-resolution pass creating NRC queries; outputs
//                     linear HDR that the NRC resolve adds onto before the
//                     separate tonemap pass
// ---------------------------------------------------------------------------
#if defined(SHARC_UPDATE) || defined(SHARC_QUERY)
#define METALLIC_PT_SHARC 1
#else
#define METALLIC_PT_SHARC 0
#endif

#if defined(NRC_UPDATE) || defined(NRC_QUERY)
#define METALLIC_PT_NRC 1
#else
#define METALLIC_PT_NRC 0
#endif

#if METALLIC_PT_SHARC
#include "../../ThirdParty/RadianceCache/Sharc/SharcCommon.h"
#endif

#include "../../ThirdParty/RadianceCache/Nrc/Nrc.hlsli"

// Cache mode pushed through the per-frame parameter buffer.
#define METALLIC_CACHE_MODE_OFF 0
#define METALLIC_CACHE_MODE_SHARC 1
#define METALLIC_CACHE_MODE_NRC 2

// Per-frame radiance-cache parameters (binding 20). Layout must match
// ScenePathTraceCacheParams in BuiltinPassCommon.h byte for byte.
struct ScenePathTraceCacheParams
{
    float4 sharcCameraPosition;
    float4 sharcCameraPositionPrev;
    float sharcSceneScale;
    uint sharcEntriesNum;
    uint sharcAccumulationFrameNum;
    uint sharcStaleFrameNumMax;
    uint frameIndex;
    uint cacheMode;
    uint sharcUpdateStride;
    uint width;
    uint height;
    uint trainingWidth;
    uint trainingHeight;
    NrcConstants nrc;
};

// Match the tightly packed CPU constants, including NRC at byte offset 76.
typedef StructuredBuffer<ScenePathTraceCacheParams, ScalarDataLayout> ScenePathTraceCacheBuffer;
#define gCacheParams resolveBuffer<ScenePathTraceCacheBuffer>(getResourceParameters<SceneResourceParameters>().cacheParams)

#if METALLIC_PT_SHARC
#define gSharcHashEntries resolveBuffer<RWStructuredBuffer<uint64_t>>(getResourceParameters<SceneResourceParameters>().sharcHashEntries)

#define gSharcAccumulation resolveBuffer<RWStructuredBuffer<SharcAccumulationData>>(getResourceParameters<SceneResourceParameters>().sharcAccumulation)

#define gSharcResolved resolveBuffer<RWStructuredBuffer<SharcPackedData>>(getResourceParameters<SceneResourceParameters>().sharcResolved)
#endif

#if METALLIC_PT_NRC
#define gNrcQueryPathInfo resolveBuffer<RWStructuredBuffer<NrcPackedQueryPathInfo>>(getResourceParameters<SceneResourceParameters>().nrcQueryPathInfo)

#define gNrcTrainingPathInfo resolveBuffer<RWStructuredBuffer<NrcPackedTrainingPathInfo>>(getResourceParameters<SceneResourceParameters>().nrcTrainingPathInfo)

#define gNrcTrainingPathVertices resolveBuffer<RWStructuredBuffer<NrcPackedPathVertex>>(getResourceParameters<SceneResourceParameters>().nrcTrainingPathVertices)

typedef RWStructuredBuffer<NrcRadianceParams, ScalarDataLayout> NRCQueryRadianceBuffer;
#define gNrcQueryRadianceParams resolveBuffer<NRCQueryRadianceBuffer>(getResourceParameters<SceneResourceParameters>().nrcQueryRadianceParams)

#define gNrcCounters resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().nrcCounters)
#endif

#define gScene resolveAccelerationStructure(getResourceParameters<SceneResourceParameters>().scene)

#if METALLIC_STREAM_RAY_QUERIES || (METALLIC_STREAM_SHADOWS && METALLIC_STREAM_TLAS)
#include "StreamSceneRayQuery.slang"
#endif

bool traceShadow(float3 origin, float3 direction, float maxDistance, ScenePathTracePush push)
{
#if METALLIC_STREAM_RAY_QUERIES
    Ray ray; ray.origin = origin; ray.direction = direction;
    HitInfo hit;
    return traceStreamClosest(ray, 0.001, max(maxDistance, 0.001), push, hit);
#elif METALLIC_STREAM_MATERIALS
    return false; // Streamed opaque shading consumes the main-light shadow pass.
#else
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = max(maxDistance, 0.001);

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
#endif
}

bool traceClosest(Ray ray, ScenePathTracePush push, out HitInfo hit)
{
#if METALLIC_STREAM_RAY_QUERIES
    return traceStreamClosest(ray, max(push.clipOrtho.x, 0.0001), max(push.clipOrtho.y, 0.001), push, hit);
#elif METALLIC_STREAM_MATERIALS
    hit = (HitInfo)0;
    return false; // Streamed opaque shading consumes the main-light shadow pass.
#else
    hit = HitInfo();
    hit.t = 0.0;
    hit.position = ray.origin;
    hit.normal = float3(0.0, 1.0, 0.0);
    hit.geometryNormal = hit.normal;
    hit.tangent = float4(1.0, 0.0, 0.0, 1.0);
    hit.bitangent = float3(0.0, 0.0, 1.0);
    hit.texcoord = float2(0.0, 0.0);
    hit.materialIndex = 0u;
    hit.instanceIndex = 0u;
    hit.primitiveIndex = 0u;
    hit.triangleIndex = 0u;
    hit.rayConeLodConstant = 0.0;
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
    PathTraceInstance instance = gInstances[instanceIndex];
    PathTracePrimitive primitive = gPrimitives[instance.primitiveIndex];
    uint triangleIndex = query.CommittedPrimitiveIndex();
    uint indexOffset = primitive.firstIndex + triangleIndex * 3u;

    uint i0 = primitive.firstVertex + gIndices[indexOffset + 0u];
    uint i1 = primitive.firstVertex + gIndices[indexOffset + 1u];
    uint i2 = primitive.firstVertex + gIndices[indexOffset + 2u];
    SceneShadingVertex v0 = resolveBuffer<StructuredBuffer<SceneShadingVertex>>(getResourceParameters<SceneResourceParameters>().vertices)[i0];
    SceneShadingVertex v1 = resolveBuffer<StructuredBuffer<SceneShadingVertex>>(getResourceParameters<SceneResourceParameters>().vertices)[i1];
    SceneShadingVertex v2 = resolveBuffer<StructuredBuffer<SceneShadingVertex>>(getResourceParameters<SceneResourceParameters>().vertices)[i2];

    float2 bary = query.CommittedTriangleBarycentrics();
    float3 weights = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

#if SCENE_RAYQUERY_ENABLE_POSITION_FETCH
    // Position fetch returns object-space positions in triangle vertex order.
    // Attributes still use the indexed vertex buffer; keep the same TBN basis.
    float3 positions[3] = query.CommittedGetIntersectionTriangleVertexPositions();
    float3 p0 = positions[0];
    float3 p1 = positions[1];
    float3 p2 = positions[2];
#else
    float3 p0 = loadSceneVertexPosition(resolveBuffer<StructuredBuffer<SceneVertexPosition>>(getResourceParameters<SceneResourceParameters>().positions), i0);
    float3 p1 = loadSceneVertexPosition(resolveBuffer<StructuredBuffer<SceneVertexPosition>>(getResourceParameters<SceneResourceParameters>().positions), i1);
    float3 p2 = loadSceneVertexPosition(resolveBuffer<StructuredBuffer<SceneVertexPosition>>(getResourceParameters<SceneResourceParameters>().positions), i2);
#endif
    float2 uv0 = v0.texcoord;
    float2 uv1 = v1.texcoord;
    float2 uv2 = v2.texcoord;
    float3 localNormal = decodeSceneNormal(v0.normal) * weights.x +
        decodeSceneNormal(v1.normal) * weights.y + decodeSceneNormal(v2.normal) * weights.z;
    float3 localNormalFallback = safeNormalize(localNormal, float3(0.0, 1.0, 0.0));
    float3 localGeometryNormal = safeTriangleNormal(
        p1 - p0,
        p2 - p0,
        localNormalFallback);
    localNormal = dot(localNormal, localNormal) > 0.000001 ? safeNormalize(localNormal, localGeometryNormal) : localGeometryNormal;
    hit.texcoord = uv0 * weights.x + uv1 * weights.y + uv2 * weights.z;

    float3x4 worldToObject = query.CommittedWorldToObject3x4();
    float3x4 objectToWorld = query.CommittedObjectToWorld3x4();
    float handedness = determinant((float3x3)objectToWorld) < 0.0 ? -1.0 : 1.0;
    hit.t = query.CommittedRayT();
    hit.position = ray.origin + ray.direction * hit.t;
    hit.geometryNormal = safeNormalize(transformObjectNormalToWorld(localGeometryNormal, worldToObject), localGeometryNormal);
    hit.normal = safeNormalize(transformObjectNormalToWorld(localNormal, worldToObject), hit.geometryNormal);
    hit.frontFace = dot(hit.geometryNormal, -ray.direction) >= 0.0;

    float3 referenceAxis = abs(hit.normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 fallbackTangent = safeNormalize(cross(referenceAxis, hit.normal), float3(1.0, 0.0, 0.0));
    float tangentSign = 1.0;
    if ((primitive.flags & kPrimitiveHasAuthoredTangents) != 0u) {
        float4 localTangent = decodeSceneTangent(v0.tangent) * weights.x +
            decodeSceneTangent(v1.tangent) * weights.y + decodeSceneTangent(v2.tangent) * weights.z;
        float3 localTangentDirection = dot(localTangent.xyz, localTangent.xyz) > 0.000001
            ? safeNormalize(localTangent.xyz, float3(1.0, 0.0, 0.0))
            : float3(1.0, 0.0, 0.0);
        tangentSign = (localTangent.w >= 0.0 ? 1.0 : -1.0) * handedness;
        float3 worldTangent = safeNormalize(mul(objectToWorld, float4(localTangentDirection, 0.0)), fallbackTangent);
        worldTangent = safeNormalize(worldTangent - hit.normal * dot(hit.normal, worldTangent), fallbackTangent);
        hit.tangent = float4(worldTangent, tangentSign);
        hit.bitangent = safeNormalize(cross(hit.normal, worldTangent) * tangentSign, cross(hit.normal, fallbackTangent));
    } else {
        float3 localTriangleTangent;
        float3 localTriangleBitangent;
        computeTriangleTangentSpace(p0, p1, p2, uv0, uv1, uv2, localNormal, localTriangleTangent, localTriangleBitangent);
        float3 worldTangent = safeNormalize(mul(objectToWorld, float4(localTriangleTangent, 0.0)), fallbackTangent);
        worldTangent = safeNormalize(worldTangent - hit.normal * dot(hit.normal, worldTangent), fallbackTangent);
        float3 worldBitangent = safeNormalize(mul(objectToWorld, float4(localTriangleBitangent, 0.0)), cross(hit.normal, worldTangent));
        worldBitangent = safeNormalize(worldBitangent - hit.normal * dot(hit.normal, worldBitangent), cross(hit.normal, worldTangent));
        tangentSign = dot(cross(hit.normal, worldTangent), worldBitangent) < 0.0 ? -1.0 : 1.0;
        hit.tangent = float4(worldTangent, tangentSign);
        hit.bitangent = worldBitangent;
    }

    hit.tangent.w *= push.bitangentFlip;
    hit.bitangent *= push.bitangentFlip;

    hit.materialIndex = instance.materialIndex;
    hit.instanceIndex = instanceIndex;
    hit.primitiveIndex = instance.primitiveIndex;
    hit.triangleIndex = triangleIndex;
    hit.rayConeLodConstant = instance.rayConeLodConstant;
    return true;
#endif
}

// Per-path radiance-cache state. Members exist only in permutations that use
// them; the plain permutation carries just the glossy cone tracker.
struct PathTraceCacheContext
{
#if METALLIC_PT_SHARC
    SharcParameters sharc;
    SharcState sharcState;
#endif
#if METALLIC_PT_NRC
    NrcContext nrc;
    NrcPathState nrcPathState;
#endif
    float materialRoughnessPrev;
};

float3 pathTrace(
    Ray ray,
    RayCone rayCone,
    ScenePathTracePush push,
    inout PathTraceCacheContext cache,
    inout uint seed,
    out float specularHitDistance,
    out Metallic.Atmosphere::AtmosphereSegment primaryAerial)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float previousBsdfPdf = 0.0;
    bool previousScatterDelta = true;
    uint maxDepth = clamp(push.maxDepth, 1u, 32u);
    cache.materialRoughnessPrev = 1.0;
    specularHitDistance = 0.0;
    bool captureSpecularHitDistance = false;
    float3 primaryEndpoint = float3(0.0);
    float3 primaryOrigin = ray.origin;
    bool primarySurface = false;
    primaryAerial.transmittance = float3(1.0);
    primaryAerial.radiance = float3(0.0);

#if METALLIC_PT_NRC
    NrcSetSampleIndex(cache.nrc, 0u);
#endif

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        if (!traceClosest(ray, push, hit)) {
            if (captureSpecularHitDistance) {
                specularHitDistance = 65504.0;
                captureSpecularHitDistance = false;
            }
            float misWeight = depth == 0u
                ? 1.0
                : environmentBsdfMisWeight(push, previousBsdfPdf, ray.direction, previousScatterDelta);
            float3 missRadiance = environmentRadiance(ray.direction, push, depth == 0u, primaryOrigin) * misWeight;
            if (depth > 0u && previousScatterDelta && push.environmentMode != kEnvironmentModeDisabled) {
                missRadiance += physicalDeltaMissDisks(getResourceParameters<SceneResourceParameters>().environment,
                    ray.direction, ray.origin) * max(push.environmentIntensity, 0.0);
            }
            radiance += throughput * missRadiance;
#if METALLIC_PT_SHARC
            SharcUpdateMiss(cache.sharc, cache.sharcState, missRadiance);
#endif
#if METALLIC_PT_NRC
            NrcUpdateOnMiss(cache.nrcPathState);
#endif
            break;
        }
        if (captureSpecularHitDistance) {
            specularHitDistance = hit.t;
            captureSpecularHitDistance = false;
        }
        if (depth == 0u) { primaryEndpoint = hit.position; primarySurface = true; }

#if METALLIC_PT_SHARC && SHARC_UPDATE
        // SHaRC accumulates each path segment independently: drop the signal
        // gathered for the previous segment before evaluating this hit.
        radiance = float3(0.0, 0.0, 0.0);
        throughput = float3(1.0, 1.0, 1.0);
#endif

        rayCone = propagateRayCone(rayCone, hit.t);
        PathTraceMaterial material = loadSceneMaterial(hit.materialIndex);
        bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
        float4 baseColorSample = loadMaterialTextureForMaterial(
            material.baseColorTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 metallicRoughnessSample = loadMaterialTextureForMaterial(
            material.metallicRoughnessTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 emissiveSample = loadMaterialTextureForMaterial(
            material.emissiveTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 normalSample = loadMaterialTextureForMaterial(
            material.normalTexture,
            hit.texcoord,
            float4(0.5, 0.5, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 occlusionSample = loadMaterialTextureForMaterial(
            material.occlusionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 transmissionSample = loadMaterialTextureForMaterial(
            material.transmissionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 thicknessSample = loadMaterialTextureForMaterial(
            material.thicknessTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 diffuseTransmissionSample = loadMaterialTextureForMaterial(
            material.diffuseTransmissionTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        float4 diffuseTransmissionColorSample = loadMaterialTextureForMaterial(
            material.diffuseTransmissionColorTexture,
            hit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push,
            rayCone,
            hit,
            ray.direction,
            deterministicGlassTextures,
            seed);
        hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
        float3 stableShadingNormal = hit.normal;
        hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);
        float occlusion = lerp(1.0, saturate(occlusionSample.r), saturate(material.textureParams.y));

        float3 baseColor = saturate(materialTextureModulate(material.baseColor.rgb, baseColorSample.rgb, material.baseColorTexture.transform0.w));
        float metallic = saturate(material.params.x * metallicRoughnessSample.b);
        float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.02, 1.0);
        float transmission = saturate(material.glassParams.x * transmissionSample.r);
        float ior = clamp(material.glassParams.y, 1.0, 3.0);
        float thickness = max(material.glassParams.z * thicknessSample.g, 0.0);
        float attenuationDistance = max(material.glassParams.w, 0.0);
        float3 attenuationColor = max(saturate(material.attenuationColor.rgb), float3(0.001, 0.001, 0.001));
        float diffuseTransmission = saturate(material.diffuseTransmission.a * diffuseTransmissionSample.a);
        float3 diffuseTransmissionColor = max(
            saturate(materialTextureModulate(material.diffuseTransmission.rgb, diffuseTransmissionColorSample.rgb, material.diffuseTransmissionColorTexture.transform0.w)),
            float3(0.001, 0.001, 0.001));
        float dielectricF0 = (ior - 1.0) / (ior + 1.0);
        dielectricF0 *= dielectricF0;
        float3 f0 = lerp(float3(dielectricF0, dielectricF0, dielectricF0), baseColor, metallic);

        float3 emissive = max(materialTextureModulate(material.emissive.rgb, emissiveSample.rgb, material.emissiveTexture.transform0.w), float3(0.0, 0.0, 0.0));
        radiance += throughput * emissive;

        if (materialDomain(material) == kMaterialDomainFiber) {
            if (!shadeSceneFiber(material, hit, stableShadingNormal, push, depth,
                    ray, rayCone, seed, throughput, radiance, previousBsdfPdf, previousScatterDelta)) { break; }
            continue;
        }

        if (transmission > 0.01) {
            float eta = hit.frontFace ? 1.0 / max(ior, 1.001) : max(ior, 1.001);
            float3 refractedDirection = float3(0.0, 0.0, 0.0);
            bool canRefract = refractDirection(ray.direction, hit.normal, eta, refractedDirection);
            float reflectProbability = canRefract
                ? fresnelDielectric(ior, saturate(dot(-ray.direction, hit.normal)))
                : 1.0;
            if (!canRefract || rand(seed) < reflectProbability) {
                ray.direction = safeNormalize(reflect(ray.direction, hit.normal), hit.normal);
                if (depth == 0u) {
                    captureSpecularHitDistance = true;
                }
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
            previousBsdfPdf = 1.0;
            previousScatterDelta = true;
            ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);
            continue;
        }

        // ------------------------------------------------------------------
        // Radiance caches (RTXGI SHaRC / NVIDIA NRC reference integration).
        // Delta-lobe segments (ideal transmission, hair) skip the caches,
        // mirroring the reference treatment of skipped delta vertices.
        // ------------------------------------------------------------------
#if METALLIC_PT_NRC
        NrcSurfaceAttributes nrcSurface = (NrcSurfaceAttributes)0;
        nrcSurface.encodedPosition = NrcEncodePosition(hit.position, cache.nrc.constants);
        nrcSurface.roughness = roughness;
        nrcSurface.specularF0 = f0;
        nrcSurface.diffuseReflectance = baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission);
        nrcSurface.shadingNormal = hit.normal;
        nrcSurface.viewVector = -ray.direction;
        nrcSurface.isDeltaLobe = transmission > 0.01;
        NrcProgressState nrcProgressState = NrcUpdateOnHit(
            cache.nrc,
            cache.nrcPathState,
            nrcSurface,
            hit.t,
            depth,
            throughput,
            radiance);
        if (nrcProgressState == NrcProgressState::TerminateImmediately) {
            break;
        }
#endif

#if METALLIC_PT_SHARC
        SharcHitData sharcHitData;
        sharcHitData.positionWorld = hit.position;
        sharcHitData.normalWorld = hit.geometryNormal;

#if SHARC_QUERY
        // Camera hits must retain exact material/normal and direct-light detail.
        // The cache is only filtered enough to query after a surface bounce.
        if (depth > 0u) {
            uint gridLevel = HashGridGetLevel(hit.position, cache.sharc.hashGridParameters);
            float voxelSize = HashGridGetVoxelSize(gridLevel, cache.sharc.hashGridParameters);
            bool isValidHit = hit.t > voxelSize * sqrt(3.0);

            float roughnessPrev = min(cache.materialRoughnessPrev, 0.99);
            float alpha = roughnessPrev * roughnessPrev;
            float footprint = hit.t * sqrt(max(0.5 * alpha * alpha / max(1.0 - alpha * alpha, 1e-6), 0.0));
            isValidHit &= footprint > voxelSize;

            float3 sharcRadiance;
            if (isValidHit && SharcGetCachedRadiance(cache.sharc, sharcHitData, sharcRadiance)) {
                radiance += sharcRadiance * throughput;
                break; // Terminate the path once we've looked up into the cache
            }
        }
#endif
#endif

        EnvironmentLightSample lightSample;
        // Sun and Moon remain fixed sources outside local RIS; physical mode
        // samples each finite disk before its projected BSDF and visibility.
        for (uint celestialSlot = 0u; celestialSlot < kCelestialLightCount; ++celestialSlot) {
            let celestial = resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameters<SceneResourceParameters>().environment.celestialLights);
            PunctualLightSample light = samplePathCelestialLightAtPosition(celestial, celestialSlot, hit.position, seed);
            float3 bsdf = punctualStandardBsdf(hit.normal, -ray.direction, light.direction,
                baseColor, f0, metallic, roughness, max(transmission, diffuseTransmission));
            if (luminanceMax(bsdf * light.illuminance) > 0.0) {
                float3 origin = offsetRay(hit.position,
                    dot(light.direction, hit.geometryNormal) >= 0.0 ? hit.geometryNormal : -hit.geometryNormal);
                if (!celestialLightCastsShadow(celestial, celestialSlot) ||
                    !traceShadow(origin, light.direction, light.distance, push)) {
                    radiance += throughput * bsdf * light.illuminance;
                }
            }
        }
        uint punctualLightIndex;
        float punctualInverseSourcePdf;
        if (selectPunctualLight(PunctualSamplingResources(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid), resolveUniform(getResourceParameters<SceneResourceParameters>().lightsPdf)), hit.position, seed, punctualLightIndex, punctualInverseSourcePdf)) {
            PunctualLightSample light = samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), punctualLightIndex, hit.position);
            float3 bsdf = punctualStandardBsdf(hit.normal, -ray.direction, light.direction,
                baseColor, f0, metallic, roughness, max(transmission, diffuseTransmission));
            if (luminanceMax(bsdf * light.illuminance) > 0.0) {
                float3 origin = offsetRay(hit.position,
                    dot(light.direction, hit.geometryNormal) >= 0.0 ? hit.geometryNormal : -hit.geometryNormal);
                if (!traceShadow(origin, light.direction,
                        max(light.distance - length(origin - hit.position), 0.0), push)) {
                    // This is the ReGIR RIS inverse source weight, not an
                    // environment solid-angle PDF. Delta-light MIS is one.
                    radiance += throughput * bsdf * light.illuminance * punctualInverseSourcePdf;
                }
            }
        }
        if (sampleEnvironmentLight(push, seed, lightSample)) {
            float3 lightDirection = lightSample.direction;
            float lightMisWeight = environmentLightMisWeight(
                push,
                lightSample.pdfSolidAngle,
                fallbackBsdfPdf(ray.direction, hit.normal, lightDirection, metallic, roughness, diffuseTransmission));
            float3 lightRadianceOverPdf =
                lightSample.radiance * lightMisWeight / max(lightSample.pdfSolidAngle, 0.000001);
            float nDotL = saturate(dot(hit.normal, lightDirection));
            if (nDotL > 0.0) {
                bool occluded = traceShadow(offsetRay(hit.position, hit.geometryNormal), lightDirection, 10000.0, push);
                if (!occluded) {
                    float3 viewDirection = -ray.direction;
                    float3 halfVector = safeNormalize(lightDirection + viewDirection, hit.normal);
                    float nDotH = saturate(dot(hit.normal, halfVector));
                    float specularPower = lerp(96.0, 8.0, roughness);
                    float3 diffuse = baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission) * (nDotL / kPi);
                    float3 specular = f0 * pow(nDotH, specularPower) * nDotL;
                    radiance += throughput * (diffuse + specular) * lightRadianceOverPdf;
                }
            }
            if (diffuseTransmission > 0.01) {
                float backNDotL = saturate(dot(-hit.normal, lightDirection));
                if (backNDotL > 0.0) {
                    float3 transmittedDiffuse =
                        baseColor *
                        diffuseTransmissionColor *
                        diffuseTransmission *
                        volumeTransmittance(
                            baseColor,
                            transmission,
                            thickness,
                            attenuationDistance,
                            attenuationColor,
                            max(thickness, 0.01)) *
                        (backNDotL / kPi);
                    radiance += throughput * transmittedDiffuse * lightRadianceOverPdf;
                }
            }
        }

#if METALLIC_PT_NRC
        // The NRC query was created at this vertex; direct lighting and
        // emissive terms have been added, so the path can terminate here.
        if (nrcProgressState == NrcProgressState::TerminateAfterDirectLighting) {
            break;
        }
#endif

#if METALLIC_PT_SHARC && SHARC_UPDATE
        // Feed the signal accumulated for this segment (emissive + direct
        // lighting) into the cache. Returning false means the path must stop.
        if (!SharcUpdateHit(cache.sharc, cache.sharcState, sharcHitData, radiance, rand(seed))) {
            break;
        }
#endif

        float3 diffuseDirection = sampleCosineHemisphere(hit.normal, seed);
        float3 incidentDirection = ray.direction;
        float3 reflectedDirection = reflect(incidentDirection, hit.normal);
        float specularChance = saturate(0.08 + metallic * 0.72 + (1.0 - roughness) * 0.18);
        bool sampledSpecularLobe = false;
        if (rand(seed) < specularChance) {
            sampledSpecularLobe = true;
            float glossMix = roughness * roughness;
            ray.direction = safeNormalize(lerp(reflectedDirection, diffuseDirection, glossMix), reflectedDirection);
            throughput *= lerp(f0, baseColor, metallic * 0.35);
        } else if (rand(seed) < diffuseTransmission * (1.0 - metallic)) {
            ray.direction = sampleCosineHemisphere(-hit.normal, seed);
            throughput *=
                baseColor *
                diffuseTransmissionColor *
                diffuseTransmission *
                volumeTransmittance(
                    baseColor,
                    transmission,
                    thickness,
                    attenuationDistance,
                    attenuationColor,
                    max(thickness, 0.01));
        } else {
            ray.direction = diffuseDirection;
            throughput *= baseColor * occlusion * (1.0 - metallic) * (1.0 - diffuseTransmission);
        }
        if (depth == 0u) {
            captureSpecularHitDistance = sampledSpecularLobe;
        }
        previousBsdfPdf = max(
            fallbackBsdfPdf(incidentDirection, hit.normal, ray.direction, metallic, roughness, diffuseTransmission),
            0.000001);
        previousScatterDelta = false;
        rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf);
        ray.origin = offsetRay(hit.position, dot(ray.direction, hit.geometryNormal) > 0.0 ? hit.geometryNormal : -hit.geometryNormal);

#if METALLIC_PT_SHARC
        // Track the effective cone spread of the path so far for the glossy
        // footprint test performed on the next hit.
        cache.materialRoughnessPrev += sampledSpecularLobe ? roughness : 1.0;
#endif
#if METALLIC_PT_NRC
        NrcSetBrdfPdf(cache.nrcPathState, previousBsdfPdf);
#endif
#if METALLIC_PT_SHARC && SHARC_UPDATE
        // Hand the segment throughput to the cache; each segment is
        // accumulated independently afterwards.
        SharcSetThroughput(cache.sharcState, throughput);
#endif

        bool allowRussianRoulette = true;
#if METALLIC_PT_NRC
        allowRussianRoulette = NrcCanUseRussianRoulette(cache.nrcPathState);
#endif
        if (allowRussianRoulette && depth >= 2u) {
            float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
            if (rand(seed) > continueProbability) {
                break;
            }
            throughput /= max(continueProbability, 0.001);
        }
    }

#if METALLIC_PT_NRC
    NrcWriteFinalPathInfo(cache.nrc, cache.nrcPathState, throughput, radiance);
#endif

    if (primarySurface) {
        primaryAerial = physicalAerialSegment(getResourceParameters<SceneResourceParameters>().environment, primaryEndpoint, primaryOrigin);
    }
#if SHARC_UPDATE || NRC_UPDATE || NRC_QUERY
    // Cache radiance stays local to the shaded point. NRC's display resolve
    // composes the completed radiance after its predicted indirect term is added.
    return radiance;
#else
    return primaryAerial.transmittance * radiance + primaryAerial.radiance;
#endif
}

PathTraceCacheContext makeCacheContext()
{
    PathTraceCacheContext cache;
    ScenePathTraceCacheParams params = gCacheParams[0];
#if METALLIC_PT_SHARC
    cache.sharc.hashGridParameters.cameraPosition = params.sharcCameraPosition.xyz;
    cache.sharc.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    cache.sharc.hashGridParameters.sceneScale = max(params.sharcSceneScale, 0.0001);
    cache.sharc.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

    cache.sharc.hashGridData.capacity = params.sharcEntriesNum;
    cache.sharc.hashGridData.hashEntriesBuffer = gSharcHashEntries;
    cache.sharc.accumulationBuffer = gSharcAccumulation;
    cache.sharc.resolvedBuffer = gSharcResolved;
    cache.sharc.radianceScale = 1.0e3;
    SharcInit(cache.sharcState);
#endif
#if METALLIC_PT_NRC
    NrcBuffers nrcBuffers;
    nrcBuffers.queryPathInfo = gNrcQueryPathInfo;
    nrcBuffers.trainingPathInfo = gNrcTrainingPathInfo;
    nrcBuffers.trainingPathVertices = gNrcTrainingPathVertices;
    nrcBuffers.queryRadianceParams = gNrcQueryRadianceParams;
    nrcBuffers.countersData = gNrcCounters;
    cache.nrc = NrcCreateContext(params.nrc, nrcBuffers, uint2(0, 0));
#endif
    cache.materialRoughnessPrev = 1.0;
    return cache;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
    uint2 pixel = dispatchThreadId.xy;
    uint samples = clamp(push.samples, 1u, 16u);
    uint2 outputPixel = pixel;
    uint seed = 0u;

#if SHARC_UPDATE
    // Sparse update: one random sample per stride x stride block of pixels,
    // with coverage distributed over time through the jitter and frame index.
    ScenePathTraceCacheParams cacheParams = gCacheParams[0];
    uint stride = max(cacheParams.sharcUpdateStride, 1u);
    if (pixel.x >= (push.width + stride - 1u) / stride || pixel.y >= (push.height + stride - 1u) / stride) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 updateJitter = float2(rand(seed), rand(seed));
    float2 samplePixel = (float2(pixel) + updateJitter) * float(stride);
    if (samplePixel.x >= float(push.width) || samplePixel.y >= float(push.height)) {
        return;
    }
    samples = 1u;
#elif NRC_UPDATE
    // NRC training pass: dispatch covers the training resolution; each thread
    // traces one path at a jittered position in the corresponding full-res area.
    ScenePathTraceCacheParams cacheParams = gCacheParams[0];
    if (pixel.x >= cacheParams.trainingWidth || pixel.y >= cacheParams.trainingHeight) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(cacheParams.trainingWidth, 1u) +
        (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 updateJitter = float2(rand(seed), rand(seed));
    float2 samplePixel = (float2(pixel) + updateJitter) *
        float2(push.width, push.height) /
        float2(max(cacheParams.trainingWidth, 1u), max(cacheParams.trainingHeight, 1u));
    samplePixel = min(samplePixel, float2(push.width, push.height) - float2(0.001, 0.001));
    samples = 1u;
#elif NRC_QUERY
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }
    ScenePathTraceCacheParams cacheParams = gCacheParams[0];
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (cacheParams.frameIndex + 1u) * 0x9e3779b9u);
    float2 samplePixel = float2(pixel) + float2(0.5, 0.5);
    samples = 1u;
#else
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }
    seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.sampleFrame + 1u) * 0x9e3779b9u);
    float2 samplePixel = float2(pixel);
#endif

    PathTraceCacheContext cache = makeCacheContext();

    float3 color = float3(0.0, 0.0, 0.0);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
#if SHARC_UPDATE || NRC_UPDATE || NRC_QUERY
        float2 jitter = float2(0.0, 0.0);
#else
        float2 jitter = push.temporalJitter != 0u
            ? float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY)
            : (samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed)));
#endif
#if !(SHARC_UPDATE || NRC_UPDATE || NRC_QUERY)
        samplePixel = float2(outputPixel) + jitter;
#endif

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
#if METALLIC_PT_NRC
        cache.nrc.pixelIndex = outputPixel;
        NrcSetSampleIndex(cache.nrc, sampleIndex);
        cache.nrcPathState = NrcCreatePathState(cache.nrc.constants, rand(seed));
#endif
        float ignoredSpecularHitDistance;
        Metallic.Atmosphere::AtmosphereSegment primaryAerial;
        color += pathTrace(ray, rayCone, push, cache, seed, ignoredSpecularHitDistance, primaryAerial);
#if NRC_QUERY
        let aerialHandle = getResourceParameters<SceneResourceParameters>().environment.primaryAerial;
        if (aerialHandle.index != 0xffffffffu) {
            let aerial = resolveBuffer<RWStructuredBuffer<float4>>(aerialHandle);
            uint count, stride;
            aerial.GetDimensions(count, stride);
            uint index = (pixel.y * push.width + pixel.x) * 2u;
            if (index + 1u < count) {
                aerial[index] = float4(primaryAerial.transmittance, 1.0);
                aerial[index + 1u] = float4(primaryAerial.radiance, 0.0);
            }
        }
#endif
    }
#if SHARC_UPDATE || NRC_UPDATE
    // Cache update passes produce no image output.
    return;
#endif
    color /= float(samples);

#if NRC_QUERY
    // Resolve adds this frame's predicted indirect light. Accumulate the
    // complete frame afterwards so direct and cached light share one weight.
    gHistoryCurrent[pixel] = float4(color, 1.0);
    return;
#endif

    if (push.outputLinear == 0u) {
        color = Metallic.WorkingColor::toLinearRec709(color);
        color *= lightingExposure(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights));
        color = color / (color + float3(1.0, 1.0, 1.0));
        color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
        color = saturate(color);
    }

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
