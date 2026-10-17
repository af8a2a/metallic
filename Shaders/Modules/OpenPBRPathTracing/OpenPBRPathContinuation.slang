implementing OpenPBRPathTracing;

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
import OpenPBR;
import OpenPBRClosure;
import OpenPBRSurface;
import SceneSurfaceMaterial;
import SurfaceLighting;
import ScenePathTracing;
import SHARC;
#if METALLIC_PT_SPLIT_PRIMARY
import OpenPBRPrimaryHit;
#endif
#if METALLIC_CUSTOM_MATERIALS
import SlabClosure;
#endif
using Metallic.OpenPBR;
using Metallic.SceneOpenPBR;
using Metallic.ScenePathTracing;
using Metallic.SHARC;

namespace Metallic.OpenPBRPathTracing
{
#if !SHARC_UPDATE
// One-ray continuation keeps only the original compact transport state live.
// The primary HitInfo and guide/material payloads are consumed in the first
// iteration. Subsequent iterations own only their current vertex HitInfo.
public float3 pathTraceOpenPbrSingleSample(uint2 pixel, Ray ray, RayCone rayCone,
    ScenePathTracePush push, uint seed, out float specularHitDistance)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float previousBsdfPdf = 0.0;
    bool previousScatterDelta = true;
    uint maxDepth = clamp(push.maxDepth, 1u, 32u);
    OpenPBRInteriorStack interiorStack = makeOpenPBRInteriorStack();
    specularHitDistance = 0.0;
    bool captureSpecularHitDistance = false;
    uint alphaLayers = 0u;
    float3 primaryEndpoint = float3(0.0);
    float3 primaryOrigin = ray.origin;
    bool primarySurface = false;
    bool primaryQueryConsumed = false;
#if (SHARC_UPDATE || SHARC_QUERY)
#if METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION
    // Select only numeric values and engine handle indices across the policy branch.
    // Materials is mandatory for every accepted production dispatch, including misses.
    let inactiveMaterials = getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneMaterialsOffset);
    float3 cacheCameraPosition = float3(0.0, 0.0, 0.0);
    float cacheLogarithmBase = 0.0;
    float cacheSceneScale = 0.0;
    float cacheLevelBias = 0.0;
    uint cacheCapacity = 0u;
    float cacheRadianceScale = 0.0;
    ResourceHandle<ByteAddressBuffer> cacheHashEntries = inactiveMaterials;
    ResourceHandle<ByteAddressBuffer> cacheAccumulation = inactiveMaterials;
    ResourceHandle<ByteAddressBuffer> cacheResolved = inactiveMaterials;
    if (kOpenPBRSingleCacheQueryEnabled) {
        // Keep these expressions and their order in sync with makeCacheContext().
        ScenePathTraceCacheParams params = scenePathCacheParams()[0];
        let resources = getSceneResourceParameterField<SceneSharcResourceParameters>(kSceneSharcOffset);
        cacheCameraPosition = params.sharcCameraPosition.xyz;
        cacheLogarithmBase = kSHARCGridLogarithmBase;
        cacheSceneScale = max(params.sharcSceneScale, 0.0001);
        cacheLevelBias = kSHARCGridLevelBias;
        cacheCapacity = params.sharcEntriesNum;
        cacheHashEntries = resources.sharcHashEntries;
        cacheAccumulation = resources.sharcAccumulation;
        cacheResolved = resources.sharcResolved;
        cacheRadianceScale = 1.0e3;
    }
    // Resolve after the merge; no opaque descriptor value crosses the branch.
    SHARCParameters selectedSharc;
    selectedSharc.hashGridParameters.cameraPosition = cacheCameraPosition;
    selectedSharc.hashGridParameters.logarithmBase = cacheLogarithmBase;
    selectedSharc.hashGridParameters.sceneScale = cacheSceneScale;
    selectedSharc.hashGridParameters.levelBias = cacheLevelBias;
    selectedSharc.hashGridData.capacity = cacheCapacity;
    selectedSharc.hashGridData.hashEntriesBuffer = resolveBuffer<RWStructuredBuffer<uint64_t>>(cacheHashEntries);
    selectedSharc.accumulationBuffer = resolveBuffer<RWStructuredBuffer<SHARCAccumulationData>>(cacheAccumulation);
    selectedSharc.resolvedBuffer = resolveBuffer<RWStructuredBuffer<SHARCPackedData>>(cacheResolved);
    selectedSharc.radianceScale = cacheRadianceScale;
    OpenPBRCacheContext cache = { selectedSharc };
#else
    OpenPBRCacheContext cache = makeOpenPbrCacheContext();
#endif
#endif

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        bool found;
        if (!primaryQueryConsumed) {
            primaryQueryConsumed = true;
#if METALLIC_PT_SPLIT_PRIMARY
            // A known miss is consumed too. BLEND may return to depth zero,
            // but subsequent queries must follow the updated continuation ray.
            found = Metallic.OpenPBRPrimaryHit::loadOpenPbrPrimaryHit(pixel, push, hit);
#else
            found = queryOpenPbrPrimaryStage(pixel, ray, rayCone, push, hit);
#endif
        } else {
            found = traceClosest(ray, push, hit);
        }
        let control = shadeOpenPBRPathVertex(found, hit, push,
            ray, rayCone, seed, depth, radiance, throughput,
            previousBsdfPdf, previousScatterDelta, interiorStack,
            specularHitDistance, captureSpecularHitDistance,
            alphaLayers, primaryEndpoint, primaryOrigin, primarySurface
#if (SHARC_UPDATE || SHARC_QUERY)
            , cache
#endif
        );
        if (control == OpenPBRPathControl.Return) { return radiance; }
        if (control == OpenPBRPathControl.Stop) { break; }
    }

    return primarySurface && push.debugView == kOpenPBRDebugViewFinal
        ? applyPrimaryAerialPerspective(radiance, primaryEndpoint, primaryOrigin) : radiance;
}

#endif
} // namespace Metallic.OpenPBRPathTracing
