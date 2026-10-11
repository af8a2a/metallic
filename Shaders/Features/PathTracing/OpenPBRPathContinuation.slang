// One-ray continuation keeps only the original compact transport state live.
// The primary HitInfo and guide/material payloads are consumed in the first
// iteration. Subsequent iterations own only their current vertex HitInfo.
float3 pathTraceOpenPbrSingleSample(uint2 pixel, Ray ray, RayCone rayCone,
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
#if METALLIC_PT_SHARC
    OpenPBRCacheContext cache = makeOpenPbrCacheContext();
#endif

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        bool found;
        if (!primaryQueryConsumed) {
            primaryQueryConsumed = true;
            found = queryOpenPbrPrimaryStage(pixel, ray, rayCone, push, hit);
        } else {
            found = traceClosest(ray, push, hit);
        }
        let control = shadeOpenPBRPathVertex(found, hit, push,
            ray, rayCone, seed, depth, radiance, throughput,
            previousBsdfPdf, previousScatterDelta, interiorStack,
            specularHitDistance, captureSpecularHitDistance,
            alphaLayers, primaryEndpoint, primaryOrigin, primarySurface
#if METALLIC_PT_SHARC
            , cache
#endif
        );
        if (control == OpenPBRPathControl.Return) { return radiance; }
        if (control == OpenPBRPathControl.Stop) { break; }
    }

    return primarySurface && push.debugView == kOpenPBRDebugViewFinal
        ? applyPrimaryAerialPerspective(radiance, primaryEndpoint, primaryOrigin) : radiance;
}
