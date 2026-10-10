// The cache stores outgoing scene-linear radiance. Its key has position and
// geometric normal, but no outgoing direction or medium. Keep directional and
// transmitting closures on the exact path; only diffuse-like surfaces populate
// this cache. Neither primary guides nor the authored TBN use cached data.
#if METALLIC_PT_SHARC
bool openPbrCacheSurfaceEligible(OpenPBRResolvedInputs inputs, PathTraceMaterial material,
    OpenPBRInteriorStack interiorStack)
{
    return interiorStack.count == 0u && material.textureParams.w < 0.5 && material.emissive.w == 0.0 &&
        inputs.geometry_opacity >= 0.999 && inputs.emission_luminance <= 0.0 &&
        inputs.base_metalness <= 0.05 && inputs.transmission_weight <= 0.001 &&
        inputs.subsurface_weight <= 0.001 && inputs.coat_weight <= 0.001 &&
        inputs.fuzz_weight <= 0.001 && inputs.thin_film_weight <= 0.001 &&
        abs(inputs.specular_roughness_anisotropy) <= 0.001 &&
        (inputs.specular_weight <= 0.001 || inputs.specular_roughness >= gCacheParams[0].sharcMinRoughness);
}

// A registered vertex already counted this path in SharcUpdateHit. Subsequent
// lighting (including a sky miss or a skipped glass/fiber vertex) contributes
// to that same sample, and must not increment its normalization count again.
void openPbrCacheAddLighting(inout PathTraceCacheContext cache, float3 lighting)
{
#if SHARC_UPDATE
    for (uint i = 0u; i < cache.sharcState.pathLength; ++i) {
        SharcAddVoxelData(cache.sharc, cache.sharcState.cacheIndices[i], lighting,
            cache.sharcState.sampleWeights[i], 0u);
    }
#endif
}

bool openPbrCacheQuery(PathTraceCacheContext cache, HitInfo hit, RayCone cone,
    uint depth, bool previousScatterDelta, out float3 radiance)
{
    radiance = float3(0.0);
#if SHARC_QUERY
    if (depth < max(gCacheParams[0].sharcQueryMinDepth, 1u) || previousScatterDelta) { return false; }
    uint level = HashGridGetLevel(hit.position, cache.sharc.hashGridParameters);
    float voxelSize = HashGridGetVoxelSize(level, cache.sharc.hashGridParameters);
    // Do not replace close contacts or a narrow glossy footprint by one voxel.
    if (hit.t <= voxelSize * sqrt(3.0) || cone.width <= voxelSize) { return false; }
    SharcHitData data;
    data.positionWorld = hit.position;
    data.normalWorld = hit.geometryNormal;
    return SharcGetCachedRadiance(cache.sharc, data, radiance);
#else
    return false;
#endif
}

bool openPbrCacheRecordHit(inout PathTraceCacheContext cache, HitInfo hit,
    bool eligible, float3 lighting, inout uint seed)
{
#if SHARC_UPDATE
    if (eligible) {
        SharcHitData data;
        data.positionWorld = hit.position;
        data.normalWorld = hit.geometryNormal;
        return SharcUpdateHit(cache.sharc, cache.sharcState, data, lighting, rand(seed));
    }
    openPbrCacheAddLighting(cache, lighting);
#endif
    return true;
}
#endif
