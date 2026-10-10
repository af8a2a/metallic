// OpenPBR keeps the two native BSDF components through direct lighting and
// continuation. Sampling still uses the complete mixture PDF.
OpenPBRDiffuseSpecular openPbrSelectComponents(OpenPBRDiffuseSpecular value, uint replacedMask)
{
    if (replacedMask & 1u) { value.diffuse = float3(0); }
    if (replacedMask & 2u) { value.specular = float3(0); }
    return value;
}

OpenPBRDiffuseSpecular openPbrDirectComponents<TContext : IOpenPBRContext, TLights : ISurfaceLightProvider>(
    OpenPBRPreparedClosure<TContext> prepared, inout TLights lights, uint replacedMask)
{
    OpenPBRDiffuseSpecular result = openPBRMakeZeroDiffuseSpecular();
    uint count = lights.count();
    for (uint index = 0u; index < count; ++index) {
        SurfaceLightSample light = lights.sample(index);
        if (light.inverseSourcePdf <= 0.0) { continue; }
        OpenPBRDiffuseSpecular weight = openPbrSelectComponents(prepared.evalProjectedComponents(light.direction), replacedMask);
        float3 contribution = openPBRGetSumOfDiffuseSpecular(weight) * light.illuminance;
        if (max(contribution.x, max(contribution.y, contribution.z)) <= 0.0) { continue; }
        float3 incident = light.illuminance * lights.visibility(light) * light.inverseSourcePdf;
        result = openPBRAddDiffuseSpecular(result, openPBRScaleDiffuseSpecular(weight, incident));
    }
    return result;
}

#if METALLIC_PT_SHARC
// Independent half-tables in the existing allocation: diffuse outgoing radiance
// and directional specular outgoing radiance, both in scene-linear units.
// No changes to SHaRC's vendor key format or the standard-material cache.
// OpenPBR resolve disables adjacent-level reprojection: these are opaque keys.
static const uint kOpenPBRCacheTrackedVertices = 2u;

struct OpenPBRCacheAddress
{
    uint64_t diffuse;
    uint64_t specular;
    uint eligibleMask;
    uint absentMask;
};

struct OpenPBRCacheContext
{
    SharcParameters sharc;
#if SHARC_UPDATE
    uint2 indices[kOpenPBRCacheTrackedVertices];
    OpenPBRDiffuseSpecular weights[kOpenPBRCacheTrackedVertices];
    uint count;
#endif
};

OpenPBRCacheContext makeOpenPbrCacheContext()
{
    OpenPBRCacheContext result;
    result.sharc = makeCacheContext().sharc;
#if SHARC_UPDATE
    result.count = 0u;
#endif
    return result;
}

uint openPbrCacheDirectionBin(float3 direction, uint resolution)
{
    float3 n = direction / max(dot(abs(direction), float3(1)), 1e-8);
    float2 p = n.xy;
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * float2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    }
    uint2 cell = min(uint2(saturate(p * 0.5 + 0.5) * float(resolution)), resolution - 1u);
    return cell.x + cell.y * resolution;
}

uint64_t openPbrCacheMixKey(uint64_t value)
{
    value = (value ^ (value >> 30)) * uint64_t(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * uint64_t(0x94d049bb133111eb);
    return (value ^ (value >> 31)) | uint64_t(1); // Never the empty key (zero).
}

OpenPBRCacheAddress openPbrCacheAddress(OpenPBRCacheContext cache, HitInfo hit,
    OpenPBRResolvedInputs inputs, PathTraceMaterial material, OpenPBRInteriorStack interiorStack, float3 wo)
{
    OpenPBRCacheAddress address = (OpenPBRCacheAddress)0;
    // Media and layered/transmitting closures continue exact transport. Sharp
    // dielectric surfaces can still cache their diffuse component independently.
    bool eligible = interiorStack.count == 0u && material.textureParams.w < 0.5 && material.emissive.w == 0.0 &&
        inputs.geometry_opacity >= 0.999 && inputs.emission_luminance <= 0.0 &&
        inputs.transmission_weight <= 0.001 && inputs.subsurface_weight <= 0.001 &&
        inputs.coat_weight <= 0.001 && inputs.fuzz_weight <= 0.001 && inputs.thin_film_weight <= 0.001 &&
        abs(inputs.specular_roughness_anisotropy) <= 0.001;
    if (!eligible) { return address; }
    // Pure metal has no diffuse lobe. All other components must either hit a
    // cache entry or be traced; low amplitude alone never counts as a cache hit.
    if (inputs.base_metalness == 1.0) { address.absentMask = 1u; }
    address.eligibleMask = address.absentMask == 0u ? 1u : 0u;
    if (inputs.specular_roughness >= max(gCacheParams[0].sharcMinRoughness, 0.2)) {
        address.eligibleMask |= 2u;
    }
    address.eligibleMask &= gCacheParams[0].sharcLobeMask;
    if (address.eligibleMask == 0u) { return address; }
    uint64_t spatial = HashGridComputeSpatialHash(hit.position, hit.geometryNormal, cache.sharc.hashGridParameters);
    // Distinguish materials, shading normals and local textured parameters so
    // neighboring surfaces cannot share a bright lobe solely by voxel position.
    uint normal = openPbrCacheDirectionBin(inputs.geometry_basis.n, 64u);
    uint3 color = uint3(saturate(inputs.base_color) * 31.0 + 0.5);
    uint materialClass = color.x | (color.y << 5) | (color.z << 10) |
        (uint(saturate(inputs.base_metalness) * 31.0 + 0.5) << 15) |
        (uint(saturate(inputs.specular_roughness) * 31.0 + 0.5) << 20);
    uint64_t surface = openPbrCacheMixKey(spatial ^ (uint64_t(hit.materialIndex) << 32));
    surface = openPbrCacheMixKey(surface ^ (uint64_t(materialClass) << 12) ^ uint64_t(normal));
    address.diffuse = openPbrCacheMixKey(surface ^ uint64_t(0x243f6a8885a308d3));
    // Angular cells shrink with the specular lobe. Roughness is already in the
    // surface key; include resolution explicitly at quantization boundaries.
    if (address.eligibleMask & 2u) {
        uint resolution = uint(clamp(ceil(4.0 / max(inputs.specular_roughness * inputs.specular_roughness, 0.0625)), 4.0, 64.0));
        uint direction = openPbrCacheDirectionBin(wo, resolution) | (resolution << 16);
        address.specular = openPbrCacheMixKey(surface ^ uint64_t(direction) ^ uint64_t(0x13198a2e03707344));
    }
    return address;
}

uint openPbrCacheBaseSlot(OpenPBRCacheContext cache, uint64_t key, uint component)
{
    uint capacity = cache.sharc.hashGridData.capacity / 2u;
    return component * capacity + HashGridGetBaseSlot(key, capacity);
}

bool openPbrCacheFind(OpenPBRCacheContext cache, uint64_t key, uint component, out float3 radiance)
{
    radiance = float3(0);
    uint index, bucketOffset;
    if (!HashGridFind(cache.sharc.hashGridData, key, openPbrCacheBaseSlot(cache, key, component),
        HASH_GRID_HASH_MAP_BUCKET_SIZE, index, bucketOffset)) { return false; }
    SharcVoxelData_SharcCommon data = SharcGetVoxelData(cache.sharc.resolvedBuffer, index);
    // A directional bin needs multiple independent samples before reuse.
    if (data.accumulatedSampleNum < (component == 0u ? 2.0 : 8.0) || data.accumulatedFrameNum < 2u) { return false; }
    radiance = data.accumulatedRadiance.radiance;
    return true;
}

uint openPbrCacheQuery(OpenPBRCacheContext cache, OpenPBRCacheAddress address, HitInfo hit, RayCone cone,
    uint depth, bool previousScatterDelta, out float3 radiance)
{
    radiance = float3(0);
    if (address.eligibleMask == 0u || depth < max(gCacheParams[0].sharcQueryMinDepth, 1u) || previousScatterDelta) { return 0u; }
    uint level = HashGridGetLevel(hit.position, cache.sharc.hashGridParameters);
    float voxelSize = HashGridGetVoxelSize(level, cache.sharc.hashGridParameters);
    if (hit.t <= voxelSize * sqrt(3.0) || cone.width <= voxelSize) { return 0u; }
    uint mask = 0u;
    float3 value;
    if ((address.eligibleMask & 1u) && openPbrCacheFind(cache, address.diffuse, 0u, value)) {
        radiance += value;
        mask |= 1u;
    }
    if ((address.eligibleMask & 2u) && openPbrCacheFind(cache, address.specular, 1u, value)) {
        radiance += value;
        mask |= 2u;
    }
    return mask;
}

// Count each registered component once. Every later light/miss contributes to
// that sample with its own first-vertex component weight, then total BSDF weights.
void openPbrCacheAddLighting(inout OpenPBRCacheContext cache, float3 lighting)
{
#if SHARC_UPDATE
    for (uint i = 0u; i < cache.count; ++i) {
        if (cache.indices[i].x != HASH_GRID_INVALID_CACHE_INDEX) {
            SharcAddVoxelData(cache.sharc, cache.indices[i].x, lighting, cache.weights[i].diffuse, 0u);
        }
        if (cache.indices[i].y != HASH_GRID_INVALID_CACHE_INDEX) {
            SharcAddVoxelData(cache.sharc, cache.indices[i].y, lighting, cache.weights[i].specular, 0u);
        }
    }
#endif
}

void openPbrCacheSetThroughput(inout OpenPBRCacheContext cache, float3 throughput)
{
#if SHARC_UPDATE
    for (uint i = 0u; i < cache.count; ++i) {
        cache.weights[i] = openPBRScaleDiffuseSpecular(cache.weights[i], throughput);
    }
#endif
}

bool openPbrCacheRecordHit(inout OpenPBRCacheContext cache, OpenPBRCacheAddress address,
    OpenPBRDiffuseSpecular lighting, float3 emission)
{
#if SHARC_UPDATE
    openPbrCacheAddLighting(cache, openPBRGetSumOfDiffuseSpecular(lighting) + emission);
    if (cache.count >= kOpenPBRCacheTrackedVertices || address.eligibleMask == 0u) { return false; }
    uint2 indices = uint2(HASH_GRID_INVALID_CACHE_INDEX);
    uint bucketOffset;
    if ((address.eligibleMask & 1u) && HashGridInsert(cache.sharc.hashGridData, address.diffuse,
        openPbrCacheBaseSlot(cache, address.diffuse, 0u), HASH_GRID_HASH_MAP_BUCKET_SIZE, indices.x, bucketOffset)) {
        SharcAddVoxelData(cache.sharc, indices.x, lighting.diffuse, float3(1), 1u);
    }
    if ((address.eligibleMask & 2u) && HashGridInsert(cache.sharc.hashGridData, address.specular,
        openPbrCacheBaseSlot(cache, address.specular, 1u), HASH_GRID_HASH_MAP_BUCKET_SIZE, indices.y, bucketOffset)) {
        SharcAddVoxelData(cache.sharc, indices.y, lighting.specular, float3(1), 1u);
    }
    if (all(indices == uint2(HASH_GRID_INVALID_CACHE_INDEX))) { return false; }
    cache.indices[cache.count] = indices;
    cache.weights[cache.count] = openPBRMakeZeroDiffuseSpecular();
    ++cache.count;
    return true;
#else
    return false;
#endif
}

void openPbrCacheScatter(inout OpenPBRCacheContext cache, OpenPBRDiffuseSpecular weight, bool registered)
{
#if SHARC_UPDATE
    openPbrCacheSetThroughput(cache, openPBRGetSumOfDiffuseSpecular(weight));
    if (registered) { cache.weights[cache.count - 1u] = weight; }
    // Full suffixes stay attached to the first two registered vertices. Never
    // drop their tail when the tracking budget is full or a bootstrap misses.
#endif
}
#endif
