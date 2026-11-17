#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// BLAS publication state, RT owner priority and representation helpers.
public static const uint kBLASInstanceDirty = 8u;
public static const uint kBLASInstanceSemanticCutDirty = 1u << 4u;
// Persist the representation actually published to the TLAS, including failed
// admissions. Address relocations and requested cuts do not change this state.
public static const uint kBLASInstancePublishedFallback = 1u << 5u;
public static const uint kBLASInstancePublishedDynamic = 2u << 5u;
public static const uint kBLASInstancePublishedMask = 3u << 5u;

public bool streamBlasDynamicReady(GPUDrivenStreamAssetInstanceBLAS blas)
{
    return (blas.flags & kBLASInstanceDynamic) != 0u &&
        (blas.flags & (kBLASInstanceFallback | kBLASInstanceOverflow)) == 0u &&
        blas.selectedClusterCount != 0u &&
        blas.insertedClusterCount == blas.selectedClusterCount &&
        blas.cachedValid == 1u && blas.storageCapacity != 0u;
}

uint2 streamPublishedBlasAddress(GPUDrivenStreamAssetInstanceBLAS blas,
    GPUDrivenStreamAssetParams params, uint2 fallbackAddress)
{
    if (streamBlasDynamicReady(blas)) {
        uint low = params.blasStorageAddressLow + blas.storageOffset;
        return uint2(low, params.blasStorageAddressHigh +
            (low < params.blasStorageAddressLow ? 1u : 0u));
    }
    return fallbackAddress;
}
public groupshared uint4 g_StreamBlasStoragePrefix[64];

// RT representations own allocations and dependency keys. TLAS instances only
// select one of three canonical slots: highest drawable cut, shared view cut,
// complete discrete cut. Raster's per-instance frontier remains independent.
uint streamRTHistogramBase(GPUDrivenStreamAssetParams p)
{
    return p.sceneGroupCount * 3u;
}

static const uint kRTCachedLevelMask = 0x1fffffffu;
static const uint kRTIdleCache = 1u << 29u;
static const uint kRTEvictIdleCache = 1u << 30u;
static const uint kRTGroupOverflow = 1u << 31u;

uint streamRTPrimitiveBase(GPUDrivenStreamAssetParams p, uint primitive)
{
    return streamRTHistogramBase(p) + p.sceneLodLevelCount * 4u + primitive * 12u;
}

public uint streamRTInstanceBase(GPUDrivenStreamAssetParams p, uint instance)
{
    return streamRTPrimitiveBase(p, p.scenePrimitiveCount) + instance * 4u;
}

uint streamRTOwnerImportanceBase(GPUDrivenStreamAssetParams p, uint owner)
{
    return streamRTInstanceBase(p, p.sceneInstanceCount) + owner * 6u;
}

uint streamRTOwnerOrderBase(GPUDrivenStreamAssetParams p)
{
    return streamRTOwnerImportanceBase(p, p.rtRepresentationCount);
}

uint streamRTOwnerPriorityHistoryBase(GPUDrivenStreamAssetParams p)
{
    return streamRTOwnerOrderBase(p) + p.rtRepresentationCount * 3u;
}

uint streamRTOwnerPriorityControlBase(GPUDrivenStreamAssetParams p)
{
    return streamRTOwnerPriorityHistoryBase(p) + p.rtRepresentationCount * 3u;
}

uint4 streamRTOwnerPriorityKey(RWStructuredBuffer<uint> state, uint owner, GPUDrivenStreamAssetParams p)
{
    if (owner == 0xffffffffu) { return uint4(0xffffffffu); }
    uint base = streamRTOwnerImportanceBase(p, owner);
    uint errorKey = ~state[base];
    return uint4(errorKey, state[base + 1u], ~state[base + 2u], owner);
}

bool streamRTPriorityBefore(uint4 a, uint4 b)
{
    return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z != b.z ? a.z < b.z : a.w < b.w;
}

public groupshared uint4 g_RTPriorityKeys[64];

void sortStreamRTOwners(uint item, GPUDrivenStreamAssetPush push, GPUDrivenStreamAssetParams p)
{
    RWStructuredBuffer<uint> state = resolveUniform(
        ResourceHandle<RWStructuredBuffer<uint>>(p.rtRepresentationStateBuffer));
    uint count = p.rtRepresentationCount;
    uint base = streamRTOwnerOrderBase(p);
    uint control = streamRTOwnerPriorityControlBase(p);
    if (push.activeBuildPhase == 41u) {
        if (item >= count || (push.traversalPhase & 1u) != 0u) { return; }
        uint importance = streamRTOwnerImportanceBase(p, item);
        uint history = streamRTOwnerPriorityHistoryBase(p) + item * 3u;
        // Exact inputs to the total order, including owners without consumers.
        // Cut/residency/budget changes are validated separately by admission.
        bool errorChanged = state[importance] != state[history];
        bool distanceChanged = state[importance + 1u] != state[history + 1u];
        if (errorChanged || distanceChanged || state[importance + 2u] != state[history + 2u]) {
            InterlockedOr(state[control], 1u);
            uint word = errorChanged ? 0u : distanceChanged ? 1u : 2u;
            InterlockedMin(state[control + 2u], item * 3u + word);
        }
        return;
    }
    if (item == 0u && push.activeBuildPhase == 35u) {
        // Phase 41 has completed. Capture the first changed comparator before
        // phase 37 replaces history; one writer avoids racing provenance pairs.
        uint changed = state[control + 2u];
        if (changed != 0xffffffffu) {
            state[control + 3u] = state[streamRTOwnerPriorityHistoryBase(p) + changed];
            state[control + 4u] = state[streamRTOwnerImportanceBase(p, changed / 3u) + changed % 3u];
            state[control + 5u] = p.frameIndex;
            state[control + 6u] = changed;
        }
    }
    // The decision is uniform across every sort workgroup, before its barriers.
    if (state[control] == 0u) {
        if (item == 0u && push.activeBuildPhase == 35u) { state[control + 1u] = 1u; }
        return;
    }
    uint lane = item & 63u;
    if (push.activeBuildPhase == 35u) {
        uint4 key = streamRTOwnerPriorityKey(state, item < count ? item : 0xffffffffu, p);
        g_RTPriorityKeys[lane] = key;
        GroupMemoryBarrierWithGroupSync();
        for (uint width = 2u; width <= 64u; width <<= 1u) {
            for (uint stride = width >> 1u; stride != 0u; stride >>= 1u) {
                uint4 other = g_RTPriorityKeys[lane ^ stride];
                bool takeMinimum = ((lane & width) == 0u) == ((lane & stride) == 0u);
                bool otherBefore = streamRTPriorityBefore(other, key);
                if (takeMinimum ? otherBefore : streamRTPriorityBefore(key, other)) { key = other; }
                GroupMemoryBarrierWithGroupSync();
                g_RTPriorityKeys[lane] = key;
                GroupMemoryBarrierWithGroupSync();
            }
        }
        if (item < count) { state[base + item] = key.w; }
        return;
    }
    if (item >= count) { return; }
    uint source = base + ((push.traversalPhase >> 8u) & 1u) * count;
    uint owner = state[source + item];
    if (push.activeBuildPhase == 37u) {
        if (source != base) { state[base + item] = owner; }
        state[base + count * 2u + owner] = item;
        uint importance = streamRTOwnerImportanceBase(p, owner);
        uint history = streamRTOwnerPriorityHistoryBase(p) + owner * 3u;
        // Publish on every real sort, including cache-disabled frames. Switching
        // the cache on therefore consumes the last submitted order's exact keys.
        state[history] = state[importance];
        state[history + 1u] = state[importance + 1u];
        state[history + 2u] = state[importance + 2u];
        return;
    }
    // Each input run is sorted. Its element's binary-search rank in the other
    // run plus its own rank gives a unique output position (owner IDs break ties).
    uint width = 1u << (push.traversalPhase & 31u);
    uint pairBase = (item / width) & ~1u;
    pairBase *= width;
    uint ownBase = (item / width) * width;
    uint otherBase = ownBase == pairBase ? min(pairBase + width, count) : pairBase;
    uint low = otherBase;
    uint high = otherBase + min(width, count - otherBase);
    uint4 key = streamRTOwnerPriorityKey(state, owner, p);
    while (low < high) {
        uint middle = low + (high - low) / 2u;
        uint4 other = streamRTOwnerPriorityKey(state, state[source + middle], p);
        if (streamRTPriorityBefore(other, key)) { low = middle + 1u; }
        else { high = middle; }
    }
    uint destination = base + (source == base ? count : 0u);
    state[destination + pairBase + (item - ownBase) + (low - otherBase)] = owner;
}

void streamAccumulateRTOwnerImportance(RWStructuredBuffer<uint> state, uint owner,
    GPUDrivenStreamAssetInstance instance, GPUDrivenStreamAssetParams params)
{
    // Primary-view estimates of terminal fallback error in render pixels serve
    // as admission proxies, not exact secondary-ray errors. Aggregates are independent.
    StructuredBuffer<float4> metadata = resolveUniform(
        ResourceHandle<StructuredBuffer<float4>>(params.rtRepresentationMetadataBuffer));
    uint metadataIndex = (params.sceneLodLevelCount + instance.primitiveIndex) * 2u;
    float4 sphere = metadata[metadataIndex];
    float error = metadata[metadataIndex + 1u].x;
    float3 extent = abs(sphere.xyz) + sphere.www;
    float3 magnitude = abs(instance.world0.xyz) * extent.x + abs(instance.world1.xyz) * extent.y +
        abs(instance.world2.xyz) * extent.z + abs(instance.world3.xyz) + abs(params.eye.xyz);
    float padding = dot(magnitude, float3(1, 1, 1)) * (32.0 * 1.1920928955078125e-7);
    float scale = conservativeTransformScale(instance.world0.xyz, instance.world1.xyz, instance.world2.xyz);
    float3 center = instance.world0.xyz * sphere.x + instance.world1.xyz * sphere.y +
        instance.world2.xyz * sphere.z + instance.world3.xyz;
    float radius = sphere.w * scale + padding;
    bool finiteBounds = all(isfinite(center)) && isfinite(radius);
    GPUDrivenStreamAssetCameraData camera = currentStreamCamera(params);
    float guard = 1.0 + 2.0 / max(min(params.viewport.y, params.viewport.z), 1.0);
    camera.viewport.w = 2.0 * atan(tan(params.viewport.w * 0.5) * guard);
    camera.clipOrtho.z *= guard;
    bool frustumVisible = !finiteBounds || streamSphereInsideFrustum(center, radius, camera);
    float3 forward = safeNormalize(params.center.xyz - params.eye.xyz, float3(0, 0, -1));
    float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 direction = float4(forward, params.upProjection.w);
    float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, 0.0);
    float maxFinite = asfloat(0x7f7fffffu);
    float projected = meshletLodRayProjectedError(sphere, error, instance.world0, instance.world1,
        instance.world2, instance.world3, eye, direction, projection, padding);
    projected = isfinite(projected) ? (projected > 0.0 ? projected : 0.0) : maxFinite;
    if (frustumVisible) {
        float raster = meshletLodProjectedError(sphere, error, instance.world0, instance.world1,
            instance.world2, instance.world3, eye, direction, projection, padding);
        raster = isfinite(raster) ? (raster > 0.0 ? raster : 0.0) : maxFinite;
        projected = max(projected, raster);
    }
    float distance = length(center - params.eye.xyz) - radius;
    distance = finiteBounds && isfinite(distance) && distance > 0.0 ? distance : 0.0;
    float coverage = 0.0;
    if (frustumVisible) {
        float4 rect;
        float2 size = params.viewport.yz;
        bool bounded = conservativeSphereScreenRect(streamWorldToView(center, currentStreamCamera(params)),
            radius, size, params.viewport.x, params.viewport.w,
            params.upProjection.w != 0.0 ? params.clipOrtho.z : 0.0, params.clipOrtho.x, rect);
        coverage = bounded ? saturate(((rect.z - rect.x + 1.0) * (rect.w - rect.y + 1.0)) /
            (size.x * size.y)) : 1.0;
    }
    uint base = streamRTOwnerImportanceBase(params, owner);
    // Zero means no consumer. A real zero approximation error retains key 1.
    InterlockedMax(state[base], asuint(projected) + 1u);
    InterlockedMin(state[base + 1u], asuint(distance));
    InterlockedMax(state[base + 2u], asuint(coverage));
    InterlockedAdd(state[base + 3u], 1u);
    InterlockedAdd(state[base + (frustumVisible ? 4u : 5u)], 1u);
}

GPUDrivenStreamAssetActiveGroup streamBlasGroup(uint index, GPUDrivenStreamAssetParams params)
{
    StructuredBuffer<uint4> groups = resolveUniform(
        ResourceHandle<StructuredBuffer<uint4>>(params.rtRepresentationGroupBuffer));
    uint4 key = groups[index];
    GPUDrivenStreamAssetActiveGroup group = (GPUDrivenStreamAssetActiveGroup)0;
    group.pageIndex = key.x;
    group.clusterSelectionMask = key.y;
    group.clusterCount = key.z;
    group.instanceIndex = key.w;
    return group;
}

public groupshared uint g_RTCacheMissing;

uint4 scanRTRepresentationCounts(uint4 demand, uint lane)
{
    g_StreamBlasStoragePrefix[lane] = demand;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 64u; stride *= 2u) {
        uint4 prior = lane >= stride ? g_StreamBlasStoragePrefix[max(lane, stride) - stride] : uint4(0u);
        GroupMemoryBarrierWithGroupSync();
        uint4 current = g_StreamBlasStoragePrefix[lane];
        g_StreamBlasStoragePrefix[lane] = current + min(prior, 0xffffffffu - current);
        GroupMemoryBarrierWithGroupSync();
    }
    return lane == 0u ? uint4(0u) : g_StreamBlasStoragePrefix[lane - 1u];
}

