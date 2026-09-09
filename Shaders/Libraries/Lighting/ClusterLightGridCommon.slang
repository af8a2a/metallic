#ifndef CLUSTER_LIGHT_GRID_COMMON_SLANG
#define CLUSTER_LIGHT_GRID_COMMON_SLANG

// Shared with ClusterLightGrid.h. All depths are positive camera-space distances,
// independent of the depth attachment's normal/reversed-Z encoding.
struct ClusterLightGridParams
{
    uint4 grid;
    uint4 viewport;
    float4 eyeNear;
    float4 rightFar;
    float4 upExtent;
    float4 forwardExtent;
    float4 zParams;
    uint4 counts;
};

// Same 64-byte physical-light ABI as GpuPunctualLight. The grid's light buffer is
// indexed by stable source slot and does not contain a metadata element at zero.
struct ClusterLightData
{
    float4 positionRange;
    float4 directionType;
    float4 colorIntensity;
    float4 spot;
};

struct ClusterLightGridCell
{
    uint offset;
    uint count;
    uint overflow;
    uint totalCount;
};

static const uint kClusterLightGridInvalidIndex = 0xffffffffu;

float clusterLightGridSliceDepth(ClusterLightGridParams params, uint slice)
{
    if (slice == 0u) {
        return params.eyeNear.w;
    }
    if (slice >= params.grid.z) {
        return params.rightFar.w;
    }
    if (params.viewport.w != 0u) {
        return params.eyeNear.w + float(slice) / params.zParams.z;
    }
    return (exp2(float(slice) / params.zParams.z) - params.zParams.y) /
        params.zParams.x;
}

bool clusterLightGridCellIndex(
    ClusterLightGridParams params, uint2 pixel, float viewDepth, out uint cellIndex)
{
    cellIndex = kClusterLightGridInvalidIndex;
    if (any(params.grid.xyz == 0u) || params.grid.w == 0u ||
        any(pixel >= params.viewport.xy) || !isfinite(viewDepth) ||
        viewDepth < params.eyeNear.w || viewDepth > params.rightFar.w) {
        return false;
    }
    float slice = params.viewport.w != 0u
        ? (viewDepth - params.eyeNear.w) * params.zParams.z
        : log2(viewDepth * params.zParams.x + params.zParams.y) * params.zParams.z;
    if (!isfinite(slice)) {
        return false;
    }
    uint3 cell = uint3(pixel / params.grid.w,
        uint(clamp(floor(slice), 0.0, float(params.grid.z - 1u))));
    if (any(cell >= params.grid.xyz)) {
        return false;
    }
    cellIndex = (cell.z * params.grid.y + cell.y) * params.grid.x + cell.x;
    return true;
}

struct ClusterLightGridLookup
{
    uint offset;
    uint count;
    uint fallback;
};

// The last finite slice is not stretched to an arbitrary world maximum. Shading
// outside the covered frustum/depth range instead uses all bounded candidates.
// Capacity overflow follows the same path, so a crowded cell cannot lose lights.
ClusterLightGridLookup clusterLightGridLookup(
    ClusterLightGridParams params,
    StructuredBuffer<ClusterLightGridCell> cells,
    uint2 pixel,
    float viewDepth)
{
    ClusterLightGridLookup result;
    result.offset = 0u;
    result.count = params.counts.x;
    result.fallback = 1u;
    uint cellIndex;
    if (clusterLightGridCellIndex(params, pixel, viewDepth, cellIndex)) {
        ClusterLightGridCell cell = cells[cellIndex];
        if (cell.overflow == 0u && cell.count <= params.viewport.z &&
            cell.totalCount == cell.count) {
            result.offset = cell.offset;
            result.count = cell.count;
            result.fallback = 0u;
        }
    }
    return result;
}

uint clusterLightGridLocalLightIndex(
    ClusterLightGridLookup lookup,
    StructuredBuffer<uint> lightIndices,
    StructuredBuffer<uint> candidates,
    uint index)
{
    if (index >= lookup.count) {
        return kClusterLightGridInvalidIndex;
    }
    return lookup.fallback != 0u ? candidates[index] : lightIndices[lookup.offset + index];
}

uint clusterLightGridGlobalLightCount(ClusterLightGridParams params)
{
    return params.counts.y + params.counts.z;
}

// Directionals precede unbounded local lights. Both are deliberately outside the
// cell lists and must be evaluated once in addition to the bounded-local lookup.
uint clusterLightGridGlobalLightIndex(
    ClusterLightGridParams params, StructuredBuffer<uint> candidates, uint index)
{
    return index < clusterLightGridGlobalLightCount(params)
        ? candidates[params.counts.x + index] : kClusterLightGridInvalidIndex;
}

#endif // CLUSTER_LIGHT_GRID_COMMON_SLANG
