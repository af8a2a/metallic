#include "../../Libraries/Lighting/ClusterLightGridCommon.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<ClusterLightGridParams> gClusterParams;

[[vk::binding(1, 0)]]
StructuredBuffer<ClusterLightData> gClusterLights;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> gClusterCandidates;

[[vk::binding(3, 0)]]
RWStructuredBuffer<ClusterLightGridCell> gClusterCells;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> gClusterLightIndices;

static const uint kClusterLightGridThreadCount = 64u;
groupshared uint gClusterIntersectionCount;
groupshared float3 gClusterCellCenter;
groupshared float3 gClusterCellExtent;

struct ClusterCellBounds
{
    float3 center;
    float3 extent;
};

ClusterCellBounds clusterLightGridCellBounds(ClusterLightGridParams params, uint3 cell)
{
    // Clamp partial edge tiles to the actual viewport. View-space +Y points up,
    // while pixel Y grows downward; the camera's forward direction is positive Z.
    uint2 pixelMin = cell.xy * params.grid.w;
    uint2 pixelMax = min((cell.xy + 1u) * params.grid.w, params.viewport.xy);
    float2 unitMin = float2(pixelMin) / float2(params.viewport.xy) * 2.0 - 1.0;
    float2 unitMax = float2(pixelMax) / float2(params.viewport.xy) * 2.0 - 1.0;
    unitMin.y = -unitMin.y;
    unitMax.y = -unitMax.y;
    float nearDepth = clusterLightGridSliceDepth(params, cell.z);
    float farDepth = clusterLightGridSliceDepth(params, cell.z + 1u);
    float3 boundsMin = float3(3.402823466e+38);
    float3 boundsMax = -boundsMin;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float depth = (corner & 4u) != 0u ? farDepth : nearDepth;
        float2 unit = float2((corner & 1u) != 0u ? unitMax.x : unitMin.x,
            (corner & 2u) != 0u ? unitMax.y : unitMin.y);
        float scale = params.viewport.w != 0u ? 1.0 : depth;
        float3 point = float3(unit * float2(params.forwardExtent.w, params.upExtent.w) * scale,
            depth);
        boundsMin = min(boundsMin, point);
        boundsMax = max(boundsMax, point);
    }
    ClusterCellBounds result;
    result.center = boundsMin * 0.5 + boundsMax * 0.5;
    result.extent = boundsMax * 0.5 - boundsMin * 0.5;
    return result;
}

float3 clusterLightGridToViewVector(ClusterLightGridParams params, float3 vector)
{
    return float3(dot(vector, params.rightFar.xyz), dot(vector, params.upExtent.xyz),
        dot(vector, params.forwardExtent.xyz));
}

bool clusterLightGridOverlaps(
    ClusterLightGridParams params, ClusterCellBounds bounds, ClusterLightData light)
{
    float3 position = clusterLightGridToViewVector(params, light.positionRange.xyz - params.eyeNear.xyz);
    float range = light.positionRange.w;
    // Source validation normally guarantees finite values. Keep uncertain bounds
    // conservatively instead of rejecting a light through overflow/NaN arithmetic.
    if (!all(isfinite(position)) || !all(isfinite(bounds.center)) ||
        !all(isfinite(bounds.extent)) || !isfinite(range)) {
        return true;
    }
    float3 delta = bounds.center - position;
    float tolerance = 1e-5 * max(1.0, max(range,
        max(max(abs(delta.x), abs(delta.y)), max(abs(delta.z), max(bounds.extent.x,
            max(bounds.extent.y, bounds.extent.z))))));
    float3 distance = max(abs(delta) - bounds.extent, 0.0);
    float expandedRange = range + tolerance;
    if (dot(distance, distance) > expandedRange * expandedRange) {
        return false;
    }

    // Unreal's conservative cone/AABB separating-plane refinement. The range
    // sphere already bounds the cone's length. Very wide cones use only the sphere.
    if (light.directionType.w > 1.5 && light.spot.y > 1e-4) {
        float3 axis = clusterLightGridToViewVector(params, light.directionType.xyz);
        float axisLengthSq = dot(axis, axis);
        if (!isfinite(axisLengthSq) || axisLengthSq < 1e-12) {
            return true;
        }
        axis *= rsqrt(axisLengthSq);
        float projected = dot(delta, axis);
        float projectedExtent = dot(abs(axis), bounds.extent);
        if (projected + projectedExtent < -tolerance) {
            return false;
        }
        float3 radial = delta - projected * axis;
        float radialLengthSq = dot(radial, radial);
        if (isfinite(radialLengthSq) && radialLengthSq > tolerance * tolerance) {
            float cosine = clamp(light.spot.y, 1e-4, 1.0);
            float tangent = sqrt(max(1.0 - cosine * cosine, 0.0)) / cosine;
            float3 planeNormal = radial * rsqrt(radialLengthSq) - tangent * axis;
            float planeDistance = dot(planeNormal, delta);
            float planeExtent = dot(abs(planeNormal), bounds.extent);
            if (planeDistance > planeExtent + tolerance * (1.0 + tangent)) {
                return false;
            }
        }
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clusterLightGridMain(uint3 groupId : SV_GroupID, uint threadIndex : SV_GroupIndex)
{
    ClusterLightGridParams params = gClusterParams[0];
    if (any(groupId >= params.grid.xyz)) {
        return;
    }
    uint cellIndex = (groupId.z * params.grid.y + groupId.y) * params.grid.x + groupId.x;
    uint offset = cellIndex * params.viewport.z;
    if (threadIndex == 0u) {
        gClusterIntersectionCount = 0u;
        ClusterCellBounds bounds = clusterLightGridCellBounds(params, groupId);
        gClusterCellCenter = bounds.center;
        gClusterCellExtent = bounds.extent;
    }
    GroupMemoryBarrierWithGroupSync();
    ClusterCellBounds bounds;
    bounds.center = gClusterCellCenter;
    bounds.extent = gClusterCellExtent;
    for (uint index = threadIndex; index < params.counts.x; index += kClusterLightGridThreadCount) {
        uint sourceIndex = gClusterCandidates[index];
        if (sourceIndex < params.counts.w &&
            clusterLightGridOverlaps(params, bounds, gClusterLights[sourceIndex])) {
            uint entry;
            InterlockedAdd(gClusterIntersectionCount, 1u, entry);
            if (entry < params.viewport.z) {
                gClusterLightIndices[offset + entry] = sourceIndex;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex == 0u) {
        ClusterLightGridCell cell;
        cell.offset = offset;
        cell.count = min(gClusterIntersectionCount, params.viewport.z);
        cell.overflow = gClusterIntersectionCount > params.viewport.z ? 1u : 0u;
        cell.totalCount = gClusterIntersectionCount;
        gClusterCells[cellIndex] = cell;
    }
}
