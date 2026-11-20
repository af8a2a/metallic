module LumenMeshSDFCulling;

import ShaderCore;
import ParameterRoot;
import ScenePathMath;
import MeshDistanceField;
import LumenMeshSDFGrid;

using namespace Metallic;
using namespace Metallic.DistanceField;
using namespace Metallic.Lumen;

struct LumenMeshSDFBounds
{
    float3 center;
    uint objectId;
    float3 extent;
    uint flags;
};

struct LumenMeshSDFCullingParams
{
    ViewCameraConstants camera;
    float4 cull; // Projection jitter in pixels, Mesh SDF influence radius, card trace end distance.
    uint4 grid; // Froxel grid dimensions, Mesh SDF instance count.
    ResourceHandle<Texture2D<float>> depth;
    ResourceHandle<RWStructuredBuffer<float2>> tileDepth;
    ResourceHandle<StructuredBuffer<LumenMeshSDFBounds>> bounds;
    ResourceHandle<StructuredBuffer<MeshDistanceFieldInstance>> instances;
    ResourceHandle<StructuredBuffer<MeshDistanceFieldMip>> mips;
    ResourceHandle<StructuredBuffer<uint>> indirection;
    ResourceHandle<Texture3D<float>> atlas;
    ResourceHandle<SamplerState> sampler;
    ResourceHandle<RWByteAddressBuffer> gridCells;
    ResourceHandle<RWStructuredBuffer<uint>> gridIndices;
    ResourceHandle<RWStructuredBuffer<uint2>> gridPairs;
    ResourceHandle<RWStructuredBuffer<uint>> gridState;
};

static const uint kPairCount = 0u;
static const uint kIndexAllocator = 1u;
static const uint kGridStateWords = 4u;
static const uint kGridCellStride = 8u;
static const uint kGridCellCountOffset = 4u;
static const uint kEmissiveMeshSDF = 1u;
static const float kMeshSDFRadiusThreshold = 0.30f;
static const uint kThreads = 64u;

groupshared float2 gTileDepth[kThreads];

uint meshSDFGridCellCount(uint3 grid)
{
    return grid.x * grid.y * grid.z;
}

// UE 5.7.4 HZB.usf RoundUpF16.
float roundUpClosestDeviceZ(float deviceZ)
{
    return f16tof32(f32tof16(deviceZ) + 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenMeshSDFGridClearMain(uint3 dispatch : SV_DispatchThreadID)
{
    LumenMeshSDFCullingParams params = getParameters<LumenMeshSDFCullingParams>();
    if (dispatch.x < kGridStateWords) {
        resolveUniform(params.gridState)[dispatch.x] = 0u;
    }
    if (dispatch.x < meshSDFGridCellCount(params.grid.xyz)) {
        resolveUniform(params.gridCells).Store2(dispatch.x * kGridCellStride, uint2(0u, 0u));
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenMeshSDFGridDepthMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID, uint laneIndex : SV_GroupIndex)
{
    LumenMeshSDFCullingParams params = getParameters<LumenMeshSDFCullingParams>();
    Texture2D<float> depth = resolveUniform(params.depth);
    uint2 extent = uint2(params.camera.viewport.yz);
    // UE 5.7.4 SceneTextureReductions.cpp:143 and LumenMeshSDFCulling.cpp:769.
    uint maxExtent = max(max(extent.x, extent.y), 2u);
    uint mipCount = max(uint(firstbithigh(maxExtent - 1u)), 1u);
    uint mip = min(kLumenMeshSDFGridPixelSizeShift - 1u, mipCount - 1u);
    uint footprint = 2u << mip;
    uint2 tileStart = group.xy << kLumenMeshSDFGridPixelSizeShift;
    float closest = 0.0f;
    float furthest = 1.0f;
    for (uint y = lane.y; y < footprint; y += 8u) {
        for (uint x = lane.x; x < footprint; x += 8u) {
            uint2 source = tileStart + uint2(x, y);
            // HZB.usf Gather4 clamps to the last in-viewport pair before loading.
            uint2 pair = min(source & ~1u, max(extent, 2u) - 2u);
            uint2 pixel = min(pair + (source & 1u), extent - 1u);
            float sampleDepth = depth.Load(int3(pixel, 0));
            closest = max(closest, sampleDepth);
            furthest = min(furthest, sampleDepth);
        }
    }
    gTileDepth[laneIndex] = float2(closest, furthest);
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = kThreads / 2u; stride > 0u; stride >>= 1u) {
        if (laneIndex < stride) {
            float2 sampleDepth = gTileDepth[laneIndex + stride];
            gTileDepth[laneIndex] = float2(max(gTileDepth[laneIndex].x, sampleDepth.x),
                min(gTileDepth[laneIndex].y, sampleDepth.y));
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (laneIndex == 0u) {
        // Closest HZB rounds once when the next four-mip batch reads its R16F parent.
        float2 depthRange = gTileDepth[0];
        if (mip >= 4u) {
            depthRange.x = roundUpClosestDeviceZ(depthRange.x);
        }
        depthRange.x = roundUpClosestDeviceZ(depthRange.x);
        depthRange.y = f16tof32(f32tof16(depthRange.y));
        resolveUniform(params.tileDepth)[group.y * params.grid.x + group.x] = depthRange;
    }
}

struct MeshSDFCullView
{
    float3 eye;
    float3 right;
    float3 up;
    float3 forward;
    float2 halfExtent;
    float2 projectionOffset;
    float2 extent;
    float nearPlane;
    float farPlane;
};

MeshSDFCullView makeCullView(ViewCameraConstants camera, float2 jitter)
{
    MeshSDFCullView result;
    result.eye = camera.eye.xyz;
    result.forward = ScenePath.cameraForwardFrom(camera.eye, camera.center);
    result.right = ScenePath.cameraRightFrom(result.forward, camera.upProjection.xyz);
    result.up = ScenePath.cameraUpFrom(result.forward, result.right);
    result.halfExtent = float2(camera.viewport.x, 1.0f) * tan(camera.viewport.w * 0.5f);
    result.extent = camera.viewport.yz;
    result.projectionOffset = -(jitter / result.extent * float2(2.0f, -2.0f));
    result.nearPlane = camera.clipOrtho.x;
    result.farPlane = camera.clipOrtho.y;
    return result;
}

// UE 5.7.4 LumenMeshSDFCulling.usf:428.
void makeFrustumSidePlanes(float2 inverseHalfExtent, float2 projectionOffset, float2 tileScale, float2 tileBias,
    out float4 frustumPlanes[6])
{
    float4 c1 = float4(inverseHalfExtent.x * tileScale.x, 0.0f, projectionOffset.x * tileScale.x + tileBias.x, 0.0f);
    float4 c2 = float4(0.0f, -inverseHalfExtent.y * tileScale.y, projectionOffset.y * tileScale.y + tileBias.y, 0.0f);
    float4 c4 = float4(0.0f, 0.0f, 1.0f, 0.0f);
    frustumPlanes[0] = c4 - c1;
    frustumPlanes[1] = c4 + c1;
    frustumPlanes[2] = c4 - c2;
    frustumPlanes[3] = c4 + c2;
    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        frustumPlanes[i] *= rcp(length(frustumPlanes[i].xyz));
    }
}

bool meshSDFBoundsIntersectView(MeshSDFCullView view, float3 viewPosition, float radius)
{
    float4 cameraPlanes[6];
    // View.ViewFrustum is unjittered and excludes the near plane.
    makeFrustumSidePlanes(rcp(view.halfExtent), float2(0.0f), float2(1.0f), float2(0.0f), cameraPlanes);
    bool intersects = viewPosition.z <= view.farPlane + radius;
    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        intersects = intersects && dot(cameraPlanes[i].xyz, viewPosition) >= -radius;
    }
    return intersects;
}

float3 meshSDFVolumeWorldExtent(MeshDistanceFieldInstance instance)
{
    // The uploaded inverse basis is orthogonal with a uniform scale. Multiplying
    // its transpose by the squared world scale recovers the volume-to-world basis.
    float4 extentAndScale = instance.volumeExtentAndWorldScale;
    float scaleSquared = extentAndScale.w * extentAndScale.w;
    return (extentAndScale.x * abs(instance.worldToVolume0.xyz) +
        extentAndScale.y * abs(instance.worldToVolume1.xyz) +
        extentAndScale.z * abs(instance.worldToVolume2.xyz)) * scaleSquared;
}

float meshSDFConvertFromDeviceZ(MeshSDFCullView view, float deviceZ)
{
    return view.nearPlane * view.farPlane / (view.nearPlane + deviceZ * (view.farPlane - view.nearPlane));
}

// UE 5.7.4 LumenMeshSDFCulling.usf:366, perspective ClipToTranslatedWorld.
float3 meshSDFGridCellWorldPosition(MeshSDFCullView view, uint3 gridSize, uint3 gridCoordinate, float3 cellOffset)
{
    float2 volumeUV = (float2(gridCoordinate.xy) + cellOffset.xy) / float2(gridSize.xy);
    float2 volumeNDC = (volumeUV * 2.0f - 1.0f) * float2(1.0f, -1.0f);
    float sceneDepth = lumenMeshSDFGridSliceDepth(float(gridCoordinate.z) + cellOffset.z);
    float2 offset = (volumeNDC - view.projectionOffset) * view.halfExtent * sceneDepth;
    return view.eye + view.forward * sceneDepth + view.right * offset.x + view.up * offset.y;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenMeshSDFGridScatterMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenMeshSDFCullingParams params = getParameters<LumenMeshSDFCullingParams>();
    uint objectIndex = group.z * kThreads + lane.x;
    if (objectIndex >= params.grid.w) {
        return;
    }

    LumenMeshSDFBounds bounds = resolveUniform(params.bounds)[objectIndex];
    MeshSDFCullView view = makeCullView(params.camera, params.cull.xy);
    float influenceRadius = params.cull.z;
    float objectRadius = length(bounds.extent);
    float3 eyeRelativePosition = bounds.center - view.eye;
    float3 viewPosition = float3(dot(eyeRelativePosition, view.right), dot(eyeRelativePosition, view.up),
        dot(eyeRelativePosition, view.forward));
    float maxDistance = params.cull.w + objectRadius;
    // UE 5.7.4 CullMeshSDFObjectsForViewCS. Instance upload filters visibility.
    if (!(dot(eyeRelativePosition, eyeRelativePosition) < maxDistance * maxDistance &&
            (objectRadius > kMeshSDFRadiusThreshold || (bounds.flags & kEmissiveMeshSDF) != 0u) &&
            meshSDFBoundsIntersectView(view, viewPosition, objectRadius + influenceRadius))) {
        return;
    }

    MeshDistanceFieldInstance instance = resolveUniform(params.instances)[objectIndex];
    float3 worldExtent = meshSDFVolumeWorldExtent(instance);
    float3 influenceExtent = float3(dot(worldExtent, abs(view.right)), dot(worldExtent, abs(view.up)),
        dot(worldExtent, abs(view.forward))) + influenceRadius;
    float4 frustumPlanes[6];
    float2 tileScale = view.extent * rcp(2.0f * float(kLumenMeshSDFGridPixelSize));
    float2 tileBias = tileScale - float2(group.xy) - 0.5f;
    makeFrustumSidePlanes(rcp(view.halfExtent), view.projectionOffset, tileScale, tileBias, frustumPlanes);
    float2 deviceDepth = resolveUniform(params.tileDepth)[group.y * params.grid.x + group.x];
    float minDepth = max(viewPosition.z - influenceExtent.z, meshSDFConvertFromDeviceZ(view, deviceDepth.x));
    float maxDepth = min(viewPosition.z + influenceExtent.z, meshSDFConvertFromDeviceZ(view, deviceDepth.y));
    uint minFroxelZ = lumenMeshSDFGridDepthSlice(minDepth, params.grid.z);
    uint maxFroxelZ = lumenMeshSDFGridDepthSlice(maxDepth, params.grid.z);

    Texture3D<float> atlas = resolveUniform(params.atlas);
    uint width, height, depth;
    atlas.GetDimensions(width, height, depth);
    MeshDistanceFieldResources meshes = {resolveUniform(params.mips), resolveUniform(params.indirection), atlas,
        resolveUniform(params.sampler), width / 8u};
    RWByteAddressBuffer cells = resolveUniform(params.gridCells);
    RWStructuredBuffer<uint> state = resolveUniform(params.gridState);
    RWStructuredBuffer<uint2> pairs = resolveUniform(params.gridPairs);
    for (uint froxelZ = minFroxelZ; froxelZ <= maxFroxelZ; ++froxelZ) {
        uint3 gridCoordinate = uint3(group.xy, froxelZ);
        float minTileZ = lumenMeshSDFGridSliceDepth(float(froxelZ));
        float maxTileZ = lumenMeshSDFGridSliceDepth(float(froxelZ + 1u));
        frustumPlanes[4] = float4(0.0f, 0.0f, 1.0f, -minTileZ);
        frustumPlanes[5] = float4(0.0f, 0.0f, -1.0f, maxTileZ);

        bool intersectsTile = true;
        [unroll]
        for (uint i = 0u; i < 6u; ++i) {
            float distanceFromPlane = dot(frustumPlanes[i], float4(viewPosition, 1.0f));
            float radius = dot(influenceExtent, abs(frustumPlanes[i].xyz));
            intersectsTile = intersectsTile && (distanceFromPlane >= -radius);
        }
        if (intersectsTile) {
            float3 worldTileCenter = meshSDFGridCellWorldPosition(view, params.grid.xyz, gridCoordinate, float3(0.5f));
            float3 worldTileCorner = meshSDFGridCellWorldPosition(view, params.grid.xyz, gridCoordinate, float3(1.0f));
            float cornerLength = length(worldTileCorner - worldTileCenter);
            worldTileCorner = meshSDFGridCellWorldPosition(view, params.grid.xyz, gridCoordinate, float3(0.0f, 0.0f, 1.0f));
            cornerLength = max(length(worldTileCorner - worldTileCenter), cornerLength);
            float tileBoundingSphereRadius = cornerLength + influenceRadius;
            float distanceToSurface = distanceToNearestMeshSurface(meshes, instance, worldTileCenter, tileBoundingSphereRadius);
            intersectsTile = distanceToSurface < tileBoundingSphereRadius;
        }
        if (intersectsTile) {
            uint gridIndex = lumenMeshSDFGridIndex(gridCoordinate, params.grid.xyz);
            cells.InterlockedAdd(gridIndex * kGridCellStride + kGridCellCountOffset, 1u);
            uint pairIndex;
            InterlockedAdd(state[kPairCount], 1u, pairIndex);
            // The CPU allocates for one pair for every combination of object and cell.
            pairs[pairIndex] = uint2(objectIndex, gridIndex);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenMeshSDFGridAllocateMain(uint3 dispatch : SV_DispatchThreadID)
{
    LumenMeshSDFCullingParams params = getParameters<LumenMeshSDFCullingParams>();
    if (dispatch.x >= meshSDFGridCellCount(params.grid.xyz)) {
        return;
    }
    RWByteAddressBuffer cells = resolveUniform(params.gridCells);
    uint cellAddress = dispatch.x * kGridCellStride;
    uint startOffset = 0u;
    uint count = cells.Load(cellAddress + kGridCellCountOffset);
    if (count != 0u) {
        RWStructuredBuffer<uint> state = resolveUniform(params.gridState);
        InterlockedAdd(state[kIndexAllocator], count, startOffset);
    }
    // Compaction increments the cleared per-cell count as it writes indices.
    cells.Store2(cellAddress, uint2(startOffset, 0u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenMeshSDFGridCompactMain(uint3 dispatch : SV_DispatchThreadID)
{
    LumenMeshSDFCullingParams params = getParameters<LumenMeshSDFCullingParams>();
    uint pairIndex = dispatch.x + dispatch.y * (65535u * kThreads);
    uint pairCount = resolveUniform(params.gridState)[kPairCount];
    if (pairIndex >= pairCount) {
        return;
    }
    uint2 pair = resolveUniform(params.gridPairs)[pairIndex];
    RWByteAddressBuffer cells = resolveUniform(params.gridCells);
    uint cellAddress = pair.y * kGridCellStride;
    // A scalar offset read must not also read the count being updated by other lanes.
    uint startOffset = cells.Load(cellAddress);
    uint indexInCell;
    cells.InterlockedAdd(cellAddress + kGridCellCountOffset, 1u, indexInCell);
    resolveUniform(params.gridIndices)[startOffset + indexInCell] = pair.x;
}
