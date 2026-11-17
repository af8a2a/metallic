#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Camera projection, LOD metrics, instance visibility and HZB generation.
float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

bool streamPayloadRangeValid(
    uint byteOffset,
    uint byteCount,
    uint payloadBytes)
{
    return byteOffset <= payloadBytes && byteCount <= payloadBytes - byteOffset;
}

bool streamPayloadArrayRangeValid(
    uint byteOffset,
    uint elementCount,
    uint elementStride,
    uint payloadBytes)
{
    return elementStride != 0u &&
        byteOffset <= payloadBytes &&
        elementCount <= (payloadBytes - byteOffset) / elementStride;
}

float3 transformStreamDirection(
    GPUDrivenStreamAssetActiveGroup group,
    float3 localDirection)
{
    return group.world0.xyz * localDirection.x +
        group.world1.xyz * localDirection.y +
        group.world2.xyz * localDirection.z;
}

public float3 transformStreamNormal(
    GPUDrivenStreamAssetActiveGroup group,
    float3 localNormal,
    float3 fallback)
{
    float3 cofactor0 = cross(group.world1.xyz, group.world2.xyz);
    float3 cofactor1 = cross(group.world2.xyz, group.world0.xyz);
    float3 cofactor2 = cross(group.world0.xyz, group.world1.xyz);
    float3 normal = cofactor0 * localNormal.x +
        cofactor1 * localNormal.y + cofactor2 * localNormal.z;
    // Cofactors scale quadratically: Sponza's 0.008 scale produces a valid
    // normal below safeNormalize's length threshold. Normalize the direction
    // independently of that scale, preserving the cofactor winding sign.
    float maxComponent = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
    return maxComponent > 0.0 && isfinite(maxComponent)
        ? normalize(normal / maxComponent) : fallback;
}

float4 projectPosition(float3 worldPosition, GPUDrivenStreamAssetParams params)
{
    if (params.renderViewport.y > 0.0) {
        params.eye = params.renderEye;
        params.center = params.renderCenter;
        params.upProjection = params.renderUpProjection;
        params.viewport = params.renderViewport;
        params.clipOrtho = params.renderClipOrtho;
    }
    float3 forward = safeNormalize(params.center.xyz - params.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(params.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));

    float3 cameraVector = worldPosition - params.eye.xyz;
    float3 viewPosition = float3(
        dot(cameraVector, right),
        dot(cameraVector, up),
        dot(cameraVector, forward));

    float zNear = max(params.clipOrtho.x, 0.0001);
    float zFar = max(params.clipOrtho.y, zNear + 0.0001);
    float ndcZ = conservativeProjectDepth(viewPosition.z, zNear, zFar,
        params.upProjection.w > 0.5);
    float aspect = max(params.viewport.x, 0.001);

    if (params.upProjection.w > 0.5) {
        float halfHeight = max(params.clipOrtho.z * 0.5, 0.0001);
        float4 clip = float4(
            viewPosition.x / (halfHeight * aspect),
            viewPosition.y / halfHeight,
            ndcZ,
            1.0);
        if (params.renderViewport.y > 0.0) {
            clip.xy -= 2.0 * float2(params.renderEye.w, -params.renderCenter.w) / params.renderViewport.yz * clip.w;
        }
        return clip;
    }

    float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = viewPosition.z;
    float4 clip = float4(
        viewPosition.x * yScale / aspect,
        viewPosition.y * yScale,
        ndcZ * w,
        w);
    if (params.renderViewport.y > 0.0) {
        clip.xy -= 2.0 * float2(params.renderEye.w, -params.renderCenter.w) / params.renderViewport.yz * clip.w;
    }
    return clip;
}

public struct GPUDrivenStreamAssetCameraData
{
    public float4 eye;
    public float4 center;
    public float4 upProjection;
    public float4 viewport;
    public float4 clipOrtho;
};

GPUDrivenStreamAssetCameraData currentStreamCamera(GPUDrivenStreamAssetParams params)
{
    GPUDrivenStreamAssetCameraData camera;
    camera.eye = params.eye;
    camera.center = params.center;
    camera.upProjection = params.upProjection;
    camera.viewport = params.viewport;
    camera.clipOrtho = params.clipOrtho;
    return camera;
}

GPUDrivenStreamAssetCameraData previousStreamCamera(GPUDrivenStreamAssetParams params)
{
    GPUDrivenStreamAssetCameraData camera;
    camera.eye = params.previousEye;
    camera.center = params.previousCenter;
    camera.upProjection = params.previousUpProjection;
    camera.viewport = params.previousViewport;
    camera.clipOrtho = params.previousClipOrtho;
    return camera;
}

float3 streamWorldToView(float3 worldPosition, GPUDrivenStreamAssetCameraData camera)
{
    float3 forward = safeNormalize(camera.center.xyz - camera.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(camera.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 cameraVector = worldPosition - camera.eye.xyz;
    return float3(
        dot(cameraVector, right),
        dot(cameraVector, up),
        dot(cameraVector, forward));
}

float streamProjectDepth(float viewZ, GPUDrivenStreamAssetCameraData camera)
{
    float zNear = max(camera.clipOrtho.x, 0.0001);
    float zFar = max(camera.clipOrtho.y, zNear + 0.0001);
    return conservativeProjectDepth(viewZ, zNear, zFar,
        camera.upProjection.w > 0.5);
}

bool streamSphereInsideFrustum(
    float3 center,
    float radius,
    GPUDrivenStreamAssetCameraData camera)
{
    float3 viewCenter = streamWorldToView(center, camera);
    float zNear = max(camera.clipOrtho.x, 0.0001);
    float zFar = max(camera.clipOrtho.y, zNear + 0.0001);
    if (viewCenter.z + radius < zNear || viewCenter.z - radius > zFar) {
        return false;
    }
    float aspect = max(camera.viewport.x, 0.001);
    if (camera.upProjection.w > 0.5) {
        float halfHeight = max(camera.clipOrtho.z * 0.5, 0.0001);
        float halfWidth = halfHeight * aspect;
        return abs(viewCenter.x) <= halfWidth + radius &&
            abs(viewCenter.y) <= halfHeight + radius;
    }
    if (viewCenter.z + radius <= 0.0) {
        return false;
    }
    float tanY = tan(clamp(camera.viewport.w, 0.017453292, 3.12413936) * 0.5);
    float tanX = tanY * aspect;
    float radiusX = radius * sqrt(1.0 + tanX * tanX);
    float radiusY = radius * sqrt(1.0 + tanY * tanY);
    return abs(viewCenter.x) <= viewCenter.z * tanX + radiusX &&
        abs(viewCenter.y) <= viewCenter.z * tanY + radiusY;
}

bool streamDemandBoundsVisible(float3 low, float3 high, GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetParams params)
{
    if (any(low > high) || !all(isfinite(low)) || !all(isfinite(high))) { return true; }
    float3 center = low * 0.5 + high * 0.5, extent = high * 0.5 - low * 0.5;
    float3 a = instance.world0.xyz, b = instance.world1.xyz, c = instance.world2.xyz;
    float3 delta = a * center.x + b * center.y + c * center.z + instance.world3.xyz - params.eye.xyz;
    float3 forward = safeNormalize(params.center.xyz - params.eye.xyz, float3(0,0,-1));
    float3 right = safeNormalize(cross(forward, params.upProjection.xyz), float3(1,0,0));
    float3 up = cross(right, forward);
    float3 magnitude = abs(a) * (abs(center.x) + extent.x) + abs(b) * (abs(center.y) + extent.y) +
        abs(c) * (abs(center.z) + extent.z) + abs(instance.world3.xyz) + abs(params.eye.xyz);
    if (!all(isfinite(delta)) || !isfinite(dot(magnitude, float3(1, 1, 1)))) { return true; }
    float padding = dot(magnitude, float3(1,1,1)) * (32.0 * 1.1920928955078125e-7);
    float z = dot(delta, forward);
    float rz = dot(abs(float3(dot(a, forward), dot(b, forward), dot(c, forward))), extent) + padding;
    if (z + rz < params.clipOrtho.x || z - rz > params.clipOrtho.y) { return false; }
    // One pixel guard plus transform roundoff. Demand must be conservative at
    // viewport edges even while temporal jitter moves the raster samples.
    float guard = 1.0 + 2.0 / max(min(params.viewport.y, params.viewport.z), 1.0);
    float ty = tan(params.viewport.w * 0.5) * guard, tx = ty * params.viewport.x;
    if (params.upProjection.w != 0.0) {
        float halfHeight = params.clipOrtho.z * 0.5 * guard;
        float rx = dot(abs(float3(dot(a, right), dot(b, right), dot(c, right))), extent) + padding;
        float ry = dot(abs(float3(dot(a, up), dot(b, up), dot(c, up))), extent) + padding;
        return abs(dot(delta, right)) <= halfHeight * params.viewport.x + rx && abs(dot(delta, up)) <= halfHeight + ry;
    }
    float3 planes[4] = {forward * tx + right, forward * tx - right, forward * ty + up, forward * ty - up};
    for (uint i = 0u; i < 4u; ++i) {
        float3 normal = planes[i];
        float radius = dot(abs(float3(dot(a, normal), dot(b, normal), dot(c, normal))), extent);
        if (dot(delta, normal) + radius + padding * length(normal) < 0.0) { return false; }
    }
    return true;
}

bool streamRefinementDemandVisible(GPUDrivenStreamAssetGroup group, GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetParams params)
{
    if (params.enableGpuLodSelection == 0u || (group.flags & 1u) != 0u) { return true; }
    return streamDemandBoundsVisible(float3(group.demandMinX, group.demandMinY, group.demandMinZ),
        float3(group.demandMaxX, group.demandMaxY, group.demandMaxZ), instance, params);
}

bool streamRayLodEnabled(GPUDrivenStreamAssetParams params)
{
    return (params.rtLodFlags & 1u) != 0u;
}

uint streamRayLodHistoryBase(uint base, uint groupCount)
{
    return base + groupCount * 3u + 4u;
}

void prepareStreamRayLodHistory(RWStructuredBuffer<uint> state, uint base, uint groupCount,
    bool visible, GPUDrivenStreamAssetParams params)
{
    if (!streamRayLodEnabled(params)) { return; }
    uint history = streamRayLodHistoryBase(base, groupCount);
    bool eligible = visible && params.enableGpuLodSelection != 0u;
    state[history] = eligible && state[history + 2u] == params.rtLodPolicyRevision ? state[history + 1u] : 0u;
    state[history + 1u] = params.rtLodFrameToken;
    state[history + 2u] = params.rtLodPolicyRevision;
}

float streamLodMetric(float4 sphere, float error, GPUDrivenStreamAssetInstance instance,
    float4 eye, float4 forward, float4 projection, GPUDrivenStreamAssetParams params,
    bool visible, float padding = 0.0)
{
    if (!streamRayLodEnabled(params)) {
        return meshletLodProjectedError(sphere, error, instance.world0, instance.world1,
            instance.world2, instance.world3, eye, forward, projection, padding);
    }
    float projected = meshletLodRayProjectedError(sphere, error, instance.world0, instance.world1,
        instance.world2, instance.world3, eye, forward, projection, padding);
    if (visible && (params.rtLodFlags & 2u) != 0u) {
        projected = max(projected, meshletLodProjectedError(sphere, error, instance.world0, instance.world1,
            instance.world2, instance.world3, eye, forward, projection, padding));
    }
    return projected;
}

// Demand owns hysteresis. Parent activation, CLAS readiness and the complete
// frontier still decide which complementary cluster masks can be published.
bool streamRayLodWasWanted(RWStructuredBuffer<uint> state, uint history, uint address)
{
    uint previousToken = state[history];
    if (previousToken == 0u || state[address] != previousToken) { return false; }
    return state[address + 1u] != 0u;
}

void publishStreamRayLodDemand(RWStructuredBuffer<uint> state, uint address, bool wanted,
    bool updateHistory, GPUDrivenStreamAssetParams params)
{
    if (!updateHistory || !streamRayLodEnabled(params)) { return; }
    // Tokens are unique across executed traversals, including cancelled CPU
    // frame gaps. A pruned group expires when the instance header advances.
    // Each group has one authoritative demand writer per token; the frontier
    // consumes its demand bit rather than evaluating and overwriting it again.
    state[address] = params.rtLodFrameToken;
    state[address + 1u] = wanted ? 1u : 0u;
}

bool streamLodGroupDemand(GPUDrivenStreamAssetGroup group, GPUDrivenStreamAssetInstance instance,
    float4 eye, float4 forward, float4 projection, uint manualLevel, GPUDrivenStreamAssetParams params,
    RWStructuredBuffer<uint> state, uint base, uint groupCount, uint localGroup, bool updateHistory = true)
{
    if ((group.flags & 1u) != 0u) { return true; }
    if (manualLevel != 0xffffffffu) { return group.lodLevel >= manualLevel; }
    bool rayEnabled = streamRayLodEnabled(params);
    uint history = streamRayLodHistoryBase(base, groupCount);
    uint address = history + 3u + localGroup * 2u;
    float4 sphere = float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius);
    bool visible = streamRefinementDemandVisible(group, instance, params);
    if (!rayEnabled && !visible) { return false; }
    float threshold = projection.w;
    if (rayEnabled) {
        threshold *= visible ? 1.0 : max(params.rtOffscreenErrorScale, 1.0);
    }
    float error = streamLodMetric(sphere, group.maxQuadricError, instance, eye, forward, projection, params, visible);
    bool wanted = error > threshold;
    if (rayEnabled && updateHistory && !wanted &&
        error > threshold * (1.0 - clamp(params.rtLodHysteresis, 0.0, 0.9))) {
        // Explicit > preserves equality and NaN behavior; an unordered error
        // must not fall through to stale retained demand.
        wanted = streamRayLodWasWanted(state, history, address);
    }
    publishStreamRayLodDemand(state, address, wanted, updateHistory, params);
    return wanted;
}

// A scheduling heuristic, not a replacement for the conservative LOD cut.
// Offscreen dependencies remain requested with zero benefit. Shared pages use
// the maximum instance benefit, independently of request-list deduplication.
float streamGroupScreenBenefit(GPUDrivenStreamAssetGroup group, GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetParams params, float4 eye, float4 direction, float4 projection)
{
    float3 a = instance.world0.xyz, b = instance.world1.xyz, c = instance.world2.xyz;
    float ab = abs(dot(a, b)), ac = abs(dot(a, c)), bc = abs(dot(b, c));
    float scale = sqrt(max(dot(a, a) + ab + ac, max(dot(b, b) + ab + bc, dot(c, c) + ac + bc)));
    float4 sphere = float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius);
    float3 center = a * sphere.x + b * sphere.y + c * sphere.z + instance.world3.xyz;
    float radius = (sphere.w + group.maxQuadricError) * scale;
    if (!streamSphereInsideFrustum(center, radius, currentStreamCamera(params))) { return 0.0; }
    float pixelsPerUnit = direction.w != 0.0 ? projection.x / max(projection.z, 1e-6) :
        projection.x / max(2.0 * projection.y, 1e-6) /
            max(dot(center - eye.xyz, direction.xyz) - radius, max(eye.w, 1e-6));
    float pixelRadius = min(radius * pixelsPerUnit, max(params.viewport.y, params.viewport.z));
    float area = min(3.14159265 * pixelRadius * pixelRadius, params.viewport.y * params.viewport.z);
    float error = meshletLodProjectedError(sphere, group.maxQuadricError, instance.world0, instance.world1,
        instance.world2, instance.world3, eye, direction, projection);
    float excess = clamp(error / max(projection.w, 1e-6) - 1.0, 1.0, 64.0);
    return clamp(max(area, 1.0) * excess, 0.0, 1e9);
}

uint streamMipDimension(uint baseDimension, uint mipLevel)
{
    uint dimension = max(baseDimension, 1u);
    for (uint level = 0u; level < mipLevel; ++level) {
        dimension = max(1u, (dimension + 1u) / 2u);
    }
    return dimension;
}

uint streamHzbMipOffset(
    GPUDrivenStreamAssetRasterBindings bindings,
    uint mipLevel)
{
    uint offset = 0u;
    uint width = max(bindings.width, 1u);
    uint height = max(bindings.height, 1u);
    for (uint level = 0u; level < mipLevel; ++level) {
        offset += width * height;
        width = max(1u, (width + 1u) / 2u);
        height = max(1u, (height + 1u) / 2u);
    }
    return offset;
}

float loadStreamHzb(
    GPUDrivenStreamAssetPush push,
    GPUDrivenStreamAssetRasterBindings bindings,
    uint hzbIndex,
    uint elementIndex)
{
    uint handleIndex = hzbIndex == 0u ? bindings.hzbBuffer0 : bindings.hzbBuffer1;
    StructuredBuffer<float> hzbHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<float>>(handleIndex));
    StructuredBuffer<float> hzb = hzbHandle;
    return hzb[elementIndex];
}

bool streamSphereOccluded(
    float3 center,
    float radius,
    GPUDrivenStreamAssetCameraData camera,
    GPUDrivenStreamAssetRasterBindings bindings,
    GPUDrivenStreamAssetPush push,
    uint hzbIndex)
{
    if (bindings.hzbMipCount == 0u) {
        return false;
    }
    float3 viewCenter = streamWorldToView(center, camera);
    float zNear = max(camera.clipOrtho.x, 0.0001);
    float4 rect;
    if (!conservativeSphereScreenRect(viewCenter, radius, float2(bindings.width, bindings.height),
        max(camera.viewport.x, 0.001), clamp(camera.viewport.w, 0.017453292, 3.12413936),
        camera.upProjection.w > 0.5 ? max(camera.clipOrtho.z, 0.0002) : 0.0, zNear, rect)) {
        return false;
    }
    float2 minPixels = rect.xy, maxPixels = rect.zw;
    float maxExtent = max(maxPixels.x - minPixels.x, maxPixels.y - minPixels.y);
    uint mipLevel = min(
        bindings.hzbMipCount - 1u,
        maxExtent > 1.0 ? uint(ceil(log2(maxExtent))) : 0u);
    uint mipWidth = streamMipDimension(bindings.width, mipLevel);
    uint mipHeight = streamMipDimension(bindings.height, mipLevel);
    float mipScale = exp2(-float(mipLevel));
    uint2 minTexel = min(uint2(minPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    uint2 maxTexel = min(uint2(maxPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    if (any(maxTexel - minTexel > 1u)) { return false; }
    uint mipOffset = streamHzbMipOffset(bindings, mipLevel);
    float depth00 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + minTexel.y * mipWidth + minTexel.x);
    float depth10 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + minTexel.y * mipWidth + maxTexel.x);
    float depth01 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + maxTexel.y * mipWidth + minTexel.x);
    float depth11 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + maxTexel.y * mipWidth + maxTexel.x);
    float nearestZ = max(viewCenter.z - radius, zNear);
    float depthBias = max(0.001, radius * 0.01);
    float nearestDepth = streamProjectDepth(max(nearestZ - depthBias, zNear), camera);
    return conservativeDepthOccluded(nearestDepth, float4(depth00, depth10, depth01, depth11));
}

bool streamClusterVisibleInOcclusionPhase(float3 center, float radius, uint instanceState,
    GPUDrivenStreamAssetParams params, GPUDrivenStreamAssetRasterBindings bindings, GPUDrivenStreamAssetPush push, out bool retryLate)
{
    retryLate = false;
    bool early = push.traversalPhase == 0u;
    if ((bindings.cullingFlags & (1u << 4u)) == 0u) {
        return instanceState == (early ? 1u : 3u);
    }
    if (early ? instanceState != 1u : (instanceState != 1u && instanceState != 3u)) { return false; }
    uint currentHzb = params.frameIndex & 1u;
    bool deferred = instanceState == 3u || (bindings.hzbValid != 0u &&
        streamSphereOccluded(center, radius, previousStreamCamera(params), bindings, push, currentHzb ^ 1u));
    if (early) { retryLate = deferred; return !deferred; }
    return deferred && !streamSphereOccluded(center, radius, currentStreamCamera(params), bindings, push, currentHzb);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetCullResetMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    // Late culling appends the instances recovered by the current-frame HZB to
    // the early-pass list, matching the common GPUScene two-phase contract.
    if (min(push.traversalPhase, 1u) != 0u) {
        return;
    }
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsBuffer = bindingsHandle;
    GPUDrivenStreamAssetRasterBindings bindings = bindingsBuffer[0];
    // The common producer owns both resets in the combined pass. Preserve its
    // already-appended instances when Stream culling follows it.
    if (bindings.instanceVisibilityResetCount == 0u) { return; }
    RWStructuredBuffer<uint> counterHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(bindings.visibleInstanceCounterBuffer));
    RWStructuredBuffer<uint> counter = counterHandle;
    if (dispatchThreadID.x == 0u) { counter[0] = 0u; }
    // The standalone Stream pass owns the full shared visibility range.
    if (dispatchThreadID.x < bindings.instanceVisibilityResetCount) {
        RWStructuredBuffer<uint> visibility =
            resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(bindings.instanceVisibilityBuffer));
        visibility[dispatchThreadID.x] = 0u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetInstanceCullMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetInstance> instanceHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(push.instanceBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsBuffer = bindingsHandle;
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings bindings = bindingsBuffer[0];
    uint rasterPhase = min(push.traversalPhase, 1u);
    uint candidateIndex = dispatchThreadID.x;
    if (candidateIndex >= bindings.candidateInstanceCount) {
        return;
    }
    StructuredBuffer<uint> candidateInstanceIds =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(bindings.candidateInstanceIdsBuffer));
    // CPU candidates exclude scene-disabled instances without renumbering owners.
    uint instanceIndex = candidateInstanceIds[candidateIndex];

    RWStructuredBuffer<uint> visibilityHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(bindings.instanceVisibilityBuffer));
    RWStructuredBuffer<uint> visibility = visibilityHandle;
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    uint gpuSceneInstanceIndex = instance.gpuSceneInstanceIndex;
    if (gpuSceneInstanceIndex == kInvalidClusterIndex) {
        return;
    }
    if (rasterPhase != 0u && visibility[gpuSceneInstanceIndex] != 2u) {
        return;
    }

    float3 center = instance.boundsCenterRadius.xyz;
    float radius = max(instance.boundsCenterRadius.w, 0.000001) + bindings.displacementBound;
    GPUDrivenStreamAssetCameraData currentCamera = currentStreamCamera(params);
    if ((bindings.cullingFlags & (1u << 0u)) != 0u &&
        !streamSphereInsideFrustum(center, radius, currentCamera)) {
        visibility[gpuSceneInstanceIndex] = 0u;
        return;
    }

    bool occluded = false;
    if ((bindings.cullingFlags & (1u << 1u)) != 0u &&
        (rasterPhase != 0u || bindings.hzbValid != 0u)) {
        uint currentHzb = params.frameIndex & 1u;
        uint hzbIndex = rasterPhase == 0u ? (currentHzb ^ 1u) : currentHzb;
        GPUDrivenStreamAssetCameraData hzbCamera = rasterPhase == 0u
            ? previousStreamCamera(params)
            : currentCamera;
        occluded = streamSphereOccluded(
            center,
            radius,
            hzbCamera,
            bindings,
            push,
            hzbIndex);
    }
    uint visibleState = rasterPhase == 0u ? 1u : 3u;
    visibility[gpuSceneInstanceIndex] = occluded
        ? (rasterPhase == 0u ? 2u : 0u)
        : visibleState;
    if (!occluded) {
        RWStructuredBuffer<uint> visibleIdsHandle =
            resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(bindings.visibleInstanceIdsBuffer));
        RWStructuredBuffer<uint> counterHandle =
            resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(bindings.visibleInstanceCounterBuffer));
        RWStructuredBuffer<uint> visibleIds = visibleIdsHandle;
        RWStructuredBuffer<uint> counter = counterHandle;
        uint outputIndex = 0u;
        InterlockedAdd(counter[0], 1u, outputIndex);
        if (outputIndex < params.sceneInstanceCount) {
            visibleIds[outputIndex] = gpuSceneInstanceIndex;
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gpuDrivenStreamAssetHzbMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsBuffer = bindingsHandle;
    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings bindings = bindingsBuffer[0];
    uint mipLevel = push.activeBuildPhase;
    if (mipLevel >= bindings.hzbMipCount) {
        return;
    }
    uint mipWidth = streamMipDimension(bindings.width, mipLevel);
    uint mipHeight = streamMipDimension(bindings.height, mipLevel);
    if (dispatchThreadID.x >= mipWidth || dispatchThreadID.y >= mipHeight) {
        return;
    }
    uint currentHzb = params.frameIndex & 1u;
    uint hzbHandleIndex = currentHzb == 0u ? bindings.hzbBuffer0 : bindings.hzbBuffer1;
    RWStructuredBuffer<float> hzbHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<float>>(hzbHandleIndex));
    RWStructuredBuffer<float> hzb = hzbHandle;
    uint destinationIndex = streamHzbMipOffset(bindings, mipLevel) +
        dispatchThreadID.y * mipWidth + dispatchThreadID.x;
    if (mipLevel == 0u) {
        Texture2D<float> depthHandle =
            resolveUniform(ResourceHandle<Texture2D<float>>(bindings.depthImage));
        Texture2D<float> depth = depthHandle;
        hzb[destinationIndex] = depth.Load(int3(dispatchThreadID.xy, 0));
        return;
    }
    uint sourceMip = mipLevel - 1u;
    uint sourceWidth = streamMipDimension(bindings.width, sourceMip);
    uint sourceHeight = streamMipDimension(bindings.height, sourceMip);
    uint sourceOffset = streamHzbMipOffset(bindings, sourceMip);
    uint2 sourceBase = dispatchThreadID.xy * 2u;
    uint2 limit = uint2(sourceWidth - 1u, sourceHeight - 1u);
    uint2 source0 = min(sourceBase, limit);
    uint2 source1 = min(sourceBase + uint2(1u, 0u), limit);
    uint2 source2 = min(sourceBase + uint2(0u, 1u), limit);
    uint2 source3 = min(sourceBase + uint2(1u, 1u), limit);
    float depth0 = hzb[sourceOffset + source0.y * sourceWidth + source0.x];
    float depth1 = hzb[sourceOffset + source1.y * sourceWidth + source1.x];
    float depth2 = hzb[sourceOffset + source2.y * sourceWidth + source2.x];
    float depth3 = hzb[sourceOffset + source3.y * sourceWidth + source3.x];
    hzb[destinationIndex] = min(min(depth0, depth1), min(depth2, depth3));
}

float3 hashColor(uint seed)
{
    float3 value = frac(sin(float3(seed, seed * 17u + 13u, seed * 31u + 7u) * 0.017453292) * 43758.5453);
    return lerp(float3(0.16, 0.39, 0.70), float3(0.92, 0.78, 0.26), value);
}

