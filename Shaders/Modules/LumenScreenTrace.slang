module LumenScreenTrace;

import ShaderCore;
import ScenePathMath;

namespace Metallic.Lumen
{

public struct LumenScreenTraceView
{
    public ViewConstants cameras;
    public uint4 state; // History valid, closest HZB mip-zero width and height, mip count.
};

public struct LumenScreenTraceInputs
{
    public StructuredBuffer<float> closestHZB;
    public Texture2D<float> previousDepth;
    public Texture2D<float4> previousColor;
    public StructuredBuffer<LumenScreenTraceView> view;
};

public struct ScreenTraceProjection
{
    float3 eye, right, up, forward;
    float2 halfExtent, extent, jitter;
    float nearPlane, farPlane;
    bool orthographic;
};

public ScreenTraceProjection makeScreenTraceProjection(ViewCameraConstants camera, float2 jitter)
{
    ScreenTraceProjection result;
    result.eye = camera.eye.xyz;
    result.forward = ScenePath.cameraForwardFrom(camera.eye, camera.center);
    result.right = ScenePath.cameraRightFrom(result.forward, camera.upProjection.xyz);
    result.up = ScenePath.cameraUpFrom(result.forward, result.right);
    result.orthographic = camera.upProjection.w > 0.5f;
    float halfHeight = result.orthographic ? camera.clipOrtho.z * 0.5f : tan(camera.viewport.w * 0.5f);
    result.halfExtent = float2(camera.viewport.x * halfHeight, halfHeight);
    result.extent = camera.viewport.yz;
    result.jitter = jitter;
    result.nearPlane = camera.clipOrtho.x;
    result.farPlane = camera.clipOrtho.y;
    return result;
}

public float4 screenTraceWorldToClip(ScreenTraceProjection view, float3 worldPosition)
{
    float3 eyeRelativePosition = worldPosition - view.eye;
    float viewZ = dot(eyeRelativePosition, view.forward);
    float clipW = view.orthographic ? 1.0f : viewZ;
    float clipZ = view.orthographic ? (view.farPlane - viewZ) / (view.farPlane - view.nearPlane)
        : (view.nearPlane * view.farPlane - view.nearPlane * viewZ) / (view.farPlane - view.nearPlane);
    float2 clipXY = float2(dot(eyeRelativePosition, view.right), dot(eyeRelativePosition, view.up)) / view.halfExtent;
    clipXY -= (view.jitter / view.extent * float2(2.0f, -2.0f)) * clipW;
    return float4(clipXY, clipZ, clipW);
}

public float screenTraceConvertFromDeviceZ(ScreenTraceProjection view, float deviceZ)
{
    return view.orthographic ? lerp(view.farPlane, view.nearPlane, deviceZ)
        : view.nearPlane * view.farPlane / (view.nearPlane + deviceZ * (view.farPlane - view.nearPlane));
}

public float3 screenTraceScreenToWorld(ScreenTraceProjection view, float2 screen, float sceneDepth)
{
    float2 unjitteredScreen = screen + view.jitter / view.extent * float2(2.0f, -2.0f);
    float2 offset = unjitteredScreen * view.halfExtent * (view.orthographic ? 1.0f : sceneDepth);
    return view.eye + view.forward * sceneDepth + view.right * offset.x + view.up * offset.y;
}

public float2 screenTraceScreenUV(float2 screenPosition)
{
    return screenPosition * float2(0.5f, -0.5f) + 0.5f;
}

public float2 screenTraceScreenPosition(float2 screenUV)
{
    return (screenUV - 0.5f) * float2(2.0f, -2.0f);
}

uint2 pointClampedPixel(float2 uv, uint2 extent)
{
    return min(uint2(saturate(uv) * float2(extent)), extent - 1u);
}

// UE CalcSceneDepth and GetWorldPositionFromScreenUV. The depth texel is
// point sampled, while the position is reconstructed at the same fractional UV.
public float3 lumenScenePositionAtUV(ViewCameraConstants camera, float2 jitter, uint2 extent,
    Texture2D<float> depth, float2 screenUV)
{
    ScreenTraceProjection view = makeScreenTraceProjection(camera, jitter);
    float deviceZ = depth.Load(int3(pointClampedPixel(screenUV, extent), 0));
    float sceneDepth = screenTraceConvertFromDeviceZ(view, deviceZ);
    return screenTraceScreenToWorld(view, screenTraceScreenPosition(screenUV), sceneDepth);
}

// UE 5.7.4 HZB.usf RoundUpF16, with the closest pyramid stored in a packed buffer.
public float roundUpScreenTraceDepth(float deviceZ)
{
    return f16tof32(f32tof16(deviceZ) + 1u);
}

// UE EncodeProbeRayDistance / DecodeProbeRayDistance, with flags stored separately.
public float decodeLumenTraceContinuationDistance(float hitDistanceCm)
{
    const uint encodedDistance = asuint(max(hitDistanceCm, 0.0f)) & 0x7ffffffcu;
    return asfloat(encodedDistance & 0x7ffffffcu) * 0.01f;
}

float sampleClosestHZB(StructuredBuffer<float> hzb, uint2 fullExtent, uint2 baseExtent,
    uint mipCount, float2 uv, float mipLevel)
{
    uint level = min(uint(mipLevel), mipCount - 1u);
    uint address = fullExtent.x * fullExtent.y;
    for (uint mip = 0u; mip < level; ++mip) {
        uint2 mipExtent = max(baseExtent >> mip, 1u);
        address += mipExtent.x * mipExtent.y;
    }
    uint2 extent = max(baseExtent >> level, 1u);
    uint2 pixel = pointClampedPixel(uv, extent);
    return roundUpScreenTraceDepth(hzb[address + pixel.y * extent.x + pixel.x]);
}

float sampleScreenTraceFullDepth(StructuredBuffer<float> hzb, uint2 extent, float2 uv)
{
    uint2 pixel = pointClampedPixel(uv, extent);
    return hzb[pixel.y * extent.x + pixel.x];
}

// UE 5.7.4 Common.ush LineBoxIntersect, retaining the far intersection used by TraceHZB.
float screenTraceLineBoxExit(float3 rayOrigin, float3 rayEnd, float3 boxMax)
{
    float3 invRayDir = 1.0f / (rayEnd - rayOrigin);
    float3 firstPlaneIntersections = -rayOrigin * invRayDir;
    float3 secondPlaneIntersections = (boxMax - rayOrigin) * invRayDir;
    float3 furthestPlaneIntersections = max(firstPlaneIntersections, secondPlaneIntersections);
    return saturate(min(furthestPlaneIntersections.x, min(furthestPlaneIntersections.y, furthestPlaneIntersections.z)));
}

// UE 5.7.4 HZBTracing.ush TraceHZB, default hierarchical full-resolution opaque path.
bool traceScreenHZB(StructuredBuffer<float> hzb, ScreenTraceProjection view, uint2 hzbExtent,
    uint mipCount, float3 rayOrigin, float3 rayDirection, float maxTraceDistance,
    float relativeDepthThickness, out float3 hitUVz, out bool hasEndpoint)
{
    hitUVz = 0.0f;
    hasEndpoint = false;
    float4 rayStartClip = screenTraceWorldToClip(view, rayOrigin);
    if (any(abs(rayStartClip.xy) > rayStartClip.w) || rayStartClip.w <= 0.0f) {
        return false;
    }
    hasEndpoint = true;

    float2 hzbBaseTexelSize = 1.0f / float2(hzbExtent);
    float2 hzbUvFactor = view.extent / (2.0f * float2(hzbExtent));
    float2 hzbUvInvFactor = 1.0f / hzbUvFactor;
    float3 rayStartScreenPosition = rayStartClip.xyz / max(rayStartClip.w, 0.01f);
    float3 rayStartScreenUV = float3(screenTraceScreenUV(rayStartScreenPosition.xy) * hzbUvFactor, rayStartScreenPosition.z);

    float viewRayDirectionZ = dot(rayDirection, view.forward);
    float sceneDepth = dot(rayOrigin - view.eye, view.forward);
    float rayEndWorldDistance = viewRayDirectionZ < 0.0f
        ? min(-0.99f * sceneDepth / viewRayDirectionZ, maxTraceDistance) : maxTraceDistance;
    float3 rayWorldEnd = rayOrigin + rayDirection * rayEndWorldDistance;
    float4 rayEndClip = screenTraceWorldToClip(view, rayWorldEnd);
    float3 rayEndScreenPosition = rayEndClip.xyz / rayEndClip.w;
    float3 rayEndScreenUV = float3(screenTraceScreenUV(rayEndScreenPosition.xy) * hzbUvFactor, rayEndScreenPosition.z);
    float screenEdgeIntersection = screenTraceLineBoxExit(rayStartScreenUV, rayEndScreenUV, float3(hzbUvFactor, 1.0f));
    rayEndScreenUV = rayStartScreenUV + (rayEndScreenUV - rayStartScreenUV) * screenEdgeIntersection;

    const float baseMipLevel = -1.0f;
    const float maxIterations = 50.0f;
    float mipLevel = baseMipLevel;
    float3 rayDirectionScreenUV = rayEndScreenUV - rayStartScreenUV;
    float currentIntersectionTime = 0.0f;
    float lastAboveSurfaceTime = 0.0f;
    float2 floorOffset = select(rayDirectionScreenUV.xy < 0.0f, 0.0f, 1.0f);
    float3 rayScreenUV = rayStartScreenUV;

    // Step out of the starting tile before testing depth.
    {
        float2 currentMipTexelSize = exp2(mipLevel) * hzbBaseTexelSize;
        float2 currentMipResolution = 1.0f / currentMipTexelSize;
        float2 uvOffset = 0.005f * currentMipTexelSize;
        uvOffset = select(rayDirectionScreenUV.xy < 0.0f, -uvOffset, uvOffset);
        float2 xyPlane = floor(rayScreenUV.xy * currentMipResolution) + floorOffset;
        xyPlane = xyPlane * currentMipTexelSize + uvOffset;
        float2 planeIntersectionTimes = (xyPlane - rayStartScreenUV.xy) / rayDirectionScreenUV.xy;
        currentIntersectionTime = min(planeIntersectionTimes.x, planeIntersectionTimes.y);
        rayScreenUV = rayStartScreenUV + currentIntersectionTime * rayDirectionScreenUV;
    }
    if (currentIntersectionTime >= 1.0f) {
        rayScreenUV = rayStartScreenUV;
    }

    float numIterations = 0.0f;
    while (mipLevel >= baseMipLevel && numIterations < maxIterations && currentIntersectionTime < 1.0f) {
        float2 currentMipTexelSize = exp2(mipLevel) * hzbBaseTexelSize;
        float2 currentMipResolution = 1.0f / currentMipTexelSize;
        float2 uvOffset = 0.005f * currentMipTexelSize;
        uvOffset = select(rayDirectionScreenUV.xy < 0.0f, -uvOffset, uvOffset);
        float2 xyPlane = floor(rayScreenUV.xy * currentMipResolution) + floorOffset;
        xyPlane = xyPlane * currentMipTexelSize + uvOffset;

        float tileZ = mipLevel < 0.0f
            ? sampleScreenTraceFullDepth(hzb, uint2(view.extent), rayScreenUV.xy * hzbUvInvFactor)
            : sampleClosestHZB(hzb, uint2(view.extent), hzbExtent, mipCount, rayScreenUV.xy, mipLevel);
        float3 boundaryPlanes = float3(xyPlane, tileZ);
        float3 planeIntersectionTimes = (boundaryPlanes - rayStartScreenUV) / rayDirectionScreenUV;
        planeIntersectionTimes.z = rayDirectionScreenUV.z < 0.0f ? planeIntersectionTimes.z : 1.0f;
        float intersectionTime = min(planeIntersectionTimes.x, planeIntersectionTimes.y);
        bool aboveSurface = rayScreenUV.z > tileZ;
        bool skippedTile = aboveSurface;
        intersectionTime = min(intersectionTime, planeIntersectionTimes.z);
        skippedTile &= intersectionTime != planeIntersectionTimes.z;

        if (skippedTile) {
            lastAboveSurfaceTime = intersectionTime;
        }
        currentIntersectionTime = aboveSurface ? intersectionTime : currentIntersectionTime;
        rayScreenUV = rayStartScreenUV + min(currentIntersectionTime, 1.0f) * rayDirectionScreenUV;
        mipLevel += skippedTile ? 1.0f : -1.0f;
        ++numIterations;
    }

    bool hit = false;
    if (mipLevel < baseMipLevel && currentIntersectionTime < 1.0f) {
        float2 sampleUV = rayScreenUV.xy * hzbUvInvFactor;
        float tileZ = sampleScreenTraceFullDepth(hzb, uint2(view.extent), sampleUV);
        float hitSceneDepth = screenTraceConvertFromDeviceZ(view, tileZ);
        float raySceneDepth = screenTraceConvertFromDeviceZ(view, rayScreenUV.z);
        hit = tileZ > 0.0f && (raySceneDepth - hitSceneDepth) < relativeDepthThickness * max(hitSceneDepth, 0.0000001f);
        if (!hit) {
            rayScreenUV = rayStartScreenUV + lastAboveSurfaceTime * rayDirectionScreenUV;
        }
    }
    hitUVz = float3(rayScreenUV.xy * hzbUvInvFactor, rayScreenUV.z);
    return hit;
}

// UE 5.7.4 SSRTRayCast.ush and RandomInterleavedGradientNoise.ush.
float screenTraceHitVignette(float2 screenPosition)
{
    float2 vignette = saturate(abs(screenPosition) * 5.0f - 4.0f);
    return saturate(1.0f - dot(vignette, vignette));
}

public float screenTraceNoise(float2 uv, float frameId)
{
    uv += frameId * (float2(47.0f, 17.0f) * 0.695f);
    const float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
    return frac(magic.z * frac(dot(uv, magic.xy)));
}

// UE 5.7.4 LumenScreenProbeTracing.usf, camera reprojection for static opaque surfaces.
public bool traceLumenScreenRay(LumenScreenTraceInputs inputs, uint2 receiverPixel, float3 receiverPosition,
    float3 receiverNormal, float3 rayDirection, float traceDistance, float maxTraceDistance, float relativeDepthThickness,
    uint2 noisePixel,
    out float3 radiance, out float distance, out float continuationDistance)
{
    radiance = 0.0f;
    distance = 0.0f;
    continuationDistance = 0.0f;
    LumenScreenTraceView traceView = inputs.view[0];
    if (traceView.state.x == 0u) {
        return false;
    }

    ScreenTraceProjection current = makeScreenTraceProjection(traceView.cameras.current, traceView.cameras.jitter.xy);
    float receiverDepth = dot(receiverPosition - current.eye, current.forward);
    float2 invViewSize = 1.0f / current.extent;
    float2 screenUV = (float2(receiverPixel) + 0.5f) * invViewSize;
    float2 cornerScreenPosition = screenTraceScreenPosition(screenUV + 0.5f * invViewSize);
    float3 cornerWorldPosition = screenTraceScreenToWorld(current, cornerScreenPosition, receiverDepth);
    float normalBias = abs(dot(cornerWorldPosition - receiverPosition, receiverNormal)) * 2.0f;
    float3 rayOrigin = receiverPosition + normalBias * receiverNormal;
    float depthThicknessScale = 1.0f;
    if (current.orthographic) {
        int exponent = int(-clamp(log10(abs(current.farPlane - current.nearPlane) * 100.0f), 0.0f, 9.0f));
        depthThicknessScale = pow(10.0f, float(exponent));
    }

    float3 hitUVz;
    bool hasEndpoint;
    bool hit = traceScreenHZB(inputs.closestHZB, current, traceView.state.yz, traceView.state.w, rayOrigin,
        rayDirection, traceDistance, relativeDepthThickness * depthThicknessScale, hitUVz, hasEndpoint);
    if (!hasEndpoint) {
        return false;
    }

    float2 hitScreenPosition = screenTraceScreenPosition(hitUVz.xy);
    float hitSceneDepth = screenTraceConvertFromDeviceZ(current, hitUVz.z);
    float3 hitWorldPosition = screenTraceScreenToWorld(current, hitScreenPosition, hitSceneDepth);
    // UE 5.7.4 ComputeRayHitSqrDistance and Encode/DecodeProbeRayDistance.
    // Pack in centimeters, before history rejection. The WRC limit uses the unencoded distance.
    float3 receiverToHitCm = (receiverPosition - hitWorldPosition) * 100.0f;
    float hitDistanceOffset = hit ? 0.0f : 2.0f;
    float hitDistanceCm = min(sqrt(dot(receiverToHitCm, receiverToHitCm)) + hitDistanceOffset,
        maxTraceDistance * 100.0f);
    distance = hitDistanceCm * 0.01f;
    continuationDistance = decodeLumenTraceContinuationDistance(hitDistanceCm);
    if (!hit) {
        return false;
    }
    // ClipToPrevClip excludes both cameras' jitter in the UE history lookup.
    current.jitter = 0.0f;
    float3 reprojectWorldPosition = screenTraceScreenToWorld(current, hitScreenPosition, hitSceneDepth);
    ScreenTraceProjection previous = makeScreenTraceProjection(traceView.cameras.previous, 0.0f);
    float4 historyClip = screenTraceWorldToClip(previous, reprojectWorldPosition);
    if (historyClip.w <= 0.0f) {
        return false;
    }
    float3 historyScreenPosition = historyClip.xyz / historyClip.w;
    float vignette = min(screenTraceHitVignette(hitScreenPosition), screenTraceHitVignette(historyScreenPosition.xy));
    float noise = screenTraceNoise(float2(noisePixel) + 0.5f, float(traceView.cameras.frame.x % 8u));
    if (vignette < noise) {
        return false;
    }

    float2 historyUV = screenTraceScreenUV(historyScreenPosition.xy);
    uint2 extent = uint2(previous.extent);
    uint2 depthPixel = pointClampedPixel(historyUV, extent);
    float historyDeviceZ = inputs.previousDepth.Load(int3(depthPixel, 0));
    float historyDepthTolerance = 0.005f * depthThicknessScale * lerp(0.5f, 2.0f, noise);
    if (!(historyDeviceZ > 0.0f && abs(historyDeviceZ - historyScreenPosition.z) < historyDepthTolerance)) {
        return false;
    }

    float2 colorHalfTexel = 0.5f / previous.extent;
    uint2 colorPixel = pointClampedPixel(clamp(historyUV, colorHalfTexel, 1.0f - colorHalfTexel), extent);
    radiance = -min(-inputs.previousColor.Load(int3(colorPixel, 0)).rgb, 0.0f);
    return true;
}

} // namespace Metallic.Lumen
