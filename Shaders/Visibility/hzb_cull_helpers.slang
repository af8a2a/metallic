float sampleHzb(uint level, uint2 coord) {
    switch (level) {
    case 0u: return hzbPyramid[0].Load(int3(coord, 0));
    case 1u: return hzbPyramid[1].Load(int3(coord, 0));
    case 2u: return hzbPyramid[2].Load(int3(coord, 0));
    case 3u: return hzbPyramid[3].Load(int3(coord, 0));
    case 4u: return hzbPyramid[4].Load(int3(coord, 0));
    case 5u: return hzbPyramid[5].Load(int3(coord, 0));
    case 6u: return hzbPyramid[6].Load(int3(coord, 0));
    case 7u: return hzbPyramid[7].Load(int3(coord, 0));
    case 8u: return hzbPyramid[8].Load(int3(coord, 0));
    default: return hzbPyramid[9].Load(int3(coord, 0));
    }
}

bool spherePreviousFrameOccluded(float4x4 prevViewProj,
                                 float4x4 prevView,
                                 float4 prevCameraWorldPos,
                                 float2 prevProjScale,
                                 float2 hzbTextureSize,
                                 uint hzbLevelCount,
                                 float occlusionBoundsScale,
                                 float occlusionDepthBias,
                                 float3 centerWS,
                                 float worldRadius) {
    if (hzbLevelCount == 0u) {
        return false;
    }

    float4 prevCenterClip = mul(prevViewProj, float4(centerWS, 1.0));
    if (prevCenterClip.w <= 1e-5) {
        return false;
    }

    float3 centerVS = mul(prevView, float4(centerWS, 1.0)).xyz;
    float viewDepth = max(abs(centerVS.z), 1e-4);
    float2 ndcRadius = float2(prevProjScale.x * worldRadius / viewDepth,
                              prevProjScale.y * worldRadius / viewDepth) *
                       occlusionBoundsScale;
    float2 centerNdc = prevCenterClip.xy / prevCenterClip.w;

    float2 ndcMin = centerNdc - ndcRadius;
    float2 ndcMax = centerNdc + ndcRadius;
    bool outsidePrevView = ndcMax.x < -1.0 || ndcMin.x > 1.0 ||
                           ndcMax.y < -1.0 || ndcMin.y > 1.0;
    if (outsidePrevView) {
        return false;
    }

    float2 minUv = float2(centerNdc.x - ndcRadius.x, centerNdc.y + ndcRadius.y) *
                       float2(0.5, -0.5) +
                   float2(0.5, 0.5);
    float2 maxUv = float2(centerNdc.x + ndcRadius.x, centerNdc.y - ndcRadius.y) *
                       float2(0.5, -0.5) +
                   float2(0.5, 0.5);
    minUv = clamp(minUv, float2(0.0, 0.0), float2(1.0, 1.0));
    maxUv = clamp(maxUv, float2(0.0, 0.0), float2(1.0, 1.0));
    float2 uvMin = min(minUv, maxUv);
    float2 uvMax = max(minUv, maxUv);

    float2 rectSizePixels = max((uvMax - uvMin) * hzbTextureSize, float2(1.0, 1.0));
    float maxExtent = max(rectSizePixels.x, rectSizePixels.y);
    uint level = 0u;
    if (maxExtent > 1.0) {
        level = min(hzbLevelCount - 1u, (uint)ceil(log2(maxExtent)));
    }

    uint2 mipSize = uint2((uint)hzbTextureSize.x, (uint)hzbTextureSize.y);
    for (uint mip = 0u; mip < level; ++mip) {
        mipSize = uint2(max((mipSize.x + 1u) / 2u, 1u),
                        max((mipSize.y + 1u) / 2u, 1u));
    }

    float2 mipScale = float2((float)mipSize.x, (float)mipSize.y);
    uint2 minTexel = uint2(floor(uvMin * mipScale));
    uint2 maxTexel = uint2(floor(uvMax * mipScale));
    uint2 mipMaxCoord = uint2(max(mipSize.x, 1u) - 1u,
                              max(mipSize.y, 1u) - 1u);
    minTexel = min(minTexel, mipMaxCoord);
    maxTexel = min(maxTexel, mipMaxCoord);

    float sampledDepth = sampleHzb(level, minTexel);
    sampledDepth = min(sampledDepth, sampleHzb(level, uint2(maxTexel.x, minTexel.y)));
    sampledDepth = min(sampledDepth, sampleHzb(level, uint2(minTexel.x, maxTexel.y)));
    sampledDepth = min(sampledDepth, sampleHzb(level, maxTexel));

    float3 prevCameraToCenter = centerWS - prevCameraWorldPos.xyz;
    float prevCameraDistanceSq = dot(prevCameraToCenter, prevCameraToCenter);
    if (prevCameraDistanceSq <= 1e-10) {
        return false;
    }

    float3 nearestPointWS = centerWS - normalize(prevCameraToCenter) * worldRadius;
    float4 nearestClip = mul(prevViewProj, float4(nearestPointWS, 1.0));
    if (nearestClip.w <= 1e-5) {
        return false;
    }

    float nearestDepth = saturate(nearestClip.z / nearestClip.w);
    return nearestDepth < sampledDepth - occlusionDepthBias;
}

#ifdef HZB_CULL_ENABLE_CURRENT_PYRAMID
float sampleCurrentHzb(uint level, uint2 coord) {
    switch (level) {
    case 0u: return currentHzbPyramid[0].Load(int3(coord, 0));
    case 1u: return currentHzbPyramid[1].Load(int3(coord, 0));
    case 2u: return currentHzbPyramid[2].Load(int3(coord, 0));
    case 3u: return currentHzbPyramid[3].Load(int3(coord, 0));
    case 4u: return currentHzbPyramid[4].Load(int3(coord, 0));
    case 5u: return currentHzbPyramid[5].Load(int3(coord, 0));
    case 6u: return currentHzbPyramid[6].Load(int3(coord, 0));
    case 7u: return currentHzbPyramid[7].Load(int3(coord, 0));
    case 8u: return currentHzbPyramid[8].Load(int3(coord, 0));
    default: return currentHzbPyramid[9].Load(int3(coord, 0));
    }
}

bool sphereCurrentFrameOccluded(float4x4 viewProj,
                                float4x4 view,
                                float4 cameraWorldPos,
                                float2 projScale,
                                float2 hzbTextureSize,
                                uint hzbLevelCount,
                                float occlusionBoundsScale,
                                float occlusionDepthBias,
                                float3 centerWS,
                                float worldRadius) {
    if (hzbLevelCount == 0u) {
        return false;
    }

    float4 centerClip = mul(viewProj, float4(centerWS, 1.0));
    if (centerClip.w <= 1e-5) {
        return false;
    }

    float3 centerVS = mul(view, float4(centerWS, 1.0)).xyz;
    float viewDepth = max(abs(centerVS.z), 1e-4);
    float2 ndcRadius = float2(projScale.x * worldRadius / viewDepth,
                              projScale.y * worldRadius / viewDepth) *
                       occlusionBoundsScale;
    float2 centerNdc = centerClip.xy / centerClip.w;

    float2 ndcMin = centerNdc - ndcRadius;
    float2 ndcMax = centerNdc + ndcRadius;
    bool outsideView = ndcMax.x < -1.0 || ndcMin.x > 1.0 ||
                       ndcMax.y < -1.0 || ndcMin.y > 1.0;
    if (outsideView) {
        return false;
    }

    float2 minUv = float2(centerNdc.x - ndcRadius.x, centerNdc.y + ndcRadius.y) *
                       float2(0.5, -0.5) +
                   float2(0.5, 0.5);
    float2 maxUv = float2(centerNdc.x + ndcRadius.x, centerNdc.y - ndcRadius.y) *
                       float2(0.5, -0.5) +
                   float2(0.5, 0.5);
    minUv = clamp(minUv, float2(0.0, 0.0), float2(1.0, 1.0));
    maxUv = clamp(maxUv, float2(0.0, 0.0), float2(1.0, 1.0));
    float2 uvMin = min(minUv, maxUv);
    float2 uvMax = max(minUv, maxUv);

    float2 rectSizePixels = max((uvMax - uvMin) * hzbTextureSize, float2(1.0, 1.0));
    float maxExtent = max(rectSizePixels.x, rectSizePixels.y);
    uint level = 0u;
    if (maxExtent > 1.0) {
        level = min(hzbLevelCount - 1u, (uint)ceil(log2(maxExtent)));
    }

    uint2 mipSize = uint2((uint)hzbTextureSize.x, (uint)hzbTextureSize.y);
    for (uint mip = 0u; mip < level; ++mip) {
        mipSize = uint2(max((mipSize.x + 1u) / 2u, 1u),
                        max((mipSize.y + 1u) / 2u, 1u));
    }

    float2 mipScale = float2((float)mipSize.x, (float)mipSize.y);
    uint2 minTexel = uint2(floor(uvMin * mipScale));
    uint2 maxTexel = uint2(floor(uvMax * mipScale));
    uint2 mipMaxCoord = uint2(max(mipSize.x, 1u) - 1u,
                              max(mipSize.y, 1u) - 1u);
    minTexel = min(minTexel, mipMaxCoord);
    maxTexel = min(maxTexel, mipMaxCoord);

    float sampledDepth = sampleCurrentHzb(level, minTexel);
    sampledDepth = min(sampledDepth, sampleCurrentHzb(level, uint2(maxTexel.x, minTexel.y)));
    sampledDepth = min(sampledDepth, sampleCurrentHzb(level, uint2(minTexel.x, maxTexel.y)));
    sampledDepth = min(sampledDepth, sampleCurrentHzb(level, maxTexel));

    float3 cameraToCenter = centerWS - cameraWorldPos.xyz;
    float cameraDistanceSq = dot(cameraToCenter, cameraToCenter);
    if (cameraDistanceSq <= 1e-10) {
        return false;
    }

    float3 nearestPointWS = centerWS - normalize(cameraToCenter) * worldRadius;
    float4 nearestClip = mul(viewProj, float4(nearestPointWS, 1.0));
    if (nearestClip.w <= 1e-5) {
        return false;
    }

    float nearestDepth = saturate(nearestClip.z / nearestClip.w);
    return nearestDepth < sampledDepth - occlusionDepthBias;
}
#endif
