implementing VolumetricCloudView;

// UE 5.7.4 VolumetricRenderTarget.usf: mode 0 with min/max tracing,
// checkerboard reconstruction and bilateral upsampling (mode 4).
static const float kCloudMaxDepthKm = 65504.0f;

struct CloudTargetSample
{
    float4 farRGBT, nearRGBT;
    float4 depthKm; // Reprojection front, traced scene depth, near and far scene depth.
    float2 weightedDepthMeters; // Metallic graph output, far and near, weighted by opacity.
};

float cloudSceneDepthKm(ScreenTraceProjection camera, float deviceDepth, float cosView)
{
    if (deviceDepth == 0.0f) { return kCloudMaxDepthKm; }
    return min(kCloudMaxDepthKm, screenTraceConvertFromDeviceZ(camera, deviceDepth) * 0.001f / cosView);
}

float4 cloudViewRGBT(VolumetricCloudViewParams p, GPUAtmosphereParameters a, CloudViewIntegral cloud,
    float3 originMeters, float3 originKm, float3 direction, DirectionalAtmosphereShadow shadow)
{
    if (cloud.depthWeightsSum == 0.0f) { return float4(0, 0, 0, 1); }
    AtmosphereSegment aerial = cloudAerialPerspective(p, a, originMeters, originKm, direction,
        cloud.representativeDepthMeters(), shadow);
    return float4(aerial.radiance * (1.0f - cloud.transmittance) +
        aerial.transmittance * spectralToWorking(cloud.luminance),
        cloud.transmittance < kStopTracingTransmittance ? 0.0f : cloud.transmittance);
}

void writeCloudTargets(VolumetricCloudViewParams p, uint2 coord, CloudTargetSample s)
{
    resolveUniform(p.writeFar)[coord] = s.farRGBT;
    resolveUniform(p.writeNear)[coord] = s.nearRGBT;
    resolveUniform(p.writeDepth)[coord] = s.depthKm;
    resolveUniform(p.writeRepresentativeDepth)[coord] = s.weightedDepthMeters;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cloudTraceTemporalMain(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricCloudViewParams>();
    uint2 fullSize = uint2(p.camera.viewport.yz);
    uint2 halfSize = (fullSize + 1u) / 2u;
    uint2 traceSize = (halfSize + 1u) / 2u;
    if (any(id.xy >= traceSize)) { return; }
    // UE removes the scene TAA jitter from the tracing camera, then jitters it
    // to the scheduled half-resolution pixel. Use the actual rounded extents.
    float2 uv = (float2(id.xy) + 0.5f) / float2(traceSize) +
        (float2(p.tracingPixelOffset) - 0.5f) / float2(halfSize);
    ScreenTraceProjection camera = makeScreenTraceProjection(p.camera, 0.0f);
    float2 offset = screenTraceScreenPosition(uv) * camera.halfExtent;
    float3 rayOffset = camera.right * offset.x + camera.up * offset.y;
    float3 direction = camera.orthographic ? camera.forward : normalize(camera.forward + rayOffset);
    float3 originMeters = camera.orthographic ? camera.eye + rayOffset : camera.eye;
    float cosView = dot(direction, camera.forward);

    // CreateQuarterResolutionDepthMinAndMaxFromDepthTexture, fused here so the
    // graph needs no additional half- and quarter-resolution depth surfaces.
    float nearZ = 0.0f, farZ = 1.0f;
    [unroll] for (uint y = 0; y < 4; ++y) {
        [unroll] for (uint x = 0; x < 4; ++x) {
            uint2 pixel = min(id.xy * 4u + uint2(x, y), fullSize - 1u);
            float z = resolveUniform(p.depth).Load(int3(pixel, 0));
            nearZ = max(nearZ, z);
            farZ = min(farZ, z);
        }
    }
    float nearKm = cloudSceneDepthKm(camera, nearZ, cosView);
    float farKm = cloudSceneDepthKm(camera, farZ, cosView);
    GPUAtmosphereParameters a = resolveUniform(p.atmosphereParameters)[0];
    float3 originKm = worldToAtmosphereKm(a, originMeters);
    let shadow = DirectionalAtmosphereShadow::forRay(p.worldShadowFaces.index, p.shadowMask, originMeters);
    float noise = lumenProbeBlueNoiseScalar(resolveUniform(p.blueNoise), id.xy * 4u + 2u, p.noiseFrameIndex);
    CloudViewIntegral nearCloud;
    float noCloudDepthKm;
    CloudViewIntegral farCloud = traceViewCloud(p, a, originKm, direction, farKm, noise, shadow,
        nearKm * 1000.0f, nearCloud, noCloudDepthKm);
    CloudTargetSample s;
    s.farRGBT = cloudViewRGBT(p, a, farCloud, originMeters, originKm, direction, shadow);
    s.nearRGBT = cloudViewRGBT(p, a, nearCloud, originMeters, originKm, direction, shadow);
    float farCloudDepth = farCloud.representativeDepthMeters();
    s.depthKm = float4(s.farRGBT.a > 0.99f ? noCloudDepthKm : farCloudDepth * 0.001f, farKm, nearKm, farKm);
    s.weightedDepthMeters = float2(farCloudDepth, nearCloud.representativeDepthMeters()) *
        (1.0f - float2(s.farRGBT.a, s.nearRGBT.a));
    writeCloudTargets(p, id.xy, s);
}

struct CloudBilinear
{
    int2 lo, hi;
    float2 f;
};

CloudBilinear cloudBilinear(float2 uv, uint2 size)
{
    float2 pos = clamp(uv * float2(size) - 0.5f, 0.0f, float2(size - 1u));
    CloudBilinear b;
    b.lo = int2(pos);
    b.hi = min(b.lo + 1, int2(size) - 1);
    b.f = frac(pos);
    return b;
}

float4 sampleCloudImage(Texture2D<float4> image, CloudBilinear b)
{
    return lerp(lerp(image.Load(int3(b.lo, 0)), image.Load(int3(b.hi.x, b.lo.y, 0)), b.f.x),
        lerp(image.Load(int3(b.lo.x, b.hi.y, 0)), image.Load(int3(b.hi, 0)), b.f.x), b.f.y);
}

float2 sampleCloudImage(Texture2D<float2> image, CloudBilinear b)
{
    return lerp(lerp(image.Load(int3(b.lo, 0)), image.Load(int3(b.hi.x, b.lo.y, 0)), b.f.x),
        lerp(image.Load(int3(b.lo.x, b.hi.y, 0)), image.Load(int3(b.hi, 0)), b.f.x), b.f.y);
}

CloudTargetSample sampleCloudTargets(VolumetricCloudViewParams p, bool previous, CloudBilinear b)
{
    CloudTargetSample s;
    s.farRGBT = sampleCloudImage(resolveUniform(previous ? p.previousFar : p.currentFar), b);
    s.nearRGBT = sampleCloudImage(resolveUniform(previous ? p.previousNear : p.currentNear), b);
    s.depthKm = sampleCloudImage(resolveUniform(previous ? p.previousDepth : p.currentDepth), b);
    s.weightedDepthMeters = sampleCloudImage(resolveUniform(previous ? p.previousRepresentativeDepth : p.currentRepresentativeDepth), b);
    return s;
}

CloudTargetSample loadCloudTargets(VolumetricCloudViewParams p, uint2 coord)
{
    CloudTargetSample s;
    s.farRGBT = resolveUniform(p.currentFar).Load(int3(coord, 0));
    s.nearRGBT = resolveUniform(p.currentNear).Load(int3(coord, 0));
    s.depthKm = resolveUniform(p.currentDepth).Load(int3(coord, 0));
    s.weightedDepthMeters = resolveUniform(p.currentRepresentativeDepth).Load(int3(coord, 0));
    return s;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cloudReconstructMain(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricCloudViewParams>();
    uint2 halfSize = (uint2(p.camera.viewport.yz) + 1u) / 2u;
    if (any(id.xy >= halfSize)) { return; }
    uint2 traceSize = (halfSize + 1u) / 2u;
    float2 uv = (float2(id.xy) + 0.5f) / float2(halfSize);
    // All previous textures are bound together when the history is valid.
    if (p.previousFar.index == 0xffffffffu) {
        writeCloudTargets(p, id.xy, sampleCloudTargets(p, false, cloudBilinear(uv, traceSize)));
        return;
    }

    CloudTargetSample fresh = loadCloudTargets(p, id.xy / 2u);
    // UE ConvertToDeviceZ(front depth) treats ray distance as view-axis depth.
    // Current and previous unjittered projections implement ClipToPrevClip.
    float3 world = screenTraceScreenToWorld(makeScreenTraceProjection(p.camera, 0.0f),
        screenTraceScreenPosition(uv), fresh.depthKm.x * 1000.0f);
    float4 clip = screenTraceWorldToClip(makeScreenTraceProjection(p.previousCamera, 0.0f), world);
    float2 previousUV = screenTraceScreenUV(clip.xy / clip.w);
    bool validUV = all(previousUV > 0.0f) && all(previousUV < 1.0f) && fresh.depthKm.x >= 0.0f;
    fresh.depthKm = select(fresh.depthKm > 60000.0f, kCloudMaxDepthKm, fresh.depthKm);
    // UE tests depth changes before choosing even the offscreen fallback. Its
    // previous depth sampler clamps outside the viewport.
    CloudBilinear historyFilter = cloudBilinear(clip.w == 0.0f ? uv : previousUV, halfSize);
    float4 historyDepth = sampleCloudImage(resolveUniform(p.previousDepth), historyFilter);
    historyDepth = select(historyDepth > 60000.0f, kCloudMaxDepthKm, historyDepth);
    bool useNew = all((id.xy % 2u) == p.tracingPixelOffset);
    if (any(fresh.depthKm.yzw < 5.0f) || any(historyDepth.yzw < 5.0f)) {
        if ((abs(fresh.depthKm.z - fresh.depthKm.w) > 2.0f &&
             abs(fresh.depthKm.z - fresh.depthKm.y) < abs(fresh.depthKm.w - fresh.depthKm.y)) ||
            historyDepth.y < fresh.depthKm.y - 2.0f || historyDepth.y > fresh.depthKm.y + 2.0f) {
            useNew = true;
        }
    }
    CloudTargetSample s;
    if (useNew) {
        s = fresh;
    } else if (validUV) {
        if (abs(fresh.depthKm.z - historyDepth.z) > 1.0f) {
            s = fresh;
        } else {
            s = sampleCloudTargets(p, true, historyFilter);
            s.depthKm = historyDepth;
        }
        if (!(all(isfinite(s.farRGBT)) && all(isfinite(s.nearRGBT)) && all(isfinite(s.depthKm.xy)))) {
            s.farRGBT = s.nearRGBT = float4(0, 0, 0, 1);
            s.depthKm = 1000.0f;
            s.weightedDepthMeters = 0.0f;
        }
    } else {
        s = sampleCloudTargets(p, false, cloudBilinear(uv, traceSize));
    }
    writeCloudTargets(p, id.xy, s);
}

// Default UE bilateral resolve for opaque geometry and sky. Metallic carries
// one additional interpolant through exactly the same tap choice as the RGBT.
float4 composeCloudSample(VolumetricCloudViewParams p, uint2 pixel, uint2 halfSize,
    float sceneKm, out float depthMeters)
{
    depthMeters = 0.0f;
    int2 center = int2(pixel / 2u);
    int2 offset = int2(pixel % 2u) * 2 - 1;
    int2 coords[4] = {center, center + int2(offset.x, 0), center + offset, center + int2(0, offset.y)};
    float depthDifference = 0.0f;
    bool sampleNear = false;
    int2 bestCoord = center;
    float minDepth = kCloudMaxDepthKm, maxDepth = 0.0f;
    bool allBehind = true, allInFrontOfNear = true, allDistant = true;
    [unroll] for (uint i = 0; i < 4; ++i) {
        int2 coord = clamp(coords[i], int2(0), int2(halfSize) - 1);
        float4 d = resolveUniform(p.currentDepth).Load(int3(coord, 0));
        // The reference visits the far then the near sample; keep the first on ties.
        float diff = abs(sceneKm - d.w);
        if (i == 0u || diff < depthDifference) {
            bestCoord = coord; sampleNear = false; depthDifference = diff;
        }
        diff = abs(sceneKm - d.z);
        if (diff < depthDifference) {
            bestCoord = coord; sampleNear = true; depthDifference = diff;
        }
        minDepth = min(minDepth, min(d.z, d.w));
        maxDepth = max(maxDepth, max(d.z, d.w));
        allBehind = allBehind && d.x > sceneKm + 0.001f;
        allInFrontOfNear = allInFrontOfNear && d.x <= d.z;
        allDistant = allDistant && all(d.yzw >= 5.0f);
    }
    if (allBehind) { return float4(0, 0, 0, 1); }
    float4 rgbt;
    float weightedDepth;
    if ((allInFrontOfNear && maxDepth - minDepth < 1.0f) || (allInFrontOfNear && allDistant)) {
        float2 sampleUV = (float2(center) + 0.25f + (float2(offset) * 0.5f + 0.5f) * 0.5f) / float2(halfSize);
        CloudBilinear b = cloudBilinear(sampleUV, halfSize);
        rgbt = sampleCloudImage(resolveUniform(p.currentFar), b);
        weightedDepth = sampleCloudImage(resolveUniform(p.currentRepresentativeDepth), b).x;
    } else {
        rgbt = resolveUniform(sampleNear ? p.currentNear : p.currentFar).Load(int3(bestCoord, 0));
        float2 weights = resolveUniform(p.currentRepresentativeDepth).Load(int3(bestCoord, 0));
        weightedDepth = sampleNear ? weights.y : weights.x;
    }
    float opacity = 1.0f - rgbt.a;
    if (opacity > 0.0f) { depthMeters = min(weightedDepth / opacity, sceneKm * 1000.0f); }
    return rgbt;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cloudComposeMain(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricCloudViewParams>();
    uint2 fullSize = uint2(p.camera.viewport.yz);
    if (any(id.xy >= fullSize)) { return; }
    uint2 pixel = id.xy;
    float4 rgbt = float4(0, 0, 0, 1);
    float depthMeters = 0.0f;
    float z = resolveUniform(p.depth).Load(int3(pixel, 0));
    if (z > 0.0f || p.skyVisible != 0u) {
        ScreenTraceProjection camera = makeScreenTraceProjection(p.camera, p.jitter);
        float2 screen = screenTraceScreenPosition((float2(pixel) + 0.5f + p.jitter) / float2(fullSize));
        float cosView = camera.orthographic ? 1.0f : rsqrt(1.0f + dot(screen * camera.halfExtent, screen * camera.halfExtent));
        rgbt = composeCloudSample(p, pixel, (fullSize + 1u) / 2u, cloudSceneDepthKm(camera, z, cosView), depthMeters);
    }
    float4 scene = resolveUniform(p.sceneColor).Load(int3(pixel, 0));
    resolveUniform(p.scattering)[pixel] = float4(rgbt.rgb, 1.0f - rgbt.a);
    resolveUniform(p.transmittance)[pixel] = rgbt.a;
    resolveUniform(p.cloudDepth)[pixel] = depthMeters;
    resolveUniform(p.color)[pixel] = float4(rgbt.rgb + rgbt.a * scene.rgb, scene.a);
}
