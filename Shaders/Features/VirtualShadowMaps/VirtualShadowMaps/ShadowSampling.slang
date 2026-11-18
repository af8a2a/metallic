#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_DIRECTIONAL_PROJECTION
#define VSM_DIRECTIONAL_PROJECTION 0
#endif

uint vsmSampleHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float vsmSampleRandom(uint2 pixel, uint dimension, uint phase)
{
    float spatial = float(vsmSampleHash(pixel.x ^ vsmSampleHash(pixel.y + dimension * 0x9e3779b9u)) >> 8) / 16777216.0;
    // Preserve deterministic raw rendering. With history enabled, rotate both
    // disk and texel/step dimensions with a bounded low-discrepancy sequence.
    return phase == 0u ? spatial : frac(spatial + float(phase) * ((dimension & 1u) != 0u ? 0.569840291 : 0.754877666));
}

float2 vsmDiskSample(uint ray, float invRayCount, float2 scramble)
{
    // Scramble the stratified disk per receiver. A common disk point at one ray
    // shifts a hard shadow instead of estimating area-source visibility.
    float2 sample = frac(float2(float(ray) * invRayCount, float(ray) * 0.61803398875) +
        scramble);
    float angle = sample.y * 6.28318530718;
    float sinAngle, cosAngle;
    sincos(angle, sinAngle, cosAngle);
    return sqrt(sample.x) * float2(cosAngle, sinAngle);
}

float2 vsmDiskSample(uint ray, uint rayCount, float2 scramble)
{
    return vsmDiskSample(ray, 1.0 / float(rayCount), scramble);
}

float2 vsmDiskSample(uint ray, float invRayCount, uint2 pixel, uint phase)
{
    return vsmDiskSample(ray, invRayCount,
        float2(vsmSampleRandom(pixel, 0u, phase), vsmSampleRandom(pixel, 1u, phase)));
}

float2 vsmDiskSample(uint ray, uint rayCount, uint2 pixel, uint phase)
{
    return vsmDiskSample(ray, 1.0 / float(rayCount), pixel, phase);
}

float3 vsmRayEndpointDirectional(float3 position, float3 toLight, float3 offset, float tanAlpha, float rayLength)
{
    return position + normalize(toLight + offset * tanAlpha) * rayLength;
}

float3 vsmRayEndpoint(float3 position, ShadowProjection projection, ShadowPush push, float3 toLight, float3 offset)
{
    if (push.lightType == 0u) {
        return position + normalize(toLight + offset * tan(push.angularRadius)) * push.rayLength;
    }
    float distance = length(projection.originNear.xyz - position);
    // Metallic's local source is a planar disk at the light center.
    float3 direction = normalize(toLight * distance + offset * push.sourceRadius);
    float sinTheta = length(cross(direction, toLight));
    float cosTheta = dot(direction, toLight);
    // A single center-light depth layer becomes a poor proxy near the emitter.
    // Limit its angular span, and stay clear of the perspective singularity.
    float distanceScale = 0.75 * saturate(1.5 / (cosTheta + sinTheta / tan(push.maxRayAngle)));
    float3 endpoint = position + direction * min(push.rayLength, distance * distanceScale);
    if (push.lightType == 1u) {
        float startW = dot(projection.rowW, float4(position, 1.0));
        float endW = dot(projection.rowW, float4(endpoint, 1.0));
        // Very wide spots can cross the near plane even with the angular cap.
        // Clip once in world space before interpolating post-projection depth.
        if (endW < projection.originNear.w) {
            endpoint = lerp(position, endpoint, (startW - projection.originNear.w) / (startW - endW));
        }
    }
    return endpoint;
}

float3 vsmRayEndpoint(float3 position, ShadowProjection projection, ShadowPush push, uint ray, uint2 pixel)
{
    float3 toLight = push.lightType == 0u ? normalize(projection.rowZ.xyz) :
        normalize(projection.originNear.xyz - position);
    float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
    float3 up = cross(toLight, right);
    float2 disk = vsmDiskSample(ray, push.rayCount, pixel, push.samplePhase);
    return vsmRayEndpoint(position, projection, push, toLight, right * disk.x + up * disk.y);
}

float vsmRaySampleTime(uint step, ShadowPush push, float stepRandom, float invSamplesPerRay)
{
    if (step == push.samplesPerRay) { return 0.0; }
    float t = 1.0 - (float(step + 1u) - stepRandom) * invSamplesPerRay;
    return t * t;
}

float vsmRaySampleTime(uint step, ShadowPush push, float stepRandom)
{
    return vsmRaySampleTime(step, push, stepRandom, 1.0 / float(push.samplesPerRay));
}

float vsmRaySampleTime(uint step, ShadowPush push, uint2 pixel, float invSamplesPerRay)
{
    if (step == push.samplesPerRay) { return 0.0; }
    return vsmRaySampleTime(step, push, vsmSampleRandom(pixel, 2u, push.samplePhase), invSamplesPerRay);
}

float vsmRaySampleTime(uint step, ShadowPush push, uint2 pixel)
{
    return vsmRaySampleTime(step, push, pixel, 1.0 / float(push.samplesPerRay));
}

float3 vsmRayPosition(float3 position, float3 endpoint, float time, ShadowProjection projection, ShadowPush push)
{
    if (push.lightType != 0u) {
        float3 forward = normalize(projection.rowW.xyz);
        if (push.lightType == 2u) {
            float3 right, up;
            virtualShadowMapPointBasis(virtualShadowMapPointFace(position - projection.originNear.xyz), right, up, forward);
        }
        float startW = dot(position - projection.originNear.xyz, forward);
        float endW = dot(endpoint - projection.originNear.xyz, forward);
        // Linear interpolation after perspective division, reconstructed into
        // world space so a point-light trace can cross canonical cube faces.
        time = time * startW / lerp(endW, startW, time);
    }
    return lerp(position, endpoint, time);
}

float vsmWorldTexelSize(float3 position, ShadowProjection projection, uint pageSize, uint lightType)
{
    if (lightType == 0u) { return projection.metric.y; }
    float3 delta = abs(position - projection.originNear.xyz);
    float axial = lightType == 2u ? max(delta.x, max(delta.y, delta.z)) :
        dot(projection.rowW, float4(position, 1.0));
    return 2.0 * max(axial, projection.originNear.w) * projection.metric.x / float(projection.page.y * pageSize);
}

float3 vsmRayDither(float3 position, ShadowProjection projection, ShadowPush push, uint ray, uint2 pixel)
{
    float3 toLight = push.lightType == 0u ? normalize(projection.rowZ.xyz) :
        normalize(projection.originNear.xyz - position);
    float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
    float3 up = cross(toLight, right);
    float2 offset = float2(vsmSampleRandom(pixel, 3u + 2u * ray, push.samplePhase),
        vsmSampleRandom(pixel, 4u + 2u * ray, push.samplePhase)) - 0.5;
    // Dither the shadow-map texel independently of the source sample. This
    // antialiases contact edges narrower than a receiver-selected texel. A
    // world-space offset remains continuous when a ray crosses a cube face.
    return (right * offset.x + up * offset.y) * vsmWorldTexelSize(position, projection, push.pageSize, push.lightType);
}

bool vsmUsesReceiverFootprint(ShadowPush push)
{
    return (push.receiverLOD & 1u) != 0u;
}

float vsmPixelFootprint(float linearDepth, ViewConstants view, ShadowPush push)
{
    if (!vsmUsesReceiverFootprint(push)) { return 0.0; }
    return view.current.upProjection.w > 0.5 ? view.current.clipOrtho.z / view.current.viewport.z :
        2.0 * linearDepth * tan(view.current.viewport.w * 0.5) / view.current.viewport.z;
}

float vsmSearchLength(float3 position, float linearDepth, VSMReceiverBasis basis, ShadowPush push)
{
    if (push.rayLengthScale == 0.0) { return push.rayLength; }
    float distance = basis.isOrtho ? linearDepth : length(position - basis.eye);
    return min(push.rayLength, push.rayLengthScale * distance);
}

float vsmSearchLength(float3 position, float linearDepth, ViewConstants view, ShadowPush push)
{
    if (push.rayLengthScale == 0.0) { return push.rayLength; }
    // Perspective follows view-origin distance, as directional SMRT does in UE.
    // Orthographic receivers use axial depth so panning cannot change the cap.
    float distance = view.current.upProjection.w > 0.5 ? linearDepth : length(position - view.current.eye.xyz);
    return min(push.rayLength, push.rayLengthScale * distance);
}

struct VSMReceiverFootprint
{
    float width;
    float3 dx, dy;
    bool hasDifferentials;
};

VSMReceiverFootprint vsmReceiverFootprint(float3 position, float3 normal, float linearDepth,
    VSMReceiverBasis basis, ShadowPush push)
{
    VSMReceiverFootprint footprint = {};
    if (!vsmUsesReceiverFootprint(push)) { return footprint; }
    footprint.width = basis.isOrtho ? (basis.orthoHalfHeight * 2.0) * basis.invViewport.y :
        (2.0 * linearDepth * basis.tanHalfFov) * basis.invViewport.y;
    float3 viewRay = basis.isOrtho ? basis.forward : position - basis.eye;
    float denominator = dot(normal, viewRay);
    // An undefined or edge-on depth plane has no finite pixel differential.
    if (abs(denominator) <= 1e-5 * length(viewRay)) { return footprint; }
    footprint.dx = basis.right - viewRay * (dot(normal, basis.right) / denominator);
    footprint.dy = basis.up - viewRay * (dot(normal, basis.up) / denominator);
    footprint.hasDifferentials = true;
    return footprint;
}

float vsmProjectedReceiverFootprint(float3 position, VSMReceiverFootprint footprint, ShadowProjection projection, ShadowPush push)
{
    if (!footprint.hasDifferentials) { return footprint.width; }
    float3 dx = footprint.dx, dy = footprint.dy;
    float3 lightRight = normalize(projection.rowX.xyz), lightUp = normalize(projection.rowY.xyz);
    if (push.lightType != 0u) {
        float3 fromLight = position - projection.originNear.xyz;
        float3 lightForward = normalize(projection.rowW.xyz);
        if (push.lightType == 2u) {
            // Every face's demand dispatch must choose the same receiver mip.
            virtualShadowMapPointBasis(virtualShadowMapPointFace(fromLight), lightRight, lightUp, lightForward);
        }
        float axial = dot(fromLight, lightForward);
        dx -= fromLight * (dot(dx, lightForward) / axial);
        dy -= fromLight * (dot(dy, lightForward) / axial);
    }
    float2 a = float2(dot(dx, lightRight), dot(dx, lightUp));
    float2 b = float2(dot(dy, lightRight), dot(dy, lightUp));
    float aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
    float maximumSquared = 0.5 * (aa + bb + length(float2(aa - bb, 2.0 * ab)));
    // The smaller singular value preserves resolution along the compressed
    // axis at grazing incidence. An area ratio misses projection shear.
    return footprint.width * abs(a.x * b.y - a.y * b.x) * rsqrt(maximumSquared);
}

float vsmProjectedReceiverFootprint(float3 position, float3 normal, float linearDepth,
    VSMReceiverBasis basis, ShadowProjection projection, ShadowPush push)
{
    return vsmProjectedReceiverFootprint(position, vsmReceiverFootprint(position, normal, linearDepth, basis, push), projection, push);
}

float vsmProjectedReceiverFootprint(float3 position, float3 normal, float linearDepth,
    ViewConstants view, ShadowProjection projection, ShadowPush push)
{
    return vsmProjectedReceiverFootprint(position, normal, linearDepth, vsmMakeReceiverBasis(view), projection, push);
}

struct VSMReceiverBasisGroup
{
    float3 eye;
    float3 forward;
    float3 right;
    float3 up;
    float3 rightAspect;
    float tanHalfFov;
    float orthoHalfHeight;
    float2 jitter;
    float2 invViewport;
    float2 ndcScale;
    int2 viewportSize;
    float nearPlane;
    float farPlane;
    float farMinusNear;
    float nearTimesFar;
    uint isOrtho;
    uint reversed;
};

// Groupshared receiver basis, directional levels, and uniform caches have been
// replaced with direct register/scalar loads to eliminate intra-workgroup barriers.

ShadowProjection vsmGetCachedProjection(StructuredBuffer<ShadowProjection> projections, uint level, bool useGroupCache = false)
{
    return projections[level];
}

uint vsmReceiverLevel(float3 position, StructuredBuffer<ShadowProjection> projections,
    ShadowPush push, float footprint, bool useGroupCache = false)
{
    uint lastLevel = push.clipmapCount - 1u;
    float bias = min(asfloat(push.receiverLOD >> 1u), float(lastLevel));
    uint wholeLevels = uint(bias);
    float scale = exp2(bias - float(wholeLevels));
    float targetTexelSize = footprint * 0.5 * scale;
    for (uint level = 0u; level < lastLevel - wholeLevels; ++level) {
        ShadowProjection projection = vsmGetCachedProjection(projections, level, useGroupCache);
        if (vsmWorldTexelSize(position, projection, push.pageSize, push.lightType) < targetTexelSize) { continue; }
        if (push.lightType == 0u && bias > 0.0) {
            // Bias the first covering clipmap even when footprint matching is disabled.
            // Its snapped projection and one-texel inset match page marking and lookup.
            float3 projected = vsmProjectPosition(position, projection);
            projected.xy *= scale;
            if (!vsmContains(projected, projection, push.pageSize, true)) { continue; }
        }
        return level + wholeLevels;
    }
    return lastLevel;
}

float3 vsmPCFPosition(float3 position, float3 normal, ShadowProjection projection,
    ShadowPush push, int2 tap)
{
    float3 right = normalize(projection.rowX.xyz), up = normalize(projection.rowY.xyz);
    float3 toLight = push.lightType == 0u ? normalize(projection.rowZ.xyz) :
        normalize(projection.originNear.xyz - position);
    if (push.lightType == 2u) {
        float3 forward;
        virtualShadowMapPointBasis(virtualShadowMapPointFace(position - projection.originNear.xyz), right, up, forward);
    }
    float scale = push.pcfRadius * vsmWorldTexelSize(position, projection, push.pageSize, push.lightType);
    float3 offset = (right * float(tap.x) + up * float(tap.y)) * scale;
    float denominator = dot(normal, toLight);
    if (abs(denominator) > 1e-5) { offset -= toLight * (dot(normal, offset) / denominator); }
    return position + offset;
}

#if VSM_DIRECTIONAL_PROJECTION
float vsmDirectionalComparisonDepth(float3 normal, float3 projected, uint2 texel,
    float extent, float invResolution, float3 normX, float3 normY,
    float invDenom, float biasZ)
{
    float2 sampleNDC = (float2(texel) + 0.5) * invResolution - 1.0;
    float2 offset = (sampleNDC - projected.xy) * extent;
    float3 delta = normX * offset.x + normY * offset.y;
    float correction = -dot(normal, delta) * invDenom;
    return projected.z + correction + biasZ;
}

float vsmDirectionalComparisonDepth(float3 projected, uint2 texel,
    float extent, float invResolution, float2 normalSlope, float biasZ)
{
    float2 sampleNDC = fma(float2(texel), float2(invResolution), float2(0.5 * invResolution - 1.0));
    float2 offset = (sampleNDC - projected.xy) * extent;
    float correction = -dot(normalSlope, offset);
    return projected.z + correction + biasZ;
}
#endif

float vsmComparisonDepth(float3 position, float3 normal, float3 projected, uint2 texel,
    ShadowProjection projection, uint pageSize)
{
#if !VSM_DIRECTIONAL_PROJECTION
    if (projection.metric.w > 0.5) { return vsmPerspectiveReceiverDepth(position, normal, texel, projection, pageSize); }
#endif
    float denominator = dot(normal, normalize(projection.rowZ.xyz));
    float correction = 0.0;
    if (abs(denominator) > 1e-5) {
        float2 sampleNDC = (float2(texel) + 0.5) * (2.0 / float(projection.page.y * pageSize)) - 1.0;
        float2 offset = (sampleNDC - projected.xy) * projection.metric.x;
        float3 delta = normalize(projection.rowX.xyz) * offset.x + normalize(projection.rowY.xyz) * offset.y;
        correction = -dot(normal, delta) * length(projection.rowZ.xyz) / denominator;
    }
    return projected.z + correction + projection.metric.z;
}

