#language slang 2026
module VirtualShadowMaps;

import ShaderCore;
import Material;
import MaterialProgram;
import VirtualShadowMapGeometry;
import VirtualShadowMapCommon;

using Metallic;
using Metallic.Material;
using Metallic.Shadow;

struct ShadowPush
{
    uint geometry, materials, pages, coverageInputs;
    uint workItems, statistics, projections, atlas;
    uint depth, shadow, debug, view;
    uint arguments, triangleCount, clipmapCount, virtualPagesPerAxis;
    uint physicalPagesX, physicalPagesY, physicalPageCount, pageSize;
    uint workCapacity, batchCount, workBase, rasterCoarse;
    uint sampler, enabled, totalPages, lightType;
    uint previousPages, reuse, owners, cacheMode;
    uint coarseBatchCount, cancelledCount, cacheEpoch, dirtyBoundsCount;
    float lightRange;
    uint projectionFace, filterMode, pointFaces;
    float pcfRadius, rayLength, sourceRadius, angularRadius;
    uint rayCount, samplesPerRay, receiverFootprint, filterReserved;
};

struct ShadowProjection
{
    float4 rowX, rowY, rowZ, rowW, metric;
    uint4 page; // Virtual-page offset, pages per axis, level, coarse flag.
    float4 originNear;
};

struct ShadowFace
{
    uint atlas, pages, projections, statistics;
    uint physicalPagesX, clipmapCount, pageSize, reserved;
};

struct ShadowMaterial
{
    float4 alpha;
    uint4 resources;
    float4 uvX, uvY;
};

struct ShadowClipVertex
{
    float4 position;
    float2 uv;
};

struct ShadowRasterVertex
{
    float4 position : SV_Position;
    [vk::location(0)] float2 uv : TEXCOORD0;
    [vk::location(1)] nointerpolation uint materialIndex : TEXCOORD1;
};

float4 vsmProjectClip(float3 position, ShadowProjection projection)
{
    float4 p = float4(position, 1.0);
    return float4(dot(projection.rowX, p), dot(projection.rowY, p), dot(projection.rowZ, p), dot(projection.rowW, p));
}

float3 vsmProjectPosition(float3 position, ShadowProjection projection)
{
    float4 clip = vsmProjectClip(position, projection);
    return projection.metric.w > 0.5 ? clip.xyz / clip.w : clip.xyz;
}

// The selected face owns both demand and query. Receivers outside a punctual
// light's range/cone contribute no direct light; near-plane truncation is counted
// separately by the query pass and deliberately returns neutral visibility.
bool vsmReceiverDomain(float3 position, ShadowProjection projection, ShadowPush push, out bool nearPlane)
{
    nearPlane = false;
    if (projection.metric.w < 0.5) { return true; }
    float3 delta = position - projection.originNear.xyz;
    float distance = length(delta);
    if (push.lightRange > 0.0 && distance >= push.lightRange) { return false; }
    float axial = dot(projection.rowW, float4(position, 1.0));
    if (projection.metric.w > 1.5) {
        if (virtualShadowMapPointFace(delta) != push.projectionFace) { return false; }
    } else {
        float outerCosine = rsqrt(1.0 + projection.metric.x * projection.metric.x);
        if (axial < distance * outerCosine) { return false; }
    }
    nearPlane = axial <= projection.originNear.w;
    return !nearPlane;
}

bool vsmContains(float3 position, ShadowProjection projection, uint pageSize, bool inset)
{
    if (projection.metric.w > 0.5) {
        // A canonical point face includes its boundary. Insetting it would
        // reject receivers exactly on a cube edge without selecting a neighbor.
        return all(position.xy >= -1.0) && all(position.xy <= 1.0) && position.z >= 0.0 && position.z <= 1.0;
    }
    float margin = inset ? 1.0 / float(projection.page.y * pageSize) : 0.0;
    return all(position.xy >= -1.0 + margin) && all(position.xy < 1.0 - margin) &&
        position.z >= 0.0 && position.z <= 1.0;
}

uint2 vsmVirtualTexel(float3 position, ShadowProjection projection, uint pageSize)
{
    uint resolution = projection.page.y * pageSize;
    uint2 texel = uint2(floor((position.xy * 0.5 + 0.5) * float(resolution)));
    return projection.metric.w > 0.5 ? min(texel, uint2(resolution - 1u)) : texel;
}

uint vsmVirtualPage(uint2 texel, ShadowProjection projection, uint pageSize)
{
    uint2 page = texel / pageSize;
    return projection.page.x + page.y * projection.page.y + page.x;
}

float4 vsmPageBounds(uint virtualPage, ShadowProjection projection)
{
    uint local = virtualPage - projection.page.x;
    uint2 page = uint2(local % projection.page.y, local / projection.page.y);
    return float4(float2(page) * (2.0 / float(projection.page.y)) - 1.0,
        float2(page + 1u) * (2.0 / float(projection.page.y)) - 1.0);
}

float2 vsmAtlasPosition(float2 pagePosition, uint physicalPage, ShadowPush push)
{
    uint2 page = uint2(physicalPage % push.physicalPagesX, physicalPage / push.physicalPagesX);
    return (float2(page) + pagePosition) * (2.0 / float2(push.physicalPagesX, push.physicalPagesY)) - 1.0;
}

// Six homogeneous half-planes turn a triangle into at most nine vertices.
// Attributes interpolate along the original edge; preserving clip.w at raster
// output then supplies perspective-correct interpolation across the clipped fan.
uint vsmClipPlane(inout ShadowClipVertex polygon[9], uint count, uint axis, float edge, float side)
{
    if (count == 0u) { return 0u; }
    ShadowClipVertex clipped[9];
    uint clippedCount = 0u;
    ShadowClipVertex previous = polygon[count - 1u];
    float previousDistance = side * (previous.position[axis] - edge * previous.position.w);
    for (uint index = 0u; index < count; ++index) {
        ShadowClipVertex current = polygon[index];
        float currentDistance = side * (current.position[axis] - edge * current.position.w);
        if ((previousDistance < 0.0 && currentDistance > 0.0) ||
            (previousDistance > 0.0 && currentDistance < 0.0)) {
            float t = previousDistance / (previousDistance - currentDistance);
            ShadowClipVertex intersection;
            intersection.position = lerp(previous.position, current.position, t);
            if (axis == 2u && edge == 1.0) {
                // At reversed-Z near, z=w. For a triangle crossing the light,
                // interpolated w subtracts large values and loses precision;
                // z stays near the small near distance. Preserve z so later
                // page clipping does not amplify that error into self-shadow.
                intersection.position.w = intersection.position.z;
            } else {
                intersection.position[axis] = edge * intersection.position.w;
            }
            intersection.uv = lerp(previous.uv, current.uv, t);
            clipped[clippedCount++] = intersection;
        }
        if (currentDistance >= 0.0) { clipped[clippedCount++] = current; }
        previous = current;
        previousDistance = currentDistance;
    }
    for (uint index = 0u; index < clippedCount; ++index) { polygon[index] = clipped[index]; }
    return clippedCount;
}

uint vsmClipProjection(inout ShadowClipVertex polygon[9], float4 bounds, bool perspective)
{
    uint count = 3u;
    if (perspective) {
        count = vsmClipPlane(polygon, count, 2u, 1.0, -1.0); // Reversed-Z near: w-z >= 0.
        count = vsmClipPlane(polygon, count, 2u, 0.0, 1.0);  // Far: z >= 0.
    }
    count = vsmClipPlane(polygon, count, 0u, bounds.x, 1.0);
    count = vsmClipPlane(polygon, count, 0u, bounds.z, -1.0);
    count = vsmClipPlane(polygon, count, 1u, bounds.y, 1.0);
    return vsmClipPlane(polygon, count, 1u, bounds.w, -1.0);
}

// Uses the actual receiver projection, including temporal jitter in pixel units.
// clipOrtho.w selects reversed (1) or standard (0) receiver depth only; shadow
// projections always use reversed depth.
bool vsmReceiver(int2 pixel, ViewConstants view, Texture2D<float> depth,
    out float3 position, out float linearDepth)
{
    position = float3(0.0);
    linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= int2(view.current.viewport.yz))) { return false; }
    float value = depth.Load(int3(pixel, 0));
    bool reversed = view.current.clipOrtho.w > 0.5;
    if (value == (reversed ? 0.0 : 1.0)) { return false; }
    float nearPlane = view.current.clipOrtho.x;
    float farPlane = view.current.clipOrtho.y;
    if (view.current.upProjection.w > 0.5) {
        linearDepth = reversed ? lerp(farPlane, nearPlane, value) : lerp(nearPlane, farPlane, value);
    } else {
        float denominator = reversed ? nearPlane + value * (farPlane - nearPlane)
            : farPlane - value * (farPlane - nearPlane);
        linearDepth = nearPlane * farPlane / denominator;
    }
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    float2 uv = (float2(pixel) + 0.5 + jitter) / view.current.viewport.yz;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float halfHeight = view.current.upProjection.w > 0.5 ? view.current.clipOrtho.z * 0.5
        : linearDepth * tan(view.current.viewport.w * 0.5);
    position = view.current.eye.xyz + forward * linearDepth +
        halfHeight * (right * ndc.x * view.current.viewport.x + up * ndc.y);
    return true;
}

float3 vsmReceiverNormal(int2 pixel, ViewConstants view, Texture2D<float> depth,
    float3 position, float linearDepth)
{
    float3 left, right, top, bottom;
    float zl, zr, zt, zb;
    bool hasLeft = vsmReceiver(pixel - int2(1, 0), view, depth, left, zl);
    bool hasRight = vsmReceiver(pixel + int2(1, 0), view, depth, right, zr);
    bool hasTop = vsmReceiver(pixel - int2(0, 1), view, depth, top, zt);
    bool hasBottom = vsmReceiver(pixel + int2(0, 1), view, depth, bottom, zb);
    bool useLeft = hasLeft && (!hasRight || abs(zl - linearDepth) < abs(zr - linearDepth));
    bool useTop = hasTop && (!hasBottom || abs(zt - linearDepth) < abs(zb - linearDepth));
    float3 dx = useLeft ? position - left : (hasRight ? right - position : float3(0.0));
    float3 dy = useTop ? position - top : (hasBottom ? bottom - position : float3(0.0));
    float3 normal = cross(dy, dx);
    float lengthSquared = dot(normal, normal);
    return lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);
}

float vsmPerspectiveReceiverDepth(float3 position, float3 normal, uint2 virtualTexel,
    ShadowProjection projection, uint pageSize)
{
    float2 sampleNDC = (float2(virtualTexel) + 0.5) *
        (2.0 / float(projection.page.y * pageSize)) - 1.0;
    float3 forward = normalize(projection.rowW.xyz);
    float3 ray = forward + projection.metric.x *
        (normalize(projection.rowX.xyz) * sampleNDC.x + normalize(projection.rowY.xyz) * sampleNDC.y);
    float3 samplePosition = position;
    float denominator = dot(normal, ray);
    if (abs(denominator) > 1e-5 * length(ray)) {
        float distanceAlongRay = dot(normal, position - projection.originNear.xyz) / denominator;
        float3 intersection = projection.originNear.xyz + ray * distanceAlongRay;
        if (distanceAlongRay > 0.0 && all(isfinite(intersection))) { samplePosition = intersection; }
    }
    float axial = dot(projection.rowW, float4(samplePosition, 1.0));
    // A plane extrapolated to/before near cannot be meaningfully compared to
    // clipped caster depth. It is nearer than every stored sample.
    if (axial <= projection.originNear.w) { return 1.0; }
    float3 delta = samplePosition - projection.originNear.xyz;
    float distance = length(delta);
    float biasScale = max(1.0 - projection.metric.z / distance, projection.originNear.w / axial);
    float3 biasedPosition = projection.originNear.xyz + delta * biasScale;
    float4 clip = vsmProjectClip(biasedPosition, projection);
    return clip.z / clip.w;
}

uint vsmSampleHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float vsmSampleRandom(uint2 pixel, uint dimension)
{
    return float(vsmSampleHash(pixel.x ^ vsmSampleHash(pixel.y + dimension * 0x9e3779b9u)) >> 8) / 16777216.0;
}

float2 vsmDiskSample(uint ray, uint rayCount, uint2 pixel)
{
    // Scramble the stratified disk per receiver. A common disk point at one ray
    // shifts a hard shadow instead of estimating area-source visibility. Keep
    // the seed spatial: this pass has no temporal reconstruction of its own.
    float2 sample = frac(float2(float(ray) / float(rayCount), float(ray) * 0.61803398875) +
        float2(vsmSampleRandom(pixel, 0u), vsmSampleRandom(pixel, 1u)));
    float angle = sample.y * 6.28318530718;
    return sqrt(sample.x) * float2(cos(angle), sin(angle));
}

float3 vsmRayEndpoint(float3 position, ShadowProjection projection, ShadowPush push, uint ray, uint2 pixel)
{
    float3 toLight = push.lightType == 0u ? normalize(projection.rowZ.xyz) :
        normalize(projection.originNear.xyz - position);
    float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
    float3 up = cross(toLight, right);
    float2 disk = vsmDiskSample(ray, push.rayCount, pixel);
    float3 offset = right * disk.x + up * disk.y;
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
    float distanceScale = 0.75 * saturate(1.5 / (cosTheta + sinTheta / tan(0.03)));
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

float vsmRaySampleTime(uint step, ShadowPush push, uint2 pixel)
{
    if (step == push.samplesPerRay) { return 0.0; }
    float t = 1.0 - (float(step + 1u) - vsmSampleRandom(pixel, 2u)) / float(push.samplesPerRay);
    return t * t;
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
    float2 offset = float2(vsmSampleRandom(pixel, 3u + 2u * ray), vsmSampleRandom(pixel, 4u + 2u * ray)) - 0.5;
    // Dither the shadow-map texel independently of the source sample. This
    // antialiases contact edges narrower than a receiver-selected texel. A
    // world-space offset remains continuous when a ray crosses a cube face.
    return (right * offset.x + up * offset.y) * vsmWorldTexelSize(position, projection, push.pageSize, push.lightType);
}

float vsmPixelFootprint(float linearDepth, ViewConstants view, ShadowPush push)
{
    if (push.receiverFootprint == 0u) { return 0.0; }
    return view.current.upProjection.w > 0.5 ? view.current.clipOrtho.z / view.current.viewport.z :
        2.0 * linearDepth * tan(view.current.viewport.w * 0.5) / view.current.viewport.z;
}

float vsmProjectedReceiverFootprint(float3 position, float3 normal, float linearDepth,
    ViewConstants view, ShadowProjection projection, ShadowPush push)
{
    float footprint = vsmPixelFootprint(linearDepth, view, push);
    if (push.receiverFootprint == 0u) { return footprint; }
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float3 viewRay = view.current.upProjection.w > 0.5 ? forward : position - view.current.eye.xyz;
    float denominator = dot(normal, viewRay);
    // An undefined or edge-on depth plane has no finite pixel differential.
    if (abs(denominator) <= 1e-5 * length(viewRay)) { return footprint; }
    float3 dx = right - viewRay * (dot(normal, right) / denominator);
    float3 dy = up - viewRay * (dot(normal, up) / denominator);
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
    return footprint * abs(a.x * b.y - a.y * b.x) * rsqrt(maximumSquared);
}

uint vsmReceiverLevel(float3 position, StructuredBuffer<ShadowProjection> projections,
    ShadowPush push, float footprint)
{
    uint level = 0u;
    while (level + 1u < push.clipmapCount &&
        vsmWorldTexelSize(position, projections[level], push.pageSize, push.lightType) < footprint * 0.5) { ++level; }
    return level;
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

float vsmComparisonDepth(float3 position, float3 normal, float3 projected, uint2 texel,
    ShadowProjection projection, uint pageSize)
{
    if (projection.metric.w > 0.5) { return vsmPerspectiveReceiverDepth(position, normal, texel, projection, pageSize); }
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

struct ShadowLookup
{
    ShadowProjection projection;
    float3 projected;
    uint2 virtualTexel;
    float storedDepth;
    uint level;
    bool valid;
};

ShadowLookup vsmFilterLookup(float3 position, ShadowPush push, uint firstLevel)
{
    ShadowLookup result;
    result.valid = false;
    result.storedDepth = 0.0;
    uint atlasHandle = push.atlas, pagesHandle = push.pages, projectionsHandle = push.projections;
    uint statisticsHandle = push.statistics, physicalPagesX = push.physicalPagesX;
    if (push.lightType == 2u) {
        StructuredBuffer<ShadowProjection> own = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        StructuredBuffer<ShadowFace> faces = resolveUniform(ResourceHandle<StructuredBuffer<ShadowFace>>(push.pointFaces));
        ShadowFace face = faces[virtualShadowMapPointFace(position - own[0].originNear.xyz)];
        atlasHandle = face.atlas; pagesHandle = face.pages; projectionsHandle = face.projections;
        statisticsHandle = face.statistics; physicalPagesX = face.physicalPagesX;
    }
    StructuredBuffer<ShadowProjection> projections = resolveNonUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(projectionsHandle));
    StructuredBuffer<uint4> pages = resolveNonUniform(ResourceHandle<StructuredBuffer<uint4>>(pagesHandle));
    StructuredBuffer<uint> faceStatistics = resolveNonUniform(ResourceHandle<StructuredBuffer<uint>>(statisticsHandle));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint physicalPage = 0xffffffffu;
    result.level = push.clipmapCount;
    InterlockedAdd(statistics[30], 1u);
    if (faceStatistics[5] == 0u) {
        for (uint level = firstLevel; level < push.clipmapCount; ++level) {
            ShadowProjection projection = projections[level];
            float3 p = vsmProjectPosition(position, projection);
            if (!vsmContains(p, projection, push.pageSize, true)) { continue; }
            uint2 texel = vsmVirtualTexel(p, projection, push.pageSize);
            uint4 entry = pages[vsmVirtualPage(texel, projection, push.pageSize)];
            if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 1u) == 0u) { continue; }
            physicalPage = entry.x; result.level = level;
            result.projection = projection; result.projected = p; result.virtualTexel = texel;
            break;
        }
    }
    if (physicalPage == 0xffffffffu) {
        result.projection = projections[push.clipmapCount];
        result.projected = vsmProjectPosition(position, result.projection);
        // The complete coarse projection encloses all represented casters.
        // A trace segment outside that volume has no represented depth layer.
        if (!vsmContains(result.projected, result.projection, push.pageSize, false)) { return result; }
        uint4 entry = pages[result.projection.page.x];
        if ((entry.y & 2u) == 0u || (entry.w & 1u) == 0u) {
            InterlockedAdd(statistics[9], 1u);
            return result;
        }
        physicalPage = entry.x;
        result.virtualTexel = vsmVirtualTexel(result.projected, result.projection, push.pageSize);
        InterlockedAdd(statistics[31], 1u);
    }
    uint2 physical = uint2(physicalPage % physicalPagesX, physicalPage / physicalPagesX);
    uint2 texel = physical * push.pageSize + result.virtualTexel % push.pageSize;
    Texture2D<float> atlas = resolveNonUniform(ResourceHandle<Texture2D<float>>(atlasHandle));
    result.storedDepth = atlas.Load(int3(texel, 0));
    result.valid = true;
    return result;
}

float vsmRayDepth(float depth, ShadowLookup sample, float3 receiver, uint pageSize)
{
    ShadowProjection projection = sample.projection;
    if (projection.metric.w < 0.5) {
        return (depth - dot(projection.rowZ, float4(receiver, 1.0))) / length(projection.rowZ.xyz);
    }
    float2 ndc = (float2(sample.virtualTexel) + 0.5) *
        (2.0 / float(projection.page.y * pageSize)) - 1.0;
    float3 forward = normalize(projection.rowW.xyz);
    float3 texelRay = forward + projection.metric.x *
        (normalize(projection.rowX.xyz) * ndc.x + normalize(projection.rowY.xyz) * ndc.y);
    float scale = dot(projection.rowZ.xyz, forward);
    float offset = dot(projection.rowZ, float4(projection.originNear.xyz, 1.0));
    // Inverse radial depth gives every cube face the same continuous history
    // coordinate. Larger values, as for directional depth, are toward the light.
    return (depth - scale) / (offset * length(texelRay));
}

float vsmFilterVisibility(float3 position, float3 normal, uint receiverLevel, uint2 pixel,
    StructuredBuffer<ShadowProjection> projections, ShadowPush push)
{
    if (push.filterMode == 1u) {
        float sum = 0.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float3 tap = vsmPCFPosition(position, normal, projections[receiverLevel], push, int2(x, y));
                ShadowLookup sample = vsmFilterLookup(tap, push, receiverLevel);
                float compare = sample.valid ? vsmComparisonDepth(tap, normal, sample.projected,
                    sample.virtualTexel, sample.projection, push.pageSize) : 1.0;
                sum += !sample.valid || sample.storedDepth == 0.0 || compare >= sample.storedDepth ? 1.0 : 0.0;
            }
        }
        return sum / 9.0;
    }
    float visibility = 0.0;
    // A ray keeps its receiver's resolution. An angular footprint shrinks to
    // zero at contact and would request entire fine clipmaps, starving the
    // receiver pages and forcing nearly all samples onto the coarse map.
    for (uint ray = 0u; ray < push.rayCount; ++ray) {
        float3 endpoint = vsmRayEndpoint(position, projections[0], push, ray, pixel);
        float3 dither = vsmRayDither(position, projections[receiverLevel], push, ray, pixel);
        float maxSlope = push.lightType == 0u ? 0.05 : 0.05 * abs(
            1.0 / length(endpoint - projections[0].originNear.xyz) -
            1.0 / length(position - projections[0].originNear.xyz));
        float historyDepth = 0.0, historyTime = 0.0, historySlope = 0.0, previousReference = 0.0;
        bool hasHistory = false;
        bool blocked = false;
        // Trace back toward the receiver, including t=0. Depth discontinuities
        // can expose only the front layer: continue the preceding surface with
        // bounded slope through that gap instead of inventing an intersection.
        for (uint step = 0u; step <= push.samplesPerRay; ++step) {
            float time = vsmRaySampleTime(step, push, pixel);
            float3 point = vsmRayPosition(position, endpoint, time, projections[0], push);
            ShadowLookup sample = vsmFilterLookup(point + dither, push, receiverLevel);
            if (!sample.valid) { continue; }
            float compare = vsmComparisonDepth(point, normal, vsmProjectPosition(point, sample.projection),
                sample.virtualTexel, sample.projection, push.pageSize);
            float reference = vsmRayDepth(compare, sample, position, push.pageSize);
            float depth = push.lightType != 0u && sample.storedDepth == 0.0 ? 0.0 :
                vsmRayDepth(sample.storedDepth, sample, position, push.pageSize);
            if (!hasHistory) {
                // This ordinary comparison also retains blockers beyond the
                // bounded search; there is no separate hard-shadow tail.
                blocked = depth > reference;
                historyDepth = depth; historyTime = time;
                hasHistory = true;
            } else {
                float tolerance = abs(reference - previousReference) * 1.05;
                float reconstructed = depth;
                if (depth - reference > tolerance) {
                    reconstructed = historyDepth + historySlope * (time - historyTime);
                } else if (depth != historyDepth) {
                    historySlope = clamp((depth - historyDepth) / (time - historyTime), -maxSlope, maxSlope);
                    historyDepth = depth; historyTime = time;
                }
                float separation = reconstructed - reference;
                blocked = separation > 0.0 && separation <= tolerance;
            }
            if (blocked) { break; }
            previousReference = reference;
        }
        visibility += blocked ? 0.0 : 1.0;
    }
    return visibility / float(push.rayCount);
}

void vsmMarkPosition(float3 position, StructuredBuffer<ShadowProjection> projections,
    RWStructuredBuffer<uint4> pages, ShadowPush push, uint firstLevel)
{
    if (push.lightType == 2u && virtualShadowMapPointFace(position - projections[0].originNear.xyz) != push.projectionFace) { return; }
    // Match the query's first usable level. Requesting finer pages here spends
    // the admission budget on pages the receiver will never sample.
    for (uint level = firstLevel; level < push.clipmapCount; ++level) {
        ShadowProjection projection = projections[level];
        float3 projected = vsmProjectPosition(position, projection);
        if (!vsmContains(projected, projection, push.pageSize, true)) { continue; }
        uint page = vsmVirtualPage(vsmVirtualTexel(projected, projection, push.pageSize), projection, push.pageSize);
        InterlockedOr(pages[page].y, 1u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmResetMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    if (id.x < push.totalPages) {
        bool coarse = id.x == push.totalPages - 1u;
        uint4 entry = uint4(coarse ? 0u : 0xffffffffu, coarse && push.enabled != 0u ? 3u : 0u,
            coarse ? push.clipmapCount : id.x / (push.virtualPagesPerAxis * push.virtualPagesPerAxis), 2u);
        uint mappingFlags = 0u;
        if ((push.cacheMode & 1u) != 0u) {
            // Fine levels occupy fixed VPA^2 strides; local mips only use the
            // packed axis^2 prefix. The remaining slots must stay unmapped.
            StructuredBuffer<uint4> changes = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.reuse));
            uint4 change = uint4(0u);
            uint2 page = uint2(0u);
            uint oldIndex = coarse ? id.x : 0xffffffffu;
            if (!coarse) {
                uint stride = push.virtualPagesPerAxis * push.virtualPagesPerAxis;
                uint level = id.x / stride, local = id.x % stride;
                uint axis = push.lightType == 0u ? push.virtualPagesPerAxis : max(1u, push.virtualPagesPerAxis >> level);
                change = changes[level];
                if (local < axis * axis) {
                    page = uint2(local % axis, local / axis);
                    int2 oldPage = int2(page) + asint(change.xy);
                    if (all(uint2(oldPage) < uint2(axis))) {
                        oldIndex = level * stride + uint(oldPage.y) * axis + uint(oldPage.x);
                    }
                }
            }
            if (oldIndex != 0xffffffffu) {
                StructuredBuffer<uint4> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.previousPages));
                uint4 old = previous[oldIndex];
                bool mappingDirty = (push.cacheMode & 4u) != 0u || (coarse && push.dirtyBoundsCount != 0u);
                // Only resident pages need rectangle tests. Count consumes these
                // flags before replacing the same scratch words with scan ranks.
                if (old.x != 0xffffffffu) {
                    for (uint rectangle = 0u; !mappingDirty && rectangle < change.w; ++rectangle) {
                        uint4 bounds = changes[change.z + rectangle];
                        mappingDirty = all(page >= bounds.xy) && all(page <= bounds.zw);
                    }
                    mappingFlags = (mappingDirty ? 1u : 0u) | (oldIndex != id.x ? 2u : 0u);
                    entry.x = old.x;
                    bool dirty = mappingDirty || (old.w & 1u) == 0u;
                    entry.w = dirty ? 2u : 1u;
                }
            }
        }
        pages[id.x] = entry;
        owners[push.physicalPageCount + id.x] = mappingFlags;
    }
    // Each counter has exactly one writer, including the diagnostic constants.
    if (id.x < 32u) {
        uint value = id.x == 10u ? push.triangleCount : id.x == 11u ? push.clipmapCount :
            id.x == 12u ? push.physicalPageCount : id.x == 14u ? 1u - push.enabled :
            id.x == 22u ? ((push.cacheMode & 8u) != 0u ? 1u : 0u) :
            id.x == 23u ? push.cacheEpoch : id.x == 25u ? push.dirtyBoundsCount :
            id.x == 26u ? ((push.cacheMode & 4u) != 0u ? 1u : 0u) :
            id.x == 27u ? ((push.cacheMode & 2u) != 0u ? 1u : 0u) :
            id.x == 28u ? push.cancelledCount : 0u;
        statistics[id.x] = value;
    }
    if (id.x < push.physicalPageCount) { owners[id.x] = 0u; }
    if (id.x < (push.batchCount + push.coarseBatchCount) * 3u) {
        arguments[id.x] = id.x % 3u == 0u ? 0u : 1u;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmMarkMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    if (push.enabled == 0u) { return; }
    StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
    ViewConstants view = views[0];
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    float3 position;
    float linearDepth;
    if (!vsmReceiver(int2(id.xy), view, depth, position, linearDepth)) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    bool nearPlane;
    if (push.lightType == 2u && push.filterMode != 0u) {
        float3 delta = abs(position - projections[0].originNear.xyz);
        if ((push.lightRange > 0.0 && length(delta) >= push.lightRange) ||
            max(delta.x, max(delta.y, delta.z)) <= projections[0].originNear.w) { return; }
    } else if (!vsmReceiverDomain(position, projections[0], push, nearPlane)) { return; }
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    // Resolution belongs to the receiver, independently of the filter kernel.
    float3 normal = float3(0.0);
    if (push.receiverFootprint != 0u || push.filterMode == 1u) {
        normal = vsmReceiverNormal(int2(id.xy), view, depth, position, linearDepth);
    }
    uint receiverLevel = vsmReceiverLevel(position, projections, push,
        vsmProjectedReceiverFootprint(position, normal, linearDepth, view, projections[0], push));
    if (push.filterMode == 0u) {
        vsmMarkPosition(position, projections, pages, push, receiverLevel);
        return;
    }
    if (push.filterMode == 1u) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                vsmMarkPosition(vsmPCFPosition(position, normal, projections[receiverLevel], push, int2(x, y)),
                    projections, pages, push, receiverLevel);
            }
        }
    } else if (push.filterMode == 2u) {
        for (uint ray = 0u; ray < push.rayCount; ++ray) {
            float3 endpoint = vsmRayEndpoint(position, projections[0], push, ray, id.xy);
            float3 dither = vsmRayDither(position, projections[receiverLevel], push, ray, id.xy);
            for (uint step = 0u; step <= push.samplesPerRay; ++step) {
                float time = vsmRaySampleTime(step, push, id.xy);
                float3 point = vsmRayPosition(position, endpoint, time, projections[0], push);
                vsmMarkPosition(point + dither, projections, pages, push, receiverLevel);
            }
        }
    }
}

// Allocation scratch follows the physical owner table in the same buffer:
// local page ranks, block counts/prefixes, then free/evictable physical slots.
// Every scan invocation participates, including rounded dispatch tails.
groupshared uint3 gVsmAllocationScan[256];
uint3 vsmAllocationPrefix(uint3 value, uint lane)
{
    gVsmAllocationScan[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        uint3 previous = lane >= offset ? gVsmAllocationScan[lane - offset] : uint3(0u);
        GroupMemoryBarrierWithGroupSync();
        gVsmAllocationScan[lane] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return gVsmAllocationScan[lane] - value;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmCountMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform ShadowPush push)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> scratch = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint3 counts = uint3(0u);
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        counts.x = id.x + 1u < push.totalPages && (entry.y & 1u) != 0u ? 1u : 0u;
        if ((push.cacheMode & 1u) != 0u && entry.x != 0xffffffffu) {
            uint mappingFlags = scratch[push.physicalPageCount + id.x];
            counts.y = mappingFlags & 1u;
            counts.z = (mappingFlags >> 1u) & 1u;
        }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    if (id.x < push.totalPages) { scratch[push.physicalPageCount + id.x] = prefix.x; }
    if (lane == 255u) {
        uint3 total = prefix + counts;
        scratch[push.physicalPageCount + push.totalPages + group.x] = total.x;
        InterlockedAdd(statistics[18], total.y);
        InterlockedAdd(statistics[19], total.z);
    }
}

// The established allocation entry now prefixes per-block demand counts;
// its one group replaces the old serial traversal of the full virtual table.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAllocateMain(uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    RWStructuredBuffer<uint> scratch = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blocksPerLane = (blockCount + 255u) / 256u;
    uint first = lane * blocksPerLane, end = min(first + blocksPerLane, blockCount);
    uint offset = push.physicalPageCount + push.totalPages;
    uint count = 0u;
    for (uint block = first; block < end; ++block) { count += scratch[offset + block]; }
    uint prefix = vsmAllocationPrefix(uint3(count, 0u, 0u), lane).x;
    for (uint block = first; block < end; ++block) {
        uint value = scratch[offset + block];
        scratch[offset + block] = prefix;
        prefix += value;
    }
    if (lane == 0u) {
        uint requested = gVsmAllocationScan[255].x;
        uint allocated = min(requested, push.physicalPageCount - 1u);
        statistics[0] = requested;
        statistics[1] = allocated;
        statistics[2] = requested - allocated;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAdmitMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform ShadowPush push)
{
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint3 counts = uint3(0u);
    uint rankOffset = push.physicalPageCount;
    uint blockOffset = rankOffset + push.totalPages;
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        bool fine = id.x + 1u < push.totalPages;
        if (fine && (entry.y & 1u) != 0u) {
            uint rank = owners[blockOffset + group.x] + owners[rankOffset + id.x];
            // Keep requested parent coverage before spending the remaining
            // budget on refinement. A missing fine page can then use a nearby
            // coarser mip instead of the complete low-resolution scene map.
            rank = statistics[0] - 1u - rank;
            if (rank < push.physicalPageCount - 1u) { entry.y |= 2u; }
            else { entry.x = 0xffffffffu; entry.w = 2u; }
        }
        pages[id.x] = entry;
        if (entry.x != 0xffffffffu) { owners[entry.x] = id.x + 1u; }
        if (fine && (entry.y & 2u) != 0u) {
            counts.x = entry.x == 0xffffffffu ? 1u : 0u;
            counts.y = entry.x != 0xffffffffu && (entry.w & 2u) != 0u ? 1u : 0u;
            counts.z = entry.x != 0xffffffffu && (entry.w & 3u) == 1u ? 1u : 0u;
        }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    if (id.x < push.totalPages) { owners[rankOffset + id.x] = counts.x != 0u ? prefix.x : 0xffffffffu; }
    if (lane == 255u) {
        uint3 total = prefix + counts;
        owners[blockOffset + group.x] = total.x;
        InterlockedAdd(statistics[16], total.y);
        InterlockedAdd(statistics[17], total.z);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmPhysicalMain(uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blocksPerLane = (blockCount + 255u) / 256u;
    uint firstBlock = lane * blocksPerLane, endBlock = min(firstBlock + blocksPerLane, blockCount);
    uint blockOffset = push.physicalPageCount + push.totalPages;
    uint candidatesOffset = blockOffset + blockCount;
    uint slotsPerLane = (push.physicalPageCount - 1u + 255u) / 256u;
    uint firstSlot = 1u + lane * slotsPerLane, endSlot = min(firstSlot + slotsPerLane, push.physicalPageCount);
    uint3 counts = uint3(0u);
    for (uint block = firstBlock; block < endBlock; ++block) { counts.x += owners[blockOffset + block]; }
    for (uint physical = firstSlot; physical < endSlot; ++physical) {
        uint owner = owners[physical];
        if (owner == 0u) { ++counts.y; }
        else if ((pages[owner - 1u].y & 2u) == 0u) { ++counts.z; }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    uint freeCount = gVsmAllocationScan[255].y;
    for (uint block = firstBlock; block < endBlock; ++block) {
        uint count = owners[blockOffset + block];
        owners[blockOffset + block] = prefix.x;
        prefix.x += count;
    }
    // Retained admitted pages are pinned. Allocation consumes every genuinely
    // free slot in ascending order before evicting an unrequested old owner.
    for (uint physical = firstSlot; physical < endSlot; ++physical) {
        uint owner = owners[physical];
        if (owner == 0u) { owners[candidatesOffset + prefix.y++] = physical; }
        else if ((pages[owner - 1u].y & 2u) == 0u) { owners[candidatesOffset + freeCount + prefix.z++] = physical; }
    }
    if (lane == 0u) {
        uint missing = gVsmAllocationScan[255].x;
        statistics[20] = missing;
        statistics[16] += missing;
        statistics[21] = push.enabled != 0u && (pages[push.totalPages - 1u].w & 2u) != 0u ? 1u : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAssignMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uniform ShadowPush push)
{
    if (id.x >= push.totalPages) { return; }
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blockOffset = push.physicalPageCount + push.totalPages;
    uint localRank = owners[push.physicalPageCount + id.x];
    if (localRank == 0xffffffffu) { return; }
    uint rank = owners[blockOffset + group.x] + localRank;
    uint physical = owners[blockOffset + blockCount + rank];
    uint evicted = owners[physical];
    if (evicted != 0u) {
        // Only missing admitted pages run this phase. An evicted unrequested
        // page has no competing invocation reading/writing its page entry.
        uint4 old = pages[evicted - 1u];
        old.x = 0xffffffffu; old.w = 2u;
        pages[evicted - 1u] = old;
    }
    uint4 entry = pages[id.x];
    entry.x = physical; entry.w = 2u;
    pages[id.x] = entry;
    owners[physical] = id.x + 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    uint triangleIndex = push.workBase + id.x;
    if (triangleIndex >= push.triangleCount || push.enabled == 0u) { return; }
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    if (statistics[16] == 0u) { return; }
    VirtualShadowMapTriangle triangle;
    if (!loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint2> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.workItems));
    for (uint level = 0u; level < push.clipmapCount; ++level) {
        ShadowProjection projection = projections[level];
        float3 lo, hi;
        if (projection.metric.w > 0.5) {
            ShadowClipVertex polygon[9];
            for (uint vertex = 0u; vertex < 3u; ++vertex) {
                polygon[vertex].position = vsmProjectClip(triangle.vertices[vertex].position.xyz, projection);
                polygon[vertex].uv = float2(0.0);
            }
            // Projecting only the original vertices misses triangles crossing
            // near/w=0. Clip the complete face before any perspective division.
            uint count = vsmClipProjection(polygon, float4(-1.0, -1.0, 1.0, 1.0), true);
            if (count < 3u) { continue; }
            lo = hi = polygon[0].position.xyz / polygon[0].position.w;
            for (uint vertex = 1u; vertex < count; ++vertex) {
                float3 projected = polygon[vertex].position.xyz / polygon[vertex].position.w;
                lo = min(lo, projected); hi = max(hi, projected);
            }
        } else {
            float3 a = vsmProjectPosition(triangle.vertices[0].position.xyz, projection);
            float3 b = vsmProjectPosition(triangle.vertices[1].position.xyz, projection);
            float3 c = vsmProjectPosition(triangle.vertices[2].position.xyz, projection);
            lo = min(a, min(b, c)); hi = max(a, max(b, c));
        }
        if (hi.z < 0.0 || lo.z > 1.0 || any(hi.xy < -1.0) || any(lo.xy > 1.0)) { continue; }
        int axis = int(projection.page.y);
        // Clip the projected bounds before integer conversion; off-clipmap
        // world coordinates need not fit the page-index integer range.
        int2 first = min(int2(floor(saturate(lo.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
        int2 last = min(int2(floor(saturate(hi.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
        for (int y = first.y; y <= last.y; ++y) {
            for (int x = first.x; x <= last.x; ++x) {
                uint page = projection.page.x + uint(y * axis + x);
                uint4 entry = pages[page];
                if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 2u) == 0u) { continue; }
                uint index;
                InterlockedAdd(statistics[3], 1u, index);
                if (index < push.workCapacity) { work[index] = uint2(triangleIndex, page); }
                else { InterlockedAdd(statistics[5], 1u); }
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmFinalizeMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    uint count = min(statistics[3], push.workCapacity);
    if (id.x == 0u) { statistics[4] = count; }
    uint coarsePage = push.totalPages - 1u;
    bool redrawCoarse = statistics[21] != 0u;
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        if (push.enabled != 0u && (entry.y & 2u) != 0u && (entry.w & 2u) != 0u) {
            if (id.x == coarsePage || statistics[5] == 0u) { entry.w |= 1u; }
            else { entry.w &= ~1u; }
            pages[id.x] = entry;
        }
        if (id.x != coarsePage && entry.x != 0xffffffffu && (entry.w & 1u) != 0u) {
            InterlockedAdd(statistics[24], 1u);
        }
    }
    if (id.x < push.batchCount + push.coarseBatchCount) {
        bool coarse = id.x >= push.batchCount;
        uint first = (coarse ? id.x - push.batchCount : id.x) * 65535u;
        uint drawCount = coarse ? (redrawCoarse ? push.triangleCount : 0u) : count;
        arguments[id.x * 3u] = first < drawCount ? min(drawCount - first, 65535u) : 0u;
        arguments[id.x * 3u + 1u] = 1u;
        arguments[id.x * 3u + 2u] = 1u;
    }
}

struct ShadowClearVertex
{
    float4 position : SV_Position;
};

[shader("fragment")]
void vsmClearFragmentMain()
{
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmClearMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowClearVertex vertices[4], out indices uint3 triangles[2], uniform ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint owner = owners[group.x];
    bool clear = false;
    if (owner != 0u && push.enabled != 0u) {
        uint4 entry = pages[owner - 1u];
        clear = (entry.y & 2u) != 0u && (entry.w & 2u) != 0u;
    }
    SetMeshOutputCounts(clear ? 4u : 0u, clear ? 2u : 0u);
    if (!clear) { return; }
    if (lane < 4u) {
        float2 corner = float2(lane & 1u, lane >> 1u);
        vertices[lane].position = float4(vsmAtlasPosition(corner, group.x, push), 0.0, 1.0);
    }
    if (lane < 2u) { triangles[lane] = lane == 0u ? uint3(0u, 2u, 1u) : uint3(1u, 2u, 3u); }
}

groupshared ShadowClipVertex g_vsmPolygon[9];
groupshared uint g_vsmVertexCount;
groupshared uint g_vsmMaterial;
groupshared uint g_vsmPhysicalPage;
groupshared float4 g_vsmBounds;

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[9], out indices uint3 triangles[7], uniform ShadowPush push)
{
    if (lane == 0u) {
        g_vsmVertexCount = 0u;
        StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        uint triangleIndex = push.workBase + group.x;
        uint virtualPage = push.totalPages - 1u;
        if (push.rasterCoarse == 0u) {
            StructuredBuffer<uint2> work = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.workItems));
            uint2 item = work[push.workBase + group.x];
            triangleIndex = item.x;
            virtualPage = item.y;
        }
        VirtualShadowMapTriangle triangle;
        if (loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) {
            uint4 page = pages[virtualPage];
            ShadowProjection projection = projections[page.z];
            float4 bounds = vsmPageBounds(virtualPage, projection);
            if (projection.metric.w < 0.5 && push.rasterCoarse == 0u) {
                uint local = virtualPage - projection.page.x;
                uint2 pageXY = uint2(local % projection.page.y, local / projection.page.y);
                // Directional originNear.xy carries the snapped page-grid
                // origin. Form a canonical world-page translation before the
                // vertex dot product: subtracting rolling clipmap bounds after
                // projection changes clipping/depth rounding in cached pages.
                float2 center = projection.originNear.xy + float2(pageXY) + 0.5 - 0.5 * float(projection.page.y);
                float halfPage = 1.0 / float(projection.page.y);
                projection.rowX.w = -center.x * (2.0 * halfPage);
                projection.rowY.w = -center.y * (2.0 * halfPage);
                bounds = float4(-halfPage, -halfPage, halfPage, halfPage);
            }
            ShadowClipVertex polygon[9];
            for (uint i = 0u; i < 3u; ++i) {
                polygon[i].position = vsmProjectClip(triangle.vertices[i].position.xyz, projection);
                polygon[i].uv = triangle.vertices[i].uv.xy;
            }
            uint count = vsmClipProjection(polygon, bounds, projection.metric.w > 0.5);
            g_vsmVertexCount = count >= 3u ? count : 0u;
            g_vsmMaterial = triangle.materialIndex;
            g_vsmPhysicalPage = page.x;
            g_vsmBounds = bounds;
            for (uint i = 0u; i < count; ++i) { g_vsmPolygon[i] = polygon[i]; }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint vertexCount = g_vsmVertexCount;
    uint triangleCount = vertexCount >= 3u ? vertexCount - 2u : 0u;
    SetMeshOutputCounts(vertexCount, triangleCount);
    if (lane < vertexCount) {
        ShadowClipVertex vertex = g_vsmPolygon[lane];
        float2 pagePosition = saturate((vertex.position.xy / vertex.position.w - g_vsmBounds.xy) /
            (g_vsmBounds.zw - g_vsmBounds.xy));
        vertices[lane].position = float4(vsmAtlasPosition(pagePosition, g_vsmPhysicalPage, push) * vertex.position.w,
            vertex.position.z, vertex.position.w);
        vertices[lane].uv = vertex.uv;
        vertices[lane].materialIndex = g_vsmMaterial;
    }
    if (lane < triangleCount) { triangles[lane] = uint3(0u, lane + 1u, lane + 2u); }
}

[shader("fragment")]
void vsmDepthFragmentMain(ShadowRasterVertex input, bool frontFace : SV_IsFrontFace, uniform ShadowPush push)
{
    StructuredBuffer<ShadowMaterial> materials = resolveUniform(ResourceHandle<StructuredBuffer<ShadowMaterial>>(push.materials));
    ShadowMaterial material = materials[input.materialIndex];
    if (!frontFace && material.resources.y == 0u) { discard; }
    if (material.alpha.z < 0.5) { return; }
    CoverageContext context;
    context.uv = input.uv;
    context.baseAlpha = material.alpha.x;
    context.cutoff = material.alpha.y;
    context.alphaMode = material.alpha.z;
    CoverageProgram program;
    program.inputs = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.coverageInputs));
    MaterialInstanceRef instance = coverageInstanceRef(input.materialIndex);
    if (program.usesBaseAlpha(instance) && material.resources.x != 0xffffffffu) {
        Texture2D<float4> texture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(material.resources.x));
        SamplerState sampler = resolveUniform(SamplerHandle(push.sampler));
        float3 uv = float3(input.uv, 1.0);
        float2 textureUV = float2(dot(material.uvX.xyz, uv), dot(material.uvY.xyz, uv));
        context.baseAlpha *= texture.Sample(sampler, textureUV).a;
    }
    if (!program.evaluateCoverage(context, instance).accepted) { discard; }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmProjectMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
    ViewConstants view = views[0];
    if (any(id.xy >= uint2(view.current.viewport.yz))) { return; }
    RWTexture2D<float> shadow = resolveUniform(ResourceHandle<RWTexture2D<float>>(push.shadow));
    RWTexture2D<float4> debug = resolveUniform(ResourceHandle<RWTexture2D<float4>>(push.debug));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    float3 position;
    float linearDepth;
    bool sharedPointLayer = push.filterReserved != 0u && push.lightType == 2u;
    if (!vsmReceiver(int2(id.xy), view, depth, position, linearDepth)) {
        if (!sharedPointLayer || push.projectionFace == 0u) {
            shadow[id.xy] = 1.0;
            debug[id.xy] = float4(0.0, 0.0, 0.0, 1.0);
        }
        InterlockedAdd(statistics[8], 1u);
        return;
    }
    InterlockedAdd(statistics[6], 1u);
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    if (sharedPointLayer && virtualShadowMapPointFace(position - projections[0].originNear.xyz) != push.projectionFace) { return; }
    shadow[id.xy] = 1.0;
    debug[id.xy] = float4(0.0, 0.0, 0.0, 1.0);
    if (push.enabled == 0u) { return; }
    bool nearPlane;
    if (!vsmReceiverDomain(position, projections[0], push, nearPlane)) {
        if (nearPlane) { InterlockedAdd(statistics[29], 1u); }
        return;
    }
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    float3 normal = vsmReceiverNormal(int2(id.xy), view, depth, position, linearDepth);
    uint receiverLevel = vsmReceiverLevel(position, projections, push,
        vsmProjectedReceiverFootprint(position, normal, linearDepth, view, projections[0], push));
    uint selectedLevel = push.clipmapCount;
    uint firstLevel = 0xffffffffu;
    uint physicalPage = 0xffffffffu;
    uint2 virtualTexel = uint2(0u);
    float3 projected = float3(0.0);
    // Any fine-work overflow invalidates the entire partial fine atlas. The
    // independent coarse draw always traverses every caster slot.
    if (statistics[5] == 0u) {
        for (uint level = receiverLevel; level < push.clipmapCount; ++level) {
            ShadowProjection projection = projections[level];
            float3 p = vsmProjectPosition(position, projection);
            if (!vsmContains(p, projection, push.pageSize, true)) { continue; }
            if (firstLevel == 0xffffffffu) { firstLevel = level; }
            uint2 texel = vsmVirtualTexel(p, projection, push.pageSize);
            uint page = vsmVirtualPage(texel, projection, push.pageSize);
            uint4 entry = pages[page];
            if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 1u) == 0u) { continue; }
            physicalPage = entry.x;
            selectedLevel = level;
            projected = p;
            virtualTexel = texel;
            break;
        }
    }
    if (physicalPage == 0xffffffffu) {
        ShadowProjection coarse = projections[push.clipmapCount];
        projected = vsmProjectPosition(position, coarse);
        if (!vsmContains(projected, coarse, push.pageSize, false)) {
            shadow[id.xy] = 0.0;
            debug[id.xy] = float4(1.0, 0.0, 1.0, 1.0);
            InterlockedAdd(statistics[9], 1u);
            return;
        }
        physicalPage = 0u;
        virtualTexel = vsmVirtualTexel(projected, coarse, push.pageSize);
        InterlockedAdd(statistics[7], 1u);
        debug[id.xy] = float4(0.0, 0.0, 1.0, 1.0);
    } else {
        if (selectedLevel > firstLevel) { InterlockedAdd(statistics[13], 1u); }
        debug[id.xy] = float4(float(selectedLevel + 1u) / float(push.clipmapCount), 0.5, 0.0, 1.0);
    }
    ShadowProjection projection = projections[selectedLevel];
    if (push.filterMode != 0u) {
        shadow[id.xy] = vsmFilterVisibility(position, normal, receiverLevel, id.xy, projections, push);
        return;
    }
    float compareDepth;
    if (projection.metric.w > 0.5) {
        compareDepth = vsmPerspectiveReceiverDepth(position, normal, virtualTexel, projection, push.pageSize);
    } else {
        float3 axisDepth = normalize(projection.rowZ.xyz);
        float denominator = dot(normal, axisDepth);
        float correction = 0.0;
        if (abs(denominator) > 1e-5) {
            float2 sampleNDC = (float2(virtualTexel) + 0.5) *
                (2.0 / float(projection.page.y * push.pageSize)) - 1.0;
            float2 offset = (sampleNDC - projected.xy) * projection.metric.x;
            float3 deltaXYWorld = normalize(projection.rowX.xyz) * offset.x + normalize(projection.rowY.xyz) * offset.y;
            correction = -dot(normal, deltaXYWorld) * length(projection.rowZ.xyz) / denominator;
        }
        compareDepth = projected.z + correction + projection.metric.z;
    }
    uint2 physical = uint2(physicalPage % push.physicalPagesX, physicalPage / push.physicalPagesX);
    uint2 texel = physical * push.pageSize + virtualTexel % push.pageSize;
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
    float storedDepth = atlas.Load(int3(texel, 0));
    // The producer's scene-Z/far margin keeps real casters above zero. A receiver
    // plane extrapolated past the scene must not turn empty depth into a blocker.
    shadow[id.xy] = storedDepth == 0.0 || compareDepth >= storedDepth ? 1.0 : 0.0;
}
