module GlobalDistanceFieldDebug;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceField;
import GlobalDistanceFieldPages;
using Metallic;
using Metallic.DistanceField;

struct GlobalDistanceFieldDebugParams
{
    ResourceHandle<Texture3D<float>> distanceField;
    SamplerHandle sampler;
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domain;
    ResourceHandle<RWTexture2D<float4>> color;
    ResourceHandle<RWTexture2D<float4>> trace;
    uint debugMode;
    uint padding0;
    uint padding1;
    ViewCameraConstants current;
};

struct GlobalDistanceFieldPagesDebugParams
{
    ResourceHandle<Texture3D<uint>> pageTable;
    ResourceHandle<Texture3D<float>> pageAtlas;
    SamplerHandle sampler;
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domain;
    ResourceHandle<RWTexture2D<float4>> color;
    ResourceHandle<RWTexture2D<float4>> trace;
    uint debugMode;
    uint padding;
    ViewCameraConstants current;
};

struct QuantizedGlobalDistanceFieldDebugParams
{
    ResourceHandle<Texture3D<uint>> pageTable;
    ResourceHandle<Texture3D<float>> pageAtlas;
    ResourceHandle<Texture3D<float>> coarseMip;
    SamplerHandle sampler;
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domain;
    ResourceHandle<RWTexture2D<float4>> color;
    ResourceHandle<RWTexture2D<float4>> trace;
    uint debugMode;
    ViewCameraConstants current;
};

struct GlobalDistanceFieldDebugRay
{
    float3 start;
    float3 direction;
    float minDistance;
    float maxDistance;
};

bool prepareGlobalDistanceFieldDebugRay(uint2 pixel, ViewCameraConstants camera, RWTexture2D<float4> color,
    out GlobalDistanceFieldDebugRay ray)
{
    uint width, height;
    color.GetDimensions(width, height);
    if (pixel.x >= width || pixel.y >= height) {
        ray = (GlobalDistanceFieldDebugRay)0;
        return false;
    }

    float3 forward = normalize(camera.center.xyz - camera.eye.xyz);
    float3 right = normalize(cross(forward, camera.upProjection.xyz));
    float3 up = cross(right, forward);
    float2 uv = (float2(pixel) + 0.5f) / float2(width, height);
    float2 ndc = float2(2.0f * uv.x - 1.0f, 1.0f - 2.0f * uv.y);
    float3 lateral = right * (ndc.x * camera.viewport.x) + up * ndc.y;
    float nearPlane = camera.clipOrtho.x;
    float farPlane = camera.clipOrtho.y;
    ray.start = camera.eye.xyz;
    ray.minDistance = nearPlane;
    ray.maxDistance = farPlane;
    if (camera.upProjection.w > 0.5f) {
        ray.start += forward * nearPlane + lateral * (0.5f * camera.clipOrtho.z);
        ray.direction = forward;
        ray.minDistance = 0.0f;
        ray.maxDistance = farPlane - nearPlane;
    } else {
        ray.direction = normalize(forward + lateral * tan(0.5f * camera.viewport.w));
    }
    return true;
}

static const float3 kClipmapColors[] = {
    float3(0.18f, 0.75f, 0.45f), float3(0.25f, 0.57f, 0.96f), float3(0.90f, 0.40f, 0.75f),
    float3(0.93f, 0.75f, 0.18f), float3(0.20f, 0.82f, 0.83f), float3(0.70f, 0.52f, 0.95f)
};

void writeGlobalDistanceFieldDebug<T : IGlobalDistanceFieldSampler>(uint2 pixel, T distanceField,
    GlobalDistanceFieldDomain domain, GlobalDistanceFieldDebugRay ray, GlobalDistanceFieldTrace result,
    uint debugMode, uint hitClipmapIndex, float traceDetail, RWTexture2D<float4> color, RWTexture2D<float4> trace)
{
    float3 display = float3(0.025f, 0.035f, 0.055f);
    bool hit = result.classification == kGlobalDistanceFieldHit;
    if (result.classification == kGlobalDistanceFieldUnavailable) {
        display = float3(0.95f, 0.28f, 0.08f);
    } else if (debugMode == 1u && result.stepCount != 0u) {
        float heat = saturate(log2(1.0f + float(result.stepCount)) / log2(1.0f + float(kGlobalDistanceFieldMaxTraceSteps)));
        display = lerp(float3(0.015f, 0.08f, 0.35f), float3(1.0f, 0.30f, 0.025f), heat);
    } else if (debugMode == 2u) {
        if (result.classification == kGlobalDistanceFieldPartial) {
            display = float3(0.29f, 0.20f, 0.41f);
        } else if (result.classification != kGlobalDistanceFieldOutside) {
            display = hit ? kClipmapColors[0] : float3(0.10f, 0.30f, 0.56f);
        }
    } else if (debugMode == 3u && hit) {
        display = kClipmapColors[hitClipmapIndex];
    } else if (hit) {
        float3 worldPosition = ray.start + ray.direction * result.hitDistance;
        float3 normal = calculateGlobalDistanceFieldNormal(distanceField, domain, worldPosition, -ray.direction);
        float illumination = saturate(dot(normal, normalize(float3(0.4f, 0.8f, 0.6f))));
        display = (normal * 0.5f + 0.5f) * (0.35f + 0.65f * illumination);
    }
    color[pixel] = float4(display, 1.0f);
    trace[pixel] = float4(result.hitDistance, float(result.stepCount), float(result.classification), traceDetail);
}

void drawGlobalDistanceField<T : IGlobalDistanceFieldSampler>(uint2 pixel, T distanceField,
    GlobalDistanceFieldDomain domain, ViewCameraConstants camera, uint debugMode,
    RWTexture2D<float4> color, RWTexture2D<float4> trace)
{
    GlobalDistanceFieldDebugRay ray;
    if (!prepareGlobalDistanceFieldDebugRay(pixel, camera, color, ray)) {
        return;
    }
    GlobalDistanceFieldTrace result = traceGlobalDistanceField(distanceField, domain,
        ray.start, ray.direction, ray.minDistance, ray.maxDistance);
    writeGlobalDistanceFieldDebug(pixel, distanceField, domain, ray, result, debugMode, 0u, result.surfaceExpansion,
        color, trace);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void globalDistanceFieldDebugMain(uint3 id : SV_DispatchThreadID)
{
    GlobalDistanceFieldDebugParams params = getParameters<GlobalDistanceFieldDebugParams>();
    StructuredBuffer<GlobalDistanceFieldDomain> domains = resolveUniform(params.domain);
    GlobalDistanceFieldDomain domain = domains[0];
    DenseGlobalDistanceFieldSampler distanceField = {resolveUniform(params.distanceField), resolveUniform(params.sampler), domain};
    drawGlobalDistanceField(id.xy, distanceField, domain, params.current, params.debugMode,
        resolveUniform(params.color), resolveUniform(params.trace));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void globalDistanceFieldPagesDebugMain(uint3 id : SV_DispatchThreadID)
{
    GlobalDistanceFieldPagesDebugParams params = getParameters<GlobalDistanceFieldPagesDebugParams>();
    StructuredBuffer<GlobalDistanceFieldDomain> domains = resolveUniform(params.domain);
    GlobalDistanceFieldDomain domain = domains[0];
    SparseGlobalDistanceFieldSampler distanceField = {resolveUniform(params.pageTable), resolveUniform(params.pageAtlas),
        resolveUniform(params.sampler), domain};
    drawGlobalDistanceField(id.xy, distanceField, domain, params.current, params.debugMode,
        resolveUniform(params.color), resolveUniform(params.trace));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void globalDistanceFieldQuantizedDebugMain(uint3 id : SV_DispatchThreadID)
{
    QuantizedGlobalDistanceFieldDebugParams params = getParameters<QuantizedGlobalDistanceFieldDebugParams>();
    StructuredBuffer<GlobalDistanceFieldDomain> domains = resolveUniform(params.domain);
    Texture3D<float> coarseMip = resolveUniform(params.coarseMip);
    QuantizedGlobalDistanceFieldSampler distanceField = {resolveUniform(params.pageTable), resolveUniform(params.pageAtlas),
        coarseMip, resolveUniform(params.sampler), domains[0], 0u};
    RWTexture2D<float4> color = resolveUniform(params.color);
    RWTexture2D<float4> trace = resolveUniform(params.trace);
    uint3 mipDimensions;
    coarseMip.GetDimensions(mipDimensions.x, mipDimensions.y, mipDimensions.z);
    uint clipmapCount = mipDimensions.z / mipDimensions.x;
    if (clipmapCount == 1u) {
        drawGlobalDistanceField(id.xy, distanceField, distanceField.domain, params.current, params.debugMode, color, trace);
        return;
    }

    GlobalDistanceFieldDebugRay ray;
    if (!prepareGlobalDistanceFieldDebugRay(id.xy, params.current, color, ray)) {
        return;
    }
    uint hitClipmapIndex;
    GlobalDistanceFieldTrace result = traceGlobalDistanceFieldClipmaps(distanceField, domains, clipmapCount,
        ray.start, ray.direction, ray.minDistance, ray.maxDistance, hitClipmapIndex);
    bool hit = result.classification == kGlobalDistanceFieldHit;
    if (hit) {
        distanceField.domain = domains[hitClipmapIndex];
        distanceField.clipmapIndex = hitClipmapIndex;
    }
    writeGlobalDistanceFieldDebug(id.xy, distanceField, distanceField.domain, ray, result, params.debugMode, hitClipmapIndex,
        hit ? float(hitClipmapIndex) : -1.0f, color, trace);
}
