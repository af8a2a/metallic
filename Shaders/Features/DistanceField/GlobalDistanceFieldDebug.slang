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

void drawGlobalDistanceField<T : IGlobalDistanceFieldSampler>(uint2 pixel, T distanceField,
    GlobalDistanceFieldDomain domain, ViewCameraConstants camera, uint debugMode,
    RWTexture2D<float4> color, RWTexture2D<float4> trace)
{
    uint width, height;
    color.GetDimensions(width, height);
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    float3 forward = normalize(camera.center.xyz - camera.eye.xyz);
    float3 right = normalize(cross(forward, camera.upProjection.xyz));
    float3 up = cross(right, forward);
    float2 uv = (float2(pixel) + 0.5f) / float2(width, height);
    float2 ndc = float2(2.0f * uv.x - 1.0f, 1.0f - 2.0f * uv.y);
    float3 lateral = right * (ndc.x * camera.viewport.x) + up * ndc.y;
    float nearPlane = camera.clipOrtho.x;
    float farPlane = camera.clipOrtho.y;
    float3 rayStart = camera.eye.xyz;
    float3 rayDirection;
    float minTraceDistance = nearPlane;
    float maxTraceDistance = farPlane;
    if (camera.upProjection.w > 0.5f) {
        rayStart += forward * nearPlane + lateral * (0.5f * camera.clipOrtho.z);
        rayDirection = forward;
        minTraceDistance = 0.0f;
        maxTraceDistance = farPlane - nearPlane;
    } else {
        rayDirection = normalize(forward + lateral * tan(0.5f * camera.viewport.w));
    }

    GlobalDistanceFieldTrace result = traceGlobalDistanceField(distanceField, domain,
        rayStart, rayDirection, minTraceDistance, maxTraceDistance);
    float3 display = float3(0.025f, 0.035f, 0.055f);
    bool hit = result.classification == kGlobalDistanceFieldHit;
    if (result.classification == kGlobalDistanceFieldUnavailable) {
        display = float3(0.95f, 0.28f, 0.08f);
    } else if (debugMode == 1u && result.stepCount != 0u) {
        float heat = log2(1.0f + float(result.stepCount)) / log2(1.0f + float(kGlobalDistanceFieldMaxTraceSteps));
        display = lerp(float3(0.015f, 0.08f, 0.35f), float3(1.0f, 0.30f, 0.025f), heat);
    } else if (debugMode == 2u) {
        if (result.classification != kGlobalDistanceFieldOutside) {
            display = hit ? float3(0.18f, 0.75f, 0.45f) : float3(0.10f, 0.30f, 0.56f);
        }
    } else if (hit) {
        float3 worldPosition = rayStart + rayDirection * result.hitDistance;
        float3 normal = calculateGlobalDistanceFieldNormal(distanceField, domain, worldPosition, -rayDirection);
        float illumination = saturate(dot(normal, normalize(float3(0.4f, 0.8f, 0.6f))));
        display = (normal * 0.5f + 0.5f) * (0.35f + 0.65f * illumination);
    }
    color[pixel] = float4(display, 1.0f);
    trace[pixel] = float4(result.hitDistance, float(result.stepCount), float(result.classification), result.surfaceExpansion);
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
