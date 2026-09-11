#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"
#include "../../../Shaders/Libraries/GPUDriven/GPUDrivenCullingCommon.slang"

[shader("compute")]
[numthreads(32, 1, 1)]
void twoPassOcclusionProbeMain(uint3 id : SV_DispatchThreadID)
{
    // Full ceil-divided 17x9 pyramid: 153 + 45 + 15 + 6 + 2 + 1.
    for (uint i = id.x; i < 222u; i += 32u) {
        METALLIC_RESOURCE(RWStructuredBuffer<float>, 1)[i] = 0.25;
        METALLIC_RESOURCE(RWStructuredBuffer<float>, 2)[i] = 1.0;
    }
    AllMemoryBarrierWithGroupSync();
    GPUDrivenPreviewParams params = {};
    params.center = params.previousCenter = float4(0, 0, 1, 0);
    params.upProjection = params.previousUpProjection = float4(0, 1, 0, 0);
    params.viewport = params.previousViewport = float4(17.0 / 9.0, 17, 9, 1.57079632679);
    params.clipOrtho = params.previousClipOrtho = float4(0.1, 10.0, 12.0, 0);
    params.width = 17;
    params.height = 9;
    params.hzbMipCount = 6;
    params.hzbValid = 1;
    params.frameIndex = 1;
    params.cullingFlags = kCullMeshletHzb;
    GPUDrivenPreviewPush push = {};
    push.hzbBuffer0 = gComputeResources.resources[1].x;
    push.hzbBuffer1 = gComputeResources.resources[2].x;
    float3 center = float3(0, 0, 5);
    float radius = 0.25;
    uint instanceState = 1;
    // History chooses a phase; only this frame's depth may reject permanently.
    switch (id.x) {
    case 0: params.hzbValid = 0; break;                       // no history
    case 1: break;                                          // disoccluded this frame
    case 2: push.hzbBuffer1 = push.hzbBuffer0; break;         // occluded in both
    case 3: params.frameIndex = 0; break;                    // already drawn early
    case 4: instanceState = 3; break;                       // recovered instance
    case 5: params.cullingFlags = 0; break;                 // disabled
    case 6: center.z = 0.2; break;                          // near-plane crossing
    case 7: params.upProjection.w = params.previousUpProjection.w = 1; break;
    case 8: params.clipOrtho.w = params.previousClipOrtho.w = 1; break;
    case 9: instanceState = 0; break;                       // frustum rejected
    case 10: instanceState = 2; break;                      // still instance occluded
    case 11: params.hzbValid = 0; instanceState = 3; break;  // late uses current HZB
    default: break;
    }
    push.passIndex = 0;
    bool early = meshletVisibleInOcclusionPhase(center, radius, instanceState, params, push);
    push.passIndex = 1;
    bool late = meshletVisibleInOcclusionPhase(center, radius, instanceState, params, push);

    // An independent surface-sampling oracle exercises off-axis projection,
    // tiny/large bounds, odd sizes, orthographic views and subpixel jitter.
    float2 size = float2(799, 293);
    float3 testCenter = float3((float(id.x % 8u) - 3.5) * 1.7,
        (float(id.x / 8u) - 1.5) * 1.5, 6.0);
    float testRadius = (id.x & 1u) ? 2.0 : 0.0001;
    bool ortho = (id.x & 2u) != 0;
    float4 rect;
    bool hasRect = conservativeSphereScreenRect(testCenter, testRadius, size,
        size.x / size.y, 1.0, ortho ? 12.0 : 0.0, 0.1, rect);
    uint rectErrors = 0;
    float2 halfExtent = (ortho ? 6.0 : tan(0.5)) * float2(size.x / size.y, 1.0);
    for (uint sample = 0; sample < 512; ++sample) {
        float z = 1.0 - 2.0 * (float(sample) + 0.5) / 512.0;
        float phi = float(sample) * 2.39996322973;
        float3 direction = float3(sqrt(1.0 - z * z) * float2(cos(phi), sin(phi)), z);
        float3 point = testCenter + direction * testRadius;
        float2 xy = point.xy / (ortho ? 1.0 : point.z) / halfExtent;
        float2 pixel = (float2(xy.x, -xy.y) * 0.5 + 0.5) * size;
        pixel += float2((sample & 1u) ? 0.5 : -0.5, (sample & 2u) ? 0.5 : -0.5);
        if (all(pixel >= 0.0) && all(pixel < size)) {
            pixel = clamp(pixel, 0.0, size - 1.0);
            rectErrors += (!hasRect || any(pixel < rect.xy - 0.001) || any(pixel > rect.zw + 0.001)) ? 1u : 0u;
        }
    }
    rectErrors += conservativeSphereScreenRect(float3(0, 0, 0.2), 0.25,
        size, 1.0, 1.0, 0.0, 0.1, rect) ? 1u : 0u;

    uint depthErrors = 0;
    float nanValue = asfloat(0x7fc00000u);
    for (uint reverse = 0; reverse < 2; ++reverse) {
        bool reversedZ = reverse != 0;
        depthErrors += conservativeDepthOccluded(0.5, float4(0.5), reversedZ) ? 1u : 0u;
        depthErrors += conservativeDepthOccluded(0.5, float4(nanValue), reversedZ) ? 1u : 0u;
        depthErrors += conservativeDepthOccluded(nanValue, float4(0.5), reversedZ) ? 1u : 0u;
        depthErrors += conservativeDepthOccluded(0.5, float4(reversedZ ? 0.0 : 1.0), reversedZ) ? 1u : 0u;
        depthErrors += !conservativeDepthOccluded(0.5, float4(reversedZ ? 0.75 : 0.25), reversedZ) ? 1u : 0u;
    }
    SceneTransform transform = {};
    transform.world0 = float4(1, 0, 0, 0);
    transform.world1 = float4(1, 1, 0, 0);
    transform.world2 = float4(0, 0, 1, 0);
    float scale = transformMaxScale(transform);
    for (uint sample = 0; sample < 128; ++sample) {
        float angle = float(sample) * (6.28318530718 / 128.0);
        float3 point = transformPoint(transform, float3(cos(angle), sin(angle), 0));
        depthErrors += length(point) > scale + 1e-6 ? 1u : 0u;
    }
    METALLIC_RESOURCE(RWStructuredBuffer<float4>, 0)[id.x] =
        float4(early ? 1.0 : 0.0, late ? 1.0 : 0.0, float(rectErrors), float(depthErrors));
}
