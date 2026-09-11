#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"
#include "../../../Shaders/Libraries/GPUDriven/GPUDrivenCullingCommon.slang"

[shader("compute")]
[numthreads(32, 1, 1)]
void gpuDrivenConeProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 16u) { return; }
    const float scales[8] = {1.0, 0.008, 0.001, 0.0001, 0.1, 10.0, 1000.0, 100000.0};
    float scale = scales[id.x % 8u];
    bool rotated = id.x >= 8u;
    float3 expected = rotated ? float3(-0.70710678118, 0.70710678118, 0.0) : float3(0, 1, 0);
    SceneTransform transform = {};
    transform.world0 = float4((rotated ? float3(0.70710678118, 0.70710678118, 0.0) : float3(1, 0, 0)) * scale, 0);
    transform.world1 = float4(expected * scale, 0);
    transform.world2 = float4(0, 0, scale, 0);
    GPUDrivenPreviewMeshlet meshlet = {};
    meshlet.coneAxisLodError = float4(0, 1, 0, 0);
    GPUDrivenPreviewParams params = {};
    params.cullingFlags = kCullMeshletNormalCone;
    params.eye = float4(expected, 0);
    bool front = meshletVisible(meshlet, 0, transform, params);
    params.eye = float4(-expected, 0);
    bool back = meshletVisible(meshlet, 0, transform, params);
    METALLIC_RESOURCE(RWStructuredBuffer<float4>, 0)[id.x * 2u] =
        float4(transformNormal(transform, float3(0, 1, 0)), front ? 1.0 : 0.0);
    METALLIC_RESOURCE(RWStructuredBuffer<float4>, 0)[id.x * 2u + 1u] = float4(expected, back ? 1.0 : 0.0);
}
