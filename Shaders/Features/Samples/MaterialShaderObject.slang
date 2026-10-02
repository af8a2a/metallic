#include "../../../Source/Runtime/Render/Core/MaterialRasterParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<MaterialRasterParameters> gParams;

struct MaterialVertexOutput
{
    float4 position : SV_Position;
    nointerpolation uint materialIndex : TEXCOORD0;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float4 projectPosition(float3 worldPosition, MaterialShaderObjectGPUParams params)
{
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
    float ndcZ = saturate((viewPosition.z - zNear) / (zFar - zNear));
    if (params.clipOrtho.w > 0.5) {
        ndcZ = 1.0 - ndcZ;
    }
    float aspect = max(params.viewport.x, 0.001);

    if (params.upProjection.w > 0.5) {
        float halfHeight = max(params.clipOrtho.z * 0.5, 0.0001);
        return float4(
            viewPosition.x / (halfHeight * aspect),
            viewPosition.y / halfHeight,
            ndcZ,
            1.0);
    }

    float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = max(viewPosition.z, 0.0001);
    return float4(
        viewPosition.x * yScale / aspect,
        viewPosition.y * yScale,
        ndcZ * w,
        w);
}

[shader("vertex")]
MaterialVertexOutput materialShaderObjectVertexMain(uint vertexIndex : SV_VertexID)
{
    MaterialVertexOutput invalid = (MaterialVertexOutput)0;
    invalid.position = float4(0, 0, 2, 1);
    uint count = min(gParams.positions.length(), gParams.materialIndices.length());
    if (gParams.vertexOffset >= count || vertexIndex >= count - gParams.vertexOffset) { return invalid; }
    uint globalVertexIndex = gParams.vertexOffset + vertexIndex;
    float4 localPosition = gParams.positions.data[globalVertexIndex];
    uint transformIndex = (uint)localPosition.w;
    if (!gParams.transforms.contains(transformIndex)) { return invalid; }
    MaterialRasterTransform transform = gParams.transforms.data[transformIndex];
    float3 worldPosition =
        transform.world0.xyz * localPosition.x +
        transform.world1.xyz * localPosition.y +
        transform.world2.xyz * localPosition.z +
        transform.world3.xyz;

    MaterialVertexOutput output;
    output.position = projectPosition(worldPosition, gParams.camera);
    output.materialIndex = gParams.materialIndices.data[globalVertexIndex];
    return output;
}

[shader("fragment")]
float4 materialShaderObjectFragmentMain(MaterialVertexOutput input) : SV_Target0
{
    return gParams.materials.contains(input.materialIndex) ? gParams.materials.data[input.materialIndex] : float4(0);

}

[shader("fragment")]
float4 materialShaderObjectAlternateFragmentMain(MaterialVertexOutput input) : SV_Target0
{
    float4 baseColor = gParams.materials.contains(input.materialIndex) ? gParams.materials.data[input.materialIndex] : float4(0);
    float3 shifted = lerp(baseColor.rgb, baseColor.gbr, 0.45);
    return float4(shifted, baseColor.a);
}
