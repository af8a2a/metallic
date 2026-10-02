#include "../../../Source/Runtime/Render/Core/BunnyWireframeParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<BunnyWireframeParameters> gParams;

struct BunnyWireframeVertexOutput
{
    float4 position : SV_Position;
    float3 barycentric : TEXCOORD0;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float4 projectPosition(float3 worldPosition, BunnyWireframeGPUParams params)
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

float lineCoverage(float3 barycentric, float thickness, float smoothing)
{
    float3 deltas = fwidth(barycentric);
    float3 barys = smoothstep(deltas * thickness, deltas * (thickness + smoothing), barycentric);
    float minBary = min(barys.x, min(barys.y, barys.z));
    return 1.0 - minBary;
}

[shader("vertex")]
BunnyWireframeVertexOutput bunnyWireframeVertexMain(uint vertexIndex : SV_VertexID)
{
    BunnyWireframeVertexOutput invalid = (BunnyWireframeVertexOutput)0;
    invalid.position = float4(0, 0, 2, 1);
    if (!gParams.positions.contains(vertexIndex)) { return invalid; }
    float4 localPosition = gParams.positions.data[vertexIndex];
    const uint transformIndex = (uint)localPosition.w;
    if (!gParams.transforms.contains(transformIndex)) { return invalid; }
    BunnyTransform transform = gParams.transforms.data[transformIndex];
    BunnyWireframeGPUParams params = gParams.settings;
    float3 worldPosition =
        transform.world0.xyz * localPosition.x +
        transform.world1.xyz * localPosition.y +
        transform.world2.xyz * localPosition.z +
        transform.world3.xyz;

    BunnyWireframeVertexOutput output;
    output.position = projectPosition(worldPosition, params);

    uint corner = vertexIndex % 3;
    output.barycentric = corner == 0 ? float3(1.0, 0.0, 0.0) :
        (corner == 1 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0));
    return output;
}

[shader("fragment")]
float4 bunnyWireframeFragmentMain(BunnyWireframeVertexOutput input) : SV_Target0
{
    BunnyWireframeGPUParams params = gParams.settings;

    float thickness = max(params.settings.x, 0.01);
    float smoothing = max(params.settings.y, 0.01);
    float coverage = lineCoverage(input.barycentric, thickness, smoothing);
    float3 color = lerp(params.clearColor.xyz, params.wireColor.xyz, coverage);
    return float4(color, 1.0);
}
