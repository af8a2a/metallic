struct BindlessHeapPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
};

struct BunnyWireframeParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 clearColor;
    float4 wireColor;
    float4 settings;
};

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

float4 projectPosition(float3 worldPosition, BunnyWireframeParams params)
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
BunnyWireframeVertexOutput bunnyWireframeVertexMain(uint vertexIndex : SV_VertexID, uniform BindlessHeapPush bindless)
{
    DescriptorHandle<StructuredBuffer<BunnyWireframeParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<BunnyWireframeParams>>(uint2(bindless.bufferShaderIndexBase + 0, 0));
    DescriptorHandle<StructuredBuffer<float4>> positionsHandle =
        DescriptorHandle<StructuredBuffer<float4>>(uint2(bindless.bufferShaderIndexBase + 1, 0));

    StructuredBuffer<BunnyWireframeParams> paramsBuffer = paramsHandle;
    StructuredBuffer<float4> positions = positionsHandle;
    BunnyWireframeParams params = paramsBuffer[0];

    BunnyWireframeVertexOutput output;
    output.position = projectPosition(positions[vertexIndex].xyz, params);

    uint corner = vertexIndex % 3;
    output.barycentric = corner == 0 ? float3(1.0, 0.0, 0.0) :
        (corner == 1 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0));
    return output;
}

[shader("fragment")]
float4 bunnyWireframeFragmentMain(BunnyWireframeVertexOutput input, uniform BindlessHeapPush bindless) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<BunnyWireframeParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<BunnyWireframeParams>>(uint2(bindless.bufferShaderIndexBase + 0, 0));
    StructuredBuffer<BunnyWireframeParams> paramsBuffer = paramsHandle;
    BunnyWireframeParams params = paramsBuffer[0];

    float thickness = max(params.settings.x, 0.01);
    float smoothing = max(params.settings.y, 0.01);
    float coverage = lineCoverage(input.barycentric, thickness, smoothing);
    float3 color = lerp(params.clearColor.xyz, params.wireColor.xyz, coverage);
    return float4(color, 1.0);
}
