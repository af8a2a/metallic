struct BindlessHeapPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
};

struct BunnyWireframeParams
{
    float4 centerScale;
    float4 viewport;
    float4 clearColor;
    float4 wireColor;
    float4 settings;
};

struct BunnyWireframeVertexOutput
{
    float4 position : SV_Position;
    float3 barycentric : TEXCOORD0;
};

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

    float3 centered = positions[vertexIndex].xyz - params.centerScale.xyz;
    float aspect = max(params.viewport.x, 0.001);
    float scale = params.centerScale.w;

    BunnyWireframeVertexOutput output;
    output.position = float4(centered.x * scale / aspect, centered.y * scale, centered.z * scale * 0.05, 1.0);

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
