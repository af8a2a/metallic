struct MaterialShaderObjectPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint positionBuffer;
    uint materialIndexBuffer;
    uint materialBuffer;
    uint paramsBuffer;
    uint vertexOffset;
    uint materialVariant;
    uint padding0;
    uint padding1;
};

struct MaterialCameraParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
};

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

float4 projectPosition(float3 worldPosition, MaterialCameraParams params)
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
MaterialVertexOutput materialShaderObjectVertexMain(uint vertexIndex : SV_VertexID, uniform MaterialShaderObjectPush push)
{
    DescriptorHandle<StructuredBuffer<float4>> positionHandle =
        DescriptorHandle<StructuredBuffer<float4>>(uint2(push.bufferShaderIndexBase + push.positionBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> materialIndexHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.materialIndexBuffer, 0));
    DescriptorHandle<StructuredBuffer<MaterialCameraParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<MaterialCameraParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));

    StructuredBuffer<float4> positions = positionHandle;
    StructuredBuffer<uint> materialIndices = materialIndexHandle;
    StructuredBuffer<MaterialCameraParams> paramsBuffer = paramsHandle;

    uint globalVertexIndex = push.vertexOffset + vertexIndex;

    MaterialVertexOutput output;
    output.position = projectPosition(positions[globalVertexIndex].xyz, paramsBuffer[0]);
    output.materialIndex = materialIndices[globalVertexIndex];
    return output;
}

[shader("fragment")]
float4 materialShaderObjectFragmentMain(MaterialVertexOutput input, uniform MaterialShaderObjectPush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<float4>> materialHandle =
        DescriptorHandle<StructuredBuffer<float4>>(uint2(push.bufferShaderIndexBase + push.materialBuffer, 0));
    StructuredBuffer<float4> materials = materialHandle;
    return materials[input.materialIndex];
}

[shader("fragment")]
float4 materialShaderObjectAlternateFragmentMain(MaterialVertexOutput input, uniform MaterialShaderObjectPush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<float4>> materialHandle =
        DescriptorHandle<StructuredBuffer<float4>>(uint2(push.bufferShaderIndexBase + push.materialBuffer, 0));
    StructuredBuffer<float4> materials = materialHandle;
    float4 baseColor = materials[input.materialIndex];
    float3 shifted = lerp(baseColor.rgb, baseColor.gbr, 0.45);
    return float4(shifted, baseColor.a);
}
