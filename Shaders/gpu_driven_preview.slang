static const uint kMaxMeshletVertices = 128;
static const uint kMaxMeshletTriangles = 128;

struct GPUDrivenPreviewPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint positionBuffer;
    uint meshletBuffer;
    uint meshletVertexBuffer;
    uint meshletTriangleBuffer;
    uint paramsBuffer;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct GPUDrivenPreviewMeshlet
{
    uint vertexOffset;
    uint vertexCount;
    uint triangleOffset;
    uint triangleCount;
    uint primitiveIndex;
    uint materialIndex;
    uint padding0;
    uint padding1;
};

struct GPUDrivenPreviewParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 clearColor;
    uint mode;
    uint meshletCount;
    uint padding0;
    uint padding1;
};

struct GPUDrivenPreviewVertexOutput
{
    float4 position : SV_Position;
    nointerpolation float3 color : COLOR0;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float4 projectPosition(float3 worldPosition, GPUDrivenPreviewParams params)
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
    float ndcZ = (viewPosition.z - zNear) / (zFar - zNear);
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
    float w = viewPosition.z;
    return float4(
        viewPosition.x * yScale / aspect,
        viewPosition.y * yScale,
        ndcZ * w,
        w);
}

float3 hashColor(uint seed)
{
    float3 value = frac(sin(float3(seed, seed * 17u + 13u, seed * 31u + 7u) * 0.017453292) * 43758.5453);
    return lerp(float3(0.18, 0.34, 0.62), float3(0.95, 0.82, 0.35), value);
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void gpuDrivenPreviewMeshMain(
    uint3 groupID : SV_GroupID,
    out vertices GPUDrivenPreviewVertexOutput vertices[kMaxMeshletVertices],
    out indices uint3 triangles[kMaxMeshletTriangles],
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<float4>> positionHandle =
        DescriptorHandle<StructuredBuffer<float4>>(uint2(push.bufferShaderIndexBase + push.positionBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> meshletHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.bufferShaderIndexBase + push.meshletBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> meshletVertexHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletVertexBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> meshletTriangleHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletTriangleBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));

    StructuredBuffer<float4> positions = positionHandle;
    StructuredBuffer<GPUDrivenPreviewMeshlet> meshlets = meshletHandle;
    StructuredBuffer<uint> meshletVertices = meshletVertexHandle;
    StructuredBuffer<uint> meshletTriangles = meshletTriangleHandle;
    StructuredBuffer<GPUDrivenPreviewParams> paramsBuffer = paramsHandle;

    GPUDrivenPreviewParams params = paramsBuffer[0];
    uint meshletIndex = groupID.x;
    if (meshletIndex >= params.meshletCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    GPUDrivenPreviewMeshlet meshlet = meshlets[meshletIndex];
    uint vertexCount = min(meshlet.vertexCount, kMaxMeshletVertices);
    uint triangleCount = min(meshlet.triangleCount, kMaxMeshletTriangles);
    SetMeshOutputCounts(vertexCount, triangleCount);

    uint colorSeed = params.mode == 1 ? meshlet.primitiveIndex : meshletIndex;
    float3 color = hashColor(colorSeed + 1u);

    for (uint vertexIndex = 0; vertexIndex < kMaxMeshletVertices; ++vertexIndex) {
        if (vertexIndex < vertexCount) {
            uint positionIndex = meshletVertices[meshlet.vertexOffset + vertexIndex];
            float4 position = projectPosition(positions[positionIndex].xyz, params);
            position.y = -position.y;
            vertices[vertexIndex].position = position;
            vertices[vertexIndex].color = color;
        }
    }

    for (uint triangleIndex = 0; triangleIndex < kMaxMeshletTriangles; ++triangleIndex) {
        if (triangleIndex < triangleCount) {
            uint baseIndex = meshlet.triangleOffset + triangleIndex * 3u;
            triangles[triangleIndex] = uint3(
                meshletTriangles[baseIndex + 0u],
                meshletTriangles[baseIndex + 1u],
                meshletTriangles[baseIndex + 2u]);
        }
    }
}

[shader("fragment")]
float4 gpuDrivenPreviewFragmentMain(GPUDrivenPreviewVertexOutput input, uniform GPUDrivenPreviewPush push) : SV_Target0
{
    return float4(input.color, 1.0);
}
