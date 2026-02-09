struct LightingUniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir;
    float4x4 invProj;
    uint     screenWidth;
    uint     screenHeight;
    uint     meshletCount;
    uint     materialCount;
    uint     textureCount;
    uint     pad0;
    uint     pad1;
    uint     pad2;
};

struct GPUMeshlet {
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
};

struct GPUMaterial {
    uint   baseColorTexIndex;
    uint   normalTexIndex;
    uint   metallicRoughnessTexIndex;
    uint   alphaMode;

    float4 baseColorFactor;
    float  metallicFactor;
    float  roughnessFactor;
    float  alphaCutoff;
    float  _pad;
};

static const uint INVALID_TEX = 0xFFFFFFFF;
static const uint INVALID_VIS = 0xFFFFFFFF;
static const uint kTextureArraySize = 96;
static const uint kInstanceBits = 5;
static const uint kVisibilityTriangleBits = 7;
static const uint kVisibilityTriangleMask = (1u << kVisibilityTriangleBits) - 1u;
static const uint kVisibilityMeshletShift = kVisibilityTriangleBits + kInstanceBits;
static const uint kVisibilityMeshletMask = (1u << (32u - kVisibilityMeshletShift)) - 1u;

// Buffer bindings
ConstantBuffer<LightingUniforms> lightUniforms;          // buffer(0)
StructuredBuffer<float>          positions;               // buffer(1)
StructuredBuffer<float>          normals;                 // buffer(2)
StructuredBuffer<GPUMeshlet>     meshlets;                // buffer(3)
StructuredBuffer<uint>           meshletVertices;         // buffer(4)
StructuredBuffer<uint>           meshletTriangles;        // buffer(5)
StructuredBuffer<float>          uvs;                     // buffer(6)
StructuredBuffer<uint>           meshletMaterialIDs;      // buffer(7)
StructuredBuffer<GPUMaterial>    materials;                // buffer(8)
// Texture bindings
Texture2D<uint>              visibilityBuffer;            // texture(0)
Texture2D<float>             depthBuffer;                 // texture(1)
RWTexture2D<float4>          outputTexture;               // texture(2)
Texture2D                    sceneTextures[kTextureArraySize]; // texture(3..)
SamplerState                 sceneSampler;                // sampler(0)

float3 projectVertex(float3 pos) {
    float4 clip = mul(lightUniforms.mvp, float4(pos, 1.0));
    float3 ndc = clip.xyz / clip.w;
    float2 screen;
    screen.x = (ndc.x * 0.5 + 0.5) * float(lightUniforms.screenWidth);
    screen.y = (1.0 - (ndc.y * 0.5 + 0.5)) * float(lightUniforms.screenHeight);
    return float3(screen, 1.0 / clip.w);
}

float3 loadPosition(uint vertexIndex) {
    return float3(
        positions[vertexIndex * 3 + 0],
        positions[vertexIndex * 3 + 1],
        positions[vertexIndex * 3 + 2]);
}

float3 loadNormal(uint vertexIndex) {
    return float3(
        normals[vertexIndex * 3 + 0],
        normals[vertexIndex * 3 + 1],
        normals[vertexIndex * 3 + 2]);
}

float2 loadUV(uint vertexIndex) {
    return float2(
        uvs[vertexIndex * 2 + 0],
        uvs[vertexIndex * 2 + 1]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= lightUniforms.screenWidth || pixel.y >= lightUniforms.screenHeight)
        return;

    uint vis = visibilityBuffer[pixel];
    if (vis == INVALID_VIS) {
        outputTexture[pixel] = float4(0.1, 0.2, 0.3, 1.0);
        return;
    }

    // Decode visibility
    uint meshletID  = (vis >> kVisibilityMeshletShift) & kVisibilityMeshletMask;
    uint triangleID = (vis >> kInstanceBits) & kVisibilityTriangleMask;
    // uint instanceID = vis & 0x1F; // unused for now

    if (meshletID >= lightUniforms.meshletCount) {
        outputTexture[pixel] = float4(0.1, 0.2, 0.3, 1.0);
        return;
    }

    // Fetch triangle vertex indices
    GPUMeshlet meshlet = meshlets[meshletID];
    if (triangleID >= meshlet.triangle_count) {
        outputTexture[pixel] = float4(0.1, 0.2, 0.3, 1.0);
        return;
    }

    uint packedTri = meshletTriangles[meshlet.triangle_offset + triangleID];
    uint localV0 = (packedTri >>  0) & 0xFF;
    uint localV1 = (packedTri >>  8) & 0xFF;
    uint localV2 = (packedTri >> 16) & 0xFF;

    if (localV0 >= meshlet.vertex_count || localV1 >= meshlet.vertex_count || localV2 >= meshlet.vertex_count) {
        outputTexture[pixel] = float4(0.1, 0.2, 0.3, 1.0);
        return;
    }

    uint globalV0 = meshletVertices[meshlet.vertex_offset + localV0];
    uint globalV1 = meshletVertices[meshlet.vertex_offset + localV1];
    uint globalV2 = meshletVertices[meshlet.vertex_offset + localV2];
    // Project vertices to screen space
    float3 s0 = projectVertex(loadPosition(globalV0));
    float3 s1 = projectVertex(loadPosition(globalV1));
    float3 s2 = projectVertex(loadPosition(globalV2));

    // Compute screen-space barycentrics
    float2 p = float2(pixel) + 0.5;
    float2 e0 = s1.xy - s0.xy;
    float2 e1 = s2.xy - s0.xy;
    float2 e2 = p - s0.xy;

    float det = e0.x * e1.y - e0.y * e1.x;
    if (abs(det) < 1e-10) {
        outputTexture[pixel] = float4(0.1, 0.2, 0.3, 1.0);
        return;
    }
    float invDet = 1.0 / det;
    float b1 = (e2.x * e1.y - e2.y * e1.x) * invDet;
    float b2 = (e0.x * e2.y - e0.y * e2.x) * invDet;
    float b0 = 1.0 - b1 - b2;

    // Perspective-correct interpolation
    float w0 = s0.z, w1 = s1.z, w2 = s2.z; // 1/clip.w
    float invW = b0 * w0 + b1 * w1 + b2 * w2;
    float W = 1.0 / invW;

    // Interpolate UVs with perspective correction
    float2 uv0 = loadUV(globalV0);
    float2 uv1 = loadUV(globalV1);
    float2 uv2 = loadUV(globalV2);
    float2 uv = (b0 * w0 * uv0 + b1 * w1 * uv1 + b2 * w2 * uv2) * W;

    // Interpolate normals with perspective correction
    float3 n0 = loadNormal(globalV0);
    float3 n1 = loadNormal(globalV1);
    float3 n2 = loadNormal(globalV2);
    float3 objNormal = normalize((b0 * w0 * n0 + b1 * w1 * n1 + b2 * w2 * n2) * W);

    // Transform normal to view space
    float3 viewNormal;
    viewNormal.x = dot(lightUniforms.modelView[0].xyz, objNormal);
    viewNormal.y = dot(lightUniforms.modelView[1].xyz, objNormal);
    viewNormal.z = dot(lightUniforms.modelView[2].xyz, objNormal);
    viewNormal = normalize(viewNormal);
    // Reconstruct view-space position from depth
    float depth = depthBuffer[pixel];
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5) / float(lightUniforms.screenWidth) * 2.0 - 1.0;
    ndc.y = 1.0 - (float(pixel.y) + 0.5) / float(lightUniforms.screenHeight) * 2.0;
    float4 clipPos = float4(ndc, depth, 1.0);
    float4 viewPos4 = mul(lightUniforms.invProj, clipPos);
    float3 viewPos = viewPos4.xyz / viewPos4.w;

    // Material lookup
    uint matID = meshletMaterialIDs[meshletID];
    if (matID >= lightUniforms.materialCount) {
        outputTexture[pixel] = float4(0.1, 0.2, 0.3, 1.0);
        return;
    }
    GPUMaterial mat = materials[matID];

    float4 baseColor = mat.baseColorFactor;
    if (mat.baseColorTexIndex != INVALID_TEX && mat.baseColorTexIndex < lightUniforms.textureCount) {
        baseColor *= sceneTextures[mat.baseColorTexIndex].Sample(sceneSampler, uv);
    }

    // Blinn-Phong lighting
    float3 N = normalize(viewNormal);
    float3 L = normalize(lightUniforms.lightDir.xyz);
    float3 V = normalize(-viewPos);
    float3 H = normalize(L + V);

    float ambient = 0.15;
    float diffuse = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 64.0);

    float3 color = baseColor.rgb * (ambient + diffuse * 0.75) + float3(1.0, 1.0, 1.0) * specular * 0.4;
    outputTexture[pixel] = float4(color, 1.0);
}
