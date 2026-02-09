struct Uniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir;
    float4   frustumPlanes[6];
    float4   cameraPos;
    uint     enableFrustumCull;
    uint     enableConeCull;
    uint     meshletBaseOffset;
    uint     instanceID;
};

struct GPUMeshlet {
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
};

struct GPUMeshletBounds {
    float4 center_radius;
    float4 cone_apex_pad;
    float4 cone_axis_cutoff;
};

struct GPUMaterial {
    uint   baseColorTexIndex;
    uint   normalTexIndex;
    uint   metallicRoughnessTexIndex;
    uint   alphaMode; // 0=OPAQUE, 1=MASK

    float4 baseColorFactor;
    float  metallicFactor;
    float  roughnessFactor;
    float  alphaCutoff;
    float  _pad;
};

// Buffer bindings (same indices as meshlet.slang)
ConstantBuffer<Uniforms>           uniforms;            // buffer(0)
StructuredBuffer<float>            positions;           // buffer(1)
StructuredBuffer<float>            normals;             // buffer(2)
StructuredBuffer<GPUMeshlet>       meshlets;            // buffer(3)
StructuredBuffer<uint>             meshletVertices;     // buffer(4)
StructuredBuffer<uint>             meshletTriangles;    // buffer(5)
StructuredBuffer<GPUMeshletBounds> meshletBounds;       // buffer(6)
StructuredBuffer<float>            uvs;                 // buffer(7)
StructuredBuffer<uint>             meshletMaterialIDs;  // buffer(8)

struct VisVertex {
    float4 clipPos : SV_Position;
    float2 uv      : TEXCOORD0;
};

struct VisPrimitive {
    uint visibility : TEXCOORD1;
    uint materialID : TEXCOORD2;
};

static const uint kMaxVertices  = 64;
static const uint kMaxTriangles = 124;
static const uint INVALID_TEX   = 0xFFFFFFFF;
static const uint kTextureArraySize = 96;
static const uint kInstanceBits = 11;
static const uint kInstanceMask = (1u << kInstanceBits) - 1u;
static const uint kVisibilityTriangleBits = 7;
static const uint kVisibilityTriangleMask = (1u << kVisibilityTriangleBits) - 1u;
static const uint kVisibilityMeshletShift = kVisibilityTriangleBits + kInstanceBits;
static const uint kVisibilityMeshletMask = (1u << (32u - kVisibilityMeshletShift)) - 1u;

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    in uint groupID       : SV_GroupID,
    in uint groupThreadID : SV_GroupThreadID,
    OutputVertices<VisVertex, kMaxVertices>    outVerts,
    OutputIndices<uint3, kMaxTriangles>        outTris,
    OutputPrimitives<VisPrimitive, kMaxTriangles> outPrims)
{
    uint globalMeshletID = groupID + uniforms.meshletBaseOffset;
    GPUMeshlet m = meshlets[globalMeshletID];
    GPUMeshletBounds b = meshletBounds[globalMeshletID];
    float3 center = b.center_radius.xyz;
    float radius = b.center_radius.w;
    float3 coneAxis = b.cone_axis_cutoff.xyz;
    float coneCutoff = b.cone_axis_cutoff.w;

    bool culled = false;

    // Frustum culling
    if (uniforms.enableFrustumCull != 0) {
        [unroll]
        for (int i = 0; i < 6; i++) {
            if (dot(uniforms.frustumPlanes[i].xyz, center) + uniforms.frustumPlanes[i].w + radius < 0.0) {
                culled = true;
                break;
            }
        }
    }

    // Cone/backface culling
    if (!culled && uniforms.enableConeCull != 0) {
        float3 cameraToCenter = center - uniforms.cameraPos.xyz;
        float lenSq = dot(cameraToCenter, cameraToCenter);
        if (lenSq > 1e-12) {
            float lenV = sqrt(lenSq);
            float lhs = dot(cameraToCenter, coneAxis);
            float rhs = coneCutoff * lenV + radius;
            if (lhs >= rhs)
                culled = true;
        }
    }

    bool visibilityOverflow = (globalMeshletID > kVisibilityMeshletMask) ||
                              (uniforms.instanceID > kInstanceMask) ||
                              (m.triangle_count > (kVisibilityTriangleMask + 1u));

    if (groupThreadID == 0)
        SetMeshOutputCounts((culled || visibilityOverflow) ? 0u : m.vertex_count,
                            (culled || visibilityOverflow) ? 0u : m.triangle_count);

    GroupMemoryBarrierWithGroupSync();

    if (culled || visibilityOverflow)
        return;

    uint matID = meshletMaterialIDs[globalMeshletID];

    if (groupThreadID < m.vertex_count) {
        uint vertexIndex = meshletVertices[m.vertex_offset + groupThreadID];

        float3 pos = float3(
            positions[vertexIndex * 3 + 0],
            positions[vertexIndex * 3 + 1],
            positions[vertexIndex * 3 + 2]);

        float2 texcoord = float2(
            uvs[vertexIndex * 2 + 0],
            uvs[vertexIndex * 2 + 1]);

        VisVertex v;
        v.clipPos = mul(uniforms.mvp, float4(pos, 1.0));
        v.uv = texcoord;
        outVerts[groupThreadID] = v;
    }

    if (groupThreadID < m.triangle_count) {
        uint packed = meshletTriangles[m.triangle_offset + groupThreadID];
        uint v0 = (packed >>  0) & 0xFF;
        uint v1 = (packed >>  8) & 0xFF;
        uint v2 = (packed >> 16) & 0xFF;
        outTris[groupThreadID] = uint3(v0, v1, v2);

        VisPrimitive prim;
        prim.visibility = (globalMeshletID << kVisibilityMeshletShift) |
                          (groupThreadID << kInstanceBits) |
                          uniforms.instanceID;
        prim.materialID = matID;
        outPrims[groupThreadID] = prim;
    }
}

// Fragment-only buffer bindings
StructuredBuffer<GPUMaterial> materials;
Texture2D sceneTextures[kTextureArraySize];
SamplerState sceneSampler;

[shader("fragment")]
uint fragmentMain(VisVertex vertIn, VisPrimitive primIn) : SV_Target {
    GPUMaterial mat = materials[primIn.materialID];

    // Alpha test for MASK materials
    if (mat.alphaMode == 1) {
        float4 baseColor = mat.baseColorFactor;
        if (mat.baseColorTexIndex != INVALID_TEX)
            baseColor *= sceneTextures[mat.baseColorTexIndex].Sample(sceneSampler, vertIn.uv);
        if (baseColor.a < mat.alphaCutoff)
            discard;
    }

    return primIn.visibility;
}
