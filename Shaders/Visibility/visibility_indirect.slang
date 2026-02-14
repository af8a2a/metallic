// GPU-driven visibility buffer mesh shader.
// Each threadgroup processes one entry from the visibleMeshlets buffer
// (populated by meshlet_cull.slang). No per-node CPU dispatch needed.

#include "../../Source/Rendering/visibility_constants.h"

struct GlobalUniforms {
    float4   lightDir;
    float4   lightColorIntensity;
};

struct InstanceData {
    float4x4 mvp;
    float4x4 modelView;
    float4x4 worldMatrix;
    uint     meshletStart;
    uint     meshletCount;
    uint     instanceID;
    uint     pad;
};

struct MeshletDrawInfo {
    uint instanceID;
    uint globalMeshletID;
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

// Buffer bindings
ConstantBuffer<GlobalUniforms>      globalUniforms;    // buffer(0)
StructuredBuffer<float>             positions;         // buffer(1)
StructuredBuffer<float>             normals;           // buffer(2)
StructuredBuffer<GPUMeshlet>        meshlets;          // buffer(3)
StructuredBuffer<uint>              meshletVertices;   // buffer(4)
StructuredBuffer<uint>              meshletTriangles;  // buffer(5)
StructuredBuffer<float>             _boundsPlaceholder;// buffer(6) — keeps indices aligned with visibility.slang
StructuredBuffer<float>             uvs;               // buffer(7)
StructuredBuffer<uint>              meshletMaterialIDs;// buffer(8)
StructuredBuffer<GPUMaterial>       materials;         // buffer(9)
StructuredBuffer<MeshletDrawInfo>   visibleMeshlets;   // buffer(10)
StructuredBuffer<InstanceData>      instanceData;      // buffer(11)

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
static const uint kInstanceBits = VISIBILITY_INSTANCE_BITS;
static const uint kInstanceMask = VISIBILITY_INSTANCE_MASK;
static const uint kVisibilityTriangleBits = VISIBILITY_TRIANGLE_BITS;
static const uint kVisibilityTriangleMask = VISIBILITY_TRIANGLE_MASK;
static const uint kVisibilityMeshletShift = VISIBILITY_MESHLET_SHIFT;
static const uint kVisibilityMeshletMask = VISIBILITY_MESHLET_MASK;

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
    // Each threadgroup = one visible meshlet from the cull pass output
    MeshletDrawInfo info = visibleMeshlets[groupID];
    InstanceData inst = instanceData[info.instanceID];
    uint globalMeshletID = info.globalMeshletID;

    GPUMeshlet m = meshlets[globalMeshletID];

    // No culling here — already done in meshlet_cull.slang

    bool visibilityOverflow = (globalMeshletID > kVisibilityMeshletMask) ||
                              (info.instanceID > kInstanceMask) ||
                              (m.triangle_count > (kVisibilityTriangleMask + 1u));

    if (groupThreadID == 0)
        SetMeshOutputCounts(visibilityOverflow ? 0u : m.vertex_count,
                            visibilityOverflow ? 0u : m.triangle_count);

    GroupMemoryBarrierWithGroupSync();

    if (visibilityOverflow)
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
        v.clipPos = mul(inst.mvp, float4(pos, 1.0));
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
                          info.instanceID;
        prim.materialID = matID;
        outPrims[groupThreadID] = prim;
    }
}

// Fragment-only buffer bindings
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
