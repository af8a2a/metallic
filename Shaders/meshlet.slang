struct Uniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir; // view-space light direction (normalized), w unused
    float4   frustumPlanes[6]; // left, right, bottom, top, near, far
    float4   cameraPos;        // object-space camera position (xyz), w unused
    uint     enableFrustumCull;
    uint     enableConeCull;
    uint     pad0;
    uint     pad1;
};

struct GPUMeshlet {
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
};

struct GPUMeshletBounds {
    float4 center_radius;    // xyz=center, w=radius
    float4 cone_apex_pad;    // xyz=cone_apex, w=unused
    float4 cone_axis_cutoff; // xyz=cone_axis, w=cone_cutoff
};

// Global buffer bindings
ConstantBuffer<Uniforms>           uniforms;        // buffer(0)
StructuredBuffer<float>            positions;       // flat float array, 3 per vertex
StructuredBuffer<float>            normals;         // flat float array, 3 per vertex
StructuredBuffer<GPUMeshlet>       meshlets;
StructuredBuffer<uint>             meshletVertices;
StructuredBuffer<uint>             meshletTriangles;
StructuredBuffer<GPUMeshletBounds> meshletBounds;

struct MeshOutput {
    float4 clipPos    : SV_Position;
    float3 viewNormal : NORMAL;
    float3 viewPos    : TEXCOORD0;
};

static const uint kMaxVertices  = 64;
static const uint kMaxTriangles = 124;

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    in uint groupID       : SV_GroupID,
    in uint groupThreadID : SV_GroupThreadID,
    OutputVertices<MeshOutput, kMaxVertices>  outVerts,
    OutputIndices<uint3, kMaxTriangles>       outTris)
{
    GPUMeshlet m = meshlets[groupID];
    GPUMeshletBounds b = meshletBounds[groupID];
    float3 center = b.center_radius.xyz;
    float radius = b.center_radius.w;
    float3 coneAxis = b.cone_axis_cutoff.xyz;
    float coneCutoff = b.cone_axis_cutoff.w;

    bool culled = false;

    // Frustum culling: test bounding sphere against 6 planes.
    if (uniforms.enableFrustumCull != 0) {
        [unroll]
        for (int i = 0; i < 6; i++) {
            if (dot(uniforms.frustumPlanes[i].xyz, center) + uniforms.frustumPlanes[i].w + radius < 0.0) {
                culled = true;
                break;
            }
        }
    }

    // Cone/backface culling.
    if (!culled && uniforms.enableConeCull != 0) {
        // Conservative sphere-based test from meshoptimizer docs.
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

    if (groupThreadID == 0)
        SetMeshOutputCounts(culled ? 0u : m.vertex_count, culled ? 0u : m.triangle_count);

    // Ensure output counts are visible before any lane writes mesh outputs.
    GroupMemoryBarrierWithGroupSync();

    if (culled)
        return;

    if (groupThreadID < m.vertex_count) {
        uint vertexIndex = meshletVertices[m.vertex_offset + groupThreadID];

        float3 pos = float3(
            positions[vertexIndex * 3 + 0],
            positions[vertexIndex * 3 + 1],
            positions[vertexIndex * 3 + 2]);

        float3 nrm = float3(
            normals[vertexIndex * 3 + 0],
            normals[vertexIndex * 3 + 1],
            normals[vertexIndex * 3 + 2]);

        MeshOutput o;
        o.clipPos = mul(uniforms.mvp, float4(pos, 1.0));

        float4 viewPosition = mul(uniforms.modelView, float4(pos, 1.0));
        o.viewPos = viewPosition.xyz;

        // Transform normal by upper-left 3x3 of modelView
        float3 n;
        n.x = dot(uniforms.modelView[0].xyz, nrm);
        n.y = dot(uniforms.modelView[1].xyz, nrm);
        n.z = dot(uniforms.modelView[2].xyz, nrm);
        o.viewNormal = normalize(n);

        outVerts[groupThreadID] = o;
    }

    if (groupThreadID < m.triangle_count) {
        uint packed = meshletTriangles[m.triangle_offset + groupThreadID];
        uint v0 = (packed >>  0) & 0xFF;
        uint v1 = (packed >>  8) & 0xFF;
        uint v2 = (packed >> 16) & 0xFF;
        outTris[groupThreadID] = uint3(v0, v1, v2);
    }
}

[shader("fragment")]
float4 fragmentMain(MeshOutput input) : SV_Target {
    float3 N = normalize(input.viewNormal);
    float3 L = normalize(uniforms.lightDir.xyz);
    float3 V = normalize(-input.viewPos);
    float3 H = normalize(L + V);

    float3 baseColor = float3(0.7, 0.7, 0.72);

    float ambient = 0.15;
    float diffuse = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 64.0);

    float3 color = baseColor * (ambient + diffuse * 0.75) + float3(1.0, 1.0, 1.0) * specular * 0.4;
    return float4(color, 1.0);
}
