// GPU meshlet culling compute shader.
// One thread per meshlet across all instances.
// Outputs a compact list of visible {instanceID, globalMeshletID} pairs.

struct CullUniforms {
    float4x4 viewProj;
    float4   cameraWorldPos;
    uint     totalDispatchCount;
    uint     instanceCount;
    uint     enableFrustumCull;
    uint     enableConeCull;
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

struct GPUMeshletBounds {
    float4 center_radius;
    float4 cone_apex_pad;
    float4 cone_axis_cutoff;
};

struct MeshletDrawInfo {
    uint instanceID;
    uint globalMeshletID;
};

ConstantBuffer<CullUniforms>        cullUniforms;    // buffer(0)
StructuredBuffer<InstanceData>      instances;       // buffer(1)
StructuredBuffer<GPUMeshletBounds>  meshletBounds;   // buffer(2)
RWStructuredBuffer<MeshletDrawInfo> visibleMeshlets; // buffer(3)
RWByteAddressBuffer                 counters;        // buffer(4)

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dtid : SV_DispatchThreadID) {
    uint flatID = dtid.x;
    if (flatID >= cullUniforms.totalDispatchCount)
        return;

    // Find which instance this thread belongs to via linear scan
    // (instance count is small, typically <100)
    uint instIdx = 0;
    uint remaining = flatID;
    for (uint i = 0; i < cullUniforms.instanceCount; i++) {
        if (remaining < instances[i].meshletCount) {
            instIdx = i;
            break;
        }
        remaining -= instances[i].meshletCount;
    }

    InstanceData inst = instances[instIdx];
    uint localMeshletIdx = remaining;
    uint globalMeshletID = inst.meshletStart + localMeshletIdx;

    GPUMeshletBounds b = meshletBounds[globalMeshletID];
    float3 centerOS = b.center_radius.xyz;
    float  radius   = b.center_radius.w;
    float3 coneAxis = b.cone_axis_cutoff.xyz;
    float  coneCutoff = b.cone_axis_cutoff.w;

    // Transform bounds center to world space
    float3 centerWS = mul(inst.worldMatrix, float4(centerOS, 1.0)).xyz;

    bool culled = false;

    // Frustum culling in clip space
    if (cullUniforms.enableFrustumCull != 0) {
        // Extract frustum planes from viewProj.
        // viewProj is transposed for Slang, so vp[i] = row i of original MVP.
        // This matches MvpToPlanes: mt = transpose(m), planes from mt[3] ± mt[0/1/2].
        float4x4 vp = cullUniforms.viewProj;
        float4 planes[6];
        planes[0] = vp[3] + vp[0]; // Left
        planes[1] = vp[3] - vp[0]; // Right
        planes[2] = vp[3] + vp[1]; // Bottom
        planes[3] = vp[3] - vp[1]; // Top
        planes[4] = vp[2];         // Near
        planes[5] = vp[3] - vp[2]; // Far

        // Scale radius by max axis scale of world matrix
        float3 sx = inst.worldMatrix[0].xyz;
        float3 sy = inst.worldMatrix[1].xyz;
        float3 sz = inst.worldMatrix[2].xyz;
        float maxScale = max(length(sx), max(length(sy), length(sz)));
        float worldRadius = radius * maxScale;

        [unroll]
        for (int i = 0; i < 6; i++) {
            float len = length(planes[i].xyz);
            if (len > 1e-6) {
                float d = (dot(planes[i].xyz, centerWS) + planes[i].w) / len;
                if (d < -worldRadius) {
                    culled = true;
                    break;
                }
            }
        }
    }

    // Cone/backface culling in world space
    if (!culled && cullUniforms.enableConeCull != 0) {
        float3 cameraPosWS = cullUniforms.cameraWorldPos.xyz;
        // Rotate cone axis to world space (direction only)
        float3 coneAxisWS = normalize(mul((float3x3)inst.worldMatrix, coneAxis));
        float3 cameraToCenter = centerWS - cameraPosWS;
        float lenSq = dot(cameraToCenter, cameraToCenter);
        if (lenSq > 1e-12) {
            float lenV = sqrt(lenSq);
            float lhs = dot(cameraToCenter, coneAxisWS);
            float3 sx2 = inst.worldMatrix[0].xyz;
            float3 sy2 = inst.worldMatrix[1].xyz;
            float3 sz2 = inst.worldMatrix[2].xyz;
            float maxScale2 = max(length(sx2), max(length(sy2), length(sz2)));
            float rhs = coneCutoff * lenV + radius * maxScale2;
            if (lhs >= rhs)
                culled = true;
        }
    }

    if (!culled) {
        uint slot;
        counters.InterlockedAdd(0, 1u, slot);
        MeshletDrawInfo info;
        info.instanceID = inst.instanceID;
        info.globalMeshletID = globalMeshletID;
        visibleMeshlets[slot] = info;
    }
}
