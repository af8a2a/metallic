#include "../../../Source/Runtime/Render/Core/VisibilityMaterialParameters.h"
[[vk::push_constant]] ConstantBuffer<VisibilityMaterialParams> gParams;
using Metallic;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

typealias MaterialPush = VisibilityMaterialSettings;
#include "VisibilityStreamDecode.slang"
#define gOutput resolveDescriptor(gParams.output)
#define gVisibility resolveDescriptor(gParams.visibility)
#define gInstances resolveDescriptor(gParams.instances)
#define gMaterials resolveDescriptor(gParams.materials)
#define gRecords resolveDescriptor(gParams.records)
#define gMeshlets resolveDescriptor(gParams.meshlets)
#define gVertices resolveDescriptor(gParams.vertices)
#define gVertexIndices resolveDescriptor(gParams.vertexIndices)
#define gTriangles resolveDescriptor(gParams.triangles)
#define gGeometries resolveDescriptor(gParams.geometries)
#define gStreamRecords resolveDescriptor(gParams.streamRecords)
#define gGroups resolveDescriptor(gParams.groups)
#define gPages resolveDescriptor(gParams.pages)
#define gPageTable resolveDescriptor(gParams.pageTable)

float3 worldPosition(GPUDrivenPreviewInstance instance, float3 p)
{
    return instance.world0.xyz * p.x + instance.world1.xyz * p.y + instance.world2.xyz * p.z + instance.world3.xyz;
}

bool decodeTriangle(uint id, MaterialPush push, out uint instanceId, out float3 a, out float3 b, out float3 c)
{
    instanceId = 0; a = b = c = float3(0);
    uint index, triangle;
    if (!unpackVisibilityId(id, index, triangle)) { return false; }
    VisibleClusterRecord record;
    if (index < push.residentCount) {
        record = gRecords[index];
        if (visibleClusterSource(record) != kVisibleClusterSourceResident) { return false; }
        GPUDrivenPreviewMeshlet meshlet = gMeshlets[record.clusterIndex];
        if (triangle >= meshlet.ranges.w) { return false; }
        GPUDrivenPreviewGeometry geometry = gGeometries[record.dataIndex];
        uint3 local = uint3(loadPackedMeshletTriangleIndex(gTriangles, meshlet.ranges.z, triangle, 0),
            loadPackedMeshletTriangleIndex(gTriangles, meshlet.ranges.z, triangle, 1),
            loadPackedMeshletTriangleIndex(gTriangles, meshlet.ranges.z, triangle, 2));
        a = gVertices[geometry.payload.x + gVertexIndices[meshlet.ranges.x + local.x]].position.xyz;
        b = gVertices[geometry.payload.x + gVertexIndices[meshlet.ranges.x + local.y]].position.xyz;
        c = gVertices[geometry.payload.x + gVertexIndices[meshlet.ranges.x + local.z]].position.xyz;
    } else {
        index -= push.residentCount;
        if (index >= push.streamCount) { return false; }
        if (!decodeStreamTriangle(index, triangle, gStreamRecords, gGroups, gPages, gPageTable, streamPageWordCount(resolveDescriptor(gParams.streamParams)), record, a, b, c)) { return false; }
    }
    uint count, stride;
    gInstances.GetDimensions(count, stride);
    instanceId = record.instanceIndex;
    if (instanceId >= count) { return false; }
    GPUDrivenPreviewInstance instance = gInstances[instanceId];
    a = worldPosition(instance, a); b = worldPosition(instance, b); c = worldPosition(instance, c);
    return true;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void visibilityBufferMaterialMain(uint3 thread : SV_DispatchThreadID)
{
    MaterialPush push = gParams.settings;
    if (thread.x >= push.width || thread.y >= push.height) { return; }
    uint id = gVisibility.Load(int3(thread.xy, 0)), instanceId;
    float3 a, b, c;
    float3 color = float3(0.025, 0.035, 0.055);
    if (decodeTriangle(id, push, instanceId, a, b, c)) {
        GPUDrivenPreviewMaterial material = gMaterials[gInstances[instanceId].identity.y];
        float3 n = cross(b - a, c - a);
        n *= rsqrt(max(dot(n, n), 1e-30));
        if (dot(n, float3(push.eyeX, push.eyeY, push.eyeZ) - (a + b + c) / 3) < 0 && material.params.w > 0.5) { n = -n; }
        float3 base = material.baseColor.rgb;
        if (push.mode == 1) { color = base; }
        else if (push.mode == 2) { color = n * 0.5 + 0.5; }
        else if (push.mode == 3) {
            uint seed = (instanceId + 1) * 1664525u + 1013904223u;
            color = float3(seed & 255u, (seed >> 8) & 255u, (seed >> 16) & 255u) / 255;
        } else {
            float3 light = normalize(float3(-0.4, 0.8, 0.5));
            float3 view = normalize(float3(push.eyeX, push.eyeY, push.eyeZ) - (a + b + c) / 3);
            float3 halfway = normalize(light + view);
            float roughness = clamp(material.params.y, 0.05, 1.0);
            float metallic = saturate(material.params.x);
            float diffuse = 0.2 + 0.8 * saturate(dot(n, light));
            float specular = pow(saturate(dot(n, halfway)), lerp(128.0, 2.0, roughness * roughness));
            color = base * diffuse * (1 - 0.6 * metallic) + lerp(float3(0.04), base, metallic) * specular + material.emissive.rgb;
        }
        color = pow(saturate(color), float3(1.0 / 2.2));
    }
    gOutput[thread.xy] = float4(color, 1);
}
