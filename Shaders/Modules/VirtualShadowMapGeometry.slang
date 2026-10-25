#language slang 2026
module VirtualShadowMapGeometry;

import ShaderCore;
import GPUDriven;

using Metallic;
using Metallic.GPUDriven;

namespace Metallic.Shadow
{

public struct VirtualShadowMapVertex
{
    public float4 position;
    public float4 uv;
};

public struct VirtualShadowMapTriangle
{
    public VirtualShadowMapVertex vertices[3];
    public uint materialIndex;
    // Scalar padding matches the C++ 112-byte triangle stride.
    public uint reserved0;
    public uint reserved1;
    public uint reserved2;
};

struct VirtualShadowMapGeometryParameters
{
    uint streamed, triangles, candidates, instances;
    uint pages, pageTable, pageWordCount, triangleCapacity;
};

struct VirtualShadowMapStreamInstance
{
    uint primitiveIndex, materialIndex, visible, gpuSceneInstanceIndex;
    float4 world0, world1, world2, world3;
    float4 boundsCenterRadius;
};

public bool loadVirtualShadowMapTriangle(uint geometryResourceIndex, uint triangleIndex,
    out VirtualShadowMapTriangle triangle)
{
    triangle = (VirtualShadowMapTriangle)0;
    StructuredBuffer<VirtualShadowMapGeometryParameters> parameters =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapGeometryParameters>>(geometryResourceIndex));
    VirtualShadowMapGeometryParameters p = parameters[0];
    if (triangleIndex >= p.triangleCapacity) { return false; }
    if (p.streamed == 0u) {
        StructuredBuffer<VirtualShadowMapTriangle> triangles =
            resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapTriangle>>(p.triangles));
        triangle = triangles[triangleIndex];
        return true;
    }

    // Immutable candidates enumerate every terminal branch and every instance,
    // including offscreen casters. No camera-visible or camera-LOD list is read.
    StructuredBuffer<uint4> candidates = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(p.candidates));
    uint4 candidate = candidates[triangleIndex / 128u];
    StructuredBuffer<VirtualShadowMapStreamInstance> instances =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapStreamInstance>>(p.instances));
    VirtualShadowMapStreamInstance instance = instances[candidate.x];
    if (instance.visible == 0u) { return false; }
    StructuredBuffer<uint> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(p.pages));
    StructuredBuffer<uint2> pageTable = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(p.pageTable));
    float3 a, b, c;
    StreamTriangleDecodeContext context;
    if (!decodeStreamTriangleAtPage(candidate.y, candidate.z, triangleIndex % 128u,
        pages, pageTable, p.pageWordCount, a, b, c, context)) { return false; }
    float3 normals[3];
    float2 uvs[3];
    float4 tangents[3];
    decodeStreamAttributes(pages, context, normals, uvs, tangents, false);
    float3 positions[3] = {a, b, c};
    for (uint i = 0u; i < 3u; ++i) {
        float3 local = positions[i];
        triangle.vertices[i].position = float4(instance.world0.xyz * local.x +
            instance.world1.xyz * local.y + instance.world2.xyz * local.z + instance.world3.xyz, 1.0);
        triangle.vertices[i].uv = float4(uvs[i], 0.0, 0.0);
    }
    triangle.materialIndex = instance.materialIndex;
    return true;
}

} // namespace Metallic.Shadow
