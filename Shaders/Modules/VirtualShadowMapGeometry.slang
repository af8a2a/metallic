#language slang 2026
module VirtualShadowMapGeometry;

import ShaderCore;
import GPUDriven;
import VisibilityTessellation;

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
    uint normals, tessellation, tessellationFactor, subdivisionDepth;
};

void displaceVirtualShadowMapTriangle(VirtualShadowMapGeometryParameters parameters,
    uint microTriangle, float3 normals[3], inout VirtualShadowMapTriangle triangle)
{
    StructuredBuffer<uint> data = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(parameters.tessellation));
    TessMaterial material = tessMaterial(data, triangle.materialIndex);
    if (material.enabled == 0u) { return; }
    uint3 rates = uint3(parameters.tessellationFactor);
    uint4 pattern = tessPattern(data, rates);
    uint leaf = microTriangle / pattern.w;
    uint3 indices = tessTriangle(data, pattern, microTriangle % pattern.w);
    float3 positions[3];
    float2 uvs[3];
    for (uint i = 0u; i < 3u; ++i) {
        positions[i] = triangle.vertices[i].position.xyz;
        uvs[i] = triangle.vertices[i].uv.xy;
    }
    // Uniform shadow patches use the raster's dyadic domain and canonical
    // shared edges without depending on its camera-visible adaptive leaf list.
    uint3 patch = uint3(0u, kTessDomainScale, kTessDomainScale << 16u);
    for (uint depth = 0u; depth < parameters.subdivisionDepth; ++depth) {
        uint3 children[4];
        tessSplitChildren(patch, 7u, children);
        uint child = (leaf >> (2u * (parameters.subdivisionDepth - depth - 1u))) & 3u;
        patch = children[child];
    }
    for (uint i = 0u; i < 3u; ++i) {
        float3 bary = tessPatchDomain(tessBary(data, pattern, indices[i]), rates, patch, positions);
        float3 position, normal;
        float2 uv;
        tessSourceDomain(bary, positions, normals, uvs, position, normal, uv);
        triangle.vertices[i].position = float4(position + normal * tessHeight(material, uv), 1.0);
        triangle.vertices[i].uv = float4(uv, 0.0, 0.0);
    }
}

void orientVirtualShadowMapTriangle(bool reflected, inout VirtualShadowMapTriangle triangle)
{
    // Preserve authored front faces for base meshes and displaced patches alike.
    // Source interpolation always precedes the final reflected output winding.
    if (reflected) {
        VirtualShadowMapVertex vertex = triangle.vertices[1];
        triangle.vertices[1] = triangle.vertices[2];
        triangle.vertices[2] = vertex;
    }
}

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
        if (p.tessellation == 0xffffffffu) {
            triangle = triangles[triangleIndex];
            orientVirtualShadowMapTriangle(triangle.reserved0 != 0u, triangle);
            return true;
        }
        StructuredBuffer<uint2> candidates = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(p.candidates));
        uint2 candidate = candidates[triangleIndex];
        triangle = triangles[candidate.x];
        StructuredBuffer<float4> sourceNormals = resolveUniform(ResourceHandle<StructuredBuffer<float4>>(p.normals));
        float3 normals[3];
        for (uint i = 0u; i < 3u; ++i) { normals[i] = sourceNormals[candidate.x * 3u + i].xyz; }
        displaceVirtualShadowMapTriangle(p, candidate.y, normals, triangle);
        orientVirtualShadowMapTriangle(triangle.reserved0 != 0u, triangle);
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
    // Ordinary shadow Coverage needs only UVs; displacement also needs the
    // published authored/generated normal stream used by raster tessellation.
    decodeStreamAttributes(pages, context, normals, uvs, tangents, p.tessellation != 0xffffffffu);
    float3 positions[3] = {a, b, c};
    bool reflected = dot(cross(instance.world0.xyz, instance.world1.xyz), instance.world2.xyz) < 0.0;
    for (uint i = 0u; i < 3u; ++i) {
        float3 local = positions[i];
        triangle.vertices[i].position = float4(instance.world0.xyz * local.x +
            instance.world1.xyz * local.y + instance.world2.xyz * local.z + instance.world3.xyz, 1.0);
        triangle.vertices[i].uv = float4(uvs[i], 0.0, 0.0);
        if (p.tessellation != 0xffffffffu) {
            // Match VisibilityTessellationStream's inverse-transpose normal,
            // including reflected instance transforms, before interpolation.
            normals[i] = tessNormal(cross(instance.world1.xyz, instance.world2.xyz) * normals[i].x +
                cross(instance.world2.xyz, instance.world0.xyz) * normals[i].y +
                cross(instance.world0.xyz, instance.world1.xyz) * normals[i].z, float3(0, 1, 0)) *
                (reflected ? -1.0 : 1.0);
        }
    }
    triangle.materialIndex = instance.materialIndex;
    if (p.tessellation != 0xffffffffu) {
        displaceVirtualShadowMapTriangle(p, candidate.w, normals, triangle);
    }
    orientVirtualShadowMapTriangle(reflected, triangle);
    return true;
}

} // namespace Metallic.Shadow
