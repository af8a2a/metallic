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
    uint residentClusterBounds, reserved0, reserved1, reserved2;
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

struct VirtualShadowMapCasterClusterBounds
{
    float4 minimum, maximum;
};

// firstTriangle is aligned to the 64-slot fine-work group. A streamed
// candidate has 128 slots, so no group straddles two terminal clusters.
public bool loadVirtualShadowMapClusterBounds(uint geometryResourceIndex, uint firstTriangle,
    out float3 minimum, out float3 maximum)
{
    minimum = maximum = float3(0.0);
    StructuredBuffer<VirtualShadowMapGeometryParameters> parameters =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapGeometryParameters>>(geometryResourceIndex));
    VirtualShadowMapGeometryParameters p = parameters[0];
    if (p.streamed == 0u) {
        StructuredBuffer<VirtualShadowMapCasterClusterBounds> bounds =
            resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapCasterClusterBounds>>(p.residentClusterBounds));
        VirtualShadowMapCasterClusterBounds cluster = bounds[firstTriangle / 64u];
        minimum = cluster.minimum.xyz;
        maximum = cluster.maximum.xyz;
    } else {
        StructuredBuffer<uint4> candidates = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(p.candidates));
        uint4 candidate = candidates[firstTriangle / 128u];
        StructuredBuffer<VirtualShadowMapStreamInstance> instances =
            resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapStreamInstance>>(p.instances));
        VirtualShadowMapStreamInstance instance = instances[candidate.x];
        if (instance.visible == 0u) { return false; }
        StructuredBuffer<uint> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(p.pages));
        StructuredBuffer<uint2> pageTable = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(p.pageTable));
        uint pageCount, stride;
        pageTable.GetDimensions(pageCount, stride);
        if (candidate.y >= pageCount) { return false; }
        uint entry = pageTable[candidate.y].x;
        if ((entry & 7u) != 2u && (entry & 7u) != 3u) { return false; }
        uint page = (entry & 0xfffffff8u) / 4u;
        if (page > p.pageWordCount || p.pageWordCount - page < 28u) { return false; }
        uint bytes = pages[page + 12u], clusters = pages[page + 9u];
        if (bytes < 112u || bytes / 4u > p.pageWordCount - page || (clusters & 3u) != 0u ||
            clusters > bytes || candidate.z >= pages[page + 2u] || candidate.z >= (bytes - clusters) / 96u) { return false; }
        uint cluster = page + clusters / 4u + candidate.z * 24u;
        if (firstTriangle % 128u >= pages[cluster + 3u]) { return false; }
        // The stream cooker builds cluster bounds from the same reconstructed
        // positions it stores: Reference quantizes before rebuilding meshlets.
        // Device compaction only removes position padding, preserving XYZ bits.
        float3 localCenter = asfloat(uint3(pages[cluster + 12u], pages[cluster + 13u], pages[cluster + 14u]));
        float radius = asfloat(pages[cluster + 15u]);
        float3 center = instance.world0.xyz * localCenter.x + instance.world1.xyz * localCenter.y +
            instance.world2.xyz * localCenter.z + instance.world3.xyz;
        // Transform the sphere's enclosing cube. Absolute columns retain a
        // conservative AABB for nonuniform scale, reflection and shear.
        float3 extent = radius * (abs(instance.world0.xyz) + abs(instance.world1.xyz) + abs(instance.world2.xyz));
        // Center and vertex transforms can round differently before large
        // local coordinates cancel against translation. Bound their arithmetic
        // using absolute terms, independently of the small translated result.
        float3 transformMagnitude = abs(instance.world0.xyz * localCenter.x) +
            abs(instance.world1.xyz * localCenter.y) + abs(instance.world2.xyz * localCenter.z) + abs(instance.world3.xyz);
        extent += (transformMagnitude + extent) * 1e-6;
        if (p.tessellation != 0xffffffffu) {
            StructuredBuffer<uint> data = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(p.tessellation));
            TessMaterial material = tessMaterial(data, instance.materialIndex);
            if (material.enabled != 0u) { extent += abs(material.magnitude) * max(abs(material.centre), abs(1.0 - material.centre)); }
        }
        minimum = center - extent;
        maximum = center + extent;
    }
    // Cover float rounding of final bound endpoints and displaced interpolation;
    // the projected page guard below is separate from this world-space bound.
    float3 rounding = max(max(abs(minimum), abs(maximum)), float3(1.0)) * 1e-6;
    minimum -= rounding;
    maximum += rounding;
    return true;
}

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
