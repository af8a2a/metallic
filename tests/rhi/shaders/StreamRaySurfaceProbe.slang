#include "UnusedPathTraceScene.hlsli"
#include "../StreamProbeParameters.h"
#include "../../../Shaders/Features/PathTracing/ScenePathTrace.slang"
#include "../../../Shaders/Features/PathTracing/StreamSceneRayQuery.slang"

[[vk::push_constant]] ConstantBuffer<StreamProbeParameters> gStreamProbe;
// Deliberately no path-tracing root or descriptor resources: the production
// surface/alpha helpers must consume only the explicitly supplied provider.
struct ProbeMaterials : IStreamRayMaterials
{
    uint test;
    bool load(uint index, out PathTraceMaterial material)
    {
        material = (PathTraceMaterial)0;
        if (index != 17u || test == 3u || test == 12u) { return false; }
        material.baseColor.a = 0.5;
        material.params.z = 0.25;
        material.textureParams.w = (test == 11u || test >= 16u) ? 0.0 : (test == 9u || test == 10u ? 2.0 : 1.0);
        return true;
    }
    float alpha(PathTraceTextureInfo texture, float2 uv)
    {
        return test == 8u ? 0.49 : test == 9u ? 0.0 : 0.5;
    }
};

[shader("compute")]
[numthreads(32, 1, 1)]
void streamRaySurfaceMain(uint3 id : SV_DispatchThreadID)
{
    StreamProbeParameters p = gStreamProbe;
    if (id.x >= 20u || !p.output.contains(24u + id.x)) { return; }
    ProbeMaterials materials;
    materials.test = id.x;
    Ray ray;
    ray.origin = float3(0.25, 0.25, id.x == 1u ? -1.0 : 1.0);
    ray.direction = float3(0, 0, id.x == 1u ? 1.0 : -1.0);
    uint instance = id.x == 4u || id.x == 13u ? 1u : 0u;
    uint triangle = id.x == 5u || id.x == 14u || id.x == 16u ? 1u : 0u;
    if (id.x == 6u) { p.stream.pages.count = 27; }
    if (id.x == 15u || id.x == 17u) { p.stream.pages.data = nullptr; }
    if (id.x == 18u) { p.stream.header.count = 2u; }
    if (id.x == 19u) { p.stream.pageTable.count = 0u; }
    float flip = id.x == 2u ? -1.0 : 1.0;
    bool valid;
    if (id.x >= 7u) {
        valid = streamRayAlphaAccepts(instance, 0u, triangle, float2(0.25),
            p.stream, materials);
    } else {
        HitInfo hit;
        valid = streamRaySurface(instance, 0u, triangle, float2(0.25), ray, 1.0,
            p.stream, materials, flip, hit);
        if (valid) {
            // Reversing the incident ray must only change frontFace, never TBN.
            valid = all(abs(hit.normal - float3(0, 0, 1)) < 1e-5) &&
                all(abs(hit.geometryNormal - float3(0, 0, 1)) < 1e-5) &&
                hit.frontFace == (id.x != 1u) && hit.materialIndex == 17u &&
                abs(dot(hit.tangent.xyz, hit.normal)) < 1e-5 &&
                abs(length(hit.tangent.xyz) - 1.0) < 1e-5 &&
                all(abs(hit.bitangent - cross(hit.normal, hit.tangent.xyz) * hit.tangent.w) < 1e-5) &&
                all(abs(hit.position - float3(0.25, 0.25, 0)) < 1e-5);
            HitInfo reference;
            if (valid && id.x == 2u) {
                valid = streamRaySurface(instance, 0u, triangle, float2(0.25), ray, 1.0,
                    p.stream, materials, 1.0, reference) &&
                    all(abs(hit.tangent.xyz - reference.tangent.xyz) < 1e-5) &&
                    hit.tangent.w == -reference.tangent.w &&
                    all(abs(hit.bitangent + reference.bitangent) < 1e-5);
            }
        }
    }
    p.output.data[24u + id.x] = valid ? 1u : 0u;
}

// UV-sensitive coverage catches accidentally sampling zero UV or the wrong vertex.
struct UVProbeMaterials : IStreamRayMaterials
{
    bool load(uint index, out PathTraceMaterial material)
    {
        material = (PathTraceMaterial)0;
        material.baseColor.a = 0.5;
        material.params.z = 0.25;
        material.textureParams.w = 1.0;
        return index == 17u;
    }
    float alpha(PathTraceTextureInfo texture, float2 uv)
    {
        return uv.x + uv.y;
    }
};

[shader("compute")]
[numthreads(8, 1, 1)]
void streamRayUVMain(uint3 id : SV_DispatchThreadID)
{
    StreamProbeParameters p = gStreamProbe;
    if (id.x >= 8u || !p.output.contains(44u + id.x)) { return; }
    p.stream.pages.data = &p.stream.pages.data[id.x * 76u];
    p.stream.pages.count = 76u;
    float2 bary = id.x == 6u ? float2(0.1) : id.x == 7u ? float2(0.75, 0) : float2(0.25);
    UVProbeMaterials materials;
    bool accepted = streamRayAlphaAccepts(0u, 0u, 0u, bary, p.stream, materials);
    float3 a, b, c, normals[3]; float2 uvs[3]; float4 tangents[3];
    uint attributes;
    bool valid = decodeStreamRaySurface(p.stream, 0u, 0u, a, b, c, normals, uvs, tangents, attributes);
    bool hasUV = id.x == 0u || id.x >= 6u;
    float2 expectedUV = hasUV ? bary : float2(0);
    valid = valid && attributes == (hasUV ? 4u : 0u) &&
        all(abs(uvs[0] * (1.0 - bary.x - bary.y) + uvs[1] * bary.x + uvs[2] * bary.y - expectedUV) < 1e-5);
    // Both surface and alpha keep the existing zero-UV fallback for invalid streams.
    p.output.data[44u + id.x] = valid && accepted == (id.x == 0u || id.x == 7u) ? 1u : 0u;
}
