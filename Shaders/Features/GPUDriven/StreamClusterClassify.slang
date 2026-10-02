#include "../../../Source/Runtime/Render/Core/StreamClassifyParameters.h"
[[vk::push_constant]] ConstantBuffer<StreamClassifyParameters> gStreamClassify;

// Classification consumes only the selected camera, instance transform and
// the cull pass's prevalidated geometry addresses. Keep the raster ABI unchanged.
struct StreamClassifyDesc
{
    float4 eye, center, upProjection, viewport, clipOrtho;
    float4 world0, world1, world2, world3;
    uint positionWord, positionStrideWords, vertexCount, triangleCount;
    uint triangleBaseByte, renderCamera;
};
groupshared StreamClassifyDesc g_StreamClassify;

// Keep the original projectPosition arithmetic, including render-camera jitter.
float4 projectStreamClassifyPosition(float3 worldPosition, StreamClassifyDesc c)
{
    float3 forward = safeNormalize(c.center.xyz - c.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(c.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));

    float3 cameraVector = worldPosition - c.eye.xyz;
    float3 viewPosition = float3(
        dot(cameraVector, right),
        dot(cameraVector, up),
        dot(cameraVector, forward));

    float zNear = max(c.clipOrtho.x, 0.0001);
    float zFar = max(c.clipOrtho.y, zNear + 0.0001);
    float ndcZ = conservativeProjectDepth(viewPosition.z, zNear, zFar,
        c.upProjection.w > 0.5, c.clipOrtho.w > 0.5);
    float aspect = max(c.viewport.x, 0.001);

    if (c.upProjection.w > 0.5) {
        float halfHeight = max(c.clipOrtho.z * 0.5, 0.0001);
        float4 clip = float4(
            viewPosition.x / (halfHeight * aspect),
            viewPosition.y / halfHeight,
            ndcZ,
            1.0);
        if (c.renderCamera != 0u) {
            clip.xy -= 2.0 * float2(c.eye.w, -c.center.w) / c.viewport.yz * clip.w;
        }
        return clip;
    }

    float fovRadians = clamp(c.viewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = viewPosition.z;
    float4 clip = float4(
        viewPosition.x * yScale / aspect,
        viewPosition.y * yScale,
        ndcZ * w,
        w);
    if (c.renderCamera != 0u) {
        clip.xy -= 2.0 * float2(c.eye.w, -c.center.w) / c.viewport.yz * clip.w;
    }
    return clip;
}

float4 streamClassifyClip(StreamClassifyDesc c, uint vertex, StreamClassifyParameters push)
{
    uint* pages = push.pages.data;
    uint word = c.positionWord + vertex * c.positionStrideWords;
    float3 local = float3(asfloat(pages[word]), asfloat(pages[word + 1u]), asfloat(pages[word + 2u]));
    float3 world = c.world0.xyz * local.x + c.world1.xyz * local.y +
        c.world2.xyz * local.z + c.world3.xyz;
    float4 clip = projectStreamClassifyPosition(world, c);
    clip.y = -clip.y;
    return clip;
}

uint streamClassifyByte(uint* pages, uint address)
{
    return (pages[address / 4u] >> ((address & 3u) * 8u)) & 255u;
}

uint3 streamClassifyTriangle(StreamClassifyDesc c, uint triangle, StreamClassifyParameters push)
{
    uint* pages = push.pages.data;
    uint address = c.triangleBaseByte + triangle * 3u;
    uint3 indices = uint3(streamClassifyByte(pages, address),
        streamClassifyByte(pages, address + 1u), streamClassifyByte(pages, address + 2u));
    // Retain the existing malformed-index fallback in the stream HW decoder.
    indices = uint3(indices.x < c.vertexCount ? indices.x : 0u,
        indices.y < c.vertexCount ? indices.y : 0u, indices.z < c.vertexCount ? indices.z : 0u);
    // A reflected instance reverses projected winding. Keep authored front
    // faces front-facing in both hardware and software rasterization; the
    // visibility ID still names the original payload triangle for shading.
    bool reflected = dot(cross(c.world0.xyz, c.world1.xyz), c.world2.xyz) < 0.0;
    return reflected ? indices.xzy : indices;
}

void classifyStreamCluster(uint3 group, uint lane, StreamClassifyParameters push, bool checkHardware)
{
    RWStructuredBuffer<uint> bins = push.bins;
    uint work = hybridFlatGroup(group);
    if (work >= bins[0]) { return; }
    uint item = hybridStreamClassifyBase(work);
    uint candidate = bins[item];
    uint record = bins[hybridCandidateBase(bins[5]) + candidate * 2u];
    if (!push.settings.contains(0u) || push.settings.data == nullptr || bins[13] == 0u ||
        push.groups.data == nullptr || !push.groups.contains(record / bins[13])) { return; }
    // P1 cull already diverts coverage and tessellation to hardware.
    if (checkHardware) {
        uint instanceIndex = push.groups.data[record / bins[13]].gpuSceneInstanceIndex;
        bool needsCoverage = push.instances.data != nullptr && push.instances.contains(instanceIndex) &&
            (push.instances.data[instanceIndex].identity.w & 12u) != 0u;
        if (push.tessellationEnabled != 0u || needsCoverage) {
            if (lane == 0u) { hybridClassifyCluster(bins, 0u, record, candidate); }
            return;
        }
    }
    // All fields have exactly one writer. The first subgroup spreads the
    // independent loads; the existing group barrier publishes them to all 128 lanes.
    if (lane < WaveGetLaneCount()) {
        GPUDrivenStreamAssetParams* params = push.settings.data;
        GPUDrivenStreamAssetActiveGroup* groups = push.groups.data;
        bool renderCamera = params[0].renderViewport.y > 0.0;
        uint groupIndex = record / bins[13];
        // Also covers subgroups smaller than the number of descriptor fields.
        for (uint field = lane; field < 14u; field += WaveGetLaneCount()) {
            switch (field) {
            case 0: g_StreamClassify.eye = renderCamera ? params[0].renderEye : params[0].eye; break;
            case 1: g_StreamClassify.center = renderCamera ? params[0].renderCenter : params[0].center; break;
            case 2: g_StreamClassify.upProjection = renderCamera ? params[0].renderUpProjection : params[0].upProjection; break;
            case 3: g_StreamClassify.viewport = renderCamera ? params[0].renderViewport : params[0].viewport; break;
            case 4: g_StreamClassify.clipOrtho = renderCamera ? params[0].renderClipOrtho : params[0].clipOrtho; break;
            case 5: g_StreamClassify.world0 = groups[groupIndex].world0; break;
            case 6: g_StreamClassify.world1 = groups[groupIndex].world1; break;
            case 7: g_StreamClassify.world2 = groups[groupIndex].world2; break;
            case 8: g_StreamClassify.world3 = groups[groupIndex].world3; break;
            case 9: g_StreamClassify.positionWord = bins[item + 1u]; break;
            case 10:
                uint counts = bins[item + 2u];
                g_StreamClassify.vertexCount = counts & 0xffffu;
                g_StreamClassify.triangleCount = (counts >> 16u) & 0x7fffu;
                g_StreamClassify.positionStrideWords = (counts & 0x80000000u) != 0u ? 3u : 4u;
                break;
            case 11: g_StreamClassify.triangleBaseByte = bins[item + 3u]; break;
            case 12: g_StreamBinHardware = 0u; break;
            case 13: g_StreamClassify.renderCamera = renderCamera ? 1u : 0u; break;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Uniform checks precede the barriers and every BDA load.
    uint64_t positionEnd = uint64_t(g_StreamClassify.positionWord) +
        uint64_t(g_StreamClassify.vertexCount) * g_StreamClassify.positionStrideWords;
    uint64_t triangleEnd = uint64_t(g_StreamClassify.triangleBaseByte) + uint64_t(g_StreamClassify.triangleCount) * 3u;
    if (push.pages.data == nullptr || g_StreamClassify.vertexCount == 0u ||
        g_StreamClassify.vertexCount > 128u || g_StreamClassify.triangleCount > 128u ||
        positionEnd > uint64_t(push.pages.length()) || triangleEnd > uint64_t(push.pages.length()) * 4u) { return; }
    if (lane < g_StreamClassify.vertexCount) { g_StreamRasterClip[lane] = streamClassifyClip(g_StreamClassify, lane, push); }
    GroupMemoryBarrierWithGroupSync();
    bool hardware = false;
    if (lane < g_StreamClassify.triangleCount) {
        uint3 tri = streamClassifyTriangle(g_StreamClassify, lane, push);
        hardware = !hybridTriangleFits(g_StreamRasterClip[tri.x], g_StreamRasterClip[tri.y], g_StreamRasterClip[tri.z],
            float2(bins[6], bins[7]), asfloat(bins[8]));
    }
    bool waveHardware = WaveActiveAnyTrue(hardware);
    if (WaveIsFirstLane() && waveHardware) { InterlockedOr(g_StreamBinHardware, 1u); }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) { hybridClassifyCluster(bins, g_StreamBinHardware != 0u ? 0u : kHybridSoftwareBin, record, candidate); }
}


[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterBinMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    classifyStreamCluster(group, lane, gStreamClassify, false);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterBinP0Main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    classifyStreamCluster(group, lane, gStreamClassify, true);
}
