#language slang 2026
module StreamCLASTessellation;

import ShaderCore;
import GPUDriven;
import Features.GPUDriven.GPUDrivenStreamAsset;
import VisibilityTessellation;
using Metallic;
using Metallic.GPUDriven;

#include "../../Interop/StreamCLASTessellation.h"
#include "../../Interop/ClusterAccelerationStructure.h"

[[vk::push_constant]] ConstantBuffer<StreamCLASTessellationPush> gPush;
groupshared uint gOutsideTemplateBounds;
groupshared float3 gTemplateCorners[3];

float templateOppositeEdgeSquared(float3 a, float3 b)
{
    // Match the native diagnostic's separately rounded FP32 operations. A dot
    // product or contracted multiply-add can select another corner near a tie.
    precise float x = a.x - b.x;
    precise float y = a.y - b.y;
    precise float z = a.z - b.z;
    precise float xx = x * x;
    precise float yy = y * y;
    precise float zz = z * z;
    precise float xy = xx + yy;
    precise float squared = xy + zz;
    return squared;
}

float3 worldPoint(GPUDrivenStreamAssetInstance instance, float3 point)
{
    return instance.world0.xyz * point.x + instance.world1.xyz * point.y +
        instance.world2.xyz * point.z + instance.world3.xyz;
}

float3 worldNormal(GPUDrivenStreamAssetInstance instance, float3 normal)
{
    float3 c0 = cross(instance.world1.xyz, instance.world2.xyz);
    float3 c1 = cross(instance.world2.xyz, instance.world0.xyz);
    float3 c2 = cross(instance.world0.xyz, instance.world1.xyz);
    return tessNormal((c0 * normal.x + c1 * normal.y + c2 * normal.z) *
        (dot(instance.world0.xyz, c0) < 0.0 ? -1.0 : 1.0), normal);
}

float3 localPoint(GPUDrivenStreamAssetInstance instance, float3 point)
{
    // Phase 0 admits only finite, nonsingular instance transforms.
    float3 v = point - instance.world3.xyz;
    float3 c0 = cross(instance.world1.xyz, instance.world2.xyz);
    return float3(dot(v, c0), dot(v, cross(instance.world2.xyz, instance.world0.xyz)),
        dot(v, cross(instance.world0.xyz, instance.world1.xyz))) / dot(instance.world0.xyz, c0);
}

TessView tessView(GPUDrivenStreamAssetParams params)
{
    TessView view;
    view.eye = params.renderEye.xyz;
    view.forward = normalize(params.renderCenter.xyz - params.renderEye.xyz);
    view.orthographic = params.renderUpProjection.w > 0.5;
    view.pixelScale = params.renderViewport.z / (view.orthographic ? params.renderClipOrtho.z :
        2.0 * tan(params.renderViewport.w * 0.5));
    view.nearPlane = params.renderClipOrtho.x;
    return view;
}

void hashWord(inout uint2 hash, uint word)
{
    hash.x = (hash.x ^ word) * 16777619u;
    hash.y = tessDebugHash(hash.y + word + 0x9e3779b9u);
}

// Count and emit use the same safe published RT cut. A failed admission leaves
// the complete base instance intact, including its offscreen contributions.
void walkInstance(StreamCLASTessellationParameters p, uint instanceIndex, bool emit)
{
    let states = (StreamCLASTessellationInstance*)p.instanceStates;
    StreamCLASTessellationInstance state = (StreamCLASTessellationInstance)0;
    if (emit) {
        state = states[instanceIndex];
        if (state.status != 1u) { return; }
    } else { states[instanceIndex] = state; }
    if (p.enabled == 0u) { return; }
    let instance = ((GPUDrivenStreamAssetInstance*)p.instances)[instanceIndex];
    if (instance.visible == 0u) { return; }
    let data = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(p.materialData));
    if (tessMaterial(data, instance.materialIndex).enabled == 0u) { return; }
    float determinant = dot(instance.world0.xyz, cross(instance.world1.xyz, instance.world2.xyz));
    float volumeScale = length(instance.world0.xyz) * length(instance.world1.xyz) * length(instance.world2.xyz);
    if (!isfinite(determinant) || abs(determinant) <= volumeScale * 1e-8 || volumeScale == 0.0) {
        state.status = 3u; states[instanceIndex] = state; return;
    }
    let params = *((GPUDrivenStreamAssetParams*)p.streamParams);
    let primitive = ((GPUDrivenStreamAssetPrimitive*)p.primitives)[instance.primitiveIndex];
    let representationState = (uint*)p.representationState;
    uint selected = representationState[streamRTInstanceBase(params, instanceIndex) + 2u];
    GPUDrivenStreamAssetInstanceBLAS representation = (GPUDrivenStreamAssetInstanceBLAS)0;
    if (selected != 0xffffffffu) { representation = ((GPUDrivenStreamAssetInstanceBLAS*)p.representations)[selected]; }
    bool dynamic = selected != 0xffffffffu && streamBlasDynamicReady(representation);
    uint groupCount = dynamic ? representation.groupCount : primitive.fallbackGroupCount;
    let pages = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(p.pageDescriptor));
    let pageTable = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(p.pageTableDescriptor));
    TessView view = tessView(params);
    uint count = 0u;
    uint2 hash = uint2(2166136261u, 0x1234567u);
    hashWord(hash, p.revision);
    for (uint axis = 0u; axis < 4u; ++axis) {
        hashWord(hash, asuint(instance.world0[axis])); hashWord(hash, asuint(instance.world1[axis]));
        hashWord(hash, asuint(instance.world2[axis])); hashWord(hash, asuint(instance.world3[axis]));
    }
    for (uint groupIndex = 0u; groupIndex < groupCount; ++groupIndex) {
        uint pageIndex, mask, clusterCount;
        if (dynamic) {
            uint4 group = ((uint4*)p.representationGroups)[representation.firstGroup + groupIndex];
            pageIndex = group.x; mask = group.y; clusterCount = group.z;
        } else {
            let group = ((GPUDrivenStreamAssetGroup*)p.groups)[primitive.fallbackGroupOffset + groupIndex];
            pageIndex = group.pageIndex; clusterCount = group.clusterCount;
            mask = clusterCount == 32u ? 0xffffffffu : (1u << clusterCount) - 1u;
        }
        uint2 entry = pageTable[pageIndex];
        if ((entry.x & 7u) != 2u && (entry.x & 7u) != 3u) { state.status = 3u; break; }
        uint page = (entry.x & 0xfffffff8u) / 4u;
        hashWord(hash, pageIndex); hashWord(hash, mask);
        for (uint cluster = 0u; cluster < clusterCount; ++cluster) {
            if ((mask & (1u << cluster)) == 0u) { continue; }
            uint clusterWord = page + pages[page + 9u] / 4u + cluster * 24u;
            uint triangles = pages[clusterWord + 3u];
            for (uint triangle = 0u; triangle < triangles; ++triangle) {
                float3 source[3], normals[3]; float2 uv[3]; float4 tangents[3];
                StreamTriangleDecodeContext context;
                if (!decodeStreamTriangleAtPage(pageIndex, cluster, triangle, pages, pageTable,
                    params.pageBufferBytes / 4u, source[0], source[1], source[2], context) ||
                    (decodeStreamAttributes(pages, context, normals, uv, tangents) & 6u) != 6u) {
                    state.status = 3u; break;
                }
                for (uint i = 0u; i < 3u; ++i) { source[i] = worldPoint(instance, source[i]); }
                uint leafCount = tessBuildLeaves(0u, source, view, p.edgePixels, p.maxFactor, p.maxSplitDepth);
                if (leafCount > p.geometry.maxPatches - min(count, p.geometry.maxPatches)) {
                    state.status = 2u; break;
                }
                for (uint leaf = 0u; leaf < leafCount; ++leaf) {
                    uint base = leaf * 3u;
                    uint3 patch = uint3(g_TessPatches.patches[base], g_TessPatches.patches[base + 1u], g_TessPatches.patches[base + 2u]);
                    uint3 rates = tessPatchRates(patch, source, view, p.edgePixels, p.maxFactor);
                    uint packedRates = rates.x | (rates.y << 8u) | (rates.z << 16u);
                    hashWord(hash, cluster); hashWord(hash, triangle);
                    hashWord(hash, patch.x); hashWord(hash, patch.y); hashWord(hash, patch.z); hashWord(hash, packedRates);
                    if (emit) {
                        uint4 pattern = tessPattern(data, rates);
                        StreamCLASTessellationLeaf record;
                        record.instanceIndex = instanceIndex; record.pageIndex = pageIndex;
                        record.clusterIndex = cluster; record.sourceTriangle = triangle;
                        record.patch0 = patch.x; record.patch1 = patch.y; record.patch2 = patch.z; record.rates = packedRates;
                        record.vertexCount = pattern.y; record.triangleCount = pattern.w;
                        record.reserved0 = record.reserved1 = 0u;
                        ((StreamCLASTessellationLeaf*)p.geometry.leaves)[state.firstPatch + count] = record;
                    }
                    ++count;
                }
            }
            if (state.status > 1u) { break; }
        }
        if (state.status > 1u) { break; }
    }
    if (!emit) {
        state.patchCount = count;
        state.hash0 = hash.x; state.hash1 = hash.y;
        states[instanceIndex] = state;
    }
}

void prefixInstances(StreamCLASTessellationParameters p)
{
    uint count = 0u, builds = 0u, rejected = 0u, invalid = 0u;
    for (uint i = 0u; i < p.instanceCount; ++i) {
        let states = (StreamCLASTessellationInstance*)p.instanceStates;
        StreamCLASTessellationInstance state = states[i];
        if (state.status == 0u && state.patchCount != 0u) {
            if (state.patchCount <= p.geometry.maxPatches - count) {
                state.status = 1u; state.firstPatch = count; state.buildIndex = builds++;
                count += state.patchCount;
            } else { state.status = 2u; }
        }
        rejected += state.status == 2u ? 1u : 0u;
        invalid += state.status == 3u ? 1u : 0u;
        states[i] = state;
    }
    let counts = (uint*)p.counts;
    counts[0] = count; counts[1] = builds; counts[2] = rejected; counts[3] = invalid;
    counts[4] = 0u;
    if (p.initializeBoundStatistics != 0u) { counts[5] = 0u; }
}

void dicePatch(StreamCLASTessellationParameters p, uint patchIndex, uint lane)
{
    if (patchIndex >= ((uint*)p.counts)[0]) { return; }
    // The host publishes three 512-pattern banks only for RightCorner.
    bool selectTemplateCorner = p.templateUnboundedCount == 3u * 512u;
    if (p.templateBoundsRequested != 0u) {
        if (lane == 0u) { gOutsideTemplateBounds = p.templateBoundsAddress == 0u ? 1u : 0u; }
        GroupMemoryBarrierWithGroupSync();
    }
    let leaf = ((StreamCLASTessellationLeaf*)p.geometry.leaves)[patchIndex];
    let instance = ((GPUDrivenStreamAssetInstance*)p.instances)[leaf.instanceIndex];
    let data = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(p.materialData));
    let pages = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(p.pageDescriptor));
    let pageTable = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(p.pageTableDescriptor));
    float3 source[3], normals[3]; float2 uv[3]; float4 tangents[3];
    StreamTriangleDecodeContext context;
    // Phase 0 validated the complete instance; stream pages cannot be changed
    // between count, emission and the last reader of this frame's TLAS.
    decodeStreamCommittedTriangleAtPage(leaf.pageIndex, leaf.clusterIndex, leaf.sourceTriangle,
        pages, pageTable, source[0], source[1], source[2], context);
    decodeStreamAttributes(pages, context, normals, uv, tangents);
    for (uint i = 0u; i < 3u; ++i) {
        source[i] = worldPoint(instance, source[i]); normals[i] = worldNormal(instance, normals[i]);
    }
    uint3 rates = uint3(leaf.rates, leaf.rates >> 8u, leaf.rates >> 16u) & 255u;
    uint4 pattern = tessPattern(data, rates);
    if (lane < leaf.vertexCount) {
        float3 bary = tessPatchDomain(tessBary(data, pattern, lane), rates,
            uint3(leaf.patch0, leaf.patch1, leaf.patch2), source);
        float3 position, normal; float2 texcoord;
        tessSourceDomain(bary, source, normals, uv, position, normal, texcoord);
        position = localPoint(instance, position + normal * tessHeight(tessMaterial(data, instance.materialIndex), texcoord));
        StreamCLASTessellationVertex vertex = (StreamCLASTessellationVertex)0;
        vertex.x = position.x; vertex.y = position.y; vertex.z = position.z;
        vertex.sourceU = bary.y; vertex.sourceV = bary.z;
        ((StreamCLASTessellationVertex*)p.geometry.vertices)[patchIndex * METALLIC_TESS_VERTICES_PER_PATCH + lane] = vertex;
        if (selectTemplateCorner) {
            // The pattern's outer edges start at 0, rates.x, rates.x+rates.y.
            // This also covers the factor-1 special case, whose indices are 0,1,2.
            if (lane == 0u) { gTemplateCorners[0] = position; }
            else if (lane == rates.x) { gTemplateCorners[1] = position; }
            else if (lane == rates.x + rates.y) { gTemplateCorners[2] = position; }
        }
        if (p.templateBoundsAddress != 0u) {
            let bounds = (float*)p.templateBoundsAddress;
            // Validate the actual final FP32 positions at the native API boundary.
            // A failed limit uses identical unbounded topology/positions, never
            // a lower-detail surface or a partially admitted instance.
            if (!all(isfinite(position)) || any(position < float3(bounds[0], bounds[1], bounds[2])) ||
                any(position > float3(bounds[3], bounds[4], bounds[5]))) {
                InterlockedOr(gOutsideTemplateBounds, 1u);
            }
        }
    }
    for (uint triangle = lane; triangle < leaf.triangleCount; triangle += 64u) {
        ((uint3*)p.geometry.triangles)[patchIndex * METALLIC_TESS_TRIANGLES_PER_PATCH + triangle] = tessTriangle(data, pattern, triangle);
    }
    if (p.templateBoundsRequested != 0u || selectTemplateCorner) { GroupMemoryBarrierWithGroupSync(); }
    if (lane == 0u) {
        if (p.buildMode == 1u) {
            uint patternIndex = (rates.x - 1u) + 8u * (rates.y - 1u) + 64u * (rates.z - 1u);
            if (selectTemplateCorner) {
                uint corner = 0u;
                float largest = templateOppositeEdgeSquared(gTemplateCorners[1], gTemplateCorners[2]);
                for (uint i = 1u; i < 3u; ++i) {
                    float squared = templateOppositeEdgeSquared(gTemplateCorners[(i + 1u) % 3u], gTemplateCorners[(i + 2u) % 3u]);
                    if (squared > largest) { largest = squared; corner = i; }
                }
                // Publish the actual choice for diagnostic verification. It is
                // outside the semantic geometry hash; the vertices do not change.
                ((StreamCLASTessellationLeaf*)p.geometry.leaves)[patchIndex].reserved0 = corner;
                patternIndex += corner * 512u;
            }
            if (p.templateBoundsRequested != 0u) {
                if (gOutsideTemplateBounds == 0u) { patternIndex += p.templateUnboundedCount; }
                else {
                    InterlockedAdd(((uint*)p.counts)[4], 1u);
                    InterlockedOr(((uint*)p.counts)[5], 1u);
                }
            }
            ClusterAccelerationStructureTriangleTemplateInstantiationInfo info =
                (ClusterAccelerationStructureTriangleTemplateInstantiationInfo)0;
            info.clusterIdOffset = p.geometry.baseClusterId + patchIndex;
            info.clusterTemplateAddress = ((uint64_t*)p.templateAddresses)[patternIndex];
            info.vertexBufferAddress = p.geometry.vertices + uint64_t(patchIndex) * METALLIC_TESS_VERTICES_PER_PATCH * 32u;
            info.vertexBufferStride = 32u;
            ((ClusterAccelerationStructureTriangleTemplateInstantiationInfo*)p.clasBuildInfos)[patchIndex] = info;
            return;
        }
        ClusterAccelerationStructureTriangleIndirectBuildInfo info = (ClusterAccelerationStructureTriangleIndirectBuildInfo)0;
        info.clusterId = p.geometry.baseClusterId + patchIndex;
        info.countsAndIndexType = METALLIC_CLAS_PACK_COUNTS(leaf.triangleCount, leaf.vertexCount, 0u, METALLIC_CLAS_INDEX_UINT32);
        // TLAS instance coverage flags retain the material's opacity policy.
        info.indexAndVertexStrides = METALLIC_CLAS_PACK_STRIDES(4u, 32u);
        info.indexBufferAddress = p.geometry.triangles + uint64_t(patchIndex) * METALLIC_TESS_TRIANGLES_PER_PATCH * 12u;
        info.vertexBufferAddress = p.geometry.vertices + uint64_t(patchIndex) * METALLIC_TESS_VERTICES_PER_PATCH * 32u;
        ((ClusterAccelerationStructureTriangleIndirectBuildInfo*)p.clasBuildInfos)[patchIndex] = info;
    }
}

void blasRecord(StreamCLASTessellationParameters p, uint instance)
{
    let state = ((StreamCLASTessellationInstance*)p.instanceStates)[instance];
    if (state.status != 1u) { return; }
    uint64_t address = p.clasAddresses + uint64_t(state.firstPatch) * 8u;
    ((uint4*)p.blasBuildInfos)[state.buildIndex] = uint4(state.patchCount, 8u, uint(address), uint(address >> 32u));
}

void publishInstance(StreamCLASTessellationParameters p, uint instance)
{
    let state = ((StreamCLASTessellationInstance*)p.instanceStates)[instance];
    if (state.status == 1u) {
        uint64_t address = ((uint64_t*)p.blasAddresses)[state.buildIndex];
        let instances = (GPUDrivenStreamAssetTLASInstance*)p.tlasInstances;
        instances[instance].accelerationStructureReferenceLow = uint(address);
        instances[instance].accelerationStructureReferenceHigh = uint(address >> 32u);
    }
    let history = (uint4*)p.history;
    uint4 previous = p.initializeHistory != 0u ? uint4(0u) : history[instance];
    uint4 current = state.status == 1u ? uint4(state.hash0, state.hash1, state.patchCount, 1u) : uint4(0u);
    if (any(previous != current)) { InterlockedAdd(((uint*)p.blasHeader)[7], 1u); }
    history[instance] = current;
    if (instance == 0u) {
        let header = (uint*)p.blasHeader;
        uint64_t geometry = p.enabled != 0u ? gPush.parameters : uint64_t(0u);
        header[32] = uint(geometry); header[33] = uint(geometry >> 32u);
        header[34] = ((uint*)p.counts)[0];
        header[35] = ((uint*)p.counts)[2] + ((uint*)p.counts)[3];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tessellationMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    let p = *((StreamCLASTessellationParameters*)gPush.parameters);
    uint index = group.x + group.y * 65535u;
    if (gPush.phase == 0u || gPush.phase == 2u) {
        if (lane == 0u && index < p.instanceCount) { walkInstance(p, index, gPush.phase == 2u); }
    } else if (gPush.phase == 1u) {
        if (lane == 0u) { prefixInstances(p); }
    } else if (gPush.phase == 3u) { dicePatch(p, index, lane); }
    else if (index * 64u + lane < p.instanceCount) {
        if (gPush.phase == 4u) { blasRecord(p, index * 64u + lane); }
        else if (gPush.phase == 5u) { publishInstance(p, index * 64u + lane); }
    }
}
