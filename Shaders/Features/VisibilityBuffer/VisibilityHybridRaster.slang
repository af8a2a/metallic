import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#include "../../../Source/Runtime/Render/Core/HybridRasterParameters.h"
[[vk::push_constant]] ConstantBuffer<HybridRasterParameters> gHybridRaster;

[shader("compute")]
[numthreads(64, 1, 1)]
void hybridResetMain(uint3 id : SV_DispatchThreadID)
{
    HybridRasterParameters push = gHybridRaster;
    DescriptorHandle<RWStructuredBuffer<uint64_t>> pixels = push.pixels;
    uint index = id.y * push.width + id.x;
    if (id.x < push.width && id.y < push.height) { pixels[index] = uint64_t(0); }
    if (all(id == 0u)) {
        DescriptorHandle<RWStructuredBuffer<uint4>> queue = push.queue;
        queue[0] = uint4(0u, push.capacity, push.width, push.height);
        queue[1] = uint4(push.width, push.height, asuint(push.maxPixels), push.capacity);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void hybridArgumentsMain()
{
    HybridRasterParameters push = gHybridRaster;
    DescriptorHandle<RWStructuredBuffer<uint4>> queue = push.queue;
    if (push.arguments.data == nullptr || !push.arguments.contains(2u)) { return; }
    uint* args = push.arguments.data;
    args[0] = (min(queue[0].x, push.capacity) + 63u) / 64u;
    args[1] = 1u;
    args[2] = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void hybridRasterMain(uint3 id : SV_DispatchThreadID)
{
    HybridRasterParameters push = gHybridRaster;
    RWStructuredBuffer<uint4> queue = push.queue;
    if (id.x >= min(queue[0].x, push.capacity)) { return; }
    uint offset = 2u + id.x * 4u;
    float4 c0 = asfloat(queue[offset]);
    float4 c1 = asfloat(queue[offset + 1u]);
    float4 c2 = asfloat(queue[offset + 2u]);
    uint4 payload = queue[offset + 3u];
    hybridRasterTriangle(c0, c1, c2, payload.x, payload.y != 0u,
        push.width, push.height, push.reversedZ, push.subpixelBits, push.pixels);
}

#include "../../../Source/Runtime/Render/Core/HybridResolveParameters.h"
[[vk::push_constant]] ConstantBuffer<HybridResolveParameters> gHybridResolve;

struct HybridVertexOutput { float4 position : SV_Position; };
[shader("vertex")]
HybridVertexOutput hybridResolveVertexMain(uint vertex : SV_VertexID)
{
    HybridVertexOutput output;
    output.position = float4(float2((vertex << 1u) & 2u, vertex & 2u) * 2.0 - 1.0, 0.0, 1.0);
    return output;
}
struct HybridFragmentOutput { uint visibility : SV_Target0; float depth : SV_Depth; };
[shader("fragment")]
HybridFragmentOutput hybridResolveFragmentMain(HybridVertexOutput input)
{
    HybridResolveParameters push = gHybridResolve;
    uint index = uint(input.position.y) * push.width + uint(input.position.x);
    // Discard can demote a fragment to a helper invocation; explicitly leave
    // the invalid path before dereferencing a physical buffer address.
    if (push.pixels.data == nullptr || !push.pixels.contains(index)) {
        discard;
        return (HybridFragmentOutput)0;
    }
    uint64_t packed = push.pixels.data[index];
    if (uint(packed) == 0u) { discard; }
    uint key = uint(packed >> 32u);
    HybridFragmentOutput output;
    output.visibility = uint(packed);
    output.depth = asfloat(push.reversedZ != 0u ? key : ~key);
    return output;
}

#include "../../../Source/Runtime/Render/Core/HybridBinParameters.h"
[[vk::push_constant]] ConstantBuffer<HybridBinParameters> gHybridBins;

[shader("compute")]
[numthreads(1, 1, 1)]
void hybridClusterResetMain()
{
    HybridBinParameters push = gHybridBins;
    RWStructuredBuffer<uint> bins = push.bins;
    for (uint i = 0; i < kHybridBinHeaderWords; ++i) { bins[i] = 0u; }
    bins[5] = push.clusterCapacity;
    bins[6] = push.width;
    bins[7] = push.height;
    bins[8] = asuint(push.maxPixels);
    bins[9] = push.reversedZ;
    bins[10] = push.subpixelBits;
    bins[11] = push.producerPixelBuffer;
    bins[12] = push.inputClusterCount;
}

groupshared uint g_BinTags[128];
groupshared uint g_BinCounts[5];
groupshared uint g_BinPrefix[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void hybridClusterHistogramMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    HybridBinParameters push = gHybridBins;
    RWStructuredBuffer<uint> bins = push.bins;
    uint block = hybridFlatGroup(group);
    uint item = block * kHybridBinBlockSize + lane;
    uint tag = 0xffffffffu;
    if (item < bins[12]) { tag = bins[hybridCandidateBase(push.clusterCapacity) + item * 2u + 1u]; }
    g_BinTags[lane] = tag;
    if (lane < kHybridBinCount) { g_BinCounts[lane] = 0u; }
    GroupMemoryBarrierWithGroupSync();
    if (tag < kHybridBinCount) {
        uint rank = 0u;
        for (uint i = 0; i < lane; ++i) { rank += g_BinTags[i] == tag ? 1u : 0u; }
        bins[hybridRankBase(push.clusterCapacity) + item] = rank;
        InterlockedAdd(g_BinCounts[tag], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint blockCapacity = (push.clusterCapacity + 127u) / 128u;
    if (lane < kHybridBinCount && block < (bins[12] + 127u) / 128u) {
        bins[hybridBlockBase(push.clusterCapacity) + lane * blockCapacity + block] = g_BinCounts[lane];
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void hybridClusterArgumentsMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    HybridBinParameters push = gHybridBins;
    DescriptorHandle<RWStructuredBuffer<uint>> bins = push.bins;
    if (push.arguments.data == nullptr || !push.arguments.contains(14u)) { return; }
    uint* args = push.arguments.data;
    uint bin = group.x;
    uint blocks = (bins[12] + 127u) / 128u;
    uint blocksPerLane = (blocks + 127u) / 128u;
    uint first = min(lane * blocksPerLane, blocks);
    uint end = min(first + blocksPerLane, blocks);
    uint base = hybridBlockBase(push.clusterCapacity) + bin * ((push.clusterCapacity + 127u) / 128u);
    uint total = 0u;
    for (uint block = first; block < end; ++block) { total += bins[base + block]; }
    g_BinPrefix[lane] = total;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 128u; stride *= 2u) {
        uint previous = lane >= stride ? g_BinPrefix[max(lane, stride) - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        g_BinPrefix[lane] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    uint offset = lane != 0u ? g_BinPrefix[max(lane, 1u) - 1u] : 0u;
    for (uint block = first; block < end; ++block) {
        uint count = bins[base + block];
        bins[base + block] = offset;
        offset += count;
    }
    if (lane == 127u) {
        uint count = g_BinPrefix[127];
        if ((push.streamMode & 1u) != 0u && bins[14] != 0u) {
            // Group masks remain intact: scatter/cull/classify have zero work.
            count = bin == 0u ? bins[15] * bins[13] : 0u;
        }
        bins[bin] = count;
        uint groups = bin == kHybridSoftwareBin ? count :
            (push.streamMode & 2u) != 0u ? count * 8u :
            (push.streamMode & 1u) != 0u ? count : (count + 31u) / 32u;
        args[bin * 3u] = min(groups, kHybridDispatchWidth);
        args[bin * 3u + 1u] = max(1u, (groups + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
        args[bin * 3u + 2u] = 1u;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void hybridClusterScatterMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    HybridBinParameters push = gHybridBins;
    uint block = hybridFlatGroup(group);
    uint item = block * kHybridBinBlockSize + lane;
    DescriptorHandle<RWStructuredBuffer<uint>> bins = push.bins;
    if (item >= bins[12]) { return; }
    uint candidate = hybridCandidateBase(push.clusterCapacity) + item * 2u;
    uint bin = bins[candidate + 1u];
    if (bin >= kHybridBinCount) { return; }
    uint blockBase = hybridBlockBase(push.clusterCapacity) + bin * ((push.clusterCapacity + 127u) / 128u);
    uint index = bins[blockBase + block] + bins[hybridRankBase(push.clusterCapacity) + item];
    if (index < push.clusterCapacity) {
        bins[kHybridBinHeaderWords + bin * push.clusterCapacity + index] = bins[candidate];
    } else {
        InterlockedAdd(bins[14], 1u);
    }
}
