import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#include "../HybridProbeParameters.h"
[shader("compute")]
[numthreads(1, 1, 1)]
void classifyMain(uint3 group : SV_GroupID, uniform HybridClusterProbeParameters push)
{
    uint item = hybridFlatGroup(group);
    if (item >= push.count || !push.input.contains(item)) { return; }
    uint2* input = push.input.data;
    DescriptorHandle<RWStructuredBuffer<uint>> bins =
        push.bins;
    uint2 candidate = input[item];
    hybridClassifyCluster(bins, candidate.y, candidate.x, item);
}

// Poison reusable scratch and seed retry masks independently of preparation.
[shader("compute")]
[numthreads(128, 1, 1)]
void seedStreamCandidatesMain(uint3 id : SV_DispatchThreadID, uniform HybridClusterProbeParameters push)
{
    uint item = id.x;
    if (item >= push.count || !push.input.contains(item)) { return; }
    uint2* input = push.input.data;
    DescriptorHandle<RWStructuredBuffer<uint>> bins =
        push.bins;
    bins[kHybridBinHeaderWords + item] = 0xa5a5a5a5u;
    bins[hybridCandidateBase(push.count) + item * 2u] = 0xa5a5a5a5u;
    bins[hybridCandidateBase(push.count) + item * 2u + 1u] = 0xa5a5a5a5u;
    bins[hybridRankBase(push.count) + item] = 0xa5a5a5a5u;
    bins[hybridRetryBase(push.count) + item] = input[item].x;
    if (item < ((push.count + 127u) / 128u) * kHybridBinCount) {
        bins[hybridBlockBase(push.count) + item] = 0xa5a5a5a5u;
    }
}
