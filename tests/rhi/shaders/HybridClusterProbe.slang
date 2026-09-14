import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

struct ProbePush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint inputBuffer;
    uint clusterBuffer;
    uint count;
};
[shader("compute")]
[numthreads(1, 1, 1)]
void classifyMain(uint3 group : SV_GroupID, uniform ProbePush push)
{
    uint item = hybridFlatGroup(group);
    if (item >= push.count) { return; }
    DescriptorHandle<StructuredBuffer<uint2>> input =
        DescriptorHandle<StructuredBuffer<uint2>>(uint2(push.bufferShaderIndexBase + push.inputBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> bins =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.clusterBuffer, 0));
    uint2 candidate = input[item];
    hybridClassifyCluster(bins, candidate.y, candidate.x, item);
}

// Poison reusable scratch and seed retry masks independently of preparation.
[shader("compute")]
[numthreads(128, 1, 1)]
void seedStreamCandidatesMain(uint3 id : SV_DispatchThreadID, uniform ProbePush push)
{
    uint item = id.x;
    if (item >= push.count) { return; }
    DescriptorHandle<StructuredBuffer<uint2>> input =
        DescriptorHandle<StructuredBuffer<uint2>>(uint2(push.bufferShaderIndexBase + push.inputBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> bins =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.clusterBuffer, 0));
    bins[kHybridBinHeaderWords + item] = 0xa5a5a5a5u;
    bins[hybridCandidateBase(push.count) + item * 2u] = 0xa5a5a5a5u;
    bins[hybridCandidateBase(push.count) + item * 2u + 1u] = 0xa5a5a5a5u;
    bins[hybridRankBase(push.count) + item] = 0xa5a5a5a5u;
    bins[hybridRetryBase(push.count) + item] = input[item].x;
    if (item < ((push.count + 127u) / 128u) * kHybridBinCount) {
        bins[hybridBlockBase(push.count) + item] = 0xa5a5a5a5u;
    }
}
