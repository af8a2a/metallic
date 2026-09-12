#include "../../../Shaders/Libraries/GPUDriven/HybridClusterBins.slang"
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
