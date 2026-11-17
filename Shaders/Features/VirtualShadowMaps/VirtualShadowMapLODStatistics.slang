#language slang 2026
module VirtualShadowMapLODStatistics;

import ShaderCore;
using Metallic;

struct LODStatisticsSource
{
    uint statistics, viewCount;
};

struct LODStatisticsPush
{
    uint sources, output, sourceCount;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void vsmAggregateLODStatisticsMain(uint word : SV_GroupIndex, uniform LODStatisticsPush push)
{
    StructuredBuffer<LODStatisticsSource> sources = resolveUniform(ResourceHandle<StructuredBuffer<LODStatisticsSource>>(push.sources));
    RWStructuredBuffer<uint> output = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.output));
    uint value = 0u;
    if (word == 0u) { value = 3u; }
    else if (word == 1u) { value = 1u; }
    else if (word != 14u && word != 15u) {
        for (uint index = 0u; index < push.sourceCount; ++index) {
            LODStatisticsSource source = sources[index];
            if (word == 2u) { value += source.viewCount; }
            else {
                StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(source.statistics));
                for (uint view = 0u; view < source.viewCount; ++view) { value += statistics[view * 32u + word]; }
            }
        }
    }
    output[word] = value;
}
