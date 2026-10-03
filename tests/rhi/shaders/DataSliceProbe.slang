import Core;
using Metallic;
#include "../TestResourceParameters.h"
import ParameterRoot;

struct DataProbeParams
{
    RWBufferSpan<uint> source;
    RWBufferSpan<uint> output;
    RWBufferSpan<uint> arguments;
    uint add;
};

[shader("compute")]
[numthreads(8, 1, 1)]
void dataProduceMain(uint3 id : SV_DispatchThreadID)
{
    DataProbeParams params = getParameters<DataProbeParams>();
    if (params.source.contains(id.x) && params.output.contains(id.x)) {
        params.output.store(id.x, params.source[id.x] + params.add);
    }
    if (id.x == 0 && params.arguments.length() >= 3) {
        params.arguments.store(0, 1);
        params.arguments.store(1, 1);
        params.arguments.store(2, 1);
    }
}

[shader("compute")]
[numthreads(8, 1, 1)]
void dataIndirectMain(uint3 id : SV_DispatchThreadID)
{
    DataProbeParams params = getParameters<DataProbeParams>();
    if (params.output.contains(id.x)) { params.output.store(id.x, params.output.load(id.x) * (2)); }
}

[shader("compute")]
[numthreads(8, 1, 1)]
void dataAdapterMain(uint3 id : SV_DispatchThreadID)
{
    RWBufferSpan<uint> output = typedBufferSpan<uint>(getResourceParameters<DataSliceProbeResources>().output);
    // An incomplete element must not expose a partial typed span.
    BufferSpan<uint> incomplete = getResourceParameters<DataSliceProbeResources>().output;
    incomplete.count = 3;
    if (typedBufferSpan<uint2>(incomplete).length() != 0u) { output.store(id.x, 0xbadbad); return; }
    if (output.contains(id.x)) { output.store(id.x, output.load(id.x) + (3)); }
}
