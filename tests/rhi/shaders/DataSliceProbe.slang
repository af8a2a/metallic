import Core;
using Metallic;
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
    RWBufferSpan<uint> output = getData<uint>(0);
    if (getData<uint2>(0).length() != 0u) { output.store(id.x, 0xbadbad); return; }
    if (output.contains(id.x)) { output.store(id.x, output.load(id.x) + (3)); }
}
