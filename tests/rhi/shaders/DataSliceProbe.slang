import Core;
using Metallic;
import ParameterRoot;

struct DataProbeParams
{
    DataSpan<uint> source;
    DataSpan<uint> output;
    DataSpan<uint> arguments;
    uint add;
};

[shader("compute")]
[numthreads(8, 1, 1)]
void dataProduceMain(uint3 id : SV_DispatchThreadID)
{
    DataProbeParams params = getParameters<DataProbeParams>();
    if (params.source.contains(id.x) && params.output.contains(id.x)) {
        params.output.data[id.x] = params.source.data[id.x] + params.add;
    }
    if (id.x == 0 && params.arguments.length() >= 3) {
        params.arguments.data[0] = 1;
        params.arguments.data[1] = 1;
        params.arguments.data[2] = 1;
    }
}

[shader("compute")]
[numthreads(8, 1, 1)]
void dataIndirectMain(uint3 id : SV_DispatchThreadID)
{
    DataProbeParams params = getParameters<DataProbeParams>();
    if (params.output.contains(id.x)) { params.output.data[id.x] *= 2; }
}

[shader("compute")]
[numthreads(8, 1, 1)]
void dataAdapterMain(uint3 id : SV_DispatchThreadID)
{
    DataSpan<uint> output = getData<uint>(0);
    if (output.contains(id.x)) { output.data[id.x] += 3; }
}
