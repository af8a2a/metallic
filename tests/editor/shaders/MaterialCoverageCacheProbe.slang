module MaterialCoverageCacheProbe;
import Material;
import MaterialProgram;
using Metallic.Material;

[[vk::binding(0, 0)]] ByteAddressBuffer inputs;
[[vk::binding(1, 0)]] RWStructuredBuffer<float2> output;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    CoverageProgram program;
    program.inputs = inputs;
    CoverageContext context;
    context.uv = float2(float(id.x) / 1024.0, 0.0);
    context.baseAlpha = 1.0;
    context.cutoff = 0.5;
    context.alphaMode = 1.0;
    let result = program.evaluateCoverage(context, coverageInstanceRef(0));
    output[id.x] = float2(result.opacity, result.accepted ? 1.0 : 0.0);
}
