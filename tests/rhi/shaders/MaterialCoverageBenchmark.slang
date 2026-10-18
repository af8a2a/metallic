// Same dispatch and outputs for both backends. Legacy VM exists only in tests.
module MaterialCoverageBenchmark;
import Core;
import Material;
import MaterialProgram;
#if METALLIC_TEST_LEGACY_COVERAGE
import LegacyCoverageOracle;
using Metallic.CoverageTest;
#endif
using Metallic;
using Metallic.Material;

struct CoverageBenchmarkPush
{
    uint size;
    uint materialCount;
    uint repeats;
    uint extremeInputs;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void materialCoverageBenchmarkMain(uint3 id : SV_DispatchThreadID)
{
    let push = getConstants<CoverageBenchmarkPush>();
    if (any(id.xy >= push.size)) { return; }
#if METALLIC_TEST_LEGACY_COVERAGE
    LegacyCoverageProgram evaluator;
#else
    CoverageProgram evaluator;
#endif
    evaluator.inputs = resolveBuffer<ByteAddressBuffer>(getResourceParameters<SceneResourceParameters>().materialValues);
    uint material = (id.x + id.y * push.size) % push.materialCount;
    MaterialInstanceRef instance = coverageInstanceRef(material);
    CoverageContext context;
    context.uv = (float2(id.xy) + 0.5) / float(push.size);
    if (push.extremeInputs != 0) { context.uv = (context.uv - 0.5) * 1e8; }
    context.baseAlpha = 0.25 + float(id.x % 4) * 0.25;
    context.cutoff = 0.53125;
    context.alphaMode = 1.0;
    float opacity = 0.0;
    uint accepted = 0;
    for (uint repeat = 0; repeat < push.repeats; ++repeat) {
        context.uv += float2(0.001953125, -0.0009765625);
        CoverageResult result = evaluator.evaluateCoverage(context, instance);
        opacity += result.opacity;
        accepted |= result.accepted ? (1u << repeat) : 0u;
    }
    resolveUniform(getResourceParameters<SceneResourceParameters>().output)[id.xy] =
        float4(opacity, float(accepted), evaluator.usesBaseAlpha(instance) ? 1.0 : 0.0, 1.0);
}
