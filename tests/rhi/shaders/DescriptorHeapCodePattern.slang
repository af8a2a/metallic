import ShaderCore;
using Metallic;

// One DR payload and identical arithmetic; only aggregate access changes.
#ifndef METALLIC_GPU_PATTERN
#define METALLIC_GPU_PATTERN 0
#endif

struct PatternValues
{
    uint a;
    uint b;
    uint c;
    uint d;
    uint e;
    uint f;
    uint g;
    uint h;
};

struct PatternPush
{
    PatternValues values;
    ResourceHandle<Texture2D<float4>> image;
    ResourceHandle<RWStructuredBuffer<PatternValues>> input;
    ResourceHandle<RWStructuredBuffer<uint>> output;
};

#if METALLIC_GPU_PATTERN != 1
[[vk::push_constant]] ConstantBuffer<PatternPush> gPush;
#endif

// A macro deliberately avoids introducing a by-value helper parameter in the
// control variants. Every field contributes to the observable result.
#define PATTERN_SUM(v) ((v).a + 2u * (v).b + 3u * (v).c + 4u * (v).d + \
    5u * (v).e + 6u * (v).f + 7u * (v).g + 8u * (v).h)

[shader("compute")]
[numthreads(128, 1, 1)]
void descriptorHeapCodePatternMain(uint3 dispatchThreadId : SV_DispatchThreadID
#if METALLIC_GPU_PATTERN == 1
    , uniform PatternPush parameters
#endif
)
{
#if METALLIC_GPU_PATTERN == 1
    uint pushSum = PATTERN_SUM(parameters.values);
#elif METALLIC_GPU_PATTERN == 2
    PatternValues push = gPush.values;
    uint pushSum = PATTERN_SUM(push);
#else
    uint pushSum = PATTERN_SUM(gPush.values);
#endif

#if METALLIC_GPU_PATTERN == 1
#define PATTERN_RESOURCES parameters
#else
#define PATTERN_RESOURCES gPush
#endif
    let gImage = resolveUniform(PATTERN_RESOURCES.image);
    let gInput = resolveUniform(PATTERN_RESOURCES.input);
    let gOutput = resolveUniform(PATTERN_RESOURCES.output);

#if METALLIC_GPU_PATTERN == 3
    PatternValues input = gInput[0];
    uint inputSum = PATTERN_SUM(input);
#elif METALLIC_GPU_PATTERN == 4
    PatternValues input;
    input.a = gInput[0].a;
    input.b = gInput[0].b;
    input.c = gInput[0].c;
    input.d = gInput[0].d;
    input.e = gInput[0].e;
    input.f = gInput[0].f;
    input.g = gInput[0].g;
    input.h = gInput[0].h;
    uint inputSum = PATTERN_SUM(input);
#else
    uint inputSum = PATTERN_SUM(gInput[0]);
#endif

    float4 texel = gImage.Load(int3(0, 0, 0));
    uint imageSum = uint(dot(texel, float4(1.0, 2.0, 3.0, 4.0)));
    gOutput[dispatchThreadId.x] = pushSum + inputSum + imageSum + dispatchThreadId.x;
}
