import Core;
using Metallic;

struct ProbePush
{
    uint index;
};

#define gInput getResource<StructuredBuffer<uint>>(0)
#define gOutput getResource<RWStructuredBuffer<uint>>(1)

[shader("compute")]
[numthreads(1, 1, 1)]
void copyValue(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = getConstants<ProbePush>();
    gOutput[push.index] = gInput[0];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void sampleImages(uint3 thread : SV_DispatchThreadID)
{
    ProbePush push = getConstants<ProbePush>();
    float a = getResourceArray<Texture2D<float4>>(0, 0).Load(int3(0, 0, 0)).x;
    float b = getResourceArray<Texture2D<float4>>(0, 1).Load(int3(0, 0, 0)).x;
    gOutput[push.index] = uint(a + b);
}
