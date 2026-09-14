import Core;
using Metallic;

#define environmentTexture getResource<Texture2D<float4>>(0)
#define result getResource<RWStructuredBuffer<float4>>(1)

[shader("compute")]
[numthreads(1, 1, 1)]
void readEnvironment(uint3 thread : SV_DispatchThreadID)
{
    result[0] = environmentTexture.Load(int3(0, 0, 0));
}
