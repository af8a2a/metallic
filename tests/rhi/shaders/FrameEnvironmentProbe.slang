[[vk::binding(0, 0)]] Texture2D<float4> environmentTexture;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> result;

[shader("compute")]
[numthreads(1, 1, 1)]
void readEnvironment(uint3 thread : SV_DispatchThreadID)
{
    result[0] = environmentTexture.Load(int3(0, 0, 0));
}
