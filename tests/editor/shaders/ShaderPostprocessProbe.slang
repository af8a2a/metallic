#language slang 2026
module ShaderPostprocessProbe;

public interface IPostprocessPolicy
{
    public static bool enableExtra();
    public uint evaluate(uint value);
}

public struct DisabledPostprocessPolicy : IPostprocessPolicy
{
    public static bool enableExtra() { return false; }
    public uint evaluate(uint value) { return value * 33u + 17u; }
}

uint evaluatePolicy<T : IPostprocessPolicy>(uint value)
{
    T policy;
    if (T.enableExtra()) {
        value = policy.evaluate(value);
    }
    return value;
}

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> output;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output[id.x] = evaluatePolicy<DisabledPostprocessPolicy>(id.x + 1u);
}
