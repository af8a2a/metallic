[[vk::binding(0, 0)]] RWStructuredBuffer<uint> output;

uint evaluate(uint value)
{
    uint result = value;
    for (uint i = 0; i < 4; ++i) {
        result = (result ^ (i * 17)) * 33 + 7;
    }
    return result;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output[id.x] = evaluate(id.x);
}
