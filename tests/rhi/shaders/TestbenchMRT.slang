[shader("vertex")]
float4 vertexMain(uint id : SV_VertexID) : SV_Position
{
    float2 points[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(points[id], 0, 1);
}

struct Targets
{
    float4 red : SV_Target0;
    float4 green : SV_Target1;
    float4 blue : SV_Target2;
    float4 yellow : SV_Target3;
};

[shader("fragment")]
Targets fragmentMain()
{
    Targets result;
    result.red = float4(1, 0, 0, 1);
    result.green = float4(0, 1, 0, 1);
    result.blue = float4(0, 0, 1, 1);
    result.yellow = float4(1, 1, 0, 1);
    return result;
}

[shader("fragment")]
void depthMain()
{
}
