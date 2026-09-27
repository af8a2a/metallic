[shader("vertex")]
float4 vertexMain(uint id : SV_VertexID) : SV_Position
{
    float2 points[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(points[id], 0, 1);
}
[shader("fragment")]
float4 redMain() : SV_Target { return float4(1, 0, 0, 1); }
[shader("fragment")]
float4 greenMain() : SV_Target { return float4(0, 1, 0, 1); }
