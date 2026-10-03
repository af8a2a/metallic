import ShaderCore;
using Metallic;

// Consumer-local legacy macros must not alter the imported module.
#define S2H_DISABLE_EMBEDDED_FONT 1
#define S2H_GLSL 1
import ShaderToHuman;
using Metallic.ShaderDebug;

struct ProbeParams
{
    RWBufferSpan<float4> output;
};
[[vk::push_constant]] ConstantBuffer<ProbeParams> gParams;
static const uint kWidth = 320;
static const uint kHeight = 256;
static const uint kPixels = kWidth * kHeight;

void writeScatterPixel(int2 pixel, float4 color)
{
    if (any(pixel < 0) || any(uint2(pixel) >= uint2(kWidth, kHeight))) { return; }
    gParams.output.store(kPixels + uint(pixel.y) * kWidth + uint(pixel.x), color);
}

struct ProbeSink : IScatterSink
{
    [mutating] void writePixel(int2 pixel, float4 color) { writeScatterPixel(pixel, color); }
}
struct IntTable : IIntTable
{
    bool lookup(uint column, uint row, out int value)
    {
        value = int(column * 10 + row) - 3;
        return row < 3;
    }
}
struct FloatTable : IFloatTable
{
    bool lookup(uint column, uint row, out float value)
    {
        value = float(column + row) * 0.125;
        return row < 3;
    }
}
struct FunctionPlot : IFunctionPlot
{
    float evaluate(uint functionId, float x) { return sin(x) * float(functionId); }
}
struct DebugScene : IScene3D
{
    void draw(inout Context3D world)
    {
        s2h_drawSphereWS(world, float3(0), float4(0.15, 0.7, 0.3, 1), 0.65);
        s2h_drawAABB(world, float3(-1, 0, 0), float3(0.2), float4(0.8, 0.2, 0.1, 1));
        s2h_drawLineWS(world, float3(0, -1, 0), float3(1, -1, 0), float4(0.2, 0.4, 0.8, 1), 0.06);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shaderToHumanGatherProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= kWidth || id.y >= kHeight) { return; }
    float2 pixel = float2(id.xy) + 0.5;
    ContextGather ui;
    s2h_init(ui, pixel);
    s2h_setCursor(ui, float2(8));
    for (uint character = 32; character < 128; ++character) {
        s2h_printCharacter(ui, character);
        if ((character & 15) == 15) { s2h_printLF(ui); }
    }
    ui.textColor = float4(0.8, 0.4, 0.2, 0.5);
    s2h_setScale(ui, 2.0);
    s2h_printTxt(ui, _S, _2, _H);
    s2h_setScale(ui, 1.0);
    s2h_printInt(ui, -42);
    s2h_printFloat(ui, 0.35);
    s2h_printLF(ui);
    s2h_printHex(ui, 0x1234abcdu);
    s2h_printSpace(ui, 1);
    s2h_printDisc(ui, float4(0, 1, 0, 0.5));
    s2h_printBox(ui, float4(1, 0, 0, 0.5));
    s2h_drawLine(ui, float2(8.5, 88.5), float2(136.5, 92.5), float4(0.2, 0.7, 1, 0.7));
    s2h_drawCrosshair(ui, float2(80.5, 112.5), 12, float4(1, 0.5, 0.1, 1));
    s2h_drawCircle(ui, float2(120, 112), 10, float4(0.4, 0.2, 1, 0.7));
    s2h_setCursor(ui, float2(8, 136));
    s2h_tableInt(ui, 1, float4(0.1, 0.2, 0.3, 0.5), int2(4, 4), true, IntTable());
    s2h_tableFloat(ui, 1, float4(0.2, 0.1, 0.3, 0.5), int2(6, 4), false, FloatTable());
    s2h_function(ui, 1, float4(0.1, 0.1, 0.1, 0.5), int2(12, 4), float2(0, 6.28), float2(-1, 1), FunctionPlot());
    s2h_setCursor(ui, float2(8, 220));
    ui.mouseInput = float4(12.5, 222.5, 1, 0);
    s2h_checkBox(ui, true);
    s2h_radioButton(ui, true);
    s2h_progress(ui, 6, 0.65);
    Context3D world;
    float2 screen = (pixel - float2(248, 112)) / 48;
    s2h_init(world, float3(0, 0, -3), normalize(float3(screen.x, -screen.y, 2)));
    sceneWithShadows(world, DebugScene());
    float4 background = float4(0.025, 0.035, 0.05, 1);
    if (world.dstColor.a > 0) { background = world.dstColor; }
    float4 result = composite(background, ui);
    gParams.output.store(id.y * kWidth + id.x, result);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void shaderToHumanScatterProbeMain()
{
    ContextScatter<ProbeSink> ui;
    s2h_init(ui, ProbeSink());
    s2h_setCursor(ui, float2(-4, 8)); // Exercise caller-owned clipping.
    s2h_printTxt(ui, _P, _r, _o, _b, _e);
    s2h_printInt(ui, -42);
    s2h_printLF(ui);
    s2h_printFloat(ui, 0.35);
    s2h_printHex(ui, 0x1234abcdu);
    s2h_setScale(ui, 2);
    s2h_printLF(ui);
    s2h_printBlock(ui, float4(0.8, 0.3, 0.1, 1));
    s2h_printDisc(ui, float4(0.1, 0.8, 0.3, 1));
    s2h_drawCrosshair(ui, float2(80, 64), 12, float4(0.2, 0.4, 0.8, 1));
}
