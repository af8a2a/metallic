#include "../../Libraries/Resources/ComputeResources.slang"

// Standalone linear-color example. A 320x192 output shows the entire overlay.
// Compile either the compute or fragment entry; only compute binds gOutput.
#include "ShaderToHuman.slang"

#define gOutput METALLIC_RESOURCE(RWTexture2D<float4>, 0)

float4 shaderToHumanExampleColor(float2 pixelCenter)
{
    float4 background = float4(0.025f, 0.035f, 0.05f, 1.0f);

    // An opaque debug sphere viewed by one normalized ray per output pixel.
    // For a scene overlay, use the actual camera ray and set world.depth to
    // the visible surface's ray distance before drawing debug geometry.
    Context3D world;
    float2 screen = (pixelCenter - float2(248.0f, 112.0f)) / 64.0f;
    s2h_init(world, float3(0.0f, 0.0f, -3.0f), normalize(float3(screen.x, -screen.y, 2.0f)));
    s2h_drawSphereWS(world, float3(0.0f, 0.0f, 0.0f), float4(0.15f, 0.7f, 0.3f, 1.0f), 0.65f);
    if (world.dstColor.a > 0.0f) {
        background = world.dstColor;
    }

    ContextGather ui;
    s2h_init(ui, pixelCenter);
    s2h_setCursor(ui, float2(8.0f, 8.0f));
    s2h_setScale(ui, 2.0f);
    s2h_printTxt(ui, _S, _2, _H);
    s2h_printSpace(ui, 1.0f);
    s2h_printInt(ui, S2H_VERSION);
    s2h_printLF(ui);

    s2h_setScale(ui, 1.0f);
    s2h_printTxt(ui, _V, _a, _l, _u, _e);
    s2h_printSpace(ui, 1.0f);
    s2h_printFloat(ui, 0.35f);
    s2h_printLF(ui);
    s2h_printTxt(ui, _I, _D);
    s2h_printSpace(ui, 1.0f);
    s2h_printHex(ui, 0x1234abcdu);

    s2h_drawLine(ui, float2(8.5f, 72.5f), float2(160.5f, 72.5f), float4(0.2f, 0.7f, 1.0f, 1.0f));
    s2h_drawCrosshair(ui, float2(80.5f, 112.5f), 12.0f, float4(1.0f, 0.5f, 0.1f, 1.0f));
    return ShaderDebug::composite(background, ui);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shaderToHumanExampleMain(uint3 dispatchId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    if (any(dispatchId.xy >= uint2(width, height))) { return; }
    gOutput[dispatchId.xy] = shaderToHumanExampleColor(float2(dispatchId.xy) + 0.5f);
}

[shader("fragment")]
float4 shaderToHumanExampleFragmentMain(float4 position : SV_Position) : SV_Target
{
    // SV_Position already addresses the pixel center.
    return shaderToHumanExampleColor(position.xy);
}
