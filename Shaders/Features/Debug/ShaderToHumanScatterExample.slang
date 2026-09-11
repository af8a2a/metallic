#include "../../Libraries/Resources/ComputeResources.slang"

// Dispatch after the destination is initialized, with a storage-write barrier
// between this dispatch and earlier image writes. One invocation owns all text.
#include "ShaderToHumanScatter.slang"

#define gOutput METALLIC_RESOURCE(RWTexture2D<float4>, 0)

void onGfxForAllScatter(int2 pxPos, float4 color)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    if (any(pxPos < 0) || any(uint2(pxPos) >= uint2(width, height))) { return; }
    gOutput[uint2(pxPos)] = color;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void shaderToHumanScatterExampleMain(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId != uint3(0, 0, 0))) { return; }
    ContextScatter ui;
    s2h_init(ui);
    s2h_setCursor(ui, float2(8.0f, 160.0f));
    s2h_printTxt(ui, _P, _r, _o, _b, _e);
    s2h_printInt(ui, 42);
    s2h_printFloat(ui, 0.35f);
    s2h_printHex(ui, 0x1234abcdu);
}
