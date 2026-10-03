import Core;
using Metallic;

// Dispatch after the destination is initialized, with a storage-write barrier
// between this dispatch and earlier image writes. One invocation owns all text.
import ShaderToHuman;
using Metallic.ShaderDebug;

#define gOutput resolveUniform(getResourceParameters<OutputImageResourceParameters>().output)

struct OutputSink : IScatterSink
{
    [mutating]
    void writePixel(int2 pixel, float4 color)
    {
        uint width, height;
        gOutput.GetDimensions(width, height);
        if (any(pixel < 0) || any(uint2(pixel) >= uint2(width, height))) { return; }
        gOutput[uint2(pixel)] = color;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void shaderToHumanScatterExampleMain(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId != uint3(0, 0, 0))) { return; }
    ContextScatter<OutputSink> ui;
    s2h_init(ui, OutputSink());
    s2h_setCursor(ui, float2(8.0f, 160.0f));
    s2h_printTxt(ui, _P, _r, _o, _b, _e);
    s2h_printInt(ui, 42);
    s2h_printFloat(ui, 0.35f);
    s2h_printHex(ui, 0x1234abcdu);
}
