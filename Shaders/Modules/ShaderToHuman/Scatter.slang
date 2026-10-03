#language slang 2026
implementing ShaderToHuman;

// Copyright (c) 2024-2025 Electronic Arts Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Slang port of ShaderToHuman v14, revision 70ea227f3d504a19d9dd0539a832b68999426cca.
// See LICENSE.txt, NOTICE.txt and Upstream.json in this directory.

namespace Metallic.ShaderDebug
{
public interface IScatterSink
{
    [mutating] void writePixel(int2 pixel, float4 color);
}

public struct ContextScatter<TSink : IScatterSink>
{
    public TSink sink;
    // RGBA, alpha 1 is assumed to be opaque
    public float4 textColor;

    // private, for internal use, might change --------

    // in pixels
    public int2 pxCursor;
    // window left top, set by s2h_init()
    public int pxLeftX;
    // 1/2/3/4, call s2h_setScale()
    public int scale;
};

public void s2h_init<TSink : IScatterSink>(out ContextScatter<TSink> ui, TSink sink)
{
    // white, opaque
    ui.textColor = float4(1, 1, 1, 1);
    ui.pxCursor = int2(0, 0);
    ui.pxLeftX = int(ui.pxCursor.x);
    ui.scale = 1;
    ui.sink = sink;
}

public void s2h_setCursor<TSink : IScatterSink>(inout ContextScatter<TSink> ui, float2 inpxLeftTop)
{
    ui.pxCursor = int2(inpxLeftTop);
    ui.pxLeftX = int(inpxLeftTop.x);
}

public void s2h_setScale<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint scale)
{
    ui.scale = scale;
}

// The sink owns clipping and writes. Invocations must use disjoint regions.

public void s2h_printCharacter<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint ascii)
{
    [loop] for (int y = 0; y < 8 * ui.scale; ++y)
    [loop] for (int x = 0; x < 8 * ui.scale; ++x)
        if (s2h_fontLookup(ascii, int2(x, y) / ui.scale))
            ui.sink.writePixel(ui.pxCursor + int2(x, y), ui.textColor);

    ui.pxCursor.x += 8 * ui.scale;
}

public void s2h_drawCrosshair<TSink : IScatterSink>(inout ContextScatter<TSink> ui, float2 pxCenter, float pxRadius, float4 color)
{
    // avoiding int math for better performance
    ui.sink.writePixel(int2(pxCenter), ui.textColor);

    [loop] for (float i = 1; i < pxRadius; ++i)
    {
        ui.sink.writePixel(int2(pxCenter + float2(i, 0)), color);
        ui.sink.writePixel(int2(pxCenter + float2(-i, 0)), color);
        ui.sink.writePixel(int2(pxCenter + float2(0, i)), color);
        ui.sink.writePixel(int2(pxCenter + float2(0, -i)), color);
    }
}

public void s2h_printTxt<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint a)
{
    s2h_printCharacter(ui, a);
}
// Retain the upstream one-to-six character overloads.
public void s2h_printTxt<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint a, uint b)
{ s2h_printTxt(ui, a); s2h_printCharacter(ui, b); }
public void s2h_printTxt<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint a, uint b, uint c)
{ s2h_printTxt(ui, a, b); s2h_printCharacter(ui, c); }
public void s2h_printTxt<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint a, uint b, uint c, uint d)
{ s2h_printTxt(ui, a, b, c); s2h_printCharacter(ui, d); }
public void s2h_printTxt<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint a, uint b, uint c, uint d, uint e)
{ s2h_printTxt(ui, a, b, c, d); s2h_printCharacter(ui, e); }
public void s2h_printTxt<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint a, uint b, uint c, uint d, uint e, uint f)
{ s2h_printTxt(ui, a, b, c, d, e); s2h_printCharacter(ui, f); }

public void s2h_printLF<TSink : IScatterSink>(inout ContextScatter<TSink> ui)
{
    ui.pxCursor.x = ui.pxLeftX;
    ui.pxCursor.y += 8 * ui.scale;
}

public void s2h_printInt<TSink : IScatterSink>(inout ContextScatter<TSink> ui, int value)
{
    // leading '-'
    if (value < 0) {
        s2h_printCharacter(ui, '-');
        value = -value;
    }
    if (value == 0) {
        s2h_printCharacter(ui, '0');
        return;
    }
    // move to right depending on number length
    {
        uint tmp = (uint)value;
        while (tmp != 0u) {
            ui.pxCursor.x += 8 * ui.scale;
            tmp /= 10u;
        }
    }
    // digits
    {
        float backup = ui.pxCursor.x;
        uint tmp = (uint)value;
        while (tmp != 0u) {
            // 0..9
            uint digit = tmp % 10u;
            tmp /= 10u;
            // go backwards
            ui.pxCursor.x -= 8 * ui.scale;
            s2h_printCharacter(ui, '0' + digit);
            // counter +=8 from printCharacter ()
            ui.pxCursor.x -= 8 * ui.scale;
        }
        ui.pxCursor.x = int(backup);
    }
}

public void s2h_printHex<TSink : IScatterSink>(inout ContextScatter<TSink> ui, uint value)
{
    // 8 nibbles
    for (int i = 7; i >= 0; --i) {
        // 0..15
        uint nibble = (value >> (i * 4)) & 0xf;
        uint start = (nibble < 10) ? '0' : ('A' - 10);
        s2h_printCharacter(ui, start + nibble);
    }
}

public void s2h_printFloat<TSink : IScatterSink>(inout ContextScatter<TSink> ui, float value)
{
    s2h_printInt(ui, int(value));
    float fractional = frac(abs(value));

    s2h_printCharacter(ui, '.');

    uint digitCount = 3u;

    // todo: unit tests, this is likely wrong at lower precision

    // fractional digits
    for (uint i = 0u; i < digitCount; ++i) {
        fractional *= 10.0f;
        // 0..9
        uint digit = uint(fractional);
        fractional = frac(fractional);
        s2h_printCharacter(ui, '0' + digit);
    }
}

public void s2h_printBlock<TSink : IScatterSink>(inout ContextScatter<TSink> ui, float4 color)
{
    [loop] for (int y = 0; y < 8 * ui.scale; ++y)
    [loop] for (int x = 0; x < 8 * ui.scale; ++x)
    {
        float2 pxLocal = (float2(x, y) ) / ui.scale - float2(3.5f, 3.5f);

        float mask = saturate(4 - max(abs(pxLocal.x), abs(pxLocal.y)));

        if (mask > 0.0f)
            ui.sink.writePixel(ui.pxCursor + int2(x,y), color);
    }

    ui.pxCursor.x += 8 * ui.scale;
}

public void s2h_printDisc<TSink : IScatterSink>(inout ContextScatter<TSink> ui, float4 color)
{
    [loop] for (int y = 0; y < 8 * ui.scale; ++y)
    [loop] for (int x = 0; x < 8 * ui.scale; ++x)
    {
        float2 pxLocal = (float2(x, y) ) / ui.scale - float2(3.5f, 3.5f);

        float mask = saturate(4 - length(pxLocal));

        if (mask > 0.0f)
            ui.sink.writePixel(ui.pxCursor + int2(x,y), color);
    }

    ui.pxCursor.x += 8 * ui.scale;
}

} // namespace Metallic.ShaderDebug
