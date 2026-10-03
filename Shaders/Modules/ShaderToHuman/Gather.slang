#language slang 2026
implementing ShaderToHuman;

// Copyright (c) 2024-2025 Electronic Arts Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Slang port of ShaderToHuman v14, revision 70ea227f3d504a19d9dd0539a832b68999426cca.
// See LICENSE.txt, NOTICE.txt and Upstream.json in this directory.

namespace Metallic.ShaderDebug
{
public struct ContextGather
{
    // in pixels, no fractional part (half pixel offset)
    public float2 pxCursor;
    // 1/2/3/4
    public float scale;
    // .xy:-100,-100 if not yet set, .xy:absolutePos, z:leftMouse 0/1, w:rightMouse 0/1, no fractional part (half pixel offset)
    public float4 mouseInput;

    // window left top, no fractional part (half pixel offset), set by s2h_setPos(), used by s2h_printLF()
    public float pxLeftX;
    // in pixels, no fractional part (half pixel offset), set by s2h_init()
    public float2 pxPos;
    // premultiplied RGBA, alpha 1 is assumed to be opaque, don't init with a color or s2h_button() will not work
    public float4 dstColor;

    //

    // RGBA, alpha 1 is assumed to be opaque, s2h_progress()
    public float4 textColor;
    // for s2h_frame()
    public float4 frameFillColor;
    // for s2h_frame()
    public float4 frameBorderColor;
    // for s2h_button(), s2h_checkbox(), s2h_radiobutton, s2h_progress(), s2h_sliderFloat()
    public float4 buttonColor;
    //
    public float lineWidth;

    // private ----------------------

    // for interactive UI, read int4 state from former frame
    public int4 s2h_State;
};

public struct s2h_Triangle
{
    public float2 A;
    public float2 B;
    public float2 C;
};

// Data sources belong to the calling program, not to this module.
public interface IIntTable
{
    bool lookup(uint column, uint row, out int value);
}

public interface IFloatTable
{
    bool lookup(uint column, uint row, out float value);
}

public interface IFunctionPlot
{
    float evaluate(uint functionId, float x);
}

public void s2h_printCharacter(inout ContextGather ui, uint ascii)
{
    int2 pxLocal = int2(floor((ui.pxPos - ui.pxCursor) / ui.scale));

    if (s2h_fontLookup(ascii, pxLocal))
        ui.dstColor = lerp(ui.dstColor, float4(ui.textColor.rgb, 1), ui.textColor.a);

    ui.pxCursor.x += s2h_fontSize() * ui.scale;
}

public void s2h_init(out ContextGather ui, float2 inPxPos)
{
    // white, opaque
    ui.textColor = float4(1, 1, 1, 1);
    ui.pxLeftX = 0.0f;
    ui.pxCursor = float2(0, 0);
    ui.scale = 1.0f;
    ui.mouseInput = float4(-100, -100, 0, 0);

    ui.pxPos = inPxPos;
    // see through
    ui.dstColor = float4(0, 0, 0, 0);
    ui.s2h_State = int4(0, 0, 0, 0);

    ui.frameFillColor = float4(0.9f, 0.9f, 0.9f, 1);
    ui.frameBorderColor = float4(0.7f, 0.7f, 0.7f, 1);
    ui.buttonColor = float4(0.5f, 0.5f, 0.5f, 1);
    ui.lineWidth = 2.0f;
}

public void s2h_setCursor(inout ContextGather ui, float2 inpxLeftTop)
{
    ui.pxCursor = inpxLeftTop;
    ui.pxLeftX = inpxLeftTop.x;
}

public void s2h_deinit(inout ContextGather ui, out int4 s2h_State)
{
    // if mouse input was set and mouse is released, we forget which button was active
    if (ui.mouseInput.x != -100.0f && ui.mouseInput.z == 0.0f)
        ui.s2h_State = int4(0,0,0,0);

    s2h_State = ui.s2h_State;
}

public void s2h_setScale(inout ContextGather ui, float scale)
{
    ui.scale = scale;
}

public void s2h_printTxt(inout ContextGather ui, uint a)
{
    s2h_printCharacter(ui, a);
}
// Retain the upstream one-to-six character overloads.
public void s2h_printTxt(inout ContextGather ui, uint a, uint b)
{ s2h_printTxt(ui, a); s2h_printCharacter(ui, b); }
public void s2h_printTxt(inout ContextGather ui, uint a, uint b, uint c)
{ s2h_printTxt(ui, a, b); s2h_printCharacter(ui, c); }
public void s2h_printTxt(inout ContextGather ui, uint a, uint b, uint c, uint d)
{ s2h_printTxt(ui, a, b, c); s2h_printCharacter(ui, d); }
public void s2h_printTxt(inout ContextGather ui, uint a, uint b, uint c, uint d, uint e)
{ s2h_printTxt(ui, a, b, c, d); s2h_printCharacter(ui, e); }
public void s2h_printTxt(inout ContextGather ui, uint a, uint b, uint c, uint d, uint e, uint f)
{ s2h_printTxt(ui, a, b, c, d, e); s2h_printCharacter(ui, f); }

public void s2h_printSpace(inout ContextGather ui, float numberOfChars)
{
    ui.pxCursor.x += s2h_fontSize() * numberOfChars * ui.scale;
}

public void s2h_printLF(inout ContextGather ui)
{
    ui.pxCursor.x = ui.pxLeftX;
    ui.pxCursor.y += s2h_fontSize() * ui.scale;
}

public void s2h_printInt(inout ContextGather ui, int value)
{
    // leading '-'
    if (value < 0) {
        s2h_printCharacter(ui, _MINUS);
        value = -value;
    }
    if (value == 0) {
        s2h_printCharacter(ui, _0);
        return;
    }
    // move to right depending on number length
    {
        uint tmp = uint(value);
        while (tmp != 0u) {
            ui.pxCursor.x += s2h_fontSize() * ui.scale;
            tmp /= 10u;
        }
    }
    // digits
    {
        float backup = ui.pxCursor.x;
        uint tmp = uint(value);
        while (tmp != 0u) {
            // 0..9
            uint digit = tmp % 10u;
            tmp /= 10u;
            // go backwards
            ui.pxCursor.x -= s2h_fontSize() * ui.scale;
            s2h_printCharacter(ui, _0 + digit);
            // counter +=s2h_fontSize() from printCharacter ()
            ui.pxCursor.x -= s2h_fontSize() * ui.scale;
        }
        ui.pxCursor.x = backup;
    }
}

public void s2h_printHex(inout ContextGather ui, uint value)
{
    // 8 nibbles
    for (int i = 7; i >= 0; --i) {
        // 0..15
        uint nibble = (value >> (uint(i) * 4u)) & 0xfu;
        uint start = (nibble < 10u) ? _0 : (_A - 10u);
        s2h_printCharacter(ui, start + nibble);
    }
}

public void s2h_printFloat(inout ContextGather ui, float value)
{
    s2h_printInt(ui, int(value));
    float fractional = frac(abs(value));

    s2h_printCharacter(ui, _PERIOD);

    uint digitCount = 3u;

    // todo: unit tests, this is likely wrong at lower precision

    // fractional digits
    for (uint i = 0u; i < digitCount; ++i) {
        fractional *= 10.0f;
        // 0..9
        uint digit = uint(fractional);
        fractional = frac(fractional);
        s2h_printCharacter(ui, _0 + digit);
    }
}

public void s2h_printBox(inout ContextGather ui, float4 color)
{
    float2 pxLocal = float2(ui.pxPos - ui.pxCursor) / float(ui.scale) - float2(4, 4);

    float mask = saturate(4.0f - max(abs(pxLocal.x), abs(pxLocal.y)));

//  dstColor = lerp(dstColor, float4(color.rgb, 1), color.a * mask);
    if (mask > 0.0f)
        ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a);

    ui.pxCursor.x += s2h_fontSize() * ui.scale;
}

public void s2h_drawDisc(inout ContextGather ui, float2 pxCenter, float pxRadius, float4 color)
{
    float2 pxLocal = ui.pxPos - pxCenter;

    float len = length(pxLocal);
    float mask = saturate(pxRadius - len);

    ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a * mask);
}

public void s2h_drawCircle(inout ContextGather ui, float2 pxCenter, float pxRadius, float4 color)
{
    float r = ui.lineWidth * 0.5f;
    float2 pxLocal = ui.pxPos - pxCenter;

    float len = length(pxLocal);
    float mask = saturate(pxRadius - len + r) * (1.0f - saturate(pxRadius - len - r));

    ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a * mask);
}

public void s2h_drawHalfSpace(inout ContextGather ui, float3 halfSpace, float2 visualizePoint, float4 color, float pxCircleRadius, float lineRadius)
{
    // normalize
    halfSpace /= length(halfSpace.xy);

    //
    float2 onPoint = visualizePoint - halfSpace.xy * dot(halfSpace, float3(visualizePoint, 1));

    float planeDist = dot(halfSpace, float3(ui.pxPos, 1));
    float diskDist = length(onPoint - ui.pxPos);

    // 0..1
    float sideMask = saturate(planeDist);
    // 0..1
    float lineMask = saturate(ui.lineWidth - abs(planeDist - ui.lineWidth)) * saturate(lineRadius - diskDist);
    // 0..1
    float semiDiskMask = saturate(pxCircleRadius - diskDist) * sideMask;
    float mask = max(semiDiskMask, lineMask);

    ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a * mask);
}

public void s2h_drawRectangle(inout ContextGather ui, float2 pxLeftTop, float2 pxBottomRight, float4 color)
{
    if (ui.pxPos.x >= pxLeftTop.x && ui.pxPos.y >= pxLeftTop.y && ui.pxPos.x < pxBottomRight.x && ui.pxPos.y < pxBottomRight.y)
        ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a);
}

public void s2h_drawRectangleAA(inout ContextGather ui, float2 pxA, float2 pxB, float4 borderColor, float4 innerColor, float pxThickness)
{
    float r = pxThickness * 0.5f;

    float2 pxCenter = (pxA + pxB) * 0.5f;
    float2 pxHalfSize = abs(pxB - pxA) * 0.5f;

    float2 pxLocalOuter = max(abs(ui.pxPos - pxCenter) - pxHalfSize, float2(0, 0));
    float2 pxLocalInner = max(abs(ui.pxPos - pxCenter) - pxHalfSize + r, float2(0, 0));

    float maskOuter = saturate(1.0f + r - length(pxLocalOuter));
    float maskInner = saturate(length(pxLocalInner) - 0.5f);

    float4 color = lerp(innerColor, float4(borderColor.rgb, 1), borderColor.a * maskInner);

    ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a * maskOuter);
}

public void s2h_drawCrosshair(inout ContextGather ui, float2 pxCenter, float pxRadius, float4 color)
{
    float2 h = float2(pxRadius, 0);
    float2 v = float2(0, pxRadius);

    s2h_drawLine(ui, pxCenter - h, pxCenter + h, color);
    s2h_drawLine(ui, pxCenter - v, pxCenter + v, color);
}

public void s2h_drawLine(inout ContextGather ui, float2 pxBegin, float2 pxEnd, float4 color)
{
    float pxThickness = ui.lineWidth + 1.0f;
    float r = pxThickness * 0.5f;
    float2 delta = pxEnd - pxBegin;
    float len = length(delta);
    if (len > 0.01f) {
        float2 tangent = delta / len;
        float2 normal = float2(tangent.y, -tangent.x);
        float2 local = float2(ui.pxPos) - pxBegin;
        float2 uv = float2(dot(local, tangent), dot(local, normal));
        // 0...1
        float mask = saturate(r - abs(uv.y)) * saturate(r - uv.x + len) * saturate(r + uv.x);

        ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a * mask);
    }
}

internal float3 s2h_getHalfSpacePlane(float2 pointA, float2 pointB)
{
    float2 ab = normalize(pointA - pointB);
    float3 abPlane = float3(-ab.y, ab.x, 0);
    abPlane.z = dot(abPlane.xy, -pointA);

    return abPlane;
}

public void s2h_drawTriangle(inout ContextGather ui, s2h_Triangle tri, float4 color)
{
    float3 abPlane = s2h_getHalfSpacePlane(tri.A, tri.B);
    float abMask = saturate(dot(abPlane, float3(ui.pxPos, 1)) - 0.5f);

    float3 bcPlane = s2h_getHalfSpacePlane(tri.B, tri.C);
    float bcMask = saturate(dot(bcPlane, float3(ui.pxPos, 1))- 0.5f);

    float3 caPlane = s2h_getHalfSpacePlane(tri.C, tri.A);
    float caMask = saturate(dot(caPlane, float3(ui.pxPos, 1)) - 0.5f);

    float mask = abMask * bcMask * caMask;
    ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a * mask);
}

public void s2h_drawArrow(inout ContextGather ui, float2 pxStart, float2 pxEnd, float4 color,  float arrowHeadLength, float arrowHeadWidth)
{
    float2 direction = float2(0,1);
    direction = normalize(pxEnd - pxStart);

    float2 lineStart = pxStart;
    // Subtract the arrow length from lineEnd - arrow fits in pxStart...pxEnd
    float2 lineEnd = pxEnd - direction * arrowHeadLength;

    float2 perpendicularDir = normalize(float2(direction.y, -direction.x));

    s2h_drawLine(ui, lineStart, lineEnd, color);

    s2h_Triangle triA;
    triA.A = lineEnd - perpendicularDir * arrowHeadWidth;
    triA.B = lineEnd + direction * arrowHeadLength;
    triA.C = lineEnd + perpendicularDir * arrowHeadWidth;
    s2h_drawTriangle(ui, triA, color);
}

public void s2h_drawSRGBRamp(inout ContextGather ui, float2 pxPos)
{
    // snap to pixel center
    pxPos = floor(pxPos) + 0.5f;

    float2 local = ui.pxPos - pxPos;

    float u = local.x / 256.0f;

    if (local.y > 16.0f)
        u = floor(u * 16.0f) / 16.0f;

    float3 col = s2h_accurateSRGBToLinear(float3(u, u, u));

    s2h_drawRectangle(ui, pxPos - 2.0f, pxPos + float2(256, 32) + 2.0f, float4(s2h_colorRampRGB(u), 1));
    s2h_drawRectangle(ui, pxPos, pxPos + float2(256, 32), float4(col, 1));

    ContextGather backup = ui;
    s2h_setScale(ui, 1.0f);
    ui.textColor = float4(1, 1, 1, 1);
    s2h_setCursor(ui, pxPos + float2(2.0f, 22));
    s2h_printTxt(ui, _0);
    s2h_setCursor(ui, pxPos + float2(128.0f - 1.5f * 8.0f, 22));
    s2h_printTxt(ui, _1, _2, _7);
    ui.textColor = float4(0, 0, 0, 1);
    s2h_setCursor(ui, pxPos + float2(256.0f - 3.2f * 8.0f, 22));
    s2h_printTxt(ui, _2, _5, _5);

    ui.pxCursor = backup.pxCursor;
    ui.scale = backup.scale;
    ui.textColor = backup.textColor;
    ui.pxLeftX = backup.pxLeftX;
}

public void s2h_coordinateSystem(inout ContextGather ui, float2 pxOrigin, float4 domain, float pxScale, float gridSize, float4 gridColor, uint flags)
{
    // floor() to snap to pixel grid for consistent visuals
    pxOrigin = floor(pxOrigin);

    bool yIsUp = (flags & 0x1u) != 0u;
    bool gridLines = (flags & 0x2u) != 0u;

    float4 color = ui.textColor;

    float pxDotSize = ui.lineWidth;
    // todo: refine math to be consistent in grid size
    float2 pxD = frac(floor(ui.pxPos - pxOrigin + pxDotSize / 2.0f) / gridSize) * gridSize;

    if (gridLines) {
        if (pxD.x > floor(pxDotSize) && pxD.y > floor(pxDotSize))
            gridColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        if (pxD.x > floor(pxDotSize) || pxD.y > floor(pxDotSize))
            gridColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float4 pxDomain = domain * pxScale + float4(pxOrigin, pxOrigin);

    s2h_drawRectangle(ui, pxDomain.xy, pxDomain.zw, gridColor);

    float backupLineWidth = ui.lineWidth;

    // X axis
    if (pxD.x < backupLineWidth)
        ui.lineWidth *= 2.0f;
    s2h_drawArrow(ui, float2(pxDomain.x, pxOrigin.y), float2(pxDomain.z, pxOrigin.y), color, 16.0f, 8.0f);

    // Y axis
    ui.lineWidth = backupLineWidth;
    if (pxD.y < backupLineWidth)
        ui.lineWidth *= 2.0f;

    if (yIsUp)
        s2h_drawArrow(ui, float2(pxOrigin.x, pxDomain.w), float2(pxOrigin.x, pxDomain.y), color, 16.0f, 8.0f);
    else
        s2h_drawArrow(ui, float2(pxOrigin.x, pxDomain.y), float2(pxOrigin.x, pxDomain.w), color, 16.0f, 8.0f);

    ui.lineWidth = backupLineWidth;
}

public void s2h_printDisc(inout ContextGather ui, float4 color)
{
    float2 pxLocal = float2(ui.pxPos - ui.pxCursor) / float(ui.scale) - float2(4, 4);

    float mask = saturate(4.0f - length(pxLocal));

//  dstColor = lerp(stColor, float4(color.rgb, 1), color.a * mask);
    // no AA for now
    if (mask > 0.0f)
        ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a);

    ui.pxCursor.x += s2h_fontSize() * ui.scale;
}

internal float s2h_computeDistToBox(inout ContextGather ui, float2 p, float2 center, float2 halfSize)
{
    float2 pxLocal = float2(p - center);// - float2(3.5f, 3.5f) * ui.scale;
    float2 dist2 = max(float2(0, 0), abs(pxLocal) - halfSize);
    return max(dist2.x, dist2.y);
}

// @param aabb .x:minx, .y:miny, z:maxx, w:maxy
internal float s2h_computeDistToBox(inout ContextGather ui, float2 p, float4 aabb)
{
    float2 center = (aabb.xy + aabb.zw) * 0.5f;
    float2 halfSize = (aabb.zw - aabb.xy) * 0.5f;
    float2 pxLocal = float2(p - center);// - float2(3.5f, 3.5f) / ui.scale;
    float2 dist2 = max(float2(0, 0), abs(pxLocal) - halfSize);
    return max(dist2.x, dist2.y);
}

public void s2h_frame(inout ContextGather ui, uint widthInCharacters)
{
    float4 aabb = float4(ui.pxCursor - float2(widthInCharacters, 0) * s2h_fontSize() * ui.scale, ui.pxCursor);

    // shrink
    aabb += float4(4, 4, -4, 4) * ui.scale;

    float dist = s2h_computeDistToBox(ui, ui.pxPos, aabb) / ui.scale;

    float rimMask = saturate(3.0f - dist);
    float outerMask = saturate(4.0f - dist);

    float4 localColor = float4(0,0,0,0);

    // no AA for now
    if (outerMask > 0.0f)
        localColor = ui.frameBorderColor;

    if (rimMask > 0.0f)
        localColor = ui.frameFillColor;

    ui.dstColor = lerp(ui.dstColor, float4(localColor.rgb, 1), localColor.a * (1.0f - ui.dstColor.a));
}

public bool s2h_button(inout ContextGather ui, uint widthInCharacters)
{
    float4 color = ui.buttonColor;
    const float border = 0.0f;

    float4 aabb = float4(ui.pxCursor - float2(widthInCharacters, 0) * s2h_fontSize() * ui.scale, ui.pxCursor);

    // shrink
    aabb += float4(4, 4, -4, 4) * ui.scale;

    float dist = s2h_computeDistToBox(ui, ui.pxPos, aabb) / ui.scale;
    bool mouseOver = s2h_computeDistToBox(ui, ui.mouseInput.xy, aabb) / ui.scale < 5.0f + border;

    float rimMask = saturate(5.0f - dist + border);
    float outerMask = saturate(4.0f - dist + border);

    float4 localColor = float4(0,0,0,0);

    if (mouseOver && rimMask > 0.0f)
        localColor = float4(1, 1, 1, 1);

    // no AA for now
    if (outerMask > 0.0f)
        localColor = color;

    ui.dstColor = lerp(ui.dstColor, float4(localColor.rgb, 1), localColor.a * (1.0f - ui.dstColor.a));

    float2 delta = round(ui.mouseInput.xy + 0.5f - ui.pxPos);

    return mouseOver && delta.x == 0.0f && delta.y == 0.0f;
}

public bool s2h_radioButton(inout ContextGather ui, bool checked)
{
    float4 color = ui.buttonColor;

    float2 pxLocal = float2(ui.pxPos - ui.pxCursor - 0.5f) / float(ui.scale) - float2(3.5f, 3.5f);
    float dist = length(pxLocal);

    float rimMask = saturate(5.0f - dist);
    float outerMask = saturate(4.0f - dist);
    float innerMask = saturate(2.5f - dist);

    bool mouseOver = length(float2(ui.mouseInput.xy - ui.pxCursor) / float(ui.scale) - float2(3.5f, 3.5f)) < 4.0f;

    if (mouseOver && rimMask > 0.0f)
        ui.dstColor = float4(1, 1, 1 ,1);

    // no AA for now
    if (outerMask > 0.0f)
        ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a);
    if (checked && innerMask > 0.0f)
        ui.dstColor = lerp(ui.dstColor, float4(ui.textColor.rgb, 1), ui.textColor.a);

    ui.pxCursor.x += s2h_fontSize() * ui.scale;

    float2 delta = round(ui.mouseInput.xy + 0.5f - ui.pxPos);

    return mouseOver && delta.x == 0.0f && delta.y == 0.0f;
}

public bool s2h_checkBox(inout ContextGather ui, bool checked)
{
    float4 color = ui.buttonColor;

    float2 pxLocal = float2(ui.pxPos - ui.pxCursor - 0.5f) / float(ui.scale) - float2(3.5f, 3.5f);
    float dist = max(abs(pxLocal.x), abs(pxLocal.y));

    float rimMask = saturate(5.0f - dist);
    float outerMask = saturate(4.0f - dist);
    float innerMask = saturate(2.5f - dist);

    bool mouseOver = length(float2(ui.mouseInput.xy - ui.pxCursor) / float(ui.scale) - float2(3.5f, 3.5f)) < 4.0f;

    if (mouseOver && rimMask > 0.0f)
        ui.dstColor = float4(1, 1, 1 ,1);

    // no AA for now
    if (outerMask > 0.0f)
        ui.dstColor = lerp(ui.dstColor, float4(color.rgb, 1), color.a);
    if (checked && innerMask > 0.0f)
        ui.dstColor = lerp(ui.dstColor, float4(ui.textColor.rgb, 1), ui.textColor.a);

    ui.pxCursor.x += s2h_fontSize() * ui.scale;

    float2 delta = round(ui.mouseInput.xy + 0.5f - ui.pxPos);

    return mouseOver && delta.x == 0.0f && delta.y == 0.0f;
}


public void s2h_progress(inout ContextGather ui, uint widthInCharacters, float fraction)
{
    float4 color = ui.buttonColor;
    float4 outerAABB = float4(ui.pxCursor, ui.pxCursor + float2(float(widthInCharacters) * s2h_fontSize(), s2h_fontSize() - 2.0f) * ui.scale);
    outerAABB += 0.5f;

    // shrink
    float4 innerAABB = outerAABB + float4(1, 1, -1, -1) * ui.scale;

    innerAABB.z = lerp(innerAABB.x, innerAABB.z, saturate(fraction));

    float sliderDist = s2h_computeDistToBox(ui, ui.pxPos, outerAABB);
    float innerDist = s2h_computeDistToBox(ui, ui.pxPos, innerAABB);

    float4 localColor = float4(0,0,0,0);

    // no AA for now
    if (sliderDist <= 0.0f)
        localColor = color;

    if (innerDist <= 0.0f)
        localColor = lerp(localColor, float4(ui.textColor.rgb, 1), ui.textColor.a);

    ui.pxCursor.x += float(widthInCharacters) * s2h_fontSize() * ui.scale;

    ui.dstColor = lerp(ui.dstColor, float4(localColor.rgb, 1), localColor.a * (1.0f - ui.dstColor.a));
}


public void s2h_sliderFloat(inout ContextGather ui, uint widthInCharacters, inout float value, float minValue, float maxValue)
{
    float4 color = ui.buttonColor;
    float4 outerAABB = float4(ui.pxCursor, ui.pxCursor + float2(float(widthInCharacters) * s2h_fontSize(), s2h_fontSize() - 2.0f) * ui.scale);
    outerAABB += 0.5f;

    float halfChar = s2h_fontSize() / 2.0f;

    // shrink
    float4 innerAABB = outerAABB + float4(1, 1, -1, -1) * ui.scale;

    float sliderDist = s2h_computeDistToBox(ui, ui.pxPos, outerAABB);

    // todo: active button should be made for all UI interactive buttons (checkbox, radio, button)
    float2 currentMouse = (ui.s2h_State.x == 0 && ui.s2h_State.y == 0) ? ui.mouseInput.xy : float2(ui.s2h_State.xy);

    bool mouseOver = s2h_computeDistToBox(ui, currentMouse, outerAABB) <= 0.0f;

    float3 knobColor = ui.textColor.rgb;

    // mouse over and left mouse button pressed
    if (mouseOver && ui.mouseInput.z != 0.0f) {
        float newFraction = saturate((ui.mouseInput.xy.x - innerAABB.x) / (innerAABB.z - innerAABB.x));
        value = lerp(minValue, maxValue, newFraction);

        knobColor = float3(1, 1, 1);

        // todo: active button should be made for all UI interactive buttons (checkbox, radio, button)
        if (ui.s2h_State.x == 0 && ui.s2h_State.y == 0)
            ui.s2h_State.xy = int2(ui.mouseInput.xy);
    }

    float fraction = saturate((value - minValue) / (maxValue - minValue));

    float knobRange = (float(widthInCharacters) - 1.0f) * s2h_fontSize() * ui.scale;
    float2 knobPos = ui.pxCursor + float2(halfChar * ui.scale, 0.0f) + float2(fraction * knobRange, 3.0f * ui.scale);
    float2 knobSize = float2(s2h_fontSize() - 4.0f, s2h_fontSize() - 4.0f) * 0.5f * ui.scale;
    float4 knobAABB = float4(knobPos - knobSize, knobPos + knobSize);
    knobAABB += 0.5f;

    float knobDist = s2h_computeDistToBox(ui, ui.pxPos, knobAABB);

    float4 localColor = float4(0,0,0,0);

    if (mouseOver && sliderDist <= 2.0)
        localColor = float4(1, 1, 1 ,1);

    // no AA for now
    if (sliderDist <= 0.0f)
        localColor = color;

//  if (innerDist <= 0.0f)
//      localColor = lerp(localColor, float4(ui.textColor.rgb, 1), ui.textColor.a);

    if (knobDist <= 0.0f)
        localColor = lerp(localColor, float4(knobColor, 1), ui.textColor.a);

    ui.pxCursor.x += float(widthInCharacters) * s2h_fontSize() * ui.scale;

    ui.dstColor = lerp(ui.dstColor, float4(localColor.rgb, 1), localColor.a * (1.0f - ui.dstColor.a));
}

public void s2h_sliderRGB(inout ContextGather ui, uint widthInCharacters, inout float3 value)
{
    float r = 3.0f * s2h_fontSize() * 0.5f * ui.scale - 1.0f;
    float4 backup = ui.buttonColor;

    float2 initialPos = ui.pxCursor;
    float2 pos = initialPos + float2(3.0f * s2h_fontSize() * ui.scale, 0.0f);

    ui.pxCursor.x = pos.x;
    ui.buttonColor = float4(1,0.1f,0.1f,1);
    s2h_sliderFloat(ui, widthInCharacters - 3u, value.r, 0.0f, 1.0f);
    s2h_printLF(ui);

    ui.pxCursor.x = pos.x;
    ui.buttonColor = float4(0,1,0,1);
    s2h_sliderFloat(ui, widthInCharacters - 3u, value.g, 0.0f, 1.0f);
    s2h_printLF(ui);

    ui.pxCursor.x = pos.x;
    ui.buttonColor = float4(0.2f,0.2f,1,1);
    s2h_sliderFloat(ui, widthInCharacters - 3u, value.b, 0.0f, 1.0f);
    s2h_printLF(ui);

    // todo: don't abuse circle drawing for disk drawing
    // todo: check if sRGB blending is right, it looks wrong with white
    s2h_drawDisc(ui, initialPos + r, r, float4(value, 1));

    ui.pxCursor = initialPos + float2(float(widthInCharacters) * s2h_fontSize() * ui.scale, 0.0f);

    ui.buttonColor = backup;
}

public void s2h_sliderRGBA(inout ContextGather ui, uint widthInCharacters, inout float4 value)
{
    float r = 3.0f * s2h_fontSize() * 0.5f * ui.scale - 1.0f;
    float4 backup = ui.buttonColor;

    float2 initialPos = ui.pxCursor;
    float2 pos = initialPos + float2(3.0f * s2h_fontSize() * ui.scale, 0.0f);

    ui.pxCursor.x = pos.x;
    ui.buttonColor = float4(1,0.1f,0.1f,1);
    s2h_sliderFloat(ui, widthInCharacters - 3u, value.r, 0.0f, 1.0f);
    s2h_printLF(ui);

    ui.pxCursor.x = pos.x;
    ui.buttonColor = float4(0,1,0,1);
    s2h_sliderFloat(ui, widthInCharacters - 3u, value.g, 0.0f, 1.0f);
    s2h_printLF(ui);

    ui.pxCursor.x = pos.x;
    ui.buttonColor = float4(0.2f,0.2f,1,1);
    s2h_sliderFloat(ui, widthInCharacters - 3u, value.b, 0.0f, 1.0f);
    s2h_printLF(ui);

    ui.pxCursor.x = pos.x;
    ui.buttonColor = float4(0.5f,0.5f,0.5f,1);
    s2h_sliderFloat(ui, widthInCharacters - 3u, value.a, 0.0f, 1.0f);
    s2h_printLF(ui);

    // todo: don't abuse circle drawing for disk drawing
    // todo: check if sRGB blending is right, it looks wrong with white
    s2h_drawDisc(ui, initialPos + r, r, value);

    ui.pxCursor = initialPos + float2(float(widthInCharacters) * s2h_fontSize() * ui.scale, 0.0f);

    ui.buttonColor = backup;
}

public void s2h_tableInt<TSource : IIntTable>(inout ContextGather ui, uint column, float4 backgroundColor, int2 sizeInCharacters, bool goRightNotDown, TSource source)
{
    float pxCharSize = s2h_fontSize() * ui.scale;
    float2 backup = ui.pxCursor;
    float2 localPos = ui.pxPos - ui.pxCursor;
    float2 pxSize = float2(sizeInCharacters) * pxCharSize;

    if (localPos.x >= 0.0f && localPos.y >= 0.0f && localPos.x < pxSize.x && localPos.y < pxSize.y) {
        ui.dstColor = lerp(ui.dstColor, float4(backgroundColor.rgb, 1), backgroundColor.a * (1.0f - ui.dstColor.a));

        int row = int(localPos.y / pxCharSize);

        ui.pxCursor.y = backup.y + floor(localPos.y / pxCharSize) * pxCharSize;

        int value = 0;
        if (source.lookup(column, row, value)) {
            s2h_printInt(ui, value);
        }
    }
    if (goRightNotDown) {
        ui.pxCursor.x = backup.x + pxSize.x;
    } else {
        s2h_printLF(ui);
        ui.pxCursor.y = backup.y + pxSize.y;
    }
}

public void s2h_tableFloat<TSource : IFloatTable>(inout ContextGather ui, uint column, float4 backgroundColor, int2 sizeInCharacters, bool goRightNotDown, TSource source)
{
    float pxCharSize = s2h_fontSize() * ui.scale;
    float2 backup = ui.pxCursor;
    float2 localPos = ui.pxPos - ui.pxCursor;
    float2 pxSize = float2(sizeInCharacters) * pxCharSize;

    if (localPos.x >= 0.0f && localPos.y >= 0.0f && localPos.x < pxSize.x && localPos.y < pxSize.y) {
        ui.dstColor = lerp(ui.dstColor, float4(backgroundColor.rgb, 1), backgroundColor.a * (1.0f - ui.dstColor.a));

        int row = int(localPos.y / pxCharSize);

        ui.pxCursor.y = backup.y + floor(localPos.y / pxCharSize) * pxCharSize;

        float value = 0.0f;
        if (source.lookup(column, row, value)) {
            s2h_printFloat(ui, value);
        }
    }
    if (goRightNotDown) {
        ui.pxCursor.x = backup.x + pxSize.x;
    } else {
        s2h_printLF(ui);
        ui.pxCursor.y = backup.y + pxSize.y;
    }
}

public void s2h_function<TSource : IFunctionPlot>(inout ContextGather ui, uint functionId, float4 backgroundColor, int2 sizeInCharacters, float2 rangeX, float2 rangeY, TSource source)
{
    float scaledFontSize = s2h_fontSize() * ui.scale;

    float2 backup = ui.pxCursor;
    float2 localPos = ui.pxPos - ui.pxCursor;
    float2 pxSize = float2(sizeInCharacters) * scaledFontSize;

    if (localPos.x >= 0.0f && localPos.y >= 0.0f && localPos.x < pxSize.x && localPos.y < pxSize.y) {
        // rangeX.x .. rangeX.y
        float x = lerp(rangeX.x, rangeX.y, localPos.x / pxSize.x);

        ui.dstColor = lerp(ui.dstColor, float4(backgroundColor.rgb, 1), backgroundColor.a * (1.0f - ui.dstColor.a));

        int row = int(localPos.y / scaledFontSize);

        ui.pxCursor.y = backup.y + floor(localPos.y / scaledFontSize) * scaledFontSize;

        // rangeY.x .. rangeY.y
        float y = source.evaluate(functionId, x);
        // 0 .. pxSize.y
        float pxY = (1.0f - (y - rangeY.x) / (rangeY.y - rangeY.x)) * pxSize.y;

        if (pxY < localPos.y)
            ui.dstColor = lerp(ui.dstColor, float4(ui.textColor.rgb, 1), ui.textColor.a);
    }

    s2h_printLF(ui);
    ui.pxCursor.y = backup.y + pxSize.y;
}

public float3 s2h_accurateLinearToSRGB(float3 linearCol)
{
    float3 sRGBLo = linearCol * 12.92;
    float3 sRGBHi = (pow(abs(linearCol), float3(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4)) * 1.055) - 0.055;
    public float3 sRGB;
    sRGB.r = linearCol.r <= 0.0031308 ? sRGBLo.r : sRGBHi.r;
    sRGB.g = linearCol.g <= 0.0031308 ? sRGBLo.g : sRGBHi.g;
    sRGB.b = linearCol.b <= 0.0031308 ? sRGBLo.b : sRGBHi.b;
    return sRGB;
}

public float3 s2h_accurateSRGBToLinear(float3 sRGBCol)
{
    float3 linearRGBLo = sRGBCol / 12.92;
    float3 linearRGBHi = pow((sRGBCol + 0.055) / 1.055, float3(2.4, 2.4, 2.4));
    public float3 linearRGB;
    linearRGB.r = sRGBCol.r <= 0.04045 ? linearRGBLo.r : linearRGBHi.r;
    linearRGB.g = sRGBCol.g <= 0.04045 ? linearRGBLo.g : linearRGBHi.g;
    linearRGB.b = sRGBCol.b <= 0.04045 ? linearRGBLo.b : linearRGBHi.b;
    return linearRGB;
}

public float3 s2h_indexToColor(uint index)
{
    uint a = (index >> 0u) & 1u;
    uint d = (index >> 1u) & 1u;
    uint g = (index >> 2u) & 1u;

    uint b = (index >> 3u) & 1u;
    uint e = (index >> 4u) & 1u;
    uint h = (index >> 5u) & 1u;

    uint c = (index >> 6u) & 1u;
    uint f = (index >> 7u) & 1u;
    uint i = (index >> 8u) & 1u;

    return float3(a * 4u + b * 2u + c, d * 4u + e * 2u + f, g * 4u + h * 2u + i) / 7.0f;
}

public float3 s2h_colorRampRGB(float value)
{
    return float3(
        saturate(1.0f - abs(value) * 2.0f),
        saturate(1.0f - abs(value - 0.5f) * 2.0f),
        saturate(1.0f - abs(value - 1.0f) * 2.0f));
}


// Premultiplied linear RGBA; exposure and output conversion remain caller-owned.
public float4 composite(float4 linearBackground, ContextGather overlay)
{
    return linearBackground * (1.0f - overlay.dstColor.a) + overlay.dstColor;
}

} // namespace Metallic.ShaderDebug
