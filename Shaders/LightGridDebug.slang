#include "ClusterLightGridCommon.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<ClusterLightGridParams> gLightGridDebugParams;

[[vk::binding(1, 0)]]
StructuredBuffer<ClusterLightGridCell> gLightGridDebugCells;

[[vk::binding(2, 0)]]
RWTexture2D<float4> gLightGridDebugOutput;

struct LightGridDebugPush
{
    uint width;
    uint height;
    uint mode;
    uint sliceIndex;
    float viewDepth;
    float heatmapMaxLights;
    uint flags;
    uint reserved;
};

static const uint kLightGridDebugShowGrid = 1u;
static const uint kLightGridDebugShowLegend = 2u;
static const uint kLightGridDebugIncludeGlobals = 4u;
static const uint kLightGridDebugShowCounts = 8u;

float3 lightGridDebugPalette(float value)
{
    float scaled = saturate(value) * 4.0;
    if (scaled < 1.0) {
        return lerp(float3(0.0, 0.15, 0.55), float3(0.0, 0.85, 1.0), scaled);
    }
    if (scaled < 2.0) {
        return lerp(float3(0.0, 0.85, 1.0), float3(0.15, 1.0, 0.25), scaled - 1.0);
    }
    if (scaled < 3.0) {
        return lerp(float3(0.15, 1.0, 0.25), float3(1.0, 0.9, 0.05), scaled - 2.0);
    }
    return lerp(float3(1.0, 0.9, 0.05), float3(1.0, 0.08, 0.02), scaled - 3.0);
}

uint lightGridDebugDigitMask(uint digit)
{
    // Row-major 3x5 glyphs, with the leftmost pixel in the low bit of each row.
    switch (digit) {
    case 0u: return 0x7b6fu;
    case 1u: return 0x749au;
    case 2u: return 0x73e7u;
    case 3u: return 0x79e7u;
    case 4u: return 0x49edu;
    case 5u: return 0x79cfu;
    case 6u: return 0x7bcfu;
    case 7u: return 0x4927u;
    case 8u: return 0x7befu;
    default: return 0x79efu;
    }
}

float3 lightGridDebugNumber(
    float3 color, uint2 pixel, uint2 center, uint value, uint maxWidth, uint scale)
{
    uint digits = 1u;
    uint divisor = 1u;
    for (uint remaining = value; remaining >= 10u; remaining /= 10u) {
        ++digits;
        divisor *= 10u;
    }
    uint textWidth = (digits * 4u - 1u) * scale;
    if (textWidth + 2u * scale > maxWidth) {
        return color;
    }
    int2 local = int2(pixel) - int2(center) + int2(textWidth / 2u, (5u * scale) / 2u);
    if (any(local < -int(scale)) || local.x >= int(textWidth + scale) ||
        local.y >= int(6u * scale)) {
        return color;
    }
    color = float3(0.025);
    if (any(local < 0) || local.x >= int(textWidth) || local.y >= int(5u * scale)) {
        return color;
    }
    uint2 glyphPixel = uint2(local) / scale;
    uint digitIndex = glyphPixel.x / 4u;
    for (uint i = 0u; i < digitIndex; ++i) {
        divisor /= 10u;
    }
    uint digit = (value / divisor) % 10u;
    uint column = glyphPixel.x % 4u;
    if (column < 3u && ((lightGridDebugDigitMask(digit) >> (glyphPixel.y * 3u + column)) & 1u) != 0u) {
        color = float3(1.0);
    }
    return color;
}

float3 lightGridDebugLegend(float3 color, uint2 pixel, LightGridDebugPush push)
{
    // Leave small render targets unobscured. Black means no coverage, the ramp
    // shows nonzero coverage up to the selected maximum, and magenta is overflow.
    if (push.width < 96u || push.height < 48u || pixel.y < push.height - 28u) {
        return color;
    }
    color = float3(0.025);
    uint barBegin = 26u;
    uint barEnd = push.width - 36u;
    bool inBar = pixel.y >= push.height - 23u && pixel.y < push.height - 11u;
    if (inBar) {
        if (pixel.x >= 8u && pixel.x < 20u) {
            color = float3(0.0);
        } else if (pixel.x >= barBegin && pixel.x < barEnd) {
            color = lightGridDebugPalette(float(pixel.x - barBegin) / float(barEnd - barBegin - 1u));
        } else if (pixel.x >= push.width - 28u && pixel.x < push.width - 8u) {
            color = float3(1.0, 0.0, 1.0);
        }
    }
    color = lightGridDebugNumber(color, pixel, uint2(14u, push.height - 6u), 0u, 16u, 1u);
    color = lightGridDebugNumber(color, pixel, uint2(barEnd - 10u, push.height - 6u),
        uint(clamp(round(push.heatmapMaxLights), 1.0, 1000000.0)), barEnd - barBegin, 1u);
    return color;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lightGridDebugMain(uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform LightGridDebugPush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    ClusterLightGridParams params = gLightGridDebugParams[0];
    if (any(params.grid.xyz == 0u) || params.grid.w == 0u || any(pixel >= params.viewport.xy)) {
        gLightGridDebugOutput[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    uint2 tile = pixel / params.grid.w;
    uint tileIndex = tile.y * params.grid.x + tile.x;
    uint planeCellCount = params.grid.x * params.grid.y;
    uint localCount = 0u;
    bool validDepth = true;
    if (push.mode == 0u) {
        // Peak is the worst per-cell list length along this screen tile. It is
        // intentionally not a sum, which would count a light in several slices.
        for (uint z = 0u; z < params.grid.z; ++z) {
            localCount = max(localCount, gLightGridDebugCells[z * planeCellCount + tileIndex].totalCount);
        }
    } else if (push.mode == 1u) {
        uint z = min(push.sliceIndex, params.grid.z - 1u);
        localCount = gLightGridDebugCells[z * planeCellCount + tileIndex].totalCount;
    } else {
        uint cellIndex;
        validDepth = clusterLightGridCellIndex(params, pixel, push.viewDepth, cellIndex);
        if (validDepth) {
            localCount = gLightGridDebugCells[cellIndex].totalCount;
        }
    }

    // Visualize the pre-truncation count. Global directionals and unbounded
    // locals are outside cell storage and never cause a local-list overflow.
    uint count = localCount;
    if (validDepth && (push.flags & kLightGridDebugIncludeGlobals) != 0u) {
        count += clusterLightGridGlobalLightCount(params);
    }
    bool overflow = localCount > params.viewport.z;
    float3 color = float3(0.0);
    if (validDepth && count != 0u) {
        color = overflow ? float3(1.0, 0.0, 1.0)
            : lightGridDebugPalette(float(count) / max(push.heatmapMaxLights, 1.0));
    }
    if ((push.flags & kLightGridDebugShowGrid) != 0u &&
        (pixel.x % params.grid.w == 0u || pixel.y % params.grid.w == 0u)) {
        color *= 0.25;
    }
    if (validDepth && (push.flags & kLightGridDebugShowCounts) != 0u) {
        uint2 tileMin = tile * params.grid.w;
        uint2 tileMax = min(tileMin + params.grid.w, params.viewport.xy);
        uint scale = params.grid.w >= 32u ? 2u : 1u;
        uint2 textCenter = (tileMin + tileMax) / 2u;
        bool overlapsLegend = (push.flags & kLightGridDebugShowLegend) != 0u &&
            push.width >= 96u && push.height >= 48u &&
            textCenter.y + 4u * scale >= push.height - 28u;
        if (tileMax.y - tileMin.y >= 7u * scale && !overlapsLegend) {
            color = lightGridDebugNumber(color, pixel, (tileMin + tileMax) / 2u,
                count, tileMax.x - tileMin.x, scale);
        }
    }
    if ((push.flags & kLightGridDebugShowLegend) != 0u) {
        color = lightGridDebugLegend(color, pixel, push);
    }
    gLightGridDebugOutput[pixel] = float4(color, 1.0);
}
