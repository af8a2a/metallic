// Shared HZB projection and occlusion-test helpers.
//
// All functions are pure (no global state, no binding references).
// Callers supply explicit VP matrices, world matrices, screen dimensions,
// and the HZB texture array so this file compiles into any shader stage.
//
// Conventions:
//   - Reversed-Z depth: 1.0 = near plane, 0.0 = far plane.
//   - Clip-space Y is +up; UV-space Y is +down (standard D3D/Vulkan).
//   - HZB mip 0 is a full-resolution copy of the depth buffer.
//   - "Visible" means the object's nearest clip depth (clipMax.z) is >=
//     the farthest (minimum) depth sample in the HZB footprint.

#include "../../Source/Rendering/hzb_constants.h"

static const float kHzbEpsilon   = 1.2e-7;
static const float kHzbDepthNudge = 2.0 / float(1 << 24);

// ── Geometry helpers ────────────────────────────────────────────────────────

uint hzbGetCullBits(float4 hPos) {
    uint bits = 0u;
    bits |= hPos.x < -hPos.w ? 1u : 0u;
    bits |= hPos.x >  hPos.w ? 2u : 0u;
    bits |= hPos.y < -hPos.w ? 4u : 0u;
    bits |= hPos.y >  hPos.w ? 8u : 0u;
    bits |= hPos.z <  0.0    ? 16u : 0u;
    bits |= hPos.z >  hPos.w ? 32u : 0u;
    bits |= hPos.w <= 0.0    ? 64u : 0u;
    return bits;
}

float3 hzbBoxCorner(float3 center, float radius, uint corner) {
    return center + float3(
        (corner & 1u) != 0u ? radius : -radius,
        (corner & 2u) != 0u ? radius : -radius,
        (corner & 4u) != 0u ? radius : -radius);
}

// Projects a sphere (approximated as an axis-aligned bounding box) into
// clip space and returns the min/max NDC extents.
//
// Returns true if the sphere intersects the frustum (commonBits == 0),
// or if frustumCull is false (caller disables frustum rejection).
// clipValid is false when any corner straddles the near plane (w < epsilon).
bool hzbProjectSphere(float3 center, float radius,
                      float4x4 worldMatrix, float4x4 vp,
                      bool frustumCull,
                      out float4 clipMin, out float4 clipMax, out bool clipValid) {
    if (radius <= 0.0) {
        clipMin    = float4(0.0, 0.0, 0.0, 0.0);
        clipMax    = float4(0.0, 0.0, 0.0, 0.0);
        clipValid  = false;
        return true;
    }

    float4 hPos = mul(vp, mul(worldMatrix, float4(hzbBoxCorner(center, radius, 0u), 1.0)));
    uint commonBits = hzbGetCullBits(hPos);
    bool valid = hPos.w >= kHzbEpsilon;
    float4 clip = float4(hPos.xyz / max(hPos.w, kHzbEpsilon), hPos.w);
    clipMin   = clip;
    clipMax   = clip;
    clipValid = valid;

    [unroll]
    for (uint corner = 1u; corner < 8u; ++corner) {
        hPos = mul(vp, mul(worldMatrix, float4(hzbBoxCorner(center, radius, corner), 1.0)));
        commonBits &= hzbGetCullBits(hPos);
        valid = hPos.w >= kHzbEpsilon;
        clip  = float4(hPos.xyz / max(hPos.w, kHzbEpsilon), hPos.w);
        clipMin   = min(clipMin, clip);
        clipMax   = max(clipMax, clip);
        clipValid = clipValid && valid;
    }

    clipMin = float4(clamp(clipMin.xy, float2(-1.0), float2(1.0)), clipMin.zw);
    clipMax = float4(clamp(clipMax.xy, float2(-1.0), float2(1.0)), clipMax.zw);
    return !frustumCull || commonBits == 0u;
}

// ── Mip selection ────────────────────────────────────────────────────────────

// Dimension of HZB mip `mip` given the base (mip-0) dimension.
uint hzbMipDimension(uint baseDimension, uint mip) {
    uint d = max(baseDimension, 1u);
    for (uint level = 0u; level < mip; ++level) {
        d = max(1u, (d + 1u) / 2u);
    }
    return d;
}

// Selects the coarsest mip whose texels still cover the given full-resolution
// pixel rect within a 4-texel footprint. Mirrors Nyx MipLevelForRect with
// HZB_TARGET_FOOTPRINT_PIXELS = 4.
int hzbMipLevelForRect(int4 pixelRect) {
    int2 rectSize    = max(pixelRect.zw - pixelRect.xy, int2(0, 0));
    int2 dominantMip = int2(firstbithigh(rectSize.x), firstbithigh(rectSize.y));
    int  mipLevel    = max(max(dominantMip.x, dominantMip.y) - 1, 0);
    int2 shifted     = pixelRect.xy >> mipLevel;
    int2 shiftedMax  = pixelRect.zw >> mipLevel;
    mipLevel += any(shiftedMax - shifted > int2(3, 3)) ? 1 : 0;
    return mipLevel;
}

// ── Depth sampling ───────────────────────────────────────────────────────────

// Reads a 4×4 texel grid from `hzbTextures[mipIdx]` clamped to [rectMin, rectMax]
// and returns the minimum (farthest in reversed-Z) depth value.
float hzbSampleMinDepth(Texture2D<float> hzbTextures[HZB_MAX_LEVELS],
                        uint mipIdx,
                        int2 rectMin, int2 rectMax) {
    int4 sampleX = min(rectMin.x + int4(0, 1, 2, 3), int4(rectMax.x));
    int4 sampleY = min(rectMin.y + int4(0, 1, 2, 3), int4(rectMax.y));

    float4 row0 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.x, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.x, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.x, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.x, 0)));
    float4 row1 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.y, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.y, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.y, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.y, 0)));
    float4 row2 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.z, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.z, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.z, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.z, 0)));
    float4 row3 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.w, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.w, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.w, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.w, 0)));

    float4 colMin = min(min(row0, row1), min(row2, row3));
    return min(min(colMin.x, colMin.y), min(colMin.z, colMin.w));
}

// ── Full occlusion test ──────────────────────────────────────────────────────

// Returns true if the projected bounds are visible (not occluded by the HZB).
// Returns true conservatively when occlusion is disabled, the pyramid is
// invalid, the object straddles the near plane, or its screen footprint is
// sub-pixel.
//
// enableOcclusion / hzbValid / hzbMipCount mirror the push-constant fields
// callers already carry; pass them through rather than re-reading params here.
bool hzbTestVisibility(float4 clipMin, float4 clipMax, bool clipValid,
                       bool enableOcclusion, bool hzbValid, uint hzbMipCount,
                       uint screenWidth, uint screenHeight,
                       uint hzbWidth, uint hzbHeight,
                       Texture2D<float> hzbTextures[HZB_MAX_LEVELS]) {
    if (!enableOcclusion || !hzbValid || hzbMipCount == 0u) {
        return true;
    }
    if (!clipValid) {
        return true;
    }

    // NDC → UV, flipping Y (clip +up → UV +down).
    float2 uvMin = float2(clipMin.x * 0.5 + 0.5, 0.5 - clipMax.y * 0.5);
    float2 uvMax = float2(clipMax.x * 0.5 + 0.5, 0.5 - clipMin.y * 0.5);
    uvMin = clamp(uvMin, float2(0.0), float2(1.0));
    uvMax = clamp(uvMax, float2(0.0), float2(1.0));

    float2 pixelSize = (uvMax - uvMin) * float2(float(screenWidth), float(screenHeight));
    if (max(pixelSize.x, pixelSize.y) < 1.0) {
        return true;
    }

    // Pixel rect: min rounds up (+0.5 bias), max rounds down (-0.5 bias).
    float4 screenSizeF = float4(float(screenWidth), float(screenHeight),
                                float(screenWidth), float(screenHeight));
    int4 pixelsRect = int4(float4(uvMin.x, uvMin.y, uvMax.x, uvMax.y) * screenSizeF
                           + float4(0.5, 0.5, -0.5, -0.5));
    pixelsRect = int4(min(pixelsRect.xy, pixelsRect.zw),
                      max(pixelsRect.xy, pixelsRect.zw));

    int  hzbMip  = min(hzbMipLevelForRect(pixelsRect), int(hzbMipCount) - 1);
    int4 mipRect = pixelsRect >> hzbMip;

    uint mipIdx = uint(hzbMip);
    int  mipW   = int(hzbMipDimension(hzbWidth,  mipIdx));
    int  mipH   = int(hzbMipDimension(hzbHeight, mipIdx));

    int2 maxCoord = int2(mipW - 1, mipH - 1);
    int2 rectMin  = clamp(mipRect.xy, int2(0, 0), maxCoord);
    int2 rectMax  = clamp(mipRect.zw, int2(0, 0), maxCoord);
    rectMin = min(rectMin, rectMax);

    float minDepth = hzbSampleMinDepth(hzbTextures, mipIdx, rectMin, rectMax);

    // Reversed-Z: object is visible if its nearest depth >= farthest HZB sample.
    return clipMax.z >= minDepth - kHzbDepthNudge;
}
