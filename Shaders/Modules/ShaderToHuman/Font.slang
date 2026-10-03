#language slang 2026
implementing ShaderToHuman;

// Copyright (c) 2024-2025 Electronic Arts Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Slang port of ShaderToHuman v14, revision 70ea227f3d504a19d9dd0539a832b68999426cca.
// See LICENSE.txt, NOTICE.txt and Upstream.json in this directory.

namespace Metallic.ShaderDebug
{
public static const int S2H_VERSION = 14;

internal static const uint kMiniFont[] = {
    0x00306c6cu, 0x30003860u, 0x18600000u, 0x00000006u,
    0x00786c6cu, 0x7cc66c60u, 0x30306630u, 0x0000000cu,
    0x00786cfeu, 0xc0cc38c0u, 0x60183c30u, 0x00000018u,
    0x0030006cu, 0x78187600u, 0x6018fffcu, 0x00fc0030u,
    0x003000feu, 0x0c30dc00u, 0x60183c30u, 0x00000060u,
    0x0000006cu, 0xf866cc00u, 0x30306630u, 0x300030c0u,
    0x0030006cu, 0x30c67600u, 0x18600000u, 0x30003080u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x60000000u,
    0x7c307878u, 0x1cfc38fcu, 0x78780000u, 0x18006078u,
    0xc670ccccu, 0x3cc060ccu, 0xcccc3030u, 0x300030ccu,
    0xce300c0cu, 0x6cf8c00cu, 0xcccc3030u, 0x60fc180cu,
    0xde303838u, 0xcc0cf818u, 0x787c0000u, 0xc0000c18u,
    0xf630600cu, 0xfe0ccc30u, 0xcc0c0000u, 0x60001830u,
    0xe630ccccu, 0x0ccccc30u, 0xcc183030u, 0x30fc3000u,
    0x7cfcfc78u, 0x1e787830u, 0x78703030u, 0x18006030u,
    0x00000000u, 0x00000000u, 0x00000060u, 0x00000000u,
    0x7c30fc3cu, 0xf8fefe3cu, 0xcc781ee6u, 0xf0c6c638u,
    0xc6786666u, 0x6c626266u, 0xcc300c66u, 0x60eee66cu,
    0xdecc66c0u, 0x666868c0u, 0xcc300c6cu, 0x60fef6c6u,
    0xdecc7cc0u, 0x667878c0u, 0xfc300c78u, 0x60fedec6u,
    0xdefc66c0u, 0x666868ceu, 0xcc30cc6cu, 0x62d6cec6u,
    0xc0cc6666u, 0x6c626066u, 0xcc30cc66u, 0x66c6c66cu,
    0x78ccfc3cu, 0xf8fef03eu, 0xcc7878e6u, 0xfec6c638u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0xfc78fc78u, 0xfcccccc6u, 0xc6ccfe78u, 0xc0781000u,
    0x66cc66ccu, 0xb4ccccc6u, 0xc6ccc660u, 0x60183800u,
    0x66cc66e0u, 0x30ccccc6u, 0x6ccc8c60u, 0x30186c00u,
    0x7ccc7c70u, 0x30ccccd6u, 0x38781860u, 0x1818c600u,
    0x60dc6c1cu, 0x30ccccfeu, 0x38303260u, 0x0c180000u,
    0x607866ccu, 0x30cc78eeu, 0x6c306660u, 0x06180000u,
    0xf01ce678u, 0x78fc30c6u, 0xc678fe78u, 0x02780000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x000000ffu,
    0x3000e000u, 0x1c003800u, 0xe0300ce0u, 0x70000000u,
    0x30006000u, 0x0c006c00u, 0x60000060u, 0x30000000u,
    0x18786078u, 0x0c786076u, 0x6c700c66u, 0x30ccf878u,
    0x000c7cccu, 0x7cccf0ccu, 0x76300c6cu, 0x30feccccu,
    0x007c66c0u, 0xccfc60ccu, 0x66300c78u, 0x30feccccu,
    0x00cc66ccu, 0xccc0607cu, 0x6630cc6cu, 0x30d6ccccu,
    0x0076dc78u, 0x7678f00cu, 0xe678cce6u, 0x78c6cc78u,
    0x00000000u, 0x000000f8u, 0x00007800u, 0x00000000u,
    0x00000000u, 0x10000000u, 0x0000001cu, 0x18e076ffu,
    0x00000000u, 0x30000000u, 0x00000030u, 0x1830dcffu,
    0xdc76dc7cu, 0x7cccccc6u, 0xc6ccfc30u, 0x183000ffu,
    0x66cc76c0u, 0x30ccccd6u, 0x6ccc98e0u, 0x001c00ffu,
    0x66cc6678u, 0x30ccccfeu, 0x38cc3030u, 0x183000ffu,
    0x7c7c600cu, 0x34cc78feu, 0x6c7c6430u, 0x183000ffu,
    0x600cf0f8u, 0x1876306cu, 0xc60cfc1cu, 0x18e000ffu,
    0xf01e0000u, 0x00000000u, 0x00f80000u, 0x000000ffu
};

public float s2h_fontSize()
{
    return 8.0f;
}

// don't use directly
// can be used for scatter and gather
// @param ascii 32..127 are valid characters
// @param pxPos int2(0..s2h_fontSize()-1, 0..s2h_fontSize-1)
// @return true if there should be a pixel, false if not or outside the valid range
public bool s2h_fontLookup(uint ascii, int2 pxPos)
{
    if (uint(pxPos.x) >= 8u || uint(pxPos.y) >= 8u)
        return false;

    if (ascii <= 32u || ascii > 127u)
        return false;

    // 0..16*6-1
    uint chr = ascii - 32u;
    // uint2(0..127, 0..47)
    uint2 chrPos = uint2(chr % 16u, chr / 16u);
    uint2 pixel = uint2(chrPos.x * 8u + uint(pxPos.x), chrPos.y * 8u + uint(pxPos.y));
    uint dwordId = pixel.x / 32u + (pixel.y * 4u);
    // 0..31
    uint bitId  = uint(pixel.x) & 0x1fu;

    // 0..ff
    uint dwordValue = kMiniFont[dwordId];

    return ((dwordValue >> (31u - bitId)) & 1u) != 0u;
}

public static const uint _A = 65u;
public static const uint _B = 66u;
public static const uint _C = 67u;
public static const uint _D = 68u;
public static const uint _E = 69u;
public static const uint _F = 70u;
public static const uint _G = 71u;
public static const uint _H = 72u;
public static const uint _I = 73u;
public static const uint _J = 74u;
public static const uint _K = 75u;
public static const uint _L = 76u;
public static const uint _M = 77u;
public static const uint _N = 78u;
public static const uint _O = 79u;
public static const uint _P = 80u;
public static const uint _Q = 81u;
public static const uint _R = 82u;
public static const uint _S = 83u;
public static const uint _T = 84u;
public static const uint _U = 85u;
public static const uint _V = 86u;
public static const uint _W = 87u;
public static const uint _X = 88u;
public static const uint _Y = 89u;
public static const uint _Z = 90u;

public static const uint _a = (_A + 32u);
public static const uint _b = (_B + 32u);
public static const uint _c = (_C + 32u);
public static const uint _d = (_D + 32u);
public static const uint _e = (_E + 32u);
public static const uint _f = (_F + 32u);
public static const uint _g = (_G + 32u);
public static const uint _h = (_H + 32u);
public static const uint _i = (_I + 32u);
public static const uint _j = (_J + 32u);
public static const uint _k = (_K + 32u);
public static const uint _l = (_L + 32u);
public static const uint _m = (_M + 32u);
public static const uint _n = (_N + 32u);
public static const uint _o = (_O + 32u);
public static const uint _p = (_P + 32u);
public static const uint _q = (_Q + 32u);
public static const uint _r = (_R + 32u);
public static const uint _s = (_S + 32u);
public static const uint _t = (_T + 32u);
public static const uint _u = (_U + 32u);
public static const uint _v = (_V + 32u);
public static const uint _w = (_W + 32u);
public static const uint _x = (_X + 32u);
public static const uint _y = (_Y + 32u);
public static const uint _z = (_Z + 32u);

public static const uint _SINGLEQUOTE = 39u;   // '
public static const uint _UNDERSCORE = 95u;    // _
public static const uint _MINUS = 45u;         // -
public static const uint _PLUS = 43u;          // +
public static const uint _ASTERISK = 42u;      // *
public static const uint _PERIOD = 46u;        // .
public static const uint _COLON = 58u;         // :
public static const uint _COMMA = 44u;         // ,
public static const uint _SPACE = 32u;         //
public static const uint _LESS = 60u;          // <
public static const uint _EQUAL = 61u;         // =
public static const uint _GREATER = 62u;       // >
public static const uint _SLASH = 47u;         // /
public static const uint _BACKSLASH = 92u;     //
public static const uint _0 = 48u;
public static const uint _1 = 49u;
public static const uint _2 = 50u;
public static const uint _3 = 51u;
public static const uint _4 = 52u;
public static const uint _5 = 53u;
public static const uint _6 = 54u;
public static const uint _7 = 55u;
public static const uint _8 = 56u;
public static const uint _9 = 57u;
public static const int _S2H_VERSION = S2H_VERSION;

} // namespace Metallic.ShaderDebug
