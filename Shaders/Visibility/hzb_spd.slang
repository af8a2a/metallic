// Single Pass Downsampler for HZB generation.
// Based on AMD FidelityFX SPD 2.0, adapted for Metallic's separate HZB textures.
// mipTextures[0] is a full-resolution depth copy. Mip 6 uses a separate globally coherent binding.

#include "../../Source/Rendering/hzb_constants.h"
#include "../../Source/Rendering/hzb_spd_constants.h"

struct HZBSpdUniforms {
    uint mipCount;
    uint numWorkGroups;
    uint2 workGroupOffset;
    uint2 sourceSize;
    uint2 _pad;
};

[[vk::push_constant]] ConstantBuffer<HZBSpdUniforms> uniforms;

[[vk::binding(HZB_SPD_SOURCE_TEXTURE_BINDING)]]
Texture2D<float> sourceTexture;

[[vk::binding(HZB_SPD_MIP_TEXTURE_BINDING_BASE)]]
RWTexture2D<float> mipTextures[HZB_MAX_LEVELS];

[[vk::binding(HZB_SPD_MIP6_TEXTURE_BINDING)]]
globallycoherent RWTexture2D<float> mip6Texture;

[[vk::binding(HZB_SPD_ATOMIC_COUNTER_BINDING)]]
globallycoherent RWByteAddressBuffer spdGlobalAtomic;

groupshared float spdIntermediate[16][16];
groupshared uint  spdCounter;

uint hzbLevelDimension(uint baseDimension, uint level) {
    uint dimension = max(baseDimension, 1u);
    for (uint currentLevel = 0u; currentLevel < level; ++currentLevel) {
        dimension = max(1u, (dimension + 1u) / 2u);
    }
    return dimension;
}

uint2 hzbLevelSize(uint level) {
    return uint2(hzbLevelDimension(uniforms.sourceSize.x, level),
                 hzbLevelDimension(uniforms.sourceSize.y, level));
}

float spdReduce4(float v0, float v1, float v2, float v3) {
    return min(min(v0, v1), min(v2, v3));
}

uint2 remap8x8(uint lane) {
    uint x = ((lane >> 1) & 1u) | (((lane >> 3) & 1u) << 1u) | (((lane >> 5) & 1u) << 2u);
    uint y = ((lane >> 0) & 1u) | (((lane >> 2) & 1u) << 1u) | (((lane >> 4) & 1u) << 2u);
    return uint2(x, y);
}

float spdReduceQuad(float v) {
    uint quad = WaveGetLaneIndex() & (~0x3u);
    float v0 = v;
    float v1 = WaveReadLaneAt(v, quad | 1u);
    float v2 = WaveReadLaneAt(v, quad | 2u);
    float v3 = WaveReadLaneAt(v, quad | 3u);
    return spdReduce4(v0, v1, v2, v3);
}

float spdLoadIntermediate(uint x, uint y) {
    return spdIntermediate[x][y];
}

void spdStoreIntermediate(uint x, uint y, float v) {
    spdIntermediate[x][y] = v;
}

float loadSourceClamped(uint2 coord) {
    uint2 maxCoord = max(uniforms.sourceSize, uint2(1u, 1u)) - uint2(1u, 1u);
    return sourceTexture.Load(int3(min(coord, maxCoord), 0));
}

float loadHzbClamped(uint mip, int2 coord) {
    if (mip >= HZB_MAX_LEVELS) {
        return 1.0;
    }

    uint2 mipSize = hzbLevelSize(mip);
    uint2 maxCoord = mipSize - uint2(1u, 1u);
    uint2 clampedCoord = min(uint2(max(coord, int2(0, 0))), maxCoord);
    if (mip == HZB_SPD_MIP6_LEVEL) {
        return mip6Texture[clampedCoord];
    }
    return mipTextures[mip][clampedCoord];
}

float spdLoadSource(int2 base) {
    uint2 p = uint2(max(base, int2(0, 0)));
    float v0 = loadSourceClamped(p + uint2(0u, 0u));
    float v1 = loadSourceClamped(p + uint2(0u, 1u));
    float v2 = loadSourceClamped(p + uint2(1u, 0u));
    float v3 = loadSourceClamped(p + uint2(1u, 1u));
    return spdReduce4(v0, v1, v2, v3);
}

void spdStore(int2 pix, float value, uint spdMip) {
    uint outputMip = spdMip + 1u;
    if (outputMip >= HZB_MAX_LEVELS || pix.x < 0 || pix.y < 0) {
        return;
    }

    uint2 mipSize = hzbLevelSize(outputMip);
    uint2 p = uint2(pix);
    if (p.x >= mipSize.x || p.y >= mipSize.y) {
        return;
    }

    if (outputMip == HZB_SPD_MIP6_LEVEL) {
        mip6Texture[p] = value;
    } else {
        mipTextures[outputMip][p] = value;
    }
}

float spdLoadMip5(int2 p) {
    return loadHzbClamped(HZB_SPD_MIP6_LEVEL, p);
}

void copySourceToMip0(uint2 workGroupID, uint localIndex) {
    uint localX = localIndex & 31u;
    uint localY = localIndex >> 5u;
    uint2 tileBase = workGroupID * SPD_TILE_SIZE;

    for (uint y = 0u; y < 8u; ++y) {
        for (uint x = 0u; x < 2u; ++x) {
            uint2 p = tileBase + uint2(localX * 2u + x, localY * 8u + y);
            if (p.x < uniforms.sourceSize.x && p.y < uniforms.sourceSize.y) {
                mipTextures[0][p] = sourceTexture.Load(int3(p, 0));
            }
        }
    }
}

void spdDownsampleMips_0_1(uint x, uint y, uint2 workGroupID, uint localIndex, uint mips) {
    float v[4];

    int2 tex = int2(workGroupID * 64u) + int2(x * 2u, y * 2u);
    int2 pix = int2(workGroupID * 32u) + int2(x, y);
    v[0] = spdLoadSource(tex);
    spdStore(pix, v[0], 0u);

    tex = int2(workGroupID * 64u) + int2(x * 2u + 32u, y * 2u);
    pix = int2(workGroupID * 32u) + int2(x + 16u, y);
    v[1] = spdLoadSource(tex);
    spdStore(pix, v[1], 0u);

    tex = int2(workGroupID * 64u) + int2(x * 2u, y * 2u + 32u);
    pix = int2(workGroupID * 32u) + int2(x, y + 16u);
    v[2] = spdLoadSource(tex);
    spdStore(pix, v[2], 0u);

    tex = int2(workGroupID * 64u) + int2(x * 2u + 32u, y * 2u + 32u);
    pix = int2(workGroupID * 32u) + int2(x + 16u, y + 16u);
    v[3] = spdLoadSource(tex);
    spdStore(pix, v[3], 0u);

    if (mips <= 1u) {
        return;
    }

    v[0] = spdReduceQuad(v[0]);
    v[1] = spdReduceQuad(v[1]);
    v[2] = spdReduceQuad(v[2]);
    v[3] = spdReduceQuad(v[3]);

    if ((localIndex % 4u) == 0u) {
        spdStore(int2(workGroupID * 16u) + int2(x / 2u, y / 2u), v[0], 1u);
        spdStoreIntermediate(x / 2u, y / 2u, v[0]);

        spdStore(int2(workGroupID * 16u) + int2(x / 2u + 8u, y / 2u), v[1], 1u);
        spdStoreIntermediate(x / 2u + 8u, y / 2u, v[1]);

        spdStore(int2(workGroupID * 16u) + int2(x / 2u, y / 2u + 8u), v[2], 1u);
        spdStoreIntermediate(x / 2u, y / 2u + 8u, v[2]);

        spdStore(int2(workGroupID * 16u) + int2(x / 2u + 8u, y / 2u + 8u), v[3], 1u);
        spdStoreIntermediate(x / 2u + 8u, y / 2u + 8u, v[3]);
    }
}

void spdDownsampleMip_2(uint x, uint y, uint2 workGroupID, uint localIndex, uint mip) {
    float v = spdLoadIntermediate(x, y);
    v = spdReduceQuad(v);
    if ((localIndex % 4u) == 0u) {
        spdStore(int2(workGroupID * 8u) + int2(x / 2u, y / 2u), v, mip);
        spdStoreIntermediate(x + (y / 2u) % 2u, y, v);
    }
}

void spdDownsampleMip_3(uint x, uint y, uint2 workGroupID, uint localIndex, uint mip) {
    if (localIndex < 64u) {
        float v = spdLoadIntermediate(x * 2u + y % 2u, y * 2u);
        v = spdReduceQuad(v);
        if ((localIndex % 4u) == 0u) {
            spdStore(int2(workGroupID * 4u) + int2(x / 2u, y / 2u), v, mip);
            spdStoreIntermediate(x * 2u + y / 2u, y * 2u, v);
        }
    }
}

void spdDownsampleMip_4(uint x, uint y, uint2 workGroupID, uint localIndex, uint mip) {
    if (localIndex < 16u) {
        float v = spdLoadIntermediate(x * 4u + y, y * 4u);
        v = spdReduceQuad(v);
        if ((localIndex % 4u) == 0u) {
            spdStore(int2(workGroupID * 2u) + int2(x / 2u, y / 2u), v, mip);
            spdStoreIntermediate(x / 2u + y, 0u, v);
        }
    }
}

void spdDownsampleMip_5(uint2 workGroupID, uint localIndex, uint mip) {
    if (localIndex < 4u) {
        float v = spdLoadIntermediate(localIndex, 0u);
        v = spdReduceQuad(v);
        if ((localIndex % 4u) == 0u) {
            spdStore(int2(workGroupID), v, mip);
        }
    }
}

void spdDownsampleNextFour(uint x, uint y, uint2 workGroupID, uint localIndex, uint baseMip, uint mips) {
    if (mips <= baseMip) {
        return;
    }
    GroupMemoryBarrierWithGroupSync();
    spdDownsampleMip_2(x, y, workGroupID, localIndex, baseMip);

    if (mips <= baseMip + 1u) {
        return;
    }
    GroupMemoryBarrierWithGroupSync();
    spdDownsampleMip_3(x, y, workGroupID, localIndex, baseMip + 1u);

    if (mips <= baseMip + 2u) {
        return;
    }
    GroupMemoryBarrierWithGroupSync();
    spdDownsampleMip_4(x, y, workGroupID, localIndex, baseMip + 2u);

    if (mips <= baseMip + 3u) {
        return;
    }
    GroupMemoryBarrierWithGroupSync();
    spdDownsampleMip_5(workGroupID, localIndex, baseMip + 3u);
}

bool spdExitWorkgroup(uint numWorkGroups, uint localIndex) {
    AllMemoryBarrierWithGroupSync();
    if (localIndex == 0u) {
        spdGlobalAtomic.InterlockedAdd(0u, 1u, spdCounter);
    }
    GroupMemoryBarrierWithGroupSync();
    return spdCounter != (numWorkGroups - 1u);
}

void spdDownsampleMips_6_7(uint x, uint y, uint mips) {
    int2 tex = int2(x * 4u + 0u, y * 4u + 0u);
    int2 pix = int2(x * 2u + 0u, y * 2u + 0u);
    float v0 = spdReduce4(spdLoadMip5(tex + int2(0, 0)), spdLoadMip5(tex + int2(0, 1)),
                          spdLoadMip5(tex + int2(1, 0)), spdLoadMip5(tex + int2(1, 1)));
    spdStore(pix, v0, 6u);

    tex = int2(x * 4u + 2u, y * 4u + 0u);
    pix = int2(x * 2u + 1u, y * 2u + 0u);
    float v1 = spdReduce4(spdLoadMip5(tex + int2(0, 0)), spdLoadMip5(tex + int2(0, 1)),
                          spdLoadMip5(tex + int2(1, 0)), spdLoadMip5(tex + int2(1, 1)));
    spdStore(pix, v1, 6u);

    tex = int2(x * 4u + 0u, y * 4u + 2u);
    pix = int2(x * 2u + 0u, y * 2u + 1u);
    float v2 = spdReduce4(spdLoadMip5(tex + int2(0, 0)), spdLoadMip5(tex + int2(0, 1)),
                          spdLoadMip5(tex + int2(1, 0)), spdLoadMip5(tex + int2(1, 1)));
    spdStore(pix, v2, 6u);

    tex = int2(x * 4u + 2u, y * 4u + 2u);
    pix = int2(x * 2u + 1u, y * 2u + 1u);
    float v3 = spdReduce4(spdLoadMip5(tex + int2(0, 0)), spdLoadMip5(tex + int2(0, 1)),
                          spdLoadMip5(tex + int2(1, 0)), spdLoadMip5(tex + int2(1, 1)));
    spdStore(pix, v3, 6u);

    if (mips <= 7u) {
        return;
    }

    float v = spdReduce4(v0, v1, v2, v3);
    spdStore(int2(x, y), v, 7u);
    spdStoreIntermediate(x, y, v);
}

[shader("compute")]
[numthreads(SPD_THREADGROUP_SIZE, 1, 1)]
void computeMain(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex) {
    uint2 workGroupID = groupID.xy + uniforms.workGroupOffset;

    copySourceToMip0(workGroupID, localIndex);

    if (uniforms.mipCount == 0u) {
        return;
    }

    uint2 subXY = remap8x8(localIndex % 64u);
    uint x = subXY.x + 8u * ((localIndex >> 6u) % 2u);
    uint y = subXY.y + 8u * (localIndex >> 7u);

    spdDownsampleMips_0_1(x, y, workGroupID, localIndex, uniforms.mipCount);
    spdDownsampleNextFour(x, y, workGroupID, localIndex, 2u, uniforms.mipCount);

    if (uniforms.mipCount <= 6u) {
        return;
    }

    if (spdExitWorkgroup(uniforms.numWorkGroups, localIndex)) {
        return;
    }

    if (localIndex == 0u) {
        spdGlobalAtomic.Store(0u, 0u);
    }

    spdDownsampleMips_6_7(x, y, uniforms.mipCount);
    spdDownsampleNextFour(x, y, uint2(0u, 0u), localIndex, 8u, uniforms.mipCount);
}
