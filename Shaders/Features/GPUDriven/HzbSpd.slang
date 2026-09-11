// Scalar depth specialization of the FidelityFX SPD algorithm (64x64 tiles,
// wave reduction, last-arriving group completes the tail), described at:
// https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/
// Unlike ordinary texture mips, this HZB keeps ceil-divided odd edges and mip 0.
// Uses full FP32 min/max, no filtering or half precision.

#ifndef HZB_SPD_WAVE_OPS
#define HZB_SPD_WAVE_OPS 1
#endif

struct HzbSpdPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint depthImage;
    uint hzbBuffer;
    uint counterBuffer;
    uint width;
    uint height;
    uint mipCount;
    uint reversedZ;
};

#if HZB_SPD_WAVE_OPS
// Only the 4x4 mip crosses wave boundaries.
groupshared float spdIntermediate[16];

// Vulkan does not guarantee any mapping between LocalInvocationIndex
// and subgroup lanes.
//
// Under full-subgroup execution this constructs a bijective [0, 255]
// workgroup index ordered by subgroup/lane. It is intentionally NOT
// assumed to equal SV_GroupIndex.
//
// SPD uses this ordering so groups of 4/16 consecutive logical indices
// correspond to consecutive lanes in the same subgroup.
// Requires subgroupSize >= 16.
uint spdInvocationIndex()
{
    uint wave = spirv_asm {
        OpCapability GroupNonUniform;
        result:$$uint = OpLoad builtin(SubgroupId:uint);
    };
    return wave * WaveGetLaneCount() + WaveGetLaneIndex();
}

uint2 spdMortonPixel(uint index)
{
    // Consecutive groups of 4/16 lanes cover aligned 2x2/4x4 pixels.
    uint2 bits = uint2(index, index >> 1u) & 0x55u;
    bits = (bits | (bits >> 1u)) & 0x33u;
    return (bits | (bits >> 2u)) & 0x0fu;
}
#else
groupshared float spdIntermediate[256];
#endif
groupshared uint spdCompletedGroups;

uint2 spdMipSize(HzbSpdPush push, uint mip)
{
    return max((uint2(push.width, push.height) + ((1u << mip) - 1u)) >> mip, uint2(1));
}

uint spdMipOffset(HzbSpdPush push, uint mip)
{
    uint offset = 0;
    for (uint level = 0; level < mip; ++level) {
        uint2 size = spdMipSize(push, level);
        offset += size.x * size.y;
    }
    return offset;
}

float spdReduce4(float a, float b, float c, float d, bool reversedZ)
{
    return reversedZ ? min(min(a, b), min(c, d)) : max(max(a, b), max(c, d));
}

#if HZB_SPD_WAVE_OPS
float spdWaveReduce4(float value, uint stride, bool reversedZ)
{
    uint lane = WaveGetLaneIndex();
    return spdReduce4(value, WaveReadLaneAt(value, lane ^ stride),
        WaveReadLaneAt(value, lane ^ (stride * 2u)),
        WaveReadLaneAt(value, lane ^ (stride * 3u)), reversedZ);
}
#endif

float spdLoad(HzbSpdPush push, uint2 pixel, uint sourceMip)
{
    uint2 size = spdMipSize(push, sourceMip);
    uint2 clamped = min(pixel, size - 1u);
    uint index = spdMipOffset(push, sourceMip) + clamped.y * size.x + clamped.x;
    if (sourceMip == 0u) {
        Texture2D<float> source = DescriptorHandle<Texture2D<float>>(
            uint2(push.imageShaderIndexBase + push.depthImage, 0));
        float depth = source.Load(int3(clamped, 0));
        // Non-finite depth must not prove occlusion. A clear-depth contribution
        // propagates through the pyramid and conservatively keeps candidates.
        depth = isfinite(depth) ? saturate(depth) : (push.reversedZ != 0u ? 0.0 : 1.0);
        if (all(pixel < size)) {
            RWStructuredBuffer<float> output = DescriptorHandle<RWStructuredBuffer<float>>(
                uint2(push.bufferShaderIndexBase + push.hzbBuffer, 0));
            output[index] = depth;
        }
        return depth;
    }
    // DescriptorHandle cannot carry HLSL globallycoherent. Atomic loads/stores
    // give mip 6 device-scope release/acquire visibility in native SPIR-V.
    RWStructuredBuffer<Atomic<uint>> output = DescriptorHandle<RWStructuredBuffer<Atomic<uint>>>(
        uint2(push.bufferShaderIndexBase + push.hzbBuffer, 0));
    return asfloat(output[index].load(MemoryOrder.Acquire));
}

void spdStore(HzbSpdPush push, uint2 pixel, float value, uint mip)
{
    if (mip >= push.mipCount) { return; }
    uint2 size = spdMipSize(push, mip);
    if (any(pixel >= size)) { return; }
    uint index = spdMipOffset(push, mip) + pixel.y * size.x + pixel.x;
    if (mip == 6u) {
        RWStructuredBuffer<Atomic<uint>> output = DescriptorHandle<RWStructuredBuffer<Atomic<uint>>>(
            uint2(push.bufferShaderIndexBase + push.hzbBuffer, 0));
        output[index].store(asuint(value), MemoryOrder.Release);
    } else {
        RWStructuredBuffer<float> output = DescriptorHandle<RWStructuredBuffer<float>>(
            uint2(push.bufferShaderIndexBase + push.hzbBuffer, 0));
        output[index] = value;
    }
}

void spdDownsampleTile(HzbSpdPush push, uint2 tile, uint lane, uint sourceMip)
{
#if HZB_SPD_WAVE_OPS
    uint2 xy = spdMortonPixel(lane);
#else
    uint2 xy = uint2(lane & 15u, lane >> 4u);
#endif
    float reduced[4];
    [unroll]
    for (uint quadrant = 0; quadrant < 4; ++quadrant) {
        uint2 pixel = tile * 32u + xy * 2u + uint2(quadrant & 1u, quadrant >> 1u);
        uint2 source = pixel * 2u;
        reduced[quadrant] = spdReduce4(
            spdLoad(push, source, sourceMip), spdLoad(push, source + uint2(1, 0), sourceMip),
            spdLoad(push, source + uint2(0, 1), sourceMip), spdLoad(push, source + uint2(1, 1), sourceMip),
            push.reversedZ != 0u);
        spdStore(push, pixel, reduced[quadrant], sourceMip + 1u);
    }
    float value = spdReduce4(reduced[0], reduced[1], reduced[2], reduced[3], push.reversedZ != 0u);
    spdStore(push, tile * 16u + xy, value, sourceMip + 2u);
#if HZB_SPD_WAVE_OPS
    if (sourceMip + 3u >= push.mipCount) { return; }
    // Keep all wave lanes participating: every shuffle source is active,
    // including lanes whose pixels are outside a partial border tile.
    value = spdWaveReduce4(value, 1u, push.reversedZ != 0u);
    if ((lane & 3u) == 0u) { spdStore(push, tile * 8u + (xy >> 1u), value, sourceMip + 3u); }
    if (sourceMip + 4u >= push.mipCount) { return; }
    value = spdWaveReduce4(value, 4u, push.reversedZ != 0u);
    if ((lane & 15u) == 0u) {
        spdStore(push, tile * 4u + (xy >> 2u), value, sourceMip + 4u);
        spdIntermediate[lane >> 4u] = value;
    }
    if (sourceMip + 5u >= push.mipCount) { return; }
    GroupMemoryBarrierWithGroupSync();
    // Gather the 16 Morton-ordered results into the first wave. Only these
    // 16 lanes participate, and all shuffle indices stay within [0, 15].
    if (lane < 16u) {
        value = spdIntermediate[lane];
        value = spdWaveReduce4(value, 1u, push.reversedZ != 0u);
        if ((lane & 3u) == 0u) {
            spdStore(push, tile * 2u + (spdMortonPixel(lane) >> 1u), value, sourceMip + 5u);
        }
        if (sourceMip + 6u < push.mipCount) {
            value = spdWaveReduce4(value, 4u, push.reversedZ != 0u);
            if (lane == 0u) { spdStore(push, tile, value, sourceMip + 6u); }
        }
    }
#else
    spdIntermediate[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    // All invocations participate in barriers, including partial border tiles.
    for (uint level = 3u; level <= 6u && sourceMip + level < push.mipCount; ++level) {
        uint side = 64u >> level;
        uint2 pixel = uint2(lane % side, lane / side);
        if (lane < side * side) {
            uint base = pixel.y * 2u * 16u + pixel.x * 2u;
            value = spdReduce4(spdIntermediate[base], spdIntermediate[base + 1u],
                spdIntermediate[base + 16u], spdIntermediate[base + 17u], push.reversedZ != 0u);
            spdStore(push, tile * side + pixel, value, sourceMip + level);
        }
        // Finish every read before reusing LDS locations for the next mip.
        GroupMemoryBarrierWithGroupSync();
        if (lane < side * side) { spdIntermediate[pixel.y * 16u + pixel.x] = value; }
        GroupMemoryBarrierWithGroupSync();
    }
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void hzbSpdMain(uint3 group : SV_GroupID, uint localIndex : SV_GroupIndex, uniform HzbSpdPush push)
{
#if HZB_SPD_WAVE_OPS
    uint lane = spdInvocationIndex();
#else
    uint lane = localIndex;
#endif
    spdDownsampleTile(push, group.xy, lane, 0u);
    if (push.mipCount <= 7u) { return; }
    RWStructuredBuffer<Atomic<uint>> counter = DescriptorHandle<RWStructuredBuffer<Atomic<uint>>>(
        uint2(push.bufferShaderIndexBase + push.counterBuffer, 0));
    // Mip 6 and this RMW share logical lane 0. The acq-rel RMW chain elects a
    // single last group without spinning or assuming workgroup scheduling order.
    if (lane == 0u) { spdCompletedGroups = counter[0].add(1u, MemoryOrder.AcquireRelease); }
    DeviceMemoryBarrierWithGroupSync();
    uint2 tiles = (uint2(push.width, push.height) + 63u) / 64u;
    if (spdCompletedGroups + 1u != tiles.x * tiles.y) { return; }
    spdDownsampleTile(push, uint2(0), lane, 6u);
    if (lane == 0u) { counter[0].store(0u, MemoryOrder.Release); }
}
