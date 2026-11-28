#language slang 2026
module TerrainCliffNoise;

// The 3D Cycles Noise Texture fBm used by the Ancient Rift Valley preview.
// Adapted from Blender v4.5.3: intern/cycles/{util/hash.h,kernel/svm/{noise,fractal_noise,noisetex}.h}.
// See Shaders/Licenses/BlenderCycles.txt for attribution and license terms.
namespace Metallic.TerrainCliff
{
internal uint rotateLeft(uint value, uint bits)
{
    return (value << bits) | (value >> (32u - bits));
}

internal uint hashFinal(uint a, uint b, uint c)
{
    c ^= b;
    c -= rotateLeft(b, 14u);
    a ^= c;
    a -= rotateLeft(c, 11u);
    b ^= a;
    b -= rotateLeft(a, 25u);
    c ^= b;
    c -= rotateLeft(b, 16u);
    a ^= c;
    a -= rotateLeft(c, 4u);
    b ^= a;
    b -= rotateLeft(a, 14u);
    c ^= b;
    c -= rotateLeft(b, 24u);
    return c;
}

internal uint hashUint3(uint x, uint y, uint z)
{
    const uint initial = 0xdeadbeefu + (3u << 2u) + 13u;
    return hashFinal(initial + x, initial + y, initial + z);
}

internal float hashFloat2ToFloat(float x, float y)
{
    const uint initial = 0xdeadbeefu + (2u << 2u) + 13u;
    return float(hashFinal(initial + asuint(x), initial + asuint(y), initial)) * (1.0 / float(0xffffffffu));
}

internal float3 randomFloat3Offset(float seed)
{
    return float3(100.0 + hashFloat2ToFloat(seed, 0.0) * 100.0,
        100.0 + hashFloat2ToFloat(seed, 1.0) * 100.0,
        100.0 + hashFloat2ToFloat(seed, 2.0) * 100.0);
}

internal float fade(float t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

internal float grad3(uint hash, float x, float y, float z)
{
    uint h = hash & 15u;
    float u = h < 8u ? x : y;
    float vt = ((h == 12u) || (h == 14u)) ? x : z;
    float v = h < 4u ? y : vt;
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);
}

internal float triMix(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7,
    float x, float y, float z)
{
    float x1 = 1.0 - x;
    float y1 = 1.0 - y;
    float z1 = 1.0 - z;
    return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) +
        z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));
}

internal float perlin3D(float3 p)
{
    float3 cell = floor(p);
    uint3 lattice = asuint(int3(cell));
    uint x = lattice.x;
    uint y = lattice.y;
    uint z = lattice.z;
    float3 f = p - cell;
    float fx = f.x;
    float fy = f.y;
    float fz = f.z;
    return triMix(grad3(hashUint3(x, y, z), fx, fy, fz),
        grad3(hashUint3(x + 1u, y, z), fx - 1.0, fy, fz),
        grad3(hashUint3(x, y + 1u, z), fx, fy - 1.0, fz),
        grad3(hashUint3(x + 1u, y + 1u, z), fx - 1.0, fy - 1.0, fz),
        grad3(hashUint3(x, y, z + 1u), fx, fy, fz - 1.0),
        grad3(hashUint3(x + 1u, y, z + 1u), fx - 1.0, fy, fz - 1.0),
        grad3(hashUint3(x, y + 1u, z + 1u), fx, fy - 1.0, fz - 1.0),
        grad3(hashUint3(x + 1u, y + 1u, z + 1u), fx - 1.0, fy - 1.0, fz - 1.0),
        fade(fx), fade(fy), fade(fz));
}

internal float signedNoise3D(float3 p)
{
    const float3 precisionCorrection = 0.5 * float3(abs(p) >= float3(1000000.0));
    p = fmod(p, float3(100000.0)) + precisionCorrection;
    return 0.9820 * perlin3D(p);
}

internal float normalizedFbm(float3 p, uint detail, float roughness)
{
    float fscale = 1.0;
    float amp = 1.0;
    float maxamp = 0.0;
    float sum = 0.0;
    // The authored detail is integral. The Cycles fractional-octave branch is unreachable here.
    for (uint i = 0u; i <= detail; ++i) {
        const float t = signedNoise3D(fscale * p);
        sum += t * amp;
        maxamp += amp;
        amp *= roughness;
        fscale *= 2.0;
    }
    return 0.5 * sum / maxamp + 0.5;
}

internal float3 noiseColor(float3 p)
{
    return float3(normalizedFbm(p, 2u, 0.63), normalizedFbm(p + randomFloat3Offset(3.0), 2u, 0.63),
        normalizedFbm(p + randomFloat3Offset(4.0), 2u, 0.63));
}

public float3 cliffTexturePhase(float3 position)
{
    return (noiseColor(position * 0.032) - 0.5) * 1.15 + (noiseColor(position * 0.093) - 0.5) * 0.42;
}

public float cliffTint(float3 position)
{
    return 0.71 + 0.59 * normalizedFbm(position * 0.023, 3u, 0.5);
}
} // namespace Metallic.TerrainCliff
