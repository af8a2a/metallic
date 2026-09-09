#ifndef METALLIC_SPHERICAL_HARMONICS
#define METALLIC_SPHERICAL_HARMONICS

// Low-order SH API inspired by/adapted from MJP's SHforHLSL (MIT).
// https://github.com/TheRealMJP/SHforHLSL
// See ../../Licenses/SHforHLSL.txt. Slang generics and free operators provide the
// HLSL 2021-style call sites without requiring a second shader compiler.
namespace SH {

static const float kPi = 3.14159265358979323846;
static const int kL1CoefficientCount = 4;
static const int kL2CoefficientCount = 9;

// L1 = two bands (l <= 1), L2 = three bands (l <= 2).
// T is the coefficient value type, e.g. float, float3, half or half3.
struct Coefficients<T : IFloat, let N : int>
{
    static const int kCoefficientCount = N;
    T c[N];

    static Coefficients<T, N> zero()
    {
        Coefficients<T, N> result;
        [unroll] for (int i = 0; i < N; ++i) { result.c[i] = T(0.0); }
        return result;
    }
};

typealias L1Of<T : IFloat> = Coefficients<T, kL1CoefficientCount>;
typealias L2Of<T : IFloat> = Coefficients<T, kL2CoefficientCount>;
typealias L1 = L1Of<float>;
typealias L1RGB = L1Of<float3>;
typealias L2 = L2Of<float>;
typealias L2RGB = L2Of<float3>;

} // namespace SH

// Slang operators are free functions; global scope keeps them usable without
// importing the SH namespace into each including shader.
SH::Coefficients<T, N> operator+<T : IFloat, let N : int>(SH::Coefficients<T, N> a, SH::Coefficients<T, N> b)
{
    [unroll] for (int i = 0; i < N; ++i) { a.c[i] = a.c[i] + b.c[i]; }
    return a;
}

SH::Coefficients<T, N> operator-<T : IFloat, let N : int>(SH::Coefficients<T, N> a, SH::Coefficients<T, N> b)
{
    [unroll] for (int i = 0; i < N; ++i) { a.c[i] = a.c[i] - b.c[i]; }
    return a;
}

SH::Coefficients<T, N> operator*<T : IFloat, let N : int>(SH::Coefficients<T, N> sh, float scale)
{
    [unroll] for (int i = 0; i < N; ++i) { sh.c[i] = sh.c[i] * T(scale); }
    return sh;
}

SH::Coefficients<T, N> operator*<T : IFloat, let N : int>(float scale, SH::Coefficients<T, N> sh)
{
    return sh * scale;
}

SH::Coefficients<T, N> operator/<T : IFloat, let N : int>(SH::Coefficients<T, N> sh, float scale)
{
    [unroll] for (int i = 0; i < N; ++i) { sh.c[i] = sh.c[i] / T(scale); }
    return sh;
}

// Component-wise tinting; this is not the SH product of two angular functions.
SH::Coefficients<vector<T, C>, N> operator*<T : __BuiltinFloatingPointType, let C : int, let N : int>(
    SH::Coefficients<vector<T, C>, N> sh, vector<T, C> tint)
{
    [unroll] for (int i = 0; i < N; ++i) { sh.c[i] *= tint; }
    return sh;
}

SH::Coefficients<vector<T, C>, N> operator/<T : __BuiltinFloatingPointType, let C : int, let N : int>(
    SH::Coefficients<vector<T, C>, N> sh, vector<T, C> tint)
{
    [unroll] for (int i = 0; i < N; ++i) { sh.c[i] /= tint; }
    return sh;
}

void operator+=<T : IFloat, let N : int>(inout SH::Coefficients<T, N> a, SH::Coefficients<T, N> b)
{
    a = a + b;
}

void operator-=<T : IFloat, let N : int>(inout SH::Coefficients<T, N> a, SH::Coefficients<T, N> b)
{
    a = a - b;
}

void operator*=<T : IFloat, let N : int>(inout SH::Coefficients<T, N> sh, float scale)
{
    sh = sh * scale;
}

void operator/=<T : IFloat, let N : int>(inout SH::Coefficients<T, N> sh, float scale)
{
    sh = sh / scale;
}

void operator*=<T : __BuiltinFloatingPointType, let C : int, let N : int>(
    inout SH::Coefficients<vector<T, C>, N> sh, vector<T, C> tint)
{
    sh = sh * tint;
}

void operator/=<T : __BuiltinFloatingPointType, let C : int, let N : int>(
    inout SH::Coefficients<vector<T, C>, N> sh, vector<T, C> tint)
{
    sh = sh / tint;
}

namespace SH {

Coefficients<T, N> lerp<T : IFloat, let N : int>(Coefficients<T, N> a, Coefficients<T, N> b, float weight)
{
    return a * (1.0 - weight) + b * weight;
}

T dotProduct<T : IFloat, let N : int>(Coefficients<T, N> a, Coefficients<T, N> b)
{
    T value = T(0.0);
    [unroll] for (int i = 0; i < N; ++i) { value = value + a.c[i] * b.c[i]; }
    return value;
}

// Unit directions. Preserve Metallic/SHforHLSL's real basis signs and order:
// [1, y, z, x, xy, yz, (3z^2-1), xz, (x^2-y^2)]. Y-up environments only
// affect lat-long direction mapping, not this coefficient convention.
float basis(uint index, float3 d)
{
    switch (index) {
    case 0u: return 0.2820947918;
    case 1u: return 0.4886025119 * d.y;
    case 2u: return 0.4886025119 * d.z;
    case 3u: return 0.4886025119 * d.x;
    case 4u: return 1.0925484306 * d.x * d.y;
    case 5u: return 1.0925484306 * d.y * d.z;
    case 6u: return 0.3153915653 * (3.0 * d.z * d.z - 1.0);
    case 7u: return 1.0925484306 * d.x * d.z;
    case 8u: return 0.5462742153 * (d.x * d.x - d.y * d.y);
    default: return 0.0;
    }
}

L1Of<T> projectOntoL1<T : IFloat>(float3 direction, T value)
{
    L1Of<T> result;
    [unroll] for (int i = 0; i < kL1CoefficientCount; ++i) { result.c[i] = T(basis(i, direction)) * value; }
    return result;
}

L2Of<T> projectOntoL2<T : IFloat>(float3 direction, T value)
{
    L2Of<T> result;
    [unroll] for (int i = 0; i < kL2CoefficientCount; ++i) { result.c[i] = T(basis(i, direction)) * value; }
    return result;
}

// Raw angular evaluation. No cosine convolution, albedo, exposure or clamping.
T evaluate<T : IFloat, let N : int>(Coefficients<T, N> sh, float3 direction)
{
    T value = T(0.0);
    [unroll] for (int i = 0; i < N; ++i) { value = value + sh.c[i] * T(basis(i, direction)); }
    return value;
}

L1Of<T> truncateToL1<T : IFloat>(L2Of<T> sh)
{
    L1Of<T> result;
    [unroll] for (int i = 0; i < kL1CoefficientCount; ++i) { result.c[i] = sh.c[i]; }
    return result;
}

Coefficients<vector<T, 3>, N> toRGB<T : __BuiltinFloatingPointType, let N : int>(Coefficients<T, N> sh)
{
    Coefficients<vector<T, 3>, N> result;
    [unroll] for (int i = 0; i < N; ++i) { result.c[i] = vector<T, 3>(sh.c[i]); }
    return result;
}

float cosineConvolutionWeight(uint coefficientIndex)
{
    return coefficientIndex == 0u ? kPi : (coefficientIndex < 4u ? 2.0 * kPi / 3.0 : kPi / 4.0);
}

// ZH arguments are per-band convolution multipliers, already including the
// sqrt(4*pi/(2*l+1)) normalization when starting from projected ZH coefficients.
L1Of<T> convolveWithZH<T : IFloat>(L1Of<T> sh, float2 bandWeights)
{
    [unroll] for (int i = 0; i < kL1CoefficientCount; ++i) { sh.c[i] = sh.c[i] * T(i == 0 ? bandWeights.x : bandWeights.y); }
    return sh;
}

L2Of<T> convolveWithZH<T : IFloat>(L2Of<T> sh, float3 bandWeights)
{
    [unroll] for (int i = 0; i < kL2CoefficientCount; ++i) {
        sh.c[i] = sh.c[i] * T(i == 0 ? bandWeights.x : (i < 4 ? bandWeights.y : bandWeights.z));
    }
    return sh;
}

Coefficients<T, N> convolveWithCosineLobe<T : IFloat, let N : int>(Coefficients<T, N> radiance)
{
    [unroll] for (int i = 0; i < N; ++i) { radiance.c[i] = radiance.c[i] * T(cosineConvolutionWeight(i)); }
    return radiance;
}

// Input must be radiance SH. Lambertian shading still applies albedo / pi.
T calculateIrradiance<T : IFloat, let N : int>(Coefficients<T, N> radiance, float3 normal)
{
    return evaluate(convolveWithCosineLobe(radiance), normal);
}

// Active rotation, using HLSL row vectors: directions transform as mul(d, R).
// R must be orthonormal. evaluate(rotate(sh, R), mul(n, R)) == evaluate(sh, n).
L1Of<T> rotate<T : IFloat>(L1Of<T> sh, float3x3 rotation)
{
    L1Of<T> result;
    result.c[0] = sh.c[0];
    result.c[3] = sh.c[3] * T(rotation[0][0]) + sh.c[1] * T(rotation[1][0]) + sh.c[2] * T(rotation[2][0]);
    result.c[1] = sh.c[3] * T(rotation[0][1]) + sh.c[1] * T(rotation[1][1]) + sh.c[2] * T(rotation[2][1]);
    result.c[2] = sh.c[3] * T(rotation[0][2]) + sh.c[1] * T(rotation[1][2]) + sh.c[2] * T(rotation[2][2]);
    return result;
}

L2Of<T> rotate<T : IFloat>(L2Of<T> sh, float3x3 rotation)
{
    L2Of<T> result;
    L1Of<T> linear = rotate(truncateToL1(sh), rotation);
    [unroll] for (int i = 0; i < kL1CoefficientCount; ++i) { result.c[i] = linear.c[i]; }

    // The l=2 band is a symmetric, traceless quadratic form n^T Q n.
    // Rotate Q with transpose(R) * Q * R, then recover the five coefficients.
    const float bXY = 1.0925484306;
    const float bZ2 = 0.3153915653;
    const float bX2Y2 = 0.5462742153;
    T q[9];
    q[0] = sh.c[8] * T(bX2Y2) - sh.c[6] * T(bZ2);
    q[4] = -sh.c[8] * T(bX2Y2) - sh.c[6] * T(bZ2);
    q[8] = sh.c[6] * T(2.0 * bZ2);
    q[1] = q[3] = sh.c[4] * T(0.5 * bXY);
    q[2] = q[6] = sh.c[7] * T(0.5 * bXY);
    q[5] = q[7] = sh.c[5] * T(0.5 * bXY);
    T rotated[9];
    [unroll] for (int row = 0; row < 3; ++row) {
        [unroll] for (int col = 0; col < 3; ++col) {
            T value = T(0.0);
            [unroll] for (int a = 0; a < 3; ++a) {
                [unroll] for (int b = 0; b < 3; ++b) {
                    value = value + q[a * 3 + b] * T(rotation[a][row] * rotation[b][col]);
                }
            }
            rotated[row * 3 + col] = value;
        }
    }
    result.c[4] = rotated[1] * T(2.0 / bXY);
    result.c[5] = rotated[5] * T(2.0 / bXY);
    result.c[6] = rotated[8] * T(0.5 / bZ2);
    result.c[7] = rotated[2] * T(2.0 / bXY);
    result.c[8] = (rotated[0] - rotated[4]) * T(0.5 / bX2Y2);
    return result;
}

// Explicit packing boundary: the environment ABI stays nine float4 slots,
// RGB coefficients in xyz, reserved w=0. Do not reinterpret a compact L2RGB.
L2RGB loadL2RGB(StructuredBuffer<float4> buffer, uint offset = 0u)
{
    L2RGB result;
    [unroll] for (int i = 0; i < kL2CoefficientCount; ++i) { result.c[i] = buffer[offset + i].xyz; }
    return result;
}

void storeL2RGB(RWStructuredBuffer<float4> buffer, uint offset, L2RGB sh)
{
    [unroll] for (int i = 0; i < kL2CoefficientCount; ++i) { buffer[offset + i] = float4(sh.c[i], 0.0); }
}

// A distinct type for already-convolved environment/probe data prevents passing
// it to calculateIrradiance() and accidentally applying the cosine lobe twice.
struct IrradianceL2RGB
{
    L2RGB coefficients;

    float3 evaluate(float3 normal)
    {
        return SH::evaluate(coefficients, normal);
    }
};

IrradianceL2RGB toIrradiance(L2RGB radiance)
{
    IrradianceL2RGB result;
    result.coefficients = convolveWithCosineLobe(radiance);
    return result;
}

IrradianceL2RGB loadIrradianceL2RGB(StructuredBuffer<float4> buffer, uint offset = 0u)
{
    IrradianceL2RGB result;
    result.coefficients = loadL2RGB(buffer, offset);
    return result;
}

} // namespace SH

#endif // METALLIC_SPHERICAL_HARMONICS
