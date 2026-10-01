// Adapted from UE 5.7.4 ACESUtils / OpenColorIO 2.4.1.
// Copyright Epic Games, Inc. All Rights Reserved.
// See Shaders/Licenses/ColorGrading (Academy / OpenColorIO BSD notices).
#include "Runtime/Render/Core/ACESTables.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace metallic::render {
namespace {
// CPU ACES2 lookup tables; matrices use column vectors.
using int32 = int32_t;
constexpr float kPi = 3.14159265358979323846f;
template <class T> struct Vector3 {
    T X{}, Y{}, Z{};
    Vector3() = default;
    Vector3(T x, T y, T z) : X(x), Y(y), Z(z) {}
    template <class U> explicit Vector3(const Vector3<U>& v) : X(T(v.X)), Y(T(v.Y)), Z(T(v.Z)) {}
    T& operator[](int i) { return i == 0 ? X : (i == 1 ? Y : Z); }
    T operator[](int i) const { return i == 0 ? X : (i == 1 ? Y : Z); }
    static Vector3 one() { return {1, 1, 1}; }
    Vector3 operator*(T s) const { return {X * s, Y * s, Z * s}; }
    Vector3 operator+(T s) const { return {X + s, Y + s, Z + s}; }
    friend Vector3 operator*(T s, const Vector3& v) { return v * s; }
};
using Float3 = Vector3<float>;
using Double3 = Vector3<double>;
struct Float2 {
    float X{}, Y{};
    float operator[](int i) const { return i == 0 ? X : Y; }
};
template <class T> struct Matrix3 {
    T m[3][3]{};
    Matrix3() = default;
    template <class U> explicit Matrix3(const Matrix3<U>& v)
    {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                m[r][c] = T(v.m[r][c]);
            }
        }
    }
    Vector3<T> transform(const Vector3<T>& v) const
    {
        return {m[0][0] * v.X + m[0][1] * v.Y + m[0][2] * v.Z, m[1][0] * v.X + m[1][1] * v.Y + m[1][2] * v.Z,
                m[2][0] * v.X + m[2][1] * v.Y + m[2][2] * v.Z};
    }
    Matrix3 operator*(const Matrix3& b) const
    {
        Matrix3 out;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < 3; ++k) {
                    out.m[r][c] += m[r][k] * b.m[k][c];
                }
            }
        }
        return out;
    }
    Matrix3 scaled(T scale) const
    {
        auto out = *this;
        for (auto& row : out.m) {
            for (auto& value : row) {
                value *= scale;
            }
        }
        return out;
    }
    Matrix3 inverse() const
    {
        Matrix3 out;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out.m[c][r] = m[(r + 1) % 3][(c + 1) % 3] * m[(r + 2) % 3][(c + 2) % 3] -
                              m[(r + 1) % 3][(c + 2) % 3] * m[(r + 2) % 3][(c + 1) % 3];
            }
        }
        const T det = m[0][0] * out.m[0][0] + m[0][1] * out.m[1][0] + m[0][2] * out.m[2][0];
        return out.scaled(T(1) / det);
    }
};
using DoubleMatrix = Matrix3<double>;
using FloatMatrix = Matrix3<float>;
enum class Primaries { ACESAP0, ACESAP1, ACESCAM16 };
enum class Adaptation { None };
struct ColorSpace {
    DoubleMatrix rgbToXYZ;
    explicit ColorSpace(Primaries space)
    {
        // Exact chromaticities from UE ColorSpace.cpp, including ACES D60.
        double xy[4][2] = {{0.7347, 0.2653}, {0, 1}, {0.0001, -0.077}, {0.32168, 0.33767}};
        if (space == Primaries::ACESAP1) {
            xy[0][0] = 0.713;
            xy[0][1] = 0.293;
            xy[1][0] = 0.165;
            xy[1][1] = 0.830;
            xy[2][0] = 0.128;
            xy[2][1] = 0.044;
        } else if (space == Primaries::ACESCAM16) {
            xy[0][0] = 0.8336;
            xy[0][1] = 0.1735;
            xy[1][0] = 2.3854;
            xy[1][1] = -1.4659;
            xy[2][0] = 0.087;
            xy[2][1] = -0.125;
            xy[3][0] = xy[3][1] = 0.333;
        }
        for (int c = 0; c < 3; ++c) {
            rgbToXYZ.m[0][c] = xy[c][0] / xy[c][1];
            rgbToXYZ.m[1][c] = 1;
            rgbToXYZ.m[2][c] = (1 - xy[c][0] - xy[c][1]) / xy[c][1];
        }
        const auto scale = rgbToXYZ.inverse().transform({xy[3][0] / xy[3][1], 1, (1 - xy[3][0] - xy[3][1]) / xy[3][1]});
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                rgbToXYZ.m[r][c] *= scale[c];
            }
        }
    }
    DoubleMatrix rgbToXYZMatrix() const { return rgbToXYZ; }
    DoubleMatrix xyzToRGBMatrix() const { return rgbToXYZ.inverse(); }
};
struct ColorSpaceTransform : DoubleMatrix {
    ColorSpaceTransform(const ColorSpace& from, const ColorSpace& to, Adaptation)
        : DoubleMatrix(to.xyzToRGBMatrix() * from.rgbToXYZMatrix())
    {
    }
};
struct ScalarMath {
    template <class T> static constexpr T Min(T a, T b) { return std::min(a, b); }
    template <class T> static constexpr T Max(T a, T b) { return std::max(a, b); }
    static float Abs(float x) { return std::abs(x); }
    static float Pow(float x, float y) { return std::pow(x, y); }
    static float CopySign(float x, float y) { return std::copysign(x, y); }
    static float Sqrt(float x) { return std::sqrt(x); }
    static float Cos(float x) { return std::cos(x); }
    static float Sin(float x) { return std::sin(x); }
    static float Fmod(float x, float y) { return std::fmod(x, y); }
    static float Atan2(float x, float y) { return std::atan2(x, y); }
    static float Loge(float x) { return std::log(x); }
    static float Lerp(float x, float y, float t) { return x + (y - x) * t; }
};

constexpr int32 TABLE_SIZE = 360;
constexpr int32 TABLE_ADDITION_ENTRIES = 2;
constexpr int32 TABLE_TotalSize = TABLE_SIZE + TABLE_ADDITION_ENTRIES;
constexpr int32 GAMUT_TABLE_BASE_INDEX = 1;

constexpr float ReferenceLuminance = 100.f;
constexpr float L_A = 100.f;
constexpr float Y_b = 20.f;
constexpr float AcResp = 1.f;
constexpr float Ra = 2.f * AcResp;
constexpr float Ba = 0.05f + (2.f - Ra);
constexpr float Surround[3] = {0.9f, 0.59f, 0.9f};

constexpr float SmoothCusps = 0.12f;
constexpr float SmoothM = 0.27f;
constexpr float CuspMidBlend = 1.3f;
constexpr float FocusGainBlend = 0.3f;
constexpr float FocusAdjustGain = 0.55f;
constexpr float FocusDistance = 1.35f;
constexpr float FocusDistanceScaling = 1.75f;
constexpr float CompressionThreshold = 0.75f;

constexpr float GammaMinimum = 0.0f;
constexpr float GammaMaximum = 5.0f;
constexpr float GammaSearchStep = 0.4f;
constexpr float GammaAccuracy = 1e-5f;

struct JMhParams {
    float F_L;
    float z;
    float A_w;
    float A_w_J;
    Float3 XYZ_w;
    Float3 D_RGB;
    FloatMatrix MATRIX_RGB_to_CAM16;
    FloatMatrix MATRIX_CAM16_to_RGB;
};

struct ToneScaleParams {
    float n;
    float n_r;
    float g;
    float t_1;
    float c_t;
    float s_2;
    float u_2;
    float m_2;
};

struct Table3D {
    static constexpr int32 BaseIndex = GAMUT_TABLE_BASE_INDEX;
    static constexpr int32 Size = TABLE_SIZE;
    static constexpr int32 TotalSize = TABLE_TotalSize;
    Float3 Data[TABLE_TotalSize];
};

struct Table1D {
    static constexpr int32 BaseIndex = GAMUT_TABLE_BASE_INDEX;
    static constexpr int32 Size = TABLE_SIZE;
    static constexpr int32 TotalSize = TABLE_TotalSize;
    float Data[TABLE_TotalSize];
};

float PanlrcForward(float Value, float F_L)
{
    const float F_L_v = ScalarMath::Pow(F_L * ScalarMath::Abs(Value) / ReferenceLuminance, 0.42f);
    return (400.f * ScalarMath::CopySign(1.f, Value) * F_L_v) / (27.13f + F_L_v);
}

float PanlrcInverse(float Value, float F_L)
{
    return ScalarMath::CopySign(1.f, Value) * ReferenceLuminance / F_L *
           ScalarMath::Pow((27.13f * ScalarMath::Abs(Value) / (400.f - ScalarMath::Abs(Value))), 1.f / 0.42f);
}

inline bool AnyBelowZero(const Float3& RGB) { return (RGB[0] < 0. || RGB[1] < 0. || RGB[2] < 0.); }

bool OutsideHull(const Float3& RGB)
{

    constexpr float MaxRGBtestVal = 1.0;
    return RGB[0] > MaxRGBtestVal || RGB[1] > MaxRGBtestVal || RGB[2] > MaxRGBtestVal;
}

float Y_to_J(float Y, const JMhParams& Params)
{
    float F_L_Y = ScalarMath::Pow(Params.F_L * ScalarMath::Abs(Y) / ReferenceLuminance, 0.42f);
    return ScalarMath::CopySign(1.f, Y) * ReferenceLuminance *
           ScalarMath::Pow(((400.f * F_L_Y) / (27.13f + F_L_Y)) / Params.A_w_J, Surround[1] * Params.z);
}

float WrapTo360(float Hue)
{
    float Y = ScalarMath::Fmod(Hue, 360.f);
    if (Y < 0.f) {
        Y = Y + 360.f;
    }

    return Y;
}

int32 HuePositionInUniformTable(float Hue, int32 TableSize)
{
    const float WrappedHue = WrapTo360(Hue);

    return int32(WrappedHue / 360.f * (float)TableSize);
}

int32 ClampToTableBounds(int32 Entry, int32 TableSize)
{
    return ScalarMath::Min(TableSize - 1, ScalarMath::Max(0, Entry));
}

float Smin(float a, float b, float s)
{
    const float h = ScalarMath::Max(s - ScalarMath::Abs(a - b), 0.f) / s;
    return ScalarMath::Min(a, b) - h * h * h * s * (1.f / 6.f);
}

Float3 JMh_to_RGB(const Float3& JMh, const JMhParams& Params)
{
    const float J = JMh[0];
    const float M = JMh[1];
    const float h = JMh[2];

    const float HRad = h * kPi / 180.f;

    const float Scale = M / (43.f * Surround[2]);
    const float A = Params.A_w * ScalarMath::Pow(J / 100.f, 1.f / (Surround[1] * Params.z));
    const float a = Scale * ScalarMath::Cos(HRad);
    const float b = Scale * ScalarMath::Sin(HRad);

    const float RedA = (460.f * A + 451.f * a + 288.f * b) / 1403.f;
    const float GrnA = (460.f * A - 891.f * a - 261.f * b) / 1403.f;
    const float BluA = (460.f * A - 220.f * a - 6300.f * b) / 1403.f;

    Float3 CamM;
    CamM.X = PanlrcInverse(RedA, Params.F_L) / Params.D_RGB[0];
    CamM.Y = PanlrcInverse(GrnA, Params.F_L) / Params.D_RGB[1];
    CamM.Z = PanlrcInverse(BluA, Params.F_L) / Params.D_RGB[2];

    return Params.MATRIX_CAM16_to_RGB.transform(CamM);
}

JMhParams InitJMhParams(const ColorSpace& InColorSpace)
{
    static ColorSpace ColorSpaceCAM16 = ColorSpace(Primaries::ACESCAM16);

    const Double3 XYZ_w = InColorSpace.rgbToXYZMatrix().transform(Double3::one() * ReferenceLuminance);
    const float Y_W = XYZ_w[1];
    const Double3 RGB_w = ColorSpaceCAM16.xyzToRGBMatrix().transform(XYZ_w);

    const float K = 1.f / (5.f * L_A + 1.f);
    const float K4 = ScalarMath::Pow(K, 4.f);
    const float N = Y_b / Y_W;
    const float F_L =
        0.2f * K4 * (5.f * L_A) + 0.1f * ScalarMath::Pow((1.f - K4), 2.f) * ScalarMath::Pow(5.f * L_A, 1.f / 3.f);
    const float z = 1.48f + ScalarMath::Sqrt(N);

    const Float3 D_RGB = {Y_W / (float)RGB_w[0], Y_W / (float)RGB_w[1], Y_W / (float)RGB_w[2]};

    const Float3 RGB_WC{D_RGB[0] * (float)RGB_w[0], D_RGB[1] * (float)RGB_w[1], D_RGB[2] * (float)RGB_w[2]};

    const Float3 RGB_AW = {PanlrcForward(RGB_WC[0], F_L), PanlrcForward(RGB_WC[1], F_L), PanlrcForward(RGB_WC[2], F_L)};

    const float A_w = Ra * RGB_AW[0] + RGB_AW[1] + Ba * RGB_AW[2];
    const float F_L_W = ScalarMath::Pow(F_L, 0.42f);
    const float A_w_J = (400.f * F_L_W) / (27.13f + F_L_W);

    JMhParams Params;
    Params.XYZ_w = Float3(Double3(XYZ_w));
    Params.F_L = F_L;
    Params.z = z;
    Params.D_RGB = D_RGB;
    Params.A_w = A_w;
    Params.A_w_J = A_w_J;

    ColorSpaceTransform ToCAM16T(InColorSpace, ColorSpaceCAM16, Adaptation::None);
    DoubleMatrix ToCAM16 = ToCAM16T.scaled(100.0);

    Params.MATRIX_RGB_to_CAM16 = FloatMatrix(ToCAM16);
    Params.MATRIX_CAM16_to_RGB = FloatMatrix(ToCAM16.inverse());

    return Params;
}

ToneScaleParams InitToneScaleParams(float PeakLuminance)
{

    const float n = PeakLuminance;

    const float n_r = 100.0f;
    const float g = 1.15f;
    const float c = 0.18f;
    const float c_d = 10.013f;
    const float w_g = 0.14f;
    const float t_1 = 0.04f;
    const float r_hit_min = 128.f;
    const float r_hit_max = 896.f;

    const float r_hit =
        r_hit_min + (r_hit_max - r_hit_min) * (ScalarMath::Loge(n / n_r) / ScalarMath::Loge(10000.f / 100.f));
    const float m_0 = (n / n_r);
    const float m_1 = 0.5f * (m_0 + ScalarMath::Sqrt(m_0 * (m_0 + 4.f * t_1)));
    const float u = ScalarMath::Pow((r_hit / m_1) / ((r_hit / m_1) + 1.f), g);
    const float m = m_1 / u;
    const float w_i = ScalarMath::Loge(n / 100.f) / ScalarMath::Loge(2.f);
    const float c_t = c_d / n_r * (1.f + w_i * w_g);
    const float g_ip = 0.5f * (c_t + ScalarMath::Sqrt(c_t * (c_t + 4.f * t_1)));
    const float g_ipp2 = -(m_1 * ScalarMath::Pow((g_ip / m), (1.f / g))) / (ScalarMath::Pow(g_ip / m, 1.f / g) - 1.f);
    const float w_2 = c / g_ipp2;
    const float s_2 = w_2 * m_1;
    const float u_2 = ScalarMath::Pow((r_hit / m_1) / ((r_hit / m_1) + w_2), g);
    const float m_2 = m_1 / u_2;

    ToneScaleParams TonescaleParams = {n, n_r, g, t_1, c_t, s_2, u_2, m_2};

    return TonescaleParams;
}

Table1D MakeReachMTable(float PeakLuminance)
{
    static ColorSpace ColorSpaceAP1 = ColorSpace(Primaries::ACESAP1);

    const JMhParams Params = InitJMhParams(ColorSpaceAP1);
    const float LimitJMax = Y_to_J(PeakLuminance, Params);

    Table1D GamutReachTable{};

    for (int32 Index = 0; Index < GamutReachTable.Size; Index++) {
        const float Hue = (float)Index;
        const float SearchRange = 50.f;

        float Low = 0.;
        float High = Low + SearchRange;
        bool Outside = false;

        while ((Outside != true) && (High < 1300.f)) {
            const Float3 SearchJMh = Float3{LimitJMax, High, Hue};
            const Float3 NewLimitRGB = JMh_to_RGB(SearchJMh, Params);
            Outside = AnyBelowZero(NewLimitRGB);

            if (Outside == false) {
                Low = High;
                High = High + SearchRange;
            }
        }

        while (High - Low > 1e-2) {
            const float SampleM = (High + Low) / 2.f;
            const Float3 SearchJMh = Float3{LimitJMax, SampleM, Hue};
            const Float3 NewLimitRGB = JMh_to_RGB(SearchJMh, Params);
            Outside = AnyBelowZero(NewLimitRGB);

            if (Outside) {
                High = SampleM;
            } else {
                Low = SampleM;
            }
        }

        GamutReachTable.Data[Index] = High;
    }

    return GamutReachTable;
}

Float3 HSV_to_RGB(const Float3& HSV)
{
    const float C = HSV[2] * HSV[1];
    const float X = C * (1.f - ScalarMath::Abs(ScalarMath::Fmod(HSV[0] * 6.f, 2.f) - 1.f));
    const float m = HSV[2] - C;

    Float3 RGB{};
    if (HSV[0] < 1.f / 6.f) {
        RGB = {C, X, 0.f};
    } else if (HSV[0] < 2. / 6.) {
        RGB = {X, C, 0.f};
    } else if (HSV[0] < 3. / 6.) {
        RGB = {0.f, C, X};
    } else if (HSV[0] < 4. / 6.) {
        RGB = {0.f, X, C};
    } else if (HSV[0] < 5. / 6.) {
        RGB = {X, 0.f, C};
    } else {
        RGB = {C, 0.f, X};
    }
    RGB = RGB + m;

    return RGB;
}

Float3 RGB_to_JMh(const Float3& RGB, const JMhParams& Params)
{
    const Float3 RGB_m = Params.MATRIX_RGB_to_CAM16.transform(RGB);

    const float RedA = PanlrcForward(RGB_m[0] * Params.D_RGB[0], Params.F_L);
    const float GrnA = PanlrcForward(RGB_m[1] * Params.D_RGB[1], Params.F_L);
    const float BluA = PanlrcForward(RGB_m[2] * Params.D_RGB[2], Params.F_L);

    const float A = 2.f * RedA + GrnA + 0.05f * BluA;
    const float a = RedA - 12.f * GrnA / 11.f + BluA / 11.f;
    const float b = (RedA + GrnA - 2.f * BluA) / 9.f;

    const float J = 100.f * ScalarMath::Pow(A / Params.A_w, Surround[1] * Params.z);

    const float M = J == 0.f ? 0.f : 43.f * Surround[2] * ScalarMath::Sqrt(a * a + b * b);

    const float h_rad = ScalarMath::Atan2(b, a);
    float h = ScalarMath::Fmod(h_rad * 180.f / kPi, 360.f);
    if (h < 0.f) {
        h += 360.f;
    }

    return {J, M, h};
}

Table3D MakeGamutTable(const ColorSpace& InLimitingColorSpace, float PeakLuminance)
{
    const JMhParams Params = InitJMhParams(InLimitingColorSpace);

    Table3D GamutCuspTableUnsorted{};
    for (int32 i = 0; i < GamutCuspTableUnsorted.Size; i++) {
        const float hNorm = (float)i / GamutCuspTableUnsorted.Size;
        const Float3 HSV = {hNorm, 1., 1.};
        const Float3 RGB = HSV_to_RGB(HSV);
        const Float3 ScaledRGB = (PeakLuminance / ReferenceLuminance) * RGB;
        const Float3 JMh = RGB_to_JMh(ScaledRGB, Params);

        GamutCuspTableUnsorted.Data[i] = JMh;
    }

    int32 MinhIndex = 0;
    for (int32 i = 0; i < GamutCuspTableUnsorted.Size; i++) {
        if (GamutCuspTableUnsorted.Data[i][2] < GamutCuspTableUnsorted.Data[MinhIndex][2])
            MinhIndex = i;
    }

    Table3D GamutCuspTable{};
    for (int32 i = 0; i < GamutCuspTableUnsorted.Size; i++) {
        GamutCuspTable.Data[i + GamutCuspTable.BaseIndex] =
            GamutCuspTableUnsorted.Data[(MinhIndex + i) % GamutCuspTableUnsorted.Size];
    }

    GamutCuspTable.Data[0] = GamutCuspTable.Data[GamutCuspTable.BaseIndex + GamutCuspTable.Size - 1];

    GamutCuspTable.Data[GamutCuspTable.BaseIndex + GamutCuspTable.Size] = GamutCuspTable.Data[GamutCuspTable.BaseIndex];

    GamutCuspTable.Data[0][2] = GamutCuspTable.Data[0][2] - 360.f;
    GamutCuspTable.Data[GamutCuspTable.Size + 1][2] = GamutCuspTable.Data[GamutCuspTable.Size + 1][2] + 360.f;

    return GamutCuspTable;
}

Float2 CuspToTable(float h, const Table3D& Gt)
{
    int32 Idx_lo = 0;
    int32 Idx_hi = Gt.BaseIndex + Gt.Size;
    int32 Idx = ClampToTableBounds(HuePositionInUniformTable(h, Gt.Size) + Gt.BaseIndex, Gt.TotalSize);

    while (Idx_lo + 1 < Idx_hi) {
        if (h > Gt.Data[Idx][2]) {
            Idx_lo = Idx;
        } else {
            Idx_hi = Idx;
        }

        Idx = ClampToTableBounds((Idx_lo + Idx_hi) / 2, Gt.TotalSize);
    }

    Idx_hi = ScalarMath::Max(1, Idx_hi);

    const Float3 Lo{Gt.Data[Idx_hi - 1][0], Gt.Data[Idx_hi - 1][1], Gt.Data[Idx_hi - 1][2]};

    const Float3 Hi{Gt.Data[Idx_hi][0], Gt.Data[Idx_hi][1], Gt.Data[Idx_hi][2]};

    const float t = (h - Lo[2]) / (Hi[2] - Lo[2]);
    const float CuspJ = ScalarMath::Lerp(Lo[0], Hi[0], t);
    const float CuspM = ScalarMath::Lerp(Lo[1], Hi[1], t);

    return Float2{CuspJ, CuspM};
}

float GetFocusGain(float J, float CuspJ, float LimitJMax)
{
    const float Thr = ScalarMath::Lerp(CuspJ, LimitJMax, FocusGainBlend);

    if (J > Thr) {

        float Gain = (LimitJMax - Thr) / ScalarMath::Max(0.0001f, (LimitJMax - ScalarMath::Min(LimitJMax, J)));
        return ScalarMath::Pow(log10(Gain), 1.f / FocusAdjustGain) + 1.f;
    } else {

        return 1.f;
    }
}

float SolveJIntersect(float J, float M, float FocusJ, float MaxJ, float SlopeGain)
{
    const float a = M / (FocusJ * SlopeGain);
    float b = 0.f;
    float c = 0.f;
    float IntersectJ = 0.f;

    if (J < FocusJ) {
        b = 1.f - M / SlopeGain;
    } else {
        b = -(1.f + M / SlopeGain + MaxJ * M / (FocusJ * SlopeGain));
    }

    if (J < FocusJ) {
        c = -J;
    } else {
        c = MaxJ * M / SlopeGain + J;
    }

    const float Root = ScalarMath::Sqrt(b * b - 4.f * a * c);

    if (J < FocusJ) {
        IntersectJ = 2.f * c / (-b - Root);
    } else {
        IntersectJ = 2.f * c / (-b + Root);
    }

    return IntersectJ;
}

Float3 FindGamutBoundaryIntersection(const Float3& JMh_s, const Float2& JM_cusp_in, float J_focus, float J_max,
                                     float SlopeGain, float gamma_top, float gamma_bottom)
{
    constexpr float s = ScalarMath::Max(0.000001f, SmoothCusps);
    const Float2 JM_cusp = {JM_cusp_in[0], JM_cusp_in[1] * (1.f + SmoothM * s)};

    const float J_intersect_source = SolveJIntersect(JMh_s[0], JMh_s[1], J_focus, J_max, SlopeGain);
    const float J_intersect_cusp = SolveJIntersect(JM_cusp[0], JM_cusp[1], J_focus, J_max, SlopeGain);

    float Slope = 0.f;
    if (J_intersect_source < J_focus) {
        Slope = J_intersect_source * (J_intersect_source - J_focus) / (J_focus * SlopeGain);
    } else {
        Slope = (J_max - J_intersect_source) * (J_intersect_source - J_focus) / (J_focus * SlopeGain);
    }

    const float M_boundary_lower = J_intersect_cusp *
                                   ScalarMath::Pow(J_intersect_source / J_intersect_cusp, 1.f / gamma_bottom) /
                                   (JM_cusp[0] / JM_cusp[1] - Slope);
    const float M_boundary_upper =
        JM_cusp[1] * (J_max - J_intersect_cusp) *
        ScalarMath::Pow((J_max - J_intersect_source) / (J_max - J_intersect_cusp), 1.f / gamma_top) /
        (Slope * JM_cusp[1] + J_max - JM_cusp[0]);
    const float M_boundary = JM_cusp[1] * Smin(M_boundary_lower / JM_cusp[1], M_boundary_upper / JM_cusp[1], s);
    const float J_boundary = J_intersect_source + Slope * M_boundary;

    return {J_boundary, M_boundary, J_intersect_source};
}

bool EvaluateGammaFit(const Float2& JMcusp, const Float3 TestJMh[3], float TopGamma, float PeakLuminance,
                      float LimitJMax, float Mid_J, float FocusDist, float LowerHullGamma,
                      const JMhParams& LimitJMhParams)
{
    const float FocusJ =
        ScalarMath::Lerp(JMcusp[0], Mid_J, ScalarMath::Min(1.f, CuspMidBlend - (JMcusp[0] / LimitJMax)));

    for (size_t TestIndex = 0; TestIndex < 3; TestIndex++) {
        const float SlopeGain = LimitJMax * FocusDist * GetFocusGain(TestJMh[TestIndex][0], JMcusp[0], LimitJMax);
        const Float3 ApproxLimit = FindGamutBoundaryIntersection(TestJMh[TestIndex], JMcusp, FocusJ, LimitJMax,
                                                                 SlopeGain, TopGamma, LowerHullGamma);
        const Float3 Approximate_JMh = {ApproxLimit[0], ApproxLimit[1], TestJMh[TestIndex][2]};
        const Float3 NewLimitRGB = JMh_to_RGB(Approximate_JMh, LimitJMhParams);
        const Float3 NewLimitRGBScaled = (ReferenceLuminance / PeakLuminance) * NewLimitRGB;

        if (!OutsideHull(NewLimitRGBScaled)) {
            return false;
        }
    }

    return true;
}

Table1D MakeUpperHullGamma(const Table3D& GamutCuspTable, float PeakLuminance, float LimitJMax, float Mid_J,
                           float FocusDist, float LowerHullGamma, const JMhParams& LimitJMhParams)
{
    const int32 TestCount = 3;
    const float TestPositions[TestCount] = {0.01f, 0.5f, 0.99f};

    Table1D GammaTable{};
    Table1D GamutTopGamma{};

    for (int32 Index = 0; Index < GammaTable.Size; Index++) {
        GammaTable.Data[Index] = -1.f;

        const float Hue = (float)Index;
        const Float2 JMcusp = CuspToTable(Hue, GamutCuspTable);

        Float3 TestJMh[TestCount]{};
        for (int32 TestIndex = 0; TestIndex < TestCount; TestIndex++) {
            const float TestJ = JMcusp[0] + ((LimitJMax - JMcusp[0]) * TestPositions[TestIndex]);
            TestJMh[TestIndex] = {TestJ, JMcusp[1], Hue};
        }

        const float SearchRange = GammaSearchStep;
        float Low = GammaMinimum;
        float High = Low + SearchRange;
        bool Outside = false;

        while (!(Outside) && (High < 5.f)) {
            const bool bGammaFound = EvaluateGammaFit(JMcusp, TestJMh, High, PeakLuminance, LimitJMax, Mid_J, FocusDist,
                                                      LowerHullGamma, LimitJMhParams);
            if (!bGammaFound) {
                Low = High;
                High = High + SearchRange;
            } else {
                Outside = true;
            }
        }

        float TestGamma = -1.f;
        while ((High - Low) > GammaAccuracy) {
            TestGamma = (High + Low) / 2.f;
            const bool bGammaFound = EvaluateGammaFit(JMcusp, TestJMh, TestGamma, PeakLuminance, LimitJMax, Mid_J,
                                                      FocusDist, LowerHullGamma, LimitJMhParams);
            if (bGammaFound) {
                High = TestGamma;
                GammaTable.Data[Index] = High;
            } else {
                Low = TestGamma;
            }
        }

        GamutTopGamma.Data[Index + GamutTopGamma.BaseIndex] = GammaTable.Data[Index];
    }

    GamutTopGamma.Data[0] = GammaTable.Data[GammaTable.Size - 1];

    GamutTopGamma.Data[GamutTopGamma.TotalSize - 1] = GammaTable.Data[0];

    return GamutTopGamma;
}

} // namespace

ACESTables makeACESTables(float peakNits)
{
    const ColorSpace ap1(Primaries::ACESAP1), ap0(Primaries::ACESAP0);
    const auto reach = MakeReachMTable(peakNits);
    const auto gamut = MakeGamutTable(ap1, peakNits);
    const auto tone = InitToneScaleParams(peakNits);
    const auto input = InitJMhParams(ap0);
    const float logPeak = std::log10(tone.n / tone.n_r);
    const auto gamma = MakeUpperHullGamma(gamut, peakNits, Y_to_J(peakNits, input), Y_to_J(tone.c_t * 100.f, input),
                                          FocusDistance + FocusDistance * FocusDistanceScaling * logPeak,
                                          1.14f + 0.07f * logPeak, InitJMhParams(ap1));
    ACESTables tables;
    std::copy_n(reach.Data, tables.reach.size(), tables.reach.begin());
    for (size_t i = 0; i < 362; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            tables.gamut[i * 3 + c] = gamut.Data[i][int(c)];
        }
    }
    std::copy_n(gamma.Data, tables.gamma.size(), tables.gamma.begin());
    return tables;
}
} // namespace metallic::render
