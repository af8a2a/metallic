// Adapted from Unreal Engine 5.7.4 / Academy ACES. Copyright Epic Games, Inc.
// All Rights Reserved. See Shaders/Licenses/ColorGrading. Forward rendering only.
#language slang 2026
implementing ColorGrading;
namespace Metallic.ColorGrading
{
static const float HALF_MAX = 65504.0;

static const float HALF_POS_INF = 65535.0f;

static const float MIN_STOP_SDR = -6.5;

static const float MAX_STOP_SDR = 6.5;

static const float MIN_STOP_RRT = -15.;

static const float MAX_STOP_RRT = 18.;

static const float MIN_LUM_SDR = 0.02;

static const float MAX_LUM_SDR = 48.0;

static const float MIN_LUM_RRT = 0.0001;

static const float MAX_LUM_RRT = 10000.0;

float pow10(float x)
{
    return pow(10, x);
}

float cubic_basis_shaper
(
    float x,
    float w
)
{
    float M[4][4] = { { -1. / 6,  3. / 6, -3. / 6,  1. / 6 },
                      {  3. / 6, -6. / 6,  3. / 6,  0. / 6 },
                      { -3. / 6,  0. / 6,  3. / 6,  0. / 6 },
                      {  1. / 6,  4. / 6,  1. / 6,  0. / 6 } };

    float knots[5] = { -0.5 * w, -0.25 * w, 0, 0.25 * w, 0.5 * w };

    float y = 0;
    if ((x > knots[0]) && (x < knots[4])) {
        float knot_coord = (x - knots[0]) * 4.0 / w;
        int j = knot_coord;
        float t = knot_coord - j;

        float monomials[4] = { t * t * t, t * t, t, 1.0 };

        if (j == 3) {
            y = monomials[0] * M[0][0] + monomials[1] * M[1][0] +
                monomials[2] * M[2][0] + monomials[3] * M[3][0];
        }
        else if (j == 2) {
            y = monomials[0] * M[0][1] + monomials[1] * M[1][1] +
                monomials[2] * M[2][1] + monomials[3] * M[3][1];
        }
        else if (j == 1) {
            y = monomials[0] * M[0][2] + monomials[1] * M[1][2] +
                monomials[2] * M[2][2] + monomials[3] * M[3][2];
        }
        else if (j == 0) {
            y = monomials[0] * M[0][3] + monomials[1] * M[1][3] +
                monomials[2] * M[2][3] + monomials[3] * M[3][3];
        }
        else {
            y = 0.0;
        }
    }

    return y * 1.5;
}

static const float3x3 M1 =
{
    {  0.5, -1.0, 0.5 },
    { -1.0,  1.0, 0.5 },
    {  0.5,  0.0, 0.0 }
};

float3 limit_to_primaries
(
    float3 XYZ,
    DisplayPrimaries LIMITING_PRI
)
{
    float3x3 XYZ_2_LIMITING_PRI_MAT = LIMITING_PRI.XYZtoRGB;
    float3x3 LIMITING_PRI_2_XYZ_MAT = LIMITING_PRI.RGBtoXYZ;

    float3 rgb = mul(XYZ_2_LIMITING_PRI_MAT, XYZ);

    float3 limitedRgb = clamp(rgb, 0.0.xxx, 1.0.xxx);

    return mul(LIMITING_PRI_2_XYZ_MAT, limitedRgb);
}

float interpolate1D(const float2 table[2], float value)
{
    float t = saturate((value - table[0].x) / (table[1].x - table[0].x));
    return lerp(table[0].y, table[1].y, t);
}

float lookup_ACESmin(float minLum)
{
    const float2 minTable[2] = { float2(log10(MIN_LUM_RRT), MIN_STOP_RRT),
                                 float2(log10(MIN_LUM_SDR), MIN_STOP_SDR) };

    return 0.18 * pow(2., interpolate1D(minTable, log10(minLum)));
}

float lookup_ACESmax(float maxLum)
{
    const float2 maxTable[2] = { float2(log10(MAX_LUM_SDR), MAX_STOP_SDR),
                                   float2(log10(MAX_LUM_RRT), MAX_STOP_RRT) };

    return 0.18 * pow(2., interpolate1D(maxTable, log10(maxLum)));
}

struct TsPoint
{
    float x;
    float y;
    float slope;
};

struct TsParams
{
    TsPoint Min;
    TsPoint Mid;
    TsPoint Max;
    float coefsLow[6];
    float coefsHigh[6];
};

void init_coefsLow(
    TsPoint TsPointLow,
    TsPoint TsPointMid,
    out float coefsLow[5]
)
{

    float knotIncLow = (log10(TsPointMid.x) - log10(TsPointLow.x)) / 3.;

    coefsLow[0] = (TsPointLow.slope * (log10(TsPointLow.x) - 0.5 * knotIncLow)) + (log10(TsPointLow.y) - TsPointLow.slope * log10(TsPointLow.x));
    coefsLow[1] = (TsPointLow.slope * (log10(TsPointLow.x) + 0.5 * knotIncLow)) + (log10(TsPointLow.y) - TsPointLow.slope * log10(TsPointLow.x));

    coefsLow[3] = (TsPointMid.slope * (log10(TsPointMid.x) - 0.5 * knotIncLow)) + (log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));
    coefsLow[4] = (TsPointMid.slope * (log10(TsPointMid.x) + 0.5 * knotIncLow)) + (log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));

    float2 bendsLow[2] = { float2(MIN_STOP_RRT, 0.18),
                           float2(MIN_STOP_SDR, 0.35) };
    float pctLow = interpolate1D(bendsLow, log2(TsPointLow.x / 0.18));
    coefsLow[2] = log10(TsPointLow.y) + pctLow * (log10(TsPointMid.y) - log10(TsPointLow.y));
}

void init_coefsHigh(
    TsPoint TsPointMid,
    TsPoint TsPointMax,
    out float coefsHigh[5]
)
{

    float knotIncHigh = (log10(TsPointMax.x) - log10(TsPointMid.x)) / 3.;

    coefsHigh[0] = (TsPointMid.slope * (log10(TsPointMid.x) - 0.5 * knotIncHigh)) + (log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));
    coefsHigh[1] = (TsPointMid.slope * (log10(TsPointMid.x) + 0.5 * knotIncHigh)) + (log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));

    coefsHigh[3] = (TsPointMax.slope * (log10(TsPointMax.x) - 0.5 * knotIncHigh)) + (log10(TsPointMax.y) - TsPointMax.slope * log10(TsPointMax.x));
    coefsHigh[4] = (TsPointMax.slope * (log10(TsPointMax.x) + 0.5 * knotIncHigh)) + (log10(TsPointMax.y) - TsPointMax.slope * log10(TsPointMax.x));

    float2 bendsHigh[2] = { float2(MAX_STOP_SDR, 0.89) ,
                            float2(MAX_STOP_RRT, 0.90) };
    float pctHigh = interpolate1D(bendsHigh, log2(TsPointMax.x / 0.18));
    coefsHigh[2] = log10(TsPointMid.y) + pctHigh * (log10(TsPointMax.y) - log10(TsPointMid.y));
}

float shift(float inValue, float expShift)
{
    return pow(2., (log2(inValue) - expShift));
}

TsParams init_TsParams(
    float minLum,
    float maxLum,
    float expShift = 0
)
{
    TsPoint MIN_PT = { lookup_ACESmin(minLum), minLum, 0.0 };
    TsPoint MID_PT = { 0.18, 4.8, 1.55 };
    TsPoint MAX_PT = { lookup_ACESmax(maxLum), maxLum, 0.0 };
    float cLow[5];
    init_coefsLow(MIN_PT, MID_PT, cLow);
    float cHigh[5];
    init_coefsHigh(MID_PT, MAX_PT, cHigh);
    MIN_PT.x = shift(lookup_ACESmin(minLum), expShift);
    MID_PT.x = shift(0.18, expShift);
    MAX_PT.x = shift(lookup_ACESmax(maxLum), expShift);

    TsParams P = {
        {MIN_PT.x, MIN_PT.y, MIN_PT.slope},
        {MID_PT.x, MID_PT.y, MID_PT.slope},
        {MAX_PT.x, MAX_PT.y, MAX_PT.slope},
        {cLow[0], cLow[1], cLow[2], cLow[3], cLow[4], cLow[4]},
        {cHigh[0], cHigh[1], cHigh[2], cHigh[3], cHigh[4], cHigh[4]}
    };

    return P;
}

float ssts
(
    const float x,
    const TsParams C
)
{
    const int N_KNOTS_LOW = 4;
    const int N_KNOTS_HIGH = 4;

    float logx = log10(max(x, 1e-10));
    float logy;

    if (logx <= log10(C.Min.x)) {

        logy = logx * C.Min.slope + (log10(C.Min.y) - C.Min.slope * log10(C.Min.x));

    }
    else if ((logx > log10(C.Min.x)) && (logx < log10(C.Mid.x))) {

        float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(C.Min.x)) / (log10(C.Mid.x) - log10(C.Min.x));
        int j = knot_coord;
        float t = knot_coord - j;

        float3 cf = { C.coefsLow[j], C.coefsLow[j + 1], C.coefsLow[j + 2] };

        float3 monomials = { t * t, t, 1.0 };
        logy = dot(monomials, mul(cf, M1));

    }
    else if ((logx >= log10(C.Mid.x)) && (logx < log10(C.Max.x))) {

        float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(C.Mid.x)) / (log10(C.Max.x) - log10(C.Mid.x));
        int j = knot_coord;
        float t = knot_coord - j;

        float3 cf = { C.coefsHigh[j], C.coefsHigh[j + 1], C.coefsHigh[j + 2] };

        float3 monomials = { t * t, t, 1.0 };
        logy = dot(monomials, mul(cf, M1));

    }
    else {

        logy = logx * C.Max.slope + (log10(C.Max.y) - C.Max.slope * log10(C.Max.x));

    }

    return pow10(logy);

}

float inv_ssts
(
    const float y,
    const TsParams C
)
{
    const int N_KNOTS_LOW = 4;
    const int N_KNOTS_HIGH = 4;

    const float KNOT_INC_LOW = (log10(C.Mid.x) - log10(C.Min.x)) / (N_KNOTS_LOW - 1.);
    const float KNOT_INC_HIGH = (log10(C.Max.x) - log10(C.Mid.x)) / (N_KNOTS_HIGH - 1.);

    float KNOT_Y_LOW[N_KNOTS_LOW];

    {
        for (int i = 0; i < N_KNOTS_LOW; i = i + 1) {
            KNOT_Y_LOW[i] = (C.coefsLow[i] + C.coefsLow[i + 1]) / 2.;
        };
    }

    float KNOT_Y_HIGH[N_KNOTS_HIGH];

    {
        for (int i = 0; i < N_KNOTS_HIGH; i = i + 1) {
            KNOT_Y_HIGH[i] = (C.coefsHigh[i] + C.coefsHigh[i + 1]) / 2.;
        };
    }

    float logy = log10(max(y, 1e-10));

    float logx;
    if (logy <= log10(C.Min.y)) {

        logx = log10(C.Min.x);

    }
    else if ((logy > log10(C.Min.y)) && (logy <= log10(C.Mid.y))) {

        int j;
        float3 cf;
        if (logy > KNOT_Y_LOW[0] && logy <= KNOT_Y_LOW[1]) {
            cf[0] = C.coefsLow[0];  cf[1] = C.coefsLow[1];  cf[2] = C.coefsLow[2];  j = 0;
        }
        else if (logy > KNOT_Y_LOW[1] && logy <= KNOT_Y_LOW[2]) {
            cf[0] = C.coefsLow[1];  cf[1] = C.coefsLow[2];  cf[2] = C.coefsLow[3];  j = 1;
        }
        else if (logy > KNOT_Y_LOW[2] && logy <= KNOT_Y_LOW[3]) {
            cf[0] = C.coefsLow[2];  cf[1] = C.coefsLow[3];  cf[2] = C.coefsLow[4];  j = 2;
        }

        const float3 tmp = mul(cf, M1);

        float a = tmp[0];
        float b = tmp[1];
        float c = tmp[2];
        c = c - logy;

        const float d = sqrt(b * b - 4. * a * c);

        const float t = (2. * c) / (-d - b);

        logx = log10(C.Min.x) + (t + j) * KNOT_INC_LOW;

    }
    else if ((logy > log10(C.Mid.y)) && (logy < log10(C.Max.y))) {

        int j;
        float3 cf;
        if (logy >= KNOT_Y_HIGH[0] && logy <= KNOT_Y_HIGH[1]) {
            cf[0] = C.coefsHigh[0];  cf[1] = C.coefsHigh[1];  cf[2] = C.coefsHigh[2];  j = 0;
        }
        else if (logy > KNOT_Y_HIGH[1] && logy <= KNOT_Y_HIGH[2]) {
            cf[0] = C.coefsHigh[1];  cf[1] = C.coefsHigh[2];  cf[2] = C.coefsHigh[3];  j = 1;
        }
        else if (logy > KNOT_Y_HIGH[2] && logy <= KNOT_Y_HIGH[3]) {
            cf[0] = C.coefsHigh[2];  cf[1] = C.coefsHigh[3];  cf[2] = C.coefsHigh[4];  j = 2;
        }

        const float3 tmp = mul(cf, M1);

        float a = tmp[0];
        float b = tmp[1];
        float c = tmp[2];
        c = c - logy;

        const float d = sqrt(b * b - 4. * a * c);

        const float t = (2. * c) / (-d - b);

        logx = log10(C.Mid.x) + (t + j) * KNOT_INC_HIGH;

    }
    else {

        logx = log10(C.Max.x);

    }

    return pow10(logx);

}

float3 ssts_f3
(
    const float3 x,
    const TsParams C
)
{
    float3 outValue;
    outValue[0] = ssts(x[0], C);
    outValue[1] = ssts(x[1], C);
    outValue[2] = ssts(x[2], C);

    return outValue;
}

static const float RRT_SAT_FACTOR = 0.96;

static const float RRT_GLOW_GAIN = 0.05;

static const float RRT_GLOW_MID = 0.08;

static const float RRT_RED_SCALE = 0.82;

static const float RRT_RED_PIVOT = 0.03;

static const float RRT_RED_HUE = 0.;

static const float RRT_RED_WIDTH = 135.;

float3 rrt_sweeteners(float3 inValue)
{
    float3 aces = inValue;

    float saturation = rgb_2_saturation(aces);
    float ycIn = rgb_2_yc(aces);
    float s = sigmoid_shaper((saturation - 0.4) / 0.2);
    float addedGlow = 1 + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
    aces *= addedGlow;

    float hue = rgb_2_hue(aces);
    float centeredHue = center_hue(hue, RRT_RED_HUE);
    float hueWeight = cubic_basis_shaper(centeredHue, RRT_RED_WIDTH);

    aces.r += hueWeight * saturation * (RRT_RED_PIVOT - aces.r) * (1. - RRT_RED_SCALE);

    aces = clamp(aces, 0, HALF_POS_INF);
    float3 rgbPre = mul(AP0_2_AP1_MAT, aces);
    rgbPre = clamp(rgbPre, 0, HALF_MAX);

    rgbPre = lerp(dot(rgbPre, AP1_RGB2Y).xxx, rgbPre, RRT_SAT_FACTOR);
    return rgbPre;
}

float Y_2_linCV(float Y, float Ymax, float Ymin)
{
    return (Y - Ymin) / (Ymax - Ymin);
}

float linCV_2_Y(float linCV, float Ymax, float Ymin)
{
    return linCV * (Ymax - Ymin) + Ymin;
}

float3 Y_2_linCV_f3(float3 Y, float Ymax, float Ymin)
{
    float3 linCV;
    linCV[0] = Y_2_linCV(Y[0], Ymax, Ymin);
    linCV[1] = Y_2_linCV(Y[1], Ymax, Ymin);
    linCV[2] = Y_2_linCV(Y[2], Ymax, Ymin);
    return linCV;
}

float3 linCV_2_Y_f3(float3 linCV, float Ymax, float Ymin)
{
    float3 Y;
    Y[0] = linCV_2_Y(linCV[0], Ymax, Ymin);
    Y[1] = linCV_2_Y(linCV[1], Ymax, Ymin);
    Y[2] = linCV_2_Y(linCV[2], Ymax, Ymin);
    return Y;
}

float3 aces1Nits(float3 aces, float peak)
{
    TsParams params = init_TsParams(1e-4, peak);
    float expShift = log2(inv_ssts(15.0, params)) - log2(0.18);
    params = init_TsParams(1e-4, peak, expShift);
    float3 rgb = ssts_f3(rrt_sweeteners(aces), params);
    float3 linear = Y_2_linCV_f3(rgb, peak, 1e-4);
    float3 xyz = mul(AP1_2_XYZ_MAT, linear);
    DisplayPrimaries ap1 = {WhiteStandard_D60, XYZ_2_AP1_MAT, AP1_2_XYZ_MAT};
    xyz = limit_to_primaries(xyz, ap1);
    linear = max(mul(XYZ_2_AP1_MAT, xyz), 0.0);
    return clamp(linCV_2_Y_f3(linear, peak, 0.0), 0.0, HALF_POS_INF);
}
}
