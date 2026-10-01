// Adapted from Unreal Engine 5.7.4 / Academy ACES. Copyright Epic Games, Inc.
// All Rights Reserved. See Shaders/Licenses/ColorGrading. Forward rendering only.
#language slang 2026
implementing ColorGrading;
namespace Metallic.ColorGrading
{
float radians_to_degrees( float radians )
{
    return radians * 180.0 / PI;
}

float degrees_to_radians( float degrees )
{
    return degrees / 180.0 * PI;
}

float copysign(float x, float y)
{
    return sign(y) * abs(x);
}

float3 copysign(float3 x, float3 y)
{
    return sign(y) * abs(x);
}

float smin(float a, float b, float s)
{
    float h = max(s - abs(a - b), 0.0) / s;
    return min(a, b) - h * h * h * s * (1.0 / 6.0);
}

float spow(float base, float exponent)
{
    return (base < 0.0 && exponent != floor(exponent)) ? 0.0 : pow(base, exponent);
}

float3 spow(float3 base, float3 exponent)
{
    return float3(
        spow(base[0], exponent[0]),
        spow(base[1], exponent[1]),
        spow(base[2], exponent[2])
    );
}

float sdiv(float a, float b)
{
    return b == 0.0 ? 0.0 : a / b;
}

struct TSParams
{
    float n;
    float n_r;
    float g;
    float t_1;
    float c_t;
    float s_2;
    float u_2;
    float m_2;
};

TSParams init_TSParams(float peakLuminance) {

    const float n = peakLuminance;

    static const float n_r = 100.0;
    static const float g = 1.15;
    static const float c = 0.18;
    static const float c_d = 10.013;
    static const float w_g = 0.14;
    static const float t_1 = 0.04;
    static const float r_hit_min = 128.;
    static const float r_hit_max = 896.;

    const float r_hit = r_hit_min + (r_hit_max - r_hit_min) * (log(n/n_r)/log(10000./100.));
    const float m_0 = (n / n_r);
    const float m_1 = 0.5 * (m_0 + sqrt(m_0 * (m_0 + 4. * t_1)));
    const float u = pow((r_hit/m_1)/((r_hit/m_1)+1),g);
    const float m = m_1 / u;
    const float w_i = log(n/100.)/log(2.);
    const float c_t = c_d/n_r * (1. + w_i * w_g);
    const float g_ip = 0.5 * (c_t + sqrt(c_t * (c_t + 4. * t_1)));
    const float g_ipp2 = -(m_1 * pow((g_ip/m),(1./g))) / (pow( g_ip/m , 1./g)-1.);
    const float w_2 = c / g_ipp2;
    const float s_2 = w_2 * m_1;
    const float u_2 = pow( (r_hit/m_1)/((r_hit/m_1) + w_2), g);
    const float m_2 = m_1 / u_2;

    TSParams TonescaleParams = {
        n,
        n_r,
        g,
        t_1,
        c_t,
        s_2,
        u_2,
        m_2
    };

    return TonescaleParams;
}

float tonescale_fwd(
    float x,
    TSParams params
)
{

    float f = params.m_2 * pow( max(0.0, x) / (x + params.s_2), params.g);
    float h = max(0., f * f / (f + params.t_1));

    return h * params.n_r;
}

static const float smoothCusps = 0.12;

static const float smoothM = 0.27;

static const float cuspMidBlend = 1.3;

static const float focusGainBlend = 0.3;

static const float focusAdjustGain = 0.55;

static const float focusDistance = 1.35;

static const float focusDistanceScaling = 1.75;

static const float compressionThreshold = 0.75;

static const int tableSize = 360;

static const int baseIndex = 1;

static const float referenceLuminance = 100.;

static const float L_A = 100.;

static const float Y_b = 20.;

static const float ac_resp = 1.0;

static const float ra = 2. * ac_resp;

static const float ba = 0.05 + (2. - ra);

static const float3 surround = float3(0.9, 0.59, 0.9);

static const float3x3 panlrcm =
{
    460.0,  451.0,  288.0 ,
    460.0, -891.0, -261.0 ,
    460.0, -220.0, -6300.0
};

struct ODTParams
{
    float peakLuminance;

    float n_r;
    float g;
    float t_1;
    float c_t;
    float s_2;
    float u_2;
    float m_2;

    float limitJmax;
    float midJ;
    float model_gamma;
    float sat;
    float sat_thr;
    float compr;
    float chromaCompressScale;

    float focusDist;

    float3x3 LIMIT_RGB_TO_XYZ;
    float3x3 LIMIT_XYZ_TO_RGB;
    float3 XYZ_w_limit;

    float3x3 OUTPUT_RGB_TO_XYZ;
    float3x3 OUTPUT_XYZ_TO_RGB;
    float3 XYZ_w_output;

    float lowerHullGamma;

    Texture2D<float> reach_table;
    Texture2D<float> gamut_table;
    Texture2D<float> gamma_table;
};

float wrap_to_360(float hue)
{
    float y = fmod(hue, 360.);
    if (y < 0.)
    {
        y = y + 360.;
    }
    return y;
}

int hue_position_in_uniform_table(float hue, int table_size)
{
    const float wrapped_hue = wrap_to_360(hue);
    int result = (wrapped_hue / 360. * table_size);
    return result;
}

int next_position_in_table(int entry, int table_size)
{
    int result = (entry + 1) % table_size;
    return result;
}

float base_hue_for_position(int i_lo, int table_size)
{
    float result = i_lo * 360. / table_size;
    return result;
}

float3 post_adaptation_non_linear_response_compression_forward(float3 RGB, float F_L)
{
    float3 F_L_RGB = spow(F_L * abs(RGB) / 100., 0.42);
    float3 RGB_c = (400. * copysign(1., RGB) * F_L_RGB) / (27.13 + F_L_RGB);

    return RGB_c;
}

float3 post_adaptation_non_linear_response_compression_inverse(float3 RGB, float F_L)
{
    float3 RGB_p = sign(RGB) * 100. / F_L * spow((27.13 * abs(RGB)) / (400. - abs(RGB)), 1.0 / 0.42);

    return RGB_p;
}

float3 XYZ_to_Hellwig2022_JMh(  float3 XYZ,
                                float3 XYZ_w)
{

    float Y_w = XYZ_w[1];

    float3 RGB_w = mul(XYZ_2_CAM16_MAT, XYZ_w);

    float D = 1.0;

    float k = 1. / (5. * L_A + 1.);
    float k4 = pow(k, 4.);
    float F_L = 0.2 * k4 * (5. * L_A) + 0.1 * pow((1. - k4), 2.) * spow(5. * L_A, 1. / 3.);
    float n = Y_b / Y_w;
    float z = 1.48 + sqrt(n);

    float3 D_RGB = D * Y_w / RGB_w + 1. - D;

    float3 RGB_wc = D_RGB * RGB_w;

    float3 RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L);
    float A_w = ra * RGB_aw[0] + RGB_aw[1] + ba * RGB_aw[2];

    float3 RGB = mul(XYZ_2_CAM16_MAT, XYZ);

    float3 RGB_c = D_RGB * RGB;

    float3 RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L);

    float a = RGB_a[0] - 12. * RGB_a[1] / 11. + RGB_a[2] / 11.;
    float b = (RGB_a[0] + RGB_a[1] - 2. * RGB_a[2]) / 9.;

    float hr = atan2(b, a);
    float h = wrap_to_360(radians_to_degrees(hr));

    float A = ra * RGB_a[0] + RGB_a[1] + ba * RGB_a[2];

    float J = 100. * pow(A / A_w, surround[1] * z);

    float M = (J == 0.0) ? 0.0 : 43. * surround[2] * sqrt(a * a + b * b);

    float3 return_JMh = float3(J, M, h);

    return return_JMh;
}

float3 Hellwig2022_JMh_to_XYZ(  float3 JMh,
                                float3 XYZ_w)
{
    float J = JMh[0];
    float M = JMh[1];
    float h = JMh[2];

    float Y_w = XYZ_w[1];

    float3 RGB_w = mul(XYZ_2_CAM16_MAT, XYZ_w);

    float D = 1.0;

    float k = 1.0 / (5.0 * L_A + 1.0);
    float k4 = pow(k, 4.0);
    float F_L = 0.2 * k4 * (5.0 * L_A) + 0.1 * pow((1.0 - k4), 2.0) * spow(5.0 * L_A, 1.0 / 3.0);
    float n = sdiv(Y_b, Y_w);
    float z = 1.48 + sqrt(n);

    float3 D_RGB = D * Y_w / RGB_w + 1.0 - D;

    float3 RGB_wc = D_RGB * RGB_w;

    float3 RGB_aw = post_adaptation_non_linear_response_compression_forward(RGB_wc, F_L);

    float A_w = ra * RGB_aw[0] + RGB_aw[1] + ba * RGB_aw[2];

    float hr = degrees_to_radians(h);

    float A = A_w * spow(J / 100.0, 1.0 / (surround[1] * z));

    float P_p_1 = 43.0 * surround[2];
    float P_p_2 = A;

    float gamma = M / P_p_1;
    float a = gamma * cos(hr);
    float b = gamma * sin(hr);

    float3 vec = {P_p_2, a, b};
    float3 RGB_a = (1.0 / 1403.0) * mul(panlrcm, vec);

    float3 RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L);

    float3 RGB = RGB_c / D_RGB;

    float3 XYZ = mul(CAM16_2_XYZ_MAT, RGB);

    return XYZ;
}

float Hellwig_J_to_Y(float J,
                     float surround = 0.59,
                     float L_A = 100.,
                     float Y_b = 20.)
{

    float k = 1.0 / (5. * L_A + 1.);
    float k4 = k * k * k * k;
    float F_L = 0.2 * k4 * (5. * L_A) + 0.1 * pow((1. - k4), 2.) * pow(5. * L_A, 1. / 3.);
    float n = Y_b / 100.;
    float z = 1.48 + sqrt(n);
    float F_L_W = pow(F_L, 0.42);
    float A_w = (400. * F_L_W) / (27.13 + F_L_W);

    float A = A_w * pow(abs(J) / 100., 1. / (surround * z));

    return sign(J) * 100. / F_L * pow((27.13 * A) / (400.0 - A), 1. / 0.42);
}

float Y_to_Hellwig_J(float Y,
                     float surround = 0.59,
                     float L_A = 100.,
                     float Y_b = 20.)
{

    float k = 1.0 / (5. * L_A + 1.);
    float k4 = k * k * k * k;
    float F_L = 0.2 * k4 * (5. * L_A) + 0.1 * pow((1. - k4), 2.) * pow(5. * L_A, 1. / 3.);
    float n = Y_b / 100.;
    float z = 1.48 + sqrt(n);
    float F_L_W = pow(F_L, 0.42);
    float A_w = (400. * F_L_W) / (27.13 + F_L_W);

    float F_L_Y = pow(F_L * abs(Y) / 100., 0.42);

    return sign(Y) * 100. * pow(((400. * F_L_Y) / (27.13 + F_L_Y)) / A_w, surround * z);
}

float3 clamp_AP1(float3 AP1, float peakLuminance)
{
    const float upper_clamp_limit = 8. * (128 + 768 * (log(peakLuminance/100.)/log(10000./100.)));

    float3 ap1_clamped = clamp(AP1, 0., upper_clamp_limit);
    float3 XYZ_clamped = mul(AP1_2_XYZ_MAT, ap1_clamped);

    return XYZ_clamped;
}

float3 aces_to_JMh(float3 aces,
                     float peakLuminance)
{

    float3 AP1 = mul(AP0_2_AP1_MAT, aces);

    float3 XYZ = clamp_AP1(AP1, peakLuminance);

    const float3 RGB_w = referenceLuminance;
    float3 XYZ_w = mul(AP0_2_XYZ_MAT, RGB_w);

    float3 XYZluminance = referenceLuminance * XYZ;
    float3 JMh = XYZ_to_Hellwig2022_JMh(XYZluminance, XYZ_w);

    return JMh;
}

ODTParams init_ODTParams(
    float peakLuminance,
    float3x3 LIMIT_RGB_TO_XYZ,
    float3x3 LIMIT_XYZ_TO_RGB,
    float3x3 OUTPUT_RGB_TO_XYZ,
    float3x3 OUTPUT_XYZ_TO_RGB,
    Texture2D<float> reach_table,
    Texture2D<float> gamut_table,
    Texture2D<float> gamma_table
)
{
    TSParams TSPARAMS = init_TSParams(peakLuminance);

    float limitJmax = Y_to_Hellwig_J(peakLuminance);
    float midJ = Y_to_Hellwig_J(TSPARAMS.c_t * 100.);

    static const float chroma_compress = 2.4;
    static const float chroma_compress_fact = 3.3;
    static const float chroma_expand = 1.3;
    static const float chroma_expand_fact = 0.69;
    static const float chroma_expand_thr = 0.5;

    const float log_peak = log10(TSPARAMS.n / TSPARAMS.n_r);
    const float compr = chroma_compress + (chroma_compress * chroma_compress_fact) * log_peak;
    const float sat = max(0.2, chroma_expand - (chroma_expand * chroma_expand_fact) * log_peak);
    const float sat_thr = chroma_expand_thr / TSPARAMS.n;
    const float chromaCompressScale = pow(0.03379 * TSPARAMS.n, 0.30596) - 0.45135;

    const float model_gamma = 1. / (surround[1] * (1.48 + sqrt(Y_b / L_A)));

    const float focusDist = focusDistance +
                            focusDistance * focusDistanceScaling * log_peak;

    static const float3 RGB_w = float3(100., 100., 100.);

    const float lowerHullGamma = 1.14 + 0.07 * log_peak;

    float3 XYZ_w_limit = mul(LIMIT_RGB_TO_XYZ, RGB_w);

    float3 XYZ_w_output = mul(OUTPUT_RGB_TO_XYZ, RGB_w);

    const ODTParams ODTPARAMS = {
        peakLuminance,

        TSPARAMS.n_r,
        TSPARAMS.g,
        TSPARAMS.t_1,
        TSPARAMS.c_t,
        TSPARAMS.s_2,
        TSPARAMS.u_2,
        TSPARAMS.m_2,

        limitJmax,
        midJ,
        model_gamma,
        sat,
        sat_thr,
        compr,
        chromaCompressScale,

        focusDist,

        LIMIT_RGB_TO_XYZ,
        LIMIT_XYZ_TO_RGB,
        XYZ_w_limit,

        OUTPUT_RGB_TO_XYZ,
        OUTPUT_XYZ_TO_RGB,
        XYZ_w_output,

        lowerHullGamma,

        reach_table,
        gamut_table,
        gamma_table
    };

    return ODTPARAMS;
}

float chromaCompressionNorm(float h,
                            ODTParams PARAMS)
{

    float hr = degrees_to_radians(h);

    float a = cos(hr);
    float b = sin(hr);
    float cos_hr2 = a * a - b * b;
    float sin_hr2 = 2.0 * a * b;
    float cos_hr3 = 4.0 * a * a * a - 3.0 * a;
    float sin_hr3 = 3.0 * b - 4.0 * b * b * b;

    float M = 11.34072 * a +
              16.46899 * cos_hr2 +
              7.88380 * cos_hr3 +
              14.66441 * b +
              -6.37224 * sin_hr2 +
              9.19364 * sin_hr3 +
              77.12896;

    return M * PARAMS.chromaCompressScale;
}

float toe_fwd(float x,
          float limit,
          float k1_in,
          float k2_in)
{
    if (x > limit)
        return x;

    const float k2 = max(k2_in, 0.001);
    const float k1 = sqrt(k1_in * k1_in + k2 * k2);
    const float k3 = (limit + k1) / (limit + k2);
    const float minus_b = k3 * x - k1;
    const float minus_c = k2 * k3 * x;

    return 0.5 * (minus_b + sqrt(minus_b * minus_b + 4. * minus_c));
}

float toe_inv(float x,
          float limit,
          float k1_in,
          float k2_in)
{
    if (x > limit)
        return x;

    float k2 = max(k2_in, 0.001);
    float k1 = sqrt(k1_in * k1_in + k2 * k2);
    float k3 = (limit + k1) / (limit + k2);

    return (x * x + k1 * x) / (k3 * (x + k2));
}

float reachMFromTable(float h,
                      Texture2D<float> reach_table)
{
    int i_lo = hue_position_in_uniform_table(h, tableSize);
    int i_hi = next_position_in_table(i_lo, tableSize);

    float t = (h - i_lo) / (i_hi - i_lo);

    return lerp(reach_table.Load(int3(i_lo, 0, 0)), reach_table.Load(int3(i_hi, 0, 0)), t);
}

float chromaCompression(float3 JMh,
                        float origJ,
                        ODTParams PARAMS,
                        bool _invert = false)
{
    float J = JMh[0];
    float M = JMh[1];
    float h = JMh[2];

    if (M == 0.0)
    {
        return M;
    }

    float nJ = J / PARAMS.limitJmax;
    float snJ = max(0., 1. - nJ);
    float Mnorm = chromaCompressionNorm(h, PARAMS);
    float limit = pow(nJ, PARAMS.model_gamma) * reachMFromTable(h, PARAMS.reach_table) / Mnorm;

    float toe_limit = limit - 0.001;
    float toe_snJ_sat = snJ * PARAMS.sat;
    float toe_sqrt_nJ_sat_thr = sqrt(nJ * nJ + PARAMS.sat_thr);
    float toe_nJ_compr = nJ * PARAMS.compr;

    if (!_invert) {

        M = M * pow(J / origJ, PARAMS.model_gamma);

        M = M / Mnorm;

        M = limit - toe_fwd(limit - M,
                        toe_limit,
                        toe_snJ_sat,
                        toe_sqrt_nJ_sat_thr);

        M = toe_fwd(M,
                limit,
                toe_nJ_compr,
                snJ);

        M = M * Mnorm;
    }
    else
    {
        M = M / Mnorm;
        M = toe_inv(M,
                limit,
                toe_nJ_compr,
                snJ);
        M = limit - toe_inv(limit - M,
                        toe_limit,
                        toe_snJ_sat,
                        toe_sqrt_nJ_sat_thr);
        M = M * Mnorm;
        M = M * pow(J / origJ, -PARAMS.model_gamma);
    }

    return M;
}

float compressionFunction(float v,
                          float thr,
                          float lim,
                          bool _invert = false)
{
    float s = (lim - thr) * (1.0 - thr) / (lim - 1.0);
    float nd = (v - thr) / s;

    float vCompressed;

    if (_invert)
    {
        if (v < thr || lim <= 1.0001 || v > thr + s)
        {
            vCompressed = v;
        }
        else
        {
            vCompressed = thr + s * (-nd / (nd - 1));
        }
    }
    else
    {
        if (v < thr || lim <= 1.0001)
        {
            vCompressed = v;
        }
        else
        {
            vCompressed = thr + s * nd / (1.0 + nd);
        }
    }

    return vCompressed;
}

int midpoint(int low, int high)
{
    return (low + high) / 2;
}

float2 cuspFromTable(float h, Texture2D<float> gamut_table)
{
    float3 lo;
    float3 hi;

    int low_i = 0;
    int high_i = baseIndex + tableSize;
    int i = hue_position_in_uniform_table(h, tableSize) + baseIndex;

    while (low_i + 1 < high_i)
    {
        if (h > gamut_table.Load(int3(2, i, 0)))
        {
            low_i = i;
        }
        else
        {
            high_i = i;
        }
        i = midpoint(low_i, high_i);
    }
    lo = float3(gamut_table.Load(int3(0, high_i - 1, 0)),
        gamut_table.Load(int3(1, high_i - 1, 0)),
        gamut_table.Load(int3(2, high_i - 1, 0)));
    hi = float3(gamut_table.Load(int3(0, high_i, 0)),
        gamut_table.Load(int3(1, high_i, 0)),
        gamut_table.Load(int3(2, high_i, 0)));

    float t = (h - lo[2]) / (hi[2] - lo[2]);
    float cuspJ = lerp(lo[0], hi[0], t);
    float cuspM = lerp(lo[1], hi[1], t);

    float2 cuspJM = float2(cuspJ, cuspM);

    return cuspJM;
}

float3 tonemapAndCompress_fwd(float3 inputJMh,
                                ODTParams PARAMS)
{
    float3 outputJMh;

    float _linear = Hellwig_J_to_Y(inputJMh[0]) / referenceLuminance;

    TSParams TSPARAMS = init_TSParams(PARAMS.peakLuminance);
    float luminanceTS = tonescale_fwd(_linear, TSPARAMS);

    float tonemappedJ = Y_to_Hellwig_J(luminanceTS);

    float3 tonemappedJMh = float3(tonemappedJ, inputJMh[1], inputJMh[2]);

    outputJMh = tonemappedJMh;
    outputJMh[1] = chromaCompression(outputJMh,
                                     inputJMh[0],
                                     PARAMS,
                                     false);

    return outputJMh;
}

float3 JMh_to_output_XYZ(float3 JMh,
                           ODTParams PARAMS)
{
    float3 XYZluminance = Hellwig2022_JMh_to_XYZ(JMh,
                                                   PARAMS.XYZ_w_limit);

    float3 XYZ = (1. / referenceLuminance) * XYZluminance;

    return XYZ;
}

float hueDependentUpperHullGamma(float h,
                                 Texture2D<float> gamma_table)
{
    uint3 gammaSize;
    gamma_table.GetDimensions(0, gammaSize.x, gammaSize.y, gammaSize.z);

    const int i_lo = hue_position_in_uniform_table(h, tableSize) + baseIndex;
    const int i_hi = next_position_in_table(i_lo, (int)gammaSize.x);

    const float base_hue = base_hue_for_position(i_lo - baseIndex, tableSize);

    const float t = wrap_to_360(h) - base_hue;

    return lerp(gamma_table.Load(int3(i_lo, 0, 0)), gamma_table.Load(int3(i_hi, 0, 0)), t);
}

float getFocusGain(float J,
                   float cuspJ,
                   float limitJmax)
{

    float thr = lerp(cuspJ, limitJmax, focusGainBlend);
    if (J > thr)
    {

        float gain = (limitJmax - thr) / max(0.0001, limitJmax - J);
        return pow(log10(gain), 1. / focusAdjustGain) + 1.;
    }
    else
    {

        return 1.;
    }
}

float solve_J_intersect(float J,
                        float M,
                        float focusJ,
                        float maxJ,
                        float slope_gain)
{
    float a = M / (focusJ * slope_gain);
    float b = 0.0;
    float c = 0.0;
    float intersectJ = 0.0;

    if (J < focusJ)
    {
        b = 1.0 - M / slope_gain;
    }
    else
    {
        b = -(1.0 + M / slope_gain + maxJ * M / (focusJ * slope_gain));
    }

    if (J < focusJ)
    {
        c = -J;
    }
    else
    {
        c = maxJ * M / slope_gain + J;
    }

    float root = sqrt(b * b - 4.0 * a * c);

    if (J < focusJ)
    {
        intersectJ = 2.0 * c / (-b - root);
    }
    else
    {
        intersectJ = 2.0 * c / (-b + root);
    }

    return intersectJ;
}

float3 findGamutBoundaryIntersection(float3 JMh_s,
                                       float2 JM_cusp_in,
                                       float J_focus,
                                       float J_max,
                                       float slope_gain,
                                       float gamma_top,
                                       float gamma_bottom)
{
    float slope = 0.0;

    float s = max(0.000001, smoothCusps);
    float2 JM_cusp = JM_cusp_in;
    JM_cusp[1] = JM_cusp_in[1] * (1.0 + smoothM * s);

    float J_intersect_source = solve_J_intersect(JMh_s[0],
                                                 JMh_s[1],
                                                 J_focus,
                                                 J_max,
                                                 slope_gain);
    float J_intersect_cusp = solve_J_intersect(JM_cusp[0],
                                               JM_cusp[1],
                                               J_focus,
                                               J_max,
                                               slope_gain);

    if (J_intersect_source < J_focus)
    {
        slope = J_intersect_source * (J_intersect_source - J_focus) / (J_focus * slope_gain);
    }
    else
    {
        slope = (J_max - J_intersect_source) * (J_intersect_source - J_focus) / (J_focus * slope_gain);
    }

    float M_boundary_lower = J_intersect_cusp * pow(J_intersect_source / J_intersect_cusp, 1. / gamma_bottom) / (JM_cusp[0] / JM_cusp[1] - slope);

    float M_boundary_upper = JM_cusp[1] * (J_max - J_intersect_cusp) * pow((J_max - J_intersect_source) / (J_max - J_intersect_cusp), 1. / gamma_top) / (slope * JM_cusp[1] + J_max - JM_cusp[0]);

    float M_boundary = JM_cusp[1] * smin(M_boundary_lower / JM_cusp[1], M_boundary_upper / JM_cusp[1], s);

    float J_boundary = J_intersect_source + slope * M_boundary;

    float3 return_JMh = {J_boundary, M_boundary, J_intersect_source};

    return return_JMh;
}

float3 getReachBoundary(float J,
                          float M,
                          float h,
                          ODTParams PARAMS,
                          float2 JMcusp,
                          float focusJ)
{
    float limitJmax = PARAMS.limitJmax;
    float midJ = PARAMS.midJ;
    float model_gamma = PARAMS.model_gamma;
    float focusDist = PARAMS.focusDist;

    const float reachMaxM = reachMFromTable(h,  PARAMS.reach_table);

    float slope_gain = limitJmax * focusDist * getFocusGain(J, JMcusp[0], limitJmax);

    float intersectJ = solve_J_intersect(J, M, focusJ, limitJmax, slope_gain);
    float slope;
    if (intersectJ < focusJ)
    {
        slope = intersectJ * (intersectJ - focusJ) / (focusJ * slope_gain);
    }
    else
    {
        slope = (limitJmax - intersectJ) * (intersectJ - focusJ) / (focusJ * slope_gain);
    }
    float boundary = limitJmax * pow(intersectJ / limitJmax, model_gamma) * reachMaxM / (limitJmax - slope * reachMaxM);
    float3 result = {J, boundary, h};
    return result;
}

float3 compressGamut(float3 JMh,
                       ODTParams PARAMS,
                       float Jx,
                       bool _invert = false)
{
    float limitJmax = PARAMS.limitJmax;
    float midJ = PARAMS.midJ;
    float focusDist = PARAMS.focusDist;
    float model_gamma = PARAMS.model_gamma;

    float2 project_from = float2(JMh[0], JMh[1]);
    float2 JMcusp = cuspFromTable(JMh[2], PARAMS.gamut_table);

    if (JMh[1] < 0.0001 || JMh[0] > limitJmax)
    {
        float3 JMh_return = float3(JMh[0], 0.0, JMh[2]);
        return JMh_return;
    }

    float focusJ = lerp(JMcusp[0], midJ, min(1., cuspMidBlend - (JMcusp[0] / limitJmax)));

    float slope_gain = limitJmax * focusDist * getFocusGain(Jx, JMcusp[0], limitJmax);

    float gamma_top = hueDependentUpperHullGamma(JMh[2], PARAMS.gamma_table);
    float gamma_bottom = PARAMS.lowerHullGamma;

    float3 boundaryReturn = findGamutBoundaryIntersection(JMh,
                                                            JMcusp,
                                                            focusJ,
                                                            limitJmax,
                                                            slope_gain,
                                                            gamma_top,
                                                            gamma_bottom);

    float2 JMboundary = float2(boundaryReturn[0], boundaryReturn[1]);
    float2 project_to = float2(boundaryReturn[2], 0.0);
    float projectJ = boundaryReturn[2];

    float3 reachBoundary = getReachBoundary(JMboundary[0],
                                              JMboundary[1],
                                              JMh[2],
                                              PARAMS,
                                              JMcusp,
                                              focusJ);

    float difference = max(1.0001, reachBoundary[1] / JMboundary[1]);
    float threshold = max(compressionThreshold, 1. / difference);

    float v = project_from[1] / JMboundary[1];

    v = compressionFunction(v,
                            threshold,
                            difference,
                            _invert);

    float2 JMcompressed = project_to + v * (JMboundary - project_to);
    float3 return_JMh = float3(JMcompressed[0], JMcompressed[1], JMh[2]);

    return return_JMh;
}

float3 gamutMap_fwd(float3 JMh,
                      ODTParams PARAMS
                    )
{
    return compressGamut(JMh, PARAMS, JMh[0], false);
}

float3 outputTransform_fwd( float3 aces,
                            float peakLuminance,
                            ODTParams PARAMS)
{
    float3 JMh = aces_to_JMh(aces, peakLuminance);

    float3 tonemappedJMh = tonemapAndCompress_fwd(JMh, PARAMS);

    float3 compressedJMh = gamutMap_fwd(tonemappedJMh, PARAMS);

    float3 XYZ = JMh_to_output_XYZ(compressedJMh,
                                      PARAMS);

    return XYZ;
}
}
