// Tonemapping operators adapted from nvpro_core2 (Apache-2.0).
struct TonemapUniforms {
    uint isActive;
    uint method;
    float exposure;
    float contrast;
    float brightness;
    float saturation;
    float vignette;
    uint dither;
    float2 invResolution;
    float2 pad;
};

enum ToneMapMethod {
    eFilmic = 0,
    eUncharted2,
    eClip,
    eACES,
    eAgX,
    eKhronosPBR,
};

ConstantBuffer<TonemapUniforms> tm;
Texture2D<float4> inputTexture;
SamplerState linearSampler;

struct VSOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOut vertexMain(uint vertexID : SV_VertexID) {
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 uvs[3] = { float2(0.0, 0.0), float2(2.0, 0.0), float2(0.0, 2.0) };

    VSOut output;
    output.position = float4(positions[vertexID], 0.0, 1.0);
    output.uv = uvs[vertexID];
    return output;
}

inline float3 toSrgb(float3 rgb)
{
    float3 low = rgb * 12.92f;
    float3 high = fma(pow(rgb, float3(1.0f / 2.4f)), float3(1.055f), float3(-0.055f));
    return lerp(low, high, float3(rgb > float3(0.0031308f)));
}

inline float luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float3 tonemapFilmic(float3 color)
{
    float3 temp = max(float3(0.0f), color - float3(0.004f));
    float3 result = (temp * (float3(6.2f) * temp + float3(0.5f))) /
                    (temp * (float3(6.2f) * temp + float3(1.7f)) + float3(0.06f));
    return result;
}

inline float3 tonemapUncharted2Impl(float3 color)
{
    const float a = 0.15f;
    const float b = 0.50f;
    const float c = 0.10f;
    const float d = 0.20f;
    const float e = 0.02f;
    const float f = 0.30f;
    return ((color * (a * color + c * b) + d * e) / (color * (a * color + b) + d * f)) - e / f;
}

inline float3 tonemapUncharted2(float3 color)
{
    const float W = 11.2f;
    const float exposure_bias = 2.0f;
    color = tonemapUncharted2Impl(color * exposure_bias);
    float3 white_scale = float3(1.0f) / tonemapUncharted2Impl(float3(W));
    return pow(color * white_scale, float3(1.0f / 2.2f));
}

inline float3 tonemapACES(float3 color)
{
    const float3x3 ACESInputMat = float3x3(
        0.59719f, 0.07600f, 0.02840f,
        0.35458f, 0.90834f, 0.13383f,
        0.04823f, 0.01566f, 0.83777f);
    color = mul(color, ACESInputMat);

    float3 a = color * (color + float3(0.0245786f)) - float3(0.000090537f);
    float3 b = color * (float3(0.983729f) * color + float3(0.4329510f)) + float3(0.238081f);
    color = a / b;

    const float3x3 ACESOutputMat = float3x3(
        1.60475f, -0.10208f, -0.00327f,
        -0.53108f, 1.10813f, -0.07276f,
        -0.07367f, -0.00605f, 1.07602f);
    color = mul(color, ACESOutputMat);
    return toSrgb(color);
}

inline float3 tonemapAgX(float3 color)
{
    const float3x3 agx_mat = float3x3(
        0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f,
        0.0784335999999992f, 0.878468636469772f, 0.0784336f,
        0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f);
    color = mul(color, agx_mat);

    const float min_ev = -12.47393f;
    const float max_ev = 4.026069f;
    color = clamp(log2(color), min_ev, max_ev);
    color = (color - min_ev) / (max_ev - min_ev);

    float3 v = fma(float3(15.5f), color, float3(-40.14f));
    v = fma(color, v, float3(31.96f));
    v = fma(color, v, float3(-6.868f));
    v = fma(color, v, float3(0.4298f));
    v = fma(color, v, float3(0.1191f));
    v = fma(color, v, float3(-0.0023f));

    const float3x3 agx_mat_inv = float3x3(
        1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f,
        -0.0980208811401368f, 1.15190312990417f, -0.0980434501171241f,
        -0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f);
    v = mul(v, agx_mat_inv);

    return v;
}

inline float3 tonemapKhronosPBR(float3 color)
{
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = min(color.x, min(color.y, color.z));
    float peak = max(color.x, max(color.y, color.z));

    float offset = x < 0.08f ? x * (-6.25f * x + 1.0f) : 0.04f;
    color -= offset;

    if (peak >= startCompression) {
        const float d = 1.0f - startCompression;
        float newPeak = 1.0f - d * d / (peak + d - startCompression);
        color *= newPeak / peak;

        float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
        color = lerp(color, float3(newPeak), float3(g));
    }
    return toSrgb(color);
}

inline float3 adjustSaturation(float3 color, float sat)
{
    float lum = luminance(color);
    return lerp(float3(lum), color, float3(sat));
}

inline float3 applyTonemap(TonemapUniforms tmData, float3 color, float2 pixel)
{
    if (tmData.isActive == 0) {
        return toSrgb(color);
    }

    color *= tmData.exposure;

    float3 c;
    switch (tmData.method) {
        case (uint)ToneMapMethod::eFilmic:
            c = tonemapFilmic(color);
            break;
        case (uint)ToneMapMethod::eUncharted2:
            c = tonemapUncharted2(color);
            break;
        case (uint)ToneMapMethod::eClip:
            c = toSrgb(color);
            break;
        case (uint)ToneMapMethod::eACES:
            c = tonemapACES(color);
            break;
        case (uint)ToneMapMethod::eAgX:
            c = tonemapAgX(color);
            break;
        case (uint)ToneMapMethod::eKhronosPBR:
            c = tonemapKhronosPBR(color);
            break;
        default:
            c = toSrgb(color);
            break;
    }

    c = clamp(lerp(float3(0.5f), c, float3(tmData.contrast)), float3(0.0f), float3(1.0f));
    float safeBrightness = max(tmData.brightness, 0.001f);
    c = pow(c, float3(1.0f / safeBrightness));
    c = adjustSaturation(c, tmData.saturation);

    float2 center_uv = (pixel * tmData.invResolution) * 2.0f - 1.0f;
    c *= 1.0f - dot(center_uv, center_uv) * tmData.vignette;
    c = clamp(c, float3(0.0f), float3(1.0f));

    if (tmData.dither != 0) {
        const float levelsMinus1 = 255.0f;
        float noise = fract(dot(pixel, float2(0.245122331f, 0.430159704f)));
        noise = 0.5f - 2.0f * abs(noise - 0.5f);
        const float trinoise = sign(noise) * (1.0f - sqrt(1.0f - 2.0f * abs(noise)));
        const bool3 useUniform = (float3(0.5f / levelsMinus1) > c) || (c > float3(1.0f - 0.5f / levelsMinus1));
        c += lerp(float3(trinoise), float3(noise), float3(useUniform)) / levelsMinus1;
    }

    return c;
}

[shader("fragment")]
float4 fragmentMain(VSOut input) : SV_Target {
    float3 color = inputTexture.Sample(linearSampler, input.uv).rgb;
    float2 pixel = input.position.xy;
    float3 mapped = applyTonemap(tm, color, pixel);
    return float4(mapped, 1.0);
}
