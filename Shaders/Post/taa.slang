// Temporal Anti-Aliasing resolve shader
// Based on "Improving Temporal Antialiasing using Adaptive Ray Tracing" (Ray Tracing Gems Ch.22)
// YCoCg variance clipping, Catmull-Rom history sampling, adaptive blend factor

struct TAAUniforms {
    float2 jitterOffset;
    float2 invResolution;
    uint   screenWidth;
    uint   screenHeight;
    float  blendMin;            // 0.05
    float  blendMax;            // 1.0
    float  varianceClipGamma;   // 1.0
    uint   frameIndex;
    float  motionWeightScale;   // motion rejection sensitivity
    float  pad;
};

ConstantBuffer<TAAUniforms> params;

Texture2D<float4>   currentColor;    // texture(0)
Texture2D<float>    depthBuffer;     // texture(1)
Texture2D<float2>   motionVectors;   // texture(2)
Texture2D<float4>   historyColor;    // texture(3)
RWTexture2D<float4> taaOutput;       // texture(4)
RWTexture2D<float4> historyWrite;    // texture(5)

SamplerState linearSampler;          // sampler(0)

// --- Color space conversions ---
float3 RGBToYCoCg(float3 rgb) {
    return float3(
         0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b,
         0.5  * rgb.r                - 0.5  * rgb.b,
        -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b
    );
}

float3 YCoCgToRGB(float3 ycocg) {
    float y  = ycocg.x;
    float co = ycocg.y;
    float cg = ycocg.z;
    return float3(y + co - cg, y + cg, y - co - cg);
}

// Karis 2014 tonemapping for firefly suppression
float3 tonemap(float3 c) {
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}

float3 untonemap(float3 c) {
    return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-5);
}

// 5-tap Catmull-Rom filter for sharper history sampling
float4 sampleHistoryCatmullRom(float2 uv) {
    float2 texSize = float2(params.screenWidth, params.screenHeight);
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - texPos1;

    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / max(w12, 1e-6);

    float2 texPos0 = (texPos1 - 1.0) * params.invResolution;
    float2 texPos3 = (texPos1 + 2.0) * params.invResolution;
    float2 texPos12 = (texPos1 + offset12) * params.invResolution;

    float4 result = float4(0.0);
    result += historyColor.SampleLevel(linearSampler, float2(texPos12.x, texPos0.y), 0) * w12.x * w0.y;
    result += historyColor.SampleLevel(linearSampler, float2(texPos0.x, texPos12.y), 0) * w0.x * w12.y;
    result += historyColor.SampleLevel(linearSampler, float2(texPos12.x, texPos12.y), 0) * w12.x * w12.y;
    result += historyColor.SampleLevel(linearSampler, float2(texPos3.x, texPos12.y), 0) * w3.x * w12.y;
    result += historyColor.SampleLevel(linearSampler, float2(texPos12.x, texPos3.y), 0) * w12.x * w3.y;

    return max(result, float4(0.0));
}

[numthreads(8, 8, 1)]
void taaMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= params.screenWidth || pixel.y >= params.screenHeight)
        return;

    float2 pixelCenter = float2(pixel) + 0.5;
    float2 uv = pixelCenter * params.invResolution;

    // Sample current frame (already rendered with jitter)
    float3 current = tonemap(currentColor[pixel].rgb);

    // Motion vector lookup and history reprojection
    float2 motion = motionVectors[pixel];
    float2 historyUV = uv - motion;

    // Reject history if reprojected UV is out of bounds
    bool historyValid = all(historyUV >= float2(0.0)) && all(historyUV <= float2(1.0));

    // Sample history with Catmull-Rom for sharpness
    float3 history = historyValid
        ? tonemap(sampleHistoryCatmullRom(historyUV).rgb)
        : current;

    // 3x3 neighborhood in YCoCg for variance clipping (Salvi 2016)
    float3 m1 = float3(0.0);
    float3 m2 = float3(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 samplePos = int2(pixel) + int2(dx, dy);
            samplePos = clamp(samplePos, int2(0), int2(params.screenWidth - 1, params.screenHeight - 1));
            float3 s = RGBToYCoCg(tonemap(currentColor[samplePos].rgb));
            m1 += s;
            m2 += s * s;
        }
    }
    m1 /= 9.0;
    m2 /= 9.0;
    float3 stddev = sqrt(max(m2 - m1 * m1, float3(0.0)));

    // Clip history to AABB in YCoCg space
    float3 historyYCoCg = RGBToYCoCg(history);
    float3 aabbMin = m1 - params.varianceClipGamma * stddev;
    float3 aabbMax = m1 + params.varianceClipGamma * stddev;
    float3 clippedYCoCg = clamp(historyYCoCg, aabbMin, aabbMax);
    float3 clippedHistory = YCoCgToRGB(clippedYCoCg);

    // Clip distance for adaptive blend
    float clipDist = length(historyYCoCg - clippedYCoCg);

    // Adaptive blend factor: more current on disocclusion/fast motion
    float motionMag = length(motion) * max(params.screenWidth, params.screenHeight);
    float motionWeight = saturate(motionMag * params.motionWeightScale);
    float blendFactor = lerp(params.blendMin, params.blendMax, saturate(motionWeight + clipDist));

    // Force full current on first frame or invalid history
    if (!historyValid || params.frameIndex == 0)
        blendFactor = 1.0;

    // Blend and undo tonemapping
    float3 resolved = lerp(clippedHistory, current, blendFactor);
    resolved = untonemap(resolved);

    taaOutput[pixel] = float4(max(resolved, float3(0.0)), 1.0);
    historyWrite[pixel] = taaOutput[pixel];
}
