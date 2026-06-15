#include "NRD.hlsli"

[[vk::binding(0, 0)]]
RWTexture2D<float4> gOutput;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gNoisyDiffuse;

[[vk::binding(2, 0)]]
RWTexture2D<float4> gNoisySpecular;

[[vk::binding(3, 0)]]
RWTexture2D<float4> gDenoisedDiffuse;

[[vk::binding(4, 0)]]
RWTexture2D<float4> gDenoisedSpecular;

[[vk::binding(5, 0)]]
RWTexture2D<float4> gNormalRoughness;

[[vk::binding(6, 0)]]
RWTexture2D<float4> gMotionVectors;

[[vk::binding(7, 0)]]
RWTexture2D<float> gViewZ;

[[vk::binding(8, 0)]]
RWTexture2D<float4> gDirectLighting;

[[vk::binding(9, 0)]]
RWTexture2D<float4> gBaseColorMetalness;

[[vk::binding(10, 0)]]
RWTexture2D<float4> gValidation;

[[vk::binding(11, 0)]]
RWTexture2D<float4> gNoisyColor;

struct ChessNrdCompositePush
{
    uint width;
    uint height;
    uint denoiserMode;
    uint visualization;
};

static const uint kNrdReblur = 0u;
static const uint kNrdRelax = 1u;
static const uint kNrdReference = 2u;

static const uint kVisualFinal = 0u;
static const uint kVisualNoisy = 1u;
static const uint kVisualDirect = 2u;
static const uint kVisualNoisyDiffuse = 3u;
static const uint kVisualNoisySpecular = 4u;
static const uint kVisualDenoisedDiffuse = 5u;
static const uint kVisualDenoisedSpecular = 6u;
static const uint kVisualNormalRoughness = 7u;
static const uint kVisualViewZ = 8u;
static const uint kVisualMotion = 9u;
static const uint kVisualBaseColorMetalness = 10u;
static const uint kVisualValidation = 11u;

float3 tonemap(float3 color)
{
    color = color / (color + float3(1.0, 1.0, 1.0));
    return pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
}

float3 unpackRadiance(float4 value, uint denoiserMode)
{
    if (denoiserMode == kNrdReblur) {
        return REBLUR_BackEnd_UnpackRadianceAndNormHitDist(value).rgb;
    }
    if (denoiserMode == kNrdRelax) {
        return RELAX_BackEnd_UnpackRadiance(value).rgb;
    }
    return value.rgb;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void chessNrdCompositeMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                           uniform ChessNrdCompositePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float3 noisyDiffuse = unpackRadiance(gNoisyDiffuse[pixel], push.denoiserMode);
    float3 noisySpecular = unpackRadiance(gNoisySpecular[pixel], push.denoiserMode);
    float3 denoisedDiffuse = unpackRadiance(gDenoisedDiffuse[pixel], push.denoiserMode);
    float3 denoisedSpecular = unpackRadiance(gDenoisedSpecular[pixel], push.denoiserMode);
    float4 direct = gDirectLighting[pixel];
    float4 baseColorMetalness = gBaseColorMetalness[pixel];
    float3 diffDemodulate = baseColorMetalness.rgb * 0.99 + 0.01;
    float3 specDemodulate = lerp(float3(0.04, 0.04, 0.04), baseColorMetalness.rgb, baseColorMetalness.a) * 0.99 + 0.01;

    float3 color = direct.rgb;
    if (direct.a > 0.0) {
        color += denoisedDiffuse * diffDemodulate;
        color += denoisedSpecular * specDemodulate;
    }

    if (push.visualization == kVisualNoisy) {
        gOutput[pixel] = gNoisyColor[pixel];
        return;
    }
    if (push.visualization == kVisualDirect) {
        gOutput[pixel] = float4(tonemap(direct.rgb), 1.0);
        return;
    }
    if (push.visualization == kVisualNoisyDiffuse) {
        gOutput[pixel] = float4(tonemap(noisyDiffuse * diffDemodulate), 1.0);
        return;
    }
    if (push.visualization == kVisualNoisySpecular) {
        gOutput[pixel] = float4(tonemap(noisySpecular * specDemodulate), 1.0);
        return;
    }
    if (push.visualization == kVisualDenoisedDiffuse) {
        gOutput[pixel] = float4(tonemap(denoisedDiffuse * diffDemodulate), 1.0);
        return;
    }
    if (push.visualization == kVisualDenoisedSpecular) {
        gOutput[pixel] = float4(tonemap(denoisedSpecular * specDemodulate), 1.0);
        return;
    }
    if (push.visualization == kVisualNormalRoughness) {
        float4 normalRoughness = NRD_FrontEnd_UnpackNormalAndRoughness(gNormalRoughness[pixel]);
        gOutput[pixel] = float4(normalRoughness.xyz * 0.5 + 0.5, 1.0);
        return;
    }
    if (push.visualization == kVisualViewZ) {
        float z = gViewZ[pixel];
        gOutput[pixel] = float4(saturate(z / 1.2).xxx, 1.0);
        return;
    }
    if (push.visualization == kVisualMotion) {
        float2 mv = gMotionVectors[pixel].xy;
        gOutput[pixel] = float4(abs(mv) * 40.0, 0.0, 1.0);
        return;
    }
    if (push.visualization == kVisualBaseColorMetalness) {
        gOutput[pixel] = float4(baseColorMetalness.rgb, 1.0);
        return;
    }
    if (push.visualization == kVisualValidation) {
        gOutput[pixel] = gValidation[pixel];
        return;
    }

    gOutput[pixel] = float4(tonemap(color), 1.0);
}
