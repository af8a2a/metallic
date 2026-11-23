module LumenReflection;

import ShaderCore;
import ParameterRoot;
import ScenePathMath;
import GlobalDistanceField;
import LumenGlobalRay;
import LumenProbeMath;
import LumenScreenTrace;

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

struct LumenReflectionParams
{
    LumenGlobalTraceBindings scene;
    ResourceHandle<Texture2D<float4>> position, normal, specular, sceneColor;
    ResourceHandle<Texture2D<float2>> blueNoise;
    ResourceHandle<RWTexture2D<float4>> outputReflections, trace, color;
    ResourceHandle<Texture2D<float4>> reflections;
    ViewCameraConstants camera;
    uint width, height, clipmapCount, frameIndex;
    float maxTraceDistance, maxRoughness;
};

static const float kGGXSamplingBias = 0.1f;
static const float kMinReflectionConeAngle = 0.00001f;
static const float kSurfaceBias = 0.05f; // UE SurfaceBias defaults to 5 cm.
static const float kMaxRayIntensity = 40.0f;
static const float kInvRoughnessFadeLength = 1.0f / 0.1f;

// UE 5.7.4 MonteCarlo.ush, isotropic material tangent basis.
float3x3 getReflectionTangentBasis(float3 tangentZ)
{
    const float signZ = tangentZ.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -rcp(signZ + tangentZ.z);
    const float b = tangentZ.x * tangentZ.y * a;
    float3 tangentX = float3(1.0f + signZ * a * (tangentZ.x * tangentZ.x), signZ * b, -signZ * tangentZ.x);
    float3 tangentY = float3(b, signZ + a * (tangentZ.y * tangentZ.y), -tangentZ.y);
    return float3x3(tangentX, tangentY, tangentZ);
}

// UE 5.7.4 MonteCarlo.ush, VisibleGGXPDF_aniso overload with
// GGX_BOUNDED_VNDF_SAMPLING and bLimitVDNFToReflection enabled.
float visibleReflectionGGXPDF(float3 v, float3 h, float2 alpha)
{
    float noV = v.z;
    float noH = h.z;
    float voH = dot(v, h);
    float a2 = alpha.x * alpha.y;
    float3 hs = float3(alpha.y * h.x, alpha.x * h.y, a2 * noH);
    float s = dot(hs, hs);
    float d = (1.0f / kLumenProbePi) * a2 * ((a2 / s) * (a2 / s));
    float lenV = length(float3(v.x * alpha.x, v.y * alpha.y, noV));
    float a = saturate(min(alpha.x, alpha.y));
    float bound = 1.0f + length(v.xy);
    float ka2 = a * a, s2 = bound * bound;
    float k = (s2 - ka2 * s2) / (s2 + ka2 * v.z * v.z);
    return (2.0f * d * voH) / (k * noV + lenV);
}

// UE 5.7.4 MonteCarlo.ush, ImportanceSampleVisibleGGX with the bounded reflection sampling range.
float4 importanceSampleVisibleReflectionGGX(float2 e, float2 alpha, float3 v)
{
    float3 vh = normalize(float3(alpha * v.xy, v.z));
    float phi = (2.0f * kLumenProbePi) * e.x;
    float a = saturate(min(alpha.x, alpha.y));
    float s = 1.0f + length(v.xy);
    float a2 = a * a, s2 = s * s;
    float k = (s2 - a2 * s2) / (s2 + a2 * v.z * v.z);
    float z = lerp(1.0f, -k * vh.z, e.y);
    float sinTheta = sqrt(saturate(1.0f - z * z));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 h = float3(x, y, z) + vh;
    h = normalize(float3(alpha * h.xy, max(0.0f, h.z)));
    return float4(h, visibleReflectionGGXPDF(v, h, alpha));
}

// UE 5.7.4 LumenReflections.usf, full-resolution, opaque non-Substrate,
// isotropic single-lobe ray generation. Distance quantization is in centimeters.
float4 generateReflectionRay(LumenReflectionParams params, uint2 pixel, float3 worldPosition, float3 normal,
    float roughness)
{
    float3 cameraVector = params.camera.upProjection.w > 0.5f
        ? ScenePath.cameraForwardFrom(params.camera.eye, params.camera.center)
        : normalize(worldPosition - params.camera.eye.xyz);

    float3 rayDirection;
    float coneAngle = 0.0f;
    if (roughness < 0.001f) {
        rayDirection = reflect(cameraVector, normal);
    } else {
        float2 e = lumenProbeBlueNoiseVec2(resolveUniform(params.blueNoise), pixel, params.frameIndex % 1024u);
        e.y *= 1.0f - kGGXSamplingBias;
        float3x3 tangentBasis = getReflectionTangentBasis(normal);
        float3 tangentV = mul(tangentBasis, -cameraVector);
        float2 alpha = (roughness * roughness).xx;
        float4 ggxSample = importanceSampleVisibleReflectionGGX(e, alpha, tangentV);
        float3 worldH = mul(ggxSample.xyz, tangentBasis);
        rayDirection = reflect(cameraVector, worldH);
        coneAngle = 1.0f / max(ggxSample.w, 0.0001f);
    }
    coneAngle = max(coneAngle, kMinReflectionConeAngle);

    // Reading the reference PF_FloatRGBA ray buffer rounds the direction and cone.
    return f16tof32(f32tof16(float4(rayDirection, coneAngle)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionTraceMain(uint3 thread : SV_DispatchThreadID)
{
    LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel = thread.xy;
    if (any(pixel >= uint2(params.width, params.height))) { return; }

    const float4 position = resolveUniform(params.position).Load(int3(pixel, 0));
    const float4 specular = resolveUniform(params.specular).Load(int3(pixel, 0));
    // UE 5.7.4 LumenReflectionsCombine.ush, opaque single-lobe fade.
    const float fadeAlpha = saturate((params.maxRoughness - specular.w) * kInvRoughnessFadeLength);
    float4 reflections = 0.0f;
    float4 debugTrace = float4(-1.0f, float(kGlobalDistanceFieldEmpty), -1.0f, 0.0f);
    if (position.w > 0.0f && any(specular.xyz > 0.0f) && fadeAlpha > 0.0f) {
        const float3 normal = normalize(resolveUniform(params.normal).Load(int3(pixel, 0)).xyz);
        const float4 rayData = generateReflectionRay(params, pixel, position.xyz, normal, specular.w);

        // UE 5.7.4 LumenReflectionTracing.usf, TraceVoxels without screen or mesh-SDF methods.
        // Their disabled permutations leave no prefix distance to pull back.
        const float3 samplePosition = position.xyz + kSurfaceBias * rayData.xyz;
        const float stepFactorNoise = lerp(0.95f, 1.0f / 0.95f, screenTraceNoise(float2(pixel),
            float(params.frameIndex % 8u)));
        const float stepFactor = lerp(stepFactorNoise, 1.0f, saturate(rayData.w / (kLumenProbePi / 256.0f)));
        const float maxTraceDistance = f16tof32(f32tof16(min(params.maxTraceDistance * 100.0f, 65504.0f))
            & 0x7fffu) * 0.01f;
        LumenGlobalTraceRay ray = {float4(samplePosition, 0.0f), float4(rayData.xyz, maxTraceDistance)};
        const LumenGlobalTraceResult hit = traceLumenReflectionGlobalRay(
            resolveLumenGlobalTraceScene(params.scene, params.clipmapCount), ray, stepFactor);

        float3 lighting = hit.radiance.xyz;
        float maxLighting = max(lighting.x, max(lighting.y, lighting.z));
        if (maxLighting > kMaxRayIntensity) {
            lighting *= kMaxRayIntensity / maxLighting;
        }
        reflections = float4(lighting, fadeAlpha);
        debugTrace = float4(hit.metrics.x, float(hit.state.x),
            hit.state.y == ~0u ? -1.0f : float(hit.state.y), float(hit.state.w));
    }
    resolveUniform(params.outputReflections)[pixel] = reflections;
    resolveUniform(params.trace)[pixel] = debugTrace;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionCompositeMain(uint3 thread : SV_DispatchThreadID)
{
    LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel = thread.xy;
    if (any(pixel >= uint2(params.width, params.height))) { return; }

    const float4 sceneColor = resolveUniform(params.sceneColor).Load(int3(pixel, 0));
    const float4 reflection = resolveUniform(params.reflections).Load(int3(pixel, 0));
    float3 lighting = 0.0f;
    // UE 5.7.4 ReflectionEnvironmentPixelShader.usf, standalone Lumen
    // reflection composite. The supported permutation has no sky or probe fallback.
    if (reflection.a > 0.0f) {
        const float3 specular = resolveUniform(params.specular).Load(int3(pixel, 0)).xyz;
        lighting = reflection.rgb * reflection.a;
        lighting *= specular;
    }
    resolveUniform(params.color)[pixel] = float4(sceneColor.rgb - min(-lighting, 0.0f), sceneColor.a);
}
