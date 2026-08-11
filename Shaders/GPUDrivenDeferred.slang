#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_ENERGY_TABLES_USE_UINT16 0
#define OPENPBR_FEATURE_EnableSheenAndCoat false
#define OPENPBR_FEATURE_EnableDispersion false
#define OPENPBR_FEATURE_EnableTranslucency true
#define OPENPBR_FEATURE_EnableMetallic true
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) OPENPBR_FEATURE_##name

#ifndef GPU_DRIVEN_OPENPBR_LUT_2D_SHADER_BASE
#define GPU_DRIVEN_OPENPBR_LUT_2D_SHADER_BASE 0
#endif
#ifndef GPU_DRIVEN_OPENPBR_LUT_3D_SHADER_BASE
#define GPU_DRIVEN_OPENPBR_LUT_3D_SHADER_BASE 0
#endif

uint openPBRLut2DSlot(int lutId)
{
    if (lutId == 1) {
        return 0u;
    }
    if (lutId == 2) {
        return 1u;
    }
    if (lutId == 4) {
        return 2u;
    }
    if (lutId == 5) {
        return 3u;
    }
    if (lutId == 6) {
        return 4u;
    }
    return 5u;
}

uint openPBRLut3DSlot(int lutId)
{
    return lutId == 3 ? 1u : 0u;
}

float4 sampleOpenPBRLut2D(int lutId, float2 uv)
{
    DescriptorHandle<Texture2D<float4>> textureHandle =
        DescriptorHandle<Texture2D<float4>>(uint2(
            GPU_DRIVEN_OPENPBR_LUT_2D_SHADER_BASE + openPBRLut2DSlot(lutId),
            0));
    Texture2D<float4> texture = textureHandle;
    uint width = 1u;
    uint height = 1u;
    texture.GetDimensions(width, height);
    float2 size = float2(max(width, 1u), max(height, 1u));
    float2 texel = clamp(uv * size - 0.5, float2(0.0), size - 1.0);
    int2 baseTexel = int2(floor(texel));
    int2 nextTexel = min(baseTexel + int2(1, 1), int2(width - 1u, height - 1u));
    float2 fraction = texel - float2(baseTexel);
    float4 v00 = texture.Load(int3(baseTexel.x, baseTexel.y, 0));
    float4 v10 = texture.Load(int3(nextTexel.x, baseTexel.y, 0));
    float4 v01 = texture.Load(int3(baseTexel.x, nextTexel.y, 0));
    float4 v11 = texture.Load(int3(nextTexel.x, nextTexel.y, 0));
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

float4 sampleOpenPBRLut3D(int lutId, float3 uvw)
{
    DescriptorHandle<Texture3D<float4>> textureHandle =
        DescriptorHandle<Texture3D<float4>>(uint2(
            GPU_DRIVEN_OPENPBR_LUT_3D_SHADER_BASE + openPBRLut3DSlot(lutId),
            0));
    Texture3D<float4> texture = textureHandle;
    uint width = 1u;
    uint height = 1u;
    uint depth = 1u;
    texture.GetDimensions(width, height, depth);
    float3 size = float3(max(width, 1u), max(height, 1u), max(depth, 1u));
    float3 texel = clamp(uvw * size - 0.5, float3(0.0), size - 1.0);
    int3 baseTexel = int3(floor(texel));
    int3 nextTexel = min(baseTexel + int3(1, 1, 1), int3(width - 1u, height - 1u, depth - 1u));
    float3 fraction = texel - float3(baseTexel);
    float4 v000 = texture.Load(int4(baseTexel.x, baseTexel.y, baseTexel.z, 0));
    float4 v100 = texture.Load(int4(nextTexel.x, baseTexel.y, baseTexel.z, 0));
    float4 v010 = texture.Load(int4(baseTexel.x, nextTexel.y, baseTexel.z, 0));
    float4 v110 = texture.Load(int4(nextTexel.x, nextTexel.y, baseTexel.z, 0));
    float4 v001 = texture.Load(int4(baseTexel.x, baseTexel.y, nextTexel.z, 0));
    float4 v101 = texture.Load(int4(nextTexel.x, baseTexel.y, nextTexel.z, 0));
    float4 v011 = texture.Load(int4(baseTexel.x, nextTexel.y, nextTexel.z, 0));
    float4 v111 = texture.Load(int4(nextTexel.x, nextTexel.y, nextTexel.z, 0));
    float4 z0 = lerp(lerp(v000, v100, fraction.x), lerp(v010, v110, fraction.x), fraction.y);
    float4 z1 = lerp(lerp(v001, v101, fraction.x), lerp(v011, v111, fraction.x), fraction.y);
    return lerp(z0, z1, fraction.z);
}

#define OPENPBR_SAMPLE_2D_TEXTURE(lut_id, uv) sampleOpenPBRLut2D(lut_id, uv)
#define OPENPBR_SAMPLE_3D_TEXTURE(lut_id, uvw) sampleOpenPBRLut3D(lut_id, uvw)

#include "../External/openpbr-bsdf/openpbr.h"

static const uint kVisibilityTriangleBits = 7u;
static const uint kVisibilityTriangleMask = (1u << kVisibilityTriangleBits) - 1u;
static const uint kInvalidTextureIndex = 0xffffffffu;
static const uint kModeMeshlet = 0u;
static const uint kModePrimitive = 1u;
static const uint kModeLod = 2u;
static const uint kModeShaded = 3u;
static const uint kModeBaseColor = 4u;
static const uint kEnvironmentModeProcedural = 0u;
static const uint kEnvironmentModeMap = 1u;
static const uint kEnvironmentModeDisabled = 2u;
static const uint kEnvironmentSampleCount = 24u;
static const float kPi = 3.14159265358979323846;

struct GPUDrivenPreviewPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint positionBuffer;
    uint meshletBuffer;
    uint meshletDrawBuffer;
    uint meshletVertexBuffer;
    uint meshletTriangleBuffer;
    uint paramsBuffer;
    uint transformBuffer;
    uint instanceBuffer;
    uint instanceVisibilityBuffer;
    uint visibleInstanceIdsBuffer;
    uint visibleInstanceCounterBuffer;
    uint visibleMeshletBuffer0;
    uint visibleMeshletBuffer1;
    uint indirectBuffer0;
    uint indirectBuffer1;
    uint hzbBuffer0;
    uint hzbBuffer1;
    uint deferredColorBuffer;
    uint depthImage;
    uint visibilityImage;
    uint passIndex;
    uint mipLevel;
    uint projectWithCullingCamera;
    uint materialBuffer;
    uint materialTextureRemapBuffer;
    uint environmentImage;
    uint environmentSHBuffer;
};

struct GPUDrivenPreviewVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float2 texcoord;
    uint flags;
    uint reserved;
};

struct SceneTransform
{
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
};

struct GPUDrivenPreviewMeshlet
{
    uint4 ranges;
    uint4 lod;
    float4 boundingSphere;
    float4 coneApexCutoff;
    float4 coneAxisLodError;
};

struct GPUDrivenPreviewMeshletDraw
{
    uint meshletIndex;
    uint instanceIndex;
    uint geometryIndex;
    uint drawBucket;
};

struct GPUDrivenPreviewGeometry
{
    uint4 source;
    uint4 counts;
    uint4 payload;
    uint4 meshletPayload;
    float4 localBoundingSphere;
    uint4 identity;
};

struct GPUDrivenPreviewInstance
{
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
    float4 previousWorld0;
    float4 previousWorld1;
    float4 previousWorld2;
    float4 previousWorld3;
    float4 localBoundingSphere;
    uint4 identity;
};

struct GPUDrivenPreviewTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint padding0;
    uint padding1;
    float4 transform0;
    float4 transform1;
};

struct GPUDrivenPreviewMaterial
{
    float4 baseColor;
    float4 emissive;
    float4 params;
    float4 textureParams;
    float4 glassParams;
    float4 attenuationColor;
    float4 diffuseTransmission;
    GPUDrivenPreviewTextureInfo baseColorTexture;
    GPUDrivenPreviewTextureInfo metallicRoughnessTexture;
    GPUDrivenPreviewTextureInfo normalTexture;
    GPUDrivenPreviewTextureInfo occlusionTexture;
    GPUDrivenPreviewTextureInfo emissiveTexture;
    GPUDrivenPreviewTextureInfo transmissionTexture;
    GPUDrivenPreviewTextureInfo thicknessTexture;
    GPUDrivenPreviewTextureInfo diffuseTransmissionTexture;
    GPUDrivenPreviewTextureInfo diffuseTransmissionColorTexture;
    uint4 identity;
};

SceneTransform instanceTransform(GPUDrivenPreviewInstance instance)
{
    SceneTransform transform;
    transform.world0 = instance.world0;
    transform.world1 = instance.world1;
    transform.world2 = instance.world2;
    transform.world3 = instance.world3;
    return transform;
}

uint loadPackedMeshletTriangleIndex(
    StructuredBuffer<uint> triangleWords,
    uint firstWord,
    uint triangleIndex,
    uint triangleCorner)
{
    uint byteOffset = triangleIndex * 3u + triangleCorner;
    uint word = triangleWords[firstWord + (byteOffset >> 2u)];
    return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
}

struct GPUDrivenPreviewParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 clearColor;
    float4 previousEye;
    float4 previousCenter;
    float4 previousUpProjection;
    float4 previousViewport;
    float4 previousClipOrtho;
    float4 renderEye;
    float4 renderCenter;
    float4 renderUpProjection;
    float4 renderViewport;
    float4 renderClipOrtho;
    uint mode;
    uint meshletOffset;
    uint meshletCount;
    uint selectedLodLevel;
    uint instanceCount;
    uint width;
    uint height;
    uint hzbMipCount;
    uint frameIndex;
    uint hzbValid;
    uint cullingFlags;
    uint materialTextureCount;
    float environmentIntensity;
    float environmentRotationRadians;
    uint environmentMode;
    uint environmentVisible;
    uint materialCount;
    uint visibleMeshletCapacity;
    uint shadingPadding1;
    uint shadingPadding2;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float3 transformPoint(SceneTransform transform, float3 localPosition)
{
    return transform.world0.xyz * localPosition.x +
        transform.world1.xyz * localPosition.y +
        transform.world2.xyz * localPosition.z +
        transform.world3.xyz;
}

float3 transformDirection(SceneTransform transform, float3 localDirection)
{
    return transform.world0.xyz * localDirection.x +
        transform.world1.xyz * localDirection.y +
        transform.world2.xyz * localDirection.z;
}

float3 transformNormal(SceneTransform transform, float3 localNormal)
{
    float3 cofactor0 = cross(transform.world1.xyz, transform.world2.xyz);
    float3 cofactor1 = cross(transform.world2.xyz, transform.world0.xyz);
    float3 cofactor2 = cross(transform.world0.xyz, transform.world1.xyz);
    return safeNormalize(
        cofactor0 * localNormal.x + cofactor1 * localNormal.y + cofactor2 * localNormal.z,
        float3(0.0, 0.0, 1.0));
}

float3 worldToView(float3 worldPosition, GPUDrivenPreviewParams params)
{
    float3 forward = safeNormalize(params.renderCenter.xyz - params.renderEye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(params.renderUpProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 cameraVector = worldPosition - params.renderEye.xyz;
    return float3(dot(cameraVector, right), dot(cameraVector, up), dot(cameraVector, forward));
}

float projectDepth(float viewZ, GPUDrivenPreviewParams params)
{
    float zNear = max(params.renderClipOrtho.x, 0.0001);
    float zFar = max(params.renderClipOrtho.y, zNear + 0.0001);
    float depth;
    if (params.renderUpProjection.w > 0.5) {
        depth = (viewZ - zNear) / (zFar - zNear);
    } else {
        float safeViewZ = abs(viewZ) > 0.000001 ? viewZ : 0.000001;
        depth = zFar / (zFar - zNear) - (zFar * zNear) / ((zFar - zNear) * safeViewZ);
    }
    return params.renderClipOrtho.w > 0.5 ? 1.0 - depth : depth;
}

float4 projectPosition(float3 worldPosition, GPUDrivenPreviewParams params)
{
    float3 viewPosition = worldToView(worldPosition, params);
    float ndcZ = projectDepth(viewPosition.z, params);
    float aspect = max(params.renderViewport.x, 0.001);
    if (params.renderUpProjection.w > 0.5) {
        float halfHeight = max(params.renderClipOrtho.z * 0.5, 0.0001);
        return float4(
            viewPosition.x / (halfHeight * aspect),
            -viewPosition.y / halfHeight,
            ndcZ,
            1.0);
    }
    float fovRadians = clamp(params.renderViewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = viewPosition.z;
    return float4(
        viewPosition.x * yScale / aspect,
        -viewPosition.y * yScale,
        ndcZ * w,
        w);
}

float2 screenPosition(float4 clipPosition, GPUDrivenPreviewParams params)
{
    float reciprocalW = 1.0 / max(abs(clipPosition.w), 0.000001);
    float2 ndc = clipPosition.xy * reciprocalW;
    return (ndc * 0.5 + 0.5) * float2(params.width, params.height);
}

float cross2(float2 left, float2 right)
{
    return left.x * right.y - left.y * right.x;
}

float3 perspectiveBarycentrics(
    float2 pixel,
    float4 clip0,
    float4 clip1,
    float4 clip2,
    GPUDrivenPreviewParams params)
{
    float2 p0 = screenPosition(clip0, params);
    float2 p1 = screenPosition(clip1, params);
    float2 p2 = screenPosition(clip2, params);
    float denominator = cross2(p1 - p0, p2 - p0);
    if (abs(denominator) <= 0.000001) {
        return float3(1.0, 0.0, 0.0);
    }
    float3 screenWeights = float3(
        cross2(p1 - pixel, p2 - pixel),
        cross2(p2 - pixel, p0 - pixel),
        cross2(p0 - pixel, p1 - pixel)) / denominator;
    float3 weights = screenWeights / float3(clip0.w, clip1.w, clip2.w);
    float weightSum = weights.x + weights.y + weights.z;
    return abs(weightSum) > 0.000001 ? weights / weightSum : float3(1.0, 0.0, 0.0);
}

float3 cameraRayDirection(float2 pixel, GPUDrivenPreviewParams params)
{
    float2 uv = pixel / float2(max(params.width, 1u), max(params.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 forward = safeNormalize(params.renderCenter.xyz - params.renderEye.xyz, float3(0.0, 0.0, -1.0));
    if (params.renderUpProjection.w > 0.5) {
        return forward;
    }
    float3 cameraUp = safeNormalize(params.renderUpProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float halfHeight = tan(clamp(params.renderViewport.w, 0.017453292, 3.12413936) * 0.5);
    return safeNormalize(
        forward + right * (ndc.x * max(params.renderViewport.x, 0.001) * halfHeight) + up * (ndc.y * halfHeight),
        forward);
}

float3 srgbToLinear(float3 color)
{
    float3 low = color / 12.92;
    float3 high = pow((color + 0.055) / 1.055, float3(2.4));
    return float3(
        color.x <= 0.04045 ? low.x : high.x,
        color.y <= 0.04045 ? low.y : high.y,
        color.z <= 0.04045 ? low.z : high.z);
}

float3 linearToSrgb(float3 color)
{
    color = max(color, float3(0.0));
    float3 low = color * 12.92;
    float3 high = 1.055 * pow(color, float3(1.0 / 2.4)) - 0.055;
    return float3(
        color.x <= 0.0031308 ? low.x : high.x,
        color.y <= 0.0031308 ? low.y : high.y,
        color.z <= 0.0031308 ? low.z : high.z);
}

float2 transformMaterialUv(GPUDrivenPreviewTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

float4 sampleImageBilinear(Texture2D<float4> texture, float2 uv, bool wrapY)
{
    uint width = 1u;
    uint height = 1u;
    texture.GetDimensions(width, height);
    float2 addressedUv = float2(frac(uv.x), wrapY ? frac(uv.y) : saturate(uv.y));
    float2 texel = addressedUv * float2(width, height) - 0.5;
    int2 baseTexel = int2(floor(texel));
    float2 fraction = frac(texel);
    int x0 = (baseTexel.x % int(width) + int(width)) % int(width);
    int x1 = (x0 + 1) % int(width);
    int y0 = wrapY
        ? (baseTexel.y % int(height) + int(height)) % int(height)
        : clamp(baseTexel.y, 0, int(height) - 1);
    int y1 = wrapY ? (y0 + 1) % int(height) : min(y0 + 1, int(height) - 1);
    float4 v00 = texture.Load(int3(x0, y0, 0));
    float4 v10 = texture.Load(int3(x1, y0, 0));
    float4 v01 = texture.Load(int3(x0, y1, 0));
    float4 v11 = texture.Load(int3(x1, y1, 0));
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

float4 sampleMaterialTexture(
    GPUDrivenPreviewTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push)
{
    if (textureInfo.textureIndex == kInvalidTextureIndex ||
        textureInfo.textureIndex >= params.materialTextureCount) {
        return fallback;
    }
    DescriptorHandle<StructuredBuffer<uint>> remapHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + push.materialTextureRemapBuffer,
            0));
    StructuredBuffer<uint> remap = remapHandle;
    DescriptorHandle<Texture2D<float4>> textureHandle =
        DescriptorHandle<Texture2D<float4>>(uint2(
            push.imageShaderIndexBase + remap[textureInfo.textureIndex],
            0));
    Texture2D<float4> texture = textureHandle;
    return sampleImageBilinear(texture, transformMaterialUv(textureInfo, uv), true);
}

float2 environmentUv(float3 direction, float rotationRadians)
{
    float phi = atan2(direction.z, direction.x) + rotationRadians;
    return float2(frac(phi / (2.0 * kPi) + 0.5), acos(clamp(direction.y, -1.0, 1.0)) / kPi);
}

float3 proceduralSky(float3 direction, float rotationRadians)
{
    float sine = sin(rotationRadians);
    float cosine = cos(rotationRadians);
    float3 sunDirection = safeNormalize(
        float3(0.35 * cosine + 0.22 * sine, 0.88, -0.35 * sine + 0.22 * cosine),
        float3(0.0, 1.0, 0.0));
    float t = saturate(direction.y * 0.5 + 0.5);
    float3 sky = lerp(float3(0.62, 0.70, 0.78), float3(0.12, 0.24, 0.48), t) * 0.45;
    return sky + pow(saturate(dot(direction, sunDirection)), 256.0) * float3(7.0, 6.4, 5.3);
}

float3 environmentRadiance(float3 direction, GPUDrivenPreviewParams params, GPUDrivenPreviewPush push)
{
    if (params.environmentMode == kEnvironmentModeDisabled || params.environmentIntensity <= 0.0) {
        return float3(0.0);
    }
    if (params.environmentMode == kEnvironmentModeProcedural) {
        return proceduralSky(direction, params.environmentRotationRadians) * params.environmentIntensity;
    }
    DescriptorHandle<Texture2D<float4>> environmentHandle =
        DescriptorHandle<Texture2D<float4>>(uint2(push.imageShaderIndexBase + push.environmentImage, 0));
    Texture2D<float4> environmentMap = environmentHandle;
    return max(
        sampleImageBilinear(
            environmentMap,
            environmentUv(safeNormalize(direction, float3(0.0, 1.0, 0.0)), params.environmentRotationRadians),
            false).rgb,
        float3(0.0)) * params.environmentIntensity;
}

float3 environmentSHRadiance(
    float3 direction,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push)
{
    float sine = sin(params.environmentRotationRadians);
    float cosine = cos(params.environmentRotationRadians);
    float3 rotatedDirection = float3(
        direction.x * cosine - direction.z * sine,
        direction.y,
        direction.x * sine + direction.z * cosine);
    float x = rotatedDirection.x;
    float y = rotatedDirection.y;
    float z = rotatedDirection.z;
    float basis[9] =
    {
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    };
    DescriptorHandle<StructuredBuffer<float4>> coefficientsHandle =
        DescriptorHandle<StructuredBuffer<float4>>(uint2(
            push.bufferShaderIndexBase + push.environmentSHBuffer,
            0));
    StructuredBuffer<float4> coefficients = coefficientsHandle;
    float3 radiance = float3(0.0);
    for (uint coefficientIndex = 0u; coefficientIndex < 9u; ++coefficientIndex) {
        radiance += coefficients[coefficientIndex].xyz * basis[coefficientIndex];
    }
    return max(radiance, float3(0.0)) * params.environmentIntensity;
}

float3 hashColor(uint seed)
{
    float3 value = frac(sin(float3(seed, seed * 17u + 13u, seed * 31u + 7u) * 0.017453292) * 43758.5453);
    return lerp(float3(0.18, 0.34, 0.62), float3(0.95, 0.82, 0.35), value);
}

float3 environmentIntegrationDirection(uint sampleIndex)
{
    // A fixed Fibonacci sphere makes the environment quadrature spatially
    // coherent: neighboring surfaces evaluate the same incident directions
    // while OpenPBR supplies their smoothly varying BSDF response.
    float unitIndex = (float(sampleIndex) + 0.5) / float(kEnvironmentSampleCount);
    float y = 1.0 - 2.0 * unitIndex;
    float radius = sqrt(max(1.0 - y * y, 0.0));
    float phi = 2.0 * kPi * frac((float(sampleIndex) + 0.5) * 0.61803398875);
    return float3(cos(phi) * radius, y, sin(phi) * radius);
}

float3 acesToneMap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

uint packColor(float3 color)
{
    uint3 rgb = uint3(saturate(color) * 255.0 + 0.5);
    return rgb.x | (rgb.y << 8u) | (rgb.z << 16u) | 0xff000000u;
}

uint packHdrColor(float3 color)
{
    return packColor(linearToSrgb(acesToneMap(max(color, float3(0.0)))));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gpuDrivenPreviewDeferredMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    GPUDrivenPreviewParams params = paramsHandle[0];
    if (dispatchThreadId.x >= params.width || dispatchThreadId.y >= params.height) {
        return;
    }

    DescriptorHandle<Texture2D<uint>> visibilityHandle =
        DescriptorHandle<Texture2D<uint>>(uint2(push.imageShaderIndexBase + push.visibilityImage, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> colorHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.deferredColorBuffer, 0));
    Texture2D<uint> visibility = visibilityHandle;
    RWStructuredBuffer<uint> colors = colorHandle;
    uint packedVisibility = visibility.Load(int3(dispatchThreadId.xy, 0));
    uint outputIndex = dispatchThreadId.y * params.width + dispatchThreadId.x;
    float2 pixelCenter = float2(dispatchThreadId.xy) + 0.5;

    if (packedVisibility == 0u) {
        if (params.mode == kModeShaded && params.environmentVisible != 0u) {
            colors[outputIndex] = packHdrColor(environmentRadiance(cameraRayDirection(pixelCenter, params), params, push));
        } else {
            colors[outputIndex] = packColor(params.clearColor.xyz);
        }
        return;
    }

    uint encodedMeshlet = packedVisibility >> kVisibilityTriangleBits;
    uint triangleIndex = packedVisibility & kVisibilityTriangleMask;
    if (encodedMeshlet == 0u) {
        colors[outputIndex] = packColor(params.clearColor.xyz);
        return;
    }
    uint meshletIndex = encodedMeshlet - 1u;

    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> meshletHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.bufferShaderIndexBase + push.meshletBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshletDraw>> meshletDrawHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshletDraw>>(uint2(
            push.bufferShaderIndexBase + push.meshletDrawBuffer,
            0));
    StructuredBuffer<GPUDrivenPreviewMeshlet> meshlets = meshletHandle;
    StructuredBuffer<GPUDrivenPreviewMeshletDraw> meshletDraws = meshletDrawHandle;
    GPUDrivenPreviewMeshletDraw meshletDraw = meshletDraws[meshletIndex];
    GPUDrivenPreviewMeshlet meshlet = meshlets[meshletDraw.meshletIndex];
    if (triangleIndex >= meshlet.ranges.w) {
        colors[outputIndex] = packColor(params.clearColor.xyz);
        return;
    }

    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewGeometry>> geometryHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewGeometry>>(uint2(
            push.bufferShaderIndexBase + push.transformBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(
            push.bufferShaderIndexBase + push.instanceBuffer,
            0));
    StructuredBuffer<GPUDrivenPreviewGeometry> geometries = geometryHandle;
    StructuredBuffer<GPUDrivenPreviewInstance> instances = instanceHandle;
    GPUDrivenPreviewGeometry geometry = geometries[meshletDraw.geometryIndex];
    GPUDrivenPreviewInstance instance = instances[meshletDraw.instanceIndex];

    if (params.mode != kModeShaded && params.mode != kModeBaseColor) {
        uint colorSeed = meshletIndex;
        if (params.mode == kModePrimitive) {
            colorSeed = geometry.source.x;
        } else if (params.mode == kModeLod) {
            colorSeed = geometry.source.x * 1009u +
                meshlet.lod.y * 131u + meshlet.lod.x * 17u + 1u;
        }
        colors[outputIndex] = packColor(hashColor(colorSeed + 1u));
        return;
    }

    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>> vertexHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>(uint2(push.bufferShaderIndexBase + push.positionBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> meshletVertexHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletVertexBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> meshletTriangleHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletTriangleBuffer, 0));
    StructuredBuffer<GPUDrivenPreviewVertex> vertices = vertexHandle;
    StructuredBuffer<uint> meshletVertices = meshletVertexHandle;
    StructuredBuffer<uint> meshletTriangles = meshletTriangleHandle;

    uint3 localIndices = uint3(
        loadPackedMeshletTriangleIndex(meshletTriangles, meshlet.ranges.z, triangleIndex, 0u),
        loadPackedMeshletTriangleIndex(meshletTriangles, meshlet.ranges.z, triangleIndex, 1u),
        loadPackedMeshletTriangleIndex(meshletTriangles, meshlet.ranges.z, triangleIndex, 2u));
    uint3 vertexIndices = uint3(
        geometry.payload.x + meshletVertices[meshlet.ranges.x + localIndices.x],
        geometry.payload.x + meshletVertices[meshlet.ranges.x + localIndices.y],
        geometry.payload.x + meshletVertices[meshlet.ranges.x + localIndices.z]);
    GPUDrivenPreviewVertex vertex0 = vertices[vertexIndices.x];
    GPUDrivenPreviewVertex vertex1 = vertices[vertexIndices.y];
    GPUDrivenPreviewVertex vertex2 = vertices[vertexIndices.z];
    SceneTransform transform = instanceTransform(instance);

    float3 world0 = transformPoint(transform, vertex0.position.xyz);
    float3 world1 = transformPoint(transform, vertex1.position.xyz);
    float3 world2 = transformPoint(transform, vertex2.position.xyz);
    float4 clip0 = projectPosition(world0, params);
    float4 clip1 = projectPosition(world1, params);
    float4 clip2 = projectPosition(world2, params);
    float3 barycentrics = perspectiveBarycentrics(pixelCenter, clip0, clip1, clip2, params);
    float3 worldPosition = world0 * barycentrics.x + world1 * barycentrics.y + world2 * barycentrics.z;
    float2 texcoord = vertex0.texcoord.xy * barycentrics.x +
        vertex1.texcoord.xy * barycentrics.y + vertex2.texcoord.xy * barycentrics.z;
    float3 localNormal = vertex0.normal.xyz * barycentrics.x +
        vertex1.normal.xyz * barycentrics.y + vertex2.normal.xyz * barycentrics.z;
    float3 geometryNormal = safeNormalize(cross(world1 - world0, world2 - world0), float3(0.0, 0.0, 1.0));
    float3 normal = transformNormal(transform, localNormal);
    float3 viewDirection = safeNormalize(params.renderEye.xyz - worldPosition, -cameraRayDirection(pixelCenter, params));
    if (dot(normal, geometryNormal) < 0.0) {
        normal = -normal;
    }

    float4 localTangent = vertex0.tangent * barycentrics.x +
        vertex1.tangent * barycentrics.y + vertex2.tangent * barycentrics.z;
    float3 tangent = transformDirection(transform, localTangent.xyz);
    tangent = safeNormalize(tangent - normal * dot(normal, tangent),
        abs(normal.z) < 0.999 ? safeNormalize(cross(float3(0.0, 0.0, 1.0), normal), float3(1.0, 0.0, 0.0)) :
            safeNormalize(cross(float3(0.0, 1.0, 0.0), normal), float3(1.0, 0.0, 0.0)));
    float tangentSign = localTangent.w < 0.0 ? -1.0 : 1.0;
    float3 bitangent = safeNormalize(cross(normal, tangent) * tangentSign, float3(0.0, 1.0, 0.0));

    uint materialIndex = min(instance.identity.y, max(params.materialCount, 1u) - 1u);
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>> materialHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>(uint2(push.bufferShaderIndexBase + push.materialBuffer, 0));
    StructuredBuffer<GPUDrivenPreviewMaterial> materials = materialHandle;
    GPUDrivenPreviewMaterial material = materials[materialIndex];

    float4 normalSample = sampleMaterialTexture(
        material.normalTexture,
        texcoord,
        float4(0.5, 0.5, 1.0, 1.0),
        params,
        push);
    float3 tangentNormal = normalSample.xyz * 2.0 - 1.0;
    tangentNormal.xy *= material.textureParams.x;
    tangentNormal = safeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));
    normal = safeNormalize(
        tangent * tangentNormal.x + bitangent * tangentNormal.y + normal * tangentNormal.z,
        normal);
    if (dot(normal, geometryNormal) < 0.0) {
        normal = -normal;
    }
    if (dot(normal, viewDirection) < 0.0) {
        normal = -normal;
    }
    tangent = safeNormalize(tangent - normal * dot(normal, tangent), tangent);
    bitangent = safeNormalize(cross(normal, tangent) * tangentSign, bitangent);

    float4 baseColorSample = sampleMaterialTexture(
        material.baseColorTexture, texcoord, float4(1.0), params, push);
    float4 metallicRoughnessSample = sampleMaterialTexture(
        material.metallicRoughnessTexture, texcoord, float4(1.0), params, push);
    float4 occlusionSample = sampleMaterialTexture(
        material.occlusionTexture, texcoord, float4(1.0), params, push);
    float4 emissiveSample = sampleMaterialTexture(
        material.emissiveTexture, texcoord, float4(1.0), params, push);
    float4 transmissionSample = sampleMaterialTexture(
        material.transmissionTexture, texcoord, float4(1.0), params, push);

    float3 baseColor = saturate(material.baseColor.rgb * srgbToLinear(baseColorSample.rgb));
    if (params.mode == kModeBaseColor) {
        colors[outputIndex] = packColor(linearToSrgb(baseColor));
        return;
    }
    float metallic = saturate(material.params.x * metallicRoughnessSample.b);
    float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.001, 1.0);
    float transmission = saturate(material.glassParams.x * transmissionSample.r) * (1.0 - metallic);
    float3 emissive = max(material.emissive.rgb * srgbToLinear(emissiveSample.rgb), float3(0.0));
    float occlusion = lerp(1.0, saturate(occlusionSample.r), saturate(material.textureParams.y));

    OpenPBR_ResolvedInputs inputs = openpbr_make_default_resolved_inputs();
    inputs.base_weight = 1.0;
    inputs.base_color = baseColor;
    inputs.base_diffuse_roughness = roughness;
    inputs.base_metalness = metallic;
    inputs.specular_weight = 1.0;
    inputs.specular_color = float3(1.0);
    inputs.specular_roughness = roughness;
    inputs.specular_roughness_anisotropy = 0.0;
    inputs.specular_ior = clamp(material.glassParams.y, 1.0, 3.0);
    inputs.specular_anisotropy_rotation_cos_sin = float2(1.0, 0.0);
    inputs.transmission_weight = transmission;
    inputs.transmission_color = sqrt(max(baseColor, float3(0.0)));
    inputs.transmission_depth = 0.0;
    inputs.subsurface_weight = 0.0;
    float emissionLuminance = max(emissive.x, max(emissive.y, emissive.z));
    inputs.emission_luminance = emissionLuminance;
    inputs.emission_color = emissionLuminance > 0.0 ? emissive / emissionLuminance : float3(1.0);
    inputs.geometry_opacity = saturate(material.baseColor.a * baseColorSample.a);
    inputs.geometry_thin_walled = false;
    inputs.geometry_basis = openpbr_make_basis(normal, tangent, bitangent);
    inputs.geometry_coat_basis = inputs.geometry_basis;

    OpenPBR_PreparedBsdf prepared = openpbr_prepare(
        inputs,
        float3(1.0),
        OpenPBR_BaseRgbWavelengths_nm,
        OpenPBR_VacuumIor,
        viewDirection);
    float3 reflectedEnvironment = float3(0.0);
    for (uint sampleIndex = 0u; sampleIndex < kEnvironmentSampleCount; ++sampleIndex) {
        float3 localDirection = environmentIntegrationDirection(sampleIndex);
        float3 lightDirection = safeNormalize(
            tangent * localDirection.x +
                normal * localDirection.y +
                bitangent * localDirection.z,
            normal);
        OpenPBR_DiffuseSpecular bsdf = openpbr_eval(prepared, lightDirection);
        float3 incidentRadiance = params.environmentMode == kEnvironmentModeMap ?
            environmentSHRadiance(lightDirection, params, push) :
            environmentRadiance(lightDirection, params, push);
        reflectedEnvironment +=
            max(openpbr_get_sum_of_diffuse_specular(bsdf), float3(0.0)) *
            incidentRadiance;
    }
    float uniformSphereWeight = 4.0 * kPi / float(kEnvironmentSampleCount);
    float3 radiance = max(prepared.emission, float3(0.0)) +
        reflectedEnvironment * (occlusion * uniformSphereWeight);
    colors[outputIndex] = packHdrColor(radiance);
}
