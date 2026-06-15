[[vk::binding(0, 0)]]
RWTexture2D<float4> gOutput;

[[vk::binding(1, 0)]]
Texture2D<float4> gEnvironment;

struct ChessHdrPrefilterPush
{
    uint width;
    uint height;
    uint layerOffset;
    uint sampleCount;
    float roughness;
    uint padding0;
    uint padding1;
    uint padding2;
};

static const float kPi = 3.14159265358979323846;

uint hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float radicalInverseVdc(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint index, uint count)
{
    return float2((float(index) + 0.5) / float(max(count, 1u)), radicalInverseVdc(index));
}

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float3 directionFromTexel(uint2 pixel, uint width, uint height)
{
    float2 uv = (float2(pixel) + 0.5) / float2(max(width, 1u), max(height, 1u));
    float phi = (uv.x - 0.5) * 2.0 * kPi;
    float theta = uv.y * kPi;
    float sinTheta = sin(theta);
    return safeNormalize(float3(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta), float3(0.0, 1.0, 0.0));
}

void basisFromNormal(float3 normal, out float3 tangent, out float3 bitangent)
{
    tangent = abs(normal.z) < 0.999
        ? safeNormalize(cross(float3(0.0, 0.0, 1.0), normal), float3(1.0, 0.0, 0.0))
        : safeNormalize(cross(float3(0.0, 1.0, 0.0), normal), float3(1.0, 0.0, 0.0));
    bitangent = cross(normal, tangent);
}

uint wrapTexel(int value, uint size)
{
    int signedSize = int(max(size, 1u));
    int wrapped = value % signedSize;
    return uint(wrapped < 0 ? wrapped + signedSize : wrapped);
}

uint clampTexel(int value, uint size)
{
    return uint(clamp(value, 0, int(max(size, 1u)) - 1));
}

float3 sampleEnvironment(float3 direction)
{
    uint width = 1u;
    uint height = 1u;
    gEnvironment.GetDimensions(width, height);

    float3 dir = safeNormalize(direction, float3(0.0, 1.0, 0.0));
    float phi = atan2(dir.z, dir.x);
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    float2 uv = float2(frac(0.5 + phi / (2.0 * kPi)), theta / kPi);
    float2 imageCoord = uv * float2(width, height) - 0.5;
    int2 baseCoord = int2(floor(imageCoord));
    float2 blend = frac(imageCoord);

    uint2 p00 = uint2(wrapTexel(baseCoord.x, width), clampTexel(baseCoord.y, height));
    uint2 p10 = uint2(wrapTexel(baseCoord.x + 1, width), p00.y);
    uint2 p01 = uint2(p00.x, clampTexel(baseCoord.y + 1, height));
    uint2 p11 = uint2(p10.x, p01.y);

    float3 c00 = gEnvironment.Load(int3(p00, 0)).rgb;
    float3 c10 = gEnvironment.Load(int3(p10, 0)).rgb;
    float3 c01 = gEnvironment.Load(int3(p01, 0)).rgb;
    float3 c11 = gEnvironment.Load(int3(p11, 0)).rgb;
    return max(lerp(lerp(c00, c10, blend.x), lerp(c01, c11, blend.x), blend.y), float3(0.0, 0.0, 0.0));
}

float3 sampleGgxHalfVector(float2 xi, float alpha)
{
    float alphaSquared = alpha * alpha;
    float phi = 2.0 * kPi * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / max(1.0 + (alphaSquared - 1.0) * xi.y, 0.000001));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void chessHdrPrefilterGlossyMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                                 uniform ChessHdrPrefilterPush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float3 normal = directionFromTexel(pixel, push.width, push.height);
    float roughness = saturate(push.roughness);
    if (roughness <= 0.0001 || push.sampleCount <= 1u) {
        gOutput[uint2(pixel.x, push.layerOffset + pixel.y)] = float4(sampleEnvironment(normal), 1.0);
        return;
    }

    float3 tangent;
    float3 bitangent;
    basisFromNormal(normal, tangent, bitangent);

    float alpha = max(roughness * roughness, 0.0001);
    uint sampleCount = max(push.sampleCount, 1u);
    uint seedOffset = hash(pixel.x + pixel.y * max(push.width, 1u) + uint(roughness * 4096.0));
    float3 accum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    for (uint sampleIndex = 0u; sampleIndex < sampleCount; ++sampleIndex) {
        float2 xi = hammersley(sampleIndex ^ seedOffset, sampleCount);
        float3 localHalf = sampleGgxHalfVector(xi, alpha);
        float3 halfVector = safeNormalize(
            tangent * localHalf.x + bitangent * localHalf.y + normal * localHalf.z,
            normal);
        float3 sampleDirection = safeNormalize(reflect(-normal, halfVector), normal);
        float nDotL = saturate(dot(normal, sampleDirection));
        if (nDotL > 0.0) {
            accum += sampleEnvironment(sampleDirection) * nDotL;
            weightSum += nDotL;
        }
    }

    float3 value = weightSum > 0.0 ? accum / weightSum : sampleEnvironment(normal);
    gOutput[uint2(pixel.x, push.layerOffset + pixel.y)] = float4(value, 1.0);
}
