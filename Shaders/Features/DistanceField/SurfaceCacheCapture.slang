module SurfaceCacheCapture;

import ShaderCore;
import ParameterRoot;
import MaterialTextureDecode;
import LumenSurfaceCache;
using Metallic;
using Metallic.Lumen;

struct SurfaceCacheCaptureVertex
{
    float4 worldPosition;
    float4 worldNormal;
    float4 worldTangent;
    float4 uv;
};

struct SurfaceCacheCaptureParams
{
    uint4 geometry;
    float4 clipRow0;
    float4 clipRow1;
    float4 clipRow2;
    float4 normalAxisX;
    float4 normalAxisY;
    float4 baseColor;
    float4 emissive;
    float4 material;
    uint4 textures;
    float4 textureTransformRows[8];
};

struct SurfaceCacheCaptureInterpolants
{
    float4 position : SV_Position;
    float3 worldNormal : TEXCOORD0;
    float4 worldTangent : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

struct SurfaceCacheCaptureOutputs
{
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 emissive : SV_Target2;
};

static const uint kInvalidTexture = 0xffffffffu;
static const float kPi = 3.1415926535897932f;

float4 sampleCardMaterialTexture(uint texture, SamplerState sampler, float2 uv, float4 row0, float4 row1)
{
    Texture2D<float4> image = resolveUniform(ResourceHandle<Texture2D<float4>>(texture));
    float2 transformedUv = float2(dot(row0.xy, uv) + row0.z, dot(row1.xy, uv) + row1.z);
    return decodeMaterialTextureSample(image.Sample(sampler, transformedUv), row0.w);
}

bool3 isFiniteSurfaceCacheValue(float3 value)
{
    return (asuint(value) & 0x7f800000u) != 0x7f800000u;
}

[shader("vertex")]
SurfaceCacheCaptureInterpolants surfaceCacheCaptureVertexMain(uint vertexIndex : SV_VertexID)
{
    SurfaceCacheCaptureParams params = getParameters<SurfaceCacheCaptureParams>();
    StructuredBuffer<SurfaceCacheCaptureVertex> vertices =
        resolveUniform(ResourceHandle<StructuredBuffer<SurfaceCacheCaptureVertex>>(params.geometry.x));
    SurfaceCacheCaptureVertex vertex = vertices[params.geometry.y + vertexIndex];
    float4 worldPosition = float4(vertex.worldPosition.xyz, 1.0f);

    SurfaceCacheCaptureInterpolants output;
    output.position = float4(dot(worldPosition, params.clipRow0), dot(worldPosition, params.clipRow1),
        dot(worldPosition, params.clipRow2), 1.0f);
    output.worldNormal = vertex.worldNormal.xyz;
    output.worldTangent = vertex.worldTangent;
    output.uv = vertex.uv.xy;
    return output;
}

[shader("fragment")]
SurfaceCacheCaptureOutputs surfaceCacheCaptureFragmentMain(SurfaceCacheCaptureInterpolants input)
{
    SurfaceCacheCaptureParams params = getParameters<SurfaceCacheCaptureParams>();
    SamplerState sampler = resolveUniform(SamplerHandle(params.geometry.z));
    float3 baseColor = params.baseColor.xyz;
    float metallic = params.material.x;
    float3 emissive = params.emissive.xyz;
    float3 worldNormal = input.worldNormal;

    if (params.textures.x != kInvalidTexture) {
        float4 sample = sampleCardMaterialTexture(params.textures.x, sampler, input.uv,
            params.textureTransformRows[0], params.textureTransformRows[1]);
        baseColor = materialTextureModulate(baseColor, sample.rgb, params.textureTransformRows[0].w);
    }
    if (params.textures.y != kInvalidTexture) {
        metallic *= sampleCardMaterialTexture(params.textures.y, sampler, input.uv,
            params.textureTransformRows[2], params.textureTransformRows[3]).b;
    }
    if (params.textures.z != kInvalidTexture) {
        float4 sample = sampleCardMaterialTexture(params.textures.z, sampler, input.uv,
            params.textureTransformRows[4], params.textureTransformRows[5]);
        emissive = materialTextureModulate(emissive, sample.rgb, params.textureTransformRows[4].w);
    }
    if (params.textures.w != kInvalidTexture && input.worldTangent.w != 0.0f) {
        float3 tangentNormal = sampleCardMaterialTexture(params.textures.w, sampler, input.uv,
            params.textureTransformRows[6], params.textureTransformRows[7]).xyz * 2.0f - 1.0f;
        tangentNormal.xy *= params.material.w;
        tangentNormal = normalize(tangentNormal);
        worldNormal = normalize(worldNormal);
        float3 tangent = normalize(input.worldTangent.xyz - worldNormal * dot(worldNormal, input.worldTangent.xyz));
        float3 bitangent = cross(worldNormal, tangent) * (input.worldTangent.w >= 0.0f ? 1.0f : -1.0f);
        worldNormal = tangent * tangentNormal.x + bitangent * tangentNormal.y + worldNormal * tangentNormal.z;
    }

    // UE 5.7.4 LumenCardBasePass.ush, opaque non-Substrate material inputs.
    float specular = 0.5f * params.material.y;
    float3 diffuseColor = baseColor - baseColor * metallic;
    float3 specularColor = lerp(0.08f * specular.xxx, baseColor, metallic.xxx);
    // EnvBRDFApproxFullyRough in BRDF.ush.
    diffuseColor += specularColor * 0.45f;

    float2 cardSpaceNormalXY = float2(0.0f, 0.0f);
    if (all(isFiniteSurfaceCacheValue(worldNormal))) {
        worldNormal = normalize(worldNormal);
        cardSpaceNormalXY = float2(dot(worldNormal, params.normalAxisX.xyz), dot(worldNormal, params.normalAxisY.xyz));
    }

    SurfaceCacheCaptureOutputs output;
    output.albedo = float4(sqrt(diffuseColor), 1.0f);
    output.normal = float4(cardSpaceNormalXY * 0.5f + 0.5f, 0.0f, 1.0f);
    output.emissive = float4(emissive * params.material.z, 0.0f);
    return output;
}

// Bindless texture formats must be carried by the Slang resource type itself.
// _Texture format identifiers 13 and 2 select r16 and rgba16f with float texels.
typealias SurfaceCacheDepthTexture = _Texture<float, __Shape2D, 0, 0, 0, 1, 0, 0, 13>;
typealias SurfaceCacheLightingTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;

struct SurfaceCacheResolveParams
{
    ResourceHandle<Texture2D<float4>> albedo;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<Texture2D<float4>> emissive;
    ResourceHandle<Texture2D<float>> hardwareDepth;
    ResourceHandle<SurfaceCacheDepthTexture> depth;
    ResourceHandle<SurfaceCacheLightingTexture> lighting;
    ResourceHandle<StructuredBuffer<SurfaceCacheCard>> cards;
    uint tileResolution;
    float4 lightDirection;
    float4 illuminance;
};

// UE 5.7.4 SurfaceCache/LumenSurfaceCache.ush:13. Reserve 1 for invalid texels.
float encodeSurfaceCacheDepth(float depth)
{
    float maxValidDepth = float(0xffff - 1 - 0.5f) / float(0xffff);
    return min(depth, maxValidDepth);
}

float3 decodeSurfaceCacheNormal(SurfaceCacheCard card, float2 encodedNormal)
{
    float3 cardSpaceNormal;
    cardSpaceNormal.xy = encodedNormal * 2.0f - 1.0f;
    cardSpaceNormal.z = sqrt(max(1.0f - dot(cardSpaceNormal.xy, cardSpaceNormal.xy), 0.0001f));
    return normalize(card.axisX.xyz * cardSpaceNormal.x + card.axisY.xyz * cardSpaceNormal.y +
        card.axisZ.xyz * cardSpaceNormal.z);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void surfaceCacheResolveMain(uint3 threadID : SV_DispatchThreadID)
{
    SurfaceCacheResolveParams params = getParameters<SurfaceCacheResolveParams>();
    Texture2D<float4> normalAtlas = resolveUniform(params.normal);
    uint width, height;
    normalAtlas.GetDimensions(width, height);
    uint2 texel = threadID.xy;
    if (texel.x >= width || texel.y >= height) {
        return;
    }

    SurfaceCacheDepthTexture depthAtlas = resolveUniform(params.depth);
    SurfaceCacheLightingTexture finalLightingAtlas = resolveUniform(params.lighting);
    float4 encodedNormal = normalAtlas.Load(int3(texel, 0));
    if (encodedNormal.w <= 0.5f) {
        depthAtlas[texel] = 1.0f;
        finalLightingAtlas[texel] = float4(0.0f);
        return;
    }

    Texture2D<float> hardwareDepthAtlas = resolveUniform(params.hardwareDepth);
    depthAtlas[texel] = encodeSurfaceCacheDepth(1.0f - hardwareDepthAtlas.Load(int3(texel, 0)));

    StructuredBuffer<SurfaceCacheCard> cards = resolveUniform(params.cards);
    uint2 tile = texel / params.tileResolution;
    uint cardIndex = tile.x + tile.y * (width / params.tileResolution);
    float3 worldNormal = decodeSurfaceCacheNormal(cards[cardIndex], encodedNormal.xy);
    // DeferredLightingCommon.ush:545, an unshadowed directional light at default diffuse scale.
    float3 directLighting = params.illuminance.xyz * saturate(dot(worldNormal, params.lightDirection.xyz));

    Texture2D<float4> albedoAtlas = resolveUniform(params.albedo);
    Texture2D<float4> emissiveAtlas = resolveUniform(params.emissive);
    float3 encodedAlbedo = albedoAtlas.Load(int3(texel, 0)).xyz;
    float3 albedo = encodedAlbedo * encodedAlbedo;
    float3 emissive = emissiveAtlas.Load(int3(texel, 0)).xyz;
    // CombineFinalLighting, with default diffuse boost and an initially empty indirect atlas.
    float3 diffuseLambert = albedo * (1.0f / kPi);
    float3 finalLighting = directLighting * diffuseLambert + emissive;
    finalLighting = max(select(!isFiniteSurfaceCacheValue(finalLighting), 0.0f, finalLighting), 0.0f);
    finalLightingAtlas[texel] = float4(finalLighting, 0.0f);
}
