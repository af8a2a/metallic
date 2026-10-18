#language slang 2026
module OpenPBRProbe;

import ShaderCore;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRSurface;
import OpenPBRMaterialParameters;
import ScenePathTypes;
import Material;
import MaterialProgram;

namespace Metallic.Tests
{
using Metallic.Material;
using Metallic.OpenPBR;
using Metallic.SceneOpenPBR;

public struct ProbeParams
{
    public BufferSpan<float4> parameters;
    public RWBufferSpan<float4> output;
    public ResourceHandle<Texture2D<float4>> texture;
    public ResourceHandle<RWByteAddressBuffer> counters;
    public uint lightCount;
};

// Raw vendor uint16 tables normalized exactly as the production texture upload.
// Buffer lookup avoids embedding giant arrays in the driver's shader compiler.
public static const uint kLutOffsets[7] = {2, 32770, 33794, 34818, 67586, 68610, 69634};
public float probeLutTexel(ProbeParams probe, int id, int3 p, int3 size)
{
    return probe.parameters.load(kLutOffsets[id] + uint((p.z * size.y + p.y) * size.x + p.x)).x;
}
public float4 probeLut(ProbeParams probe, int id, float3 uvw, int3 size)
{
    float3 p = clamp(uvw * float3(size) - 0.5, float3(0), float3(size - 1));
    int3 lo = int3(floor(p));
    int3 hi = min(lo + 1, size - 1);
    float3 f = p - float3(lo);
    float a = lerp(lerp(probeLutTexel(probe, id, lo, size), probeLutTexel(probe, id, int3(hi.x, lo.yz), size), f.x),
        lerp(probeLutTexel(probe, id, int3(lo.x, hi.y, lo.z), size), probeLutTexel(probe, id, int3(hi.xy, lo.z), size), f.x), f.y);
    float b = lerp(lerp(probeLutTexel(probe, id, int3(lo.xy, hi.z), size), probeLutTexel(probe, id, int3(hi.x, lo.y, hi.z), size), f.x),
        lerp(probeLutTexel(probe, id, int3(lo.x, hi.yz), size), probeLutTexel(probe, id, hi, size), f.x), f.y);
    return float4(lerp(a, b, f.z));
}
public struct ProbeOpenPBRContext : IOpenPBRContext
{
    public static const bool kEnableSheenAndCoat = false;
    public static const bool kEnableDispersion = false;
    public ProbeParams probe;
    public static bool enableSheenAndCoat() { return false; }
    public static bool enableDispersion() { return false; }
    public static bool enableTranslucency() { return true; }
    public static bool enableMetallic() { return true; }
    public float4 sample2D(int id, float2 uv) { return probeLut(probe, id, float3(uv, 0.5), int3(32, id == 6 ? 1 : 32, 1)); }
    public float4 sample3D(int id, float3 uvw) { return probeLut(probe, id, uvw, int3(32)); }
};

// Exercise a second specialization with nonempty context state. Both prepare
// and subsequent evaluation must use the supplied LUT context, not a default.
public struct StatefulOpenPBRContext : IOpenPBRContext
{
    public static const bool kEnableSheenAndCoat = true;
    public static const bool kEnableDispersion = false;
    public ProbeOpenPBRContext base;
    public float scale;
    public static bool enableSheenAndCoat() { return true; }
    public static bool enableDispersion() { return false; }
    public static bool enableTranslucency() { return true; }
    public static bool enableMetallic() { return true; }
    public float4 sample2D(int id, float2 uv) { return base.sample2D(id, uv) * scale; }
    public float4 sample3D(int id, float3 uvw) { return base.sample3D(id, uvw) * scale; }
};

public void countStage(ProbeParams probe, uint index)
{
    uint previous;
    resolveUniform(probe.counters).InterlockedAdd(index * 4, 1, previous);
}

public struct ProbeSource : IOpenPBRMaterialSource
{
    public ProbeParams probe;
    public typealias Context = ProbeOpenPBRContext;
    public bool hasTextures(ScenePathTracePush push) { return true; }
    public Context context() { Context result; result.probe = probe; return result; }
    public OpenPBRMaterialParameters loadParameters(MaterialInstanceRef instance)
    {
        countStage(probe, 0);
        uint variant = instance.resourceOffset;
        float4 factors = probe.parameters.load(instance.parameterOffset / 16);
        OpenPBRMaterialParameters material = (OpenPBRMaterialParameters)0;
        material.baseColor = float4(factors.rgb, 1);
        material.specular = float4(1);
        material.params = float4(variant == 1 ? 1 : 0, variant == 3 ? 0.001 : 0.35, 0, 0);
        material.glassParams = float4(variant == 2 || variant == 3 || variant == 6 ? 1 : 0, 1.5, 0, 0);
        material.textureParams = float4(1, 1, 0, 0);
        material.emissive = float4(variant == 4 ? factors.rgb : float3(0), variant == 5 ? 1 : 0);
        material.baseColorTexture.textureIndex = 1;
        material.baseColorTexture.transform0 = float4(1, 0, 0, 1);
        material.baseColorTexture.transform1 = float4(0, 1, 0, 0);
        material.normalTexture.textureIndex = 2;
        if (variant == 7) { material.baseColor = float4(0); material.specular = float4(0); }
        return material;
    }
    public float4 loadTexture(MaterialTextureInfo info, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint, out bool sampled)
    {
        countStage(probe, 1);
        // Deliberately independent of push.materialTextureCount: generic
        // material resolution must respect the provider's presence decision.
        sampled = info.textureIndex != 0;
        if (!sampled) { return fallback; }
        countStage(probe, 2);
        uint2 texel = info.textureIndex == 2 ? uint2(0, 1) : min(uint2(frac(uv) * 2), uint2(1));
        float4 sample = resolveUniform(probe.texture).Load(int3(texel, 0));
        // Index 3 exercises nontrivial opacity without changing the texture
        // payload shared by the existing normal and lighting probes.
        if (info.textureIndex == 3) { sample.a = 0.25; }
        return sample;
    }
};

public SurfaceMaterialResult<T.Closure> evaluateHit<T : ISurfaceMaterialProgram>(
    Material<T> material, SurfaceMaterialContext context, MaterialInstanceRef instance)
{
    return material.evaluate(context, instance);
}


} // namespace Metallic.Tests
