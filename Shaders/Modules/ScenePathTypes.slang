#language slang 2026
module ScenePathTypes;

import MaterialProgram;

namespace Metallic
{
using Metallic.Material;

// Matches the CPU packed push flag; no resource/push layout change.
public static const uint kScenePathTraceExportUpscalerGuides = 1u << 26u;
public static const uint kScenePathTraceUseGlobalView = 1u << 27u;
public static const uint kScenePathTraceUseFP16Precision = 1u << 28u;
public static const uint kScenePathTraceStrictOpaqueShadows = 1u << 29u;
#if defined(METALLIC_PT_CACHE_QUERY_SPECIALIZATION) && METALLIC_PT_CACHE_QUERY_SPECIALIZATION != 0 && METALLIC_PT_CACHE_QUERY_SPECIALIZATION != 1
#error METALLIC_PT_CACHE_QUERY_SPECIALIZATION must be 0 or 1.
#endif
#if METALLIC_PT_CACHE_QUERY_SPECIALIZATION
#if !defined(SHARC_QUERY) || SHARC_QUERY != 1 || (defined(SHARC_UPDATE) && SHARC_UPDATE != 0)
#error Cache-query specialization requires globally configured QUERY1 and UPDATE0.
#endif
#if defined(METALLIC_PT_SINGLE_SAMPLE) || defined(MATERIAL_CLASS) || defined(MATERIAL_PROGRAM_ID) || defined(METALLIC_DEFERRED_FP16) || defined(METALLIC_DEFERRED_RUNTIME_PRECISION) || defined(METALLIC_DEFERRED_CLASS_SPECIALIZATION)
#error Cache-query specialization excludes SINGLE and manual material/deferred shapes.
#endif
#if !defined(METALLIC_CUSTOM_MATERIALS) || METALLIC_CUSTOM_MATERIALS != 0 || !defined(METALLIC_GLOBAL_VIEW) || METALLIC_GLOBAL_VIEW != 0 || !defined(METALLIC_HAS_RTXCR) || METALLIC_HAS_RTXCR != 1
#error Cache-query specialization requires the builtin local-view RTXCR family.
#endif
#if !defined(METALLIC_DEFERRED_LIGHT_GRID) || METALLIC_DEFERRED_LIGHT_GRID != 0 || !defined(METALLIC_REALTIME_DEFERRED) || METALLIC_REALTIME_DEFERRED != 0 || !defined(METALLIC_DEFERRED_UPSCALER_GUIDES) || METALLIC_DEFERRED_UPSCALER_GUIDES != 0
#error Cache-query specialization excludes Deferred source families.
#endif
#if !defined(METALLIC_WORKING_SPACE_ACESCG) || METALLIC_WORKING_SPACE_ACESCG != 1
#error Cache-query specialization is restricted to ACEScg.
#endif
// Native policy must be supplied explicitly after actual ID/type qualification.
[SpecializationConstant]
public const bool kOpenPBRCacheQueryEnabled = false;
#endif

#if METALLIC_PT_GUIDE_SPECIALIZATION
#if METALLIC_PT_GUIDE_SPECIALIZATION != 1
#error Guide specialization must be 0 or 1.
#endif
#if !defined(METALLIC_PT_CACHE_QUERY_SPECIALIZATION) || METALLIC_PT_CACHE_QUERY_SPECIALIZATION != 1
#error Guide specialization requires the accepted cache-query specialization family.
#endif
// The parent cache marker enforces QUERY1/UPDATE0, builtin local RTXCR/ACEScg,
// and excludes SINGLE and known manual/deferred shapes. Native IDs remain unobserved.
[SpecializationConstant]
public const bool kOpenPBRGuideExportEnabled = false;
#endif
public struct ScenePathTracePush
{
    public float4 eye;
    public float4 center;
    public float4 upProjection;
    public float4 viewport;
    public float4 clipOrtho;
    public float4 previousEye;
    public float4 previousCenter;
    public float4 previousUpProjection;
    public float4 previousViewport;
    public float4 previousClipOrtho;
    public uint width;
    public uint height;
    public uint maxDepth;
    public uint samples;
    public uint accumulationFrame;
    public uint hasHistory;
    public uint enableAccumulation;
    public uint previousCameraValid;
    public uint materialTextureCount;
    public float bitangentFlip;
    public float environmentIntensity;
    public float environmentRotationRadians;
    public uint environmentMode;
    public uint environmentVisible;
    public uint outputLinear;      // when set, write linear HDR for the downstream composite
    public uint cacheMode;         // METALLIC_CACHE_MODE_*
    public uint debugView;
    public uint debugFlags;
    public float jitterOffsetX;
    public float jitterOffsetY;
    public uint sampleFrame;
    public uint temporalJitter;
    public uint deferredSettings; // bin0:15, samples16:20, depth21:25, guides26, global view27, FP16 precision28, strict opaque shadows29
    public uint textureFeedbackEnabled; // host capture decision, then launch-tile sampling
};

// Independent of the path RNG. All invocations of an 8x8 compute group share
// this decision, including SHARC updates where launch and sample pixels differ.
// Ray-pipeline invocation ordering does not guarantee the same subgroup layout.
public ScenePathTracePush withTextureFeedbackSampling(ScenePathTracePush push, uint2 launchPixel)
{
    uint phase = (launchPixel.x >> 3u) ^ (launchPixel.y >> 3u) ^ push.sampleFrame;
    push.textureFeedbackEnabled = push.textureFeedbackEnabled != 0u && (phase & 7u) == 0u ? 1u : 0u;
    return push;
}

public struct Ray
{
    public float3 origin;
    public float3 direction;
};

public struct RayCone
{
    public float width;
    public float spreadAngle;
};

public struct HitInfo
{
#if METALLIC_PT_PATH_DIAGNOSTICS
    public uint diagnosticStreamInstance;
    public uint diagnosticCluster;
    public float2 diagnosticBarycentrics;
#endif
    public float t;
    public float3 position;
    public float3 normal;
    public float3 geometryNormal;
    public float4 tangent;
    public float3 bitangent;
    public float2 texcoord;
    public uint materialIndex;
    public uint instanceIndex;
    public uint primitiveIndex;
    public uint triangleIndex;
    public float rayConeLodConstant;
    public bool frontFace;
};

// Both ray hits and VBuffer reconstruction already produce HitInfo. Preserve
// the authored TBN; UV1 is not present in this legacy vertex/hit representation.
public SurfaceMaterialContext sceneSurfaceMaterialContext(HitInfo hit, TextureFootprint footprint)
{
    SurfaceMaterialContext context = (SurfaceMaterialContext)0;
    context.positionWS = hit.position;
    context.geometryNormalWS = hit.geometryNormal;
    context.shadingNormalWS = hit.normal;
    context.tangentWS = hit.tangent.xyz;
    context.bitangentWS = hit.bitangent;
    context.texCoord0 = hit.texcoord;
    context.texCoordMask = 1u;
    context.frontFace = hit.frontFace ? 1u : 0u;
    context.footprint = footprint;
    return context;
}

} // namespace Metallic
