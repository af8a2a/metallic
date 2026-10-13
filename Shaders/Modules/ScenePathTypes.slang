#language slang 2026
module ScenePathTypes;

import MaterialProgram;

namespace Metallic
{
using Metallic.Material;

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
    public uint deferredSettings; // material shading class
};

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
