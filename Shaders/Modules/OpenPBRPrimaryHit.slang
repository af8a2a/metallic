#language slang 2026
module OpenPBRPrimaryHit;

import Core;
import ScenePathTypes;

namespace Metallic.OpenPBRPrimaryHit
{
using Metallic;

// Explicit scalar wire layout: 24 authored words plus found/frontFace flags.
// No closure, evaluated normal, RNG or propagated ray cone crosses this boundary.
// Retain position and bitangent bits rather than recomputing them in another kernel.
public void storeOpenPbrPrimaryHit(uint2 pixel, ScenePathTracePush push, bool found, HitInfo hit)
{
    let handle = getResourceParameterField<ScenePrimarySplitResourceParameters,
        ResourceHandle<ByteAddressBuffer>>(kScenePrimaryHitsOffset);
    RWByteAddressBuffer buffer = resolveBuffer<RWByteAddressBuffer>(handle);
    uint offset = (pixel.x + pixel.y * push.width) * kOpenPBRPrimaryHitStride;
    if (!found) { hit = (HitInfo)0; }
    buffer.Store4(offset, asuint(float4(hit.t, hit.position)));
    buffer.Store4(offset + 16u, asuint(float4(hit.normal, hit.geometryNormal.x)));
    buffer.Store4(offset + 32u, asuint(float4(hit.geometryNormal.yz, hit.tangent.xy)));
    buffer.Store4(offset + 48u, asuint(float4(hit.tangent.zw, hit.bitangent.xy)));
    buffer.Store4(offset + 64u, uint4(asuint(hit.bitangent.z), asuint(hit.texcoord), hit.materialIndex));
    buffer.Store4(offset + 80u, uint4(hit.instanceIndex, hit.primitiveIndex, hit.triangleIndex,
        asuint(hit.rayConeLodConstant)));
    buffer.Store(offset + 96u, (found ? 1u : 0u) | (hit.frontFace ? 2u : 0u));
}

public bool loadOpenPbrPrimaryHit(uint2 pixel, ScenePathTracePush push, out HitInfo hit)
{
    let handle = getResourceParameterField<ScenePrimarySplitResourceParameters,
        ResourceHandle<ByteAddressBuffer>>(kScenePrimaryHitsOffset);
    ByteAddressBuffer buffer = resolveBuffer<ByteAddressBuffer>(handle);
    uint offset = (pixel.x + pixel.y * push.width) * kOpenPBRPrimaryHitStride;
    uint flags = buffer.Load(offset + 96u);
    hit = (HitInfo)0;
    if ((flags & 1u) == 0u) { return false; }
    float4 a = asfloat(buffer.Load4(offset));
    float4 b = asfloat(buffer.Load4(offset + 16u));
    float4 c = asfloat(buffer.Load4(offset + 32u));
    float4 d = asfloat(buffer.Load4(offset + 48u));
    uint4 e = buffer.Load4(offset + 64u);
    uint4 f = buffer.Load4(offset + 80u);
    hit.t = a.x;
    hit.position = a.yzw;
    hit.normal = b.xyz;
    hit.geometryNormal = float3(b.w, c.xy);
    hit.tangent = float4(c.zw, d.xy);
    hit.bitangent = float3(d.zw, asfloat(e.x));
    hit.texcoord = asfloat(e.yz);
    hit.materialIndex = e.w;
    hit.instanceIndex = f.x;
    hit.primitiveIndex = f.y;
    hit.triangleIndex = f.z;
    hit.rayConeLodConstant = asfloat(f.w);
    hit.frontFace = (flags & 2u) != 0u;
    return true;
}
} // namespace Metallic.OpenPBRPrimaryHit
