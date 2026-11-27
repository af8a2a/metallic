module FogSkyOcclusion;

import ShaderCore;
import GlobalDistanceField;
import GlobalDistanceFieldPages;
using Metallic;
using Metallic.DistanceField;

namespace Metallic.Fog
{
// rcpFast is from Shader Fast Math Lib (v0.41), Copyright (c) 2014 Michal Drobot.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
float rcpFast(float x)
{
    int i = asint(x);
    i = 0x7EF311C2 - i;
    return asfloat(i);
}

// UE 5.7.4 GlobalDistanceFieldShared.ush, SampleGlobalDistanceField.
float sampleGlobalDistanceField(float3 worldPosition, float maxDistance, uint minClipmapIndex,
    StructuredBuffer<GlobalDistanceFieldDomain> domains, uint clipmapCount,
    Texture3D<uint> pageTable, Texture3D<float> pageAtlas, SamplerState sampler)
{
    for (uint clipmapIndex = minClipmapIndex; clipmapIndex < clipmapCount; ++clipmapIndex) {
        GlobalDistanceFieldDomain domain = domains[clipmapIndex];
        float3 gridPosition = (worldPosition - domain.minAndVoxelSize.xyz) / domain.minAndVoxelSize.w;
        float3 pagePosition;
        uint page = loadGlobalDistanceFieldPage(pageTable, gridPosition, pagePosition, clipmapIndex);
        float distanceFieldValue = sampleGlobalDistanceFieldEncodedPage(pageAtlas, sampler, page, pagePosition);
        if (distanceFieldValue < 1.0f) {
            return decodeGlobalDistanceFieldPageDistance(distanceFieldValue, domain.maxAndInfluenceRadius.w);
        }
    }
    return maxDistance;
}

// UE 5.7.4 VolumetricFog.usf, HemisphereConeTraceAgainstGlobalDistanceFieldClipmap.
float hemisphereConeTraceAgainstGlobalDistanceFieldClipmap(uint clipmapIndex, float clipmapExtent,
    float3 worldShadingPosition, float3 coneDirection, float tanConeHalfAngle, float maxOcclusionDistance,
    StructuredBuffer<GlobalDistanceFieldDomain> domains, uint clipmapCount,
    Texture3D<uint> pageTable, Texture3D<float> pageAtlas, SamplerState sampler)
{
    // UE's fast reciprocal depends on the input bits. March in centimetres and
    // convert to metres only at the boundary of the Metallic distance field.
    float clipmapExtentCm = clipmapExtent * 100.0f;
    float maxOcclusionDistanceCm = maxOcclusionDistance * 100.0f;
    float minStepSize = clipmapExtentCm * 2.0f / 100.0f;
    float invMaxOcclusionDistance = 1.0f / maxOcclusionDistanceCm;
    float minVisibility = 1.0f;
    float worldStepOffset = 2.0f;
    [loop]
    for (uint stepIndex = 0; stepIndex < 10 && worldStepOffset < maxOcclusionDistanceCm; ++stepIndex) {
        float3 worldSamplePosition = worldShadingPosition + coneDirection * (worldStepOffset * 0.01f);
        float distanceToOccluder = sampleGlobalDistanceField(worldSamplePosition, maxOcclusionDistance,
            clipmapIndex, domains, clipmapCount, pageTable, pageAtlas, sampler) * 100.0f;
        float sphereRadius = worldStepOffset * tanConeHalfAngle;
        float invSphereRadius = rcpFast(sphereRadius);
        float visibility = saturate(distanceToOccluder * invSphereRadius);
        float occluderDistanceFraction = (worldStepOffset + distanceToOccluder) * invMaxOcclusionDistance;
        visibility = max(visibility, saturate(occluderDistanceFraction * occluderDistanceFraction * 0.6f));
        minVisibility = min(minVisibility, visibility);
        worldStepOffset += max(distanceToOccluder, minStepSize);
    }
    return minVisibility;
}

// UE 5.7.4 VolumetricFog.usf, ComputeSkyVisibility and HemisphereConeTraceAgainstGlobalDistanceField.
public float computeSkyVisibility(float3 worldPosition, ResourceHandle<Texture3D<uint>> pageTable,
    ResourceHandle<Texture3D<float>> pageAtlas, ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domains,
    SamplerHandle sampler, float maxOcclusionDistance, uint clipmapCount)
{
    if (clipmapCount == 0u) {
        return 1.0f;
    }
    let domainBuffer = resolveUniform(domains);
    for (uint clipmapIndex = 0; clipmapIndex < clipmapCount; ++clipmapIndex) {
        GlobalDistanceFieldDomain domain = domainBuffer[clipmapIndex];
        float3 center = (domain.minAndVoxelSize.xyz + domain.maxAndInfluenceRadius.xyz) * 0.5f;
        float3 extent = (domain.maxAndInfluenceRadius.xyz - domain.minAndVoxelSize.xyz) * 0.5f;
        float3 distancesToMin = max(worldPosition - center + extent, 0.0f);
        float3 distancesToMax = max(center + extent - worldPosition, 0.0f);
        float3 closestDistances = min(distancesToMin, distancesToMax);
        float distanceFromClipmap = min(closestDistances.x, min(closestDistances.y, closestDistances.z));
        if (distanceFromClipmap > maxOcclusionDistance) {
            float tanConeHalfAngle = tan(3.14159265358979323846f / 4.0f);
            return hemisphereConeTraceAgainstGlobalDistanceFieldClipmap(clipmapIndex, extent.x, worldPosition,
                float3(0, 1, 0), tanConeHalfAngle, maxOcclusionDistance, domainBuffer, clipmapCount,
                resolveUniform(pageTable), resolveUniform(pageAtlas), resolveUniform(sampler));
        }
    }
    return 1.0f;
}
} // namespace Metallic.Fog
