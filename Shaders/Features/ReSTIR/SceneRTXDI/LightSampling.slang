#language slang 2026
implementing SceneRTXDI;

import Core;
import Lighting;
using Metallic;
using Metallic.Lighting;

#define gEnvironmentMap resolveUniform(getResourceParameters<SceneResourceParameters>().environmentMap)
#define gEnvironmentPdf resolveUniform(getResourceParameters<SceneResourceParameters>().environmentPdf)

float loadImportanceWeight(Texture2D<float> pdfTexture, uint2 position, uint mipLevel)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    pdfTexture.GetDimensions(mipLevel, width, height, mipCount);
    if (position.x >= width || position.y >= height) {
        return 0.0;
    }
    return max(pdfTexture.Load(int3(position, mipLevel)), 0.0);
}

bool sampleImportanceMipmap(
    Texture2D<float> pdfTexture,
    inout uint seed,
    out uint2 position,
    out float pdf)
{
    position = uint2(0u, 0u);
    pdf = 1.0;
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    pdfTexture.GetDimensions(0u, width, height, mipCount);
    uint2 pdfTextureSize = uint2(width, height);
    if (pdfTextureSize.x == 0u || pdfTextureSize.y == 0u || mipCount == 0u) {
        pdf = 0.0;
        return false;
    }
    if (mipCount == 1u) {
        return loadImportanceWeight(pdfTexture, position, 0u) > 0.0;
    }

    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
        float4 weights = float4(
            loadImportanceWeight(pdfTexture, position + uint2(0u, 0u), uint(mipLevel)),
            loadImportanceWeight(pdfTexture, position + uint2(0u, 1u), uint(mipLevel)),
            loadImportanceWeight(pdfTexture, position + uint2(1u, 0u), uint(mipLevel)),
            loadImportanceWeight(pdfTexture, position + uint2(1u, 1u), uint(mipLevel)));
        float weightSum = weights.x + weights.y + weights.z + weights.w;
        if (weightSum <= 0.0) {
            pdf = 0.0;
            return false;
        }
        weights /= weightSum;
        float randomValue = randomFloat(seed);
        if (randomValue < weights.x) {
            pdf *= weights.x;
        } else if (randomValue < weights.x + weights.y) {
            position += uint2(0u, 1u);
            pdf *= weights.y;
        } else if (randomValue < weights.x + weights.y + weights.z) {
            position += uint2(1u, 0u);
            pdf *= weights.z;
        } else {
            position += uint2(1u, 1u);
            pdf *= weights.w;
        }
    }
    return pdf > 0.0 && all(position < pdfTextureSize);
}

float3 visualizeReGIRCell(float3 worldPosition)
{
    uint recordCount, stride;
    resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid).GetDimensions(recordCount, stride);
    if (recordCount < kPunctualReGIRHeaderRecordCount) {
        return float3(0.0);
    }
    uint4 grid = resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid)[0];
    uint4 centerAndCellSize = resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid)[1];
    uint4 metadata = resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid)[2];
    float cellSize = asfloat(centerAndCellSize.w);
    if (metadata.x != kPunctualReGIRMagic ||
        any(grid.xyz == uint3(0u, 0u, 0u)) ||
        !(cellSize > 0.0) || !isfinite(cellSize)) {
        return float3(0.0, 0.0, 0.0);
    }

    float3 gridCenter = asfloat(centerAndCellSize.xyz);
    float3 gridOrigin = gridCenter - float3(grid.xyz) * (cellSize * 0.5);
    float3 cellCoordinates = (worldPosition - gridOrigin) / cellSize;
    int3 cellPosition = int3(floor(cellCoordinates));
    if (any(cellPosition < int3(0, 0, 0)) || any(cellPosition >= int3(grid.xyz))) {
        return float3(0.0, 0.0, 0.0);
    }

    uint cellIndex = uint(cellPosition.x) +
        (uint(cellPosition.y) + uint(cellPosition.z) * grid.y) * grid.x;
    float3 cellColor = hashColor(cellIndex + 0x7f4a7c15u);
    float3 centerDistance = abs(frac(cellCoordinates) - 0.5) * 2.0;
    return cellColor * lerp(0.35, 1.0, max(centerDistance.x, max(centerDistance.y, centerDistance.z)));
}

bool sampleLocalLight(
    SceneRTXDIPush push,
    float3 worldPosition,
    inout uint seed,
    out uint lightIndex,
    out float inverseSourcePdf)
{
    lightIndex = 0u;
    inverseSourcePdf = 0.0;
    if (push.lightCount == 0u) {
        return false;
    }
    if (behaviorEnabled(push, kBehaviorReGIR)) {
        // An empty slot in a valid cell stays a zero candidate. Only the
        // shared selector's out-of-grid path falls back to the global PDF.
        return selectPunctualLight(PunctualSamplingResources(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid), resolveUniform(getResourceParameters<SceneResourceParameters>().lightsPdf)), worldPosition, seed, lightIndex, inverseSourcePdf);
    }
    if (behaviorEnabled(push, kBehaviorLocalLightImportance)) {
        return selectPunctualLightGlobal(PunctualSamplingResources(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid), resolveUniform(getResourceParameters<SceneResourceParameters>().lightsPdf)), seed, lightIndex, inverseSourcePdf);
    }

    lightIndex = min(uint(randomFloat(seed) * float(push.lightCount)), push.lightCount - 1u);
    inverseSourcePdf = float(push.lightCount);
    return true;
}

bool sampleEnvironmentTexel(
    SceneRTXDIPush push,
    inout uint seed,
    out uint2 texel,
    out float texelPdf)
{
    texel = uint2(0u, 0u);
    texelPdf = 0.0;
    uint width = 0u;
    uint height = 0u;
    gEnvironmentMap.GetDimensions(width, height);
    if (!behaviorEnabled(push, kBehaviorEnvironmentEnabled) || width == 0u || height == 0u) {
        return false;
    }

    uint environmentTexelCount = width * height;
    if (!behaviorEnabled(push, kBehaviorEnvironmentImportance)) {
        uint texelIndex = min(
            uint(randomFloat(seed) * float(max(environmentTexelCount, 1u))),
            max(environmentTexelCount, 1u) - 1u);
        texel = uint2(texelIndex % width, texelIndex / width);
        texelPdf = 1.0 / float(max(environmentTexelCount, 1u));
        return texel.y < height;
    }

    if (sampleImportanceMipmap(
            gEnvironmentPdf,
            seed,
            texel,
            texelPdf) &&
        texel.x < width &&
        texel.y < height &&
        texelPdf > 0.0) {
        return true;
    }

    uint texelIndex = min(
        uint(randomFloat(seed) * float(max(environmentTexelCount, 1u))),
        max(environmentTexelCount, 1u) - 1u);
    texel = uint2(texelIndex % width, texelIndex / width);
    texelPdf = 1.0 / float(max(environmentTexelCount, 1u));
    return texel.y < height;
}
