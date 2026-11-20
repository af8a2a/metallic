module GlobalDistanceFieldMip;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceFieldPages;
using Metallic;
using Metallic.DistanceField;

struct GlobalDistanceFieldMipParams
{
    ResourceHandle<Texture3D<uint>> pageTable;
    ResourceHandle<Texture3D<float>> pageAtlas;
    SamplerHandle sampler;
    ResourceHandle<RWTexture3D<unorm float>> previousMip;
    ResourceHandle<RWTexture3D<unorm float>> outputMip;
    uint logicalResolution;
    uint coarseResolution;
    uint clipmapCount;
};

static const float kMipDownsampleFactor = 4.0f;
static const float kMipInfluenceRangeInVoxels = 4.0f;

// UE 5.7.4 GlobalDistanceFieldMip.usf. Both the seed dispatch and the
// subsequent dispatches solve the same Eikonal update on encoded distances.
float eikonal1(float x, float step)
{
    return x + step;
}

float eikonal2(float x, float y, float step)
{
    float sumU = x + y;
    float sumUSq = x * x + y * y;
    float distanceSq = sumU * sumU - 2.0f * (sumUSq - step * step);
    return (1.0f / 2.0f) * (sumU + sqrt(distanceSq));
}

float eikonal3(float x, float y, float z, float step)
{
    float sumU = x + y + z;
    float sumUSq = x * x + y * y + z * z;
    float distanceSq = sumU * sumU - 3.0f * (sumUSq - step * step);
    return (1.0f / 3.0f) * (sumU + sqrt(distanceSq));
}

float minimumEikonalDistance(float center, float candidate)
{
    // Match HLSL min when a negative discriminant produces a NaN. SPIR-V
    // GLSL.std.450 FMin alone does not specify the result for a NaN operand.
    return isnan(candidate) ? center : min(center, candidate);
}

interface IMipDistanceSource
{
    float load(int3 mipCoord);
};

struct FinePageDistanceSource : IMipDistanceSource
{
    Texture3D<uint> pageTable;
    Texture3D<float> pageAtlas;
    SamplerState sampler;
    uint logicalResolution;
    float oneOverCoarseResolution;
    uint clipmapIndex;

    float load(int3 mipCoord)
    {
        // Each rebuilt clipmap is centered on its current page-snapped origin.
        float3 clipmapUV = (float3(mipCoord) + 0.5f) * oneOverCoarseResolution;
        float3 fineGridPosition = clipmapUV * float(logicalResolution);
        float3 pagePosition;
        uint pageIndex = loadGlobalDistanceFieldPage(pageTable, fineGridPosition, pagePosition, clipmapIndex);
        float distanceFieldValue = sampleGlobalDistanceFieldEncodedPage(pageAtlas, sampler, pageIndex, pagePosition);
        if (distanceFieldValue < 1.0f) {
            const float scale = 1.0f / kMipDownsampleFactor;
            const float bias = 0.5f - 0.5f / kMipDownsampleFactor;
            distanceFieldValue = distanceFieldValue * scale + bias;
        }
        return distanceFieldValue;
    }
};

struct PreviousMipDistanceSource : IMipDistanceSource
{
    RWTexture3D<unorm float> texture;
    int clipmapOffsetZ;

    float load(int3 mipCoord)
    {
        mipCoord.z += clipmapOffsetZ;
        return texture[mipCoord];
    }
};

float loadPrevDistanceFieldValue<T : IMipDistanceSource>(T source, int3 mipCoord, int3 offset, int lastTexel)
{
    return source.load(clamp(mipCoord + offset, 0, lastTexel));
}

float propagateMipDistance<T : IMipDistanceSource>(T source, int3 mipCoord, int lastTexel)
{
    float center = loadPrevDistanceFieldValue(source, mipCoord, int3(0, 0, 0), lastTexel);
    float3 v;
    v.x = min(loadPrevDistanceFieldValue(source, mipCoord, int3(-1, 0, 0), lastTexel),
        loadPrevDistanceFieldValue(source, mipCoord, int3(+1, 0, 0), lastTexel));
    v.y = min(loadPrevDistanceFieldValue(source, mipCoord, int3(0, -1, 0), lastTexel),
        loadPrevDistanceFieldValue(source, mipCoord, int3(0, +1, 0), lastTexel));
    v.z = min(loadPrevDistanceFieldValue(source, mipCoord, int3(0, 0, -1), lastTexel),
        loadPrevDistanceFieldValue(source, mipCoord, int3(0, 0, +1), lastTexel));

    float minV = min(min(v.x, v.y), v.z);
    float maxV = max(max(v.x, v.y), v.z);
    float3 sortedV;
    sortedV.x = minV;
    sortedV.y = (v.x != minV && v.x != maxV) ? v.x : ((v.y != minV && v.y != maxV) ? v.y : v.z);
    sortedV.z = maxV;

    const float step = 1.0f / (2.0f * kMipInfluenceRangeInVoxels);
    center = min(center, eikonal1(sortedV.x, step));
    center = minimumEikonalDistance(center, eikonal2(sortedV.x, sortedV.y, step));
    center = minimumEikonalDistance(center, eikonal3(sortedV.x, sortedV.y, sortedV.z, step));
    return center;
}

float quantizeMipDistanceR8(float value)
{
    // Use the Direct3D FLOAT-to-UNORM reference rounding at each write.
    // Vulkan's choice of adjacent UNORM codes can otherwise accumulate.
    return floor(saturate(value) * 255.0f + 0.5f) / 255.0f;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void globalDistanceFieldSeedMipMain(uint3 packedCoord : SV_DispatchThreadID)
{
    GlobalDistanceFieldMipParams params = getParameters<GlobalDistanceFieldMipParams>();
    uint mipSize = params.coarseResolution;
    if (all(packedCoord.xy < mipSize) && packedCoord.z < mipSize * params.clipmapCount) {
        uint clipmapIndex = packedCoord.z / mipSize;
        int3 mipCoord = int3(packedCoord.xy, packedCoord.z % mipSize);
        FinePageDistanceSource source = {resolveUniform(params.pageTable), resolveUniform(params.pageAtlas),
            resolveUniform(params.sampler), params.logicalResolution, 1.0f / float(mipSize), clipmapIndex};
        RWTexture3D<unorm float> outputMip = resolveUniform(params.outputMip);
        outputMip[packedCoord] = quantizeMipDistanceR8(propagateMipDistance(source, mipCoord, int(mipSize) - 1));
    }
}

[shader("compute")]
[numthreads(4, 4, 4)]
void globalDistanceFieldPropagateMipMain(uint3 packedCoord : SV_DispatchThreadID)
{
    GlobalDistanceFieldMipParams params = getParameters<GlobalDistanceFieldMipParams>();
    uint mipSize = params.coarseResolution;
    if (all(packedCoord.xy < mipSize) && packedCoord.z < mipSize * params.clipmapCount) {
        int3 mipCoord = int3(packedCoord.xy, packedCoord.z % mipSize);
        int clipmapOffsetZ = int(packedCoord.z) - mipCoord.z;
        PreviousMipDistanceSource source = {resolveUniform(params.previousMip), clipmapOffsetZ};
        RWTexture3D<unorm float> outputMip = resolveUniform(params.outputMip);
        outputMip[packedCoord] = quantizeMipDistanceR8(propagateMipDistance(source, mipCoord, int(mipSize) - 1));
    }
}
