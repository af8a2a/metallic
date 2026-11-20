module GlobalDistanceFieldPageBuild;

import ShaderCore;
import ParameterRoot;
import MeshDistanceField;
import GlobalDistanceField;
import GlobalDistanceFieldPages;
using Metallic;
using Metallic.DistanceField;

struct GlobalDistanceFieldPageBuildParams
{
    ResourceHandle<StructuredBuffer<MeshDistanceFieldInstance>> instances;
    ResourceHandle<StructuredBuffer<MeshDistanceFieldMip>> mips;
    ResourceHandle<StructuredBuffer<uint>> indirection;
    ResourceHandle<Texture3D<float>> meshAtlas;
    SamplerHandle meshSampler;
    ResourceHandle<RWTexture3D<uint>> pageTable;
#if defined(METALLIC_GDF_PAGE_ATLAS_R8)
    ResourceHandle<RWTexture3D<unorm float>> pageAtlas;
#else
    ResourceHandle<RWTexture3D<float>> pageAtlas;
#endif
    ResourceHandle<RWStructuredBuffer<GlobalDistanceFieldDomain>> domainOutput;
    ResourceHandle<RWStructuredBuffer<uint>> statistics;
    uint instanceCount;
    uint logicalResolution;
    uint maxPhysicalPages;
    uint clipmapCount;
    // Match the CPU's 16-byte domain alignment in scalar parameter storage.
    uint padding0;
    uint padding1;
    uint padding2;
    GlobalDistanceFieldDomain domains[6];
};

static const uint kRequiredPages = 0u;
static const uint kResidentPages = 1u;
static const uint kPositivePages = 2u;
static const uint kNegativePages = 3u;
static const uint kOverflowPages = 4u;
static const uint kPageStatisticCount = 5u;
static const uint kPageGroupSize = 64u;
static const uint kSamplesPerPage = kGlobalDistanceFieldPageSamples *
    kGlobalDistanceFieldPageSamples * kGlobalDistanceFieldPageSamples;
static const uint kBelowPositiveBand = 1u;
static const uint kAboveNegativeBand = 2u;

groupshared float sharedPageDistances[kSamplesPerPage];
groupshared uint sharedPageVariation;
groupshared uint sharedPhysicalPage;

[shader("compute")]
[numthreads(1, 1, 1)]
void globalDistanceFieldClearPageStatsMain()
{
    GlobalDistanceFieldPageBuildParams params = getParameters<GlobalDistanceFieldPageBuildParams>();
    RWStructuredBuffer<uint> statistics = resolveUniform(params.statistics);
    for (uint index = 0; index < kPageStatisticCount; ++index) {
        statistics[index] = 0u;
    }
}

[shader("compute")]
[numthreads(4, 4, 4)]
void globalDistanceFieldBuildPagesMain(uint3 pageCoordinate : SV_GroupID, uint threadIndex : SV_GroupIndex)
{
    GlobalDistanceFieldPageBuildParams params = getParameters<GlobalDistanceFieldPageBuildParams>();
    RWTexture3D<uint> pageTable = resolveUniform(params.pageTable);
    RWStructuredBuffer<uint> statistics = resolveUniform(params.statistics);
    uint3 tableDimensions;
    pageTable.GetDimensions(tableDimensions.x, tableDimensions.y, tableDimensions.z);
    uint clipmapIndex = pageCoordinate.z / tableDimensions.x;
    uint3 localPage = uint3(pageCoordinate.xy, pageCoordinate.z % tableDimensions.x);
    GlobalDistanceFieldDomain domain = params.domains[clipmapIndex];
    if (threadIndex == 0u && all(localPage == 0u)) {
        RWStructuredBuffer<GlobalDistanceFieldDomain> domainOutput = resolveUniform(params.domainOutput);
        domainOutput[clipmapIndex] = domain;
    }

    if (params.instanceCount == 0u) {
        if (threadIndex == 0u) {
            pageTable[pageCoordinate] = kGlobalDistanceFieldPositivePage;
            InterlockedAdd(statistics[kPositivePages], 1u);
        }
        return;
    }

    if (threadIndex == 0u) {
        sharedPageVariation = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    Texture3D<float> meshAtlas = resolveUniform(params.meshAtlas);
    uint3 meshAtlasDimensions;
    meshAtlas.GetDimensions(meshAtlasDimensions.x, meshAtlasDimensions.y, meshAtlasDimensions.z);
    MeshDistanceFieldResources meshes = {resolveUniform(params.mips), resolveUniform(params.indirection),
        meshAtlas, resolveUniform(params.meshSampler), meshAtlasDimensions.x / 8u};
    StructuredBuffer<MeshDistanceFieldInstance> instances = resolveUniform(params.instances);
    uint3 firstSample = localPage * kGlobalDistanceFieldPageCells;
    float influenceRadius = domain.maxAndInfluenceRadius.w;
    uint variation = 0u;
    for (uint index = threadIndex; index < kSamplesPerPage; index += kPageGroupSize) {
        uint3 local = globalDistanceFieldPageAtlasOffset(index, uint3(kGlobalDistanceFieldPageSamples));
        uint3 sampleCoordinate = min(firstSample + local, params.logicalResolution);
        float3 worldPosition = globalDistanceFieldVoxelCenter(domain, sampleCoordinate);
        float distance = compositeGlobalDistanceField(meshes, instances, params.instanceCount, worldPosition, influenceRadius);
        sharedPageDistances[index] = distance;
        variation |= distance < influenceRadius ? kBelowPositiveBand : 0u;
        variation |= distance > -influenceRadius ? kAboveNegativeBand : 0u;
    }
    InterlockedOr(sharedPageVariation, variation);
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex == 0u) {
        // Collapse only pages proven constant at a signed band limit. A missing
        // page never stands for empty space when the physical pool overflows.
        uint pageIndex;
        if ((sharedPageVariation & kBelowPositiveBand) == 0u) {
            pageIndex = kGlobalDistanceFieldPositivePage;
            InterlockedAdd(statistics[kPositivePages], 1u);
        } else if ((sharedPageVariation & kAboveNegativeBand) == 0u) {
            pageIndex = kGlobalDistanceFieldNegativePage;
            InterlockedAdd(statistics[kNegativePages], 1u);
        } else {
            uint allocation;
            InterlockedAdd(statistics[kRequiredPages], 1u, allocation);
            if (allocation < params.maxPhysicalPages) {
                pageIndex = allocation;
                InterlockedAdd(statistics[kResidentPages], 1u);
            } else {
                pageIndex = kGlobalDistanceFieldOverflowPage;
                InterlockedAdd(statistics[kOverflowPages], 1u);
            }
        }
        sharedPhysicalPage = pageIndex;
        pageTable[pageCoordinate] = pageIndex;
    }
    GroupMemoryBarrierWithGroupSync();

    if (sharedPhysicalPage < kGlobalDistanceFieldOverflowPage) {
#if defined(METALLIC_GDF_PAGE_ATLAS_R8)
        RWTexture3D<unorm float> pageAtlas = resolveUniform(params.pageAtlas);
#else
        RWTexture3D<float> pageAtlas = resolveUniform(params.pageAtlas);
#endif
        uint3 atlasDimensions;
        pageAtlas.GetDimensions(atlasDimensions.x, atlasDimensions.y, atlasDimensions.z);
        uint3 firstAtlasSample = globalDistanceFieldPageAtlasOffset(sharedPhysicalPage,
            atlasDimensions / kGlobalDistanceFieldPageSamples) * kGlobalDistanceFieldPageSamples;
        for (uint index = threadIndex; index < kSamplesPerPage; index += kPageGroupSize) {
            uint3 local = globalDistanceFieldPageAtlasOffset(index, uint3(kGlobalDistanceFieldPageSamples));
            float value = sharedPageDistances[index];
#if defined(METALLIC_GDF_PAGE_ATLAS_R8)
            value = encodeGlobalDistanceFieldPageDistance(value, influenceRadius);
#endif
            pageAtlas[firstAtlasSample + local] = value;
        }
    }
}
