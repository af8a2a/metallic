module GlobalDistanceFieldPageBuild;

import ShaderCore;
import ParameterRoot;
import MeshDistanceField;
import GlobalDistanceField;
import GlobalDistanceFieldPages;
#if defined(METALLIC_GDF_BUILD_OBJECT_GRID)
import GlobalDistanceFieldObjectGrid;
#endif
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
#if defined(METALLIC_GDF_BUILD_OBJECT_GRID)
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldObjectGridSource>> objectGridObjects;
    ResourceHandle<RWStructuredBuffer<uint4>> objectGrid;
    ResourceHandle<RWStructuredBuffer<uint>> objectGridScene;
#else
    uint objectGridObjects;
    uint objectGrid;
    uint objectGridScene;
#endif
    uint2 sceneIdentity;
    // Match the CPU's 16-byte domain alignment in scalar parameter storage.
    uint padding0;
    uint padding1;
    GlobalDistanceFieldDomain domains[6];
#if defined(METALLIC_GLOBAL_DISTANCE_FIELD_PERSISTENT_PAGES)
    ResourceHandle<RWTexture3D<uint>> pageHistory;
    ResourceHandle<RWStructuredBuffer<int>> freeCount;
    ResourceHandle<RWStructuredBuffer<uint>> freePages;
    uint fullUpdate;
    int4 pageMovements[6];
#endif
};

struct GlobalDistanceFieldUpdateStatsParams
{
    ResourceHandle<RWStructuredBuffer<uint>> updateStatistics;
    uint fullUpdate;
    uint updatedPages;
    uint preservedPages;
    uint updatedLevels;
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
[numthreads(1, 1, 1)]
void globalDistanceFieldPublishUpdateStatsMain()
{
    GlobalDistanceFieldUpdateStatsParams params = getParameters<GlobalDistanceFieldUpdateStatsParams>();
    RWStructuredBuffer<uint> updateStatistics = resolveUniform(params.updateStatistics);
    updateStatistics[0] = params.fullUpdate;
    updateStatistics[1] = params.updatedPages;
    updateStatistics[2] = params.preservedPages;
    updateStatistics[3] = params.updatedLevels;
}

#if defined(METALLIC_GLOBAL_DISTANCE_FIELD_PERSISTENT_PAGES)
static const uint kFreePageInitRowSize = 65536u;

bool isLocalPageInRange(int3 page, uint pageCountPerAxis)
{
    return all(page >= 0) && all(page < int(pageCountPerAxis));
}

void countRetainedPage(RWStructuredBuffer<uint> statistics, uint pageIndex)
{
    if (pageIndex == kGlobalDistanceFieldPositivePage) {
        InterlockedAdd(statistics[kPositivePages], 1u);
    } else if (pageIndex == kGlobalDistanceFieldNegativePage) {
        InterlockedAdd(statistics[kNegativePages], 1u);
    } else {
        InterlockedAdd(statistics[kRequiredPages], 1u);
        InterlockedAdd(statistics[pageIndex < kGlobalDistanceFieldOverflowPage ? kResidentPages : kOverflowPages], 1u);
    }
}

// UE 5.7.4 GlobalDistanceField.usf:351, InitPageFreeListCS.
[shader("compute")]
[numthreads(64, 1, 1)]
void globalDistanceFieldInitFreePagesMain(uint3 thread : SV_DispatchThreadID)
{
    GlobalDistanceFieldPageBuildParams params = getParameters<GlobalDistanceFieldPageBuildParams>();
    uint listIndex = thread.x + thread.y * kFreePageInitRowSize;
    if (listIndex < params.maxPhysicalPages) {
        resolveUniform(params.freePages)[listIndex] = params.maxPhysicalPages - 1u - listIndex;
    }
    if (listIndex == 0u) {
        resolveUniform(params.freeCount)[0] = int(params.maxPhysicalPages);
    }
}

// The published table uses logical coordinates. Snapshot moved levels before
// overwriting them and return only pages whose old coordinates leave the clipmap.
[shader("compute")]
[numthreads(4, 4, 4)]
void globalDistanceFieldPreparePageScrollMain(uint3 thread : SV_DispatchThreadID)
{
    GlobalDistanceFieldPageBuildParams params = getParameters<GlobalDistanceFieldPageBuildParams>();
    RWTexture3D<uint> pageTable = resolveUniform(params.pageTable);
    uint3 tableDimensions;
    pageTable.GetDimensions(tableDimensions.x, tableDimensions.y, tableDimensions.z);
    if (any(thread >= tableDimensions)) {
        return;
    }

    uint clipmapIndex = thread.z / tableDimensions.x;
    int4 movement = params.pageMovements[clipmapIndex];
    if (movement.w == 0) {
        return;
    }

    uint pageIndex = pageTable[thread];
    resolveUniform(params.pageHistory)[thread] = pageIndex;
    int3 oldLocalPage = int3(thread.xy, thread.z % tableDimensions.x);
    if (pageIndex < kGlobalDistanceFieldOverflowPage &&
        !isLocalPageInRange(oldLocalPage - movement.xyz, tableDimensions.x)) {
        int freeListLastElementIndex;
        InterlockedAdd(resolveUniform(params.freeCount)[0], 1, freeListLastElementIndex);
        resolveUniform(params.freePages)[freeListLastElementIndex] = pageIndex;
    }
}

// UE 5.7.4 GlobalDistanceField.usf:582, after potentially failed pops.
[shader("compute")]
[numthreads(1, 1, 1)]
void globalDistanceFieldFinalizeFreePagesMain()
{
    GlobalDistanceFieldPageBuildParams params = getParameters<GlobalDistanceFieldPageBuildParams>();
    RWStructuredBuffer<int> freeCount = resolveUniform(params.freeCount);
    freeCount[0] = max(freeCount[0], 0);
}
#endif

[shader("compute")]
[numthreads(4, 4, 4)]
void globalDistanceFieldBuildPagesMain(uint3 pageCoordinate : SV_GroupID, uint3 cellCoordinate : SV_GroupThreadID,
    uint threadIndex : SV_GroupIndex)
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
#if defined(METALLIC_GDF_BUILD_OBJECT_GRID)
        if (clipmapIndex == 0u) {
            RWStructuredBuffer<uint> objectGridScene = resolveUniform(params.objectGridScene);
            objectGridScene[0] = params.sceneIdentity.x;
            objectGridScene[1] = params.sceneIdentity.y;
        }
#endif
    }

#if defined(METALLIC_GLOBAL_DISTANCE_FIELD_PERSISTENT_PAGES)
    if (params.fullUpdate == 0u) {
        int4 movement = params.pageMovements[clipmapIndex];
        int3 oldLocalPage = int3(localPage) + movement.xyz;
        if (isLocalPageInRange(oldLocalPage, tableDimensions.x)) {
            if (threadIndex == 0u) {
                uint pageIndex;
                if (movement.w != 0) {
                    uint3 oldPage = uint3(oldLocalPage) + uint3(0, 0, clipmapIndex * tableDimensions.x);
                    pageIndex = resolveUniform(params.pageHistory)[oldPage];
                    pageTable[pageCoordinate] = pageIndex;
                } else {
                    pageIndex = pageTable[pageCoordinate];
                }
                countRetainedPage(statistics, pageIndex);
            }
            return;
        }
    }
#endif

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
#if defined(METALLIC_GLOBAL_DISTANCE_FIELD_PERSISTENT_PAGES)
            // UE 5.7.4 GlobalDistanceField.usf:504, pop the last free element.
            int freeListSize;
            InterlockedAdd(statistics[kRequiredPages], 1u);
            InterlockedAdd(resolveUniform(params.freeCount)[0], -1, freeListSize);
            if (freeListSize > 0) {
                pageIndex = resolveUniform(params.freePages)[freeListSize - 1];
#else
            uint allocation;
            InterlockedAdd(statistics[kRequiredPages], 1u, allocation);
            if (allocation < params.maxPhysicalPages) {
                pageIndex = allocation;
#endif
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
#if defined(METALLIC_GDF_BUILD_OBJECT_GRID)
        compositeGlobalDistanceFieldObjectGridPage(domain, localPage, cellCoordinate, threadIndex, sharedPhysicalPage,
            meshes, instances, resolveUniform(params.objectGridObjects), params.instanceCount,
            resolveUniform(params.objectGrid));
#endif
    }
}
