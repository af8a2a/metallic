#language slang 2026
implementing GPUDrivenStreamAsset;

// Shared push data, GPU records and streaming constants.
public static const uint kMaxMeshletVertices = 128;
public static const uint kMaxMeshletTriangles = 128;
public static const uint kMeshletTriangleChunkSize = 64u;
public static const uint kMeshletTriangleChunkCount = 2u;
public static const uint kPayloadHeaderBytes = 112u;
public static const uint kPayloadClusterBytes = 96u;
public static const uint kPayloadClusterWords = kPayloadClusterBytes / 4u;
public static const uint kPayloadAttributePosition = 1u << 0u;
public static const uint kPayloadAttributeNormal = 1u << 1u;
public static const uint kPayloadAttributeTexcoord0 = 1u << 2u;
public static const uint kPayloadAttributeTangent = 1u << 4u;
public static const uint kPayloadFormatFloat32x2 = 1u;
public static const uint kPayloadFormatFloat32x4 = 2u;
public static const uint kPayloadFormatFloat32x3 = 4u;
public static const uint kInvalidStreamDeviceOffset = 0xffffffffu;
public static const uint kInvalidClusterIndex = 0xffffffffu;
public static const uint kUnloadClusterIndex = 0xfffffffeu;
public static const uint kStreamStatePendingUpload = 1u;
public static const uint kStreamStateResident = 2u;
public static const uint kStreamStateLockedFallback = 3u;
public static const uint kStreamPageStateMask = 0x00000007u;
public static const uint kStreamPageDeviceOffsetMask = 0xfffffff8u;
public static const uint kNoDebugLODOverride = 0xffffffffu;
public static const uint kActiveGroupResident = 1u << 0u;
public static const uint kActiveGroupLoadRequest = 1u << 1u;
public static const uint kActiveGroupUnloadRequest = 1u << 2u;
public static const uint kTraversalPhaseLoad = 0u;
public static const uint kTraversalPhaseUnload = 1u;
public static const uint kTraversalPhaseClearPriority = 2u;
public static const uint kTraversalPhaseGatherPriority = 3u;
public static const uint kActiveBuildPhaseReset = 0u;
public static const uint kActiveBuildPhaseBuild = 1u;
public static const uint kActiveBuildPhaseFinalize = 2u;
public static const uint kActiveBuildPhaseSeed = 3u;
public static const uint kActiveBuildPhaseRun = 4u;
public static const uint kActiveBuildPhaseFrontier = 5u;
public static const uint kActiveBuildPhasePrefix = 6u;
public static const uint kActiveBuildPhaseEmit = 7u;
public static const uint kActiveBuildPhaseInitializeLODState = 8u;
public static const uint kActiveBuildPhasePrefetch = 9u;
public static const uint kActiveBuildPhaseClear = 10u;
public static const uint kActiveBuildPhaseMask = 11u;
public static const uint kActiveBuildPhaseDemandReset = 12u;
public static const uint kActiveBuildPhaseDemand = 13u;
public static const uint kStreamDemandStatsWords = 32u;
public static const uint kLODTelemetryFrameMask = 0x03ffffffu;
public static const uint kLODObservedDemand = 1u;
public static const uint kLODObservedSelected = 2u;
public static const uint kLODPreviouslySelected = 4u;
public static const uint kLODThresholdCandidate = 8u;
public static const uint kLODCatchupCandidate = 16u;
public static const uint kLODObservationEligible = 32u;
public static const uint kBLASInputPhaseReset = 0u;
public static const uint kBLASInputPhaseCount = 1u;
public static const uint kBLASInputPhaseSetup = 2u;
public static const uint kBLASInputPhaseInsert = 3u;
public static const uint kCLASPageStateActive = 1u;
public static const uint kCLASPageAddressOffsetMask = 0x3fffffffu;
public static const uint kCLASPageStateShift = 30u;
public static const uint kBLASInstanceFallback = 1u << 0u;
public static const uint kBLASInstanceDynamic = 1u << 1u;
public static const uint kBLASInstanceOverflow = 1u << 2u;
public static const uint kTraversalStackCapacity = 128u;
public static const uint kStreamRequestHeaderWords = 16u;
public static const uint kStreamUpdateHeaderWords = 16u;
public static const uint kStreamRequestMaxLoadsWord = 0u;
public static const uint kStreamRequestMaxUnloadsWord = 1u;
public static const uint kStreamRequestLoadCounterWord = 2u;
public static const uint kStreamRequestUnloadCounterWord = 3u;
public static const uint kStreamRequestFrameIndexWord = 4u;
public static const uint kStreamRequestLoadOverflowCounterWord = 5u;
public static const uint kStreamRequestUnloadOverflowCounterWord = 6u;
public static const uint kStreamRequestInvalidPageCounterWord = 7u;
public static const uint kStreamRequestLastLoadOverflowFrameWord = 8u;
public static const uint kStreamRequestLastUnloadOverflowFrameWord = 9u;
public static const uint kStreamRequestLastInvalidPageFrameWord = 10u;
public static const uint kStreamRequestLoadPriorityOffsetWord = 11u;
public static const uint kStreamRequestPriorityTableOffsetWord = 12u;
public static const uint kStreamRequestPrefetchLimitWord = 13u;
public static const uint kStreamRequestPrefetchCounterWord = 14u;
public static const uint kStreamRequestPrefetchDroppedWord = 15u;
public static const uint kStreamPrefetchPageTag = 1u << 31u;

public struct GPUDrivenStreamAssetPush
{
    public uint pageBuffer;
    public uint activeGroupBuffer;
    public uint pageTableBuffer;
    public uint paramsBuffer;
    public uint requestBuffer;
    public uint residentPageBuffer;
    public uint updateBuffer;
    public uint activeHeaderBuffer;
    public uint instanceBuffer;
    public uint primitiveBuffer;
    public uint lodLevelBuffer;
    public uint groupBuffer;
    public uint nodeBuffer;
    public uint drawIndirectBuffer;
    public uint traversalHeaderBuffer;
    public uint traversalWorkBuffer;
    public uint clasAddressBuffer;
    public uint clasPageTableBuffer;
    public uint blasHeaderBuffer;
    public uint instanceBlasBuffer;
    public uint blasBuildInfoBuffer;
    public uint blasClusterReferenceBuffer;
    public uint fallbackBlasAddressBuffer;
    public uint dynamicBlasAddressBuffer;
    public uint tlasInstanceBuffer;
    public uint traversalPhase;
    public uint activeBuildPhase;
    public uint rasterBindingsBuffer;
    public uint hybridQueueBuffer;
    public uint hybridClusterBuffer;
    public uint clasPublicationRevision;
    public float tessellationEdgePixels;
    public uint tessellationMaxFactor;
    public uint tessellationMaxSplitDepth;
    public uint pageTableMutationSerial;
};

public struct GPUDrivenStreamAssetRasterBindings
{
    public uint visibleClusterBuffer;
    public uint instanceVisibilityBuffer;
    public uint hzbBuffer0;
    public uint hzbBuffer1;
    public uint depthImage;
    public uint visibilityImage;
    public uint deferredColorBuffer;
    public uint visibleInstanceIdsBuffer;
    public uint visibleRecordBase;
    public uint visibleRecordCapacity;
    public uint hzbMipCount;
    public uint hzbValid;
    public uint cullingFlags;
    public uint width;
    public uint height;
    public uint visibleInstanceCounterBuffer;
    public uint gpuSceneInstanceBuffer;
    public uint tessellationBuffer;
    public float displacementBound;
    public uint classificationFlags; // bit 0 disables metadata classification for A/B
    public uint materialBuffer;
    public uint materialTextureRemapBuffer;
    public uint materialTextureCount;
    public uint coverageInputs;
    public uint candidateInstanceIdsBuffer;
    public uint candidateInstanceCount;
    public uint instanceVisibilityResetCount;
};

public struct GPUDrivenStreamAssetActiveHeader
{
    public uint activeGroupCount;
    public uint activeGroupCapacity;
    public uint maxActiveGroupClusters;
    public uint overflowCount;
    public uint frameIndex;
    public uint padding0;
    public uint padding1;
    public uint padding2;
};

public struct GPUDrivenStreamAssetActiveGroup
{
    public uint pageDeviceOffsetBytes;
    public uint pageIndex;
    public uint clusterCount;
    public uint primitiveIndex;
    public uint lodLevel;
    public uint materialIndex;
    public uint clusterSelectionMask;
    public uint flags;
    public uint instanceIndex;
    public uint gpuSceneInstanceIndex;
    public uint padding1;
    public uint padding2;
    public float4 world0;
    public float4 world1;
    public float4 world2;
    public float4 world3;
};

public struct GPUDrivenStreamAssetInstance
{
    public uint primitiveIndex;
    public uint materialIndex;
    public uint visible;
    public uint gpuSceneInstanceIndex;
    public float4 world0;
    public float4 world1;
    public float4 world2;
    public float4 world3;
    public float4 boundsCenterRadius;
};

public struct GPUDrivenStreamAssetPrimitive
{
    public uint lodLevelOffset;
    public uint lodLevelCount;
    public uint pageOffset;
    public uint pageCount;
    public uint fallbackPageOffset;
    public uint fallbackPageCount;
    public uint groupOffset;
    public uint groupCount;
    public uint fallbackGroupOffset;
    public uint fallbackGroupCount;
    public uint materialIndex;
    public uint nodeOffset;
    public uint nodeCount;
    public uint lodBvhOffset;
    public uint lodBvhNodeCount;
    public uint lodTileOffset;
};

public struct GPUDrivenStreamAssetLODLevel
{
    public uint pageOffset;
    public uint pageCount;
    public uint lodLevel;
    public uint clusterCount;
    public float minBoundingSphereRadius;
    public float minMaxQuadricError;
    public uint padding0;
    public uint padding1;
};

public struct GPUDrivenStreamAssetGroup
{
    public uint primitiveIndex;
    public uint pageIndex;
    public uint lodLevel;
    public uint clusterCount;
    public float boundsCenterX;
    public float boundsCenterY;
    public float boundsCenterZ;
    public float boundsRadius;
    public float maxQuadricError;
    public uint clusterRefinedOffset;
    public uint flags;
    public uint parentOffset;
    public uint parentCount;
    public float demandMinX;
    public float demandMinY;
    public float demandMinZ;
    public float demandMaxX;
    public float demandMaxY;
    public float demandMaxZ;
};

public struct GPUDrivenStreamAssetNode
{
    public uint primitiveIndex;
    public uint childOffset;
    public uint childCount;
    public uint groupIndex;
    public float4 boundsCenterRadius;
    public float maxQuadricError;
    public uint lodLevel;
    public uint padding0;
    public uint padding1;
};

public struct GPUDrivenStreamAssetDrawIndirect
{
    public uint groupCountX;
    public uint groupCountY;
    public uint groupCountZ;
};

public struct GPUDrivenStreamAssetTraversalHeader
{
    public uint readCounter;
    public uint writeCounter;
    public uint taskCounter;
    public uint overflowCount;
    public uint frameIndex;
    public uint padding0;
    public uint padding1;
    public uint padding2;
};

public struct GPUDrivenStreamAssetTraversalWorkItem
{
    public uint instanceIndex;
    public uint nodeIndex;
    public uint readyFrame;
    public uint padding0;
};

public struct GPUDrivenStreamAssetCLASPageEntry
{
    public uint addressOffsetAndState;
    public uint publicationGeneration;
};

uint clasPageAddressOffset(GPUDrivenStreamAssetCLASPageEntry entry)
{
    return entry.addressOffsetAndState & kCLASPageAddressOffsetMask;
}

uint clasPageState(GPUDrivenStreamAssetCLASPageEntry entry)
{
    return entry.addressOffsetAndState >> kCLASPageStateShift;
}

public struct GPUDrivenStreamAssetBLASHeader
{
    public uint clusterReferenceCount;
    public uint blasBuildCount;
    public uint clusterReferenceCapacity;
    public uint overflowCount;
    public uint frameIndex;
    public uint padding0;
    public uint padding1;
    public uint semanticChangeCount;
    public uint requestedClusterReferences;
    public uint requestedInstances;
    public uint referenceBudgetRejected;
    public uint buildBudgetRejected;
    public uint oversizedInstances;
    public uint missingClasInstances;
    public uint invalidGroups;
    public uint liveClusterReferences;
    public uint dirtyInstances;
    public uint reusedInstances;
    public uint storageRejected;
    public uint arenaUsedBytes;
    public uint arenaRepack;
    public uint arenaBase;
    public uint admittedInstances;
    public uint publicationInvalidated;
    public uint representationCount;
    public uint sharingInstanceCount;
    public uint mergedInstanceCount;
    public uint cachedInstanceCount;
    public uint representationGroupCount;
    public uint representationGroupOverflow;
    public uint cacheDependencyInvalidations;
    public uint representationPadding;
    public uint tessGeometryAddressLow, tessGeometryAddressHigh;
    public uint tessPatchCount, tessRejectedInstances;
};

public struct GPUDrivenStreamAssetInstanceBLAS
{
    public uint clusterReferenceOffset;
    public uint clusterReferenceCapacity;
    public uint selectedClusterCount;
    public uint insertedClusterCount;
    public uint blasBuildIndex;
    public uint flags;
    public uint firstGroup;
    public uint groupCount;
    public uint previousFirstGroup;
    public uint previousGroupCount;
    public uint storageOffset;
    public uint storageCapacity;
    public uint allocationPrefix;
    public uint packedPrefix;
    public uint buildPrefix;
    // 0: no cut keys; 1: live BLAS; 2: examined cut rejected by exact storage.
    public uint cachedValid;
};

public struct GPUDrivenStreamAssetBLASBuildInfo
{
    public uint clusterReferencesCount;
    public uint clusterReferencesStride;
    public uint clusterReferencesAddressLow;
    public uint clusterReferencesAddressHigh;
};

public struct GPUDrivenStreamAssetTLASInstance
{
    public float4 transform0;
    public float4 transform1;
    public float4 transform2;
    public uint customIndexMask;
    public uint shaderBindingTableFlags;
    public uint accelerationStructureReferenceLow;
    public uint accelerationStructureReferenceHigh;
};

public struct GPUDrivenStreamAssetPageTableEntry
{
    public uint deviceOffsetAndState;
    public uint lastRequestFrame;
};

uint streamPageDeviceOffset(GPUDrivenStreamAssetPageTableEntry entry)
{
    return (entry.deviceOffsetAndState & kStreamPageStateMask) == 0u
        ? kInvalidStreamDeviceOffset
        : entry.deviceOffsetAndState & kStreamPageDeviceOffsetMask;
}

uint streamPageState(GPUDrivenStreamAssetPageTableEntry entry)
{
    return entry.deviceOffsetAndState & kStreamPageStateMask;
}

public struct GPUDrivenStreamAssetRequestHeader
{
    public uint maxLoadRequests;
    public uint maxUnloadRequests;
    public uint loadCounter;
    public uint unloadCounter;
    public uint frameIndex;
    public uint loadOverflowCounter;
    public uint unloadOverflowCounter;
    public uint invalidPageCounter;
    public uint lastLoadOverflowFrame;
    public uint lastUnloadOverflowFrame;
    public uint lastInvalidPageFrame;
    public uint padding0;
    public uint padding1;
    public uint padding2;
    public uint padding3;
    public uint padding4;
};

public struct GPUDrivenStreamAssetUpdateHeader
{
    public uint patchUnloadPageCount;
    public uint patchPageCount;
    public uint frameIndex;
    public uint patchOverflowCounter;
    public uint padding0;
    public uint padding1;
    public uint padding2;
    public uint padding3;
    public uint padding4;
    public uint padding5;
    public uint padding6;
    public uint padding7;
    public uint padding8;
    public uint padding9;
    public uint padding10;
    public uint padding11;
};

public struct GPUDrivenStreamAssetParams
{
    public float4 eye;
    public float4 center;
    public float4 upProjection;
    public float4 viewport;
    public float4 clipOrtho;
    public float4 clearColor;
    public uint debugColorMode;
    public uint pageBufferBytes;
    public uint drawTaskCount;
    public uint frameIndex;
    public uint maxGpuPageRequests;
    public uint maxGpuPageUnloadRequests;
    public uint activeGroupCount;
    public uint maxActiveGroupClusters;
    public uint sceneInstanceCount;
    public uint scenePrimitiveCount;
    public uint sceneLodLevelCount;
    public uint scenePageCount;
    public uint selectedLodLevel;
    public uint enableGpuLodSelection;
    public uint enableGpuUnloadRequests;
    public uint sceneGroupCount;
    public uint maxPrimitiveGroupCount;
    public uint sceneNodeCount;
    public uint traversalWorkerCount;
    public uint traversalWorkCapacity;
    public uint blasClusterReferenceAddressLow;
    public uint blasClusterReferenceAddressHigh;
    public uint blasClusterReferenceCapacity;
    public uint blasBuildCapacity;
    public float4 previousEye;
    public float4 previousCenter;
    public float4 previousUpProjection;
    public float4 previousViewport;
    public float4 previousClipOrtho;
    public float lodPixelError; // Display-pixel budget converted to internal render pixels by the CPU.
    public uint lodTopologyBuffer;
    public uint lodStateBuffer;
    public uint lodInstanceOffsetsOffset;
    public float4 renderEye;
    public float4 renderCenter;
    public float4 renderUpProjection;
    public float4 renderViewport;
    public float4 renderClipOrtho;
    public float4 prefetchParams;
    public uint demandBuffer;
    public uint demandTaskOffset;
    public uint demandTaskCount;
    public uint splitFrontier;
    public uint demandInstanceOffsetsOffset;
    public uint demandStatsBuffer;
    public uint maxBlasClustersPerBuild;
    public uint blasStorageBytes;
    public uint blasStorageAddressLow;
    public uint blasStorageAddressHigh;
    public uint blasInstanceReusePadding;
    public uint gpuSceneInstanceBufferIndexPlusOne;
    public uint blasStoragePadding[32];
    public float4 prefetchEye;
    public float4 prefetchCenter;
    public float4 prefetchUp;
    public uint enableLodTransitionTelemetry;
    public float previousLodPixelError;
    public uint rayClasPageTableBuffer;
    public uint enableTerminalMaskDiagnostics;
    public float rtOffscreenErrorScale;
    public float rtLodHysteresis;
    public uint rtLodFlags;
    public uint rtLodPolicyRevision;
    public uint rtLodFrameToken;
    public uint rtRepresentationBuffer;
    public uint rtRepresentationGroupBuffer;
    public uint rtRepresentationHeaderBuffer;
    public uint rtRepresentationStateBuffer;
    public uint rtRepresentationMetadataBuffer;
    public uint rtRepresentationCount;
    public uint rtRepresentationGroupCapacity;
    public uint enableBlasRepresentations;
    public uint rtRepresentationCacheBuffer;
    public uint rtRepresentationCachePadding0;
    public uint rtRepresentationCachePadding1;
    public uint rtRepresentationCachePadding2;
    public uint blasGPUAllocatorPadding;
    public uint blasPriorityPadding;
    public uint blasPriorityCachePadding;
    public uint blasMergedSharingPadding;
    public uint blasFallbackPadding;
};

public struct GPUDrivenStreamAssetVertexOutput
{
    public float4 position : SV_Position;
    public nointerpolation uint doubleSided : TEXCOORD1;
    public float2 texcoord : TEXCOORD2;
    public nointerpolation uint instanceIndex : TEXCOORD3;
};

// IDs vary per triangle, so they must not use a shared vertex's flat attribute.
public struct GPUDrivenStreamAssetPrimitiveOutput
{
    public uint visibilityId : SV_PrimitiveID;
};

public struct GPUDrivenStreamAssetFullscreenOutput
{
    public float4 position : SV_Position;
};

