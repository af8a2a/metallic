#pragma once

#include "Runtime/Render/RenderFrameContext.h"

#include <unordered_set>

#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/Profiling/CpuProfile.h"
#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/ResourceRegistry.h"
#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/Streamer/MeshletStreamClas.h"
#include "Runtime/Render/Streamer/MeshletStreamResidency.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <json.hpp>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {

class MeshletStreamClasPool;
struct StreamSceneReadiness {
    uint32_t requiredPages = 0;
    uint32_t completedPages = 0;
    bool ready = true;
    float fraction() const { return requiredPages ? float(completedPages) / float(requiredPages) : 0.f; }
};
struct DebugResourceBinding;

inline constexpr const char* kMeshletStreamShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
inline constexpr const char* kMeshletStreamShaderModuleName = "Features/GPUDriven/GPUDrivenStreamAsset";
inline constexpr const char* kMeshletStreamMeshEntryPoint = "gpuDrivenStreamAssetMeshMain";
inline constexpr const char* kMeshletStreamFragmentEntryPoint = "gpuDrivenStreamAssetFragmentMain";
inline constexpr const char* kMeshletStreamDeferredEntryPoint = "gpuDrivenStreamAssetDeferredMain";
inline constexpr const char* kMeshletStreamCompositeVertexEntryPoint =
    "gpuDrivenStreamAssetCompositeVertexMain";
inline constexpr const char* kMeshletStreamCompositeFragmentEntryPoint =
    "gpuDrivenStreamAssetCompositeFragmentMain";
inline constexpr const char* kMeshletStreamCullResetEntryPoint =
    "gpuDrivenStreamAssetCullResetMain";
inline constexpr const char* kMeshletStreamInstanceCullEntryPoint =
    "gpuDrivenStreamAssetInstanceCullMain";
inline constexpr const char* kMeshletStreamHzbEntryPoint =
    "gpuDrivenStreamAssetHzbMain";
inline constexpr const char* kMeshletStreamPageTableInitEntryPoint = "gpuDrivenStreamAssetInitializePageTableMain";
inline constexpr const char* kMeshletStreamUpdateEntryPoint = "gpuDrivenStreamAssetApplyUpdatesMain";
inline constexpr const char* kMeshletStreamTraversalEntryPoint = "gpuDrivenStreamAssetTraversalMain";
inline constexpr const char* kMeshletStreamActiveBuildEntryPoint = "gpuDrivenStreamAssetBuildActiveMain";
inline constexpr const char* kMeshletStreamCooperativeBuildEntryPoint = "streamCooperativeLodMain";
inline constexpr const char* kMeshletStreamDemandEntryPoint = "streamDistributedDemandMain";
inline constexpr const char* kMeshletStreamBlasInputEntryPoint = "gpuDrivenStreamAssetBuildBlasInputMain";
inline constexpr const char* kMeshletStreamTlasInputEntryPoint = "gpuDrivenStreamAssetBuildTlasInputMain";

inline constexpr uint32_t kMeshletStreamDebugPage = 0;
inline constexpr uint32_t kMeshletStreamDebugLod = 1;
inline constexpr uint32_t kMeshletStreamDebugPrimitive = 2;
inline constexpr uint32_t kMeshletStreamDebugInstance = 3;
inline constexpr uint32_t kMeshletStreamDebugMeshlet = 4;
inline constexpr uint32_t kMeshletStreamDebugShaded = 5;
inline constexpr uint32_t kMeshletStreamNoDebugLodOverride = UINT32_MAX;
inline constexpr uint32_t kMeshletStreamInvalidClusterIndex = UINT32_MAX;
inline constexpr uint32_t kMeshletStreamUnloadClusterIndex = UINT32_MAX - 1u;
inline constexpr uint32_t kMeshletStreamDefaultMaxGpuPageRequests = 65536;
inline constexpr uint32_t kMeshletStreamActiveGroupResident = 1u << 0;
inline constexpr uint32_t kMeshletStreamActiveGroupLoadRequest = 1u << 1;
inline constexpr uint32_t kMeshletStreamActiveGroupUnloadRequest = 1u << 2;
inline constexpr uint32_t kMeshletStreamTraversalLoadPhase = 0;
inline constexpr uint32_t kMeshletStreamTraversalUnloadPhase = 1;
inline constexpr uint32_t kMeshletStreamActiveBuildResetPhase = 0;
inline constexpr uint32_t kMeshletStreamActiveBuildBuildPhase = 1;
inline constexpr uint32_t kMeshletStreamActiveBuildFinalizePhase = 2;
inline constexpr uint32_t kMeshletStreamActiveBuildSeedPhase = 3;
inline constexpr uint32_t kMeshletStreamActiveBuildRunPhase = 4;
inline constexpr uint32_t kMeshletStreamActiveBuildFrontierPhase = 5;
inline constexpr uint32_t kMeshletStreamActiveBuildPrefixPhase = 6;
inline constexpr uint32_t kMeshletStreamActiveBuildEmitPhase = 7;
inline constexpr uint32_t kMeshletStreamActiveBuildInitializeLodStatePhase = 8;
inline constexpr uint32_t kMeshletStreamActiveBuildPrefetchPhase = 9;
inline constexpr uint32_t kMeshletStreamActiveBuildClearPhase = 10;
inline constexpr uint32_t kMeshletStreamActiveBuildMaskPhase = 11;
inline constexpr uint32_t kMeshletStreamActiveBuildDemandResetPhase = 12;
inline constexpr uint32_t kMeshletStreamActiveBuildDemandPhase = 13;
inline constexpr uint32_t kMeshletStreamDemandStatsWords = 32;
inline constexpr uint32_t kMeshletStreamBlasInputResetPhase = 0;
inline constexpr uint32_t kMeshletStreamBlasInputCountPhase = 1;
inline constexpr uint32_t kMeshletStreamBlasInputSetupPhase = 2;
inline constexpr uint32_t kMeshletStreamBlasInputInsertPhase = 3;
inline constexpr uint32_t kMeshletStreamBlasInstanceFallback = 1u << 0;
inline constexpr uint32_t kMeshletStreamBlasInstanceDynamic = 1u << 1;
inline constexpr uint32_t kMeshletStreamBlasInstanceOverflow = 1u << 2;
inline constexpr uint32_t kMeshletStreamDefaultMaxActiveGroups = 262144;
inline constexpr uint32_t kMeshletStreamDefaultTraversalWorkers = 1024;
inline constexpr uint32_t kMeshletStreamDefaultTraversalWorkItems = 1048576;
inline constexpr uint32_t kMeshletStreamDefaultMaxBlasBuilds = 65536;
inline constexpr uint32_t kMeshletStreamTriangleChunkSize = 64;
inline constexpr uint32_t kMeshletStreamTriangleChunkCount = 2;
inline constexpr uint32_t kMeshletStreamMaxActiveGroupClusters = 32;
inline constexpr uint32_t kMeshletStreamMaxTraversalWorkers = 65535u * 64u;
inline constexpr uint32_t kMeshletStreamMaxTraversalWorkItems = 16777216;

static_assert(
    static_cast<uint64_t>(kMeshletStreamDefaultMaxActiveGroups) *
        kMeshletStreamMaxActiveGroupClusters <=
    kVisibilityMaxRecordCount);

struct MeshletStreamGpuActiveHeader {
    uint32_t activeGroupCount = 0;
    uint32_t activeGroupCapacity = 0;
    uint32_t maxActiveGroupClusters = 0;
    uint32_t overflowCount = 0;
    uint32_t frameIndex = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

struct MeshletStreamGpuActiveGroup {
    uint32_t pageDeviceOffsetBytes = kInvalidStreamDeviceOffsetBytes;
    uint32_t pageIndex = 0;
    uint32_t clusterCount = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t materialIndex = 0;
    uint32_t clusterSelectionMask = 0;
    uint32_t flags = 0;
    uint32_t instanceIndex = 0;
    uint32_t gpuSceneInstanceIndex = kMeshletStreamInvalidClusterIndex;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
    float world0[4] = {};
    float world1[4] = {};
    float world2[4] = {};
    float world3[4] = {};
};

struct MeshletStreamGpuInstance {
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t visible = 0;
    uint32_t gpuSceneInstanceIndex = kMeshletStreamInvalidClusterIndex;
    float world0[4] = {};
    float world1[4] = {};
    float world2[4] = {};
    float world3[4] = {};
    float boundsCenterRadius[4] = {};
};

struct MeshletStreamGpuPrimitive {
    uint32_t lodLevelOffset = 0;
    uint32_t lodLevelCount = 0;
    uint32_t pageOffset = 0;
    uint32_t pageCount = 0;
    uint32_t fallbackPageOffset = 0;
    uint32_t fallbackPageCount = 0;
    uint32_t groupOffset = 0;
    uint32_t groupCount = 0;
    uint32_t fallbackGroupOffset = 0;
    uint32_t fallbackGroupCount = 0;
    uint32_t materialIndex = 0;
    uint32_t nodeOffset = 0;
    uint32_t nodeCount = 0;
    // Word offset in resident LOD topology and count of preorder BVH nodes.
    uint32_t lodBvhOffset = 0;
    uint32_t lodBvhNodeCount = 0;
    uint32_t lodTileOffset = UINT32_MAX;
};

struct MeshletStreamGpuLodLevel {
    uint32_t pageOffset = 0;
    uint32_t pageCount = 0;
    uint32_t lodLevel = 0;
    uint32_t clusterCount = 0;
    float minBoundingSphereRadius = 0.0f;
    float minMaxQuadricError = 0.0f;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct MeshletStreamGpuGroup {
    uint32_t primitiveIndex = 0;
    uint32_t pageIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t clusterCount = 0;
    float boundsCenterRadius[4] = {};
    float maxQuadricError = 0.0f;
    uint32_t clusterRefinedOffset = 0;
    uint32_t flags = 0;
    uint32_t parentOffset = 0;
    uint32_t parentCount = 0;
    MeshletLodRefinementBounds refinementBounds;
};

struct MeshletStreamGpuNode {
    uint32_t primitiveIndex = 0;
    uint32_t childOffset = 0;
    uint32_t childCount = 0;
    uint32_t groupIndex = kMeshletStreamInvalidClusterIndex;
    float boundsCenterRadius[4] = {};
    float maxQuadricError = 0.0f;
    uint32_t lodLevel = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct MeshletStreamGpuDrawIndirect {
    uint32_t groupCountX = 0;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
};
// Entry 0: ordinary 64-triangle chunks. Entry 1: recursive tessellation tasks.
inline constexpr uint32_t kMeshletStreamDrawIndirectCommandCount = 2;

struct MeshletStreamGpuTraversalHeader {
    uint32_t readCounter = 0;
    uint32_t writeCounter = 0;
    uint32_t taskCounter = 0;
    uint32_t overflowCount = 0;
    uint32_t frameIndex = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

struct MeshletStreamGpuTraversalWorkItem {
    uint32_t instanceIndex = 0;
    uint32_t nodeIndex = 0;
    uint32_t readyFrame = 0;
    uint32_t padding0 = 0;
};

struct MeshletStreamGpuBlasHeader {
    uint32_t clusterReferenceCount = 0;
    uint32_t blasBuildCount = 0;
    uint32_t clusterReferenceCapacity = 0;
    uint32_t overflowCount = 0;
    uint32_t frameIndex = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

struct MeshletStreamGpuInstanceBlas {
    uint32_t clusterReferenceOffset = 0;
    uint32_t clusterReferenceCapacity = 0;
    uint32_t selectedClusterCount = 0;
    uint32_t insertedClusterCount = 0;
    uint32_t blasBuildIndex = kMeshletStreamInvalidClusterIndex;
    uint32_t flags = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct MeshletStreamGpuBlasBuildInfo {
    uint32_t clusterReferencesCount = 0;
    uint32_t clusterReferencesStride = sizeof(uint64_t);
    uint32_t clusterReferencesAddressLow = 0;
    uint32_t clusterReferencesAddressHigh = 0;
};

struct MeshletStreamGpuParams {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    float clearColor[4] = {};
    uint32_t debugColorMode = kMeshletStreamDebugPage;
    uint32_t pageBufferBytes = 0;
    uint32_t drawTaskCount = 0;
    uint32_t frameIndex = 0;
    uint32_t maxGpuPageRequests = 0;
    uint32_t maxGpuPageUnloadRequests = 0;
    uint32_t activeGroupCount = 0;
    uint32_t maxActiveGroupClusters = 0;
    uint32_t sceneInstanceCount = 0;
    uint32_t scenePrimitiveCount = 0;
    uint32_t sceneLodLevelCount = 0;
    uint32_t scenePageCount = 0;
    uint32_t selectedLodLevel = kMeshletStreamNoDebugLodOverride;
    uint32_t enableGpuLodSelection = 1;
    uint32_t enableGpuUnloadRequests = 1;
    uint32_t sceneGroupCount = 0;
    uint32_t maxPrimitiveGroupCount = 0;
    uint32_t sceneNodeCount = 0;
    uint32_t traversalWorkerCount = 0;
    uint32_t traversalWorkCapacity = 0;
    uint32_t blasClusterReferenceAddressLow = 0;
    uint32_t blasClusterReferenceAddressHigh = 0;
    uint32_t blasClusterReferenceCapacity = 0;
    uint32_t blasBuildCapacity = 0;
    float previousEye[4] = {};
    float previousCenter[4] = {};
    float previousUpProjection[4] = {};
    float previousViewport[4] = {};
    float previousClipOrtho[4] = {};
    float lodPixelError = 1.5f;
    uint32_t lodTopologyBuffer = UINT32_MAX;
    uint32_t lodStateBuffer = UINT32_MAX;
    uint32_t lodInstanceOffsetsOffset = 0;
    float renderEye[4] = {};
    float renderCenter[4] = {};
    float renderUpProjection[4] = {};
    float renderViewport[4] = {};
    float renderClipOrtho[4] = {};
    float prefetchParams[4] = {}; // Frustum expansion, LOD error scale, enabled, reserved.
    uint32_t demandBuffer = UINT32_MAX;
    uint32_t demandTaskOffset = 0; // Tile root indices in LOD topology, grouped by instance.
    uint32_t demandTaskCount = 0;
    uint32_t splitFrontier = 0;
    uint32_t demandInstanceOffsetsOffset = 0; // Group/tile bit bases and task start/count per instance.
    uint32_t demandStatsBuffer = UINT32_MAX;
    uint32_t demandPadding[2] = {};
};

struct MeshletStreamGpuRasterBindings {
    uint32_t visibleClusterBuffer = 0;
    uint32_t instanceVisibilityBuffer = 0;
    uint32_t hzbBuffer0 = 0;
    uint32_t hzbBuffer1 = 0;
    uint32_t depthImage = 0;
    uint32_t visibilityImage = 0;
    uint32_t deferredColorBuffer = 0;
    uint32_t visibleInstanceIdsBuffer = 0;
    // Stream records keep a local storage index while visibility IDs address
    // the common resident + stream record namespace.
    uint32_t visibleRecordBase = 0;
    uint32_t visibleRecordCapacity = 0;
    uint32_t hzbMipCount = 0;
    uint32_t hzbValid = 0;
    uint32_t cullingFlags = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t visibleInstanceCounterBuffer = 0;
    uint32_t gpuSceneInstanceBuffer = UINT32_MAX;
    uint32_t tessellationBuffer = UINT32_MAX;
    float displacementBound = 0.0f;
    uint32_t classificationFlags = 0; // bit 0: disable metadata fast classification
    uint32_t materialBuffer = UINT32_MAX;
    uint32_t materialTextureRemapBuffer = UINT32_MAX;
    uint32_t materialTextureCount = 0;
    uint32_t materialPadding = 0;
};

// Non-owning resources required by a unified deferred consumer. The stream
// runtime retains ownership; a consumer registers these buffers in its own
// bindless heap so one deferred dispatch can decode both resident and streamed
// visibility IDs.
struct MeshletStreamDeferredGpuResourcesView {
    Buffer* instanceBuffer = nullptr;
    Buffer* pageBuffer = nullptr;
    Buffer* activeGroupBuffer = nullptr;
    Buffer* pageTableBuffer = nullptr;
    Buffer* activeHeaderBuffer = nullptr;
    Buffer* paramsBuffer = nullptr;
    Buffer* visibleClusterBuffer = nullptr;
    uint32_t visibleRecordCapacity = 0;
    RayTracingAccelerationStructure* accelerationStructure = nullptr;

    bool valid() const
    {
        return pageBuffer != nullptr &&
            activeGroupBuffer != nullptr &&
            pageTableBuffer != nullptr &&
            activeHeaderBuffer != nullptr &&
            paramsBuffer != nullptr &&
            visibleClusterBuffer != nullptr &&
            visibleRecordCapacity != 0;
    }
};

struct MeshletStreamUserPush {
    uint32_t pageBuffer = 0;
    uint32_t activeGroupBuffer = 0;
    uint32_t pageTableBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t requestBuffer = 0;
    uint32_t residentPageBuffer = 0;
    uint32_t updateBuffer = 0;
    uint32_t activeHeaderBuffer = 0;
    uint32_t instanceBuffer = 0;
    uint32_t primitiveBuffer = 0;
    uint32_t lodLevelBuffer = 0;
    uint32_t groupBuffer = 0;
    uint32_t nodeBuffer = 0;
    uint32_t drawIndirectBuffer = 0;
    uint32_t traversalHeaderBuffer = 0;
    uint32_t traversalWorkBuffer = 0;
    uint32_t clasAddressBuffer = 0;
    uint32_t clasPageTableBuffer = 0;
    uint32_t blasHeaderBuffer = 0;
    uint32_t instanceBlasBuffer = 0;
    uint32_t blasBuildInfoBuffer = 0;
    uint32_t blasClusterReferenceBuffer = 0;
    uint32_t fallbackBlasAddressBuffer = 0;
    uint32_t dynamicBlasAddressBuffer = 0;
    uint32_t tlasInstanceBuffer = 0;
    uint32_t traversalPhase = kMeshletStreamTraversalLoadPhase;
    uint32_t activeBuildPhase = kMeshletStreamActiveBuildBuildPhase;
    uint32_t rasterBindingsBuffer = UINT32_MAX;
    uint32_t hybridQueueBuffer = UINT32_MAX;
    uint32_t hybridClusterBuffer = UINT32_MAX;
    uint32_t clasPublicationRevision = 0;
    float tessellationEdgePixels = 8.0f;
    uint32_t tessellationMaxFactor = 4;
    uint32_t tessellationMaxSplitDepth = 2;
};

static_assert(sizeof(MeshletStreamGpuActiveHeader) == 32);
static_assert(sizeof(MeshletStreamGpuActiveGroup) == 112);
static_assert(sizeof(MeshletStreamGpuInstance) == 96);
static_assert(sizeof(MeshletStreamGpuPrimitive) == 64);
static_assert(sizeof(MeshletStreamGpuLodLevel) == 32);
static_assert(sizeof(MeshletStreamGpuGroup) == 76);
static_assert(sizeof(MeshletStreamGpuNode) == 48);
static_assert(sizeof(MeshletStreamGpuDrawIndirect) == 12);
static_assert(sizeof(MeshletStreamGpuTraversalHeader) == 32);
static_assert(sizeof(MeshletStreamGpuTraversalWorkItem) == 16);
static_assert(sizeof(MeshletStreamGpuBlasHeader) == 32);
static_assert(sizeof(MeshletStreamGpuInstanceBlas) == 32);
static_assert(sizeof(MeshletStreamGpuBlasBuildInfo) == 16);
static_assert(sizeof(StreamPageTableEntry) == 8);
static_assert(sizeof(MeshletStreamGpuParams) == 416);
// VisibilityStreamDecode.slang reads the pool capacity from the immutable frame params.
static_assert(offsetof(MeshletStreamGpuParams, pageBufferBytes) == 100);
static_assert(sizeof(MeshletStreamGpuRasterBindings) == 96);
static_assert(sizeof(MeshletStreamUserPush) == 136);

struct MeshletStreamRuntimeDesc {
    std::filesystem::path sourcePath;
    std::filesystem::path streamAssetPath;
    bool autoBuildStreamAsset = false;
    uint64_t maxResidentBytes = 0;
    uint32_t maxResidentPages = 4096;
    uint32_t maxLockedFallbackPages = 1024;
    uint32_t maxPageUploadsPerFrame = 64;
    uint64_t maxUploadBytesPerFrame = 8ull * 1024ull * 1024ull; // Zero disables the byte limit.
    uint32_t maxGpuPageRequests = kMeshletStreamDefaultMaxGpuPageRequests;
    uint32_t maxGpuPageUnloadRequests = kMeshletStreamDefaultMaxGpuPageRequests;
    uint32_t maxActiveGroups = kMeshletStreamDefaultMaxActiveGroups;
    uint32_t maxTraversalWorkers = kMeshletStreamDefaultTraversalWorkers;
    uint32_t maxTraversalWorkItems = kMeshletStreamDefaultTraversalWorkItems;
    uint32_t pageLoadConcurrency = 2;
    uint32_t maxPageLoadsInFlight = 128;
    uint32_t queuedFrameCount = 3;
    bool enableClusterRtx = false;
    bool enableClas = false; // Build resident CLAS independently of per-frame BLAS/TLAS.
    bool compactClas = false;
    uint32_t coldPageRetentionFrames = 0; // Zero preserves budget-only eviction.
    uint64_t maxClasBytes = 512ull * 1024ull * 1024ull;
    uint32_t maxClasBuildClusters = 0;
    uint32_t maxBlasClusterReferences = 0;
    uint64_t maxBlasBytes = 512ull * 1024ull * 1024ull;
    uint32_t maxBlasBuilds = kMeshletStreamDefaultMaxBlasBuilds;
    uint64_t maxFallbackBlasBytes = 512ull * 1024ull * 1024ull;
    bool screenSpacePagePriority = true;
    bool viewDrivenPageDemand = true;
    bool distributedPageDemand = true; // Ordered safe cut consumes the independent demand bitmap.
    uint32_t distributedDemandMinGroups = 65536; // Zero forces distribution; otherwise use completed request feedback.
    bool measurePageLatency = true;
    bool lowLatencyRequests = true;
    bool completionDrivenUploads = true;
    bool enableGpuDecompression = false; // Opt-in GPU-ready assets; unsupported devices decode on CPU.
    uint64_t gpuDecompressionMinBatchBytes = 1024 * 1024;
    bool prefetchPages = true;
    uint32_t rasterMaterialTextureCapacity = 0;
    bool compactShadingAttributes = false;

    bool operator==(const MeshletStreamRuntimeDesc&) const = default;
};

struct MeshletStreamCameraDesc {
    float3 eye{0.0f, 0.0f, 0.0f};
    float3 center{0.0f, 0.0f, -1.0f};
    float3 up{0.0f, 1.0f, 0.0f};
    float fovDegrees = 60.0f;
    float znear = 0.1f;
    float zfar = 1000.0f;
    bool orthographic = false;
    bool reversedZ = true;
    float orthoHeight = 10.0f;
};

struct MeshletStreamFrameDesc {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t selectedLodLevel = kMeshletStreamNoDebugLodOverride;
    bool enableGpuLodSelection = true;
    float lodPixelError = 1.5f;
    float lodBias = 0.0f;
    uint32_t debugColorMode = kMeshletStreamDebugPage;
    MeshletStreamCameraDesc camera;
    MeshletStreamCameraDesc renderCamera;
    bool useSeparateRenderCamera = false;
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    // Diagnostic raster A/B: retain the last published GPU cut and page mappings.
    bool freezeRasterSnapshot = false;
};

class MeshletStreamRuntime {
public:
    MeshletStreamRuntime();
    ~MeshletStreamRuntime();

    MeshletStreamRuntime(const MeshletStreamRuntime&) = delete;
    MeshletStreamRuntime& operator=(const MeshletStreamRuntime&) = delete;

    MeshletStreamRuntime(MeshletStreamRuntime&&) noexcept = delete;
    MeshletStreamRuntime& operator=(MeshletStreamRuntime&&) noexcept = delete;

    // The optional caller-owned cache is used only during initialization.
    // Its owner handles persistence; no cache pointer survives this call.
    Result initialize(Device& device, const MeshletStreamRuntimeDesc& desc, std::string& log,
        PipelineCache* pipelineCache = nullptr);
    Result syncRuntimeScene(const scene::Scene& scene, std::string& log);
    Result syncRuntimeScene(
        const scene::Scene& scene,
        std::span<const uint32_t> runtimeRenderNodeIndices,
        std::string& log);
    Result syncGPUSceneInstanceMapping(std::span<const uint32_t> mapping);
    void reset();

    bool ready() const;
    // Complete fallback coverage, including RT fallback resources when used.
    // Resource initialization alone does not make a scene presentable.
    StreamSceneReadiness sceneReadiness() const;
    // Cached for the root resource lifetime; loading progress refreshes per frame.
    bool sceneReady() const { return sceneReadiness().ready; }
    bool tlasReady() const { return tlasBuilt_; }
    RayTracingAccelerationStructure* accelerationStructure() const;

    Result cmdBeginFrame(CommandBuffer& commandBuffer, Streamer& streamer, const MeshletStreamFrameDesc& frame,
        const std::function<void()>& flushUploads = {});
    // CPU-only, non-blocking maintenance for the next recorded frame. A caller
    // may invoke this before pacing; cmdBeginFrame remains the fallback owner.
    void prepareMaintenance(CpuProfileRecorder* profiler = nullptr, bool allowLegacyReadback = false);
    const CpuProfileRecorder& beginFrameCpuProfile() const { return beginFrameCpuProfile_; }
    using TraversalCheckpoint = std::function<void(std::string_view)>;
    Result cmdPreTraversal(CommandBuffer& commandBuffer, const MeshletStreamFrameDesc& frame,
        const TraversalCheckpoint& checkpoint = {});
    Result cmdPostTraversal(CommandBuffer& commandBuffer);
    Result cmdEndFrame(CommandBuffer& commandBuffer);

    ResourceRegistry* resourceRegistry() const { return registry_.get(); }
    BindlessHeap* bindlessHeap() const { return registry_ ? registry_->heap() : nullptr; }
    MeshletStreamUserPush userPush() const;
    Result updateRasterBindings(const MeshletStreamGpuRasterBindings& bindings);
    Result cmdPrepareVisibility(CommandBuffer& commandBuffer);
    Result cmdPrepareDeferred(CommandBuffer& commandBuffer);
    MeshletStreamDeferredGpuResourcesView deferredGpuResources() const;
    uint32_t frameIndex() const { return frameIndex_; }
    uint32_t visibleClusterCapacity() const;
    uint32_t drawTaskCount() const;
    void cmdDrawMeshTasks(CommandBuffer& commandBuffer, bool tessellation = false) const;
    const scene::Bounds& bounds() const { return drawBounds_; }
    const scene::MeshletStreamAsset& asset() const { return asset_; }
    const MeshletStreamResidencyManager& residency() const { return residency_; }
    void setDebugReadbackEnabled(bool enabled) { debugReadbackEnabled_ = enabled; }
    void appendDebugBindings(std::vector<DebugResourceBinding>& bindings, const std::string& prefix) const;
    nlohmann::json debugSnapshot(bool includePages = true) const;
    SceneStreamingProfile profilingStats() const;
    MeshletStreamClasPool* clasPool() const { return clasPool_.get(); }

private:
    struct SceneReadinessCache {
        StreamSceneReadiness value{.ready = false};
        bool valid = false;
        bool rootsInvalidated = false;
        uint64_t scans = 0;
    };
    // Callbacks retain only this generation's state, never the runtime itself.
    std::shared_ptr<SceneReadinessCache> sceneReadinessCache_ = std::make_shared<SceneReadinessCache>();
    CpuProfileRecorder beginFrameCpuProfile_;
    bool maintenancePrepared_ = false;
    bool rasterSnapshotFrozen_ = false;
    std::shared_ptr<bool> blasCacheInitialized_ = std::make_shared<bool>(false);
    struct FrameUploads {
        std::unique_ptr<Buffer> params, raster, clear;
        ResourceLease paramsHandle, rasterHandle;
        GpuCompletionPoint completion;
    };
    std::vector<FrameUploads> frameUploads_;
    uint32_t currentUploadSlot_ = 0;
    struct FallbackBlasPrimitive {
        uint32_t primitiveIndex = 0;
        uint32_t referenceCount = 0;
        uint64_t referenceOffset = 0;
        uint64_t storageOffset = 0;
        bool built = false;
        std::shared_ptr<SubmissionTransaction> buildTransaction;
        bool recorded() const { return built && buildTransaction && !buildTransaction->cancelled(); }
        bool submitted() const { return recorded() && buildTransaction->resolved(); }
    };

    struct ResidentPageFrame {
        std::unique_ptr<Buffer> buffer;
        ResourceLease handle;
    };

    class UpdatePass;
    class TraversalPass;
    class ActiveBuildPass;
    class BlasInputPass;
    class TlasInputPass;

    uint32_t computeMaxActiveGroups(uint32_t capacity) const;
    uint32_t computeMaxPrimitiveGroups() const;
    Result initializeSceneMetadataBuffers(Device& device, std::string& log);

    Result initializePageTableIfNeeded(CommandBuffer& commandBuffer);
    Result applyPageTablePatches(CommandBuffer& commandBuffer);
    Result clearRequestBuffer(CommandBuffer& commandBuffer);
    Result dispatchTraversal(CommandBuffer& commandBuffer, uint32_t threadCount, uint32_t traversalPhase);
    Result buildActiveTable(CommandBuffer& commandBuffer, const TraversalCheckpoint& checkpoint);
    Result buildBlasInputs(CommandBuffer& commandBuffer, const TraversalCheckpoint& checkpoint);
    Result cmdBuildBlas(CommandBuffer& commandBuffer);
    Result cmdBuildFallbackBlas(CommandBuffer& commandBuffer);
    Result buildTlasInstances(CommandBuffer& commandBuffer);
    Result cmdBuildTlas(CommandBuffer& commandBuffer);
    Result copyRequestBufferForReadback(CommandBuffer& commandBuffer);
    Result updateParamsBuffer(const MeshletStreamFrameDesc& frame);
    Result transitionPageBufferForTraversal(CommandBuffer& commandBuffer);
    void consumeGpuRequestReadback(CpuProfileRecorder* profiler, bool allowLegacyReadback);

    scene::MeshletStreamAsset asset_;
    MeshletStreamResidencyManager residency_;
    scene::Bounds drawBounds_;
    uint64_t sceneTransformRevision_ = 0;
    uint64_t sceneVisibilityRevision_ = 0;
    uint64_t sceneResourceIdentity_ = 0;
    std::vector<uint32_t> runtimeRenderNodeIndices_;
    std::vector<uint32_t> gpuSceneInstanceMapping_;
    std::unique_ptr<Buffer> pageBuffer_;
    std::unique_ptr<Buffer> activeGroupBuffer_;
    std::unique_ptr<Buffer> activeHeaderBuffer_;
    std::unique_ptr<Buffer> pageTableBuffer_;
    std::unique_ptr<Buffer> requestBuffer_;
    std::unique_ptr<Buffer> requestReadbackBuffer_;
    struct RequestReadback {
        std::unique_ptr<Buffer> buffer;
        GpuCompletionPoint completion;
        std::shared_ptr<SubmissionTransaction> submission;
        uint32_t frame = 0;
    };
    std::vector<RequestReadback> requestReadbacks_;
    uint32_t consumedRequestFrame_ = 0;
    std::unique_ptr<Buffer> requestClearBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<Buffer> visibleClusterBuffer_;
    std::unique_ptr<Buffer> rasterBindingsBuffer_;
    std::vector<ResidentPageFrame> residentPageFrames_;
    std::unique_ptr<Buffer> instanceBuffer_;
    std::unique_ptr<Buffer> primitiveBuffer_;
    std::unique_ptr<Buffer> lodLevelBuffer_;
    std::unique_ptr<Buffer> groupBuffer_;
    std::unique_ptr<Buffer> lodTopologyBuffer_;
    std::unique_ptr<Buffer> lodStateBuffer_;
    std::unique_ptr<Buffer> demandBuffer_;
    std::unique_ptr<Buffer> nodeBuffer_;
    std::unique_ptr<Buffer> drawIndirectBuffer_;
    std::unique_ptr<Buffer> traversalHeaderBuffer_;
    std::unique_ptr<Buffer> traversalWorkBuffer_;
    std::unique_ptr<Buffer> blasHeaderBuffer_;
    std::unique_ptr<Buffer> instanceBlasBuffer_;
    std::unique_ptr<Buffer> blasBuildInfoBuffer_;
    std::unique_ptr<Buffer> blasClusterReferenceBuffer_;
    std::unique_ptr<Buffer> blasStorageBuffer_;
    std::unique_ptr<Buffer> blasScratchBuffer_;
    std::unique_ptr<Buffer> blasAddressBuffer_;
    std::unique_ptr<Buffer> blasSizeBuffer_;
    std::unique_ptr<Buffer> fallbackBlasStorageBuffer_;
    std::unique_ptr<Buffer> fallbackBlasScratchBuffer_;
    std::unique_ptr<Buffer> fallbackBlasReferenceBuffer_;
    std::unique_ptr<Buffer> fallbackBlasBuildInfoBuffer_;
    std::unique_ptr<Buffer> fallbackBlasDestinationBuffer_;
    std::unique_ptr<Buffer> fallbackBlasAddressBuffer_;
    std::unique_ptr<Buffer> tlasInstanceBuffer_;
    std::unique_ptr<Buffer> tlasScratchBuffer_;
    std::unique_ptr<RayTracingAccelerationStructure> tlas_;
    std::shared_ptr<ResourceRegistry> registry_;
    std::unique_ptr<UpdatePass> updatePass_;
    std::unique_ptr<TraversalPass> traversalPass_;
    std::unique_ptr<ActiveBuildPass> activeBuildPass_;
    std::unique_ptr<BlasInputPass> blasInputPass_;
    std::unique_ptr<TlasInputPass> tlasInputPass_;
    std::unique_ptr<MeshletStreamClasPool> clasPool_;
    std::unordered_map<uint32_t, MeshletStreamClasPagePlan> pendingClasPlans_;
    std::deque<uint32_t> pendingClasPages_;
    std::unordered_set<uint32_t> queuedClasPages_;
    uint32_t maxClasBuildClusters_ = 0;
    uint32_t coldPageRetentionFrames_ = 0;
    bool clusterRtxEnabled_ = false;
    MeshletStreamGpuBlasHeader recentBlasHeader_;
    ResourceLease pageHandle_;
    ResourceLease activeGroupHandle_;
    ResourceLease activeHeaderHandle_;
    ResourceLease pageTableHandle_;
    ResourceLease paramsHandle_;
    ResourceLease visibleClusterHandle_;
    ResourceLease rasterBindingsHandle_;
    ResourceLease requestHandle_;
    ResourceLease instanceHandle_;
    ResourceLease primitiveHandle_;
    ResourceLease lodLevelHandle_;
    ResourceLease groupHandle_;
    ResourceLease lodTopologyHandle_;
    ResourceLease lodStateHandle_;
    ResourceLease demandHandle_;
    uint32_t demandTaskOffset_ = 0;
    uint32_t demandTaskCount_ = 0;
    uint32_t demandInstanceOffsetsOffset_ = 0;
    ResourceState demandBufferState_ = ResourceState::Undefined;
    uint32_t lodInstanceOffsetsOffset_ = 0;
    ResourceState lodStateBufferState_ = ResourceState::Undefined;
    ResourceLease nodeHandle_;
    ResourceLease drawIndirectHandle_;
    ResourceLease traversalHeaderHandle_;
    ResourceLease traversalWorkHandle_;
    ResourceLease clasAddressHandle_;
    ResourceLease clasPageTableHandle_;
    ResourceLease blasHeaderHandle_;
    ResourceLease instanceBlasHandle_;
    ResourceLease blasBuildInfoHandle_;
    ResourceLease blasClusterReferenceHandle_;
    ResourceLease fallbackBlasAddressHandle_;
    ResourceLease dynamicBlasAddressHandle_;
    ResourceLease tlasInstanceHandle_;
    ResourceState pageBufferState_ = ResourceState::Undefined;
    ResourceState activeGroupBufferState_ = ResourceState::Undefined;
    ResourceState activeHeaderBufferState_ = ResourceState::Undefined;
    ResourceState pageTableState_ = ResourceState::Undefined;
    ResourceState requestBufferState_ = ResourceState::Undefined;
    ResourceState visibleClusterBufferState_ = ResourceState::Undefined;
    ResourceState drawIndirectBufferState_ = ResourceState::Undefined;
    ResourceState traversalHeaderBufferState_ = ResourceState::Undefined;
    ResourceState traversalWorkBufferState_ = ResourceState::Undefined;
    ResourceState blasHeaderBufferState_ = ResourceState::Undefined;
    ResourceState instanceBlasBufferState_ = ResourceState::Undefined;
    ResourceState blasBuildInfoBufferState_ = ResourceState::Undefined;
    ResourceState blasClusterReferenceBufferState_ = ResourceState::Undefined;
    ResourceState tlasInstanceBufferState_ = ResourceState::Undefined;
    std::shared_ptr<bool> pageTableInitialized_ = std::make_shared<bool>(false);
    uint32_t currentFrameOrderedUploadCount_ = 0;
    bool requestReadbackValid_ = false;
    uint32_t frameIndex_ = 0;
    bool debugReadbackEnabled_ = false;
    uint64_t debugGeneration_ = 0;
    uint32_t debugRequestSourceFrame_ = 0;
    bool debugRequestSourceKnown_ = false;
    uint32_t maxResidentPages_ = 0;
    uint32_t maxPageUploadsPerFrame_ = 0;
    uint64_t maxUploadBytesPerFrame_ = 0;
    uint32_t maxGpuPageRequests_ = 0;
    bool screenSpacePagePriority_ = false;
    bool viewDrivenPageDemand_ = false;
    bool distributedPageDemand_ = false;
    bool currentFrameDistributedDemand_ = true;
    uint32_t distributedDemandMinGroups_ = 65536;
    uint32_t recentDemandGroupTests_ = UINT32_MAX;
    bool prefetchPages_ = false;
    bool currentFramePrefetch_ = false;
    uint32_t recentGpuRequestCount_ = 0;
    uint32_t maxGpuPageUnloadRequests_ = 0;
    uint32_t maxUpdatePatches_ = 0;
    uint32_t residentPageCapacity_ = 0;
    uint32_t currentResidentPageCount_ = 0;
    uint64_t maxResidentBytes_ = 0;
    std::vector<uint32_t> lockedFallbackPages_;
    uint32_t maxActiveGroups_ = 0;
    uint32_t maxActiveGroupClusters_ = 0;
    uint32_t maxPrimitiveGroupCount_ = 0;
    uint32_t traversalWorkerCount_ = 0;
    uint32_t traversalWorkCapacity_ = 0;
    uint32_t blasClusterReferenceCapacity_ = 0;
    uint32_t blasBuildCapacity_ = 0;
    uint32_t maxBlasClustersPerBuild_ = 0;
    uint64_t blasClusterReferenceAddress_ = 0;
    bool tlasBuilt_ = false;
    std::vector<FallbackBlasPrimitive> fallbackBlasPrimitives_;
    uint32_t currentFrameUploadCount_ = 0;
    MeshletStreamGpuParams previousFrameParams_;
    bool previousFrameParamsValid_ = false;
};

} // namespace metallic::render
