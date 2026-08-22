#pragma once

#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/MeshletStreamResidency.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {

class MeshletStreamClasPool;

inline constexpr const char* kMeshletStreamShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
inline constexpr const char* kMeshletStreamShaderModuleName = "GPUDrivenStreamAsset";
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
    uint32_t padding2 = 0;
    uint32_t padding3 = 0;
    uint32_t padding4 = 0;
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
};

// Non-owning resources required by a unified deferred consumer. The stream
// runtime retains ownership; a consumer registers these buffers in its own
// bindless heap so one deferred dispatch can decode both resident and streamed
// visibility IDs.
struct MeshletStreamDeferredGpuResourcesView {
    Buffer* pageBuffer = nullptr;
    Buffer* activeGroupBuffer = nullptr;
    Buffer* pageTableBuffer = nullptr;
    Buffer* activeHeaderBuffer = nullptr;
    Buffer* paramsBuffer = nullptr;
    Buffer* visibleClusterBuffer = nullptr;
    uint32_t visibleRecordCapacity = 0;

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
    uint32_t rasterBindingsBuffer = 0;
};

static_assert(sizeof(MeshletStreamGpuActiveHeader) == 32);
static_assert(sizeof(MeshletStreamGpuActiveGroup) == 112);
static_assert(sizeof(MeshletStreamGpuInstance) == 96);
static_assert(sizeof(MeshletStreamGpuPrimitive) == 64);
static_assert(sizeof(MeshletStreamGpuLodLevel) == 32);
static_assert(sizeof(MeshletStreamGpuGroup) == 36);
static_assert(sizeof(MeshletStreamGpuNode) == 48);
static_assert(sizeof(MeshletStreamGpuDrawIndirect) == 12);
static_assert(sizeof(MeshletStreamGpuTraversalHeader) == 32);
static_assert(sizeof(MeshletStreamGpuTraversalWorkItem) == 16);
static_assert(sizeof(MeshletStreamGpuBlasHeader) == 32);
static_assert(sizeof(MeshletStreamGpuInstanceBlas) == 32);
static_assert(sizeof(MeshletStreamGpuBlasBuildInfo) == 16);
static_assert(sizeof(StreamPageTableEntry) == 8);
static_assert(sizeof(MeshletStreamGpuParams) == 272);
static_assert(sizeof(MeshletStreamGpuRasterBindings) == 64);
static_assert(sizeof(MeshletStreamUserPush) == 112);

struct MeshletStreamRuntimeDesc {
    std::filesystem::path sourcePath;
    std::filesystem::path streamAssetPath;
    bool autoBuildStreamAsset = false;
    uint64_t maxResidentBytes = 0;
    uint32_t maxResidentPages = 4096;
    uint32_t maxLockedFallbackPages = 1024;
    uint32_t maxPageUploadsPerFrame = 64;
    uint32_t maxGpuPageRequests = kMeshletStreamDefaultMaxGpuPageRequests;
    uint32_t maxGpuPageUnloadRequests = kMeshletStreamDefaultMaxGpuPageRequests;
    uint32_t maxActiveGroups = kMeshletStreamDefaultMaxActiveGroups;
    uint32_t maxTraversalWorkers = kMeshletStreamDefaultTraversalWorkers;
    uint32_t maxTraversalWorkItems = kMeshletStreamDefaultTraversalWorkItems;
    uint32_t pageLoadConcurrency = 2;
    uint32_t maxPageLoadsInFlight = 128;
    uint32_t queuedFrameCount = 3;
    bool enableClusterRtx = false;
    uint64_t maxClasBytes = 512ull * 1024ull * 1024ull;
    uint32_t maxClasBuildClusters = 0;
    uint32_t maxBlasClusterReferences = 0;
    uint64_t maxBlasBytes = 512ull * 1024ull * 1024ull;
    uint32_t maxBlasBuilds = kMeshletStreamDefaultMaxBlasBuilds;
    uint64_t maxFallbackBlasBytes = 512ull * 1024ull * 1024ull;
};

struct MeshletStreamCameraDesc {
    float3 eye{0.0f, 0.0f, 0.0f};
    float3 center{0.0f, 0.0f, -1.0f};
    float3 up{0.0f, 1.0f, 0.0f};
    float fovDegrees = 60.0f;
    float znear = 0.1f;
    float zfar = 1000.0f;
};

struct MeshletStreamFrameDesc {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t selectedLodLevel = kMeshletStreamNoDebugLodOverride;
    bool enableGpuLodSelection = true;
    uint32_t debugColorMode = kMeshletStreamDebugPage;
    MeshletStreamCameraDesc camera;
};

class MeshletStreamRuntime {
public:
    MeshletStreamRuntime();
    ~MeshletStreamRuntime();

    MeshletStreamRuntime(const MeshletStreamRuntime&) = delete;
    MeshletStreamRuntime& operator=(const MeshletStreamRuntime&) = delete;

    MeshletStreamRuntime(MeshletStreamRuntime&&) noexcept = delete;
    MeshletStreamRuntime& operator=(MeshletStreamRuntime&&) noexcept = delete;

    Result initialize(Device& device, const MeshletStreamRuntimeDesc& desc, std::string& log);
    Result syncRuntimeScene(const scene::Scene& scene, std::string& log);
    Result syncRuntimeScene(
        const scene::Scene& scene,
        std::span<const uint32_t> runtimeRenderNodeIndices,
        std::string& log);
    Result syncGPUSceneInstanceMapping(std::span<const uint32_t> mapping);
    void reset();

    bool ready() const;
    bool tlasReady() const { return tlasBuilt_; }
    RayTracingAccelerationStructure* accelerationStructure() const;

    Result cmdBeginFrame(CommandBuffer& commandBuffer, Streamer& streamer, const MeshletStreamFrameDesc& frame);
    Result cmdPreTraversal(CommandBuffer& commandBuffer, const MeshletStreamFrameDesc& frame);
    Result cmdPostTraversal(CommandBuffer& commandBuffer);
    Result cmdEndFrame(CommandBuffer& commandBuffer);

    BindlessHeap* bindlessHeap() const { return bindlessHeap_.get(); }
    MeshletStreamUserPush userPush() const;
    Result updateRasterBindings(const MeshletStreamGpuRasterBindings& bindings);
    Result cmdPrepareVisibility(CommandBuffer& commandBuffer);
    Result cmdPrepareDeferred(CommandBuffer& commandBuffer);
    MeshletStreamDeferredGpuResourcesView deferredGpuResources() const;
    uint32_t frameIndex() const { return frameIndex_; }
    uint32_t visibleClusterCapacity() const;
    uint32_t drawTaskCount() const;
    void cmdDrawMeshTasks(CommandBuffer& commandBuffer) const;
    const scene::Bounds& bounds() const { return drawBounds_; }
    const scene::MeshletStreamAsset& asset() const { return asset_; }
    const MeshletStreamResidencyManager& residency() const { return residency_; }
    MeshletStreamClasPool* clasPool() const { return clasPool_.get(); }

private:
    struct FallbackBlasPrimitive {
        uint32_t primitiveIndex = 0;
        uint32_t referenceCount = 0;
        uint64_t referenceOffset = 0;
        uint64_t storageOffset = 0;
        bool built = false;
    };

    struct ResidentPageFrame {
        std::unique_ptr<Buffer> buffer;
        BindlessHandle handle;
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
    Result buildActiveTable(CommandBuffer& commandBuffer);
    Result buildBlasInputs(CommandBuffer& commandBuffer);
    Result cmdBuildBlas(CommandBuffer& commandBuffer);
    Result cmdBuildFallbackBlas(CommandBuffer& commandBuffer);
    Result buildTlasInstances(CommandBuffer& commandBuffer);
    Result cmdBuildTlas(CommandBuffer& commandBuffer);
    Result copyRequestBufferForReadback(CommandBuffer& commandBuffer);
    Result updateParamsBuffer(const MeshletStreamFrameDesc& frame);
    Result transitionPageBufferForTraversal(CommandBuffer& commandBuffer);
    void consumeGpuRequestReadback();

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
    std::unique_ptr<Buffer> requestClearBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<Buffer> visibleClusterBuffer_;
    std::unique_ptr<Buffer> rasterBindingsBuffer_;
    std::vector<ResidentPageFrame> residentPageFrames_;
    std::unique_ptr<Buffer> instanceBuffer_;
    std::unique_ptr<Buffer> primitiveBuffer_;
    std::unique_ptr<Buffer> lodLevelBuffer_;
    std::unique_ptr<Buffer> groupBuffer_;
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
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    std::unique_ptr<UpdatePass> updatePass_;
    std::unique_ptr<TraversalPass> traversalPass_;
    std::unique_ptr<ActiveBuildPass> activeBuildPass_;
    std::unique_ptr<BlasInputPass> blasInputPass_;
    std::unique_ptr<TlasInputPass> tlasInputPass_;
    std::unique_ptr<MeshletStreamClasPool> clasPool_;
    BindlessHandle pageHandle_;
    BindlessHandle activeGroupHandle_;
    BindlessHandle activeHeaderHandle_;
    BindlessHandle pageTableHandle_;
    BindlessHandle paramsHandle_;
    BindlessHandle visibleClusterHandle_;
    BindlessHandle rasterBindingsHandle_;
    BindlessHandle requestHandle_;
    BindlessHandle instanceHandle_;
    BindlessHandle primitiveHandle_;
    BindlessHandle lodLevelHandle_;
    BindlessHandle groupHandle_;
    BindlessHandle nodeHandle_;
    BindlessHandle drawIndirectHandle_;
    BindlessHandle traversalHeaderHandle_;
    BindlessHandle traversalWorkHandle_;
    BindlessHandle clasAddressHandle_;
    BindlessHandle clasPageTableHandle_;
    BindlessHandle blasHeaderHandle_;
    BindlessHandle instanceBlasHandle_;
    BindlessHandle blasBuildInfoHandle_;
    BindlessHandle blasClusterReferenceHandle_;
    BindlessHandle fallbackBlasAddressHandle_;
    BindlessHandle dynamicBlasAddressHandle_;
    BindlessHandle tlasInstanceHandle_;
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
    bool pageTableInitialized_ = false;
    bool requestReadbackValid_ = false;
    uint32_t frameIndex_ = 0;
    uint32_t maxResidentPages_ = 0;
    uint32_t maxPageUploadsPerFrame_ = 0;
    uint32_t maxGpuPageRequests_ = 0;
    uint32_t maxGpuPageUnloadRequests_ = 0;
    uint32_t maxUpdatePatches_ = 0;
    uint32_t residentPageCapacity_ = 0;
    uint32_t currentResidentPageCount_ = 0;
    uint64_t maxResidentBytes_ = 0;
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
