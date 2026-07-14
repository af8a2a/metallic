#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/MeshletStreamResidency.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {

inline constexpr const char* kMeshletStreamShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
inline constexpr const char* kMeshletStreamShaderModuleName = "gpu_driven_streamasset";
inline constexpr const char* kMeshletStreamMeshEntryPoint = "gpuDrivenStreamAssetMeshMain";
inline constexpr const char* kMeshletStreamFragmentEntryPoint = "gpuDrivenStreamAssetFragmentMain";
inline constexpr const char* kMeshletStreamUpdateEntryPoint = "gpuDrivenStreamAssetApplyUpdatesMain";
inline constexpr const char* kMeshletStreamTraversalEntryPoint = "gpuDrivenStreamAssetTraversalMain";
inline constexpr const char* kMeshletStreamActiveBuildEntryPoint = "gpuDrivenStreamAssetBuildActiveMain";

inline constexpr uint32_t kMeshletStreamDebugPage = 0;
inline constexpr uint32_t kMeshletStreamDebugLod = 1;
inline constexpr uint32_t kMeshletStreamDebugPrimitive = 2;
inline constexpr uint32_t kMeshletStreamDebugInstance = 3;
inline constexpr uint32_t kMeshletStreamDebugMeshlet = 4;
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
inline constexpr uint32_t kMeshletStreamDefaultMaxActiveGroups = 262144;
inline constexpr uint32_t kMeshletStreamDefaultTraversalWorkers = 1024;
inline constexpr uint32_t kMeshletStreamDefaultTraversalWorkItems = 1048576;
inline constexpr uint32_t kMeshletStreamMaxTraversalWorkers = 65535u * 64u;
inline constexpr uint32_t kMeshletStreamMaxTraversalWorkItems = 16777216;

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
    uint32_t padding0 = 0;
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
    uint32_t padding0 = 0;
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

struct MeshletStreamGpuPageInfo {
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t pageIndex = 0;
    uint32_t clusterCount = 0;
};

struct MeshletStreamGpuGroup {
    uint32_t primitiveIndex = 0;
    uint32_t pageIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t clusterRefOffset = 0;
    uint32_t clusterCount = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
    float boundsCenterRadius[4] = {};
    float maxQuadricError = 0.0f;
    uint32_t padding3 = 0;
    uint32_t padding4 = 0;
    uint32_t padding5 = 0;
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
};

struct MeshletStreamUserPush {
    uint32_t pageBuffer = 0;
    uint32_t activeGroupBuffer = 0;
    uint32_t pageTableBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t requestBuffer = 0;
    uint32_t updateBuffer = 0;
    uint32_t activeHeaderBuffer = 0;
    uint32_t instanceBuffer = 0;
    uint32_t primitiveBuffer = 0;
    uint32_t lodLevelBuffer = 0;
    uint32_t pageInfoBuffer = 0;
    uint32_t groupBuffer = 0;
    uint32_t clusterRefBuffer = 0;
    uint32_t nodeBuffer = 0;
    uint32_t drawIndirectBuffer = 0;
    uint32_t traversalHeaderBuffer = 0;
    uint32_t traversalWorkBuffer = 0;
    uint32_t traversalPhase = kMeshletStreamTraversalLoadPhase;
    uint32_t activeBuildPhase = kMeshletStreamActiveBuildBuildPhase;
};

static_assert(sizeof(MeshletStreamGpuActiveHeader) == 32);
static_assert(sizeof(MeshletStreamGpuActiveGroup) == 112);
static_assert(sizeof(MeshletStreamGpuInstance) == 96);
static_assert(sizeof(MeshletStreamGpuPrimitive) == 64);
static_assert(sizeof(MeshletStreamGpuLodLevel) == 32);
static_assert(sizeof(MeshletStreamGpuPageInfo) == 16);
static_assert(sizeof(MeshletStreamGpuGroup) == 64);
static_assert(sizeof(MeshletStreamGpuNode) == 48);
static_assert(sizeof(MeshletStreamGpuDrawIndirect) == 12);
static_assert(sizeof(MeshletStreamGpuTraversalHeader) == 32);
static_assert(sizeof(MeshletStreamGpuTraversalWorkItem) == 16);
static_assert(sizeof(StreamPageTableEntry) == 32);
static_assert(sizeof(MeshletStreamGpuParams) == 176);
static_assert(sizeof(MeshletStreamUserPush) == 76);

struct MeshletStreamRuntimeDesc {
    std::filesystem::path sourcePath;
    std::filesystem::path streamAssetPath;
    bool autoBuildStreamAsset = false;
    uint64_t maxResidentBytes = 0;
    uint32_t maxResidentPages = 4096;
    uint32_t maxPageUploadsPerFrame = 64;
    uint32_t maxGpuPageRequests = kMeshletStreamDefaultMaxGpuPageRequests;
    uint32_t maxGpuPageUnloadRequests = kMeshletStreamDefaultMaxGpuPageRequests;
    uint32_t maxActiveGroups = kMeshletStreamDefaultMaxActiveGroups;
    uint32_t maxTraversalWorkers = kMeshletStreamDefaultTraversalWorkers;
    uint32_t maxTraversalWorkItems = kMeshletStreamDefaultTraversalWorkItems;
    uint32_t pageLoadWorkerCount = 2;
    uint32_t maxPageLoadsInFlight = 128;
    uint32_t queuedFrameCount = 3;
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
    void reset();

    bool ready() const;

    Result cmdBeginFrame(CommandBuffer& commandBuffer, Streamer& streamer, const MeshletStreamFrameDesc& frame);
    Result cmdPreTraversal(CommandBuffer& commandBuffer, const MeshletStreamFrameDesc& frame);
    Result cmdPostTraversal(CommandBuffer& commandBuffer);
    Result cmdEndFrame(CommandBuffer& commandBuffer);

    BindlessHeap* bindlessHeap() const { return bindlessHeap_.get(); }
    MeshletStreamUserPush userPush() const;
    uint32_t drawTaskCount() const;
    void cmdDrawMeshTasks(CommandBuffer& commandBuffer) const;
    const scene::Bounds& bounds() const { return drawBounds_; }
    const scene::MeshletStreamAsset& asset() const { return asset_; }
    const MeshletStreamResidencyManager& residency() const { return residency_; }

private:
    class UpdatePass;
    class TraversalPass;
    class ActiveBuildPass;

    uint32_t computeMaxActiveGroups(uint32_t capacity) const;
    uint32_t computeMaxPageClusters() const;
    uint32_t computeMaxPrimitiveGroups() const;
    Result initializeSceneMetadataBuffers(Device& device, std::string& log);

    Result initializePageTableIfNeeded(CommandBuffer& commandBuffer);
    Result applyPageTablePatches(CommandBuffer& commandBuffer);
    Result clearRequestBuffer(CommandBuffer& commandBuffer);
    Result dispatchTraversal(CommandBuffer& commandBuffer, uint32_t threadCount, uint32_t traversalPhase);
    Result buildActiveTable(CommandBuffer& commandBuffer);
    Result copyRequestBufferForReadback(CommandBuffer& commandBuffer);
    Result updateParamsBuffer(const MeshletStreamFrameDesc& frame);
    Result transitionPageBufferForTraversal(CommandBuffer& commandBuffer);
    void consumeGpuRequestReadback();

    scene::MeshletStreamAsset asset_;
    MeshletStreamResidencyManager residency_;
    scene::Bounds drawBounds_;
    std::vector<StreamPageTableEntry> pageTable_;
    std::unique_ptr<Buffer> pageBuffer_;
    std::unique_ptr<Buffer> activeGroupBuffer_;
    std::unique_ptr<Buffer> activeHeaderBuffer_;
    std::unique_ptr<Buffer> pageTableBuffer_;
    std::unique_ptr<Buffer> pageTableUploadBuffer_;
    std::unique_ptr<Buffer> requestBuffer_;
    std::unique_ptr<Buffer> requestReadbackBuffer_;
    std::unique_ptr<Buffer> requestClearBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<Buffer> instanceBuffer_;
    std::unique_ptr<Buffer> primitiveBuffer_;
    std::unique_ptr<Buffer> lodLevelBuffer_;
    std::unique_ptr<Buffer> pageInfoBuffer_;
    std::unique_ptr<Buffer> groupBuffer_;
    std::unique_ptr<Buffer> clusterRefBuffer_;
    std::unique_ptr<Buffer> nodeBuffer_;
    std::unique_ptr<Buffer> drawIndirectBuffer_;
    std::unique_ptr<Buffer> traversalHeaderBuffer_;
    std::unique_ptr<Buffer> traversalWorkBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    std::unique_ptr<UpdatePass> updatePass_;
    std::unique_ptr<TraversalPass> traversalPass_;
    std::unique_ptr<ActiveBuildPass> activeBuildPass_;
    BindlessHandle pageHandle_;
    BindlessHandle activeGroupHandle_;
    BindlessHandle activeHeaderHandle_;
    BindlessHandle pageTableHandle_;
    BindlessHandle paramsHandle_;
    BindlessHandle requestHandle_;
    BindlessHandle instanceHandle_;
    BindlessHandle primitiveHandle_;
    BindlessHandle lodLevelHandle_;
    BindlessHandle pageInfoHandle_;
    BindlessHandle groupHandle_;
    BindlessHandle clusterRefHandle_;
    BindlessHandle nodeHandle_;
    BindlessHandle drawIndirectHandle_;
    BindlessHandle traversalHeaderHandle_;
    BindlessHandle traversalWorkHandle_;
    ResourceState pageBufferState_ = ResourceState::Undefined;
    ResourceState activeGroupBufferState_ = ResourceState::Undefined;
    ResourceState activeHeaderBufferState_ = ResourceState::Undefined;
    ResourceState pageTableState_ = ResourceState::Undefined;
    ResourceState requestBufferState_ = ResourceState::Undefined;
    ResourceState drawIndirectBufferState_ = ResourceState::Undefined;
    ResourceState traversalHeaderBufferState_ = ResourceState::Undefined;
    ResourceState traversalWorkBufferState_ = ResourceState::Undefined;
    bool pageTableInitialized_ = false;
    bool requestReadbackValid_ = false;
    uint32_t frameIndex_ = 0;
    uint32_t maxResidentPages_ = 0;
    uint32_t maxPageUploadsPerFrame_ = 0;
    uint32_t maxGpuPageRequests_ = 0;
    uint32_t maxGpuPageUnloadRequests_ = 0;
    uint32_t maxUpdatePatches_ = 0;
    uint64_t maxResidentBytes_ = 0;
    uint32_t maxActiveGroups_ = 0;
    uint32_t maxActiveGroupClusters_ = 0;
    uint32_t maxPrimitiveGroupCount_ = 0;
    uint32_t traversalWorkerCount_ = 0;
    uint32_t traversalWorkCapacity_ = 0;
    uint32_t currentFrameUploadCount_ = 0;
};

} // namespace metallic::render
