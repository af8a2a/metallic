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

inline constexpr uint32_t kMeshletStreamDebugPage = 0;
inline constexpr uint32_t kMeshletStreamDebugLod = 1;
inline constexpr uint32_t kMeshletStreamDebugPrimitive = 2;
inline constexpr uint32_t kMeshletStreamInvalidClusterIndex = UINT32_MAX;
inline constexpr uint32_t kMeshletStreamUnloadClusterIndex = UINT32_MAX - 1u;
inline constexpr uint32_t kMeshletStreamDefaultMaxGpuPageRequests = 65536;
inline constexpr uint32_t kMeshletStreamActiveGroupResident = 1u << 0;
inline constexpr uint32_t kMeshletStreamActiveGroupLoadRequest = 1u << 1;
inline constexpr uint32_t kMeshletStreamActiveGroupUnloadRequest = 1u << 2;

struct MeshletStreamGpuActiveGroup {
    uint32_t pageDeviceOffsetBytes = kInvalidStreamDeviceOffsetBytes;
    uint32_t pageIndex = 0;
    uint32_t clusterCount = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t materialIndex = 0;
    uint32_t colorSeed = 0;
    uint32_t flags = 0;
    float world0[4] = {};
    float world1[4] = {};
    float world2[4] = {};
    float world3[4] = {};
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
};

struct MeshletStreamUserPush {
    uint32_t pageBuffer = 0;
    uint32_t activeGroupBuffer = 0;
    uint32_t pageTableBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t requestBuffer = 0;
    uint32_t updateBuffer = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

static_assert(sizeof(MeshletStreamGpuActiveGroup) == 96);
static_assert(sizeof(StreamPageTableEntry) == 32);
static_assert(sizeof(MeshletStreamGpuParams) == 128);

struct MeshletStreamRuntimeDesc {
    std::filesystem::path sourcePath;
    std::filesystem::path streamAssetPath;
    bool autoBuildStreamAsset = true;
    uint64_t maxResidentBytes = 0;
    uint32_t maxResidentPages = 4096;
    uint32_t maxPageUploadsPerFrame = 64;
    uint32_t maxGpuPageRequests = kMeshletStreamDefaultMaxGpuPageRequests;
    uint32_t maxGpuPageUnloadRequests = kMeshletStreamDefaultMaxGpuPageRequests;
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
    uint32_t selectedLodLevel = 0;
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
    const scene::Bounds& bounds() const { return drawBounds_; }
    const scene::MeshletStreamAsset& asset() const { return asset_; }
    const MeshletStreamResidencyManager& residency() const { return residency_; }

private:
    class UpdatePass;

    uint32_t computeMaxActiveGroups() const;
    uint32_t computeMaxPageClusters() const;
    void appendResidentPageGroup(
        const scene::MeshletStreamInstanceInfo& instance,
        const scene::MeshletStreamPageInfo& page,
        uint32_t pageIndex);
    void appendLoadRequestGroup(
        const scene::MeshletStreamInstanceInfo& instance,
        const scene::MeshletStreamPageInfo& page,
        uint32_t pageIndex);
    bool appendResidentPageRange(
        const scene::MeshletStreamInstanceInfo& instance,
        uint32_t pageOffset,
        uint32_t pageCount);
    void appendRequestPageRange(
        const scene::MeshletStreamInstanceInfo& instance,
        uint32_t pageOffset,
        uint32_t pageCount);
    void buildFrameActiveGroups(uint32_t selectedLodLevel);

    Result initializePageTableIfNeeded(CommandBuffer& commandBuffer);
    Result applyPageTablePatches(CommandBuffer& commandBuffer);
    Result clearRequestBuffer(CommandBuffer& commandBuffer);
    Result copyRequestBufferForReadback(CommandBuffer& commandBuffer);
    Result updateActiveGroupBuffer();
    Result updateParamsBuffer(const MeshletStreamFrameDesc& frame);
    Result transitionPageBufferForTraversal(CommandBuffer& commandBuffer);
    void consumeGpuRequestReadback();

    scene::MeshletStreamAsset asset_;
    MeshletStreamResidencyManager residency_;
    scene::Bounds drawBounds_;
    std::vector<MeshletStreamGpuActiveGroup> activeGroups_;
    std::vector<StreamPageTableEntry> pageTable_;
    std::unique_ptr<Buffer> pageBuffer_;
    std::unique_ptr<Buffer> activeGroupBuffer_;
    std::unique_ptr<Buffer> pageTableBuffer_;
    std::unique_ptr<Buffer> pageTableUploadBuffer_;
    std::unique_ptr<Buffer> requestBuffer_;
    std::unique_ptr<Buffer> requestReadbackBuffer_;
    std::unique_ptr<Buffer> requestClearBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    std::unique_ptr<UpdatePass> updatePass_;
    BindlessHandle pageHandle_;
    BindlessHandle activeGroupHandle_;
    BindlessHandle pageTableHandle_;
    BindlessHandle paramsHandle_;
    BindlessHandle requestHandle_;
    ResourceState pageBufferState_ = ResourceState::Undefined;
    ResourceState pageTableState_ = ResourceState::Undefined;
    ResourceState requestBufferState_ = ResourceState::Undefined;
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
    uint32_t currentFrameUploadCount_ = 0;
};

} // namespace metallic::render
