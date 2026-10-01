#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/MeshletStreamGPUCodec.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

struct CPUProfileRecorder;

struct MeshletStreamCLASClusterInput {
    uint32_t clusterId = 0;
    uint32_t pageIndex = 0;
    uint32_t clusterIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexOffsetBytes = 0;
    uint32_t vertexCount = 0;
    uint32_t vertexStrideBytes = 16;
    uint32_t triangleOffsetBytes = 0;
    uint32_t triangleCount = 0;
};

struct MeshletStreamCLASPagePlan {
    uint32_t pageIndex = 0;
    uint32_t firstClusterId = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t payloadByteSize = 0;
    std::vector<MeshletStreamCLASClusterInput> clusters;
};

// The sideband must have passed inspectMeshletStreamGpuPage in the page loader.
bool buildMeshletStreamClasGpuPagePlan(const scene::MeshletStreamGPUPage& page,
    uint32_t pageIndex, uint32_t firstClusterId, MeshletStreamCLASPagePlan& outPlan, std::string& reason);

inline constexpr uint32_t kInvalidMeshletStreamCLASAddressOffset = UINT32_MAX;

enum class MeshletStreamCLASPageState : uint32_t {
    Empty = 0,
    Active = 1,
    Retiring = 2,
};

inline constexpr uint32_t kMeshletStreamCLASPageAddressOffsetMask = 0x3fffffffu;
inline constexpr uint32_t kMeshletStreamCLASPageStateShift = 30u;

struct MeshletStreamCLASPageEntry {
    uint32_t addressOffsetAndState = 0;
    uint32_t publicationGeneration = 0;
};

static_assert(sizeof(MeshletStreamCLASPageEntry) == 8);

inline constexpr uint32_t packMeshletStreamClasPageEntry(
    uint32_t addressOffset,
    MeshletStreamCLASPageState state)
{
    return (addressOffset & kMeshletStreamCLASPageAddressOffsetMask) |
        (static_cast<uint32_t>(state) << kMeshletStreamCLASPageStateShift);
}

inline constexpr uint32_t meshletStreamClasPageAddressOffset(
    const MeshletStreamCLASPageEntry& entry)
{
    return (entry.addressOffsetAndState >> kMeshletStreamCLASPageStateShift) ==
            static_cast<uint32_t>(MeshletStreamCLASPageState::Empty)
        ? kInvalidMeshletStreamCLASAddressOffset
        : entry.addressOffsetAndState & kMeshletStreamCLASPageAddressOffsetMask;
}

inline constexpr MeshletStreamCLASPageState meshletStreamClasPageState(
    const MeshletStreamCLASPageEntry& entry)
{
    return static_cast<MeshletStreamCLASPageState>(
        entry.addressOffsetAndState >> kMeshletStreamCLASPageStateShift);
}

struct MeshletStreamCLASPoolDesc {
    const scene::MeshletStreamAsset* asset = nullptr;
    uint64_t maxStorageBytes = 512ull * 1024ull * 1024ull;
    uint32_t maxBuildClusters = 2048;
    uint32_t queuedFrameCount = 3;
    bool compactStorage = false; // Opt-in until the compact path is validated.
    uint64_t startStorageBytes = 0; // Initial physical commitment; may shrink to zero when unused.
    uint64_t growStorageBytes = 64ull * 1024ull * 1024ull;
    uint32_t emptyChunkRetentionFrames = 0; // After all GPU readers finish; zero releases immediately.
    uint64_t persistentGrowStorageBytes = 0; // Zero uses the common growth quantum.
    std::span<const uint32_t> persistentPages; // Copied at initialization; segregate locked roots from evictable pages.
};

struct MeshletStreamCLASPageBuild {
    uint32_t pageIndex = UINT32_MAX;
    uint64_t deviceOffsetBytes = UINT64_MAX;
    const MeshletStreamCLASPagePlan* plan = nullptr;
};

struct MeshletStreamCLASPoolStats {
    uint64_t publicationRevision = 0;
    uint32_t pageCapacity = 0;
    uint32_t trackedPageCount = 0;
    uint32_t clusterSlotCapacity = 0;
    uint32_t builtPageCount = 0;
    uint32_t builtClusterCount = 0;
    uint32_t retiringPageCount = 0;
    uint32_t retiringClusterCount = 0;
    uint32_t frameBuiltPageCount = 0;
    uint32_t frameBuiltClusterCount = 0;
    uint32_t frameRejectedPageCount = 0;
    uint64_t totalBuiltPageCount = 0;
    uint64_t totalBuiltClusterCount = 0;
    uint64_t totalRejectedPageCount = 0;
    uint64_t storageBytes = 0; // Physical backing, including empty chunks awaiting GPU completion.
    uint64_t storageBudgetBytes = 0;
    uint32_t storageChunkCount = 0;
    uint64_t startStorageBytes = 0, growStorageBytes = 0;
    uint64_t emptyStorageBytes = 0;
    uint64_t persistentStorageBytes = 0, persistentUsedBytes = 0, persistentGrowStorageBytes = 0;
    uint64_t transientStorageBytes = 0, transientUsedBytes = 0;
    uint64_t fragmentedFreeBytes = 0; // Sum of free bytes outside each chunk's largest hole.
    uint64_t totalStorageGrowthCount = 0, totalStorageReleasedBytes = 0;
    uint32_t frameStorageGrowthCount = 0, frameStorageReleaseCount = 0;
    uint64_t usedStorageBytes = 0;
    uint64_t clusterStrideBytes = 0;
    uint64_t scratchBytes = 0;
    uint64_t encodedStorageBytes = 0;
    uint64_t worstCaseStorageBytes = 0;
    uint64_t retiringStorageBytes = 0;
    uint32_t frameMovedPageCount = 0;
    uint32_t frameMovedClusterCount = 0;
};

class MeshletStreamCompactCLASPool;

class MeshletStreamCLASPool {
public:
    MeshletStreamCLASPool();
    ~MeshletStreamCLASPool();

    MeshletStreamCLASPool(const MeshletStreamCLASPool&) = delete;
    MeshletStreamCLASPool& operator=(const MeshletStreamCLASPool&) = delete;

    MeshletStreamCLASPool(MeshletStreamCLASPool&&) noexcept;
    MeshletStreamCLASPool& operator=(MeshletStreamCLASPool&&) noexcept;

    Result<> initialize(Device& device, const MeshletStreamCLASPoolDesc& desc, std::string& log);
    void clear();
    void beginFrame(CPUProfileRecorder* profiler = nullptr);

    Result<> cmdBuildPages(
        CommandBuffer& commandBuffer,
        Buffer& pageBuffer,
        std::span<const MeshletStreamCLASPageBuild> pages,
        std::string& log,
        Buffer* sizeOutput = nullptr);
    void retirePages(std::span<const uint32_t> pageIndices);
    // Empty span denotes a whole-pool clear. Called before destructive changes.
    void setInvalidationObserver(std::function<void(std::span<const uint32_t>)> observer)
    {
        invalidationObserver_ = std::move(observer);
    }

    bool ready() const;
    bool pageHasClas(uint32_t pageIndex) const;
    bool pageBuildPending(uint32_t pageIndex) const;
    uint64_t pageStorageBytes(uint32_t pageIndex) const;
    // Only the legacy staging builder has one contiguous storage buffer.
    Buffer* storageBuffer() const;
    Buffer* pageStorageBuffer(uint32_t pageIndex) const;
    uint32_t pageClasAddressOffset(uint32_t pageIndex) const;
    uint64_t clusterAddress(uint32_t pageIndex, uint32_t clusterIndex) const;
    Buffer* clusterAddressBuffer() const;
    Buffer* pageTableBuffer() const;
    MeshletStreamCLASPoolStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<MeshletStreamCompactCLASPool> compact_;
    std::function<void(std::span<const uint32_t>)> invalidationObserver_;
};

bool buildMeshletStreamPageClusterOffsets(
    const scene::MeshletStreamAsset& asset,
    std::vector<uint32_t>& outOffsets,
    uint32_t& outClusterCount,
    std::string& reason);

bool buildMeshletStreamClasPagePlan(
    const scene::MeshletStreamPageInfo& page,
    std::span<const uint8_t> devicePayload,
    uint32_t pageIndex,
    uint32_t firstClusterId,
    MeshletStreamCLASPagePlan& outPlan,
    std::string& reason);

bool buildMeshletStreamClasPagePlan(
    const scene::MeshletStreamAsset& asset,
    uint32_t pageIndex,
    uint32_t firstClusterId,
    MeshletStreamCLASPagePlan& outPlan,
    std::string& reason);

} // namespace metallic::render
