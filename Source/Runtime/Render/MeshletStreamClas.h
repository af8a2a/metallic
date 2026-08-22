#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

struct MeshletStreamClasClusterInput {
    uint32_t clusterId = 0;
    uint32_t pageIndex = 0;
    uint32_t clusterIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexOffsetBytes = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleOffsetBytes = 0;
    uint32_t triangleCount = 0;
};

struct MeshletStreamClasPagePlan {
    uint32_t pageIndex = 0;
    uint32_t firstClusterId = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t payloadByteSize = 0;
    std::vector<MeshletStreamClasClusterInput> clusters;
};

inline constexpr uint32_t kInvalidMeshletStreamClasAddressOffset = UINT32_MAX;

enum class MeshletStreamClasPageState : uint32_t {
    Empty = 0,
    Active = 1,
    Retiring = 2,
};

inline constexpr uint32_t kMeshletStreamClasPageAddressOffsetMask = 0x3fffffffu;
inline constexpr uint32_t kMeshletStreamClasPageStateShift = 30u;

struct MeshletStreamClasPageEntry {
    uint32_t addressOffsetAndState = 0;
};

static_assert(sizeof(MeshletStreamClasPageEntry) == 4);

inline constexpr uint32_t packMeshletStreamClasPageEntry(
    uint32_t addressOffset,
    MeshletStreamClasPageState state)
{
    return (addressOffset & kMeshletStreamClasPageAddressOffsetMask) |
        (static_cast<uint32_t>(state) << kMeshletStreamClasPageStateShift);
}

inline constexpr uint32_t meshletStreamClasPageAddressOffset(
    const MeshletStreamClasPageEntry& entry)
{
    return (entry.addressOffsetAndState >> kMeshletStreamClasPageStateShift) ==
            static_cast<uint32_t>(MeshletStreamClasPageState::Empty)
        ? kInvalidMeshletStreamClasAddressOffset
        : entry.addressOffsetAndState & kMeshletStreamClasPageAddressOffsetMask;
}

inline constexpr MeshletStreamClasPageState meshletStreamClasPageState(
    const MeshletStreamClasPageEntry& entry)
{
    return static_cast<MeshletStreamClasPageState>(
        entry.addressOffsetAndState >> kMeshletStreamClasPageStateShift);
}

struct MeshletStreamClasPoolDesc {
    const scene::MeshletStreamAsset* asset = nullptr;
    uint64_t maxStorageBytes = 512ull * 1024ull * 1024ull;
    uint32_t maxBuildClusters = 2048;
    uint32_t queuedFrameCount = 3;
};

struct MeshletStreamClasPageBuild {
    uint32_t pageIndex = UINT32_MAX;
    uint64_t deviceOffsetBytes = UINT64_MAX;
};

struct MeshletStreamClasPoolStats {
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
    uint64_t storageBytes = 0;
    uint64_t usedStorageBytes = 0;
    uint64_t clusterStrideBytes = 0;
    uint64_t scratchBytes = 0;
};

class MeshletStreamClasPool {
public:
    MeshletStreamClasPool();
    ~MeshletStreamClasPool();

    MeshletStreamClasPool(const MeshletStreamClasPool&) = delete;
    MeshletStreamClasPool& operator=(const MeshletStreamClasPool&) = delete;

    MeshletStreamClasPool(MeshletStreamClasPool&&) noexcept;
    MeshletStreamClasPool& operator=(MeshletStreamClasPool&&) noexcept;

    Result initialize(Device& device, const MeshletStreamClasPoolDesc& desc, std::string& log);
    void clear();
    void beginFrame();

    Result cmdBuildPages(
        CommandBuffer& commandBuffer,
        Buffer& pageBuffer,
        std::span<const MeshletStreamClasPageBuild> pages,
        std::string& log);
    void retirePages(std::span<const uint32_t> pageIndices);

    bool ready() const;
    bool pageHasClas(uint32_t pageIndex) const;
    uint32_t pageClasAddressOffset(uint32_t pageIndex) const;
    uint64_t clusterAddress(uint32_t pageIndex, uint32_t clusterIndex) const;
    Buffer* clusterAddressBuffer() const;
    Buffer* pageTableBuffer() const;
    MeshletStreamClasPoolStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
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
    MeshletStreamClasPagePlan& outPlan,
    std::string& reason);

bool buildMeshletStreamClasPagePlan(
    const scene::MeshletStreamAsset& asset,
    uint32_t pageIndex,
    uint32_t firstClusterId,
    MeshletStreamClasPagePlan& outPlan,
    std::string& reason);

} // namespace metallic::render
