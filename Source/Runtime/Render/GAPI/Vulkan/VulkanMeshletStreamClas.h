#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace metallic::render::vulkan {

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
    MeshletStreamClasPoolStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::vulkan
