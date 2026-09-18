#pragma once

#include "Runtime/Scene/MeshletStreamGpuCodec.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace metallic::render {

inline constexpr uint32_t kMeshletStreamMaxPageLoadConcurrency = 32;

struct MeshletStreamPageLoadResult {
    uint32_t pageIndex = UINT32_MAX;
    std::vector<uint8_t> payload;
    bool gpuEncoded = false;
    scene::MeshletStreamGpuPage gpuPage;
    std::string failureReason;
    uint64_t startedMicroseconds = 0;
    uint64_t completedMicroseconds = 0;

    bool success() const { return pageIndex != UINT32_MAX && !payload.empty(); }
};

class MeshletStreamPageLoader {
public:
    MeshletStreamPageLoader();
    ~MeshletStreamPageLoader();

    MeshletStreamPageLoader(const MeshletStreamPageLoader&) = delete;
    MeshletStreamPageLoader& operator=(const MeshletStreamPageLoader&) = delete;

    MeshletStreamPageLoader(MeshletStreamPageLoader&&) noexcept = delete;
    MeshletStreamPageLoader& operator=(MeshletStreamPageLoader&&) noexcept = delete;

    bool initialize(const scene::MeshletStreamAsset& asset, uint32_t concurrency, std::string& reason,
        bool gpuDecompression = false);
    void reset();

    // A per-request choice: small cohorts decode on the worker without waiting
    // for more requests or adding CPU decode work to the render thread.
    bool enqueue(uint32_t pageIndex, bool allowGpuDecompression = true);
    bool tryPop(MeshletStreamPageLoadResult& outResult);

    bool ready() const;
    uint32_t concurrency() const;
    uint32_t pendingCount() const;
    uint32_t activeCount() const;
    uint32_t completedCount() const;
    uint32_t outstandingCount() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render
