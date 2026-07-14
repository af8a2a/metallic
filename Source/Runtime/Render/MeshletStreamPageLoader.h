#pragma once

#include "Runtime/Scene/MeshletStreamAsset.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace metallic::render {

inline constexpr uint32_t kMeshletStreamMaxPageLoadWorkers = 32;

struct MeshletStreamPageLoadResult {
    uint32_t pageIndex = UINT32_MAX;
    std::vector<uint8_t> payload;
    std::string failureReason;

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

    bool initialize(const scene::MeshletStreamAsset& asset, uint32_t workerCount, std::string& reason);
    void reset();

    bool enqueue(uint32_t pageIndex);
    bool tryPop(MeshletStreamPageLoadResult& outResult);

    bool ready() const;
    uint32_t workerCount() const;
    uint32_t pendingCount() const;
    uint32_t activeCount() const;
    uint32_t completedCount() const;
    uint32_t outstandingCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
