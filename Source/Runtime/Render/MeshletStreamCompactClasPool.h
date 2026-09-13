#pragma once
#include "Runtime/Render/MeshletStreamClas.h"

namespace metallic::render {
// Optional compact backend. The existing pool remains the temporary builder and
// the compatibility path; this backend only owns relocation and publication.
class MeshletStreamCompactClasPool {
  public:
    MeshletStreamCompactClasPool();
    ~MeshletStreamCompactClasPool();
    Result initialize(Device&, const MeshletStreamClasPoolDesc&, std::string&);
    void beginFrame();
    Result cmdBuildPages(CommandBuffer&, Buffer&, std::span<const MeshletStreamClasPageBuild>, std::string&);
    void retirePages(std::span<const uint32_t>);
    bool ready() const;
    bool pageHasClas(uint32_t) const;
    bool pageBuildPending(uint32_t) const;
    uint64_t pageStorageBytes(uint32_t) const;
    uint32_t pageClasAddressOffset(uint32_t) const;
    uint64_t clusterAddress(uint32_t, uint32_t) const;
    Buffer* storageBuffer() const;
    Buffer* clusterAddressBuffer() const;
    Buffer* pageTableBuffer() const;
    MeshletStreamClasPoolStats stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace metallic::render
