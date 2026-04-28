#pragma once

#include "cluster_types.h"
#include "hzb_constants.h"
#include "hzb_spd_constants.h"
#include "rhi_backend.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

struct ClusterOcclusionState {
    static constexpr uint32_t kPhase0 = 0u;
    static constexpr uint32_t kPhase1 = 1u;
    static constexpr uint32_t kCounterPhase0Visible = 0u;
    static constexpr uint32_t kCounterPhase0Recheck = 4u;
    static constexpr uint32_t kCounterPhase1Visible = 8u;
    static constexpr uint32_t kIndirectPhase0Offset = 0u;
    static constexpr uint32_t kIndirectPhase1Offset = 12u;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t maxClusters = 0;
    uint32_t maxInstances = 0;
    uint32_t mipCount = 0;
    bool hizValid[2] = {false, false};
    bool worklistValid[2] = {false, false};

    std::unique_ptr<RhiBuffer> phase0VisibleWorklist;
    std::unique_ptr<RhiBuffer> phase0RecheckWorklist;
    std::unique_ptr<RhiBuffer> phase1VisibleWorklist;
    std::unique_ptr<RhiBuffer> counters;
    std::unique_ptr<RhiBuffer> indirectArgs;
    std::unique_ptr<RhiBuffer> spdAtomicCounter;
    std::array<std::unique_ptr<RhiTexture>, kHzbMaxLevels> hzb[2];

    // Instance culling buffers
    std::unique_ptr<RhiBuffer> visibleInstanceBuffer;
    std::unique_ptr<RhiBuffer> instanceCounters;
    std::unique_ptr<RhiBuffer> instanceIndirectArgs;

    bool ensure(RhiFrameGraphBackend& factory,
                uint32_t newWidth,
                uint32_t newHeight,
                uint32_t newMaxClusters,
                uint32_t newMaxInstances = 0) {
        newWidth = std::max(newWidth, 1u);
        newHeight = std::max(newHeight, 1u);
        newMaxClusters = std::max(newMaxClusters, 1u);
        newMaxInstances = std::max(newMaxInstances, 1u);

        const SpdSetupInfo spdInfo = spdSetup(newWidth, newHeight);
        const uint32_t newMipCount = std::min(spdInfo.mipCount + 1u, kHzbMaxLevels);

        const bool needsResize =
            width != newWidth ||
            height != newHeight ||
            mipCount != newMipCount ||
            maxClusters < newMaxClusters ||
            !phase0VisibleWorklist ||
            !phase0RecheckWorklist ||
            !phase1VisibleWorklist ||
            !counters ||
            !indirectArgs ||
            !spdAtomicCounter ||
            !hizTexturesReady(0u, newMipCount) ||
            !hizTexturesReady(1u, newMipCount);

        const bool needsInstanceResize =
            maxInstances < newMaxInstances ||
            !visibleInstanceBuffer ||
            !instanceCounters ||
            !instanceIndirectArgs;

        if (!needsResize && !needsInstanceResize) {
            return true;
        }

        if (needsResize) {
            width = newWidth;
            height = newHeight;
            maxClusters = newMaxClusters;
            mipCount = newMipCount;
            hizValid[0] = false;
            hizValid[1] = false;
            worklistValid[0] = false;
            worklistValid[1] = false;

            phase0VisibleWorklist = createBuffer(factory,
                                                 maxClusters * sizeof(ClusterInfo),
                                                 "ClusterCull Phase0 Visible");
            phase0RecheckWorklist = createBuffer(factory,
                                                 maxClusters * sizeof(ClusterInfo),
                                                 "ClusterCull Phase0 Recheck");
            phase1VisibleWorklist = createBuffer(factory,
                                                 maxClusters * sizeof(ClusterInfo),
                                                 "ClusterCull Phase1 Visible");
            counters = createBuffer(factory, 16u, "ClusterCull Counters");
            indirectArgs = createBuffer(factory, 24u, "ClusterCull Mesh Indirect Args");
            const uint32_t zero = 0u;
            spdAtomicCounter = createBuffer(factory,
                                            sizeof(uint32_t),
                                            "ClusterCull HZB SPD Atomic Counter",
                                            true,
                                            &zero);
            for (uint32_t pyramid = 0; pyramid < 2u; ++pyramid) {
                for (auto& texture : hzb[pyramid]) {
                    texture.reset();
                }
                for (uint32_t level = 0; level < mipCount; ++level) {
                    hzb[pyramid][level] = factory.createTexture(makeHzbTextureDesc(width, height, level));
                }
            }
        }

        if (needsInstanceResize) {
            maxInstances = newMaxInstances;
            // visibleInstanceBuffer: stores uint indices of visible instances
            visibleInstanceBuffer = createBuffer(factory,
                                                 maxInstances * sizeof(uint32_t),
                                                 "InstanceCull Visible Instances");
            // instanceCounters: [0]=phase0 visible, [4]=phase0 rejected, [8]=phase1 visible
            instanceCounters = createBuffer(factory, 16u, "InstanceCull Counters");
            // instanceIndirectArgs: dispatch args for downstream passes
            instanceIndirectArgs = createBuffer(factory, 12u, "InstanceCull Indirect Args");
        }

        return phase0VisibleWorklist &&
               phase0RecheckWorklist &&
               phase1VisibleWorklist &&
               counters &&
               indirectArgs &&
               spdAtomicCounter &&
               visibleInstanceBuffer &&
               instanceCounters &&
               instanceIndirectArgs &&
               hizTexturesReady(0u) &&
               hizTexturesReady(1u);
    }

    void resetHistory() {
        hizValid[0] = false;
        hizValid[1] = false;
    }

    void resetWorklists() {
        worklistValid[0] = false;
        worklistValid[1] = false;
    }

    RhiBuffer* visibleWorklist(uint32_t phase) const {
        return phase == kPhase1 ? phase1VisibleWorklist.get() : phase0VisibleWorklist.get();
    }

    RhiBuffer* recheckWorklist() const { return phase0RecheckWorklist.get(); }

    RhiBuffer* spdCounter() const {
        return spdAtomicCounter.get();
    }

    RhiTexture* hizTexture(uint32_t index, uint32_t level) const {
        if (level >= mipCount || level >= kHzbMaxLevels) {
            return nullptr;
        }
        return hzb[index & 1u][level].get();
    }

    bool hizTexturesReady(uint32_t index) const {
        return hizTexturesReady(index, mipCount);
    }

    uint32_t indirectOffset(uint32_t phase) const {
        return phase == kPhase1 ? kIndirectPhase1Offset : kIndirectPhase0Offset;
    }

    uint32_t mipWidth(uint32_t level) const {
        if (mipCount == 0u) {
            return 1u;
        }
        return hzbLevelDimension(width, std::min(level, mipCount - 1u));
    }

    uint32_t mipHeight(uint32_t level) const {
        if (mipCount == 0u) {
            return 1u;
        }
        return hzbLevelDimension(height, std::min(level, mipCount - 1u));
    }

private:
    bool hizTexturesReady(uint32_t index, uint32_t levels) const {
        if (levels == 0u || levels > kHzbMaxLevels) {
            return false;
        }
        for (uint32_t level = 0; level < levels; ++level) {
            if (!hzb[index & 1u][level]) {
                return false;
            }
        }
        return true;
    }

    static std::unique_ptr<RhiBuffer> createBuffer(RhiFrameGraphBackend& factory,
                                                   size_t size,
                                                   const char* debugName,
                                                   bool hostVisible = false,
                                                   const void* initialData = nullptr) {
        RhiBufferDesc desc;
        desc.size = std::max<size_t>(size, 4u);
        desc.initialData = initialData;
        desc.hostVisible = hostVisible;
        desc.debugName = debugName;
        return factory.createBuffer(desc);
    }
};
