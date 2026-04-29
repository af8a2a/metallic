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
    static constexpr uint32_t kDagQueueCount0 = 16u;
    static constexpr uint32_t kDagQueueCount1 = 20u;
    static constexpr uint32_t kDagNodeOverflow = 24u;
    static constexpr uint32_t kDagNodeProcessed = 28u;
    static constexpr uint32_t kDagClusterOverflow = 32u;
    static constexpr uint32_t kDagSeededInstances = 36u;
    static constexpr uint32_t kInstanceCounterPhase0Visible = 0u;
    static constexpr uint32_t kInstanceCounterPhase0Rejected = 4u;
    static constexpr uint32_t kInstanceCounterPhase1Visible = 8u;
    static constexpr uint32_t kIndirectPhase0Offset = 0u;
    static constexpr uint32_t kIndirectPhase1Offset = 12u;
    static constexpr uint32_t kDagCounterBytes = 64u;

    struct DagNodeTask {
        uint32_t instanceID = 0;
        uint32_t nodeID = 0;
    };
    static_assert(sizeof(DagNodeTask) == sizeof(ClusterInfo),
                  "DAG node task must remain 8 bytes");

    struct InstanceCullStats {
        bool countersReadable = false;
        bool indirectReadable = false;
        uint32_t phase0Visible = 0;
        uint32_t phase0Rejected = 0;
        uint32_t phase1Visible = 0;
        uint32_t dispatchGroups = 0;
    };

    struct DagCullStats {
        bool readable = false;
        uint32_t phase0Visible = 0;
        uint32_t phase0Recheck = 0;
        uint32_t phase1Visible = 0;
        uint32_t queueCount0 = 0;
        uint32_t queueCount1 = 0;
        uint32_t nodeOverflow = 0;
        uint32_t nodeProcessed = 0;
        uint32_t clusterOverflow = 0;
        uint32_t seededInstances = 0;
    };

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
    std::unique_ptr<RhiBuffer> dagNodeQueues[2];
    std::unique_ptr<RhiBuffer> spdAtomicCounter;
    std::array<std::unique_ptr<RhiTexture>, kHzbMaxLevels> hzb[2];

    // Instance culling buffers
    std::unique_ptr<RhiBuffer> visibleInstanceBuffer;
    std::unique_ptr<RhiBuffer> instanceVisibilityBuffer;
    std::unique_ptr<RhiBuffer> instanceCounters;
    std::unique_ptr<RhiBuffer> instanceIndirectArgs;
    bool instanceVisibilityValid = false;
    uint32_t instanceVisibilityFrameIndex = 0;

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
            !dagNodeQueues[0] ||
            !dagNodeQueues[1] ||
            !spdAtomicCounter ||
            !hizTexturesReady(0u, newMipCount) ||
            !hizTexturesReady(1u, newMipCount);

        const bool needsInstanceResize =
            maxInstances < newMaxInstances ||
            !visibleInstanceBuffer ||
            !instanceVisibilityBuffer ||
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
            const uint32_t zeroDagCounters[kDagCounterBytes / sizeof(uint32_t)] = {};
            counters = createBuffer(factory,
                                    kDagCounterBytes,
                                    "ClusterCull Counters",
                                    true,
                                    zeroDagCounters);
            indirectArgs = createBuffer(factory, 24u, "ClusterCull Mesh Indirect Args");
            dagNodeQueues[0] = createBuffer(factory,
                                            maxClusters * sizeof(DagNodeTask),
                                            "DagClusterCull Node Queue 0");
            dagNodeQueues[1] = createBuffer(factory,
                                            maxClusters * sizeof(DagNodeTask),
                                            "DagClusterCull Node Queue 1");
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
            const uint32_t zeroCounters[4] = {};
            const uint32_t zeroIndirectArgs[3] = {};
            // visibleInstanceBuffer: stores uint indices of visible instances
            visibleInstanceBuffer = createBuffer(factory,
                                                 maxInstances * sizeof(uint32_t),
                                                 "InstanceCull Visible Instances");
            // instanceVisibilityBuffer: one uint flag per instance for O(1) cluster worklist filtering
            instanceVisibilityBuffer = createBuffer(factory,
                                                    maxInstances * sizeof(uint32_t),
                                                    "InstanceCull Visibility Flags");
            // instanceCounters: [0]=phase0 visible, [4]=phase0 rejected, [8]=phase1 visible
            instanceCounters = createBuffer(factory,
                                            16u,
                                            "InstanceCull Counters",
                                            true,
                                            zeroCounters);
            // instanceIndirectArgs: dispatch args for downstream passes
            instanceIndirectArgs = createBuffer(factory,
                                                12u,
                                                "InstanceCull Indirect Args",
                                                true,
                                                zeroIndirectArgs);
        }

        return phase0VisibleWorklist &&
               phase0RecheckWorklist &&
               phase1VisibleWorklist &&
               counters &&
               indirectArgs &&
               dagNodeQueues[0] &&
               dagNodeQueues[1] &&
               spdAtomicCounter &&
               visibleInstanceBuffer &&
               instanceVisibilityBuffer &&
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
    RhiBuffer* dagNodeQueue(uint32_t index) const { return dagNodeQueues[index & 1u].get(); }

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

    InstanceCullStats readInstanceCullStats() {
        InstanceCullStats stats{};
        if (instanceCounters && instanceCounters->size() >= 12u) {
            if (const auto* counters = static_cast<const uint32_t*>(instanceCounters->mappedData())) {
                stats.countersReadable = true;
                stats.phase0Visible = counters[kInstanceCounterPhase0Visible / sizeof(uint32_t)];
                stats.phase0Rejected = counters[kInstanceCounterPhase0Rejected / sizeof(uint32_t)];
                stats.phase1Visible = counters[kInstanceCounterPhase1Visible / sizeof(uint32_t)];
            }
        }
        if (instanceIndirectArgs && instanceIndirectArgs->size() >= 4u) {
            if (const auto* indirectArgs = static_cast<const uint32_t*>(instanceIndirectArgs->mappedData())) {
                stats.indirectReadable = true;
                stats.dispatchGroups = indirectArgs[0];
            }
        }
        return stats;
    }

    DagCullStats readDagCullStats() {
        DagCullStats stats{};
        if (!counters || counters->size() < kDagCounterBytes) {
            return stats;
        }

        if (const auto* values = static_cast<const uint32_t*>(counters->mappedData())) {
            stats.readable = true;
            stats.phase0Visible = values[kCounterPhase0Visible / sizeof(uint32_t)];
            stats.phase0Recheck = values[kCounterPhase0Recheck / sizeof(uint32_t)];
            stats.phase1Visible = values[kCounterPhase1Visible / sizeof(uint32_t)];
            stats.queueCount0 = values[kDagQueueCount0 / sizeof(uint32_t)];
            stats.queueCount1 = values[kDagQueueCount1 / sizeof(uint32_t)];
            stats.nodeOverflow = values[kDagNodeOverflow / sizeof(uint32_t)];
            stats.nodeProcessed = values[kDagNodeProcessed / sizeof(uint32_t)];
            stats.clusterOverflow = values[kDagClusterOverflow / sizeof(uint32_t)];
            stats.seededInstances = values[kDagSeededInstances / sizeof(uint32_t)];
        }
        return stats;
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
