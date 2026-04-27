#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

class ClusterStreamingService {
public:
    static constexpr uint32_t kStreamingAgeHistogramBucketCount = 16u;
    static constexpr uint32_t kStreamingLodBucketCount = 16u;

    enum class BudgetPreset : uint32_t {
        Auto = 0,
        Low,
        Medium,
        High,
        Custom,
    };

    struct MemoryBudgetInfo {
        bool available = false;
        uint32_t heapCount = 0;
        uint64_t totalBudgetBytes = 0;
        uint64_t totalUsageBytes = 0;
        uint64_t totalHeadroomBytes = 0;
        uint64_t deviceLocalBudgetBytes = 0;
        uint64_t deviceLocalUsageBytes = 0;
        uint64_t deviceLocalHeadroomBytes = 0;
        uint64_t targetStorageBytes = kDefaultStorageCapacityBytes;
    };

    struct DebugStats {
        bool resourcesReady = false;
        uint32_t activeResidencyGroupCount = 0;
        uint32_t pendingResidencyGroupCount = 0;
        uint32_t pendingUnloadGroupCount = 0;
        uint32_t confirmedUnloadGroupCount = 0;
        uint32_t residentHeapUsed = 0;
        uint32_t residentHeapCapacity = 0;
        uint32_t streamingTaskCapacity = 0;
        uint32_t freeStreamingTaskCount = 0;
        uint32_t preparedStreamingTaskCount = 0;
        uint32_t transferSubmittedTaskCount = 0;
        uint32_t updateQueuedTaskCount = 0;
        uint32_t selectedTransferTaskIndex = std::numeric_limits<uint32_t>::max();
        uint64_t selectedTransferBytes = 0;
        uint32_t selectedUpdateTaskIndex = std::numeric_limits<uint32_t>::max();
        uint32_t selectedUpdatePatchCount = 0;
        uint64_t selectedUpdateTransferWaitValue = 0;
    };

    struct StreamingStats {
        uint32_t loadRequestsThisFrame = 0;
        uint32_t unloadRequestsThisFrame = 0;
        uint32_t loadsExecutedThisFrame = 0;
        uint32_t loadsDeferredThisFrame = 0;
        uint32_t unloadsExecutedThisFrame = 0;
        uint32_t failedAllocations = 0;
        float smoothedFailedAllocations = 0.0f;
        uint32_t residentGroupCount = 0;
        uint32_t alwaysResidentGroupCount = 0;
        uint32_t dynamicResidentGroupCount = 0;
        uint32_t configuredAgeThreshold = 0;
        uint32_t effectiveAgeThreshold = 0;
        uint32_t adaptiveBudgetAdjustmentCount = 0;
        uint64_t storagePoolUsedBytes = 0;
        uint64_t storagePoolCapacityBytes = kDefaultStorageCapacityBytes;
        float smoothedStorageUtilization = 0.0f;
        uint64_t transferBytesThisFrame = 0;
        float transferUtilization = 0.0f;
        bool cpuUnloadFallbackActive = false;
        uint32_t cpuUnloadFallbackFrameIndex = 0;
        uint32_t cpuUnloadFallbackGroupCount = 0;
        bool graphicsTransferFallbackActive = false;
        bool gpuStatsValid = false;
        uint32_t gpuStatsFrameIndex = 0;
        uint32_t gpuUnloadRequestCount = 0;
        float gpuAverageUnloadAge = 0.0f;
        uint32_t gpuAppliedPatchCount = 0;
        uint64_t gpuCopiedBytes = 0;
        uint32_t gpuErrorUpdateCount = 0;
        uint32_t gpuErrorAgeFilterCount = 0;
        uint32_t gpuErrorAllocationCount = 0;
        uint32_t gpuErrorPageTableCount = 0;
        bool gpuAgeFilterDispatchMissing = false;
        uint32_t gpuAgeFilterDispatchMissingFrameIndex = 0;
        uint32_t ageHistogramBucketWidth = 1;
        uint32_t ageHistogramMaxAge = 0;
        std::array<uint32_t, kStreamingAgeHistogramBucketCount> ageHistogram = {};
        std::array<uint32_t, kStreamingLodBucketCount> totalGroupsPerLod = {};
        std::array<uint32_t, kStreamingLodBucketCount> residentGroupsPerLod = {};
    };

    static const char* budgetPresetLabel(BudgetPreset preset) {
        switch (preset) {
        case BudgetPreset::Auto: return "Auto";
        case BudgetPreset::Low: return "Low";
        case BudgetPreset::Medium: return "Medium";
        case BudgetPreset::High: return "High";
        case BudgetPreset::Custom: return "Custom";
        }
        return "Custom";
    }

    void applyAutoMemoryBudget(const MemoryBudgetInfo& info) {
        m_memoryBudget = info;
        if (m_budgetPreset == BudgetPreset::Auto) {
            const uint64_t headroomTarget = info.available
                ? std::clamp(info.deviceLocalHeadroomBytes / 4u,
                             kMinStorageCapacityBytes,
                             kMaxStorageCapacityBytes)
                : kDefaultStorageCapacityBytes;
            m_storageCapacityBytes = headroomTarget;
            m_memoryBudget.targetStorageBytes = m_storageCapacityBytes;
            m_streamingStats.storagePoolCapacityBytes = m_storageCapacityBytes;
        }
    }

    void resetForPipelineReload() {}

    const MemoryBudgetInfo& memoryBudgetInfo() const { return m_memoryBudget; }
    const DebugStats& debugStats() const { return m_debugStats; }
    const StreamingStats& streamingStats() const { return m_streamingStats; }

    bool streamingEnabled() const { return false; }
    uint64_t streamingStorageCapacityBytes() const { return m_storageCapacityBytes; }
    uint32_t streamingBudgetGroups() const { return m_streamingBudgetGroups; }
    BudgetPreset budgetPreset() const { return m_budgetPreset; }

    void setBudgetPreset(BudgetPreset preset) {
        m_budgetPreset = preset;
        switch (preset) {
        case BudgetPreset::Auto:
            applyAutoMemoryBudget(m_memoryBudget);
            break;
        case BudgetPreset::Low:
            setStorageCapacityFromPreset(128ull * kMiB);
            break;
        case BudgetPreset::Medium:
            setStorageCapacityFromPreset(256ull * kMiB);
            break;
        case BudgetPreset::High:
            setStorageCapacityFromPreset(512ull * kMiB);
            break;
        case BudgetPreset::Custom:
            break;
        }
    }

    void setStreamingStorageCapacityBytes(uint64_t bytes) {
        m_budgetPreset = BudgetPreset::Custom;
        m_storageCapacityBytes =
            std::clamp(bytes, kMinStorageCapacityBytes, kMaxStorageCapacityBytes);
        m_memoryBudget.targetStorageBytes = m_storageCapacityBytes;
        m_streamingStats.storagePoolCapacityBytes = m_storageCapacityBytes;
    }

    void setStreamingBudgetGroups(uint32_t groups) {
        m_budgetPreset = BudgetPreset::Custom;
        m_streamingBudgetGroups = groups;
    }

    bool gpuStatsReadbackEnabled() const { return m_gpuStatsReadbackEnabled; }
    void setGpuStatsReadbackEnabled(bool enabled) { m_gpuStatsReadbackEnabled = enabled; }

    bool adaptiveBudgetEnabled() const { return m_adaptiveBudgetEnabled; }
    void setAdaptiveBudgetEnabled(bool enabled) { m_adaptiveBudgetEnabled = enabled; }

    uint64_t effectiveStreamingTransferCapacityBytes() const { return 0; }
    uint64_t consumePendingTransferWaitValue() { return 0; }

private:
    static constexpr uint64_t kMiB = 1024ull * 1024ull;
    static constexpr uint64_t kMinStorageCapacityBytes = 64ull * kMiB;
    static constexpr uint64_t kDefaultStorageCapacityBytes = 256ull * kMiB;
    static constexpr uint64_t kMaxStorageCapacityBytes = 2048ull * kMiB;

    void setStorageCapacityFromPreset(uint64_t bytes) {
        m_storageCapacityBytes = bytes;
        m_memoryBudget.targetStorageBytes = m_storageCapacityBytes;
        m_streamingStats.storagePoolCapacityBytes = m_storageCapacityBytes;
    }

    MemoryBudgetInfo m_memoryBudget{};
    DebugStats m_debugStats{};
    StreamingStats m_streamingStats{};
    BudgetPreset m_budgetPreset = BudgetPreset::Auto;
    uint64_t m_storageCapacityBytes = kDefaultStorageCapacityBytes;
    uint32_t m_streamingBudgetGroups = 0;
    bool m_gpuStatsReadbackEnabled = false;
    bool m_adaptiveBudgetEnabled = false;
};
