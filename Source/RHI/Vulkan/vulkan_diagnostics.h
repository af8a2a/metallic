#pragma once

#ifdef _WIN32

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

struct VulkanPipelineStatisticsSnapshot {
    bool valid = false;
    uint64_t inputAssemblyVertices = 0;
    uint64_t inputAssemblyPrimitives = 0;
    uint64_t vertexShaderInvocations = 0;
    uint64_t clippingInvocations = 0;
    uint64_t clippingPrimitives = 0;
    uint64_t fragmentShaderInvocations = 0;
    uint64_t computeShaderInvocations = 0;
    uint64_t taskShaderInvocations = 0;
    uint64_t meshShaderInvocations = 0;
};

struct VulkanGpuScopeTiming {
    std::string label;
    double durationMs = 0.0;
    VulkanPipelineStatisticsSnapshot pipelineStats{};
};

struct VulkanGpuFrameDiagnostics {
    uint64_t frameIndex = 0;
    double totalGpuMs = 0.0;
    std::vector<VulkanGpuScopeTiming> scopes;
};

struct VulkanToolingInfo {
    bool debugUtils = false;
    bool validationMessenger = false;
    bool renderDocLayerAvailable = false;
    bool diagnosticCheckpoints = false;
    bool deviceFault = false;
    bool pipelineStatistics = false;
};

class VulkanGpuProfiler {
public:
    struct ScopeHandle {
        uint32_t pendingIndex = UINT32_MAX;
        bool active = false;
    };

    VulkanGpuProfiler() = default;
    ~VulkanGpuProfiler() = default;

    VulkanGpuProfiler(const VulkanGpuProfiler&) = delete;
    VulkanGpuProfiler& operator=(const VulkanGpuProfiler&) = delete;

    bool init(VkPhysicalDevice physicalDevice,
              VkDevice device,
              uint32_t framesInFlight,
              bool meshShaders);
    void destroy();

    void beginFrame(uint32_t frameIndex, uint64_t completedFrameIndex);
    void resetActiveFrameQueries(VkCommandBuffer commandBuffer);
    void beginScope(VkCommandBuffer commandBuffer,
                    const char* label,
                    ScopeHandle& outHandle,
                    bool allowPipelineStats = true);
    void endScope(VkCommandBuffer commandBuffer, ScopeHandle& handle);

    const VulkanGpuFrameDiagnostics& latestFrame() const { return m_latestFrame; }
    bool supportsPipelineStatistics() const { return m_pipelineStatisticsMask != 0; }

private:
    struct PendingScope {
        std::string label;
        uint32_t startQuery = UINT32_MAX;
        uint32_t endQuery = UINT32_MAX;
        uint32_t pipelineStatsQuery = UINT32_MAX;
    };

    struct FrameState {
        VkQueryPool timestampPool = VK_NULL_HANDLE;
        VkQueryPool pipelineStatsPool = VK_NULL_HANDLE;
        uint32_t nextTimestampQuery = 0;
        uint32_t nextPipelineStatsQuery = 0;
        uint64_t frameIndex = 0;
        bool hasSubmittedFrame = false;
        std::vector<PendingScope> scopes;
    };

    void collectFrame(FrameState& frame);
    VulkanPipelineStatisticsSnapshot buildPipelineStatsSnapshot(const uint64_t* values) const;
    void resetFrameQueries(VkCommandBuffer commandBuffer, FrameState& frame) const;

    VkDevice m_device = VK_NULL_HANDLE;
    float m_timestampPeriod = 0.0f;
    uint32_t m_framesInFlight = 0;
    uint32_t m_activeFrameIndex = 0;
    VkQueryPipelineStatisticFlags m_pipelineStatisticsMask = 0;
    uint32_t m_pipelineStatisticValueCount = 0;
    std::vector<FrameState> m_frames;
    VulkanGpuFrameDiagnostics m_latestFrame{};
};

#endif // _WIN32
