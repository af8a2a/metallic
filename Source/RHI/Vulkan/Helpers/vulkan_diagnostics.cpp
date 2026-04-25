#include "vulkan_diagnostics.h"

#ifdef _WIN32
#include <algorithm>
#include <spdlog/spdlog.h>
#include <vector>

namespace {

constexpr uint32_t kMaxTimestampQueriesPerFrame = 512;
constexpr uint32_t kMaxPipelineStatisticQueriesPerFrame = 128;

} // namespace

bool VulkanGpuProfiler::init(VkPhysicalDevice physicalDevice,
                             VkDevice device,
                             uint32_t framesInFlight,
                             bool meshShaders) {
    destroy();

    if (physicalDevice == VK_NULL_HANDLE || device == VK_NULL_HANDLE || framesInFlight == 0) {
        return false;
    }

    m_device = device;
    m_framesInFlight = framesInFlight;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    m_timestampPeriod = properties.limits.timestampPeriod;

    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(physicalDevice, &features);

    if (features.pipelineStatisticsQuery == VK_TRUE) {
        m_pipelineStatisticsMask =
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
#ifdef VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT
        if (meshShaders) {
            m_pipelineStatisticsMask |= VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT;
        }
#endif
#ifdef VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT
        if (meshShaders) {
            m_pipelineStatisticsMask |= VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;
        }
#endif
    }

    auto bitCount = [](uint32_t value) {
        uint32_t count = 0;
        while (value != 0) {
            count += (value & 1u);
            value >>= 1u;
        }
        return count;
    };
    m_pipelineStatisticValueCount = bitCount(m_pipelineStatisticsMask);

    m_frames.resize(framesInFlight);
    for (FrameState& frame : m_frames) {
        VkQueryPoolCreateInfo timestampInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        timestampInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        timestampInfo.queryCount = kMaxTimestampQueriesPerFrame;
        if (vkCreateQueryPool(device, &timestampInfo, nullptr, &frame.timestampPool) != VK_SUCCESS) {
            spdlog::warn("VulkanGpuProfiler: failed to create timestamp query pool; GPU timing disabled");
            destroy();
            return false;
        }

        if (m_pipelineStatisticsMask != 0) {
            VkQueryPoolCreateInfo pipelineStatsInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            pipelineStatsInfo.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            pipelineStatsInfo.queryCount = kMaxPipelineStatisticQueriesPerFrame;
            pipelineStatsInfo.pipelineStatistics = m_pipelineStatisticsMask;
            if (vkCreateQueryPool(device, &pipelineStatsInfo, nullptr, &frame.pipelineStatsPool) != VK_SUCCESS) {
                spdlog::warn("VulkanGpuProfiler: failed to create pipeline statistics pool; statistics disabled");
                if (frame.pipelineStatsPool != VK_NULL_HANDLE) {
                    vkDestroyQueryPool(device, frame.pipelineStatsPool, nullptr);
                    frame.pipelineStatsPool = VK_NULL_HANDLE;
                }
                m_pipelineStatisticsMask = 0;
                m_pipelineStatisticValueCount = 0;
            }
        }
    }

    return true;
}

void VulkanGpuProfiler::destroy() {
    for (FrameState& frame : m_frames) {
        if (frame.timestampPool != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_device, frame.timestampPool, nullptr);
        }
        if (frame.pipelineStatsPool != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_device, frame.pipelineStatsPool, nullptr);
        }
        frame.timestampPool = VK_NULL_HANDLE;
        frame.pipelineStatsPool = VK_NULL_HANDLE;
        frame.scopes.clear();
        frame.nextTimestampQuery = 0;
        frame.nextPipelineStatsQuery = 0;
        frame.frameIndex = 0;
        frame.hasSubmittedFrame = false;
    }
    m_frames.clear();
    m_device = VK_NULL_HANDLE;
    m_timestampPeriod = 0.0f;
    m_framesInFlight = 0;
    m_activeFrameIndex = 0;
    m_pipelineStatisticsMask = 0;
    m_pipelineStatisticValueCount = 0;
    m_latestFrame = {};
}

void VulkanGpuProfiler::beginFrame(uint32_t frameIndex, uint64_t completedFrameIndex) {
    if (m_device == VK_NULL_HANDLE || frameIndex >= m_frames.size()) {
        return;
    }

    FrameState& frame = m_frames[frameIndex];
    if (frame.hasSubmittedFrame) {
        collectFrame(frame);
    }

    frame.frameIndex = completedFrameIndex;
    frame.hasSubmittedFrame = true;
    frame.nextTimestampQuery = 0;
    frame.nextPipelineStatsQuery = 0;
    frame.scopes.clear();
    m_activeFrameIndex = frameIndex;
}

void VulkanGpuProfiler::resetActiveFrameQueries(VkCommandBuffer commandBuffer) {
    if (m_device == VK_NULL_HANDLE ||
        commandBuffer == VK_NULL_HANDLE ||
        m_activeFrameIndex >= m_frames.size()) {
        return;
    }

    resetFrameQueries(commandBuffer, m_frames[m_activeFrameIndex]);
}

void VulkanGpuProfiler::beginScope(VkCommandBuffer commandBuffer,
                                   const char* label,
                                   ScopeHandle& outHandle,
                                   bool allowPipelineStats) {
    outHandle = {};
    if (m_device == VK_NULL_HANDLE || commandBuffer == VK_NULL_HANDLE || m_frames.empty()) {
        return;
    }

    FrameState& frame = m_frames[m_activeFrameIndex];
    if ((frame.nextTimestampQuery + 2u) > kMaxTimestampQueriesPerFrame) {
        return;
    }

    PendingScope scope{};
    scope.label = (label && label[0] != '\0') ? label : "Unnamed Pass";
    scope.startQuery = frame.nextTimestampQuery++;
    scope.endQuery = frame.nextTimestampQuery++;

    vkCmdWriteTimestamp2(commandBuffer,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         frame.timestampPool,
                         scope.startQuery);

    if (allowPipelineStats &&
        frame.pipelineStatsPool != VK_NULL_HANDLE &&
        frame.nextPipelineStatsQuery < kMaxPipelineStatisticQueriesPerFrame) {
        scope.pipelineStatsQuery = frame.nextPipelineStatsQuery++;
        vkCmdBeginQuery(commandBuffer, frame.pipelineStatsPool, scope.pipelineStatsQuery, 0);
    }

    frame.scopes.push_back(std::move(scope));
    outHandle.pendingIndex = static_cast<uint32_t>(frame.scopes.size() - 1);
    outHandle.active = true;
}

void VulkanGpuProfiler::endScope(VkCommandBuffer commandBuffer, ScopeHandle& handle) {
    if (!handle.active || handle.pendingIndex == UINT32_MAX || m_frames.empty() ||
        commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    FrameState& frame = m_frames[m_activeFrameIndex];
    if (handle.pendingIndex >= frame.scopes.size()) {
        handle = {};
        return;
    }

    PendingScope& scope = frame.scopes[handle.pendingIndex];
    if (scope.pipelineStatsQuery != UINT32_MAX && frame.pipelineStatsPool != VK_NULL_HANDLE) {
        vkCmdEndQuery(commandBuffer, frame.pipelineStatsPool, scope.pipelineStatsQuery);
    }

    vkCmdWriteTimestamp2(commandBuffer,
                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         frame.timestampPool,
                         scope.endQuery);
    handle = {};
}

void VulkanGpuProfiler::collectFrame(FrameState& frame) {
    if (frame.timestampPool == VK_NULL_HANDLE || frame.nextTimestampQuery == 0) {
        return;
    }

    std::vector<uint64_t> timestamps(frame.nextTimestampQuery, 0);
    const VkResult timestampResult = vkGetQueryPoolResults(m_device,
                                                           frame.timestampPool,
                                                           0,
                                                           frame.nextTimestampQuery,
                                                           timestamps.size() * sizeof(uint64_t),
                                                           timestamps.data(),
                                                           sizeof(uint64_t),
                                                           VK_QUERY_RESULT_64_BIT);
    if (timestampResult != VK_SUCCESS) {
        spdlog::warn("VulkanGpuProfiler: failed to read timestamp queries ({})",
                     static_cast<int>(timestampResult));
        return;
    }

    std::vector<uint64_t> pipelineStats;
    if (frame.pipelineStatsPool != VK_NULL_HANDLE &&
        frame.nextPipelineStatsQuery > 0 &&
        m_pipelineStatisticValueCount > 0) {
        pipelineStats.resize(static_cast<size_t>(frame.nextPipelineStatsQuery) * m_pipelineStatisticValueCount, 0);
        const VkResult statsResult = vkGetQueryPoolResults(m_device,
                                                           frame.pipelineStatsPool,
                                                           0,
                                                           frame.nextPipelineStatsQuery,
                                                           pipelineStats.size() * sizeof(uint64_t),
                                                           pipelineStats.data(),
                                                           sizeof(uint64_t) * m_pipelineStatisticValueCount,
                                                           VK_QUERY_RESULT_64_BIT);
        if (statsResult != VK_SUCCESS) {
            spdlog::warn("VulkanGpuProfiler: failed to read pipeline statistics ({})",
                         static_cast<int>(statsResult));
            pipelineStats.clear();
        }
    }

    VulkanGpuFrameDiagnostics completed{};
    completed.frameIndex = frame.frameIndex;
    completed.scopes.reserve(frame.scopes.size());

    for (const PendingScope& scope : frame.scopes) {
        if (scope.startQuery == UINT32_MAX || scope.endQuery == UINT32_MAX ||
            scope.endQuery >= timestamps.size() || scope.startQuery >= timestamps.size()) {
            continue;
        }

        VulkanGpuScopeTiming timing{};
        timing.label = scope.label;
        if (timestamps[scope.endQuery] >= timestamps[scope.startQuery]) {
            const uint64_t delta = timestamps[scope.endQuery] - timestamps[scope.startQuery];
            timing.durationMs = (static_cast<double>(delta) * static_cast<double>(m_timestampPeriod)) / 1'000'000.0;
            completed.totalGpuMs += timing.durationMs;
        }

        if (!pipelineStats.empty() &&
            scope.pipelineStatsQuery != UINT32_MAX &&
            m_pipelineStatisticValueCount > 0) {
            const size_t valueOffset =
                static_cast<size_t>(scope.pipelineStatsQuery) * m_pipelineStatisticValueCount;
            timing.pipelineStats = buildPipelineStatsSnapshot(pipelineStats.data() + valueOffset);
        }

        completed.scopes.push_back(std::move(timing));
    }

    std::sort(completed.scopes.begin(),
              completed.scopes.end(),
              [](const VulkanGpuScopeTiming& lhs, const VulkanGpuScopeTiming& rhs) {
                  return lhs.durationMs > rhs.durationMs;
              });

    m_latestFrame = std::move(completed);
}

VulkanPipelineStatisticsSnapshot VulkanGpuProfiler::buildPipelineStatsSnapshot(const uint64_t* values) const {
    VulkanPipelineStatisticsSnapshot snapshot{};
    if (!values || m_pipelineStatisticsMask == 0) {
        return snapshot;
    }

    uint32_t cursor = 0;
    auto consume = [&](VkQueryPipelineStatisticFlags flag, uint64_t& outValue) {
        if ((m_pipelineStatisticsMask & flag) != 0) {
            outValue = values[cursor++];
        }
    };

    consume(VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT, snapshot.inputAssemblyVertices);
    consume(VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT, snapshot.inputAssemblyPrimitives);
    consume(VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT, snapshot.vertexShaderInvocations);
    consume(VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT, snapshot.clippingInvocations);
    consume(VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT, snapshot.clippingPrimitives);
    consume(VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT, snapshot.fragmentShaderInvocations);
    consume(VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT, snapshot.computeShaderInvocations);
#ifdef VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT
    consume(VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT, snapshot.taskShaderInvocations);
#endif
#ifdef VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT
    consume(VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT, snapshot.meshShaderInvocations);
#endif
    snapshot.valid = true;
    return snapshot;
}

void VulkanGpuProfiler::resetFrameQueries(VkCommandBuffer commandBuffer, FrameState& frame) const {
    if (commandBuffer == VK_NULL_HANDLE) {
        return;
    }
    if (frame.timestampPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(commandBuffer, frame.timestampPool, 0, kMaxTimestampQueriesPerFrame);
    }
    if (frame.pipelineStatsPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(commandBuffer,
                            frame.pipelineStatsPool,
                            0,
                            kMaxPipelineStatisticQueriesPerFrame);
    }
}

#endif // _WIN32
