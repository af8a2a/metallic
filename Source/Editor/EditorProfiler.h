#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace metallic {

class EditorProfiler {
public:
    using Clock = std::chrono::steady_clock;

    struct GraphicsCaptureControls {
        bool sdkCompiled = false;
        bool runtimeEnabled = false;
        bool canCapture = false;
        bool capturePending = false;
        const char* statusText = "";
        const char* capturePath = "";
    };

    struct Node {
        std::string name;
        uint32_t color = 0;
        double cpuMilliseconds = 0.0;
        double gpuMilliseconds = 0.0;
        bool gpuTimingAvailable = false;
        uint64_t renderGraphExecutionId = UINT64_MAX;
        uint32_t renderGraphNodeId = UINT32_MAX;
        uint32_t renderGraphSectionIndex = UINT32_MAX;
        render::QueueType queue = render::QueueType::Graphics;
        size_t parent = 0;
        std::vector<size_t> children;
        Clock::time_point beginTime;
    };

    struct Frame {
        std::vector<Node> nodes;
        uint64_t index = 0;
        bool profilingOverflow = false;
    };

    struct StreamingHistory {
        std::string passName;
        std::string assetPath;
        uint64_t generation = 0;
        std::vector<render::SceneStreamingProfile> samples;
    };

    // Latest completed GPU sample keeps CPU/GPU rows on the same execution.
    const Frame& displayFrame() const;
    const std::vector<Frame>& history() const { return history_; }
    const std::vector<StreamingHistory>& streamingHistory() const { return streamingHistory_; }

    class FrameScope {
    public:
        FrameScope() = default;
        explicit FrameScope(EditorProfiler* profiler);
        ~FrameScope();

        FrameScope(FrameScope&& other) noexcept;
        FrameScope& operator=(FrameScope&& other) noexcept;

        FrameScope(const FrameScope&) = delete;
        FrameScope& operator=(const FrameScope&) = delete;

    private:
        EditorProfiler* profiler_ = nullptr;
    };

    class Scope {
    public:
        Scope() = default;
        Scope(EditorProfiler* profiler, size_t nodeIndex);
        ~Scope();

        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&& other) noexcept;

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        EditorProfiler* profiler_ = nullptr;
        size_t nodeIndex_ = 0;
    };

    FrameScope beginFrame();
    Scope scope(std::string_view name, uint32_t color = 0);
    void addRenderGraphStats(const render::RenderGraphExecutionStats& stats);
    void updateRenderGraphGpuStats(const render::RenderGraphExecutionStats& stats);
    bool drawWindow(bool* open, const GraphicsCaptureControls& graphicsCapture);

private:
    size_t beginSection(std::string_view name, uint32_t color);
    void endSection(size_t nodeIndex);
    size_t addFinishedSection(size_t parent, std::string name, uint32_t color, double cpuMilliseconds);
    void endFrame();

    static uint32_t colorFromName(std::string_view name);

    bool frameActive_ = false;
    std::vector<Node> currentNodes_;
    std::vector<size_t> stack_;
    Frame latestFrame_;
    std::vector<Frame> history_;
    uint64_t frameIndex_ = 0;
    uint64_t graphGeneration_ = UINT64_MAX;
    bool currentOverflow_ = false;
    bool detailed_ = false;
    bool streamingShowBudget_ = false;
    int chartMetric_ = 1;
    std::vector<std::string> chartPath_;
    std::string selectedStream_;
    std::vector<render::SceneStreamingProfile> currentStreaming_;
    std::vector<StreamingHistory> streamingHistory_;
};

} // namespace metallic
