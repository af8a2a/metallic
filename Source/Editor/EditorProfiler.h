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

    struct Node {
        std::string name;
        uint32_t color = 0;
        double cpuMilliseconds = 0.0;
        double gpuMilliseconds = 0.0;
        bool gpuTimingAvailable = false;
        uint64_t renderGraphExecutionId = UINT64_MAX;
        uint32_t renderGraphNodeId = UINT32_MAX;
        size_t parent = 0;
        std::vector<size_t> children;
        Clock::time_point beginTime;
    };

    struct Frame {
        std::vector<Node> nodes;
    };

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
    void drawWindow(bool* open);

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
};

} // namespace metallic
