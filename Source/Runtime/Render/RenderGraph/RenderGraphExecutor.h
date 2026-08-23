#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphNode.h"
#include "Runtime/Render/RenderGraph/RenderGraphStreamingSubsystem.h"
#include "Runtime/Scene/SceneLoad.h"

namespace metallic::render {
struct RenderGraphSubmitDesc {
    Queue* graphicsQueue = nullptr;
    Queue* computeQueue = nullptr;
    Queue* copyQueue = nullptr;
};

struct RenderGraphCompileOptions {
    std::vector<std::string> extraOutputs;
    bool enablePreviewOutputAccess = false;
};

struct RenderGraphNodeExecutionStat {
    uint32_t id = 0;
    std::string name;
    std::string type;
    double cpuMilliseconds = 0.0;
    double gpuMilliseconds = 0.0;
    bool gpuTimingAvailable = false;
};

struct RenderGraphExecutionStats {
    uint64_t executionId = 0;
    double cpuMilliseconds = 0.0;
    double gpuMilliseconds = 0.0;
    bool gpuTimingAvailable = false;
    std::vector<RenderGraphNodeExecutionStat> nodes;
};

class RenderGraphExecutor {
public:
    RenderGraphExecutor();
    RenderGraphExecutor(RenderSubsystemHost& subsystemHost, RenderWorld& world);
    ~RenderGraphExecutor();

    RenderGraphExecutor(RenderGraphExecutor&&) noexcept;
    RenderGraphExecutor& operator=(RenderGraphExecutor&&) noexcept;

    RenderGraphExecutor(const RenderGraphExecutor&) = delete;
    RenderGraphExecutor& operator=(const RenderGraphExecutor&) = delete;

    Result compile(
        Device& device,
        const RenderGraph& graph,
        uint32_t width,
        uint32_t height,
        std::string& log);
    Result compile(
        Device& device,
        const RenderGraph& graph,
        uint32_t width,
        uint32_t height,
        const RenderGraphCompileOptions& options,
        std::string& log);
    Result execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources = nullptr);
    Result execute(const RenderGraphSubmitDesc& desc);
    Result waitForSubmittedWork(uint64_t timeoutNanoseconds = UINT64_MAX);
    void bindRuntimeScene(const scene::Scene* scene);
    void bindRenderWorld(RenderWorld* world);
    RenderSubsystemHost* subsystemHost();
    const RenderSubsystemHost* subsystemHost() const;
    Result beginSceneResourcePreparation(
        Device& device,
        const RenderGraphProperties& properties,
        const scene::Scene& scene,
        std::string& log);
    Result pumpSceneResourcePreparation(
        const scene::Scene& scene,
        double budgetMilliseconds,
        bool& complete,
        scene::SceneLoadProgress& progress,
        std::string& log);
    void cancelSceneResourcePreparation();
    void acceptSceneResourcePreparation();
    bool syncProperties(const RenderGraph& graph);
    bool syncRuntimeProperties(const RenderGraph& graph);
    Result transitionOutput(
        CommandBuffer& commandBuffer,
        std::string_view fullName,
        ResourceState state);

    RenderGraphResource* outputResource(std::string_view fullName);
    const RenderGraphResource* outputResource(std::string_view fullName) const;
    const RenderGraphExecutionStats& executionStats() const;
    Result collectCompletedGpuExecutionStats(std::vector<RenderGraphExecutionStats>& outStats);
    const RenderGraphStreamingStats& streamingStats() const;
    bool compiled() const;
    uint32_t width() const;
    uint32_t height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RenderGraphPreviewRenderer {
public:
    RenderGraphPreviewRenderer();
    ~RenderGraphPreviewRenderer();

    RenderGraphPreviewRenderer(RenderGraphPreviewRenderer&&) noexcept;
    RenderGraphPreviewRenderer& operator=(RenderGraphPreviewRenderer&&) noexcept;

    RenderGraphPreviewRenderer(const RenderGraphPreviewRenderer&) = delete;
    RenderGraphPreviewRenderer& operator=(const RenderGraphPreviewRenderer&) = delete;

    Result initialize(bool enableValidation = false, bool enableRayQuery = false);
    Result render(RenderGraph& graph, uint32_t width, uint32_t height);
    Result render(RenderGraph& graph, uint32_t width, uint32_t height, std::string_view outputName);
    void setEnvironment(EnvironmentSettings environment);
    RenderSubsystemHost* subsystemHost();
    const RenderSubsystemHost* subsystemHost() const;
    const std::vector<uint32_t>& pixels() const;
    uint32_t width() const;
    uint32_t height() const;
    const std::string& lastLog() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
