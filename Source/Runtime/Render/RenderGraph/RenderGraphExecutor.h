#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphNode.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/Profiling/SchedulingDiagnostics.h"
#include "Runtime/Render/Streamer/StreamingUploads.h"
#include "Runtime/Scene/SceneLoad.h"
#include "Runtime/Scene/SceneLighting.h"

namespace metallic::render {
struct RenderGraphSubmitDesc {
    Queue* graphicsQueue = nullptr;
    Queue* computeQueue = nullptr;
    Queue* copyQueue = nullptr;
    HistoryResourceManager* historyResources = nullptr;
    // Dependencies supplied by the caller, retained until this graph completes.
    std::span<const GpuCompletionPoint> waitCompletions;
    uint64_t slotWaitTimeoutNanoseconds = UINT64_MAX;
    // 0 uses up to eight TaskSystem workers; 1 records inline. A queue change
    // starts a batch. Pipelining requires every pass's explicit safety contract.
    uint32_t recordingWorkerLimit = 0;
    uint32_t recordingBatchWorkload = 8;
    // Pure preparation uses the same worker limit, with independent CPU batches.
    uint32_t preparationBatchWorkload = 1;
    FrameSubmissionMode submissionMode = FrameSubmissionMode::Pipelined;
    // CPU attribution only; can also be enabled process-wide by
    // METALLIC_RENDER_SCHEDULING_DIAGNOSTICS=1. No diagnostic clocks run while disabled.
    bool schedulingDiagnostics = false;
};

struct RenderGraphCompileOptions {
    std::vector<std::string> extraOutputs;
    bool enablePreviewOutputAccess = false;
    DisplayOutputParameters displayOutput;
};

struct RenderGraphNodeExecutionStat {
    uint32_t id = 0;
    std::string name;
    std::string type;
    double cpuMilliseconds = 0.0;
    double gpuMilliseconds = 0.0;
    bool gpuTimingAvailable = false;
    QueueType queue = QueueType::Graphics;
    std::vector<RenderGraphProfileSection> sections;
};

struct RenderGraphExecutionStats {
    uint32_t asyncComputeBranches = 0;
    uint64_t executionId = 0;
    uint64_t graphGeneration = 0;
    double cpuMilliseconds = 0.0;
    double gpuMilliseconds = 0.0;
    bool gpuTimingAvailable = false;
    std::vector<RenderGraphNodeExecutionStat> nodes;
    std::vector<SceneStreamingProfile> streaming;
    bool profilingOverflow = false;
    std::vector<RenderGraphProfileSection> preparation;
    std::vector<std::string> overlapBlockingPasses;
    uint32_t drainReasonMask = 0; // 1: pass contract, 2: scene revision; legacy bit 4 no longer drains
    uint32_t externalCompletionCount = 0; // Unfinished external consumers carried as GPU dependencies
    uint32_t recordingBatchCount = 0;
    uint32_t parallelRecordedPassCount = 0;
    uint32_t recordingTaskCount = 0;
    uint32_t preparationTaskCount = 0;
    bool pipelinedSubmission = false;
    uint32_t submittedBatchCount = 0;
    uint32_t batchesSubmittedWhileRecording = 0;
    std::vector<std::string> submissionBlockingPasses;
    profiling::SchedulingMetrics scheduling;
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

    Result<> compile(
        Device& device,
        const RenderGraph& graph,
        uint32_t width,
        uint32_t height,
        std::string& log);
    Result<> compile(
        Device& device,
        const RenderGraph& graph,
        uint32_t width,
        uint32_t height,
        const RenderGraphCompileOptions& options,
        std::string& log);
    // Call between frames, with no recorded-but-unsubmitted command buffers that
    // reference this graph. On failure the current passes and resources stay valid.
    Result<> reloadShaders(std::string& log);
    Result<> execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources = nullptr);
    Result<> execute(const RenderGraphSubmitDesc& desc);
    GpuCompletionPoint lastSubmittedCompletion() const;
    Result<> waitForSubmittedWork(uint64_t timeoutNanoseconds = UINT64_MAX);
    void bindRuntimeScene(const scene::Scene* scene);
    void bindRenderWorld(RenderWorld* world);
    // One view per executor; separate viewports/executors retain independent history.
    void bindRenderView(RenderView* view);
    RenderView* renderView();
    RenderSubsystemHost* subsystemHost();
    const RenderSubsystemHost* subsystemHost() const;
    Result<> beginSceneResourcePreparation(
        Device& device,
        const RenderGraphProperties& properties,
        const scene::Scene& scene,
        std::string& log);
    Result<> pumpSceneResourcePreparation(
        const scene::Scene& scene,
        double budgetMilliseconds,
        bool& complete,
        scene::SceneLoadProgress& progress,
        std::string& log);
    void cancelSceneResourcePreparation();
    void acceptSceneResourcePreparation();
    bool syncProperties(const RenderGraph& graph);
    bool syncRuntimeProperties(const RenderGraph& graph);
    // Attach before compilation; observer and device outlive submitted work.
    void setDebugObserver(IRenderDebugObserver* observer);
    Result<> transitionOutput(
        CommandBuffer& commandBuffer,
        std::string_view fullName,
        ResourceState state);

    RenderGraphResource* outputResource(std::string_view fullName);
    const RenderGraphResource* outputResource(std::string_view fullName) const;
    const RenderGraphExecutionStats& executionStats() const;
    // Owner-thread control/query. Disabling capture preserves the last snapshot;
    // returned immutable values can outlive the executor and its GPU resources.
    void setExecutionCaptureEnabled(bool enabled);
    std::shared_ptr<const RenderGraphExecutionSnapshot> executionSnapshot() const;
    Result<> collectCompletedGpuExecutionStats(std::vector<RenderGraphExecutionStats>& outStats);
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

    Result<> initialize(bool enableValidation = false, bool enableRayQuery = false, bool enableAftermath = true);
    Result<> render(RenderGraph& graph, uint32_t width, uint32_t height);
    // Disabling readback still completes the frame, but leaves pixels() empty.
    Result<> render(RenderGraph& graph, uint32_t width, uint32_t height, std::string_view outputName, bool readback = true);
    // Bind before rendering; the scene must outlive the preview renderer.
    // This keeps scene-owned lighting and world overrides in the same scene.
    void bindRuntimeScene(const scene::Scene* scene);
    void setDebugObserver(IRenderDebugObserver* observer);
    // Update a live camera without recompiling the graph or clearing HZB history.
    // The view must outlive the preview renderer.
    void bindRenderView(RenderView* view);
    // Same policy as RenderGraphSubmitDesc; useful for reproducible CPU comparisons.
    void setRecordingWorkerLimit(uint32_t limit);
    void setEnvironment(EnvironmentSettings environment);
    bool setLighting(scene::LightingSettings lighting);
    RenderSubsystemHost* subsystemHost();
    const RenderSubsystemHost* subsystemHost() const;
    const std::vector<uint32_t>& pixels() const;
    uint32_t width() const;
    uint32_t height() const;
    const std::string& lastLog() const;
    const RenderGraphExecutionStats& executionStats() const;
    void setExecutionCaptureEnabled(bool enabled);
    std::shared_ptr<const RenderGraphExecutionSnapshot> executionSnapshot() const;
    Result<> collectCompletedGpuExecutionStats(std::vector<RenderGraphExecutionStats>& outStats);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
