#pragma once

#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/SceneResourceManager.h"
#include "Runtime/Render/Streamer/StreamingUploads.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <optional>

namespace metallic::render {

// One owner for scene resources, geometry residency, page IO, CLAS pools and
// upload completion. Raster passes borrow a session; GPUScene owns identities.
class StreamerSubsystem final : public IRenderSubsystem {
public:
    struct Desc {};
    static constexpr RenderSubsystemId kSubsystemId = "render.streamer";

    Result<> initialize(const RenderSubsystemInitContext& context, std::string& log) override;
    Result<> beginFrame(
        const RenderSubsystemFrameContext& context,
        RenderChangeBits& changes,
        std::string& log) override;
    Result<> recordPostGraph(const RenderSubsystemFrameContext& context, std::string& log) override;
    void endFrame(const RenderSubsystemFrameContext& context) override;
    void shutdown() override;

    Result<> prepareScene(const SceneStreamingRequirements& requirements,
        const RenderGraphProperties& properties, const scene::Scene* scene,
        std::shared_ptr<PreparedSceneResources>& prepared, std::string& log, bool debugReadback = false);
    Result<> recordSceneBegin(PreparedSceneResources& prepared, const SceneStreamingRequirements& requirements,
        RenderGraphExecutionContext& context, const MeshletStreamFrameDesc& view, std::string& log);
    Result<> recordSceneTraversal(PreparedSceneResources& prepared, RenderGraphExecutionContext& context,
        const MeshletStreamFrameDesc& view, const MeshletStreamRuntime::TraversalCheckpoint& checkpoint);
    Result<> recordSceneEnd(PreparedSceneResources& prepared, RenderGraphExecutionContext& context);

    [[nodiscard]] Result<std::shared_ptr<MeshletStreamRuntime>> acquireStream(
        const MeshletStreamRuntimeDesc& desc,
        bool debugReadback,
        std::string& log,
        PipelineCache* cache = nullptr);
    size_t streamCount() const { return streams_.size(); }
    StreamSceneReadiness sceneReadiness() const;
    void collectReleasedStreams();
    void prepareBeforePacing(CpuProfileRecorder* profiler = nullptr);
    [[nodiscard]] Result<> flush(CommandBuffer& commands, const StreamUploadPhaseCallback& phase = {});
    Streamer* streamer() const { return uploads_.streamer(); }
    const RenderGraphStreamingStats& stats() const { return uploads_.stats(); }
    SceneResourceManager& manager() { return resources_; }
    const SceneResourceManager& manager() const { return resources_; }

private:
    Result<> completeInitialLoads(std::string& log);
    struct TextureFrame {
        std::shared_ptr<ScenePathTraceResources> resources;
        Buffer* feedback = nullptr;
        uint64_t frameIndex = 0;
    };
    std::unordered_map<const ScenePathTraceResources*, TextureFrame> textureFrames_;
    Device* device_ = nullptr;
    StreamingUploads uploads_;
    SceneResourceManager resources_;
    // Sessions are view-specific: traversal feedback must not be consumed by
    // another view. The subsystem retains ownership until all borrowers retire.
    std::vector<std::shared_ptr<MeshletStreamRuntime>> streams_;
    // Do not populate replacement roots while old passes still own their stream.
    struct InitialLoad {
        std::weak_ptr<MeshletStreamRuntime> runtime;
        std::optional<Error> failure;
        std::string failureLog;
        bool metadataOnly = false;
    };
    std::vector<InitialLoad> initialLoads_;
};

} // namespace metallic::render
