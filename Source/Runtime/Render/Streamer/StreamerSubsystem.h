#pragma once

#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/SceneResourceManager.h"
#include "Runtime/Render/Streamer/StreamingUploads.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

namespace metallic::render {

// One owner for scene resources, geometry residency, page IO, CLAS pools and
// upload completion. Raster passes borrow a session; GPUScene owns identities.
class StreamerSubsystem final : public IRenderSubsystem {
public:
    struct Desc {};
    static constexpr RenderSubsystemId kSubsystemId = "render.streamer";

    Result initialize(const RenderSubsystemInitContext& context, std::string& log) override;
    Result beginFrame(const RenderSubsystemFrameContext& context, RenderChangeBits& changes, std::string& log) override;
    void endFrame(const RenderSubsystemFrameContext& context) override;
    void shutdown() override;

    Result acquireStream(const MeshletStreamRuntimeDesc& desc, bool debugReadback,
        std::shared_ptr<MeshletStreamRuntime>& outSession, std::string& log, PipelineCache* cache = nullptr);
    size_t streamCount() const { return streams_.size(); }
    StreamSceneReadiness sceneReadiness() const;
    void collectReleasedStreams();
    void flush(CommandBuffer& commands, const StreamUploadPhaseCallback& phase = {});
    Streamer* streamer() const { return uploads_.streamer(); }
    const RenderGraphStreamingStats& stats() const { return uploads_.stats(); }
    SceneResourceManager& manager() { return resources_; }
    const SceneResourceManager& manager() const { return resources_; }

private:
    Device* device_ = nullptr;
    StreamingUploads uploads_;
    SceneResourceManager resources_;
    // Sessions are view-specific: traversal feedback must not be consumed by
    // another view. The subsystem retains ownership until all borrowers retire.
    std::vector<std::shared_ptr<MeshletStreamRuntime>> streams_;
};

} // namespace metallic::render
