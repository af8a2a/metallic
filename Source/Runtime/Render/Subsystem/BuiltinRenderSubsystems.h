#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphStreamingSubsystem.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

namespace metallic::render {

class RenderUploadSubsystem final : public IRenderSubsystem {
public:
    struct Desc {};

    static constexpr RenderSubsystemId kSubsystemId = "render.upload";

    Result initialize(const RenderSubsystemInitContext& context, std::string& log) override;
    Result beginFrame(
        const RenderSubsystemFrameContext& context,
        RenderChangeBits& changes,
        std::string& log) override;
    void endFrame(const RenderSubsystemFrameContext& context) override;
    void shutdown() override;

    void flush(CommandBuffer& commandBuffer);
    Streamer* streamer() const { return streaming_.streamer(); }
    const RenderGraphStreamingStats& stats() const { return streaming_.stats(); }

private:
    RenderGraphStreamingSubsystem streaming_;
};

class SceneResourcesSubsystem final : public IRenderSubsystem {
public:
    struct Desc {};

    static constexpr RenderSubsystemId kSubsystemId = "render.scene-resources";

    void shutdown() override;

    SceneResourceManager& manager() { return manager_; }
    const SceneResourceManager& manager() const { return manager_; }

private:
    SceneResourceManager manager_;
};

bool registerBuiltInRenderSubsystems(RenderSubsystemHost& host, std::string& log);

} // namespace metallic::render
