#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphNode.h"

namespace metallic::render {
struct RenderGraphSubmitDesc {
    Queue* graphicsQueue = nullptr;
    Queue* computeQueue = nullptr;
    Queue* copyQueue = nullptr;
};

class RenderGraphExecutor {
public:
    RenderGraphExecutor();
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
    Result execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources = nullptr);
    Result execute(const RenderGraphSubmitDesc& desc);
    Result waitForSubmittedWork(uint64_t timeoutNanoseconds = UINT64_MAX);
    bool syncProperties(const RenderGraph& graph);
    Result transitionOutput(
        CommandBuffer& commandBuffer,
        std::string_view fullName,
        ResourceState state);

    RenderGraphResource* outputResource(std::string_view fullName);
    const RenderGraphResource* outputResource(std::string_view fullName) const;
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
    const std::vector<uint32_t>& pixels() const;
    uint32_t width() const;
    uint32_t height() const;
    const std::string& lastLog() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
