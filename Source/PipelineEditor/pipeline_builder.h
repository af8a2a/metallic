#pragma once

#include "pipeline_asset.h"
#include "frame_graph.h"
#include "pass_registry.h"
#include <Metal/Metal.hpp>
#include <unordered_map>
#include <vector>

struct RenderContext;
struct FrameContext;
struct PipelineRuntimeContext;

class PipelineBuilder {
public:
    PipelineBuilder(const RenderContext& ctx);

    // Build FrameGraph from pipeline asset (stores internally)
    bool build(const PipelineAsset& asset,
               const PipelineRuntimeContext& rtCtx,
               int width, int height);

    // Check if rebuild is needed (resolution changed or never built)
    bool needsRebuild(int width, int height) const;

    // Per-frame update: swap backbuffer, set frame context, reset transients
    void updateFrame(MTL::Texture* backbuffer, const FrameContext* frameCtx);

    // Forward to internal FrameGraph
    void compile();
    void execute(MTL::CommandBuffer* cmdBuf, MTL::Device* device, TracyMetalCtxHandle tracyCtx);

    // Access internal FrameGraph (for debug UI, graphviz export, etc.)
    FrameGraph& frameGraph() { return m_fg; }

    // Set frame context on all passes (call before execute each frame)
    void setFrameContext(const FrameContext* ctx);

    // Set runtime context on all passes
    void setRuntimeContext(const PipelineRuntimeContext* ctx);

    // Get last error message
    const std::string& lastError() const { return m_lastError; }

    // Get resource handle by name (after build)
    FGResource getResource(const std::string& name) const;

    // Get all created passes
    const std::vector<RenderPass*>& passes() const { return m_passes; }

private:
    MTL::PixelFormat parsePixelFormat(const std::string& format) const;
    FGTextureDesc parseTextureDesc(const ResourceDecl& decl, int width, int height) const;

    const RenderContext& m_ctx;
    std::string m_lastError;
    std::unordered_map<std::string, FGResource> m_resourceMap;
    std::vector<RenderPass*> m_passes;

    // Cached state
    FrameGraph m_fg;
    FGResource m_backbufferRes;
    int m_builtWidth = 0;
    int m_builtHeight = 0;
    bool m_built = false;
};
