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

    // Build FrameGraph from pipeline asset
    // Returns list of created passes for later context injection
    bool build(const PipelineAsset& asset, FrameGraph& fg,
               const PipelineRuntimeContext& rtCtx,
               int width, int height);

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
};
