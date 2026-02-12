#pragma once

#include "pipeline_asset.h"
#include "frame_graph.h"
#include "pass_registry.h"
#include <Metal/Metal.hpp>
#include <unordered_map>

struct RenderContext;

// Runtime context for pipeline building (pipelines, textures, etc.)
struct PipelineRuntimeContext {
    MTL::Device* device = nullptr;

    // Pipeline states (keyed by shader name or type)
    std::unordered_map<std::string, MTL::RenderPipelineState*> renderPipelines;
    std::unordered_map<std::string, MTL::ComputePipelineState*> computePipelines;

    // Samplers
    std::unordered_map<std::string, MTL::SamplerState*> samplers;

    // Imported textures (atmosphere, fallbacks, etc.)
    std::unordered_map<std::string, MTL::Texture*> importedTextures;

    // Current frame's drawable
    MTL::Texture* backbuffer = nullptr;

    // Command buffer for ImGui pass
    MTL::CommandBuffer* commandBuffer = nullptr;
};

class PipelineBuilder {
public:
    PipelineBuilder(const RenderContext& ctx, const PipelineRuntimeContext& rtCtx);

    // Build FrameGraph from pipeline asset
    bool build(const PipelineAsset& asset, FrameGraph& fg, int width, int height);

    // Get last error message
    const std::string& lastError() const { return m_lastError; }

    // Get resource handle by name (after build)
    FGResource getResource(const std::string& name) const;

private:
    MTL::PixelFormat parsePixelFormat(const std::string& format) const;
    FGTextureDesc parseTextureDesc(const ResourceDecl& decl, int width, int height) const;

    const RenderContext& m_ctx;
    const PipelineRuntimeContext& m_rtCtx;
    std::string m_lastError;
    std::unordered_map<std::string, FGResource> m_resourceMap;
};
