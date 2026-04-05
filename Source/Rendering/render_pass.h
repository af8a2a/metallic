#pragma once

#include "frame_graph.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "material_loader.h"
#include "scene_graph.h"
#include "raytraced_shadows.h"
#include <tracy/Tracy.hpp>
#include <microprofile.h>
#include <unordered_map>

// Forward declarations
struct PassConfig;
struct FrameContext;
struct PipelineRuntimeContext;

// Static context (scene data, depth state, etc.)
struct RenderContext {
    const LoadedMesh& sceneMesh;
    const MeshletData& meshletData;
    const LoadedMaterials& materials;
    const SceneGraph& sceneGraph;
    const RaytracedShadowResources& shadowResources;
    RhiDepthStencilStateHandle depthState;
    RhiTextureHandle shadowDummyTex;
    RhiTextureHandle skyFallbackTex;
    double depthClearValue;
};

class RenderPass {
public:
    virtual ~RenderPass();
    virtual FGPassType passType() const = 0;
    virtual const char* name() const = 0;
    virtual void setup(FGBuilder& builder) = 0;
    virtual void prepareResources(RhiCommandBuffer&) {}
    virtual void executeRender(RhiRenderCommandEncoder&) {}
    virtual void executeCompute(RhiComputeCommandEncoder&) {}
    virtual void executeBlit(RhiBlitCommandEncoder&) {}
    virtual void renderUI() {}

    // Data-driven configuration support
    virtual void configure(const PassConfig& config) {}

    // Set per-frame context before execution
    virtual void setFrameContext(const FrameContext* ctx) { m_frameContext = ctx; }

    // Set runtime context (pipelines, samplers)
    virtual void setRuntimeContext(const PipelineRuntimeContext* ctx) { m_runtimeContext = ctx; }
    void setSideEffectEnabled(bool enabled) { m_hasSideEffect = enabled; }
    bool hasSideEffectEnabled() const { return m_hasSideEffect; }

    // Get output resources by name (for pipeline builder to wire up dependencies)
    virtual FGResource getOutput(const std::string& name) const { return FGResource{}; }

    // Set input resources by name (called by pipeline builder)
    virtual void setInput(const std::string& name, FGResource resource) {
        m_inputResources[name] = resource;
    }

    // Bind an authored output slot to an imported/backbuffer target resource.
    virtual void setOutputTarget(const std::string& name, FGResource resource) {
        m_outputTargets[name] = resource;
    }

    // Get input resource by name
    FGResource getInput(const std::string& name) const {
        auto it = m_inputResources.find(name);
        return it != m_inputResources.end() ? it->second : FGResource{};
    }

    FGResource getOutputTarget(const std::string& name) const {
        auto it = m_outputTargets.find(name);
        return it != m_outputTargets.end() ? it->second : FGResource{};
    }

    FrameGraph* m_frameGraph = nullptr;

protected:
    const FrameContext* m_frameContext = nullptr;
    const PipelineRuntimeContext* m_runtimeContext = nullptr;
    std::unordered_map<std::string, FGResource> m_inputResources;
    std::unordered_map<std::string, FGResource> m_outputTargets;
    bool m_hasSideEffect = false;
};
