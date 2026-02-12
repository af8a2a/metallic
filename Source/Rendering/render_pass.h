#pragma once

#include "frame_graph.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "material_loader.h"
#include "scene_graph.h"
#include "raytraced_shadows.h"
#include <tracy/Tracy.hpp>

// Forward declaration for PassConfig (defined in PipelineEditor/pass_registry.h)
struct PassConfig;

struct RenderContext {
    const LoadedMesh& sceneMesh;
    const MeshletData& meshletData;
    const LoadedMaterials& materials;
    const SceneGraph& sceneGraph;
    const RaytracedShadowResources& shadowResources;
    MTL::DepthStencilState* depthState;
    MTL::Texture* shadowDummyTex;
    double depthClearValue;
};

class RenderPass {
public:
    virtual ~RenderPass();
    virtual FGPassType passType() const = 0;
    virtual const char* name() const = 0;
    virtual void setup(FGBuilder& builder) = 0;
    virtual void executeRender(MTL::RenderCommandEncoder*) {}
    virtual void executeCompute(MTL::ComputeCommandEncoder*) {}
    virtual void executeBlit(MTL::BlitCommandEncoder*) {}
    virtual void renderUI() {}

    // Data-driven configuration support
    virtual void configure(const PassConfig& config) {}

    // Get output resources by name (for pipeline builder)
    virtual FGResource getOutput(const std::string& name) const { return FGResource{}; }

    FrameGraph* m_frameGraph = nullptr;
};
