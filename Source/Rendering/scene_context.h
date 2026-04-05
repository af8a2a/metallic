#pragma once

#include <string>

#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "cluster_lod_builder.h"
#include "material_loader.h"
#include "scene_graph.h"
#include "raytraced_shadows.h"
#include "render_pass.h"

struct AtmosphereTextureSet {
    RhiTextureHandle transmittance;
    RhiTextureHandle scattering;
    RhiTextureHandle irradiance;
    RhiSamplerHandle sampler;

    bool isValid() const;
    void release();
};

class SceneContext {
public:
    SceneContext(RhiDeviceHandle device, RhiCommandQueueHandle queue, const char* projectRoot);
    ~SceneContext();

    bool loadAll(const char* gltfPath);

    const LoadedMesh& mesh() const { return m_mesh; }
    const MeshletData& meshlets() const { return m_meshlets; }
    const ClusterLODData& clusterLod() const { return m_clusterLod; }
    const LoadedMaterials& materials() const { return m_materials; }
    SceneGraph& sceneGraph() { return m_sceneGraph; }
    const SceneGraph& sceneGraph() const { return m_sceneGraph; }
    RaytracedShadowResources& shadowResources() { return m_shadowResources; }
    bool rtShadowsAvailable() const { return m_rtShadowsAvailable; }

    const RhiTextureHandle& imguiDepthDummy() const { return m_imguiDepthDummy; }
    double depthClearValue() const { return m_depthClearValue; }

    bool atmosphereLoaded() const { return m_atmosphereLoaded; }
    const AtmosphereTextureSet& atmosphereTextures() const { return m_atmosphereTextures; }

    RenderContext renderContext() const;

private:
    RhiDeviceHandle m_device;
    RhiCommandQueueHandle m_queue;
    std::string m_projectRoot;

    LoadedMesh m_mesh;
    MeshletData m_meshlets;
    ClusterLODData m_clusterLod;
    LoadedMaterials m_materials;
    SceneGraph m_sceneGraph;
    RaytracedShadowResources m_shadowResources;
    bool m_rtShadowsAvailable = false;

    AtmosphereTextureSet m_atmosphereTextures;
    bool m_atmosphereLoaded = false;

    RhiDepthStencilStateHandle m_depthState;
    RhiTextureHandle m_shadowDummyTex;
    RhiTextureHandle m_skyFallbackTex;
    RhiTextureHandle m_imguiDepthDummy;
    double m_depthClearValue = 1.0;
};
