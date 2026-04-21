#pragma once

#include <memory>
#include <string>

#include "rhi_backend.h"
#include "scene.h"
#include "scene_gpu.h"
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

    bool loadScene(const std::string& gltfPath);
    void unloadScene();
    bool isSceneLoaded() const;

    // Legacy compat: loadAll delegates to loadScene with hardcoded Sponza path
    bool loadAll(const char* gltfPath);

    const LoadedMesh& mesh() const { return m_sceneGpu->mesh(); }
    const MeshletData& meshlets() const { return m_sceneGpu->meshlets(); }
    const ClusterLODData& clusterLod() const { return m_sceneGpu->clusterLod(); }
    const GpuSceneTables& gpuScene() const { return m_sceneGpu->gpuScene(); }
    const LoadedMaterials& materials() const { return m_sceneGpu->materials(); }
    SceneGraph& sceneGraph() { return m_sceneGpu->sceneGraph(); }
    const SceneGraph& sceneGraph() const { return m_sceneGpu->sceneGraph(); }
    const Scene& scene() const { return m_scene; }

    RaytracedShadowResources& shadowResources() { return m_shadowResources; }
    void setShadowResources(const RaytracedShadowResources& res) { m_shadowResources = res; }
    void setRtShadowsAvailable(bool available) { m_rtShadowsAvailable = available; }
    bool rtShadowsAvailable() const { return m_rtShadowsAvailable; }

    const RhiTextureHandle& imguiDepthDummy() const { return m_imguiDepthDummy; }
    const RhiTextureHandle& shadowDummyTex() const { return m_shadowDummyTex; }
    const RhiTextureHandle& skyFallbackTex() const { return m_skyFallbackTex; }
    double depthClearValue() const { return m_depthClearValue; }

    bool atmosphereLoaded() const { return m_atmosphereLoaded; }
    const AtmosphereTextureSet& atmosphereTextures() const { return m_atmosphereTextures; }

    void updateGpuScene();
    RenderContext renderContext() const;

private:
    bool initFallbackResources();

    RhiDeviceHandle m_device;
    RhiCommandQueueHandle m_queue;
    std::string m_projectRoot;

    Scene m_scene;
    std::unique_ptr<SceneGpu> m_sceneGpu;

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
