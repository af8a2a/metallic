#pragma once

#include <Metal/Metal.hpp>
#include <string>

#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "material_loader.h"
#include "scene_graph.h"
#include "raytraced_shadows.h"
#include "render_pass.h"

struct AtmosphereTextureSet {
    MTL::Texture* transmittance = nullptr;
    MTL::Texture* scattering = nullptr;
    MTL::Texture* irradiance = nullptr;
    MTL::SamplerState* sampler = nullptr;

    bool isValid() const {
        return transmittance && scattering && irradiance && sampler;
    }

    void release() {
        if (transmittance) { transmittance->release(); transmittance = nullptr; }
        if (scattering) { scattering->release(); scattering = nullptr; }
        if (irradiance) { irradiance->release(); irradiance = nullptr; }
        if (sampler) { sampler->release(); sampler = nullptr; }
    }
};

class SceneContext {
public:
    SceneContext(MTL::Device* device, MTL::CommandQueue* queue, const char* projectRoot);
    ~SceneContext();

    bool loadAll(const char* gltfPath);

    const LoadedMesh& mesh() const { return m_mesh; }
    const MeshletData& meshlets() const { return m_meshlets; }
    const LoadedMaterials& materials() const { return m_materials; }
    SceneGraph& sceneGraph() { return m_sceneGraph; }
    const SceneGraph& sceneGraph() const { return m_sceneGraph; }
    RaytracedShadowResources& shadowResources() { return m_shadowResources; }
    bool rtShadowsAvailable() const { return m_rtShadowsAvailable; }

    MTL::DepthStencilState* depthState() const { return m_depthState; }
    MTL::Texture* shadowDummyTex() const { return m_shadowDummyTex; }
    MTL::Texture* skyFallbackTex() const { return m_skyFallbackTex; }
    MTL::Texture* imguiDepthDummy() const { return m_imguiDepthDummy; }
    double depthClearValue() const { return m_depthClearValue; }

    bool atmosphereLoaded() const { return m_atmosphereLoaded; }
    AtmosphereTextureSet& atmosphereTextures() { return m_atmosphereTextures; }

    RenderContext renderContext() const;

private:
    MTL::Device* m_device;
    MTL::CommandQueue* m_queue;
    std::string m_projectRoot;

    LoadedMesh m_mesh;
    MeshletData m_meshlets;
    LoadedMaterials m_materials;
    SceneGraph m_sceneGraph;
    RaytracedShadowResources m_shadowResources;
    bool m_rtShadowsAvailable = false;

    AtmosphereTextureSet m_atmosphereTextures;
    bool m_atmosphereLoaded = false;

    MTL::DepthStencilState* m_depthState = nullptr;
    MTL::Texture* m_shadowDummyTex = nullptr;
    MTL::Texture* m_skyFallbackTex = nullptr;
    MTL::Texture* m_imguiDepthDummy = nullptr;
    double m_depthClearValue = 1.0;
};
