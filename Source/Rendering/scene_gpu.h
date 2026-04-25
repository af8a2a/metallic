#pragma once

#include "rhi_backend.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "cluster_lod_builder.h"
#include "material_loader.h"
#include "scene_graph.h"
#include "gpu_scene.h"

#include <string>

class Scene;

class SceneGpu {
public:
    SceneGpu(RhiDeviceHandle device, RhiCommandQueueHandle queue);
    ~SceneGpu();

    bool create(const Scene& scene, const std::string& cacheDir);
    void destroy();
    void updatePerFrame();

    bool isValid() const { return m_valid; }

    const LoadedMesh& mesh() const { return m_mesh; }
    const MeshletData& meshlets() const { return m_meshlets; }
    const ClusterLODData& clusterLod() const { return m_clusterLod; }
    const LoadedMaterials& materials() const { return m_materials; }
    SceneGraph& sceneGraph() { return m_sceneGraph; }
    const SceneGraph& sceneGraph() const { return m_sceneGraph; }
    const GpuSceneTables& gpuScene() const { return m_gpuScene; }

private:
    bool createMeshBuffers(const Scene& scene);
    bool createMeshlets(const Scene& scene, const std::string& cacheDir);
    bool createClusterLod(const Scene& scene, const std::string& cacheDir);
    bool createMaterials(const Scene& scene);
    bool createSceneGraph(const Scene& scene);
    bool createGpuSceneTables();

    RhiDeviceHandle m_device;
    RhiCommandQueueHandle m_queue;
    bool m_valid = false;

    LoadedMesh m_mesh;
    MeshletData m_meshlets;
    ClusterLODData m_clusterLod;
    LoadedMaterials m_materials;
    SceneGraph m_sceneGraph;
    GpuSceneTables m_gpuScene;
};
