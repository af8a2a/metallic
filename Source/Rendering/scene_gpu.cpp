#include "scene_gpu.h"
#include "scene.h"
#include "rhi_resource_utils.h"

#include <spdlog/spdlog.h>
#include <cstring>
#include <algorithm>
#include <filesystem>

static constexpr uint32_t MAX_SCENE_TEXTURES_GPU = 96;

static void releaseMeshBuffers(LoadedMesh& mesh) {
    rhiReleaseHandle(mesh.positionBuffer);
    rhiReleaseHandle(mesh.normalBuffer);
    rhiReleaseHandle(mesh.uvBuffer);
    rhiReleaseHandle(mesh.indexBuffer);
    mesh.cpuPositions.clear();
    mesh.cpuIndices.clear();
    mesh.primitiveGroups.clear();
    mesh.meshRanges.clear();
    mesh.vertexCount = 0;
    mesh.indexCount = 0;
    mesh.bboxMin[0] = mesh.bboxMin[1] = mesh.bboxMin[2] = 0.0f;
    mesh.bboxMax[0] = mesh.bboxMax[1] = mesh.bboxMax[2] = 0.0f;
    mesh.hasBakedRootScale = false;
    mesh.bakedRootScale = 1.0f;
}

static void releaseMeshletBuffers(MeshletData& m) {
    rhiReleaseHandle(m.meshletBuffer);
    rhiReleaseHandle(m.meshletVertices);
    rhiReleaseHandle(m.meshletTriangles);
    rhiReleaseHandle(m.boundsBuffer);
    rhiReleaseHandle(m.materialIDs);
    m.meshletCount = 0;
    m.meshletsPerGroup.clear();
    m.cpuMeshlets.clear();
    m.cpuMeshletVertices.clear();
    m.cpuMeshletTriangles.clear();
    m.cpuBounds.clear();
    m.cpuMaterialIDs.clear();
}

static void resetSceneGraph(SceneGraph& graph) {
    graph.nodes.clear();
    graph.rootNodes.clear();
    graph.selectedNode = -1;
    graph.sunLightNode = -1;
}

static void releaseMaterialResources(LoadedMaterials& mat) {
    for (auto& tex : mat.textures)
        rhiReleaseHandle(tex);
    mat.textures.clear();
    mat.textureViews.clear();
    rhiReleaseHandle(mat.materialBuffer);
    rhiReleaseHandle(mat.sampler);
    mat.materialCount = 0;
}

SceneGpu::SceneGpu(RhiDeviceHandle device, RhiCommandQueueHandle queue)
    : m_device(device), m_queue(queue) {}

SceneGpu::~SceneGpu() {
    destroy();
}

void SceneGpu::destroy() {
    releaseGpuSceneTables(m_gpuScene);
    releaseClusterLOD(m_clusterLod);
    releaseMeshletBuffers(m_meshlets);
    releaseMaterialResources(m_materials);
    releaseMeshBuffers(m_mesh);

    resetSceneGraph(m_sceneGraph);
    m_valid = false;
}

bool SceneGpu::create(const Scene& scene, const std::string& cacheDir) {
    destroy();

    if (!createMeshBuffers(scene)) return false;
    spdlog::info("SceneGpu: creating meshlets");
    if (!createMeshlets(scene, cacheDir)) {
        spdlog::error("SceneGpu: meshlet creation failed");
        destroy();
        return false;
    }
    spdlog::info("SceneGpu: creating cluster LOD");
    createClusterLod(scene, cacheDir);
    spdlog::info("SceneGpu: creating materials");
    if (!createMaterials(scene)) {
        spdlog::error("SceneGpu: material creation failed");
        destroy();
        return false;
    }
    spdlog::info("SceneGpu: creating scene graph");
    if (!createSceneGraph(scene)) {
        spdlog::error("SceneGpu: scene graph creation failed");
        destroy();
        return false;
    }
    spdlog::info("SceneGpu: creating GPU scene tables");
    if (!createGpuSceneTables()) {
        spdlog::warn("GPU scene tables failed, continuing without");
        releaseGpuSceneTables(m_gpuScene);
    }

    m_valid = true;
    return true;
}

bool SceneGpu::createMeshBuffers(const Scene& scene) {
    releaseMeshBuffers(m_mesh);
    m_mesh.cpuPositions = scene.positions;
    m_mesh.cpuIndices = scene.indices;
    m_mesh.vertexCount = static_cast<uint32_t>(scene.positions.size() / 3);
    m_mesh.indexCount = static_cast<uint32_t>(scene.indices.size());
    std::memcpy(m_mesh.bboxMin, scene.bboxMin, sizeof(m_mesh.bboxMin));
    std::memcpy(m_mesh.bboxMax, scene.bboxMax, sizeof(m_mesh.bboxMax));
    m_mesh.hasBakedRootScale = scene.hasBakedRootScale;
    m_mesh.bakedRootScale = scene.bakedRootScale;

    m_mesh.primitiveGroups.reserve(scene.primitives.size());
    for (const auto& sp : scene.primitives) {
        LoadedMesh::PrimitiveGroup pg;
        pg.indexOffset = sp.indexOffset;
        pg.indexCount = sp.indexCount;
        pg.vertexOffset = sp.vertexOffset;
        pg.vertexCount = sp.vertexCount;
        pg.materialIndex = sp.materialIndex;
        m_mesh.primitiveGroups.push_back(pg);
    }

    m_mesh.meshRanges.reserve(scene.meshInfos.size());
    for (const auto& mi : scene.meshInfos) {
        LoadedMesh::MeshPrimitiveRange r;
        r.firstGroup = mi.firstPrimitive;
        r.groupCount = mi.primitiveCount;
        m_mesh.meshRanges.push_back(r);
    }

    const RhiDevice& dev = m_device;
    m_mesh.positionBuffer = rhiCreateSharedBuffer(
        dev, m_mesh.cpuPositions.data(),
        m_mesh.cpuPositions.size() * sizeof(float), "Mesh Positions");
    m_mesh.normalBuffer = rhiCreateSharedBuffer(
        dev, scene.normals.data(),
        scene.normals.size() * sizeof(float), "Mesh Normals");
    m_mesh.uvBuffer = rhiCreateSharedBuffer(
        dev, scene.uvs.data(),
        scene.uvs.size() * sizeof(float), "Mesh UVs");
    m_mesh.indexBuffer = rhiCreateSharedBuffer(
        dev, m_mesh.cpuIndices.data(),
        m_mesh.cpuIndices.size() * sizeof(uint32_t), "Mesh Indices");

    if (!m_mesh.positionBuffer.nativeHandle() || !m_mesh.indexBuffer.nativeHandle()) {
        spdlog::error("Failed to create mesh GPU buffers");
        return false;
    }

    spdlog::info("SceneGpu: mesh {} verts, {} indices", m_mesh.vertexCount, m_mesh.indexCount);
    return true;
}

bool SceneGpu::createMeshlets(const Scene& scene, const std::string& cacheDir) {
    const RhiDevice& dev = m_device;
    if (!cacheDir.empty()) {
        std::filesystem::create_directories(cacheDir);
        if (loadOrBuildMeshlets(dev, m_mesh, scene.filePath(), cacheDir, m_meshlets))
            return true;
    }
    return buildMeshlets(dev, m_mesh, m_meshlets);
}

bool SceneGpu::createClusterLod(const Scene& scene, const std::string& cacheDir) {
    const RhiDevice& dev = m_device;
    if (!cacheDir.empty()) {
        if (loadOrBuildClusterLOD(dev, m_mesh, m_meshlets, scene.filePath(), cacheDir, m_clusterLod))
            return true;
    }
    if (!buildClusterLOD(dev, m_mesh, m_meshlets, m_clusterLod)) {
        spdlog::warn("SceneGpu: cluster LOD build failed, continuing without");
        releaseClusterLOD(m_clusterLod);
        return false;
    }
    return true;
}

bool SceneGpu::createMaterials(const Scene& scene) {
    const RhiDevice& dev = m_device;
    const RhiCommandQueue& queue = m_queue;

    uint32_t imageCount = std::min(static_cast<uint32_t>(scene.images.size()), MAX_SCENE_TEXTURES_GPU);
    if (scene.images.size() > imageCount)
        spdlog::warn("Scene has {} images, clamping to {}", scene.images.size(), imageCount);

    m_materials.textures.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        const auto& img = scene.images[i];
        if (img.pixels.empty() || img.width <= 0 || img.height <= 0) continue;

        uint32_t mipLevels = 1;
        { int dim = std::max(img.width, img.height); while (dim > 1) { dim >>= 1; mipLevels++; } }

        m_materials.textures[i] = rhiCreateTexture2D(
            dev, static_cast<uint32_t>(img.width), static_cast<uint32_t>(img.height),
            RhiFormat::RGBA8Unorm, true, mipLevels,
            RhiTextureStorageMode::Shared, RhiTextureUsage::ShaderRead);

        if (m_materials.textures[i].nativeHandle()) {
            rhiUploadTexture2D(m_materials.textures[i],
                               static_cast<uint32_t>(img.width),
                               static_cast<uint32_t>(img.height),
                               img.pixels.data(),
                               static_cast<size_t>(img.width) * 4u);
            rhiGenerateMipmaps(queue, m_materials.textures[i]);
        }
    }

    std::vector<GPUMaterial> gpuMats(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const auto& sm = scene.materials[i];
        auto& gm = gpuMats[i];
        std::memcpy(gm.baseColorFactor, sm.baseColorFactor, sizeof(gm.baseColorFactor));
        gm.metallicFactor = sm.metallicFactor;
        gm.roughnessFactor = sm.roughnessFactor;
        gm.alphaCutoff = sm.alphaCutoff;
        gm.alphaMode = sm.alphaMode;
        gm.baseColorTexIndex = sm.baseColorTexture >= 0
            ? static_cast<uint32_t>(sm.baseColorTexture) : INVALID_TEXTURE_INDEX;
        gm.normalTexIndex = sm.normalTexture >= 0
            ? static_cast<uint32_t>(sm.normalTexture) : INVALID_TEXTURE_INDEX;
        gm.metallicRoughnessTexIndex = sm.metallicRoughnessTexture >= 0
            ? static_cast<uint32_t>(sm.metallicRoughnessTexture) : INVALID_TEXTURE_INDEX;
        gm._pad = 0.0f;
    }

    if (!gpuMats.empty()) {
        m_materials.materialBuffer = rhiCreateSharedBuffer(
            dev, gpuMats.data(), gpuMats.size() * sizeof(GPUMaterial), "Materials");
    }
    m_materials.materialCount = static_cast<uint32_t>(gpuMats.size());

    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.mipFilter = RhiSamplerMipFilterMode::Linear;
    samplerDesc.addressModeS = RhiSamplerAddressMode::Repeat;
    samplerDesc.addressModeT = RhiSamplerAddressMode::Repeat;
    m_materials.sampler = rhiCreateSampler(dev, samplerDesc);

    m_materials.textureViews.clear();
    m_materials.textureViews.reserve(m_materials.textures.size());
    for (const auto& tex : m_materials.textures)
        m_materials.textureViews.push_back(&tex);

    spdlog::info("SceneGpu: {} materials, {} textures", m_materials.materialCount, imageCount);
    return true;
}

bool SceneGpu::createSceneGraph(const Scene& scene) {
    resetSceneGraph(m_sceneGraph);

    // Build meshlet prefix sums for offset computation
    std::vector<uint32_t> meshletGroupPrefix(m_mesh.primitiveGroups.size() + 1, 0);
    for (size_t i = 0; i < m_mesh.primitiveGroups.size(); ++i) {
        uint32_t count = (i < m_meshlets.meshletsPerGroup.size())
            ? m_meshlets.meshletsPerGroup[i] : 0;
        meshletGroupPrefix[i + 1] = meshletGroupPrefix[i] + count;
    }

    auto assignPrimGroup = [&](SceneNode& gn,
                               int32_t meshIdx,
                               uint32_t firstGroup,
                               uint32_t groupCount,
                               uint32_t firstPrimitiveInMesh) {
        gn.meshIndex = meshIdx;
        gn.primitiveGroupStart = firstGroup;
        gn.primitiveGroupCount = groupCount;
        gn.primitiveIndexInMesh = firstPrimitiveInMesh;

        uint32_t totalGroups = static_cast<uint32_t>(m_mesh.primitiveGroups.size());
        uint32_t cFirst = std::min(firstGroup, totalGroups);
        uint32_t cLast = std::min(firstGroup + groupCount, totalGroups);
        if (cFirst >= cLast) return;

        if (cLast <= meshletGroupPrefix.size()) {
            gn.meshletStart = meshletGroupPrefix[cFirst];
            gn.meshletCount = meshletGroupPrefix[cLast] - gn.meshletStart;
        }

        gn.indexStart = m_mesh.primitiveGroups[cFirst].indexOffset;
        const auto& lastPg = m_mesh.primitiveGroups[cLast - 1];
        gn.indexCount = (lastPg.indexOffset + lastPg.indexCount) - gn.indexStart;

        if (groupCount == 1)
            gn.materialIndex = m_mesh.primitiveGroups[cFirst].materialIndex;

        // ClusterLOD owns the traversal root contract: selector-chain mode stores
        // the selector root, Nyx-style mode stores the top-level BVH root.
        if (groupCount == 1 && cFirst < m_clusterLod.primitiveGroupLodRoots.size()) {
            gn.lodRootNode = m_clusterLod.primitiveGroupLodRoots[cFirst];
            if (cFirst < m_clusterLod.primitiveGroupLod0Roots.size()) {
                gn.lod0RootNode = m_clusterLod.primitiveGroupLod0Roots[cFirst];
            } else {
                gn.lod0RootNode = gn.lodRootNode;
            }
        }
    };

    // First pass: create nodes matching scene hierarchy
    m_sceneGraph.nodes.resize(scene.nodes.size());
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const auto& sn = scene.nodes[i];
        auto& gn = m_sceneGraph.nodes[i];
        gn.name = sn.name;
        gn.id = static_cast<uint32_t>(i);
        gn.parent = sn.parent;
        gn.children.assign(sn.children.begin(), sn.children.end());
        gn.visible = true;

        if (sn.hasMatrix) {
            std::memcpy(&gn.transform.localMatrix, sn.localMatrix, sizeof(float) * 16);
            gn.transform.useLocalMatrix = true;
        } else {
            gn.transform.translation = float3(sn.translation[0], sn.translation[1], sn.translation[2]);
            gn.transform.rotation = float4(sn.rotation[0], sn.rotation[1], sn.rotation[2], sn.rotation[3]);
            gn.transform.scale = float3(sn.scale[0], sn.scale[1], sn.scale[2]);
            gn.transform.useLocalMatrix = false;
        }
        gn.transform.dirty = true;

        if (sn.light >= 0 && sn.light < static_cast<int>(scene.lights.size())) {
            gn.lightIndex = sn.light;
            gn.hasLight = true;
            gn.light.type = LightType::Directional;
            const auto& sl = scene.lights[sn.light];
            gn.light.directional.color = float3(sl.color[0], sl.color[1], sl.color[2]);
            gn.light.directional.intensity = sl.intensity;
            gn.light.directional.direction = normalize(float3(0.5f, 1.0f, 0.8f));
            if (m_sceneGraph.sunLightNode < 0 && sl.type == SceneLight::Directional)
                m_sceneGraph.sunLightNode = static_cast<int32_t>(i);
        }

        if (sn.camera >= 0 && sn.camera < static_cast<int>(scene.cameras.size())) {
            gn.cameraIndex = sn.camera;
            const auto& sc = scene.cameras[sn.camera];
            gn.camera.type = (sc.type == SceneCamera::Perspective)
                ? CameraType::Perspective
                : CameraType::Orthographic;
            gn.camera.yfov = sc.yfov;
            gn.camera.znear = sc.znear;
            gn.camera.zfar = sc.zfar;
            gn.camera.aspectRatio = sc.aspectRatio;
        }
    }

    // Second pass: assign mesh data, splitting multi-primitive meshes into child nodes
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const auto& sn = scene.nodes[i];
        if (sn.mesh < 0 || sn.mesh >= static_cast<int>(scene.meshInfos.size()))
            continue;

        const auto& mi = scene.meshInfos[sn.mesh];

        if (mi.primitiveCount <= 1) {
            assignPrimGroup(m_sceneGraph.nodes[i], sn.mesh, mi.firstPrimitive, mi.primitiveCount, 0);
        } else {
            // Multi-primitive mesh: create a child node per primitive group
            m_sceneGraph.nodes[i].meshIndex = sn.mesh;
            std::string parentName = m_sceneGraph.nodes[i].name;
            for (uint32_t pg = 0; pg < mi.primitiveCount; ++pg) {
                uint32_t groupIdx = mi.firstPrimitive + pg;
                uint32_t childIdx = static_cast<uint32_t>(m_sceneGraph.nodes.size());

                SceneNode child;
                child.id = childIdx;
                child.parent = static_cast<int32_t>(i);
                child.visible = true;
                child.transform.dirty = true;
                child.generatedPrimitive = true;

                if (groupIdx < m_mesh.primitiveGroups.size()) {
                    child.name = parentName + "/Prim_" + std::to_string(pg)
                        + " [Mat " + std::to_string(m_mesh.primitiveGroups[groupIdx].materialIndex) + "]";
                } else {
                    child.name = parentName + "/Prim_" + std::to_string(pg);
                }

                assignPrimGroup(child, sn.mesh, groupIdx, 1, pg);
                m_sceneGraph.nodes.push_back(std::move(child));
                m_sceneGraph.nodes[i].children.push_back(childIdx);
            }
        }
    }

    for (int ri : scene.rootNodes)
        m_sceneGraph.rootNodes.push_back(static_cast<uint32_t>(ri));

    if (!m_sceneGraph.rootNodes.empty())
        m_sceneGraph.selectedNode = static_cast<int32_t>(m_sceneGraph.rootNodes.front());

    if (scene.hasBakedRootScale)
        m_sceneGraph.applyBakedSingleRootScale(m_mesh);

    m_sceneGraph.updateTransforms();

    if (m_sceneGraph.sunLightNode < 0)
        m_sceneGraph.addDirectionalLightNode("Sun", float3(0.5f, 1.0f, 0.8f), true);

    return true;
}

bool SceneGpu::createGpuSceneTables() {
    const ClusterLODData* lodPtr = m_clusterLod.nodes.empty() ? nullptr : &m_clusterLod;
    return buildGpuSceneTables(m_device, m_mesh, m_meshlets, lodPtr, m_sceneGraph, m_gpuScene);
}

void SceneGpu::updatePerFrame() {
    m_sceneGraph.updateTransforms();
    updateGpuSceneTables(m_sceneGraph, m_gpuScene);
}
