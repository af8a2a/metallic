#include "scene_context.h"

#include "rhi_resource_utils.h"

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>
#include <fstream>
#include <vector>

bool AtmosphereTextureSet::isValid() const {
    return transmittance.nativeHandle() &&
           scattering.nativeHandle() &&
           irradiance.nativeHandle() &&
           sampler.nativeHandle();
}

void AtmosphereTextureSet::release() {
    rhiReleaseHandle(transmittance);
    rhiReleaseHandle(scattering);
    rhiReleaseHandle(irradiance);
    rhiReleaseHandle(sampler);
}

// --- Atmosphere texture helpers (moved from main.cpp) ---

static std::vector<float> loadFloatData(const std::string& path, size_t expectedCount) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::warn("Atmosphere: missing texture data {}", path);
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    if (size == 0 || size % sizeof(float) != 0) {
        spdlog::warn("Atmosphere: invalid data size {} ({} bytes)", path, size);
        return {};
    }

    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    if (!file) {
        spdlog::warn("Atmosphere: failed to read {}", path);
        return {};
    }

    if (expectedCount > 0 && data.size() != expectedCount) {
        spdlog::warn("Atmosphere: unexpected element count in {} ({} vs {})",
                     path, data.size(), expectedCount);
    }
    return data;
}

static bool loadAtmosphereTextures(const RhiDevice& device, const char* projectRoot,
                                   AtmosphereTextureSet& out) {
    constexpr int kTransmittanceWidth = 256;
    constexpr int kTransmittanceHeight = 64;
    constexpr int kScatteringWidth = 256;
    constexpr int kScatteringHeight = 128;
    constexpr int kScatteringDepth = 32;
    constexpr int kIrradianceWidth = 64;
    constexpr int kIrradianceHeight = 16;

    std::string basePath = std::string(projectRoot) + "/Asset/Atmosphere/";
    auto transmittance = loadFloatData(
        basePath + "transmittance.dat",
        static_cast<size_t>(kTransmittanceWidth) * kTransmittanceHeight * 4);
    auto scattering = loadFloatData(
        basePath + "scattering.dat",
        static_cast<size_t>(kScatteringWidth) * kScatteringHeight * kScatteringDepth * 4);
    auto irradiance = loadFloatData(
        basePath + "irradiance.dat",
        static_cast<size_t>(kIrradianceWidth) * kIrradianceHeight * 4);

    if (transmittance.empty() || scattering.empty() || irradiance.empty()) {
        return false;
    }

    out.transmittance = rhiCreateTexture2D(
        device,
        kTransmittanceWidth,
        kTransmittanceHeight,
        RhiFormat::RGBA32Float,
        false,
        1,
        RhiTextureStorageMode::Shared,
        RhiTextureUsage::ShaderRead);
    out.scattering = rhiCreateTexture3D(
        device,
        kScatteringWidth,
        kScatteringHeight,
        kScatteringDepth,
        RhiFormat::RGBA32Float,
        RhiTextureStorageMode::Shared,
        RhiTextureUsage::ShaderRead);
    out.irradiance = rhiCreateTexture2D(
        device,
        kIrradianceWidth,
        kIrradianceHeight,
        RhiFormat::RGBA32Float,
        false,
        1,
        RhiTextureStorageMode::Shared,
        RhiTextureUsage::ShaderRead);

    if (!out.transmittance.nativeHandle() ||
        !out.scattering.nativeHandle() ||
        !out.irradiance.nativeHandle()) {
        out.release();
        return false;
    }

    rhiUploadTexture2D(out.transmittance,
                       kTransmittanceWidth,
                       kTransmittanceHeight,
                       transmittance.data(),
                       static_cast<size_t>(kTransmittanceWidth) * 4 * sizeof(float));
    rhiUploadTexture3D(out.scattering,
                       kScatteringWidth,
                       kScatteringHeight,
                       kScatteringDepth,
                       scattering.data(),
                       static_cast<size_t>(kScatteringWidth) * 4 * sizeof(float),
                       static_cast<size_t>(kScatteringWidth) * kScatteringHeight * 4 * sizeof(float));
    rhiUploadTexture2D(out.irradiance,
                       kIrradianceWidth,
                       kIrradianceHeight,
                       irradiance.data(),
                       static_cast<size_t>(kIrradianceWidth) * 4 * sizeof(float));

    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.mipFilter = RhiSamplerMipFilterMode::None;
    samplerDesc.addressModeS = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeT = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeR = RhiSamplerAddressMode::ClampToEdge;
    out.sampler = rhiCreateSampler(device, samplerDesc);

    if (!out.sampler.nativeHandle()) {
        out.release();
        return false;
    }

    return true;
}

// --- SceneContext implementation ---

SceneContext::SceneContext(RhiDeviceHandle device, RhiCommandQueueHandle queue, const char* projectRoot)
    : m_device(device), m_queue(queue), m_projectRoot(projectRoot) {}

SceneContext::~SceneContext() {
    m_shadowResources.release();
    rhiReleaseHandle(m_imguiDepthDummy);
    rhiReleaseHandle(m_shadowDummyTex);
    rhiReleaseHandle(m_skyFallbackTex);
    m_atmosphereTextures.release();
    releaseGpuSceneTables(m_gpuScene);
    releaseClusterLOD(m_clusterLod);
    rhiReleaseHandle(m_meshlets.meshletBuffer);
    rhiReleaseHandle(m_meshlets.meshletVertices);
    rhiReleaseHandle(m_meshlets.meshletTriangles);
    rhiReleaseHandle(m_meshlets.boundsBuffer);
    rhiReleaseHandle(m_meshlets.materialIDs);
    for (auto& texture : m_materials.textures) {
        rhiReleaseHandle(texture);
    }
    rhiReleaseHandle(m_materials.materialBuffer);
    rhiReleaseHandle(m_materials.sampler);
    rhiReleaseHandle(m_mesh.positionBuffer);
    rhiReleaseHandle(m_mesh.normalBuffer);
    rhiReleaseHandle(m_mesh.uvBuffer);
    rhiReleaseHandle(m_mesh.indexBuffer);
    rhiReleaseHandle(m_depthState);
}

bool SceneContext::loadAll(const char* gltfPath) {
    ZoneScopedN("SceneContext::loadAll");
    const std::string meshletCacheDir = m_projectRoot + "/Asset/MeshletCache";

    releaseClusterLOD(m_clusterLod);
    m_clusterLod = ClusterLODData{};
    releaseGpuSceneTables(m_gpuScene);
    m_gpuScene = GpuSceneTables{};

    if (!loadGLTFMesh(m_device, gltfPath, m_mesh)) {
        spdlog::error("Failed to load scene mesh");
        return false;
    }

    if (!loadOrBuildMeshlets(m_device, m_mesh, gltfPath, meshletCacheDir, m_meshlets)) {
        spdlog::error("Failed to load or build meshlets");
        return false;
    }

    if (!loadOrBuildClusterLOD(m_device, m_mesh, m_meshlets, gltfPath, meshletCacheDir, m_clusterLod)) {
        spdlog::warn("Failed to load or build meshlet LOD hierarchy; continuing without meshlet LODs");
        releaseClusterLOD(m_clusterLod);
        m_clusterLod = ClusterLODData{};
    }

    if (!loadGLTFMaterials(m_device, m_queue, gltfPath, m_materials)) {
        spdlog::error("Failed to load materials");
        return false;
    }

    const ClusterLODData* clusterLodData = m_clusterLod.nodes.empty() ? nullptr : &m_clusterLod;
    if (!m_sceneGraph.buildFromGLTF(gltfPath, m_mesh, m_meshlets, clusterLodData)) {
        spdlog::error("Failed to build scene graph");
        return false;
    }
    if (!m_sceneGraph.applyBakedSingleRootScale(m_mesh)) {
        spdlog::error("Failed to apply baked scene root scale");
        return false;
    }
    m_sceneGraph.updateTransforms();
    if (!buildGpuSceneTables(m_device, m_mesh, m_meshlets, &m_clusterLod, m_sceneGraph, m_gpuScene)) {
        spdlog::warn("Failed to build shared GPU scene tables; visibility passes will fall back");
        releaseGpuSceneTables(m_gpuScene);
        m_gpuScene = GpuSceneTables{};
    }

    // Raytracing acceleration structures (non-fatal)
    if (rhiSupportsRaytracing(m_device)) {
        ZoneScopedN("Build Acceleration Structures");
        if (buildAccelerationStructures(m_device, m_queue, m_mesh, m_sceneGraph, m_shadowResources) &&
            createShadowPipeline(m_device, m_shadowResources, m_projectRoot.c_str())) {
            m_rtShadowsAvailable = true;
            spdlog::info("Raytraced shadows enabled");
        } else {
            spdlog::error("Failed to initialize raytraced shadows");
            m_shadowResources.release();
        }
    } else {
        spdlog::info("Raytracing not supported on this device");
    }

    // Atmosphere textures (non-fatal)
    m_atmosphereLoaded = loadAtmosphereTextures(m_device, m_projectRoot.c_str(), m_atmosphereTextures);
    if (!m_atmosphereLoaded) {
        spdlog::warn("Atmosphere textures not found or invalid; sky pass will use fallback");
    }

    // Depth state (reversed-Z aware)
    m_depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;
    m_depthState = rhiCreateDepthStencilState(m_device, true, ML_DEPTH_REVERSED);

    // 1x1 depth texture for ImGui pipeline matching
    m_imguiDepthDummy = rhiCreateTexture2D(m_device,
                                           1,
                                           1,
                                           RhiFormat::D32Float,
                                           false,
                                           1,
                                           RhiTextureStorageMode::Private,
                                           RhiTextureUsage::RenderTarget);

    // 1x1 shadow texture for non-RT paths (white = fully lit)
    m_shadowDummyTex = rhiCreateTexture2D(m_device,
                                          1,
                                          1,
                                          RhiFormat::R8Unorm,
                                          false,
                                          1,
                                          RhiTextureStorageMode::Shared,
                                          RhiTextureUsage::ShaderRead);
    uint8_t shadowClear = 0xFF;
    rhiUploadTexture2D(m_shadowDummyTex, 1, 1, &shadowClear, 1);

    // 1x1 sky fallback texture (BGRA8)
    m_skyFallbackTex = rhiCreateTexture2D(m_device,
                                          1,
                                          1,
                                          RhiFormat::BGRA8Unorm,
                                          false,
                                          1,
                                          RhiTextureStorageMode::Shared,
                                          RhiTextureUsage::ShaderRead);
    uint8_t skyFallbackColor[4] = {77, 51, 26, 255};
    rhiUploadTexture2D(m_skyFallbackTex, 1, 1, skyFallbackColor, 4);

    return true;
}

void SceneContext::updateGpuScene() {
    updateGpuSceneTables(m_sceneGraph, m_gpuScene);
}

RenderContext SceneContext::renderContext() const {
    RenderContext ctx{
        m_mesh, m_meshlets, m_materials, m_sceneGraph,
        m_gpuScene, m_clusterLod,
        m_shadowResources,
        m_depthState,
        m_shadowDummyTex,
        m_skyFallbackTex,
        m_depthClearValue
    };
    return ctx;
}
