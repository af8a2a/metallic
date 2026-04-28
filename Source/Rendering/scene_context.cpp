#include "scene_context.h"

#include "rhi_resource_utils.h"

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>
#include <fstream>
#include <vector>
#include <filesystem>

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

static std::string resolveScenePath(const std::string& projectRoot, const std::string& gltfPath) {
    std::filesystem::path path(gltfPath);
    if (!path.is_absolute() && !projectRoot.empty()) {
        path = std::filesystem::path(projectRoot) / path;
    }
    return path.lexically_normal().generic_string();
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
        device, kTransmittanceWidth, kTransmittanceHeight,
        RhiFormat::RGBA32Float, false, 1,
        RhiTextureStorageMode::Shared, RhiTextureUsage::ShaderRead);
    rhiUploadTexture2D(out.transmittance, kTransmittanceWidth, kTransmittanceHeight,
                       transmittance.data(),
                       static_cast<size_t>(kTransmittanceWidth) * 4 * sizeof(float));

    out.scattering = rhiCreateTexture3D(
        device, kScatteringWidth, kScatteringHeight, kScatteringDepth,
        RhiFormat::RGBA32Float,
        RhiTextureStorageMode::Shared, RhiTextureUsage::ShaderRead);
    size_t scatterBytesPerRow = static_cast<size_t>(kScatteringWidth) * 4 * sizeof(float);
    size_t scatterBytesPerImage = scatterBytesPerRow * kScatteringHeight;
    rhiUploadTexture3D(out.scattering, kScatteringWidth, kScatteringHeight, kScatteringDepth,
                       scattering.data(), scatterBytesPerRow, scatterBytesPerImage);

    out.irradiance = rhiCreateTexture2D(
        device, kIrradianceWidth, kIrradianceHeight,
        RhiFormat::RGBA32Float, false, 1,
        RhiTextureStorageMode::Shared, RhiTextureUsage::ShaderRead);
    rhiUploadTexture2D(out.irradiance, kIrradianceWidth, kIrradianceHeight,
                       irradiance.data(),
                       static_cast<size_t>(kIrradianceWidth) * 4 * sizeof(float));

    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.mipFilter = RhiSamplerMipFilterMode::None;
    samplerDesc.addressModeS = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeT = RhiSamplerAddressMode::ClampToEdge;
    out.sampler = rhiCreateSampler(device, samplerDesc);

    return out.isValid();
}

SceneContext::SceneContext(RhiDeviceHandle device, RhiCommandQueueHandle queue, const char* projectRoot)
    : m_device(device), m_queue(queue), m_projectRoot(projectRoot) {}

SceneContext::~SceneContext() {
    unloadScene();
    m_atmosphereTextures.release();
    rhiReleaseHandle(m_depthState);
    rhiReleaseHandle(m_shadowDummyTex);
    rhiReleaseHandle(m_skyFallbackTex);
    rhiReleaseHandle(m_imguiDepthDummy);
}

bool SceneContext::initFallbackResources() {
    if (m_shadowDummyTex.nativeHandle()) return true;

    const RhiDevice& dev = m_device;

    m_depthState = rhiCreateDepthStencilState(dev, true, ML_DEPTH_REVERSED);
    m_depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;

    m_imguiDepthDummy = rhiCreateTexture2D(dev, 1, 1, RhiFormat::D32Float, false, 1,
                                           RhiTextureStorageMode::Private,
                                           RhiTextureUsage::RenderTarget);

    m_shadowDummyTex = rhiCreateTexture2D(dev, 1, 1, RhiFormat::R8Unorm, false, 1,
                                          RhiTextureStorageMode::Shared,
                                          RhiTextureUsage::ShaderRead);
    uint8_t shadowClear = 0xFF;
    rhiUploadTexture2D(m_shadowDummyTex, 1, 1, &shadowClear, 1);

    m_skyFallbackTex = rhiCreateTexture2D(dev, 1, 1, RhiFormat::BGRA8Unorm, false, 1,
                                          RhiTextureStorageMode::Shared,
                                          RhiTextureUsage::ShaderRead);
    uint8_t skyFallbackColor[4] = {77, 51, 26, 255};
    rhiUploadTexture2D(m_skyFallbackTex, 1, 1, skyFallbackColor, 4);

    if (!m_atmosphereLoaded) {
        m_atmosphereLoaded = loadAtmosphereTextures(dev, m_projectRoot.c_str(), m_atmosphereTextures);
    }

    return true;
}

bool SceneContext::loadScene(const std::string& gltfPath) {
    ZoneScoped;

    unloadScene();
    initFallbackResources();

    const std::string resolvedGltfPath = resolveScenePath(m_projectRoot, gltfPath);
    if (!m_scene.load(resolvedGltfPath)) {
        spdlog::error("Failed to load scene: {}", resolvedGltfPath);
        return false;
    }

    std::string cacheDir = m_projectRoot + "/Asset/MeshletCache";
    m_sceneGpu = std::make_unique<SceneGpu>(m_device, m_queue);
    if (!m_sceneGpu->create(m_scene, cacheDir)) {
        spdlog::error("Failed to create GPU resources for scene: {}", resolvedGltfPath);
        m_sceneGpu.reset();
        m_scene.clear();
        return false;
    }

    spdlog::info("Scene loaded successfully: {}", resolvedGltfPath);
    return true;
}

void SceneContext::unloadScene() {
    m_sceneGpu.reset();
    m_scene.clear();
}

bool SceneContext::isSceneLoaded() const {
    return m_sceneGpu && m_sceneGpu->isValid();
}

bool SceneContext::loadAll(const char* gltfPath) {
    return loadScene(gltfPath);
}

void SceneContext::updateGpuScene() {
    if (m_sceneGpu)
        m_sceneGpu->updatePerFrame();
}

RenderContext SceneContext::renderContext() const {
    RenderContext ctx{
        m_sceneGpu->mesh(),
        m_sceneGpu->meshlets(),
        m_sceneGpu->materials(),
        m_sceneGpu->sceneGraph(),
        m_sceneGpu->gpuScene(),
        m_sceneGpu->clusterLod(),
        m_shadowResources,
        m_depthState,
        m_shadowDummyTex,
        m_skyFallbackTex,
        m_depthClearValue
    };
    return ctx;
}
