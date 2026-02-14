#include "scene_context.h"

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>
#include <fstream>
#include <vector>

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

static MTL::Texture* createTexture2D(MTL::Device* device, int width, int height,
                                     const float* data) {
    auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA32Float, width, height, false);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    auto* tex = device->newTexture(desc);
    desc->release();
    if (!tex) return nullptr;
    size_t bytesPerRow = static_cast<size_t>(width) * 4 * sizeof(float);
    tex->replaceRegion(MTL::Region(0, 0, 0, width, height, 1), 0, data, bytesPerRow);
    return tex;
}

static MTL::Texture* createTexture3D(MTL::Device* device, int width, int height, int depth,
                                     const float* data) {
    auto* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType3D);
    desc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setDepth(depth);
    desc->setMipmapLevelCount(1);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    auto* tex = device->newTexture(desc);
    desc->release();
    if (!tex) return nullptr;
    size_t bytesPerRow = static_cast<size_t>(width) * 4 * sizeof(float);
    size_t bytesPerImage = bytesPerRow * static_cast<size_t>(height);
    tex->replaceRegion(MTL::Region(0, 0, 0, width, height, depth), 0, 0,
                       data, bytesPerRow, bytesPerImage);
    return tex;
}

static bool loadAtmosphereTextures(MTL::Device* device, const char* projectRoot,
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

    out.transmittance = createTexture2D(
        device, kTransmittanceWidth, kTransmittanceHeight, transmittance.data());
    out.scattering = createTexture3D(
        device, kScatteringWidth, kScatteringHeight, kScatteringDepth, scattering.data());
    out.irradiance = createTexture2D(
        device, kIrradianceWidth, kIrradianceHeight, irradiance.data());

    if (!out.transmittance || !out.scattering || !out.irradiance) {
        out.release();
        return false;
    }

    auto* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
    samplerDesc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    samplerDesc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    samplerDesc->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
    out.sampler = device->newSamplerState(samplerDesc);
    samplerDesc->release();

    if (!out.sampler) {
        out.release();
        return false;
    }

    return true;
}

// --- SceneContext implementation ---

SceneContext::SceneContext(MTL::Device* device, MTL::CommandQueue* queue, const char* projectRoot)
    : m_device(device), m_queue(queue), m_projectRoot(projectRoot) {}

SceneContext::~SceneContext() {
    m_shadowResources.release();
    if (m_imguiDepthDummy) m_imguiDepthDummy->release();
    if (m_shadowDummyTex) m_shadowDummyTex->release();
    if (m_skyFallbackTex) m_skyFallbackTex->release();
    m_atmosphereTextures.release();
    if (m_meshlets.meshletBuffer) m_meshlets.meshletBuffer->release();
    if (m_meshlets.meshletVertices) m_meshlets.meshletVertices->release();
    if (m_meshlets.meshletTriangles) m_meshlets.meshletTriangles->release();
    if (m_meshlets.boundsBuffer) m_meshlets.boundsBuffer->release();
    if (m_meshlets.materialIDs) m_meshlets.materialIDs->release();
    for (auto* tex : m_materials.textures) {
        if (tex) tex->release();
    }
    if (m_materials.materialBuffer) m_materials.materialBuffer->release();
    if (m_materials.sampler) m_materials.sampler->release();
    if (m_mesh.positionBuffer) m_mesh.positionBuffer->release();
    if (m_mesh.normalBuffer) m_mesh.normalBuffer->release();
    if (m_mesh.uvBuffer) m_mesh.uvBuffer->release();
    if (m_mesh.indexBuffer) m_mesh.indexBuffer->release();
    if (m_depthState) m_depthState->release();
}

bool SceneContext::loadAll(const char* gltfPath) {
    ZoneScopedN("SceneContext::loadAll");

    if (!loadGLTFMesh(m_device, gltfPath, m_mesh)) {
        spdlog::error("Failed to load scene mesh");
        return false;
    }

    if (!buildMeshlets(m_device, m_mesh, m_meshlets)) {
        spdlog::error("Failed to build meshlets");
        return false;
    }

    if (!loadGLTFMaterials(m_device, m_queue, gltfPath, m_materials)) {
        spdlog::error("Failed to load materials");
        return false;
    }

    if (!m_sceneGraph.buildFromGLTF(gltfPath, m_mesh, m_meshlets)) {
        spdlog::error("Failed to build scene graph");
        return false;
    }
    m_sceneGraph.updateTransforms();

    // Raytracing acceleration structures (non-fatal)
    if (m_device->supportsRaytracing()) {
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
    MTL::DepthStencilDescriptor* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(
        ML_DEPTH_REVERSED ? MTL::CompareFunctionGreater : MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(true);
    m_depthState = m_device->newDepthStencilState(depthDesc);
    depthDesc->release();

    // 1x1 depth texture for ImGui pipeline matching
    auto* imguiDepthTexDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, 1, 1, false);
    imguiDepthTexDesc->setStorageMode(MTL::StorageModePrivate);
    imguiDepthTexDesc->setUsage(MTL::TextureUsageRenderTarget);
    m_imguiDepthDummy = m_device->newTexture(imguiDepthTexDesc);

    // 1x1 shadow texture for non-RT paths (white = fully lit)
    auto* shadowDummyDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatR8Unorm, 1, 1, false);
    shadowDummyDesc->setStorageMode(MTL::StorageModeShared);
    shadowDummyDesc->setUsage(MTL::TextureUsageShaderRead);
    m_shadowDummyTex = m_device->newTexture(shadowDummyDesc);
    uint8_t shadowClear = 0xFF;
    m_shadowDummyTex->replaceRegion(MTL::Region(0, 0, 0, 1, 1, 1), 0, &shadowClear, 1);

    // 1x1 sky fallback texture (BGRA8)
    auto* skyFallbackDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatBGRA8Unorm, 1, 1, false);
    skyFallbackDesc->setStorageMode(MTL::StorageModeShared);
    skyFallbackDesc->setUsage(MTL::TextureUsageShaderRead);
    m_skyFallbackTex = m_device->newTexture(skyFallbackDesc);
    uint8_t skyFallbackColor[4] = {77, 51, 26, 255};
    m_skyFallbackTex->replaceRegion(MTL::Region(0, 0, 0, 1, 1, 1), 0, skyFallbackColor, 4);

    return true;
}

RenderContext SceneContext::renderContext() const {
    return RenderContext{
        m_mesh, m_meshlets, m_materials, m_sceneGraph,
        m_shadowResources, m_depthState, m_shadowDummyTex, m_skyFallbackTex, m_depthClearValue
    };
}
