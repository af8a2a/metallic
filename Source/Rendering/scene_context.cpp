#include "scene_context.h"

#include <Metal/Metal.hpp>

#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>
#include <fstream>
#include <vector>

static MTL::Buffer* metalBuffer(void* handle) {
    return static_cast<MTL::Buffer*>(handle);
}

static MTL::Texture* metalTexture(void* handle) {
    return static_cast<MTL::Texture*>(handle);
}

static MTL::SamplerState* metalSampler(void* handle) {
    return static_cast<MTL::SamplerState*>(handle);
}

bool AtmosphereTextureSet::isValid() const {
    return transmittance && scattering && irradiance && sampler;
}

void AtmosphereTextureSet::release() {
    if (transmittance) { transmittance->release(); transmittance = nullptr; }
    if (scattering) { scattering->release(); scattering = nullptr; }
    if (irradiance) { irradiance->release(); irradiance = nullptr; }
    if (sampler) { sampler->release(); sampler = nullptr; }
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

static void syncMeshRhiViews(LoadedMesh& mesh) {
    mesh.positionBufferRhi.setNativeHandle(mesh.positionBuffer);
    mesh.normalBufferRhi.setNativeHandle(mesh.normalBuffer);
    mesh.uvBufferRhi.setNativeHandle(mesh.uvBuffer);
    mesh.indexBufferRhi.setNativeHandle(mesh.indexBuffer);
}

static void syncMeshletRhiViews(MeshletData& meshlets) {
    meshlets.meshletBufferRhi.setNativeHandle(meshlets.meshletBuffer);
    meshlets.meshletVerticesRhi.setNativeHandle(meshlets.meshletVertices);
    meshlets.meshletTrianglesRhi.setNativeHandle(meshlets.meshletTriangles);
    meshlets.boundsBufferRhi.setNativeHandle(meshlets.boundsBuffer);
    meshlets.materialIDsRhi.setNativeHandle(meshlets.materialIDs);
}

static void syncMaterialRhiViews(LoadedMaterials& materials) {
    materials.textureHandles.clear();
    materials.textureViews.clear();
    materials.textureHandles.reserve(materials.textures.size());
    materials.textureViews.reserve(materials.textures.size());
    for (void* textureHandle : materials.textures) {
        auto* texture = metalTexture(textureHandle);
        materials.textureHandles.emplace_back(textureHandle,
                                              texture ? static_cast<uint32_t>(texture->width()) : 0,
                                              texture ? static_cast<uint32_t>(texture->height()) : 0);
    }
    for (auto& textureHandle : materials.textureHandles) {
        materials.textureViews.push_back(&textureHandle);
    }
    materials.materialBufferRhi.setNativeHandle(materials.materialBuffer);
    materials.samplerRhi.setNativeHandle(materials.sampler);
}

static void syncShadowRhiViews(RaytracedShadowResources& shadowResources) {
    shadowResources.blasHandles.clear();
    shadowResources.blasHandles.reserve(shadowResources.blasArray.size());
    for (auto* blas : shadowResources.blasArray) {
        shadowResources.blasHandles.emplace_back(blas);
    }
    shadowResources.tlasRhi.setNativeHandle(shadowResources.tlas);
    shadowResources.instanceDescriptorBufferRhi.setNativeHandle(shadowResources.instanceDescriptorBuffer);
    shadowResources.scratchBufferRhi.setNativeHandle(shadowResources.scratchBuffer);
    shadowResources.pipelineRhi.setNativeHandle(shadowResources.pipeline);
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
    if (m_meshlets.meshletBuffer) metalBuffer(m_meshlets.meshletBuffer)->release();
    if (m_meshlets.meshletVertices) metalBuffer(m_meshlets.meshletVertices)->release();
    if (m_meshlets.meshletTriangles) metalBuffer(m_meshlets.meshletTriangles)->release();
    if (m_meshlets.boundsBuffer) metalBuffer(m_meshlets.boundsBuffer)->release();
    if (m_meshlets.materialIDs) metalBuffer(m_meshlets.materialIDs)->release();
    for (void* texHandle : m_materials.textures) {
        auto* tex = metalTexture(texHandle);
        if (tex) tex->release();
    }
    if (m_materials.materialBuffer) metalBuffer(m_materials.materialBuffer)->release();
    if (m_materials.sampler) metalSampler(m_materials.sampler)->release();
    if (m_mesh.positionBuffer) metalBuffer(m_mesh.positionBuffer)->release();
    if (m_mesh.normalBuffer) metalBuffer(m_mesh.normalBuffer)->release();
    if (m_mesh.uvBuffer) metalBuffer(m_mesh.uvBuffer)->release();
    if (m_mesh.indexBuffer) metalBuffer(m_mesh.indexBuffer)->release();
    if (m_depthState) m_depthState->release();
}

bool SceneContext::loadAll(const char* gltfPath) {
    ZoneScopedN("SceneContext::loadAll");

    if (!loadGLTFMesh(m_device, gltfPath, m_mesh)) {
        spdlog::error("Failed to load scene mesh");
        return false;
    }
    syncMeshRhiViews(m_mesh);

    if (!buildMeshlets(m_device, m_mesh, m_meshlets)) {
        spdlog::error("Failed to build meshlets");
        return false;
    }
    syncMeshletRhiViews(m_meshlets);

    if (!loadGLTFMaterials(m_device, m_queue, gltfPath, m_materials)) {
        spdlog::error("Failed to load materials");
        return false;
    }
    syncMaterialRhiViews(m_materials);

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
            syncShadowRhiViews(m_shadowResources);
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
    RenderContext ctx{
        m_mesh, m_meshlets, m_materials, m_sceneGraph,
        m_shadowResources,
        RhiDepthStencilStateHandle(m_depthState),
        RhiTextureHandle(m_shadowDummyTex,
                         m_shadowDummyTex ? static_cast<uint32_t>(m_shadowDummyTex->width()) : 0,
                         m_shadowDummyTex ? static_cast<uint32_t>(m_shadowDummyTex->height()) : 0),
        RhiTextureHandle(m_skyFallbackTex,
                         m_skyFallbackTex ? static_cast<uint32_t>(m_skyFallbackTex->width()) : 0,
                         m_skyFallbackTex ? static_cast<uint32_t>(m_skyFallbackTex->height()) : 0),
        m_depthClearValue
    };
    return ctx;
}
