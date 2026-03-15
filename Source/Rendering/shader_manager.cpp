#include "shader_manager.h"

#include "rhi_resource_utils.h"
#include "rhi_shader_utils.h"
#include "slang_compiler.h"

#include <spdlog/spdlog.h>
#include <cstring>

namespace {

#ifdef __APPLE__
constexpr RhiBackendType kShaderBackend = RhiBackendType::Metal;
#else
constexpr RhiBackendType kShaderBackend = RhiBackendType::Vulkan;
#endif

// On Vulkan, compile to SPIR-V binary and pack into a string for rhiCreatePipelineFromSource.
// On Metal, compile to MSL source string directly.
std::string compileGraphics(const char* shaderPath, const char* projectRoot) {
#ifdef __APPLE__
    return compileSlangGraphicsSource(kShaderBackend, shaderPath, projectRoot);
#else
    auto spirv = compileSlangGraphicsBinary(kShaderBackend, shaderPath, projectRoot);
    if (spirv.empty()) return {};
    return std::string(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(uint32_t));
#endif
}

std::string compileMesh(const char* shaderPath, const char* projectRoot) {
#ifdef __APPLE__
    return compileSlangMeshSource(kShaderBackend, shaderPath, projectRoot);
#else
    auto spirv = compileSlangMeshBinary(kShaderBackend, shaderPath, projectRoot);
    if (spirv.empty()) return {};
    return std::string(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(uint32_t));
#endif
}

std::string compileCompute(const char* shaderPath, const char* projectRoot, const char* entryPoint) {
#ifdef __APPLE__
    return compileSlangComputeSource(kShaderBackend, shaderPath, projectRoot, entryPoint);
#else
    auto spirv = compileSlangComputeBinary(kShaderBackend, shaderPath, projectRoot, entryPoint);
    if (spirv.empty()) return {};
    return std::string(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(uint32_t));
#endif
}

template <typename Handle>
void releaseOwnedHandle(Handle& handle) {
    rhiReleaseHandle(handle);
}

const char* defaultError(const char* fallback) {
    return fallback ? fallback : "Unknown error";
}

std::string formatError(const std::string* errorMessage, const char* fallback) {
    if (errorMessage && !errorMessage->empty()) {
        return *errorMessage;
    }
    return defaultError(fallback);
}

} // namespace

ShaderManager::ShaderManager(RhiDeviceHandle device, const char* projectRoot)
    : m_device(device)
    , m_projectRoot(projectRoot)
    , m_rtCtx(new PipelineRuntimeContext{})
{}

ShaderManager::~ShaderManager() {
    releaseOwnedHandle(m_vertexPipeline);
    releaseOwnedHandle(m_meshPipeline);
    releaseOwnedHandle(m_visPipeline);
    releaseOwnedHandle(m_visIndirectPipeline);
    releaseOwnedHandle(m_computePipeline);
    releaseOwnedHandle(m_cullPipeline);
    releaseOwnedHandle(m_buildIndirectPipeline);
    releaseOwnedHandle(m_meshletVisPipeline);
    releaseOwnedHandle(m_skyPipeline);
    releaseOwnedHandle(m_tonemapPipeline);
    releaseOwnedHandle(m_outputPipeline);
    releaseOwnedHandle(m_histogramPipeline);
    releaseOwnedHandle(m_autoExposurePipeline);
    releaseOwnedHandle(m_taaPipeline);
    releaseOwnedHandle(m_tonemapSampler);
    releaseOwnedHandle(m_vertexDesc);
    delete m_rtCtx;
}

PipelineRuntimeContext& ShaderManager::runtimeContext() { return *m_rtCtx; }
bool ShaderManager::hasSkyPipeline() const { return m_skyPipeline.nativeHandle() != nullptr; }

void ShaderManager::importTexture(const std::string& name, const RhiTexture& texture) {
    m_rtCtx->importedTexturesRhi[name].setNativeHandle(
        texture.nativeHandle(),
        texture.width(),
        texture.height());
}

void ShaderManager::importSampler(const std::string& name, const RhiSampler& sampler) {
    m_rtCtx->samplersRhi[name].setNativeHandle(sampler.nativeHandle());
}

void ShaderManager::createVertexDescriptor() {
    releaseOwnedHandle(m_vertexDesc);
    m_vertexDesc = rhiCreateVertexDescriptor();

    rhiVertexDescriptorSetAttribute(m_vertexDesc, 0, RhiVertexFormat::Float3, 0, 1);
    rhiVertexDescriptorSetAttribute(m_vertexDesc, 1, RhiVertexFormat::Float3, 0, 2);
    rhiVertexDescriptorSetLayout(m_vertexDesc, 1, 12);
    rhiVertexDescriptorSetLayout(m_vertexDesc, 2, 12);
}

void ShaderManager::syncRuntimeContext() {
    m_rtCtx->renderPipelinesRhi["ForwardPass"] = m_vertexPipeline;
    m_rtCtx->renderPipelinesRhi["ForwardMeshPass"] = m_meshPipeline;
    m_rtCtx->renderPipelinesRhi["VisibilityPass"] = m_visPipeline;
    m_rtCtx->renderPipelinesRhi["VisibilityIndirectPass"] = m_visIndirectPipeline;
    m_rtCtx->renderPipelinesRhi["SkyPass"] = m_skyPipeline;
    m_rtCtx->renderPipelinesRhi["TonemapPass"] = m_tonemapPipeline;
    m_rtCtx->renderPipelinesRhi["OutputPass"] = m_outputPipeline;

    m_rtCtx->computePipelinesRhi["DeferredLightingPass"] = m_computePipeline;
    m_rtCtx->computePipelinesRhi["MeshletCullPass"] = m_cullPipeline;
    m_rtCtx->computePipelinesRhi["BuildIndirectPass"] = m_buildIndirectPipeline;
    if (m_meshletVisPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["MeshletVisualizePass"] = m_meshletVisPipeline;
    if (m_histogramPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["HistogramPass"] = m_histogramPipeline;
    if (m_autoExposurePipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["AutoExposurePass"] = m_autoExposurePipeline;
    if (m_taaPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["TAAPass"] = m_taaPipeline;

    m_rtCtx->samplersRhi["tonemap"] = m_tonemapSampler;
}

bool ShaderManager::buildAll() {
    createVertexDescriptor();

    std::string errorMessage;

    m_vertexPipeline = reloadVertexShader("Shaders/Vertex/bunny", &errorMessage);
    if (!m_vertexPipeline.nativeHandle()) {
        spdlog::error("Failed to create vertex pipeline: {}",
                      formatError(&errorMessage, "Slang vertex shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_meshPipeline = reloadMeshShader("Shaders/Mesh/meshlet",
                                      patchMeshShaderSource,
                                      RhiFormat::RGBA16Float,
                                      RhiFormat::D32Float,
                                      &errorMessage);
    if (!m_meshPipeline.nativeHandle()) {
        spdlog::error("Failed to create mesh pipeline: {}",
                      formatError(&errorMessage, "Slang mesh shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_visPipeline = reloadMeshShader("Shaders/Visibility/visibility",
                                     patchVisibilityShaderSource,
                                     RhiFormat::R32Uint,
                                     RhiFormat::D32Float,
                                     &errorMessage);
    if (!m_visPipeline.nativeHandle()) {
        spdlog::error("Failed to create visibility pipeline: {}",
                      formatError(&errorMessage, "Slang visibility shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_visIndirectPipeline = reloadMeshShader("Shaders/Visibility/visibility_indirect",
                                             patchVisibilityShaderSource,
                                             RhiFormat::R32Uint,
                                             RhiFormat::D32Float,
                                             &errorMessage);
    if (!m_visIndirectPipeline.nativeHandle()) {
        spdlog::error("Failed to create visibility indirect pipeline: {}",
                      formatError(&errorMessage, "Slang visibility indirect shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_cullPipeline = reloadComputeShader("Shaders/Visibility/meshlet_cull",
                                         "computeMain",
                                         nullptr,
                                         &errorMessage);
    if (!m_cullPipeline.nativeHandle()) {
        spdlog::error("Failed to create meshlet cull pipeline: {}",
                      formatError(&errorMessage, "Slang meshlet cull shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_buildIndirectPipeline = reloadComputeShader("Shaders/Visibility/build_indirect",
                                                  "computeMain",
                                                  nullptr,
                                                  &errorMessage);
    if (!m_buildIndirectPipeline.nativeHandle()) {
        spdlog::error("Failed to create build indirect pipeline: {}",
                      formatError(&errorMessage, "Slang build indirect shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_computePipeline = reloadComputeShader("Shaders/Visibility/deferred_lighting",
                                            "computeMain",
                                            patchComputeShaderSource,
                                            &errorMessage);
    if (!m_computePipeline.nativeHandle()) {
        spdlog::error("Failed to create deferred lighting pipeline: {}",
                      formatError(&errorMessage, "Slang deferred lighting shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_meshletVisPipeline = reloadComputeShader("Shaders/Visibility/meshlet_visualize",
                                               "computeMain",
                                               nullptr,
                                               &errorMessage);
    if (!m_meshletVisPipeline.nativeHandle()) {
        spdlog::warn("Failed to compile meshlet visualize shader; visualization disabled: {}",
                     formatError(&errorMessage, "Slang meshlet visualize shader compilation failed"));
    }

    errorMessage.clear();
    m_skyPipeline = reloadFullscreenShader("Shaders/Atmosphere/sky",
                                           RhiFormat::RGBA16Float,
                                           &errorMessage);
    if (!m_skyPipeline.nativeHandle()) {
        spdlog::warn("Sky shader compile failed; atmosphere sky disabled: {}",
                     formatError(&errorMessage, "Slang sky shader compilation failed"));
    }

    errorMessage.clear();
    m_tonemapPipeline = reloadFullscreenShader("Shaders/Post/tonemap",
                                               RhiFormat::BGRA8Unorm,
                                               &errorMessage);
    if (!m_tonemapPipeline.nativeHandle()) {
        spdlog::error("Failed to create tonemap pipeline: {}",
                      formatError(&errorMessage, "Slang tonemap shader compilation failed"));
        return false;
    }

    RhiSamplerDesc samplerDesc;
    samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
    samplerDesc.mipFilter = RhiSamplerMipFilterMode::None;
    samplerDesc.addressModeS = RhiSamplerAddressMode::ClampToEdge;
    samplerDesc.addressModeT = RhiSamplerAddressMode::ClampToEdge;
    m_tonemapSampler = rhiCreateSampler(m_device, samplerDesc);
    if (!m_tonemapSampler.nativeHandle()) {
        spdlog::error("Failed to create tonemap sampler state");
        return false;
    }

    errorMessage.clear();
    m_outputPipeline = reloadFullscreenShader("Shaders/Post/passthrough",
                                              RhiFormat::BGRA8Unorm,
                                              &errorMessage);
    if (!m_outputPipeline.nativeHandle()) {
        spdlog::error("Failed to create output passthrough pipeline: {}",
                      formatError(&errorMessage, "Slang passthrough shader compilation failed"));
        return false;
    }

    errorMessage.clear();
    m_histogramPipeline = reloadComputeShader("Shaders/Post/auto_exposure",
                                              "histogramMain",
                                              nullptr,
                                              &errorMessage);
    if (!m_histogramPipeline.nativeHandle()) {
        spdlog::warn("Failed to compile histogram shader; auto-exposure disabled: {}",
                     formatError(&errorMessage, "Slang histogram shader compilation failed"));
    }

    errorMessage.clear();
    m_autoExposurePipeline = reloadComputeShader("Shaders/Post/auto_exposure",
                                                 "exposureMain",
                                                 nullptr,
                                                 &errorMessage);
    if (!m_autoExposurePipeline.nativeHandle()) {
        spdlog::warn("Failed to compile auto-exposure shader; auto-exposure disabled: {}",
                     formatError(&errorMessage, "Slang auto-exposure shader compilation failed"));
    }

    errorMessage.clear();
    m_taaPipeline = reloadComputeShader("Shaders/Post/taa",
                                        "taaMain",
                                        nullptr,
                                        &errorMessage);
    if (!m_taaPipeline.nativeHandle()) {
        spdlog::warn("Failed to compile TAA shader; TAA disabled: {}",
                     formatError(&errorMessage, "Slang TAA shader compilation failed"));
    }

    syncRuntimeContext();
    return true;
}

RhiGraphicsPipelineHandle ShaderManager::reloadVertexShader(const char* shaderPath, std::string* errorMessage) {
    std::string source = compileGraphics(shaderPath, m_projectRoot.c_str());
    if (source.empty()) {
        if (errorMessage) {
            *errorMessage = "Slang vertex shader compilation returned empty source";
        }
        return {};
    }

    RhiRenderPipelineSourceDesc pipelineDesc;
    pipelineDesc.vertexEntry = "vertexMain";
    pipelineDesc.fragmentEntry = "fragmentMain";
    pipelineDesc.colorFormat = RhiFormat::RGBA16Float;
    pipelineDesc.depthFormat = RhiFormat::D32Float;
    pipelineDesc.vertexDescriptor = &m_vertexDesc;

    std::string localError;
    RhiGraphicsPipelineHandle pipeline = rhiCreateRenderPipelineFromSource(m_device, source, pipelineDesc, localError);
    if (!pipeline.nativeHandle() && errorMessage) {
        *errorMessage = std::move(localError);
    }
    return pipeline;
}

RhiGraphicsPipelineHandle ShaderManager::reloadFullscreenShader(const char* shaderPath,
                                                                RhiFormat colorFormat,
                                                                std::string* errorMessage) {
    std::string source = compileGraphics(shaderPath, m_projectRoot.c_str());
    if (source.empty()) {
        if (errorMessage) {
            *errorMessage = "Slang fullscreen shader compilation returned empty source";
        }
        return {};
    }

    RhiRenderPipelineSourceDesc pipelineDesc;
    pipelineDesc.vertexEntry = "vertexMain";
    pipelineDesc.fragmentEntry = "fragmentMain";
    pipelineDesc.colorFormat = colorFormat;

    std::string localError;
    RhiGraphicsPipelineHandle pipeline = rhiCreateRenderPipelineFromSource(m_device, source, pipelineDesc, localError);
    if (!pipeline.nativeHandle() && errorMessage) {
        *errorMessage = std::move(localError);
    }
    return pipeline;
}

RhiGraphicsPipelineHandle ShaderManager::reloadMeshShader(const char* shaderPath,
                                                          std::string (*patchFn)(RhiBackendType, const std::string&),
                                                          RhiFormat colorFormat,
                                                          RhiFormat depthFormat,
                                                          std::string* errorMessage) {
    std::string source = compileMesh(shaderPath, m_projectRoot.c_str());
    if (source.empty()) {
        if (errorMessage) {
            *errorMessage = "Slang mesh shader compilation returned empty source";
        }
        return {};
    }

    if (patchFn) {
        source = patchFn(kShaderBackend, source);
    }

    RhiRenderPipelineSourceDesc pipelineDesc;
    pipelineDesc.meshEntry = "meshMain";
    pipelineDesc.fragmentEntry = "fragmentMain";
    pipelineDesc.colorFormat = colorFormat;
    pipelineDesc.depthFormat = depthFormat;

    std::string localError;
    RhiGraphicsPipelineHandle pipeline = rhiCreateRenderPipelineFromSource(m_device, source, pipelineDesc, localError);
    if (!pipeline.nativeHandle() && errorMessage) {
        *errorMessage = std::move(localError);
    }
    return pipeline;
}

RhiComputePipelineHandle ShaderManager::reloadComputeShader(const char* shaderPath,
                                                            const char* entryPoint,
                                                            std::string (*patchFn)(RhiBackendType, const std::string&),
                                                            std::string* errorMessage) {
    std::string source = compileCompute(shaderPath, m_projectRoot.c_str(), entryPoint);
    if (source.empty()) {
        if (errorMessage) {
            *errorMessage = "Slang compute shader compilation returned empty source";
        }
        return {};
    }

    if (patchFn) {
        source = patchFn(kShaderBackend, source);
    }

    std::string localError;
    RhiComputePipelineHandle pipeline = rhiCreateComputePipelineFromSource(m_device, source, entryPoint, localError);
    if (!pipeline.nativeHandle() && errorMessage) {
        *errorMessage = std::move(localError);
    }
    return pipeline;
}

std::pair<int, int> ShaderManager::reloadAll() {
    int reloaded = 0;
    int failed = 0;
    std::string errorMessage;

    auto vertexPipeline = reloadVertexShader("Shaders/Vertex/bunny", &errorMessage);
    if (vertexPipeline.nativeHandle()) {
        releaseOwnedHandle(m_vertexPipeline);
        m_vertexPipeline = vertexPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload vertex PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto meshPipeline = reloadMeshShader("Shaders/Mesh/meshlet",
                                         patchMeshShaderSource,
                                         RhiFormat::RGBA16Float,
                                         RhiFormat::D32Float,
                                         &errorMessage);
    if (meshPipeline.nativeHandle()) {
        releaseOwnedHandle(m_meshPipeline);
        m_meshPipeline = meshPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload mesh PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto visPipeline = reloadMeshShader("Shaders/Visibility/visibility",
                                        patchVisibilityShaderSource,
                                        RhiFormat::R32Uint,
                                        RhiFormat::D32Float,
                                        &errorMessage);
    if (visPipeline.nativeHandle()) {
        releaseOwnedHandle(m_visPipeline);
        m_visPipeline = visPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload visibility PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto visIndirectPipeline = reloadMeshShader("Shaders/Visibility/visibility_indirect",
                                                patchVisibilityShaderSource,
                                                RhiFormat::R32Uint,
                                                RhiFormat::D32Float,
                                                &errorMessage);
    if (visIndirectPipeline.nativeHandle()) {
        releaseOwnedHandle(m_visIndirectPipeline);
        m_visIndirectPipeline = visIndirectPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload visibility indirect PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto cullPipeline = reloadComputeShader("Shaders/Visibility/meshlet_cull",
                                            "computeMain",
                                            nullptr,
                                            &errorMessage);
    if (cullPipeline.nativeHandle()) {
        releaseOwnedHandle(m_cullPipeline);
        m_cullPipeline = cullPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload meshlet cull PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto buildIndirectPipeline = reloadComputeShader("Shaders/Visibility/build_indirect",
                                                     "computeMain",
                                                     nullptr,
                                                     &errorMessage);
    if (buildIndirectPipeline.nativeHandle()) {
        releaseOwnedHandle(m_buildIndirectPipeline);
        m_buildIndirectPipeline = buildIndirectPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload build indirect PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto deferredLightingPipeline = reloadComputeShader("Shaders/Visibility/deferred_lighting",
                                                        "computeMain",
                                                        patchComputeShaderSource,
                                                        &errorMessage);
    if (deferredLightingPipeline.nativeHandle()) {
        releaseOwnedHandle(m_computePipeline);
        m_computePipeline = deferredLightingPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload deferred lighting PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto meshletVisPipeline = reloadComputeShader("Shaders/Visibility/meshlet_visualize",
                                                  "computeMain",
                                                  nullptr,
                                                  &errorMessage);
    if (meshletVisPipeline.nativeHandle()) {
        releaseOwnedHandle(m_meshletVisPipeline);
        m_meshletVisPipeline = meshletVisPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload meshlet visualize PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto skyPipeline = reloadFullscreenShader("Shaders/Atmosphere/sky",
                                              RhiFormat::RGBA16Float,
                                              &errorMessage);
    if (skyPipeline.nativeHandle()) {
        releaseOwnedHandle(m_skyPipeline);
        m_skyPipeline = skyPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload sky PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto tonemapPipeline = reloadFullscreenShader("Shaders/Post/tonemap",
                                                  RhiFormat::BGRA8Unorm,
                                                  &errorMessage);
    if (tonemapPipeline.nativeHandle()) {
        releaseOwnedHandle(m_tonemapPipeline);
        m_tonemapPipeline = tonemapPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload tonemap PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto outputPipeline = reloadFullscreenShader("Shaders/Post/passthrough",
                                                 RhiFormat::BGRA8Unorm,
                                                 &errorMessage);
    if (outputPipeline.nativeHandle()) {
        releaseOwnedHandle(m_outputPipeline);
        m_outputPipeline = outputPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload output PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto histogramPipeline = reloadComputeShader("Shaders/Post/auto_exposure",
                                                 "histogramMain",
                                                 nullptr,
                                                 &errorMessage);
    if (histogramPipeline.nativeHandle()) {
        releaseOwnedHandle(m_histogramPipeline);
        m_histogramPipeline = histogramPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload histogram PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto autoExposurePipeline = reloadComputeShader("Shaders/Post/auto_exposure",
                                                    "exposureMain",
                                                    nullptr,
                                                    &errorMessage);
    if (autoExposurePipeline.nativeHandle()) {
        releaseOwnedHandle(m_autoExposurePipeline);
        m_autoExposurePipeline = autoExposurePipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload auto-exposure PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    errorMessage.clear();
    auto taaPipeline = reloadComputeShader("Shaders/Post/taa",
                                           "taaMain",
                                           nullptr,
                                           &errorMessage);
    if (taaPipeline.nativeHandle()) {
        releaseOwnedHandle(m_taaPipeline);
        m_taaPipeline = taaPipeline;
        reloaded++;
    } else {
        spdlog::error("Hot-reload TAA PSO: {}", formatError(&errorMessage, "Unknown failure"));
        failed++;
    }

    syncRuntimeContext();
    return {reloaded, failed};
}
