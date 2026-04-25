#include "shader_manager.h"

#include "frame_context.h"
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

constexpr const char* kSlangVisibilityPerPrimitiveIssueUrl =
    "https://github.com/shader-slang/slang/issues/7019";

// On Vulkan, compile to SPIR-V binary and pack into a string for rhiCreatePipelineFromSource.
// On Metal, compile to MSL source string directly.
std::string compileGraphics(const char* shaderPath, const char* projectRoot,
                            const SlangCompileOptions* options) {
#ifdef __APPLE__
    return compileSlangGraphicsSource(kShaderBackend, shaderPath, projectRoot, options);
#else
    auto spirv = compileSlangGraphicsBinary(kShaderBackend, shaderPath, projectRoot, options);
    if (spirv.empty()) return {};
    return std::string(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(uint32_t));
#endif
}

std::string compileMesh(const char* shaderPath, const char* projectRoot,
                        const SlangCompileOptions* options) {
#ifdef __APPLE__
    return compileSlangMeshSource(kShaderBackend, shaderPath, projectRoot, options);
#else
    auto spirv = compileSlangMeshBinary(kShaderBackend, shaderPath, projectRoot, options);
    if (spirv.empty()) return {};
    return std::string(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(uint32_t));
#endif
}

std::string compileCompute(const char* shaderPath, const char* projectRoot,
                           const char* entryPoint, const SlangCompileOptions* options) {
#ifdef __APPLE__
    return compileSlangComputeSource(kShaderBackend, shaderPath, projectRoot, entryPoint, options);
#else
    auto spirv = compileSlangComputeBinary(kShaderBackend, shaderPath, projectRoot, entryPoint, options);
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

ShaderManager::ShaderManager(RhiDeviceHandle device,
                             const char* projectRoot,
                             bool supportsMeshShaders,
                             bool validateVisibilityPipelines,
                             ShaderManagerProfile profile,
                             ShaderCompileMode compileMode)
    : m_device(device)
    , m_projectRoot(projectRoot)
    , m_supportsMeshShaders(supportsMeshShaders)
    , m_validateVisibilityPipelines(validateVisibilityPipelines)
    , m_profile(profile)
    , m_compileMode(compileMode)
    , m_rtCtx(new PipelineRuntimeContext{})
{}

ShaderManager::~ShaderManager() {
    releaseOwnedHandle(m_vertexPipeline);
    releaseOwnedHandle(m_meshPipeline);
    releaseOwnedHandle(m_visPipeline);
    releaseOwnedHandle(m_visIndirectPipeline);
    releaseOwnedHandle(m_computePipeline);
    releaseOwnedHandle(m_clusterStreamingUpdatePipeline);
    releaseOwnedHandle(m_instanceClassifyPipeline);
    releaseOwnedHandle(m_cullPipeline);
    releaseOwnedHandle(m_clusterStreamingAgeFilterPipeline);
    releaseOwnedHandle(m_hzbBuildPipeline);
    releaseOwnedHandle(m_buildIndirectPipeline);
    releaseOwnedHandle(m_meshletVisPipeline);
    releaseOwnedHandle(m_skyPipeline);
    releaseOwnedHandle(m_tonemapPipeline);
    releaseOwnedHandle(m_outputPipeline);
    releaseOwnedHandle(m_histogramPipeline);
    releaseOwnedHandle(m_autoExposurePipeline);
    releaseOwnedHandle(m_taaPipeline);
    releaseOwnedHandle(m_clusterRenderPipeline);
    releaseOwnedHandle(m_tonemapSampler);
    releaseOwnedHandle(m_vertexDesc);
    delete m_rtCtx;
}

PipelineRuntimeContext& ShaderManager::runtimeContext() { return *m_rtCtx; }
bool ShaderManager::hasSkyPipeline() const { return m_skyPipeline.nativeHandle() != nullptr; }

void ShaderManager::setGlobalDefines(const std::vector<std::pair<std::string, std::string>>& defines) {
    m_globalDefines = defines;
}

void ShaderManager::setCompileMode(ShaderCompileMode mode) {
    m_compileMode = mode;
}

ShaderCompileMode ShaderManager::compileMode() const {
    return m_compileMode;
}

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
    m_rtCtx->renderPipelinesRhi.clear();
    if (m_vertexPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["ForwardPass"] = m_vertexPipeline;
    if (m_meshPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["ForwardMeshPass"] = m_meshPipeline;
    if (m_visPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["VisibilityPass"] = m_visPipeline;
    if (m_visIndirectPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["VisibilityIndirectPass"] = m_visIndirectPipeline;
    if (m_clusterRenderPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["ClusterRenderPass"] = m_clusterRenderPipeline;
    if (m_skyPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["SkyPass"] = m_skyPipeline;
    if (m_tonemapPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["TonemapPass"] = m_tonemapPipeline;
    if (m_outputPipeline.nativeHandle())
        m_rtCtx->renderPipelinesRhi["OutputPass"] = m_outputPipeline;

    m_rtCtx->computePipelinesRhi.clear();
    if (m_computePipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["DeferredLightingPass"] = m_computePipeline;
    if (m_clusterStreamingUpdatePipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["ClusterStreamingUpdatePass"] =
            m_clusterStreamingUpdatePipeline;
    if (m_instanceClassifyPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["InstanceClassifyPass"] = m_instanceClassifyPipeline;
    if (m_cullPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["MeshletCullPass"] = m_cullPipeline;
    if (m_clusterStreamingAgeFilterPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["ClusterStreamingAgeFilterPass"] =
            m_clusterStreamingAgeFilterPipeline;
    if (m_hzbBuildPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["HZBBuildPass"] = m_hzbBuildPipeline;
    if (m_buildIndirectPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["BuildIndirectPass"] = m_buildIndirectPipeline;
    if (m_meshletVisPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["MeshletVisualizePass"] = m_meshletVisPipeline;
    if (m_histogramPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["HistogramPass"] = m_histogramPipeline;
    if (m_autoExposurePipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["AutoExposurePass"] = m_autoExposurePipeline;
    if (m_taaPipeline.nativeHandle())
        m_rtCtx->computePipelinesRhi["TAAPass"] = m_taaPipeline;

    if (m_tonemapSampler.nativeHandle())
        m_rtCtx->samplersRhi["tonemap"] = m_tonemapSampler;
    else
        m_rtCtx->samplersRhi.erase("tonemap");
}

bool ShaderManager::buildAll() {
    createVertexDescriptor();

    std::string errorMessage;

    if (m_profile.forwardVertex) {
        m_vertexPipeline = reloadVertexShader("Shaders/Vertex/bunny", &errorMessage);
        if (!m_vertexPipeline.nativeHandle()) {
            spdlog::error("Failed to create vertex pipeline: {}",
                          formatError(&errorMessage, "Slang vertex shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_vertexPipeline);
    }

    errorMessage.clear();
    if (m_profile.forwardMesh && m_supportsMeshShaders) {
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
    } else if (m_profile.forwardMesh) {
        spdlog::info("Skipping mesh pipeline validation because mesh shaders are not supported");
        releaseOwnedHandle(m_meshPipeline);
    } else {
        releaseOwnedHandle(m_meshPipeline);
    }

    errorMessage.clear();
    if (m_profile.visibility && m_supportsMeshShaders && m_validateVisibilityPipelines) {
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
    } else if (m_profile.visibility && !m_supportsMeshShaders) {
        spdlog::info("Skipping visibility pipeline validation because mesh shaders are not supported");
        releaseOwnedHandle(m_visPipeline);
    } else if (m_profile.visibility) {
        spdlog::info("Skipping visibility pipeline validation on Vulkan due to Slang PerPrimitiveEXT blocker: {}",
                     kSlangVisibilityPerPrimitiveIssueUrl);
        releaseOwnedHandle(m_visPipeline);
    } else {
        releaseOwnedHandle(m_visPipeline);
    }

    errorMessage.clear();
    if (m_profile.visibilityIndirect && m_supportsMeshShaders && m_validateVisibilityPipelines) {
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
    } else if (m_profile.visibilityIndirect && !m_supportsMeshShaders) {
        spdlog::info("Skipping visibility indirect pipeline validation because mesh shaders are not supported");
        releaseOwnedHandle(m_visIndirectPipeline);
    } else if (m_profile.visibilityIndirect) {
        spdlog::info("Skipping visibility indirect pipeline validation on Vulkan due to Slang PerPrimitiveEXT blocker: {}",
                     kSlangVisibilityPerPrimitiveIssueUrl);
        releaseOwnedHandle(m_visIndirectPipeline);
    } else {
        releaseOwnedHandle(m_visIndirectPipeline);
    }

    errorMessage.clear();
    if (m_profile.clusterRender && m_supportsMeshShaders) {
        m_clusterRenderPipeline = reloadMeshShader("Shaders/Mesh/cluster_render",
                                                    nullptr,
                                                    RhiFormat::RGBA8Unorm,
                                                    RhiFormat::D32Float,
                                                    &errorMessage);
        if (!m_clusterRenderPipeline.nativeHandle()) {
            spdlog::warn("Failed to create cluster render pipeline: {}",
                         formatError(&errorMessage, "Slang cluster render shader compilation failed"));
        }
    } else {
        releaseOwnedHandle(m_clusterRenderPipeline);
    }

    errorMessage.clear();
    if (m_profile.meshletCull) {
        m_clusterStreamingUpdatePipeline =
            reloadComputeShader("Shaders/Streaming/stream_update_scene",
                                "computeMain",
                                nullptr,
                                &errorMessage);
        if (!m_clusterStreamingUpdatePipeline.nativeHandle()) {
            spdlog::error("Failed to create cluster streaming update pipeline: {}",
                          formatError(&errorMessage,
                                      "Slang cluster streaming update shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_clusterStreamingUpdatePipeline);
    }

    errorMessage.clear();
    if (m_profile.meshletCull) {
        m_instanceClassifyPipeline = reloadComputeShader("Shaders/Visibility/instance_classify",
                                                         "computeMain",
                                                         nullptr,
                                                         &errorMessage);
        if (!m_instanceClassifyPipeline.nativeHandle()) {
            spdlog::error("Failed to create instance classify pipeline: {}",
                          formatError(&errorMessage, "Slang instance classify shader compilation failed"));
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
        m_clusterStreamingAgeFilterPipeline =
            reloadComputeShader("Shaders/Streaming/stream_agefilter_groups",
                                "computeMain",
                                nullptr,
                                &errorMessage);
        if (!m_clusterStreamingAgeFilterPipeline.nativeHandle()) {
            spdlog::error("Failed to create cluster streaming age filter pipeline: {}",
                          formatError(&errorMessage,
                                      "Slang cluster streaming age filter shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_instanceClassifyPipeline);
        releaseOwnedHandle(m_cullPipeline);
        releaseOwnedHandle(m_clusterStreamingAgeFilterPipeline);
    }

    errorMessage.clear();
    if (m_profile.hzbBuild) {
        m_hzbBuildPipeline = reloadComputeShader("Shaders/Visibility/hzb_build",
                                                 "computeMain",
                                                 nullptr,
                                                 &errorMessage);
        if (!m_hzbBuildPipeline.nativeHandle()) {
            spdlog::error("Failed to create HZB build pipeline: {}",
                          formatError(&errorMessage, "Slang HZB build shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_hzbBuildPipeline);
    }

    errorMessage.clear();
    if (m_profile.buildIndirect) {
        m_buildIndirectPipeline = reloadComputeShader("Shaders/Visibility/build_indirect",
                                                      "computeMain",
                                                      nullptr,
                                                      &errorMessage);
        if (!m_buildIndirectPipeline.nativeHandle()) {
            spdlog::error("Failed to create build indirect pipeline: {}",
                          formatError(&errorMessage, "Slang build indirect shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_buildIndirectPipeline);
    }

    errorMessage.clear();
    if (m_profile.deferredLighting) {
        m_computePipeline = reloadComputeShader("Shaders/Visibility/deferred_lighting",
                                                "computeMain",
                                                patchComputeShaderSource,
                                                &errorMessage);
        if (!m_computePipeline.nativeHandle()) {
            spdlog::error("Failed to create deferred lighting pipeline: {}",
                          formatError(&errorMessage, "Slang deferred lighting shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_computePipeline);
    }

    errorMessage.clear();
    if (m_profile.meshletVisualize) {
        m_meshletVisPipeline = reloadComputeShader("Shaders/Visibility/meshlet_visualize",
                                                   "computeMain",
                                                   nullptr,
                                                   &errorMessage);
        if (!m_meshletVisPipeline.nativeHandle()) {
            spdlog::warn("Failed to compile meshlet visualize shader; visualization disabled: {}",
                         formatError(&errorMessage, "Slang meshlet visualize shader compilation failed"));
        }
    } else {
        releaseOwnedHandle(m_meshletVisPipeline);
    }

    errorMessage.clear();
    if (m_profile.sky) {
        m_skyPipeline = reloadFullscreenShader("Shaders/Atmosphere/sky",
                                               RhiFormat::RGBA16Float,
                                               &errorMessage);
        if (!m_skyPipeline.nativeHandle()) {
            spdlog::warn("Sky shader compile failed; atmosphere sky disabled: {}",
                         formatError(&errorMessage, "Slang sky shader compilation failed"));
        }
    } else {
        releaseOwnedHandle(m_skyPipeline);
    }

    errorMessage.clear();
    if (m_profile.tonemap) {
        m_tonemapPipeline = reloadFullscreenShader("Shaders/Post/tonemap",
                                                   RhiFormat::RGBA8Srgb,
                                                   &errorMessage);
        if (!m_tonemapPipeline.nativeHandle()) {
            spdlog::error("Failed to create tonemap pipeline: {}",
                          formatError(&errorMessage, "Slang tonemap shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_tonemapPipeline);
    }

    if (m_profile.tonemap || m_profile.output) {
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
    } else {
        releaseOwnedHandle(m_tonemapSampler);
    }

    errorMessage.clear();
    if (m_profile.output) {
        m_outputPipeline = reloadFullscreenShader("Shaders/Post/passthrough",
                                                  RhiFormat::BGRA8Unorm,
                                                  &errorMessage);
        if (!m_outputPipeline.nativeHandle()) {
            spdlog::error("Failed to create output passthrough pipeline: {}",
                          formatError(&errorMessage, "Slang passthrough shader compilation failed"));
            return false;
        }
    } else {
        releaseOwnedHandle(m_outputPipeline);
    }

    errorMessage.clear();
    if (m_profile.autoExposure) {
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
    } else {
        releaseOwnedHandle(m_histogramPipeline);
        releaseOwnedHandle(m_autoExposurePipeline);
    }

    errorMessage.clear();
    if (m_profile.taa) {
        m_taaPipeline = reloadComputeShader("Shaders/Post/taa",
                                            "taaMain",
                                            nullptr,
                                            &errorMessage);
        if (!m_taaPipeline.nativeHandle()) {
            spdlog::warn("Failed to compile TAA shader; TAA disabled: {}",
                         formatError(&errorMessage, "Slang TAA shader compilation failed"));
        }
    } else {
        releaseOwnedHandle(m_taaPipeline);
    }

    syncRuntimeContext();
    return true;
}

RhiGraphicsPipelineHandle ShaderManager::reloadVertexShader(const char* shaderPath, std::string* errorMessage) {
    SlangCompileOptions opts;
    opts.optimized = (m_compileMode == ShaderCompileMode::Release);
    opts.generateDebugInfo = (m_compileMode == ShaderCompileMode::Debug);
    opts.defines = m_globalDefines;

    std::string source = compileGraphics(shaderPath, m_projectRoot.c_str(), &opts);
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
    SlangCompileOptions opts;
    opts.optimized = (m_compileMode == ShaderCompileMode::Release);
    opts.generateDebugInfo = (m_compileMode == ShaderCompileMode::Debug);
    opts.defines = m_globalDefines;

    std::string source = compileGraphics(shaderPath, m_projectRoot.c_str(), &opts);
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
    SlangCompileOptions opts;
    opts.optimized = (m_compileMode == ShaderCompileMode::Release);
    opts.generateDebugInfo = (m_compileMode == ShaderCompileMode::Debug);
    opts.defines = m_globalDefines;

    std::string source = compileMesh(shaderPath, m_projectRoot.c_str(), &opts);
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
    SlangCompileOptions opts;
    opts.optimized = (m_compileMode == ShaderCompileMode::Release);
    opts.generateDebugInfo = (m_compileMode == ShaderCompileMode::Debug);
    opts.defines = m_globalDefines;

    std::string source = compileCompute(shaderPath, m_projectRoot.c_str(), entryPoint, &opts);
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

    auto reloadPipeline = [&](bool enabled, auto& target, const char* label, auto&& reloadFn) {
        if (!enabled) {
            releaseOwnedHandle(target);
            return;
        }

        errorMessage.clear();
        auto pipeline = reloadFn(errorMessage);
        if (pipeline.nativeHandle()) {
            releaseOwnedHandle(target);
            target = pipeline;
            reloaded++;
        } else {
            spdlog::error("Hot-reload {}: {}", label, formatError(&errorMessage, "Unknown failure"));
            failed++;
        }
    };

    reloadPipeline(m_profile.forwardVertex,
                   m_vertexPipeline,
                   "vertex PSO",
                   [&](std::string& localError) {
                       return reloadVertexShader("Shaders/Vertex/bunny", &localError);
                   });

    if (m_profile.forwardMesh) {
        if (m_supportsMeshShaders) {
            reloadPipeline(true,
                           m_meshPipeline,
                           "mesh PSO",
                           [&](std::string& localError) {
                               return reloadMeshShader("Shaders/Mesh/meshlet",
                                                       patchMeshShaderSource,
                                                       RhiFormat::RGBA16Float,
                                                       RhiFormat::D32Float,
                                                       &localError);
                           });
        } else {
            spdlog::info("Skipping mesh PSO hot-reload because mesh shaders are not supported");
            releaseOwnedHandle(m_meshPipeline);
        }
    } else {
        releaseOwnedHandle(m_meshPipeline);
    }

    if (m_profile.visibility) {
        if (!m_supportsMeshShaders) {
            spdlog::info("Skipping visibility PSO hot-reload because mesh shaders are not supported");
            releaseOwnedHandle(m_visPipeline);
        } else if (!m_validateVisibilityPipelines) {
            spdlog::info("Skipping visibility PSO hot-reload on Vulkan due to Slang PerPrimitiveEXT blocker: {}",
                         kSlangVisibilityPerPrimitiveIssueUrl);
            releaseOwnedHandle(m_visPipeline);
        } else {
            reloadPipeline(true,
                           m_visPipeline,
                           "visibility PSO",
                           [&](std::string& localError) {
                               return reloadMeshShader("Shaders/Visibility/visibility",
                                                       patchVisibilityShaderSource,
                                                       RhiFormat::R32Uint,
                                                       RhiFormat::D32Float,
                                                       &localError);
                           });
        }
    } else {
        releaseOwnedHandle(m_visPipeline);
    }

    if (m_profile.visibilityIndirect) {
        if (!m_supportsMeshShaders) {
            spdlog::info("Skipping visibility indirect PSO hot-reload because mesh shaders are not supported");
            releaseOwnedHandle(m_visIndirectPipeline);
        } else if (!m_validateVisibilityPipelines) {
            spdlog::info("Skipping visibility indirect PSO hot-reload on Vulkan due to Slang PerPrimitiveEXT blocker: {}",
                         kSlangVisibilityPerPrimitiveIssueUrl);
            releaseOwnedHandle(m_visIndirectPipeline);
        } else {
            reloadPipeline(true,
                           m_visIndirectPipeline,
                           "visibility indirect PSO",
                           [&](std::string& localError) {
                               return reloadMeshShader("Shaders/Visibility/visibility_indirect",
                                                       patchVisibilityShaderSource,
                                                       RhiFormat::R32Uint,
                                                       RhiFormat::D32Float,
                                                       &localError);
                           });
        }
    } else {
        releaseOwnedHandle(m_visIndirectPipeline);
    }

    if (m_profile.clusterRender && m_supportsMeshShaders) {
        reloadPipeline(true,
                       m_clusterRenderPipeline,
                       "cluster render PSO",
                       [&](std::string& localError) {
                           return reloadMeshShader("Shaders/Mesh/cluster_render",
                                                   nullptr,
                                                   RhiFormat::RGBA8Unorm,
                                                   RhiFormat::D32Float,
                                                   &localError);
                       });
    } else {
        releaseOwnedHandle(m_clusterRenderPipeline);
    }

    reloadPipeline(m_profile.meshletCull,
                   m_clusterStreamingUpdatePipeline,
                   "cluster streaming update PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Streaming/stream_update_scene",
                                                  "computeMain",
                                                  nullptr,
                                                  &localError);
                   });

    reloadPipeline(m_profile.meshletCull,
                   m_instanceClassifyPipeline,
                   "instance classify PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Visibility/instance_classify",
                                                  "computeMain",
                                                  nullptr,
                                                  &localError);
                   });

    reloadPipeline(m_profile.meshletCull,
                   m_cullPipeline,
                   "meshlet cull PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Visibility/meshlet_cull",
                                                  "computeMain",
                                                  nullptr,
                                                  &localError);
                   });

    reloadPipeline(m_profile.meshletCull,
                   m_clusterStreamingAgeFilterPipeline,
                   "cluster streaming age filter PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Streaming/stream_agefilter_groups",
                                                  "computeMain",
                                                  nullptr,
                                                  &localError);
                   });

    reloadPipeline(m_profile.hzbBuild,
                   m_hzbBuildPipeline,
                   "HZB build PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Visibility/hzb_build",
                                                  "computeMain",
                                                  nullptr,
                                                  &localError);
                   });

    reloadPipeline(m_profile.buildIndirect,
                   m_buildIndirectPipeline,
                   "build indirect PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Visibility/build_indirect",
                                                  "computeMain",
                                                  nullptr,
                                                  &localError);
                   });

    reloadPipeline(m_profile.deferredLighting,
                   m_computePipeline,
                   "deferred lighting PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Visibility/deferred_lighting",
                                                  "computeMain",
                                                  patchComputeShaderSource,
                                                  &localError);
                   });

    reloadPipeline(m_profile.meshletVisualize,
                   m_meshletVisPipeline,
                   "meshlet visualize PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Visibility/meshlet_visualize",
                                                  "computeMain",
                                                  nullptr,
                                                  &localError);
                   });

    reloadPipeline(m_profile.sky,
                   m_skyPipeline,
                   "sky PSO",
                   [&](std::string& localError) {
                       return reloadFullscreenShader("Shaders/Atmosphere/sky",
                                                     RhiFormat::RGBA16Float,
                                                     &localError);
                   });

    reloadPipeline(m_profile.tonemap,
                   m_tonemapPipeline,
                   "tonemap PSO",
                   [&](std::string& localError) {
                       return reloadFullscreenShader("Shaders/Post/tonemap",
                                                     RhiFormat::RGBA8Srgb,
                                                     &localError);
                   });

    if (m_profile.tonemap || m_profile.output) {
        if (!m_tonemapSampler.nativeHandle()) {
            RhiSamplerDesc samplerDesc;
            samplerDesc.minFilter = RhiSamplerFilterMode::Linear;
            samplerDesc.magFilter = RhiSamplerFilterMode::Linear;
            samplerDesc.mipFilter = RhiSamplerMipFilterMode::None;
            samplerDesc.addressModeS = RhiSamplerAddressMode::ClampToEdge;
            samplerDesc.addressModeT = RhiSamplerAddressMode::ClampToEdge;
            m_tonemapSampler = rhiCreateSampler(m_device, samplerDesc);
            if (!m_tonemapSampler.nativeHandle()) {
                spdlog::error("Hot-reload tonemap sampler: Failed to create tonemap sampler state");
                failed++;
            }
        }
    } else {
        releaseOwnedHandle(m_tonemapSampler);
    }

    reloadPipeline(m_profile.output,
                   m_outputPipeline,
                   "output PSO",
                   [&](std::string& localError) {
                       return reloadFullscreenShader("Shaders/Post/passthrough",
                                                     RhiFormat::BGRA8Unorm,
                                                     &localError);
                   });

    if (m_profile.autoExposure) {
        reloadPipeline(true,
                       m_histogramPipeline,
                       "histogram PSO",
                       [&](std::string& localError) {
                           return reloadComputeShader("Shaders/Post/auto_exposure",
                                                      "histogramMain",
                                                      nullptr,
                                                      &localError);
                       });

        reloadPipeline(true,
                       m_autoExposurePipeline,
                       "auto-exposure PSO",
                       [&](std::string& localError) {
                           return reloadComputeShader("Shaders/Post/auto_exposure",
                                                      "exposureMain",
                                                      nullptr,
                                                      &localError);
                       });
    } else {
        releaseOwnedHandle(m_histogramPipeline);
        releaseOwnedHandle(m_autoExposurePipeline);
    }

    reloadPipeline(m_profile.taa,
                   m_taaPipeline,
                   "TAA PSO",
                   [&](std::string& localError) {
                       return reloadComputeShader("Shaders/Post/taa",
                                                  "taaMain",
                                                  nullptr,
                                                  &localError);
                   });

    syncRuntimeContext();
    return {reloaded, failed};
}
