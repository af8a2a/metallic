#pragma once

#include <string>
#include <utility>
#include <vector>

#include "rhi_backend.h"

struct PipelineRuntimeContext;
struct SlangCompileOptions;

enum class ShaderCompileMode {
    Release,  // optimized=true,  generateDebugInfo=false
    Debug,    // optimized=false, generateDebugInfo=true
};

struct ShaderManagerProfile {
    bool forwardVertex = true;
    bool forwardMesh = true;
    bool sky = true;
    bool tonemap = true;
    bool output = true;
    bool autoExposure = true;
    bool taa = true;
    bool clusterRender = true;
    bool clusterOcclusion = true;

    static ShaderManagerProfile full() { return {}; }

    static ShaderManagerProfile preview() {
        ShaderManagerProfile profile;
        profile.forwardVertex = true;
        profile.forwardMesh = false;
        profile.output = false;
        profile.taa = false;
        profile.clusterOcclusion = false;
        return profile;
    }

    static ShaderManagerProfile vulkanVisibility() {
        ShaderManagerProfile profile;
        profile.forwardVertex = false;
        profile.forwardMesh = false;
        profile.sky = false;
        profile.tonemap = true;
        profile.output = true;
        profile.autoExposure = false;
        profile.taa = false;
        profile.clusterRender = true;
        profile.clusterOcclusion = true;
        return profile;
    }
};

class ShaderManager {
public:
    ShaderManager(RhiDeviceHandle device,
                  const char* projectRoot,
                  bool supportsMeshShaders = true,
                  bool validateVisibilityPipelines = true,
                  ShaderManagerProfile profile = ShaderManagerProfile::full(),
                  ShaderCompileMode compileMode = ShaderCompileMode::Release);
    ~ShaderManager();

    // Initial creation of all pipelines + samplers. Returns false on fatal failure.
    bool buildAll();

    // F5 hot-reload (swap-on-success per pipeline). Returns {reloaded, failed}.
    std::pair<int,int> reloadAll();

    // Import external textures/samplers into the runtime context.
    void importTexture(const std::string& name, const RhiTexture& texture);
    void importSampler(const std::string& name, const RhiSampler& sampler);

    // Engine-wide preprocessor defines applied to all shader compilations.
    void setGlobalDefines(const std::vector<std::pair<std::string, std::string>>& defines);

    // Change compile mode (takes effect on next buildAll/reloadAll).
    void setCompileMode(ShaderCompileMode mode);
    ShaderCompileMode compileMode() const;

    PipelineRuntimeContext& runtimeContext();
    bool hasSkyPipeline() const;

private:
    RhiDeviceHandle m_device;
    std::string m_projectRoot;
    bool m_supportsMeshShaders = true;
    bool m_validateVisibilityPipelines = true;
    ShaderManagerProfile m_profile;
    ShaderCompileMode m_compileMode = ShaderCompileMode::Release;
    std::vector<std::pair<std::string, std::string>> m_globalDefines;
    PipelineRuntimeContext* m_rtCtx;
    RhiVertexDescriptorHandle m_vertexDesc;

    // Owned pipeline states
    RhiGraphicsPipelineHandle m_vertexPipeline;
    RhiGraphicsPipelineHandle m_meshPipeline;
    RhiGraphicsPipelineHandle m_skyPipeline;
    RhiGraphicsPipelineHandle m_tonemapPipeline;
    RhiGraphicsPipelineHandle m_outputPipeline;
    RhiComputePipelineHandle m_histogramPipeline;
    RhiComputePipelineHandle m_autoExposurePipeline;
    RhiComputePipelineHandle m_taaPipeline;
    RhiComputePipelineHandle m_clusterCullResetPipeline;
    RhiComputePipelineHandle m_clusterCullMainPipeline;
    RhiComputePipelineHandle m_clusterCullFinalizePipeline;
    RhiComputePipelineHandle m_dagClusterCullResetPipeline;
    RhiComputePipelineHandle m_dagClusterCullMainPipeline;
    RhiComputePipelineHandle m_dagClusterCullFinalizePipeline;
    RhiComputePipelineHandle m_clusterHizBuildPipeline;
    RhiComputePipelineHandle m_instanceCullResetPipeline;
    RhiComputePipelineHandle m_instanceCullMainPipeline;
    RhiComputePipelineHandle m_instanceCullFinalizePipeline;
    RhiGraphicsPipelineHandle m_clusterRenderPipeline;
    RhiSamplerHandle m_tonemapSampler;

    void createVertexDescriptor();
    void syncRuntimeContext();

    // Internal reload helpers (return empty handle on failure)
    RhiGraphicsPipelineHandle reloadVertexShader(const char* shaderPath, std::string* errorMessage = nullptr);
    RhiGraphicsPipelineHandle reloadFullscreenShader(const char* shaderPath,
                                                     RhiFormat colorFormat,
                                                     std::string* errorMessage = nullptr);
    RhiGraphicsPipelineHandle reloadMeshShader(
        const char* shaderPath,
        std::string (*patchFn)(RhiBackendType, const std::string&),
        RhiFormat colorFormat, RhiFormat depthFormat,
        std::string* errorMessage = nullptr);
    RhiComputePipelineHandle reloadComputeShader(
        const char* shaderPath, const char* entryPoint,
        std::string (*patchFn)(RhiBackendType, const std::string&),
        std::string* errorMessage = nullptr);
};
