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
    bool visibility = true;
    bool visibilityIndirect = true;
    bool meshletCull = true;
    bool hzbBuild = true;
    bool buildIndirect = true;
    bool deferredLighting = true;
    bool meshletVisualize = true;
    bool sky = true;
    bool tonemap = true;
    bool output = true;
    bool autoExposure = true;
    bool taa = true;

    static ShaderManagerProfile full() { return {}; }

    static ShaderManagerProfile preview() {
        ShaderManagerProfile profile;
        profile.forwardVertex = true;
        profile.forwardMesh = false;
        profile.visibility = false;
        profile.visibilityIndirect = false;
        profile.meshletCull = false;
        profile.hzbBuild = false;
        profile.buildIndirect = false;
        profile.deferredLighting = false;
        profile.meshletVisualize = false;
        profile.output = false;
        profile.taa = false;
        return profile;
    }

    static ShaderManagerProfile vulkanVisibility() {
        ShaderManagerProfile profile;
        profile.forwardVertex = false;
        profile.forwardMesh = false;
        profile.visibility = true;
        profile.visibilityIndirect = true;
        profile.meshletCull = true;
        profile.hzbBuild = true;
        profile.buildIndirect = true;
        profile.deferredLighting = true;
        profile.meshletVisualize = false;
        profile.sky = true;
        profile.tonemap = true;
        profile.output = true;
        profile.autoExposure = true;
        profile.taa = true;
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
    RhiGraphicsPipelineHandle m_visPipeline;
    RhiGraphicsPipelineHandle m_visIndirectPipeline;
    RhiComputePipelineHandle m_computePipeline;
    RhiComputePipelineHandle m_instanceClassifyPipeline;
    RhiComputePipelineHandle m_cullPipeline;
    RhiComputePipelineHandle m_hzbBuildPipeline;
    RhiComputePipelineHandle m_buildIndirectPipeline;
    RhiComputePipelineHandle m_meshletVisPipeline;
    RhiGraphicsPipelineHandle m_skyPipeline;
    RhiGraphicsPipelineHandle m_tonemapPipeline;
    RhiGraphicsPipelineHandle m_outputPipeline;
    RhiComputePipelineHandle m_histogramPipeline;
    RhiComputePipelineHandle m_autoExposurePipeline;
    RhiComputePipelineHandle m_taaPipeline;
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
