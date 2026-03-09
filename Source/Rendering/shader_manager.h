#pragma once

#include <string>
#include <utility>

#include "rhi_backend.h"

struct PipelineRuntimeContext;

class ShaderManager {
public:
    ShaderManager(void* deviceHandle, const char* projectRoot);
    ~ShaderManager();

    // Initial creation of all pipelines + samplers. Returns false on fatal failure.
    bool buildAll();

    // F5 hot-reload (swap-on-success per pipeline). Returns {reloaded, failed}.
    std::pair<int,int> reloadAll();

    // Import external textures/samplers into the runtime context.
    void importTexture(const std::string& name, void* textureHandle);
    void importSampler(const std::string& name, void* samplerHandle);

    PipelineRuntimeContext& runtimeContext();
    bool hasSkyPipeline() const;

private:
    void* m_device = nullptr;
    std::string m_projectRoot;
    PipelineRuntimeContext* m_rtCtx;
    void* m_vertexDesc = nullptr;

    // Owned pipeline states
    void* m_vertexPipeline = nullptr;
    void* m_meshPipeline = nullptr;
    void* m_visPipeline = nullptr;
    void* m_visIndirectPipeline = nullptr;
    void* m_computePipeline = nullptr;
    void* m_cullPipeline = nullptr;
    void* m_buildIndirectPipeline = nullptr;
    void* m_meshletVisPipeline = nullptr;
    void* m_skyPipeline = nullptr;
    void* m_tonemapPipeline = nullptr;
    void* m_outputPipeline = nullptr;
    void* m_histogramPipeline = nullptr;
    void* m_autoExposurePipeline = nullptr;
    void* m_taaPipeline = nullptr;
    void* m_tonemapSampler = nullptr;

    void createVertexDescriptor();
    void syncRuntimeContext();

    // Internal reload helpers (return new PSO on success, nullptr on failure)
    void* reloadVertexShader(const char* shaderPath);
    void* reloadFullscreenShader(const char* shaderPath, RhiFormat colorFormat);
    void* reloadMeshShader(
        const char* shaderPath,
        std::string (*patchFn)(const std::string&),
        RhiFormat colorFormat, RhiFormat depthFormat);
    void* reloadComputeShader(
        const char* shaderPath, const char* entryPoint,
        std::string (*patchFn)(const std::string&));
};
