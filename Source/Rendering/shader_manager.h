#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <utility>

struct PipelineRuntimeContext;

class ShaderManager {
public:
    ShaderManager(MTL::Device* device, const char* projectRoot);
    ~ShaderManager();

    // Initial creation of all pipelines + samplers. Returns false on fatal failure.
    bool buildAll();

    // F5 hot-reload (swap-on-success per pipeline). Returns {reloaded, failed}.
    std::pair<int,int> reloadAll();

    // Import external textures/samplers into the runtime context.
    void importTexture(const std::string& name, MTL::Texture* tex);
    void importSampler(const std::string& name, MTL::SamplerState* sampler);

    PipelineRuntimeContext& runtimeContext();
    bool hasSkyPipeline() const;

private:
    MTL::Device* m_device;
    std::string m_projectRoot;
    PipelineRuntimeContext* m_rtCtx;
    MTL::VertexDescriptor* m_vertexDesc = nullptr;

    // Owned pipeline states
    MTL::RenderPipelineState* m_vertexPipeline = nullptr;
    MTL::RenderPipelineState* m_meshPipeline = nullptr;
    MTL::RenderPipelineState* m_visPipeline = nullptr;
    MTL::RenderPipelineState* m_visIndirectPipeline = nullptr;
    MTL::ComputePipelineState* m_computePipeline = nullptr;
    MTL::ComputePipelineState* m_cullPipeline = nullptr;
    MTL::ComputePipelineState* m_buildIndirectPipeline = nullptr;
    MTL::RenderPipelineState* m_skyPipeline = nullptr;
    MTL::RenderPipelineState* m_tonemapPipeline = nullptr;
    MTL::RenderPipelineState* m_outputPipeline = nullptr;
    MTL::SamplerState* m_tonemapSampler = nullptr;

    void createVertexDescriptor();
    void syncRuntimeContext();

    // Internal reload helpers (return new PSO on success, nullptr on failure)
    MTL::RenderPipelineState* reloadVertexShader(const char* shaderPath);
    MTL::RenderPipelineState* reloadFullscreenShader(const char* shaderPath, MTL::PixelFormat colorFormat);
    MTL::RenderPipelineState* reloadMeshShader(
        const char* shaderPath,
        std::string (*patchFn)(const std::string&),
        MTL::PixelFormat colorFormat, MTL::PixelFormat depthFormat);
    MTL::ComputePipelineState* reloadComputeShader(
        const char* shaderPath, const char* entryPoint,
        std::string (*patchFn)(const std::string&));
};
