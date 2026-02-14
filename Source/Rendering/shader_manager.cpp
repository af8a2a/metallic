#include "shader_manager.h"
#include "slang_compiler.h"
#include "frame_context.h"

#include <Foundation/Foundation.hpp>
#include <spdlog/spdlog.h>

ShaderManager::ShaderManager(MTL::Device* device, const char* projectRoot)
    : m_device(device)
    , m_projectRoot(projectRoot)
    , m_rtCtx(new PipelineRuntimeContext{})
{
    m_rtCtx->device = device;
}

ShaderManager::~ShaderManager() {
    if (m_vertexPipeline) m_vertexPipeline->release();
    if (m_meshPipeline) m_meshPipeline->release();
    if (m_visPipeline) m_visPipeline->release();
    if (m_visIndirectPipeline) m_visIndirectPipeline->release();
    if (m_computePipeline) m_computePipeline->release();
    if (m_cullPipeline) m_cullPipeline->release();
    if (m_buildIndirectPipeline) m_buildIndirectPipeline->release();
    if (m_skyPipeline) m_skyPipeline->release();
    if (m_tonemapPipeline) m_tonemapPipeline->release();
    if (m_outputPipeline) m_outputPipeline->release();
    if (m_tonemapSampler) m_tonemapSampler->release();
    if (m_vertexDesc) m_vertexDesc->release();
    delete m_rtCtx;
}

PipelineRuntimeContext& ShaderManager::runtimeContext() { return *m_rtCtx; }
bool ShaderManager::hasSkyPipeline() const { return m_skyPipeline != nullptr; }

void ShaderManager::importTexture(const std::string& name, MTL::Texture* tex) {
    m_rtCtx->importedTextures[name] = tex;
}

void ShaderManager::importSampler(const std::string& name, MTL::SamplerState* sampler) {
    m_rtCtx->samplers[name] = sampler;
}

void ShaderManager::createVertexDescriptor() {
    m_vertexDesc = MTL::VertexDescriptor::alloc()->init();
    // attribute(0) = position: float3 from buffer 1
    m_vertexDesc->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
    m_vertexDesc->attributes()->object(0)->setOffset(0);
    m_vertexDesc->attributes()->object(0)->setBufferIndex(1);
    // attribute(1) = normal: float3 from buffer 2
    m_vertexDesc->attributes()->object(1)->setFormat(MTL::VertexFormatFloat3);
    m_vertexDesc->attributes()->object(1)->setOffset(0);
    m_vertexDesc->attributes()->object(1)->setBufferIndex(2);
    // layout for buffer 1 (positions)
    m_vertexDesc->layouts()->object(1)->setStride(12);
    m_vertexDesc->layouts()->object(1)->setStepFunction(MTL::VertexStepFunctionPerVertex);
    // layout for buffer 2 (normals)
    m_vertexDesc->layouts()->object(2)->setStride(12);
    m_vertexDesc->layouts()->object(2)->setStepFunction(MTL::VertexStepFunctionPerVertex);
}

void ShaderManager::syncRuntimeContext() {
    m_rtCtx->renderPipelines["ForwardPass"] = m_vertexPipeline;
    m_rtCtx->renderPipelines["ForwardMeshPass"] = m_meshPipeline;
    m_rtCtx->renderPipelines["VisibilityPass"] = m_visPipeline;
    m_rtCtx->renderPipelines["VisibilityIndirectPass"] = m_visIndirectPipeline;
    m_rtCtx->renderPipelines["SkyPass"] = m_skyPipeline;
    m_rtCtx->renderPipelines["TonemapPass"] = m_tonemapPipeline;
    m_rtCtx->renderPipelines["OutputPass"] = m_outputPipeline;
    m_rtCtx->computePipelines["DeferredLightingPass"] = m_computePipeline;
    m_rtCtx->computePipelines["MeshletCullPass"] = m_cullPipeline;
    m_rtCtx->computePipelines["BuildIndirectPass"] = m_buildIndirectPipeline;
    m_rtCtx->samplers["tonemap"] = m_tonemapSampler;
}

bool ShaderManager::buildAll() {
    createVertexDescriptor();
    const char* root = m_projectRoot.c_str();

    // 1. Vertex pipeline (forward)
    {
        std::string src = compileSlangToMetal("Shaders/Vertex/bunny", root);
        if (src.empty()) { spdlog::error("Failed to compile vertex shader"); return false; }
        spdlog::info("Slang compiled Metal shader ({} bytes)", src.size());

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* vf = lib->newFunction(NS::String::string("vertexMain", NS::UTF8StringEncoding));
        auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

        auto* desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vf);
        desc->setFragmentFunction(ff);
        desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
        desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
        desc->setVertexDescriptor(m_vertexDesc);

        m_vertexPipeline = m_device->newRenderPipelineState(desc, &error);
        desc->release();
        vf->release(); ff->release(); lib->release();
        if (!m_vertexPipeline) {
            spdlog::error("Failed to create pipeline state: {}", error->localizedDescription()->utf8String());
            return false;
        }
    }
// PLACEHOLDER_BUILD_ALL_MESH

    // 2. Mesh shader pipeline
    {
        std::string src = compileSlangMeshShaderToMetal("Shaders/Mesh/meshlet", root);
        if (src.empty()) { spdlog::error("Failed to compile mesh shader"); return false; }
        src = patchMeshShaderMetalSource(src);
        spdlog::info("Mesh shader compiled ({} bytes)", src.size());

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create mesh Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* mf = lib->newFunction(NS::String::string("meshMain", NS::UTF8StringEncoding));
        auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

        auto* desc = MTL::MeshRenderPipelineDescriptor::alloc()->init();
        desc->setMeshFunction(mf);
        desc->setFragmentFunction(ff);
        desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
        desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

        NS::Error* meshError = nullptr;
        MTL::RenderPipelineReflection* refl = nullptr;
        m_meshPipeline = m_device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &meshError);
        desc->release();
        mf->release(); ff->release(); lib->release();
        if (!m_meshPipeline) {
            spdlog::error("Failed to create mesh pipeline state: {}", meshError->localizedDescription()->utf8String());
            return false;
        }
    }

    // 3. Visibility buffer mesh shader pipeline
    {
        std::string src = compileSlangMeshShaderToMetal("Shaders/Visibility/visibility", root);
        if (src.empty()) { spdlog::error("Failed to compile visibility shader"); return false; }
        src = patchVisibilityShaderMetalSource(src);
        spdlog::info("Visibility shader compiled ({} bytes)", src.size());
// PLACEHOLDER_BUILD_ALL_VIS

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create visibility Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* mf = lib->newFunction(NS::String::string("meshMain", NS::UTF8StringEncoding));
        auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

        auto* desc = MTL::MeshRenderPipelineDescriptor::alloc()->init();
        desc->setMeshFunction(mf);
        desc->setFragmentFunction(ff);
        desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR32Uint);
        desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

        NS::Error* visError = nullptr;
        MTL::RenderPipelineReflection* refl = nullptr;
        m_visPipeline = m_device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &visError);
        desc->release();
        mf->release(); ff->release(); lib->release();
        if (!m_visPipeline) {
            spdlog::error("Failed to create visibility pipeline state: {}", visError->localizedDescription()->utf8String());
            return false;
        }
    }

    // 3b. Visibility indirect mesh shader pipeline (GPU-driven)
    {
        std::string src = compileSlangMeshShaderToMetal("Shaders/Visibility/visibility_indirect", root);
        if (src.empty()) { spdlog::error("Failed to compile visibility indirect shader"); return false; }
        src = patchVisibilityShaderMetalSource(src);
        spdlog::info("Visibility indirect shader compiled ({} bytes)", src.size());

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create visibility indirect Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* mf = lib->newFunction(NS::String::string("meshMain", NS::UTF8StringEncoding));
        auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

        auto* desc = MTL::MeshRenderPipelineDescriptor::alloc()->init();
        desc->setMeshFunction(mf);
        desc->setFragmentFunction(ff);
        desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR32Uint);
        desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

        NS::Error* visError = nullptr;
        MTL::RenderPipelineReflection* refl = nullptr;
        m_visIndirectPipeline = m_device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &visError);
        desc->release();
        mf->release(); ff->release(); lib->release();
        if (!m_visIndirectPipeline) {
            spdlog::error("Failed to create visibility indirect pipeline state: {}", visError->localizedDescription()->utf8String());
            return false;
        }
    }

    // 3c. Meshlet cull compute pipeline
    {
        std::string src = compileSlangComputeShaderToMetal("Shaders/Visibility/meshlet_cull", root);
        if (src.empty()) { spdlog::error("Failed to compile meshlet cull shader"); return false; }
        spdlog::info("Meshlet cull shader compiled ({} bytes)", src.size());

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create meshlet cull Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* fn = lib->newFunction(NS::String::string("computeMain", NS::UTF8StringEncoding));
        m_cullPipeline = m_device->newComputePipelineState(fn, &error);
        fn->release(); lib->release();
        if (!m_cullPipeline) {
            spdlog::error("Failed to create meshlet cull pipeline state: {}", error->localizedDescription()->utf8String());
            return false;
        }
    }

    // 3d. Build indirect compute pipeline
    {
        std::string src = compileSlangComputeShaderToMetal("Shaders/Visibility/build_indirect", root);
        if (src.empty()) { spdlog::error("Failed to compile build indirect shader"); return false; }
        spdlog::info("Build indirect shader compiled ({} bytes)", src.size());

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create build indirect Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* fn = lib->newFunction(NS::String::string("computeMain", NS::UTF8StringEncoding));
        m_buildIndirectPipeline = m_device->newComputePipelineState(fn, &error);
        fn->release(); lib->release();
        if (!m_buildIndirectPipeline) {
            spdlog::error("Failed to create build indirect pipeline state: {}", error->localizedDescription()->utf8String());
            return false;
        }
    }

    // 4. Deferred lighting compute pipeline
    {
        std::string src = compileSlangComputeShaderToMetal("Shaders/Visibility/deferred_lighting", root);
        if (src.empty()) { spdlog::error("Failed to compile deferred lighting shader"); return false; }
        src = patchComputeShaderMetalSource(src);
        spdlog::info("Compute shader compiled ({} bytes)", src.size());

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create compute Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* fn = lib->newFunction(NS::String::string("computeMain", NS::UTF8StringEncoding));
        m_computePipeline = m_device->newComputePipelineState(fn, &error);
        fn->release(); lib->release();
        if (!m_computePipeline) {
            spdlog::error("Failed to create compute pipeline state: {}", error->localizedDescription()->utf8String());
            return false;
        }
    }
// PLACEHOLDER_BUILD_ALL_SKY

    // 5. Atmosphere sky pipeline (non-fatal)
    {
        std::string src = compileSlangToMetal("Shaders/Atmosphere/sky", root);
        if (!src.empty()) {
            NS::Error* error = nullptr;
            auto* compileOpts = MTL::CompileOptions::alloc()->init();
            auto* lib = m_device->newLibrary(
                NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
            compileOpts->release();
            if (!lib) {
                spdlog::warn("Failed to create sky Metal library: {}", error->localizedDescription()->utf8String());
            } else {
                auto* vf = lib->newFunction(NS::String::string("vertexMain", NS::UTF8StringEncoding));
                auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

                auto* desc = MTL::RenderPipelineDescriptor::alloc()->init();
                desc->setVertexFunction(vf);
                desc->setFragmentFunction(ff);
                desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);

                m_skyPipeline = m_device->newRenderPipelineState(desc, &error);
                if (!m_skyPipeline) {
                    spdlog::warn("Failed to create sky pipeline state: {}", error->localizedDescription()->utf8String());
                }

                desc->release();
                if (vf) vf->release();
                if (ff) ff->release();
                lib->release();
            }
        } else {
            spdlog::warn("Sky shader compile failed; atmosphere sky disabled");
        }
    }

    // 6. Tonemap pipeline
    {
        std::string src = compileSlangToMetal("Shaders/Post/tonemap", root);
        if (src.empty()) { spdlog::error("Failed to compile tonemap shader"); return false; }

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = m_device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create tonemap Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }
// PLACEHOLDER_BUILD_ALL_TONEMAP

        auto* vf = lib->newFunction(NS::String::string("vertexMain", NS::UTF8StringEncoding));
        auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

        auto* desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vf);
        desc->setFragmentFunction(ff);
        desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

        m_tonemapPipeline = m_device->newRenderPipelineState(desc, &error);
        desc->release();
        if (vf) vf->release();
        if (ff) ff->release();
        lib->release();
        if (!m_tonemapPipeline) {
            spdlog::error("Failed to create tonemap pipeline state: {}", error->localizedDescription()->utf8String());
            return false;
        }
    }

    // Tonemap sampler
    {
        auto* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
        samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
        samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
        samplerDesc->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
        samplerDesc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
        samplerDesc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
        m_tonemapSampler = m_device->newSamplerState(samplerDesc);
        samplerDesc->release();
        if (!m_tonemapSampler) {
            spdlog::error("Failed to create tonemap sampler state");
            return false;
        }
    }

    // 7. Output passthrough pipeline
    m_outputPipeline = reloadFullscreenShader("Shaders/Post/passthrough", MTL::PixelFormatBGRA8Unorm);
    if (!m_outputPipeline) {
        spdlog::error("Failed to create output passthrough pipeline");
        return false;
    }

    syncRuntimeContext();
    return true;
}
// PLACEHOLDER_RELOAD_HELPERS

// --- Reload helpers ---

MTL::RenderPipelineState* ShaderManager::reloadVertexShader(const char* shaderPath) {
    std::string src = compileSlangToMetal(shaderPath, m_projectRoot.c_str());
    if (src.empty()) return nullptr;

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = m_device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
    compileOpts->release();
    if (!lib) {
        spdlog::error("Hot-reload vertex lib: {}", error->localizedDescription()->utf8String());
        return nullptr;
    }

    auto* vf = lib->newFunction(NS::String::string("vertexMain", NS::UTF8StringEncoding));
    auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    auto* desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vf);
    desc->setFragmentFunction(ff);
    desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
    desc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    desc->setVertexDescriptor(m_vertexDesc);

    auto* pso = m_device->newRenderPipelineState(desc, &error);
    desc->release();
    if (vf) vf->release();
    if (ff) ff->release();
    lib->release();
    if (!pso) {
        spdlog::error("Hot-reload vertex PSO: {}", error->localizedDescription()->utf8String());
    }
    return pso;
}

MTL::RenderPipelineState* ShaderManager::reloadFullscreenShader(
    const char* shaderPath, MTL::PixelFormat colorFormat)
{
    std::string src = compileSlangToMetal(shaderPath, m_projectRoot.c_str());
    if (src.empty()) return nullptr;

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = m_device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
    compileOpts->release();
    if (!lib) {
        spdlog::error("Hot-reload fullscreen lib: {}", error->localizedDescription()->utf8String());
        return nullptr;
    }
// PLACEHOLDER_RELOAD_HELPERS_2

    auto* vf = lib->newFunction(NS::String::string("vertexMain", NS::UTF8StringEncoding));
    auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    auto* desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vf);
    desc->setFragmentFunction(ff);
    desc->colorAttachments()->object(0)->setPixelFormat(colorFormat);

    auto* pso = m_device->newRenderPipelineState(desc, &error);
    desc->release();
    if (vf) vf->release();
    if (ff) ff->release();
    lib->release();
    if (!pso) {
        spdlog::error("Hot-reload fullscreen PSO: {}", error->localizedDescription()->utf8String());
    }
    return pso;
}

MTL::RenderPipelineState* ShaderManager::reloadMeshShader(
    const char* shaderPath,
    std::string (*patchFn)(const std::string&),
    MTL::PixelFormat colorFormat, MTL::PixelFormat depthFormat)
{
    std::string src = compileSlangMeshShaderToMetal(shaderPath, m_projectRoot.c_str());
    if (src.empty()) return nullptr;
    src = patchFn(src);

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = m_device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
    compileOpts->release();
    if (!lib) {
        spdlog::error("Hot-reload mesh lib: {}", error->localizedDescription()->utf8String());
        return nullptr;
    }

    auto* mf = lib->newFunction(NS::String::string("meshMain", NS::UTF8StringEncoding));
    auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    auto* desc = MTL::MeshRenderPipelineDescriptor::alloc()->init();
    desc->setMeshFunction(mf);
    desc->setFragmentFunction(ff);
    desc->colorAttachments()->object(0)->setPixelFormat(colorFormat);
    desc->setDepthAttachmentPixelFormat(depthFormat);

    MTL::RenderPipelineReflection* refl = nullptr;
    auto* pso = m_device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &error);
    desc->release();
    if (mf) mf->release();
    if (ff) ff->release();
    lib->release();
    if (!pso) {
        spdlog::error("Hot-reload mesh PSO: {}", error->localizedDescription()->utf8String());
    }
    return pso;
}
// PLACEHOLDER_RELOAD_COMPUTE_AND_ALL

MTL::ComputePipelineState* ShaderManager::reloadComputeShader(
    const char* shaderPath, const char* entryPoint,
    std::string (*patchFn)(const std::string&))
{
    std::string src = compileSlangComputeShaderToMetal(shaderPath, m_projectRoot.c_str());
    if (src.empty()) return nullptr;
    if (patchFn) src = patchFn(src);

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = m_device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
    compileOpts->release();
    if (!lib) {
        spdlog::error("Hot-reload compute lib: {}", error->localizedDescription()->utf8String());
        return nullptr;
    }

    auto* fn = lib->newFunction(NS::String::string(entryPoint, NS::UTF8StringEncoding));
    auto* pso = m_device->newComputePipelineState(fn, &error);
    if (fn) fn->release();
    lib->release();
    if (!pso) {
        spdlog::error("Hot-reload compute PSO: {}", error->localizedDescription()->utf8String());
    }
    return pso;
}

std::pair<int,int> ShaderManager::reloadAll() {
    int reloaded = 0, failed = 0;

    // 1. Vertex shader
    if (auto* p = reloadVertexShader("Shaders/Vertex/bunny")) {
        m_vertexPipeline->release();
        m_vertexPipeline = p;
        reloaded++;
    } else { failed++; }

    // 2. Mesh shader
    if (auto* p = reloadMeshShader("Shaders/Mesh/meshlet",
            patchMeshShaderMetalSource, MTL::PixelFormatRGBA16Float, MTL::PixelFormatDepth32Float)) {
        m_meshPipeline->release();
        m_meshPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3. Visibility shader
    if (auto* p = reloadMeshShader("Shaders/Visibility/visibility",
            patchVisibilityShaderMetalSource, MTL::PixelFormatR32Uint, MTL::PixelFormatDepth32Float)) {
        m_visPipeline->release();
        m_visPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3b. Visibility indirect shader
    if (auto* p = reloadMeshShader("Shaders/Visibility/visibility_indirect",
            patchVisibilityShaderMetalSource, MTL::PixelFormatR32Uint, MTL::PixelFormatDepth32Float)) {
        if (m_visIndirectPipeline) m_visIndirectPipeline->release();
        m_visIndirectPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3c. Meshlet cull shader
    if (auto* p = reloadComputeShader("Shaders/Visibility/meshlet_cull",
            "computeMain", nullptr)) {
        if (m_cullPipeline) m_cullPipeline->release();
        m_cullPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3d. Build indirect shader
    if (auto* p = reloadComputeShader("Shaders/Visibility/build_indirect",
            "computeMain", nullptr)) {
        if (m_buildIndirectPipeline) m_buildIndirectPipeline->release();
        m_buildIndirectPipeline = p;
        reloaded++;
    } else { failed++; }

    // 4. Compute shader (deferred lighting)
    if (auto* p = reloadComputeShader("Shaders/Visibility/deferred_lighting",
            "computeMain", patchComputeShaderMetalSource)) {
        m_computePipeline->release();
        m_computePipeline = p;
        reloaded++;
    } else { failed++; }

    // 5. Sky shader
    if (auto* p = reloadFullscreenShader("Shaders/Atmosphere/sky", MTL::PixelFormatRGBA16Float)) {
        if (m_skyPipeline) m_skyPipeline->release();
        m_skyPipeline = p;
        reloaded++;
    } else { failed++; }

    // 6. Tonemap shader
    if (auto* p = reloadFullscreenShader("Shaders/Post/tonemap", MTL::PixelFormatBGRA8Unorm)) {
        m_tonemapPipeline->release();
        m_tonemapPipeline = p;
        reloaded++;
    } else { failed++; }

    // 7. Passthrough (output) shader
    if (auto* p = reloadFullscreenShader("Shaders/Post/passthrough", MTL::PixelFormatBGRA8Unorm)) {
        m_outputPipeline->release();
        m_outputPipeline = p;
        reloaded++;
    } else { failed++; }

    syncRuntimeContext();
    return {reloaded, failed};
}
