#include "shader_manager.h"
#include "slang_compiler.h"
#include "frame_context.h"

#include <Foundation/Foundation.hpp>
#include <spdlog/spdlog.h>

static MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

static MTL::Texture* metalTexture(void* handle) {
    return static_cast<MTL::Texture*>(handle);
}

static MTL::SamplerState* metalSampler(void* handle) {
    return static_cast<MTL::SamplerState*>(handle);
}

static MTL::VertexDescriptor* metalVertexDescriptor(void* handle) {
    return static_cast<MTL::VertexDescriptor*>(handle);
}

static MTL::RenderPipelineState* metalRenderPipeline(void* handle) {
    return static_cast<MTL::RenderPipelineState*>(handle);
}

static MTL::ComputePipelineState* metalComputePipeline(void* handle) {
    return static_cast<MTL::ComputePipelineState*>(handle);
}

static MTL::PixelFormat metalPixelFormat(RhiFormat format) {
    switch (format) {
    case RhiFormat::R32Uint: return MTL::PixelFormatR32Uint;
    case RhiFormat::RGBA16Float: return MTL::PixelFormatRGBA16Float;
    case RhiFormat::BGRA8Unorm: return MTL::PixelFormatBGRA8Unorm;
    case RhiFormat::D32Float: return MTL::PixelFormatDepth32Float;
    default: return MTL::PixelFormatInvalid;
    }
}

ShaderManager::ShaderManager(void* deviceHandle, const char* projectRoot)
    : m_device(deviceHandle)
    , m_projectRoot(projectRoot)
    , m_rtCtx(new PipelineRuntimeContext{})
{}

ShaderManager::~ShaderManager() {
    if (m_vertexPipeline) metalRenderPipeline(m_vertexPipeline)->release();
    if (m_meshPipeline) metalRenderPipeline(m_meshPipeline)->release();
    if (m_visPipeline) metalRenderPipeline(m_visPipeline)->release();
    if (m_visIndirectPipeline) metalRenderPipeline(m_visIndirectPipeline)->release();
    if (m_computePipeline) metalComputePipeline(m_computePipeline)->release();
    if (m_cullPipeline) metalComputePipeline(m_cullPipeline)->release();
    if (m_buildIndirectPipeline) metalComputePipeline(m_buildIndirectPipeline)->release();
    if (m_skyPipeline) metalRenderPipeline(m_skyPipeline)->release();
    if (m_tonemapPipeline) metalRenderPipeline(m_tonemapPipeline)->release();
    if (m_outputPipeline) metalRenderPipeline(m_outputPipeline)->release();
    if (m_histogramPipeline) metalComputePipeline(m_histogramPipeline)->release();
    if (m_autoExposurePipeline) metalComputePipeline(m_autoExposurePipeline)->release();
    if (m_taaPipeline) metalComputePipeline(m_taaPipeline)->release();
    if (m_tonemapSampler) metalSampler(m_tonemapSampler)->release();
    if (m_vertexDesc) metalVertexDescriptor(m_vertexDesc)->release();
    delete m_rtCtx;
}

PipelineRuntimeContext& ShaderManager::runtimeContext() { return *m_rtCtx; }
bool ShaderManager::hasSkyPipeline() const { return m_skyPipeline != nullptr; }

void ShaderManager::importTexture(const std::string& name, void* textureHandle) {
    auto* tex = metalTexture(textureHandle);
    m_rtCtx->importedTexturesRhi[name].setNativeHandle(
        textureHandle,
        tex ? static_cast<uint32_t>(tex->width()) : 0,
        tex ? static_cast<uint32_t>(tex->height()) : 0);
}

void ShaderManager::importSampler(const std::string& name, void* samplerHandle) {
    m_rtCtx->samplersRhi[name].setNativeHandle(samplerHandle);
}

void ShaderManager::createVertexDescriptor() {
    m_vertexDesc = MTL::VertexDescriptor::alloc()->init();
    auto* vertexDesc = metalVertexDescriptor(m_vertexDesc);
    // attribute(0) = position: float3 from buffer 1
    vertexDesc->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(0)->setOffset(0);
    vertexDesc->attributes()->object(0)->setBufferIndex(1);
    // attribute(1) = normal: float3 from buffer 2
    vertexDesc->attributes()->object(1)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(1)->setOffset(0);
    vertexDesc->attributes()->object(1)->setBufferIndex(2);
    // layout for buffer 1 (positions)
    vertexDesc->layouts()->object(1)->setStride(12);
    vertexDesc->layouts()->object(1)->setStepFunction(MTL::VertexStepFunctionPerVertex);
    // layout for buffer 2 (normals)
    vertexDesc->layouts()->object(2)->setStride(12);
    vertexDesc->layouts()->object(2)->setStepFunction(MTL::VertexStepFunctionPerVertex);
}

void ShaderManager::syncRuntimeContext() {
    m_rtCtx->renderPipelinesRhi["ForwardPass"].setNativeHandle(m_vertexPipeline);
    m_rtCtx->renderPipelinesRhi["ForwardMeshPass"].setNativeHandle(m_meshPipeline);
    m_rtCtx->renderPipelinesRhi["VisibilityPass"].setNativeHandle(m_visPipeline);
    m_rtCtx->renderPipelinesRhi["VisibilityIndirectPass"].setNativeHandle(m_visIndirectPipeline);
    m_rtCtx->renderPipelinesRhi["SkyPass"].setNativeHandle(m_skyPipeline);
    m_rtCtx->renderPipelinesRhi["TonemapPass"].setNativeHandle(m_tonemapPipeline);
    m_rtCtx->renderPipelinesRhi["OutputPass"].setNativeHandle(m_outputPipeline);

    m_rtCtx->computePipelinesRhi["DeferredLightingPass"].setNativeHandle(m_computePipeline);
    m_rtCtx->computePipelinesRhi["MeshletCullPass"].setNativeHandle(m_cullPipeline);
    m_rtCtx->computePipelinesRhi["BuildIndirectPass"].setNativeHandle(m_buildIndirectPipeline);
    if (m_meshletVisPipeline)
        m_rtCtx->computePipelinesRhi["MeshletVisualizePass"].setNativeHandle(m_meshletVisPipeline);
    if (m_histogramPipeline)
        m_rtCtx->computePipelinesRhi["HistogramPass"].setNativeHandle(m_histogramPipeline);
    if (m_autoExposurePipeline)
        m_rtCtx->computePipelinesRhi["AutoExposurePass"].setNativeHandle(m_autoExposurePipeline);
    if (m_taaPipeline)
        m_rtCtx->computePipelinesRhi["TAAPass"].setNativeHandle(m_taaPipeline);

    m_rtCtx->samplersRhi["tonemap"].setNativeHandle(m_tonemapSampler);
}

bool ShaderManager::buildAll() {
    createVertexDescriptor();
    const char* root = m_projectRoot.c_str();
    auto* device = metalDevice(m_device);

    // 1. Vertex pipeline (forward)
    {
        std::string src = compileSlangToMetal("Shaders/Vertex/bunny", root);
        if (src.empty()) { spdlog::error("Failed to compile vertex shader"); return false; }
        spdlog::info("Slang compiled Metal shader ({} bytes)", src.size());

        NS::Error* error = nullptr;
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = device->newLibrary(
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
        desc->setVertexDescriptor(metalVertexDescriptor(m_vertexDesc));

        m_vertexPipeline = device->newRenderPipelineState(desc, &error);
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
        auto* lib = device->newLibrary(
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
        m_meshPipeline = device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &meshError);
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
        auto* lib = device->newLibrary(
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
        m_visPipeline = device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &visError);
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
        auto* lib = device->newLibrary(
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
        m_visIndirectPipeline = device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &visError);
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
        auto* lib = device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create meshlet cull Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* fn = lib->newFunction(NS::String::string("computeMain", NS::UTF8StringEncoding));
        m_cullPipeline = device->newComputePipelineState(fn, &error);
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
        auto* lib = device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create build indirect Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* fn = lib->newFunction(NS::String::string("computeMain", NS::UTF8StringEncoding));
        m_buildIndirectPipeline = device->newComputePipelineState(fn, &error);
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
        auto* lib = device->newLibrary(
            NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
        compileOpts->release();
        if (!lib) {
            spdlog::error("Failed to create compute Metal library: {}", error->localizedDescription()->utf8String());
            return false;
        }

        auto* fn = lib->newFunction(NS::String::string("computeMain", NS::UTF8StringEncoding));
        m_computePipeline = device->newComputePipelineState(fn, &error);
        fn->release(); lib->release();
        if (!m_computePipeline) {
            spdlog::error("Failed to create compute pipeline state: {}", error->localizedDescription()->utf8String());
            return false;
        }
    }

    // 4b. Meshlet visualize compute pipeline (non-fatal)
    {
        std::string src = compileSlangComputeShaderToMetal("Shaders/Visibility/meshlet_visualize", root);
        if (!src.empty()) {
            NS::Error* error = nullptr;
            auto* compileOpts = MTL::CompileOptions::alloc()->init();
            auto* lib = device->newLibrary(
                NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
            compileOpts->release();
            if (!lib) {
                spdlog::warn("Failed to create meshlet visualize Metal library: {}", error->localizedDescription()->utf8String());
            } else {
                auto* fn = lib->newFunction(NS::String::string("computeMain", NS::UTF8StringEncoding));
                m_meshletVisPipeline = device->newComputePipelineState(fn, &error);
                fn->release(); lib->release();
                if (!m_meshletVisPipeline) {
                    spdlog::warn("Failed to create meshlet visualize pipeline: {}", error->localizedDescription()->utf8String());
                }
            }
        } else {
            spdlog::warn("Failed to compile meshlet visualize shader");
        }
    }
// PLACEHOLDER_BUILD_ALL_SKY

    // 5. Atmosphere sky pipeline (non-fatal)
    {
        std::string src = compileSlangToMetal("Shaders/Atmosphere/sky", root);
        if (!src.empty()) {
            NS::Error* error = nullptr;
            auto* compileOpts = MTL::CompileOptions::alloc()->init();
            auto* lib = device->newLibrary(
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

                m_skyPipeline = device->newRenderPipelineState(desc, &error);
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
        auto* lib = device->newLibrary(
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

        m_tonemapPipeline = device->newRenderPipelineState(desc, &error);
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
        m_tonemapSampler = device->newSamplerState(samplerDesc);
        samplerDesc->release();
        if (!m_tonemapSampler) {
            spdlog::error("Failed to create tonemap sampler state");
            return false;
        }
    }

    // 7. Output passthrough pipeline
    m_outputPipeline = reloadFullscreenShader("Shaders/Post/passthrough", RhiFormat::BGRA8Unorm);
    if (!m_outputPipeline) {
        spdlog::error("Failed to create output passthrough pipeline");
        return false;
    }

    // 8. Auto-exposure compute pipelines (non-fatal)
    {
        auto* histPso = reloadComputeShader("Shaders/Post/auto_exposure", "histogramMain", nullptr);
        if (histPso) {
            m_histogramPipeline = histPso;
        } else {
            spdlog::warn("Failed to compile histogram shader; auto-exposure disabled");
        }

        auto* expPso = reloadComputeShader("Shaders/Post/auto_exposure", "exposureMain", nullptr);
        if (expPso) {
            m_autoExposurePipeline = expPso;
        } else {
            spdlog::warn("Failed to compile auto-exposure shader; auto-exposure disabled");
        }
    }

    // 9. TAA compute pipeline (non-fatal)
    {
        auto* taaPso = reloadComputeShader("Shaders/Post/taa", "taaMain", nullptr);
        if (taaPso) {
            m_taaPipeline = taaPso;
        } else {
            spdlog::warn("Failed to compile TAA shader; TAA disabled");
        }
    }

    syncRuntimeContext();
    return true;
}
// PLACEHOLDER_RELOAD_HELPERS

// --- Reload helpers ---

void* ShaderManager::reloadVertexShader(const char* shaderPath) {
    auto* device = metalDevice(m_device);
    std::string src = compileSlangToMetal(shaderPath, m_projectRoot.c_str());
    if (src.empty()) return nullptr;

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
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
    desc->setVertexDescriptor(metalVertexDescriptor(m_vertexDesc));

    auto* pso = device->newRenderPipelineState(desc, &error);
    desc->release();
    if (vf) vf->release();
    if (ff) ff->release();
    lib->release();
    if (!pso) {
        spdlog::error("Hot-reload vertex PSO: {}", error->localizedDescription()->utf8String());
    }
    return pso;
}

void* ShaderManager::reloadFullscreenShader(
    const char* shaderPath, RhiFormat colorFormat)
{
    auto* device = metalDevice(m_device);
    std::string src = compileSlangToMetal(shaderPath, m_projectRoot.c_str());
    if (src.empty()) return nullptr;

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
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
    desc->colorAttachments()->object(0)->setPixelFormat(metalPixelFormat(colorFormat));

    auto* pso = device->newRenderPipelineState(desc, &error);
    desc->release();
    if (vf) vf->release();
    if (ff) ff->release();
    lib->release();
    if (!pso) {
        spdlog::error("Hot-reload fullscreen PSO: {}", error->localizedDescription()->utf8String());
    }
    return pso;
}

void* ShaderManager::reloadMeshShader(
    const char* shaderPath,
    std::string (*patchFn)(const std::string&),
    RhiFormat colorFormat, RhiFormat depthFormat)
{
    auto* device = metalDevice(m_device);
    std::string src = compileSlangMeshShaderToMetal(shaderPath, m_projectRoot.c_str());
    if (src.empty()) return nullptr;
    src = patchFn(src);

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
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
    desc->colorAttachments()->object(0)->setPixelFormat(metalPixelFormat(colorFormat));
    desc->setDepthAttachmentPixelFormat(metalPixelFormat(depthFormat));

    MTL::RenderPipelineReflection* refl = nullptr;
    auto* pso = device->newRenderPipelineState(desc, MTL::PipelineOptionNone, &refl, &error);
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

void* ShaderManager::reloadComputeShader(
    const char* shaderPath, const char* entryPoint,
    std::string (*patchFn)(const std::string&))
{
    auto* device = metalDevice(m_device);
    std::string src = compileSlangComputeShaderToMetal(shaderPath, m_projectRoot.c_str(), entryPoint);
    if (src.empty()) return nullptr;
    if (patchFn) src = patchFn(src);

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding), compileOpts, &error);
    compileOpts->release();
    if (!lib) {
        spdlog::error("Hot-reload compute lib: {}", error->localizedDescription()->utf8String());
        return nullptr;
    }

    auto* fn = lib->newFunction(NS::String::string(entryPoint, NS::UTF8StringEncoding));
    auto* pso = device->newComputePipelineState(fn, &error);
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
        metalRenderPipeline(m_vertexPipeline)->release();
        m_vertexPipeline = p;
        reloaded++;
    } else { failed++; }

    // 2. Mesh shader
    if (auto* p = reloadMeshShader("Shaders/Mesh/meshlet",
            patchMeshShaderMetalSource, RhiFormat::RGBA16Float, RhiFormat::D32Float)) {
        metalRenderPipeline(m_meshPipeline)->release();
        m_meshPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3. Visibility shader
    if (auto* p = reloadMeshShader("Shaders/Visibility/visibility",
            patchVisibilityShaderMetalSource, RhiFormat::R32Uint, RhiFormat::D32Float)) {
        metalRenderPipeline(m_visPipeline)->release();
        m_visPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3b. Visibility indirect shader
    if (auto* p = reloadMeshShader("Shaders/Visibility/visibility_indirect",
            patchVisibilityShaderMetalSource, RhiFormat::R32Uint, RhiFormat::D32Float)) {
        if (m_visIndirectPipeline) metalRenderPipeline(m_visIndirectPipeline)->release();
        m_visIndirectPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3c. Meshlet cull shader
    if (auto* p = reloadComputeShader("Shaders/Visibility/meshlet_cull",
            "computeMain", nullptr)) {
        if (m_cullPipeline) metalComputePipeline(m_cullPipeline)->release();
        m_cullPipeline = p;
        reloaded++;
    } else { failed++; }

    // 3d. Build indirect shader
    if (auto* p = reloadComputeShader("Shaders/Visibility/build_indirect",
            "computeMain", nullptr)) {
        if (m_buildIndirectPipeline) metalComputePipeline(m_buildIndirectPipeline)->release();
        m_buildIndirectPipeline = p;
        reloaded++;
    } else { failed++; }

    // 4. Compute shader (deferred lighting)
    if (auto* p = reloadComputeShader("Shaders/Visibility/deferred_lighting",
            "computeMain", patchComputeShaderMetalSource)) {
        metalComputePipeline(m_computePipeline)->release();
        m_computePipeline = p;
        reloaded++;
    } else { failed++; }

    // 4b. Meshlet visualize compute shader
    if (auto* p = reloadComputeShader("Shaders/Visibility/meshlet_visualize",
            "computeMain", nullptr)) {
        if (m_meshletVisPipeline) metalComputePipeline(m_meshletVisPipeline)->release();
        m_meshletVisPipeline = p;
        reloaded++;
    } else { failed++; }

    // 5. Sky shader
    if (auto* p = reloadFullscreenShader("Shaders/Atmosphere/sky", RhiFormat::RGBA16Float)) {
        if (m_skyPipeline) metalRenderPipeline(m_skyPipeline)->release();
        m_skyPipeline = p;
        reloaded++;
    } else { failed++; }

    // 6. Tonemap shader
    if (auto* p = reloadFullscreenShader("Shaders/Post/tonemap", RhiFormat::BGRA8Unorm)) {
        metalRenderPipeline(m_tonemapPipeline)->release();
        m_tonemapPipeline = p;
        reloaded++;
    } else { failed++; }

    // 7. Passthrough (output) shader
    if (auto* p = reloadFullscreenShader("Shaders/Post/passthrough", RhiFormat::BGRA8Unorm)) {
        metalRenderPipeline(m_outputPipeline)->release();
        m_outputPipeline = p;
        reloaded++;
    } else { failed++; }

    // 8. Auto-exposure compute shaders
    if (auto* p = reloadComputeShader("Shaders/Post/auto_exposure", "histogramMain", nullptr)) {
        if (m_histogramPipeline) metalComputePipeline(m_histogramPipeline)->release();
        m_histogramPipeline = p;
        reloaded++;
    } else { failed++; }

    if (auto* p = reloadComputeShader("Shaders/Post/auto_exposure", "exposureMain", nullptr)) {
        if (m_autoExposurePipeline) metalComputePipeline(m_autoExposurePipeline)->release();
        m_autoExposurePipeline = p;
        reloaded++;
    } else { failed++; }

    // 9. TAA compute shader
    if (auto* p = reloadComputeShader("Shaders/Post/taa", "taaMain", nullptr)) {
        if (m_taaPipeline) metalComputePipeline(m_taaPipeline)->release();
        m_taaPipeline = p;
        reloaded++;
    } else { failed++; }

    syncRuntimeContext();
    return {reloaded, failed};
}
