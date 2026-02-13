#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <ml.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "glfw_metal_bridge.h"
#include "imgui_metal_bridge.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "material_loader.h"
#include "camera.h"
#include "input.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "scene_graph.h"
#include "scene_graph_ui.h"
#include "raytraced_shadows.h"

#include <slang.h>
#include <slang-com-ptr.h>

#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include <fstream>
#include <functional>
#include <memory>

#include <tracy/Tracy.hpp>
#include "tracy_metal.h"
#include "frame_graph.h"
#include "visibility_constants.h"
#include "render_uniforms.h"
#include "render_pass.h"
#include "blit_pass.h"
#include "tonemap_pass.h"
#include "imgui_overlay_pass.h"
#include "visibility_pass.h"
#include "shadow_ray_pass.h"
#include "deferred_lighting_pass.h"
#include "forward_pass.h"
#include "sky_pass.h"
#include "pipeline_asset.h"
#include "pipeline_builder.h"
#include "frame_context.h"


static std::string compileSlangToMetal(const char* shaderPath, const char* searchPath = nullptr) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    spdlog::info("Loading shader: {} (search path: {})", shaderPath, searchPath ? searchPath : "<cwd>");
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
        spdlog::error("Slang load error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    Slang::ComPtr<slang::IEntryPoint> vertexEntry;
    module->findEntryPointByName("vertexMain", vertexEntry.writeRef());
    Slang::ComPtr<slang::IEntryPoint> fragmentEntry;
    module->findEntryPointByName("fragmentMain", fragmentEntry.writeRef());

    std::vector<slang::IComponentType*> components = {module, vertexEntry, fragmentEntry};
    Slang::ComPtr<slang::IComponentType> program;
    session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    program->link(linkedProgram.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IBlob> metalCode;
    linkedProgram->getTargetCode(0, metalCode.writeRef(), diagnostics.writeRef());
    if (!metalCode) {
        if (diagnostics)
        spdlog::error("Slang compile error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

static std::string compileSlangMeshShaderToMetal(const char* shaderPath, const char* searchPath = nullptr) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    spdlog::info("Loading shader: {} (search path: {})", shaderPath, searchPath ? searchPath : "<cwd>");
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
        spdlog::error("Slang load error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    Slang::ComPtr<slang::IEntryPoint> meshEntry;
    module->findEntryPointByName("meshMain", meshEntry.writeRef());
    Slang::ComPtr<slang::IEntryPoint> fragmentEntry;
    module->findEntryPointByName("fragmentMain", fragmentEntry.writeRef());

    std::vector<slang::IComponentType*> components = {module, meshEntry, fragmentEntry};
    Slang::ComPtr<slang::IComponentType> program;
    session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    program->link(linkedProgram.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IBlob> metalCode;
    linkedProgram->getTargetCode(0, metalCode.writeRef(), diagnostics.writeRef());
    if (!metalCode) {
        if (diagnostics)
        spdlog::error("Slang mesh shader compile error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

static std::string compileSlangComputeShaderToMetal(const char* shaderPath, const char* searchPath = nullptr) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    spdlog::info("Loading shader: {} (search path: {})", shaderPath, searchPath ? searchPath : "<cwd>");
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
        spdlog::error("Slang load error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    Slang::ComPtr<slang::IEntryPoint> computeEntry;
    module->findEntryPointByName("computeMain", computeEntry.writeRef());

    std::vector<slang::IComponentType*> components = {module, computeEntry};
    Slang::ComPtr<slang::IComponentType> program;
    session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    program->link(linkedProgram.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IBlob> metalCode;
    linkedProgram->getTargetCode(0, metalCode.writeRef(), diagnostics.writeRef());
    if (!metalCode) {
        if (diagnostics)
        spdlog::error("Slang compute shader compile error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

static std::string patchMeshShaderMetalSource(const std::string& source) {
    // Slang v2026.1.2 bug: mesh output struct members lack [[user(...)]] attributes
    // that the fragment shader's [[stage_in]] expects. We patch them in.
    std::string patched = source;

    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewNormal\w*)\s*;)"),
        "$1 [[user(NORMAL)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewPos\w*)\s*;)"),
        "$1 [[user(TEXCOORD)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((float2\s+\w*uv\w*)\s*;)"),
        "$1 [[user(TEXCOORD_1)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((\[\[flat\]\]\s+uint\s+\w*materialID\w*)\s*;)"),
        "$1 [[user(TEXCOORD_2)]];");

    // Slang doesn't emit [[texture(0)]] on texture array parameters.
    // Patch both mesh and fragment function signatures.
    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+))"),
        "$1 [[texture(0)]]");

    return patched;
}

static std::string patchVisibilityShaderMetalSource(const std::string& source) {
    std::string patched = source;

    // Patch [[user(...)]] on VisVertex members in mesh output struct
    patched = std::regex_replace(patched,
        std::regex(R"((float2\s+\w*uv\w*)\s*;)"),
        "$1 [[user(TEXCOORD)]];");

    // Patch [[user(...)]] on VisPrimitive members
    patched = std::regex_replace(patched,
        std::regex(R"((uint\s+\w*visibility\w*)\s*;)"),
        "$1 [[user(TEXCOORD_1)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((uint\s+\w*materialID\w*)\s*;)"),
        "$1 [[user(TEXCOORD_2)]];");

    // Patch [[texture(0)]] on texture array parameters
    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+))"),
        "$1 [[texture(0)]]");

    return patched;
}

static std::string patchComputeShaderMetalSource(const std::string& source) {
    std::string patched = source;

    // Slang emits [[texture(3)]] on the KernelContext struct member but NOT on
    // the function parameter. Patch the function parameter to add [[texture(3)]].
    // Match array texture params that don't already have [[texture(...)]]
    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+)(\s*,))"),
        "$1 [[texture(3)]]$2");

    return patched;
}

struct AtmosphereTextureSet {
    MTL::Texture* transmittance = nullptr;
    MTL::Texture* scattering = nullptr;
    MTL::Texture* irradiance = nullptr;
    MTL::SamplerState* sampler = nullptr;

    bool isValid() const {
        return transmittance && scattering && irradiance && sampler;
    }

    void release() {
        if (transmittance) { transmittance->release(); transmittance = nullptr; }
        if (scattering) { scattering->release(); scattering = nullptr; }
        if (irradiance) { irradiance->release(); irradiance = nullptr; }
        if (sampler) { sampler->release(); sampler = nullptr; }
    }
};

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
    if (!tex) {
        return nullptr;
    }
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
    if (!tex) {
        return nullptr;
    }
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
    constexpr int kScatteringWidth = 256; // NU_SIZE * MU_S_SIZE
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


// --- Shader hot-reload helpers ---
// Each returns a new pipeline on success, nullptr on failure (caller keeps old pipeline).

static MTL::RenderPipelineState* reloadVertexShader(
    MTL::Device* device, const char* shaderPath, const char* searchPath,
    MTL::VertexDescriptor* vertexDesc)
{
    std::string src = compileSlangToMetal(shaderPath, searchPath);
    if (src.empty()) return nullptr;

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding),
        compileOpts, &error);
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
    desc->setVertexDescriptor(vertexDesc);

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

static MTL::RenderPipelineState* reloadFullscreenShader(
    MTL::Device* device, const char* shaderPath, const char* searchPath,
    MTL::PixelFormat colorFormat)
{
    std::string src = compileSlangToMetal(shaderPath, searchPath);
    if (src.empty()) return nullptr;

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding),
        compileOpts, &error);
    compileOpts->release();
    if (!lib) {
        spdlog::error("Hot-reload fullscreen lib: {}", error->localizedDescription()->utf8String());
        return nullptr;
    }

    auto* vf = lib->newFunction(NS::String::string("vertexMain", NS::UTF8StringEncoding));
    auto* ff = lib->newFunction(NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    auto* desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vf);
    desc->setFragmentFunction(ff);
    desc->colorAttachments()->object(0)->setPixelFormat(colorFormat);

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

static MTL::RenderPipelineState* reloadMeshShader(
    MTL::Device* device, const char* shaderPath, const char* searchPath,
    std::function<std::string(const std::string&)> patchFn,
    MTL::PixelFormat colorFormat, MTL::PixelFormat depthFormat)
{
    std::string src = compileSlangMeshShaderToMetal(shaderPath, searchPath);
    if (src.empty()) return nullptr;
    src = patchFn(src);

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding),
        compileOpts, &error);
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

static MTL::ComputePipelineState* reloadComputeShader(
    MTL::Device* device, const char* shaderPath, const char* searchPath,
    const char* entryPoint,
    std::function<std::string(const std::string&)> patchFn)
{
    std::string src = compileSlangComputeShaderToMetal(shaderPath, searchPath);
    if (src.empty()) return nullptr;
    src = patchFn(src);

    NS::Error* error = nullptr;
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    auto* lib = device->newLibrary(
        NS::String::string(src.c_str(), NS::UTF8StringEncoding),
        compileOpts, &error);
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

static bool loadPipelineAssetChecked(const std::string& path,
                                     const char* label,
                                     PipelineAsset& outAsset) {
    PipelineAsset loaded = PipelineAsset::load(path);
    if (loaded.name.empty()) {
        spdlog::error("Failed to load {} pipeline from '{}'", label, path);
        return false;
    }

    std::string validationError;
    if (!loaded.validate(validationError)) {
        spdlog::error("Invalid {} pipeline '{}': {}", label, path, validationError);
        return false;
    }

    outAsset = std::move(loaded);
    spdlog::info("Loaded {} pipeline: {} ({} passes, {} resources)",
                 label, outAsset.name, outAsset.passes.size(), outAsset.resources.size());
    return true;
}



int main() {
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Metallic - Sponza", nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return 1;
    }

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        spdlog::error("Metal is not supported on this device");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    spdlog::info("Metal device: {}", device->name()->utf8String());

    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Tracy GPU profiling context
    TracyMetalCtxHandle tracyGpuCtx = tracyMetalCreate(device);

    CA::MetalLayer* metalLayer = static_cast<CA::MetalLayer*>(
        attachMetalLayerToGLFWWindow(window));
    metalLayer->setDevice(device);
    metalLayer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    metalLayer->setFramebufferOnly(false);

    // Load scene mesh
    LoadedMesh sceneMesh;
    if (!loadGLTFMesh(device, "Asset/Sponza/glTF/Sponza.gltf", sceneMesh)) {
        spdlog::error("Failed to load scene mesh");
        return 1;
    }

    // Build meshlets for mesh shader rendering
    MeshletData meshletData;
    if (!buildMeshlets(device, sceneMesh, meshletData)) {
        spdlog::error("Failed to build meshlets");
        return 1;
    }

    // Load materials and textures
    LoadedMaterials materials;
    if (!loadGLTFMaterials(device, commandQueue, "Asset/Sponza/glTF/Sponza.gltf", materials)) {
        spdlog::error("Failed to load materials");
        return 1;
    }

    // Build scene graph from glTF node hierarchy
    SceneGraph sceneGraph;
    if (!sceneGraph.buildFromGLTF("Asset/Sponza/glTF/Sponza.gltf", sceneMesh, meshletData)) {
        spdlog::error("Failed to build scene graph");
        return 1;
    }
    sceneGraph.updateTransforms();

    const char* projectRoot = PROJECT_SOURCE_DIR;

    // Build raytracing acceleration structures for shadows
    RaytracedShadowResources shadowResources;
    bool rtShadowsAvailable = false;
    if (device->supportsRaytracing()) {
        ZoneScopedN("Build Acceleration Structures");
        if (buildAccelerationStructures(device, commandQueue, sceneMesh, sceneGraph, shadowResources) &&
            createShadowPipeline(device, shadowResources, projectRoot)) {
            rtShadowsAvailable = true;
            spdlog::info("Raytraced shadows enabled");
        } else {
            spdlog::error("Failed to initialize raytraced shadows");
            shadowResources.release();
        }
    } else {
        spdlog::info("Raytracing not supported on this device");
    }

    // Init orbit camera
    OrbitCamera camera;
    camera.initFromBounds(sceneMesh.bboxMin, sceneMesh.bboxMax);

    // Setup input
    InputState inputState;
    inputState.camera = &camera;
    setupInputCallbacks(window, &inputState);

    // Init Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui_ImplGlfw_InitForOther(window, true);
    imguiInit(device);

    // Compile Slang shader to Metal source
    std::string metalSource = compileSlangToMetal("Shaders/Vertex/bunny", projectRoot);
    if (metalSource.empty()) {
        spdlog::error("Failed to compile Slang shader");
        return 1;
    }
    spdlog::info("Slang compiled Metal shader ({} bytes)", metalSource.size());

    // Create Metal library from compiled source
    NS::Error* error = nullptr;
    NS::String* sourceStr = NS::String::string(metalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* compileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* library = device->newLibrary(sourceStr, compileOpts, &error);
    compileOpts->release();
    if (!library) {
        spdlog::error("Failed to create Metal library: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }

    MTL::Function* vertexFn = library->newFunction(
        NS::String::string("vertexMain", NS::UTF8StringEncoding));
    MTL::Function* fragmentFn = library->newFunction(
        NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    // Vertex descriptor: buffer 1 = positions, buffer 2 = normals
    // (buffer 0 is reserved for Slang's uniform ConstantBuffer)
    MTL::VertexDescriptor* vertexDesc = MTL::VertexDescriptor::alloc()->init();
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

    // Create render pipeline
    MTL::RenderPipelineDescriptor* pipelineDesc =
        MTL::RenderPipelineDescriptor::alloc()->init();
    pipelineDesc->setVertexFunction(vertexFn);
    pipelineDesc->setFragmentFunction(fragmentFn);
    pipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
    pipelineDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    pipelineDesc->setVertexDescriptor(vertexDesc);
    // vertexDesc kept alive for shader hot-reload

    MTL::RenderPipelineState* pipelineState =
        device->newRenderPipelineState(pipelineDesc, &error);
    if (!pipelineState) {
        spdlog::error("Failed to create pipeline state: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }
    pipelineDesc->release();
    vertexFn->release();
    fragmentFn->release();
    library->release();

    // --- Mesh shader pipeline ---
    std::string meshMetalSource = compileSlangMeshShaderToMetal("Shaders/Mesh/meshlet", projectRoot);
    if (meshMetalSource.empty()) {
        spdlog::error("Failed to compile Slang mesh shader");
        return 1;
    }
    meshMetalSource = patchMeshShaderMetalSource(meshMetalSource);
    spdlog::info("Mesh shader compiled ({} bytes)", meshMetalSource.size());

    NS::String* meshSourceStr = NS::String::string(meshMetalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* meshCompileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* meshLibrary = device->newLibrary(meshSourceStr, meshCompileOpts, &error);
    meshCompileOpts->release();
    if (!meshLibrary) {
        spdlog::error("Failed to create mesh Metal library: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }

    MTL::Function* meshFn = meshLibrary->newFunction(
        NS::String::string("meshMain", NS::UTF8StringEncoding));
    MTL::Function* meshFragFn = meshLibrary->newFunction(
        NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    auto* meshPipelineDesc = MTL::MeshRenderPipelineDescriptor::alloc()->init();
    meshPipelineDesc->setMeshFunction(meshFn);
    meshPipelineDesc->setFragmentFunction(meshFragFn);
    meshPipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
    meshPipelineDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    NS::Error* meshError = nullptr;
    MTL::RenderPipelineReflection* meshReflection = nullptr;
    MTL::RenderPipelineState* meshPipelineState = device->newRenderPipelineState(
        meshPipelineDesc, MTL::PipelineOptionNone, &meshReflection, &meshError);
    if (!meshPipelineState) {
        spdlog::error("Failed to create mesh pipeline state: {}",
                      meshError->localizedDescription()->utf8String());
        return 1;
    }
    meshPipelineDesc->release();
    meshFn->release();
    meshFragFn->release();
    meshLibrary->release();

    // --- Visibility buffer mesh shader pipeline ---
    std::string visMetalSource = compileSlangMeshShaderToMetal("Shaders/Visibility/visibility", projectRoot);
    if (visMetalSource.empty()) {
        spdlog::error("Failed to compile visibility shader");
        return 1;
    }
    visMetalSource = patchVisibilityShaderMetalSource(visMetalSource);
    spdlog::info("Visibility shader compiled ({} bytes)", visMetalSource.size());

    NS::String* visSourceStr = NS::String::string(visMetalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* visCompileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* visLibrary = device->newLibrary(visSourceStr, visCompileOpts, &error);
    visCompileOpts->release();
    if (!visLibrary) {
        spdlog::error("Failed to create visibility Metal library: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }

    MTL::Function* visMeshFn = visLibrary->newFunction(
        NS::String::string("meshMain", NS::UTF8StringEncoding));
    MTL::Function* visFragFn = visLibrary->newFunction(
        NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    auto* visPipelineDesc = MTL::MeshRenderPipelineDescriptor::alloc()->init();
    visPipelineDesc->setMeshFunction(visMeshFn);
    visPipelineDesc->setFragmentFunction(visFragFn);
    visPipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR32Uint);
    visPipelineDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    NS::Error* visError = nullptr;
    MTL::RenderPipelineReflection* visReflection = nullptr;
    MTL::RenderPipelineState* visPipelineState = device->newRenderPipelineState(
        visPipelineDesc, MTL::PipelineOptionNone, &visReflection, &visError);
    if (!visPipelineState) {
        spdlog::error("Failed to create visibility pipeline state: {}",
                      visError->localizedDescription()->utf8String());
        return 1;
    }
    visPipelineDesc->release();
    visMeshFn->release();
    visFragFn->release();
    visLibrary->release();

    // --- Deferred lighting compute pipeline ---
    std::string computeMetalSource = compileSlangComputeShaderToMetal("Shaders/Visibility/deferred_lighting", projectRoot);
    if (computeMetalSource.empty()) {
        spdlog::error("Failed to compile deferred lighting shader");
        return 1;
    }
    computeMetalSource = patchComputeShaderMetalSource(computeMetalSource);
    spdlog::info("Compute shader compiled ({} bytes)", computeMetalSource.size());

    NS::String* computeSourceStr = NS::String::string(computeMetalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* computeCompileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* computeLibrary = device->newLibrary(computeSourceStr, computeCompileOpts, &error);
    computeCompileOpts->release();
    if (!computeLibrary) {
        spdlog::error("Failed to create compute Metal library: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }

    MTL::Function* computeFn = computeLibrary->newFunction(
        NS::String::string("computeMain", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* computePipelineState =
        device->newComputePipelineState(computeFn, &error);
    if (!computePipelineState) {
        spdlog::error("Failed to create compute pipeline state: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }
    computeFn->release();
    computeLibrary->release();

    // --- Atmosphere sky pipeline ---
    MTL::RenderPipelineState* skyPipelineState = nullptr;
    std::string skyMetalSource = compileSlangToMetal("Shaders/Atmosphere/sky", projectRoot);
    if (!skyMetalSource.empty()) {
        NS::String* skySourceStr = NS::String::string(skyMetalSource.c_str(), NS::UTF8StringEncoding);
        MTL::CompileOptions* skyCompileOpts = MTL::CompileOptions::alloc()->init();
        MTL::Library* skyLibrary = device->newLibrary(skySourceStr, skyCompileOpts, &error);
        skyCompileOpts->release();
        if (!skyLibrary) {
            spdlog::warn("Failed to create sky Metal library: {}",
                         error->localizedDescription()->utf8String());
        } else {
            MTL::Function* skyVertexFn = skyLibrary->newFunction(
                NS::String::string("vertexMain", NS::UTF8StringEncoding));
            MTL::Function* skyFragmentFn = skyLibrary->newFunction(
                NS::String::string("fragmentMain", NS::UTF8StringEncoding));

            auto* skyPipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
            skyPipelineDesc->setVertexFunction(skyVertexFn);
            skyPipelineDesc->setFragmentFunction(skyFragmentFn);
            skyPipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);

            skyPipelineState = device->newRenderPipelineState(skyPipelineDesc, &error);
            if (!skyPipelineState) {
                spdlog::warn("Failed to create sky pipeline state: {}",
                             error->localizedDescription()->utf8String());
            }

            skyPipelineDesc->release();
            if (skyVertexFn) skyVertexFn->release();
            if (skyFragmentFn) skyFragmentFn->release();
            skyLibrary->release();
        }
    } else {
        spdlog::warn("Sky shader compile failed; atmosphere sky disabled");
    }

    // --- Tonemap pipeline ---
    std::string tonemapMetalSource = compileSlangToMetal("Shaders/Post/tonemap", projectRoot);
    if (tonemapMetalSource.empty()) {
        spdlog::error("Failed to compile tonemap shader");
        return 1;
    }

    NS::String* tonemapSourceStr = NS::String::string(tonemapMetalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* tonemapCompileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* tonemapLibrary = device->newLibrary(tonemapSourceStr, tonemapCompileOpts, &error);
    tonemapCompileOpts->release();
    if (!tonemapLibrary) {
        spdlog::error("Failed to create tonemap Metal library: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }

    MTL::Function* tonemapVertexFn = tonemapLibrary->newFunction(
        NS::String::string("vertexMain", NS::UTF8StringEncoding));
    MTL::Function* tonemapFragmentFn = tonemapLibrary->newFunction(
        NS::String::string("fragmentMain", NS::UTF8StringEncoding));

    auto* tonemapPipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    tonemapPipelineDesc->setVertexFunction(tonemapVertexFn);
    tonemapPipelineDesc->setFragmentFunction(tonemapFragmentFn);
    tonemapPipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    MTL::RenderPipelineState* tonemapPipelineState =
        device->newRenderPipelineState(tonemapPipelineDesc, &error);
    if (!tonemapPipelineState) {
        spdlog::error("Failed to create tonemap pipeline state: {}",
                      error->localizedDescription()->utf8String());
        return 1;
    }
    tonemapPipelineDesc->release();
    if (tonemapVertexFn) tonemapVertexFn->release();
    if (tonemapFragmentFn) tonemapFragmentFn->release();
    tonemapLibrary->release();

    auto* tonemapSamplerDesc = MTL::SamplerDescriptor::alloc()->init();
    tonemapSamplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    tonemapSamplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    tonemapSamplerDesc->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
    tonemapSamplerDesc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    tonemapSamplerDesc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    MTL::SamplerState* tonemapSampler = device->newSamplerState(tonemapSamplerDesc);
    tonemapSamplerDesc->release();
    if (!tonemapSampler) {
        spdlog::error("Failed to create tonemap sampler state");
        return 1;
    }

    // Passthrough (output) pipeline — same pattern as tonemap but no processing
    MTL::RenderPipelineState* outputPipelineState =
        reloadFullscreenShader(device, "Shaders/Post/passthrough", projectRoot, MTL::PixelFormatBGRA8Unorm);
    if (!outputPipelineState) {
        spdlog::error("Failed to create output passthrough pipeline");
        return 1;
    }

    AtmosphereTextureSet atmosphereTextures;
    bool atmosphereLoaded = loadAtmosphereTextures(device, projectRoot, atmosphereTextures);
    if (!atmosphereLoaded) {
        spdlog::warn("Atmosphere textures not found or invalid; sky pass will use fallback");
    }
    bool skyAvailable = atmosphereLoaded && skyPipelineState;

    int renderMode = 0; // 0=Vertex, 1=Mesh, 2=Visibility Buffer
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool enableRTShadows = true;
    bool enableAtmosphereSky = skyAvailable;
    float skyExposure = 10.0f;
    bool enableTonemap = true;
    int tonemapMethod = 0;
    float tonemapExposure = 1.0f;
    float tonemapContrast = 1.0f;
    float tonemapBrightness = 1.0f;
    float tonemapSaturation = 1.0f;
    float tonemapVignette = 0.0f;
    bool tonemapDither = true;
    bool showGraphDebug = false;
    bool showSceneGraphWindow = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool exportGraphKeyDown = false;
    bool reloadKeyDown = false;
    bool pipelineReloadKeyDown = false;

    // Load pipeline assets
    PipelineAsset visPipelineAsset;
    PipelineAsset fwdPipelineAsset;
    std::string visPipelinePath = std::string(projectRoot) + "/Pipelines/visibility_buffer.json";
    std::string fwdPipelinePath = std::string(projectRoot) + "/Pipelines/forward.json";

    if (!loadPipelineAssetChecked(visPipelinePath, "visibility buffer", visPipelineAsset) ||
        !loadPipelineAssetChecked(fwdPipelinePath, "forward", fwdPipelineAsset)) {
        return 1;
    }

    const double depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;

    // Create depth stencil state
    MTL::DepthStencilDescriptor* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(
        ML_DEPTH_REVERSED ? MTL::CompareFunctionGreater : MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(true);
    MTL::DepthStencilState* depthState = device->newDepthStencilState(depthDesc);
    depthDesc->release();

    // 1x1 depth texture used only to tell ImGui about the depth pixel format
    auto* imguiDepthTexDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, 1, 1, false);
    imguiDepthTexDesc->setStorageMode(MTL::StorageModePrivate);
    imguiDepthTexDesc->setUsage(MTL::TextureUsageRenderTarget);
    MTL::Texture* imguiDepthDummy = device->newTexture(imguiDepthTexDesc);

    // 1x1 shadow texture for non-RT paths (white = fully lit)
    auto* shadowDummyDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatR8Unorm, 1, 1, false);
    shadowDummyDesc->setStorageMode(MTL::StorageModeShared);
    shadowDummyDesc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* shadowDummyTex = device->newTexture(shadowDummyDesc);
    uint8_t shadowClear = 0xFF;
    shadowDummyTex->replaceRegion(MTL::Region(0, 0, 0, 1, 1, 1), 0, &shadowClear, 1);

    // 1x1 sky fallback texture (BGRA8)
    auto* skyFallbackDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatBGRA8Unorm, 1, 1, false);
    skyFallbackDesc->setStorageMode(MTL::StorageModeShared);
    skyFallbackDesc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* skyFallbackTex = device->newTexture(skyFallbackDesc);
    uint8_t skyFallbackColor[4] = {77, 51, 26, 255}; // B, G, R, A (~0.3, 0.2, 0.1)
    skyFallbackTex->replaceRegion(MTL::Region(0, 0, 0, 1, 1, 1), 0, skyFallbackColor, 4);

    // Create initial framebuffer size (used for metalLayer)
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

    // Persistent render context and pipeline builder (hoisted out of frame loop)
    RenderContext ctx{sceneMesh, meshletData, materials, sceneGraph,
                      shadowResources, depthState, shadowDummyTex, skyFallbackTex, depthClearValue};
    PipelineRuntimeContext rtCtx;
    rtCtx.device = device;
    rtCtx.renderPipelines["VisibilityPass"] = visPipelineState;
    rtCtx.renderPipelines["SkyPass"] = skyPipelineState;
    rtCtx.renderPipelines["TonemapPass"] = tonemapPipelineState;
    rtCtx.renderPipelines["ForwardPass"] = pipelineState;
    rtCtx.renderPipelines["ForwardMeshPass"] = meshPipelineState;
    rtCtx.renderPipelines["OutputPass"] = outputPipelineState;
    rtCtx.computePipelines["DeferredLightingPass"] = computePipelineState;
    rtCtx.samplers["tonemap"] = tonemapSampler;
    rtCtx.samplers["atmosphere"] = atmosphereTextures.sampler;
    rtCtx.importedTextures["transmittance"] = atmosphereTextures.transmittance;
    rtCtx.importedTextures["scattering"] = atmosphereTextures.scattering;
    rtCtx.importedTextures["irradiance"] = atmosphereTextures.irradiance;

    PipelineBuilder pipelineBuilder(ctx);
    bool pipelineNeedsRebuild = true;
    int lastRenderMode = -1;

    while (!glfwWindowShouldClose(window)) {
        ZoneScopedN("Frame");
        glfwPollEvents();

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            pool->release();
            continue;
        }
        metalLayer->setDrawableSize(CGSizeMake(width, height));

        CA::MetalDrawable* drawable = metalLayer->nextDrawable();
        if (!drawable) {
            pool->release();
            continue;
        }

        // Compute matrices
        float aspect;
        float4x4 view, proj, model, modelView, mvp;
        float4 worldLightDir, viewLightDir;
        float4 cameraWorldPos;
        Uniforms uniforms;
        AtmosphereUniforms skyUniforms;
        {
            ZoneScopedN("Matrix Computation");
            aspect = (float)width / (float)height;
            view = camera.viewMatrix();
            proj = camera.projectionMatrix(aspect);
            model = float4x4::Identity();
            modelView = view * model;
            mvp = proj * modelView;

            // Light data from scene graph sun source.
            DirectionalLight sunLight = sceneGraph.getSunDirectionalLight();
            worldLightDir = float4(sunLight.direction, 0.0f);
            viewLightDir = view * worldLightDir;

            uniforms.mvp = transpose(mvp);
            uniforms.modelView = transpose(modelView);
            uniforms.lightDir = viewLightDir;
            uniforms.lightColorIntensity = float4(
                sunLight.color.x,
                sunLight.color.y,
                sunLight.color.z,
                sunLight.intensity);

            // Extract frustum planes from non-transposed MVP (object-space planes)
            extractFrustumPlanes(mvp, uniforms.frustumPlanes);

            // Camera position in world space
            float cosA = std::cos(camera.azimuth), sinA = std::sin(camera.azimuth);
            float cosE = std::cos(camera.elevation), sinE = std::sin(camera.elevation);
            cameraWorldPos = float4(
                camera.target.x + camera.distance * cosE * sinA,
                camera.target.y + camera.distance * sinE,
                camera.target.z + camera.distance * cosE * cosA,
                1.0f);

            // Backface cone data is generated in object-space, so the camera needs to
            // be transformed to object-space for robust culling when model != identity.
            float4x4 invModel = model;
            invModel.Invert();
            uniforms.cameraPos = invModel * cameraWorldPos;

            uniforms.enableFrustumCull = enableFrustumCull ? 1 : 0;
            uniforms.enableConeCull = enableConeCull ? 1 : 0;
            uniforms.meshletBaseOffset = 0;
            uniforms.instanceID = 0;

            sceneGraph.updateTransforms();

            if (skyAvailable) {
                float4x4 viewProj = proj * view;
                float4x4 invViewProj = viewProj;
                invViewProj.Invert();
                skyUniforms.invViewProj = transpose(invViewProj);
                skyUniforms.cameraWorldPos = cameraWorldPos;
                skyUniforms.sunDirection = worldLightDir;
                skyUniforms.params = float4(skyExposure, 0.0f, 0.0f, 0.0f);
                skyUniforms.screenWidth = static_cast<uint32_t>(width);
                skyUniforms.screenHeight = static_cast<uint32_t>(height);
                skyUniforms.pad0 = 0;
                skyUniforms.pad1 = 0;
            }
        }

        // Render pass
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

        // Collect GPU timestamps from previous frames
        tracyMetalCollect(tracyGpuCtx);

        // ImGui new frame (needs a render pass descriptor for Metal backend)
        {
            ZoneScopedN("ImGui Frame");
        MTL::RenderPassDescriptor* imguiFramePass = MTL::RenderPassDescriptor::alloc()->init();
        imguiFramePass->colorAttachments()->object(0)->setTexture(drawable->texture());
        // Always provide a depth attachment so the ImGui pipeline matches even when
        // switching render modes within the same frame.
        imguiFramePass->depthAttachment()->setTexture(imguiDepthDummy);
        imguiNewFrame(imguiFramePass);
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Scene Graph", nullptr, &showSceneGraphWindow);
                ImGui::MenuItem("Render Passes", nullptr, &showRenderPassUI);
                ImGui::MenuItem("FrameGraph", nullptr, &showGraphDebug);
                ImGui::MenuItem("ImGui Demo", nullptr, &showImGuiDemo);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Renderer");
        ImGui::Text("%.1f FPS (%.3f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Separator();
        ImGui::RadioButton("Vertex Shader", &renderMode, 0);
        ImGui::RadioButton("Mesh Shader", &renderMode, 1);
        ImGui::RadioButton("Visibility Buffer", &renderMode, 2);
        if (renderMode >= 1) {
            ImGui::Text("Meshlets: %u", meshletData.meshletCount);
            ImGui::Checkbox("Frustum Culling", &enableFrustumCull);
            ImGui::Checkbox("Backface Culling", &enableConeCull);
        }
        if (renderMode == 2 && rtShadowsAvailable) {
            ImGui::Checkbox("RT Shadows", &enableRTShadows);
        }
        if (skyAvailable) {
            ImGui::Checkbox("Atmosphere Sky", &enableAtmosphereSky);
            ImGui::SliderFloat("Sky Exposure", &skyExposure, 0.1f, 20.0f, "%.2f");
        } else {
            ImGui::TextDisabled("Atmosphere Sky (missing textures)");
        }
        ImGui::Separator();
        ImGui::Text("Tonemapping");
        ImGui::Checkbox("Enable Tonemap", &enableTonemap);
        const char* tonemapLabels[] = {
            "Filmic", "Uncharted2", "Clip", "ACES", "AgX", "Khronos PBR"
        };
        ImGui::Combo("Tonemap Method", &tonemapMethod, tonemapLabels, IM_ARRAYSIZE(tonemapLabels));
        ImGui::SliderFloat("Exposure", &tonemapExposure, 0.1f, 4.0f, "%.2f");
        ImGui::SliderFloat("Contrast", &tonemapContrast, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Brightness", &tonemapBrightness, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Saturation", &tonemapSaturation, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Vignette", &tonemapVignette, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Dither", &tonemapDither);
        ImGui::Checkbox("Show Graph", &showGraphDebug);
        ImGui::Separator();
        bool f5Down = glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
        bool triggerReload = ImGui::Button("Reload Shaders (F5)") || (f5Down && !reloadKeyDown);
        reloadKeyDown = f5Down;
        if (triggerReload) {
            spdlog::info("Reloading shaders...");
            int reloaded = 0, failed = 0;

            // 1. Vertex shader
            if (auto* p = reloadVertexShader(device, "Shaders/Vertex/bunny", projectRoot, vertexDesc)) {
                pipelineState->release();
                pipelineState = p;
                reloaded++;
            } else { failed++; }

            // 2. Mesh shader
            if (auto* p = reloadMeshShader(device, "Shaders/Mesh/meshlet", projectRoot,
                    patchMeshShaderMetalSource, MTL::PixelFormatRGBA16Float, MTL::PixelFormatDepth32Float)) {
                meshPipelineState->release();
                meshPipelineState = p;
                reloaded++;
            } else { failed++; }

            // 3. Visibility shader
            if (auto* p = reloadMeshShader(device, "Shaders/Visibility/visibility", projectRoot,
                    patchVisibilityShaderMetalSource, MTL::PixelFormatR32Uint, MTL::PixelFormatDepth32Float)) {
                visPipelineState->release();
                visPipelineState = p;
                reloaded++;
            } else { failed++; }

            // 4. Compute shader (deferred lighting)
            if (auto* p = reloadComputeShader(device, "Shaders/Visibility/deferred_lighting", projectRoot,
                    "computeMain", patchComputeShaderMetalSource)) {
                computePipelineState->release();
                computePipelineState = p;
                reloaded++;
            } else { failed++; }

            // 5. Sky shader
            if (auto* p = reloadFullscreenShader(device, "Shaders/Atmosphere/sky", projectRoot,
                    MTL::PixelFormatRGBA16Float)) {
                if (skyPipelineState) skyPipelineState->release();
                skyPipelineState = p;
                skyAvailable = atmosphereLoaded && skyPipelineState;
                reloaded++;
            } else { failed++; }

            // 6. Tonemap shader
            if (auto* p = reloadFullscreenShader(device, "Shaders/Post/tonemap", projectRoot,
                    MTL::PixelFormatBGRA8Unorm)) {
                tonemapPipelineState->release();
                tonemapPipelineState = p;
                reloaded++;
            } else { failed++; }

            // 7. Passthrough (output) shader
            if (auto* p = reloadFullscreenShader(device, "Shaders/Post/passthrough", projectRoot,
                    MTL::PixelFormatBGRA8Unorm)) {
                outputPipelineState->release();
                outputPipelineState = p;
                reloaded++;
            } else { failed++; }

            // 8. Shadow ray shader
            if (rtShadowsAvailable) {
                if (reloadShadowPipeline(device, shadowResources, projectRoot)) {
                    reloaded++;
                } else { failed++; }
            }

            if (failed == 0)
                spdlog::info("All {} shaders reloaded successfully", reloaded);
            else
                spdlog::warn("{} shaders reloaded, {} failed (keeping old pipelines)", reloaded, failed);

            // Update runtime context with reloaded pipelines
            rtCtx.renderPipelines["VisibilityPass"] = visPipelineState;
            rtCtx.renderPipelines["SkyPass"] = skyPipelineState;
            rtCtx.renderPipelines["TonemapPass"] = tonemapPipelineState;
            rtCtx.renderPipelines["ForwardPass"] = pipelineState;
            rtCtx.renderPipelines["ForwardMeshPass"] = meshPipelineState;
            rtCtx.renderPipelines["OutputPass"] = outputPipelineState;
            rtCtx.computePipelines["DeferredLightingPass"] = computePipelineState;
            pipelineNeedsRebuild = true;
        }
        ImGui::End();
        imguiFramePass->release();
        } // end ImGui Frame zone

        if (showSceneGraphWindow)
            drawSceneGraphUI(sceneGraph);

        if (showImGuiDemo)
            ImGui::ShowDemoWindow(&showImGuiDemo);

        // Pipeline hot-reload (F6)
        bool f6Down = glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS;
        if (f6Down && !pipelineReloadKeyDown) {
            bool reloadedAnyPipeline = false;

            PipelineAsset reloadedVis;
            if (loadPipelineAssetChecked(visPipelinePath, "visibility buffer", reloadedVis)) {
                visPipelineAsset = std::move(reloadedVis);
                reloadedAnyPipeline = true;
            } else {
                spdlog::warn("Keeping previous visibility buffer pipeline: {}", visPipelineAsset.name);
            }

            PipelineAsset reloadedFwd;
            if (loadPipelineAssetChecked(fwdPipelinePath, "forward", reloadedFwd)) {
                fwdPipelineAsset = std::move(reloadedFwd);
                reloadedAnyPipeline = true;
            } else {
                spdlog::warn("Keeping previous forward pipeline: {}", fwdPipelineAsset.name);
            }

            pipelineNeedsRebuild = reloadedAnyPipeline;
        }
        pipelineReloadKeyDown = f6Down;

        std::vector<uint32_t> visibleMeshletNodes;
        std::vector<uint32_t> visibleIndexNodes;
        visibleMeshletNodes.reserve(sceneGraph.nodes.size());
        visibleIndexNodes.reserve(sceneGraph.nodes.size());
        for (const auto& node : sceneGraph.nodes) {
            if (!sceneGraph.isNodeVisible(node.id))
                continue;
            if (node.meshletCount > 0)
                visibleMeshletNodes.push_back(node.id);
            if (node.indexCount > 0)
                visibleIndexNodes.push_back(node.id);
        }

        TonemapUniforms tonemapUniforms{};
        tonemapUniforms.isActive = enableTonemap ? 1u : 0u;
        tonemapUniforms.method = static_cast<uint32_t>(tonemapMethod);
        tonemapUniforms.exposure = tonemapExposure;
        tonemapUniforms.contrast = tonemapContrast;
        tonemapUniforms.brightness = tonemapBrightness;
        tonemapUniforms.saturation = tonemapSaturation;
        tonemapUniforms.vignette = tonemapVignette;
        tonemapUniforms.dither = tonemapDither ? 1u : 0u;
        tonemapUniforms.invResolution = float2(1.0f / float(width), 1.0f / float(height));
        tonemapUniforms.pad = float2(0.0f, 0.0f);

        // --- Build FrameGraph (unified data-driven path) ---
        MTL::Buffer* instanceTransformBuffer = nullptr;

        FrameContext frameCtx;

        // Visibility buffer mode needs instance transform buffer
        uint32_t visibilityInstanceCount = 0;
        if (renderMode == 2) {
            ZoneScopedN("Visibility Instance Setup");

            static bool warnedInstanceOverflow = false;
            if (!warnedInstanceOverflow &&
                visibleMeshletNodes.size() > static_cast<size_t>(kVisibilityInstanceMask + 1)) {
                spdlog::warn("Visibility buffer instance limit exceeded ({} > {}), extra nodes will be skipped in this mode",
                             visibleMeshletNodes.size(), kVisibilityInstanceMask + 1);
                warnedInstanceOverflow = true;
            }

            static bool warnedMeshletOverflow = false;
            if (!warnedMeshletOverflow &&
                meshletData.meshletCount > (kVisibilityMeshletMask + 1u)) {
                spdlog::warn("Visibility meshlet id limit exceeded ({} > {}), overflowing meshlets will be culled",
                             meshletData.meshletCount, kVisibilityMeshletMask + 1);
                warnedMeshletOverflow = true;
            }

            visibilityInstanceCount =
                static_cast<uint32_t>(std::min<size_t>(visibleMeshletNodes.size(),
                                                       kVisibilityInstanceMask + 1));

            std::vector<SceneInstanceTransform> visibilityInstanceTransforms;
            visibilityInstanceTransforms.reserve(std::max<size_t>(visibilityInstanceCount, 1));
            for (uint32_t instanceID = 0; instanceID < visibilityInstanceCount; instanceID++) {
                const auto& node = sceneGraph.nodes[visibleMeshletNodes[instanceID]];
                float4x4 nodeModelView = view * node.transform.worldMatrix;
                float4x4 nodeMVP = proj * nodeModelView;
                visibilityInstanceTransforms.push_back({transpose(nodeMVP), transpose(nodeModelView)});
            }

            if (visibilityInstanceTransforms.empty()) {
                visibilityInstanceTransforms.push_back({transpose(mvp), transpose(modelView)});
            }

            instanceTransformBuffer = device->newBuffer(
                visibilityInstanceTransforms.data(),
                visibilityInstanceTransforms.size() * sizeof(SceneInstanceTransform),
                MTL::ResourceStorageModeShared);
        }

        // Populate frame context (shared across all modes)
        frameCtx.width = width;
        frameCtx.height = height;
        frameCtx.view = view;
        frameCtx.proj = proj;
        frameCtx.cameraWorldPos = cameraWorldPos;
        frameCtx.worldLightDir = worldLightDir;
        frameCtx.viewLightDir = viewLightDir;
        frameCtx.lightColorIntensity = uniforms.lightColorIntensity;
        frameCtx.baseUniforms = uniforms;
        frameCtx.skyUniforms = skyUniforms;
        frameCtx.tonemapUniforms = tonemapUniforms;
        frameCtx.visibleMeshletNodes = visibleMeshletNodes;
        frameCtx.visibleIndexNodes = visibleIndexNodes;
        frameCtx.visibilityInstanceCount = visibilityInstanceCount;
        frameCtx.instanceTransformBuffer = instanceTransformBuffer;
        frameCtx.commandBuffer = commandBuffer;
        frameCtx.depthClearValue = depthClearValue;
        frameCtx.cameraFarZ = camera.farZ;
        frameCtx.enableFrustumCull = enableFrustumCull;
        frameCtx.enableConeCull = enableConeCull;
        frameCtx.enableRTShadows = rtShadowsAvailable && enableRTShadows;
        frameCtx.enableAtmosphereSky = skyAvailable && enableAtmosphereSky;
        frameCtx.renderMode = renderMode;

        if (renderMode == 2) {
            // Lighting uniforms (only needed for visibility buffer deferred lighting)
            LightingUniforms lightUniforms;
            lightUniforms.mvp = transpose(mvp);
            lightUniforms.modelView = transpose(modelView);
            lightUniforms.lightDir = viewLightDir;
            lightUniforms.lightColorIntensity = uniforms.lightColorIntensity;
            float4x4 invProj = proj;
            invProj.Invert();
            lightUniforms.invProj = transpose(invProj);
            lightUniforms.screenWidth = (uint32_t)width;
            lightUniforms.screenHeight = (uint32_t)height;
            lightUniforms.meshletCount = meshletData.meshletCount;
            lightUniforms.materialCount = materials.materialCount;
            lightUniforms.textureCount = static_cast<uint32_t>(materials.textures.size());
            lightUniforms.instanceCount = visibilityInstanceCount;
            lightUniforms.shadowEnabled = frameCtx.enableRTShadows ? 1 : 0;
            lightUniforms.pad2 = 0;
            frameCtx.lightingUniforms = lightUniforms;
        }

        // Select active pipeline asset based on render mode
        const PipelineAsset& activePipelineAsset = (renderMode == 2) ? visPipelineAsset : fwdPipelineAsset;

        // Detect mode switch
        if (renderMode != lastRenderMode) {
            pipelineNeedsRebuild = true;
            lastRenderMode = renderMode;
        }

        // Rebuild pipeline only when needed (first frame, F6 reload, resolution change, mode switch)
        if (pipelineNeedsRebuild || pipelineBuilder.needsRebuild(width, height)) {
            rtCtx.backbuffer = drawable->texture();
            bool buildSucceeded = pipelineBuilder.build(activePipelineAsset, rtCtx, width, height);
            if (!buildSucceeded) {
                spdlog::error("Failed to build pipeline: {}", pipelineBuilder.lastError());
            } else {
                pipelineBuilder.compile();
            }
            pipelineNeedsRebuild = !buildSucceeded;
        }

        // Per-frame: swap backbuffer, update frame context, reset transients
        pipelineBuilder.updateFrame(drawable->texture(), &frameCtx);

        FrameGraph& activeFg = pipelineBuilder.frameGraph();

        if (showGraphDebug)
            activeFg.debugImGui();

        if (showRenderPassUI)
            activeFg.renderPassUI();

        ImGui::Render();

        bool gKeyDown = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
        if (gKeyDown && !exportGraphKeyDown) {
            std::ofstream dot("framegraph.dot");
            if (dot.is_open()) {
                activeFg.exportGraphviz(dot);
                spdlog::info("Exported framegraph.dot");
            }
        }
        exportGraphKeyDown = gKeyDown;

        // Update TLAS with current scene transforms before executing the frame graph
        if (rtShadowsAvailable && renderMode == 2) {
            ZoneScopedN("Update TLAS");
            updateTLAS(commandBuffer, sceneGraph, shadowResources);
        }

        activeFg.execute(commandBuffer, device, tracyGpuCtx);

        commandBuffer->presentDrawable(drawable);
        commandBuffer->commit();

        if (instanceTransformBuffer)
            instanceTransformBuffer->release();

        FrameMark;
        pool->release();
    }

    // Cleanup ImGui
    imguiShutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup Tracy GPU context
    tracyMetalDestroy(tracyGpuCtx);

    // Cleanup
    shadowResources.release();
    imguiDepthDummy->release();
    shadowDummyTex->release();
    skyFallbackTex->release();
    atmosphereTextures.release();
    meshletData.meshletBuffer->release();
    meshletData.meshletVertices->release();
    meshletData.meshletTriangles->release();
    meshletData.boundsBuffer->release();
    meshletData.materialIDs->release();
    for (auto* tex : materials.textures) {
        if (tex) tex->release();
    }
    materials.materialBuffer->release();
    materials.sampler->release();
    sceneMesh.positionBuffer->release();
    sceneMesh.normalBuffer->release();
    sceneMesh.uvBuffer->release();
    sceneMesh.indexBuffer->release();
    depthState->release();
    vertexDesc->release();
    if (skyPipelineState) skyPipelineState->release();
    computePipelineState->release();
    tonemapSampler->release();
    tonemapPipelineState->release();
    outputPipelineState->release();
    visPipelineState->release();
    meshPipelineState->release();
    pipelineState->release();
    commandQueue->release();
    device->release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
