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
#include "imgui_overlay_pass.h"
#include "visibility_pass.h"
#include "shadow_ray_pass.h"
#include "deferred_lighting_pass.h"
#include "forward_pass.h"


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
    desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
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
    pipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
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
    meshPipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
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

    int renderMode = 0; // 0=Vertex, 1=Mesh, 2=Visibility Buffer
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool enableRTShadows = true;
    bool showGraphDebug = false;
    bool showSceneGraphWindow = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool exportGraphKeyDown = false;
    bool reloadKeyDown = false;

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

    // Create initial framebuffer size (used for metalLayer)
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

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

        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::SetNextWindowBgAlpha(0.7f);
        ImGui::Begin("##fps", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
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
                    patchMeshShaderMetalSource, MTL::PixelFormatBGRA8Unorm, MTL::PixelFormatDepth32Float)) {
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

            // 5. Shadow ray shader
            if (rtShadowsAvailable) {
                if (reloadShadowPipeline(device, shadowResources, projectRoot)) {
                    reloaded++;
                } else { failed++; }
            }

            if (failed == 0)
                spdlog::info("All {} shaders reloaded successfully", reloaded);
            else
                spdlog::warn("{} shaders reloaded, {} failed (keeping old pipelines)", reloaded, failed);
        }
        ImGui::End();
        imguiFramePass->release();
        } // end ImGui Frame zone

        if (showSceneGraphWindow)
            drawSceneGraphUI(sceneGraph);

        if (showImGuiDemo)
            ImGui::ShowDemoWindow(&showImGuiDemo);

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

        // --- Build FrameGraph ---
        FrameGraph fg;
        auto drawableRes = fg.import("drawable", drawable->texture());
        MTL::Buffer* instanceTransformBuffer = nullptr;

        RenderContext ctx{sceneMesh, meshletData, materials, sceneGraph,
                          shadowResources, depthState, shadowDummyTex, depthClearValue};

        if (renderMode == 2) {
            // === VISIBILITY BUFFER MODE ===
            ZoneScopedN("Visibility Buffer Graph Build");

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

            uint32_t visibilityInstanceCount =
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

            auto visPass = std::make_unique<VisibilityPass>(
                ctx, visPipelineState, uniforms,
                visibleMeshletNodes, visibilityInstanceCount,
                view, proj, cameraWorldPos, width, height);
            auto* visPassPtr = visPass.get();
            fg.addPass(std::move(visPass));
            FGResource visDepth = visPassPtr->depth;
            FGResource visVisibility = visPassPtr->visibility;
            bool shadowPassActive = rtShadowsAvailable && enableRTShadows;
            FGResource shadowMapRes{};
            if (rtShadowsAvailable) {
                auto shadowPass = std::make_unique<ShadowRayPass>(
                    ctx, visDepth, view, proj,
                    worldLightDir, camera.farZ, shadowPassActive,
                    width, height);
                auto* shadowPassPtr = shadowPass.get();
                fg.addPass(std::move(shadowPass));
                shadowMapRes = shadowPassPtr->shadowMap;
            }

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
            lightUniforms.shadowEnabled = (rtShadowsAvailable && enableRTShadows) ? 1 : 0;
            lightUniforms.pad2 = 0;

            auto lightingPass = std::make_unique<DeferredLightingPass>(
                ctx, computePipelineState,
                visVisibility, visDepth,
                shadowMapRes, instanceTransformBuffer,
                lightUniforms, width, height);
            auto* lightingPassPtr = lightingPass.get();
            fg.addPass(std::move(lightingPass));
            FGResource lightingOutput = lightingPassPtr->output;

            auto blitPass = std::make_unique<BlitPass>(
                lightingOutput, drawableRes, width, height);
            fg.addPass(std::move(blitPass));

            auto imguiPass = std::make_unique<ImGuiOverlayPass>(
                drawableRes, visDepth, depthClearValue, commandBuffer);
            fg.addPass(std::move(imguiPass));
        } else {
            // === FORWARD RENDERING (Vertex or Mesh shader) ===
            ZoneScopedN("Forward Graph Build");

            auto fwdPass = std::make_unique<ForwardPass>(
                ctx, drawableRes, renderMode,
                pipelineState, meshPipelineState, uniforms,
                visibleMeshletNodes, visibleIndexNodes,
                view, proj, cameraWorldPos,
                commandBuffer, width, height);
            fg.addPass(std::move(fwdPass));
        }

        fg.compile();

        if (showGraphDebug)
            fg.debugImGui();

        if (showRenderPassUI)
            fg.renderPassUI();

        ImGui::Render();

        bool gKeyDown = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
        if (gKeyDown && !exportGraphKeyDown) {
            std::ofstream dot("framegraph.dot");
            if (dot.is_open()) {
                fg.exportGraphviz(dot);
                spdlog::info("Exported framegraph.dot");
            }
        }
        exportGraphKeyDown = gKeyDown;

        // Update TLAS with current scene transforms before executing the frame graph
        if (rtShadowsAvailable && renderMode == 2) {
            ZoneScopedN("Update TLAS");
            updateTLAS(commandBuffer, sceneGraph, shadowResources);
        }

        fg.execute(commandBuffer, device, tracyGpuCtx);

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
    computePipelineState->release();
    visPipelineState->release();
    meshPipelineState->release();
    pipelineState->release();
    commandQueue->release();
    device->release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
