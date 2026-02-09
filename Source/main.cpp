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

#include <slang.h>
#include <slang-com-ptr.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include <fstream>

#include <tracy/Tracy.hpp>
#include "tracy_metal.h"
#include "frame_graph.h"


struct Uniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir;
    float4   frustumPlanes[6];
    float4   cameraPos; // object-space camera position
    uint32_t enableFrustumCull;
    uint32_t enableConeCull;
    uint32_t meshletBaseOffset;
    uint32_t instanceID;
};

static constexpr uint32_t kVisibilityInstanceBits = 11;
static constexpr uint32_t kVisibilityInstanceMask = (1u << kVisibilityInstanceBits) - 1u;

static void extractFrustumPlanes(const float4x4& mvp, float4* planes) {
    // Gribb-Hartmann method: extract planes from MVP matrix rows
    float4 row0 = mvp.Row(0);
    float4 row1 = mvp.Row(1);
    float4 row2 = mvp.Row(2);
    float4 row3 = mvp.Row(3);

    planes[0] = row3 + row0; // left
    planes[1] = row3 - row0; // right
    planes[2] = row3 + row1; // bottom
    planes[3] = row3 - row1; // top
    planes[4] = row2;        // near  (Metal NDC z in [0,1])
    planes[5] = row3 - row2; // far

    // Normalize each plane
    for (int i = 0; i < 6; i++) {
        float len = length(planes[i].xyz);
        if (len > 0.0f)
            planes[i] /= len;
    }
}

static std::string compileSlangToMetal(const char* shaderPath) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
            std::cerr << "Slang load error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
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
            std::cerr << "Slang compile error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

static std::string compileSlangMeshShaderToMetal(const char* shaderPath) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
            std::cerr << "Slang load error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
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
            std::cerr << "Slang mesh shader compile error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

static std::string compileSlangComputeShaderToMetal(const char* shaderPath) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
            std::cerr << "Slang load error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
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
            std::cerr << "Slang compute shader compile error: "
                      << static_cast<const char*>(diagnostics->getBufferPointer()) << std::endl;
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


struct LightingUniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir;
    float4x4 invProj;
    uint32_t screenWidth;
    uint32_t screenHeight;
    uint32_t meshletCount;
    uint32_t materialCount;
    uint32_t textureCount;
    uint32_t instanceCount;
    uint32_t pad1;
    uint32_t pad2;
};

struct SceneInstanceTransform {
    float4x4 mvp;
    float4x4 modelView;
};

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "RenderGraph - Sponza", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    std::cout << "Metal device: " << device->name()->utf8String() << std::endl;

    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Tracy GPU profiling context
    TracyMetalCtxHandle tracyGpuCtx = tracyMetalCreate(device);

    CA::MetalLayer* metalLayer = static_cast<CA::MetalLayer*>(
        attachMetalLayerToGLFWWindow(window));
    metalLayer->setDevice(device);
    metalLayer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    // Load scene mesh
    LoadedMesh sceneMesh;
    if (!loadGLTFMesh(device, "Asset/Sponza/glTF/Sponza.gltf", sceneMesh)) {
        std::cerr << "Failed to load scene mesh" << std::endl;
        return 1;
    }

    // Build meshlets for mesh shader rendering
    MeshletData meshletData;
    if (!buildMeshlets(device, sceneMesh, meshletData)) {
        std::cerr << "Failed to build meshlets" << std::endl;
        return 1;
    }

    // Load materials and textures
    LoadedMaterials materials;
    if (!loadGLTFMaterials(device, commandQueue, "Asset/Sponza/glTF/Sponza.gltf", materials)) {
        std::cerr << "Failed to load materials" << std::endl;
        return 1;
    }

    // Build scene graph from glTF node hierarchy
    SceneGraph sceneGraph;
    if (!sceneGraph.buildFromGLTF("Asset/Sponza/glTF/Sponza.gltf", sceneMesh, meshletData)) {
        std::cerr << "Failed to build scene graph" << std::endl;
        return 1;
    }
    sceneGraph.updateTransforms();

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
    std::string metalSource = compileSlangToMetal("Shaders/bunny");
    if (metalSource.empty()) {
        std::cerr << "Failed to compile Slang shader" << std::endl;
        return 1;
    }
    std::cout << "Slang compiled Metal shader (" << metalSource.size() << " bytes)" << std::endl;

    // Create Metal library from compiled source
    NS::Error* error = nullptr;
    NS::String* sourceStr = NS::String::string(metalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* compileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* library = device->newLibrary(sourceStr, compileOpts, &error);
    compileOpts->release();
    if (!library) {
        std::cerr << "Failed to create Metal library: "
                  << error->localizedDescription()->utf8String() << std::endl;
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
    vertexDesc->release();

    MTL::RenderPipelineState* pipelineState =
        device->newRenderPipelineState(pipelineDesc, &error);
    if (!pipelineState) {
        std::cerr << "Failed to create pipeline state: "
                  << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }
    pipelineDesc->release();
    vertexFn->release();
    fragmentFn->release();
    library->release();

    // --- Mesh shader pipeline ---
    std::string meshMetalSource = compileSlangMeshShaderToMetal("Shaders/meshlet");
    if (meshMetalSource.empty()) {
        std::cerr << "Failed to compile Slang mesh shader" << std::endl;
        return 1;
    }
    meshMetalSource = patchMeshShaderMetalSource(meshMetalSource);
    std::cout << "Mesh shader compiled (" << meshMetalSource.size() << " bytes)" << std::endl;

    NS::String* meshSourceStr = NS::String::string(meshMetalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* meshCompileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* meshLibrary = device->newLibrary(meshSourceStr, meshCompileOpts, &error);
    meshCompileOpts->release();
    if (!meshLibrary) {
        std::cerr << "Failed to create mesh Metal library: "
                  << error->localizedDescription()->utf8String() << std::endl;
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
        std::cerr << "Failed to create mesh pipeline state: "
                  << meshError->localizedDescription()->utf8String() << std::endl;
        return 1;
    }
    meshPipelineDesc->release();
    meshFn->release();
    meshFragFn->release();
    meshLibrary->release();

    // --- Visibility buffer mesh shader pipeline ---
    std::string visMetalSource = compileSlangMeshShaderToMetal("Shaders/visibility");
    if (visMetalSource.empty()) {
        std::cerr << "Failed to compile visibility shader" << std::endl;
        return 1;
    }
    visMetalSource = patchVisibilityShaderMetalSource(visMetalSource);
    std::cout << "Visibility shader compiled (" << visMetalSource.size() << " bytes)" << std::endl;

    NS::String* visSourceStr = NS::String::string(visMetalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* visCompileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* visLibrary = device->newLibrary(visSourceStr, visCompileOpts, &error);
    visCompileOpts->release();
    if (!visLibrary) {
        std::cerr << "Failed to create visibility Metal library: "
                  << error->localizedDescription()->utf8String() << std::endl;
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
        std::cerr << "Failed to create visibility pipeline state: "
                  << visError->localizedDescription()->utf8String() << std::endl;
        return 1;
    }
    visPipelineDesc->release();
    visMeshFn->release();
    visFragFn->release();
    visLibrary->release();

    // --- Deferred lighting compute pipeline ---
    std::string computeMetalSource = compileSlangComputeShaderToMetal("Shaders/deferred_lighting");
    if (computeMetalSource.empty()) {
        std::cerr << "Failed to compile deferred lighting shader" << std::endl;
        return 1;
    }
    computeMetalSource = patchComputeShaderMetalSource(computeMetalSource);
    std::cout << "Compute shader compiled (" << computeMetalSource.size() << " bytes)" << std::endl;

    NS::String* computeSourceStr = NS::String::string(computeMetalSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* computeCompileOpts = MTL::CompileOptions::alloc()->init();
    MTL::Library* computeLibrary = device->newLibrary(computeSourceStr, computeCompileOpts, &error);
    computeCompileOpts->release();
    if (!computeLibrary) {
        std::cerr << "Failed to create compute Metal library: "
                  << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    MTL::Function* computeFn = computeLibrary->newFunction(
        NS::String::string("computeMain", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* computePipelineState =
        device->newComputePipelineState(computeFn, &error);
    if (!computePipelineState) {
        std::cerr << "Failed to create compute pipeline state: "
                  << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }
    computeFn->release();
    computeLibrary->release();

    int renderMode = 0; // 0=Vertex, 1=Mesh, 2=Visibility Buffer
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool showGraphDebug = false;
    bool showSceneGraphWindow = true;
    bool showImGuiDemo = false;
    bool exportGraphKeyDown = false;

    // Create depth stencil state
    MTL::DepthStencilDescriptor* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(true);
    MTL::DepthStencilState* depthState = device->newDepthStencilState(depthDesc);
    depthDesc->release();

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

            // Light direction in view space (from upper-right-front)
            worldLightDir = float4(normalize(float3(0.5f, 1.0f, 0.8f)), 0.0f);
            viewLightDir = view * worldLightDir;

            uniforms.mvp = transpose(mvp);
            uniforms.modelView = transpose(modelView);
            uniforms.lightDir = viewLightDir;

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
        imguiNewFrame(imguiFramePass);
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Scene Graph", nullptr, &showSceneGraphWindow);
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
        ImGui::Checkbox("Show Graph", &showGraphDebug);
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

        if (renderMode == 2) {
            // === VISIBILITY BUFFER MODE ===
            ZoneScopedN("Visibility Buffer Graph Build");

            static bool warnedInstanceOverflow = false;
            if (!warnedInstanceOverflow &&
                visibleMeshletNodes.size() > static_cast<size_t>(kVisibilityInstanceMask + 1)) {
                std::cerr << "Visibility buffer instance limit exceeded ("
                          << visibleMeshletNodes.size() << " > "
                          << (kVisibilityInstanceMask + 1)
                          << "), extra nodes will be skipped in this mode" << std::endl;
                warnedInstanceOverflow = true;
            }

            constexpr uint32_t kVisibilityTriangleBits = 7;
            constexpr uint32_t kVisibilityMeshletMask =
                (1u << (32u - (kVisibilityTriangleBits + kVisibilityInstanceBits))) - 1u;
            static bool warnedMeshletOverflow = false;
            if (!warnedMeshletOverflow && meshletData.meshletCount > kVisibilityMeshletMask) {
                std::cerr << "Visibility meshlet id limit exceeded ("
                          << meshletData.meshletCount << " > "
                          << (kVisibilityMeshletMask + 1)
                          << "), overflowing meshlets will be culled" << std::endl;
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

            struct VisPassData {
                FGResource visibility;
                FGResource depth;
            };
            auto& visData = fg.addRenderPass<VisPassData>("Visibility Pass",
                [&](FGBuilder& builder, VisPassData& data) {
                    data.visibility = builder.create("visibility",
                        FGTextureDesc::renderTarget(width, height, MTL::PixelFormatR32Uint));
                    data.depth = builder.create("depth",
                        FGTextureDesc::depthTarget(width, height));
                    builder.setColorAttachment(0, data.visibility,
                        MTL::LoadActionClear, MTL::StoreActionStore,
                        MTL::ClearColor(0xFFFFFFFF, 0, 0, 0));
                    builder.setDepthAttachment(data.depth,
                        MTL::LoadActionClear, MTL::StoreActionStore, 1.0);
                },
                [&](const VisPassData&, MTL::RenderCommandEncoder* enc) {
                    enc->setDepthStencilState(depthState);
                    enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
                    enc->setCullMode(MTL::CullModeBack);
                    enc->setRenderPipelineState(visPipelineState);
                    // Bind shared buffers once
                    enc->setMeshBuffer(sceneMesh.positionBuffer, 0, 1);
                    enc->setMeshBuffer(sceneMesh.normalBuffer, 0, 2);
                    enc->setMeshBuffer(meshletData.meshletBuffer, 0, 3);
                    enc->setMeshBuffer(meshletData.meshletVertices, 0, 4);
                    enc->setMeshBuffer(meshletData.meshletTriangles, 0, 5);
                    enc->setMeshBuffer(meshletData.boundsBuffer, 0, 6);
                    enc->setMeshBuffer(sceneMesh.uvBuffer, 0, 7);
                    enc->setMeshBuffer(meshletData.materialIDs, 0, 8);
                    enc->setMeshBuffer(materials.materialBuffer, 0, 9);
                    enc->setFragmentBuffer(sceneMesh.positionBuffer, 0, 1);
                    enc->setFragmentBuffer(sceneMesh.normalBuffer, 0, 2);
                    enc->setFragmentBuffer(meshletData.meshletBuffer, 0, 3);
                    enc->setFragmentBuffer(meshletData.meshletVertices, 0, 4);
                    enc->setFragmentBuffer(meshletData.meshletTriangles, 0, 5);
                    enc->setFragmentBuffer(meshletData.boundsBuffer, 0, 6);
                    enc->setFragmentBuffer(sceneMesh.uvBuffer, 0, 7);
                    enc->setFragmentBuffer(meshletData.materialIDs, 0, 8);
                    enc->setFragmentBuffer(materials.materialBuffer, 0, 9);
                    if (!materials.textures.empty()) {
                        enc->setFragmentTextures(
                            const_cast<MTL::Texture* const*>(materials.textures.data()),
                            NS::Range(0, materials.textures.size()));
                        enc->setMeshTextures(
                            const_cast<MTL::Texture* const*>(materials.textures.data()),
                            NS::Range(0, materials.textures.size()));
                    }
                    enc->setFragmentSamplerState(materials.sampler, 0);
                    enc->setMeshSamplerState(materials.sampler, 0);

                    // Per-node dispatch
                    for (uint32_t instanceID = 0; instanceID < visibilityInstanceCount; instanceID++) {
                        const auto& node = sceneGraph.nodes[visibleMeshletNodes[instanceID]];
                        float4x4 nodeModelView = view * node.transform.worldMatrix;
                        float4x4 nodeMVP = proj * nodeModelView;
                        Uniforms nodeUniforms = uniforms;
                        nodeUniforms.mvp = transpose(nodeMVP);
                        nodeUniforms.modelView = transpose(nodeModelView);
                        extractFrustumPlanes(nodeMVP, nodeUniforms.frustumPlanes);
                        float4x4 invModel = node.transform.worldMatrix;
                        invModel.Invert();
                        nodeUniforms.cameraPos = invModel * cameraWorldPos;
                        nodeUniforms.meshletBaseOffset = node.meshletStart;
                        nodeUniforms.instanceID = instanceID;
                        enc->setMeshBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                        enc->setFragmentBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                        enc->drawMeshThreadgroups(
                            MTL::Size(node.meshletCount, 1, 1),
                            MTL::Size(1, 1, 1),
                            MTL::Size(128, 1, 1));
                    }
                });

            // Compute lighting uniforms
            LightingUniforms lightUniforms;
            lightUniforms.mvp = transpose(mvp);
            lightUniforms.modelView = transpose(modelView);
            lightUniforms.lightDir = viewLightDir;
            float4x4 invProj = proj;
            invProj.Invert();
            lightUniforms.invProj = transpose(invProj);
            lightUniforms.screenWidth = (uint32_t)width;
            lightUniforms.screenHeight = (uint32_t)height;
            lightUniforms.meshletCount = meshletData.meshletCount;
            lightUniforms.materialCount = materials.materialCount;
            lightUniforms.textureCount = static_cast<uint32_t>(materials.textures.size());
            lightUniforms.instanceCount = visibilityInstanceCount;
            lightUniforms.pad1 = 0;
            lightUniforms.pad2 = 0;

            struct ComputePassData {
                FGResource visibility;
                FGResource depth;
                FGResource output;
            };
            auto& computeData = fg.addComputePass<ComputePassData>("Deferred Lighting",
                [&](FGBuilder& builder, ComputePassData& data) {
                    data.visibility = builder.read(visData.visibility);
                    data.depth = builder.read(visData.depth);
                    data.output = builder.create("output",
                        FGTextureDesc::storageTexture(width, height, MTL::PixelFormatBGRA8Unorm));
                },
                [&, lightUniforms](const ComputePassData& data, MTL::ComputeCommandEncoder* enc) {
                    enc->setComputePipelineState(computePipelineState);
                    enc->setBytes(&lightUniforms, sizeof(lightUniforms), 0);
                    enc->setBuffer(sceneMesh.positionBuffer, 0, 1);
                    enc->setBuffer(sceneMesh.normalBuffer, 0, 2);
                    enc->setBuffer(meshletData.meshletBuffer, 0, 3);
                    enc->setBuffer(meshletData.meshletVertices, 0, 4);
                    enc->setBuffer(meshletData.meshletTriangles, 0, 5);
                    enc->setBuffer(sceneMesh.uvBuffer, 0, 6);
                    enc->setBuffer(meshletData.materialIDs, 0, 7);
                    enc->setBuffer(materials.materialBuffer, 0, 8);
                    enc->setBuffer(instanceTransformBuffer, 0, 9);
                    enc->setTexture(fg.getTexture(data.visibility), 0);
                    enc->setTexture(fg.getTexture(data.depth), 1);
                    enc->setTexture(fg.getTexture(data.output), 2);
                    if (!materials.textures.empty()) {
                        enc->setTextures(
                            const_cast<MTL::Texture* const*>(materials.textures.data()),
                            NS::Range(3, materials.textures.size()));
                    }
                    enc->setSamplerState(materials.sampler, 0);
                    MTL::Size tgSize(8, 8, 1);
                    MTL::Size grid((width + 7) / 8, (height + 7) / 8, 1);
                    enc->dispatchThreadgroups(grid, tgSize);
                });

            struct BlitPassData {
                FGResource output;
                FGResource drawable;
            };
            fg.addBlitPass<BlitPassData>("Blit to Drawable",
                [&](FGBuilder& builder, BlitPassData& data) {
                    data.output = builder.read(computeData.output);
                    data.drawable = builder.write(drawableRes);
                    builder.setSideEffect();
                },
                [&, w = width, h = height](const BlitPassData& data, MTL::BlitCommandEncoder* enc) {
                    enc->copyFromTexture(fg.getTexture(data.output), 0, 0,
                        MTL::Origin(0, 0, 0), MTL::Size(w, h, 1),
                        fg.getTexture(data.drawable), 0, 0, MTL::Origin(0, 0, 0));
                });

            struct ImGuiOverlayData {};
            fg.addRenderPass<ImGuiOverlayData>("ImGui Overlay",
                [&](FGBuilder& builder, ImGuiOverlayData&) {
                    builder.setColorAttachment(0, drawableRes,
                        MTL::LoadActionLoad, MTL::StoreActionStore);
                    builder.setSideEffect();
                },
                [&](const ImGuiOverlayData&, MTL::RenderCommandEncoder* enc) {
                    imguiRenderDrawData(commandBuffer, enc);
                });
        } else {
            // === FORWARD RENDERING (Vertex or Mesh shader) ===
            ZoneScopedN("Forward Graph Build");

            struct ForwardPassData {
                FGResource depth;
            };
            fg.addRenderPass<ForwardPassData>("Forward Pass",
                [&](FGBuilder& builder, ForwardPassData& data) {
                    data.depth = builder.create("depth",
                        FGTextureDesc::depthTarget(width, height));
                    builder.setColorAttachment(0, drawableRes,
                        MTL::LoadActionClear, MTL::StoreActionStore,
                        MTL::ClearColor(0.1, 0.2, 0.3, 1.0));
                    builder.setDepthAttachment(data.depth,
                        MTL::LoadActionClear, MTL::StoreActionDontCare, 1.0);
                    builder.setSideEffect();
                },
                [&](const ForwardPassData&, MTL::RenderCommandEncoder* enc) {
                    enc->setDepthStencilState(depthState);
                    enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
                    enc->setCullMode(MTL::CullModeBack);

                    if (renderMode == 1) {
                        enc->setRenderPipelineState(meshPipelineState);
                        // Bind shared buffers once
                        enc->setMeshBuffer(sceneMesh.positionBuffer, 0, 1);
                        enc->setMeshBuffer(sceneMesh.normalBuffer, 0, 2);
                        enc->setMeshBuffer(meshletData.meshletBuffer, 0, 3);
                        enc->setMeshBuffer(meshletData.meshletVertices, 0, 4);
                        enc->setMeshBuffer(meshletData.meshletTriangles, 0, 5);
                        enc->setMeshBuffer(meshletData.boundsBuffer, 0, 6);
                        enc->setMeshBuffer(sceneMesh.uvBuffer, 0, 7);
                        enc->setMeshBuffer(meshletData.materialIDs, 0, 8);
                        enc->setMeshBuffer(materials.materialBuffer, 0, 9);
                        enc->setFragmentBuffer(sceneMesh.positionBuffer, 0, 1);
                        enc->setFragmentBuffer(sceneMesh.normalBuffer, 0, 2);
                        enc->setFragmentBuffer(meshletData.meshletBuffer, 0, 3);
                        enc->setFragmentBuffer(meshletData.meshletVertices, 0, 4);
                        enc->setFragmentBuffer(meshletData.meshletTriangles, 0, 5);
                        enc->setFragmentBuffer(meshletData.boundsBuffer, 0, 6);
                        enc->setFragmentBuffer(sceneMesh.uvBuffer, 0, 7);
                        enc->setFragmentBuffer(meshletData.materialIDs, 0, 8);
                        enc->setFragmentBuffer(materials.materialBuffer, 0, 9);
                        if (!materials.textures.empty()) {
                            enc->setFragmentTextures(
                                const_cast<MTL::Texture* const*>(materials.textures.data()),
                                NS::Range(0, materials.textures.size()));
                            enc->setMeshTextures(
                                const_cast<MTL::Texture* const*>(materials.textures.data()),
                                NS::Range(0, materials.textures.size()));
                        }
                        enc->setFragmentSamplerState(materials.sampler, 0);
                        enc->setMeshSamplerState(materials.sampler, 0);

                        // Per-node dispatch
                        for (uint32_t nodeID : visibleMeshletNodes) {
                            const auto& node = sceneGraph.nodes[nodeID];
                            float4x4 nodeModelView = view * node.transform.worldMatrix;
                            float4x4 nodeMVP = proj * nodeModelView;
                            Uniforms nodeUniforms = uniforms;
                            nodeUniforms.mvp = transpose(nodeMVP);
                            nodeUniforms.modelView = transpose(nodeModelView);
                            extractFrustumPlanes(nodeMVP, nodeUniforms.frustumPlanes);
                            float4x4 invModel = node.transform.worldMatrix;
                            invModel.Invert();
                            nodeUniforms.cameraPos = invModel * cameraWorldPos;
                            nodeUniforms.meshletBaseOffset = node.meshletStart;
                            nodeUniforms.instanceID = 0;
                            enc->setMeshBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                            enc->setFragmentBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                            enc->drawMeshThreadgroups(
                                MTL::Size(node.meshletCount, 1, 1),
                                MTL::Size(1, 1, 1),
                                MTL::Size(128, 1, 1));
                        }
                    } else {
                        enc->setRenderPipelineState(pipelineState);
                        enc->setVertexBuffer(sceneMesh.positionBuffer, 0, 1);
                        enc->setVertexBuffer(sceneMesh.normalBuffer, 0, 2);

                        // Per-node dispatch for vertex pipeline
                        for (uint32_t nodeID : visibleIndexNodes) {
                            const auto& node = sceneGraph.nodes[nodeID];
                            float4x4 nodeModelView = view * node.transform.worldMatrix;
                            float4x4 nodeMVP = proj * nodeModelView;
                            Uniforms nodeUniforms = uniforms;
                            nodeUniforms.mvp = transpose(nodeMVP);
                            nodeUniforms.modelView = transpose(nodeModelView);
                            float4x4 invModel = node.transform.worldMatrix;
                            invModel.Invert();
                            nodeUniforms.cameraPos = invModel * cameraWorldPos;
                            enc->setVertexBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                            enc->setFragmentBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                            enc->drawIndexedPrimitives(
                                MTL::PrimitiveTypeTriangle,
                                node.indexCount, MTL::IndexTypeUInt32,
                                sceneMesh.indexBuffer, node.indexStart * sizeof(uint32_t));
                        }
                    }

                    imguiRenderDrawData(commandBuffer, enc);
                });
        }

        fg.compile();

        if (showGraphDebug)
            fg.debugImGui();

        ImGui::Render();

        bool gKeyDown = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
        if (gKeyDown && !exportGraphKeyDown) {
            std::ofstream dot("framegraph.dot");
            if (dot.is_open()) {
                fg.exportGraphviz(dot);
                std::cout << "Exported framegraph.dot" << std::endl;
            }
        }
        exportGraphKeyDown = gKeyDown;

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
