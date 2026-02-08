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
#include "camera.h"
#include "input.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include <slang.h>
#include <slang-com-ptr.h>

#include <iostream>
#include <string>
#include <vector>
#include <regex>

struct Uniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir;
    float4   frustumPlanes[6];
    float4   cameraPos;
    uint32_t enableFrustumCull;
    uint32_t enableConeCull;
    uint32_t pad0;
    uint32_t pad1;
};

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

static std::string patchMeshShaderMetalSource(const std::string& source) {
    std::cout << "=== Raw Slang-generated Metal source (mesh shader) ===" << std::endl;
    std::cout << source << std::endl;
    std::cout << "=== End of raw Metal source ===" << std::endl;

    // Slang v2026.1.2 bug: mesh output struct members lack [[user(...)]] attributes
    // that the fragment shader's [[stage_in]] expects. We patch them in.
    std::string patched = source;

    // Pattern: find fields like "float3 viewNormal_0;" and add [[user(NORMAL)]]
    // and "float3 viewPos_0;" and add [[user(TEXCOORD0)]]
    // The exact field names depend on Slang's mangling, so we use regex.
    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewNormal\w*)\s*;)"),
        "$1 [[user(NORMAL)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewPos\w*)\s*;)"),
        "$1 [[user(TEXCOORD)]];");

    return patched;
}

static MTL::Texture* createDepthTexture(MTL::Device* device, int width, int height) {
    auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, width, height, false);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setUsage(MTL::TextureUsageRenderTarget);
    return device->newTexture(desc);
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "RenderGraph - Stanford Bunny", nullptr, nullptr);
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

    CA::MetalLayer* metalLayer = static_cast<CA::MetalLayer*>(
        attachMetalLayerToGLFWWindow(window));
    metalLayer->setDevice(device);
    metalLayer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    // Load bunny mesh
    LoadedMesh bunny;
    if (!loadGLTFMesh(device, "Asset/StandfordBunny/scene.gltf", bunny)) {
        std::cerr << "Failed to load bunny mesh" << std::endl;
        return 1;
    }
    std::cout << "Loaded bunny: " << bunny.vertexCount << " vertices, "
              << bunny.indexCount << " indices" << std::endl;

    // Build meshlets for future mesh shader rendering
    MeshletData meshletData;
    if (!buildMeshlets(device, bunny, meshletData)) {
        std::cerr << "Failed to build meshlets" << std::endl;
        return 1;
    }

    // Init orbit camera
    OrbitCamera camera;
    camera.initForBunny(bunny.bboxMin, bunny.bboxMax);

    // Setup input
    InputState inputState;
    inputState.camera = &camera;
    setupInputCallbacks(window, &inputState);

    // Init Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
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

    bool useMeshShader = false;
    bool enableFrustumCull = false;
    bool enableConeCull = false;

    // Create depth stencil state
    MTL::DepthStencilDescriptor* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(true);
    MTL::DepthStencilState* depthState = device->newDepthStencilState(depthDesc);
    depthDesc->release();

    // Create initial depth texture
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    MTL::Texture* depthTexture = createDepthTexture(device, fbWidth, fbHeight);
    int prevWidth = fbWidth, prevHeight = fbHeight;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            pool->release();
            continue;
        }
        metalLayer->setDrawableSize(CGSizeMake(width, height));

        // Recreate depth texture on resize
        if (width != prevWidth || height != prevHeight) {
            depthTexture->release();
            depthTexture = createDepthTexture(device, width, height);
            prevWidth = width;
            prevHeight = height;
        }

        CA::MetalDrawable* drawable = metalLayer->nextDrawable();
        if (!drawable) {
            pool->release();
            continue;
        }

        // Compute matrices
        float aspect = (float)width / (float)height;
        float4x4 view = camera.viewMatrix();
        float4x4 proj = camera.projectionMatrix(aspect);
        float4x4 model = float4x4::Identity();
        float4x4 modelView = view * model;
        float4x4 mvp = proj * modelView;

        // Light direction in view space (from upper-right-front)
        float4 worldLightDir = float4(normalize(float3(0.5f, 1.0f, 0.8f)), 0.0f);
        float4 viewLightDir = view * worldLightDir;

        Uniforms uniforms;
        uniforms.mvp = transpose(mvp);
        uniforms.modelView = transpose(modelView);
        uniforms.lightDir = viewLightDir;

        // Extract frustum planes from non-transposed MVP (world-space planes)
        extractFrustumPlanes(mvp, uniforms.frustumPlanes);

        // Camera position in world space
        float cosA = std::cos(camera.azimuth), sinA = std::sin(camera.azimuth);
        float cosE = std::cos(camera.elevation), sinE = std::sin(camera.elevation);
        uniforms.cameraPos = float4(
            camera.target.x + camera.distance * cosE * sinA,
            camera.target.y + camera.distance * sinE,
            camera.target.z + camera.distance * cosE * cosA,
            0.0f);

        uniforms.enableFrustumCull = enableFrustumCull ? 1 : 0;
        uniforms.enableConeCull = enableConeCull ? 1 : 0;
        uniforms.pad0 = 0;
        uniforms.pad1 = 0;

        // Render pass
        MTL::RenderPassDescriptor* renderPass = MTL::RenderPassDescriptor::alloc()->init();
        auto* colorAttachment = renderPass->colorAttachments()->object(0);
        colorAttachment->setTexture(drawable->texture());
        colorAttachment->setLoadAction(MTL::LoadActionClear);
        colorAttachment->setClearColor(MTL::ClearColor(0.1, 0.2, 0.3, 1.0));
        colorAttachment->setStoreAction(MTL::StoreActionStore);

        auto* depthAttachment = renderPass->depthAttachment();
        depthAttachment->setTexture(depthTexture);
        depthAttachment->setLoadAction(MTL::LoadActionClear);
        depthAttachment->setClearDepth(1.0);
        depthAttachment->setStoreAction(MTL::StoreActionDontCare);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::RenderCommandEncoder* encoder =
            commandBuffer->renderCommandEncoder(renderPass);

        // ImGui new frame
        imguiNewFrame(renderPass);
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin("##fps", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
        ImGui::Text("%.1f FPS (%.3f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::Separator();
        int renderMode = useMeshShader ? 1 : 0;
        ImGui::RadioButton("Vertex Shader", &renderMode, 0);
        ImGui::RadioButton("Mesh Shader", &renderMode, 1);
        useMeshShader = (renderMode == 1);
        if (useMeshShader) {
            ImGui::Text("Meshlets: %u", meshletData.meshletCount);
            ImGui::Checkbox("Frustum Culling", &enableFrustumCull);
            ImGui::Checkbox("Backface Culling", &enableConeCull);
        }
        ImGui::End();
        ImGui::Render();

        encoder->setDepthStencilState(depthState);
        encoder->setFrontFacingWinding(MTL::WindingCounterClockwise);
        encoder->setCullMode(MTL::CullModeBack);

        if (useMeshShader) {
            encoder->setRenderPipelineState(meshPipelineState);
            encoder->setMeshBytes(&uniforms, sizeof(uniforms), 0);
            encoder->setMeshBuffer(bunny.positionBuffer, 0, 1);
            encoder->setMeshBuffer(bunny.normalBuffer, 0, 2);
            encoder->setMeshBuffer(meshletData.meshletBuffer, 0, 3);
            encoder->setMeshBuffer(meshletData.meshletVertices, 0, 4);
            encoder->setMeshBuffer(meshletData.meshletTriangles, 0, 5);
            encoder->setMeshBuffer(meshletData.boundsBuffer, 0, 6);
            encoder->setFragmentBytes(&uniforms, sizeof(uniforms), 0);
            encoder->drawMeshThreadgroups(
                MTL::Size(meshletData.meshletCount, 1, 1),
                MTL::Size(1, 1, 1),
                MTL::Size(128, 1, 1));
        } else {
            encoder->setRenderPipelineState(pipelineState);
            encoder->setVertexBuffer(bunny.positionBuffer, 0, 1);
            encoder->setVertexBuffer(bunny.normalBuffer, 0, 2);
            encoder->setVertexBytes(&uniforms, sizeof(uniforms), 0);
            encoder->setFragmentBytes(&uniforms, sizeof(uniforms), 0);
            encoder->drawIndexedPrimitives(
                MTL::PrimitiveTypeTriangle,
                bunny.indexCount, MTL::IndexTypeUInt32,
                bunny.indexBuffer, 0);
        }

        // Render ImGui on top
        imguiRenderDrawData(commandBuffer, encoder);

        encoder->endEncoding();
        commandBuffer->presentDrawable(drawable);
        commandBuffer->commit();

        renderPass->release();
        pool->release();
    }

    // Cleanup ImGui
    imguiShutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup
    depthTexture->release();
    meshletData.meshletBuffer->release();
    meshletData.meshletVertices->release();
    meshletData.meshletTriangles->release();
    meshletData.boundsBuffer->release();
    bunny.positionBuffer->release();
    bunny.normalBuffer->release();
    bunny.indexBuffer->release();
    depthState->release();
    meshPipelineState->release();
    pipelineState->release();
    commandQueue->release();
    device->release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
