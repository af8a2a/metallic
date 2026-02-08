#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "glfw_metal_bridge.h"

#include <slang.h>
#include <slang-com-ptr.h>

#include <iostream>
#include <string>
#include <vector>
#include <cstddef>

struct Vertex {
    float position[3];
    float color[3];
};

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

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "RenderGraph", nullptr, nullptr);
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

    // Compile Slang shader to Metal source
    std::string metalSource = compileSlangToMetal("Shaders/triangle");
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

    // Create render pipeline state
    MTL::RenderPipelineDescriptor* pipelineDesc =
        MTL::RenderPipelineDescriptor::alloc()->init();
    pipelineDesc->setVertexFunction(vertexFn);
    pipelineDesc->setFragmentFunction(fragmentFn);
    pipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    // Vertex descriptor matching Slang's attribute layout
    MTL::VertexDescriptor* vertexDesc = MTL::VertexDescriptor::alloc()->init();
    // attribute(0) = position: float3 at offset 0
    vertexDesc->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(0)->setOffset(0);
    vertexDesc->attributes()->object(0)->setBufferIndex(0);
    // attribute(1) = color: float3 at offset 12
    vertexDesc->attributes()->object(1)->setFormat(MTL::VertexFormatFloat3);
    vertexDesc->attributes()->object(1)->setOffset(offsetof(Vertex, color));
    vertexDesc->attributes()->object(1)->setBufferIndex(0);
    // layout for buffer 0
    vertexDesc->layouts()->object(0)->setStride(sizeof(Vertex));
    vertexDesc->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);
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

    // Create vertex buffer with triangle data
    const Vertex triangleVertices[] = {
        {{ 0.0f,  0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},  // top - red
        {{-0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},  // bottom left - green
        {{ 0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},  // bottom right - blue
    };
    MTL::Buffer* vertexBuffer = device->newBuffer(
        triangleVertices, sizeof(triangleVertices), MTL::ResourceStorageModeShared);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        metalLayer->setDrawableSize(CGSizeMake(width, height));

        CA::MetalDrawable* drawable = metalLayer->nextDrawable();
        if (!drawable) {
            pool->release();
            continue;
        }

        MTL::RenderPassDescriptor* renderPass = MTL::RenderPassDescriptor::alloc()->init();
        auto* colorAttachment = renderPass->colorAttachments()->object(0);
        colorAttachment->setTexture(drawable->texture());
        colorAttachment->setLoadAction(MTL::LoadActionClear);
        colorAttachment->setClearColor(MTL::ClearColor(0.1, 0.2, 0.3, 1.0));
        colorAttachment->setStoreAction(MTL::StoreActionStore);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::RenderCommandEncoder* encoder =
            commandBuffer->renderCommandEncoder(renderPass);

        encoder->setRenderPipelineState(pipelineState);
        encoder->setVertexBuffer(vertexBuffer, 0, 0);
        encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));

        encoder->endEncoding();
        commandBuffer->presentDrawable(drawable);
        commandBuffer->commit();

        renderPass->release();
        pool->release();
    }

    vertexBuffer->release();
    pipelineState->release();
    commandQueue->release();
    device->release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}