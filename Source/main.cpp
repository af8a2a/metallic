#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "glfw_metal_bridge.h"

#include <iostream>

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
        MTL::RenderPassColorAttachmentDescriptor* colorAttachment =
            renderPass->colorAttachments()->object(0);
        colorAttachment->setTexture(drawable->texture());
        colorAttachment->setLoadAction(MTL::LoadActionClear);
        colorAttachment->setClearColor(MTL::ClearColor(0.1, 0.2, 0.3, 1.0));
        colorAttachment->setStoreAction(MTL::StoreActionStore);

        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
        MTL::RenderCommandEncoder* encoder =
            commandBuffer->renderCommandEncoder(renderPass);
        encoder->endEncoding();

        commandBuffer->presentDrawable(drawable);
        commandBuffer->commit();

        renderPass->release();
        pool->release();
    }

    commandQueue->release();
    device->release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
