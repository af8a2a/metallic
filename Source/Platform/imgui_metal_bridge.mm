#import <Metal/Metal.h>

#include "imgui.h"
#include "imgui_impl_metal.h"
#include "imgui_metal_bridge.h"

extern "C" void imguiInit(void* mtlDevice) {
    ImGui_ImplMetal_Init((__bridge id<MTLDevice>)mtlDevice);
}

extern "C" void imguiNewFrame(void* mtlRenderPassDescriptor) {
    ImGui_ImplMetal_NewFrame((__bridge MTLRenderPassDescriptor*)mtlRenderPassDescriptor);
}

extern "C" void imguiNewFrameForTargets(void* colorTextureHandle, void* depthTextureHandle) {
    MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
    renderPassDescriptor.colorAttachments[0].texture = (__bridge id<MTLTexture>)colorTextureHandle;
    renderPassDescriptor.depthAttachment.texture = (__bridge id<MTLTexture>)depthTextureHandle;
    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
}

extern "C" void imguiRenderDrawData(void* mtlCommandBuffer, void* mtlRenderCommandEncoder) {
    ImGui_ImplMetal_RenderDrawData(
        ImGui::GetDrawData(),
        (__bridge id<MTLCommandBuffer>)mtlCommandBuffer,
        (__bridge id<MTLRenderCommandEncoder>)mtlRenderCommandEncoder);
}

extern "C" void imguiShutdown(void) {
    ImGui_ImplMetal_Shutdown();
}
