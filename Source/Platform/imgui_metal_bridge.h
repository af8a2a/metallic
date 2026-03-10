#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void imguiInit(void* mtlDevice);
void imguiNewFrame(void* mtlRenderPassDescriptor);
void imguiNewFrameForTargets(void* colorTextureHandle, void* depthTextureHandle);
void imguiRenderDrawData(void* mtlCommandBuffer, void* mtlRenderCommandEncoder);
void imguiShutdown(void);

#ifdef __cplusplus
}
#endif
