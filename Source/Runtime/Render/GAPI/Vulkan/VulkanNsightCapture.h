#pragma once
#include <cstdint>
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
#include <NGFX_GraphicsCapture_Common.h>
#if __has_include(<NGFX_GPUTrace_Common.h>)
#include <NGFX_GPUTrace_Common.h>
#define METALLIC_HAS_NSIGHT_GPU_TRACE 1
#endif
#endif
namespace metallic::render { class Queue; class Texture; }
namespace metallic::render::vulkan {
bool nsightInjectionActive();
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result initializeNsightTrace();
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result startNsightTrace();
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result stopNsightTrace(Queue* queue);
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result injectNsightTrace(const NGFX_PathChar* path, NGFX_GPUTrace_InjectionSettings settings);
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result injectNsightCapture(const NGFX_PathChar* path, NGFX_GraphicsCapture_InjectionSettings settings);
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result initializeNsightCapture();
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result activateNsightTrace(Queue& queue);
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result requestNsightCapture(bool explicitBoundaries, uint32_t framesBeforeStart, uint32_t framesToCapture);
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result nsightFrameBoundary(Queue& queue, Texture* output);
#endif
} // namespace metallic::render::vulkan
