#include "VulkanNsightCapture.h"
#include "VulkanNative.h"
#include <atomic>
#ifdef _WIN32
#include <Windows.h>
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
#include <NGFX_GraphicsCapture_Vulkan.h>
#include <NGFX_Vulkan.h>
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
#include <NGFX_GPUTrace_Vulkan.h>
#endif
#endif

namespace metallic::render::vulkan {
namespace { std::atomic_bool captureInjected{false}; }
bool nsightInjectionActive()
{
    if (captureInjected.load(std::memory_order_relaxed)) { return true; }
#ifdef _WIN32
    return GetModuleHandleW(L"ngfx-capture-interception.dll") != nullptr;
#else
    return false;
#endif
}
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result initializeNsightTrace()
{
    NGFX_GPUTrace_InitializeActivity_Vulkan_Params initialize{};
    initialize.version = NGFX_GPUTrace_InitializeActivity_Vulkan_Params_VER;
    return NGFX_GPUTrace_InitializeActivity_Vulkan(&initialize);
}
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result startNsightTrace()
{
    NGFX_GPUTrace_StartTrace_Vulkan_Params start{};
    start.version = NGFX_GPUTrace_StartTrace_Vulkan_Params_VER;
    return NGFX_GPUTrace_StartTrace_Vulkan(&start);
}
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result stopNsightTrace(Queue* queue)
{
    NGFX_GPUTrace_StopTrace_Vulkan_Params stop{};
    stop.version = NGFX_GPUTrace_StopTrace_Vulkan_Params_VER;
    stop.queue = queue ? nativeQueue(*queue).queue : VK_NULL_HANDLE;
    return NGFX_GPUTrace_StopTrace_Vulkan(&stop);
}
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result injectNsightTrace(const NGFX_PathChar* path, NGFX_GPUTrace_InjectionSettings settings)
{
    NGFX_GPUTrace_Inject_Vulkan_Params inject{};
    inject.version = NGFX_GPUTrace_Inject_Vulkan_Params_VER;
    inject.installationPath = path;
    inject.settings = &settings;
    return NGFX_GPUTrace_Inject_Vulkan(&inject);
}
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result injectNsightCapture(const NGFX_PathChar* path, NGFX_GraphicsCapture_InjectionSettings settings)
{
    NGFX_GraphicsCapture_Inject_Vulkan_Params injectParams{};
    injectParams.version = NGFX_GraphicsCapture_Inject_Vulkan_Params_VER;
    injectParams.installationPath = path;
    injectParams.settings = &settings;
    const auto result = NGFX_GraphicsCapture_Inject_Vulkan(&injectParams);
    // Injection outlives activity initialization, including failures.
    if (result == NGFX_Result_Success) { captureInjected.store(true, std::memory_order_relaxed); }
    return result;
}
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result initializeNsightCapture()
{
    NGFX_GraphicsCapture_InitializeActivity_Vulkan_Params initializeParams{};
    initializeParams.version = NGFX_GraphicsCapture_InitializeActivity_Vulkan_Params_VER;
    return NGFX_GraphicsCapture_InitializeActivity_Vulkan(&initializeParams);
}
#endif
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
NGFX_Result activateNsightTrace(Queue& queue)
{
    NGFX_GPUTrace_ActivateTrace_Vulkan_Params activate{};
    activate.version = NGFX_GPUTrace_ActivateTrace_Vulkan_Params_VER;
    activate.queue = nativeQueue(queue).queue;
    return NGFX_GPUTrace_ActivateTrace_Vulkan(&activate);
}
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result requestNsightCapture(bool explicitBoundaries, uint32_t framesBeforeStart, uint32_t framesToCapture)
{
    NGFX_GraphicsCapture_RequestCapture_Vulkan_Params captureParams{};
    captureParams.version = NGFX_GraphicsCapture_RequestCapture_Vulkan_Params_VER;
    captureParams.delimiter = explicitBoundaries
        ? NGFX_GraphicsCapture_Delimiter_FrameBoundary : NGFX_GraphicsCapture_Delimiter_Present;
    captureParams.framesBeforeStart = framesBeforeStart;
    captureParams.framesToCapture = framesToCapture;
    return NGFX_GraphicsCapture_RequestCapture_Vulkan(&captureParams);
}
#endif
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
NGFX_Result nsightFrameBoundary(Queue& queue, Texture* output)
{
    NGFX_FrameBoundary_Vulkan_Params params{};
    params.version = NGFX_FrameBoundary_Vulkan_Params_VER;
    params.queue = nativeQueue(queue).queue;
    NGFX_ResourceDescription_Vulkan resource{};
    if (output != nullptr) {
        resource.version = NGFX_ResourceDescription_Vulkan_VER;
        resource.type = NGFX_ResourceType_Vulkan_VkImage;
        resource.image = nativeTexture(*output).image;
        params.outputResources = &resource;
        params.numOutputResources = 1;
    }
    return NGFX_FrameBoundary_Vulkan(&params);
}
#endif
} // namespace metallic::render::vulkan
