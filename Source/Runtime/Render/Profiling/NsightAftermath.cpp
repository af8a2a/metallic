#include "NsightAftermath.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanAftermath.h"

namespace metallic::render::profiling {
bool nsightAftermathSdkAvailable() { return vulkan::nsightAftermathSdkAvailable(); }
bool nsightAftermathInitialized() { return vulkan::nsightAftermathInitialized(); }
void initializeNsightAftermath(const char* name) { vulkan::initializeNsightAftermath(name); }
void registerNsightAftermathShaderBinary(const uint32_t* code, uint64_t size) { vulkan::registerNsightAftermathShaderBinary(code, size); }
void handleNsightAftermathDeviceLost() { vulkan::handleNsightAftermathDeviceLost(); }
} // namespace metallic::render::profiling
