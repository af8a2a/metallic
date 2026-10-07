#pragma once
#include <volk.h>
#include <cstdint>
#include <string>
#include <vector>

namespace metallic::render::vulkan {
// Immutable process-lifetime backend callbacks. No application profiling types
// cross the RHI boundary, and disabled SDKs retain their explicit failure paths.
struct ToolingHooks {
    bool (*captureInjected)();
    bool (*requiresPresentDrain)();
    bool (*instanceExtensions)(std::vector<const char*>&, uint32_t apiVersion, bool validation, std::string& error);
    bool (*deviceExtensions)(VkInstance, VkPhysicalDevice, std::vector<const char*>&, std::string& error);
    void (*initializeDiagnostics)(const char* applicationName);
    bool (*diagnosticsInitialized)();
    void (*shaderBinary)(const uint32_t*, uint64_t byteSize);
    void (*deviceLost)();
};
const ToolingHooks& toolingHooks();
} // namespace metallic::render::vulkan
