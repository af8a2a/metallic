#pragma once

#include <cstdint>

namespace metallic::render::profiling {

bool nsightAftermathSdkAvailable();
bool nsightAftermathInitialized();
void initializeNsightAftermath(const char* applicationName);
void registerNsightAftermathShaderBinary(const uint32_t* code, uint64_t byteSize);
void handleNsightAftermathDeviceLost();

} // namespace metallic::render::profiling
