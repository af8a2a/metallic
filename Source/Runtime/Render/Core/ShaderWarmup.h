#pragma once

#include <string_view>

namespace metallic::render {

// Shared by the standalone compiler and application startup. Startup always
// warms the complete catalog in the current Slang shader debug/descriptor mode.
int runShaderWarmup(int argc, char** argv);
int warmupShadersForStartup(bool skip);
// Arm device-first source/native warmup without compiling or querying an adapter.
// The actual rendering device performs both stages before createDevice returns.
int prepareStartupShaderWarmup(bool skip);
// Explicit application preparation or successful legacy source warmup enables
// the device phase. Standalone/offline compilation does not enable this gate.
bool startupShaderWarmupEnabled() noexcept;

struct ShaderWarmupLaunchOptions {
    bool skip = false;

    static constexpr const char* kUsage =
        "  --skip-shader-warmup         Skip startup SPIR-V and native pipeline warmup (compile on demand)";

    bool consume(const char* argument)
    {
        if (std::string_view(argument) != "--skip-shader-warmup") {
            return false;
        }
        skip = true;
        return true;
    }
};

} // namespace metallic::render
