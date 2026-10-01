#pragma once

#include <string_view>

namespace metallic::render {

// Shared by the standalone compiler and application startup. Startup always
// warms the complete catalog in the current Slang shader debug/descriptor mode.
int runShaderWarmup(int argc, char** argv);
int warmupShadersForStartup(bool skip);

struct ShaderWarmupLaunchOptions {
    bool skip = false;

    static constexpr const char* kUsage =
        "  --skip-shader-warmup         Skip startup shader cache warmup (compile on demand)";

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
