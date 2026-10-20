#pragma once

#include <string_view>

namespace metallic::render {

// Explicit full-catalog warmup, shared by the standalone compiler and opt-in
// application startup. Default startup compiles the active graph/UI requests
// through their runtime owners, including exact native specialization values.
int runShaderWarmup(int argc, char** argv);
int warmupShadersForStartup(bool skip);
// Arm opt-in full-catalog source/native warmup without querying an adapter.
// Skipping leaves compilation to runtime graph/UI initialization. Otherwise the
// actual rendering device performs both stages before createDevice returns.
int prepareStartupShaderWarmup(bool skip);
// Explicit application preparation or successful legacy source warmup enables
// the device phase. Standalone/offline compilation does not enable this gate.
bool startupShaderWarmupEnabled() noexcept;

struct ShaderWarmupLaunchOptions {
    bool skip = true;

    static constexpr const char* kUsage =
        "  Default: compile shader/PSO variants requested by the active graph and UI\n"
        "  --full-shader-warmup         Warm the complete shader/compute PSO catalog before initialization\n"
        "  --skip-shader-warmup         Compile on demand (default; overrides an earlier --full-shader-warmup)";

    bool consume(const char* argument)
    {
        const std::string_view value(argument);
        if (value == "--full-shader-warmup") {
            skip = false;
            return true;
        }
        if (value == "--skip-shader-warmup") {
            skip = true;
            return true;
        }
        return false;
    }
};

} // namespace metallic::render
