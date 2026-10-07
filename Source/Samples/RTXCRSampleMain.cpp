#include "Runtime/Render/Profiling/RHIProfiling.h"
#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Editor/EditorApplication.h"
#include "Editor/NsightLaunchOptions.h"

#include <spdlog/spdlog.h>

#include <string_view>

namespace {

constexpr const char* kRTXCRSampleId = "rtxcr-material-sample";

void printUsage()
{
    std::puts(metallic::render::ShaderWarmupLaunchOptions::kUsage);
    std::puts(metallic::NsightLaunchOptions::kUsage);
    spdlog::info(
        "MetallicRTXCRSample options:\n"
        "  --smoke-test                 Render one frame and exit\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization");
}

} // namespace

int main(int argc, char** argv)
{
    metallic::render::profiling::initializeRHIProfiling();
    metallic::NsightLaunchOptions nsightOptions;
    metallic::render::ShaderWarmupLaunchOptions warmupOptions;
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    for (int index = 1; index < argc; ++index) {
        if (warmupOptions.consume(argv[index])) { continue; }
        const int nsightArgument = nsightOptions.consume(argc, argv, index);
        if (nsightArgument < 0) { return 1; }
        if (nsightArgument > 0) { continue; }
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            printUsage();
            return 0;
        }
        if (argument == "--smoke-test") {
            smokeTest = true;
            continue;
        }
        if (argument == "--wait-for-graphics-debugger") {
            waitForGraphicsDebugger = true;
            continue;
        }

        spdlog::error("Unknown argument: {}", argument);
        printUsage();
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(smokeTest, waitForGraphicsDebugger, kRTXCRSampleId, nullptr, nullptr, nsightOptions.mode, false, false, false, warmupOptions.skip);
}
