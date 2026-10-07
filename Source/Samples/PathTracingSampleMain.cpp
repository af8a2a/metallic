#include "Runtime/Render/Profiling/RHIProfiling.h"
#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Runtime/Render/RenderSample.h"
#include "Editor/EditorApplication.h"
#include "Editor/NsightLaunchOptions.h"

#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace {

constexpr const char* kPathTracingSampleId = "pathtracing-sample";
constexpr const char* kPathTracingDLSSSRSampleId = "pathtracing-sample-dlss-sr";

void printUsage()
{
    std::puts(metallic::render::ShaderWarmupLaunchOptions::kUsage);
    std::puts(metallic::NsightLaunchOptions::kUsage);
    spdlog::info(
        "MetallicPathTracingSample options:\n"
        "  Default: DLSS-RR with DLSS Super Resolution Quality\n"
        "  --native                    Use native-resolution path tracing without DLSS\n"
        "  --dlss-sr                   Use the NVIDIA DLSS-SR upscaling graph\n"
        "  --dlss-rr                   Use DLSS-RR with Super Resolution Quality (default)\n"
        "  --smoke-test                 Render one frame and exit\n"
        "  --scene <path>               Override the sample glTF scene\n"
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
    const char* sampleId = metallic::render::kDefaultPathTracingSampleId;
    std::string scenePath;
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
        if (argument == "--dlss-rr") {
            sampleId = metallic::render::kDefaultPathTracingSampleId;
            continue;
        }
        if (argument == "--native") {
            sampleId = kPathTracingSampleId;
            continue;
        }
        if (argument == "--dlss-sr") {
            sampleId = kPathTracingDLSSSRSampleId;
            continue;
        }
        if (argument == "--wait-for-graphics-debugger") {
            waitForGraphicsDebugger = true;
            continue;
        }
        if (argument == "--scene" && index + 1 < argc) {
            scenePath = argv[++index];
            continue;
        }

        spdlog::error("Unknown argument: {}", argument);
        printUsage();
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(
        smokeTest,
        waitForGraphicsDebugger,
        sampleId,
        scenePath.empty() ? nullptr : scenePath.c_str(),
        nullptr,
        nsightOptions.mode, false, false, false, warmupOptions.skip);
}
