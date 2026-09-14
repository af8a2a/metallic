#include "Editor/EditorApplication.h"
#include "Runtime/Render/RenderSample.h"

#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace {

void printUsage()
{
    spdlog::info(
        "MetallicGPUDrivenSample options:\n"
        "  Default: streamed MiniZorah with realtime deferred lighting, auto exposure and DLSS-SR\n"
        "  --smoke-test                 Render one frame and exit\n"
        "  --debug-control              Enable local Agent debug control\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization\n"
        "  --scene <source>             Override streaming metadata scene (requires matching cook)\n"
        "  --streamasset-path <file>    Override the cooked StreamAsset path\n"
        "  --sample <id>                Explicitly select a diagnostic or another sample");
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    bool debugControl = false;
    const char* sampleId = metallic::render::kDefaultGPUDrivenSampleId;
    std::string scenePath;
    std::string streamAssetPath;
    for (int index = 1; index < argc; ++index) {
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
        if (argument == "--sample") {
            if (index + 1 >= argc) { spdlog::error("--sample requires an id"); return 1; }
            sampleId = argv[++index];
            continue;
        }
        if (argument == "--minizorah" || argument == "--minizorah-vbuffer" || argument == "--streamasset") {
            sampleId = metallic::render::kDefaultGPUDrivenSampleId;
            continue;
        }
        if (argument == "--debug-control") {
            debugControl = true;
            continue;
        }
        if (argument == "--scene" || argument == "--streamasset-path") {
            if (index + 1 >= argc) {
                spdlog::error("{} requires a path", argument);
                return 1;
            }
            std::string& path = argument == "--scene" ? scenePath : streamAssetPath;
            path = argv[++index];
            continue;
        }

        spdlog::error("Unknown argument: {}", argument);
        printUsage();
        return 1;
    }

    if (!scenePath.empty() && streamAssetPath.empty() &&
        std::string_view(sampleId) == metallic::render::kDefaultGPUDrivenSampleId) {
        spdlog::error("--scene requires --streamasset-path with a matching pre-cooked stream asset");
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(
        smokeTest,
        waitForGraphicsDebugger,
        sampleId,
        scenePath.empty() ? nullptr : scenePath.c_str(),
        streamAssetPath.empty() ? nullptr : streamAssetPath.c_str(),
        false, false, debugControl);
}
