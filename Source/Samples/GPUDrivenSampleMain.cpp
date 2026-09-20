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
        "  --minizorah                 Load MiniZorah (default)\n"
        "  --zorah-full                 Load the full textured Zorah scene\n"
        "  --list-scenes               List the two supported scenes and exit\n"
        "  --streamasset-path <file>    Use a matching cook for the selected startup scene\n"
        "  --sample <id>                Compatibility: gpu-driven-sample or gpu-driven-zorah-full");
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    bool debugControl = false;
    const char* sampleId = metallic::render::kDefaultGPUDrivenSampleId;
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
        if (argument == "--list-scenes") {
            for (const auto& scene : metallic::render::listGPUDrivenSceneSamples()) {
                spdlog::info("{} ({}) - {}", scene.name, scene.id, scene.scenePath);
            }
            return 0;
        }
        if (argument == "--minizorah") {
            sampleId = metallic::render::kDefaultGPUDrivenSampleId;
            continue;
        }
        if (argument == "--zorah-full") {
            sampleId = metallic::render::kGPUDrivenZorahFullSampleId;
            continue;
        }
        if (argument == "--debug-control") {
            debugControl = true;
            continue;
        }
        if (argument == "--streamasset-path") {
            if (index + 1 >= argc) {
                spdlog::error("{} requires a path", argument);
                return 1;
            }
            streamAssetPath = argv[++index];
            continue;
        }

        spdlog::error("Unknown argument: {}", argument);
        printUsage();
        return 1;
    }

    if (!metallic::render::isGPUDrivenSceneSample(sampleId)) {
        spdlog::error("GPUDrivenSample supports only MiniZorah and ZorahFull; use --list-scenes");
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(
        smokeTest,
        waitForGraphicsDebugger,
        sampleId,
        nullptr,
        streamAssetPath.empty() ? nullptr : streamAssetPath.c_str(),
        false, false, debugControl, true);
}
