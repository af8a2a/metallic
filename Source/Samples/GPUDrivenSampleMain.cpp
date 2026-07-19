#include "Editor/EditorApplication.h"

#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace {

constexpr const char* kGPUDrivenSampleId = "gpu-driven-sample";
constexpr const char* kGPUDrivenStreamAssetSampleId = "gpu-driven-streamasset";
constexpr const char* kGPUDrivenRtasVisualizationSampleId = "gpu-driven-rtas-visualization";

void printUsage()
{
    spdlog::info(
        "MetallicGPUDrivenSample options:\n"
        "  --smoke-test                 Render one frame and exit\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization\n"
        "  --visibility-buffer          Load the visibility-buffer variant (default)\n"
        "  --streamasset                Load the default meshlet StreamAsset variant\n"
        "  --legacy-preloaded           Alias for the visibility-buffer variant\n"
        "  --rtas-visualization         Load the RTAS visualization variant\n"
        "  --scene <source.gltf>        Override the sample source scene\n"
        "  --streamasset-path <file>    Override the StreamAsset cache path");
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    const char* sampleId = kGPUDrivenSampleId;
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
        if (argument == "--streamasset") {
            sampleId = kGPUDrivenStreamAssetSampleId;
            continue;
        }
        if (argument == "--visibility-buffer" || argument == "--legacy-preloaded") {
            sampleId = kGPUDrivenSampleId;
            continue;
        }
        if (argument == "--rtas-visualization") {
            sampleId = kGPUDrivenRtasVisualizationSampleId;
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

    if (!streamAssetPath.empty() && std::string_view(sampleId) == kGPUDrivenSampleId) {
        spdlog::error("--streamasset-path cannot be used with the visibility-buffer variant");
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(
        smokeTest,
        waitForGraphicsDebugger,
        sampleId,
        scenePath.empty() ? nullptr : scenePath.c_str(),
        streamAssetPath.empty() ? nullptr : streamAssetPath.c_str());
}
