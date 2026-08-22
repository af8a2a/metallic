#include "Editor/EditorApplication.h"

#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace {

constexpr const char* kMaterialVisualizationSampleId = "material-visualization-abeautiful-game";

void printUsage()
{
    spdlog::info(
        "MetallicMaterialVisualizationSample options:\n"
        "  --smoke-test                 Render one frame and exit\n"
        "  --scene <path>               Override the sample glTF scene\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization");
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    std::string scenePath;
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
        kMaterialVisualizationSampleId,
        scenePath.empty() ? nullptr : scenePath.c_str());
}
