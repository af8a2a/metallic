#include "Editor/EditorApplication.h"
#include "Runtime/Render/RenderSample.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <string_view>

namespace {

constexpr const char* kDefaultLookDevSampleId = "openpbr-lookdev";

void printUsage()
{
    spdlog::info(
        "LookDev material playground options:\n"
        "  Default: MaterialX OpenPBR shaderball with fixed exposure and sRGB display\n"
        "  --sample <id>               Load a built-in sample (default: openpbr-lookdev)\n"
        "  --list-samples              List available sample IDs and exit\n"
        "  --scene <path>              Override the selected sample's scene\n"
        "  --smoke-test                Render one frame and exit\n"
        "  --debug-control             Enable local Agent debug control\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization");
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    bool debugControl = false;
    bool listSamples = false;
    std::string sampleId = kDefaultLookDevSampleId;
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
        if (argument == "--debug-control") {
            debugControl = true;
            continue;
        }
        if (argument == "--list-samples") {
            listSamples = true;
            continue;
        }
        if (argument == "--sample" || argument == "--scene") {
            if (index + 1 >= argc || std::string_view(argv[index + 1]).empty() ||
                std::string_view(argv[index + 1]).starts_with("--")) {
                spdlog::error("{} requires a value", argument);
                return 1;
            }
            (argument == "--sample" ? sampleId : scenePath) = argv[++index];
            continue;
        }
        spdlog::error("Unknown argument: {}", argument);
        printUsage();
        return 1;
    }

    const auto samples = metallic::render::listBuiltInRenderSamples();
    if (listSamples) {
        for (const auto& sample : samples) {
            spdlog::info("{} [{}] {}", sample.id, sample.category, sample.name);
        }
        return 0;
    }
    if (std::none_of(samples.begin(), samples.end(), [&](const auto& sample) { return sample.id == sampleId; })) {
        spdlog::error("Unknown sample '{}'. Use --list-samples to see available IDs.", sampleId);
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(
        smokeTest,
        waitForGraphicsDebugger,
        sampleId.c_str(),
        scenePath.empty() ? nullptr : scenePath.c_str(),
        nullptr,
        false,
        false,
        debugControl);
}
