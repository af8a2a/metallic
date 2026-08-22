#include "Editor/EditorApplication.h"

#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace {

constexpr const char* kPathTracingSampleId = "pathtracing-sample";
constexpr const char* kPathTracingDlssRrSampleId = "pathtracing-sample-dlss-rr";

void printUsage()
{
    spdlog::info(
        "MetallicPathTracingSample options:\n"
        "  --dlss-rr                   Use the NVIDIA DLSS-RR denoiser graph\n"
        "  --smoke-test                 Render one frame and exit\n"
        "  --scene <path>               Override the sample glTF scene\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization");
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    const char* sampleId = kPathTracingSampleId;
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
        if (argument == "--dlss-rr") {
            sampleId = kPathTracingDlssRrSampleId;
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
        scenePath.empty() ? nullptr : scenePath.c_str());
}
