#include "Editor/EditorApplication.h"

#include <iostream>
#include <string_view>

namespace {

constexpr const char* kPathTracingSampleId = "pathtracing-sample";
constexpr const char* kPathTracingDlssRrSampleId = "pathtracing-sample-dlss-rr";

void printUsage()
{
    std::cout
        << "MetallicPathTracingSample options:\n"
        << "  --dlss-rr                   Use the NVIDIA DLSS-RR denoiser graph\n"
        << "  --smoke-test                 Render one frame and exit\n"
        << "  --wait-for-graphics-debugger Wait before Vulkan initialization\n";
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    const char* sampleId = kPathTracingSampleId;
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

        std::cerr << "Unknown argument: " << argument << '\n';
        printUsage();
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(smokeTest, waitForGraphicsDebugger, sampleId);
}
