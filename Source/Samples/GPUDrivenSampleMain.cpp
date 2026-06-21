#include "Editor/EditorApplication.h"

#include <iostream>
#include <string_view>

namespace {

constexpr const char* kGPUDrivenSampleId = "gpu-driven-sample";

void printUsage()
{
    std::cout
        << "MetallicGPUDrivenSample options:\n"
        << "  --smoke-test                 Render one frame and exit\n"
        << "  --wait-for-graphics-debugger Wait before Vulkan initialization\n";
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
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

        std::cerr << "Unknown argument: " << argument << '\n';
        printUsage();
        return 1;
    }

    metallic::EditorApplication app;
    return app.run(smokeTest, waitForGraphicsDebugger, kGPUDrivenSampleId);
}
