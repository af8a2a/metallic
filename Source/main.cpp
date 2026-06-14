#include "Editor/editor_application.h"
#include "Runtime/Render/GAPI/rhi.h"

#include <cstdlib>
#include <string>
#include <string_view>

namespace {

bool isEnabledValue(std::string_view value)
{
    return !value.empty() && value != "0" && value != "false" && value != "FALSE";
}

bool waitForGraphicsDebuggerFromEnv()
{
#if defined(_WIN32)
    size_t requiredSize = 0;
    if (getenv_s(&requiredSize, nullptr, 0, "METALLIC_WAIT_FOR_GRAPHICS_DEBUGGER") != 0 ||
        requiredSize == 0) {
        return false;
    }

    std::string value(requiredSize, '\0');
    if (getenv_s(&requiredSize, value.data(), value.size(), "METALLIC_WAIT_FOR_GRAPHICS_DEBUGGER") != 0) {
        return false;
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return isEnabledValue(value);
#else
    const char* value = std::getenv("METALLIC_WAIT_FOR_GRAPHICS_DEBUGGER");
    return value != nullptr && isEnabledValue(value);
#endif
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool rhiSmokeTest = false;
    bool rhiTrianglePreviewTest = false;
    bool rhiValidation = true;
    bool waitForGraphicsDebugger = waitForGraphicsDebuggerFromEnv();
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--smoke-test") {
            smokeTest = true;
        } else if (argument == "--rhi-smoke-test") {
            rhiSmokeTest = true;
        } else if (argument == "--rhi-triangle-preview-test") {
            rhiTrianglePreviewTest = true;
        } else if (argument == "--rhi-no-validation") {
            rhiValidation = false;
        } else if (argument == "--wait-for-graphics-debugger") {
            waitForGraphicsDebugger = true;
        }
    }

    if (rhiTrianglePreviewTest) {
        return metallic::render::runRhiTrianglePreviewTest(rhiValidation);
    }

    if (rhiSmokeTest) {
        return metallic::render::runRhiSmokeTest(rhiValidation);
    }

    metallic::EditorApplication app;
    return app.run(smokeTest, waitForGraphicsDebugger);
}
