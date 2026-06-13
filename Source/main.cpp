#include "Editor/editor_application.h"
#include "Runtime/Render/GAPI/rhi.h"

#include <string_view>

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool rhiSmokeTest = false;
    bool rhiValidation = true;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--smoke-test") {
            smokeTest = true;
        } else if (argument == "--rhi-smoke-test") {
            rhiSmokeTest = true;
        } else if (argument == "--rhi-no-validation") {
            rhiValidation = false;
        }
    }

    if (rhiSmokeTest) {
        return metallic::render::runRhiSmokeTest(rhiValidation);
    }

    metallic::EditorApplication app;
    return app.run(smokeTest);
}
