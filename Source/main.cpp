#include "Editor/editor_application.h"

#include <string_view>

int main(int argc, char** argv)
{
    bool smokeTest = false;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--smoke-test") {
            smokeTest = true;
        }
    }

    metallic::EditorApplication app;
    return app.run(smokeTest);
}
