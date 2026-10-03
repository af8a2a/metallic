#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>

namespace metallic::tests {
namespace {

class ShaderToHumanShaderCompileTest final : public RHITest {
public:
    ShaderToHumanShaderCompileTest()
    {
        type = RHITestType::Resource;
        name = "shader_to_human_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        struct ShaderEntry {
            const char* module;
            const char* entry;
            const char* dependency;
        };
        constexpr ShaderEntry entries[] = {
            {"Features/Debug/ShaderToHumanExample", "shaderToHumanExampleMain", "/include/s2h.hlsl"},
            {"Features/Debug/ShaderToHumanExample", "shaderToHumanExampleFragmentMain", "/include/s2h_3d.hlsl"},
            {"Features/Debug/ShaderToHumanScatterExample", "shaderToHumanScatterExampleMain", "/include/s2h_scatter.hlsl"},
        };
        for (const ShaderEntry& entry : entries) {
            render::ShaderCompileResult shader;
            const auto result = render::compileSlangShaderToSpirv({
                .moduleName = entry.module,
                .entryPointName = entry.entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            if (!result || shader.spirv.empty()) {
                return RHITestResult::fail(std::string(entry.entry) + ": " + shader.diagnostics);
            }
            // The vendored HLSL must participate in Slang cache invalidation
            // and hot reload just like the surrounding .slang files.
            if (!std::any_of(shader.dependencies.begin(), shader.dependencies.end(),
                    [&](const std::string& path) { return path.ends_with(entry.dependency); })) {
                return RHITestResult::fail(std::string("Missing ShaderToHuman dependency: ") + entry.dependency);
            }
        }
        return RHITestResult::pass("ShaderToHuman gather/3D in compute and fragment, scatter callback, HLSL dependency tracking");
    }
};

METALLIC_REGISTER_RHI_TEST(ShaderToHumanShaderCompileTest);

} // namespace
} // namespace metallic::tests
