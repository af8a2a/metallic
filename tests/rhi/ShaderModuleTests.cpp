#include "RhiTest.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace metallic::tests {
namespace {

class SlangShaderModuleTest : public RhiTest {
public:
    SlangShaderModuleTest()
    {
        type = RhiTestType::Resource;
        name = "slang_shader_modules_and_vendor_interop";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        struct TrackingGuard {
            TrackingGuard()
            {
                render::resetSlangShaderHotReloadTracking();
            }
            ~TrackingGuard()
            {
                render::resetSlangShaderHotReloadTracking();
            }
        } tracking;
        const auto root = std::filesystem::absolute(context.outputDirectory / name);
        const auto write = [&](const char* path, const std::string& text) {
            std::error_code error;
            std::filesystem::create_directories((root / path).parent_path(), error);
            std::ofstream stream(root / path, std::ios::binary | std::ios::trunc);
            stream << text;
            return !error && static_cast<bool>(stream);
        };
        // Both branches import the same owner. Include guards alone cannot
        // deduplicate vendor declarations across independent Slang modules.
        if (!write("Modules/ModuleMath.slang",
                "#language slang 2026\nmodule ModuleMath;\n"
                "__include \"Math/Value.slang\";\n") ||
            !write("Modules/Math/Value.slang",
                "#language slang 2026\nimplementing ModuleMath;\n"
                "internal uint hiddenValue() { return 3u; }\n"
                "public uint moduleValue() { return hiddenValue(); }\n") ||
            !write("Interop/Vendor/Value.hlsli",
                "#ifndef VENDOR_VALUE_HEADER\n#define VENDOR_VALUE_HEADER\n"
                "uint vendorValue() { return VENDOR_SCALE * 5u; }\n#endif\n") ||
            !write("Interop/VendorAdapter.slang",
                "#language slang 2026\nmodule VendorAdapter;\n"
                "#ifndef VENDOR_SCALE\n#define VENDOR_SCALE 2u\n#endif\n"
                "#include \"Vendor/Value.hlsli\"\n"
                "public uint adapterValue() { return vendorValue(); }\n") ||
            !write("Modules/ModuleLeft.slang",
                "module ModuleLeft;\nimport ModuleMath;\nimport VendorAdapter;\n"
                "public uint leftValue() { return moduleValue() + adapterValue(); }\n") ||
            !write("Modules/ModuleRight.slang",
                "module ModuleRight;\nimport ModuleMath;\nimport VendorAdapter;\n"
                "public uint rightValue() { return moduleValue() * adapterValue(); }\n") ||
            !write("Program.slang",
                "#define VENDOR_SCALE 99u\n"
                "import ModuleLeft;\nimport ModuleRight;\n"
                "#ifdef VENDOR_VALUE_HEADER\n#error Vendor macros leaked through import\n#endif\n"
                "RWStructuredBuffer<uint> outputBuffer;\n"
                "[shader(\"compute\")] [numthreads(1, 1, 1)]\n"
                "void main() { outputBuffer[0] = leftValue() + rightValue(); }\n")) {
            return RhiTestResult::fail("could not create shader module fixtures");
        }

        const std::string sourceRoot = root.string();
        const std::string cacheRoot = (root / "cache").string();
        const render::SlangShaderDesc desc{
            .moduleName = "Program",
            .entryPointName = "main",
            .searchPath = sourceRoot.c_str(),
        };
        bool cacheHit = false;
        const render::SlangShaderCacheOptions cache{
            .cacheDirectory = cacheRoot.c_str(),
            .outCacheHit = &cacheHit,
        };
        render::ShaderCompileResult first;
        if (!render::compileSlangShaderToSpirv(desc, cache, first)) {
            return RhiTestResult::fail(first.diagnostics);
        }
        for (const char* dependency : {"Modules/ModuleMath.slang", "Modules/Math/Value.slang",
                 "Modules/ModuleLeft.slang", "Modules/ModuleRight.slang",
                 "Interop/VendorAdapter.slang", "Interop/Vendor/Value.hlsli", "Program.slang"}) {
            const std::string path = (root / dependency).lexically_normal().generic_string();
            if (std::count(first.dependencies.begin(), first.dependencies.end(), path) != 1) {
                return RhiTestResult::fail("missing or duplicate transitive dependency: " + path);
            }
        }
        render::ShaderCompileResult cached;
        if (!render::compileSlangShaderToSpirv(desc, cache, cached) || !cacheHit ||
            first.spirv != cached.spirv || first.dependencies != cached.dependencies) {
            return RhiTestResult::fail("module dependency cache did not round trip");
        }
        for (const auto& [file, source] : {
                 std::pair{"Modules/Math/Value.slang",
                     "implementing ModuleMath;\npublic uint moduleValue() { return 7u; }\n"},
                 std::pair{"Interop/Vendor/Value.hlsli",
                     "uint vendorValue() { return VENDOR_SCALE * 11u; }\n"}}) {
            if (!write(file, source)) {
                return RhiTestResult::fail("could not edit module dependency");
            }
            const auto changes = render::pollSlangShaderChanges(0, 0);
            if (changes != std::vector<std::string>{(root / file).lexically_normal().generic_string()}) {
                return RhiTestResult::fail("module or vendor edit was not tracked precisely");
            }
            render::ShaderCompileResult changed;
            if (!render::compileSlangShaderToSpirv(desc, cache, changed) || cacheHit ||
                changed.spirv == cached.spirv) {
                return RhiTestResult::fail("module or vendor edit reused stale IR/SPIR-V: " + changed.diagnostics);
            }
            cached = std::move(changed);
            render::acknowledgeSlangShaderChanges();
        }

        // An importing program's local define must not configure the adapter.
        // A session define must configure it, and must get a different cache key.
        const render::SlangMacroDefine macro{"VENDOR_SCALE", "3u"};
        auto variantDesc = desc;
        variantDesc.macroDefines = &macro;
        variantDesc.macroDefineCount = 1;
        render::ShaderCompileResult variant;
        if (!render::compileSlangShaderToSpirv(variantDesc, cache, variant) ||
            variant.spirv == cached.spirv) {
            return RhiTestResult::fail("SDK macro variant reused the wrong module: " + variant.diagnostics);
        }
        render::ShaderCompileResult variantCached;
        if (!render::compileSlangShaderToSpirv(variantDesc, cache, variantCached) || !cacheHit ||
            variantCached.spirv != variant.spirv) {
            return RhiTestResult::fail("SDK macro variant did not retain its own cache entry");
        }
        if (!write("Program.slang",
                "import Core;\nusing Metallic;\nRWStructuredBuffer<float3> outputBuffer;\n"
                "[shader(\"compute\")] [numthreads(1, 1, 1)]\n"
                "void main() { outputBuffer[0] = decodeSceneOctahedron(float2(0)); }\n")) {
            return RhiTestResult::fail("could not write access control probe");
        }
        render::ShaderCompileResult inaccessible;
        if (render::compileSlangShaderToSpirv(desc, cache, inaccessible) ||
            inaccessible.diagnostics.find("not accessible") == std::string::npos) {
            return RhiTestResult::fail("Core implementation detail escaped its module boundary");
        }
        return RhiTestResult::pass("validated module ownership, visibility, SDK macro isolation, cache and hot reload");
    }
};

METALLIC_REGISTER_RHI_TEST(SlangShaderModuleTest);

} // namespace
} // namespace metallic::tests
