#include "Runtime/Render/SlangCompiler.h"
#include "ShaderWarmupRequests.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace metallic::render;

namespace {

std::vector<const char*> stringPointers(const std::vector<std::string>& strings)
{
    std::vector<const char*> pointers;
    for (const auto& string : strings) {
        pointers.push_back(string.c_str());
    }
    return pointers;
}

int run(int argc, char** argv)
{
    std::string filter;
    std::string cacheDirectory;
    bool listOnly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            std::cout << "MetallicShaderCompiler [--list] [--filter substring] [--cache-dir path]\n"
                         "                       [--debug-mode disabled|capture|debug]\n"
                         "Defaults to the runtime .cache/shaders/spirv cache. No GPU is required.\n";
            return 0;
        }
        if (argument == "--list") {
            listOnly = true;
        } else if ((argument == "--filter" || argument == "--cache-dir" || argument == "--debug-mode") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (argument == "--filter") {
                filter = value;
            } else if (argument == "--cache-dir") {
                cacheDirectory = value;
            } else if (value == "disabled") {
                setSlangShaderDebugMode(SlangShaderDebugMode::Disabled);
            } else if (value == "capture") {
                setSlangShaderDebugMode(SlangShaderDebugMode::CaptureSymbols);
            } else if (value == "debug") {
                setSlangShaderDebugMode(SlangShaderDebugMode::ShaderDebug);
            } else {
                std::cerr << "Unknown debug mode: " << value << '\n';
                return 2;
            }
        } else {
            std::cerr << "Unknown argument or missing value: " << argument << '\n';
            return 2;
        }
    }

    size_t selected = 0;
    size_t hits = 0;
    size_t failures = 0;
    for (const auto& request : metallic::tools::shaderWarmupRequests()) {
        const std::string name = request.module + "." + request.entry;
        if (!filter.empty() && name.find(filter) == std::string::npos) {
            continue;
        }
        ++selected;
        std::cout << '[' << selected << "] " << name;
        for (const auto& define : request.defines) {
            std::cout << " " << define.first << '=' << define.second;
        }
        std::cout << std::endl;
        if (listOnly) {
            continue;
        }
        const auto capabilities = stringPointers(request.capabilities);
        const auto searchPaths = stringPointers(request.searchPaths);
        std::vector<SlangMacroDefine> defines;
        for (const auto& define : request.defines) {
            defines.push_back({define.first.c_str(), define.second.c_str()});
        }
        const SlangShaderDesc desc{
            .moduleName = request.module.c_str(),
            .entryPointName = request.entry.c_str(),
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .additionalSearchPaths = searchPaths.data(),
            .additionalSearchPathCount = static_cast<uint32_t>(searchPaths.size()),
            .capabilities = capabilities.data(),
            .capabilityCount = static_cast<uint32_t>(capabilities.size()),
            .macroDefines = defines.data(),
            .macroDefineCount = static_cast<uint32_t>(defines.size()),
        };
        bool cacheHit = false;
        const SlangShaderCacheOptions options{
            .cacheDirectory = cacheDirectory.empty() ? nullptr : cacheDirectory.c_str(),
            .outCacheHit = &cacheHit,
        };
        ShaderCompileResult compiled;
        const auto result = compileSlangShaderToSpirv(desc, options, compiled);
        if (!result) {
            ++failures;
            std::cerr << name << ": " << resultToString(result) << '\n' << compiled.diagnostics << '\n';
            continue;
        }
        hits += cacheHit ? 1 : 0;
        // Runtime compilation tolerates cache write failures. Warmup must verify
        // that the result can actually be read back from disk.
        if (!cacheHit) {
            ShaderCompileResult cached;
            const auto verified = compileSlangShaderToSpirv(desc, options, cached);
            if (!verified || !cacheHit || cached.spirv != compiled.spirv) {
                ++failures;
                std::cerr << name << ": cache read-back verification failed\n";
            }
        }
    }
    std::cout << "Shader warmup: " << selected << " requests, " << hits
              << " existing cache hits, " << failures << " failures"
              << (listOnly ? " (list only)" : "") << '\n';
    if (selected == 0) {
        std::cerr << "No shader requests matched the filter\n";
        return 2;
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Shader warmup failed: " << error.what() << '\n';
        return 1;
    }
}
