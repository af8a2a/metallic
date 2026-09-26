#include "Runtime/Render/SlangCompiler.h"
#include "ShaderWarmupRequests.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
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

struct CompileOutcome {
    bool cacheHit = false;
    std::string error;
};

CompileOutcome compileRequest(const metallic::tools::ShaderWarmupRequest& request, const std::string& cacheDirectory)
{
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
        return {false, std::string(resultToString(result)) + "\n" + compiled.diagnostics};
    }
    const bool existingCacheHit = cacheHit;
    // Every worker owns its compiler sessions and output. Keep cache read-back
    // validation identical to the serial warmup.
    if (!cacheHit) {
        ShaderCompileResult cached;
        const auto verified = compileSlangShaderToSpirv(desc, options, cached);
        if (!verified || !cacheHit || cached.spirv != compiled.spirv) {
            return {false, "cache read-back verification failed"};
        }
    }
    return {existingCacheHit, {}};
}

int run(int argc, char** argv)
{
    std::string filter;
    std::string cacheDirectory;
    bool listOnly = false;
    size_t jobs = std::clamp(std::thread::hardware_concurrency(), 1u, 4u);
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            std::cout << "MetallicShaderCompiler [--list] [--filter substring] [--cache-dir path]\n"
                         "                       [--debug-mode disabled|capture|debug] [--jobs N]\n"
                         "Default workers: min(CPU threads, 4); --jobs 1 compiles serially.\n"
                         "Defaults to the runtime .cache/shaders/spirv cache. No GPU is required.\n";
            return 0;
        }
        if (argument == "--list") {
            listOnly = true;
        } else if ((argument == "--filter" || argument == "--cache-dir" || argument == "--debug-mode" || argument == "--jobs") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (argument == "--filter") {
                filter = value;
            } else if (argument == "--cache-dir") {
                cacheDirectory = value;
            } else if (argument == "--jobs") {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), jobs);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || jobs == 0) {
                    std::cerr << "--jobs requires a positive integer\n";
                    return 2;
                }
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

    auto requests = metallic::tools::shaderWarmupRequests();
    std::erase_if(requests, [&](const auto& request) {
        return !filter.empty() && (request.module + "." + request.entry).find(filter) == std::string::npos;
    });
    if (requests.empty()) {
        std::cerr << "No shader requests matched the filter\n";
        return 2;
    }

    // Keep progress readable without changing logging in the application.
    spdlog::set_level(spdlog::level::warn);
    const auto start = std::chrono::steady_clock::now();
    size_t selected = 0;
    size_t hits = 0;
    size_t failures = 0;
    const auto progress = [&](const char* status) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "[" << selected << '/' << requests.size() << " "
                  << (selected * 100 / requests.size()) << "%] " << status
                  << " | cached: " << hits << " | failed: " << failures
                  << " | elapsed: " << std::fixed << std::setprecision(1) << seconds << "s";
        std::cout << std::endl;
    };
    if (listOnly) {
        for (const auto& request : requests) {
            std::cout << '[' << ++selected << '/' << requests.size() << "] " << request.module << '.' << request.entry;
            for (const auto& define : request.defines) {
                std::cout << " " << define.first << '=' << define.second;
            }
            std::cout << '\n';
        }
    } else {
        jobs = std::min(jobs, requests.size());
        std::cout << "Shader warmup workers: " << jobs << std::endl;
        progress("Starting");
        std::atomic<size_t> nextRequest{0};
        std::mutex outputMutex;
        {
            // jthread also joins already-started workers if thread creation fails.
            // All shared state outlives the workers; counters and output share a lock.
            std::vector<std::jthread> workers;
            workers.reserve(jobs);
            for (size_t worker = 0; worker < jobs; ++worker) {
                workers.emplace_back([&] {
                    for (;;) {
                        const size_t index = nextRequest.fetch_add(1, std::memory_order_relaxed);
                        if (index >= requests.size()) {
                            return;
                        }
                        const auto& request = requests[index];
                        const std::string name = request.module + "." + request.entry;
                        {
                            std::scoped_lock lock(outputMutex);
                            std::cout << "  Processing " << (index + 1) << '/' << requests.size()
                                      << ": " << name << std::endl;
                        }
                        CompileOutcome outcome;
                        try {
                            outcome = compileRequest(request, cacheDirectory);
                        } catch (const std::exception& error) {
                            outcome.error = error.what();
                        } catch (...) {
                            outcome.error = "unknown compilation exception";
                        }
                        std::scoped_lock lock(outputMutex);
                        ++selected;
                        hits += outcome.cacheHit ? 1 : 0;
                        if (!outcome.error.empty()) {
                            ++failures;
                            std::cerr << name << ": " << outcome.error << '\n';
                        }
                        const std::string status = std::string(!outcome.error.empty() ? "Failed: " :
                            outcome.cacheHit ? "Cached: " : "Compiled: ") + name;
                        progress(status.c_str());
                    }
                });
            }
        }
    }
    std::cout << "Shader warmup: " << selected << " requests, " << hits
              << " existing cache hits, " << failures << " failures"
              << (listOnly ? " (list only)" : "") << '\n';
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
