#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/ShaderTarget.h"
#include "ShaderWarmupRequests.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace metallic::render;

namespace {

std::atomic<bool> startupWarmupEnabled{false};

struct CompileOutcome {
    bool cacheHit = false;
    std::string error;
    ShaderCompileMetrics metrics;
};

// Request completion is provisional until the whole source batch is validated.
// Keep this separate from the accepted-result summary and its timing format.
class WarmupBatchReporter {
public:
    explicit WarmupBatchReporter(std::span<const SlangShaderDesc> requests)
        : requests_(requests), start_(std::chrono::steady_clock::now()), heartbeat_([this] {
            std::unique_lock lock(mutex_);
            while (!wake_.wait_for(lock, std::chrono::seconds(2), [this] { return stopped_; })) {
                printLocked("working");
            }
        })
    {
    }

    ~WarmupBatchReporter()
    {
        {
            std::lock_guard lock(mutex_);
            stopped_ = true;
        }
        wake_.notify_all();
        heartbeat_.join();
    }

    void update(const SlangWarmupBatchProgress& event)
    {
        std::lock_guard lock(mutex_);
        if (std::string_view(event.phase) == "request-complete") {
            ++ready_;
            hits_ += event.cacheHit ? 1 : 0;
            const auto& request = requests_[event.requestIndex];
            const auto status = std::string(event.cacheHit ? "cache ready: " : "compile ready: ") +
                request.moduleName + "." + request.entryPointName;
            printLocked(status.c_str());
        } else {
            phase_ = event.phase;
            printLocked("phase changed");
        }
    }

private:
    void printLocked(const char* status)
    {
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        std::cout << "[Warmup] " << phase_ << " | ready: " << ready_ << '/' << requests_.size()
                  << " | cached: " << hits_ << " | elapsed: " << std::fixed << std::setprecision(1)
                  << seconds << "s | " << status << " | awaiting batch acceptance" << std::endl;
    }

    std::span<const SlangShaderDesc> requests_;
    std::chrono::steady_clock::time_point start_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_ = false;
    size_t ready_ = 0;
    size_t hits_ = 0;
    std::string phase_ = "preparing";
    std::thread heartbeat_;
};

int compilePriority(const metallic::tools::ShaderWarmupRequest& request)
{
    // Start expensive programs while every worker is available. Catalog order
    // remains stable for listing and request identification.
    if (request.module.starts_with("Features/PathTracing/") &&
        request.module != "Features/PathTracing/SceneSharcMaintenance" &&
        request.module != "Features/PathTracing/RayMaterialQueue") {
        return request.module.find("OpenPBR") != std::string::npos ? 4 : 3;
    }
    if (request.module == "Features/VisibilityBuffer/VisibilityBufferDeferred") { return 2; }
    if (request.module == "Features/Lighting/SceneRealtimeLighting") { return 3; }
    if (request.module == "Features/ReSTIR/SceneRTXDI") { return 1; }
    return 0;
}

CompileOutcome compileRequest(const metallic::tools::ShaderWarmupRequest& request,
    const std::string& cacheDirectory, DescriptorHeapShaderStrides strides)
{
    const ShaderRequestView source(request);
    SlangShaderDesc desc = source.desc();
    desc.descriptorHeapStrides = strides;
    bool cacheHit = false;
    const SlangShaderCacheOptions options{
        .cacheDirectory = cacheDirectory.empty() ? nullptr : cacheDirectory.c_str(),
        .outCacheHit = &cacheHit,
    };
    std::string diagnostics;
    const auto compiled = ShaderRegistry::instance().getShader(desc, options, diagnostics);
    if (!compiled) {
        return {false, std::string(resultToString(compiled)) + "\n" + diagnostics};
    }
    const bool existingCacheHit = cacheHit;
    // Every worker owns its compiler sessions and output. Keep cache read-back
    // validation identical to the serial warmup.
    if (!cacheHit) {
        const auto cached = ShaderRegistry::instance().getShader(desc, options, diagnostics);
        if (!cached || !cacheHit || cached->spirv != compiled->spirv) {
            return {false, "cache read-back verification failed"};
        }
    }
    return {existingCacheHit, {}, compiled->metrics};
}

int run(int argc, char** argv)
{
    std::string filter;
    std::string cacheDirectory;
    bool listOnly = false;
    DescriptorHeapShaderStrides explicitStrides;
    bool resourceStrideSpecified = false;
    bool samplerStrideSpecified = false;
    size_t jobs = std::clamp(std::thread::hardware_concurrency(), 1u, 12u);
    bool jobsSpecified = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            std::cout << "MetallicShaderCompiler [--list] [--filter substring] [--cache-dir path]\n"
                         "                       [--debug-mode disabled|capture|debug] [--jobs N]\n"
                         "                       [--resource-heap-stride N --sampler-heap-stride N]\n"
                         "Default workers: min(CPU threads, 12); --jobs 1 compiles serially.\n"
                         "METALLIC_SHADER_WARMUP_JOBS overrides the default, including application startup.\n"
                         "--jobs takes precedence over the environment variable.\n"
                         "Native shaders query a supported adapter unless both byte strides are supplied.\n"
                         "Defaults to the runtime .cache/shaders/spirv cache. No GPU device is created.\n";
            return 0;
        }
        if (argument == "--list") {
            listOnly = true;
        } else if ((argument == "--filter" || argument == "--cache-dir" || argument == "--debug-mode" || argument == "--jobs" ||
            argument == "--resource-heap-stride" || argument == "--sampler-heap-stride") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (argument == "--filter") {
                filter = value;
            } else if (argument == "--cache-dir") {
                cacheDirectory = value;
            } else if (argument == "--jobs") {
                jobsSpecified = true;
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), jobs);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || jobs == 0) {
                    std::cerr << "--jobs requires a positive integer\n";
                    return 2;
                }
            } else if (argument == "--resource-heap-stride" || argument == "--sampler-heap-stride") {
                uint32_t& stride = argument == "--resource-heap-stride" ? explicitStrides.resource : explicitStrides.sampler;
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), stride);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || stride == 0) {
                    std::cerr << argument << " requires a positive byte stride\n";
                    return 2;
                }
                (argument == "--resource-heap-stride" ? resourceStrideSpecified : samplerStrideSpecified) = true;
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
    if (resourceStrideSpecified != samplerStrideSpecified) {
        std::cerr << "--resource-heap-stride and --sampler-heap-stride must be supplied together\n";
        return 2;
    }
    if (!jobsSpecified && !listOnly) {
        const char* environmentJobs = std::getenv("METALLIC_SHADER_WARMUP_JOBS");
        if (environmentJobs != nullptr && environmentJobs[0] != '\0') {
            const std::string_view value(environmentJobs);
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), jobs);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || jobs == 0) {
                std::cerr << "METALLIC_SHADER_WARMUP_JOBS requires a positive integer\n";
                return 2;
            }
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

    std::vector<bool> nativeRequests(requests.size(), false);
    DescriptorHeapShaderStrides nativeStrides;
    if (!listOnly) {
        for (size_t index = 0; index < requests.size(); ++index) {
            const ShaderRequestView source(requests[index]);
            nativeRequests[index] = slangUsesNativeDescriptorHeap(source.desc());
        }
        if (std::ranges::any_of(nativeRequests, [](bool native) { return native; })) {
            if (resourceStrideSpecified) {
                nativeStrides = explicitStrides;
            } else {
                std::string diagnostics;
                if (!queryVulkanDescriptorHeapShaderStrides(nativeStrides, diagnostics) ||
                    nativeStrides.resource == 0 || nativeStrides.sampler == 0) {
                    std::cerr << "Native shader warmup cannot determine descriptor heap strides: " << diagnostics << '\n';
                    return 1;
                }
            }
            std::cout << "Native shader warmup descriptor strides: resource=" << nativeStrides.resource
                      << ", sampler=" << nativeStrides.sampler << " bytes" << std::endl;
        }
    }
    const size_t totalRequests = requests.size();

    // Restore application logging and console formatting, including failure paths.
    struct OutputStateGuard {
        spdlog::level::level_enum level = spdlog::get_level();
        std::ios::fmtflags flags = std::cout.flags();
        std::streamsize precision = std::cout.precision();

        ~OutputStateGuard()
        {
            spdlog::set_level(level);
            std::cout.flags(flags);
            std::cout.precision(precision);
        }
    } outputStateGuard;
    spdlog::set_level(spdlog::level::warn);
    const auto start = std::chrono::steady_clock::now();
    size_t selected = 0;
    size_t hits = 0;
    size_t failures = 0;
    uint64_t compilerTimeNanoseconds = 0;
    uint64_t moduleLoadTimeNanoseconds = 0;
    uint64_t codeGenerationTimeNanoseconds = 0;
    const auto progress = [&](const char* status) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "[" << selected << '/' << totalRequests << " "
                  << (selected * 100 / totalRequests) << "%] " << status
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
        std::vector<std::unique_ptr<ShaderRequestView>> views;
        std::vector<SlangShaderDesc> descriptors;
        views.reserve(requests.size()); descriptors.reserve(requests.size());
        for (size_t index = 0; index < requests.size(); ++index) {
            views.push_back(std::make_unique<ShaderRequestView>(requests[index]));
            auto desc = views.back()->desc();
            desc.descriptorHeapStrides = nativeRequests[index] ? nativeStrides : DescriptorHeapShaderStrides{};
            descriptors.push_back(desc);
        }
        std::cout << "Shader warmup workers: " << jobs << std::endl;
        progress("Starting validated immutable source batch");
        std::string diagnostics;
        auto batch = [&] {
            WarmupBatchReporter reporter(descriptors);
            const SlangWarmupBatchOptions batchOptions{
                .cache = {.cacheDirectory = cacheDirectory.empty() ? nullptr : cacheDirectory.c_str()},
                .jobs = jobs, .shareSessions = true,
                .onProgress = [&](const SlangWarmupBatchProgress& event) { reporter.update(event); }};
            return ShaderRegistry::instance().getWarmupShaders(descriptors, batchOptions, diagnostics);
        }();
        if (!batch || !batch->completedAllRequests || batch->shaders.size() != totalRequests) {
            ++failures; std::cerr << "Complete shader warmup batch rejected: " << diagnostics << '\n';
        } else {
            for (size_t index = 0; index < totalRequests; ++index) {
                const auto& metrics = batch->shaders[index].metrics;
                ++selected; hits += metrics.cacheHit ? 1 : 0;
                compilerTimeNanoseconds += metrics.totalTimeNanoseconds;
                moduleLoadTimeNanoseconds += metrics.moduleLoadTimeNanoseconds;
                codeGenerationTimeNanoseconds += metrics.codeGenerationTimeNanoseconds;
                const auto name = requests[index].module + "." + requests[index].entry;
                std::cout << std::setprecision(3) << "  Request " << (index + 1) << " timing: total="
                          << double(metrics.totalTimeNanoseconds) / 1e9 << "s, setup="
                          << double(metrics.compilerSetupTimeNanoseconds) / 1e9 << "s, module="
                          << double(metrics.moduleLoadTimeNanoseconds) / 1e9 << "s, codegen="
                          << double(metrics.codeGenerationTimeNanoseconds) / 1e9 << "s" << std::endl;
                const auto status = std::string(metrics.cacheHit ? "Cached: " : "Compiled: ") + name;
                progress(status.c_str());
            }
            if (!batch->completeSourceEpochValidated) {
                std::cout << "Shader warmup used fresh sessions without persistent cache reuse\n";
            }
            std::cout << "Warmup source batch: " << batch->sessionGroups << " groups; static estimate schedule; snapshot="
                      << double(batch->snapshotCaptureTimeNanoseconds) / 1e9 << "s; boundary="
                      << double(batch->boundaryValidationTimeNanoseconds) / 1e9 << "s; cache publication/readback="
                      << double(batch->cachePublicationTimeNanoseconds) / 1e9 << "s; live_fs_atomic=false\n";
            std::cout << "Warmup V3 publication phases: proof encode="
                      << double(batch->cacheProofEncodeTimeNanoseconds) / 1e9 << "s; atomic write="
                      << double(batch->cacheAtomicWriteTimeNanoseconds) / 1e9 << "s; file read/decode="
                      << double(batch->cacheFileReadDecodeTimeNanoseconds) / 1e9 << "s; full source validation="
                      << double(batch->cacheFullSourceValidationTimeNanoseconds) / 1e9 << "s\n";
        }
    }
    std::cout << "Shader warmup: " << selected << " requests, " << hits
              << " existing cache hits, " << failures << " failures"
              << (listOnly ? " (list only)" : "") << '\n';
    if (!listOnly) {
        std::cout << std::setprecision(3) << "Shader warmup elapsed: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << "s; summed request time: " << double(compilerTimeNanoseconds) / 1e9
                  << "s (module=" << double(moduleLoadTimeNanoseconds) / 1e9
                  << "s, codegen=" << double(codeGenerationTimeNanoseconds) / 1e9 << "s)\n";
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int metallic::render::runShaderWarmup(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Shader warmup failed: " << error.what() << '\n';
        return 1;
    }
}

int metallic::render::warmupShadersForStartup(bool skip)
{
    startupWarmupEnabled.store(false, std::memory_order_release);
    if (skip) {
        return 0;
    }
    std::cout << "[Startup] Warming shader cache before application initialization" << std::endl;
    char program[] = "MetallicShaderWarmup";
    char* arguments[] = {program, nullptr};
    const int result = runShaderWarmup(1, arguments);
    if (result != 0) {
        std::cerr << "[Startup] Shader warmup failed; application startup cancelled. "
                     "Use --skip-shader-warmup to compile on demand.\n";
    }
    startupWarmupEnabled.store(result == 0, std::memory_order_release);
    return result;
}

bool metallic::render::startupShaderWarmupEnabled() noexcept
{
    return startupWarmupEnabled.load(std::memory_order_acquire);
}
