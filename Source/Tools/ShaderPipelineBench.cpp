#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/Core/ShaderResourceABI.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"
#include "Tools/ShaderWarmupRequests.h"
#include "Tools/SPIRVStatistics.h"

#include <SDL3/SDL.h>
#include <json.hpp>
#include <slang.h>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
using namespace metallic::render;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string filter, outputDirectory, cacheDirectory, pipelineCacheFile;
    std::string shaderCache = "disabled", pipelineCache = "empty";
    std::string descriptorMode = "request", debugMode = "disabled";
    std::string mode = "serial", cacheLockScope = "metadata", recipeFile;
    std::string slangOptimization = "default", computeBackend = "pipeline", nativeBinaryCacheDirectory;
    std::string spirvPostprocess = "default";
    uint32_t iterations = 3;
    uint32_t pipelineReplicas = 1;
    std::optional<uint32_t> pipelineWorkers;
    std::optional<uint32_t> requestIndex;
    bool list = false, all = false, compileOnly = false, validation = false;
    bool explicitMode = false, mergeShards = false, explicitComputeBackend = false;
    bool disableDriverOptimization = false, slangPerfDiagnostics = false;
};

struct CapturedValidationMessage {
    ValidationSeverity severity;
    ValidationCategory type;
    int32_t id;
    std::string idName, text;
};

struct ValidationCapture {
    std::atomic<uint32_t> errors = 0, dropped = 0;
    std::mutex mutex;
    std::vector<CapturedValidationMessage> messages;

    static void callback(void* context, const ValidationMessage& message) noexcept
    {
        auto& capture = *static_cast<ValidationCapture*>(context);
        if (message.severity == ValidationSeverity::Error) { ++capture.errors; }
        try {
            std::lock_guard lock(capture.mutex);
            if (capture.messages.size() == 256) { ++capture.dropped; return; }
            capture.messages.push_back({message.severity, message.type, message.messageId,
                message.messageIdName ? message.messageIdName : "", message.message ? message.message : ""});
        } catch (...) { ++capture.dropped; }
    }

    void snapshot(Json& report)
    {
        Json captured = Json::array();
        std::lock_guard lock(mutex);
        for (const auto& message : messages) {
            const char* severity = "unknown";
            switch (message.severity) {
            case ValidationSeverity::Verbose: severity = "verbose"; break;
            case ValidationSeverity::Info: severity = "info"; break;
            case ValidationSeverity::Warning: severity = "warning"; break;
            case ValidationSeverity::Error: severity = "error"; break;
            default: break;
            }
            captured.push_back({{"severity", severity}, {"type", static_cast<uint8_t>(message.type)},
                {"id", message.id}, {"id_name", message.idName}, {"text", message.text}});
        }
        report["validation_messages"] = std::move(captured);
        report["validation_messages_dropped"] = dropped.load();
        report["validation_errors"] = errors.load();
    }
};

double milliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

uint32_t number(const std::string& value, const std::string& option)
{
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument(option + " requires an unsigned integer");
    }
    return result;
}

Options parse(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list") { options.list = true; }
        else if (arg == "--all") { options.all = true; }
        else if (arg == "--compile-only") { options.compileOnly = true; }
        else if (arg == "--validation") { options.validation = true; }
        else if (arg == "--merge-shards") { options.mergeShards = true; }
        else if (arg == "--disable-driver-optimization") { options.disableDriverOptimization = true; }
        else if (arg == "--slang-perf-diagnostics") { options.slangPerfDiagnostics = true; }
        else {
            if (i + 1 == argc) { throw std::invalid_argument("Missing value for " + arg); }
            const std::string value = argv[++i];
            if (arg == "--filter") { options.filter = value; }
            else if (arg == "--request-index") { options.requestIndex = number(value, arg); }
            else if (arg == "--iterations") { options.iterations = number(value, arg); }
            else if (arg == "--pipeline-workers" || arg == "--workers") { options.pipelineWorkers = number(value, arg); }
            else if (arg == "--pipeline-replicas") { options.pipelineReplicas = number(value, arg); }
            else if (arg == "--mode") { options.mode = value; options.explicitMode = true; }
            else if (arg == "--cache-lock-scope") { options.cacheLockScope = value; }
            else if (arg == "--recipe-file") { options.recipeFile = value; }
            else if (arg == "--output-dir") { options.outputDirectory = value; }
            else if (arg == "--cache-dir") { options.cacheDirectory = value; }
            else if (arg == "--shader-cache") { options.shaderCache = value; }
            else if (arg == "--pipeline-cache") { options.pipelineCache = value; }
            else if (arg == "--pipeline-cache-file") { options.pipelineCacheFile = value; }
            else if (arg == "--descriptor-mode") { options.descriptorMode = value; }
            else if (arg == "--debug-mode") { options.debugMode = value; }
            else if (arg == "--slang-optimization") { options.slangOptimization = value; }
            else if (arg == "--spirv-postprocess") { options.spirvPostprocess = value; }
            else if (arg == "--compute-backend") { options.computeBackend = value; options.explicitComputeBackend = true; }
            else if (arg == "--native-binary-cache-dir") { options.nativeBinaryCacheDirectory = value; }
            else { throw std::invalid_argument("Unknown argument: " + arg); }
        }
    }
    if (!options.iterations) { throw std::invalid_argument("--iterations must be positive"); }
    if (!options.pipelineReplicas) { throw std::invalid_argument("--pipeline-replicas must be positive"); }
    if (options.mode != "serial" && options.mode != "parallel-shared" && options.mode != "parallel-sharded") {
        throw std::invalid_argument("--mode requires serial, parallel-shared or parallel-sharded");
    }
    if (options.cacheLockScope != "metadata" && options.cacheLockScope != "creation") {
        throw std::invalid_argument("--cache-lock-scope requires metadata or creation");
    }
    if (!options.explicitMode && options.pipelineWorkers) { options.mode = "parallel-shared"; }
    if (options.explicitMode || options.pipelineReplicas > 1) {
        options.pipelineWorkers = options.pipelineWorkers.value_or(1);
    }
    if (options.mode == "serial" && options.pipelineWorkers.value_or(1) != 1) {
        throw std::invalid_argument("--mode serial requires --workers 1");
    }
    if (options.mergeShards && options.mode != "parallel-sharded") {
        throw std::invalid_argument("--merge-shards requires --mode parallel-sharded");
    }
    if (options.pipelineWorkers && !*options.pipelineWorkers) { throw std::invalid_argument("--pipeline-workers must be positive"); }
    if (options.pipelineWorkers && options.compileOnly) { throw std::invalid_argument("--pipeline-workers cannot be combined with --compile-only"); }
    if (options.shaderCache != "disabled" && options.shaderCache != "reuse") {
        throw std::invalid_argument("--shader-cache requires disabled or reuse");
    }
    if (options.pipelineCache != "empty" && options.pipelineCache != "reuse") {
        throw std::invalid_argument("--pipeline-cache requires empty or reuse");
    }
    if (!options.pipelineCacheFile.empty() && (options.pipelineCache != "reuse" || options.mode == "parallel-sharded")) {
        throw std::invalid_argument("--pipeline-cache-file requires --pipeline-cache reuse and serial or parallel-shared mode");
    }
    if (options.descriptorMode != "request" && options.descriptorMode != "mapped" && options.descriptorMode != "native") {
        throw std::invalid_argument("--descriptor-mode requires request, mapped or native");
    }
    if (options.debugMode != "disabled" && options.debugMode != "capture" && options.debugMode != "debug") {
        throw std::invalid_argument("--debug-mode requires disabled, capture or debug");
    }
    if (!options.list && (options.outputDirectory.empty() ||
        (!options.all && options.filter.empty() && !options.requestIndex && options.recipeFile.empty()))) {
        throw std::invalid_argument("Runs require --output-dir and --filter, --request-index or --all");
    }
    if (options.slangOptimization.size() == 2 && options.slangOptimization[0] == 'o') { options.slangOptimization[0] = 'O'; }
    if (options.slangOptimization != "default" && options.slangOptimization != "O0" && options.slangOptimization != "O1" &&
        options.slangOptimization != "O2" && options.slangOptimization != "O3") {
        throw std::invalid_argument("--slang-optimization requires default, O0, O1, O2 or O3");
    }
    if (options.spirvPostprocess != "default" && options.spirvPostprocess != "none" &&
        options.spirvPostprocess != "dead-branches" && options.spirvPostprocess != "slang-dead-branches") {
        throw std::invalid_argument("--spirv-postprocess requires default, none, dead-branches or slang-dead-branches");
    }
    if (options.computeBackend != "pipeline" && options.computeBackend != "shader-object" && options.computeBackend != "pipeline-binary") {
        throw std::invalid_argument("--compute-backend requires pipeline, shader-object or pipeline-binary");
    }
    if (options.computeBackend == "shader-object" && options.disableDriverOptimization) {
        throw std::invalid_argument("Shader Object has no DISABLE_OPTIMIZATION pipeline flag");
    }
    if (!options.nativeBinaryCacheDirectory.empty() && options.computeBackend == "pipeline" &&
        (options.recipeFile.empty() || options.explicitComputeBackend)) {
        throw std::invalid_argument("--native-binary-cache-dir requires shader-object or pipeline-binary backend");
    }
    if (options.computeBackend != "pipeline" && (options.pipelineCache != "empty" || options.mergeShards || options.cacheLockScope != "metadata")) {
        throw std::invalid_argument("Experimental compute backends do not use VkPipelineCache or its merge/lock controls");
    }
    if (!options.recipeFile.empty() && (options.all || !options.filter.empty() || options.requestIndex || options.compileOnly)) {
        throw std::invalid_argument("--recipe-file cannot be combined with catalog selectors or --compile-only");
    }
    if (!options.recipeFile.empty()) { options.pipelineWorkers = options.pipelineWorkers.value_or(1); }
    if (!options.recipeFile.empty() && options.slangOptimization != "default") {
        throw std::invalid_argument("Recipe replay uses captured SPIR-V; --slang-optimization requires a catalog request");
    }
    if (!options.recipeFile.empty() && options.spirvPostprocess != "default") {
        throw std::invalid_argument("--recipe-file contains finalized bytecode and cannot change SPIR-V postprocessing");
    }
    return options;
}

SlangShaderOptimizationLevel optimizationLevel(const std::string& name)
{
    if (name == "O0") { return SlangShaderOptimizationLevel::O0; }
    if (name == "O1") { return SlangShaderOptimizationLevel::O1; }
    if (name == "O2") { return SlangShaderOptimizationLevel::O2; }
    if (name == "O3") { return SlangShaderOptimizationLevel::O3; }
    return SlangShaderOptimizationLevel::Default;
}

SlangShaderPostprocessMode postprocessMode(const std::string& name)
{
    if (name == "none") { return SlangShaderPostprocessMode::None; }
    if (name == "dead-branches") { return SlangShaderPostprocessMode::DeadBranches; }
    if (name == "slang-dead-branches") { return SlangShaderPostprocessMode::SlangDeadBranches; }
    return SlangShaderPostprocessMode::Default;
}

const char* computeBackendName(ComputePipelineBackend backend)
{
    switch (backend) {
    case ComputePipelineBackend::Pipeline: return "pipeline";
    case ComputePipelineBackend::ShaderObject: return "shader-object";
    case ComputePipelineBackend::PipelineBinary: return "pipeline-binary";
    default: return "unknown";
    }
}

const char* pipelineTypeName(PipelineRecipeType type)
{
    switch (type) {
    case PipelineRecipeType::Compute: return "compute";
    case PipelineRecipeType::Graphics: return "graphics";
    case PipelineRecipeType::RayTracing: return "ray_tracing";
    default: return "unknown";
    }
}

Json spirvStatisticsJson(std::span<const uint32_t> code)
{
    const auto stats = metallic::tools::inspectSPIRV(code);
    return {{"id_bound", stats.idBound}, {"instruction_count", stats.instructions},
        {"function_count", stats.functions}, {"function_call_count", stats.functionCalls},
        {"basic_block_count", stats.basicBlocks}, {"loop_merge_count", stats.loopMerges},
        {"selection_merge_count", stats.selectionMerges}, {"branch_count", stats.branches},
        {"conditional_branch_count", stats.conditionalBranches}, {"switch_count", stats.switches},
        {"phi_count", stats.phiNodes}, {"max_function_instructions", stats.maxFunctionInstructions},
        {"max_function_basic_blocks", stats.maxFunctionBasicBlocks},
        {"largest_function_instruction_counts", stats.largestFunctionInstructions}};
}

Json requestJson(const ShaderRequest& request, size_t index)
{
    return {{"index", index}, {"module", request.module}, {"entry", request.entry},
        {"defines", request.defines}, {"capabilities", request.capabilities},
        {"search_path", request.searchPath}, {"additional_search_paths", request.searchPaths},
        {"profile", request.profile}, {"catalog_descriptor_mode", static_cast<int>(request.descriptorHeapMode)}};
}

std::string hashSpirv(std::span<const uint32_t> code)
{
    uint64_t hash = 14695981039346656037ull;
    for (uint32_t word : code) {
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            hash = (hash ^ ((word >> shift) & 255)) * 1099511628211ull;
        }
    }
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

Json deviceShaderJson(ShaderModule& module)
{
    const auto code = vulkan::nativeShaderSpirv(module);
    return {{"device_spirv_bytes", code.size_bytes()}, {"device_spirv_fnv1a64", hashSpirv(code)},
        {"device_spirv_statistics", spirvStatisticsJson(code)}, {"device_shader_content_hash", module.contentHash()}};
}

// Slang compiles one entry per request. Read its actual SPIR-V stage/name;
// graphics stages have no complete PSO state in the warmup catalog.
std::pair<uint32_t, std::string> entryPoint(std::span<const uint32_t> code)
{
    if (code.size() < 5 || code[0] != 0x07230203) { throw std::runtime_error("Invalid SPIR-V header"); }
    for (size_t offset = 5; offset < code.size();) {
        const uint32_t count = code[offset] >> 16;
        if (!count || count > code.size() - offset) { throw std::runtime_error("Invalid SPIR-V instruction"); }
        if ((code[offset] & 65535) == 15 && count >= 4) {
            std::string name;
            for (size_t i = offset + 3; i < offset + count; ++i) {
                for (uint32_t shift = 0; shift < 32; shift += 8) {
                    const char c = char((code[i] >> shift) & 255);
                    if (!c) { return {code[offset + 1], name}; }
                    name += c;
                }
            }
            throw std::runtime_error("Invalid SPIR-V entry point name");
        }
        offset += count;
    }
    throw std::runtime_error("No SPIR-V entry point");
}

Json summarize(const Json& samples)
{
    // Catalog index identifies the full variant, including ordered macros/paths.
    std::map<size_t, Json> groups;
    for (const auto& sample : samples) {
        const size_t index = sample.at("request").at("index");
        auto& group = groups[index];
        if (group.is_null()) {
            group = {{"request", sample.at("request")}, {"successful_samples", 0},
                {"failed_samples", 0}, {"pipelines_created", 0}, {"pipelines_skipped", 0}, {"values", Json::object()}};
        }
        if (sample.at("status") != "passed") { group["failed_samples"] = group["failed_samples"].get<size_t>() + 1; continue; }
        group["successful_samples"] = group["successful_samples"].get<size_t>() + 1;
        for (const char* metric : {"compile_wall_ms", "shader_module_wall_ms", "pipeline_wall_ms"}) {
            if (sample.contains(metric)) { group["values"][metric].push_back(sample.at(metric)); }
        }
        if (sample.contains("pipeline")) {
            group["pipelines_created"] = group["pipelines_created"].get<size_t>() + 1;
            group["values"]["native_create_ms"].push_back(sample.at("pipeline").at("native_create_ms"));
            group["values"]["cache_lock_wait_ms"].push_back(sample.at("pipeline").at("cache_lock_wait_ms"));
            if (!sample.at("pipeline").at("driver_ms").is_null()) {
                group["values"]["driver_ms"].push_back(sample.at("pipeline").at("driver_ms"));
            }
        } else { group["pipelines_skipped"] = group["pipelines_skipped"].get<size_t>() + 1; }
    }
    Json result = Json::array();
    for (auto& [index, group] : groups) {
        for (auto& [metric, values] : group["values"].items()) {
            auto sorted = values.get<std::vector<double>>();
            std::sort(sorted.begin(), sorted.end());
            const size_t middle = sorted.size() / 2;
            const double median = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) * 0.5;
            group["timings"][metric] = {{"count", sorted.size()}, {"min", sorted.front()}, {"median", median},
                {"p95", sorted[static_cast<size_t>(std::ceil(sorted.size() * 0.95)) - 1]}, {"max", sorted.back()}};
        }
        group.erase("values");
        result.push_back(std::move(group));
    }
    return result;
}

template<typename T>
T require(Result<T> result, const char* operation)
{
    if (!result) { throw std::runtime_error(std::string(operation) + ": " + resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

Json metricsJson(const PipelineCreationStats& stats)
{
    return {{"native_cache_hit", stats.feedbackValid ? Json(stats.nativeCacheHit) : Json(nullptr)},
        {"feedback_valid", stats.feedbackValid}, {"native_create_ms", double(stats.nativeCreateTimeNanoseconds) / 1e6},
        {"driver_ms", stats.feedbackValid ? Json(double(stats.driverDurationNanoseconds) / 1e6) : Json(nullptr)},
        {"cache_lock_wait_ms", double(stats.cacheLockWaitTimeNanoseconds) / 1e6},
        {"native_object_reused", stats.nativeObjectReused}, {"compute_backend", computeBackendName(stats.computeBackend)},
        {"native_binary_cache_hit", stats.nativeBinaryCacheHit}, {"native_binary_persisted", stats.nativeBinaryPersisted},
        {"native_binary_driver_rejected", stats.nativeBinaryDriverRejected}, {"native_binary_data_bytes", stats.nativeBinaryDataSize},
        {"native_binary_import_ms", double(stats.nativeBinaryImportTimeNanoseconds) / 1e6},
        {"native_binary_export_ms", double(stats.nativeBinaryExportTimeNanoseconds) / 1e6}};
}

Json compilationMetricsJson(const ShaderCompileMetrics& metrics)
{
    return {{"total_ms", double(metrics.totalTimeNanoseconds) / 1e6},
        {"cache_lookup_ms", double(metrics.cacheLookupTimeNanoseconds) / 1e6},
        {"source_epoch_acquire_ms", double(metrics.sourceEpochAcquireTimeNanoseconds) / 1e6},
        {"source_epoch_capture_ms", double(metrics.sourceEpochCaptureTimeNanoseconds) / 1e6},
        {"source_epoch_reused", metrics.sourceEpochReused},
        {"producer_acquire_ms", double(metrics.producerAcquireTimeNanoseconds) / 1e6},
        {"producer_validation_ms", double(metrics.producerValidationTimeNanoseconds) / 1e6},
        {"source_validation_ms", double(metrics.sourceValidationTimeNanoseconds) / 1e6},
        {"source_notification_validation_ms", double(metrics.sourceNotificationValidationTimeNanoseconds) / 1e6},
        {"compiler_setup_ms", double(metrics.compilerSetupTimeNanoseconds) / 1e6},
        {"module_load_ms", double(metrics.moduleLoadTimeNanoseconds) / 1e6},
        {"dependency_tracking_ms", double(metrics.dependencyTrackingTimeNanoseconds) / 1e6},
        {"entry_point_composition_ms", double(metrics.entryPointCompositionTimeNanoseconds) / 1e6},
        {"link_ms", double(metrics.linkTimeNanoseconds) / 1e6},
        {"code_generation_ms", double(metrics.codeGenerationTimeNanoseconds) / 1e6},
        {"target_finalize_ms", double(metrics.targetFinalizeTimeNanoseconds) / 1e6},
        {"postprocess_ms", double(metrics.postprocessTimeNanoseconds) / 1e6},
        {"raw_spirv_bytes", metrics.rawSpirvByteSize},
        {"output_spirv_bytes", metrics.outputSpirvByteSize},
        {"dependency_validation_ms", double(metrics.dependencyValidationTimeNanoseconds) / 1e6},
        {"cache_write_ms", double(metrics.cacheWriteTimeNanoseconds) / 1e6},
        {"compiler_reported_total_ms", double(metrics.compilerReportedTotalTimeNanoseconds) / 1e6},
        {"compiler_reported_downstream_ms", double(metrics.compilerReportedDownstreamTimeNanoseconds) / 1e6},
        {"cache_hit", metrics.cacheHit}, {"compiler_executed", metrics.compilerExecuted},
        {"detailed_compiler_perf_requested", metrics.detailedCompilerPerfRequested},
        {"compiler_reported_timing_available", metrics.compilerReportedTimingAvailable}};
}

template<typename T>
Json pipelineJson(const T& pipeline)
{
    auto result = metricsJson(pipeline.creationStats());
    result["pso_hash"] = pipeline.psoHash();
    result["metadata_hit"] = pipeline.pipelineCacheHit();
    return result;
}

struct PreparedPipeline {
    size_t sampleIndex;
    std::unique_ptr<ShaderModule> module;
    std::string entry;
    const PipelineRecipe* recipe = nullptr;
    std::array<std::unique_ptr<ShaderModule>, 4> graphicsModules;
    std::array<std::unique_ptr<ShaderModule>, 4> rayTracingModules;
    bool disableOptimization = false;
    ComputePipelineBackend backend = ComputePipelineBackend::Pipeline;
    std::string nativeBinaryCacheDirectory;
};

void createPreparedPipeline(Device& device, PipelineCache* cache, const PreparedPipeline& item, Json& result)
{
    if (item.recipe) {
        uint64_t capturedHash = 0;
        switch (item.recipe->type) {
        case PipelineRecipeType::Compute:
            capturedHash = detail::computePipelineStateHash(item.recipe->computeDesc(item.module.get()));
            break;
        case PipelineRecipeType::Graphics:
            capturedHash = detail::graphicsPipelineStateHash(item.recipe->graphicsDesc(item.graphicsModules[0].get(),
                item.graphicsModules[1].get(), item.graphicsModules[2].get(), item.graphicsModules[3].get()));
            break;
        case PipelineRecipeType::RayTracing:
            capturedHash = detail::rayTracingPipelineStateHash(item.recipe->rayTracingDesc(item.rayTracingModules[0].get(),
                item.rayTracingModules[1].get(), item.rayTracingModules[2].get(), item.rayTracingModules[3].get()));
            break;
        }
        result["recipe_baseline_pso_hash"] = capturedHash;
        if (capturedHash != item.recipe->expectedPsoHash) {
            throw std::runtime_error("Captured recipe baseline PSO hash differs on this device before experiment overrides");
        }
    }
    const auto begin = Clock::now();
    PipelineCreationStats metrics;
    if (item.recipe && item.recipe->type == PipelineRecipeType::RayTracing) {
        auto desc = item.recipe->rayTracingDesc(item.rayTracingModules[0].get(),
            item.rayTracingModules[1].get(), item.rayTracingModules[2].get(), item.rayTracingModules[3].get(), cache);
        desc.disableOptimization |= item.disableOptimization;
        result["disable_driver_optimization"] = desc.disableOptimization;
        result["native_create_scope"] = "vkCreateRayTracingPipelinesKHR only; excludes shader modules, layout, SBT allocation/upload and cache I/O";
        result["pipeline_wall_scope"] = "Device::createRayTracingPipeline including pipeline layout, native creation and SBT construction";
        result["ray_tracing_invocation_reorder"] = desc.enableInvocationReorder;
        auto created = device.createRayTracingPipeline(desc, &metrics);
        result["pipeline_wall_ms"] = milliseconds(begin);
        result["pipeline_attempt"] = metricsJson(metrics);
        auto pipeline = require(std::move(created), "create ray tracing pipeline");
        result["pipeline"] = pipelineJson(*pipeline);
    } else if (item.recipe && item.recipe->type == PipelineRecipeType::Graphics) {
        auto desc = item.recipe->graphicsDesc(item.graphicsModules[0].get(),
                item.graphicsModules[1].get(), item.graphicsModules[2].get(), item.graphicsModules[3].get(), cache);
        desc.disableOptimization |= item.disableOptimization;
        result["disable_driver_optimization"] = desc.disableOptimization;
        auto created = device.createGraphicsPipeline(desc, &metrics);
        result["pipeline_wall_ms"] = milliseconds(begin);
        result["pipeline_attempt"] = metricsJson(metrics);
        auto pipeline = require(std::move(created), "create graphics pipeline");
        result["pipeline"] = pipelineJson(*pipeline);
    } else {
        auto desc = item.recipe ? item.recipe->computeDesc(item.module.get(), cache) :
            ComputePipelineDesc{.computeShader = {item.module.get(), item.entry.c_str()},
                .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(GPUBufferSpan), .pipelineCache = cache};
        desc.disableOptimization |= item.disableOptimization;
        result["disable_driver_optimization"] = desc.disableOptimization;
        desc.backend = item.backend;
        desc.nativeBinaryCacheDirectory = item.nativeBinaryCacheDirectory.empty() ? nullptr : item.nativeBinaryCacheDirectory.c_str();
        if (item.backend != ComputePipelineBackend::Pipeline) { desc.pipelineCache = nullptr; }
        auto created = device.createComputePipeline(desc, &metrics);
        result["pipeline_wall_ms"] = milliseconds(begin);
        result["pipeline_attempt"] = metricsJson(metrics);
        auto pipeline = require(std::move(created), "create compute pipeline");
        result["pipeline"] = pipelineJson(*pipeline);
    }
    result["pipeline_load_and_create_wall_ms"] = result.value("pipeline_cache_preparation_wall_ms", 0.0) +
        result.at("pipeline_wall_ms").get<double>();
    if (item.recipe) {
        result["pipeline"]["expected_pso_hash"] = item.recipe->expectedPsoHash;
        result["pipeline"]["recipe_hash_matches"] = result["pipeline"].at("pso_hash") == item.recipe->expectedPsoHash;
        result["pipeline"]["recipe_state_overridden"] = (item.disableOptimization && !item.recipe->disableOptimization) ||
            (item.recipe->type == PipelineRecipeType::Compute && item.backend != item.recipe->computeBackend);
        if (!result["pipeline"]["recipe_hash_matches"].get<bool>() && !result["pipeline"]["recipe_state_overridden"].get<bool>()) {
            throw std::runtime_error("Replayed PSO hash differs from captured recipe; device features or backend state changed");
        }
    }
    result["pipeline_status"] = "created";
    result["status"] = "passed";
}

// Each worker owns one result slot at a time; the report is updated only after
// every worker has joined. Compilation and module/cache creation precede this.
Json createPipelineBatch(Device& device, const std::vector<PipelineCache*>& caches,
    const std::vector<PreparedPipeline>& prepared, uint32_t requestedWorkers,
    const std::string& mode, Json& samples, size_t& failures)
{
    const uint32_t workerCount = static_cast<uint32_t>(std::min<size_t>(requestedWorkers, prepared.size()));
    std::vector<Json> results(prepared.size());
    std::barrier start(static_cast<std::ptrdiff_t>(workerCount + 1));
    std::vector<std::jthread> workers;
    workers.reserve(workerCount);
    try {
        for (uint32_t worker = 0; worker < workerCount; ++worker) {
            workers.emplace_back([&, worker] {
                start.arrive_and_wait();
                // Stable striding keeps each shard's workload identical across rounds.
                auto* cache = caches[caches.size() == 1 ? 0 : worker];
                for (size_t i = worker; i < prepared.size(); i += workerCount) {
                    const auto& item = prepared[i];
                    auto& result = results[i];
                    result["pipeline_cache_preparation_wall_ms"] = samples[item.sampleIndex].value("pipeline_cache_preparation_wall_ms", 0.0);
                    const auto begin = Clock::now();
                    try {
                        createPreparedPipeline(device, cache, item, result);
                    } catch (const std::exception& error) {
                        result["pipeline_wall_ms"] = milliseconds(begin);
                        result["failed_phase"] = "creating_pipeline";
                        result["status"] = "failed";
                        result["error"] = error.what();
                    }
                    result["pipeline_worker"] = worker;
                    result["pipeline_cache_shard"] = caches.size() == 1 ? 0 : worker;
                }
            });
        }
    } catch (...) {
        // Release already launched threads if the host cannot launch the full
        // team; jthread then drains them before the exception escapes.
        for (size_t i = workers.size(); i < workerCount; ++i) { start.arrive_and_drop(); }
        start.arrive_and_wait();
        throw;
    }
    const auto begin = Clock::now();
    start.arrive_and_wait();
    for (auto& worker : workers) { worker.join(); }
    const double elapsed = milliseconds(begin);
    double wallSum = 0, nativeSum = 0, waitSum = 0, attemptedNativeSum = 0, attemptedWaitSum = 0, driverSum = 0;
    size_t validDriverSamples = 0;
    size_t successful = 0, metadataHits = 0, nativeHits = 0, nativeMisses = 0, unknown = 0;
    std::vector<double> wallTimes, nativeTimes, waitTimes;
    Json sampleIndices = Json::array();
    for (size_t i = 0; i < prepared.size(); ++i) {
        auto& sample = samples[prepared[i].sampleIndex];
        sample.update(results[i]);
        sampleIndices.push_back(prepared[i].sampleIndex);
        wallSum += sample.at("pipeline_wall_ms").get<double>();
        wallTimes.push_back(sample.at("pipeline_wall_ms").get<double>());
        if (sample.contains("pipeline_attempt")) {
            const auto& attempt = sample.at("pipeline_attempt");
            attemptedNativeSum += attempt.at("native_create_ms").get<double>();
            attemptedWaitSum += attempt.at("cache_lock_wait_ms").get<double>();
            if (!attempt.at("driver_ms").is_null()) {
                driverSum += attempt.at("driver_ms").get<double>();
                ++validDriverSamples;
            }
        }
        if (sample.at("status") == "passed") {
            ++successful;
            nativeSum += sample.at("pipeline").at("native_create_ms").get<double>();
            waitSum += sample.at("pipeline").at("cache_lock_wait_ms").get<double>();
            nativeTimes.push_back(sample.at("pipeline").at("native_create_ms").get<double>());
            waitTimes.push_back(sample.at("pipeline").at("cache_lock_wait_ms").get<double>());
            metadataHits += sample.at("pipeline").at("metadata_hit").get<bool>();
            if (!sample.at("pipeline").at("feedback_valid").get<bool>()) { ++unknown; }
            else if (sample.at("pipeline").at("native_cache_hit").get<bool>()) { ++nativeHits; }
            else { ++nativeMisses; }
        } else { ++failures; }
    }
    const auto p95 = [](std::vector<double>& values) -> Json {
        if (values.empty()) { return nullptr; }
        std::sort(values.begin(), values.end());
        return values[static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1];
    };
    return {{"mode", mode}, {"cache_count", caches.size()},
        {"requested_workers", requestedWorkers}, {"actual_workers", workerCount},
        {"sample_indices", sampleIndices}, {"pipeline_count", prepared.size()}, {"successful_pipelines", successful},
        {"pipeline_batch_wall_ms", elapsed}, {"pipeline_wall_sum_ms", wallSum},
        {"successful_native_create_sum_ms", nativeSum}, {"successful_cache_lock_wait_sum_ms", waitSum},
        {"attempted_native_create_sum_ms", attemptedNativeSum}, {"attempted_cache_lock_wait_sum_ms", attemptedWaitSum},
        {"valid_driver_feedback_count", validDriverSamples},
        {"driver_duration_sum_ms", validDriverSamples ? Json(driverSum) : Json(nullptr)},
        {"pipeline_wall_p95_ms", p95(wallTimes)}, {"native_create_p95_ms", p95(nativeTimes)},
        {"cache_lock_wait_p95_ms", p95(waitTimes)},
        {"throughput_pipelines_per_second", elapsed > 0 ? Json(successful * 1000.0 / elapsed) : Json(nullptr)},
        {"metadata_hits", metadataHits}, {"metadata_misses", successful - metadataHits},
        {"native_hits", nativeHits}, {"native_misses", nativeMisses}, {"unknown_feedback", unknown}};
}

int run(const Options& options)
{
    std::optional<PipelineRecipe> recipe;
    if (!options.recipeFile.empty()) {
        std::string log;
        auto loaded = readPipelineRecipe(std::filesystem::absolute(options.recipeFile), log);
        if (!loaded) { throw std::invalid_argument("Cannot read pipeline recipe: " + log); }
        recipe = std::move(*loaded);
    }
    const auto backend = recipe && recipe->type == PipelineRecipeType::Compute && !options.explicitComputeBackend ? recipe->computeBackend :
        options.computeBackend == "shader-object" ? ComputePipelineBackend::ShaderObject :
        options.computeBackend == "pipeline-binary" ? ComputePipelineBackend::PipelineBinary : ComputePipelineBackend::Pipeline;
    if (recipe && recipe->type != PipelineRecipeType::Compute && backend != ComputePipelineBackend::Pipeline) {
        throw std::invalid_argument("Experimental compute backend requires a compute recipe");
    }
    const auto effectivePostprocess = slangShaderPostprocessMode(
        SlangShaderDesc{.postprocessMode = postprocessMode(options.spirvPostprocess)});
    if (backend == ComputePipelineBackend::Pipeline && !options.nativeBinaryCacheDirectory.empty()) {
        throw std::invalid_argument("--native-binary-cache-dir requires shader-object or pipeline-binary backend");
    }
    if (backend == ComputePipelineBackend::ShaderObject && (options.disableDriverOptimization || (recipe && recipe->disableOptimization))) {
        throw std::invalid_argument("Shader Object has no DISABLE_OPTIMIZATION pipeline flag");
    }
    if (backend != ComputePipelineBackend::Pipeline && (options.pipelineCache != "empty" || options.mergeShards || options.cacheLockScope != "metadata")) {
        throw std::invalid_argument("Experimental compute recipe does not use VkPipelineCache or merge/lock controls");
    }
    if (backend != ComputePipelineBackend::Pipeline && options.mode != "serial") {
        throw std::invalid_argument("Experimental compute backends require --mode serial; VkPipelineCache concurrency modes use the pipeline backend");
    }
    auto catalog = metallic::tools::shaderWarmupRequests();
    // Manual experiment catalog only: normal application warmup does not pay
    // for the opt-in split's additional modules.
    const std::string splitRtxcrInclude = METALLIC_RTXCR_SHADER_INCLUDE_DIR;
    const SceneShaderOptions splitOptions{.hasRTXCR = !splitRtxcrInclude.empty(), .rtxcrInclude = splitRtxcrInclude};
    const SlangMacroDefine splitDefines[]{{"METALLIC_PT_SINGLE_SAMPLE", "1"}};
    for (bool primaryStage : {true, false}) {
        catalog.push_back(makeOpenPBRPrimarySplitShaderRequest(splitOptions, splitDefines, primaryStage));
    }
    std::vector<size_t> selected;
    for (size_t i = 0; i < catalog.size(); ++i) {
        if (options.requestIndex && i != *options.requestIndex) { continue; }
        if (!options.filter.empty() && (catalog[i].module + "." + catalog[i].entry).find(options.filter) == std::string::npos) { continue; }
        selected.push_back(i);
    }
    if (recipe) { selected = {0}; }
    if (selected.empty()) { throw std::invalid_argument("No shader requests matched"); }
    if (options.list) {
        if (recipe) {
            std::cout << Json{{"recipe_file", options.recipeFile}, {"expected_pso_hash", recipe->expectedPsoHash},
                {"pipeline_type", pipelineTypeName(recipe->type)}}.dump() << '\n';
            return 0;
        }
        for (size_t i : selected) { std::cout << requestJson(catalog[i], i).dump() << '\n'; }
        return 0;
    }
    const auto output = std::filesystem::absolute(options.outputDirectory);
    const auto reportPath = output / "report.json";
    if (std::filesystem::exists(reportPath)) { throw std::invalid_argument("Report already exists; choose a new --output-dir"); }
    std::filesystem::create_directories(output);
    const auto cache = options.cacheDirectory.empty() ? output / "cache" : std::filesystem::absolute(options.cacheDirectory);
    const std::string shaderCache = (cache / "spirv").string();
    const std::string pipelineCache = options.pipelineCacheFile.empty() ? (cache / "pipeline.pso").string() :
        std::filesystem::absolute(options.pipelineCacheFile).string();
    Json report{{"schema_version", 5}, {"status", "initializing"}, {"samples", Json::array()}, {"pipeline_batches", Json::array()},
        {"pipeline_cache_preparation_wall_ms", 0.0}, {"pipeline_cache_preparations", Json::array()},
        {"pipeline_cache_preparation_scope", "createPipelineCache wall time includes file loading and native cache creation. Report totals count each cache once; concurrent batch preparation belongs to the batch. Load-and-create fields sum preparation and creation segments, excluding device/shader module creation and cache save."},
        {"build_configuration", METALLIC_BENCH_BUILD_CONFIGURATION},
        {"slang_build", spGetBuildTagString()},
        {"shader_target", {{"id", vulkanSpirvTarget().id}, {"revision", vulkanSpirvTarget().revision}}},
        {"configuration", {{"filter", options.filter}, {"iterations", options.iterations},
            {"pipeline_workers", options.pipelineWorkers.value_or(1)}, {"staged_pipeline_batches", options.pipelineWorkers.has_value()},
            {"mode", options.mode}, {"workers", options.pipelineWorkers.value_or(1)},
            {"pipeline_replicas", options.pipelineReplicas}, {"cache_lock_scope", options.cacheLockScope},
            {"merge_shards", options.mergeShards}, {"recipe_file", options.recipeFile},
            {"shader_cache", options.shaderCache}, {"pipeline_cache", options.pipelineCache},
            {"pipeline_cache_file", pipelineCache},
            {"descriptor_mode", options.descriptorMode}, {"debug_mode", options.debugMode},
            {"slang_optimization", options.slangOptimization},
            {"spirv_postprocess", options.spirvPostprocess},
            {"spirv_postprocess_tool_version", slangShaderPostprocessToolVersion(effectivePostprocess)},
            {"slang_perf_diagnostics", options.slangPerfDiagnostics},
            {"requested_disable_driver_optimization", options.disableDriverOptimization},
            {"disable_driver_optimization", options.disableDriverOptimization || (recipe && recipe->disableOptimization)},
            {"compute_backend", computeBackendName(backend)}, {"native_binary_cache_directory", options.nativeBinaryCacheDirectory},
            {"application_pipeline_cache_used", backend == ComputePipelineBackend::Pipeline && !options.compileOnly},
            {"compile_only", options.compileOnly}, {"validation", options.validation},
            {"cache_directory", cache.string()}, {"selected_requests", selected}}},
        {"scope", "Serial preparation and optional concurrent PSO creation; no dispatch or frame timing. Empty application caches do not guarantee cold driver/OS caches."}};
    for (const char* name : {"METALLIC_SLANG_DESCRIPTOR_MODE", "METALLIC_WORKING_COLOR_SPACE",
        "METALLIC_VK_INTERNAL_PIPELINE_CACHE", "METALLIC_VK_LOG_PIPELINE_KEYS", "METALLIC_SLANG_OPTIMIZATION", "METALLIC_SPIRV_POSTPROCESS",
        "METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION", "METALLIC_VK_PIPELINE_BINARY",
        "METALLIC_VK_PIPELINE_STATISTICS", "METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY"}) {
        const char* value = std::getenv(name);
        report["environment"][name] = value ? Json(value) : Json(nullptr);
    }
    ValidationCapture validation;
    const auto save = [&] {
        validation.snapshot(report);
        std::ofstream stream(reportPath, std::ios::trunc);
        stream << report.dump(2) << '\n';
        if (!stream) { throw std::runtime_error("Cannot write " + reportPath.string()); }
    };
    save();
    bool videoInitialized = false;
    std::unique_ptr<Device> device;
    std::unique_ptr<PipelineCache> reusedPipelineCache;
    std::vector<std::unique_ptr<PipelineCache>> reusedShardCaches;
    std::unique_ptr<PipelineCache> mergedPipelineCache;
    DescriptorHeapShaderStrides nativeStrides;
    size_t failures = 0;
    std::vector<PreparedPipeline> prepared;
    const auto configurePrepared = [&](PreparedPipeline& item) {
        item.disableOptimization = options.disableDriverOptimization;
        item.backend = backend;
        item.nativeBinaryCacheDirectory = options.nativeBinaryCacheDirectory;
    };
    const auto createMeasuredCache = [&](const PipelineCacheDesc& desc, const char* operation, Json* sample = nullptr) {
        const auto begin = Clock::now();
        auto created = device->createPipelineCache(desc);
        const double elapsed = milliseconds(begin);
        report["pipeline_cache_preparation_wall_ms"] = report["pipeline_cache_preparation_wall_ms"].get<double>() + elapsed;
        report["pipeline_cache_preparations"].push_back({{"operation", operation},
            {"file", desc.filePath ? desc.filePath : ""}, {"wall_ms", elapsed}});
        if (sample) { (*sample)["pipeline_cache_preparation_wall_ms"] = elapsed; }
        auto preparedCache = require(std::move(created), operation);
        report["pipeline_cache_preparations"].back()["load_status"] = static_cast<int>(preparedCache->stats().loadStatus);
        return preparedCache;
    };
    uint32_t batchIteration = 0;
    const auto flushBatch = [&] {
        if (prepared.empty()) { return; }
        const uint32_t workerCount = static_cast<uint32_t>(std::min<size_t>(*options.pipelineWorkers, prepared.size()));
        const uint32_t cacheCount = options.mode == "parallel-sharded" ? workerCount : 1;
        std::vector<std::unique_ptr<PipelineCache>> emptyCaches;
        std::vector<PipelineCache*> activeCaches;
        const double initialCachePreparation = report["pipeline_cache_preparation_wall_ms"].get<double>();
        for (const auto& item : prepared) { report["samples"][item.sampleIndex]["status"] = "creating_pipeline_cache"; }
        save();
        for (uint32_t shard = 0; shard < cacheCount; ++shard) {
            if (backend != ComputePipelineBackend::Pipeline) {
                activeCaches.push_back(nullptr);
            } else if (options.pipelineCache == "empty") {
                emptyCaches.push_back(createMeasuredCache({.saveOnDestroy = false,
                    .serializeCreation = options.cacheLockScope == "creation"}, "create empty batch cache"));
                activeCaches.push_back(emptyCaches.back().get());
            } else if (options.mode == "parallel-sharded") {
                if (shard >= reusedShardCaches.size()) {
                    const auto path = (cache / ("pipeline-shard-" + std::to_string(shard) + ".pso")).string();
                    reusedShardCaches.push_back(createMeasuredCache({.filePath = path.c_str(),
                        .saveOnDestroy = false, .serializeCreation = options.cacheLockScope == "creation"}, "create reusable shard cache"));
                    report["pipeline_cache_shard_load_status"].push_back(static_cast<int>(reusedShardCaches.back()->stats().loadStatus));
                }
                activeCaches.push_back(reusedShardCaches[shard].get());
            } else { activeCaches.push_back(reusedPipelineCache.get()); }
        }
        for (const auto& item : prepared) {
            auto& sample = report["samples"][item.sampleIndex];
            sample["status"] = "creating_pipeline";
            sample["pipeline_batch"] = report["pipeline_batches"].size();
        }
        save();
        const uint32_t initialErrors = validation.errors.load();
        auto batch = createPipelineBatch(*device, activeCaches, prepared, *options.pipelineWorkers, options.mode, report["samples"], failures);
        batch["pipeline_cache_preparation_wall_ms"] = report["pipeline_cache_preparation_wall_ms"].get<double>() - initialCachePreparation;
        batch["pipeline_load_and_create_wall_ms"] = batch["pipeline_cache_preparation_wall_ms"].get<double>() +
            batch["pipeline_batch_wall_ms"].get<double>();
        batch["iteration"] = batchIteration;
        batch["validation_errors"] = validation.errors.load() - initialErrors;
        if (options.mergeShards) {
            if (!mergedPipelineCache) {
                const auto path = (output / "pipeline-merged.pso").string();
                mergedPipelineCache = createMeasuredCache({.filePath = path.c_str(), .saveOnDestroy = false}, "create merge destination");
            }
            const auto begin = Clock::now();
            require(mergedPipelineCache->merge(activeCaches), "merge shard caches");
            batch["pipeline_cache_merge_wall_ms"] = milliseconds(begin);
            report["pipeline_cache_merge_wall_ms"] = report.value("pipeline_cache_merge_wall_ms", 0.0) + batch["pipeline_cache_merge_wall_ms"].get<double>();
        }
        report["pipeline_batches"].push_back(std::move(batch));
        prepared.clear();
        save();
    };
    const auto ensureDevice = [&](bool rayQuery) {
        if (device) { return; }
        const auto begin = Clock::now();
        if (!SDL_Init(SDL_INIT_VIDEO)) { throw std::runtime_error(std::string("Initialize SDL video: ") + SDL_GetError()); }
        videoInitialized = true;
        DeviceDesc desc{.applicationName = "MetallicShaderPipelineBench",
            .enableValidation = options.validation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = rayQuery, .enableRayQuery = rayQuery,
            .enableClusterAccelerationStructure = rayQuery,
            .validationSink = {.callback = ValidationCapture::callback, .context = &validation}};
        desc.enablePipelineBinary = backend == ComputePipelineBackend::PipelineBinary;
        if (recipe) {
            const auto& caps = recipe->sourceDevice.capabilities;
            desc.enableBindlessDescriptorHeap = caps.bindlessDescriptorHeap;
            desc.enableMeshShader = caps.meshShader;
            desc.enableTaskShader = caps.taskShader;
            desc.enableTaskShaderSubgroupBallot = caps.taskShaderSubgroupBallot;
            desc.enableGeometryShader = caps.geometryShader;
            desc.enableSubgroupSizeControl = caps.subgroupSizeControl;
            desc.enableComputeFullSubgroups = caps.computeFullSubgroups;
            desc.enableRayTracingAccelerationStructure = caps.rayTracingAccelerationStructure;
            desc.enableRayQuery = caps.rayQuery;
            desc.enableRayTracingPipeline = caps.rayTracingPipeline;
            desc.enableRayTracingInvocationReorder = caps.rayTracingInvocationReorder;
            desc.enableClusterAccelerationStructure = caps.clusterAccelerationStructure;
            desc.enablePartitionedAccelerationStructure = caps.partitionedAccelerationStructure;
            desc.enableDeviceGeneratedCommands = caps.deviceGeneratedCommands;
        }
        device = require(createDevice(desc), "create device");
        report["device_initialization_wall_ms"] = milliseconds(begin);
        const auto vk = vulkan::nativeDevice(*device);
        const auto& properties = *vk.properties;
        std::ostringstream uuid;
        for (const auto byte : properties.core.pipelineCacheUUID) {
            uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
        }
        report["device"] = {{"name", properties.core.deviceName}, {"vendor_id", properties.core.vendorID},
            {"device_id", properties.core.deviceID}, {"api_version", properties.core.apiVersion},
            {"driver_version", properties.core.driverVersion}, {"driver_name", properties.driver.driverName},
            {"driver_info", properties.driver.driverInfo}, {"pipeline_cache_uuid", uuid.str()},
            {"validation_active", vk.validationMessengerActive}};
        report["device"]["ray_tracing_invocation_reorder"] = device->capabilities().rayTracingInvocationReorder;
        report["device"]["ray_tracing_invocation_reorder_reordering_hint"] =
            device->capabilities().rayTracingInvocationReorderReorderingHint;
        if (recipe) {
            const auto& source = recipe->sourceDevice;
            report["recipe_source_device"] = {{"name", source.deviceName}, {"vendor_id", source.vendorId},
                {"device_id", source.deviceId}, {"driver_version", source.driverVersion}, {"api_version", source.apiVersion},
                {"driver_name", source.driverName}, {"pipeline_cache_uuid", source.pipelineCacheUUID},
                {"validation", source.validationEnabled}, {"synchronization_validation", source.synchronizationValidationEnabled}};
            report["recipe_source_device_matches"] = source.vendorId == properties.core.vendorID &&
                source.deviceId == properties.core.deviceID && source.driverVersion == properties.core.driverVersion;
        }
    };
    const auto ensureReusedCache = [&](Json* sample = nullptr) {
        if (options.pipelineCache != "reuse" || options.mode == "parallel-sharded" || reusedPipelineCache) { return; }
        reusedPipelineCache = createMeasuredCache({.filePath = pipelineCache.c_str(), .saveOnDestroy = false,
            .serializeCreation = options.cacheLockScope == "creation"}, "create reusable cache", sample);
        report["pipeline_cache_load_status"] = static_cast<int>(reusedPipelineCache->stats().loadStatus);
    };
    setSlangShaderDebugMode(options.debugMode == "capture" ? SlangShaderDebugMode::CaptureSymbols :
        options.debugMode == "debug" ? SlangShaderDebugMode::ShaderDebug : SlangShaderDebugMode::Disabled);
    try {
        bool rayQuery = false;
        for (size_t i : selected) {
            rayQuery |= std::find(catalog[i].capabilities.begin(), catalog[i].capabilities.end(), "spvRayQueryKHR") != catalog[i].capabilities.end();
        }
        std::vector<std::tuple<size_t, uint32_t, uint32_t>> work;
        if (options.pipelineWorkers) {
            for (uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
                for (size_t index : selected) {
                    for (uint32_t replica = 0; replica < options.pipelineReplicas; ++replica) { work.emplace_back(index, iteration, replica); }
                }
            }
        } else {
            for (size_t index : selected) {
                for (uint32_t iteration = 0; iteration < options.iterations; ++iteration) { work.emplace_back(index, iteration, 0); }
            }
        }
        for (const auto& [index, iteration, replicaIndex] : work) {
            if (options.pipelineWorkers && iteration != batchIteration) {
                flushBatch();
                batchIteration = iteration;
            }
            if (recipe) {
                const bool graphics = recipe->type == PipelineRecipeType::Graphics;
                const bool rayTracing = recipe->type == PipelineRecipeType::RayTracing;
                const auto& mainShader = rayTracing ? recipe->rayGenerationShader :
                    graphics ? (!recipe->meshShader.spirv.empty() ? recipe->meshShader : recipe->vertexShader) : recipe->computeShader;
                report["samples"].push_back({{"request", {{"index", 0}, {"module", mainShader.moduleName},
                    {"entry", mainShader.sourceEntryPoint}, {"recipe_file", options.recipeFile}}},
                    {"iteration", iteration}, {"pipeline_replica", replicaIndex}, {"pipeline_type", pipelineTypeName(recipe->type)},
                    {"status", "creating_device"}, {"shader_source", "recipe_spirv"}});
                const size_t sampleIndex = report["samples"].size() - 1;
                auto& sample = report["samples"].back();
                sample["pipeline_cache_preparation_wall_ms"] = 0.0;
                save();
                try {
                    ensureDevice(false);
                    ensureReusedCache(&sample);
                    PreparedPipeline item{.sampleIndex = sampleIndex, .recipe = &*recipe};
                    configurePrepared(item);
                    sample["status"] = "creating_shader_module";
                    save();
                    const auto begin = Clock::now();
                    double moduleWall = 0;
                    size_t bytes = 0;
                    sample["shader_stages"] = Json::array();
                    const auto makeModule = [&](const PipelineRecipeShader& stage, const char* name) -> std::unique_ptr<ShaderModule> {
                        if (stage.spirv.empty()) { return {}; }
                        bytes += stage.spirv.size() * sizeof(uint32_t);
                        sample["shader_stages"].push_back({{"stage", name}, {"module", stage.moduleName},
                            {"entry", stage.entryPoint}, {"source_entry", stage.sourceEntryPoint},
                            {"debug_mode", stage.debugMode}, {"descriptor_mode", stage.descriptorHeapMode},
                            {"slang_optimization", stage.optimizationLevel},
                            {"spirv_postprocess", stage.spirvPostprocess},
                            {"spirv_postprocess_tool_version", stage.spirvPostprocessToolVersion},
                            {"source_identity_ambiguous", stage.sourceIdentityAmbiguous},
                            {"spirv_bytes", stage.spirv.size() * sizeof(uint32_t)}, {"spirv_fnv1a64", hashSpirv(stage.spirv)},
                            {"spirv_statistics", spirvStatisticsJson(stage.spirv)}});
                        auto& shaderSample = sample["shader_stages"].back();
                        shaderSample["specialization_constants"] = Json::array();
                        for (const auto& value : stage.specializationConstants) {
                            shaderSample["specialization_constants"].push_back({{"id", value.id},
                                {"type", value.type == ShaderSpecializationType::Int32 ? "int32" : "bool32"},
                                {"size", 4}, {"bits", value.bits}});
                        }
                        const auto moduleBegin = Clock::now();
                        auto module = require(device->createShaderModule({.spirv = stage.spirv, .debugName = stage.moduleName.c_str()}), "create recipe shader module");
                        moduleWall += milliseconds(moduleBegin);
                        sample["shader_stages"].back().update(deviceShaderJson(*module));
                        return module;
                    };
                    if (rayTracing) {
                        item.rayTracingModules[0] = makeModule(recipe->rayGenerationShader, "ray_generation");
                        item.rayTracingModules[1] = makeModule(recipe->missShader, "miss");
                        item.rayTracingModules[2] = makeModule(recipe->closestHitShader, "closest_hit");
                        item.rayTracingModules[3] = makeModule(recipe->anyHitShader, "any_hit");
                        sample["ray_tracing_state"] = {{"uses_bindless_heap", recipe->usesBindlessHeap},
                            {"bindless_user_push_data_size", recipe->bindlessUserPushDataSize},
                            {"max_recursion_depth", recipe->maxRecursionDepth},
                            {"cluster_acceleration_structure", recipe->clusterAccelerationStructure},
                            {"disable_optimization", recipe->disableOptimization}};
                    } else if (graphics) {
                        item.graphicsModules[0] = makeModule(recipe->vertexShader, "vertex");
                        item.graphicsModules[1] = makeModule(recipe->taskShader, "task");
                        item.graphicsModules[2] = makeModule(recipe->meshShader, "mesh");
                        item.graphicsModules[3] = makeModule(recipe->fragmentShader, "fragment");
                    } else { item.module = makeModule(recipe->computeShader, "compute"); }
                    if (!graphics && !rayTracing) {
                        sample["spirv_statistics"] = spirvStatisticsJson(mainShader.spirv);
                        sample["effective_slang_optimization"] = mainShader.optimizationLevel;
                        sample["effective_spirv_postprocess"] = mainShader.spirvPostprocess;
                        sample["spirv_fnv1a64"] = hashSpirv(mainShader.spirv);
                        sample.update(deviceShaderJson(*item.module));
                    }
                    sample["spirv_bytes"] = bytes;
                    sample["shader_module_wall_ms"] = moduleWall;
                    sample["recipe_preparation_wall_ms"] = milliseconds(begin);
                    sample["status"] = "prepared";
                    prepared.push_back(std::move(item));
                } catch (const std::exception& error) {
                    sample["failed_phase"] = sample["status"];
                    sample["status"] = "failed";
                    sample["error"] = error.what();
                    ++failures;
                }
                save();
                continue;
            }
            auto request = catalog[index];
            if (options.descriptorMode != "request") {
                request.descriptorHeapMode = options.descriptorMode == "native" ? SlangDescriptorHeapMode::Native : SlangDescriptorHeapMode::Mapped;
            }
            const ShaderRequestView view(request);
            auto source = view.desc();
            source.optimizationLevel = optimizationLevel(options.slangOptimization);
            source.postprocessMode = postprocessMode(options.spirvPostprocess);
            const bool native = slangUsesNativeDescriptorHeap(source);
            if (native && !nativeStrides.resource) {
                std::string diagnostics;
                if (!queryVulkanDescriptorHeapShaderStrides(nativeStrides, diagnostics)) { throw std::runtime_error(diagnostics); }
            }
            source.descriptorHeapStrides = nativeStrides;
            report["samples"].push_back({{"request", requestJson(catalog[index], index)}, {"iteration", iteration}, {"pipeline_replica", replicaIndex},
                {"effective_descriptor_mode", native ? "native" : "mapped"},
                {"resource_heap_stride", native ? nativeStrides.resource : 0},
                {"sampler_heap_stride", native ? nativeStrides.sampler : 0}, {"status", "compiling"}});
            report["samples"].back()["effective_slang_optimization"] = slangShaderOptimizationLevelName(slangShaderOptimizationLevel(source));
            report["samples"].back()["effective_spirv_postprocess"] = slangShaderPostprocessModeName(slangShaderPostprocessMode(source));
            auto& sample = report["samples"].back();
            sample["pipeline_cache_preparation_wall_ms"] = 0.0;
            const uint32_t initialValidationErrors = validation.errors.load();
            report["status"] = "running";
            save();
            std::cout << "[ShaderPipelineBench] request=" << index << " iteration=" << iteration << ' ' << request.module << '.' << request.entry << std::endl;
            try {
                bool cacheHit = false;
                std::string diagnostics;
                auto begin = Clock::now();
                auto compiled = ShaderRegistry::instance().getShader(source,
                    {.cacheDirectory = shaderCache.c_str(), .enableDiskCache = options.shaderCache == "reuse", .outCacheHit = &cacheHit,
                        .reportDetailedCompilerPerf = options.slangPerfDiagnostics}, diagnostics);
                sample["compile_wall_ms"] = milliseconds(begin);
                sample["shader_cache_hit"] = cacheHit;
                sample["diagnostics"] = diagnostics;
                auto code = require(std::move(compiled), "compile shader");
                sample["compilation_phases"] = compilationMetricsJson(code.metrics);
                sample["spirv_bytes"] = code.spirv.size() * sizeof(uint32_t);
                sample["spirv_fnv1a64"] = hashSpirv(code.spirv);
                sample["spirv_statistics"] = spirvStatisticsJson(code.spirv);
                sample["dependencies"] = code.dependencies;
                const auto [model, entry] = entryPoint(code.spirv);
                sample["execution_model"] = model;
                sample["pipeline_type"] = model == 5 ? "compute" : "graphics";
                sample["spirv_entry"] = entry;
                if (options.compileOnly || model != 5) {
                    sample["status"] = "passed";
                    sample["pipeline_status"] = options.compileOnly ? "compile_only" : "skipped_non_compute";
                    sample["validation_errors"] = validation.errors.load() - initialValidationErrors;
                    save();
                    continue;
                }
                if (!device) {
                    sample["status"] = "creating_device";
                    save();
                    ensureDevice(rayQuery);
                }
                if (native && slangDescriptorHeapShaderStrides() != nativeStrides) {
                    throw std::runtime_error("Selected device descriptor strides differ from the native compiler adapter query");
                }
                if (options.pipelineCache == "reuse" && options.mode != "parallel-sharded" && !reusedPipelineCache) {
                    sample["status"] = "creating_pipeline_cache";
                    save();
                    ensureReusedCache(&sample);
                }
                sample["status"] = "creating_shader_module";
                save();
                begin = Clock::now();
                auto moduleResult = device->createShaderModule({.spirv = code.spirv, .debugName = request.module.c_str()});
                sample["shader_module_wall_ms"] = milliseconds(begin);
                auto module = require(std::move(moduleResult), "create shader module");
                sample.update(deviceShaderJson(*module));
                if (options.pipelineWorkers) {
                    prepared.push_back({report["samples"].size() - 1, std::move(module), entry});
                    configurePrepared(prepared.back());
                    sample["status"] = "prepared";
                    sample["preparation_validation_errors"] = validation.errors.load() - initialValidationErrors;
                    save();
                    continue;
                }
                std::unique_ptr<PipelineCache> emptyCache;
                if (backend == ComputePipelineBackend::Pipeline && options.pipelineCache == "empty") {
                    sample["status"] = "creating_pipeline_cache";
                    save();
                    emptyCache = createMeasuredCache({.saveOnDestroy = false,
                        .serializeCreation = options.cacheLockScope == "creation"}, "create empty cache", &sample);
                }
                auto* activeCache = emptyCache ? emptyCache.get() : reusedPipelineCache.get();
                sample["status"] = "creating_pipeline";
                save();
                PreparedPipeline item{report["samples"].size() - 1, std::move(module), entry};
                configurePrepared(item);
                createPreparedPipeline(*device, activeCache, item, sample);
            } catch (const std::exception& error) {
                sample["failed_phase"] = sample["status"];
                sample["status"] = "failed";
                sample["error"] = error.what();
                ++failures;
                std::cerr << error.what() << '\n';
            }
            sample["validation_errors"] = validation.errors.load() - initialValidationErrors;
            save();
        }
        if (options.pipelineWorkers) { flushBatch(); }
        if (reusedPipelineCache) {
            report["status"] = "saving_pipeline_cache";
            save();
            const auto begin = Clock::now();
            auto saved = reusedPipelineCache->save();
            report["pipeline_cache_save_wall_ms"] = milliseconds(begin);
            require(std::move(saved), "save pipeline cache");
        }
        for (const auto& shard : reusedShardCaches) {
            const auto begin = Clock::now();
            require(shard->save(), "save shard cache");
            report["pipeline_cache_save_wall_ms"] = report.value("pipeline_cache_save_wall_ms", 0.0) + milliseconds(begin);
        }
        if (mergedPipelineCache) {
            const auto begin = Clock::now();
            require(mergedPipelineCache->save(), "save merged cache");
            report["pipeline_cache_merged_save_wall_ms"] = milliseconds(begin);
            const auto stats = mergedPipelineCache->stats();
            report["merged_cache"] = {{"stored_pso_count", stats.storedPsoCount}, {"backend_data_bytes", stats.backendDataSize},
                {"file", (output / "pipeline-merged.pso").string()}};
        }
    } catch (const std::exception& error) {
        report["error"] = error.what();
        ++failures;
        std::cerr << error.what() << '\n';
    }
    prepared.clear();
    reusedPipelineCache.reset();
    reusedShardCaches.clear();
    mergedPipelineCache.reset();
    device.reset();
    if (videoInitialized) { SDL_Quit(); }
    const uint32_t validationErrors = validation.errors.load();
    report["failures"] = failures;
    report["summary"] = summarize(report["samples"]);
    size_t createdCount = 0, skippedCount = 0;
    for (const auto& sample : report["samples"]) {
        if (sample.value("pipeline_status", "") == "created") { ++createdCount; }
        else if (sample.contains("pipeline_status")) { ++skippedCount; }
    }
    report["pipelines_created"] = createdCount;
    report["pipelines_skipped"] = skippedCount;
    report["status"] = failures || validationErrors ? "failed" : "passed";
    save();
    std::cout << reportPath.string() << '\n';
    return failures || validationErrors ? 1 : 0;
}
} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    // This process is an unattended timing harness. Let native failures return
    // their exit code so runners preserve evidence without modal OS dialogs.
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "MetallicShaderPipelineBench [--list] [--filter substring | --request-index N | --all]\n"
            "  --output-dir path [--iterations N] [--compile-only] [--validation]\n"
            "  [--mode serial|parallel-shared|parallel-sharded] [--workers N] [--pipeline-replicas N]\n"
            "  [--cache-lock-scope metadata|creation] [--merge-shards] [--recipe-file path]\n"
            "  [--pipeline-workers N] (legacy alias for parallel-shared --workers N)\n"
            "  [--shader-cache disabled|reuse] [--pipeline-cache empty|reuse] [--cache-dir path]\n"
            "  [--descriptor-mode request|mapped|native] [--debug-mode disabled|capture|debug]\n"
            "  [--slang-optimization default|O0|O1|O2|O3]\n"
            "  [--spirv-postprocess default|none|dead-branches|slang-dead-branches]\n"
            "  [--disable-driver-optimization]\n"
            "  [--compute-backend pipeline|shader-object|pipeline-binary] [--native-binary-cache-dir path]\n"
            "Defaults: 3 iterations, disabled shader cache, empty pipeline cache, request descriptor mode, disabled debug.\n"
            "Explicit modes stage one batch per iteration. Recipe replay uses captured PSO state and SPIR-V.\n"
            "Graphics catalog requests compile only; use --recipe-file for actual graphics or ray-tracing PSOs.\n";
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::invalid_argument& error) { std::cerr << error.what() << '\n'; return 2; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
