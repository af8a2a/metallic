#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/Core/ShaderResourceABI.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Tools/ShaderWarmupRequests.h"

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
    std::string filter, outputDirectory, cacheDirectory;
    std::string shaderCache = "disabled", pipelineCache = "empty";
    std::string descriptorMode = "request", debugMode = "disabled";
    std::string mode = "serial", cacheLockScope = "metadata", recipeFile;
    uint32_t iterations = 3;
    uint32_t pipelineReplicas = 1;
    std::optional<uint32_t> pipelineWorkers;
    std::optional<uint32_t> requestIndex;
    bool list = false, all = false, compileOnly = false, validation = false;
    bool explicitMode = false, mergeShards = false;
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
            else if (arg == "--descriptor-mode") { options.descriptorMode = value; }
            else if (arg == "--debug-mode") { options.debugMode = value; }
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
    if (!options.recipeFile.empty() && (options.all || !options.filter.empty() || options.requestIndex || options.compileOnly)) {
        throw std::invalid_argument("--recipe-file cannot be combined with catalog selectors or --compile-only");
    }
    if (!options.recipeFile.empty()) { options.pipelineWorkers = options.pipelineWorkers.value_or(1); }
    return options;
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
        {"cache_lock_wait_ms", double(stats.cacheLockWaitTimeNanoseconds) / 1e6}};
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
};

void createPreparedPipeline(Device& device, PipelineCache& cache, const PreparedPipeline& item, Json& result)
{
    const auto begin = Clock::now();
    PipelineCreationStats metrics;
    if (item.recipe && item.recipe->type == PipelineRecipeType::Graphics) {
        auto created = device.createGraphicsPipeline(item.recipe->graphicsDesc(item.graphicsModules[0].get(),
            item.graphicsModules[1].get(), item.graphicsModules[2].get(), item.graphicsModules[3].get(), &cache), &metrics);
        result["pipeline_wall_ms"] = milliseconds(begin);
        result["pipeline_attempt"] = metricsJson(metrics);
        auto pipeline = require(std::move(created), "create graphics pipeline");
        result["pipeline"] = pipelineJson(*pipeline);
    } else {
        const auto desc = item.recipe ? item.recipe->computeDesc(item.module.get(), &cache) :
            ComputePipelineDesc{.computeShader = {item.module.get(), item.entry.c_str()},
                .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(GPUBufferSpan), .pipelineCache = &cache};
        auto created = device.createComputePipeline(desc, &metrics);
        result["pipeline_wall_ms"] = milliseconds(begin);
        result["pipeline_attempt"] = metricsJson(metrics);
        auto pipeline = require(std::move(created), "create compute pipeline");
        result["pipeline"] = pipelineJson(*pipeline);
    }
    if (item.recipe) {
        result["pipeline"]["expected_pso_hash"] = item.recipe->expectedPsoHash;
        result["pipeline"]["recipe_hash_matches"] = result["pipeline"].at("pso_hash") == item.recipe->expectedPsoHash;
        if (!result["pipeline"]["recipe_hash_matches"].get<bool>()) {
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
                auto& cache = *caches[caches.size() == 1 ? 0 : worker];
                for (size_t i = worker; i < prepared.size(); i += workerCount) {
                    const auto& item = prepared[i];
                    auto& result = results[i];
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
    const auto catalog = metallic::tools::shaderWarmupRequests();
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
                {"pipeline_type", recipe->type == PipelineRecipeType::Graphics ? "graphics" : "compute"}}.dump() << '\n';
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
    const std::string pipelineCache = (cache / "pipeline.pso").string();
    Json report{{"schema_version", 3}, {"status", "initializing"}, {"samples", Json::array()}, {"pipeline_batches", Json::array()},
        {"build_configuration", METALLIC_BENCH_BUILD_CONFIGURATION},
        {"slang_build", spGetBuildTagString()},
        {"shader_target", {{"id", vulkanSpirvTarget().id}, {"revision", vulkanSpirvTarget().revision}}},
        {"configuration", {{"filter", options.filter}, {"iterations", options.iterations},
            {"pipeline_workers", options.pipelineWorkers.value_or(1)}, {"staged_pipeline_batches", options.pipelineWorkers.has_value()},
            {"mode", options.mode}, {"workers", options.pipelineWorkers.value_or(1)},
            {"pipeline_replicas", options.pipelineReplicas}, {"cache_lock_scope", options.cacheLockScope},
            {"merge_shards", options.mergeShards}, {"recipe_file", options.recipeFile},
            {"shader_cache", options.shaderCache}, {"pipeline_cache", options.pipelineCache},
            {"descriptor_mode", options.descriptorMode}, {"debug_mode", options.debugMode},
            {"compile_only", options.compileOnly}, {"validation", options.validation},
            {"cache_directory", cache.string()}, {"selected_requests", selected}}},
        {"scope", "Serial preparation and optional concurrent PSO creation; no dispatch or frame timing. Empty application caches do not guarantee cold driver/OS caches."}};
    for (const char* name : {"METALLIC_SLANG_DESCRIPTOR_MODE", "METALLIC_WORKING_COLOR_SPACE",
        "METALLIC_VK_INTERNAL_PIPELINE_CACHE", "METALLIC_VK_LOG_PIPELINE_KEYS"}) {
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
    uint32_t batchIteration = 0;
    const auto flushBatch = [&] {
        if (prepared.empty()) { return; }
        const uint32_t workerCount = static_cast<uint32_t>(std::min<size_t>(*options.pipelineWorkers, prepared.size()));
        const uint32_t cacheCount = options.mode == "parallel-sharded" ? workerCount : 1;
        std::vector<std::unique_ptr<PipelineCache>> emptyCaches;
        std::vector<PipelineCache*> activeCaches;
        for (const auto& item : prepared) { report["samples"][item.sampleIndex]["status"] = "creating_pipeline_cache"; }
        save();
        for (uint32_t shard = 0; shard < cacheCount; ++shard) {
            if (options.pipelineCache == "empty") {
                emptyCaches.push_back(require(device->createPipelineCache({.saveOnDestroy = false,
                    .serializeCreation = options.cacheLockScope == "creation"}), "create empty batch cache"));
                activeCaches.push_back(emptyCaches.back().get());
            } else if (options.mode == "parallel-sharded") {
                if (shard >= reusedShardCaches.size()) {
                    const auto path = (cache / ("pipeline-shard-" + std::to_string(shard) + ".pso")).string();
                    reusedShardCaches.push_back(require(device->createPipelineCache({.filePath = path.c_str(),
                        .saveOnDestroy = false, .serializeCreation = options.cacheLockScope == "creation"}), "create reusable shard cache"));
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
        batch["iteration"] = batchIteration;
        batch["validation_errors"] = validation.errors.load() - initialErrors;
        if (options.mergeShards) {
            if (!mergedPipelineCache) {
                const auto path = (output / "pipeline-merged.pso").string();
                mergedPipelineCache = require(device->createPipelineCache({.filePath = path.c_str(), .saveOnDestroy = false}), "create merge destination");
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
    const auto ensureReusedCache = [&] {
        if (options.pipelineCache != "reuse" || options.mode == "parallel-sharded" || reusedPipelineCache) { return; }
        reusedPipelineCache = require(device->createPipelineCache({.filePath = pipelineCache.c_str(), .saveOnDestroy = false,
            .serializeCreation = options.cacheLockScope == "creation"}), "create reusable cache");
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
                const auto& mainShader = graphics ? (!recipe->meshShader.spirv.empty() ? recipe->meshShader : recipe->vertexShader) : recipe->computeShader;
                report["samples"].push_back({{"request", {{"index", 0}, {"module", mainShader.moduleName},
                    {"entry", mainShader.sourceEntryPoint}, {"recipe_file", options.recipeFile}}},
                    {"iteration", iteration}, {"pipeline_replica", replicaIndex}, {"pipeline_type", graphics ? "graphics" : "compute"},
                    {"status", "creating_device"}, {"shader_source", "recipe_spirv"}});
                const size_t sampleIndex = report["samples"].size() - 1;
                auto& sample = report["samples"].back();
                save();
                try {
                    ensureDevice(false);
                    ensureReusedCache();
                    PreparedPipeline item{.sampleIndex = sampleIndex, .recipe = &*recipe};
                    sample["status"] = "creating_shader_module";
                    save();
                    const auto begin = Clock::now();
                    size_t bytes = 0;
                    sample["shader_stages"] = Json::array();
                    const auto makeModule = [&](const PipelineRecipeShader& stage, const char* name) -> std::unique_ptr<ShaderModule> {
                        if (stage.spirv.empty()) { return {}; }
                        bytes += stage.spirv.size() * sizeof(uint32_t);
                        sample["shader_stages"].push_back({{"stage", name}, {"module", stage.moduleName},
                            {"entry", stage.entryPoint}, {"source_entry", stage.sourceEntryPoint},
                            {"debug_mode", stage.debugMode}, {"descriptor_mode", stage.descriptorHeapMode},
                            {"spirv_bytes", stage.spirv.size() * sizeof(uint32_t)}, {"spirv_fnv1a64", hashSpirv(stage.spirv)}});
                        return require(device->createShaderModule({.spirv = stage.spirv, .debugName = stage.moduleName.c_str()}), "create recipe shader module");
                    };
                    if (graphics) {
                        item.graphicsModules[0] = makeModule(recipe->vertexShader, "vertex");
                        item.graphicsModules[1] = makeModule(recipe->taskShader, "task");
                        item.graphicsModules[2] = makeModule(recipe->meshShader, "mesh");
                        item.graphicsModules[3] = makeModule(recipe->fragmentShader, "fragment");
                    } else { item.module = makeModule(recipe->computeShader, "compute"); }
                    sample["spirv_bytes"] = bytes;
                    sample["shader_module_wall_ms"] = milliseconds(begin);
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
            auto& sample = report["samples"].back();
            const uint32_t initialValidationErrors = validation.errors.load();
            report["status"] = "running";
            save();
            std::cout << "[ShaderPipelineBench] request=" << index << " iteration=" << iteration << ' ' << request.module << '.' << request.entry << std::endl;
            try {
                bool cacheHit = false;
                std::string diagnostics;
                auto begin = Clock::now();
                auto compiled = ShaderRegistry::instance().getShader(source,
                    {.cacheDirectory = shaderCache.c_str(), .enableDiskCache = options.shaderCache == "reuse", .outCacheHit = &cacheHit}, diagnostics);
                sample["compile_wall_ms"] = milliseconds(begin);
                sample["shader_cache_hit"] = cacheHit;
                sample["diagnostics"] = diagnostics;
                auto code = require(std::move(compiled), "compile shader");
                sample["spirv_bytes"] = code.spirv.size() * sizeof(uint32_t);
                sample["spirv_fnv1a64"] = hashSpirv(code.spirv);
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
                    ensureReusedCache();
                }
                sample["status"] = "creating_shader_module";
                save();
                begin = Clock::now();
                auto moduleResult = device->createShaderModule({.spirv = code.spirv, .debugName = request.module.c_str()});
                sample["shader_module_wall_ms"] = milliseconds(begin);
                auto module = require(std::move(moduleResult), "create shader module");
                if (options.pipelineWorkers) {
                    prepared.push_back({report["samples"].size() - 1, std::move(module), entry});
                    sample["status"] = "prepared";
                    sample["preparation_validation_errors"] = validation.errors.load() - initialValidationErrors;
                    save();
                    continue;
                }
                std::unique_ptr<PipelineCache> emptyCache;
                if (options.pipelineCache == "empty") {
                    sample["status"] = "creating_pipeline_cache";
                    save();
                    emptyCache = require(device->createPipelineCache({.saveOnDestroy = false,
                        .serializeCreation = options.cacheLockScope == "creation"}), "create empty cache");
                }
                auto& activeCache = emptyCache ? *emptyCache : *reusedPipelineCache;
                sample["status"] = "creating_pipeline";
                save();
                PreparedPipeline item{report["samples"].size() - 1, std::move(module), entry};
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
            "Defaults: 3 iterations, disabled shader cache, empty pipeline cache, request descriptor mode, disabled debug.\n"
            "Explicit modes stage one batch per iteration. Recipe replay uses captured PSO state and SPIR-V.\n"
            "Graphics catalog requests compile only; use --recipe-file for actual graphics PSOs.\n";
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::invalid_argument& error) { std::cerr << error.what() << '\n'; return 2; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
