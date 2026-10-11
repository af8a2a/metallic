#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ShaderResourceABI.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Tools/ShaderWarmupRequests.h"

#include <SDL3/SDL.h>
#include <json.hpp>
#include <slang.h>
#include <algorithm>
#include <atomic>
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

namespace {
using namespace metallic::render;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string filter, outputDirectory, cacheDirectory;
    std::string shaderCache = "disabled", pipelineCache = "empty";
    std::string descriptorMode = "request", debugMode = "disabled";
    uint32_t iterations = 3;
    std::optional<uint32_t> requestIndex;
    bool list = false, all = false, compileOnly = false, validation = false;
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
        else {
            if (i + 1 == argc) { throw std::invalid_argument("Missing value for " + arg); }
            const std::string value = argv[++i];
            if (arg == "--filter") { options.filter = value; }
            else if (arg == "--request-index") { options.requestIndex = number(value, arg); }
            else if (arg == "--iterations") { options.iterations = number(value, arg); }
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
        (!options.all && options.filter.empty() && !options.requestIndex))) {
        throw std::invalid_argument("Runs require --output-dir and --filter, --request-index or --all");
    }
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
        } else { group["pipelines_skipped"] = group["pipelines_skipped"].get<size_t>() + 1; }
    }
    Json result = Json::array();
    for (auto& [index, group] : groups) {
        for (auto& [metric, values] : group["values"].items()) {
            auto sorted = values.get<std::vector<double>>();
            std::sort(sorted.begin(), sorted.end());
            const size_t middle = sorted.size() / 2;
            const double median = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) * 0.5;
            group["timings"][metric] = {{"count", sorted.size()}, {"min", sorted.front()}, {"median", median}, {"max", sorted.back()}};
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

int run(const Options& options)
{
    const auto catalog = metallic::tools::shaderWarmupRequests();
    std::vector<size_t> selected;
    for (size_t i = 0; i < catalog.size(); ++i) {
        if (options.requestIndex && i != *options.requestIndex) { continue; }
        if (!options.filter.empty() && (catalog[i].module + "." + catalog[i].entry).find(options.filter) == std::string::npos) { continue; }
        selected.push_back(i);
    }
    if (selected.empty()) { throw std::invalid_argument("No shader requests matched"); }
    if (options.list) {
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
    Json report{{"schema_version", 1}, {"status", "initializing"}, {"samples", Json::array()},
        {"slang_build", spGetBuildTagString()},
        {"shader_target", {{"id", vulkanSpirvTarget().id}, {"revision", vulkanSpirvTarget().revision}}},
        {"configuration", {{"filter", options.filter}, {"iterations", options.iterations},
            {"shader_cache", options.shaderCache}, {"pipeline_cache", options.pipelineCache},
            {"descriptor_mode", options.descriptorMode}, {"debug_mode", options.debugMode},
            {"compile_only", options.compileOnly}, {"validation", options.validation},
            {"cache_directory", cache.string()}, {"selected_requests", selected}}},
        {"scope", "Serial host compilation and compute pipeline creation; no dispatch or frame timing. Empty application caches do not guarantee cold driver/OS caches."}};
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
    DescriptorHeapShaderStrides nativeStrides;
    size_t failures = 0;
    setSlangShaderDebugMode(options.debugMode == "capture" ? SlangShaderDebugMode::CaptureSymbols :
        options.debugMode == "debug" ? SlangShaderDebugMode::ShaderDebug : SlangShaderDebugMode::Disabled);
    try {
        bool rayQuery = false;
        for (size_t i : selected) {
            rayQuery |= std::find(catalog[i].capabilities.begin(), catalog[i].capabilities.end(), "spvRayQueryKHR") != catalog[i].capabilities.end();
        }
        for (size_t index : selected) {
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
            for (uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
                report["samples"].push_back({{"request", requestJson(catalog[index], index)}, {"iteration", iteration},
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
                        begin = Clock::now();
                        if (!videoInitialized) {
                            if (!SDL_Init(SDL_INIT_VIDEO)) {
                                throw std::runtime_error(std::string("Initialize SDL video: ") + SDL_GetError());
                            }
                            videoInitialized = true;
                        }
                        device = require(createDevice({.applicationName = "MetallicShaderPipelineBench",
                            .enableValidation = options.validation, .enableBindlessDescriptorHeap = true,
                            .enableRayTracingAccelerationStructure = rayQuery, .enableRayQuery = rayQuery,
                            .enableClusterAccelerationStructure = rayQuery,
                            .validationSink = {.callback = ValidationCapture::callback, .context = &validation}}), "create device");
                        report["device_initialization_wall_ms"] = milliseconds(begin);
                        const auto vk = vulkan::nativeDevice(*device);
                        const auto& properties = *vk.properties;
                        report["device"] = {{"name", properties.core.deviceName}, {"vendor_id", properties.core.vendorID},
                            {"device_id", properties.core.deviceID}, {"api_version", properties.core.apiVersion},
                            {"driver_version", properties.core.driverVersion}, {"driver_name", properties.driver.driverName},
                            {"driver_info", properties.driver.driverInfo}, {"validation_active", vk.validationMessengerActive}};
                    }
                    if (native && slangDescriptorHeapShaderStrides() != nativeStrides) {
                        throw std::runtime_error("Selected device descriptor strides differ from the native compiler adapter query");
                    }
                    if (options.pipelineCache == "reuse" && !reusedPipelineCache) {
                        sample["status"] = "creating_pipeline_cache";
                        save();
                        reusedPipelineCache = require(device->createPipelineCache({.filePath = pipelineCache.c_str(), .saveOnDestroy = false}), "create reusable cache");
                        report["pipeline_cache_load_status"] = static_cast<int>(reusedPipelineCache->stats().loadStatus);
                    }
                    sample["status"] = "creating_shader_module";
                    save();
                    begin = Clock::now();
                    auto moduleResult = device->createShaderModule({.spirv = code.spirv, .debugName = request.module.c_str()});
                    sample["shader_module_wall_ms"] = milliseconds(begin);
                    auto module = require(std::move(moduleResult), "create shader module");
                    std::unique_ptr<PipelineCache> emptyCache;
                    if (options.pipelineCache == "empty") {
                        sample["status"] = "creating_pipeline_cache";
                        save();
                        emptyCache = require(device->createPipelineCache({.saveOnDestroy = false}), "create empty cache");
                    }
                    auto& activeCache = emptyCache ? *emptyCache : *reusedPipelineCache;
                    const auto before = activeCache.stats();
                    sample["status"] = "creating_pipeline";
                    save();
                    begin = Clock::now();
                    auto created = device->createComputePipeline({.computeShader = {module.get(), entry.c_str()},
                        .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(GPUBufferSpan), .pipelineCache = &activeCache});
                    sample["pipeline_wall_ms"] = milliseconds(begin);
                    auto pipeline = require(std::move(created), "create compute pipeline");
                    const auto after = activeCache.stats();
                    const bool validFeedback = after.unknownFeedbackCount == before.unknownFeedbackCount;
                    sample["pipeline"] = {{"pso_hash", pipeline->psoHash()}, {"metadata_hit", pipeline->pipelineCacheHit()},
                        {"native_cache_hit", validFeedback ? Json(after.nativeHitCount > before.nativeHitCount) : Json(nullptr)},
                        {"feedback_valid", validFeedback}, {"native_create_ms", double(after.lastNativeCreateTimeNanoseconds) / 1e6},
                        {"driver_ms", validFeedback ? Json(double(after.lastDriverDurationNanoseconds) / 1e6) : Json(nullptr)},
                        {"cache_lock_wait_ms", double(after.lastCacheLockWaitTimeNanoseconds) / 1e6}};
                    sample["status"] = "passed";
                    sample["pipeline_status"] = "created";
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
        }
        if (reusedPipelineCache) {
            report["status"] = "saving_pipeline_cache";
            save();
            const auto begin = Clock::now();
            auto saved = reusedPipelineCache->save();
            report["pipeline_cache_save_wall_ms"] = milliseconds(begin);
            require(std::move(saved), "save pipeline cache");
        }
    } catch (const std::exception& error) {
        report["error"] = error.what();
        ++failures;
        std::cerr << error.what() << '\n';
    }
    reusedPipelineCache.reset();
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
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "MetallicShaderPipelineBench [--list] [--filter substring | --request-index N | --all]\n"
            "  --output-dir path [--iterations N] [--compile-only] [--validation]\n"
            "  [--shader-cache disabled|reuse] [--pipeline-cache empty|reuse] [--cache-dir path]\n"
            "  [--descriptor-mode request|mapped|native] [--debug-mode disabled|capture|debug]\n"
            "Defaults: 3 iterations, disabled shader cache, empty pipeline cache, request descriptor mode, disabled debug.\n"
            "Graphics requests compile only; their PSO state is not part of the shader catalog.\n";
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::invalid_argument& error) { std::cerr << error.what() << '\n'; return 2; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
