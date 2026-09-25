#include "Runtime/Render/Profiling/NvPerf.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"
#include <json.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <Windows.h>
#endif

#if METALLIC_HAS_NVPERF
#include <nvperf_host_impl.h>
#include <NvPerfVulkan.h>
#include <NvPerfCounterConfiguration.h>
#include <NvPerfCounterData.h>
#endif

namespace metallic::render::profiling {
namespace {
using Json = nlohmann::json;
std::atomic_bool passActive = false;
std::atomic_bool rangeFailed = false;
std::atomic_uint rangeCount = 0;
VkDevice activeDevice = VK_NULL_HANDLE;
std::mutex sessionMutex;

void require(bool ok, const char* message)
{
    if (!ok) { throw std::runtime_error(message); }
}

#if METALLIC_HAS_NVPERF
void check(NVPA_Status status, const char* operation)
{
    if (status != NVPA_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + nv::perf::FormatStatus(status));
    }
}

void initialize()
{
    static const bool initialized = [] {
        const char* paths[] = {METALLIC_NVPERF_LIBRARY_DIR};
        NVPW_SetLibraryLoadPaths_Params load{NVPW_SetLibraryLoadPaths_Params_STRUCT_SIZE};
        load.numPaths = 1; load.ppPaths = paths;
        check(NVPW_SetLibraryLoadPaths(&load), "SetLibraryLoadPaths");
        require(nv::perf::InitializeNvPerf(), "NvPerf initialization failed");
        return true;
    }();
    (void)initialized;
}

void deduplicate(std::vector<const char*>& names)
{
    std::set<std::string> seen;
    std::erase_if(names, [&](const char* name) { return !seen.insert(name).second; });
}
#endif
} // namespace

bool nvPerfRequested()
{
    const auto* value = std::getenv("METALLIC_NVPERF");
    return value && std::strcmp(value, "1") == 0;
}

bool nvPerfPassActive() { return passActive.load(std::memory_order_acquire); }

bool nvPerfInstanceExtensions(std::vector<const char*>& extensions, uint32_t apiVersion, std::string& error)
{
    if (!nvPerfRequested()) { return true; }
    try {
#if METALLIC_HAS_NVPERF
        require(!NsightGraphicsCapture::vulkanInjectionActive(), "NvPerf cannot run under Graphics Capture injection");
#ifdef _WIN32
        require(!GetModuleHandleW(L"WarpVizTarget.dll") && !GetModuleHandleW(L"renderdoc.dll"),
            "NvPerf cannot run under another GPU profiler/capture");
#endif
        initialize();
        require(nv::perf::VulkanAppendInstanceRequiredExtensions(extensions, apiVersion), "NvPerf instance extensions unavailable");
        deduplicate(extensions);
        return true;
#else
        throw std::runtime_error("NvPerf backend not compiled; enable METALLIC_ENABLE_NVPERF with METALLIC_NVPERF_SDK_ROOT");
#endif
    } catch (const std::exception& ex) { error = ex.what(); return false; }
}

bool nvPerfDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
    std::vector<const char*>& extensions, std::string& error)
{
    if (!nvPerfRequested()) { return true; }
    try {
#if METALLIC_HAS_NVPERF
        require(nv::perf::VulkanAppendDeviceRequiredExtensions(instance, physicalDevice,
            reinterpret_cast<void*>(vkGetInstanceProcAddr), extensions), "NvPerf device extensions unavailable");
        deduplicate(extensions);
        return true;
#else
        throw std::runtime_error("NvPerf backend not compiled");
#endif
    } catch (const std::exception& ex) { error = ex.what(); return false; }
}

struct NvPerfSession::Impl {
    std::unique_lock<std::mutex> lock;
    std::filesystem::path output;
    Json report{{"protocol", "metallic-nvperf-v1"}, {"status", "uninitialized"},
        {"measurementKind", "diagnostic"}, {"backend", "nvperf-vulkan-range"},
        {"scope", "in-frame-command-ranges-on-graphics-queue"}, {"clockPolicy", "unaltered"}};
#if METALLIC_HAS_NVPERF
    VkQueue queue = VK_NULL_HANDLE;
    bool session = false, inPass = false;
    std::thread worker;
    std::atomic_int workerStatus = NVPA_STATUS_SUCCESS;
    nv::perf::MetricsEvaluator evaluator;
    nv::perf::CounterConfiguration configuration;
    std::vector<uint8_t> availability, image, scratch;
    std::vector<std::string> metrics;
    std::vector<NVPW_MetricEvalRequest> requests;

    void bytes(const char* name, const std::vector<uint8_t>& data)
    {
        std::ofstream file(output / name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        require(bool(file), "NvPerf artifact write failed");
    }

    void endSession()
    {
        passActive = false;
        if (inPass) {
            NVPW_VK_Profiler_Queue_EndPass_Params end{NVPW_VK_Profiler_Queue_EndPass_Params_STRUCT_SIZE};
            end.queue = queue;
            const auto status = NVPW_VK_Profiler_Queue_EndPass(&end);
            inPass = false;
            if (status != NVPA_STATUS_SUCCESS) { report["cleanupEndPassStatus"] = int(status); }
        }
        if (session) {
            NVPW_VK_Profiler_Queue_EndSession_Params end{NVPW_VK_Profiler_Queue_EndSession_Params_STRUCT_SIZE};
            end.queue = queue; end.timeout = 10000;
            const auto status = NVPW_VK_Profiler_Queue_EndSession(&end);
            session = false;
            // The outer process runner has a hard timeout if the driver cannot unwind.
            if (worker.joinable()) { worker.join(); }
            require(status == NVPA_STATUS_SUCCESS && !end.timeoutExpired, "NvPerf EndSession failed/timed out");
            check(static_cast<NVPA_Status>(workerStatus.load()), "ServicePendingGpuOperations");
        }
    }
#endif
    void save()
    {
        if (output.empty()) { return; }
        std::ofstream file(output / "NvPerf.json");
        file << report.dump(2) << '\n';
        require(bool(file), "NvPerf report write failed");
    }
};

NvPerfSession::NvPerfSession() : impl_(std::make_unique<Impl>()) {}
NvPerfSession::~NvPerfSession() { cancel(); }

bool NvPerfSession::begin(Device& device, Queue& queue, const std::filesystem::path& output, std::string& error)
{
    auto& s = *impl_;
    try {
        require(nvPerfRequested(), "NvPerf not requested at device creation");
        require(!s.lock.owns_lock(), "NvPerf session already owns a collection");
        s.lock = std::unique_lock(sessionMutex, std::try_to_lock);
        require(s.lock.owns_lock(), "NvPerf session already active in process");
        require(!std::filesystem::exists(output), "NvPerf output must be fresh");
        std::filesystem::create_directories(output);
        s.output = output; s.report["status"] = "preparing";
#if METALLIC_HAS_NVPERF
        initialize();
        const auto native = vulkan::nativeDevice(device);
        const auto q = vulkan::nativeQueue(queue);
        NVPW_VK_Profiler_GetRequiredInstanceExtensions_Params support{NVPW_VK_Profiler_GetRequiredInstanceExtensions_Params_STRUCT_SIZE};
        support.apiVersion = native.apiVersion;
        check(NVPW_VK_Profiler_GetRequiredInstanceExtensions(&support), "Vulkan version support");
        s.report["vulkanApiVersion"] = native.apiVersion;
        s.report["vulkanVersionOfficiallySupported"] = bool(support.isOfficiallySupportedVersion);
        require(nv::perf::VulkanLoadDriver(native.instance), "NvPerf Vulkan driver unavailable");
        require(nv::perf::profiler::VulkanIsGpuSupported(native.instance, native.physicalDevice, native.device,
            vkGetInstanceProcAddr, vkGetDeviceProcAddr), "NvPerf GPU/driver unsupported");
        const auto identifiers = nv::perf::VulkanGetDeviceIdentifiers(native.instance, native.physicalDevice,
            native.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr);
        require(identifiers.pChipName != nullptr, "NvPerf missing chip identity");
        s.report["chip"] = identifiers.pChipName;
        s.report["libraryDirectory"] = METALLIC_NVPERF_LIBRARY_DIR;
        s.report["queueFamily"] = q.familyIndex;
        s.queue = q.queue;
        NVPW_VK_Profiler_Queue_GetCounterAvailability_Params available{NVPW_VK_Profiler_Queue_GetCounterAvailability_Params_STRUCT_SIZE};
        available.instance = native.instance; available.physicalDevice = native.physicalDevice;
        available.device = native.device; available.queue = q.queue;
        available.pfnGetInstanceProcAddr = reinterpret_cast<void*>(vkGetInstanceProcAddr);
        available.pfnGetDeviceProcAddr = reinterpret_cast<void*>(vkGetDeviceProcAddr);
        check(NVPW_VK_Profiler_Queue_GetCounterAvailability(&available), "CounterAvailability size");
        require(available.counterAvailabilityImageSize > 0, "Empty counter availability");
        s.availability.resize(available.counterAvailabilityImageSize);
        available.pCounterAvailabilityImage = s.availability.data();
        check(NVPW_VK_Profiler_Queue_GetCounterAvailability(&available), "CounterAvailability image");
        s.bytes("CounterAvailability.bin", s.availability);
        std::vector<uint8_t> evalScratch(nv::perf::VulkanCalculateMetricsEvaluatorScratchBufferSize(identifiers.pChipName));
        require(!evalScratch.empty(), "NvPerf metric evaluator unavailable");
        auto* evaluator = nv::perf::VulkanCreateMetricsEvaluator(evalScratch.data(), evalScratch.size(), identifiers.pChipName);
        require(evaluator != nullptr, "NvPerf metric evaluator initialization failed");
        s.evaluator = nv::perf::MetricsEvaluator(evaluator, std::move(evalScratch));
        s.metrics = {"gpu__time_duration.sum", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"};
        if (const char* path = std::getenv("METALLIC_NVPERF_METRICS")) {
            std::ifstream file(path); Json request; file >> request;
            require(request.is_array() && !request.empty() && request.size() <= 16, "NvPerf metrics require 1..16 names");
            s.metrics = request.get<std::vector<std::string>>();
        }
        require(std::set<std::string>(s.metrics.begin(), s.metrics.end()).size() == s.metrics.size(), "Duplicate NvPerf metric");
        s.report["metricNames"] = s.metrics;
        for (const auto& name : s.metrics) {
            NVPW_MetricEvalRequest request{};
            require(s.evaluator.ToMetricEvalRequest(name.c_str(), request), ("Unknown metric: " + name).c_str());
            s.requests.push_back(request);
        }
        auto* raw = nv::perf::profiler::VulkanCreateRawCounterConfig(identifiers.pChipName);
        require(raw != nullptr, "NvPerf raw counter config unavailable");
        nv::perf::MetricsConfigBuilder builder;
        require(builder.Initialize(s.evaluator, raw, identifiers.pChipName), "NvPerf config builder failed");
        NVPW_RawCounterConfig_SetCounterAvailability_Params setAvailability{NVPW_RawCounterConfig_SetCounterAvailability_Params_STRUCT_SIZE};
        setAvailability.pRawCounterConfig = raw; setAvailability.pCounterAvailabilityImage = s.availability.data();
        check(NVPW_RawCounterConfig_SetCounterAvailability(&setAvailability), "SetCounterAvailability");
        require(builder.AddMetrics(s.requests.data(), s.requests.size()), "Metric counters unavailable");
        require(nv::perf::CreateConfiguration(builder, s.configuration), "NvPerf configuration failed");
        s.report["requiredPasses"] = s.configuration.numPasses;
        s.bytes("ConfigImage.bin", s.configuration.configImage);
        s.bytes("CounterDataPrefix.bin", s.configuration.counterDataPrefix);
        require(s.configuration.numPasses == 1, "multipass_requires_restored_workload: choose a single-pass metric set");

        NVPW_VK_Profiler_CounterDataImageOptions options{NVPW_VK_Profiler_CounterDataImageOptions_STRUCT_SIZE};
        options.pCounterDataPrefix = s.configuration.counterDataPrefix.data();
        options.counterDataPrefixSize = s.configuration.counterDataPrefix.size();
        options.maxNumRanges = 4; options.maxNumRangeTreeNodes = 8; options.maxRangeNameLength = 64;
        NVPW_VK_Profiler_CounterDataImage_CalculateSize_Params size{NVPW_VK_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
        size.pOptions = &options; size.counterDataImageOptionsSize = options.structSize;
        check(NVPW_VK_Profiler_CounterDataImage_CalculateSize(&size), "CounterData size");
        s.image.resize(size.counterDataImageSize);
        NVPW_VK_Profiler_CounterDataImage_Initialize_Params init{NVPW_VK_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
        init.pOptions = &options; init.counterDataImageOptionsSize = options.structSize;
        init.counterDataImageSize = s.image.size(); init.pCounterDataImage = s.image.data();
        check(NVPW_VK_Profiler_CounterDataImage_Initialize(&init), "CounterData initialize");
        NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchSize{NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
        scratchSize.counterDataImageSize = s.image.size(); scratchSize.pCounterDataImage = s.image.data();
        check(NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize(&scratchSize), "CounterData scratch size");
        s.scratch.resize(scratchSize.counterDataScratchBufferSize);
        NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params scratchInit{NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
        scratchInit.counterDataImageSize = s.image.size(); scratchInit.pCounterDataImage = s.image.data();
        scratchInit.counterDataScratchBufferSize = s.scratch.size(); scratchInit.pCounterDataScratchBuffer = s.scratch.data();
        check(NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer(&scratchInit), "CounterData scratch initialize");
        NVPW_VK_Profiler_CalcTraceBufferSize_Params trace{NVPW_VK_Profiler_CalcTraceBufferSize_Params_STRUCT_SIZE};
        trace.maxRangesPerPass = 4; trace.avgRangeNameLength = 64;
        check(NVPW_VK_Profiler_CalcTraceBufferSize(&trace), "Trace buffer size");
        NVPW_VK_Profiler_Queue_BeginSession_Params start{NVPW_VK_Profiler_Queue_BeginSession_Params_STRUCT_SIZE};
        start.instance = native.instance; start.physicalDevice = native.physicalDevice; start.device = native.device;
        start.queue = s.queue; start.pfnGetInstanceProcAddr = reinterpret_cast<void*>(vkGetInstanceProcAddr);
        start.pfnGetDeviceProcAddr = reinterpret_cast<void*>(vkGetDeviceProcAddr);
        start.numTraceBuffers = 1; start.traceBufferSize = trace.traceBufferSize; start.maxRangesPerPass = 4; start.maxLaunchesPerPass = 4;
        check(NVPW_VK_Profiler_Queue_BeginSession(&start), "BeginSession");
        s.session = true;
        s.worker = std::thread([&s] {
            NVPW_VK_Queue_ServicePendingGpuOperations_Params service{NVPW_VK_Queue_ServicePendingGpuOperations_Params_STRUCT_SIZE};
            service.queue = s.queue; service.numOperations = 0; service.timeout = 0xffffffffu;
            s.workerStatus = NVPW_VK_Queue_ServicePendingGpuOperations(&service);
        });
        NVPW_VK_Profiler_Queue_SetConfig_Params config{NVPW_VK_Profiler_Queue_SetConfig_Params_STRUCT_SIZE};
        config.queue = s.queue; config.pConfig = s.configuration.configImage.data(); config.configSize = s.configuration.configImage.size();
        config.minNestingLevel = 1; config.numNestingLevels = 1; config.targetNestingLevel = 1;
        check(NVPW_VK_Profiler_Queue_SetConfig(&config), "SetConfig");
        NVPW_VK_Profiler_Queue_BeginPass_Params begin{NVPW_VK_Profiler_Queue_BeginPass_Params_STRUCT_SIZE};
        begin.queue = s.queue;
        check(NVPW_VK_Profiler_Queue_BeginPass(&begin), "BeginPass");
        s.inPass = true;
        activeDevice = native.device; rangeCount = 0; rangeFailed = false; passActive = true;
        s.report["status"] = "collecting"; s.save();
        return true;
#else
        throw std::runtime_error("NvPerf backend not compiled");
#endif
    } catch (const std::exception& ex) {
        error = ex.what(); s.report["status"] = "failed"; s.report["error"] = error; cancel(); return false;
    }
}

bool NvPerfSession::finish(std::string& error)
{
    auto& s = *impl_;
    try {
#if METALLIC_HAS_NVPERF
        require(s.inPass && rangeCount == 2 && !rangeFailed, "Missing/failed production range commands");
        passActive = false;
        NVPW_VK_Profiler_Queue_EndPass_Params end{NVPW_VK_Profiler_Queue_EndPass_Params_STRUCT_SIZE};
        end.queue = s.queue;
        check(NVPW_VK_Profiler_Queue_EndPass(&end), "EndPass");
        s.inPass = false;
        require(end.allPassesSubmitted, "NvPerf unexpectedly requires another pass");
        NVPW_VK_Profiler_Queue_DecodeCounters_Params decode{NVPW_VK_Profiler_Queue_DecodeCounters_Params_STRUCT_SIZE};
        decode.queue = s.queue; decode.counterDataImageSize = s.image.size(); decode.pCounterDataImage = s.image.data();
        decode.counterDataScratchBufferSize = s.scratch.size(); decode.pCounterDataScratchBuffer = s.scratch.data();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        do {
            check(NVPW_VK_Profiler_Queue_DecodeCounters(&decode), "DecodeCounters");
            require(!decode.numRangesDropped && !decode.numTraceBytesDropped, "NvPerf counter buffer overflow");
            if (decode.allPassesCollected) { break; }
            require(std::chrono::steady_clock::now() < deadline, "NvPerf decode timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (true);
        s.bytes("CounterDataImage.bin", s.image);
        require(s.evaluator.MetricsEvaluatorSetDeviceAttributes(s.image.data(), s.image.size()), "Metric device attributes failed");
        require(nv::perf::CounterDataGetNumRanges(s.image.data()) == 2, "NvPerf decoded range count mismatch");
        Json ranges = Json::array(); std::set<std::string> names;
        for (size_t i = 0; i < 2; ++i) {
            const auto name = nv::perf::profiler::CounterDataGetRangeName(s.image.data(), i, '/');
            require((name == "WorkControl/early" || name == "WorkControl/late") && names.insert(name).second, "Wrong/duplicate NvPerf range");
            std::vector<double> values(s.metrics.size());
            require(s.evaluator.EvaluateToGpuValues(s.image.data(), s.image.size(), i, s.requests.size(), s.requests.data(), values.data()), "Metric evaluation failed");
            Json metrics = Json::array();
            for (size_t j = 0; j < values.size(); ++j) {
                require(std::isfinite(values[j]) && values[j] >= 0, "Invalid NvPerf metric value");
                std::vector<NVPW_DimUnitFactor> units;
                require(s.evaluator.GetMetricDimUnits(s.requests[j], units), "Metric units unavailable");
                Json dimensions = Json::array();
                for (const auto& unit : units) {
                    const char* name = nv::perf::ToCString(s.evaluator, static_cast<NVPW_DimUnitName>(unit.dimUnit), false);
                    require(name && *name, "Metric dimensional unit name unavailable");
                    dimensions.push_back({{"unit", unit.dimUnit}, {"name", name}, {"exponent", unit.exponent}});
                }
                metrics.push_back({{"name", s.metrics[j]}, {"value", values[j]}, {"dimUnits", dimensions}});
            }
            ranges.push_back({{"name", name}, {"index", i}, {"metrics", metrics}});
        }
        s.endSession();
        s.report.update({{"status", "complete"}, {"ranges", ranges}, {"passesCollected", 1},
            {"numRangesDropped", 0}, {"numTraceBytesDropped", 0}});
        s.save(); s.lock.unlock();
        return true;
#else
        throw std::runtime_error("NvPerf backend not compiled");
#endif
    } catch (const std::exception& ex) {
        error = ex.what(); s.report["status"] = "failed"; s.report["error"] = error; cancel(); return false;
    }
}

void NvPerfSession::cancel()
{
    auto& s = *impl_;
    if (!s.lock.owns_lock()) { return; }
    try {
#if METALLIC_HAS_NVPERF
        s.endSession();
#endif
        if (s.report["status"] != "failed") { s.report["status"] = "cancelled"; }
        s.save();
    } catch (const std::exception& ex) { spdlog::error("[NvPerf] cleanup: {}", ex.what()); }
    s.lock.unlock();
}

NvPerfRange::NvPerfRange(CommandBuffer& commands, const char* name)
{
#if METALLIC_HAS_NVPERF
    if (!nvPerfPassActive()) { return; }
    if (vulkan::nativeCommandBufferDevice(commands) != activeDevice) { rangeFailed = true; return; }
    commands_ = vulkan::nativeCommandBuffer(commands);
    if (!nv::perf::profiler::VulkanPushRange(commands_, name)) { rangeFailed = true; commands_ = VK_NULL_HANDLE; }
    else { ++rangeCount; }
#endif
}

NvPerfRange::~NvPerfRange()
{
#if METALLIC_HAS_NVPERF
    if (commands_ && !nv::perf::profiler::VulkanPopRange(commands_)) { rangeFailed = true; }
#endif
}
} // namespace metallic::render::profiling

