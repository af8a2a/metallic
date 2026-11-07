#include "Editor/EditorApplication.h"
#include "Runtime/Task/TaskSystem.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace metallic {
namespace {

using Json = render::RenderGraphProperties;
using Clock = std::chrono::steady_clock;

int64_t timestamp(Clock::time_point time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

Json memorySnapshot(const render::DeviceMemoryBudget& budget)
{
    Json heaps = Json::array(), domains = Json::array();
    for (const auto& heap : budget.heaps) {
        heaps.push_back({{"sizeBytes", heap.sizeBytes}, {"budgetBytes", heap.budgetBytes},
            {"usageBytes", heap.usageBytes}, {"blockBytes", heap.blockBytes},
            {"allocationBytes", heap.allocationBytes}, {"blockCount", heap.blockCount},
            {"allocationCount", heap.allocationCount}, {"deviceLocal", heap.deviceLocal}});
    }
    for (const auto& domain : budget.domains) {
        domains.push_back({{"allocationBytes", domain.allocationBytes},
            {"peakAllocationBytes", domain.peakAllocationBytes}, {"deviceLocalBytes", domain.deviceLocalBytes},
            {"allocationCount", domain.allocationCount}});
    }
    return {{"heaps", heaps}, {"domains", domains}, {"driverBudget", budget.driverBudget},
        {"reservedBytes", budget.reservedBytes}, {"deniedAllocations", budget.deniedAllocations},
        {"availableBytes", budget.availableBytes}, {"primaryDeviceLocalHeap", budget.primaryDeviceLocalHeap},
        {"trackedAllocationBytes", budget.trackedAllocationBytes}, {"trackedDeviceLocalBytes", budget.trackedDeviceLocalBytes},
        {"peakTrackedAllocationBytes", budget.peakTrackedAllocationBytes},
        {"peakTrackedDeviceLocalBytes", budget.peakTrackedDeviceLocalBytes}};
}

Json gpuSnapshot(NVMLMonitor& monitor)
{
    monitor.refresh();
    const auto sampledAt = monitor.lastSampleTime();
    const double sampleAge = std::chrono::duration<double, std::milli>(Clock::now() - sampledAt).count();
    Json result = Json::array();
    for (const auto& device : monitor.devices()) {
        const auto* sample = monitor.currentSample(device);
        if (sample && sample->valid) {
            result.push_back({{"device", device.name}, {"driver", monitor.driverVersion()},
                {"sampleSteadyNs", timestamp(sampledAt)}, {"sampleAgeMs", sampleAge},
                {"sampleIntervalMs", monitor.sampleIntervalMilliseconds()},
                {"gpuLoadPercent", sample->gpuLoad}, {"wholeDeviceMemoryUsedBytes", sample->memoryUsed},
                {"graphicsClockMHz", sample->graphicsClockMHz}, {"temperatureC", sample->temperature},
                {"powerMilliwatts", sample->powerMilliwatts}, {"throttleReasons", sample->throttleReasons}});
        }
    }
    return result;
}

class BenchmarkTaskSink final : public task::ITaskEventSink {
public:
    explicit BenchmarkTaskSink(const std::filesystem::path& path) : file_(path)
    {
        if (!file_) { throw std::runtime_error("Cannot create TaskGraph diagnostic output"); }
    }
    void onGraphSubmitted(const task::TaskGraphSnapshot&) override {}
    void onTaskStateChanged(const task::TaskNodeEvent&) override {}
    void onGraphCompleted(const task::TaskGraphSnapshot& graph) override
    {
        try {
            Json nodes = Json::array(), edges = Json::array();
            for (const auto& node : graph.nodes) {
                nodes.push_back({{"nodeIndex", node.handle.nodeIndex}, {"name", node.desc.name},
                    {"category", node.desc.category}, {"userTag", node.desc.userTag},
                    {"state", task::taskStateName(node.state)}, {"workerThreadId", node.workerThreadId},
                    {"scheduledNs", timestamp(node.scheduledTime)}, {"startNs", timestamp(node.startTime)},
                    {"finishNs", timestamp(node.finishTime)}});
            }
            for (const auto& edge : graph.edges) {
                edges.push_back({edge.prerequisite.nodeIndex, edge.dependent.nodeIndex});
            }
            const Json row{{"graphId", graph.graphId}, {"executionId", graph.executionId}, {"name", graph.name},
                {"submittedNs", timestamp(graph.submittedTime)}, {"completedNs", timestamp(graph.completedTime)},
                {"status", task::taskGraphStatusName(graph.status)}, {"nodes", nodes}, {"edges", edges}};
            std::lock_guard lock(mutex_);
            file_ << row.dump() << '\n';
            file_.flush();
            if (!file_) { writeFailed_.store(true, std::memory_order_relaxed); }
        } catch (...) {
            // TaskSystem isolates observer exceptions. Keep an explicit failure
            // for the coordinator to check before accepting diagnostic evidence.
            writeFailed_.store(true, std::memory_order_relaxed);
        }
    }
    bool writeFailed() const
    {
        return writeFailed_.load(std::memory_order_relaxed);
    }
private:
    std::atomic_bool writeFailed_{false};
    std::mutex mutex_;
    std::ofstream file_;
};

Json readFinalImage(render::Device& device, render::Queue& queue, render::RenderGraphExecutor& graph,
    const std::string& outputName, const std::filesystem::path& directory)
{
    auto* output = graph.outputResource(outputName);
    if (!output || (output->desc.format != render::Format::RGBA8Unorm &&
        output->desc.format != render::Format::RGBA16Sfloat)) {
        throw std::runtime_error("Benchmark final output requires RGBA8 or RGBA16F");
    }
    const bool half = output->desc.format == render::Format::RGBA16Sfloat;
    const uint64_t bytes = uint64_t(output->desc.width) * output->desc.height * (half ? 8 : 4);
    auto readback = device.createBuffer({.size = bytes, .usage = render::BufferUsageBits::TransferDestination,
        .memoryLocation = render::MemoryLocation::HostReadback});
    auto pool = device.createCommandPool(queue);
    if (!readback || !pool) { throw std::runtime_error("Benchmark readback allocation failed"); }
    auto commands = (*pool)->createCommandBuffer();
    render::QueueSubmissionTracker tracker;
    render::RenderFrameContext frame;
    if (!commands || !tracker.initialize(device, queue) || !frame.begin(0) ||
        !(*commands)->begin(frame.submissionContext()) ||
        !graph.transitionOutput(**commands, outputName, render::ResourceState::TransferSource) ||
        !(*readback)->slice().and_then([&](const auto& slice) {
            return (*commands)->copyTextureToBuffer({.texture = output->texture, .buffer = slice,
                .width = output->desc.width, .height = output->desc.height});
        }) || !graph.transitionOutput(**commands, outputName, render::ResourceState::ShaderRead) ||
        !(*commands)->end()) { throw std::runtime_error("Benchmark image copy recording failed"); }
    render::CommandBuffer* buffers[]{commands->get()};
    if (!tracker.submit({.commandBuffers = {buffers, 1}}, frame) || !frame.wait()) {
        throw std::runtime_error("Benchmark image copy completion failed");
    }
    (*readback)->invalidate();
    const auto* mapped = static_cast<const uint8_t*>((*readback)->map());
    if (!mapped) { throw std::runtime_error("Benchmark image map failed"); }
    std::ofstream file(directory / "FinalColor.bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(mapped), static_cast<std::streamsize>(bytes));
    (*readback)->unmap();
    if (!file) { throw std::runtime_error("Benchmark image export failed"); }
    return {{"file", "FinalColor.bin"}, {"width", output->desc.width}, {"height", output->desc.height},
        {"format", half ? "RGBA16Sfloat" : "RGBA8Unorm"}, {"bytes", bytes},
        {"colorEncoding", static_cast<int>(output->colorEncoding)}};
}

} // namespace

bool EditorApplication::configureCPUJobBenchmark(uint32_t& workerCount)
{
    const char* path = std::getenv("METALLIC_CPU_JOB_BENCH_CONFIG");
    const char* output = std::getenv("METALLIC_CPU_JOB_BENCH_OUTPUT");
    if (!path && !output) { return true; }
    try {
        if (!path || !output) { throw std::runtime_error("CPU job benchmark needs CONFIG and OUTPUT"); }
        std::ifstream input(path);
        if (!input) { throw std::runtime_error("Cannot open CPU job benchmark config"); }
        cpuJobBenchmarkConfig_ = Json::parse(input);
        auto& config = cpuJobBenchmarkConfig_;
        if (!config.is_object()) { throw std::runtime_error("CPU job config must be an object"); }
        if (config.value("synchronizationValidation", false) && !config.value("validation", false)) {
            throw std::runtime_error("Synchronization validation requires the validation configuration");
        }
        workerCount = config.at("workerCount").get<uint32_t>();
        const uint32_t maximum = std::max(1u, std::thread::hardware_concurrency() -
            (std::thread::hardware_concurrency() > 1 ? 1u : 0u));
        const auto frames = config.at("frames").get<uint32_t>();
        const auto warmup = config.at("warmupFrames").get<uint32_t>();
        const auto width = config.at("width").get<uint32_t>(), height = config.at("height").get<uint32_t>();
        if (!workerCount || workerCount > maximum || !frames || frames > 60000 || warmup > 6000 ||
            width < 64 || height < 64 || width > 8192 || height > 8192) {
            throw std::runtime_error("Invalid CPU job worker/frame/extent limits");
        }
        config.at("sample").get_ref<const std::string&>();
        const auto source = config.at("scenePath").get<std::string>();
        if (source.empty() || !std::filesystem::is_regular_file(source)) { throw std::runtime_error("Benchmark scene missing"); }
        if (!config.at("camera").is_object()) { throw std::runtime_error("Benchmark camera must be an object"); }
        render::RenderView checkedView;
        if (!checkedView.setCameraProperties(config.at("camera"))) { throw std::runtime_error("Invalid benchmark camera"); }
        cpuJobRecordingWorkerLimit_ = config.value("recordingWorkerLimit", workerCount);
        if (!cpuJobRecordingWorkerLimit_ || cpuJobRecordingWorkerLimit_ > workerCount) {
            throw std::runtime_error("Benchmark recording limit must fit the shared pool");
        }
        return true;
    } catch (const std::exception& error) {
        spdlog::error("[CPU Jobs] Configuration failed: {}", error.what());
        return false;
    }
}

bool EditorApplication::runCPUJobBenchmark()
{
    const std::filesystem::path output(std::getenv("METALLIC_CPU_JOB_BENCH_OUTPUT"));
    if (std::filesystem::exists(output / "Capture.json")) {
        spdlog::error("[CPU Jobs] Benchmark output already exists: {}", output.generic_string());
        return false;
    }
    Json report{{"protocol", "metallic-cpu-jobs-production-v1"}, {"status", "failed"},
        {"config", cpuJobBenchmarkConfig_}};
    task::TaskEventSinkToken sinkToken = 0;
    std::shared_ptr<BenchmarkTaskSink> taskSink;
    bool passed = false;
    try {
        std::filesystem::create_directories(output);
        const auto& config = cpuJobBenchmarkConfig_;
        const bool diagnostics = config.value("diagnostics", false);
        if (diagnostics) {
            taskSink = std::make_shared<BenchmarkTaskSink>(output / "TaskGraphs.jsonl");
            sinkToken = task::taskSystem().subscribe(taskSink);
            report["taskGraphDiagnostics"] = "TaskGraphs.jsonl";
        }
        const bool validation = debugRuntime_ && std::getenv("METALLIC_DEBUG_VALIDATION") &&
            std::string_view(std::getenv("METALLIC_DEBUG_VALIDATION")) == "1";
        if (validation != config.value("validation", false)) { throw std::runtime_error("Validation mode mismatch"); }
        report["validationRequested"] = validation;
        report["synchronizationValidationRequested"] = config.value("synchronizationValidation", false);
        const auto native = render::vulkan::nativeDevice(*device_);
        report["validationEnabled"] = native.validationEnabled;
        report["synchronizationValidationEnabled"] = native.synchronizationValidationEnabled;
        report["validationMessengerActive"] = native.validationMessengerActive;
        if (native.validationEnabled != validation ||
            native.synchronizationValidationEnabled != config.value("synchronizationValidation", false) ||
            (validation && !native.validationMessengerActive)) {
            throw std::runtime_error("Active Vulkan validation differs from benchmark configuration");
        }
        report["diagnostics"] = diagnostics;
        report["graphicsCaptureInjected"] = render::profiling::NsightGraphicsCapture::vulkanInjectionActive();
        const auto shaderDebug = render::slangShaderDebugMode();
        report["shaderDebugMode"] = shaderDebug == render::SlangShaderDebugMode::ShaderDebug ? "unoptimized-debug"
            : shaderDebug == render::SlangShaderDebugMode::CaptureSymbols ? "capture-symbols" : "disabled";
        report["measurementKind"] = validation ? "validation" : diagnostics ? "diagnostic" : "performance";
        if (!validation && !diagnostics &&
            (report["graphicsCaptureInjected"].get<bool>() || shaderDebug != render::SlangShaderDebugMode::Disabled)) {
            throw std::runtime_error("Performance capture requires uninjected graphics and optimized shaders without capture symbols");
        }
        const char* noVsync = std::getenv("METALLIC_FULL_ROAM_NO_VSYNC");
        const bool vsyncRequested = !(noVsync && std::string_view(noVsync) == "1");
        report["vsyncRequested"] = vsyncRequested;
        report["presentMode"] = nullptr;
        report["presentModeEvidence"] = "Actual Vulkan mode is logged in Console.log [Swapchain] presentMode; no public Swapchain query exposes it";
        report["framesInFlight"] = frameSlots_.size();
        report["swapchainImageCount"] = swapchain_->imageCount();
        if (config.contains("expectedVsync") && config.at("expectedVsync").get<bool>() != vsyncRequested) {
            throw std::runtime_error("VSync request differs from benchmark expectation");
        }
        report["workerCount"] = task::taskSystem().workerCount();
        report["recordingWorkerLimit"] = cpuJobRecordingWorkerLimit_;
        report["scene"] = std::filesystem::weakly_canonical(config.at("scenePath").get<std::string>()).generic_string();
        report["camera"] = config.at("camera");
        report["hiddenWindow"] = true;
        report["trajectoryClock"] = "fixed camera; no scene edits or camera route during capture";
        report["cpuFramePathContract"] = "nonoverlapping-coordinator-wall-segments-v1";
        report["frameTimeDefinition"] = "start-to-start editor loop including frame-slot wait, rendering, submission, Present and telemetry; final interval ends after final telemetry; no per-frame GPU drain/readback";
        report["resourceDefinition"] = "VMA/driver budget snapshots after selected frames; tracked peaks are simultaneous RHI allocation totals over this device lifetime, excluding untracked SDK/driver allocations; domain peaks are independent and must not be summed; OS monitor spans process lifetime";
        fullRoamActive_ = true;
        fullRoamWidth_ = config.at("width"); fullRoamHeight_ = config.at("height");
        if (config.contains("materialTextureBudgetMiB")) {
            const auto budget = config.at("materialTextureBudgetMiB").get<uint32_t>();
            if (!budget || budget > 65536) { throw std::runtime_error("Invalid material texture budget"); }
            for (const auto& node : renderGraph_.nodes()) {
                if (node.properties.contains("materialTextureBudgetMiB")) {
                    auto properties = node.properties;
                    properties["materialTextureBudgetMiB"] = budget;
                    if (!renderGraph_.setNodeProperties(node.id, std::move(properties))) {
                        throw std::runtime_error("Failed to configure benchmark texture budget");
                    }
                }
            }
        }
        if (config.contains("geometryBudgetMiB")) {
            const auto budget = config.at("geometryBudgetMiB").get<uint32_t>();
            if (!budget || budget > 65536) { throw std::runtime_error("Invalid geometry pool budget"); }
            const auto* visibility = renderGraph_.findNode("VBuffer");
            if (!visibility || !visibility->properties.contains("maxResidentBytes")) {
                throw std::runtime_error("Benchmark graph has no configurable geometry pool");
            }
            auto properties = visibility->properties;
            properties["maxResidentBytes"] = uint64_t(budget) * 1024 * 1024;
            if (!renderGraph_.setNodeProperties(visibility->id, std::move(properties))) {
                throw std::runtime_error("Failed to configure benchmark geometry pool");
            }
        }
        applyViewportCameraProperties({{"camera", config.at("camera")}}, nullptr);
        report["camera"] = viewportView_.cameraProperties();
        const auto* producer = renderGraph_.findNode("VBuffer");
        if (!producer || std::filesystem::weakly_canonical(producer->properties.at("path").get<std::string>()) !=
            std::filesystem::weakly_canonical(startupScenePath_)) { throw std::runtime_error("Effective graph scene mismatch"); }

        struct Sample {
            uint64_t frame = 0;
            int64_t originNs = 0;
            double frameMs = 0;
            Json segments = Json::array(), memory = nullptr, scheduling, gpu = nullptr;
            render::vulkan::StreamlineFrameBeginProfile frameBegin;
            uint64_t executionId = 0, graphGeneration = 0;
        };
        const auto draw = [&]() {
            Sample sample;
            sample.frame = profiler_.nextFrameIndex();
            const auto begin = Clock::now();
            sample.originNs = timestamp(begin);
            double previous = 0;
            const auto mark = [&](const char* name) {
                const double end = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
                sample.segments.push_back({{"name", name}, {"startMs", previous}, {"endMs", end}});
                previous = end;
            };
            {
                auto profile = profiler_.beginFrame(); mark("Profiler begin");
                if (!waitForFrameSlotBeforeInput()) { throw std::runtime_error("Frame-slot wait failed"); }
                mark("Frame-slot wait");
                {
                    const render::vulkan::StreamlineFrameScope streamlineFrame(
                        (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) == 0 && ImGui::GetPlatformIO().Viewports.Size <= 1,
                        &sample.frameBegin);
                    mark("Streamline begin");
                    { auto scope = profiler_.scope("Poll Events"); pollEvents(); }
                    if (!running_) { throw std::runtime_error("Benchmark cancelled"); }
                    mark("Poll events");
                    { auto scope = profiler_.scope("Render Frame");
                        if (!renderFrame() || viewportCompileFailed_) { throw std::runtime_error("Production frame failed"); }
                    }
                    mark("Render/submit/Present");
                }
                mark("Streamline end");
            }
            mark("Profiler end");
            const auto& stats = graphExecutor_->executionStats();
            sample.executionId = stats.executionId; sample.graphGeneration = stats.graphGeneration;
            if (stats.scheduling.enabled) { sample.scheduling = stats.scheduling; }
            return sample;
        };
        const auto drain = [&] {
            if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { throw std::runtime_error("Final GPU drain failed"); }
            auto stats = graphExecutor_->collectCompletedGpuExecutionStats();
            if (!stats) { throw std::runtime_error("GPU timing readback failed"); }
            for (const auto& execution : *stats) { profiler_.updateRenderGraphGpuStats(execution); }
        };
        const auto loading = Clock::now();
        uint32_t loadingFrames = 0;
        for (;;) {
            draw(); ++loadingFrames;
            const auto readiness = subsystemHost_.get<render::StreamerSubsystem>()->sceneReadiness();
            if (!pendingSceneLoad_.valid() && !pendingSceneResourcePreparation_ && viewportPreviewValid_ &&
                (producer->properties.value("sceneBinding", std::string("world")) == "asset"
                    ? readiness.ready && readiness.requiredPages != 0 : scene_.valid())) { break; }
            if (Clock::now() - loading > std::chrono::minutes(10)) { throw std::runtime_error("Scene readiness timed out"); }
        }
        report["loadingSeconds"] = std::chrono::duration<double>(Clock::now() - loading).count();
        report["loadingFrames"] = loadingFrames;
        report["deviceMemoryAfterLoad"] = memorySnapshot(device_->memoryBudget());
        report["gpuAfterLoad"] = gpuSnapshot(nvmlMonitor_);
        for (uint32_t frame = 0; frame < config.at("warmupFrames").get<uint32_t>(); ++frame) { draw(); }
        const auto reflex = render::vulkan::streamlineReflexStatus();
        report["reflex"] = {{"available", reflex.available}, {"suspended", reflex.suspended},
            {"requestedMode", static_cast<uint32_t>(reflex.options.mode)}, {"frameLimitUs", reflex.options.frameLimitUs}};
        if (config.contains("expectedReflexMode") &&
            config.at("expectedReflexMode").get<uint32_t>() != static_cast<uint32_t>(reflex.options.mode)) {
            throw std::runtime_error("Reflex mode differs from benchmark expectation");
        }
        if (config.contains("expectedReflexFrameLimitUs") &&
            config.at("expectedReflexFrameLimitUs").get<uint32_t>() != reflex.options.frameLimitUs) {
            throw std::runtime_error("Reflex frame limit differs from benchmark expectation");
        }
        report["deviceMemoryAfterWarmup"] = memorySnapshot(device_->memoryBudget());
        report["graph"] = Json::array();
        for (const auto& node : renderGraph_.nodes()) {
            auto properties = node.properties; properties.merge_patch(node.runtimeProperties);
            report["graph"].push_back({{"name", node.name}, {"type", node.type}, {"properties", properties}});
        }
        report["outputExtent"] = {viewportTextureWidth_, viewportTextureHeight_};
        const auto* depth = graphExecutor_->outputResource("VBuffer.depth");
        if (!depth) { throw std::runtime_error("Benchmark graph has no visibility depth"); }
        report["renderExtent"] = {depth->desc.width, depth->desc.height};
        const uint32_t frameCount = config.at("frames");
        const uint32_t memoryInterval = config.value("memorySampleIntervalFrames", 10u);
        if (!memoryInterval) { throw std::runtime_error("Memory sampling interval must be positive"); }
        std::vector<Sample> samples;
        samples.reserve(frameCount);
        profiler_.beginCapture();
        const auto measuredStart = Clock::now();
        for (uint32_t frame = 0; frame < frameCount; ++frame) {
            samples.push_back(draw());
            if (samples.size() > 1) {
                samples[samples.size() - 2].frameMs = double(samples.back().originNs - samples[samples.size() - 2].originNs) / 1.e6;
            }
            if (frame % memoryInterval == 0 || frame + 1 == frameCount) {
                samples.back().memory = memorySnapshot(device_->memoryBudget());
                samples.back().gpu = gpuSnapshot(nvmlMonitor_);
            }
            if (profiler_.captureOverflow() || viewportTextureWidth_ != fullRoamWidth_ ||
                viewportTextureHeight_ != fullRoamHeight_ ||
                (frame && samples.back().graphGeneration != samples.front().graphGeneration)) {
                throw std::runtime_error("Production capture topology/extent changed");
            }
        }
        samples.back().frameMs = double(timestamp(Clock::now()) - samples.back().originNs) / 1.e6;
        report["measuredSeconds"] = std::chrono::duration<double>(Clock::now() - measuredStart).count();
        profiler_.endCapture();
        drain();
        if (taskSink) {
            report["taskGraphWriteFailed"] = taskSink->writeFailed();
            if (taskSink->writeFailed()) { throw std::runtime_error("TaskGraph diagnostic export failed"); }
        }
        report["deviceMemoryAfterCapture"] = memorySnapshot(device_->memoryBudget());
        const auto& frames = profiler_.capturedFrames();
        if (frames.size() != samples.size()) { throw std::runtime_error("Profiler frame identity count mismatch"); }
        std::ofstream stream(output / "Frames.jsonl");
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        Json definitions = Json::array();
        std::map<std::string, uint32_t> ids;
        size_t missingGPU = 0;
        for (size_t i = 0; i < frames.size(); ++i) {
            const auto& frame = frames[i]; const auto& sample = samples[i];
            if (frame.index != sample.frame || frame.profilingOverflow) { throw std::runtime_error("Profiler frame mismatch/overflow"); }
            auto segments = sample.segments;
            segments.push_back({{"name", "Telemetry/bookkeeping"}, {"startMs", segments.back().at("endMs")}, {"endMs", sample.frameMs}});
            Json scopes = Json::array(), streaming = Json::array(); std::vector<std::string> paths;
            bool gpu = false;
            for (size_t n = 0; n < frame.nodes.size(); ++n) {
                const auto& node = frame.nodes[n];
                const auto path = (n && node.parent < n ? paths[node.parent] + "/" : "") + node.name;
                paths.push_back(path);
                auto [id, inserted] = ids.emplace(path, uint32_t(ids.size()));
                if (inserted) { definitions.push_back({{"id", id->second}, {"path", path}, {"cpuOnly", node.cpuOnly},
                    {"cpuIdle", node.cpuIdle}, {"cpuEditor", node.cpuEditor}}); }
                scopes.push_back({{"id", id->second}, {"cpuMs", node.cpuMilliseconds},
                    {"gpuMs", node.gpuTimingAvailable ? Json(node.gpuMilliseconds) : Json(nullptr)},
                    {"executionId", node.renderGraphExecutionId == UINT64_MAX ? Json(nullptr) : Json(node.renderGraphExecutionId)}});
                if (node.name == "RenderGraph GPU envelope") { gpu = node.gpuTimingAvailable; }
            }
            for (const auto& state : frame.streaming) {
                streaming.push_back({{"pass", state.passName}, {"asset", state.assetPath},
                    {"frame", state.frameIndex}, {"feedbackFrame", state.feedbackFrame},
                    {"residentPages", state.residentPages}, {"pendingPages", state.pendingPages},
                    {"geometryBytes", state.geometryUsedBytes}, {"clasAllocatedBytes", state.clasAllocatedBytes},
                    {"textureResidentBytes", state.textureResidentBytes}, {"texturePendingBytes", state.texturePendingBytes},
                    {"texturePendingImages", state.texturePendingImages}, {"textureRefinedImages", state.textureRefinedImages},
                    {"loadFailures", state.loadFailures}, {"allocationFailures", state.allocationFailures}});
            }
            if (!gpu) { ++missingGPU; }
            const auto& frameBegin = sample.frameBegin;
            stream << Json{{"frame", frame.index}, {"originSteadyNs", sample.originNs}, {"frameMs", sample.frameMs},
                {"cpuFramePathContract", "nonoverlapping-coordinator-wall-segments-v1"},
                {"cpuFramePathSegmentsMs", segments}, {"executionId", sample.executionId},
                {"graphGeneration", sample.graphGeneration}, {"scopes", scopes},
                {"scheduling", sample.scheduling}, {"streaming", streaming},
                {"streamlineBegin", {{"active", frameBegin.active},
                    {"effectiveMode", static_cast<uint32_t>(frameBegin.effectiveOptions.mode)},
                    {"frameLimitUs", frameBegin.effectiveOptions.frameLimitUs}, {"sleepCalled", frameBegin.sleepCalled},
                    {"totalMs", frameBegin.totalMs}, {"sleepMs", frameBegin.sleepMs},
                    {"available", frameBegin.cachedStatus.available}, {"suspended", frameBegin.cachedStatus.suspended}}},
                {"deviceMemory", sample.memory}, {"gpu", sample.gpu}}.dump() << '\n';
        }
        stream.close(); report["scopes"] = definitions; report["frames"] = frames.size(); report["missingGpuFrames"] = missingGPU;
        if (missingGPU) { throw std::runtime_error("Missing completed GPU frame timings"); }
        if (config.value("captureImage", false)) { report["image"] = readFinalImage(*device_, *graphicsQueue_, *graphExecutor_, activePreviewOutput_, output); }
        report["status"] = "passed";
        passed = true;
    } catch (const std::exception& error) {
        report["error"] = error.what(); spdlog::error("[CPU Jobs] {}", error.what());
    }
    if (sinkToken) { task::taskSystem().unsubscribe(sinkToken); }
    fullRoamActive_ = false; fullRoamWidth_ = fullRoamHeight_ = 0;
    std::ofstream reportFile(output / "Capture.json"); reportFile << report.dump(2) << '\n';
    return passed && bool(reportFile);
}

} // namespace metallic
