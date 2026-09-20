#include "Editor/EditorApplication.h"
#include "Editor/EditorRasterWorkload.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "imgui.h"
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace metallic {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

Json streamSample(const render::SceneStreamingProfile& s)
{
    return {{"softwareRaster",s.softwareRasterIdentity.empty() ? Json(nullptr) : Json::parse(s.softwareRasterIdentity)},
        {"frame",s.frameIndex},{"feedbackFrame",s.feedbackFrame},{"generation",s.generation},
        {"geometryBytes",s.geometryUsedBytes},{"geometryCapacity",s.geometryBudgetBytes},
        {"clasBytes",s.clasUsedBytes},{"clasCapacity",s.clasCapacityBytes},{"clasScratchBytes",s.clasScratchBytes},
        {"clasBuiltClusters",s.clasBuiltClusters},{"clasMovedClusters",s.clasMovedClusters},
        {"clasPendingPages",s.clasPendingPages},{"residentPages",s.residentPages},{"pendingPages",s.pendingPages},
        {"ioQueued",s.ioQueued},{"ioActive",s.ioActive},{"uploadQueued",s.uploadQueued},
        {"requests",s.requests},{"uploads",s.uploads},{"evictions",s.evictions},{"uploadBytes",s.uploadBytes},
        {"requestOverflows",s.requestOverflows},{"allocationFailures",s.allocationFailures},{"loadFailures",s.loadFailures},
        {"blasFeedbackAvailable",s.blasFeedbackAvailable},{"blasFeedbackFrame",s.blasFeedbackFrame},
        {"blasBuildCount",s.blasBuildCount},{"blasClusterReferences",s.blasClusterReferences},{"blasOverflowCount",s.blasOverflowCount},
        {"textureResidentBytes",s.textureResidentBytes},{"textureRetiredBytes",s.textureRetiredBytes},
        {"texturePendingBytes",s.texturePendingBytes},{"textureBudgetBytes",s.textureBudgetBytes},
        {"textureRefinedImages",s.textureRefinedImages},{"textureRequestedImages",s.textureRequestedImages},
        {"textureUpgrades",s.textureUpgrades},{"textureDowngrades",s.textureDowngrades},
        {"textureBudgetDeferrals",s.textureBudgetDeferrals},{"textureMaxRequestFrames",s.textureMaxRequestFrames},
        {"lastTextureUploadSequence",s.textureUpload.sequence}};
}

Json uploadSample(const render::TextureUploadProfile& s)
{
    return {{"sequence",s.sequence},{"requestFrame",s.requestFrame},{"submitFrame",s.submitFrame},
        {"completionFrame",s.completionFrame},{"bytes",s.bytes},{"images",s.images},{"queue","graphics"},
        {"gpuMs",s.gpuTimingAvailable ? Json(s.gpuMilliseconds) : Json(nullptr)},
        {"completionObservedMs",s.completionObservedMilliseconds}};
}
} // namespace

bool EditorApplication::runZorahFullRoamBenchmark()
{
    const char* outputEnvironment = std::getenv("METALLIC_FULL_ROAM_OUTPUT");
    const std::filesystem::path output = outputEnvironment ? std::filesystem::path(outputEnvironment) :
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Captures" / "FullRoam" /
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    if (std::filesystem::exists(output / "Capture.json")) {
        spdlog::error("[Full Roam] Capture already exists: {}", output.string());
        return false;
    }
    Json report{{"protocol","zorah-full-editor-roam-v1"},{"status","failed"},
        {"frameTimeDefinition","start-to-start editor loop; includes wait, events, recording, submit, present and capture bookkeeping"},
        {"gpuTimeDefinition","RenderGraph envelope excludes editor composite/present and independent texture submissions; do not sum nested scopes"},
        {"textureLatencyDefinition","request enqueue to publish frames; excludes pre-admission budget waiting"}};
    fullRoamActive_ = true;
    fullRoamWidth_ = viewportTextureWidth_;
    fullRoamHeight_ = viewportTextureHeight_;
    bool passed = false;
    RasterWorkloadObserver workloadObserver;
    try {
        std::filesystem::create_directories(output);
        Json config = Json::object();
        if (const char* path = std::getenv("METALLIC_FULL_ROAM_CONFIG")) {
            std::ifstream input(path);
            config = Json::parse(input);
        }
        if (config.value("rasterComparison", false)) { return runZorahFullRasterComparison(config, output); }
        const uint32_t workloadEvery = config.value("workloadEvery", 0u);
        if (workloadEvery && workloadEvery < 10) { throw std::runtime_error("workloadEvery must be 0 or at least 10"); }
        report["workloadEvery"] = workloadEvery;
        report["diagnosticRun"] = workloadEvery != 0;
        const double duration = config.value("durationSeconds",180.0);
        const double warmup = config.value("warmupSeconds",5.0);
        const double distance = config.value("distance",6.0);
        if (!std::isfinite(duration) || duration < 1 || duration > 1800 ||
            !std::isfinite(warmup) || warmup < 0 || warmup > 120 ||
            !std::isfinite(distance) || distance < 0 || distance > 100) { throw std::runtime_error("Invalid route configuration"); }
        if (config.contains("width") || config.contains("height")) {
            const int width = config.value("width",0), height = config.value("height",0);
            if (width < 64 || height < 64 || width > 8192 || height > 8192) { throw std::runtime_error("Explicit viewport requires width and height in [64,8192]"); }
            fullRoamWidth_ = uint32_t(width); fullRoamHeight_ = uint32_t(height);
        }
        // Fractions of route duration, forward distance, yaw degrees. Actual
        // absolute camera keyframes are exported for exact reuse and inspection.
        const Json points = config.value("keyframes",Json::array({
            {{"t",0.0},{"forward",0.0},{"yaw",0.0},{"stage","static"}},
            {{"t",1.0/12},{"forward",0.0},{"yaw",0.0},{"stage","first-visit"}},
            {{"t",1.0/3},{"forward",1.0},{"yaw",0.0},{"stage","near-wall"}},
            {{"t",4.0/9},{"forward",1.0},{"yaw",65.0},{"stage","turn"}},
            {{"t",2.0/3},{"forward",1.0},{"yaw",-65.0},{"stage","return"}},
            {{"t",8.0/9},{"forward",0.0},{"yaw",0.0},{"stage","settle"}},
            {{"t",1.0},{"forward",0.0},{"yaw",0.0},{"stage","end"}}}));
        if (!points.is_array() || points.size() < 2 || points.size() > 128 ||
            points.front().at("t").get<double>() != 0 || points.back().at("t").get<double>() != 1) {
            throw std::runtime_error("Route keyframes must span [0,1]");
        }
        double lastT = -1;
        for (const auto& point : points) {
            const double t=point.at("t"), f=point.at("forward"), yaw=point.at("yaw");
            if (!std::isfinite(t) || !std::isfinite(f) || !std::isfinite(yaw) || t <= lastT || t > 1 ||
                std::abs(f)>10 || std::abs(yaw)>360 || !point.at("stage").is_string()) { throw std::runtime_error("Invalid route keyframe"); }
            lastT=t;
        }
        if (workloadEvery) { graphExecutor_->setDebugObserver(&workloadObserver); }
        loadBuiltInSample(render::kGPUDrivenZorahFullSampleId);
        const auto workloadSetting = [&](bool enabled) {
            const auto* node = renderGraph_.findNode("VBuffer");
            if (!node) { throw std::runtime_error("Missing VBuffer"); }
            renderGraph_.setNodeRuntimeProperty(node->id, "softwareRasterWorkload", enabled);
        };
        const auto draw = [&]() {
            auto frame = profiler_.beginFrame();
            if (!waitForFrameSlotBeforeInput()) { return false; }
            const render::vulkan::StreamlineFrameScope streamlineFrame(
                (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED)==0 && ImGui::GetPlatformIO().Viewports.Size<=1);
            { auto scope = profiler_.scope("Poll Events"); pollEvents(); }
            if (!running_ || SDL_GetKeyboardState(nullptr)[SDL_SCANCODE_ESCAPE]) { return false; }
            auto scope = profiler_.scope("Render Frame");
            return renderFrame() && !viewportCompileFailed_;
        };
        const auto loadingStart = Clock::now();
        auto readyAt = Clock::time_point{};
        uint32_t loadingFrames = 0;
        while (true) {
            if (!draw()) { throw std::runtime_error("Full loading or rendering failed/cancelled"); }
            ++loadingFrames;
            const auto readiness = subsystemHost_.get<render::StreamerSubsystem>()->sceneReadiness();
            const auto now=Clock::now();
            if (readiness.ready && readiness.requiredPages && viewportPreviewValid_) {
                if (readyAt == Clock::time_point{}) { readyAt=now; }
                if (std::chrono::duration<double>(now-readyAt).count() >= warmup) { break; }
            }
            if (std::chrono::duration<double>(now-loadingStart).count()>600) { throw std::runtime_error("Full readiness timed out"); }
        }
        if (workloadEvery) { graphExecutor_->setDebugObserver(nullptr); }
        if (!fullRoamWidth_) { fullRoamWidth_=viewportTextureWidth_; fullRoamHeight_=viewportTextureHeight_; }
        const auto original = viewportCameraProperties();
        const auto cameraAt = [&](double forward, double yaw) {
            auto camera = original;
            const auto& base = original.at("camera");
            const double dx=base["center"][0].get<double>()-base["eye"][0].get<double>();
            const double dz=base["center"][2].get<double>()-base["eye"][2].get<double>();
            const double length=std::hypot(dx,dz);
            if (length<1e-6) { throw std::runtime_error("Route requires a horizontal view direction"); }
            const double angle=yaw*3.14159265358979323846/180;
            camera["camera"]["eye"][0]=base["eye"][0].get<double>()+dx/length*distance*forward;
            camera["camera"]["eye"][2]=base["eye"][2].get<double>()+dz/length*distance*forward;
            camera["camera"]["center"][0]=camera["camera"]["eye"][0].get<double>()+std::cos(angle)*dx+std::sin(angle)*dz;
            camera["camera"]["center"][2]=camera["camera"]["eye"][2].get<double>()-std::sin(angle)*dx+std::cos(angle)*dz;
            return camera;
        };
        report["config"]={{"durationSeconds",duration},{"warmupSeconds",warmup},{"distance",distance},{"keyframes",points}};
        report["absoluteKeyframes"]=Json::array();
        for (const auto& p : points) { report["absoluteKeyframes"].push_back({{"seconds",p.at("t").get<double>()*duration},
            {"stage",p.at("stage")},{"camera",cameraAt(p.at("forward"),p.at("yaw"))}}); }
        report["outputExtent"]={fullRoamWidth_,fullRoamHeight_};
        auto* depth=graphExecutor_->outputResource("VBuffer.depth");
        if (!depth) { throw std::runtime_error("Missing VBuffer render extent"); }
        report["renderExtent"]={depth->desc.width,depth->desc.height};
        report["hidden"]=std::getenv("METALLIC_FULL_ROAM_HIDDEN")!=nullptr;
        report["vsync"]=true;
        report["frameSlots"]=frameSlots_.size();
        report["loadingFrames"]=loadingFrames;
        report["loadingSeconds"]=std::chrono::duration<double>(readyAt-loadingStart).count();
        report["validationRequested"]=debugRuntime_ && std::getenv("METALLIC_DEBUG_VALIDATION");
        report["graph"]=Json::array();
        for (const auto& node : renderGraph_.nodes()) { auto properties=node.properties; properties.merge_patch(node.runtimeProperties);
            report["graph"].push_back({{"name",node.name},{"type",node.type},{"properties",properties}}); }
        const auto graphGeneration = graphExecutor_->executionStats().graphGeneration;
        struct Sample { uint64_t frame; double seconds, ms; size_t segment; uint64_t availableBytes; Json graphPreparation; bool diagnostic = false; };
        std::vector<Sample> samples;
        samples.reserve(size_t(duration*120));
        profiler_.beginCapture();
        const auto start=Clock::now();
        auto previous=start;
        while (true) {
            const auto now=Clock::now();
            const double seconds=std::chrono::duration<double>(now-start).count();
            if (!samples.empty()) { samples.back().ms=std::chrono::duration<double,std::milli>(now-previous).count(); }
            if (seconds>=duration) { break; }
            previous=now;
            size_t segment=0;
            while (segment+2<points.size() && seconds/duration >= points[segment+1]["t"].get<double>()) { ++segment; }
            const auto& a=points[segment]; const auto& b=points[segment+1];
            const double u=std::clamp((seconds/duration-a["t"].get<double>())/(b["t"].get<double>()-a["t"].get<double>()),0.0,1.0);
            const auto blend=[&](const char* key) { return a[key].get<double>()*(1-u)+b[key].get<double>()*u; };
            applyViewportCameraProperties(cameraAt(blend("forward"),blend("yaw")),nullptr);
            samples.push_back({profiler_.nextFrameIndex(),seconds,0,segment,device_->memoryBudget().availableBytes});
            const bool diagnostic = workloadEvery && (samples.size() - 1) % workloadEvery == 0;
            samples.back().diagnostic = diagnostic;
            if (workloadEvery) {
                workloadSetting(diagnostic);
                workloadObserver.capture = diagnostic;
                workloadObserver.editorFrame = samples.back().frame;
                workloadObserver.camera = viewportCameraProperties();
                graphExecutor_->setDebugObserver(diagnostic ? &workloadObserver : nullptr);
            }
            if (!draw()) { throw std::runtime_error("Roam rendering failed/cancelled"); }
            const auto& execution = graphExecutor_->executionStats();
            samples.back().graphPreparation = {{"executionId", execution.executionId},
                {"drainReasonMask", execution.drainReasonMask}, {"externalCompletionCount", execution.externalCompletionCount},
                {"overlapBlockingPasses", execution.overlapBlockingPasses}};
            if (profiler_.captureOverflow()) { throw std::runtime_error("Profiler capture limit exceeded"); }
            if (viewportTextureWidth_!=fullRoamWidth_ || viewportTextureHeight_!=fullRoamHeight_ ||
                graphExecutor_->executionStats().graphGeneration!=graphGeneration) { throw std::runtime_error("Viewport or graph changed during capture"); }
        }
        profiler_.endCapture();
        // Resolve outstanding queries after measurement; no per-frame readback wait.
        if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { throw std::runtime_error("GPU drain failed"); }
        std::vector<render::RenderGraphExecutionStats> completed;
        if (!graphExecutor_->collectCompletedGpuExecutionStats(completed)) { throw std::runtime_error("GPU query resolve failed"); }
        for (const auto& stats : completed) { profiler_.updateRenderGraphGpuStats(stats); }
        if (workloadEvery) {
            report["workloads"] = workloadObserver.takeAfterDrain();
            graphExecutor_->setDebugObserver(nullptr);
            workloadSetting(false);
        }
        const auto& frames=profiler_.capturedFrames();
        if (frames.size()!=samples.size()) { throw std::runtime_error("CPU frame count mismatch"); }
        std::ofstream frameFile(output/"Frames.jsonl"), uploadFile(output/"Uploads.jsonl");
        frameFile.exceptions(std::ios::badbit|std::ios::failbit); uploadFile.exceptions(std::ios::badbit|std::ios::failbit);
        std::map<std::string,uint32_t> scopeIds;
        std::set<std::pair<uint64_t,uint64_t>> uploads;
        Json definitions=Json::array();
        size_t missingGpu=0;
        for (size_t i=0; i<frames.size(); ++i) {
            const auto& f=frames[i]; const auto& sample=samples[i];
            if (f.index!=sample.frame || f.profilingOverflow) { throw std::runtime_error("Frame identity/section overflow"); }
            Json row{{"frame",f.index},{"seconds",sample.seconds},{"frameMs",sample.ms},
                {"stage",points[sample.segment]["stage"]},{"availableBytes",sample.availableBytes},
                {"graphPreparation",sample.graphPreparation},{"diagnostic",sample.diagnostic},
                {"scopes",Json::array()},{"streaming",Json::array()}};
            std::vector<std::string> paths;
            bool graphGpu=false;
            for (size_t n=0; n<f.nodes.size(); ++n) {
                const auto& node=f.nodes[n];
                const std::string path=(n && node.parent<n ? paths[node.parent]+"/" : "")+node.name;
                paths.push_back(path);
                auto [it,inserted]=scopeIds.emplace(path,uint32_t(scopeIds.size()));
                if (inserted) { definitions.push_back({{"id",it->second},{"path",path},{"cpuOnly",node.cpuOnly},
                    {"queue",(node.cpuOnly || node.renderGraphExecutionId==UINT64_MAX) ? "cpu" : node.queue==render::QueueType::Compute ? "compute" : node.queue==render::QueueType::Copy ? "copy" : "graphics"}}); }
                row["scopes"].push_back({{"id",it->second},{"cpuMs",node.cpuMilliseconds},
                    {"gpuMs",node.gpuTimingAvailable ? Json(node.gpuMilliseconds) : Json(nullptr)},
                    {"executionId",node.renderGraphExecutionId==UINT64_MAX ? Json(nullptr) : Json(node.renderGraphExecutionId)}});
                if (node.name=="RenderGraph GPU envelope") { graphGpu=node.gpuTimingAvailable; }
            }
            if (!graphGpu) { ++missingGpu; }
            for (const auto& s : f.streaming) {
                row["streaming"].push_back(streamSample(s));
                if (s.textureUpload.sequence && uploads.insert({s.generation,s.textureUpload.sequence}).second) {
                    auto upload=uploadSample(s.textureUpload); upload["streamGeneration"]=s.generation;
                    uploadFile<<upload.dump()<<'\n';
                }
            }
            frameFile<<row.dump()<<'\n';
        }
        frameFile.close(); uploadFile.close();
        report["scopes"]=std::move(definitions);
        report["frames"]=frames.size(); report["missingGpuFrames"]=missingGpu;
        report["uploadSamples"]=uploads.size(); report["graphGeneration"]=graphGeneration;
        if (missingGpu) { throw std::runtime_error("Incomplete GPU timing coverage"); }
        report["status"]="capture_complete";
        passed=true;
    } catch (const std::exception& error) {
        report["error"]=error.what();
        spdlog::error("[Full Roam] {}",error.what());
    }
    profiler_.endCapture();
    // Keep readback allocations alive until all submitted copies complete, also on failure.
    if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { passed = false; report["status"] = "failed"; }
    graphExecutor_->setDebugObserver(debugRuntime_.get());
    fullRoamActive_=false; fullRoamWidth_=fullRoamHeight_=0;
    profiler_.beginCapture(); profiler_.endCapture(); // Release retained capture samples after export.
    try {
        std::ofstream file(output/"Capture.json");
        file.exceptions(std::ios::badbit|std::ios::failbit);
        file<<report.dump(2)<<'\n';
    } catch (const std::exception& error) { spdlog::error("[Full Roam] Export failed: {}",error.what()); passed=false; }
    spdlog::info("[Full Roam] {}: {}",passed ? "Capture complete" : "Failed",output.string());
    return passed;
}
} // namespace metallic
