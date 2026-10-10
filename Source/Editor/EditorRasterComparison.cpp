#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Profiling/NvPerf.h"
#include "Runtime/Render/Profiling/WorkControlReplay.h"
#include "Editor/EditorApplication.h"
#include "Editor/EditorRasterWorkload.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "imgui.h"
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace metallic {
namespace {
using namespace render;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
void checkRaster(bool ok, const char* message)
{
    if (!ok) { throw std::runtime_error(message); }
}
std::string rasterHash(const void* data, size_t bytes)
{
    uint64_t hash = 14695981039346656037ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) { hash = (hash ^ p[i]) * 1099511628211ull; }
    return std::to_string(hash);
}
// Attached only to diagnostic frames (and initial compilation to opt into
// transfer-source resource usage). No readbacks/snapshots in measured frames.
class RasterComparisonObserver final : public IRenderDebugObserver {
public:
    bool capture = false;
    RasterWorkloadObserver workload;
    Device* device = nullptr;
    std::map<std::string, std::unique_ptr<Buffer>> copies;
    Json dispatches = Json::array();
    Json graph;
    WorkControlShaderTrace* trace = nullptr;
    void compiled(Json value) override { graph=std::move(value); }
    void beginExecution(Device& d, debug::DebugEvidenceStamp stamp, RenderSubsystemHost* host) override
    {
        device=&d; workload.beginExecution(d,stamp,host);
        graph["id"]=stamp.graph; graph["generation"]=stamp.generation;
        if (trace) { trace->beginExecution(stamp); }
    }
    void endExecution(bool) override {}
    void boundary(CommandBuffer& commands, std::string_view checkpoint, uint32_t,
        std::string_view pass, std::span<const DebugResourceBinding> resources, const Json& values) override
    {
        if (!capture || pass != "VBuffer") { return; }
        if (checkpoint == "BeforeStreamEarlySoftware" || checkpoint == "BeforeStreamLateSoftware") {
            auto binding = values;
            if (trace && trace->bind(commands,pass,values)) {
                binding["scope"]="diagnostic-dispatch"; binding["instrumentation"]="Printf";
                binding["dispatchToken"]=trace->plan().at("dispatchToken");
                binding["uninstrumentedSpirvFnv1a64"]=binding.at("spirvFnv1a64"); binding.erase("spirvFnv1a64");
                binding["diagnosticCompilerSpirvSha256"]=trace->variant().at("compilerSpirvSha256");
            }
            dispatches.push_back(std::move(binding));
        }
        workload.capture = true;
        workload.boundary(commands, checkpoint, 0, pass, resources, values);
        for (const auto& resource : resources) {
            std::string key;
            uint64_t bytes = 0;
            if (checkpoint == "AfterTraversal" && resource.id == "streaming.VBuffer.activeHeader") { key = "header"; }
            if (checkpoint == "AfterTraversal" && resource.id == "streaming.VBuffer.activeGroups") { key = "groups"; }
            if (checkpoint == "AfterTraversal" && resource.id == "streaming.VBuffer.pageTable") { key = "pages"; }
            if ((checkpoint == "AfterStreamEarlyBins" || checkpoint == "AfterStreamLateBins" || checkpoint == "AfterStreamEarlyClusterCull" || checkpoint == "AfterStreamLateClusterCull") && resource.id == "hybrid.VBuffer.clusters") {
                key = checkpoint;
                bytes = checkpoint.ends_with("Bins") ? resource.size : 64;
            }
            if ((checkpoint == "AfterStreamEarlyBins" || checkpoint == "AfterStreamLateBins") && resource.id == "hybrid.VBuffer.arguments") {
                key = std::string(checkpoint) + ".arguments"; bytes = resource.size;
            }
            if (checkpoint == "AfterPass" && (resource.id == "VBuffer.visibility" || resource.id == "VBuffer.depth")) { key = resource.id; }
            if (key.empty()) { continue; }
            if (resource.texture) {
                checkRaster(resource.texture->desc().format == Format::R32Uint || resource.texture->desc().format == Format::D32Sfloat,
                    "Unexpected visibility/depth format");
                bytes = uint64_t(resource.texture->desc().width) * resource.texture->desc().height * 4;
            } else if (!bytes) { bytes = resource.size ? resource.size : resource.buffer->desc().size; }
            auto& copy = copies[key];
            if (!copy || copy->desc().size != bytes) {
                checkRaster(bool(device->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { copy = std::move(rhiValue); })), "Cannot allocate raster diagnostic readback");
            }
            if (resource.texture) {
                TextureBarrierDesc barrier{
                    .texture = resource.texture,
                    .oldLayout = metallic::render::textureLayoutForResourceState(resource.state),
                    .newLayout = TextureLayout::TransferSource,
                    .before = metallic::render::resourceSyncScope(resource.state, metallic::render::PipelineStageBits::AllCommands),
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                };
                if (auto commandResult = commands.synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
                if (auto commandResult = (copy.get())->slice().and_then([&](const auto& bufferSlice) { return commands.copyTextureToBuffer({.texture = resource.texture, .buffer = bufferSlice,
                    .width = resource.texture->desc().width, .height = resource.texture->desc().height}); }); !commandResult) { throw std::runtime_error(std::string("copyTextureToBuffer failed: ") + metallic::render::resultToString(commandResult)); }
                std::swap(barrier.before, barrier.after); std::swap(barrier.oldLayout, barrier.newLayout);
                if (auto commandResult = commands.synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
            } else {
                BufferBarrierDesc barrier{
                    .buffer = resource.buffer,
                    .before = metallic::render::resourceSyncScope(resource.state, metallic::render::PipelineStageBits::AllCommands),
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                    .range = {.offset = resource.offset, .size = bytes},
                };
                if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
                {
                    auto sourceSlice = resource.buffer->slice({resource.offset, bytes});
                    if (!sourceSlice) { throw std::runtime_error(std::string("source slice failed: ") + metallic::render::resultToString(sourceSlice)); }
                    auto destinationSlice = copy.get()->slice({0, bytes});
                    if (!destinationSlice) { throw std::runtime_error(std::string("destination slice failed: ") + metallic::render::resultToString(destinationSlice)); }
                    if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { throw std::runtime_error(std::string("copyBuffer failed: ") + metallic::render::resultToString(commandResult)); }
                }
                std::swap(barrier.before, barrier.after);
                if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
            }
        }
    }
    template<typename T> std::vector<T> read(const std::string& name)
    {
        checkRaster(copies.contains(name), ("Missing raster snapshot: " + name).c_str());
        auto& b = copies.at(name);
        b->invalidate();
        const void* data = b->map();
        checkRaster(data != nullptr, "Cannot map raster snapshot");
        std::vector<T> result(b->desc().size / sizeof(T));
        std::memcpy(result.data(), data, result.size() * sizeof(T));
        b->unmap();
        return result;
    }
    Json snapshot(const std::filesystem::path& output, const std::string& name)
    {
        const auto header = read<MeshletStreamGPUActiveHeader>("header").front();
        auto groups = read<MeshletStreamGPUActiveGroup>("groups");
        checkRaster(header.activeGroupCount <= groups.size(), "Active cut overflow");
        groups.resize(header.activeGroupCount);
        const auto pages = read<StreamPageTableEntry>("pages");
        std::vector<uint32_t> mappings;
        for (const auto& page : pages) { mappings.push_back(page.deviceOffsetAndState); }
        Json result{{"activeGroups",groups.size()},{"cutHash",rasterHash(groups.data(),groups.size()*sizeof(groups[0]))},
            {"pageMappingsHash",rasterHash(mappings.data(),mappings.size()*4)},
            {"hashAlgorithm","FNV1a64; page mappings exclude mutable lastRequestFrame"}};
        for (const auto* phase : {"AfterStreamEarlyBins","AfterStreamLateBins"}) {
            const auto bins = read<uint32_t>(phase);
            const size_t first = 16ull + 4ull * bins[5];
            checkRaster(bins[4] <= bins[5] && first + bins[4] <= bins.size(), "Software bin list overflow");
            const auto arguments = read<uint32_t>(std::string(phase) + ".arguments");
            checkRaster(arguments.size() >= 15, "Missing software indirect arguments");
            result[phase] = {{"hardwareClusters",bins[0]+bins[1]+bins[2]+bins[3]},
                {"softwareClusters",bins[4]},{"candidates",bins[12]},{"capacity",bins[5]},
                {"softwareListHash",rasterHash(bins.data()+first, size_t(bins[4])*4)},
                {"softwareDispatch",{arguments[12],arguments[13],arguments[14]}}};
        }
        for (const auto* phase : {"AfterStreamEarlyClusterCull","AfterStreamLateClusterCull"}) {
            const auto counters = read<uint32_t>(phase);
            result[phase] = {{"exactClusters",counters[0]},{"fastSoftware",counters[1]},
                {"fastHardware",counters[2]},{"candidateOverflow",counters[14]}};
            checkRaster(counters[14] == 0, "Raster candidate overflow");
        }
        for (const auto* resource : {"VBuffer.visibility","VBuffer.depth"}) {
            const auto data = read<uint32_t>(resource);
            const auto file = name + "-" + resource + ".bin";
            std::ofstream stream(output/file,std::ios::binary);
            stream.exceptions(std::ios::badbit|std::ios::failbit);
            stream.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*4));
            result[resource] = {{"file",file},{"hash",rasterHash(data.data(),data.size()*4)},{"pixels",data.size()}};
        }
        result["workloads"] = workload.takeAfterDrain();
        result["productionDispatches"] = std::move(dispatches);
        dispatches = Json::array();
        return result;
    }
};
} // namespace

bool EditorApplication::runZorahFullRasterComparison(const Json& config, const std::filesystem::path& output)
{
    Json report{{"protocol","zorah-full-raster-comparison-v1"},{"status","failed"},{"cases",Json::array()},
        {"scope","Frozen geometry/CLAS/cut/TLAS and texture publication; hidden full editor graph; jitter disabled for exact camera samples"}};
    RasterComparisonObserver observer;
    bool passed = false;
    bool traceActive = false;
    profiling::NvPerfSession nvPerf;
    const bool nvPerfRequested = profiling::nvPerfRequested();
    const auto drain = [&]() {
        checkRaster(bool(frameSubmissions_.wait()) && bool(graphExecutor_->waitForSubmittedWork()), "Raster benchmark GPU drain failed");
        std::vector<RenderGraphExecutionStats> completed;
        checkRaster(bool(graphExecutor_->collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })), "Raster benchmark query resolve failed");
        for (const auto& stats : completed) { profiler_.updateRenderGraphGpuStats(stats); }
    };
    try {
        checkRaster(config.contains("shaderTrace") == bool(shaderTrace_),"Shader trace startup/config mismatch");
        if (shaderTrace_) { shaderTrace_->qualify(*device_,*graphicsQueue_,output); }
        const uint32_t samples = config.value("sampleFrames",64u);
        const uint32_t settle = config.value("settleFrames",8u);
        const uint32_t rounds = config.value("rounds",3u);
        const bool workloadCase = config.contains("workloadCase");
        const bool groupComparison = config.value("swGroupComparison", false);
        const double roamSeconds = config.value("swGroupRoamSeconds", 0.0);
        checkRaster(std::isfinite(roamSeconds) && roamSeconds >= 0 && roamSeconds <= 600 &&
            (!roamSeconds || groupComparison), "Invalid group correctness roam duration");
        const std::map<std::string, uint32_t> registeredVariants{{"swLegacy",10},{"swPrepared",11},
            {"swPlane",12},{"swCooperative",13},{"swWorkBins",14},{"swWorkControl",15}};
        uint32_t selectedMode = 0;
        if (workloadCase) {
            const auto& selection = config.at("workloadCase");
            checkRaster(selection.is_object() && selection.contains("id") && selection.at("id").is_string() &&
                !selection.at("id").get<std::string>().empty(), "Workload case needs a nonempty id");
            const std::string variant = selection.at("variant").get<std::string>();
            checkRaster(registeredVariants.contains(variant), "Unregistered workload variant");
            checkRaster(selection.value("scope", std::string{}) == "in-frame-early-late", "Unsupported workload scope");
            selectedMode = registeredVariants.at(variant);
            report["protocol"] = "metallic-workload-case-v1";
            report["workloadCase"] = selection;
        }
        const double profileHoldSeconds = config.value("profileHoldSeconds", 0.0);
        const uint32_t traceFrames = config.value("nsightTraceFrames", 0u);
        const bool primeHistory = config.contains("primeCameraOffset");
        if (primeHistory) {
            const auto& offset = config.at("primeCameraOffset");
            checkRaster((workloadCase || groupComparison) && offset.is_array() && offset.size() == 3 &&
                profileHoldSeconds == 0 && traceFrames <= 1, "Invalid history priming configuration");
            bool nonzero = false;
            for (const auto& component : offset) {
                checkRaster(component.is_number() && std::isfinite(component.get<double>()) &&
                    std::abs(component.get<double>()) <= 100, "Invalid history camera offset");
                nonzero |= component.get<double>() != 0;
            }
            checkRaster(nonzero, "History camera offset must be nonzero");
        }
        checkRaster(traceFrames <= 3 && (!traceFrames || (workloadCase && rounds == 1 && profileHoldSeconds == 0)),
            "SDK trace needs one workload round, 1-3 frames and no timed hold");
        checkRaster(std::isfinite(profileHoldSeconds) && profileHoldSeconds >= 0 && profileHoldSeconds <= 300,
            "Invalid SW profiler hold duration");
        const uint32_t width = config.value("width",1797u), height = config.value("height",660u);
        checkRaster(samples >= 8 && samples <= 512 && settle >= 2 && settle <= 120 && rounds >= 1 && rounds <= 3,
            "Invalid raster comparison sample count");
        checkRaster(width >= 64 && height >= 64 && width <= 8192 && height <= 8192,"Invalid raster comparison extent");
        fullRoamWidth_ = width; fullRoamHeight_ = height; fullRoamActive_ = true;
        drain();
        graphExecutor_->setDebugObserver(&observer);
        const std::string sampleId = config.value("sampleId", std::string(kGPUDrivenZorahFullSampleId));
        checkRaster(sampleId == kGPUDrivenZorahFullSampleId || sampleId == kDefaultGPUDrivenSampleId,
            "Unregistered workload sample");
        loadBuiltInSample(sampleId.c_str());
        auto view = viewportCameraProperties();
        if (config.contains("benchmarkCamera")) {
            const auto& camera = config.at("benchmarkCamera");
            checkRaster(camera.is_object() && camera.contains("eye") && camera.contains("center"), "Invalid benchmark camera");
            for (const auto* key : {"eye", "center"}) {
                checkRaster(camera[key].is_array() && camera[key].size() == 3, "Invalid benchmark camera vector");
                for (const auto& component : camera[key]) {
                    checkRaster(component.is_number() && std::isfinite(component.get<double>()), "Nonfinite benchmark camera");
                }
                view["camera"][key] = camera[key];
            }
        }
        view["temporalJitter"] = false;
        applyViewportCameraProperties(view,nullptr);
        viewportView_.setTemporalJitter(false);
        const auto set = [&](const char* node, const char* key, Json value) {
            const auto* found = renderGraph_.findNode(node);
            checkRaster(found != nullptr,"Missing raster comparison graph node");
            renderGraph_.setNodeRuntimeProperty(found->id,key,std::move(value));
        };
        set("VBuffer","debugStreamingPages",false);
        set("VBuffer","temporalJitter",false);
        set("VBuffer","asyncSoftwareRaster",false);
        bool reapplyCamera = false;
        const auto draw = [&]() {
            if (reapplyCamera) { applyViewportCameraProperties(viewportCameraProperties(), nullptr); }
            checkRaster(!viewportView_.temporalJitter(), "Raster comparison jitter was re-enabled");
            auto frame = profiler_.beginFrame();
            checkRaster(waitForFrameSlotBeforeInput(),"Frame slot wait failed");
            vulkan::StreamlineFrameBeginProfile frameBegin;
            const vulkan::StreamlineFrameScope streamlineFrame(
                (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED)==0 && ImGui::GetPlatformIO().Viewports.Size<=1, &frameBegin);
            if (frameBegin.sleepCalled) { profiler_.addIdleSample("Reflex Frame Pacing", frameBegin.sleepMs); }
            { auto scope = profiler_.scope("Poll Events"); pollEvents(); }
            checkRaster(running_ && !SDL_GetKeyboardState(nullptr)[SDL_SCANCODE_ESCAPE],"Raster comparison cancelled");
            auto scope = profiler_.scope("Render Frame");
            checkRaster(renderFrame() && !viewportCompileFailed_,"Raster comparison rendering failed");
        };
        const auto start = Clock::now();
        auto readyAt = Clock::time_point{};
        const double warmup = config.value("warmupSeconds",10.0);
        checkRaster(std::isfinite(warmup) && warmup >= 0 && warmup <= 120,"Invalid warmup duration");
        while (true) {
            draw();
            const auto now = Clock::now();
            const auto readiness = subsystemHost_.get<StreamerSubsystem>()->sceneReadiness();
            if (readiness.ready && readiness.requiredPages && viewportPreviewValid_) {
                if (readyAt == Clock::time_point{}) { readyAt=now; }
                if (std::chrono::duration<double>(now-readyAt).count() >= warmup) { break; }
            }
            checkRaster(std::chrono::duration<double>(now-start).count()<600,"Full raster warmup timed out");
        }
        drain();
        graphExecutor_->setDebugObserver(nullptr);
        set("VBuffer","benchmarkFreezeStreaming",true);
        set("Deferred","benchmarkFreezeStreaming",true);
        const auto generation = graphExecutor_->executionStats().graphGeneration;
        report["config"] = config;
        // Version the expected behavior so older causal captures remain readable.
        report["historyInvalidationPolicy"] = "reprojection-v1";
        report["validationRequested"] = bool(shaderTrace_) || (debugRuntime_ && std::getenv("METALLIC_DEBUG_VALIDATION"));
        report["hidden"] = std::getenv("METALLIC_FULL_ROAM_HIDDEN") != nullptr;
        report["graphicsCaptureInjected"] = profiling::NsightGraphicsCapture::vulkanInjectionActive();
#ifdef _WIN32
        report["gpuTraceInjected"] = GetModuleHandleW(L"WarpVizTarget.dll") != nullptr;
        report["renderDocInjected"] = GetModuleHandleW(L"renderdoc.dll") != nullptr;
#endif
        const char* pipelineStatistics = std::getenv("METALLIC_VK_PIPELINE_STATISTICS");
        const bool pipelineStatisticsRequested = pipelineStatistics && std::strcmp(pipelineStatistics, "1") == 0;
        report["pipelineStatisticsRequested"] = pipelineStatisticsRequested;
        report["nvPerfRequested"] = nvPerfRequested;
        report["workControlReplayRequested"] = profiling::workControlReplayRequested();
        report["shaderTraceRequested"] = bool(shaderTrace_);
        if (shaderTrace_) {
            checkRaster(workloadCase && selectedMode==15 && rounds==1 && primeHistory && !traceFrames &&
                profileHoldSeconds==0 && !pipelineStatisticsRequested && !nvPerfRequested &&
                !report["graphicsCaptureInjected"].get<bool>() && !report.value("gpuTraceInjected",false) &&
                !report.value("renderDocInjected",false),"Shader trace requires one primed WorkControl case without other profilers");
        }
        if (nvPerfRequested) {
            checkRaster(workloadCase && selectedMode == 15 && rounds == 1 && primeHistory && !traceFrames &&
                profileHoldSeconds == 0 && !pipelineStatisticsRequested && !report["validationRequested"].get<bool>() &&
                !report["graphicsCaptureInjected"].get<bool>() && !report.value("gpuTraceInjected", false) &&
                !report.value("renderDocInjected", false), "NvPerf needs one primed WorkControl round without other instrumentation");
        }
        report["measurementKind"] = shaderTrace_ || nvPerfRequested || profileHoldSeconds > 0 || traceFrames || pipelineStatisticsRequested
            ? "diagnostic" : "normal-timing";
        report["camera"] = viewportCameraProperties();
        auto targetCamera = viewportCameraProperties();
        const auto routeCamera = targetCamera;
        auto primeCamera = targetCamera;
        if (primeHistory) {
            for (const auto* key : {"eye", "center"}) {
                for (size_t axis = 0; axis < 3; ++axis) {
                    primeCamera["camera"][key][axis] = targetCamera["camera"][key][axis].get<double>() +
                        config.at("primeCameraOffset")[axis].get<double>();
                }
            }
            report["historyPriming"] = {{"camera", primeCamera}, {"frames", 4},
                {"policy", "four unmeasured prime frames before every target frame; target GPU drained before next prime"}};
        }
        const auto primeTarget = [&]() {
            if (!primeHistory) { return; }
            applyViewportCameraProperties(primeCamera, nullptr);
            for (uint32_t i = 0; i < 4; ++i) { draw(); }
            drain();
            applyViewportCameraProperties(targetCamera, nullptr);
        };
        report["outputExtent"] = {width,height};
        const auto* depth = graphExecutor_->outputResource("VBuffer.depth");
        checkRaster(depth != nullptr,"Missing raster extent");
        report["renderExtent"] = {depth->desc.width,depth->desc.height};
        report["graphGeneration"] = generation;
        report["loadingAndWarmupSeconds"] = std::chrono::duration<double>(Clock::now()-start).count();
        report["graph"] = Json::array();
        for (const auto& node : renderGraph_.nodes()) {
            auto properties=node.properties; properties.merge_patch(node.runtimeProperties);
            report["graph"].push_back({{"name",node.name},{"type",node.type},{"properties",properties}});
        }
        // Reverse and rotate order to expose warm-cache/clock/order effects.
        const bool historyComparison = config.value("historyComparison", false);
        checkRaster(!groupComparison || (!workloadCase && !historyComparison &&
            device_->capabilities().subgroupSize == 32 && device_->capabilities().minSubgroupSize == 32 &&
            device_->capabilities().maxSubgroupSize == 32), "Group comparison requires fixed wave32 and its own suite");
        constexpr uint32_t groupOrders[3][4] = {{15,17,18,19},{19,18,17,15},{18,15,19,17}};
        const bool swWorkComparison = groupComparison || workloadCase || historyComparison || config.value("swWorkComparison", false);
        const bool swLoadComparison = config.value("swLoadComparison", false);
        const bool swComparison = swWorkComparison || swLoadComparison || config.value("swComparison", false);
        constexpr uint32_t historyOrders[3][3] = {{0,15,16},{16,15,0},{15,0,16}};
        constexpr uint32_t workOrders[3][5] = {{0,10,13,15,14},{14,15,13,10,0},{13,0,15,14,10}};
        constexpr uint32_t loadOrders[3][3] = {{0,10,13},{13,10,0},{10,0,13}};
        constexpr uint32_t swOrders[3][4] = {{0,10,11,12},{12,11,10,0},{11,0,12,10}};
        const bool metadataComparison = config.value("metadataComparison", false);
        constexpr uint32_t metadataOrders[3][3] = {{0,8,9},{9,8,0},{8,0,9}};
        constexpr uint32_t orders[3][5] = {{0,1,2,4,8},{8,4,2,1,0},{2,0,8,1,4}};
        std::string cutHash, mappingHash;
        profiling::WorkControlReplay* replayToArm = nullptr;
        const auto checkpoint = [&](const std::string& name) {
            drain();
            primeTarget();
            if (replayToArm) { replayToArm->arm(); replayToArm = nullptr; }
            graphExecutor_->setDebugObserver(&observer); observer.capture=true;
            set("VBuffer", "softwareRasterWorkload", config.value("workloadCounters", false));
            observer.workload.editorFrame = profiler_.nextFrameIndex();
            observer.workload.camera = viewportCameraProperties();
            draw(); drain();
            set("VBuffer", "softwareRasterWorkload", false);
            graphExecutor_->setDebugObserver(nullptr); observer.capture=false;
            auto snap = observer.snapshot(output,name);
            if (cutHash.empty()) { cutHash=snap["cutHash"]; mappingHash=snap["pageMappingsHash"]; }
            checkRaster(snap["cutHash"]==cutHash && snap["pageMappingsHash"]==mappingHash,
                "Frozen cut or resident page mappings changed between cases");
            return snap;
        };
        report["liveRoam"] = Json::array();
        for (uint32_t round=0; round<rounds; ++round) {
            if (roamSeconds > 0) {
                report["measurementKind"] = "correctness-diagnostic";
                drain();
                set("VBuffer", "benchmarkFreezeStreaming", false);
                set("Deferred", "benchmarkFreezeStreaming", false);
                set("VBuffer", "benchmarkSoftwareGroupSize", config.value("swGroupUseDefault32", false) ? 0u : 32u);
                set("VBuffer", "benchmarkSoftwareReference128", false);
                const auto segmentStart = Clock::now();
                uint64_t liveFrames = 0;
                Json live = Json::array();
                profiler_.beginCapture();
                while (true) {
                    const double elapsed = std::chrono::duration<double>(Clock::now()-segmentStart).count();
                    const double t = std::min(elapsed / roamSeconds, 1.0);
                    const double progress = (round + t) / rounds;
                    const double angle = std::sin(progress * 6.28318530718) * 0.7;
                    auto camera = routeCamera;
                    const auto& base = routeCamera.at("camera");
                    const double dx = base["center"][0].get<double>() - base["eye"][0].get<double>();
                    const double dz = base["center"][2].get<double>() - base["eye"][2].get<double>();
                    const double distance = std::hypot(dx, dz);
                    checkRaster(distance > 1e-6, "Roam needs horizontal camera direction");
                    const double travel = 6.0 * std::sin(progress * 3.14159265359);
                    camera["camera"]["eye"][0] = base["eye"][0].get<double>() + dx / distance * travel;
                    camera["camera"]["eye"][2] = base["eye"][2].get<double>() + dz / distance * travel;
                    camera["camera"]["center"][0] = camera["camera"]["eye"][0].get<double>() + std::cos(angle)*dx + std::sin(angle)*dz;
                    camera["camera"]["center"][2] = camera["camera"]["eye"][2].get<double>() - std::sin(angle)*dx + std::cos(angle)*dz;
                    applyViewportCameraProperties(camera, nullptr);
                    draw(); ++liveFrames;
                    if (liveFrames % 30 == 0 || t == 1.0) {
                        const auto profiles = subsystemHost_.get<StreamerSubsystem>()->sceneReadiness();
                        live.push_back({{"frame", profiler_.nextFrameIndex()-1}, {"seconds",elapsed},
                            {"camera",viewportCameraProperties()}, {"ready",profiles.ready}});
                    }
                    if (t == 1.0) { break; }
                }
                profiler_.endCapture(); drain();
                checkRaster(!profiler_.captureOverflow(), "Live roam capture overflow");
                Json telemetry = Json::array();
                for (const auto& frame : profiler_.capturedFrames()) {
                    for (const auto& stream : frame.streaming) {
                        checkRaster(!stream.softwareRasterIdentity.empty(), "Missing live raster identity");
                        const auto identity = Json::parse(stream.softwareRasterIdentity);
                        checkRaster(identity.value("groupSize", 0u) == 32, "Live roam lost 32-thread pipeline");
                        telemetry.push_back({{"frame",frame.index}, {"shader",identity},
                            {"residentPages",stream.residentPages}, {"uploads",stream.uploads}, {"evictions",stream.evictions},
                            {"loadFailures",stream.loadFailures}, {"requestOverflows",stream.requestOverflows},
                            {"blasOverflowCount",stream.blasOverflowCount}, {"geometryBytes",stream.geometryUsedBytes},
                            {"clasBytes",stream.clasUsedBytes}, {"textureBytes",stream.textureResidentBytes}});
                    }
                }
                checkRaster(telemetry.size() > 1, "Missing live streaming telemetry");
                const auto telemetryFile = "round"+std::to_string(round+1)+"-live.json";
                std::ofstream(output/telemetryFile) << telemetry.dump() << '\n';
                targetCamera = viewportCameraProperties();
                primeCamera = targetCamera;
                if (primeHistory) {
                    for (const auto* key : {"eye", "center"}) {
                        for (size_t axis=0; axis<3; ++axis) {
                            primeCamera["camera"][key][axis] = targetCamera["camera"][key][axis].get<double>() +
                                config.at("primeCameraOffset")[axis].get<double>();
                        }
                    }
                }
                set("VBuffer", "benchmarkFreezeStreaming", true);
                set("Deferred", "benchmarkFreezeStreaming", true);
                report["camera"] = targetCamera;
                report["liveRoam"].push_back({{"round",round+1}, {"frames",liveFrames},
                    {"seconds",std::chrono::duration<double>(Clock::now()-segmentStart).count()}, {"samples",live}, {"telemetryFile",telemetryFile},
                    {"targetCamera",targetCamera}, {"primeCamera",primeCamera}});
                cutHash.clear(); mappingHash.clear();
                spdlog::info("[Raster Comparison] Live 32-thread segment {} complete: {} frames", round+1, liveFrames);
            }
            const auto sequence = groupComparison ? std::span<const uint32_t>(groupOrders[round]) : workloadCase ? std::span<const uint32_t>(&selectedMode, 1) : historyComparison ? std::span<const uint32_t>(historyOrders[round]) : swWorkComparison ? std::span<const uint32_t>(workOrders[round]) : swLoadComparison ? std::span<const uint32_t>(loadOrders[round]) : swComparison ? std::span<const uint32_t>(swOrders[round]) : metadataComparison ? std::span<const uint32_t>(metadataOrders[round]) : std::span<const uint32_t>(orders[round]);
            for (uint32_t mode : sequence) {
                const std::string variant = !mode ? "0" : mode == 17 ? "swGroup32" : mode == 18 ? "swGroup64" : mode == 19 ? "swGroup128" : swComparison ? (mode == 16 ? "swCameraReapply" : mode == 10 ? "swLegacy" : mode == 15 ? "swWorkControl" : mode == 14 ? "swWorkBins" : mode == 13 ? "swCooperative" : mode == 11 ? "swPrepared" : "swPlane") : metadataComparison ? (mode == 8 ? "exact8" : "fast8") : std::to_string(mode);
                set("VBuffer", "benchmarkSoftwareGroupSize", mode == 17 && !config.value("swGroupUseDefault32", false) ? 32u : mode == 18 ? 64u : mode == 19 ? 128u : 0u);
                set("VBuffer", "benchmarkSoftwareReference128", mode == 15 || mode == 16);
                reapplyCamera = historyComparison && mode == 16;
                const std::string name = "round"+std::to_string(round+1)+"-"+variant;
                set("VBuffer","softwareRasterPreparedVertices", swComparison && (mode == 11 || mode == 12));
                set("VBuffer","softwareRasterCooperativeLoad", !swComparison || ((swLoadComparison || swWorkComparison) && mode >= 13));
                set("VBuffer","softwareRasterWorkBins", swWorkComparison && mode == 14);
                set("VBuffer","softwareRasterSharedScreenVertices", !swComparison || (swWorkComparison && mode >= 15));
                set("VBuffer","softwareRasterIncrementalDepth", swComparison && mode == 12);
                set("VBuffer","metadataFastClassification", !metadataComparison || mode != 8);
                set("VBuffer","benchmarkForceHardwareRaster",mode==0);
                set("VBuffer","softwareRasterMaxPixels",(metadataComparison || swComparison) ? 8u : mode ? mode : 8u);
                for (uint32_t i=0; i<settle; ++i) { draw(); }
                const auto before = checkpoint(name+"-before");
                if (profiling::workControlReplayRequested()) {
                    checkRaster(workloadCase && primeHistory && rounds == 1 && mode == 15 &&
                        !shaderTrace_ && !traceFrames && !pipelineStatisticsRequested,
                        "Isolated correctness replay requires one frozen primed WorkControl round without another profiler");
                    report["measurementKind"] = "diagnostic";
                    report["normalTimingEligible"] = false;
                    const char* phase = std::getenv("METALLIC_WORK_CONTROL_REPLAY_PHASE");
                    profiling::WorkControlReplay replay(*device_, phase ? phase : "early", output / "replay");
                    replayToArm = &replay;
                    const auto control = checkpoint(name+"-control");
                    // checkpoint drained the exact captured frame. No draw, frame
                    // slot rotation or streamer publication occurs inside run().
                    report["replay"] = replay.run(*graphicsQueue_, {{"workloadCase", report.at("workloadCase")},
                        {"snapshot", control}, {"camera", targetCamera}, {"renderExtent", report.at("renderExtent")}});
                    const auto restored = checkpoint(name+"-restored");
                    report["cases"].push_back({{"name", name}, {"variant", variant},
                        {"before", before}, {"control", control}, {"after", restored}});
                    continue;
                }
                // Readback/copy cache effects are outside measurement, followed
                // by the same unmeasured recovery frames in every variant.
                for (uint32_t i=0; i<settle; ++i) { draw(); }
                if (shaderTrace_) {
                    const auto& selection=config.at("shaderTrace");
                    checkRaster(selection.is_object(),"Shader trace selection must be an object");
                    for (auto field=selection.begin();field!=selection.end();++field) {
                        checkRaster(field.key()=="site" || field.key()=="phase" || field.key()=="group" || field.key()=="localIndex" || field.key()=="predicate", "Unknown shader trace selection field");
                    }
                    const std::string phase=selection.at("phase");
                    checkRaster(phase=="early" || phase=="late","Shader trace phase must be early or late");
                    shaderTrace_->configure(observer.graph);
                    const auto& site=shaderTrace_->site(selection.value("site",std::string("stream.after-triangle-prepare")),phase);
                    Json watch{{"version",1},{"generation",observer.graph.at("generation")},
                        {"target",{{"site",site.at("name")},{"phase",phase},{"expectedSiteSchemaHash",site.at("schemaHash")}}},
                        {"invocation",{{"group",selection.value("group",Json::array({0,0,0}))},{"localIndex",selection.value("localIndex",0u)}}},
                        {"limits",{{"targetFrames",1},{"maxRecords",16},{"timeoutMs",30000}}}};
                    if (selection.contains("predicate")) { watch["predicate"]=selection.at("predicate"); }
                    shaderTrace_->prepare(*device_,watch,{{"workloadCase",report.at("workloadCase")},{"baseline",before},
                        {"camera",targetCamera},{"renderExtent",report.at("renderExtent")}});
                    observer.trace=shaderTrace_.get();
                    const auto diagnostic=checkpoint(name+"-diagnostic");
                    shaderTrace_->targetDrained(*graphicsQueue_);
                    const auto restored=checkpoint(name+"-restored");
                    const auto equalReadback = [&](const Json& snapshot) {
                        for (const char* key : {"cutHash","pageMappingsHash","AfterStreamEarlyBins","AfterStreamLateBins",
                            "AfterStreamEarlyClusterCull","AfterStreamLateClusterCull"}) {
                            if (snapshot.at(key)!=before.at(key)) { return false; }
                        }
                        for (const char* key : {"VBuffer.visibility","VBuffer.depth"}) {
                            if (snapshot.at(key).at("hash")!=before.at(key).at("hash") || snapshot.at(key).at("pixels")!=before.at(key).at("pixels")) { return false; }
                        }
                        return true;
                    };
                    const bool same=equalReadback(diagnostic) && equalReadback(restored) && restored.at("productionDispatches")==before.at("productionDispatches");
                    shaderTrace_->restoration({{"before",before},{"diagnostic",diagnostic},{"after",restored},
                        {"productionBindingRestored",restored.at("productionDispatches")==before.at("productionDispatches")},
                        {"readbacksIdentical",same}},same);
                    observer.trace=nullptr;
                    checkRaster(same,"Shader observation perturbed input/output or production restoration failed");
                    report["cases"].push_back({{"name",name},{"variant",variant},{"before",before},{"diagnostic",diagnostic},{"after",restored}});
                    report["shaderTrace"]={{"job",shaderTrace_->job()},{"phase",phase},{"collectionBoundary","case-process-instance-destroyed"}};
                    continue; // No timing samples from an instrumented process.
                }
                if (nvPerfRequested) {
                    primeTarget(); drain();
                    std::string error;
                    const bool started = nvPerf.begin(*device_, *graphicsQueue_, output / "nvperf", error);
                    checkRaster(started, error.c_str());
                    report["nvPerf"] = {{"startFrame", profiler_.nextFrameIndex()}, {"frames", 1},
                        {"workloadCase", report["workloadCase"]}, {"snapshot", before}, {"complete", false}};
                    draw(); drain();
                    const bool stopped = nvPerf.finish(error);
                    checkRaster(stopped, error.c_str());
                    report["nvPerf"]["complete"] = true;
                }
                if (traceFrames) {
                    primeTarget();
                    drain();
                    std::string error;
                    const bool started = profiling::beginExternalNsightGpuTrace(error);
                    checkRaster(started, error.c_str());
                    traceActive = true;
                    report["sdkTrace"] = {{"startFrame",profiler_.nextFrameIndex()}, {"frames",traceFrames},
                        {"workloadCase",report["workloadCase"]}, {"snapshot",before}, {"complete",false}};
                    for (uint32_t i=0; i<traceFrames; ++i) { draw(); }
                    drain();
                    const bool stopped = profiling::endExternalNsightGpuTrace(error);
                    checkRaster(stopped, error.c_str());
                    traceActive = false;
                    report["sdkTrace"]["complete"] = true;
                    // SDK boundary success does not prove artifact/export success.
                    report["sdkTrace"]["artifactVerified"] = false;
                }
                if (((workloadCase && mode == selectedMode) || (!workloadCase && swComparison && mode == 10)) && round == 0 && profileHoldSeconds > 0) {
                    std::ofstream(output / "ProfileReady.json") << Json({{"case", name}, {"snapshot", before},
                        {"workloadCase", report.value("workloadCase", Json(nullptr))},
                        {"holdSeconds", profileHoldSeconds}, {"camera", report["camera"]}}).dump(2) << '\n';
                    spdlog::info("[Raster Comparison] SW profiler hold begin: {} seconds", profileHoldSeconds);
                    const auto holdStart = Clock::now();
                    while (std::chrono::duration<double>(Clock::now() - holdStart).count() < profileHoldSeconds) {
                        draw();
                        checkRaster(viewportCameraProperties() == report["camera"], "Profiler camera changed");
                    }
                    spdlog::info("[Raster Comparison] SW profiler hold end");
                    drain();
                }
                if (!primeHistory) { profiler_.beginCapture(); }
                const auto unixMs = []() {
                    return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();
                };
                const double measurementBeginUnixMs = unixMs();
                std::vector<double> wall;
                std::vector<EditorProfiler::Frame> primedFrames;
                for (uint32_t i=0; i<samples; ++i) {
                    if (primeHistory) { primeTarget(); profiler_.beginCapture(); }
                    const auto begin=Clock::now(); draw();
                    wall.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
                    if (primeHistory) {
                        profiler_.endCapture(); drain();
                        checkRaster(!profiler_.captureOverflow() && profiler_.capturedFrames().size() == 1,
                            "History target frame capture failed");
                        primedFrames.push_back(profiler_.capturedFrames().front());
                    }
                    checkRaster(graphExecutor_->executionStats().graphGeneration==generation &&
                        viewportTextureWidth_==width && viewportTextureHeight_==height,"Raster graph or viewport changed");
                    checkRaster(viewportCameraProperties()==report["camera"],"Raster camera changed");
                }
                profiler_.endCapture(); drain();
                const double measurementEndUnixMs = unixMs();
                checkRaster(!profiler_.captureOverflow(),"Raster profiler overflow");
                const auto& frames = primeHistory ? primedFrames : profiler_.capturedFrames();
                checkRaster(frames.size()==samples,"Raster frame count mismatch");
                Json rows=Json::array();
                for (size_t i=0; i<frames.size(); ++i) {
                    const auto& f=frames[i];
                    Json row{{"frame",f.index},{"editorLoopMs",wall[i]},{"scopes",Json::array()},{"streaming",Json::array()}};
                    std::vector<std::string> paths;
                    bool graphGpu=false, classified=false, software=false;
                    for (size_t j=0; j<f.nodes.size(); ++j) {
                        const auto& n=f.nodes[j];
                        const auto path=(j && n.parent<j ? paths[n.parent]+"/" : "")+n.name;
                        paths.push_back(path);
                        if (n.name=="RenderGraph GPU envelope") { graphGpu=n.gpuTimingAvailable; }
                        if (path.find("/Stream early/")!=std::string::npos || path.find("/Stream late/")!=std::string::npos) {
                            classified |= n.name=="Soft/hard classification";
                            software |= n.name=="Software raster";
                        }
                        row["scopes"].push_back({{"path",path},{"cpuMs",n.cpuMilliseconds},
                            {"gpuMs",n.gpuTimingAvailable ? Json(n.gpuMilliseconds) : Json(nullptr)}});
                    }
                    checkRaster(graphGpu && !f.profilingOverflow,"Missing raster GPU timestamps");
                    if (!mode) { checkRaster(!classified && !software,"Full HW still executed geometry classification or SW raster"); }
                    for (const auto& s : f.streaming) {
                        row["streaming"].push_back({{"softwareRaster",s.softwareRasterIdentity.empty() ? Json(nullptr) : Json::parse(s.softwareRasterIdentity)},{"geometryBytes",s.geometryUsedBytes},{"residentPages",s.residentPages},
                            {"clasBytes",s.clasUsedBytes},{"textureBytes",s.textureResidentBytes},
                            {"textureUpgrades",s.textureUpgrades},{"textureDowngrades",s.textureDowngrades}});
                    }
                    rows.push_back(std::move(row));
                }
                const auto after=checkpoint(name+"-after");
                if (config.value("requireNonzeroLate", false)) {
                    checkRaster(before["AfterStreamLateBins"]["softwareClusters"].get<uint32_t>() > 0 &&
                        after["AfterStreamLateBins"]["softwareClusters"].get<uint32_t>() > 0,
                        "Correctness case did not exercise nonzero late software raster");
                }
                if (!mode) { checkRaster(after["AfterStreamEarlyBins"]["softwareClusters"]==0 &&
                    after["AfterStreamLateBins"]["softwareClusters"]==0,"Full HW emitted software clusters"); }
                const auto file=name+"-frames.json";
                std::ofstream framesFile(output/file); framesFile.exceptions(std::ios::badbit|std::ios::failbit);
                framesFile<<rows.dump()<<'\n'; framesFile.close();
                report["cases"].push_back({{"name",name},{"round",round+1},{"maxPixels",(metadataComparison || swComparison) ? 8u : mode},{"variant",variant},
                    {"fullHardware",mode==0},{"framesFile",file},{"frames",samples},{"before",before},{"after",after},
                    {"measurementBeginUnixMs",measurementBeginUnixMs},{"measurementEndUnixMs",measurementEndUnixMs}});
                std::ofstream(output/"Progress.json")<<report.dump(2)<<'\n';
                spdlog::info("[Raster Comparison] {} complete, {} measured frames, cut={} pages={}",name,samples,cutHash,mappingHash);
            }
        }
        reapplyCamera = false;
        set("VBuffer","benchmarkFreezeStreaming",false);
        set("Deferred","benchmarkFreezeStreaming",false);
        set("VBuffer","benchmarkForceHardwareRaster",false);
        set("VBuffer","softwareRasterMaxPixels",8.0f);
        set("VBuffer","metadataFastClassification",true);
        set("VBuffer","softwareRasterPreparedVertices",false);
        set("VBuffer","softwareRasterCooperativeLoad",true);
        set("VBuffer","softwareRasterWorkBins",false);
        set("VBuffer","softwareRasterSharedScreenVertices",true);
        set("VBuffer","softwareRasterIncrementalDepth",false);
        set("VBuffer","benchmarkSoftwareGroupSize",0u);
        set("VBuffer","benchmarkSoftwareReference128",false);
        report["status"]="capture_complete"; passed=true;
    } catch (const std::exception& e) {
        if (shaderTrace_) { shaderTrace_->abort(e.what()); }
        report["error"]=e.what();
        spdlog::error("[Raster Comparison] {}",e.what());
    }
    profiler_.endCapture();
    // Lifetime of observer/readbacks extends past every submitted reference,
    // including an exception in a diagnostic draw.
    const auto frameDrain = frameSubmissions_.wait();
    const auto graphDrain = graphExecutor_->waitForSubmittedWork();
    if (traceActive) {
        std::string error;
        if (!profiling::endExternalNsightGpuTrace(error)) { report["traceCleanupError"] = error; }
    }
    if (!frameDrain || !graphDrain) { report["status"]="failed"; report["error"]="Final GPU drain failed"; passed=false; }
    graphExecutor_->setDebugObserver(debugRuntime_.get());
    profiler_.beginCapture(); profiler_.endCapture();
    fullRoamActive_=false; fullRoamWidth_=fullRoamHeight_=0;
    std::ofstream(output/"Capture.json")<<report.dump(2)<<'\n';
    return passed;
}
} // namespace metallic
