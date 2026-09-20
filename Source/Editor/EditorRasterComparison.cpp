#include "Editor/EditorApplication.h"
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
    Device* device = nullptr;
    std::map<std::string, std::unique_ptr<Buffer>> copies;
    void compiled(Json) override {}
    void beginExecution(Device& d, debug::DebugEvidenceStamp, RenderSubsystemHost*) override { device = &d; }
    void endExecution(bool) override {}
    void boundary(CommandBuffer& commands, std::string_view checkpoint, uint32_t,
        std::string_view pass, std::span<const DebugResourceBinding> resources, const Json&) override
    {
        if (!capture || pass != "VBuffer") { return; }
        for (const auto& resource : resources) {
            std::string key;
            uint64_t bytes = 0;
            if (checkpoint == "AfterTraversal" && resource.id == "streaming.VBuffer.activeHeader") { key = "header"; }
            if (checkpoint == "AfterTraversal" && resource.id == "streaming.VBuffer.activeGroups") { key = "groups"; }
            if (checkpoint == "AfterTraversal" && resource.id == "streaming.VBuffer.pageTable") { key = "pages"; }
            if ((checkpoint == "AfterStreamEarlyBins" || checkpoint == "AfterStreamLateBins" || checkpoint == "AfterStreamEarlyClusterCull" || checkpoint == "AfterStreamLateClusterCull") && resource.id == "hybrid.VBuffer.clusters") {
                key = checkpoint; bytes = 64;
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
                    .memoryLocation = MemoryLocation::HostReadback}, copy)), "Cannot allocate raster diagnostic readback");
            }
            if (resource.texture) {
                TextureBarrierDesc barrier{.texture = resource.texture, .before = resource.state, .after = ResourceState::TransferSource};
                commands.barrier({.textures = &barrier, .textureCount = 1});
                commands.copyTextureToBuffer({.texture = resource.texture, .buffer = copy.get(),
                    .width = resource.texture->desc().width, .height = resource.texture->desc().height});
                std::swap(barrier.before, barrier.after);
                commands.barrier({.textures = &barrier, .textureCount = 1});
            } else {
                BufferBarrierDesc barrier{.buffer = resource.buffer, .before = resource.state, .after = ResourceState::TransferSource,
                    .offset = resource.offset, .size = bytes};
                commands.barrier({.buffers = &barrier, .bufferCount = 1});
                commands.copyBuffer({.source = resource.buffer, .destination = copy.get(), .sourceOffset = resource.offset, .size = bytes});
                std::swap(barrier.before, barrier.after);
                commands.barrier({.buffers = &barrier, .bufferCount = 1});
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
        const auto header = read<MeshletStreamGpuActiveHeader>("header").front();
        auto groups = read<MeshletStreamGpuActiveGroup>("groups");
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
            result[phase] = {{"hardwareClusters",bins[0]+bins[1]+bins[2]+bins[3]},
                {"softwareClusters",bins[4]},{"candidates",bins[12]},{"capacity",bins[5]}};
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
    const auto drain = [&]() {
        checkRaster(bool(frameSubmissions_.wait()) && bool(graphExecutor_->waitForSubmittedWork()), "Raster benchmark GPU drain failed");
        std::vector<RenderGraphExecutionStats> completed;
        checkRaster(bool(graphExecutor_->collectCompletedGpuExecutionStats(completed)), "Raster benchmark query resolve failed");
        for (const auto& stats : completed) { profiler_.updateRenderGraphGpuStats(stats); }
    };
    try {
        const uint32_t samples = config.value("sampleFrames",64u);
        const uint32_t settle = config.value("settleFrames",8u);
        const uint32_t rounds = config.value("rounds",3u);
        const uint32_t width = config.value("width",1797u), height = config.value("height",660u);
        checkRaster(samples >= 8 && samples <= 512 && settle >= 2 && settle <= 120 && rounds >= 1 && rounds <= 3,
            "Invalid raster comparison sample count");
        checkRaster(width >= 64 && height >= 64 && width <= 8192 && height <= 8192,"Invalid raster comparison extent");
        fullRoamWidth_ = width; fullRoamHeight_ = height; fullRoamActive_ = true;
        drain();
        graphExecutor_->setDebugObserver(&observer);
        loadBuiltInSample(kGPUDrivenZorahFullSampleId);
        auto view = viewportCameraProperties();
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
        const auto draw = [&]() {
            checkRaster(!viewportView_.temporalJitter(), "Raster comparison jitter was re-enabled");
            auto frame = profiler_.beginFrame();
            checkRaster(waitForFrameSlotBeforeInput(),"Frame slot wait failed");
            const vulkan::StreamlineFrameScope streamlineFrame(
                (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED)==0 && ImGui::GetPlatformIO().Viewports.Size<=1);
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
        report["validationRequested"] = debugRuntime_ && std::getenv("METALLIC_DEBUG_VALIDATION");
        report["hidden"] = std::getenv("METALLIC_FULL_ROAM_HIDDEN") != nullptr;
        report["camera"] = viewportCameraProperties();
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
        const bool swComparison = config.value("swComparison", false);
        constexpr uint32_t swOrders[3][4] = {{0,10,11,12},{12,11,10,0},{11,0,12,10}};
        const bool metadataComparison = config.value("metadataComparison", false);
        constexpr uint32_t metadataOrders[3][3] = {{0,8,9},{9,8,0},{8,0,9}};
        constexpr uint32_t orders[3][5] = {{0,1,2,4,8},{8,4,2,1,0},{2,0,8,1,4}};
        std::string cutHash, mappingHash;
        const auto checkpoint = [&](const std::string& name) {
            drain();
            graphExecutor_->setDebugObserver(&observer); observer.capture=true;
            draw(); drain();
            graphExecutor_->setDebugObserver(nullptr); observer.capture=false;
            auto snap = observer.snapshot(output,name);
            if (cutHash.empty()) { cutHash=snap["cutHash"]; mappingHash=snap["pageMappingsHash"]; }
            checkRaster(snap["cutHash"]==cutHash && snap["pageMappingsHash"]==mappingHash,
                "Frozen cut or resident page mappings changed between cases");
            return snap;
        };
        for (uint32_t round=0; round<rounds; ++round) {
            const auto sequence = swComparison ? std::span<const uint32_t>(swOrders[round]) : metadataComparison ? std::span<const uint32_t>(metadataOrders[round]) : std::span<const uint32_t>(orders[round]);
            for (uint32_t mode : sequence) {
                const std::string variant = !mode ? "0" : swComparison ? (mode == 10 ? "swLegacy" : mode == 11 ? "swPrepared" : "swPlane") : metadataComparison ? (mode == 8 ? "exact8" : "fast8") : std::to_string(mode);
                const std::string name = "round"+std::to_string(round+1)+"-"+variant;
                set("VBuffer","softwareRasterPreparedVertices", swComparison && mode != 10);
                set("VBuffer","softwareRasterIncrementalDepth", swComparison && mode == 12);
                set("VBuffer","metadataFastClassification", !metadataComparison || mode != 8);
                set("VBuffer","benchmarkForceHardwareRaster",mode==0);
                set("VBuffer","softwareRasterMaxPixels",(metadataComparison || swComparison) ? 8u : mode ? mode : 8u);
                for (uint32_t i=0; i<settle; ++i) { draw(); }
                const auto before = checkpoint(name+"-before");
                // Readback/copy cache effects are outside measurement, followed
                // by the same unmeasured recovery frames in every variant.
                for (uint32_t i=0; i<settle; ++i) { draw(); }
                profiler_.beginCapture();
                std::vector<double> wall;
                for (uint32_t i=0; i<samples; ++i) {
                    const auto begin=Clock::now(); draw();
                    wall.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
                    checkRaster(graphExecutor_->executionStats().graphGeneration==generation &&
                        viewportTextureWidth_==width && viewportTextureHeight_==height,"Raster graph or viewport changed");
                    checkRaster(viewportCameraProperties()==report["camera"],"Raster camera changed");
                }
                profiler_.endCapture(); drain();
                checkRaster(!profiler_.captureOverflow(),"Raster profiler overflow");
                const auto& frames = profiler_.capturedFrames();
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
                        row["streaming"].push_back({{"geometryBytes",s.geometryUsedBytes},{"residentPages",s.residentPages},
                            {"clasBytes",s.clasUsedBytes},{"textureBytes",s.textureResidentBytes},
                            {"textureUpgrades",s.textureUpgrades},{"textureDowngrades",s.textureDowngrades}});
                    }
                    rows.push_back(std::move(row));
                }
                const auto after=checkpoint(name+"-after");
                if (!mode) { checkRaster(after["AfterStreamEarlyBins"]["softwareClusters"]==0 &&
                    after["AfterStreamLateBins"]["softwareClusters"]==0,"Full HW emitted software clusters"); }
                const auto file=name+"-frames.json";
                std::ofstream framesFile(output/file); framesFile.exceptions(std::ios::badbit|std::ios::failbit);
                framesFile<<rows.dump()<<'\n'; framesFile.close();
                report["cases"].push_back({{"name",name},{"round",round+1},{"maxPixels",(metadataComparison || swComparison) ? 8u : mode},{"variant",variant},
                    {"fullHardware",mode==0},{"framesFile",file},{"frames",samples},{"before",before},{"after",after}});
                std::ofstream(output/"Progress.json")<<report.dump(2)<<'\n';
                spdlog::info("[Raster Comparison] {} complete, {} measured frames, cut={} pages={}",name,samples,cutHash,mappingHash);
            }
        }
        set("VBuffer","benchmarkFreezeStreaming",false);
        set("Deferred","benchmarkFreezeStreaming",false);
        set("VBuffer","benchmarkForceHardwareRaster",false);
        set("VBuffer","softwareRasterMaxPixels",8.0f);
        set("VBuffer","metadataFastClassification",true);
        set("VBuffer","softwareRasterPreparedVertices",false);
        set("VBuffer","softwareRasterIncrementalDepth",false);
        report["status"]="capture_complete"; passed=true;
    } catch (const std::exception& e) {
        report["error"]=e.what();
        spdlog::error("[Raster Comparison] {}",e.what());
    }
    profiler_.endCapture();
    // Lifetime of observer/readbacks extends past every submitted reference,
    // including an exception in a diagnostic draw.
    const auto frameDrain = frameSubmissions_.wait();
    const auto graphDrain = graphExecutor_->waitForSubmittedWork();
    if (!frameDrain || !graphDrain) { report["status"]="failed"; report["error"]="Final GPU drain failed"; passed=false; }
    graphExecutor_->setDebugObserver(debugRuntime_.get());
    profiler_.beginCapture(); profiler_.endCapture();
    fullRoamActive_=false; fullRoamWidth_=fullRoamHeight_=0;
    std::ofstream(output/"Capture.json")<<report.dump(2)<<'\n';
    return passed;
}
} // namespace metallic
