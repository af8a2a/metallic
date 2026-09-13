#include "RhiTest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#ifdef _WIN32
#include <Windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#endif

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void checkRoam(bool valid, const std::string& message)
{
    if (!valid) { throw std::runtime_error(message); }
}

// Raw, bounded snapshots only at checkpoints. Timed frames detach this
// observer and skip both output and debug readback.
class RoamingObserver final : public IRenderDebugObserver {
public:
    Json latest = Json::object();
    bool capture = false;
    std::map<std::string, std::unique_ptr<Buffer>> copies;
    Device* device = nullptr;
    std::unique_ptr<TimestampQueryPool> timestamps;
    static constexpr std::array<std::string_view, 19> stages = {"BeforeStreamUpdates", "AfterStreamUpdates",
        "AfterStreamFrontier", "AfterStreamPrefix", "AfterStreamEmit", "AfterTraversal", "AfterEarlyCull",
        "AfterStreamEarlyCandidates", "AfterStreamEarlyClassify", "AfterStreamEarlyBins", "AfterStreamEarlyRaster", "AfterStreamEarlyResolve",
        "AfterLateCull", "AfterStreamLateCandidates", "AfterStreamLateClassify", "AfterStreamLateBins", "AfterStreamLateRaster", "AfterStreamLateResolve", "AfterPass"};
    void compiled(Json) override {}
    void beginExecution(Device& d, debug::DebugEvidenceStamp, RenderSubsystemHost*) override { device = &d; }
    void endExecution(bool) override {}
    void boundary(CommandBuffer& commands, std::string_view checkpoint, uint32_t,
        std::string_view pass, std::span<const DebugResourceBinding> resources, const Json& values) override
    {
        if (pass != "GPUDriven") { return; }
        if (values.contains("streaming")) { latest = values.at("streaming").at("instances").at(0); }
        if (!capture) { return; }
        const auto stage = std::find(stages.begin(), stages.end(), checkpoint);
        if (stage != stages.end()) {
            if (!timestamps) {
                checkRoam(bool(device->createTimestampQueryPool(*device->getQueue(QueueType::Graphics),
                    {.queryCount = static_cast<uint32_t>(stages.size())}, timestamps)), "Cannot allocate checkpoint timestamps");
            }
            const uint32_t index = static_cast<uint32_t>(stage - stages.begin());
            if (index == 0) { checkRoam(bool(commands.resetTimestampQueries(*timestamps, 0, static_cast<uint32_t>(stages.size()))), "Cannot reset checkpoint timestamps"); }
            checkRoam(bool(commands.writeTimestamp(*timestamps, index, PipelineStageBits::BottomOfPipe)), "Cannot timestamp checkpoint");
        }
        for (const auto& resource : resources) {
            uint64_t size = 0;
            std::string key;
            if (checkpoint == "AfterTraversal" && resource.id == "streaming.GPUDriven.activeHeader") {
                size = sizeof(MeshletStreamGpuActiveHeader); key = "header";
            } else if (checkpoint == "AfterTraversal" && resource.id == "streaming.GPUDriven.activeGroups") {
                size = resource.size != 0 ? resource.size : resource.buffer->desc().size; key = "groups";
            } else if (checkpoint == "AfterEarlyCull" && resource.id.ends_with(".instanceVisibility")) {
                size = resource.size != 0 ? resource.size : resource.buffer->desc().size; key = "instances";
            } else if (checkpoint == "AfterLateCull" && resource.id.ends_with(".instanceVisibility")) {
                size = resource.size != 0 ? resource.size : resource.buffer->desc().size; key = "lateInstances";
            } else if ((checkpoint == "AfterStreamEarlyBins" || checkpoint == "AfterStreamLateBins") &&
                resource.id == "hybrid.GPUDriven.clusters") {
                size = 64; key = std::string(checkpoint);
            } else if (checkpoint == "AfterPass" && resource.id == "GPUDriven.rasterInfo") {
                size = sizeof(VisibilityBufferFrameInfo); key = "rasterInfo";
            } else if (checkpoint == "AfterPass" && (resource.id == "streaming.GPUDriven.requestHeader" ||
                resource.id == "streaming.GPUDriven.loadPriorities")) {
                size = resource.size; key = resource.id;
            }
            if (size == 0 || !resource.buffer) { continue; }
            auto& buffer = copies[key];
            if (!buffer || buffer->desc().size != size) {
                checkRoam(bool(device->createBuffer({.size = size, .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}, buffer)), "Cannot allocate roaming snapshot");
            }
            BufferBarrierDesc barrier{.buffer = resource.buffer, .before = resource.state,
                .after = ResourceState::TransferSource, .offset = resource.offset, .size = size};
            commands.barrier({.buffers = &barrier, .bufferCount = 1});
            commands.copyBuffer({.source = resource.buffer, .destination = buffer.get(),
                .sourceOffset = resource.offset, .size = size});
            std::swap(barrier.before, barrier.after);
            commands.barrier({.buffers = &barrier, .bufferCount = 1});
        }
    }
    template<typename T> std::vector<T> read(const std::string& name)
    {
        auto& buffer = copies.at(name);
        std::vector<T> result(buffer->desc().size / sizeof(T));
        buffer->invalidate();
        void* data = buffer->map();
        checkRoam(data != nullptr, "Cannot map roaming snapshot");
        std::memcpy(result.data(), data, result.size() * sizeof(T));
        buffer->unmap();
        return result;
    }
    Json readPagePriorities()
    {
        const auto header = read<StreamRequestBufferHeader>("streaming.GPUDriven.requestHeader").front();
        if (header.loadPriorityOffset == 0) { return {{"enabled", false}}; }
        const auto priorities = read<float>("streaming.GPUDriven.loadPriorities");
        const uint32_t count = std::min(header.loadCounter, header.maxLoadRequests);
        checkRoam(count <= priorities.size(), "Page benefit feedback exceeded its buffer");
        uint32_t positive = 0;
        float maximum = 0.f;
        for (uint32_t i = 0; i < count; ++i) {
            checkRoam(std::isfinite(priorities[i]) && priorities[i] >= 0.f && priorities[i] <= 1e9f,
                "Non-finite screen benefit reached the readback");
            positive += priorities[i] > 0.f;
            maximum = std::max(maximum, priorities[i]);
        }
        return {{"enabled", true}, {"requests", count}, {"positive", positive}, {"maximumBenefit", maximum}};
    }
    Json readTimings()
    {
        std::array<TimestampQueryResult, stages.size()> values{};
        checkRoam(timestamps && bool(timestamps->readResults(0, static_cast<uint32_t>(values.size()), values.data())), "Cannot read checkpoint timestamps");
        Json result;
        for (uint32_t i = 0; i < values.size(); ++i) {
            checkRoam(values[i].available, "Incomplete checkpoint timestamp");
            if (i != 0) { result[std::string(stages[i])] = timestamps->durationMilliseconds(values[i-1].value, values[i].value); }
        }
        result["afterTraversalToEnd"] = timestamps->durationMilliseconds(values[5].value, values.back().value);
        return result;
    }
};

Json roamingMemory()
{
    Json result;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX memory{};
    memory.cb = sizeof(memory);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory))) {
        result["processCommitBytes"] = memory.PrivateUsage;
        result["workingSetBytes"] = memory.WorkingSetSize;
    }
    IDXGIFactory1* factory = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        uint64_t local = 0, nonLocal = 0;
        bool available = false;
        for (UINT index = 0;; ++index) {
            IDXGIAdapter1* adapter = nullptr;
            if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) { break; }
            if (!adapter) { break; }
            IDXGIAdapter3* memoryAdapter = nullptr;
            if (SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&memoryAdapter)))) {
                DXGI_QUERY_VIDEO_MEMORY_INFO info{};
                if (SUCCEEDED(memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
                    local += info.CurrentUsage; available = true;
                }
                if (SUCCEEDED(memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) { nonLocal += info.CurrentUsage; }
                memoryAdapter->Release();
            }
            adapter->Release();
        }
        factory->Release();
        if (available) { result["processLocalGpuBytes"] = local; result["processNonLocalGpuBytes"] = nonLocal; }
    }
#endif
    return result;
}

// Reconstruct each emitted instance's DAG cut from terminal clusters. Every
// removed coarse cluster must have a complete replacement, including shared
// child groups. Check all reached parents agree on that replacement, and reject
// unreachable/duplicate selected groups. This detects holes and double cuts
// independently of how many non-background pixels happen to be on screen.
Json validateRoamingCut(RoamingObserver& observer, const scene::MeshletStreamAsset& asset, const Json& camera)
{
    const auto header = observer.read<MeshletStreamGpuActiveHeader>("header").at(0);
    const auto rows = observer.read<MeshletStreamGpuActiveGroup>("groups");
    checkRoam(header.activeGroupCount <= rows.size() && header.overflowCount < 2, "Invalid/empty capacity fallback");
    const auto instanceStates = observer.read<uint32_t>("instances");
    const auto lateInstanceStates = observer.read<uint32_t>("lateInstances");
    std::vector<std::map<uint32_t, uint32_t>> selected(asset.instanceCount());
    uint64_t clusters = 0, fineGroups = 0;
    uint64_t earlyCandidates = 0, lateCandidateLimit = 0, recoveredCandidates = 0;
    for (uint32_t i = 0; i < header.activeGroupCount; ++i) {
        const auto& row = rows[i];
        checkRoam(row.instanceIndex < selected.size() && row.pageIndex < asset.pageCount() &&
            row.pageDeviceOffsetBytes != kInvalidStreamDeviceOffsetBytes, "Invalid active record");
        const auto& page = asset.pages()[row.pageIndex];
        const uint32_t groupId = page.lodGroupIndex;
        checkRoam(groupId < asset.groups().size() && asset.groups()[groupId].pageIndex == row.pageIndex,
            "Page/group identity mismatch");
        checkRoam(selected[row.instanceIndex].emplace(groupId, row.clusterSelectionMask).second, "Duplicate selected group");
        checkRoam(asset.instances()[row.instanceIndex].primitiveIndex == row.primitiveIndex, "Instance identity mismatch");
        clusters += std::popcount(row.clusterSelectionMask);
        checkRoam(row.gpuSceneInstanceIndex < instanceStates.size() && row.gpuSceneInstanceIndex < lateInstanceStates.size(),
            "Missing candidate instance state");
        const auto clusterCount = std::popcount(row.clusterSelectionMask);
        if (instanceStates[row.gpuSceneInstanceIndex] == 1u) { earlyCandidates += clusterCount; }
        const auto lateState = lateInstanceStates[row.gpuSceneInstanceIndex];
        if (lateState == 1u || lateState == 3u) { lateCandidateLimit += clusterCount; }
        if (lateState == 3u) { recoveredCandidates += clusterCount; }
        fineGroups += (asset.groups()[groupId].flags & 1u) == 0u;
    }
    uint32_t verified = 0;
    uint64_t overTarget = 0, unbounded = 0;
    float maxFiniteError = 0;
    MeshletLodView view;
    for (uint32_t axis = 0; axis < 3; ++axis) { view.eye[axis] = camera.at("eye")[axis].get<float>(); }
    view.eye[3] = camera.at("znear").get<float>();
    float3 direction(camera.at("center")[0].get<float>() - view.eye[0], camera.at("center")[1].get<float>() - view.eye[1],
        camera.at("center")[2].get<float>() - view.eye[2]);
    direction = normalize(direction);
    view.forward = {direction.x, direction.y, direction.z, 0};
    view.projection = {1080, std::tan(camera.at("fovDegrees").get<float>() * .00872664626f), 1, 1.5f};
    for (uint32_t instance = 0; instance < asset.instanceCount(); ++instance) {
        const auto& masks = selected[instance];
        // MiniZorah metadata preserves the cache's primitive-instance order.
        checkRoam(instance < instanceStates.size(), "Missing instance visibility states");
        checkRoam(instanceStates[instance] == 0u || !masks.empty(), "A visible/deferred instance lost its complete cut");
        if (masks.empty()) { continue; }
        ++verified;
        GPUSceneGpuInstanceRecord metricInstance{};
        std::copy_n(asset.instances()[instance].worldMatrix, 16, metricInstance.worldMatrix.begin());
        std::unordered_set<uint32_t> measured;
        for (const auto& [id, mask] : masks) {
            const auto& group = asset.groups()[id];
            for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                if ((mask & (1u << cluster)) == 0) { continue; }
                const uint32_t child = asset.refinedGroups()[group.clusterRefinedOffset + cluster];
                if (child >= asset.groups().size() || !measured.insert(child).second) { continue; }
                const auto& refine = asset.groups()[child];
                MeshletLodGroupRecord metric;
                std::copy_n(refine.boundsCenterRadius, 4, metric.sphere.begin());
                metric.error = refine.maxQuadricError;
                const float error = meshletLodPixelError(metric, metricInstance, view);
                if (error > 1.5001f) { ++overTarget; }
                if (error > 1e30f) { ++unbounded; }
                else { maxFiniteError = std::max(maxFiniteError, error); }
            }
        }
        const auto roots = asset.primitiveTerminalGroups(asset.instances()[instance].primitiveIndex);
        std::vector<uint32_t> pending(roots.begin(), roots.end());
        std::unordered_set<uint32_t> reached;
        while (!pending.empty()) {
            const uint32_t id = pending.back(); pending.pop_back();
            if (!reached.insert(id).second) { continue; }
            const auto& group = asset.groups()[id];
            const auto found = masks.find(id);
            const uint32_t mask = found == masks.end() ? 0u : found->second;
            const uint32_t valid = group.clusterCount == 32 ? UINT32_MAX : (1u << group.clusterCount) - 1u;
            checkRoam((mask & ~valid) == 0, "Selected cluster outside group");
            for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                if ((mask & (1u << cluster)) != 0) { continue; }
                const uint32_t child = asset.refinedGroups()[group.clusterRefinedOffset + cluster];
                checkRoam(child < asset.groups().size(), "Uncovered cluster has no refinement");
                pending.push_back(child);
            }
        }
        for (const auto& [id, mask] : masks) { checkRoam(reached.contains(id) && mask != 0, "Unreachable selected group"); }
        for (uint32_t id : reached) {
            const auto& group = asset.groups()[id];
            uint32_t expected = 0;
            for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                const uint32_t child = asset.refinedGroups()[group.clusterRefinedOffset + cluster];
                if (!reached.contains(child)) { expected |= 1u << cluster; }
            }
            const auto found = masks.find(id);
            checkRoam(expected == (found == masks.end() ? 0u : found->second), "Shared refinement parents disagree");
        }
    }
    const auto early = observer.read<uint32_t>("AfterStreamEarlyBins");
    const auto late = observer.read<uint32_t>("AfterStreamLateBins");
    checkRoam(early[12] == earlyCandidates && late[12] >= recoveredCandidates && late[12] <= lateCandidateLimit &&
        early[14] == 0 && late[14] == 0 && early[0] + early[4] <= early[12] && late[0] + late[4] <= late[12],
        "Phase candidates disagree with visible/recovered instances or overflowed");
    return {{"verifiedInstances", verified}, {"activeGroups", header.activeGroupCount}, {"fineGroups", fineGroups},
        {"selectedClusters", clusters}, {"earlyCandidates", early[12]}, {"lateCandidates", late[12]},
        {"candidateCapacity", early[5]}, {"earlyHardware", early[0]},
        {"earlySoftware", early[4]}, {"lateHardware", late[0]}, {"lateSoftware", late[4]},
        {"capacityFallback", header.overflowCount != 0}, {"overTargetRefinements", overTarget},
        {"nearPlaneUnboundedRefinements", unbounded}, {"maxFiniteRefinementErrorPixels", maxFiniteError}};
}

Json percentiles(std::vector<double> samples)
{
    if (samples.empty()) { return Json(); }
    std::sort(samples.begin(), samples.end());
    const auto p = [&](double fraction) { return samples[size_t(fraction * double(samples.size() - 1))]; };
    return {{"samples", samples.size()}, {"p50", p(.5)}, {"p95", p(.95)}, {"p99", p(.99)}, {"max", samples.back()}};
}

class MiniZorahRoamingTest final : public RhiTest {
public:
    MiniZorahRoamingTest() { type = RhiTestType::Rendering; name = "minizorah_roaming"; }
    RhiTestResult run(RhiTestContext& context) override;
};

RhiTestResult MiniZorahRoamingTest::run(RhiTestContext& context)
{
    if (!std::getenv("METALLIC_TEST_MINIZORAH") || std::string_view(std::getenv("METALLIC_TEST_MINIZORAH")) != "1") {
        return RhiTestResult::skip("Set METALLIC_TEST_MINIZORAH=1 for the cooked scene roaming test");
    }
    const auto setting = [](const char* name, uint32_t fallback) {
        const char* value = std::getenv(name);
        return value ? static_cast<uint32_t>(std::stoul(value)) : fallback;
    };
    const uint32_t duration = setting("METALLIC_MINIZORAH_ROAM_SECONDS", 660);
    const uint64_t budget = uint64_t(setting("METALLIC_MINIZORAH_ROAM_MIB", 1024)) << 20u;
    Json report{{"status", "running"}, {"durationSeconds", duration}, {"pageBudgetBytes", budget},
        {"resolution", {1920, 1080}}, {"targetPixelError", 1.5}, {"routePeriodSeconds", 60},
        {"timingScope", "Synchronous GPUDriven + MaterialResolve offscreen graph; no presentation blit, output/debug readback on timed frames; excludes checkpoint verification"},
        {"samples", Json::array()}};
    RoamingObserver observer;
    RenderView viewport;
    RenderGraphPreviewRenderer preview;
    std::vector<double> frameTimes, gpuTimes, cpuTimes;
    double elapsed = 0;
    const auto save = [&]() {
        report["renderSeconds"] = elapsed;
        report["frameMilliseconds"] = percentiles(frameTimes);
        report["gpuMilliseconds"] = percentiles(gpuTimes);
        report["cpuRecordMilliseconds"] = percentiles(cpuTimes);
        std::ofstream(context.outputDirectory / "MiniZorahRoamingReport.json") << report.dump(2) << '\n';
    };
    try {
        std::filesystem::create_directories(context.outputDirectory);
        checkRoam(duration > 0 && duration <= 3600 && budget >= (16ull << 20), "Invalid roaming duration/budget");
        RenderSampleLoadResult sample;
        std::string log;
        checkRoam(loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log), log);
        auto graph = std::move(sample.graph);
        // There is no swapchain in this test. Excluding the presentation copy
        // also lets the existing graphics timing envelope cover HW/SW joins.
        graph.removeNode(graph.findNode("FinalBlit")->id);
        graph.markOutput("MaterialResolve.color");
        graph.markOutput("GPUDriven.visibility");
        const auto node = graph.findNode("GPUDriven")->id;
        auto& props = graph.findNode(node)->properties;
        props["maxResidentBytes"] = budget;
        props["screenSpacePagePriority"] = setting("METALLIC_MINIZORAH_PAGE_PRIORITY", 1) != 0;
        report["screenSpacePagePriority"] = props["screenSpacePagePriority"];
        props["debugStreamingPages"] = false;
        const Json original = graph.viewProperties().at("camera");
        checkRoam(props.value("viewBinding", "") == "global" && viewport.setCameraProperties(original),
            "MiniZorah must initialize a shared viewport camera");
        preview.bindRenderView(&viewport);
        report["cameraControl"] = "Shared RenderView, same as editor viewport input";
        scene::MeshletStreamAsset asset;
        checkRoam(asset.open(std::filesystem::path(PROJECT_SOURCE_DIR) / props.at("streamAssetPath").get<std::string>(), log), log);
        scene::Bounds bounds;
        for (const auto& instance : asset.instances()) {
            const auto& b = asset.primitives()[instance.primitiveIndex].bounds;
            const auto& m = instance.worldMatrix;
            for (uint32_t corner = 0; corner < 8; ++corner) {
                const float x = corner & 1 ? b.max[0] : b.min[0], y = corner & 2 ? b.max[1] : b.min[1], z = corner & 4 ? b.max[2] : b.min[2];
                bounds.include(float3(m[0]*x + m[4]*y + m[8]*z + m[12], m[1]*x + m[5]*y + m[9]*z + m[13], m[2]*x + m[6]*y + m[10]*z + m[14]));
            }
        }
        const auto center = (bounds.min + bounds.max) * .5f;
        const float radius = length(bounds.max - bounds.min) * .5f;
        Json farCamera = original;
        farCamera["eye"] = {center.x + radius*2, center.y + radius*1.4f, center.z + radius*2};
        farCamera["center"] = {center.x, center.y, center.z}; farCamera["znear"] = radius*.001f; farCamera["zfar"] = radius*5;
        const auto cameraAt = [&](double seconds) {
            const double phase = std::fmod(seconds, 60.0);
            if ((phase >= 20 && phase < 25) || (phase >= 50 && phase < 55)) { return farCamera; }
            Json camera = original;
            const float move = phase >= 15 && phase < 30 ? 2.7f : 2.f * (1.f - std::cos(float(phase) * .2f));
            const float yaw = phase >= 30 && phase < 50 ? .9f * std::sin(float(phase - 30) * .3f) : .25f * std::sin(float(phase) * .3f);
            const float dx = original["center"][0].get<float>() - original["eye"][0].get<float>();
            const float dz = original["center"][2].get<float>() - original["eye"][2].get<float>();
            const float distance = std::sqrt(dx*dx + dz*dz);
            camera["eye"][0] = original["eye"][0].get<float>() + dx / distance * move;
            camera["eye"][2] = original["eye"][2].get<float>() + dz / distance * move;
            camera["center"][0] = camera["eye"][0].get<float>() + dx*std::cos(yaw) - dz*std::sin(yaw);
            camera["center"][2] = camera["eye"][2].get<float>() + dx*std::sin(yaw) + dz*std::cos(yaw);
            return camera;
        };
        checkRoam(bool(preview.initialize(context.enableValidation, false, false)), preview.lastLog());
        preview.setDebugObserver(&observer);
        for (uint32_t frame = 0; frame < 240 && !observer.latest.value("terminalReady", false); ++frame) {
            checkRoam(bool(preview.render(graph, 1920, 1080, "MaterialResolve.color", false)), preview.lastLog());
        }
        checkRoam(observer.latest.value("terminalReady", false), "Terminal pages did not become ready");
        double nextCheckpoint = 0;
        uint64_t maxFineGroups = 0, maxDeferred = 0;
        while (elapsed < duration) {
            const auto camera = cameraAt(elapsed);
            checkRoam(viewport.setCameraProperties(camera), "Viewport rejected roaming camera");
            if (elapsed >= nextCheckpoint) {
                observer.capture = true;
                preview.setDebugObserver(&observer);
                checkRoam(bool(preview.render(graph, 1920, 1080, "GPUDriven.visibility")), preview.lastLog());
                const auto rasterInfo = observer.read<VisibilityBufferFrameInfo>("rasterInfo").at(0);
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    checkRoam(std::abs(rasterInfo.eye[axis] - camera.at("eye")[axis].get<float>()) < 1e-5f &&
                        std::abs(rasterInfo.center[axis] - camera.at("center")[axis].get<float>()) < 1e-5f,
                        "Viewport position/rotation did not reach the rendered camera");
                }
                checkRoam(!graph.findNode(node)->runtimeProperties.contains("camera"),
                    "Roaming must not bypass the viewport by rewriting the pass camera");
                const auto cut = validateRoamingCut(observer, asset, camera);
                const Json stageTimes = observer.readTimings();
                std::vector<RenderGraphExecutionStats> capturedTimings;
                checkRoam(bool(preview.collectCompletedGpuExecutionStats(capturedTimings)), "Cannot collect checkpoint GPU timings");
                Json nodeTimes;
                for (const auto& timing : capturedTimings) {
                    if (timing.executionId == preview.executionStats().executionId) {
                        for (const auto& pass : timing.nodes) {
                            if (pass.gpuTimingAvailable) { nodeTimes[pass.name] = pass.gpuMilliseconds; }
                        }
                    }
                }
                const auto covered = std::count_if(preview.pixels().begin(), preview.pixels().end(), [](uint32_t p) { return p != 0; });
                checkRoam(covered > 64, "Roaming camera lost useful geometry coverage");
                const auto stats = observer.latest.at("stats");
                checkRoam(observer.latest.value("terminalReady", false) && stats.at("usedResidentBytes").get<uint64_t>() <= budget,
                    "Root coverage/page budget violated");
                checkRoam(stats.at("totalPageLoadFailureCount") == 0 && stats.at("totalGpuInvalidRequestCount") == 0 &&
                    stats.at("frameEvictionScanCount").get<uint32_t>() <= 1, "Streaming error or repeated eviction scan");
                maxFineGroups = std::max(maxFineGroups, cut.at("fineGroups").get<uint64_t>());
                maxDeferred = std::max(maxDeferred, stats.at("frameAllocationDeferredCount").get<uint64_t>());
                report["samples"].push_back({{"seconds", elapsed}, {"camera", camera}, {"coverage", covered}, {"cut", cut},
                    {"streaming", observer.latest}, {"pagePriorities", observer.readPagePriorities()}, {"memory", roamingMemory()},
                    {"checkpointGpuMilliseconds", stageTimes}, {"passGpuMilliseconds", nodeTimes}});
                if (nextCheckpoint <= 60) {
                    observer.capture = false;
                    checkRoam(bool(preview.render(graph, 1920, 1080, "MaterialResolve.color")), preview.lastLog());
                    checkRoam(saveRgba8Png(context.outputDirectory / ("roam-" + std::to_string(int(nextCheckpoint)) + ".png"),
                        reinterpret_cast<const uint8_t*>(preview.pixels().data()), 1920, 1080, log), log);
                }
                std::printf("[MiniZorahRoaming] %.1fs groups=%u candidates=%llu pool=%.1fMiB queued=%u\n", elapsed,
                    cut.at("activeGroups").get<uint32_t>(), static_cast<unsigned long long>(cut.at("selectedClusters").get<uint64_t>()),
                    stats.at("usedResidentBytes").get<double>() / 1048576.0, stats.at("queuedUploadCount").get<uint32_t>());
                std::fflush(stdout);
                nextCheckpoint += 5;
                save();
                std::vector<RenderGraphExecutionStats> checkpointTimings;
                checkRoam(bool(preview.collectCompletedGpuExecutionStats(checkpointTimings)), "Cannot retire checkpoint timings");
            }
            preview.setDebugObserver(nullptr);
            const auto start = Clock::now();
            checkRoam(bool(preview.render(graph, 1920, 1080, "MaterialResolve.color", false)), preview.lastLog());
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            elapsed += ms * .001;
            frameTimes.push_back(ms);
            cpuTimes.push_back(preview.executionStats().cpuMilliseconds);
            checkRoam(preview.pixels().empty(), "Timed frame performed an output readback");
            std::vector<RenderGraphExecutionStats> timings;
            checkRoam(bool(preview.collectCompletedGpuExecutionStats(timings)), "Cannot collect GPU timings");
            for (const auto& timing : timings) {
                if (timing.gpuTimingAvailable) { gpuTimes.push_back(timing.gpuMilliseconds); }
            }
        }
        checkRoam(maxFineGroups > 0, "Roaming used only the terminal LOD");
        if (duration >= 660) {
            uint64_t warmPeak = 0, finalPeak = 0;
            for (const auto& sample : report.at("samples")) {
                const double seconds = sample.at("seconds");
                const uint64_t bytes = sample.at("memory").value("processLocalGpuBytes", uint64_t(0));
                if (seconds >= 120 && seconds < 180) { warmPeak = std::max(warmPeak, bytes); }
                if (seconds >= 600) { finalPeak = std::max(finalPeak, bytes); }
            }
            checkRoam(warmPeak > 0 && finalPeak > 0, "Process GPU memory measurements unavailable");
            checkRoam(finalPeak <= warmPeak + (64ull << 20), "Process GPU memory grew beyond the warm-cycle allowance");
            report["gpuMemoryPlateau"] = {{"warmCyclePeakBytes", warmPeak}, {"finalCyclePeakBytes", finalPeak},
                {"allowedGrowthBytes", 64ull << 20}};
        }
        report["maxFineGroups"] = maxFineGroups; report["maxAllocationDeferred"] = maxDeferred;
        report["tenMinuteCruise"] = duration >= 660;
        report["status"] = "passed";
        save();
        // Snapshot buffers belong to the preview device and must die first.
        observer.copies.clear();
        observer.timestamps.reset();
        return RhiTestResult::pass("Fixed-budget route and DAG coverage verified");
    } catch (const std::exception& error) {
        report["status"] = "failed"; report["error"] = error.what(); save();
        observer.copies.clear();
        observer.timestamps.reset();
        return RhiTestResult::fail(error.what());
    }
}

METALLIC_REGISTER_RHI_TEST(MiniZorahRoamingTest);
} // namespace
} // namespace metallic::tests
