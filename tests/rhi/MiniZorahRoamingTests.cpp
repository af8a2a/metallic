#include "RhiTest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Profiling/CpuPhaseTrace.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/RenderView.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

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
#include <unordered_map>
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
    static constexpr std::array<std::string_view, 26> stages = {"BeforeStreamUpdates", "AfterStreamUpdates",
        "AfterStreamPriorityClear", "AfterStreamStateClear", "AfterStreamDemand", "AfterStreamFrontier", "AfterStreamMask", "AfterStreamPrefix", "AfterStreamEmit", "AfterStreamPrefetch", "AfterTraversal", "AfterEarlyCull",
        "AfterStreamEarlyCandidates", "AfterStreamEarlyClusterCull", "AfterStreamEarlyClassify", "AfterStreamEarlyBins", "AfterStreamEarlyRaster", "AfterStreamEarlyResolve",
        "AfterLateCull", "AfterStreamLateCandidates", "AfterStreamLateClusterCull", "AfterStreamLateClassify", "AfterStreamLateBins", "AfterStreamLateRaster", "AfterStreamLateResolve", "AfterPass"};
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
            } else if (checkpoint == "AfterTraversal" && resource.id == "streaming.GPUDriven.lodState") {
                size = resource.size != 0 ? resource.size : resource.buffer->desc().size; key = "lodState";
            } else if (checkpoint == "AfterTraversal" && resource.id == "streaming.GPUDriven.demandStats") {
                size = resource.size; key = "demandStats";
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
                resource.id == "streaming.GPUDriven.loadPriorities" || resource.id == "streaming.GPUDriven.blasHeader")) {
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
        checkRoam(copies.contains(name), "Missing roaming snapshot: " + name);
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
        result["afterTraversalToEnd"] = timestamps->durationMilliseconds(values[std::find(stages.begin(), stages.end(), "AfterTraversal") - stages.begin()].value, values.back().value);
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
    const auto rasterInfo = observer.read<VisibilityBufferFrameInfo>("rasterInfo").front();
    const auto header = observer.read<MeshletStreamGpuActiveHeader>("header").at(0);
    const auto rows = observer.read<MeshletStreamGpuActiveGroup>("groups");
    checkRoam(header.activeGroupCount <= rows.size() && header.overflowCount < 2, "Invalid/empty capacity fallback");
    const auto instanceStates = observer.read<uint32_t>("instances");
    const auto lateInstanceStates = observer.read<uint32_t>("lateInstances");
    std::vector<std::map<uint32_t, uint32_t>> selected(asset.instanceCount());
    const auto lodState = observer.read<uint32_t>("lodState");
    uint64_t visitedNodes = 0, testedGroups = 0, flatTileTests = 0, demandSeedRootTests = 0;
    uint64_t base = uint64_t(asset.instanceCount()) * 4;
    std::vector<uint32_t> flatTiles(asset.primitiveCount());
    for (uint32_t p = 0; p < asset.primitiveCount(); ++p) {
        const auto& primitive = asset.primitives()[p];
        for (uint32_t end = primitive.groupCount; end != 0;) {
            uint32_t first = end - 1;
            while (first != 0 && end - first < 64 && asset.groups()[primitive.groupOffset + first - 1].lodLevel ==
                asset.groups()[primitive.groupOffset + end - 1].lodLevel) { --first; }
            ++flatTiles[p]; end = first;
        }
    }
    for (const auto& instance : asset.instances()) {
        if (instance.primitiveIndex >= asset.primitiveCount()) { continue; }
        const auto count = asset.primitives()[instance.primitiveIndex].groupCount;
        const uint64_t sparse = base + uint64_t(count) * 2;
        checkRoam(sparse + 3 < lodState.size(), "Traversal stats outside state");
        visitedNodes += lodState[sparse + 1];
        testedGroups += lodState[sparse + 2];
        demandSeedRootTests += lodState[sparse + 3];
        if (lodState[sparse + 1] != 0) { flatTileTests += flatTiles[instance.primitiveIndex]; }
        base += 4ull + count * 3ull;
    }
    Json demandStats;
    if (observer.copies.contains("demandStats")) {
        const auto values = observer.read<uint32_t>("demandStats");
        uint64_t histogramCount = 0;
        for (size_t bin = 8; bin < 16; ++bin) { histogramCount += values[bin]; }
        checkRoam(values[1] <= observer.latest.at("demandTaskCount").get<uint32_t>() && values[1] == values[16] && histogramCount == values[1] &&
            values[4] <= 8 && values[3] <= values[7], "Incomplete or unbounded distributed demand tasks");
        demandStats = {{"distributed", values[17] != 0}, {"groupTestsForPolicy", values[18]},
            {"tasks", values[1]}, {"seedRootTests", demandSeedRootTests}, {"visitedNodes", values[2]}, {"testedGroups", values[3]},
            {"maxNodesPerTask", values[4]}, {"maxGroupsPerTask", values[5]}, {"nonemptyTasks", values[6]},
            {"waveSlots", values[7]}, {"groupCountHistogram", std::vector<uint32_t>(values.begin() + 8, values.begin() + 16)}};
    }
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
    uint64_t overTarget = 0, unbounded = 0, visibleOverTarget = 0, visibleUnbounded = 0;
    float maxFiniteError = 0, maxVisibleError = 0;
    MeshletLodView view;
    for (uint32_t axis = 0; axis < 3; ++axis) { view.eye[axis] = camera.at("eye")[axis].get<float>(); }
    view.eye[3] = camera.at("znear").get<float>();
    float3 direction(camera.at("center")[0].get<float>() - view.eye[0], camera.at("center")[1].get<float>() - view.eye[1],
        camera.at("center")[2].get<float>() - view.eye[2]);
    direction = normalize(direction);
    view.forward = {direction.x, direction.y, direction.z, 0};
    view.projection = {float(rasterInfo.height), std::tan(camera.at("fovDegrees").get<float>() * .00872664626f), 1, 1.5f};
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
                MeshletLodRefinementBounds bounds;
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    const double radius = double(metric.sphere[3]) + metric.error;
                    bounds.min[axis] = std::nextafter(float(double(metric.sphere[axis]) - radius), -INFINITY);
                    bounds.max[axis] = std::nextafter(float(double(metric.sphere[axis]) + radius), INFINITY);
                }
                // Measure only the referenced refinement's own error envelope;
                // ancestor demand bounds deliberately contain additional descendants.
                if (meshletLodBoundsVisible(bounds, metricInstance, view, {0, 1, 0}, float(rasterInfo.width) / rasterInfo.height,
                    camera.at("zfar").get<float>())) {
                    visibleOverTarget += error > 1.5001f;
                    visibleUnbounded += error > 1e30f;
                    maxVisibleError = std::max(maxVisibleError, error);
                }
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
        {"capacityFallback", header.overflowCount != 0}, {"fallbackInstances", header.padding2},
        {"traversalVisitedNodes", visitedNodes}, {"traversalTestedGroups", testedGroups}, {"demand", demandStats},
        {"traversalFlatTileBaseline", flatTileTests}, {"overTargetRefinements", overTarget},
        {"nearPlaneUnboundedRefinements", unbounded}, {"maxFiniteRefinementErrorPixels", maxFiniteError},
        {"visibleOverTargetRefinements", visibleOverTarget}, {"visibleUnboundedRefinements", visibleUnbounded},
        {"maxVisibleRefinementErrorPixels", maxVisibleError}};
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
    const uint32_t fixedView = setting("METALLIC_MINIZORAH_FIXED_VIEW", UINT32_MAX);
    const bool transitionChecks = setting("METALLIC_MINIZORAH_TRANSITION_CHECKS", 0) != 0;
    const bool latencyOnly = setting("METALLIC_MINIZORAH_LATENCY_ONLY", 0) != 0;
    const uint32_t startupTraceFrames = std::min(setting("METALLIC_MINIZORAH_STARTUP_TRACE_FRAMES", 0),
        static_cast<uint32_t>(profiling::CpuPhaseTrace::kMaxFrames));
    auto startupTrace = startupTraceFrames != 0 ? std::make_unique<profiling::CpuPhaseTrace>() : nullptr;
    Json tracedFrames = Json::array(), tracedGpu = Json::array();
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
        if (startupTrace) {
            Json events = Json::array();
            Json gpuSpans = Json::array();
            for (const auto& event : startupTrace->events) {
                events.push_back({{"name", event.name}, {"frame", event.frame}, {"depth", event.depth},
                    {"startMilliseconds", double(event.startNanoseconds) / 1e6},
                    {"milliseconds", double(event.durationNanoseconds) / 1e6}, {"value", event.value}});
            }
            for (const auto& span : startupTrace->gpuSpans) {
                const auto mappedTime = [&](uint64_t timestamp) {
                    return double((static_cast<long double>(timestamp) - span.calibrationTimestamp) *
                        span.periodNanoseconds + span.hostReferenceNanoseconds) / 1e6;
                };
                gpuSpans.push_back({{"executionId", span.executionId},
                    {"startMilliseconds", mappedTime(span.beginTimestamp)}, {"endMilliseconds", mappedTime(span.endTimestamp)},
                    {"alignmentUncertaintyMilliseconds", double(span.calibrationCallNanoseconds / 2 + span.maxDeviationNanoseconds) / 1e6}});
            }
            std::ofstream(context.outputDirectory / "MiniZorahStartupTrace.json") << Json{
                {"events", std::move(events)}, {"frames", tracedFrames}, {"gpu", tracedGpu},
                {"calibratedGpuSpans", std::move(gpuSpans)},
                {"droppedEvents", startupTrace->dropped}}.dump(2) << '\n';
        }
    };
    try {
        std::filesystem::create_directories(context.outputDirectory);
        checkRoam(duration > 0 && duration <= 3600 && budget >= (16ull << 20), "Invalid roaming duration/budget");
        RenderSampleLoadResult sample;
        std::string log;
        checkRoam(loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log), log);
        auto graph = std::move(sample.graph);
        uint64_t previewFrame = 0;
        const auto renderPreview = [&](const char* output, bool readback, const char* phase) {
            const uint64_t frame = previewFrame++;
            auto* trace = startupTrace && frame < startupTraceFrames ? startupTrace.get() : nullptr;
            profiling::CpuPhaseTraceFrame scope(trace, frame);
            const auto start = trace ? Clock::now() : Clock::time_point{};
            const auto result = preview.render(graph, 1920, 1080, output, readback);
            if (trace) {
                tracedFrames.push_back({{"frame", frame}, {"phase", phase},
                    {"milliseconds", std::chrono::duration<double, std::milli>(Clock::now() - start).count()},
                    {"executionId", preview.executionStats().executionId},
                    {"recordMilliseconds", preview.executionStats().cpuMilliseconds}});
            }
            return result;
        };
        // There is no swapchain in this test. Excluding the presentation copy
        // also lets the existing graphics timing envelope cover HW/SW joins.
        graph.removeNode(graph.findNode("FinalBlit")->id);
        graph.markOutput("MaterialResolve.color");
        graph.markOutput("GPUDriven.visibility");
        const auto node = graph.findNode("GPUDriven")->id;
        auto& props = graph.findNode(node)->properties;
        props["maxResidentBytes"] = budget;
        props["screenSpacePagePriority"] = setting("METALLIC_MINIZORAH_PAGE_PRIORITY", 1) != 0;
        props["viewDrivenPageDemand"] = setting("METALLIC_MINIZORAH_VIEW_DEMAND", 1) != 0;
        props["distributedPageDemand"] = setting("METALLIC_MINIZORAH_DISTRIBUTED_DEMAND", 1) != 0;
        props["prefetchPages"] = setting("METALLIC_MINIZORAH_PREFETCH", 1) != 0;
        props["lowLatencyRequests"] = setting("METALLIC_MINIZORAH_LOW_LATENCY", 1) != 0;
        props["completionDrivenUploads"] = setting("METALLIC_MINIZORAH_COMPLETION_UPLOADS", 1) != 0;
        report["screenSpacePagePriority"] = props["screenSpacePagePriority"];
        report["viewDrivenPageDemand"] = props["viewDrivenPageDemand"];
        report["distributedPageDemand"] = props["distributedPageDemand"];
        report["prefetchPages"] = props["prefetchPages"];
        report["lowLatencyRequests"] = props["lowLatencyRequests"];
        report["completionDrivenUploads"] = props["completionDrivenUploads"];
        report["transitionChecks"] = transitionChecks;
        report["latencyOnly"] = latencyOnly;
        props["debugStreamingPages"] = false;
        const Json original = graph.viewProperties().at("camera");
        checkRoam(props.value("viewBinding", "") == "global" && viewport.setCameraProperties(original),
            "MiniZorah must initialize a shared viewport camera");
        preview.bindRenderView(&viewport);
        report["cameraControl"] = "Shared RenderView, same as editor viewport input";
        report["fixedViewSeconds"] = fixedView == UINT32_MAX ? Json(nullptr) : Json(fixedView);
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
            if (fixedView != UINT32_MAX) { seconds = double(fixedView); }
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
        const auto terminalStart = Clock::now();
        uint32_t terminalFrames = 0;
        for (; terminalFrames < 240 && !observer.latest.value("terminalReady", false); ++terminalFrames) {
            checkRoam(bool(renderPreview("MaterialResolve.color", false, "terminal")), preview.lastLog());
        }
        checkRoam(observer.latest.value("terminalReady", false), "Terminal pages did not become ready");
        report["terminalReadyFrames"] = terminalFrames;
        report["terminalReadyWallSeconds"] = std::chrono::duration<double>(Clock::now() - terminalStart).count();
        // Measure wall-clock request tails without interleaving expensive cut
        // readbacks/PNG saves; retain one final integrity/latency checkpoint.
        double nextCheckpoint = latencyOnly ? double(duration - 1u) : 0.0;
        const uint32_t convergenceDeadline = setting("METALLIC_MINIZORAH_CONVERGENCE_SECONDS",
            fixedView != UINT32_MAX && budget >= (1ull << 30) ? 5u : UINT32_MAX);
        report["convergenceDeadlineSeconds"] = convergenceDeadline == UINT32_MAX ? Json(nullptr) : Json(convergenceDeadline);
        report["firstConvergedSeconds"] = nullptr;
        report["firstConvergedWallSeconds"] = nullptr;
        const auto roamingStart = Clock::now();
        uint64_t maxFineGroups = 0, maxDeferred = 0;
        while (elapsed < duration) {
            const auto camera = cameraAt(elapsed);
            checkRoam(viewport.setCameraProperties(camera), "Viewport rejected roaming camera");
            if (elapsed >= nextCheckpoint) {
                observer.capture = true;
                preview.setDebugObserver(&observer);
                checkRoam(bool(renderPreview("GPUDriven.visibility", true, "checkpoint")), preview.lastLog());
                const auto rasterInfo = observer.read<VisibilityBufferFrameInfo>("rasterInfo").at(0);
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    checkRoam(std::abs(rasterInfo.eye[axis] - camera.at("eye")[axis].get<float>()) < 1e-5f &&
                        std::abs(rasterInfo.center[axis] - camera.at("center")[axis].get<float>()) < 1e-5f,
                        "Viewport position/rotation did not reach the rendered camera");
                }
                checkRoam(!graph.findNode(node)->runtimeProperties.contains("camera"),
                    "Roaming must not bypass the viewport by rewriting the pass camera");
                const auto cut = validateRoamingCut(observer, asset, camera);
                if (cut.at("visibleOverTargetRefinements") == 0 && report["firstConvergedSeconds"].is_null()) {
                    report["firstConvergedSeconds"] = elapsed;
                    report["firstConvergedWallSeconds"] = std::chrono::duration<double>(Clock::now() - roamingStart).count();
                }
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
                report["samples"].push_back({{"seconds", elapsed},
                    {"wallSeconds", std::chrono::duration<double>(Clock::now() - roamingStart).count()},
                    {"timedFrames", frameTimes.size()}, {"camera", camera}, {"coverage", covered}, {"cut", cut},
                    {"streaming", observer.latest}, {"pagePriorities", observer.readPagePriorities()}, {"memory", roamingMemory()},
                    {"checkpointGpuMilliseconds", stageTimes}, {"passGpuMilliseconds", nodeTimes}});
                if (!latencyOnly && nextCheckpoint <= 60 && std::floor(nextCheckpoint) == nextCheckpoint) {
                    observer.capture = false;
                    checkRoam(bool(renderPreview("MaterialResolve.color", true, "image")), preview.lastLog());
                    checkRoam(saveRgba8Png(context.outputDirectory / ("roam-" + std::to_string(int(nextCheckpoint)) + ".png"),
                        reinterpret_cast<const uint8_t*>(preview.pixels().data()), 1920, 1080, log), log);
                }
                std::printf("[MiniZorahRoaming] %.1fs groups=%u candidates=%llu pool=%.1fMiB queued=%u visibleOverTarget=%llu\n", elapsed,
                    cut.at("activeGroups").get<uint32_t>(), static_cast<unsigned long long>(cut.at("selectedClusters").get<uint64_t>()),
                    stats.at("usedResidentBytes").get<double>() / 1048576.0, stats.at("queuedUploadCount").get<uint32_t>(),
                    static_cast<unsigned long long>(cut.at("visibleOverTargetRefinements").get<uint64_t>()));
                std::fflush(stdout);
                const double phase = std::fmod(nextCheckpoint, 60.0);
                const bool transition = transitionChecks && fixedView == UINT32_MAX && (phase < 2 ||
                    (phase >= 20 && phase < 22) || (phase >= 25 && phase < 27) || (phase >= 30 && phase < 32) ||
                    (phase >= 50 && phase < 52) || (phase >= 55 && phase < 57));
                nextCheckpoint = transition ? nextCheckpoint + .25 : fixedView != UINT32_MAX && nextCheckpoint < 5 ?
                    nextCheckpoint + .5 : (std::floor(nextCheckpoint / 5.0) + 1.0) * 5.0;
                if (latencyOnly) { nextCheckpoint = double(duration); }
                save();
                checkRoam(elapsed < convergenceDeadline || cut.at("visibleOverTargetRefinements") == 0,
                    "Visible refinements failed the 1.5 px convergence deadline");
                std::vector<RenderGraphExecutionStats> checkpointTimings;
                checkRoam(bool(preview.collectCompletedGpuExecutionStats(checkpointTimings)), "Cannot retire checkpoint timings");
            }
            preview.setDebugObserver(nullptr);
            const auto start = Clock::now();
            checkRoam(bool(renderPreview("MaterialResolve.color", false, "timed")), preview.lastLog());
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            elapsed += ms * .001;
            frameTimes.push_back(ms);
            cpuTimes.push_back(preview.executionStats().cpuMilliseconds);
            checkRoam(preview.pixels().empty(), "Timed frame performed an output readback");
            std::vector<RenderGraphExecutionStats> timings;
            checkRoam(bool(preview.collectCompletedGpuExecutionStats(timings)), "Cannot collect GPU timings");
            for (const auto& timing : timings) {
                if (timing.gpuTimingAvailable) { gpuTimes.push_back(timing.gpuMilliseconds); }
                if (startupTrace && timing.executionId <= startupTraceFrames) {
                    tracedGpu.push_back({{"executionId", timing.executionId},
                        {"available", timing.gpuTimingAvailable}, {"milliseconds", timing.gpuMilliseconds}});
                }
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

// Fixed input replay, separate from the wall-clock-driven quality/roaming test.
// Timing runs keep multiple submissions in flight and have no debug observer.
class MiniZorahBaselineTest final : public RhiTest {
public:
    MiniZorahBaselineTest()
    {
        type = RhiTestType::Rendering;
        name = "minizorah_fixed_baseline";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (!std::getenv("METALLIC_TEST_MINIZORAH")) {
            return RhiTestResult::skip("Set METALLIC_TEST_MINIZORAH=1");
        }
        const auto setting = [](const char* name, uint32_t fallback) {
            const char* value = std::getenv(name);
            return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : fallback;
        };
        const bool clas = setting("METALLIC_MINIZORAH_BENCH_CLAS", 1) != 0;
        const bool quality = setting("METALLIC_MINIZORAH_BENCH_QUALITY", 0) != 0;
        const bool realtime = setting("METALLIC_MINIZORAH_BENCH_REALTIME", 0) != 0;
        if (realtime && !context.device.capabilities().streamlineDlssSr) {
            return RhiTestResult::skip("Realtime replay requires --rhi-realtime and DLSS-SR");
        }
        uint32_t frameCount = setting("METALLIC_MINIZORAH_BENCH_FRAMES", 8400);
        Json replay;
        Json report{{"protocol", "minizorah-fixed-v1"}, {"status", "running"},
            {"clasEnabled", clas}, {"qualityRun", quality}, {"validation", context.enableValidation},
            {"resolution", {1920, 1080}}, {"lodPixelError", 1.5}, {"frameCount", frameCount},
            {"routeStepSeconds", 1.0 / 60.0}, {"paced", false},
            {"timingScope", "Multi-frame offscreen GPUDriven + MaterialResolve; GPU graphics envelope includes async joins. CPU execute includes frame-slot wait. No UI, presentation, Streamline, Aftermath or timed-frame debug readback."},
            {"cacheScope", "New process/device GPU residency; existing cook, OS file cache and persistent PSO cache. Repeated route retains normal eviction policy."},
            {"quality", Json::array()}};
        report["realtime"] = realtime;
        if (realtime) {
            report["timingScope"] = "Offscreen streamed realtime graph including BLAS/TLAS, shadows, deferred lighting, DLSS-SR, exposure and final blit. No UI/present or timed-frame output readback. CPU phase trace enabled in both baseline and candidate.";
        }
        std::filesystem::create_directories(context.outputDirectory);
        const auto save = [&]() { std::ofstream(context.outputDirectory / "Baseline.json") << report.dump(2) << '\n'; };
        std::unique_ptr<Device> device;
        RoamingObserver observer;
        try {
            if (const char* replayPath = std::getenv("METALLIC_MINIZORAH_REPLAY")) {
                std::ifstream input(replayPath);
                checkRoam(input.good(), "Cannot open external camera replay");
                replay = Json::parse(input);
                checkRoam(replay.at("frames").is_array(), "Invalid camera replay");
                frameCount = static_cast<uint32_t>(replay.at("frames").size());
                report["protocol"] = replay.at("protocol");
                report["externalReplay"] = replayPath;
                report["frameCount"] = frameCount;
            }
            checkRoam(frameCount >= 600 && frameCount <= 8400, "Baseline frame count must be 600..8400");
            RenderSampleLoadResult sample;
            std::string log;
            checkRoam(loadBuiltInRenderSample(realtime ? kDefaultGPUDrivenSampleId : "gpu-driven-minizorah-vbuffer", sample, log), log);
            auto graph = std::move(sample.graph);
            if (realtime) {
                checkRoam(graph.renameNode(graph.findNode("VBuffer")->id, "GPUDriven"), "Cannot name replay producer");
            } else {
                graph.removeNode(graph.findNode("FinalBlit")->id);
                graph.markOutput("MaterialResolve.color");
            }
            graph.markOutput("GPUDriven.visibility");
            auto& props = graph.findNode("GPUDriven")->properties;
            props["enableClas"] = clas;
            props["compactClas"] = true;
            props["maxResidentBytes"] = 1ull << 30;
            props["maxClasBytes"] = 512ull << 20;
            props["maxClasBuildClusters"] = 8192;
            props["coldPageRetentionFrames"] = 120;
            props["debugStreamingPages"] = false;
            if (const char* path = std::getenv("METALLIC_MINIZORAH_STREAM_ASSET")) { props["streamAssetPath"] = path; }
            if (const char* mode = std::getenv("METALLIC_MINIZORAH_GPU_DECOMPRESSION")) {
                checkRoam(std::string_view(mode) == "0" || std::string_view(mode) == "1", "Invalid GPU decompression setting");
                props["enableGpuDecompression"] = std::string_view(mode) == "1";
            }
            if (const char* queues = std::getenv("METALLIC_MINIZORAH_RASTER_QUEUES")) {
                const std::string policy(queues);
                checkRoam(policy == "Off" || policy == "Early" || policy == "All", "Invalid raster queue policy");
                props["asyncSoftwareRaster"] = policy != "Off";
                props["asyncLateRaster"] = policy == "All";
            }
            report["rasterQueues"] = {{"asyncSoftwareRaster", props.value("asyncSoftwareRaster", true)},
                {"asyncLateRaster", props.value("asyncLateRaster", false)}};
            for (const char* key : {"screenSpacePagePriority", "viewDrivenPageDemand", "distributedPageDemand", "prefetchPages",
                     "lowLatencyRequests", "completionDrivenUploads", "measurePageLatency"}) { props[key] = true; }
            props["distributedPageDemand"] = setting("METALLIC_MINIZORAH_DISTRIBUTED_DEMAND", 1) != 0;
            if (const char* demand = std::getenv("METALLIC_MINIZORAH_DISTRIBUTED_DEMAND"); demand && std::string_view(demand) == "1") {
                props["distributedDemandMinGroups"] = 0u;
            }
            props["maxTraversalWorkers"] = setting("METALLIC_MINIZORAH_DEMAND_WORKERS", props.value("maxTraversalWorkers", 1024u));
            if (!replay.is_null()) {
                auto viewProperties = graph.viewProperties();
                viewProperties["camera"] = replay.at("originalCamera");
                graph.setViewProperties(std::move(viewProperties));
            }
            report["graph"] = Json::parse(serializeRenderGraphToString(graph));
            const Json original = graph.viewProperties().at("camera");
            scene::MeshletStreamAsset asset;
            const auto assetPath = std::filesystem::path(PROJECT_SOURCE_DIR) / props.at("streamAssetPath").get<std::string>();
            checkRoam(asset.open(assetPath, log), log);
            report["asset"] = {{"path", assetPath.generic_string()}, {"fileBytes", std::filesystem::file_size(assetPath)},
                {"pages", asset.pageCount()}, {"instances", asset.instanceCount()}};
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
            farCamera["center"] = {center.x, center.y, center.z};
            farCamera["znear"] = radius*.001f; farCamera["zfar"] = radius*5;
            const auto cameraAt = [&](uint32_t f) {
                if (!replay.is_null()) { return replay.at("frames").at(f).at("camera"); }
                if (f < 600 || f >= 7800) { return original; }
                const double phase = double((f - 600) % 3600) / 60.0;
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
            const auto phaseAt = [&](uint32_t f) -> std::string {
                if (!replay.is_null()) { return replay.at("frames").at(f).at("phase").get<std::string>(); }
                if (f < 300) { return "cold_start"; }
                if (f < 600) { return "static_warm"; }
                if (f < 4200) { return "roam_first"; }
                if (f < 7800) { return "roam_repeat"; }
                return f < 8100 ? "settle" : "static_return";
            };
            std::vector<Json> cameras;
            for (uint32_t f = 0; f < frameCount; ++f) { cameras.push_back(cameraAt(f)); }
            std::ofstream(context.outputDirectory / "Cameras.json") << Json(cameras).dump();
            DeviceDesc desc;
            desc.applicationName = "MiniZorah fixed baseline";
            desc.enableValidation = context.enableValidation;
            desc.enableBindlessDescriptorHeap = true;
            desc.enableShaderObject = true;
            desc.enableMeshShader = true;
            desc.enableTaskShader = true;
            desc.enableTaskShaderSubgroupBallot = true;
            desc.enableGeometryShader = true;
            desc.enableSubgroupSizeControl = true;
            desc.enableComputeFullSubgroups = true;
            desc.preferredTaskSubgroupSize = 32;
            desc.enableAsyncCompute = true;
            // Both configurations use the same enabled device capabilities.
            desc.enableRayTracingAccelerationStructure = true;
            desc.enablePushDescriptor = true;
            desc.enableRayQuery = true;
            desc.enableClusterAccelerationStructure = true;
            auto start = Clock::now();
            if (!realtime) {
                const auto result = createDevice(desc, device);
                checkRoam(bool(result), std::string("Baseline device: ") + toString(result));
            }
            Device* replayDevice = realtime ? &context.device : device.get();
            report["deviceCreateMs"] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            RenderView view;
            checkRoam(view.setCameraProperties(original), "Invalid initial camera");
            view.setTemporalJitter(realtime);
            scene::Scene runtimeScene;
            RenderWorld world;
            if (realtime) {
                world.setEnvironment({.enabled = true, .path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
                scene::LightingSettings lighting;
                lighting.autoExposure.enabled = true;
                lighting.exposureEV100 = 2;
                scene::PunctualLight sun;
                sun.properties.type = "directional"; sun.properties.intensity = 10;
                sun.direction = float3(.6f, -1, -.3f);
                lighting.lights.push_back(sun);
                world.setLighting(lighting);
                report["lighting"] = {{"sunDirection", {.6f, -1.0f, -.3f}}, {"sunIntensity", 10},
                    {"environment", sample.desc.environment->path}, {"autoExposure", true}};
            }
            HistoryResourceManager history;
            checkRoam(bool(history.initialize(*replayDevice)), "History initialization failed");
            RenderGraphExecutor executor;
            executor.bindRenderView(&view);
            if (realtime) { executor.bindRenderWorld(&world); }
            else { executor.bindRuntimeScene(&runtimeScene); }
            // Register transfer-readable debug resources before compilation. The
            // observer is detached during timed frames, so no copies are recorded.
            executor.setDebugObserver(&observer);
            start = Clock::now();
            checkRoam(bool(executor.compile(*replayDevice, graph, 1920, 1080, log)), log);
            report["compileMs"] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            struct Frame {
                double executeMs = 0, hostMs = 0;
                bool overlap = false;
                std::map<std::string, double> cpuPhases;
                RenderGraphExecutionStats stats;
            };
            std::vector<Frame> frames(frameCount);
            std::unordered_map<uint64_t, uint32_t> executionFrames;
            const auto collect = [&]() {
                std::vector<RenderGraphExecutionStats> completed;
                checkRoam(bool(executor.collectCompletedGpuExecutionStats(completed)), "GPU timing collection failed");
                for (auto& stats : completed) {
                    const auto found = executionFrames.find(stats.executionId);
                    if (found == executionFrames.end()) { continue; } // Final diagnostic frame is untimed.
                    auto& frame = frames[found->second];
                    checkRoam(!frame.stats.gpuTimingAvailable && stats.gpuTimingAvailable && !stats.profilingOverflow,
                        "Missing, duplicate or overflowing GPU timing sample");
                    frame.stats = std::move(stats);
                }
            };
            const auto submit = [&]() {
                checkRoam(bool(executor.execute({.graphicsQueue = replayDevice->getQueue(QueueType::Graphics),
                    .computeQueue = replayDevice->getQueue(QueueType::Compute), .historyResources = &history,
                    .slotWaitTimeoutNanoseconds = 30000000000ull})), "Baseline execute failed");
                if (realtime) { checkRoam(bool(vulkan::notifyStreamlineOffscreenFrame()), "Streamline offscreen frame bookkeeping failed"); }
            };
            const auto checkpoint = [&](uint32_t f, const Json& camera, bool final) {
                checkRoam(bool(executor.waitForSubmittedWork(30000000000ull)), "Checkpoint completion failed");
                collect();
                Json entry{{"frame", f}, {"phase", final ? "final_diagnostic" : phaseAt(f)},
                    {"camera", camera}, {"stream", observer.latest}};
                if (observer.latest.value("terminalReady", false)) {
                    const auto cut = validateRoamingCut(observer, asset, camera);
                    entry["cut"] = cut;
                    const auto info = observer.read<VisibilityBufferFrameInfo>("rasterInfo").front();
                    entry["renderExtent"] = {info.width, info.height};
                    if (realtime && observer.copies.contains("streaming.GPUDriven.blasHeader")) {
                        const auto blas = observer.read<MeshletStreamGpuBlasHeader>("streaming.GPUDriven.blasHeader").front();
                        entry["blas"] = {{"builds", blas.blasBuildCount}, {"references", blas.clusterReferenceCount},
                            {"cacheDirty", blas.padding0}, {"activeGroups", blas.padding1}, {"clasRevision", blas.padding2}};
                    }
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        checkRoam(std::abs(info.eye[axis] - camera["eye"][axis].get<float>()) < 1e-5f &&
                            std::abs(info.center[axis] - camera["center"][axis].get<float>()) < 1e-5f,
                            "Camera replay did not reach VBuffer");
                    }
                } else { checkRoam(!final, "Final terminal cut not ready"); }
                report["quality"].push_back(std::move(entry));
                if (final) { checkRoam(report["quality"].back().at("cut").at("visibleOverTargetRefinements") == 0,
                    "Final held view did not converge to 1.5 render px"); }
            };
            const auto runStart = Clock::now();
            profiling::CpuPhaseTrace cpuTrace;
            for (uint32_t f = 0; f < frameCount; ++f) {
                const bool capture = quality && (f == 29 || f == 59 || f == 119 || f == 179 || (f + 1) % 300 == 0);
                if (capture) { checkRoam(bool(executor.waitForSubmittedWork(30000000000ull)), "Pre-checkpoint drain failed"); }
                observer.capture = capture;
                executor.setDebugObserver(capture ? &observer : nullptr);
                const auto hostStart = Clock::now();
                checkRoam(view.setCameraProperties(cameras[f]), "Invalid replay camera");
                frames[f].overlap = executor.lastSubmittedCompletion().valid() && !executor.lastSubmittedCompletion().isComplete();
                const auto executeStart = Clock::now();
                cpuTrace.events.clear();
                cpuTrace.gpuSpans.clear();
                {
                    profiling::CpuPhaseTraceFrame traceFrame(&cpuTrace, f);
                    submit();
                }
                frames[f].executeMs = std::chrono::duration<double, std::milli>(Clock::now() - executeStart).count();
                executionFrames.emplace(executor.executionStats().executionId, f);
                collect();
                frames[f].hostMs = std::chrono::duration<double, std::milli>(Clock::now() - hostStart).count();
                checkRoam(cpuTrace.dropped == 0, "CPU phase trace overflow");
                for (const auto& event : cpuTrace.events) {
                    frames[f].cpuPhases[event.name] += double(event.durationNanoseconds) / 1e6;
                }
                if (capture) { checkpoint(f, cameras[f], false); }
            }
            checkRoam(bool(executor.waitForSubmittedWork(30000000000ull)), "Final GPU drain failed");
            collect();
            report["runWallSeconds"] = std::chrono::duration<double>(Clock::now() - runStart).count();
            report["memoryAfterReplay"] = roamingMemory();
            // No diagnostic work perturbs the preceding performance samples.
            executor.setDebugObserver(&observer);
            observer.capture = true;
            submit();
            checkpoint(frameCount, cameras.back(), true);
            executor.setDebugObserver(nullptr);
            report["finalStream"] = observer.latest;
            std::ofstream output(context.outputDirectory / "Frames.jsonl");
            std::map<std::string, std::vector<double>> gpuByPhase, cpuByPhase, hostByPhase;
            uint32_t overlapCount = 0;
            bool sawAsyncRaster = false;
            for (uint32_t f = 0; f < frameCount; ++f) {
                const auto& frame = frames[f];
                const auto& stats = frame.stats;
                checkRoam(stats.gpuTimingAvailable && stats.streaming.size() == 1, "Frame timing/stream sample missing");
                const auto& s = stats.streaming.front();
                checkRoam(s.clasEnabled == clas && s.geometryUsedBytes <= s.geometryBudgetBytes &&
                    s.clasUsedBytes <= s.clasCapacityBytes && !s.loadFailures && !s.requestOverflows,
                    "Streaming budget, loading or request invariant failed");
                checkRoam(s.clasBuiltClusters <= 8192, "CLAS build budget exceeded");
                const auto& work = s.cpuWork;
                checkRoam(work.demandVisited == work.demandNewerThanFeedback + work.demandUnused +
                    work.demandRefreshed + work.demandIncompleteProtected, "Resident feedback work counts do not partition visits");
                checkRoam(work.coldVisited == work.coldStateRejected + work.coldAgeRejected +
                    work.coldScheduleFailed + work.coldPressureScheduled + work.coldRetentionScheduled &&
                    work.coldClasLookups <= work.coldVisited - work.coldStateRejected,
                    "Cold scheduling work counts do not partition visits");
                Json nodes = Json::array();
                for (const auto& pass : stats.nodes) {
                    checkRoam(pass.gpuTimingAvailable, "Pass GPU timing missing");
                    Json sections = Json::array();
                    for (const auto& section : pass.sections) {
                        checkRoam(section.cpuOnly ? !section.gpuTimingAvailable : section.gpuTimingAvailable,
                            "Scope timing domain mismatch");
                        sawAsyncRaster |= section.name == "Software raster" && section.queue == QueueType::Compute;
                        sections.push_back({{"name", section.name}, {"parent", section.parent},
                            {"queue", section.cpuOnly ? "cpu" : section.queue == QueueType::Compute ? "compute" : "graphics"},
                            {"cpuOnly", section.cpuOnly}, {"gpuTimingAvailable", section.gpuTimingAvailable},
                            {"gpuMs", section.gpuMilliseconds}, {"cpuMs", section.cpuMilliseconds}});
                    }
                    const auto streamBegin = std::find_if(pass.sections.begin(), pass.sections.end(),
                        [](const auto& section) { return section.name == "Stream Begin"; });
                    if (streamBegin != pass.sections.end()) {
                        const auto beginIndex = uint32_t(streamBegin - pass.sections.begin());
                        uint32_t children = 0;
                        double childCpuMs = 0;
                        std::vector<double> childSums(pass.sections.size(), 0.0);
                        for (uint32_t i = 0; i < pass.sections.size(); ++i) {
                            const auto& section = pass.sections[i];
                            if (!section.cpuOnly) { continue; }
                            checkRoam(section.parent < i && std::isfinite(section.cpuMilliseconds) &&
                                section.cpuMilliseconds >= 0 && section.gpuMilliseconds == 0,
                                "Invalid CPU profile hierarchy or duration");
                            childSums[section.parent] += section.cpuMilliseconds;
                            if (section.parent == beginIndex) { ++children; childCpuMs += section.cpuMilliseconds; }
                        }
                        checkRoam(children >= 7 && childCpuMs <= streamBegin->cpuMilliseconds + .01,
                            "Stream Begin CPU breakdown missing or double-counted");
                        for (uint32_t i = 0; i < pass.sections.size(); ++i) {
                            if (pass.sections[i].cpuOnly) {
                                checkRoam(childSums[i] <= pass.sections[i].cpuMilliseconds + .01,
                                    "Nested request/reclaim CPU scopes overlap or exceed their parent");
                            }
                        }
                    }
                    nodes.push_back({{"name", pass.name}, {"gpuMs", pass.gpuMilliseconds},
                        {"cpuMs", pass.cpuMilliseconds}, {"sections", std::move(sections)}});
                }
                const std::string phase = phaseAt(f);
                gpuByPhase[phase].push_back(stats.gpuMilliseconds);
                cpuByPhase[phase].push_back(frame.executeMs);
                hostByPhase[phase].push_back(frame.hostMs);
                overlapCount += frame.overlap;
                output << Json{{"frame", f}, {"phase", phase}, {"executionId", stats.executionId},
                    {"gpuMs", stats.gpuMilliseconds}, {"cpuRecordMs", stats.cpuMilliseconds},
                    {"cpuExecuteMs", frame.executeMs}, {"hostFrameMs", frame.hostMs},
                    {"overlap", frame.overlap}, {"nodes", std::move(nodes)},
                    {"cpuPhases", frame.cpuPhases},
                    {"stream", {{"frame", s.frameIndex}, {"feedbackFrame", s.feedbackFrame},
                        {"geometryBytes", s.geometryUsedBytes}, {"geometryBudgetBytes", s.geometryBudgetBytes},
                        {"clasBytes", s.clasUsedBytes}, {"clasEncodedBytes", s.clasEncodedBytes},
                        {"clasCapacityBytes", s.clasCapacityBytes}, {"clasScratchBytes", s.clasScratchBytes},
                        {"clasPages", s.clasResidentPages}, {"clasClusters", s.clasResidentClusters},
                        {"clasPending", s.clasPendingPages}, {"clasRetiring", s.clasRetiringPages},
                        {"clasBuilt", s.clasBuiltClusters}, {"clasMoved", s.clasMovedClusters},
                        {"clasDeferred", s.clasRejectedPages}, {"residentPages", s.residentPages},
                        {"pendingPages", s.pendingPages}, {"ioQueued", s.ioQueued}, {"ioActive", s.ioActive},
                        {"uploadQueued", s.uploadQueued}, {"requests", s.requests}, {"evictions", s.evictions},
                        {"allocationFailures", s.allocationFailures}, {"uploadBytes", s.uploadBytes},
                        {"storedUploadBytes", s.storedUploadBytes}, {"totalStoredUploadBytes", s.totalStoredUploadBytes},
                        {"gpuDecompressedPages", s.gpuDecompressedPages}, {"totalGpuDecompressedPages", s.totalGpuDecompressedPages},
                        {"cpuWork", {{"demandVisited", work.demandVisited},
                            {"demandNewerThanFeedback", work.demandNewerThanFeedback},
                            {"demandUnused", work.demandUnused}, {"demandRefreshed", work.demandRefreshed},
                            {"demandIncompleteProtected", work.demandIncompleteProtected},
                            {"coldCandidates", work.coldCandidates}, {"coldVisited", work.coldVisited},
                            {"coldStateRejected", work.coldStateRejected}, {"coldClasLookups", work.coldClasLookups},
                            {"coldAgeRejected", work.coldAgeRejected}, {"coldScheduleFailed", work.coldScheduleFailed},
                            {"coldPressureScheduled", work.coldPressureScheduled},
                            {"coldRetentionScheduled", work.coldRetentionScheduled}, {"pendingFreePages", work.pendingFreePages}}},
                        {"totalUploadBytes", s.totalUploadBytes}}}}.dump() << '\n';
            }
            for (const auto& [phase, values] : gpuByPhase) {
                report["phases"][phase] = {{"gpuMs", percentiles(values)},
                    {"cpuExecuteMs", percentiles(cpuByPhase[phase])}, {"hostFrameMs", percentiles(hostByPhase[phase])}};
            }
            checkRoam(sawAsyncRaster == props.value("asyncSoftwareRaster", true), "Raster queue policy not exercised");
            checkRoam(overlapCount > frameCount / 2, "CPU recording/submission overlap not exercised");
            report["sawAsyncRaster"] = sawAsyncRaster;
            if (clas) {
                const auto& last = frames.back().stats.streaming.front();
                checkRoam(last.clasPendingPages == 0 && last.clasResidentPages == last.residentPages,
                    "Final held view CLAS backlog did not converge");
            }
            report["overlappingFrames"] = overlapCount;
            report["status"] = "passed";
            save();
            return RhiTestResult::pass(std::to_string(frameCount) + " fixed-step frames with complete timing, budget and final quality checks");
        } catch (const std::exception& error) {
            report["status"] = "failed"; report["error"] = error.what(); save();
            return RhiTestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(MiniZorahBaselineTest);
} // namespace
} // namespace metallic::tests
