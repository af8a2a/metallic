#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLights.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numbers>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

void requireAcceptance(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T>
void requireAcceptance(const Result<T>& result, const std::string& message)
{
    requireAcceptance(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

std::string hashBytes(const std::vector<std::byte>& bytes)
{
    return debug::debugSha256(std::span(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()));
}

Json hdrReceipt(Format format, uint32_t width, uint32_t height, std::span<const std::byte> bytes)
{
    requireAcceptance(format == Format::RGBA32Sfloat || format == Format::RGBA16Sfloat,
        "HDR comparison requires a linear floating-point output");
    const bool half = format == Format::RGBA16Sfloat;
    const uint32_t componentBytes = half ? 2 : 4;
    requireAcceptance(bytes.size() == uint64_t(width) * height * 4 * componentBytes,
        "HDR readback size differs from the actual output extent");
    uint64_t nonfinite = 0;
    for (size_t offset = 0; offset < bytes.size(); offset += componentBytes) {
        if (half) {
            uint16_t bits;
            std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
            nonfinite += (bits & 0x7c00u) == 0x7c00u;
        } else {
            float value;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            nonfinite += !std::isfinite(value);
        }
    }
    return {{"format", half ? "RGBA16Sfloat" : "RGBA32Sfloat"},
        {"width", width}, {"height", height},
        {"components", bytes.size() / componentBytes}, {"nonfiniteComponents", nonfinite}};
}

Json hdrReceipt(const RenderGraphPreviewRenderer& preview)
{
    return hdrReceipt(preview.readbackFormat(), preview.width(), preview.height(), preview.readbackBytes());
}

// Readbacks run only in diagnostic mode. Ordinary timings use the same production
// graph without an observer or image copies. Preview rendering waits for completion.
class VSMAcceptanceReceipt final : public IRenderDebugObserver {
public:
    bool captureImages = false;
    bool captureHDR = false;
    std::string hdrOutput, hdrPass;
    bool collectGPUWork = false;
    bool completed = false;
    uint32_t actualRenderWidth = 0, actualRenderHeight = 0;
    std::string error;
    std::map<std::string, std::unique_ptr<Buffer>> buffers;
    std::map<std::string, uint64_t> captured;
    std::map<std::string, std::array<uint64_t, 3>> imageLayouts;

    void compiled(debug::DebugValue) override {}
    void beginExecution(Device& device, debug::DebugEvidenceStamp, RenderSubsystemHost*) override
    {
        device_ = &device;
        completed = false;
        actualRenderWidth = actualRenderHeight = 0;
        error.clear();
        captured.clear();
    }
    void endExecution(bool success) override { completed = success; }
    void boundary(CommandBuffer& commands, std::string_view checkpoint, uint32_t,
        std::string_view pass, std::span<const DebugResourceBinding> resources, const debug::DebugValue&) override
    {
        if (checkpoint != "AfterPass" || (pass != "VBuffer" && pass != "Shadows" && (!captureHDR || pass != hdrPass)) || !error.empty()) { return; }
        for (const auto& binding : resources) {
            if (binding.id == "VBuffer.depth" && binding.texture) {
                actualRenderWidth = binding.texture->desc().width;
                actualRenderHeight = binding.texture->desc().height;
            }
            const bool stats = binding.id == "Shadows.statistics" || binding.id == "Shadows.lights" || binding.id == "Shadows.memory" ||
                (collectGPUWork && (binding.id == "Shadows.workload" || binding.id == "Shadows.buildDiagnostics"));
            const bool image = (captureImages && (binding.id == "VBuffer.depth" ||
                binding.id == "VBuffer.visibility" || binding.id == "Shadows.shadows")) ||
                (captureHDR && pass == hdrPass && binding.id == hdrOutput);
            if (!stats && !image) { continue; }
            try {
                copy(commands, binding, image);
            } catch (const std::exception& exception) { error = exception.what(); return; }
        }
    }
    std::vector<std::byte> read(const std::string& name)
    {
        requireAcceptance(completed && error.empty(), "Diagnostic readback failed: " + error);
        requireAcceptance(captured.contains(name), "Missing diagnostic resource: " + name);
        auto& buffer = buffers.at(name);
        buffer->invalidate();
        const void* mapped = buffer->map();
        requireAcceptance(mapped != nullptr, "Map diagnostic resource: " + name);
        std::vector<std::byte> bytes(static_cast<size_t>(captured.at(name)));
        if (const auto found = imageLayouts.find(name); found != imageLayouts.end()) {
            const auto [layerBytes, stride, layers] = found->second;
            for (uint64_t layer = 0; layer < layers; ++layer) {
                std::memcpy(bytes.data() + layer * layerBytes,
                    static_cast<const std::byte*>(mapped) + layer * stride, static_cast<size_t>(layerBytes));
            }
        } else { std::memcpy(bytes.data(), mapped, bytes.size()); }
        buffer->unmap();
        return bytes;
    }
    Json shadow()
    {
        const auto table = read("Shadows.lights");
        requireAcceptance(table.size() >= sizeof(VirtualShadowMapLightsHeader), "Truncated VSM light header");
        VirtualShadowMapLightsHeader header;
        std::memcpy(&header, table.data(), sizeof(header));
        requireAcceptance(header.version == kVirtualShadowMapLightsVersion && header.count <= header.capacity,
            "Invalid VSM light header");
        const auto raw = read("Shadows.statistics");
        requireAcceptance(raw.size() % kVirtualShadowMapFaceStatisticsSize == 0, "Invalid VSM statistics size");
        Json faces = Json::array();
        std::array<uint64_t, 32> totals{};
        for (size_t offset = 0; offset < raw.size(); offset += kVirtualShadowMapFaceStatisticsSize) {
            std::array<uint32_t, 32> values;
            std::memcpy(values.data(), raw.data() + offset, sizeof(values));
            faces.push_back(values);
            for (size_t index = 0; index < values.size(); ++index) { totals[index] += values[index]; }
        }
        Json result{{"header", {{"version", header.version}, {"count", header.count}, {"capacity", header.capacity},
            {"totalFaces", header.totalFaces}, {"admittedCount", header.admittedCount},
            {"budgetDroppedCount", header.budgetDroppedCount}, {"disabledCount", header.disabledCount},
            {"physicalPageBudget", header.physicalPageBudget}, {"sceneIdentity", header.sceneIdentity}}},
            {"statistics", faces}, {"totals", totals}};
        result["gpuWork"] = {{"version", 1}, {"enabled", collectGPUWork}};
        result["gpuBuildDiagnostics"] = {{"version", 3}, {"enabled", collectGPUWork}};
        if (collectGPUWork) {
            const auto rawWorkload = read("Shadows.workload");
            requireAcceptance(rawWorkload.size() == uint64_t(header.capacity) * 6 * kVirtualShadowMapWorkloadSize,
                "Unexpected VSM GPU workload layout");
            Json workFaces = Json::array(), workTotals = Json::object();
            std::array<uint64_t, kVirtualShadowMapWorkCounterCount> sums{};
            for (size_t offset = 0; offset < rawWorkload.size(); offset += kVirtualShadowMapWorkloadSize) {
                std::array<uint32_t, kVirtualShadowMapWorkCounterCount * 2> words;
                std::memcpy(words.data(), rawWorkload.data() + offset, sizeof(words));
                Json counters = Json::object();
                for (size_t index = 0; index < sums.size(); ++index) {
                    const uint64_t value = uint64_t(words[index * 2]) | (uint64_t(words[index * 2 + 1]) << 32);
                    counters[std::string(kVirtualShadowMapWorkCounterNames[index])] = value;
                    sums[index] += value;
                }
                const size_t layer = offset / kVirtualShadowMapWorkloadSize;
                workFaces.push_back({{"lightIndex", layer / 6}, {"face", layer % 6}, {"counters", std::move(counters)}});
            }
            for (size_t index = 0; index < sums.size(); ++index) {
                workTotals[std::string(kVirtualShadowMapWorkCounterNames[index])] = sums[index];
            }
            result["gpuWork"]["faces"] = std::move(workFaces);
            result["gpuWork"]["totals"] = std::move(workTotals);

            const auto rawBuild = read("Shadows.buildDiagnostics");
            requireAcceptance(rawBuild.size() == uint64_t(header.capacity) * 6 * kVirtualShadowMapBuildDiagnosticSize,
                "Unexpected VSM build diagnostic layout");
            Json buildFaces = Json::array(), buildTotals = Json::object();
            std::array<uint64_t, kVirtualShadowMapBuildDiagnosticCounterCount> buildSums{};
            for (size_t offset = 0; offset < rawBuild.size(); offset += kVirtualShadowMapBuildDiagnosticSize) {
                std::array<uint32_t, kVirtualShadowMapBuildDiagnosticCounterCount * 2> words;
                std::memcpy(words.data(), rawBuild.data() + offset, sizeof(words));
                Json counters = Json::object();
                for (size_t index = 0; index < buildSums.size(); ++index) {
                    const uint64_t value = uint64_t(words[index * 2]) | (uint64_t(words[index * 2 + 1]) << 32);
                    counters[std::string(kVirtualShadowMapBuildDiagnosticCounterNames[index])] = value;
                    buildSums[index] += value;
                }
                const size_t layer = offset / kVirtualShadowMapBuildDiagnosticSize;
                buildFaces.push_back({{"lightIndex", layer / 6}, {"face", layer % 6}, {"counters", std::move(counters)}});
            }
            for (size_t index = 0; index < buildSums.size(); ++index) {
                buildTotals[std::string(kVirtualShadowMapBuildDiagnosticCounterNames[index])] = buildSums[index];
            }
            result["gpuBuildDiagnostics"]["faces"] = std::move(buildFaces);
            result["gpuBuildDiagnostics"]["totals"] = std::move(buildTotals);
        }
        if (captured.contains("Shadows.memory")) {
            const auto rawMemory = read("Shadows.memory");
            requireAcceptance(rawMemory.size() == 64, "Unexpected VSM memory telemetry layout");
            std::array<uint64_t, 8> values;
            std::memcpy(values.data(), rawMemory.data(), sizeof(values));
            result["resourceMemory"] = values;
        }
        return result;
    }
    Json hdr()
    {
        const auto bytes = read(hdrOutput);
        auto result = hdrReceipt(hdrFormat_, hdrWidth_, hdrHeight_, bytes);
        result["output"] = hdrOutput;
        result["rawSha256"] = hashBytes(bytes);
        return result;
    }
private:
    void copy(CommandBuffer& commands, const DebugResourceBinding& binding, bool image)
    {
        const auto* texture = binding.texture;
        const bool hdr = image && binding.id == hdrOutput;
        uint32_t pixelBytes = binding.id == "Shadows.shadows" ? 1 : 4;
        if (hdr) {
            requireAcceptance(texture && texture->desc().layerCount == 1 &&
                (texture->desc().format == Format::RGBA16Sfloat || texture->desc().format == Format::RGBA32Sfloat),
                "Diagnostic HDR capture requires a single floating-point RGBA layer");
            hdrFormat_ = texture->desc().format; hdrWidth_ = texture->desc().width; hdrHeight_ = texture->desc().height;
            pixelBytes = hdrFormat_ == Format::RGBA16Sfloat ? 8 : 16;
        }
        uint64_t bytes = image ? uint64_t(texture->desc().width) * texture->desc().height *
            texture->desc().layerCount * pixelBytes :
            (binding.size ? binding.size : binding.buffer->desc().size - binding.offset);
        uint64_t allocationBytes = bytes;
        if (image) {
            const uint64_t layerBytes = bytes / texture->desc().layerCount;
            const uint64_t alignment = std::max(uint64_t(4), device_->capabilities().textureUploadBufferOffsetAlignment);
            const uint64_t stride = ((layerBytes + alignment - 1) / alignment) * alignment;
            imageLayouts[binding.id] = {layerBytes, stride, texture->desc().layerCount};
            allocationBytes = stride * texture->desc().layerCount;
        }
        auto& readback = buffers[binding.id];
        if (!readback || readback->desc().size != allocationBytes) {
            auto allocation = device_->createBuffer({.size = allocationBytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
            requireAcceptance(allocation, "Allocate diagnostic readback");
            readback = std::move(*allocation);
        }
        if (image) {
            const auto& desc = texture->desc();
            requireAcceptance(hdr || desc.format == Format::R8Unorm || desc.format == Format::R32Uint ||
                desc.format == Format::D32Sfloat, "Unexpected diagnostic image format");
            TextureBarrierDesc barrier{.texture = binding.texture,
                .oldLayout = textureLayoutForResourceState(binding.state), .newLayout = TextureLayout::TransferSource,
                .before = resourceSyncScope(binding.state, PipelineStageBits::AllCommands),
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
            requireAcceptance(commands.synchronize({.textures = {&barrier, 1}}), "Diagnostic image transition");
            const auto [layerBytes, stride, layers] = imageLayouts.at(binding.id);
            for (uint32_t layer = 0; layer < desc.layerCount; ++layer) {
                auto destination = readback->slice({stride * layer, layerBytes});
                requireAcceptance(destination, "Diagnostic image slice");
                requireAcceptance(commands.copyTextureToBuffer({.texture = binding.texture, .buffer = *destination,
                    .width = desc.width, .height = desc.height, .baseLayer = layer}), "Copy diagnostic image");
            }
            std::swap(barrier.before, barrier.after); std::swap(barrier.oldLayout, barrier.newLayout);
            requireAcceptance(commands.synchronize({.textures = {&barrier, 1}}), "Restore diagnostic image");
        } else {
            BufferBarrierDesc barrier{.buffer = binding.buffer,
                .before = resourceSyncScope(binding.state, PipelineStageBits::AllCommands),
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .range = {.offset = binding.offset, .size = bytes}};
            requireAcceptance(commands.synchronize({.buffers = {&barrier, 1}}), "Diagnostic buffer transition");
            auto source = binding.buffer->slice({binding.offset, bytes});
            auto destination = readback->slice();
            requireAcceptance(source, "Diagnostic source slice");
            requireAcceptance(destination, "Diagnostic destination slice");
            requireAcceptance(commands.copyBuffer(*source, *destination), "Copy diagnostic buffer");
            std::swap(barrier.before, barrier.after);
            requireAcceptance(commands.synchronize({.buffers = {&barrier, 1}}), "Restore diagnostic buffer");
        }
        const MemoryBarrierDesc host{.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .after = {PipelineStageBits::Host, AccessBits::HostRead}};
        requireAcceptance(commands.synchronize({.memory = {&host, 1}}), "Diagnostic host visibility");
        captured[binding.id] = bytes;
    }
    Device* device_ = nullptr;
    Format hdrFormat_{};
    uint32_t hdrWidth_ = 0, hdrHeight_ = 0;
};

Json memorySnapshot(const Device& device)
{
    const auto budget = device.memoryBudget();
    constexpr std::array names{"Other", "Geometry", "CLAS", "CLASScratch", "RayTracing", "MaterialTextures", "FrameResources", "Upload"};
    Json result{{"domains", Json::object()}, {"heaps", Json::array()}, {"driverBudget", budget.driverBudget},
        {"reservedBytes", budget.reservedBytes}, {"deniedAllocations", budget.deniedAllocations}};
    uint64_t liveBytes = 0, liveCount = 0;
    for (size_t index = 0; index < budget.domains.size(); ++index) {
        const auto& domain = budget.domains[index];
        liveBytes += domain.allocationBytes; liveCount += domain.allocationCount;
        result["domains"][names[index]] = {{"allocationBytes", domain.allocationBytes},
            {"allocationCount", domain.allocationCount}, {"deviceLocalBytes", domain.deviceLocalBytes},
            {"peakAllocationBytes", domain.peakAllocationBytes}};
    }
    for (const auto& heap : budget.heaps) {
        result["heaps"].push_back({{"deviceLocal", heap.deviceLocal}, {"sizeBytes", heap.sizeBytes},
            {"budgetBytes", heap.budgetBytes}, {"usageBytes", heap.usageBytes}, {"blockBytes", heap.blockBytes},
            {"allocationBytes", heap.allocationBytes}, {"allocationCount", heap.allocationCount}, {"blockCount", heap.blockCount}});
    }
    result["liveAllocationBytes"] = liveBytes; result["liveAllocationCount"] = liveCount;
    result["rayTracingAllocationBytes"] = budget.domains[size_t(MemoryBudgetDomain::RayTracing)].allocationBytes;
    result["clasAllocationBytes"] = budget.domains[size_t(MemoryBudgetDomain::CLAS)].allocationBytes;
    return result;
}

Json frameStatistics(RenderGraphPreviewRenderer& preview)
{
    auto completed = preview.collectCompletedGpuExecutionStats();
    requireAcceptance(completed, "Read GPU timestamps");
    requireAcceptance(completed->size() == 1 && completed->front().gpuTimingAvailable &&
        !completed->front().profilingOverflow, "Missing, ambiguous or overflowed GPU timestamp frame");
    const auto& stats = completed->front();
    requireAcceptance(std::isfinite(stats.gpuMilliseconds) && stats.gpuMilliseconds > 0, "Invalid graph GPU duration");
    const auto serializeSection = [](const RenderGraphProfileSection& section) {
        Json work = Json::array();
        for (const auto& counter : section.cpuWork) {
            work.push_back({{"name", counter.name}, {"value", counter.value}});
        }
        return Json{{"name", section.name}, {"parent", section.parent},
            {"cpuMs", section.cpuMilliseconds}, {"cpuOnly", section.cpuOnly},
            {"cpuWork", std::move(work)},
            {"cpuThreadProfile", section.cpuThreadProfile ? Json(*section.cpuThreadProfile) : Json(nullptr)},
            {"gpuMs", section.gpuTimingAvailable ? Json(section.gpuMilliseconds) : Json(nullptr)}};
    };
    Json result{{"executionId", stats.executionId}, {"graphMs", stats.gpuMilliseconds}, {"cpuMs", stats.cpuMilliseconds},
        {"preparation", Json::array()}, {"nodes", Json::array()}, {"streaming", Json::array()}};
    for (const auto& section : stats.preparation) {
        result["preparation"].push_back(serializeSection(section));
    }
    for (const auto& node : stats.nodes) {
        Json entry{{"name", node.name}, {"type", node.type},
            {"gpuMs", node.gpuTimingAvailable ? Json(node.gpuMilliseconds) : Json(nullptr)},
            {"cpuMs", node.cpuMilliseconds}, {"sections", Json::array()}};
        for (const auto& section : node.sections) {
            entry["sections"].push_back(serializeSection(section));
        }
        result["nodes"].push_back(std::move(entry));
    }
    for (const auto& stream : stats.streaming) {
        result["streaming"].push_back({{"passName", stream.passName}, {"assetPath", stream.assetPath},
            {"generation", stream.generation}, {"diagnosticSourceIdentity", stream.diagnosticSourceIdentity},
            {"textureSemanticIdentitySha256", stream.diagnosticTextureSemanticIdentity.empty() ? std::string{} :
                debug::debugSha256(stream.diagnosticTextureSemanticIdentity)},
            {"textureSemanticIdentityBytes", stream.diagnosticTextureSemanticIdentity.size()},
            {"originalTopologyHash", stream.originalTopologyHash}, {"primitiveMetadataHash", stream.primitiveMetadataHash},
            {"residentPages", stream.residentPages}, {"pendingPages", stream.pendingPages},
            {"geometryUsedBytes", stream.geometryUsedBytes}, {"geometryBudgetBytes", stream.geometryBudgetBytes},
            {"clasUsedBytes", stream.clasUsedBytes}, {"textureResidentBytes", stream.textureResidentBytes},
            {"loadFailures", stream.loadFailures}, {"allocationFailures", stream.allocationFailures},
            {"requestOverflows", stream.requestOverflows}});
    }
    return result;
}

Json comparisonIdentity(const Json& stats)
{
    Json identity = Json::array();
    for (const auto& stream : stats.at("streaming")) {
        identity.push_back({{"assetPath", stream.at("assetPath")}, {"generation", stream.at("generation")},
            {"source", stream.at("diagnosticSourceIdentity")}, {"textures", stream.at("textureSemanticIdentitySha256")},
            {"topology", stream.at("originalTopologyHash")}, {"primitives", stream.at("primitiveMetadataHash")}});
    }
    return identity;
}

class VirtualShadowMapAcceptanceTest final : public RHITest {
public:
    VirtualShadowMapAcceptanceTest() { name = "virtual_shadow_map_acceptance"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        const char* requested = std::getenv("METALLIC_VSM_ACCEPTANCE_CASES");
        if (!requested || !*requested) { return RHITestResult::skip("Set METALLIC_VSM_ACCEPTANCE_CASES to an explicit acceptance manifest"); }
        const auto directory = std::filesystem::absolute(context.outputDirectory);
        std::filesystem::create_directories(directory);
        Json report{{"protocol", "virtual-shadow-map-acceptance-v2"}, {"status", "running"},
            {"validation", context.enableValidation}, {"cases", Json::array()}};
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapAcceptance.json") << report.dump(2) << '\n'; };
        bench::ValidationRecorder validation;
        try {
            const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
            const auto manifestPath = std::filesystem::absolute(requested);
            std::ifstream manifestFile(manifestPath);
            const Json manifest = Json::parse(manifestFile);
            report["manifest"] = manifest;
            requireAcceptance(manifest.at("version").get<uint32_t>() == 1, "Unsupported manifest version");
            const std::string mode = manifest.at("mode");
            requireAcceptance(mode == "diagnostic" || mode == "timing", "Mode must be diagnostic or timing");
            const bool diagnostic = mode == "diagnostic";
            requireAcceptance(diagnostic || !context.enableValidation, "Ordinary timings must disable validation");
            report["mode"] = mode; report["performanceEligible"] = !diagnostic && !context.enableValidation;
            report["timingScope"] = "Completed production graph GPU timestamps; preview serializes each submitted frame";
            report["memoryScope"] = "Device live allocations after submission completion; driver heap usage includes other processes";
            report["extentMeasurement"] = "Diagnostic reads each VBuffer depth descriptor; timing reads every frame's CPU execution snapshot. Snapshots add CPU collection work without changing GPU commands, scheduling or resource lifetimes; no post-warmup GPU samples are excluded.";
            report["statisticalUnit"] = "Independent process/run; frames describe this run and are not independent A/B samples";
            requireAcceptance(manifest.at("cases").is_array() && manifest.at("cases").size() == 1,
                "Acceptance requires one scene/device per process; run each named case manifest separately");
            std::map<std::string, bool> caseIds;
            for (const auto& spec : manifest.at("cases")) {
                const std::string id = spec.at("id");
                requireAcceptance(caseIds.emplace(id, true).second, "Duplicate acceptance case id: " + id);
                requireAcceptance(!id.empty() && std::filesystem::path(id).filename() == std::filesystem::path(id) &&
                    id != "." && id != "..", "Case id must be one filename component");
                const uint32_t frames = spec.value("frames", manifest.value("frames", 512u));
                const uint32_t warmup = spec.value("warmupFrames", manifest.value("warmupFrames", 16u));
                requireAcceptance(frames > warmup && frames <= 16384, "Invalid frame budget");
                const auto comparisons = spec.value("comparisonFrames", std::vector<uint32_t>{});
                const auto captures = spec.value("captureFrames", std::vector<uint32_t>{});
                const bool enableStreamline = spec.value("enableStreamline", false);
                const bool requireNoRayCapabilities = spec.value("requireNoRayCapabilities", !enableStreamline);
                const bool synchronizationValidation = spec.value("synchronizationValidation",
                    manifest.value("synchronizationValidation", context.enableValidation));
                requireAcceptance(!synchronizationValidation || context.enableValidation,
                    "Synchronization validation requires validation to be enabled");
                requireAcceptance(diagnostic || (comparisons.empty() && captures.empty()), "Timing mode forbids diagnostic captures");
                for (const auto frame : comparisons) { requireAcceptance(frame >= warmup && frame < frames, "Comparison frame outside post-warmup range"); }
                for (const auto frame : captures) { requireAcceptance(frame < frames, "Capture frame outside run"); }
                report["cases"].push_back({{"id", id}, {"status", "running"}, {"spec", spec},
                    {"frames", Json::array()}, {"comparisons", Json::array()}, {"captures", Json::array()}});
                auto& output = report["cases"].back();
                output["synchronizationValidation"] = synchronizationValidation;
                output["requireNoRayCapabilities"] = requireNoRayCapabilities;
                output["enableStreamline"] = enableStreamline;
                output["rayCapabilityContract"] = requireNoRayCapabilities ? "RayQuery and RTAS capabilities must be disabled" :
                    "Capabilities are reported; Streamline can require RayQuery/RTAS. RHI RayTracing and CLAS allocations must remain zero.";
                save();
                RenderGraph graph;
                std::string log;
                requireAcceptance(loadRenderGraphFromFile(manifestPath.parent_path() / spec.at("graph").get<std::string>(), graph, log), log);
                auto* shadows = graph.findNode("Shadows");
                requireAcceptance(shadows && shadows->type == "VirtualShadowMapLightsPass", "Acceptance requires a production VSM Lights pass named Shadows");
                const bool cpuProfileDiagnostics = shadows->properties.value("benchmarkCPUProfile", false);
                const bool gpuProfileDiagnostics = shadows->properties.value("benchmarkGPUProfile", false);
                const bool buildSubstageDiagnostics = shadows->properties.value("benchmarkBuildSubstages", false);
                const bool gpuWorkDiagnostics = shadows->properties.value("benchmarkGPUWork", false);
                requireAcceptance(!buildSubstageDiagnostics || gpuProfileDiagnostics,
                    "Build substage attribution requires detailed GPU profiling");
                requireAcceptance(!cpuProfileDiagnostics || diagnostic, "CPU profile attribution requires diagnostic mode");
                requireAcceptance(!gpuWorkDiagnostics || diagnostic, "Timing mode forbids GPU workload counters");
                requireAcceptance(!cpuProfileDiagnostics || !gpuProfileDiagnostics,
                    "CPU attribution and detailed GPU profiling must run separately");
                output["cpuProfileDiagnostics"] = cpuProfileDiagnostics;
                report["cpuProfileDiagnostics"] = cpuProfileDiagnostics;
                output["gpuProfileDiagnostics"] = gpuProfileDiagnostics;
                report["gpuProfileDiagnostics"] = gpuProfileDiagnostics;
                output["buildSubstageDiagnostics"] = buildSubstageDiagnostics;
                report["buildSubstageDiagnostics"] = buildSubstageDiagnostics;
                if (buildSubstageDiagnostics) {
                    report["buildSubstageScope"] = "Optional nested timestamps attribute build cost and can perturb scheduling; they must not enter ordinary optimization A/B acceptance. Finalize includes its trailing read/draw barriers.";
                }
                output["gpuWorkDiagnostics"] = gpuWorkDiagnostics;
                report["gpuWorkDiagnostics"] = gpuWorkDiagnostics;
                output["gpuBuildDiagnostics"] = gpuWorkDiagnostics;
                report["gpuBuildDiagnostics"] = gpuWorkDiagnostics;
                if (gpuWorkDiagnostics) {
                    report["performanceEligible"] = false;
                    report["gpuWorkScope"] = "Optional GPU diagnostic counters include shader atomic overhead; timings from this run are not ordinary performance evidence.";
                    report["gpuBuildDiagnosticScope"] = "Build precheck counters and scan-length histograms are diagnostic-only sidecars; they do not change the public statistics or workload layouts and are not ordinary timing evidence.";
                }
                if (cpuProfileDiagnostics) {
                    report["performanceEligible"] = false;
                    report["cpuProfileScope"] = "CPU attribution includes diagnostic clocks and counters; nested sections are inclusive and must not be summed with their children. This is not ordinary performance acceptance.";
                }
                const uint32_t shadowId = shadows->id;
                auto* raster = graph.findNode("VBuffer");
                requireAcceptance(raster && raster->type == "VisibilityBufferPass", "Acceptance requires production VBuffer");
                const uint32_t rasterId = raster->id;
                const bool worldBinding = raster->properties.value("sceneBinding", std::string("world")) == "world";
                if (diagnostic) { graph.setNodeRuntimeProperty(rasterId, "benchmarkCaptureSemanticSource", true); }
                for (const auto& event : spec.value("events", Json::array())) {
                    requireAcceptance(!event.value("reloadScene", false) || (worldBinding && !spec.at("scene").is_null()),
                        "reloadScene requires world scene binding and an explicit scene; asset graph reload is not scene handoff");
                }
                requireAcceptance(!shadows->properties.value("forceRedraw", false), "Acceptance begins with ordinary caching");
                const std::string displayOutput = spec.value("output", std::string("FinalBlit.color"));
                const std::string comparisonOutput = spec.value("comparisonOutput", std::string("Deferred.color"));
                graph.markOutput(displayOutput);
                if (!comparisons.empty() || (diagnostic && !captures.empty())) { graph.markOutput(comparisonOutput); }
                output["resolvedGraph"] = Json::parse(serializeRenderGraphToString(graph));
                std::vector<std::unique_ptr<scene::SceneDocument>> documents;
                RenderView view;
                RenderGraphPreviewRenderer preview;
                VSMAcceptanceReceipt receipt;
                receipt.collectGPUWork = gpuWorkDiagnostics;
                receipt.hdrOutput = comparisonOutput;
                receipt.hdrPass = comparisonOutput.substr(0, comparisonOutput.rfind('.'));
                struct ReceiptLifetimeScope
                {
                    RenderGraphPreviewRenderer& preview;
                    ~ReceiptLifetimeScope()
                    {
                        preview.setDebugObserver(nullptr);
                        // A failed render can leave an accepted submission prefix.
                        // Receipt buffers must retire while the preview device lives.
                        if (auto* device = preview.subsystemHost()->device()) { (void)device->waitIdle(); }
                    }
                } receiptLifetime{preview};
                requireAcceptance(preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                    {.initialDecodeTimeoutMilliseconds = 10000}, log), log);
                uint32_t width = spec.at("width"), height = spec.at("height");
                uint32_t renderWidth = spec.value("renderWidth", width), renderHeight = spec.value("renderHeight", height);
                requireAcceptance(view.setRenderResolution(renderWidth, renderHeight), "Invalid render dimensions");
                Json camera = spec.contains("camera") ? spec.at("camera") : graph.viewProperties().at("camera");
                requireAcceptance(view.setCameraProperties(camera), "Invalid camera");
                const bool temporalJitter = spec.value("temporalJitter", graph.viewProperties().value("temporalJitter", false));
                requireAcceptance(comparisons.empty() || !temporalJitter,
                    "Exact cache comparisons require an explicitly unjittered diagnostic workload");
                view.setTemporalJitter(temporalJitter);
                output["temporalJitter"] = temporalJitter;
                preview.bindRenderView(&view);
                const auto loadScene = [&]() {
                    if (spec.at("scene").is_null()) { preview.bindRuntimeScene(nullptr); return; }
                    auto document = std::make_unique<scene::SceneDocument>();
                    const auto path = root / spec.at("scene").get<std::string>();
                    requireAcceptance(spec.value("metadataOnly", false) ? document->loadStreamMetadata(path) : document->load(path),
                        "Acceptance scene load: " + path.string());
                    requireAcceptance(document->documentWarning().empty(), document->documentWarning());
                    preview.bindRuntimeScene(document.get());
                    preview.setEnvironment(document->environment());
                    auto lighting = document->lighting(); lighting.autoExposure.enabled = false;
                    lighting.exposureEV100 = spec.value("exposureEV100", 0.0f);
                    requireAcceptance(preview.setLighting(lighting), "Invalid scene lighting");
                    documents.push_back(std::move(document));
                };
                const auto setLighting = [&]() {
                    if (spec.contains("environment")) {
                        const auto& env = spec.at("environment");
                        preview.setEnvironment({.enabled = env.value("enabled", true),
                            .path = root / env.value("path", std::string{}), .intensity = env.value("intensity", 1.0f),
                            .rotationDegrees = env.value("rotationDegrees", 0.0f), .visible = env.value("visible", false)});
                    }
                    if (spec.contains("lighting")) {
                        const auto& sun = spec.at("lighting").at("sun");
                        const auto direction = sun.at("direction").get<std::array<float, 3>>();
                        const auto color = sun.value("color", std::array<float, 3>{1, 1, 1});
                        environment::WorldEnvironment world;
                        world.sun.enabled = true; world.sun.direction = {direction[0], direction[1], direction[2]};
                        world.sun.color = {color[0], color[1], color[2]}; world.sun.illuminance = sun.at("illuminance");
                        world.sun.angularRadius = sun.value("angularRadius", 0.00465f);
                        requireAcceptance(preview.setWorldEnvironment(world), "Invalid acceptance sun");
                    }
                };
                loadScene(); setLighting();
                requireAcceptance(preview.initialize(context.enableValidation, false, false, synchronizationValidation,
                    validation.sink(), enableStreamline), "Initialize acceptance preview");
                if (diagnostic) { preview.setDebugObserver(&receipt); }
                preview.setRawReadbackEnabled(true);
                Json previousSceneIdentity;
                uint32_t actualRenderWidth = 0, actualRenderHeight = 0;
                bool noRayCapabilities = true, noRayTracingAllocations = true;
                for (uint32_t frame = 0; frame < frames; ++frame) {
                    bool sceneHandoff = false;
                    for (const auto& event : spec.value("events", Json::array())) {
                        if (event.at("frame").get<uint32_t>() != frame) { continue; }
                        if (event.contains("width") || event.contains("height")) {
                            width = event.value("width", width); height = event.value("height", height);
                            renderWidth = event.value("renderWidth", width); renderHeight = event.value("renderHeight", height);
                            requireAcceptance(view.setRenderResolution(renderWidth, renderHeight), "Invalid resize dimensions");
                        }
                        if (event.contains("camera")) { camera = event.at("camera"); }
                        if (event.value("cameraCut", false)) { view.cameraCut(); }
                        if (event.value("reloadScene", false)) { loadScene(); setLighting(); graph.markDirty(); sceneHandoff = true; }
                    }
                    Json currentCamera = camera;
                    if (spec.contains("cameraMotion")) {
                        const auto& motion = spec.at("cameraMotion");
                        const uint32_t axis = motion.value("axis", 0u);
                        const double period = motion.value("period", 128.0);
                        requireAcceptance(axis < 3 && period > 0, "Invalid camera motion");
                        const double delta = motion.at("amplitude").get<double>() * std::sin(2 * std::numbers::pi * frame / period);
                        currentCamera["eye"][axis] = camera.at("eye").at(axis).get<double>() + delta;
                        currentCamera["center"][axis] = camera.at("center").at(axis).get<double>() + delta;
                    }
                    requireAcceptance(view.setCameraProperties(currentCamera), "Invalid frame camera");
                    if (spec.contains("freezeStreamingFrame") && frame == spec.at("freezeStreamingFrame").get<uint32_t>()) {
                        graph.setNodeRuntimeProperty(rasterId, "benchmarkFreezeStreaming", true);
                    }
                    validation.phase(id + "/" + std::to_string(frame));
                    const bool capture = std::find(captures.begin(), captures.end(), frame) != captures.end();
                    receipt.captureImages = diagnostic && capture;
                    receipt.captureHDR = diagnostic && capture;
                    const bool extentSnapshot = !diagnostic;
                    preview.setExecutionCaptureEnabled(extentSnapshot);
                    const auto begin = std::chrono::steady_clock::now();
                    requireAcceptance(preview.render(graph, width, height, displayOutput, capture), preview.lastLog());
                    auto row = frameStatistics(preview);
                    row["frame"] = frame; row["warmup"] = frame < warmup; row["width"] = width; row["height"] = height;
                    row["requestedRenderWidth"] = renderWidth; row["requestedRenderHeight"] = renderHeight;
                    row["extentSnapshot"] = extentSnapshot;
                    row["timingEligible"] = frame >= warmup;
                    if (diagnostic) {
                        actualRenderWidth = receipt.actualRenderWidth;
                        actualRenderHeight = receipt.actualRenderHeight;
                    } else if (extentSnapshot) {
                        const auto snapshot = preview.executionSnapshot();
                        requireAcceptance(snapshot && snapshot->success, "Missing successful execution extent snapshot");
                        const auto depth = std::find_if(snapshot->resources.begin(), snapshot->resources.end(), [](const auto& resource) {
                            return resource.name == "VBuffer.depth" ||
                                std::find(resource.aliases.begin(), resource.aliases.end(), "VBuffer.depth") != resource.aliases.end();
                        });
                        requireAcceptance(depth != snapshot->resources.end() && depth->type == RenderGraphResourceType::Texture2D,
                            "Execution snapshot has no VBuffer depth texture");
                        actualRenderWidth = depth->textureDesc.width;
                        actualRenderHeight = depth->textureDesc.height;
                    }
                    requireAcceptance(actualRenderWidth != 0 && actualRenderHeight != 0, "Missing actual raster extent");
                    row["actualRenderWidth"] = actualRenderWidth; row["actualRenderHeight"] = actualRenderHeight;
                    row["extentObservedThisFrame"] = diagnostic || extentSnapshot;
                    row["wallMs"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                    const auto* device = preview.subsystemHost()->device();
                    const auto& caps = device->capabilities();
                    row["rtCaps"] = {{"rayQuery", caps.rayQuery},
                        {"rayTracingAccelerationStructure", caps.rayTracingAccelerationStructure},
                        {"rayTracingPipeline", caps.rayTracingPipeline},
                        {"clusterAccelerationStructure", caps.clusterAccelerationStructure}};
                    noRayCapabilities &= !caps.rayQuery && !caps.rayTracingAccelerationStructure;
                    row["memory"] = memorySnapshot(*device);
                    if (auto* streamer = preview.subsystemHost()->get<StreamerSubsystem>()) {
                        const auto ready = streamer->sceneReadiness();
                        row["readiness"] = {{"ready", ready.ready}, {"requiredPages", ready.requiredPages}, {"completedPages", ready.completedPages}};
                        requireAcceptance(frame < warmup || ready.ready, "Required scene pages were not ready after warmup");
                    }
                    for (const auto& stream : row["streaming"]) {
                        requireAcceptance(stream["loadFailures"] == 0 && stream["allocationFailures"] == 0 &&
                            stream["requestOverflows"] == 0, "Streaming failed or overflowed during acceptance");
                    }
                    if (diagnostic) {
                        row["shadow"] = receipt.shadow();
                        const auto identity = row["shadow"]["header"]["sceneIdentity"];
                        row["sceneHandoff"] = sceneHandoff;
                        requireAcceptance(!sceneHandoff || (!previousSceneIdentity.is_null() && identity != previousSceneIdentity),
                            "Scene handoff did not change the published VSM scene identity");
                        previousSceneIdentity = identity;
                    }
                    output["frames"].push_back(row);
                    noRayTracingAllocations &= row["memory"]["domains"]["RayTracing"]["peakAllocationBytes"] == 0 &&
                        row["memory"]["domains"]["CLAS"]["peakAllocationBytes"] == 0;
                    output["noRayQueryOrRTAS"] = noRayCapabilities;
                    output["noRayTracingAllocations"] = noRayTracingAllocations;
                    requireAcceptance(!requireNoRayCapabilities || noRayCapabilities,
                        "Strict acceptance device unexpectedly enabled RayQuery/RTAS capabilities");
                    requireAcceptance(noRayTracingAllocations, "VSM graph allocated RHI ray tracing or CLAS memory");
                    if (diagnostic) {
                        requireAcceptance(!spec.value("requireShadow", true) || row["shadow"]["header"]["admittedCount"].get<uint32_t>() > 0,
                            "No source admitted for shadows");
                        requireAcceptance(frame < warmup || !spec.value("requireShadow", true) ||
                            row["shadow"]["totals"][6].get<uint64_t>() > 0, "No visible receiver after warmup");
                        requireAcceptance(row["shadow"]["header"]["budgetDroppedCount"] == 0, "VSM light budget overflow");
                        requireAcceptance(row["shadow"]["totals"][5] == 0 && row["shadow"]["totals"][9] == 0,
                            "VSM work overflow or invalid coarse fallback");
                    }
                    if (capture) {
                        const auto filename = id + "-" + std::to_string(frame) + ".png";
                        requireAcceptance(preview.width() == width && preview.height() == height &&
                            preview.pixels().size() == uint64_t(width) * height,
                            "Presentation capture dimensions differ from the requested output");
                        requireAcceptance(saveRgba8Png(directory / filename, reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                            width, height, log), log);
                        output["captures"].push_back({{"frame", frame}, {"file", filename}, {"rawSha256", hashBytes(preview.readbackBytes())}});
                        if (diagnostic) {
                            auto& captureReceipt = output["captures"].back();
                            captureReceipt["linearHDR"] = receipt.hdr();
                            captureReceipt["depthSha256"] = hashBytes(receipt.read("VBuffer.depth"));
                            captureReceipt["shadowSha256"] = hashBytes(receipt.read("Shadows.shadows"));
                            requireAcceptance(captureReceipt["linearHDR"]["nonfiniteComponents"] == 0, "Nonfinite diagnostic HDR capture");
                        }
                    }
                    if (std::find(comparisons.begin(), comparisons.end(), frame) != comparisons.end()) {
                        receipt.captureImages = true;
                        requireAcceptance(preview.render(graph, width, height, comparisonOutput, true), preview.lastLog());
                        const auto cachedHDR = preview.readbackBytes();
                        const auto cachedHDRReceipt = hdrReceipt(preview);
                        const auto cachedStats = frameStatistics(preview);
                        const auto cachedDepth = receipt.read("VBuffer.depth"), cachedVisibility = receipt.read("VBuffer.visibility");
                        const auto cachedShadows = receipt.read("Shadows.shadows");
                        graph.setNodeRuntimeProperty(shadowId, "forceRedraw", true);
                        requireAcceptance(preview.render(graph, width, height, comparisonOutput, true), preview.lastLog());
                        const auto redrawnHDRReceipt = hdrReceipt(preview);
                        const auto redrawnStats = frameStatistics(preview);
                        const auto redrawnDepth = receipt.read("VBuffer.depth"), redrawnVisibility = receipt.read("VBuffer.visibility");
                        const auto redrawnShadows = receipt.read("Shadows.shadows");
                        const bool stable = cachedDepth == redrawnDepth && cachedVisibility == redrawnVisibility &&
                            comparisonIdentity(cachedStats) == comparisonIdentity(redrawnStats);
                        uint64_t changed = 0;
                        if (cachedShadows.size() == redrawnShadows.size()) {
                            for (size_t index = 0; index < cachedShadows.size(); ++index) { changed += cachedShadows[index] != redrawnShadows[index]; }
                        } else { changed = std::max(cachedShadows.size(), redrawnShadows.size()); }
                        output["comparisons"].push_back({{"frame", frame}, {"geometryIdentityStable", stable},
                            {"status", stable ? (changed ? "failed" : "passed") : "incomparable"},
                            {"changedShadowBytes", changed}, {"byteEqual", changed == 0},
                            {"hdrByteEqual", cachedHDR == preview.readbackBytes()},
                            {"cachedHDR", cachedHDRReceipt}, {"redrawnHDR", redrawnHDRReceipt},
                            {"cachedShadowSha256", hashBytes(cachedShadows)}, {"redrawnShadowSha256", hashBytes(redrawnShadows)},
                            {"cachedDepthSha256", hashBytes(cachedDepth)}, {"redrawnDepthSha256", hashBytes(redrawnDepth)},
                            {"cachedStreaming", cachedStats["streaming"]}, {"redrawnStreaming", redrawnStats["streaming"]}});
                        graph.setNodeRuntimeProperty(shadowId, "forceRedraw", false);
                        receipt.captureImages = false;
                        requireAcceptance(cachedHDRReceipt["nonfiniteComponents"] == 0 &&
                            redrawnHDRReceipt["nonfiniteComponents"] == 0, "Nonfinite HDR comparison output");
                        requireAcceptance(!stable || changed == 0, "Cached/forced shadow mismatch with stable geometry inputs");
                    }
                    if (frame % 16 == 0 || frame + 1 == frames) {
                        std::printf("[VSM acceptance] %s %u/%u GPU=%.3fms live=%.1fMiB\n", id.c_str(), frame + 1, frames,
                            row["graphMs"].get<double>(), row["memory"]["liveAllocationBytes"].get<uint64_t>() / 1048576.0);
                        std::fflush(stdout); save();
                    }
                }
                output["noRayQueryOrRTAS"] = noRayCapabilities;
                output["noRayTracingAllocations"] = noRayTracingAllocations;
                output["status"] = "passed";
                if (!comparisons.empty()) {
                    const bool verified = std::any_of(output["comparisons"].begin(), output["comparisons"].end(),
                        [](const Json& value) { return value.at("status") == "passed"; });
                    requireAcceptance(verified, "All cache comparisons were incomparable; no cache equivalence evidence");
                }
                requireAcceptance(!validation.failed(), "Vulkan validation failed");
                save();
            }
            requireAcceptance(!validation.failed(), "Vulkan validation failed during preview cleanup");
            report["status"] = "passed";
            report["validationMessages"] = validation.snapshot();
            save();
            return RHITestResult::pass("Production VSM acceptance completed without RHI ray tracing allocations; see per-case capability contract");
        } catch (const std::exception& exception) {
            report["status"] = "failed"; report["error"] = exception.what();
            if (!report["cases"].empty()) { report["cases"].back()["status"] = "failed"; }
            report["validationMessages"] = validation.snapshot(); save();
            return RHITestResult::fail(exception.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapAcceptanceTest);
} // namespace
} // namespace metallic::tests
