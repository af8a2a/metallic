#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Debug/LumenProbeTraceReadback.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Task/TaskSystem.h"

#include <SDL3/SDL.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>

namespace {
using namespace metallic;
using namespace metallic::render;
using debug::DebugValue;

constexpr uint32_t kExtent = 384;
constexpr uint32_t kLitFrames = 256;
constexpr uint32_t kDarkFrames = 8;
constexpr uint32_t kContactProbePhases = 8;
constexpr std::string_view kOutput = "FinalBlit.color";
constexpr uint32_t kTileSize = 16;
constexpr uint32_t kProbesPerTile = 9;
constexpr uint32_t kRaysPerProbe = 64;
constexpr uint32_t kProbeCount = (kExtent / kTileSize) * (kExtent / kTileSize) * kProbesPerTile;
constexpr uint32_t kPixelCount = kExtent * kExtent;
// Full depth followed by all eight packed mips from 256x256 to 2x2.
constexpr uint32_t kHZBElements = kExtent * kExtent + (4u * 256u * 256u - 4u) / 3u;

struct RayCaptureRegion {
    std::array<uint32_t, 4> tiles;
    bool captureHZB;

    constexpr std::array<uint32_t, 4> pixels() const
    {
        return {tiles[0] * kTileSize, tiles[1] * kTileSize, tiles[2] * kTileSize, tiles[3] * kTileSize};
    }
    constexpr uint32_t traces() const { return tiles[2] * tiles[3] * kProbesPerTile * kRaysPerProbe; }
};

constexpr RayCaptureRegion kDetailRayRegion{{16, 18, 4, 4}, false};
constexpr RayCaptureRegion kFloorRayRegion{{4, 18, 15, 5}, true};

struct BaselineOptions {
    bool trace = false;
    bool detail = false;
    bool screenGDFContinuation = false;
    bool floorRays = false;
    bool gdfPrefixValidation = false;
    bool gdfPrefixRejectReceding = false;
    bool gdfPrefixMeshValidation = false;
    bool contactVisibility = false;
    bool contactTemporalPhases = false;
};

struct BaselineCapture final : IRenderDebugObserver {
    // Contact visibility reads the full Deferred guides and both ray buffers in one frame.
    explicit BaselineCapture(bool contactVisibility)
        : inspector(contactVisibility ? debug::DebugLimits{.frameBytes = 40ull << 20} : debug::DebugLimits{},
              RenderDebugDetail::Resources)
    {}

    RenderGraphPreviewRenderer preview;
    RenderDebugRuntime inspector;
    DebugValue request;
    DebugValue reply;

    ~BaselineCapture() override { preview.setDebugObserver(nullptr); }
    bool detailedSnapshots() const override { return false; }
    void compiled(DebugValue graph) override { inspector.compiled(std::move(graph)); }
    void beginExecution(Device& device, debug::DebugEvidenceStamp stamp, RenderSubsystemHost* subsystems) override
    {
        if (!request.is_null()) {
            // A material edit can rebuild the graph before the first dark frame.
            reply = inspector.core().dispatch(request);
            request = nullptr;
        }
        inspector.beginExecution(device, std::move(stamp), subsystems);
    }
    void boundary(CommandBuffer& commands, std::string_view checkpoint, uint32_t passId, std::string_view pass,
        std::span<const DebugResourceBinding> resources, const DebugValue& values) override
    {
        inspector.boundary(commands, checkpoint, passId, pass, resources, values);
    }
    void endExecution(bool success) override { inspector.endExecution(success); }
};

void renderFrame(RenderGraphPreviewRenderer& preview, RenderGraph& graph)
{
    if (!preview.render(graph, kExtent, kExtent, kOutput, false)) {
        throw std::runtime_error("Cornell Box render failed: " + preview.lastLog());
    }
}

std::shared_ptr<const debug::DebugCapture> getCapture(BaselineCapture& capture, size_t jobIndex)
{
    auto& core = capture.inspector.core();
    const auto job = capture.reply.at("result").at("jobs").at(jobIndex).at("job").get<std::string>();
    if (auto result = core.completedCapture(job)) { return result; }
    throw std::runtime_error("Cornell Box capture did not complete: " +
        core.dispatch({{"method", "jobs.get"}, {"params", {{"job", job}}}}).dump());
}

void checkSameFrame(const debug::DebugEvidenceStamp& expected, const debug::DebugEvidenceStamp& actual)
{
    if (expected.execution != actual.execution || expected.generation != actual.generation) {
        throw std::runtime_error("Cornell Box captures came from different graph executions");
    }
}

void saveReadback(const debug::DebugArtifact& artifact, const std::filesystem::path& file,
    std::string_view layout, size_t bytes)
{
    if (artifact.layout.name != layout || artifact.bytes.size() != bytes) {
        throw std::runtime_error("Unexpected Cornell Box readback layout or size: " + artifact.metadata.dump());
    }
    std::ofstream stream(file, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(artifact.bytes.data()), std::streamsize(artifact.bytes.size()));
    if (!stream) { throw std::runtime_error("Could not write " + file.string()); }
}

void addRayCaptureBatches(DebugValue& batches, const RayCaptureRegion& region)
{
    const auto pixels = region.pixels();
    const DebugValue roi{{"x", pixels[0]}, {"y", pixels[1]},
        {"width", pixels[2]}, {"height", pixels[3]}};
    batches.push_back({{"pass", "Deferred"}, {"checkpoint", "AfterPass"}, {"resources", DebugValue::array({
        {{"id", "Deferred.lumenPosition"}, {"roi", roi}},
        {{"id", "Deferred.lumenNormal"}, {"roi", roi}},
    })}});
    batches.push_back({{"pass", "Gather"}, {"checkpoint", "AfterPass"}, {"resources", DebugValue::array({
        {{"id", "Gather.probePositions"}, {"layout", "u32"}, {"count", kProbeCount * 2}},
        {{"id", "Gather.probeTraceReadback"}, {"layout", kLumenProbeTraceReadbackLayout}, {"count", region.traces()}},
        {{"id", "Gather.probeRadiance"}, {"layout", "u32"}, {"count", kProbeCount * kRaysPerProbe * 4}},
        {{"id", "Gather.probeDistance"}, {"layout", "u32"}, {"count", kProbeCount * kRaysPerProbe}},
    })}});
    if (region.captureHZB) {
        batches.push_back({{"pass", "ScreenTrace"}, {"checkpoint", "AfterPass"}, {"resources", DebugValue::array({
            {{"id", "ScreenTrace.closestHZB"}, {"layout", "u32"}, {"count", kHZBElements}},
        })}});
    }
}

DebugValue saveRayCapture(BaselineCapture& capture, const std::filesystem::path& directory,
    std::string_view label, const debug::DebugEvidenceStamp& frame, const RayCaptureRegion& region)
{
    const auto deferred = getCapture(capture, 2);
    const auto gather = getCapture(capture, 3);
    checkSameFrame(frame, deferred->snapshot.evidence);
    checkSameFrame(frame, gather->snapshot.evidence);
    const auto pixels = region.pixels();
    const size_t pixelCount = pixels[2] * pixels[3];
    const auto& geometry = deferred->artifacts;
    const auto& traces = gather->artifacts;
    DebugValue result{{"frame", frame.execution}, {"tiles", region.tiles}, {"pixel_rect", pixels},
        {"evidence", {deferred->snapshot.evidence.value(), gather->snapshot.evidence.value()}}};
    const auto prefix = label == "on" ? "rays-" : std::string(label) + "-rays-";
    auto saveRay = [&](std::string_view key, std::string_view suffix, const debug::DebugArtifact& artifact,
                       std::string_view layout, size_t bytes) {
        const auto file = prefix + std::string(suffix) + ".bin";
        saveReadback(artifact, directory / file, layout, bytes);
        result[key] = file;
    };
    saveRay("position", "position", geometry.at(0), "RGBA32F", pixelCount * 16);
    saveRay("normal", "normal", geometry.at(1), "RGBA16F", pixelCount * 8);
    saveRay("probe_positions", "probe-positions", traces.at(0), "u32", kProbeCount * 8);
    saveRay("trace", "trace", traces.at(1), kLumenProbeTraceReadbackLayout,
        region.traces() * sizeof(LumenProbeTraceReadbackRecord));
    saveRay("radiance", "radiance", traces.at(2), "u32", kProbeCount * kRaysPerProbe * 16);
    saveRay("distance", "distance", traces.at(3), "u32", kProbeCount * kRaysPerProbe * 4);
    if (region.captureHZB) {
        const auto screen = getCapture(capture, 4);
        checkSameFrame(frame, screen->snapshot.evidence);
        saveRay("closest_hzb", "closest-hzb", screen->artifacts.at(0), "u32", kHZBElements * 4);
        result["evidence"].push_back(screen->snapshot.evidence.value());
    }
    return result;
}

struct ContactResource {
    std::string_view key;
    std::string_view id;
    std::string_view layout;
    std::string_view format;
    uint32_t bytes;
};

constexpr std::array kContactDeferred{
    ContactResource{"position", "Deferred.lumenPosition", "RGBA32F", "rgba32f", kPixelCount * 16},
    ContactResource{"normal", "Deferred.lumenNormal", "RGBA16F", "rgba16f", kPixelCount * 8},
    ContactResource{"diffuse", "Deferred.lumenDiffuse", "RGBA16F", "rgba16f", kPixelCount * 8},
    ContactResource{"direct_color", "Deferred.color", "RGBA32F", "rgba32f", kPixelCount * 16},
};
constexpr std::array kContactGather{
    ContactResource{"interpolation", "Gather.probeInterpolationReadback", "u32", "u32x4_f32x4", kPixelCount * 32},
    ContactResource{"filtered_radiance", "Gather.filteredProbeRadiance", "u32", "rgba32f", kProbeCount * kRaysPerProbe * 16},
    ContactResource{"gather_indirect", "Gather.indirectDiffuse", "RGBA16F", "rgba16f", kPixelCount * 8},
    ContactResource{"gather_color", "Gather.color", "RGBA32F", "rgba32f", kPixelCount * 16},
};
constexpr std::array kContactTemporal{
    ContactResource{"history_weight", "Temporal.historyWeight", "f32", "f32", kPixelCount * 4},
};
constexpr std::array kContactRaw{
    ContactResource{"raw_radiance", "Gather.probeRadiance", "u32", "rgba32f", kProbeCount * kRaysPerProbe * 16},
};

void addContactBatch(DebugValue& batches, std::string_view pass, std::span<const ContactResource> resources,
    const DebugValue& roi)
{
    auto requests = DebugValue::array();
    for (const auto& resource : resources) {
        if (resource.layout == "u32") {
            requests.push_back({{"id", resource.id}, {"layout", resource.layout}, {"count", resource.bytes / sizeof(uint32_t)}});
        } else {
            requests.push_back({{"id", resource.id}, {"roi", roi}});
        }
    }
    batches.push_back({{"pass", pass}, {"checkpoint", "AfterPass"}, {"resources", std::move(requests)}});
}

DebugValue saveContactVisibilityCapture(BaselineCapture& capture, const std::filesystem::path& directory,
    std::string_view label, const debug::DebugEvidenceStamp& frame, size_t jobIndex, const DebugValue* rayCapture)
{
    DebugValue result{{"frame", frame.execution}, {"evidence", DebugValue::array()}};
    auto saveBatch = [&](std::span<const ContactResource> resources) {
        const auto job = getCapture(capture, jobIndex++);
        checkSameFrame(frame, job->snapshot.evidence);
        result["evidence"].push_back(job->snapshot.evidence.value());
        for (size_t i = 0; i < resources.size(); ++i) {
            const auto& resource = resources[i];
            const auto file = std::string(label) + "-contact-" + std::string(resource.key) + ".bin";
            saveReadback(job->artifacts.at(i), directory / file, resource.layout, resource.bytes);
            result[resource.key] = {{"file", file}, {"format", resource.format}, {"frame", frame.execution}};
        }
    };
    saveBatch(kContactDeferred);
    saveBatch(kContactGather);
    saveBatch(kContactTemporal);
    if (rayCapture) {
        result["raw_radiance"] = {{"file", rayCapture->at("radiance")},
            {"format", kContactRaw[0].format}, {"frame", frame.execution}};
    } else {
        saveBatch(kContactRaw);
    }
    return result;
}

DebugValue captureFrame(BaselineCapture& capture, RenderGraph& graph,
    const std::filesystem::path& directory, std::string_view label, uint32_t accumulatedFrames,
    DebugValue* rayCapture = nullptr, const RayCaptureRegion& region = kDetailRayRegion, bool contactVisibility = false)
{
    constexpr std::array<std::string_view, 3> keys{"path_trace", "realtime", "indirect"};
    constexpr std::array<std::string_view, 3> ids{"PathTrace.color", "Temporal.color", "Temporal.indirectDiffuse"};
    constexpr std::array<std::string_view, 3> formats{"rgba32f", "rgba32f", "rgba16f"};
    constexpr std::array<std::string_view, 3> layouts{"RGBA32F", "RGBA32F", "RGBA16F"};
    constexpr std::array<size_t, 3> strides{16, 16, 8};
    const DebugValue roi{{"x", 0}, {"y", 0}, {"width", kExtent}, {"height", kExtent}};
    DebugValue batches = DebugValue::array({
        {{"pass", "PathTrace"}, {"checkpoint", "AfterPass"},
            {"resources", DebugValue::array({{{"id", ids[0]}, {"roi", roi}}})}},
        {{"pass", "Temporal"}, {"checkpoint", "AfterPass"},
            {"resources", DebugValue::array({{{"id", ids[1]}, {"roi", roi}}, {{"id", ids[2]}, {"roi", roi}}})}},
    });
    if (rayCapture) { addRayCaptureBatches(batches, region); }
    const size_t contactJobIndex = batches.size();
    if (contactVisibility) {
        addContactBatch(batches, "Deferred", kContactDeferred, roi);
        addContactBatch(batches, "Gather", kContactGather, roi);
        addContactBatch(batches, "Temporal", kContactTemporal, roi);
        if (!rayCapture) { addContactBatch(batches, "Gather", kContactRaw, roi); }
    }
    capture.request = {{"method", "capture.batch"}, {"params", {{"batches", std::move(batches)}}}};
    renderFrame(capture.preview, graph);
    if (capture.reply.value("status", "") != "ok") {
        throw std::runtime_error("Cornell Box capture enqueue failed: " + capture.reply.dump());
    }
    capture.inspector.poll();
    const auto pathTrace = getCapture(capture, 0);
    const auto realtime = getCapture(capture, 1);
    const auto& pathStamp = pathTrace->snapshot.evidence;
    const auto& realtimeStamp = realtime->snapshot.evidence;
    checkSameFrame(pathStamp, realtimeStamp);
    const uint64_t frame = pathStamp.execution;
    DebugValue output{{"frame", frame}, {"accumulated_frames", accumulatedFrames},
        {"path_samples", accumulatedFrames * 4u}, {"evidence", {pathStamp.value(), realtimeStamp.value()}}};
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto& artifact = i == 0 ? pathTrace->artifacts.at(0) : realtime->artifacts.at(i - 1);
        const auto file = std::string(label) + "-" + std::string(keys[i]) + ".bin";
        saveReadback(artifact, directory / file, layouts[i], kExtent * kExtent * strides[i]);
        output[keys[i]] = {{"file", file}, {"format", formats[i]}, {"frame", frame}};
    }
    if (rayCapture) { *rayCapture = saveRayCapture(capture, directory, label, pathStamp, region); }
    if (contactVisibility) {
        output["contact_visibility"] = saveContactVisibilityCapture(capture, directory, label, pathStamp,
            contactJobIndex, rayCapture);
    }
    std::cout << "Captured " << label << " at graph execution " << frame << std::endl;
    return output;
}

void runBaseline(const std::filesystem::path& directory, const DebugValue& regions, BaselineOptions options)
{
    RenderSampleLoadResult sample;
    std::string log;
    if (!loadBuiltInRenderSample("cornell-box-compare", sample, log)) {
        throw std::runtime_error("Could not load GI Comparison: " + log);
    }
    const auto gather = *sample.graph.findNode("Gather");
    const auto gdf = *sample.graph.findNode("GlobalDistanceFieldPages");
    auto gatherProperties = gather.properties;
    auto gdfProperties = gdf.properties;
    const auto& rayRegion = options.floorRays ? kFloorRayRegion : kDetailRayRegion;
    if (options.trace) {
        // The fixed triangle and contact witnesses were captured at these tracing settings.
        gatherProperties["fixedProbeDirections"] = true;
        gatherProperties.erase("tracingOctahedronResolution");
        gatherProperties["probeTraceReadback"] = rayRegion.tiles;
        gatherProperties["screenTraceRelativeDepthThickness"] = 0.02f;
        const auto temporal = *sample.graph.findNode("Temporal");
        auto temporalProperties = temporal.properties;
        temporalProperties.erase("maxFramesAccumulated");
        temporalProperties.erase("stochasticHistoryQuantization");
        if (!sample.graph.setNodeProperties(temporal.id, temporalProperties)) {
            throw std::runtime_error("Could not configure the fixed Cornell Box temporal history");
        }
        gdfProperties["resolution"] = 112;
        gdfProperties["maxPhysicalPages"] = 4096;
        if (!sample.graph.setNodeProperties(gdf.id, gdfProperties)) {
            throw std::runtime_error("Could not configure the fixed Cornell Box distance-field grid");
        }
    }
    if (options.contactVisibility) { gatherProperties["probeInterpolationReadback"] = {0, 0, kExtent, kExtent}; }
    if (options.screenGDFContinuation) { gatherProperties["screenGDFContinuation"] = true; }
    if (options.gdfPrefixValidation) { gatherProperties["gdfPrefixValidation"] = true; }
    if (options.gdfPrefixRejectReceding) { gatherProperties["gdfPrefixRejectReceding"] = true; }
    if (options.gdfPrefixMeshValidation) { gatherProperties["gdfPrefixMeshValidation"] = true; }
    if (options.detail) { gatherProperties["detailTracing"] = true; }
    if (options.detail || options.gdfPrefixMeshValidation) {
        gatherProperties["detailDistanceFieldPaths"] =
            gdfProperties.at("distanceFieldPaths");
    }
    if (!sample.graph.setNodeProperties(gather.id, gatherProperties) ||
        (options.detail && !sample.graph.addEdge("VBuffer.depth", "Gather.detailDepth")) ||
        !saveRenderGraphToFile(sample.graph, directory / "graph.metallic_graph.json", log)) {
        throw std::runtime_error("Could not configure the Cornell Box capture graph: " + log);
    }
    RenderView view;
    scene::SceneDocument scene;
    if (!scene.load(sample.desc.scenePath)) {
        throw std::runtime_error(scene.lastLoadResult().error);
    }
    if (!view.setCameraProperties(sample.graph.viewProperties().at("camera")) ||
        !view.setRenderResolution(kExtent, kExtent)) {
        throw std::runtime_error("Could not configure the Cornell Box view");
    }
    const auto& materials = scene.materials();
    int32_t emitter = -1;
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto emission = materials[i].emissiveFactor;
        if (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f) {
            if (emitter >= 0) { throw std::runtime_error("Cornell Box must have a single emissive material"); }
            emitter = int32_t(i);
        }
    }
    if (emitter < 0) { throw std::runtime_error("Cornell Box has no emissive material"); }

    std::atomic_uint32_t errors = 0;
    const ValidationSink validation{.callback = [](void* context, const ValidationMessage& message) noexcept {
        if (message.severity == ValidationSeverity::Error) {
            static_cast<std::atomic_uint32_t*>(context)->fetch_add(1);
            std::cerr << message.message << '\n';
        }
    }, .context = &errors};
    DebugValue captures = DebugValue::object();
    DebugValue rayCapture;
    DebugValue temporalSeed;
    DebugValue temporalPhases = DebugValue::array();
    {
        BaselineCapture capture(options.contactVisibility);
        auto& preview = capture.preview;
        preview.bindRenderView(&view);
        preview.bindRuntimeScene(&scene);
        if (!preview.initialize(true, true, false, true, validation) ||
            !preview.setWorldEnvironment(scene.worldEnvironment()) || !preview.setLighting(scene.lighting())) {
            throw std::runtime_error("Cornell Box requires a Vulkan preview device with ray queries: " + preview.lastLog());
        }
        preview.setEnvironment({.enabled = false, .visible = false});
        preview.setDebugObserver(&capture);
        constexpr uint32_t temporalSeedFrame = kLitFrames - kContactProbePhases;
        for (uint32_t frame = 1; frame <= kLitFrames; ++frame) {
            const bool finalLitFrame = frame == kLitFrames;
            const bool temporalPhase = options.contactTemporalPhases && frame > temporalSeedFrame;
            if (frame == kLitFrames / 2) {
                captures["prior_on"] = captureFrame(capture, sample.graph, directory, "prior_on", frame);
            } else if (options.contactTemporalPhases && frame == temporalSeedFrame) {
                temporalSeed = captureFrame(capture, sample.graph, directory, "temporal_seed", frame);
            } else if (temporalPhase || finalLitFrame) {
                const auto label = finalLitFrame ? "on" : "phase_" + std::to_string(frame - temporalSeedFrame - 1);
                DebugValue phaseRays;
                auto phase = captureFrame(capture, sample.graph, directory, label, frame,
                    options.trace ? &phaseRays : nullptr, rayRegion, options.contactVisibility);
                if (temporalPhase) {
                    temporalPhases.push_back({{"capture", phase}, {"ray_capture", phaseRays}});
                }
                if (finalLitFrame) {
                    captures["on"] = std::move(phase);
                    rayCapture = std::move(phaseRays);
                }
            } else {
                renderFrame(preview, sample.graph);
            }
        }
        auto darkMaterial = materials[size_t(emitter)];
        darkMaterial.emissiveFactor = float3(0.0f);
        if (!scene.setMaterialProperties(emitter, darkMaterial)) {
            throw std::runtime_error("Could not zero the Cornell Box emissive material");
        }
        for (uint32_t frame = 1; frame <= kDarkFrames; ++frame) {
            if (frame == 1 || frame == kDarkFrames) {
                const auto label = frame == 1 ? "first_off" : "off";
                captures[label] = captureFrame(capture, sample.graph, directory, label, frame, nullptr, rayRegion,
                    options.contactTemporalPhases || (options.contactVisibility && frame == kDarkFrames));
            } else {
                renderFrame(preview, sample.graph);
            }
        }
    }
    if (errors != 0) { throw std::runtime_error("Cornell Box produced Vulkan validation errors"); }
    const double gdfExtent = gdfProperties.at("extent").get<double>();
    const uint32_t gdfResolution = gdfProperties.at("resolution").get<uint32_t>();
    DebugValue manifest{{"sample", sample.desc.id}, {"color_space", "acescg"},
        {"resolution", {kExtent, kExtent}}, {"view", sample.graph.viewProperties()}, {"regions", regions},
        {"detail_tracing", options.detail}, {"screen_traces", gatherProperties.at("screenTraces")},
        {"screen_trace_relative_depth_thickness", gatherProperties.value("screenTraceRelativeDepthThickness", 0.02f)},
        {"gdf_extent_m", gdfExtent}, {"gdf_resolution", gdfResolution},
        {"gdf_voxel_size_m", 2.0 * gdfExtent / gdfResolution},
        {"fixed_probe_directions", options.trace},
        {"screen_gdf_continuation", options.screenGDFContinuation || options.detail},
        {"gdf_prefix_validation", options.gdfPrefixValidation},
        {"gdf_prefix_reject_receding", options.gdfPrefixRejectReceding},
        {"gdf_prefix_mesh_validation", options.gdfPrefixMeshValidation},
        {"captures", captures}, {"vulkan_validation_errors", errors.load()}};
    if (options.trace) { manifest["ray_capture"] = std::move(rayCapture); }
    if (options.contactTemporalPhases) {
        manifest["temporal_seed"] = std::move(temporalSeed);
        manifest["temporal_phases"] = std::move(temporalPhases);
    }
    std::ofstream stream(directory / "capture.json");
    stream << manifest.dump(2) << '\n';
    if (!stream) { throw std::runtime_error("Could not write Cornell Box capture manifest"); }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: MetallicCornellBoxGIBaseline <new-output-directory> <regions.json>"
            " [--trace|--floor-rays] [--detail] [--screen-gdf-continuation|--gdf-prefix-validation]"
            " [--gdf-prefix-reject-receding|--gdf-prefix-mesh-validation]"
            " [--contact-visibility|--contact-temporal-phases]\n";
        return 1;
    }
    try {
        BaselineOptions options;
        for (int i = 3; i < argc; ++i) {
            const std::string_view argument = argv[i];
            if (argument == "--trace") { options.trace = true; }
            else if (argument == "--floor-rays") { options.trace = options.floorRays = true; }
            else if (argument == "--detail") { options.detail = true; }
            else if (argument == "--screen-gdf-continuation") { options.screenGDFContinuation = true; }
            else if (argument == "--gdf-prefix-validation") {
                options.screenGDFContinuation = options.gdfPrefixValidation = true;
            }
            else if (argument == "--gdf-prefix-reject-receding") {
                options.screenGDFContinuation = options.gdfPrefixValidation = options.gdfPrefixRejectReceding = true;
            }
            else if (argument == "--gdf-prefix-mesh-validation") {
                options.screenGDFContinuation = options.gdfPrefixValidation = true;
                options.gdfPrefixRejectReceding = options.gdfPrefixMeshValidation = true;
            }
            else if (argument == "--contact-visibility" || argument == "--contact-temporal-phases") {
                options.contactTemporalPhases |= argument == "--contact-temporal-phases";
                options.contactVisibility = options.trace = options.floorRays = true;
                options.screenGDFContinuation = options.gdfPrefixValidation = true;
                options.gdfPrefixRejectReceding = options.gdfPrefixMeshValidation = true;
            }
            else { throw std::runtime_error("Unknown Cornell Box capture option: " + std::string(argument)); }
        }
        if (options.contactVisibility && options.detail) {
            throw std::runtime_error("Cornell Box contact visibility captures require group F with Detail Tracing disabled");
        }
        if (sceneWorkingColorSpace() != SceneWorkingColorSpace::ACEScg) {
            throw std::runtime_error("Cornell Box baseline requires METALLIC_WORKING_COLOR_SPACE=acescg");
        }
        std::ifstream input(argv[2]);
        DebugValue regions;
        input >> regions;
        const std::filesystem::path directory = argv[1];
        if (!std::filesystem::create_directories(directory)) {
            throw std::runtime_error("Choose a new output directory");
        }
        if (!SDL_Init(SDL_INIT_VIDEO)) { throw std::runtime_error(SDL_GetError()); }
        struct SdlScope { ~SdlScope() { SDL_Quit(); } } sdl;
        if (!task::initializeTaskSystem()) { throw std::runtime_error("Could not initialize the task system"); }
        struct TaskScope { ~TaskScope() { task::shutdownTaskSystem(); } } tasks;
        runBaseline(directory, regions, options);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
