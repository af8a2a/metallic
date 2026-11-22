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
#include <stdexcept>

namespace {
using namespace metallic;
using namespace metallic::render;
using debug::DebugValue;

constexpr uint32_t kExtent = 384;
constexpr uint32_t kLitFrames = 256;
constexpr uint32_t kDarkFrames = 8;
constexpr std::string_view kOutput = "FinalBlit.color";
constexpr uint32_t kTileSize = 16;
constexpr uint32_t kProbesPerTile = 9;
constexpr uint32_t kRaysPerProbe = 64;
constexpr uint32_t kProbeCount = (kExtent / kTileSize) * (kExtent / kTileSize) * kProbesPerTile;
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
};

struct BaselineCapture final : IRenderDebugObserver {
    RenderGraphPreviewRenderer preview;
    RenderDebugRuntime inspector{{}, RenderDebugDetail::Resources};
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
    const debug::DebugEvidenceStamp& frame, const RayCaptureRegion& region)
{
    const auto deferred = getCapture(capture, 2);
    const auto gather = getCapture(capture, 3);
    checkSameFrame(frame, deferred->snapshot.evidence);
    checkSameFrame(frame, gather->snapshot.evidence);
    const auto pixels = region.pixels();
    const size_t pixelCount = pixels[2] * pixels[3];
    const auto& geometry = deferred->artifacts;
    const auto& traces = gather->artifacts;
    saveReadback(geometry.at(0), directory / "rays-position.bin", "RGBA32F", pixelCount * 16);
    saveReadback(geometry.at(1), directory / "rays-normal.bin", "RGBA16F", pixelCount * 8);
    saveReadback(traces.at(0), directory / "rays-probe-positions.bin", "u32", kProbeCount * 8);
    saveReadback(traces.at(1), directory / "rays-trace.bin", kLumenProbeTraceReadbackLayout,
        region.traces() * sizeof(LumenProbeTraceReadbackRecord));
    saveReadback(traces.at(2), directory / "rays-radiance.bin", "u32", kProbeCount * kRaysPerProbe * 16);
    saveReadback(traces.at(3), directory / "rays-distance.bin", "u32", kProbeCount * kRaysPerProbe * 4);
    DebugValue result{{"frame", frame.execution}, {"tiles", region.tiles}, {"pixel_rect", pixels},
        {"position", "rays-position.bin"}, {"normal", "rays-normal.bin"},
        {"probe_positions", "rays-probe-positions.bin"}, {"trace", "rays-trace.bin"},
        {"radiance", "rays-radiance.bin"}, {"distance", "rays-distance.bin"},
        {"evidence", {deferred->snapshot.evidence.value(), gather->snapshot.evidence.value()}}};
    if (region.captureHZB) {
        const auto screen = getCapture(capture, 4);
        checkSameFrame(frame, screen->snapshot.evidence);
        saveReadback(screen->artifacts.at(0), directory / "rays-closest-hzb.bin", "u32", kHZBElements * 4);
        result["closest_hzb"] = "rays-closest-hzb.bin";
        result["evidence"].push_back(screen->snapshot.evidence.value());
    }
    return result;
}

DebugValue captureFrame(BaselineCapture& capture, RenderGraph& graph,
    const std::filesystem::path& directory, std::string_view label, uint32_t accumulatedFrames,
    DebugValue* rayCapture = nullptr, const RayCaptureRegion& region = kDetailRayRegion)
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
    if (rayCapture) { *rayCapture = saveRayCapture(capture, directory, pathStamp, region); }
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
    auto gatherProperties = gather.properties;
    const auto& rayRegion = options.floorRays ? kFloorRayRegion : kDetailRayRegion;
    if (options.trace) { gatherProperties["probeTraceReadback"] = rayRegion.tiles; }
    if (options.screenGDFContinuation) { gatherProperties["screenGDFContinuation"] = true; }
    if (options.gdfPrefixValidation) { gatherProperties["gdfPrefixValidation"] = true; }
    if (options.gdfPrefixRejectReceding) { gatherProperties["gdfPrefixRejectReceding"] = true; }
    if (options.gdfPrefixMeshValidation) { gatherProperties["gdfPrefixMeshValidation"] = true; }
    if (options.detail) { gatherProperties["detailTracing"] = true; }
    if (options.detail || options.gdfPrefixMeshValidation) {
        gatherProperties["detailDistanceFieldPaths"] =
            sample.graph.findNode("GlobalDistanceFieldPages")->properties.at("distanceFieldPaths");
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
    {
        BaselineCapture capture;
        auto& preview = capture.preview;
        preview.bindRenderView(&view);
        preview.bindRuntimeScene(&scene);
        if (!preview.initialize(true, true, false, true, validation) ||
            !preview.setWorldEnvironment(scene.worldEnvironment()) || !preview.setLighting(scene.lighting())) {
            throw std::runtime_error("Cornell Box requires a Vulkan preview device with ray queries: " + preview.lastLog());
        }
        preview.setEnvironment({.enabled = false, .visible = false});
        preview.setDebugObserver(&capture);
        for (uint32_t frame = 1; frame <= kLitFrames; ++frame) {
            if (frame == kLitFrames / 2 || frame == kLitFrames) {
                const auto label = frame == kLitFrames ? "on" : "prior_on";
                captures[label] = captureFrame(capture, sample.graph, directory, label, frame,
                    options.trace && frame == kLitFrames ? &rayCapture : nullptr, rayRegion);
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
                captures[label] = captureFrame(capture, sample.graph, directory, label, frame);
            } else {
                renderFrame(preview, sample.graph);
            }
        }
    }
    if (errors != 0) { throw std::runtime_error("Cornell Box produced Vulkan validation errors"); }
    DebugValue manifest{{"sample", sample.desc.id}, {"color_space", "acescg"},
        {"resolution", {kExtent, kExtent}}, {"view", sample.graph.viewProperties()}, {"regions", regions},
        {"detail_tracing", options.detail}, {"screen_traces", gatherProperties.at("screenTraces")},
        {"screen_gdf_continuation", options.screenGDFContinuation || options.detail},
        {"gdf_prefix_validation", options.gdfPrefixValidation},
        {"gdf_prefix_reject_receding", options.gdfPrefixRejectReceding},
        {"gdf_prefix_mesh_validation", options.gdfPrefixMeshValidation},
        {"captures", captures}, {"vulkan_validation_errors", errors.load()}};
    if (options.trace) { manifest["ray_capture"] = std::move(rayCapture); }
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
            " [--gdf-prefix-reject-receding|--gdf-prefix-mesh-validation]\n";
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
            else { throw std::runtime_error("Unknown Cornell Box capture option: " + std::string(argument)); }
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
