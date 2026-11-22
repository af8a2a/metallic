#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenScreenProbePlacement.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Debug/LumenProbeTraceReadback.h"
#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <vector>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kScreenProbeGatherABI = 0x4c554d454e010013ull;
constexpr uint64_t kMeshSDFCullingABI = 0x4c554d454e060001ull;
constexpr uint32_t kGatherSamplesPerProbe = 8 * 8;
constexpr uint32_t kTraceStride = 4 * sizeof(float);
constexpr uint32_t kScreenTraceViewStride = 224;
constexpr uint32_t kRadianceCacheClipmapStride = 48;
constexpr uint32_t kSpatialPassCount = 3;
constexpr float kScreenTraceRelativeDepthThickness = 0.02f; // UE 5.7.4 HZBTraversal.RelativeDepthThickness.
constexpr uint32_t kIrradianceProbeWithBorderRes = 8;
constexpr uint32_t kProbeSlotsPerTileAxis = 3;
static_assert(kProbeSlotsPerTileAxis * kProbeSlotsPerTileAxis == kLumenScreenProbeSlotsPerTile);
constexpr uint32_t kMeshSDFGridTileSize = 64;
constexpr uint32_t kMeshSDFGridMaxDepth = 22;
constexpr uint32_t kMeshSDFCullThreads = 64;
constexpr uint32_t kMeshSDFMaxDispatchGroups = 65535;
constexpr uint32_t kMeshSDFMaxPairs = (128u * 1024 * 1024) / (2 * sizeof(uint32_t));
constexpr float kMeshSDFTraceDistance = 1.8f;
constexpr float kCardTraceEndDistance = 40.0f;
constexpr uint64_t kMaxProbeReadbackBytes = 128u * 1024u * 1024u;
constexpr uint32_t kProbeInterpolationReadbackStride = 4 * sizeof(uint32_t) + 4 * sizeof(float);

using ProbeReadbackRectangle = std::array<uint32_t, 4>;

uint64_t probeReadbackElementCount(const ProbeReadbackRectangle& rectangle)
{
    return uint64_t(rectangle[2]) * rectangle[3];
}

std::optional<ProbeReadbackRectangle> readProbeReadbackRectangle(const RenderGraphProperties& value, uint64_t bytesPerElement)
{
    if (!value.is_array() || value.size() != 4) { return std::nullopt; }
    ProbeReadbackRectangle rectangle{};
    for (size_t index = 0; index < rectangle.size(); ++index) {
        if (!value[index].is_number_integer()) { return std::nullopt; }
        const uint64_t component = value[index].get<uint64_t>();
        if (component > std::numeric_limits<uint32_t>::max()) { return std::nullopt; }
        rectangle[index] = static_cast<uint32_t>(component);
    }
    const uint64_t count = probeReadbackElementCount(rectangle);
    if (count == 0 || count > kMaxProbeReadbackBytes / bytesPerElement) {
        return std::nullopt;
    }
    return rectangle;
}

uint32_t meshSDFGridTiles(uint32_t extent)
{
    return (extent + kMeshSDFGridTileSize - 1u) / kMeshSDFGridTileSize;
}

struct LumenMeshSDFGatherBindings {
    MeshDistanceFieldGPUBindings meshes;
    ShaderBuffer instances, gridCells, gridIndices;
    std::array<uint32_t, 4> grid;
    std::array<float, 3> cameraPosition;
    float maxMeshSDFTraceDistance, cardTraceEndDistance;
    uint32_t bricksPerAxis;
};
static_assert(sizeof(LumenMeshSDFGatherBindings) == 68);

struct LumenMeshSDFCullingParams {
    ViewCameraConstants camera;
    std::array<float, 4> cull;
    std::array<uint32_t, 4> grid;
    ShaderSampledImage depth;
    ShaderBuffer tileDepth, bounds, instances;
    MeshDistanceFieldGPUBindings meshes;
    ShaderBuffer gridCells, gridIndices, gridPairs, gridState;
};
static_assert(sizeof(LumenMeshSDFCullingParams) == 160 && offsetof(LumenMeshSDFCullingParams, meshes) == 128);

struct LumenScreenProbeGatherParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderSampledImage position, normal, diffuse, directColor;
    ShaderBuffer traceOutput, traceInput;
    ShaderStorageImage indirectDiffuse, color;
    uint32_t width, height, probeWidth, probeHeight, clipmapCount;
    float maxTraceDistance;
    ShaderBuffer probePositions, traceDistance, closestHZB;
    ShaderSampledImage previousDepth, previousColor;
    ShaderBuffer traceView;
    uint32_t frameIndex;
    ShaderBuffer cacheIndirection, cacheClipmaps;
    ShaderSampledImage cacheRadiance;
    LumenMeshSDFGatherBindings detail;
    ShaderBuffer traceReadback;
    ProbeReadbackRectangle traceReadbackTiles;
    ShaderBuffer interpolationReadback;
    ProbeReadbackRectangle interpolationReadbackPixels;
    ShaderSampledImage blueNoiseVec2;
    ShaderSampledImage probeIrradiance;
    ShaderSampler irradianceSampler;
    float screenTraceRelativeDepthThickness;
    ShaderSampledImage shDirectional;
    uint32_t rayDirectionFrameIndex;
};
static_assert(sizeof(LumenScreenProbeGatherParams) == 272 && offsetof(LumenScreenProbeGatherParams, detail) == 140 &&
    offsetof(LumenScreenProbeGatherParams, traceReadback) == 208 &&
    offsetof(LumenScreenProbeGatherParams, interpolationReadback) == 228 &&
    offsetof(LumenScreenProbeGatherParams, blueNoiseVec2) == 248);

class LumenScreenProbeGatherPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override;
    Result<> prepare(const RenderGraphCompileContext& context, std::string& log) override
    {
        // A resource rebuild can reuse this pass while changing the trace shader's thread group size.
        const auto requested = properties().value("tracingOctahedronResolution", RenderGraphProperties(8));
        if (compiledTraceResolution_ && (!requested.is_number_integer() || requested != compiledTraceResolution_)) {
            return compile(context, log);
        }
        return {};
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        if (!optionEnabled("finalGather") || optionEnabled("fixedProbeDirections")) { return {}; }
        return {{.key = "tracingOctahedronResolution", .label = "Rays per Probe",
            .type = RenderGraphRuntimeSettingType::Enum, .defaultValue = 8,
            .options = {{"64 (Standard)", 8}, {"256 (Reference)", 16}},
            .invalidateHistory = true, .rebuildGraph = true}};
    }

private:
    bool optionEnabled(std::string_view name) const
    {
        const auto value = properties().find(name);
        return value != properties().end() && value->is_boolean() && value->get<bool>();
    }

    bool useSH3() const
    {
        const auto value = properties().find("irradianceFormat");
        return value != properties().end() && *value == "SH3";
    }

    uint32_t traceResolution() const
    {
        const auto value = properties().find("tracingOctahedronResolution");
        return value != properties().end() && *value == 16 ? 16u : 8u;
    }

    uint64_t traceReadbackBytesPerTile() const
    {
        const auto resolution = traceResolution();
        return kLumenScreenProbeSlotsPerTile * resolution * resolution * sizeof(LumenProbeTraceReadbackRecord);
    }

    Device* device_ = nullptr;
    ComputeKernel trace_;
    ComputeKernel integrate_;
    ComputeKernel place_;
    ComputeKernel composite_;
    ComputeKernel filter_;
    ComputeKernel convertToIrradiance_;
    std::unique_ptr<TextureView> blueNoise_;
    MeshDistanceFieldGPU meshFields_;
    ComputeKernel meshGridClear_, meshGridDepth_, meshGridScatter_, meshGridAllocate_, meshGridCompact_;
    uint32_t compiledTraceResolution_ = 0;
    uint32_t maxRayDirections_ = 8;
    float maxTraceDistance_ = 4.0f;
    float screenTraceRelativeDepthThickness_ = kScreenTraceRelativeDepthThickness;
    float meshSDFTraceDistance_ = kMeshSDFTraceDistance;
    std::optional<ProbeReadbackRectangle> traceReadbackTiles_;
    std::optional<ProbeReadbackRectangle> interpolationReadbackPixels_;
};

RenderPassReflection LumenScreenProbeGatherPass::reflect(const RenderGraphCompileContext& context) const
{
    RenderPassReflection reflection;
    reflectLumenGlobalTraceInputs(reflection);
    reflection.addTextureInput("position", "Receiver world position and positive linear view depth")
        .sampledRead().format = Format::RGBA32Sfloat;
    reflection.addTextureInput("normal", "Receiver shading normal").sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("diffuse", "Receiver diffuse response").sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("directColor", "Unexposed deferred scene color before indirect diffuse")
        .sampledRead().format = Format::RGBA32Sfloat;

    if (optionEnabled("screenTraces")) {
        reflection.addBufferInput("closestHZB", "Current closest reversed-Z depth hierarchy")
            .buffer(0, sizeof(float)).storageRead();
        reflection.addTextureInput("previousDepth", "Previous device depth").sampledRead().format = Format::R32Sfloat;
        reflection.addTextureInput("previousColor", "Previous scene-linear direct and emissive color")
            .sampledRead().format = Format::RGBA32Sfloat;
        reflection.addBufferInput("traceView", "Screen trace view and history validity")
            .buffer(kScreenTraceViewStride, kScreenTraceViewStride).storageRead();
    }
    if (optionEnabled("detailTracing")) {
        reflection.addTextureInput("detailDepth", "Current reversed-Z depth for Mesh SDF froxel culling")
            .sampledRead().format = Format::D32Sfloat;
        const uint64_t tiles = uint64_t(meshSDFGridTiles(context.width)) * meshSDFGridTiles(context.height);
        const uint64_t cells = tiles * kMeshSDFGridMaxDepth;
        const uint64_t maxInstances = context.runtimeScene ? context.runtimeScene->renderNodes().size() : 0u;
        const uint64_t capacity = std::clamp(cells * maxInstances, uint64_t{1}, uint64_t{kMeshSDFMaxPairs});
        reflection.addBufferOutput("detailTileDepth", "Closest and furthest reversed-Z depth per froxel tile")
            .buffer(tiles * 8, 8).storageReadWrite();
        reflection.addBufferOutput("detailGridCells", "Mesh SDF candidate start and count per froxel")
            .buffer(cells * 8, 8).storageReadWrite().bufferLayout("RG32U");
        reflection.addBufferOutput("detailGridPairs", "Scattered Mesh SDF instance and froxel indices")
            .buffer(capacity * 8, 8).storageReadWrite();
        reflection.addBufferOutput("detailGridIndices", "Compacted Mesh SDF instance indices")
            .buffer(capacity * 4, 4).storageReadWrite().bufferLayout("u32");
        reflection.addBufferOutput("detailGridState", "Candidate count and compacted index allocator")
            .buffer(4 * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite().bufferLayout("u32");
    }

    const uint64_t probeCount = uint64_t(lumenScreenProbeTiles(context.width)) * lumenScreenProbeTiles(context.height);
    const bool finalGather = optionEnabled("finalGather");
    const uint64_t probePositionsBytes = probeCount * kLumenScreenProbeSlotsPerTile * 2 * sizeof(uint32_t);
    const bool radianceCache = optionEnabled("radianceCache");
    if (radianceCache) {
        reflection.addBufferInput("placedProbes", "Uniform and adaptive receiver pixels used to mark the World Radiance Cache")
            .buffer(probePositionsBytes, 2 * sizeof(uint32_t)).storageRead();
        reflection.addBufferInput("cacheIndirection", "World Radiance Cache clipmap cell-to-probe indices")
            .buffer(0, sizeof(uint32_t)).storageRead();
        reflection.addBufferInput("cacheClipmaps", "World Radiance Cache clipmap geometry and atlas layout")
            .buffer(0, kRadianceCacheClipmapStride).storageRead();
        auto& cacheRadiance = reflection.addTextureInput("cacheRadiance", "Bordered directional World Radiance Cache atlas")
            .sampledRead();
        cacheRadiance.format = Format::RGBA16Sfloat;
        cacheRadiance.matchOutputExtent = false;
    }
    const uint64_t slots = probeCount * (finalGather ? kLumenScreenProbeSlotsPerTile : 1u);
    const uint64_t rayCount = slots * traceResolution() * traceResolution();
    const uint64_t gatherCount = slots * kGatherSamplesPerProbe;
    reflection.addBufferOutput("probeRadiance", "Probe ray RGB and trace classification in tile order")
        .buffer(rayCount * kTraceStride, kTraceStride).storageReadWrite()
        .transient(RenderGraphInitialization::FullOverwrite);
    if (finalGather) {
        if (!radianceCache) {
            reflection.addBufferOutput("probePositions", "Uniform and adaptive receiver pixels by tile and slot")
                .buffer(probePositionsBytes, 2 * sizeof(uint32_t)).storageReadWrite()
                .transient(RenderGraphInitialization::FullOverwrite);
        }
        reflection.addBufferOutput("probeDistance", "Per-direction trace distance in meters")
            .buffer(rayCount * sizeof(float), sizeof(float)).storageReadWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
        if (!optionEnabled("fixedProbeDirections")) {
            reflection.addBufferOutput("compositedProbeDistance", "Minimum trace distance per gather direction in meters")
                .buffer(gatherCount * sizeof(float), sizeof(float)).storageReadWrite()
                .transient(RenderGraphInitialization::FullOverwrite);
            const uint32_t atlasWidth = lumenScreenProbeTiles(context.width) * kProbeSlotsPerTileAxis;
            const uint32_t atlasHeight = lumenScreenProbeTiles(context.height) * kProbeSlotsPerTileAxis;
            const auto addAtlas = [&](const char* name, const char* description, uint32_t width, uint32_t height, Format format) {
                auto& atlas = reflection.addTextureOutput(name, description).texture2D(width, height)
                    .storageWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead)
                    .transient(RenderGraphInitialization::FullOverwrite);
                atlas.format = format;
                atlas.matchOutputExtent = false;
            };
            if (useSH3()) {
                addAtlas("probeSHAmbient", "Screen probe ambient SH coefficient", atlasWidth, atlasHeight, Format::B10G11R11UfloatPack32);
                addAtlas("probeSHDirectional", "Screen probe directional SH coefficients in six horizontal slices",
                    atlasWidth * 6u, atlasHeight, Format::RGBA16Sfloat);
            } else {
                addAtlas("probeIrradiance", "Bordered directional diffuse irradiance per screen probe",
                    atlasWidth * kIrradianceProbeWithBorderRes, atlasHeight * kIrradianceProbeWithBorderRes, Format::B10G11R11UfloatPack32);
            }
        }
        if (const auto readback = properties().find("probeTraceReadback"); readback != properties().end()) {
            if (const auto tiles = readProbeReadbackRectangle(*readback, traceReadbackBytesPerTile())) {
                reflection.addBufferOutput("probeTraceReadback", "Screen, Detail, GDF and WRC state per selected probe ray")
                    .buffer(probeReadbackElementCount(*tiles) * traceReadbackBytesPerTile(),
                        sizeof(LumenProbeTraceReadbackRecord))
                    .storageWrite().transient(RenderGraphInitialization::FullOverwrite)
                    .bufferLayout(kLumenProbeTraceReadbackLayout);
            }
        }
        if (const auto readback = properties().find("probeInterpolationReadback"); readback != properties().end()) {
            if (const auto pixels = readProbeReadbackRectangle(*readback, kProbeInterpolationReadbackStride)) {
                reflection.addBufferOutput("probeInterpolationReadback", "Final probe indices and FP32 weights per selected pixel")
                    .buffer(probeReadbackElementCount(*pixels) * kProbeInterpolationReadbackStride, kProbeInterpolationReadbackStride)
                    .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
            }
        }
        reflection.addBufferOutput("filteredProbeRadiance", "Spatially filtered probe ray RGB")
            .buffer(gatherCount * kTraceStride, kTraceStride).storageReadWrite().transient(RenderGraphInitialization::FullOverwrite);
        reflection.addBufferOutput("probeFilterScratch", "Spatial filtering ping-pong buffer")
            .buffer(gatherCount * kTraceStride, kTraceStride).storageReadWrite().transient(RenderGraphInitialization::FullOverwrite);
    }

    auto& indirect = reflection.addTextureOutput("indirectDiffuse", "Unexposed indirect diffuse after receiver response")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    indirect.format = Format::RGBA16Sfloat;
    indirect.colorEncoding = DisplayColorEncoding::SceneLinear;
    auto& color = reflection.addTextureOutput("color", "Unexposed scene color with indirect diffuse")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    color.format = Format::RGBA32Sfloat;
    color.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenScreenProbeGatherPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    for (std::string_view name : {"finalGather", "screenTraces", "stochasticInterpolation", "radianceCache",
            "detailTracing", "screenGDFContinuation", "gdfPrefixValidation", "gdfPrefixRejectReceding",
            "gdfPrefixMeshValidation", "fixedProbeDirections"}) {
        const auto value = properties().find(name);
        if (value != properties().end() && !value->is_boolean()) {
            log += "Lumen Screen Probe Gather requires a boolean " + std::string(name);
            return makeError(Error::InvalidArgument);
        }
    }
    const bool finalGather = optionEnabled("finalGather");
    const bool jitterDirections = finalGather && !optionEnabled("fixedProbeDirections");
    const bool screenTraces = optionEnabled("screenTraces");
    const bool stochasticInterpolation = optionEnabled("stochasticInterpolation");
    const bool radianceCache = optionEnabled("radianceCache");
    const bool detailTracing = optionEnabled("detailTracing");
    if (const auto format = properties().find("irradianceFormat"); format != properties().end() &&
        *format != "SH3" && *format != "OCT") {
        log = "Lumen irradianceFormat must be OCT or SH3";
        return makeError(Error::InvalidArgument);
    }
    if (const auto resolution = properties().find("tracingOctahedronResolution"); resolution != properties().end() &&
        (!resolution->is_number_integer() || (*resolution != 8 && (*resolution != 16 || !jitterDirections)))) {
        log = "Lumen tracingOctahedronResolution must be 8, or 16 with jittered Final Gather";
        return makeError(Error::InvalidArgument);
    }
    compiledTraceResolution_ = traceResolution();
    const bool screenGDFContinuation = optionEnabled("screenGDFContinuation");
    const bool gdfPrefixValidation = optionEnabled("gdfPrefixValidation");
    const bool gdfPrefixRejectReceding = optionEnabled("gdfPrefixRejectReceding");
    const bool gdfPrefixMeshValidation = optionEnabled("gdfPrefixMeshValidation");
    if (screenTraces && !finalGather) {
        log += "Lumen screen tracing requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    if (screenGDFContinuation && !screenTraces) {
        log += "Lumen screenGDFContinuation requires screenTraces";
        return makeError(Error::InvalidArgument);
    }
    if (gdfPrefixValidation && (!screenGDFContinuation || detailTracing)) {
        log += "Lumen gdfPrefixValidation requires screenGDFContinuation with detailTracing disabled";
        return makeError(Error::InvalidArgument);
    }
    if (gdfPrefixRejectReceding && !gdfPrefixValidation) {
        log += "Lumen gdfPrefixRejectReceding requires gdfPrefixValidation";
        return makeError(Error::InvalidArgument);
    }
    if (gdfPrefixMeshValidation && !gdfPrefixValidation) {
        log += "Lumen gdfPrefixMeshValidation requires gdfPrefixValidation";
        return makeError(Error::InvalidArgument);
    }
    if (stochasticInterpolation && !finalGather) {
        log += "Lumen stochastic probe interpolation requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    if (radianceCache && !finalGather) {
        log += "Lumen World Radiance Cache requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    if (detailTracing && !finalGather) {
        log = "Lumen Detail Tracing requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    traceReadbackTiles_.reset();
    if (const auto readback = properties().find("probeTraceReadback"); readback != properties().end()) {
        const auto tiles = readProbeReadbackRectangle(*readback, traceReadbackBytesPerTile());
        const uint32_t tileWidth = lumenScreenProbeTiles(context.width);
        const uint32_t tileHeight = lumenScreenProbeTiles(context.height);
        if (!finalGather || !tiles || uint64_t((*tiles)[0]) + (*tiles)[2] > tileWidth ||
            uint64_t((*tiles)[1]) + (*tiles)[3] > tileHeight) {
            log = "Lumen probeTraceReadback requires Final Gather and an in-bounds [tileX, tileY, width, height] "
                "rectangle with at most 128 MiB of records";
            return makeError(Error::InvalidArgument);
        }
        traceReadbackTiles_ = tiles;
    }
    interpolationReadbackPixels_.reset();
    if (const auto readback = properties().find("probeInterpolationReadback"); readback != properties().end()) {
        const auto pixels = readProbeReadbackRectangle(*readback, kProbeInterpolationReadbackStride);
        if (!finalGather || !pixels || uint64_t((*pixels)[0]) + (*pixels)[2] > context.width ||
            uint64_t((*pixels)[1]) + (*pixels)[3] > context.height) {
            log = "Lumen probeInterpolationReadback requires Final Gather and an in-bounds [x, y, width, height] "
                "pixel rectangle with at most 128 MiB of records";
            return makeError(Error::InvalidArgument);
        }
        interpolationReadbackPixels_ = pixels;
    }
    const auto phases = properties().find("maxRayDirections");
    if (phases != properties().end() && (!phases->is_number_integer() || *phases < 1 ||
        *phases > std::numeric_limits<int32_t>::max())) {
        log = "Lumen maxRayDirections must be a positive 32-bit integer";
        return makeError(Error::InvalidArgument);
    }
    maxRayDirections_ = properties().value("maxRayDirections", 8u);
    const auto thickness = properties().find("screenTraceRelativeDepthThickness");
    if (thickness != properties().end() && (!thickness->is_number() || !std::isfinite(thickness->get<float>()) ||
            thickness->get<float>() < 0.0f)) {
        log = "Lumen Screen Probe Gather screenTraceRelativeDepthThickness must be finite and nonnegative";
        return makeError(Error::InvalidArgument);
    }
    screenTraceRelativeDepthThickness_ = properties().value("screenTraceRelativeDepthThickness", kScreenTraceRelativeDepthThickness);
    const auto distance = properties().find("maxTraceDistance");
    if (distance != properties().end()) {
        if (!distance->is_number() || !std::isfinite(distance->get<float>()) || distance->get<float>() <= 0.0001f) {
            log += "Lumen Screen Probe Gather maxTraceDistance must be finite and greater than 0.0001 meters";
            return makeError(Error::InvalidArgument);
        }
        maxTraceDistance_ = distance->get<float>();
    }

    std::vector<SlangMacroDefine> defines;
    if (finalGather) { defines.push_back({"METALLIC_LUMEN_FINAL_GATHER", "1"}); }
    if (jitterDirections) {
        if (!context.graphicsQueue) {
            log = "Lumen Screen Probe Gather requires a graphics queue to upload blue noise";
            return makeError(Error::InvalidArgument);
        }
        auto blueNoise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Vec2, log);
        if (!blueNoise) { return makeError(blueNoise.error()); }
        blueNoise_ = std::move(*blueNoise);
        defines.push_back({"METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS", "1"});
    }
    if (compiledTraceResolution_ == 16u) { defines.push_back({"METALLIC_LUMEN_TRACE_RESOLUTION_16", "1"}); }
    if (screenTraces) { defines.push_back({"METALLIC_LUMEN_SCREEN_TRACES", "1"}); }
    if (screenGDFContinuation) { defines.push_back({"METALLIC_LUMEN_SCREEN_GDF_CONTINUATION", "1"}); }
    if (gdfPrefixValidation) { defines.push_back({"METALLIC_LUMEN_GDF_PREFIX_VALIDATION", "1"}); }
    if (gdfPrefixRejectReceding) { defines.push_back({"METALLIC_LUMEN_GDF_PREFIX_REJECT_RECEDING", "1"}); }
    if (gdfPrefixMeshValidation) { defines.push_back({"METALLIC_LUMEN_GDF_PREFIX_MESH_VALIDATION", "1"}); }
    if (stochasticInterpolation) { defines.push_back({"METALLIC_LUMEN_STOCHASTIC_INTERPOLATION", "1"}); }
    if (radianceCache) { defines.push_back({"METALLIC_LUMEN_RADIANCE_CACHE", "1"}); }
    if (traceReadbackTiles_) { defines.push_back({"METALLIC_LUMEN_PROBE_TRACE_READBACK", "1"}); }
    if (detailTracing || gdfPrefixMeshValidation) {
        const auto paths = properties().find("detailDistanceFieldPaths");
        if (!context.runtimeScene || !context.graphicsQueue || paths == properties().end() || !paths->is_array() || paths->empty()) {
            log = "Lumen Mesh SDF tracing requires a scene, graphics queue and detailDistanceFieldPaths";
            return makeError(Error::InvalidArgument);
        }
        std::vector<std::filesystem::path> fields;
        for (const auto& path : *paths) {
            if (!path.is_string()) {
                log = "Lumen detailDistanceFieldPaths must contain paths";
                return makeError(Error::InvalidArgument);
            }
            fields.push_back(std::filesystem::path(PROJECT_SOURCE_DIR) / path.get_ref<const std::string&>());
        }
        if (auto result = meshFields_.initialize(*device_, *context.graphicsQueue, *context.runtimeScene, fields, log); !result) {
            return result;
        }
    }
    if (detailTracing) {
        defines.push_back({"METALLIC_LUMEN_DETAIL_TRACING", "1"});
        meshSDFTraceDistance_ = kMeshSDFTraceDistance;
        if (const auto traceDistance = properties().find("meshSDFTraceDistance"); traceDistance != properties().end()) {
            if (!traceDistance->is_number() || !std::isfinite(traceDistance->get<float>()) || traceDistance->get<float>() <= 0.0001f) {
                log = "Lumen Detail Tracing meshSDFTraceDistance must be finite and greater than 0.0001 meters";
                return makeError(Error::InvalidArgument);
            }
            meshSDFTraceDistance_ = traceDistance->get<float>();
        }
        meshSDFTraceDistance_ = std::min(meshSDFTraceDistance_, maxTraceDistance_);
        using KernelEntry = std::pair<const char*, ComputeKernel*>;
        const KernelEntry culling[]{
            {"lumenMeshSDFGridClearMain", &meshGridClear_}, {"lumenMeshSDFGridDepthMain", &meshGridDepth_},
            {"lumenMeshSDFGridScatterMain", &meshGridScatter_}, {"lumenMeshSDFGridAllocateMain", &meshGridAllocate_},
            {"lumenMeshSDFGridCompactMain", &meshGridCompact_},
        };
        for (const auto& [entry, kernel] : culling) {
            auto result = ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/LumenMeshSDFCulling", .entryPointName = entry,
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<LumenMeshSDFCullingParams>(kMeshSDFCullingABI), .debugName = entry}, *kernel, log);
            if (!result) { return result; }
        }
    }
    const auto createGather = [&](const char* entry, const char* name, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenScreenProbeGather", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .macroDefines = defines},
            {.parameters = parameterAbi<LumenScreenProbeGatherParams>(kScreenProbeGatherABI), .debugName = name}, kernel, log);
    };
    if (auto result = createGather("lumenTraceScreenProbesMain", "LumenTraceScreenProbes", trace_); !result) {
        return result;
    }
    if (jitterDirections && useSH3()) { defines.push_back({"METALLIC_LUMEN_IRRADIANCE_SH3", "1"}); }
    if (interpolationReadbackPixels_) { defines.push_back({"METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK", "1"}); }
    if (auto result = createGather("lumenIntegrateScreenProbesMain", "LumenIntegrateScreenProbes", integrate_); !result || !finalGather) {
        return result;
    }

    std::vector<SlangMacroDefine> filterDefines;
    if (jitterDirections) { filterDefines.push_back({"METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS", "1"}); }
    if (compiledTraceResolution_ == 16u) { filterDefines.push_back({"METALLIC_LUMEN_TRACE_RESOLUTION_16", "1"}); }
    if (jitterDirections && useSH3()) { filterDefines.push_back({"METALLIC_LUMEN_IRRADIANCE_SH3", "1"}); }
    const auto createFilter = [&](const char* entry, const char* name, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenScreenProbeFilter", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = filterDefines},
            {.parameters = parameterAbi<LumenScreenProbeFilterParams>(kScreenProbeFilterABI), .debugName = name}, kernel, log);
    };
    if (!radianceCache) {
        if (auto result = createFilter("lumenPlaceAdaptiveProbesMain", "LumenPlaceAdaptiveProbes", place_); !result) {
            return result;
        }
    }
    if (jitterDirections) {
        if (auto result = createFilter("lumenCompositeScreenProbesMain", "LumenCompositeScreenProbes", composite_); !result) {
            return result;
        }
        if (auto result = createFilter("lumenConvertScreenProbesToIrradianceMain", "LumenConvertScreenProbesToIrradiance",
                convertToIrradiance_); !result) {
            return result;
        }
    }
    return createFilter("lumenFilterScreenProbesMain", "LumenFilterScreenProbes", filter_);
}

Result<> LumenScreenProbeGatherPass::execute(RenderGraphExecutionContext& context)
{
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    LumenScreenProbeGatherParams params{};
    params.scene = bindLumenGlobalTrace(context, writer);
    params.position = writer.sampledImageHandle(context.inputTexture("position").view());
    params.normal = writer.sampledImageHandle(context.inputTexture("normal").view());
    params.diffuse = writer.sampledImageHandle(context.inputTexture("diffuse").view());
    params.directColor = writer.sampledImageHandle(context.inputTexture("directColor").view());
    params.traceOutput = writer.buffer(context.outputBuffer("probeRadiance").buffer());
    params.traceInput = params.traceOutput;
    params.indirectDiffuse = writer.storageImageHandle(context.outputTexture("indirectDiffuse").view());
    params.color = writer.storageImageHandle(context.outputTexture("color").view());
    params.width = context.width();
    params.height = context.height();
    params.probeWidth = lumenScreenProbeTiles(params.width);
    params.probeHeight = lumenScreenProbeTiles(params.height);
    params.clipmapCount = *clipmapCount;
    params.maxTraceDistance = maxTraceDistance_;
    params.frameIndex = static_cast<uint32_t>(context.frameIndex());
    params.rayDirectionFrameIndex = params.frameIndex % maxRayDirections_;
    params.screenTraceRelativeDepthThickness = screenTraceRelativeDepthThickness_;
    const bool finalGather = optionEnabled("finalGather");
    const bool jitterDirections = finalGather && !optionEnabled("fixedProbeDirections");
    const bool screenTraces = optionEnabled("screenTraces");
    const bool radianceCache = optionEnabled("radianceCache");
    const bool detailTracing = optionEnabled("detailTracing");
    const bool sh3 = jitterDirections && useSH3();
    const char* irradianceName = sh3 ? "probeSHAmbient" : "probeIrradiance";
    const char* probePositionsName = radianceCache ? "placedProbes" : "probePositions";
    ShaderBuffer scratch;
    ShaderBuffer compositedDistance;
    if (finalGather) {
        params.probePositions = writer.buffer(radianceCache ? context.inputBuffer(probePositionsName).buffer() :
            context.outputBuffer(probePositionsName).buffer());
        params.traceDistance = writer.buffer(context.outputBuffer("probeDistance").buffer());
        params.traceInput = writer.buffer(context.outputBuffer("filteredProbeRadiance").buffer());
        scratch = writer.buffer(context.outputBuffer("probeFilterScratch").buffer());
        if (jitterDirections) {
            params.blueNoiseVec2 = writer.sampledImageHandle(blueNoise_.get());
            compositedDistance = writer.buffer(context.outputBuffer("compositedProbeDistance").buffer());
            params.probeIrradiance = writer.sampledImageHandle(context.outputTexture(irradianceName).view());
            if (sh3) {
                params.shDirectional = writer.sampledImageHandle(context.outputTexture("probeSHDirectional").view());
            } else {
                params.irradianceSampler = writer.samplerHandle({});
            }
        }
    }
    if (screenTraces) {
        params.closestHZB = writer.buffer(context.inputBuffer("closestHZB").buffer());
        params.previousDepth = writer.sampledImageHandle(context.inputTexture("previousDepth").view());
        params.previousColor = writer.sampledImageHandle(context.inputTexture("previousColor").view());
        params.traceView = writer.buffer(context.inputBuffer("traceView").buffer());
    }
    if (radianceCache) {
        params.cacheIndirection = writer.buffer(context.inputBuffer("cacheIndirection").buffer());
        params.cacheClipmaps = writer.buffer(context.inputBuffer("cacheClipmaps").buffer());
        params.cacheRadiance = writer.sampledImageHandle(context.inputTexture("cacheRadiance").view());
    }
    if (traceReadbackTiles_) {
        params.traceReadback = writer.buffer(context.outputBuffer("probeTraceReadback").buffer());
        params.traceReadbackTiles = *traceReadbackTiles_;
    }
    if (interpolationReadbackPixels_) {
        params.interpolationReadback = writer.buffer(context.outputBuffer("probeInterpolationReadback").buffer());
        params.interpolationReadbackPixels = *interpolationReadbackPixels_;
    }
    EncodedParameters meshGridParams;
    std::unique_ptr<Buffer> meshInstances, meshBounds, validationBounds;
    uint32_t meshCellGroups = 0, meshObjectGroups = 0, meshCompactGroups = 0;
    if (detailTracing) {
        const auto* view = context.viewConstants();
        if (!view || view->current.upProjection[3] > 0.5f) {
            spdlog::error("Lumen Detail Tracing requires a perspective graph camera");
            return makeError(Error::InvalidArgument);
        }
        const auto* world = context.world();
        if (world && (world->environment().enabled ||
            world->worldEnvironment().source == environment::EnvironmentSource::PhysicalAtmosphere)) {
            spdlog::error("This Lumen Detail Tracing permutation requires skylight to be disabled");
            return makeError(Error::InvalidArgument);
        }
        const float depthRange = kCardTraceEndDistance - view->current.clipOrtho[0];
        const uint32_t slices = depthRange > 0.0f ? uint32_t(std::max(0.0f, std::trunc(std::log2(depthRange) * 4.0f))) + 1u : 1u;
        uint32_t instanceCount = 0, unsupportedTransforms = 0;
        auto instances = meshFields_.uploadInstances(*context.runtimeScene(), instanceCount, unsupportedTransforms, &meshBounds);
        if (!instances) { return makeError(instances.error()); }
        meshInstances = std::move(*instances);
        const std::array grid{meshSDFGridTiles(params.width), meshSDFGridTiles(params.height), slices, instanceCount};
        const uint32_t cellCount = grid[0] * grid[1] * grid[2];
        const uint32_t capacity = uint32_t(context.outputBuffer("detailGridPairs").desc().size / 8u);
        meshCellGroups = (cellCount + kMeshSDFCullThreads - 1u) / kMeshSDFCullThreads;
        meshObjectGroups = (instanceCount + kMeshSDFCullThreads - 1u) / kMeshSDFCullThreads;
        if (uint64_t(cellCount) * instanceCount > capacity || meshCellGroups > kMeshSDFMaxDispatchGroups ||
            meshObjectGroups > kMeshSDFMaxDispatchGroups) {
            spdlog::error("Lumen Detail Tracing candidate pool cannot hold {} instances across {} froxels", instanceCount, cellCount);
            return makeError(Error::Unsupported);
        }
        const uint32_t candidateSlots = cellCount * instanceCount;
        meshCompactGroups = (candidateSlots + kMeshSDFCullThreads - 1u) / kMeshSDFCullThreads;
        const auto bindOutput = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
        const auto meshBindings = meshFields_.bind(writer);
        const auto instancesHandle = writer.buffer(meshInstances.get());
        params.detail = {meshBindings, instancesHandle, bindOutput("detailGridCells"), bindOutput("detailGridIndices"),
            grid, {view->current.eye[0], view->current.eye[1], view->current.eye[2]}, meshSDFTraceDistance_,
            kCardTraceEndDistance, meshFields_.bricksPerAxis()};
        const auto& cull = writer.encode(LumenMeshSDFCullingParams{view->current,
            {view->jitter[0], view->jitter[1], meshSDFTraceDistance_, kCardTraceEndDistance}, grid,
            writer.sampledImageHandle(context.inputTexture("detailDepth").view()), bindOutput("detailTileDepth"),
            writer.buffer(meshBounds.get()), instancesHandle, meshBindings, params.detail.gridCells, params.detail.gridIndices,
            bindOutput("detailGridPairs"), bindOutput("detailGridState")}, kMeshSDFCullingABI);
        if (!cull) { return makeError(cull.error()); }
        meshGridParams = *cull;
        if (context.detailedDebugSnapshots()) {
            context.debugCheckpoint("Lumen Mesh SDF Grid", {}, {{"instanceCount", instanceCount},
                {"unsupportedTransforms", unsupportedTransforms}, {"cellCount", cellCount}, {"capacity", capacity}});
        }
    } else if (optionEnabled("gdfPrefixMeshValidation")) {
        uint32_t instanceCount = 0, unsupportedTransforms = 0;
        auto instances = meshFields_.uploadInstances(*context.runtimeScene(), instanceCount, unsupportedTransforms,
            nullptr, &validationBounds);
        if (!instances) { return makeError(instances.error()); }
        meshInstances = std::move(*instances);
        params.detail.meshes = meshFields_.bind(writer);
        params.detail.instances = writer.buffer(meshInstances.get());
        // The prefix permutation uses this slot for object bounds instead of froxel cells.
        params.detail.gridCells = writer.buffer(validationBounds.get());
        params.detail.grid[3] = instanceCount;
        params.detail.bricksPerAxis = meshFields_.bricksPerAxis();
    }
    const auto encoded = writer.encode(params, kScreenProbeGatherABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    std::vector<RenderGraphStageUse> traceUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead},
        {"coarseMip", Access::TextureSampleRead}, {"domain", Access::BufferStorageRead},
        {"objectGrid", Access::BufferStorageRead}, {"objectGridScene", Access::BufferStorageRead},
        {"cards", Access::BufferStorageRead}, {"cardScene", Access::BufferStorageRead},
        {"cardDepth", Access::TextureSampleRead}, {"cardLighting", Access::TextureSampleRead},
        {"probeRadiance", Access::BufferStorageWrite},
    };
    std::vector<RenderGraphStageUse> integrateUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"diffuse", Access::TextureSampleRead}, {"directColor", Access::TextureSampleRead},
        {"indirectDiffuse", Access::TextureStorageWrite},
        {"color", Access::TextureStorageWrite},
    };
    if (jitterDirections) {
        integrateUses.push_back({irradianceName, Access::TextureSampleRead});
        if (sh3) { integrateUses.push_back({"probeSHDirectional", Access::TextureSampleRead}); }
    } else {
        integrateUses.push_back({finalGather ? "filteredProbeRadiance" : "probeRadiance", Access::BufferStorageRead});
    }
    if (screenTraces) {
        traceUses.insert(traceUses.end(), {{"closestHZB", Access::BufferStorageRead},
            {"previousDepth", Access::TextureSampleRead}, {"previousColor", Access::TextureSampleRead},
            {"traceView", Access::BufferStorageRead}});
    }
    if (radianceCache) {
        traceUses.insert(traceUses.end(), {{"cacheIndirection", Access::BufferStorageRead},
            {"cacheClipmaps", Access::BufferStorageRead},
            {"cacheRadiance", Access::TextureSampleRead}});
    }
    if (traceReadbackTiles_) {
        traceUses.push_back({"probeTraceReadback", Access::BufferStorageWrite});
    }
    if (interpolationReadbackPixels_) {
        integrateUses.push_back({"probeInterpolationReadback", Access::BufferStorageWrite});
    }

    const RenderGraphStageUse placeUses[] = {{"position", Access::TextureSampleRead},
        {"normal", Access::TextureSampleRead}, {"probePositions", Access::BufferStorageWrite}};
    std::array<std::array<RenderGraphStageUse, 5>, kSpatialPassCount> filterUses;
    EncodedParameters placeParams;
    EncodedParameters compositeParams;
    EncodedParameters irradianceParams;
    std::vector<RenderGraphStageUse> irradianceUses = {{probePositionsName, Access::BufferStorageRead},
        {"filteredProbeRadiance", Access::BufferStorageRead}, {irradianceName, Access::TextureStorageWrite}};
    if (sh3) { irradianceUses.push_back({"probeSHDirectional", Access::TextureStorageWrite}); }
    const RenderGraphStageUse compositeUses[] = {{probePositionsName, Access::BufferStorageRead},
        {"probeRadiance", Access::BufferStorageRead}, {"probeDistance", Access::BufferStorageRead},
        {"probeFilterScratch", Access::BufferStorageWrite}, {"compositedProbeDistance", Access::BufferStorageWrite}};
    std::array<EncodedParameters, kSpatialPassCount> filteredParams;
    std::vector<RenderGraphStage> stages;
    const RenderGraphStageUse gridClearUses[]{{"detailGridCells", Access::BufferStorageWrite},
        {"detailGridState", Access::BufferStorageWrite}};
    const RenderGraphStageUse gridDepthUses[]{{"detailDepth", Access::TextureSampleRead},
        {"detailTileDepth", Access::BufferStorageWrite}};
    const RenderGraphStageUse gridScatterUses[]{{"detailTileDepth", Access::BufferStorageRead},
        {"detailGridCells", Access::BufferStorageReadWrite}, {"detailGridPairs", Access::BufferStorageWrite},
        {"detailGridState", Access::BufferStorageReadWrite}};
    const RenderGraphStageUse gridAllocateUses[]{{"detailGridCells", Access::BufferStorageReadWrite},
        {"detailGridState", Access::BufferStorageReadWrite}};
    const RenderGraphStageUse gridCompactUses[]{{"detailGridCells", Access::BufferStorageReadWrite},
        {"detailGridPairs", Access::BufferStorageRead}, {"detailGridState", Access::BufferStorageRead},
        {"detailGridIndices", Access::BufferStorageWrite}};
    if (detailTracing) {
        traceUses.insert(traceUses.end(), {{"detailGridCells", Access::BufferStorageRead},
            {"detailGridIndices", Access::BufferStorageRead}});
        stages.push_back({"Clear Mesh SDF candidate ranges", gridClearUses, [&](CommandBuffer& cmd) {
            return meshGridClear_.dispatch(cmd, meshGridParams, meshCellGroups, 1, 1);
        }});
        if (meshObjectGroups != 0) {
            stages.push_back({"Reduce closest and furthest froxel depth", gridDepthUses, [&](CommandBuffer& cmd) {
                return meshGridDepth_.dispatch(cmd, meshGridParams, params.detail.grid[0], params.detail.grid[1], 1);
            }});
            stages.push_back({"Cull Mesh SDFs to view froxels", gridScatterUses, [&](CommandBuffer& cmd) {
                return meshGridScatter_.dispatch(cmd, meshGridParams, params.detail.grid[0], params.detail.grid[1], meshObjectGroups);
            }});
            stages.push_back({"Allocate Mesh SDF candidate ranges", gridAllocateUses, [&](CommandBuffer& cmd) {
                return meshGridAllocate_.dispatch(cmd, meshGridParams, meshCellGroups, 1, 1);
            }});
            stages.push_back({"Compact Mesh SDF candidates", gridCompactUses, [&](CommandBuffer& cmd) {
                return meshGridCompact_.dispatch(cmd, meshGridParams, std::min(meshCompactGroups, kMeshSDFMaxDispatchGroups),
                    (meshCompactGroups + kMeshSDFMaxDispatchGroups - 1u) / kMeshSDFMaxDispatchGroups, 1);
            }});
        }
    }
    if (finalGather) {
        traceUses.insert(traceUses.end(), {{probePositionsName, Access::BufferStorageRead},
            {"probeDistance", Access::BufferStorageWrite}});
        integrateUses.push_back({probePositionsName, Access::BufferStorageRead});
        LumenScreenProbeFilterParams filterParams{params.position, params.normal, params.probePositions,
            params.probePositions, {}, params.traceDistance, {}, params.width, params.height, params.probeWidth,
            params.probeHeight, params.frameIndex, params.blueNoiseVec2, {}};
        filterParams.rayDirectionFrameIndex = params.rayDirectionFrameIndex;
        if (!radianceCache) {
            auto placement = writer.encode(filterParams, kScreenProbeFilterABI);
            if (!placement) { return makeError(placement.error()); }
            placeParams = std::move(*placement);
            stages.push_back({"Place uniform and adaptive probes", placeUses, [&](CommandBuffer& cmd) {
                return place_.dispatch(cmd, placeParams, params.probeWidth, params.probeHeight, 1);
            }});
        }

        if (jitterDirections) {
            filterParams.inputRadiance = params.traceOutput;
            filterParams.outputRadiance = scratch;
            filterParams.outputDistance = compositedDistance;
            auto composite = writer.encode(filterParams, kScreenProbeFilterABI);
            if (!composite) { return makeError(composite.error()); }
            compositeParams = std::move(*composite);
            filterParams.inputDistance = compositedDistance;
        }
        const std::array inputNames{jitterDirections ? "probeFilterScratch" : "probeRadiance",
            "filteredProbeRadiance", "probeFilterScratch"};
        constexpr std::array outputNames{"filteredProbeRadiance", "probeFilterScratch", "filteredProbeRadiance"};
        const std::array inputs{jitterDirections ? scratch : params.traceOutput, params.traceInput, scratch};
        const std::array outputs{params.traceInput, scratch, params.traceInput};
        for (uint32_t index = 0; index < kSpatialPassCount; ++index) {
            filterUses[index] = {{{"position", Access::TextureSampleRead}, {probePositionsName, Access::BufferStorageRead},
                {jitterDirections ? "compositedProbeDistance" : "probeDistance", Access::BufferStorageRead},
                {inputNames[index], Access::BufferStorageRead},
                {outputNames[index], Access::BufferStorageWrite}}};
            filterParams.inputRadiance = inputs[index];
            filterParams.outputRadiance = outputs[index];
            auto filter = writer.encode(filterParams, kScreenProbeFilterABI);
            if (!filter) { return makeError(filter.error()); }
            filteredParams[index] = std::move(*filter);
        }
        if (jitterDirections) {
            filterParams.inputRadiance = params.traceInput;
            filterParams.irradianceOutput = writer.storageImageHandle(context.outputTexture(irradianceName).view());
            if (sh3) {
                filterParams.shDirectionalOutput = writer.storageImageHandle(context.outputTexture("probeSHDirectional").view());
            }
            auto irradiance = writer.encode(filterParams, kScreenProbeFilterABI);
            if (!irradiance) { return makeError(irradiance.error()); }
            irradianceParams = std::move(*irradiance);
        }
    }
    stages.push_back({"Trace screen probes", traceUses, [&](CommandBuffer& cmd) {
        return trace_.dispatch(cmd, *encoded, params.probeWidth, params.probeHeight, finalGather ? kLumenScreenProbeSlotsPerTile : 1);
    }});
    if (jitterDirections) {
        stages.push_back({"Composite jittered rays into gather directions", compositeUses, [&](CommandBuffer& cmd) {
            return composite_.dispatch(cmd, compositeParams, params.probeWidth, params.probeHeight, kLumenScreenProbeSlotsPerTile);
        }});
    }
    for (uint32_t index = 0; finalGather && index < kSpatialPassCount; ++index) {
        stages.push_back({"Spatially filter probe radiance", filterUses[index], [&, index](CommandBuffer& cmd) {
            return filter_.dispatch(cmd, filteredParams[index], params.probeWidth, params.probeHeight, kLumenScreenProbeSlotsPerTile);
        }});
    }
    if (jitterDirections) {
        stages.push_back({"Convert screen probe radiance to irradiance", irradianceUses, [&](CommandBuffer& cmd) {
            return convertToIrradiance_.dispatch(cmd, irradianceParams, params.probeWidth, params.probeHeight, kLumenScreenProbeSlotsPerTile);
        }});
    }
    stages.push_back({"Integrate and composite diffuse", integrateUses, [&](CommandBuffer& cmd) {
        return integrate_.dispatch(cmd, *encoded, (params.width + 7u) / 8u, (params.height + 7u) / 8u, 1);
    }});
    return context.executeStages(stages);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenScreenProbeGatherPass()
{
    return std::make_unique<LumenScreenProbeGatherPass>();
}

} // namespace metallic::render::builtin_pass
