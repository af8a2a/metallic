#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenMeshSDFGrid.h"
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

constexpr uint64_t kScreenProbeGatherABI = 0x4c554d454e010026ull;
constexpr uint64_t kActiveScreenProbeGatherABI = 0x4c554d454e010027ull;
constexpr uint64_t kActiveScreenProbeFilterABI = 0x4c554d454e010016ull;
constexpr uint64_t kScreenProbeScheduleABI = 0x4c554d454e010017ull;
constexpr uint64_t kScreenProbeClearABI = 0x4c554d454e010018ull;
constexpr uint64_t kScreenProbeRayScheduleABI = 0x4c554d454e010019ull;
constexpr uint64_t kScreenProbeFallbackABI = 0x4c554d454e010028ull;
constexpr uint64_t kCompactScreenProbeScheduleABI = 0x4c554d454e01001bull;
constexpr uint64_t kCompactScreenProbeRayScheduleABI = 0x4c554d454e01001cull;
constexpr uint64_t kCompactScreenProbeGatherABI = 0x4c554d454e010029ull;
constexpr uint64_t kCompactScreenProbeFallbackABI = 0x4c554d454e01002aull;
constexpr uint64_t kCompactScreenProbeFilterABI = 0x4c554d454e01001full;
constexpr uint64_t kScreenProbeImportanceABI = 0x4c554d454e010030ull;
constexpr uint64_t kStructuredImportanceABIBit = 0x0000000000800000ull;
constexpr uint64_t kFastLightingUpdateABIBit = 0x0000000000400000ull;
constexpr uint64_t kRoughSpecularABIBit = 0x0000000000200000ull;
constexpr const char* kStructuredImportanceDefine = "METALLIC_LUMEN_STRUCTURED_IMPORTANCE_SAMPLING";
constexpr const char* kFastLightingUpdateDefine = "METALLIC_LUMEN_FAST_LIGHTING_UPDATE";
constexpr uint32_t kProbeCompactionThreads = 64;
constexpr uint32_t kProbeDispatchArgumentBytes = 3 * sizeof(uint32_t);
constexpr uint32_t kGatherSamplesPerProbe = 8 * 8;
constexpr uint32_t kTraceStride = 4 * sizeof(float);
constexpr uint32_t kScreenTraceViewStride = 224;
constexpr uint32_t kRadianceCacheClipmapStride = 48;
constexpr uint32_t kSpatialPassCount = 3;
constexpr float kScreenTraceRelativeDepthThickness = 0.02f; // UE 5.7.4 HZBTraversal.RelativeDepthThickness.
constexpr uint32_t kIrradianceProbeWithBorderRes = 8;
constexpr uint32_t kProbeSlotsPerTileAxis = 3;
static_assert(kProbeSlotsPerTileAxis * kProbeSlotsPerTileAxis == kLumenScreenProbeSlotsPerTile);
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
    ShaderLumenMeshSDFGridBindings detail;
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
    ShaderSampledImage jitterDepth;
    ShaderBuffer jitterView, jitterOffsets;
    float fullResolutionJitterWidth;
    ShaderStorageImage indirectIrradiance;
    uint32_t writeIndirectIrradiance;
};
static_assert(sizeof(LumenScreenProbeGatherParams) == 296 && offsetof(LumenScreenProbeGatherParams, detail) == 140 &&
    offsetof(LumenScreenProbeGatherParams, traceReadback) == 208 &&
    offsetof(LumenScreenProbeGatherParams, interpolationReadback) == 228 &&
    offsetof(LumenScreenProbeGatherParams, blueNoiseVec2) == 248);

template <typename Params>
struct LumenActiveScreenProbeParams {
    Params base;
    ShaderBuffer workList, workState;
};
static_assert(sizeof(LumenActiveScreenProbeParams<LumenScreenProbeGatherParams>) == 304);
static_assert(sizeof(LumenActiveScreenProbeParams<LumenScreenProbeFilterParams>) == 76);

struct LumenScreenProbeScheduleParams {
    ShaderBuffer probePositions, workList, workState, indirectArgs;
    uint32_t capacity;
};
static_assert(sizeof(LumenScreenProbeScheduleParams) == 20);

struct LumenScreenProbeRayScheduleParams {
    LumenScreenProbeScheduleParams probes;
    ShaderBuffer traceRadiance, rayWorkList, rayWorkCount, rayIndirectArgs;
};
static_assert(sizeof(LumenScreenProbeRayScheduleParams) == 36);

struct LumenScreenProbeFallbackParams {
    LumenActiveScreenProbeParams<LumenScreenProbeGatherParams> gather;
    ShaderBuffer rayWorkList, rayWorkCount;
};
static_assert(sizeof(LumenScreenProbeFallbackParams) == 312);

template <typename Params>
struct CompactScreenProbeScheduleParams {
    Params base;
    ShaderBuffer adaptiveCount;
    uint32_t uniformCount;
};

struct CompactProbeAtlasBindings {
    ShaderBuffer origins, tileCounts, tileIndices;
};

template <typename Params>
struct CompactScreenProbeGatherParams {
    Params base;
    CompactProbeAtlasBindings atlas;
};

struct CompactScreenProbeFilterParams {
    LumenActiveScreenProbeParams<LumenScreenProbeFilterParams> base;
    ShaderBuffer probeOrigins;
};

template <typename Params>
struct StructuredImportanceParams {
    Params base;
    ShaderBuffer structuredRayInfo;
};

struct ProbeTraceMotionBindings {
    ShaderSampledImage previousPosition;
    ShaderBuffer probeTraceMoving, probeMoving, detailMotion;
};
static_assert(sizeof(ProbeTraceMotionBindings) == 16);

struct ProbeFilterMotionBindings {
    ShaderBuffer probeTraceMoving, compositedProbeMoving, probeMoving;
};
static_assert(sizeof(ProbeFilterMotionBindings) == 12);

struct ProbeRoughSpecularBindings {
    ShaderSampledImage material;
    ShaderStorageImage lighting;
    uint32_t writeLighting;
};
static_assert(sizeof(ProbeRoughSpecularBindings) == 12);

template <typename Params, typename Motion>
struct ScreenProbeMotionParams {
    Params base;
    Motion motion;
};

template <typename Params>
struct ScreenProbeRoughSpecularParams {
    Params base;
    ProbeRoughSpecularBindings roughSpecular;
};

struct LumenScreenProbeImportanceParams {
    ShaderSampledImage position, normal, blueNoiseVec2;
    ShaderBuffer probePositions, probeWorkList, probeWorkState, view, cacheIndirection, cacheClipmaps;
    ShaderSampledImage cacheRadiance;
    ShaderBuffer brdfPdf, lightingPdf, rayInfo, filteredRadiance, previousRadiance, previousGeometry, savedRadiance, savedGeometry;
    uint32_t width, height, probeWidth, probeHeight, frameIndex, rayDirectionFrameIndex, historyValid;
    ShaderSampler cacheSampler;
};
static_assert(sizeof(LumenScreenProbeImportanceParams) == 104 && offsetof(LumenScreenProbeImportanceParams, width) == 72);
using ActiveGatherParams = LumenActiveScreenProbeParams<LumenScreenProbeGatherParams>;
static_assert(sizeof(CompactScreenProbeScheduleParams<LumenScreenProbeScheduleParams>) == 28);
static_assert(sizeof(CompactScreenProbeScheduleParams<LumenScreenProbeRayScheduleParams>) == 44);
static_assert(sizeof(CompactScreenProbeGatherParams<ActiveGatherParams>) == 316);
static_assert(sizeof(CompactScreenProbeGatherParams<LumenScreenProbeFallbackParams>) == 324);
static_assert(sizeof(CompactScreenProbeFilterParams) == 80);

struct LumenScreenProbeClearParams {
    ShaderBuffer workList, workState, traceRadiance, traceDistance, filteredRadiance, scratch, compositedDistance;
    ShaderStorageImage irradiance, shDirectional;
    ShaderBuffer traceReadback;
    ProbeReadbackRectangle traceReadbackTiles;
    uint32_t probeWidth, probeHeight;
};
static_assert(sizeof(LumenScreenProbeClearParams) == 64);

class LumenScreenProbeGatherPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override;
    Result<> prepare(const RenderGraphCompileContext& context, std::string& log) override
    {
        // A resource rebuild can reuse this pass while changing its trace topology.
        const auto requested = properties().value("tracingOctahedronResolution", RenderGraphProperties(8));
        const auto active = properties().value("activeProbeDispatch", RenderGraphProperties(false));
        const auto split = properties().value("splitProbeTracing", RenderGraphProperties(false));
        const auto compact = properties().value("compactProbeAtlas", RenderGraphProperties(false));
        const auto cache = properties().value("radianceCache", RenderGraphProperties(false));
        const auto fraction = lumenAdaptiveProbeFraction(properties());
        const auto jitterWidth = properties().value("fullResolutionJitterWidth", RenderGraphProperties(0.0f));
        const auto compareJitter = properties().value("probeJitterComparison", RenderGraphProperties(false));
        const auto exportIrradiance = properties().value("exportIndirectIrradiance", RenderGraphProperties(false));
        const auto structured = properties().value("structuredImportanceSampling", RenderGraphProperties(false));
        const auto fastLightingUpdate = properties().value("fastLightingUpdate", RenderGraphProperties(false));
        const auto roughSpecular = properties().value("roughSpecular", RenderGraphProperties(false));
        if (compiledTraceResolution_ && (!requested.is_number_integer() || requested != compiledTraceResolution_ ||
                !active.is_boolean() || active != compiledActiveProbeDispatch_ ||
                !split.is_boolean() || split != compiledSplitProbeTracing_ ||
                !compact.is_boolean() || compact != compiledCompactProbeAtlas_ ||
                !cache.is_boolean() || cache != compiledRadianceCache_ ||
                fraction != compiledAdaptiveProbeAllocationFraction_ || !jitterWidth.is_number() ||
                jitterWidth.get<float>() != fullResolutionJitterWidth_ || !compareJitter.is_boolean() ||
                compareJitter != compiledProbeJitterComparison_ || !exportIrradiance.is_boolean() ||
                exportIrradiance != compiledExportIndirectIrradiance_ || !structured.is_boolean() ||
                (structured == true && requested == 8) != compiledStructuredImportance_ ||
                !fastLightingUpdate.is_boolean() || fastLightingUpdate != compiledFastLightingUpdate_ ||
                !roughSpecular.is_boolean() || roughSpecular != compiledRoughSpecular_)) {
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
            .invalidateHistory = true, .rebuildGraph = true},
            {.key = "structuredImportanceSampling", .label = "Structured Importance Sampling (64 Rays)",
                .type = RenderGraphRuntimeSettingType::Bool, .defaultValue = false,
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
    ComputeKernel reduceProbeMotion_;
    ComputeKernel filter_;
    ComputeKernel convertToIrradiance_;
    ComputeKernel brdfPdf_, lightingPdf_, structuredRays_, saveProbeHistory_;
    ComputeKernel resetProbeSchedule_, compactProbes_, buildProbeDispatch_, clearInactiveProbes_;
    ComputeKernel compactProbeRays_, buildRayDispatch_, traceFallback_;
    ComputeKernel compactDetailRays_, buildDetailDispatch_, traceDetail_;
    LumenCompactProbePlacement placement_;
    ComputeKernel clearCompactReadback_;
    std::unique_ptr<TextureView> blueNoise_;
    MeshDistanceFieldGPU meshFields_;
    LumenMeshSDFGrid meshGrid_;
    uint32_t compiledTraceResolution_ = 0;
    bool compiledActiveProbeDispatch_ = false;
    bool compiledSplitProbeTracing_ = false;
    bool compiledCompactProbeAtlas_ = false;
    bool compiledRadianceCache_ = false;
    bool compiledProbeJitterComparison_ = false;
    bool compiledExportIndirectIrradiance_ = false;
    bool compiledStructuredImportance_ = false;
    bool compiledFastLightingUpdate_ = false;
    bool compiledRoughSpecular_ = false;
    float fullResolutionJitterWidth_ = 0.0f;
    double compiledAdaptiveProbeAllocationFraction_ = kLumenAdaptiveProbeAllocationFraction;
    uint32_t maxRayDirections_ = 8;
    float maxTraceDistance_ = 4.0f;
    float screenTraceRelativeDepthThickness_ = kScreenTraceRelativeDepthThickness;
    float meshSDFTraceDistance_ = kLumenDefaultMeshSDFTraceDistance;
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
    if (optionEnabled("roughSpecular")) {
        reflection.addTextureInput("specular", "Receiver integrated specular response and perceptual roughness")
            .sampledRead().format = Format::RGBA16Sfloat;
        auto& rough = reflection.addTextureOutput("roughSpecular", "Unmodulated rough specular radiance from screen probes")
            .storageWrite();
        rough.format = Format::RGBA16Sfloat;
        rough.colorEncoding = DisplayColorEncoding::SceneLinear;
    }
    reflection.addTextureInput("directColor", "Unexposed deferred scene color before indirect diffuse")
        .sampledRead().format = Format::RGBA32Sfloat;
    const auto jitterWidth = properties().find("fullResolutionJitterWidth");
    if (jitterWidth != properties().end() && jitterWidth->is_number() && *jitterWidth > 0.0f) {
        reflection.addTextureInput("jitterDepth", "Current reversed-Z depth for full-resolution probe interpolation")
            .sampledRead().format = Format::D32Sfloat;
    }
    const bool compareJitter = optionEnabled("probeJitterComparison");

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
        LumenMeshSDFGrid::reflect(reflection, context);
    }

    const uint32_t probeWidth = lumenScreenProbeTiles(context.width);
    const uint32_t probeHeight = lumenScreenProbeTiles(context.height);
    const uint32_t probeCount = probeWidth * probeHeight;
    const bool finalGather = optionEnabled("finalGather");
    const bool compactAtlas = finalGather && optionEnabled("activeProbeDispatch") && optionEnabled("compactProbeAtlas");
    const uint64_t slots = compactAtlas ? probeCount + lumenAdaptiveProbeCapacity(properties(), probeCount) :
        uint64_t(probeCount) * (finalGather ? kLumenScreenProbeSlotsPerTile : 1u);
    const uint64_t probePositionsBytes = slots * 2 * sizeof(uint32_t);
    const bool radianceCache = optionEnabled("radianceCache");
    if (radianceCache) {
        if (!compactAtlas) {
            reflection.addBufferInput("placedProbes", "Uniform and adaptive receiver pixels used to mark the World Radiance Cache")
                .buffer(probePositionsBytes, 2 * sizeof(uint32_t)).storageRead();
        }
        reflection.addBufferInput("cacheIndirection", "World Radiance Cache clipmap cell-to-probe indices")
            .buffer(0, sizeof(uint32_t)).storageRead();
        reflection.addBufferInput("cacheClipmaps", "World Radiance Cache clipmap geometry and atlas layout")
            .buffer(0, kRadianceCacheClipmapStride).storageRead();
        auto& cacheRadiance = reflection.addTextureInput("cacheRadiance", "Bordered directional World Radiance Cache atlas")
            .sampledRead();
        cacheRadiance.format = Format::RGBA16Sfloat;
        cacheRadiance.matchOutputExtent = false;
    }
    const uint64_t rayCount = slots * traceResolution() * traceResolution();
    const uint64_t gatherCount = slots * kGatherSamplesPerProbe;
    if (finalGather && !optionEnabled("fixedProbeDirections") && optionEnabled("fastLightingUpdate")) {
        reflection.addTextureInput("previousPosition", "Previous world position of each current receiver")
            .sampledRead().format = Format::RGBA32Sfloat;
        reflection.addBufferOutput("probeTraceMoving", "Relative hit motion flag in original ray order")
            .buffer(rayCount * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite().bufferLayout("u32");
        reflection.addBufferOutput("compositedProbeMoving", "Quantized relative motion in gather direction order")
            .buffer(gatherCount * sizeof(float), sizeof(float)).storageReadWrite().bufferLayout("f32");
        reflection.addBufferOutput("probeMoving", "Fraction of moving directions per screen probe")
            .buffer(slots * sizeof(float), sizeof(float)).storageReadWrite().bufferLayout("f32");
    }
    if (finalGather && optionEnabled("structuredImportanceSampling") && traceResolution() == 8u) {
        reflection.addBufferOutput("probeBrdfPdfSH", "Directional BRDF PDF with nine SH3 coefficients per probe")
            .buffer(slots * 9u * sizeof(float), sizeof(float)).storageReadWrite().bufferLayout("f32");
        reflection.addBufferOutput("probeLightingPdf", "Normalized incoming-lighting PDF in trace tile order")
            .buffer(rayCount * sizeof(float), sizeof(float)).storageReadWrite().bufferLayout("f32");
        reflection.addBufferOutput("structuredRayInfo", "UE packed octahedral coordinate and mip per trace lane")
            .buffer(rayCount * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite().bufferLayout("u32");
        const uint64_t uniformRadianceBytes = uint64_t(probeCount) * kGatherSamplesPerProbe * kTraceStride;
        const uint64_t uniformGeometryBytes = uint64_t(probeCount) * kTraceStride;
        reflection.addBufferInput("previousImportanceRadiance", "Previous filtered radiance of the uniform probe grid")
            .buffer(uniformRadianceBytes, kTraceStride).storageRead().temporalRead("importanceRadiance", -1, true);
        reflection.addBufferInput("previousImportanceGeometry", "Previous world position and depth of uniform screen probes")
            .buffer(uniformGeometryBytes, kTraceStride).storageRead().temporalRead("importanceGeometry", -1, true);
        reflection.addBufferOutput("savedImportanceRadiance", "Filtered uniform probe radiance for next frame's sampling")
            .buffer(uniformRadianceBytes, kTraceStride).storageWrite().temporalWrite("importanceRadiance", true);
        reflection.addBufferOutput("savedImportanceGeometry", "Uniform probe world position and depth for next frame")
            .buffer(uniformGeometryBytes, kTraceStride).storageWrite().temporalWrite("importanceGeometry", true);
    }
    const auto initializeLegacyAllocation = [&](auto& field) {
        if (!compactAtlas) { field.transient(RenderGraphInitialization::FullOverwrite); }
    };
    initializeLegacyAllocation(reflection.addBufferOutput("probeRadiance", "Probe ray RGB and trace classification in atlas order")
        .buffer(rayCount * kTraceStride, kTraceStride).storageReadWrite());
    if (finalGather) {
        if (optionEnabled("activeProbeDispatch")) {
            reflection.addBufferOutput("probeWorkList", "Compacted original indices for active and inactive probe slots")
                .buffer(slots * (compactAtlas ? 1u : 2u) * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite();
            reflection.addBufferOutput("probeWorkState", "Active and inactive probe counts")
                .buffer(2 * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite()
                .transient(RenderGraphInitialization::FullOverwrite);
            if (optionEnabled("splitProbeTracing")) {
                const bool detailTracing = optionEnabled("detailTracing");
                reflection.addBufferOutput("probeRayWorkList", "Original atlas addresses of unfinished screen probe rays")
                    .buffer(rayCount * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite();
                reflection.addBufferOutput("probeRayWorkCount", "Unfinished probe ray counts for GDF and optional Detail Tracing")
                    .buffer((detailTracing ? 2u : 1u) * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite()
                    .transient(RenderGraphInitialization::FullOverwrite);
                reflection.addBufferOutput("probeRayIndirectArgs", "Dispatch commands for ray compaction and software fallback")
                    .buffer((detailTracing ? 3u : 2u) * kProbeDispatchArgumentBytes, sizeof(uint32_t)).storageWrite()
                    .stageAccess(RenderGraphResourceAccess::BufferIndirectRead)
                    .transient(RenderGraphInitialization::FullOverwrite);
            }
            reflection.addBufferOutput("probeIndirectArgs", "Dispatch commands for active and inactive probe slots")
                .buffer(2 * kProbeDispatchArgumentBytes, sizeof(uint32_t)).storageWrite()
                .stageAccess(RenderGraphResourceAccess::BufferIndirectRead)
                .transient(RenderGraphInitialization::FullOverwrite);
        }
        if (compactAtlas) {
            LumenCompactProbePlacement::reflect(reflection, probeCount, slots, radianceCache);
        } else if (!radianceCache) {
            initializeLegacyAllocation(reflection.addBufferOutput("probePositions", "Uniform and adaptive receiver pixels in atlas order")
                .buffer(probePositionsBytes, 2 * sizeof(uint32_t)).storageReadWrite());
        }
        initializeLegacyAllocation(reflection.addBufferOutput("probeDistance", "Per-direction trace distance in meters")
            .buffer(rayCount * sizeof(float), sizeof(float)).storageReadWrite());
        if (!optionEnabled("fixedProbeDirections")) {
            initializeLegacyAllocation(reflection.addBufferOutput("compositedProbeDistance", "Minimum trace distance per gather direction in meters")
                .buffer(gatherCount * sizeof(float), sizeof(float)).storageReadWrite());
            const uint32_t atlasWidth = compactAtlas ? probeWidth : probeWidth * kProbeSlotsPerTileAxis;
            const uint32_t atlasHeight = compactAtlas ? uint32_t((slots + probeWidth - 1u) / probeWidth) :
                probeHeight * kProbeSlotsPerTileAxis;
            const auto addAtlas = [&](const char* name, const char* description, uint32_t width, uint32_t height, Format format) {
                auto& atlas = reflection.addTextureOutput(name, description).texture2D(width, height)
                    .storageWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead);
                initializeLegacyAllocation(atlas);
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
                auto& field = reflection.addBufferOutput("probeTraceReadback", "Screen, Detail, GDF and WRC state per selected probe ray")
                    .buffer(probeReadbackElementCount(*tiles) * traceReadbackBytesPerTile(),
                        sizeof(LumenProbeTraceReadbackRecord))
                    .transient(RenderGraphInitialization::FullOverwrite)
                    .bufferLayout(kLumenProbeTraceReadbackLayout);
                if (optionEnabled("splitProbeTracing")) { field.storageReadWrite(); }
                else { field.storageWrite(); }
            }
        }
        if (const auto readback = properties().find("probeInterpolationReadback"); readback != properties().end()) {
            if (const auto pixels = readProbeReadbackRectangle(*readback, kProbeInterpolationReadbackStride)) {
                reflection.addBufferOutput("probeInterpolationReadback", "Final probe indices and FP32 weights per selected pixel")
                    .buffer(probeReadbackElementCount(*pixels) * kProbeInterpolationReadbackStride, kProbeInterpolationReadbackStride)
                    .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
                if (compareJitter) {
                    reflection.addBufferOutput("jitterComparisonInterpolationReadback", "Probe indices and weights after interpolation jitter")
                        .buffer(probeReadbackElementCount(*pixels) * kProbeInterpolationReadbackStride, kProbeInterpolationReadbackStride)
                        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
                    reflection.addBufferOutput("probeJitterOffsets", "Proposed XY and accepted XY interpolation offsets in pixels")
                        .buffer(probeReadbackElementCount(*pixels) * 4 * sizeof(float), 4 * sizeof(float))
                        .storageWrite().transient(RenderGraphInitialization::FullOverwrite).bufferLayout("RGBA32F");
                }
            }
        }
        initializeLegacyAllocation(reflection.addBufferOutput("filteredProbeRadiance", "Spatially filtered probe ray RGB")
            .buffer(gatherCount * kTraceStride, kTraceStride).storageReadWrite());
        initializeLegacyAllocation(reflection.addBufferOutput("probeFilterScratch", "Spatial filtering ping-pong buffer")
            .buffer(gatherCount * kTraceStride, kTraceStride).storageReadWrite());
    }

    auto& indirect = reflection.addTextureOutput("indirectDiffuse", "Unexposed indirect diffuse after receiver response")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    indirect.format = Format::RGBA16Sfloat;
    indirect.colorEncoding = DisplayColorEncoding::SceneLinear;
    if (optionEnabled("exportIndirectIrradiance")) {
        auto& irradiance = reflection.addTextureOutput("indirectIrradiance", "Unexposed indirect irradiance over pi before material response")
            .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        irradiance.format = Format::RGBA16Sfloat;
        irradiance.colorEncoding = DisplayColorEncoding::SceneLinear;
    }
    auto& color = reflection.addTextureOutput("color", "Unexposed scene color with indirect diffuse")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    color.format = Format::RGBA32Sfloat;
    color.colorEncoding = DisplayColorEncoding::SceneLinear;
    if (compareJitter) {
        auto& comparisonDiffuse = reflection.addTextureOutput("jitterComparisonIndirectDiffuse", "Indirect diffuse with interpolation jitter")
            .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        comparisonDiffuse.format = Format::RGBA16Sfloat;
        comparisonDiffuse.colorEncoding = DisplayColorEncoding::SceneLinear;
        auto& comparisonColor = reflection.addTextureOutput("jitterComparisonColor", "Unexposed scene color with interpolation jitter")
            .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        comparisonColor.format = Format::RGBA32Sfloat;
        comparisonColor.colorEncoding = DisplayColorEncoding::SceneLinear;
    }
    return reflection;
}

Result<> LumenScreenProbeGatherPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    for (std::string_view name : {"finalGather", "screenTraces", "stochasticInterpolation", "radianceCache",
            "detailTracing", "screenGDFContinuation", "gdfPrefixValidation", "gdfPrefixRejectReceding",
            "gdfPrefixMeshValidation", "fixedProbeDirections", "activeProbeDispatch", "splitProbeTracing", "compactProbeAtlas",
            "probeJitterComparison", "exportIndirectIrradiance", "fastLightingUpdate", "roughSpecular"}) {
        const auto value = properties().find(name);
        if (value != properties().end() && !value->is_boolean()) {
            log += "Lumen Screen Probe Gather requires a boolean " + std::string(name);
            return makeError(Error::InvalidArgument);
        }
    }
    const bool finalGather = optionEnabled("finalGather");
    compiledExportIndirectIrradiance_ = optionEnabled("exportIndirectIrradiance");
    if (compiledExportIndirectIrradiance_ && !finalGather) {
        log = "Lumen exportIndirectIrradiance requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    const auto jitterWidth = properties().value("fullResolutionJitterWidth", RenderGraphProperties(0.0f));
    if (!jitterWidth.is_number() || !std::isfinite(jitterWidth.get<float>()) || jitterWidth < 0.0f ||
        (jitterWidth > 0.0f && !finalGather)) {
        log = "Lumen fullResolutionJitterWidth requires a finite nonnegative width and Final Gather when enabled";
        return makeError(Error::InvalidArgument);
    }
    fullResolutionJitterWidth_ = jitterWidth.get<float>();
    compiledProbeJitterComparison_ = optionEnabled("probeJitterComparison");
    if (compiledProbeJitterComparison_ && fullResolutionJitterWidth_ == 0.0f) {
        log = "Lumen probeJitterComparison requires a positive fullResolutionJitterWidth";
        return makeError(Error::InvalidArgument);
    }
    const bool activeProbeDispatch = optionEnabled("activeProbeDispatch");
    if (activeProbeDispatch && !finalGather) {
        log = "Lumen activeProbeDispatch requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    const bool jitterDirections = finalGather && !optionEnabled("fixedProbeDirections");
    compiledFastLightingUpdate_ = optionEnabled("fastLightingUpdate");
    if (compiledFastLightingUpdate_ && !jitterDirections) {
        log = "Lumen fastLightingUpdate requires jittered Final Gather";
        return makeError(Error::InvalidArgument);
    }
    compiledRoughSpecular_ = optionEnabled("roughSpecular");
    if (compiledRoughSpecular_ && !jitterDirections) {
        log = "Lumen roughSpecular requires jittered Final Gather radiance";
        return makeError(Error::InvalidArgument);
    }
    const auto structured = properties().value("structuredImportanceSampling", RenderGraphProperties(false));
    if (!structured.is_boolean() || (structured == true && !jitterDirections)) {
        log = "Lumen structuredImportanceSampling requires jittered Final Gather";
        return makeError(Error::InvalidArgument);
    }
    compiledStructuredImportance_ = structured == true && traceResolution() == 8u;
    const bool screenTraces = optionEnabled("screenTraces");
    const bool splitProbeTracing = optionEnabled("splitProbeTracing");
    if (splitProbeTracing && (!activeProbeDispatch || !screenTraces)) {
        log = "Lumen splitProbeTracing requires activeProbeDispatch and screenTraces";
        return makeError(Error::InvalidArgument);
    }
    const bool stochasticInterpolation = optionEnabled("stochasticInterpolation");
    const bool radianceCache = optionEnabled("radianceCache");
    const bool detailTracing = optionEnabled("detailTracing");
    const bool compactAtlas = optionEnabled("compactProbeAtlas");
    if (compactAtlas && !activeProbeDispatch) {
        log = "Lumen compactProbeAtlas requires activeProbeDispatch";
        return makeError(Error::InvalidArgument);
    }
    const auto fraction = lumenAdaptiveProbeFraction(properties());
    if (!fraction || (!compactAtlas && properties().contains("adaptiveProbeAllocationFraction"))) {
        log = "Lumen adaptiveProbeAllocationFraction requires compactProbeAtlas and a value from 0 to 8";
        return makeError(Error::InvalidArgument);
    }
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
    compiledActiveProbeDispatch_ = activeProbeDispatch;
    compiledSplitProbeTracing_ = splitProbeTracing;
    compiledCompactProbeAtlas_ = compactAtlas;
    compiledRadianceCache_ = radianceCache;
    compiledAdaptiveProbeAllocationFraction_ = *fraction;
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
    if (activeProbeDispatch) { defines.push_back({"METALLIC_LUMEN_ACTIVE_PROBES", "1"}); }
    if (compactAtlas) { defines.push_back({"METALLIC_LUMEN_COMPACT_PROBE_ATLAS", "1"}); }
    if (jitterDirections || fullResolutionJitterWidth_ > 0.0f) {
        if (!context.graphicsQueue) {
            log = "Lumen Screen Probe Gather requires a graphics queue to upload blue noise";
            return makeError(Error::InvalidArgument);
        }
        auto blueNoise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Vec2, log);
        if (!blueNoise) { return makeError(blueNoise.error()); }
        blueNoise_ = std::move(*blueNoise);
    }
    if (jitterDirections) { defines.push_back({"METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS", "1"}); }
    if (compiledStructuredImportance_) { defines.push_back({kStructuredImportanceDefine, "1"}); }
    if (compiledFastLightingUpdate_) { defines.push_back({kFastLightingUpdateDefine, "1"}); }
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
        meshSDFTraceDistance_ = kLumenDefaultMeshSDFTraceDistance;
        if (const auto traceDistance = properties().find("meshSDFTraceDistance"); traceDistance != properties().end()) {
            if (!traceDistance->is_number() || !std::isfinite(traceDistance->get<float>()) || traceDistance->get<float>() <= 0.0001f) {
                log = "Lumen Detail Tracing meshSDFTraceDistance must be finite and greater than 0.0001 meters";
                return makeError(Error::InvalidArgument);
            }
            meshSDFTraceDistance_ = traceDistance->get<float>();
        }
        meshSDFTraceDistance_ = std::min(meshSDFTraceDistance_, maxTraceDistance_);
        if (auto result = meshGrid_.compile(*device_, log); !result) { return result; }
    }
    const auto probeABI = [&]<typename Params, typename Motion = ProbeTraceMotionBindings>(uint64_t key, bool rough = false) {
        const auto integrateABI = [&]<typename Base>(uint64_t baseKey) {
            return rough ? parameterAbi<ScreenProbeRoughSpecularParams<Base>>(baseKey | kRoughSpecularABIBit) :
                parameterAbi<Base>(baseKey);
        };
        const auto motionABI = [&]<typename Base>(uint64_t baseKey) {
            return compiledFastLightingUpdate_ ?
                integrateABI.operator()<ScreenProbeMotionParams<Base, Motion>>(baseKey | kFastLightingUpdateABIBit) :
                integrateABI.operator()<Base>(baseKey);
        };
        return compiledStructuredImportance_ ?
            motionABI.operator()<StructuredImportanceParams<Params>>(key | kStructuredImportanceABIBit) : motionABI.operator()<Params>(key);
    };
    const auto gatherABIFor = [&](bool rough) {
        if (compactAtlas) {
            return probeABI.operator()<CompactScreenProbeGatherParams<ActiveGatherParams>>(kCompactScreenProbeGatherABI, rough);
        }
        return activeProbeDispatch ? probeABI.operator()<ActiveGatherParams>(kActiveScreenProbeGatherABI, rough) :
            probeABI.operator()<LumenScreenProbeGatherParams>(kScreenProbeGatherABI, rough);
    };
    const auto gatherABI = gatherABIFor(false);
    const auto createGather = [&](const char* entry, const char* name, ComputeKernel& kernel, ParameterABI abi) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenScreenProbeGather", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .macroDefines = defines},
            {.parameters = abi, .debugName = name}, kernel, log);
    };
    if (splitProbeTracing) {
        if (detailTracing) { defines.push_back({"METALLIC_LUMEN_SPLIT_DETAIL_TRACING", "1"}); }
        const auto fallbackABI = compactAtlas ?
            probeABI.operator()<CompactScreenProbeGatherParams<LumenScreenProbeFallbackParams>>(kCompactScreenProbeFallbackABI) :
            probeABI.operator()<LumenScreenProbeFallbackParams>(kScreenProbeFallbackABI);
        defines.push_back({"METALLIC_LUMEN_PROBE_TRACE_SCREEN_ONLY", "1"});
        if (auto result = createGather("lumenTraceScreenProbesMain", "LumenTraceScreenProbeHZB", trace_, gatherABI); !result) {
            return result;
        }
        defines.back() = {"METALLIC_LUMEN_PROBE_TRACE_FALLBACK_ONLY", "1"};
        if (auto result = createGather("lumenTraceScreenProbeFallbackMain", "LumenTraceScreenProbeFallback", traceFallback_,
                fallbackABI); !result) {
            return result;
        }
        if (detailTracing) {
            defines.back() = {"METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY", "1"};
            if (auto result = createGather("lumenTraceScreenProbeDetailMain", "LumenTraceScreenProbeDetail", traceDetail_,
                    fallbackABI); !result) {
                return result;
            }
            defines.pop_back();
        }
        defines.pop_back();
    } else if (auto result = createGather("lumenTraceScreenProbesMain", "LumenTraceScreenProbes", trace_, gatherABI); !result) {
        return result;
    }
    if (jitterDirections && useSH3()) { defines.push_back({"METALLIC_LUMEN_IRRADIANCE_SH3", "1"}); }
    if (interpolationReadbackPixels_) { defines.push_back({"METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK", "1"}); }
    if (fullResolutionJitterWidth_ > 0.0f) { defines.push_back({"METALLIC_LUMEN_FULL_RESOLUTION_JITTER", "1"}); }
    if (compiledProbeJitterComparison_ && interpolationReadbackPixels_) {
        defines.push_back({"METALLIC_LUMEN_PROBE_JITTER_READBACK", "1"});
    }
    if (compiledExportIndirectIrradiance_) { defines.push_back({"METALLIC_LUMEN_EXPORT_INDIRECT_IRRADIANCE", "1"}); }
    if (compiledRoughSpecular_) { defines.push_back({"METALLIC_LUMEN_ROUGH_SPECULAR", "1"}); }
    const auto integration = createGather("lumenIntegrateScreenProbesMain", "LumenIntegrateScreenProbes", integrate_,
        gatherABIFor(compiledRoughSpecular_));
    if (compiledRoughSpecular_) { defines.pop_back(); }
    if (!integration || !finalGather) { return integration; }
    if (compactAtlas) {
        if (!radianceCache) {
            if (auto result = placement_.compile(*device_, log); !result) { return result; }
        }
        if (traceReadbackTiles_) {
            if (auto result = createGather("lumenClearCompactProbeTraceReadbackMain", "LumenClearCompactProbeTraceReadback",
                    clearCompactReadback_, gatherABI); !result) { return result; }
        }
    }

    if (activeProbeDispatch) {
        std::vector<std::pair<const char*, ComputeKernel*>> scheduleKernels{
            {"lumenResetScreenProbeScheduleMain", &resetProbeSchedule_},
            {"lumenCompactScreenProbesMain", &compactProbes_},
            {"lumenBuildScreenProbeIndirectArgsMain", &buildProbeDispatch_},
        };
        std::vector<SlangMacroDefine> scheduleDefines;
        if (compactAtlas) { scheduleDefines.push_back({"METALLIC_LUMEN_COMPACT_PROBE_ATLAS", "1"}); }
        const auto& caps = device_->capabilities();
        if (caps.computeSubgroupBallotArithmetic && (!splitProbeTracing || (caps.subgroupSize == 32u && caps.computeSubgroupShuffle))) {
            scheduleDefines.push_back({"METALLIC_LUMEN_PROBE_COMPACTION_WAVE_OPS", "1"});
        }
        if (splitProbeTracing) {
            scheduleDefines.push_back({"METALLIC_LUMEN_SPLIT_PROBE_TRACING", "1"});
            if (detailTracing) { scheduleDefines.push_back({"METALLIC_LUMEN_SPLIT_DETAIL_TRACING", "1"}); }
            if (compiledTraceResolution_ == 16u) { scheduleDefines.push_back({"METALLIC_LUMEN_TRACE_RESOLUTION_16", "1"}); }
            scheduleKernels.emplace_back("lumenCompactScreenProbeRaysMain", &compactProbeRays_);
            scheduleKernels.emplace_back("lumenBuildScreenProbeRayIndirectArgsMain", &buildRayDispatch_);
        }
        auto scheduleABI = splitProbeTracing ? parameterAbi<LumenScreenProbeRayScheduleParams>(kScreenProbeRayScheduleABI) :
            parameterAbi<LumenScreenProbeScheduleParams>(kScreenProbeScheduleABI);
        if (compactAtlas) {
            scheduleABI = splitProbeTracing ?
                parameterAbi<CompactScreenProbeScheduleParams<LumenScreenProbeRayScheduleParams>>(kCompactScreenProbeRayScheduleABI) :
                parameterAbi<CompactScreenProbeScheduleParams<LumenScreenProbeScheduleParams>>(kCompactScreenProbeScheduleABI);
        }
        const auto createSchedule = [&](const char* entry, const char* name, ComputeKernel& kernel) {
            return ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/LumenScreenProbeSchedule", .entryPointName = entry,
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = scheduleDefines},
                {.parameters = scheduleABI, .debugName = name}, kernel, log);
        };
        for (const auto& [entry, kernel] : scheduleKernels) {
            if (auto result = createSchedule(entry, entry, *kernel); !result) { return result; }
        }
        if (splitProbeTracing && detailTracing) {
            scheduleDefines.push_back({"METALLIC_LUMEN_COMPACT_DETAIL_RAYS", "1"});
            if (auto result = createSchedule("lumenCompactScreenProbeRaysMain", "LumenCompactDetailRays", compactDetailRays_);
                !result) { return result; }
            if (auto result = createSchedule("lumenBuildScreenProbeRayIndirectArgsMain", "LumenBuildDetailDispatch", buildDetailDispatch_);
                !result) { return result; }
        }
        if (!compactAtlas) {
            if (auto result = ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/LumenScreenProbeGather",
                    .entryPointName = "lumenClearInactiveScreenProbesMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                    .macroDefines = defines},
                {.parameters = parameterAbi<LumenScreenProbeClearParams>(kScreenProbeClearABI),
                    .debugName = "LumenClearInactiveScreenProbes"}, clearInactiveProbes_, log); !result) {
                return result;
            }
        }
    }

    std::vector<SlangMacroDefine> filterDefines;
    if (compiledFastLightingUpdate_) { filterDefines.push_back({kFastLightingUpdateDefine, "1"}); }
    if (compiledStructuredImportance_) {
        filterDefines.push_back({kStructuredImportanceDefine, "1"});
        std::vector<SlangMacroDefine> importanceDefines;
        if (activeProbeDispatch) { importanceDefines.push_back({"METALLIC_LUMEN_ACTIVE_PROBES", "1"}); }
        if (compactAtlas) { importanceDefines.push_back({"METALLIC_LUMEN_COMPACT_PROBE_ATLAS", "1"}); }
        if (radianceCache) { importanceDefines.push_back({"METALLIC_LUMEN_RADIANCE_CACHE", "1"}); }
        const std::pair<const char*, ComputeKernel*> importanceKernels[]{
            {"lumenComputeProbeBrdfPdfMain", &brdfPdf_}, {"lumenComputeProbeLightingPdfMain", &lightingPdf_},
            {"lumenGenerateStructuredProbeRaysMain", &structuredRays_}, {"lumenSaveProbeLightingHistoryMain", &saveProbeHistory_},
        };
        for (const auto& [entry, kernel] : importanceKernels) {
            if (auto result = ShaderRegistry::instance().getComputeKernel(*device_,
                    {.moduleName = "Features/DistanceField/LumenScreenProbeImportance", .entryPointName = entry,
                        .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = importanceDefines},
                    {.parameters = parameterAbi<LumenScreenProbeImportanceParams>(kScreenProbeImportanceABI), .debugName = entry},
                    *kernel, log); !result) { return result; }
        }
    }
    if (activeProbeDispatch) { filterDefines.push_back({"METALLIC_LUMEN_ACTIVE_PROBES", "1"}); }
    if (compactAtlas) { filterDefines.push_back({"METALLIC_LUMEN_COMPACT_PROBE_ATLAS", "1"}); }
    if (jitterDirections) { filterDefines.push_back({"METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS", "1"}); }
    if (compiledTraceResolution_ == 16u) { filterDefines.push_back({"METALLIC_LUMEN_TRACE_RESOLUTION_16", "1"}); }
    if (jitterDirections && useSH3()) { filterDefines.push_back({"METALLIC_LUMEN_IRRADIANCE_SH3", "1"}); }
    auto filterABI = activeProbeDispatch ?
        probeABI.operator()<LumenActiveScreenProbeParams<LumenScreenProbeFilterParams>, ProbeFilterMotionBindings>(kActiveScreenProbeFilterABI) :
        probeABI.operator()<LumenScreenProbeFilterParams, ProbeFilterMotionBindings>(kScreenProbeFilterABI);
    if (compactAtlas) {
        filterABI = probeABI.operator()<CompactScreenProbeFilterParams, ProbeFilterMotionBindings>(kCompactScreenProbeFilterABI);
    }
    const auto createFilter = [&](const char* entry, const char* name, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenScreenProbeFilter", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = filterDefines},
            {.parameters = filterABI, .debugName = name}, kernel, log);
    };
    if (!radianceCache && !compactAtlas) {
        if (auto result = createFilter("lumenPlaceAdaptiveProbesMain", "LumenPlaceAdaptiveProbes", place_); !result) {
            return result;
        }
    }
    if (jitterDirections) {
        if (auto result = createFilter("lumenCompositeScreenProbesMain", "LumenCompositeScreenProbes", composite_); !result) {
            return result;
        }
        if (compiledFastLightingUpdate_) {
            if (auto result = createFilter("lumenReduceScreenProbeMotionMain", "LumenReduceScreenProbeMotion", reduceProbeMotion_);
                !result) {
                return result;
            }
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
    if (compiledExportIndirectIrradiance_) {
        params.indirectIrradiance = writer.storageImageHandle(context.outputTexture("indirectIrradiance").view());
        params.writeIndirectIrradiance = 1u;
    }
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
    const bool activeProbeDispatch = compiledActiveProbeDispatch_;
    const bool splitProbeTracing = compiledSplitProbeTracing_;
    const bool compactAtlas = compiledCompactProbeAtlas_;
    const bool fastLightingUpdate = compiledFastLightingUpdate_;
    const bool jitterDirections = finalGather && !optionEnabled("fixedProbeDirections");
    const bool screenTraces = optionEnabled("screenTraces");
    const bool radianceCache = optionEnabled("radianceCache");
    const bool detailTracing = optionEnabled("detailTracing");
    const bool splitDetailTracing = splitProbeTracing && detailTracing;
    const bool sh3 = jitterDirections && useSH3();
    const char* irradianceName = sh3 ? "probeSHAmbient" : "probeIrradiance";
    const char* probePositionsName = radianceCache && !compactAtlas ? "placedProbes" : "probePositions";
    ShaderBuffer scratch;
    ShaderBuffer compositedDistance;
    if (finalGather) {
        params.probePositions = writer.buffer(radianceCache ? context.inputBuffer(probePositionsName).buffer() :
            context.outputBuffer(probePositionsName).buffer());
        params.traceDistance = writer.buffer(context.outputBuffer("probeDistance").buffer());
        params.traceInput = writer.buffer(context.outputBuffer("filteredProbeRadiance").buffer());
        scratch = writer.buffer(context.outputBuffer("probeFilterScratch").buffer());
        if (jitterDirections || fullResolutionJitterWidth_ > 0.0f) {
            params.blueNoiseVec2 = writer.sampledImageHandle(blueNoise_.get());
        }
        if (jitterDirections) {
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
    if (fullResolutionJitterWidth_ > 0.0f || compiledRoughSpecular_) {
        auto* viewBuffer = context.viewConstantsBuffer();
        if (!viewBuffer) {
            spdlog::error("Lumen full-resolution probe interpolation jitter and rough specular require graph view constants");
            return makeError(Error::InvalidArgument);
        }
        params.jitterView = writer.buffer(viewBuffer);
    }
    if (fullResolutionJitterWidth_ > 0.0f) {
        params.jitterDepth = writer.sampledImageHandle(context.inputTexture("jitterDepth").view());
        if (!compiledProbeJitterComparison_) { params.fullResolutionJitterWidth = fullResolutionJitterWidth_; }
    }
    std::optional<LumenMeshSDFGrid::Frame> meshGridFrame;
    std::unique_ptr<Buffer> meshInstances, validationBounds;
    if (detailTracing) {
        const auto* world = context.world();
        if (world && (world->environment().enabled ||
            world->worldEnvironment().source == environment::EnvironmentSource::PhysicalAtmosphere)) {
            spdlog::error("This Lumen Detail Tracing permutation requires skylight to be disabled");
            return makeError(Error::InvalidArgument);
        }
        auto grid = meshGrid_.prepare(context, writer, meshFields_, meshSDFTraceDistance_, kLumenCardTraceEndDistance,
            fastLightingUpdate);
        if (!grid) { return makeError(grid.error()); }
        meshGridFrame.emplace(std::move(*grid));
        params.detail = meshGridFrame->bindings;
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
    LumenScreenProbeScheduleParams schedule{};
    CompactProbeAtlasBindings atlas{};
    ShaderBuffer adaptiveCount;
    const uint32_t uniformCount = params.probeWidth * params.probeHeight;
    const uint32_t adaptiveCapacity = compactAtlas ? lumenAdaptiveProbeCapacity(properties(), uniformCount) : 0u;
    if (compactAtlas) {
        if (radianceCache) {
            // Gather sizes its ray buffers before the producer resources are bound.
            constexpr uint32_t word = sizeof(uint32_t);
            const uint64_t capacity = uniformCount + uint64_t(adaptiveCapacity);
            const struct {
                const char* name;
                uint64_t size;
                uint32_t stride;
            } inputs[]{
                {probePositionsName, capacity * 2 * word, 2 * word},
                {"probeOrigins", capacity * word, word},
                {"probeAdaptiveCount", word, word},
                {"probeTileAdaptiveCounts", uint64_t(uniformCount) * word, word},
                {"probeTileAdaptiveIndices", uint64_t(uniformCount) * (kLumenScreenProbeSlotsPerTile - 1) * word, word},
            };
            for (const auto& input : inputs) {
                const auto& desc = context.inputBuffer(input.name).desc();
                if (desc.size != input.size || desc.structureStride != input.stride) {
                    spdlog::error("Lumen Gather shared placement input '{}' must match its compact atlas layout and capacity", input.name);
                    return makeError(Error::InvalidArgument);
                }
            }
        }
        const auto bindAtlas = [&](const char* name) {
            return writer.buffer(radianceCache ? context.inputBuffer(name).buffer() : context.outputBuffer(name).buffer());
        };
        atlas = {bindAtlas("probeOrigins"), bindAtlas("probeTileAdaptiveCounts"), bindAtlas("probeTileAdaptiveIndices")};
        adaptiveCount = bindAtlas("probeAdaptiveCount");
    }
    if (activeProbeDispatch) {
        schedule = {params.probePositions, writer.buffer(context.outputBuffer("probeWorkList").buffer()),
            writer.buffer(context.outputBuffer("probeWorkState").buffer()),
            writer.buffer(context.outputBuffer("probeIndirectArgs").buffer()),
            compactAtlas ? uniformCount + adaptiveCapacity : uniformCount * kLumenScreenProbeSlotsPerTile};
    }
    ShaderBuffer structuredRayInfo{};
    EncodedParameters importanceParams;
    bool importanceHistoryValid = false;
    if (compiledStructuredImportance_) {
        auto* viewBuffer = context.viewConstantsBuffer();
        if (!viewBuffer) {
            spdlog::error("Lumen structured probe importance sampling requires graph view constants");
            return makeError(Error::InvalidArgument);
        }
        structuredRayInfo = writer.buffer(context.outputBuffer("structuredRayInfo").buffer());
        // A camera cut invalidates reprojection even when the graph history slots remain valid.
        importanceHistoryValid = context.viewConstants()->frame[1] != 0u &&
            context.temporalValid("previousImportanceRadiance") &&
            context.temporalValid("previousImportanceGeometry");
        const auto historyBuffer = [&](const char* name) { return writer.buffer(context.temporalBuffer(name).buffer.buffer()); };
        auto result = writer.encode(LumenScreenProbeImportanceParams{
            params.position, params.normal, params.blueNoiseVec2, params.probePositions, schedule.workList, schedule.workState,
            writer.buffer(viewBuffer), params.cacheIndirection, params.cacheClipmaps, params.cacheRadiance,
            writer.buffer(context.outputBuffer("probeBrdfPdfSH").buffer()),
            writer.buffer(context.outputBuffer("probeLightingPdf").buffer()), structuredRayInfo, params.traceInput,
            importanceHistoryValid ? historyBuffer("previousImportanceRadiance") : ShaderBuffer{},
            importanceHistoryValid ? historyBuffer("previousImportanceGeometry") : ShaderBuffer{},
            historyBuffer("savedImportanceRadiance"), historyBuffer("savedImportanceGeometry"),
            params.width, params.height, params.probeWidth, params.probeHeight, params.frameIndex,
            params.rayDirectionFrameIndex, importanceHistoryValid, params.scene.sampler}, kScreenProbeImportanceABI);
        if (!result) { return makeError(result.error()); }
        importanceParams = *result;
    }
    ProbeTraceMotionBindings traceMotion{};
    ProbeFilterMotionBindings filterMotion{};
    if (fastLightingUpdate) {
        const auto bindOutput = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
        const ShaderBuffer traceMoving = bindOutput("probeTraceMoving");
        const ShaderBuffer probeMoving = bindOutput("probeMoving");
        traceMotion = {writer.sampledImageHandle(context.inputTexture("previousPosition").view()),
            traceMoving, probeMoving, meshGridFrame ? writer.buffer(meshGridFrame->motion.get()) : ShaderBuffer{}};
        filterMotion = {traceMoving, bindOutput("compositedProbeMoving"), probeMoving};
    }
    const auto encodeProbeParams = [&]<typename Params, typename Motion>(const Params& bindings, uint64_t key, const Motion& motion,
        const ProbeRoughSpecularBindings* rough = nullptr) {
        const auto encodeIntegration = [&]<typename Base>(const Base& base, uint64_t baseKey) {
            return rough ? writer.encode(ScreenProbeRoughSpecularParams<Base>{base, *rough}, baseKey | kRoughSpecularABIBit) :
                writer.encode(base, baseKey);
        };
        const auto encodeMotion = [&]<typename Base>(const Base& base, uint64_t baseKey) {
            return fastLightingUpdate ? encodeIntegration(ScreenProbeMotionParams<Base, Motion>{base, motion},
                baseKey | kFastLightingUpdateABIBit) : encodeIntegration(base, baseKey);
        };
        return compiledStructuredImportance_ ?
            encodeMotion(StructuredImportanceParams<Params>{bindings, structuredRayInfo}, key | kStructuredImportanceABIBit) :
            encodeMotion(bindings, key);
    };
    const ActiveGatherParams activeParams{params, schedule.workList, schedule.workState};
    const auto encodeGather = [&](const ActiveGatherParams& bindings, const ProbeRoughSpecularBindings* rough = nullptr) {
        if (compactAtlas) {
            return encodeProbeParams(CompactScreenProbeGatherParams<ActiveGatherParams>{bindings, atlas},
                kCompactScreenProbeGatherABI, traceMotion, rough);
        }
        return activeProbeDispatch ? encodeProbeParams(bindings, kActiveScreenProbeGatherABI, traceMotion, rough) :
            encodeProbeParams(bindings.base, kScreenProbeGatherABI, traceMotion, rough);
    };
    const auto encoded = encodeGather(activeParams);
    if (!encoded) { return makeError(encoded.error()); }
    ProbeRoughSpecularBindings roughSpecular{};
    if (compiledRoughSpecular_) {
        roughSpecular = {writer.sampledImageHandle(context.inputTexture("specular").view()),
            writer.storageImageHandle(context.outputTexture("roughSpecular").view()), 1u};
    }
    const auto encodedIntegrate = compiledRoughSpecular_ ? encodeGather(activeParams, &roughSpecular) : encoded;
    if (!encodedIntegrate) { return makeError(encodedIntegrate.error()); }
    EncodedParameters comparisonParams;
    if (compiledProbeJitterComparison_) {
        auto comparison = activeParams;
        comparison.base.writeIndirectIrradiance = 0u;
        comparison.base.fullResolutionJitterWidth = fullResolutionJitterWidth_;
        comparison.base.indirectDiffuse = writer.storageImageHandle(context.outputTexture("jitterComparisonIndirectDiffuse").view());
        comparison.base.color = writer.storageImageHandle(context.outputTexture("jitterComparisonColor").view());
        if (interpolationReadbackPixels_) {
            comparison.base.interpolationReadback = writer.buffer(context.outputBuffer("jitterComparisonInterpolationReadback").buffer());
            comparison.base.jitterOffsets = writer.buffer(context.outputBuffer("probeJitterOffsets").buffer());
        }
        auto roughComparison = roughSpecular;
        roughComparison.writeLighting = 0u;
        const auto comparisonEncoded = encodeGather(comparison, compiledRoughSpecular_ ? &roughComparison : nullptr);
        if (!comparisonEncoded) { return makeError(comparisonEncoded.error()); }
        comparisonParams = *comparisonEncoded;
    }
    LumenScreenProbeRayScheduleParams raySchedule{};
    EncodedParameters fallbackParams;
    if (splitProbeTracing) {
        raySchedule = {schedule, params.traceOutput,
            writer.buffer(context.outputBuffer("probeRayWorkList").buffer()),
            writer.buffer(context.outputBuffer("probeRayWorkCount").buffer()),
            writer.buffer(context.outputBuffer("probeRayIndirectArgs").buffer())};
        const LumenScreenProbeFallbackParams fallbackBindings{activeParams, raySchedule.rayWorkList, raySchedule.rayWorkCount};
        const auto fallback = compactAtlas ? encodeProbeParams(CompactScreenProbeGatherParams<LumenScreenProbeFallbackParams>{
            fallbackBindings, atlas}, kCompactScreenProbeFallbackABI, traceMotion) :
            encodeProbeParams(fallbackBindings, kScreenProbeFallbackABI, traceMotion);
        if (!fallback) { return makeError(fallback.error()); }
        fallbackParams = *fallback;
    }

    using Access = RenderGraphResourceAccess;
    std::vector<RenderGraphStageUse> screenUses;
    std::vector<RenderGraphStageUse> detailUses;
    std::vector<RenderGraphStageUse> traceUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead},
        {"coarseMip", Access::TextureSampleRead}, {"domain", Access::BufferStorageRead},
        {"objectGrid", Access::BufferStorageRead}, {"objectGridScene", Access::BufferStorageRead},
        {"cards", Access::BufferStorageRead}, {"cardScene", Access::BufferStorageRead},
        {"cardDepth", Access::TextureSampleRead}, {"cardLighting", Access::TextureSampleRead},
        {"probeRadiance", splitProbeTracing ? Access::BufferStorageReadWrite : Access::BufferStorageWrite},
    };
    std::vector<RenderGraphStageUse> integrateUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"diffuse", Access::TextureSampleRead}, {"directColor", Access::TextureSampleRead},
    };
    if (fullResolutionJitterWidth_ > 0.0f) { integrateUses.push_back({"jitterDepth", Access::TextureSampleRead}); }
    if (jitterDirections) {
        integrateUses.push_back({irradianceName, Access::TextureSampleRead});
        if (sh3) { integrateUses.push_back({"probeSHDirectional", Access::TextureSampleRead}); }
    } else {
        integrateUses.push_back({finalGather ? "filteredProbeRadiance" : "probeRadiance", Access::BufferStorageRead});
    }
    if (screenTraces) {
        auto& historyUses = splitProbeTracing ? screenUses : traceUses;
        historyUses.insert(historyUses.end(), {{"closestHZB", Access::BufferStorageRead},
            {"previousDepth", Access::TextureSampleRead}, {"previousColor", Access::TextureSampleRead},
            {"traceView", Access::BufferStorageRead}});
    }
    if (radianceCache) {
        const RenderGraphStageUse cacheUses[]{{"cacheIndirection", Access::BufferStorageRead},
            {"cacheClipmaps", Access::BufferStorageRead}, {"cacheRadiance", Access::TextureSampleRead}};
        traceUses.insert(traceUses.end(), std::begin(cacheUses), std::end(cacheUses));
        if (splitProbeTracing) { screenUses.insert(screenUses.end(), std::begin(cacheUses), std::end(cacheUses)); }
    }
    if (traceReadbackTiles_) {
        traceUses.push_back({"probeTraceReadback", splitProbeTracing ? Access::BufferStorageReadWrite : Access::BufferStorageWrite});
        if (splitProbeTracing) { screenUses.push_back({"probeTraceReadback", Access::BufferStorageWrite}); }
    }
    if (compactAtlas) {
        traceUses.push_back({"probeOrigins", Access::BufferStorageRead});
        if (splitProbeTracing) { screenUses.push_back({"probeOrigins", Access::BufferStorageRead}); }
        integrateUses.insert(integrateUses.end(), {{"probeTileAdaptiveCounts", Access::BufferStorageRead},
            {"probeTileAdaptiveIndices", Access::BufferStorageRead}});
        if (interpolationReadbackPixels_) { integrateUses.push_back({"probeOrigins", Access::BufferStorageRead}); }
    }
    if (fastLightingUpdate) {
        const RenderGraphStageUse motionTraceUses[]{{"previousPosition", Access::TextureSampleRead},
            {"probeTraceMoving", Access::BufferStorageWrite}};
        auto& initialTraceUses = splitProbeTracing ? screenUses : traceUses;
        initialTraceUses.insert(initialTraceUses.end(), std::begin(motionTraceUses), std::end(motionTraceUses));
        if (splitDetailTracing) {
            detailUses.insert(detailUses.end(), std::begin(motionTraceUses), std::end(motionTraceUses));
        }
        integrateUses.push_back({"probeMoving", Access::BufferStorageRead});
    }

    const RenderGraphStageUse placeUses[] = {{"position", Access::TextureSampleRead},
        {"normal", Access::TextureSampleRead}, {"probePositions", Access::BufferStorageWrite}};
    const RenderGraphStageUse clearCompactReadbackUses[]{{"probeTraceReadback", Access::BufferStorageWrite}};
    std::array<std::vector<RenderGraphStageUse>, kSpatialPassCount> filterUses;
    EncodedParameters placeParams;
    EncodedParameters compositeParams;
    EncodedParameters irradianceParams;
    std::vector<RenderGraphStageUse> irradianceUses = {{probePositionsName, Access::BufferStorageRead},
        {"filteredProbeRadiance", Access::BufferStorageRead}, {irradianceName, Access::TextureStorageWrite}};
    if (sh3) { irradianceUses.push_back({"probeSHDirectional", Access::TextureStorageWrite}); }
    std::vector<RenderGraphStageUse> compositeUses = {{probePositionsName, Access::BufferStorageRead},
        {"probeRadiance", Access::BufferStorageRead}, {"probeDistance", Access::BufferStorageRead},
        {"probeFilterScratch", Access::BufferStorageWrite}, {"compositedProbeDistance", Access::BufferStorageWrite}};
    std::vector<RenderGraphStageUse> reduceMotionUses;
    if (fastLightingUpdate) {
        compositeUses.insert(compositeUses.end(), {{"probeTraceMoving", Access::BufferStorageRead},
            {"compositedProbeMoving", Access::BufferStorageWrite}});
        reduceMotionUses = {{"compositedProbeMoving", Access::BufferStorageRead}, {"probeMoving", Access::BufferStorageWrite}};
    }
    std::array<EncodedParameters, kSpatialPassCount> filteredParams;
    std::vector<RenderGraphStage> stages;
    const RenderGraphStageUse workUses[]{{"probeWorkList", Access::BufferStorageRead},
        {"probeWorkState", Access::BufferStorageRead}, {"probeIndirectArgs", Access::BufferIndirectRead}};
    const auto useProbeSchedule = [&](std::vector<RenderGraphStageUse>& uses) {
        if (activeProbeDispatch) { uses.insert(uses.end(), std::begin(workUses), std::end(workUses)); }
    };
    const auto dispatchIndirect = [&](ComputeKernel& kernel, const EncodedParameters& parameters, const char* argumentBuffer,
                                       uint32_t argumentIndex, CommandBuffer& cmd) -> Result<> {
        const auto args = context.outputBuffer(argumentBuffer).buffer()->slice(
            {argumentIndex * kProbeDispatchArgumentBytes, kProbeDispatchArgumentBytes});
        if (!args) { return makeError(args.error()); }
        return kernel.dispatchIndirect(cmd, parameters, *args);
    };
    const auto dispatchProbes = [&](ComputeKernel& kernel, const EncodedParameters& parameters, CommandBuffer& cmd) {
        if (activeProbeDispatch) { return dispatchIndirect(kernel, parameters, "probeIndirectArgs", 0, cmd); }
        return kernel.dispatch(cmd, parameters, params.probeWidth, params.probeHeight,
            finalGather ? kLumenScreenProbeSlotsPerTile : 1u);
    };
    const RenderGraphStageUse resetProbeUses[]{{"probeWorkState", Access::BufferStorageWrite}};
    std::vector<RenderGraphStageUse> compactProbeUses{{probePositionsName, Access::BufferStorageRead},
        {"probeWorkState", Access::BufferStorageReadWrite}, {"probeWorkList", Access::BufferStorageWrite}};
    if (compactAtlas) { compactProbeUses.push_back({"probeAdaptiveCount", Access::BufferStorageRead}); }
    std::vector<RenderGraphStageUse> buildProbeDispatchUses{{"probeWorkState", Access::BufferStorageRead},
        {"probeIndirectArgs", Access::BufferStorageWrite}};
    std::vector<RenderGraphStageUse> clearProbeUses;
    EncodedParameters scheduleParams, clearParams;
    if (meshGridFrame) {
        LumenMeshSDFGrid::appendTraceUses(splitDetailTracing ? detailUses : traceUses);
        meshGrid_.appendStages(stages, *meshGridFrame);
    }
    if (finalGather) {
        traceUses.insert(traceUses.end(), {{probePositionsName, Access::BufferStorageRead},
            {"probeDistance", splitProbeTracing ? Access::BufferStorageReadWrite : Access::BufferStorageWrite}});
        integrateUses.push_back({probePositionsName, Access::BufferStorageRead});
        LumenScreenProbeFilterParams filterParams{params.position, params.normal, params.probePositions,
            params.probePositions, {}, params.traceDistance, {}, params.width, params.height, params.probeWidth,
            params.probeHeight, params.frameIndex, params.blueNoiseVec2, {}};
        filterParams.rayDirectionFrameIndex = params.rayDirectionFrameIndex;
        const auto encodeFilter = [&] {
            const LumenActiveScreenProbeParams<LumenScreenProbeFilterParams> bindings{filterParams, schedule.workList, schedule.workState};
            if (compactAtlas) {
                return encodeProbeParams(CompactScreenProbeFilterParams{bindings, atlas.origins}, kCompactScreenProbeFilterABI, filterMotion);
            }
            return activeProbeDispatch ? encodeProbeParams(bindings, kActiveScreenProbeFilterABI, filterMotion) :
                encodeProbeParams(filterParams, kScreenProbeFilterABI, filterMotion);
        };
        if (!radianceCache && compactAtlas) {
            if (auto result = placement_.appendStages(context, writer, adaptiveCapacity, stages); !result) { return result; }
        } else if (!radianceCache) {
            auto placement = encodeFilter();
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
            auto composite = encodeFilter();
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
            filterUses[index] = {{"position", Access::TextureSampleRead}, {probePositionsName, Access::BufferStorageRead},
                {jitterDirections ? "compositedProbeDistance" : "probeDistance", Access::BufferStorageRead},
                {inputNames[index], Access::BufferStorageRead},
                {outputNames[index], Access::BufferStorageWrite}};
            useProbeSchedule(filterUses[index]);
            if (fastLightingUpdate) { filterUses[index].push_back({"probeMoving", Access::BufferStorageRead}); }
            if (compactAtlas && !jitterDirections) { filterUses[index].push_back({"probeOrigins", Access::BufferStorageRead}); }
            filterParams.inputRadiance = inputs[index];
            filterParams.outputRadiance = outputs[index];
            auto filter = encodeFilter();
            if (!filter) { return makeError(filter.error()); }
            filteredParams[index] = std::move(*filter);
        }
        if (jitterDirections) {
            filterParams.inputRadiance = params.traceInput;
            filterParams.irradianceOutput = writer.storageImageHandle(context.outputTexture(irradianceName).view());
            if (sh3) {
                filterParams.shDirectionalOutput = writer.storageImageHandle(context.outputTexture("probeSHDirectional").view());
            }
            auto irradiance = encodeFilter();
            if (!irradiance) { return makeError(irradiance.error()); }
            irradianceParams = std::move(*irradiance);
        }
        if (activeProbeDispatch) {
            const auto encodeProbeSchedule = [&] {
                if (compactAtlas) {
                    return splitProbeTracing ? writer.encode(CompactScreenProbeScheduleParams<LumenScreenProbeRayScheduleParams>{
                        raySchedule, adaptiveCount, uniformCount}, kCompactScreenProbeRayScheduleABI) :
                        writer.encode(CompactScreenProbeScheduleParams<LumenScreenProbeScheduleParams>{
                            schedule, adaptiveCount, uniformCount}, kCompactScreenProbeScheduleABI);
                }
                return splitProbeTracing ? writer.encode(raySchedule, kScreenProbeRayScheduleABI) : writer.encode(schedule, kScreenProbeScheduleABI);
            };
            const auto encodedSchedule = encodeProbeSchedule();
            if (!encodedSchedule) { return makeError(encodedSchedule.error()); }
            scheduleParams = *encodedSchedule;
            if (splitProbeTracing) {
                buildProbeDispatchUses.insert(buildProbeDispatchUses.end(), {{"probeRayWorkCount", Access::BufferStorageWrite},
                    {"probeRayIndirectArgs", Access::BufferStorageWrite}});
            }
            stages.push_back({"Reset screen probe work counts", resetProbeUses, [&](CommandBuffer& cmd) {
                return resetProbeSchedule_.dispatch(cmd, scheduleParams, 1, 1, 1);
            }});
            stages.push_back({"Compact screen probe work", compactProbeUses, [&](CommandBuffer& cmd) {
                return compactProbes_.dispatch(cmd, scheduleParams,
                    (schedule.capacity + kProbeCompactionThreads - 1u) / kProbeCompactionThreads, 1, 1);
            }});
            stages.push_back({"Build screen probe indirect dispatches", buildProbeDispatchUses, [&](CommandBuffer& cmd) {
                return buildProbeDispatch_.dispatch(cmd, scheduleParams, 1, 1, 1);
            }});
            if (compactAtlas) {
                if (traceReadbackTiles_) {
                    stages.push_back({"Initialize compact probe debug ROI", clearCompactReadbackUses, [&](CommandBuffer& cmd) {
                        return clearCompactReadback_.dispatch(cmd, *encoded, params.traceReadbackTiles[2],
                            params.traceReadbackTiles[3], kLumenScreenProbeSlotsPerTile);
                    }});
                }
            } else {
                const auto encodedClear = writer.encode(LumenScreenProbeClearParams{schedule.workList, schedule.workState,
                    params.traceOutput, params.traceDistance, params.traceInput, scratch, compositedDistance,
                    filterParams.irradianceOutput, filterParams.shDirectionalOutput, params.traceReadback,
                    params.traceReadbackTiles, params.probeWidth, params.probeHeight}, kScreenProbeClearABI);
                if (!encodedClear) { return makeError(encodedClear.error()); }
                clearParams = *encodedClear;
                clearProbeUses = {{"probeRadiance", Access::BufferStorageWrite}, {"probeDistance", Access::BufferStorageWrite},
                    {"filteredProbeRadiance", Access::BufferStorageWrite}, {"probeFilterScratch", Access::BufferStorageWrite}};
                if (jitterDirections) {
                    clearProbeUses.insert(clearProbeUses.end(), {{"compositedProbeDistance", Access::BufferStorageWrite},
                        {irradianceName, Access::TextureStorageWrite}});
                    if (sh3) { clearProbeUses.push_back({"probeSHDirectional", Access::TextureStorageWrite}); }
                }
                if (traceReadbackTiles_) { clearProbeUses.push_back({"probeTraceReadback", Access::BufferStorageWrite}); }
                useProbeSchedule(clearProbeUses);
                stages.push_back({"Clear inactive screen probes", clearProbeUses, [&](CommandBuffer& cmd) {
                    return dispatchIndirect(clearInactiveProbes_, clearParams, "probeIndirectArgs", 1, cmd);
                }});
            }
        }
    }
    std::vector<RenderGraphStageUse> brdfPdfUses, lightingPdfUses, generateRayUses;
    if (compiledStructuredImportance_) {
        brdfPdfUses = {{"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
            {probePositionsName, Access::BufferStorageRead}, {"probeBrdfPdfSH", Access::BufferStorageWrite}};
        lightingPdfUses = {{"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
            {probePositionsName, Access::BufferStorageRead}, {"probeLightingPdf", Access::BufferStorageWrite}};
        if (importanceHistoryValid) {
            lightingPdfUses.insert(lightingPdfUses.end(), {{"previousImportanceRadiance", Access::BufferStorageRead},
                {"previousImportanceGeometry", Access::BufferStorageRead}});
        }
        if (radianceCache) {
            lightingPdfUses.insert(lightingPdfUses.end(), {{"cacheIndirection", Access::BufferStorageRead},
                {"cacheClipmaps", Access::BufferStorageRead}, {"cacheRadiance", Access::TextureSampleRead}});
        }
        generateRayUses = {{probePositionsName, Access::BufferStorageRead}, {"probeBrdfPdfSH", Access::BufferStorageRead},
            {"probeLightingPdf", Access::BufferStorageRead}, {"structuredRayInfo", Access::BufferStorageWrite}};
        useProbeSchedule(brdfPdfUses);
        useProbeSchedule(lightingPdfUses);
        useProbeSchedule(generateRayUses);
        stages.push_back({"Compute screen probe BRDF PDF", brdfPdfUses, [&](CommandBuffer& cmd) {
            return dispatchProbes(brdfPdf_, importanceParams, cmd);
        }});
        stages.push_back({"Compute screen probe lighting PDF", lightingPdfUses, [&](CommandBuffer& cmd) {
            return dispatchProbes(lightingPdf_, importanceParams, cmd);
        }});
        stages.push_back({"Generate structured screen probe rays", generateRayUses, [&](CommandBuffer& cmd) {
            return dispatchProbes(structuredRays_, importanceParams, cmd);
        }});
        traceUses.push_back({"structuredRayInfo", Access::BufferStorageRead});
        if (splitProbeTracing) { screenUses.push_back({"structuredRayInfo", Access::BufferStorageRead}); }
        if (splitDetailTracing) { detailUses.push_back({"structuredRayInfo", Access::BufferStorageRead}); }
        compositeUses.push_back({"structuredRayInfo", Access::BufferStorageRead});
    }
    const RenderGraphStageUse compactRayUses[]{{"probeWorkList", Access::BufferStorageRead},
        {"probeWorkState", Access::BufferStorageRead}, {"probeRadiance", Access::BufferStorageRead},
        {"probeRayIndirectArgs", Access::BufferIndirectRead}, {"probeRayWorkList", Access::BufferStorageWrite},
        {"probeRayWorkCount", Access::BufferStorageReadWrite}};
    const RenderGraphStageUse buildRayDispatchUses[]{{"probeRayWorkCount", Access::BufferStorageRead},
        {"probeRayIndirectArgs", Access::BufferStorageWrite}};
    if (splitProbeTracing) {
        screenUses.insert(screenUses.end(), {{"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
            {probePositionsName, Access::BufferStorageRead}, {"probeRadiance", Access::BufferStorageWrite},
            {"probeDistance", Access::BufferStorageWrite}});
        useProbeSchedule(screenUses);
        traceUses.insert(traceUses.end(), {{"probeRayWorkList", Access::BufferStorageRead},
            {"probeRayWorkCount", Access::BufferStorageRead}, {"probeRayIndirectArgs", Access::BufferIndirectRead}});
        if (splitDetailTracing) {
            screenUses.insert(screenUses.end(), {{"cardScene", Access::BufferStorageRead},
                {"objectGridScene", Access::BufferStorageRead}});
            detailUses.insert(detailUses.end(), {{"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
                {probePositionsName, Access::BufferStorageRead}, {"cards", Access::BufferStorageRead},
                {"cardScene", Access::BufferStorageRead}, {"cardDepth", Access::TextureSampleRead},
                {"cardLighting", Access::TextureSampleRead}, {"probeRadiance", Access::BufferStorageWrite},
                {"probeDistance", Access::BufferStorageReadWrite}, {"probeRayWorkList", Access::BufferStorageRead},
                {"probeRayWorkCount", Access::BufferStorageRead}, {"probeRayIndirectArgs", Access::BufferIndirectRead}});
            if (traceReadbackTiles_) {
                detailUses.push_back({"probeTraceReadback", Access::BufferStorageReadWrite});
            }
            if (compactAtlas) { detailUses.push_back({"probeOrigins", Access::BufferStorageRead}); }
        }
        stages.push_back({"Trace screen probe rays against HZB", screenUses, [&](CommandBuffer& cmd) {
            return dispatchProbes(trace_, *encoded, cmd);
        }});
        if (splitDetailTracing) {
            stages.push_back({"Compact probe rays for Mesh SDF", compactRayUses, [&](CommandBuffer& cmd) {
                return dispatchIndirect(compactDetailRays_, scheduleParams, "probeRayIndirectArgs", 0, cmd);
            }});
            stages.push_back({"Build Mesh SDF probe trace dispatch", buildRayDispatchUses, [&](CommandBuffer& cmd) {
                return buildDetailDispatch_.dispatch(cmd, scheduleParams, 1, 1, 1);
            }});
            stages.push_back({"Trace compacted probe rays through Mesh SDF", detailUses, [&](CommandBuffer& cmd) {
                return dispatchIndirect(traceDetail_, fallbackParams, "probeRayIndirectArgs", 2, cmd);
            }});
        }
        stages.push_back({"Compact unfinished screen probe rays", compactRayUses, [&](CommandBuffer& cmd) {
            return dispatchIndirect(compactProbeRays_, scheduleParams, "probeRayIndirectArgs", 0, cmd);
        }});
        stages.push_back({"Build software probe trace dispatch", buildRayDispatchUses, [&](CommandBuffer& cmd) {
            return buildRayDispatch_.dispatch(cmd, scheduleParams, 1, 1, 1);
        }});
        stages.push_back({"Trace compacted probe rays through GDF", traceUses, [&](CommandBuffer& cmd) {
            return dispatchIndirect(traceFallback_, fallbackParams, "probeRayIndirectArgs", 1, cmd);
        }});
    } else {
        useProbeSchedule(traceUses);
        stages.push_back({"Trace screen probes", traceUses, [&](CommandBuffer& cmd) {
            return dispatchProbes(trace_, *encoded, cmd);
        }});
    }
    if (jitterDirections) {
        useProbeSchedule(compositeUses);
        stages.push_back({"Composite jittered rays into gather directions", compositeUses, [&](CommandBuffer& cmd) {
            return dispatchProbes(composite_, compositeParams, cmd);
        }});
        if (fastLightingUpdate) {
            useProbeSchedule(reduceMotionUses);
            stages.push_back({"Reduce screen probe motion", reduceMotionUses, [&](CommandBuffer& cmd) {
                return dispatchProbes(reduceProbeMotion_, compositeParams, cmd);
            }});
        }
    }
    for (uint32_t index = 0; finalGather && index < kSpatialPassCount; ++index) {
        stages.push_back({"Spatially filter probe radiance", filterUses[index], [&, index](CommandBuffer& cmd) {
            return dispatchProbes(filter_, filteredParams[index], cmd);
        }});
    }
    const RenderGraphStageUse saveImportanceUses[]{{"position", Access::TextureSampleRead},
        {probePositionsName, Access::BufferStorageRead}, {"filteredProbeRadiance", Access::BufferStorageRead},
        {"savedImportanceRadiance", Access::BufferStorageWrite}, {"savedImportanceGeometry", Access::BufferStorageWrite}};
    if (compiledStructuredImportance_) {
        stages.push_back({"Save filtered screen probe lighting history", saveImportanceUses, [&](CommandBuffer& cmd) {
            return saveProbeHistory_.dispatch(cmd, importanceParams, params.probeWidth, params.probeHeight, 1);
        }});
    }
    if (jitterDirections) {
        useProbeSchedule(irradianceUses);
        stages.push_back({"Convert screen probe radiance to irradiance", irradianceUses, [&](CommandBuffer& cmd) {
            return dispatchProbes(convertToIrradiance_, irradianceParams, cmd);
        }});
    }
    // Both integrations read the same frame's probe data. Only the baseline feeds lighting history.
    std::vector<RenderGraphStageUse> comparisonUses;
    if (compiledProbeJitterComparison_) {
        comparisonUses = integrateUses;
        comparisonUses.insert(comparisonUses.end(), {{"jitterComparisonIndirectDiffuse", Access::TextureStorageWrite},
            {"jitterComparisonColor", Access::TextureStorageWrite}});
        if (interpolationReadbackPixels_) {
            comparisonUses.insert(comparisonUses.end(), {{"jitterComparisonInterpolationReadback", Access::BufferStorageWrite},
                {"probeJitterOffsets", Access::BufferStorageWrite}});
        }
        stages.push_back({"Compare jittered probe interpolation", comparisonUses, [&](CommandBuffer& cmd) {
            return integrate_.dispatch(cmd, comparisonParams, (params.width + 7u) / 8u, (params.height + 7u) / 8u, 1);
        }});
    }
    integrateUses.insert(integrateUses.end(), {{"indirectDiffuse", Access::TextureStorageWrite}, {"color", Access::TextureStorageWrite}});
    if (compiledRoughSpecular_) {
        integrateUses.insert(integrateUses.end(), {{"specular", Access::TextureSampleRead},
            {"filteredProbeRadiance", Access::BufferStorageRead}, {"roughSpecular", Access::TextureStorageWrite}});
    }
    if (compiledExportIndirectIrradiance_) { integrateUses.push_back({"indirectIrradiance", Access::TextureStorageWrite}); }
    if (interpolationReadbackPixels_) { integrateUses.push_back({"probeInterpolationReadback", Access::BufferStorageWrite}); }
    stages.push_back({compiledRoughSpecular_ ? "Integrate diffuse and rough specular" : "Integrate and composite diffuse",
        integrateUses, [&](CommandBuffer& cmd) {
        return integrate_.dispatch(cmd, *encodedIntegrate, (params.width + 7u) / 8u, (params.height + 7u) / 8u, 1);
    }});
    auto result = context.executeStages(stages);
    if (result && compiledStructuredImportance_) {
        context.setTemporalWritten("savedImportanceRadiance", true);
        context.setTemporalWritten("savedImportanceGeometry", true);
    }
    return result;
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenScreenProbeGatherPass()
{
    return std::make_unique<LumenScreenProbeGatherPass>();
}

} // namespace metallic::render::builtin_pass
