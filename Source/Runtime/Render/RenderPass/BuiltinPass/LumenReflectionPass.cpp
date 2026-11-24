#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenLightingHistory.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenMeshSDFGrid.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kLumenReflectionABI = 0x4c554d454e050003ull;
constexpr uint64_t kLumenReflectionTiledABI = 0x4c554d454e050004ull;
constexpr uint64_t kLumenReflectionScheduleABI = 0x4c554d454e050005ull;
constexpr uint64_t kLumenReflectionTemporalABI = 0x4c554d454e060001ull;
constexpr uint64_t kLumenReflectionBilateralABI = 0x4c554d454e070002ull;
constexpr char kShaderModule[] = "Features/DistanceField/LumenReflection";
constexpr char kScheduleModule[] = "Features/DistanceField/LumenReflectionSchedule";
constexpr std::array kHistoryNames{"lighting", "depth", "age"};
constexpr std::array kHistoryFormats{Format::RGBA16Sfloat, Format::R32Sfloat, Format::R8Unorm};
constexpr uint32_t kScreenTraceViewStride = sizeof(ViewConstants) + 4 * sizeof(uint32_t);
constexpr uint32_t kReflectionTileSize = 8;
constexpr uint32_t kReflectionMaxDispatchX = 65535;
constexpr uint32_t kReflectionMaxExtent = 1u << 14u;
constexpr uint32_t kReflectionDispatchBytes = 3 * sizeof(uint32_t);

enum class ReflectionDispatch : uint32_t { ActiveTiles, ClearTiles, CompactRays, DetailRays, GDFRays, Count };

uint32_t reflectionTileCount(uint32_t pixels)
{
    return (pixels + kReflectionTileSize - 1u) / kReflectionTileSize;
}

struct LumenReflectionScreenBindings {
    ShaderBuffer closestHZB;
    ShaderSampledImage previousDepth, previousColor;
    ShaderBuffer traceView;
};

struct LumenReflectionParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderSampledImage position, normal, specular, sceneColor, blueNoise;
    ShaderStorageImage outputReflections, trace, color;
    ShaderSampledImage reflections;
    ViewCameraConstants camera;
    uint32_t width, height, clipmapCount, frameIndex;
    float maxTraceDistance, maxRoughness;
    ShaderStorageImage outputRayData, outputRayHitDistance, outputResolvedReflections, outputResolvedHitDistance;
    ShaderSampledImage rayData, rayHitDistance, bilateralReflections;
    LumenReflectionScreenBindings screen;
    ShaderLumenMeshSDFGridBindings detail;
};
static_assert(offsetof(LumenReflectionParams, camera) == 80 && offsetof(LumenReflectionParams, width) == 160 &&
    offsetof(LumenReflectionParams, maxTraceDistance) == 176 &&
    offsetof(LumenReflectionParams, outputRayData) == 184 && offsetof(LumenReflectionParams, rayData) == 200 &&
    offsetof(LumenReflectionParams, screen) == 212 && offsetof(LumenReflectionParams, detail) == 228 &&
    sizeof(LumenReflectionParams) == 304);

struct LumenReflectionTiledParams {
    LumenReflectionParams base;
    ShaderBuffer activeTiles, clearTiles, workCounts, rayList;
    uint32_t maxDispatchX;
};
static_assert(offsetof(LumenReflectionTiledParams, activeTiles) == 304 &&
    offsetof(LumenReflectionTiledParams, maxDispatchX) == 320 && sizeof(LumenReflectionTiledParams) == 336);

struct LumenReflectionScheduleParams {
    ShaderSampledImage position, specular, rayData, rayHitDistance, trace;
    ShaderBuffer tileMask, activeTiles, clearTiles, workCounts, rayList, indirectArgs;
    uint32_t width, height, maxDispatchX;
    float maxRoughness, meshSDFTraceDistance, cardTraceEndDistance, maxTraceDistance;
};
static_assert(offsetof(LumenReflectionScheduleParams, width) == 44 && sizeof(LumenReflectionScheduleParams) == 72);

struct LumenReflectionTemporalParams {
    ShaderSampledImage position, normal, specular, previousPosition;
    ShaderSampledImage resolvedReflections, resolvedHitDistance, previousLighting, previousDepth, previousAge;
    ShaderStorageImage temporalReflections, temporalState, currentLighting, currentDepth, currentAge;
    uint32_t width, height;
    ViewCameraConstants currentCamera, previousCamera;
    uint32_t historyValid, dynamicReprojection, frameIndex;
};
static_assert(offsetof(LumenReflectionTemporalParams, temporalReflections) == 36 &&
    offsetof(LumenReflectionTemporalParams, width) == 56 &&
    offsetof(LumenReflectionTemporalParams, currentCamera) == 64 &&
    offsetof(LumenReflectionTemporalParams, previousCamera) == 144 &&
    offsetof(LumenReflectionTemporalParams, historyValid) == 224 &&
    sizeof(LumenReflectionTemporalParams) == 240);

struct LumenReflectionBilateralParams {
    ViewCameraConstants camera;
    ShaderSampledImage position, normal, specular, temporalReflections, currentLighting, currentAge;
    ShaderStorageImage bilateralReflections;
    uint32_t width, height, frameIndex;
};
static_assert(offsetof(LumenReflectionBilateralParams, position) == 80 &&
    offsetof(LumenReflectionBilateralParams, bilateralReflections) == 104 &&
    offsetof(LumenReflectionBilateralParams, frameIndex) == 116 && sizeof(LumenReflectionBilateralParams) == 128);

struct HistoryContent {
    uint64_t reprojectionRevision;
    uint64_t sceneIdentity;
    uint64_t sceneLifetimeRevision;
    uint64_t sceneStructuralRevision;
    uint64_t sceneContentRevision;
    uint64_t geometryTransformRevision;
    uint64_t visibilityRevision;
    uint64_t nonSunLightingRevision;
    uint64_t environmentRevision;
    uint32_t width;
    uint32_t height;
    uint32_t cameraCutSerial;
    bool operator==(const HistoryContent&) const = default;
};

class LumenReflectionPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }

private:
    bool optionEnabled(std::string_view name) const
    {
        const auto value = properties().find(name);
        return value != properties().end() && *value == true;
    }
    Result<> compileClassifiedTracing(const std::vector<SlangMacroDefine>& traceDefines, std::string& log);

    Device* device_ = nullptr;
    std::unique_ptr<TextureView> blueNoise_;
    ComputeKernel trace_;
    ComputeKernel initWork_, markTiles_, buildTileLists_, setupTileArgs_;
    ComputeKernel generateTiledRays_, clearTiles_, traceScreenTiles_;
    ComputeKernel compactDetailRays_, setupDetailArgs_, traceDetailRays_;
    ComputeKernel compactGDFRays_, setupGDFArgs_, traceGDFRays_;
    ComputeKernel resolve_;
    ComputeKernel temporal_;
    ComputeKernel bilateral_;
    ComputeKernel composite_;
    MeshDistanceFieldGPU meshFields_;
    LumenMeshSDFGrid meshGrid_;
    float maxTraceDistance_ = 200.0f;
    float maxRoughness_ = 0.4f;
    float meshSDFTraceDistance_ = kLumenDefaultMeshSDFTraceDistance;
    // Preserve reprojectable history across ordinary camera motion, as diffuse gather does.
    HistoryResourceManager history_;
    std::optional<HistoryContent> lastContent_;
    uint64_t lastFrame_ = 0;
    float previousSunIrradianceMax_ = 0.0f;
};

RenderPassReflection LumenReflectionPass::reflect(const RenderGraphCompileContext& context) const
{
    using Access = RenderGraphResourceAccess;
    using Initialization = RenderGraphInitialization;
    RenderPassReflection reflection;
    reflectLumenGlobalTraceInputs(reflection);
    reflection.addTextureInput("position", "World receiver position and positive linear depth")
        .sampledRead().format = Format::RGBA32Sfloat;
    reflection.addTextureInput("normal", "World receiver shading normal")
        .sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("specular", "Integrated receiver specular response and perceptual roughness")
        .sampledRead().format = Format::RGBA16Sfloat;
    if (optionEnabled("dynamicReprojection")) {
        reflection.addTextureInput("previousPosition", "Previous world position of the current receiver and validity")
            .sampledRead().format = Format::RGBA32Sfloat;
    }
    reflection.addTextureInput("sceneColor", "Scene-linear color before specular reflections")
        .sampledRead().format = Format::RGBA32Sfloat;
    if (optionEnabled("screenTraces")) {
        reflection.addBufferInput("closestHZB", "Current closest reversed-Z depth hierarchy")
            .buffer(0, sizeof(float)).storageRead();
        reflection.addTextureInput("previousDepth", "Previous device depth").sampledRead().format = Format::R32Sfloat;
        reflection.addTextureInput("previousColor", "Previous scene-linear color for reflection screen hits")
            .sampledRead().format = Format::RGBA32Sfloat;
        reflection.addBufferInput("traceView", "Screen trace view and history validity")
            .buffer(kScreenTraceViewStride, kScreenTraceViewStride).storageRead();
    }
    if (optionEnabled("detailTracing")) { LumenMeshSDFGrid::reflect(reflection, context); }
    const bool classifiedTracing = optionEnabled("classifiedTracing");
    if (classifiedTracing) {
        const uint64_t tileCount = uint64_t(reflectionTileCount(context.width)) * reflectionTileCount(context.height);
        const uint64_t pixelCount = uint64_t(context.width) * context.height;
        const auto addWorkBuffer = [&](const char* name, const char* description, uint64_t words) -> auto& {
            return reflection.addBufferOutput(name, description).buffer(words * sizeof(uint32_t), sizeof(uint32_t))
                .storageReadWrite().bufferLayout("u32");
        };
        addWorkBuffer("reflectionTileMask", "Per-tile reflection participation bit", tileCount)
            .transient(Initialization::FullOverwrite);
        addWorkBuffer("reflectionActiveTiles", "Packed full-resolution reflection tiles", tileCount);
        addWorkBuffer("reflectionClearTiles", "Packed tiles without reflection rays", tileCount);
        addWorkBuffer("reflectionWorkCounts", "Active tiles, clear tiles, Mesh SDF rays and GDF rays", 4)
            .transient(Initialization::FullOverwrite);
        addWorkBuffer("reflectionRayList", "Packed pixel coordinates from the latest unfinished-ray compaction", pixelCount);
        reflection.addBufferOutput("reflectionDispatchArgs", "Active tiles, clear tiles, compaction, Mesh SDF and GDF dispatches")
            .buffer(uint32_t(ReflectionDispatch::Count) * kReflectionDispatchBytes, sizeof(uint32_t)).storageWrite()
            .stageAccess(Access::BufferIndirectRead).transient(Initialization::FullOverwrite);
    }

    auto& reflections = reflection.addTextureOutput("reflections", "Raw scene-linear reflection radiance and roughness fade")
        .storageWrite().stageAccess(Access::TextureSampleRead);
    reflections.format = Format::RGBA16Sfloat;
    reflections.colorEncoding = DisplayColorEncoding::SceneLinear;
    auto& trace = reflection.addTextureOutput("trace",
        "Hit distance in meters, trace classification (5 Screen, 9 Mesh SDF), GDF clipmap and card cache classification")
        .storageWrite();
    trace.format = Format::RGBA32Sfloat;
    if (classifiedTracing) { trace.stageAccess(Access::TextureSampleRead); }
    reflection.addTextureOutput("rayData", "Reflection ray direction and reciprocal PDF")
        .storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite)
        .format = Format::RGBA16Sfloat;
    auto& rayHitDistance = reflection.addTextureOutput("rayHitDistance", "Encoded reflection ray distance in centimeters")
        .storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite);
    rayHitDistance.format = Format::R16Sfloat;
    if (classifiedTracing) { rayHitDistance.stageAccess(Access::TextureStorageReadWrite); }
    auto& resolvedReflections = reflection.addTextureOutput("resolvedReflections",
        "Resolved scene-linear reflection radiance and roughness fade");
    resolvedReflections.storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite);
    resolvedReflections.format = Format::RGBA16Sfloat;
    resolvedReflections.colorEncoding = DisplayColorEncoding::SceneLinear;
    reflection.addTextureOutput("resolvedHitDistance", "Resolved reflection hit distance in centimeters")
        .storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite)
        .format = Format::R16Sfloat;
    auto& temporalReflections = reflection.addTextureOutput("temporalReflections", "Accumulated scene-linear reflection radiance and roughness fade")
        .storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite);
    temporalReflections.format = Format::RGBA16Sfloat;
    temporalReflections.colorEncoding = DisplayColorEncoding::SceneLinear;
    reflection.addTextureOutput("temporalState", "Reflection history age and reprojection method")
        .storageWrite().transient(Initialization::FullOverwrite).format = Format::RG16Sfloat;
    auto& bilateralReflections = reflection.addTextureOutput("bilateralReflections",
        "Bilateral-filtered scene-linear reflection radiance and roughness fade")
        .storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite);
    bilateralReflections.format = Format::RGBA16Sfloat;
    bilateralReflections.colorEncoding = DisplayColorEncoding::SceneLinear;
    auto& color = reflection.addTextureOutput("color", "Scene-linear color with the reflection response applied")
        .storageWrite();
    color.format = Format::RGBA32Sfloat;
    color.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenReflectionPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    if (!context.device || !context.graphicsQueue) {
        log = "Lumen Reflection requires a rendering device and a graphics queue";
        return makeError(Error::InvalidArgument);
    }
    for (std::string_view option : {"dynamicReprojection", "screenTraces", "detailTracing", "classifiedTracing"}) {
        const auto value = properties().find(option);
        if (value != properties().end() && !value->is_boolean()) {
            log = "Lumen Reflection " + std::string(option) + " must be a boolean";
            return makeError(Error::InvalidArgument);
        }
    }
    if (optionEnabled("classifiedTracing") && (context.width > kReflectionMaxExtent || context.height > kReflectionMaxExtent)) {
        log = "Lumen classified reflection coordinates require a render extent of at most 16384 pixels per axis";
        return makeError(Error::InvalidArgument);
    }
    try {
        // UE 5.7.4 FPostProcessSettings defaults to 20000 cm and roughness 0.4.
        maxTraceDistance_ = properties().value("maxTraceDistance", 200.0f);
        maxRoughness_ = properties().value("maxRoughness", 0.4f);
        meshSDFTraceDistance_ = properties().value("meshSDFTraceDistance", kLumenDefaultMeshSDFTraceDistance);
    } catch (const RenderGraphProperties::exception& error) {
        log = "Invalid Lumen Reflection properties: " + std::string(error.what());
        return makeError(Error::InvalidArgument);
    }
    if (!(std::isfinite(maxTraceDistance_) && maxTraceDistance_ >= 0.0001f) ||
        !(maxRoughness_ >= 0.0f && maxRoughness_ <= 1.0f)) {
        log = "Lumen Reflection requires a finite trace distance of at least 0.0001 m and roughness in [0, 1]";
        return makeError(Error::InvalidArgument);
    }

    device_ = context.device;
    std::vector<SlangMacroDefine> traceDefines;
    if (optionEnabled("screenTraces")) { traceDefines.push_back({"METALLIC_LUMEN_REFLECTION_SCREEN_TRACES", "1"}); }
    if (optionEnabled("detailTracing")) {
        const auto paths = properties().find("detailDistanceFieldPaths");
        if (!context.runtimeScene || paths == properties().end() || !paths->is_array() || paths->empty()) {
            log = "Lumen Reflection detail tracing requires a scene and detailDistanceFieldPaths";
            return makeError(Error::InvalidArgument);
        }
        if (!(std::isfinite(meshSDFTraceDistance_) && meshSDFTraceDistance_ >= 0.0001f)) {
            log = "Lumen Reflection meshSDFTraceDistance must be finite and at least 0.0001 m";
            return makeError(Error::InvalidArgument);
        }
        meshSDFTraceDistance_ = std::min(meshSDFTraceDistance_, maxTraceDistance_);
        std::vector<std::filesystem::path> fields;
        for (const auto& path : *paths) {
            if (!path.is_string()) {
                log = "Lumen Reflection detailDistanceFieldPaths must contain paths";
                return makeError(Error::InvalidArgument);
            }
            fields.push_back(std::filesystem::path(PROJECT_SOURCE_DIR) / path.get_ref<const std::string&>());
        }
        if (auto result = meshFields_.initialize(*device_, *context.graphicsQueue, *context.runtimeScene, fields, log); !result) {
            return result;
        }
        if (auto result = meshGrid_.compile(*device_, log); !result) { return result; }
        traceDefines.push_back({"METALLIC_LUMEN_REFLECTION_DETAIL_TRACING", "1"});
    }
    lastContent_.reset();
    if (auto result = history_.initialize(*device_); !result) { return result; }
    auto blueNoise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Vec2, log);
    if (!blueNoise) { return makeError(blueNoise.error()); }
    blueNoise_ = std::move(*blueNoise);
    auto& registry = ShaderRegistry::instance();
    if (optionEnabled("classifiedTracing")) {
        if (auto result = compileClassifiedTracing(traceDefines, log); !result) { return result; }
    } else {
        if (auto result = registry.getComputeKernel(*device_,
                {.moduleName = kShaderModule, .entryPointName = "lumenReflectionTraceMain",
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = traceDefines},
                {.parameters = parameterAbi<LumenReflectionParams>(kLumenReflectionABI), .debugName = "LumenReflectionTrace"},
                trace_, log); !result) {
            return result;
        }
        if (auto result = registry.getComputeKernel(*device_,
                {.moduleName = kShaderModule, .entryPointName = "lumenReflectionResolveMain",
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<LumenReflectionParams>(kLumenReflectionABI), .debugName = "LumenReflectionResolve"},
                resolve_, log); !result) {
            return result;
        }
    }
    auto result = registry.getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "lumenReflectionTemporalMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenReflectionTemporalParams>(kLumenReflectionTemporalABI),
            .debugName = "LumenReflectionTemporal"}, temporal_, log);
    if (!result) { return result; }
    result = registry.getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "lumenReflectionBilateralMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenReflectionBilateralParams>(kLumenReflectionBilateralABI),
            .debugName = "LumenReflectionBilateral"}, bilateral_, log);
    if (!result) { return result; }
    return registry.getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "lumenReflectionCompositeMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenReflectionParams>(kLumenReflectionABI), .debugName = "LumenReflectionComposite"},
        composite_, log);
}

Result<> LumenReflectionPass::compileClassifiedTracing(const std::vector<SlangMacroDefine>& traceDefines, std::string& log)
{
    auto& registry = ShaderRegistry::instance();
    const auto caps = device_->capabilities();
    std::vector<SlangMacroDefine> scheduleDefines;
    // UE's wave permutation assigns 16 tiles to one 1024-lane group.
    if (caps.subgroupSize == 32u && caps.minSubgroupSize == 32u && caps.maxComputeWorkgroupSubgroups >= 32u &&
        caps.computeSubgroupBallotArithmetic && caps.computeSubgroupShuffle) {
        scheduleDefines.push_back({"METALLIC_LUMEN_REFLECTION_WAVE_OPS", "1"});
    }
    const auto schedule = [&](const char* entry, const char* label, ComputeKernel& kernel) {
        return registry.getComputeKernel(*device_,
            {.moduleName = kScheduleModule, .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = scheduleDefines},
            {.parameters = parameterAbi<LumenReflectionScheduleParams>(kLumenReflectionScheduleABI), .debugName = label},
            kernel, log);
    };
    struct KernelEntry { const char* entry; const char* label; ComputeKernel& kernel; };
    const KernelEntry scheduleEntries[]{
        {"lumenReflectionInitWorkMain", "LumenReflectionInitWork", initWork_},
        {"lumenReflectionMarkTilesMain", "LumenReflectionMarkTiles", markTiles_},
        {"lumenReflectionBuildTileListsMain", "LumenReflectionBuildTileLists", buildTileLists_},
        {"lumenReflectionSetupTileArgsMain", "LumenReflectionSetupTileArgs", setupTileArgs_},
        {"lumenReflectionCompactRaysMain", "LumenReflectionCompactGDFRays", compactGDFRays_},
        {"lumenReflectionSetupGDFArgsMain", "LumenReflectionSetupGDFArgs", setupGDFArgs_},
    };
    for (const auto& [entry, label, kernel] : scheduleEntries) {
        if (auto result = schedule(entry, label, kernel); !result) { return result; }
    }
    if (optionEnabled("detailTracing")) {
        if (auto result = schedule("lumenReflectionSetupDetailArgsMain", "LumenReflectionSetupDetailArgs", setupDetailArgs_); !result) {
            return result;
        }
        scheduleDefines.push_back({"METALLIC_LUMEN_REFLECTION_COMPACT_DETAIL", "1"});
        if (auto result = schedule("lumenReflectionCompactRaysMain", "LumenReflectionCompactDetailRays", compactDetailRays_); !result) {
            return result;
        }
    }
    auto tiledDefines = traceDefines;
    tiledDefines.push_back({"METALLIC_LUMEN_REFLECTION_TILED_TRACING", "1"});
    const auto tiled = [&](const char* entry, const char* label, ComputeKernel& kernel) {
        return registry.getComputeKernel(*device_,
            {.moduleName = kShaderModule, .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = tiledDefines},
            {.parameters = parameterAbi<LumenReflectionTiledParams>(kLumenReflectionTiledABI), .debugName = label}, kernel, log);
    };
    const KernelEntry tiledEntries[]{
        {"lumenReflectionGenerateTiledRaysMain", "LumenReflectionGenerateTiledRays", generateTiledRays_},
        {"lumenReflectionClearTilesMain", "LumenReflectionClearTiles", clearTiles_},
        {"lumenReflectionTraceGDFTiledMain", "LumenReflectionTraceGDFRays", traceGDFRays_},
        {"lumenReflectionResolveTiledMain", "LumenReflectionResolveTiles", resolve_},
    };
    for (const auto& [entry, label, kernel] : tiledEntries) {
        if (auto result = tiled(entry, label, kernel); !result) { return result; }
    }
    if (optionEnabled("screenTraces")) {
        if (auto result = tiled("lumenReflectionTraceScreenTiledMain", "LumenReflectionTraceScreenTiles", traceScreenTiles_); !result) {
            return result;
        }
    }
    if (optionEnabled("detailTracing")) {
        return tiled("lumenReflectionTraceDetailTiledMain", "LumenReflectionTraceDetailRays", traceDetailRays_);
    }
    return {};
}

Result<> LumenReflectionPass::execute(RenderGraphExecutionContext& context)
{
    const auto* world = context.world();
    if (world && (world->environment().enabled ||
        world->worldEnvironment().source == environment::EnvironmentSource::PhysicalAtmosphere)) {
        spdlog::error("The opaque Lumen Reflection software path requires skylight to be disabled");
        return makeError(Error::Unsupported);
    }
    const auto* view = context.viewConstants();
    if (!view) {
        spdlog::error("Lumen Reflection requires the graph camera");
        return makeError(Error::InvalidArgument);
    }
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }
    const auto& output = context.outputTexture("color").desc();
    const auto* scene = world ? world->scene() : context.runtimeScene();
    const auto* globalHistory = context.historyResources();
    const bool dynamic = optionEnabled("dynamicReprojection");
    const bool screenTraces = optionEnabled("screenTraces");
    const uint64_t reprojectionRevision = globalHistory
        ? (dynamic ? globalHistory->motionReprojectionInvalidationRevision() :
            globalHistory->sunLightingReprojectionInvalidationRevision())
        : 0;
    const HistoryContent content{
        .reprojectionRevision = reprojectionRevision,
        .sceneIdentity = scene ? scene->resourceIdentity() : 0,
        .sceneLifetimeRevision = dynamic && scene ? scene->sceneGraph().lifetimeRevision() : 0,
        .sceneStructuralRevision = dynamic && scene ? scene->sceneGraph().structuralRevision() : 0,
        .sceneContentRevision = scene ? scene->contentRevision() : 0,
        .geometryTransformRevision = !dynamic && scene ? scene->geometryTransformRevision() : 0,
        .visibilityRevision = scene ? scene->visibilityRevision() : 0,
        .nonSunLightingRevision = world ? world->nonSunLightingRevision() : 0,
        .environmentRevision = world ? world->environmentRevision() : 0,
        .width = output.width,
        .height = output.height,
        .cameraCutSerial = view->frame[3],
    };
    const uint64_t frameIndex = context.frameIndex();
    const float sunIrradianceMax = world ? world->sunIrradianceMax() : 0.0f;
    history_.beginFrame(frameIndex);
    if (lastContent_ != content || lastFrame_ + 1 != frameIndex || view->frame[1] == 0 ||
        lumenGlobalSunLightingChanged(previousSunIrradianceMax_, sunIrradianceMax)) {
        history_.invalidateAll();
    }
    for (size_t index = 0; index < kHistoryNames.size(); ++index) {
        const TextureDesc desc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage,
            .format = kHistoryFormats[index],
            .width = content.width,
            .height = content.height,
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        auto result = history_.ensureTexture(kHistoryNames[index], desc, TextureViewDesc{.format = desc.format});
        if (!result) { return result; }
    }
    const auto currentLighting = history_.texture(kHistoryNames[0], HistorySlot::Current);
    const auto currentDepth = history_.texture(kHistoryNames[1], HistorySlot::Current);
    const auto currentAge = history_.texture(kHistoryNames[2], HistorySlot::Current);
    const auto previousLighting = history_.texture(kHistoryNames[0], HistorySlot::Previous);
    const auto previousDepth = history_.texture(kHistoryNames[1], HistorySlot::Previous);
    const auto previousAge = history_.texture(kHistoryNames[2], HistorySlot::Previous);
    const bool historyValid = previousLighting.valid && previousDepth.valid && previousAge.valid;

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    const auto sampled = [&](const char* name) { return writer.sampledImageHandle(context.inputTexture(name).view()); };
    const auto sampledOutput = [&](const char* name) { return writer.sampledImageHandle(context.outputTexture(name).view()); };
    const auto storage = [&](const char* name) { return writer.storageImageHandle(context.outputTexture(name).view()); };
    LumenReflectionParams parameters{
        .scene = bindLumenGlobalTrace(context, writer),
        .position = sampled("position"),
        .normal = sampled("normal"),
        .specular = sampled("specular"),
        .sceneColor = sampled("sceneColor"),
        .blueNoise = writer.sampledImageHandle(blueNoise_.get()),
        .outputReflections = storage("reflections"),
        .trace = storage("trace"),
        .color = storage("color"),
        .reflections = sampledOutput("reflections"),
        .camera = view->current,
        .width = output.width,
        .height = output.height,
        .clipmapCount = *clipmapCount,
        .frameIndex = view->frame[0],
        .maxTraceDistance = maxTraceDistance_,
        .maxRoughness = maxRoughness_,
        .outputRayData = storage("rayData"),
        .outputRayHitDistance = storage("rayHitDistance"),
        .outputResolvedReflections = storage("resolvedReflections"),
        .outputResolvedHitDistance = storage("resolvedHitDistance"),
        .rayData = sampledOutput("rayData"),
        .rayHitDistance = sampledOutput("rayHitDistance"),
        .bilateralReflections = sampledOutput("bilateralReflections"),
    };
    if (screenTraces) {
        parameters.screen = {writer.buffer(context.inputBuffer("closestHZB").buffer()), sampled("previousDepth"),
            sampled("previousColor"), writer.buffer(context.inputBuffer("traceView").buffer())};
    }
    std::optional<LumenMeshSDFGrid::Frame> meshGridFrame;
    if (optionEnabled("detailTracing")) {
        auto grid = meshGrid_.prepare(context, writer, meshFields_, meshSDFTraceDistance_, kLumenCardTraceEndDistance);
        if (!grid) { return makeError(grid.error()); }
        meshGridFrame.emplace(std::move(*grid));
        parameters.detail = meshGridFrame->bindings;
    }
    const auto encoded = writer.encode(parameters, kLumenReflectionABI);
    if (!encoded) { return makeError(encoded.error()); }

    const auto resolvedReflections = sampledOutput("resolvedReflections");
    const auto resolvedHitDistance = sampledOutput("resolvedHitDistance");
    const LumenReflectionTemporalParams temporalParams{
        .position = parameters.position,
        .normal = parameters.normal,
        .specular = parameters.specular,
        .previousPosition = dynamic ? sampled("previousPosition") : parameters.position,
        .resolvedReflections = resolvedReflections,
        .resolvedHitDistance = resolvedHitDistance,
        .previousLighting = historyValid ? writer.sampledImageHandle(previousLighting.view) : resolvedReflections,
        .previousDepth = historyValid ? writer.sampledImageHandle(previousDepth.view) : resolvedHitDistance,
        .previousAge = historyValid ? writer.sampledImageHandle(previousAge.view) : resolvedHitDistance,
        .temporalReflections = storage("temporalReflections"),
        .temporalState = storage("temporalState"),
        .currentLighting = writer.storageImageHandle(currentLighting.view),
        .currentDepth = writer.storageImageHandle(currentDepth.view),
        .currentAge = writer.storageImageHandle(currentAge.view),
        .width = parameters.width,
        .height = parameters.height,
        .currentCamera = view->current,
        .previousCamera = view->previous,
        .historyValid = historyValid,
        .dynamicReprojection = dynamic,
        .frameIndex = parameters.frameIndex,
    };
    const auto encodedTemporal = writer.encode(temporalParams, kLumenReflectionTemporalABI);
    if (!encodedTemporal) { return makeError(encodedTemporal.error()); }

    const LumenReflectionBilateralParams bilateralParams{
        .camera = parameters.camera,
        .position = parameters.position,
        .normal = parameters.normal,
        .specular = parameters.specular,
        .temporalReflections = sampledOutput("temporalReflections"),
        .currentLighting = writer.sampledImageHandle(currentLighting.view),
        .currentAge = writer.sampledImageHandle(currentAge.view),
        .bilateralReflections = storage("bilateralReflections"),
        .width = parameters.width,
        .height = parameters.height,
        .frameIndex = parameters.frameIndex,
    };
    const auto encodedBilateral = writer.encode(bilateralParams, kLumenReflectionBilateralABI);
    if (!encodedBilateral) { return makeError(encodedBilateral.error()); }

    using Access = RenderGraphResourceAccess;
    std::vector<RenderGraphStageUse> traceUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"specular", Access::TextureSampleRead}, {"pageTable", Access::TextureSampleRead},
        {"pageAtlas", Access::TextureSampleRead}, {"coarseMip", Access::TextureSampleRead},
        {"domain", Access::BufferStorageRead}, {"objectGrid", Access::BufferStorageRead},
        {"objectGridScene", Access::BufferStorageRead}, {"cards", Access::BufferStorageRead},
        {"cardScene", Access::BufferStorageRead}, {"cardDepth", Access::TextureSampleRead},
        {"cardLighting", Access::TextureSampleRead}, {"reflections", Access::TextureStorageWrite},
        {"trace", Access::TextureStorageWrite}, {"rayData", Access::TextureStorageWrite},
        {"rayHitDistance", Access::TextureStorageWrite},
    };
    if (screenTraces) {
        traceUses.insert(traceUses.end(), {{"closestHZB", Access::BufferStorageRead},
            {"previousDepth", Access::TextureSampleRead}, {"previousColor", Access::TextureSampleRead},
            {"traceView", Access::BufferStorageRead}});
    }
    const RenderGraphStageUse resolveUses[]{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"specular", Access::TextureSampleRead}, {"reflections", Access::TextureSampleRead},
        {"rayData", Access::TextureSampleRead}, {"rayHitDistance", Access::TextureSampleRead},
        {"resolvedReflections", Access::TextureStorageWrite}, {"resolvedHitDistance", Access::TextureStorageWrite},
    };
    std::vector<RenderGraphStageUse> temporalUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"specular", Access::TextureSampleRead}, {"resolvedReflections", Access::TextureSampleRead},
        {"resolvedHitDistance", Access::TextureSampleRead}, {"temporalReflections", Access::TextureStorageWrite},
        {"temporalState", Access::TextureStorageWrite}, {"currentLighting", Access::TextureStorageWrite},
        {"currentDepth", Access::TextureStorageWrite}, {"currentAge", Access::TextureStorageWrite},
    };
    if (dynamic) { temporalUses.push_back({"previousPosition", Access::TextureSampleRead}); }
    const auto import = [](std::string_view name, const HistoryTextureRef& texture) {
        return RenderGraphTextureImport{name, texture.texture, texture.view, texture.state, ResourceState::ShaderRead};
    };
    std::vector<RenderGraphTextureImport> textures{
        import("currentLighting", currentLighting), import("currentDepth", currentDepth), import("currentAge", currentAge),
    };
    if (historyValid) {
        temporalUses.insert(temporalUses.end(), {{"previousLighting", Access::TextureSampleRead},
            {"previousReflectionDepth", Access::TextureSampleRead}, {"previousAge", Access::TextureSampleRead}});
        textures.insert(textures.end(), {import("previousLighting", previousLighting), import("previousReflectionDepth", previousDepth),
            import("previousAge", previousAge)});
    }
    const RenderGraphStageUse bilateralUses[]{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"specular", Access::TextureSampleRead}, {"temporalReflections", Access::TextureSampleRead},
        {"currentLighting", Access::TextureSampleRead}, {"currentAge", Access::TextureSampleRead},
        {"bilateralReflections", Access::TextureStorageWrite},
    };
    const RenderGraphStageUse compositeUses[]{
        {"sceneColor", Access::TextureSampleRead}, {"bilateralReflections", Access::TextureSampleRead},
        {"specular", Access::TextureSampleRead}, {"color", Access::TextureStorageWrite},
    };
    const uint32_t groupsX = reflectionTileCount(parameters.width);
    const uint32_t groupsY = reflectionTileCount(parameters.height);
    std::vector<RenderGraphStage> stages;
    const auto executeWithDenoising = [&] {
        stages.insert(stages.end(), {
            {"Reproject and accumulate reflections", temporalUses, [&](CommandBuffer& cmd) {
                return temporal_.dispatch(cmd, *encodedTemporal, groupsX, groupsY);
            }},
            {"Bilateral filter reflections", bilateralUses, [&](CommandBuffer& cmd) {
                return bilateral_.dispatch(cmd, *encodedBilateral, groupsX, groupsY);
            }},
            {"Composite reflections", compositeUses, [&](CommandBuffer& cmd) {
                return composite_.dispatch(cmd, *encoded, groupsX, groupsY);
            }},
        });
        return context.executeStages(stages, {}, textures);
    };
    Result<> result;
    if (optionEnabled("classifiedTracing")) {
        const auto work = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
        const LumenReflectionTiledParams tiledParams{parameters, work("reflectionActiveTiles"),
            work("reflectionClearTiles"), work("reflectionWorkCounts"), work("reflectionRayList"), kReflectionMaxDispatchX};
        const auto tiled = writer.encode(tiledParams, kLumenReflectionTiledABI);
        if (!tiled) { return makeError(tiled.error()); }
        const auto schedule = writer.encode(LumenReflectionScheduleParams{
            parameters.position, parameters.specular, parameters.rayData, parameters.rayHitDistance, sampledOutput("trace"),
            work("reflectionTileMask"), tiledParams.activeTiles, tiledParams.clearTiles, tiledParams.workCounts,
            tiledParams.rayList, work("reflectionDispatchArgs"), parameters.width, parameters.height, kReflectionMaxDispatchX,
            parameters.maxRoughness, meshSDFTraceDistance_, kLumenCardTraceEndDistance, parameters.maxTraceDistance},
            kLumenReflectionScheduleABI);
        if (!schedule) { return makeError(schedule.error()); }
        const auto indirect = [&](ComputeKernel& kernel, const EncodedParameters& args, ReflectionDispatch kind,
                                  CommandBuffer& cmd) -> Result<> {
            const auto dispatch = context.outputBuffer("reflectionDispatchArgs").buffer()->slice(
                {uint32_t(kind) * kReflectionDispatchBytes, kReflectionDispatchBytes});
            if (!dispatch) { return makeError(dispatch.error()); }
            return kernel.dispatchIndirect(cmd, args, *dispatch);
        };

        constexpr Access read = Access::BufferStorageRead;
        constexpr Access write = Access::BufferStorageWrite;
        constexpr Access readWrite = Access::BufferStorageReadWrite;
        constexpr Access sample = Access::TextureSampleRead;
        constexpr Access store = Access::TextureStorageWrite;
        constexpr RenderGraphStageUse dispatchArgs{"reflectionDispatchArgs", Access::BufferIndirectRead};
        constexpr RenderGraphStageUse workCounts{"reflectionWorkCounts", read};
        constexpr RenderGraphStageUse activeTiles{"reflectionActiveTiles", read};
        const RenderGraphStageUse initUses[]{{"reflectionWorkCounts", write}};
        const RenderGraphStageUse markUses[]{{"position", sample}, {"specular", sample}, {"reflectionTileMask", write}};
        const RenderGraphStageUse listUses[]{{"reflectionTileMask", read}, {"reflectionWorkCounts", readWrite},
            {"reflectionActiveTiles", write}, {"reflectionClearTiles", write}};
        const RenderGraphStageUse setupUses[]{workCounts, {"reflectionDispatchArgs", write}};
        const RenderGraphStageUse generateUses[]{dispatchArgs, workCounts, activeTiles,
            {"position", sample}, {"normal", sample}, {"specular", sample},
            {"reflections", store}, {"trace", store}, {"rayData", store}, {"rayHitDistance", store}};
        const RenderGraphStageUse clearUses[]{dispatchArgs, workCounts, {"reflectionClearTiles", read},
            {"reflections", store}, {"trace", store}, {"rayData", store}, {"rayHitDistance", store},
            {"resolvedReflections", store}, {"resolvedHitDistance", store}};
        const RenderGraphStageUse screenUses[]{dispatchArgs, workCounts, activeTiles,
            {"position", sample}, {"normal", sample}, {"specular", sample}, {"rayData", sample},
            {"closestHZB", read}, {"previousDepth", sample}, {"previousColor", sample}, {"traceView", read},
            {"reflections", store}, {"trace", store}, {"rayHitDistance", store}};
        const RenderGraphStageUse compactUses[]{dispatchArgs, activeTiles, {"reflectionWorkCounts", readWrite},
            {"position", sample}, {"rayData", sample}, {"rayHitDistance", sample}, {"trace", sample},
            {"reflectionRayList", write}};
        const RenderGraphStageUse traceListUses[]{dispatchArgs, workCounts, {"reflectionRayList", read},
            {"position", sample}, {"specular", sample}, {"rayData", sample},
            {"reflections", store}, {"trace", store}, {"rayHitDistance", Access::TextureStorageReadWrite},
            {"pageTable", sample}, {"pageAtlas", sample}, {"coarseMip", sample}, {"domain", read},
            {"objectGrid", read}, {"objectGridScene", read}, {"cards", read}, {"cardScene", read},
            {"cardDepth", sample}, {"cardLighting", sample}};
        std::vector<RenderGraphStageUse> detailUses(std::begin(traceListUses), std::end(traceListUses));
        std::vector<RenderGraphStageUse> tiledResolveUses(std::begin(resolveUses), std::end(resolveUses));
        tiledResolveUses.insert(tiledResolveUses.end(), {dispatchArgs, workCounts, activeTiles});
        stages.insert(stages.end(), {
            {"Initialize reflection work counts", initUses, [&](CommandBuffer& cmd) {
                return initWork_.dispatch(cmd, *schedule, 1, 1);
            }},
            {"Mark reflection tiles", markUses, [&](CommandBuffer& cmd) {
                return markTiles_.dispatch(cmd, *schedule, groupsX, groupsY);
            }},
            {"Build active and clear reflection tile lists", listUses, [&](CommandBuffer& cmd) {
                return buildTileLists_.dispatch(cmd, *schedule, reflectionTileCount(groupsX), reflectionTileCount(groupsY));
            }},
            {"Set up reflection tile dispatches", setupUses, [&](CommandBuffer& cmd) {
                return setupTileArgs_.dispatch(cmd, *schedule, 1, 1);
            }},
            {"Generate reflection rays in active tiles", generateUses, [&](CommandBuffer& cmd) {
                return indirect(generateTiledRays_, *tiled, ReflectionDispatch::ActiveTiles, cmd);
            }},
            {"Clear unused reflection tiles", clearUses, [&](CommandBuffer& cmd) {
                return indirect(clearTiles_, *tiled, ReflectionDispatch::ClearTiles, cmd);
            }},
        });
        if (screenTraces) {
            stages.push_back({"Trace screen reflections in active tiles", screenUses, [&](CommandBuffer& cmd) {
                return indirect(traceScreenTiles_, *tiled, ReflectionDispatch::ActiveTiles, cmd);
            }});
        }
        if (meshGridFrame) {
            LumenMeshSDFGrid::appendTraceUses(detailUses);
            meshGrid_.appendStages(stages, *meshGridFrame);
            stages.insert(stages.end(), {
                {"Compact unfinished reflections for Mesh SDF", compactUses, [&](CommandBuffer& cmd) {
                    return indirect(compactDetailRays_, *schedule, ReflectionDispatch::CompactRays, cmd);
                }},
                {"Set up reflection Mesh SDF dispatch", setupUses, [&](CommandBuffer& cmd) {
                    return setupDetailArgs_.dispatch(cmd, *schedule, 1, 1);
                }},
                {"Trace compacted reflections through Mesh SDF", detailUses, [&](CommandBuffer& cmd) {
                    return indirect(traceDetailRays_, *tiled, ReflectionDispatch::DetailRays, cmd);
                }},
            });
        }
        stages.insert(stages.end(), {
            {"Compact remaining reflections for GDF", compactUses, [&](CommandBuffer& cmd) {
                return indirect(compactGDFRays_, *schedule, ReflectionDispatch::CompactRays, cmd);
            }},
            {"Set up reflection GDF dispatch", setupUses, [&](CommandBuffer& cmd) {
                return setupGDFArgs_.dispatch(cmd, *schedule, 1, 1);
            }},
            {"Trace compacted reflections through GDF", traceListUses, [&](CommandBuffer& cmd) {
                return indirect(traceGDFRays_, *tiled, ReflectionDispatch::GDFRays, cmd);
            }},
            {"Resolve active reflection tiles", tiledResolveUses, [&](CommandBuffer& cmd) {
                return indirect(resolve_, *tiled, ReflectionDispatch::ActiveTiles, cmd);
            }},
        });
        result = executeWithDenoising();
    } else {
        if (meshGridFrame) {
            LumenMeshSDFGrid::appendTraceUses(traceUses);
            meshGrid_.appendStages(stages, *meshGridFrame);
        }
        stages.insert(stages.end(), {
            {"Trace software reflections", traceUses, [&](CommandBuffer& cmd) {
                return trace_.dispatch(cmd, *encoded, groupsX, groupsY);
            }},
            {"Resolve reflections", resolveUses, [&](CommandBuffer& cmd) {
                return resolve_.dispatch(cmd, *encoded, groupsX, groupsY);
            }},
        });
        result = executeWithDenoising();
    }
    if (!result) { return result; }
    for (const auto* name : kHistoryNames) {
        result = history_.publishTextureState(commands, name, HistorySlot::Current, ResourceState::ShaderRead, true);
        if (!result) { return result; }
    }
    lastContent_ = content;
    lastFrame_ = frameIndex;
    previousSunIrradianceMax_ = sunIrradianceMax;
    return {};
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenReflectionPass()
{
    return std::make_unique<LumenReflectionPass>();
}

} // namespace metallic::render::builtin_pass
