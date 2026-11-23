#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenLightingHistory.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <cmath>
#include <optional>
#include <string_view>
#include <vector>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kLumenReflectionABI = 0x4c554d454e050002ull;
constexpr uint64_t kLumenReflectionTemporalABI = 0x4c554d454e060001ull;
constexpr uint64_t kLumenReflectionBilateralABI = 0x4c554d454e070002ull;
constexpr char kShaderModule[] = "Features/DistanceField/LumenReflection";
constexpr std::array kHistoryNames{"lighting", "depth", "age"};
constexpr std::array kHistoryFormats{Format::RGBA16Sfloat, Format::R32Sfloat, Format::R8Unorm};

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
};
static_assert(offsetof(LumenReflectionParams, camera) == 80 && offsetof(LumenReflectionParams, width) == 160 &&
    offsetof(LumenReflectionParams, maxTraceDistance) == 176 &&
    offsetof(LumenReflectionParams, outputRayData) == 184 && offsetof(LumenReflectionParams, rayData) == 200 &&
    sizeof(LumenReflectionParams) == 224);

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
    bool dynamicReprojection() const
    {
        const auto value = properties().find("dynamicReprojection");
        return value != properties().end() && *value == true;
    }

    Device* device_ = nullptr;
    std::unique_ptr<TextureView> blueNoise_;
    ComputeKernel trace_;
    ComputeKernel resolve_;
    ComputeKernel temporal_;
    ComputeKernel bilateral_;
    ComputeKernel composite_;
    float maxTraceDistance_ = 200.0f;
    float maxRoughness_ = 0.4f;
    // Preserve reprojectable history across ordinary camera motion, as diffuse gather does.
    HistoryResourceManager history_;
    std::optional<HistoryContent> lastContent_;
    uint64_t lastFrame_ = 0;
    float previousSunIrradianceMax_ = 0.0f;
};

RenderPassReflection LumenReflectionPass::reflect(const RenderGraphCompileContext&) const
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
    if (dynamicReprojection()) {
        reflection.addTextureInput("previousPosition", "Previous world position of the current receiver and validity")
            .sampledRead().format = Format::RGBA32Sfloat;
    }
    reflection.addTextureInput("sceneColor", "Scene-linear color before specular reflections")
        .sampledRead().format = Format::RGBA32Sfloat;

    auto& reflections = reflection.addTextureOutput("reflections", "Raw scene-linear reflection radiance and roughness fade")
        .storageWrite().stageAccess(Access::TextureSampleRead);
    reflections.format = Format::RGBA16Sfloat;
    reflections.colorEncoding = DisplayColorEncoding::SceneLinear;
    reflection.addTextureOutput("trace", "Hit distance in meters, GDF classification, clipmap index and card cache classification")
        .storageWrite().format = Format::RGBA32Sfloat;
    reflection.addTextureOutput("rayData", "Reflection ray direction and reciprocal PDF")
        .storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite)
        .format = Format::RGBA16Sfloat;
    reflection.addTextureOutput("rayHitDistance", "Encoded reflection ray distance in centimeters")
        .storageWrite().stageAccess(Access::TextureSampleRead).transient(Initialization::FullOverwrite)
        .format = Format::R16Sfloat;
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
    const auto dynamic = properties().find("dynamicReprojection");
    if (dynamic != properties().end() && !dynamic->is_boolean()) {
        log = "Lumen Reflection dynamicReprojection must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    try {
        // UE 5.7.4 FPostProcessSettings defaults to 20000 cm and roughness 0.4.
        maxTraceDistance_ = properties().value("maxTraceDistance", 200.0f);
        maxRoughness_ = properties().value("maxRoughness", 0.4f);
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
    lastContent_.reset();
    if (auto result = history_.initialize(*device_); !result) { return result; }
    auto blueNoise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Vec2, log);
    if (!blueNoise) { return makeError(blueNoise.error()); }
    blueNoise_ = std::move(*blueNoise);
    auto& registry = ShaderRegistry::instance();
    auto result = registry.getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "lumenReflectionTraceMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenReflectionParams>(kLumenReflectionABI), .debugName = "LumenReflectionTrace"},
        trace_, log);
    if (!result) { return result; }
    result = registry.getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "lumenReflectionResolveMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenReflectionParams>(kLumenReflectionABI), .debugName = "LumenReflectionResolve"},
        resolve_, log);
    if (!result) { return result; }
    result = registry.getComputeKernel(*device_,
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

Result<> LumenReflectionPass::execute(RenderGraphExecutionContext& context)
{
    const auto* world = context.world();
    if (world && (world->environment().enabled ||
        world->worldEnvironment().source == environment::EnvironmentSource::PhysicalAtmosphere)) {
        spdlog::error("The opaque Lumen Reflection GDF branch requires skylight to be disabled");
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
    const bool dynamic = dynamicReprojection();
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
    const LumenReflectionParams parameters{
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
    const RenderGraphStageUse traceUses[]{
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
            {"previousDepth", Access::TextureSampleRead}, {"previousAge", Access::TextureSampleRead}});
        textures.insert(textures.end(), {import("previousLighting", previousLighting), import("previousDepth", previousDepth),
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
    const uint32_t groupsX = (parameters.width + 7u) / 8u;
    const uint32_t groupsY = (parameters.height + 7u) / 8u;
    const RenderGraphStage stages[]{
        {"Trace software reflections", traceUses, [&](CommandBuffer& cmd) {
            return trace_.dispatch(cmd, *encoded, groupsX, groupsY);
        }},
        {"Resolve reflections", resolveUses, [&](CommandBuffer& cmd) {
            return resolve_.dispatch(cmd, *encoded, groupsX, groupsY);
        }},
        {"Reproject and accumulate reflections", temporalUses, [&](CommandBuffer& cmd) {
            return temporal_.dispatch(cmd, *encodedTemporal, groupsX, groupsY);
        }},
        {"Bilateral filter reflections", bilateralUses, [&](CommandBuffer& cmd) {
            return bilateral_.dispatch(cmd, *encodedBilateral, groupsX, groupsY);
        }},
        {"Composite reflections", compositeUses, [&](CommandBuffer& cmd) {
            return composite_.dispatch(cmd, *encoded, groupsX, groupsY);
        }},
    };
    auto result = context.executeStages(stages, {}, textures);
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
