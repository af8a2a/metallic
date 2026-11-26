#include "Runtime/Render/Environment/VolumetricFogParameters.h"
#include "Runtime/Render/Environment/LocalFogVolumePacking.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/Shadow/VirtualShadowMapWorld.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

namespace metallic::render::builtin_pass {
namespace {

using Access = RenderGraphResourceAccess;
using StageUse = RenderGraphStageUse;

struct FogView {
    ViewCameraConstants camera;
    FogFloat4 gridZ, distances;
    FogUInt4 grid;
    uint32_t enabled;
};

Result<> compileFogKernel(Device& device, const char* entry, ComputeKernel& kernel, std::string& log)
{
    return ShaderRegistry::instance().getComputeKernel(device,
        {.moduleName = "Features/Environment/VolumetricFog", .entryPointName = entry,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<VolumetricFogParams>(kVolumetricFogABI), .debugName = entry}, kernel, log);
}

// RenderUtils.h::CalculateGridZParams. N, F and the near offset are centimeters.
FogFloat4 gridZParams(float nearPlane, float farPlane, uint32_t sizeZ)
{
    const double N = nearPlane + .095 * 100;
    const double F = farPlane;
    constexpr double S = 32.0;
    const double O = (F - N * std::exp2(sizeZ / S)) / (F - N);
    const double B = (1 - O) / N;
    return {float(B), float(O), float(S), 0};
}

void reflectMetadata(RenderGraphField& field, uint32_t size)
{
    field.buffer(size, size).shaderRead();
    field.memoryLocation = MemoryLocation::HostUpload;
}

template<typename T>
Result<> readMetadata(BufferHandle buffer, T& value)
{
    if (!buffer.valid() || buffer.desc().size != sizeof(T) || buffer.desc().memoryLocation != MemoryLocation::HostUpload) {
        return makeError(Error::InvalidArgument);
    }
    const auto* mapped = buffer.buffer()->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(&value, mapped, sizeof(T));
    buffer.buffer()->unmap();
    return {};
}

class VolumetricFogPass final : public ComputePass {
public:
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"depth", "rasterInfo", "accelerationStructure"}};
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{EnvironmentLightingSubsystem::kSubsystemId, GPUSceneSubsystem::kSubsystemId};
        return required;
    }
    Result<> declareResourceRequests(const RenderGraphResourceRequestContext& context,
        RenderGraphResourceRequestBuilder& builder, std::string& log) override;
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> prepare(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;
    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override;
    QueueType queueType() const override { return QueueType::Graphics; }

private:
    Device* device_ = nullptr;
    ComputeKernel material_, shadow_, lighting_, integration_, depth_, localTileCulling_;
    HistoryResourceManager history_;
    struct Previous {
        ViewCameraConstants camera{};
        environment::VolumetricFogState fog;
        FogFloat4 gridZ{};
        uint64_t frame = UINT64_MAX, sceneIdentity = 0, reprojectionRevision = 0;
        uint32_t cutSerial = 0, viewIndex = 0, viewGeneration = 0;
    } previous_;
};

RenderPassReflection VolumetricFogPass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    auto& depth = reflection.addTextureInput("depth", "Main raster reversed-Z depth").sampledRead();
    depth.format = Format::D32Sfloat;
    depth.matchOutputExtent = false;
    reflectMetadata(reflection.addBufferInput("rasterInfo", "Camera that produced the depth"), sizeof(VisibilityBufferFrameInfo));
    reflection.addBufferInput("worldShadowFaces", "Optional world-space VSM faces for punctual fog lights")
        .buffer(0, sizeof(VirtualShadowMapWorldFace)).bufferLayout("VirtualShadowMapWorldFace")
        .storageRead().setOptional().sharedProducerInput = "rasterInfo";
    const auto tile = properties().value("gridPixelSize", 16u);
    const auto slices = properties().value("gridSizeZ", 64u);
    const auto output = [&](const char* name, Format format, bool volume, bool sampled) {
        auto& field = reflection.addTextureOutput(name);
        if (volume) { field.texture3D(0, 0, slices); }
        field.matchOutputExtent = false;
        field.extentSourceInput = "depth";
        field.extentDivisor = tile;
        field.storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        field.stageAccess(Access::TextureTransferWrite);
        field.format = format;
        if (sampled) { field.stageAccess(Access::TextureSampleRead); }
    };
    output("material", Format::RGBA16Sfloat, true, true);
    if (properties().value("localFogVolumes", false)) {
        output("emissive", Format::RGBA16Sfloat, true, true);
        auto& tiles = reflection.addTextureOutput("localFogTiles", "Count and sorted sphere indices per 128-pixel tile");
        tiles.matchOutputExtent = false;
        tiles.extentSourceInput = "depth";
        tiles.extentDivisor = kLocalFogTilePixels;
        tiles.layerCount = kLocalFogMaxInstances + 1;
        tiles.format = Format::R8Uint;
        tiles.storageWrite().stageAccess(Access::TextureSampleRead).stageAccess(Access::TextureTransferWrite);
        // Layers beyond a tile's count are unused, as in LocalFogVolumeTiledCullingCS.
        tiles.transient(RenderGraphInitialization::Unknown);
    }
    output("scattering", Format::RGBA16Sfloat, true, true);
    output("integrated", Format::RGBA16Sfloat, true, false);
    reflection.addAccelerationStructureInput("accelerationStructure", "Scene TLAS when directional fog shadows are enabled")
        .accelerationStructureRead().setOptional();
    if (properties().value("rayTracedShadows", false)) {
        output("shadow", Format::R16Sfloat, true, true);
    }
    if (properties().value("conservativeDepth", true)) {
        output("conservativeDepth", Format::R16Sfloat, false, true);
    }
    reflectMetadata(reflection.addBufferOutput("view", "Camera, depth mapping and extents of the integrated volume"), sizeof(FogView));
    return reflection;
}

std::vector<RenderGraphRuntimeSetting> VolumetricFogPass::runtimeSettings() const
{
    std::vector settings{
        runtimeIntSetting("gridPixelSize", "Pixels per froxel", 16, 1, 128),
        runtimeIntSetting("gridSizeZ", "Depth slices", 64, 2, 256),
        runtimeBoolSetting("rayTracedShadows", "Ray-traced directional shadows", false),
        runtimeBoolSetting("conservativeDepth", "Conservative depth", true),
        runtimeBoolSetting("localFogVolumes", "Local fog volumes", false),
    };
    for (auto& setting : settings) { setting.rebuildGraph = true; }
    settings.push_back(runtimeFloatSetting("localFogStartDistanceMeters", "Local fog start distance (m)", 20.0f, 0.1f, 10000.0f, true));
    settings.push_back(runtimeBoolSetting("temporalReprojection", "Temporal reprojection", true, true));
    settings.push_back(runtimeBoolSetting("jitter", "Sub-voxel jitter", true, true));
    return settings;
}

Result<> VolumetricFogPass::prepare(const RenderGraphCompileContext& context, std::string& log)
{
    const auto tile = properties().value("gridPixelSize", 16);
    const auto slices = properties().value("gridSizeZ", 64);
    if (!context.device || tile < 1 || tile > 128 || slices < 2 || slices > 256) {
        log = "VolumetricFogPass requires a device, gridPixelSize in [1,128] and gridSizeZ in [2,256]";
        return makeError(Error::InvalidArgument);
    }
    // A settings rebuild can reuse this pass instance and introduce an optional shader.
    if (material_.valid() && ((properties().value("rayTracedShadows", false) && !shadow_.valid()) ||
        (properties().value("conservativeDepth", true) && !depth_.valid()) ||
        (properties().value("localFogVolumes", false) && !localTileCulling_.valid()))) {
        return compile(context, log);
    }
    return {};
}

Result<> VolumetricFogPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    if (properties().value("localFogVolumes", false)) {
        if (auto r = compileFogKernel(*device_, "localFogTiledCulling", localTileCulling_, log); !r) { return r; }
    }
    if (properties().value("rayTracedShadows", false)) {
        if (!device_->capabilities().rayQuery || !device_->capabilities().rayTracingAccelerationStructure) {
            log = "Directional fog shadows require ray query and ray tracing acceleration structures";
            return makeError(Error::Unsupported);
        }
        if (auto r = compileFogKernel(*device_, "directionalShadow", shadow_, log); !r) { return r; }
    }
    if (properties().value("conservativeDepth", true)) {
        if (auto r = compileFogKernel(*device_, "conservativeDepth", depth_, log); !r) { return r; }
    }
    if (auto r = compileFogKernel(*device_, "materialSetup", material_, log); !r) { return r; }
    if (auto r = compileFogKernel(*device_, "lightScattering", lighting_, log); !r) { return r; }
    if (auto r = compileFogKernel(*device_, "finalIntegration", integration_, log); !r) { return r; }
    previous_ = {};
    return history_.initialize(*device_);
}

Result<> VolumetricFogPass::declareResourceRequests(const RenderGraphResourceRequestContext& context,
    RenderGraphResourceRequestBuilder& builder, std::string& log)
{
    const auto environment = resolveWorldEnvironment(context.runtimeScene(), context.world());
    if (!environment.volumetricFog.enabled || environment.source != environment::EnvironmentSource::PhysicalAtmosphere ||
        !environment.celestial[0].enabled) { return {}; }
    std::array<double, 3> observer{};
    if (const auto* view = context.viewConstants()) {
        observer = {view->current.eye[0], view->current.eye[1], view->current.eye[2]};
    }
    const auto declared = context.subsystem<EnvironmentLightingSubsystem>()->declareEnvironmentRequest(context, builder,
        {.environment = environment, .observerWorldMetres = observer, .viewId = context.viewKey(), .needCelestial = false}, log);
    return declared ? Result<>{} : makeError(declared.error());
}

Result<> VolumetricFogPass::execute(RenderGraphExecutionContext& context)
{
    const auto depth = context.inputTexture("depth");
    const auto material = context.outputTexture("material");
    const auto scattering = context.outputTexture("scattering");
    const auto integrated = context.outputTexture("integrated");
    VisibilityBufferFrameInfo raster;
    if (auto r = readMetadata(context.inputBuffer("rasterInfo"), raster); !r) { return r; }
    if (!depth.valid() || raster.frameIndex != context.frameIndex() || raster.width != depth.desc().width ||
        raster.height != depth.desc().height || !context.runtimeScene() ||
        raster.sceneIdentity != context.runtimeScene()->resourceIdentity()) {
        spdlog::error("VolumetricFogPass requires depth and raster metadata from the current scene and frame");
        return makeError(Error::InvalidArgument);
    }
    const auto env = resolveWorldEnvironment(context.runtimeScene(), context.world());
    const auto& fog = env.volumetricFog;
    const float nearCm = std::max(raster.clipOrtho[0], fog.startDistanceMeters) * 100.0f;
    const float farCm = fog.distanceMeters * 100.0f;
    const bool active = fog.enabled && farCm > nearCm + 9.5f;
    const auto& extent = integrated.desc();
    const bool conservative = context.properties().value("conservativeDepth", true);
    const bool temporal = context.properties().value("temporalReprojection", true) && active;
    const bool rayTraced = context.properties().value("rayTracedShadows", false);
    const bool localVolumes = context.properties().value("localFogVolumes", false);
    const auto sun = buildCelestialLightRecords(env)[0];

    VolumetricFogParams p{};
    std::copy_n(raster.eye, 4, p.current.eye);
    std::copy_n(raster.center, 4, p.current.center);
    std::copy_n(raster.upProjection, 4, p.current.upProjection);
    std::copy_n(raster.viewport, 4, p.current.viewport);
    std::copy_n(raster.clipOrtho, 4, p.current.clipOrtho);
    p.grid = {extent.width, extent.height, extent.depth, context.properties().value("gridPixelSize", 16u)};
    // An empty interval publishes identity fog. Use a finite mapping to fill every output.
    p.gridZ = active ? gridZParams(nearCm, farCm, extent.depth) : gridZParams(1.0f, 6000.0f, extent.depth);
    if (active) {
        p.fogLayer0 = {fog.density / 1000.0f, fog.heightFalloff / 1000.0f, fog.heightMeters * 100.0f, 0};
        p.fogLayer1 = {fog.secondDensity / 1000.0f, fog.secondHeightFalloff / 1000.0f,
            (fog.heightMeters + fog.secondHeightOffsetMeters) * 100.0f, 0};
        const auto albedo = color::fromLinearRec709({fog.albedo.x, fog.albedo.y, fog.albedo.z});
        p.albedoExtinction = {albedo[0], albedo[1], albedo[2], fog.extinctionScale};
        const auto emissive = color::fromLinearRec709({fog.emissive.x, fog.emissive.y, fog.emissive.z});
        p.emissive = {emissive[0] / 10000.0f, emissive[1] / 10000.0f, emissive[2] / 10000.0f, 0};
        p.lightDirectionPhase = {-sun.direction[0], -sun.direction[1], -sun.direction[2], fog.scatteringDistribution};
        p.lightColorIntensity = {sun.irradiance[0], sun.irradiance[1], sun.irradiance[2],
            (sun.flags & kGPUCelestialLightEnabled) ? fog.directionalScatteringIntensity : 0.0f};
    }
    p.distances = {fog.startDistanceMeters * 100.0f, fog.nearFadeInDistanceMeters > 0.0f ?
        1.0f / (fog.nearFadeInDistanceMeters * 100.0f) : 100000000.0f, 0.0f,
        context.properties().value("jitter", true) && temporal ? 1.0f : 0.0f};
    p.temporal = {uint32_t(context.frameIndex()), temporal, conservative, 4u};

    const FogView fogView{p.current, p.gridZ, p.distances, p.grid, active};
    auto metadata = context.outputBuffer("view");
    void* mapped = metadata.buffer()->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, &fogView, sizeof(fogView));
    metadata.buffer()->flush({0, sizeof(fogView)});
    metadata.buffer()->unmap();
    if (!active) {
        std::vector<StageUse> outputs{{"material", Access::TextureTransferWrite}, {"scattering", Access::TextureTransferWrite},
            {"integrated", Access::TextureTransferWrite}};
        if (rayTraced) { outputs.push_back({"shadow", Access::TextureTransferWrite}); }
        if (conservative) { outputs.push_back({"conservativeDepth", Access::TextureTransferWrite}); }
        if (localVolumes) {
            outputs.push_back({"emissive", Access::TextureTransferWrite});
            outputs.push_back({"localFogTiles", Access::TextureTransferWrite});
        }
        const RenderGraphStage stages[]{{"Clear inactive fog", outputs, [&](CommandBuffer& cmd) -> Result<> {
            for (const auto& output : outputs) {
                const ColorValue clear = output.resource == "integrated" ? ColorValue{0, 0, 0, 1} :
                    output.resource == "shadow" ? ColorValue{1, 0, 0, 0} : ColorValue{0, 0, 0, 0};
                if (auto r = cmd.clearColorTexture(*context.outputTexture(output.resource).texture(),
                        TextureLayout::TransferDestination, clear); !r) { return r; }
            }
            return {};
        }}};
        return context.executeStages(stages);
    }

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    p.sampler = writer.samplerHandle({});
    p.depth = writer.sampledImageHandle(depth.view());
    p.materialOut = writer.storageImageHandle(material.view());
    p.material = writer.sampledImageHandle(material.view());
    p.scatteringOut = writer.storageImageHandle(scattering.view());
    p.scattering = writer.sampledImageHandle(scattering.view());
    p.integratedOut = writer.storageImageHandle(integrated.view());
    const auto* grid = context.subsystem<GPUSceneSubsystem>()->lightGrid(
        {raster.lightGridViewIndex, raster.lightGridViewGeneration}, raster.lightGridFrameSlot);
    if (!grid || !grid->valid()) {
        spdlog::error("Volumetric fog requires the light grid published for its raster view");
        return makeError(Error::InvalidArgument);
    }
    if (grid->params.counts[1] > 0) {
        p.lightGrid = writer.buffer(grid->parameters);
        p.lights = writer.buffer(grid->lights);
        p.lightCells = writer.buffer(grid->cells);
        p.lightIndices = writer.buffer(grid->lightIndices);
        p.lightCandidates = writer.buffer(grid->candidates);
    }
    if (const auto* atmosphere = context.resources<EnvironmentResourceSet>()) {
        const auto& lighting = atmosphere->lighting;
        p.atmosphereParameters = writer.buffer(lighting.atmosphereParametersBuffer);
        p.transmittance = writer.sampledImageHandle(lighting.transmittanceView);
        if (lighting.cloudShadowView) { p.cloudShadow = writer.sampledImageHandle(lighting.cloudShadowView); }
    }

    const auto currentDepth = context.outputTexture("conservativeDepth");
    if (conservative) {
        p.currentDepth = writer.sampledImageHandle(currentDepth.view());
        p.currentDepthOut = writer.storageImageHandle(currentDepth.view());
    }
    std::vector<StageUse> shadowUses{{"shadow", Access::TextureStorageWrite}};
    if (rayTraced) {
        auto shadow = context.outputTexture("shadow");
        p.shadowOut = writer.storageImageHandle(shadow.view());
        p.shadow = writer.sampledImageHandle(shadow.view());
        if (p.lightColorIntensity[3] > 0.0f) {
            auto* tlas = context.inputAccelerationStructure("accelerationStructure");
            if (!tlas) {
                spdlog::error("Directional fog shadows require a published scene TLAS");
                return makeError(Error::InvalidArgument);
            }
            p.scene = writer.accelerationStructure(tlas).value;
            shadowUses.push_back({"accelerationStructure", Access::AccelerationStructureShaderRead});
        }
    }

    std::vector<StageUse> depthUses{{"depth", Access::TextureSampleRead}, {"conservativeDepth", Access::TextureStorageWrite}};
    std::vector<StageUse> materialUses{{"material", Access::TextureStorageWrite}};
    std::vector<StageUse> lightUses{{"material", Access::TextureSampleRead}, {"scattering", Access::TextureStorageWrite}};
    if (const auto faces = context.inputBuffer("worldShadowFaces"); faces.valid()) {
        p.worldShadowFaces = writer.buffer(faces.buffer());
        lightUses.push_back({"worldShadowFaces", Access::BufferStorageRead});
    }
    if (localVolumes) {
        const auto instances = packLocalFogVolumes(fog.localVolumes, p.current);
        p.localFog.startDistanceCm = std::max(10.0f, context.properties().value("localFogStartDistanceMeters", 20.0f) * 100.0f);
        const auto emissive = context.outputTexture("emissive");
        p.localFog.emissive = writer.sampledImageHandle(emissive.view());
        p.localFog.emissiveOut = writer.storageImageHandle(emissive.view());
        materialUses.push_back({"emissive", Access::TextureStorageWrite});
        lightUses.push_back({"emissive", Access::TextureSampleRead});
        if (!instances.empty()) {
            p.localFog.instances = writer.dataSpan(instances.data(), instances.size() * sizeof(PackedLocalFogVolume), sizeof(FogUInt4), 16);
            auto* tiles = context.outputTexture("localFogTiles").view();
            p.localFog.tiles = writer.sampledImageHandle(tiles);
            p.localFog.tilesOut = writer.storageImageHandle(tiles);
            materialUses.push_back({"localFogTiles", Access::TextureSampleRead});
        }
    }
    if (conservative) { lightUses.push_back({"conservativeDepth", Access::TextureSampleRead}); }
    if (rayTraced) { lightUses.push_back({"shadow", Access::TextureSampleRead}); }
    const auto* globalHistory = context.historyResources();
    const auto revision = globalHistory ? globalHistory->reprojectionInvalidationRevision() : 0;
    const auto* view = context.viewConstants();
    const auto cut = view ? view->frame[3] : 0;
    std::vector<RenderGraphTextureImport> imports;
    if (temporal) {
        history_.beginFrame(context.frameIndex());
        if (previous_.frame + 1 != context.frameIndex() || previous_.sceneIdentity != raster.sceneIdentity ||
            previous_.viewIndex != raster.lightGridViewIndex || previous_.viewGeneration != raster.lightGridViewGeneration ||
            previous_.fog != fog || previous_.gridZ != p.gridZ || previous_.reprojectionRevision != revision ||
            previous_.camera.viewport[1] != p.current.viewport[1] || previous_.camera.viewport[2] != p.current.viewport[2] ||
            previous_.cutSerial != cut || (view && !view->frame[1])) {
            history_.invalidateAll();
        }
        const auto prepareHistory = [&](const char* name, const TextureDesc& texture) {
            auto desc = texture;
            desc.usage = TextureUsageBits::Storage | TextureUsageBits::Sampled;
            return history_.ensureTexture(name, desc, {.format = desc.format});
        };
        if (auto r = prepareHistory("scattering", scattering.desc()); !r) { return r; }
        if (conservative) {
            if (auto r = prepareHistory("depth", currentDepth.desc()); !r) { return r; }
        }
        const auto import = [&](const char* name, HistoryTextureRef h) {
            imports.push_back({name, h.texture, h.view, h.state, ResourceState::ShaderRead});
            return h.view;
        };
        p.historyScatteringOut = writer.storageImageHandle(import("currentScattering", history_.texture("scattering", HistorySlot::Current)));
        lightUses.push_back({"currentScattering", Access::TextureStorageWrite});
        if (conservative) {
            p.historyDepthOut = writer.storageImageHandle(import("currentHistoryDepth", history_.texture("depth", HistorySlot::Current)));
            depthUses.push_back({"currentHistoryDepth", Access::TextureStorageWrite});
        }
        auto history = history_.texture("scattering", HistorySlot::Previous);
        auto oldDepth = history_.texture("depth", HistorySlot::Previous);
        if (history.valid && history.writtenFrameIndex + 1 == context.frameIndex() &&
            (!conservative || (oldDepth.valid && oldDepth.writtenFrameIndex == history.writtenFrameIndex))) {
            p.previous = previous_.camera;
            p.distances[2] = 0.9f;
            p.history = writer.sampledImageHandle(import("previousScattering", history));
            lightUses.push_back({"previousScattering", Access::TextureSampleRead});
            if (conservative) {
                p.previousDepth = writer.sampledImageHandle(import("previousDepth", oldDepth));
                lightUses.push_back({"previousDepth", Access::TextureSampleRead});
            }
        }
    }
    if (!p.distances[2]) { p.previous = p.current; }
    auto packet = writer.encode(p, kVolumetricFogABI);
    if (!packet) { return makeError(packet.error()); }
    const StageUse integrationUses[]{{"scattering", Access::TextureSampleRead}, {"integrated", Access::TextureStorageWrite}};
    std::vector<RenderGraphStage> stages;
    const StageUse cullUses[]{{"localFogTiles", Access::TextureStorageWrite}};
    const StageUse clearTileUses[]{{"localFogTiles", Access::TextureTransferWrite}};
    if (localVolumes) {
        if (p.localFog.instances.count) {
            stages.push_back({"Cull local fog tiles", cullUses, [&](CommandBuffer& cmd) {
                const auto& tiles = context.outputTexture("localFogTiles").desc();
                return localTileCulling_.dispatch(cmd, *packet, (tiles.width + 7u) / 8u, (tiles.height + 7u) / 8u, 1);
            }});
        } else {
            stages.push_back({"Clear empty local fog tiles", clearTileUses, [&](CommandBuffer& cmd) {
                return cmd.clearColorTexture(*context.outputTexture("localFogTiles").texture(), TextureLayout::TransferDestination, {});
            }});
        }
    }
    const auto stage = [&](const char* label, const ComputeKernel& kernel, std::span<const StageUse> uses, bool volume) {
        const uint32_t group = volume ? 4u : 8u;
        const auto x = (extent.width + group - 1) / group, y = (extent.height + group - 1) / group;
        const auto z = volume ? (extent.depth + 3u) / 4u : 1u;
        stages.push_back({label, uses, [&kernel, &packet, x, y, z](CommandBuffer& cmd) {
            return kernel.dispatch(cmd, *packet, x, y, z);
        }});
    };
    if (conservative) { stage("Fog conservative depth", depth_, depthUses, false); }
    stage("Fog material setup", material_, materialUses, true);
    if (rayTraced) { stage("Fog directional shadow", shadow_, shadowUses, true); }
    stage("Fog light scattering", lighting_, lightUses, true);
    stage("Fog final integration", integration_, integrationUses, false);
    if (auto r = context.executeStages(stages, {}, imports); !r) { return r; }
    if (temporal) {
        if (auto r = history_.publishTextureState(commands, "scattering", HistorySlot::Current, ResourceState::ShaderRead, true); !r) {
            return r;
        }
        if (conservative) {
            if (auto r = history_.publishTextureState(commands, "depth", HistorySlot::Current, ResourceState::ShaderRead, true); !r) {
                return r;
            }
        }
        previous_ = {p.current, fog, p.gridZ, context.frameIndex(), raster.sceneIdentity, revision, cut,
            raster.lightGridViewIndex, raster.lightGridViewGeneration};
    }
    return {};
}

class VolumetricFogCompositePass final : public ComputePass {
public:
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"integrated", "view"}};
    }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection r;
        r.addTextureInput("sceneColor", "Linear HDR before fog").sampledRead()
            .stageAccess(Access::TextureTransferRead).format = Format::RGBA32Sfloat;
        r.addTextureInput("depth", "Main raster reversed-Z depth").sampledRead().format = Format::D32Sfloat;
        r.addTextureInput("integrated", "Integrated in-scattering and transmittance").texture3D(0, 0, 0)
            .sampledRead().format = Format::RGBA16Sfloat;
        reflectMetadata(r.addBufferInput("view", "Mapping of the integrated volume"), sizeof(FogView));
        auto& output = r.addTextureOutput("color").storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        output.stageAccess(Access::TextureTransferWrite);
        output.format = Format::RGBA32Sfloat;
        output.colorEncoding = DisplayColorEncoding::SceneLinear;
        return r;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        return device_ ? compileFogKernel(*device_, "compositeFog", kernel_, log) : makeError(Error::InvalidArgument);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        FogView view;
        if (auto r = readMetadata(context.inputBuffer("view"), view); !r) { return r; }
        const auto volume = context.inputTexture("integrated");
        if (volume.desc().width != view.grid[0] || volume.desc().height != view.grid[1] ||
            volume.desc().depth != view.grid[2] || context.width() != uint32_t(view.camera.viewport[1]) ||
            context.height() != uint32_t(view.camera.viewport[2])) {
            spdlog::error("Volumetric fog composite requires scene depth and a matching integrated volume and view");
            return makeError(Error::InvalidArgument);
        }
        if (!view.enabled) {
            const StageUse uses[]{{"sceneColor", Access::TextureTransferRead}, {"color", Access::TextureTransferWrite}};
            const RenderGraphStage stages[]{{"Copy scene color", uses, [&](CommandBuffer& cmd) {
                return cmd.copyTexture({.source = context.inputTexture("sceneColor").texture(),
                    .destination = context.outputTexture("color").texture(), .width = context.width(), .height = context.height()});
            }}};
            return context.executeStages(stages);
        }
        auto reg = ResourceRegistry::forDevice(*device_);
        if (!reg) { return makeError(reg.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **reg, RenderFrameContext::from(commands));
        VolumetricFogParams p{};
        p.current = view.camera;
        p.gridZ = view.gridZ;
        p.distances = view.distances;
        p.grid = view.grid;
        p.depth = writer.sampledImageHandle(context.inputTexture("depth").view());
        p.sceneColor = writer.sampledImageHandle(context.inputTexture("sceneColor").view());
        p.integrated = writer.sampledImageHandle(volume.view());
        p.colorOut = writer.storageImageHandle(context.outputTexture("color").view());
        p.sampler = writer.samplerHandle({});
        auto packet = writer.encode(p, kVolumetricFogABI);
        if (!packet) { return makeError(packet.error()); }
        return kernel_.dispatch(commands, *packet, (context.width() + 7u) / 8u, (context.height() + 7u) / 8u, 1);
    }
private:
    Device* device_ = nullptr;
    ComputeKernel kernel_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createVolumetricFogPass()
{
    return std::make_unique<VolumetricFogPass>();
}

std::unique_ptr<RenderGraphPass> createVolumetricFogCompositePass()
{
    return std::make_unique<VolumetricFogCompositePass>();
}

} // namespace metallic::render::builtin_pass
