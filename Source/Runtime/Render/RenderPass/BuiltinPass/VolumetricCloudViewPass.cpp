#include "Runtime/Render/Environment/AtmosphereResources.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/Environment/VolumetricCloudViewParameters.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/Shadow/VirtualShadowMapWorld.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

namespace metallic::render::builtin_pass {
namespace {

using Access = RenderGraphResourceAccess;
using StageUse = RenderGraphStageUse;

// Each traced quantity has a quarter-resolution producer and a half-resolution history.
struct CloudChannel {
    const char* trace;
    const char* current;
    const char* previous;
    Format format;
    ColorValue clear;
    ShaderSampledImage VolumetricCloudViewParams::*input;
    ShaderSampledImage VolumetricCloudViewParams::*past;
    ShaderStorageImage VolumetricCloudViewParams::*output;
};
using CloudParams = VolumetricCloudViewParams;
constexpr float kDefaultDistanceToSampleMaxCountKm = 15.0f;
constexpr std::array<CloudChannel, 4> kCloudChannels{{
    {"traceFar", "currentFar", "previousFar", Format::RGBA32Sfloat, {0, 0, 0, 1},
        &CloudParams::currentFar, &CloudParams::previousFar, &CloudParams::writeFar},
    {"traceNear", "currentNear", "previousNear", Format::RGBA32Sfloat, {0, 0, 0, 1},
        &CloudParams::currentNear, &CloudParams::previousNear, &CloudParams::writeNear},
    {"traceDepth", "currentDepth", "previousDepth", Format::RGBA32Sfloat, {65504, 65504, 65504, 65504},
        &CloudParams::currentDepth, &CloudParams::previousDepth, &CloudParams::writeDepth},
    {"traceRepresentativeDepth", "currentRepresentative", "previousRepresentative", Format::RG32Sfloat, {},
        &CloudParams::currentRepresentativeDepth, &CloudParams::previousRepresentativeDepth, &CloudParams::writeRepresentativeDepth},
}};

struct CloudHistoryContent {
    uint64_t scene, discontinuity, atmosphereRevision, rasterSettings;
    uint32_t viewIndex, viewGeneration, cutSerial, width, height, skyVisible;
    float invDistanceToSampleCountMax;
    environment::WeatherState weather;
    std::array<environment::CelestialLight, environment::kCelestialLightCount> authoredLights;
    environment::EnvironmentTimeState authoredTime;
    environment::AstronomyState astronomy;
    bool operator==(const CloudHistoryContent&) const = default;
};

class VolumetricCloudViewPass final : public ComputePass {
public:
    QueueType queueType() const override { return QueueType::Graphics; }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        auto setting = runtimeBoolSetting("temporalReconstruction", "Temporal cloud reconstruction", true, true);
        setting.rebuildGraph = true;
        return {std::move(setting), runtimeFloatSetting("distanceToSampleMaxCountKm", "Distance to maximum sample count (km)",
            kDefaultDistanceToSampleMaxCountKm, 0.0f, 350.0f, true)};
    }

    RenderGraphSceneDependency sceneDependency() const override
    {
        return {.source = RenderGraphSceneSource::Input, .inputs = {"depth", "rasterInfo"}, .recompileOnTransform = false};
    }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{EnvironmentLightingSubsystem::kSubsystemId};
        return required;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection r;
        r.addTextureInput("sceneColor", "Scene-linear HDR with the clear camera sky and aerial perspective")
            .sampledRead().format = Format::RGBA32Sfloat;
        r.addTextureInput("depth", "Opaque reversed-Z raster depth").sampledRead().format = Format::D32Sfloat;
        auto& info = r.addBufferInput("rasterInfo", "Camera and jitter that produced the opaque depth")
            .buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo)).shaderRead();
        info.memoryLocation = MemoryLocation::HostUpload;
        constexpr uint32_t side = AtmosphereResourcesGPU::kAerialSize;
        r.addBufferInput("cameraAerial", "Gas aerial radiance and transmittance for this raster view")
            .buffer(side * side * side * 2 * 16, 16).storageRead().setOptional().sharedProducerInput = "rasterInfo";
        r.addBufferInput("worldShadowFaces", "Directional VSM pages for cloud and atmosphere lighting")
            .buffer(0, sizeof(VirtualShadowMapWorldFace)).bufferLayout("VirtualShadowMapWorldFace")
            .storageRead().setOptional().sharedProducerInput = "rasterInfo";

        const auto output = [&](const char* name, const char* description, Format format) -> RenderGraphField& {
            auto& field = r.addTextureOutput(name, description).storageWrite().transient(RenderGraphInitialization::FullOverwrite);
            field.format = format;
            return field;
        };
        output("scattering", "Premultiplied cloud radiance with aerial perspective, alpha is cloud coverage",
            Format::RGBA32Sfloat).colorEncoding = DisplayColorEncoding::SceneLinear;
        output("transmittance", "Cloud transmittance along the visible segment", Format::R32Sfloat);
        output("cloudDepth", "Representative cloud distance along the camera ray in metres, zero on a miss",
            Format::R32Sfloat);
        output("color", "Scene-linear HDR after cloud compositing", Format::RGBA32Sfloat).colorEncoding = DisplayColorEncoding::SceneLinear;
        if (temporalEnabled()) {
            for (const auto& channel : kCloudChannels) {
                auto& field = output(channel.trace, "Quarter-resolution cloud trace", channel.format);
                field.matchOutputExtent = false;
                field.extentSourceInput = "depth";
                field.extentDivisor = 4;
                field.stageAccess(Access::TextureSampleRead).stageAccess(Access::TextureTransferWrite);
            }
        }
        return r;
    }

    Result<> declareResourceRequests(const RenderGraphResourceRequestContext& context,
        RenderGraphResourceRequestBuilder& builder, std::string& log) override
    {
        const auto world = resolveWorldEnvironment(context.runtimeScene(), context.world());
        const auto& weather = world.weather;
        if (world.source != environment::EnvironmentSource::PhysicalAtmosphere || !weather.cloudEnabled ||
            weather.cloudCoverage == 0.0f || weather.cloudDensity == 0.0f || weather.cloudExtinctionPerKm == 0.0f) {
            return {};
        }
        auto observer = context.inputObserver("rasterInfo");
        if (!observer) {
            const auto* view = context.viewConstants();
            if (!view) {
                log = "VolumetricCloudViewPass requires a published raster observer or RenderView for cloud lighting";
                return makeError(Error::InvalidArgument);
            }
            observer = std::array<double, 3>{view->current.eye[0], view->current.eye[1], view->current.eye[2]};
        }
        auto result = context.subsystem<EnvironmentLightingSubsystem>()->declareEnvironmentRequest(context, builder,
            {.environment = world, .observerWorldMetres = *observer, .viewId = context.viewKey(), .needCelestial = false}, log);
        return result ? Result<>{} : makeError(result.error());
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const auto property = properties().find("temporalReconstruction");
        if (property != properties().end() && !property->is_boolean()) {
            log = "VolumetricCloudViewPass temporalReconstruction must be a boolean";
            return makeError(Error::InvalidArgument);
        }
        const auto distance = properties().find("distanceToSampleMaxCountKm");
        if (distance != properties().end() && (!distance->is_number() || !std::isfinite(distance->get<float>()) ||
            distance->get<float>() < 0.0f)) {
            log = "VolumetricCloudViewPass distanceToSampleMaxCountKm must be a finite non-negative distance";
            return makeError(Error::InvalidArgument);
        }
        device_ = context.device;
        const auto compileKernel = [&](const char* entry, ComputeKernel& kernel) {
            return ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = "Features/Environment/VolumetricCloudView", .entryPointName = entry,
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<VolumetricCloudViewParams>(kVolumetricCloudViewABI), .debugName = entry}, kernel, log);
        };
        auto result = compileKernel("volumetricCloudViewMain", kernel_);
        if (!result) { return result; }
        result = compileKernel("clearCloudViewMain", clearKernel_);
        if (!result) { return result; }
        if (temporalEnabled()) {
            if (!(result = compileKernel("cloudTraceTemporalMain", trace_))) { return result; }
            if (!(result = compileKernel("cloudReconstructMain", reconstruct_))) { return result; }
            if (!(result = compileKernel("cloudComposeMain", compose_))) { return result; }
        }
        auto noise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Scalar, log);
        if (!noise) { return makeError(noise.error()); }
        blueNoise_ = std::move(*noise);
        lastContent_.reset();
        return history_.initialize(*device_);
    }

    Result<> prepare(const RenderGraphCompileContext& context, std::string& log) override
    {
        return temporalEnabled() && kernel_.valid() && !trace_.valid() ? compile(context, log) : Result<>{};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto metadata = context.inputBuffer("rasterInfo");
        if (metadata.desc().size != sizeof(VisibilityBufferFrameInfo) || metadata.desc().memoryLocation != MemoryLocation::HostUpload) {
            spdlog::error("VolumetricCloudViewPass requires visibility raster metadata in a host upload buffer");
            return makeError(Error::InvalidArgument);
        }
        VisibilityBufferFrameInfo raster;
        const auto* mapped = metadata.buffer()->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(&raster, mapped, sizeof(raster));
        metadata.buffer()->unmap();
        if (raster.width != context.width() || raster.height != context.height() ||
            raster.frameIndex != context.frameIndex() || raster.sceneIdentity != context.runtimeScene()->resourceIdentity()) {
            spdlog::error("VolumetricCloudViewPass requires depth and camera metadata from the current raster scene and frame");
            return makeError(Error::InvalidArgument);
        }

        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        VolumetricCloudViewParams p{};
        static_assert(offsetof(VisibilityBufferFrameInfo, width) == sizeof(ViewCameraConstants));
        std::memcpy(&p.camera, raster.eye, sizeof(p.camera));
        p.jitter[0] = raster.jitter[0];
        p.jitter[1] = raster.jitter[1];
        p.sceneColor = writer.sampledImageHandle(context.inputTexture("sceneColor").view());
        p.color = writer.storageImageHandle(context.outputTexture("color").view());
        p.scattering = writer.storageImageHandle(context.outputTexture("scattering").view());
        p.transmittance = writer.storageImageHandle(context.outputTexture("transmittance").view());
        p.cloudDepth = writer.storageImageHandle(context.outputTexture("cloudDepth").view());
        const auto* set = context.resources<EnvironmentResourceSet>();
        if (set) {
            // UE 5.7.4 VolumetricCloudRendering.cpp clamps the denominator to one centimetre.
            const float distanceKm = properties().value("distanceToSampleMaxCountKm", kDefaultDistanceToSampleMaxCountKm);
            p.invDistanceToSampleCountMax = 1.0f / std::max(0.01f, 1000.0f * distanceKm);
            p.depth = writer.sampledImageHandle(context.inputTexture("depth").view());
            const auto& lighting = set->lighting;
            p.atmosphereParameters = writer.buffer(lighting.atmosphereParametersBuffer);
            p.transmittanceLut = writer.sampledImageHandle(lighting.transmittanceView);
            p.multiScatteringLut = writer.sampledImageHandle(lighting.multiScatteringView);
            p.skyVisible = lighting.settings.visible;
            p.blueNoise = writer.sampledImageHandle(blueNoise_.get());
            if (auto aerial = context.inputBuffer("cameraAerial"); aerial.valid()) {
                p.aerialPerspective = writer.buffer(aerial.buffer());
            }
            if (auto faces = context.inputBuffer("worldShadowFaces"); faces.valid()) {
                const auto world = resolveWorldEnvironment(context.runtimeScene(), context.world());
                for (uint32_t slot = 0; slot < world.celestial.size(); ++slot) {
                    if (world.celestial[slot].enabled && world.celestial[slot].castShadowsOnAtmosphere) {
                        p.shadowMask |= 1u << slot;
                    }
                }
                p.worldShadowFaces = writer.buffer(faces.buffer());
            }
        }
        if (set && temporalEnabled() && p.camera.upProjection[3] == 0.0f) {
            return executeTemporal(context, raster, p, writer);
        }
        lastContent_.reset();
        auto packet = writer.encode(p, kVolumetricCloudViewABI);
        if (!packet) { return makeError(packet.error()); }
        auto& kernel = set ? kernel_ : clearKernel_;
        if (auto result = kernel.dispatch(commands, *packet, (raster.width + 7u) / 8u, (raster.height + 7u) / 8u, 1u); !result) {
            return result;
        }
        if (temporalEnabled()) {
            // The diagnostic trace outputs are public even for an empty or orthographic view.
            std::array<StageUse, kCloudChannels.size()> uses;
            for (size_t i = 0; i < kCloudChannels.size(); ++i) { uses[i] = {kCloudChannels[i].trace, Access::TextureTransferWrite}; }
            RenderGraphStage clear{"Clear cloud trace", uses, [&](CommandBuffer& cmd) -> Result<> {
                for (const auto& channel : kCloudChannels) {
                    if (auto r = cmd.clearColorTexture(*context.outputTexture(channel.trace).texture(),
                        TextureLayout::TransferDestination, channel.clear); !r) { return r; }
                }
                return {};
            }};
            return context.executeStages({&clear, 1});
        }
        return {};
    }

private:
    bool temporalEnabled() const
    {
        const auto value = properties().find("temporalReconstruction");
        return value == properties().end() || *value == true;
    }

    Result<> executeTemporal(RenderGraphExecutionContext& context, const VisibilityBufferFrameInfo& raster,
        CloudParams& p, ParameterWriter& writer);

    Device* device_ = nullptr;
    ComputeKernel kernel_, clearKernel_;
    ComputeKernel trace_, reconstruct_, compose_;
    std::unique_ptr<TextureView> blueNoise_;
    HistoryResourceManager history_;
    std::optional<CloudHistoryContent> lastContent_;
    ViewCameraConstants previousCamera_{};
    uint32_t frameId_ = 0;
};

Result<> VolumetricCloudViewPass::executeTemporal(RenderGraphExecutionContext& context, const VisibilityBufferFrameInfo& raster,
    CloudParams& p, ParameterWriter& writer)
{
    const auto environment = resolveWorldEnvironment(context.runtimeScene(), context.world());
    const auto* view = context.viewConstants();
    const auto* shared = context.historyResources();
    // Weather revision advances with wind and celestial revision advances with time of day.
    // Compare the authored controls so ordinary playback can reuse the last cloud image.
    const CloudHistoryContent content{raster.sceneIdentity,
        (shared ? *shared : context.graphHistoryResources()).geometryReprojectionInvalidationRevision(),
        environment.atmosphereRevision, raster.rasterSettingsRevision, raster.lightGridViewIndex, raster.lightGridViewGeneration,
        view ? view->frame[3] : 0u, raster.width, raster.height, p.skyVisible, p.invDistanceToSampleCountMax,
        environment.weather, environment.authoredCelestial,
        environment.time, environment.astronomy};
    history_.beginFrame(context.frameIndex());
    const uint32_t halfWidth = (raster.width + 1u) / 2u, halfHeight = (raster.height + 1u) / 2u;
    bool valid = lastContent_ == content && (!view || view->frame[1] != 0u);
    for (const auto& channel : kCloudChannels) {
        auto desc = context.outputTexture(channel.trace).desc();
        desc.width = halfWidth;
        desc.height = halfHeight;
        desc.usage = TextureUsageBits::Storage | TextureUsageBits::Sampled;
        if (auto r = history_.ensureTexture(channel.current, desc, {.format = desc.format}); !r) { return r; }
        const auto previous = history_.texture(channel.current, HistorySlot::Previous);
        valid = valid && previous.valid && previous.writtenFrameIndex + 1 == context.frameIndex();
    }

    // FVolumetricRenderTargetViewStateData::Initialise increments FrameId before selecting the phase.
    constexpr std::array<uint32_t, 4> orderDithering{0, 2, 3, 1};
    const uint32_t frameId = ((valid ? frameId_ : 0u) + 1u) % 4u;
    const uint32_t offset = orderDithering[frameId];
    p.tracingPixelOffset = {offset % 2u, offset / 2u};
    p.noiseFrameIndex = uint32_t(context.frameIndex() % 8u);
    p.previousCamera = valid ? previousCamera_ : p.camera;
    auto reconstruction = p, composite = p;
    std::vector<RenderGraphTextureImport> imports;
    const auto import = [&](const char* name, const HistoryTextureRef& h) {
        imports.push_back({name, h.texture, h.view, h.state, ResourceState::ShaderRead});
        return h.view;
    };
    std::vector<StageUse> traceUses{{"depth", Access::TextureSampleRead}};
    if (context.inputBuffer("cameraAerial").valid()) { traceUses.push_back({"cameraAerial", Access::BufferStorageRead}); }
    if (context.inputBuffer("worldShadowFaces").valid()) { traceUses.push_back({"worldShadowFaces", Access::BufferStorageRead}); }
    std::vector<StageUse> reconstructUses;
    std::vector<StageUse> composeUses{{"sceneColor", Access::TextureSampleRead}, {"depth", Access::TextureSampleRead},
        {"color", Access::TextureStorageWrite}, {"scattering", Access::TextureStorageWrite},
        {"transmittance", Access::TextureStorageWrite}, {"cloudDepth", Access::TextureStorageWrite}};
    for (const auto& channel : kCloudChannels) {
        auto* traceView = context.outputTexture(channel.trace).view();
        auto* current = import(channel.current, history_.texture(channel.current, HistorySlot::Current));
        p.*channel.output = writer.storageImageHandle(traceView);
        reconstruction.*channel.input = writer.sampledImageHandle(traceView);
        reconstruction.*channel.output = writer.storageImageHandle(current);
        composite.*channel.input = writer.sampledImageHandle(current);
        traceUses.push_back({channel.trace, Access::TextureStorageWrite});
        reconstructUses.push_back({channel.trace, Access::TextureSampleRead});
        reconstructUses.push_back({channel.current, Access::TextureStorageWrite});
        composeUses.push_back({channel.current, Access::TextureSampleRead});
        if (valid) {
            auto* previous = import(channel.previous, history_.texture(channel.current, HistorySlot::Previous));
            reconstruction.*channel.past = writer.sampledImageHandle(previous);
            reconstructUses.push_back({channel.previous, Access::TextureSampleRead});
        }
    }
    auto tracePacket = writer.encode(p, kVolumetricCloudViewABI);
    if (!tracePacket) { return makeError(tracePacket.error()); }
    auto reconstructionPacket = writer.encode(reconstruction, kVolumetricCloudViewABI);
    if (!reconstructionPacket) { return makeError(reconstructionPacket.error()); }
    auto composePacket = writer.encode(composite, kVolumetricCloudViewABI);
    if (!composePacket) { return makeError(composePacket.error()); }
    const auto& traceExtent = context.outputTexture("traceFar").desc();
    const RenderGraphStage stages[]{
        {"Cloud quarter-resolution trace", traceUses, [&](CommandBuffer& cmd) {
            return trace_.dispatch(cmd, *tracePacket, (traceExtent.width + 7u) / 8u, (traceExtent.height + 7u) / 8u, 1u);
        }},
        {valid ? "Cloud temporal reconstruction" : "Cloud spatial reconstruction", reconstructUses, [&](CommandBuffer& cmd) {
            return reconstruct_.dispatch(cmd, *reconstructionPacket, (halfWidth + 7u) / 8u, (halfHeight + 7u) / 8u, 1u);
        }},
        {"Cloud full-resolution composite", composeUses, [&](CommandBuffer& cmd) {
            return compose_.dispatch(cmd, *composePacket, (raster.width + 7u) / 8u, (raster.height + 7u) / 8u, 1u);
        }},
    };
    if (auto r = context.executeStages(stages, {}, imports); !r) { return r; }
    for (const auto& channel : kCloudChannels) {
        if (auto r = history_.publishTextureState(context.commandBuffer(), channel.current,
            HistorySlot::Current, ResourceState::ShaderRead, true); !r) { return r; }
    }
    lastContent_ = content;
    previousCamera_ = p.camera;
    frameId_ = frameId;
    return {};
}

} // namespace

std::unique_ptr<RenderGraphPass> createVolumetricCloudViewPass()
{
    return std::make_unique<VolumetricCloudViewPass>();
}

} // namespace metallic::render::builtin_pass
