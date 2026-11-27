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

class VolumetricCloudViewPass final : public ComputePass {
public:
    QueueType queueType() const override { return QueueType::Graphics; }

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
        device_ = context.device;
        auto result = ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/Environment/VolumetricCloudView", .entryPointName = "volumetricCloudViewMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<VolumetricCloudViewParams>(kVolumetricCloudViewABI), .debugName = "VolumetricCloudView"},
            kernel_, log);
        if (!result) { return result; }
        auto noise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Scalar, log);
        if (!noise) { return makeError(noise.error()); }
        blueNoise_ = std::move(*noise);
        return {};
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
        p.depth = writer.sampledImageHandle(context.inputTexture("depth").view());
        p.color = writer.storageImageHandle(context.outputTexture("color").view());
        p.scattering = writer.storageImageHandle(context.outputTexture("scattering").view());
        p.transmittance = writer.storageImageHandle(context.outputTexture("transmittance").view());
        p.cloudDepth = writer.storageImageHandle(context.outputTexture("cloudDepth").view());
        if (const auto* set = context.resources<EnvironmentResourceSet>()) {
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
        auto packet = writer.encode(p, kVolumetricCloudViewABI);
        if (!packet) { return makeError(packet.error()); }
        return kernel_.dispatch(commands, *packet, (raster.width + 7u) / 8u, (raster.height + 7u) / 8u, 1u);
    }

private:
    Device* device_ = nullptr;
    ComputeKernel kernel_;
    std::unique_ptr<TextureView> blueNoise_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createVolumetricCloudViewPass()
{
    return std::make_unique<VolumetricCloudViewPass>();
}

} // namespace metallic::render::builtin_pass
