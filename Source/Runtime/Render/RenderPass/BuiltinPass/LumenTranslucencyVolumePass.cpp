#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenTranslucencyVolumeParameters.h"
#include "Runtime/Render/GPUDrivenRaster.h"

namespace metallic::render::builtin_pass {
namespace {

using Access = RenderGraphResourceAccess;
using Use = RenderGraphStageUse;

uint32_t gridSliceCount(float nearMetres, float farMetres)
{
    const float range = (farMetres * 100.0f - nearMetres * 100.0f) * 0.01f;
    return range > 1.0f ? static_cast<uint32_t>(std::log2(range) * 4.0f) + 1u : 1u;
}

class LumenTranslucencyVolumePass final : public ComputePass {
public:
    QueueType queueType() const override { return QueueType::Graphics; }
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {.source = RenderGraphSceneSource::Input, .inputs = {"depth", "rasterInfo"},
            .recompileOnTransform = false};
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{EnvironmentLightingSubsystem::kSubsystemId};
        return required;
    }
    Result<> declareResourceRequests(const RenderGraphResourceRequestContext& context,
        RenderGraphResourceRequestBuilder& builder, std::string& log) override
    {
        return declareLumenSky(context, builder, log);
    }
    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {runtimeIntSetting("gridPixelSize", "Pixels per GI cell", 32, 1, 128),
            runtimeFloatSetting("endDistanceMeters", "GI volume distance (m)", 80.0f, 0.01f, 1000.0f),
            runtimeFloatSetting("maxRayIntensity", "Maximum ray intensity", 20.0f, 0.0f, 65504.0f),
            runtimeBoolSetting("spatialFilter", "Spatial filter", true),
            runtimeBoolSetting("jitter", "Jitter probe positions and directions", true)};
    }
    Result<> prepare(const RenderGraphCompileContext& context, std::string& log) override;
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    Device* device_ = nullptr;
    ComputeKernel depth_, trace_, filterX_, filterY_, filterZ_, integrate_;
    std::unique_ptr<TextureView> blueNoise_;
};

Result<> LumenTranslucencyVolumePass::prepare(const RenderGraphCompileContext& context, std::string& log)
{
    const auto tile = properties().value("gridPixelSize", 32);
    const auto endDistance = properties().value("endDistanceMeters", 80.0f);
    const auto maxRayIntensity = properties().value("maxRayIntensity", 20.0f);
    if (!context.device || !context.graphicsQueue || tile < 1 || tile > 128 ||
        !(endDistance >= 0.01f && endDistance <= 1000.0f) || !(maxRayIntensity >= 0.0f && maxRayIntensity <= 65504.0f)) {
        log = "LumenTranslucencyVolumePass requires a device, graphics queue, gridPixelSize in [1,128], "
            "endDistanceMeters in [0.01,1000] and maxRayIntensity in [0,65504]";
        return makeError(Error::InvalidArgument);
    }
    return {};
}

RenderPassReflection LumenTranslucencyVolumePass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection r;
    auto& depth = r.addTextureInput("depth", "Reversed-Z depth from the raster view").sampledRead();
    depth.format = Format::D32Sfloat;
    depth.matchOutputExtent = false;
    auto& raster = r.addBufferInput("rasterInfo", "Camera that produced depth")
        .buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo)).shaderRead();
    raster.memoryLocation = MemoryLocation::HostUpload;
    reflectLumenGlobalTraceInputs(r);
    auto& grid = r.addBufferOutput("giGrid", "GI log-depth mapping (B, O, S, active slices)").buffer(16, 16).shaderRead();
    grid.memoryLocation = MemoryLocation::HostUpload;
    const auto tile = properties().value("gridPixelSize", 32u);
    const auto slices = gridSliceCount(0.0f, properties().value("endDistanceMeters", 80.0f));
    const auto output = [&](const char* name, Format format, uint32_t depthSlices) {
        auto& field = r.addTextureOutput(name);
        if (depthSlices) { field.texture3D(0, 0, depthSlices); }
        field.format = format;
        field.matchOutputExtent = false;
        field.extentSourceInput = "depth";
        field.extentDivisor = tile;
        field.storageWrite().stageAccess(Access::TextureSampleRead).transient(RenderGraphInitialization::FullOverwrite);
    };
    output("maxVisibleDepth", Format::R32Sfloat, 0);
    output("traceRadiance", Format::RGBA16Sfloat, slices * 9u);
    if (properties().value("spatialFilter", true)) { output("filteredRadiance", Format::RGBA16Sfloat, slices * 9u); }
    output("giAmbient", Format::RGBA16Sfloat, slices);
    output("giDirectional", Format::RGBA16Sfloat, slices);
    return r;
}

Result<> LumenTranslucencyVolumePass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    const auto kernel = [&](const char* name, ComputeKernel& result) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenTranslucencyVolume", .entryPointName = name,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<LumenTranslucencyVolumeParameters>(kLumenTranslucencyVolumeABI), .debugName = name},
            result, log);
    };
    if (auto r = kernel("reduceVisibleDepth", depth_); !r) { return r; }
    if (auto r = kernel("traceVolume", trace_); !r) { return r; }
    if (auto r = kernel("filterX", filterX_); !r) { return r; }
    if (auto r = kernel("filterY", filterY_); !r) { return r; }
    if (auto r = kernel("filterZ", filterZ_); !r) { return r; }
    if (auto r = kernel("integrateVolume", integrate_); !r) { return r; }
    auto noise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Vec2, log);
    if (!noise) { return makeError(noise.error()); }
    blueNoise_ = std::move(*noise);
    return {};
}

Result<> LumenTranslucencyVolumePass::execute(RenderGraphExecutionContext& context)
{
    auto rasterInfo = context.inputBuffer("rasterInfo");
    const auto depth = context.inputTexture("depth");
    const auto ambient = context.outputTexture("giAmbient");
    VisibilityBufferFrameInfo raster{};
    auto* mapped = rasterInfo.buffer()->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(&raster, mapped, sizeof(raster));
    rasterInfo.buffer()->unmap();
    if (!context.runtimeScene() || raster.sceneIdentity != context.runtimeScene()->resourceIdentity() ||
        raster.frameIndex != context.frameIndex() || raster.width != depth.desc().width || raster.height != depth.desc().height) {
        spdlog::error("LumenTranslucencyVolumePass requires depth and raster metadata for its current scene and frame");
        return makeError(Error::InvalidArgument);
    }
    const auto clipmaps = lumenGlobalTraceClipmapCount(context);
    if (!clipmaps) { return makeError(Error::InvalidArgument); }
    const auto& extent = ambient.desc();
    const uint32_t activeSlices = gridSliceCount(raster.clipOrtho[0], properties().value("endDistanceMeters", 80.0f));
    const std::array gridZ{0.01f, 1.0f, 4.0f, static_cast<float>(activeSlices)};
    auto* grid = context.outputBuffer("giGrid").buffer();
    mapped = grid->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, gridZ.data(), sizeof(gridZ));
    grid->flush({0, sizeof(gridZ)});
    grid->unmap();

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(context.commandBuffer()));
    const bool jitter = properties().value("jitter", true);
    const bool spatialFilter = properties().value("spatialFilter", true);
    const uint32_t frame = uint32_t(context.frameIndex());
    LumenTranslucencyVolumeParameters p{};
    std::copy_n(raster.eye, 4, p.camera.eye);
    std::copy_n(raster.center, 4, p.camera.center);
    std::copy_n(raster.upProjection, 4, p.camera.upProjection);
    std::copy_n(raster.viewport, 4, p.camera.viewport);
    std::copy_n(raster.clipOrtho, 4, p.camera.clipOrtho);
    p.grid = {extent.width, extent.height, activeSlices, properties().value("gridPixelSize", 32u)};
    p.tracing = {*clipmaps, jitter ? frame % 8u : 0u, extent.depth, spatialFilter};
    const uint32_t sequence = frame & 1023u;
    p.frameJitter = jitter ? VolumeFloat3{radicalInverse(sequence, 2), radicalInverse(sequence, 3),
        radicalInverse(sequence, 5)} : VolumeFloat3{0.5f, 0.5f, 0.5f};
    p.maxRayIntensity = properties().value("maxRayIntensity", 20.0f);
    p.sky = bindLumenSky(context, writer);
    p.scene = bindLumenGlobalTrace(context, writer);
    p.blueNoise = writer.sampledImageHandle(blueNoise_.get());
    p.depth = writer.sampledImageHandle(depth.view());
    p.maxVisibleDepth = writer.sampledImageHandle(context.outputTexture("maxVisibleDepth").view());
    p.maxVisibleDepthOut = writer.storageImageHandle(context.outputTexture("maxVisibleDepth").view());
    p.trace = writer.sampledImageHandle(context.outputTexture("traceRadiance").view());
    p.traceOut = writer.storageImageHandle(context.outputTexture("traceRadiance").view());
    if (spatialFilter) {
        p.filtered = writer.sampledImageHandle(context.outputTexture("filteredRadiance").view());
        p.filteredOut = writer.storageImageHandle(context.outputTexture("filteredRadiance").view());
    }
    p.ambientOut = writer.storageImageHandle(ambient.view());
    p.directionalOut = writer.storageImageHandle(context.outputTexture("giDirectional").view());
    auto packet = writer.encode(p, kLumenTranslucencyVolumeABI);
    if (!packet) { return makeError(packet.error()); }

    const Use depthUses[]{{"depth", Access::TextureSampleRead}, {"maxVisibleDepth", Access::TextureStorageWrite}};
    const Use traceUses[]{{"depth", Access::TextureSampleRead}, {"maxVisibleDepth", Access::TextureSampleRead},
        {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead}, {"coarseMip", Access::TextureSampleRead},
        {"domain", Access::BufferStorageRead}, {"objectGrid", Access::BufferStorageRead}, {"objectGridScene", Access::BufferStorageRead},
        {"cards", Access::BufferStorageRead}, {"cardScene", Access::BufferStorageRead}, {"cardDepth", Access::TextureSampleRead},
        {"cardLighting", Access::TextureSampleRead}, {"traceRadiance", Access::TextureStorageWrite}};
    const Use filterForward[]{{"traceRadiance", Access::TextureSampleRead}, {"filteredRadiance", Access::TextureStorageWrite}};
    const Use filterBackward[]{{"filteredRadiance", Access::TextureSampleRead}, {"traceRadiance", Access::TextureStorageWrite}};
    const Use integrateUses[]{{spatialFilter ? "filteredRadiance" : "traceRadiance", Access::TextureSampleRead},
        {"maxVisibleDepth", Access::TextureSampleRead}, {"giAmbient", Access::TextureStorageWrite},
        {"giDirectional", Access::TextureStorageWrite}};
    std::vector<RenderGraphStage> stages;
    const auto add = [&](const char* label, const ComputeKernel& kernel, std::span<const Use> uses,
                         uint32_t groupXY, uint32_t groupsZ) {
        uint32_t groupsX = (extent.width + groupXY - 1u) / groupXY;
        uint32_t groupsY = (extent.height + groupXY - 1u) / groupXY;
        stages.push_back({label, uses, [&kernel, &packet, groupsX, groupsY, groupsZ](CommandBuffer& cmd) {
            return kernel.dispatch(cmd, *packet, groupsX, groupsY, groupsZ);
        }});
    };
    const uint32_t traceGroupsZ = (extent.depth * 9u + 3u) / 4u;
    add("Lumen volume furthest depth", depth_, depthUses, 8, 1);
    add("Trace Lumen volume", trace_, traceUses, 4, traceGroupsZ);
    if (spatialFilter) {
        add("Filter Lumen volume X", filterX_, filterForward, 4, traceGroupsZ);
        add("Filter Lumen volume Y", filterY_, filterBackward, 4, traceGroupsZ);
        add("Filter Lumen volume Z", filterZ_, filterForward, 4, traceGroupsZ);
    }
    add("Integrate Lumen volume SH", integrate_, integrateUses, 4, (extent.depth + 3u) / 4u);
    return context.executeStages(stages);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenTranslucencyVolumePass()
{
    return std::make_unique<LumenTranslucencyVolumePass>();
}
} // namespace metallic::render::builtin_pass
