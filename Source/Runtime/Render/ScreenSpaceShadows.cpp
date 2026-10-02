#include "Runtime/Render/Core/ShadowTraceParameters.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/RenderGraph/NRDRuntime.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace metallic::render {
namespace {

#if METALLIC_HAS_NRD
void writeCameraMatrices(const ViewCameraConstants& camera, float* worldToView, float* viewToClip)
{
    const float3 eye(camera.eye[0], camera.eye[1], camera.eye[2]);
    const float3 forward = normalize(float3(camera.center[0], camera.center[1], camera.center[2]) - eye);
    const float3 right = normalize(cross(forward, float3(camera.upProjection[0], camera.upProjection[1], camera.upProjection[2])));
    const float3 up = cross(right, forward);
    const float view[] = {
        right.x, up.x, forward.x, 0, right.y, up.y, forward.y, 0,
        right.z, up.z, forward.z, 0, -dot(right, eye), -dot(up, eye), -dot(forward, eye), 1,
    };
    std::memcpy(worldToView, view, sizeof(view));
    std::fill_n(viewToClip, 16, 0.0f);
    const float nearPlane = camera.clipOrtho[0], farPlane = camera.clipOrtho[1];
    const bool ortho = camera.upProjection[3] > 0.5f;
    viewToClip[5] = ortho ? 2.0f / camera.clipOrtho[2] : 1.0f / std::tan(camera.viewport[3] * 0.5f);
    viewToClip[0] = viewToClip[5] / camera.viewport[0];
    viewToClip[10] = (ortho ? 1.0f : farPlane) / (farPlane - nearPlane);
    viewToClip[14] = -nearPlane * (ortho ? 1.0f : farPlane) / (farPlane - nearPlane);
    viewToClip[ortho ? 15 : 11] = 1.0f;
}
#endif

} // namespace

std::vector<GPUPunctualLight> buildScreenSpaceShadowLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& lighting)
{
    const auto sources = buildSceneLightRecords(scene != nullptr
        ? std::span<const scene::RenderLight>(scene->lights()) : std::span<const scene::RenderLight>(), lighting.lights);
    std::vector<GPUPunctualLight> lights(sources.size() + 1);
    lights[0].positionRange[0] = static_cast<float>(sources.size());
    for (size_t i = 0; i < sources.size(); ++i) { lights[i + 1] = sources[i].gpu; }
    return lights;
}

uint32_t selectScreenSpaceShadowLight(std::span<const GPUPunctualLight> lights, int32_t requestedIndex)
{
    const auto enabled = [&](size_t entry) {
        return entry < lights.size() && lights[entry].colorIntensity[3] > 0.0f;
    };
    if (requestedIndex >= 0 && enabled(size_t(requestedIndex) + 1)) {
        return static_cast<uint32_t>(requestedIndex);
    }
    uint32_t selected = UINT32_MAX;
    for (size_t i = 1; i < lights.size(); ++i) {
        if (!enabled(i)) { continue; }
        if (selected == UINT32_MAX) { selected = static_cast<uint32_t>(i - 1); }
        if (lights[i].directionType[3] < 0.5f) { return static_cast<uint32_t>(i - 1); }
    }
    return selected;
}

struct ScreenSpaceShadows::State {
    std::array<std::unique_ptr<Texture>, 5> textures; // penumbra, normal, viewZ, MV, shadow
    std::array<std::unique_ptr<TextureView>, 5> views;
    std::shared_ptr<Buffer> parameters;
    std::vector<std::shared_ptr<Buffer>> parameterPool;
    uint32_t width = 0, height = 0;
    uint64_t sceneRevision = 0;
    uint64_t transformRevision = 0;
    GPUPunctualLight light{};
    ScreenSpaceShadowSettings settings;
    uint32_t lightIndex = UINT32_MAX;
    bool initialized = false;
    bool cancelled = false;
    bool traceEnabled = false;
#if METALLIC_HAS_NRD
    NRDRuntime sigma;
#endif
};

void ScreenSpaceShadows::clear()
{
    state_.reset();
    for (auto& trace : traces_) { trace.clear(); }
}

Result<ScreenSpaceShadowResult> ScreenSpaceShadows::record(
    Device& device,
    CommandBuffer& commands,
    Streamer& streamer,
    TextureView& depth,
    const ViewConstants& view,
    std::span<const GPUPunctualLight> lights,
    uint64_t sceneRevision,
    uint64_t transformRevision,
    const ScreenSpaceShadowSettings& settings,
    std::string& log,
    ScenePathTraceResources* geometry,
    const MeshletStreamDeferredGPUResourcesView* streamGeometry,
    CPUProfileRecorder* profiler,
    RayTracingAccelerationStructure* accelerationStructure)
{
    ScreenSpaceShadowResult output{};
    CPUProfileScope profile(profiler, "Validate shadow resources");
    output = {};
    const uint32_t width = static_cast<uint32_t>(view.current.viewport[1]);
    const uint32_t height = static_cast<uint32_t>(view.current.viewport[2]);
    if (width == 0 || height == 0 || width > UINT16_MAX || height > UINT16_MAX ||
        !std::isfinite(settings.maxDistance) || settings.maxDistance <= 0 || settings.maxDistance > 1000000 ||
        !std::isfinite(settings.normalBias) || settings.normalBias < 0 ||
        !std::isfinite(settings.angularRadiusDegrees) || settings.angularRadiusDegrees < 0 || settings.angularRadiusDegrees > 10 ||
        !std::isfinite(settings.lightRadius) || settings.lightRadius < 0) {
        log = "Invalid ray-traced shadow dimensions or trace settings";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().rayQuery || !device.capabilities().rayTracingAccelerationStructure) {
        log = "Ray-traced shadows require ray queries and a scene acceleration structure";
        return makeError(Error::Unsupported);
    }
    const bool streamed = streamGeometry != nullptr;
    const bool streamTlas = streamed && (accelerationStructure || streamGeometry->accelerationStructure != nullptr);
    if (!streamed && (geometry == nullptr || !geometry->valid())) {
        log = "Ray-traced shadows require prepared scene geometry";
        return makeError(Error::InvalidArgument);
    }
    const auto* neural = streamed ? nullptr : &geometry->neuralTextures();
    const bool ntc = neural != nullptr && neural->active();
    const bool coop = ntc && neural->cooperativeVectorActive();
    profile.next("Prepare trace pipeline");
    auto& trace = traces_[streamed ? (streamTlas ? 3 : 4) : (ntc ? (coop ? 2 : 1) : 0)];
    const uint32_t textureCount = streamed && !streamTlas ? 0 : geometry->materialTextureCount();
    if (!trace.valid()) {
        std::vector<const char*> capabilities{"spvRayQueryKHR"};
        if (coop) { capabilities.push_back("spvCooperativeVectorNV"); }
        std::vector<const char*> searchPaths;
#if METALLIC_HAS_NTC
        if (ntc) { searchPaths.push_back(METALLIC_NTC_SHADER_INCLUDE_DIR); }
#endif
        const SlangMacroDefine defines[] = {
            {.name = "METALLIC_STREAM_SHADOWS", .value = streamed ? "1" : "0"},
            {.name = "METALLIC_STREAM_TLAS", .value = streamTlas ? "1" : "0"},
            {.name = "METALLIC_HAS_NTC", .value = ntc ? "1" : "0"},
            {.name = "METALLIC_NTC_COOPERATIVE_VECTOR", .value = coop ? "1" : "0"},
        };
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({
            .moduleName = "Features/Lighting/ScreenSpaceShadows",
            .entryPointName = "rayTracedShadowsMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .additionalSearchPaths = searchPaths,
            .capabilities = capabilities,
            .macroDefines = {defines, 4},
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result.transform([&] { return std::move(output); }); }
        result = trace.initialize(device, {
            .spirv = shader.spirv,
            .parameters = parameterAbi<ShadowTraceParameters>(kShadowTraceABI, ParameterTransport::InlinePush),
            .debugName = "Ray-traced shadows",
        }, log);
        if (!result) { return makeError(result.error()); }
    }
    profile.next("Prepare shadow images");
    if (!state_ || state_->cancelled || state_->width != width || state_->height != height) {
        auto next = std::make_shared<State>();
        const Format formats[] = {Format::R16Sfloat, nrdNormalRoughnessFormat(), Format::R32Sfloat,
            Format::RGBA16Sfloat, Format::R8Unorm};
        for (size_t i = 0; i < next->textures.size(); ++i) {
            auto result = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::Storage |
                TextureUsageBits::TransferDestination | TextureUsageBits::TransferSource,
                .format = formats[i], .width = width, .height = height}).transform([&](auto rhiValue) { next->textures[i] = std::move(rhiValue); });
            if (!result) { return makeError(result.error()); }
            result = device.createTextureView(*next->textures[i], {.format = formats[i]}).transform([&](auto rhiValue) { next->views[i] = std::move(rhiValue); });
            if (!result) { return makeError(result.error()); }
        }
        next->width = width;
        next->height = height;
        state_ = std::move(next);
    }
    auto state = state_;
    if (auto* frame = commands.frameContext()) { frame->retain(state); }
    profile.next("Prepare shadow parameters");
    state->parameters.reset();
    for (auto& candidate : state->parameterPool) {
        if (candidate.use_count() == 1) { state->parameters = candidate; break; }
    }
    if (!state->parameters) {
        std::unique_ptr<Buffer> buffer;
        auto allocated = device.createBuffer({.size = sizeof(ScreenSpaceShadowParameters),
            .structureStride = sizeof(ScreenSpaceShadowParameters), .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
        if (!allocated) { return makeError(allocated.error()); }
        state->parameters = std::move(buffer);
        state->parameterPool.push_back(state->parameters);
    }
    if (auto* frame = commands.frameContext()) { frame->retain(state->parameters); }
    auto result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>([] {},
        [state] { state->cancelled = true; }));
    if (!result) { return makeError(result.error()); }

    // Deferred compares this index with the stable LightGrid source slot.
    const uint32_t selected = selectScreenSpaceShadowLight(lights, settings.lightIndex);
    ScreenSpaceShadowParameters parameters{};
    parameters.view = view;
    if (selected != UINT32_MAX) { parameters.light = lights[selected + 1]; }
    parameters.trace[0] = settings.maxDistance;
    parameters.trace[2] = settings.normalBias;
    parameters.trace[3] = std::tan(settings.angularRadiusDegrees * 0.01745329252f);
    parameters.control[0] = settings.enabled && selected != UINT32_MAX && (!streamed || streamTlas);
    parameters.control[2] = selected;
    parameters.control[3] = settings.debug;
    parameters.shape[0] = settings.lightRadius;
    const bool denoise = METALLIC_HAS_NRD && settings.denoise && parameters.control[0];
    parameters.shape[1] = denoise ? 1.0f : 0.0f;
    void* mapped = state->parameters->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, &parameters, sizeof(parameters));
    state->parameters->flush({0, sizeof(parameters)});
    state->parameters->unmap();
    commands.hostWriteBarrier();

    profile.next("Plan shadow accesses");
    std::array<detail::GraphAccessResource, 5> accessResources;
    std::array<detail::GraphAccessBinding, 5> accessBindings;
    std::array<detail::GraphAccessPass, 4> accessStages;
    for (size_t i = 0; i < accessResources.size(); ++i) {
        accessResources[i] = {.type = RenderGraphResourceType::Texture2D,
            .state = state->initialized ? (i == 4 ? ResourceState::ShaderRead : ResourceState::General) : ResourceState::Undefined,
            .scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
        accessBindings[i] = {.texture = state->textures[i].get()};
        if (!state->initialized) {
            accessStages[0].uses.push_back({i, ResourceState::TransferDestination,
                {PipelineStageBits::Transfer, AccessBits::TransferWrite}, true});
        }
        accessStages[1].uses.push_back({i, ResourceState::General,
            {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}, true});
        if (denoise) {
            accessStages[2].uses.push_back({i, ResourceState::General,
                {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}, true});
        }
    }
    accessStages[3].uses.push_back({4, ResourceState::ShaderRead,
        {PipelineStageBits::AllCommands, AccessBits::ShaderRead}, false});
    auto accessPlan = detail::buildGraphAccessPlan(accessResources, accessStages);
    if (!accessPlan) { return makeError(accessPlan.error()); }
    const auto enterStage = [&](size_t index) {
        return detail::recordGraphAccessBarriers(commands, accessPlan->passes[index], accessBindings);
    };
    if (!state->initialized) {
        result = enterStage(0);
        if (!result) { return makeError(result.error()); }
        for (size_t i = 0; i < accessResources.size(); ++i) {
            commands.clearColorTexture(*state->textures[i], ResourceState::TransferDestination, {1, 1, 1, 1});
        }
    }
    profile.next("Encode shadow parameters");
    auto registry = device.resourceRegistry();
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, commands.frameContext());
    ShadowTraceParameters params{};
    params.settings = writer.dataBuffer(state->parameters.get(), sizeof(ScreenSpaceShadowParameters), 4);
    params.depth = writer.sampledImage(&depth);
    params.penumbra = writer.storageImage(state->views[0].get());
    params.normal = writer.storageImage(state->views[1].get());
    params.viewZ = writer.storageImage(state->views[2].get());
    params.motion = writer.storageImage(state->views[3].get());
    params.shadow = writer.storageImage(state->views[4].get());
    if (!streamed || streamTlas) {
        CPUProfileScope resources(profiler, "Prepare material textures");
        writer.retain(std::make_shared<ScenePathTraceResources>(*geometry));
        PathTraceParameters scene{};
        scene.scene = writer.accelerationStructure(accelerationStructure ? accelerationStructure :
            (streamed ? streamGeometry->accelerationStructure : geometry->accelerationStructure().accelerationStructure()));
        scene.materials = writer.buffer(geometry->materialBuffer());
        scene.materialTextures = writer.sampledImages({geometry->materialTextureViews().data(), textureCount});
        params.materialTextureCount = textureCount;
        if (streamed) {
            params.streamScene = streamGeometry->encodeRayQuerySnapshot(writer);
        } else {
            scene.vertices = writer.dataBuffer(geometry->shadingVertexBuffer(), 16, 8);
            scene.indices = writer.dataBuffer(geometry->indexBuffer(), 4, 4);
            scene.primitives = writer.dataBuffer(geometry->primitiveBuffer(), 32, 4);
            scene.instances = writer.dataBuffer(geometry->instanceBuffer(), 16, 4);
            params.ntcTextureSetCount = neural->textureSetCount();
        }
        if (ntc) {
            scene.ntcLatents = writer.sampledImages(neural->latentTextureViews());
            scene.ntcConstants = writer.buffer(neural->constantsBuffer());
            scene.ntcWeights = writer.buffer(neural->weightsBuffer());
            scene.ntcInfo = writer.buffer(neural->setInfoBuffer());
            scene.ntcSampler = writer.sampler(neural->latentSampler());
        }
        params.scene = writer.data(&scene, sizeof(scene));
    }
    auto encoded = writer.encode(params, kShadowTraceABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    profile.next("Record trace dispatch");
    result = enterStage(1);
    if (!result) { return makeError(result.error()); }
    commands.beginDebugLabel({.name = "Ray-traced shadows"});
    result = trace.dispatch(commands, *encoded, (width + 7) / 8, (height + 7) / 8);
    commands.endDebugLabel();
    if (!result) { return makeError(result.error()); }
    profile.next("Record denoising");
#if METALLIC_HAS_NRD
    if (denoise) {
        result = enterStage(2);
        if (!result) { return makeError(result.error()); }
        if (!state->sigma.valid()) {
            NRDUserTexturePool pool{};
            const denoising::ResourceType resources[] = {denoising::ResourceType::IN_PENUMBRA,
                denoising::ResourceType::IN_NORMAL_ROUGHNESS, denoising::ResourceType::IN_VIEWZ,
                denoising::ResourceType::IN_MV, denoising::ResourceType::OUT_SHADOW_TRANSLUCENCY};
            for (size_t i = 0; i < 5; ++i) { pool[static_cast<size_t>(resources[i])] = {state->textures[i].get(), state->views[i].get()}; }
            result = state->sigma.initialize(device, static_cast<uint16_t>(width), static_cast<uint16_t>(height), pool, log, true);
            if (!result) { return makeError(result.error()); }
        }
        denoising::CommonSettings common;
        writeCameraMatrices(view.current, common.worldToViewMatrix, common.viewToClipMatrix);
        writeCameraMatrices(view.previous, common.worldToViewMatrixPrev, common.viewToClipMatrixPrev);
        for (auto* size : {common.resourceSize, common.resourceSizePrev, common.rectSize, common.rectSizePrev}) {
            size[0] = static_cast<uint16_t>(width);
            size[1] = static_cast<uint16_t>(height);
        }
        for (size_t i = 0; i < 2; ++i) {
            common.cameraJitter[i] = view.frame[2] ? view.jitter[i] : 0.0f;
            common.cameraJitterPrev[i] = view.frame[2] ? view.jitter[i + 2] : 0.0f;
        }
        common.frameIndex = view.frame[0];
        common.denoisingRange = std::min(view.current.clipOrtho[1], 500000.0f);
        common.timeDeltaBetweenFrames = 1000.0f / 60.0f;
        common.accumulationMode = !state->initialized || !view.frame[1] || sceneRevision != state->sceneRevision ||
            transformRevision != state->transformRevision || parameters.control[0] != state->traceEnabled ||
            selected != state->lightIndex || settings != state->settings ||
            std::memcmp(&parameters.light, &state->light, sizeof(GPUPunctualLight)) != 0
            ? denoising::AccumulationMode::CLEAR_AND_RESTART : denoising::AccumulationMode::CONTINUE;
        result = state->sigma.setCommonSettings(common);
        if (!result) { return makeError(result.error()); }
        denoising::SigmaSettings sigma;
        sigma.maxStabilizedFrameNum = std::min(settings.historyLength, denoising::SIGMA_MAX_HISTORY_FRAME_NUM);
        if (parameters.light.directionType[3] < 0.5f) {
            for (size_t i = 0; i < 3; ++i) { sigma.lightDirection[i] = -parameters.light.directionType[i]; }
        }
        result = state->sigma.setSigmaSettings(sigma);
        if (result) { result = state->sigma.denoise(NRDDenoiserMode::Sigma, commands); }
        if (!result) { return makeError(result.error()); }
    }
#endif
    profile.next("Finalize shadow state");
    result = enterStage(3);
    if (!result) { return makeError(result.error()); }
    state->initialized = true;
    state->sceneRevision = sceneRevision;
    state->transformRevision = transformRevision;
    state->lightIndex = selected;
    state->light = parameters.light;
    state->settings = settings;
    state->traceEnabled = parameters.control[0] != 0;
    output = {.texture = state->textures[4].get(), .shadow = state->views[4].get(), .parameters = state->parameters.get()};
    return output;
}

} // namespace metallic::render
