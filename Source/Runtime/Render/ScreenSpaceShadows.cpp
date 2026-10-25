#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceMember.h"
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

std::vector<ScreenSpaceShadowLightRecord> buildScreenSpaceShadowLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& lighting,
    const environment::EnvironmentSnapshot& environment)
{
    const auto sources = buildSceneLightRecords(scene != nullptr
        ? std::span<const scene::RenderLight>(scene->lights()) : std::span<const scene::RenderLight>(), lighting.lights);
    const auto celestial = buildCelestialLightRecords(environment);
    std::vector<ScreenSpaceShadowLightRecord> lights(celestial.size() + sources.size());
    for (size_t i = 0; i < celestial.size(); ++i) {
        lights[i] = {.celestial = celestial[i], .sourceIndex = static_cast<uint32_t>(i),
            .isCelestial = true, .enabled = (celestial[i].flags & kGPUCelestialLightEnabled) != 0};
        lights[i].celestial.shadowImportance *= std::max(-celestial[i].direction[1], 0.0f);
    }
    for (size_t i = 0; i < sources.size(); ++i) {
        lights[celestial.size() + i] = {.local = sources[i].gpu,
            .sourceIndex = static_cast<uint32_t>(i), .enabled = sources[i].enabled};
    }
    return lights;
}

uint32_t selectScreenSpaceShadowLight(std::span<const ScreenSpaceShadowLightRecord> lights, int32_t requestedIndex)
{
    const auto enabled = [&](size_t entry) {
        return entry < lights.size() && lights[entry].enabled;
    };
    if (requestedIndex >= 0 && enabled(size_t(requestedIndex))) {
        return static_cast<uint32_t>(requestedIndex);
    }
    uint32_t selected = UINT32_MAX;
    uint32_t selectedCelestial = UINT32_MAX;
    float celestialImportance = 0.0f;
    for (size_t i = 0; i < lights.size(); ++i) {
        if (!enabled(i)) { continue; }
        if (!lights[i].isCelestial && selected == UINT32_MAX) { selected = static_cast<uint32_t>(i); }
        if (lights[i].isCelestial && lights[i].celestial.shadowImportance > celestialImportance) {
            celestialImportance = lights[i].celestial.shadowImportance;
            selectedCelestial = static_cast<uint32_t>(i);
        }
    }
    return selectedCelestial != UINT32_MAX ? selectedCelestial : selected;
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
    float angularRadius = 0.0f;
#if METALLIC_HAS_NRD
    NRDRuntime sigma;
#endif
};

void ScreenSpaceShadows::clear()
{
    state_.reset();
    trace_.clear();
    traceEncoder_.clear();
}

Result<ScreenSpaceShadowResult> ScreenSpaceShadows::record(
    Device& device,
    CommandBuffer& commands,
    Streamer& streamer,
    TextureView& depth,
    const ViewConstants& view,
    std::span<const ScreenSpaceShadowLightRecord> lights,
    uint64_t sceneRevision,
    uint64_t transformRevision,
    const ScreenSpaceShadowSettings& settings,
    std::string& log,
    Buffer& textureSamplingMetadata,
    ScenePathTraceResources* geometry,
    const MeshletStreamDeferredGPUResourcesView* streamGeometry,
    CPUProfileRecorder* profiler,
    RayTracingAccelerationStructure* accelerationStructure,
    const std::function<void(std::string_view)>& checkpoint)
{
    ScreenSpaceShadowResult output{};
    CPUProfileScope profile(profiler, "Validate shadow resources");
    output = {};
    const uint32_t width = static_cast<uint32_t>(view.current.viewport[1]);
    const uint32_t height = static_cast<uint32_t>(view.current.viewport[2]);
    if (width == 0 || height == 0 || width > UINT16_MAX || height > UINT16_MAX ||
        !std::isfinite(settings.maxDistance) || settings.maxDistance <= 0 || settings.maxDistance > 1000000 ||
        !std::isfinite(settings.normalBias) || settings.normalBias < 0 ||
        !std::isfinite(settings.lightRadius) || settings.lightRadius < 0) {
        log = "Invalid ray-traced shadow dimensions or trace settings";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().rayQuery || !device.capabilities().rayTracingAccelerationStructure || !device.capabilities().clusterAccelerationStructure) {
        log = "Ray-traced shadows require ray queries and a scene acceleration structure";
        return makeError(Error::Unsupported);
    }
    if (streamGeometry == nullptr || geometry == nullptr || !geometry->valid()) {
        log = "Ray-traced shadows require prepared streaming geometry and material resources";
        return makeError(Error::InvalidArgument);
    }
    if (streamGeometry->accelerationStructure == nullptr) {
        log = "Ray-traced shadows require a ready CLAS scene; defer rendering until streaming initialization completes";
        return makeError(Error::InvalidArgument);
    }
    if (accelerationStructure != nullptr && accelerationStructure != streamGeometry->accelerationStructure) {
        log = "Ray-traced shadows require the CLAS TLAS and geometry buffers from the same streaming session";
        return makeError(Error::InvalidArgument);
    }
    profile.next("Prepare trace pipeline");
    auto& trace = trace_;
    constexpr ComputeResourceMember shadowOutputs[] = {
        METALLIC_RESOURCE_MEMBER(SceneResourceParameters, penumbra),
        METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowNormal),
        METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowViewZ),
        METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowMotion),
        METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowOutput)
    };
    const uint32_t textureCount = geometry->materialTextureCount();
    if (trace.valid() && traceTextureCount_ != textureCount) {
        if (auto* frame = metallic::render::RenderFrameContext::from(commands)) { frame->retain(std::make_shared<ComputeKernel>(std::move(trace))); }
        else { (void)device.waitIdle(); trace.clear(); }
        traceEncoder_.clear();
    }
    traceTextureCount_ = textureCount;
    if (!trace.valid()) {
        const auto request = makeShadowShaderRequest();
        const ShaderRequestView source(request);
        ShaderCompileResult shader;
        auto result = ShaderRegistry::instance().getShader(source.desc(), shader.diagnostics)
            .transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result.transform([&] { return std::move(output); }); }
        std::vector<ComputeResourceBindingDesc> layout = {
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowParameters)}, {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowDepth), .kind = ComputeResourceBindingKind::SampledImage},
        };
        for (const auto member : shadowOutputs) {
            layout.push_back({.binding = member, .kind = ComputeResourceBindingKind::StorageImage});
        }
        {
            layout.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene), .kind = ComputeResourceBindingKind::AccelerationStructure});
            layout.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues)});
        }
        {
            layout.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials)});
            layout.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata)});
            layout.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), .kind = ComputeResourceBindingKind::SampledImage, .descriptorCount = textureCount});
            for (const auto member : {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPages), METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPageTable), METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamInstances), METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamHeader), METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamChanges), METALLIC_RESOURCE_MEMBER(SceneResourceParameters, streamPageCount)}) { layout.push_back({.binding = member}); }
        }
        result = initializeResourceKernel(device, {
            .spirv = shader.spirv,
            .pushConstantSize = sizeof(uint32_t),
            .bindings = layout,
            .debugName = "Ray-traced shadows",
            .requiresRayQuery = true,
            .resourceParameterSize = sizeof(SceneResourceParameters),
        }, trace, traceEncoder_, log);
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
    if (auto* frame = metallic::render::RenderFrameContext::from(commands)) { frame->retain(state); }
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
    if (auto* frame = metallic::render::RenderFrameContext::from(commands)) { frame->retain(state->parameters); }
    auto result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>([] {},
        [state] { state->cancelled = true; }));
    if (!result) { return makeError(result.error()); }

    // Deferred compares the domain-tagged identity with the lighting sample.
    const uint32_t selected = selectScreenSpaceShadowLight(lights, settings.lightIndex);
    ScreenSpaceShadowParameters parameters{};
    parameters.view = view;
    if (selected != UINT32_MAX) {
        const auto& light = lights[selected];
        parameters.control[1] = light.isCelestial ? 1u : 0u;
        parameters.control[2] = light.sourceIndex | (light.isCelestial ? kCelestialLightSourceTag : 0u);
        if (light.isCelestial) {
            for (size_t i = 0; i < 3; ++i) {
                parameters.light.directionType[i] = light.celestial.direction[i];
                parameters.light.colorIntensity[i] = light.celestial.irradiance[i];
            }
            parameters.light.colorIntensity[3] = 1.0f;
            parameters.trace[3] = static_cast<float>(std::tan(std::min(
                double(light.celestial.angularRadius), 1.57079632679489661923 - 1e-6)));
        } else {
            parameters.light = light.local;
        }
    } else {
        parameters.control[2] = UINT32_MAX;
    }
    parameters.trace[0] = settings.maxDistance;
    parameters.trace[2] = settings.normalBias;
    parameters.control[0] = settings.enabled && selected != UINT32_MAX;
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
            if (auto commandResult = commands.clearColorTexture(*state->textures[i], TextureLayout::TransferDestination, {1, 1, 1, 1}); !commandResult) { return makeError(commandResult.error()); }
        }
    }
    profile.next("Prepare dispatch bindings");
    TextureView* depthView = &depth;
    std::vector<ComputeDispatchBinding> bindings{
        {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowParameters), .buffer = state->parameters.get()},
        {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, shadowDepth), .textureViews = {&depthView, 1}},
    };
    for (uint32_t i = 0; i < 5; ++i) {
        bindings.push_back({.binding = shadowOutputs[i], .textureView = state->views[i].get()});
    }
    uint32_t geometryPush = 0;
    {
        CPUProfileScope resources(profiler, "Prepare material textures");
        if (auto* frame = metallic::render::RenderFrameContext::from(commands)) { frame->retain(std::make_shared<ScenePathTraceResources>(*geometry)); }
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene),
            .accelerationStructure = streamGeometry->accelerationStructure});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials), .buffer = geometry->materialBuffer()});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata), .buffer = &textureSamplingMetadata});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), .textureViews = {geometry->materialTextureViews().data(), textureCount}, .sampledImages = geometry->materialTextureSnapshot()});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPages), .buffer = streamGeometry->pageBuffer});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPageTable), .buffer = streamGeometry->pageTableBuffer});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamInstances), .buffer = streamGeometry->instanceBuffer});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamHeader), .buffer = streamGeometry->activeHeaderBuffer});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamChanges), .buffer = streamGeometry->shadingChangeBuffer});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, streamPageCount), .buffer = streamGeometry->paramsBuffer});
        geometryPush = textureCount;
    }

    profile.next("Record trace dispatch");
    if (geometry) {
        const auto inputs = geometry->materialBinding();
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues), .buffer = inputs->valueBuffer() ? inputs->valueBuffer() : inputs->buffer()});
        if (auto* frame = RenderFrameContext::from(commands)) { frame->retain(inputs); }
    }
    result = enterStage(1);
    if (!result) { return makeError(result.error()); }
    // The trace interval starts after the access barrier so it measures the
    // actual dispatch separately from setup and final output transitions.
    if (checkpoint) { checkpoint("BeforeShadowTrace"); }
    commands.beginDebugLabel({.name = "Ray-traced shadows"});
    result = dispatchResources(trace, traceEncoder_, {
        .commandBuffer = &commands,
        .bindings = bindings,
        .pushData = &geometryPush,
        .pushDataSize = sizeof(geometryPush),
        .groupCountX = (width + 7) / 8,
        .groupCountY = (height + 7) / 8,
        .profiler = profiler,
    });
    commands.endDebugLabel();
    if (checkpoint) { checkpoint("AfterShadowTrace"); }
    if (!result) { return makeError(result.error()); }
    profile.next("Record denoising");
#if METALLIC_HAS_NRD
    if (denoise) {
        if (checkpoint) { checkpoint("BeforeShadowDenoise"); }
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
            transformRevision != state->transformRevision || (parameters.control[0] != 0) != state->traceEnabled ||
            selected != state->lightIndex || settings != state->settings ||
            parameters.trace[3] != state->angularRadius ||
            std::memcmp(&parameters.light, &state->light, sizeof(GPUPunctualLight)) != 0
            ? denoising::AccumulationMode::CLEAR_AND_RESTART : denoising::AccumulationMode::CONTINUE;
        result = state->sigma.setCommonSettings(common);
        if (!result) { return makeError(result.error()); }
        denoising::SigmaSettings sigma;
        sigma.maxStabilizedFrameNum = std::min(settings.historyLength, denoising::SIGMA_MAX_HISTORY_FRAME_NUM);
        if (parameters.control[1] != 0) {
            for (size_t i = 0; i < 3; ++i) { sigma.lightDirection[i] = -parameters.light.directionType[i]; }
        }
        result = state->sigma.setSigmaSettings(sigma);
        if (result) { result = state->sigma.denoise(NRDDenoiserMode::Sigma, commands); }
        if (checkpoint) { checkpoint("AfterShadowDenoise"); }
        if (!result) { return makeError(result.error()); }
    }
#endif
    profile.next("Finalize shadow state");
    if (checkpoint) { checkpoint("BeforeShadowFinalize"); }
    result = enterStage(3);
    if (!result) { return makeError(result.error()); }
    state->initialized = true;
    state->sceneRevision = sceneRevision;
    state->transformRevision = transformRevision;
    state->lightIndex = selected;
    state->light = parameters.light;
    state->settings = settings;
    state->traceEnabled = parameters.control[0] != 0;
    state->angularRadius = parameters.trace[3];
    output = {.texture = state->textures[4].get(), .shadow = state->views[4].get(), .parameters = state->parameters.get()};
    if (checkpoint) { checkpoint("AfterShadowFinalize"); }
    return output;
}

} // namespace metallic::render
