#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/RenderGraph/NrdRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"

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

std::vector<GpuPunctualLight> buildScreenSpaceShadowLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& lighting)
{
    const auto sources = buildSceneLightRecords(scene != nullptr
        ? std::span<const scene::RenderLight>(scene->lights()) : std::span<const scene::RenderLight>(), lighting.lights);
    std::vector<GpuPunctualLight> lights(sources.size() + 1);
    lights[0].positionRange[0] = static_cast<float>(sources.size());
    for (size_t i = 0; i < sources.size(); ++i) { lights[i + 1] = sources[i].gpu; }
    return lights;
}

uint32_t selectScreenSpaceShadowLight(std::span<const GpuPunctualLight> lights, int32_t requestedIndex)
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
    std::unique_ptr<Buffer> parameters;
    uint32_t width = 0, height = 0;
    uint64_t sceneRevision = 0;
    uint64_t transformRevision = 0;
    GpuPunctualLight light{};
    ScreenSpaceShadowSettings settings;
    uint32_t lightIndex = UINT32_MAX;
    bool initialized = false;
    bool cancelled = false;
#if METALLIC_HAS_NRD
    NrdRuntime sigma;
#endif
};

void ScreenSpaceShadows::clear()
{
    state_.reset();
    for (auto& trace : traces_) { trace.clear(); }
}

Result ScreenSpaceShadows::record(Device& device, CommandBuffer& commands, Streamer& streamer,
    TextureView& depth, const ViewConstants& view, std::span<const GpuPunctualLight> lights,
    uint64_t sceneRevision, uint64_t transformRevision, const ScreenSpaceShadowSettings& settings,
    ScreenSpaceShadowResult& output, std::string& log, ScenePathTraceResources* geometry)
{
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
    if (geometry == nullptr || !geometry->valid()) {
        log = "Ray-traced shadows require prepared scene geometry";
        return makeError(Error::InvalidArgument);
    }
    const auto* neural = &geometry->neuralTextures();
    const bool ntc = neural->active();
    const bool coop = ntc && neural->cooperativeVectorActive();
    auto& trace = traces_[ntc ? (coop ? 2 : 1) : 0];
    constexpr uint32_t kShadowBinding = 80; // Keep the shared scene alpha-mask resource slots.
    if (!trace.valid()) {
        std::vector<const char*> capabilities{"spvRayQueryKHR"};
        if (coop) { capabilities.push_back("spvCooperativeVectorNV"); }
        std::vector<const char*> searchPaths;
#if METALLIC_HAS_NTC
        if (ntc) { searchPaths.push_back(METALLIC_NTC_SHADER_INCLUDE_DIR); }
#endif
        const SlangMacroDefine defines[] = {
            {.name = "METALLIC_HAS_NTC", .value = ntc ? "1" : "0"},
            {.name = "METALLIC_NTC_COOPERATIVE_VECTOR", .value = coop ? "1" : "0"},
        };
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({.moduleName = "Features/Lighting/ScreenSpaceShadows",
            .entryPointName = "rayTracedShadowsMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .additionalSearchPaths = searchPaths.data(), .additionalSearchPathCount = uint32_t(searchPaths.size()),
            .capabilities = capabilities.data(), .capabilityCount = uint32_t(capabilities.size()),
            .macroDefines = defines, .macroDefineCount = 2}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        std::vector<ComputeProgramBindingDesc> layout = {
            {.binding = kShadowBinding}, {.binding = kShadowBinding + 1, .kind = ComputeResourceBindingKind::SampledImage},
        };
        for (uint32_t i = 2; i < 7; ++i) {
            layout.push_back({.binding = kShadowBinding + i, .kind = ComputeResourceBindingKind::StorageImage});
        }
        layout.push_back({.binding = 0, .kind = ComputeResourceBindingKind::AccelerationStructure});
        for (uint32_t i = 2; i <= 6; ++i) { layout.push_back({.binding = i}); }
        layout.push_back({.binding = 9, .kind = ComputeResourceBindingKind::SampledImage,
            .descriptorCount = kScenePathTraceMaxMaterialTextures});
        if (ntc) {
            layout.push_back({.binding = kNeuralTextureLatentsBinding, .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kMaxNeuralTextureSets});
            layout.push_back({.binding = kNeuralTextureConstantsBinding});
            layout.push_back({.binding = kNeuralTextureWeightsBinding});
            layout.push_back({.binding = kNeuralTextureSetInfoBinding});
            layout.push_back({.binding = kNeuralTextureSamplerBinding, .kind = ComputeResourceBindingKind::Sampler});
        }
        result = trace.initialize(device, {.spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
            .pushConstantSize = 8u,
            .bindings = layout.data(), .bindingCount = uint32_t(layout.size()),
            .debugName = "Ray-traced shadows", .requiresRayQuery = true}, log);
        if (!result) { return result; }
    }
    if (!state_ || state_->cancelled || state_->width != width || state_->height != height) {
        auto next = std::make_shared<State>();
        const Format formats[] = {Format::R16Sfloat, nrdNormalRoughnessFormat(), Format::R32Sfloat,
            Format::Rgba16Sfloat, Format::R8Unorm};
        for (size_t i = 0; i < next->textures.size(); ++i) {
            auto result = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::Storage |
                TextureUsageBits::TransferDestination | TextureUsageBits::TransferSource,
                .format = formats[i], .width = width, .height = height}, next->textures[i]);
            if (!result) { return result; }
            result = device.createTextureView(*next->textures[i], {.format = formats[i]}, next->views[i]);
            if (!result) { return result; }
        }
        auto result = device.createBuffer({.size = sizeof(ScreenSpaceShadowParameters), .structureStride = sizeof(ScreenSpaceShadowParameters),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, next->parameters);
        if (!result) { return result; }
        next->width = width;
        next->height = height;
        state_ = std::move(next);
    }
    auto state = state_;
    if (auto* frame = commands.frameContext()) { frame->retain(state); }
    auto result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>([] {},
        [state] { state->cancelled = true; }));
    if (!result) { return result; }

    // Deferred compares this index with the stable LightGrid source slot.
    const uint32_t selected = selectScreenSpaceShadowLight(lights, settings.lightIndex);
    ScreenSpaceShadowParameters parameters{};
    parameters.view = view;
    if (selected != UINT32_MAX) { parameters.light = lights[selected + 1]; }
    parameters.trace[0] = settings.maxDistance;
    parameters.trace[2] = settings.normalBias;
    parameters.trace[3] = std::tan(settings.angularRadiusDegrees * 0.01745329252f);
    parameters.control[0] = settings.enabled && selected != UINT32_MAX;
    parameters.control[2] = selected;
    parameters.control[3] = settings.debug;
    parameters.shape[0] = settings.lightRadius;
    const bool denoise = METALLIC_HAS_NRD && settings.denoise && parameters.control[0];
    parameters.shape[1] = denoise ? 1.0f : 0.0f;
    void* mapped = state->parameters->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, &parameters, sizeof(parameters));
    state->parameters->flush(0, sizeof(parameters));
    state->parameters->unmap();
    commands.hostWriteBarrier();

    std::array<TextureBarrierDesc, 5> barriers;
    for (size_t i = 0; i < barriers.size(); ++i) {
        barriers[i] = {.texture = state->textures[i].get(),
            .before = state->initialized ? (i == 4 ? ResourceState::ShaderRead : ResourceState::General) : ResourceState::Undefined,
            .after = state->initialized ? ResourceState::General : ResourceState::TransferDestination,
            .mipCount = 1, .layerCount = 1};
    }
    commands.barrier({.textures = barriers.data(), .textureCount = 5});
    if (!state->initialized) {
        for (size_t i = 0; i < barriers.size(); ++i) {
            commands.clearColorTexture(*state->textures[i], ResourceState::TransferDestination, {1, 1, 1, 1});
            barriers[i].before = ResourceState::TransferDestination;
            barriers[i].after = ResourceState::General;
        }
        commands.barrier({.textures = barriers.data(), .textureCount = 5});
    }
    TextureView* depthView = &depth;
    std::vector<ComputeDispatchBinding> bindings{
        {.binding = kShadowBinding, .buffer = state->parameters.get()},
        {.binding = kShadowBinding + 1, .textureViews = &depthView, .textureViewCount = 1},
    };
    for (uint32_t i = 0; i < 5; ++i) {
        bindings.push_back({.binding = kShadowBinding + i + 2, .textureView = state->views[i].get()});
    }
    uint32_t geometryPush[2]{};
    result = geometry->uploadMaterialTextures(commands);
    if (!result) { return result; }
    if (auto* frame = commands.frameContext()) { frame->retain(std::make_shared<ScenePathTraceResources>(*geometry)); }
    bindings.push_back({.binding = 0, .accelerationStructure = geometry->accelerationStructure().accelerationStructure()});
    bindings.push_back({.binding = 2, .buffer = geometry->shadingVertexBuffer()});
    bindings.push_back({.binding = 3, .buffer = geometry->indexBuffer()});
    bindings.push_back({.binding = 4, .buffer = geometry->primitiveBuffer()});
    bindings.push_back({.binding = 5, .buffer = geometry->instanceBuffer()});
    bindings.push_back({.binding = 6, .buffer = geometry->materialBuffer()});
    bindings.push_back({.binding = 9, .textureViews = geometry->materialTextureViews().data(),
        .textureViewCount = kScenePathTraceMaxMaterialTextures});
    geometryPush[0] = geometry->materialTextureCount();
    geometryPush[1] = neural->textureSetCount();
    if (ntc) {
        bindings.push_back({.binding = kNeuralTextureLatentsBinding, .textureViews = neural->latentTextureViews().data(),
            .textureViewCount = kMaxNeuralTextureSets});
        bindings.push_back({.binding = kNeuralTextureConstantsBinding, .buffer = neural->constantsBuffer()});
        bindings.push_back({.binding = kNeuralTextureWeightsBinding, .buffer = neural->weightsBuffer()});
        bindings.push_back({.binding = kNeuralTextureSetInfoBinding, .buffer = neural->setInfoBuffer()});
        bindings.push_back({.binding = kNeuralTextureSamplerBinding, .sampler = &neural->latentSampler()});
    }
    commands.beginDebugLabel({.name = "Ray-traced shadows"});
    result = trace.dispatch({.commandBuffer = &commands, .bindings = bindings.data(), .bindingCount = uint32_t(bindings.size()),
        .pushData = geometryPush, .pushDataSize = sizeof(geometryPush),
        .groupCountX = (width + 7) / 8, .groupCountY = (height + 7) / 8});
    commands.endDebugLabel();
    if (!result) { return result; }
#if METALLIC_HAS_NRD
    if (denoise) {
        if (!state->sigma.valid()) {
            NrdUserTexturePool pool{};
            const denoising::ResourceType resources[] = {denoising::ResourceType::IN_PENUMBRA,
                denoising::ResourceType::IN_NORMAL_ROUGHNESS, denoising::ResourceType::IN_VIEWZ,
                denoising::ResourceType::IN_MV, denoising::ResourceType::OUT_SHADOW_TRANSLUCENCY};
            for (size_t i = 0; i < 5; ++i) { pool[static_cast<size_t>(resources[i])] = {state->textures[i].get(), state->views[i].get()}; }
            result = state->sigma.initialize(device, static_cast<uint16_t>(width), static_cast<uint16_t>(height), pool, log, true);
            if (!result) { return result; }
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
            transformRevision != state->transformRevision ||
            selected != state->lightIndex || settings != state->settings ||
            std::memcmp(&parameters.light, &state->light, sizeof(GpuPunctualLight)) != 0
            ? denoising::AccumulationMode::CLEAR_AND_RESTART : denoising::AccumulationMode::CONTINUE;
        result = state->sigma.setCommonSettings(common);
        if (!result) { return result; }
        denoising::SigmaSettings sigma;
        sigma.maxStabilizedFrameNum = std::min(settings.historyLength, denoising::SIGMA_MAX_HISTORY_FRAME_NUM);
        if (parameters.light.directionType[3] < 0.5f) {
            for (size_t i = 0; i < 3; ++i) { sigma.lightDirection[i] = -parameters.light.directionType[i]; }
        }
        result = state->sigma.setSigmaSettings(sigma);
        if (result) { result = state->sigma.denoise(NrdDenoiserMode::Sigma, commands, streamer); }
        if (!result) { return result; }
    }
#endif
    const TextureBarrierDesc shadowReady{.texture = state->textures[4].get(),
        .before = ResourceState::General, .after = ResourceState::ShaderRead, .mipCount = 1, .layerCount = 1};
    commands.barrier({.textures = &shadowReady, .textureCount = 1});
    state->initialized = true;
    state->sceneRevision = sceneRevision;
    state->transformRevision = transformRevision;
    state->lightIndex = selected;
    state->light = parameters.light;
    state->settings = settings;
    output = {.texture = state->textures[4].get(), .shadow = state->views[4].get(), .parameters = state->parameters.get()};
    return {};
}

} // namespace metallic::render
