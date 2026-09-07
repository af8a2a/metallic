#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <chrono>

namespace metallic::render::builtin_pass {
namespace {

struct AutoExposurePush {
    uint32_t width, height, tileCount, resetHistory;
    float minEV100, maxEV100, compensation, manualEV100;
    float lowPercent, highPercent, histogramMin, histogramMax;
    float speedUp, speedDown, transitionDistance, deltaSeconds;
    uint32_t automatic, toneCurve;
    float sourceExposure, artisticExposure;
};
static_assert(sizeof(AutoExposurePush) == 80);

class AutoExposurePass final : public ComputePass {
public:
    // All adaptation dispatches stay on graphics. The explicit history barrier
    // orders consecutive frames on that queue without a CPU readback or wait.
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return false; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        auto& source = reflection.addTextureInput("source", "Linear physical HDR radiance; no tone mapping");
        source.sampledRead();
        source.format = Format::Unknown;
        reflection.addTextureOutput("color", "Exposed and tone-mapped display color")
            .storageReadWrite().format = Format::Rgba8Unorm;
        reflection.addBufferOutput("histogram", "64-bin luminance histogram per 16x16 tile")
            .buffer(uint64_t((context.width + 15) / 16) * ((context.height + 15) / 16) * 64 * 4, 4)
            .storageReadWrite();
        reflection.addBufferOutput("exposure", "float4: multiplier, adapted EV100, target EV100, luminance")
            .buffer(16, 16).storageReadWrite();
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeEnumSetting("toneCurve", "Tone Curve", "reinhard",
                {{"Reinhard", "reinhard"}, {"Exponential (RTXDI)", "exponential"},
                    {"None (sRGB)", "none"}}),
            runtimeFloatSetting("artisticExposure", "Output Multiplier", 1.0f, 0.001f, 16.0f),
            runtimeFloatSetting("sourceExposure", "Input Pre-exposure", 1.0f, 0.000001f, 65536.0f),
            RenderGraphRuntimeSetting{.key = "resetSerial", .label = "Reset Adaptation",
                .type = RenderGraphRuntimeSettingType::ActionCounter, .defaultValue = 0},
        };
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) { return makeError(Error::InvalidArgument); }
        const ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = ComputeResourceBindingKind::SampledImage},
            {.binding = 1, .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = 2, .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = 3, .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = 4, .kind = ComputeResourceBindingKind::StorageImage},
        };
        const char* entries[] = {"autoExposureHistogramMain", "autoExposureReduceMain", "autoExposureApplyMain"};
        for (size_t i = 0; i < programs_.size(); ++i) {
            if (programs_[i].valid()) { continue; }
            ShaderCompileResult shader;
            Result result = compileSlangShaderToSpirv({.moduleName = "AutoExposure",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
            if (!result) { log += shader.diagnostics; return result; }
            result = programs_[i].initialize(*context.device, {.spirv = shader.spirv.data(),
                .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = sizeof(AutoExposurePush),
                .bindings = bindings, .bindingCount = 5, .debugName = entries[i], .requiresRayQuery = false}, log);
            if (!result) { return result; }
        }
        state_ = std::make_shared<State>();
        return context.device->createBuffer({.size = 16, .structureStride = 16,
            .usage = BufferUsageBits::Storage}, state_->history);
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        const TextureHandle source = context.inputTexture("source");
        const TextureHandle color = context.outputTexture("color");
        const BufferHandle histogram = context.outputBuffer("histogram");
        const BufferHandle exposure = context.outputBuffer("exposure");
        if (!source.valid() || source.view() == nullptr || !color.valid() || color.view() == nullptr ||
            !histogram.valid() || !exposure.valid() || state_ == nullptr || state_->history == nullptr ||
            source.desc().width != context.width() || source.desc().height != context.height()) {
            return makeError(Error::InvalidArgument);
        }
        switch (source.desc().format) {
        case Format::Rgba16Sfloat:
        case Format::Rgba32Sfloat:
        case Format::Rg32Sfloat:
        case Format::R32Sfloat:
        case Format::B10G11R11UfloatPack32:
            break;
        default:
            return makeError(Error::InvalidArgument);
        }
        const scene::LightingSettings fallback;
        const auto& lighting = context.world() != nullptr ? context.world()->lighting() : fallback;
        const auto& settings = lighting.autoExposure;
        if (!scene::validAutoExposureSettings(settings)) { return makeError(Error::InvalidArgument); }
        const auto now = std::chrono::steady_clock::now();
        const float elapsed = std::chrono::duration<float>(now - lastTime_).count();
        const uint64_t identity = context.runtimeScene() != nullptr ? context.runtimeScene()->resourceIdentity() : 0;
        const int resetSerial = context.properties().value("resetSerial", 0);
        const bool reset = !state_->valid || width_ != context.width() || height_ != context.height() ||
            sceneIdentity_ != identity || resetSerial_ != resetSerial || automatic_ != settings.enabled;
        // A fixed timestep is useful for offline rendering and deterministic GPU tests.
        const float fixedDelta = context.properties().value("adaptationDeltaSeconds", 0.0f);
        const float delta = std::isfinite(fixedDelta) && fixedDelta > 0.0f ? fixedDelta : elapsed;
        const std::string toneCurve = context.properties().value("toneCurve", "reinhard");
        AutoExposurePush push{
            context.width(), context.height(), ((context.width() + 15) / 16) * ((context.height() + 15) / 16),
            reset ? 1u : 0u, settings.minEV100, settings.maxEV100, settings.compensation, lighting.exposureEV100,
            settings.lowPercent * 0.01f, settings.highPercent * 0.01f,
            settings.histogramMinEV100, settings.histogramMaxEV100,
            settings.speedUp, settings.speedDown, settings.transitionDistance, std::clamp(delta, 0.0f, 1.0f),
            settings.enabled ? 1u : 0u, toneCurve == "none" ? 2u : (toneCurve == "exponential" ? 1u : 0u),
            finiteProperty(context.properties(), "sourceExposure", 1.0f, 0.000001f, 65536.0f),
            finiteProperty(context.properties(), "artisticExposure", 1.0f, 0.001f, 16.0f),
        };
        auto& commands = context.commandBuffer();
        if (auto* frame = commands.frameContext()) { frame->retain(state_); }
        Result result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
            [] {}, [state = state_] { state->valid = false; }));
        if (!result) { return result; }
        const BufferBarrierDesc historyBarrier{.buffer = state_->history.get(),
            .before = state_->valid ? ResourceState::General : ResourceState::Undefined, .after = ResourceState::General};
        commands.barrier({.buffers = &historyBarrier, .bufferCount = 1});
        TextureView* sourceView = source.view();
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureViews = &sourceView, .textureViewCount = 1},
            {.binding = 1, .buffer = histogram.buffer()},
            {.binding = 2, .buffer = state_->history.get()},
            {.binding = 3, .buffer = exposure.buffer()},
            {.binding = 4, .textureView = color.view()},
        };
        ComputeDispatchDesc dispatch{.commandBuffer = &commands, .bindings = bindings, .bindingCount = 5,
            .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = (push.width + 15) / 16, .groupCountY = (push.height + 15) / 16};
        result = programs_[0].dispatch(dispatch);
        if (!result) { return result; }
        const BufferBarrierDesc histogramBarrier{.buffer = histogram.buffer(),
            .before = ResourceState::General, .after = ResourceState::General};
        commands.barrier({.buffers = &histogramBarrier, .bufferCount = 1});
        dispatch.groupCountX = dispatch.groupCountY = 1;
        result = programs_[1].dispatch(dispatch);
        if (!result) { return result; }
        const BufferBarrierDesc exposureBarrier{.buffer = exposure.buffer(),
            .before = ResourceState::General, .after = ResourceState::General};
        commands.barrier({.buffers = &exposureBarrier, .bufferCount = 1});
        dispatch.groupCountX = (push.width + 7) / 8;
        dispatch.groupCountY = (push.height + 7) / 8;
        result = programs_[2].dispatch(dispatch);
        if (!result) { return result; }
        state_->valid = true;
        width_ = context.width();
        height_ = context.height();
        sceneIdentity_ = identity;
        resetSerial_ = resetSerial;
        automatic_ = settings.enabled;
        lastTime_ = now;
        return {};
    }

private:
    static float finiteProperty(const RenderGraphProperties& properties, const char* key,
        float fallback, float minimum, float maximum)
    {
        const auto value = properties.find(key);
        if (value == properties.end() || !value->is_number()) { return fallback; }
        const float number = value->get<float>();
        return std::isfinite(number) ? std::clamp(number, minimum, maximum) : fallback;
    }

    struct State {
        std::unique_ptr<Buffer> history;
        bool valid = false;
    };
    std::array<ComputeProgram, 3> programs_;
    std::shared_ptr<State> state_;
    std::chrono::steady_clock::time_point lastTime_{};
    uint64_t sceneIdentity_ = 0;
    uint32_t width_ = 0, height_ = 0;
    int resetSerial_ = 0;
    bool automatic_ = false;
};

} // namespace

std::unique_ptr<RenderGraphPass> createAutoExposurePass()
{
    return std::make_unique<AutoExposurePass>();
}

} // namespace metallic::render::builtin_pass
