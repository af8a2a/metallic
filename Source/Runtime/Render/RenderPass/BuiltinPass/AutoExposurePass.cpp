#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PostProcessParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <chrono>

namespace metallic::render::builtin_pass {
namespace {

class AutoExposurePass final : public ComputePass {
public:
    // Adaptation stays on graphics. The private history import lets the stage
    // planner order consecutive frames without a CPU readback or wait.
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return false; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        auto& source = reflection.addTextureInput("source", "Linear physical HDR radiance; no tone mapping");
        source.sampledRead();
        source.format = Format::Unknown;
        auto& color = reflection.addTextureOutput("color", "Exposed color; HDR display mapping occurs in FinalBlit");
        // Apply dispatches cover every output texel, including every tone-curve
        // branch; adaptation history remains in the separate persistent buffer.
        color.storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        color.format = Format::RGBA16Sfloat;
        color.colorEncoding = DisplayColorEncoding::ExposedLinear;
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
            runtimeFloatSetting("artisticExposure", "Output Multiplier", 1.0f, 0.001f, 16.0f),
            runtimeFloatSetting("sourceExposure", "Input Pre-exposure", 1.0f, 0.000001f, 65536.0f),
            RenderGraphRuntimeSetting{.key = "resetSerial", .label = "Reset Adaptation",
                .type = RenderGraphRuntimeSettingType::ActionCounter, .defaultValue = 0},
        };
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        if (context.device == nullptr) { return makeError(Error::InvalidArgument); }
        const char* entries[] = {"autoExposureHistogramMain", "autoExposureReduceMain", "autoExposureApplyMain"};
        for (size_t i = 0; i < programs_.size(); ++i) {
            if (programs_[i].valid()) { continue; }
            ShaderCompileResult shader;
            Result<> result = compileSlangShaderToSpirv({.moduleName = "Features/PostProcess/AutoExposure",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            if (!result) { log += shader.diagnostics; return result; }
            result = programs_[i].initialize(*context.device, {
                .spirv = shader.spirv,
                .parameters = parameterAbi<AutoExposureParams>(kAutoExposureABI, ParameterTransport::InlinePush),
                .debugName = entries[i],
            }, log);
            if (!result) { return result; }
        }
        state_ = std::make_shared<State>();
        return context.device->createBuffer({.size = 16, .structureStride = 16,
            .usage = BufferUsageBits::Storage}).transform([&](auto rhiValue) { state_->history = std::move(rhiValue); });
    }

    Result<> execute(RenderGraphExecutionContext& context) override
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
        case Format::RGBA16Sfloat:
        case Format::RGBA32Sfloat:
        case Format::RG32Sfloat:
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
        AutoExposurePush push{
            context.width(), context.height(), ((context.width() + 15) / 16) * ((context.height() + 15) / 16),
            reset ? 1u : 0u, settings.minEV100, settings.maxEV100, settings.compensation, lighting.exposureEV100,
            settings.lowPercent * 0.01f, settings.highPercent * 0.01f,
            settings.histogramMinEV100, settings.histogramMaxEV100,
            settings.speedUp, settings.speedDown, settings.transitionDistance, std::clamp(delta, 0.0f, 1.0f),
            settings.enabled ? 1u : 0u,
            finiteProperty(context.properties(), "sourceExposure", 1.0f, 0.000001f, 65536.0f),
            finiteProperty(context.properties(), "artisticExposure", 1.0f, 0.001f, 16.0f),
        };
        auto& commands = context.commandBuffer();
        if (auto* frame = metallic::render::RenderFrameContext::from(commands)) { frame->retain(state_); }
        Result<> result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
            [] {}, [state = state_] { state->valid = false; }));
        if (!result) { return result; }
        auto registry = metallic::render::ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, metallic::render::RenderFrameContext::from(commands));
        AutoExposureParams params{};
        params.source = writer.sampledImageHandle(source.view());
        params.output = writer.storageImageHandle(color.view());
        params.histogram = writer.bufferSpan(histogram.buffer(), {}, 4, 4);
        params.history = writer.bufferSpan(state_->history.get(), {}, 16, 16);
        params.exposure = writer.bufferSpan(exposure.buffer(), {}, 16, 16);
        params.display = push;
        auto encoded = writer.encode(params, kAutoExposureABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        const auto record = [&](size_t program, CommandBuffer& stageCommands, uint32_t x, uint32_t y) {
            return programs_[program].dispatch(stageCommands, *encoded, x, y);
        };
        using Access = RenderGraphResourceAccess;
        const RenderGraphStageUse histogramUses[] = {
            {"source", Access::TextureSampleRead}, {"histogram", Access::BufferStorageWrite},
        };
        const RenderGraphStageUse reduceUses[] = {
            {"histogram", Access::BufferStorageRead}, {"history", Access::BufferStorageReadWrite},
            {"exposure", Access::BufferStorageWrite},
        };
        const RenderGraphStageUse applyUses[] = {
            {"source", Access::TextureSampleRead}, {"exposure", Access::BufferStorageRead},
            {"color", Access::TextureStorageWrite},
        };
        const RenderGraphComputeStage stages[] = {
            {"Histogram", histogramUses, [&](CommandBuffer& stageCommands) {
                return record(0, stageCommands, (push.width + 15) / 16, (push.height + 15) / 16);
            }},
            {"Reduce", reduceUses, [&](CommandBuffer& stageCommands) { return record(1, stageCommands, 1, 1); }},
            {"Apply", applyUses, [&](CommandBuffer& stageCommands) {
                return record(2, stageCommands, (push.width + 7) / 8, (push.height + 7) / 8);
            }},
        };
        auto history = state_->history->slice();
        if (!history) { return makeError(history.error()); }
        // Content reset does not erase prior accepted GPU accesses. Keep the
        // conservative history contract even on reset/cancellation/first use.
        const RenderGraphBufferImport imports[] = {{"history", *history, Access::BufferStorageReadWrite}};
        result = context.executeComputeStages(stages, imports);
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
    Device* device_ = nullptr;
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
    std::array<ComputeKernel, 3> programs_;
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
