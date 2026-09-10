#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDlssNr.h"

namespace metallic::render::builtin_pass {
namespace {

struct SliderDebugPush {
    float splitPosition;
    uint32_t horizontal;
    uint32_t swapSides;
};
static_assert(sizeof(SliderDebugPush) == 12);

class DlssNrPass final : public UnsafePass {
public:
    bool supportsFrameOverlap() const override { return false; }
    bool supportsAsyncQueue() const override { return false; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        auto& input = reflection.addTextureInput("inputColor", "Tone-mapped sRGB display color (RGBA8 UNORM)")
            .texture2D(context.width, context.height).storageReadWrite();
        input.format = Format::Rgba8Unorm;
        input.usage = input.usage | TextureUsageBits::TransferSource | TextureUsageBits::Sampled;
        auto& motion = reflection.addTextureInput("motionVectors", "Current-to-previous UV motion, without jitter")
            .texture2D(context.width, context.height).storageReadWrite();
        motion.format = Format::Rg16Sfloat;
        motion.usage = motion.usage | TextureUsageBits::Sampled;
        auto& depth = reflection.addTextureInput("depth", "Normalized hardware depth, R32F")
            .texture2D(context.width, context.height).storageReadWrite();
        depth.format = Format::R32Sfloat;
        depth.usage = depth.usage | TextureUsageBits::Sampled;
        auto& output = reflection.addTextureOutput("color", "Experimental DLSS Neural Rendering output")
            .texture2D(context.width, context.height).storageReadWrite();
        output.format = Format::Rgba8Unorm;
        output.usage = output.usage | TextureUsageBits::TransferDestination;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        auto enabled = runtimeBoolSetting("enabled", "Enable DLSS-NR (Experimental)", true, true);
        enabled.rebuildGraph = true;
        auto preset = runtimeIntSetting("preset", "Preset", 0, 0, 3, true);
        preset.rebuildGraph = true;
        auto fallback = runtimeBoolSetting("fallbackToInput", "Pass Through When Unavailable", true);
        fallback.rebuildGraph = true;
        return {
            enabled, preset, fallback,
            runtimeIntSetting("style", "Style (0 Default, 1 Natural, 2 Cinematic)", 0, 0, 2, true),
            runtimeFloatSetting("intensity", "Intensity", 1.0f, 0.0f, 1.0f, true),
            runtimeFloatSetting("localToneStrength", "Local Tone", 1.0f, 0.0f, 1.0f, true),
            runtimeFloatSetting("localStructureStrength", "Local Structure", 1.0f, 0.0f, 1.0f, true),
            runtimeFloatSetting("skinStructureStrength", "Skin Structure (-1 Auto)", -1.0f, -1.0f, 1.0f, true),
            runtimeBoolSetting("depthInverted", "Reversed Z", true, true),
            runtimeBoolSetting("useAutoMask", "Auto Mask", false, true),
            runtimeBoolSetting("uiCorrection", "UI Correction", false, true),
            runtimeActionCounterSetting("resetSerial", "Reset NR History", true),
            runtimeBoolSetting("sliderDebug", "Slider Debug (Before / After)", false),
            runtimeFloatSetting("splitPosition", "Split Position", 0.5f, 0.0f, 1.0f),
            runtimeEnumSetting("orientation", "Divider", "vertical",
                {{"Vertical (left / right)", "vertical"}, {"Horizontal (top / bottom)", "horizontal"}}),
            runtimeBoolSetting("swapSides", "Swap Before / After", false),
        };
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        runtime_.reset();
        sliderProgram_.clear();
        device_ = context.device;
        hasHistory_ = false;
        failed_ = false;
        if (context.device == nullptr || context.graphicsQueue == nullptr ||
            context.width == 0 || context.height == 0) {
            log = "DlssNrPass requires a device, graphics queue and non-zero dimensions";
            return makeError(Error::InvalidArgument);
        }
        if (!properties().value("enabled", true)) { return {}; }
        auto runtime = std::make_unique<vulkan::DlssNrContext>();
        auto result = runtime->initialize(*context.device, log);
        if (!result) {
            if (!properties().value("fallbackToInput", true)) { return result; }
            spdlog::warn("[DLSS-NR] {}; passing through input color", log);
            return {};
        }
        runtime_ = std::move(runtime);
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        const auto input = context.inputTexture("inputColor");
        const auto output = context.outputTexture("color");
        const auto motion = context.inputTexture("motionVectors");
        const auto depth = context.inputTexture("depth");
        const auto& properties = context.properties();
        vulkan::DlssNrDesc desc{
            .inputColor = {input.texture(), input.view()},
            .outputColor = {output.texture(), output.view()},
            .motionVectors = {motion.texture(), motion.view()},
            .depth = {depth.texture(), depth.view()},
        };
        auto& settings = desc.settings;
        try {
            settings.preset = properties.value("preset", 0u);
            settings.style = properties.value("style", 0u);
            settings.intensity = properties.value("intensity", 1.0f);
            settings.localToneStrength = properties.value("localToneStrength", 1.0f);
            settings.localStructureStrength = properties.value("localStructureStrength", 1.0f);
            settings.skinStructureStrength = properties.value("skinStructureStrength", -1.0f);
            settings.depthInverted = properties.value("depthInverted", true);
            settings.useAutoMask = properties.value("useAutoMask", false);
            settings.uiCorrection = properties.value("uiCorrection", false);
        } catch (const RenderGraphProperties::exception& error) {
            spdlog::error("[DLSS-NR] Invalid pass properties: {}", error.what());
            return makeError(Error::InvalidArgument);
        }
        std::string log;
        auto result = vulkan::validateDlssNrDesc(desc, log);
        if (!result) { spdlog::error("[DLSS-NR] {}", log); return result; }
        if (runtime_ == nullptr || failed_ || !properties.value("enabled", true) || settings.intensity == 0.0f) {
            hasHistory_ = false;
            copyColor(context.commandBuffer(), input, output);
            return {};
        }
        const uint64_t revision = context.historyResources() != nullptr
            ? context.historyResources()->invalidationRevision() : 0;
        const uint64_t scene = context.runtimeScene() != nullptr ? context.runtimeScene()->resourceIdentity() : 0;
        auto historyProperties = properties;
        // These controls only reveal the original pixels after NR evaluation;
        // moving the divider must not reset the neural temporal history.
        for (const char* key : {"sliderDebug", "splitPosition", "orientation", "swapSides"}) {
            historyProperties.erase(key);
        }
        settings.reset = !hasHistory_ || lastFrame_ + 1 != context.frameIndex() ||
            lastRevision_ != revision || lastScene_ != scene || lastProperties_ != historyProperties;
        const bool sliderDebug = properties.value("sliderDebug", false);
        if (sliderDebug) {
            result = initializeSliderDebug(log);
            if (!result) { spdlog::error("[DLSS-NR] Slider debug: {}", log); return result; }
        }
        // Metallic guides are already current-to-previous UV displacements.
        // NGX feature 18 takes pixels: unlike Unity, do not negate them.
        settings.motionVectorScaleX = static_cast<float>(input.desc().width);
        settings.motionVectorScaleY = static_cast<float>(input.desc().height);
        result = runtime_->evaluate(context.commandBuffer(), desc, log);
        if (!result) {
            hasHistory_ = false;
            if (!properties.value("fallbackToInput", true) || hasError(result, Error::InvalidArgument) ||
                hasError(result, Error::DeviceLost)) {
                spdlog::error("[DLSS-NR] {}", log);
                return result;
            }
            spdlog::warn("[DLSS-NR] {}; passing through until the pass is recompiled", log);
            // Keep the context alive: failed create/evaluate may have recorded
            // commands that reference its resources in the current submission.
            failed_ = true;
            copyColor(context.commandBuffer(), input, output);
            return {};
        }
        hasHistory_ = true;
        lastFrame_ = context.frameIndex();
        lastRevision_ = revision;
        lastScene_ = scene;
        lastProperties_ = std::move(historyProperties);
        return sliderDebug ? drawSliderDebug(context, input, output) : Result{};
    }

private:
    Result initializeSliderDebug(std::string& log)
    {
        if (sliderProgram_.valid()) { return {}; }
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({.moduleName = "Features/Debug/SliderDebug",
            .entryPointName = "sliderDebugOverlayMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = ComputeResourceBindingKind::SampledImage},
            {.binding = 2, .kind = ComputeResourceBindingKind::StorageImage},
        };
        return sliderProgram_.initialize(*device_, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = sizeof(SliderDebugPush),
            .bindings = bindings, .bindingCount = 2, .debugName = "DlssNrSliderDebug", .requiresRayQuery = false}, log);
    }

    Result drawSliderDebug(RenderGraphExecutionContext& context, TextureHandle input, TextureHandle output)
    {
        const auto& properties = context.properties();
        const float split = properties.value("splitPosition", 0.5f);
        const SliderDebugPush push{
            std::isfinite(split) ? std::clamp(split, 0.0f, 1.0f) : 0.5f,
            properties.value("orientation", "vertical") == "horizontal" ? 1u : 0u,
            properties.value("swapSides", false) ? 1u : 0u,
        };
        auto& command = context.commandBuffer();
        const TextureBarrierDesc barriers[] = {
            {.texture = input.texture(), .before = ResourceState::General, .after = ResourceState::ShaderRead},
            {.texture = output.texture(), .before = ResourceState::General, .after = ResourceState::General},
        };
        command.barrier({.textures = barriers, .textureCount = 2});
        auto* source = input.view();
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureViews = &source, .textureViewCount = 1},
            {.binding = 2, .textureView = output.view()},
        };
        auto result = sliderProgram_.dispatch({.commandBuffer = &command,
            .bindings = bindings, .bindingCount = 2, .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8, .groupCountY = (context.height() + 7) / 8});
        const TextureBarrierDesc restore{
            .texture = input.texture(), .before = ResourceState::ShaderRead, .after = ResourceState::General};
        command.barrier({.textures = &restore, .textureCount = 1});
        return result;
    }

    static void copyColor(CommandBuffer& command, TextureHandle input, TextureHandle output)
    {
        TextureBarrierDesc barriers[] = {
            {.texture = input.texture(), .before = ResourceState::General, .after = ResourceState::TransferSource},
            {.texture = output.texture(), .before = ResourceState::General, .after = ResourceState::TransferDestination},
        };
        command.barrier({.textures = barriers, .textureCount = 2});
        command.copyTexture({.source = input.texture(), .destination = output.texture(),
            .width = output.desc().width, .height = output.desc().height, .depth = 1});
        for (auto& barrier : barriers) { barrier.before = barrier.after; barrier.after = ResourceState::General; }
        command.barrier({.textures = barriers, .textureCount = 2});
    }

    Device* device_ = nullptr;
    ComputeProgram sliderProgram_;
    std::unique_ptr<vulkan::DlssNrContext> runtime_;
    bool hasHistory_ = false;
    bool failed_ = false;
    uint64_t lastFrame_ = 0;
    uint64_t lastRevision_ = 0;
    uint64_t lastScene_ = 0;
    RenderGraphProperties lastProperties_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createDlssNrPass()
{
    return std::make_unique<DlssNrPass>();
}

} // namespace metallic::render::builtin_pass
