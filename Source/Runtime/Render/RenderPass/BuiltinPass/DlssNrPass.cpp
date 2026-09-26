#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDlssNr.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"

#include <array>
#include <vector>

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
    bool supportsFrameOverlap() const override { return !properties().value("enabled", true); }
    bool supportsAsyncQueue() const override { return false; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        auto& input = reflection.addTextureInput("inputColor", "Tone-mapped sRGB display color (RGBA8 UNORM)")
            .texture2D(context.width, context.height).storageReadWrite();
        const bool hdr = context.displayOutput.mode == DisplayOutputMode::HdrScRgb;
        const Format colorFormat = hdr ? Format::Rgba16Sfloat : Format::Rgba8Unorm;
        input.format = colorFormat;
        input.usage = input.usage | TextureUsageBits::TransferSource | TextureUsageBits::Sampled;
        input.stageAccess(RenderGraphResourceAccess::TextureTransferRead)
            .stageAccess(RenderGraphResourceAccess::TextureSampleRead);
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
        output.format = colorFormat;
        output.colorEncoding = hdr ? DisplayColorEncoding::ExposedLinear : DisplayColorEncoding::Srgb;
        output.usage = output.usage | TextureUsageBits::TransferDestination;
        output.stageAccess(RenderGraphResourceAccess::TextureTransferWrite);
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

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        runtime_.reset();
        sliderProgram_.clear();
        device_ = context.device;
        hasHistory_ = false;
        failed_ = false;
        hdrBypass_ = context.displayOutput.mode == DisplayOutputMode::HdrScRgb;
        if (context.device == nullptr || context.graphicsQueue == nullptr ||
            context.width == 0 || context.height == 0) {
            log = "DlssNrPass requires a device, graphics queue and non-zero dimensions";
            return makeError(Error::InvalidArgument);
        }
        if (!properties().value("enabled", true)) { return {}; }
        if (hdrBypass_) {
            log = "DLSS-NR currently requires SDR RGBA8 input; preserving HDR input without NR";
            if (!properties().value("fallbackToInput", true)) { return makeError(Error::Unsupported); }
            spdlog::warn("[DLSS-NR] {}", log);
            return {};
        }
        auto runtime = std::make_unique<vulkan::DlssNrContext>();
        auto result = runtime->initialize(*context.device, log);
        if (!result) {
            if (!properties().value("fallbackToInput", true)) { return result; }
            spdlog::warn("[DLSS-NR] {}; passing through input color", log);
            failed_ = true;
            return {};
        }
        runtime_ = std::move(runtime);
        return {};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto input = context.inputTexture("inputColor");
        const auto output = context.outputTexture("color");
        const auto motion = context.inputTexture("motionVectors");
        const auto depth = context.inputTexture("depth");
        const auto& properties = context.properties();
        if (hdrBypass_) {
            if (properties.value("enabled", true) && !properties.value("fallbackToInput", true)) {
                return makeError(Error::Unsupported);
            }
            return executeCopy(context, input, output);
        }
        // Enabling a bypassed NR node can reuse the compiled pass. Initialize
        // lazily here so the toggle actually activates the feature.
        if (runtime_ == nullptr && !failed_ && properties.value("enabled", true)) {
            auto runtime = std::make_unique<vulkan::DlssNrContext>();
            std::string log;
            auto result = runtime->initialize(*device_, log);
            if (!result) {
                if (!properties.value("fallbackToInput", true)) { return result; }
                spdlog::warn("[DLSS-NR] {}; passing through input color", log);
                failed_ = true;
            } else {
                runtime_ = std::move(runtime);
            }
        }
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
            return executeCopy(context, input, output);
        }
        const auto* view = context.viewConstants();
        const uint64_t revision = context.historyResources() != nullptr
            ? (view != nullptr ? context.historyResources()->reprojectionInvalidationRevision()
                : context.historyResources()->invalidationRevision()) : 0;
        const uint64_t scene = context.runtimeScene() != nullptr ? context.runtimeScene()->resourceIdentity() : 0;
        auto historyProperties = properties;
        if (view != nullptr) { historyProperties.erase("camera"); }
        // These controls only reveal the original pixels after NR evaluation;
        // moving the divider must not reset the neural temporal history.
        for (const char* key : {"sliderDebug", "splitPosition", "orientation", "swapSides"}) {
            historyProperties.erase(key);
        }
        settings.reset = !hasHistory_ || lastFrame_ + 1 != context.frameIndex() ||
            (view != nullptr && view->frame[1] == 0) ||
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
        using Access = RenderGraphResourceAccess;
        const std::array sdkUses{RenderGraphStageUse{"inputColor", Access::TextureStorageReadWrite},
            RenderGraphStageUse{"motionVectors", Access::TextureStorageReadWrite},
            RenderGraphStageUse{"depth", Access::TextureStorageReadWrite},
            RenderGraphStageUse{"color", Access::TextureStorageReadWrite}};
        const std::array sliderUses{RenderGraphStageUse{"inputColor", Access::TextureSampleRead},
            RenderGraphStageUse{"color", Access::TextureStorageReadWrite}};
        bool fallback = false;
        std::vector<RenderGraphStage> stages;
        stages.push_back({"DLSS NR evaluate", sdkUses, [&](CommandBuffer& command) -> Result<> {
            auto evaluated = runtime_->evaluate(command, desc, log);
            if (evaluated) { return {}; }
            hasHistory_ = false;
            if (!properties.value("fallbackToInput", true) || hasError(evaluated, Error::InvalidArgument) ||
                hasError(evaluated, Error::DeviceLost)) {
                spdlog::error("[DLSS-NR] {}", log);
                return evaluated;
            }
            spdlog::warn("[DLSS-NR] {}; passing through until the pass is recompiled", log);
            // The SDK may already have recorded resource accesses on failure.
            // Keep it alive. The local fallback plan restores the opaque SDK
            // stage's General boundary; successful frames never enter transfer
            // layouts. Reflection includes the fallback's transfer accesses.
            failed_ = true;
            fallback = true;
            return copyColorAfterSdkFailure(command, input, output);
        }, RenderGraphPassKind::Unsafe});
        if (sliderDebug) {
            stages.push_back({"DLSS NR slider", sliderUses, [&](CommandBuffer&) -> Result<> {
                return fallback ? Result<>{} : drawSliderDebug(context, input, output);
            }});
        }
        result = context.executeStages(stages);
        if (!result || fallback) { return result; }
        hasHistory_ = true;
        lastFrame_ = context.frameIndex();
        lastRevision_ = revision;
        lastScene_ = scene;
        lastProperties_ = std::move(historyProperties);
        return {};
    }

private:
    Result<> initializeSliderDebug(std::string& log)
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

    Result<> drawSliderDebug(RenderGraphExecutionContext& context, TextureHandle input, TextureHandle output)
    {
        const auto& properties = context.properties();
        const float split = properties.value("splitPosition", 0.5f);
        const SliderDebugPush push{
            std::isfinite(split) ? std::clamp(split, 0.0f, 1.0f) : 0.5f,
            properties.value("orientation", "vertical") == "horizontal" ? 1u : 0u,
            properties.value("swapSides", false) ? 1u : 0u,
        };
        auto& command = context.commandBuffer();
        auto* source = input.view();
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureViews = &source, .textureViewCount = 1},
            {.binding = 2, .textureView = output.view()},
        };
        return sliderProgram_.dispatch({.commandBuffer = &command,
            .bindings = bindings, .bindingCount = 2, .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8, .groupCountY = (context.height() + 7) / 8});
    }

    static void copyColor(CommandBuffer& command, TextureHandle input, TextureHandle output)
    {
        command.copyTexture({.source = input.texture(), .destination = output.texture(),
            .width = output.desc().width, .height = output.desc().height, .depth = 1});
    }

    static Result<> copyColorAfterSdkFailure(CommandBuffer& command, TextureHandle input, TextureHandle output)
    {
        using namespace detail;
        const SyncScope externalScope{PipelineStageBits::AllCommands,
            AccessBits::MemoryRead | AccessBits::MemoryWrite};
        // An SDK failure can follow partial GPU recording. Preserve the existing
        // NGX General-layout contract and conservatively synchronize its work.
        if (input.texture()->retainAllocation() == output.texture()->retainAllocation()) {
            return makeError(Error::InvalidArgument);
        }
        const GraphAccessResource resources[] = {
            {RenderGraphResourceType::Texture2D, ResourceState::General, externalScope},
            {RenderGraphResourceType::Texture2D, ResourceState::General, externalScope},
        };
        const GraphAccessBinding bindings[] = {
            {.texture = input.texture(), .mipCount = input.desc().mipCount, .layerCount = input.desc().layerCount},
            {.texture = output.texture(), .mipCount = output.desc().mipCount, .layerCount = output.desc().layerCount},
        };
        const GraphAccessPass accesses[] = {
            {.uses = {
                {0, ResourceState::TransferSource, {PipelineStageBits::Transfer, AccessBits::TransferRead}, false},
                {1, ResourceState::TransferDestination, {PipelineStageBits::Transfer, AccessBits::TransferWrite}, true},
            }},
            {.uses = {
                {0, ResourceState::General, externalScope, false},
                {1, ResourceState::General, externalScope, false},
            }},
        };
        auto plan = buildGraphAccessPlan(resources, accesses);
        if (!plan) { return makeError(plan.error()); }
        auto result = recordGraphAccessBarriers(command, plan->passes.front(), bindings);
        if (!result) { return result; }
        copyColor(command, input, output);
        return recordGraphAccessBarriers(command, plan->passes.back(), bindings);
    }

    static Result<> executeCopy(RenderGraphExecutionContext& context, TextureHandle input, TextureHandle output)
    {
        const std::array uses{RenderGraphStageUse{"inputColor", RenderGraphResourceAccess::TextureTransferRead},
            RenderGraphStageUse{"color", RenderGraphResourceAccess::TextureTransferWrite}};
        const std::array stages{RenderGraphStage{"DLSS NR pass through", uses,
            [&](CommandBuffer& command) -> Result<> { copyColor(command, input, output); return {}; }}};
        return context.executeStages(stages);
    }

    Device* device_ = nullptr;
    ComputeProgram sliderProgram_;
    std::unique_ptr<vulkan::DlssNrContext> runtime_;
    bool hasHistory_ = false;
    bool failed_ = false;
    bool hdrBypass_ = false;
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
