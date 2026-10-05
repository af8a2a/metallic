#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RTXDIPostProcessParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Core/RenderFrameContext.h"

namespace metallic::render::builtin_pass {
namespace {

constexpr uint32_t kComputeGradient = 0;
constexpr uint32_t kFilterAToB = 1;
constexpr uint32_t kFilterBToA = 2;
constexpr uint32_t kResolveA = 3;
constexpr uint32_t kResolveB = 4;
constexpr uint32_t kGradientFactor = 3;
constexpr uint32_t kMaximumFilterPasses = 6;

struct ConfidenceHistoryViews {
    TextureView* current = nullptr;
    TextureView* previous = nullptr;
    bool previousValid = false;
    HistoryTextureRef currentResource;
    HistoryTextureRef previousResource;
};

struct ConfidenceGradientTexture {
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    std::shared_ptr<ResourceState> state = std::make_shared<ResourceState>(ResourceState::Undefined);
};

class RTXDIConfidencePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("noisyDiffuse", "RTXDI diffuse radiance and hit distance")
            .storageRead()
            .format = Format::RGBA16Sfloat;
        reflection.addTextureInput("noisySpecular", "RTXDI specular radiance and hit distance")
            .storageRead()
            .format = Format::RGBA16Sfloat;
        reflection.addTextureInput("baseColorMetalness", "Base color and metalness")
            .storageRead()
            .format = Format::RGBA8Unorm;
        reflection.addTextureInput("motionVectors", "Previous-minus-current UV motion")
            .storageRead()
            .format = Format::RGBA16Sfloat;
        reflection.addTextureOutput("diffuseConfidence", "NRD diffuse history confidence")
            .storageWrite()
            // Resolve writes every texel; temporal data lives in separate imports.
            .transient(RenderGraphInitialization::FullOverwrite)
            .format = Format::R8Unorm;
        reflection.addTextureOutput("specularConfidence", "NRD specular history confidence")
            .storageWrite()
            .transient(RenderGraphInitialization::FullOverwrite)
            .format = Format::R8Unorm;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeIntSetting(
                "gradientFilterPasses",
                "Gradient A-Trous Passes",
                4,
                0,
                static_cast<int32_t>(kMaximumFilterPasses),
                true),
            runtimeFloatSetting(
                "gradientLogDarknessBias",
                "Darkness Bias (EV)",
                -12.0f,
                -16.0f,
                -4.0f,
                true),
            runtimeFloatSetting(
                "gradientSensitivity",
                "Gradient Sensitivity",
                8.0f,
                1.0f,
                20.0f,
                true),
            runtimeFloatSetting(
                "confidenceHistoryLength",
                "Confidence History",
                0.75f,
                0.0f,
                3.0f,
                true),
            runtimeActionCounterSetting("resetSerial", "Reset", true),
        };
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            log = "RTXDIConfidencePass requires a device";
            return makeError(Error::InvalidArgument);
        }

        const uint32_t gradientWidth = (context.width + kGradientFactor - 1u) / kGradientFactor;
        const uint32_t gradientHeight = (context.height + kGradientFactor - 1u) / kGradientFactor;
        Result<> result = ensureGradientTextures(
            *context.device,
            std::max(gradientWidth, 1u),
            std::max(gradientHeight, 1u),
            log);
        if (!result || program_.valid()) {
            return result;
        }

        device_ = context.device;
        ShaderCompileResult compileResult;
        result = ShaderRegistry::instance().getShader(SlangShaderDesc{
                .moduleName = kRTXDIConfidenceShaderModuleName,
                .entryPointName = kRTXDIConfidenceEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
            }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result) {
            log = resultMessage(
                "compileSlangShaderToSpirv(RTXDIConfidence.rtxdiConfidenceMain)",
                result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            return result;
        }

        std::string programLog;
        result = program_.initialize(
            *context.device,
            ComputeKernelDesc{
                .spirv = compileResult.spirv,
                .parameters = parameterAbi<RTXDIConfidenceParams>(kRTXDIConfidenceABI, ParameterTransport::InlinePush),
                .debugName = "RTXDIConfidencePass",
            },
            programLog);
        if (!programLog.empty()) {
            log += programLog;
        }
        if (!result) {
            program_.clear();
        }
        return result;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        if (device_ == nullptr || !program_.valid()) {
            return makeError(Error::InvalidArgument);
        }
        // RenderGraph can reuse the compiled pass when only the viewport changes.
        // Refresh private resources before encoding handles or resolving history.
        std::string gradientLog;
        const auto gradientResult = ensureGradientTextures(*device_,
            std::max((context.width() + kGradientFactor - 1u) / kGradientFactor, 1u),
            std::max((context.height() + kGradientFactor - 1u) / kGradientFactor, 1u), gradientLog);
        if (!gradientResult) { return gradientResult; }
        TextureHandle noisyDiffuse = context.inputTexture("noisyDiffuse");
        TextureHandle noisySpecular = context.inputTexture("noisySpecular");
        TextureHandle baseColorMetalness = context.inputTexture("baseColorMetalness");
        TextureHandle motionVectors = context.inputTexture("motionVectors");
        TextureHandle diffuseConfidence = context.outputTexture("diffuseConfidence");
        TextureHandle specularConfidence = context.outputTexture("specularConfidence");
        if (!validTexture(noisyDiffuse) ||
            !validTexture(noisySpecular) ||
            !validTexture(baseColorMetalness) ||
            !validTexture(motionVectors) ||
            !validTexture(diffuseConfidence) ||
            !validTexture(specularConfidence) ||
            !program_.valid() ||
            gradientA_.texture == nullptr ||
            gradientA_.view == nullptr ||
            gradientB_.texture == nullptr ||
            gradientB_.view == nullptr ||
            context.historyResources() == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const uint32_t resetSerial = uintProperty(
            context.properties(),
            "resetSerial",
            0,
            0,
            std::numeric_limits<uint32_t>::max());
        if (resetSerial != lastResetSerial_) {
            resetHistory_ = true;
            lastResetSerial_ = resetSerial;
        }

        ConfidenceHistoryViews luminanceHistory;
        ConfidenceHistoryViews diffuseConfidenceHistory;
        ConfidenceHistoryViews specularConfidenceHistory;
        Result<> result = prepareHistoryTexture(
            context,
            "luminance",
            Format::RG16Sfloat,
            luminanceHistory);
        if (!result) {
            return result;
        }
        result = prepareHistoryTexture(
            context,
            "diffuseConfidence",
            Format::R8Unorm,
            diffuseConfidenceHistory);
        if (!result) {
            return result;
        }
        result = prepareHistoryTexture(
            context,
            "specularConfidence",
            Format::R8Unorm,
            specularConfidenceHistory);
        if (!result) {
            return result;
        }

        RTXDIConfidencePush push{};
        push.width = context.width();
        push.height = context.height();
        push.gradientWidth = gradientWidth_;
        push.gradientHeight = gradientHeight_;
        push.hasHistory = !resetHistory_ &&
            luminanceHistory.previousValid &&
            diffuseConfidenceHistory.previousValid &&
            specularConfidenceHistory.previousValid
            ? 1u
            : 0u;
        const float logDarknessBias = floatProperty(
            context.properties(),
            "gradientLogDarknessBias",
            -12.0f,
            -16.0f,
            -4.0f);
        push.darknessBias = std::exp2(logDarknessBias);
        push.sensitivity = floatProperty(
            context.properties(),
            "gradientSensitivity",
            8.0f,
            1.0f,
            20.0f);
        const float historyLength = floatProperty(
            context.properties(),
            "confidenceHistoryLength",
            0.75f,
            0.0f,
            3.0f);
        push.blendFactor = 1.0f / (historyLength + 1.0f);

        auto registry = metallic::render::ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, metallic::render::RenderFrameContext::from(context.commandBuffer()));
        RTXDIConfidenceParams params{
            .noisyDiffuse = writer.storageImage(noisyDiffuse.view()),
            .noisySpecular = writer.storageImage(noisySpecular.view()),
            .baseColorMetalness = writer.storageImage(baseColorMetalness.view()),
            .motionVectors = writer.storageImage(motionVectors.view()),
            .previousLuminance = writer.storageImage(luminanceHistory.previous),
            .currentLuminance = writer.storageImage(luminanceHistory.current),
            .gradientA = writer.storageImage(gradientA_.view.get()),
            .gradientB = writer.storageImage(gradientB_.view.get()),
            .previousDiffuseConfidence = writer.storageImage(diffuseConfidenceHistory.previous),
            .previousSpecularConfidence = writer.storageImage(specularConfidenceHistory.previous),
            .diffuseConfidence = writer.storageImage(diffuseConfidence.view()),
            .specularConfidence = writer.storageImage(specularConfidence.view()),
            .currentDiffuseConfidence = writer.storageImage(diffuseConfidenceHistory.current),
            .currentSpecularConfidence = writer.storageImage(specularConfidenceHistory.current),
            .settings = push,
        };
        auto dispatch = [&](CommandBuffer& commands, uint32_t mode,
                            uint32_t width, uint32_t height, uint32_t filterStep = 0u) -> Result<> {
            auto stagePush = push;
            stagePush.mode = mode;
            stagePush.filterStep = filterStep;
            params.settings = stagePush;
            auto encoded = writer.encode(params, kRTXDIConfidenceABI, ParameterTransport::InlinePush);
            if (!encoded) { return makeError(encoded.error()); }
            return program_.dispatch(commands, *encoded, (width + 7u) / 8u, (height + 7u) / 8u);
        };

        const uint32_t filterPassCount = uintProperty(
            context.properties(),
            "gradientFilterPasses",
            4,
            0,
            kMaximumFilterPasses);
        const bool finalGradientIsA = (filterPassCount & 1u) == 0u;
        using Access = RenderGraphResourceAccess;
        // Every mode uses the same statically bound storage-image descriptors.
        // Establish their layouts before the first dispatch, even with zero
        // filter passes or invalid history; content validity is independent.
        const RenderGraphStageUse prepareUses[] = {
            {"luminanceCurrent", Access::TextureStorageReadWrite},
            {"luminancePrevious", Access::TextureStorageReadWrite},
            {"diffuseCurrent", Access::TextureStorageReadWrite},
            {"diffusePrevious", Access::TextureStorageReadWrite},
            {"specularCurrent", Access::TextureStorageReadWrite},
            {"specularPrevious", Access::TextureStorageReadWrite},
            {"gradientA", Access::TextureStorageReadWrite},
            {"gradientB", Access::TextureStorageReadWrite},
        };
        const RenderGraphStageUse gradientUses[] = {
            {"noisyDiffuse", Access::TextureStorageRead},
            {"noisySpecular", Access::TextureStorageRead},
            {"baseColorMetalness", Access::TextureStorageRead},
            {"motionVectors", Access::TextureStorageRead},
            {"luminancePrevious", Access::TextureStorageRead},
            {"luminanceCurrent", Access::TextureStorageWrite},
            {"gradientA", Access::TextureStorageWrite},
        };
        const RenderGraphStageUse filterAToB[] = {
            {"gradientA", Access::TextureStorageRead}, {"gradientB", Access::TextureStorageWrite},
        };
        const RenderGraphStageUse filterBToA[] = {
            {"gradientB", Access::TextureStorageRead}, {"gradientA", Access::TextureStorageWrite},
        };
        const RenderGraphStageUse resolveUses[] = {
            {finalGradientIsA ? "gradientA" : "gradientB", Access::TextureStorageRead},
            {"motionVectors", Access::TextureStorageRead},
            {"diffusePrevious", Access::TextureStorageRead},
            {"specularPrevious", Access::TextureStorageRead},
            {"diffuseCurrent", Access::TextureStorageWrite},
            {"specularCurrent", Access::TextureStorageWrite},
            {"diffuseConfidence", Access::TextureStorageWrite},
            {"specularConfidence", Access::TextureStorageWrite},
        };
        const auto import = [](std::string_view name, const HistoryTextureRef& texture) {
            return RenderGraphTextureImport{name, texture.texture, texture.view,
                texture.state, ResourceState::General};
        };
        const RenderGraphTextureImport textures[] = {
            import("luminanceCurrent", luminanceHistory.currentResource),
            import("luminancePrevious", luminanceHistory.previousResource),
            import("diffuseCurrent", diffuseConfidenceHistory.currentResource),
            import("diffusePrevious", diffuseConfidenceHistory.previousResource),
            import("specularCurrent", specularConfidenceHistory.currentResource),
            import("specularPrevious", specularConfidenceHistory.previousResource),
            {"gradientA", gradientA_.texture.get(), gradientA_.view.get(),
                *gradientA_.state, ResourceState::General},
            {"gradientB", gradientB_.texture.get(), gradientB_.view.get(),
                *gradientB_.state, ResourceState::General},
        };
        std::vector<RenderGraphStage> stages;
        stages.reserve(filterPassCount + 3u);
        stages.push_back({"PrepareDescriptors", prepareUses, [](CommandBuffer&) -> Result<> { return {}; }});
        stages.push_back({"Gradient", gradientUses, [&](CommandBuffer& commands) {
            return dispatch(commands, kComputeGradient, gradientWidth_, gradientHeight_);
        }});
        for (uint32_t passIndex = 0; passIndex < filterPassCount; ++passIndex) {
            const bool sourceA = (passIndex & 1u) == 0u;
            stages.push_back({sourceA ? "FilterAToB" : "FilterBToA",
                sourceA ? std::span<const RenderGraphStageUse>(filterAToB)
                        : std::span<const RenderGraphStageUse>(filterBToA),
                [&, passIndex, sourceA](CommandBuffer& commands) {
                    return dispatch(commands, sourceA ? kFilterAToB : kFilterBToA,
                        gradientWidth_, gradientHeight_, 1u << passIndex);
                }});
        }
        stages.push_back({"Resolve", resolveUses, [&](CommandBuffer& commands) {
            return dispatch(commands, finalGradientIsA ? kResolveA : kResolveB,
                context.width(), context.height());
        }});
        result = context.executeStages(stages, {}, textures);
        if (!result) {
            return result;
        }

        HistoryResourceManager& history = *context.historyResources();
        for (const auto suffix : {"luminance", "diffuseConfidence", "specularConfidence"}) {
            const auto name = historyNameForContext(context, suffix);
            result = history.publishTextureState(context.commandBuffer(), name,
                HistorySlot::Current, ResourceState::General, true);
            if (!result) { return result; }
            result = history.publishTextureState(context.commandBuffer(), name,
                HistorySlot::Previous, ResourceState::General);
            if (!result) { return result; }
        }
        for (const auto& state : {gradientA_.state, gradientB_.state}) {
            const auto before = *state;
            result = context.commandBuffer().addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
                [] {}, [state, before] { *state = before; }));
            if (!result) { return result; }
            *state = ResourceState::General;
        }
        resetHistory_ = false;
        return {};
    }

private:
    static bool validTexture(TextureHandle texture)
    {
        return texture.valid() && texture.texture() != nullptr && texture.view() != nullptr;
    }

    static uint32_t uintProperty(
        const RenderGraphProperties& properties,
        const char* key,
        uint32_t fallback,
        uint32_t minimum,
        uint32_t maximum)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        const int64_t value = iter->is_number_integer()
            ? iter->get<int64_t>()
            : static_cast<int64_t>(iter->get<double>());
        return static_cast<uint32_t>(std::clamp<int64_t>(value, minimum, maximum));
    }

    static float floatProperty(
        const RenderGraphProperties& properties,
        const char* key,
        float fallback,
        float minimum,
        float maximum)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        const float value = iter->get<float>();
        return std::isfinite(value) ? std::clamp(value, minimum, maximum) : fallback;
    }

    Result<> ensureGradientTextures(
        Device& device,
        uint32_t width,
        uint32_t height,
        std::string& log)
    {
        if (gradientA_.texture != nullptr &&
            gradientB_.texture != nullptr &&
            gradientWidth_ == width &&
            gradientHeight_ == height) {
            return {};
        }

        ConfidenceGradientTexture nextA;
        ConfidenceGradientTexture nextB;
        Result<> result = createGradientTexture(device, width, height, "A", nextA, log);
        if (!result) {
            return result;
        }
        result = createGradientTexture(device, width, height, "B", nextB, log);
        if (!result) {
            return result;
        }
        gradientA_ = std::move(nextA);
        gradientB_ = std::move(nextB);
        gradientWidth_ = width;
        gradientHeight_ = height;
        resetHistory_ = true;
        return {};
    }

    static Result<> createGradientTexture(
        Device& device,
        uint32_t width,
        uint32_t height,
        std::string_view label,
        ConfidenceGradientTexture& outTexture,
        std::string& log)
    {
        Result<> result = device.createTexture(TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::Storage,
                .format = Format::RGBA16Sfloat,
                .width = width,
                .height = height,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            }).transform([&](auto rhiValue) { outTexture.texture = std::move(rhiValue); });
        if (!result || outTexture.texture == nullptr) {
            log = resultMessage(
                std::string("createTexture(RTXDIConfidence gradient ") + std::string(label) + ')',
                result);
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createTextureView(*outTexture.texture,
            TextureViewDesc{.format = Format::RGBA16Sfloat}).transform([&](auto rhiValue) { outTexture.view = std::move(rhiValue); });
        if (!result || outTexture.view == nullptr) {
            log = resultMessage(
                std::string("createTextureView(RTXDIConfidence gradient ") + std::string(label) + ')',
                result);
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    static Result<> prepareHistoryTexture(
        RenderGraphExecutionContext& context,
        std::string_view suffix,
        Format format,
        ConfidenceHistoryViews& outViews)
    {
        HistoryResourceManager* history = context.historyResources();
        if (history == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const TextureDesc desc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Storage | TextureUsageBits::TransferSource,
            .format = format,
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        const std::string name = historyNameForContext(context, suffix);
        Result<> result = history->ensureTexture(name, desc, TextureViewDesc{.format = format});
        if (!result) {
            return result;
        }

        const HistoryTextureRef current = history->texture(name, HistorySlot::Current);
        const HistoryTextureRef previous = history->texture(name, HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr ||
            previous.texture == nullptr || previous.view == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        outViews.current = current.view;
        outViews.previous = previous.view;
        outViews.previousValid = previous.valid;
        outViews.currentResource = current;
        outViews.previousResource = previous;
        return {};
    }

    static std::string historyNameForContext(
        const RenderGraphExecutionContext& context,
        std::string_view suffix)
    {
        std::string name("RTXDIConfidencePass.");
        name += context.passName();
        name += '.';
        name += suffix;
        return name;
    }

    Device* device_ = nullptr;
    ComputeKernel program_;
    ConfidenceGradientTexture gradientA_;
    ConfidenceGradientTexture gradientB_;
    uint32_t gradientWidth_ = 0;
    uint32_t gradientHeight_ = 0;
    uint32_t lastResetSerial_ = 0;
    bool resetHistory_ = true;
};

} // namespace

std::unique_ptr<RenderGraphPass> createRtxdiConfidencePass()
{
    return std::make_unique<RTXDIConfidencePass>();
}

} // namespace metallic::render::builtin_pass
