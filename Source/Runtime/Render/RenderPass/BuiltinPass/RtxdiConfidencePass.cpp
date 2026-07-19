#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

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
};

struct ConfidenceGradientTexture {
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    ResourceState state = ResourceState::Undefined;
};

class RtxdiConfidencePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("noisyDiffuse", "RTXDI diffuse radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("noisySpecular", "RTXDI specular radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("baseColorMetalness", "Base color and metalness")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        reflection.addTextureInput("motionVectors", "Previous-minus-current UV motion")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("diffuseConfidence", "NRD diffuse history confidence")
            .storageReadWrite()
            .format = Format::R8Unorm;
        reflection.addTextureOutput("specularConfidence", "NRD specular history confidence")
            .storageReadWrite()
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

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            log = "RtxdiConfidencePass requires a device";
            return makeError(Error::InvalidArgument);
        }

        const uint32_t gradientWidth = (context.width + kGradientFactor - 1u) / kGradientFactor;
        const uint32_t gradientHeight = (context.height + kGradientFactor - 1u) / kGradientFactor;
        Result result = ensureGradientTextures(
            *context.device,
            std::max(gradientWidth, 1u),
            std::max(gradientHeight, 1u),
            log);
        if (!result || program_.valid()) {
            return result;
        }

        ShaderCompileResult compileResult;
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kRtxdiConfidenceShaderModuleName,
                .entryPointName = kRtxdiConfidenceEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
            },
            compileResult);
        if (!result) {
            log = resultMessage(
                "compileSlangShaderToSpirv(RtxdiConfidence.rtxdiConfidenceMain)",
                result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            return result;
        }

        const SceneRayQueryBindingDesc bindings[] = {
            {.binding = 0, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 1, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 2, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 3, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 4, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 5, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 6, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 7, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 8, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 9, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 10, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 11, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 12, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 13, .kind = SceneRayQueryBindingKind::StorageImage},
        };
        std::string programLog;
        result = program_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(RtxdiConfidencePush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "RtxdiConfidencePass",
                .descriptorSetCount = kMaximumFilterPasses + 2u,
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

    Result execute(RenderGraphExecutionContext& context) override
    {
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
        Result result = prepareHistoryTexture(
            context,
            "luminance",
            Format::Rg16Sfloat,
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

        transitionGradientTextures(context.commandBuffer());

        RtxdiConfidencePush push;
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

        const SceneRayQueryDispatchBinding bindings[] = {
            {.binding = 0, .textureView = noisyDiffuse.view()},
            {.binding = 1, .textureView = noisySpecular.view()},
            {.binding = 2, .textureView = baseColorMetalness.view()},
            {.binding = 3, .textureView = motionVectors.view()},
            {.binding = 4, .textureView = luminanceHistory.previous},
            {.binding = 5, .textureView = luminanceHistory.current},
            {.binding = 6, .textureView = gradientA_.view.get()},
            {.binding = 7, .textureView = gradientB_.view.get()},
            {.binding = 8, .textureView = diffuseConfidenceHistory.previous},
            {.binding = 9, .textureView = specularConfidenceHistory.previous},
            {.binding = 10, .textureView = diffuseConfidence.view()},
            {.binding = 11, .textureView = specularConfidence.view()},
            {.binding = 12, .textureView = diffuseConfidenceHistory.current},
            {.binding = 13, .textureView = specularConfidenceHistory.current},
        };
        auto dispatch = [&](uint32_t mode, uint32_t descriptorSetIndex, uint32_t width, uint32_t height) {
            push.mode = mode;
            return program_.dispatch(SceneRayQueryDispatchDesc{
                .commandBuffer = &context.commandBuffer(),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .pushData = &push,
                .pushDataSize = sizeof(push),
                .groupCountX = (width + 7u) / 8u,
                .groupCountY = (height + 7u) / 8u,
                .groupCountZ = 1,
                .descriptorSetIndex = descriptorSetIndex,
            });
        };

        result = dispatch(kComputeGradient, 0u, gradientWidth_, gradientHeight_);
        if (!result) {
            return result;
        }
        synchronizeGradientTexture(context.commandBuffer(), gradientA_);

        const uint32_t filterPassCount = uintProperty(
            context.properties(),
            "gradientFilterPasses",
            4,
            0,
            kMaximumFilterPasses);
        for (uint32_t passIndex = 0; passIndex < filterPassCount; ++passIndex) {
            const bool sourceA = (passIndex & 1u) == 0u;
            push.filterStep = 1u << passIndex;
            result = dispatch(
                sourceA ? kFilterAToB : kFilterBToA,
                passIndex + 1u,
                gradientWidth_,
                gradientHeight_);
            if (!result) {
                return result;
            }
            synchronizeGradientTexture(
                context.commandBuffer(),
                sourceA ? gradientB_ : gradientA_);
        }

        const bool finalGradientIsA = (filterPassCount & 1u) == 0u;
        result = dispatch(
            finalGradientIsA ? kResolveA : kResolveB,
            kMaximumFilterPasses + 1u,
            context.width(),
            context.height());
        if (!result) {
            return result;
        }

        HistoryResourceManager& history = *context.historyResources();
        history.markWritten(historyNameForContext(context, "luminance"));
        history.markWritten(historyNameForContext(context, "diffuseConfidence"));
        history.markWritten(historyNameForContext(context, "specularConfidence"));
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

    Result ensureGradientTextures(
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
        Result result = createGradientTexture(device, width, height, "A", nextA, log);
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

    static Result createGradientTexture(
        Device& device,
        uint32_t width,
        uint32_t height,
        std::string_view label,
        ConfidenceGradientTexture& outTexture,
        std::string& log)
    {
        Result result = device.createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::Storage,
                .format = Format::Rgba16Sfloat,
                .width = width,
                .height = height,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            },
            outTexture.texture);
        if (!result || outTexture.texture == nullptr) {
            log = resultMessage(
                std::string("createTexture(RtxdiConfidence gradient ") + std::string(label) + ')',
                result);
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createTextureView(
            *outTexture.texture,
            TextureViewDesc{.format = Format::Rgba16Sfloat},
            outTexture.view);
        if (!result || outTexture.view == nullptr) {
            log = resultMessage(
                std::string("createTextureView(RtxdiConfidence gradient ") + std::string(label) + ')',
                result);
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    void transitionGradientTextures(CommandBuffer& commandBuffer)
    {
        TextureBarrierDesc barriers[] = {
            TextureBarrierDesc{
                .texture = gradientA_.texture.get(),
                .before = gradientA_.state,
                .after = ResourceState::General,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            TextureBarrierDesc{
                .texture = gradientB_.texture.get(),
                .before = gradientB_.state,
                .after = ResourceState::General,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = barriers,
            .textureCount = static_cast<uint32_t>(std::size(barriers)),
        });
        gradientA_.state = ResourceState::General;
        gradientB_.state = ResourceState::General;
    }

    static void synchronizeGradientTexture(
        CommandBuffer& commandBuffer,
        ConfidenceGradientTexture& texture)
    {
        TextureBarrierDesc barrier{
            .texture = texture.texture.get(),
            .before = ResourceState::General,
            .after = ResourceState::General,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer.barrier(BarrierDesc{.textures = &barrier, .textureCount = 1});
    }

    static Result prepareHistoryTexture(
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
        Result result = history->ensureTexture(name, desc, TextureViewDesc{.format = format});
        if (!result) {
            return result;
        }

        const HistoryTextureRef current = history->texture(name, HistorySlot::Current);
        const HistoryTextureRef previous = history->texture(name, HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr ||
            previous.texture == nullptr || previous.view == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        result = history->transitionTexture(
            context.commandBuffer(),
            name,
            HistorySlot::Current,
            ResourceState::General);
        if (!result) {
            return result;
        }
        result = history->transitionTexture(
            context.commandBuffer(),
            name,
            HistorySlot::Previous,
            ResourceState::General);
        if (!result) {
            return result;
        }
        outViews.current = current.view;
        outViews.previous = previous.view;
        outViews.previousValid = previous.valid;
        return {};
    }

    static std::string historyNameForContext(
        const RenderGraphExecutionContext& context,
        std::string_view suffix)
    {
        std::string name("RtxdiConfidencePass.");
        name += context.passName();
        name += '.';
        name += suffix;
        return name;
    }

    SceneRayQueryProgram program_;
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
    return std::make_unique<RtxdiConfidencePass>();
}

} // namespace metallic::render::builtin_pass
