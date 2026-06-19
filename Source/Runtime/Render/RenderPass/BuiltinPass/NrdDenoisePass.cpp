#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNrdWrapper.h"

namespace metallic::render::builtin_pass {
namespace {

class NrdDenoisePass final : public ComputePass {
public:
    ~NrdDenoisePass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("noisyDiffuse", "Noisy diffuse radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("noisySpecular", "Noisy specular radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("normalRoughness", "NRD packed normal and roughness")
            .storageReadWrite()
            .format = vulkan::nrdNormalRoughnessFormat();
        reflection.addTextureInput("motionVectors", "NRD motion vectors")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("viewZ", "NRD linear view depth")
            .storageReadWrite()
            .format = Format::R16Sfloat;
        reflection.addTextureInput("baseColorMetalness", "Base color and metalness")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        reflection.addTextureOutput("denoisedDiffuse", "NRD denoised diffuse radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("denoisedSpecular", "NRD denoised specular radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("validation", "NRD validation/debug output")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
#if !METALLIC_HAS_NRD
        log = "NrdDenoisePass requires the NRD SDK target";
        return makeError(Error::Unsupported);
#else
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "NrdDenoisePass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().pushDescriptor) {
            log = "NrdDenoisePass requires VK_KHR_push_descriptor";
            return makeError(Error::Unsupported);
        }

        device_ = context.device;
        queue_ = context.graphicsQueue;
        return {};
#endif
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
#if !METALLIC_HAS_NRD
        (void)context;
        return makeError(Error::Unsupported);
#else
        TextureHandle noisyDiffuse = context.inputTexture("noisyDiffuse");
        TextureHandle noisySpecular = context.inputTexture("noisySpecular");
        TextureHandle normalRoughness = context.inputTexture("normalRoughness");
        TextureHandle motionVectors = context.inputTexture("motionVectors");
        TextureHandle viewZ = context.inputTexture("viewZ");
        TextureHandle baseColorMetalness = context.inputTexture("baseColorMetalness");
        TextureHandle denoisedDiffuse = context.outputTexture("denoisedDiffuse");
        TextureHandle denoisedSpecular = context.outputTexture("denoisedSpecular");
        TextureHandle validation = context.outputTexture("validation");

        if (!validTexture(noisyDiffuse) ||
            !validTexture(noisySpecular) ||
            !validTexture(normalRoughness) ||
            !validTexture(motionVectors) ||
            !validTexture(viewZ) ||
            !validTexture(baseColorMetalness) ||
            !validTexture(denoisedDiffuse) ||
            !validTexture(denoisedSpecular) ||
            !validTexture(validation) ||
            device_ == nullptr ||
            queue_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const uint32_t denoiserMode = denoiserModeFromProperties(context.properties());
        const uint32_t resetSerial = uintProperty(
            context.properties(),
            "resetSerial",
            0,
            0,
            std::numeric_limits<uint32_t>::max());
        if (lastDenoiserMode_ != denoiserMode || lastResetSerial_ != resetSerial) {
            frameIndex_ = 0;
            lastDenoiserMode_ = denoiserMode;
            lastResetSerial_ = resetSerial;
        }

        Result result = ensureNrd(
            context,
            noisyDiffuse,
            noisySpecular,
            normalRoughness,
            motionVectors,
            viewZ,
            baseColorMetalness,
            denoisedDiffuse,
            denoisedSpecular,
            validation);
        if (!result) {
            return result;
        }

        result = runNrd(context, denoiserMode, noisyDiffuse, noisySpecular, denoisedDiffuse, denoisedSpecular);
        if (result) {
            ++frameIndex_;
        }
        return result;
#endif
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
        uint32_t value = fallback;
        if (iter->is_number_unsigned()) {
            value = iter->get<uint32_t>();
        } else if (iter->is_number_integer()) {
            const int64_t signedValue = iter->get<int64_t>();
            value = signedValue > 0 ? static_cast<uint32_t>(signedValue) : minimum;
        } else {
            value = static_cast<uint32_t>(std::max(iter->get<float>(), static_cast<float>(minimum)));
        }
        return std::clamp(value, minimum, maximum);
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
        if (!std::isfinite(value)) {
            return fallback;
        }
        return std::clamp(value, minimum, maximum);
    }

    static uint32_t denoiserModeFromProperties(const RenderGraphProperties& properties)
    {
        if (properties.is_object()) {
            auto iter = properties.find("denoiser");
            if (iter != properties.end() && iter->is_string()) {
                const std::string value = iter->get<std::string>();
                if (value == "RELAX" || value == "Relax" || value == "relax") {
                    return kNrdDenoiserModeRelax;
                }
                if (value == "REFERENCE" || value == "Reference" || value == "reference") {
                    return kNrdDenoiserModeReference;
                }
            }
        }
        return kNrdDenoiserModeReblur;
    }

#if METALLIC_HAS_NRD
    static void setIdentity(float matrix[16])
    {
        for (uint32_t index = 0; index < 16; ++index) {
            matrix[index] = 0.0f;
        }
        matrix[0] = 1.0f;
        matrix[5] = 1.0f;
        matrix[10] = 1.0f;
        matrix[15] = 1.0f;
    }

    static vulkan::NrdDenoiserMode wrapperMode(uint32_t denoiserMode)
    {
        if (denoiserMode == kNrdDenoiserModeRelax) {
            return vulkan::NrdDenoiserMode::Relax;
        }
        if (denoiserMode == kNrdDenoiserModeReference) {
            return vulkan::NrdDenoiserMode::Reference;
        }
        return vulkan::NrdDenoiserMode::Reblur;
    }

    Result ensureNrd(
        RenderGraphExecutionContext& context,
        TextureHandle noisyDiffuse,
        TextureHandle noisySpecular,
        TextureHandle normalRoughness,
        TextureHandle motionVectors,
        TextureHandle viewZ,
        TextureHandle baseColorMetalness,
        TextureHandle denoisedDiffuse,
        TextureHandle denoisedSpecular,
        TextureHandle validation)
    {
        if (context.width() > std::numeric_limits<uint16_t>::max() ||
            context.height() > std::numeric_limits<uint16_t>::max()) {
            return makeError(Error::InvalidArgument);
        }

        vulkan::NrdUserTexturePool pool{};
        auto put = [&pool](nrd::ResourceType resource, TextureHandle texture) {
            pool[static_cast<size_t>(resource)] = vulkan::NrdTextureRef{
                .texture = texture.texture(),
                .view = texture.view(),
            };
        };
        put(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, noisyDiffuse);
        put(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, noisySpecular);
        put(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, denoisedDiffuse);
        put(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, denoisedSpecular);
        put(nrd::ResourceType::IN_NORMAL_ROUGHNESS, normalRoughness);
        put(nrd::ResourceType::IN_MV, motionVectors);
        put(nrd::ResourceType::IN_VIEWZ, viewZ);
        put(nrd::ResourceType::IN_BASECOLOR_METALNESS, baseColorMetalness);
        put(nrd::ResourceType::OUT_VALIDATION, validation);
        put(nrd::ResourceType::IN_SIGNAL, noisyDiffuse);
        put(nrd::ResourceType::OUT_SIGNAL, denoisedDiffuse);

        const uint16_t width = static_cast<uint16_t>(context.width());
        const uint16_t height = static_cast<uint16_t>(context.height());
        const bool sizeChanged = nrd_ == nullptr ||
            !nrd_->valid() ||
            nrd_->width() != width ||
            nrd_->height() != height;
        if (sizeChanged) {
            nrd_ = std::make_unique<vulkan::NrdDenoiser>();
            std::string log;
            Result result = nrd_->initialize(*device_, *queue_, width, height, pool, log);
            if (!result) {
                nrd_.reset();
                return result;
            }
            frameIndex_ = 0;
            return {};
        }

        nrd_->setUserPoolTexture(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, *noisyDiffuse.texture(), *noisyDiffuse.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, *noisySpecular.texture(), *noisySpecular.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, *denoisedDiffuse.texture(), *denoisedDiffuse.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, *denoisedSpecular.texture(), *denoisedSpecular.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_NORMAL_ROUGHNESS, *normalRoughness.texture(), *normalRoughness.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_MV, *motionVectors.texture(), *motionVectors.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_VIEWZ, *viewZ.texture(), *viewZ.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_BASECOLOR_METALNESS, *baseColorMetalness.texture(), *baseColorMetalness.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::OUT_VALIDATION, *validation.texture(), *validation.view());
        return {};
    }

    Result runNrd(
        RenderGraphExecutionContext& context,
        uint32_t denoiserMode,
        TextureHandle noisyDiffuse,
        TextureHandle noisySpecular,
        TextureHandle denoisedDiffuse,
        TextureHandle denoisedSpecular)
    {
        if (nrd_ == nullptr || !nrd_->valid()) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphProperties& properties = context.properties();
        nrd::CommonSettings commonSettings;
        setIdentity(commonSettings.viewToClipMatrix);
        setIdentity(commonSettings.viewToClipMatrixPrev);
        setIdentity(commonSettings.worldToViewMatrix);
        setIdentity(commonSettings.worldToViewMatrixPrev);
        commonSettings.motionVectorScale[0] = floatProperty(properties, "motionVectorScaleX", 1.0f, -65504.0f, 65504.0f);
        commonSettings.motionVectorScale[1] = floatProperty(properties, "motionVectorScaleY", 1.0f, -65504.0f, 65504.0f);
        commonSettings.motionVectorScale[2] = 0.0f;
        commonSettings.resourceSize[0] = static_cast<uint16_t>(context.width());
        commonSettings.resourceSize[1] = static_cast<uint16_t>(context.height());
        commonSettings.resourceSizePrev[0] = static_cast<uint16_t>(context.width());
        commonSettings.resourceSizePrev[1] = static_cast<uint16_t>(context.height());
        commonSettings.rectSize[0] = static_cast<uint16_t>(context.width());
        commonSettings.rectSize[1] = static_cast<uint16_t>(context.height());
        commonSettings.rectSizePrev[0] = static_cast<uint16_t>(context.width());
        commonSettings.rectSizePrev[1] = static_cast<uint16_t>(context.height());
        commonSettings.frameIndex = frameIndex_;
        commonSettings.timeDeltaBetweenFrames = floatProperty(properties, "timeDeltaSeconds", 1.0f / 60.0f, 0.0f, 1.0f);
        commonSettings.denoisingRange = floatProperty(properties, "denoisingRange", 10000.0f, 0.0f, 1000000.0f);
        commonSettings.accumulationMode = frameIndex_ == 0
            ? nrd::AccumulationMode::CLEAR_AND_RESTART
            : nrd::AccumulationMode::CONTINUE;
        commonSettings.isMotionVectorInWorldSpace = boolProperty(&properties, "motionVectorInWorldSpace", false);
        commonSettings.isBaseColorMetalnessAvailable = true;
        commonSettings.enableValidation = boolProperty(&properties, "enableValidation", true);

        Result result = nrd_->setCommonSettings(commonSettings);
        if (!result) {
            return result;
        }

        if (denoiserMode == kNrdDenoiserModeReference) {
            nrd_->setUserPoolTexture(nrd::ResourceType::IN_SIGNAL, *noisyDiffuse.texture(), *noisyDiffuse.view());
            nrd_->setUserPoolTexture(nrd::ResourceType::OUT_SIGNAL, *denoisedDiffuse.texture(), *denoisedDiffuse.view());
            nrd::Identifier referenceDiffuse = static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE);
            result = nrd_->denoiseIdentifiers(&referenceDiffuse, 1, context.commandBuffer());
            if (!result) {
                return result;
            }

            nrd_->setUserPoolTexture(nrd::ResourceType::IN_SIGNAL, *noisySpecular.texture(), *noisySpecular.view());
            nrd_->setUserPoolTexture(nrd::ResourceType::OUT_SIGNAL, *denoisedSpecular.texture(), *denoisedSpecular.view());
            nrd::Identifier referenceSpecular = static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE) + 1;
            return nrd_->denoiseIdentifiers(&referenceSpecular, 1, context.commandBuffer());
        }

        if (denoiserMode == kNrdDenoiserModeRelax) {
            nrd::RelaxSettings relaxSettings;
            result = nrd_->setRelaxSettings(relaxSettings);
            if (!result) {
                return result;
            }
        } else {
            nrd::ReblurSettings reblurSettings;
            result = nrd_->setReblurSettings(reblurSettings);
            if (!result) {
                return result;
            }
        }
        return nrd_->denoise(wrapperMode(denoiserMode), context.commandBuffer());
    }
#endif

    Device* device_ = nullptr;
    Queue* queue_ = nullptr;
    uint32_t frameIndex_ = 0;
    uint32_t lastDenoiserMode_ = std::numeric_limits<uint32_t>::max();
    uint32_t lastResetSerial_ = 0;
#if METALLIC_HAS_NRD
    std::unique_ptr<vulkan::NrdDenoiser> nrd_;
#endif
};

} // namespace

std::unique_ptr<RenderGraphPass> createNrdDenoisePass()
{
    return std::make_unique<NrdDenoisePass>();
}

} // namespace metallic::render::builtin_pass
