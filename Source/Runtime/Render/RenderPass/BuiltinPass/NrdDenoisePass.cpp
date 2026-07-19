#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNrdWrapper.h"

namespace metallic::render::builtin_pass {
namespace {

struct NrdCameraSnapshot {
    float3 eye{0.0f, 0.25f, 3.0f};
    float3 center{0.0f, 0.15f, 0.0f};
    float3 up{0.0f, 1.0f, 0.0f};
    float fovRadians = 0.87266463f;
    float aspectRatio = 1.0f;
    float zNear = 0.001f;
    float zFar = 10000.0f;
    float orthoHeight = 1.0f;
    bool orthographic = false;
};

class NrdDenoisePass final : public ComputePass {
public:
    ~NrdDenoisePass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        RenderGraphField& noisyDiffuse = reflection.addTextureInput(
            "noisyDiffuse",
            "Noisy diffuse radiance and hit distance").storageReadWrite();
        noisyDiffuse.format = Format::Rgba16Sfloat;
        noisyDiffuse.usage = TextureUsageBits::Sampled;
        RenderGraphField& noisySpecular = reflection.addTextureInput(
            "noisySpecular",
            "Noisy specular radiance and hit distance").storageReadWrite();
        noisySpecular.format = Format::Rgba16Sfloat;
        noisySpecular.usage = TextureUsageBits::Sampled;
        RenderGraphField& normalRoughness = reflection.addTextureInput(
            "normalRoughness",
            "NRD packed normal and roughness").storageReadWrite();
        normalRoughness.format = vulkan::nrdNormalRoughnessFormat();
        normalRoughness.usage = TextureUsageBits::Sampled;
        RenderGraphField& motionVectors = reflection.addTextureInput(
            "motionVectors",
            "NRD motion vectors").storageReadWrite();
        motionVectors.format = Format::Rgba16Sfloat;
        motionVectors.usage = TextureUsageBits::Sampled;
        RenderGraphField& viewZ = reflection.addTextureInput(
            "viewZ",
            "NRD linear view depth").storageReadWrite();
        viewZ.format = Format::R16Sfloat;
        viewZ.usage = TextureUsageBits::Sampled;
        RenderGraphField& baseColorMetalness = reflection.addTextureInput(
            "baseColorMetalness",
            "Base color and metalness").storageReadWrite();
        baseColorMetalness.format = Format::Rgba8Unorm;
        baseColorMetalness.usage = TextureUsageBits::Sampled;
        RenderGraphField& diffuseConfidence = reflection.addTextureInput(
            "diffuseConfidence",
            "Diffuse history confidence").storageReadWrite();
        diffuseConfidence.format = Format::R8Unorm;
        diffuseConfidence.usage = TextureUsageBits::Sampled;
        RenderGraphField& specularConfidence = reflection.addTextureInput(
            "specularConfidence",
            "Specular history confidence").storageReadWrite();
        specularConfidence.format = Format::R8Unorm;
        specularConfidence.usage = TextureUsageBits::Sampled;
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

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeEnumSetting(
                "denoiser",
                "Denoiser",
                "REBLUR",
                {{"REBLUR", "REBLUR"}, {"RELAX", "RELAX"}, {"REFERENCE", "REFERENCE"}},
                true),
            runtimeBoolSetting("enableValidation", "Validation", true),
            runtimeIntSetting("relaxHistoryLength", "RELAX History", 30, 0, 255, true),
            runtimeIntSetting("relaxFastHistoryLength", "RELAX Fast History", 6, 0, 255, true),
            runtimeIntSetting("relaxAtrousIterations", "RELAX A-Trous Iterations", 5, 2, 8, true),
            runtimeFloatSetting("relaxDiffusePrepassRadius", "RELAX Diffuse Prepass", 30.0f, 0.0f, 50.0f, true),
            runtimeFloatSetting("relaxSpecularPrepassRadius", "RELAX Specular Prepass", 50.0f, 0.0f, 50.0f, true),
            runtimeFloatSetting("relaxMinHitDistanceWeight", "RELAX Min Hit Weight", 0.1f, 0.001f, 0.2f, true),
            runtimeBoolSetting("relaxAntiFirefly", "RELAX Anti-Firefly", true, true),
            runtimeBoolSetting("relaxConfidenceInputs", "RELAX Confidence Inputs", true, true),
            runtimeActionCounterSetting("resetSerial", "Reset", true),
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 0.25f, 3.0f},
            std::array<float, 3>{0.0f, 0.15f, 0.0f},
            50.0f);
        return settings;
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
        TextureHandle diffuseConfidence = context.inputTexture("diffuseConfidence");
        TextureHandle specularConfidence = context.inputTexture("specularConfidence");
        TextureHandle denoisedDiffuse = context.outputTexture("denoisedDiffuse");
        TextureHandle denoisedSpecular = context.outputTexture("denoisedSpecular");
        TextureHandle validation = context.outputTexture("validation");

        if (!validTexture(noisyDiffuse) ||
            !validTexture(noisySpecular) ||
            !validTexture(normalRoughness) ||
            !validTexture(motionVectors) ||
            !validTexture(viewZ) ||
            !validTexture(baseColorMetalness) ||
            !validTexture(diffuseConfidence) ||
            !validTexture(specularConfidence) ||
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
            hasPreviousCamera_ = false;
            lastDenoiserMode_ = denoiserMode;
            lastResetSerial_ = resetSerial;
        }

        const NrdCameraSnapshot currentCamera = cameraFromProperties(
            context.width(),
            context.height(),
            context.properties());

        Result result = ensureNrd(
            context,
            noisyDiffuse,
            noisySpecular,
            normalRoughness,
            motionVectors,
            viewZ,
            baseColorMetalness,
            diffuseConfidence,
            specularConfidence,
            denoisedDiffuse,
            denoisedSpecular,
            validation);
        if (!result) {
            return result;
        }

        const NrdCameraSnapshot& previousCamera = hasPreviousCamera_
            ? previousCamera_
            : currentCamera;
        result = runNrd(
            context,
            denoiserMode,
            noisyDiffuse,
            noisySpecular,
            denoisedDiffuse,
            denoisedSpecular,
            currentCamera,
            previousCamera);
        if (result) {
            ++frameIndex_;
            previousCamera_ = currentCamera;
            hasPreviousCamera_ = true;
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
    static float finiteOr(float value, float fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    static const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return nullptr;
        }
        auto iter = properties.find("camera");
        return iter != properties.end() && iter->is_object() ? &(*iter) : nullptr;
    }

    static float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        return iter != camera->end() && iter->is_number()
            ? finiteOr(iter->get<float>(), fallback)
            : fallback;
    }

    static float3 cameraVec3(
        const RenderGraphProperties* camera,
        const char* key,
        const float3& fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_array() || iter->size() < 3) {
            return fallback;
        }
        float values[3] = {fallback.x, fallback.y, fallback.z};
        for (size_t index = 0; index < 3; ++index) {
            if ((*iter)[index].is_number()) {
                values[index] = finiteOr((*iter)[index].get<float>(), values[index]);
            }
        }
        return float3(values[0], values[1], values[2]);
    }

    static bool cameraIsOrthographic(const RenderGraphProperties* camera)
    {
        if (camera == nullptr) {
            return false;
        }
        auto iter = camera->find("projection");
        if (iter == camera->end() || !iter->is_string()) {
            return false;
        }
        const std::string projection = iter->get<std::string>();
        return projection == "orthographic" || projection == "ortho";
    }

    static NrdCameraSnapshot cameraFromProperties(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties)
    {
        const RenderGraphProperties* camera = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        NrdCameraSnapshot snapshot;
        snapshot.aspectRatio = height == 0
            ? 1.0f
            : static_cast<float>(width) / static_cast<float>(height);
        const float fovDegrees = std::clamp(cameraFloat(camera, "fovDegrees", 50.0f), 1.0f, 179.0f);
        snapshot.fovRadians = fovDegrees * (kPi / 180.0f);
        snapshot.eye = cameraVec3(camera, "eye", snapshot.eye);
        snapshot.center = cameraVec3(camera, "center", snapshot.center);
        snapshot.up = cameraVec3(camera, "up", snapshot.up);
        snapshot.zNear = std::max(cameraFloat(camera, "znear", snapshot.zNear), 0.0001f);
        snapshot.zFar = std::max(
            cameraFloat(camera, "zfar", snapshot.zFar),
            snapshot.zNear + 0.001f);
        const float cameraDistance = std::max(length(snapshot.eye - snapshot.center), 0.001f);
        const float defaultOrthoHeight = std::max(
            2.0f * cameraDistance * std::tan(snapshot.fovRadians * 0.5f),
            0.0001f);
        snapshot.orthoHeight = std::max(
            cameraFloat(camera, "orthoHeight", defaultOrthoHeight),
            0.0001f);
        snapshot.orthographic = cameraIsOrthographic(camera);
        return snapshot;
    }

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

    static float3 normalizeOr(const float3& value, const float3& fallback)
    {
        const float lengthSquared = dot(value, value);
        return lengthSquared > 0.00000001f
            ? value / std::sqrt(lengthSquared)
            : fallback;
    }

    static void writeWorldToViewMatrix(const NrdCameraSnapshot& camera, float matrix[16])
    {
        const float3 forward = normalizeOr(camera.center - camera.eye, float3(0.0f, 0.0f, -1.0f));
        const float3 right = normalizeOr(cross(forward, camera.up), float3(1.0f, 0.0f, 0.0f));
        const float3 up = normalizeOr(cross(right, forward), float3(0.0f, 1.0f, 0.0f));
        matrix[0] = right.x;
        matrix[1] = up.x;
        matrix[2] = forward.x;
        matrix[3] = 0.0f;
        matrix[4] = right.y;
        matrix[5] = up.y;
        matrix[6] = forward.y;
        matrix[7] = 0.0f;
        matrix[8] = right.z;
        matrix[9] = up.z;
        matrix[10] = forward.z;
        matrix[11] = 0.0f;
        matrix[12] = -dot(right, camera.eye);
        matrix[13] = -dot(up, camera.eye);
        matrix[14] = -dot(forward, camera.eye);
        matrix[15] = 1.0f;
    }

    static void writeViewToClipMatrix(const NrdCameraSnapshot& camera, float matrix[16])
    {
        setIdentity(matrix);
        const float zNear = std::max(camera.zNear, 0.0001f);
        const float zFar = std::max(camera.zFar, zNear + 0.001f);
        if (camera.orthographic) {
            const float height = std::max(camera.orthoHeight, 0.0001f);
            const float width = std::max(height * camera.aspectRatio, 0.0001f);
            matrix[0] = 2.0f / width;
            matrix[5] = 2.0f / height;
            matrix[10] = 1.0f / (zFar - zNear);
            matrix[14] = -zNear / (zFar - zNear);
            return;
        }

        const float yScale = 1.0f / std::tan(camera.fovRadians * 0.5f);
        const float xScale = yScale / std::max(camera.aspectRatio, 0.001f);
        std::fill(matrix, matrix + 16, 0.0f);
        matrix[0] = xScale;
        matrix[5] = yScale;
        matrix[10] = zFar / (zFar - zNear);
        matrix[11] = 1.0f;
        matrix[14] = -(zNear * zFar) / (zFar - zNear);
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
        TextureHandle diffuseConfidence,
        TextureHandle specularConfidence,
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
        put(nrd::ResourceType::IN_DIFF_CONFIDENCE, diffuseConfidence);
        put(nrd::ResourceType::IN_SPEC_CONFIDENCE, specularConfidence);
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
            hasPreviousCamera_ = false;
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
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_DIFF_CONFIDENCE, *diffuseConfidence.texture(), *diffuseConfidence.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_SPEC_CONFIDENCE, *specularConfidence.texture(), *specularConfidence.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::OUT_VALIDATION, *validation.texture(), *validation.view());
        return {};
    }

    Result runNrd(
        RenderGraphExecutionContext& context,
        uint32_t denoiserMode,
        TextureHandle noisyDiffuse,
        TextureHandle noisySpecular,
        TextureHandle denoisedDiffuse,
        TextureHandle denoisedSpecular,
        const NrdCameraSnapshot& currentCamera,
        const NrdCameraSnapshot& previousCamera)
    {
        if (nrd_ == nullptr || !nrd_->valid()) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphProperties& properties = context.properties();
        nrd::CommonSettings commonSettings;
        writeViewToClipMatrix(currentCamera, commonSettings.viewToClipMatrix);
        writeViewToClipMatrix(previousCamera, commonSettings.viewToClipMatrixPrev);
        writeWorldToViewMatrix(currentCamera, commonSettings.worldToViewMatrix);
        writeWorldToViewMatrix(previousCamera, commonSettings.worldToViewMatrixPrev);
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
        commonSettings.isHistoryConfidenceAvailable =
            denoiserMode == kNrdDenoiserModeRelax &&
            boolProperty(&properties, "relaxConfidenceInputs", true);
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
            const uint32_t historyLength = uintProperty(
                properties,
                "relaxHistoryLength",
                relaxSettings.diffuseMaxAccumulatedFrameNum,
                0,
                nrd::RELAX_MAX_HISTORY_FRAME_NUM);
            const uint32_t fastHistoryLength = std::min(
                uintProperty(
                    properties,
                    "relaxFastHistoryLength",
                    relaxSettings.diffuseMaxFastAccumulatedFrameNum,
                    0,
                    nrd::RELAX_MAX_HISTORY_FRAME_NUM),
                historyLength);
            relaxSettings.diffuseMaxAccumulatedFrameNum = historyLength;
            relaxSettings.specularMaxAccumulatedFrameNum = historyLength;
            relaxSettings.diffuseMaxFastAccumulatedFrameNum = fastHistoryLength;
            relaxSettings.specularMaxFastAccumulatedFrameNum = fastHistoryLength;
            relaxSettings.atrousIterationNum = uintProperty(
                properties,
                "relaxAtrousIterations",
                relaxSettings.atrousIterationNum,
                2,
                8);
            relaxSettings.diffusePrepassBlurRadius = floatProperty(
                properties,
                "relaxDiffusePrepassRadius",
                relaxSettings.diffusePrepassBlurRadius,
                0.0f,
                50.0f);
            relaxSettings.specularPrepassBlurRadius = floatProperty(
                properties,
                "relaxSpecularPrepassRadius",
                relaxSettings.specularPrepassBlurRadius,
                0.0f,
                50.0f);
            relaxSettings.minHitDistanceWeight = floatProperty(
                properties,
                "relaxMinHitDistanceWeight",
                relaxSettings.minHitDistanceWeight,
                0.001f,
                0.2f);
            relaxSettings.enableAntiFirefly = boolProperty(
                &properties,
                "relaxAntiFirefly",
                true);
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
    NrdCameraSnapshot previousCamera_;
    bool hasPreviousCamera_ = false;
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
