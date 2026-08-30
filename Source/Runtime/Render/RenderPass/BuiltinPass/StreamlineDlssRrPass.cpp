#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"

#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {
namespace {

struct DlssRrCameraSnapshot {
    float eye[3] = {};
    float center[3] = {};
    float up[3] = {};
    float fovRadians = 0.87266463f;
    float aspectRatio = 1.0f;
    float zNear = 0.001f;
    float zFar = 10000.0f;
    float orthoHeight = 1.0f;
    bool orthographic = false;
};

class StreamlineDlssRrPass final : public UnsafePass {
public:
    ~StreamlineDlssRrPass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        const vulkan::StreamlineDlssRrMode mode = modeFromProperties(properties());
        const bool hasPreparedExtent = preparedFor(mode, context.width, context.height);
        const uint32_t renderWidth = hasPreparedExtent ? preparedSettings_.renderWidth : 0;
        const uint32_t renderHeight = hasPreparedExtent ? preparedSettings_.renderHeight : 0;
        RenderPassReflection reflection;
        RenderGraphField& inputColor = reflection.addTextureInput("inputColor", "DLSS-RR noisy HDR input color")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite();
        inputColor.format = Format::Rgba16Sfloat;
        inputColor.usage = inputColor.usage | TextureUsageBits::TransferSource;

        reflection.addTextureInput("albedo", "DLSS-RR diffuse albedo guide")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("specularAlbedo", "DLSS-RR specular albedo guide")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("normalRoughness", "DLSS-RR packed normal and roughness guide")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("motionVectors", "DLSS-RR motion vector guide")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite()
            .format = Format::Rg16Sfloat;
        reflection.addTextureInput("linearDepth", "DLSS-RR linear depth guide")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite()
            .format = Format::R32Sfloat;
        reflection.addTextureInput("specularHitDistance", "DLSS-RR specular hit distance guide")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite()
            .format = Format::R32Sfloat;

        RenderGraphField& outputColor = reflection.addTextureOutput("color", "DLSS-RR denoised HDR output color")
            .texture2D(context.width, context.height)
            .storageReadWrite();
        outputColor.format = Format::Rgba16Sfloat;
        outputColor.usage = outputColor.usage | TextureUsageBits::TransferDestination;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeEnumSetting(
                "mode",
                "Mode",
                "Balanced",
                {
                    {"Balanced", "Balanced"},
                    {"Quality", "Quality"},
                    {"Performance", "Performance"},
                    {"Ultra Performance", "UltraPerformance"},
                    {"Ultra Quality", "UltraQuality"},
                    {"DLAA", "DLAA"},
                    {"Off", "Off"},
                },
                true,
                true),
            runtimeActionCounterSetting("resetSerial", "Reset", true),
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 0.2f, 2.5f},
            std::array<float, 3>{0.0f, 0.0f, 0.0f},
            50.0f,
            true);
        return settings;
    }

    Result prepare(const RenderGraphCompileContext& context, std::string& log) override
    {
        forceReset_ = true;
        preparedValid_ = false;
        preparedSettings_ = {};
        preparedMode_ = modeFromProperties(properties());
        preparedOutputWidth_ = context.width;
        preparedOutputHeight_ = context.height;
        if (context.device == nullptr) {
            return {};
        }
        if (context.width == 0 || context.height == 0) {
            log = "StreamlineDlssRrPass requires non-zero output dimensions";
            return makeError(Error::InvalidArgument);
        }
        if (preparedMode_ == vulkan::StreamlineDlssRrMode::Off) {
            preparedSettings_.renderWidth = context.width;
            preparedSettings_.renderHeight = context.height;
            preparedSettings_.renderWidthMin = context.width;
            preparedSettings_.renderHeightMin = context.height;
            preparedSettings_.renderWidthMax = context.width;
            preparedSettings_.renderHeightMax = context.height;
            preparedValid_ = true;
            return {};
        }

        Result result = vulkan::getStreamlineDlssRrOptimalSettings(
            preparedMode_,
            context.width,
            context.height,
            preparedSettings_,
            log);
        preparedValid_ = result.has_value();
        return result;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
#if !METALLIC_HAS_STREAMLINE
        log = "StreamlineDlssRrPass requires the NVIDIA Streamline SDK target";
        return makeError(Error::Unsupported);
#else
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "StreamlineDlssRrPass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().streamline ||
            !context.device->capabilities().streamlineDlssRr) {
            log = "StreamlineDlssRrPass requires DeviceCapabilities::streamlineDlssRr";
            return makeError(Error::Unsupported);
        }
        if (!preparedFor(modeFromProperties(properties()), context.width, context.height)) {
            log = "StreamlineDlssRrPass was not prepared with optimal input dimensions";
            return makeError(Error::InvalidArgument);
        }
        return {};
#endif
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle inputColor = context.inputTexture("inputColor");
        TextureHandle outputColor = context.outputTexture("color");
        TextureHandle albedo = context.inputTexture("albedo");
        TextureHandle specularAlbedo = context.inputTexture("specularAlbedo");
        TextureHandle normalRoughness = context.inputTexture("normalRoughness");
        TextureHandle motionVectors = context.inputTexture("motionVectors");
        TextureHandle linearDepth = context.inputTexture("linearDepth");
        TextureHandle specularHitDistance = context.inputTexture("specularHitDistance");
        if (!validTexture(inputColor) ||
            !validTexture(outputColor) ||
            !validTexture(albedo) ||
            !validTexture(specularAlbedo) ||
            !validTexture(normalRoughness) ||
            !validTexture(motionVectors) ||
            !validTexture(linearDepth) ||
            !validTexture(specularHitDistance)) {
            return makeError(Error::InvalidArgument);
        }

        const vulkan::StreamlineDlssRrMode mode = modeFromProperties(context.properties());
        const uint32_t renderWidth = inputColor.desc().width;
        const uint32_t renderHeight = inputColor.desc().height;
        const uint32_t outputWidth = outputColor.desc().width;
        const uint32_t outputHeight = outputColor.desc().height;
        if (!preparedFor(mode, outputWidth, outputHeight) ||
            preparedSettings_.renderWidth != renderWidth ||
            preparedSettings_.renderHeight != renderHeight) {
            spdlog::error(
                "[Streamline] DLSS-RR pass extent mismatch: mode={} input={}x{} output={}x{} "
                "preparedInput={}x{} preparedOutput={}x{}",
                static_cast<uint32_t>(mode),
                renderWidth,
                renderHeight,
                outputWidth,
                outputHeight,
                preparedSettings_.renderWidth,
                preparedSettings_.renderHeight,
                preparedOutputWidth_,
                preparedOutputHeight_);
            return makeError(Error::InvalidArgument);
        }
        if (mode == vulkan::StreamlineDlssRrMode::Off) {
            if (renderWidth != outputWidth || renderHeight != outputHeight) {
                return makeError(Error::InvalidArgument);
            }
            copyInputToOutput(context.commandBuffer(), inputColor, outputColor);
            lastMode_ = vulkan::StreamlineDlssRrMode::Off;
            lastRenderWidth_ = renderWidth;
            lastRenderHeight_ = renderHeight;
            lastOutputWidth_ = outputWidth;
            lastOutputHeight_ = outputHeight;
            hasPreviousCamera_ = false;
            forceReset_ = false;
            return {};
        }

        const uint32_t resetSerial = uintProperty(
            context.properties(),
            "resetSerial",
            0,
            0,
            std::numeric_limits<uint32_t>::max());
        const bool reset =
            forceReset_ ||
            lastMode_ != mode ||
            lastResetSerial_ != resetSerial ||
            lastRenderWidth_ != renderWidth ||
            lastRenderHeight_ != renderHeight ||
            lastOutputWidth_ != outputWidth ||
            lastOutputHeight_ != outputHeight;
        vulkan::StreamlineDlssRrCamera camera = cameraFromProperties(
            renderWidth,
            renderHeight,
            context.properties());
        const DlssRrCameraSnapshot currentCamera = cameraSnapshotFrom(camera);
        const bool previousCameraValid =
            !reset &&
            hasPreviousCamera_ &&
            previousCameraWidth_ == outputWidth &&
            previousCameraHeight_ == outputHeight;
        applyPreviousCamera(previousCameraValid ? previousCamera_ : currentCamera, camera);
        camera.previousValid = previousCameraValid;

        std::string log;
        Result result = vulkan::evaluateStreamlineDlssRr(
            context.commandBuffer(),
            vulkan::StreamlineDlssRrDesc{
                .inputColor = textureRef(inputColor),
                .outputColor = textureRef(outputColor),
                .albedo = textureRef(albedo),
                .specularAlbedo = textureRef(specularAlbedo),
                .normalRoughness = textureRef(normalRoughness),
                .motionVectors = textureRef(motionVectors),
                .linearDepth = textureRef(linearDepth),
                .specularHitDistance = textureRef(specularHitDistance),
                .renderWidth = renderWidth,
                .renderHeight = renderHeight,
                .outputWidth = outputWidth,
                .outputHeight = outputHeight,
                .camera = camera,
                .mode = mode,
                .reset = reset || !previousCameraValid,
            },
            log);
        if (!result && !log.empty()) {
            spdlog::error("[Streamline] DLSS-RR evaluate failed: {}", log);
        }
        if (result) {
            lastMode_ = mode;
            lastResetSerial_ = resetSerial;
            lastRenderWidth_ = renderWidth;
            lastRenderHeight_ = renderHeight;
            lastOutputWidth_ = outputWidth;
            lastOutputHeight_ = outputHeight;
            previousCamera_ = currentCamera;
            previousCameraWidth_ = outputWidth;
            previousCameraHeight_ = outputHeight;
            hasPreviousCamera_ = true;
            forceReset_ = false;
        }
        return result;
    }

private:
    bool preparedFor(
        vulkan::StreamlineDlssRrMode mode,
        uint32_t outputWidth,
        uint32_t outputHeight) const
    {
        return preparedValid_ &&
            preparedMode_ == mode &&
            preparedOutputWidth_ == outputWidth &&
            preparedOutputHeight_ == outputHeight;
    }

    static bool validTexture(TextureHandle texture)
    {
        return texture.valid() && texture.texture() != nullptr && texture.view() != nullptr;
    }

    static vulkan::StreamlineDlssRrTextureRef textureRef(TextureHandle texture)
    {
        return vulkan::StreamlineDlssRrTextureRef{
            .texture = texture.texture(),
            .view = texture.view(),
        };
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

    static std::string stringProperty(
        const RenderGraphProperties& properties,
        const char* key,
        std::string_view fallback)
    {
        if (!properties.is_object()) {
            return std::string(fallback);
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_string()) {
            return std::string(fallback);
        }
        return iter->get<std::string>();
    }

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
        if (iter == properties.end() || !iter->is_object()) {
            return nullptr;
        }
        return &(*iter);
    }

    static float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_number()) {
            return fallback;
        }
        return finiteOr(iter->get<float>(), fallback);
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
            const RenderGraphProperties& component = (*iter)[index];
            if (component.is_number()) {
                values[index] = finiteOr(component.get<float>(), values[index]);
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

    static void writeFloat3(const float3& value, float out[3])
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
    }

    static void copyFloat3(const float source[3], float target[3])
    {
        std::copy(source, source + 3, target);
    }

    static vulkan::StreamlineDlssRrCamera cameraFromProperties(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties)
    {
        vulkan::StreamlineDlssRrCamera camera;
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 50.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float3 eye = cameraVec3(cameraProperties, "eye", float3(0.0f, 0.2f, 2.5f));
        const float3 center = cameraVec3(cameraProperties, "center", float3(0.0f, 0.0f, 0.0f));
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.001f), 0.0001f);
        const float zFar = std::max(cameraFloat(cameraProperties, "zfar", 10000.0f), zNear + 0.001f);
        const float cameraDistance = std::max(length(eye - center), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(
            cameraFloat(cameraProperties, "orthoHeight", defaultOrthoHeight),
            0.0001f);

        writeFloat3(eye, camera.eye);
        writeFloat3(center, camera.center);
        writeFloat3(up, camera.up);
        camera.fovRadians = fovRadians;
        camera.aspectRatio = aspect;
        camera.zNear = zNear;
        camera.zFar = zFar;
        camera.orthoHeight = orthoHeight;
        camera.orthographic = cameraIsOrthographic(cameraProperties);
        return camera;
    }

    static DlssRrCameraSnapshot cameraSnapshotFrom(const vulkan::StreamlineDlssRrCamera& camera)
    {
        DlssRrCameraSnapshot snapshot;
        copyFloat3(camera.eye, snapshot.eye);
        copyFloat3(camera.center, snapshot.center);
        copyFloat3(camera.up, snapshot.up);
        snapshot.fovRadians = camera.fovRadians;
        snapshot.aspectRatio = camera.aspectRatio;
        snapshot.zNear = camera.zNear;
        snapshot.zFar = camera.zFar;
        snapshot.orthoHeight = camera.orthoHeight;
        snapshot.orthographic = camera.orthographic;
        return snapshot;
    }

    static void applyPreviousCamera(
        const DlssRrCameraSnapshot& previous,
        vulkan::StreamlineDlssRrCamera& camera)
    {
        copyFloat3(previous.eye, camera.previousEye);
        copyFloat3(previous.center, camera.previousCenter);
        copyFloat3(previous.up, camera.previousUp);
        camera.previousFovRadians = previous.fovRadians;
        camera.previousAspectRatio = previous.aspectRatio;
        camera.previousZNear = previous.zNear;
        camera.previousZFar = previous.zFar;
        camera.previousOrthoHeight = previous.orthoHeight;
        camera.previousOrthographic = previous.orthographic;
    }

    static vulkan::StreamlineDlssRrMode modeFromProperties(const RenderGraphProperties& properties)
    {
        const std::string mode = stringProperty(properties, "mode", "Balanced");
        if (mode == "Off" || mode == "off") {
            return vulkan::StreamlineDlssRrMode::Off;
        }
        if (mode == "DLAA" || mode == "Dlaa" || mode == "dlaa") {
            return vulkan::StreamlineDlssRrMode::Dlaa;
        }
        if (mode == "Quality" || mode == "quality") {
            return vulkan::StreamlineDlssRrMode::Quality;
        }
        if (mode == "Performance" || mode == "performance") {
            return vulkan::StreamlineDlssRrMode::Performance;
        }
        if (mode == "UltraPerformance" || mode == "Ultra Performance" || mode == "ultraPerformance") {
            return vulkan::StreamlineDlssRrMode::UltraPerformance;
        }
        if (mode == "UltraQuality" || mode == "Ultra Quality" || mode == "ultraQuality") {
            return vulkan::StreamlineDlssRrMode::UltraQuality;
        }
        return vulkan::StreamlineDlssRrMode::Balanced;
    }

    static void copyInputToOutput(CommandBuffer& commandBuffer, TextureHandle inputColor, TextureHandle outputColor)
    {
        TextureBarrierDesc toTransfer[] = {
            TextureBarrierDesc{
                .texture = inputColor.texture(),
                .before = ResourceState::General,
                .after = ResourceState::TransferSource,
                .baseMip = 0,
                .mipCount = inputColor.desc().mipCount,
                .baseLayer = 0,
                .layerCount = inputColor.desc().layerCount,
            },
            TextureBarrierDesc{
                .texture = outputColor.texture(),
                .before = ResourceState::General,
                .after = ResourceState::TransferDestination,
                .baseMip = 0,
                .mipCount = outputColor.desc().mipCount,
                .baseLayer = 0,
                .layerCount = outputColor.desc().layerCount,
            },
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = toTransfer,
            .textureCount = static_cast<uint32_t>(std::size(toTransfer)),
        });
        commandBuffer.copyTexture(TextureCopyDesc{
            .source = inputColor.texture(),
            .destination = outputColor.texture(),
            .width = outputColor.desc().width,
            .height = outputColor.desc().height,
            .depth = 1,
            .sourceMipLevel = 0,
            .sourceBaseLayer = 0,
            .destinationMipLevel = 0,
            .destinationBaseLayer = 0,
        });
        TextureBarrierDesc toGeneral[] = {
            TextureBarrierDesc{
                .texture = inputColor.texture(),
                .before = ResourceState::TransferSource,
                .after = ResourceState::General,
                .baseMip = 0,
                .mipCount = inputColor.desc().mipCount,
                .baseLayer = 0,
                .layerCount = inputColor.desc().layerCount,
            },
            TextureBarrierDesc{
                .texture = outputColor.texture(),
                .before = ResourceState::TransferDestination,
                .after = ResourceState::General,
                .baseMip = 0,
                .mipCount = outputColor.desc().mipCount,
                .baseLayer = 0,
                .layerCount = outputColor.desc().layerCount,
            },
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = toGeneral,
            .textureCount = static_cast<uint32_t>(std::size(toGeneral)),
        });
    }

    vulkan::StreamlineDlssRrMode lastMode_ = vulkan::StreamlineDlssRrMode::Off;
    uint32_t lastResetSerial_ = 0;
    uint32_t lastRenderWidth_ = 0;
    uint32_t lastRenderHeight_ = 0;
    uint32_t lastOutputWidth_ = 0;
    uint32_t lastOutputHeight_ = 0;
    DlssRrCameraSnapshot previousCamera_;
    uint32_t previousCameraWidth_ = 0;
    uint32_t previousCameraHeight_ = 0;
    bool hasPreviousCamera_ = false;
    vulkan::StreamlineDlssRrOptimalSettings preparedSettings_;
    vulkan::StreamlineDlssRrMode preparedMode_ = vulkan::StreamlineDlssRrMode::Off;
    uint32_t preparedOutputWidth_ = 0;
    uint32_t preparedOutputHeight_ = 0;
    bool preparedValid_ = false;
    bool forceReset_ = true;
};

} // namespace

std::unique_ptr<RenderGraphPass> createStreamlineDlssRrPass()
{
    return std::make_unique<StreamlineDlssRrPass>();
}

} // namespace metallic::render::builtin_pass
