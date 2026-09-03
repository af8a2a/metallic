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

enum class DlssVariant : uint8_t {
    SuperResolution,
    RayReconstruction,
};

inline constexpr const char* kStreamlineDlssSupportShaderModuleName =
    "StreamlineDlssSupport";
inline constexpr const char* kStreamlineDlssDepthVertexEntryPoint =
    "streamlineDlssDepthVertexMain";
inline constexpr const char* kStreamlineDlssDepthFragmentEntryPoint =
    "streamlineDlssDepthFragmentMain";
inline constexpr const char* kStreamlineDlssAlphaEntryPoint =
    "streamlineDlssAlphaMain";

struct StreamlineDlssAlphaUserPush {
    uint32_t outputImage = 0;
};

class StreamlineDlssPass final : public UnsafePass {
public:
    explicit StreamlineDlssPass(DlssVariant variant)
        : variant_(variant)
    {
    }

    ~StreamlineDlssPass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        const vulkan::StreamlineDlssRrMode mode = modeFromProperties(properties());
        const bool hasPreparedExtent = preparedFor(mode, context.width, context.height);
        const uint32_t renderWidth = hasPreparedExtent ? preparedSettings_.renderWidth : 0;
        const uint32_t renderHeight = hasPreparedExtent ? preparedSettings_.renderHeight : 0;
        RenderPassReflection reflection;
        const bool rayReconstruction = variant_ == DlssVariant::RayReconstruction;
        RenderGraphField& inputColor = reflection.addTextureInput(
            "inputColor",
            rayReconstruction ? "DLSS-RR noisy HDR input color" : "DLSS-SR HDR input color")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite();
        inputColor.format = Format::Rgba16Sfloat;
        inputColor.usage = inputColor.usage | TextureUsageBits::TransferSource;

        if (rayReconstruction) {
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
            reflection.addTextureInput("specularHitDistance", "DLSS-RR specular hit distance guide")
                .texture2D(renderWidth, renderHeight)
                .storageReadWrite()
                .format = Format::R32Sfloat;
        }
        reflection.addTextureInput(
            "motionVectors",
            rayReconstruction ? "DLSS-RR motion vector guide" : "DLSS-SR motion vectors")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite()
            .format = Format::Rg16Sfloat;
        const char* depthFieldName = rayReconstruction ? "linearDepth" : "depth";
        RenderGraphField& depth = reflection.addTextureInput(
            depthFieldName,
            rayReconstruction ? "DLSS-RR linear depth guide" : "DLSS-SR normalized hardware depth")
            .texture2D(renderWidth, renderHeight)
            .storageReadWrite();
        depth.format = Format::R32Sfloat;
        if (!rayReconstruction) {
            depth.usage = depth.usage | TextureUsageBits::Sampled;
        }

        RenderGraphField& outputColor = reflection.addTextureOutput(
            "color",
            rayReconstruction ? "DLSS-RR denoised HDR output color" : "DLSS-SR upscaled HDR output color")
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
            log = std::string(passTypeName()) + " requires non-zero output dimensions";
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

        Result result = variant_ == DlssVariant::RayReconstruction
            ? vulkan::getStreamlineDlssRrOptimalSettings(
                  preparedMode_,
                  context.width,
                  context.height,
                  preparedSettings_,
                  log)
            : vulkan::getStreamlineDlssSrOptimalSettings(
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
        log = std::string(passTypeName()) + " requires the NVIDIA Streamline SDK target";
        return makeError(Error::Unsupported);
#else
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = std::string(passTypeName()) + " requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        const bool featureSupported = variant_ == DlssVariant::RayReconstruction
            ? context.device->capabilities().streamlineDlssRr
            : context.device->capabilities().streamlineDlssSr;
        if (!context.device->capabilities().streamline || !featureSupported) {
            log = std::string(passTypeName()) + " requires DeviceCapabilities::" +
                (variant_ == DlssVariant::RayReconstruction ? "streamlineDlssRr" : "streamlineDlssSr");
            return makeError(Error::Unsupported);
        }
        if (!preparedFor(modeFromProperties(properties()), context.width, context.height)) {
            log = std::string(passTypeName()) + " was not prepared with optimal input dimensions";
            return makeError(Error::InvalidArgument);
        }
        if (variant_ == DlssVariant::SuperResolution &&
            preparedMode_ != vulkan::StreamlineDlssRrMode::Off) {
            if (!context.device->capabilities().bindlessDescriptorHeap) {
                log = "StreamlineDlssSrPass requires DeviceCapabilities::bindlessDescriptorHeap "
                    "for depth export and alpha resolve";
                return makeError(Error::Unsupported);
            }
            return prepareSuperResolutionResources(
                *context.device,
                preparedSettings_.renderWidth,
                preparedSettings_.renderHeight,
                log);
        }
        return {};
#endif
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle inputColor = context.inputTexture("inputColor");
        TextureHandle outputColor = context.outputTexture("color");
        TextureHandle motionVectors = context.inputTexture("motionVectors");
        const bool rayReconstruction = variant_ == DlssVariant::RayReconstruction;
        TextureHandle depth = context.inputTexture(rayReconstruction ? "linearDepth" : "depth");
        TextureHandle albedo = rayReconstruction ? context.inputTexture("albedo") : TextureHandle{};
        TextureHandle specularAlbedo = rayReconstruction
            ? context.inputTexture("specularAlbedo")
            : TextureHandle{};
        TextureHandle normalRoughness = rayReconstruction
            ? context.inputTexture("normalRoughness")
            : TextureHandle{};
        TextureHandle specularHitDistance = rayReconstruction
            ? context.inputTexture("specularHitDistance")
            : TextureHandle{};
        if (!validTexture(inputColor) ||
            !validTexture(outputColor) ||
            !validTexture(motionVectors) ||
            !validTexture(depth) ||
            (rayReconstruction &&
                (!validTexture(albedo) ||
                    !validTexture(specularAlbedo) ||
                    !validTexture(normalRoughness) ||
                    !validTexture(specularHitDistance)))) {
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
                "[Streamline] {} pass extent mismatch: mode={} input={}x{} output={}x{} "
                "preparedInput={}x{} preparedOutput={}x{}",
                featureName(),
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
        Result result;
        if (rayReconstruction) {
            result = vulkan::evaluateStreamlineDlssRr(
                context.commandBuffer(),
                vulkan::StreamlineDlssRrDesc{
                    .inputColor = textureRef(inputColor),
                    .outputColor = textureRef(outputColor),
                    .albedo = textureRef(albedo),
                    .specularAlbedo = textureRef(specularAlbedo),
                    .normalRoughness = textureRef(normalRoughness),
                    .motionVectors = textureRef(motionVectors),
                    .linearDepth = textureRef(depth),
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
        } else {
            Result depthResult = exportSuperResolutionDepth(
                context.commandBuffer(),
                depth,
                renderWidth,
                renderHeight);
            if (!depthResult) {
                return depthResult;
            }
            result = vulkan::evaluateStreamlineDlssSr(
                context.commandBuffer(),
                vulkan::StreamlineDlssSrDesc{
                    .inputColor = textureRef(inputColor),
                    .outputColor = textureRef(outputColor),
                    .motionVectors = textureRef(motionVectors),
                    .depth = vulkan::StreamlineDlssSrTextureRef{
                        .texture = dlssDepth_.get(),
                        .view = dlssDepthView_.get(),
                    },
                    .renderWidth = renderWidth,
                    .renderHeight = renderHeight,
                    .outputWidth = outputWidth,
                    .outputHeight = outputHeight,
                    .camera = camera,
                    .mode = mode,
                    .reset = reset || !previousCameraValid,
                },
                log);
            if (result) {
                result = resolveSuperResolutionAlpha(
                    context.commandBuffer(),
                    outputColor);
                if (!result) {
                    log = "DLSS-SR alpha resolve failed";
                }
            }
        }
        if (!result && !log.empty()) {
            spdlog::error("[Streamline] {} evaluate failed: {}", featureName(), log);
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
    const char* passTypeName() const
    {
        return variant_ == DlssVariant::RayReconstruction
            ? "StreamlineDlssRrPass"
            : "StreamlineDlssSrPass";
    }

    const char* featureName() const
    {
        return variant_ == DlssVariant::RayReconstruction ? "DLSS-RR" : "DLSS-SR";
    }

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

    Result prepareSuperResolutionResources(
        Device& device,
        uint32_t renderWidth,
        uint32_t renderHeight,
        std::string& log)
    {
        if (renderWidth == 0 || renderHeight == 0) {
            log = "StreamlineDlssSrPass depth export requires non-zero dimensions";
            return makeError(Error::InvalidArgument);
        }

        Result result;
        if (auxiliaryHeap_ == nullptr) {
            result = device.createBindlessHeap(
                BindlessHeapDesc{
                    .maxSampledImages = 1,
                    .maxStorageImages = 1,
                },
                auxiliaryHeap_);
            if (!result || auxiliaryHeap_ == nullptr) {
                log += resultMessage("createBindlessHeap(StreamlineDlssSrPass)", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
            result = auxiliaryHeap_->allocateSampledImage(depthGuideHandle_);
            if (!result || !depthGuideHandle_.valid() || depthGuideHandle_.index != 0) {
                log = "StreamlineDlssSrPass failed to allocate its depth guide descriptor";
                return result ? makeError(Error::Failure) : result;
            }
            result = auxiliaryHeap_->allocateStorageImage(outputColorHandle_);
            if (!result || !outputColorHandle_.valid()) {
                log = "StreamlineDlssSrPass failed to allocate its output descriptor";
                return result ? makeError(Error::Failure) : result;
            }
        }

        if (depthExportPipeline_ == nullptr) {
            result = createSlangShaderModule(
                device,
                kStreamlineDlssSupportShaderModuleName,
                kStreamlineDlssDepthVertexEntryPoint,
                depthVertexShader_,
                log);
            if (!result) {
                return result;
            }
            result = createSlangShaderModule(
                device,
                kStreamlineDlssSupportShaderModuleName,
                kStreamlineDlssDepthFragmentEntryPoint,
                depthFragmentShader_,
                log);
            if (!result) {
                return result;
            }
            result = device.createGraphicsPipeline(
                GraphicsPipelineDesc{
                    .vertexShader = depthVertexShader_.get(),
                    .fragmentShader = depthFragmentShader_.get(),
                    .depthStencilFormat = Format::D32Sfloat,
                    .topology = PrimitiveTopology::TriangleList,
                    .depthStencil = DepthStencilState{
                        .depthTestEnable = true,
                        .depthWriteEnable = true,
                        .depthCompareOp = CompareOp::Always,
                    },
                    .usesBindlessHeap = true,
                },
                depthExportPipeline_);
            if (!result || depthExportPipeline_ == nullptr) {
                log += resultMessage("createGraphicsPipeline(StreamlineDlssSrPass depth export)", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
        }

        if (alphaResolvePipeline_ == nullptr) {
            result = createSlangShaderModule(
                device,
                kStreamlineDlssSupportShaderModuleName,
                kStreamlineDlssAlphaEntryPoint,
                alphaShader_,
                log);
            if (!result) {
                return result;
            }
            result = device.createComputePipeline(
                ComputePipelineDesc{
                    .computeShader = alphaShader_.get(),
                    .usesBindlessHeap = true,
                    .bindlessUserPushDataSize = sizeof(StreamlineDlssAlphaUserPush),
                },
                alphaResolvePipeline_);
            if (!result || alphaResolvePipeline_ == nullptr) {
                log += resultMessage("createComputePipeline(StreamlineDlssSrPass alpha resolve)", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
        }

        if (dlssDepth_ != nullptr &&
            dlssDepthWidth_ == renderWidth &&
            dlssDepthHeight_ == renderHeight) {
            return {};
        }

        dlssDepthView_.reset();
        dlssDepth_.reset();
        dlssDepthState_ = ResourceState::Undefined;
        result = device.createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::DepthStencilAttachment | TextureUsageBits::Sampled,
                .format = Format::D32Sfloat,
                .width = renderWidth,
                .height = renderHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            },
            dlssDepth_);
        if (!result || dlssDepth_ == nullptr) {
            log += resultMessage("createTexture(StreamlineDlssSrPass D32 depth)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createTextureView(
            *dlssDepth_,
            TextureViewDesc{
                .format = Format::D32Sfloat,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            dlssDepthView_);
        if (!result || dlssDepthView_ == nullptr) {
            log += resultMessage("createTextureView(StreamlineDlssSrPass D32 depth)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        dlssDepthWidth_ = renderWidth;
        dlssDepthHeight_ = renderHeight;
        return {};
    }

    Result exportSuperResolutionDepth(
        CommandBuffer& commandBuffer,
        TextureHandle depthGuide,
        uint32_t renderWidth,
        uint32_t renderHeight)
    {
        if (!validTexture(depthGuide) ||
            dlssDepth_ == nullptr ||
            dlssDepthView_ == nullptr ||
            auxiliaryHeap_ == nullptr ||
            depthExportPipeline_ == nullptr ||
            renderWidth != dlssDepthWidth_ ||
            renderHeight != dlssDepthHeight_) {
            return makeError(Error::InvalidArgument);
        }

        Result result = auxiliaryHeap_->writeSampledImage(
            depthGuideHandle_,
            *depthGuide.view(),
            ResourceState::ShaderRead);
        if (!result) {
            return result;
        }

        TextureBarrierDesc toDepthExport[] = {
            TextureBarrierDesc{
                .texture = depthGuide.texture(),
                .before = ResourceState::General,
                .after = ResourceState::ShaderRead,
                .baseMip = 0,
                .mipCount = depthGuide.desc().mipCount,
                .baseLayer = 0,
                .layerCount = depthGuide.desc().layerCount,
            },
            TextureBarrierDesc{
                .texture = dlssDepth_.get(),
                .before = dlssDepthState_,
                .after = ResourceState::DepthStencilAttachment,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = toDepthExport,
            .textureCount = static_cast<uint32_t>(std::size(toDepthExport)),
        });
        dlssDepthState_ = ResourceState::DepthStencilAttachment;

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = renderWidth,
            .height = renderHeight,
        };
        RenderingAttachmentDesc depthAttachment{
            .view = dlssDepthView_.get(),
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearDepth = 1.0f,
        };
        commandBuffer.beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .depthStencilAttachment = &depthAttachment,
        });
        commandBuffer.setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(renderWidth),
            .height = static_cast<float>(renderHeight),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        commandBuffer.setScissor(renderArea);
        commandBuffer.bindBindlessHeap(*auxiliaryHeap_);
        commandBuffer.bindGraphicsPipeline(*depthExportPipeline_);
        commandBuffer.draw(3);
        commandBuffer.endRendering();

        TextureBarrierDesc toStreamline[] = {
            TextureBarrierDesc{
                .texture = depthGuide.texture(),
                .before = ResourceState::ShaderRead,
                .after = ResourceState::General,
                .baseMip = 0,
                .mipCount = depthGuide.desc().mipCount,
                .baseLayer = 0,
                .layerCount = depthGuide.desc().layerCount,
            },
            TextureBarrierDesc{
                .texture = dlssDepth_.get(),
                .before = dlssDepthState_,
                .after = ResourceState::General,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = toStreamline,
            .textureCount = static_cast<uint32_t>(std::size(toStreamline)),
        });
        dlssDepthState_ = ResourceState::General;
        return {};
    }

    Result resolveSuperResolutionAlpha(
        CommandBuffer& commandBuffer,
        TextureHandle outputColor)
    {
        if (!validTexture(outputColor) ||
            auxiliaryHeap_ == nullptr ||
            alphaResolvePipeline_ == nullptr ||
            !outputColorHandle_.valid()) {
            return makeError(Error::InvalidArgument);
        }
        Result result = auxiliaryHeap_->writeStorageImage(
            outputColorHandle_,
            *outputColor.view());
        if (!result) {
            return result;
        }

        TextureBarrierDesc outputBarrier{
            .texture = outputColor.texture(),
            .before = ResourceState::General,
            .after = ResourceState::General,
            .baseMip = 0,
            .mipCount = outputColor.desc().mipCount,
            .baseLayer = 0,
            .layerCount = outputColor.desc().layerCount,
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = &outputBarrier,
            .textureCount = 1,
        });
        commandBuffer.bindBindlessHeap(*auxiliaryHeap_);
        const StreamlineDlssAlphaUserPush push{
            .outputImage = outputColorHandle_.index,
        };
        commandBuffer.bindComputePipeline(
            *alphaResolvePipeline_,
            &push,
            sizeof(push));
        commandBuffer.dispatch(
            (outputColor.desc().width + 7u) / 8u,
            (outputColor.desc().height + 7u) / 8u,
            1);
        return {};
    }

    DlssVariant variant_ = DlssVariant::SuperResolution;
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
    std::unique_ptr<BindlessHeap> auxiliaryHeap_;
    BindlessHandle depthGuideHandle_;
    BindlessHandle outputColorHandle_;
    std::unique_ptr<ShaderModule> depthVertexShader_;
    std::unique_ptr<ShaderModule> depthFragmentShader_;
    std::unique_ptr<ShaderModule> alphaShader_;
    std::unique_ptr<GraphicsPipeline> depthExportPipeline_;
    std::unique_ptr<ComputePipeline> alphaResolvePipeline_;
    std::unique_ptr<Texture> dlssDepth_;
    std::unique_ptr<TextureView> dlssDepthView_;
    uint32_t dlssDepthWidth_ = 0;
    uint32_t dlssDepthHeight_ = 0;
    ResourceState dlssDepthState_ = ResourceState::Undefined;
};

} // namespace

std::unique_ptr<RenderGraphPass> createStreamlineDlssRrPass()
{
    return std::make_unique<StreamlineDlssPass>(DlssVariant::RayReconstruction);
}

std::unique_ptr<RenderGraphPass> createStreamlineDlssSrPass()
{
    return std::make_unique<StreamlineDlssPass>(DlssVariant::SuperResolution);
}

} // namespace metallic::render::builtin_pass
