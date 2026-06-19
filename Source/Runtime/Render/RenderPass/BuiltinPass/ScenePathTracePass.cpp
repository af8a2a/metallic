#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class ScenePathTracePass final : public ComputePass {
public:
    ~ScenePathTracePass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Path-traced glTF scene")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "ScenePathTracePass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().rayTracingAccelerationStructure ||
            !context.device->capabilities().rayQuery) {
            log = "ScenePathTracePass requires rayTracingAccelerationStructure and rayQuery capabilities";
            return makeError(Error::Unsupported);
        }

        Result result = sceneResources_.prepare(*context.device, *context.graphicsQueue, properties(), log);
        if (!result) {
            return result;
        }
        const uint64_t resourceRevision = sceneResources_.revision();
        if (resourceRevision != sceneResourceRevision_) {
            sceneResourceRevision_ = resourceRevision;
            resetAccumulation_ = true;
        }
        if (rayQueryProgram_.valid()) {
            return {};
        }

        ShaderCompileResult computeCompile;
        const char* capabilities[] = {"spvRayQueryKHR"};
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kScenePathTraceShaderModuleName,
                .entryPointName = kScenePathTraceEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .profileName = "glsl_460",
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            computeCompile);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += kScenePathTraceShaderModuleName;
            log += ".";
            log += kScenePathTraceEntryPoint;
            log += ") returned ";
            log += resultToString(result);
            if (!computeCompile.diagnostics.empty()) {
                log += ": ";
                log += computeCompile.diagnostics;
            }
            log += '\n';
            rayQueryProgram_.clear();
            return result;
        }

        const SceneRayQueryBindingDesc bindings[] = {
            SceneRayQueryBindingDesc{
                .binding = 0,
                .kind = SceneRayQueryBindingKind::AccelerationStructure,
            },
            SceneRayQueryBindingDesc{
                .binding = 1,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 2,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 3,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 4,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 5,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 6,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 7,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 8,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 9,
                .kind = SceneRayQueryBindingKind::SampledImage,
                .descriptorCount = kScenePathTraceMaxMaterialTextures,
            },
        };
        std::string programLog;
        result = rayQueryProgram_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = computeCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(computeCompile.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(ScenePathTracePush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "ScenePathTracePass",
            },
            programLog);
        if (!programLog.empty()) {
            if (!log.empty() && log.back() != '\n') {
                log += '\n';
            }
            log += programLog;
        }
        if (!result) {
            rayQueryProgram_.clear();
            return result;
        }
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        const auto& materialTextureViews = sceneResources_.materialTextureViews();
        if (!color.valid() ||
            color.view() == nullptr ||
            !rayQueryProgram_.valid() ||
            !sceneResources_.valid() ||
            materialTextureViews[0] == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        ScenePathTracePush push;
        buildPush(context.width(), context.height(), context.properties(), sceneResources_.bounds(), push);
        push.materialTextureCount = sceneResources_.materialTextureCount();

        TextureView* historyCurrentView = color.view();
        TextureView* historyPreviousView = color.view();
        Result result = prepareHistoryTextures(
            context,
            *color.view(),
            push,
            historyCurrentView,
            historyPreviousView);
        if (!result) {
            return result;
        }
        if (historyCurrentView == nullptr || historyPreviousView == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        result = sceneResources_.uploadMaterialTextures(context.commandBuffer());
        if (!result) {
            return result;
        }

        const SceneRayQueryDispatchBinding bindings[] = {
            SceneRayQueryDispatchBinding{
                .binding = 0,
                .accelerationStructure = &sceneResources_.accelerationStructure(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 1,
                .textureView = color.view(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 2,
                .buffer = sceneResources_.vertexBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 3,
                .buffer = sceneResources_.indexBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 4,
                .buffer = sceneResources_.primitiveBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 5,
                .buffer = sceneResources_.instanceBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 6,
                .buffer = sceneResources_.materialBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 7,
                .textureView = historyCurrentView,
            },
            SceneRayQueryDispatchBinding{
                .binding = 8,
                .textureView = historyPreviousView,
            },
            SceneRayQueryDispatchBinding{
                .binding = 9,
                .textureViews = materialTextureViews.data(),
                .textureViewCount = static_cast<uint32_t>(materialTextureViews.size()),
            },
        };
        result = rayQueryProgram_.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
            .groupCountZ = 1,
        });
        if (!result) {
            return result;
        }
        if (push.enableAccumulation != 0 && context.historyResources() != nullptr) {
            context.historyResources()->markWritten(historyNameForContext(context));
        }
        return {};
    }

private:
    Result prepareHistoryTextures(
        RenderGraphExecutionContext& context,
        TextureView& fallbackView,
        ScenePathTracePush& push,
        TextureView*& outCurrentView,
        TextureView*& outPreviousView)
    {
        HistoryResourceManager* history = context.historyResources();
        const bool accumulationEnabled = boolProperty(context.properties(), "accumulate", true);
        push.enableAccumulation = accumulationEnabled && history != nullptr ? 1u : 0u;
        push.hasHistory = 0;
        push.accumulationFrame = 0;
        outCurrentView = &fallbackView;
        outPreviousView = &fallbackView;
        if (push.enableAccumulation == 0) {
            accumulationFrame_ = 0;
            resetAccumulation_ = true;
            return {};
        }

        const TextureDesc historyDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled |
                TextureUsageBits::Storage |
                TextureUsageBits::TransferSource,
            .format = Format::Rgba8Unorm,
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        const std::string historyName = historyNameForContext(context);
        Result result = history->ensureTexture(
            historyName,
            historyDesc,
            TextureViewDesc{.format = Format::Rgba8Unorm});
        if (!result) {
            return result;
        }

        HistoryTextureRef current = history->texture(historyName, HistorySlot::Current);
        HistoryTextureRef previous = history->texture(historyName, HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr || previous.texture == nullptr || previous.view == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        result = history->transitionTexture(
            context.commandBuffer(),
            historyName,
            HistorySlot::Current,
            ResourceState::General);
        if (!result) {
            return result;
        }
        result = history->transitionTexture(
            context.commandBuffer(),
            historyName,
            HistorySlot::Previous,
            ResourceState::General);
        if (!result) {
            return result;
        }

        outCurrentView = current.view;
        outPreviousView = previous.view;

        if (previous.valid && !resetAccumulation_) {
            ++accumulationFrame_;
            push.hasHistory = 1;
        } else {
            accumulationFrame_ = 0;
            push.hasHistory = 0;
        }
        resetAccumulation_ = false;
        push.accumulationFrame = accumulationFrame_;
        return {};
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

    static bool boolProperty(const RenderGraphProperties& properties, const char* key, bool fallback)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_boolean()) {
            return fallback;
        }
        return iter->get<bool>();
    }

    static std::string historyNameForContext(const RenderGraphExecutionContext& context)
    {
        std::string name(kScenePathTraceHistoryPrefix);
        name += context.passName();
        name += ".accumulation";
        return name;
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

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    static void buildPush(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const scene::Bounds& drawBounds,
        ScenePathTracePush& outPush)
    {
        outPush = ScenePathTracePush{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 50.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y + radius * 0.12f, center.z + defaultDistance);
        const float3 eye = cameraVec3(cameraProperties, "eye", defaultEye);
        const float3 target = cameraVec3(cameraProperties, "center", center);
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.001f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(cameraProperties, "zfar", defaultDistance + radius * 4.0f),
            zNear + 0.001f);
        const float cameraDistance = std::max(length(eye - target), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(
            cameraFloat(cameraProperties, "orthoHeight", defaultOrthoHeight),
            0.0001f);

        writeParamVec3(eye, outPush.eye, 0.0f);
        writeParamVec3(target, outPush.center, 0.0f);
        writeParamVec3(up, outPush.upProjection, cameraIsOrthographic(cameraProperties) ? 1.0f : 0.0f);
        outPush.viewport[0] = aspect;
        outPush.viewport[1] = static_cast<float>(width);
        outPush.viewport[2] = static_cast<float>(height);
        outPush.viewport[3] = fovRadians;
        outPush.clipOrtho[0] = zNear;
        outPush.clipOrtho[1] = zFar;
        outPush.clipOrtho[2] = orthoHeight;
        outPush.clipOrtho[3] = 0.0f;
        outPush.width = width;
        outPush.height = height;
        outPush.maxDepth = uintProperty(properties, "maxDepth", kDefaultPathTraceMaxDepth, 1, 12);
        outPush.samples = uintProperty(properties, "samples", kDefaultPathTraceSamples, 1, 16);
    }

    ScenePathTraceResources sceneResources_;
    SceneRayQueryProgram rayQueryProgram_;
    uint64_t sceneResourceRevision_ = 0;
    uint32_t accumulationFrame_ = 0;
    bool resetAccumulation_ = false;
};

} // namespace

std::unique_ptr<RenderGraphPass> createScenePathTracePass()
{
    return std::make_unique<ScenePathTracePass>();
}

} // namespace metallic::render::builtin_pass
