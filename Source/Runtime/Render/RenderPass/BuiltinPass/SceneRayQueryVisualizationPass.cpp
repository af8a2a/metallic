#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/NamedComputeParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/StreamRayTracingBindings.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kNamedComputeABI = 0x4e43500000000002ull;

class SceneRayQueryVisualizationPass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::AccelerationStructure, .geometry = true};
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }

    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }

    bool supportsFrameOverlap() const override { return true; }

    ~SceneRayQueryVisualizationPass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureInput("accelerationStructure", "Optional graph-managed scene TLAS/PTLAS")
            .accelerationStructureRead().setOptional();
        reflection.addTextureOutput("color", "RayQuery acceleration-structure visualization")
            .storageWrite()
            .format = Format::RGBA8Unorm;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeEnumSetting(
                "granularity",
                "Granularity",
                "instance",
                {{"Instance", "instance"}, {"Primitive", "primitive"}, {"Cluster ID", "cluster-id"}}),
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 0.0f, 3.0f},
            std::array<float, 3>{0.0f, 0.0f, 0.0f},
            60.0f);
        return settings;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "SceneRayQueryVisualizationPass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().rayTracingAccelerationStructure ||
            !context.device->capabilities().rayQuery) {
            log = "SceneRayQueryVisualizationPass requires rayTracingAccelerationStructure and rayQuery capabilities";
            return makeError(Error::Unsupported);
        }
        if (!context.device->capabilities().clusterAccelerationStructure) {
            log = "SceneRayQueryVisualizationPass requires CLAS support";
            return makeError(Error::Unsupported);
        }
        if (!context.preparedScene || !context.preparedScene->geometry) {
            log = "Scene geometry was not prepared by StreamerSubsystem";
            return makeError(Error::InvalidArgument);
        }
        device_ = context.device;
        geometry_ = context.preparedScene->geometry;
        drawBounds_ = geometry_->bounds();
        if (rayQueryProgram_.valid()) { return {}; }
        Result<> result;
        ShaderCompileResult computeCompile;
        const char* clusterIdCapabilities[] = {
            "spvRayQueryKHR",
            "SPV_NV_cluster_acceleration_structure",
            "spvRayTracingClusterAccelerationStructureNV",
        };
        result = ShaderRegistry::instance().getShader(SlangShaderDesc{
            .moduleName = kSceneRayQueryVisualizationShaderModuleName,
            .entryPointName = kSceneRayQueryVisualizationEntryPoint,
            .searchPath = kTriangleShaderSearchPath,
            .capabilities = clusterIdCapabilities,
        }, computeCompile.diagnostics).transform([&](auto value) { computeCompile = std::move(value); });
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += kSceneRayQueryVisualizationShaderModuleName;
            log += ".";
            log += kSceneRayQueryVisualizationEntryPoint;
            log += ") returned ";
            log += resultToString(result);
            if (!computeCompile.diagnostics.empty()) {
                log += ": ";
                log += computeCompile.diagnostics;
            }
            log += '\n';
            return result;
        }

        result = rayQueryProgram_.initialize(
            *context.device,
            ComputeKernelDesc{
                .spirv = computeCompile.spirv,
                .parameters = parameterAbi<NamedComputeParameters>(kNamedComputeABI),
                .debugName = "SceneRayQueryVisualizationPass",
            },
            log);
        if (!result) {
            rayQueryProgram_.clear();
            return result;
        }
        return {};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            color.view() == nullptr ||
            !rayQueryProgram_.valid() ||
            !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }
        SceneRayQueryVisualizationPush push;
        buildPush(context.width(), context.height(), context.properties(), geometry_->bounds(), push);
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(context.commandBuffer()));
        SceneResourceParameters resources{};
        resources.scene = writer.accelerationStructure(context.inputAccelerationStructure("accelerationStructure")
            ? context.inputAccelerationStructure("accelerationStructure") : geometry_->accelerationStructure()).value;
        resources.output = writer.storageImageHandle(color.view());
        auto encoded = encodeNamedParameters(writer, resources, push, kNamedComputeABI);
        if (!encoded) { return makeError(encoded.error()); }
        return rayQueryProgram_.dispatch(context.commandBuffer(), *encoded,
            (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
    }

    void describeSceneView(const RenderGraphExecutionContext& context, MeshletStreamFrameDesc& view) const override
    {
        SceneRayQueryVisualizationPush push;
        buildPush(context.width(), context.height(), context.properties(), geometry_->bounds(), push);
        describeStreamRayView(push, view);
        view.displayHeight = context.displayHeight();
        view.lodPixelError = context.properties().value("lodPixelError", 1.5f);
        view.lodBias = context.properties().value("lodBias", 0.0f);
    }

private:
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

    static uint32_t visualizationModeFromProperties(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return kRayQueryVisualizationGranularityInstance;
        }
        auto iter = properties.find("granularity");
        if (iter == properties.end() || !iter->is_string()) {
            return kRayQueryVisualizationGranularityInstance;
        }
        const std::string value = iter->get<std::string>();
        if (value == "primitive" || value == "per primitive" || value == "perPrimitive") {
            return kRayQueryVisualizationGranularityPrimitive;
        }
        if (value == "cluster" ||
            value == "cluster-id" ||
            value == "clusterId" ||
            value == "cluster id" ||
            value == "ClusterIDNV") {
            return kRayQueryVisualizationGranularityClusterId;
        }
        return kRayQueryVisualizationGranularityInstance;
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
        SceneRayQueryVisualizationPush& outPush)
    {
        outPush = SceneRayQueryVisualizationPush{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 60.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y, center.z + defaultDistance);
        const float3 eye = cameraVec3(cameraProperties, "eye", defaultEye);
        const float3 target = cameraVec3(cameraProperties, "center", center);
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.001f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(cameraProperties, "zfar", defaultDistance + radius * 3.0f),
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
        outPush.clipOrtho[3] = 1.0f;
        outPush.mode = visualizationModeFromProperties(properties);
        outPush.width = width;
        outPush.height = height;
    }

    std::shared_ptr<MeshletStreamRuntime> geometry_;
    ComputeKernel rayQueryProgram_;
    scene::Bounds drawBounds_;
    Device* device_ = nullptr;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSceneRayQueryVisualizationPass()
{
    return std::make_unique<SceneRayQueryVisualizationPass>();
}

} // namespace metallic::render::builtin_pass
