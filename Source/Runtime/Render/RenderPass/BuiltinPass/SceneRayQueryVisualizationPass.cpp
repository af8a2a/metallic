#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class SceneRayQueryVisualizationPass final : public ComputePass {
public:
    ~SceneRayQueryVisualizationPass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "RayQuery acceleration-structure visualization")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
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
        if (rayQueryProgram_.valid() && rtxBuilder_.valid()) {
            return {};
        }

        scene::Scene loadedScene;
        const std::filesystem::path path = scenePathFromProperties(properties());
        if (!loadedScene.load(path)) {
            log = "SceneRayQueryVisualizationPass failed to load glTF: " + loadedScene.lastLoadResult().error;
            return makeError(Error::Failure);
        }
        if (!loadedScene.bounds().valid) {
            log = "SceneRayQueryVisualizationPass scene bounds are unavailable";
            return makeError(Error::Failure);
        }

        drawBounds_ = loadedScene.bounds();
        Result result = rtxBuilder_.build(*context.device, *context.graphicsQueue, loadedScene, log);
        if (!result) {
            return result;
        }

        ShaderCompileResult computeCompile;
        const char* capabilities[] = {"spvRayQueryKHR"};
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kSceneRayQueryVisualizationShaderModuleName,
                .entryPointName = kSceneRayQueryVisualizationEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .profileName = "glsl_460",
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            computeCompile);
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

        const SceneRayQueryBindingDesc bindings[] = {
            SceneRayQueryBindingDesc{
                .binding = 0,
                .kind = SceneRayQueryBindingKind::AccelerationStructure,
            },
            SceneRayQueryBindingDesc{
                .binding = 1,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
        };
        result = rayQueryProgram_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = computeCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(computeCompile.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(SceneRayQueryVisualizationPush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "SceneRayQueryVisualizationPass",
            },
            log);
        if (!result) {
            rayQueryProgram_.clear();
            return result;
        }

        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            color.view() == nullptr ||
            !rayQueryProgram_.valid() ||
            !rtxBuilder_.valid() ||
            !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        SceneRayQueryVisualizationPush push;
        buildPush(context.width(), context.height(), context.properties(), drawBounds_, push);

        const SceneRayQueryDispatchBinding bindings[] = {
            SceneRayQueryDispatchBinding{
                .binding = 0,
                .accelerationStructure = &rtxBuilder_,
            },
            SceneRayQueryDispatchBinding{
                .binding = 1,
                .textureView = color.view(),
            },
        };
        return rayQueryProgram_.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
            .groupCountZ = 1,
        });
    }

private:
    static std::filesystem::path scenePathFromProperties(const RenderGraphProperties& props)
    {
        if (props.contains("path") && props["path"].is_string()) {
            std::filesystem::path path = props["path"].get<std::string>();
            if (path.is_relative()) {
                path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
            }
            return path;
        }
        return kDefaultMaterialScenePath;
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
        return value == "primitive" || value == "per primitive" || value == "perPrimitive"
            ? kRayQueryVisualizationGranularityPrimitive
            : kRayQueryVisualizationGranularityInstance;
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
        outPush.clipOrtho[3] = 0.0f;
        outPush.mode = visualizationModeFromProperties(properties);
        outPush.width = width;
        outPush.height = height;
    }

    SceneRtxBuilder rtxBuilder_;
    SceneRayQueryProgram rayQueryProgram_;
    scene::Bounds drawBounds_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSceneRayQueryVisualizationPass()
{
    return std::make_unique<SceneRayQueryVisualizationPass>();
}

} // namespace metallic::render::builtin_pass
