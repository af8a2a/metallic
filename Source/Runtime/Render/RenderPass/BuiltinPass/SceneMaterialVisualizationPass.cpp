#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"

#ifndef METALLIC_HAS_NTC
#define METALLIC_HAS_NTC 0
#endif

#ifndef METALLIC_NTC_SHADER_INCLUDE_DIR
#define METALLIC_NTC_SHADER_INCLUDE_DIR ""
#endif

namespace metallic::render::builtin_pass {
namespace {

class SceneMaterialVisualizationPass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::Geometry | SceneResourceFeatureBits::Materials |
            SceneResourceFeatureBits::MaterialTextures | SceneResourceFeatureBits::StandardAccelerationStructure};
    }

    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }

    bool supportsFrameOverlap() const override { return true; }

    ~SceneMaterialVisualizationPass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureInput("accelerationStructure", "Optional graph-managed scene TLAS/PTLAS")
            .accelerationStructureRead().setOptional();
        reflection.addTextureOutput("color", "glTF material diagnostic visualization")
            .storageWrite()
            .transient(RenderGraphInitialization::FullOverwrite)
            .format = Format::RGBA8Unorm;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeEnumSetting(
                "mode",
                "Mode",
                "material",
                {
                    {"Material", "material"},
                    {"Base Color", "baseColor"},
                    {"World Normal RGB", "normal"},
                    {"Roughness", "roughness"},
                    {"Metallic", "metallic"},
                    {"AO", "ao"},
                    {"Geometry Normal RGB", "geometryNormal"},
                    {"Vertex Normal RGB", "vertexNormal"},
                    {"Normal Texture RGB", "normalTexture"},
                    {"Tangent RGB", "tangent"},
                    {"Bitangent RGB", "bitangent"},
                    {"NRD Packed Normal", "nrdNormalRoughness"},
                    {"Normal Deviation", "normalDeviation"},
                }),
            runtimeBoolSetting("flipBitangent", "Flip Bitangent", false),
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 0.42f, 1.15f},
            std::array<float, 3>{0.0f, 0.075f, 0.0f},
            45.0f);
        return settings;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "SceneMaterialVisualizationPass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().rayTracingAccelerationStructure ||
            !context.device->capabilities().rayQuery) {
            log = "SceneMaterialVisualizationPass requires rayTracingAccelerationStructure and rayQuery capabilities";
            return makeError(Error::Unsupported);
        }
        device_ = context.device;
        graphicsQueue_ = context.graphicsQueue;
        if (!context.preparedScene || !context.preparedScene->snapshot ||
            !context.preparedScene->snapshot->pathTraceResources) {
            log = "Scene resources were not prepared by StreamerSubsystem";
            return makeError(Error::InvalidArgument);
        }
        sceneResources_ = *context.preparedScene->snapshot->pathTraceResources;
        Result<> result;
        const bool ntcActive = sceneResources_.neuralTextures().active();
        const bool positionFetch = context.device->capabilities().rayTracingPositionFetch;
        const bool ntcCooperativeVector =
            sceneResources_.neuralTextures().cooperativeVectorActive();
        if (rayQueryProgram_.valid() &&
            compiledTextureCount_ == sceneResources_.materialTextureCount() &&
            compiledPositionFetch_ == positionFetch &&
            compiledNtcActive_ == ntcActive &&
            compiledNtcCooperativeVector_ == ntcCooperativeVector) {
            return {};
        }
        rayQueryProgram_.clear();
        rayQueryProgramEncoder_.clear();

        ShaderCompileResult computeCompile;
        SceneRayQueryOptions shaderOptions{
            .positionFetch = positionFetch, .hasNTC = ntcActive, .cooperativeVector = ntcCooperativeVector,
        };
#if METALLIC_HAS_NTC
        shaderOptions.ntcInclude = METALLIC_NTC_SHADER_INCLUDE_DIR;
#endif
        const auto request = makeSceneRayQueryRequest(SceneRayQueryProgram::MaterialVisualization, shaderOptions);
        const ShaderRequestView source(request);
        result = ShaderRegistry::instance().getShader(source.desc(), computeCompile.diagnostics)
            .transform([&](auto value) { computeCompile = std::move(value); });
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += kSceneMaterialVisualizationShaderModuleName;
            log += ".";
            log += kSceneMaterialVisualizationEntryPoint;
            log += ") returned ";
            log += resultToString(result);
            if (!computeCompile.diagnostics.empty()) {
                log += ": ";
                log += computeCompile.diagnostics;
            }
            log += '\n';
            rayQueryProgram_.clear();
            rayQueryProgramEncoder_.clear();
            return result;
        }

        // Keep the conventional binding table stable; append NTC descriptors only when active.
        std::vector<ComputeResourceBindingDesc> bindings{
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene),
                .kind = ComputeResourceBindingKind::AccelerationStructure,
            },
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, output),
                .kind = ComputeResourceBindingKind::StorageImage,
            },
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, vertices),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, indices),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, primitives),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, instances),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            },
            ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures),
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = sceneResources_.materialTextureCount(),
            },
        };
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues), .kind = ComputeResourceBindingKind::StorageBuffer});
        if (!positionFetch) {
            bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, positions), .kind = ComputeResourceBindingKind::StorageBuffer});
        }
        if (ntcActive) {
            bindings.push_back(ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcLatents),
                .kind = ComputeResourceBindingKind::SampledImage,
                .descriptorCount = kMaxNeuralTextureSets,
            });
            bindings.push_back(ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcConstants),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            bindings.push_back(ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcWeights),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            bindings.push_back(ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcInfo),
                .kind = ComputeResourceBindingKind::StorageBuffer,
            });
            bindings.push_back(ComputeResourceBindingDesc{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcSampler),
                .kind = ComputeResourceBindingKind::Sampler,
            });
        }
        std::string programLog;
        result = initializeResourceKernel(
            *context.device,
            ResourceComputeKernelDesc{
                .spirv = computeCompile.spirv,
                .pushConstantSize = sizeof(SceneMaterialVisualizationPush),
                .bindings = bindings,
                .debugName = "SceneMaterialVisualizationPass",
                .resourceParameters = resourceParameterLayout<SceneResourceParameters>(),
            }, rayQueryProgram_, rayQueryProgramEncoder_,
            programLog);
        if (!programLog.empty()) {
            if (!log.empty() && log.back() != '\n') {
                log += '\n';
            }
            log += programLog;
        }
        if (!result) {
            rayQueryProgram_.clear();
            rayQueryProgramEncoder_.clear();
            return result;
        }
        compiledNtcActive_ = ntcActive;
        compiledTextureCount_ = sceneResources_.materialTextureCount();
        compiledPositionFetch_ = positionFetch;
        compiledNtcCooperativeVector_ = ntcCooperativeVector;
        return {};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        std::string syncLog;
        TextureHandle color = context.outputTexture("color");
        const auto& materialTextureViews = sceneResources_.materialTextureViews();
        if (!color.valid() ||
            color.view() == nullptr ||
            !rayQueryProgram_.valid() ||
            !sceneResources_.valid() ||
            materialTextureViews[0] == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        SceneMaterialVisualizationPush push;
        buildPush(context.width(), context.height(), context.properties(), sceneResources_.bounds(), push);
        push.materialTextureCount = sceneResources_.materialTextureCount();
        push.ntcTextureSetCount = sceneResources_.neuralTextures().textureSetCount();

        Result<> result;

        std::vector<ComputeDispatchBinding> bindings{
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene),
                .accelerationStructure =
                    context.inputAccelerationStructure("accelerationStructure")
                        ? context.inputAccelerationStructure("accelerationStructure")
                        : sceneResources_.accelerationStructure().accelerationStructure(),
            },
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, output),
                .textureView = color.view(),
            },
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, vertices),
                .buffer = sceneResources_.shadingVertexBuffer(),
            },
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, indices),
                .buffer = sceneResources_.indexBuffer(),
            },
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, primitives),
                .buffer = sceneResources_.primitiveBuffer(),
            },
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, instances),
                .buffer = sceneResources_.instanceBuffer(),
            },
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials),
                .buffer = sceneResources_.materialBuffer(),
            },
            ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures),
                .textureViews = materialTextureViews,
            },
        };
        const NeuralTextureResources& neuralTextures = sceneResources_.neuralTextures();
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues), .buffer = sceneResources_.materialBinding()->valueBuffer()
            ? sceneResources_.materialBinding()->valueBuffer() : sceneResources_.materialBuffer()});
        if (sceneResources_.fallbackPositionBuffer() != nullptr) {
            bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, positions), .buffer = sceneResources_.fallbackPositionBuffer()});
        }
        if (neuralTextures.active()) {
            const auto& latentViews = neuralTextures.latentTextureViews();
            bindings.push_back(ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcLatents),
                .textureViews = latentViews,
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcConstants),
                .buffer = neuralTextures.constantsBuffer(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcWeights),
                .buffer = neuralTextures.weightsBuffer(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcInfo),
                .buffer = neuralTextures.setInfoBuffer(),
            });
            bindings.push_back(ComputeDispatchBinding{
                .binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, ntcSampler),
                .sampler = &neuralTextures.latentSampler(),
            });
        }
        result = dispatchResources(rayQueryProgram_, rayQueryProgramEncoder_, ComputeDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
            .groupCountZ = 1,
        });
        return result;
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
            return kSceneMaterialVisualizationModeMaterial;
        }
        auto iter = properties.find("mode");
        if (iter == properties.end() || !iter->is_string()) {
            return kSceneMaterialVisualizationModeMaterial;
        }
        const std::string mode = iter->get<std::string>();
        if (mode == "baseColor" || mode == "basecolor" || mode == "base color") {
            return kSceneMaterialVisualizationModeBaseColor;
        }
        if (mode == "normal") {
            return kSceneMaterialVisualizationModeNormal;
        }
        if (mode == "roughness") {
            return kSceneMaterialVisualizationModeRoughness;
        }
        if (mode == "metallic") {
            return kSceneMaterialVisualizationModeMetallic;
        }
        if (mode == "ao" || mode == "AO" || mode == "occlusion") {
            return kSceneMaterialVisualizationModeAo;
        }
        if (mode == "geometryNormal" || mode == "geometry normal" || mode == "geometricNormal") {
            return kSceneMaterialVisualizationModeGeometryNormal;
        }
        if (mode == "vertexNormal" || mode == "vertex normal" || mode == "shadingNormal" || mode == "shading normal") {
            return kSceneMaterialVisualizationModeVertexNormal;
        }
        if (mode == "normalTexture" || mode == "normal texture" || mode == "normalMap" || mode == "normal map") {
            return kSceneMaterialVisualizationModeNormalTexture;
        }
        if (mode == "tangent") {
            return kSceneMaterialVisualizationModeTangent;
        }
        if (mode == "bitangent") {
            return kSceneMaterialVisualizationModeBitangent;
        }
        if (
            mode == "nrdNormalRoughness" ||
            mode == "nrd normal roughness" ||
            mode == "normalRoughness" ||
            mode == "normal roughness") {
            return kSceneMaterialVisualizationModeNRDNormalRoughness;
        }
        if (mode == "normalDeviation" || mode == "normal deviation") {
            return kSceneMaterialVisualizationModeNormalDeviation;
        }
        return kSceneMaterialVisualizationModeMaterial;
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
        SceneMaterialVisualizationPush& outPush)
    {
        outPush = SceneMaterialVisualizationPush{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 45.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y + radius * 0.45f, center.z + defaultDistance);
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
        outPush.clipOrtho[3] = 1.0f;
        outPush.width = width;
        outPush.height = height;
        outPush.mode = visualizationModeFromProperties(properties);
        outPush.bitangentFlip = metallic::render::builtin_pass::boolProperty(&properties, "flipBitangent", false)
            ? -1.0f
            : 1.0f;
    }

    ScenePathTraceResources sceneResources_;
    Device* device_ = nullptr;
    Queue* graphicsQueue_ = nullptr;
    ComputeKernel rayQueryProgram_;
    ComputeResourceEncoder rayQueryProgramEncoder_;
    bool compiledNtcActive_ = false;
    uint32_t compiledTextureCount_ = 0;
    bool compiledPositionFetch_ = false;
    bool compiledNtcCooperativeVector_ = false;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSceneMaterialVisualizationPass()
{
    return std::make_unique<SceneMaterialVisualizationPass>();
}

} // namespace metallic::render::builtin_pass
