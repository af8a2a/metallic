#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kMeshDistanceFieldDebugABI = 0x4d44464400000001ull;

struct alignas(16) MeshDistanceFieldDebugParams {
    ShaderBuffer instances, mips, indirection;
    ShaderSampledImage atlas;
    ShaderSampler atlasSampler;
    ShaderStorageImage color, trace;
    uint32_t instanceCount;
    std::array<std::array<float, 4>, 4> clipToWorld;
    std::array<float, 4> cameraEyeAndOrthographic;
    uint32_t bricksPerAxis;
    float minTraceDistance, maxTraceDistance, nearClipZ, farClipZ;
    uint32_t debugMode, padding0, padding1;
};

static_assert(sizeof(MeshDistanceFieldDebugParams) == 144);
static_assert(offsetof(MeshDistanceFieldDebugParams, clipToWorld) == 32);
static_assert(offsetof(MeshDistanceFieldDebugParams, farClipZ) == 128);

void setCamera(const ViewCameraConstants& camera, MeshDistanceFieldDebugParams& params)
{
    const float3 eye(camera.eye[0], camera.eye[1], camera.eye[2]);
    const float3 center(camera.center[0], camera.center[1], camera.center[2]);
    const float3 up(camera.upProjection[0], camera.upProjection[1], camera.upProjection[2]);
    const float3 forward = (center - eye) / std::sqrt(dot(center - eye, center - eye));
    const float3 cameraRight = cross(forward, up);
    const float3 right = cameraRight / std::sqrt(dot(cameraRight, cameraRight));
    const bool orthographic = camera.upProjection[3] > 0.5f;
    const float nearPlane = camera.clipOrtho[0];
    const float farPlane = camera.clipOrtho[1];
    const float halfHeight = orthographic ? camera.clipOrtho[2] * 0.5f : nearPlane * std::tan(camera.viewport[3] * 0.5f);
    const float3 horizontal = right * (halfHeight * camera.viewport[0]);
    const float3 vertical = cross(right, forward) * halfHeight;
    // Perspective unprojects only the near plane. Orthographic rays use both
    // affine endpoints, with depth set to 0 and 1 by the debug pass.
    const float3 depth = forward * (orthographic ? farPlane - nearPlane : 0.0f);
    const float3 nearCenter = eye + forward * nearPlane;
    params.clipToWorld = {{{horizontal.x, vertical.x, depth.x, nearCenter.x},
        {horizontal.y, vertical.y, depth.y, nearCenter.y},
        {horizontal.z, vertical.z, depth.z, nearCenter.z}, {0.0f, 0.0f, 0.0f, 1.0f}}};
    params.cameraEyeAndOrthographic = {eye.x, eye.y, eye.z, orthographic ? 1.0f : 0.0f};
    params.nearClipZ = 0.0f;
    params.farClipZ = 1.0f;
    params.minTraceDistance = orthographic ? 0.0f : nearPlane;
    params.maxTraceDistance = orthographic ? farPlane - nearPlane : farPlane;
}

class MeshDistanceFieldDebugPass final : public ComputePass {
public:
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Mesh distance field diagnostic color")
            .storageWrite().format = Format::RGBA16Sfloat;
        reflection.addTextureOutput("trace", "Hit distance in meters, step count, instance ID, hit flag")
            .storageWrite().format = Format::RGBA32Sfloat;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {runtimeEnumSetting("debugMode", "Display", "normals",
            {{"Normals", "normals"}, {"Trace Steps", "steps"}, {"Instances", "instances"}})};
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.device || !context.graphicsQueue || !context.runtimeScene) {
            log = "Mesh Distance Field Debug requires a world scene and a graphics queue";
            return makeError(Error::InvalidArgument);
        }
        const std::array paths{std::filesystem::path(PROJECT_SOURCE_DIR) /
            properties().value("distanceFieldPath", std::string{})};
        device_ = context.device;

        auto result = ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/MeshDistanceFieldDebug",
                .entryPointName = "meshDistanceFieldDebugMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<MeshDistanceFieldDebugParams>(kMeshDistanceFieldDebugABI),
                .debugName = "MeshDistanceFieldDebug"}, program_, log);
        if (result) {
            result = meshFields_.initialize(*device_, *context.graphicsQueue, *context.runtimeScene, paths, log);
            if (!result && log.empty()) {
                log = "Mesh distance field GPU upload failed: " + std::string(resultToString(result));
            }
        }
        return result;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        MeshDistanceFieldDebugParams params{};
        uint32_t unsupportedInstances = 0;
        auto instanceBuffer = meshFields_.uploadInstances(*context.runtimeScene(), params.instanceCount, unsupportedInstances);
        if (!instanceBuffer) { return makeError(instanceBuffer.error()); }
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        params.instances = writer.buffer(instanceBuffer->get());
        const auto meshBindings = meshFields_.bind(writer);
        params.mips = meshBindings.mips;
        params.indirection = meshBindings.indirection;
        params.atlas = meshBindings.atlas;
        params.atlasSampler = meshBindings.atlasSampler;
        params.color = writer.storageImageHandle(context.outputTexture("color").view());
        params.trace = writer.storageImageHandle(context.outputTexture("trace").view());
        params.bricksPerAxis = meshFields_.bricksPerAxis();
        const auto mode = context.properties().value("debugMode", std::string("normals"));
        params.debugMode = mode == "instances" ? 2u : (mode == "steps" ? 1u : 0u);
        if (const auto* view = context.viewConstants()) {
            setCamera(view->current, params);
        } else {
            setCamera(RenderView{}.constants(0, context.width(), context.height(), context.width(), context.height()).current,
                params);
        }

        auto encoded = writer.encode(params, kMeshDistanceFieldDebugABI);
        if (!encoded) { return makeError(encoded.error()); }
        auto result = program_.dispatch(commands, *encoded,
            (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
        if (result) {
            const DebugResourceBinding atlas{.id = "atlas", .texture = &meshFields_.atlas(), .state = ResourceState::ShaderRead};
            context.debugCheckpoint("MeshDistanceField", {&atlas, 1}, {
                {"assetCount", 1}, {"instanceCount", params.instanceCount},
                {"unsupportedTransforms", unsupportedInstances}, {"atlasBricksPerAxis", params.bricksPerAxis}});
        }
        return result;
    }

    std::vector<std::string> debugCheckpoints() const override { return {"AfterPass", "MeshDistanceField"}; }

private:
    Device* device_ = nullptr;
    ComputeKernel program_;
    MeshDistanceFieldGPU meshFields_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createMeshDistanceFieldDebugPass()
{
    return std::make_unique<MeshDistanceFieldDebugPass>();
}

} // namespace metallic::render::builtin_pass
