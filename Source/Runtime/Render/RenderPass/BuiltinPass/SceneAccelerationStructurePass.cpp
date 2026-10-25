#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "StreamLODSettings.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

class SceneAccelerationStructurePass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        const auto features = properties().value("rtTessellation", false)
            ? SceneResourceFeatureBits::AccelerationStructure | SceneResourceFeatureBits::Materials |
                SceneResourceFeatureBits::MaterialTextures
            : SceneResourceFeatureBits::AccelerationStructure;
        return {.features = features, .geometry = true};
    }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }
    QueueType queueType() const override
    {
        return properties().value("AsyncComputePreferred", true) ? QueueType::Compute : QueueType::Graphics;
    }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{{.key = "AsyncComputePreferred", .label = "Async Compute Preferred",
            .type = RenderGraphRuntimeSettingType::Bool, .defaultValue = true, .rebuildGraph = true}};
        appendStreamRayLODRuntimeSettings(settings);
        return settings;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureOutput("accelerationStructure", "Streamed CLAS scene TLAS")
            .buildReadWrite();
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.preparedScene || !context.preparedScene->geometry ||
            !context.preparedScene->geometry->ready()) {
            log = "SceneAccelerationStructurePass requires a prepared CLAS geometry stream.";
            return makeError(Error::Unsupported);
        }
        // The first traversal builds the TLAS, including an all-hidden scene.
        // Compilation only requires the initialized streaming resources.
        geometry_ = context.preparedScene->geometry;
        return {};
    }

    void describeSceneView(const RenderGraphExecutionContext& context, MeshletStreamFrameDesc& view) const override
    {
        view.width = context.width();
        view.height = context.height();
        view.displayHeight = context.displayHeight();
        const auto& bounds = geometry_->bounds();
        const auto& properties = context.properties();
        const auto camera = properties.value("camera", RenderGraphProperties::object());
        const float radius = std::max(bounds.radius(), 0.01f);
        const float aspect = float(std::max(view.width, 1u)) / float(std::max(view.height, 1u));
        const auto halfExtent = (bounds.max - bounds.min) * 0.5f;
        view.camera.fovDegrees = std::clamp(camera.value("fovDegrees", 60.0f), 1.0f, 179.0f);
        const float distance = std::max(std::max(halfExtent.y, halfExtent.x / aspect) /
            (0.72f * std::tan(view.camera.fovDegrees * 3.14159265359f / 360.0f)) + radius, 0.05f);
        const auto vector = [&](const char* name, float3 fallback) {
            const auto found = camera.find(name);
            if (found == camera.end() || !found->is_array() || found->size() < 3) { return fallback; }
            return float3((*found)[0].get<float>(), (*found)[1].get<float>(), (*found)[2].get<float>());
        };
        view.camera.center = vector("center", bounds.center());
        view.camera.eye = vector("eye", bounds.center() + float3(0, 0, distance));
        view.camera.up = vector("up", float3(0, 1, 0));
        view.camera.znear = camera.value("znear", 0.001f);
        view.camera.zfar = camera.value("zfar", distance + radius * 3.0f);
        const auto projection = camera.value("projection", std::string("perspective"));
        view.camera.orthographic = projection == "orthographic" || projection == "ortho";
        view.camera.orthoHeight = camera.value("orthoHeight", 2.0f * distance *
            std::tan(view.camera.fovDegrees * 3.14159265359f / 360.0f));
        view.lodPixelError = properties.value("lodPixelError", 1.5f);
        view.lodBias = properties.value("lodBias", 0.0f);
        applyStreamRayLODSettings(properties, view);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        if (!geometry_) { return makeError(Error::InvalidArgument); }
        if (geometry_->topLevelBuildPending()) {
            const auto result = geometry_->cmdBuildTopLevelAccelerationStructure(context.commandBuffer());
            if (!result) { return result; }
        }
        return context.publishAccelerationStructure("accelerationStructure", geometry_->accelerationStructure());
    }

private:
    std::shared_ptr<MeshletStreamRuntime> geometry_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSceneAccelerationStructurePass()
{
    return std::make_unique<SceneAccelerationStructurePass>();
}

} // namespace metallic::render::builtin_pass
