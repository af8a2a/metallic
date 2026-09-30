#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"

namespace metallic::render::builtin_pass {
namespace {

class SceneAccelerationStructurePass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::StandardAccelerationStructure,
            .graphManagedAccelerationStructure = true};
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
        return {{.key = "AsyncComputePreferred", .label = "Async Compute Preferred",
            .type = RenderGraphRuntimeSettingType::Bool, .defaultValue = true, .rebuildGraph = true}};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureOutput("accelerationStructure", "Scene TLAS/PTLAS with graph-managed refits")
            .buildReadWrite();
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.device || !context.preparedScene || !context.preparedScene->snapshot ||
            !context.preparedScene->snapshot->pathTraceResources) {
            log = "SceneAccelerationStructurePass requires prepared scene acceleration structures.";
            return makeError(Error::InvalidArgument);
        }
        sceneResources_ = *context.preparedScene->snapshot->pathTraceResources;
        device_ = context.device;
        if (!sceneResources_.accelerationStructure().valid()) {
            log = "SceneAccelerationStructurePass scene acceleration structures are unavailable.";
            return makeError(Error::Unsupported);
        }
        return {};
    }

    Result<> prepareExecution(RenderGraphExecutionContext& context) override
    {
        if (!device_ || !context.runtimeScene()) {
            return makeError(Error::InvalidArgument);
        }
        return sceneResources_.accelerationStructure().prepareInstanceTransformUpdate(
            *device_, *context.runtimeScene(), log_);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto& builder = sceneResources_.accelerationStructure();
        Result<> result = context.commandBuffer().retainResource(std::make_shared<ScenePathTraceResources>(sceneResources_));
        if (!result) { return result; }
        result = context.publishAccelerationStructure("accelerationStructure", builder.accelerationStructure());
        if (!result || !builder.hasPendingInstanceTransformUpdate()) {
            return result;
        }
        Buffer* instances = builder.instanceTransformUpdateBuffer();
        Buffer* scratch = builder.instanceTransformUpdateScratchBuffer();
        if (!instances || !scratch) {
            return makeError(Error::InvalidArgument);
        }
        auto instanceSlice = instances->slice();
        auto scratchSlice = scratch->slice();
        if (!instanceSlice || !scratchSlice) {
            return makeError(Error::InvalidArgument);
        }
        using Access = RenderGraphResourceAccess;
        const RenderGraphStageUse buildUses[] = {
            {"accelerationStructure", Access::AccelerationStructureBuildReadWrite},
            {"instances", Access::BufferAccelerationStructureBuildRead},
            {"scratch", Access::BufferAccelerationStructureScratchReadWrite},
        };
        const RenderGraphStage stages[] = {{"Scene TLAS/PTLAS refit", buildUses,
            [&](CommandBuffer& commands) { return builder.recordInstanceTransformUpdate(commands, log_); }}};
        const RenderGraphBufferImport buffers[] = {
            {"instances", *instanceSlice, Access::BufferAccelerationStructureBuildRead},
            {"scratch", *scratchSlice, Access::BufferAccelerationStructureScratchReadWrite},
        };
        return context.executeStages(stages, buffers);
    }

private:
    ScenePathTraceResources sceneResources_;
    Device* device_ = nullptr;
    std::string log_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSceneAccelerationStructurePass()
{
    return std::make_unique<SceneAccelerationStructurePass>();
}

} // namespace metallic::render::builtin_pass
