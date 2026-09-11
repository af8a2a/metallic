#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/ScreenSpaceShadowPassCommon.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/RenderFrameContext.h"

namespace metallic::render::builtin_pass {
namespace {

class RayTracedShadowPass final : public UnsafePass {
public:
    // SIGMA reads its previous output and the trace updates singleton uploads.
    bool supportsFrameOverlap() const override { return false; }

    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"depth", "rasterInfo"}};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("depth", "Hardware depth from the visibility raster")
            .sampledRead().format = Format::D32Sfloat;
        auto& info = reflection.addBufferInput("rasterInfo", "Matching raster camera and scene identity")
            .buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo)).shaderRead();
        info.memoryLocation = MemoryLocation::HostUpload;
        reflection.addTextureOutput("shadow", "SIGMA-encoded visibility for deferred direct lighting")
            .transferWrite().format = Format::R8Unorm;
        auto& parameters = reflection.addBufferOutput("parameters", "Selected light and shadow settings")
            .buffer(sizeof(ScreenSpaceShadowParameters), sizeof(ScreenSpaceShadowParameters)).shaderRead();
        parameters.memoryLocation = MemoryLocation::HostUpload;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return screenSpaceShadowRuntimeSettings(properties());
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) { return makeError(Error::InvalidArgument); }
        device_ = context.device;
        if (!device_->capabilities().rayQuery || !device_->capabilities().rayTracingAccelerationStructure) {
            log = "RayTracedShadowPass requires ray queries and a scene acceleration structure";
            return makeError(Error::Unsupported);
        }
        // Prepare the same shared geometry snapshot as Deferred, outside execute().
        {
            if (context.graphicsQueue == nullptr) { return makeError(Error::InvalidArgument); }
            auto* manager = context.sceneResourceManager != nullptr
                ? context.sceneResourceManager : &fallbackSceneResourceManager_;
            std::shared_ptr<SceneResourceSnapshot> snapshot;
            const auto result = manager->acquire(*device_, *context.graphicsQueue, properties(), context.runtimeScene,
                SceneResourceFeatureBits::Geometry | SceneResourceFeatureBits::Materials |
                SceneResourceFeatureBits::MaterialTextures | SceneResourceFeatureBits::StandardAccelerationStructure,
                snapshot, log);
            if (!result) { return result; }
            if (snapshot == nullptr || snapshot->pathTraceResources == nullptr) { return makeError(Error::Failure); }
            geometry_ = *snapshot->pathTraceResources;
        }
        history_ = std::make_shared<History>();
        shadows_.clear();
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        const auto depth = context.inputTexture("depth");
        const auto metadata = context.inputBuffer("rasterInfo");
        const auto output = context.outputTexture("shadow");
        const auto parameters = context.outputBuffer("parameters");
        const auto* scene = context.runtimeScene();
        if (device_ == nullptr || history_ == nullptr || context.streamer() == nullptr || scene == nullptr ||
            !depth.valid() || depth.view() == nullptr || !metadata.valid() || !output.valid() || !parameters.valid() ||
            metadata.desc().size != sizeof(VisibilityBufferFrameInfo) ||
            metadata.desc().memoryLocation != MemoryLocation::HostUpload ||
            parameters.desc().memoryLocation != MemoryLocation::HostUpload ||
            depth.desc().width != context.width() || depth.desc().height != context.height()) {
            return makeError(Error::InvalidArgument);
        }
        VisibilityBufferFrameInfo info;
        const void* mapped = metadata.buffer()->map();
        if (mapped == nullptr) { return makeError(Error::Failure); }
        std::memcpy(&info, mapped, sizeof(info));
        metadata.buffer()->unmap();
        if (info.width != context.width() || info.height != context.height() ||
            info.frameIndex != context.frameIndex() || info.sceneIdentity != scene->resourceIdentity()) {
            spdlog::error("RayTracedShadowPass requires depth and camera metadata from the current raster scene/view");
            return makeError(Error::InvalidArgument);
        }

        ViewConstants view{};
        const auto* sharedView = context.viewConstants();
        const bool previousValid = history_->valid && history_->sceneIdentity == info.sceneIdentity &&
            history_->view.current.viewport[1] == float(info.width) &&
            history_->view.current.viewport[2] == float(info.height) &&
            (sharedView == nullptr || (sharedView->frame[1] != 0 &&
                sharedView->frame[3] == history_->view.frame[3]));
        static_assert(offsetof(VisibilityBufferFrameInfo, width) == sizeof(ViewCameraConstants));
        std::memcpy(&view.current, info.eye, sizeof(ViewCameraConstants));
        // Raster eye.w and center.w carry jitter, not camera coordinates.
        view.current.eye[3] = view.current.center[3] = 0.0f;
        view.previous = previousValid ? history_->view.current : view.current;
        view.jitter[0] = info.jitter[0];
        view.jitter[1] = info.jitter[1];
        view.jitter[2] = previousValid ? history_->view.jitter[0] : info.jitter[0];
        view.jitter[3] = previousValid ? history_->view.jitter[1] : info.jitter[1];
        view.frame[0] = static_cast<uint32_t>(info.frameIndex);
        view.frame[1] = previousValid ? 1u : 0u;
        view.frame[2] = sharedView != nullptr ? sharedView->frame[2] : info.temporalJitter;
        view.frame[3] = sharedView != nullptr ? sharedView->frame[3] : 0u;
        auto& commands = context.commandBuffer();
        auto result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
            [] {}, [history = history_] { history->valid = false; }));
        if (!result) { return result; }
        const auto lights = buildScreenSpaceShadowLightRecords(scene, resolveSceneLighting(scene, context.world()));
        ScreenSpaceShadowResult shadow;
        std::string log;
        result = shadows_.record(*device_, commands, *context.streamer(), *depth.view(), view,
            lights, scene->contentRevision(), scene->transformRevision(),
            screenSpaceShadowSettings(context.properties()), shadow, log, &geometry_);
        if (!result) {
            spdlog::error("RayTracedShadowPass: {} ({})", log, resultToString(result));
            return result;
        }

        // Keep SIGMA's private output for the next history copy; publish a graph-owned snapshot.
        TextureBarrierDesc barrier{.texture = shadow.texture, .before = ResourceState::ShaderRead,
            .after = ResourceState::TransferSource, .mipCount = 1, .layerCount = 1};
        commands.barrier({.textures = &barrier, .textureCount = 1});
        commands.copyTexture({.source = shadow.texture, .destination = output.texture(),
            .width = context.width(), .height = context.height(), .depth = 1});
        barrier.before = ResourceState::TransferSource;
        barrier.after = ResourceState::ShaderRead;
        commands.barrier({.textures = &barrier, .textureCount = 1});
        mapped = shadow.parameters->map();
        if (mapped == nullptr) { return makeError(Error::Failure); }
        void* destination = parameters.buffer()->map();
        if (destination == nullptr) { shadow.parameters->unmap(); return makeError(Error::Failure); }
        std::memcpy(destination, mapped, sizeof(ScreenSpaceShadowParameters));
        shadow.parameters->unmap();
        parameters.buffer()->flush(0, sizeof(ScreenSpaceShadowParameters));
        parameters.buffer()->unmap();
        commands.hostWriteBarrier();
        history_->view = view;
        history_->sceneIdentity = info.sceneIdentity;
        history_->valid = true;
        return {};
    }

private:
    struct History {
        ViewConstants view;
        uint64_t sceneIdentity = 0;
        bool valid = false;
    };
    Device* device_ = nullptr;
    ScreenSpaceShadows shadows_;
    ScenePathTraceResources geometry_;
    SceneResourceManager fallbackSceneResourceManager_;
    std::shared_ptr<History> history_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createScreenSpaceShadowPass()
{
    return std::make_unique<RayTracedShadowPass>();
}

} // namespace metallic::render::builtin_pass
