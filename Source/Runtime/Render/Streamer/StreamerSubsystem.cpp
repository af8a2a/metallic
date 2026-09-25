#include "Runtime/Render/Streamer/StreamerSubsystem.h"

#include <algorithm>
#include <numeric>
#include "Runtime/Render/RayTracing/SceneAccelerationStructureExtensions.h"
#include "Runtime/Render/Streamer/StreamedImage.h"
#include "Runtime/Render/Streamer/SceneStreamingConfig.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Profiling/CpuProfile.h"

namespace metallic::render {
struct SceneStreamingState {
    MeshletStreamRuntimeDesc desc;
    SceneStreamKind kind = SceneStreamKind::None;
    uint64_t sceneIdentity = 0;
    uint64_t structuralRevision = 0;
    bool debugReadback = false;
};

Result StreamerSubsystem::initialize(const RenderSubsystemInitContext& context, std::string& log)
{
    device_ = &context.device;
    return uploads_.initialize(context.device, log, context.host.frameSlotCount());
}

Result StreamerSubsystem::beginFrame(const RenderSubsystemFrameContext& context, RenderChangeBits&, std::string&)
{
    collectReleasedStreams();
    textureFrames_.clear();
    if (context.frameResources) { return uploads_.beginFrame(*context.frameResources); }
    uploads_.beginFrame();
    return {};
}

void StreamerSubsystem::endFrame(const RenderSubsystemFrameContext&)
{
    uploads_.endFrame();
}

void StreamerSubsystem::prepareBeforePacing(CpuProfileRecorder* profiler)
{
    for (const auto& stream : streams_) {
        // A released view must not advance residency or launch more I/O.
        if (stream.use_count() > 1) { stream->prepareMaintenance(profiler); }
    }
}

void StreamerSubsystem::shutdown()
{
    // The host waits for submitted work and retires graph passes before this.
    streams_.clear();
    resources_.clear();
    uploads_.reset();
    device_ = nullptr;
}

Result StreamerSubsystem::prepareScene(const SceneStreamingRequirements& requirements,
    const RenderGraphProperties& properties, const scene::Scene* scene,
    std::shared_ptr<PreparedSceneResources>& prepared, std::string& log, bool debugReadback)
{
    if (!device_) { return makeError(Error::InvalidArgument); }
    if (requirements.features == SceneResourceFeatureBits::None && requirements.geometry == SceneStreamKind::None && !requirements.sampledImage) {
        prepared.reset();
        return {};
    }
    if (!prepared) { prepared = std::make_shared<PreparedSceneResources>(); }
    if (requirements.sampledImage) {
        const auto path = normalizedScenePath(properties.value("path", std::string(PROJECT_SOURCE_DIR "/Asset/statue-1275469_1280.jpg")));
        if (!prepared->image || prepared->imagePath != path) {
            auto image = std::make_shared<StreamedImage>();
            const auto result = image->prepare(*device_, path.string(), log);
            if (!result) { return result; }
            prepared->image = std::move(image);
            prepared->imagePath = path;
            prepared->imageView = prepared->image->imageView_.get();
        }
    }
    if (requirements.features != SceneResourceFeatureBits::None) {
        const auto result = resources_.acquire(*device_, *device_->getQueue(QueueType::Graphics),
            properties, scene, requirements.features, prepared->snapshot, log);
        if (!result) { return result; }
    }
    if ((uint32_t(requirements.features) & uint32_t(SceneResourceFeatureBits::ClusterAccelerationStructure)) != 0) {
        if (!scene) { return makeError(Error::InvalidArgument); }
        const std::array stamp{scene->resourceIdentity(), scene->sceneGraph().structuralRevision(),
            scene->transformRevision(), scene->visibilityRevision()};
        if (!prepared->clusterAccelerationStructure || prepared->clusterRevision != stamp) {
            auto cluster = std::make_shared<SceneClusterAccelerationStructureBuilder>();
            auto* queue = device_->getQueue(QueueType::Compute);
            if (!queue) { queue = device_->getQueue(QueueType::Graphics); }
            std::string clusterLog;
            const auto result = cluster->build(*device_, *queue, *scene, clusterLog);
            if (!result && !requirements.optionalClusterAccelerationStructure) { log = clusterLog; return result; }
            if (!result) { cluster->clear(); }
            prepared->clusterAccelerationStructure = std::move(cluster);
            prepared->clusterRevision = stamp;
        }
    }
    using namespace streaming_detail;
    const bool wantsGeometry = requirements.geometry == SceneStreamKind::Asset ||
        (requirements.geometry == SceneStreamKind::Visibility && previewStreamEnabled(properties));
    if (!wantsGeometry) { prepared->geometry.reset(); prepared->state.reset(); return {}; }
    PreviewStreamSourceSelection source;
    MeshletStreamRuntimeDesc desc;
    if (requirements.geometry == SceneStreamKind::Visibility) {
        if (!resolvePreviewStreamSource(properties, scene,
                normalizedScenePath(properties.value("path", std::string{})), source, log)) {
            return makeError(Error::InvalidArgument);
        }
        desc = previewStreamRuntimeDesc(properties, source.sourcePath, uint32_t(scene->textures().size()) + 1);
    } else {
        desc = assetStreamRuntimeDesc(properties);
        source.sourcePath = desc.sourcePath;
    }
    const uint64_t identity = scene ? scene->resourceIdentity() : 0;
    const uint64_t structural = scene ? scene->sceneGraph().structuralRevision() : 0;
    if (!prepared->state || prepared->state->desc != desc || prepared->state->sceneIdentity != identity ||
        prepared->state->structuralRevision != structural || prepared->state->debugReadback != debugReadback || prepared->streamSourceId != source.sourceId) {
        std::unique_ptr<PipelineCache> cache;
        auto result = device_->createPipelineCache({.filePath = PROJECT_SOURCE_DIR "/.cache/pso/SceneStreaming.pso"}, cache);
        if (!result) { return result; }
        std::shared_ptr<MeshletStreamRuntime> stream;
        result = acquireStream(desc, debugReadback, stream, log, cache.get());
        if (!result) { return result; }
        prepared->geometry = std::move(stream);
        prepared->state = std::make_shared<SceneStreamingState>(SceneStreamingState{desc, requirements.geometry, identity, structural, debugReadback});
        prepared->streamSourceId = source.sourceId;
        prepared->streamSourcePath = source.sourcePath;
        prepared->streamAssetPath = desc.streamAssetPath;
        if (cache) { (void)cache->save(); }
    }
    return {};
}

Result StreamerSubsystem::recordSceneBegin(PreparedSceneResources& prepared,
    const SceneStreamingRequirements& requirements, RenderGraphExecutionContext& context, const MeshletStreamFrameDesc& view, std::string& log)
{
    prepared.ready = !prepared.snapshot || !prepared.snapshot->pathTraceResources ||
        prepared.snapshot->pathTraceResources->textureUploadsReady();
    if (!prepared.ready) { return {}; }
    if (prepared.image) {
        const auto result = prepared.image->upload(context.commandBuffer());
        if (!result) { return result; }
        if (auto* frame = context.commandBuffer().frameContext()) { frame->retain(prepared.image); }
    }
    if (prepared.clusterAccelerationStructure) {
        if (auto* frame = context.commandBuffer().frameContext()) { frame->retain(prepared.clusterAccelerationStructure); }
    }
    prepared.textureFeedback = nullptr;
    if (prepared.geometry) {
        auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
        if (!gpuScene) { return makeError(Error::InvalidArgument); }
        auto& stream = *prepared.geometry;
        auto* scene = context.runtimeScene();
        const bool cacheOnly = context.properties().value("streamAssetOnly", false) &&
            prepared.state->kind == SceneStreamKind::Asset;
        std::vector<uint32_t> renderNodes(stream.asset().instanceCount(), UINT32_MAX);
        std::vector<uint32_t> mapping(renderNodes.size(), UINT32_MAX);
        prepared.geometryOwnerMask.assign(std::max<size_t>(gpuScene->instances().size(), 1), 0);
        prepared.mappedInstanceCount = 0;
        if (cacheOnly) {
            gpuScene->scene().ensureDrawSet();
            std::iota(mapping.begin(), mapping.end(), 0u);
        } else {
            if (!scene) { return makeError(Error::InvalidArgument); }
            for (size_t i = 0; i < renderNodes.size(); ++i) {
                const auto local = stream.asset().instances()[i].renderNodeIndex;
                renderNodes[i] = prepared.streamSourceId.empty() ? local :
                    uint32_t(scene->renderNodeIndexForSource(prepared.streamSourceId, int32_t(local)));
                if (renderNodes[i] == UINT32_MAX) { log = "Stream source has no matching render node"; return makeError(Error::InvalidArgument); }
                const auto instance = gpuScene->instanceForRenderNode(renderNodes[i]);
                if (instance.valid() && instance.index < prepared.geometryOwnerMask.size()) {
                    mapping[i] = instance.index;
                    prepared.geometryOwnerMask[instance.index] = 1;
                    ++prepared.mappedInstanceCount;
                }
            }
            auto result = stream.syncRuntimeScene(*scene, renderNodes, log);
            if (!result) { return result; }
            if (scene->hasStreamGeometry() && prepared.state->kind == SceneStreamKind::Visibility &&
                prepared.mappedInstanceCount != gpuScene->instances().size()) {
                log = "Stream metadata contains instances without cooked geometry";
                return makeError(Error::InvalidArgument);
            }
        }
        auto result = stream.syncGPUSceneInstanceMapping(mapping);
        if (!result) { return result; }
        auto scope = context.profileScope("Stream Begin");
        result = stream.cmdBeginFrame(context.commandBuffer(), *streamer(), view, [&] {
            auto upload = context.profileScope("Upload preflight");
            flush(context.commandBuffer(), [&](const char* name) { upload.next(name); });
        });
        context.publishCpuProfile(stream.beginFrameCpuProfile().sections);
        if (!result) { return result; }
        if (auto* frame = context.commandBuffer().frameContext()) { frame->retain(prepared.geometry); }
    }
    if (prepared.snapshot && prepared.snapshot->pathTraceResources) {
        auto resources = prepared.snapshot->pathTraceResources;
        if (requirements.textureFeedback) {
            auto [entry, inserted] = textureFrames_.try_emplace(resources.get(), nullptr);
            if (inserted) {
                CpuProfileRecorder profiler;
                Result result;
                {
                    CpuProfileScope profile(&profiler, "Texture streaming");
                    result = resources->beginTextureStreaming(context.commandBuffer(), context.frameIndex(),
                        entry->second, &profiler, context.properties().value("benchmarkFreezeStreaming", false));
                }
                context.publishCpuProfile(profiler.sections);
                if (!result) { textureFrames_.erase(entry); return result; }
            }
            prepared.textureFeedback = entry->second;
        }
        const auto result = resources->uploadMaterialTextures(context.commandBuffer());
        if (!result) { return result; }
    }
    return {};
}

Result StreamerSubsystem::recordSceneTraversal(PreparedSceneResources& prepared,
    RenderGraphExecutionContext& context, const MeshletStreamFrameDesc& view,
    const MeshletStreamRuntime::TraversalCheckpoint& checkpoint)
{
    if (!prepared.geometry) { return {}; }
    auto traversalProfile = context.profileScope("Stream traversal");
    auto phaseProfile = context.profileScope("Traversal setup");
    const auto result = prepared.geometry->cmdPreTraversal(context.commandBuffer(), view,
        [&](std::string_view point) {
            if (point == "BeforeStreamUpdates") { phaseProfile.next("Page updates"); }
            else if (point == "AfterStreamUpdates") { phaseProfile.next("Priority clear"); }
            else if (point == "AfterStreamPriorityClear") { phaseProfile.next("LOD clear / demand seed"); }
            else if (point == "AfterStreamStateClear") { phaseProfile.next("Detail demand"); }
            else if (point == "AfterStreamDemand") { phaseProfile.next("LOD frontier"); }
            else if (point == "AfterStreamFrontier") { phaseProfile.next("LOD mask"); }
            else if (point == "AfterStreamMask") { phaseProfile.next("LOD prefix"); }
            else if (point == "AfterStreamPrefix") { phaseProfile.next("LOD emit"); }
            else if (point == "AfterStreamEmit") { phaseProfile.next("Prefetch"); }
            else if (point == "AfterStreamPrefetch") { phaseProfile.next("Traversal finalize"); }
            else if (point == "BeforeStreamClasBuild") { phaseProfile.next("CLAS build"); }
            else if (point == "AfterStreamClasBuild") { phaseProfile.end(); }
            else if (point == "BeforeFallbackBlas") { phaseProfile.next("Fallback BLAS"); }
            else if (point == "BeforeBlasReset") { phaseProfile.next("BLAS reset"); }
            else if (point == "BeforeBlasCompare") { phaseProfile.next("BLAS cut compare"); }
            else if (point == "BeforeBlasCount") { phaseProfile.next("BLAS count"); }
            else if (point == "BeforeBlasSetup") { phaseProfile.next("BLAS setup"); }
            else if (point == "BeforeBlasInsert") { phaseProfile.next("BLAS insert"); }
            else if (point == "BeforeBlasBuild") { phaseProfile.next("BLAS build"); }
            else if (point == "BeforeTlasInput") { phaseProfile.next("TLAS input"); }
            else if (point == "BeforeTlasBuild") { phaseProfile.next("TLAS build"); }
            else if (point == "AfterTlasBuild") { phaseProfile.end(); }
            if (checkpoint) { checkpoint(point); }
        });
    phaseProfile.end();
    if (result && checkpoint) { checkpoint("AfterTraversal"); }
    return result;
}

Result StreamerSubsystem::recordSceneEnd(PreparedSceneResources& prepared, RenderGraphExecutionContext& context)
{
    if (!prepared.geometry) { return {}; }
    auto scope = context.profileScope("Stream End");
    auto result = prepared.geometry->cmdPostTraversal(context.commandBuffer());
    if (result) { result = prepared.geometry->cmdEndFrame(context.commandBuffer()); }
    if (result) {
        auto profile = prepared.geometry->profilingStats();
        profile.softwareRasterIdentity = prepared.softwareRasterIdentity;
        if (prepared.snapshot && prepared.snapshot->pathTraceResources) {
            const auto textures = prepared.snapshot->pathTraceResources->textureStats();
            profile.textureUpload = textures.lastUpload;
            profile.textureStreaming = textures.streamingEnabled;
            profile.textureResidentBytes = textures.residentAllocationBytes;
            profile.textureBudgetBytes = textures.budgetBytes;
            profile.texturePendingBytes = textures.pendingAllocationBytes;
            profile.textureRetiredBytes = textures.retiredAllocationBytes;
            profile.textureUpgrades = textures.upgrades; profile.textureDowngrades = textures.downgrades;
            profile.textureBudgetDeferrals = textures.budgetDeferrals; profile.textureFeedbackFrames = textures.feedbackFrames;
            profile.textureUploadBytes = textures.streamingUploadBytes; profile.textureMaxRequestFrames = textures.maxRequestLatencyFrames;
            profile.textureRefinedImages = textures.refinedImages; profile.textureRequestedImages = textures.requestedImages;
            profile.texturePendingImages = textures.pendingImages;
        }
        context.publishStreamingProfile(std::move(profile));
    }
    return result;
}

Result StreamerSubsystem::acquireStream(const MeshletStreamRuntimeDesc& desc, bool debugReadback,
    std::shared_ptr<MeshletStreamRuntime>& outSession, std::string& log, PipelineCache* cache)
{
    if (!device_) { return makeError(Error::InvalidArgument); }
    collectReleasedStreams();
    auto session = std::make_shared<MeshletStreamRuntime>();
    session->setDebugReadbackEnabled(debugReadback);
    Result result = session->initialize(*device_, desc, log, cache);
    if (!result) { return result; }
    streams_.push_back(session);
    outSession = std::move(session);
    return {};
}

StreamSceneReadiness StreamerSubsystem::sceneReadiness() const
{
    StreamSceneReadiness result;
    for (const auto& stream : streams_) {
        if (stream.use_count() == 1) { continue; }
        const auto state = stream->sceneReadiness();
        result.requiredPages += state.requiredPages;
        result.completedPages += state.completedPages;
        result.ready = result.ready && state.ready;
    }
    return result;
}

void StreamerSubsystem::collectReleasedStreams()
{
    std::erase_if(streams_, [](const auto& session) { return session.use_count() == 1; });
}

void StreamerSubsystem::flush(CommandBuffer& commands, const StreamUploadPhaseCallback& phase)
{
    if (streamer() && streamer()->stats().pendingCopies.copyCount() != 0) { uploads_.flush(commands, phase); }
}

} // namespace metallic::render
