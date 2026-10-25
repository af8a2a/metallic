#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Debug/RenderDebug.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <thread>
#include <spdlog/spdlog.h>
#include "Runtime/Scene/SceneDocument.h"
#include <fstream>
#include "Runtime/Render/Streamer/MeshletStreamInitialLoader.h"
#include "Runtime/Render/Streamer/StreamedImage.h"
#include "Runtime/Render/Streamer/SceneStreamingConfig.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Profiling/CPUProfile.h"

namespace metallic::render {
struct SceneStreamingState {
    MeshletStreamRuntimeDesc desc;
    uint64_t sceneIdentity = 0;
    uint64_t structuralRevision = 0;
    bool debugReadback = false;
};

Result<> StreamerSubsystem::initialize(const RenderSubsystemInitContext& context, std::string& log)
{
    device_ = &context.device;
    return uploads_.initialize(context.device, log, context.host.frameSlotCount());
}

Result<> StreamerSubsystem::beginFrame(const RenderSubsystemFrameContext& context, RenderChangeBits&, std::string& log)
{
    collectReleasedStreams();
    // Scene/pass replacement has finished and old borrowers have retired. Run
    // initialization before this frame opens the normal graph upload stream.
    const auto result = completeInitialLoads(log);
    if (!result) { return result; }
    textureFrames_.clear();
    debugFrameStreams_.clear();
    if (context.frameResources) { return uploads_.beginFrame(*context.frameResources); }
    uploads_.beginFrame();
    return {};
}

Result<> StreamerSubsystem::recordPostGraph(const RenderSubsystemFrameContext& context, std::string& log)
{
    if (textureFrames_.empty()) { return {}; }
    if (!context.commandBuffer) { return makeError(Error::InvalidArgument); }
    // The graph epilogue joins every consumer, including multiple Deferred
    // passes sharing these scene resources and passes without streamed geometry.
    for (const auto& entry : textureFrames_) {
        const auto& frame = entry.second;
        const auto result = frame.resources->endTextureStreaming(*context.commandBuffer, frame.frameIndex);
        if (!result) {
            log = "Texture feedback readback recording failed: " + std::string(resultToString(result));
            return result;
        }
    }
    return {};
}

void StreamerSubsystem::endFrame(const RenderSubsystemFrameContext&)
{
    textureFrames_.clear();
    uploads_.endFrame();
}

void StreamerSubsystem::prepareBeforePacing(CPUProfileRecorder* profiler)
{
    for (const auto& stream : streams_) {
        // A released view must not advance residency or launch more I/O.
        if (stream.use_count() > 1) { stream->prepareMaintenance(profiler); }
    }
}

void StreamerSubsystem::appendDebugBindings(const RenderSubsystemFrameContext&, std::vector<DebugResourceBinding>& bindings)
{
    std::erase_if(debugStreamIds_, [&](const auto& item) {
        return item.second.owner.expired() ||
            std::none_of(streams_.begin(), streams_.end(), [&](const auto& stream) { return stream.get() == item.first; });
    });
    for (const auto& stream : streams_) {
        if (!stream->ready() || !debugFrameStreams_.contains(stream.get())) { continue; }
        auto [identity, inserted] = debugStreamIds_.try_emplace(stream.get(), DebugStreamIdentity{stream, nextDebugStreamId_});
        if (inserted) { ++nextDebugStreamId_; }
        const size_t begin = bindings.size();
        stream->appendDebugBindings(bindings, "stream" + std::to_string(identity->second.id) + ".");
        for (size_t i = begin; i < bindings.size(); ++i) {
            auto& binding = bindings[i];
            binding.metadata["streamId"] = identity->second.id;
            if (binding.layout == "CompactStreamVisibleRecord") {
                binding.metadata["captureSupported"] = false;
                binding.metadata["reason"] = "Inspect visible clusters at the producer's AfterPass checkpoint; sparse slots need its visibility evidence";
            }
        }
    }
}

void StreamerSubsystem::shutdown()
{
    // The host waits for submitted work and retires graph passes before this.
    initialLoads_.clear();
    streams_.clear();
    debugStreamIds_.clear();
    debugFrameStreams_.clear();
    textureFrames_.clear();
    resources_.clear();
    uploads_.reset();
    device_ = nullptr;
}

Result<> StreamerSubsystem::prepareScene(const SceneStreamingRequirements& requirements,
    const RenderGraphProperties& properties, const scene::Scene* scene,
    std::shared_ptr<PreparedSceneResources>& prepared, std::string& log, bool debugReadback)
{
    if (!device_) { return makeError(Error::InvalidArgument); }
    if (requirements.features == SceneResourceFeatureBits::None && !requirements.geometry && !requirements.sampledImage) {
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
    const bool tessellation = properties.value("rtTessellation", false);
    const auto features = tessellation
        ? requirements.features | SceneResourceFeatureBits::AccelerationStructure |
            SceneResourceFeatureBits::Materials | SceneResourceFeatureBits::MaterialTextures
        : requirements.features;
    if (features != SceneResourceFeatureBits::None) {
        auto resourceProperties = properties;
        const auto result = resources_.acquire(*device_, *device_->getQueue(QueueType::Graphics), resourceProperties, scene, features, log).transform([&](auto value) { prepared->snapshot = std::move(value); });
        if (!result) { return result; }
    }
    using namespace streaming_detail;
    constexpr auto geometryFeatures = SceneResourceFeatureBits::Geometry | SceneResourceFeatureBits::Meshlets |
        SceneResourceFeatureBits::AccelerationStructure;
    const bool wantsGeometry = requirements.geometry ||
        (uint32_t(features) & uint32_t(geometryFeatures)) != 0;
    if (!wantsGeometry) { prepared->geometry.reset(); prepared->state.reset(); return {}; }
    if (!scene) {
        const auto resolved = resources_.resolveScene(properties, scene, log);
        if (!resolved) { return makeError(resolved.error()); }
        scene = *resolved;
    }
    prepared->sourceScene = scene;
    auto sourcePath = scene->filename().empty() ? std::filesystem::path{} : normalizedScenePath(scene->filename());
    auto desc = sceneStreamRuntimeDesc(properties, sourcePath, uint32_t(scene->textures().size()) + 1);
    const bool rayTracing = (uint32_t(features) &
        uint32_t(SceneResourceFeatureBits::AccelerationStructure)) != 0 ||
        properties.value("enableClusterRtx", false);
    desc.enableClusterRtx = rayTracing;
    desc.enableClas = rayTracing;
    if (rayTracing && (!device_->capabilities().clusterAccelerationStructure ||
        !device_->capabilities().rayTracingAccelerationStructure)) {
        log = "Scene ray tracing requires CLAS support";
        return makeError(Error::Unsupported);
    }
    const uint64_t identity = scene->resourceIdentity();
    const uint64_t structural = scene->sceneGraph().structuralRevision();
    // A session rebuild uses a unique cache for runtime instance edits and
    // composed scenes. Never overwrite an asset still mapped by in-flight work.
    const bool runtimeCook = scene->isComposed() ||
        (prepared->state && (prepared->state->sceneIdentity != identity ||
         prepared->state->structuralRevision != structural));
    const auto selectRuntimeCache = [&]() -> Result<> {
        const auto directory = std::filesystem::path(PROJECT_SOURCE_DIR) / ".cache" / "scene-streams";
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) { log = "Cannot create scene streaming cache: " + error.message(); return makeError(Error::Failure); }
        static const auto sessionId = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto key = std::to_string(sessionId) + "-" + std::to_string(identity) + "-" + std::to_string(structural);
        sourcePath = directory / (key + ".source");
        if (!std::filesystem::exists(sourcePath)) {
            std::ofstream marker(sourcePath);
            marker << scene->filename().generic_string() << '\n' << key;
            if (!marker) { log = "Cannot write scene streaming source identity"; return makeError(Error::Failure); }
        }
        desc.sourcePath = sourcePath;
        desc.streamAssetPath = directory / (key + ".streamasset");
        return {};
    };
    if (runtimeCook || sourcePath.empty()) {
        const auto result = selectRuntimeCache();
        if (!result) { return result; }
    }
    if (prepared->state && prepared->state->sceneIdentity == identity &&
        prepared->state->structuralRevision == structural) {
        // Preserve the temporary cache chosen for the current runtime topology.
        desc.sourcePath = prepared->state->desc.sourcePath;
        desc.streamAssetPath = prepared->state->desc.streamAssetPath;
    }
    if (!prepared->state || prepared->state->desc != desc || prepared->state->sceneIdentity != identity ||
        prepared->state->structuralRevision != structural || prepared->state->debugReadback != debugReadback) {
        scene::MeshletStreamAsset asset;
        std::string reason;
        bool compatible = asset.open(desc.streamAssetPath, reason) &&
            asset.isRuntimeCompatibleForSource(desc.sourcePath, reason);
        if (compatible) {
            // Node edits can change instances without changing the source file.
            compatible = asset.instanceCount() == scene->renderNodes().size();
            for (const auto& instance : asset.instances()) {
                if (!compatible) { break; }
                compatible = instance.renderNodeIndex < scene->renderNodes().size() &&
                    instance.primitiveIndex < asset.primitives().size();
                // Offline cooks use canonical source geometry ordinals; stream
                // metadata creates a primitive per instance. Their indices are
                // not comparable. Source dependencies and the structural
                // revision above establish that metadata retains source topology.
                if (compatible && !scene->hasStreamGeometry()) {
                    compatible = asset.primitives()[instance.primitiveIndex].renderPrimitiveIndex ==
                        uint32_t(scene->renderNodes()[instance.renderNodeIndex].renderPrimitiveIndex);
                }
            }
        }
        asset.close();
        if (!compatible) {
            // A previous session can still map this file. Cook replacements to
            // a separate identity rather than truncating its live backing file.
            if (std::filesystem::exists(desc.streamAssetPath)) {
                const auto result = selectRuntimeCache();
                if (!result) { return result; }
            }
            const scene::Scene* geometryScene = scene;
            scene::SceneDocument importedGeometry;
            if (scene->hasStreamGeometry()) {
                if (!importedGeometry.load(scene->filename())) {
                    log = "Cannot import geometry for scene streaming: " + importedGeometry.lastLoadResult().error;
                    return makeError(Error::Failure);
                }
                geometryScene = &importedGeometry;
            }
            if (!scene::buildMeshletStreamAsset({.scene = geometryScene, .instanceScene = scene,
                    .sourcePath = desc.sourcePath, .outputPath = desc.streamAssetPath}, reason)) {
                log = "Cannot cook scene streaming geometry: " + reason;
                return makeError(Error::Failure);
            }
        }
        std::shared_ptr<MeshletStreamRuntime> stream;
        auto result = acquireStream(desc, debugReadback, log).transform([&](auto value) { stream = std::move(value); });
        if (!result) { return result; }
        prepared->geometry = std::move(stream);
        prepared->state = std::make_shared<SceneStreamingState>(SceneStreamingState{desc, identity, structural, debugReadback});
        prepared->streamSourcePath = desc.sourcePath;
        prepared->streamAssetPath = desc.streamAssetPath;
    }
    return {};
}

Result<> StreamerSubsystem::recordSceneBegin(PreparedSceneResources& prepared,
    const SceneStreamingRequirements& requirements, RenderGraphExecutionContext& context, const MeshletStreamFrameDesc& view, std::string& log)
{
    prepared.textureFeedback = {};
    prepared.textureFeedbackEnabled = false;
    prepared.ready = !prepared.snapshot || !prepared.snapshot->pathTraceResources ||
        prepared.snapshot->pathTraceResources->textureUploadsReady();
    if (!prepared.ready) { return {}; }
    if (prepared.snapshot && prepared.snapshot->pathTraceResources) {
        // A TLAS retains addresses into BLAS and backend dependencies owned by the scene snapshot.
        // External recordings also need that ownership until commands retire.
        const auto result = context.commandBuffer().retainResource(prepared.snapshot->pathTraceResources);
        if (!result) { return result; }
    }
    if (prepared.geometry) {
        const auto result = context.commandBuffer().retainResource(prepared.geometry);
        if (!result) { return result; }
    }
    if (prepared.image) {
        const auto result = prepared.image->upload(context.commandBuffer());
        if (!result) { return result; }
        if (auto* frame = metallic::render::RenderFrameContext::from(context.commandBuffer())) { frame->retain(prepared.image); }
    }
    if (prepared.geometry) {
        auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
        if (!gpuScene) { return makeError(Error::InvalidArgument); }
        auto& stream = *prepared.geometry;
        const auto* scene = prepared.sourceScene;
        const bool profilePrepare = context.properties().value("benchmarkPrepareProfile", false);
        CPUProfileRecorder prepareRecorder;
        prepareRecorder.threadDiagnostics = context.properties().value("benchmarkCPUThreadDiagnostics", false);
        CPUProfileScope preparePhase(profilePrepare ? &prepareRecorder : nullptr, "Stream instance mapping");
        CPUProfileScope* prepareWork = profilePrepare ? &preparePhase : nullptr;
        std::vector<uint32_t> renderNodes(stream.asset().instanceCount(), UINT32_MAX);
        std::vector<uint32_t> mapping(renderNodes.size(), UINT32_MAX);
        prepared.geometryOwnerMask.assign(std::max<size_t>(gpuScene->instances().size(), 1), 0);
        prepared.mappedInstanceCount = 0;
        {
            if (!scene) { return makeError(Error::InvalidArgument); }
            for (size_t i = 0; i < renderNodes.size(); ++i) {
                const auto local = stream.asset().instances()[i].renderNodeIndex;
                renderNodes[i] = local;
                if (renderNodes[i] == UINT32_MAX) { log = "Stream source has no matching render node"; return makeError(Error::InvalidArgument); }
                const auto instance = gpuScene->instanceForRenderNode(renderNodes[i]);
                if (instance.valid() && instance.index < prepared.geometryOwnerMask.size()) {
                    mapping[i] = instance.index;
                    prepared.geometryOwnerMask[instance.index] = 1;
                    ++prepared.mappedInstanceCount;
                }
            }
            preparePhase.work("instanceRecordsVisited", renderNodes.size());
            preparePhase.work("mappedInstances", prepared.mappedInstanceCount);
            preparePhase.work("ownerMaskBytes", prepared.geometryOwnerMask.size() * sizeof(uint32_t));
            preparePhase.work("mappingArrayBytes", (renderNodes.size() + mapping.size()) * sizeof(uint32_t));
            preparePhase.next("Stream runtime scene sync");
            auto result = stream.syncRuntimeScene(*scene, renderNodes, log, prepareWork);
            if (!result) { return result; }

        }
        preparePhase.next("Stream GPUScene mapping sync");
        auto result = stream.syncGPUSceneInstanceMapping(mapping, prepareWork);
        if (!result) { return result; }
        preparePhase.next("Stream GPUScene binding");
        preparePhase.work("descriptorBindings", 1);
        // recordPreGraph publishes GPUScene instance uploads with an AllCommands
        // shader-read barrier, including this stream TLAS input compute pass.
        result = stream.bindGPUSceneInstances(*gpuScene->globalBufferViews().instances.buffer);
        if (!result) { return result; }
        preparePhase.end();
        if (!prepareRecorder.sections.empty()) { context.publishCpuProfile(prepareRecorder.sections); }
        auto scope = context.profileScope("Stream Begin");
        result = stream.cmdBeginFrame(context.commandBuffer(), *streamer(), view, [&] {
            auto upload = context.profileScope("Upload preflight");
            return flush(context.commandBuffer(), [&](const char* name) { upload.next(name); });
        });
        context.publishCpuProfile(stream.beginFrameCpuProfile().sections);
        if (!result) { return result; }
        if (context.debugEnabled()) { debugFrameStreams_.insert(prepared.geometry.get()); }
        if (auto* frame = metallic::render::RenderFrameContext::from(context.commandBuffer())) { frame->retain(prepared.geometry); }
    }
    if (prepared.snapshot && prepared.snapshot->pathTraceResources) {
        auto resources = prepared.snapshot->pathTraceResources;
        if (requirements.textureFeedback) {
            auto [entry, inserted] = textureFrames_.try_emplace(resources.get());
            if (inserted) {
                entry->second.resources = resources;
                entry->second.frameIndex = context.frameIndex();
                CPUProfileRecorder profiler;
                Result<> result;
                {
                    CPUProfileScope profile(&profiler, "Texture streaming");
                    result = resources->beginTextureStreaming(context.commandBuffer(), context.frameIndex(),
                        entry->second.feedback, &profiler, context.properties().value("benchmarkFreezeStreaming", false) ||
                            context.properties().value("benchmarkFreezeResidency", false),
                        &entry->second.captureDemand);
                }
                context.publishCpuProfile(profiler.sections);
                if (!result) { textureFrames_.erase(entry); return result; }
            }
            prepared.textureFeedback = entry->second.feedback;
            prepared.textureFeedbackEnabled = entry->second.captureDemand;
        }
        const auto result = resources->uploadMaterialTextures(context.commandBuffer());
        if (!result) { return result; }
        // Scene begin precedes describeSceneView; allocation settings come from
        // the pass properties here, while traversal receives its resolved view.
        if (prepared.geometry && context.properties().value("rtTessellation", false)) {
            const StreamCLASTessellationSettings settings{
                .enabled = true, .maxPatches = context.properties().value("rtTessellationMaxPatches", 16384u),
                .buildMode = parseStreamCLASTessellationBuildMode(
                    context.properties().value("rtTessellationBuildMode", std::string("triangles")))};
            const auto configured = prepared.geometry->configureTessellation(*device_, *prepared.sourceScene,
                *resources, settings, log);
            if (!configured) { return configured; }
        }
    }
    return {};
}

Result<> StreamerSubsystem::recordSceneTraversal(PreparedSceneResources& prepared,
    RenderGraphExecutionContext& context, const MeshletStreamFrameDesc& view,
    const MeshletStreamRuntime::TraversalCheckpoint& checkpoint)
{
    if (!prepared.geometry) { return {}; }
    auto traversalProfile = context.profileScope("Stream traversal");
    auto phaseProfile = context.profileScope("Traversal setup");
    RenderGraphExecutionContext::ProfileScope clasProfile;
    RenderGraphExecutionContext::ProfileScope rtCutProfile;
    RenderGraphExecutionContext::ProfileScope tessellationClasProfile;
    const bool profilePrepare = context.properties().value("benchmarkPrepareProfile", false);
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
            else if (point == "BeforeStreamClasBuild") {
                phaseProfile.next("CLAS build");
                clasProfile = context.profileScope("CLAS preparation");
            }
            else if (point == "BeforeCLASScratchBuild") { clasProfile.next("CLAS scratch build"); }
            else if (point == "BeforeCLASAllocate") { clasProfile.next("CLAS allocate"); }
            else if (point == "BeforeCLASMove") { clasProfile.next("CLAS move"); }
            else if (point == "BeforeCLASPublish") { clasProfile.next("CLAS publish"); }
            else if (point == "AfterStreamClasBuild") { clasProfile.end(); phaseProfile.end(); }
            else if (point == "BeforeFallbackBlas") { phaseProfile.next("Fallback BLAS"); }
            else if (point == "BeforeRTRepresentationClassify") { phaseProfile.next("RT representation classify"); }
            else if (point == "BeforeRTRepresentationCut") {
                phaseProfile.next("RT representation cut");
                if (profilePrepare) { rtCutProfile = context.profileScope("RT cut count"); }
            }
            else if (point == "BeforeRTCutAdmission") { if (profilePrepare) { rtCutProfile.next("RT cut admission"); } }
            else if (point == "BeforeRTCutCacheSelection") { if (profilePrepare) { rtCutProfile.next("RT cut cache selection"); } }
            else if (point == "BeforeRTCutPublish") { if (profilePrepare) { rtCutProfile.next("RT cut publish"); } }
            else if (point == "BeforeBlasReset") { rtCutProfile.end(); phaseProfile.next("BLAS reset"); }
            else if (point == "BeforeBlasCompare") { phaseProfile.next("BLAS cut compare"); }
            else if (point == "BeforeBlasCount") { phaseProfile.next("BLAS count"); }
            else if (point == "BeforeBlasSetup") { phaseProfile.next("BLAS setup"); }
            else if (point == "BeforeBlasAllocate") { phaseProfile.next("BLAS prepare allocation"); }
            else if (point == "BeforeBlasInsert") { phaseProfile.next("BLAS insert"); }
            else if (point == "BeforeBlasBuild") { phaseProfile.next("BLAS build"); }
            else if (point == "BeforeBlasGPUAllocate") { phaseProfile.next("BLAS GPU allocate"); }
            else if (point == "BeforeBlasMove") { phaseProfile.next("BLAS move"); }
            else if (point == "BeforeBlasPublish") { phaseProfile.next("BLAS publish"); }
            else if (point == "BeforeBlasStatistics") { phaseProfile.next("BLAS statistics"); }
            else if (point == "BeforeTessellation") { phaseProfile.next("CLAS tessellation"); }
            else if (point == "BeforeTessellationCLAS") { tessellationClasProfile = context.profileScope("Tessellation CLAS build"); }
            else if (point == "AfterTessellationCLAS") { tessellationClasProfile.end(); }
            else if (point == "AfterTessellation") { phaseProfile.end(); }
            else if (point == "BeforeTessellationPublish") { phaseProfile.next("Tessellation publish"); }
            else if (point == "AfterTessellationPublish") { phaseProfile.end(); }
            else if (point == "BeforeTlasInput") { phaseProfile.next("TLAS input"); }
            else if (point == "BeforeTlasBuild") { phaseProfile.next("TLAS build"); }
            else if (point == "AfterTlasBuild") { phaseProfile.end(); }
            if (checkpoint) { checkpoint(point); }
        }, context.output("accelerationStructure") != nullptr &&
            context.properties().value("AsyncComputePreferred", true) && context.supportsParallelCompute());
    clasProfile.end();
    rtCutProfile.end();
    tessellationClasProfile.end();
    phaseProfile.end();
    if (result && checkpoint) { checkpoint("AfterTraversal"); }
    return result;
}

Result<> StreamerSubsystem::recordSceneEnd(PreparedSceneResources& prepared, RenderGraphExecutionContext& context)
{
    if (!prepared.geometry) { return {}; }
    auto scope = context.profileScope("Stream End");
    auto result = prepared.geometry->cmdPostTraversal(context.commandBuffer());
    if (result) { result = prepared.geometry->cmdEndFrame(context.commandBuffer()); }
    if (result) {
        auto profile = prepared.geometry->profilingStats();
        const bool captureSemanticSource = context.properties().value("benchmarkCaptureSemanticSource", false);
        if (context.properties().value("benchmarkFreezeResidency", false) || captureSemanticSource) {
            const auto push = prepared.geometry->userPush();
            profile.diagnosticSourceIdentity = std::to_string(push.pageBuffer) + ":" +
                std::to_string(push.pageTableBuffer) + ":" + std::to_string(push.pageTableMutationSerial) + ":" +
                std::to_string(push.clasPublicationRevision);
            if (prepared.snapshot && prepared.snapshot->pathTraceResources) {
                profile.diagnosticSourceIdentity += ":" +
                    prepared.snapshot->pathTraceResources->diagnosticTextureResidencyIdentity(context.frameIndex());
                if (captureSemanticSource) {
                    profile.diagnosticTextureSemanticIdentity =
                        prepared.snapshot->pathTraceResources->diagnosticTextureSemanticIdentity(context.frameIndex());
                }
            }
        }
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

Result<std::shared_ptr<MeshletStreamRuntime>> StreamerSubsystem::acquireStream(
    const MeshletStreamRuntimeDesc& desc,
    bool debugReadback,
    std::string& log,
    PipelineCache* cache)
{
    std::shared_ptr<MeshletStreamRuntime> outSession{};
    if (!device_) { return makeError(Error::InvalidArgument); }
    collectReleasedStreams();
    auto session = std::make_shared<MeshletStreamRuntime>();
    session->setDebugReadbackEnabled(debugReadback);
    Result<> result = session->initialize(*device_, desc, log, cache);
    if (!result) { return makeError(result.error()); }
    if (desc.initialLoad || desc.deviceImmutableMetadata) {
        // Immutable Device tables are mandatory before any consumer, even when
        // root pages keep their existing lazy-loading policy.
        initialLoads_.push_back({.runtime = session, .metadataOnly = !desc.initialLoad});
    }
    streams_.push_back(session);
    outSession = std::move(session);
    return outSession;
}

Result<> StreamerSubsystem::completeInitialLoads(std::string& log)
{
    for (auto& pending : initialLoads_) {
        const auto session = pending.runtime.lock();
        if (!session) { continue; }
        if (pending.failure) { log = pending.failureLog; return makeError(*pending.failure); }
        const auto fail = [&](Result<> result) {
            // A failed decode/build can consume a root request. Require a fresh
            // session instead of retrying the same partial state every frame.
            pending.failure = result.error();
            pending.failureLog = log;
            return result;
        };
        // No graph command has used this new session. Keep initialization
        // submissions independent of render frames, and drain before recording.
        MeshletStreamInitialLoader loader;
        auto result = loader.initialize(*device_, log);
        if (!result) { return fail(result); }
        const auto start = std::chrono::steady_clock::now();
        auto progressAt = start;
        bool complete = false;
        while (!complete) {
            const auto beforeBytes = loader.stats().uploadBytes;
            const auto beforePages = session->sceneReadiness().completedPages;
            result = loader.pump(*session, 16.0, complete, log, pending.metadataOnly);
            if (!result) { return fail(result); }
            const auto now = std::chrono::steady_clock::now();
            const auto readiness = session->sceneReadiness();
            if (!complete && now - progressAt >= std::chrono::seconds(5)) {
                if (pending.metadataOnly) {
                    spdlog::info("[StreamMetadataLoad] batches={} uploadedMiB={:.1f}",
                        loader.stats().batches, double(session->immutableMetadataUploadedBytes()) / (1024.0 * 1024.0));
                } else {
                    spdlog::info("[StreamInitialLoad] resources={}/{} batches={} uploadedMiB={:.1f}",
                        readiness.completedPages, readiness.requiredPages, loader.stats().batches,
                        double(loader.stats().uploadBytes) / (1024.0 * 1024.0));
                }
                progressAt = now;
            }
            if (!complete && now - start >= std::chrono::minutes(5)) {
                log = pending.metadataOnly ? "Immutable geometry metadata did not become ready within 5 minutes" :
                    "Initial geometry loading did not become ready within 5 minutes";
                return fail(makeError(Error::Failure));
            }
            if (!complete && beforeBytes == loader.stats().uploadBytes && beforePages == readiness.completedPages) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        const auto stats = loader.stats();
        result = loader.reset();
        if (!result) { log = "Initial geometry loader handoff failed"; return fail(result); }
        if (pending.metadataOnly) {
            spdlog::info("[StreamMetadataLoad] ready batches={} uploadedMiB={:.1f} elapsedMs={:.3f} gpuWaitMs={:.3f}",
                stats.batches, double(session->immutableMetadataUploadedBytes()) / (1024.0 * 1024.0),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                stats.gpuWaitMilliseconds);
        } else {
            spdlog::info("[StreamInitialLoad] ready resources={} batches={} uploadedMiB={:.1f} elapsedMs={:.3f} gpuWaitMs={:.3f}",
                session->sceneReadiness().requiredPages, stats.batches,
                double(stats.uploadBytes) / (1024.0 * 1024.0),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                stats.gpuWaitMilliseconds);
        }
        pending.runtime.reset();
    }
    initialLoads_.clear();
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

Result<> StreamerSubsystem::flush(CommandBuffer& commands, const StreamUploadPhaseCallback& phase)
{
    if (streamer() && streamer()->stats().pendingCopies.copyCount() != 0) { if (auto commandResult = uploads_.flush(commands, phase); !commandResult) { return commandResult; } }
    return {};
}

} // namespace metallic::render
