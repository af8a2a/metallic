#include "RhiTest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void requireFull(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

class ZorahFullFirstFrameTest final : public RhiTest {
public:
    ZorahFullFirstFrameTest() { type = RhiTestType::Rendering; name = "zorah_full_first_frame"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (!std::getenv("METALLIC_TEST_ZORAH_FULL")) {
            return RhiTestResult::skip("Set METALLIC_TEST_ZORAH_FULL=1 after the complete Z5 cook");
        }
        const auto directory = std::filesystem::absolute(context.outputDirectory);
        std::filesystem::create_directories(directory);
        Json report{{"status", "running"}, {"protocol", "zorah-full-first-frame-v1"},
            {"width", 960}, {"height", 540}, {"dlss", false}, {"switchFromMiniZorah", true}, {"validation",context.enableValidation},
            {"skipMeshesApplied", false}, {"runs", Json::array()}};
        const auto save = [&]() { std::ofstream(directory / "ZorahFullFirstFrame.json") << report.dump(2) << '\n'; };
        save();
        try {
            std::string log;
            RenderSampleLoadResult sample;
            requireFull(loadBuiltInRenderSample(kGPUDrivenZorahFullSampleId, sample, log), log);
            requireFull(!sample.desc.loadSceneInEditor, "Full sample must bypass resident import");
            auto graph = sample.graph;
            graph.markOutput("VBuffer.color");
            // Headless evidence uses native resolution. The interactive sample
            // retains DLSS-SR; its timing is not compared to this readback run.
            graph.removeNode(graph.findNode("DlssSr")->id);
            graph.removeNode(graph.findNode("DlssNr")->id);
            graph.addEdge("Deferred.color", "AutoExposure.source");
            graph.addEdge("AutoExposure.color", "FinalBlit.source");
            auto view = graph.viewProperties(); view["temporalJitter"] = false; graph.setViewProperties(view);
            report["camera"] = view["camera"];
            report["properties"] = graph.findNode("VBuffer")->properties;
            const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath;
            scene::MeshletStreamAsset asset;
            requireFull(asset.open(std::filesystem::path(PROJECT_SOURCE_DIR) /
                graph.findNode("VBuffer")->properties.at("streamAssetPath").get<std::string>(), log), log);
            requireFull(asset.isCurrentForSource(source), "Full cache is stale");
            requireFull(asset.primitiveCount() == 5715 && asset.instanceCount() == 43068,
                "Full cook lost geometries or primitive instances");
            report["cook"] = {{"primitives",asset.primitiveCount()}, {"instances",asset.instanceCount()},
                {"pages",asset.pageCount()}, {"revision",asset.cookRevision()}};
            uint32_t cycles = 2;
            if (const char* value = std::getenv("METALLIC_ZORAH_FULL_CYCLES")) { cycles = std::clamp(std::atoi(value), 1, 4); }
            scene::SceneDocument scene;
            RenderGraphPreviewRenderer preview;
            const auto deviceStarted = Clock::now();
            requireFull(bool(preview.initialize(context.enableValidation, true, false)), preview.lastLog());
            RenderGraph warmup;
            warmup.addNode("FinalBlitPass", "Warmup"); warmup.markOutput("Warmup.color");
            requireFull(bool(preview.render(warmup, 32, 32, "Warmup.color")), preview.lastLog());
            auto* budgetDevice = preview.subsystemHost()->device();
            auto budgetPolicy = budgetDevice->memoryBudget().policy;
            budgetPolicy.enabled = true;
            budgetDevice->setMemoryBudgetPolicy(budgetPolicy);
            report["deviceInitializeSeconds"] = std::chrono::duration<double>(Clock::now()-deviceStarted).count();
            for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
                // Populate the executor-owned queued frame slots with MiniZorah
                // before Full. A readback-only warmup does not cover editor switches.
                RenderSampleLoadResult mini;
                requireFull(loadBuiltInRenderSample(kDefaultGPUDrivenSampleId, mini, log), log);
                mini.graph.removeNode(mini.graph.findNode("DlssSr")->id);
                mini.graph.removeNode(mini.graph.findNode("DlssNr")->id);
                mini.graph.addEdge("Deferred.color", "AutoExposure.source");
                mini.graph.addEdge("AutoExposure.color", "FinalBlit.source");
                preview.bindRuntimeScene(nullptr);
                for (uint32_t frame = 0; frame < 8; ++frame) {
                    requireFull(bool(preview.render(mini.graph,960,540,mini.desc.previewOutput,false)),preview.lastLog());
                }
                requireFull(preview.subsystemHost()->get<StreamerSubsystem>()->streamCount()==1,
                    "MiniZorah warmup did not retain exactly one stream");
                graph.markDirty();
                const auto started = Clock::now();
                const auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now()-started).count(); };
                requireFull(scene.loadStreamMetadata(source), scene.lastLoadResult().error);
                // Exercise the standalone asset entry, then the editor world entry.
                const bool worldBinding = (cycle % 2u) != 0u;
                preview.bindRuntimeScene(worldBinding ? &scene : nullptr);
                graph.setNodeRuntimeProperty(graph.findNode("VBuffer")->id,
                    "sceneBinding", worldBinding ? "world" : "asset");
                requireFull(scene.renderNodes().size() == asset.instanceCount(), "Runtime/cook instance mismatch");
                for (const auto& primitive : scene.renderPrimitives()) {
                    requireFull(primitive.positions.empty() && primitive.indices.empty(), "Full geometry imported resident");
                }
                Json run{{"metadataSeconds",elapsed()}, {"cycle",cycle},
                    {"sceneBinding",worldBinding ? "world" : "asset"}, {"frames",Json::array()}};
                preview.setEnvironment({.enabled=true,
                    .path=std::filesystem::path(PROJECT_SOURCE_DIR)/sample.desc.environment->path, .visible=false});
                scene::LightingSettings lighting;
                lighting.autoExposure.enabled = false; lighting.exposureEV100 = 2;
                scene::PunctualLight sun;
                sun.properties.type = "directional"; sun.properties.intensity = 10;
                sun.properties.color = float3(1, .8f, .5f);
                sun.direction = float3(-.6f, -.7f, -.36f);
                lighting.lights.push_back(sun);
                requireFull(preview.setLighting(lighting), "Invalid validation lighting");
                uint32_t readyFrame = 0;
                const auto capture = [&](const std::string& name) {
                    requireFull(saveRgba8Png(directory / (name+"-"+std::to_string(cycle)+".png"),
                        reinterpret_cast<const uint8_t*>(preview.pixels().data()),960,540,log),log);
                    const std::set<uint32_t> colors(preview.pixels().begin(),preview.pixels().end());
                    requireFull(colors.size()>256, "Full frame has no useful geometry image (environment background disabled)");
                };
                for (uint32_t frame=0; frame<1200; ++frame) {
                    const double begin=elapsed();
                    requireFull(bool(preview.render(graph,960,540,"FinalBlit.color",true)),preview.lastLog());
                    auto* streamer=preview.subsystemHost()->get<StreamerSubsystem>();
                    requireFull(streamer && streamer->streamCount()==1, "Expected exactly one stream session");
                    const auto readiness=streamer->sceneReadiness();
                    const auto& stats=preview.executionStats();
                    requireFull(stats.streaming.size()==1, "Missing Full streaming telemetry");
                    const auto& stream=stats.streaming.front();
                    Json row{{"frame",frame+1}, {"elapsedSeconds",elapsed()}, {"wallMilliseconds",(elapsed()-begin)*1000},
                        {"ready",readiness.ready}, {"requiredPages",readiness.requiredPages}, {"completedPages",readiness.completedPages},
                        {"geometryUsedBytes",stream.geometryUsedBytes}, {"geometryBudgetBytes",stream.geometryBudgetBytes},
                        {"clasUsedBytes",stream.clasUsedBytes}, {"clasCapacityBytes",stream.clasCapacityBytes},
                        {"clasScratchBytes",stream.clasScratchBytes}, {"residentPages",stream.residentPages},
                        {"clasResidentPages",stream.clasResidentPages}, {"pendingPages",stream.pendingPages},
                        {"loadFailures",stream.loadFailures}, {"allocationFailures",stream.allocationFailures},
                        {"requestOverflows",stream.requestOverflows}};
                    run["frames"].push_back(row);
                    requireFull(stream.geometryUsedBytes<=stream.geometryBudgetBytes && stream.clasUsedBytes<=stream.clasCapacityBytes,
                        "Streaming pool exceeded budget");
                    requireFull(stream.loadFailures==0, "Full streaming page IO failed");
                    if (readyFrame) { requireFull(readiness.ready,"Full fallback readiness regressed"); }
                    if (!readyFrame && readiness.ready) {
                        readyFrame=frame+1; run["firstReadyFrame"]=readyFrame; run["firstReadySeconds"]=elapsed();
                        graph.setNodeRuntimeProperty(graph.findNode("VBuffer")->id,"visualization","meshlet");
                        requireFull(bool(preview.render(graph,960,540,"VBuffer.color",true)),preview.lastLog());
                        capture("ZorahFull-meshlets");
                        graph.setNodeRuntimeProperty(graph.findNode("VBuffer")->id,"visualization","none");
                        requireFull(bool(preview.render(graph,960,540,"FinalBlit.color",true)),preview.lastLog());
                        capture("ZorahFull-first-ready");
                    }
                    if (frame%30==0 || (readyFrame && frame+1==readyFrame)) {
                        std::printf("[ZorahFull] cycle=%u frame=%u ready=%u/%u geometry=%.1fMiB CLAS=%.1fMiB time=%.2fs\n",
                            cycle,frame+1,readiness.completedPages,readiness.requiredPages,
                            stream.geometryUsedBytes/1048576.,stream.clasUsedBytes/1048576.,elapsed());
                        std::fflush(stdout);
                        report["currentRun"]=run; save();
                    }
                    if (readyFrame && frame+1>=readyFrame+120) { break; }
                }
                requireFull(readyFrame!=0,"Full terminal cut did not become ready in 1200 frames");
                capture("ZorahFull-settled");
                auto* host=preview.subsystemHost();
                auto* gpuScene=host->get<GPUSceneSubsystem>();
                requireFull(gpuScene && !gpuScene->globalBufferViews().vertices.buffer, "Resident GPU vertex upload detected");
                auto* streamer=host->get<StreamerSubsystem>();
                auto* device=host->device();
                std::shared_ptr<SceneResourceSnapshot> materials;
                const scene::Scene* renderedScene=&scene;
                if (!worldBinding) {
                    requireFull(bool(streamer->manager().resolveScene(graph.findNode("VBuffer")->properties,
                        nullptr,renderedScene,log)),log);
                    requireFull(renderedScene && renderedScene->hasStreamGeometry() &&
                        renderedScene->renderNodes().size()==asset.instanceCount(),"Asset entry did not resolve Full metadata");
                }
                requireFull(bool(streamer->manager().acquire(*device,*device->getQueue(QueueType::Graphics),
                    graph.findNode("Deferred")->properties,renderedScene,SceneResourceFeatureBits::Materials,materials,log)),log);
                const auto tex=materials->pathTraceResources->textureStats();
                const auto memory = device->memoryBudget();
                const auto& textureDomain = memory.domains[size_t(MemoryBudgetDomain::MaterialTextures)];
                requireFull(textureDomain.allocationBytes <= tex.residentAllocationBytes + tex.pendingAllocationBytes + tex.retiredAllocationBytes + 1024 * 1024 &&
                    textureDomain.allocationCount <= tex.residentImageCount + 32,
                    "Scene consumers retained duplicate material texture owners");
                run["materialDomainBytes"] = textureDomain.allocationBytes;
                run["materialDomainImages"] = textureDomain.allocationCount;
                const auto& localHeap = memory.heaps[memory.primaryDeviceLocalHeap];
                run["localHeap"] = {{"usageBytes", localHeap.usageBytes}, {"blockBytes", localHeap.blockBytes},
                    {"allocationBytes", localHeap.allocationBytes}, {"blockCount", localHeap.blockCount}};
                run["textures"]={{"logical",tex.logicalTextureCount},{"residentIncludingFallback",tex.residentImageCount},
                    {"maxDimension",tex.selectedMaxDimension},{"maskImages",tex.maskImageCount},{"maskMaxDimension",tex.maskMaxDimension},{"payloadBytes",tex.residentPayloadBytes},
                    {"allocationBytes",tex.residentAllocationBytes},{"budgetBytes",tex.budgetBytes},{"peakStagingBytes",tex.peakStagingBytes},
                    {"upgrades",tex.upgrades},{"downgrades",tex.downgrades},{"refinedImages",tex.refinedImages},
                    {"feedbackFrames",tex.feedbackFrames},{"peakLiveBytes",tex.peakLiveAllocationBytes}};
                report["currentRun"]=run; save();
                requireFull(tex.logicalTextureCount==4418 && tex.ktxImageCount==4418 &&
                    tex.residentImageCount==4419 && tex.selectedMaxDimension<=256 &&
                    tex.maskImageCount==110 && tex.maskMaxDimension==512,
                    "Full texture tails or protected MASK quality differ");
                for (uint32_t index : materials->pathTraceResources->logicalTextureIndices()) {
                    requireFull(index>0 && index<materials->pathTraceResources->materialTextureCount(),
                        "Full logical texture references the fallback descriptor");
                }
                requireFull(tex.residentAllocationBytes<=tex.budgetBytes,"Texture allocations exceeded budget");
                requireFull(!materials->pathTraceResources->shadingVertexBuffer() &&
                    !materials->pathTraceResources->accelerationStructure().valid(),"Material owner imported full resident RTAS");
                const auto deferredId = graph.findNode("Deferred")->id;
                graph.setNodeRuntimeProperty(deferredId,"debugView","baseColor");
                requireFull(bool(preview.render(graph,960,540,"Deferred.color",true)),preview.lastLog());
                capture("ZorahFull-base-color");
                run["baseColorNonblackPixels"]=std::count_if(preview.pixels().begin(),preview.pixels().end(),
                    [](uint32_t pixel) { return (pixel & 0x00ffffffu)!=0; });
                requireFull(run["baseColorNonblackPixels"].get<size_t>()>10000,"Full visibility has negligible material coverage");
                graph.setNodeRuntimeProperty(deferredId,"materialBinning",false);
                requireFull(bool(preview.render(graph,960,540,"Deferred.color",true)),preview.lastLog());
                run["unbinnedBaseColorNonblackPixels"] = std::count_if(preview.pixels().begin(),preview.pixels().end(),
                    [](uint32_t pixel) { return (pixel & 0x00ffffffu)!=0; });
                requireFull(run["unbinnedBaseColorNonblackPixels"].get<size_t>()>10000,"Unbinned Full decode has negligible geometry coverage");
                graph.setNodeRuntimeProperty(deferredId,"materialBinning",true);
                graph.setNodeRuntimeProperty(deferredId,"debugView","final");
                if (std::getenv("METALLIC_TEST_TEXTURE_ROAM")) {
                    const auto originalView = graph.viewProperties();
                    const auto row = [&](const char* phase, uint32_t frame) {
                        const auto t=materials->pathTraceResources->textureStats();
                        run["textureRoam"].push_back({{"phase",phase},{"frame",frame},{"residentBytes",t.residentAllocationBytes},
                            {"pendingBytes",t.pendingAllocationBytes},{"retiredBytes",t.retiredAllocationBytes},
                            {"upgrades",t.upgrades},{"downgrades",t.downgrades},{"refined",t.refinedImages},
                            {"requested",t.requestedImages},{"feedbackFrames",t.feedbackFrames},{"maxRequestFrames",t.maxRequestLatencyFrames}});
                        requireFull(t.peakLiveAllocationBytes<=t.budgetBytes,"Texture replacement exceeded total budget");
                    };
                    requireFull(tex.streamingEnabled && tex.upgrades>0 && tex.feedbackFrames>0,"Full shading produced no refinement demand");
                    auto moved=originalView;
                    for (size_t axis=0;axis<3;++axis) {
                        const double eye=originalView["camera"]["eye"][axis], center=originalView["camera"]["center"][axis];
                        moved["camera"]["eye"][axis]=eye+(center-eye)*.5;
                        moved["camera"]["center"][axis]=center+(center-eye)*.5;
                    }
                    graph.setViewProperties(moved);
                    for (uint32_t frame=0;frame<180;++frame) {
                        requireFull(bool(preview.render(graph,960,540,"FinalBlit.color",true)),preview.lastLog());
                        if (frame%15==0) { row("forward",frame); }
                    }
                    capture("ZorahFull-refined");
                    const auto hot=materials->pathTraceResources->textureStats();
                    auto away=originalView;
                    away["camera"]["eye"]={0,1000000,0}; away["camera"]["center"]={0,1000001,0}; away["camera"]["up"]={0,0,1};
                    graph.setViewProperties(away);
                    for (uint32_t frame=0;frame<420;++frame) {
                        requireFull(bool(preview.render(graph,960,540,"FinalBlit.color",true)),preview.lastLog());
                        if (frame%15==0) { row("away",frame); }
                    }
                    const auto cold=materials->pathTraceResources->textureStats();
                    requireFull(cold.downgrades>hot.downgrades && cold.residentAllocationBytes<hot.residentAllocationBytes,
                        "Full cold textures did not release physical image allocations");
                    graph.setViewProperties(originalView);
                    for (uint32_t frame=0;frame<120;++frame) {
                        requireFull(bool(preview.render(graph,960,540,"FinalBlit.color",true)),preview.lastLog());
                        if (frame%15==0) { row("return",frame); }
                    }
                    capture("ZorahFull-return");
                    requireFull(materials->pathTraceResources->textureStats().upgrades>cold.upgrades,"Full textures did not refine after returning");
                    report["currentRun"]=run; save();
                }
                // Exercise graph removal and retirement before the next full load.
                RenderGraph empty; empty.addNode("FinalBlitPass","Empty"); empty.markOutput("Empty.color");
                for (uint32_t frame=0; frame<=host->frameSlotCount(); ++frame) {
                    requireFull(bool(preview.render(empty,32,32,"Empty.color")),preview.lastLog());
                }
                streamer->collectReleasedStreams();
                requireFull(streamer->streamCount()==0,"Full stream survived graph removal");
                run["retired"]=true; run["totalSeconds"]=elapsed();
                report["runs"].push_back(run); report.erase("currentRun"); save();
            }
            report["status"]="passed"; save();
            return RhiTestResult::pass("Full source instances, budgeted texture tails with MASK floor, bounded first frame and repeated release/load");
        } catch (const std::exception& error) {
            report["status"]="failed"; report["error"]=error.what(); save();
            return RhiTestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(ZorahFullFirstFrameTest);
} // namespace
} // namespace metallic::tests
