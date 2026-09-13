#include "RhiTest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <psapi.h>
#endif

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

class StreamStartupObserver final : public IRenderDebugObserver {
public:
    RenderDebugRuntime debug;
    Json latest;

    void compiled(Json graph) override { debug.compiled(std::move(graph)); }
    void beginExecution(Device& device, debug::DebugEvidenceStamp evidence, RenderSubsystemHost* subsystems) override
    {
        debug.beginExecution(device, std::move(evidence), subsystems);
    }
    void boundary(CommandBuffer& commands, std::string_view checkpoint, uint32_t passId,
        std::string_view pass, std::span<const DebugResourceBinding> resources, const Json& values) override
    {
        if (values.contains("streaming")) {
            latest = values.at("streaming").at("instances").at(0);
            latest.erase("pages");
        }
        debug.boundary(commands, checkpoint, passId, pass, resources, values);
    }
    void endExecution(bool success) override { debug.endExecution(success); }
};

RhiTestResult runStreamStartup(RhiTestContext& context, bool miniZorah, bool unified = false)
{
    const char* enabled = std::getenv("METALLIC_TEST_MINIZORAH");
    if (miniZorah && (enabled == nullptr || std::string_view(enabled) != "1")) {
        return RhiTestResult::skip("Set METALLIC_TEST_MINIZORAH=1 to run the full cooked scene milestone");
    }
    const std::string label = std::string(miniZorah ? "MiniZorah" : "StreamOnlyBunny") + (unified ? "VBuffer" : "");
    Json report{{"status", "running"}, {"cases", Json::array()}, {"cacheState", "OS file cache not flushed"}};
    StreamStartupObserver observer;
    RenderGraphPreviewRenderer preview;
    const auto started = Clock::now();
    const auto seconds = [&]() { return std::chrono::duration<double>(Clock::now() - started).count(); };
    const auto saveReport = [&]() {
        report["elapsedSeconds"] = seconds();
        report["streaming"] = observer.latest;
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX memory{};
        memory.cb = sizeof(memory);
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory))) {
            report["peakProcessCommitBytes"] = memory.PeakPagefileUsage;
            report["peakWorkingSetBytes"] = memory.PeakWorkingSetSize;
        }
        IO_COUNTERS io{};
        if (GetProcessIoCounters(GetCurrentProcess(), &io)) {
            report["processIoReadBytes"] = io.ReadTransferCount;
            report["processIoWriteBytes"] = io.WriteTransferCount;
        }
#endif
        std::ofstream output(context.outputDirectory / (label + "FirstFrameReport.json"));
        output << report.dump(2) << '\n';
    };
    try {
        RenderGraph graph;
        std::filesystem::path source, cache;
        std::string log;
        if (miniZorah) {
            RenderSampleLoadResult sample;
            require(loadBuiltInRenderSample(unified ? "gpu-driven-minizorah-vbuffer" : "gpu-driven-minizorah", sample, log), log);
            require(!sample.desc.loadSceneInEditor && !sample.desc.requiresStreamline, "MiniZorah startup must skip resident import and DLSS");
            graph = std::move(sample.graph);
            const auto& props = graph.findNode("GPUDriven")->properties;
            source = std::filesystem::path(PROJECT_SOURCE_DIR) / props.at("path").get<std::string>();
            cache = std::filesystem::path(PROJECT_SOURCE_DIR) / props.at("streamAssetPath").get<std::string>();
        } else {
            source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
            cache = std::filesystem::absolute(context.outputDirectory / "StreamOnlyBunny.meshstream.bin");
            require(scene::buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = cache,
                .meshletOptions = {.maxWorkers = 1}}, log), log);
            graph.addNode("GPUDrivenStreamAssetPass", "GPUDriven", {
                {"path", source.generic_string()}, {"streamAssetPath", cache.generic_string()},
                {"streamAssetOnly", true}, {"maxResidentPages", 64}, {"maxLockedFallbackPages", 64},
                {"maxActiveGroups", 1024}, {"maxPageUploadsPerFrame", 64},
                {"maxGpuPageRequests", 1024}, {"maxGpuPageUnloadRequests", 1024},
                {"maxTraversalWorkers", 64}, {"maxTraversalWorkItems", 4096},
                {"clusterNormalConeCull", false}, {"debugColorMode", "shaded"},
                {"camera", {{"eye", {-.0168404f, .110154f, .22f}},
                    {"center", {-.0168404f, .110154f, -.00153695f}},
                    {"up", {0, 1, 0}}, {"znear", .001f}, {"zfar", 10.f}, {"fovDegrees", 60.f}}}});
        }
        const auto node = graph.findNode("GPUDriven")->id;
        if (unified) {
            graph.findNode(node)->type = "VisibilityBufferPass";
            auto& properties = graph.findNode(node)->properties;
            properties["sceneBinding"] = "asset";
            properties["meshletNormalConeCull"] = false;
            properties["visualization"] = "triangle";
            if (!miniZorah) {
                graph.addNode("VisibilityBufferMaterialPass", "MaterialResolve", {});
                graph.addEdge("GPUDriven.visibility", "MaterialResolve.visibility");
                graph.addEdge("GPUDriven.rasterInfo", "MaterialResolve.rasterInfo");
            }
            graph.markOutput("MaterialResolve.color");
        }
        auto pass = createRenderGraphPass(unified ? "VisibilityBufferPass" : "GPUDrivenStreamAssetPass");
        pass->setProperties(graph.findNode(node)->properties);
        require(pass->sceneDependency().source == (unified ? RenderGraphSceneSource::World : RenderGraphSceneSource::None),
            "Stream-only pass still declares a Scene dependency");
        graph.markOutput("GPUDriven.color");
        graph.markOutput("GPUDriven.visibility");

        std::vector<scene::MeshletStreamInstanceInfo> instances;
        std::vector<uint32_t> rootPages;
        scene::Bounds worldBounds;
        {
            scene::MeshletStreamAsset asset;
            require(asset.open(cache, log) && asset.isCurrentForSource(source), "Invalid cooked cache: " + log);
            report["source"] = source.generic_string(); report["cache"] = cache.generic_string();
            report["primitives"] = asset.primitiveCount(); report["instances"] = asset.instanceCount();
            report["pages"] = asset.pageCount(); report["terminalPages"] = asset.terminalGroups().size();
            if (miniZorah) { require(asset.primitiveCount() == 3163 && asset.instanceCount() == 19144, "Incomplete MiniZorah cache"); }
            instances.assign(asset.instances().begin(), asset.instances().end());
            for (uint32_t index = 0; index < asset.primitiveCount(); ++index) {
                const auto roots = asset.primitiveTerminalGroups(index);
                require(roots.size() == 1 && asset.groups()[roots[0]].clusterCount == 1,
                    "This fixture's root coverage check expects one terminal cluster per primitive");
                rootPages.push_back(asset.groups()[roots[0]].pageIndex);
            }
            for (const auto& instance : instances) {
                const auto& bounds = asset.primitives()[instance.primitiveIndex].bounds;
                for (uint32_t corner = 0; corner < 8; ++corner) {
                    float p[3];
                    for (uint32_t axis = 0; axis < 3; ++axis) { p[axis] = (corner & (1u << axis)) ? bounds.max[axis] : bounds.min[axis]; }
                    const auto& m = instance.worldMatrix;
                    worldBounds.include(float3(m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12],
                        m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13], m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14]));
                }
            }
        }
        report["metadataCheckSeconds"] = seconds();
        const auto initialized = preview.initialize(context.enableValidation, false, false);
        if (hasError(initialized, Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders and descriptor heaps"); }
        require(bool(initialized), preview.lastLog());
        preview.setDebugObserver(&observer);
        uint32_t width = miniZorah ? 1920 : 256, height = miniZorah ? 1080 : 192;
        report["width"] = width; report["height"] = height;
        report["profile"] = graph.findNode(node)->properties;
        uint32_t frames = 0;
        const auto renderFrame = [&](const char* output = "GPUDriven.visibility") {
            const auto result = preview.render(graph, width, height, output);
            observer.debug.poll();
            require(bool(result), preview.lastLog());
            ++frames;
            if (frames == 1) { report["firstRenderCompleteSeconds"] = seconds(); }
            auto* gpuScene = preview.subsystemHost()->get<GPUSceneSubsystem>();
            if (unified) {
                require(gpuScene && gpuScene->instances().size() == instances.size(), "Incomplete global GPUScene instances");
                require(gpuScene->geometries().size() == report.at("primitives").get<size_t>(), "Incomplete global geometry identities");
                const auto& views = gpuScene->globalBufferViews();
                require(views.vertices.buffer == nullptr && views.indices.buffer == nullptr &&
                    views.meshletDraws.buffer == nullptr && views.lodGroups.buffer == nullptr &&
                    gpuScene->rasterDrawLayout().maxRangeCount == 0, "Stream metadata allocated resident payload");
                return;
            }
            require(gpuScene && gpuScene->instances().empty() && gpuScene->geometries().empty(),
                "Stream-only mode created resident GPUScene geometry or instances");
            require(gpuScene->materials().empty() && gpuScene->drawSet().generation != 0,
                "Stream-only view must have an empty but valid DrawSet");
        };
        const auto coveredPixels = [&]() {
            return std::count_if(preview.pixels().begin(), preview.pixels().end(), [](uint32_t value) { return value != 0u; });
        };
        for (uint32_t frame = 0; frame < 240; ++frame) {
            renderFrame();
            if (!report.contains("firstPixelSeconds") && coveredPixels() != 0) { report["firstPixelSeconds"] = seconds(); }
            if (frame % 8u == 0) {
                std::printf("[%s] frame=%u terminal=%u/%u time=%.2fs\n", label.c_str(), frames,
                    observer.latest.value("terminalResidentPageCount", 0u), uint32_t(rootPages.size()), seconds());
                std::fflush(stdout);
            }
            if (observer.latest.value("terminalReady", false)) { report["terminalReadySeconds"] = seconds(); break; }
        }
        require(report.contains("terminalReadySeconds") && report.contains("firstPixelSeconds"), "Terminal cut did not become drawable");
        const auto call = [&](const char* method, Json params) {
            auto response = observer.debug.core().dispatch({{"id", "stream-startup"}, {"method", method}, {"params", params}});
            require(response.value("status", "") == "ok", response.dump());
            return response.at("result");
        };
        const auto captureHeader = [&](bool allRoots) {
            Json resources = {{{"id", "streaming.GPUDriven.activeHeader"}, {"count", 1}}};
            if (allRoots) { resources.push_back({{"id", "streaming.GPUDriven.activeGroups"}, {"count", instances.size()}}); }
            const std::string job = call("capture.batch", {{"pass", "GPUDriven"}, {"checkpoint", "AfterTraversal"}, {"resources", resources}}).at("job");
            renderFrame();
            return job;
        };
        const auto read = [&](const std::string& job, const std::string& resource) {
            Json rows = Json::array();
            uint64_t total = 0;
            do {
                const auto page = call("eval", {{"job", job}, {"expression", "buffers[\"" + resource + "\"]"},
                    {"offset", rows.size()}, {"count", 4096}}).at("value");
                total = page.at("total");
                for (const auto& row : page.at("items")) { rows.push_back(row); }
            } while (rows.size() < total);
            return rows;
        };
        // Measure startup with the shipped automatic LOD profile first. Then
        // force the complete coarse cut to verify every cached instance on GPU.
        graph.setNodeRuntimeProperty(node, "autoLod", false);
        graph.setNodeRuntimeProperty(node, "lodLevel", 31);
        const auto rootJob = captureHeader(true);
        const auto header = read(rootJob, "streaming.GPUDriven.activeHeader").at(0);
        require(header.at("activeGroupCount") == instances.size() && header.at("overflowCount") == 0,
            "GPU terminal cut is incomplete: " + header.dump());
        const auto rows = read(rootJob, "streaming.GPUDriven.activeGroups");
        std::vector<bool> seen(instances.size());
        for (const auto& row : rows) {
            const uint32_t index = row.at("instanceIndex");
            require(index < instances.size() && !seen[index], "Invalid or duplicate stream instance ID");
            seen[index] = true;
            const auto& expected = instances[index];
            require(row.at("gpuSceneInstanceIndex") == index && row.at("primitiveIndex") == expected.primitiveIndex &&
                row.at("materialIndex") == expected.materialIndex && row.at("pageIndex") == rootPages[expected.primitiveIndex] &&
                row.at("clusterSelectionMask") == 1u, "GPU root instance mapping differs from the cache");
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t c = 0; c < 4; ++c) {
                    require(std::abs(row.at("world").at(r*4+c).get<float>() - expected.worldMatrix[r*4+c]) < 1e-5f,
                        "GPU instance transform differs from the cache");
                }
            }
        }
        require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "Missing GPU root instances");
        report["gpuRootCoverage"] = {{"verifiedInstances", seen.size()}, {"header", header}};
        if (unified) {
            auto* gpuScene = preview.subsystemHost()->get<GPUSceneSubsystem>();
            uint32_t blendInstances = 0;
            for (const auto& instance : gpuScene->instances()) {
                const auto& material = gpuScene->materials()[instance.material.index];
                require(instance.drawKey.bucket != GPUSceneDrawBucket::Blend, "Effective-opaque instance was dropped as BLEND");
                if (material.material.alphaMode == "BLEND") { ++blendInstances; }
            }
            if (miniZorah) { require(blendInstances == 606, "Lost MiniZorah constant-alpha BLEND instances"); }
            report["effectiveOpaqueBlendInstances"] = blendInstances;
        }
        // Freeze the cut and remove history dependence for producer comparison.
        if (unified) { graph.setNodeRuntimeProperty(node, "hybridRaster", false); }
        graph.setNodeRuntimeProperty(node, "instanceHzbCull", false);
        graph.setNodeRuntimeProperty(node, "meshletHzbCull", false);
        for (uint32_t frame = 0; frame < 3; ++frame) { renderFrame(); }
        std::vector<uint32_t> hardware(preview.pixels().begin(), preview.pixels().end());
        {
            std::ofstream output(context.outputDirectory / (label + "-root-visibility.bin"), std::ios::binary);
            output.write(reinterpret_cast<const char*>(hardware.data()), hardware.size() * sizeof(uint32_t));
        }
        if (unified) {
            const auto resolveNode = graph.findNode("MaterialResolve")->id;
            graph.setNodeRuntimeProperty(resolveNode, "visualization", "baseColor");
            renderFrame("MaterialResolve.color");
            auto* gpuScene = preview.subsystemHost()->get<GPUSceneSubsystem>();
            uint32_t maxColorError = 0;
            uint64_t checkedPixels = 0;
            for (size_t pixel = 0; pixel < hardware.size(); ++pixel) {
                if (hardware[pixel] == 0) { continue; }
                const uint32_t group = ((hardware[pixel] >> kVisibilityTriangleBits) - 1u) /
                    header.at("maxActiveGroupClusters").get<uint32_t>();
                require(group < rows.size(), "Visibility ID outside the terminal frontier");
                const uint32_t instanceId = rows[group].at("gpuSceneInstanceIndex");
                const auto& instance = gpuScene->instances()[instanceId];
                const auto base = gpuScene->materials()[instance.material.index].material.baseColorFactor;
                const float components[] = {base.x, base.y, base.z};
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const int expected = int(std::round(std::pow(std::clamp(components[channel], 0.f, 1.f), 1.f / 2.2f) * 255));
                    const int actual = (preview.pixels()[pixel] >> (channel * 8)) & 255;
                    maxColorError = std::max(maxColorError, uint32_t(std::abs(expected - actual)));
                }
                ++checkedPixels;
            }
            require(checkedPixels > 64 && maxColorError <= 1, "Scalar resolve does not preserve source base color");
            report["baseColorValidation"] = {{"checkedPixels", checkedPixels}, {"maxByteError", maxColorError}};
            graph.setNodeRuntimeProperty(resolveNode, "visualization", "shaded");
            graph.setNodeRuntimeProperty(node, "hybridRaster", true);
            graph.setNodeRuntimeProperty(node, "asyncSoftwareRaster", true);
            graph.setNodeRuntimeProperty(node, "softwareRasterMaxPixels", 64.0f);
            for (uint32_t frame = 0; frame < 3; ++frame) { renderFrame(); }
            uint64_t mismatch = 0, coverageMismatch = 0, interiorMismatch = 0;
            for (size_t pixel = 0; pixel < hardware.size(); ++pixel) {
                mismatch += hardware[pixel] != preview.pixels()[pixel];
                coverageMismatch += (hardware[pixel] == 0) != (preview.pixels()[pixel] == 0);
                if (hardware[pixel] != preview.pixels()[pixel] && pixel >= width && pixel + width < hardware.size() &&
                    pixel % width != 0 && pixel % width != width - 1 &&
                    hardware[pixel - 1] == hardware[pixel] && hardware[pixel + 1] == hardware[pixel] &&
                    hardware[pixel - width] == hardware[pixel] && hardware[pixel + width] == hardware[pixel] &&
                    preview.pixels()[pixel - 1] == preview.pixels()[pixel] && preview.pixels()[pixel + 1] == preview.pixels()[pixel] &&
                    preview.pixels()[pixel - width] == preview.pixels()[pixel] && preview.pixels()[pixel + width] == preview.pixels()[pixel]) {
                    ++interiorMismatch;
                }
            }
            report["hardwareHybridComparison"] = {{"pixels", hardware.size()}, {"idMismatch", mismatch},
                {"coverageMismatch", coverageMismatch}, {"interiorIdMismatch", interiorMismatch},
                {"asyncComputeBranches", preview.executionStats().asyncComputeBranches}};
            {
                std::ofstream output(context.outputDirectory / (label + "-root-hybrid.bin"), std::ios::binary);
                output.write(reinterpret_cast<const char*>(preview.pixels().data()), hardware.size() * sizeof(uint32_t));
            }
            require(mismatch <= std::max<size_t>(8, hardware.size() / 1000), "HW/async hybrid visibility mismatch");
            graph.setNodeRuntimeProperty(node, "softwareRasterMaxPixels", 8.0f);
        }
        graph.setNodeRuntimeProperty(node, "instanceHzbCull", true);
        graph.setNodeRuntimeProperty(node, "meshletHzbCull", true);
        std::printf("[%s] verified all %zu GPU root instances at %.2fs\n", label.c_str(), seen.size(), seconds());
        std::fflush(stdout);
        graph.setNodeRuntimeProperty(node, "autoLod", true);
        graph.setNodeRuntimeProperty(node, "lodPixelError", 1.5f);
        const auto originalCamera = graph.findNode(node)->properties.at("camera");
        const float3 center = worldBounds.center();
        const float radius = std::max(worldBounds.radius(), 1.0f);
        report["worldBounds"] = {{"min", {worldBounds.min.x, worldBounds.min.y, worldBounds.min.z}},
            {"max", {worldBounds.max.x, worldBounds.max.y, worldBounds.max.z}}, {"radius", radius}};
        Json cameras = Json::array({{{"name", "original"}, {"camera", originalCamera}}});
        if (miniZorah) {
            auto farCamera = originalCamera;
            farCamera["eye"] = {center.x + radius * 2, center.y + radius * 1.4f, center.z + radius * 2};
            farCamera["center"] = {center.x, center.y, center.z};
            farCamera["znear"] = std::max(originalCamera["znear"].get<float>(), radius * .001f);
            farCamera["zfar"] = radius * 5;
            cameras.push_back({{"name", "far"}, {"camera", farCamera}});
            auto nearCamera = originalCamera;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                nearCamera["eye"][axis] = originalCamera["eye"][axis].get<float>() * .7f + originalCamera["center"][axis].get<float>() * .3f;
            }
            cameras.push_back({{"name", "near"}, {"camera", nearCamera}});
        }
        for (const auto& camera : cameras) {
            report["currentCamera"] = camera;
            graph.setNodeRuntimeProperty(node, "camera", camera.at("camera"));
            for (uint32_t frame = 0; frame < (miniZorah ? 48u : 16u); ++frame) { renderFrame(); }
            const auto job = captureHeader(false);
            const auto active = read(job, "streaming.GPUDriven.activeHeader").at(0);
            const uint64_t covered = coveredPixels();
            require(covered > 64, "Camera produced no useful geometry coverage: " + camera.dump() + " active=" + active.dump());
            require(observer.latest.value("terminalReady", false), "Terminal pages lost while changing camera");
            Json result{{"name", camera.at("name")}, {"camera", camera.at("camera")},
                {"coveredPixels", covered}, {"activeHeader", active}, {"streaming", observer.latest}};
            renderFrame(unified ? "MaterialResolve.color" : "GPUDriven.color");
            const auto imagePath = context.outputDirectory / (label + "-" + camera.at("name").get<std::string>() + ".png");
            require(saveRgba8Png(imagePath, reinterpret_cast<const uint8_t*>(preview.pixels().data()), width, height, log), log);
            result["image"] = imagePath.generic_string();
            report["cases"].push_back(std::move(result));
            std::printf("[%s] camera=%s pixels=%llu groups=%u time=%.2fs\n", label.c_str(),
                camera.at("name").get<std::string>().c_str(), static_cast<unsigned long long>(covered),
                active.at("activeGroupCount").get<uint32_t>(), seconds());
            std::fflush(stdout);
        }
        require(!observer.latest.value("clusterRtxEnabled", true), "Unexpected CLAS/BLAS path");
        require(observer.latest.at("stats").at("totalPageLoadFailureCount") == 0 &&
            observer.latest.at("stats").at("totalGpuInvalidRequestCount") == 0, "Stream page load/request failure");
        if (unified) {
            width = 321; height = 197;
            renderFrame("MaterialResolve.color");
            require(preview.pixels().size() == size_t(width) * height, "Resize failed");
            // Remove the producer and its borrowed stream resources, then reopen.
            RenderGraph empty;
            empty.addNode("FinalBlitPass", "Empty", {});
            require(bool(preview.render(empty, width, height, "Empty.color")), preview.lastLog());
            auto* gpuScene = preview.subsystemHost()->get<GPUSceneSubsystem>();
            require(gpuScene->instances().empty(), "Scene source lease survived producer removal");
            graph.setNodeRuntimeProperty(node, "autoLod", false);
            graph.setNodeRuntimeProperty(node, "lodLevel", 31);
            for (uint32_t frame = 0; frame < 64; ++frame) {
                renderFrame();
                if (observer.latest.value("terminalReady", false) && coveredPixels() > 0) { break; }
            }
            require(observer.latest.value("terminalReady", false) && coveredPixels() > 0, "Scene reopen failed");
            report["resizeReleaseReopen"] = "passed";
        }
        report["globalGPUSceneInstances"] = unified ? instances.size() : 0;
        report["residentVertexBytes"] = 0; report["residentMeshletDrawBytes"] = 0;
        report["frames"] = frames; report["status"] = "passed";
        saveReport();
        return RhiTestResult::pass(label + " complete root coverage and first frames verified");
    } catch (const std::exception& error) {
        report["status"] = "failed"; report["error"] = error.what();
        saveReport();
        return RhiTestResult::fail(error.what());
    }
}

class StreamAssetOnlyFirstFrameTest final : public RhiTest {
public:
    StreamAssetOnlyFirstFrameTest() { type = RhiTestType::Rendering; name = "streamasset_only_first_frame"; }
    RhiTestResult run(RhiTestContext& context) override { return runStreamStartup(context, false); }
};
class MiniZorahFirstFrameTest final : public RhiTest {
public:
    MiniZorahFirstFrameTest() { type = RhiTestType::Rendering; name = "minizorah_stream_first_frame"; }
    RhiTestResult run(RhiTestContext& context) override { return runStreamStartup(context, true); }
};
METALLIC_REGISTER_RHI_TEST(StreamAssetOnlyFirstFrameTest);
METALLIC_REGISTER_RHI_TEST(MiniZorahFirstFrameTest);
class StreamMetadataVBufferTest final : public RhiTest {
public:
    StreamMetadataVBufferTest() { type = RhiTestType::Rendering; name = "stream_metadata_vbuffer"; }
    RhiTestResult run(RhiTestContext& context) override { return runStreamStartup(context, false, true); }
};
class MiniZorahVBufferTest final : public RhiTest {
public:
    MiniZorahVBufferTest() { type = RhiTestType::Rendering; name = "minizorah_vbuffer"; }
    RhiTestResult run(RhiTestContext& context) override { return runStreamStartup(context, true, true); }
};
METALLIC_REGISTER_RHI_TEST(StreamMetadataVBufferTest);
METALLIC_REGISTER_RHI_TEST(MiniZorahVBufferTest);

class StreamMetadataContractTest final : public RhiTest {
public:
    StreamMetadataContractTest() { type = RhiTestType::Rendering; name = "stream_metadata_contract"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        try {
            const auto path = std::filesystem::absolute(context.outputDirectory / "MetadataOnly.gltf");
            Json metadata = {{"asset", {{"version", "2.0"}}}, {"scene", 0},
                {"extensionsRequired", {"EXT_meshopt_compression"}},
                {"scenes", {{{"nodes", {0, 1}}}}},
                {"nodes", {{{"mesh", 0}}, {{"mesh", 0}, {"translation", {2, 0, 0}}}}},
                {"buffers", {{{"uri", "intentionally-absent-10gb.bin"}, {"byteLength", 10000000000ull}}}},
                {"bufferViews", {{{"buffer", 0}, {"byteOffset", 5000000000ull}, {"byteLength", 36}}}},
                {"accessors", {{{"bufferView", 0}, {"componentType", 5126}, {"count", 3},
                    {"type", "VEC3"}, {"min", {0, 0, 0}}, {"max", {1, 1, 0}}}}},
                {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}}}, {"material", 0}}}}}}},
                {"materials", {{{"alphaMode", "BLEND"}, {"doubleSided", true},
                    {"pbrMetallicRoughness", {{"baseColorFactor", {.2, .4, .6, 1.0}}}}}}}};
            { std::ofstream file(path); file << metadata.dump(); }
            scene::Scene source;
            require(source.loadStreamMetadata(path), source.lastLoadResult().error);
            require(source.valid() && source.hasStreamGeometry() && source.renderNodes().size() == 2, "Metadata scene topology");
            require(source.bounds().max.x == 3 && source.materials()[0].alphaMode == "BLEND", "Metadata transforms/material preservation");
            for (const auto& primitive : source.renderPrimitives()) {
                require(primitive.storage == scene::GeometryStorage::StreamAsset && primitive.positions.empty() &&
                    primitive.indices.empty() && primitive.meshletLodClusters.empty() && primitive.triangleCount == 1, "Metadata imported geometry");
            }
            GPUScene gpuScene;
            std::string log;
            auto view = GPUSceneSourceView::fromScene(source);
            require(bool(gpuScene.rebuild(view, log)), log);
            require(gpuScene.geometries().size() == 1 && gpuScene.instances().size() == 2 &&
                gpuScene.materials()[0].bucket == GPUSceneDrawBucket::OpaqueDoubleSided, "Global identity or constant-alpha classification");
            const auto id = gpuScene.instanceForRenderNode(1);
            require(gpuScene.sync(view) != GPUSceneSyncResult::RebuildRequired && gpuScene.instanceForRenderNode(1) == id, "Unstable identity");
            view.constantAlphaOpaque = false;
            require(gpuScene.sync(view) == GPUSceneSyncResult::RebuildRequired, "Classification mode change was ignored");
            auto material = source.materials()[0]; material.baseColorFactor.w = .5f;
            require(classifyGPUSceneMaterial(material, true) == GPUSceneDrawBucket::Blend, "Real transparency incorrectly promoted");
            SceneResourceManager manager;
            const scene::Scene* resolved = nullptr;
            require(bool(manager.resolveScene({{"path", path.generic_string()}}, &source, resolved, log)) && resolved == &source,
                "Metadata source fell back to resident import");
            metadata["images"] = {{{"uri", "unsupported.png"}}};
            { std::ofstream file(path); file << metadata.dump(); }
            require(!source.loadStreamMetadata(path) && !source.valid() && !source.hasStreamGeometry(), "Unsupported metadata load retained stale scene");
            metadata.erase("images");
            metadata.erase("extensionsRequired");
            metadata["scenes"][0]["nodes"] = {0};
            metadata["buffers"][0] = {{"uri", "ScalarTriangle.bin"}, {"byteLength", 36}};
            metadata["bufferViews"][0]["byteOffset"] = 0;
            const float positions[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
            { std::ofstream file(path.parent_path() / "ScalarTriangle.bin", std::ios::binary); file.write(reinterpret_cast<const char*>(positions), sizeof(positions)); }
            { std::ofstream file(path); file << metadata.dump(); }
            const auto cache = path.parent_path() / "ScalarTriangle.meshstream.bin";
            require(scene::buildMeshletStreamAssetOffline({.sourcePath = path, .outputPath = cache}, log), log);
            require(source.loadStreamMetadata(path), source.lastLoadResult().error);
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&source);
            require(bool(preview.initialize(context.enableValidation, false, false)), preview.lastLog());
            RenderGraph graph;
            graph.addNode("VisibilityBufferPass", "Raster", {{"path", path.generic_string()}, {"streamAssetPath", cache.generic_string()},
                {"streamAssetOnly", true}, {"maxResidentPages", 4}, {"maxLockedFallbackPages", 4},
                {"maxActiveGroups", 64}, {"maxGpuPageRequests", 64}, {"maxTraversalWorkers", 32}, {"maxTraversalWorkItems", 128},
                {"autoLod", false}, {"instanceHzbCull", false}, {"meshletHzbCull", false}, {"meshletNormalConeCull", true},
                {"hybridRaster", false}, {"camera", {{"eye", {.25, .25, -2}}, {"center", {.25, .25, 0}}, {"up", {0, 1, 0}},
                    {"znear", .01}, {"zfar", 10}, {"fovDegrees", 60}}}});
            graph.markOutput("Raster.visibility");
            const auto render = [&]() { require(bool(preview.render(graph, 128, 128, "Raster.visibility")), preview.lastLog()); };
            for (uint32_t frame = 0; frame < 16; ++frame) { render(); }
            const auto hardware = preview.pixels();
            require(std::count_if(hardware.begin(), hardware.end(), [](uint32_t id) { return id != 0; }) > 128,
                "Double-sided back faces were culled by hardware or normal cone");
            const auto raster = graph.findNode("Raster")->id;
            graph.setNodeRuntimeProperty(raster, "hybridRaster", true);
            graph.setNodeRuntimeProperty(raster, "asyncSoftwareRaster", true);
            graph.setNodeRuntimeProperty(raster, "softwareRasterMaxPixels", 1024.f);
            render();
            require(hardware == preview.pixels(), "Double-sided HW/SW coverage or IDs differ");
            auto singleSided = source.materials()[0]; singleSided.doubleSided = false;
            require(source.setMaterialProperties(0, singleSided), "Material edit failed");
            for (uint32_t frame = 0; frame < 16; ++frame) { render(); }
            require(std::all_of(preview.pixels().begin(), preview.pixels().end(), [](uint32_t id) { return id == 0; }),
                "Single-sided back faces survived after material edit");
            // Exercise both halves of the independent material consumer: first
            // ordinary resident records, then streamed IDs with a nonzero base.
            require(source.load(path), source.lastLoadResult().error);
            auto opaque = source.materials()[0]; opaque.alphaMode = "OPAQUE";
            require(source.setMaterialProperties(0, opaque), "Resident fixture material edit failed");
            graph.setNodeRuntimeProperty(raster, "streamAssetOnly", false);
            graph.setNodeRuntimeProperty(raster, "enableMeshletStreaming", false);
            graph.addNode("VisibilityBufferMaterialPass", "Resolve", {{"visualization", "baseColor"}});
            graph.addEdge("Raster.visibility", "Resolve.visibility");
            graph.addEdge("Raster.rasterInfo", "Resolve.rasterInfo");
            graph.markOutput("Resolve.color");
            require(bool(preview.render(graph, 128, 128, "Resolve.color")), preview.lastLog());
            const auto residentColors = preview.pixels();
            const auto expectedChannel = [](float value) { return uint32_t(std::round(std::pow(value, 1.f / 2.2f) * 255)); };
            const uint32_t expectedColor = expectedChannel(.2f) | (expectedChannel(.4f) << 8u) |
                (expectedChannel(.6f) << 16u) | 0xff000000u;
            require(std::count(residentColors.begin(), residentColors.end(), expectedColor) > 128,
                "Resident material resolve did not preserve base color");
            graph.setNodeRuntimeProperty(raster, "enableMeshletStreaming", true);
            for (uint32_t frame = 0; frame < 16; ++frame) {
                require(bool(preview.render(graph, 128, 128, "Resolve.color")), preview.lastLog());
            }
            auto* subsystem = preview.subsystemHost()->get<GPUSceneSubsystem>();
            require(subsystem->rasterDrawLayout().maxRangeCount > 0, "Mixed fixture lost its resident layout");
            require(residentColors == preview.pixels(), "Resident and mixed-stream material resolve differ");
            return RhiTestResult::pass("Metadata import never reads external buffers, preserves identity, and classifies only constant opaque alpha");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMetadataContractTest);
} // namespace
} // namespace metallic::tests
