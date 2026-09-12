#include "RhiTest.h"
#include "Runtime/Render/ResidentMeshletLod.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {

using namespace render;

struct LodFixture {
    scene::RenderPrimitive primitive;
    std::vector<MeshletLodGroupRecord> groups;
    std::vector<GPUSceneGpuMeshletRecord> clusters;
    std::vector<GPUSceneGpuInstanceRecord> instances;
    std::vector<VisibleClusterRecord> candidates;
};

LodFixture makeLodFixture()
{
    LodFixture f;
    f.primitive.positions = {float3(0, 0, 0), float3(1, 0, 0), float3(0, 1, 0)};
    f.primitive.meshletLodVertices = {0, 1, 2};
    f.primitive.meshletLodTriangles = {0, 1, 2};
    const uint32_t counts[] = {2, 2, 1, 3, 1};
    const uint32_t levels[] = {0, 0, 0, 1, 2};
    const float errors[] = {.1f, .5f, .02f, 2.f, 2.5f};
    for (uint32_t g = 0; g < 5; ++g) {
        scene::MeshletLodGroup group;
        group.clusterOffset = static_cast<uint32_t>(f.clusters.size());
        group.clusterCount = counts[g];
        group.lodLevel = levels[g];
        group.maxQuadricError = errors[g];
        group.boundingSphereCenter = float3(g == 0 ? -2.f : g == 1 ? 2.f : 0.f, 0, 0);
        group.boundingSphereRadius = g < 3 ? 1.f : 5.f;
        f.primitive.meshletLodGroups.push_back(group);
        for (uint32_t c = 0; c < counts[g]; ++c) {
            const uint32_t refined = g == 3 ? (c < 2 ? 0u : 1u) : g == 4 ? 3u : UINT32_MAX;
            scene::MeshletCluster cluster;
            cluster.vertexCount = 3;
            cluster.triangleCount = 1;
            cluster.lodLevel = levels[g];
            cluster.lodGroupIndex = g;
            cluster.refinedGroupIndex = static_cast<int32_t>(refined);
            f.primitive.meshletLodClusters.push_back(cluster);
            GPUSceneGpuMeshletRecord gpu;
            gpu.lod = {levels[g], g, refined, 0};
            f.clusters.push_back(gpu);
        }
    }
    GPUSceneGpuInstanceRecord instance;
    instance.worldMatrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    instance.identity[3] = GPUSceneGpuInstanceVisible;
    f.instances.push_back(instance);
    for (uint32_t c = 0; c < f.clusters.size(); ++c) { f.candidates.push_back({c, 0, 7, 0}); }
    return f;
}

class MeshletLodReferenceTest final : public RhiTest {
public:
    MeshletLodReferenceTest() { type = RhiTestType::Validation; name = "meshlet_lod_reference_cut_and_metadata"; }
    RhiTestResult run(RhiTestContext&) override
    {
        auto f = makeLodFixture();
        std::string reason;
        if (!buildMeshletLodMetadata(f.primitive, f.groups, reason)) { return RhiTestResult::fail(reason); }
        MeshletLodView view;
        view.forward[3] = 1;
        view.projection = {1000, .5f, 100, 1.5f};
        const auto check = [&](std::initializer_list<uint32_t> expected, uint32_t manual = UINT32_MAX) {
            auto result = selectMeshletLodReference(f.groups, f.clusters, f.instances, f.candidates, view, 17, manual);
            std::vector<uint32_t> ids;
            for (const auto& selected : result) {
                if (selected.instanceIndex != 0 || selected.geometryIndex != 7 || selected.recordIndex != selected.clusterIndex + 17) { return false; }
                ids.push_back(selected.clusterIndex);
            }
            return ids == std::vector<uint32_t>(expected);
        };
        if (!check({2, 3, 4, 5, 6}) || !check({0, 1, 2, 3, 4}, 0) ||
            !check({4, 5, 6, 7}, 1) || !check({4, 8}, 31)) {
            return RhiTestResult::fail("mixed cut, shared generating group or early terminal coverage is incorrect");
        }
        view.projection[3] = .5f;
        if (!check({0, 1, 2, 3, 4})) { return RhiTestResult::fail("smaller pixel error did not refine"); }
        view.projection[3] = 8;
        if (!check({4, 5, 6, 7})) { return RhiTestResult::fail("larger pixel error did not coarsen"); }
        view.projection[3] = 30;
        if (!check({4, 8})) { return RhiTestResult::fail("root cut omitted an early terminal"); }
        view.forward[3] = 0;
        view.eye = {0, 0, 100, .1f};
        auto& instance = f.instances.front();
        float base = meshletLodPixelError(f.groups[0], instance, view);
        instance.worldMatrix[4] = 3;
        float sheared = meshletLodPixelError(f.groups[0], instance, view);
        if (!(sheared > base)) { return RhiTestResult::fail("shear is not included in projected error"); }
        view.eye[2] = .1f;
        if (meshletLodPixelError(f.groups[0], instance, view) != FLT_MAX) { return RhiTestResult::fail("near-plane intersection did not request detail"); }
        f.primitive.meshletLodClusters.back().refinedGroupIndex = 4;
        if (buildMeshletLodMetadata(f.primitive, f.groups, reason) || !f.groups.empty()) {
            return RhiTestResult::fail("cyclic metadata was accepted");
        }
        return RhiTestResult::pass("mixed-depth roots, many-to-one refinement, pixel threshold, manual cuts, shear and invalid DAG");
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletLodReferenceTest);

#define LOD_REQUIRE(expr) do { auto checked = (expr); if (!checked) { return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

class MeshletLodGpuTest final : public RhiTest {
public:
    MeshletLodGpuTest() { type = RhiTestType::Rendering; name = "meshlet_lod_gpu_matches_reference"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        auto f = makeLodFixture();
        if (!buildMeshletLodMetadata(f.primitive, f.groups, log)) { return RhiTestResult::fail(log); }
        for (uint32_t i = 1; i < 1027; ++i) {
            auto instance = f.instances.front();
            instance.worldMatrix[0] = i % 2 ? -1.5f : 1.f;
            instance.worldMatrix[4] = float(i % 4) * .3f;
            instance.worldMatrix[5] = .5f + float(i % 3);
            instance.worldMatrix[12] = float(i) * .7f;
            instance.worldMatrix[14] = -float(i) * 1.3f;
            if (i % 9 == 0) { instance.identity[3] = 0; }
            f.instances.push_back(instance);
            for (uint32_t c = 0; c < f.clusters.size(); ++c) { f.candidates.push_back({c, i, 7, 0}); }
        }
        std::unique_ptr<Device> device;
        auto created = createDevice({.applicationName = "Resident meshlet LOD regression", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless compute"); }
        LOD_REQUIRE(created);
        std::unique_ptr<BindlessHeap> heap;
        LOD_REQUIRE(device->createBindlessHeap({.maxBuffers = 7}, heap));
        std::array<std::unique_ptr<Buffer>, 4> buffers;
        GPUSceneConsumerBindings bindings;
        const GPUSceneGlobalBufferKind kinds[] = {GPUSceneGlobalBufferKind::LodGroups, GPUSceneGlobalBufferKind::Meshlets,
            GPUSceneGlobalBufferKind::Instances, GPUSceneGlobalBufferKind::MeshletDraws};
        const void* data[] = {f.groups.data(), f.clusters.data(), f.instances.data(), f.candidates.data()};
        const uint32_t strides[] = {32, 80, 160, 16};
        const size_t counts[] = {f.groups.size(), f.clusters.size(), f.instances.size(), f.candidates.size()};
        for (uint32_t i = 0; i < 4; ++i) {
            LOD_REQUIRE(device->createBuffer({.size = counts[i] * strides[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, buffers[i]));
            void* mapped = buffers[i]->map();
            if (!mapped) { return RhiTestResult::fail("input map"); }
            std::memcpy(mapped, data[i], counts[i] * strides[i]); buffers[i]->flush(); buffers[i]->unmap();
            auto& handle = bindings.buffers[static_cast<size_t>(kinds[i])];
            LOD_REQUIRE(heap->allocateBuffer(handle)); LOD_REQUIRE(heap->writeStorageBuffer(handle, *buffers[i]));
        }
        ResidentMeshletLod selector;
        const uint32_t capacity = static_cast<uint32_t>(f.candidates.size());
        LOD_REQUIRE(selector.initialize(*device, capacity, log));
        BindlessHandle selections, arguments, scratch;
        LOD_REQUIRE(heap->allocateBuffer(selections)); LOD_REQUIRE(heap->writeStorageBuffer(selections, selector.selections()));
        LOD_REQUIRE(heap->allocateBuffer(arguments)); LOD_REQUIRE(heap->writeStorageBuffer(arguments, selector.arguments()));
        LOD_REQUIRE(heap->allocateBuffer(scratch)); LOD_REQUIRE(heap->writeStorageBuffer(scratch, selector.scratch()));
        std::unique_ptr<Buffer> readback;
        const uint64_t selectionBytes = (uint64_t(capacity) + 1u) * 16u;
        LOD_REQUIRE(device->createBuffer({.size = selectionBytes + 24, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}, readback));
        Queue* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        LOD_REQUIRE(device->createCommandPool(*queue, pool));
        LOD_REQUIRE(pool->createCommandBuffer(commands)); LOD_REQUIRE(device->createFence(false, fence));
        for (uint32_t test = 0; test < 24; ++test) {
            MeshletLodView view;
            view.eye = {float(test % 3) * 20, 1, float(test % 5) * 25, .1f};
            view.forward[3] = test % 2 ? 1.f : 0.f;
            view.projection = {test % 3 ? 1080.f : 540.f, .577350269f, 100.f, .2f + float(test % 7)};
            uint32_t manual = test >= 20 ? test - 20 : UINT32_MAX;
            const uint32_t count = test == 19 ? 0u : capacity;
            auto expected = selectMeshletLodReference(f.groups, f.clusters, f.instances,
                std::span(f.candidates).first(count), view, 0, manual);
            if (test != 0) { LOD_REQUIRE(fence->reset()); LOD_REQUIRE(pool->reset()); }
            LOD_REQUIRE(commands->begin());
            if (test == 0) {
                for (auto& buffer : buffers) {
                    BufferBarrierDesc barrier{.buffer = buffer.get(), .before = ResourceState::Undefined, .after = ResourceState::ShaderRead};
                    commands->barrier({.buffers = &barrier, .bufferCount = 1});
                }
            }
            LOD_REQUIRE(selector.record(*commands, *heap, bindings, view, {0, count},
                static_cast<uint32_t>(f.instances.size()), static_cast<uint32_t>(f.groups.size()), selections, arguments, scratch, manual));
            BufferBarrierDesc barriers[] = {
                {.buffer = &selector.selections(), .before = ResourceState::ShaderRead, .after = ResourceState::TransferSource},
                {.buffer = &selector.arguments(), .before = ResourceState::IndirectArgument, .after = ResourceState::TransferSource}};
            commands->barrier({.buffers = barriers, .bufferCount = 2});
            commands->copyBuffer({.source = &selector.selections(), .destination = readback.get(), .size = selectionBytes});
            commands->copyBuffer({.source = &selector.arguments(), .destination = readback.get(), .destinationOffset = selectionBytes, .size = 24});
            for (auto& barrier : barriers) { std::swap(barrier.before, barrier.after); }
            commands->barrier({.buffers = barriers, .bufferCount = 2});
            LOD_REQUIRE(commands->end());
            CommandBuffer* list[] = {commands.get()};
            LOD_REQUIRE(queue->submit({.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()}));
            LOD_REQUIRE(fence->wait());
            readback->invalidate();
            auto* mapped = static_cast<const uint32_t*>(readback->map());
            if (!mapped) { return RhiTestResult::fail("selection readback map"); }
            uint32_t actualCount = mapped[0], overflow = mapped[3];
            if (actualCount > capacity || overflow != 0) { readback->unmap(); return RhiTestResult::fail("selection overflow"); }
            std::vector<MeshletLodSelection> actual(actualCount);
            std::memcpy(actual.data(), mapped + 4, actualCount * sizeof(MeshletLodSelection));
            const uint32_t* args = mapped + selectionBytes / 4u;
            bool validArguments = args[0] == actualCount && args[1] == 1 && args[2] == 1 &&
                args[3] == (actualCount + 31) / 32 && args[4] == 1 && args[5] == 1;
            readback->unmap();
            if (actual != expected || !validArguments) {
                return RhiTestResult::fail("CPU/GPU LOD mismatch in case " + std::to_string(test) +
                    ": expected " + std::to_string(expected.size()) + ", actual " + std::to_string(actual.size()));
            }
        }
        return RhiTestResult::pass("24 GPU/reference comparisons: perspective/ortho, near plane, render resolution, shear/reflection, hidden instances, empty cut, manual LOD and indirect arguments");
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletLodGpuTest);

class MeshletLodSceneTest final : public RhiTest {
public:
    MeshletLodSceneTest() { type = RhiTestType::Rendering; name = "meshlet_lod_resident_scene_runtime_cut"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using debug::DebugValue;
        RenderDebugRuntime debug;
        RenderGraphPreviewRenderer preview;
        auto initialized = preview.initialize(context.enableValidation, false, false);
        if (hasError(initialized, Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders"); }
        if (!initialized) { return RhiTestResult::fail(preview.lastLog()); }
        preview.setDebugObserver(&debug);
        RenderGraph graph;
        graph.addNode("VisibilityBufferPass", "VBuffer", {{"path", "Asset/StandfordBunny/scene.gltf"},
            {"camera", {{"eye", {-.0168404f, .110154f, .22f}}, {"center", {-.0168404f, .110154f, -.00153695f}},
                {"znear", .001f}, {"zfar", 10.f}, {"orthoHeight", .24f}, {"fovDegrees", 60.f}}}});
        graph.markOutput("VBuffer.visibility");
        graph.markOutput("VBuffer.color");
        const auto node = graph.findNode("VBuffer")->id;
        const auto render = [&]() {
            const auto result = preview.render(graph, 193, 157);
            debug.poll();
            if (!result) { throw std::runtime_error(preview.lastLog()); }
        };
        const auto call = [&](const char* method, DebugValue params) {
            auto response = debug.core().dispatch({{"id", "lod-test"}, {"method", method}, {"params", params}});
            if (response.value("status", "") != "ok") { throw std::runtime_error(response.dump()); }
            return response.at("result");
        };
        const auto capture = [&](const char* checkpoint, DebugValue resources) -> std::string {
            return call("capture.batch", {{"pass", "VBuffer"}, {"checkpoint", checkpoint}, {"resources", resources}}).at("job");
        };
        const auto read = [&](const std::string& job, const std::string& id) {
            auto rows = DebugValue::array();
            uint64_t total = 0;
            do {
                auto page = call("eval", {{"job", job}, {"expression", "buffers[\"" + id + "\"]"},
                    {"offset", rows.size()}, {"count", 4096}}).at("value");
                total = page.at("total");
                for (auto& row : page.at("items")) { rows.push_back(std::move(row)); }
            } while (rows.size() < total);
            return rows;
        };
        try {
            render();
            auto* gpuScene = preview.subsystemHost()->get<GPUSceneSubsystem>();
            if (!gpuScene) { return RhiTestResult::fail("Missing GPUScene"); }
            const auto& views = gpuScene->globalBufferViews();
            const auto range = gpuScene->rasterDrawLayout().adaptiveRange;
            if (views.lodGroups.size == 0 || range.count <= gpuScene->rasterDrawLayout().baseRange.count) {
                return RhiTestResult::fail("Bunny hierarchy fell back to the base range");
            }
            auto globals = capture("AfterPass", {
                {{"id", "gpuScene.VBuffer.lodGroups"}, {"count", views.lodGroups.size / sizeof(MeshletLodGroupRecord)}},
                {{"id", "gpuScene.VBuffer.meshlets"}, {"count", views.meshlets.size / sizeof(GPUSceneGpuMeshletRecord)}},
                {{"id", "gpuScene.VBuffer.instances"}, {"count", views.instances.size / sizeof(GPUSceneGpuInstanceRecord)}},
                {{"id", "gpuScene.VBuffer.meshletDraws"}, {"offset", range.offset}, {"count", range.count}}});
            render();
            std::vector<MeshletLodGroupRecord> groups;
            std::vector<GPUSceneGpuMeshletRecord> clusters;
            std::vector<GPUSceneGpuInstanceRecord> instances;
            std::vector<VisibleClusterRecord> candidates;
            for (const auto& row : read(globals, "gpuScene.VBuffer.lodGroups")) {
                groups.push_back({row.at("sphere").get<std::array<float, 4>>(), row.at("error").get<float>(),
                    row.at("level").get<uint32_t>(), row.at("flags").get<uint32_t>()});
            }
            for (const auto& row : read(globals, "gpuScene.VBuffer.meshlets")) {
                GPUSceneGpuMeshletRecord cluster;
                cluster.lod = row.at("lod").get<std::array<uint32_t, 4>>();
                clusters.push_back(cluster);
            }
            for (const auto& row : read(globals, "gpuScene.VBuffer.instances")) {
                GPUSceneGpuInstanceRecord instance;
                instance.worldMatrix = row.at("world").get<std::array<float, 16>>();
                instance.identity = row.at("identity").get<std::array<uint32_t, 4>>();
                instances.push_back(instance);
            }
            for (const auto& row : read(globals, "gpuScene.VBuffer.meshletDraws")) {
                candidates.push_back({row.at("clusterIndex"), row.at("instanceIndex"), row.at("dataIndex"), row.at("flags")});
            }
            size_t finestCount = 0, coarsestCount = SIZE_MAX;
            DebugValue report{{"asset", "StandfordBunny"}, {"groups", groups.size()}, {"candidateCount", range.count},
                {"cases", DebugValue::array()}};
            const char* modes[] = {"none", "meshlet", "triangle", "coverage", "lod", "depth"};
            for (uint32_t test = 0; test < 16; ++test) {
                const bool ortho = test >= 8;
                const uint32_t configuration = test % 8;
                const uint32_t manual = configuration >= 3 && configuration <= 5 ?
                    std::array{0u, 1u, 31u}[configuration - 3u] : UINT32_MAX;
                const float error = configuration == 0 ? .05f : configuration < 3 ? 16.f : 1.5f;
                const float bias = configuration == 2 ? 2.f : 0.f;
                const float eyeZ = configuration == 7 ? .8f : .22f;
                graph.setNodeRuntimeProperty(node, "autoLod", manual == UINT32_MAX);
                graph.setNodeRuntimeProperty(node, "lodLevel", manual == UINT32_MAX ? 0u : manual);
                graph.setNodeRuntimeProperty(node, "lodPixelError", error);
                graph.setNodeRuntimeProperty(node, "lodBias", bias);
                graph.setNodeRuntimeProperty(node, "visualization", modes[configuration % 6]);
                graph.setNodeRuntimeProperty(node, "camera.projection", ortho ? "orthographic" : "perspective");
                graph.setNodeRuntimeProperty(node, "camera.eye", {-.0168404f, .110154f, eyeZ});
                auto job = capture("AfterResidentLod", {{{"id", "lod.VBuffer.header"}, {"count", 1}},
                    {{"id", "lod.VBuffer.selections"}, {"count", range.count}}});
                render();
                auto header = read(job, "lod.VBuffer.header").at(0);
                if (header.at("overflow") != 0 || header.at("candidateCount") != range.count) {
                    return RhiTestResult::fail("Invalid scene selection header");
                }
                MeshletLodView view;
                view.eye = {-.0168404f, .110154f, eyeZ, .001f};
                view.forward[3] = ortho ? 1.f : 0.f;
                view.projection = {157.f, std::tan(3.14159265358979323846f / 6.f), .24f, error * std::exp2(bias)};
                const auto expected = selectMeshletLodReference(groups, clusters, instances, candidates, view, range.offset, manual);
                auto selected = read(job, "lod.VBuffer.selections");
                if (header.at("count") != expected.size() || expected.empty()) {
                    return RhiTestResult::fail("Scene CPU/GPU count mismatch in case " + std::to_string(test));
                }
                for (size_t i = 0; i < expected.size(); ++i) {
                    const auto& row = selected.at(i);
                    const MeshletLodSelection actual{row.at("instanceIndex"), row.at("clusterIndex"), row.at("recordIndex"), row.at("geometryIndex")};
                    if (actual != expected[i]) { return RhiTestResult::fail("Scene CPU/GPU cut mismatch in case " + std::to_string(test)); }
                }
                // The visibility output must decode exclusively through this frame's cut.
                std::vector<bool> live(range.count, false);
                for (const auto& item : expected) { live[item.recordIndex - range.offset] = true; }
                size_t covered = 0;
                for (uint32_t id : preview.pixels()) {
                    if (id == 0) { continue; }
                    const uint32_t record = (id >> kVisibilityTriangleBits) - 1u;
                    if (record < range.offset || record >= range.offset + range.count || !live[record - range.offset]) {
                        return RhiTestResult::fail("Rasterized visibility referenced a cluster outside the current cut");
                    }
                    ++covered;
                }
                if (covered < 100) { return RhiTestResult::fail("LOD switch lost Bunny coverage in case " + std::to_string(test)); }
                DebugValue histogram = DebugValue::object();
                for (const auto& item : expected) {
                    const std::string level = std::to_string(clusters[item.clusterIndex].lod[0]);
                    histogram[level] = histogram.value(level, 0u) + 1u;
                }
                report["cases"].push_back({{"case", test}, {"orthographic", ortho}, {"manualLevel", manual},
                    {"targetPixels", view.projection[3]}, {"eyeZ", eyeZ}, {"clusters", expected.size()},
                    {"coveredPixels", covered}, {"levels", histogram}});
                if (manual == 0) { finestCount = expected.size(); }
                if (manual == 31) { coarsestCount = expected.size(); }
                if (configuration == 3 || configuration == 5 || configuration == 6) {
                    graph.setNodeRuntimeProperty(node, "visualization", "triangle");
                    if (!preview.render(graph, 193, 157, "VBuffer.color")) { return RhiTestResult::fail(preview.lastLog()); }
                    std::string log;
                    if (!saveRgba8Png(context.outputDirectory / ("LodBunny-" + std::to_string(test) + ".png"),
                        reinterpret_cast<const uint8_t*>(preview.pixels().data()), 193, 157, log)) { return RhiTestResult::fail(log); }
                }
            }
            if (coarsestCount >= finestCount) { return RhiTestResult::fail("Manual coarse cut did not reduce resident cluster count"); }
            std::ofstream reportFile(context.outputDirectory / "MeshletLodReport.json");
            reportFile << report.dump(2);
            if (!reportFile) { return RhiTestResult::fail("Could not write LOD validation report"); }
            return RhiTestResult::pass("16 real-scene CPU/GPU cut comparisons and live VBuffer ID checks; finest " +
                std::to_string(finestCount) + ", coarsest " + std::to_string(coarsestCount) + " clusters");
        } catch (const std::exception& error) {
            return RhiTestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletLodSceneTest);

} // namespace
} // namespace metallic::tests
