#include "TestResourceParameters.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

class GPUDrivenInstanceCandidateTest final : public RHITest {
public:
    GPUDrivenInstanceCandidateTest()
    {
        type = RHITestType::Rendering;
        name = "gpu_driven_cpu_instance_candidates";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        using namespace render::builtin_pass;
        std::string log;
#define INSTANCE_REQUIRE(expr) do { const auto result = (expr); if (!result) { \
    return RHITestResult::fail(std::string(#expr) + ": " + resultToString(result) + " " + log); } } while (false)
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "CPU instance candidate regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
        INSTANCE_REQUIRE(created);
        constexpr uint32_t globalCount = 137;
        constexpr uint32_t streamCount = 9;
        enum Input { Params, Instances, Candidates, Owners, Visibility, VisibleIds, Counter,
            StreamParams, StreamInstances, StreamCandidates, StreamBindings, InputCount };
        const uint32_t strides[InputCount] = {sizeof(GPUDrivenPreviewGPUParams), sizeof(GPUSceneGPUInstanceRecord),
            4, 4, 4, 4, 4, sizeof(MeshletStreamGPUParams), sizeof(MeshletStreamGPUInstance),
            4, sizeof(MeshletStreamGPURasterBindings)};
        const uint32_t counts[InputCount] = {1, globalCount, globalCount, globalCount, globalCount,
            globalCount, 1, 1, streamCount, streamCount, 1};
        std::unique_ptr<BindlessHeap> heap;
        INSTANCE_REQUIRE(device->createBindlessHeap({.maxBuffers = InputCount})
            .transform([&](auto value) { heap = std::move(value); }));
        std::array<std::unique_ptr<Buffer>, InputCount> buffers;
        std::array<BindlessHandle, InputCount> handles;
        for (uint32_t i = 0; i < InputCount; ++i) {
            INSTANCE_REQUIRE(device->createBuffer({.size = uint64_t(strides[i]) * counts[i],
                .structureStride = strides[i], .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload})
                .transform([&](auto value) { buffers[i] = std::move(value); }));
            INSTANCE_REQUIRE(heap->allocate(BindlessHandleKind::Buffer)
                .transform([&](auto value) { handles[i] = std::move(value); }));
            INSTANCE_REQUIRE(buffers[i]->slice().and_then([&](const auto& slice) {
                return heap->writeStorageBuffer(handles[i], slice);
            }));
        }
        const auto upload = [&](Input index, const void* data, size_t size) {
            void* mapped = buffers[index]->map();
            if (mapped == nullptr) { return false; }
            if (size != 0) { std::memcpy(mapped, data, size); }
            buffers[index]->flush();
            buffers[index]->unmap();
            return true;
        };
        GPUDrivenPreviewGPUParams params;
        params.instanceCount = globalCount;
        params.cullingFlags = 0;
        MeshletStreamGPUParams streamParams;
        streamParams.sceneInstanceCount = streamCount;
        std::array<GPUSceneGPUInstanceRecord, globalCount> instances{};
        std::array<MeshletStreamGPUInstance, streamCount> streamInstances{};
        std::array<uint32_t, globalCount> owners{}, poison{};
        for (auto& instance : instances) {
            instance.worldMatrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            instance.localBoundingSphere = {0, 0, 0, 1};
            instance.identity[3] = GPUSceneGPUInstanceVisible;
        }
        streamInstances[2].gpuSceneInstanceIndex = 9;
        streamInstances[7].gpuSceneInstanceIndex = 127;
        for (auto& instance : streamInstances) {
            instance.visible = 1;
            instance.boundsCenterRadius[3] = 1.0f;
        }
        owners[9] = owners[127] = 1;
        poison.fill(3u);
        if (!upload(Params, &params, sizeof(params)) || !upload(Instances, instances.data(), sizeof(instances)) ||
            !upload(StreamParams, &streamParams, sizeof(streamParams)) ||
            !upload(StreamInstances, streamInstances.data(), sizeof(streamInstances)) ||
            !upload(Owners, owners.data(), sizeof(owners)) || !upload(Visibility, poison.data(), sizeof(poison))) {
            return RHITestResult::fail("Cannot upload instance candidate inputs");
        }
        std::array<std::unique_ptr<ShaderModule>, 4> shaders;
        std::array<std::unique_ptr<ComputePipeline>, 4> pipelines;
        const char* entries[] = {"gpuDrivenPreviewResetMain", "gpuDrivenPreviewInstanceCullMain",
            "gpuDrivenStreamAssetInstanceCullMain", "gpuDrivenStreamAssetCullResetMain"};
        for (uint32_t i = 0; i < pipelines.size(); ++i) {
            ShaderCompileResult compiled;
            const auto compiledResult = compileSlangShaderToSpirv({
                .moduleName = i >= 2 ? "Features/GPUDriven/GPUDrivenStreamAsset" : "Features/GPUDriven/GPUDrivenCulling",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log)
                .transform([&](auto value) { compiled = std::move(value); });
            INSTANCE_REQUIRE(compiledResult);
            INSTANCE_REQUIRE(device->createShaderModule({.spirv = compiled.spirv})
                .transform([&](auto value) { shaders[i] = std::move(value); }));
            INSTANCE_REQUIRE(device->createComputePipeline({.computeShader = {shaders[i].get()},
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = i >= 2 ? uint32_t(sizeof(MeshletStreamUserPush)) : uint32_t(sizeof(GPUDrivenPreviewUserPush))})
                .transform([&](auto value) { pipelines[i] = std::move(value); }));
        }
        constexpr uint64_t readbackBytes = (globalCount * 2u + 1u) * sizeof(uint32_t);
        std::unique_ptr<Buffer> readback;
        INSTANCE_REQUIRE(device->createBuffer({.size = readbackBytes, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { readback = std::move(value); }));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        INSTANCE_REQUIRE(device->createCommandPool(*queue).transform([&](auto value) { pool = std::move(value); }));
        INSTANCE_REQUIRE(pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }));
        INSTANCE_REQUIRE(device->createFence(false).transform([&](auto value) { fence = std::move(value); }));
        struct Case { std::vector<uint32_t> candidates, streamCandidates; bool late = false; bool standaloneStream = false; };
        // Non-identity candidate ordering catches accidental use of the compact index as a global ID.
        const Case cases[] = {{{136, 9, 0, 127, 64}, {7, 2}}, {{136, 0, 127}, {7}},
            {{136, 9, 0, 127, 64}, {2, 7}}, {{136, 9, 0, 127, 64}, {7, 2}, true},
            {{}, {}}, {{}, {}, true}, {{64, 9}, {2}},
            {{9, 127}, {7, 2}, false, true}, {{127}, {7}, false, true},
            {{9, 127}, {2, 7}, false, true}, {{}, {}, false, true}, {{9}, {2}, false, true}};
        for (uint32_t caseIndex = 0; caseIndex < std::size(cases); ++caseIndex) {
            const Case& test = cases[caseIndex];
            GPUDrivenPreviewUserPush push;
            push.paramsBuffer = handles[Params].shaderIndex;
            push.instanceBuffer = handles[Instances].shaderIndex;
            push.instanceVisibilityBuffer = handles[Visibility].shaderIndex;
            push.visibleInstanceIdsBuffer = handles[VisibleIds].shaderIndex;
            push.visibleInstanceCounterBuffer = handles[Counter].shaderIndex;
            push.streamOwnerMaskBuffer = handles[Owners].shaderIndex;
            push.candidateInstanceIdsBuffer = handles[Candidates].shaderIndex;
            push.candidateInstanceCount = static_cast<uint32_t>(test.candidates.size());
            push.passIndex = test.late ? 1u : 0u;
            MeshletStreamGPURasterBindings bindings;
            bindings.instanceVisibilityBuffer = handles[Visibility].shaderIndex;
            bindings.visibleInstanceIdsBuffer = handles[VisibleIds].shaderIndex;
            bindings.visibleInstanceCounterBuffer = handles[Counter].shaderIndex;
            bindings.candidateInstanceIdsBuffer = handles[StreamCandidates].shaderIndex;
            bindings.candidateInstanceCount = static_cast<uint32_t>(test.streamCandidates.size());
            bindings.instanceVisibilityResetCount = test.standaloneStream ? globalCount : 0u;
            MeshletStreamUserPush streamPush;
            streamPush.paramsBuffer = handles[StreamParams].shaderIndex;
            streamPush.instanceBuffer = handles[StreamInstances].shaderIndex;
            streamPush.rasterBindingsBuffer = handles[StreamBindings].shaderIndex;
            streamPush.traversalPhase = push.passIndex;
            if (!upload(Candidates, test.candidates.data(), test.candidates.size() * sizeof(uint32_t)) ||
                !upload(StreamCandidates, test.streamCandidates.data(), test.streamCandidates.size() * sizeof(uint32_t)) ||
                !upload(StreamBindings, &bindings, sizeof(bindings))) {
                return RHITestResult::fail("Cannot upload compact candidate lists");
            }
            if (test.late && !test.candidates.empty()) {
                std::array<uint32_t, globalCount> earlyStates{};
                for (uint32_t id : test.candidates) { earlyStates[id] = 1; }
                earlyStates[9] = earlyStates[64] = 2;
                const uint32_t earlyIds[] = {136, 0, 127}, earlyCount = 3;
                if (!upload(Visibility, earlyStates.data(), sizeof(earlyStates)) ||
                    !upload(VisibleIds, earlyIds, sizeof(earlyIds)) || !upload(Counter, &earlyCount, sizeof(earlyCount))) {
                    return RHITestResult::fail("Cannot seed early HZB states");
                }
            }
            if (caseIndex != 0) { INSTANCE_REQUIRE(fence->reset()); INSTANCE_REQUIRE(pool->reset()); }
            INSTANCE_REQUIRE(commands->begin());
            INSTANCE_REQUIRE(commands->bindBindlessHeap(*heap));
            if (test.standaloneStream) {
                INSTANCE_REQUIRE(commands->pushBindlessData(&streamPush, sizeof(streamPush)));
                INSTANCE_REQUIRE(commands->bindExecution(pipelines[3]->execution()));
            } else {
                INSTANCE_REQUIRE(commands->pushBindlessData(&push, sizeof(push)));
                INSTANCE_REQUIRE(commands->bindExecution(pipelines[0]->execution()));
            }
            INSTANCE_REQUIRE(commands->dispatch((globalCount + 63u) / 64u));
            std::array<BufferBarrierDesc, 3> barriers;
            const Input outputs[] = {Visibility, VisibleIds, Counter};
            for (uint32_t i = 0; i < barriers.size(); ++i) {
                barriers[i] = {.buffer = buffers[outputs[i]].get(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
            }
            INSTANCE_REQUIRE(commands->synchronize({.buffers = barriers}));
            if (!test.standaloneStream && !test.candidates.empty()) {
                INSTANCE_REQUIRE(commands->bindExecution(pipelines[1]->execution()));
                INSTANCE_REQUIRE(commands->dispatch((push.candidateInstanceCount + 63u) / 64u));
                INSTANCE_REQUIRE(commands->synchronize({.buffers = barriers}));
            }
            if (!test.standaloneStream) {
                INSTANCE_REQUIRE(commands->pushBindlessData(&streamPush, sizeof(streamPush)));
                INSTANCE_REQUIRE(commands->bindExecution(pipelines[3]->execution()));
                INSTANCE_REQUIRE(commands->dispatch(1));
                INSTANCE_REQUIRE(commands->synchronize({.buffers = barriers}));
            }
            if (!test.streamCandidates.empty()) {
                INSTANCE_REQUIRE(commands->pushBindlessData(&streamPush, sizeof(streamPush)));
                INSTANCE_REQUIRE(commands->bindExecution(pipelines[2]->execution()));
                INSTANCE_REQUIRE(commands->dispatch((bindings.candidateInstanceCount + 63u) / 64u));
            }
            for (auto& barrier : barriers) { barrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead}; }
            INSTANCE_REQUIRE(commands->synchronize({.buffers = barriers}));
            uint64_t offset = 0;
            for (Input output : outputs) {
                const uint64_t bytes = buffers[output]->desc().size;
                auto source = buffers[output]->slice();
                auto destination = readback->slice({offset, bytes});
                INSTANCE_REQUIRE(source);
                INSTANCE_REQUIRE(destination);
                INSTANCE_REQUIRE(commands->copyBuffer(*source, *destination));
                offset += bytes;
            }
            for (auto& barrier : barriers) { std::swap(barrier.before, barrier.after); }
            INSTANCE_REQUIRE(commands->synchronize({.buffers = barriers}));
            INSTANCE_REQUIRE(commands->end());
            CommandBuffer* submitted[] = {commands.get()};
            INSTANCE_REQUIRE(queue->submit({.commandBuffers = submitted, .signalFence = fence.get()}));
            INSTANCE_REQUIRE(fence->wait());
            readback->invalidate();
            const auto* data = static_cast<const uint32_t*>(readback->map());
            if (data == nullptr) { return RHITestResult::fail("Cannot read candidate culling results"); }
            std::array<uint32_t, globalCount> expected{};
            for (uint32_t id : test.candidates) { expected[id] = 1; }
            if (test.late && !test.candidates.empty()) { expected[9] = expected[64] = 3; }
            bool valid = std::equal(expected.begin(), expected.end(), data);
            const uint32_t outputCount = data[globalCount * 2u];
            valid = valid && outputCount == test.candidates.size();
            if (outputCount <= globalCount) {
                std::vector<uint32_t> actualIds(data + globalCount, data + globalCount + outputCount);
                auto expectedIds = test.candidates;
                std::sort(actualIds.begin(), actualIds.end());
                std::sort(expectedIds.begin(), expectedIds.end());
                valid = valid && actualIds == expectedIds;
            }
            readback->unmap();
            if (!valid) {
                return RHITestResult::fail("Candidate culling lost stable IDs or retained stale visibility in case " + std::to_string(caseIndex));
            }
        }
#undef INSTANCE_REQUIRE
        return RHITestResult::pass("Ordinary and Stream sparse candidates preserve global IDs across hide, re-enable, empty frames and late HZB recovery");
    }
};

METALLIC_REGISTER_RHI_TEST(GPUDrivenInstanceCandidateTest);

class GPUDrivenSceneVisibilityCandidateTest final : public RHITest {
public:
    GPUDrivenSceneVisibilityCandidateTest()
    {
        type = RHITestType::Rendering;
        name = "gpu_driven_scene_visibility_candidates";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = context.outputDirectory / "instance-candidate-scene";
        std::filesystem::create_directories(directory);
        const auto path = directory / "triangles.gltf";
        {
            constexpr float vertices[] = {-0.4f, -0.5f, 0, 0.4f, -0.5f, 0, 0, 0.5f, 0};
            std::ofstream binary(directory / "triangle.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(vertices), sizeof(vertices));
            std::ofstream gltf(path);
            gltf << R"json({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],
                "nodes":[{"mesh":0,"translation":[-0.65,0,0]},{"mesh":0,"translation":[0.65,0,0]}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
                "materials":[{"doubleSided":true}],"buffers":[{"uri":"triangle.bin","byteLength":36}],
                "bufferViews":[{"buffer":0,"byteLength":36}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
                    "min":[-0.4,-0.5,0],"max":[0.4,0.5,0]}]})json";
            if (!binary || !gltf) { return RHITestResult::fail("Cannot write scene visibility fixture"); }
        }
        scene::SceneDocument document;
        if (!document.load(path)) { return RHITestResult::fail(document.lastLoadResult().error); }
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) {
            return RHITestResult::skip("Scene candidate test requires preview mesh/ray capabilities");
        }
        if (!initialized) { return RHITestResult::fail("Scene candidate preview initialization failed"); }
        preview.bindRuntimeScene(&document);
        render::RenderGraph graph;
        graph.addNode("VisibilityBufferPass", "VBuffer", {{"path", path.string()},
            {"visualization", "meshlet"}, {"meshletNormalConeCull", false}, {"autoLod", false}});
        graph.markOutput("VBuffer.color");
        graph.setViewProperties({{"camera", {{"eye", {0, 0, 3}}, {"center", {0, 0, 0}},
            {"up", {0, 1, 0}}, {"fovDegrees", 60.0}, {"znear", 0.1}, {"zfar", 10.0}}},
            {"temporalJitter", false}});
        const std::array<scene::SceneEntity, 2> objects = {
            document.objectForNode(0).entity(), document.objectForNode(1).entity()};
        const std::array<std::array<bool, 2>, 6> cases = {{{true, true}, {false, true},
            {true, true}, {false, false}, {false, false}, {true, true}}};
        std::vector<uint32_t> reference;
        size_t referenceCoverage = 0;
        uint32_t background = 0;
        std::array<bool, 2> currentVisibility{true, true};
        for (uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
            const auto& enabled = cases[caseIndex];
            for (uint32_t i = 0; i < objects.size(); ++i) {
                if (currentVisibility[i] == enabled[i]) { continue; }
                if (!document.setObjectVisible(objects[i], enabled[i])) {
                    return RHITestResult::fail("Cannot update scene object visibility");
                }
                currentVisibility[i] = enabled[i];
            }
            // Reuse all frame slots so stale candidate lists and visibility cannot hide behind fresh allocations.
            for (uint32_t frame = 0; frame < 4; ++frame) {
                const auto rendered = preview.render(graph, 193, 129);
                if (render::hasError(rendered, render::Error::Unsupported)) { return RHITestResult::skip(preview.lastLog()); }
                if (!rendered) { return RHITestResult::fail(preview.lastLog()); }
                const auto& pixels = preview.pixels();
                if (caseIndex == 0 && frame == 0) {
                    reference = pixels;
                    background = pixels.front();
                    referenceCoverage = std::count_if(pixels.begin(), pixels.end(),
                        [&](uint32_t pixel) { return pixel != background; });
                    if (referenceCoverage < 200) { return RHITestResult::fail("Scene candidate reference has insufficient coverage"); }
                }
                const size_t coverage = std::count_if(pixels.begin(), pixels.end(),
                    [&](uint32_t pixel) { return pixel != background; });
                const bool correct = enabled[0] && enabled[1] ? pixels == reference :
                    !enabled[0] && !enabled[1] ? coverage == 0 : coverage > 0 && coverage < referenceCoverage;
                spdlog::info("[Scene candidates] case={} frame={} coverage={}", caseIndex, frame, coverage);
                if (!correct || frame == 3) {
                    std::string log;
                    if (!saveRgba8Png(directory / ("case-" + std::to_string(caseIndex) + ".png"),
                            reinterpret_cast<const uint8_t*>(pixels.data()), preview.width(), preview.height(), log)) {
                        return RHITestResult::fail(log);
                    }
                }
                if (!correct) {
                    return RHITestResult::fail("Scene candidate visibility changed incorrectly at case " +
                        std::to_string(caseIndex) + " frame " + std::to_string(frame));
                }
            }
        }
        return RHITestResult::pass("Bound scene visibility excludes disabled objects and restores identical raster output across empty candidate frames");
    }
};

METALLIC_REGISTER_RHI_TEST(GPUDrivenSceneVisibilityCandidateTest);

class GPUDrivenConeProbe final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(64 * 16, 16).storageReadWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        const char* searchPaths[] = {PROJECT_SOURCE_DIR "/Shaders/Features/GPUDriven"};
        auto result = render::compileSlangShaderToSpirv({.moduleName = "GPUDrivenConeProbe",
            .entryPointName = "gpuDrivenConeProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .additionalSearchPaths = searchPaths}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc bindings[] = {{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::GPUDrivenConeProbeResources, output)}};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {bindings, 1},
            .requiresRayQuery = false,
            .resourceParameterSize = sizeof(metallic::tests::GPUDrivenConeProbeResources),
        }, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding bindings[] = {{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::GPUDrivenConeProbeResources, output), .buffer = context.outputBuffer("data").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {bindings, 1}});
    }
private:
    render::ComputeProgram program_;
};

class GPUDrivenConeScaleTest final : public RHITest {
public:
    GPUDrivenConeScaleTest() { type = RHITestType::Rendering; name = "gpu_driven_cone_scale_invariance"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "Cone scale regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
        if (!initialized) { return RHITestResult::fail("Cone probe device creation failed"); }
        render::registerRenderGraphPassType("GPUDrivenConeProbe", "Normal cone scale probe",
            [] { return std::make_unique<GPUDrivenConeProbe>(); });
        render::RenderGraph graph;
        graph.addNode("GPUDrivenConeProbe", "Probe");
        graph.markOutput("Probe.data");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(*device, graph, 1, 1, log)) { return RHITestResult::fail(log); }
        if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
            !executor.waitForSubmittedWork()) { return RHITestResult::fail("Cone probe dispatch failed"); }
        auto* buffer = executor.outputResource("Probe.data")->buffer;
        buffer->invalidate();
        const auto* data = static_cast<const std::array<float, 4>*>(buffer->map());
        if (data == nullptr) { return RHITestResult::fail("Cone probe readback failed"); }
        bool valid = true;
        for (uint32_t index = 0; index < 32; ++index) {
            const auto& axis = data[index * 2];
            const auto& expected = data[index * 2 + 1];
            for (uint32_t channel = 0; channel < 3; ++channel) {
                valid = valid && std::isfinite(axis[channel]) && std::abs(axis[channel] - expected[channel]) < 1e-5f;
            }
            valid = valid && axis[3] == 1.0f && expected[3] == 0.0f;
            spdlog::info("[Cone] case={} axis=({},{},{}) front={} back={}", index, axis[0], axis[1], axis[2], axis[3], expected[3]);
        }
        buffer->unmap();
        return valid ? RHITestResult::pass("Preview and streamed normal cones survive uniform scales and rotation")
            : RHITestResult::fail("Object scale changed the cone axis or rejected a front-facing meshlet");
    }
};

class TwoPassOcclusionProbe final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(32 * 16, 16).storageReadWrite();
        reflection.addBufferOutput("history").buffer(222 * 4, 4).storageReadWrite();
        reflection.addBufferOutput("current").buffer(222 * 4, 4).storageReadWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "TwoPassOcclusionProbe",
            .entryPointName = "twoPassOcclusionProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc bindings[] = {{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::TwoPassOcclusionProbeResources, output)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::TwoPassOcclusionProbeResources, previousHzb)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::TwoPassOcclusionProbeResources, currentHzb)}};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {bindings, 3},
            .requiresRayQuery = false,
            .resourceParameterSize = sizeof(metallic::tests::TwoPassOcclusionProbeResources),
        }, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::TwoPassOcclusionProbeResources, output), .buffer = context.outputBuffer("data").buffer()},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::TwoPassOcclusionProbeResources, previousHzb), .buffer = context.outputBuffer("history").buffer()},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::TwoPassOcclusionProbeResources, currentHzb), .buffer = context.outputBuffer("current").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {bindings, 3}});
    }
private:
    render::ComputeProgram program_;
};

class GPUDrivenTwoPassOcclusionTest final : public RHITest {
public:
    GPUDrivenTwoPassOcclusionTest() { type = RHITestType::Rendering; name = "gpu_driven_two_pass_occlusion"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "Two-pass occlusion regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
        if (!initialized) { return RHITestResult::fail("Occlusion probe device creation failed"); }
        render::registerRenderGraphPassType("TwoPassOcclusionProbe", "Two-pass occlusion probe",
            [] { return std::make_unique<TwoPassOcclusionProbe>(); });
        render::RenderGraph graph;
        graph.addNode("TwoPassOcclusionProbe", "Probe");
        graph.markOutput("Probe.data");
        graph.markOutput("Probe.history");
        graph.markOutput("Probe.current");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(*device, graph, 1, 1, log)) { return RHITestResult::fail(log); }
        if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
            !executor.waitForSubmittedWork()) { return RHITestResult::fail("Occlusion probe dispatch failed"); }
        auto* buffer = executor.outputResource("Probe.data")->buffer;
        buffer->invalidate();
        const auto* data = static_cast<const std::array<float, 4>*>(buffer->map());
        if (data == nullptr) { return RHITestResult::fail("Occlusion probe readback failed"); }
        const std::array<std::array<float, 2>, 12> expected{{
            {1, 0}, {0, 1}, {0, 0}, {1, 0}, {0, 1}, {1, 0},
            {1, 0}, {0, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 1}}};
        bool valid = true;
        for (uint32_t index = 0; index < 32; ++index) {
            const auto& value = data[index];
            valid = valid && value[2] == 0 && value[3] == 0;
            if (index < expected.size()) {
                valid = valid && value[0] == expected[index][0] && value[1] == expected[index][1];
            }
            spdlog::info("[Two-pass] case={} early={} late={} projectionErrors={} depth/scaleErrors={}",
                index, value[0], value[1], value[2], value[3]);
        }
        buffer->unmap();
        return valid ? RHITestResult::pass("GPU occlusion rejection/recovery and conservative projection/depth/scale bounds")
            : RHITestResult::fail("Occlusion lost a disoccluded meshlet, drew a meshlet twice, or violated conservative bounds");
    }
};

class GPUDrivenSponzaCullingTest final : public RHITest {
public:
    GPUDrivenSponzaCullingTest() { type = RHITestType::Rendering; name = "gpu_driven_sponza_culling_equivalence"; }
    RHITestResult run(RHITestContext& context) override
    {
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Sponza/glTF/Sponza.gltf")) {
            return RHITestResult::fail(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders"); }
        if (!initialized) { return RHITestResult::fail("Preview initialization failed"); }
        preview.bindRuntimeScene(&scene);
        render::RenderGraph graph;
        const uint32_t raster = graph.addNode("VisibilityBufferPass", "VBuffer",
            {{"path", "Asset/Sponza/glTF/Sponza.gltf"}, {"visualization", "meshlet"}})->id;
        graph.markOutput("VBuffer.color");
        const std::array<const char*, 5> flags{
            "instanceFrustumCull", "instanceHzbCull", "meshletFrustumCull", "meshletNormalConeCull", "meshletHzbCull"};
        const std::array<std::array<double, 6>, 3> cameras{{
            {-5.646879, 11.323325, -0.051334, 29.722519, -15.616591, -5.377987},
            {-3.325359, 9.545668, -0.429330, 31.658249, -17.904564, -5.697884},
            {-4.960654, 10.867253, -0.183054, 29.629982, -17.087643, -5.392424}}};
        std::string log;
        bool valid = true;
        for (uint32_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex) {
            const auto& c = cameras[cameraIndex];
            graph.setViewProperties({{"camera", {{"eye", {c[0], c[1], c[2]}}, {"center", {c[3], c[4], c[5]}},
                {"up", {0, 1, 0}}, {"fovDegrees", 45.0}, {"znear", 0.018548}, {"zfar", 1854.789185}}},
                {"temporalJitter", false}});
            std::vector<uint32_t> reference;
            // No culling is the oracle. Then exercise all culling and disable one
            // stage at a time, including several frames of temporal HZB reuse.
            for (int configuration = -2; configuration < int(flags.size()); ++configuration) {
                for (int flag = 0; flag < int(flags.size()); ++flag) {
                    graph.setNodeRuntimeProperty(raster, flags[flag], configuration != -2 && configuration != flag);
                }
                for (uint32_t frame = 0; frame < 3; ++frame) {
                    if (!preview.render(graph, 799, 292)) { return RHITestResult::fail(preview.lastLog()); }
                    if (configuration != -2 && preview.pixels() != reference) { valid = false; }
                }
                const auto& pixels = preview.pixels();
                if (configuration == -2) {
                    reference = pixels;
                    const auto covered = std::count_if(pixels.begin(), pixels.end(),
                        [](uint32_t pixel) { return (pixel & 255u) >= 64u; });
                    if (covered < 10000) { return RHITestResult::fail("Sponza reference contains too little geometry"); }
                }
                size_t changed = 0;
                for (size_t pixel = 0; pixel < pixels.size(); ++pixel) { changed += pixels[pixel] != reference[pixel]; }
                spdlog::info("[Sponza culling] camera={} configuration={} changed={}", cameraIndex, configuration, changed);
                if (!saveRgba8Png(context.outputDirectory / ("Sponza" + std::to_string(cameraIndex) +
                        "Cull" + std::to_string(configuration) + ".png"), reinterpret_cast<const uint8_t*>(pixels.data()),
                        preview.width(), preview.height(), log)) { return RHITestResult::fail(log); }
            }
        }
        // Review artifact with actual materials/lighting at the failing camera.
        // Native resolution isolates culling from temporal reconstruction.
        render::RenderSampleLoadResult sample;
        if (!render::loadBuiltInRenderSample("realtime-lighting", sample, log)) { return RHITestResult::fail(log); }
        const auto* sr = sample.graph.findNode("DLSSSR");
        const auto* nr = sample.graph.findNode("DLSSNR");
        if (sr == nullptr || nr == nullptr || !sample.desc.environment.has_value()) {
            return RHITestResult::fail("Missing realtime sample reconstruction or environment settings");
        }
        const auto srId = sr->id, nrId = nr->id;
        sample.graph.removeNode(srId);
        sample.graph.removeNode(nrId);
        sample.graph.addEdge("Deferred.color", "AutoExposure.source");
        sample.graph.addEdge("AutoExposure.color", "FinalBlit.source");
        const auto& c = cameras[1];
        sample.graph.setViewProperties({{"camera", {{"eye", {c[0], c[1], c[2]}}, {"center", {c[3], c[4], c[5]}},
            {"up", {0, 1, 0}}, {"fovDegrees", 45.0}, {"znear", 0.018548}, {"zfar", 1854.789185}}},
            {"temporalJitter", false}});
        preview.setEnvironment({.enabled = true, .path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
        for (uint32_t frame = 0; frame < 32; ++frame) {
            if (!preview.render(sample.graph, 1198, 438)) { return RHITestResult::fail(preview.lastLog()); }
        }
        if (!saveRgba8Png(context.outputDirectory / "SponzaShaded.png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                preview.width(), preview.height(), log)) { return RHITestResult::fail(log); }
        return valid ? RHITestResult::pass("Sponza captured cameras match unculled visibility with every culling stage enabled")
            : RHITestResult::fail("Culling changed visible Sponza geometry");
    }
};

class GPUDrivenTemporalOcclusionTest final : public RHITest {
public:
    GPUDrivenTemporalOcclusionTest() { type = RHITestType::Rendering; name = "gpu_driven_temporal_occlusion_equivalence"; }
    RHITestResult run(RHITestContext& context) override
    {
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Sponza/glTF/Sponza.gltf")) {
            return RHITestResult::fail(scene.lastLoadResult().error);
        }
        // Independent histories, identical cameras/jitter. Only occlusion differs.
        render::RenderView view;
        view.setTemporalJitter(true);
        render::RenderGraphPreviewRenderer reference, culled;
        for (auto* preview : {&reference, &culled}) {
            const auto initialized = preview->initialize(context.enableValidation, true);
            if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders"); }
            if (!initialized) { return RHITestResult::fail("Temporal preview initialization failed"); }
            preview->bindRuntimeScene(&scene);
            preview->bindRenderView(&view);
        }
        render::RenderGraph referenceGraph, culledGraph;
        for (auto* graph : {&referenceGraph, &culledGraph}) {
            const bool enableOcclusion = graph == &culledGraph;
            graph->addNode("VisibilityBufferPass", "VBuffer", {{"visualization", "triangle"},
                {"instanceHzbCull", enableOcclusion}, {"meshletHzbCull", enableOcclusion}});
            graph->markOutput("VBuffer.color");
        }
        for (uint32_t frame = 0; frame < 30; ++frame) {
            const double motion = std::sin(frame * 1.7);
            if (!view.setCameraProperties({
                    {"eye", {-4.5 + motion * 2.0, 10.0 + motion * 1.5, -0.2 + motion}},
                    {"center", {30.0 - motion * 4.0, -17.0 + motion * 2.0, -5.5}},
                    {"up", {0, 1, 0}}, {"fovDegrees", 45.0}, {"znear", 0.018548}, {"zfar", 1854.789185},
                    {"projection", frame >= 20 ? "orthographic" : "perspective"}, {"orthoHeight", 25.0}})) {
                return RHITestResult::fail("Temporal test camera is invalid");
            }
            if (frame == 8) { view.cameraCut(); }
            const uint32_t width = frame < 16 ? 257 : 193;
            const uint32_t height = frame < 16 ? 131 : 157;
            if (!reference.render(referenceGraph, width, height)) { return RHITestResult::fail(reference.lastLog()); }
            if (!culled.render(culledGraph, width, height)) { return RHITestResult::fail(culled.lastLog()); }
            const auto covered = std::count_if(reference.pixels().begin(), reference.pixels().end(),
                [](uint32_t pixel) { return (pixel & 255u) >= 64u; });
            if (covered < 100) { return RHITestResult::fail("Temporal reference contains too little geometry"); }
            size_t changed = 0;
            for (size_t pixel = 0; pixel < reference.pixels().size(); ++pixel) {
                changed += reference.pixels()[pixel] != culled.pixels()[pixel];
            }
            spdlog::info("[Temporal occlusion] frame={} size={}x{} changed={}", frame, width, height, changed);
            if (changed != 0) {
                std::string log;
                saveRgba8Png(context.outputDirectory / "TemporalReference.png",
                    reinterpret_cast<const uint8_t*>(reference.pixels().data()), width, height, log);
                saveRgba8Png(context.outputDirectory / "TemporalCulled.png",
                    reinterpret_cast<const uint8_t*>(culled.pixels().data()), width, height, log);
                return RHITestResult::fail("Temporal HZB changed visibility at frame " + std::to_string(frame));
            }
        }
        return RHITestResult::pass("30 moving/jittered reversed-Z Sponza views match with occlusion on/off, including resize and perspective/orthographic projection");
    }
};

class GPUDrivenSponzaVSMFloorTest final : public RHITest {
public:
    GPUDrivenSponzaVSMFloorTest() { type = RHITestType::Rendering; name = "gpu_driven_sponza_vsm_floor"; }
    RHITestResult run(RHITestContext& context) override
    {
        scene::SceneDocument scene;
        const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Sponza/SponzaVSM.metallic_scene.json";
        if (!scene.load(path)) { return RHITestResult::fail(scene.lastLoadResult().error); }
        render::RenderView view;
        view.setTemporalJitter(false);
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders"); }
        if (!initialized) { return RHITestResult::fail("Sponza VSM preview initialization failed"); }
        preview.bindRuntimeScene(&scene);
        preview.setEnvironment(scene.environment());
        if (!preview.setLighting(scene.lighting())) { return RHITestResult::fail("Invalid Sponza VSM lighting"); }
        preview.bindRenderView(&view);
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample(render::kVSMSponzaSampleId, sample, log)) { return RHITestResult::fail(log); }
        auto& graph = sample.graph;
        const auto raster = graph.findNode("VBuffer")->id;
        graph.setNodeRuntimeProperty(raster, "visualization", "triangle");
        graph.clearOutputs();
        graph.markOutput("VBuffer.color");
        const std::array<const char*, 5> flags{
            "instanceFrustumCull", "instanceHzbCull", "meshletFrustumCull", "meshletNormalConeCull", "meshletHzbCull"};
        // Initial sample view and low, downward-facing views from the floor-disappearance recording.
        const std::array<std::array<double, 6>, 3> cameras{{
            {8.7, 1.65, 0.15, 7.700582, 1.629054, 0.123070},
            {5.917828, 0.585971, -0.344645, 5.132805, -0.028311, -0.424636},
            {6.790932, 1.210763, -0.288578, 5.978194, 0.629214, -0.340765}}};
        bool valid = true;
        for (uint32_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex) {
            const auto& c = cameras[cameraIndex];
            if (!view.setCameraProperties({{"eye", {c[0], c[1], c[2]}}, {"center", {c[3], c[4], c[5]}},
                    {"up", {-0.020938, 0.999781, -0.000564}}, {"fovDegrees", 60.0}, {"znear", 0.05}, {"zfar", 100.0}})) {
                return RHITestResult::fail("Invalid Sponza VSM regression camera");
            }
            std::vector<uint32_t> reference;
            // Keep all other stages enabled, isolating the production streamed normal-cone path.
            for (bool coneCull : {false, true}) {
                for (const char* flag : flags) { graph.setNodeRuntimeProperty(raster, flag, true); }
                graph.setNodeRuntimeProperty(raster, "meshletNormalConeCull", coneCull);
                // Let initial streaming uploads and all frame slots settle before the pixel oracle.
                for (uint32_t frame = 0; frame < 8; ++frame) {
                    if (!preview.render(graph, 960, 540, "VBuffer.color")) { return RHITestResult::fail(preview.lastLog()); }
                }
                for (uint32_t frame = 0; frame < 4; ++frame) {
                    if (!preview.render(graph, 960, 540, "VBuffer.color")) { return RHITestResult::fail(preview.lastLog()); }
                    if (!coneCull && frame == 0) {
                        reference = preview.pixels();
                        const auto covered = std::count_if(reference.begin(), reference.end(),
                            [](uint32_t pixel) { return (pixel & 255u) >= 64u; });
                        if (covered < 10000) { return RHITestResult::fail("Sponza VSM reference contains too little geometry"); }
                    }
                    size_t changed = 0;
                    for (size_t pixel = 0; pixel < reference.size(); ++pixel) {
                        changed += preview.pixels()[pixel] != reference[pixel];
                    }
                    valid = valid && changed == 0;
                    spdlog::info("[Sponza VSM floor] camera={} cone={} frame={} changed={}", cameraIndex, coneCull, frame, changed);
                }
                if (!saveRgba8Png(context.outputDirectory / ("SponzaVSMFloor" + std::to_string(cameraIndex) +
                        (coneCull ? "Culled.png" : "Reference.png")), reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                        preview.width(), preview.height(), log)) { return RHITestResult::fail(log); }
            }
        }
        // Also exercise the full VSM/deferred graph at the recorded low views.
        if (!render::loadBuiltInRenderSample(render::kVSMSponzaSampleId, sample, log)) { return RHITestResult::fail(log); }
        sample.graph.markOutput(sample.desc.previewOutput);
        for (uint32_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex) {
            const auto& c = cameras[cameraIndex];
            if (!view.setCameraProperties({{"eye", {c[0], c[1], c[2]}}, {"center", {c[3], c[4], c[5]}},
                    {"up", {-0.020938, 0.999781, -0.000564}}, {"fovDegrees", 60.0}, {"znear", 0.05}, {"zfar", 100.0}})) {
                return RHITestResult::fail("Invalid Sponza VSM shaded camera");
            }
            for (uint32_t frame = 0; frame < 8; ++frame) {
                if (!preview.render(sample.graph, 960, 540)) { return RHITestResult::fail(preview.lastLog()); }
            }
            if (!saveRgba8Png(context.outputDirectory / ("SponzaVSMFloor" + std::to_string(cameraIndex) + "Shaded.png"),
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), preview.width(), preview.height(), log)) {
                return RHITestResult::fail(log);
            }
        }
        return valid ? RHITestResult::pass("Sponza VSM floor visibility matches with streamed cone culling on/off across low camera views and HZB history")
            : RHITestResult::fail("Streamed normal-cone culling changed visible Sponza VSM floor geometry");
    }
};

METALLIC_REGISTER_RHI_TEST(GPUDrivenSponzaVSMFloorTest);
METALLIC_REGISTER_RHI_TEST(GPUDrivenConeScaleTest);
METALLIC_REGISTER_RHI_TEST(GPUDrivenTwoPassOcclusionTest);
METALLIC_REGISTER_RHI_TEST(GPUDrivenSponzaCullingTest);
METALLIC_REGISTER_RHI_TEST(GPUDrivenTemporalOcclusionTest);
} // namespace
} // namespace metallic::tests
