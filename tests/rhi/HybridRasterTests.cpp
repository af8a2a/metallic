#include "RhiTest.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/RenderSample.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

#define HYBRID_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

class HybridRasterDepthTest final : public RhiTest {
public:
    HybridRasterDepthTest() { type = RhiTestType::Rendering; name = "hybrid_raster_depth_coverage_and_overflow"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Hybrid raster regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableMeshShader = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders and bindless heap"); }
        HYBRID_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RhiTestResult::skip("Requires 64-bit buffer atomics and <=8 subpixel bits");
        }
        constexpr uint32_t width = 97, height = 73, pixelCount = width * height;
        std::vector<std::array<float, 4>> vertices;
        auto triangle = [&](std::array<float, 3> a, std::array<float, 3> b, std::array<float, 3> c, bool perspective = false) {
            uint32_t corner = 0;
            for (auto point : {a, b, c}) {
                const float w = perspective ? 0.7f + float(corner++) * 0.6f : 1.0f;
                vertices.push_back({(point[0] / width * 2.0f - 1.0f) * w,
                    (point[1] / height * 2.0f - 1.0f) * w, point[2] * w, w});
            }
        };
        // A large HW receiver with overlapping small foreground/background triangles.
        triangle({0, 0, .48f}, {0, 73, .48f}, {97, 0, .48f});
        triangle({97, 0, .48f}, {0, 73, .48f}, {97, 73, .48f});
        for (uint32_t i = 0; i < 180; ++i) {
            const float x = float((i * 17) % 93) + .17f;
            const float y = float((i * 13) % 69) + .31f;
            const float size = .3f + float(i % 19);
            const float z = .05f + float(i % 89) * .01f;
            auto a = std::array<float, 3>{x, y, z};
            auto b = std::array<float, 3>{x + .2f, y + size, z + .003f};
            auto c = std::array<float, 3>{x + size, y + .4f, z + .007f};
            if (i % 3 == 0) { std::swap(b, c); }
            triangle(a, b, c, i % 2 != 0);
        }
        // Adjacent triangles through exact pixel centers exercise the top-left rule.
        triangle({24.5f, 30.5f, .97f}, {24.5f, 38.5f, .97f}, {32.5f, 30.5f, .97f});
        triangle({32.5f, 30.5f, .97f}, {24.5f, 38.5f, .97f}, {32.5f, 38.5f, .97f});
        // Near/far clipping, viewport clipping and degenerate/subpixel coverage.
        triangle({5, 7, -.2f}, {5, 20, .3f}, {17, 7, .5f});
        triangle({40, 7, 1.2f}, {40, 20, .7f}, {52, 7, .5f});
        triangle({-3, 48, .9f}, {-3, 57, .91f}, {5, 48, .92f});
        triangle({2, 2, .8f}, {2, 2, .8f}, {3, 3, .8f});

        triangle({62, 7, 0.f}, {62, 13, 0.f}, {68, 7, 0.f});
        triangle({74, 7, 1.f}, {74, 13, 1.f}, {80, 7, 1.f});

        std::unique_ptr<Buffer> input;
        HYBRID_REQUIRE(device->createBuffer({.size = vertices.size() * 16, .structureStride = 16,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, input));
        void* mapped = input->map();
        if (!mapped) { return RhiTestResult::fail("Vertex upload map failed"); }
        std::memcpy(mapped, vertices.data(), vertices.size() * 16); input->flush(); input->unmap();
        std::unique_ptr<BindlessHeap> heap;
        HYBRID_REQUIRE(device->createBindlessHeap({.maxBuffers = 2}, heap));
        BindlessHandle inputHandle, queueHandle;
        HYBRID_REQUIRE(heap->allocateBuffer(inputHandle)); HYBRID_REQUIRE(heap->allocateBuffer(queueHandle));
        HYBRID_REQUIRE(heap->writeStorageBuffer(inputHandle, *input));
        std::array<std::unique_ptr<ShaderModule>, 2> shaders;
        const char* entries[] = {"probeMeshMain", "probeFragmentMain"};
        const char* capabilities[] = {"spvMeshShadingEXT"};
        for (size_t i = 0; i < shaders.size(); ++i) {
            ShaderCompileResult shader;
            const auto compiled = compileSlangShaderToSpirv({.moduleName = "HybridRasterProbe", .entryPointName = entries[i],
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .capabilities = capabilities, .capabilityCount = 1}, shader);
            log = shader.diagnostics; HYBRID_REQUIRE(compiled);
            HYBRID_REQUIRE(device->createShaderModule({.code = shader.spirv.data(), .byteSize = shader.spirv.size() * 4}, shaders[i]));
        }
        std::array<std::unique_ptr<Texture>, 2> textures;
        std::array<std::unique_ptr<TextureView>, 2> views;
        std::array<std::unique_ptr<Buffer>, 2> readback;
        for (size_t i = 0; i < 2; ++i) {
            const auto format = i == 0 ? Format::R32Uint : Format::D32Sfloat;
            HYBRID_REQUIRE(device->createTexture({.usage = TextureUsageBits::TransferSource |
                (i == 0 ? TextureUsageBits::ColorAttachment : TextureUsageBits::DepthStencilAttachment),
                .format = format, .width = width, .height = height}, textures[i]));
            HYBRID_REQUIRE(device->createTextureView(*textures[i], {.format = format}, views[i]));
            HYBRID_REQUIRE(device->createBuffer({.size = pixelCount * 4, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}, readback[i]));
        }
        std::unique_ptr<Buffer> queueReadback, pixelReadback;
        HYBRID_REQUIRE(device->createBuffer({.size = 32, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}, queueReadback));
        HYBRID_REQUIRE(device->createBuffer({.size = pixelCount * 8, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}, pixelReadback));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        HYBRID_REQUIRE(device->createCommandPool(*queue, pool));
        HYBRID_REQUIRE(pool->createCommandBuffer(commands)); HYBRID_REQUIRE(device->createFence(false, fence));
        bool submitted = false;
        size_t softwarePixels = 0, overflows = 0, cases = 0;
        for (bool reversed : {false, true}) {
            for (bool doubleSided : {false, true}) {
                std::unique_ptr<GraphicsPipeline> pipeline;
                HYBRID_REQUIRE(device->createGraphicsPipeline({.meshShader = shaders[0].get(), .fragmentShader = shaders[1].get(),
                    .colorFormat = Format::R32Uint, .depthStencilFormat = Format::D32Sfloat,
                    .rasterization = {.cullMode = doubleSided ? CullMode::None : CullMode::Back, .frontFace = FrontFace::CounterClockwise},
                    .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true,
                        .depthCompareOp = reversed ? CompareOp::GreaterEqual : CompareOp::LessEqual}, .usesBindlessHeap = true}, pipeline));
                std::vector<uint32_t> referenceIds;
                std::vector<float> referenceDepth;
                for (uint32_t configuration = 0; configuration < 5; ++configuration) {
                    VisibilityHybridRasterizer rasterizer;
                    const uint32_t capacity = configuration == 4 ? 1u : 262144u;
                    HYBRID_REQUIRE(rasterizer.initialize(*device, width, height, log, capacity));
                    HYBRID_REQUIRE(heap->writeStorageBuffer(queueHandle, rasterizer.queueBuffer()));
                    if (submitted) { HYBRID_REQUIRE(fence->reset()); HYBRID_REQUIRE(pool->reset()); }
                    HYBRID_REQUIRE(commands->begin());
                    const TextureBarrierDesc transitions[] = {
                        {.texture = textures[0].get(), .before = submitted ? ResourceState::TransferSource : ResourceState::Undefined,
                            .after = ResourceState::ColorAttachment},
                        {.texture = textures[1].get(), .before = submitted ? ResourceState::TransferSource : ResourceState::Undefined,
                            .after = ResourceState::DepthStencilAttachment}};
                    commands->barrier({.textures = transitions, .textureCount = 2});
                    const bool hybrid = configuration != 0;
                    if (hybrid) { rasterizer.begin(*commands, configuration == 1 ? 1.f : configuration == 2 ? 8.f : 32.f, reversed); }
                    const RenderingAttachmentDesc color{.view = views[0].get(), .state = ResourceState::ColorAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
                    const RenderingAttachmentDesc depth{.view = views[1].get(), .state = ResourceState::DepthStencilAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = reversed ? 0.f : 1.f};
                    commands->beginRendering({.renderArea = {.width = width, .height = height}, .colorAttachments = &color,
                        .colorAttachmentCount = 1, .depthStencilAttachment = &depth});
                    commands->setViewport({.width = float(width), .height = float(height), .maxDepth = 1.f});
                    commands->setScissor({.width = width, .height = height});
                    commands->bindBindlessHeap(*heap); commands->bindGraphicsPipeline(*pipeline);
                    const uint32_t push[] = {inputHandle.index, hybrid ? queueHandle.index : UINT32_MAX, doubleSided ? 1u : 0u};
                    commands->pushBindlessData(push, sizeof(push));
                    commands->drawMeshTasks(uint32_t(vertices.size() / 3)); commands->endRendering();
                    if (hybrid) {
                        HYBRID_REQUIRE(rasterizer.resolve(*commands, *textures[0], *views[0], *textures[1], *views[1]));
                        const BufferBarrierDesc bufferTransitions[] = {
                            {.buffer = &rasterizer.queueBuffer(), .before = ResourceState::ShaderRead, .after = ResourceState::TransferSource},
                            {.buffer = &rasterizer.pixelBuffer(), .before = ResourceState::ShaderRead, .after = ResourceState::TransferSource}};
                        commands->barrier({.buffers = bufferTransitions, .bufferCount = 2});
                        commands->copyBuffer({.source = &rasterizer.queueBuffer(), .destination = queueReadback.get(), .size = 32});
                        commands->copyBuffer({.source = &rasterizer.pixelBuffer(), .destination = pixelReadback.get(), .size = pixelCount * 8});
                    }
                    TextureBarrierDesc outputTransitions[2];
                    for (size_t i = 0; i < 2; ++i) {
                        outputTransitions[i] = {.texture = textures[i].get(), .before = transitions[i].after, .after = ResourceState::TransferSource};
                    }
                    commands->barrier({.textures = outputTransitions, .textureCount = 2});
                    for (size_t i = 0; i < 2; ++i) {
                        commands->copyTextureToBuffer({.texture = textures[i].get(), .buffer = readback[i].get(), .width = width, .height = height});
                    }
                    HYBRID_REQUIRE(commands->end());
                    CommandBuffer* list[] = {commands.get()};
                    HYBRID_REQUIRE(queue->submit({.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()}));
                    HYBRID_REQUIRE(fence->wait()); submitted = true;
                    std::vector<uint32_t> ids(pixelCount); std::vector<float> depths(pixelCount);
                    for (size_t i = 0; i < 2; ++i) {
                        readback[i]->invalidate(); const void* data = readback[i]->map();
                        if (!data) { return RhiTestResult::fail("Output readback failed"); }
                        std::memcpy(i == 0 ? static_cast<void*>(ids.data()) : depths.data(), data, pixelCount * 4); readback[i]->unmap();
                    }
                    if (!hybrid) { referenceIds = ids; referenceDepth = depths; continue; }
                    queueReadback->invalidate(); const auto* header = static_cast<const uint32_t*>(queueReadback->map());
                    if (!header || header[0] == 0) { return RhiTestResult::fail("GPU producer did not enqueue software triangles"); }
                    overflows += header[0] > header[1]; queueReadback->unmap();
                    pixelReadback->invalidate(); const auto* pixels = static_cast<const uint64_t*>(pixelReadback->map());
                    if (!pixels) { return RhiTestResult::fail("Software pixel readback failed"); }
                    for (size_t i = 0; i < pixelCount; ++i) { softwarePixels += uint32_t(pixels[i]) != 0; }
                    pixelReadback->unmap();
                    for (size_t i = 0; i < pixelCount; ++i) {
                        if (ids[i] != referenceIds[i] || !std::isfinite(depths[i]) || std::abs(depths[i] - referenceDepth[i]) > 2e-6f) {
                            return RhiTestResult::fail("HW/SW mismatch: reversed=" + std::to_string(reversed) + " doubleSided=" +
                                std::to_string(doubleSided) + " config=" + std::to_string(configuration) + " pixel=" + std::to_string(i) +
                                " id=" + std::to_string(ids[i]) + "/" + std::to_string(referenceIds[i]) +
                                " depth=" + std::to_string(depths[i]) + "/" + std::to_string(referenceDepth[i]));
                        }
                    }
                    ++cases;
                }
            }
        }
        if (softwarePixels < 1000 || overflows != 4) { return RhiTestResult::fail("Software writes or bounded-queue overflow were not exercised"); }
        return RhiTestResult::pass(std::to_string(cases) + " HW/SW comparisons; software pixels=" + std::to_string(softwarePixels) +
            "; clipping, shared edges, perspective depth, both windings/Z directions, thresholds and four forced overflows");
    }
};
METALLIC_REGISTER_RHI_TEST(HybridRasterDepthTest);
// Exercise the real stable compaction and argument generation on the GPU. IDs
// deliberately differ from candidate indices so a namespace remap cannot pass.
class HybridClusterBinTest final : public RhiTest {
public:
    HybridClusterBinTest() { type = RhiTestType::Rendering; name = "hybrid_cluster_stable_bins_and_indirect_limits"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Hybrid cluster bin regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless heap"); }
        HYBRID_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RhiTestResult::skip("Requires hybrid raster capabilities");
        }
        std::unique_ptr<BindlessHeap> heap;
        HYBRID_REQUIRE(device->createBindlessHeap({.maxBuffers = 2}, heap));
        BindlessHandle inputHandle, binHandle;
        HYBRID_REQUIRE(heap->allocateBuffer(inputHandle)); HYBRID_REQUIRE(heap->allocateBuffer(binHandle));
        ShaderCompileResult compiled;
        const auto compile = compileSlangShaderToSpirv({.moduleName = "HybridClusterProbe", .entryPointName = "classifyMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, compiled);
        log = compiled.diagnostics; HYBRID_REQUIRE(compile);
        std::unique_ptr<ShaderModule> shader;
        std::unique_ptr<ComputePipeline> pipeline;
        HYBRID_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4}, shader));
        HYBRID_REQUIRE(device->createComputePipeline({.computeShader = shader.get(), .usesBindlessHeap = true,
            .bindlessUserPushDataSize = 12}, pipeline));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        HYBRID_REQUIRE(device->createCommandPool(*queue, pool));
        HYBRID_REQUIRE(pool->createCommandBuffer(commands)); HYBRID_REQUIRE(device->createFence(false, fence));
        struct Case { uint32_t count; uint32_t mode; bool stream; };
        const Case cases[] = {{0, 0, false}, {513, 0, false}, {65537, 1, false},
            {65537, 2, true}, {65535u * 32u + 1u, 2, false}, {129, 3, true}};
        bool submitted = false;
        for (const auto test : cases) {
            const uint32_t capacity = std::max(1u, test.count);
            VisibilityHybridRasterizer rasterizer;
            HYBRID_REQUIRE(rasterizer.initialize(*device, 1, 1, log, 1, capacity));
            std::vector<std::array<uint32_t, 2>> candidates(capacity);
            std::array<std::vector<uint32_t>, 5> expected;
            for (uint32_t i = 0; i < test.count; ++i) {
                const uint32_t bin = test.mode == 1 ? 4u : test.mode == 2 ? 0u :
                    test.mode == 3 ? UINT32_MAX : i % 7u;
                const uint32_t id = 17u + i * 3u;
                candidates[i] = {id, bin};
                if (bin < expected.size()) { expected[bin].push_back(id); }
            }
            std::unique_ptr<Buffer> input, readback, arguments;
            HYBRID_REQUIRE(device->createBuffer({.size = capacity * 8ull, .structureStride = 8,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, input));
            void* mapped = input->map();
            if (!mapped) { return RhiTestResult::fail("Cluster input map failed"); }
            std::memcpy(mapped, candidates.data(), candidates.size() * 8); input->flush(); input->unmap();
            HYBRID_REQUIRE(heap->writeStorageBuffer(inputHandle, *input));
            HYBRID_REQUIRE(heap->writeStorageBuffer(binHandle, rasterizer.clusterBuffer()));
            const uint64_t readbackBytes = (16ull + 5ull * capacity) * 4u;
            HYBRID_REQUIRE(device->createBuffer({.size = readbackBytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}, readback));
            HYBRID_REQUIRE(device->createBuffer({.size = 60, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}, arguments));
            if (submitted) { HYBRID_REQUIRE(fence->reset()); HYBRID_REQUIRE(pool->reset()); }
            HYBRID_REQUIRE(commands->begin());
            // Capacity validation must reject oversized dispatches before recording GPU work.
            if (!hasError(rasterizer.beginClusters(*commands, 8, true, 0, capacity + 1u, test.stream), Error::InvalidArgument)) {
                return RhiTestResult::fail("Oversized cluster input was accepted");
            }
            HYBRID_REQUIRE(rasterizer.beginClusters(*commands, 8, true, 0, test.count, test.stream));
            commands->bindBindlessHeap(*heap); commands->bindComputePipeline(*pipeline);
            const uint32_t push[] = {inputHandle.index, binHandle.index, test.count};
            commands->pushBindlessData(push, sizeof(push));
            if (test.count != 0) {
                commands->dispatch(std::min(test.count, 65535u), (test.count + 65534u) / 65535u);
            }
            rasterizer.finishClusterBins(*commands);
            const BufferBarrierDesc barriers[] = {
                {.buffer = &rasterizer.clusterBuffer(), .before = ResourceState::ShaderRead, .after = ResourceState::TransferSource},
                {.buffer = &rasterizer.clusterArguments(), .before = ResourceState::IndirectArgument, .after = ResourceState::TransferSource}};
            commands->barrier({.buffers = barriers, .bufferCount = 2});
            commands->copyBuffer({.source = &rasterizer.clusterBuffer(), .destination = readback.get(), .size = readbackBytes});
            commands->copyBuffer({.source = &rasterizer.clusterArguments(), .destination = arguments.get(), .size = 60});
            HYBRID_REQUIRE(commands->end());
            CommandBuffer* list[] = {commands.get()};
            HYBRID_REQUIRE(queue->submit({.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()}));
            HYBRID_REQUIRE(fence->wait()); submitted = true;
            readback->invalidate(); arguments->invalidate();
            const auto* bins = static_cast<const uint32_t*>(readback->map());
            const auto* args = static_cast<const uint32_t*>(arguments->map());
            if (!bins || !args) { return RhiTestResult::fail("Cluster result map failed"); }
            bool valid = bins[5] == capacity && bins[12] == test.count && bins[14] == 0u;
            for (size_t bin = 0; bin < 5; ++bin) {
                const uint32_t count = static_cast<uint32_t>(expected[bin].size());
                const uint32_t groups = bin == 4 ? count : test.stream ? count * 2u : (count + 31u) / 32u;
                valid = valid && bins[bin] == count && args[bin * 3u] == std::min(groups, 65535u) &&
                    args[bin * 3u + 1u] == std::max(1u, (groups + 65534u) / 65535u) && args[bin * 3u + 2u] == 1u;
                valid = valid && std::equal(expected[bin].begin(), expected[bin].end(), bins + 16u + bin * capacity);
            }
            readback->unmap(); arguments->unmap();
            if (!valid) { return RhiTestResult::fail("Stable bins/indirect args mismatch for count=" + std::to_string(test.count)); }
        }
        return RhiTestResult::pass("Stable IDs, five bins, empty/culled/partial blocks, capacity rejection; 2D resident HW, stream HW and SW dispatch limits");
    }
};
METALLIC_REGISTER_RHI_TEST(HybridClusterBinTest);
#undef HYBRID_REQUIRE

class HybridRasterSceneTest final : public RhiTest {
public:
    HybridRasterSceneTest() { type = RhiTestType::Rendering; name = "hybrid_raster_scene_equivalence"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, false);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders"); }
        if (!result) { return RhiTestResult::fail(preview.lastLog()); }
        render::RenderGraph graph;
        graph.addNode("VisibilityBufferPass", "VBuffer", {{"path", "Asset/StandfordBunny/scene.gltf"},
            {"visualization", "triangle"}, {"camera", {{"eye", {-.0168404f, .110154f, .22f}},
                {"center", {-.0168404f, .110154f, -.00153695f}}, {"znear", .001f}, {"zfar", 10.f}, {"orthoHeight", .24f}}}});
        graph.markOutput("VBuffer.color");
        const auto node = graph.findNode("VBuffer")->id;
        size_t cases = 0;
        for (bool orthographic : {false, true}) {
            for (bool reversed : {false, true}) {
                graph.setNodeRuntimeProperty(node, "camera.projection", orthographic ? "orthographic" : "perspective");
                graph.setNodeRuntimeProperty(node, "camera.reversedZ", reversed);
                graph.setNodeRuntimeProperty(node, "hybridRaster", false);
                if (!preview.render(graph, 193, 157)) { return RhiTestResult::fail(preview.lastLog()); }
                const auto reference = preview.pixels();
                for (uint32_t configuration = 0; configuration < 9; ++configuration) {
                    const float threshold = std::array{1.f, 8.f, 32.f}[configuration % 3];
                    graph.setNodeRuntimeProperty(node, "clusterPrebin", configuration >= 3);
                    graph.setNodeRuntimeProperty(node, "asyncSoftwareRaster", configuration >= 6);
                    graph.setNodeRuntimeProperty(node, "hybridRaster", true);
                    graph.setNodeRuntimeProperty(node, "softwareRasterMaxPixels", threshold);
                    for (uint32_t frame = 0; frame < 3; ++frame) {
                        if (!preview.render(graph, 193, 157)) { return RhiTestResult::fail(preview.lastLog()); }
                        const bool independent = preview.subsystemHost()->device()->capabilities().independentComputeQueue;
                        const uint32_t branches = preview.executionStats().asyncComputeBranches;
                        if ((configuration >= 6 && independent) ? branches < 2u : branches != 0u) {
                            return RhiTestResult::fail("Async setting did not select the expected hardware/software queue topology");
                        }
                        size_t mismatches = 0;
                        for (size_t i = 0; i < reference.size(); ++i) { mismatches += reference[i] != preview.pixels()[i]; }
                        if (mismatches != 0) { return RhiTestResult::fail("Scene HW/SW triangle IDs differ at " + std::to_string(mismatches) + " pixels"); }
                        ++cases;
                    }
                }
            }
        }
        std::string log;
        if (!saveRgba8Png(context.outputDirectory / "HybridBunny.png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
            preview.width(), preview.height(), log)) { return RhiTestResult::fail(log); }
        return RhiTestResult::pass(std::to_string(cases) + " stationary HZB frames agree with hardware across asynchronous/serial cluster and triangle modes, projections, Z conventions and thresholds");
    }
};
METALLIC_REGISTER_RHI_TEST(HybridRasterSceneTest);
} // namespace
} // namespace metallic::tests
