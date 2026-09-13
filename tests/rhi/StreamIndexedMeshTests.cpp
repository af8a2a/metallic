#include "RhiTest.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {
using namespace render;
#define MESH_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

// Compare actual depth and full 32-bit visibility attachments against the
// previous duplicated-vertex shader, including its two-workgroup draw order.
class StreamIndexedMeshTest final : public RhiTest {
public:
    StreamIndexedMeshTest() { type = RhiTestType::Rendering; name = "stream_indexed_mesh_raster_equivalence"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Indexed stream mesh regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            // Match VisibilityBufferPass: Slang emits Geometry for fragment SV_PrimitiveID.
            .enableMeshShader = true, .enableGeometryShader = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires mesh/primitive-ID features and bindless heap"); }
        MESH_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RhiTestResult::skip("Requires hybrid raster capabilities");
        }
        constexpr uint32_t width = 97, height = 73, pixels = width * height, capacity = 4, pageBytes = 8192;
        enum Input { Header, Groups, Params, Pages, PageTable, Bindings, Visibility, Records, Instances, Bins, Queue, InputCount };
        const uint32_t strides[] = {sizeof(MeshletStreamGpuActiveHeader), sizeof(MeshletStreamGpuActiveGroup),
            sizeof(MeshletStreamGpuParams), 4, sizeof(StreamPageTableEntry), sizeof(MeshletStreamGpuRasterBindings),
            4, sizeof(VisibleClusterRecord), sizeof(GPUSceneGpuInstanceRecord), 4};
        const uint32_t counts[] = {1, 2, 1, pageBytes / 4, 1, 1, 2, capacity, 2, 16 + 5 * capacity};
        std::unique_ptr<BindlessHeap> heap;
        MESH_REQUIRE(device->createBindlessHeap({.maxBuffers = InputCount}, heap));
        std::array<BindlessHandle, InputCount> handles;
        for (auto& handle : handles) { MESH_REQUIRE(heap->allocateBuffer(handle)); }
        std::array<std::unique_ptr<Buffer>, Queue> inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            MESH_REQUIRE(device->createBuffer({.size = uint64_t(strides[i]) * counts[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, inputs[i]));
            MESH_REQUIRE(heap->writeStorageBuffer(handles[i], *inputs[i]));
        }
        const auto upload = [&](size_t index, const void* data, size_t size) -> Result {
            void* mapped = inputs[index]->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data, size); inputs[index]->flush(); inputs[index]->unmap(); return {};
        };
        VisibilityHybridRasterizer rasterizer;
        MESH_REQUIRE(rasterizer.initialize(*device, width, height, log));
        MESH_REQUIRE(heap->writeStorageBuffer(handles[Queue], rasterizer.queueBuffer()));
        std::array<std::unique_ptr<ShaderModule>, 4> shaders;
        const char* entries[] = {"referenceStreamMeshMain", "referenceStreamFragmentMain",
            "gpuDrivenStreamAssetMeshMain", "gpuDrivenStreamAssetFragmentMain"};
        const char* capabilities[] = {"spvMeshShadingEXT"};
        const char* paths[] = {PROJECT_SOURCE_DIR "/Shaders"};
        for (size_t i = 0; i < shaders.size(); ++i) {
            ShaderCompileResult compiled;
            const auto result = compileSlangShaderToSpirv({
                .moduleName = i < 2 ? "StreamMeshProbe" : "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = entries[i], .searchPath = i < 2 ? PROJECT_SOURCE_DIR "/tests/rhi/shaders" : paths[0],
                .additionalSearchPaths = paths, .additionalSearchPathCount = 1,
                .capabilities = capabilities, .capabilityCount = 1}, compiled);
            log = compiled.diagnostics; MESH_REQUIRE(result);
            MESH_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4}, shaders[i]));
        }
        std::array<std::unique_ptr<GraphicsPipeline>, 4> pipelines;
        for (uint32_t reversed = 0; reversed < 2; ++reversed) {
            for (uint32_t indexed = 0; indexed < 2; ++indexed) {
                MESH_REQUIRE(device->createGraphicsPipeline({.meshShader = shaders[indexed * 2].get(),
                    .fragmentShader = shaders[indexed * 2 + 1].get(), .colorFormat = Format::R32Uint,
                    .depthStencilFormat = Format::D32Sfloat,
                    .rasterization = {.cullMode = CullMode::None, .frontFace = FrontFace::CounterClockwise},
                    .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true,
                        .depthCompareOp = reversed ? CompareOp::GreaterEqual : CompareOp::LessEqual},
                    .usesBindlessHeap = true}, pipelines[reversed * 2 + indexed]));
            }
        }
        std::array<std::unique_ptr<Texture>, 2> textures;
        std::array<std::unique_ptr<TextureView>, 2> views;
        std::array<std::unique_ptr<Buffer>, 2> readbacks;
        for (size_t i = 0; i < 2; ++i) {
            const auto format = i == 0 ? Format::R32Uint : Format::D32Sfloat;
            MESH_REQUIRE(device->createTexture({.usage = TextureUsageBits::TransferSource |
                (i == 0 ? TextureUsageBits::ColorAttachment : TextureUsageBits::DepthStencilAttachment),
                .format = format, .width = width, .height = height}, textures[i]));
            MESH_REQUIRE(device->createTextureView(*textures[i], {.format = format}, views[i]));
            MESH_REQUIRE(device->createBuffer({.size = pixels * 4, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}, readbacks[i]));
        }
        std::unique_ptr<Buffer> queueReadback;
        MESH_REQUIRE(device->createBuffer({.size = 4, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}, queueReadback));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        MESH_REQUIRE(device->createCommandPool(*queue, pool));
        MESH_REQUIRE(pool->createCommandBuffer(commands)); MESH_REQUIRE(device->createFence(false, fence));
        bool submitted = false, sawSecondChunk = false, sawHighId = false;
        uint32_t softwareTriangles = 0, comparisons = 0;
        const uint32_t partialCounts[] = {0, 1, 63, 64, 65, 127, 128, 65};
        for (uint32_t test = 0; test < 8; ++test) {
            const bool ortho = (test & 1) != 0, reversed = (test & 2) != 0, doubleSided = (test & 4) != 0;
            std::vector<uint8_t> page(pageBytes);
            scene::MeshletStreamPayloadHeader h;
            h.clusterCount = 2; h.vertexCount = 256; h.triangleIndexCount = 768;
            h.clusterOffsetBytes = sizeof(h); h.positionOffsetBytes = h.clusterOffsetBytes + 2 * sizeof(scene::MeshletStreamPayloadCluster);
            h.triangleOffsetBytes = h.positionOffsetBytes + h.vertexCount * 16;
            h.payloadByteSize = h.triangleOffsetBytes + h.triangleIndexCount; h.attributeFlags = 1;
            std::memcpy(page.data(), &h, sizeof(h));
            for (uint32_t c = 0; c < 2; ++c) {
                scene::MeshletStreamPayloadCluster cluster;
                cluster.vertexOffset = c * 128; cluster.vertexCount = 128;
                cluster.triangleOffset = c * 384; cluster.triangleCount = c == 0 ? 128 : partialCounts[test];
                cluster.boundingSphere[2] = 2; cluster.boundingSphere[3] = 12; cluster.coneApexCutoff[3] = 1;
                std::memcpy(page.data() + h.clusterOffsetBytes + c * sizeof(cluster), &cluster, sizeof(cluster));
                for (uint32_t v = 0; v < 128; ++v) {
                    const uint32_t grid = v == 127 ? 80 : v % 81;
                    const float p[] = {-.9f + float(grid % 9) * .225f, -.9f + float(grid / 9) * .225f,
                        grid == 0 ? .005f : grid == 8 ? 12.f : 2.f, 1.f};
                    std::memcpy(page.data() + h.positionOffsetBytes + (c * 128 + v) * 16, p, sizeof(p));
                }
                for (uint32_t t = 0; t < 128; ++t) {
                    const uint32_t a = (t / 2 / 8) * 9 + (t / 2 % 8);
                    std::array<uint8_t, 3> tri = t % 2 == 0 ? std::array<uint8_t, 3>{uint8_t(a), uint8_t(a + 1), uint8_t(a + 9)} :
                        std::array<uint8_t, 3>{uint8_t(a + 1), uint8_t(a + 10), uint8_t(a + 9)};
                    if (t % 7 == 0) { std::swap(tri[1], tri[2]); }
                    for (auto& index : tri) { if (index == 80) { index = 127; } }
                    if (t == 5) { tri[0] = 255; } // Existing invalid-index-to-zero fallback.
                    std::memcpy(page.data() + h.triangleOffsetBytes + c * 384 + t * 3, tri.data(), 3);
                }
            }
            MeshletStreamGpuActiveHeader header{.activeGroupCount = 2, .activeGroupCapacity = 2, .maxActiveGroupClusters = 2};
            std::array<MeshletStreamGpuActiveGroup, 2> groups;
            std::array<GPUSceneGpuInstanceRecord, 2> instances;
            for (uint32_t i = 0; i < 2; ++i) {
                groups[i].clusterCount = 2; groups[i].clusterSelectionMask = 3; groups[i].gpuSceneInstanceIndex = i;
                groups[i].world0[0] = groups[i].world1[1] = groups[i].world2[2] = groups[i].world3[3] = 1;
                groups[i].world3[0] = i == 0 ? 0.f : .13f;
                instances[i].identity[3] = doubleSided ? 2 : 0;
            }
            MeshletStreamGpuParams params;
            params.center[2] = 1; params.upProjection[1] = 1; params.upProjection[3] = ortho ? 1.f : 0.f;
            params.viewport[0] = float(width) / height; params.viewport[1] = width; params.viewport[2] = height; params.viewport[3] = 1.57079632679f;
            params.clipOrtho[0] = .1f; params.clipOrtho[1] = 10; params.clipOrtho[2] = 2.4f; params.clipOrtho[3] = reversed ? 1.f : 0.f;
            params.pageBufferBytes = pageBytes; params.drawTaskCount = capacity * 2; params.scenePageCount = 1;
            if (test == 7) {
                std::memcpy(params.renderCenter, params.center, 16); std::memcpy(params.renderUpProjection, params.upProjection, 16);
                std::memcpy(params.renderViewport, params.viewport, 16); std::memcpy(params.renderClipOrtho, params.clipOrtho, 16);
                params.renderEye[0] = .1f; params.renderEye[3] = .35f; params.renderCenter[3] = -.2f;
            }
            const StreamPageTableEntry entry{.deviceOffsetAndState = packStreamPageTableEntry(0, MeshletStreamPageResidencyState::Resident)};
            MeshletStreamGpuRasterBindings bindings{.visibleClusterBuffer = handles[Records].index,
                .instanceVisibilityBuffer = handles[Visibility].index, .visibleRecordBase = kVisibilityMaxRecordCount - capacity,
                .visibleRecordCapacity = capacity, .gpuSceneInstanceBuffer = handles[Instances].index};
            const std::array<uint32_t, 2> visibility{test == 7 ? 3u : 1u, test == 7 ? 3u : 1u};
            std::array<uint32_t, 16 + 5 * capacity> bins{};
            bins[0] = capacity; bins[5] = capacity;
            // Deliberately reordered records exercise stable primitive order at equal depth.
            bins[16] = 2; bins[17] = 0; bins[18] = 3; bins[19] = 1;
            MESH_REQUIRE(upload(Header, &header, sizeof(header))); MESH_REQUIRE(upload(Groups, groups.data(), sizeof(groups)));
            MESH_REQUIRE(upload(Params, &params, sizeof(params))); MESH_REQUIRE(upload(Pages, page.data(), page.size()));
            MESH_REQUIRE(upload(PageTable, &entry, sizeof(entry))); MESH_REQUIRE(upload(Bindings, &bindings, sizeof(bindings)));
            MESH_REQUIRE(upload(Visibility, visibility.data(), sizeof(visibility))); MESH_REQUIRE(upload(Instances, instances.data(), sizeof(instances)));
            MESH_REQUIRE(upload(Bins, bins.data(), sizeof(bins)));
            for (uint32_t mode = 0; mode < 3; ++mode) {
                const bool prebinned = mode == 0, hybridQueue = mode == 2;
                std::array<std::vector<uint32_t>, 2> reference;
                for (uint32_t indexed = 0; indexed < 2; ++indexed) {
                    if (submitted) { MESH_REQUIRE(fence->reset()); MESH_REQUIRE(pool->reset()); }
                    MESH_REQUIRE(commands->begin());
                    const TextureBarrierDesc transitions[] = {
                        {.texture = textures[0].get(), .before = submitted ? ResourceState::TransferSource : ResourceState::Undefined, .after = ResourceState::ColorAttachment},
                        {.texture = textures[1].get(), .before = submitted ? ResourceState::TransferSource : ResourceState::Undefined, .after = ResourceState::DepthStencilAttachment}};
                    commands->barrier({.textures = transitions, .textureCount = 2});
                    if (hybridQueue) { rasterizer.begin(*commands, 8, reversed); }
                    const RenderingAttachmentDesc color{.view = views[0].get(), .state = ResourceState::ColorAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
                    const RenderingAttachmentDesc depth{.view = views[1].get(), .state = ResourceState::DepthStencilAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = reversed ? 0.f : 1.f};
                    commands->beginRendering({.renderArea = {.width = width, .height = height},
                        .colorAttachments = &color, .colorAttachmentCount = 1, .depthStencilAttachment = &depth});
                    commands->setViewport({.width = float(width), .height = float(height), .maxDepth = 1.f});
                    commands->setScissor({.width = width, .height = height});
                    commands->bindBindlessHeap(*heap); commands->bindGraphicsPipeline(*pipelines[(reversed ? 2 : 0) + indexed]);
                    MeshletStreamUserPush push{.pageBuffer = handles[Pages].index, .activeGroupBuffer = handles[Groups].index,
                        .pageTableBuffer = handles[PageTable].index, .paramsBuffer = handles[Params].index,
                        .activeHeaderBuffer = handles[Header].index, .traversalPhase = test == 7 ? 1u : 0u,
                        .rasterBindingsBuffer = handles[Bindings].index,
                        .hybridQueueBuffer = hybridQueue ? handles[Queue].index : UINT32_MAX,
                        .hybridClusterBuffer = prebinned ? handles[Bins].index : UINT32_MAX};
                    commands->pushBindlessData(&push, sizeof(push));
                    commands->drawMeshTasks(prebinned && indexed ? capacity : capacity * 2);
                    commands->endRendering();
                    if (hybridQueue) {
                        MESH_REQUIRE(rasterizer.resolve(*commands, *textures[0], *views[0], *textures[1], *views[1]));
                        const BufferBarrierDesc copy{.buffer = &rasterizer.queueBuffer(), .before = ResourceState::ShaderRead, .after = ResourceState::TransferSource};
                        commands->barrier({.buffers = &copy, .bufferCount = 1});
                        commands->copyBuffer({.source = &rasterizer.queueBuffer(), .destination = queueReadback.get(), .size = 4});
                        const BufferBarrierDesc restore{.buffer = &rasterizer.queueBuffer(), .before = ResourceState::TransferSource, .after = ResourceState::ShaderRead};
                        commands->barrier({.buffers = &restore, .bufferCount = 1});
                    }
                    for (size_t i = 0; i < 2; ++i) {
                        const TextureBarrierDesc copy{.texture = textures[i].get(), .before = transitions[i].after, .after = ResourceState::TransferSource};
                        commands->barrier({.textures = &copy, .textureCount = 1});
                        commands->copyTextureToBuffer({.texture = textures[i].get(), .buffer = readbacks[i].get(), .width = width, .height = height});
                    }
                    MESH_REQUIRE(commands->end());
                    CommandBuffer* list[] = {commands.get()};
                    MESH_REQUIRE(queue->submit({.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()}));
                    MESH_REQUIRE(fence->wait()); submitted = true;
                    for (size_t attachment = 0; attachment < 2; ++attachment) {
                        readbacks[attachment]->invalidate(); const auto* mapped = static_cast<const uint32_t*>(readbacks[attachment]->map());
                        if (!mapped) { return RhiTestResult::fail("Cannot map mesh output"); }
                        std::vector<uint32_t> actual(mapped, mapped + pixels); readbacks[attachment]->unmap();
                        if (indexed == 0) { reference[attachment] = actual; }
                        else if (actual != reference[attachment]) {
                            const auto mismatch = std::mismatch(actual.begin(), actual.end(), reference[attachment].begin()).first - actual.begin();
                            return RhiTestResult::fail("Mesh attachment mismatch: case=" + std::to_string(test) + " mode=" + std::to_string(mode) +
                                " attachment=" + std::to_string(attachment) + " pixel=" + std::to_string(mismatch));
                        }
                        if (attachment == 0) {
                            if (std::count_if(actual.begin(), actual.end(), [](uint32_t id) { return id != 0; }) < 100) {
                                return RhiTestResult::fail("Insufficient mesh coverage");
                            }
                            for (uint32_t id : actual) { sawSecondChunk |= id != 0 && (id & 127) >= 64; sawHighId |= (id & 0x80000000u) != 0; }
                        }
                    }
                    if (hybridQueue) {
                        queueReadback->invalidate(); const auto* mapped = static_cast<const uint32_t*>(queueReadback->map());
                        if (!mapped || *mapped == 0) { return RhiTestResult::fail("Legacy queue was not exercised"); }
                        softwareTriangles += *mapped; queueReadback->unmap();
                    }
                }
                ++comparisons;
            }
        }
        if (!sawSecondChunk || !sawHighId || softwareTriangles == 0) { return RhiTestResult::fail("Missing primitive ID or queue coverage"); }
        return RhiTestResult::pass(std::to_string(comparisons) +
            " bit-exact HW/legacy-queue attachment comparisons: indexed vertices, 0/1/63/64/65/127/128 triangles, high IDs, clipping, equal depth, both windings/Z and render jitter");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamIndexedMeshTest);
#undef MESH_REQUIRE
} // namespace
} // namespace metallic::tests
