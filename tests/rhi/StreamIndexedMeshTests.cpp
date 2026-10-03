#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "RHITest.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {
using namespace render;
#define MESH_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RHITestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

// Compare actual depth and full 32-bit visibility attachments against the
// previous duplicated-vertex shader, including its two-workgroup draw order.
class StreamIndexedMeshTest final : public RHITest {
public:
    StreamIndexedMeshTest() { type = RHITestType::Rendering; name = "stream_indexed_mesh_raster_equivalence"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Indexed stream mesh regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            // Match VisibilityBufferPass: Slang emits Geometry for fragment SV_PrimitiveID.
            .enableMeshShader = true, .enableGeometryShader = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires mesh/primitive-ID features and bindless heap"); }
        MESH_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RHITestResult::skip("Requires hybrid raster capabilities");
        }
        constexpr uint32_t width = 97, height = 73, pixels = width * height, capacity = 4, pageBytes = 8192;
        enum Input { Header, Groups, Params, Pages, PageTable, Bindings, Visibility, Records, Instances, Bins, Queue, InputCount };
        const uint32_t strides[] = {sizeof(MeshletStreamGPUActiveHeader), sizeof(MeshletStreamGPUActiveGroup),
            sizeof(MeshletStreamGPUParams), 4, sizeof(StreamPageTableEntry), sizeof(MeshletStreamGPURasterBindings),
            4, sizeof(CompactStreamVisibleRecord), sizeof(GPUSceneGPUInstanceRecord), 4};
        const uint32_t counts[] = {1, 2, 1, pageBytes / 4, 1, 1, 2, capacity, 2, 16 + 9 * capacity + 5};
        std::unique_ptr<BindlessHeap> heap;
        MESH_REQUIRE(device->createBindlessHeap({.maxBuffers = InputCount}).transform([&](auto rhiValue) { heap = std::move(rhiValue); }));
        std::array<BindlessHandle, InputCount> handles;
        for (auto& handle : handles) { MESH_REQUIRE(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { handle = std::move(rhiValue); })); }
        std::array<std::unique_ptr<Buffer>, Queue> inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            MESH_REQUIRE(device->createBuffer({.size = uint64_t(strides[i]) * counts[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { inputs[i] = std::move(rhiValue); }));
            MESH_REQUIRE((*inputs[i]).slice().and_then([&](const auto& bufferSlice) { return heap->writeStorageBuffer(handles[i], bufferSlice); }));
        }
        const auto upload = [&](size_t index, const void* data, size_t size) -> Result<> {
            void* mapped = inputs[index]->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data, size); inputs[index]->flush(); inputs[index]->unmap(); return {};
        };
        VisibilityHybridRasterizer rasterizer;
        MESH_REQUIRE(rasterizer.initialize(*device, width, height, log));
        MESH_REQUIRE((rasterizer.queueBuffer()).slice().and_then([&](const auto& bufferSlice) { return heap->writeStorageBuffer(handles[Queue], bufferSlice); }));
        std::array<std::unique_ptr<ShaderModule>, 4> shaders;
        const char* entries[] = {"referenceStreamMeshMain", "referenceStreamFragmentMain",
            "gpuDrivenStreamAssetMeshMain", "gpuDrivenStreamAssetFragmentMain"};
        const char* capabilities[] = {"spvMeshShadingEXT"};
        const char* paths[] = {PROJECT_SOURCE_DIR "/Shaders"};
        for (size_t i = 0; i < shaders.size(); ++i) {
            ShaderCompileResult compiled;
            const auto result = compileSlangShaderToSpirv({
                .moduleName = i < 2 ? "StreamMeshProbe" : "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = entries[i],
                .searchPath = i < 2 ? PROJECT_SOURCE_DIR "/tests/rhi/shaders" : paths[0],
                .additionalSearchPaths = {paths, 1},
                .capabilities = {capabilities, 1},
            }, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); });
            log = compiled.diagnostics; MESH_REQUIRE(result);
            MESH_REQUIRE(device->createShaderModule({.spirv = compiled.spirv}).transform([&](auto rhiValue) { shaders[i] = std::move(rhiValue); }));
        }
        std::array<std::unique_ptr<GraphicsPipeline>, 4> pipelines;
        for (uint32_t reversed = 0; reversed < 2; ++reversed) {
            for (uint32_t indexed = 0; indexed < 2; ++indexed) {
                MESH_REQUIRE(device->createGraphicsPipeline({
                    .meshShader = {shaders[indexed * 2].get()},
                    .fragmentShader = {shaders[indexed * 2 + 1].get()},
                    .colorFormat = Format::R32Uint,
                    .depthStencilFormat = Format::D32Sfloat,
                    .rasterization = {.cullMode = CullMode::None, .frontFace = FrontFace::CounterClockwise},
                    .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true,
                        .depthCompareOp = reversed ? CompareOp::GreaterEqual : CompareOp::LessEqual},
                    .usesBindlessHeap = true,
                }).transform([&](auto rhiValue) { pipelines[reversed * 2 + indexed] = std::move(rhiValue); }));
            }
        }
        std::array<std::unique_ptr<Texture>, 2> textures;
        std::array<std::unique_ptr<TextureView>, 2> views;
        std::array<std::unique_ptr<Buffer>, 2> readbacks;
        for (size_t i = 0; i < 2; ++i) {
            const auto format = i == 0 ? Format::R32Uint : Format::D32Sfloat;
            MESH_REQUIRE(device->createTexture({.usage = TextureUsageBits::TransferSource |
                (i == 0 ? TextureUsageBits::ColorAttachment : TextureUsageBits::DepthStencilAttachment),
                .format = format, .width = width, .height = height}).transform([&](auto rhiValue) { textures[i] = std::move(rhiValue); }));
            MESH_REQUIRE(device->createTextureView(*textures[i], {.format = format}).transform([&](auto rhiValue) { views[i] = std::move(rhiValue); }));
            MESH_REQUIRE(device->createBuffer({.size = pixels * 4, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readbacks[i] = std::move(rhiValue); }));
        }
        std::unique_ptr<Buffer> queueReadback;
        MESH_REQUIRE(device->createBuffer({.size = 4, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { queueReadback = std::move(rhiValue); }));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        MESH_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        MESH_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); })); MESH_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
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
            MeshletStreamGPUActiveHeader header{.activeGroupCount = 2, .activeGroupCapacity = 2, .maxActiveGroupClusters = 2};
            std::array<MeshletStreamGPUActiveGroup, 2> groups;
            std::array<GPUSceneGPUInstanceRecord, 2> instances;
            for (uint32_t i = 0; i < 2; ++i) {
                groups[i].clusterCount = 2; groups[i].clusterSelectionMask = 3; groups[i].gpuSceneInstanceIndex = i;
                groups[i].world0[0] = groups[i].world1[1] = groups[i].world2[2] = groups[i].world3[3] = 1;
                groups[i].world3[0] = i == 0 ? 0.f : .13f;
                instances[i].identity[3] = doubleSided ? 2 : 0;
            }
            MeshletStreamGPUParams params;
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
            MeshletStreamGPURasterBindings bindings{.visibleClusterBuffer = handles[Records].shaderIndex,
                .instanceVisibilityBuffer = handles[Visibility].shaderIndex, .visibleRecordBase = kVisibilityMaxRecordCount - capacity,
                .visibleRecordCapacity = capacity, .gpuSceneInstanceBuffer = handles[Instances].shaderIndex};
            const std::array<uint32_t, 2> visibility{test == 7 ? 3u : 1u, test == 7 ? 3u : 1u};
            std::array<uint32_t, 16 + 9 * capacity + 5> bins{};
            bins[0] = capacity; bins[5] = capacity;
            // Deliberately reordered records exercise stable primitive order at equal depth.
            bins[16] = 2; bins[17] = 0; bins[18] = 3; bins[19] = 1;
            MESH_REQUIRE(upload(Header, &header, sizeof(header))); MESH_REQUIRE(upload(Groups, groups.data(), sizeof(groups)));
            MESH_REQUIRE(upload(Params, &params, sizeof(params))); MESH_REQUIRE(upload(Pages, page.data(), page.size()));
            MESH_REQUIRE(upload(PageTable, &entry, sizeof(entry))); MESH_REQUIRE(upload(Bindings, &bindings, sizeof(bindings)));
            MESH_REQUIRE(upload(Visibility, visibility.data(), sizeof(visibility))); MESH_REQUIRE(upload(Instances, instances.data(), sizeof(instances)));
            for (uint32_t mode = 0; mode < 4; ++mode) {
                const bool fallback = mode == 3;
                const bool prebinned = mode == 0 || fallback, hybridQueue = mode == 2;
                if (fallback) {
                    // Four record IDs through two candidate slots. Bin 0 holds group masks.
                    bins.fill(0); bins[0] = capacity; bins[5] = 2; bins[13] = 2;
                    bins[14] = 2; bins[15] = 2; bins[16] = bins[17] = 3;
                }
                MESH_REQUIRE(upload(Bins, bins.data(), sizeof(bins)));
                std::array<std::vector<uint32_t>, 2> reference;
                for (uint32_t indexed = 0; indexed < 2; ++indexed) {
                    if (submitted) { MESH_REQUIRE(fence->reset()); MESH_REQUIRE(pool->reset()); }
                    MESH_REQUIRE(commands->begin());
                    const TextureBarrierDesc transitions[] = {
                        {
                            .texture = textures[0].get(),
                            .oldLayout = metallic::render::textureLayoutForResourceState(submitted ? ResourceState::TransferSource : ResourceState::Undefined),
                            .newLayout = TextureLayout::ColorAttachment,
                            .before = metallic::render::resourceSyncScope(submitted ? ResourceState::TransferSource : ResourceState::Undefined, metallic::render::PipelineStageBits::AllCommands),
                            .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
                        },
                        {
                            .texture = textures[1].get(),
                            .oldLayout = metallic::render::textureLayoutForResourceState(submitted ? ResourceState::TransferSource : ResourceState::Undefined),
                            .newLayout = TextureLayout::DepthStencilAttachment,
                            .before = metallic::render::resourceSyncScope(submitted ? ResourceState::TransferSource : ResourceState::Undefined, metallic::render::PipelineStageBits::AllCommands),
                            .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite},
                        }};
                    if (auto commandResult = commands->synchronize({.textures = {transitions, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                    if (hybridQueue) { if (auto commandResult = rasterizer.begin(*commands, 8, reversed); !commandResult) { return RHITestResult::fail(std::string("begin failed: ") + render::resultToString(commandResult)); } }
                    const RenderingAttachmentDesc color{.view = views[0].get(), .layout = TextureLayout::ColorAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
                    const RenderingAttachmentDesc depth{.view = views[1].get(), .layout = TextureLayout::DepthStencilAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = reversed ? 0.f : 1.f};
                    if (auto commandResult = commands->beginRendering({
                        .renderArea = {.width = width, .height = height},
                        .colorAttachments = {&color, 1},
                        .depthStencilAttachment = &depth,
                    }); !commandResult) { return RHITestResult::fail(std::string("beginRendering failed: ") + render::resultToString(commandResult)); }
                    if (auto commandResult = commands->setViewport({.width = float(width), .height = float(height), .maxDepth = 1.f}); !commandResult) { return RHITestResult::fail(std::string("setViewport failed: ") + render::resultToString(commandResult)); }
                    commands->setScissor({.width = width, .height = height});
                    commands->bindBindlessHeap(*heap); if (auto commandResult = commands->bindExecution((pipelines[(reversed ? 2 : 0) + indexed])->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
                    MeshletStreamUserPush push{.pageBuffer = handles[Pages].shaderIndex, .activeGroupBuffer = handles[Groups].shaderIndex,
                        .pageTableBuffer = handles[PageTable].shaderIndex, .paramsBuffer = handles[Params].shaderIndex,
                        .activeHeaderBuffer = handles[Header].shaderIndex, .traversalPhase = test == 7 ? 1u : 0u,
                        .rasterBindingsBuffer = handles[Bindings].shaderIndex,
                        .hybridQueueBuffer = hybridQueue ? handles[Queue].shaderIndex : UINT32_MAX,
                        .hybridClusterBuffer = prebinned && (!fallback || indexed != 0) ? handles[Bins].shaderIndex : UINT32_MAX};
                    commands->pushBindlessData(&push, sizeof(push));
                    if (auto commandResult = commands->drawMeshTasks(prebinned && indexed ? capacity : capacity * 2); !commandResult) { return RHITestResult::fail(std::string("drawMeshTasks failed: ") + render::resultToString(commandResult)); }
                    commands->endRendering();
                    if (hybridQueue) {
                        MESH_REQUIRE(rasterizer.resolve(*commands, *textures[0], *views[0], *textures[1], *views[1]));
                        const BufferBarrierDesc copy{
                            .buffer = &rasterizer.queueBuffer(),
                            .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                        };
                        if (auto commandResult = commands->synchronize({.buffers = {&copy, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                        {
                            auto sourceSlice = (&rasterizer.queueBuffer())->slice({0, 4});
                            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                            auto destinationSlice = queueReadback.get()->slice({0, 4});
                            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                            if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
                        }
                        const BufferBarrierDesc restore{
                            .buffer = &rasterizer.queueBuffer(),
                            .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                            .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                        };
                        if (auto commandResult = commands->synchronize({.buffers = {&restore, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                    }
                    for (size_t i = 0; i < 2; ++i) {
                        const TextureBarrierDesc copy{
                            .texture = textures[i].get(),
                            .oldLayout = transitions[i].newLayout,
                            .newLayout = TextureLayout::TransferSource,
                            .before = transitions[i].after,
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                        };
                        if (auto commandResult = commands->synchronize({.textures = {&copy, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                        if (auto commandResult = (readbacks[i].get())->slice().and_then([&](const auto& bufferSlice) { return commands->copyTextureToBuffer({.texture = textures[i].get(), .buffer = bufferSlice, .width = width, .height = height}); }); !commandResult) { return RHITestResult::fail(std::string("copyTextureToBuffer failed: ") + render::resultToString(commandResult)); }
                    }
                    MESH_REQUIRE(commands->end());
                    CommandBuffer* list[] = {commands.get()};
                    MESH_REQUIRE(queue->submit({.commandBuffers = {list, 1}, .signalFence = fence.get()}));
                    MESH_REQUIRE(fence->wait()); submitted = true;
                    for (size_t attachment = 0; attachment < 2; ++attachment) {
                        readbacks[attachment]->invalidate(); const auto* mapped = static_cast<const uint32_t*>(readbacks[attachment]->map());
                        if (!mapped) { return RHITestResult::fail("Cannot map mesh output"); }
                        std::vector<uint32_t> actual(mapped, mapped + pixels); readbacks[attachment]->unmap();
                        if (indexed == 0) { reference[attachment] = actual; }
                        else if (actual != reference[attachment]) {
                            const auto mismatch = std::mismatch(actual.begin(), actual.end(), reference[attachment].begin()).first - actual.begin();
                            return RHITestResult::fail("Mesh attachment mismatch: case=" + std::to_string(test) + " mode=" + std::to_string(mode) +
                                " attachment=" + std::to_string(attachment) + " pixel=" + std::to_string(mismatch));
                        }
                        if (attachment == 0) {
                            if (std::count_if(actual.begin(), actual.end(), [](uint32_t id) { return id != 0; }) < 100) {
                                return RHITestResult::fail("Insufficient mesh coverage");
                            }
                            for (uint32_t id : actual) { sawSecondChunk |= id != 0 && (id & 127) >= 64; sawHighId |= (id & 0x80000000u) != 0; }
                        }
                    }
                    if (hybridQueue) {
                        queueReadback->invalidate(); const auto* mapped = static_cast<const uint32_t*>(queueReadback->map());
                        if (!mapped || *mapped == 0) { return RHITestResult::fail("Legacy queue was not exercised"); }
                        softwareTriangles += *mapped; queueReadback->unmap();
                    }
                }
                ++comparisons;
            }
        }
        if (!sawSecondChunk || !sawHighId || softwareTriangles == 0) { return RHITestResult::fail("Missing primitive ID or queue coverage"); }
        return RHITestResult::pass(std::to_string(comparisons) +
            " bit-exact HW/overflow/legacy-queue attachment comparisons: indexed vertices, 0/1/63/64/65/127/128 triangles, high IDs, clipping, equal depth, both windings/Z and render jitter");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamIndexedMeshTest);
#undef MESH_REQUIRE
} // namespace
} // namespace metallic::tests
