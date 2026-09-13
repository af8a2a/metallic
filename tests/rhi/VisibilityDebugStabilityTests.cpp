#include "RhiTest.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <cstring>

namespace metallic::tests {
namespace {
using namespace render;
#define DEBUG_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

class VisibilityDebugStabilityTest final : public RhiTest {
public:
    VisibilityDebugStabilityTest()
    {
        type = RhiTestType::Rendering;
        name = "visibility_debug_stable_geometry_identity";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        std::unique_ptr<Device> device;
        const Result created = createDevice({.applicationName = "Visibility debug stability",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless heap"); }
        DEBUG_REQUIRE(created);
        constexpr uint32_t width = 12, capacity = 32;
        enum Input { Params, Resident, Clusters, Stream, Groups, InputCount };
        const uint32_t strides[] = {sizeof(builtin_pass::GPUDrivenPreviewGpuParams), sizeof(VisibleClusterRecord),
            80, sizeof(VisibleClusterRecord), sizeof(MeshletStreamGpuActiveGroup)};
        const uint32_t counts[] = {1, capacity, 8, capacity, 8};
        std::unique_ptr<BindlessHeap> heap;
        DEBUG_REQUIRE(device->createBindlessHeap({.maxSampledImages = 2, .maxBuffers = InputCount}, heap));
        std::array<BindlessHandle, InputCount> handles;
        std::array<std::unique_ptr<Buffer>, InputCount> buffers;
        for (uint32_t i = 0; i < InputCount; ++i) {
            DEBUG_REQUIRE(device->createBuffer({.size = uint64_t(strides[i]) * counts[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, buffers[i]));
            DEBUG_REQUIRE(heap->allocateBuffer(handles[i]));
            DEBUG_REQUIRE(heap->writeStorageBuffer(handles[i], *buffers[i]));
        }
        const auto upload = [](Buffer& buffer, const void* data, size_t size) -> Result {
            void* mapped = buffer.map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data, size);
            buffer.flush(); buffer.unmap(); return {};
        };
        std::unique_ptr<ShaderModule> vertex, fragment;
        for (uint32_t i = 0; i < 2; ++i) {
            ShaderCompileResult compiled;
            const Result result = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityBufferComposite",
                .entryPointName = i == 0 ? "visibilityBufferCompositeVertexMain" : "visibilityBufferCompositeFragmentMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled);
            log = compiled.diagnostics; DEBUG_REQUIRE(result);
            DEBUG_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4},
                i == 0 ? vertex : fragment));
        }
        std::unique_ptr<GraphicsPipeline> pipeline;
        DEBUG_REQUIRE(device->createGraphicsPipeline({.vertexShader = vertex.get(), .fragmentShader = fragment.get(),
            .colorFormat = Format::Rgba8Unorm, .rasterization = {.cullMode = CullMode::None}, .usesBindlessHeap = true}, pipeline));
        std::array<std::unique_ptr<Texture>, 3> textures;
        std::array<std::unique_ptr<TextureView>, 3> views;
        std::array<std::unique_ptr<Buffer>, 2> uploads;
        std::array<BindlessHandle, 2> images;
        for (uint32_t i = 0; i < 3; ++i) {
            const Format format = i == 0 ? Format::R32Uint : i == 1 ? Format::R32Sfloat : Format::Rgba8Unorm;
            DEBUG_REQUIRE(device->createTexture({.usage = i < 2 ? TextureUsageBits::Sampled | TextureUsageBits::TransferDestination :
                TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
                .format = format, .width = width, .height = 1}, textures[i]));
            DEBUG_REQUIRE(device->createTextureView(*textures[i], {.format = format}, views[i]));
            if (i < 2) {
                DEBUG_REQUIRE(heap->allocateSampledImage(images[i]));
                DEBUG_REQUIRE(heap->writeSampledImage(images[i], *views[i], ResourceState::ShaderRead));
                DEBUG_REQUIRE(device->createBuffer({.size = width * 4, .usage = BufferUsageBits::TransferSource,
                    .memoryLocation = MemoryLocation::HostUpload}, uploads[i]));
            }
        }
        std::unique_ptr<Buffer> readback;
        DEBUG_REQUIRE(device->createBuffer({.size = width * 4, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}, readback));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        DEBUG_REQUIRE(device->createCommandPool(*queue, pool));
        DEBUG_REQUIRE(pool->createCommandBuffer(commands));
        DEBUG_REQUIRE(device->createFence(false, fence));
        bool submitted = false;
        std::array<float, width> depth;
        depth.fill(.25f);
        DEBUG_REQUIRE(upload(*uploads[1], depth.data(), sizeof(depth)));
        for (uint32_t mode = 1; mode <= 6; ++mode) {
            std::array<uint32_t, width> reference{};
            // Same geometry, but records, active groups, resident base and physical pages move.
            for (uint32_t permutation = 0; permutation < 2; ++permutation) {
                const uint32_t base = permutation == 0 ? 8 : 16;
                const auto slot = [permutation](uint32_t i) { return permutation == 0 ? i : 15u - i; };
                const auto groupSlot = [permutation](uint32_t i) { return permutation == 0 ? i : 7u - i; };
                std::array<VisibleClusterRecord, capacity> resident{}, stream{};
                std::array<MeshletStreamGpuActiveGroup, 8> groups{};
                std::array<std::array<uint32_t, 20>, 8> meshlets{};
                meshlets[2][4] = 3; meshlets[4][4] = 1;
                resident[slot(0)] = {.clusterIndex = 2, .instanceIndex = 7};
                resident[slot(1)] = {.clusterIndex = 2, .instanceIndex = 9};
                resident[slot(2)] = {.clusterIndex = 4};
                for (uint32_t g = 0; g < 3; ++g) {
                    auto& group = groups[groupSlot(g)];
                    group.pageIndex = g == 2 ? 101 : 100;
                    group.clusterCount = 4; group.lodLevel = g == 2 ? 5 : 3;
                    group.gpuSceneInstanceIndex = g + 10;
                    group.pageDeviceOffsetBytes = (permutation * 8 + g) * 4096;
                }
                const auto record = [&](uint32_t cluster, uint32_t g) {
                    return VisibleClusterRecord{.clusterIndex = cluster, .instanceIndex = g + 10,
                        .dataIndex = groupSlot(g), .flags = visibleClusterFlags(VisibleClusterSource::StreamPage)};
                };
                stream[slot(0)] = record(2, 0); stream[slot(1)] = record(2, 1);
                stream[slot(2)] = record(2, 2); stream[slot(3)] = record(3, 0);
                stream[slot(4)] = record(0, 0); stream[slot(4)].flags = 0; // Wrong producer.
                stream[slot(5)] = record(0, 0); stream[slot(5)].dataIndex = 99;
                const auto id = [](uint32_t r, uint32_t t = 0) { return ((r + 1) << 7) | t; };
                const std::array<uint32_t, width> visibility{ id(slot(0)), id(slot(1)), id(slot(2), 1),
                    id(base + slot(0)), id(base + slot(1)), id(base + slot(0), 1),
                    id(base + slot(2)), id(base + slot(3)), 0, id(base + slot(4)), id(base + slot(5)), id(base + capacity) };
                builtin_pass::GPUDrivenPreviewGpuParams params;
                params.width = width; params.height = 1; params.mode = mode; params.clearColor[3] = 1;
                DEBUG_REQUIRE(upload(*buffers[Params], &params, sizeof(params)));
                DEBUG_REQUIRE(upload(*buffers[Resident], resident.data(), sizeof(resident)));
                DEBUG_REQUIRE(upload(*buffers[Stream], stream.data(), sizeof(stream)));
                DEBUG_REQUIRE(upload(*buffers[Groups], groups.data(), sizeof(groups)));
                DEBUG_REQUIRE(upload(*buffers[Clusters], meshlets.data(), sizeof(meshlets)));
                DEBUG_REQUIRE(upload(*uploads[0], visibility.data(), sizeof(visibility)));
                if (submitted) { DEBUG_REQUIRE(fence->reset()); DEBUG_REQUIRE(pool->reset()); }
                DEBUG_REQUIRE(commands->begin());
                for (uint32_t i = 0; i < 2; ++i) {
                    TextureBarrierDesc barrier{.texture = textures[i].get(),
                        .before = submitted ? ResourceState::ShaderRead : ResourceState::Undefined, .after = ResourceState::TransferDestination};
                    commands->barrier({.textures = &barrier, .textureCount = 1});
                    commands->copyBufferToTexture({.buffer = uploads[i].get(), .texture = textures[i].get(), .width = width, .height = 1});
                    barrier.before = ResourceState::TransferDestination; barrier.after = ResourceState::ShaderRead;
                    commands->barrier({.textures = &barrier, .textureCount = 1});
                }
                TextureBarrierDesc barrier{.texture = textures[2].get(),
                    .before = submitted ? ResourceState::TransferSource : ResourceState::Undefined, .after = ResourceState::ColorAttachment};
                commands->barrier({.textures = &barrier, .textureCount = 1});
                const RenderingAttachmentDesc color{.view = views[2].get(), .state = ResourceState::ColorAttachment,
                    .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
                commands->beginRendering({.renderArea = {.width = width, .height = 1}, .colorAttachments = &color, .colorAttachmentCount = 1});
                commands->setViewport({.width = float(width), .height = 1.f, .maxDepth = 1.f});
                commands->setScissor({.width = width, .height = 1});
                commands->bindBindlessHeap(*heap); commands->bindGraphicsPipeline(*pipeline);
                const VisibilityBufferCompositeUserPush push{.paramsBuffer = handles[Params].index, .visibilityImage = images[0].index,
                    .depthImage = images[1].index, .residentRecords = handles[Resident].index, .meshletBuffer = handles[Clusters].index,
                    .residentRecordCapacity = base, .streamRecords = handles[Stream].index, .streamGroups = handles[Groups].index};
                commands->pushBindlessData(&push, sizeof(push)); commands->draw(3); commands->endRendering();
                barrier.before = ResourceState::ColorAttachment; barrier.after = ResourceState::TransferSource;
                commands->barrier({.textures = &barrier, .textureCount = 1});
                commands->copyTextureToBuffer({.texture = textures[2].get(), .buffer = readback.get(), .width = width, .height = 1});
                DEBUG_REQUIRE(commands->end());
                CommandBuffer* list[] = {commands.get()};
                DEBUG_REQUIRE(queue->submit({.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()}));
                DEBUG_REQUIRE(fence->wait()); submitted = true;
                readback->invalidate(); const void* mapped = readback->map();
                if (!mapped) { return RhiTestResult::fail("Cannot read debug attachment"); }
                std::array<uint32_t, width> actual;
                std::memcpy(actual.data(), mapped, sizeof(actual)); readback->unmap();
                if (permutation == 0) { reference = actual; }
                else if (actual != reference) { return RhiTestResult::fail("Debug color changed after record/group/page relocation, mode=" + std::to_string(mode)); }
                if (mode <= 3) {
                    if (actual[0] != actual[1] || actual[3] != actual[4] || actual[0] == actual[8] || actual[3] == actual[8] ||
                        actual[9] != actual[8] || actual[10] != actual[8] || actual[11] != actual[8]) {
                        return RhiTestResult::fail("Invalid identity coverage or instance-dependent color");
                    }
                    if (mode == 1 && (actual[3] != actual[5] || actual[3] == actual[6] || actual[3] == actual[7])) {
                        return RhiTestResult::fail("Meshlet color does not distinguish page/local cluster independently of triangle");
                    }
                    if (mode == 2 && (actual[0] != actual[3] || actual[3] != actual[5] || actual[3] != actual[7] || actual[3] == actual[6])) {
                        return RhiTestResult::fail("Stream LOD colors do not use actual LOD shared with resident geometry");
                    }
                    if (mode == 3 && actual[3] == actual[5]) { return RhiTestResult::fail("Triangle ID not distinguished"); }
                }
            }
        }
        return RhiTestResult::pass("Six actual composite modes stable across record/group/base/page relocation; meshlet, triangle and true stream LOD identities verified");
    }
};
METALLIC_REGISTER_RHI_TEST(VisibilityDebugStabilityTest);
#undef DEBUG_REQUIRE
} // namespace
} // namespace metallic::tests
