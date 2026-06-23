#include "RhiTest.h"

#include "Runtime/Render/MeshletStreamResidency.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::tests {
namespace {

RhiTestResult createCommandResources(
    render::Device& device,
    render::Queue& queue,
    std::unique_ptr<render::CommandPool>& outCommandPool,
    std::unique_ptr<render::CommandBuffer>& outCommandBuffer,
    std::unique_ptr<render::Fence>& outFence)
{
    render::Result result = device.createCommandPool(queue, outCommandPool);
    if (!result || outCommandPool == nullptr) {
        return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
    }

    result = outCommandPool->createCommandBuffer(outCommandBuffer);
    if (!result || outCommandBuffer == nullptr) {
        return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
    }

    result = device.createFence(false, outFence);
    if (!result || outFence == nullptr) {
        return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
    }

    return RhiTestResult::pass();
}

RhiTestResult submitAndWait(
    render::Queue& queue,
    render::CommandBuffer& commandBuffer,
    render::Fence& fence)
{
    render::CommandBuffer* commandBuffers[] = {&commandBuffer};
    render::Result result = queue.submit(render::QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = &fence,
    });
    if (!result) {
        return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
    }

    result = fence.wait(5'000'000'000ull);
    if (!result) {
        return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
    }
    return RhiTestResult::pass();
}

bool readBufferBytes(render::Buffer& buffer, void* outData, uint64_t byteSize)
{
    buffer.invalidate(0, byteSize);
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return false;
    }
    std::memcpy(outData, mapped, static_cast<size_t>(byteSize));
    buffer.unmap();
    return true;
}

render::StreamerDesc makeTestStreamerDesc(uint64_t dynamicSizePerFrame = 1024)
{
    render::StreamerDesc desc;
    desc.constantBufferSize = 4096;
    desc.dynamicBufferSizePerFrame = dynamicSizePerFrame;
    desc.queuedFrameCount = 2;
    desc.dynamicBufferDesc.usage = render::BufferUsageBits::TransferSource;
    return desc;
}

RhiTestResult buildBunnyStreamAssetForTest(
    const std::filesystem::path& outputPath,
    scene::MeshletStreamAsset& outAsset)
{
    const std::filesystem::path sourcePath =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";

    scene::Scene scene;
    if (!scene.load(sourcePath)) {
        return RhiTestResult::fail("Scene::load failed: " + scene.lastLoadResult().error);
    }

    std::string reason;
    if (!scene::buildMeshletStreamAsset(
            scene::MeshletStreamAssetBuildDesc{
                .scene = &scene,
                .sourcePath = sourcePath,
                .outputPath = outputPath,
            },
            reason)) {
        return RhiTestResult::fail("buildMeshletStreamAsset failed: " + reason);
    }

    if (!outAsset.open(outputPath, reason)) {
        return RhiTestResult::fail("MeshletStreamAsset::open failed: " + reason);
    }
    if (outAsset.pageCount() == 0) {
        return RhiTestResult::fail("streamasset has no pages");
    }
    return RhiTestResult::pass();
}

std::vector<uint32_t> fallbackPagesFor(const scene::MeshletStreamAsset& asset)
{
    std::vector<uint32_t> fallbackPages;
    for (const scene::MeshletStreamPrimitiveInfo& primitive : asset.primitives()) {
        for (uint32_t page = 0; page < primitive.fallbackPageCount; ++page) {
            fallbackPages.push_back(primitive.fallbackPageOffset + page);
        }
    }
    return fallbackPages;
}

std::vector<uint32_t> nonFallbackPagesFor(
    const scene::MeshletStreamAsset& asset,
    std::span<const uint32_t> fallbackPages)
{
    std::vector<uint8_t> isFallback(asset.pageCount(), 0);
    for (uint32_t page : fallbackPages) {
        if (page < isFallback.size()) {
            isFallback[page] = 1;
        }
    }

    std::vector<uint32_t> pages;
    for (uint32_t page = 0; page < asset.pageCount(); ++page) {
        if (isFallback[page] == 0) {
            pages.push_back(page);
        }
    }
    return pages;
}

class StreamerBufferUploadTest : public RhiTest {
public:
    StreamerBufferUploadTest()
    {
        type = RhiTestType::Command;
        name = "streamer_buffer_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr std::array<uint32_t, 4> kExpected{
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };
        constexpr uint64_t kByteSize = kExpected.size() * sizeof(uint32_t);

        std::unique_ptr<render::Streamer> streamer;
        render::Result result = context.device.createStreamer(makeTestStreamerDesc(), streamer);
        if (!result || streamer == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = kByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        const render::StreamDataChunk chunks[] = {
            render::StreamDataChunk{
                .data = kExpected.data(),
                .size = 2 * sizeof(uint32_t),
            },
            render::StreamDataChunk{
                .data = kExpected.data() + 2,
                .size = 2 * sizeof(uint32_t),
            },
        };
        render::BufferOffset streamed = streamer->streamBufferData(render::StreamBufferDataDesc{
            .dataChunks = chunks,
            .dataChunkCount = static_cast<uint32_t>(std::size(chunks)),
            .placementAlignment = 4,
            .dstBuffer = readbackBuffer.get(),
            .dstOffset = 0,
        });
        if (!streamed.valid()) {
            return RhiTestResult::fail("streamBufferData returned an invalid source");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::BufferBarrierDesc toTransfer{
            .buffer = readbackBuffer.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::TransferDestination,
            .offset = 0,
            .size = kByteSize,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = &toTransfer,
            .bufferCount = 1,
        });
        commandBuffer->copyStreamedData(*streamer);
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }

        std::array<uint32_t, 4> actual{};
        if (!readBufferBytes(*readbackBuffer, actual.data(), kByteSize)) {
            return RhiTestResult::fail("readback buffer did not map");
        }
        if (actual != kExpected) {
            return RhiTestResult::fail("streamed buffer bytes did not match expected pattern");
        }
        return RhiTestResult::pass();
    }
};

class StreamerTextureUploadTest : public RhiTest {
public:
    StreamerTextureUploadTest()
    {
        type = RhiTestType::Command;
        name = "streamer_texture_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint32_t kWidth = 4;
        constexpr uint32_t kHeight = 4;
        constexpr uint32_t kTightRowPitch = kWidth * 4;
        constexpr uint32_t kSourceRowPitch = 32;
        constexpr uint64_t kPixelByteSize = kWidth * kHeight * 4ull;

        std::array<uint8_t, kSourceRowPitch * kHeight> source{};
        std::array<uint8_t, kPixelByteSize> expected{};
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                const uint8_t r = static_cast<uint8_t>(x * 40 + 3);
                const uint8_t g = static_cast<uint8_t>(y * 35 + 7);
                const uint8_t b = static_cast<uint8_t>(x + y * 10 + 11);
                const uint8_t a = 255;
                const uint32_t sourceIndex = y * kSourceRowPitch + x * 4;
                const uint32_t expectedIndex = y * kTightRowPitch + x * 4;
                source[sourceIndex + 0] = r;
                source[sourceIndex + 1] = g;
                source[sourceIndex + 2] = b;
                source[sourceIndex + 3] = a;
                expected[expectedIndex + 0] = r;
                expected[expectedIndex + 1] = g;
                expected[expectedIndex + 2] = b;
                expected[expectedIndex + 3] = a;
            }
        }

        std::unique_ptr<render::Streamer> streamer;
        render::Result result = context.device.createStreamer(makeTestStreamerDesc(), streamer);
        if (!result || streamer == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Texture> texture;
        result = context.device.createTexture(
            render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::TransferDestination | render::TextureUsageBits::TransferSource,
                .format = render::Format::Rgba8Unorm,
                .width = kWidth,
                .height = kHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            },
            texture);
        if (!result || texture == nullptr) {
            return RhiTestResult::fail(std::string("createTexture returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = kPixelByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        render::BufferOffset streamed = streamer->streamTextureData(render::StreamTextureDataDesc{
            .data = source.data(),
            .dataRowPitch = kSourceRowPitch,
            .dataSlicePitch = kSourceRowPitch * kHeight,
            .dstTexture = texture.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
        });
        if (!streamed.valid()) {
            return RhiTestResult::fail("streamTextureData returned an invalid source");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::TextureBarrierDesc textureToTransfer{
            .texture = texture.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::TransferDestination,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .textures = &textureToTransfer,
            .textureCount = 1,
        });
        commandBuffer->copyStreamedData(*streamer);
        render::TextureBarrierDesc textureToSource{
            .texture = texture.get(),
            .before = render::ResourceState::TransferDestination,
            .after = render::ResourceState::TransferSource,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .textures = &textureToSource,
            .textureCount = 1,
        });
        commandBuffer->copyTextureToBuffer(render::TextureBufferCopyDesc{
            .texture = texture.get(),
            .buffer = readbackBuffer.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
        });
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }

        std::array<uint8_t, kPixelByteSize> actual{};
        if (!readBufferBytes(*readbackBuffer, actual.data(), actual.size())) {
            return RhiTestResult::fail("texture readback buffer did not map");
        }
        if (actual != expected) {
            return RhiTestResult::fail("streamed texture pixels did not match expected pattern");
        }
        return RhiTestResult::pass();
    }
};

class StreamerConstantUploadTest : public RhiTest {
public:
    StreamerConstantUploadTest()
    {
        type = RhiTestType::Resource;
        name = "streamer_constant_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr std::array<uint32_t, 4> kFirst{
            0x01020304u,
            0x11121314u,
            0x21222324u,
            0x31323334u,
        };
        constexpr std::array<uint32_t, 2> kSecond{
            0xA0A1A2A3u,
            0xB0B1B2B3u,
        };

        std::unique_ptr<render::Streamer> streamer;
        render::StreamerDesc desc = makeTestStreamerDesc();
        desc.constantBufferSize = 4096;
        render::Result result = context.device.createStreamer(desc, streamer);
        if (!result || streamer == nullptr || streamer->constantBuffer() == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        const uint64_t firstOffset = streamer->streamConstantData(
            kFirst.data(),
            kFirst.size() * sizeof(uint32_t));
        const uint64_t secondOffset = streamer->streamConstantData(
            kSecond.data(),
            kSecond.size() * sizeof(uint32_t));
        if (firstOffset == std::numeric_limits<uint64_t>::max() ||
            secondOffset == std::numeric_limits<uint64_t>::max()) {
            return RhiTestResult::fail("streamConstantData returned an invalid offset");
        }
        if (firstOffset != 0) {
            return RhiTestResult::fail("first constant upload did not start at offset zero");
        }

        const uint64_t alignment = std::max<uint64_t>(
            context.device.capabilities().constantBufferOffsetAlignment,
            1);
        if (secondOffset % alignment != 0 ||
            secondOffset < kFirst.size() * sizeof(uint32_t)) {
            return RhiTestResult::fail("second constant upload was not aligned after first upload");
        }

        render::Buffer* constantBuffer = streamer->constantBuffer();
        constantBuffer->invalidate(0, desc.constantBufferSize);
        void* mapped = constantBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("constant buffer did not map");
        }

        bool firstMatches = std::memcmp(
            static_cast<uint8_t*>(mapped) + firstOffset,
            kFirst.data(),
            kFirst.size() * sizeof(uint32_t)) == 0;
        bool secondMatches = std::memcmp(
            static_cast<uint8_t*>(mapped) + secondOffset,
            kSecond.data(),
            kSecond.size() * sizeof(uint32_t)) == 0;
        constantBuffer->unmap();
        if (!firstMatches || !secondMatches) {
            return RhiTestResult::fail("constant buffer contents did not match streamed data");
        }
        streamer->endFrame();
        return RhiTestResult::pass();
    }
};

class StreamerGraphUploadPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Streamed graph data")
            .buffer(kExpected.size() * sizeof(uint32_t), sizeof(uint32_t))
            .transferWrite();
        return reflection;
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        render::Streamer* streamer = context.streamer();
        render::BufferHandle output = context.outputBuffer("data");
        if (streamer == nullptr || !output.valid()) {
            return render::makeError(render::Error::InvalidArgument);
        }

        const render::StreamDataChunk chunk{
            .data = kExpected.data(),
            .size = kExpected.size() * sizeof(uint32_t),
        };
        render::BufferOffset streamed = streamer->streamBufferData(render::StreamBufferDataDesc{
            .dataChunks = &chunk,
            .dataChunkCount = 1,
            .placementAlignment = 4,
            .dstBuffer = output.buffer(),
            .dstOffset = 0,
        });
        return streamed.valid() ? render::Result{} : render::makeError(render::Error::Failure);
    }

    static constexpr std::array<uint32_t, 4> kExpected{
        0x11223344u,
        0xAABBCCDDu,
        0xDEADBEEFu,
        0xCAFEBABEu,
    };
};

void registerStreamerGraphPass()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;
    render::registerRenderGraphPassType(
        "StreamerGraphUploadPass",
        "Test-only pass that streams into a graph output buffer",
        []() { return std::make_unique<StreamerGraphUploadPass>(); });
}

class StreamerRenderGraphFlushTest : public RhiTest {
public:
    StreamerRenderGraphFlushTest()
    {
        type = RhiTestType::Command;
        name = "streamer_render_graph_flush";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerStreamerGraphPass();

        render::RenderGraph graph;
        graph.setName("StreamerGraph");
        graph.addNode("StreamerGraphUploadPass", "Upload");
        graph.markOutput("Upload.data");

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result result = executor.compile(context.device, graph, 1, 1, log);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        result = executor.execute(*commandBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        if (!submit.passed) {
            return submit;
        }

        render::RenderGraphResource* output = executor.outputResource("Upload.data");
        if (output == nullptr || output->buffer == nullptr) {
            return RhiTestResult::fail("streamer graph output resource is missing");
        }

        std::array<uint32_t, 4> actual{};
        if (!readBufferBytes(
                *output->buffer,
                actual.data(),
                actual.size() * sizeof(uint32_t))) {
            return RhiTestResult::fail("streamer graph output did not map");
        }
        if (actual != StreamerGraphUploadPass::kExpected) {
            return RhiTestResult::fail("streamer graph output bytes did not match expected pattern");
        }
        return RhiTestResult::pass();
    }
};

class StreamerMeshletResidencyUploadTest : public RhiTest {
public:
    StreamerMeshletResidencyUploadTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_residency_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        const std::filesystem::path streamAssetPath = context.outputDirectory / "streamer_residency.meshstream.bin";
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(streamAssetPath, asset);
        if (!build.passed) {
            return build;
        }

        std::string reason;
        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        const uint32_t maxResidentPages = std::max<uint32_t>(
            static_cast<uint32_t>(fallbackPages.size()) + 2u,
            4u);

        render::MeshletStreamResidencyManager residency;
        if (!residency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentPages = maxResidentPages,
                    .queuedFrameCount = 2,
                },
                reason)) {
            return RhiTestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RhiTestResult::fail("lockFallbackPages failed: " + reason);
        }

        std::unique_ptr<render::Streamer> streamer;
        render::Result result = context.device.createStreamer(makeTestStreamerDesc(), streamer);
        if (!result || streamer == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = residency.pageBufferSize(),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            pageBuffer);
        if (!result || pageBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(pageBuffer) returned ") + toString(result));
        }

        residency.beginFrame();
        if (fallbackPages.empty() || residency.queuedUploadCount() == 0) {
            return RhiTestResult::fail("lockFallbackPages did not queue fallback uploads");
        }
        const uint32_t pageIndex = fallbackPages.front();
        std::vector<render::StreamPageTableEntry> initialTable(asset.pageCount());
        residency.buildInitialPageTable(initialTable);
        if (initialTable[pageIndex].slot != UINT32_MAX ||
            initialTable[pageIndex].state !=
                static_cast<uint32_t>(render::MeshletStreamPageResidencyState::Unloaded) ||
            initialTable[pageIndex].payloadBytes != asset.pages()[pageIndex].payloadSize) {
            return RhiTestResult::fail("initial stream page table entry did not encode missing fallback page");
        }
        residency.clearPendingPatches();

        const bool alreadyResident = residency.requestPage(pageIndex);
        if (alreadyResident || residency.queuedUploadCount() == 0) {
            return RhiTestResult::fail("fallback page was resident before upload");
        }
        const uint32_t uploaded = residency.processUploads(*streamer, *pageBuffer, 1);
        if (uploaded != 1) {
            return RhiTestResult::fail("processUploads did not schedule exactly one upload");
        }
        if (residency.pageState(pageIndex) != render::MeshletStreamPageResidencyState::PendingUpload) {
            return RhiTestResult::fail("uploaded page did not enter PendingUpload state");
        }
        std::span<const render::StreamPageTablePatch> patches = residency.pendingPatches();
        if (patches.size() != 1 ||
            patches[0].pageId != pageIndex ||
            patches[0].slot == UINT32_MAX ||
            patches[0].state != static_cast<uint32_t>(render::MeshletStreamPageResidencyState::PendingUpload)) {
            return RhiTestResult::fail("pending upload did not produce expected page table patch");
        }
        residency.clearPendingPatches();

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::BufferBarrierDesc toTransfer{
            .buffer = pageBuffer.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::TransferDestination,
            .offset = 0,
            .size = pageBuffer->desc().size,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = &toTransfer,
            .bufferCount = 1,
        });
        commandBuffer->copyStreamedData(*streamer);
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }

        residency.beginFrame();
        if (residency.pageResident(pageIndex)) {
            return RhiTestResult::fail("page became resident before queued frame delay elapsed");
        }
        if (!residency.pendingPatches().empty()) {
            return RhiTestResult::fail("residency produced a patch before pending upload completed");
        }
        residency.beginFrame();
        if (!residency.pageResident(pageIndex)) {
            return RhiTestResult::fail("page did not become resident after queued frame delay elapsed");
        }
        patches = residency.pendingPatches();
        if (patches.size() != 1 ||
            patches[0].pageId != pageIndex ||
            patches[0].slot == UINT32_MAX ||
            patches[0].state != static_cast<uint32_t>(render::MeshletStreamPageResidencyState::LockedFallback)) {
            return RhiTestResult::fail("resident fallback did not produce expected page table patch");
        }

        const uint32_t slot = residency.slotForPage(pageIndex);
        if (slot == UINT32_MAX) {
            return RhiTestResult::fail("resident page has no slot");
        }

        std::vector<uint8_t> actual(asset.pages()[pageIndex].payloadSize);
        pageBuffer->invalidate(
            static_cast<uint64_t>(slot) * residency.pageStride(),
            actual.size());
        void* mapped = pageBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("page buffer did not map");
        }
        std::memcpy(
            actual.data(),
            static_cast<uint8_t*>(mapped) + static_cast<uint64_t>(slot) * residency.pageStride(),
            actual.size());
        pageBuffer->unmap();

        const std::span<const uint8_t> expected = asset.pagePayload(pageIndex);
        if (actual.size() != expected.size() ||
            std::memcmp(actual.data(), expected.data(), expected.size()) != 0) {
            return RhiTestResult::fail("streamed page payload bytes did not match streamasset payload");
        }
        return RhiTestResult::pass();
    }
};

class StreamerMeshletResidencyGpuRequestPatchTest : public RhiTest {
public:
    StreamerMeshletResidencyGpuRequestPatchTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_residency_gpu_request_patches";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_gpu_request.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RhiTestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
        }

        render::MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentPages = static_cast<uint32_t>(fallbackPages.size()) + 1u,
                    .queuedFrameCount = 2,
                },
                reason)) {
            return RhiTestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RhiTestResult::fail("lockFallbackPages failed: " + reason);
        }
        residency.clearPendingPatches();

        const uint32_t firstPage = streamablePages[0];
        const uint32_t secondPage = streamablePages[1];
        const std::array<uint32_t, 4> gpuRequests = {
            firstPage,
            firstPage,
            secondPage,
            secondPage,
        };
        const uint32_t consumed = residency.consumeGpuRequests(gpuRequests);
        if (consumed != 2) {
            return RhiTestResult::fail("consumeGpuRequests did not deduplicate page ids");
        }
        if (residency.slotForPage(firstPage) != UINT32_MAX) {
            return RhiTestResult::fail("first requested page was not evicted under slot pressure");
        }
        if (residency.slotForPage(secondPage) == UINT32_MAX) {
            return RhiTestResult::fail("second requested page did not receive the single streamable slot");
        }

        const std::span<const render::StreamPageTablePatch> patches = residency.pendingPatches();
        if (patches.size() != 1 ||
            patches[0].pageId != firstPage ||
            patches[0].slot != UINT32_MAX ||
            patches[0].state != static_cast<uint32_t>(render::MeshletStreamPageResidencyState::Unloaded)) {
            return RhiTestResult::fail("evicted request did not produce expected invalid page table patch");
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(StreamerBufferUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerTextureUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerConstantUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerRenderGraphFlushTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyGpuRequestPatchTest);

} // namespace
} // namespace metallic::tests
