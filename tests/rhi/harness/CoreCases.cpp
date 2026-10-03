#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Fixtures.h"
#include "TraceRecorder.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ComputeKernel.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>

namespace metallic::tests {
namespace {
using namespace render;
#define CASE_REQUIRE(expression) do { const auto& checked = (expression); if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + resultToString(checked)); } } while (false)

bool writeBuffer(Buffer& buffer, std::span<const std::byte> bytes)
{
    auto* mapped = buffer.map();
    if (!mapped) { return false; }
    std::memcpy(mapped, bytes.data(), bytes.size());
    buffer.flush(); buffer.unmap();
    return true;
}

RHITestResult compare(RHITestContext& context, Buffer& buffer, std::span<const std::byte> expected, const std::string& prefix)
{
    const auto* mapped = static_cast<const std::byte*>(buffer.map());
    if (!mapped) { return RHITestResult::fail("readback mapping failed"); }
    buffer.invalidate();
    std::vector<std::byte> actual(mapped, mapped + expected.size());
    buffer.unmap();
    const auto difference = std::mismatch(actual.begin(), actual.end(), expected.begin());
    const bool equal = difference.first == actual.end();
    if (context.evidence) {
        std::string artifact = prefix;
        for (uint32_t index = 1; std::filesystem::exists(context.evidence->root() / (artifact + "-actual.bin")); ++index) {
            artifact = prefix + "-" + std::to_string(index);
        }
        context.evidence->bytes(artifact + "-expected.bin", expected);
        context.evidence->bytes(artifact + "-actual.bin", actual);
        context.evidence->json(artifact + "-diff.json", {{"equal", equal}, {"bytes", actual.size()},
            {"firstMismatch", size_t(difference.first - actual.begin())}});
    }
    return equal ? RHITestResult::pass() : RHITestResult::fail(prefix + " readback mismatch");
}

class TextureSubresourceCopyTest final : public RHITest {
public:
    TextureSubresourceCopyTest() { type = RHITestType::Command; name = "texture_odd_mips_layers_volume_readback"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"texture.copy.offset.padding.readback", "texture.copy.mip.layer.readback",
            "texture.copy.volume.preserve.readback"}, bench::Layer::RHI, "core", "core",
            {"array-expected.bin", "array-actual.bin", "array-diff.json", "array-regions.json",
             "volume-expected.bin", "volume-actual.bin", "volume-diff.json", "volume-regions.json"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        for (bool volume : {false, true}) {
            const auto type = volume ? TextureType::Texture3D : TextureType::Texture2D;
            const uint32_t layers = volume ? 1 : 3, depth = volume ? 5 : 1;
            auto texture = context.device.createTexture({.type = type,
                .usage = TextureUsageBits::TransferSource | TextureUsageBits::TransferDestination,
                .format = Format::RGBA8Unorm, .width = 13, .height = 9, .depth = depth, .mipCount = 3, .layerCount = layers});
            CASE_REQUIRE(texture);
            struct Region { uint32_t mip, layer, width, height, depth, row, slice; uint64_t offset; };
            std::vector<Region> regions;
            uint64_t size = 64;
            for (uint32_t layer = 0; layer < layers; ++layer) {
                for (uint32_t mip = 0; mip < 3; ++mip) {
                    const uint32_t w = std::max(1u, 13u >> mip), h = std::max(1u, 9u >> mip), d = std::max(1u, depth >> mip);
                    const uint32_t row = (w + 3) * 4, slice = row * (h + 2);
                    regions.push_back({mip, layer, w, h, d, row, slice, size});
                    size += uint64_t(slice) * d + 64;
                }
            }
            std::vector<std::byte> source(size + 128, std::byte{0x33}), expected(size, std::byte{0xa7});
            const uint64_t seed = context.evidence ? bench::readJson(context.evidence->root() / "input.json").at("seed").get<uint64_t>() : 1;
            bench::Json description = bench::Json::array();
            for (const auto& r : regions) {
                description.push_back({{"mip", r.mip}, {"layer", r.layer}, {"width", r.width}, {"height", r.height},
                    {"depth", r.depth}, {"offset", r.offset}, {"rowPitch", r.row}, {"slicePitch", r.slice}});
                for (uint32_t z = 0; z < r.depth; ++z) {
                    for (uint32_t y = 0; y < r.height; ++y) {
                        for (uint32_t x = 0; x < r.width * 4; ++x) {
                            const auto i = r.offset + z * r.slice + y * r.row + x;
                            source[i] = std::byte((seed + 31 * r.layer + 11 * r.mip + 7 * z + 3 * y + x) & 255);
                            expected[i] = source[i];
                        }
                    }
                }
            }
            // Replace one interior rectangle of one mip/layer/z plane. Every
            // other texel and every host padding byte must retain its value.
            const auto& target = regions[volume ? 1 : 4];
            const uint32_t targetZ = volume ? 1 : 0;
            std::fill(source.begin() + size, source.end(), std::byte{0xde});
            for (uint32_t y = 0; y < 2; ++y) {
                for (uint32_t x = 0; x < 12; ++x) {
                    expected[target.offset + targetZ * target.slice + (y + 1) * target.row + 4 + x] = std::byte{0xde};
                }
            }
            auto upload = context.device.createBuffer({.size = source.size(), .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload});
            auto readback = context.device.createBuffer({.size = size, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback});
            CASE_REQUIRE(upload); CASE_REQUIRE(readback);
            std::vector<std::byte> sentinel(size, std::byte{0xa7});
            if (!writeBuffer(**upload, source) || !writeBuffer(**readback, sentinel)) { return RHITestResult::fail("upload mapping failed"); }
            bench::GPUCommands recording(context.graphicsQueue);
            CASE_REQUIRE(recording.initialize(context.device));
            auto& commands = *recording.commands;
            TextureBarrierDesc barrier{.texture = texture->get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .before = {},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}, .range = {0, 3, 0, layers}};
            CASE_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
            for (const auto& r : regions) {
                commands.copyBufferToTexture({.buffer = upload->get(), .texture = texture->get(), .bufferOffset = r.offset,
                    .bufferRowPitch = r.row, .bufferSlicePitch = r.slice, .width = r.width, .height = r.height,
                    .depth = r.depth, .mipLevel = r.mip, .baseLayer = r.layer});
            }
            barrier.oldLayout = TextureLayout::TransferDestination;
            barrier.before = barrier.after;
            CASE_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
            commands.copyBufferToTexture({.buffer = upload->get(), .texture = texture->get(), .bufferOffset = size,
                .textureOffsetX = 1, .textureOffsetY = 1, .textureOffsetZ = int32_t(targetZ),
                .width = 3, .height = 2, .depth = 1, .mipLevel = target.mip, .baseLayer = target.layer});
            barrier.newLayout = TextureLayout::TransferSource;
            barrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead};
            CASE_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
            for (const auto& r : regions) {
                commands.copyTextureToBuffer({.texture = texture->get(), .buffer = readback->get(), .bufferOffset = r.offset,
                    .bufferRowPitch = r.row, .bufferSlicePitch = r.slice, .width = r.width, .height = r.height,
                    .depth = r.depth, .mipLevel = r.mip, .baseLayer = r.layer});
            }
            CASE_REQUIRE(recording.submitAndWait());
            const std::string prefix = volume ? "volume" : "array";
            if (context.evidence) { context.evidence->json(prefix + "-regions.json", {{"seed", seed}, {"regions", description}}); }
            if (context.evidence) {
                bench::Json visuals = bench::Json::array();
                const auto path = context.evidence->root() / "visuals.json";
                if (std::filesystem::exists(path)) { visuals = bench::readJson(path); }
                for (const auto& region : regions) {
                    if (region.layer > 0 || region.mip > 0) { continue; }
                    visuals.push_back({{"format", "rgba8"}, {"label", prefix + " mip 0 / layer 0 / z 0"},
                        {"actual", prefix + "-actual.bin"}, {"expected", prefix + "-expected.bin"},
                        {"width", region.width}, {"height", region.height}, {"rowPitch", region.row}, {"offset", region.offset}});
                }
                context.evidence->json("visuals.json", visuals);
            }
            auto result = compare(context, **readback, expected, prefix);
            if (!result.passed) { return result; }
            if (context.trace) {
                const auto identity = context.trace->textureId(**texture);
                bool initial = false, toRead = false;
                const bool unified = context.device.capabilities().unifiedImageLayouts;
                const auto snapshot = context.trace->snapshot();
                for (const auto& event : snapshot.at("events")) {
                    if (event.at("kind") != "barrier") { continue; }
                    for (const auto& encoded : event.at("images")) {
                        if (encoded.at("resource").get<uint64_t>() != identity) { continue; }
                        const bench::Json range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 3, 0, layers};
                        if (encoded.at("range") != range ||
                            encoded.at("sourceFamily").get<uint32_t>() != VK_QUEUE_FAMILY_IGNORED ||
                            encoded.at("destinationFamily").get<uint32_t>() != VK_QUEUE_FAMILY_IGNORED) {
                            return RHITestResult::fail("encoded texture barrier changed mip/layer range or queue ownership");
                        }
                        initial |= encoded.at("oldLayout").get<int>() == VK_IMAGE_LAYOUT_UNDEFINED &&
                            encoded.at("newLayout").get<int>() == (unified ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) &&
                            encoded.at("after").at("access").get<uint64_t>() == VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        toRead |= encoded.at("newLayout").get<int>() == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                            encoded.at("before").at("access").get<uint64_t>() == VK_ACCESS_2_TRANSFER_WRITE_BIT &&
                            encoded.at("after").at("access").get<uint64_t>() == VK_ACCESS_2_TRANSFER_READ_BIT;
                    }
                    if (unified) {
                        for (const auto& requested : event.at("requested")) {
                            if (requested.value("resource", uint64_t(0)) != identity || requested.at("kind") != "image" ||
                                requested.at("newLayout").get<int>() != int(TextureLayout::TransferSource)) { continue; }
                            for (const auto& encoded : event.at("memory")) {
                                toRead |= (encoded.at("before").at("access").get<uint64_t>() & VK_ACCESS_2_TRANSFER_WRITE_BIT) &&
                                    (encoded.at("after").at("access").get<uint64_t>() & VK_ACCESS_2_TRANSFER_READ_BIT);
                            }
                        }
                    }
                }
                if (context.evidence) { context.evidence->json(prefix + "-trace-checks.json", {{"initialTransition", initial}, {"copyReadVisibility", toRead}}); }
                if (!initial || !toRead) { return RHITestResult::fail("encoded texture layout/visibility missing from trace"); }
            }
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(TextureSubresourceCopyTest);

class GraphicsExecutionSwitchTest final : public RHITest {
public:
    GraphicsExecutionSwitchTest() { type = RHITestType::Rendering; name = "graphics_pipeline_shader_object_aba_readback"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graphics.pipeline.shaderObject.aba.readback", "graphics.viewport.scissor.readback", "graphics.stage.contract"},
            bench::Layer::RHI, "core", "core", {"aba-expected.bin", "aba-actual.bin", "aba-diff.json"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        if (!hasError(context.device.createGraphicsPipeline({}), Error::InvalidArgument) ||
            !hasError(context.device.createGraphicsShaderObjectProgram({}), Error::InvalidArgument)) {
            return RHITestResult::fail("empty graphics stages did not return InvalidArgument");
        }
        std::array<std::unique_ptr<ShaderModule>, 3> modules;
        const char* entries[]{"vertexMain", "redMain", "greenMain"};
        for (size_t i = 0; i < modules.size(); ++i) {
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "TestbenchState", .entryPointName = entries[i],
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
            if (!shader) { return RHITestResult::fail(log); }
            auto module = context.device.createShaderModule({.spirv = shader->spirv});
            CASE_REQUIRE(module); modules[i] = std::move(*module);
        }
        auto pipeline = context.device.createGraphicsPipeline({.vertexShader = {modules[0].get()},
            .fragmentShader = {modules[1].get()}, .colorFormat = Format::RGBA8Unorm});
        auto shaders = context.device.createGraphicsShaderObjectProgram({.vertexShader = {modules[0].get()},
            .fragmentShader = {modules[2].get()}});
        CASE_REQUIRE(pipeline); CASE_REQUIRE(shaders);
        constexpr uint32_t width = 39, height = 11;
        auto texture = context.device.createTexture({.usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
            .format = Format::RGBA8Unorm, .width = width, .height = height});
        CASE_REQUIRE(texture);
        auto view = context.device.createTextureView(**texture, {});
        auto readback = context.device.createBuffer({.size = width * height * 4, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback});
        CASE_REQUIRE(view); CASE_REQUIRE(readback);
        bench::GPUCommands recording(context.graphicsQueue);
        CASE_REQUIRE(recording.initialize(context.device));
        auto& commands = *recording.commands;
        TextureBarrierDesc barrier{.texture = texture->get(), .oldLayout = TextureLayout::Undefined,
            .newLayout = TextureLayout::ColorAttachment, .before = {},
            .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorWrite}, .range = {0, 1, 0, 1}};
        CASE_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
        RenderingAttachmentDesc attachment{.view = view->get(), .layout = TextureLayout::ColorAttachment,
            .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearColor = {0, 0, 0, 1}};
        CASE_REQUIRE(commands.beginRendering({.renderArea = {0, 0, width, height}, .colorAttachments = {&attachment, 1}}));
        for (uint32_t i = 0; i < 3; ++i) {
            CASE_REQUIRE(commands.bindExecution(i == 1 ? (*shaders)->execution() : (*pipeline)->execution()));
            commands.setViewport({float(i * 13), 0, 13, float(height), 0, 1});
            commands.setScissor({int32_t(i * 13 + 1), 1, 11, height - 2});
            commands.draw(3);
        }
        commands.endRendering();
        barrier.oldLayout = TextureLayout::ColorAttachment; barrier.newLayout = TextureLayout::TransferSource;
        barrier.before = barrier.after; barrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead};
        CASE_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
        commands.copyTextureToBuffer({.texture = texture->get(), .buffer = readback->get(), .width = width, .height = height});
        CASE_REQUIRE(recording.submitAndWait());
        std::vector<std::byte> expected(width * height * 4);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                const auto offset = (y * width + x) * 4;
                expected[offset + 3] = std::byte{255};
                if (y > 0 && y < height - 1 && x % 13 > 0 && x % 13 < 12) {
                    expected[offset + (x / 13 == 1 ? 1 : 0)] = std::byte{255};
                }
            }
        }
        if (context.evidence) {
            context.evidence->json("visuals.json", bench::Json::array({{{"format", "rgba8"}, {"label", "Pipeline / shader object / pipeline"},
                {"actual", "aba-actual.bin"}, {"expected", "aba-expected.bin"}, {"width", width}, {"height", height}}}));
        }
        return compare(context, **readback, expected, "aba");
    }
};
METALLIC_REGISTER_RHI_TEST(GraphicsExecutionSwitchTest);

class CopyTimestampReuseTest final : public RHITest {
public:
    CopyTimestampReuseTest() { type = RHITestType::Command; name = "copy_timestamp_host_reset_reuse_readback"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async",
            .requirements = {.validation = bench::Validation::Synchronization,
                .capabilities = {bench::Capability::TimestampQueries, bench::Capability::IndependentCopy},
                .queues = {QueueType::Graphics, QueueType::Copy}, .timestampQueues = {QueueType::Copy}},
            .coverage = {"query.copyQueue.hostReset.readback", "query.partialReset.contract", "query.completedReuse.readback"},
            .artifacts = {"copy-queries.json", "query-copy-actual.bin", "query-copy-expected.bin", "query-copy-diff.json"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto* queue = context.device.getQueue(QueueType::Copy);
        if (!queue || !queue->timestampValidBits()) { return RHITestResult::skip("copy timestamps unavailable"); }
        auto queries = context.device.createTimestampQueryPool(*queue, {.queryCount = 2});
        auto upload = context.device.createBuffer({.size = 256, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload, .queueAccess = QueueAccessBits::Copy});
        auto output = context.device.createBuffer({.size = 256, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback, .queueAccess = QueueAccessBits::Copy});
        CASE_REQUIRE(queries); CASE_REQUIRE(upload); CASE_REQUIRE(output);
        bench::Json records = bench::Json::array();
        for (uint32_t iteration = 0; iteration < 6; ++iteration) {
            std::array<std::byte, 256> expected;
            for (uint32_t i = 0; i < expected.size(); ++i) { expected[i] = std::byte((iteration * 19 + i) & 255); }
            if (!writeBuffer(**upload, expected)) { return RHITestResult::fail("copy query upload failed"); }
            CASE_REQUIRE((*queries)->reset(0, 2));
            bench::GPUCommands recording(*queue);
            CASE_REQUIRE(recording.initialize(context.device));
            CASE_REQUIRE(recording.commands->writeTimestamp(**queries, 0, PipelineStageBits::TopOfPipe));
            auto from = (*upload)->slice(), to = (*output)->slice();
            CASE_REQUIRE(from); CASE_REQUIRE(to);
            CASE_REQUIRE(recording.commands->copyBuffer(*from, *to));
            CASE_REQUIRE(recording.commands->writeTimestamp(**queries, 1, PipelineStageBits::BottomOfPipe));
            CASE_REQUIRE(recording.submitAndWait());
            std::array<TimestampQueryResult, 2> values{};
            CASE_REQUIRE((*queries)->readResults(0, values));
            const auto duration = (*queries)->durationMilliseconds(values[0].value, values[1].value);
            if (!values[0].available || !values[1].available || !std::isfinite(duration) || duration < 0) {
                return RHITestResult::fail("completed copy timestamps unavailable/invalid");
            }
            if (!hasError((*queries)->reset(1, UINT32_MAX), Error::InvalidArgument) ||
                !hasError((*queries)->reset(2, 1), Error::InvalidArgument)) {
                return RHITestResult::fail("query reset range was not rejected");
            }
            std::array<TimestampQueryResult, 2> unchanged{};
            CASE_REQUIRE((*queries)->readResults(0, unchanged));
            if (!unchanged[0].available || !unchanged[1].available || unchanged[0].value != values[0].value || unchanged[1].value != values[1].value) {
                return RHITestResult::fail("rejected reset changed completed query state");
            }
            CASE_REQUIRE((*queries)->reset(0, 1));
            CASE_REQUIRE((*queries)->readResults(0, unchanged));
            records.push_back({{"iteration", iteration}, {"begin", values[0].value}, {"end", values[1].value},
                {"milliseconds", duration}, {"resetAvailable", unchanged[0].available}, {"retainedAvailable", unchanged[1].available}});
            if (context.evidence) { context.evidence->json("copy-queries.json", records); }
            if (unchanged[0].available || !unchanged[1].available || unchanged[1].value != values[1].value) {
                return RHITestResult::fail("partial reset changed the wrong query");
            }
            const auto result = compare(context, **output, expected, "query-copy");
            if (!result.passed) { return result; }
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(CopyTimestampReuseTest);

class NonuniformSampledBindingsTest final : public RHITest {
public:
    NonuniformSampledBindingsTest() { type = RHITestType::Rendering; name = "binding_nonuniform_images_samplers_reuse"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"binding.sampledImage.sampler.nonuniform.readback", "binding.completedReuse.readback"},
            bench::Layer::RHI, "binding", "binding", {"sample0-actual.bin", "sample0-expected.bin", "sample0-diff.json",
                "sample1-actual.bin", "sample1-expected.bin", "sample1-diff.json"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        struct Params { GPUBufferSpan images; ShaderSampler samplers[2]; ShaderBuffer output; };
        static_assert(sizeof(Params) == 24);
        constexpr uint64_t abi = 0x544253414d504c45ull;
        std::string log;
        auto shader = compileSlangShaderToSpirv({.moduleName = "TestbenchBindings", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log);
        if (!shader) { return RHITestResult::fail(log); }
        ComputeKernel kernel;
        CASE_REQUIRE(kernel.initialize(context.device, {.spirv = shader->spirv, .parameters = parameterAbi<Params>(abi)}, log));
        auto registry = metallic::render::ResourceRegistry::forDevice(context.device);
        CASE_REQUIRE(registry);
        std::array<std::unique_ptr<Texture>, 2> images;
        std::array<std::unique_ptr<TextureView>, 2> views;
        for (uint32_t i = 0; i < 2; ++i) {
            auto image = context.device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::RGBA8Unorm});
            CASE_REQUIRE(image); images[i] = std::move(*image);
            auto view = context.device.createTextureView(*images[i], {});
            CASE_REQUIRE(view); views[i] = std::move(*view);
        }
        auto output = context.device.createBuffer({.size = 33 * 16, .structureStride = 16, .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostReadback});
        CASE_REQUIRE(output);
        ParameterWriter writer(context.device, **registry);
        TextureView* pointers[]{views[0].get(), views[1].get()};
        const Params params{writer.sampledImages(pointers), {writer.sampler({.minFilter = SamplerFilter::Nearest,
            .magFilter = SamplerFilter::Nearest}), writer.sampler({})}, writer.buffer(output->get())};
        auto packet = writer.encode(params, abi);
        CASE_REQUIRE(packet);
        for (uint32_t iteration = 0; iteration < 2; ++iteration) {
            std::array<std::byte, 33 * 16> sentinel;
            sentinel.fill(std::byte{0xa7});
            if (!writeBuffer(**output, sentinel)) { return RHITestResult::fail("binding sentinel map failed"); }
            bench::GPUCommands recording(context.graphicsQueue);
            CASE_REQUIRE(recording.initialize(context.device));
            auto& commands = *recording.commands;
            for (uint32_t i = 0; i < 2; ++i) {
                TextureBarrierDesc barrier{.texture = images[i].get(),
                    .oldLayout = iteration ? TextureLayout::ShaderRead : TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination,
                    .before = iteration ? SyncScope{PipelineStageBits::ComputeShader, AccessBits::ShaderRead} : SyncScope{},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}, .range = {0, 1, 0, 1}};
                CASE_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
                commands.clearColorTexture(*images[i], TextureLayout::TransferDestination,
                    i == iteration ? ColorValue{1, 0, 0, 1} : ColorValue{0, 1, 0, 1});
                barrier.oldLayout = TextureLayout::TransferDestination; barrier.newLayout = TextureLayout::ShaderRead;
                barrier.before = barrier.after; barrier.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
                CASE_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
            }
            CASE_REQUIRE(kernel.dispatch(commands, *packet, 1));
            CASE_REQUIRE(recording.submitAndWait());
            std::array<uint32_t, 33 * 4> expected;
            expected.fill(0xa7a7a7a7u);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                expected[lane * 4] = (lane & 1) == iteration ? 255 : 0;
                expected[lane * 4 + 1] = (lane & 1) == iteration ? 0 : 255;
                expected[lane * 4 + 2] = 0; expected[lane * 4 + 3] = 255;
            }
            const auto result = compare(context, **output, std::as_bytes(std::span(expected)), "sample" + std::to_string(iteration));
            if (!result.passed) { return result; }
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(NonuniformSampledBindingsTest);
#undef CASE_REQUIRE
} // namespace
} // namespace metallic::tests
