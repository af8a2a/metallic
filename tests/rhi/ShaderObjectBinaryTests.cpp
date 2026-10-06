#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <string_view>

namespace metallic::tests {
namespace {
using namespace render;

constexpr uint32_t kWidth = 32;
constexpr uint32_t kHeight = 24;
using ShaderCodes = std::array<ShaderCompileResult, 3>;
using ShaderModules = std::array<std::unique_ptr<ShaderModule>, 3>;

bool compileShaderCodes(ShaderCodes& output, std::string& log)
{
    const char* entries[]{"vertexMain", "redMain", "greenMain"};
    for (size_t i = 0; i < output.size(); ++i) {
        auto code = ShaderRegistry::instance().getShader({.moduleName = "TestbenchState",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
        if (!code) { return false; }
        output[i] = std::move(*code);
    }
    return true;
}

Result<ShaderModules> createShaderModules(Device& device, const ShaderCodes& code)
{
    ShaderModules modules;
    for (size_t i = 0; i < modules.size(); ++i) {
        auto module = ShaderRegistry::instance().getShaderModule(device, {.spirv = code[i].spirv});
        if (!module) { return std::unexpected(module.error()); }
        modules[i] = std::move(*module);
    }
    return modules;
}

Result<std::vector<std::byte>> renderReadback(Device& device, Queue& queue,
    const GraphicsShaderObjectProgram& program)
{
#define BINARY_RENDER_REQUIRE(expression) do { const auto& checked = (expression); \
    if (!checked) { return std::unexpected(checked.error()); } } while (false)
    auto texture = device.createTexture({.usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
        .format = Format::RGBA8Unorm, .width = kWidth, .height = kHeight});
    BINARY_RENDER_REQUIRE(texture);
    auto view = device.createTextureView(**texture, {});
    auto readback = device.createBuffer({.size = kWidth * kHeight * 4, .usage = BufferUsageBits::TransferDestination,
        .memoryLocation = MemoryLocation::HostReadback});
    BINARY_RENDER_REQUIRE(view);
    BINARY_RENDER_REQUIRE(readback);
    bench::GPUCommands recording(queue);
    BINARY_RENDER_REQUIRE(recording.initialize(device));
    auto& commands = *recording.commands;
    TextureBarrierDesc barrier{.texture = texture->get(), .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::ColorAttachment, .before = {},
        .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorWrite}, .range = {0, 1, 0, 1}};
    BINARY_RENDER_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
    RenderingAttachmentDesc attachment{.view = view->get(), .layout = TextureLayout::ColorAttachment,
        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearColor = {0, 0, 0, 1}};
    BINARY_RENDER_REQUIRE(commands.beginRendering({.renderArea = {0, 0, kWidth, kHeight},
        .colorAttachments = {&attachment, 1}}));
    BINARY_RENDER_REQUIRE(commands.bindExecution(program.execution()));
    BINARY_RENDER_REQUIRE(commands.setViewport({0, 0, float(kWidth), float(kHeight), 0, 1}));
    commands.setScissor({0, 0, kWidth, kHeight});
    BINARY_RENDER_REQUIRE(commands.draw(3));
    commands.endRendering();
    barrier.oldLayout = TextureLayout::ColorAttachment;
    barrier.newLayout = TextureLayout::TransferSource;
    barrier.before = barrier.after;
    barrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead};
    BINARY_RENDER_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
    auto slice = (*readback)->slice();
    BINARY_RENDER_REQUIRE(slice);
    BINARY_RENDER_REQUIRE(commands.copyTextureToBuffer({.texture = texture->get(), .buffer = *slice,
        .width = kWidth, .height = kHeight}));
    BINARY_RENDER_REQUIRE(recording.submitAndWait());
    const auto* mapped = static_cast<const std::byte*>((*readback)->map());
    if (!mapped) { return std::unexpected(Error::Failure); }
    (*readback)->invalidate();
    std::vector<std::byte> bytes(mapped, mapped + kWidth * kHeight * 4);
    (*readback)->unmap();
    return bytes;
#undef BINARY_RENDER_REQUIRE
}

bool matchesSolidColor(std::span<const std::byte> bytes, size_t channel)
{
    if (bytes.size() != kWidth * kHeight * 4) { return false; }
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (bytes[i] != std::byte{uint8_t(i % 4 == channel || i % 4 == 3 ? 255 : 0)}) { return false; }
    }
    return true;
}

uint64_t readbackChecksum(std::span<const std::byte> bytes)
{
    uint64_t checksum = 14695981039346656037ull;
    for (const auto byte : bytes) {
        checksum = (checksum ^ std::to_integer<uint8_t>(byte)) * 1099511628211ull;
    }
    return checksum;
}

void logCreation(const char* phase, const GraphicsShaderObjectProgram& program,
    std::span<const std::byte> bytes)
{
    const auto stats = program.cacheStats();
    std::cout << "[ShaderObjectBinaryTest] phase=" << phase
        << " binaryCacheHit=" << int(stats.binaryCacheHit) << " persisted=" << int(stats.persisted)
        << " loadStatus=" << int(stats.loadStatus) << " driverRejected=" << int(stats.driverRejected)
        << " programHash=" << stats.programHash << " binaryDataSize=" << stats.binaryDataSize
        << " creationTimeNanoseconds=" << stats.creationTimeNanoseconds
        << " readbackChecksum=" << readbackChecksum(bytes) << '\n';
}

bench::Metadata binaryMetadata(std::vector<std::string> coverage)
{
    auto result = bench::gpuMetadata(std::move(coverage), bench::Layer::RHI, "core", "core");
    result.requirements.capabilities.push_back(bench::Capability::ShaderObject);
    return result;
}

class ShaderObjectBinaryPersistenceTest final : public RHITest {
public:
    ShaderObjectBinaryPersistenceTest() { type = RHITestType::Rendering; name = "shader_object_binary_persistence_readback"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return binaryMetadata({"graphics.shaderObject.binary.persistence.readback",
            "graphics.shaderObject.binary.invalidation", "graphics.shaderObject.binary.deviceLifetime"});
    }
    RHITestResult run(RHITestContext& context) override
    {
#define BINARY_TEST_REQUIRE(expression) do { const auto& checked = (expression); \
    if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + resultToString(checked)); } } while (false)
        if (!context.device.capabilities().shaderObject) { return RHITestResult::skip("VK_EXT_shader_object is unavailable"); }
        const auto root = std::filesystem::absolute(context.outputDirectory / "shader-object-binary" /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const auto cacheDirectory = root.string();
        ShaderCodes code;
        std::string log;
        if (!compileShaderCodes(code, log)) { return RHITestResult::fail("shader object source compilation failed: " + log); }
        auto modules = createShaderModules(context.device, code);
        BINARY_TEST_REQUIRE(modules);
        auto& registry = ShaderRegistry::instance();
        const GraphicsShaderObjectProgramDesc desc{.vertexShader = {(*modules)[0].get()},
            .fragmentShader = {(*modules)[1].get()}, .binaryCacheDirectory = cacheDirectory.c_str()};
        auto cold = registry.getGraphicsShaderObjectProgram(context.device, desc);
        BINARY_TEST_REQUIRE(cold);
        const auto coldStats = (*cold)->cacheStats();
        if (coldStats.loadStatus != PipelineCacheLoadStatus::NotFound || coldStats.binaryCacheHit ||
            !coldStats.persisted || coldStats.binaryDataSize == 0) {
            return RHITestResult::fail("cold SPIR-V shader object creation did not persist a binary pair");
        }
        const std::filesystem::path cacheFile((*cold)->binaryCacheFilePath());
        if (!std::filesystem::is_regular_file(cacheFile)) { return RHITestResult::fail("shader object binary file is missing"); }
        auto coldPixels = renderReadback(context.device, context.graphicsQueue, **cold);
        BINARY_TEST_REQUIRE(coldPixels);
        if (!matchesSolidColor(*coldPixels, 0)) { return RHITestResult::fail("cold shader object red readback mismatch"); }
        logCreation("cold", **cold, *coldPixels);
        auto hot = registry.getGraphicsShaderObjectProgram(context.device, desc);
        BINARY_TEST_REQUIRE(hot);
        auto hotPixels = renderReadback(context.device, context.graphicsQueue, **hot);
        BINARY_TEST_REQUIRE(hotPixels);
        if (!(*hot)->cacheStats().binaryCacheHit || (*hot)->cacheStats().loadStatus != PipelineCacheLoadStatus::Loaded ||
            (*hot)->cacheStats().programHash != coldStats.programHash || *hotPixels != *coldPixels) {
            return RHITestResult::fail("native BINARY creation was not reused or changed its pixels");
        }
        logCreation("hot", **hot, *hotPixels);
        auto variantDesc = desc;
        variantDesc.fragmentShader.module = (*modules)[2].get();
        auto variant = registry.getGraphicsShaderObjectProgram(context.device, variantDesc);
        BINARY_TEST_REQUIRE(variant);
        auto variantPixels = renderReadback(context.device, context.graphicsQueue, **variant);
        BINARY_TEST_REQUIRE(variantPixels);
        if ((*variant)->cacheStats().binaryCacheHit || (*variant)->cacheStats().programHash == coldStats.programHash ||
            !(*variant)->cacheStats().persisted || !matchesSolidColor(*variantPixels, 1) || *variantPixels == *coldPixels) {
            return RHITestResult::fail("fragment variant reused the wrong binary identity or pixels");
        }
        logCreation("fragment-variant", **variant, *variantPixels);
        const std::string concurrentDirectory = (root / "concurrent").string();
        auto concurrentDesc = desc;
        concurrentDesc.binaryCacheDirectory = concurrentDirectory.c_str();
        std::array<std::future<Result<std::unique_ptr<GraphicsShaderObjectProgram>>>, 4> jobs;
        for (auto& job : jobs) {
            job = std::async(std::launch::async, [&] { return registry.getGraphicsShaderObjectProgram(context.device, concurrentDesc); });
        }
        uint32_t coldCreates = 0;
        for (auto& job : jobs) {
            auto concurrent = job.get();
            BINARY_TEST_REQUIRE(concurrent);
            if (!(*concurrent)->cacheStats().binaryCacheHit) { ++coldCreates; }
            auto concurrentPixels = renderReadback(context.device, context.graphicsQueue, **concurrent);
            BINARY_TEST_REQUIRE(concurrentPixels);
            if (!(*concurrent)->cacheStats().persisted || *concurrentPixels != *coldPixels) {
                return RHITestResult::fail("concurrent shader binary requests changed output or did not persist");
            }
        }
        if (coldCreates != 1) { return RHITestResult::fail("parallel requests redundantly compiled the same uncached binary pair"); }

        // An unavailable cache directory must not prevent a valid shader from rendering.
        const std::string blockedDirectory = (root / "blocked").string();
        { std::ofstream blocker(blockedDirectory); blocker << "cache directory is a regular file"; }
        auto blockedDesc = desc;
        blockedDesc.binaryCacheDirectory = blockedDirectory.c_str();
        auto unsaved = registry.getGraphicsShaderObjectProgram(context.device, blockedDesc);
        BINARY_TEST_REQUIRE(unsaved);
        auto unsavedPixels = renderReadback(context.device, context.graphicsQueue, **unsaved);
        BINARY_TEST_REQUIRE(unsavedPixels);
        if ((*unsaved)->cacheStats().persisted || (*unsaved)->cacheStats().binaryCacheHit || *unsavedPixels != *coldPixels) {
            return RHITestResult::fail("cache I/O failure prevented correct shader object rendering");
        }
        for (const bool truncate : {true, false}) {
            if (truncate) {
                std::ofstream damaged(cacheFile, std::ios::binary | std::ios::trunc);
                damaged << "broken";
                if (!damaged) { return RHITestResult::fail("could not truncate the binary fixture"); }
            } else {
                std::fstream damaged(cacheFile, std::ios::binary | std::ios::in | std::ios::out);
                damaged.seekg(-1, std::ios::end);
                char last = 0;
                damaged.read(&last, 1);
                last ^= 0x5a;
                damaged.seekp(-1, std::ios::end);
                damaged.write(&last, 1);
                if (!damaged) { return RHITestResult::fail("could not corrupt the binary fixture"); }
            }
            auto fallback = registry.getGraphicsShaderObjectProgram(context.device, desc);
            BINARY_TEST_REQUIRE(fallback);
            auto fallbackPixels = renderReadback(context.device, context.graphicsQueue, **fallback);
            BINARY_TEST_REQUIRE(fallbackPixels);
            if ((*fallback)->cacheStats().loadStatus != PipelineCacheLoadStatus::Invalid ||
                (*fallback)->cacheStats().binaryCacheHit || !(*fallback)->cacheStats().persisted ||
                *fallbackPixels != *coldPixels) {
                return RHITestResult::fail("damaged binary was not rejected, recompiled and repaired with identical pixels");
            }
            logCreation(truncate ? "truncated-fallback" : "corrupt-fallback", **fallback, *fallbackPixels);
            auto repaired = registry.getGraphicsShaderObjectProgram(context.device, desc);
            BINARY_TEST_REQUIRE(repaired);
            if (!(*repaired)->cacheStats().binaryCacheHit) { return RHITestResult::fail("repaired binary file was not reusable"); }
        }
        // Reconstruct modules on the second device so this checks file reuse without borrowing native handles.
        auto secondDevice = bench::createTestDevice(context, {.applicationName = "Shader Object Binary lifecycle",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = context.device.capabilities().bindlessDescriptorHeap,
            .enableShaderObject = true}, true);
        BINARY_TEST_REQUIRE(secondDevice);
        auto* secondQueue = secondDevice->get()->getQueue(QueueType::Graphics);
        if (!secondQueue) { return RHITestResult::fail("second device has no graphics queue"); }
        auto secondModules = createShaderModules(**secondDevice, code);
        BINARY_TEST_REQUIRE(secondModules);
        auto secondDesc = desc;
        secondDesc.vertexShader.module = (*secondModules)[0].get();
        secondDesc.fragmentShader.module = (*secondModules)[1].get();
        auto second = registry.getGraphicsShaderObjectProgram(**secondDevice, secondDesc);
        BINARY_TEST_REQUIRE(second);
        auto secondPixels = renderReadback(**secondDevice, *secondQueue, **second);
        BINARY_TEST_REQUIRE(secondPixels);
        if (!(*second)->cacheStats().binaryCacheHit || (*second)->cacheStats().programHash != coldStats.programHash ||
            (*second)->execution().deviceIdentity() != secondDevice->get()->identity() ||
            (*second)->execution().deviceIdentity() == context.device.identity() || *secondPixels != *coldPixels) {
            return RHITestResult::fail("independent device did not load the binary pair with identical output and its own native handles");
        }
        logCreation("second-device", **second, *secondPixels);
        std::string imageMessage;
        if (!saveRgba8Png(root / "red.png", reinterpret_cast<const uint8_t*>(coldPixels->data()), kWidth, kHeight, imageMessage) ||
            !saveRgba8Png(root / "green.png", reinterpret_cast<const uint8_t*>(variantPixels->data()), kWidth, kHeight, imageMessage)) {
            return RHITestResult::fail("could not save shader object output: " + imageMessage);
        }
        return RHITestResult::pass("SPIR-V export, native binary reload, variant invalidation, corrupt/truncated repair and independent-device pixel equality");
#undef BINARY_TEST_REQUIRE
    }
};
METALLIC_REGISTER_RHI_TEST(ShaderObjectBinaryPersistenceTest);

class ShaderObjectBinaryProcessTest final : public RHITest {
public:
    ShaderObjectBinaryProcessTest() { type = RHITestType::Rendering; name = "shader_object_binary_process_readback"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return binaryMetadata({"graphics.shaderObject.binary.process.readback"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().shaderObject) { return RHITestResult::skip("VK_EXT_shader_object is unavailable"); }
        const auto root = std::filesystem::absolute(context.outputDirectory / "binary-process");
        std::filesystem::create_directories(root);
        const auto cacheDirectory = root.string();
        ShaderCodes code;
        std::string log;
        if (!compileShaderCodes(code, log)) { return RHITestResult::fail("shader object source compilation failed: " + log); }
        auto modules = createShaderModules(context.device, code);
        if (!modules) { return RHITestResult::fail(std::string("shader module creation failed: ") + resultToString(modules)); }
        auto program = ShaderRegistry::instance().getGraphicsShaderObjectProgram(context.device,
            {.vertexShader = {(*modules)[0].get()}, .fragmentShader = {(*modules)[1].get()},
                .binaryCacheDirectory = cacheDirectory.c_str()});
        if (!program) { return RHITestResult::fail(std::string("shader object creation failed: ") + resultToString(program)); }
        auto pixels = renderReadback(context.device, context.graphicsQueue, **program);
        if (!pixels || !matchesSolidColor(*pixels, 0)) { return RHITestResult::fail("cross-process shader object red readback failed"); }
        const auto stats = (*program)->cacheStats();
        if (stats.binaryDataSize == 0 || (!stats.binaryCacheHit && !stats.persisted)) {
            return RHITestResult::fail("cross-process shader object neither loaded nor persisted a binary pair");
        }
        logCreation("process", **program, *pixels);
        std::ofstream output(root / "process-readback.bin", std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(pixels->data()), std::streamsize(pixels->size()));
        if (!output) { return RHITestResult::fail("could not save cross-process readback bytes"); }
        std::ofstream observations(root / "process-observations.json", std::ios::trunc);
        observations << "{\n  \"binaryCacheHit\": " << (stats.binaryCacheHit ? "true" : "false")
            << ",\n  \"loadStatus\": " << int(stats.loadStatus)
            << ",\n  \"persisted\": " << (stats.persisted ? "true" : "false")
            << ",\n  \"programHash\": " << stats.programHash
            << ",\n  \"binaryDataSize\": " << stats.binaryDataSize
            << ",\n  \"creationTimeNanoseconds\": " << stats.creationTimeNanoseconds
            << ",\n  \"readbackChecksum\": " << readbackChecksum(*pixels) << "\n}\n";
        if (!observations) { return RHITestResult::fail("could not save cross-process creation observations"); }
        std::string imageMessage;
        if (!saveRgba8Png(root / "process-readback.png", reinterpret_cast<const uint8_t*>(pixels->data()),
            kWidth, kHeight, imageMessage)) { return RHITestResult::fail("could not save cross-process image: " + imageMessage); }
        return RHITestResult::pass(stats.binaryCacheHit ? "native binary pair reused across process with red readback" :
            "cold SPIR-V pair persisted for the next process with red readback");
    }
};
METALLIC_REGISTER_RHI_TEST(ShaderObjectBinaryProcessTest);

class ShaderObjectSceneBinaryFramesTest final : public RHITest {
public:
    ShaderObjectSceneBinaryFramesTest() { type = RHITestType::Rendering; name = "shader_object_scene_binary_frames"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return binaryMetadata({"graphics.shaderObject.binary.scene.frames",
            "graphics.shaderObject.binary.material.readback", "graphics.shaderObject.binary.wireframe.readback"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint32_t width = 128, height = 96, frameCount = 8;
        const auto root = std::filesystem::absolute(context.outputDirectory / "shader-object-scene");
        std::filesystem::create_directories(root);
        auto cameraGraph = RenderGraph::createDefaultBunnyGraph();
        const auto* cameraNode = cameraGraph.findNode("Bunny");
        if (!cameraNode) { return RHITestResult::fail("default Bunny graph camera properties are missing"); }
        const RenderGraphProperties sceneProperties = cameraNode->properties;
        const char* passTypes[]{"SceneMaterialShaderObjectPass", "BunnyWireframePass"};
        for (const auto* passType : passTypes) {
            std::array<std::vector<std::byte>, frameCount> reference;
            for (uint32_t lifetime = 0; lifetime < 2; ++lifetime) {
                std::cout << "[ShaderObjectBinaryTest] scene=" << passType << " lifetime=" << lifetime << " begin" << std::endl;
                // A new preview owns a new Device and pass; it must reconstruct executables from disk.
                RenderGraphPreviewRenderer preview;
                auto initialized = preview.initialize(context.enableValidation, false, false);
                if (!initialized) {
                    if (hasError(initialized, Error::Unsupported)) { return RHITestResult::skip("shader object preview capabilities are unavailable"); }
                    return RHITestResult::fail(std::string("shader object scene preview initialization failed: ") + resultToString(initialized));
                }
                preview.setRawReadbackEnabled(true);
                RenderGraph graph;
                graph.setName("ShaderObjectBinaryScene");
                RenderGraphProperties properties = sceneProperties;
                if (std::string_view(passType) == "SceneMaterialShaderObjectPass") { properties["debugAlternateShaders"] = true; }
                graph.addNode(passType, "Scene", std::move(properties));
                graph.markOutput("Scene.color");
                for (uint32_t frame = 0; frame < frameCount; ++frame) {
                    auto rendered = preview.render(graph, width, height, "Scene.color");
                    if (!rendered) {
                        return RHITestResult::fail(std::string(passType) + " frame " + std::to_string(frame) +
                            " render failed: " + resultToString(rendered) + ": " + preview.lastLog());
                    }
                    const auto& bytes = preview.readbackBytes();
                    if (preview.readbackFormat() != Format::RGBA8Unorm || bytes.size() != width * height * 4 ||
                        preview.pixels().size() != width * height) {
                        return RHITestResult::fail(std::string(passType) + " produced an unexpected readback layout");
                    }
                    const uint32_t background = preview.pixels().front();
                    const auto foreground = std::count_if(preview.pixels().begin(), preview.pixels().end(),
                        [&](uint32_t pixel) { return pixel != background; });
                    if (foreground < 128) { return RHITestResult::fail(std::string(passType) + " rendered only a clear or nearly uniform image"); }
                    if (lifetime == 0) { reference[frame] = bytes; }
                    else if (bytes != reference[frame]) {
                        return RHITestResult::fail(std::string(passType) + " cached shader object changed pixels at frame " + std::to_string(frame));
                    }
                    if (frame == 0 || frame + 1 == frameCount) {
                        const auto stem = std::string(passType) + "-lifetime" + std::to_string(lifetime) + "-frame" + std::to_string(frame);
                        std::string imageMessage;
                        if (!saveRgba8Png(root / (stem + ".png"), reinterpret_cast<const uint8_t*>(bytes.data()),
                            width, height, imageMessage)) { return RHITestResult::fail("could not save scene output: " + imageMessage); }
                        std::cout << "[ShaderObjectBinaryTest] scene=" << passType << " lifetime=" << lifetime << " frame=" << frame
                            << " foregroundPixels=" << foreground << " readbackChecksum=" << readbackChecksum(bytes) << '\n';
                    }
                }
                std::cout << "[ShaderObjectBinaryTest] scene=" << passType << " lifetime=" << lifetime << " frames=" << frameCount
                    << " complete" << std::endl;
            }
        }
        return RHITestResult::pass("material and wireframe scene passes: 8 frames per independent preview lifetime, matching raw RGBA8 pixels and visible geometry");
    }
};
METALLIC_REGISTER_RHI_TEST(ShaderObjectSceneBinaryFramesTest);

} // namespace
} // namespace metallic::tests
