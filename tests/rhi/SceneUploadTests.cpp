#include "RHITest.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Scene/SceneLoader.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <spdlog/spdlog.h>

namespace metallic::tests {
namespace {

#define UPLOAD_REQUIRE(expression) do { const auto result = (expression); \
    if (!result) { return RHITestResult::fail(std::string(#expression) + ": " + toString(result) + " " + log); } } while (false)

constexpr uint32_t kTextureCount = 60;
constexpr uint32_t kMipCount = 7;

render::ValidationSink uploadValidationSink(RHITestContext& context)
{
    return {.callback = [](void* data, const render::ValidationMessage& message) noexcept {
        if (data != nullptr && message.messageIdName != nullptr && std::strstr(message.messageIdName, "VUID-") != nullptr) {
            ++*static_cast<std::atomic_uint*>(data);
        }
    }, .context = context.validationMessageCount};
}

std::array<uint32_t, 4> imagePixel(uint32_t image, uint32_t mip, uint32_t x, uint32_t y)
{
    return {(image * 3u + mip * 11u + x) & 255u, (image + y * 7u + mip * 13u) & 255u,
        (x + y + mip * 17u) & 255u, 255u};
}

std::filesystem::path writeUploadScene(const std::filesystem::path& directory)
{
    std::filesystem::create_directories(directory);
    const float positions[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    {
        std::ofstream binary(directory / "scene.bin", std::ios::binary);
        binary.write(reinterpret_cast<const char*>(positions), sizeof(positions));
    }
    auto source = nlohmann::json::parse(R"({
        "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
        "nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
        "buffers":[{"uri":"scene.bin","byteLength":36}],
        "bufferViews":[{"buffer":0,"byteLength":36}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}]
    })");
    for (uint32_t index = 0; index < kTextureCount; ++index) {
        source["images"].push_back({{"uri", std::to_string(index) + ".png"}});
        source["textures"].push_back({{"source", index}});
        source["materials"].push_back({{"pbrMetallicRoughness", {{"baseColorTexture", {{"index", index}}}}}});
    }
    const auto path = directory / "scene.gltf";
    std::ofstream(path) << source.dump();
    return path;
}

// Construct after scene resources so even a failed assertion releases the GPU
// gate before their destructor waits for submitted upload work.
struct UploadQueueDrain {
    render::Semaphore& gate;
    render::Queue& copy;
    render::Queue& graphics;
    uint64_t value = 1;
    ~UploadQueueDrain()
    {
        if (gate.currentValue() < value) { (void)gate.signal(value); }
        (void)copy.waitIdle();
        (void)graphics.waitIdle();
    }
};

class SceneUploadPipelineTest final : public RHITest {
public:
    SceneUploadPipelineTest() { type = RHITestType::Resource; name = "scene_upload_pipeline"; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::string log;
        std::unique_ptr<Device> device;
        const auto initialized = createDevice({.applicationName = "Scene upload pipeline test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .validationSink = uploadValidationSink(context)}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(initialized, Error::Unsupported)) { return RHITestResult::skip("Requires ray tracing and bindless resources"); }
        UPLOAD_REQUIRE(initialized);
        auto* graphics = device->getQueue(QueueType::Graphics);
        auto* copy = device->getQueue(QueueType::Copy);
        if (!graphics) { return RHITestResult::fail("No graphics queue"); }
        if (!copy) { copy = graphics; }
        const auto path = writeUploadScene(context.outputDirectory / "scene-upload");
        scene::Scene scene;
        if (!scene.load(path)) { return RHITestResult::fail(scene.lastLoadResult().error); }
        for (uint32_t index = 0; index < kTextureCount; ++index) {
            std::vector<scene::RenderImage::Mip> mips;
            uint32_t width = 65, height = 33;
            for (uint32_t level = 0; level < kMipCount; ++level) {
                scene::RenderImage::Mip mip{.width = width, .height = height};
                mip.pixels.resize(width * height * 4u);
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        const auto color = imagePixel(index, level, x, y);
                        for (uint32_t c = 0; c < 4; ++c) { mip.pixels[(y * width + x) * 4u + c] = uint8_t(color[c]); }
                    }
                }
                mips.push_back(std::move(mip));
                width = std::max(width / 2u, 1u);
                height = std::max(height / 2u, 1u);
            }
            if (!scene.setImageDecodeResult(index, std::move(mips), {})) { return RHITestResult::fail("Failed to set fixture mip chain"); }
        }
        std::unique_ptr<Semaphore> gate;
        UPLOAD_REQUIRE(device->createSemaphore({}).transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        ScenePathTraceResources resources;
        UploadQueueDrain drain{*gate, *copy, *graphics};
        const SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1, .stages = PipelineStageBits::AllCommands};
        UPLOAD_REQUIRE(copy->submit({.waitSemaphores = {&wait, 1}}));
        UPLOAD_REQUIRE(resources.beginPrepareAsync(*device, *graphics, {{"path", path.string()}}, scene, log));
        bool complete = false;
        scene::SceneLoadProgress progress;
        for (uint32_t pump = 0; pump < 20 && resources.uploadStats().submittedBatches < 3; ++pump) {
            UPLOAD_REQUIRE(resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { complete = std::move(value); }));
        }
        auto stats = resources.uploadStats();
        if (stats.submittedBatches != 3 || stats.inFlightBatches != 3 || stats.completedBatches != 0 ||
            complete || resources.textureUploadsReady()) {
            return RHITestResult::fail("Expected three concurrent batches before releasing the GPU gate");
        }
        for (uint32_t pump = 0; pump < 3; ++pump) { UPLOAD_REQUIRE(resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { complete = std::move(value); })); }
        if (resources.uploadStats().submittedBatches != 3) { return RHITestResult::fail("Upload backpressure exceeded three batches"); }
        UPLOAD_REQUIRE(gate->signal(1));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!complete && std::chrono::steady_clock::now() < deadline) {
            UPLOAD_REQUIRE(resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { complete = std::move(value); }));
            if (!complete) { std::this_thread::yield(); }
        }
        stats = resources.uploadStats();
        if (!complete || !resources.valid() || !resources.textureUploadsReady() ||
            resources.materialTextureCount() != kTextureCount + 1 || stats.submittedBatches < 4 ||
            stats.submittedBatches != stats.completedBatches || stats.inFlightBatches != 0 || stats.peakInFlightBatches != 3) {
            return RHITestResult::fail("Upload pipeline did not drain every batch");
        }

        ShaderCompileResult shader;
        UPLOAD_REQUIRE(compileSlangShaderToSpirv({.moduleName = "SceneUploadProbe", .entryPointName = "sceneUploadProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
        ComputeProgram program;
        const ComputeProgramBindingDesc layout[] = {{0, ComputeResourceBindingKind::SampledImage}, {1}};
        UPLOAD_REQUIRE(program.initialize(*device, {
            .spirv = shader.spirv,
            .pushConstantSize = 4,
            .bindings = {layout, 2},
            .requiresRayQuery = false,
        }, log));
        std::unique_ptr<Buffer> output;
        UPLOAD_REQUIRE(device->createBuffer({.size = kTextureCount * kMipCount * 2u * 16u, .structureStride = 16,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        QueueSubmissionTracker tracker;
        UPLOAD_REQUIRE(tracker.initialize(*device, *graphics));
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        UPLOAD_REQUIRE(device->createCommandPool(*graphics).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        UPLOAD_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        RenderFrameContext frame;
        struct FrameDrain {
            RenderFrameContext& frame;
            CommandPool& pool;
            ~FrameDrain() { if (frame.completion().isSubmitted()) { (void)frame.wait(); } (void)pool.reset(); (void)frame.reset(); }
        } frameDrain{frame, *pool};
        UPLOAD_REQUIRE(frame.begin(0));
        UPLOAD_REQUIRE(commands->begin(&frame));
        for (uint32_t index = 0; index < kTextureCount; ++index) {
            const ComputeDispatchBinding bindings[] = {
                {.binding = 0, .textureViews = {resources.materialTextureViews().data() + index + 1, 1}},
                {.binding = 1, .buffer = output.get()},
            };
            const uint32_t offset = index * kMipCount * 2u;
            UPLOAD_REQUIRE(program.dispatch({
                .commandBuffer = commands.get(),
                .bindings = {bindings, 2},
                .pushData = &offset,
                .pushDataSize = 4,
            }));
        }
        UPLOAD_REQUIRE(commands->end());
        CommandBuffer* submitted[] = {commands.get()};
        UPLOAD_REQUIRE(tracker.submit({.commandBuffers = {submitted, 1}}, frame));
        UPLOAD_REQUIRE(frame.wait(30'000'000'000ull));
        output->invalidate();
        const auto* pixels = static_cast<const std::array<uint32_t, 4>*>(output->map());
        if (!pixels) { return RHITestResult::fail("Readback map failed"); }
        bool matches = true;
        for (uint32_t index = 0; index < kTextureCount; ++index) {
            uint32_t width = 65, height = 33;
            for (uint32_t mip = 0; mip < kMipCount; ++mip) {
                const uint32_t offset = (index * kMipCount + mip) * 2u;
                matches &= pixels[offset] == imagePixel(index, mip, 0, 0);
                matches &= pixels[offset + 1] == imagePixel(index, mip, width - 1, height - 1);
                width = std::max(width / 2u, 1u); height = std::max(height / 2u, 1u);
            }
        }
        output->unmap();
        if (!matches) { return RHITestResult::fail("Mip pixels changed across batch retirement/staging reuse"); }

        // Rebuild and cancel with uploads still pending. Signal from another
        // thread so clear() must drain work before releasing its resources.
        resources.clear();
        drain.value = 2;
        const SemaphoreSubmitDesc nextWait{.semaphore = gate.get(), .value = 2, .stages = PipelineStageBits::AllCommands};
        UPLOAD_REQUIRE(copy->submit({.waitSemaphores = {&nextWait, 1}}));
        UPLOAD_REQUIRE(resources.beginPrepareAsync(*device, *graphics, {{"path", path.string()}}, scene, log));
        for (uint32_t pump = 0; pump < 20 && resources.uploadStats().submittedBatches < 3; ++pump) {
            UPLOAD_REQUIRE(resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { complete = std::move(value); }));
        }
        if (resources.uploadStats().inFlightBatches != 3) { return RHITestResult::fail("Cancel probe did not fill upload window"); }
        std::jthread release([&] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); (void)gate->signal(2); });
        resources.clear();
        if (resources.valid() || resources.preparing() || !resources.gpuWorkComplete() || resources.uploadStats().inFlightBatches != 0) {
            return RHITestResult::fail("Clear left pending scene uploads");
        }
        return RHITestResult::pass("Three batches in flight, bounded backpressure, 420 mips verified, in-flight clear drained");
    }
};

METALLIC_REGISTER_RHI_TEST(SceneUploadPipelineTest);

class SuperSponzaUploadSmokeTest final : public RHITest {
public:
    SuperSponzaUploadSmokeTest() { type = RHITestType::Resource; name = "super_sponza_upload_smoke"; }
    RHITestResult run(RHITestContext& context) override
    {
        if (!std::getenv("METALLIC_TEST_SUPER_SPONZA_UPLOAD")) {
            return RHITestResult::skip("Set METALLIC_TEST_SUPER_SPONZA_UPLOAD=1 to load the large fixture");
        }
        const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf";
        if (!std::filesystem::exists(path)) { return RHITestResult::skip("Super Sponza fixture is unavailable"); }
        std::string log;
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "Super Sponza upload smoke",
            .enableValidation = context.enableValidation, .enableRayTracingAccelerationStructure = true,
            .validationSink = uploadValidationSink(context)}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires ray tracing"); }
        UPLOAD_REQUIRE(initialized);
        scene::SceneLoader loader;
        auto handle = loader.request(path);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!handle.complete() && std::chrono::steady_clock::now() < deadline) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        auto loaded = handle.takeResult();
        if (!loaded) { return RHITestResult::fail("CPU load failed: " + handle.progress().error); }
        render::ScenePathTraceResources resources;
        const auto begin = std::chrono::steady_clock::now();
        UPLOAD_REQUIRE(resources.beginPrepareAsync(*device, *device->getQueue(render::QueueType::Graphics),
            {{"path", path.string()}, {"materialTextureBudgetMiB", 8192}}, *loaded, log));
        scene::SceneLoadProgress progress;
        bool complete = false;
        while (!complete && std::chrono::steady_clock::now() < deadline) {
            UPLOAD_REQUIRE(resources.pumpPrepareAsync(8.0, progress, log).transform([&](auto value) { complete = std::move(value); }));
            if (!complete) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        }
        if (!complete || !resources.valid() || !resources.gpuWorkComplete()) { return RHITestResult::fail("GPU preparation timed out: " + log); }
        const auto stats = resources.uploadStats();
        const auto textures = resources.textureStats();
        std::ofstream(context.outputDirectory / "SuperSponzaUploadAllocation.json") <<
            nlohmann::json{{"logicalTextures", textures.logicalTextureCount}, {"residentIncludingFallback", textures.residentImageCount},
                {"payloadBytes", textures.residentPayloadBytes}, {"allocationBytes", textures.residentAllocationBytes},
                {"peakLiveBytes", textures.peakLiveAllocationBytes}, {"budgetBytes", textures.budgetBytes},
                {"peakStagingBytes", textures.peakStagingBytes}, {"submittedUploadBytes", stats.submittedBytes}}.dump(2) << '\n';
        spdlog::info("[SceneUploadSmoke] GPU preparation including AS: {:.2f} ms, batches={}, peakInFlight={}, bytes={}",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(),
            stats.completedBatches, stats.peakInFlightBatches, stats.submittedBytes);
        return RHITestResult::pass("Super Sponza CPU decode, material uploads, and acceleration structures completed");
    }
};

METALLIC_REGISTER_RHI_TEST(SuperSponzaUploadSmokeTest);

class SponzaAsyncRtasTest final : public RHITest {
public:
    SponzaAsyncRtasTest() { type = RHITestType::Resource; name = "sponza_async_scene_rtas"; }
    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().rayTracingAccelerationStructure ||
            !context.device.capabilities().opacityMicromap) {
            return RHITestResult::skip("Requires --rhi-realtime with OMM enabled");
        }
        const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Sponza/glTF/Sponza.gltf";
        std::string log;
        scene::SceneLoader loader;
        auto handle = loader.request(path);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!handle.complete() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto loaded = handle.takeResult();
        if (!loaded) { return RHITestResult::fail("CPU load failed: " + handle.progress().error); }
        render::ScenePathTraceResources resources;
        UPLOAD_REQUIRE(resources.beginPrepareAsync(context.device, context.graphicsQueue,
            {{"path", path.string()}}, *loaded, log));
        scene::SceneLoadProgress progress;
        bool complete = false;
        while (!complete && std::chrono::steady_clock::now() < deadline) {
            UPLOAD_REQUIRE(resources.pumpPrepareAsync(8.0, progress, log).transform([&](auto value) { complete = std::move(value); }));
            // Also surfaces device loss when an asynchronous fence poll is not ready.
            UPLOAD_REQUIRE(context.graphicsQueue.waitIdle());
        }
        if (!complete || !resources.valid() || !resources.gpuWorkComplete()) {
            return RHITestResult::fail("Sponza GPU preparation timed out: " + log);
        }
        const auto& stats = resources.accelerationStructure().stats();
        if (stats.blasCount != 103 || stats.opacityMicromapCount != 10 ||
            stats.opacityMicromapTriangleCount != 34908 || stats.compactionSavedBytes == 0) {
            return RHITestResult::fail("Sponza did not exercise OMM and BLAS compaction");
        }
        return RHITestResult::pass("Sponza async upload, 10 OMMs, 103 BLASes, compaction and TLAS completed");
    }
};

METALLIC_REGISTER_RHI_TEST(SponzaAsyncRtasTest);

#undef UPLOAD_REQUIRE
} // namespace
} // namespace metallic::tests
