#include "TestResourceLayouts.h"
#include "RHITest.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/HZBSPD.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

uint32_t hzbElements(uint32_t width, uint32_t height, uint32_t& mips)
{
    uint32_t count = 0;
    mips = 0;
    for (;;) {
        ++mips;
        count += width * height;
        if (width == 1 && height == 1) { return count; }
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }
}

class HZBSPDProbe final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        auto& depth = reflection.addTextureOutput("depth").storageReadWrite();
        depth.format = render::Format::R32Sfloat;
        depth.usage = depth.usage | render::TextureUsageBits::Sampled;
        uint32_t mips = 0;
        reflection.addBufferOutput("data").buffer(hzbElements(context.width, context.height, mips) * 4ull, 4)
            .storageReadWrite().memoryLocation = render::MemoryLocation::HostReadback;
        reflection.addBufferOutput("counter").buffer(8, 4).storageReadWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "HZBSPDFixture",
            .entryPointName = "hzbSpdFixtureMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageImage}, {.binding = 1}};
        result = fixture_.initialize(*context.device, {
            .spirv = shader.spirv,
            .pushConstantSize = 16,
            .bindings = {bindings, 2},
            .requiresRayQuery = false,
            .resourceParameters = metallic::tests::kHZBSPDFixtureLayout,
        }, log);
        if (!result) { return result; }
        const render::SlangMacroDefine waveDefine{render::kHZBSPDWaveOpsDefine,
            properties().value("waveOps", true) ? "1" : "0"};
        result = render::compileSlangShaderToSpirv({
            .moduleName = render::kHZBSPDModule,
            .entryPointName = render::kHZBSPDEntryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .macroDefines = {&waveDefine, 1},
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        // Wave indexing relies on full subgroups, guaranteed by SPIR-V 1.6
        // with numthreads.x=256 and the supported subgroup-size range.
        if (shader.spirv.size() < 2 || shader.spirv[1] < 0x00010600u) {
            log = "SPD wave operations require SPIR-V 1.6";
            return render::makeError(render::Error::Failure);
        }
        result = context.device->createShaderModule({.spirv = shader.spirv}).transform([&](auto rhiValue) { shader_ = std::move(rhiValue); });
        if (!result) { return result; }
        result = context.device->createComputePipeline({
            .computeShader = {shader_.get()},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(render::HZBSPDUserPush),
        }).transform([&](auto rhiValue) { pipeline_ = std::move(rhiValue); });
        if (!result) { return result; }
        result = context.device->createBindlessHeap({.maxSampledImages = 1, .maxBuffers = 2}).transform([&](auto rhiValue) { heap_ = std::move(rhiValue); });
        if (result) { result = heap_->allocate(metallic::render::BindlessHandleKind::SampledImage).transform([&](auto rhiValue) { depth_ = std::move(rhiValue); }); }
        if (result) { result = heap_->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { data_ = std::move(rhiValue); }); }
        if (result) { result = heap_->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { counter_ = std::move(rhiValue); }); }
        return result;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const uint32_t seed = uint32_t(context.frameIndex());
        const uint32_t reversed = context.properties().value("reversedZ", true) ? 1u : 0u;
        uint32_t constants[] = {context.width(), context.height(), seed, reversed};
        const auto depth = context.outputTexture("depth");
        auto* data = context.outputBuffer("data").buffer();
        auto* counter = context.outputBuffer("counter").buffer();
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureView = depth.view()}, {.binding = 1, .buffer = counter}};
        auto result = fixture_.dispatch({
            .commandBuffer = &context.commandBuffer(),
            .bindings = {bindings, 2},
            .pushData = constants,
            .pushDataSize = sizeof(constants),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
        });
        if (!result) { return result; }
        render::TextureBarrierDesc depthBarrier{
            .texture = depth.texture(),
            .oldLayout = render::TextureLayout::General,
            .newLayout = render::TextureLayout::ShaderRead,
            .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead},
        };
        const render::BufferBarrierDesc counterBarrier{
            .buffer = counter,
            .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
        };
        if (auto commandResult = context.commandBuffer().synchronize({.textures = {&depthBarrier, 1}, .buffers = {&counterBarrier, 1}}); !commandResult) { return commandResult; }
        result = heap_->writeSampledImage(depth_, *depth.view(), render::TextureLayout::ShaderRead);
        if (result) { result = heap_->writeStorageBuffer(data_, *data); }
        if (result) { result = heap_->writeStorageBuffer(counter_, *counter); }
        if (!result) { return result; }
        uint32_t mips = 0;
        hzbElements(context.width(), context.height(), mips);
        const render::HZBSPDUserPush push{.depthImage = depth_.shaderIndex, .hzbBuffer = data_.shaderIndex,
            .counterBuffer = counter_.shaderIndex, .width = context.width(), .height = context.height(),
            .mipCount = mips, .reversedZ = reversed};
        context.commandBuffer().bindBindlessHeap(*heap_);
        if (auto commandResult = context.commandBuffer().bindExecution((pipeline_)->execution()); !commandResult) { return commandResult; }
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().dispatch((context.width() + 63) / 64, (context.height() + 63) / 64);
        std::swap(depthBarrier.before, depthBarrier.after); std::swap(depthBarrier.oldLayout, depthBarrier.newLayout);
        if (auto commandResult = context.commandBuffer().synchronize({.textures = {&depthBarrier, 1}}); !commandResult) { return commandResult; }
        return {};
    }
private:
    render::ComputeProgram fixture_;
    std::unique_ptr<render::ShaderModule> shader_;
    std::unique_ptr<render::ComputePipeline> pipeline_;
    std::unique_ptr<render::BindlessHeap> heap_;
    render::BindlessHandle depth_, data_, counter_;
};

class HZBSPDReductionTest final : public RHITest {
public:
    HZBSPDReductionTest() { type = RHITestType::Rendering; name = "hzb_spd_conservative_reduction"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "SPD HZB reduction",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
        if (!result) { return RHITestResult::fail("SPD device creation failed"); }
        render::registerRenderGraphPassType("HZBSPDProbe", "SPD reduction probe", [] { return std::make_unique<HZBSPDProbe>(); });
        render::RenderGraph graph;
        const auto node = graph.addNode("HZBSPDProbe", "Probe")->id;
        graph.markOutput("Probe.data");
        graph.markOutput("Probe.counter");
        render::RenderGraphExecutor executor;
        const std::array<std::array<uint32_t, 2>, 14> sizes{{
            {1, 1}, {1, 131}, {133, 1}, {7, 5}, {15, 16}, {16, 16}, {17, 31}, {32, 33},
            {63, 65}, {64, 64}, {65, 129}, {799, 293}, {1920, 1080}, {4096, 4095}}};
        std::string log;
        spdlog::info("[SPD] subgroup range {}..{} shuffle={}", device->capabilities().minSubgroupSize,
            device->capabilities().maxSubgroupSize, device->capabilities().computeSubgroupShuffle);
        uint32_t pyramids = 0;
        for (bool waveOps : {false, true}) {
            if (waveOps && !render::supportsHzbSpdWaveOps(device->capabilities())) {
                spdlog::info("[SPD] wave variant unavailable; validated LDS fallback only");
                continue;
            }
            for (bool reversed : {false, true}) {
                graph.setNodeProperties(node, {{"reversedZ", reversed}, {"waveOps", waveOps}});
                for (auto size : sizes) {
                    if (!executor.compile(*device, graph, size[0], size[1], log)) { return RHITestResult::fail(log); }
                    // Repeated builds with different depth reveal stale tail/counter data.
                    for (uint32_t frame = 0; frame < 3; ++frame) {
                        if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
                            !executor.waitForSubmittedWork()) { return RHITestResult::fail("SPD dispatch failed"); }
                        auto* counter = executor.outputResource("Probe.counter")->buffer;
                        counter->invalidate();
                        const auto* count = static_cast<const uint32_t*>(counter->map());
                        if (!count) { return RHITestResult::fail("SPD counter readback failed"); }
                        bool valid = count[0] == 0;
                        const uint32_t seed = count[1];
                        counter->unmap();
                        auto* buffer = executor.outputResource("Probe.data")->buffer;
                        buffer->invalidate();
                        const auto* data = static_cast<const float*>(buffer->map());
                        if (!data) { return RHITestResult::fail("SPD readback failed"); }
                        uint32_t width = size[0], height = size[1], offset = 0;
                        for (uint32_t y = 0; y < height; ++y) {
                            for (uint32_t x = 0; x < width; ++x) {
                                const uint32_t tile = ((x / 64u) * 7u + (y / 64u) * 13u + seed * 17u) % 64u;
                                float expected = float(tile * 1024u + (x * 73u + y * 137u + seed * 31u) % 1024u + 1u) / 65537.0f;
                                if (x + 1 == width || y + 1 == height || (x == 0 && y == 0)) {
                                    expected = reversed ? 0.0f : 1.0f;
                                }
                                valid = valid && std::abs(data[y * width + x] - expected) < 1e-7f;
                            }
                        }
                        while (width > 1 || height > 1) {
                            const uint32_t nextWidth = (width + 1) / 2, nextHeight = (height + 1) / 2;
                            const uint32_t nextOffset = offset + width * height;
                            for (uint32_t y = 0; y < nextHeight; ++y) {
                                for (uint32_t x = 0; x < nextWidth; ++x) {
                                    float expected = reversed ? 1.0f : 0.0f;
                                    for (uint32_t dy = 0; dy < 2; ++dy) {
                                        for (uint32_t dx = 0; dx < 2; ++dx) {
                                            const float sample = data[offset + std::min(y * 2 + dy, height - 1) * width + std::min(x * 2 + dx, width - 1)];
                                            expected = reversed ? std::min(expected, sample) : std::max(expected, sample);
                                        }
                                    }
                                    valid = valid && data[nextOffset + y * nextWidth + x] == expected;
                                }
                            }
                            offset = nextOffset; width = nextWidth; height = nextHeight;
                        }
                        buffer->unmap();
                        if (!valid) { return RHITestResult::fail("SPD mip/counter mismatch at " + std::to_string(size[0]) + "x" + std::to_string(size[1]) + " waveOps=" + std::to_string(waveOps)); }
                    }
                    spdlog::info("[SPD] {}x{} reversed={} waveOps={} every mip exact", size[0], size[1], reversed, waveOps);
                }
            }
            pyramids += uint32_t(sizes.size()) * 2u * 3u;
        }
        return RHITestResult::pass(std::to_string(pyramids) + " SPD wave/LDS pyramids exactly match CPU min/max, including odd edges, 1D, NaN and repeated tails");
    }
};

class HZBSPDVisibilityTest final : public RHITest {
public:
    HZBSPDVisibilityTest() { type = RHITestType::Rendering; name = "hzb_spd_visibility_equivalence_timing"; }
    RHITestResult run(RHITestContext& context) override
    {
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Sponza/glTF/Sponza.gltf")) {
            return RHITestResult::fail(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders"); }
        if (!initialized) { return RHITestResult::fail("SPD preview initialization failed"); }
        preview.bindRuntimeScene(&scene);
        render::RenderGraph graph;
        const auto node = graph.addNode("VisibilityBufferPass", "VBuffer", {{"visualization", "triangle"}})->id;
        graph.setViewProperties({{"camera", {{"eye", {-3.325359, 9.545668, -0.429330}},
            {"center", {31.658249, -17.904564, -5.697884}}, {"fovDegrees", 45.0},
            {"znear", 0.018548}, {"zfar", 1854.789185}, {"reversedZ", true}}}, {"temporalJitter", false}});
        graph.markOutput("VBuffer.color");
        // Measure the complete VisibilityBuffer pass, including both HZB builds
        // and counter reset copies. These are diagnostic timings, not thresholds.
        for (auto size : {std::array{799u, 293u}, std::array{1920u, 1080u}, std::array{4097u, 65u}}) {
            std::vector<uint32_t> reference;
            for (uint32_t mode = 0; mode < 3; ++mode) {
                const bool spd = mode != 0;
                const bool waveOps = mode == 2;
                graph.setNodeRuntimeProperty(node, "hzbSpd", spd);
                graph.setNodeRuntimeProperty(node, "hzbSpdWaveOps", waveOps);
                std::vector<double> timings;
                for (uint32_t frame = 0; frame < 48; ++frame) {
                    if (!preview.render(graph, size[0], size[1])) { return RHITestResult::fail(preview.lastLog()); }
                    if (frame == 47 && !spd) { reference = preview.pixels(); }
                    if (spd && preview.pixels() != reference) { return RHITestResult::fail("SPD changed visible Sponza triangles"); }
                    std::vector<render::RenderGraphExecutionStats> completed;
                    if (!preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) { return RHITestResult::fail("SPD timing readback failed"); }
                    if (frame >= 16) {
                        for (const auto& execution : completed) {
                            for (const auto& pass : execution.nodes) {
                                if (pass.name == "VBuffer" && pass.gpuTimingAvailable) { timings.push_back(pass.gpuMilliseconds); }
                            }
                        }
                    }
                }
                if (!timings.empty()) {
                    std::sort(timings.begin(), timings.end());
                    const auto* device = preview.subsystemHost()->device();
                    const bool waveSupported = device && render::supportsHzbSpdWaveOps(device->capabilities());
                    const bool actualSpd = spd && size[0] <= render::kHZBSPDMaxDimension && size[1] <= render::kHZBSPDMaxDimension;
                    spdlog::info("[SPD timing] {}x{} requestedSpd={} requestedWaveOps={} actualSpd={} actualWaveOps={} VBuffer median={} ms samples={}",
                        size[0], size[1], spd, waveOps, actualSpd, actualSpd && waveOps && waveSupported,
                        timings[timings.size() / 2], timings.size());
                }
            }
        }
        return RHITestResult::pass("SPD wave/LDS/multi-dispatch Sponza visibility matches, including >4096 fallback; timings include both HZBs");
    }
};

METALLIC_REGISTER_RHI_TEST(HZBSPDReductionTest);
METALLIC_REGISTER_RHI_TEST(HZBSPDVisibilityTest);
} // namespace
} // namespace metallic::tests
