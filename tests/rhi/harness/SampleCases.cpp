#include "Fixtures.h"
#include "RHITest.h"
#include "Evidence.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <array>
#include <cstring>
#include <thread>

namespace metallic::tests {
namespace {

class BufferRangeContractTest final : public RHITest {
public:
    BufferRangeContractTest() { type = RHITestType::Validation; name = "buffer_range_cpu_contract"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .requirements = {.requiresDevice = false,
            .validation = bench::Validation::Off, .queues = {}}, .coverage = {"buffer.range.contract"}, .artifacts = {"ranges.json"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        bench::Evidence evidence(context.outputDirectory / ("cpu-contract-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())));
        return runCpu(evidence);
    }
    RHITestResult runCpu(bench::Evidence& evidence) override
    {
        const auto tail = render::BufferRange{8}.resolve(16);
        const auto empty = render::BufferRange{16, 0}.resolve(16);
        const auto overflow = render::BufferRange{8, UINT64_MAX - 1}.resolve(16);
        const auto outside = render::BufferRange{17, 0}.resolve(16);
        evidence.json("ranges.json", {{"tailSize", tail ? tail->size : UINT64_MAX}, {"emptyAccepted", bool(empty)},
            {"overflowError", render::resultToString(overflow)}, {"outsideError", render::resultToString(outside)}});
        if (!tail || tail->size != 8 || !empty || empty->size != 0 ||
            !render::hasError(overflow, render::Error::InvalidArgument) || !render::hasError(outside, render::Error::InvalidArgument)) {
            return RHITestResult::fail("range resolution/overflow contract violated");
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(BufferRangeContractTest);

class BufferCopyReadbackTest final : public RHITest {
public:
    BufferCopyReadbackTest() { type = RHITestType::Command; name = "buffer_copy_offset_readback"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.coverage = {"buffer.copy.readback", "buffer.copy.offset", "buffer.copy.preserve"},
            .artifacts = {"source.bin", "expected.bin", "actual.bin", "diff.json"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto upload = context.device.createBuffer({.size = 256, .usage = render::BufferUsageBits::TransferSource,
            .memoryLocation = render::MemoryLocation::HostUpload});
        auto readback = context.device.createBuffer({.size = 256, .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback});
        auto pool = context.device.createCommandPool(context.graphicsQueue);
        if (!upload || !readback || !pool) { return RHITestResult::fail("copy fixture allocation failed"); }
        auto commands = (*pool)->createCommandBuffer();
        auto fence = context.device.createFence(false);
        if (!commands || !fence) { return RHITestResult::fail("copy fixture command/fence allocation failed"); }
        std::array<std::byte, 256> source, expected, actual;
        const uint64_t seed = context.evidence ? bench::readJson(context.evidence->root() / "input.json").at("seed").get<uint64_t>() : 1;
        for (size_t i = 0; i < source.size(); ++i) { source[i] = std::byte((i * 29 + seed) & 255); }
        expected.fill(std::byte{0xa7});
        std::memcpy(expected.data() + 64, source.data() + 16, 128);
        auto* input = (*upload)->map();
        auto* output = (*readback)->map();
        if (!input || !output) { return RHITestResult::fail("copy fixture mapping failed"); }
        std::memcpy(input, source.data(), source.size());
        std::memset(output, 0xa7, actual.size());
        (*upload)->flush(); (*readback)->flush();
        (*upload)->unmap(); (*readback)->unmap();
#define TB_REQUIRE(expression) do { const auto checked = (expression); if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + render::resultToString(checked)); } } while (false)
        TB_REQUIRE((*commands)->begin());
        (*commands)->hostWriteBarrier();
        auto src = (*upload)->slice({16, 128});
        auto dst = (*readback)->slice({64, 128});
        if (!src || !dst) { return RHITestResult::fail("copy fixture slicing failed"); }
        TB_REQUIRE((*commands)->copyBuffer(*src, *dst));
        TB_REQUIRE((*commands)->end());
        render::CommandBuffer* submitted[]{commands->get()};
        TB_REQUIRE(context.graphicsQueue.submit({.commandBuffers = submitted, .signalFence = fence->get()}));
        // A failed timed wait must not release resources still in use. The parent
        // process watchdog bounds the drain on that exceptional path.
        const auto completion = (*fence)->wait(5'000'000'000ull);
        if (!completion) {
            const auto drain = context.device.waitIdle();
            return RHITestResult::fail(std::string("copy completion: ") + render::resultToString(completion) +
                "; drain: " + render::resultToString(drain));
        }
        output = (*readback)->map();
        if (!output) { return RHITestResult::fail("copy readback mapping failed"); }
        (*readback)->invalidate();
        std::memcpy(actual.data(), output, actual.size());
        (*readback)->unmap();
        if (context.evidence) {
            context.evidence->bytes("source.bin", source);
            context.evidence->bytes("expected.bin", expected);
            context.evidence->bytes("actual.bin", actual);
            size_t first = 0;
            while (first < actual.size() && actual[first] == expected[first]) { ++first; }
            context.evidence->json("diff.json", {{"equal", actual == expected}, {"firstMismatch", first},
                {"sourceOffset", 16}, {"destinationOffset", 64}, {"size", 128}, {"seed", seed}});
        }
        return actual == expected ? RHITestResult::pass() : RHITestResult::fail("offset copy changed data/sentinel; see diff.json");
#undef TB_REQUIRE
    }
};
METALLIC_REGISTER_RHI_TEST(BufferCopyReadbackTest);

// Safe activation probe: record a transfer WAW hazard, then discard the
// command buffer. The invalid sequence is never submitted to the GPU.
class SyncActivationProbe final : public RHITest {
public:
    SyncActivationProbe() { type = RHITestType::Validation; name = "sync_activation_probe"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "validation-probe", .layer = bench::Layer::Harness};
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto texture = context.device.createTexture({.usage = render::TextureUsageBits::TransferDestination,
            .format = render::Format::RGBA8Unorm, .width = 4, .height = 4});
        if (!texture) { return RHITestResult::fail("probe allocation failed"); }
        bench::GPUCommands recording(context.graphicsQueue);
        const auto initialized = recording.initialize(context.device);
        if (!initialized) { return RHITestResult::fail(render::resultToString(initialized)); }
        const render::TextureBarrierDesc barrier{.texture = texture->get(),
            .oldLayout = render::TextureLayout::Undefined, .newLayout = render::TextureLayout::TransferDestination,
            .before = {}, .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite}, .range = {0, 1, 0, 1}};
        const auto transition = recording.commands->synchronize({.textures = {&barrier, 1}});
        if (!transition) { return RHITestResult::fail(render::resultToString(transition)); }
        for (uint32_t i = 0; i < 2; ++i) {
            recording.commands->clearColorTexture(**texture, render::ResourceState::TransferDestination, {1, 0, 0, 1});
        }
        const auto ended = recording.commands->end();
        return ended ? RHITestResult::pass() : RHITestResult::fail(render::resultToString(ended));
    }
};

// Explicit harness-fixtures suite only. These never run in the legacy registry.
class FaultCase final : public RHITest {
public:
    explicit FaultCase(std::string mode) : mode_(std::move(mode)) { type = RHITestType::Validation; name = mode_.c_str(); }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "harness-fixtures", .layer = bench::Layer::Harness,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .timeout = std::chrono::milliseconds(mode_ == "fixture_00_timeout" ? 1500 : 30000)};
    }
    RHITestResult run(RHITestContext&) override { return RHITestResult::fail("fixture must run through testbench"); }
    RHITestResult runCpu(bench::Evidence& evidence) override
    {
        if (mode_ == "fixture_crash") { std::_Exit(3); }
        if (mode_ == "fixture_00_timeout") { std::this_thread::sleep_for(std::chrono::seconds(60)); }
        if (mode_ == "fixture_fail_cleanup") { return RHITestResult::fail("first run failure"); }
        if (mode_ == "fixture_skip") { return RHITestResult::skip("unexpected inner skip"); }
        if (mode_ == "fixture_shader_failure") {
            std::string diagnostics;
            const auto shader = render::compileSlangShaderToSpirv({.moduleName = "Features/Samples/Triangle",
                .entryPointName = "missingTestbenchEntryPoint", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, diagnostics);
            evidence.json("shader.json", {{"error", render::resultToString(shader)}, {"diagnostics", diagnostics}});
            return RHITestResult::fail(shader ? "invalid shader unexpectedly compiled" : "shader compilation failed: " + diagnostics);
        }
        return RHITestResult::pass();
    }
    void cleanupCpu() override
    {
        if (mode_ == "fixture_cleanup" || mode_ == "fixture_fail_cleanup") { throw std::runtime_error("cleanup failure"); }
    }
private:
    std::string mode_;
};

} // namespace

std::vector<RHITestRegistry::Factory> testbenchFaultFactories()
{
    std::vector<RHITestRegistry::Factory> result;
    result.emplace_back([] { return std::make_unique<SyncActivationProbe>(); });
    for (const auto* mode : {"fixture_pass", "fixture_cleanup", "fixture_fail_cleanup", "fixture_crash",
        "fixture_00_timeout", "fixture_skip", "fixture_shader_failure"}) {
        result.emplace_back([mode] { return std::make_unique<FaultCase>(mode); });
    }
    return result;
}
} // namespace metallic::tests
