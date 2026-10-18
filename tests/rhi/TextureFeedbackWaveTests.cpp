#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using UInt4 = std::array<uint32_t, 4>;
constexpr uint64_t kWaveFeedbackABI = 0x5445585741564501ull;
constexpr uint32_t kTextureCount = 64;
struct TextureFeedbackWaveParams
{
    GPUBufferSpan inputs, feedback;
    uint32_t laneCount, iterations;
};
static_assert(sizeof(TextureFeedbackWaveParams) == 32);

void requireWaveFeedback(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}
template<typename T> void requireWaveFeedback(const Result<T>& result, const std::string& message)
{
    requireWaveFeedback(bool(result), message);
}

constexpr std::array kPatternNames{"one-texture", "two-textures", "three-textures", "distinct-lanes",
    "sparse-lanes", "lane-31-only", "empty", "cross-wave-hotspots", "partial-group"};

RHITestResult runWaveFeedback(RHITestContext& context, bool benchmark)
{
    try {
        bench::TestDevice device;
        requireWaveFeedback(bench::createTestDevice(context, {.applicationName = "Texture feedback wave aggregation",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); }), "Device");
        auto* queue = device->getQueue(QueueType::Graphics);
        if (!device->capabilities().shaderMaximalReconvergence || !device->capabilities().computeSubgroupBallotArithmetic) {
            return RHITestResult::skip("Shader maximal reconvergence unavailable for wave feedback aggregation");
        }
        if (benchmark && (!device->capabilities().timestampQueries || !queue->timestampValidBits())) {
            return RHITestResult::skip("GPU timestamps unavailable");
        }
        ResourceRegistry registry;
        requireWaveFeedback(registry.initialize(*device), "Registry");
        constexpr uint32_t kMaximumLanes = 65536;
        constexpr uint64_t kInputBytes = kMaximumLanes * sizeof(UInt4);
        constexpr uint64_t kFeedbackBytes = kTextureCount * 8 * sizeof(uint32_t);
        std::unique_ptr<Buffer> inputUpload, feedbackUpload, inputs, feedback, readback;
        auto allocate = [&](uint64_t bytes, MemoryLocation memory, BufferUsageBits usage, auto& target) {
            requireWaveFeedback(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = memory})
                .transform([&](auto value) { target = std::move(value); }), "Buffer");
        };
        allocate(kInputBytes, MemoryLocation::HostUpload, BufferUsageBits::TransferSource, inputUpload);
        allocate(kFeedbackBytes, MemoryLocation::HostUpload, BufferUsageBits::TransferSource, feedbackUpload);
        allocate(kInputBytes, MemoryLocation::Device, BufferUsageBits::Storage | BufferUsageBits::TransferDestination, inputs);
        allocate(kFeedbackBytes, MemoryLocation::Device,
            BufferUsageBits::Storage | BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination, feedback);
        allocate(kFeedbackBytes, MemoryLocation::HostReadback, BufferUsageBits::TransferDestination, readback);
        auto write = [&](Buffer& buffer, const void* data, size_t bytes) {
            auto* mapped = buffer.map(); requireWaveFeedback(mapped != nullptr, "Upload map");
            std::memcpy(mapped, data, bytes); buffer.flush(); buffer.unmap();
        };
        auto inputUploadSlice = inputUpload->slice(), inputSlice = inputs->slice();
        auto feedbackUploadSlice = feedbackUpload->slice(), feedbackSlice = feedback->slice(), readbackSlice = readback->slice();
        requireWaveFeedback(inputUploadSlice, "Input upload slice"); requireWaveFeedback(inputSlice, "Input slice");
        requireWaveFeedback(feedbackUploadSlice, "Feedback upload slice"); requireWaveFeedback(feedbackSlice, "Feedback slice");
        requireWaveFeedback(readbackSlice, "Readback slice");
        std::filesystem::create_directories(context.outputDirectory);
        std::array<ComputeKernel, 2> kernels;
        constexpr std::array entries{"textureFeedbackWaveReference", "textureFeedbackWaveProduction"};
        for (uint32_t side = 0; side < 2; ++side) {
            std::string log;
            const SlangMacroDefine macros[] = {{"METALLIC_TEXTURE_FEEDBACK_WAVE", side ? "1" : "0"}};
            const char* capabilities[] = {"spvMaximalReconvergenceKHR"};
            const auto shader = compileSlangShaderToSpirv({.moduleName = "TextureFeedbackWaveProbe", .entryPointName = entries[side],
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .capabilities = side ? std::span<const char* const>(capabilities) : std::span<const char* const>{},
                .macroDefines = macros}, {.enableDiskCache = false}, log);
            requireWaveFeedback(shader, "Compile: " + log);
            requireWaveFeedback(kernels[side].initialize(*device, {.spirv = shader->spirv,
                .parameters = parameterAbi<TextureFeedbackWaveParams>(kWaveFeedbackABI, ParameterTransport::InlinePush)}, log), "Kernel: " + log);
            std::ofstream spirv(context.outputDirectory / (std::string(entries[side]) + ".spv"), std::ios::binary);
            spirv.write(reinterpret_cast<const char*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
        }
        std::unique_ptr<TimestampQueryPool> timestamps;
        if (benchmark) {
            requireWaveFeedback(device->createTimestampQueryPool(*queue, {.queryCount = 2})
                .transform([&](auto value) { timestamps = std::move(value); }), "Timestamps");
        }
        bench::Json samples = bench::Json::array();
        for (uint32_t pattern = 0; pattern < kPatternNames.size(); ++pattern) {
            const uint32_t lanes = pattern == 8 ? 37 : (benchmark ? kMaximumLanes : 4093);
            const uint32_t iterations = benchmark ? 8 : 3;
            std::vector<UInt4> inputData(kMaximumLanes);
            std::array<uint32_t, kTextureCount * 8> initial{};
            for (uint32_t texture = 0; texture < kTextureCount; ++texture) {
                for (uint32_t word = 0; word < 8; ++word) { initial[texture * 8 + word] = 0x12340000u + texture * 8 + word; }
                // Include records whose existing mip is finer than every request,
                // and nonzero counts, to catch overwrite rather than accumulation.
                initial[texture * 8 + 4] = texture % 5 == 4 ? 0 : UINT32_MAX;
                initial[texture * 8 + 5] = texture + 7;
            }
            auto expected = initial;
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                uint32_t texture = 0;
                if (pattern == 1) { texture = lane % 2; }
                if (pattern == 2 || pattern == 4 || pattern == 8) { texture = lane % 3; }
                if (pattern == 3) { texture = lane % 64; }
                if (pattern == 7) { texture = ((lane * 1664525u + 1013904223u) >> 16) % 11; }
                const bool active = pattern == 4 ? lane % 5 == 1 : pattern == 5 ? lane % 32 == 31 : pattern != 6;
                const uint32_t mip = (lane * 7 + lane / 32) & 15;
                inputData[lane] = {texture, mip, uint32_t(active), 0};
                if (active) {
                    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
                        expected[texture * 8 + 4] = std::min(expected[texture * 8 + 4], (mip + iteration) & 15);
                        ++expected[texture * 8 + 5];
                    }
                }
            }
            write(*inputUpload, inputData.data(), kInputBytes);
            write(*feedbackUpload, initial.data(), kFeedbackBytes);
            {
                bench::GPUCommands gpu(*queue); requireWaveFeedback(gpu.initialize(*device), "Input commands");
                gpu.commands->hostWriteBarrier();
                requireWaveFeedback(gpu.commands->copyBuffer(*inputUploadSlice, *inputSlice), "Upload input");
                const MemoryBarrierDesc barrier{.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
                requireWaveFeedback(gpu.commands->synchronize({.memory = {&barrier, 1}}), "Input visibility");
                requireWaveFeedback(gpu.submitAndWait(), "Input completion");
            }
            ParameterWriter writer(*device, registry);
            const TextureFeedbackWaveParams params{writer.bufferSpan<UInt4>(inputs.get()), writer.bufferSpan<uint32_t>(feedback.get()), lanes, iterations};
            auto encoded = writer.encode(params, kWaveFeedbackABI, ParameterTransport::InlinePush);
            requireWaveFeedback(encoded, "Parameters");
            std::array<PreparedComputeDispatch, 2> dispatches;
            for (uint32_t side = 0; side < 2; ++side) {
                auto prepared = kernels[side].prepareDispatch(*encoded, (lanes + 63) / 64);
                requireWaveFeedback(prepared, "Prepare dispatch"); dispatches[side] = std::move(*prepared);
            }
            constexpr std::array order{0u, 1u, 1u, 0u};
            for (uint32_t batch = 0; batch < (benchmark ? 5u : 1u); ++batch) {
                for (uint32_t position = 0; position < (benchmark ? 4u : 2u); ++position) {
                    const uint32_t side = order[position];
                    if (timestamps) { requireWaveFeedback(timestamps->reset(0, 2), "Reset timestamps"); }
                    bench::GPUCommands gpu(*queue); requireWaveFeedback(gpu.initialize(*device), "Dispatch commands");
                    gpu.commands->hostWriteBarrier();
                    const MemoryBarrierDesc reuse{.before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
                    requireWaveFeedback(gpu.commands->synchronize({.memory = {&reuse, 1}}), "Reset visibility");
                    requireWaveFeedback(gpu.commands->copyBuffer(*feedbackUploadSlice, *feedbackSlice), "Reset feedback");
                    const MemoryBarrierDesc compute{.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
                    requireWaveFeedback(gpu.commands->synchronize({.memory = {&compute, 1}}), "Feedback visibility");
                    if (timestamps) { requireWaveFeedback(gpu.commands->writeTimestamp(*timestamps, 0, PipelineStageBits::AllCommands), "Begin timestamp"); }
                    requireWaveFeedback(dispatches[side].record(*gpu.commands), "Dispatch");
                    if (timestamps) { requireWaveFeedback(gpu.commands->writeTimestamp(*timestamps, 1, PipelineStageBits::AllCommands), "End timestamp"); }
                    const MemoryBarrierDesc transfer{.before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    requireWaveFeedback(gpu.commands->synchronize({.memory = {&transfer, 1}}), "Readback visibility");
                    requireWaveFeedback(gpu.commands->copyBuffer(*feedbackSlice, *readbackSlice), "Readback copy");
                    requireWaveFeedback(gpu.submitAndWait(), "Completion");
                    readback->invalidate();
                    const auto* mapped = static_cast<const uint32_t*>(readback->map()); requireWaveFeedback(mapped != nullptr, "Readback map");
                    std::array<uint32_t, kTextureCount * 8> actual;
                    std::memcpy(actual.data(), mapped, kFeedbackBytes); readback->unmap();
                    for (uint32_t word = 0; word < actual.size(); ++word) {
                        requireWaveFeedback(actual[word] == expected[word], std::string(kPatternNames[pattern]) + " " + entries[side] +
                            " mismatch at word " + std::to_string(word) + ": " + std::to_string(actual[word]) + " != " + std::to_string(expected[word]));
                    }
                    if (timestamps) {
                        std::array<TimestampQueryResult, 2> times;
                        requireWaveFeedback(timestamps->readResults(0, times), "Read timestamps");
                        requireWaveFeedback(times[0].available && times[1].available, "Unavailable timestamps");
                        samples.push_back({{"pattern", kPatternNames[pattern]}, {"batch", batch}, {"order", position},
                            {"side", side ? "production" : "reference"}, {"warmup", batch == 0}, {"lanes", lanes},
                            {"iterations", iterations}, {"gpu_ms", timestamps->durationMilliseconds(times[0].value, times[1].value)}, {"correct", true}});
                    }
                }
            }
        }
        if (benchmark) {
            std::ofstream report(context.outputDirectory / "TextureFeedbackWaveTiming.json");
            report << bench::Json{{"scope", "isolated compute feedback dispatch; no texture sampling or ray queries"},
                {"validation", false}, {"warmup_dispatches_per_pattern", 4}, {"measured_abba_batches_per_pattern", 4},
                {"samples", samples}}.dump(2);
        }
        return RHITestResult::pass(benchmark ? "Nine patterns: exact readback for all ABBA dispatches; timings are microbenchmark evidence only" :
            "Nine patterns: per-lane and production min/hit counts match CPU oracle, including sparse/tail lanes and cross-wave contention");
    } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
}

class TextureFeedbackWaveTest final : public RHITest
{
public:
    TextureFeedbackWaveTest() { type = RHITestType::Rendering; name = "texture_feedback_wave"; }
    RHITestResult run(RHITestContext& context) override { return runWaveFeedback(context, false); }
};
METALLIC_REGISTER_RHI_TEST(TextureFeedbackWaveTest);

class TextureFeedbackWaveBenchmark final : public RHITest
{
public:
    TextureFeedbackWaveBenchmark() { type = RHITestType::Rendering; name = "texture_feedback_wave_benchmark"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto* enabled = std::getenv("METALLIC_RUN_TEXTURE_FEEDBACK_WAVE_BENCHMARK");
        if (!enabled || std::string_view(enabled) != "1" || context.enableValidation) {
            return RHITestResult::skip("Manual benchmark: set METALLIC_RUN_TEXTURE_FEEDBACK_WAVE_BENCHMARK=1 and disable validation");
        }
        return runWaveFeedback(context, true);
    }
};
METALLIC_REGISTER_RHI_TEST(TextureFeedbackWaveBenchmark);
} // namespace
} // namespace metallic::tests
