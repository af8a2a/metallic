#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;
constexpr uint64_t kABI = 0x4f50425246524501ull;
// Predeclared before running the candidate; critical-angle samples use the same
// bound. Oracle error is reported separately because the old float formula is
// itself ill-conditioned near eta=1 and the TIR boundary.
constexpr double kAbsoluteTolerance = 3e-5;
constexpr double kRelativeTolerance = 3e-5;
constexpr uint32_t kRowsPerSample = 5;
struct FresnelProbeParams { GPUBufferSpan inputs, output; };
static_assert(sizeof(FresnelProbeParams) == 24);

void check(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}
template<typename T> void check(const Result<T>& value, const std::string& message)
{
    check(bool(value), message);
}

enum class SampleGroup { Grid, NearUnity, CriticalAngle, HostRange, Extreme, Count };
constexpr std::array kGroupNames{"grid", "near-unity", "critical-angle", "host-range", "extreme"};
struct Sample { Float4 input; SampleGroup group; };

// Independent double oracle using Snell's law and polarization amplitudes.
// Inputs are the exact uploaded floats, not the original unrounded generators.
double fresnelOracle(double eta, double cosine)
{
    if (eta == 1.0) { return 0.0; }
    const double sinT2 = ((1.0 - cosine) * (1.0 + cosine)) / (eta * eta);
    if (sinT2 >= 1.0) { return 1.0; }
    const double transmitted = std::sqrt(1.0 - sinT2);
    const double rs = (cosine - eta * transmitted) / (cosine + eta * transmitted);
    const double rp = (eta * cosine - transmitted) / (eta * cosine + transmitted);
    return (rs * rs + rp * rp) * 0.5;
}

std::vector<Sample> makeSamples()
{
    std::vector<Sample> samples;
    const auto add = [&](float eta, float cosine, SampleGroup group) {
        samples.push_back({{eta, eta, eta, cosine}, group});
    };
    std::vector<float> etas{0.01f, 0.1f, 1.0f / 3, 0.5f, 1.0f / 1.5f, 0.9f, 1.0f, 1.1f, 1.33f, 1.5f, 2, 3, 10, 100};
    for (uint32_t i = 0; i <= 96; ++i) { etas.push_back(float(std::pow(10.0, -2.0 + 4.0 * i / 96))); }
    for (float eta : etas) {
        for (uint32_t i = 0; i <= 128; ++i) { add(eta, float(i) / 128, SampleGroup::Grid); }
        for (uint32_t i = 0; i <= 32; ++i) { add(eta, float(std::pow(10.0, -8.0 + 7.0 * i / 32)), SampleGroup::Grid); }
    }
    // Both adjacent representable IORs and a wider logarithmic grazing sweep.
    for (int direction : {-1, 1}) {
        float eta = 1;
        for (uint32_t ulps = 0; ulps <= 256; ++ulps) {
            if (ulps == 0 || ulps == 1 || ulps == 2 || ulps == 4 || ulps == 16 || ulps == 64 || ulps == 256) {
                etas.push_back(eta);
                for (uint32_t i = 0; i <= 128; ++i) { add(eta, float(std::pow(10.0, -8.0 + 8.0 * i / 128)), SampleGroup::NearUnity); }
                add(eta, 0, SampleGroup::NearUnity);
            }
            eta = std::nextafter(eta, direction < 0 ? 0.0f : 2.0f);
        }
    }
    for (float eta : etas) {
        if (eta >= 1) { continue; }
        const float critical = float(std::sqrt((1.0 - double(eta)) * (1.0 + double(eta))));
        for (int direction : {-1, 1}) {
            float cosine = critical;
            for (uint32_t ulps = 0; ulps <= 32; ++ulps) {
                if (ulps <= 4 || ulps == 8 || ulps == 16 || ulps == 32) { add(eta, cosine, SampleGroup::CriticalAngle); }
                cosine = std::nextafter(cosine, direction < 0 ? 0.0f : 1.0f);
            }
        }
    }
    // glTF import clamps IOR to [1,3], but Scene::validMaterialProperties accepts
    // any finite IOR >=1. Specular-weight conversion caps F0 at 0.9999, yielding
    // an effective IOR around 39991 and its inside-to-outside reciprocal.
    const float rootF0 = std::sqrt(0.9999f);
    const float weightedMaximum = (1.0f + rootF0) / (1.0f - rootF0);
    for (float eta : {1.0f, 1.33f, 1.5f, 3.0f, 100.0f, 1000.0f, 10000.0f, weightedMaximum}) {
        for (uint32_t i = 0; i <= 128; ++i) {
            add(eta, float(i) / 128, SampleGroup::HostRange);
            add(1.0f / eta, float(i) / 128, SampleGroup::HostRange);
        }
    }
    // Preserve the old classification even where squaring eta under/overflows.
    // A finite candidate result is not automatically accepted over a legacy NaN.
    const float maximum = std::numeric_limits<float>::max();
    for (float eta : {1e10f, 1e20f, std::nextafter(maximum, 0.0f), maximum}) {
        for (float cosine : {0.0f, 1e-8f, 1e-4f, 0.1f, 0.5f, std::nextafter(1.0f, 0.0f), 1.0f}) {
            add(eta, cosine, SampleGroup::Extreme);
            add(1.0f / eta, cosine, SampleGroup::Extreme);
        }
    }
    // RGB channels deliberately straddle index matching and TIR. Dispersion-off
    // must broadcast red; it must not average or select the green channel.
    for (Float4 input : {Float4{1.48f, 1.5f, 1.53f, 0}, Float4{0.65f, 1.0f, 1.55f, 0},
            Float4{std::nextafter(1.0f, 0.0f), 1.0f, std::nextafter(1.0f, 2.0f), 0}}) {
        for (uint32_t i = 0; i <= 256; ++i) {
            input[3] = float(i) / 256;
            samples.push_back({input, SampleGroup::HostRange});
        }
    }
    return samples;
}

struct ErrorStats
{
    uint64_t compared = 0, nonfinite = 0, nan = 0, infinity = 0, indexMatched = 0, tir = 0;
    uint64_t candidateOutsideRange = 0, referenceOutsideRange = 0;
    double maxDifference = 0, maxCandidateOracle = 0, maxReferenceOracle = 0;
    uint32_t worstDifferenceSample = 0, worstCandidateSample = 0, worstReferenceSample = 0;
};

class OpenPBRFresnelTest final : public RHITest
{
public:
    OpenPBRFresnelTest() { type = RHITestType::Rendering; name = "material_openpbr_fresnel_numerics"; }

    RHITestResult run(RHITestContext& context) override
    {
        try {
            const auto samples = makeSamples();
            std::vector<Float4> inputs;
            for (const auto& sample : samples) { inputs.push_back(sample.input); }
            bench::TestDevice device;
            check(bench::createTestDevice(context, {.applicationName = "OpenPBR Fresnel numerics",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { device = std::move(value); }), "Create device");
            ResourceRegistry registry;
            check(registry.initialize(*device), "Registry");
            std::unique_ptr<Buffer> input, output;
            check(device->createBuffer({.size = inputs.size() * sizeof(Float4), .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto value) { input = std::move(value); }), "Input allocation");
            check(device->createBuffer({.size = inputs.size() * kRowsPerSample * sizeof(Float4), .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { output = std::move(value); }), "Output allocation");
            auto* upload = input->map(); check(upload != nullptr, "Input mapping");
            std::memcpy(upload, inputs.data(), inputs.size() * sizeof(Float4)); input->flush(); input->unmap();
            std::string log;
            const auto shader = compileSlangShaderToSpirv({.moduleName = "OpenPBRFresnelProbe", .entryPointName = "openPBRFresnelMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, {.enableDiskCache = false}, log);
            check(shader, "Compile: " + log);
            ComputeKernel kernel;
            check(kernel.initialize(*device, {.spirv = shader->spirv,
                .parameters = parameterAbi<FresnelProbeParams>(kABI, ParameterTransport::InlinePush)}, log), "Kernel: " + log);
            ParameterWriter writer(*device, registry);
            const FresnelProbeParams params{writer.bufferSpan<Float4>(input.get()), writer.bufferSpan<Float4>(output.get())};
            auto encoded = writer.encode(params, kABI, ParameterTransport::InlinePush); check(encoded, "Parameters");
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            check(gpu.initialize(*device), "Commands");
            gpu.commands->hostWriteBarrier();
            check(kernel.dispatch(*gpu.commands, *encoded, uint32_t((samples.size() + 63) / 64)), "Dispatch");
            check(gpu.submitAndWait(), "Completion");
            output->invalidate();
            const auto* mapped = static_cast<const Float4*>(output->map()); check(mapped != nullptr, "Readback mapping");
            const std::vector<Float4> readback(mapped, mapped + samples.size() * kRowsPerSample);
            output->unmap();
            bench::readbackEvidence(context, "fresnel-inputs.bin", std::span<const Float4>(inputs));
            bench::readbackEvidence(context, "fresnel-readback.bin", std::span<const Float4>(readback));
            std::array<ErrorStats, size_t(SampleGroup::Count)> stats{};
            uint64_t failures = 0;
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream failuresFile(context.outputDirectory / "FresnelFailures.csv");
            failuresFile << "sample,group,operation,channel,eta,cosine,candidate,reference,oracle,reason\n" << std::setprecision(17);
            for (uint32_t index = 0; index < samples.size(); ++index) {
                const auto& sample = samples[index];
                auto& stat = stats[size_t(sample.group)];
                const auto& flags = readback[index * kRowsPerSample + 4];
                check(flags[3] == 1, "Missing Fresnel lane " + std::to_string(index));
                for (uint32_t operation = 0; operation < 2; ++operation) {
                    const auto& candidate = readback[index * kRowsPerSample + operation * 2];
                    const auto& reference = readback[index * kRowsPerSample + operation * 2 + 1];
                    for (uint32_t channel = 0; channel < 4; ++channel) {
                        const uint32_t etaChannel = operation == 0 && channel < 3 ? channel : 0;
                        const float eta = sample.input[etaChannel];
                        const double single = fresnelOracle(eta, sample.input[3]);
                        const double oracle = operation == 0 && channel == 3 ? 2 * single / (1 + single) : single;
                        const float actual = candidate[channel], expected = reference[channel];
                        std::string reason;
                        ++stat.compared;
                        if (!std::isfinite(actual) || !std::isfinite(expected)) {
                            ++stat.nonfinite;
                            if (std::isnan(actual) || std::isnan(expected)) { ++stat.nan; }
                            if (std::isinf(actual) || std::isinf(expected)) { ++stat.infinity; }
                            const bool same = (std::isnan(actual) && std::isnan(expected)) ||
                                (std::isinf(actual) && std::isinf(expected) && std::signbit(actual) == std::signbit(expected));
                            if (!same) { reason = "nonfinite-classification"; }
                        } else {
                            const double difference = std::abs(double(actual) - expected);
                            const double candidateError = std::abs(double(actual) - oracle);
                            const double referenceError = std::abs(double(expected) - oracle);
                            if (difference > stat.maxDifference) { stat.maxDifference = difference; stat.worstDifferenceSample = index; }
                            if (candidateError > stat.maxCandidateOracle) { stat.maxCandidateOracle = candidateError; stat.worstCandidateSample = index; }
                            if (referenceError > stat.maxReferenceOracle) { stat.maxReferenceOracle = referenceError; stat.worstReferenceSample = index; }
                            if (difference > kAbsoluteTolerance + kRelativeTolerance * std::abs(expected)) { reason = "legacy-difference"; }
                            if (eta == 1 || flags[etaChannel] == 1) {
                                if (eta == 1) { ++stat.indexMatched; }
                                else { ++stat.tir; }
                                if (actual != (eta == 1 ? 0.0f : 1.0f)) { reason = "index-match-or-TIR"; }
                            }
                            if (actual < 0 || actual > 1) { ++stat.candidateOutsideRange; }
                            if (expected < 0 || expected > 1) { ++stat.referenceOutsideRange; }
                            // The legacy GPU division can overshoot 1 by one ULP
                            // at grazing incidence. Record it and reject any new
                            // or larger excursion without widening the error bound.
                            if (actual < std::min(0.0f, expected) || actual > std::max(1.0f, expected)) {
                                reason = "reflectance-range-regression";
                            }
                            if (candidateError > referenceError + kAbsoluteTolerance + kRelativeTolerance * std::abs(oracle)) {
                                reason = "oracle-regression";
                            }
                        }
                        if (!reason.empty()) {
                            ++failures;
                            if (failures <= 128) {
                                failuresFile << index << ',' << kGroupNames[size_t(sample.group)] << ',' << operation << ',' << channel << ','
                                    << eta << ',' << sample.input[3] << ',' << actual << ',' << expected << ',' << oracle << ',' << reason << '\n';
                            }
                        }
                    }
                }
            }
            std::ofstream summary(context.outputDirectory / "FresnelErrors.csv");
            summary << "group,components,nonfinite,nan,infinity,index_matched,tir,max_candidate_reference,max_candidate_oracle,max_reference_oracle,worst_difference_sample,worst_candidate_sample,worst_reference_sample,candidate_outside_range,reference_outside_range\n"
                    << std::setprecision(17);
            for (size_t group = 0; group < stats.size(); ++group) {
                const auto& stat = stats[group];
                summary << kGroupNames[group] << ',' << stat.compared << ',' << stat.nonfinite << ',' << stat.nan << ',' << stat.infinity << ','
                    << stat.indexMatched << ',' << stat.tir << ','
                    << stat.maxDifference << ',' << stat.maxCandidateOracle << ',' << stat.maxReferenceOracle << ','
                    << stat.worstDifferenceSample << ',' << stat.worstCandidateSample << ',' << stat.worstReferenceSample << ','
                    << stat.candidateOutsideRange << ',' << stat.referenceOutsideRange << '\n';
            }
            check(failures == 0, std::to_string(failures) + " Fresnel numerical failures; see FresnelFailures.csv and FresnelErrors.csv");
            return RHITestResult::pass(std::to_string(samples.size()) + " Fresnel cases: frozen formula, double oracle, exact index-match/TIR, RGB dispersion and thin wall");
        } catch (const std::exception& failure) { return RHITestResult::fail(failure.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(OpenPBRFresnelTest);

class OpenPBRFresnelBenchmark final : public RHITest
{
public:
    OpenPBRFresnelBenchmark() { type = RHITestType::Rendering; name = "material_openpbr_fresnel_benchmark"; }

    RHITestResult run(RHITestContext& context) override
    {
        const auto* enabled = std::getenv("METALLIC_RUN_FRESNEL_BENCHMARK");
        if (!enabled || std::string_view(enabled) != "1") {
            return RHITestResult::skip("Manual benchmark: set METALLIC_RUN_FRESNEL_BENCHMARK=1 and disable validation");
        }
        if (context.enableValidation) { return RHITestResult::skip("Fresnel timing requires validation disabled"); }
        try {
            constexpr uint32_t lanes = 65536;
            std::vector<Float4> inputs(lanes);
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                const float eta = 1.05f + float(lane % 1024) * (1.95f / 1023);
                inputs[lane] = {lane % 4 == 0 ? 1.0f / eta : eta, 0, 0, float(lane % 257) / 257};
            }
            bench::TestDevice device;
            check(bench::createTestDevice(context, {.applicationName = "OpenPBR Fresnel manual benchmark",
                .enableValidation = false, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { device = std::move(value); }), "Create benchmark device");
            ResourceRegistry registry; check(registry.initialize(*device), "Benchmark registry");
            std::unique_ptr<Buffer> upload, input, output, readback;
            const uint64_t bytes = lanes * sizeof(Float4);
            check(device->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto value) { upload = std::move(value); }), "Benchmark upload");
            check(device->createBuffer({.size = bytes, .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::Device}).transform([&](auto value) { input = std::move(value); }), "Benchmark input");
            check(device->createBuffer({.size = bytes, .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::Device}).transform([&](auto value) { output = std::move(value); }), "Benchmark output");
            check(device->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { readback = std::move(value); }), "Benchmark readback");
            auto* mapped = upload->map(); check(mapped != nullptr, "Benchmark upload mapping");
            std::memcpy(mapped, inputs.data(), bytes); upload->flush(); upload->unmap();
            auto uploadSlice = upload->slice(), inputSlice = input->slice(), outputSlice = output->slice(), readbackSlice = readback->slice();
            check(uploadSlice, "Upload slice"); check(inputSlice, "Input slice"); check(outputSlice, "Output slice"); check(readbackSlice, "Readback slice");
            auto* queue = device->getQueue(QueueType::Graphics);
            {
                bench::GPUCommands gpu(*queue); check(gpu.initialize(*device), "Upload commands");
                gpu.commands->hostWriteBarrier();
                check(gpu.commands->copyBuffer(*uploadSlice, *inputSlice), "Input upload");
                const MemoryBarrierDesc barrier{.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
                check(gpu.commands->synchronize({.memory = {&barrier, 1}}), "Upload visibility");
                check(gpu.submitAndWait(), "Upload completion");
            }
            std::array<ComputeKernel, 2> kernels;
            const std::array entries{"openPBRFresnelBenchmarkOriginal", "openPBRFresnelBenchmarkProduction"};
            std::filesystem::create_directories(context.outputDirectory);
            for (uint32_t side = 0; side < 2; ++side) {
                std::string log;
                const auto shader = compileSlangShaderToSpirv({.moduleName = "OpenPBRFresnelProbe", .entryPointName = entries[side],
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, {.enableDiskCache = false}, log);
                check(shader, "Benchmark compile: " + log);
                std::ofstream spirv(context.outputDirectory / (std::string(entries[side]) + ".spv"), std::ios::binary);
                spirv.write(reinterpret_cast<const char*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
                check(kernels[side].initialize(*device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<FresnelProbeParams>(kABI, ParameterTransport::InlinePush)}, log), "Benchmark kernel: " + log);
            }
            ParameterWriter writer(*device, registry);
            const FresnelProbeParams params{writer.bufferSpan<Float4>(input.get()), writer.bufferSpan<Float4>(output.get())};
            auto encoded = writer.encode(params, kABI, ParameterTransport::InlinePush); check(encoded, "Benchmark parameters");
            std::array<PreparedComputeDispatch, 2> dispatches;
            for (uint32_t side = 0; side < 2; ++side) {
                auto prepared = kernels[side].prepareDispatch(*encoded, lanes / 64); check(prepared, "Benchmark dispatch preparation");
                dispatches[side] = std::move(*prepared);
            }
            std::unique_ptr<TimestampQueryPool> timestamps;
            check(device->createTimestampQueryPool(*queue, {.queryCount = 2})
                .transform([&](auto value) { timestamps = std::move(value); }), "Benchmark timestamps");
            std::ofstream timings(context.outputDirectory / "FresnelTiming.csv");
            timings << "batch,order,side,warmup,validation,lanes,iterations,gpu_ms\n" << std::setprecision(17);
            std::array<std::vector<Float4>, 2> results;
            constexpr std::array order{0u, 1u, 1u, 0u};
            for (uint32_t batch = 0; batch < 5; ++batch) {
                for (uint32_t position = 0; position < order.size(); ++position) {
                    const uint32_t side = order[position];
                    check(timestamps->reset(0, 2), "Reset benchmark timestamps");
                    bench::GPUCommands gpu(*queue); check(gpu.initialize(*device), "Benchmark commands");
                    const MemoryBarrierDesc reuse{.before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
                    check(gpu.commands->synchronize({.memory = {&reuse, 1}}), "Output reuse");
                    check(gpu.commands->writeTimestamp(*timestamps, 0, PipelineStageBits::AllCommands), "Begin benchmark timestamp");
                    check(dispatches[side].record(*gpu.commands), "Benchmark dispatch");
                    check(gpu.commands->writeTimestamp(*timestamps, 1, PipelineStageBits::AllCommands), "End benchmark timestamp");
                    if (batch == 4) {
                        const MemoryBarrierDesc transfer{.before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                        check(gpu.commands->synchronize({.memory = {&transfer, 1}}), "Benchmark readback visibility");
                        check(gpu.commands->copyBuffer(*outputSlice, *readbackSlice), "Benchmark readback copy");
                    }
                    check(gpu.submitAndWait(), "Benchmark completion");
                    std::array<TimestampQueryResult, 2> times;
                    check(timestamps->readResults(0, times), "Read benchmark timestamps");
                    check(times[0].available && times[1].available, "Unavailable benchmark timestamps");
                    timings << batch << ',' << position << ',' << (side ? "production" : "original") << ',' << (batch == 0)
                            << ",0," << lanes << ",128," << timestamps->durationMilliseconds(times[0].value, times[1].value) << '\n';
                    if (batch == 4) {
                        readback->invalidate();
                        const auto* values = static_cast<const Float4*>(readback->map()); check(values != nullptr, "Benchmark readback mapping");
                        results[side].assign(values, values + lanes); readback->unmap();
                    }
                }
            }
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                const auto& original = results[0][lane]; const auto& production = results[1][lane];
                check(std::isfinite(production[0]) && original[3] == 1 && production[3] == 1 &&
                    std::abs(double(production[0]) - original[0]) <= 128 * kAbsoluteTolerance + kRelativeTolerance * std::abs(original[0]),
                    "Benchmark checksum mismatch at lane " + std::to_string(lane));
            }
            bench::readbackEvidence(context, "fresnel-benchmark-original.bin", std::span<const Float4>(results[0]));
            bench::readbackEvidence(context, "fresnel-benchmark-production.bin", std::span<const Float4>(results[1]));
            return RHITestResult::pass("Manual function timing: 4 warmups + 4 ABBA batches; raw dispatch timings are correlated within one process, not production frame evidence");
        } catch (const std::exception& failure) { return RHITestResult::fail(failure.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(OpenPBRFresnelBenchmark);
} // namespace
} // namespace metallic::tests
