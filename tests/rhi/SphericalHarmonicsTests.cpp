#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <cmath>
#include <cstring>
#include <numbers>

namespace metallic::tests {
namespace {

using ProbeValue = std::array<float, 4>;
constexpr size_t kOutputCount = 41;
constexpr double kPi = std::numbers::pi;

std::array<ProbeValue, 14> probeInput()
{
    std::array<ProbeValue, 14> values{};
    values[0] = {0.36f, 0.48f, 0.8f, 0.0f};
    values[1] = {0.0f, 0.0f, 1.0f, 0.0f};
    values[2] = {1.25f, 0.5f, 2.0f, 0.0f};
    values[3] = {2.0f, 0.75f, 0.125f, 0.0f};
    // Deliberately nonzero buffer offset and padding. Stored environment data
    // already represents irradiance, not radiance awaiting cosine convolution.
    values[4] = {-123.0f, -456.0f, -789.0f, -1.0f};
    for (size_t c = 0; c < 3; ++c) {
        values[5][c] = static_cast<float>(values[3][c] * kPi * std::sqrt(4.0 * kPi));
    }
    for (size_t i = 5; i < values.size(); ++i) { values[i][3] = static_cast<float>(100 + i); }
    return values;
}

class SphericalHarmonicsProbePass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(kOutputCount * sizeof(ProbeValue), sizeof(ProbeValue)).storageReadWrite();
        return reflection;
    }

    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        const auto values = probeInput();
        auto result = context.device->createBuffer({.size = sizeof(values), .structureStride = sizeof(ProbeValue),
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostUpload}, input_);
        if (!result) { return result; }
        void* mapped = input_->map();
        if (mapped == nullptr) { log = "SH fixture input map failed"; return render::makeError(render::Error::Failure); }
        std::memcpy(mapped, values.data(), sizeof(values));
        input_->flush();
        input_->unmap();

        render::ShaderCompileResult shader;
        result = render::compileSlangShaderToSpirv({.moduleName = "SphericalHarmonicsProbe",
            .entryPointName = "sphericalHarmonicsProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .bindings = bindings,
            .bindingCount = 2, .requiresRayQuery = false}, log);
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .buffer = context.outputBuffer("data").buffer()},
            {.binding = 1, .buffer = input_.get()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings, .bindingCount = 2});
    }

private:
    std::unique_ptr<render::Buffer> input_;
    render::ComputeProgram program_;
};

class SphericalHarmonicsMathTest final : public RhiTest {
public:
    SphericalHarmonicsMathTest() { type = RhiTestType::Rendering; name = "spherical_harmonics_math_and_packing"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        // Validate fp16 generic specialization without requiring fp16 execution
        // support from the test device. Runtime readback below uses fp32.
        render::ShaderCompileResult halfShader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "SphericalHarmonicsProbe",
            .entryPointName = "sphericalHarmonicsHalfCompileMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, halfShader);
        if (!result) { return RhiTestResult::fail(halfShader.diagnostics); }

        std::unique_ptr<render::Device> device;
        result = render::createDevice({.applicationName = "SH math probe", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("requires bindless descriptors"); }
        if (!result) { return RhiTestResult::fail("SH probe device creation failed"); }
        render::registerRenderGraphPassType("SphericalHarmonicsProbePass", "SH math probe",
            [] { return std::make_unique<SphericalHarmonicsProbePass>(); });
        render::RenderGraph graph;
        graph.addNode("SphericalHarmonicsProbePass", "Probe");
        graph.markOutput("Probe.data");
        render::RenderGraphExecutor executor;
        std::string log;
        result = executor.compile(*device, graph, 1, 1, log);
        if (!result) { return RhiTestResult::fail(log); }
        result = executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)});
        if (!result) { return RhiTestResult::fail("SH probe dispatch failed: " + std::string(toString(result))); }
        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) { return RhiTestResult::fail("SH probe wait failed"); }
        auto* buffer = executor.outputResource("Probe.data")->buffer;
        buffer->invalidate();
        void* mapped = buffer->map();
        if (mapped == nullptr) { return RhiTestResult::fail("SH probe readback failed"); }
        std::array<ProbeValue, kOutputCount> values;
        std::memcpy(values.data(), mapped, sizeof(values));
        buffer->unmap();

        const auto input = probeInput();
        std::array<ProbeValue, kOutputCount> expected{};
        const double x = input[0][0], y = input[0][1], z = input[0][2];
        const std::array<double, 9> basis{
            std::sqrt(1.0 / (4.0 * kPi)), std::sqrt(3.0 / (4.0 * kPi)) * y,
            std::sqrt(3.0 / (4.0 * kPi)) * z, std::sqrt(3.0 / (4.0 * kPi)) * x,
            std::sqrt(15.0 / (4.0 * kPi)) * x * y, std::sqrt(15.0 / (4.0 * kPi)) * y * z,
            std::sqrt(5.0 / (16.0 * kPi)) * (3.0 * z * z - 1.0),
            std::sqrt(15.0 / (4.0 * kPi)) * x * z, std::sqrt(15.0 / (16.0 * kPi)) * (x * x - y * y)};
        // Independent analytic SH addition theorem and diffuse convolution,
        // not a second invocation of the shader's coefficient loop.
        const double cosine = z;
        const double p2 = 0.5 * (3.0 * cosine * cosine - 1.0);
        const double l1Kernel = (1.0 + 3.0 * cosine) / (4.0 * kPi);
        const double l2Kernel = (1.0 + 3.0 * cosine + 5.0 * p2) / (4.0 * kPi);
        const double irradianceKernel = 0.25 + 0.5 * cosine + 5.0 / 16.0 * p2;
        for (size_t c = 0; c < 3; ++c) {
            const double radiance = input[2][c];
            const double constant = input[3][c];
            for (size_t i = 0; i < basis.size(); ++i) { expected[i][c] = static_cast<float>(radiance * basis[i]); }
            expected[9][c] = expected[14][c] = static_cast<float>(radiance * l2Kernel);
            expected[10][c] = static_cast<float>(radiance * irradianceKernel);
            expected[12][c] = static_cast<float>(radiance * (0.25 + 0.5 * cosine));
            expected[13][c] = static_cast<float>(radiance * radiance * 9.0 / (4.0 * kPi));
            expected[15][c] = static_cast<float>(0.7 * radiance * l2Kernel + 0.3 * constant * 9.0 / (4.0 * kPi));
            expected[16][c] = expected[17][c] = expected[18][c] = static_cast<float>(constant * kPi);
            expected[19][c] = static_cast<float>(constant);
            expected[23][c] = static_cast<float>(radiance * (1.0 + 6.0 * cosine + 15.0 * p2) / (4.0 * kPi));
            expected[24][c] = static_cast<float>(l1Kernel);
            for (size_t i = 0; i < 9; ++i) { expected[32 + i][c] = input[5 + i][c]; }
        }
        expected[11][0] = static_cast<float>(l1Kernel);
        expected[26][0] = static_cast<float>((1.0 + 6.0 * cosine) / (4.0 * kPi));
        for (size_t i = 0; i < values.size(); ++i) {
            if (i >= 27 && i < 32) { continue; } // Unwritten padding between result groups.
            for (size_t c = 0; c < 4; ++c) {
                if (!std::isfinite(values[i][c]) || std::abs(values[i][c] - expected[i][c]) > 0.00002f) {
                    return RhiTestResult::fail("SH result " + std::to_string(i) + ", channel " + std::to_string(c) +
                        ": expected " + std::to_string(expected[i][c]) + ", got " + std::to_string(values[i][c]));
                }
            }
        }
        return RhiTestResult::pass("L1/L2 projection, operators, interpolation, rotation, irradiance units, fp16 compilation and float4 packing");
    }
};

METALLIC_REGISTER_RHI_TEST(SphericalHarmonicsMathTest);

} // namespace
} // namespace metallic::tests
