#include "RHITest.h"
#include "TestResourceParameters.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "harness/Fixtures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace metallic::tests {
namespace {

double referenceMisPowerHeuristic(float pdfA, float pdfB)
{
    const double a = std::max(double(pdfA), 0.0);
    const double b = std::max(double(pdfB), 0.0);
    const double sum = a * a + b * b;
    return sum > 0.0 ? a * a / sum : 1.0;
}

class MisPowerHeuristicGPUTest final : public RHITest {
public:
    MisPowerHeuristicGPUTest()
    {
        type = RHITestType::Resource;
        name = "path_trace_mis_power_heuristic_finite_pdfs";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        using Float4 = std::array<float, 4>;
        constexpr float kMaxFloat = std::numeric_limits<float>::max();
        constexpr std::array<Float4, 16> kCases{{
            {1.0f, 2.0f, 0, 0},
            {2.0f, 1.0f, 0, 0},
            {1e20f, 2e20f, 0, 0},
            {1e30f, 2e30f, 0, 0},
            {1e-20f, 2e-20f, 0, 0},
            {1e-30f, 2e-30f, 0, 0},
            {kMaxFloat, kMaxFloat, 0, 0},
            {kMaxFloat, kMaxFloat * 0.5f, 0, 0},
            {1e30f, 1e-30f, 0, 0},
            {0, 0, 0, 0},
            {2.0f, 0, 0, 0},
            {0, 2.0f, 0, 0},
            {-1.0f, 2.0f, 0, 0},
            {-1.0f, -2.0f, 0, 0},
            // First failing environment misses traced in M06 glass and S05 transmission.
            {2.1279323521372127e19f, 0.1219605877995491f, 0, 0},
            {2.445145195090123e19f, 0.12511590123176575f, 0, 0},
        }};
        bench::TestDevice device;
        auto result = bench::createTestDevice(context, {.applicationName = "Path trace MIS math probe",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptors"); }
        if (!result) { return RHITestResult::fail("MIS probe device creation failed"); }
        std::unique_ptr<Buffer> inputs, output;
        result = device->createBuffer({.size = sizeof(kCases), .structureStride = sizeof(Float4),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload})
            .transform([&](auto value) { inputs = std::move(value); });
        if (!result) { return RHITestResult::fail("MIS probe input allocation failed"); }
        result = device->createBuffer({.size = kCases.size() * 2 * sizeof(Float4), .structureStride = sizeof(Float4),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
            .transform([&](auto value) { output = std::move(value); });
        if (!result) { return RHITestResult::fail("MIS probe output allocation failed"); }
        auto* mapped = inputs->map();
        if (!mapped) { return RHITestResult::fail("MIS probe input mapping failed"); }
        std::memcpy(mapped, kCases.data(), sizeof(kCases));
        inputs->flush();
        inputs->unmap();

        std::string log;
        auto shader = compileSlangShaderToSpirv({.moduleName = "MisPowerHeuristicProbe",
            .entryPointName = "misPowerHeuristicProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
        if (!shader) { return RHITestResult::fail("MIS probe shader compile failed: " + log); }
        const ComputeResourceBindingDesc layout[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, input), .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, output), .kind = ComputeResourceBindingKind::StorageBuffer},
        };
        ComputeProgram program;
        if (!program.initialize(*device, {.spirv = shader->spirv, .bindings = layout,
            .requiresRayQuery = false, .resourceParameterSize = sizeof(metallic::tests::FrameCopyProbeResources)}, log)) {
            return RHITestResult::fail("MIS probe program initialization failed: " + log);
        }
        bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
        if (!gpu.initialize(*device)) { return RHITestResult::fail("MIS probe command initialization failed"); }
        const ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, input), .buffer = inputs.get()},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, output), .buffer = output.get()},
        };
        if (!program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = bindings,
            .groupCountX = uint32_t(kCases.size())}) || !gpu.submitAndWait()) {
            return RHITestResult::fail("MIS probe dispatch/wait failed");
        }
        output->invalidate();
        mapped = output->map();
        if (!mapped) { return RHITestResult::fail("MIS probe output mapping failed"); }
        std::array<Float4, kCases.size() * 2> actual;
        std::memcpy(actual.data(), mapped, sizeof(actual));
        output->unmap();
        for (size_t index = 0; index < kCases.size(); ++index) {
            const double forward = referenceMisPowerHeuristic(kCases[index][0], kCases[index][1]);
            const double reverse = referenceMisPowerHeuristic(kCases[index][1], kCases[index][0]);
            const std::array<double, 4> expected{forward, reverse, forward, reverse};
            for (size_t channel = 0; channel < 4; ++channel) {
                const float value = actual[index * 2][channel];
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f ||
                    std::abs(double(value) - expected[channel]) > 2e-6) {
                    return RHITestResult::fail("MIS finite-PDF mismatch at case " + std::to_string(index) +
                        ", channel " + std::to_string(channel) + ": actual=" + std::to_string(value) +
                        ", expected=" + std::to_string(expected[channel]));
                }
                if (actual[index * 2 + 1][channel] != 1.0f) {
                    return RHITestResult::fail("MIS delta/procedural/disabled bypass changed at case " +
                        std::to_string(index) + ", channel " + std::to_string(channel));
                }
            }
        }
        return RHITestResult::pass("Production MIS helper and environment wrappers match double reference for 16 finite PDF pairs");
    }
};

METALLIC_REGISTER_RHI_TEST(MisPowerHeuristicGPUTest);

} // namespace
} // namespace metallic::tests
