#include "TestResourceParameters.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <array>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

constexpr uint32_t kMotionCaseCount = 32;
constexpr uint32_t kMotionResultCount = kMotionCaseCount * 4;

class DLSSMotionVectorProbePass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("motion").buffer(kMotionResultCount * 16, 16).storageReadWrite();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        const char* capabilities[] = {"spvRayQueryKHR"};
        auto result = render::compileSlangShaderToSpirv({
            .moduleName = "DLSSMotionVectorProbe",
            .entryPointName = "dlssMotionVectorProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .capabilities = {capabilities, 1},
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc binding{
            .binding = METALLIC_RESOURCE_MEMBER(metallic::tests::DLSSMotionVectorProbeResources, motionResults), .kind = render::ComputeResourceBindingKind::StorageBuffer};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {&binding, 1},
            .requiresRayQuery = false,
            .resourceParameterSize = sizeof(metallic::tests::DLSSMotionVectorProbeResources),
        }, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding binding{
            .binding = METALLIC_RESOURCE_MEMBER(metallic::tests::DLSSMotionVectorProbeResources, motionResults), .buffer = context.outputBuffer("motion").buffer()};
        return program_.dispatch({
            .commandBuffer = &context.commandBuffer(),
            .bindings = {&binding, 1},
        });
    }

private:
    render::ComputeProgram program_;
};

class DLSSMotionVectorReprojectionTest final : public RHITest {
public:
    DLSSMotionVectorReprojectionTest()
    {
        type = RHITestType::Rendering;
        name = "dlss_motion_vector_reprojection";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::registerRenderGraphPassType("DLSSMotionVectorProbe", "DLSS motion probe",
            [] { return std::make_unique<DLSSMotionVectorProbePass>(); });
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "DLSS motion vectors",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(initialized, render::Error::Unsupported)) {
            return RHITestResult::skip("Requires bindless descriptors");
        }
        if (!initialized) { return RHITestResult::fail("Device initialization failed"); }
        render::RenderGraph graph;
        graph.addNode("DLSSMotionVectorProbe", "Probe");
        graph.markOutput("Probe.motion");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(*device, graph, 1, 1, log)) { return RHITestResult::fail(log); }
        if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
            !executor.waitForSubmittedWork()) {
            return RHITestResult::fail("Motion probe execution failed");
        }
        auto* buffer = executor.outputResource("Probe.motion")->buffer;
        buffer->invalidate();
        const void* mapped = buffer->map();
        if (mapped == nullptr) { return RHITestResult::fail("Motion probe readback failed"); }
        std::array<std::array<float, 4>, kMotionResultCount> values;
        std::memcpy(values.data(), mapped, sizeof(values));
        buffer->unmap();
        for (uint32_t index = 0; index < kMotionCaseCount; ++index) {
            const float halfHeight = index < 16 ? 4.0f : 2.0f;
            const std::array<float, 4> translated{
                0.5f / (2.0f * halfHeight * (640.0f / 360.0f)),
                -0.25f / (2.0f * halfHeight), 0.0f, 0.0f};
            std::array<float, 4> rotated{};
            if (index < 16) {
                const float u = (219.5f + float(index % 4) * 0.25f - 0.375f) / 640.0f;
                const float v = (173.5f + float(index / 4) * 0.25f - 0.375f) / 360.0f;
                const float x = (2.0f * u - 1.0f) * (640.0f / 360.0f);
                const float y = 1.0f - 2.0f * v;
                const float z = std::cos(0.1f) - x * std::sin(0.1f);
                const float motionX = 0.5f + (std::sin(0.1f) + x * std::cos(0.1f)) /
                    (2.0f * z * (640.0f / 360.0f)) - u;
                const float motionY = 0.5f - y / (2.0f * z) - v;
                rotated = {motionX, motionY, motionX, motionY};
            }
            for (uint32_t phase = 0; phase < 4; ++phase) {
                for (uint32_t component = 0; component < 4; ++component) {
                    const float expected = phase == 1 ? translated[component] :
                        (phase == 3 ? rotated[component] : 0.0f);
                    const float actual = values[index * 4 + phase][component];
                    if (!std::isfinite(actual) || std::abs(actual - expected) > 0.000002f) {
                        return RHITestResult::fail("DLSS reprojection mismatch: case " + std::to_string(index) +
                            ", phase " + std::to_string(phase) + ", component " + std::to_string(component) +
                            ", expected " + std::to_string(expected) + ", got " + std::to_string(actual));
                    }
                }
            }
        }
        return RHITestResult::pass("GPU: 16 jitter offsets, perspective/orthographic, static/translated/rotated cameras, sky and reset");
    }
};

METALLIC_REGISTER_RHI_TEST(DLSSMotionVectorReprojectionTest);

} // namespace
} // namespace metallic::tests
