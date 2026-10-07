#include "TestResourceLayouts.h"
#include "RHITest.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numbers>
#include <sstream>

namespace metallic::tests {
namespace {

constexpr uint32_t kFootprintCaseCount = 32;
constexpr uint32_t kHighResolutionCaseStart = 16;

struct TextureFootprintProbePush {
    uint32_t width;
    uint32_t height;
    uint32_t index;
    uint32_t orthographic;
    float aspect;
    float verticalFovRadians;
    float orthographicHeight;
    float reserved = 0.0f;
};
static_assert(sizeof(TextureFootprintProbePush) == 32);

std::array<TextureFootprintProbePush, kFootprintCaseCount> footprintCases()
{
    constexpr float kRadians = std::numbers::pi_v<float> / 180.0f;
    std::array<TextureFootprintProbePush, kFootprintCaseCount> cases{{
        {1920, 1080, 0, 0, 16.0f / 9.0f, 60.0f * kRadians, 4.0f},
        {3840, 2160, 0, 0, 16.0f / 9.0f, 60.0f * kRadians, 4.0f},
        {640, 360, 0, 0, 16.0f / 9.0f, 90.0f * kRadians, 4.0f},
        {640, 360, 0, 0, 16.0f / 9.0f, 30.0f * kRadians, 4.0f},
        {1024, 1024, 0, 0, 2.0f, 90.0f * kRadians, 4.0f},
        {4096, 1024, 0, 0, 2.0f, 90.0f * kRadians, 4.0f},
        {1280, 720, 0, 1, 16.0f / 9.0f, 60.0f * kRadians, 4.0f},
        {640, 720, 0, 1, 16.0f / 9.0f, 60.0f * kRadians, 4.0f},
        {2560, 720, 0, 1, 16.0f / 9.0f, 60.0f * kRadians, 4.0f},
        {1280, 720, 0, 1, 16.0f / 9.0f, 30.0f * kRadians, 8.0f},
        {1920, 1080, 0, 0, 16.0f / 9.0f, 0.0001f, 4.0f},
        {1920, 1080, 0, 0, 16.0f / 9.0f, 120.0f * kRadians, 4.0f},
        {65536, 65536, 0, 0, 1.0f, 60.0f * kRadians, 4.0f},
        {16777216, 16777216, 0, 0, 1.0f, 60.0f * kRadians, 4.0f},
        {65536, 65536, 0, 1, 1.0f, 60.0f * kRadians, 4.0f},
        {16777216, 16777216, 0, 1, 1.0f, 60.0f * kRadians, 4.0f},
    }};
    for (uint32_t index = kHighResolutionCaseStart; index < kFootprintCaseCount; ++index) {
        // Adjacent high-resolution projections must produce smoothly shrinking
        // positive cones instead of acos/dot quantization plateaus or zeros.
        const uint32_t height = 1048576u + (index - kHighResolutionCaseStart) * 4096u;
        cases[index] = {height * 2, height, index, 0, 2.0f, 60.0f * kRadians, 4.0f};
    }
    for (uint32_t index = 0; index < kFootprintCaseCount; ++index) {
        cases[index].index = index;
    }
    return cases;
}

class TextureFootprintProbePass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        auto& cones = reflection.addBufferOutput("cones")
            .buffer(kFootprintCaseCount * sizeof(float) * 2, sizeof(float) * 2).storageReadWrite();
        cones.memoryLocation = render::MemoryLocation::HostReadback;
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({
            .moduleName = "TextureFootprintProbe",
            .entryPointName = "textureFootprintProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc binding{
            .binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .pushConstantSize = sizeof(TextureFootprintProbePush),
            .bindings = {&binding, 1},
            .requiresRayQuery = false,
            .resourceParameters = metallic::tests::kTextureFootprintProbeLayout,
        }, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding binding{
            .binding = 0, .buffer = context.outputBuffer("cones").buffer()};
        for (const auto& input : footprintCases()) {
            auto result = program_.dispatch({
                .commandBuffer = &context.commandBuffer(),
                .bindings = {&binding, 1},
                .pushData = &input,
                .pushDataSize = sizeof(input),
            });
            if (!result) { return result; }
        }
        return {};
    }

private:
    render::ComputeProgram program_;
};

class TextureFootprintTest final : public RHITest {
public:
    TextureFootprintTest()
    {
        type = RHITestType::Rendering;
        name = "texture_primary_ray_cone_pixel_footprint";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::registerRenderGraphPassType("TextureFootprintProbe", "Pixel cone projection probe",
            [] { return std::make_unique<TextureFootprintProbePass>(); });
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "Texture footprint",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (render::hasError(initialized, render::Error::Unsupported)) {
            return RHITestResult::skip("Requires bindless descriptors");
        }
        if (!initialized) { return RHITestResult::fail("Device initialization failed"); }
        render::RenderGraph graph;
        graph.addNode("TextureFootprintProbe", "Probe");
        graph.markOutput("Probe.cones");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(*device, graph, 1, 1, log)) { return RHITestResult::fail(log); }
        if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
            !executor.waitForSubmittedWork()) {
            return RHITestResult::fail("Pixel cone probe execution failed");
        }
        auto* buffer = executor.outputResource("Probe.cones")->buffer;
        buffer->invalidate();
        const void* mapped = buffer->map();
        if (mapped == nullptr) { return RHITestResult::fail("Pixel cone probe readback failed"); }
        std::array<std::array<float, 2>, kFootprintCaseCount> values{};
        std::memcpy(values.data(), mapped, sizeof(values));
        buffer->unmap();
        const auto cases = footprintCases();
        for (uint32_t index = 0; index < kFootprintCaseCount; ++index) {
            const auto& input = cases[index];
            // Project the top and bottom sides of the image plane in double
            // precision, then divide that extent into pixels along each axis.
            const double imageHeight = input.orthographic ? double(input.orthographicHeight) :
                2.0 * std::tan(double(input.verticalFovRadians) * 0.5);
            const double diameter = std::max(imageHeight / double(input.height),
                imageHeight * double(input.aspect) / double(input.width));
            for (uint32_t component = 0; component < 2; ++component) {
                const double expected = component == (input.orthographic ? 0u : 1u) ? diameter : 0.0;
                const double actual = values[index][component];
                // GPU tan and sin/cos both measured 24.3 ppm error at the
                // extra 0.0001-radian FOV stress case (below the editor limit).
                // Allow that intrinsic precision only for ultra-narrow FOV;
                // normal projections retain the original tolerance. Separate
                // positivity/continuity checks still reject quantized cones.
                const double relativeTolerance = !input.orthographic && input.verticalFovRadians < 0.001f
                    ? 1e-4 : 3e-6;
                const double tolerance = std::max(1e-12, expected * relativeTolerance);
                if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
                    std::ostringstream failure;
                    failure << std::scientific << std::setprecision(10)
                        << "Pixel diameter mismatch: case " << index << ", component " << component
                        << ", expected " << expected << ", actual " << actual
                        << ", tolerance " << tolerance
                        << ", relative error " << (expected != 0.0 ? (actual - expected) / expected : actual)
                        << ", fov " << input.verticalFovRadians
                        << ", render " << input.width << "x" << input.height;
                    return RHITestResult::fail(failure.str());
                }
            }
            if (values[index][input.orthographic ? 0 : 1] <= 0.0f) {
                return RHITestResult::fail("Pixel cone became non-positive at case " + std::to_string(index));
            }
            if (index > kHighResolutionCaseStart && values[index][1] >= values[index - 1][1]) {
                return RHITestResult::fail("High-resolution pixel cones lost continuity at case " + std::to_string(index));
            }
        }
        return RHITestResult::pass("GPU: pixel diameter, FOV, orthographic scale, unequal axes and positive continuous cones up to 16M resolution");
    }
};

METALLIC_REGISTER_RHI_TEST(TextureFootprintTest);

} // namespace
} // namespace metallic::tests
