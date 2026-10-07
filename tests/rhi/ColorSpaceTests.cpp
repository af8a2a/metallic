#include "RHITest.h"
#include "TestResourceLayouts.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/SceneColorConversion.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>

namespace metallic::tests {
using namespace render;
TEST(ColorSpace, RoundTripNeutralAndKnownPrimaries)
{
    using namespace color;
    // Independently calculated reference Rec.709/D65 red -> Bradford AP1/D60.
    const auto red = rec709ToACEScg({1, 0, 0});
    EXPECT_NEAR(red[0], 0.6130974, 2e-6);
    EXPECT_NEAR(red[1], 0.0701937, 2e-6);
    EXPECT_NEAR(red[2], 0.0206156, 2e-6);
    for (auto c : {RGB{0, 0, 0}, RGB{0.18f, 0.18f, 0.18f}, RGB{1, 0, 0}, RGB{0, 1, 0}, RGB{0, 0, 1}, RGB{-0.2f, 4, 100}}) {
        const auto back = ACEScgToRec709(rec709ToACEScg(c));
        for (int i = 0; i < 3; ++i) { EXPECT_NEAR(back[i], c[i], 3e-5); }
    }
    const auto neutral = rec709ToACEScg({0.18f, 0.18f, 0.18f});
    for (float v : neutral) { EXPECT_NEAR(v, 0.18f, 1e-6); }
    const auto white = transform(kD65ToD60, {0.950455927f, 1, 1.089057751f});
    EXPECT_NEAR(white[0], 0.952646075, 1e-6);
    EXPECT_NEAR(white[1], 1.0, 1e-6);
    EXPECT_NEAR(white[2], 1.008825184, 1e-6);
}
TEST(ColorSpace, MetadataAndAuthoringRemainIndependent)
{
    const color::RGB rgb{0.2f, 0.5f, 0.8f};
    EXPECT_EQ(color::fromLinearRec709(rgb, SceneWorkingColorSpace::LinearRec709), rgb);
    EXPECT_EQ(color::fromSource(rgb, kACEScg, SceneWorkingColorSpace::ACEScg), rgb);
    scene::RenderMaterial authored;
    authored.baseColorFactor = float4(1, 0, 0, 0.37f);
    authored.roughnessFactor = 0.23f;
    const auto working = resolveWorkingMaterial(authored);
    EXPECT_EQ(authored.baseColorFactor.x, 1);
    EXPECT_EQ(working.baseColorFactor.w, 0.37f);
    EXPECT_EQ(working.roughnessFactor, 0.23f);
    EXPECT_EQ(authored.normalTexture.colorMetadata.semantic, TextureSemantic::Data);
    EXPECT_EQ(authored.baseColorTexture.colorMetadata.semantic, TextureSemantic::Color);
    EXPECT_EQ(textureColorFlags({TextureSemantic::Data, kACEScg}), 5u);
    EXPECT_THROW(textureColorFlags({TextureSemantic::Color, {ColorPrimaries::ACESAP1, WhitePoint::D65, TransferFunction::Linear}}), std::invalid_argument);
}

class WorkingColorGPUTest final : public RHITest {
public:
    WorkingColorGPUTest() { type = RHITestType::Rendering; name = "working_color_cpu_gpu_texture_contract"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Working color probe",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!created) { return RHITestResult::fail("Cannot initialize bindless color-probe device"); }
        auto& gpuDevice = *device;
        auto& graphicsQueue = *device->getQueue(QueueType::Graphics);
        std::string log;
        auto fail = [&](const char* message) { return RHITestResult::fail(std::string(message) + ": " + log); };
        ShaderCompileResult shader;
        const SlangMacroDefine reservedDefine{"METALLIC_WORKING_SPACE_ACESCG", "0"};
        const auto overrideResult = compileSlangShaderToSpirv({.moduleName = "WorkingColorProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = {&reservedDefine, 1}}, log);
        if (!hasError(overrideResult, Error::InvalidArgument)) { return fail("Per-shader basis override was accepted"); }
        const SlangMacroDefine adapterDefine{"METALLIC_TEST_NRD_ADAPTER", "1"};
        if (!compileSlangShaderToSpirv({.moduleName = "WorkingColorProbe", .entryPointName = "main",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = {&adapterDefine, 1}}, log).transform([&](auto value) { shader = std::move(value); })) { return fail("compile"); }
        const ComputeResourceBindingDesc layout{.binding = 0, .kind = ComputeResourceBindingKind::StorageBuffer};
        ComputeProgram program;
        if (!program.initialize(gpuDevice, {.spirv = shader.spirv, .bindings = {&layout, 1},
                .requiresRayQuery = false, .resourceParameters = kBatchBarrierProbeLayout}, log)) { return fail("program"); }
        std::unique_ptr<Buffer> output;
        if (!gpuDevice.createBuffer({.size = 11*16, .structureStride = 16, .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { output = std::move(value); })) { return fail("buffer"); }
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        if (!gpuDevice.createCommandPool(graphicsQueue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !gpuDevice.createFence({}).transform([&](auto value) { fence = std::move(value); }) || !commands->begin()) { return fail("commands"); }
        const ComputeDispatchBinding binding{.binding = 0, .buffer = output.get()};
        if (!program.dispatch({.commandBuffer = commands.get(), .bindings = {&binding, 1}}) || !commands->end()) { return fail("dispatch"); }
        CommandBuffer* raw = commands.get();
        if (!graphicsQueue.submit({.commandBuffers = {&raw, 1}, .signalFence = fence.get()}) || !fence->wait()) { return fail("submit"); }
        output->invalidate();
        const auto* actual = static_cast<const float*>(output->map());
        if (!actual) { return fail("map"); }
        const color::RGB sample{0.5f, 0.2f, 0.8f};
        auto linear = sample;
        for (auto& c : linear) { c = color::decodeSRGB(c); }
        const auto red = color::fromLinearRec709({1, 0, 0});
        auto product = linear;
        product[0] *= 0.8f; product[1] *= 0.3f; product[2] *= 0.7f;
        const std::array<color::RGB, 8> expected{red, color::RGB{1, 0, 0}, color::fromLinearRec709(linear),
            sample, color::fromSource(sample, kACEScg), color::fromLinearRec709(product),
            color::rec709ToACEScg({0.18f, 0.18f, 0.18f}), color::RGB{4, -0.2f, 10}};
        bool matches = true;
        for (size_t row = 0; row < expected.size(); ++row) {
            for (size_t channel = 0; channel < 3; ++channel) {
                if (!std::isfinite(actual[row*4 + channel]) || std::abs(actual[row*4 + channel] - expected[row][channel]) > 1e-5f) { matches = false; }
            }
        }
        matches &= std::abs(actual[3] - color::luminance(red)) < 1e-6f;
        matches &= std::abs(actual[32] - 0.8f) < 1e-5f && std::abs(actual[33] - 0.3f) < 1e-5f &&
            std::abs(actual[34] - 0.1f) < 1e-5f && std::abs(actual[35] - 2.0f) < 1e-5f;
        auto ap1Product = color::rec709ToACEScg({0.8f, 0.3f, 0.7f});
        for (size_t c = 0; c < 3; ++c) { ap1Product[c] *= sample[c]; }
        const auto expectedAP1 = color::fromSource(ap1Product, kACEScg);
        for (size_t c = 0; c < 3; ++c) { matches &= std::abs(actual[36 + c] - expectedAP1[c]) < 1e-5f; }
        const auto expectedKey = color::fromLinearRec709({5.2f, 4.8f, 4.2f});
        for (size_t c = 0; c < 3; ++c) { matches &= std::abs(actual[40 + c] - expectedKey[c]) < 1e-5f; }
        output->unmap();
        return matches ? RHITestResult::pass() : RHITestResult::fail("CPU/GPU matrix, texture Color/Data or factor modulation mismatch");
    }
};
METALLIC_REGISTER_RHI_TEST(WorkingColorGPUTest);
} // namespace metallic::tests
