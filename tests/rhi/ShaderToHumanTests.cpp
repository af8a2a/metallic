#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include "Runtime/Render/Core/ComputeKernel.h"
#include "harness/Fixtures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

class ShaderToHumanShaderCompileTest final : public RHITest {
public:
    ShaderToHumanShaderCompileTest()
    {
        type = RHITestType::Resource;
        name = "shader_to_human_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        struct ShaderEntry {
            const char* module;
            const char* entry;
        };
        constexpr ShaderEntry entries[] = {
            {"Features/Debug/ShaderToHumanExample", "shaderToHumanExampleMain"},
            {"Features/Debug/ShaderToHumanExample", "shaderToHumanExampleFragmentMain"},
            {"Features/Debug/ShaderToHumanScatterExample", "shaderToHumanScatterExampleMain"},
        };
        for (const ShaderEntry& entry : entries) {
            render::ShaderCompileResult shader;
            const auto result = render::compileSlangShaderToSpirv({
                .moduleName = entry.module,
                .entryPointName = entry.entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            if (!result || shader.spirv.empty()) {
                return RHITestResult::fail(std::string(entry.entry) + ": " + shader.diagnostics);
            }
            for (const char* dependency : {"/Modules/ShaderToHuman.slang",
                    "/ShaderToHuman/Font.slang", "/ShaderToHuman/Gather.slang",
                    "/ShaderToHuman/Geometry.slang", "/ShaderToHuman/Scatter.slang"}) {
                if (std::count_if(shader.dependencies.begin(), shader.dependencies.end(),
                        [&](const std::string& path) { return path.ends_with(dependency); }) != 1) {
                    return RHITestResult::fail(std::string("Missing or duplicate module dependency: ") + dependency);
                }
            }
            if (std::any_of(shader.dependencies.begin(), shader.dependencies.end(),
                    [](const std::string& path) { return path.find("/ShaderToHuman/include/") != std::string::npos; })) {
                return RHITestResult::fail("Production module still depends on upstream HLSL");
            }
        }
        return RHITestResult::pass("ShaderToHuman module: compute/fragment gather, generic scatter, unique Slang dependencies");
    }
};

METALLIC_REGISTER_RHI_TEST(ShaderToHumanShaderCompileTest);

class ShaderToHumanGPUBehaviorTest final : public RHITest {
public:
    ShaderToHumanGPUBehaviorTest()
    {
        type = RHITestType::Rendering;
        name = "shader_to_human_module_gpu_behavior";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        using Pixel = std::array<float, 4>;
        constexpr uint32_t kWidth = 320, kHeight = 256;
        constexpr size_t kPixels = kWidth * kHeight;
        constexpr uint64_t kABI = 0x53324850524f0001ull;
        struct Params { GPUBufferSpan output; };
        static_assert(sizeof(Params) == 12);
        bench::TestDevice device;
        auto result = bench::createTestDevice(context, {.applicationName = "ShaderToHuman behavior",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptors"); }
        if (!result) { return RHITestResult::fail("ShaderToHuman device creation failed"); }
        ResourceRegistry registry;
        if (!registry.initialize(*device)) { return RHITestResult::fail("registry initialization failed"); }
        std::vector<Pixel> pixels;
        std::unique_ptr<Buffer> output;
        result = device->createBuffer({.size = kPixels * 2 * sizeof(Pixel), .structureStride = sizeof(Pixel),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
            .transform([&](auto value) { output = std::move(value); });
        if (!result) { return RHITestResult::fail("probe buffer creation failed"); }
        auto* mapped = output->map();
        if (!mapped) { return RHITestResult::fail("probe buffer map failed"); }
        std::memset(mapped, 0, kPixels * 2 * sizeof(Pixel));
        output->flush();
        output->unmap();
        for (uint32_t scatter = 0; scatter < 2; ++scatter) {
            ShaderCompileResult shader;
            result = compileSlangShaderToSpirv({.moduleName = "ShaderToHumanProbe",
                .entryPointName = scatter ? "shaderToHumanScatterProbeMain" : "shaderToHumanGatherProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
                shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            if (!result) { return RHITestResult::fail(shader.diagnostics); }
            ComputeKernel kernel;
            std::string log;
            result = kernel.initialize(*device, {.spirv = shader.spirv,
                .parameters = parameterAbi<Params>(kABI, ParameterTransport::InlinePush)}, log);
            if (!result) { return RHITestResult::fail(log); }
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            if (!gpu.initialize(*device)) { return RHITestResult::fail("probe command initialization failed"); }
            ParameterWriter writer(*device, registry);
            const Params params{writer.bufferSpan<Pixel>(output.get())};
            auto encoded = writer.encode(params, kABI, ParameterTransport::InlinePush);
            if (!encoded) { return RHITestResult::fail("probe parameter encoding failed"); }
            result = kernel.dispatch(*gpu.commands, *encoded, scatter ? 1 : kWidth / 8, scatter ? 1 : kHeight / 8);
            if (!result || !gpu.submitAndWait()) { return RHITestResult::fail("probe dispatch/wait failed"); }
        }
        output->invalidate();
        mapped = output->map();
        if (!mapped) { return RHITestResult::fail("probe readback failed"); }
        pixels.resize(kPixels * 2);
        std::memcpy(pixels.data(), mapped, kPixels * 2 * sizeof(Pixel));
        output->unmap();
        size_t scatterPixels = 0, gatherPixels = 0;
        const Pixel background{0.025f, 0.035f, 0.05f, 1.0f};
        for (size_t pixel = 0; pixel < kPixels * 2; ++pixel) {
            for (float value : pixels[pixel]) {
                if (!std::isfinite(value)) { return RHITestResult::fail("non-finite GPU pixel"); }
            }
            if (pixel >= kPixels && pixels[pixel][3] > 0) { ++scatterPixels; }
            if (pixel < kPixels && pixels[pixel] != background) { ++gatherPixels; }
        }
        if (scatterPixels < 100 || gatherPixels < 1000) {
            return RHITestResult::fail("probe produced insufficient visible content");
        }
        const auto matches = [&](size_t pixel, Pixel expected) {
            for (size_t channel = 0; channel < 4; ++channel) {
                if (std::abs(pixels[pixel][channel] - expected[channel]) > 1e-6f) { return false; }
            }
            return true;
        };
        // Independent known colors: untouched background, gather crosshair,
        // clipped scatter block, scatter disc, and untouched scatter output.
        if (!matches(kPixels - 1, background) ||
            !matches(112 * kWidth + 80, {1.0f, 0.5f, 0.1f, 1.0f}) ||
            !matches(kPixels + 40 * kWidth + 4, {0.8f, 0.3f, 0.1f, 1.0f}) ||
            !matches(kPixels + 40 * kWidth + 20, {0.1f, 0.8f, 0.3f, 1.0f}) ||
            !matches(kPixels * 2 - 1, {0, 0, 0, 0})) {
            return RHITestResult::fail("known gather/scatter pixel color mismatch");
        }
        std::vector<uint8_t> preview(kPixels * 2 * 4);
        for (size_t pixel = 0; pixel < kPixels * 2; ++pixel) {
            for (size_t channel = 0; channel < 3; ++channel) {
                preview[pixel * 4 + channel] = static_cast<uint8_t>(
                    std::clamp(pixels[pixel][channel], 0.0f, 1.0f) * 255.0f);
            }
            preview[pixel * 4 + 3] = 255;
        }
        std::filesystem::create_directories(context.outputDirectory);
        std::string log;
        if (!saveRgba8Png(context.outputDirectory / "ShaderToHuman.png", preview.data(),
                kWidth, kHeight * 2, log)) { return RHITestResult::fail(log); }
        return RHITestResult::pass("GPU gather/scatter finite values, coverage and known pixel colors validated");
    }
};

METALLIC_REGISTER_RHI_TEST(ShaderToHumanGPUBehaviorTest);

} // namespace
} // namespace metallic::tests
