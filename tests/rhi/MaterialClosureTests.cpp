#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;
struct ClosureProbeParams
{
    GPUBufferSpan parameters, output;
    ShaderSampledImage texture;
    ShaderBuffer counters;
    uint32_t width, height, mode, lightCount;
};
static_assert(sizeof(ClosureProbeParams) == 48 && offsetof(ClosureProbeParams, width) == 32);
constexpr uint64_t kClosureProbeABI = 0x434c4f5355520001ull;

void require(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}
template<typename T>
void require(const Result<T>& value, const std::string& message)
{
    require(bool(value), message);
}

class MaterialClosureTest final : public RHITest
{
public:
    MaterialClosureTest()
    {
        name = "material_closure_lambert_stages";
        type = RHITestType::Rendering;
    }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::string log;
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            require(bench::createTestDevice(context, {.applicationName = "Material closure stages",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if ((message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
                        (message.type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}}).transform([&](auto value) { device = std::move(value); }), "Device failed");
            ResourceRegistry registry;
            require(registry.initialize(*device), "Registry failed");
            const std::array<Float4, 2> factors{{{0.8f, 0.6f, 0.4f, 0.01f}, {0.4f, 0.7f, 0.9f, 0.02f}}};
            const std::array<Float4, 4> texels{{{1, 0.5f, 0.25f, 1}, {0.25f, 1, 0.5f, 1},
                {0.5f, 0.25f, 1, 1}, {1, 1, 1, 1}}};
            std::unique_ptr<Buffer> parameters, upload, output, counters;
            const auto makeBuffer = [&](size_t bytes, MemoryLocation location, BufferUsageBits usage, auto& target) {
                require(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = location})
                    .transform([&](auto value) { target = std::move(value); }), "Buffer allocation failed");
            };
            makeBuffer(sizeof(factors), MemoryLocation::HostUpload, BufferUsageBits::Storage, parameters);
            makeBuffer(sizeof(texels), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
            makeBuffer(192 * 128 * sizeof(Float4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
            makeBuffer(5 * sizeof(uint32_t), MemoryLocation::HostReadback, BufferUsageBits::Storage, counters);
            const auto write = [](Buffer& buffer, const void* data, size_t size) {
                auto* mapped = buffer.map();
                require(mapped != nullptr, "Upload mapping failed");
                std::memcpy(mapped, data, size); buffer.flush(); buffer.unmap();
            };
            write(*parameters, factors.data(), sizeof(factors));
            write(*upload, texels.data(), sizeof(texels));
            std::unique_ptr<Texture> texture;
            std::unique_ptr<TextureView> view;
            require(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::RGBA32Sfloat, .width = 2, .height = 2})
                .transform([&](auto value) { texture = std::move(value); }), "Texture allocation failed");
            require(device->createTextureView(*texture, {.format = Format::RGBA32Sfloat})
                .transform([&](auto value) { view = std::move(value); }), "Texture view failed");
            {
                bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                require(gpu.initialize(*device), "Upload commands failed");
                const TextureBarrierDesc before{.texture = texture.get(), .oldLayout = TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination, .before = {},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
                require(gpu.commands->synchronize({.textures = {&before, 1}}), "Upload barrier failed");
                require(upload->slice().and_then([&](const auto& slice) {
                    return gpu.commands->copyBufferToTexture({.texture = texture.get(), .buffer = slice, .width = 2, .height = 2});
                }), "Texture copy failed");
                const TextureBarrierDesc after{.texture = texture.get(), .oldLayout = TextureLayout::TransferDestination,
                    .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
                require(gpu.commands->synchronize({.textures = {&after, 1}}), "Read barrier failed");
                require(gpu.submitAndWait(), "Texture upload failed");
            }
            ComputeKernel contract, renderer;
            const auto compile = [&](const char* entry, ComputeKernel& kernel) {
                auto compiled = compileSlangShaderToSpirv({.moduleName = "MaterialClosureProbe", .entryPointName = entry,
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, {.enableDiskCache = false}, log);
                require(compiled, "Closure compile failed: " + log);
                for (const auto& path : compiled->dependencies) {
                    require(path.find("OpenPBR") == std::string::npos && path.find("SceneSurface") == std::string::npos,
                        "Diagnostic closure depends on concrete production scene/material code");
                }
                auto result = kernel.initialize(*device, {.spirv = compiled->spirv,
                    .parameters = parameterAbi<ClosureProbeParams>(kClosureProbeABI, ParameterTransport::InlinePush)}, log);
                require(result, "Closure kernel failed: " + log);
            };
            compile("materialClosureContractMain", contract);
            compile("materialClosureRenderMain", renderer);
            std::array<uint32_t, 5> counts{};
            const auto dispatch = [&](ComputeKernel& kernel, uint32_t width, uint32_t height, uint32_t mode,
                uint32_t lights, bool probe) {
                counts.fill(0); write(*counters, counts.data(), sizeof(counts));
                bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                require(gpu.initialize(*device), "Dispatch commands failed");
                ParameterWriter writer(*device, registry);
                ClosureProbeParams params{writer.bufferSpan<Float4>(parameters.get()), writer.bufferSpan<Float4>(output.get()),
                    writer.sampledImageHandle(view.get()), writer.buffer(counters.get()), width, height, mode, lights};
                auto encoded = writer.encode(params, kClosureProbeABI, ParameterTransport::InlinePush);
                require(encoded, "Parameter encode failed");
                require(kernel.dispatch(*gpu.commands, *encoded, probe ? 16 : (width + 7) / 8, probe ? 1 : (height + 7) / 8),
                    "Kernel dispatch failed");
                require(gpu.submitAndWait(), "GPU dispatch failed");
                output->invalidate(); counters->invalidate();
                const auto* countData = counters->map();
                require(countData != nullptr, "Counter readback failed");
                std::memcpy(counts.data(), countData, sizeof(counts)); counters->unmap();
                const auto* pixels = static_cast<const Float4*>(output->map());
                require(pixels != nullptr, "Output readback failed");
                std::vector<Float4> result(pixels, pixels + (probe ? 10240 : width * height));
                output->unmap();
                for (const auto& pixel : result) {
                    for (float value : pixel) { require(std::isfinite(value), "Non-finite closure result"); }
                }
                return result;
            };
            const auto data = dispatch(contract, 1024, 1, 0, 8, true);
            require(counts == std::array<uint32_t, 5>{1024, 4096, 13312, 2048, 1024}, "Contract stage counts changed");
            const auto near = [](float a, float b) { return std::abs(a - b) <= 3e-5f; };
            double meanCosine = 0;
            for (size_t lane = 0; lane < 1024; ++lane) {
                const auto base = lane * 10;
                const auto& factor = factors[lane & 1];
                const auto& texel = texels[lane & 3];
                const auto& wi = data[base + 1];
                const float cosine = (lane & 1) ? wi[1] * 0.6f + wi[2] * 0.8f : wi[2];
                meanCosine += cosine;
                require(cosine > 0 && near(wi[0] * wi[0] + wi[1] * wi[1] + wi[2] * wi[2], 1), "Invalid cosine sample");
                require(near(wi[3], cosine / std::numbers::pi_v<float>) && near(wi[3], data[base + 3][3]), "Sample/eval PDF mismatch");
                require(near(data[base][3], 1 / std::numbers::pi_v<float>) && near(data[base + 2][3], 1) &&
                    data[base + 6][3] == 5 && data[base + 8][3] == 0, "BSDF flags, eta or material flags changed");
                for (size_t channel = 0; channel < 3; ++channel) {
                    const float albedo = factor[channel] * texel[channel];
                    const float expected = albedo / std::numbers::pi_v<float>;
                    for (size_t slot : {0u, 3u, 4u, 7u}) {
                        require(near(data[base + slot][channel], expected), "View-independent Lambert eval or transport mismatch");
                    }
                    require(near(data[base + 2][channel], albedo), "White-furnace throughput differs from reflectance");
                    require(near(data[base + 6][channel], 8 * expected), "Multilight evaluation mismatch");
                    require(near(data[base + 8][channel], factor[3]), "Emission was not resolved once");
                }
                for (size_t slot : {5u, 9u}) {
                    for (float value : data[base + slot]) { require(value == 0, "Back-facing closure did not return invalid/zero"); }
                }
            }
            require(std::abs(meanCosine / 1024 - 2.0 / 3.0) < 0.001, "Cosine-sampling distribution mismatch");
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream report(context.outputDirectory / "MaterialClosureStages.txt");
            report << "contract: evaluate=1024 prepare=4096 eval=13312 sample=2048 texture=1024\n";
            report << "cosine-sample mean=" << meanCosine / 1024 << " expected=2/3; tolerance=0.001\n";
            uint32_t primaryHits = 0;
            for (uint32_t mode = 0; mode < 3; ++mode) {
                const uint32_t lights = mode == 0 ? 1 : 8;
                const auto pixels = dispatch(renderer, 192, 128, mode == 2 ? 1 : 0, lights, false);
                require(counts[0] > 1000 && counts[0] == counts[1] && counts[0] == counts[4] &&
                    counts[2] == counts[0] * lights && counts[3] == (mode == 2 ? counts[0] : 0), "Per-hit rendering stage counts changed");
                if (mode == 0) { primaryHits = counts[0]; }
                if (mode == 1) { require(counts[0] == primaryHits, "More lights reevaluated material textures"); }
                if (mode == 2) { require(counts[0] > primaryHits, "Path continuation did not shade secondary hits"); }
                const std::string label = mode == 0 ? "LambertOneLight" : mode == 1 ? "LambertEightLights" : "LambertPathTrace";
                report << label << ": evaluate=" << counts[0] << " prepare=" << counts[1] << " eval=" << counts[2]
                    << " sample=" << counts[3] << " texture=" << counts[4] << '\n';
                std::ofstream raw(context.outputDirectory / (label + ".rgba32f"), std::ios::binary);
                raw.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(Float4));
                require(bool(raw), "HDR artifact write failed");
                std::vector<uint8_t> rgba(pixels.size() * 4);
                for (size_t pixel = 0; pixel < pixels.size(); ++pixel) {
                    for (size_t channel = 0; channel < 3; ++channel) {
                        const float value = std::clamp(pixels[pixel][channel], 0.0f, 1.0f);
                        const float srgb = value <= 0.0031308f ? 12.92f * value : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
                        rgba[pixel * 4 + channel] = static_cast<uint8_t>(std::lround(srgb * 255));
                    }
                    rgba[pixel * 4 + 3] = 255;
                }
                require(saveRgba8Png(context.outputDirectory / (label + ".png"), rgba.data(), 192, 128, log), log);
            }
            require(bool(report), "Stage report write failed");
            require(validationErrors == 0, "Closure workload caused Vulkan validation errors");
            return RHITestResult::pass("Lambert evaluate/prepare/eval/sample, white furnace, 1/8-light texture reuse and secondary-hit continuation verified");
        } catch (const std::exception& exception) {
            return RHITestResult::fail(exception.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialClosureTest);
} // namespace
} // namespace metallic::tests
