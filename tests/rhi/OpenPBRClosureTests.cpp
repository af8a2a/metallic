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
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;
struct OpenPBRProbeParams
{
    GPUBufferSpan parameters, output;
    ShaderSampledImage texture;
    ShaderBuffer counters;
    uint32_t lightCount;
};
static_assert(sizeof(OpenPBRProbeParams) == 36 && offsetof(OpenPBRProbeParams, lightCount) == 32);
constexpr uint64_t kOpenPBRProbeABI = 0x4f504252434c0001ull;

static constexpr uint16_t kLut0[] = {
#include "../../External/openpbr-bsdf/impl/data/openpbr_ideal_dielectric_energy_complement_data.h"
};
static constexpr uint16_t kLut1[] = {
#include "../../External/openpbr-bsdf/impl/data/openpbr_ideal_dielectric_avg_energy_complement_data.h"
};
static constexpr uint16_t kLut2[] = {
#include "../../External/openpbr-bsdf/impl/data/openpbr_ideal_dielectric_reflection_ratio_data.h"
};
static constexpr uint16_t kLut3[] = {
#include "../../External/openpbr-bsdf/impl/data/openpbr_opaque_dielectric_energy_complement_data.h"
};
static constexpr uint16_t kLut4[] = {
#include "../../External/openpbr-bsdf/impl/data/openpbr_opaque_dielectric_avg_energy_complement_data.h"
};
static constexpr uint16_t kLut5[] = {
#include "../../External/openpbr-bsdf/impl/data/openpbr_ideal_metal_energy_complement_data.h"
};
static constexpr uint16_t kLut6[] = {
#include "../../External/openpbr-bsdf/impl/data/openpbr_ideal_metal_avg_energy_complement_data.h"
};

void require(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}
template<typename T>
void require(const Result<T>& value, const std::string& message)
{
    require(bool(value), message);
}

class OpenPBRClosureTest final : public RHITest
{
public:
    OpenPBRClosureTest()
    {
        name = "material_closure_openpbr_stages";
        type = RHITestType::Rendering;
    }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::string log;
            // This retained legacy preview entry has a separate resource ABI;
            // it shares Closure preparation but is not the production VBuffer.
            auto legacy = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityBufferShading",
                .entryPointName = "visibilityBufferShadeMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log);
            require(legacy, "Legacy visibility closure compile failed: " + log);
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            require(bench::createTestDevice(context, {.applicationName = "OpenPBR closure stages",
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
            std::vector<Float4> parameterData(factors.begin(), factors.end());
            const auto appendLut = [&](const auto& values) {
                for (const auto value : values) { parameterData.push_back({float(value) * (1.0f / 65535.0f), 0, 0, 0}); }
            };
            appendLut(kLut0); appendLut(kLut1); appendLut(kLut2); appendLut(kLut3);
            appendLut(kLut4); appendLut(kLut5); appendLut(kLut6);
            require(parameterData.size() == 69666, "Vendor LUT dimensions changed");
            std::unique_ptr<Buffer> parameters, upload, output, counters;
            const auto makeBuffer = [&](size_t bytes, MemoryLocation location, BufferUsageBits usage, auto& target) {
                require(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = location})
                    .transform([&](auto value) { target = std::move(value); }), "Buffer allocation failed");
            };
            makeBuffer(parameterData.size() * sizeof(Float4), MemoryLocation::HostUpload, BufferUsageBits::Storage, parameters);
            makeBuffer(sizeof(texels), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
            makeBuffer(1024 * 8 * sizeof(Float4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
            makeBuffer(5 * sizeof(uint32_t), MemoryLocation::HostReadback, BufferUsageBits::Storage, counters);
            const auto write = [](Buffer& buffer, const void* data, size_t size) {
                auto* mapped = buffer.map();
                require(mapped != nullptr, "Upload mapping failed");
                std::memcpy(mapped, data, size); buffer.flush(); buffer.unmap();
            };
            write(*parameters, parameterData.data(), parameterData.size() * sizeof(Float4));
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
            std::string shaderLog;
            auto shader = compileSlangShaderToSpirv({.moduleName = "OpenPBRClosureProbe",
                .entryPointName = "openPBRClosureProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
                {.enableDiskCache = false}, shaderLog);
            require(shader, "OpenPBR compile failed: " + shaderLog);
            ComputeKernel kernel;
            require(kernel.initialize(*device, {.spirv = shader->spirv,
                .parameters = parameterAbi<OpenPBRProbeParams>(kOpenPBRProbeABI, ParameterTransport::InlinePush)}, log), log);
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream report(context.outputDirectory / "OpenPBRClosureStages.txt");
            for (uint32_t lightCount : {1u, 8u}) {
                std::array<uint32_t, 5> counts{};
                write(*counters, counts.data(), sizeof(counts));
                bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                require(gpu.initialize(*device), "Dispatch commands failed");
                ParameterWriter writer(*device, registry);
                OpenPBRProbeParams params{writer.bufferSpan<Float4>(parameters.get()), writer.bufferSpan<Float4>(output.get()),
                    writer.sampledImageHandle(view.get()), writer.buffer(counters.get()), lightCount};
                auto encoded = writer.encode(params, kOpenPBRProbeABI, ParameterTransport::InlinePush);
                require(encoded, "Parameter encode failed");
                require(kernel.dispatch(*gpu.commands, *encoded, 16), "Dispatch failed");
                require(gpu.submitAndWait(), "GPU dispatch failed");
                output->invalidate(); counters->invalidate();
                const auto* countData = counters->map();
                require(countData != nullptr, "Counter readback failed");
                std::memcpy(counts.data(), countData, sizeof(counts)); counters->unmap();
                const std::array<uint32_t, 5> expectedCounts{1024, 8192, 2048, 1024, lightCount * 1024};
                require(counts == expectedCounts, "Material/texture reads grew with lighting or repeated preparation");
                const auto* mapped = static_cast<const Float4*>(output->map());
                require(mapped != nullptr, "Output mapping failed");
                std::vector<Float4> data(mapped, mapped + 1024 * 8);
                output->unmap();
                bench::readbackEvidence(context, "OpenPBR-" + std::to_string(lightCount) + "-lights.bin", std::span<const Float4>(data));
                uint32_t events = 0, invalid = 0, entering = 0, exiting = 0;
                float maxError = 0;
                for (uint32_t lane = 0; lane < 1024; ++lane) {
                    const auto base = lane * 8;
                    const auto variant = lane / 128;
                    for (uint32_t slot = 0; slot < 8; ++slot) {
                        for (float value : data[base + slot]) { require(std::isfinite(value), "Non-finite OpenPBR output"); }
                    }
                    for (uint32_t slot : {0u, 1u, 4u}) {
                        for (uint32_t c = 0; c < 3; ++c) { maxError = std::max(maxError, data[base + slot][c]); }
                    }
                    require(maxError <= 3e-4f, "OpenPBR native eval/sample/emission or f/pdf conversion differs: " + std::to_string(maxError));
                    require(data[base][3] == 0 && data[base + 5][3] == 0, "Unsupported Importance or material flags are nonzero");
                    const auto flags = uint32_t(data[base + 4][3]);
                    events |= flags;
                    invalid += data[base + 1][3] == 0;
                    require(data[base + 1][3] == data[base + 7][3], "Sample densities differ");
                    const float eta = data[base + 3][3];
                    if (flags & 2) {
                        const float expectedEta = variant == 6 ? 1.5f / 1.1f : 1.1f / 1.5f;
                        require(std::abs(eta - expectedEta) < 1e-5f, "Transmission eta is not eta_i / eta_t");
                        entering += variant != 6; exiting += variant == 6;
                    } else { require(eta == 1, "Reflection or failed sample eta differs from one"); }
                    for (uint32_t c = 0; c < 3; ++c) {
                        const float expectedBase = variant == 7 ? 0 : factors[lane & 1][c] * texels[lane & 3][c];
                        require(std::abs(data[base + 2][c] - expectedBase) < 1e-6f, "Program parameter/texture mapping differs");
                    }
                    require(std::abs(data[base + 3][0]) < 1e-6f &&
                        std::abs(data[base + 3][1] + 1 / std::sqrt(5.0f)) < 1e-6f &&
                        std::abs(data[base + 3][2] - 2 / std::sqrt(5.0f)) < 1e-6f, "Authored TBN or normal map changed");
                }
                report << "lights=" << lightCount << " parameterLoads=" << counts[0] << " textureRequests=" << counts[1]
                    << " textureLoads=" << counts[2] << " primaryPrepare=" << counts[3] << " directEval=" << counts[4]
                    << " maxError=" << maxError << " eventMask=" << events << " invalid=" << invalid
                    << " entering=" << entering << " exiting=" << exiting << '\n';
                // The current vendor implements even smooth glass as finite
                // GGX and emits Glossy, never Specular/delta. Do not claim delta
                // transport coverage from the 0.001-roughness material.
                require(events == 15 && invalid > 0 && entering > 0 && exiting > 0,
                    "Probe did not exercise reflection/transmission/diffuse/glossy/invalid and both IOR sides; mask=" +
                    std::to_string(events) + " invalid=" + std::to_string(invalid) + " entering=" +
                    std::to_string(entering) + " exiting=" + std::to_string(exiting));
            }
            require(validationErrors == 0, "Vulkan validation errors");
            return RHITestResult::pass("Shared production OpenPBR program; 1/8 lights, vendor equivalence, normal mapping and both IOR sides");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(OpenPBRClosureTest);
} // namespace
} // namespace metallic::tests
