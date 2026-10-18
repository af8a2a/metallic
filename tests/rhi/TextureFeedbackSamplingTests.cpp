#include "harness/Fixtures.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;

void requireFeedback(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}
template<typename T> void requireFeedback(const Result<T>& result, const std::string& message)
{
    requireFeedback(bool(result), message);
}

class TextureFeedbackSamplingTest final : public RHITest
{
public:
    TextureFeedbackSamplingTest() { type = RHITestType::Rendering; name = "texture_feedback_sampling"; }

    RHITestResult run(RHITestContext& context) override
    {
        try {
            bench::TestDevice device;
            requireFeedback(bench::createTestDevice(context, {.applicationName = "Texture feedback sampling",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { device = std::move(value); }), "Device");
            constexpr uint32_t kCases = 48, kPixels = 64 * 64;
            const std::array<Float4, 5> texels{{{1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {0,1,0,1}}};
            std::array<uint32_t, kCases * 8> initialFeedback{};
            for (uint32_t i = 0; i < kCases; ++i) {
                initialFeedback[i * 8] = 8;
                initialFeedback[i * 8 + 1] = 8;
                initialFeedback[i * 8 + 2] = 4;
                initialFeedback[i * 8 + 4] = UINT32_MAX;
                initialFeedback[i * 8 + 6] = 0x3f800000u; // Sampling floor 1.0, even when demand is off.
            }
            std::unique_ptr<Buffer> upload, feedback, output;
            auto allocate = [&](size_t bytes, MemoryLocation memory, BufferUsageBits usage, auto& target) {
                requireFeedback(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = memory})
                    .transform([&](auto value) { target = std::move(value); }), "Buffer");
            };
            allocate(sizeof(texels), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
            allocate(sizeof(initialFeedback), MemoryLocation::HostReadback, BufferUsageBits::Storage, feedback);
            allocate(kCases * kPixels * 2 * sizeof(Float4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
            auto write = [&](Buffer& buffer, const void* data, size_t bytes) {
                auto* mapped = buffer.map(); requireFeedback(mapped != nullptr, "Upload map");
                std::memcpy(mapped, data, bytes); buffer.flush(); buffer.unmap();
            };
            write(*upload, texels.data(), sizeof(texels));
            write(*feedback, initialFeedback.data(), sizeof(initialFeedback));
            std::unique_ptr<Texture> texture;
            std::unique_ptr<TextureView> view;
            requireFeedback(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::RGBA32Sfloat, .width = 2, .height = 2, .mipCount = 2})
                .transform([&](auto value) { texture = std::move(value); }), "Texture");
            requireFeedback(device->createTextureView(*texture, {.format = Format::RGBA32Sfloat, .range = {.mipCount = 2}})
                .transform([&](auto value) { view = std::move(value); }), "View");
            std::string log;
            const SlangMacroDefine macros[] = {{"METALLIC_CUSTOM_MATERIALS", "0"}, {"METALLIC_REALTIME_DEFERRED", "0"}};
            const auto shader = compileSlangShaderToSpirv({.moduleName = "TextureFeedbackSamplingProbe",
                .entryPointName = "textureFeedbackSamplingMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .macroDefines = macros}, log);
            requireFeedback(shader, "Compile: " + log);
            const ComputeResourceBindingDesc layout[] = {
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureFeedback)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), ComputeResourceBindingKind::SampledImage, kCases}};
            ComputeProgram program;
            requireFeedback(program.initialize(*device, {.spirv = shader->spirv, .bindings = layout, .requiresRayQuery = false,
                .resourceParameterSize = sizeof(SceneResourceParameters)}, log), "Program: " + log);
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            requireFeedback(gpu.initialize(*device), "Commands");
            gpu.commands->hostWriteBarrier();
            const TextureBarrierDesc before{.texture = texture.get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .range = {.mipCount = 2}};
            requireFeedback(gpu.commands->synchronize({.textures = {&before, 1}}), "Upload barrier");
            for (uint32_t mip = 0; mip < 2; ++mip) {
                requireFeedback(upload->slice({mip ? 4 * sizeof(Float4) : 0}).and_then([&](const auto& slice) {
                    return gpu.commands->copyBufferToTexture({.texture = texture.get(), .buffer = slice,
                        .width = mip ? 1u : 2u, .height = mip ? 1u : 2u, .mipLevel = mip});
                }), "Copy mip");
            }
            const TextureBarrierDesc after{.texture = texture.get(), .oldLayout = TextureLayout::TransferDestination,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}, .range = {.mipCount = 2}};
            requireFeedback(gpu.commands->synchronize({.textures = {&after, 1}}), "Sample barrier");
            std::array<TextureView*, kCases> views; views.fill(view.get());
            const ComputeDispatchBinding bindings[] = {
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput), .buffer = output.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureFeedback), .buffer = feedback.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), .textureViews = views}};
            requireFeedback(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = bindings,
                .groupCountX = 8, .groupCountY = 8, .groupCountZ = kCases}), "Dispatch");
            requireFeedback(gpu.submitAndWait(), "Completion");
            output->invalidate(); feedback->invalidate();
            const auto* mapped = static_cast<const Float4*>(output->map()); requireFeedback(mapped != nullptr, "Output map");
            std::vector<Float4> actual(mapped, mapped + kCases * kPixels * 2); output->unmap();
            const auto* words = static_cast<const uint32_t*>(feedback->map()); requireFeedback(words != nullptr, "Feedback map");
            std::array<uint32_t, kCases * 8> result; std::memcpy(result.data(), words, sizeof(result)); feedback->unmap();
            bench::readbackEvidence(context, "texture-feedback-sampling.bin", std::span<const Float4>(actual));
            bench::readbackEvidence(context, "texture-feedback-records.bin", std::span<const uint32_t>(result));
            for (uint32_t test = 0; test < kCases; ++test) {
                const bool enabled = (test & 8) != 0, mip0 = test >= 16;
                uint32_t selected = 0;
                for (uint32_t pixel = 0; pixel < kPixels; ++pixel) {
                    const auto& sampled = actual[(test * kPixels + pixel) * 2];
                    const auto& state = actual[(test * kPixels + pixel) * 2 + 1];
                    const bool gate = enabled && (((pixel % 64 / 8) ^ (pixel / 64 / 8) ^ (test & 7)) & 7) == 0;
                    selected += gate;
                    requireFeedback(state[0] == float(gate) && state[1] == 1 && state[2] == float(mip0) && state[3] == 1,
                        "Gate, floor or RNG changed at case " + std::to_string(test) + " pixel " + std::to_string(pixel));
                    requireFeedback(sampled == Float4{0,1,0,1}, "Sampling floor was bypassed");
                }
                requireFeedback(selected == (enabled ? kPixels / 8 : 0), "Tile selection density");
                const auto minMip = result[test * 8 + 4], count = result[test * 8 + 5];
                requireFeedback(enabled ? (minMip == (mip0 ? 0u : 2u) && count == 9 * kPixels / 8) : (minMip == UINT32_MAX && count == 0),
                    "Demand or Mip0 gate changed at case " + std::to_string(test));
                for (uint32_t word : {0u, 1u, 2u, 3u, 6u, 7u}) {
                    requireFeedback(result[test * 8 + word] == initialFeedback[test * 8 + word], "Read-only metadata changed");
                }
            }
            return RHITestResult::pass("48 frame/capture/regular-Mip0-alpha cases: tile gates, demand suppression, residency floor and RNG preserved");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(TextureFeedbackSamplingTest);
} // namespace
} // namespace metallic::tests
