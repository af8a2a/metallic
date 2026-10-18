#include "harness/Fixtures.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/TextureStreamingResources.h"

#include <array>
#include <algorithm>
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
            constexpr uint32_t kFeedbackCases = 48, kFootprintCases = 64;
            constexpr uint32_t kCases = kFeedbackCases + 2 * kFootprintCases, kPixels = 64 * 64, kOutputs = 3;
            const std::array<Float4, 5> texels{{{1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {1,0,0,0}, {0,1,0,1}}};
            std::array<TextureSamplingMetadata, kCases> initialMetadata{};
            std::array<uint32_t, kCases> initialMip, initialHits{};
            initialMip.fill(UINT32_MAX);
            for (uint32_t slot = 0; slot < kCases; ++slot) {
                // Non-streamable or pinned slots sample normally but never record demand.
                const uint32_t flags = slot < kFeedbackCases && slot % 3u == 1u ? 0u : kTextureDemandEnabled;
                initialMetadata[slot] = {slot < kFeedbackCases ? 1.0f : 0.0f,
                    makeTextureLodTerm(2, 2), 2, 2, 2, flags | 2u};
            }
            std::unique_ptr<Buffer> upload, metadata, demandMip, demandHits, output;
            auto allocate = [&](size_t bytes, MemoryLocation memory, BufferUsageBits usage, auto& target) {
                requireFeedback(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = memory})
                    .transform([&](auto value) { target = std::move(value); }), "Buffer");
            };
            allocate(sizeof(texels), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
            allocate(sizeof(initialMetadata), MemoryLocation::HostReadback, BufferUsageBits::Storage, metadata);
            allocate(sizeof(initialMip), MemoryLocation::HostReadback, BufferUsageBits::Storage, demandMip);
            allocate(sizeof(initialHits), MemoryLocation::HostReadback, BufferUsageBits::Storage, demandHits);
            allocate(kCases * kPixels * kOutputs * sizeof(Float4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
            auto write = [&](Buffer& buffer, const void* data, size_t bytes) {
                auto* mapped = buffer.map(); requireFeedback(mapped != nullptr, "Upload map");
                std::memcpy(mapped, data, bytes); buffer.flush(); buffer.unmap();
            };
            write(*upload, texels.data(), sizeof(texels));
            write(*metadata, initialMetadata.data(), sizeof(initialMetadata));
            write(*demandMip, initialMip.data(), sizeof(initialMip));
            write(*demandHits, initialHits.data(), sizeof(initialHits));
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
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandMip)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandHits)},
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
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata), .buffer = metadata.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandMip), .buffer = demandMip.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandHits), .buffer = demandHits.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), .textureViews = views}};
            requireFeedback(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = bindings,
                .groupCountX = 8, .groupCountY = 8, .groupCountZ = kCases}), "Dispatch");
            requireFeedback(gpu.submitAndWait(), "Completion");
            output->invalidate(); metadata->invalidate(); demandMip->invalidate(); demandHits->invalidate();
            const auto* mapped = static_cast<const Float4*>(output->map()); requireFeedback(mapped != nullptr, "Output map");
            std::vector<Float4> actual(mapped, mapped + kCases * kPixels * kOutputs); output->unmap();
            std::array<uint32_t, kCases> mips, hits;
            auto read = [&](Buffer& buffer, void* target, size_t bytes) {
                auto* mapped = buffer.map(); requireFeedback(mapped != nullptr, "Readback map");
                std::memcpy(target, mapped, bytes); buffer.unmap();
            };
            read(*demandMip, mips.data(), sizeof(mips));
            read(*demandHits, hits.data(), sizeof(hits));
            std::array<TextureSamplingMetadata, kCases> finalMetadata;
            read(*metadata, finalMetadata.data(), sizeof(finalMetadata));
            requireFeedback(std::memcmp(initialMetadata.data(), finalMetadata.data(), sizeof(initialMetadata)) == 0,
                "Immutable sampling metadata changed");
            bench::readbackEvidence(context, "texture-feedback-sampling.bin", std::span<const Float4>(actual));
            bench::readbackEvidence(context, "texture-feedback-mips.bin", std::span<const uint32_t>(mips));
            bench::readbackEvidence(context, "texture-feedback-hits.bin", std::span<const uint32_t>(hits));
            for (uint32_t test = 0; test < kFeedbackCases; ++test) {
                const bool enabled = (test & 8) != 0, mip0 = test >= 16;
                uint32_t selected = 0;
                for (uint32_t pixel = 0; pixel < kPixels; ++pixel) {
                    const auto& sampled = actual[(test * kPixels + pixel) * kOutputs];
                    const auto& state = actual[(test * kPixels + pixel) * kOutputs + 1];
                    const bool gate = enabled && (((pixel % 64 / 8) ^ (pixel / 64 / 8) ^ (test & 7)) & 7) == 0;
                    selected += gate;
                    requireFeedback(state[0] == float(gate) && state[1] == 1 && state[2] == float(mip0) && state[3] == 1,
                        "Gate, floor or RNG changed at case " + std::to_string(test) + " pixel " + std::to_string(pixel));
                    requireFeedback(sampled == Float4{0,1,0,1}, "Sampling floor was bypassed");
                }
                requireFeedback(selected == (enabled ? kPixels / 8 : 0), "Tile selection density");
                const auto minMip = mips[test], count = hits[test];
                const bool demandEnabled = enabled && test % 3u != 1u;
                requireFeedback(demandEnabled ? (minMip == (mip0 ? 0u : 2u) && count == 9 * kPixels / 8) : (minMip == UINT32_MAX && count == 0),
                    "Demand, slot eligibility or Mip0 gate changed at case " + std::to_string(test));

            }
            constexpr std::array<float, 4> kConeWidths{0.125f, -0.5f, 0.0f, 2.0f};
            for (uint32_t test = 0; test < kFootprintCases; ++test) {
                const auto legacy = kFeedbackCases + test * 2, resolved = legacy + 1;
                const bool mip0 = (test & 16u) != 0u, absent = (test & 32u) != 0u;
                const uint32_t coneCase = test & 3u;
                const float expectedLod = (test & 4u) != 0u ? -0.75f
                    : 0.25f + std::log2(std::max(std::abs(kConeWidths[coneCase]), 0.000001f)
                        / (coneCase < 2u ? 1.0f : 0.25f));
                const uint32_t expectedMip = absent ? UINT32_MAX : mip0 ? 0u
                    : static_cast<uint32_t>(std::floor(std::clamp(expectedLod + 3.0f, 0.0f, 3.0f)));
                const uint32_t expectedHits = absent ? 0u : 9u * kPixels / 8u;
                requireFeedback(mips[legacy] == expectedMip && mips[resolved] == expectedMip
                    && hits[legacy] == expectedHits && hits[resolved] == expectedHits,
                    "Resolved footprint demand changed at case " + std::to_string(test));
                for (uint32_t pixel = 0; pixel < kPixels; ++pixel) {
                    const auto legacyIndex = (legacy * kPixels + pixel) * kOutputs;
                    const auto resolvedIndex = (resolved * kPixels + pixel) * kOutputs;
                    for (uint32_t field = 0; field < kOutputs; ++field) {
                        requireFeedback(actual[legacyIndex + field] == actual[resolvedIndex + field],
                            "Resolved footprint sample, RNG or metadata mismatch at case " + std::to_string(test)
                            + " pixel " + std::to_string(pixel) + " field " + std::to_string(field));
                    }
                    const auto& value = actual[resolvedIndex];
                    const auto& state = actual[resolvedIndex + 1];
                    const auto& footprint = actual[resolvedIndex + 2];
                    requireFeedback(state[2] == float(!absent) && state[3] == 1.0f
                        && std::isfinite(footprint[0]) && std::abs(footprint[0] - expectedLod) < 0.00001f,
                        "Footprint validity, gradient preservation or LOD failed at case " + std::to_string(test));
                    const uint32_t seed = static_cast<uint32_t>(state[0]) | (static_cast<uint32_t>(state[1]) << 16u);
                    requireFeedback((seed == 12345u + pixel) == (mip0 || absent), "Footprint sampling RNG consumption");
                    requireFeedback(absent ? value == Float4{-1,-1,-1,-1}
                        : (value == Float4{1,0,0,0} || value == Float4{0,1,0,1}), "Footprint texture or fallback sample");
                }
            }
            return RHITestResult::pass("48 feedback cases and 64 paired legacy/resolved footprint cases: cone LOD, anisotropic UV gradients, Mip0, absent slots, samples, feedback and RNG preserved");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(TextureFeedbackSamplingTest);
} // namespace
} // namespace metallic::tests
