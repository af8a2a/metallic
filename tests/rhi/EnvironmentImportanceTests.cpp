#include "RHITest.h"
#include "TestComputeProgram.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

void require(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

struct ImportanceFixture
{
    const char* name;
    uint32_t width;
    uint32_t height;
    uint32_t mipCount;
    uint32_t pattern;
    uint32_t baseMip = 0;
};

class EnvironmentImportanceTest final : public RHITest
{
public:
    EnvironmentImportanceTest()
    {
        name = "environment_importance_mip_dimensions";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            require(bool(bench::createTestDevice(context, {.applicationName = "Environment importance mip dimensions",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if (message.severity == ValidationSeverity::Error && hasFlag(message.type, ValidationCategory::Validation)) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}}).transform([&](auto value) { device = std::move(value); })), "Device creation failed");
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "EnvironmentImportanceProbe",
                .entryPointName = "environmentImportanceProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
            require(bool(shader), log);
            const ComputeResourceBindingDesc bindings[] = {
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, environmentPdf), ComputeResourceBindingKind::SampledImage}};
            ComputeProgram program;
            require(bool(program.initialize(*device, {.spirv = shader->spirv, .bindings = bindings,
                .requiresRayQuery = false, .resourceParameterSize = sizeof(SceneResourceParameters)}, log)), log);
            constexpr ImportanceFixture fixtures[] = {
                {"single", 1, 1, 1, 0}, {"single-zero", 1, 1, 1, 1},
                {"rectangular", 8, 4, 4, 0}, {"thin-x", 8, 1, 4, 0}, {"thin-y", 1, 8, 4, 0},
                {"odd", 7, 5, 3, 0}, {"odd-thin", 5, 1, 3, 0}, {"partial-chain", 8, 4, 2, 0},
                {"zero", 8, 4, 4, 1}, {"sparse", 8, 4, 4, 2}, {"negative", 7, 5, 3, 3},
                {"zero-intermediate", 8, 4, 4, 4}, {"view-base-mip", 7, 5, 3, 0, 1}};
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream report(context.outputDirectory / "EnvironmentImportance.txt");
            require(bool(report), "Could not write environment importance evidence");
            report << "256 seeds per fixture; exact GPU original/production/cached-entry-derived/cached-per-level result, texel, probability, RNG and weight comparisons.\n"
                << "Partial and NPOT chains preserve the original algorithm; this is not a normalization proof for those chains.\n";
            for (const auto& fixture : fixtures) {
                std::vector<float> texels;
                std::vector<size_t> offsets;
                for (uint32_t mip = 0; mip < fixture.mipCount; ++mip) {
                    const uint32_t width = std::max(fixture.width >> mip, 1u);
                    const uint32_t height = std::max(fixture.height >> mip, 1u);
                    offsets.push_back(texels.size());
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            float value = float(1 + (x * 3 + y * 5) % 11);
                            if (mip > 0) {
                                value = 0;
                                const uint32_t previousWidth = std::max(fixture.width >> (mip - 1), 1u);
                                const uint32_t previousHeight = std::max(fixture.height >> (mip - 1), 1u);
                                for (uint32_t dy = 0; dy < 2; ++dy) {
                                    for (uint32_t dx = 0; dx < 2; ++dx) {
                                        if (x * 2 + dx < previousWidth && y * 2 + dy < previousHeight) {
                                            value += std::max(texels[offsets[mip - 1] + (y * 2 + dy) * previousWidth + x * 2 + dx], 0.0f);
                                        }
                                    }
                                }
                                value *= 0.25f;
                            } else if (fixture.pattern == 2) {
                                value = x == 0 && y == 0 ? 8.0f : 0.0f;
                            } else if (fixture.pattern == 3 && (x + y) % 3 == 0) {
                                value = -value;
                            }
                            if (fixture.pattern == 1) { value = 0; }
                            texels.push_back(value);
                        }
                    }
                }
                // Fail after a successful top-level selection to check partial RNG advancement.
                if (fixture.pattern == 4) {
                    std::fill(texels.begin() + offsets[1], texels.begin() + offsets[2], 0.0f);
                }
                using UInt4 = std::array<uint32_t, 4>;
                constexpr uint32_t kLaneCount = 256;
                constexpr uint32_t kRowsPerLane = 8;
                std::unique_ptr<Buffer> upload, output;
                auto allocate = [&](size_t bytes, MemoryLocation location, BufferUsageBits usage, auto& target) {
                    require(bool(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = location})
                        .transform([&](auto value) { target = std::move(value); })), "Buffer creation failed");
                };
                allocate(texels.size() * sizeof(float), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
                allocate(kLaneCount * kRowsPerLane * sizeof(UInt4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
                void* mapped = upload->map();
                require(mapped != nullptr, "Upload mapping failed");
                std::memcpy(mapped, texels.data(), texels.size() * sizeof(float));
                upload->flush();
                upload->unmap();
                std::unique_ptr<Texture> texture;
                std::unique_ptr<TextureView> view;
                require(bool(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                    .format = Format::R32Sfloat, .width = fixture.width << fixture.baseMip,
                    .height = fixture.height << fixture.baseMip, .mipCount = fixture.mipCount + fixture.baseMip})
                    .transform([&](auto value) { texture = std::move(value); })), "Texture creation failed");
                require(bool(device->createTextureView(*texture, {.format = Format::R32Sfloat,
                    .range = {.baseMip = fixture.baseMip, .mipCount = fixture.mipCount}})
                    .transform([&](auto value) { view = std::move(value); })), "Texture view creation failed");
                bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                require(bool(gpu.initialize(*device)), "GPU commands failed");
                const TextureBarrierDesc before{.texture = texture.get(), .oldLayout = TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .range = {.baseMip = fixture.baseMip, .mipCount = fixture.mipCount}};
                require(bool(gpu.commands->synchronize({.textures = {&before, 1}})), "Upload barrier failed");
                for (uint32_t mip = 0; mip < fixture.mipCount; ++mip) {
                    require(bool(upload->slice({offsets[mip] * sizeof(float)}).and_then([&](const auto& slice) {
                        return gpu.commands->copyBufferToTexture({.texture = texture.get(), .buffer = slice,
                            .width = std::max(fixture.width >> mip, 1u), .height = std::max(fixture.height >> mip, 1u),
                            .mipLevel = mip + fixture.baseMip});
                    })), "Mip upload failed");
                }
                const TextureBarrierDesc after{.texture = texture.get(), .oldLayout = TextureLayout::TransferDestination,
                    .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead},
                    .range = {.baseMip = fixture.baseMip, .mipCount = fixture.mipCount}};
                require(bool(gpu.commands->synchronize({.textures = {&after, 1}})), "Sampling barrier failed");
                TextureView* views[] = {view.get()};
                const ComputeDispatchBinding dispatch[] = {
                    {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput), .buffer = output.get()},
                    {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, environmentPdf), .textureViews = views}};
                require(bool(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = dispatch, .groupCountX = kLaneCount})), "Dispatch failed");
                require(bool(gpu.submitAndWait()), "GPU execution failed");
                output->invalidate();
                const auto* data = static_cast<const UInt4*>(output->map());
                require(data != nullptr, "Readback failed");
                std::array<UInt4, kLaneCount * kRowsPerLane> actual;
                std::memcpy(actual.data(), data, sizeof(actual));
                output->unmap();
                uint32_t successful = 0, outOfBounds = 0;
                for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
                    const auto* row = &actual[lane * kRowsPerLane];
                    const std::string where = std::string(fixture.name) + " lane=" + std::to_string(lane);
                    require(row[0] == row[1], where + " sample/texel/RNG mismatch");
                    require(row[5] == row[1], where + " cached-entry-derived sample/texel/RNG mismatch");
                    require(row[6] == row[1], where + " cached-per-level sample/texel/RNG mismatch");
                    require(row[2][0] == row[2][1] && row[2][2] == row[2][3], where + " probability bits mismatch");
                    require(row[7][0] == row[2][1], where + " cached-entry-derived probability bits mismatch");
                    require(row[7][1] == row[2][1], where + " cached-per-level probability bits mismatch");
                    require(row[3][0] == row[3][1] && row[3][0] == row[3][2], where + " weight bits mismatch");
                    require(row[7][2] == row[3][2], where + " cached texture weight bits mismatch");
                    const uint32_t mip = row[3][3];
                    const uint32_t width = std::max(fixture.width >> mip, 1u), height = std::max(fixture.height >> mip, 1u);
                    const bool inBounds = row[4][0] < width && row[4][1] < height;
                    const float expectedWeight = inBounds ? std::max(texels[offsets[mip] + row[4][1] * width + row[4][0]], 0.0f) : 0.0f;
                    require(row[3][0] == std::bit_cast<uint32_t>(expectedWeight), where + " uploaded weight/OOB mismatch");
                    const bool queryInBounds = row[4][2] < fixture.width && row[4][3] < fixture.height;
                    const float total = std::max(texels[offsets.back()], 0.0f) * float(fixture.width * fixture.height);
                    const float expectedProbability = queryInBounds && total > 0
                        ? std::max(texels[row[4][3] * fixture.width + row[4][2]], 0.0f) / total : 0.0f;
                    const float actualProbability = std::bit_cast<float>(row[2][2]);
                    require(std::isfinite(actualProbability) && std::abs(actualProbability - expectedProbability) <= 1e-6f,
                        where + " CPU probability oracle mismatch");
                    successful += row[0][0];
                    outOfBounds += !inBounds;
                }
                require(outOfBounds > 0, "Fixture did not exercise out-of-bounds weights");
                report << fixture.name << " dimensions=" << fixture.width << 'x' << fixture.height
                    << " mips=" << fixture.mipCount << " baseMip=" << fixture.baseMip
                    << " exactCases=" << kLaneCount << " variants=production,cached-entry-derived,cached-per-level success=" << successful
                    << " outOfBounds=" << outOfBounds << " mismatches=0\n";
                report.flush();
            }
            require(validationErrors == 0, "Vulkan validation errors");
            return RHITestResult::pass("3328 exact GPU cases against original for production, cached-entry-derived and cached-per-level: dimensions, sampling RNG, probabilities, OOB and negative weights");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(EnvironmentImportanceTest);
} // namespace
} // namespace metallic::tests
