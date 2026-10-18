#include "harness/Fixtures.h"
#include "TestComputeProgram.h"
#include "TestResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/TextureStreamingResources.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;
struct LodCase { std::array<uint32_t, 4> dimensions; Float4 constants; };
static_assert(sizeof(LodCase) == 32);
void requireLod(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}
template<typename T> void requireLod(const Result<T>& value, const std::string& message)
{
    requireLod(bool(value), message);
}
class TextureLodMetadataTest final : public RHITest
{
public:
    TextureLodMetadataTest() { type = RHITestType::Rendering; name = "texture_lod_metadata"; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            bench::TestDevice device;
            requireLod(bench::createTestDevice(context, {.applicationName = "Texture LOD metadata",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { device = std::move(value); }), "Device");
            constexpr std::array dimensions{std::array{8u,8u}, std::array{13u,7u}, std::array{1u,1024u},
                std::array{1024u,1u}, std::array{3u,1024u}};
            std::vector<LodCase> cases;
            std::vector<TextureSamplingMetadata> metadata;
            for (auto size : dimensions) {
                const uint32_t mipCount = std::bit_width(std::max(size[0], size[1]));
                for (uint32_t first = 0; first < mipCount; ++first) {
                    const auto width = std::max(size[0] >> first, 1u), height = std::max(size[1] >> first, 1u);
                    cases.push_back({{size[0], size[1], mipCount, first}, {makeTextureLodTerm(size[0], size[1]), 0.5f * std::log2(float(size[0]) * float(size[1])), 0, 0}});
                    metadata.push_back({first % 2 ? std::min(0.75f, float(mipCount - first - 1)) : 0.0f,
                        makeTextureLodTerm(width, height), width, height, mipCount - first, first});
                }
            }
            const auto count = uint32_t(cases.size());
            std::vector<std::unique_ptr<Texture>> textures(count);
            std::vector<std::unique_ptr<TextureView>> ownedViews(count);
            std::vector<TextureView*> views(count);
            std::vector<Float4> texels;
            std::vector<std::vector<uint64_t>> offsets(count);
            for (uint32_t index = 0; index < count; ++index) {
                const auto& state = metadata[index];
                requireLod(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                    .format = Format::RGBA32Sfloat, .width = state.residentWidth, .height = state.residentHeight,
                    .mipCount = state.residentMipCount}).transform([&](auto value) { textures[index] = std::move(value); }), "Texture");
                requireLod(device->createTextureView(*textures[index], {.format = Format::RGBA32Sfloat,
                    .range = {.mipCount = state.residentMipCount}}).transform([&](auto value) { ownedViews[index] = std::move(value); }), "View");
                views[index] = ownedViews[index].get();
                for (uint32_t mip = 0; mip < state.residentMipCount; ++mip) {
                    offsets[index].push_back(texels.size() * sizeof(Float4));
                    const auto pixels = std::max(state.residentWidth >> mip, 1u) * std::max(state.residentHeight >> mip, 1u);
                    texels.insert(texels.end(), pixels, Float4{float(mip + cases[index].dimensions[3]) / 16.0f, 0, 0, 1});
                }
            }
            std::unique_ptr<Buffer> upload, caseBuffer, metadataBuffer, output;
            auto allocate = [&](uint64_t bytes, MemoryLocation memory, BufferUsageBits usage, auto& target) {
                requireLod(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = memory})
                    .transform([&](auto value) { target = std::move(value); }), "Buffer");
            };
            allocate(texels.size() * sizeof(Float4), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
            allocate(cases.size() * sizeof(LodCase), MemoryLocation::HostUpload, BufferUsageBits::Storage, caseBuffer);
            allocate(metadata.size() * sizeof(TextureSamplingMetadata), MemoryLocation::HostUpload, BufferUsageBits::Storage, metadataBuffer);
            allocate(count * 64 * 3 * sizeof(Float4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
            auto write = [&](Buffer& buffer, const void* data, size_t size) {
                auto* mapped = buffer.map(); requireLod(mapped != nullptr, "Map");
                std::memcpy(mapped, data, size); buffer.flush(); buffer.unmap();
            };
            write(*upload, texels.data(), texels.size() * sizeof(Float4));
            write(*caseBuffer, cases.data(), cases.size() * sizeof(LodCase));
            write(*metadataBuffer, metadata.data(), metadata.size() * sizeof(TextureSamplingMetadata));
            std::string log;
            const SlangMacroDefine macros[] = {{"METALLIC_CUSTOM_MATERIALS", "0"}, {"METALLIC_REALTIME_DEFERRED", "0"}};
            auto shader = compileSlangShaderToSpirv({.moduleName = "TextureLodMetadataProbe", .entryPointName = "main",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = macros}, log);
            requireLod(shader, "Compile: " + log);
            const ComputeResourceBindingDesc layout[]{
                {METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, cases)},
                {METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, samplingMetadata)},
                {METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, output)},
                {METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, textures), ComputeResourceBindingKind::SampledImage, count},
                {METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, sampler), ComputeResourceBindingKind::Sampler}};
            ComputeProgram program;
            requireLod(program.initialize(*device, {.spirv = shader->spirv, .bindings = layout, .requiresRayQuery = false,
                .resourceParameterSize = sizeof(TextureLodProbeResources)}, log), "Program: " + log);
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            requireLod(gpu.initialize(*device), "Commands");
            gpu.commands->hostWriteBarrier();
            for (uint32_t index = 0; index < count; ++index) {
                const auto& state = metadata[index];
                const TextureBarrierDesc before{.texture = textures[index].get(), .oldLayout = TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .range = {.mipCount = state.residentMipCount}};
                requireLod(gpu.commands->synchronize({.textures = {&before,1}}), "Upload barrier");
                for (uint32_t mip = 0; mip < state.residentMipCount; ++mip) {
                    requireLod(upload->slice({offsets[index][mip]}).and_then([&](const auto& slice) {
                        return gpu.commands->copyBufferToTexture({.texture = textures[index].get(), .buffer = slice,
                            .width = std::max(state.residentWidth >> mip, 1u), .height = std::max(state.residentHeight >> mip, 1u), .mipLevel = mip});
                    }), "Upload mip");
                }
                const TextureBarrierDesc after{.texture = textures[index].get(), .oldLayout = TextureLayout::TransferDestination,
                    .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}, .range = {.mipCount = state.residentMipCount}};
                requireLod(gpu.commands->synchronize({.textures = {&after,1}}), "Sampling barrier");
            }
            const SamplerDesc sampler{.mipFilter = SamplerFilter::Linear, .addressU = SamplerAddressMode::Repeat, .addressV = SamplerAddressMode::Repeat};
            const ComputeDispatchBinding bindings[]{
                {.binding = METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, cases), .buffer = caseBuffer.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, samplingMetadata), .buffer = metadataBuffer.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, output), .buffer = output.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, textures), .textureViews = views},
                {.binding = METALLIC_RESOURCE_MEMBER(TextureLodProbeResources, sampler), .sampler = &sampler}};
            requireLod(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = bindings, .groupCountX = 1, .groupCountY = count}), "Dispatch");
            requireLod(gpu.submitAndWait(), "Completion");
            output->invalidate();
            const auto* mapped = static_cast<const Float4*>(output->map()); requireLod(mapped != nullptr, "Output map");
            std::vector<Float4> actual(mapped, mapped + count * 64 * 3); output->unmap();
            bench::readbackEvidence(context, "texture-lod-metadata.bin", std::span<const Float4>(actual));
            float maxLodError = 0, maxSampleError = 0, maxNaiveError = 0;
            uint32_t sourceFloorDifferences = 0, residentFloorDifferences = 0, maxUlp = 0;
            bench::Json differences = bench::Json::array();
            for (uint32_t index = 0; index < count * 64; ++index) {
                const auto& lod = actual[index * 3]; const auto& sample = actual[index * 3 + 1]; const auto& dims = actual[index * 3 + 2];
                const auto& state = metadata[index / 64];
                requireLod(dims[0] == float(state.residentWidth) && dims[1] == float(state.residentHeight) && dims[2] == float(state.residentMipCount), "Resident view dimensions differ");
                for (float value : lod) { requireLod(std::isfinite(value), "Nonfinite LOD"); }
                for (uint32_t component : {0u,2u}) {
                    maxLodError = std::max(maxLodError, std::abs(lod[component] - lod[component + 1]));
                    const auto a = std::bit_cast<uint32_t>(lod[component]), b = std::bit_cast<uint32_t>(lod[component + 1]);
                    maxUlp = std::max(maxUlp, a > b ? a - b : b - a);
                }
                const bool sourceDifferent = std::floor(lod[2]) != std::floor(lod[3]);
                const bool residentDifferent = std::floor(lod[0]) != std::floor(lod[1]);
                sourceFloorDifferences += sourceDifferent; residentFloorDifferences += residentDifferent;
                if (sourceDifferent || residentDifferent) {
                    differences.push_back({{"case", index / 64}, {"variant", index % 64}, {"dimensions", cases[index / 64].dimensions}, {"lod", lod}});
                }
                maxSampleError = std::max(maxSampleError, std::abs(sample[0] - sample[1]));
                maxNaiveError = std::max(maxNaiveError, std::abs(lod[0] - sample[2]));
            }
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream(context.outputDirectory / "TextureLodMetadata.json") << bench::Json{
                {"cases", count}, {"variants_per_case", 64}, {"max_lod_absolute_error", maxLodError}, {"max_lod_ulp", maxUlp},
                {"max_sample_absolute_error", maxSampleError}, {"naive_source_minus_first_max_error", maxNaiveError},
                {"source_floor_differences", sourceFloorDifferences}, {"resident_floor_differences", residentFloorDifferences},
                {"floor_difference_records", differences}, {"qualification", "Exact sample and demand/resident-floor equality required; raw numerical differences are retained"}}.dump(2);
            requireLod(maxLodError <= 0.00001f, "LOD numerical tolerance exceeded");
            requireLod(sourceFloorDifferences == 0 && residentFloorDifferences == 0, "Source or resident mip floor changed");
            requireLod(maxSampleError == 0.0f, "Mip-color sample changed");
            requireLod(maxNaiveError > 1.0f, "Thin-texture regression case was not exercised");
            return RHITestResult::pass("PO2, NPOT and 1xN resident tails sampled with production helpers; integer-boundary mip floors and mip-color samples match exactly; ULP evidence retained");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(TextureLodMetadataTest);
} // namespace
} // namespace metallic::tests
