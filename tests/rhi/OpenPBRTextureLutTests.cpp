#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Material/OpenPBRLutData.h"

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
constexpr uint32_t kLanes = 8 * 32768;
constexpr uint64_t kABI = 0x4f5042524c555401ull;
struct LutParams
{
    GPUBufferSpan output;
    std::array<GPUResourceHandle<ResourceViewKind::SampledImage>, 8> textures;
    GPUSamplerHandle linearClamp;
    uint32_t mode;
};
static_assert(sizeof(LutParams) == 52);

void check(bool ok, const std::string& message)
{
    if (!ok) { throw std::runtime_error(message); }
}
template<typename T> void check(const Result<T>& result, const std::string& message)
{
    check(bool(result), message);
}

Float4 payloadTexel(const openpbr::LutPayload& payload, uint32_t index)
{
    if (payload.format == Format::R16Unorm) {
        return {static_cast<const uint16_t*>(payload.pixels)[index] * (1.0f / 65535.0f), 0, 0, 1};
    }
    Float4 value;
    std::memcpy(value.data(), static_cast<const std::byte*>(payload.pixels) + index * sizeof(Float4), sizeof(value));
    return value;
}

uint32_t hash(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

// Independent double-precision interpolation oracle and local derivative bound.
// Hardware's sub-texel fraction is allowed 1/256 error per axis, plus UNORM
// conversion rounding. This is a filter precision contract, not BSDF equivalence.
std::pair<Float4, Float4> oracle(uint32_t id, uint32_t index, uint32_t mode)
{
    const auto& p = openpbr::kLutPayloads[id];
    const std::array size{p.width, p.height, p.depth};
    std::array<double, 3> uv{(index % 32 + 0.5) / 32, ((index / 32) % p.height + 0.5) / p.height,
        ((index / 1024) % p.depth + 0.5) / p.depth};
    if (mode) {
        // Mirror only coordinate generation's float rounding, not interpolation.
        for (uint32_t axis = 0; axis < 3; ++axis) { uv[axis] = float(hash(index * 3 + axis)) / 4294967296.0f; }
        if (index < 64) { uv = {double(index % 4) * 0.5 - 0.25, double((index / 4) % 4) * 0.5 - 0.25, double(index / 16) * 0.5 - 0.25}; }
        if (index == 64) { uv = {0, 0, 0}; }
        if (index == 65) { uv = {1, 1, 1}; }
    }
    std::array<uint32_t, 3> lo{}, hi{};
    std::array<double, 3> f{};
    for (uint32_t axis = 0; axis < 3; ++axis) {
        double t = std::clamp(uv[axis] * size[axis] - 0.5, 0.0, double(size[axis] - 1));
        lo[axis] = uint32_t(t); hi[axis] = std::min(lo[axis] + 1, size[axis] - 1); f[axis] = t - lo[axis];
    }
    std::array<Float4, 8> corners;
    for (uint32_t corner = 0; corner < 8; ++corner) {
        uint32_t x = corner & 1 ? hi[0] : lo[0], y = corner & 2 ? hi[1] : lo[1], z = corner & 4 ? hi[2] : lo[2];
        corners[corner] = payloadTexel(p, (z * p.height + y) * p.width + x);
    }
    Float4 expected{}, bound{};
    for (uint32_t c = 0; c < 4; ++c) {
        double sum = 0, gradient = 0;
        for (uint32_t corner = 0; corner < 8; ++corner) {
            double w = 1;
            for (uint32_t axis = 0; axis < 3; ++axis) { w *= corner & (1 << axis) ? f[axis] : 1 - f[axis]; }
            sum += w * corners[corner][c];
        }
        for (uint32_t axis = 0; axis < 3; ++axis) {
            double derivative = 0;
            for (uint32_t corner = 0; corner < 8; ++corner) {
                derivative = std::max(derivative, std::abs(double(corners[corner][c]) - corners[corner ^ (1 << axis)][c]));
            }
            gradient += derivative;
        }
        expected[c] = float(sum);
        bound[c] = float((mode ? gradient / 256 : 0) + 2e-5 * std::max(1.0, std::abs(sum)));
    }
    return {expected, bound};
}

class OpenPBRTextureLutTest final : public RHITest
{
public:
    OpenPBRTextureLutTest() { type = RHITestType::Rendering; name = "material_openpbr_texture_luts"; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::atomic_uint errors{0};
            bench::TestDevice device;
            check(bench::createTestDevice(context, {.applicationName = "OpenPBR texture LUTs",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if ((message.severity == render::ValidationSeverity::Error) &&
                        (render::hasFlag(message.type, render::ValidationCategory::Validation))) { ++*static_cast<std::atomic_uint*>(target); }
                }, &errors}}).transform([&](auto value) { device = std::move(value); }), "Create device");
            ResourceRegistry registry;
            check(registry.initialize(*device), "Registry");
            std::array<std::array<std::unique_ptr<Texture>, 8>, 2> textures;
            std::array<std::array<std::unique_ptr<TextureView>, 8>, 2> views;
            for (uint32_t side = 0; side < 2; ++side) {
                for (uint32_t id = 0; id < 8; ++id) {
                    const auto& payload = openpbr::kLutPayloads[id];
                    std::vector<Float4> expanded;
                    const void* bytes = payload.pixels;
                    uint64_t byteSize = payload.byteSize;
                    Format format = payload.format;
                    if (side == 0) {
                        for (uint32_t i = 0; i < payload.width * payload.height * payload.depth; ++i) { expanded.push_back(payloadTexel(payload, i)); }
                        bytes = expanded.data(); byteSize = expanded.size() * sizeof(Float4); format = Format::RGBA32Sfloat;
                    }
                    std::unique_ptr<Buffer> upload;
                    check(device->createBuffer({.size = byteSize, .usage = BufferUsageBits::TransferSource,
                        .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto v) { upload = std::move(v); }), "Upload allocation");
                    void* mapped = upload->map(); check(mapped != nullptr, "Map upload");
                    std::memcpy(mapped, bytes, byteSize); upload->flush(); upload->unmap();
                    check(device->createTexture({.type = payload.depth > 1 ? TextureType::Texture3D : TextureType::Texture2D,
                        .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination, .format = format,
                        .width = payload.width, .height = payload.height, .depth = payload.depth, .memoryLocation = MemoryLocation::Device})
                        .transform([&](auto v) { textures[side][id] = std::move(v); }), "LUT texture");
                    check(device->createTextureView(*textures[side][id], {.format = format})
                        .transform([&](auto v) { views[side][id] = std::move(v); }), "LUT view");
                    bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                    check(gpu.initialize(*device), "Upload commands");
                    TextureBarrierDesc barrier{.texture = textures[side][id].get(), .oldLayout = TextureLayout::Undefined,
                        .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
                    check(gpu.commands->synchronize({.textures = {&barrier, 1}}), "Upload barrier");
                    check(upload->slice().and_then([&](const auto& slice) {
                        return gpu.commands->copyBufferToTexture({.texture = textures[side][id].get(), .buffer = slice,
                            .width = payload.width, .height = payload.height, .depth = payload.depth});
                    }), "LUT copy");
                    barrier.oldLayout = TextureLayout::TransferDestination; barrier.newLayout = TextureLayout::ShaderRead;
                    barrier.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite};
                    barrier.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
                    check(gpu.commands->synchronize({.textures = {&barrier, 1}}), "LUT visibility");
                    check(gpu.submitAndWait(), "Upload submission");
                }
            }
            std::unique_ptr<Buffer> output, readback;
            check(device->createBuffer({.size = kLanes * sizeof(Float4), .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::Device}).transform([&](auto v) { output = std::move(v); }), "Output");
            check(device->createBuffer({.size = kLanes * sizeof(Float4), .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto v) { readback = std::move(v); }), "Readback");
            std::filesystem::create_directories(context.outputDirectory);
            const auto save = [&](std::string name, const void* data, size_t size) {
                std::ofstream file(context.outputDirectory / name, std::ios::binary);
                file.write(static_cast<const char*>(data), size); check(bool(file), "Save " + name);
            };
            std::array<ComputeKernel, 2> kernels;
            for (uint32_t side = 0; side < 2; ++side) {
                const SlangMacroDefine define{"METALLIC_OPENPBR_FILTERED", side ? "1" : "0"};
                std::string log;
                auto shader = compileSlangShaderToSpirv({.moduleName = "OpenPBRTextureLutProbe", .entryPointName = "openPBRTextureLutMain",
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = {&define, 1}}, {.enableDiskCache = false}, log);
                check(shader, log);
                save(side ? "Filtered.spv" : "Manual.spv", shader->spirv.data(), shader->spirv.size() * sizeof(uint32_t));
                check(kernels[side].initialize(*device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<LutParams>(kABI, ParameterTransport::InlinePush)}, log), log);
            }
            std::unique_ptr<TimestampQueryPool> times;
            check(device->createTimestampQueryPool(*device->getQueue(QueueType::Graphics), {.queryCount = 2})
                .transform([&](auto v) { times = std::move(v); }), "Timestamps");
            std::ofstream timing(context.outputDirectory / "TextureLutTiming.csv"), report(context.outputDirectory / "TextureLut.txt");
            timing << "mode,block,side,warmup,validation,dispatches,gpu_ms\n";
            for (uint32_t mode : {0u, 1u}) {
                std::vector<Float4> expectedValues(kLanes), filterBounds(kLanes);
                for (uint32_t lane = 0; lane < kLanes; ++lane) {
                    auto [value, bound] = oracle(lane % 8, lane / 8, mode);
                    expectedValues[lane] = value; filterBounds[lane] = bound;
                }
                save("Oracle-" + std::to_string(mode) + ".bin", expectedValues.data(), expectedValues.size() * sizeof(Float4));
                save("Bound-" + std::to_string(mode) + ".bin", filterBounds.data(), filterBounds.size() * sizeof(Float4));
                for (uint32_t block = 0; block < 10; ++block) {
                    for (uint32_t order = 0; order < 2; ++order) {
                        uint32_t side = order ^ (block & 1);
                        ParameterWriter writer(*device, registry);
                        LutParams params{.output = writer.bufferSpan<Float4>(output.get()), .mode = mode};
                        for (uint32_t id = 0; id < 8; ++id) { params.textures[id] = writer.sampledImageHandle(views[side][id].get()); }
                        params.linearClamp = writer.samplerHandle(SamplerDesc{});
                        auto encoded = writer.encode(params, kABI, ParameterTransport::InlinePush); check(encoded, "Parameters");
                        check(times->reset(0, 2), "Reset timestamps");
                        bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                        check(gpu.initialize(*device), "Commands");
                        auto dispatch = kernels[side].prepareDispatch(*encoded, kLanes / 64);
                        check(dispatch, "Prepare dispatch");
                        check(gpu.commands->writeTimestamp(*times, 0, PipelineStageBits::AllCommands), "Begin timestamp");
                        for (uint32_t repeat = 0; repeat < 8; ++repeat) {
                            const MemoryBarrierDesc waw{.before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
                            if (repeat) { check(gpu.commands->synchronize({.memory = {&waw, 1}}), "WAW"); }
                            check(dispatch->record(*gpu.commands), "Dispatch");
                        }
                        check(gpu.commands->writeTimestamp(*times, 1, PipelineStageBits::AllCommands), "End timestamp");
                        if (block == 9) {
                            const MemoryBarrierDesc barrier{.before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                            check(gpu.commands->synchronize({.memory = {&barrier, 1}}), "Readback barrier");
                            auto src = output->slice(), dst = readback->slice(); check(src, "Source"); check(dst, "Destination");
                            check(gpu.commands->copyBuffer(*src, *dst), "Readback copy");
                        }
                        check(gpu.submitAndWait(), "Completion");
                        std::array<TimestampQueryResult, 2> results;
                        check(times->readResults(0, results), "Read timestamps");
                        check(results[0].available && results[1].available, "Unavailable timestamps");
                        timing << mode << ',' << block << ',' << side << ',' << (block < 2) << ',' << context.enableValidation << ",8,"
                               << times->durationMilliseconds(results[0].value, results[1].value) / 8 << '\n';
                        if (block == 9) {
                            readback->invalidate(); const auto* pixels = static_cast<const Float4*>(readback->map()); check(pixels != nullptr, "Map readback");
                            std::vector<Float4> copy(pixels, pixels + kLanes); readback->unmap();
                            save(std::string(side ? "Filtered-" : "Manual-") + std::to_string(mode) + ".bin", copy.data(), copy.size() * sizeof(Float4));
                            double maxError = 0;
                            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                                const auto& expected = expectedValues[lane];
                                const auto& bound = filterBounds[lane];
                                for (uint32_t c = 0; c < 4; ++c) {
                                    double error = std::abs(double(copy[lane][c]) - expected[c]);
                                    double limit = side ? bound[c] : 2e-5 * std::max(1.0f, std::abs(expected[c]));
                                    check(std::isfinite(copy[lane][c]) && error <= limit,
                                        "LUT mismatch lane=" + std::to_string(lane) + " channel=" + std::to_string(c) + " error=" + std::to_string(error) + " bound=" + std::to_string(limit));
                                    maxError = std::max(maxError, error / std::max(1.0, std::abs(double(expected[c]))));
                                }
                            }
                            report << "mode=" << mode << " side=" << side << " cases=" << kLanes << " maxScaledError=" << maxError << '\n';
                        }
                    }
                }
            }
            check(errors == 0, "Vulkan validation errors");
            return RHITestResult::pass("Eight production texture payloads: all texel centers, random interpolation and clamp boundaries match CPU oracle");
        } catch (const std::exception& e) { return RHITestResult::fail(e.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(OpenPBRTextureLutTest);
} // namespace
} // namespace metallic::tests
