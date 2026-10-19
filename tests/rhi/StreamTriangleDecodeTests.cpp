#include "RHITest.h"
#include "TestComputeProgram.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using UInt4 = std::array<uint32_t, 4>;

struct DecodeFixture
{
    std::string name;
    std::vector<uint32_t> words;
    uint32_t cluster = 0, triangle = 1, state = 2, expected = 1;
};

DecodeFixture makeDecodeFixture(uint32_t alignment, uint32_t positionFormat, uint32_t optionalMask,
    bool compact, uint32_t cluster)
{
    DecodeFixture fixture;
    fixture.name = "align" + std::to_string(alignment) + "-position" + std::to_string(positionFormat) +
        "-optional" + std::to_string(optionalMask) + (compact ? "-packed" : "-float") + "-cluster" + std::to_string(cluster);
    fixture.cluster = cluster;
    const uint32_t triangleOffset = 16u * cluster + ((alignment + 1u) & 3u);
    // The selected triangle is the last payload. Alignment 0/1 must not read a
    // second uint; alignment 2/3 has exactly one additional uint available.
    const uint32_t bytes = (640u + triangleOffset + 6u + 3u) & ~3u;
    fixture.words.resize(bytes / 4u);
    auto& words = fixture.words;
    words[2] = 2; words[9] = 112; words[10] = 304; words[11] = 640; words[12] = bytes;
    const uint32_t attributes = ((optionalMask & 1u) ? 2u : 0u) | ((optionalMask & 2u) ? 4u : 0u) |
        ((optionalMask & 4u) ? 16u : 0u);
    words[14] = 1u | attributes;
    words[16] = 400; words[17] = 496; words[20] = positionFormat;
    words[21] = compact ? 5u : 2u; words[22] = 1; words[24] = 544; words[25] = compact ? 6u : 2u;
    fixture.expected = 1u | (attributes << 1u);
    for (uint32_t c = 0; c < 2; ++c) {
        const uint32_t base = 28u + c * 24u;
        words[base] = c + 1; words[base + 1] = 4;
        words[base + 2] = c == cluster ? triangleOffset : 0u;
        words[base + 3] = 2;
    }
    const auto storeFloat = [&](uint32_t word, float value) { words[word] = std::bit_cast<uint32_t>(value); };
    for (uint32_t v = 0; v < 6; ++v) {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            storeFloat(76u + v * (positionFormat == 4u ? 3u : 4u) + lane, float(v * 3u + lane) * 0.125f - 2.0f);
        }
        if (compact) {
            words[100u + v] = 0x12345678u + v * 0x03050911u;
            words[136u + v] = 0x89abcdefu + v * 0x02040713u;
        } else {
            for (uint32_t lane = 0; lane < 4; ++lane) {
                storeFloat(100u + v * 4u + lane, float(v + lane) * 0.0625f - 0.5f);
                storeFloat(136u + v * 4u + lane, lane == 3 ? (v & 1u ? -1.0f : 1.0f) : float(v * 4u + lane) * 0.03125f);
            }
        }
        storeFloat(124u + v * 2u, float(v) * 0.25f - 0.5f);
        storeFloat(125u + v * 2u, float(v) * -0.125f + 1.0f);
    }
    const auto byte = [&](uint32_t address, uint32_t value) {
        const uint32_t shift = (address & 3u) * 8u;
        words[address / 4u] = (words[address / 4u] & ~(255u << shift)) | (value << shift);
    };
    for (uint32_t i = 0; i < 6; ++i) {
        constexpr uint32_t indices[] = {0, 2, 1, 1, 3, 0};
        byte(640u + triangleOffset + i, indices[i]);
    }
    return fixture;
}

class StreamTriangleDecodeTest final : public RHITest
{
public:
    StreamTriangleDecodeTest() { type = RHITestType::Rendering; name = "stream_triangle_decode_context_equivalence"; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::vector<DecodeFixture> fixtures;
            for (uint32_t alignment = 0; alignment < 4; ++alignment) {
                for (uint32_t format : {2u, 4u}) {
                    for (uint32_t attributes = 0; attributes < 8; ++attributes) {
                        for (bool compact : {false, true}) {
                            for (uint32_t cluster = 0; cluster < 2; ++cluster) {
                                fixtures.push_back(makeDecodeFixture(alignment, format, attributes, compact, cluster));
                            }
                        }
                    }
                }
            }
            auto mutate = [&](const char* name, uint32_t expected, const std::function<void(DecodeFixture&)>& change) {
                auto fixture = makeDecodeFixture(3, 4, 7, true, 1);
                fixture.name = name; fixture.expected = expected; change(fixture); fixtures.push_back(std::move(fixture));
            };
            constexpr uint32_t allAttributes = 2u | 4u | 16u;
            mutate("normal-range", 1u | ((allAttributes & ~2u) << 1u), [](auto& f) { f.words[16] = f.words[12] - 4u; });
            mutate("normal-alignment", 1u | ((allAttributes & ~2u) << 1u), [](auto& f) { f.words[16] += 1; });
            mutate("normal-format", 1u | ((allAttributes & ~2u) << 1u), [](auto& f) { f.words[21] = 77; });
            mutate("uv-range", 1u | ((allAttributes & ~4u) << 1u), [](auto& f) { f.words[17] = f.words[12] + 4u; });
            mutate("uv-alignment", 1u | ((allAttributes & ~4u) << 1u), [](auto& f) { f.words[17] += 2; });
            mutate("uv-format", 1u | ((allAttributes & ~4u) << 1u), [](auto& f) { f.words[22] = 0; });
            mutate("tangent-range", 1u | ((allAttributes & ~16u) << 1u), [](auto& f) { f.words[24] = f.words[12] - 4u; });
            mutate("tangent-alignment", 1u | ((allAttributes & ~16u) << 1u), [](auto& f) { f.words[24] += 3; });
            mutate("tangent-format", 1u | ((allAttributes & ~16u) << 1u), [](auto& f) { f.words[25] = 77; });
            mutate("resident-state-three", 1u | (allAttributes << 1u), [](auto& f) { f.state = 3; });
            mutate("nonresident", 0, [](auto& f) { f.state = 0; });
            mutate("position-format", 0, [](auto& f) { f.words[20] = 0; });
            mutate("position-flag", 0, [](auto& f) { f.words[14] &= ~1u; });
            mutate("position-alignment", 0, [](auto& f) { f.words[10] += 1; });
            mutate("cluster-alignment", 0, [](auto& f) { f.words[9] += 1; });
            mutate("cluster-range", 0, [](auto& f) { f.cluster = 2; });
            mutate("cluster-storage", 0, [](auto& f) { f.words[9] = f.words[12] - 4u; });
            mutate("triangle-range", 0, [](auto& f) { f.triangle = 2; });
            mutate("triangle-storage", 0, [](auto& f) { f.words[11] = f.words[12]; });
            mutate("triangle-offset", 0, [](auto& f) { f.words[54] = 0xffffffffu; });
            mutate("vertex-offset", 0, [](auto& f) { f.words[52] = 0xffffffffu; });
            mutate("vertex-count", 0, [](auto& f) { f.words[53] = 0xffffffffu; });
            mutate("index-equals-count", 0, [](auto& f) {
                const uint32_t address = f.words[11] + f.words[54] + 3u;
                const uint32_t shift = (address & 3u) * 8u;
                f.words[address / 4u] = (f.words[address / 4u] & ~(255u << shift)) | (4u << shift);
            });
            mutate("short-page-header", 0, [](auto& f) { f.words[12] = 108; });
            mutate("page-exceeds-pool", 0, [](auto& f) { f.words[12] = 0xfffffffcu; });
            // Keep this valid align=1 page physically last in the descriptor;
            // a needless second-word load would cross the real buffer boundary.
            fixtures.push_back(makeDecodeFixture(1, 4, 0, false, 1));
            fixtures.back().name = "physical-buffer-tail-single-word";
            std::vector<uint32_t> pages(8, 0);
            std::vector<std::array<uint32_t, 2>> table;
            std::vector<UInt4> cases;
            std::vector<std::string> names;
            for (const auto& fixture : fixtures) {
                if (pages.size() & 1u) { pages.push_back(0); }
                const uint32_t pageIndex = uint32_t(table.size());
                table.push_back({uint32_t(pages.size() * 4u) | fixture.state, 0});
                pages.insert(pages.end(), fixture.words.begin(), fixture.words.end());
                cases.push_back({pageIndex, fixture.cluster, fixture.triangle, fixture.expected});
                names.push_back(fixture.name);
            }
            cases.push_back({uint32_t(table.size()), 0, 0, 0}); names.push_back("page-table-index-range");
            table.push_back({uint32_t(pages.size() * 4u) | 2u, 0});
            cases.push_back({uint32_t(table.size() - 1), 0, 0, 0}); names.push_back("physical-short-header");
            // The table grew above, so keep the explicit out-of-range case out.
            cases[cases.size() - 2][0] = uint32_t(table.size());
            const size_t decodeCaseCount = cases.size();
            const auto addressCase = [&](uint32_t id, uint32_t stride) {
                // Nonzero denominators are the helper's contract. Zero is
                // tested on streamRaySurface by MaterialCoverageProbe.
                cases.push_back({id, stride, id / stride, 0xffffffffu});
                names.push_back("cluster-address-id" + std::to_string(id) + "-stride" + std::to_string(stride));
            };
            for (uint32_t stride : {1u, 2u, 3u, 7u, 12u, 16u, 31u, 32u, 33u, 63u, 64u, 65u,
                     255u, 256u, 257u, 65535u, 65536u, 65537u, 0x7fffffffu, 0x80000000u, 0xfffffffeu, 0xffffffffu}) {
                for (uint32_t id : {0u, 1u, 30u, 31u, 32u, 33u, 63u, 64u, 65u, 65535u, 65536u,
                         0x7fffffffu, 0x80000000u, 0xfffffffdu, 0xfffffffeu, 0xffffffffu}) {
                    addressCase(id, stride);
                }
                addressCase(stride - 1u, stride);
                addressCase(stride, stride);
                if (stride != 0xffffffffu) { addressCase(stride + 1u, stride); }
                addressCase(0xffffffffu - 0xffffffffu % stride, stride);
            }
            uint32_t random = 0x3289abcdu;
            for (uint32_t i = 0; i < 4096u; ++i) {
                random = random * 1664525u + 1013904223u;
                const uint32_t id = random;
                random = random * 1664525u + 1013904223u;
                addressCase(id, random == 0u ? 1u : random);
            }
            const auto require = [](bool condition, const std::string& message) {
                if (!condition) { throw std::runtime_error(message); }
            };
            auto created = bench::createTestDevice(context, {.applicationName = "Stream triangle decode regression",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Bindless buffers unavailable"); }
            require(bool(created), "Device creation failed");
            auto& device = **created;
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "StreamTriangleDecodeProbe",
                .entryPointName = "streamTriangleDecodeProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
            require(bool(shader), "Shader compile failed: " + log);
            const ComputeResourceBindingDesc layout[] = {
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPages)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPageTable)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamHeader)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput)}};
            ComputeProgram program;
            const auto initialized = program.initialize(device, {.spirv = shader->spirv, .pushConstantSize = 8,
                .bindings = layout, .requiresRayQuery = false, .resourceParameterSize = sizeof(SceneResourceParameters)}, log);
            require(bool(initialized), std::string("Program failed: ") + toString(initialized) + " " + log);
            std::array<std::unique_ptr<Buffer>, 4> buffers;
            const size_t bytes[] = {pages.size() * 4, table.size() * 8, cases.size() * 16, cases.size() * 16};
            const uint32_t strides[] = {4, 8, 16, 16};
            const void* data[] = {pages.data(), table.data(), cases.data()};
            for (uint32_t i = 0; i < 4; ++i) {
                auto buffer = device.createBuffer({.size = bytes[i], .structureStride = strides[i], .usage = BufferUsageBits::Storage,
                    .memoryLocation = i == 3 ? MemoryLocation::HostReadback : MemoryLocation::HostUpload});
                require(bool(buffer), "Buffer creation failed"); buffers[i] = std::move(*buffer);
                if (i < 3) {
                    void* mapped = buffers[i]->map(); require(mapped != nullptr, "Upload map failed");
                    std::memcpy(mapped, data[i], bytes[i]); buffers[i]->flush(); buffers[i]->unmap();
                }
            }
            bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics));
            require(bool(gpu.initialize(device)), "Commands failed");
            const ComputeDispatchBinding bindings[] = {
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPages), .buffer = buffers[0].get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamPageTable), .buffer = buffers[1].get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, rayStreamHeader), .buffer = buffers[2].get()},
                {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput), .buffer = buffers[3].get()}};
            const std::array<uint32_t, 2> push{uint32_t(cases.size()), uint32_t(pages.size())};
            require(bool(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = bindings, .pushData = push.data(),
                .pushDataSize = sizeof(push), .groupCountX = (push[0] + 63u) / 64u})), "Dispatch failed");
            require(bool(gpu.submitAndWait()), "GPU execution failed");
            buffers[3]->invalidate();
            const void* mapped = buffers[3]->map(); require(mapped != nullptr, "Readback map failed");
            std::vector<UInt4> actual(cases.size()); std::memcpy(actual.data(), mapped, bytes[3]); buffers[3]->unmap();
            bench::readbackEvidence(context, "readback.bin", std::span<const UInt4>(actual));
            std::ofstream report(context.outputDirectory / "StreamTriangleDecode.txt");
            require(bool(report), "Evidence write failed");
            for (size_t i = 0; i < actual.size(); ++i) {
                report << names[i] << " failures=" << actual[i][0] << " valid=" << actual[i][1] << " attributes=" << actual[i][2] << '\n';
                require(actual[i][0] == 0, names[i] + " decode mismatch bits=" + std::to_string(actual[i][0]));
                if (i >= decodeCaseCount) {
                    require(actual[i][1] == cases[i][0] / cases[i][1] && actual[i][2] == cases[i][0] % cases[i][1],
                        names[i] + " CPU quotient/remainder mismatch");
                }
            }
            return RHITestResult::pass(std::to_string(decodeCaseCount) + " synthetic cooked-page cases and " +
                std::to_string(cases.size() - decodeCaseCount) + " production cluster-address cases: exact attributes, flags, packed indices, quotient/remainder and cleared context");
        } catch (const std::exception& error) {
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamTriangleDecodeTest);
} // namespace
} // namespace metallic::tests
