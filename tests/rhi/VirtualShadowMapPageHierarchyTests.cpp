#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

void checkHierarchy(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}
template<class T> void checkHierarchy(const Result<T>& result, const std::string& message)
{
    checkHierarchy(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct HierarchyProbePush
{
    uint32_t owners, statistics, queries, output;
    uint32_t axis, levels, physicalPages, queryCount;
    uint32_t lightType, reserved0, reserved1, reserved2;
};
static_assert(sizeof(HierarchyProbePush) == 48);

class VirtualShadowMapPageHierarchyTest final : public RHITest
{
public:
    VirtualShadowMapPageHierarchyTest()
    {
        name = "virtual_shadow_map_page_hierarchy";
        type = RHITestType::Rendering;
    }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapPageHierarchy");
        std::filesystem::create_directories(directory);
        Json report{{"passed", false}, {"cases", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapPageHierarchy.json") << report.dump(2); };
        try {
            auto created = createDevice({.applicationName = "VSM page hierarchy probe",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless mesh-capable Vulkan"); }
            checkHierarchy(created, "Hierarchy device");
            auto& device = **created;
            auto registered = ResourceRegistry::forDevice(device); checkHierarchy(registered, "Hierarchy registry");
            auto& registry = **registered;
            constexpr uint64_t abi = 0x56534d484d495031ull;
            const std::array capabilities{"spvMeshShadingEXT"};
            const std::array macros{SlangMacroDefine{"VSM_RECEIVER_MASK", "1"},
                SlangMacroDefine{"VSM_DIRTY_PAGE_HIERARCHY", "1"}, SlangMacroDefine{"VSM_CLUSTER_PAGE_BITSET", "1"},
                SlangMacroDefine{"VSM_WORKLOAD_DIAGNOSTICS", "1"}};
            const std::array entries{"pageHierarchyResetProbeMain", "pageHierarchyBuildProbeMain", "pageHierarchyQueryProbeMain"};
            const std::array paths{PROJECT_SOURCE_DIR "/Shaders", PROJECT_SOURCE_DIR "/Shaders/Features/VirtualShadowMaps"};
            std::array<ComputeKernel, 3> kernels;
            std::string log;
            for (uint32_t index = 0; index < kernels.size(); ++index) {
                auto shader = compileSlangShaderToSpirv({.moduleName = "VirtualShadowMapPageHierarchyProbe",
                    .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .additionalSearchPaths = paths,
                    .capabilities = capabilities, .macroDefines = macros}, log);
                checkHierarchy(shader, "Hierarchy compile: " + log);
                checkHierarchy(kernels[index].initialize(device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<HierarchyProbePush>(abi, ParameterTransport::InlinePush)}, log), "Hierarchy pipeline: " + log);
            }
            uint64_t testedQueries = 0, falsePositives = 0;
            for (uint32_t axis : {1u, 2u, 3u, 7u, 31u, 32u, 33u, 127u, 128u}) {
                for (uint32_t lightType : {0u, 1u}) {
                    for (uint32_t pattern = 0; pattern < 4; ++pattern) {
                        constexpr uint32_t levels = 3, physicalPages = 2, sentinel = 0xcdcdcdcdu;
                        const uint32_t stride = axis * axis, totalPages = levels * stride + 1u;
                        const uint32_t bitsetOffset = physicalPages * 2u + totalPages + (totalPages + 255u) / 256u;
                        const uint32_t bitsetWords = (totalPages + 31u) / 32u;
                        const uint32_t hierarchyOffset = bitsetOffset + bitsetWords + totalPages * 2u;
                        const uint32_t paddedAxis = std::bit_ceil(axis), levelWords = 4u + (paddedAxis * paddedAxis - 1u) / 3u;
                        std::vector<uint32_t> owners(hierarchyOffset + levels * levelWords + 4u, sentinel);
                        std::fill(owners.begin() + bitsetOffset, owners.begin() + hierarchyOffset, 0u);
                        std::vector<uint8_t> dirty(levels * stride, 0);
                        std::vector<std::array<uint32_t, 4>> queryWords;
                        std::vector<bool> oracle;
                        uint32_t random = 0x937f32adu;
                        const auto next = [&]() { random ^= random << 13; random ^= random >> 17; random ^= random << 5; return random; };
                        for (uint32_t level = 0; level < levels; ++level) {
                            const uint32_t localAxis = lightType == 0 ? axis : std::max(1u, axis >> level);
                            for (uint32_t y = 0; y < localAxis; ++y) {
                                for (uint32_t x = 0; x < localAxis; ++x) {
                                    const bool hit = pattern == 3 || (pattern == 1 && x == localAxis / 2u && y == localAxis / 2u) ||
                                        (pattern == 2 && (x + y * 19u + level * 3u) % 37u == 0u);
                                    const uint32_t page = level * stride + y * localAxis + x;
                                    dirty[page] = uint8_t(hit);
                                    if (hit) { owners[bitsetOffset + page / 32u] |= 1u << (page % 32u); }
                                }
                            }
                            const auto add = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
                                queryWords.push_back({x0, y0, x1, y1}); queryWords.push_back({level, 0, 0, 0});
                                bool hit = false;
                                for (uint32_t y = y0; y <= y1; ++y) {
                                    for (uint32_t x = x0; x <= x1; ++x) { hit |= dirty[level * stride + y * localAxis + x] != 0; }
                                }
                                oracle.push_back(hit);
                            };
                            add(0, 0, localAxis - 1u, localAxis - 1u);
                            for (uint32_t i = 0; i < localAxis; ++i) {
                                add(i, 0, i, localAxis - 1u); add(0, i, localAxis - 1u, i);
                                add(i, i, i, i);
                            }
                            if (localAxis <= 7) {
                                for (uint32_t y0 = 0; y0 < localAxis; ++y0) {
                                    for (uint32_t x0 = 0; x0 < localAxis; ++x0) {
                                        for (uint32_t y1 = y0; y1 < localAxis; ++y1) {
                                            for (uint32_t x1 = x0; x1 < localAxis; ++x1) { add(x0, y0, x1, y1); }
                                        }
                                    }
                                }
                            } else {
                                for (uint32_t i = 0; i < 256; ++i) {
                                    uint32_t x0 = next() % localAxis, x1 = next() % localAxis;
                                    uint32_t y0 = next() % localAxis, y1 = next() % localAxis;
                                    add(std::min(x0, x1), std::min(y0, y1), std::max(x0, x1), std::max(y0, y1));
                                }
                            }
                        }
                        std::vector<uint32_t> stats((128u + kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize) / 4u, 0u);
                        stats[16] = uint32_t(std::ranges::count(dirty, uint8_t(1)));
                        std::vector<std::array<uint32_t, 4>> output(oracle.size() * 2u + 1u, {sentinel, sentinel, sentinel, sentinel});
                        const std::array<uint64_t, 4> sizes{owners.size() * 4u, stats.size() * 4u, queryWords.size() * 16u, output.size() * 16u};
                        const std::array strides{4u, 4u, 16u, 16u};
                        const std::array<const void*, 4> initial{owners.data(), stats.data(), queryWords.data(), output.data()};
                        std::array<std::unique_ptr<Buffer>, 4> buffers;
                        for (uint32_t i = 0; i < buffers.size(); ++i) {
                            auto result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
                                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
                            checkHierarchy(result, "Hierarchy buffer"); buffers[i] = std::move(*result);
                            void* mapped = buffers[i]->map(); checkHierarchy(mapped != nullptr, "Hierarchy upload map");
                            std::memcpy(mapped, initial[i], sizes[i]); buffers[i]->flush(); buffers[i]->unmap();
                        }
                        ParameterWriter writer(device, registry);
                        const HierarchyProbePush push{writer.buffer(buffers[0].get()).index, writer.buffer(buffers[1].get()).index,
                            writer.buffer(buffers[2].get()).index, writer.buffer(buffers[3].get()).index,
                            axis, levels, physicalPages, uint32_t(oracle.size()), lightType, 0, 0, 0};
                        auto encoded = writer.encode(push, abi, ParameterTransport::InlinePush); checkHierarchy(encoded, "Hierarchy parameters");
                        bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics)); checkHierarchy(gpu.initialize(device), "Hierarchy commands");
                        gpu.commands->hostWriteBarrier();
                        checkHierarchy(kernels[0].dispatch(*gpu.commands, *encoded, (levels * levelWords + 63u) / 64u), "Hierarchy reset");
                        BufferBarrierDesc ready{.buffer = buffers[0].get(),
                            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
                        checkHierarchy(gpu.commands->synchronize({.buffers = {&ready, 1}}), "Hierarchy reset publication");
                        checkHierarchy(kernels[1].dispatch(*gpu.commands, *encoded, (bitsetWords + 63u) / 64u), "Hierarchy build");
                        ready.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
                        checkHierarchy(gpu.commands->synchronize({.buffers = {&ready, 1}}), "Hierarchy publication");
                        checkHierarchy(kernels[2].dispatch(*gpu.commands, *encoded, 1), "Hierarchy queries");
                        std::array<BufferBarrierDesc, 4> hostReady;
                        for (uint32_t i = 0; i < buffers.size(); ++i) {
                            hostReady[i] = {.buffer = buffers[i].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                                .after = {PipelineStageBits::Host, AccessBits::HostRead}};
                        }
                        checkHierarchy(gpu.commands->synchronize({.buffers = hostReady}), "Hierarchy readback");
                        checkHierarchy(gpu.submitAndWait(), "Hierarchy completion");
                        const auto read = [&](uint32_t i, void* destination) {
                            buffers[i]->invalidate(); const void* mapped = buffers[i]->map(); checkHierarchy(mapped != nullptr, "Hierarchy readback map");
                            std::memcpy(destination, mapped, sizes[i]); buffers[i]->unmap();
                        };
                        auto actualOwners = owners; read(0, actualOwners.data()); read(3, output.data());
                        uint32_t localFalsePositives = 0, maximumNodes = 0;
                        for (uint32_t i = 0; i < oracle.size(); ++i) {
                            const auto& actual = output[i * 2u];
                            checkHierarchy(!oracle[i] || actual[0] != 0, "Hierarchy false negative at query " + std::to_string(i));
                            if (pattern <= 1) { checkHierarchy((actual[0] != 0) == oracle[i], "Empty/single-page hierarchy query mismatch"); }
                            checkHierarchy(actual[1] <= 4u, "Hierarchy query exceeded four node reads");
                            checkHierarchy(actual[2] == levelWords && actual[3] == hierarchyOffset, "Hierarchy ABI/layout mismatch");
                            maximumNodes = std::max(maximumNodes, actual[1]);
                            localFalsePositives += !oracle[i] && actual[0] != 0 ? 1u : 0u;
                            const auto& rect = queryWords[i * 2u];
                            const uint32_t x0 = i & 7u, y0 = (i >> 3u) & 7u;
                            const uint32_t x1 = std::min(7u, x0 + ((rect[2] - rect[0]) & 7u));
                            const uint32_t y1 = std::min(7u, y0 + ((rect[3] - rect[1]) & 7u));
                            uint64_t mask = 0;
                            for (uint32_t y = y0; y <= y1; ++y) {
                                for (uint32_t x = x0; x <= x1; ++x) { mask |= uint64_t(1) << (y * 8u + x); }
                            }
                            const std::array<uint32_t, 4> expectedMask{uint32_t(mask), uint32_t(mask >> 32), uint32_t(mask), uint32_t(mask >> 32)};
                            checkHierarchy(output[i * 2u + 1u] == expectedMask, "Receiver rectangle/absolute-texel mask mismatch");
                        }
                        if (pattern != 0) { checkHierarchy(maximumNodes != 0, "Hierarchy diagnostics did not record reads"); }
                        for (uint32_t level = 0; level < levels; ++level) {
                            const uint32_t localAxis = lightType == 0 ? axis : std::max(1u, axis >> level);
                            const uint32_t base = hierarchyOffset + level * levelWords;
                            std::array<uint32_t, 4> bounds{UINT32_MAX, UINT32_MAX, 0u, 0u};
                            for (uint32_t y = 0; y < localAxis; ++y) {
                                for (uint32_t x = 0; x < localAxis; ++x) {
                                    if (dirty[level * stride + y * localAxis + x]) {
                                        bounds[0] = std::min(bounds[0], x); bounds[1] = std::min(bounds[1], y);
                                        bounds[2] = std::max(bounds[2], x); bounds[3] = std::max(bounds[3], y);
                                    }
                                }
                            }
                            for (uint32_t i = 0; i < 4; ++i) { checkHierarchy(actualOwners[base + i] == bounds[i], "Hierarchy tight bounds mismatch"); }
                            uint32_t offset = base + 4u, span = 2u;
                            for (uint32_t mipAxis = paddedAxis / 2u; mipAxis; mipAxis /= 2u, span *= 2u) {
                                for (uint32_t y = 0; y < mipAxis; ++y) {
                                    for (uint32_t x = 0; x < mipAxis; ++x) {
                                        bool occupied = false;
                                        for (uint32_t py = y * span; py < std::min((y + 1u) * span, localAxis); ++py) {
                                            for (uint32_t px = x * span; px < std::min((x + 1u) * span, localAxis); ++px) {
                                                occupied |= dirty[level * stride + py * localAxis + px] != 0;
                                            }
                                        }
                                        checkHierarchy(actualOwners[offset + y * mipAxis + x] == uint32_t(occupied), "Hierarchy OR mip mismatch");
                                    }
                                }
                                offset += mipAxis * mipAxis;
                            }
                        }
                        checkHierarchy(std::equal(owners.begin(), owners.begin() + hierarchyOffset, actualOwners.begin()), "Hierarchy modified allocation/bitset prefix");
                        checkHierarchy(std::ranges::all_of(std::span(actualOwners).last(4), [](auto value) { return value == sentinel; }) &&
                            output.back() == std::array<uint32_t, 4>{sentinel, sentinel, sentinel, sentinel}, "Hierarchy buffer overrun");
                        testedQueries += oracle.size(); falsePositives += localFalsePositives;
                        report["cases"].push_back({{"axis", axis}, {"lightType", lightType}, {"pattern", pattern},
                            {"queries", oracle.size()}, {"falsePositives", localFalsePositives}, {"maximumNodeReads", maximumNodes}});
                    }
                }
            }
            report["validation"] = validation.snapshot(); checkHierarchy(!validation.failed(), "Hierarchy Vulkan validation error");
            report["passed"] = true; report["testedQueries"] = testedQueries; report["falsePositives"] = falsePositives;
            report["scope"] = "Production hierarchy reset/build/query and receiver rectangle/texel masks. Conservative hierarchy false positives allowed; zero false negatives required.";
            save(); return RHITestResult::pass("Production HMip bounds/tree and conservative queries match independent dirty-page oracle; at most four node reads");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save(); return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapPageHierarchyTest);
} // namespace
} // namespace metallic::tests
