#include "RHITest.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Material/RayMaterialQueue.h"
#include "harness/Fixtures.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <json.hpp>

namespace metallic::tests {
namespace {
using namespace render;
struct ProbeParameters
{
    GPUBufferSpan hits, programs, bins, indices, output, visits;
    uint32_t hitCount, programCount, queued, generationLo, generationHi, bounce;
};
constexpr uint64_t kProbeABI = 0x52415950524f0001ull;
static_assert(sizeof(ProbeParameters) == 96);
class RayMaterialExecutionTest final : public RHITest
{
public:
    RayMaterialExecutionTest()
    {
        name = "ray_material_execution_queue";
        type = RHITestType::Rendering;
    }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            auto check = [](bool ok, const std::string& reason) {
                if (!ok) {
                    throw std::runtime_error(reason);
                }
            };
            std::string log;
            std::array<MaterialProgramKey, 7> keys{};
            for (auto& key : keys) {
                key.definitionHash = 1;
                key.irHash = 7;
            }
            keys[2].irHash = 8; // same closure family, different program
            keys[3].domain = MaterialDomain::Fiber;
            keys[4].qualityProfile = 1;
            keys[5].targetCapabilities = 1;
            auto table = buildRayMaterialProgramTable(keys);
            check(table.programs.size() == 5 && table.materialPrograms == std::vector<uint32_t>{0, 0, 1, 2, 3, 4, 0},
                  "Full executable key classification failed");
            table.materialPrograms[6] = 99; // Deliberately corrupt mapping must be diagnosed on GPU.
            std::atomic_uint validationErrors{0};
            auto created = bench::createTestDevice(
                context,
                {.applicationName = "Unified Ray Materials",
                 .enableValidation = context.enableValidation,
                 .enableBindlessDescriptorHeap = true,
                 .validationSink = {[](void* pointer, const ValidationMessage& message) noexcept {
                                        if ((message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
                                            ++*static_cast<std::atomic_uint*>(pointer);
                                        }
                                    },
                                    &validationErrors}});
            check(bool(created), "Create ray material device");
            auto device = std::move(*created);
            ResourceRegistry registry;
            check(bool(registry.initialize(*device)), "Initialize registry");
            RayMaterialQueue queue;
            check(bool(queue.initialize(*device, log)), log);
            ComputeKernel reset, execute;
            check(bool(ShaderRegistry::instance().getComputeKernel(
                      *device,
                      {.moduleName = "RayMaterialProbe",
                       .entryPointName = "resetProbeMain",
                       .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
                      {.parameters = parameterAbi<ProbeParameters>(kProbeABI)}, reset, log)),
                  log);
            check(bool(ShaderRegistry::instance().getComputeKernel(
                      *device,
                      {.moduleName = "RayMaterialProbe",
                       .entryPointName = "executeProbeMain",
                       .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
                      {.parameters = parameterAbi<ProbeParameters>(kProbeABI)}, execute, log)),
                  log);
            auto buffer = [&](uint64_t size, MemoryLocation location) {
                auto result = device->createBuffer(
                    {.size = std::max(size, 16ull), .usage = BufferUsageBits::Storage, .memoryLocation = location});
                check(bool(result), "Create queue buffer");
                return std::move(*result);
            };
            auto upload = [&](Buffer& target, const void* data, size_t size) {
                void* mapped = target.map();
                check(mapped != nullptr, "Map queue input");
                if (size) {
                    std::memcpy(mapped, data, size);
                }
                target.flush();
                target.unmap();
            };
            auto read = [&](Buffer& target) {
                target.invalidate();
                void* mapped = target.map();
                check(mapped != nullptr, "Map queue output");
                std::vector<uint32_t> words(target.desc().size / 4);
                std::memcpy(words.data(), mapped, words.size() * 4);
                target.unmap();
                return words;
            };
            constexpr uint64_t generation = 0x0123456789abcdefull;
            const uint32_t programs = uint32_t(table.programs.size());
            uint32_t cases = 0, compared = 0;
            for (uint32_t count : {0u, 1u, 63u, 65u, 257u}) {
                std::vector<RayMaterialHitKey> hits(count);
                std::vector<uint32_t> expected(programs, 0);
                std::array<uint32_t, 4> expectedStatus{};
                for (uint32_t i = 0; i < count; ++i) {
                    auto& hit = hits[i];
                    hit = {i % 7, 0x80000000u + i * 7919u, uint32_t(generation), uint32_t(generation >> 32)};
                    if (i % 19 == 0) {
                        hit.materialIndex = UINT32_MAX;
                    }
                    else if (i % 23 == 0) {
                        ++hit.generationHi;
                    }
                    else if (i % 29 == 0) {
                        hit.materialIndex = 1000;
                    }
                    if (hit.materialIndex == UINT32_MAX) {
                        ++expectedStatus[0];
                    }
                    else if (hit.generationHi != uint32_t(generation >> 32)) {
                        ++expectedStatus[2];
                    }
                    else if (hit.materialIndex >= 7 || table.materialPrograms[hit.materialIndex] >= programs) {
                        ++expectedStatus[1];
                    }
                    else {
                        ++expected[table.materialPrograms[hit.materialIndex]];
                    }
                }
                auto hitBuffer = buffer(count * 16ull, MemoryLocation::HostUpload);
                auto mappings = buffer(table.materialPrograms.size() * 4, MemoryLocation::HostUpload);
                auto bins = buffer(programs * 16ull, MemoryLocation::HostReadback);
                auto indices = buffer(count * 4ull, MemoryLocation::HostReadback);
                auto cursors = buffer(programs * 4ull, MemoryLocation::Device);
                auto status = buffer(16, MemoryLocation::HostReadback);
                auto output = buffer(count * 48ull, MemoryLocation::HostReadback);
                auto visits = buffer(count * 4ull, MemoryLocation::HostReadback);
                upload(*hitBuffer, hits.data(), hits.size() * 16);
                upload(*mappings, table.materialPrograms.data(), table.materialPrograms.size() * 4);
                auto run = [&](bool queued, uint32_t capacity, uint32_t bounce) {
                    bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                    check(bool(gpu.initialize(*device)), "Commands");
                    ParameterWriter writer(*device, registry);
                    if (queued) {
                        check(bool(queue.classify(*gpu.commands, writer, *hitBuffer, *mappings, *bins, *indices,
                                                  *cursors, *status, count, 7, programs, capacity, generation)),
                              "Classify ray queue");
                    }
                    ProbeParameters params{writer.bufferSpan<RayMaterialHitKey>(hitBuffer.get()),
                                           writer.bufferSpan<uint32_t>(mappings.get()),
                                           writer.bufferSpan<RayMaterialBin>(bins.get()),
                                           writer.bufferSpan<uint32_t>(indices.get()),
                                           writer.bufferSpan<std::array<float, 4>>(output.get()),
                                           writer.bufferSpan<uint32_t>(visits.get()),
                                           count,
                                           programs,
                                           uint32_t(queued),
                                           uint32_t(generation),
                                           uint32_t(generation >> 32),
                                           bounce};
                    auto encoded = writer.encode(params, kProbeABI);
                    check(bool(encoded), "Encode probe");
                    const auto groups = std::max(1u, (count + 63) / 64);
                    check(bool(reset.dispatch(*gpu.commands, *encoded, groups)), "Reset outputs");
                    const std::array barriers{
                        BufferBarrierDesc{.buffer = output.get(),
                                          .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                                          .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}},
                        BufferBarrierDesc{.buffer = visits.get(),
                                          .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                                          .after = {PipelineStageBits::ComputeShader,
                                                    AccessBits::ShaderRead | AccessBits::ShaderWrite}}};
                    check(bool(gpu.commands->synchronize({.buffers = barriers})), "Probe barrier");
                    check(bool(execute.dispatch(*gpu.commands, *encoded, groups)), "Consume ray queue");
                    check(bool(gpu.submitAndWait()), "Submit queue");
                    return read(*output);
                };
                std::vector<uint32_t> firstBounce;
                for (uint32_t bounce : {1u, 4u}) {
                    auto reference = run(false, count, bounce);
                    auto actual = run(true, count, bounce);
                    check(actual == reference, "Queue ordering changed material eval/sample, path RNG or preparation");
                    auto actualBins = read(*bins), actualStatus = read(*status), actualVisits = read(*visits);
                    uint32_t offset = 0;
                    for (uint32_t p = 0; p < programs; ++p) {
                        check(actualBins[p * 4] == offset && actualBins[p * 4 + 1] == expected[p] &&
                                  actualBins[p * 4 + 2] == expected[p] && actualBins[p * 4 + 3] == 0,
                              "Queue bin prefix/count mismatch");
                        offset += expected[p];
                    }
                    check(std::equal(expectedStatus.begin(), expectedStatus.end(), actualStatus.begin()),
                          "Miss/invalid/stale diagnostics mismatch");
                    auto order = read(*indices);
                    std::vector<uint32_t> seen(count, 0);
                    for (uint32_t p = 0; p < programs; ++p) {
                        for (uint32_t j = 0; j < expected[p]; ++j) {
                            uint32_t index = order[actualBins[p * 4] + j];
                            check(index < count, "Queue index OOB");
                            check(table.materialPrograms[hits[index].materialIndex] == p && ++seen[index] == 1,
                                  "Wrong program or duplicate work");
                        }
                    }
                    for (uint32_t i = 0; i < count; ++i) {
                        check(actualVisits[i] == seen[i], "Ray evaluated more/less than once");
                        for (uint32_t c = 0; c < 12; ++c) {
                            if (c != 7) {
                                check(std::isfinite(std::bit_cast<float>(actual[i * 12 + c])),
                                      "Nonfinite ray material result");
                            }
                        }
                    }
                    if (bounce == 1) {
                        firstBounce = actual;
                    }
                    else if (count > 1) {
                        bool surfaceChanged = false, fiberChanged = false;
                        for (uint32_t i = 0; i < count; ++i) {
                            if (!seen[i]) { continue; }
                            // Compare scattering RGB only, excluding bounce/seed metadata.
                            const bool changed = actual[i*12] != firstBounce[i*12] ||
                                actual[i*12+1] != firstBounce[i*12+1] || actual[i*12+2] != firstBounce[i*12+2];
                            if (table.materialPrograms[hits[i].materialIndex] == 2) { fiberChanged |= changed; }
                            else { surfaceChanged |= changed; }
                        }
                        check(surfaceChanged && fiberChanged,
                            "Secondary footprint/wo did not change both Surface and Fiber scattering");
                    }
                    compared += offset;
                    ++cases;
                }
                // Capacity zero and a small queue must report every discarded hit.
                for (uint32_t capacity : {0u, std::min(count, 3u)}) {
                    run(true, capacity, 1);
                    auto actualBins = read(*bins), actualStatus = read(*status);
                    uint32_t total = 0, kept = 0, dropped = 0;
                    for (uint32_t p = 0; p < programs; ++p) {
                        total += expected[p];
                        kept += actualBins[p * 4 + 1];
                        dropped += actualBins[p * 4 + 3];
                        check(actualBins[p * 4] + actualBins[p * 4 + 1] <= capacity,
                              "Queue overflow wrote outside budget");
                    }
                    check(kept == std::min(total, capacity) && dropped == total - kept && actualStatus[3] == dropped,
                          "Silent ray queue truncation");
                    ++cases;
                }
            }
            std::ofstream(context.outputDirectory / "RayMaterialMetrics.json") << nlohmann::json{
                {"queueCases", cases},
                {"evaluatedCompared", compared},
                {"programs", programs},
                {"bitwiseEqual", true}}.dump(2);
            check(validationErrors == 0, "Ray material queue caused Vulkan validation errors");
            return RHITestResult::pass("Full-key Surface/Fiber bins, fresh ray contexts, path-owned RNG, queue/fused "
                                       "equality, miss/stale/invalid/overflow and empty-tail cases");
        }
        catch (const std::exception& e) {
            return RHITestResult::fail(e.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(RayMaterialExecutionTest);
} // namespace
} // namespace metallic::tests
