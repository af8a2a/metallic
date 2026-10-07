#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Material/MaterialClosureClassification.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <json.hpp>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;
using UInt2 = std::array<uint32_t, 2>;
using Json = nlohmann::json;
void check(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}
template<typename T> void require(const Result<T>& result, const char* message)
{
    check(bool(result), message);
}
struct SchedulingParams
{
    GPUBufferSpan visibility, packets, programTasks, control, familyIds, payload, familyTasks, output, visits;
    uint32_t width, height, programCount, slot, mode;
};
static_assert(sizeof(SchedulingParams) == 128);
constexpr uint64_t kSchedulingABI = 0x434c534348000001ull;

class MaterialClosureClassificationTest final : public RHITest
{
public:
    MaterialClosureClassificationTest() { name = "material_closure_family_classification"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext&) override
    {
        try {
            using F = MaterialClosureFamily;
            const std::array<std::optional<F>, 6> programs{std::nullopt, F::DualSlabClosure, F::OpenPBRCompositeClosure,
                F::SingleSlabClosure, F::DualSlabClosure, F::OpenPBRCompositeClosure};
            const auto table = MaterialClosureClassification::create(programs);
            check(table.programCount == 5 && table.closureFamilyCount == 3, "Program:family counts include background");
            check(table.programFamilyBins == std::vector<uint32_t>{0, 1, 2, 3, 1, 2} &&
                table.programOrder == std::vector<uint32_t>{0, 1, 4, 2, 5, 3} &&
                table.familyProgramOffsets == std::vector<uint32_t>{0, 1, 3, 5, 6}, "Family grouping lost original bin identity");
            const std::array<std::optional<F>, 1> background{std::nullopt};
            const auto empty = MaterialClosureClassification::create(background);
            check(empty.programCount == 0 && empty.closureFamilyCount == 0 && empty.programOrder == std::vector<uint32_t>{0},
                "Background-only schedule invalid");
            for (auto invalid : {std::vector<std::optional<F>>{}, {F::DualSlabClosure}, {std::nullopt, std::nullopt},
                    {std::nullopt, static_cast<F>(999)}}) {
                bool rejected = false;
                try { MaterialClosureClassification::create(invalid); } catch (const std::invalid_argument&) { rejected = true; }
                check(rejected, "Invalid family table accepted");
            }
            return RHITestResult::pass("Program/family ratios, stable grouping, background and invalid mappings");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialClosureClassificationTest);

class MaterialClosureSchedulingTest final : public RHITest
{
public:
    MaterialClosureSchedulingTest() { name = "material_closure_fused_split_ab"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            require(bench::createTestDevice(context, {.applicationName = "Closure scheduling A/B",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if ((message.severity == render::ValidationSeverity::Error) &&
                        (render::hasFlag(message.type, render::ValidationCategory::Validation))) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}}).transform([&](auto value) { device = std::move(value); }), "Device failed");
            if (!device->capabilities().computeSubgroupBallotArithmetic || device->capabilities().subgroupSize != 32 ||
                !device->capabilities().timestampQueries) { return RHITestResult::skip("Requires wave32 and GPU timestamps"); }
            ResourceRegistry registry;
            require(registry.initialize(*device), "Registry failed");
            std::filesystem::create_directories(context.outputDirectory);
            Json report{{"version", 1}, {"validation", context.enableValidation}, {"warmupPairs", 2}, {"measuredPairs", 6},
                {"scope", "prototype device-local material evaluation + optional closure write/classification + 16-light shading; readback excluded"},
                {"samples", Json::array()}, {"shaders", Json::array()}};
            const char* statistics = std::getenv("METALLIC_VK_PIPELINE_STATISTICS");
            report["pipelineStatistics"] = statistics && std::string_view(statistics) == "1";
            std::array<MaterialClosurePacket, 8> packets;
            std::vector<std::optional<MaterialClosureFamily>> families{std::nullopt};
            for (uint32_t p = 0; p < 8; ++p) {
                std::array<MaterialClosureNode, 3> nodes;
                nodes[0].slab.reflectance = {0.1f + 0.04f * p, 0.22f, 0.35f, 0};
                nodes[0].slab.opticalDepth = {0.1f, 0.2f, 0.3f, 0};
                nodes[1].slab.reflectance = {0.7f, 0.4f, 0.15f, 0};
                nodes[2].op = p % 2 ? MaterialClosureOp::Layer : MaterialClosureOp::Mix;
                nodes[2].operands = {0, 1}; nodes[2].weight = 0.35f;
                auto lowered = lowerMaterialClosure(MaterialClosureIR::create(nodes, 2));
                packets[p] = lowered.packet; families.push_back(lowered.family);
            }
            const auto classification = MaterialClosureClassification::create(families);
            check(classification.programCount == 8 && classification.closureFamilyCount == 1, "Distinct graphs did not share a family");
            std::array<ComputeKernel, 8> fused, evaluate, switching;
            ComputeKernel reset, programClassify, arguments, familyClassify, light;
            std::string log;
            std::set<uint64_t> fusedHashes;
            const auto compile = [&](const char* entry, uint32_t program, ComputeKernel& kernel) {
                const auto id = std::to_string(program);
                const SlangMacroDefine defines[] = {{"CLOSURE_PROGRAM", id.c_str()}};
                auto shader = compileSlangShaderToSpirv({.moduleName = "ClosureSchedulingProbe", .entryPointName = entry,
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = defines}, log);
                check(bool(shader), "Closure scheduling compile: " + log);
                uint64_t hash = 14695981039346656037ull;
                const auto* bytes = reinterpret_cast<const uint8_t*>(shader->spirv.data());
                for (size_t i = 0; i < shader->spirv.size() * 4; ++i) { hash = (hash ^ bytes[i]) * 1099511628211ull; }
                const auto label = std::string("ClosureSchedulingProbe.") + entry + ".P" + id;
                auto initialized = kernel.initialize(*device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<SchedulingParams>(kSchedulingABI, ParameterTransport::InlinePush), .debugName = label.c_str()}, log);
                check(bool(initialized), "Closure scheduling pipeline: " + log);
                report["shaders"].push_back({{"label", label}, {"inputSpirvFnv1a64", hash}, {"bytes", shader->spirv.size() * 4}});
                if (std::string_view(entry) == "fusedMain") { fusedHashes.insert(hash); }
            };
            for (uint32_t p = 0; p < 8; ++p) {
                compile("fusedMain", p, fused[p]); compile("evaluateMain", p, evaluate[p]); compile("switchMain", p, switching[p]);
            }
            check(fusedHashes.size() == 8, "Program IDs merely relabeled identical executables");
            compile("resetMain", 0, reset); compile("programClassifyMain", 0, programClassify);
            compile("argumentsMain", 0, arguments); compile("familyClassifyMain", 0, familyClassify); compile("lightFamilyMain", 0, light);
            std::unique_ptr<TimestampQueryPool> timestamps;
            require(device->createTimestampQueryPool(*device->getQueue(QueueType::Graphics), {.queryCount = 5})
                .transform([&](auto value) { timestamps = std::move(value); }), "Timestamp allocation failed");
            const auto make = [&](uint64_t size, MemoryLocation memory, BufferUsageBits usage) {
                auto value = device->createBuffer({.size = size, .usage = usage, .memoryLocation = memory});
                require(value, "Buffer allocation failed"); return std::move(*value);
            };
            auto packetBuffer = make(sizeof(packets), MemoryLocation::HostUpload, BufferUsageBits::Storage);
            auto* packetMap = packetBuffer->map(); check(packetMap != nullptr, "Packet map failed");
            std::memcpy(packetMap, packets.data(), sizeof(packets)); packetBuffer->flush(); packetBuffer->unmap();
            // The small cases exercise empty bins and partial waves; only large
            // fixtures are candidates for timing interpretation.
            for (auto fixture : {std::array<uint32_t, 4>{1, 1, 8, 0}, {17, 9, 8, 1}, {257, 129, 8, 1},
                    {1024, 512, 2, 0}, {1024, 512, 2, 1}, {1024, 512, 8, 0}, {1024, 512, 8, 1}}) {
                const auto [width, height, count, mixed] = fixture;
                const uint32_t pixels = width * height, tiles = (pixels + 31) / 32;
                std::vector<uint32_t> visibility(pixels);
                for (uint32_t i = 0; i < pixels; ++i) {
                    visibility[i] = i % 29 == 0 ? ~0u : mixed ? (i * 13u + i / width) % count : (i % width) * count / width;
                }
                auto input = make(pixels * 4ull, MemoryLocation::HostUpload, BufferUsageBits::Storage);
                auto* mapped = input->map(); check(mapped != nullptr, "Visibility map failed");
                std::memcpy(mapped, visibility.data(), pixels * 4ull); input->flush(); input->unmap();
                const auto storage = BufferUsageBits::Storage;
                auto tasks = make(count * tiles * 8ull, MemoryLocation::Device, storage);
                auto control = make(64 * 4, MemoryLocation::Device, storage | BufferUsageBits::Indirect | BufferUsageBits::TransferSource);
                auto ids = make(pixels * 4ull, MemoryLocation::Device, storage);
                auto payload = make(pixels * 96ull, MemoryLocation::Device, storage);
                auto familyTasks = make(tiles * 8ull, MemoryLocation::Device, storage);
                auto output = make(std::max(pixels, 512u) * 16ull, MemoryLocation::Device, storage | BufferUsageBits::TransferSource);
                auto visits = make(pixels * 8ull, MemoryLocation::Device, storage | BufferUsageBits::TransferSource);
                auto readback = make(output->desc().size + visits->desc().size + 256, MemoryLocation::HostReadback, BufferUsageBits::TransferDestination);
                ParameterWriter writer(*device, registry);
                SchedulingParams params{writer.bufferSpan<uint32_t>(input.get()), writer.bufferSpan<Float4>(packetBuffer.get()),
                    writer.bufferSpan<UInt2>(tasks.get()), writer.bufferSpan<uint32_t>(control.get()), writer.bufferSpan<uint32_t>(ids.get()),
                    writer.bufferSpan<Float4>(payload.get()), writer.bufferSpan<UInt2>(familyTasks.get()), writer.bufferSpan<Float4>(output.get()),
                    writer.bufferSpan<uint32_t>(visits.get()), width, height, count, 0, 0};
                auto encoded = writer.encode(params, kSchedulingABI, ParameterTransport::InlinePush); require(encoded, "Encode failed");
                const auto barrier = [&](CommandBuffer& commands, std::span<Buffer* const> buffers, SyncScope before, SyncScope after) {
                    std::vector<BufferBarrierDesc> barriers;
                    for (auto* buffer : buffers) { barriers.push_back({.buffer = buffer, .before = before, .after = after}); }
                    require(commands.synchronize({.buffers = barriers}), "Buffer barrier failed");
                };
                const SyncScope computeRW{PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite};
                const SyncScope indirect{PipelineStageBits::DrawIndirect, AccessBits::IndirectRead};
                const SyncScope transfer{PipelineStageBits::Transfer, AccessBits::TransferRead};
                const std::array scratch{tasks.get(), control.get(), ids.get(), payload.get(), familyTasks.get(), output.get(), visits.get()};
                std::vector<Float4> reference;
                for (uint32_t pair = 0; pair < 8; ++pair) {
                    for (uint32_t side = 0; side < 2; ++side) {
                        const bool split = (side ^ (pair & 1u)) != 0;
                        require(timestamps->reset(0, 5), "Timestamp reset failed");
                        bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics)); require(gpu.initialize(*device), "Commands failed");
                        auto& commands = *gpu.commands;
                        // Previous submissions have completed, but explicitly make
                        // their device writes/transfer reads precede scratch reuse.
                        barrier(commands, scratch, {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}, computeRW);
                        require(reset.dispatch(commands, *encoded, (std::max(pixels, 64u) + 63) / 64), "Reset failed");
                        barrier(commands, scratch, computeRW, computeRW);
                        require(commands.writeTimestamp(*timestamps, 0, PipelineStageBits::AllCommands), "Timestamp failed");
                        require(programClassify.dispatch(commands, *encoded, tiles), "Program classify failed");
                        barrier(commands, std::array{control.get(), tasks.get()}, computeRW, computeRW);
                        require(arguments.dispatch(commands, *encoded, 1), "Program arguments failed");
                        barrier(commands, std::array{control.get()}, computeRW, indirect);
                        require(commands.writeTimestamp(*timestamps, 1, PipelineStageBits::AllCommands), "Timestamp failed");
                        std::vector<ComputeIndirectParameters> programDispatches;
                        for (uint32_t p = 0; p < count; ++p) {
                            auto slice = control->slice({uint64_t(16 + p * 3) * 4, 12}); require(slice, "Argument slice failed");
                            programDispatches.push_back({*encoded, *slice, split ? &evaluate[p] : &fused[p]});
                        }
                        auto prepared = fused[0].prepareIndirectBatch(programDispatches); require(prepared, "Batch prepare failed");
                        require(prepared->record(commands), "Material batch failed");
                        barrier(commands, std::array{ids.get(), payload.get(), output.get(), visits.get()}, computeRW, computeRW);
                        require(commands.writeTimestamp(*timestamps, 2, PipelineStageBits::AllCommands), "Timestamp failed");
                        if (split) {
                            barrier(commands, std::array{control.get()}, indirect, computeRW);
                            require(familyClassify.dispatch(commands, *encoded, tiles), "Family classify failed");
                            barrier(commands, std::array{control.get(), familyTasks.get()}, computeRW, computeRW);
                            require(arguments.dispatch(commands, *encoded, 1), "Family arguments failed");
                            barrier(commands, std::array{control.get()}, computeRW, indirect);
                        }
                        require(commands.writeTimestamp(*timestamps, 3, PipelineStageBits::AllCommands), "Timestamp failed");
                        if (split) {
                            auto slice = control->slice({40 * 4, 12}); require(slice, "Family argument slice failed");
                            require(light.dispatchIndirect(commands, *encoded, *slice), "Family lighting failed");
                        }
                        require(commands.writeTimestamp(*timestamps, 4, PipelineStageBits::AllCommands), "Timestamp failed");
                        barrier(commands, std::array{output.get(), visits.get()}, computeRW, transfer);
                        barrier(commands, std::array{control.get()}, {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}, transfer);
                        uint64_t offset = 0;
                        for (auto* buffer : {output.get(), visits.get(), control.get()}) {
                            auto source = buffer->slice(); auto destination = readback->slice({offset, buffer->desc().size});
                            require(source, "Readback source failed"); require(destination, "Readback destination failed");
                            require(commands.copyBuffer(*source, *destination), "Readback copy failed"); offset += buffer->desc().size;
                        }
                        require(gpu.submitAndWait(), "Submit failed");
                        std::array<TimestampQueryResult, 5> times; require(timestamps->readResults(0, times), "Timestamp read failed");
                        for (const auto& time : times) { check(time.available, "Unavailable timestamp"); }
                        const auto duration = [&](uint32_t a, uint32_t b) { return timestamps->durationMilliseconds(times[a].value, times[b].value); };
                        readback->invalidate(); const auto* data = static_cast<const std::byte*>(readback->map()); check(data, "Readback map failed");
                        std::vector<Float4> image(pixels); std::memcpy(image.data(), data, pixels * sizeof(Float4));
                        std::vector<uint32_t> counters(pixels * 2); std::memcpy(counters.data(), data + output->desc().size, counters.size() * 4);
                        std::array<uint32_t, 64> counts; std::memcpy(counts.data(), data + output->desc().size + visits->desc().size, 256);
                        readback->unmap();
                        uint32_t expectedFamilyTasks = 0;
                        std::vector<uint32_t> expectedProgramTasks(count);
                        for (uint32_t tile = 0; tile < tiles; ++tile) {
                            uint32_t mask = 0;
                            for (uint32_t lane = 0; lane < 32 && tile * 32 + lane < pixels; ++lane) {
                                auto program = visibility[tile * 32 + lane]; if (program < count) { mask |= 1u << program; }
                            }
                            expectedFamilyTasks += mask != 0;
                            for (uint32_t p = 0; p < count; ++p) { expectedProgramTasks[p] += (mask >> p) & 1u; }
                        }
                        for (uint32_t p = 0; p < count; ++p) { check(counts[p] == expectedProgramTasks[p] && counts[16 + p * 3] == counts[p], "Program task count mismatch"); }
                        check(counts[8] == (split ? expectedFamilyTasks : 0) && counts[40] == counts[8], "Family task/indirect count mismatch");
                        for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
                            const bool active = visibility[pixel] < count;
                            check(counters[pixel * 2] == uint32_t(active) && counters[pixel * 2 + 1] == uint32_t(active), "Missing/duplicate material or lighting invocation");
                            check(image[pixel][3] == (active ? float(visibility[pixel] + 1) : 0), "Wrong compiled Material Program");
                            for (size_t c = 0; c < 3; ++c) { check(std::isfinite(image[pixel][c]) && image[pixel][c] >= 0, "Invalid radiance"); }
                            if (!active) { check(image[pixel] == Float4{}, "Background was shaded"); }
                            if (active && pixel % 997 == 1) {
                                const auto program = visibility[pixel];
                                const double u = (pixel % width + 0.5) / width, v = (pixel / width + 0.5) / height;
                                const double nx = (std::fmod(u * 4, 1.0) * 2 - 1) * 0.65;
                                const double ny = (std::fmod(v * 2, 1.0) * 2 - 1) * 0.65;
                                const double nz = std::sqrt(1 - nx * nx - ny * ny);
                                const double pattern = 0.65 + 0.25 * std::sin(u * (7 + program) + v * 3);
                                for (size_t channel = 0; channel < 3; ++channel) {
                                    const auto& packet = packets[program];
                                    const double a = packet.first.reflectance[channel] * pattern, b = packet.second.reflectance[channel];
                                    double expected = 0;
                                    for (uint32_t l = 0; l < 16; ++l) {
                                        const double phi = l * 0.392699081699;
                                        const double mu = nx * 0.6 * std::cos(phi) + ny * 0.6 * std::sin(phi) + nz * 0.8;
                                        if (mu <= 0) { continue; }
                                        const double reflectance = program % 2 ? a + (1-a) * (1-a) * b *
                                            std::exp(-packet.first.opticalDepth[channel] * (1/mu + 1/nz)) :
                                            (1-packet.control[1]) * a + packet.control[1] * b;
                                        expected += reflectance * mu * 0.125 / 3.141592653589793;
                                    }
                                    check(std::abs(expected - image[pixel][channel]) < 2e-5, "Independent CPU lighting oracle mismatch");
                                }
                            }
                        }
                        if (reference.empty()) { reference = image; }
                        double maxDifference = 0;
                        for (uint32_t pixel = 0; pixel < pixels; ++pixel) {
                            for (size_t c = 0; c < 4; ++c) { maxDifference = std::max(maxDifference, double(std::abs(image[pixel][c] - reference[pixel][c]))); }
                        }
                        check(maxDifference <= 2e-5, "Fused/split or frame-order output mismatch");
                        report["samples"].push_back({{"width", width}, {"height", height}, {"programCount", count}, {"closureFamilyCount", 1},
                            {"mixed", mixed != 0}, {"pair", pair}, {"warmup", pair < 2}, {"split", split},
                            {"programClassificationMs", duration(0, 1)}, {"materialMs", duration(1, 2)},
                            {"familyClassificationMs", duration(2, 3)}, {"lightingMs", duration(3, 4)}, {"totalMs", duration(1, 4)},
                            {"maxOutputDifference", maxDifference}, {"closureBytes", uint64_t(pixels) * 96},
                            {"splitExtraBytes", uint64_t(pixels) * 100 + uint64_t(tiles) * 8}, {"familyTasks", expectedFamilyTasks}});
                        if (width == 1024 && count == 8 && pair == 0 && side == 0) {
                            std::vector<uint8_t> png(pixels * 4, 255);
                            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                                for (size_t c = 0; c < 3; ++c) {
                                    float v = std::clamp(image[pixel][c], 0.0f, 1.0f);
                                    png[pixel * 4 + c] = uint8_t(std::lround(255 * (v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1/2.4f) - 0.055f)));
                                }
                            }
                            const bool saved = saveRgba8Png(context.outputDirectory / (mixed ? "Mixed.png" : "Coherent.png"), png.data(), width, height, log);
                            check(saved, log);
                        }
                    }
                }
                // Isolate pipeline switching from real material/lighting work.
                if (width == 1024 && mixed == 0) {
                    for (uint32_t sample = 0; sample < 8; ++sample) {
                        for (uint32_t side = 0; side < 2; ++side) {
                            const bool distinct = (side ^ (sample & 1u)) != 0;
                            require(timestamps->reset(0, 2), "Switch timestamp reset failed");
                            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics)); require(gpu.initialize(*device), "Switch commands failed");
                            barrier(*gpu.commands, std::array{output.get()}, {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}, computeRW);
                            std::vector<PreparedComputeDispatch> batches;
                            for (uint32_t slot = 0; slot < count * 64; ++slot) {
                                params.slot = slot;
                                auto wire = writer.encode(params, kSchedulingABI, ParameterTransport::InlinePush); require(wire, "Switch encode failed");
                                auto dispatch = switching[distinct ? slot % count : 0].prepareDispatch(*wire, 1); require(dispatch, "Switch prepare failed");
                                batches.push_back(std::move(*dispatch));
                            }
                            require(gpu.commands->writeTimestamp(*timestamps, 0, PipelineStageBits::AllCommands), "Switch timestamp failed");
                            for (const auto& dispatch : batches) { require(dispatch.record(*gpu.commands), "Switch dispatch failed"); }
                            require(gpu.commands->writeTimestamp(*timestamps, 1, PipelineStageBits::AllCommands), "Switch timestamp failed");
                            require(gpu.submitAndWait(), "Switch submit failed");
                            std::array<TimestampQueryResult, 2> times; require(timestamps->readResults(0, times), "Switch timing read failed");
                            check(times[0].available && times[1].available, "Switch timestamps unavailable");
                            report["switchSamples"].push_back({{"programCount", count}, {"sample", sample}, {"warmup", sample < 2},
                                {"distinct", distinct}, {"dispatches", count * 64}, {"gpuMs", timestamps->durationMilliseconds(times[0].value, times[1].value)}});
                        }
                    }
                }
            }
            check(validationErrors == 0, "Closure scheduling produced Vulkan validation errors");
            report["validationErrors"] = validationErrors.load();
            std::ofstream file(context.outputDirectory / "ClosureScheduling.json"); file << report.dump(2) << '\n'; check(bool(file), "Report write failed");
            return RHITestResult::pass("Distinct Slab Programs share one family; fused/split GPU A/B with per-pixel identity, exactly-once evaluation/lighting and readback equivalence");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialClosureSchedulingTest);
} // namespace
} // namespace metallic::tests
