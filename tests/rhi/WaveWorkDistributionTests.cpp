#include "WaveWorkProbeParameters.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>

namespace metallic::tests {
namespace {

#define WAVE_WORK_REQUIRE(expr) do { const auto result = (expr); if (!result) { \
    return RHITestResult::fail(std::string(#expr) + ": " + toString(result)); } } while (false)

class WaveWorkDistributionTest final : public RHITest {
public:
    WaveWorkDistributionTest() { type = RHITestType::Rendering; name = "stream_wave_work_distribution"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Wave work distribution",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless compute"); }
        WAVE_WORK_REQUIRE(created);
        if (!device->capabilities().computeSubgroupBallotArithmetic) {
            return RHITestResult::skip("Requires compute ballot and arithmetic");
        }
        constexpr uint32_t groupCount = 48, threadCount = groupCount * 128, poison = 0xa5a5a5a5u;
        using Record = std::array<uint32_t, 4>;
        std::vector<uint32_t> counts(threadCount);
        for (uint32_t i = 0; i < threadCount; ++i) {
            const uint32_t test = i / 128, lane = i % 128;
            uint32_t hash = i * 747796405u + 2891336453u;
            hash = ((hash >> ((hash >> 28) + 4)) ^ hash) * 277803737u;
            switch (test) {
            case 0: counts[i] = 0; break;
            case 1: counts[i] = 1; break;
            case 2: counts[i] = 64; break;
            case 3: counts[i] = lane % 32 == 31 ? 64 : 0; break;
            case 4: counts[i] = lane % 32 == 0 ? 33 : 0; break;
            case 5: counts[i] = lane % 2 == 0 ? 0 : 1; break;
            case 6: counts[i] = lane % 2 == 0 ? 31 : 33; break;
            case 7: counts[i] = lane % 32 == 31 ? 1 : 0; break;
            default: counts[i] = hash % (test % 3 == 0 ? 65 : 5); break;
            }
        }
        std::vector<Record> records(threadCount * 65u, Record{poison, poison, poison, poison});
        std::array<std::unique_ptr<Buffer>, 2> buffers;
        const uint64_t sizes[] = {counts.size() * sizeof(uint32_t), records.size() * sizeof(Record)};
        const void* data[] = {counts.data(), records.data()};
        for (uint32_t i = 0; i < 2; ++i) {
            WAVE_WORK_REQUIRE(device->createBuffer({.size = sizes[i], .structureStride = i == 0 ? 4u : 16u,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { buffers[i] = std::move(rhiValue); }));


            void* mapped = buffers[i]->map();
            if (!mapped) { return RHITestResult::fail("Cannot map wave work input"); }
            std::memcpy(mapped, data[i], sizes[i]);
            buffers[i]->flush(); buffers[i]->unmap();
        }
        ShaderCompileResult compiled;
        const auto compilation = compileSlangShaderToSpirv({.moduleName = "WaveWorkProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); });
        if (!compilation) { return RHITestResult::fail(compiled.diagnostics); }
        ComputeKernel kernel;
        std::string log;
        WAVE_WORK_REQUIRE(kernel.initialize(*device, {.spirv = compiled.spirv,
            .parameters = parameterAbi<WaveWorkProbeParameters>(kWaveWorkProbeABI, ParameterTransport::InlinePush)}, log));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        WAVE_WORK_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        WAVE_WORK_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        WAVE_WORK_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
        WAVE_WORK_REQUIRE(commands->begin());
        commands->hostWriteBarrier();
        const BufferBarrierDesc barriers[] = {
            {.buffer = buffers[0].get(), .before = {}, .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}},
            {.buffer = buffers[1].get(), .before = {}, .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}}};
        if (auto commandResult = commands->synchronize({.buffers = {barriers, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        auto registry = device->resourceRegistry();
        if (!registry) { return RHITestResult::fail("Missing probe registry"); }
        ParameterWriter writer(*device, **registry);
        const WaveWorkProbeParameters params{writer.dataBuffer(buffers[0].get(), 4, 4),
            writer.dataBuffer(buffers[1].get(), 16, 4)};
        auto encoded = writer.encode(params, kWaveWorkProbeABI, ParameterTransport::InlinePush);
        if (!encoded) { return RHITestResult::fail("Cannot encode probe parameters"); }
        WAVE_WORK_REQUIRE(kernel.dispatch(*commands, *encoded, groupCount));
        WAVE_WORK_REQUIRE(commands->end());
        CommandBuffer* list[] = {commands.get()};
        WAVE_WORK_REQUIRE(queue->submit({.commandBuffers = {list, 1}, .signalFence = fence.get()}));
        WAVE_WORK_REQUIRE(fence->wait());
        buffers[1]->invalidate();
        const auto* mapped = buffers[1]->map();
        if (!mapped) { return RHITestResult::fail("Cannot read wave work output"); }
        std::memcpy(records.data(), mapped, sizes[1]); buffers[1]->unmap();
        // Recover actual subgroup membership from the shader. No assumption
        // about the mapping from workgroup IDs to subgroup IDs or lane order.
        std::map<uint32_t, std::vector<uint32_t>> waves;
        for (uint32_t i = 0; i < threadCount; ++i) {
            const auto& header = records[i * 65u];
            if (header[0] >= 128 || header[1] >= threadCount || header[2] != counts[i]) {
                return RHITestResult::fail("Invalid wave membership/header");
            }
            waves[header[1]].push_back(i);
        }
        uint32_t expanded = 0;
        for (auto& [key, members] : waves) {
            std::sort(members.begin(), members.end(), [&](uint32_t a, uint32_t b) {
                return records[a * 65u][0] < records[b * 65u][0];
            });
            uint32_t total = 0;
            for (uint32_t owner : members) { total += counts[owner]; }
            uint32_t prefix = 0, lane = 0;
            for (uint32_t owner : members) {
                if (records[owner * 65u][0] != lane || records[owner * 65u][3] != total) {
                    return RHITestResult::fail("Wave lane/total mismatch");
                }
                for (uint32_t item = 0; item < 64; ++item) {
                    const Record expected = item < counts[owner] ? Record{prefix + item, owner, item, lane} :
                        Record{poison, poison, poison, poison};
                    if (records[owner * 65u + 1u + item] != expected) {
                        return RHITestResult::fail("Missing, duplicated, reordered or out-of-bounds wave work");
                    }
                }
                prefix += counts[owner];
                ++lane;
            }
            expanded += total;
        }
        return RHITestResult::pass(std::to_string(waves.size()) + " waves / " + std::to_string(expanded) +
            " items match serial expansion; empty, sparse, dense, multi-batch, partial tail and multiple waves/workgroup");
    }
};
METALLIC_REGISTER_RHI_TEST(WaveWorkDistributionTest);
#undef WAVE_WORK_REQUIRE
} // namespace
} // namespace metallic::tests
