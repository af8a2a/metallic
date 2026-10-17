#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ComputeResourceParameters.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <span>

namespace metallic::tests {
namespace {

class InlineComputeRootShaderTest final : public RHITest
{
public:
    InlineComputeRootShaderTest()
    {
        type = RHITestType::Resource;
        name = "inline_compute_root_shader_fields_and_bounds";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.inline.root.fields.bounds", "compute.inline.root.constants"},
            bench::Layer::Core, "binding", "binding", {"inline-compute-root-readback.bin"});
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        constexpr uint64_t kIndirectABI = 0x434f4d50524f4f01ull;
        constexpr uint64_t kDirectABI = 0x434f4d50524f4f02ull;
        constexpr uint32_t kWordCount = 16;
        constexpr uint32_t kMarker = 0x524f4f54u;
        static_assert(sizeof(ComputeResourceParameters) == 24);
        static_assert(sizeof(GPUProbeResourceParameters) == 8);
        using Readback = std::array<uint32_t, kWordCount>;
        struct ProbeCase
        {
            uint32_t resourceWordCount;
            bool sentinel;
            uint32_t dynamicOffset;
        };
        constexpr std::array<ProbeCase, 7> kCases{{
            {2, false, 0},
            {2, false, 4},
            {1, false, 0},
            {2, true, 4},
            {2, false, 1},
            {2, false, 8},
            {2, false, 0xfffffffcu},
        }};

        bench::TestDevice fixture;
        auto result = bench::createTestDevice(context, {.applicationName = "Inline compute root fields",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { fixture = std::move(value); });
        if (hasError(result, Error::Unsupported)) {
            return RHITestResult::skip("Inline compute root probe requires bindless descriptors");
        }
        if (!result) { return RHITestResult::fail("Inline root probe device creation failed"); }
        auto& device = *fixture;
        auto* queue = device.getQueue(QueueType::Graphics);
        if (queue == nullptr) { return RHITestResult::fail("Inline root probe requires a graphics queue"); }
        auto registry = ResourceRegistry::forDevice(device);
        if (!registry) { return RHITestResult::fail("Inline root probe registry initialization failed"); }
        std::unique_ptr<Buffer> output;
        result = device.createBuffer({.size = sizeof(Readback), .structureStride = sizeof(uint32_t),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback,
            .queueAccess = QueueAccessBits::Graphics})
            .transform([&](auto value) { output = std::move(value); });
        if (!result) { return RHITestResult::fail("Inline root probe output allocation failed"); }

        std::vector<uint32_t> evidence;
        std::array<Readback, kCases.size()> mappedReference{};
        for (auto mode : {SlangDescriptorHeapMode::Mapped, SlangDescriptorHeapMode::Native}) {
            std::array<Readback, kCases.size()> indirectReference{};
            for (uint32_t direct = 0; direct < 2; ++direct) {
                const auto transport = direct != 0 ? ParameterTransport::InlinePush : ParameterTransport::DescriptorBuffer;
                const uint64_t abiId = direct != 0 ? kDirectABI : kIndirectABI;
                const SlangMacroDefine defines[] = {{"METALLIC_INLINE_COMPUTE_ROOT", direct != 0 ? "1" : "0"}};
                std::string log;
                auto shader = compileSlangShaderToSpirv({.moduleName = "ComputeRootProbe", .entryPointName = "main",
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = defines,
                    .descriptorHeapMode = mode}, log);
                if (!shader) { return RHITestResult::fail("Inline root probe compile failed: " + log); }
                ComputeKernel kernel;
                result = kernel.initialize(device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<ComputeResourceParameters>(abiId, transport)}, log);
                if (!result) { return RHITestResult::fail("Inline root probe kernel initialization failed: " + log); }

                for (size_t caseIndex = 0; caseIndex < kCases.size(); ++caseIndex) {
                    const auto& probe = kCases[caseIndex];
                    const uint32_t inputValue = probe.sentinel ? UINT32_MAX : 0x12345678u;
                    const uint32_t outputValue = probe.sentinel ? UINT32_MAX : 0x87654321u;
                    // Nonzero span offsets and guard words distinguish a root
                    // layout failure from merely loading the first buffer word.
                    const std::array<uint32_t, 4> resourceWords{0xdeadc0deu, inputValue, outputValue, 0xbad0c0deu};
                    ParameterWriter writer(device, **registry);
                    const auto outputHandle = writer.buffer(output.get());
                    const std::array<uint32_t, 6> constantWords{
                        0xdeadbeefu, outputHandle.index, kMarker, probe.dynamicOffset, 0u, 0xbadbeefu};
                    ComputeResourceParameters root;
                    root.resources = writer.dataSpan(resourceWords.data(), sizeof(resourceWords), 4, 4);
                    root.resources.byteOffset += sizeof(uint32_t);
                    root.resources.count = probe.resourceWordCount;
                    root.constants = writer.dataSpan(constantWords.data(), sizeof(constantWords), 4, 4);
                    root.constants.byteOffset += sizeof(uint32_t);
                    root.constants.count = 4;
                    auto encoded = writer.encode(root, abiId, transport);
                    if (!encoded) { return RHITestResult::fail("Inline root probe parameter encoding failed"); }
                    if (encoded->inlineData().size() != (direct != 0 ? sizeof(root) : 0u)) {
                        return RHITestResult::fail("Inline root probe received the wrong transport bytes");
                    }

                    // This helper drains submitted work before encoded uploads,
                    // the writer, kernel or output buffer can be destroyed.
                    bench::GPUCommands gpu(*queue);
                    result = gpu.initialize(device);
                    if (!result) { return RHITestResult::fail("Inline root probe command initialization failed"); }
                    gpu.commands->hostWriteBarrier();
                    result = kernel.dispatch(*gpu.commands, *encoded, 1);
                    if (!result) { return RHITestResult::fail("Inline root probe dispatch failed"); }
                    result = gpu.submitAndWait();
                    if (!result) { return RHITestResult::fail("Inline root probe submission/wait failed"); }
                    output->invalidate();
                    const auto* mapped = static_cast<const uint32_t*>(output->map());
                    if (mapped == nullptr) { return RHITestResult::fail("Inline root probe output mapping failed"); }
                    Readback actual;
                    std::copy_n(mapped, actual.size(), actual.begin());
                    output->unmap();
                    evidence.insert(evidence.end(), actual.begin(), actual.end());

                    const bool completePacket = probe.resourceWordCount >= 2;
                    uint32_t dynamicValue = 0;
                    if (completePacket && probe.dynamicOffset == 0) { dynamicValue = inputValue; }
                    else if (completePacket && probe.dynamicOffset == 4) { dynamicValue = outputValue; }
                    const Readback expected{
                        24, root.resources.byteOffset, root.resources.count,
                        root.constants.byteOffset, root.constants.count, kMarker,
                        completePacket ? inputValue : 0u, completePacket ? outputValue : 0u,
                        completePacket ? inputValue : 0u, completePacket ? outputValue : 0u,
                        0, 0, 0, 0, dynamicValue, direct};
                    if (actual != expected) {
                        for (size_t word = 0; word < actual.size(); ++word) {
                            if (actual[word] == expected[word]) { continue; }
                            return RHITestResult::fail("Inline root probe mismatch: mode=" +
                                std::to_string(static_cast<uint32_t>(mode)) + ", direct=" + std::to_string(direct) +
                                ", case=" + std::to_string(caseIndex) + ", word=" + std::to_string(word) +
                                ", actual=" + std::to_string(actual[word]) + ", expected=" + std::to_string(expected[word]));
                        }
                    }
                    // Upload offsets and the deliberate transport marker can
                    // differ; every transported value and guard must agree.
                    actual[1] = actual[3] = actual[15] = 0;
                    if (direct == 0) {
                        indirectReference[caseIndex] = actual;
                        if (mode == SlangDescriptorHeapMode::Mapped) { mappedReference[caseIndex] = actual; }
                        else if (actual != mappedReference[caseIndex]) {
                            return RHITestResult::fail("Mapped/native inline root reference mismatch");
                        }
                    } else if (actual != indirectReference[caseIndex]) {
                        return RHITestResult::fail("Direct/indirect compute root mismatch");
                    }
                }
            }
        }
        bench::readbackEvidence(context, "inline-compute-root-readback.bin", std::span<const uint32_t>(evidence));
        return RHITestResult::pass("Mapped/native 24-byte direct and 12-byte indirect roots agree for full/short packets, offsets, sentinels and constants");
    }
};

METALLIC_REGISTER_RHI_TEST(InlineComputeRootShaderTest);

} // namespace
} // namespace metallic::tests
