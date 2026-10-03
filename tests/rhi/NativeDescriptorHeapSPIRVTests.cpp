#include "Runtime/Render/GAPI/Vulkan/NativeDescriptorHeapSPIRV.h"
#include "Runtime/Render/GAPI/Vulkan/DescriptorHeapSPIRV.h"
#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapSPIRV.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <initializer_list>

namespace metallic::tests {
namespace {

// Small instruction fixtures isolate the normalizer's policy/data-flow contract.
// Full valid Slang modules are checked separately by GPU tests and spirv-val.
using Words = std::vector<uint32_t>;
void emit(Words& words, uint32_t op, std::initializer_list<uint32_t> operands)
{
    words.push_back((uint32_t(operands.size() + 1) << 16) | op);
    words.insert(words.end(), operands.begin(), operands.end());
}

Words mixedModule(uint32_t stride = 8, uint32_t offset = 0, uint32_t signedness = 0,
    bool block = true, bool soleMember = true, bool memberOffset = true)
{
    Words code{0x07230203, 0x10600, 0, 256, 0};
    emit(code, 17, {5128}); emit(code, 17, {4473});
    if (block) { emit(code, 71, {5, 2}); }
    emit(code, 71, {4, 6, stride});
    if (memberOffset) { emit(code, 72, {5, 0, 35, offset}); }
    emit(code, 21, {1, 32, 0}); emit(code, 21, {2, 64, signedness});
    emit(code, 29, {4, 2});
    if (soleMember) { emit(code, 30, {5, 4}); }
    else { emit(code, 30, {5, 1, 4}); }
    emit(code, 32, {6, 12, 5}); emit(code, 32, {7, 12, 2});
    emit(code, 29, {10, 1}); emit(code, 30, {11, 10});
    emit(code, 32, {12, 12, 11}); emit(code, 32, {13, 12, 1});
    emit(code, 43, {2, 104, 1, 0});
    emit(code, 54, {1, 200, 0, 201});
    emit(code, 5119, {6, 20, 100});
    emit(code, 65, {7, 21, 20, 101, 101});
    emit(code, 83, {7, 22, 21});
    emit(code, 169, {7, 23, 102, 21, 22});
    // Forward edge and back edge: the policy must propagate to a fixed point.
    emit(code, 245, {7, 24, 23, 202, 26, 203});
    emit(code, 83, {7, 26, 24});
    emit(code, 68, {1, 30, 20, 0});
    emit(code, 234, {2, 40, 26, 101, 101, 104});
    emit(code, 5119, {12, 50, 100});
    emit(code, 65, {13, 51, 50, 101, 101});
    emit(code, 234, {1, 52, 51, 101, 101, 101});
    return code;
}

Words instruction(const Words& code, uint32_t op, uint32_t result)
{
    for (size_t i = 5; i < code.size(); i += code[i] >> 16) {
        const uint32_t count = code[i] >> 16;
        if ((code[i] & 0xffffu) == op && count > 2 && code[i + 2] == result) {
            return Words(code.begin() + i, code.begin() + i + count);
        }
    }
    return {};
}

void expectRejected(const Words& code, const char* diagnostic)
{
    Words output{12345};
    std::string error;
    EXPECT_FALSE(render::vulkan::normalizeNativeDescriptorHeapSpirv(code, output, error));
    EXPECT_EQ(output, Words{12345});
    EXPECT_NE(error.find(diagnostic), std::string::npos) << error;
    Words alias = code;
    EXPECT_FALSE(render::vulkan::normalizeNativeDescriptorHeapSpirv(alias, alias, error));
    EXPECT_EQ(alias, code);
}

TEST(SpirvWalker, RejectsMalformedFramingWithoutExposingPartialInstructions)
{
    using render::vulkan::SpirvWalker;
    for (const Words& code : std::vector<Words>{{}, {0x07230203, 0, 0, 1},
             {0, 0x10600, 0, 1, 0}, {0x07230203, 0x10600, 0, 1, 0, 0},
             {0x07230203, 0x10600, 0, 1, 0, (1u << 16), (65535u << 16) | 17u}}) {
        const SpirvWalker walker(code);
        EXPECT_FALSE(walker.valid());
        EXPECT_NE(walker.error(), nullptr);
        EXPECT_TRUE(walker.instructions().empty());
    }
}

TEST(SpirvWalker, DecodesOffsetsAndUnknownOpcodesWithoutSemanticValidation)
{
    using render::vulkan::SpirvWalker;
    Words code{0x07230203, 0x10600, 0, 0, 0};
    EXPECT_TRUE(SpirvWalker(code).valid()); // ID-bound policy belongs to the transform.
    emit(code, 17, {4472});
    emit(code, 65535, {});
    const SpirvWalker walker(code);
    ASSERT_TRUE(walker.valid());
    ASSERT_EQ(walker.instructions().size(), 2u);
    EXPECT_EQ(walker.instructions()[0].offset, 5u);
    EXPECT_EQ(walker.instructions()[0].wordCount, 2u);
    EXPECT_EQ(walker.instructions()[0].opcode, 17u);
    EXPECT_EQ(walker.instructions()[1].offset, 7u);
    EXPECT_EQ(walker.instructions()[1].wordCount, 1u);
    EXPECT_EQ(walker.instructions()[1].opcode, 65535u);
}

TEST(OpacityMicromapSPIRV, InPlaceTransformIsIdempotentAndFailuresPreserveOutput)
{
    using render::vulkan::enableOpacityMicromapSpirv;
    Words code{0x07230203, 0x10600, 0, 32, 0};
    emit(code, 17, {4472}); // RayQuery capability
    emit(code, 14, {0, 1});
    emit(code, 15, {5, 10, 0});
    emit(code, 54, {1, 10, 0, 2});
    emit(code, 56, {});
    for (bool useExt : {false, true}) {
        Words expected;
        ASSERT_TRUE(enableOpacityMicromapSpirv(code, expected, useExt));
        EXPECT_NE(expected, code);
        auto alias = code;
        ASSERT_TRUE(enableOpacityMicromapSpirv(alias, alias, useExt));
        EXPECT_EQ(alias, expected);
        ASSERT_TRUE(enableOpacityMicromapSpirv(alias, alias, useExt));
        EXPECT_EQ(alias, expected);
        auto malformed = code;
        malformed.push_back((4u << 16) | 17u);
        auto preserved = malformed;
        EXPECT_FALSE(enableOpacityMicromapSpirv(malformed, malformed, useExt));
        EXPECT_EQ(malformed, preserved);
        EXPECT_FALSE(enableOpacityMicromapSpirv(malformed, alias, useExt));
        EXPECT_EQ(alias, expected);
    }
    // Semantic failure after a valid instruction walk is also transactional.
    code[3] = 0xffffffffu;
    const auto original = code;
    EXPECT_FALSE(enableOpacityMicromapSpirv(code, code));
    EXPECT_EQ(code, original);
}

TEST(NativeDescriptorHeapSPIRV, PreservesEntireUInt64ChainAndNormalizesUInt32)
{
    auto input = mixedModule();
    Words output;
    std::string error;
    ASSERT_TRUE(render::vulkan::normalizeNativeDescriptorHeapSpirv(input, output, error)) << error;
    for (auto [op, id] : {std::pair{5119u, 20u}, {65u, 21u}, {83u, 22u},
            {169u, 23u}, {245u, 24u}, {83u, 26u}, {68u, 30u}, {234u, 40u}}) {
        EXPECT_EQ(instruction(input, op, id), instruction(output, op, id));
    }
    EXPECT_NE(instruction(input, 5119, 50), instruction(output, 5119, 50));
    EXPECT_TRUE(instruction(output, 65, 51).empty());
    EXPECT_FALSE(instruction(output, 4419, 51).empty());
    Words second;
    error = "stale diagnostic";
    ASSERT_TRUE(render::vulkan::normalizeNativeDescriptorHeapSpirv(output, second, error)) << error;
    EXPECT_EQ(output, second);
    EXPECT_TRUE(error.empty());
    ASSERT_TRUE(render::vulkan::normalizeNativeDescriptorHeapSpirv(input, input, error));
    EXPECT_EQ(output, input);
}

TEST(NativeDescriptorHeapSPIRV, RejectsUnprovenLayoutsBeforeMixedAtomicsReachDriver)
{
    for (const auto& code : {mixedModule(16), mixedModule(8, 8), mixedModule(8, 0, 1),
            mixedModule(8, 0, 0, false), mixedModule(8, 0, 0, true, false),
            mixedModule(8, 0, 0, true, true, false)}) {
        expectRejected(code, "mixed 32/64-bit untyped");
    }
}

TEST(NativeDescriptorHeapSPIRV, RejectsMixedPolicySelectAndPhi)
{
    for (uint32_t op : {169u, 245u}) {
        auto code = mixedModule();
        // A second uint64 pointer derived from a non-whitelisted block.
        emit(code, 30, {60, 1, 4}); emit(code, 32, {61, 12, 60});
        emit(code, 5119, {61, 62, 100}); emit(code, 65, {7, 63, 62, 101, 101});
        if (op == 169) { emit(code, op, {7, 64, 102, 21, 63}); }
        else { emit(code, op, {7, 64, 21, 202, 63, 203}); }
        expectRejected(code, "mixed typed/untyped pointer policy");
    }
}

TEST(NativeDescriptorHeapSPIRV, RejectsUntrackedPointerMergeAndEscapes)
{
    auto select = mixedModule(); emit(select, 169, {7, 64, 102, 21, 199});
    expectRejected(select, "mixed pointer select");
    auto phi = mixedModule(); emit(phi, 245, {7, 64, 21, 202, 199, 203});
    expectRejected(phi, "mixed pointer phi");
    auto call = mixedModule(); emit(call, 57, {1, 64, 199, 21});
    expectRejected(call, "escaped into a function");
    auto store = mixedModule(); emit(store, 62, {199, 21});
    expectRejected(store, "unsupported buffer pointer escape");
    auto copy = mixedModule(); emit(copy, 63, {21, 199});
    expectRejected(copy, "unsupported buffer pointer escape");
}

TEST(NativeDescriptorHeapSPIRV, RejectsPartiallyTypedChains)
{
    auto code = mixedModule();
    emit(code, 4417, {70, 12});
    emit(code, 4419, {70, 71, 5, 20, 101, 101});
    expectRejected(code, "typed uint64 chain contains an untyped pointer");
}

TEST(NativeDescriptorHeapSPIRV, ChecksAtomicLoadStoreAndCompareExchangeOperands)
{
    // Replace the uint64 Add with atomics whose operand layouts differ.
    for (uint32_t op : {227u, 228u, 230u, 231u, 232u, 233u}) {
        auto code = mixedModule(16);
        for (size_t i = 5; i < code.size(); i += code[i] >> 16) {
            if ((code[i] & 0xffffu) == 234 && code[i + 2] == 40) {
                code.erase(code.begin() + i, code.begin() + i + (code[i] >> 16));
                break;
            }
        }
        if (op == 228) { emit(code, op, {26, 101, 101, 104}); }
        else if (op == 230 || op == 231) { emit(code, op, {2, 40, 26, 101, 101, 101, 104, 104}); }
        else { emit(code, op, {2, 40, 26, 101, 101}); }
        expectRejected(code, "mixed 32/64-bit untyped");
    }
}

TEST(NativeDescriptorHeapSPIRV, TracksAtomicStoreSpecializationAndUndefinedValues)
{
    for (uint32_t op : {1u, 52u}) {
        auto code = mixedModule(16);
        Words value;
        if (op == 1) { emit(value, op, {2, 105}); }
        else { emit(value, op, {2, 105, 128, 104, 104}); }
        auto function = std::find(code.begin() + 5, code.end(), (5u << 16) | 54u);
        code.insert(function, value.begin(), value.end());
        emit(code, 228, {26, 101, 101, 105});
        expectRejected(code, "mixed 32/64-bit untyped");
    }
    auto bound = mixedModule(); bound[3] = 20;
    expectRejected(bound, "pointer id exceeds bound");
}

TEST(NativeDescriptorHeapSPIRV, MappedIsByteIdenticalAndMalformedInputIsTransactional)
{
    Words mapped{0x07230203, 0x10600, 0, 20, 0};
    emit(mapped, 21, {1, 32, 0});
    Words output;
    std::string error;
    ASSERT_TRUE(render::vulkan::normalizeNativeDescriptorHeapSpirv(mapped, output, error));
    EXPECT_EQ(output, mapped);
    expectRejected({0x07230203, 0x10600, 0, 10, 0, 0}, "truncated instruction");
    auto incomplete = mixedModule(); incomplete.push_back((3u << 16) | 230); incomplete.insert(incomplete.end(), {2, 99});
    expectRejected(incomplete, "incomplete instruction");
    auto invalidPhi = mixedModule(); emit(invalidPhi, 245, {7, 64, 21, 202, 21});
    expectRejected(invalidPhi, "invalid phi operands");
}

TEST(DescriptorHeapSPIRV, ResolvesDifferentOpaqueSizesWithoutChangingIdsOrExpressions)
{
    Words code{0x07230203, 0x10600, 0, 64, 0};
    emit(code, 21, {1, 32, 0}); emit(code, 21, {2, 64, 0});
    emit(code, 25, {3, 1, 1, 0, 0, 0, 1, 0});
    emit(code, 5115, {4, 12}); emit(code, 26, {5});
    emit(code, 5129, {1, 10, 3}); emit(code, 5129, {1, 11, 4}); emit(code, 5129, {2, 12, 5});
    emit(code, 52, {1, 13, 169, 20, 10, 11});
    Words result;
    ASSERT_TRUE(render::vulkan::specializeDescriptorHeapSizes(code, result, 48, 16, 8));
    EXPECT_EQ(instruction(result, 43, 10), (Words{(4u << 16) | 43u, 1, 10, 48}));
    EXPECT_EQ(instruction(result, 43, 11), (Words{(4u << 16) | 43u, 1, 11, 16}));
    EXPECT_EQ(instruction(result, 43, 12), (Words{(5u << 16) | 43u, 2, 12, 8, 0}));
    EXPECT_EQ(instruction(result, 52, 13), instruction(code, 52, 13));
    EXPECT_EQ(result[3], code[3]);
    auto repeated = result;
    ASSERT_TRUE(render::vulkan::specializeDescriptorHeapSizes(repeated, repeated, 48, 16, 8));
    EXPECT_EQ(repeated, result);
}

TEST(DescriptorHeapSPIRV, MappedUnchangedAndMalformedInputTransactional)
{
    Words mapped{0x07230203, 0x10600, 0, 64, 0};
    emit(mapped, 21, {1, 32, 0}); emit(mapped, 43, {1, 2, 123});
    Words result;
    ASSERT_TRUE(render::vulkan::specializeDescriptorHeapSizes(mapped, result, 0, 0, 0));
    EXPECT_EQ(result, mapped);
    auto truncated = mapped; truncated.push_back((4u << 16) | 5129u);
    ASSERT_FALSE(render::vulkan::specializeDescriptorHeapSizes(truncated, result, 48, 16, 8));
    EXPECT_EQ(result, mapped);
    Words invalidWidth{0x07230203, 0x10600, 0, 64, 0};
    emit(invalidWidth, 21, {1, 16, 0}); emit(invalidWidth, 5115, {2, 12});
    emit(invalidWidth, 5129, {1, 3, 2});
    ASSERT_FALSE(render::vulkan::specializeDescriptorHeapSizes(invalidWidth, result, 48, 16, 8));
    EXPECT_EQ(result, mapped);
    auto missingSize = invalidWidth; missingSize[7] = 32;
    ASSERT_FALSE(render::vulkan::specializeDescriptorHeapSizes(missingSize, result, 48, 0, 8));
    EXPECT_EQ(result, mapped);
}

} // namespace
} // namespace metallic::tests
