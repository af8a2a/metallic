#include "Tools/SPIRVStatistics.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using metallic::tools::inspectSPIRV;

TEST(SPIRVStatistics, CountsStructuredControlFlowAndFunctionSizes)
{
    // Grammar-sized instructions: operands only supply sizes for this parser.
    const std::vector<uint32_t> words{
        0x07230203, 0x00010600, 0, 32, 0,
        (5u << 16) | 54, 1, 2, 0, 3,
        (2u << 16) | 248, 4,
        (4u << 16) | 57, 1, 5, 8,
        (3u << 16) | 247, 6, 0,
        (4u << 16) | 250, 7, 6, 6,
        (2u << 16) | 248, 6,
        (5u << 16) | 245, 1, 9, 5, 4,
        (1u << 16) | 253,
        (1u << 16) | 56,
        (5u << 16) | 54, 1, 8, 0, 3,
        (2u << 16) | 248, 10,
        (4u << 16) | 246, 11, 10, 0,
        (2u << 16) | 249, 11,
        (2u << 16) | 248, 11,
        (3u << 16) | 251, 9, 12,
        (1u << 16) | 253,
        (1u << 16) | 56};
    const auto stats = inspectSPIRV(words);
    EXPECT_EQ(stats.idBound, 32u);
    EXPECT_EQ(stats.instructions, 17u);
    EXPECT_EQ(stats.functions, 2u);
    EXPECT_EQ(stats.basicBlocks, 4u);
    EXPECT_EQ(stats.functionCalls, 1u);
    EXPECT_EQ(stats.loopMerges, 1u);
    EXPECT_EQ(stats.selectionMerges, 1u);
    EXPECT_EQ(stats.conditionalBranches, 1u);
    EXPECT_EQ(stats.branches, 1u);
    EXPECT_EQ(stats.switches, 1u);
    EXPECT_EQ(stats.phiNodes, 1u);
    EXPECT_EQ(stats.maxFunctionInstructions, 9u);
    EXPECT_EQ(stats.maxFunctionBasicBlocks, 2u);
    EXPECT_EQ(stats.largestFunctionInstructions, (std::vector<uint64_t>{9, 8}));
}

TEST(SPIRVStatistics, RejectsTruncatedAndUnterminatedModules)
{
    const std::vector<uint32_t> header{0x07230203, 0x00010600, 0, 8, 0};
    EXPECT_THROW(inspectSPIRV({}), std::runtime_error);
    auto words = header;
    words.push_back(0);
    EXPECT_THROW(inspectSPIRV(words), std::runtime_error);
    words.back() = (5u << 16) | 54;
    EXPECT_THROW(inspectSPIRV(words), std::runtime_error);
    words.insert(words.end(), {1, 2, 0, 3});
    EXPECT_THROW(inspectSPIRV(words), std::runtime_error);
    words = header;
    words.push_back((1u << 16) | 56);
    EXPECT_THROW(inspectSPIRV(words), std::runtime_error);
}
} // namespace
