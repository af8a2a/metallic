#include "Runtime/Render/GAPI/NativeBinaryCacheFile.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iterator>

namespace metallic::render::detail {
namespace {

class NativeBinaryCacheFile : public testing::Test {
protected:
    std::filesystem::path directory;
    std::filesystem::path path;
    NativeBinaryCacheData data;

    void SetUp() override
    {
        static std::atomic<uint64_t> sequence{0};
        directory = std::filesystem::temp_directory_path() / ("MetallicNativeBinaryTests-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence.fetch_add(1)));
        path = directory / "nested" / "program.pipelinebin";
        data = {.compatibilityKey = {1, 2, 3}, .programHash = 42, .kind = 2,
            .entries = {{{7, 8}, {9, 10, 11}}, {{12}, {13, 14}}}};
    }

    void TearDown() override
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

TEST_F(NativeBinaryCacheFile, PreservesOrderedKeysAndReplacesAtomically)
{
    std::string reason;
    ASSERT_TRUE(saveNativeBinaryCacheFile(path, data, reason)) << reason;
    data.entries[1].data = {33, 44, 55, 66};
    ASSERT_TRUE(saveNativeBinaryCacheFile(path, data, reason)) << reason;
    NativeBinaryCacheData loaded;
    ASSERT_EQ(loadNativeBinaryCacheFile(path, data, loaded, reason), NativeBinaryCacheLoadStatus::Loaded) << reason;
    ASSERT_EQ(loaded.entries.size(), 2);
    EXPECT_EQ(loaded.entries[0].key, data.entries[0].key);
    EXPECT_EQ(loaded.entries[1].data, data.entries[1].data);
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(path.parent_path()), std::filesystem::directory_iterator{}), 1);
}

TEST_F(NativeBinaryCacheFile, RejectsChangedCompatibilityAndProgramAndKind)
{
    std::string reason;
    ASSERT_TRUE(saveNativeBinaryCacheFile(path, data, reason));
    for (uint32_t variant = 0; variant < 3; ++variant) {
        auto expected = data;
        if (variant == 0) { expected.compatibilityKey[0] ^= 1; }
        if (variant == 1) { ++expected.programHash; }
        if (variant == 2) { ++expected.kind; }
        NativeBinaryCacheData loaded;
        EXPECT_EQ(loadNativeBinaryCacheFile(path, expected, loaded, reason), NativeBinaryCacheLoadStatus::Incompatible);
        EXPECT_TRUE(loaded.entries.empty());
    }
}

TEST_F(NativeBinaryCacheFile, RejectsCorruptionAndTruncationBeforeImport)
{
    std::string reason;
    ASSERT_TRUE(saveNativeBinaryCacheFile(path, data, reason));
    std::ifstream input(path, std::ios::binary);
    const std::vector<uint8_t> original{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    input.close();
    for (uint32_t variant = 0; variant < 3; ++variant) {
        auto bytes = original;
        if (variant == 0) { bytes[bytes.size() / 2] ^= 1; }
        if (variant == 1) { bytes.resize(24); }
        if (variant == 2) { bytes.push_back(0); }
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); output.close();
        NativeBinaryCacheData loaded;
        EXPECT_EQ(loadNativeBinaryCacheFile(path, data, loaded, reason), NativeBinaryCacheLoadStatus::Invalid);
        EXPECT_TRUE(loaded.entries.empty());
    }
}

TEST_F(NativeBinaryCacheFile, RejectsEmptyAndExcessEntriesAndKeepsGoodFile)
{
    std::string reason;
    ASSERT_TRUE(saveNativeBinaryCacheFile(path, data, reason));
    for (uint32_t variant = 0; variant < 3; ++variant) {
        auto invalid = data;
        if (variant == 0) { invalid.entries[0].data.clear(); }
        if (variant == 1) { invalid.entries.resize(65, invalid.entries[0]); }
        if (variant == 2) { invalid.compatibilityKey.resize(257); }
        EXPECT_FALSE(saveNativeBinaryCacheFile(path, invalid, reason));
    }
    NativeBinaryCacheData loaded;
    EXPECT_EQ(loadNativeBinaryCacheFile(path, data, loaded, reason), NativeBinaryCacheLoadStatus::Loaded);
}

} // namespace
} // namespace metallic::render::detail
