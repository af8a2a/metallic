#include "Runtime/Render/GAPI/ShaderObjectCacheFile.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>

namespace metallic::render::detail {
namespace {

constexpr size_t kHeaderSize = 72;
constexpr size_t kBinarySizesOffset = 48;

class ShaderObjectCacheFile : public testing::Test {
protected:
    std::filesystem::path directory;
    std::filesystem::path path;
    ShaderObjectCacheFileIdentity identity;
    ShaderObjectCacheFileData data;

    void SetUp() override
    {
        static std::atomic<uint64_t> sequence{0};
        directory = std::filesystem::temp_directory_path() /
            ("MetallicShaderObjectCacheTests-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
             std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        path = directory / "nested" / "program.shaderbin";
        identity.binaryUUID = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        identity.binaryVersion = 4;
        identity.programHash = 0x9ABCFEDC76543210ull;
        data.binaries = {{{1, 2, 3, 4}, {5, 6}}};
    }

    void TearDown() override
    {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    std::vector<uint8_t> readBytes() const
    {
        std::ifstream stream(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    void writeBytes(const std::vector<uint8_t>& bytes) const
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(stream);
    }

    template <typename T>
    static void overwrite(std::vector<uint8_t>& bytes, size_t offset, const T& value)
    {
        ASSERT_LE(offset + sizeof(value), bytes.size());
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    void expectInvalid() const
    {
        ShaderObjectCacheFileData loaded = data;
        std::string reason;
        EXPECT_EQ(loadShaderObjectCacheFile(path, identity, loaded, reason),
            ShaderObjectCacheFileLoadStatus::Invalid);
        EXPECT_FALSE(reason.empty());
        EXPECT_TRUE(loaded.binaries[0].empty());
        EXPECT_TRUE(loaded.binaries[1].empty());
    }
};

TEST_F(ShaderObjectCacheFile, RoundTripAndAtomicReplacement)
{
    std::string reason;
    ShaderObjectCacheFileData loaded = data;
    EXPECT_EQ(loadShaderObjectCacheFile(path, identity, loaded, reason),
        ShaderObjectCacheFileLoadStatus::NotFound);
    EXPECT_TRUE(reason.empty());
    EXPECT_TRUE(loaded.binaries[0].empty());
    ASSERT_TRUE(saveShaderObjectCacheFile(path, identity, data, reason)) << reason;
    EXPECT_EQ(std::filesystem::file_size(path), kHeaderSize + 6);
    ASSERT_EQ(loadShaderObjectCacheFile(path, identity, loaded, reason),
        ShaderObjectCacheFileLoadStatus::Loaded) << reason;
    EXPECT_EQ(loaded.binaries, data.binaries);

    data.binaries = {{{9, 8}, {7, 6, 5, 4, 3}}};
    ASSERT_TRUE(saveShaderObjectCacheFile(path, identity, data, reason)) << reason;
    ASSERT_EQ(loadShaderObjectCacheFile(path, identity, loaded, reason),
        ShaderObjectCacheFileLoadStatus::Loaded) << reason;
    EXPECT_EQ(loaded.binaries, data.binaries);
    EXPECT_TRUE(reason.empty());
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
        EXPECT_EQ(entry.path(), path) << "Temporary cache file survived replacement";
    }
}

TEST_F(ShaderObjectCacheFile, OlderBinaryVersionIsCompatible)
{
    std::string reason;
    ASSERT_TRUE(saveShaderObjectCacheFile(path, identity, data, reason)) << reason;
    ShaderObjectCacheFileData loaded;
    auto currentIdentity = identity;
    ++currentIdentity.binaryVersion;
    EXPECT_EQ(loadShaderObjectCacheFile(path, currentIdentity, loaded, reason),
        ShaderObjectCacheFileLoadStatus::Loaded) << reason;
    EXPECT_EQ(loaded.binaries, data.binaries);

    currentIdentity.binaryVersion = identity.binaryVersion - 1;
    EXPECT_EQ(loadShaderObjectCacheFile(path, currentIdentity, loaded, reason),
        ShaderObjectCacheFileLoadStatus::Incompatible);
    EXPECT_FALSE(reason.empty());
    EXPECT_TRUE(loaded.binaries[0].empty());
    EXPECT_TRUE(loaded.binaries[1].empty());
}

TEST_F(ShaderObjectCacheFile, RejectsUUIDAndProgramMismatch)
{
    std::string reason;
    ASSERT_TRUE(saveShaderObjectCacheFile(path, identity, data, reason)) << reason;
    for (bool changeUUID : {false, true}) {
        auto currentIdentity = identity;
        if (changeUUID) {
            currentIdentity.binaryUUID[7] ^= 1;
        } else {
            currentIdentity.programHash ^= 1;
        }
        ShaderObjectCacheFileData loaded = data;
        EXPECT_EQ(loadShaderObjectCacheFile(path, currentIdentity, loaded, reason),
            ShaderObjectCacheFileLoadStatus::Incompatible);
        EXPECT_FALSE(reason.empty());
        EXPECT_TRUE(loaded.binaries[0].empty());
        EXPECT_TRUE(loaded.binaries[1].empty());
    }
}

TEST_F(ShaderObjectCacheFile, RejectsTruncationTrailingBytesAndChecksumFailure)
{
    std::string reason;
    ASSERT_TRUE(saveShaderObjectCacheFile(path, identity, data, reason)) << reason;
    const auto original = readBytes();
    for (const size_t truncatedSize : {size_t(0), kHeaderSize - 1, original.size() - 1}) {
        auto truncated = original;
        truncated.resize(truncatedSize);
        writeBytes(truncated);
        expectInvalid();
    }

    auto extended = original;
    extended.push_back(0);
    writeBytes(extended);
    expectInvalid();
    auto corrupt = original;
    corrupt.back() ^= 1;
    writeBytes(corrupt);
    expectInvalid();
}

TEST_F(ShaderObjectCacheFile, RejectsMalformedHeaderAndSizesBeforeAllocation)
{
    std::string reason;
    ASSERT_TRUE(saveShaderObjectCacheFile(path, identity, data, reason)) << reason;
    const auto original = readBytes();
    for (const size_t offset : {size_t(0), size_t(8), size_t(12), size_t(20)}) {
        auto malformed = original;
        malformed[offset] ^= 1;
        writeBytes(malformed);
        expectInvalid();
    }
    for (const uint64_t size : {uint64_t(0), uint64_t((64ull << 20u) + 1),
            std::numeric_limits<uint64_t>::max()}) {
        for (size_t stage = 0; stage < 2; ++stage) {
            auto malformed = original;
            overwrite(malformed, kBinarySizesOffset + stage * sizeof(uint64_t), size);
            writeBytes(malformed);
            expectInvalid();
        }
    }

    // The total length remains valid, but changing the stage boundary must fail the checksum.
    auto changedBoundary = original;
    overwrite(changedBoundary, kBinarySizesOffset, uint64_t(3));
    overwrite(changedBoundary, kBinarySizesOffset + sizeof(uint64_t), uint64_t(3));
    writeBytes(changedBoundary);
    expectInvalid();
}

TEST_F(ShaderObjectCacheFile, InvalidSavePreservesPreviousCache)
{
    std::string reason;
    ASSERT_TRUE(saveShaderObjectCacheFile(path, identity, data, reason)) << reason;
    const auto previous = readBytes();
    for (size_t stage = 0; stage < 2; ++stage) {
        auto invalid = data;
        invalid.binaries[stage].clear();
        EXPECT_FALSE(saveShaderObjectCacheFile(path, identity, invalid, reason));
        EXPECT_FALSE(reason.empty());
        EXPECT_EQ(readBytes(), previous);
    }
    auto wrongExtension = path;
    wrongExtension.replace_extension(".pso");
    EXPECT_FALSE(saveShaderObjectCacheFile(wrongExtension, identity, data, reason));
    ShaderObjectCacheFileData loaded;
    EXPECT_EQ(loadShaderObjectCacheFile(wrongExtension, identity, loaded, reason),
        ShaderObjectCacheFileLoadStatus::Invalid);
    EXPECT_EQ(readBytes(), previous);
}

TEST_F(ShaderObjectCacheFile, ReplacementFailureCleansTemporaryFile)
{
    std::filesystem::create_directories(path);
    std::string reason;
    EXPECT_FALSE(saveShaderObjectCacheFile(path, identity, data, reason));
    EXPECT_FALSE(reason.empty());
    EXPECT_TRUE(std::filesystem::is_directory(path));
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
        EXPECT_EQ(entry.path(), path) << "Temporary cache file survived failed replacement";
    }
}

TEST_F(ShaderObjectCacheFile, IndependentConcurrentWritersRoundTrip)
{
    std::array<std::future<void>, 4> writers;
    for (size_t writer = 0; writer < writers.size(); ++writer) {
        writers[writer] = std::async(std::launch::async, [&, writer] {
            const auto writerPath = directory / ("concurrent-" + std::to_string(writer) + ".shaderbin");
            auto writerData = data;
            writerData.binaries[0].push_back(static_cast<uint8_t>(writer));
            for (size_t iteration = 0; iteration < 8; ++iteration) {
                std::string reason;
                ASSERT_TRUE(saveShaderObjectCacheFile(writerPath, identity, writerData, reason)) << reason;
                ShaderObjectCacheFileData loaded;
                ASSERT_EQ(loadShaderObjectCacheFile(writerPath, identity, loaded, reason),
                    ShaderObjectCacheFileLoadStatus::Loaded) << reason;
                EXPECT_EQ(loaded.binaries, writerData.binaries);
            }
        });
    }
    for (auto& writer : writers) {
        writer.get();
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        EXPECT_EQ(entry.path().extension(), ".shaderbin") << "Temporary cache file survived concurrent writes";
    }
}

} // namespace
} // namespace metallic::render::detail
