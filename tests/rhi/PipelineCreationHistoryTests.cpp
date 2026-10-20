#include "Runtime/Render/Core/PipelineCreationHistory.h"

#include <gtest/gtest.h>
#include <json.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace metallic::tests {
namespace {

using render::detail::PipelineCreationHistory;
using Json = nlohmann::json;

class PipelineCreationHistoryTest : public testing::Test {
protected:
    void SetUp() override
    {
        static std::atomic<uint64_t> sequence{0};
        directory_ = std::filesystem::temp_directory_path() /
            ("metallic-pipeline-history-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                "-" + std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directories(directory_);
        path_ = directory_ / "history.json";
        identity_[0] = 0x81;
        identity_[31] = 0xf4;
    }

    void TearDown() override
    {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
    }

    void write(const std::string& text)
    {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        output << text;
        ASSERT_TRUE(output.good());
    }

    Json validDocument()
    {
        PipelineCreationHistory history(path_, identity_);
        history.observe(1, 70, true, false);
        std::string reason;
        EXPECT_TRUE(history.save(reason)) << reason;
        std::ifstream input(path_);
        return Json::parse(input);
    }

    std::filesystem::path directory_;
    std::filesystem::path path_;
    std::array<uint8_t, 32> identity_{};
};

TEST_F(PipelineCreationHistoryTest, MissingFileLoadsAsEmptyHistory)
{
    PipelineCreationHistory history(path_, identity_);
    history.observe(1, 50, true, false);
    std::string reason = "previous failure";
    EXPECT_TRUE(history.load(reason)) << reason;
    EXPECT_TRUE(reason.empty());
    EXPECT_FALSE(history.duration(1));
    EXPECT_FALSE(std::filesystem::exists(path_));
}

TEST_F(PipelineCreationHistoryTest, RoundTripPreservesIndependentFullWidthKeysAndDurations)
{
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    PipelineCreationHistory history(path_, identity_);
    EXPECT_EQ(history.observe(0, 7, true, false), 1);
    EXPECT_EQ(history.observe(maximum, maximum, true, false), 2);
    EXPECT_EQ(history.observe(maximum - 1, 190, true, false), 3);
    std::string reason;
    ASSERT_TRUE(history.save(reason)) << reason;
    PipelineCreationHistory restored(path_, identity_);
    ASSERT_TRUE(restored.load(reason)) << reason;
    EXPECT_EQ(restored.duration(0), 7);
    EXPECT_EQ(restored.duration(maximum), maximum);
    EXPECT_EQ(restored.duration(maximum - 1), 190);
    EXPECT_FALSE(restored.duration(3));
    EXPECT_EQ(restored.observe(maximum, 1, true, false), 0);
}

TEST_F(PipelineCreationHistoryTest, OnlyValidColdSamplesCanRaisePeakAndRevision)
{
    PipelineCreationHistory history(path_, identity_);
    EXPECT_EQ(history.observe(5, 100, true, false), 1);
    EXPECT_EQ(history.observe(5, 1, true, false), 0);
    EXPECT_EQ(history.observe(5, 100, true, false), 0);
    EXPECT_EQ(history.observe(5, 1000, true, true), 0);
    EXPECT_EQ(history.observe(5, 2000, false, false), 0);
    EXPECT_EQ(history.observe(5, 0, true, false), 0);
    EXPECT_EQ(history.observe(6, 1000, true, true), 0);
    EXPECT_EQ(history.observe(6, 1000, false, false), 0);
    EXPECT_EQ(history.observe(6, 0, true, false), 0);
    EXPECT_FALSE(history.duration(6));
    EXPECT_EQ(history.duration(5), 100);
    EXPECT_EQ(history.observe(5, 150, true, false), 2);
    EXPECT_EQ(history.observe(6, 50, true, false), 3);
    EXPECT_EQ(history.duration(5), 150);
    EXPECT_EQ(history.duration(6), 50);
}

TEST_F(PipelineCreationHistoryTest, RejectsMalformedTruncatedAndDuplicateMemberPayloads)
{
    const auto valid = validDocument().dump();
    const std::vector<std::string> invalid = {
        "", "not JSON", valid.substr(0, valid.size() - 4), valid + "trailing",
        "{\"version\":1,\"version\":1,\"identity\":\"x\",\"entries\":[]}",
        std::string(16 * 1024 * 1024 + 1, ' '),
        std::string(100, '[') + std::string(100, ']'),
    };
    for (const auto& payload : invalid) {
        write(payload);
        PipelineCreationHistory history(path_, identity_);
        history.observe(9, 10, true, false);
        std::string reason;
        EXPECT_FALSE(history.load(reason));
        EXPECT_FALSE(reason.empty());
        EXPECT_FALSE(history.duration(1));
        EXPECT_FALSE(history.duration(9));
    }
}

TEST_F(PipelineCreationHistoryTest, RejectsWrongSchemaVersionIdentityAndInvalidEntriesWithoutPartialLoad)
{
    const Json valid = validDocument();
    std::vector<Json> invalid;
    auto addChange = [&](const auto& change) {
        auto candidate = valid;
        change(candidate);
        invalid.push_back(std::move(candidate));
    };
    addChange([](auto& value) { value["version"] = 2; });
    addChange([](auto& value) { value["version"] = 1.0; });
    addChange([](auto& value) { value["identity"] = std::string(64, '0'); });
    addChange([](auto& value) { value.erase("entries"); });
    addChange([](auto& value) { value["extra"] = true; });
    addChange([](auto& value) { value["entries"] = Json::object(); });
    addChange([](auto& value) { value["entries"].push_back(value["entries"][0]); });
    addChange([](auto& value) { value["entries"].push_back({{"psoHash", "0000000000000002"}, {"nativeNanoseconds", 0}}); });
    addChange([](auto& value) { value["entries"][0]["nativeNanoseconds"] = -1; });
    addChange([](auto& value) { value["entries"][0]["nativeNanoseconds"] = 7.5; });
    addChange([](auto& value) { value["entries"][0]["psoHash"] = "fffffffffffffffff"; });
    addChange([](auto& value) { value["entries"][0]["psoHash"] = "-000000000000001"; });
    addChange([](auto& value) { value["entries"][0]["psoHash"] = "000000000000000g"; });
    addChange([](auto& value) { value["entries"][0]["extra"] = true; });
    addChange([](auto& value) {
        const Json entry = value["entries"][0];
        value["entries"] = Json::array();
        for (size_t index = 0; index < 65537; ++index) { value["entries"].push_back(entry); }
    });
    invalid.push_back(Json::array());
    for (const auto& document : invalid) {
        write(document.dump());
        PipelineCreationHistory history(path_, identity_);
        std::string reason;
        EXPECT_FALSE(history.load(reason));
        EXPECT_FALSE(reason.empty());
        EXPECT_FALSE(history.duration(1));
        EXPECT_FALSE(history.duration(2));
    }
}

TEST_F(PipelineCreationHistoryTest, ConcurrentObservationLookupAndSaveRetainAllPeaks)
{
    PipelineCreationHistory history(path_, identity_);
    std::atomic<bool> start{false};
    std::atomic<bool> saveFailed{false};
    std::vector<std::thread> workers;
    for (uint64_t worker = 0; worker < 4; ++worker) {
        workers.emplace_back([&, worker] {
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            for (uint64_t sample = 1; sample <= 100; ++sample) {
                history.observe(worker, sample, true, false);
                history.observe(99, sample, true, false);
                (void)history.duration(worker);
            }
        });
    }
    for (size_t saver = 0; saver < 2; ++saver) {
        workers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            for (size_t index = 0; index < 10; ++index) {
                std::string reason;
                if (!history.save(reason)) { saveFailed.store(true); }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) { worker.join(); }
    EXPECT_FALSE(saveFailed.load());
    std::string reason;
    ASSERT_TRUE(history.save(reason)) << reason;
    PipelineCreationHistory restored(path_, identity_);
    ASSERT_TRUE(restored.load(reason)) << reason;
    for (uint64_t worker = 0; worker < 4; ++worker) { EXPECT_EQ(restored.duration(worker), 100); }
    EXPECT_EQ(restored.duration(99), 100);
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        EXPECT_EQ(entry.path(), path_);
    }
}

TEST_F(PipelineCreationHistoryTest, FailedReplacementPreservesDestination)
{
    // An existing directory cannot be replaced by the temporary regular file.
    std::filesystem::create_directory(path_);
    const auto sentinel = path_ / "existing-data";
    { std::ofstream output(sentinel); output << "preserved"; }
    PipelineCreationHistory history(path_, identity_);
    history.observe(1, 100, true, false);
    std::string reason;
    EXPECT_FALSE(history.save(reason));
    EXPECT_FALSE(reason.empty());
    std::ifstream input(sentinel);
    std::string contents;
    input >> contents;
    EXPECT_EQ(contents, "preserved");
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        EXPECT_EQ(entry.path(), path_);
    }
}

#if defined(_WIN32)
TEST_F(PipelineCreationHistoryTest, FailedReplacementOfLockedFilePreservesPreviousHistory)
{
    const auto original = validDocument();
    PipelineCreationHistory history(path_, identity_);
    history.observe(2, 100, true, false);
    const HANDLE lockedFile = CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    ASSERT_NE(lockedFile, INVALID_HANDLE_VALUE);
    std::string reason;
    const bool saved = history.save(reason);
    CloseHandle(lockedFile);
    EXPECT_FALSE(saved);
    EXPECT_FALSE(reason.empty());
    std::ifstream input(path_);
    EXPECT_EQ(Json::parse(input), original);
}
#endif

} // namespace
} // namespace metallic::tests
