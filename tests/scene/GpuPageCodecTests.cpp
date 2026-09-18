#include "../rhi/GpuPageCodecChecks.h"
#include <gtest/gtest.h>
#include "Runtime/Render/Streamer/MeshletStreamPageLoader.h"
#include "Runtime/Render/Streamer/MeshletStreamThroughput.h"
#include "Runtime/Task/TaskSystem.h"
#include <chrono>
#include <thread>

TEST(GpuPageCodec, TranscodeAndCpuDecode)
{
    const auto directory = std::filesystem::path(PROJECT_SOURCE_DIR) / ".cache/fast-streaming/cpu-tests";
    std::filesystem::create_directories(directory);
    EXPECT_NO_THROW(metallic::tests::checkGpuPageCodec(directory));
}

TEST(GpuPageCodec, WorkerCpuFallbackAndGpuSideband)
{
    using namespace metallic;
    const auto directory = std::filesystem::path(PROJECT_SOURCE_DIR) / ".cache/fast-streaming/loader-tests";
    std::filesystem::create_directories(directory);
    ASSERT_NO_THROW(tests::checkGpuPageCodec(directory));
    ASSERT_TRUE(task::initializeTaskSystem().has_value());
    struct Shutdown { ~Shutdown() { task::shutdownTaskSystem(); } } shutdown;
    scene::MeshletStreamAsset asset;
    std::string reason;
    ASSERT_TRUE(asset.open(directory / "gpu_page_compressed.meshstream.bin", reason)) << reason;
    render::MeshletStreamPageLoader loader;
    for (bool gpu : {false, true}) {
        ASSERT_TRUE(loader.initialize(asset, 2, reason, gpu)) << reason;
        const uint32_t count = std::min(8u, asset.pageCount());
        for (uint32_t page = 0; page < count; ++page) { ASSERT_TRUE(loader.enqueue(page, page % 2 == 0)); }
        ASSERT_TRUE(loader.enqueue(asset.pageCount()));
        uint32_t completed = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (completed < count + 1 && std::chrono::steady_clock::now() < deadline) {
            render::MeshletStreamPageLoadResult result;
            if (!loader.tryPop(result)) { std::this_thread::yield(); continue; }
            ++completed;
            if (result.pageIndex == asset.pageCount()) { EXPECT_FALSE(result.success()); continue; }
            ASSERT_TRUE(result.success()) << result.failureReason;
            const bool encoded = gpu && result.pageIndex % 2 == 0;
            ASSERT_EQ(result.gpuEncoded, encoded);
            if (encoded) {
                EXPECT_TRUE(std::ranges::equal(result.payload, asset.pagePayload(result.pageIndex)));
                EXPECT_EQ(result.gpuPage.clusters.size(), asset.pages()[result.pageIndex].clusterCount);
            } else {
                std::vector<uint8_t> reference;
                ASSERT_TRUE(scene::decodeMeshletStreamGpuPage(asset.pages()[result.pageIndex], asset.pagePayload(result.pageIndex), reference, reason)) << reason;
                EXPECT_EQ(reference, result.payload);
            }
        }
        ASSERT_EQ(completed, count + 1);
        loader.reset();
    }
}

TEST(GpuPageCodec, ThroughputWallClockAndIdle)
{
    using namespace metallic::render;
    MeshletStreamThroughputTracker tracker;
    MeshletStreamTraffic totals;
    tracker.sample(1'000'000, totals);
    totals.loadedPages = 10;
    totals.loadedStoredBytes = 2 * 1024 * 1024;
    totals.geometryReadyPages = 4;
    totals.geometryReadyBytes = 1024 * 1024;
    EXPECT_EQ(tracker.snapshot(1'000'000, totals).loadedPagesPerSecond, 0);
    tracker.sample(1'500'000, totals);
    auto speed = tracker.snapshot(1'500'000, totals);
    EXPECT_DOUBLE_EQ(speed.loadedPagesPerSecond, 20);
    EXPECT_DOUBLE_EQ(speed.loadedStoredMiBPerSecond, 4);
    EXPECT_DOUBLE_EQ(speed.geometryReadyPagesPerSecond, 8);
    EXPECT_DOUBLE_EQ(speed.geometryReadyMiBPerSecond, 2);
    // No new work: speed must decay to zero, despite nonzero cumulative totals.
    tracker.sample(2'500'000, totals);
    speed = tracker.snapshot(2'500'000, totals);
    EXPECT_EQ(speed.loadedPagesPerSecond, 0);
    EXPECT_EQ(speed.geometryReadyMiBPerSecond, 0);
    EXPECT_EQ(speed.totals.loadedPages, 10);
    // A new generation, including a clock rewind, must not underflow counters.
    totals = {};
    tracker.sample(100, totals);
    EXPECT_EQ(tracker.snapshot(100, totals).windowSeconds, 0);
    totals.loadedPages = 1;
    EXPECT_DOUBLE_EQ(tracker.snapshot(100'100, totals).loadedPagesPerSecond, 10);
}
