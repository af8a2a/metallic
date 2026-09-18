#include "../rhi/GpuPageCodecChecks.h"
#include <gtest/gtest.h>
#include "Runtime/Render/Streamer/MeshletStreamPageLoader.h"
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
        for (uint32_t page = 0; page < count; ++page) { ASSERT_TRUE(loader.enqueue(page)); }
        ASSERT_TRUE(loader.enqueue(asset.pageCount()));
        uint32_t completed = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (completed < count + 1 && std::chrono::steady_clock::now() < deadline) {
            render::MeshletStreamPageLoadResult result;
            if (!loader.tryPop(result)) { std::this_thread::yield(); continue; }
            ++completed;
            if (result.pageIndex == asset.pageCount()) { EXPECT_FALSE(result.success()); continue; }
            ASSERT_TRUE(result.success()) << result.failureReason;
            ASSERT_EQ(result.gpuEncoded, gpu);
            if (gpu) {
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
