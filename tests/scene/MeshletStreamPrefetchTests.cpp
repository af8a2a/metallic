#include "Runtime/Render/Streamer/MeshletStreamPrefetch.h"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace {
using namespace metallic::render;

MeshletStreamPrefetchCamera translated(float x, float y = 0, float z = 0)
{
    MeshletStreamPrefetchCamera camera;
    camera.eye = {x, y, z};
    camera.center = {x, y, z - 1};
    return camera;
}

MeshletStreamPrefetchCamera yawed(float degrees)
{
    MeshletStreamPrefetchCamera camera;
    const double angle = double(degrees) * 0.017453292519943295;
    camera.center = {float(-std::sin(angle)), 0, float(-std::cos(angle))};
    return camera;
}

TEST(MeshletStreamPrefetch, TranslationUsesMeasuredDemandLatencyInMilliseconds)
{
    MeshletStreamPrefetchPredictor predictor;
    const auto first = predictor.update(translated(0), 0.02, {.count = 20, .p95 = 80});
    EXPECT_TRUE(first.historyReset);
    EXPECT_FALSE(first.active);
    const auto result = predictor.update(translated(0.04f), 0.02, {.count = 20, .p95 = 80});
    EXPECT_TRUE(result.active);
    EXPECT_TRUE(result.measuredLatency);
    EXPECT_FALSE(result.historyReset);
    EXPECT_NEAR(result.horizonSeconds, 0.1, 1e-9);
    EXPECT_NEAR(result.linearSpeed, 2, 1e-6);
    EXPECT_NEAR(result.translationDistance, 0.2, 1e-6);
    EXPECT_NEAR(result.camera.eye[0], 0.24f, 1e-6f);
    EXPECT_NEAR(result.camera.center[0], 0.24f, 1e-6f);
    EXPECT_FLOAT_EQ(result.camera.center[2], -1);
}

TEST(MeshletStreamPrefetch, ForecastScalesWithLatencyAndIsBounded)
{
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(translated(0), 0.01, {});
    const auto early = predictor.update(translated(0.01f), 0.01, {.count = 10, .p95 = 20});
    const auto late = predictor.update(translated(0.02f), 0.01, {.count = 20, .p95 = 200});
    EXPECT_GT(late.translationDistance, early.translationDistance * 6);
    EXPECT_NEAR(late.horizonSeconds, 0.21, 1e-9);
    const auto stalled = predictor.update(translated(1.02f), 0.01, {.count = 30, .p95 = 100000});
    EXPECT_DOUBLE_EQ(stalled.horizonSeconds, 0.25);
    EXPECT_DOUBLE_EQ(stalled.translationDistance, 2);
    EXPECT_NEAR(stalled.camera.eye[0], 3.02f, 1e-6f);
}

TEST(MeshletStreamPrefetch, MissingOrInvalidLatencyUsesExplicitFallback)
{
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(translated(0), 0.02, {});
    auto result = predictor.update(translated(0.01f), 0.02, {.p95 = 900});
    EXPECT_FALSE(result.measuredLatency);
    EXPECT_NEAR(result.horizonSeconds, 0.12, 1e-9);
    result = predictor.update(translated(0.02f), 0.02,
        {.count = 1, .p95 = std::numeric_limits<double>::quiet_NaN()});
    EXPECT_FALSE(result.measuredLatency);
    EXPECT_NEAR(result.horizonSeconds, 0.12, 1e-9);
    result = predictor.update(translated(0.03f), 0.02, {.count = 1, .p95 = 0});
    EXPECT_TRUE(result.measuredLatency);
    EXPECT_DOUBLE_EQ(result.horizonSeconds, 0.025);
}

TEST(MeshletStreamPrefetch, StopsImmediatelyWithoutResidualVelocity)
{
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(translated(0), 0.02, {});
    EXPECT_TRUE(predictor.update(translated(0.2f), 0.02, {}).active);
    const auto stopped = predictor.update(translated(0.2f), 0.02, {});
    EXPECT_FALSE(stopped.active);
    EXPECT_DOUBLE_EQ(stopped.translationDistance, 0);
    EXPECT_EQ(stopped.camera.eye, translated(0.2f).eye);
}

TEST(MeshletStreamPrefetch, PredictsYawAndCapsAngularLead)
{
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(yawed(0), 0.02, {});
    const auto result = predictor.update(yawed(2), 0.02, {.count = 20, .p95 = 80});
    EXPECT_TRUE(result.active);
    EXPECT_NEAR(result.angularSpeedDegrees, 100, 1e-4);
    EXPECT_NEAR(result.rotationDegrees, 10, 1e-5);
    EXPECT_NEAR(result.camera.center[0], yawed(12).center[0], 1e-6f);
    EXPECT_NEAR(result.camera.center[2], yawed(12).center[2], 1e-6f);
    const auto fast = predictor.update(yawed(7), 0.02, {.count = 20, .p95 = 1000});
    EXPECT_DOUBLE_EQ(fast.rotationDegrees, 20);
    EXPECT_NEAR(fast.camera.center[0], yawed(27).center[0], 1e-6f);
}

TEST(MeshletStreamPrefetch, PredictsRollAndKeepsOrthonormalBasis)
{
    MeshletStreamPrefetchPredictor predictor;
    MeshletStreamPrefetchCamera camera;
    predictor.update(camera, 0.02, {});
    const double angle = 2 * 0.017453292519943295;
    camera.up = {float(-std::sin(angle)), float(std::cos(angle)), 0};
    const auto result = predictor.update(camera, 0.02, {.count = 20, .p95 = 80});
    EXPECT_TRUE(result.active);
    EXPECT_NEAR(result.rotationDegrees, 10, 1e-5);
    EXPECT_NEAR(result.camera.up[0], -std::sin(angle * 6), 1e-6f);
    EXPECT_NEAR(result.camera.up[1], std::cos(angle * 6), 1e-6f);
    EXPECT_NEAR(result.camera.up[0] * result.camera.up[0] + result.camera.up[1] * result.camera.up[1], 1, 1e-6f);
    EXPECT_NEAR(result.camera.center[0], 0, 1e-6f);
    EXPECT_NEAR(result.camera.center[2], -1, 1e-6f);
}

TEST(MeshletStreamPrefetch, OrthographicPredictionPreservesProjectionAndFocus)
{
    MeshletStreamPrefetchPredictor predictor;
    auto camera = translated(0);
    camera.orthographic = true;
    camera.orthoHeight = 17;
    camera.center[2] = -5;
    predictor.update(camera, 0.02, {});
    camera.eye[0] = camera.center[0] = 0.04f;
    const auto result = predictor.update(camera, 0.02, {.count = 20, .p95 = 80});
    EXPECT_TRUE(result.active);
    EXPECT_TRUE(result.camera.orthographic);
    EXPECT_FLOAT_EQ(result.camera.orthoHeight, 17);
    EXPECT_FLOAT_EQ(result.camera.center[2], -5);
    EXPECT_NEAR(result.camera.eye[0], 0.24f, 1e-6f);
}

TEST(MeshletStreamPrefetch, TeleportAndCameraCutRebaseWithoutExtrapolation)
{
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(translated(0), 0.02, {});
    auto result = predictor.update(translated(100), 0.02, {});
    EXPECT_TRUE(result.historyReset);
    EXPECT_FALSE(result.active);
    EXPECT_EQ(result.camera.eye, translated(100).eye);
    result = predictor.update(translated(100.1f), 0.02, {});
    EXPECT_FALSE(result.historyReset);
    EXPECT_TRUE(result.active);
    result = predictor.update(translated(100.2f), 0.02, {}, true);
    EXPECT_TRUE(result.historyReset);
    EXPECT_FALSE(result.active);
    predictor.reset();
    result = predictor.update(translated(100.3f), 0.02, {});
    EXPECT_TRUE(result.historyReset);
    EXPECT_FALSE(result.active);
}

TEST(MeshletStreamPrefetch, LargeTurnsAndProjectionChangesResetHistory)
{
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(yawed(0), 0.02, {});
    const auto turn = predictor.update(yawed(90), 0.02, {});
    EXPECT_TRUE(turn.historyReset);
    EXPECT_FALSE(turn.active);
    auto camera = yawed(91);
    EXPECT_TRUE(predictor.update(camera, 0.02, {}).active);
    camera.orthographic = true;
    EXPECT_TRUE(predictor.update(camera, 0.02, {}).historyReset);
    camera.orthoHeight = 12;
    EXPECT_TRUE(predictor.update(camera, 0.02, {}).historyReset);
    camera.fovDegrees = 75;
    EXPECT_TRUE(predictor.update(camera, 0.02, {}).historyReset);
}

TEST(MeshletStreamPrefetch, InvalidTimeAndCameraCannotPoisonLaterForecast)
{
    for (const double delta : {0.0, -1.0, 1.0, std::numeric_limits<double>::quiet_NaN()}) {
        MeshletStreamPrefetchPredictor predictor;
        predictor.update(translated(0), 0.02, {});
        const auto invalid = predictor.update(translated(0.1f), delta, {});
        EXPECT_FALSE(invalid.active);
        EXPECT_TRUE(invalid.historyReset);
        const auto next = predictor.update(translated(0.2f), 0.02, {});
        EXPECT_TRUE(next.active);
        EXPECT_NEAR(next.linearSpeed, 5, 1e-5);
    }
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(translated(0), 0.02, {});
    auto camera = translated(0.1f);
    camera.eye[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(predictor.update(camera, 0.02, {}).active);
    EXPECT_TRUE(predictor.update(translated(0.2f), 0.02, {}).historyReset);
    EXPECT_TRUE(predictor.update(translated(0.3f), 0.02, {}).active);
    camera = translated(0.4f);
    camera.up = {0, 0, 1};
    EXPECT_FALSE(predictor.update(camera, 0.02, {}).active);
    camera.center = camera.eye;
    EXPECT_FALSE(predictor.update(camera, 0.02, {}).active);
}

TEST(MeshletStreamPrefetch, RotationCanCrossTheAngleWrapWithoutBecomingACut)
{
    MeshletStreamPrefetchPredictor predictor;
    predictor.update(yawed(179), 0.02, {});
    const auto result = predictor.update(yawed(-179), 0.02, {.count = 20, .p95 = 80});
    EXPECT_FALSE(result.historyReset);
    EXPECT_TRUE(result.active);
    EXPECT_NEAR(result.rotationDegrees, 10, 1e-5);
    EXPECT_NEAR(result.camera.center[0], yawed(-169).center[0], 1e-6f);
    EXPECT_NEAR(result.camera.center[2], yawed(-169).center[2], 1e-6f);
}

TEST(MeshletStreamPrefetch, RecentLatencyExpiresAndRejectsClockReversal)
{
    MeshletStreamLatencyTracker tracker;
    tracker.request(0, 1, 1, false, 1'000'000);
    tracker.complete(0, 2, 1'080'500);
    auto recent = tracker.recentDemandLatency(1'080'500);
    ASSERT_EQ(recent.count, 1u);
    EXPECT_DOUBLE_EQ(recent.p95, 80.5);
    EXPECT_DOUBLE_EQ(recent.mean, 80.5);
    EXPECT_EQ(tracker.recentDemandLatency(1'080'499).count, 0u);
    EXPECT_EQ(tracker.recentDemandLatency(6'080'500).count, 1u);
    EXPECT_EQ(tracker.recentDemandLatency(6'080'501).count, 0u);
    // Historical profiling remains cumulative after the prediction sample expires.
    EXPECT_EQ(tracker.snapshot(6'080'501).milliseconds[size_t(MeshletStreamLatencyStage::DemandToDrawable)].count, 1u);
}

TEST(MeshletStreamPrefetch, RecentLatencyRingRetainsOnlyTheLatest64Completions)
{
    MeshletStreamLatencyTracker tracker;
    for (uint32_t page = 0; page < 80; ++page) {
        tracker.request(page, 1, 1, false, 1'000'000);
        tracker.complete(page, 2, 1'000'000 + (page + 1) * 1000);
    }
    const auto recent = tracker.recentDemandLatency(1'080'000);
    ASSERT_EQ(recent.count, 64u);
    // The remaining durations are 17..80 ms, with nearest-rank percentiles.
    EXPECT_DOUBLE_EQ(recent.p50, 48);
    EXPECT_DOUBLE_EQ(recent.p95, 77);
    EXPECT_DOUBLE_EQ(recent.p99, 80);
    EXPECT_DOUBLE_EQ(recent.maximum, 80);
    EXPECT_DOUBLE_EQ(recent.mean, 48.5);
}

TEST(MeshletStreamPrefetch, PurePrefetchDoesNotTrainDemandAndPromotionStartsAtDemand)
{
    MeshletStreamLatencyTracker tracker;
    tracker.request(0, 1, 1, true, 1'000'000);
    tracker.complete(0, 2, 2'000'000);
    EXPECT_EQ(tracker.recentDemandLatency(2'000'000).count, 0u);
    tracker.request(1, 1, 1, true, 2'000'000);
    tracker.request(1, 2, 2, false, 2'800'000);
    tracker.complete(1, 3, 2'850'000);
    const auto recent = tracker.recentDemandLatency(2'850'000);
    ASSERT_EQ(recent.count, 1u);
    EXPECT_DOUBLE_EQ(recent.p95, 50);
    // Retirement also makes duplicate completion harmless.
    tracker.complete(1, 4, 2'900'000);
    EXPECT_EQ(tracker.recentDemandLatency(2'900'000).count, 1u);
}

} // namespace
