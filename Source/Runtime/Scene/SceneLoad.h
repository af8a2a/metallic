#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace metallic::scene {

enum class SceneLoadPhase : uint8_t {
    Idle,
    Queued,
    Parsing,
    Geometry,
    Images,
    GpuUpload,
    AccelerationStructures,
    Finalizing,
    Completed,
    Failed,
    Cancelled,
};

enum class SceneLoadStatus : uint8_t {
    Idle,
    Running,
    Succeeded,
    Failed,
    Cancelled,
};

struct SceneLoadProgress {
    SceneLoadStatus status = SceneLoadStatus::Idle;
    SceneLoadPhase phase = SceneLoadPhase::Idle;
    float fraction = 0.0f;
    uint64_t completedUnits = 0;
    uint64_t totalUnits = 0;
    std::string currentItem;
    std::string error;
    std::chrono::steady_clock::duration elapsed{};
};

struct SceneLoadOptions {
    // Independent stage limits, clamped to available workers. Zero selects up
    // to eight workers while leaving one worker for other tasks when possible.
    uint32_t decodeConcurrency = 0;
    // Estimated pixel working set for unfinished decode/mip jobs. Completed scene
    // pixels and codec-internal scratch are excluded. Zero disables this limit;
    // an image exceeding it runs alone. 1 GiB accommodates eight 4K RGBA decodes.
    uint64_t maxDecodedBytesInFlight = 1024ull * 1024ull * 1024ull;
    uint32_t mipConcurrency = 0;
};

using SceneLoadProgressCallback = std::function<bool(const SceneLoadProgress&)>;

inline float clampSceneLoadFraction(float fraction)
{
    return std::clamp(fraction, 0.0f, 1.0f);
}

const char* sceneLoadPhaseName(SceneLoadPhase phase);
const char* sceneLoadStatusName(SceneLoadStatus status);

} // namespace metallic::scene
