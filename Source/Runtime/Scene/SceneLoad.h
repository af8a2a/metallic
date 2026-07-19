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
    uint32_t decodeConcurrency = 0;
    uint64_t maxDecodedBytesInFlight = 512ull * 1024ull * 1024ull;
};

using SceneLoadProgressCallback = std::function<bool(const SceneLoadProgress&)>;

inline float clampSceneLoadFraction(float fraction)
{
    return std::clamp(fraction, 0.0f, 1.0f);
}

const char* sceneLoadPhaseName(SceneLoadPhase phase);
const char* sceneLoadStatusName(SceneLoadStatus status);

} // namespace metallic::scene
