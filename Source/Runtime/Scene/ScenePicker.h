#pragma once

#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <limits>
#include <memory>

namespace metallic::scene {

struct ScenePickRay {
    float3 origin{0.0f, 0.0f, 0.0f};
    float3 direction{0.0f, 0.0f, -1.0f};
    // Inclusive world-space interval measured along the normalized ray direction.
    float minimumDistance = 0.0f;
    float maximumDistance = std::numeric_limits<float>::infinity();
};

struct ScenePickResult {
    SceneEntity object = kNullSceneEntity;
    int32_t nodeIndex = kInvalidSceneIndex;
    int32_t renderNodeIndex = kInvalidSceneIndex;
    int32_t renderPrimitiveIndex = kInvalidSceneIndex;
    uint32_t triangleIndex = 0;
    float distance = 0.0f;

    bool hit() const { return object != kNullSceneEntity; }
};

class ScenePicker {
public:
    ScenePicker();
    ~ScenePicker();

    ScenePicker(ScenePicker&&) noexcept;
    ScenePicker& operator=(ScenePicker&&) noexcept;

    ScenePicker(const ScenePicker&) = delete;
    ScenePicker& operator=(const ScenePicker&) = delete;

    ScenePickResult pick(const Scene& scene, const ScenePickRay& ray);
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::scene
