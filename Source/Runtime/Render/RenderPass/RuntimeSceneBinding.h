#pragma once

#include "Runtime/Scene/Scene.h"

#include <filesystem>
#include <system_error>

namespace metallic::render {

inline std::filesystem::path normalizedScenePath(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::path normalized = std::filesystem::weakly_canonical(path, error);
    if (!error) {
        return normalized;
    }
    normalized = std::filesystem::absolute(path, error);
    return error ? path.lexically_normal() : normalized.lexically_normal();
}

inline const scene::Scene* runtimeSceneForPath(
    const scene::Scene* runtimeScene,
    const std::filesystem::path& path)
{
    if (runtimeScene == nullptr || !runtimeScene->valid()) {
        return nullptr;
    }
    return normalizedScenePath(runtimeScene->filename()) == normalizedScenePath(path)
        ? runtimeScene
        : nullptr;
}

} // namespace metallic::render
