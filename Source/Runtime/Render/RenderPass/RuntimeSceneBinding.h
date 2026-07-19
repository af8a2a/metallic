#pragma once

#include "Runtime/Scene/Scene.h"

#include <filesystem>
#include <system_error>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {

inline std::filesystem::path resolvedScenePath(const std::filesystem::path& path)
{
    if (path.empty() || path.is_absolute()) {
        return path;
    }
    return std::filesystem::path(PROJECT_SOURCE_DIR) / path;
}

inline std::filesystem::path normalizedScenePath(const std::filesystem::path& path)
{
    const std::filesystem::path resolved = resolvedScenePath(path);
    std::error_code error;
    std::filesystem::path normalized = std::filesystem::weakly_canonical(resolved, error);
    if (!error) {
        return normalized;
    }
    normalized = std::filesystem::absolute(resolved, error);
    return error ? resolved.lexically_normal() : normalized.lexically_normal();
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
