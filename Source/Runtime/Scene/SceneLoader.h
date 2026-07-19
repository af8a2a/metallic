#pragma once

#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Scene/SceneLoad.h"

#include <filesystem>
#include <memory>

namespace metallic::scene {

class SceneLoadHandle {
public:
    SceneLoadHandle() = default;

    bool valid() const;
    bool complete() const;
    SceneLoadProgress progress() const;
    bool cancel();
    std::unique_ptr<SceneDocument> takeResult();

private:
    struct State;
    explicit SceneLoadHandle(std::shared_ptr<State> state);
    void refreshTerminalState() const;

    std::shared_ptr<State> state_;

    friend class SceneLoader;
};

class SceneLoader {
public:
    SceneLoadHandle request(
        const std::filesystem::path& path,
        const SceneLoadOptions& options = {}) const;
};

} // namespace metallic::scene
