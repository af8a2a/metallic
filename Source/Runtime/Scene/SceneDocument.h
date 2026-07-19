#pragma once

#include "Runtime/Scene/Scene.h"

#include <filesystem>
#include <string>

namespace metallic::scene {

class SceneDocument : public Scene {
public:
    bool load(const std::filesystem::path& path);
    bool load(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback);
    bool loadDeferredMeshlets(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback);
    void clear();
    bool save(std::string& message);
    bool revert(std::string& message);
    bool setNodeLocalMatrix(int32_t nodeIndex, const float4x4& localMatrix);

    bool dirty() const { return dirty_; }
    void setDirty(bool dirty) { dirty_ = dirty; }
    const std::filesystem::path& sourcePath() const { return sourcePath_; }
    const std::filesystem::path& documentPath() const { return documentPath_; }
    const std::string& documentWarning() const { return documentWarning_; }

    static std::filesystem::path sidecarPathForSource(const std::filesystem::path& sourcePath);

private:
    bool loadInternal(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback,
        bool deferMeshletBuild);
    bool applySidecar(const std::filesystem::path& path);

    std::filesystem::path sourcePath_;
    std::filesystem::path documentPath_;
    std::string documentWarning_;
    bool dirty_ = false;
};

} // namespace metallic::scene
