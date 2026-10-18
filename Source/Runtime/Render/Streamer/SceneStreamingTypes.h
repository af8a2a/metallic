#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <memory>
#include <vector>

namespace metallic::scene { class Scene; }
namespace metallic::render {
class Buffer;
class TextureView;
struct StreamedImage;
class MeshletStreamRuntime;
class ScenePathTraceResources;
struct SceneStreamingState;

enum class SceneResourceFeatureBits : uint32_t {
    None = 0,
    Geometry = 1u << 0,
    Materials = 1u << 1,
    MaterialTextures = 1u << 2,
    Meshlets = 1u << 3,
    AccelerationStructure = 1u << 4,
};

constexpr SceneResourceFeatureBits operator|(
    SceneResourceFeatureBits lhs,
    SceneResourceFeatureBits rhs)
{
    return static_cast<SceneResourceFeatureBits>(
        static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

struct SceneResourceSnapshot {
    std::filesystem::path scenePath;
    SceneResourceFeatureBits features = SceneResourceFeatureBits::None;
    uint64_t sourceResourceIdentity = 0;
    uint64_t sourceStructuralRevision = 0;
    uint64_t sourceTransformRevision = 0;
    uint64_t sourceVisibilityRevision = 0;
    uint64_t sourceMaterialRevision = 0;
    std::shared_ptr<ScenePathTraceResources> pathTraceResources;
};

// Passes declare resource needs; StreamerSubsystem owns all geometry streaming.
struct SceneStreamingRequirements {
    SceneResourceFeatureBits features = SceneResourceFeatureBits::None;
    bool geometry = false;
    bool textureFeedback = false;
    bool sampledImage = false;
    bool operator==(const SceneStreamingRequirements&) const = default;
};

struct PreparedSceneResources {
    const scene::Scene* sourceScene = nullptr;
    std::shared_ptr<SceneResourceSnapshot> snapshot;
    std::shared_ptr<SceneStreamingState> state;
    std::string softwareRasterIdentity;
    bool ready = false;
    std::filesystem::path streamSourcePath;
    std::filesystem::path streamAssetPath;
    std::shared_ptr<MeshletStreamRuntime> geometry;
    std::vector<uint32_t> geometryOwnerMask;
    uint32_t mappedInstanceCount = 0;
    Buffer* textureFeedback = nullptr;
    bool textureFeedbackEnabled = false;
    std::shared_ptr<StreamedImage> image;
    std::filesystem::path imagePath;
    TextureView* imageView = nullptr;
};
} // namespace metallic::render
