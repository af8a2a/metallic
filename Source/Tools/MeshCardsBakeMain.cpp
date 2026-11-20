#include "Runtime/Scene/MeshCardsBake.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/Scene.h"

#include <charconv>
#include <cstdio>
#include <string_view>

namespace {

template<class T>
bool readInteger(std::string_view text, T& result)
{
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, result);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace metallic::scene;
    if (argc < 4) {
        std::fprintf(stderr, "Usage: MetallicMeshCardsBake <source.gltf|source.glb> <mesh-index> <output.meshcards> "
            "[--max-cards=N] [--distance-field=PATH]\n");
        return 1;
    }
    MeshCards field;
    if (!readInteger(argv[2], field.sourceMeshIndex) || field.sourceMeshIndex < 0) {
        std::fprintf(stderr, "Expected a nonnegative source mesh index\n");
        return 1;
    }

    MeshCardsBakeOptions options;
    std::filesystem::path distanceFieldPath;
    for (int i = 4; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg.starts_with("--max-cards=") && readInteger(arg.substr(12), options.maxCards)) { continue; }
        if (arg.starts_with("--distance-field=") && arg.size() > 17) {
            distanceFieldPath = arg.substr(17);
            continue;
        }
        std::fprintf(stderr, "Invalid bake option: %s\n", argv[i]);
        return 1;
    }

    Scene scene;
    if (!scene.loadDeferredMeshlets(argv[1], {})) {
        std::fprintf(stderr, "Could not load source mesh: %s\n", scene.lastLoadResult().error.c_str());
        return 1;
    }
    std::string error;
    MeshDistanceFieldGeometry geometry;
    if (!getMeshDistanceFieldGeometry(scene, field.sourceMeshIndex, geometry, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    field.sourceFingerprint = geometry.fingerprint;
    if (!distanceFieldPath.empty()) {
        MeshDistanceField distanceField;
        if (!loadMeshDistanceField(distanceFieldPath, distanceField, error) ||
            distanceField.sourceMeshIndex != field.sourceMeshIndex ||
            distanceField.sourceFingerprint != field.sourceFingerprint) {
            std::fprintf(stderr, "Distance field does not match source mesh: %s\n", error.c_str());
            return 1;
        }
        options.bounds = std::array{distanceField.localBoundsMin, distanceField.localBoundsMax};
    }

    if (!bakeMeshCards(geometry.positions, geometry.indices, options, field.cards, error) ||
        !saveMeshCards(argv[3], field, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("Baked %zu mesh cards\n", field.cards.size());
    return 0;
}
