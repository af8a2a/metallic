#include "Runtime/Scene/MeshDistanceFieldBake.h"
#include "Runtime/Scene/scene.h"

#include <charconv>
#include <cstdio>
#include <string_view>

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "Usage: MetallicMeshDistanceFieldBake <source.gltf|source.glb> <mesh-index> "
            "<output.meshdf> [resolution-scale]\n");
        return 1;
    }
    int32_t meshIndex = -1;
    const std::string_view indexArgument = argv[2];
    const char* indexEnd = indexArgument.data() + indexArgument.size();
    const auto index = std::from_chars(indexArgument.data(), indexEnd, meshIndex);
    if (index.ec != std::errc{} || index.ptr != indexEnd || meshIndex < 0) {
        std::fprintf(stderr, "Expected a nonnegative source mesh index\n");
        return 1;
    }

    metallic::scene::MeshDistanceFieldBakeOptions options;
    if (argc == 5) {
        const std::string_view scaleArgument = argv[4];
        const char* scaleEnd = scaleArgument.data() + scaleArgument.size();
        const auto scale = std::from_chars(scaleArgument.data(), scaleEnd, options.resolutionScale);
        if (scale.ec != std::errc{} || scale.ptr != scaleEnd) {
            std::fprintf(stderr, "Expected a positive resolution scale\n");
            return 1;
        }
    }

    const std::filesystem::path source = argv[1];
    if (source.extension() != ".gltf" && source.extension() != ".glb") {
        std::fprintf(stderr, "Expected a glTF or GLB source mesh\n");
        return 1;
    }
    metallic::scene::Scene scene;
    if (!scene.loadDeferredMeshlets(source, {})) {
        std::fprintf(stderr, "Could not load source mesh: %s\n", scene.lastLoadResult().error.c_str());
        return 1;
    }

    std::string error;
    metallic::scene::MeshDistanceFieldGeometry geometry;
    metallic::scene::MeshDistanceField field;
    if (!metallic::scene::getMeshDistanceFieldGeometry(scene, meshIndex, geometry, error) ||
        !metallic::scene::bakeMeshDistanceField(geometry.positions, geometry.indices, options, field, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    field.sourceMeshIndex = meshIndex;
    field.sourceFingerprint = geometry.fingerprint;
    if (!metallic::scene::saveMeshDistanceField(argv[3], field, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    for (size_t i = 0; i < field.mips.size(); ++i) {
        const auto& mip = field.mips[i];
        const auto& dimensions = mip.indirectionDimensions;
        constexpr uint32_t brickSize = metallic::scene::kMeshDistanceFieldBrickSize;
        std::printf("Mip %zu: %ux%ux%u, %zu bricks\n", i,
            dimensions[0], dimensions[1], dimensions[2], mip.bricks.size() / (brickSize * brickSize * brickSize));
    }
    return 0;
}
