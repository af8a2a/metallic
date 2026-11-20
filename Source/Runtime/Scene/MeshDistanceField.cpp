#include "Runtime/Scene/MeshDistanceField.h"

#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <type_traits>

namespace metallic::scene {
namespace {

constexpr std::array<char, 8> kMagic{'M', 'T', 'L', 'M', 'D', 'F', '0', '1'};
constexpr uint32_t kMaxIndirectionDimension = 1023;
constexpr uint32_t kBrickBytes =
    kMeshDistanceFieldBrickSize * kMeshDistanceFieldBrickSize * kMeshDistanceFieldBrickSize;
constexpr uint64_t kFingerprintOffset = 14695981039346656037ull;
constexpr uint64_t kFingerprintPrime = 1099511628211ull;

void hashBytes(uint64_t& hash, const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * kFingerprintPrime;
    }
}

template <typename T>
bool read(std::ifstream& file, T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

template <typename T>
void write(std::ofstream& file, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

bool validBounds(const MeshDistanceField& field)
{
    for (size_t axis = 0; axis != 3; ++axis) {
        if (!std::isfinite(field.localBoundsMin[axis]) ||
            !std::isfinite(field.localBoundsMax[axis] - field.localBoundsMin[axis]) ||
            field.localBoundsMax[axis] <= field.localBoundsMin[axis]) {
            return false;
        }
    }
    return true;
}

uint64_t indirectionCount(const MeshDistanceFieldMip& mip)
{
    uint64_t count = 1;
    for (uint32_t dimension : mip.indirectionDimensions) {
        if (dimension == 0 || dimension > kMaxIndirectionDimension) {
            return 0;
        }
        count *= dimension;
    }
    return count;
}

bool validMip(const MeshDistanceFieldMip& mip)
{
    const uint64_t count = indirectionCount(mip);
    const auto brickCount = mip.bricks.size() / kBrickBytes;
    const auto finite = [](float value) { return std::isfinite(value); };
    return count != 0 && mip.indirection.size() == count &&
        mip.bricks.size() % kBrickBytes == 0 && brickCount <= count &&
        std::ranges::all_of(mip.volumeToVirtualUVScale, [](float value) {
            return std::isfinite(value) && value > 0.0f;
        }) &&
        std::ranges::all_of(mip.volumeToVirtualUVAdd, finite) &&
        std::ranges::all_of(mip.distanceFieldToVolumeScaleBias, finite) &&
        mip.distanceFieldToVolumeScaleBias[0] > 0.0f &&
        std::ranges::all_of(mip.indirection, [brickCount](uint32_t index) {
            return index == kMeshDistanceFieldInvalidBrick || index < brickCount;
        });
}

} // namespace

bool getMeshDistanceFieldGeometry(const Scene& scene, int32_t meshIndex,
    MeshDistanceFieldGeometry& result, std::string& error)
{
    // Imported render primitives may be repeated for different instances.
    std::map<int32_t, const RenderPrimitive*> sections;
    for (const RenderPrimitive& primitive : scene.renderPrimitives()) {
        if (primitive.meshIndex == meshIndex) {
            sections.try_emplace(primitive.primitiveIndex, &primitive);
        }
    }
    if (meshIndex < 0 || sections.empty()) {
        error = "Mesh index does not identify any loaded source primitives";
        return false;
    }

    MeshDistanceFieldGeometry geometry;
    geometry.fingerprint = kFingerprintOffset;
    for (const auto& [sectionIndex, primitive] : sections) {
        const auto& materials = scene.materials();
        if (primitive->materialIndex >= 0 &&
            static_cast<size_t>(primitive->materialIndex) < materials.size()) {
            const auto& material = materials[primitive->materialIndex];
            if (material.alphaMode != "OPAQUE" || material.doubleSided ||
                material.displacementMagnitude != 0.0f) {
                error = "Mesh distance fields require single-sided, opaque, undisplaced source sections";
                return false;
            }
        }
        const uint64_t positionCount = primitive->positions.size();
        const uint64_t indexCount = primitive->indices.empty() ? positionCount : primitive->indices.size();
        if (primitive->mode != 4 || positionCount == 0 || indexCount == 0 || indexCount % 3 != 0 ||
            positionCount > std::numeric_limits<uint32_t>::max() - geometry.positions.size() ||
            std::ranges::any_of(primitive->indices, [positionCount](uint32_t index) { return index >= positionCount; })) {
            error = "Mesh distance fields require populated source triangles";
            return false;
        }
        const auto baseIndex = static_cast<uint32_t>(geometry.positions.size());
        const size_t firstIndex = geometry.indices.size();
        for (auto position : primitive->positions) {
            geometry.positions.push_back({position.x, position.y, position.z});
        }
        if (primitive->indices.empty()) {
            for (uint32_t index = 0; index < positionCount; ++index) {
                geometry.indices.push_back(baseIndex + index);
            }
        } else {
            for (uint32_t index : primitive->indices) {
                geometry.indices.push_back(baseIndex + index);
            }
        }
        hashBytes(geometry.fingerprint, &sectionIndex, sizeof(sectionIndex));
        hashBytes(geometry.fingerprint, &positionCount, sizeof(positionCount));
        hashBytes(geometry.fingerprint, geometry.positions.data() + baseIndex,
            positionCount * sizeof(geometry.positions[0]));
        hashBytes(geometry.fingerprint, &indexCount, sizeof(indexCount));
        hashBytes(geometry.fingerprint, geometry.indices.data() + firstIndex,
            indexCount * sizeof(geometry.indices[0]));
    }
    result = std::move(geometry);
    return true;
}

bool loadMeshDistanceField(const std::filesystem::path& path, MeshDistanceField& result, std::string& error)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "Could not open mesh distance field: " + path.string();
        return false;
    }
    const auto size = file.tellg();
    file.seekg(0);
    std::array<char, kMagic.size()> magic{};
    MeshDistanceField field;
    if (!read(file, magic) || magic != kMagic ||
        !read(file, field.sourceMeshIndex) || field.sourceMeshIndex < 0 ||
        !read(file, field.sourceFingerprint) ||
        !read(file, field.localBoundsMin) || !read(file, field.localBoundsMax) || !validBounds(field)) {
        error = "Invalid mesh distance field header: " + path.string();
        return false;
    }

    for (auto& mip : field.mips) {
        uint32_t brickCount = 0;
        if (!read(file, mip.indirectionDimensions) || !read(file, mip.volumeToVirtualUVScale) ||
            !read(file, mip.volumeToVirtualUVAdd) || !read(file, mip.distanceFieldToVolumeScaleBias) ||
            !read(file, brickCount)) {
            error = "Truncated mesh distance field mip header: " + path.string();
            return false;
        }
        const uint64_t count = indirectionCount(mip);
        const uint64_t tableBytes = count * sizeof(uint32_t);
        const uint64_t brickBytes = uint64_t(brickCount) * kBrickBytes;
        if (!count || brickCount > count || size - file.tellg() < std::streamoff(tableBytes + brickBytes)) {
            error = "Invalid mesh distance field mip size: " + path.string();
            return false;
        }
        mip.indirection.resize(static_cast<size_t>(count));
        mip.bricks.resize(static_cast<size_t>(brickBytes));
        file.read(reinterpret_cast<char*>(mip.indirection.data()), static_cast<std::streamsize>(tableBytes));
        if (brickBytes != 0) {
            file.read(reinterpret_cast<char*>(mip.bricks.data()), static_cast<std::streamsize>(brickBytes));
        }
        if (!file || !validMip(mip)) {
            error = "Invalid mesh distance field mip data: " + path.string();
            return false;
        }
    }
    if (file.tellg() != size) {
        error = "Unexpected data at the end of mesh distance field: " + path.string();
        return false;
    }
    result = std::move(field);
    return true;
}

bool saveMeshDistanceField(const std::filesystem::path& path, const MeshDistanceField& field, std::string& error)
{
    if (field.sourceMeshIndex < 0 || !validBounds(field) || !std::ranges::all_of(field.mips, validMip)) {
        error = "Invalid mesh distance field data";
        return false;
    }
    std::ofstream file(path, std::ios::binary);
    write(file, kMagic);
    write(file, field.sourceMeshIndex);
    write(file, field.sourceFingerprint);
    write(file, field.localBoundsMin);
    write(file, field.localBoundsMax);
    for (const auto& mip : field.mips) {
        write(file, mip.indirectionDimensions);
        write(file, mip.volumeToVirtualUVScale);
        write(file, mip.volumeToVirtualUVAdd);
        write(file, mip.distanceFieldToVolumeScaleBias);
        write(file, static_cast<uint32_t>(mip.bricks.size() / kBrickBytes));
        file.write(reinterpret_cast<const char*>(mip.indirection.data()),
            static_cast<std::streamsize>(mip.indirection.size() * sizeof(uint32_t)));
        if (!mip.bricks.empty()) {
            file.write(reinterpret_cast<const char*>(mip.bricks.data()), static_cast<std::streamsize>(mip.bricks.size()));
        }
    }
    file.close();
    if (!file) {
        error = "Could not write mesh distance field: " + path.string();
        return false;
    }
    return true;
}

} // namespace metallic::scene
