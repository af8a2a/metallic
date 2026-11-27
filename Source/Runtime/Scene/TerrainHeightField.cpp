#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>

#include "Runtime/Scene/TerrainHeightField.h"
#include "Runtime/Scene/ScenePicker.h"

namespace metallic::scene {
namespace {

constexpr size_t kHeaderBytes = 32;
constexpr uint32_t kFormatVersion = 1;

uint32_t readUInt32(const std::array<uint8_t, kHeaderBytes>& header, size_t offset)
{
    return uint32_t(header[offset]) | (uint32_t(header[offset + 1]) << 8u) |
        (uint32_t(header[offset + 2]) << 16u) | (uint32_t(header[offset + 3]) << 24u);
}

float readFloat(const std::array<uint8_t, kHeaderBytes>& header, size_t offset)
{
    return std::bit_cast<float>(readUInt32(header, offset));
}

std::vector<float> makeAxis(uint32_t count, float lower, float upper)
{
    std::vector<float> values(count);
    const double spacing = (double(upper) - lower) / (count - 1);
    for (uint32_t i = 0; i < count - 1; ++i) {
        values[i] = static_cast<float>(lower + spacing * i);
    }
    values.back() = upper;
    return values;
}

size_t cellAt(const std::vector<float>& coordinates, double value)
{
    const auto upper = std::upper_bound(coordinates.begin(), coordinates.end() - 1, value);
    return static_cast<size_t>(upper - coordinates.begin() - 1);
}

double gradient(const std::vector<float>& values, size_t start, size_t stride,
    size_t index, size_t count, double spacing, bool quadraticEdges)
{
    const auto at = [&](size_t i) { return double(values[start + i * stride]); };
    if (index == 0) {
        const double first = at(1) - at(0);
        return (quadraticEdges ? (3.0 * first - (at(2) - at(1))) * 0.5 : first) / spacing;
    }
    if (index + 1 == count) {
        const double last = at(index) - at(index - 1);
        return (quadraticEdges ? (3.0 * last - (at(index - 1) - at(index - 2))) * 0.5 : last) / spacing;
    }
    return (at(index + 1) - at(index - 1)) / (2.0 * spacing);
}

bool clipAxis(double lower, double upper, double origin, double direction, double& entry, double& exit)
{
    if (direction == 0.0) {
        return lower <= origin && origin <= upper;
    }
    double a = (lower - origin) / direction;
    double b = (upper - origin) / direction;
    if (a > b) {
        std::swap(a, b);
    }
    entry = std::max(entry, a);
    exit = std::min(exit, b);
    return entry <= exit;
}

double nextBoundary(const std::vector<float>& coordinates, size_t cell, double origin, double direction)
{
    return direction == 0.0 ? std::numeric_limits<double>::infinity() :
        (coordinates[cell + (direction > 0.0)] - origin) / direction;
}

std::optional<double> crossingDistance(double start, double end, double first, double last, double tolerance)
{
    if (std::abs(first) <= tolerance) {
        return start;
    }
    if ((first < 0.0) != (last < 0.0)) {
        return std::lerp(start, end, first / (first - last));
    }
    if (std::abs(last) <= tolerance) {
        return end;
    }
    return std::nullopt;
}

bool finite(const float3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

} // namespace

std::optional<TerrainHeightField> TerrainHeightField::load(const std::filesystem::path& path, std::string& error)
{
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        error = "Cannot open terrain heightfield: " + path.string();
        return std::nullopt;
    }
    const std::streamoff size = stream.tellg();
    stream.seekg(0);
    std::array<uint8_t, kHeaderBytes> header;
    if (!stream.read(reinterpret_cast<char*>(header.data()), header.size()) ||
        !std::equal(header.begin(), header.begin() + 4, "MTHF") || readUInt32(header, 4) != kFormatVersion) {
        error = "Invalid MTHF terrain heightfield header: " + path.string();
        return std::nullopt;
    }
    const uint32_t columns = readUInt32(header, 8);
    const uint32_t rows = readUInt32(header, 12);
    const auto payloadSize = size - static_cast<std::streamoff>(kHeaderBytes);
    const uint64_t count = uint64_t(columns) * rows;
    if (columns < 2 || rows < 2 || payloadSize < 0 || payloadSize % sizeof(float) != 0 ||
        uint64_t(payloadSize / sizeof(float)) != count || count > std::vector<float>().max_size()) {
        error = "Invalid MTHF terrain heightfield dimensions or payload size: " + path.string();
        return std::nullopt;
    }
    std::vector<float> values(static_cast<size_t>(count));
    if (!stream.read(reinterpret_cast<char*>(values.data()), payloadSize)) {
        error = "Cannot read terrain heightfield samples: " + path.string();
        return std::nullopt;
    }
    if constexpr (std::endian::native == std::endian::big) {
        for (float& value : values) {
            value = std::bit_cast<float>(std::byteswap(std::bit_cast<uint32_t>(value)));
        }
    }
    return create(columns, rows, float2(readFloat(header, 16), readFloat(header, 20)),
        float2(readFloat(header, 24), readFloat(header, 28)), std::move(values), error);
}

std::optional<TerrainHeightField> TerrainHeightField::create(uint32_t columns, uint32_t rows,
    float2 minimumXZ, float2 maximumXZ, std::vector<float> worldHeights, std::string& error)
{
    if (columns < 2 || rows < 2 || uint64_t(columns) * rows != worldHeights.size() ||
        !std::isfinite(minimumXZ.x) || !std::isfinite(minimumXZ.y) ||
        !std::isfinite(maximumXZ.x) || !std::isfinite(maximumXZ.y) ||
        !(minimumXZ.x < maximumXZ.x && minimumXZ.y < maximumXZ.y) ||
        !std::all_of(worldHeights.begin(), worldHeights.end(), [](float height) { return std::isfinite(height); })) {
        error = "Terrain heightfield requires a finite, nonempty grid with strictly ordered world extents";
        return std::nullopt;
    }

    TerrainHeightField field;
    field.xCoordinates_ = makeAxis(columns, minimumXZ.x, maximumXZ.x);
    field.zCoordinates_ = makeAxis(rows, minimumXZ.y, maximumXZ.y);
    if (std::adjacent_find(field.xCoordinates_.begin(), field.xCoordinates_.end()) != field.xCoordinates_.end() ||
        std::adjacent_find(field.zCoordinates_.begin(), field.zCoordinates_.end()) != field.zCoordinates_.end()) {
        error = "Terrain heightfield grid spacing is below world coordinate precision";
        return std::nullopt;
    }

    const auto [minimum, maximum] = std::minmax_element(worldHeights.begin(), worldHeights.end());
    field.minimumHeight_ = *minimum;
    field.maximumHeight_ = *maximum;
    field.heights_ = std::move(worldHeights);
    field.normals_.reserve(field.heights_.size());
    const double spacingX = (double(maximumXZ.x) - minimumXZ.x) / (columns - 1);
    const double spacingZ = (double(maximumXZ.y) - minimumXZ.y) / (rows - 1);
    // Match the source mesh baker's central differences and one-sided edges.
    const bool quadraticEdges = columns >= 3 && rows >= 3;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            const double dx = gradient(field.heights_, row * columns, 1, column, columns, spacingX, quadraticEdges);
            const double dz = gradient(field.heights_, column, columns, row, rows, spacingZ, quadraticEdges);
            const double length = std::hypot(dx, 1.0, dz);
            field.normals_.emplace_back(static_cast<float>(-dx / length), static_cast<float>(1.0 / length),
                static_cast<float>(-dz / length));
        }
    }
    error.clear();
    return field;
}

TerrainHeightField::Triangle TerrainHeightField::triangle(size_t column, size_t row, double u, double v) const
{
    const size_t stride = xCoordinates_.size();
    const size_t topLeft = row * stride + column;
    if (u + v <= 1.0) {
        return {topLeft, topLeft + 1, topLeft + stride, u, v};
    }
    const size_t bottomRight = topLeft + stride + 1;
    return {bottomRight, bottomRight - 1, bottomRight - stride, 1.0 - u, 1.0 - v};
}

double TerrainHeightField::height(const Triangle& weights) const
{
    const double y = heights_[weights.center];
    return y + (heights_[weights.x] - y) * weights.weightX + (heights_[weights.z] - y) * weights.weightZ;
}

std::optional<TerrainSurfaceSample> TerrainHeightField::sample(float worldX, float worldZ) const
{
    if (!(xCoordinates_.front() <= worldX && worldX <= xCoordinates_.back()) ||
        !(zCoordinates_.front() <= worldZ && worldZ <= zCoordinates_.back())) {
        return std::nullopt;
    }
    const size_t column = cellAt(xCoordinates_, worldX);
    const size_t row = cellAt(zCoordinates_, worldZ);
    const double x0 = xCoordinates_[column];
    const double z0 = zCoordinates_[row];
    const auto weights = triangle(column, row, (worldX - x0) / (xCoordinates_[column + 1] - x0),
        (worldZ - z0) / (zCoordinates_[row + 1] - z0));
    const float u = static_cast<float>(weights.weightX);
    const float v = static_cast<float>(weights.weightZ);
    const float3 normal = normals_[weights.center] * (1.0f - u - v) + normals_[weights.x] * u + normals_[weights.z] * v;
    return TerrainSurfaceSample{float3(worldX, static_cast<float>(height(weights)), worldZ), normalize(normal)};
}

std::optional<TerrainRayHit> TerrainHeightField::raycast(const ScenePickRay& ray) const
{
    if (!finite(ray.origin) || !finite(ray.direction) || !std::isfinite(ray.minimumDistance) ||
        ray.minimumDistance < 0.0f || !(ray.maximumDistance >= ray.minimumDistance)) {
        return std::nullopt;
    }
    const double length = std::hypot(double(ray.direction.x), double(ray.direction.y), double(ray.direction.z));
    if (length == 0.0) {
        return std::nullopt;
    }
    const double dx = ray.direction.x / length;
    const double dy = ray.direction.y / length;
    const double dz = ray.direction.z / length;
    const double ox = ray.origin.x;
    const double oy = ray.origin.y;
    const double oz = ray.origin.z;
    double entry = ray.minimumDistance;
    double exit = ray.maximumDistance;
    if (!clipAxis(xCoordinates_.front(), xCoordinates_.back(), ox, dx, entry, exit) ||
        !clipAxis(zCoordinates_.front(), zCoordinates_.back(), oz, dz, entry, exit) ||
        !clipAxis(minimumHeight_, maximumHeight_, oy, dy, entry, exit)) {
        return std::nullopt;
    }

    const auto worldX = [&](double t) { return std::clamp(std::fma(dx, t, ox), double(xCoordinates_.front()), double(xCoordinates_.back())); };
    const auto worldZ = [&](double t) { return std::clamp(std::fma(dz, t, oz), double(zCoordinates_.front()), double(zCoordinates_.back())); };
    size_t column = cellAt(xCoordinates_, worldX(entry));
    size_t row = cellAt(zCoordinates_, worldZ(entry));
    while (true) {
        const double crossX = nextBoundary(xCoordinates_, column, ox, dx);
        const double crossZ = nextBoundary(zCoordinates_, row, oz, dz);
        const double cellExit = std::min({exit, crossX, crossZ});
        const double x0 = xCoordinates_[column];
        const double z0 = zCoordinates_[row];
        const double width = xCoordinates_[column + 1] - x0;
        const double depth = zCoordinates_[row + 1] - z0;
        const size_t stride = xCoordinates_.size();
        const size_t corner = row * stride + column;
        // Allow double-precision roundoff when a segment lands on a shared vertex.
        const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(double(heights_[corner])), std::abs(double(heights_[corner + 1])),
                std::abs(double(heights_[corner + stride])), std::abs(double(heights_[corner + stride + 1]))});
        const double u = (worldX(entry) - x0) / width;
        const double v = (worldZ(entry) - z0) / depth;
        const auto residual = [&](double t) {
            const double localX = (worldX(t) - x0) / width;
            const double localZ = (worldZ(t) - z0) / depth;
            return std::fma(dy, t, oy) - height(triangle(column, row, localX, localZ));
        };
        const double first = residual(entry);
        const double last = residual(cellExit);
        // Within a cell the height along the ray is affine on either side of its diagonal.
        const double diagonalRate = dx / width + dz / depth;
        const double diagonal = diagonalRate != 0.0 ? entry + (1.0 - u - v) / diagonalRate : entry;
        std::optional<double> hit;
        if (entry < diagonal && diagonal < cellExit) {
            const double middle = residual(diagonal);
            hit = crossingDistance(entry, diagonal, first, middle, tolerance);
            if (!hit) {
                hit = crossingDistance(diagonal, cellExit, middle, last, tolerance);
            }
        } else {
            hit = crossingDistance(entry, cellExit, first, last, tolerance);
        }
        if (hit) {
            const auto surface = sample(static_cast<float>(worldX(*hit)), static_cast<float>(worldZ(*hit)));
            return TerrainRayHit{surface->position, surface->normal, static_cast<float>(*hit)};
        }
        if (cellExit == exit) {
            return std::nullopt;
        }
        if (crossX == cellExit) {
            column = dx > 0.0 ? column + 1 : column - 1;
        }
        if (crossZ == cellExit) {
            row = dz > 0.0 ? row + 1 : row - 1;
        }
        entry = cellExit;
    }
}

} // namespace metallic::scene
