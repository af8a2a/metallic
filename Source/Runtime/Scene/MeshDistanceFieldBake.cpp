#include "Runtime/Scene/MeshDistanceFieldBake.h"

#include "Runtime/Task/ParallelFor.h"

#include <embree4/rtcore.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace metallic::scene {
namespace {

// UE 5.7.4 MeshDistanceFieldUtilities.cpp uses float storage for geometry and
// mip parameters, and double arithmetic for voxel placement and point projection.
template <typename T>
using Vector3 = std::array<T, 3>;

template <typename To, typename From>
Vector3<To> castVector(const Vector3<From>& value)
{
    return {static_cast<To>(value[0]), static_cast<To>(value[1]), static_cast<To>(value[2])};
}

template <typename T>
Vector3<T> operator+(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

template <typename T>
Vector3<T> operator-(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template <typename T>
Vector3<T> operator*(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

template <typename T>
Vector3<T> operator*(const Vector3<T>& value, T scale)
{
    return {value[0] * scale, value[1] * scale, value[2] * scale};
}

template <typename T>
Vector3<T> operator/(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
}

template <typename T>
T dot(const Vector3<T>& a, const Vector3<T>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
Vector3<T> cross(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

constexpr float kSmallNumber = 1.e-8f;
constexpr float kCentimetersPerMeter = 100.0f;
constexpr uint32_t kObjectBorder = 1;
constexpr uint32_t kMaxIndirectionDimension = 1024;
constexpr float kBandSizeInVoxels = 4.0f;
constexpr uint32_t kBrickSize = kMeshDistanceFieldBrickSize;
constexpr uint32_t kUniqueBrickSize = kMeshDistanceFieldUniqueBrickSize;
constexpr uint32_t kBrickVoxelCount = kBrickSize * kBrickSize * kBrickSize;
constexpr uint32_t kHemisphereGridSize = 7;
constexpr uint32_t kHemisphereSampleCount = kHemisphereGridSize * kHemisphereGridSize;

using Brick = std::array<uint8_t, kBrickVoxelCount>;
using SampleDirections = std::array<Vector3<float>, 2 * kHemisphereSampleCount>;

template <typename T>
Vector3<T> safeNormal(const Vector3<T>& vector)
{
    const T squareSum = dot(vector, vector);
    if (squareSum == 1.0f) { return vector; }
    if (squareSum < kSmallNumber) { return {}; }
    return vector * (T{1} / std::sqrt(squareSum));
}

int32_t roundToInt(float value)
{
#if defined(_M_X64) || defined(__x86_64__)
    return _mm_cvt_ss2si(_mm_set_ss(value + value + 0.5f)) >> 1;
#else
    return static_cast<int32_t>(std::floor(value + 0.5f));
#endif
}

int32_t floorToInt(float value)
{
#if defined(_M_X64) || defined(__x86_64__)
    return _mm_cvt_ss2si(_mm_set_ss(value + value - 0.5f)) >> 1;
#else
    return static_cast<int32_t>(std::floor(value));
#endif
}

float randomFraction(uint32_t& seed)
{
    seed = seed * 196314165u + 907633515u;
    return std::bit_cast<float>(0x3f800000u | (seed >> 9)) - 1.0f;
}

// Keep the reference's float and double transitions in the concentric mapping.
Vector3<float> uniformSampleHemisphere(double x, double y)
{
    x = x * 2.0f - 1.0f;
    y = y * 2.0f - 1.0f;
    if (x == 0.0 && y == 0.0) { return {}; }

    float radius;
    float theta;
    if (std::abs(x) > std::abs(y)) {
        radius = static_cast<float>(x);
        theta = static_cast<float>(std::numbers::pi_v<float> / 4 * (y / x));
    } else {
        radius = static_cast<float>(y);
        theta = static_cast<float>(std::numbers::pi_v<float> / 2 - std::numbers::pi_v<float> / 4 * (x / y));
    }
    const float u = radius * std::cos(theta);
    const float v = radius * std::sin(theta);
    const float radiusSquared = radius * radius;
    return {u * std::sqrt(2 - radiusSquared), v * std::sqrt(2 - radiusSquared), 1.0f - radiusSquared};
}

SampleDirections generateSampleDirections()
{
    uint32_t seed = 0;
    SampleDirections directions;
    for (uint32_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
        for (uint32_t x = 0; x < kHemisphereGridSize; ++x) {
            for (uint32_t y = 0; y < kHemisphereGridSize; ++y) {
                const float u1 = randomFraction(seed);
                const float u2 = randomFraction(seed);
                const float fraction1 = (static_cast<float>(x) + u1) / static_cast<float>(kHemisphereGridSize);
                const float fraction2 = (static_cast<float>(y) + u2) / static_cast<float>(kHemisphereGridSize);
                auto sample = uniformSampleHemisphere(fraction1, fraction2);
                if (hemisphere == 1) { sample[2] *= -1.0f; }
                directions[hemisphere * kHemisphereSampleCount + x * kHemisphereGridSize + y] = sample;
            }
        }
    }
    return directions;
}

Vector3<double> closestPointOnSegment(const Vector3<double>& point, const Vector3<double>& start,
    const Vector3<double>& end)
{
    const auto segment = end - start;
    const auto toPoint = point - start;
    const double dot1 = dot(toPoint, segment);
    if (dot1 <= 0.0) { return start; }
    const double dot2 = dot(segment, segment);
    if (dot2 <= dot1) { return end; }
    return start + segment * (dot1 / dot2);
}

Vector3<double> closestPointOnTriangle(const Vector3<double>& point, const Vector3<double>& a,
    const Vector3<double>& b, const Vector3<double>& c)
{
    const auto ba = a - b;
    const auto ac = c - a;
    const auto cb = b - c;
    const auto triangleNormal = cross(ba, cb);
    const Vector3<double> planeOrigins[] = {b, a, c};
    const Vector3<double> planeNormals[] = {cross(triangleNormal, ba), cross(triangleNormal, ac),
        cross(triangleNormal, cb)};
    uint32_t planeMask = 0;
    for (uint32_t planeIndex = 0; planeIndex < 3; ++planeIndex) {
        if (dot(planeNormals[planeIndex], point) - dot(planeOrigins[planeIndex], planeNormals[planeIndex]) > 0.0f) {
            planeMask |= 1u << planeIndex;
        }
    }
    switch (planeMask) {
    case 0: {
        const auto normal = safeNormal(cross(b - a, c - a));
        return point - normal * (dot(normal, point) - dot(a, normal));
    }
    case 1: return closestPointOnSegment(point, b, a);
    case 2: return closestPointOnSegment(point, a, c);
    case 3: return a;
    case 4: return closestPointOnSegment(point, b, c);
    case 5: return b;
    case 6: return c;
    default: return point;
    }
}

struct EmbreeScene {
    std::vector<Vector3<float>> vertices;
    std::vector<uint32_t> indices;
    RTCDevice device = nullptr;
    RTCScene scene = nullptr;

    ~EmbreeScene()
    {
        if (scene) { rtcReleaseScene(scene); }
        if (device) { rtcReleaseDevice(device); }
    }
};

struct PointQueryData {
    const EmbreeScene& geometry;
    float closestDistanceSquared;
};

bool pointQuery(RTCPointQueryFunctionArguments* arguments)
{
    auto& data = *static_cast<PointQueryData*>(arguments->userPtr);
    const auto& indices = data.geometry.indices;
    const auto& vertices = data.geometry.vertices;
    const size_t index = size_t{arguments->primID} * 3;
    const Vector3<float> queryPosition{arguments->query->x, arguments->query->y, arguments->query->z};
    const auto closestPoint = castVector<float>(closestPointOnTriangle(castVector<double>(queryPosition),
        castVector<double>(vertices[indices[index]]), castVector<double>(vertices[indices[index + 1]]),
        castVector<double>(vertices[indices[index + 2]])));
    const auto difference = closestPoint - queryPosition;
    const float distanceSquared = dot(difference, difference);
    if (distanceSquared < data.closestDistanceSquared) {
        data.closestDistanceSquared = distanceSquared;
        arguments->query->radius = std::sqrt(distanceSquared);
        return true;
    }
    return false;
}

// Keep the reference's geometry-filter path. The baker accepts every
// intersection of its single-sided input triangles.
void intersectFilter(const RTCFilterFunctionNArguments*)
{
}

bool initializeEmbree(EmbreeScene& scene, std::string& error)
{
    scene.device = rtcNewDevice(nullptr);
    if (!scene.device) {
        error = "Could not create an Embree device (error " + std::to_string(rtcGetDeviceError(nullptr)) + ")";
        return false;
    }
    const auto version = rtcGetDeviceProperty(scene.device, RTC_DEVICE_PROPERTY_VERSION);
    if (version != 40303) {
        error = "Mesh distance field baking requires Embree 4.3.3 (device version " + std::to_string(version) + ")";
        return false;
    }
    scene.scene = rtcNewScene(scene.device);
    if (!scene.scene) {
        error = "Could not create the Embree scene (error " + std::to_string(rtcGetDeviceError(scene.device)) + ")";
        return false;
    }
    rtcSetSceneFlags(scene.scene, RTC_SCENE_FLAG_NONE);
    RTCGeometry geometry = rtcNewGeometry(scene.device, RTC_GEOMETRY_TYPE_TRIANGLE);
    if (!geometry) {
        error = "Could not create the Embree geometry (error " + std::to_string(rtcGetDeviceError(scene.device)) + ")";
        return false;
    }
    // Embree can read one extra float from the final vertex.
    rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, scene.vertices.data(),
        0, sizeof(Vector3<float>), scene.vertices.size() - 1);
    rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, scene.indices.data(),
        0, sizeof(uint32_t) * 3, scene.indices.size() / 3);
    rtcSetGeometryIntersectFilterFunction(geometry, intersectFilter);
    rtcCommitGeometry(geometry);
    rtcAttachGeometry(scene.scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(scene.scene);
    if (const RTCError code = rtcGetDeviceError(scene.device); code != RTC_ERROR_NONE) {
        error = "Could not commit the Embree scene (error " + std::to_string(code) + ")";
        return false;
    }
    return true;
}

bool lineBoxIntersection(const Vector3<double>& minimum, const Vector3<double>& maximum,
    const Vector3<double>& start, const Vector3<double>& end, const Vector3<double>& direction)
{
    Vector3<double> time{};
    bool startIsOutside = false;
    for (size_t axis = 0; axis < 3; ++axis) {
        const double inverseDirection = direction[axis] != 0.0 ? 1.0 / direction[axis] : 3.4e+38f;
        if (start[axis] < minimum[axis]) {
            startIsOutside = true;
            if (end[axis] < minimum[axis]) { return false; }
            time[axis] = (minimum[axis] - start[axis]) * inverseDirection;
        } else if (start[axis] > maximum[axis]) {
            startIsOutside = true;
            if (end[axis] > maximum[axis]) { return false; }
            time[axis] = (maximum[axis] - start[axis]) * inverseDirection;
        }
    }
    if (!startIsOutside) { return true; }

    const double maxTime = std::max({time[0], time[1], time[2]});
    if (maxTime < 0.0 || maxTime > 1.0) { return false; }
    const auto hit = start + direction * maxTime;
    constexpr double kBoxSideThreshold = 0.1f;
    return hit[0] > minimum[0] - kBoxSideThreshold && hit[0] < maximum[0] + kBoxSideThreshold &&
        hit[1] > minimum[1] - kBoxSideThreshold && hit[1] < maximum[1] + kBoxSideThreshold &&
        hit[2] > minimum[2] - kBoxSideThreshold && hit[2] < maximum[2] + kBoxSideThreshold;
}

float signedDistance(const EmbreeScene& scene, const SampleDirections& directions, float traceDistance,
    const Vector3<double>& volumeMinimum, const Vector3<double>& volumeMaximum, const Vector3<double>& voxelPosition)
{
    const auto position = castVector<float>(voxelPosition);
    RTCPointQuery query{position[0], position[1], position[2], 0.0f, traceDistance};
    RTCPointQueryContext pointContext;
    rtcInitPointQueryContext(&pointContext);
    PointQueryData queryData{scene, (traceDistance * 2.0f) * (traceDistance * 2.0f)};
    rtcPointQuery(scene.scene, &query, &pointContext, pointQuery, &queryData);

    const float closestDistance = std::sqrt(queryData.closestDistanceSquared);
    float minDistance = std::min(traceDistance, closestDistance);
    if (closestDistance <= traceDistance) {
        uint32_t hitCount = 0;
        uint32_t backfaceCount = 0;
        for (const auto& sample : directions) {
            const auto unitRayDirection = castVector<double>(sample);
            constexpr float kPullbackEpsilon = 1.e-4f;
            const auto start = voxelPosition - unitRayDirection * static_cast<double>(kPullbackEpsilon * traceDistance);
            const auto end = voxelPosition + unitRayDirection * static_cast<double>(traceDistance);
            if (!lineBoxIntersection(volumeMinimum, volumeMaximum, voxelPosition, end, unitRayDirection)) {
                continue;
            }
            const auto origin = castVector<float>(start);
            const auto rayDirection = castVector<float>(end - voxelPosition);

            RTCRayHit ray{};
            ray.ray.org_x = origin[0];
            ray.ray.org_y = origin[1];
            ray.ray.org_z = origin[2];
            ray.ray.dir_x = rayDirection[0];
            ray.ray.dir_y = rayDirection[1];
            ray.ray.dir_z = rayDirection[2];
            ray.ray.tnear = 0.0f;
            ray.ray.tfar = 1.0f;
            ray.ray.mask = UINT32_MAX;
            ray.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            ray.hit.primID = RTC_INVALID_GEOMETRY_ID;
            ray.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

            RTCRayQueryContext rayContext;
            rtcInitRayQueryContext(&rayContext);
            RTCIntersectArguments arguments;
            rtcInitIntersectArguments(&arguments);
            arguments.context = &rayContext;
            rtcIntersect1(scene.scene, &ray, &arguments);

            if (ray.hit.geomID != RTC_INVALID_GEOMETRY_ID && ray.hit.primID != RTC_INVALID_GEOMETRY_ID) {
                ++hitCount;
                const auto hitNormal = safeNormal(Vector3<float>{-ray.hit.Ng_x, -ray.hit.Ng_y, -ray.hit.Ng_z});
                if (dot(unitRayDirection, castVector<double>(hitNormal)) > 0.0) { ++backfaceCount; }
            }
        }
        if (hitCount > 0 && static_cast<float>(backfaceCount) > 0.25f * static_cast<float>(directions.size())) {
            minDistance *= -1.0f;
        }
    }
    return minDistance;
}

bool bakeMip(const EmbreeScene& scene, const SampleDirections& directions, const Vector3<float>& boundsMin,
    const Vector3<float>& boundsMax, float maxExtent, const Vector3<uint32_t>& dimensions,
    MeshDistanceFieldMip& result, std::string& error)
{
    constexpr float uniqueBrickSize = static_cast<float>(kUniqueBrickSize);
    const float localToVolume = 1.0f / maxExtent;
    const Vector3<uint32_t> indirectionVoxels = {dimensions[0] * kUniqueBrickSize,
        dimensions[1] * kUniqueBrickSize, dimensions[2] * kUniqueBrickSize};
    const Vector3<uint32_t> interiorVoxels = {indirectionVoxels[0] - 2 * kObjectBorder,
        indirectionVoxels[1] - 2 * kObjectBorder, indirectionVoxels[2] - 2 * kObjectBorder};
    const auto texelSize = (boundsMax - boundsMin) / castVector<float>(interiorVoxels);
    const auto volumeMinimum = boundsMin - texelSize;
    const auto volumeMaximum = boundsMax + texelSize;
    const auto indirectionVoxelSize = (volumeMaximum - volumeMinimum) / castVector<float>(dimensions);
    const auto volumeIndirectionSize = indirectionVoxelSize * localToVolume;
    // The UE 5.7 Win64 cooker factors the rounded brick reciprocal out of
    // the squared sum. Preserve that order for the serialized scale and bias.
    constexpr float inverseBrickSize = 1.0f / uniqueBrickSize;
    constexpr float inverseBrickSizeSquared = inverseBrickSize * inverseBrickSize;
    const float maxEncodedDistance = std::sqrt(dot(volumeIndirectionSize, volumeIndirectionSize) *
        inverseBrickSizeSquared) * kBandSizeInVoxels;
    const float traceDistance = maxEncodedDistance * maxExtent;
    const float distanceScale = 2.0f * maxEncodedDistance;
    const float distanceBias = -maxEncodedDistance;
    if (!std::isfinite((traceDistance * 2.0f) * (traceDistance * 2.0f)) || traceDistance <= 0.0f) {
        error = "Mesh bounds are outside the supported distance field bake range";
        return false;
    }

    const auto minimum = castVector<double>(volumeMinimum);
    const auto maximum = castVector<double>(volumeMaximum);
    const auto placementIndirectionSize = (maximum - minimum) / castVector<double>(dimensions);
    const auto placementVoxelSize = placementIndirectionSize / Vector3<double>{uniqueBrickSize, uniqueBrickSize, uniqueBrickSize};
    const uint32_t planeSize = dimensions[0] * dimensions[1];
    const uint32_t brickCount = planeSize * dimensions[2];
    std::vector<Brick> candidates(brickCount);
    std::vector<uint8_t> occupied(brickCount);
    const auto outcome = task::parallelFor(0, brickCount, {.minGrain = 1},
        [&](uint32_t begin, uint32_t end, task::WorkerContext&) {
            for (uint32_t index = begin; index < end; ++index) {
                const Vector3<uint32_t> coordinate = {index % dimensions[0], (index / dimensions[0]) % dimensions[1],
                    index / planeSize};
                const auto brickMinimum = minimum + castVector<double>(coordinate) * placementIndirectionSize;
                uint8_t brickMin = UINT8_MAX;
                uint8_t brickMax = 0;
                Brick& brick = candidates[index];
                for (uint32_t z = 0; z < kBrickSize; ++z) {
                    for (uint32_t y = 0; y < kBrickSize; ++y) {
                        for (uint32_t x = 0; x < kBrickSize; ++x) {
                            const auto voxelPosition = Vector3<double>{double(x), double(y), double(z)} *
                                placementVoxelSize + brickMinimum;
                            const float localDistance = signedDistance(scene, directions, traceDistance,
                                minimum, maximum, voxelPosition);
                            const float volumeDistance = localDistance * localToVolume;
                            const float rescaledDistance = static_cast<float>((double(volumeDistance) - double(distanceBias)) /
                                double(distanceScale));
                            const auto sample = static_cast<uint8_t>(std::clamp(floorToInt(rescaledDistance * 255.0f + 0.5f), 0, 255));
                            brick[(z * kBrickSize + y) * kBrickSize + x] = sample;
                            brickMin = std::min(brickMin, sample);
                            brickMax = std::max(brickMax, sample);
                        }
                    }
                }
                occupied[index] = brickMin < UINT8_MAX && brickMax > 0;
            }
        });
    if (!outcome) {
        error = outcome.error();
        return false;
    }

    result.indirectionDimensions = dimensions;
    result.indirection.resize(brickCount, kMeshDistanceFieldInvalidBrick);
    result.distanceFieldToVolumeScaleBias = {distanceScale, distanceBias};

    result.bricks.reserve(static_cast<size_t>(std::count(occupied.begin(), occupied.end(), uint8_t{1})) * kBrickVoxelCount);
    for (uint32_t index = 0; index < brickCount; ++index) {
        if (occupied[index]) {
            result.indirection[index] = static_cast<uint32_t>(result.bricks.size() / kBrickVoxelCount);
            result.bricks.insert(result.bricks.end(), candidates[index].begin(), candidates[index].end());
        }
    }

    constexpr float objectBorder = static_cast<float>(kObjectBorder);
    const auto virtualUVMin = Vector3<float>{objectBorder, objectBorder, objectBorder} / castVector<float>(indirectionVoxels);
    const auto virtualUVSize = castVector<float>(interiorVoxels) / castVector<float>(indirectionVoxels);
    const auto volumePositionExtent = (boundsMax - boundsMin) * 0.5f * localToVolume;
    result.volumeToVirtualUVScale = virtualUVSize / (volumePositionExtent * 2.0f);
    result.volumeToVirtualUVAdd = volumePositionExtent * result.volumeToVirtualUVScale + virtualUVMin;
    return true;
}

} // namespace

bool bakeMeshDistanceField(std::span<const std::array<float, 3>> positions, std::span<const uint32_t> indices,
    const MeshDistanceFieldBakeOptions& options, MeshDistanceField& result, std::string& error)
{
    if (positions.empty() || positions.size() > UINT32_MAX ||
        indices.empty() || indices.size() % 3 != 0 || indices.size() / 3 > UINT32_MAX) {
        error = "Mesh distance field baking requires indexed triangles";
        return false;
    }
    const float voxelDensity = options.voxelDensity * options.resolutionScale;
    if (options.voxelDensity <= 0.0f || options.resolutionScale <= 0.0f ||
        !std::isfinite(voxelDensity) || voxelDensity <= 0.0f || options.maxResolution < 8) {
        error = "Mesh distance field baking requires positive density and resolution scale and a maximum resolution of at least 8";
        return false;
    }

    EmbreeScene scene;
    scene.vertices.reserve(positions.size() + 1);
    constexpr float maxFloat = std::numeric_limits<float>::max();
    Vector3<float> boundsMin{maxFloat, maxFloat, maxFloat};
    Vector3<float> boundsMax{-maxFloat, -maxFloat, -maxFloat};
    for (const auto& position : positions) {
        const auto centimeters = position * kCentimetersPerMeter;
        for (size_t axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(centimeters[axis])) {
                error = "Mesh distance field positions must be finite in centimeters";
                return false;
            }
            boundsMin[axis] = std::min(boundsMin[axis], centimeters[axis]);
            boundsMax[axis] = std::max(boundsMax[axis], centimeters[axis]);
        }
        scene.vertices.push_back(centimeters);
    }
    scene.vertices.emplace_back();

    scene.indices.reserve(indices.size());
    for (size_t index = 0; index < indices.size(); index += 3) {
        const uint32_t i0 = indices[index];
        // The reference uses clockwise triangles and negates Embree's hit normal.
        const uint32_t i1 = indices[index + 2];
        const uint32_t i2 = indices[index + 1];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
            error = "Mesh distance field triangle index is out of range";
            return false;
        }
        const auto& v0 = scene.vertices[i0];
        const auto& v1 = scene.vertices[i1];
        const auto& v2 = scene.vertices[i2];
        const auto normal = cross(v1 - v2, v0 - v2);
        if (dot(normal, normal) < kSmallNumber) { continue; }
        scene.indices.insert(scene.indices.end(), {i0, i1, i2});
    }
    if (scene.indices.empty()) {
        error = "No triangles remain after removing degenerate triangles";
        return false;
    }

    // Match the reference box-sphere bounds round trip before enforcing the
    // one-centimeter minimum half extent.
    const auto sourceExtent = (boundsMax - boundsMin) * 0.5f;
    const auto sourceOrigin = boundsMin + sourceExtent;
    boundsMin = sourceOrigin - sourceExtent;
    boundsMax = sourceOrigin + sourceExtent;
    const auto center = (boundsMin + boundsMax) * 0.5f;
    auto extent = (boundsMax - boundsMin) * 0.5f;
    for (auto& axis : extent) { axis = std::max(axis, 1.0f); }
    boundsMin = center - extent;
    boundsMax = center + extent;
    const auto paddedExtent = (boundsMax - boundsMin) * 0.5f;
    if (!std::isfinite(dot(paddedExtent, paddedExtent)) ||
        std::min({paddedExtent[0], paddedExtent[1], paddedExtent[2]}) <= 0.0f) {
        error = "Mesh bounds are outside the supported distance field bake range";
        return false;
    }

    const float maxExtent = std::max({paddedExtent[0], paddedExtent[1], paddedExtent[2]});
    const uint32_t perMeshMax = options.resolutionScale <= 1.0f ? options.maxResolution / 2 : options.maxResolution;
    const auto maxBlocks = static_cast<uint32_t>(std::min<uint64_t>(
        (uint64_t{perMeshMax} + kUniqueBrickSize / 2) / kUniqueBrickSize, kMaxIndirectionDimension - 1));
    constexpr float uniqueBrickSize = static_cast<float>(kUniqueBrickSize);
    const auto desiredDimensions = ((boundsMax - boundsMin) * voxelDensity) /
        Vector3<float>{uniqueBrickSize, uniqueBrickSize, uniqueBrickSize};
    Vector3<uint32_t> mipZeroDimensions{};
    for (size_t axis = 0; axis < 3; ++axis) {
        // The x64 reference doubles the input before integer conversion.
        if (!(desiredDimensions[axis] < 0x1p30f)) {
            error = "Mesh bounds and voxel density exceed the supported distance field bake range";
            return false;
        }
        mipZeroDimensions[axis] = static_cast<uint32_t>(std::clamp(roundToInt(desiredDimensions[axis]),
            1, static_cast<int32_t>(maxBlocks)));
    }

    if (!initializeEmbree(scene, error)) { return false; }
    const auto directions = generateSampleDirections();
    MeshDistanceField field;
    for (size_t axis = 0; axis < 3; ++axis) {
        field.localBoundsMin[axis] = boundsMin[axis] / kCentimetersPerMeter;
        field.localBoundsMax[axis] = boundsMax[axis] / kCentimetersPerMeter;
    }
    for (uint32_t mip = 0; mip < kMeshDistanceFieldMipCount; ++mip) {
        const uint32_t divisor = 1u << mip;
        const Vector3<uint32_t> dimensions = {(mipZeroDimensions[0] + divisor - 1) / divisor,
            (mipZeroDimensions[1] + divisor - 1) / divisor, (mipZeroDimensions[2] + divisor - 1) / divisor};
        if (!bakeMip(scene, directions, boundsMin, boundsMax, maxExtent, dimensions, field.mips[mip], error)) {
            return false;
        }
    }
    result = std::move(field);
    return true;
}

} // namespace metallic::scene
