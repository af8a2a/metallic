#include "Runtime/Scene/MeshCardsBake.h"

#include <embree4/rtcore.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace metallic::scene {
namespace {

// UE 5.7.4 MeshCardRepresentationUtilities.cpp performs the build in centimeters.
constexpr float kCentimetersPerMeter = 100.0f;
constexpr float kSmallNumber = 1.e-8f;
constexpr float kMaxFloat = std::numeric_limits<float>::max();
constexpr uint32_t kNumAxisAlignedDirections = 6;
constexpr uint32_t kMaxCardsPerMesh = 32;

template <typename T> using Vector3 = std::array<T, 3>;
using Int3 = Vector3<int32_t>;
using Float3 = Vector3<float>;
using Double3 = Vector3<double>;

template <typename T> Vector3<T> operator+(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

template <typename T> Vector3<T> operator-(const Vector3<T>& a, const Vector3<T>& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template <typename T> Vector3<T> operator-(const Vector3<T>& value)
{
    return {-value[0], -value[1], -value[2]};
}

template <typename T> Vector3<T> operator*(const Vector3<T>& value, T scale)
{
    return {value[0] * scale, value[1] * scale, value[2] * scale};
}

template <typename T> Vector3<T> operator/(const Vector3<T>& value, T scale)
{
    return {value[0] / scale, value[1] / scale, value[2] / scale};
}

template <typename To, typename From> Vector3<To> castVector(const Vector3<From>& value)
{
    return {static_cast<To>(value[0]), static_cast<To>(value[1]), static_cast<To>(value[2])};
}

template <typename T> T dot(const Vector3<T>& a, const Vector3<T>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Float3 cross(const Float3& a, const Float3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

struct Box {
    Double3 min;
    Double3 max;

    Double3 center() const { return (min + max) * 0.5; }
    Double3 extent() const { return (max - min) * 0.5; }
};

struct Rotation {
    Float3 x;
    Float3 y;
    Float3 z;

    Float3 transform(const Float3& point) const
    {
        return (x * point[0] + y * point[1]) + z * point[2];
    }
};

Box transformBoxToLocal(const Box& box, const Rotation& rotation, const Float3& offset)
{
    const Box shifted{box.min - castVector<double>(offset), box.max - castVector<double>(offset)};
    const Double3 center = shifted.center();
    const Double3 extent = shifted.extent();
    const Double3 resultCenter{dot(center, castVector<double>(rotation.x)),
        dot(center, castVector<double>(rotation.y)), dot(center, castVector<double>(rotation.z))};
    const auto absoluteDot = [&](const Float3& axis) {
        return std::abs(extent[0] * axis[0]) + std::abs(extent[1] * axis[1]) + std::abs(extent[2] * axis[2]);
    };
    const Double3 resultExtent{absoluteDot(rotation.x), absoluteDot(rotation.y), absoluteDot(rotation.z)};
    return {resultCenter - resultExtent, resultCenter + resultExtent};
}

struct EmbreeScene {
    std::vector<Float3> vertices;
    std::vector<uint32_t> indices;
    RTCDevice device = nullptr;
    RTCScene scene = nullptr;

    ~EmbreeScene()
    {
        if (scene) { rtcReleaseScene(scene); }
        if (device) { rtcReleaseDevice(device); }
    }
};

struct RayContext : RTCRayQueryContext {
    uint32_t skipPrimId;
};

void intersectFilter(const RTCFilterFunctionNArguments* arguments)
{
    const auto& context = *static_cast<const RayContext*>(arguments->context);
    if (context.skipPrimId != RTC_INVALID_GEOMETRY_ID &&
        context.skipPrimId == RTCHitN_primID(arguments->hit, 1, 0)) {
        arguments->valid[0] = 0;
    }
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
        error = "Mesh Cards baking requires Embree 4.3.3 (device version " + std::to_string(version) + ")";
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
    // Embree reads one extra float from the final vertex.
    rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, scene.vertices.data(),
        0, sizeof(Float3), scene.vertices.size() - 1);
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

RTCRayHit traceRay(const EmbreeScene& scene, const Float3& origin, const Float3& direction,
    float nearDistance, uint32_t skipPrimId = RTC_INVALID_GEOMETRY_ID)
{
    RTCRayHit result{};
    result.ray.org_x = origin[0];
    result.ray.org_y = origin[1];
    result.ray.org_z = origin[2];
    result.ray.dir_x = direction[0];
    result.ray.dir_y = direction[1];
    result.ray.dir_z = direction[2];
    result.ray.tnear = nearDistance;
    result.ray.tfar = kMaxFloat;
    result.ray.mask = 0xffffffffu;
    result.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    result.hit.primID = RTC_INVALID_GEOMETRY_ID;
    result.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    RayContext context;
    rtcInitRayQueryContext(&context);
    context.skipPrimId = skipPrimId;
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    args.context = &context;
    rtcIntersect1(scene.scene, &result, &args);
    return result;
}

bool hasHit(const RTCRayHit& hit)
{
    return hit.hit.geomID != RTC_INVALID_GEOMETRY_ID && hit.hit.primID != RTC_INVALID_GEOMETRY_ID;
}

Float3 getHitNormal(const RTCRayHit& hit)
{
    const Float3 normal{-hit.hit.Ng_x, -hit.hit.Ng_y, -hit.hit.Ng_z};
    const float squareSum = dot(normal, normal);
    if (squareSum == 1.0f) { return normal; }
    if (squareSum < kSmallNumber) { return {}; }
    return normal * (1.0f / std::sqrt(squareSum));
}

uint32_t reverseBits(uint32_t bits)
{
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x00ff00ffu) << 8) | ((bits & 0xff00ff00u) >> 8);
    bits = ((bits & 0x0f0f0f0fu) << 4) | ((bits & 0xf0f0f0f0u) >> 4);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xccccccccu) >> 2);
    return ((bits & 0x55555555u) << 1) | ((bits & 0xaaaaaaaau) >> 1);
}

float randomFraction(uint32_t& seed)
{
    seed = seed * 196314165u + 907633515u;
    return std::bit_cast<float>(0x3f800000u | (seed >> 9)) - 1.0f;
}

Float3 uniformSampleHemisphere(double x, double y)
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

std::vector<Float3> generateStratifiedUniformHemisphereSamples(int32_t numSamples)
{
    const int32_t numSamplesDim = static_cast<int32_t>(std::sqrt(static_cast<float>(numSamples)));
    std::vector<Float3> samples;
    samples.reserve(static_cast<size_t>(numSamplesDim) * numSamplesDim);
    uint32_t seed = 0;
    for (int32_t x = 0; x < numSamplesDim; ++x) {
        for (int32_t y = 0; y < numSamplesDim; ++y) {
            const float u1 = randomFraction(seed);
            const float u2 = randomFraction(seed);
            const float fraction1 = (x + u1) / static_cast<float>(numSamplesDim);
            const float fraction2 = (y + u2) / static_cast<float>(numSamplesDim);
            const auto sample = uniformSampleHemisphere(fraction1, fraction2);
            samples.push_back(sample);
        }
    }
    return samples;
}

Rotation getTangentBasisFrisvad(const Float3& tangentZ)
{
    Float3 tangentX;
    Float3 tangentY;
    if (tangentZ[2] < -0.9999999f) {
        tangentX = {0.0f, -1.0f, 0.0f};
        tangentY = {-1.0f, 0.0f, 0.0f};
    } else {
        const float a = 1.0f / (1.0f + tangentZ[2]);
        const float b = -tangentZ[0] * tangentZ[1] * a;
        tangentX = {1.0f - tangentZ[0] * tangentZ[0] * a, b, -tangentZ[0]};
        tangentY = {b, 1.0f - tangentZ[1] * tangentZ[1] * a, -tangentZ[1]};
    }
    return {tangentX, tangentY, tangentZ};
}

// UE's unstable order for equal candidates affects the surfel visibility rays and card order.
template <typename T, typename Compare>
void heapSort(T* data, int32_t count, const Compare& predicate)
{
    const auto reverse = [&](const T& a, const T& b) { return predicate(b, a); };
    const auto siftDown = [&](int32_t index, int32_t size) {
        while (index * 2 + 1 < size) {
            const int32_t left = index * 2 + 1;
            const int32_t right = left + 1;
            const int32_t child = right < size && !reverse(data[left], data[right]) ? right : left;
            if (!reverse(data[child], data[index])) { break; }
            std::swap(data[index], data[child]);
            index = child;
        }
    };
    for (int32_t index = (count - 2) / 2; index >= 0; --index) {
        siftDown(index, count);
    }
    for (int32_t index = count - 1; index > 0; --index) {
        std::swap(data[0], data[index]);
        siftDown(0, index);
    }
}

template <typename T, typename Compare>
void sortLikeUnreal(std::vector<T>& values, Compare predicate)
{
    if (values.size() < 2) { return; }
    const auto introSort = [&](auto&& self, int32_t min, int32_t max, uint32_t maxDepth) -> void {
        const int32_t count = max - min + 1;
        if (maxDepth == 0) {
            heapSort(values.data() + min, count, predicate);
            return;
        }
        if (count <= 8) {
            while (max > min) {
                int32_t maxItem = min;
                for (int32_t item = min + 1; item <= max; ++item) {
                    if (predicate(values[maxItem], values[item])) { maxItem = item; }
                }
                std::swap(values[maxItem], values[max--]);
            }
            return;
        }
        std::swap(values[min + count / 2], values[min]);
        int32_t innerMin = min;
        int32_t innerMax = max + 1;
        for (;;) {
            while (++innerMin <= max && !predicate(values[min], values[innerMin])) {}
            while (--innerMax > min && !predicate(values[innerMax], values[min])) {}
            if (innerMin > innerMax) { break; }
            std::swap(values[innerMin], values[innerMax]);
        }
        std::swap(values[min], values[innerMax]);
        --maxDepth;
        if (innerMax - 1 - min >= max - innerMin) {
            if (max > innerMin) { self(self, innerMin, max, maxDepth); }
            if (min + 1 < innerMax) { self(self, min, innerMax - 1, maxDepth); }
        } else {
            if (min + 1 < innerMax) { self(self, min, innerMax - 1, maxDepth); }
            if (max > innerMin) { self(self, innerMin, max, maxDepth); }
        }
    };
    introSort(introSort, 0, static_cast<int32_t>(values.size()) - 1,
        static_cast<uint32_t>(std::log(static_cast<float>(values.size())) * 2.0f));
}

using SurfelIndex = uint16_t;

struct Surfel {
    Int3 coord;
    int32_t minRayZ;
    float coverage;
    float weightedCoverage;
};

struct SurfelScene {
    std::array<std::vector<Surfel>, kNumAxisAlignedDirections> directions;
    int32_t numSurfels = 0;
};

struct AxisAlignedDirectionBasis {
    Rotation localToWorldRotation;
    Float3 localToWorldOffset;
    Int3 volumeSize;
};

struct ClusteringParams {
    float voxelSize = 0.0f;
    float minClusterCoverage = 0.0f;
    float minOuterClusterCoverage = 0.0f;
    float minDensityPerCluster = 0.0f;
    uint32_t maxLumenMeshCards = 0;
    std::array<AxisAlignedDirectionBasis, kNumAxisAlignedDirections> clusterBasis;
};

struct SurfelSample {
    Float3 position;
    Float3 normal;
    int32_t minRayZ;
    int32_t cellZ;
};

struct SurfelVisibility {
    float visibility;
    bool valid;
};

SurfelVisibility computeSurfelVisibility(const EmbreeScene& scene, std::span<const SurfelSample> surfelSamples,
    std::span<const Float3> rayDirectionsOverHemisphere)
{
    size_t surfelSampleIndex = 0;
    uint32_t numHits = 0;
    uint32_t numBackFaceHits = 0;
    const float surfaceRayBias = 0.1f;
    float visibilitySum = 0.0f;

    for (const auto& direction : rayDirectionsOverHemisphere) {
        const auto& surfelSample = surfelSamples[surfelSampleIndex];
        const auto surfaceBasis = getTangentBasisFrisvad(surfelSample.normal);
        const auto rayOrigin = surfelSample.position;
        const auto rayDirection = surfaceBasis.transform(direction);
        const auto hit = traceRay(scene, rayOrigin, rayDirection, surfaceRayBias);
        if (hasHit(hit)) {
            ++numHits;
            if (dot(castVector<double>(rayDirection), castVector<double>(getHitNormal(hit))) > 0.0f) {
                ++numBackFaceHits;
            } else {
                visibilitySum += 0.0f;
            }
        } else {
            visibilitySum += 1.0f;
        }
        surfelSampleIndex = (surfelSampleIndex + 1) % surfelSamples.size();
    }

    const auto numRayDirections = static_cast<float>(rayDirectionsOverHemisphere.size());
    const bool insideGeometry = numHits > 0.8f * numRayDirections && numBackFaceHits > 0.2f * numRayDirections;
    return {visibilitySum / numRayDirections, !insideGeometry};
}

void generateSurfelsForDirection(const EmbreeScene& scene, const AxisAlignedDirectionBasis& clusterBasis,
    std::span<const Float3> rayDirectionsOverHemisphere, const ClusteringParams& clusteringParams,
    std::vector<Surfel>& surfels)
{
    const float normalWeightThreshold = 0.25f;
    const Float3 rayDirection = clusterBasis.localToWorldRotation.z;
    constexpr uint32_t numSurfelSamples = 32;
    constexpr int32_t minSurfelSamples = 1;
    std::vector<SurfelSample> surfelSamples;
    std::vector<int32_t> numSurfelSamplesPerCell(clusterBasis.volumeSize[2]);
    std::vector<int32_t> surfelSamplesOffsetPerCell(clusterBasis.volumeSize[2]);

    for (int32_t coordY = 0; coordY < clusterBasis.volumeSize[1]; ++coordY) {
        for (int32_t coordX = 0; coordX < clusterBasis.volumeSize[0]; ++coordX) {
            surfelSamples.clear();
            std::fill(numSurfelSamplesPerCell.begin(), numSurfelSamplesPerCell.end(), 0);
            std::fill(surfelSamplesOffsetPerCell.begin(), surfelSamplesOffsetPerCell.end(), 0);

            for (uint32_t sampleIndex = 0; sampleIndex < numSurfelSamples; ++sampleIndex) {
                const float jitterX = (sampleIndex + 0.5f) / numSurfelSamples;
                const float jitterY = static_cast<float>(static_cast<double>(reverseBits(sampleIndex)) /
                    static_cast<double>(0x100000000LL));
                Float3 rayOrigin = clusterBasis.localToWorldRotation.transform({coordX + jitterX, coordY + jitterY, 0.0f})
                    * clusteringParams.voxelSize + clusterBasis.localToWorldOffset;
                const float nearPlaneOffset = 2.0f * clusteringParams.voxelSize;
                rayOrigin = rayOrigin - rayDirection * nearPlaneOffset;

                int32_t lastHitCoordZ = -2;
                uint32_t skipPrimId = RTC_INVALID_GEOMETRY_ID;
                float rayTNear = 0.0f;
                while (lastHitCoordZ + 1 < clusterBasis.volumeSize[2]) {
                    const auto hit = traceRay(scene, rayOrigin, rayDirection, rayTNear, skipPrimId);
                    if (!hasHit(hit)) { break; }

                    const int32_t hitCoordZ = static_cast<int32_t>(std::clamp(
                        (hit.ray.tfar - nearPlaneOffset) / clusteringParams.voxelSize, 0.0f,
                        static_cast<float>(clusterBasis.volumeSize[2] - 1)));
                    const auto surfaceNormal = getHitNormal(hit);
                    const float nDotD = static_cast<float>(dot(castVector<double>(-rayDirection), castVector<double>(surfaceNormal)));
                    const bool passProjectionTest = nDotD >= normalWeightThreshold;
                    if (passProjectionTest && hitCoordZ >= 0 && hitCoordZ > lastHitCoordZ + 1 && hitCoordZ < clusterBasis.volumeSize[2]) {
                        auto& surfelSample = surfelSamples.emplace_back();
                        surfelSample.position = rayOrigin + rayDirection * hit.ray.tfar;
                        surfelSample.normal = surfaceNormal;
                        surfelSample.cellZ = hitCoordZ;
                        surfelSample.minRayZ = 0;
                        if (lastHitCoordZ >= 0) {
                            surfelSample.minRayZ = std::max(surfelSample.minRayZ, lastHitCoordZ + 1);
                        }
                    }

                    lastHitCoordZ = hitCoordZ;
                    const float safeTFar = std::max(hit.ray.tfar, hit.ray.tnear);
                    rayTNear = std::nextafter(std::max(nearPlaneOffset + (lastHitCoordZ + 1) * clusteringParams.voxelSize,
                        safeTFar), std::numeric_limits<float>::infinity());
                    skipPrimId = hit.hit.primID;
                }
            }

            sortLikeUnreal(surfelSamples, [](const SurfelSample& a, const SurfelSample& b) {
                if (a.cellZ != b.cellZ) { return a.cellZ < b.cellZ; }
                return a.minRayZ > b.minRayZ;
            });
            for (const auto& surfelSample : surfelSamples) {
                ++numSurfelSamplesPerCell[surfelSample.cellZ];
            }
            for (int32_t coordZ = 1; coordZ < clusterBasis.volumeSize[2]; ++coordZ) {
                surfelSamplesOffsetPerCell[coordZ] = surfelSamplesOffsetPerCell[coordZ - 1] + numSurfelSamplesPerCell[coordZ - 1];
            }

            for (int32_t coordZ = 0; coordZ < clusterBasis.volumeSize[2]; ++coordZ) {
                const int32_t cellNumSurfelSamples = numSurfelSamplesPerCell[coordZ];
                const int32_t cellSurfelSamplesOffset = surfelSamplesOffsetPerCell[coordZ];
                int32_t surfelSampleSpanBegin = 0;
                while (surfelSampleSpanBegin + 1 < cellNumSurfelSamples) {
                    int32_t surfelSampleSpanSize = 0;
                    // Match UE's candidate span indexing before applying the cell offset to visibility.
                    for (int32_t sampleIndex = surfelSampleSpanBegin; sampleIndex < cellNumSurfelSamples; ++sampleIndex) {
                        if (surfelSamples[sampleIndex].minRayZ == surfelSamples[surfelSampleSpanBegin].minRayZ) {
                            ++surfelSampleSpanSize;
                        } else {
                            break;
                        }
                    }
                    if (surfelSampleSpanSize >= minSurfelSamples) {
                        const auto visibility = computeSurfelVisibility(scene,
                            std::span(surfelSamples).subspan(cellSurfelSamplesOffset + surfelSampleSpanBegin, surfelSampleSpanSize),
                            rayDirectionsOverHemisphere);
                        const float coverage = surfelSampleSpanSize / static_cast<float>(numSurfelSamples);
                        if (visibility.valid) {
                            const int32_t medianMinRayZ = surfelSamples[cellSurfelSamplesOffset + surfelSampleSpanBegin].minRayZ;
                            surfels.push_back({{coordX, coordY, coordZ}, medianMinRayZ, coverage, coverage * (visibility.visibility + 1.0f)});
                            assert(coordZ > medianMinRayZ || medianMinRayZ == 0);
                        }
                    }
                    surfelSampleSpanBegin += surfelSampleSpanSize;
                }
            }
        }
    }
}

Float3 axisAlignedDirectionIndexToNormal(uint32_t axisAlignedDirectionIndex)
{
    Float3 normal{};
    normal[axisAlignedDirectionIndex / 2] = axisAlignedDirectionIndex & 1u ? 1.0f : -1.0f;
    return normal;
}

void initClusteringParams(ClusteringParams& clusteringParams, const Box& meshCardsBounds, int32_t maxVoxels,
    uint32_t maxLumenMeshCards)
{
    const float targetVoxelSize = 10.0f;
    const Float3 meshCardsBoundsSize = castVector<float>(meshCardsBounds.extent()) * 2.0f;
    const float maxMeshCardsBounds = std::max({meshCardsBoundsSize[0], meshCardsBoundsSize[1], meshCardsBoundsSize[2]});
    const float maxSizeInVoxels = std::clamp(maxMeshCardsBounds / targetVoxelSize + 0.5f, 1.0f, static_cast<float>(maxVoxels));
    const float voxelSize = std::max(targetVoxelSize, maxMeshCardsBounds / maxSizeInVoxels);

    Int3 sizeInVoxels;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        sizeInVoxels[axis] = static_cast<int32_t>(std::clamp(std::floor(meshCardsBoundsSize[axis] / voxelSize + 0.5f),
            1.0f, static_cast<float>(maxVoxels)));
    }

    const Float3 voxelBoundsCenter = castVector<float>(meshCardsBounds.center());
    const Float3 voxelBoundsExtent = (castVector<float>(sizeInVoxels) * voxelSize) * 0.5f;
    const Float3 voxelBoundsMin = voxelBoundsCenter - voxelBoundsExtent;
    const Float3 voxelBoundsMax = voxelBoundsCenter + voxelBoundsExtent;

    for (uint32_t axisAlignedDirectionIndex = 0; axisAlignedDirectionIndex < kNumAxisAlignedDirections; ++axisAlignedDirectionIndex) {
        auto& clusterBasis = clusteringParams.clusterBasis[axisAlignedDirectionIndex];
        Float3 xAxis;
        Float3 yAxis;
        switch (axisAlignedDirectionIndex / 2) {
        case 0:
            xAxis = {0.0f, 1.0f, 0.0f};
            yAxis = {0.0f, 0.0f, 1.0f};
            break;
        case 1:
            xAxis = {1.0f, 0.0f, 0.0f};
            yAxis = {0.0f, 0.0f, 1.0f};
            break;
        default:
            xAxis = {1.0f, 0.0f, 0.0f};
            yAxis = {0.0f, 1.0f, 0.0f};
            break;
        }
        const Float3 zAxis = axisAlignedDirectionIndexToNormal(axisAlignedDirectionIndex);
        clusterBasis.localToWorldRotation = {xAxis, yAxis, -zAxis};
        clusterBasis.localToWorldOffset = voxelBoundsMin;
        if (axisAlignedDirectionIndex & 1u) {
            clusterBasis.localToWorldOffset[axisAlignedDirectionIndex / 2] = voxelBoundsMax[axisAlignedDirectionIndex / 2];
        }

        switch (axisAlignedDirectionIndex / 2) {
        case 0:
            clusterBasis.volumeSize = {sizeInVoxels[1], sizeInVoxels[2], sizeInVoxels[0]};
            break;
        case 1:
            clusterBasis.volumeSize = {sizeInVoxels[0], sizeInVoxels[2], sizeInVoxels[1]};
            break;
        default:
            clusterBasis.volumeSize = {sizeInVoxels[0], sizeInVoxels[1], sizeInVoxels[2]};
            break;
        }
    }

    const float averageFaceArea = 2.0f * (sizeInVoxels[0] * sizeInVoxels[1] + sizeInVoxels[0] * sizeInVoxels[2] +
        sizeInVoxels[1] * sizeInVoxels[2]) / 6.0f;
    clusteringParams.voxelSize = voxelSize;
    clusteringParams.minDensityPerCluster = 0.2f / 3.0f;
    clusteringParams.minClusterCoverage = 15.0f;
    clusteringParams.minOuterClusterCoverage = std::min(clusteringParams.minClusterCoverage, 0.5f * averageFaceArea);
    clusteringParams.maxLumenMeshCards = maxLumenMeshCards;
}

void initSurfelScene(const EmbreeScene& scene, const Box& meshCardsBounds, uint32_t maxLumenMeshCards,
    SurfelScene& surfelScene, ClusteringParams& clusteringParams)
{
    constexpr int32_t numRayDirectionsOverHemisphere = 32;
    const auto rayDirections = generateStratifiedUniformHemisphereSamples(numRayDirectionsOverHemisphere);
    int32_t targetNumSurfels = 10000;
    float maxVoxels = 64;

    do {
        initClusteringParams(clusteringParams, meshCardsBounds, static_cast<int32_t>(maxVoxels), maxLumenMeshCards);
        for (uint32_t direction = 0; direction < kNumAxisAlignedDirections; ++direction) {
            auto& surfels = surfelScene.directions[direction];
            surfels.clear();
            generateSurfelsForDirection(scene, clusteringParams.clusterBasis[direction], rayDirections, clusteringParams, surfels);
        }
        surfelScene.numSurfels = 0;
        for (const auto& surfels : surfelScene.directions) {
            surfelScene.numSurfels += static_cast<int32_t>(surfels.size());
        }
        maxVoxels = maxVoxels / 2;
    } while (surfelScene.numSurfels > targetNumSurfels && maxVoxels > 1);
}

struct IntBox {
    Int3 min{INT32_MAX, INT32_MAX, INT32_MAX};
    Int3 max{-INT32_MAX, -INT32_MAX, -INT32_MAX};

    void add(const Int3& point)
    {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            min[axis] = std::min(min[axis], point[axis]);
            max[axis] = std::max(max[axis], point[axis]);
        }
    }

    int32_t getFaceArea() const
    {
        assert(max[0] >= min[0] && max[1] >= min[1]);
        return (max[0] + 1 - min[0]) * (max[1] + 1 - min[1]);
    }
};

struct SurfelCluster {
    IntBox bounds;
    std::vector<SurfelIndex> surfelIndices;
    int32_t nearPlane = 0;
    float coverage = 0.0f;
    float weightedCoverage = 0.0f;

    void init(int32_t inNearPlane)
    {
        bounds = {};
        surfelIndices.clear();
        nearPlane = inNearPlane;
        coverage = 0.0f;
        weightedCoverage = 0.0f;
    }

    bool isValid(const ClusteringParams& clusteringParams) const
    {
        return weightedCoverage >= (nearPlane == 0 ? clusteringParams.minOuterClusterCoverage : clusteringParams.minClusterCoverage)
            && getDensity() > clusteringParams.minDensityPerCluster;
    }

    float getDensity() const
    {
        return coverage / static_cast<float>(bounds.getFaceArea());
    }

    void addSurfel(const std::vector<Surfel>& surfels, SurfelIndex surfelIndex)
    {
        const auto& surfel = surfels[surfelIndex];
        if (surfel.coord[2] >= nearPlane && surfel.minRayZ <= nearPlane) {
            surfelIndices.push_back(surfelIndex);
            bounds.add(surfel.coord);
            coverage += surfel.coverage;
            weightedCoverage += surfel.weightedCoverage;
            assert(nearPlane <= bounds.min[2]);
        }
    }
};

using MeshCards = std::array<std::vector<SurfelCluster>, kNumAxisAlignedDirections>;

void buildCluster(int32_t nearPlane, const std::vector<Surfel>& surfels, const std::vector<bool>& surfelAssignedToAnyCluster,
    SurfelCluster& cluster)
{
    cluster.init(nearPlane);
    for (size_t index = 0; index < surfels.size(); ++index) {
        if (!surfelAssignedToAnyCluster[index]) {
            cluster.addSurfel(surfels, static_cast<SurfelIndex>(index));
        }
    }
}

void commitCluster(std::vector<SurfelCluster>& clusters, std::vector<bool>& surfelAssignedToAnyCluster,
    const SurfelCluster& cluster)
{
    for (SurfelIndex index : cluster.surfelIndices) {
        surfelAssignedToAnyCluster[index] = true;
    }
    clusters.push_back(cluster);
}

void limitClusters(const ClusteringParams& clusteringParams, MeshCards& meshCards)
{
    size_t numClusters = 0;
    for (auto& clusters : meshCards) {
        sortLikeUnreal(clusters, [](const SurfelCluster& a, const SurfelCluster& b) {
            return a.weightedCoverage > b.weightedCoverage;
        });
        numClusters += clusters.size();
    }

    while (numClusters > clusteringParams.maxLumenMeshCards) {
        float smallestClusterWeightedCoverage = kMaxFloat;
        uint32_t smallestClusterDirectionIndex = 0;
        for (uint32_t direction = 0; direction < kNumAxisAlignedDirections; ++direction) {
            const auto& clusters = meshCards[direction];
            if (!clusters.empty()) {
                const auto& cluster = clusters.back();
                if (cluster.weightedCoverage < smallestClusterWeightedCoverage) {
                    smallestClusterDirectionIndex = direction;
                    smallestClusterWeightedCoverage = cluster.weightedCoverage;
                }
            }
        }
        meshCards[smallestClusterDirectionIndex].pop_back();
        --numClusters;
    }
}

void buildSurfelClusters(const SurfelScene& surfelScene, const ClusteringParams& clusteringParams, MeshCards& meshCards)
{
    for (uint32_t direction = 0; direction < kNumAxisAlignedDirections; ++direction) {
        const auto& surfels = surfelScene.directions[direction];
        auto& clusters = meshCards[direction];
        const auto& clusterBasis = clusteringParams.clusterBasis[direction];
        assert(surfels.size() <= std::numeric_limits<SurfelIndex>::max());
        std::vector<bool> surfelAssignedToAnyCluster(surfels.size(), false);

        SurfelCluster tempCluster;
        buildCluster(0, surfels, surfelAssignedToAnyCluster, tempCluster);
        if (tempCluster.isValid(clusteringParams)) {
            commitCluster(clusters, surfelAssignedToAnyCluster, tempCluster);
        }

        SurfelCluster bestCluster;
        bool canAddCluster = true;
        while (canAddCluster) {
            bestCluster.init(-1);
            for (int32_t nearPlane = 1; nearPlane < clusterBasis.volumeSize[2]; ++nearPlane) {
                buildCluster(nearPlane, surfels, surfelAssignedToAnyCluster, tempCluster);
                if (tempCluster.isValid(clusteringParams) && tempCluster.weightedCoverage > bestCluster.weightedCoverage) {
                    bestCluster = tempCluster;
                }
            }
            canAddCluster = bestCluster.isValid(clusteringParams);
            if (canAddCluster) {
                commitCluster(clusters, surfelAssignedToAnyCluster, bestCluster);
            }
        }
    }
    limitClusters(clusteringParams, meshCards);
}

void serializeLod(const ClusteringParams& clusteringParams, const MeshCards& meshCards, const Box& meshCardsBounds,
    std::vector<MeshCardBuildRecord>& cards)
{
    for (uint32_t direction = 0; direction < kNumAxisAlignedDirections; ++direction) {
        const auto& clusterBasis = clusteringParams.clusterBasis[direction];
        const auto& rotation = clusterBasis.localToWorldRotation;
        const auto localMeshCardsBounds = transformBoxToLocal(meshCardsBounds, rotation, clusterBasis.localToWorldOffset);
        const auto localMin = castVector<float>(localMeshCardsBounds.min);
        const auto localMax = castVector<float>(localMeshCardsBounds.max);
        for (const auto& cluster : meshCards[direction]) {
            Float3 clusterBoundsMin = (castVector<float>(cluster.bounds.min) - Float3{0.0f, 0.0f, 0.5f}) * clusteringParams.voxelSize;
            Float3 clusterBoundsMax = (castVector<float>(cluster.bounds.max) + Float3{1.0f, 1.0f, 1.5f}) * clusteringParams.voxelSize;

            const float marginZ = 10.0f;
            clusterBoundsMin[0] = std::max(clusterBoundsMin[0], localMin[0]);
            clusterBoundsMin[1] = std::max(clusterBoundsMin[1], localMin[1]);
            clusterBoundsMin[2] = std::max(clusterBoundsMin[2], localMin[2] - marginZ);
            clusterBoundsMax[0] = std::min(clusterBoundsMax[0], localMax[0]);
            clusterBoundsMax[1] = std::min(clusterBoundsMax[1], localMax[1]);
            clusterBoundsMax[2] = std::min(clusterBoundsMax[2], localMax[2] + marginZ);

            const Float3 clusterBoundsOrigin = (clusterBoundsMax + clusterBoundsMin) * 0.5f;
            const Float3 clusterBoundsExtent = (clusterBoundsMax - clusterBoundsMin) * 0.5f;
            const Float3 meshClusterBoundsOrigin = rotation.transform(clusterBoundsOrigin) + clusterBasis.localToWorldOffset;
            cards.push_back({meshClusterBoundsOrigin / kCentimetersPerMeter, rotation.x, rotation.y, -rotation.z,
                clusterBoundsExtent / kCentimetersPerMeter, direction});
        }
    }
}

void buildMeshCards(const Box& meshBounds, const EmbreeScene& scene, uint32_t maxLumenMeshCards,
    std::vector<MeshCardBuildRecord>& cards)
{
    const Double3 meshCardsBoundsCenter = meshBounds.center();
    Double3 meshCardsBoundsExtent = meshBounds.extent() + Double3{1.0, 1.0, 1.0};
    for (auto& extent : meshCardsBoundsExtent) { extent = std::max(extent, 1.0); }
    const Box meshCardsBounds{meshCardsBoundsCenter - meshCardsBoundsExtent, meshCardsBoundsCenter + meshCardsBoundsExtent};

    SurfelScene surfelScene;
    ClusteringParams clusteringParams;
    initSurfelScene(scene, meshCardsBounds, maxLumenMeshCards, surfelScene, clusteringParams);
    MeshCards meshCards;
    buildSurfelClusters(surfelScene, clusteringParams, meshCards);
    serializeLod(clusteringParams, meshCards, meshCardsBounds, cards);
}

} // namespace

bool bakeMeshCards(std::span<const std::array<float, 3>> positions, std::span<const uint32_t> indices,
    const MeshCardsBakeOptions& options, std::vector<MeshCardBuildRecord>& cards, std::string& error)
{
    if (options.maxCards > kMaxCardsPerMesh) {
        error = "Mesh Cards baking supports at most 32 cards per mesh";
        return false;
    }
    if (options.maxCards == 0) {
        cards.clear();
        return true;
    }
    if (positions.empty() || positions.size() > UINT32_MAX ||
        indices.empty() || indices.size() % 3 != 0 || indices.size() / 3 > UINT32_MAX) {
        error = "Mesh Cards baking requires indexed triangles";
        return false;
    }

    EmbreeScene scene;
    scene.vertices.reserve(positions.size() + 1);
    Float3 boundsMin{kMaxFloat, kMaxFloat, kMaxFloat};
    Float3 boundsMax{-kMaxFloat, -kMaxFloat, -kMaxFloat};
    for (const auto& position : positions) {
        const Float3 centimeters = position * kCentimetersPerMeter;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(centimeters[axis])) {
                error = "Mesh Cards positions must be finite in centimeters";
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
        // UE supplies clockwise triangles and negates Embree's geometric normal.
        const uint32_t i1 = indices[index + 2];
        const uint32_t i2 = indices[index + 1];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
            error = "Mesh Cards triangle index is out of range";
            return false;
        }
        const auto& v0 = scene.vertices[i0];
        const auto& v1 = scene.vertices[i1];
        const auto& v2 = scene.vertices[i2];
        const auto triangleNormal = cross(v1 - v2, v0 - v2);
        if (dot(triangleNormal, triangleNormal) < kSmallNumber) { continue; }
        scene.indices.insert(scene.indices.end(), {i0, i1, i2});
    }
    if (scene.indices.empty()) {
        error = "No triangles remain after removing degenerate triangles";
        return false;
    }

    if (options.bounds) {
        boundsMin = (*options.bounds)[0] * kCentimetersPerMeter;
        boundsMax = (*options.bounds)[1] * kCentimetersPerMeter;
    } else {
        const Float3 extent = (boundsMax - boundsMin) * 0.5f;
        const Float3 origin = boundsMin + extent;
        boundsMin = origin - extent;
        boundsMax = origin + extent;
    }
    for (uint32_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(boundsMin[axis] + boundsMax[axis]) || !std::isfinite(boundsMax[axis] - boundsMin[axis]) ||
            boundsMin[axis] > boundsMax[axis]) {
            error = "Mesh Cards bounds must be finite and ordered in centimeters";
            return false;
        }
    }

    if (!initializeEmbree(scene, error)) { return false; }
    std::vector<MeshCardBuildRecord> result;
    result.reserve(options.maxCards);
    buildMeshCards({castVector<double>(boundsMin), castVector<double>(boundsMax)}, scene, options.maxCards, result);
    cards = std::move(result);
    return true;
}

} // namespace metallic::scene
