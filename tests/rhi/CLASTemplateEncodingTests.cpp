#include "RHITest.h"
#include "TestComputeProgram.h"
#include "TestResourceParameters.h"
#include "harness/Fixtures.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"
#include "Runtime/Render/TessellationPatterns.h"
#include "../../Shaders/Interop/StreamCLASTessellation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = bench::Json;
constexpr uint32_t kPatchCount = 2048;
constexpr uint32_t kImageSide = 128;
constexpr float kTolerance = .0002f;

template<typename T>
T checked(Result<T> result)
{
    if (!result) { throw std::runtime_error(resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

void require(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<typename T>
std::span<const uint8_t> bytes(const std::vector<T>& value)
{
    return {reinterpret_cast<const uint8_t*>(value.data()), value.size() * sizeof(T)};
}

std::vector<uint8_t> loadBytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), "Cannot read " + path.string());
    const auto size = input.tellg();
    require(size > 0 && size <= 128 * 1024 * 1024, "Invalid diagnostic capture size");
    std::vector<uint8_t> result(static_cast<size_t>(size));
    input.seekg(0); input.read(reinterpret_cast<char*>(result.data()), size);
    require(bool(input), "Incomplete diagnostic capture read");
    return result;
}

template<typename T>
std::vector<T> loadCapture(const Json& manifest, const std::filesystem::path& root,
    const char* key, size_t count)
{
    const std::string filename = manifest.at(key).get<std::string>();
    const auto raw = loadBytes(root / filename);
    require(raw.size() == count * sizeof(T), std::string(key) + " byte count mismatch");
    require(manifest.at("sha256").at(filename).get<std::string>() == debug::debugSha256(raw),
        std::string(key) + " SHA256 mismatch");
    std::vector<T> result(count);
    std::memcpy(result.data(), raw.data(), raw.size());
    return result;
}

std::unique_ptr<Buffer> buffer(Device& device, uint64_t size, MemoryLocation location = MemoryLocation::Device)
{
    return checked(device.createBuffer({.size = size,
        .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
        .memoryLocation = location}));
}

BufferSlice slice(Buffer& value, BufferRange range = {}) { return checked(value.slice(range)); }
BufferSlice alignedSlice(Buffer& value, uint64_t alignment, uint64_t size)
{
    return slice(value, {(alignment - value.deviceAddress() % alignment) % alignment, size});
}

void upload(Buffer& value, const void* data, size_t size)
{
    void* mapped = value.map(); require(mapped != nullptr, "Upload map failed");
    std::memcpy(mapped, data, size); value.flush(); value.unmap();
}

template<typename T>
std::vector<T> read(Buffer& value)
{
    value.invalidate(); const void* mapped = value.map(); require(mapped != nullptr, "Readback map failed");
    std::vector<T> result(value.desc().size / sizeof(T));
    std::memcpy(result.data(), mapped, result.size() * sizeof(T)); value.unmap(); return result;
}

void hostReadBarrier(CommandBuffer& commands)
{
    const MemoryBarrierDesc barrier{.before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
    checked(commands.synchronize({.memory = {&barrier, 1}}));
}

struct Vec3 { double x, y, z; };
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
Vec3 position(const StreamCLASTessellationVertex& v) { return {v.x, v.y, v.z}; }

struct Ray { float origin[3], minimum; float direction[3], maximum; };
struct Observation {
    uint32_t hit, cluster, geometry, primitive;
    float distance, u, v, front;
    float position[3], sourceU;
    float normal[3], sourceV;
};
struct Guide { float x, y, z; };
struct Triangle { Vec3 a, edge0, edge1; };
static_assert(sizeof(Ray) == 32 && sizeof(Observation) == 64 && sizeof(Guide) == 12);

Json floatBits(const std::vector<float>& values)
{
    std::array<uint64_t, 24> trailing{};
    std::array<uint64_t, 256> exponents{};
    uint32_t minimum = 23, bitOr = 0;
    uint64_t zeroCount = 0;
    for (float value : values) {
        require(std::isfinite(value), "Non-finite geometry position in diagnostic capture");
        const uint32_t word = std::bit_cast<uint32_t>(value);
        const uint32_t mantissa = word & 0x7fffffu;
        const uint32_t zeros = mantissa == 0 ? 23u : uint32_t(std::countr_zero(mantissa));
        ++trailing[zeros]; ++exponents[(word >> 23u) & 255u];
        minimum = std::min(minimum, zeros); bitOr |= mantissa;
        if (value == 0.f) { ++zeroCount; }
    }
    Json samples = Json::array();
    std::set<uint32_t> seen;
    for (float value : values) {
        const uint32_t word = std::bit_cast<uint32_t>(value);
        if (seen.insert(word).second && samples.size() < 24) { samples.push_back({{"value", value}, {"uint32", word}}); }
    }
    return {{"floatCount", values.size()}, {"sha256", debug::debugSha256(bytes(values))},
        {"minimumTrailingMantissaZeros", minimum}, {"mantissaBitwiseOr", bitOr},
        {"trailingMantissaZeroHistogram", trailing}, {"exponentHistogram", exponents},
        {"zeroCount", zeroCount}, {"firstDistinctWords", samples}};
}

struct Capture {
    Json manifest;
    std::vector<StreamCLASTessellationLeaf> leaves;
    std::vector<StreamCLASTessellationVertex> vertices;
    std::vector<uint32_t> indices, sizes;
    std::vector<uint8_t> indices8;
    std::array<float, 6> globalBounds{};
    std::vector<std::array<float, 6>> tightBounds;
    std::vector<TessellationPattern> patterns;
    std::vector<Triangle> triangles;
    std::vector<Ray> rays;
    uint32_t imageOffset = 0;
    float rayExtent = 0;
    Json bitAnalysis;
};

Capture loadGeometry(const std::filesystem::path& path)
{
    Capture c;
    c.manifest = bench::readJson(path);
    require(c.manifest.at("schemaVersion") == 1 && c.manifest.at("patchCount") == kPatchCount &&
        c.manifest.at("verticesPerPatch") == METALLIC_TESS_VERTICES_PER_PATCH &&
        c.manifest.at("trianglesPerPatch") == METALLIC_TESS_TRIANGLES_PER_PATCH &&
        c.manifest.at("leafStride") == sizeof(StreamCLASTessellationLeaf) &&
        c.manifest.at("vertexStride") == sizeof(StreamCLASTessellationVertex) &&
        c.manifest.at("unusedVertexSlotsZeroed") == true, "Unsupported production geometry capture schema");
    const auto& identity = c.manifest.at("identity");
    require(identity.is_object() && identity.at("mode") == "templates" && identity.at("templateShape") == "right" &&
        identity.at("instantiationFlags") == "fastTrace" && identity.at("templateBoundsMode") == "geometry" &&
        identity.at("boundedTemplatesReady") == true, "Capture is not the fixed C3 bounded-template workload");
    const auto root = path.parent_path();
    c.leaves = loadCapture<StreamCLASTessellationLeaf>(c.manifest, root, "leavesFile", kPatchCount);
    c.vertices = loadCapture<StreamCLASTessellationVertex>(c.manifest, root, "verticesFile", kPatchCount * METALLIC_TESS_VERTICES_PER_PATCH);
    c.sizes = loadCapture<uint32_t>(c.manifest, root, "sizesFile", kPatchCount);
    const auto counts = loadCapture<uint32_t>(c.manifest, root, "countsFile", 6);
    require(counts[0] == kPatchCount && counts[1] > 0 && counts[2] == 0 && counts[3] == 0 &&
        counts[4] == 0 && counts[5] == 0, "Capture contains fallback/rejected/invalid instances");
    c.globalBounds = c.manifest.at("templateBounds").get<std::array<float, 6>>();
    for (uint32_t axis = 0; axis < 3; ++axis) {
        require(std::isfinite(c.globalBounds[axis]) && std::isfinite(c.globalBounds[axis + 3]) &&
            c.globalBounds[axis] < c.globalBounds[axis + 3], "Invalid global bounds");
    }
    c.indices.resize(kPatchCount * METALLIC_TESS_TRIANGLES_PER_PATCH * 3);
    c.indices8.resize(c.indices.size()); c.tightBounds.resize(kPatchCount); c.patterns.reserve(kPatchCount);
    std::array<std::vector<float>, 3> axisValues;
    for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
        const auto& leaf = c.leaves[patch];
        const std::array<uint32_t, 3> rates{leaf.rates & 255u, (leaf.rates >> 8u) & 255u, (leaf.rates >> 16u) & 255u};
        for (uint32_t rate : rates) { require(rate >= 1 && rate <= 8, "Invalid captured pattern rate"); }
        auto pattern = makeTessellationPattern(rates);
        require(leaf.vertexCount == pattern.vertices.size() && leaf.triangleCount == pattern.triangles.size() &&
            leaf.vertexCount <= METALLIC_TESS_VERTICES_PER_PATCH && leaf.triangleCount <= METALLIC_TESS_TRIANGLES_PER_PATCH,
            "Captured topology differs from unchanged makeTessellationPattern");
        auto& bounds = c.tightBounds[patch];
        bounds = {INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (uint32_t vertex = 0; vertex < leaf.vertexCount; ++vertex) {
            const auto& v = c.vertices[patch * METALLIC_TESS_VERTICES_PER_PATCH + vertex];
            const std::array<float, 3> xyz{v.x, v.y, v.z};
            for (uint32_t axis = 0; axis < 3; ++axis) {
                require(std::isfinite(xyz[axis]) && xyz[axis] >= c.globalBounds[axis] && xyz[axis] <= c.globalBounds[axis + 3],
                    "Actual generated vertex is outside captured global template bound");
                bounds[axis] = std::min(bounds[axis], xyz[axis]); bounds[axis + 3] = std::max(bounds[axis + 3], xyz[axis]);
                axisValues[axis].push_back(xyz[axis]);
            }
        }
        for (uint32_t triangle = 0; triangle < leaf.triangleCount; ++triangle) {
            const auto indices = pattern.triangles[triangle];
            const size_t offset = (patch * METALLIC_TESS_TRIANGLES_PER_PATCH + triangle) * 3;
            for (uint32_t corner = 0; corner < 3; ++corner) {
                require(indices[corner] < leaf.vertexCount, "Invalid diagnostic topology");
                c.indices[offset + corner] = indices[corner]; c.indices8[offset + corner] = uint8_t(indices[corner]);
            }
            const auto base = patch * METALLIC_TESS_VERTICES_PER_PATCH;
            const Vec3 a = position(c.vertices[base + indices[0]]), b = position(c.vertices[base + indices[1]]),
                d = position(c.vertices[base + indices[2]]);
            c.triangles.push_back({a, b - a, d - a});
        }
        c.patterns.push_back(std::move(pattern));
    }
    if (c.manifest.contains("indicesFile")) {
        const auto actual = loadCapture<uint32_t>(c.manifest, root, "indicesFile", c.indices.size());
        for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
            const size_t offset = patch * METALLIC_TESS_TRIANGLES_PER_PATCH * 3;
            require(std::equal(c.indices.begin() + offset, c.indices.begin() + offset + c.leaves[patch].triangleCount * 3,
                actual.begin() + offset), "GPU index order differs from diagnostic reconstruction");
        }
    }
    c.bitAnalysis = {{"x", floatBits(axisValues[0])}, {"y", floatBits(axisValues[1])}, {"z", floatBits(axisValues[2])}};
    Vec3 low{c.globalBounds[0], c.globalBounds[1], c.globalBounds[2]}, high{c.globalBounds[3], c.globalBounds[4], c.globalBounds[5]};
    c.rayExtent = float(std::sqrt(dot(high - low, high - low))) * 2.f;
    for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
        const auto indices = c.patterns[patch].triangles.front();
        const uint32_t base = patch * METALLIC_TESS_VERTICES_PER_PATCH;
        const Vec3 a = position(c.vertices[base + indices[0]]), b = position(c.vertices[base + indices[1]]),
            d = position(c.vertices[base + indices[2]]);
        const Vec3 normal = cross(b - a, d - a);
        require(dot(normal, normal) > 1e-24, "Captured first triangle is degenerate");
        const Vec3 n = normal * (1.0 / std::sqrt(dot(normal, normal)));
        // Irrational interior weights avoid the regular tessellation edges.
        const Vec3 center = a * .217 + b * .319 + d * .464;
        for (double sign : {-1.0, 1.0}) {
            const Vec3 origin = center + n * (sign * c.rayExtent * .025);
            c.rays.push_back({{float(origin.x), float(origin.y), float(origin.z)}, 1e-6f,
                {float(-n.x * sign), float(-n.y * sign), float(-n.z * sign)}, c.rayExtent});
        }
    }
    c.imageOffset = uint32_t(c.rays.size());
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t u = (axis + 1) % 3, v = (axis + 2) % 3;
        for (uint32_t y = 0; y < kImageSide; ++y) {
            for (uint32_t x = 0; x < kImageSide; ++x) {
                Ray ray{}; ray.minimum = 1e-6f; ray.maximum = c.rayExtent * 2.f;
                ray.origin[axis] = c.globalBounds[axis + 3] + c.rayExtent * .25f;
                ray.origin[u] = c.globalBounds[u] + (float(x) + .371f) / kImageSide * (c.globalBounds[u + 3] - c.globalBounds[u]);
                ray.origin[v] = c.globalBounds[v] + (float(y) + .613f) / kImageSide * (c.globalBounds[v + 3] - c.globalBounds[v]);
                ray.direction[axis] = -1.f; c.rays.push_back(ray);
            }
        }
    }
    return c;
}

std::vector<Guide> guides(const Capture& c, bool dyadic)
{
    std::vector<Guide> result(kPatchCount * METALLIC_TESS_VERTICES_PER_PATCH);
    for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
        for (uint32_t vertex = 0; vertex < c.leaves[patch].vertexCount; ++vertex) {
            const auto& bary = c.patterns[patch].vertices[vertex];
            const auto coordinate = [&](uint32_t q) {
                return dyadic ? float(std::lround(double(q) * 32768.0 / 65535.0)) / 32768.f : float(q) / 65535.f;
            };
            result[patch * METALLIC_TESS_VERTICES_PER_PATCH + vertex] = {coordinate(bary[1]), coordinate(bary[2]), 0};
        }
    }
    return result;
}

struct BuiltClusters {
    std::unique_ptr<Buffer> storage;
    BufferSlice storageSlice;
    std::vector<uint64_t> addresses;
    std::vector<uint32_t> sizes;
};

BuiltClusters buildClusters(RHITestContext& context, ClusterAccelerationStructureTriangleOperation operation,
    std::span<const uint8_t> records, uint64_t stride, std::span<const BufferSlice> inputs, BufferSlice templates = {})
{
    auto& device = context.device;
    const auto properties = checked(device.queryClusterAccelerationStructureProperties());
    const ClusterAccelerationStructureTriangleBuildSizesDesc limits{
        .maxClusterTriangleCount = METALLIC_TESS_TRIANGLES_PER_PATCH,
        .maxClusterVertexCount = METALLIC_TESS_VERTICES_PER_PATCH,
        .maxTotalTriangleCount = kPatchCount * METALLIC_TESS_TRIANGLES_PER_PATCH,
        .maxTotalVertexCount = kPatchCount * METALLIC_TESS_VERTICES_PER_PATCH,
        .maxAccelerationStructureCount = kPatchCount, .operation = operation,
        .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace};
    const auto sizes = checked(device.queryClusterAccelerationStructureTriangleBuildSizes(limits, ClusterAccelerationStructureDestinationMode::Implicit));
    const uint64_t alignment = operation == ClusterAccelerationStructureTriangleOperation::BuildTemplate
        ? properties.templateStorageAlignment : properties.clusterStorageAlignment;
    BuiltClusters result;
    result.storage = buffer(device, sizes.accelerationStructureSize + alignment);
    result.storageSlice = alignedSlice(*result.storage, alignment, sizes.accelerationStructureSize);
    auto infos = buffer(device, records.size(), MemoryLocation::HostUpload); upload(*infos, records.data(), records.size());
    auto addresses = buffer(device, kPatchCount * 8, MemoryLocation::HostReadback);
    auto outputSizes = buffer(device, kPatchCount * 4, MemoryLocation::HostReadback);
    auto scratch = buffer(device, sizes.buildScratchSize + properties.scratchAlignment);
    bench::GPUCommands commands(context.graphicsQueue); checked(commands.initialize(device));
    checked(commands.commands->buildClusterAccelerationStructureTrianglesIndirect({.limits = limits,
        .buildInfoBuffer = slice(*infos), .buildInfoStride = stride, .destinationStorageBuffer = result.storageSlice,
        .destinationAddressBuffer = slice(*addresses), .destinationSizeBuffer = slice(*outputSizes),
        .scratchBuffer = slice(*scratch), .geometryBuffers = inputs, .templateStorageBuffer = templates}));
    hostReadBarrier(*commands.commands); checked(commands.submitAndWait());
    result.addresses = read<uint64_t>(*addresses); result.sizes = read<uint32_t>(*outputSizes);
    for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
        const uint64_t address = result.addresses[patch];
        require(address >= result.storageSlice.deviceAddress() && address % alignment == 0 && result.sizes[patch] > 0 &&
            address <= result.storageSlice.deviceAddress() + result.storageSlice.size() &&
            result.sizes[patch] <= result.storageSlice.deviceAddress() + result.storageSlice.size() - address,
            "Native cluster output outside arena");
    }
    return result;
}

std::vector<Observation> trace(RHITestContext& context, const Capture& capture, const BuiltClusters& clusters,
    Buffer& vertices, Buffer& indices, Buffer& rays, ComputeProgram& program)
{
    auto& device = context.device; auto& queue = context.graphicsQueue;
    const auto properties = checked(device.queryClusterAccelerationStructureProperties());
    const auto blasSizes = checked(device.queryClusterAccelerationStructureBottomLevelBuildSizes({
        .maxClusterCountPerAccelerationStructure = kPatchCount, .maxTotalClusterCount = kPatchCount}));
    auto blas = buffer(device, blasSizes.accelerationStructureSize + properties.bottomLevelStorageAlignment);
    const auto blasStorage = alignedSlice(*blas, properties.bottomLevelStorageAlignment, blasSizes.accelerationStructureSize);
    auto references = buffer(device, kPatchCount * 8, MemoryLocation::HostUpload);
    upload(*references, clusters.addresses.data(), kPatchCount * 8);
    const ClusterAccelerationStructureBottomLevelBuildInfo info{.clusterReferencesCount = kPatchCount,
        .clusterReferencesAddress = references->deviceAddress()};
    auto infos = buffer(device, sizeof(info), MemoryLocation::HostUpload); upload(*infos, &info, sizeof(info));
    auto address = buffer(device, 8, MemoryLocation::HostReadback);
    auto blasScratch = buffer(device, blasSizes.buildScratchSize + properties.scratchAlignment);
    {
        bench::GPUCommands commands(queue); checked(commands.initialize(device));
        checked(commands.commands->buildClusterAccelerationStructureBottomLevels({
            .maxClusterCountPerAccelerationStructure = kPatchCount, .maxTotalClusterCount = kPatchCount,
            .buildInfoBuffer = slice(*infos), .destinationStorageBuffer = blasStorage,
            .destinationAddressBuffer = slice(*address), .scratchBuffer = slice(*blasScratch)}));
        hostReadBarrier(*commands.commands); checked(commands.submitAndWait());
    }
    RayTracingGPUInstance instance;
    instance.customIndexAndMask = 1u << 24;
    instance.shaderBindingTableRecordOffsetAndFlags = uint32_t(RayTracingInstanceFlags::TriangleFacingCullDisable) << 24;
    instance.accelerationStructureReference = read<uint64_t>(*address)[0];
    auto instances = buffer(device, sizeof(instance), MemoryLocation::HostUpload); upload(*instances, &instance, sizeof(instance));
    const auto tlasSizes = checked(device.queryRayTracingAccelerationStructureBuildSizes({.instanceCount = 1}));
    auto tlas = checked(device.createRayTracingAccelerationStructure({.size = tlasSizes.accelerationStructureSize}));
    const auto rtProperties = checked(device.queryRayTracingAccelerationStructureProperties());
    auto scratch = buffer(device, tlasSizes.buildScratchSize + rtProperties.scratchAlignment);
    {
        bench::GPUCommands commands(queue); checked(commands.initialize(device));
        checked(commands.commands->buildRayTracingAccelerationStructure({.destination = tlas.get(),
            .instanceBuffer = slice(*instances), .instanceCount = 1, .scratchBuffer = slice(*scratch)}));
        checked(commands.submitAndWait());
    }
    struct Constants { uint64_t rays, vertices, indices; uint32_t count, verticesPerPatch, trianglesPerPatch, patchCount; };
    const Constants constants{rays.deviceAddress(), vertices.deviceAddress(), indices.deviceAddress(), uint32_t(capture.rays.size()),
        METALLIC_TESS_VERTICES_PER_PATCH, METALLIC_TESS_TRIANGLES_PER_PATCH, kPatchCount};
    auto output = buffer(device, capture.rays.size() * sizeof(Observation), MemoryLocation::HostReadback);
    auto pool = checked(device.createCommandPool(queue)); auto commands = checked(pool->createCommandBuffer());
    QueueSubmissionTracker tracker; checked(tracker.initialize(device, queue)); RenderFrameContext frame;
    struct Drain {
        RenderFrameContext& frame; CommandPool& pool;
        ~Drain() { if (frame.completion().isSubmitted()) { (void)frame.wait(); } (void)pool.reset(); (void)frame.reset(); }
    } drain{frame, *pool};
    checked(frame.begin(0)); checked(commands->begin(frame.submissionContext())); commands->hostWriteBarrier();
    for (Buffer* input : {&vertices, &indices, &rays}) { checked(commands->retainResource(input->retainAllocation())); }
    const ComputeDispatchBinding bindings[]{
        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), .accelerationStructure = tlas.get()},
        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output), .buffer = output.get()}};
    checked(program.dispatch({.commandBuffer = commands.get(), .bindings = bindings,
        .pushData = &constants, .pushDataSize = sizeof(constants), .groupCountX = (constants.count + 63) / 64}));
    hostReadBarrier(*commands); checked(commands->end()); CommandBuffer* submitted[]{commands.get()};
    checked(tracker.submit({.commandBuffers = submitted}, frame)); checked(frame.wait(5'000'000'000ull));
    return read<Observation>(*output);
}

double cpuDistance(const Ray& ray, const std::vector<Triangle>& triangles)
{
    const Vec3 origin{ray.origin[0], ray.origin[1], ray.origin[2]}, direction{ray.direction[0], ray.direction[1], ray.direction[2]};
    double nearest = INFINITY;
    for (const auto& triangle : triangles) {
        const Vec3 p = cross(direction, triangle.edge1);
        const double determinant = dot(triangle.edge0, p);
        if (std::abs(determinant) < 1e-20) { continue; }
        const Vec3 delta = origin - triangle.a;
        const double u = dot(delta, p) / determinant;
        if (u < 0 || u > 1) { continue; }
        const Vec3 q = cross(delta, triangle.edge0);
        const double v = dot(direction, q) / determinant;
        if (v < 0 || u + v > 1) { continue; }
        const double distance = dot(triangle.edge1, q) / determinant;
        if (distance >= ray.minimum && distance <= ray.maximum) { nearest = std::min(nearest, distance); }
    }
    return nearest;
}

Json compare(const Capture& capture, const std::vector<Observation>& actual, const std::vector<Observation>& reference)
{
    uint32_t hits = 0, imageHits = 0, differentIds = 0;
    float maxDistance = 0, maxPosition = 0, maxAttributes = 0;
    for (uint32_t index = 0; index < actual.size(); ++index) {
        const auto& a = actual[index]; const auto& b = reference[index];
        const std::string at = " at ray " + std::to_string(index);
        require(a.hit == b.hit, "Native encoding changed coverage" + at);
        if (!a.hit) { continue; }
        ++hits; if (index >= capture.imageOffset) { ++imageHits; }
        require(a.cluster < kPatchCount && a.geometry == 0 && a.primitive < capture.leaves[a.cluster].triangleCount,
            "Native hit IDs outside capture" + at);
        require(std::isfinite(a.distance) && std::abs(a.distance - b.distance) <= kTolerance,
            "Native encoding changed hit distance" + at);
        maxDistance = std::max(maxDistance, std::abs(a.distance - b.distance));
        for (uint32_t axis = 0; axis < 3; ++axis) {
            require(std::isfinite(a.position[axis]) && std::abs(a.position[axis] - b.position[axis]) <= kTolerance,
                "Native encoding changed hit position" + at);
            maxPosition = std::max(maxPosition, std::abs(a.position[axis] - b.position[axis]));
        }
        require(std::isfinite(a.u) && std::isfinite(a.v) && a.u >= -kTolerance && a.v >= -kTolerance &&
            a.u + a.v <= 1.f + kTolerance, "Native hit barycentrics outside triangle" + at);
        const auto indices = capture.patterns[a.cluster].triangles[a.primitive];
        const uint32_t base = a.cluster * METALLIC_TESS_VERTICES_PER_PATCH;
        const Vec3 reconstructed = position(capture.vertices[base + indices[0]]) * (1.0 - a.u - a.v) +
            position(capture.vertices[base + indices[1]]) * a.u + position(capture.vertices[base + indices[2]]) * a.v;
        require(std::abs(reconstructed.x - a.position[0]) <= kTolerance &&
            std::abs(reconstructed.y - a.position[1]) <= kTolerance &&
            std::abs(reconstructed.z - a.position[2]) <= kTolerance,
            "Native hit does not lie on unchanged actual triangle" + at);
        // Overlapping triangles and shared edges can select a different valid
        // primitive. Coverage and position still compare for these ties.
        if (a.cluster != b.cluster || a.primitive != b.primitive) { ++differentIds; continue; }
        const std::array<float, 8> attributesA{a.u, a.v, a.front, a.sourceU, a.sourceV, a.normal[0], a.normal[1], a.normal[2]};
        const std::array<float, 8> attributesB{b.u, b.v, b.front, b.sourceU, b.sourceV, b.normal[0], b.normal[1], b.normal[2]};
        for (uint32_t component = 0; component < attributesA.size(); ++component) {
            const float difference = std::abs(attributesA[component] - attributesB[component]);
            require(std::isfinite(attributesA[component]) && difference <= kTolerance, "Native encoding changed hit attributes" + at);
            maxAttributes = std::max(maxAttributes, difference);
        }
    }
    require(hits >= kPatchCount && imageHits >= 1000, "Insufficient native diagnostic ray/image coverage");
    return {{"rays", actual.size()}, {"hits", hits}, {"imageHits", imageHits}, {"differentPrimitiveIds", differentIds},
        {"maximumDistanceDifference", maxDistance}, {"maximumPositionDifference", maxPosition},
        {"maximumSamePrimitiveAttributeDifference", maxAttributes}, {"tolerance", kTolerance}};
}

void saveObservations(const std::filesystem::path& directory, const std::string& name,
    const Capture& capture, const std::vector<Observation>& observations)
{
    std::ofstream output(directory / (name + ".bin"), std::ios::binary);
    output.write(reinterpret_cast<const char*>(observations.data()), observations.size() * sizeof(Observation));
    require(bool(output), "Cannot save native ray observations");
    std::vector<uint8_t> rgba(kImageSide * kImageSide * 3 * 4);
    for (uint32_t index = capture.imageOffset; index < observations.size(); ++index) {
        const auto& value = observations[index];
        const uint8_t shade = value.hit ? uint8_t(std::clamp(1.f - value.distance / (capture.rayExtent * 2.f), 0.f, 1.f) * 254.f + 1.f) : 0;
        const uint32_t offset = (index - capture.imageOffset) * 4;
        rgba[offset] = rgba[offset + 1] = rgba[offset + 2] = shade; rgba[offset + 3] = 255;
    }
    std::string log;
    require(saveRgba8Png(directory / (name + ".png"), rgba.data(), kImageSide, kImageSide * 3, log), "Image write failed: " + log);
}

class CLASTemplateEncodingTest final : public RHITest {
public:
    CLASTemplateEncodingTest() { type = RHITestType::Rendering; name = "clas_template_encoding_diagnostic"; }
    std::optional<bench::Metadata> metadata() const override
    {
        auto result = bench::gpuMetadata({"clas.template.encoding.fixed-production-geometry"},
            bench::Layer::RHI, "ray-query", "extensions", {"encoding-matrix.json"});
        result.timeout = std::chrono::seconds(180); return result;
    }
    RHITestResult run(RHITestContext& context) override
    {
        const char* filename = std::getenv("METALLIC_CLAS_ENCODING_CAPTURE");
        if (!filename || !*filename) { return RHITestResult::skip("Set METALLIC_CLAS_ENCODING_CAPTURE to a production geometry receipt"); }
        auto created = bench::createTestDevice(context, {.applicationName = "CLAS fixed-geometry encoding diagnostic",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .enableRayQuery = true, .enableClusterAccelerationStructure = true});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("CLAS/ray query unavailable"); }
        if (!created) { return RHITestResult::fail("CLAS diagnostic device creation failed"); }
        auto& device = **created;
        if (!device.capabilities().clusterAccelerationStructure || !device.capabilities().rayQuery) {
            return RHITestResult::skip("CLAS/ray query unavailable");
        }
        RHITestContext active{device, *device.getQueue(QueueType::Graphics), context.outputDirectory,
            context.enableValidation, context.validationMessageCount, context.nsightCapture, context.evidence, context.deviceDesc};
        try { return runDevice(active, filename); }
        catch (const std::exception& error) { (void)device.waitIdle(); return RHITestResult::fail(error.what()); }
    }
private:
    RHITestResult runDevice(RHITestContext& context, const std::filesystem::path& filename)
    {
        auto& device = context.device;
        const Capture capture = loadGeometry(filename);
        const auto directory = context.outputDirectory / name; std::filesystem::create_directories(directory);
        auto vertices = buffer(device, bytes(capture.vertices).size(), MemoryLocation::HostUpload);
        upload(*vertices, capture.vertices.data(), bytes(capture.vertices).size());
        auto indices32 = buffer(device, bytes(capture.indices).size(), MemoryLocation::HostUpload);
        upload(*indices32, capture.indices.data(), bytes(capture.indices).size());
        auto indices8 = buffer(device, capture.indices8.size(), MemoryLocation::HostUpload);
        upload(*indices8, capture.indices8.data(), capture.indices8.size());
        auto rays = buffer(device, bytes(capture.rays).size(), MemoryLocation::HostUpload);
        upload(*rays, capture.rays.data(), bytes(capture.rays).size());
        const char* capabilities[]{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"};
        std::string log;
        const auto shader = checked(compileSlangShaderToSpirv({.moduleName = "CLASTemplateEncodingProbe", .entryPointName = "clasTemplateEncodingProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .capabilities = capabilities,
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log));
        const ComputeResourceBindingDesc layout[]{
            {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), ComputeResourceBindingKind::AccelerationStructure},
            {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output)}};
        ComputeProgram program;
        checked(program.initialize(device, {.spirv = shader.spirv, .pushConstantSize = 40,
            .bindings = layout, .resourceParameterSize = sizeof(UnifiedTopLevelProbeResources)}, log));
        const auto properties = checked(device.queryClusterAccelerationStructureProperties());
        Json report{{"schemaVersion", 1}, {"capture", filename.string()}, {"captureManifestSha256", debug::debugSha256(loadBytes(filename))},
            {"captureIdentity", capture.manifest.at("identity")}, {"patchCount", kPatchCount},
            {"actualVertexBytesSha256", debug::debugSha256(bytes(capture.vertices))},
            {"actualPositionBits", capture.bitAnalysis}, {"index32Sha256", debug::debugSha256(bytes(capture.indices))},
            {"index8Sha256", debug::debugSha256(capture.indices8)},
            {"indicesProvenance", capture.manifest.contains("indicesFile") ? "GPU captured and checked" : "derived from unchanged makeTessellationPattern and captured rates/counts"},
            {"scope", "untimed native encoding diagnostic; merged object-space geometry; guide-only changes; no precision or tessellation reduction"},
            {"tightBoundsScope", "exact current generated FP32 extrema only; no claim of future-frame conservativeness"},
            {"nativeParameters", {{"vertexFormat", "R32G32B32_SFLOAT"}, {"actualVertexStride", 32}, {"guideVertexStride", 12},
                {"positionTruncateBitCount", 0}, {"minPositionTruncateBitCount", 0}, {"flags", "PreferFastTrace"},
                {"maxClusterTriangleCount", METALLIC_TESS_TRIANGLES_PER_PATCH}, {"maxClusterVertexCount", METALLIC_TESS_VERTICES_PER_PATCH},
                {"maxTotalTriangleCount", kPatchCount * METALLIC_TESS_TRIANGLES_PER_PATCH},
                {"maxTotalVertexCount", kPatchCount * METALLIC_TESS_VERTICES_PER_PATCH},
                {"maxAccelerationStructureCount", kPatchCount}, {"maxGeometryIndexValue", 0}, {"maxClusterUniqueGeometryCount", 1}}},
            {"capturedNativeSizes", capture.sizes}, {"arms", Json::array()}};
        report["globalBounds"] = capture.globalBounds;
        report["tightBounds"] = capture.tightBounds;
        report["rayInputSha256"] = debug::debugSha256(bytes(capture.rays));
        report["rayLayout"] = {{"bytesPerRay", sizeof(Ray)}, {"preciseRayCount", capture.imageOffset},
            {"imageCount", 3}, {"imageWidth", kImageSide}, {"imageHeight", kImageSide},
            {"bytesPerObservation", sizeof(Observation)}};
        struct Arm { std::string name; bool triangles, dyadic, narrow, opaque; uint32_t bounds; bool packedPositions = false; };
        std::vector<Arm> arms{{"triangles-i32", true, false, false, false, 0}, {"triangles-i8", true, false, true, false, 0}};
        for (bool dyadic : {false, true}) {
            for (bool narrow : {false, true}) {
                for (uint32_t bounds = 0; bounds < 3; ++bounds) {
                    arms.push_back({std::string(dyadic ? "dyadic" : "current") + (narrow ? "-i8-" : "-i32-") +
                        std::array{"none", "global", "tight"}[bounds], false, dyadic, narrow, false, bounds});
                }
            }
        }
        arms.push_back({"current-i32-none-opaque", false, false, false, true, 0});
        arms.push_back({"current-i32-global-stride12", false, false, false, false, 1, true});
        std::vector<Guide> packedPositions(capture.vertices.size());
        for (uint32_t vertex = 0; vertex < packedPositions.size(); ++vertex) {
            const auto& value = capture.vertices[vertex];
            packedPositions[vertex] = {value.x, value.y, value.z};
        }
        auto packedVertices = buffer(device, bytes(packedPositions).size(), MemoryLocation::HostUpload);
        upload(*packedVertices, packedPositions.data(), bytes(packedPositions).size());
        report["packedActualXYZSha256"] = debug::debugSha256(bytes(packedPositions));
        std::vector<Observation> reference;
        std::array<std::vector<Observation>, 4> unbounded;
        for (const auto& arm : arms) {
            const uint32_t indexStride = arm.narrow ? 1u : 4u;
            Buffer& indices = arm.narrow ? *indices8 : *indices32;
            const auto guideData = guides(capture, arm.dyadic);
            auto canonical = buffer(device, bytes(guideData).size(), MemoryLocation::HostUpload);
            upload(*canonical, guideData.data(), bytes(guideData).size());
            const uint64_t boundsStride = (24 + properties.templateBoundsAlignment - 1) / properties.templateBoundsAlignment * properties.templateBoundsAlignment;
            auto bounds = buffer(device, kPatchCount * boundsStride + properties.templateBoundsAlignment, MemoryLocation::HostUpload);
            const auto boundsSlice = alignedSlice(*bounds, properties.templateBoundsAlignment, kPatchCount * boundsStride);
            std::vector<uint8_t> boundsData(bounds->desc().size);
            for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
                const auto& values = arm.bounds == 2 ? capture.tightBounds[patch] : capture.globalBounds;
                std::memcpy(boundsData.data() + boundsSlice.offset() + patch * boundsStride, values.data(), 24);
            }
            upload(*bounds, boundsData.data(), boundsData.size());
            std::vector<ClusterAccelerationStructureTriangleTemplateIndirectBuildInfo> templateInfos(kPatchCount);
            std::vector<ClusterAccelerationStructureTriangleIndirectBuildInfo> triangleInfos(kPatchCount);
            for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
                auto& info = templateInfos[patch];
                info.triangle.clusterId = arm.triangles ? patch : 0;
                info.triangle.countsAndIndexType = METALLIC_CLAS_PACK_COUNTS(capture.leaves[patch].triangleCount,
                    capture.leaves[patch].vertexCount, 0u, arm.narrow ? METALLIC_CLAS_INDEX_UINT8 : METALLIC_CLAS_INDEX_UINT32);
                info.triangle.geometryIndexAndFlags = arm.opaque ? METALLIC_CLAS_GEOMETRY_OPAQUE : 0;
                info.triangle.indexAndVertexStrides = METALLIC_CLAS_PACK_STRIDES(indexStride, arm.triangles ? 32u : 12u);
                info.triangle.indexBufferAddress = indices.deviceAddress() + uint64_t(patch) * METALLIC_TESS_TRIANGLES_PER_PATCH * 3 * indexStride;
                info.triangle.vertexBufferAddress = (arm.triangles ? vertices->deviceAddress() : canonical->deviceAddress()) +
                    uint64_t(patch) * METALLIC_TESS_VERTICES_PER_PATCH * (arm.triangles ? 32u : 12u);
                info.instantiationBoundingBoxLimitAddress = arm.bounds ? boundsSlice.deviceAddress() + patch * boundsStride : 0;
                triangleInfos[patch] = info.triangle;
            }
            const std::array<BufferSlice, 3> inputs{slice(indices), arm.triangles ? slice(*vertices) : slice(*canonical), boundsSlice};
            BuiltClusters templates, clusters;
            if (arm.triangles) {
                clusters = buildClusters(context, ClusterAccelerationStructureTriangleOperation::Build, bytes(triangleInfos), sizeof(triangleInfos[0]), inputs);
            } else {
                templates = buildClusters(context, ClusterAccelerationStructureTriangleOperation::BuildTemplate, bytes(templateInfos), sizeof(templateInfos[0]), inputs);
                std::vector<ClusterAccelerationStructureTriangleTemplateInstantiationInfo> instantiate(kPatchCount);
                for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
                    instantiate[patch] = {.clusterIdOffset = patch, .clusterTemplateAddress = templates.addresses[patch],
                        .vertexBufferAddress = (arm.packedPositions ? packedVertices->deviceAddress() : vertices->deviceAddress()) +
                            uint64_t(patch) * METALLIC_TESS_VERTICES_PER_PATCH * (arm.packedPositions ? 12u : 32u),
                        .vertexBufferStride = arm.packedPositions ? 12u : 32u};
                }
                const std::array<BufferSlice, 1> instanceInputs{arm.packedPositions ? slice(*packedVertices) : slice(*vertices)};
                clusters = buildClusters(context, ClusterAccelerationStructureTriangleOperation::InstantiateTemplate,
                    bytes(instantiate), sizeof(instantiate[0]), instanceInputs, templates.storageSlice);
            }
            const auto observations = trace(context, capture, clusters, *vertices, *indices32, *rays, program);
            saveObservations(directory, arm.name, capture, observations);
            if (arm.name == "current-i32-global") {
                require(clusters.sizes == capture.sizes,
                    "Native diagnostic baseline does not reproduce the captured production CLAS sizes");
            }
            if (reference.empty()) {
                reference = observations;
                uint32_t checkedRays = 0;
                std::array<uint32_t, 3> imageHits{};
                double maxError = 0;
                for (uint32_t sample = 0; sample < 448; ++sample) {
                    const uint32_t imageSample = sample >= 256 ? sample - 256 : 0;
                    const uint32_t view = imageSample / 64;
                    const uint32_t cell = imageSample % 64;
                    const uint32_t x = ((cell % 8) * 2 + 1) * kImageSide / 16;
                    const uint32_t y = ((cell / 8) * 2 + 1) * kImageSide / 16;
                    const uint32_t index = sample < 256 ? sample * capture.imageOffset / 256 :
                        capture.imageOffset + view * kImageSide * kImageSide + y * kImageSide + x;
                    const double expected = cpuDistance(capture.rays[index], capture.triangles);
                    const auto& actual = observations[index];
                    require(bool(actual.hit) == std::isfinite(expected), "CPU oracle coverage mismatch at ray " + std::to_string(index));
                    if (actual.hit) {
                        if (sample >= 256) { ++imageHits[view]; }
                        const double error = std::abs(actual.distance - expected); maxError = std::max(maxError, error);
                        require(error <= kTolerance, "CPU oracle distance mismatch at ray " + std::to_string(index));
                    }
                    ++checkedRays;
                }
                require(std::accumulate(imageHits.begin(), imageHits.end(), 0u) > 0, "CPU image oracle sampled no geometry");
                report["cpuOracle"] = {{"rayCount", checkedRays}, {"maximumDistanceDifference", maxError},
                    {"imageSampleCount", 192}, {"imageHitCounts", imageHits},
                    {"imageSampling", "fixed 8x8 cell centers in each of three orthographic views"},
                    {"method", "double precision two-sided Moller-Trumbore over every captured triangle"}};
            }
            Json row{{"name", arm.name}, {"triangles", arm.triangles}, {"guide", arm.dyadic ? "round(q * 32768 / 65535) / 32768" : "q / 65535"},
                {"indexBits", arm.narrow ? 8 : 32}, {"geometryOpaque", arm.opaque}, {"bounds", std::array{"none", "global", "tight"}[arm.bounds]},
                {"actualVertexStride", arm.packedPositions ? 12 : 32},
                {"nativeSizes", clusters.sizes}, {"nativeTotalBytes", std::accumulate(clusters.sizes.begin(), clusters.sizes.end(), uint64_t(0))},
                {"templateNativeSizes", templates.sizes}, {"guideSha256", debug::debugSha256(bytes(guideData))},
                {"comparisonToTriangles", compare(capture, observations, reference)}};
            if (!arm.triangles && !arm.opaque) {
                const uint32_t key = uint32_t(arm.dyadic) * 2 + uint32_t(arm.narrow);
                if (arm.bounds == 0) { unbounded[key] = observations; }
                else { row["comparisonToUnbounded"] = compare(capture, observations, unbounded[key]); }
            }
            std::vector<float> guideValues;
            for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
                for (uint32_t vertex = 0; vertex < capture.leaves[patch].vertexCount; ++vertex) {
                    const auto& guide = guideData[patch * METALLIC_TESS_VERTICES_PER_PATCH + vertex];
                    guideValues.push_back(guide.x); guideValues.push_back(guide.y);
                }
            }
            row["guideXYBits"] = floatBits(guideValues);
            report["arms"].push_back(std::move(row));
            // Keep partial diagnostic evidence if a later arm fails. The
            // harness's atomic JSON writer intentionally finalizes files once.
            std::ofstream partial(directory / "EncodingMatrix.partial.json", std::ios::trunc);
            partial << report.dump(2) << '\n'; require(bool(partial), "Cannot write partial diagnostic report");
        }
        require(debug::debugSha256(bytes(read<StreamCLASTessellationVertex>(*vertices))) ==
            debug::debugSha256(bytes(capture.vertices)), "Actual vertex bytes changed during native matrix");
        report["actualVertexBytesUnchanged"] = true;
        report["complete"] = true;
        bench::writeJson(directory / "EncodingMatrix.json", report);
        if (context.evidence) { context.evidence->json("encoding-matrix.json", report); }
        return RHITestResult::pass("16 fixed-geometry native CLAS encoding arms, 2048 production patches, 448 CPU oracle rays, full ray/depth equivalence");
    }
};

METALLIC_REGISTER_RHI_TEST(CLASTemplateEncodingTest);

// Production-ray replay is deliberately separate from production timing. The
// captured fixed sample must have a single opaque instance; no alpha evaluator
// is substituted here for an arbitrary scene.
struct CapturedShadowRay {
    float originTMin[4], directionTMax[4];
    uint32_t control[4];
    float distances[4];
    uint32_t ids[4];
};
struct TraversalObservation {
    uint32_t hit, proceedCalls, candidates, cluster;
    float distance;
    uint32_t primitive, instance, accepted;
};
static_assert(sizeof(CapturedShadowRay) == 80 && sizeof(TraversalObservation) == 32);

struct TraversalScene {
    BuiltClusters templates, clusters;
    std::unique_ptr<Buffer> blas;
    std::unique_ptr<RayTracingAccelerationStructure> tlas;
    Json sizes;
};

void buildTraversalScene(RHITestContext& context, TraversalScene& result, RayTracingGPUInstance instance)
{
    auto& device = context.device;
    const auto properties = checked(device.queryClusterAccelerationStructureProperties());
    const auto limits = ClusterAccelerationStructureBottomLevelBuildSizesDesc{
        .maxClusterCountPerAccelerationStructure = kPatchCount, .maxTotalClusterCount = kPatchCount};
    const auto sizes = checked(device.queryClusterAccelerationStructureBottomLevelBuildSizes(limits));
    result.blas = buffer(device, sizes.accelerationStructureSize + properties.bottomLevelStorageAlignment);
    const auto storage = alignedSlice(*result.blas, properties.bottomLevelStorageAlignment, sizes.accelerationStructureSize);
    auto references = buffer(device, kPatchCount * 8, MemoryLocation::HostUpload);
    upload(*references, result.clusters.addresses.data(), kPatchCount * 8);
    const ClusterAccelerationStructureBottomLevelBuildInfo info{.clusterReferencesCount = kPatchCount,
        .clusterReferencesAddress = references->deviceAddress()};
    auto infos = buffer(device, sizeof(info), MemoryLocation::HostUpload); upload(*infos, &info, sizeof(info));
    auto address = buffer(device, 8, MemoryLocation::HostReadback);
    auto scratch = buffer(device, sizes.buildScratchSize + properties.scratchAlignment);
    {
        bench::GPUCommands commands(context.graphicsQueue); checked(commands.initialize(device));
        checked(commands.commands->buildClusterAccelerationStructureBottomLevels({
            .maxClusterCountPerAccelerationStructure = kPatchCount, .maxTotalClusterCount = kPatchCount,
            .buildInfoBuffer = slice(*infos), .destinationStorageBuffer = storage,
            .destinationAddressBuffer = slice(*address), .scratchBuffer = slice(*scratch)}));
        hostReadBarrier(*commands.commands); checked(commands.submitAndWait());
    }
    instance.accelerationStructureReference = read<uint64_t>(*address)[0];
    auto instances = buffer(device, sizeof(instance), MemoryLocation::HostUpload);
    upload(*instances, &instance, sizeof(instance));
    const auto tlasSizes = checked(device.queryRayTracingAccelerationStructureBuildSizes({.instanceCount = 1}));
    result.tlas = checked(device.createRayTracingAccelerationStructure({.size = tlasSizes.accelerationStructureSize}));
    const auto rtProperties = checked(device.queryRayTracingAccelerationStructureProperties());
    auto rtScratch = buffer(device, tlasSizes.buildScratchSize + rtProperties.scratchAlignment);
    bench::GPUCommands commands(context.graphicsQueue); checked(commands.initialize(device));
    checked(commands.commands->buildRayTracingAccelerationStructure({.destination = result.tlas.get(),
        .instanceBuffer = slice(*instances), .instanceCount = 1, .scratchBuffer = slice(*rtScratch)}));
    checked(commands.submitAndWait());
}

Json traceTraversalBatch(RHITestContext& context, TraversalScene& scene, Buffer& rays, Buffer& output, Buffer& readback,
    ComputeProgram& program, uint32_t width, uint32_t height, uint32_t selection, uint32_t warmup, uint32_t samples)
{
    auto& device = context.device;
    std::unique_ptr<TimestampQueryPool> queries;
    if (samples != 0) { queries = checked(device.createTimestampQueryPool(context.graphicsQueue, {.queryCount = samples * 2})); }
    struct Constants { uint64_t rays; uint32_t width, height, selection, padding; };
    const Constants constants{rays.deviceAddress(), width, height, selection, 0};
    auto pool = checked(device.createCommandPool(context.graphicsQueue)); auto commands = checked(pool->createCommandBuffer());
    QueueSubmissionTracker tracker; checked(tracker.initialize(device, context.graphicsQueue)); RenderFrameContext frame;
    struct Drain {
        RenderFrameContext& frame; CommandPool& pool;
        ~Drain() { if (frame.completion().isSubmitted()) { (void)frame.wait(); } (void)pool.reset(); (void)frame.reset(); }
    } drain{frame, *pool};
    checked(frame.begin(0)); checked(commands->begin(frame.submissionContext())); commands->hostWriteBarrier();
    checked(commands->retainResource(rays.retainAllocation()));
    if (queries) { checked(commands->resetTimestampQueries(*queries, 0, samples * 2)); }
    const ComputeDispatchBinding bindings[]{
        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), .accelerationStructure = scene.tlas.get()},
        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output), .buffer = &output}};
    // Serialize complete dispatches before the next TopOfPipe timestamp, so
    // an earlier dispatch cannot enter the following sample's measured window.
    const MemoryBarrierDesc between{.before = {PipelineStageBits::AllCommands, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderWrite}};
    const uint32_t dispatches = samples == 0 ? 1 : warmup + samples;
    for (uint32_t sample = 0; sample < dispatches; ++sample) {
        if (queries && sample >= warmup) { checked(commands->writeTimestamp(*queries, (sample - warmup) * 2, PipelineStageBits::TopOfPipe)); }
        checked(program.dispatch({.commandBuffer = commands.get(), .bindings = bindings,
            .pushData = &constants, .pushDataSize = sizeof(constants),
            .groupCountX = (width + 7) / 8, .groupCountY = (height + 7) / 8}));
        if (queries && sample >= warmup) { checked(commands->writeTimestamp(*queries, (sample - warmup) * 2 + 1, PipelineStageBits::BottomOfPipe)); }
        checked(commands->synchronize({.memory = {&between, 1}}));
    }
    const MemoryBarrierDesc transfer{.before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
    checked(commands->synchronize({.memory = {&transfer, 1}}));
    checked(commands->copyBuffer(slice(output), slice(readback)));
    hostReadBarrier(*commands); checked(commands->end()); CommandBuffer* submitted[]{commands.get()};
    checked(tracker.submit({.commandBuffers = submitted}, frame)); checked(frame.wait(30'000'000'000ull));
    if (!queries) { return {{"dispatches", dispatches}}; }
    std::vector<TimestampQueryResult> timestamps(samples * 2); checked(queries->readResults(0, timestamps));
    std::vector<double> timings;
    for (uint32_t sample = 0; sample < samples; ++sample) {
        require(timestamps[sample * 2].available && timestamps[sample * 2 + 1].available, "Replay timestamps unavailable");
        const double ms = queries->durationMilliseconds(timestamps[sample * 2].value, timestamps[sample * 2 + 1].value);
        require(std::isfinite(ms) && ms > 0, "Invalid replay timestamp duration"); timings.push_back(ms);
    }
    auto sorted = timings; std::sort(sorted.begin(), sorted.end());
    return {{"samplesMs", timings}, {"medianMs", (sorted[samples / 2 - 1] + sorted[samples / 2]) * .5},
        {"warmupDispatches", warmup}, {"timedDispatches", samples}};
}

class CLASTemplateTraversalTest final : public RHITest {
public:
    CLASTemplateTraversalTest() { type = RHITestType::Rendering; name = "clas_template_traversal_diagnostic"; }
    std::optional<bench::Metadata> metadata() const override
    {
        auto result = bench::gpuMetadata({"clas.template.traversal.production-shadow-rays"},
            bench::Layer::RHI, "ray-query", "extensions", {"traversal-matrix.json"});
        result.timeout = std::chrono::seconds(180); return result;
    }
    RHITestResult run(RHITestContext& context) override
    {
        const char* captureFile = std::getenv("METALLIC_CLAS_TRAVERSAL_CAPTURE");
        const char* guideFile = std::getenv("METALLIC_CLAS_TRAVERSAL_GUIDES");
        if (!captureFile || !guideFile) { return RHITestResult::skip("Set production shadow capture and guide manifest paths"); }
        auto created = bench::createTestDevice(context, {.applicationName = "CLAS production-ray traversal diagnostic",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .enableRayQuery = true, .enableClusterAccelerationStructure = true});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("CLAS/ray query unavailable"); }
        if (!created) { return RHITestResult::fail("CLAS diagnostic device creation failed"); }
        auto& device = **created;
        RHITestContext active{device, *device.getQueue(QueueType::Graphics), context.outputDirectory,
            context.enableValidation, context.validationMessageCount, context.nsightCapture, context.evidence, context.deviceDesc};
        try { return runDevice(active, captureFile, guideFile); }
        catch (const std::exception& error) { (void)device.waitIdle(); return RHITestResult::fail(error.what()); }
    }
private:
    RHITestResult runDevice(RHITestContext& context, const std::filesystem::path& captureFile,
        const std::filesystem::path& guideFile)
    {
        const auto readJson = [](const std::filesystem::path& path) { std::ifstream input(path); Json value; input >> value; return value; };
        const Json shadow = readJson(captureFile), guideManifest = readJson(guideFile);
        const auto geometryFile = captureFile.parent_path() / shadow.at("geometryFile").get<std::string>();
        const auto capture = loadGeometry(geometryFile);
        require(shadow.at("geometryManifestSha256") == debug::debugSha256(loadBytes(geometryFile)), "Shadow/geometry frame identity mismatch");
        require(guideManifest.at("geometryManifestSha") == debug::debugSha256(loadBytes(geometryFile)), "Guides derived from another geometry capture");
        require(shadow.at("recordStride") == sizeof(CapturedShadowRay) && shadow.at("queryFlags") == 0 && shadow.at("queryMask") == 255,
            "Unexpected production shadow query contract");
        const uint32_t width = shadow.at("width"), height = shadow.at("height");
        require(width == 1444 && height == 694, "Replay requires the fixed full-resolution workload");
        const auto rays = loadCapture<CapturedShadowRay>(shadow, captureFile.parent_path(), "recordsFile", size_t(width) * height);
        require(capture.manifest.at("instanceCount") == 1, "Replay requires one captured stream instance");
        const auto instances = loadCapture<RayTracingGPUInstance>(capture.manifest, geometryFile.parent_path(), "tlasInstancesFile", 1);
        const auto instance = instances[0];
        const RayTracingGPUInstance identity;
        const uint32_t flags = instance.shaderBindingTableRecordOffsetAndFlags >> 24;
        require(std::memcmp(instance.transform, identity.transform, sizeof(instance.transform)) == 0 &&
            (instance.customIndexAndMask >> 24) == 255 &&
            (flags & uint32_t(RayTracingInstanceFlags::ForceOpaque)) != 0 && (flags & ~7u) == 0,
            "Replay requires captured identity transform, mask 255 and FORCE_OPAQUE instance");
        require(std::ranges::all_of(capture.leaves, [](const auto& leaf) { return leaf.instanceIndex == 0; }), "Multiple geometry instances");
        const uint32_t baseId = capture.manifest.at("generatedBaseClusterId");
        uint64_t activeCount = 0, hitCount = 0;
        std::array<std::vector<uint32_t>, 2> oraclePixels;
        for (uint32_t pixel = 0; pixel < rays.size(); ++pixel) {
            const auto& ray = rays[pixel];
            require(ray.control[0] <= 1 && ray.control[1] <= 1, "Invalid captured shadow state");
            if (!ray.control[0]) { continue; }
            ++activeCount; hitCount += ray.control[1];
            if (ray.control[1]) { require(std::isfinite(ray.distances[0]), "Non-finite captured hit distance"); }
            require(ray.control[2] == 1 && ray.control[3] == 0 && ray.ids[3] == 0,
                "Production opacity path is not the single opaque traversal expected by this replay");
            require(ray.originTMin[3] == .001f && ray.directionTMax[3] >= ray.originTMin[3], "Unexpected production ray interval");
            for (float value : ray.originTMin) { require(std::isfinite(value), "Non-finite captured ray"); }
            for (float value : ray.directionTMax) { require(std::isfinite(value), "Non-finite captured ray"); }
            oraclePixels[ray.control[1]].push_back(pixel);
        }
        require(activeCount > 0 && hitCount > 0 && hitCount < activeCount, "Expected both actual shadow hits and misses");
        auto& device = context.device;
        auto vertexBuffer = buffer(device, bytes(capture.vertices).size(), MemoryLocation::HostUpload);
        upload(*vertexBuffer, capture.vertices.data(), bytes(capture.vertices).size());
        auto indexBuffer = buffer(device, bytes(capture.indices).size(), MemoryLocation::HostUpload);
        upload(*indexBuffer, capture.indices.data(), bytes(capture.indices).size());
        auto rayUpload = buffer(device, bytes(rays).size(), MemoryLocation::HostUpload);
        upload(*rayUpload, rays.data(), bytes(rays).size());
        auto rayBuffer = buffer(device, bytes(rays).size());
        {
            bench::GPUCommands commands(context.graphicsQueue); checked(commands.initialize(device));
            checked(commands.commands->copyBuffer(slice(*rayUpload), slice(*rayBuffer)));
            const MemoryBarrierDesc ready{.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
            checked(commands.commands->synchronize({.memory = {&ready, 1}})); checked(commands.submitAndWait());
        }
        auto output = buffer(device, rays.size() * sizeof(TraversalObservation));
        auto readback = buffer(device, rays.size() * sizeof(TraversalObservation), MemoryLocation::HostReadback);
        auto timingOutput = buffer(device, rays.size() * sizeof(float));
        auto timingReadback = buffer(device, rays.size() * sizeof(float), MemoryLocation::HostReadback);
        const char* capabilities[]{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"};
        std::string log;
        const auto shader = checked(compileSlangShaderToSpirv({.moduleName = "CLASTemplateTraversalProbe", .entryPointName = "clasTemplateTraversalProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .capabilities = capabilities,
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log));
        const auto timingShader = checked(compileSlangShaderToSpirv({.moduleName = "CLASTemplateTraversalProbe", .entryPointName = "clasTemplateTraversalTimingMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .capabilities = capabilities,
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log));
        const ComputeResourceBindingDesc layout[]{
            {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), ComputeResourceBindingKind::AccelerationStructure},
            {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output)}};
        ComputeProgram program; checked(program.initialize(device, {.spirv = shader.spirv, .pushConstantSize = 24,
            .bindings = layout, .resourceParameterSize = sizeof(UnifiedTopLevelProbeResources)}, log));
        ComputeProgram timingProgram; checked(timingProgram.initialize(device, {.spirv = timingShader.spirv, .pushConstantSize = 24,
            .bindings = layout, .resourceParameterSize = sizeof(UnifiedTopLevelProbeResources)}, log));
        const auto directory = context.outputDirectory / name; std::filesystem::create_directories(directory);
        const auto saveShader = [&](const char* filename, const std::vector<uint32_t>& spirv) {
            std::ofstream output(directory / filename, std::ios::binary);
            output.write(reinterpret_cast<const char*>(spirv.data()), std::streamsize(spirv.size() * sizeof(uint32_t)));
            require(bool(output), "Cannot save executed replay SPIR-V");
        };
        saveShader("correctness.spv", shader.spirv);
        saveShader("timing.spv", timingShader.spirv);
        Json report{{"schemaVersion", 2}, {"scope", "counter-free isolated production-ray replay; diagnostic only, not whole Shadow pass or acceptance timing"},
            {"shadowManifestSha256", debug::debugSha256(loadBytes(captureFile))}, {"guideManifestSha256", debug::debugSha256(loadBytes(guideFile))},
            {"correctnessSpirvSha256", debug::debugSha256(bytes(shader.spirv))},
            {"timingSpirvSha256", debug::debugSha256(bytes(timingShader.spirv))}, {"validation", context.enableValidation},
            {"correctnessEntry", "clasTemplateTraversalProbeMain"}, {"timingEntry", "clasTemplateTraversalTimingMain"},
            {"correctnessOutputStride", sizeof(TraversalObservation)}, {"timingOutputStride", sizeof(float)},
            {"rayInputStride", sizeof(CapturedShadowRay)}, {"counterFree", true},
            {"templateSharing", "per-patch templates; not production shared pattern pool"},
            {"width", width}, {"height", height}, {"activeRays", activeCount}, {"hitRays", hitCount},
            {"queryFlags", 0}, {"queryMask", 255}, {"instanceCustomIndexAndMask", instance.customIndexAndMask},
            {"instanceSbtAndFlags", instance.shaderBindingTableRecordOffsetAndFlags},
            {"arms", Json::array()}, {"correctness", Json::array()}, {"batches", Json::array()}};
        const std::array<std::string, 5> names{"triangles", "current", "actual", "affine", "rightCorner"};
        std::array<TraversalScene, 5> scenes;
        for (uint32_t arm = 0; arm < names.size(); ++arm) {
            auto& scene = scenes[arm];
            std::vector<ClusterAccelerationStructureTriangleTemplateIndirectBuildInfo> infos(kPatchCount);
            std::vector<ClusterAccelerationStructureTriangleIndirectBuildInfo> triangleInfos(kPatchCount);
            std::unique_ptr<Buffer> guideBuffer;
            if (arm != 0) {
                const auto& record = guideManifest.at("guides").at(names[arm]);
                const auto raw = loadBytes(guideFile.parent_path() / record.at("file").get<std::string>());
                require(raw.size() == size_t(kPatchCount) * METALLIC_TESS_VERTICES_PER_PATCH * sizeof(Guide) &&
                    record.at("bytes") == raw.size() && record.at("sha256") == debug::debugSha256(raw), "Invalid guide receipt");
                guideBuffer = buffer(device, raw.size(), MemoryLocation::HostUpload); upload(*guideBuffer, raw.data(), raw.size());
            }
            for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
                auto& info = infos[patch].triangle;
                info.clusterId = arm == 0 ? baseId + patch : 0;
                info.countsAndIndexType = METALLIC_CLAS_PACK_COUNTS(capture.leaves[patch].triangleCount,
                    capture.leaves[patch].vertexCount, 0u, METALLIC_CLAS_INDEX_UINT32);
                info.indexAndVertexStrides = METALLIC_CLAS_PACK_STRIDES(4u, arm == 0 ? 32u : 12u);
                info.indexBufferAddress = indexBuffer->deviceAddress() + uint64_t(patch) * METALLIC_TESS_TRIANGLES_PER_PATCH * 12u;
                info.vertexBufferAddress = (arm == 0 ? vertexBuffer : guideBuffer)->deviceAddress() +
                    uint64_t(patch) * METALLIC_TESS_VERTICES_PER_PATCH * (arm == 0 ? 32u : 12u);
                triangleInfos[patch] = info;
            }
            const std::array inputs{slice(*indexBuffer), slice(*(arm == 0 ? vertexBuffer : guideBuffer))};
            if (arm == 0) {
                scene.clusters = buildClusters(context, ClusterAccelerationStructureTriangleOperation::Build,
                    bytes(triangleInfos), sizeof(triangleInfos[0]), inputs);
            } else {
                scene.templates = buildClusters(context, ClusterAccelerationStructureTriangleOperation::BuildTemplate,
                    bytes(infos), sizeof(infos[0]), inputs);
                std::vector<ClusterAccelerationStructureTriangleTemplateInstantiationInfo> instantiate(kPatchCount);
                for (uint32_t patch = 0; patch < kPatchCount; ++patch) {
                    instantiate[patch] = {.clusterIdOffset = baseId + patch, .clusterTemplateAddress = scene.templates.addresses[patch],
                        .vertexBufferAddress = vertexBuffer->deviceAddress() + uint64_t(patch) * METALLIC_TESS_VERTICES_PER_PATCH * 32u,
                        .vertexBufferStride = 32};
                }
                const std::array actualInputs{slice(*vertexBuffer)};
                scene.clusters = buildClusters(context, ClusterAccelerationStructureTriangleOperation::InstantiateTemplate,
                    bytes(instantiate), sizeof(instantiate[0]), actualInputs, scene.templates.storageSlice);
                if (arm == 1) { require(scene.clusters.sizes == capture.sizes, "Current-guide replay does not reproduce native production sizes"); }
            }
            buildTraversalScene(context, scene, instance);
            report["arms"].push_back({{"name", names[arm]}, {"nativeSizes", scene.clusters.sizes},
                {"templateCount", arm == 0 ? 0u : kPatchCount},
                {"nativeTotalBytes", std::accumulate(scene.clusters.sizes.begin(), scene.clusters.sizes.end(), uint64_t(0))}});
        }
        double oracleError = 0;
        for (const auto& pixels : oraclePixels) {
            for (uint32_t sample = 0; sample < 256; ++sample) {
                const auto& input = rays[pixels[size_t(sample) * pixels.size() / 256]];
                Ray ray{}; std::memcpy(&ray, &input, sizeof(ray));
                const double expected = cpuDistance(ray, capture.triangles);
                require(std::isfinite(expected) == bool(input.control[1]), "Production shadow/CPU oracle coverage mismatch");
                if (std::isfinite(expected)) {
                    oracleError = std::max(oracleError, std::abs(expected - input.distances[0]));
                    require(std::abs(expected - input.distances[0]) <= kTolerance, "Production shadow/CPU oracle distance mismatch");
                }
            }
        }
        report["cpuOracle"] = {{"rayCount", 512}, {"hitSamples", 256}, {"missSamples", 256}, {"maximumDistanceDifference", oracleError}};
        for (uint32_t arm = 0; arm < names.size(); ++arm) {
            Json correctness = traceTraversalBatch(context, scenes[arm], *rayBuffer, *output, *readback,
                program, width, height, 0, 0, 0);
            const auto observations = read<TraversalObservation>(*readback);
            std::ofstream raw(directory / (names[arm] + ".bin"), std::ios::binary);
            raw.write(reinterpret_cast<const char*>(observations.data()), observations.size() * sizeof(TraversalObservation));
            require(bool(raw), "Cannot save ray replay observations");
            uint64_t hitDifferences = 0, idDifferences = 0, candidates = 0, proceedCalls = 0;
            double maximumDistanceDifference = 0;
            for (size_t pixel = 0; pixel < rays.size(); ++pixel) {
                const auto& input = rays[pixel]; const auto& actual = observations[pixel];
                const bool expectedHit = input.control[0] && input.control[1];
                hitDifferences += actual.hit != uint32_t(expectedHit);
                candidates += actual.candidates; proceedCalls += actual.proceedCalls;
                if (expectedHit && actual.hit) {
                    require(std::isfinite(actual.distance), "Non-finite native replay hit distance");
                    maximumDistanceDifference = std::max(maximumDistanceDifference, double(std::abs(actual.distance - input.distances[0])));
                    idDifferences += actual.instance != input.ids[0] || actual.cluster != input.ids[1] || actual.primitive != input.ids[2];
                    require(actual.cluster >= baseId && actual.cluster < baseId + kPatchCount &&
                        actual.primitive < capture.leaves[actual.cluster - baseId].triangleCount, "Native replay hit ID outside captured geometry");
                }
            }
            require(hitDifferences == 0 && idDifferences == 0 && candidates == 0 && maximumDistanceDifference <= kTolerance,
                "Native replay differs from production closest opaque shadow: hits=" + std::to_string(hitDifferences) +
                ", IDs=" + std::to_string(idDifferences) + ", candidates=" + std::to_string(candidates) +
                ", maxDistance=" + std::to_string(maximumDistanceDifference));
            require(proceedCalls == activeCount, "Opaque replay query invocation count differs");
            correctness.update({{"arm", names[arm]}, {"selection", "all"}, {"hitDifferences", hitDifferences},
                {"idDifferences", idDifferences}, {"nonOpaqueCandidates", candidates}, {"proceedCalls", proceedCalls},
                {"maximumDistanceDifference", maximumDistanceDifference}});
            report["correctness"].push_back(std::move(correctness));
            std::vector<uint8_t> pixels(size_t(width) * height * 4, 255);
            for (size_t pixel = 0; pixel < observations.size(); ++pixel) {
                for (uint32_t channel = 0; channel < 3; ++channel) { pixels[pixel * 4 + channel] = observations[pixel].hit ? 0 : 255; }
            }
            require(saveRgba8Png(directory / (names[arm] + ".png"), pixels.data(), width, height, log), "Cannot save replay visibility image");
        }
        for (uint32_t round = 0; round < 2; ++round) {
            for (uint32_t order = 0; order < names.size(); ++order) {
                const uint32_t arm = round == 0 ? order : uint32_t(names.size()) - 1 - order;
                for (uint32_t selection = 0; selection < 3; ++selection) {
                    Json batch = traceTraversalBatch(context, scenes[arm], *rayBuffer, *timingOutput, *timingReadback,
                        timingProgram, width, height, selection, 8, 32);
                    const auto distances = read<float>(*timingReadback);
                    uint64_t hitDifferences = 0;
                    double maximumDistanceDifference = 0;
                    for (size_t pixel = 0; pixel < rays.size(); ++pixel) {
                        const auto& input = rays[pixel]; const float distance = distances[pixel];
                        const bool selected = input.control[0] && (selection == 0 || (selection == 1) == bool(input.control[1]));
                        const bool expectedHit = selected && input.control[1];
                        require(std::isfinite(distance), "Non-finite timed replay distance");
                        const bool actualHit = distance >= 0;
                        hitDifferences += actualHit != expectedHit;
                        if (expectedHit && actualHit) {
                            maximumDistanceDifference = std::max(maximumDistanceDifference, double(std::abs(distance - input.distances[0])));
                        } else if (!actualHit) {
                            require(distance == -1.f, "Invalid timed replay miss sentinel");
                        }
                    }
                    require(hitDifferences == 0 && maximumDistanceDifference <= kTolerance,
                        "Timed replay differs from production closest opaque shadow: hits=" + std::to_string(hitDifferences) +
                        ", maxDistance=" + std::to_string(maximumDistanceDifference));
                    batch.update({{"arm", names[arm]}, {"round", round}, {"selection", std::array{"all", "hits", "misses"}[selection]},
                        {"hitDifferences", hitDifferences}, {"maximumDistanceDifference", maximumDistanceDifference}});
                    report["batches"].push_back(std::move(batch));
                }
            }
        }
        require(debug::debugSha256(bytes(read<StreamCLASTessellationVertex>(*vertexBuffer))) == debug::debugSha256(bytes(capture.vertices)),
            "Replay changed actual generated vertices");
        report["actualVertexBytesUnchanged"] = true;
        report["complete"] = true;
        bench::writeJson(directory / "TraversalMatrix.json", report);
        if (context.evidence) { context.evidence->json("traversal-matrix.json", report); }
        return RHITestResult::pass("Five fixed-geometry guide controls, actual opaque production shadow rays, closest-hit and image checks");
    }
};
METALLIC_REGISTER_RHI_TEST(CLASTemplateTraversalTest);

} // namespace
} // namespace metallic::tests
