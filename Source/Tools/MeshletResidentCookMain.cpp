#include "Runtime/Scene/GeometryAttributes.h"
#include "Runtime/Scene/Scene.h"

#include <meshoptimizer.h>
#include <json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#endif

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using metallic::scene::Bounds;
using metallic::scene::RenderPrimitive;
using metallic::scene::Scene;

constexpr uint32_t kResidentFormatVersion = 4;
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

struct Options {
    fs::path source;
    fs::path output;
    fs::path report;
};

double elapsedSeconds(Clock::time_point begin)
{
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

void require(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view name(argv[index]);
        require(index + 1 < argc, "Missing value for " + std::string(name));
        fs::path* value = name == "--source" ? &options.source :
            name == "--output" ? &options.output :
            name == "--report" ? &options.report : nullptr;
        require(value != nullptr, "Unknown option: " + std::string(name));
        require(value->empty(), "Duplicate option: " + std::string(name));
        *value = fs::u8path(argv[++index]);
    }
    require(!options.source.empty() && !options.output.empty() && !options.report.empty(),
        "--source, --output and --report are required");
    return options;
}

bool sameComponent(const fs::path& left, const fs::path& right)
{
#if defined(_WIN32)
    auto a = left.native();
    auto b = right.native();
    std::transform(a.begin(), a.end(), a.begin(), [](wchar_t c) { return std::towlower(c); });
    std::transform(b.begin(), b.end(), b.begin(), [](wchar_t c) { return std::towlower(c); });
    return a == b;
#else
    return left == right;
#endif
}

bool pathIsInside(const fs::path& path, const fs::path& directory)
{
    auto part = path.begin();
    for (const auto& expected : directory) {
        if (part == path.end() || !sameComponent(*part++, expected)) { return false; }
    }
    return part != path.end();
}

bool pathIsSame(const fs::path& left, const fs::path& right)
{
    auto a = left.begin();
    auto b = right.begin();
    for (; a != left.end() && b != right.end(); ++a, ++b) {
        if (!sameComponent(*a, *b)) { return false; }
    }
    return a == left.end() && b == right.end();
}

bool pathRefersToSameFile(const fs::path& left, const fs::path& right)
{
    if (pathIsSame(left, right)) { return true; }
    std::error_code error;
    return fs::equivalent(left, right, error) && !error;
}

bool pathExists(const fs::path& path)
{
    std::error_code error;
    const auto status = fs::symlink_status(path, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        throw fs::filesystem_error("Cannot inspect path", path, error);
    }
    return fs::exists(status);
}

fs::path cachePathFor(fs::path source)
{
    source += ".meshlets.bin";
    return source;
}

struct TemporarySource {
    fs::path source;
    fs::path cache;

    ~TemporarySource()
    {
        std::error_code ignored;
        if (!cache.empty()) { fs::remove(cache, ignored); }
        if (!source.empty()) { fs::remove(source, ignored); }
    }

    void create(const fs::path& original, const fs::path& assetRoot)
    {
        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (uint32_t attempt = 0; attempt < 64; ++attempt) {
            const fs::path candidate = original.parent_path() /
                (original.stem().string() + ".resident-cook-" + std::to_string(nonce) +
                    "-" + std::to_string(attempt) + original.extension().string());
            const fs::path candidateCache = cachePathFor(candidate);
            require(pathIsInside(candidate, assetRoot) && pathIsInside(candidateCache, assetRoot),
                "Temporary paths must stay inside the repository Asset directory");
            if (pathExists(candidate) || pathExists(candidateCache)) { continue; }
            source = candidate;
            std::error_code error;
            if (!fs::copy_file(original, candidate, fs::copy_options::none, error)) {
                if (error == std::errc::file_exists) {
                    source.clear();
                    continue;
                }
                throw fs::filesystem_error("Cannot copy temporary source", original, candidate, error);
            }
            cache = candidateCache;
            fs::last_write_time(source, fs::last_write_time(original));
            return;
        }
        throw std::runtime_error("Cannot allocate a unique temporary source path");
    }
};

uint64_t hashBytes(uint64_t hash, const void* data, size_t size)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
uint64_t hashValue(uint64_t hash, const T& value)
{
    return hashBytes(hash, &value, sizeof(value));
}

// Match the resident cache's source geometry hash, excluding float3 padding.
uint64_t geometryHash(const RenderPrimitive& primitive)
{
    uint64_t hash = kFnvOffset;
    hash = hashValue(hash, primitive.meshIndex);
    hash = hashValue(hash, primitive.primitiveIndex);
    hash = hashValue(hash, primitive.mode);
    hash = hashValue(hash, primitive.vertexCount);
    hash = hashValue(hash, primitive.indexCount);
    hash = hashValue(hash, primitive.triangleCount);
    for (const auto* values : {&primitive.positions, &primitive.normals}) {
        hash = hashValue(hash, static_cast<uint64_t>(values->size()));
        for (const auto& value : *values) {
            hash = hashValue(hash, value.x);
            hash = hashValue(hash, value.y);
            hash = hashValue(hash, value.z);
        }
    }
    hash = hashValue(hash, static_cast<uint64_t>(primitive.texcoords0.size()));
    for (const auto& value : primitive.texcoords0) {
        hash = hashValue(hash, value.x);
        hash = hashValue(hash, value.y);
    }
    hash = hashValue(hash, static_cast<uint64_t>(primitive.tangents.size()));
    for (const auto& value : primitive.tangents) {
        hash = hashValue(hash, value.x);
        hash = hashValue(hash, value.y);
        hash = hashValue(hash, value.z);
        hash = hashValue(hash, value.w);
    }
    hash = hashValue(hash, static_cast<uint64_t>(primitive.indices.size()));
    return hashBytes(hash, primitive.indices.data(), primitive.indices.size() * sizeof(uint32_t));
}

Json boundsJson(const Bounds& bounds)
{
    return { {"valid", bounds.valid},
        {"min", {bounds.min.x, bounds.min.y, bounds.min.z}},
        {"max", {bounds.max.x, bounds.max.y, bounds.max.z}} };
}

Json primitiveJson(const RenderPrimitive& primitive)
{
    return { {"name", primitive.name}, {"meshIndex", primitive.meshIndex},
        {"primitiveIndex", primitive.primitiveIndex}, {"materialIndex", primitive.materialIndex},
        {"mode", primitive.mode}, {"vertexCount", primitive.vertexCount},
        {"indexCount", primitive.indexCount}, {"triangleCount", primitive.triangleCount},
        {"normalCount", primitive.normals.size()}, {"uvCount", primitive.texcoords0.size()},
        {"tangentCount", primitive.tangents.size()},
        {"geometryHash", geometryHash(primitive)}, {"bounds", boundsJson(primitive.localBounds)},
        {"meshletClusterCount", primitive.meshletClusters.size()},
        {"meshletLodLevelCount", primitive.meshletLodLevels.size()},
        {"meshletLodGroupCount", primitive.meshletLodGroups.size()},
        {"meshletLodClusterCount", primitive.meshletLodClusters.size()},
        {"meshletLodVertexReferences", primitive.meshletLodVertices.size()},
        {"meshletLodTriangleIndices", primitive.meshletLodTriangles.size()} };
}

using Triangle = std::array<uint32_t, 3>;

Triangle directedTriangle(uint32_t a, uint32_t b, uint32_t c)
{
    return std::min({Triangle{a, b, c}, Triangle{b, c, a}, Triangle{c, a, b}});
}

uint64_t validateLod0(const RenderPrimitive& primitive)
{
    std::string reason;
    const bool validAttributes = metallic::scene::validateGeometryAttributes(primitive, reason);
    require(validAttributes, "Invalid source attributes: " + reason);
    if (primitive.mode != 4) {
        require(primitive.meshletLodClusters.empty(), "Non-triangle primitive has LOD clusters");
        return 0;
    }
    require(primitive.positions.size() <= std::numeric_limits<uint32_t>::max(),
        "Source vertex count exceeds resident index range");
    std::vector<uint32_t> indices = primitive.indices;
    if (indices.empty()) {
        indices.resize((primitive.positions.size() / 3) * 3);
        for (size_t index = 0; index < indices.size(); ++index) {
            indices[index] = static_cast<uint32_t>(index);
        }
    }
    if (indices.empty()) {
        require(primitive.meshletLodClusters.empty(), "Empty source has LOD clusters");
        return 0;
    }

    // The builder welds only identical full attribute tuples. Preserve UV,
    // normal and tangent seams while accepting these equivalent source IDs.
    std::array<meshopt_Stream, 4> streams{};
    size_t streamCount = 0;
    streams[streamCount++] = {primitive.positions.data(), 3 * sizeof(float), sizeof(float3)};
    if (!primitive.normals.empty()) {
        streams[streamCount++] = {primitive.normals.data(), 3 * sizeof(float), sizeof(float3)};
    }
    if (!primitive.texcoords0.empty()) {
        streams[streamCount++] = {primitive.texcoords0.data(), 2 * sizeof(float), sizeof(float2)};
    }
    if (!primitive.tangents.empty()) {
        streams[streamCount++] = {primitive.tangents.data(), 4 * sizeof(float), sizeof(float4)};
    }
    std::vector<unsigned int> remap(primitive.positions.size());
    meshopt_generateVertexRemapMulti(remap.data(), indices.data(), indices.size(),
        primitive.positions.size(), streams.data(), streamCount);
    std::vector<Triangle> expected;
    expected.reserve(indices.size() / 3);
    for (size_t index = 0; index < indices.size(); index += 3) {
        expected.push_back(directedTriangle(
            remap[indices[index]], remap[indices[index + 1]], remap[indices[index + 2]]));
    }
    // Release the copied index buffer before constructing the second multiset.
    std::vector<uint32_t>().swap(indices);
    std::vector<Triangle> actual;
    actual.reserve(expected.size());
    for (const auto& cluster : primitive.meshletLodClusters) {
        if (cluster.lodLevel != 0) { continue; }
        require(uint64_t(cluster.vertexOffset) + cluster.vertexCount <= primitive.meshletLodVertices.size() &&
            uint64_t(cluster.triangleOffset) + uint64_t(cluster.triangleCount) * 3 <= primitive.meshletLodTriangles.size(),
            "LOD0 cluster references an invalid payload range");
        for (uint32_t triangle = 0; triangle < cluster.triangleCount; ++triangle) {
            Triangle vertices{};
            for (size_t corner = 0; corner < 3; ++corner) {
                const uint32_t local = primitive.meshletLodTriangles[
                    size_t(cluster.triangleOffset) + size_t(triangle) * 3 + corner];
                require(local < cluster.vertexCount, "LOD0 triangle has an invalid local vertex");
                const uint32_t global = primitive.meshletLodVertices[size_t(cluster.vertexOffset) + local];
                require(global < remap.size() && remap[global] != ~0u,
                    "LOD0 triangle references a vertex absent from the source topology");
                vertices[corner] = remap[global];
            }
            actual.push_back(directedTriangle(vertices[0], vertices[1], vertices[2]));
        }
    }
    require(actual.size() == expected.size(), "LOD0 triangle count differs from source");
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    require(actual == expected, "LOD0 directed triangle multiset differs from source (including winding)");
    return actual.size();
}

void validateCacheHeader(const fs::path& cache, uint64_t sourceSize, int64_t sourceTime)
{
    std::ifstream input(cache, std::ios::binary);
    std::array<char, 8> magic{};
    uint32_t version = 0;
    uint32_t endian = 0;
    uint64_t size = 0;
    int64_t time = 0;
    input.read(magic.data(), magic.size());
    input.read(reinterpret_cast<char*>(&version), sizeof(version));
    input.read(reinterpret_cast<char*>(&endian), sizeof(endian));
    input.read(reinterpret_cast<char*>(&size), sizeof(size));
    input.read(reinterpret_cast<char*>(&time), sizeof(time));
    require(input.good() && magic == std::array<char, 8>{'M', 'T', 'L', 'M', 'S', 'H', 'L', 'T'} &&
        version == kResidentFormatVersion && endian == 0x01020304 &&
        size == sourceSize && time == sourceTime, "Resident cache header does not match original source/version");
}

void moveWithoutReplacing(const fs::path& source, const fs::path& output)
{
    require(!pathExists(output), "Output already exists: " + output.string());
#if defined(_WIN32)
    if (!MoveFileExW(source.c_str(), output.c_str(), MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error("Cannot stage cache; Win32 error " + std::to_string(GetLastError()));
    }
#else
    // Creating a hard link is atomic and refuses to replace an existing file.
    fs::create_hard_link(source, output);
    fs::remove(source);
#endif
}

void writeReport(const fs::path& path, const Json& report)
{
    if (!path.parent_path().empty()) { fs::create_directories(path.parent_path()); }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.is_open(), "Cannot open report: " + path.string());
    output << report.dump(2) << '\n';
    output.close();
    require(!output.fail(), "Cannot write report: " + path.string());
}

void cook(Options& options, Json& report, bool& reportSafe)
{
    const auto begin = Clock::now();
    const fs::path assetRoot = fs::canonical(METALLIC_ASSET_ROOT);
    options.source = fs::canonical(options.source);
    options.output = fs::weakly_canonical(fs::absolute(options.output));
    options.report = fs::weakly_canonical(fs::absolute(options.report));
    require(pathIsInside(options.source, assetRoot) && pathIsInside(options.output, assetRoot),
        "Source and staged output must stay inside the repository Asset directory");
    require(fs::is_regular_file(options.source), "Source is not a regular file");
    const auto extension = options.source.extension().string();
    require(extension == ".gltf" || extension == ".glb" || extension == ".usda" || extension == ".usd" || extension == ".usdc",
        "Source must be glTF/GLB/USD");
    require(!pathIsSame(options.output, options.source) &&
        !pathIsSame(options.output, cachePathFor(options.source)), "Output must be a separate staged path");
    require(!pathRefersToSameFile(options.report, options.source) && !pathRefersToSameFile(options.report, options.output) &&
        !pathRefersToSameFile(options.report, cachePathFor(options.source)), "Report must not overwrite source or cache");
    reportSafe = true;
    require(!pathExists(options.output), "Output already exists: " + options.output.string());
    const uint64_t sourceSize = fs::file_size(options.source);
    const auto sourceTime = fs::last_write_time(options.source);
    report["source"] = options.source.generic_string();
    report["output"] = options.output.generic_string();
    report["asset"] = options.output.generic_string();
    report["formatVersion"] = kResidentFormatVersion;
    report["sourceBytes"] = sourceSize;
    report["sourceWriteTimeTicks"] = sourceTime.time_since_epoch().count();
    report["format"] = { {"kind", "resident"}, {"magic", "MTLMSHLT"},
        {"version", kResidentFormatVersion}, {"referenceCompression", false} };
    const unsigned int hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    report["internalWorkers"] = std::max(1u, hardwareThreads / 2u);
    report["workerPolicy"] = "default: max(1, hardware_concurrency / 2)";
    TemporarySource temporary;
    temporary.create(options.source, assetRoot);
    require(!pathIsSame(options.report, temporary.source) && !pathIsSame(options.report, temporary.cache),
        "Report must not overwrite temporary source or cache");
    require(fs::file_size(temporary.source) == sourceSize && fs::last_write_time(temporary.source) == sourceTime,
        "Temporary source size/mtime differs from original");
    report["temporarySourceMetadataMatches"] = true;
    report["temporarySource"] = temporary.source.generic_string();
    report["seconds"]["copySource"] = elapsedSeconds(begin);

    Json expected = Json::array();
    {
        Scene fresh;
        auto phase = Clock::now();
        const bool sourceLoaded = fresh.loadDeferredMeshlets(temporary.source, {});
        require(sourceLoaded, "Source load failed: " + fresh.lastLoadResult().error);
        report["seconds"]["loadSource"] = elapsedSeconds(phase);
        require(!fresh.lastLoadResult().meshletCacheLoaded && fresh.hasDeferredMeshlets(),
            "Unique temporary source did not request a fresh resident build");
        require(pathIsSame(fs::weakly_canonical(fresh.lastLoadResult().meshletCachePath), temporary.cache),
            "Scene selected an unexpected cache path");
        phase = Clock::now();
        double buildSeconds = 0.0;
        for (size_t index = 0; index < fresh.renderPrimitives().size(); ++index) {
            const auto primitiveBegin = Clock::now();
            require(fresh.buildDeferredMeshlet(index), "Deferred build failed for primitive " + std::to_string(index));
            const double primitiveSeconds = elapsedSeconds(primitiveBegin);
            buildSeconds += primitiveSeconds;
            expected.push_back(primitiveJson(fresh.renderPrimitives()[index]));
            report["primitiveBuildSeconds"].push_back(primitiveSeconds);
            std::printf("Built primitive %zu / %zu\n", index + 1, fresh.renderPrimitives().size());
            std::fflush(stdout);
        }
        report["seconds"]["buildMeshlets"] = buildSeconds;
        report["seconds"]["buildLoopIncludingSourceHashes"] = elapsedSeconds(phase);
        phase = Clock::now();
        const bool finalized = fresh.finalizeDeferredMeshlets();
        require(finalized && fresh.lastLoadResult().meshletCacheSaved,
            "Resident cache was not saved: " + fresh.lastLoadResult().warning);
        report["seconds"]["saveCache"] = elapsedSeconds(phase);
        report["cacheSaved"] = true;
        report["loadWarning"] = fresh.lastLoadResult().warning;
        report["bounds"] = boundsJson(fresh.bounds());
    }

    // Release the fresh hierarchy before loading the saved resident payload.
    Scene reloaded;
    auto phase = Clock::now();
    const bool cacheLoaded = reloaded.loadDeferredMeshlets(temporary.source, {});
    require(cacheLoaded, "Cache reload failed: " + reloaded.lastLoadResult().error);
    report["seconds"]["reloadCache"] = elapsedSeconds(phase);
    require(reloaded.lastLoadResult().meshletCacheLoaded && !reloaded.hasDeferredMeshlets(),
        "Saved cache failed source hash/topology checks: " + reloaded.lastLoadResult().warning);
    require(reloaded.renderPrimitives().size() == expected.size(), "Reload primitive count differs from source");
    phase = Clock::now();
    uint64_t vertexCount = 0;
    uint64_t triangleCount = 0;
    uint64_t clusterCount = 0;
    for (size_t index = 0; index < expected.size(); ++index) {
        const auto& primitive = reloaded.renderPrimitives()[index];
        require(primitiveJson(primitive) == expected[index],
            "Reload geometry hash/counts/bounds differ for primitive " + std::to_string(index));
        try {
            expected[index]["validatedLod0Triangles"] = validateLod0(primitive);
        } catch (const std::exception& error) {
            throw std::runtime_error("Primitive " + std::to_string(index) + ": " + error.what());
        }
        vertexCount += primitive.vertexCount;
        triangleCount += expected[index]["validatedLod0Triangles"].get<uint64_t>();
        clusterCount += primitive.meshletLodClusters.size();
    }
    validateCacheHeader(temporary.cache, sourceSize,
        static_cast<int64_t>(sourceTime.time_since_epoch().count()));
    require(fs::file_size(options.source) == sourceSize && fs::last_write_time(options.source) == sourceTime,
        "Original source changed during cook");
    report["seconds"]["validateSourceAndLod0"] = elapsedSeconds(phase);
    report["cacheLoaded"] = true;
    report["sourceGeometryHashAndTopologyVerified"] = true;
    report["lod0DirectedTriangleMultisetVerified"] = true;
    report["validation"] = { {"cacheReload", true}, {"sourceGeometryHashMatches", true},
        {"sourceTopologyMatches", true}, {"directedTriangleMatch", true} };
    report["triangleComparison"] = "Exact P/N/UV/T tuples; cyclic rotations allowed; winding and multiplicity preserved";
    report["primitiveCount"] = expected.size();
    report["vertexCount"] = vertexCount;
    report["sourceLod0Triangles"] = triangleCount;
    report["lodClusterCount"] = clusterCount;
    report["primitives"] = std::move(expected);
    report["outputBytes"] = fs::file_size(temporary.cache);
    report["fileBytes"] = report["outputBytes"];
    fs::create_directories(options.output.parent_path());
    const fs::path outputDirectory = fs::canonical(options.output.parent_path());
    require(pathIsSame(outputDirectory, assetRoot) || pathIsInside(outputDirectory, assetRoot),
        "Output directory escaped the repository Asset directory");
    moveWithoutReplacing(temporary.cache, options.output);
    report["status"] = "complete";
    report["seconds"]["total"] = elapsedSeconds(begin);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::puts("MetallicResidentMeshletCook --source ORIGINAL --output STAGED --report JSON\n"
            "Rebuild resident MTLMSHLT v4 caches with default internal workers (hardware / 2).\n"
            "Source and staged output must be inside the repository Asset directory; existing outputs are refused.\n"
            "Verifies source geometry and LOD0 directed triangles after cache reload.\n"
            "Resident caches do not use meshstream Reference compression.");
        return 0;
    }
    Options options;
    Json report = {{"status", "failed"}};
    bool reportSafe = false;
    try {
        options = parseOptions(argc, argv);
        cook(options, report, reportSafe);
        writeReport(options.report, report);
        std::printf("Validated resident cache staged at %s\n", options.output.string().c_str());
        return 0;
    } catch (const std::exception& error) {
        report["status"] = "failed";
        report["error"] = error.what();
        std::fprintf(stderr, "%s\n", error.what());
        if (reportSafe) {
            try { writeReport(options.report, report); }
            catch (const std::exception& reportError) {
                std::fprintf(stderr, "Failure report could not be written: %s\n", reportError.what());
            }
        }
        return 1;
    }
}
