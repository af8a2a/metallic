#include "Runtime/Scene/Scene.h"
#include "Runtime/Task/TaskSystem.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace metallic::scene;
using Clock = std::chrono::steady_clock;

#if defined(METALLIC_MESHLET_BENCH_LEGACY) && METALLIC_MESHLET_BENCH_LEGACY
constexpr const char* kBackend = "legacy-independent-pool";
#else
constexpr const char* kBackend = "shared-task-runtime";
#endif

struct Options {
    std::vector<uint32_t> cells{64, 128};
    std::vector<uint32_t> workers{1, 2, 4, 8};
    uint32_t warmup = 1;
    uint32_t rounds = 5;
    bool attributes = true;
    bool seams = false;
    std::string output;
};

uint32_t parseNumber(std::string_view text)
{
    uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Expected an unsigned integer");
    }
    return value;
}

std::vector<uint32_t> parseList(std::string_view text)
{
    std::vector<uint32_t> values;
    for (;;) {
        const size_t comma = text.find(',');
        values.push_back(parseNumber(text.substr(0, comma)));
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return values;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view name(argv[index]);
        if (index + 1 == argc) { throw std::invalid_argument("Missing option value"); }
        const std::string_view value(argv[++index]);
        if (name == "--cells") { options.cells = parseList(value); }
        else if (name == "--workers") { options.workers = parseList(value); }
        else if (name == "--rounds") { options.rounds = parseNumber(value); }
        else if (name == "--warmup") { options.warmup = parseNumber(value); }
        else if (name == "--attributes" || name == "--seams") {
            const auto flag = parseNumber(value);
            if (flag > 1) { throw std::invalid_argument("Boolean options require 0 or 1"); }
            if (name == "--attributes") { options.attributes = flag != 0; }
            else { options.seams = flag != 0; }
        } else if (name == "--output") { options.output = value; }
        else { throw std::invalid_argument("Unknown option: " + std::string(name)); }
    }
    for (const auto cells : options.cells) {
        if (cells == 0 || cells > 512) { throw std::invalid_argument("Cells must be in [1,512]"); }
    }
    for (const auto workers : options.workers) {
        if (workers == 0 || workers > 64) { throw std::invalid_argument("Workers must be in [1,64]"); }
    }
    if (options.rounds == 0 || options.rounds > 10000 || options.warmup > 1000) {
        throw std::invalid_argument("Rounds must be in [1,10000] and warmup in [0,1000]");
    }
    if (options.seams && !options.attributes) { throw std::invalid_argument("Seams require attributes"); }
    return options;
}

RenderPrimitive makeGrid(uint32_t cells, bool attributes, bool protectedSeams)
{
    RenderPrimitive primitive;
    primitive.name = "CPUJobGrid";
    for (uint32_t y = 0; y <= cells; ++y) {
        for (uint32_t x = 0; x <= cells; ++x) {
            const float u = float(x) / cells;
            const float v = float(y) / cells;
            primitive.positions.emplace_back(u, v, .01f * std::sin(u * 6.f) * std::cos(v * 6.f));
            if (attributes) {
                primitive.normals.emplace_back(0.f, 0.f, 1.f);
                primitive.texcoords0.emplace_back(u, v);
                primitive.tangents.emplace_back(1.f, 0.f, 0.f, 1.f);
            }
        }
    }
    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
            const uint32_t a = y * (cells + 1) + x;
            const uint32_t b = a + 1, c = a + cells + 1, d = c + 1;
            primitive.indices.insert(primitive.indices.end(), {a, b, c, b, d, c});
        }
    }
    if (protectedSeams) {
        RenderPrimitive soup;
        soup.name = primitive.name;
        for (size_t index = 0; index < primitive.indices.size(); ++index) {
            const uint32_t vertex = primitive.indices[index];
            soup.indices.push_back(static_cast<uint32_t>(soup.positions.size()));
            soup.positions.push_back(primitive.positions[vertex]);
            soup.normals.push_back(primitive.normals[vertex]);
            soup.texcoords0.push_back(primitive.texcoords0[vertex]);
            soup.tangents.push_back(primitive.tangents[vertex]);
            const uint32_t chart = static_cast<uint32_t>(index / 3u) % 4u;
            soup.texcoords0.back().x += float(chart);
            soup.tangents.back().w = (chart & 1u) ? -1.f : 1.f;
        }
        primitive = std::move(soup);
    }
    primitive.vertexCount = primitive.positions.size();
    primitive.indexCount = primitive.indices.size();
    primitive.triangleCount = primitive.indices.size() / 3;
    primitive.hasAuthoredNormals = attributes;
    primitive.hasAuthoredTangents = attributes;
    for (const auto& position : primitive.positions) { primitive.localBounds.include(position); }
    return primitive;
}

// FNV-1a over a versioned logical-field stream. Integers are encoded low byte
// first, floats as their exact IEEE bits; no structure padding or pointer bytes.
class GeometryHash {
public:
    void add(uint8_t value)
    {
        hash_ ^= value;
        hash_ *= 1099511628211ull;
    }

    void add(bool value) { add(uint8_t(value)); }
    void add(int8_t value) { add(static_cast<uint8_t>(value)); }
    void add(int32_t value) { add(static_cast<uint32_t>(value)); }

    void add(uint32_t value)
    {
        for (unsigned shift = 0; shift < 32; shift += 8) { add(static_cast<uint8_t>(value >> shift)); }
    }

    void add(uint64_t value)
    {
        for (unsigned shift = 0; shift < 64; shift += 8) { add(static_cast<uint8_t>(value >> shift)); }
    }

    void add(float value) { add(std::bit_cast<uint32_t>(value)); }
    void add(const float2& value) { add(value.x); add(value.y); }
    void add(const float3& value) { add(value.x); add(value.y); add(value.z); }
    void add(const float4& value) { add(value.x); add(value.y); add(value.z); add(value.w); }

    void add(const std::string& value)
    {
        add(uint64_t(value.size()));
        for (const unsigned char byte : value) { add(uint8_t(byte)); }
    }

    void add(const Bounds& value)
    {
        add(value.min); add(value.max); add(value.valid);
    }

    void add(const MeshletCluster& value)
    {
        add(value.vertexOffset); add(value.vertexCount);
        add(value.triangleOffset); add(value.triangleCount);
        add(value.lodLevel); add(value.lodGroupChildIndex);
        add(value.lodGroupIndex); add(value.refinedGroupIndex); add(value.lodError);
        add(value.bounds); add(value.boundingSphereCenter); add(value.boundingSphereRadius);
        add(value.coneApex); add(value.coneAxis); add(value.coneCutoff);
        for (const auto component : value.packedCone) { add(component); }
    }

    void add(const MeshletLODGroup& value)
    {
        add(value.clusterOffset); add(value.clusterCount); add(value.lodLevel);
        add(value.bounds); add(value.boundingSphereCenter);
        add(value.boundingSphereRadius); add(value.maxQuadricError);
    }

    void add(const MeshletLODLevel& value)
    {
        add(value.groupOffset); add(value.groupCount); add(value.clusterOffset); add(value.clusterCount);
        add(value.minBoundingSphereRadius); add(value.minMaxQuadricError);
    }

    template <typename T>
    void add(const std::vector<T>& values)
    {
        add(uint64_t(values.size()));
        for (const auto& value : values) { add(value); }
    }

    void add(const RenderPrimitive& value)
    {
        add(std::string("MetallicMeshletGeometry-v1"));
        add(value.name); add(value.meshIndex); add(value.primitiveIndex); add(value.materialIndex); add(value.mode);
        add(value.vertexCount); add(value.indexCount); add(value.triangleCount); add(value.localBounds);
        add(value.positions); add(value.normals); add(value.tangents); add(value.texcoords0);
        add(value.hasAuthoredNormals); add(value.hasAuthoredTangents); add(value.indices);
        add(value.meshletClusters); add(value.meshletVertices); add(value.meshletTriangles);
        add(value.meshletLodLevels); add(value.meshletLodGroups); add(value.meshletLodClusters);
        add(value.meshletLodVertices); add(value.meshletLodTriangles);
    }

    uint64_t value() const { return hash_; }

private:
    uint64_t hash_ = 14695981039346656037ull;
};

std::string hexHash(uint64_t value)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

struct Sample {
    uint64_t hash = 0;
    uint64_t checksum = 0;
    uint64_t wallNanoseconds = 0;
    uint64_t levels = 0, groups = 0, clusters = 0;
    uint64_t lodVertices = 0, lodTriangleBytes = 0;
};

Sample measure(const RenderPrimitive& source, uint32_t workers)
{
    // Input copying, result hashing and destruction are outside the timed build.
    // Every submission owns its primitive and worker scratch/results exclusively.
    RenderPrimitive result = source;
    const auto begin = Clock::now();
    const bool succeeded = buildStreamMeshletsForPrimitive(result, {.maxWorkers = workers});
    const auto end = Clock::now();
    if (!succeeded) { throw std::runtime_error("buildStreamMeshletsForPrimitive failed"); }
    GeometryHash hash;
    hash.add(result);
    uint64_t checksum = std::accumulate(result.meshletLodVertices.begin(), result.meshletLodVertices.end(), uint64_t{0});
    checksum = std::accumulate(result.meshletLodTriangles.begin(), result.meshletLodTriangles.end(), checksum);
    return {
        .hash = hash.value(), .checksum = checksum,
        .wallNanoseconds = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()),
        .levels = result.meshletLodLevels.size(), .groups = result.meshletLodGroups.size(),
        .clusters = result.meshletLodClusters.size(), .lodVertices = result.meshletLodVertices.size(),
        .lodTriangleBytes = result.meshletLodTriangles.size(),
    };
}

void emitSample(std::ostream& output, const Options& options, uint32_t cells, uint32_t workers,
    uint32_t round, const char* phase, const Sample& sample, uint64_t reference)
{
    const uint32_t hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    output << "{\"type\":\"sample\",\"backend\":\"" << kBackend
        << "\",\"workload\":\"generated-grid-full-meshlet-lod-build\",\"cells\":" << cells
        << ",\"attributes\":" << (options.attributes ? "true" : "false")
        << ",\"seams\":" << (options.seams ? "true" : "false")
        << ",\"requestedWorkers\":" << workers << ",\"effectiveWorkerLimit\":" << std::min(workers, hardwareThreads)
        << ",\"phase\":\"" << phase << "\",\"round\":" << round
        << ",\"wallNs\":" << sample.wallNanoseconds
        << ",\"geometryHash\":\"" << hexHash(sample.hash)
        << "\",\"serialReferenceHash\":\"" << hexHash(reference)
        << "\",\"checksum\":\"" << sample.checksum << "\",\"correct\":" << (sample.hash == reference ? "true" : "false")
        << ",\"lodLevels\":" << sample.levels << ",\"lodGroups\":" << sample.groups
        << ",\"lodClusters\":" << sample.clusters << ",\"lodVertices\":" << sample.lodVertices
        << ",\"lodTriangleBytes\":" << sample.lodTriangleBytes << "}\n";
    output.flush();
    if (sample.hash != reference) { throw std::runtime_error("Parallel geometry differs from the serial reference"); }
}

class TaskSystemScope {
public:
    explicit TaskSystemScope(uint32_t workers)
    {
        const auto initialized = metallic::task::initializeTaskSystem({.workerCount = workers});
        if (!initialized) { throw std::runtime_error(initialized.error().message); }
    }

    ~TaskSystemScope() { metallic::task::shutdownTaskSystem(); }
};

void run(const Options& options, std::ostream& output)
{
    const uint32_t runtimeWorkers = *std::max_element(options.workers.begin(), options.workers.end());
    TaskSystemScope runtime(runtimeWorkers);
    output << "{\"type\":\"metadata\",\"schema\":1,\"backend\":\"" << kBackend
        << "\",\"hardwareThreads\":" << std::thread::hardware_concurrency()
        << ",\"runtimeWorkers\":" << runtimeWorkers << ",\"warmupRounds\":" << options.warmup
        << ",\"measuredRounds\":" << options.rounds
        << ",\"hashAlgorithm\":\"fnv1a64-logical-fields-v1\",\"timedScope\":\"buildStreamMeshletsForPrimitive only\""
        << ",\"inputCopyTimed\":false,\"hashTimed\":false,\"vulkanUsed\":false"
        << ",\"note\":\"Runtime initialization and generated source creation are outside timing. Serial references run first; "
           "later samples reuse runtime, allocator and OS caches. Worker order rotates and reverses each round.\"}\n";

    std::vector<RenderPrimitive> sources;
    std::vector<uint64_t> references;
    for (const uint32_t cells : options.cells) {
        sources.push_back(makeGrid(cells, options.attributes, options.seams));
        const auto reference = measure(sources.back(), 1);
        references.push_back(reference.hash);
        emitSample(output, options, cells, 1, 0, "serial-reference-first-build", reference, reference.hash);
    }

    for (uint32_t round = 0; round < options.warmup + options.rounds; ++round) {
        auto order = options.workers;
        std::rotate(order.begin(), order.begin() + round % order.size(), order.end());
        if (round % 2 != 0) { std::reverse(order.begin(), order.end()); }
        for (const uint32_t workers : order) {
            for (size_t scenario = 0; scenario < sources.size(); ++scenario) {
                const auto sample = measure(sources[scenario], workers);
                const bool warmup = round < options.warmup;
                emitSample(output, options, options.cells[scenario], workers,
                    warmup ? round : round - options.warmup, warmup ? "warmup" : "steady",
                    sample, references[scenario]);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "MeshletSceneBench [--cells 64,128] [--workers 1,2,4,8] [--rounds 5] [--warmup 1] "
                     "[--attributes 0|1] [--seams 0|1] [--output samples.jsonl]\n";
        return 0;
    }
    try {
        const auto options = parseOptions(argc, argv);
        if (options.output.empty()) { run(options, std::cout); }
        else {
            std::ofstream output(options.output, std::ios::binary);
            if (!output) { throw std::runtime_error("Cannot create output JSONL file"); }
            output.exceptions(std::ios::failbit | std::ios::badbit);
            run(options, output);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MeshletSceneBench failed: " << error.what() << '\n';
        return 1;
    }
}
