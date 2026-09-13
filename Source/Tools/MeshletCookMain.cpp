#include "Runtime/Scene/MeshletStreamAsset.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <Windows.h>
#include <psapi.h>
#endif

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace metallic::scene;

struct MemorySample {
    uint64_t privateBytes = 0;
    uint64_t workingSetBytes = 0;
    uint64_t peakPrivateBytes = 0;
    uint64_t peakWorkingSetBytes = 0;
};

MemorySample sampleMemory()
{
    MemorySample sample;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))) {
        sample = {counters.PrivateUsage, counters.WorkingSetSize,
            counters.PeakPagefileUsage, counters.PeakWorkingSetSize};
    }
#endif
    return sample;
}

struct MemoryBudget {
#ifdef _WIN32
    HANDLE job = nullptr;
    ~MemoryBudget() { if (job) { CloseHandle(job); } }
#endif
    void apply(uint64_t bytes)
    {
        if (bytes == 0) { return; }
#ifdef _WIN32
        job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        limits.ProcessMemoryLimit = static_cast<SIZE_T>(bytes);
        if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                &limits, sizeof(limits)) || !AssignProcessToJobObject(job, GetCurrentProcess())) {
            throw std::runtime_error("Cannot install process commit budget: Windows error " +
                std::to_string(GetLastError()));
        }
#else
        throw std::runtime_error("--memory-mib currently requires Windows Job Objects");
#endif
    }
};

uint64_t parseNumber(std::string_view text)
{
    uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error("Invalid non-negative integer: " + std::string(text));
    }
    return value;
}

uint64_t alignedPageBytes(uint64_t bytes)
{
    return (bytes + 255u) & ~uint64_t(255u);
}

Json inspectAsset(const MeshletStreamAsset& asset, bool validatePayloads)
{
    Json levels = Json::object();
    uint64_t storedBytes = 0, deviceBytes = 0, clusters = 0, triangles = 0;
    std::vector<uint8_t> scratch;
    for (uint32_t index = 0; index < asset.pageCount(); ++index) {
        const auto& page = asset.pages()[index];
        storedBytes += page.payloadSize;
        deviceBytes += alignedPageBytes(page.uncompressedSize);
        clusters += page.clusterCount;
        triangles += page.triangleIndexCount / 3u;
        Json& level = levels[std::to_string(page.lodLevel)];
        if (level.is_null()) { level = Json::object(); }
        for (const auto& [key, value] : std::initializer_list<std::pair<const char*, uint64_t>>{
                {"pages", 1}, {"clusters", page.clusterCount},
                {"triangles", page.triangleIndexCount / 3u},
                {"storedBytes", page.payloadSize}, {"deviceBytes", alignedPageBytes(page.uncompressedSize)}}) {
            level[key] = level.value(key, uint64_t(0)) + value;
        }
        if (validatePayloads) {
            std::span<const uint8_t> payload;
            std::string reason;
            if (!decodeMeshletStreamPayloadForDevice(page, asset.pagePayload(index), scratch, payload, reason)) {
                throw std::runtime_error("Payload " + std::to_string(index) + ": " + reason);
            }
            if ((index + 1u) % 16384u == 0u || index + 1u == asset.pageCount()) {
                std::printf("Validated payloads %u/%u\n", index + 1u, asset.pageCount());
                std::fflush(stdout);
            }
        }
    }
    Json terminalPages = Json::array();
    uint64_t terminalBytes = 0;
    for (uint32_t group : asset.terminalGroups()) {
        const auto pageId = asset.groups()[group].pageIndex;
        terminalPages.push_back(pageId);
        terminalBytes += alignedPageBytes(asset.pages()[pageId].uncompressedSize);
    }
    uint64_t terminalInstanceGroups = 0, terminalInstanceClusters = 0, stateBytes = 0;
    Json geometries = Json::array();
    std::vector<uint32_t> instanceCounts(asset.primitiveCount(), 0);
    for (const auto& instance : asset.instances()) {
        ++instanceCounts[instance.primitiveIndex];
        const auto roots = asset.primitiveTerminalGroups(instance.primitiveIndex);
        terminalInstanceGroups += roots.size();
        for (uint32_t root : roots) { terminalInstanceClusters += asset.groups()[root].clusterCount; }
        stateBytes += 32u + uint64_t(asset.primitives()[instance.primitiveIndex].groupCount) * 12u;
    }
    for (uint32_t index = 0; index < asset.primitiveCount(); ++index) {
        const auto& primitive = asset.primitives()[index];
        geometries.push_back({{"primitive", index}, {"sourcePrimitive", primitive.renderPrimitiveIndex},
            {"instances", instanceCounts[index]}, {"groups", primitive.groupCount},
            {"levels", primitive.lodLevelCount}, {"terminalGroups", asset.primitiveTerminalGroups(index).size()}});
    }
    const uint64_t directoryBytes = asset.primitives().size_bytes() + asset.instances().size_bytes() +
        asset.geometries().size_bytes() + asset.lodLevels().size_bytes() + asset.groups().size_bytes() +
        asset.refinedGroups().size_bytes() + asset.nodes().size_bytes() + asset.pages().size_bytes() +
        asset.pagePayloadOffsets().size_bytes();
    return {{"status", "complete"}, {"formatVersion", 9}, {"asset", std::filesystem::absolute(asset.path()).string()},
        {"fileBytes", std::filesystem::file_size(asset.path())}, {"primitives", asset.primitiveCount()},
        {"instances", asset.instanceCount()}, {"groups", asset.groupCount()}, {"pages", asset.pageCount()},
        {"clustersAllLods", clusters}, {"trianglesAllLods", triangles}, {"levels", levels},
        {"storedPayloadBytes", storedBytes}, {"allPagesDeviceBytesAligned256", deviceBytes},
        {"maxPagePayloadBytes", asset.maxPagePayloadBytes()}, {"directoryBytes", directoryBytes},
        {"runtimeFrontierStateBytes", stateBytes}, {"terminalPages", terminalPages},
        {"terminalPageCount", terminalPages.size()}, {"terminalPageBytesAligned256", terminalBytes},
        {"minimumResidentBytesIncludingOneStreamPage", terminalBytes +
            (terminalPages.size() < asset.pageCount() ? alignedPageBytes(asset.maxPagePayloadBytes()) : 0)},
        {"terminalInstanceGroups", terminalInstanceGroups}, {"terminalInstanceClusters", terminalInstanceClusters},
        {"payloadValidation", validatePayloads ? "all-pages" : "not-requested"}, {"geometries", geometries}};
}

int run(int argc, char** argv)
{
    MeshletStreamAssetOfflineBuildDesc desc;
    std::filesystem::path manifestPath;
    uint64_t memoryMiB = 0;
    bool inspectOnly = false, validatePayloads = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--inspect") { inspectOnly = true; continue; }
        if (option == "--validate-payloads") { validatePayloads = true; continue; }
        if (option == "--help") {
            std::puts("MetallicMeshletCook --source file.gltf --output file.meshstream.bin\n"
                "  --report file.json --workers N --memory-mib N\n"
                "  --max-geometries N --checkpoint-interval N --compression none|byte-rle\n"
                "  --inspect (skip cooking) --validate-payloads (check every page)\n"
                "Exit 2 means a recoverable geometry-budget pause. Zero budgets preserve library defaults.");
            return 0;
        }
        if (++i >= argc) { throw std::runtime_error("Missing value for " + std::string(option)); }
        const std::string_view value(argv[i]);
        if (option == "--source") { desc.sourcePath = argv[i]; }
        else if (option == "--output") { desc.outputPath = argv[i]; }
        else if (option == "--report") { manifestPath = argv[i]; }
        else if (option == "--compression") {
            if (value == "none") { desc.compressionMode = MeshletStreamPayloadCompression::None; }
            else if (value == "byte-rle") { desc.compressionMode = MeshletStreamPayloadCompression::ByteRle; }
            else { throw std::runtime_error("Unknown compression"); }
        } else if (option == "--memory-mib") { memoryMiB = parseNumber(value); }
        else if (option == "--workers" || option == "--max-geometries" || option == "--checkpoint-interval") {
            const uint64_t number = parseNumber(value);
            if (number > UINT32_MAX) { throw std::runtime_error("Option exceeds uint32 range"); }
            if (option == "--workers") { desc.meshletOptions.maxWorkers = static_cast<uint32_t>(number); }
            else if (option == "--max-geometries") { desc.maxNewGeometriesPerInvocation = static_cast<uint32_t>(number); }
            else { desc.partialCheckpointGeometryInterval = static_cast<uint32_t>(number); }
        } else { throw std::runtime_error("Unknown option " + std::string(option)); }
    }
    if (desc.outputPath.empty() || (!inspectOnly && desc.sourcePath.empty())) {
        throw std::runtime_error("--output and (for cooking) --source are required");
    }
    if (memoryMiB > UINT64_MAX / (1024u * 1024u)) { throw std::runtime_error("Memory budget overflow"); }
    MemoryBudget budget;
    budget.apply(memoryMiB * 1024u * 1024u);
    if (manifestPath.empty()) { manifestPath = desc.outputPath.string() + ".json"; }
    if (manifestPath.has_parent_path()) { std::filesystem::create_directories(manifestPath.parent_path()); }
    std::ofstream events(manifestPath.string() + ".events.jsonl", std::ios::app);
    if (!events) { throw std::runtime_error("Cannot open cook events"); }
    const auto started = Clock::now();
    desc.progress = [&](const MeshletStreamCookProgress& progress) {
        const auto memory = sampleMemory();
        Json event = {{"phase", progress.phase}, {"sourcePrimitive", progress.sourcePrimitiveIndex},
            {"mesh", progress.meshIndex}, {"primitive", progress.primitiveIndex},
            {"completedGeometries", progress.completedGeometries}, {"vertices", progress.vertices},
            {"triangles", progress.triangles}, {"clusters", progress.clusters}, {"groups", progress.groups},
            {"payloadBytes", progress.payloadBytes}, {"decodeSeconds", progress.decodeSeconds},
            {"buildSeconds", progress.buildSeconds}, {"encodeAndCheckpointSeconds", progress.encodeSeconds},
            {"elapsedSeconds", std::chrono::duration<double>(Clock::now() - started).count()},
            {"privateBytes", memory.privateBytes}, {"peakPrivateBytes", memory.peakPrivateBytes},
            {"peakWorkingSetBytes", memory.peakWorkingSetBytes}};
        events << event.dump() << '\n';
        events.flush();
        std::printf("[%s] geometry=%u mesh=%u triangles=%llu elapsed=%.2fs peakCommit=%.1f MiB\n",
            progress.phase, progress.sourcePrimitiveIndex, progress.meshIndex,
            static_cast<unsigned long long>(progress.triangles),
            std::chrono::duration<double>(Clock::now() - started).count(), memory.peakPrivateBytes / 1048576.0);
        std::fflush(stdout);
    };
    MeshletStreamAssetOfflineBuildStats stats;
    desc.stats = &stats;
    std::string reason;
    if (!inspectOnly && !buildMeshletStreamAssetOffline(desc, reason)) {
        std::fprintf(stderr, "%s\n", reason.c_str());
        return reason.find("paused after geometry budget") != std::string::npos ? 2 : 1;
    }
    MeshletStreamAsset asset;
    if (!asset.open(desc.outputPath, reason)) { throw std::runtime_error(reason); }
    if (!desc.sourcePath.empty() && !asset.isCurrentForSource(desc.sourcePath)) {
        throw std::runtime_error("Cooked asset does not match source dependencies");
    }
    Json report = inspectAsset(asset, validatePayloads);
    const auto memory = sampleMemory();
    report["source"] = desc.sourcePath.string();
    report["elapsedSecondsThisInvocation"] = std::chrono::duration<double>(Clock::now() - started).count();
    report["processMemoryBudgetBytes"] = memoryMiB * 1024u * 1024u;
    report["peakProcessCommitBytes"] = memory.peakPrivateBytes;
    report["peakWorkingSetBytes"] = memory.peakWorkingSetBytes;
    report["maxWorkers"] = desc.meshletOptions.maxWorkers;
    report["sourceRangeReadBytesThisInvocation"] = stats.accessorRangeReadBytes;
    report["maxSourceRangeReadBytesThisInvocation"] = stats.maxAccessorRangeReadBytes;
    report["checkpointsThisInvocation"] = stats.partialCheckpointCount;
    const auto temporary = manifestPath.string() + ".tmp";
    std::ofstream output(temporary);
    output << report.dump(2) << '\n';
    output.close();
    if (!output) { throw std::runtime_error("Cannot write manifest"); }
#ifdef _WIN32
    if (!MoveFileExW(std::filesystem::path(temporary).c_str(), manifestPath.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        throw std::runtime_error("Cannot publish manifest");
    }
#else
    std::filesystem::rename(temporary, manifestPath);
#endif
    std::printf("Completed: geometries=%u instances=%u pages=%u; manifest=%s\n",
        asset.geometryCount(), asset.instanceCount(), asset.pageCount(), manifestPath.string().c_str());
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    try { return run(argc, argv); }
    catch (const std::exception& exception) {
        std::fprintf(stderr, "Cook failed: %s. A published partial checkpoint can be resumed.\n", exception.what());
        return 1;
    }
}
