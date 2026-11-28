#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/Profiling/SchedulingDiagnostics.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
using Page = std::array<uint32_t, 4>;
constexpr uint32_t kWidth = 129, kHeight = 97, kPixels = kWidth * kHeight;

void requirePool(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T> void requirePool(const Result<T>& result, const std::string& message)
{
    requirePool(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct PoolCommands
{
    RenderFrameContext frame;
    QueueSubmissionTracker tracker;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    uint64_t index = 0;

    explicit PoolCommands(Device& device)
    {
        auto& queue = *device.getQueue(QueueType::Graphics);
        requirePool(tracker.initialize(device, queue), "Pool submission tracker");
        requirePool(device.createCommandPool(queue).transform([&](auto p) { pool = std::move(p); }), "Pool command pool");
        requirePool(pool->createCommandBuffer().transform([&](auto c) { commands = std::move(c); }), "Pool commands");
    }
    ~PoolCommands()
    {
        frame.cancel();
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        (void)pool->reset();
        (void)frame.reset();
    }
    void begin()
    {
        requirePool(frame.begin(index++), "Pool frame begin");
        requirePool(pool->reset(), "Pool reset");
        requirePool(commands->begin(frame.submissionContext()), "Pool command begin");
    }
    void finish(bool cancel = false)
    {
        requirePool(commands->end(), "Pool command end");
        if (cancel) { frame.cancel(); return; }
        CommandBuffer* list[]{commands.get()};
        requirePool(tracker.submit({.commandBuffers = list}, frame), "Pool submit");
        requirePool(frame.wait(), "Pool wait");
    }
};

std::filesystem::path writePoolFixture(const std::filesystem::path& directory, bool splitSources = false)
{
    std::filesystem::create_directories(directory);
    const float positions[]{-4,-4,0, 4,-4,0, 4,4,0, -4,4,0,
        -.75f,-.75f,1, .75f,-.75f,1, .75f,.75f,1, -.75f,.75f,1};
    const float normals[]{0,0,1, 0,0,1, 0,0,1, 0,0,1};
    const uint32_t indices[]{0,1,2, 0,2,3};
    {
        std::ofstream binary(directory / "Geometry.bin", std::ios::binary);
        binary.write(reinterpret_cast<const char*>(positions), sizeof(positions));
        binary.write(reinterpret_cast<const char*>(normals), sizeof(normals));
        binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
        requirePool(binary.good(), "Write pool geometry");
    }
    auto doc = Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],
      "nodes":[{"mesh":0},{"mesh":1}],"buffers":[{"uri":"Geometry.bin","byteLength":168}],
      "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":96},
        {"buffer":0,"byteOffset":96,"byteLength":48},{"buffer":0,"byteOffset":144,"byteLength":24}],
      "accessors":[{"bufferView":0,"byteOffset":0,"componentType":5126,"count":4,"type":"VEC3","min":[-4,-4,0],"max":[4,4,0]},
        {"bufferView":0,"byteOffset":48,"componentType":5126,"count":4,"type":"VEC3","min":[-0.75,-0.75,1],"max":[0.75,0.75,1]},
        {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},
        {"bufferView":2,"componentType":5125,"count":6,"type":"SCALAR"}],
      "materials":[{"doubleSided":true}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":2},"indices":3,"material":0}]},
        {"primitives":[{"attributes":{"POSITION":1,"NORMAL":2},"indices":3,"material":0}]}]})");
    if (splitSources) {
        // Toggling the blocker mount changes the visible caster list while
        // preserving scene identity and the floor's geometry.
        doc["scenes"][0]["nodes"] = Json::array({0});
        std::ofstream floor(directory / "Floor.gltf"); floor << doc.dump(2);
        requirePool(floor.good(), "Write caster-change floor");
        doc["nodes"][0]["mesh"] = 1;
        std::ofstream blocker(directory / "Blocker.gltf"); blocker << doc.dump(2);
        requirePool(blocker.good(), "Write caster-change blocker");
        const auto path = directory / "Sources.metallic_scene.json";
        const Json manifest{{"version", 4}, {"sources", {
            {{"id", "floor"}, {"path", "Floor.gltf"}},
            {{"id", "blocker"}, {"path", "Blocker.gltf"}},
        }}};
        std::ofstream file(path); file << manifest.dump(2);
        requirePool(file.good(), "Write caster-change composition");
        return path;
    }
    const auto path = directory / "Pool.gltf";
    std::ofstream file(path); file << doc.dump(2);
    requirePool(file.good(), "Write pool fixture");
    return path;
}

struct PoolFace
{
    uint32_t identity;
    VirtualShadowMapLight light;
    uint32_t sourceSlot = UINT32_MAX;
};

enum class HybridTriangleMix { None, LongEdge, Masked };

std::filesystem::path writeClusterFixture(const std::filesystem::path& directory, bool withRoof,
    bool withHierarchy = false, HybridTriangleMix mix = HybridTriangleMix::None)
{
    std::filesystem::create_directories(directory);
    std::vector<std::array<float, 3>> positions;
    // Each grid is exactly one resident 64-triangle group. The small blocker
    // crosses two virtual page boundaries over the hardware-rasterized floor.
    const auto grid = [&](float extent, float height) {
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 8; ++x) {
                const float x0 = extent * (float(x) / 4.0f - 1.0f);
                const float x1 = x0 + extent / 4.0f;
                const float y0 = extent * (float(y) / 2.0f - 1.0f);
                const float y1 = y0 + extent / 2.0f;
                const std::array<float, 3> a{x0, y0, height}, b{x1, y0, height}, c{x1, y1, height}, d{x0, y1, height};
                if (height > 0 && x >= 4) {
                    positions.insert(positions.end(), {a,c,b, a,d,c});
                } else {
                    positions.insert(positions.end(), {a,b,c, a,c,d});
                }
            }
        }
    };
    grid(4, 0);
    const size_t floorVertices = positions.size();
    grid(.1875f, 1);
    constexpr std::array<std::array<float, 3>, 3> maskFootprint{{{-.14f, -.14f, 1}, {-.04f, -.14f, 1}, {-.14f, -.04f, 1}}};
    if (mix == HybridTriangleMix::LongEdge) {
        // The unclipped edge spans 76 finest texels. Its central-page fragment
        // shares a ticket with the original microtriangles.
        positions[floorVertices] = {-2.25f, -.1875f, 1};
        positions[floorVertices + 1] = {.125f, -.1875f, 1};
        positions[floorVertices + 2] = {.125f, -.09375f, 1};
    } else if (mix == HybridTriangleMix::Masked) {
        // Leave one slot for MASK. Identical XY bounds put it and the opaque
        // triangle on the same Build wave iteration and page.
        positions.resize(positions.size() - 3);
        std::ranges::copy(maskFootprint, positions.begin() + floorVertices);
        std::ranges::copy(maskFootprint, positions.end() - 3);
    }
    const size_t blockerVertices = positions.size() - floorVertices;
    if (mix == HybridTriangleMix::Masked) {
        for (const auto& position : maskFootprint) { positions.push_back({position[0], position[1], 2}); }
    }
    const auto bytes = positions.size() * sizeof(positions.front());
    std::ofstream binary(directory / "Clusters.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(positions.data()), bytes);
    requirePool(binary.good(), "Write hybrid cluster geometry");
    auto model = Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],
      "nodes":[{"mesh":0},{"mesh":1}],"buffers":[{"uri":"Clusters.bin"}],"bufferViews":[{"buffer":0}],
      "accessors":[{"bufferView":0,"componentType":5126,"type":"VEC3","min":[-4,-4,0],"max":[4,4,0]},
        {"bufferView":0,"componentType":5126,"type":"VEC3","min":[-0.1875,-0.1875,1],"max":[0.1875,0.1875,1]}],
      "materials":[{"doubleSided":true},{"doubleSided":false}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]},
        {"primitives":[{"attributes":{"POSITION":1},"material":1}]}]})");
    if (withRoof) {
        // A separate 64-triangle cluster at z=2 completely covers the smaller blocker at z=1.
        const float scale = withHierarchy ? 0.375f : 0.2f;
        model["nodes"].push_back({{"mesh", 0}, {"scale", {scale, scale, 1.0f}}, {"translation", {0, 0, 2}}});
        model["scenes"][0]["nodes"].push_back(2);
    }
    if (withHierarchy) {
        // The roof covers groups of adjacent instances and a leaf whose
        // sibling is visible at x = -2.25. Neither case can rely on page tickets.
        model["nodes"][1] = {{"mesh", 1}, {"scale", {0.375f, 0.375f, 1.0f}}, {"translation", {-0.75f, 0, 0}}};
        for (float x : {-0.375f, -0.25f, 0.25f, 0.375f, -2.25f}) {
            model["scenes"][0]["nodes"].push_back(model["nodes"].size());
            model["nodes"].push_back({{"mesh", 1}, {"scale", {0.375f, 0.375f, 1.0f}}, {"translation", {x, 0, 0}}});
        }
    }
    model["accessors"][0]["count"] = floorVertices;
    model["accessors"][1]["count"] = blockerVertices;
    model["accessors"][1]["byteOffset"] = floorVertices * sizeof(positions.front());
    if (mix == HybridTriangleMix::LongEdge) {
        model["accessors"][1]["min"][0] = -2.25f;
    } else if (mix == HybridTriangleMix::Masked) {
        model["accessors"].push_back({{"bufferView", 0}, {"componentType", 5126}, {"type", "VEC3"}, {"count", 3},
            {"byteOffset", (floorVertices + blockerVertices) * sizeof(positions.front())},
            {"min", {-.14f, -.14f, 2}}, {"max", {-.04f, -.04f, 2}}});
        model["materials"].push_back({{"doubleSided", true}, {"alphaMode", "MASK"}, {"alphaCutoff", .5f},
            {"pbrMetallicRoughness", {{"baseColorFactor", {1, 1, 1, 0}}}}});
        model["meshes"][1]["primitives"].push_back({{"attributes", {{"POSITION", 2}}}, {"material", 2}});
    }
    model["buffers"][0]["byteLength"] = bytes;
    model["bufferViews"][0]["byteLength"] = bytes;
    const auto path = directory / "Clusters.gltf";
    std::ofstream file(path);
    file << model.dump(2);
    requirePool(file.good(), "Write cluster fixture");
    return path;
}

struct TwoPassTickets
{
    uint32_t deferred = 0;
    uint32_t rejected = 0;
    uint32_t recovered = 0;
};

struct HybridLaneRouting
{
    uint32_t mixedTickets = 0;
    uint32_t softwareTriangles = 0;
    uint32_t hardwareTriangles = 0;
};

struct HierarchyHZBCounts
{
    uint32_t deferredInstances = 0, deferredNodes = 0;
    uint32_t rejectedInstances = 0, rejectedNodes = 0;
    uint32_t recoveredInstances = 0, recoveredNodes = 0;
};

struct PoolFaceSnapshot
{
    std::array<uint32_t, 32> stats{};
    std::array<uint32_t, kVirtualShadowMapCacheDiagnosticCount> cache{};
    std::vector<Page> pages;
    std::vector<uint64_t> requestedMasks;
    std::vector<uint8_t> visibility;
    std::array<uint32_t, 3> hybrid{};
    HybridLaneRouting hybridRouting;
    std::array<uint32_t, 2> activeViews{}; // Fine level mask and compact initial Build row count.
    TwoPassTickets hzb;
    HierarchyHZBCounts hierarchyHZB;
    uint64_t instanceClusterTickets = 0;
    uint64_t hardwarePrimitives = 0;
    uint64_t hardwareFineGroups = 0;
    uint64_t coarseGroups = 0;
};

struct PoolSnapshot
{
    std::map<uint32_t, PoolFaceSnapshot> faces;
    std::vector<uint32_t> atlas;
    uint32_t atlasWidth = 0, atlasHeight = 0, atlasLayers = 0, pagesX = 0, pageSize = 0;
    VirtualShadowMapMemoryUsage memory;
    std::map<std::string, uint64_t> commands;
    profiling::SchedulingMetrics nativeCommands;
};

std::map<std::string, uint64_t> poolRecordingCommands(const CPUProfileRecorder& recorder)
{
    std::map<std::string, uint64_t> counts;
    for (const auto& section : recorder.sections) {
        if (section.name != "Record VSM build and raster batches" && section.name != "Record VSM build and raster" &&
            section.name != "Acquire shared VSM frame buffers" && section.name != "Prepare VSM caster layers") { continue; }
        for (const auto& work : section.cpuWork) { counts[work.name] += work.value; }
    }
    for (const char* name : {"faces", "computeDispatches", "prepareDispatches", "meshDraws", "renderingPasses", "computeBins", "rasterBins"}) {
        requirePool(counts.contains(name), std::string("Missing actual VSM recording counter: ") + name);
    }
    return counts;
}

void requireUniqueOwners(const PoolSnapshot& snapshot, uint32_t budget, bool allowWorkOverflow = false)
{
    std::set<uint32_t> owners;
    uint32_t grants = 0;
    for (const auto& [identity, face] : snapshot.faces) {
        uint32_t resident = 0;
        for (const auto& page : face.pages) {
            if ((page[1] & 2u) == 0) { continue; }
            requirePool(page[0] < budget, "Admitted page has no physical slot for face " + std::to_string(identity));
            requirePool(owners.insert(page[0]).second, "Two virtual pages own the same shared physical slot");
            ++resident;
        }
        requirePool((face.pages.back()[1] & 3u) == 3u, "Missing complete coarse coverage");
        requirePool(resident == face.stats[1] + 1u && face.stats[12] == resident, "GPU grant differs from resident pages");
        requirePool(face.stats[0] == face.stats[1] + face.stats[2], "Requested/admitted/rejected accounting differs");
        requirePool((allowWorkOverflow || face.stats[5] == 0u) && face.stats[9] == 0u,
            "Shared raster overflow or uninitialized coarse query");
        grants += face.stats[12];
    }
    requirePool(grants <= budget, "Shared allocations exceed the physical budget");
}

// Compare retained D32 texels, so adding a cold face cannot silently clear a
// clean atlas tile even if a separate projection path happens to hide it.
uint32_t requireRetainedDepth(const PoolSnapshot& previous, const PoolSnapshot& current, uint32_t identity)
{
    const auto& before = previous.faces.at(identity).pages;
    const auto& after = current.faces.at(identity).pages;
    uint32_t retained = 0;
    for (size_t page = 0; page < before.size(); ++page) {
        if ((before[page][1] & after[page][1] & 2u) == 0 || before[page][0] != after[page][0]) { continue; }
        const uint32_t slot = before[page][0];
        const uint32_t layer = slot / 4096u, tile = slot % 4096u;
        for (uint32_t y = 0; y < current.pageSize; ++y) {
            const uint64_t offset = uint64_t(layer) * current.atlasWidth * current.atlasHeight +
                uint64_t((tile / current.pagesX) * current.pageSize + y) * current.atlasWidth +
                (tile % current.pagesX) * current.pageSize;
            requirePool(std::equal(previous.atlas.begin() + offset, previous.atlas.begin() + offset + current.pageSize,
                current.atlas.begin() + offset), "Unchanged face lost retained physical depth");
        }
        ++retained;
    }
    requirePool(retained > 1u, "Retention case did not exercise fine physical pages");
    return retained;
}

uint32_t poolPageTexel(const PoolSnapshot& snapshot, uint32_t slot, uint32_t x, uint32_t y)
{
    const uint32_t tile = slot % 4096u;
    return snapshot.atlas[(uint64_t(slot / 4096u) * snapshot.atlasHeight + (tile / snapshot.pagesX) * snapshot.pageSize + y)
        * snapshot.atlasWidth + (tile % snapshot.pagesX) * snapshot.pageSize + x];
}

float compareRasterDepth(const PoolSnapshot& actual, const PoolSnapshot& hardware, uint32_t identity,
    float depthTolerance = 0.0f, bool expectMixedFinePage = true, bool expectMixedVisibility = true,
    bool expectCoarseDepth = false)
{
    const auto& face = actual.faces.at(identity);
    const auto& reference = hardware.faces.at(identity);
    requirePool(face.visibility == reference.visibility, "Projected visibility differs from the reference");
    uint32_t mixedFinePages = 0;
    float maxError = 0.0f;
    for (size_t page = 0; page < face.pages.size(); ++page) {
        requirePool((face.pages[page][1] & 2u) == (reference.pages[page][1] & 2u), "Raster paths disagree on page residency");
        if ((face.pages[page][1] & 2u) == 0) { continue; }
        bool floor = false, blocker = false;
        for (uint32_t y = 0; y < actual.pageSize; ++y) {
            for (uint32_t x = 0; x < actual.pageSize; ++x) {
                const uint32_t depth = poolPageTexel(actual, face.pages[page][0], x, y);
                const uint32_t expected = poolPageTexel(hardware, reference.pages[page][0], x, y);
                const float z = std::bit_cast<float>(depth);
                const float difference = std::abs(z - std::bit_cast<float>(expected));
                const bool sameDepth = depthTolerance == 0 ? depth == expected :
                    ((depth == 0u) == (expected == 0u) && difference <= depthTolerance);
                if (!sameDepth) {
                    throw std::runtime_error("D32 mismatch on face " + std::to_string(identity) +
                        " page " + std::to_string(page) + " texel (" + std::to_string(x) + ", " + std::to_string(y) +
                        "): " + std::to_string(depth) + " vs hardware " + std::to_string(expected));
                }
                maxError = std::max(maxError, difference);
                floor |= z > 0.0f && z < 0.5f;
                blocker |= z > 0.5f;
            }
        }
        mixedFinePages += floor && blocker && page + 1 < face.pages.size();
        if (expectCoarseDepth && page + 1 == face.pages.size()) {
            requirePool(floor || blocker, "Coarse-only raster did not produce visible depth");
        }
    }
    if (expectMixedFinePage && depthTolerance == 0) {
        requirePool(mixedFinePages > 0, "No fine page contains both the large floor and the microtriangle blocker");
    }
    if (expectMixedVisibility) {
        requirePool(std::ranges::find(face.visibility, uint8_t(0)) != face.visibility.end() &&
            std::ranges::find(face.visibility, uint8_t(255)) != face.visibility.end(), "Missing blocked or lit receivers");
    }
    return maxError;
}

enum class PoolScenario { Allocation, HybridRaster, TwoPassHZB, HierarchyHZB, UnifiedMark, CasterChanges,
    ActiveViews, DistantLights, LightCacheRecovery, PackedCoarseRaster };

class VirtualShadowMapPhysicalPoolTest : public RHITest
{
public:
    explicit VirtualShadowMapPhysicalPoolTest(PoolScenario scenario = PoolScenario::Allocation) : scenario_(scenario)
    {
        name = scenario == PoolScenario::UnifiedMark ? "virtual_shadow_map_unified_mark" :
            scenario == PoolScenario::HybridRaster ? "virtual_shadow_map_cluster_hybrid_raster" :
            scenario == PoolScenario::TwoPassHZB ? "virtual_shadow_map_two_pass_hzb" :
            scenario == PoolScenario::HierarchyHZB ? "virtual_shadow_map_hierarchy_hzb" :
            scenario == PoolScenario::ActiveViews ? "virtual_shadow_map_active_views" :
            scenario == PoolScenario::DistantLights ? "virtual_shadow_map_distant_lights" :
            scenario == PoolScenario::LightCacheRecovery ? "virtual_shadow_map_light_cache_recovery" :
            scenario == PoolScenario::PackedCoarseRaster ? "virtual_shadow_map_packed_coarse_raster" :
            scenario == PoolScenario::CasterChanges ? "virtual_shadow_map_shared_caster_changes" : "virtual_shadow_map_physical_pool";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        const bool hybrid = scenario_ == PoolScenario::HybridRaster;
        const bool twoPass = scenario_ == PoolScenario::TwoPassHZB;
        const bool hierarchyHZB = scenario_ == PoolScenario::HierarchyHZB;
        const bool unifiedMark = scenario_ == PoolScenario::UnifiedMark;
        const bool casterChanges = scenario_ == PoolScenario::CasterChanges;
        const bool activeViews = scenario_ == PoolScenario::ActiveViews;
        const bool distantLights = scenario_ == PoolScenario::DistantLights;
        const bool lightCacheRecovery = scenario_ == PoolScenario::LightCacheRecovery;
        const bool packedCoarse = scenario_ == PoolScenario::PackedCoarseRaster;
        const auto directory = std::filesystem::absolute(context.outputDirectory / (unifiedMark ? "VirtualShadowMapUnifiedMark" :
            hierarchyHZB ? "VirtualShadowMapHierarchyHZB" : twoPass ? "VirtualShadowMapTwoPassHZB" :
            hybrid ? "VirtualShadowMapClusterHybridRaster" : casterChanges ? "VirtualShadowMapSharedCasterChanges" :
            activeViews ? "VirtualShadowMapActiveViews" :
            distantLights ? "VirtualShadowMapDistantLights" :
            lightCacheRecovery ? "VirtualShadowMapLightCacheRecovery" :
            packedCoarse ? "VirtualShadowMapPackedCoarseRaster" :
            "VirtualShadowMapPhysicalPool"));
        Json report{{"passed", false}, {"frames", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&] { std::filesystem::create_directories(directory); std::ofstream(directory / "Report.json") << report.dump(2); };
        try {
            requirePool(VirtualShadowMapSettings{}.batchBuildRaster, "Shared Build/Raster batching must default on");
            scene::SceneDocument scene;
            requirePool(scene.load((hybrid || twoPass || hierarchyHZB || packedCoarse) ?
                writeClusterFixture(directory, twoPass || hierarchyHZB, hierarchyHZB,
                    packedCoarse ? HybridTriangleMix::Masked : HybridTriangleMix::None) : writePoolFixture(directory, casterChanges)),
                "Load pool scene");
            auto created = createDevice({.applicationName = "VSM shared physical pool",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless"); }
            requirePool(created, "Pool device");
            auto& device = **created;
            if (hybrid && (!device.capabilities().computeSubgroupBallotArithmetic || device.capabilities().subPixelPrecisionBits > 8)) {
                return RHITestResult::skip("Hybrid raster requires subgroup ballot/arithmetic and at most 8 subpixel bits");
            }
            if ((twoPass || hierarchyHZB) && !device.capabilities().computeSubgroupBallotArithmetic) {
                return RHITestResult::skip("Two-pass HZB requires subgroup ballot/arithmetic for packed cluster Build");
            }
            if (activeViews && !device.capabilities().computeSubgroupBallotArithmetic) {
                return RHITestResult::skip("Fine view compaction requires subgroup ballot/arithmetic");
            }
            if (packedCoarse && !device.capabilities().computeSubgroupBallotArithmetic) {
                return RHITestResult::skip("Coarse raster packing requires subgroup ballot/arithmetic");
            }
            PoolCommands gpu(device);
            RenderView camera;
            requirePool(camera.setCamera({.eye = {0,0,4}, .center = {0,0,0}, .nearPlane = .01f,
                .farPlane = 10, .orthoHeight = 4, .orthographic = true}), "Pool camera");
            auto depth = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::D32Sfloat, .width = kWidth, .height = kHeight});
            requirePool(depth, "Pool depth");
            auto depthView = device.createTextureView(**depth, {.format = Format::D32Sfloat});
            requirePool(depthView, "Pool depth view");
            auto upload = device.createBuffer({.size = kPixels * sizeof(float), .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload});
            requirePool(upload, "Pool depth upload");
            auto* values = static_cast<float*>((*upload)->map());
            requirePool(values != nullptr, "Map pool depth");
            std::fill(values, values + kPixels, (10.f - 4.f) / (10.f - .01f));
            (*upload)->flush(); (*upload)->unmap();
            gpu.begin();
            const TextureBarrierDesc depthBegin{.texture = depth->get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
            requirePool(gpu.commands->synchronize({.textures = {&depthBegin, 1}}), "Pool depth transition");
            requirePool((*upload)->slice().and_then([&](const auto& slice) {
                return gpu.commands->copyBufferToTexture({.texture = depth->get(), .buffer = slice, .width = kWidth, .height = kHeight});
            }), "Pool depth copy");
            const TextureBarrierDesc depthEnd{.texture = depth->get(), .oldLayout = TextureLayout::TransferDestination,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
            requirePool(gpu.commands->synchronize({.textures = {&depthEnd, 1}}), "Pool depth ready");
            gpu.finish();

            const uint32_t variants = unifiedMark || hierarchyHZB || packedCoarse ? 1u : twoPass ? 4u : 2u;
            for (uint32_t variant = 0; variant < variants; ++variant) {
                const bool hybridHZB = twoPass && variant == 1u;
                const bool standaloneHZB = twoPass && variant == 2u;
                const bool serialHZB = twoPass && variant == 3u;
                const bool layered = unifiedMark || (twoPass ? serialHZB : variant == 1u);
                if (hybridHZB && device.capabilities().subPixelPrecisionBits > 8) { continue; }
                VirtualShadowMapPhysicalPool cachedPool, referencePool, serialPool;
                std::array<VirtualShadowMaps, 9> cached, reference, serial;
                VirtualShadowMapCasterCache casterCache;
                VirtualShadowMapSettings settings;
                settings.pageSize = 16; settings.virtualPagesPerAxis = 16; settings.clipmapCount = 3;
                settings.cacheEnabled = true; settings.cacheLayering = layered;
                settings.framesStaticThreshold = 2;
                settings.firstClipmapExtent = 4; settings.maxWorkItems = 65535;
                if (casterChanges) { settings.maxLightCacheAge = 0; }
                const ClusterLightGridSnapshot* markGrid = nullptr;
                auto record = [&](std::span<const PoolFace> specs, uint32_t budget, bool force, bool cancel = false,
                                  bool pointGroup = false, bool demandDriven = true, bool serialized = false,
                                  bool standalone = false) {
                    gpu.begin();
                    std::string log;
                    const VirtualShadowMapCasterSettings casterSettings{.instanceBuild = settings.instanceBuild,
                        .hierarchyHZB = hierarchyHZB};
                    auto casters = casterCache.prepare(device, *gpu.commands, scene, nullptr, log, casterSettings);
                    requirePool(casters, "Prepare shared pool casters: " + log);
                    requirePool(!settings.instanceBuild || casters->instanceTable != UINT32_MAX,
                        "Instance Build test did not publish an instance table");
                    if (settings.shadowLOD) {
                        casters = casterCache.prepareGPUShadowLOD(device, *gpu.commands, scene, nullptr, *casters, casterSettings, log);
                        requirePool(casters, "Prepare shared pool caster GPU LOD: " + log);
                        requirePool(casters->gpuLOD && casters->gpuLOD->context.clusterCount > 0,
                            "Packed coarse test did not produce an automatic shadow LOD hierarchy");
                    }
                    // A compact point light occupies the last six input faces and one output layer.
                    const uint32_t pointStart = uint32_t(specs.size()) - (pointGroup ? 6u : 0u);
                    const uint32_t layers = pointStart + uint32_t(pointGroup);
                    auto output = device.createTexture({.usage = TextureUsageBits::Storage | TextureUsageBits::TransferSource,
                        .format = Format::R8Unorm, .width = kWidth, .height = kHeight, .layerCount = layers});
                    requirePool(output, "Pool projection output");
                    const TextureBarrierDesc outputBegin{.texture = output->get(), .oldLayout = TextureLayout::Undefined,
                        .newLayout = TextureLayout::General, .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .range = {0, 1, 0, layers}};
                    requirePool(gpu.commands->synchronize({.textures = {&outputBegin, 1}}), "Pool output ready");
                    std::vector<VirtualShadowMapFaceRequest> requests;
                    for (uint32_t index = 0; index < specs.size(); ++index) {
                        requests.push_back({.shadow = &(serialized ? serial : force ? reference : cached)[specs[index].identity], .light = specs[index].light,
                            .output = {.shadow = output->get(), .layer = std::min(index, pointStart), .sharedPointLayer = index >= pointStart},
                            .sourceSlot = specs[index].sourceSlot});
                    }
                    auto frameSettings = settings;
                    frameSettings.forceRedraw = settings.forceRedraw || (force && !unifiedMark && !hierarchyHZB && !activeViews);
                    if (packedCoarse) { frameSettings.packedCoarseRaster = !force; }
                    if (hierarchyHZB) { frameSettings.hierarchyHZB = !force; }
                    if (unifiedMark) { frameSettings.unifiedMark = !force; }
                    if (activeViews) { frameSettings.compactShadowViews = !force; }
                    if (distantLights) { frameSettings.distantLightSinglePage = !force; }
                    if (hybrid && force) { frameSettings.clusterHybridRaster = false; }
                    if (twoPass && force) { frameSettings.twoPassHZB = false; }
                    if (serialized) { frameSettings.batchBuildRaster = false; }
                    if (!unifiedMark) { frameSettings.filter = pointGroup ? VirtualShadowMapFilter::PCF : VirtualShadowMapFilter::Hard; }
                    const auto view = camera.constants(gpu.index, kWidth, kHeight, kWidth, kHeight);
                    auto& pool = serialized ? serialPool : force ? referencePool : cachedPool;
                    CPUProfileRecorder cpu;
                    profiling::SchedulingMetrics nativeCommands;
                    auto results = [&]() -> Result<std::vector<VirtualShadowMapResult>> {
                        profiling::SchedulingCapture capture(&nativeCommands);
                        if (standalone) {
                            frameSettings.physicalPageCount = budget;
                            const auto& request = requests.front();
                            auto single = request.shadow->record(device, *gpu.commands, **depthView, view, request.light,
                                *casters, scene.materials(), {}, frameSettings, log, true, &request.output, &cpu);
                            if (!single) { return std::unexpected(single.error()); }
                            return std::vector<VirtualShadowMapResult>{std::move(*single)};
                        }
                        return VirtualShadowMaps::recordShared(pool, device, *gpu.commands, **depthView, view, requests,
                            *casters, scene.materials(), {}, frameSettings, budget, demandDriven, log, &cpu, nullptr, nullptr, markGrid);
                    }();
                    requirePool(results, "Record shared pool: " + log);
                    std::vector<VirtualShadowMapProjectionGroup> groups(layers);
                    for (uint32_t index = 0; index < specs.size(); ++index) {
                        requirePool((*results)[index].atlas == results->front().atlas, "Faces allocated independent physical atlases");
                        auto& group = groups[std::min(index, pointStart)];
                        const bool point = index >= pointStart;
                        const uint32_t face = point ? index - pointStart : 0u;
                        group.faces[face] = requests[index].shadow;
                        group.results[face] = (*results)[index];
                        group.faceCount = point ? 6u : 1u;
                        group.filteredPoint = point;
                    }
                    requirePool(VirtualShadowMaps::projectShared(*gpu.commands, groups, log), "Project shared pool: " + log);
                    PoolSnapshot snapshot;
                    snapshot.nativeCommands = nativeCommands;
                    if (casterChanges) {
                        for (const auto& section : cpu.sections) {
                            if (section.name == "Prepare VSM projections and dirty bounds") {
                                for (const auto& work : section.cpuWork) { snapshot.commands[work.name] += work.value; }
                            }
                        }
                    }
                    if (unifiedMark) {
                        for (const auto& section : cpu.sections) {
                            if (section.name == "Record VSM view mark") {
                                for (const auto& work : section.cpuWork) { snapshot.commands[work.name] += work.value; }
                            }
                        }
                        requirePool(snapshot.commands["markDispatches"] == uint32_t(!force),
                            "Unexpected count of full-screen view Mark dispatches");
                        if (!force) {
                            requirePool(snapshot.commands.at("faces") == specs.size() && snapshot.commands.at("lightGrid") == uint32_t(markGrid != nullptr),
                                "Unified Mark used an unexpected face count or grid permutation");
                        }
                    }
                    if (twoPass && !force) {
                        const bool perFace = serialized || standalone;
                        const auto sectionName = perFace ? "Record VSM build and raster" : "Record VSM build and raster batches";
                        const auto sections = std::ranges::count_if(cpu.sections,
                            [&](const auto& section) { return section.name == sectionName; });
                        requirePool(sections == (perFace ? specs.size() : 1u), "Two-pass HZB used an unexpected recording path");
                    }
                    if (scenario_ == PoolScenario::Allocation) {
                        snapshot.commands = poolRecordingCommands(cpu);
                        requirePool(snapshot.commands.at("faces") == specs.size(), "Recording counters omitted an admitted face");
                        requirePool(nativeCommands.invalidScopes == 0 && nativeCommands.drawCalls == snapshot.commands.at("meshDraws") &&
                            nativeCommands.dispatchCalls >= snapshot.commands.at("computeDispatches") + snapshot.commands.at("prepareDispatches"),
                            "Build/Raster counters disagree with the independent RHI command observer");
                    }
                    snapshot.memory = standalone ? requests.front().shadow->memoryUsage() : pool.memoryUsage();
                    const auto& atlasDesc = results->front().atlas->desc();
                    snapshot.atlasWidth = atlasDesc.width; snapshot.atlasHeight = atlasDesc.height;
                    snapshot.atlasLayers = atlasDesc.layerCount; snapshot.pagesX = results->front().physicalPagesX;
                    snapshot.pageSize = frameSettings.pageSize;
                    const uint64_t visibilityBytes = uint64_t(kPixels) * layers;
                    const uint64_t atlasOffset = (visibilityBytes + 3u) & ~uint64_t(3u);
                    const uint64_t atlasBytes = uint64_t(atlasDesc.width) * atlasDesc.height * atlasDesc.layerCount * sizeof(uint32_t);
                    uint64_t bytes = atlasOffset + atlasBytes;
                    std::vector<uint64_t> offsets;
                    for (uint32_t index = 0; index < specs.size(); ++index) {
                        auto& face = snapshot.faces[specs[index].identity];
                        face.pages.resize((*results)[index].clipmapCount * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1u);
                        offsets.push_back(bytes);
                        bytes += sizeof(face.stats) + face.pages.size() * sizeof(Page) + sizeof(face.cache);
                        if (packedCoarse) { bytes += sizeof(face.coarseGroups); }
                        if (activeViews) { bytes += sizeof(face.activeViews); }
                        if (hybrid) {
                            bytes += sizeof(face.hybrid) + sizeof(face.hybridRouting) +
                                sizeof(face.hardwarePrimitives) + sizeof(face.hardwareFineGroups);
                        }
                        if (frameSettings.twoPassHZB) { bytes += sizeof(face.hzb); }
                        if (hierarchyHZB) { bytes += sizeof(face.hierarchyHZB) + sizeof(face.instanceClusterTickets); }
                        if (unifiedMark && (*results)[index].requestedReceiverMasks) {
                            face.requestedMasks.resize(face.pages.size());
                            bytes += face.requestedMasks.size() * sizeof(uint64_t);
                        }
                    }
                    auto readback = device.createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                        .memoryLocation = MemoryLocation::HostReadback});
                    requirePool(readback, "Pool readback");
                    const TextureBarrierDesc textureReady[]{
                        {.texture = output->get(), .oldLayout = TextureLayout::General, .newLayout = TextureLayout::TransferSource,
                            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .range = {0, 1, 0, layers}},
                        {.texture = results->front().atlas, .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .range = {0, 1, 0, atlasDesc.layerCount}}};
                    std::vector<BufferBarrierDesc> bufferReady;
                    for (const auto& result : *results) {
                        for (auto* buffer : {result.statistics, result.pageTable, result.cacheDiagnostics,
                                unifiedMark ? result.requestedReceiverMasks : nullptr}) {
                            if (!buffer) { continue; }
                            bufferReady.push_back({.buffer = buffer,
                                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}});
                        }
                    }
                    requirePool(gpu.commands->synchronize({.textures = textureReady, .buffers = bufferReady}), "Pool readback ready");
                    requirePool((*readback)->slice({0, visibilityBytes}).and_then([&](const auto& slice) {
                        return gpu.commands->copyTextureToBuffer({.texture = output->get(), .buffer = slice,
                            .width = kWidth, .height = kHeight, .layerCount = layers});
                    }), "Pool visibility readback");
                    requirePool((*readback)->slice({atlasOffset, atlasBytes}).and_then([&](const auto& slice) {
                        return gpu.commands->copyTextureToBuffer({.texture = results->front().atlas, .buffer = slice,
                            .width = atlasDesc.width, .height = atlasDesc.height, .layerCount = atlasDesc.layerCount});
                    }), "Pool atlas readback");
                    for (uint32_t index = 0; index < specs.size(); ++index) {
                        const auto& face = snapshot.faces.at(specs[index].identity);
                        uint64_t offset = offsets[index];
                        const auto copyCounterRange = [&](Buffer* buffer, uint64_t sourceOffset, uint64_t size) {
                            auto from = buffer->slice({sourceOffset, size}), to = (*readback)->slice({offset, size});
                            requirePool(from, "Pool readback source"); requirePool(to, "Pool readback destination");
                            requirePool(gpu.commands->copyBuffer(*from, *to), "Pool buffer readback");
                            offset += size;
                        };
                        for (const auto& [buffer, size] : {std::pair{(*results)[index].statistics, uint64_t(sizeof(face.stats))},
                            std::pair{(*results)[index].pageTable, uint64_t(face.pages.size() * sizeof(Page))},
                            std::pair{(*results)[index].cacheDiagnostics, uint64_t(sizeof(face.cache))}}) {
                            copyCounterRange(buffer, 0, size);
                        }
                        if (packedCoarse) {
                            copyCounterRange((*results)[index].statistics, sizeof(face.stats) + sizeof(uint64_t) * kVSMWorkRasterCoarseGroups,
                                sizeof(face.coarseGroups));
                        }
                        if (activeViews) {
                            static_assert(kVSMActiveInitialBuildRows == kVSMActiveFineViewMask + 1u);
                            copyCounterRange((*results)[index].cacheDiagnostics, sizeof(uint32_t) * kVSMActiveFineViewMask,
                                sizeof(face.activeViews));
                        }
                        if (hybrid) {
                            if (frameSettings.clusterHybridRaster) {
                                copyCounterRange((*results)[index].cacheDiagnostics, sizeof(uint32_t) * kVSMHybridSoftwareTickets, sizeof(face.hybrid));
                                static_assert(kVSMHybridSoftwareTriangles == kVSMHybridMixedTickets + 1u &&
                                    kVSMHybridHardwareTriangles == kVSMHybridMixedTickets + 2u &&
                                    sizeof(HybridLaneRouting) == 3 * sizeof(uint32_t));
                                copyCounterRange((*results)[index].cacheDiagnostics, sizeof(uint32_t) * kVSMHybridMixedTickets,
                                    sizeof(face.hybridRouting));
                            } else { offset += sizeof(face.hybrid) + sizeof(face.hybridRouting); }
                            copyCounterRange((*results)[index].statistics, sizeof(face.stats) + sizeof(uint64_t) * kVSMWorkRasterOutputPrimitives,
                                sizeof(face.hardwarePrimitives));
                            copyCounterRange((*results)[index].statistics, sizeof(face.stats) + sizeof(uint64_t) * kVSMWorkRasterFineGroups,
                                sizeof(face.hardwareFineGroups));
                        }
                        if (frameSettings.twoPassHZB) {
                            static_assert(kVSMHZBRejectedTickets == kVSMHZBDeferredTickets + 1 &&
                                kVSMHZBRecoveredTickets == kVSMHZBDeferredTickets + 2 && sizeof(TwoPassTickets) == 3 * sizeof(uint32_t));
                            copyCounterRange((*results)[index].cacheDiagnostics, sizeof(uint32_t) * kVSMHZBDeferredTickets, sizeof(face.hzb));
                        }
                        if (hierarchyHZB) {
                            static_assert(sizeof(HierarchyHZBCounts) == 6 * sizeof(uint32_t));
                            static_assert(kVSMHierarchyHZBRecoveredNodes == kVSMHierarchyHZBDeferredInstances + 5u);
                            copyCounterRange((*results)[index].cacheDiagnostics,
                                sizeof(uint32_t) * kVSMHierarchyHZBDeferredInstances, sizeof(face.hierarchyHZB));
                            copyCounterRange((*results)[index].statistics, sizeof(uint32_t) * kVirtualShadowMapBuildDiagnosticWordOffset +
                                sizeof(uint64_t) * kVSMBuildDiagnosticInstanceClusterTickets, sizeof(face.instanceClusterTickets));
                        }
                        if (!face.requestedMasks.empty()) {
                            copyCounterRange((*results)[index].requestedReceiverMasks, (*results)[index].requestedReceiverMaskOffset,
                                face.requestedMasks.size() * sizeof(uint64_t));
                        }
                    }
                    const TextureBarrierDesc atlasReady{.texture = results->front().atlas, .oldLayout = TextureLayout::TransferSource,
                        .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}, .range = {0, 1, 0, atlasDesc.layerCount}};
                    requirePool(gpu.commands->synchronize({.textures = {&atlasReady, 1}}), "Pool atlas restore");
                    gpu.finish(cancel);
                    if (cancel) { return snapshot; }
                    (*readback)->invalidate();
                    const auto* data = static_cast<const uint8_t*>((*readback)->map());
                    requirePool(data != nullptr, "Map pool results");
                    snapshot.atlas.resize(atlasBytes / sizeof(uint32_t));
                    std::memcpy(snapshot.atlas.data(), data + atlasOffset, atlasBytes);
                    for (uint32_t index = 0; index < specs.size(); ++index) {
                        auto& face = snapshot.faces.at(specs[index].identity);
                        std::memcpy(face.stats.data(), data + offsets[index], sizeof(face.stats));
                        std::memcpy(face.pages.data(), data + offsets[index] + sizeof(face.stats), face.pages.size() * sizeof(Page));
                        std::memcpy(face.cache.data(), data + offsets[index] + sizeof(face.stats) + face.pages.size() * sizeof(Page), sizeof(face.cache));
                        if (packedCoarse) {
                            std::memcpy(&face.coarseGroups, data + offsets[index] + sizeof(face.stats) + face.pages.size() * sizeof(Page) +
                                sizeof(face.cache), sizeof(face.coarseGroups));
                        }
                        if (activeViews) {
                            std::memcpy(face.activeViews.data(), data + offsets[index] + sizeof(face.stats) +
                                face.pages.size() * sizeof(Page) + sizeof(face.cache), sizeof(face.activeViews));
                        }
                        if (hybrid) {
                            const auto* counters = data + offsets[index] + sizeof(face.stats) + face.pages.size() * sizeof(Page) + sizeof(face.cache);
                            if (frameSettings.clusterHybridRaster) {
                                std::memcpy(face.hybrid.data(), counters, sizeof(face.hybrid));
                                std::memcpy(&face.hybridRouting, counters + sizeof(face.hybrid), sizeof(face.hybridRouting));
                            }
                            counters += sizeof(face.hybrid) + sizeof(face.hybridRouting);
                            std::memcpy(&face.hardwarePrimitives, counters, sizeof(face.hardwarePrimitives));
                            std::memcpy(&face.hardwareFineGroups, counters + sizeof(face.hardwarePrimitives), sizeof(face.hardwareFineGroups));
                        }
                        if (frameSettings.twoPassHZB) {
                            std::memcpy(&face.hzb, data + offsets[index] + sizeof(face.stats) + face.pages.size() * sizeof(Page) +
                                sizeof(face.cache), sizeof(face.hzb));
                        }
                        if (hierarchyHZB) {
                            const auto* counters = data + offsets[index] + sizeof(face.stats) + face.pages.size() * sizeof(Page) +
                                sizeof(face.cache) + sizeof(face.hzb);
                            std::memcpy(&face.hierarchyHZB, counters, sizeof(face.hierarchyHZB));
                            std::memcpy(&face.instanceClusterTickets, counters + sizeof(face.hierarchyHZB), sizeof(face.instanceClusterTickets));
                        }
                        if (!face.requestedMasks.empty()) {
                            std::memcpy(face.requestedMasks.data(), data + offsets[index] + sizeof(face.stats) +
                                face.pages.size() * sizeof(Page) + sizeof(face.cache), face.requestedMasks.size() * sizeof(uint64_t));
                        }
                        const uint64_t offset = uint64_t(std::min(index, pointStart)) * kPixels;
                        face.visibility.assign(data + offset, data + offset + kPixels);
                    }
                    (*readback)->unmap();
                    if (!standalone) { requireUniqueOwners(snapshot, budget, packedCoarse && settings.maxWorkItems == 1); }
                    requirePool((snapshot.memory.staticAtlasBytes != 0) == layered, "Layered shared atlas memory missing or double counted");
                    return snapshot;
                };
                if (packedCoarse) {
                    settings.pageSize = 64;
                    settings.collectGPUWork = true;
                    VirtualShadowMapLight sun;
                    sun.direction = {0, 0, -1};
                    const std::array faces{PoolFace{0, sun}};
                    constexpr uint32_t kBudget = 256;
                    const auto coarseDepth = [](const PoolSnapshot& image) {
                        const uint32_t slot = image.faces.at(0).pages.back()[0];
                        std::vector<uint32_t> texels(image.pageSize * image.pageSize);
                        for (uint32_t y = 0; y < image.pageSize; ++y) {
                            for (uint32_t x = 0; x < image.pageSize; ++x) {
                                texels[y * image.pageSize + x] = poolPageTexel(image, slot, x, y);
                            }
                        }
                        return texels;
                    };
                    const auto compare = [&](const char* phase, bool expectWarm, bool expectOverflow) {
                        validation.phase(phase);
                        const auto packed = record(faces, kBudget, false);
                        const auto scalar = record(faces, kBudget, true);
                        const auto& actual = packed.faces.at(0);
                        const auto& reference = scalar.faces.at(0);
                        const auto actualDepth = coarseDepth(packed);
                        const bool equalDepth = actualDepth == coarseDepth(scalar);
                        const bool equalVisibility = actual.visibility == reference.visibility;
                        report["frames"].push_back({{"phase", phase}, {"equalCoarseD32", equalDepth}, {"equalVisibility", equalVisibility},
                            {"packedCoarseGroups", actual.coarseGroups}, {"scalarCoarseGroups", reference.coarseGroups},
                            {"packedStatistics", actual.stats}, {"scalarStatistics", reference.stats}});
                        save();
                        requirePool(equalDepth && equalVisibility && std::ranges::any_of(actualDepth, [](uint32_t z) { return z != 0; }),
                            std::string(phase) + ": packed coarse depth or projected visibility differs from scalar redraw");
                        requirePool(reference.coarseGroups > 0 && (expectWarm ? actual.coarseGroups == 0 :
                            actual.coarseGroups > 0 && actual.coarseGroups < reference.coarseGroups),
                            std::string(phase) + ": packed coarse submission did not reuse or reduce workgroups");
                        if (expectOverflow) {
                            requirePool(actual.stats[5] > 0 && reference.stats[5] > 0 && actual.stats[24] == 0 && reference.stats[24] == 0 &&
                                actual.stats[7] == actual.stats[6] && reference.stats[7] == reference.stats[6] && actual.stats[6] > 0,
                                "Fine overflow did not resolve receivers using the valid coarse page");
                        } else {
                            requirePool(actual.stats[5] == 0 && reference.stats[5] == 0 && actual.stats[24] > 0,
                                "Normal packed coarse comparison did not retain fine-page coverage");
                        }
                    };
                    compare("Cold", false, false);
                    compare("Cached", true, false);
                    settings.forceRedraw = true;
                    settings.maxWorkItems = 1;
                    compare("FineOverflow", false, true);
                    settings.maxWorkItems = 65535;
                    compare("ForcedRecovery", false, false);
                    settings.shadowLOD = true;
                    compare("AutomaticShadowLOD", false, false);
                    continue;
                }
                if (lightCacheRecovery) {
                    VirtualShadowMapLight sun;
                    sun.direction = {1, 0, -1};
                    VirtualShadowMapLight spot;
                    spot.type = VirtualShadowMapLightType::Spot; spot.position = {0, 0, 3};
                    spot.direction = {0, 0, -1}; spot.range = 10; spot.outerConeAngle = .8f;
                    auto claimant = spot;
                    claimant.position = {3.5f, 0, 2}; claimant.outerConeAngle = .18f;
                    const std::array initial{PoolFace{1, spot}, PoolFace{0, sun}};
                    const std::array reordered{initial[1], initial[0]};
                    const std::array onlySun{initial[1]};
                    const std::array onlySpot{initial[0]};
                    const std::array onlyClaimant{PoolFace{2, claimant}};
                    const std::array sunAndClaimant{initial[1], onlyClaimant[0]};
                    constexpr uint32_t ampleBudget = 512;
                    constexpr uint32_t pressuredBudget = 8;
                    constexpr float depthTolerance = 1e-8f;
                    constexpr uint32_t spotId = 1;
                    constexpr uint32_t claimantId = 2;
                    const auto render = [&](const char* phase, std::span<const PoolFace> active, uint32_t budget) {
                        validation.phase(std::string(layered ? "Layered-" : "Combined-") + phase);
                        auto snapshot = record(active, budget, false);
                        Json entry{{"phase", phase}, {"layered", layered}, {"faces", Json::array()}};
                        for (const auto& [identity, face] : snapshot.faces) {
                            entry["faces"].push_back({{"identity", identity}, {"cacheReused", face.cache[kVSMCacheReused]},
                                {"transformChanges", face.cache[kVSMCacheTransformChanges]}, {"residentPages", face.stats[12]},
                                {"dirtyFinePages", face.cache[kVSMCacheCompositeDirtyFinePages]},
                                {"dirtyCoarsePages", face.cache[kVSMCacheCompositeDirtyCoarsePages]}, {"coarseSlot", face.pages.back()[0]}});
                        }
                        report["frames"].push_back(std::move(entry)); save();
                        return snapshot;
                    };

                    const auto original = render("Cold", initial, ampleBudget);
                    const auto missing = render("MissingWithSpareSlots", sunAndClaimant, ampleBudget);
                    const uint32_t spareSlot = missing.faces.at(claimantId).pages.back()[0];
                    requirePool(std::ranges::none_of(original.faces.at(spotId).pages, [spareSlot](const Page& page) {
                        return (page[1] & 2u) != 0u && page[0] == spareSlot;
                    }), "An unused slot did not take priority over a dormant spot page");
                    (void)render("StillMissing", onlySun, ampleBudget);
                    const uint64_t emptyFrame = gpu.index++;
                    for (const auto& face : sunAndClaimant) {
                        requirePool(cached[face.identity].retainInactiveCache(emptyFrame, settings.maxLightCacheAge),
                            "A primary-view frame without any lights released a recent shared snapshot");
                    }
                    requirePool(cached[spotId].retainInactiveCache(emptyFrame, settings.maxLightCacheAge),
                        "A primary-view frame without any lights released the absent spot snapshot");
                    const auto restored = render("ReturnClean", reordered, ampleBudget);
                    const auto& clean = restored.faces.at(spotId);
                    requirePool(clean.cache[kVSMCacheReused] == 1 && clean.cache[kVSMCacheCompositeDirtyFinePages] == 0 &&
                        clean.cache[kVSMCacheCompositeDirtyCoarsePages] == 0 && clean.stats[20] == 0,
                        "Returning spot did not reuse clean depth after consecutive missing frames");
                    requirePool(requireRetainedDepth(original, restored, spotId) == original.faces.at(spotId).stats[12] &&
                        clean.visibility == original.faces.at(spotId).visibility,
                        "Reordering active faces on return changed retained physical pages or visibility");

                    (void)render("MissingBeforeMove", onlySun, ampleBudget);
                    const auto originalTransform = scene.nodes()[1].localMatrix;
                    auto translation = originalTransform;
                    translation.SetupByTranslation({.6f, 0, 0});
                    requirePool(scene.setNodeLocalMatrix(1, translation * originalTransform), "Move blocker while spot is absent");
                    (void)render("MissingAfterMove", onlySun, ampleBudget);
                    validation.phase(layered ? "Layered-CancelReturn" : "Combined-CancelReturn");
                    (void)record(reordered, ampleBudget, false, true);
                    const auto moved = render("ReturnAfterMove", reordered, ampleBudget);
                    const auto& movedSpot = moved.faces.at(spotId);
                    requirePool(movedSpot.cache[kVSMCacheReused] == 1 && movedSpot.cache[kVSMCacheTransformChanges] == 1 &&
                        movedSpot.cache[kVSMCacheCompositeDirtyFinePages] > 0 && movedSpot.cache[kVSMCacheCompositeDirtyCoarsePages] == 1 &&
                        movedSpot.visibility != clean.visibility, "Returned spot lost a caster change during its absence or canceled return");
                    const auto redraw = record(reordered, ampleBudget, true);
                    compareRasterDepth(moved, redraw, spotId, depthTolerance, false, false);
                    requirePool(scene.setNodeLocalMatrix(1, originalTransform), "Restore blocker for pressure case");

                    const auto full = render("PressureCold", onlySpot, pressuredBudget);
                    requirePool(full.faces.at(spotId).stats[12] == pressuredBudget, "Pressure fixture did not occupy the whole pool");
                    const auto claimed = render("PressureMissing", onlyClaimant, pressuredBudget);
                    const auto& claimantFace = claimed.faces.at(claimantId);
                    requirePool(claimantFace.stats[0] == 0 && claimantFace.stats[12] == 1,
                        "Offscreen claimant did not allocate exactly one coarse page");
                    const uint32_t claimedSlot = claimantFace.pages.back()[0];
                    requirePool(std::ranges::any_of(full.faces.at(spotId).pages, [claimedSlot](const Page& page) {
                        return (page[1] & 2u) != 0u && page[0] == claimedSlot;
                    }), "Claimant did not take a previously resident slot from the absent spot");
                    bool replacedDepth = false;
                    for (uint32_t y = 0; y < claimed.pageSize; ++y) {
                        for (uint32_t x = 0; x < claimed.pageSize; ++x) {
                            const uint32_t current = poolPageTexel(claimed, claimedSlot, x, y);
                            replacedDepth |= current != 0u && current != poolPageTexel(full, claimedSlot, x, y);
                        }
                    }
                    requirePool(replacedDepth, "Claimant did not write different D32 depth into the stolen physical page");

                    const auto recovered = render("PressureReturn", onlySpot, pressuredBudget);
                    const auto& recoveredSpot = recovered.faces.at(spotId);
                    requirePool(recoveredSpot.cache[kVSMCacheReused] == 1 && recoveredSpot.stats[12] == pressuredBudget &&
                        recoveredSpot.cache[kVSMCacheCompositeDirtyFinePages] + recoveredSpot.cache[kVSMCacheCompositeDirtyCoarsePages] == 1,
                        "Pressure return did not reject just the overwritten page");
                    requirePool(requireRetainedDepth(full, recovered, spotId) >= pressuredBudget - 1u,
                        "Pressure return discarded untouched resident pages");
                    compareRasterDepth(recovered, record(onlySpot, pressuredBudget, true), spotId, depthTolerance, false, false);

                    std::vector<uint32_t> pixels(kPixels);
                    std::transform(movedSpot.visibility.begin(), movedSpot.visibility.end(), pixels.begin(),
                        [](uint8_t value) { return uint32_t(value) * 0x010101u | 0xff000000u; });
                    std::string imageLog;
                    requirePool(saveRgba8Png(directory / (layered ? "LayeredRestoredSpot.png" : "RestoredSpot.png"),
                        reinterpret_cast<const uint8_t*>(pixels.data()), kWidth, kHeight, imageLog), "Save restored spot shadow: " + imageLog);
                    continue;
                }
                if (distantLights) {
                    settings.pageSize = 64;
                    settings.virtualPagesPerAxis = 4;
                    settings.collectGPUWork = true;
                    settings.unifiedMark = true;
                    settings.compactShadowViews = true;
                    const auto& caps = device.capabilities();
                    settings.clusterHybridRaster = layered && caps.computeSubgroupBallotArithmetic &&
                        caps.subPixelPrecisionBits >= 1 && caps.subPixelPrecisionBits <= 8;
                    VirtualShadowMapLight sun;
                    sun.direction = {1, 0, -1};
                    VirtualShadowMapLight spot;
                    spot.type = VirtualShadowMapLightType::Spot;
                    spot.position = {0, 0, 3}; spot.direction = {0, 0, -1};
                    spot.range = 3.8f; spot.outerConeAngle = .8f;
                    auto unbounded = spot;
                    unbounded.range = 0;
                    // A single-page first face cannot supply the hybrid fine-raster scratch.
                    std::vector<PoolFace> faces{{1, spot}, {0, sun}, {2, unbounded}};
                    auto point = spot;
                    point.type = VirtualShadowMapLightType::PointFace;
                    point.position[2] = 2.2f;
                    for (uint32_t face = 0; face < 6; ++face) {
                        point.face = face;
                        faces.push_back({3u + face, point});
                    }
                    const auto blockerTransform = scene.nodes()[1].localMatrix;
                    auto seamTranslation = blockerTransform;
                    seamTranslation.SetupByTranslation({.8f, 0, 0});
                    requirePool(scene.setNodeLocalMatrix(1, seamTranslation * blockerTransform), "Position blocker across point face boundary");
                    const auto nearCamera = camera.camera();
                    auto farCamera = nearCamera;
                    // The constant receiver depth represents the fixture floor.
                    // Keep the wider camera footprint inside that floor.
                    farCamera.orthoHeight = 6;
                    constexpr uint32_t budget = 512;
                    // Moving a page to a different atlas tile changes hardware depth interpolation rounding.
                    constexpr float depthTolerance = 1e-8f;
                    const uint32_t fullPages = settings.clipmapCount * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1u;
                    const auto local = [](uint32_t identity) { return identity == 1u || identity >= 3u; };
                    const auto coarseDepth = [](const PoolSnapshot& snapshot, uint32_t identity) {
                        const uint32_t slot = snapshot.faces.at(identity).pages.back()[0];
                        std::vector<uint32_t> depth(snapshot.pageSize * snapshot.pageSize);
                        for (uint32_t y = 0; y < snapshot.pageSize; ++y) {
                            for (uint32_t x = 0; x < snapshot.pageSize; ++x) {
                                depth[y * snapshot.pageSize + x] = poolPageTexel(snapshot, slot, x, y);
                            }
                        }
                        return depth;
                    };
                    const auto audit = [&](std::string_view phase, const PoolSnapshot& snapshot, bool distant) {
                        Json entry{{"phase", phase}, {"layered", layered}, {"clusterHybridRaster", settings.clusterHybridRaster},
                            {"faces", Json::array()}};
                        for (const auto& [identity, face] : snapshot.faces) {
                            const bool singlePage = distant && local(identity);
                            requirePool(face.pages.size() == (singlePage ? 1u : fullPages),
                                std::string(phase) + ": wrong virtual page count for face " + std::to_string(identity));
                            if (singlePage) {
                                requirePool(face.stats[0] == 0 && face.stats[1] == 0 && face.stats[2] == 0 &&
                                    face.stats[4] == 0 && face.stats[12] == 1 && face.stats[16] == 0,
                                    std::string(phase) + ": a single-page face requested, admitted or built fine pages");
                            }
                            entry["faces"].push_back({{"identity", identity}, {"virtualPages", face.pages.size()},
                                {"requestedFinePages", face.stats[0]}, {"residentPages", face.stats[12]},
                                {"reusedCache", face.cache[kVSMCacheReused]},
                                {"dirtyCoarsePages", face.cache[kVSMCacheCompositeDirtyCoarsePages]}});
                        }
                        report["frames"].push_back(std::move(entry));
                        save();
                    };
                    const auto compareFull = [&](std::string_view phase) {
                        validation.phase(std::string(phase));
                        const auto actual = record(faces, budget, false, false, true);
                        const auto reference = record(faces, budget, true, false, true);
                        audit(phase, actual, false);
                        for (const auto& face : faces) {
                            compareRasterDepth(actual, reference, face.identity, depthTolerance, false, false);
                        }
                        requirePool(actual.faces.at(1).stats[0] > 0 && actual.faces.at(8).stats[0] > 0,
                            std::string(phase) + ": near spot or point face requested no fine pages");
                    };
                    compareFull("Near");

                    requirePool(camera.setCamera(farCamera), "Zoom out for distant local lights");
                    validation.phase("Far");
                    const auto far = record(faces, budget, false, false, true);
                    audit("Far", far, true);
                    const auto full = record(faces, budget, true, false, true);
                    requirePool(full.faces.at(1).stats[0] > 0 && full.faces.at(8).stats[0] > 0 &&
                        far.faces.at(0).stats[0] > 0 && far.faces.at(2).stats[0] > 0,
                        "The mixed case did not suppress normal fine requests selectively");
                    for (uint32_t identity : {1u, 3u, 8u}) {
                        const auto& face = far.faces.at(identity);
                        requirePool(std::ranges::any_of(coarseDepth(far, identity), [](uint32_t depth) { return depth != 0; }) &&
                            face.cache[kVSMCacheReused] == 0,
                            "Switching to single-page mode did not rasterize the newly laid out page");
                    }
                    const auto& pointShadow = far.faces.at(8).visibility;
                    const uint32_t rowCenter = (kHeight / 2) * kWidth + kWidth / 2;
                    // At the receiver, these samples straddle the -Z/+X seam of the point light.
                    const uint32_t seam = uint32_t(point.position[2] * kHeight / farCamera.orthoHeight);
                    requirePool(pointShadow[rowCenter + seam] < 255 && pointShadow[rowCenter + seam + 1u] < 255,
                        "Distant point shadows must cross the cube face boundary");

                    validation.phase("FarWarm");
                    const auto warm = record(faces, budget, false, false, true);
                    audit("FarWarm", warm, true);
                    for (const auto& [identity, face] : warm.faces) {
                        compareRasterDepth(warm, far, identity, 0.0f, false, false);
                        if (local(identity)) {
                            requirePool(face.cache[kVSMCacheReused] == 1 && face.cache[kVSMCacheCompositeDirtyCoarsePages] == 0 &&
                                face.pages.front()[0] == far.faces.at(identity).pages.front()[0],
                                "A stable single-page face did not reuse its clean physical page");
                        }
                    }

                    auto movedTranslation = blockerTransform;
                    movedTranslation.SetupByTranslation({-.8f, 0, 0});
                    requirePool(scene.setNodeLocalMatrix(1, movedTranslation * blockerTransform), "Move the distant blocker");
                    validation.phase("FarMoved");
                    const auto moved = record(faces, budget, false, false, true);
                    audit("FarMoved", moved, true);
                    const auto& movedSpot = moved.faces.at(1);
                    requirePool(movedSpot.cache[kVSMCacheReused] == 1 &&
                        movedSpot.cache[kVSMCacheCompositeDirtyCoarsePages] == 1 && movedSpot.visibility != warm.faces.at(1).visibility,
                        "A changed caster did not invalidate and rerasterize the single cached page");

                    requirePool(camera.setCamera(nearCamera), "Zoom back in for local mips");
                    validation.phase("CancelNear");
                    (void)record(faces, budget, false, true, true);
                    compareFull("NearAfterCancel");

                    requirePool(camera.setCamera(farCamera), "Restore the distant comparison camera");
                    validation.phase("CoarseReference");
                    // The legacy control has no budget for fine pages. Select its one-page mip
                    // so its PCF tap footprint equals the actual fallback page's texel size.
                    settings.resolutionLodBias = float(settings.clipmapCount - 1u);
                    const uint32_t coarseBudget = uint32_t(faces.size());
                    const auto compact = record(faces, coarseBudget, false, false, true);
                    const auto coarse = record(faces, coarseBudget, true, false, true);
                    audit("CoarseReference", compact, true);
                    for (const auto& [identity, face] : compact.faces) {
                        const auto& reference = coarse.faces.at(identity);
                        requirePool(face.stats[12] == 1 && reference.stats[12] == 1 &&
                            face.visibility == reference.visibility, "Coarse-only reference disagrees on resident pages or PCF visibility");
                        const auto referenceDepth = coarseDepth(coarse, identity);
                        requirePool(coarseDepth(compact, identity) == referenceDepth,
                            "Single-page physical depth differs from the full table's coarse page");
                        if (local(identity)) {
                            const bool sameDepth = std::ranges::equal(coarseDepth(moved, identity), referenceDepth,
                                [depthTolerance](uint32_t a, uint32_t b) {
                                    return (a == 0) == (b == 0) &&
                                        std::abs(std::bit_cast<float>(a) - std::bit_cast<float>(b)) <= depthTolerance;
                                });
                            requirePool(sameDepth && moved.faces.at(identity).visibility == reference.visibility,
                                "Rerasterizing the resident single page differs from a coarse-only forced redraw");
                        }
                    }
                    std::vector<uint32_t> pixels(kPixels);
                    std::transform(pointShadow.begin(), pointShadow.end(), pixels.begin(),
                        [](uint8_t value) { return uint32_t(value) * 0x010101u | 0xff000000u; });
                    std::string imageLog;
                    requirePool(saveRgba8Png(directory / (layered ? "LayeredFarPoint.png" : "FarPoint.png"),
                        reinterpret_cast<const uint8_t*>(pixels.data()), kWidth, kHeight, imageLog), "Save distant point shadow: " + imageLog);
                    requirePool(camera.setCamera(nearCamera), "Restore the pool camera");
                    requirePool(scene.setNodeLocalMatrix(1, blockerTransform), "Restore the pool blocker");
                    continue;
                }
                if (activeViews) {
                    settings.collectGPUWork = true;
                    settings.compactBuild = layered;
                    settings.dirtyPageHierarchy = layered;
                    // Start receiver marking at level 1 so active faces have
                    // real coarse-mip work and an unused finest view.
                    settings.receiverFootprint = false;
                    settings.resolutionLodBias = 1.0f;
                    VirtualShadowMapLight sun;
                    sun.direction = {1, 0, -1};
                    VirtualShadowMapLight spot;
                    spot.type = VirtualShadowMapLightType::Spot; spot.position = {3.5f, 0, 3};
                    spot.direction = {0, 0, -1}; spot.range = 10; spot.outerConeAngle = .15f;
                    std::vector<PoolFace> faces{{0, sun}, {1, spot}};
                    auto point = spot;
                    point.type = VirtualShadowMapLightType::PointFace; point.position = {0, 0, 4};
                    for (uint32_t face = 0; face < 6; ++face) {
                        point.face = face;
                        faces.push_back({3u + face, point});
                    }
                    constexpr uint32_t budget = 512u;
                    const uint32_t pagesPerLevel = settings.virtualPagesPerAxis * settings.virtualPagesPerAxis;
                    const uint32_t expectedColdMask = ((1u << settings.clipmapCount) - 1u) & ~1u;
                    const auto compare = [&](const char* phase) {
                        const std::string label = std::string(layered ? "Layered-" : "Combined-") + phase;
                        validation.phase(label);
                        const auto actual = record(faces, budget, false, false, true);
                        const auto reference = record(faces, budget, true, false, true);
                        Json entry{{"phase", phase}, {"layered", layered}, {"faces", Json::array()}};
                        for (const auto& [identity, face] : actual.faces) {
                            const auto& referenceFace = reference.faces.at(identity);
                            uint32_t expectedMask = 0;
                            for (uint32_t index = 0; index + 1u < referenceFace.pages.size(); ++index) {
                                const auto& page = referenceFace.pages[index];
                                if (page[0] != UINT32_MAX && (page[1] & 2u) != 0u && (page[3] & 10u) != 0u) {
                                    expectedMask |= 1u << (index / pagesPerLevel);
                                }
                            }
                            entry["faces"].push_back({{"identity", identity}, {"activeFineViewMask", face.activeViews[0]},
                                {"expectedFineViewMask", expectedMask}, {"initialBuildRows", face.activeViews[1]}, {"cache", face.cache}});
                            const std::string location = label + " face " + std::to_string(identity);
                            requirePool(referenceFace.activeViews == decltype(face.activeViews){},
                                location + ": the reference published compact view work");
                            requirePool(face.activeViews[0] == expectedMask && face.activeViews[1] == uint32_t(expectedMask != 0u),
                                location + ": expected mask " + std::to_string(expectedMask) + ", compact GPU mask " +
                                std::to_string(face.activeViews[0]) + ", initial Build rows " + std::to_string(face.activeViews[1]));
                            requirePool(face.pages == referenceFace.pages && face.cache == referenceFace.cache,
                                location + ": compacting fine views changed page ownership or invalidation");
                            compareRasterDepth(actual, reference, identity, 0.0f, false, identity == 0u, identity == 1u);
                        }
                        report["frames"].push_back(std::move(entry));
                        save();
                        return actual;
                    };
                    const auto cold = compare("Cold");
                    requirePool(cold.faces.at(0).activeViews[0] == expectedColdMask &&
                        cold.faces.at(8).activeViews[0] == expectedColdMask,
                        "Cold directional clipmaps and the visible point face did not exercise partial fine-view masks");
                    requirePool(cold.faces.at(1).activeViews[1] == 0u && cold.faces.at(7).activeViews[1] == 0u &&
                        cold.faces.at(1).cache[kVSMCacheCompositeDirtyCoarsePages] == 1u &&
                        cold.faces.at(7).cache[kVSMCacheCompositeDirtyCoarsePages] == 1u,
                        "Cold spot and unused point face did not exercise coarse-only raster with no fine Build row");
                    std::vector<uint32_t> pixels(kPixels);
                    std::transform(cold.faces.at(0).visibility.begin(), cold.faces.at(0).visibility.end(), pixels.begin(),
                        [](uint8_t value) { return uint32_t(value) * 0x010101u | 0xff000000u; });
                    std::string imageLog;
                    requirePool(saveRgba8Png(directory / (layered ? "LayeredShadow.png" : "CombinedShadow.png"),
                        reinterpret_cast<const uint8_t*>(pixels.data()), kWidth, kHeight, imageLog), "Save compact-view shadow: " + imageLog);

                    const auto warm = compare("Warm");
                    for (const auto& [identity, face] : warm.faces) {
                        requirePool(face.activeViews[1] == 0u && face.visibility == cold.faces.at(identity).visibility,
                            "A clean cached view produced Build work or changed visibility");
                    }
                    requireRetainedDepth(cold, warm, 0u);
                    requireRetainedDepth(cold, warm, 8u);

                    const auto originalMatrix = scene.nodes()[1].localMatrix;
                    auto translation = originalMatrix;
                    translation.SetupByTranslation({.6f, 0, .1f});
                    requirePool(scene.setNodeLocalMatrix(1, translation * originalMatrix), "Move compact-view blocker");
                    std::swap(faces[0], faces[1]);
                    for (auto& face : std::span(faces).subspan(2)) { face.light.position[2] = -4; }
                    validation.phase(layered ? "Layered-Cancel" : "Combined-Cancel");
                    (void)record(faces, budget, false, true, true);
                    (void)record(faces, budget, true, true, true);
                    const auto moved = compare("MovedAfterCancel");
                    requirePool(moved.faces.at(0).activeViews[1] == 1u &&
                        moved.faces.at(0).visibility != warm.faces.at(0).visibility,
                        "Moved caster failed to republish and rasterize fine work after cancellation");
                    requirePool(moved.faces.at(7).activeViews[1] == 1u && moved.faces.at(8).activeViews[1] == 0u,
                        "Moving the point light failed to transfer work to a different face after cancellation");
                    if (layered) {
                        (void)compare("Settle");
                        const auto demoted = compare("Demote");
                        const auto& sunFace = demoted.faces.at(0);
                        requirePool(sunFace.cache[kVSMCacheDynamicCasters] == 0u &&
                            sunFace.cache[kVSMCacheCompositeDirtyFinePages] == 0u &&
                            sunFace.cache[kVSMCacheStaticDirtyFinePages] > 0u && sunFace.activeViews[1] == 1u,
                            "Static-only merge did not keep its fine views in the compact Build");
                    }
                    requirePool(scene.setNodeLocalMatrix(1, originalMatrix), "Restore compact-view blocker");
                    continue;
                }
                if (casterChanges) {
                    VirtualShadowMapLight sun;
                    sun.direction = {0, 0, -1};
                    VirtualShadowMapLight spot;
                    spot.type = VirtualShadowMapLightType::Spot; spot.position = {0, 0, 3};
                    spot.direction = sun.direction; spot.range = 10; spot.outerConeAngle = .8f;
                    const std::array faces{PoolFace{0, sun}, PoolFace{1, spot}};
                    constexpr uint32_t budget = 512u;
                    const auto compare = [&](const char* phase, std::span<const PoolFace> active,
                                             uint32_t compared, uint32_t reused) {
                        validation.phase(std::string(layered ? "Layered-" : "Combined-") + phase);
                        const auto actual = record(active, budget, false);
                        const auto redraw = record(active, budget, true);
                        const auto serialized = record(active, budget, false, false, false, true, true);
                        Json entry{{"phase", phase}, {"layered", layered}, {"faces", Json::array()},
                            {"batchCounters", actual.commands}, {"serialCounters", serialized.commands}};
                        for (const auto& [identity, face] : actual.faces) {
                            entry["faces"].push_back({{"identity", identity}, {"cache", face.cache}});
                        }
                        report["frames"].push_back(std::move(entry));
                        save();
                        for (const auto* snapshot : {&actual, &serialized}) {
                            const auto& counters = snapshot->commands;
                            requirePool(counters.contains("comparedCasterStates") && counters.contains("reusedCasterChanges") &&
                                counters.at("comparedCasterStates") == compared && counters.at("reusedCasterChanges") == reused,
                                std::string(phase) + ": unexpected shared caster comparison count");
                        }
                        for (const auto& [identity, face] : actual.faces) {
                            const auto& referenceFace = serialized.faces.at(identity);
                            requirePool(face.cache == referenceFace.cache && face.pages == referenceFace.pages,
                                std::string(phase) + ": batched cache or page ownership differs from serialized rendering");
                            compareRasterDepth(actual, serialized, identity, 0.0f, false, false);
                            compareRasterDepth(actual, redraw, identity, 0.0f, false, false);
                        }
                        return actual;
                    };
                    const auto moveBlocker = [&](float x) {
                        float4x4 matrix;
                        matrix.SetupByTranslation({x, 0, 0});
                        requirePool(scene.setSourceMountMatrix("blocker", matrix), "Move caster-change blocker");
                    };
                    const auto requireCounter = [&](const PoolSnapshot& snapshot, uint32_t counter, uint32_t expected,
                                                    const char* message) {
                        for (const auto& [identity, face] : snapshot.faces) {
                            requirePool(face.cache[counter] == expected, message);
                        }
                    };
                    const auto cold = compare("Cold", faces, 0, 0);
                    const auto warm = compare("Warm", faces, 0, 0);
                    requireRetainedDepth(cold, warm, 0);
                    moveBlocker(.25f);
                    (void)record(faces, budget, false, true);
                    (void)record(faces, budget, false, true, false, true, true);
                    const auto moved = compare("MoveAfterCancel", faces, 2, 1);
                    requireCounter(moved, kVSMCacheTransformChanges, 1, "Lost caster transform change after cancellation");
                    requirePool(moved.faces.at(0).visibility != warm.faces.at(0).visibility,
                        "Moved blocker did not change directional shadows");
                    if (layered) {
                        requireCounter(moved, kVSMCachePromotedCasters, 1, "Moved caster was not promoted on both faces");
                        requireCounter(moved, kVSMCacheDynamicCasters, 1, "Moved caster was not dynamic on both faces");
                    }
                    (void)compare("Settle", faces, 0, 0);
                    const auto demoted = compare("Demote", faces, 0, 0);
                    if (layered) {
                        requireCounter(demoted, kVSMCacheDynamicCasters, 0, "Settled caster did not return to static depth");
                        requireCounter(demoted, kVSMCacheStaticDirtyCoarsePages, 1, "Demotion did not update static coarse depth");
                    }
                    requirePool(scene.setSourceEnabled("blocker", false), "Hide caster-change blocker");
                    const auto removed = compare("Remove", faces, 1, 1);
                    requireCounter(removed, kVSMCacheRemovedCasters, 1, "Removed caster was not recorded on both faces");
                    requirePool(removed.faces.at(0).visibility != moved.faces.at(0).visibility,
                        "Removing the blocker did not clear directional shadows");
                    requirePool(scene.setSourceEnabled("blocker", true), "Show caster-change blocker");
                    const auto added = compare("Add", faces, 2, 1);
                    requireCounter(added, kVSMCacheAddedCasters, 1, "Restored caster was not recorded on both faces");
                    (void)compare("OmitSpot", std::span(faces).first(1), 0, 0);
                    moveBlocker(0);
                    const auto returned = compare("ReturnSpot", faces, 2, 0);
                    requirePool(returned.faces.at(0).cache[kVSMCacheTransformChanges] == 1 &&
                        returned.faces.at(1).cache[kVSMCacheReused] == 0,
                        "Returning face consumed another face's cache history");
                    continue;
                }
                if (unifiedMark) {
                    VirtualShadowMapLight sun;
                    sun.direction = {.25f, 0, -1};
                    sun.angularRadius = .06f;
                    VirtualShadowMapLight spot;
                    spot.type = VirtualShadowMapLightType::Spot;
                    spot.position = {0, 0, 3};
                    spot.direction = {0, 0, -1};
                    spot.range = 8;
                    spot.outerConeAngle = .8f;
                    spot.sourceRadius = .20f;
                    auto offscreen = spot;
                    offscreen.position[0] = 50;
                    std::vector<PoolFace> faces{{0, sun, 0}, {1, spot, 2}, {2, offscreen, 3}};
                    auto point = spot;
                    point.type = VirtualShadowMapLightType::PointFace;
                    point.position = {.25f, .125f, 1.5f};
                    for (uint32_t face = 0; face < 6; ++face) {
                        point.face = face;
                        faces.push_back({3u + face, point, 4});
                    }
                    settings.smrtRayCount = 2;
                    settings.smrtSamplesPerRay = 3;
                    settings.smrtMaxRayAngle = .25f;
                    const auto compare = [&](const char* label, VirtualShadowMapFilter filter, uint32_t budget) {
                        settings.filter = filter;
                        validation.phase(label);
                        const bool compact = filter != VirtualShadowMapFilter::Hard;
                        const auto actual = record(faces, budget, false, false, compact);
                        const auto expected = record(faces, budget, true, false, compact);
                        uint32_t requestedPointFaces = 0, rejected = 0, partialMasks = 0;
                        Json requests = Json::array();
                        for (const auto& [id, face] : actual.faces) {
                            const auto& ref = expected.faces.at(id);
                            const auto requested = std::ranges::count_if(face.pages, [](const Page& p) { return (p[1] & 1u) != 0u; });
                            requirePool(requested == face.stats[0] + 1u && face.pages == ref.pages,
                                std::string(label) + ": requested pages or allocation differ on face " + std::to_string(id));
                            requirePool(face.requestedMasks == ref.requestedMasks, std::string(label) + ": receiver masks differ");
                            requirePool(face.visibility == ref.visibility, std::string(label) + ": shadow output differs");
                            requirePool(face.cache == ref.cache, std::string(label) + ": cache invalidation differs");
                            if (id >= 3 && face.stats[0] > 0) { ++requestedPointFaces; }
                            rejected += face.stats[2];
                            partialMasks += uint32_t(std::ranges::count_if(face.requestedMasks,
                                [](uint64_t mask) { return mask != 0 && mask != UINT64_MAX; }));
                            requests.push_back(face.stats[0]);
                        }
                        requirePool(actual.atlas == expected.atlas, std::string(label) + ": physical D32 differs");
                        requirePool(requestedPointFaces >= 3 && actual.faces.at(0).stats[0] && actual.faces.at(1).stats[0] &&
                            actual.faces.at(2).stats[0] == 0, std::string(label) + ": light and point seam coverage");
                        if (filter == VirtualShadowMapFilter::SMRT) {
                            requirePool(partialMasks > 0, "No partially covered directional receiver mask");
                        }
                        if (budget == 37) { requirePool(rejected > 0, "Missing rejected requests in pressured Mark comparison"); }
                        report["frames"].push_back({{"phase",label}, {"faces",faces.size()}, {"markDispatches",actual.commands.at("markDispatches")},
                            {"lightGrid",actual.commands.at("lightGrid")},
                            {"requestedFinePages",requests}, {"pointFacesWithRequests",requestedPointFaces},
                            {"rejectedPages",rejected}, {"partialReceiverMasks",partialMasks}});
                        save();
                    };
                    compare("MixedHard", VirtualShadowMapFilter::Hard, 256);
                    GPUScene gridScene;
                    gridScene.setDefaultFrameSlotCount(1);
                    const auto gridView = gridScene.createView();
                    std::vector<scene::PunctualLight> sources;
                    for (const auto& light : {spot, offscreen, point}) {
                        scene::PunctualLight source;
                        source.properties.type = light.type == VirtualShadowMapLightType::Spot ? "spot" : "point";
                        source.properties.intensity = 100;
                        source.properties.range = light.range;
                        source.properties.outerConeAngle = light.outerConeAngle;
                        source.position = {light.position[0], light.position[1], light.position[2]};
                        source.direction = {light.direction[0], light.direction[1], light.direction[2]};
                        sources.push_back(source);
                    }
                    requirePool(gridScene.syncLights({}, sources), "Publish Mark light sources");
                    RenderSubsystemHost host;
                    std::string gridLog;
                    requirePool(host.initialize(device, 1, gridLog), "Initialize Mark light grid host");
                    ClusterLightGrid grid;
                    const auto prepareGrid = [&](uint32_t capacity) {
                        ClusterLightGridDesc desc{.width = kWidth, .height = kHeight, .tileSize = 32, .depthSliceCount = 4,
                            .maxLightsPerCell = capacity, .eye = {0,0,4}, .center = {0,0,0}, .aspect = float(kWidth)/kHeight,
                            .zNear = .01f, .zFar = 10, .orthoHeight = 4};
                        requirePool(gridScene.prepareView(gridView, 0, {.width = kWidth, .height = kHeight}), "Prepare Mark grid view");
                        gpu.begin();
                        requirePool(host.beginFrame(gpu.index - 1, 0, nullptr, gridLog, &gpu.frame), "Begin Mark grid frame");
                        requirePool(grid.record(device, *gpu.commands, host, gridScene, gridView, 0, desc, gridLog), "Record Mark grid");
                        markGrid = grid.snapshot(gridScene);
                        requirePool(markGrid && markGrid->valid(), "Mark grid snapshot missing");
                        uint32_t center = 0;
                        requirePool(clusterLightGridCellIndex(markGrid->params, kWidth / 2, kHeight / 2, 4.f, center), "Mark grid center");
                        auto readback = device.createBuffer({.size = sizeof(ClusterLightGridCell), .usage = BufferUsageBits::TransferDestination,
                            .memoryLocation = MemoryLocation::HostReadback});
                        requirePool(readback, "Mark grid cell readback");
                        const BufferBarrierDesc ready{.buffer = markGrid->cells, .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                        requirePool(gpu.commands->synchronize({.buffers = {&ready, 1}}), "Mark grid cell ready");
                        auto from = markGrid->cells->slice({uint64_t(center) * sizeof(ClusterLightGridCell), sizeof(ClusterLightGridCell)});
                        auto to = (*readback)->slice();
                        requirePool(from, "Mark grid source cell"); requirePool(to, "Mark grid destination cell");
                        requirePool(gpu.commands->copyBuffer(*from, *to), "Read Mark grid center");
                        host.endFrame();
                        gpu.finish();
                        (*readback)->invalidate();
                        const auto* cell = static_cast<const ClusterLightGridCell*>((*readback)->map());
                        requirePool(cell && cell->totalCount >= 2 && (cell->overflow != 0) == (capacity == 1),
                            "Mark grid did not exercise the expected bounded-light or overflow path");
                        (*readback)->unmap();
                    };
                    prepareGrid(8);
                    compare("GriddedHard", VirtualShadowMapFilter::Hard, 256);
                    compare("CompactPCF", VirtualShadowMapFilter::PCF, 256);
                    prepareGrid(1);
                    compare("CompactSMRT", VirtualShadowMapFilter::SMRT, 37);
                    compare("CachedSMRT", VirtualShadowMapFilter::SMRT, 37);
                    continue;
                }
                if (hierarchyHZB) {
                    settings.clipmapCount = 2;
                    settings.instanceBuild = true;
                    settings.twoPassHZB = true;
                    settings.clusterPageRaster = true;
                    settings.collectGPUWork = true;
                    settings.receiverFootprint = false;
                    VirtualShadowMapLight sun;
                    sun.direction = {0, 0, -1};
                    VirtualShadowMapLight spot;
                    spot.type = VirtualShadowMapLightType::Spot;
                    // Keep the larger roof's perspective shadow inside the
                    // receiver's horizontal extent, with lit pixels on both sides.
                    spot.position = {0, 0, 8}; spot.direction = sun.direction;
                    spot.range = 12; spot.nearPlane = .05f; spot.outerConeAngle = 1.0f;
                    const std::array faces{PoolFace{0, sun}, PoolFace{1, spot}};
                    const auto compare = [&](const char* phase, uint32_t budget = 768) {
                        auto actual = record(faces, budget, false);
                        const auto baseline = record(faces, budget, true);
                        for (const auto& [identity, face] : actual.faces) {
                            const auto& hzb = face.hierarchyHZB;
                            const auto& ref = baseline.faces.at(identity);
                            report["frames"].push_back({{"phase", phase}, {"face", identity},
                                {"deferredInstances", hzb.deferredInstances}, {"deferredNodes", hzb.deferredNodes},
                                {"rejectedInstances", hzb.rejectedInstances}, {"rejectedNodes", hzb.rejectedNodes},
                                {"recoveredInstances", hzb.recoveredInstances}, {"recoveredNodes", hzb.recoveredNodes},
                                {"instanceClusterTickets", face.instanceClusterTickets},
                                {"referenceInstanceClusterTickets", ref.instanceClusterTickets}});
                            compareRasterDepth(actual, baseline, identity, 0.0f, false);
                            requirePool(hzb.deferredInstances == hzb.rejectedInstances + hzb.recoveredInstances &&
                                hzb.deferredNodes == hzb.rejectedNodes + hzb.recoveredNodes,
                                "Deferred instances or nodes were lost or tested twice");
                            requirePool(ref.hierarchyHZB.deferredInstances == 0 && ref.hierarchyHZB.deferredNodes == 0 &&
                                (identity == 0 || (hzb.deferredInstances == 0 && hzb.deferredNodes == 0)),
                                "Hierarchy culling ran on the reference or unsupported local-light view");
                            if (identity == 0 && std::string_view(phase) == "MoveUnderRoof") {
                                requirePool(ref.instanceClusterTickets > face.instanceClusterTickets,
                                    "Rejected hierarchy work still emitted the reference instance Build tickets");
                            }
                        }
                        return actual;
                    };
                    const auto roof = scene.nodes()[2].localMatrix;
                    const auto move = [&](uint32_t node, const float4x4& original, float x) {
                        float4x4 translation;
                        translation.SetupByTranslation({x, 0, 0});
                        requirePool(scene.setNodeLocalMatrix(node, translation * original), "Move hierarchy HZB caster");
                    };
                    const auto cold = compare("Cold");
                    for (uint32_t node : {1u, 3u, 4u, 5u, 6u}) { move(node, scene.nodes()[node].localMatrix, .015625f); }
                    move(2, roof, .015625f);
                    const auto hidden = compare("MoveUnderRoof");
                    move(2, roof, 2.5f);
                    const auto revealed = compare("RemoveOccluder");
                    const auto warm = compare("Warm");
                    const auto replaced = compare("ReplacePhysicalPool", 1024);
                    const auto& underRoof = hidden.faces.at(0).hierarchyHZB;
                    const auto& exposed = revealed.faces.at(0).hierarchyHZB;
                    requirePool(underRoof.rejectedInstances > 0 && underRoof.rejectedNodes > 0,
                        "Expected both an occluded instance and a fully occluded subtree to survive retest as occluded");
                    requirePool(exposed.recoveredInstances > 0 && exposed.recoveredNodes > 0,
                        "Disocclusion did not recover both an instance and a subtree in the current frame");
                    for (const auto* snapshot : {&cold, &warm, &replaced}) {
                        const auto& counts = snapshot->faces.at(0).hierarchyHZB;
                        requirePool(counts.deferredInstances == 0 && counts.deferredNodes == 0,
                            "A cold, clean or newly allocated pool used unavailable HZB history");
                    }
                    requireRetainedDepth(revealed, warm, 0);
                    const auto& exposedShadow = revealed.faces.at(0).visibility;
                    requirePool(exposedShadow != hidden.faces.at(0).visibility, "Removing roof did not uncover any receivers");
                    std::vector<uint32_t> pixels(exposedShadow.size());
                    std::ranges::transform(exposedShadow, pixels.begin(), [](uint8_t v) { return uint32_t(v) * 0x010101u | 0xff000000u; });
                    std::string log;
                    requirePool(saveRgba8Png(directory / "Revealed.png", reinterpret_cast<const uint8_t*>(pixels.data()),
                        kWidth, kHeight, log), "Save hierarchy HZB shadow: " + log);
                    save();
                    continue;
                }
                if (twoPass) {
                    settings.twoPassHZB = true;
                    settings.clusterPageRaster = true;
                    settings.clusterHybridRaster = hybridHZB;
                    settings.collectGPUWork = true;
                    settings.receiverFootprint = false;
                    settings.framesStaticThreshold = 120;
                    VirtualShadowMapLight sun;
                    sun.direction = {0, 0, -1};
                    VirtualShadowMapLight spot;
                    spot.type = VirtualShadowMapLightType::Spot;
                    spot.position = {0, 0, 4}; spot.direction = sun.direction;
                    spot.range = 12; spot.nearPlane = .05f; spot.outerConeAngle = 1.0f;
                    const std::array faces{PoolFace{0, sun}, PoolFace{1, spot}};
                    const auto activeFaces = std::span(faces).first(standaloneHZB ? 1u : faces.size());
                    const char* path = standaloneHZB ? "standalone" : serialHZB ? "sharedSerialLayered" : "sharedBatch";
                    const auto compare = [&](const char* phase, uint32_t budget = 512) {
                        const auto actual = record(activeFaces, budget, false, false, false, true, serialHZB, standaloneHZB);
                        const auto reference = record(activeFaces, budget, true, false, false, true, false, standaloneHZB);
                        for (const auto& [identity, face] : actual.faces) {
                            const auto& hzb = face.hzb;
                            report["frames"].push_back({{"phase",phase}, {"recording",path}, {"hybridRaster",hybridHZB}, {"face",identity},
                                {"deferred",hzb.deferred}, {"rejected",hzb.rejected}, {"recovered",hzb.recovered}});
                            compareRasterDepth(actual, reference, identity, 0.0f, identity == 0u);
                            requirePool(hzb.deferred == hzb.rejected + hzb.recovered,
                                "Deferred cluster/page tickets were lost or tested twice");
                        }
                        return actual;
                    };
                    const auto blocker = scene.nodes()[1].localMatrix;
                    const auto roof = scene.nodes()[2].localMatrix;
                    const auto move = [&](uint32_t node, const float4x4& original, float x) {
                        float4x4 translation;
                        translation.SetupByTranslation({x, 0, 0});
                        requirePool(scene.setNodeLocalMatrix(node, translation * original), "Move two-pass HZB fixture caster");
                    };
                    const auto cold = compare("Cold");
                    move(1, blocker, .125f);
                    // Promote both casters in the layered path. The composite now owns their depth,
                    // so the retained static roof cannot eliminate the blocker before the two-pass test.
                    if (serialHZB) { move(2, roof, .03125f); }
                    const auto hidden = compare("MoveUnderRoof");
                    move(2, roof, 2.0f);
                    const auto revealed = compare("RemoveOccluder");
                    const auto warm = compare("Warm");
                    for (const auto& [identity, face] : cold.faces) {
                        const auto& underRoof = hidden.faces.at(identity);
                        const auto& exposed = revealed.faces.at(identity);
                        requirePool(face.hzb.deferred == 0, "Cold frame tested stale HZB history");
                        if (serialHZB) {
                            requirePool(underRoof.cache[kVSMCacheDynamicCasters] >= 2 && underRoof.cache[kVSMCacheRestoredPages] > 0,
                                "Layered serial test did not promote both casters and restore static depth");
                        } else {
                            requirePool(underRoof.hzb.rejected > 0 && underRoof.visibility == face.visibility,
                                "Moving a fully occluded cluster did not exercise final HZB rejection");
                        }
                        requirePool(exposed.hzb.recovered > 0 && exposed.visibility != underRoof.visibility,
                            "Removing the occluder did not recover a deferred ticket and expose the receivers");
                        // The roof has left this central patch. The smaller blocker must still shadow some
                        // pixels, while adjacent receivers previously hidden by the roof are now lit.
                        uint32_t blockerPixels = 0, uncoveredPixels = 0;
                        for (uint32_t y = kHeight / 2 - 8; y <= kHeight / 2 + 8; ++y) {
                            for (uint32_t x = kWidth / 2 - 8; x <= kWidth / 2 + 8; ++x) {
                                const uint32_t pixel = y * kWidth + x;
                                blockerPixels += exposed.visibility[pixel] == 0;
                                uncoveredPixels += underRoof.visibility[pixel] == 0 && exposed.visibility[pixel] == 255;
                            }
                        }
                        requirePool(blockerPixels > 0 && uncoveredPixels > 0, "Missing exposed blocker or newly lit receiver");
                        requirePool(warm.faces.at(identity).hzb.deferred == 0 && warm.faces.at(identity).stats[17] > 0,
                            "Clean resident pages generated two-pass raster work");
                        requireRetainedDepth(revealed, warm, identity);
                    }
                    if (!standaloneHZB && !serialHZB) {
                        const auto newPool = compare("ReplacePhysicalPool", 768);
                        for (const auto& [identity, face] : newPool.faces) {
                            requirePool(face.hzb.deferred == 0 && face.stats[16] > 0 && face.stats[17] == 0,
                                "New physical pool reused stale history on face " + std::to_string(identity));
                        }
                    }
                    const auto& shadow = revealed.faces.at(0).visibility;
                    std::vector<uint32_t> pixels(shadow.size());
                    std::ranges::transform(shadow, pixels.begin(), [](uint8_t v) { return uint32_t(v) * 0x010101u | 0xff000000u; });
                    std::string log;
                    const char* imageName = standaloneHZB ? "RevealedStandalone.png" : serialHZB ? "RevealedSerialLayered.png" :
                        hybridHZB ? "RevealedHybrid.png" : "RevealedHardware.png";
                    requirePool(saveRgba8Png(directory / imageName,
                        reinterpret_cast<const uint8_t*>(pixels.data()), kWidth, kHeight, log), "Save two-pass HZB shadow: " + log);
                    requirePool(scene.setNodeLocalMatrix(1, blocker) && scene.setNodeLocalMatrix(2, roof),
                        "Restore two-pass HZB fixture");
                    save();
                    continue;
                }
                if (hybrid) {
                    static_assert(kVSMHybridHardwareTickets == kVSMHybridSoftwareTickets + 1 &&
                        kVSMHybridCoveredSamples == kVSMHybridHardwareTickets + 1);
                    settings.clusterPageRaster = true;
                    settings.clusterHybridRaster = true;
                    settings.collectGPUWork = true;
                    settings.receiverFootprint = false;
                    VirtualShadowMapLight light;
                    light.direction = {0, 0, -1};
                    const std::array faces{PoolFace{0, light}, PoolFace{1, light}};
                    const std::array single{PoolFace{2, light}};
                    auto compareCold = [&](const char* path, const PoolSnapshot& mixed, const PoolSnapshot& hardware) {
                        for (const auto& [identity, face] : mixed.faces) {
                            const auto& reference = hardware.faces.at(identity);
                            report["frames"].push_back({{"path", path}, {"layered", layered}, {"face", identity},
                                {"softwareTickets", face.hybrid[0]}, {"hardwareTickets", face.hybrid[1]}, {"coveredSamples", face.hybrid[2]},
                                {"hardwarePrimitives", face.hardwarePrimitives}, {"referencePrimitives", reference.hardwarePrimitives},
                                {"hardwareFineGroups", face.hardwareFineGroups}, {"referenceFineGroups", reference.hardwareFineGroups}});
                            compareRasterDepth(mixed, hardware, identity);
                            requirePool(face.hybrid[0] > 0 && face.hybrid[1] > 0 && face.hybrid[2] > 0,
                                "No mixed software/hardware fine tickets or covered samples on " + std::string(path));
                            requirePool(face.hardwarePrimitives > 0 && face.hardwarePrimitives < reference.hardwarePrimitives,
                                "Hybrid fine tickets were also emitted by the hardware mesh shader on " + std::string(path));
                            requirePool(face.hardwareFineGroups == face.hybrid[1] && face.hardwareFineGroups < reference.hardwareFineGroups,
                                "Packed mesh group count does not match the compact hardware ticket list on " + std::string(path));
                        }
                    };
                    const auto baseline = record(faces, 512, true);
                    const auto batch = record(faces, 512, false);
                    compareCold("batched", batch, baseline);
                    const auto& shadow = batch.faces.at(0).visibility;
                    constexpr size_t center = (kHeight / 2u) * kWidth + kWidth / 2u;
                    requirePool(shadow[center - 2] != shadow[center + 2],
                        "Opposite windings of the single-sided software blocker cast the same shadow");
                    compareCold("serial", record(faces, 512, false, false, false, true, true), baseline);
                    const auto warm = record(faces, 512, false);
                    for (const auto& [identity, face] : warm.faces) {
                        requirePool(face.hybrid == std::array<uint32_t, 3>{} && face.stats[17] > 0, "Clean frame reran hybrid raster");
                        compareRasterDepth(warm, baseline, identity);
                        requireRetainedDepth(batch, warm, identity);
                    }
                    const auto soloBaseline = record(single, 256, true, false, false, true, false, true);
                    const auto solo = record(single, 256, false, false, false, true, false, true);
                    compareCold("standalone", solo, soloBaseline);
                    const auto original = scene.nodes()[1].localMatrix;
                    auto translation = original;
                    // Align a blocker edge halfway between two raster subpixels at a texel center.
                    const float texelWorldSize = 2.0f * settings.firstClipmapExtent /
                        float(settings.virtualPagesPerAxis * settings.pageSize);
                    const float subpixel = std::ldexp(1.0f, -int(device.capabilities().subPixelPrecisionBits));
                    const float edgeOffset = 0.5f * (1.0f + subpixel) * texelWorldSize;
                    translation.SetupByTranslation({.75f + edgeOffset, -.5f - edgeOffset, 0.f});
                    requirePool(scene.setNodeLocalMatrix(1, translation * original), "Move software blocker");
                    const auto moved = record(faces, 512, false);
                    const auto movedHardware = record(faces, 512, true);
                    for (const auto& [identity, face] : moved.faces) {
                            compareRasterDepth(moved, movedHardware, identity);
                        requirePool(face.visibility != batch.faces.at(identity).visibility && face.hybrid[0] && face.hybrid[2],
                            "Moving the software blocker did not change the shadow using software samples");
                        if (layered) {
                            requirePool(face.cache[kVSMCacheStaticCasters] == 1 && face.cache[kVSMCacheDynamicCasters] == 1 &&
                                face.cache[kVSMCachePromotedCasters] == 1 && face.cache[kVSMCacheRestoredPages] > 0,
                                "Moving blocker did not exercise dynamic composition over restored static depth");
                        }
                        report["frames"].push_back({{"path","moved"}, {"layered",layered}, {"face",identity},
                            {"tickets",face.hybrid}, {"restoredStaticPages",face.cache[kVSMCacheRestoredPages]}});
                    }
                    for (bool hybrid : {false, true}) {
                        settings.clusterHybridRaster = hybrid;
                        const auto switched = record(faces, 512, false);
                        for (const auto& [identity, face] : switched.faces) {
                            compareRasterDepth(switched, movedHardware, identity);
                            requirePool((face.cache[kVSMCacheReasons] & kVSMCacheReasonSettings) != 0 &&
                                face.stats[16] > 0 && face.stats[17] == 0 && (face.hybrid[2] > 0) == hybrid,
                                "Changing the raster backend failed to redraw resident depth through the selected backend");
                            report["frames"].push_back({{"path",hybrid ? "enable" : "disable"}, {"layered",layered},
                                {"face",identity}, {"dirtyFinePages",face.stats[16]}, {"reusedFinePages",face.stats[17]}});
                        }
                    }
                    requirePool(scene.setNodeLocalMatrix(1, original), "Restore software blocker");
                    if (!layered) {
                        VirtualShadowMapLight spot;
                        spot.type = VirtualShadowMapLightType::Spot;
                        spot.position = {0, 0, 3};
                        spot.direction = {.2f, .1f, -1};
                        spot.range = 12; spot.nearPlane = .05f; spot.outerConeAngle = 1.0f;
                        const std::array local{PoolFace{3, spot}};
                        const auto spotHardware = record(local, 256, true);
                        const auto spotMixed = record(local, 256, false);
                        // The oblique spot produces a depth gradient even on the horizontal blocker.
                        // Check coverage and hard visibility exactly, allowing 2e-6 in normalized D32 only.
                        constexpr float depthTolerance = 2e-6f;
                        const float maxError = compareRasterDepth(spotMixed, spotHardware, 3, depthTolerance);
                        const auto& face = spotMixed.faces.at(3);
                        requirePool(face.hybrid[0] > 0 && face.hybrid[1] > 0 && face.hybrid[2] > 0 &&
                            face.hardwarePrimitives < spotHardware.faces.at(3).hardwarePrimitives,
                            "Oblique spotlight did not route nonempty single-sided work through software");
                        report["frames"].push_back({{"path","obliqueSpot"}, {"tickets",face.hybrid},
                            {"maxDepthError",maxError}, {"depthTolerance",depthTolerance}});

                        for (const auto& [label, mix] : {std::pair{"MixedLongEdge", HybridTriangleMix::LongEdge},
                                 std::pair{"MixedMasked", HybridTriangleMix::Masked}}) {
                            validation.phase(label);
                            const auto split = directory / label;
                            std::filesystem::create_directories(split);
                            requirePool(scene.load(writeClusterFixture(split, false, false, mix)), "Load hybrid lane fixture");
                            const auto splitHardware = record(single, 256, true);
                            const auto splitMixed = record(single, 256, false);
                            const auto& mixedFace = splitMixed.faces.at(single.front().identity);
                            const auto& routed = mixedFace.hybridRouting;
                            const auto& hardwareFace = splitHardware.faces.at(single.front().identity);
                            report["frames"].push_back({{"path", label}, {"tickets", mixedFace.hybrid},
                                {"mixedTickets", routed.mixedTickets}, {"softwareTriangles", routed.softwareTriangles},
                                {"hardwareTriangles", routed.hardwareTriangles}, {"hardwarePrimitives", mixedFace.hardwarePrimitives},
                                {"referencePrimitives", hardwareFace.hardwarePrimitives}});
                            compareRasterDepth(splitMixed, splitHardware, single.front().identity, 0.0f, false);
                            requirePool(routed.mixedTickets > 0 && routed.mixedTickets <= std::min(mixedFace.hybrid[0], mixedFace.hybrid[1]) &&
                                routed.softwareTriangles >= mixedFace.hybrid[0] && routed.hardwareTriangles >= mixedFace.hybrid[1],
                                "A ticket did not split its lanes between software and hardware on " + std::string(label));
                            requirePool(mixedFace.hybrid[2] > 0 && mixedFace.hardwareFineGroups == mixedFace.hybrid[1] &&
                                mixedFace.hardwarePrimitives < hardwareFace.hardwarePrimitives,
                                "Software lanes did not replace hardware primitives on " + std::string(label));
                        }
                        requirePool(scene.load(directory / "Clusters.gltf"), "Restore original hybrid fixture");
                    }
                    std::vector<uint32_t> pixels(shadow.size());
                    std::ranges::transform(shadow, pixels.begin(), [](uint8_t value) { return uint32_t(value) * 0x010101u | 0xff000000u; });
                    std::string log;
                    requirePool(saveRgba8Png(directory / (layered ? "LayeredShadow.png" : "CombinedShadow.png"),
                        reinterpret_cast<const uint8_t*>(pixels.data()), kWidth, kHeight, log), "Save hybrid output: " + log);
                    save();
                    continue;
                }
                auto compare = [&](const char* label, std::span<const PoolFace> faces, uint32_t budget,
                                   bool pointGroup = false, bool demandDriven = true) {
                    validation.phase(std::string(layered ? "Layered-" : "Combined-") + label);
                    report["activeCase"] = {{"name", label}, {"layered", layered}};
                    auto actual = record(faces, budget, false, false, pointGroup, demandDriven);
                    const auto expected = record(faces, budget, true, false, pointGroup, demandDriven);
                    const auto serialResult = record(faces, budget, false, false, pointGroup, demandDriven, true);
                    requirePool(actual.atlas == serialResult.atlas,
                        std::string("Batched shared D32 atlas differs from serialized rendering at ") + label);
                    Json entry{{"name", label}, {"layered", layered}, {"budget", budget}, {"faces", Json::array()},
                        {"batchBuildRaster", settings.batchBuildRaster}, {"commands", actual.commands},
                        {"compactBuild", settings.compactBuild}, {"instanceBuild", settings.instanceBuild},
                        {"dirtyPageHierarchy", settings.dirtyPageHierarchy}, {"pageWindowRaster", settings.pageWindowRaster},
                        {"collectGPUWork", settings.collectGPUWork},
                        {"redrawCommands", expected.commands}, {"serialCommands", serialResult.commands},
                        {"nativeCommands", actual.nativeCommands}, {"serialNativeCommands", serialResult.nativeCommands}};
                    for (const auto& [identity, face] : actual.faces) {
                        requirePool(face.visibility == expected.faces.at(identity).visibility,
                            std::string("Cached shared visibility differs from forced redraw at ") + label);
                        requirePool(face.visibility == serialResult.faces.at(identity).visibility &&
                            face.pages == serialResult.faces.at(identity).pages,
                            std::string("Batched visibility or page ownership differs from serialized rendering at ") + label);
                        entry["faces"].push_back({{"identity", identity}, {"stats", face.stats}, {"cache", face.cache}});
                    }
                    if (settings.batchBuildRaster) {
                        requirePool(actual.commands.at("renderingPasses") == (layered ? 2u : 1u),
                            "Shared batch did not use one rendering instance per depth layer");
                        requirePool(actual.commands.at("renderingPasses") < serialResult.commands.at("renderingPasses"),
                            "Batching did not reduce rendering instances across faces");
                        if (pointGroup || faces.size() == 8) {
                            requirePool(actual.commands.at("computeDispatches") < serialResult.commands.at("computeDispatches") &&
                                actual.commands.at("meshDraws") < serialResult.commands.at("meshDraws") &&
                                actual.nativeCommands.dispatchCalls < serialResult.nativeCommands.dispatchCalls &&
                                actual.nativeCommands.drawCalls < serialResult.nativeCommands.drawCalls &&
                                actual.nativeCommands.renderingScopes < serialResult.nativeCommands.renderingScopes,
                                "Compatible light faces were not submitted through shared compute and raster batches");
                        }
                    }
                    report["frames"].push_back(entry); save();
                    return actual;
                };
                VirtualShadowMapLight hot;
                hot.type = VirtualShadowMapLightType::Spot; hot.position = {0,0,3}; hot.direction = {0,0,-1};
                hot.outerConeAngle = .8f; hot.range = 10;
                auto idle = hot; idle.position = {100,0,3};
                std::vector<PoolFace> faces{{0, hot}, {1, idle}};
                auto cold = compare("cold-borrow", faces, 64);
                requirePool(cold.commands.at("sharedFrameBufferCreates") == 3, "Cold shared pool did not allocate its three frame buffers");
                if (layered) {
                    requirePool(cold.commands.at("casterLayerMaskCreates") > 0 &&
                        cold.commands.at("casterLayerMaskUploads") > 0 &&
                        cold.commands.at("casterLayerParameterCreates") > 0 &&
                        cold.commands.at("casterLayerParameterUploads") > cold.commands.at("casterLayerMaskUploads"),
                        "Cold static and dynamic caster layers did not share a mask");
                }
                requirePool(cold.faces.at(0).stats[0] > 32 && cold.faces.at(0).stats[12] > 32,
                    "Busy face did not borrow the idle face's unused fine capacity in its first frame");
                requirePool(cold.faces.at(1).stats[0] == 0 && cold.faces.at(1).stats[12] == 1,
                    "Out-of-view face should retain only coarse coverage");
                requirePool(std::count(cold.faces.at(0).visibility.begin(), cold.faces.at(0).visibility.end(), uint8_t(0)) > 0 &&
                    std::count(cold.faces.at(0).visibility.begin(), cold.faces.at(0).visibility.end(), uint8_t(255)) > 0,
                    "Pool fixture did not exercise lit and shadowed receivers");
                std::vector<uint32_t> pixels(kPixels);
                std::transform(cold.faces.at(0).visibility.begin(), cold.faces.at(0).visibility.end(), pixels.begin(),
                    [](uint8_t value) { return uint32_t(value) * 0x010101u | 0xff000000u; });
                std::string imageLog;
                requirePool(saveRgba8Png(directory / (layered ? "LayeredShadow.png" : "CombinedShadow.png"),
                    reinterpret_cast<const uint8_t*>(pixels.data()), kWidth, kHeight, imageLog), "Save pool shadow: " + imageLog);
                auto warm = compare("warm", faces, 64);
                requirePool(warm.faces.at(0).stats[17] > 0 && warm.faces.at(0).stats[20] == 0, "Warm shared pages were not reused");
                requireRetainedDepth(cold, warm, 0);
                // The first two frames fill both slots. The third reuses a completed slot
                // while the other slot still supplies the committed history.
                auto reused = compare("warm-buffer-reuse", faces, 64);
                requirePool(reused.commands.at("sharedFrameBufferCreates") == 0, "Warm shared pool recreated frame buffers");
                if (layered) {
                    for (const char* counter : {"casterLayerMaskCreates", "casterLayerParameterCreates",
                             "casterLayerMaskUploads", "casterLayerParameterUploads"}) {
                        requirePool(reused.commands.at(counter) == 0,
                            std::string("Warm resident caster layers rebuilt stable data: ") + counter);
                    }
                }
                requireRetainedDepth(warm, reused, 0);
                faces.push_back({2, idle});
                auto added = compare("cold-add", faces, 64);
                requirePool(added.faces.at(0).stats[17] > 0, "Adding another face discarded unrelated clean pages");
                requireRetainedDepth(warm, added, 0);
                std::reverse(faces.begin(), faces.end());
                auto reordered = compare("reorder", faces, 64);
                requireRetainedDepth(added, reordered, 0);
                faces.erase(faces.begin());
                auto removed = compare("remove", faces, 64);
                requireRetainedDepth(reordered, removed, 0);
                const std::vector<PoolFace> cancelled{{2, hot}, {1, idle}, {0, idle}};
                (void)record(cancelled, 37, false, true);
                auto retried = compare("cancel-retry", faces, 64);
                requirePool(retried.faces.at(0).visibility == removed.faces.at(0).visibility, "Cancelled pool recording changed committed visibility");
                requireRetainedDepth(removed, retried, 0);
                faces = {{0, idle}, {1, hot}};
                auto transferred = compare("same-frame-transfer", faces, 64);
                requirePool(transferred.faces.at(1).stats[12] > 32 && transferred.faces.at(0).stats[12] == 1,
                    "Newly busy face did not receive released physical capacity in the same frame");
                std::set<uint32_t> oldSlots;
                for (const auto& page : retried.faces.at(0).pages) { if ((page[1] & 2u) != 0) { oldSlots.insert(page[0]); } }
                uint32_t transferredSlots = 0;
                for (const auto& page : transferred.faces.at(1).pages) {
                    transferredSlots += (page[1] & 2u) != 0 && oldSlots.contains(page[0]);
                }
                requirePool(transferredSlots > 1, "Cross-light test never transferred a real physical page");
                (void)compare("shrink", faces, 19);
                (void)compare("grow", faces, 97);
                auto equal = compare("equal-grants", faces, 64, false, false);
                requirePool(equal.faces.at(1).stats[12] <= 32, "Equal-budget mode borrowed another face's logical grant");
                VirtualShadowMapLight directional;
                directional.direction = {1,0,-1};
                faces = {{0, directional}, {1, idle}};
                auto directionalCold = compare("directional-static-hzb-cold", faces, 64);
                auto directionalWarm = compare("directional-static-hzb-warm", faces, 64);
                requireRetainedDepth(directionalCold, directionalWarm, 0);
                std::vector<PoolFace> pointFaces;
                for (uint32_t face = 0; face < 6; ++face) {
                    auto light = hot; light.type = VirtualShadowMapLightType::PointFace; light.face = face;
                    pointFaces.push_back({3u + face, light});
                }
                (void)compare("point-six-cold", pointFaces, 64, true);
                (void)compare("point-six-warm", pointFaces, 64, true);
                faces = {{0, directional}, {1, hot}};
                faces.insert(faces.end(), pointFaces.begin(), pointFaces.end());
                (void)compare("mixed-eight-cold", faces, 256);
                auto mixedWarm = compare("mixed-eight-warm", faces, 256);
                settings.batchBuildRaster = false;
                auto switched = compare("mixed-eight-serial-switch", faces, 256);
                requirePool(switched.atlas == mixedWarm.atlas, "Switching to serial recording invalidated shared depth");
                settings.batchBuildRaster = true;
                auto restored = compare("mixed-eight-batch-switch", faces, 256);
                requirePool(restored.atlas == switched.atlas, "Switching back to batching invalidated shared depth");
                const auto originalMatrix = scene.nodes()[1].localMatrix;
                auto translated = originalMatrix;
                translated.SetupByTranslation({.25f, 0.f, .1f});
                requirePool(scene.setNodeLocalMatrix(1, translated * originalMatrix), "Move shared batched caster");
                auto moving = compare("mixed-eight-dynamic", faces, 256);
                if (layered) {
                    requirePool(moving.faces.at(0).cache[kVSMCacheDynamicCasters] == 1 &&
                        moving.faces.at(0).cache[kVSMCachePromotedCasters] == 1,
                        "Mixed batch did not promote the moving caster from static depth");
                }
                (void)compare("mixed-eight-settle", faces, 256);
                auto merged = compare("mixed-eight-static-merge", faces, 256);
                if (layered) {
                    requirePool(merged.faces.at(0).cache[kVSMCacheDynamicCasters] == 0 &&
                        merged.faces.at(0).cache[kVSMCacheStaticDirtyCoarsePages] == 1,
                        "Mixed batch did not merge the settled caster into static depth");
                }
                requirePool(scene.setNodeLocalMatrix(1, originalMatrix), "Restore shared batched caster");
                (void)compare("mixed-eight-restore", faces, 256);
                // Exercise indirect Build with real dirty work, first through
                // compact clusters and then directional instances/windows.
                // Local lights retain compact scalar raster in both variants,
                // so the second variant also mixes distinct compute/raster bins.
                settings.compactBuild = true;
                settings.dirtyPageHierarchy = true;
                settings.collectGPUWork = true;
                for (bool instanceWindow : {false, true}) {
                    settings.instanceBuild = instanceWindow;
                    settings.pageWindowRaster = instanceWindow;
                    // A new directional projection starts from a static base
                    // before promotion exercises static raster -> HZB -> dynamic.
                    directional.direction = {instanceWindow ? 1.25f : .75f, 0, -1};
                    faces[0].light = directional;
                    const auto branchCold = compare(instanceWindow ? "instance-window-hierarchy-cold" : "compact-hierarchy-cold", faces, 256);
                    const auto& coldFace = branchCold.faces.at(0);
                    requirePool(coldFace.stats[16] > 0 && coldFace.stats[4] > 0,
                        "Indirect batch cold case did not generate dirty fine raster work");
                    if (layered) {
                        requirePool(coldFace.cache[kVSMCacheStaticCasters] == 2 &&
                            coldFace.cache[kVSMCacheStaticHZBBuiltPages] > 0,
                            "Indirect batch cold case did not establish static depth and HZB");
                    }
                    auto branchTranslation = originalMatrix;
                    branchTranslation.SetupByTranslation({instanceWindow ? .35f : .2f, 0.f, .1f});
                    requirePool(scene.setNodeLocalMatrix(1, branchTranslation * originalMatrix), "Move indirect batch caster");
                    const auto branchDirty = compare(instanceWindow ? "instance-window-hierarchy-dirty" : "compact-hierarchy-dirty", faces, 256);
                    const auto& dirtyFace = branchDirty.faces.at(0);
                    requirePool(dirtyFace.stats[16] > 0 && dirtyFace.stats[4] > 0,
                        "Indirect batch motion case did not generate dirty fine raster work");
                    if (layered) {
                        requirePool(dirtyFace.cache[kVSMCachePromotedCasters] == 1 &&
                            dirtyFace.cache[kVSMCacheStaticCasters] == 1 && dirtyFace.cache[kVSMCacheDynamicCasters] == 1 &&
                            dirtyFace.cache[kVSMCacheStaticDirtyFinePages] > 0 && dirtyFace.cache[kVSMCacheStaticHZBBuiltPages] > 0 &&
                            dirtyFace.cache[kVSMCacheStaticDecodedTriangles] > 0 && dirtyFace.cache[kVSMCacheDynamicDecodedTriangles] > 0,
                            "Indirect batch did not execute nonempty static/HZB/dynamic composition after promotion");
                    }
                    requirePool(scene.setNodeLocalMatrix(1, originalMatrix), "Restore indirect batch caster");
                }
                settings.compactBuild = false;
                settings.instanceBuild = false;
                settings.dirtyPageHierarchy = false;
                settings.pageWindowRaster = false;
                settings.collectGPUWork = false;
                settings.framesStaticThreshold = 100;
                // Four-texel tiles keep this two-layer atlas small. Fine-only
                // receiver selection makes one face demand more than 4096
                // pages, exercising raster, projection and static HZB bank 1.
                settings.pageSize = 4; settings.virtualPagesPerAxis = 128; settings.clipmapCount = 1;
                settings.firstClipmapExtent = 3; settings.receiverFootprint = false;
                directional.direction = {0,0,-1};
                faces = {{0, directional}, {1, idle}};
                auto arrayCold = compare("array-layer-borrow", faces, 5000);
                requirePool(arrayCold.atlasLayers == 2 && arrayCold.faces.at(0).stats[12] > 4096,
                    "Large shared pool never granted a hot face more than 4096 physical pages");
                const auto& arrayPages = arrayCold.faces.at(0).pages;
                requirePool(std::any_of(arrayPages.begin(), arrayPages.end() - 1, [](const auto& page) {
                    return (page[1] & 2u) != 0 && page[0] >= 4096 && page[0] != UINT32_MAX;
                }), "Array test did not allocate a fine page in layer 1");
                const auto secondLayer = arrayCold.atlas.begin() + uint64_t(arrayCold.atlasWidth) * arrayCold.atlasHeight;
                requirePool(std::any_of(secondLayer, arrayCold.atlas.end(), [](uint32_t depth) { return depth != 0; }),
                    "Second atlas layer contains no rasterized depth");
                // Independent ground/blocker ray oracle for receivers whose
                // selected fine page is physically in layer 1. This also
                // catches an atlas.Load accidentally hard-coded to layer 0.
                uint32_t secondLayerReceivers = 0;
                for (uint32_t y = 0; y < kHeight; ++y) {
                    for (uint32_t x = 0; x < kWidth; ++x) {
                        const float worldX = ((float(x) + .5f) * 2.f / kWidth - 1.f) * 2.f * kWidth / kHeight;
                        const float worldY = (1.f - (float(y) + .5f) * 2.f / kHeight) * 2.f;
                        const uint32_t pageX = uint32_t(std::floor((.5f - worldX / 6.f) * 128.f));
                        const uint32_t pageY = uint32_t(std::floor((.5f + worldY / 6.f) * 128.f));
                        const auto& page = arrayPages[pageY * 128u + pageX];
                        if ((page[1] & 2u) == 0 || page[0] < 4096u) { continue; }
                        const bool blocked = std::abs(worldX) < .70f && std::abs(worldY) < .70f;
                        const bool lit = std::abs(worldX) > .80f || std::abs(worldY) > .80f;
                        if (!blocked && !lit) { continue; }
                        requirePool(arrayCold.faces.at(0).visibility[y * kWidth + x] == (blocked ? 0u : 255u),
                            "Second-layer projection disagrees with the independent ground/blocker ray oracle");
                        ++secondLayerReceivers;
                    }
                }
                requirePool(secondLayerReceivers > 10, "No useful receivers sampled second-layer fine pages");
                auto arrayWarm = compare("array-layer-warm", faces, 5000);
                requirePool(arrayWarm.faces.at(0).stats[17] > 4096 && arrayWarm.faces.at(0).stats[20] == 0,
                    "Large shared pool did not retain its second-layer fine pages");
                requireRetainedDepth(arrayCold, arrayWarm, 0);
                // The forced-redraw side must also write the correct array
                // layer through both packed mesh raster entrypoints.
                settings.clusterPageRaster = true;
                (void)compare("array-layer-cluster-raster", faces, 5000);
                settings.clusterPageRaster = false;
                settings.pageWindowRaster = true;
                (void)compare("array-layer-window-raster", faces, 5000);
            }
            report["validation"] = validation.snapshot();
            requirePool(!validation.failed(), "Shared pool Vulkan validation error");
            report["passed"] = true; save();
            return RHITestResult::pass(unifiedMark ? "View Mark matched per-face requests, receiver masks, cached pages and shadow images" :
                activeViews ? "Fine view compaction preserved cached pages, resident D32 and point visibility" :
                packedCoarse ? "Packed coarse raster preserved D32 and projected visibility on cache reuse and fine overflow" :
                distantLights ? "Distant local lights used one cached page per face and matched coarse-only depth and point filtering" :
                lightCacheRecovery ? "Returning lights reused clean depth and rebuilt casters or reclaimed physical pages" :
                casterChanges ? "Caster change lists were built once per snapshot pair and preserved per-face invalidation" :
                hierarchyHZB ? "Instance and hierarchy HZB preserved resident depth and shadow visibility through disocclusion" :
                twoPass ? "Two-pass HZB preserved resident depth and shadow visibility through disocclusion" :
                hybrid ? "Mixed cluster raster preserved resident D32 texels, visibility and cached pages" :
                "Batched and serialized shared rendering preserve depth, visibility, ownership and caches while reducing actual RHI submissions");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
private:
    PoolScenario scenario_;
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapPhysicalPoolTest);
class VirtualShadowMapPackedCoarseRasterTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapPackedCoarseRasterTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::PackedCoarseRaster) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapPackedCoarseRasterTest);
class VirtualShadowMapClusterHybridRasterTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapClusterHybridRasterTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::HybridRaster) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapClusterHybridRasterTest);
class VirtualShadowMapTwoPassHZBTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapTwoPassHZBTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::TwoPassHZB) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapTwoPassHZBTest);
class VirtualShadowMapHierarchyHZBTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapHierarchyHZBTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::HierarchyHZB) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapHierarchyHZBTest);
class VirtualShadowMapUnifiedMarkTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapUnifiedMarkTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::UnifiedMark) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapUnifiedMarkTest);
class VirtualShadowMapSharedCasterChangesTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapSharedCasterChangesTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::CasterChanges) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapSharedCasterChangesTest);
class VirtualShadowMapActiveViewsTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapActiveViewsTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::ActiveViews) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapActiveViewsTest);

class VirtualShadowMapDistantLightsTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapDistantLightsTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::DistantLights) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapDistantLightsTest);
class VirtualShadowMapLightCacheRecoveryTest final : public VirtualShadowMapPhysicalPoolTest
{
public:
    VirtualShadowMapLightCacheRecoveryTest() : VirtualShadowMapPhysicalPoolTest(PoolScenario::LightCacheRecovery) {}
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapLightCacheRecoveryTest);
} // namespace
} // namespace metallic::tests
