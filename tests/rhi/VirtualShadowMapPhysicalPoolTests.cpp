#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
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

std::filesystem::path writePoolFixture(const std::filesystem::path& directory)
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
    const auto doc = Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],
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
    const auto path = directory / "Pool.gltf";
    std::ofstream file(path); file << doc.dump(2);
    requirePool(file.good(), "Write pool fixture");
    return path;
}

struct PoolFace
{
    uint32_t identity;
    VirtualShadowMapLight light;
};

struct PoolFaceSnapshot
{
    std::array<uint32_t, 32> stats{};
    std::vector<Page> pages;
    std::vector<uint8_t> visibility;
};

struct PoolSnapshot
{
    std::map<uint32_t, PoolFaceSnapshot> faces;
    std::vector<uint32_t> atlas;
    uint32_t atlasWidth = 0, atlasHeight = 0, atlasLayers = 0, pagesX = 0, pageSize = 0;
    VirtualShadowMapMemoryUsage memory;
};

void requireUniqueOwners(const PoolSnapshot& snapshot, uint32_t budget)
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
        requirePool(face.stats[5] == 0u && face.stats[9] == 0u, "Shared raster overflow or uninitialized coarse query");
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

class VirtualShadowMapPhysicalPoolTest final : public RHITest
{
public:
    VirtualShadowMapPhysicalPoolTest() { name = "virtual_shadow_map_physical_pool"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapPhysicalPool");
        Json report{{"passed", false}, {"frames", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&] { std::filesystem::create_directories(directory); std::ofstream(directory / "Report.json") << report.dump(2); };
        try {
            scene::SceneDocument scene;
            requirePool(scene.load(writePoolFixture(directory)), "Load pool scene");
            auto created = createDevice({.applicationName = "VSM shared physical pool",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless"); }
            requirePool(created, "Pool device");
            auto& device = **created;
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

            for (bool layered : {false, true}) {
                VirtualShadowMapPhysicalPool cachedPool, referencePool;
                std::array<VirtualShadowMaps, 9> cached, reference;
                VirtualShadowMapCasterCache casterCache;
                VirtualShadowMapSettings settings;
                settings.pageSize = 16; settings.virtualPagesPerAxis = 16; settings.clipmapCount = 3;
                settings.cacheEnabled = true; settings.cacheLayering = layered;
                settings.firstClipmapExtent = 4; settings.maxWorkItems = 65535;
                auto record = [&](std::span<const PoolFace> specs, uint32_t budget, bool force, bool cancel = false,
                                  bool pointGroup = false, bool demandDriven = true) {
                    gpu.begin();
                    std::string log;
                    auto casters = casterCache.prepare(device, *gpu.commands, scene, nullptr, log);
                    requirePool(casters, "Prepare shared pool casters: " + log);
                    const uint32_t layers = pointGroup ? 1u : uint32_t(specs.size());
                    auto output = device.createTexture({.usage = TextureUsageBits::Storage | TextureUsageBits::TransferSource,
                        .format = Format::R8Unorm, .width = kWidth, .height = kHeight, .layerCount = layers});
                    requirePool(output, "Pool projection output");
                    const TextureBarrierDesc outputBegin{.texture = output->get(), .oldLayout = TextureLayout::Undefined,
                        .newLayout = TextureLayout::General, .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .range = {0, 1, 0, layers}};
                    requirePool(gpu.commands->synchronize({.textures = {&outputBegin, 1}}), "Pool output ready");
                    std::vector<VirtualShadowMapFaceRequest> requests;
                    for (uint32_t index = 0; index < specs.size(); ++index) {
                        requests.push_back({.shadow = &(force ? reference : cached)[specs[index].identity], .light = specs[index].light,
                            .output = {.shadow = output->get(), .layer = pointGroup ? 0u : index, .sharedPointLayer = pointGroup}});
                    }
                    auto frameSettings = settings;
                    frameSettings.forceRedraw = force;
                    frameSettings.filter = pointGroup ? VirtualShadowMapFilter::PCF : VirtualShadowMapFilter::Hard;
                    const auto view = camera.constants(gpu.index, kWidth, kHeight, kWidth, kHeight);
                    auto& pool = force ? referencePool : cachedPool;
                    auto results = VirtualShadowMaps::recordShared(pool, device, *gpu.commands, **depthView, view, requests,
                        *casters, scene.materials(), {}, frameSettings, budget, demandDriven, log);
                    requirePool(results, "Record shared pool: " + log);
                    std::vector<VirtualShadowMapProjectionGroup> groups(pointGroup ? 1 : specs.size());
                    for (uint32_t index = 0; index < specs.size(); ++index) {
                        requirePool((*results)[index].atlas == results->front().atlas, "Faces allocated independent physical atlases");
                        auto& group = groups[pointGroup ? 0u : index];
                        const uint32_t face = pointGroup ? index : 0u;
                        group.faces[face] = requests[index].shadow;
                        group.results[face] = (*results)[index];
                        group.faceCount = pointGroup ? 6u : 1u;
                        group.filteredPoint = pointGroup;
                    }
                    requirePool(VirtualShadowMaps::projectShared(*gpu.commands, groups, log), "Project shared pool: " + log);
                    PoolSnapshot snapshot;
                    snapshot.memory = pool.memoryUsage();
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
                        bytes += sizeof(face.stats) + face.pages.size() * sizeof(Page);
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
                        for (auto* buffer : {result.statistics, result.pageTable}) {
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
                        for (const auto& [buffer, size] : {std::pair{(*results)[index].statistics, uint64_t(sizeof(face.stats))},
                            std::pair{(*results)[index].pageTable, uint64_t(face.pages.size() * sizeof(Page))}}) {
                            auto from = buffer->slice({0, size}), to = (*readback)->slice({offset, size});
                            requirePool(from, "Pool readback source"); requirePool(to, "Pool readback destination");
                            requirePool(gpu.commands->copyBuffer(*from, *to), "Pool buffer readback");
                            offset += size;
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
                        const uint64_t offset = uint64_t(pointGroup ? 0u : index) * kPixels;
                        face.visibility.assign(data + offset, data + offset + kPixels);
                    }
                    (*readback)->unmap();
                    requireUniqueOwners(snapshot, budget);
                    requirePool((snapshot.memory.staticAtlasBytes != 0) == layered, "Layered shared atlas memory missing or double counted");
                    return snapshot;
                };
                auto compare = [&](const char* label, std::span<const PoolFace> faces, uint32_t budget,
                                   bool pointGroup = false, bool demandDriven = true) {
                    validation.phase(std::string(layered ? "Layered-" : "Combined-") + label);
                    report["activeCase"] = {{"name", label}, {"layered", layered}};
                    auto actual = record(faces, budget, false, false, pointGroup, demandDriven);
                    const auto expected = record(faces, budget, true, false, pointGroup, demandDriven);
                    Json entry{{"name", label}, {"layered", layered}, {"budget", budget}, {"faces", Json::array()}};
                    for (const auto& [identity, face] : actual.faces) {
                        requirePool(face.visibility == expected.faces.at(identity).visibility,
                            std::string("Cached shared visibility differs from forced redraw at ") + label);
                        entry["faces"].push_back({{"identity", identity}, {"stats", face.stats}});
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
            return RHITestResult::pass("Shared physical ownership, same-frame borrowing, retention, cancellation, resize and six-face projection match forced redraw");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapPhysicalPoolTest);
} // namespace
} // namespace metallic::tests
