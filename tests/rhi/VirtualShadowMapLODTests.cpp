#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLOD.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
constexpr uint32_t kWidth = 193, kHeight = 157, kPixels = kWidth * kHeight;

void check(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T> void check(const Result<T>& result, const std::string& message)
{
    check(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct LODCommands
{
    RenderFrameContext frame;
    QueueSubmissionTracker tracker;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    uint64_t index = 0;

    explicit LODCommands(Device& device)
    {
        auto& queue = *device.getQueue(QueueType::Graphics);
        check(tracker.initialize(device, queue), "LOD submission tracker");
        check(device.createCommandPool(queue).transform([&](auto p) { pool = std::move(p); }), "LOD pool");
        check(pool->createCommandBuffer().transform([&](auto c) { commands = std::move(c); }), "LOD commands");
    }
    ~LODCommands()
    {
        frame.cancel();
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        (void)pool->reset();
        (void)frame.reset();
    }
    void begin()
    {
        check(frame.begin(index++), "LOD frame begin");
        check(pool->reset(), "LOD pool reset");
        check(commands->begin(frame.submissionContext()), "LOD command begin");
    }
    void submit(Semaphore* gate = nullptr)
    {
        check(commands->end(), "LOD command end");
        CommandBuffer* list[]{commands.get()};
        const SemaphoreSubmitDesc wait{.semaphore = gate, .value = 1};
        check(tracker.submit({.waitSemaphores = {gate ? &wait : nullptr, gate ? 1u : 0u},
            .commandBuffers = list}, frame), "LOD submit");
    }
    void finish(bool cancel = false)
    {
        if (cancel) {
            check(commands->end(), "LOD cancelled command end");
            frame.cancel();
            return;
        }
        submit();
        check(frame.wait(), "LOD wait");
    }
};

void verifyMetricAndCoverage()
{
    GPUSceneGPUInstanceRecord instance;
    instance.worldMatrix = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    instance.identity[3] = GPUSceneGPUInstanceVisible;
    MeshletLODGroupRecord group{.sphere = {0,0,5,.2f}, .error = .01f};
    VirtualShadowMapLODProjection projection{.rowX = {.5f,0,0,0}, .rowY = {0,.5f,0,0}, .resolution = 256};
    const auto error = [&] { return virtualShadowMapLodPixelError(group, instance, {&projection, 1}); };
    check(std::abs(error() - .64f) < 1e-5f, "Orthographic texel metric does not match analytic error");
    instance.worldMatrix[4] = 2;
    check(error() >= .64f * 2.4142f, "Shear metric underestimates largest affine singular value");
    instance.worldMatrix[4] = 0;
    projection.rowX = {1,0,0,0}; projection.rowY = {0,1,0,0}; projection.rowW = {0,0,1,0}; projection.nearPlane = .1f;
    const float nearError = error(); group.sphere[2] = 50;
    check(error() < nearError * .11f && error() > 0, "Perspective distance did not reduce shadow texel error");
    group.sphere[2] = .2f;
    check(error() == std::numeric_limits<float>::max(), "Near-plane crossing must refine conservatively");
    group.sphere[2] = -5;
    check(error() == 0, "Entirely behind-light group requested refinement");
    group.sphere = {100,0,5,.2f};
    check(error() == 0, "Entirely outside-face group requested refinement");
    group.flags = kMeshletLODTerminalGroup; group.error = std::numeric_limits<float>::max();
    const auto terminal = buildVirtualShadowMapLodRequirements({&group, 1}, instance, {&projection, 1}, 1);
    check(terminal.needsFine == std::vector<uint8_t>{1} && terminal.maxProjectedError == 0,
        "Terminal stop sentinel polluted the projected error or omitted fallback");

    // Shared descendants require every parent before refinement. Coverage atoms
    // exercise the shadow override rather than borrowing the raster camera cut.
    std::vector<MeshletLODGroupRecord> groups;
    for (uint32_t i = 0; i < 6; ++i) {
        groups.push_back({.sphere = {0,0,5,1}, .error = .1f, .level = i < 3 ? 0u : i < 5 ? 1u : 2u,
            .flags = i == 2 || i == 5 ? kMeshletLODTerminalGroup : 0u});
    }
    const std::array<MeshletLODGroupRange, 6> ranges{{{0,2},{2,2},{4,1},{5,2},{7,1},{8,2}}};
    const std::array<uint32_t, 10> refined{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,0,1,0,3,4};
    const std::array<uint32_t, 10> coverage{1,2,4,8,16,1,12,2,13,2};
    const std::array<uint8_t, 6> wanted{1,1,0,1,1,0};
    std::vector<MeshletLODBVHNode> bvh;
    std::string reason;
    check(buildMeshletLodBvh(groups, bvh, reason), "Shadow reference BVH: " + reason);
    MeshletLODView raster;
    raster.eye = {10000,10000,10000,.1f}; raster.projection[3] = 1e9f;
    for (uint32_t mask = 0; mask < 16; ++mask) {
        std::array<uint8_t, 6> drawable{uint8_t(mask & 1),uint8_t((mask >> 1) & 1),1,
            uint8_t((mask >> 2) & 1),uint8_t((mask >> 3) & 1),1};
        const auto cut = selectStreamMeshletLodReference(groups, ranges, refined, drawable, instance, raster,
            UINT32_MAX, UINT32_MAX, {}, bvh, wanted);
        check(cut.valid, "Shadow override produced an invalid DAG cut: " + cut.reason);
        uint32_t covered = 0;
        for (uint32_t cluster : cut.selectedClusters) {
            check((covered & coverage[cluster]) == 0, "Shadow override duplicated surface coverage");
            covered |= coverage[cluster];
        }
        check(covered == 31, "Shadow override omitted surface coverage");
        if (mask == 15) { check(cut.selectedClusters == std::vector<uint32_t>{0,1,2,3,4}, "Raster BVH pruned shadow refinement"); }
    }
}

std::filesystem::path writeBunnyFixture(const std::filesystem::path& directory)
{
    std::filesystem::create_directories(directory);
    const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny";
    std::ifstream input(source / "scene.gltf");
    Json document; input >> document;
    std::filesystem::copy_file(source / "scene.bin", directory / "scene.bin", std::filesystem::copy_options::overwrite_existing);
    // Preserve the genuine cooked multi-level Bunny, enlarged to world meters.
    document["nodes"][0]["matrix"] = {20,0,0,0, 0,20,0,0, 0,0,20,0, .3368,-2.203,1.3,1};
    const float positions[]{-6,-5,0, 7,-5,0, 7,5,0, -6,5,0};
    const uint32_t indices[]{0,1,2,0,2,3};
    std::ofstream binary(directory / "Plane.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(positions), sizeof(positions));
    binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
    check(binary.good(), "Write shadow LOD receiver plane"); binary.close();
    const auto buffer = document["buffers"].size(), view = document["bufferViews"].size(), accessor = document["accessors"].size();
    document["buffers"].push_back({{"uri","Plane.bin"},{"byteLength",sizeof(positions) + sizeof(indices)}});
    document["bufferViews"].push_back({{"buffer",buffer},{"byteOffset",0},{"byteLength",sizeof(positions)}});
    document["bufferViews"].push_back({{"buffer",buffer},{"byteOffset",sizeof(positions)},{"byteLength",sizeof(indices)}});
    document["accessors"].push_back({{"bufferView",view},{"componentType",5126},{"count",4},{"type","VEC3"},{"min",{-6,-5,0}},{"max",{7,5,0}}});
    document["accessors"].push_back({{"bufferView",view + 1},{"componentType",5125},{"count",6},{"type","SCALAR"}});
    const auto mesh = document["meshes"].size(), node = document["nodes"].size();
    document["meshes"].push_back({{"primitives",Json::array({{{"attributes",{{"POSITION",accessor}}},{"indices",accessor + 1}}})}});
    document["nodes"].push_back({{"mesh",mesh},{"name","Shadow receiver"}});
    document["scenes"][0]["nodes"].push_back(node);
    const auto path = directory / "scene.gltf";
    std::ofstream(path) << document.dump(2);
    return path;
}

struct LODCapture
{
    std::vector<uint8_t> visibility;
    std::vector<uint32_t> atlas;
    std::vector<std::array<uint32_t, 4>> pages;
    std::array<uint32_t, 32> statistics{}, diagnostics{};
    uint32_t atlasWidth = 0, physicalPagesX = 0;
    Json selection;
};

uint32_t fullyResidentFinestClusterCount(const scene::MeshletStreamAsset& asset)
{
    uint32_t count = 0;
    for (const auto& sourceInstance : asset.instances()) {
        const auto& primitive = asset.primitives()[sourceInstance.primitiveIndex];
        std::vector<MeshletLODGroupRecord> groups;
        std::vector<MeshletLODGroupRange> ranges;
        std::vector<uint32_t> refined;
        for (uint32_t local = 0; local < primitive.groupCount; ++local) {
            const auto& source = asset.groups()[primitive.groupOffset + local];
            groups.push_back({.level = source.lodLevel, .flags = source.flags});
            ranges.push_back({uint32_t(refined.size()), source.clusterCount});
            for (uint32_t cluster = 0; cluster < source.clusterCount; ++cluster) {
                const uint32_t child = asset.refinedGroups()[source.clusterRefinedOffset + cluster];
                refined.push_back(child == UINT32_MAX ? child : child - primitive.groupOffset);
            }
        }
        const std::vector<uint8_t> available(groups.size(), 1);
        GPUSceneGPUInstanceRecord instance;
        instance.worldMatrix = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        instance.identity[3] = GPUSceneGPUInstanceVisible;
        const auto cut = selectStreamMeshletLodReference(groups, ranges, refined, available, instance,
            MeshletLODView{}, UINT32_MAX, UINT32_MAX, {}, {}, available);
        check(cut.valid, "Fully resident finest CPU reference is invalid: " + cut.reason);
        count += uint32_t(cut.selectedClusters.size());
    }
    return count;
}

Json compareSameCut(const LODCapture& actual, const LODCapture& reference, uint32_t pageSize)
{
    check(actual.visibility == reference.visibility, "Same-cut cached/forced visibility differs");
    check(actual.pages.size() == reference.pages.size(), "Same-cut page table sizes differ");
    Json diagnostics{{"comparedDepthTexels",0},{"unadmittedValidityDifferences",Json::array()}};
    uint64_t comparedDepthTexels = 0;
    for (size_t page = 0; page < actual.pages.size(); ++page) {
        const auto& a = actual.pages[page]; const auto& b = reference.pages[page];
        const std::string detail = " at virtual page " + std::to_string(page) + ": actual=" + Json(a).dump() + ", reference=" + Json(b).dump();
        check((a[1] & 3u) == (b[1] & 3u), "Same-cut page demand/admission differs" + detail);
        if ((a[1] & 2u) == 0) {
            // Reset preserves clean resident pages even when Mark no longer
            // requests them. ForceRedraw dirties those pages; Args redraws only
            // admitted pages. Projection never queries unadmitted mappings.
            if ((a[3] & 1u) != (b[3] & 1u)) {
                diagnostics["unadmittedValidityDifferences"].push_back({{"page",page},{"actual",a},{"reference",b}});
            }
            continue;
        }
        check((a[3] & 1u) == (b[3] & 1u), "Same-cut admitted page validity differs" + detail);
        if ((a[3] & 1u) == 0) { continue; }
        check(a[0] < actual.atlas.size() / (pageSize * pageSize) && b[0] < reference.atlas.size() / (pageSize * pageSize),
            "Same-cut admitted valid page has no physical depth mapping" + detail);
        for (uint32_t y = 0; y < pageSize; ++y) { for (uint32_t x = 0; x < pageSize; ++x) {
            const auto texel = [&](const LODCapture& capture, uint32_t physical) {
                return size_t(physical / capture.physicalPagesX * pageSize + y) * capture.atlasWidth + physical % capture.physicalPagesX * pageSize + x;
            };
            check(actual.atlas[texel(actual, a[0])] == reference.atlas[texel(reference, b[0])],
                "Same-cut cached/forced D32 differs" + detail + ", texel=" + std::to_string(x) + "," + std::to_string(y));
            ++comparedDepthTexels;
        }}
    }
    diagnostics["comparedDepthTexels"] = comparedDepthTexels;
    return diagnostics;
}

Json saveQuality(const std::filesystem::path& directory, const std::string& name,
    const LODCapture& capture, const LODCapture& reference)
{
    std::vector<uint8_t> image(kPixels * 4), difference(kPixels * 4);
    uint64_t absoluteError = 0, changed = 0, shadowed = 0;
    uint32_t maximum = 0;
    for (uint32_t i = 0; i < kPixels; ++i) {
        const auto delta = uint32_t(std::abs(int(capture.visibility[i]) - int(reference.visibility[i])));
        absoluteError += delta; maximum = std::max(maximum, delta); changed += delta != 0; shadowed += capture.visibility[i] < 255;
        for (uint32_t c = 0; c < 3; ++c) { image[i * 4 + c] = capture.visibility[i]; difference[i * 4 + c] = uint8_t(delta); }
        image[i * 4 + 3] = difference[i * 4 + 3] = 255;
    }
    std::string message;
    check(saveRgba8Png(directory / (name + ".png"), image.data(), kWidth, kHeight, message), "Write LOD visibility: " + message);
    check(saveRgba8Png(directory / (name + "-difference.png"), difference.data(), kWidth, kHeight, message), "Write LOD difference: " + message);
    return {{"changedPixels",changed},{"meanAbsoluteVisibilityError",double(absoluteError) / (kPixels * 255.)},
        {"maxVisibilityError",maximum},{"shadowedPixels",shadowed},{"visibilitySHA256",debug::debugSha256(capture.visibility)}};
}


LODCapture readCapture(Device& device, LODCommands& gpu, const VirtualShadowMapResult& result,
    const VirtualShadowMapSettings& settings, bool cancel)
{
    const uint32_t faceCount = 1;
    const std::array results{result};
    std::vector<LODCapture> captures(faceCount);
    std::vector<std::unique_ptr<Buffer>> readbacks;
    const uint64_t tableBytes = uint64_t(settings.clipmapCount * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1) * 16u;
    for (uint32_t face = 0; face < faceCount; ++face) {
        const auto& shadow = results[face];
        auto& capture = captures[face];
        const auto atlasDesc = shadow.atlas->desc();
        capture.atlasWidth = atlasDesc.width; capture.physicalPagesX = shadow.physicalPagesX;
        capture.atlas.resize(size_t(atlasDesc.width) * atlasDesc.height); capture.pages.resize(tableBytes / 16u);
        capture.visibility.resize(kPixels);
        const uint64_t atlasBytes = capture.atlas.size() * sizeof(uint32_t);
        const uint64_t statsOffset = atlasBytes + tableBytes, diagnosticsOffset = statsOffset + sizeof(capture.statistics);
        const uint64_t visibilityOffset = diagnosticsOffset + sizeof(capture.diagnostics);
        auto readback = device.createBuffer({.size = visibilityOffset + kPixels, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback});
        check(readback, "LOD readback");
        const TextureBarrierDesc textures[]{
            {.texture = shadow.atlas, .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::DepthStencilWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
            {.texture = shadow.texture, .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
        const BufferBarrierDesc buffers[]{
            {.buffer = shadow.pageTable, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
            {.buffer = shadow.statistics, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
            {.buffer = shadow.cacheDiagnostics, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
        check(gpu.commands->synchronize({.textures = textures, .buffers = buffers}), "LOD readback barriers");
        check((*readback)->slice({0, atlasBytes}).and_then([&](const auto& slice) {
            return gpu.commands->copyTextureToBuffer({.texture = shadow.atlas, .buffer = slice, .width = atlasDesc.width, .height = atlasDesc.height});
        }), "LOD atlas copy");
        check((*readback)->slice({visibilityOffset, kPixels}).and_then([&](const auto& slice) {
            return gpu.commands->copyTextureToBuffer({.texture = shadow.texture, .buffer = slice, .width = kWidth, .height = kHeight});
        }), "LOD visibility copy");
        for (const auto& [source, offset, bytes] : {std::tuple{shadow.pageTable, atlasBytes, tableBytes},
            std::tuple{shadow.statistics, statsOffset, uint64_t(sizeof(capture.statistics))},
            std::tuple{shadow.cacheDiagnostics, diagnosticsOffset, uint64_t(sizeof(capture.diagnostics))}}) {
            auto from = source->slice({0, bytes}), to = (*readback)->slice({offset, bytes});
            check(from, "LOD buffer source"); check(to, "LOD buffer destination");
            check(gpu.commands->copyBuffer(*from, *to), "LOD buffer copy");
        }
        const TextureBarrierDesc restore[]{
            {.texture = shadow.atlas, .oldLayout = TextureLayout::TransferSource, .newLayout = TextureLayout::ShaderRead,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}},
            {.texture = shadow.texture, .oldLayout = TextureLayout::TransferSource, .newLayout = TextureLayout::ShaderRead,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}}};
        check(gpu.commands->synchronize({.textures = restore}), "Restore LOD atlas");
        readbacks.push_back(std::move(*readback));
    }
    gpu.finish(cancel);
    if (cancel) { return captures[0]; }
    for (uint32_t face = 0; face < faceCount; ++face) {
        auto& capture = captures[face];
        const uint64_t atlasBytes = capture.atlas.size() * sizeof(uint32_t);
        const uint64_t statsOffset = atlasBytes + tableBytes, diagnosticsOffset = statsOffset + sizeof(capture.statistics);
        const uint64_t visibilityOffset = diagnosticsOffset + sizeof(capture.diagnostics);
        readbacks[face]->invalidate(); const auto* bytes = static_cast<const std::byte*>(readbacks[face]->map());
        check(bytes != nullptr, "Map LOD capture");
        std::memcpy(capture.atlas.data(), bytes, atlasBytes); std::memcpy(capture.pages.data(), bytes + atlasBytes, tableBytes);
        std::memcpy(capture.statistics.data(), bytes + statsOffset, sizeof(capture.statistics));
        std::memcpy(capture.diagnostics.data(), bytes + diagnosticsOffset, sizeof(capture.diagnostics));
        std::memcpy(capture.visibility.data(), bytes + visibilityOffset, kPixels); readbacks[face]->unmap();
    }
    return captures[0];
}

class VirtualShadowMapLODTest final : public RHITest
{
public:
    VirtualShadowMapLODTest() { name = "virtual_shadow_map_shadow_lod"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapLOD");
        std::filesystem::create_directories(directory);
        Json report{{"passed",false},{"states",Json::array()},{"streamWarmup",Json::array()},
            {"scope","Correctness and workload counts; no GPU timing or performance claim"},
            {"quality","Changed-cut hard-shadow differences are measured, not assumed bit-identical or bounded by geometry texel error"}};
        bench::ValidationRecorder validation;
        const auto save = [&] { std::ofstream(directory / "VirtualShadowMapLOD.json") << report.dump(2); };
        try {
            verifyMetricAndCoverage(); report["metricAndDAG"] = "passed"; save();
            const auto path = writeBunnyFixture(directory / "Fixture");
            const auto streamPath = directory / "Bunny.meshstream.bin";
            std::string log;
            check(scene::buildMeshletStreamAssetOffline({.sourcePath = path, .outputPath = streamPath,
                .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), "Cook shadow LOD Bunny: " + log);
            scene::MeshletStreamAsset asset;
            check(asset.open(streamPath, log), "Open shadow LOD Bunny: " + log);
            check(asset.lodLevels().size() > 1 && asset.groups().size() > asset.terminalGroups().size(), "Bunny did not produce genuine multi-level geometry");
            const uint32_t expectedFinestClusters = fullyResidentFinestClusterCount(asset);
            report["asset"] = {{"pages",asset.pageCount()},{"groups",asset.groups().size()},{"levels",asset.lodLevels().size()}};
            report["asset"]["fullyResidentFinestClusters"] = expectedFinestClusters;
            auto deviceResult = createDevice({.applicationName = "VSM shadow LOD",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(deviceResult, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless"); }
            check(deviceResult, "Shadow LOD device"); auto& device = **deviceResult;
            LODCommands gpu(device);
            std::unique_ptr<Streamer> streamer;
            check(createStreamer(device, {.constantBufferSize = 1u << 20}).transform([&](auto s) { streamer = std::move(s); }), "LOD streamer");
            std::unique_ptr<Texture> depth; std::unique_ptr<TextureView> depthView;
            check(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::D32Sfloat, .width = kWidth, .height = kHeight})
                .transform([&](auto t) { depth = std::move(t); }), "LOD receiver depth");
            check(device.createTextureView(*depth, {.format = Format::D32Sfloat})
                .transform([&](auto v) { depthView = std::move(v); }), "LOD receiver view");
            auto upload = device.createBuffer({.size = kPixels * sizeof(float), .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload});
            check(upload, "LOD receiver upload");
            auto* values = static_cast<float*>((*upload)->map()); check(values != nullptr, "Map LOD receiver");
            std::fill(values, values + kPixels, (20.f - 8.f) / (20.f - .01f)); (*upload)->flush(); (*upload)->unmap();
            gpu.begin();
            const TextureBarrierDesc begin{.texture = depth.get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
            check(gpu.commands->synchronize({.textures = {&begin, 1}}), "LOD depth upload barrier");
            check((*upload)->slice().and_then([&](const auto& slice) {
                return gpu.commands->copyBufferToTexture({.texture = depth.get(), .buffer = slice, .width = kWidth, .height = kHeight});
            }), "LOD receiver copy");
            const TextureBarrierDesc ready{.texture = depth.get(), .oldLayout = TextureLayout::TransferDestination,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
            check(gpu.commands->synchronize({.textures = {&ready, 1}}), "LOD receiver ready"); gpu.finish();

            for (bool streamed : {false, true}) {
                const std::string label = streamed ? "Stream" : "Resident";
                scene::SceneDocument scene;
                check(streamed ? scene.loadStreamMetadata(path) : scene.load(path), "Load LOD scene");
                std::shared_ptr<MeshletStreamRuntime> runtime;
                if (streamed) {
                    runtime = std::make_shared<MeshletStreamRuntime>();
                    check(runtime->initialize(device, {.sourcePath = path, .streamAssetPath = streamPath,
                        .maxResidentBytes = 64ull << 20, .maxResidentPages = 128, .maxLockedFallbackPages = 128,
                        .maxGpuPageRequests = 1024, .maxGpuPageUnloadRequests = 1024, .maxActiveGroups = 4096,
                        .maxRasterCandidates = 4096, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
                        .pageLoadConcurrency = 0, .queuedFrameCount = 1}, log), "LOD runtime: " + log);
                    check(runtime->syncRuntimeScene(scene, log), "LOD stream mapping: " + log);
                    for (uint32_t attempt = 0; attempt < 32 && !runtime->sceneReady(); ++attempt) {
                        gpu.begin(); check(streamer->beginFrame(gpu.frame), "LOD initial frame");
                        check(runtime->cmdLoadInitialResources(*gpu.commands, *streamer), "LOD initial load");
                        gpu.finish(); streamer->endFrame(); runtime->prepareMaintenance();
                    }
                    check(runtime->sceneReady(), "LOD terminal pages never became ready");
                }
                RenderView camera;
                const auto setCamera = [&](bool offscreen) {
                    const float x = offscreen ? 4.5f : 1.5f;
                    check(camera.setCamera({.eye = {x,0,8}, .center = {x,0,0}, .nearPlane = .01f,
                        .farPlane = 20, .orthoHeight = offscreen ? 3.f : 8.f, .orthographic = true}), "LOD camera");
                };
                setCamera(false);
                VirtualShadowMapLight light; light.direction = {2,0,-1};
                VirtualShadowMapSettings settings;
                settings.pageSize = 32; settings.virtualPagesPerAxis = 8; settings.clipmapCount = 3;
                settings.firstClipmapExtent = 4; settings.physicalPageCount = 256;
                settings.cacheEnabled = true; settings.collectGPUWork = true; settings.maxWorkItems = 1u << 20;
                VirtualShadowMapCasterSettings casterSettings;
                VirtualShadowMapCasterCache casterCache;
                std::array<VirtualShadowMaps, 4> maps;
                const auto selectionJSON = [](const VirtualShadowMapCasterResources& selected) {
                    const auto& d = selected.shadowLOD;
                    return Json{{"selectedClusters",d.selectedClusters},{"triangleSlots",d.selectedTriangleSlots},
                        {"requestedPages",d.requestedPages},{"blockedPages",d.blockedPages},{"fallbackCasters",d.fallbackCasters},
                        {"protectedCasters",d.protectedCasters},{"testedGroups",d.testedGroups},
                        {"selectedLevels",d.selectedLevels},{"selectedClusterIDs",d.selectedClusterIDs}};
                };
                const auto prepare = [&](float threshold, bool freeze, bool pump, auto&& consume) {
                    gpu.begin();
                    if (runtime) {
                        check(runtime->syncRuntimeScene(scene, log), "LOD synchronize stream: " + log);
                        if (pump) {
                            check(streamer->beginFrame(gpu.frame), "LOD stream frame");
                            MeshletStreamFrameDesc frame;
                            frame.width = kWidth; frame.height = kHeight;
                            // Raster snapshot freeze retains source mappings and
                            // blocks shadow requests without requiring RT BLAS.
                            // freezeSourceResidency also rebuilds RT representations.
                            frame.freezeRasterSnapshot = freeze;
                            // Publish through the normal stream frame while the primary
                            // view looks away. No visibility preparation or draw follows.
                            frame.camera.eye = {100,100,100}; frame.camera.center = {100,100,101};
                            check(runtime->cmdBeginFrame(*gpu.commands, *streamer, frame), "LOD stream publication");
                            check(streamer->copyStreamedData(*gpu.commands), "LOD copy stream uploads");
                            check(runtime->cmdPreTraversal(*gpu.commands, frame), "LOD publish page table");
                            check(runtime->cmdPostTraversal(*gpu.commands), "LOD post traversal");
                            check(runtime->cmdEndFrame(*gpu.commands), "LOD end stream frame");
                        }
                        check(runtime->cmdPrepareShadowGeometry(*gpu.commands), "LOD shadow geometry access");
                    }
                    auto stream = runtime ? runtime->shadowGeometryResources() : MeshletStreamShadowGeometryView{};
                    stream.owner = runtime;
                    auto base = casterCache.prepare(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, casterSettings);
                    check(base, "Prepare LOD base: " + log);
                    const auto view = camera.constants(gpu.index, kWidth, kHeight, kWidth, kHeight);
                    const auto projections = VirtualShadowMaps::makeShadowLODProjections(base->bounds, view, light, settings, log);
                    check(projections, "LOD light projections: " + log);
                    auto selected = prepareVirtualShadowMapShadowLOD(device, *gpu.commands, scene,
                        runtime ? &stream : nullptr, *base, casterSettings, *projections, threshold, log);
                    check(selected, "Select shadow LOD: " + log);
                    auto result = consume(*base, *selected, view);
                    if (runtime && pump) { streamer->endFrame(); runtime->prepareMaintenance(); }
                    return result;
                };
                const auto probe = [&](float threshold, bool freeze) {
                    return prepare(threshold, freeze, true, [&](const auto&, const auto& selected, const auto&) {
                        const auto result = selectionJSON(selected); gpu.finish(); return result;
                    });
                };
                if (runtime) {
                    const auto frozen = probe(0, true);
                    report["streamWarmup"].push_back({{"phase","FrozenOffscreen"},
                        {"freezeMode","rasterSnapshot"},{"selection",frozen}}); save();
                    check(frozen.at("requestedPages") == 0, "Frozen shadow traversal enqueued streaming requests");
                    check(frozen.at("blockedPages").get<uint32_t>() > 0, "Freeze did not exercise missing fine pages");
                    const uint32_t frozenClusters = frozen.at("selectedClusters");
                    bool converged = false; uint64_t requested = 0;
                    for (uint32_t frame = 0; frame < 40; ++frame) {
                        const auto selected = probe(0, false);
                        requested += selected.at("requestedPages").get<uint32_t>();
                        report["streamWarmup"].push_back({{"phase","OffscreenRefine"},{"frame",frame},{"selection",selected}}); save();
                        if (selected.at("selectedClusters").get<uint32_t>() > frozenClusters &&
                            selected.at("selectedClusters").get<uint32_t>() == expectedFinestClusters &&
                            selected.at("fallbackCasters") == 0 && selected.at("requestedPages") == 0) { converged = true; break; }
                    }
                    check(requested > 0 && converged, "Offscreen shadow-only streaming failed to request and refine");
                }
                // The same fully available Bunny must respond to light distance,
                // independently of the receiver camera and primary stream view.
                const auto directional = light;
                light.type = VirtualShadowMapLightType::Spot; light.direction = {0,0,-1};
                light.position = {0,0,5}; light.outerConeAngle = 1.4f; light.range = 100;
                const auto nearSelection = probe(1, false);
                light.position[2] = 40;
                const auto farSelection = probe(1, false);
                report["lightDistance"][label] = {{"near",nearSelection},{"far",farSelection}}; save();
                check(nearSelection.at("selectedClusters").get<uint32_t>() > farSelection.at("selectedClusters").get<uint32_t>(),
                    "Light near/far distance did not change actual Bunny cluster selection");
                light = directional;
                const auto record = [&](uint32_t arm, float threshold, bool cancel = false, bool baseReference = false) {
                    return prepare(threshold, false, false, [&](const auto& base, const auto& selected, const auto& view) {
                        auto options = settings; options.cacheLayering = arm == 0; options.forceRedraw = arm >= 2;
                        auto result = maps[arm].record(device, *gpu.commands, *depthView, view, light,
                            baseReference ? base : selected, scene.materials(), {}, options, log);
                        check(result, "Record shadow LOD: " + log);
                        auto capture = readCapture(device, gpu, *result, options, cancel);
                        capture.selection = selectionJSON(selected);
                        return capture;
                    });
                };
                const auto compare = [&](const std::string& state, float threshold, const LODCapture* reference) {
                    report["activeState"] = label + "-" + state; validation.phase(label + "-" + state);
                    const auto layered = record(0, threshold), single = record(1, threshold), forced = record(2, threshold);
                    const auto snapshot = [](const LODCapture& capture) {
                        return Json{{"pages",capture.pages},{"statistics",capture.statistics},{"diagnostics",capture.diagnostics},
                            {"selection",capture.selection},{"visibilitySHA256",debug::debugSha256(capture.visibility)}};
                    };
                    report["captureComparisons"].push_back({{"name",label + "-" + state},
                        {"layered",snapshot(layered)},{"single",snapshot(single)},{"forced",snapshot(forced)}}); save();
                    check(layered.selection == single.selection && single.selection == forced.selection, "A/B shadow cuts changed between arms");
                    report["captureComparisons"].back()["layeredVsSingle"] = compareSameCut(layered, single, settings.pageSize);
                    report["captureComparisons"].back()["layeredVsForced"] = compareSameCut(layered, forced, settings.pageSize); save();
                    check(layered.statistics[5] == 0 && forced.statistics[5] == 0, "LOD correctness fixture overflowed build work");
                    check(layered.diagnostics[kVSMCacheDynamicCasters] == 0 && layered.diagnostics[kVSMCachePromotedCasters] == 0 &&
                        layered.diagnostics[kVSMCacheChangedCasters] == 0,
                        "LOD-only change promoted an authored static caster into dynamic cache");
                    auto quality = saveQuality(directory, label + "-" + state, layered, reference ? *reference : layered);
                    report["states"].push_back({{"name",label + "-" + state},{"threshold",threshold},
                        {"selection",layered.selection},{"layeredDiagnostics",layered.diagnostics},
                        {"layeredStatistics",layered.statistics},{"forcedStatistics",forced.statistics},{"quality",quality}}); save();
                    check(quality.at("shadowedPixels").get<uint64_t>() > 0, "LOD fixture did not produce a visible shadow");
                    return layered;
                };
                const auto finest = compare("Finest", 0, nullptr);
                if (!streamed) { compareSameCut(finest, record(3, 0, false, true), settings.pageSize); }
                const auto warm = compare("FinestWarm", 0, &finest);
                check(warm.diagnostics[kVSMCacheStaticDirtyFinePages] == 0 && warm.diagnostics[kVSMCacheStaticDirtyCoarsePages] == 0,
                    "Unchanged LOD cut redrew static depth");
                const auto terminal = compare("TerminalBaseline", 1e9f, &finest);
                check(terminal.diagnostics[kVSMCacheLODChangedCasters] > 0,
                    "Finest-to-terminal cut change did not report independent LOD invalidation");
                check(terminal.selection.at("selectedClusters").get<uint32_t>() < finest.selection.at("selectedClusters").get<uint32_t>(),
                    "Real Bunny did not reduce selected clusters for the coarse shadow cut");
                if (streamed) { compareSameCut(terminal, record(3, 1e9f, false, true), settings.pageSize); }
                (void)compare("TexelBudget", 1, &finest);
                (void)record(0, 1e9f, true);
                (void)compare("CancelledCutRollback", 1, &finest);
                (void)compare("ThresholdBackToFinest", 0, &finest);
                setCamera(true);
                const auto offscreenFinest = record(3, 0);
                (void)compare("OffscreenCaster", 1, &offscreenFinest);
            }
            report["validation"] = validation.snapshot();
            check(!validation.failed(), "Shadow LOD Vulkan validation error");
            report["passed"] = true; save();
            return RHITestResult::pass("Bunny shadow LOD: exact same-cut D32/visibility, independent offscreen streaming, metric/DAG guards, cache layering and cancellation; changed-cut PNG differences recorded");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapLODTest);

} // namespace
} // namespace metallic::tests
