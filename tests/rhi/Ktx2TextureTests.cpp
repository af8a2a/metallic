#include "RhiTest.h"
#include "Runtime/Render/Streamer/Ktx2Texture.h"
#include "Runtime/Render/Streamer/SceneResourceManager.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/ComputeProgram.h"
#include "json.hpp"
#include <zstd.h>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <bit>
#include <thread>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
void require(bool okay, const std::string& message)
{
    if (!okay) {
        throw std::runtime_error(message);
    }
}
void require(Result result, const std::string& message)
{
    require(bool(result), message + ": " + std::string(resultToString(result)));
}
template <typename T> void put(std::vector<uint8_t>& bytes, size_t offset, T value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}
std::vector<uint8_t> constantBlock(uint32_t format)
{
    if (format == 139) {
        return {77, 77, 0, 0, 0, 0, 0, 0};
    }
    if (format == 141) {
        return {64, 64, 0, 0, 0, 0, 0, 0, 192, 192, 0, 0, 0, 0, 0, 0};
    }
    // BC7 mode 6, constant RGBA=(128,64,32,254), both endpoint p-bits zero.
    std::vector<uint8_t> block(16);
    uint32_t bit = 0;
    const auto write = [&](uint32_t value, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i, ++bit) {
            block[bit / 8] |= uint8_t(((value >> i) & 1) << (bit % 8));
        }
    };
    write(64, 7);
    for (uint32_t channel : {64u, 32u, 16u, 127u}) {
        write(channel, 7);
        write(channel, 7);
    }
    write(0, 2);
    write(0, 3);
    for (uint32_t i = 1; i < 16; ++i) {
        write(0, 4);
    }
    require(bit == 128, "BC7 fixture bit count");
    return block;
}
std::filesystem::path makeKtx(const std::filesystem::path& directory, const std::string& name,
                              uint32_t format, const std::string& swizzle, uint32_t width = 7,
                              uint32_t height = 5)
{
    const std::array<uint8_t, 12> magic{0xab, 0x4b, 0x54, 0x58, 0x20, 0x32,
                                        0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a};
    const uint32_t mipCount = std::bit_width(std::max(width, height));
    const uint32_t dfdOffset = 80 + mipCount * 24;
    std::vector<uint8_t> data(dfdOffset + 28);
    std::copy(magic.begin(), magic.end(), data.begin());
    put(data, 12, format);
    put(data, 16, 1u);
    put(data, 20, width);
    put(data, 24, height);
    put(data, 36, 1u);
    put(data, 40, mipCount);
    put(data, 44, 2u);
    put(data, 48, dfdOffset);
    put(data, 52, 28u);
    put(data, dfdOffset, 28u);
    std::string pair = "KTXswizzle";
    pair.push_back(0);
    pair += swizzle;
    pair.push_back(0);
    const uint32_t kvOffset = uint32_t(data.size()), kvSize = 4 + ((uint32_t(pair.size()) + 3) & ~3u);
    data.resize(data.size() + kvSize);
    put(data, 56, kvOffset);
    put(data, 60, kvSize);
    put(data, kvOffset, uint32_t(pair.size()));
    std::memcpy(data.data() + kvOffset + 4, pair.data(), pair.size());
    const auto block = constantBlock(format);
    for (uint32_t mip = 0; mip < mipCount; ++mip) {
        const auto blocks = ((std::max(width >> mip, 1u) + 3) / 4) * ((std::max(height >> mip, 1u) + 3) / 4);
        std::vector<uint8_t> raw;
        for (uint32_t i = 0; i < blocks; ++i) {
            raw.insert(raw.end(), block.begin(), block.end());
        }
        std::vector<uint8_t> encoded(ZSTD_compressBound(raw.size()));
        const auto size = ZSTD_compress(encoded.data(), encoded.size(), raw.data(), raw.size(), 3);
        require(!ZSTD_isError(size), "Zstd fixture compression");
        encoded.resize(size);
        put(data, 80 + mip * 24, uint64_t(data.size()));
        put(data, 88 + mip * 24, uint64_t(encoded.size()));
        put(data, 96 + mip * 24, uint64_t(raw.size()));
        data.insert(data.end(), encoded.begin(), encoded.end());
    }
    const auto path = directory / name;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return path;
}
std::filesystem::path makeScene(const std::filesystem::path& directory, bool pressure = false)
{
    Json root =
        Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "buffers":[{"uri":"unused-geometry.bin","byteLength":36}],"bufferViews":[{"buffer":0,"byteLength":36}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
        "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":299}},"normalTexture":{"index":1}}]})");
    for (uint32_t i = 0; i < 300; ++i) {
        const uint32_t format = i == 0 ? 139 : (i == 1 || i == 2 ? 141 : (i == 4 ? 145 : 146));
        const std::string swizzle = i == 0 ? "111r" : (i == 1 ? "rgba" : (i == 2 ? "1rg1" : "rgba"));
        const auto path = pressure && i < 2
                              ? makeKtx(directory, std::to_string(i) + ".ktx2", format, swizzle, 1024, 1024)
                              : makeKtx(directory, std::to_string(i) + ".ktx2", format, swizzle);
        root["images"].push_back({{"uri", path.filename().string()}, {"mimeType", "image/png"}});
        root["textures"].push_back({{"source", i}});
    }
    const auto path = directory / "fixture.gltf";
    std::ofstream file(path);
    file << root;
    return path;
}

std::array<float, 12> sampleTexture(RhiTestContext& context, ScenePathTraceResources& resources,
                                    uint32_t index, uint32_t mip, uint32_t flags)
{
    ShaderCompileResult shader;
    std::string log;
    require(compileSlangShaderToSpirv({.moduleName = "Features/Debug/TextureResourceProbe",
                                       .entryPointName = "main",
                                       .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                                      shader),
            shader.diagnostics);
    const ComputeProgramBindingDesc layout[] = {
        {.binding = 0,
         .kind = ComputeResourceBindingKind::SampledImage,
         .descriptorCount = resources.materialTextureCount()},
        {.binding = 1, .kind = ComputeResourceBindingKind::StorageBuffer}};
    ComputeProgram program;
    require(program.initialize(context.device,
                               {.spirv = shader.spirv.data(),
                                .byteSize = shader.spirv.size() * 4,
                                .pushConstantSize = 16,
                                .bindings = layout,
                                .bindingCount = 2,
                                .requiresRayQuery = false},
                               log),
            log);
    std::unique_ptr<Buffer> output;
    require(context.device.createBuffer({.size = 48,
                                         .structureStride = 16,
                                         .usage = BufferUsageBits::Storage,
                                         .memoryLocation = MemoryLocation::HostReadback},
                                        output),
            "sample buffer");
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    std::unique_ptr<Fence> fence;
    require(context.device.createCommandPool(context.graphicsQueue, pool), "sample pool");
    require(pool->createCommandBuffer(commands), "sample commands");
    require(context.device.createFence({}, fence), "sample fence");
    require(commands->begin(), "sample begin");
    const BufferBarrierDesc ready{
        .buffer = output.get(), .before = ResourceState::Undefined, .after = ResourceState::General};
    commands->barrier({.buffers = &ready, .bufferCount = 1});
    const ComputeDispatchBinding bindings[] = {{.binding = 0,
                                                .textureViews = resources.materialTextureViews().data(),
                                                .textureViewCount = resources.materialTextureCount()},
                                               {.binding = 1, .buffer = output.get()}};
    const uint32_t push[] = {index, mip, flags, 0};
    require(program.dispatch({.commandBuffer = commands.get(),
                              .bindings = bindings,
                              .bindingCount = 2,
                              .pushData = push,
                              .pushDataSize = 16}),
            log);
    require(commands->end(), "sample end");
    CommandBuffer* command = commands.get();
    require(context.graphicsQueue.submit(
                {.commandBuffers = &command, .commandBufferCount = 1, .signalFence = fence.get()}),
            "sample submit");
    require(fence->wait(), "sample wait");
    output->invalidate();
    std::array<float, 12> result;
    auto* mapped = output->map();
    require(mapped != nullptr, "sample map");
    std::memcpy(result.data(), mapped, 48);
    output->unmap();
    return result;
}

Json statsJson(const ScenePathTraceResources& resources)
{
    const auto s = resources.textureStats();
    const auto upload = resources.uploadStats();
    Json report{{"logicalTextures", s.logicalTextureCount},
            {"ktxImages", s.ktxImageCount},
            {"mimeMismatches", s.mimeMismatchCount},
            {"residentImagesIncludingFallback", s.residentImageCount},
            {"descriptorCount", resources.materialTextureCount()},
            {"selectedMaxDimension", s.selectedMaxDimension},
            {"budgetBytes", s.budgetBytes},
            {"plannedPayloadBytes", s.plannedPayloadBytes},
            {"plannedAllocationBytes", s.plannedAllocationBytes},
            {"residentPayloadBytes", s.residentPayloadBytes},
            {"residentAllocationBytes", s.residentAllocationBytes},
            {"peakStagingBytes", s.peakStagingBytes},
            {"uploadBatches", upload.submittedBatches},
            {"peakInFlightBatches", upload.peakInFlightBatches}};
    report["uploadProfile"] = {
        {"wallMs", upload.textureWallMs}, {"headerMs", upload.textureHeaderMs}, {"planMs", upload.texturePlanMs},
        {"buildMs", upload.textureBuildMs}, {"openMs", upload.ktx.openMs}, {"readMs", upload.ktx.readMs},
        {"decodeMs", upload.ktx.decodeMs}, {"imageCreateMs", upload.imageCreateMs}, {"stagingMs", upload.stagingMs},
        {"memcpyMs", upload.stagingCopyMs}, {"flushMs", upload.stagingFlushMs},
        {"commandSetupMs", upload.commandSetupMs}, {"recordMs", upload.recordMs},
        {"copySubmitMs", upload.copySubmitMs}, {"acquireSubmitMs", upload.acquireSubmitMs},
        {"backpressureMs", upload.backpressureMs}, {"finalWaitMs", upload.finalWaitMs},
        {"fileOpens", upload.ktx.fileOpens}, {"decodedMips", upload.ktx.decodedMips},
        {"decoderCreations", upload.ktx.decoderCreations}, {"storedBytes", upload.ktx.storedBytes},
        {"decodedBytes", upload.ktx.decodedBytes},
        {"workers", upload.prefetch.workers}, {"prefetchPeakJobs", upload.prefetch.peakJobs},
        {"prefetchByteLimit", upload.prefetch.byteLimit}, {"prefetchPeakBytes", upload.prefetch.peakBytes},
        {"decodeWaitMs", upload.decodeWaitMs},
        {"copyCompletionObservedMs", upload.copyCompletionObservedMs},
        {"acquireCompletionObservedMs", upload.acquireCompletionObservedMs},
        {"copyCompletionSamples", upload.copyCompletionSamples}, {"acquireCompletionSamples", upload.acquireCompletionSamples},
        {"completionTimingMeaning", "Host-observed latency, includes scheduling and polling; not GPU execution time"}};
    report["slowestTextures"] = Json::array();
    for (const auto& t : upload.slowestTextures) {
        report["slowestTextures"].push_back({{"path", t.path}, {"totalMs", t.totalMs},
            {"openMs", t.openMs}, {"readMs", t.readMs}, {"decodeMs", t.decodeMs},
            {"imageMs", t.imageCreateMs}, {"stagingMs", t.stagingMs}, {"copyMs", t.copyMs}, {"flushMs", t.flushMs}});
    }
    return report;
}

class KtxTextureResourcesTest final : public RhiTest {
  public:
    KtxTextureResourcesTest()
    {
        type = RhiTestType::Rendering;
        name = "ktx2_texture_resources";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!context.device.capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("Requires --rhi-bindless, --rhi-realtime or --rhi-streamline");
        }
        const auto directory = context.outputDirectory / "ktx2";
        std::filesystem::create_directories(directory);
        const auto path = makeScene(directory);
        scene::Scene scene;
        require(scene.loadStreamMetadata(path), scene.lastLoadResult().error);
        SceneResourceManager manager;
        std::shared_ptr<SceneResourceSnapshot> first, same, lower;
        const auto features =
            SceneResourceFeatureBits::Materials | SceneResourceFeatureBits::MaterialTextures;
        RenderGraphProperties props{
            {"path", path.string()}, {"materialTextureMaxDimension", 512}, {"materialTextureBudgetMiB", 16}};
        std::string log;
        require(manager.acquire(context.device, context.graphicsQueue, props, &scene, features, first, log),
                log);
        auto& r = *first->pathTraceResources;
        require(r.materialTextureCount() == 301 && r.logicalTextureIndices().size() == 300,
                "descriptor count capped or logical mapping lost");
        for (uint32_t i = 0; i < 300; ++i) {
            require(r.logicalTextureIndices()[i] == i + 1, "unstable logical mapping");
        }
        require(manager.acquire(context.device, context.graphicsQueue, props, &scene, features, same, log),
                log);
        require(same->pathTraceResources == first->pathTraceResources, "texture owner not shared");
        const auto near = [](float a, float b) { return std::isfinite(a) && std::abs(a - b) < 0.003f; };
        for (uint32_t mip = 0; mip < 3; ++mip) {
            auto bc4 = sampleTexture(context, r, 1, mip, 0);
            require(near(bc4[0], 1) && near(bc4[1], 1) && near(bc4[2], 1) && near(bc4[3], 77.f / 255),
                    "BC4 111r swizzle or tail upload: " + Json(bc4).dump());
            auto normal = sampleTexture(context, r, 2, mip, 2);
            const float x = 64.f / 255 * 2 - 1, y = 192.f / 255 * 2 - 1;
            require(near(normal[4], 64.f / 255) && near(normal[5], 192.f / 255) &&
                        near(normal[6], std::sqrt(1 - x * x - y * y) * 0.5f + 0.5f),
                    "BC5 normal reconstruction");
            auto mr = sampleTexture(context, r, 3, mip, 0);
            require(near(mr[0], 1) && near(mr[1], 64.f / 255) && near(mr[2], 192.f / 255) && near(mr[3], 1),
                    "BC5 1rg1 swizzle");
            auto color = sampleTexture(context, r, 300, mip, 1);
            require(near(color[0], std::pow((128.f / 255 + 0.055f) / 1.055f, 2.4f)) &&
                        near(color[8], color[0]) && near(color[3], 254.f / 255),
                    "BC7 hardware sRGB or descriptor >255");
            auto linear = sampleTexture(context, r, 5, mip, 1);
            require(near(linear[0], 128.f / 255) && near(linear[8], linear[0]),
                    "BC7 linear texture was gamma decoded");
        }
        Ktx2TextureInfo info;
        require(readKtx2TextureInfo(directory / "0.ktx2", info, log), log);
        require(info.firstMipForDimension(4) == 1 && info.tailBytes(1) == 16, "NPOT budget/mip tail");
        std::vector<uint8_t> decoded;
        require(decodeKtx2Mip(info, 2, decoded, log), log);
        require(decoded == constantBlock(139), "Zstd CPU decode differs");
        Ktx2MipReader reader;
        require(reader.open(info, log), log);
        for (uint32_t mip = 0; mip < info.levels.size(); ++mip) {
            require(reader.decode(mip, decoded, log), log);
            const auto block = constantBlock(139);
            require(decoded.size() == info.levels[mip].decodedBytes, "reader decoded length changed");
            for (size_t i = 0; i < decoded.size(); ++i) {
                require(decoded[i] == block[i % block.size()], "reader decoded bytes changed");
            }
        }
        require(!reader.decode(99, decoded, log) && decoded.empty(), "invalid mip retained stale bytes");
        reader.close();
        require(!reader.decode(0, decoded, log), "closed reader accepted decode");
        require(reader.open(info, log) && reader.decode(2, decoded, log), "reader failed to reopen");
        require(reader.stats().fileOpens == 2 && reader.stats().decoderCreations == 1,
                "reader did not reuse decoder across opens");
        reader.close();
        const auto truncatedPath = directory / "truncated.ktx2";
        std::filesystem::copy_file(info.path, truncatedPath, std::filesystem::copy_options::overwrite_existing);
        std::filesystem::resize_file(truncatedPath, std::filesystem::file_size(truncatedPath) - 1);
        auto truncatedInfo = info;
        truncatedInfo.path = truncatedPath;
        require(reader.open(truncatedInfo, log) && !reader.decode(2, decoded, log) && decoded.empty(),
                "payload truncation after header inspection accepted");
        reader.close();
        auto badInfo = info;
        badInfo.path = directory / "missing.ktx2";
        require(!reader.open(badInfo, log) && !reader.decode(0, decoded, log), "failed open reused old file");
        require(reader.open(info, log) && reader.decode(2, decoded, log), "reader did not recover from failure");
        reader.close();
        const auto damagedPath = directory / "damaged.ktx2";
        std::filesystem::copy_file(info.path, damagedPath, std::filesystem::copy_options::overwrite_existing);
        {
            std::fstream damaged(damagedPath, std::ios::binary | std::ios::in | std::ios::out);
            damaged.seekp(info.levels[2].offset);
            std::vector<char> zeros(size_t(info.levels[2].storedBytes), 0);
            damaged.write(zeros.data(), zeros.size());
        }
        auto damagedInfo = info;
        damagedInfo.path = damagedPath;
        require(reader.open(damagedInfo, log) && !reader.decode(2, decoded, log) && decoded.empty(),
                "damaged Zstd payload accepted");
        reader.close();
        require(reader.open(info, log) && reader.decode(2, decoded, log) && decoded == constantBlock(139),
                "reused decoder failed after corrupt frame");
        reader.close();
        auto rawInfo = info;
        rawInfo.path = directory / "raw-payload.bin";
        rawInfo.supercompression = 0;
        const auto rawBytes = constantBlock(139);
        rawInfo.levels = {{0, rawBytes.size(), rawBytes.size()}};
        {
            std::ofstream raw(rawInfo.path, std::ios::binary);
            raw.write(reinterpret_cast<const char*>(rawBytes.data()), rawBytes.size());
        }
        require(reader.open(rawInfo, log) && reader.decode(0, decoded, log) && decoded == rawBytes,
                "reused reader raw payload failed");
        reader.close();
        const auto loadProfile = r.uploadStats();
        require(loadProfile.ktx.fileOpens == r.textureStats().ktxImageCount && loadProfile.ktx.decoderCreations >= 1 &&
                    loadProfile.ktx.decoderCreations <= loadProfile.prefetch.workers,
                "GPU texture preparation reopened files/decoders per mip");
        for (const uint32_t workers : {1u, 2u, 4u, 8u}) {
            std::vector<Ktx2PrefetchRequest> requests;
            for (uint32_t i = 0; i < 17; ++i) { requests.push_back({info, i % 3}); }
            const uint64_t limit = Ktx2TexturePrefetch::requiredBytes(info, 0) * 2;
            Ktx2TexturePrefetch loader(std::move(requests), workers, limit);
            for (uint32_t i = 0; i < 17; ++i) {
                loader.wait();
                const auto* ready = loader.front();
                require(ready && ready->error.empty(), "parallel decode failed");
                for (uint32_t mip = i % 3; mip < info.levels.size(); ++mip) {
                    require(decodeKtx2Mip(info, mip, decoded, log) && ready->mips[mip - i % 3] == decoded,
                        "parallel decode order or bytes changed");
                }
                loader.pop();
            }
            const auto bounded = loader.stats();
            require(bounded.peakBytes <= limit && bounded.peakJobs <= workers * 2,
                "prefetch exceeded byte/job credits");
            // Destruction with completed results and blocked producers must join.
            Ktx2TexturePrefetch cancelled({{info,0},{info,0},{info,0},{info,0}}, workers,
                Ktx2TexturePrefetch::requiredBytes(info,0));
            cancelled.wait();
        }
        {
            Ktx2TexturePrefetch loader({{damagedInfo,0},{info,0}}, 2,
                Ktx2TexturePrefetch::requiredBytes(info,0) * 2);
            loader.wait();
            require(loader.front() && !loader.front()->error.empty(), "parallel corrupt frame accepted");
            loader.pop();
            loader.wait();
            require(loader.front() && loader.front()->error.empty(), "parallel decoder failed after error");
            loader.pop();
        }
        {
            ScenePathTraceResources cancelled;
            require(cancelled.beginPrepareAsync(context.device, context.graphicsQueue, props, scene, log, true), log);
            bool complete = false;
            scene::SceneLoadProgress progress;
            require(cancelled.pumpPrepareAsync(0.1, complete, progress, log), log);
            cancelled.clear();
            require(cancelled.prepare(context.device, context.graphicsQueue, props, &scene, log), log);
            require(cancelled.logicalTextureIndices().size() == r.logicalTextureIndices().size() &&
                std::equal(cancelled.logicalTextureIndices().begin(), cancelled.logicalTextureIndices().end(),
                    r.logicalTextureIndices().begin()), "cancel/restart changed logical texture mapping");
            cancelled.clear();
            Ktx2TextureInfo lateInfo;
            require(readKtx2TextureInfo(directory / "299.ktx2", lateInfo, log), log);
            require(cancelled.beginPrepareAsync(context.device, context.graphicsQueue, props, scene, log, true), log);
            // The bounded queue cannot have admitted image 299 before any consumption.
            {
                std::fstream damaged(lateInfo.path, std::ios::binary | std::ios::in | std::ios::out);
                damaged.seekp(lateInfo.levels.back().offset);
                damaged.put(0);
            }
            complete = false;
            Result outcome;
            while (outcome && !complete) {
                outcome = cancelled.pumpPrepareAsync(5, complete, progress, log);
                std::this_thread::yield();
            }
            makeKtx(directory, "299.ktx2", 146, "rgba");
            require(!outcome && !cancelled.valid() && log.find("Zstd") != std::string::npos,
                "late prefetch failure published an incomplete texture scene");
            cancelled.clear();
            require(cancelled.prepare(context.device, context.graphicsQueue, props, &scene, log), log);
        }
        require(loadProfile.copyCompletionSamples == loadProfile.completedBatches &&
                    loadProfile.textureWallMs > 0 && !loadProfile.slowestTextures.empty(),
                "upload profile missing completed observations");
        if (context.device.capabilities().independentCopyQueue) {
            require(loadProfile.acquireCompletionSamples == loadProfile.completedBatches,
                    "upload profile lost graphics acquire observations");
        }
        {
            std::fstream corrupt(directory / "0.ktx2", std::ios::binary | std::ios::in | std::ios::out);
            corrupt.seekp(96);
            const uint64_t bad = 999;
            corrupt.write(reinterpret_cast<const char*>(&bad), 8);
        }
        require(!readKtx2TextureInfo(directory / "0.ktx2", info, log), "bad block length accepted");
        makeKtx(directory, "0.ktx2", 139, "111r");
        props["materialTextureMaxDimension"] = 1;
        lower = first;
        require(manager.acquire(context.device, context.graphicsQueue, props, &scene, features, lower, log),
                log);
        require(lower != first && lower->pathTraceResources->textureStats().selectedMaxDimension == 1,
                "budget settings reused stale resources");
        require(near(sampleTexture(context, *lower->pathTraceResources, 300, 0, 1)[0],
                     sampleTexture(context, r, 300, 2, 1)[0]),
                "rebased mip sampling differs");
        Json report{{"fullTail", statsJson(r)},
                    {"oneTexelTail", statsJson(*lower->pathTraceResources)},
                    {"gpuSampling", "passed"}};
        const auto pressureDirectory = directory / "budget";
        std::filesystem::create_directories(pressureDirectory);
        const auto pressurePath = makeScene(pressureDirectory, true);
        scene::Scene pressureScene;
        require(pressureScene.loadStreamMetadata(pressurePath), pressureScene.lastLoadResult().error);
        std::shared_ptr<SceneResourceSnapshot> pressure;
        require(manager.acquire(context.device, context.graphicsQueue,
                                {{"path", pressurePath.string()},
                                 {"materialTextureMaxDimension", 1024},
                                 {"materialTextureBudgetMiB", 1}},
                                &pressureScene, features, pressure, log),
                log);
        const auto pressureStats = pressure->pathTraceResources->textureStats();
        require(pressureStats.selectedMaxDimension < 1024 &&
                    pressureStats.residentAllocationBytes <= 1024 * 1024,
                "allocation budget did not reduce mip tails");
        require(near(sampleTexture(context, *pressure->pathTraceResources, 1, 0, 0)[3], 77.f / 255),
                "budgeted BC tail upload differs");
        report["budgetPressure"] = statsJson(*pressure->pathTraceResources);
        std::ofstream(directory / "validation.json") << report.dump(2);
        return RhiTestResult::pass(
            "BC4/5/7, swizzle, sRGB, NPOT/sub-block mips, Zstd, 300 logical textures and shared owners");
    }
};

class ZorahTextureResourcesTest final : public RhiTest {
  public:
    ZorahTextureResourcesTest()
    {
        type = RhiTestType::Rendering;
        name = "zorah_texture_resources";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!std::getenv("METALLIC_ZORAH_Z3_FULL")) {
            return RhiTestResult::skip("Set METALLIC_ZORAH_Z3_FULL for Full texture-only upload");
        }
        if (!context.device.capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("Requires --rhi-bindless, --rhi-realtime or --rhi-streamline");
        }
        const std::filesystem::path path =
            PROJECT_SOURCE_DIR "/Asset/ZorahFull/zorah_textured_public.v1.gltf";
        scene::Scene scene;
        require(scene.loadStreamMetadata(path), scene.lastLoadResult().error);
        ScenePathTraceResources resources;
        std::string log;
        require(resources.prepare(context.device, context.graphicsQueue,
                                  {{"path", path.string()},
                                   {"materialTextureMaxDimension", 512},
                                   {"materialTextureBudgetMiB", 2048},
                                   {"materialTextureLoadWorkers", std::getenv("METALLIC_KTX_LOAD_WORKERS")
                                       ? std::atoi(std::getenv("METALLIC_KTX_LOAD_WORKERS")) : 4}},
                                  &scene, log),
                log);
        const auto stats = resources.textureStats();
        require(stats.logicalTextureCount == 4418 && stats.ktxImageCount == 4418 &&
                    resources.materialTextureCount() == 4419,
                "Full texture references missing");
        require(resources.uploadStats().ktx.fileOpens == 4418 &&
                    resources.uploadStats().ktx.decodedMips == 44140 &&
                    resources.uploadStats().ktx.decoderCreations >= 1 &&
                    resources.uploadStats().ktx.decoderCreations <= resources.uploadStats().prefetch.workers &&
                    resources.uploadStats().ktx.decodedBytes + 4 == stats.residentPayloadBytes,
                "Full mip reader reuse/count/payload mismatch");
        require(resources.uploadStats().prefetch.peakBytes <= resources.uploadStats().prefetch.byteLimit &&
                resources.uploadStats().prefetch.peakJobs <= resources.uploadStats().prefetch.workers * 2,
                "Full prefetch exceeded bounds");
        require(stats.selectedMaxDimension == 512 && stats.residentAllocationBytes <= stats.budgetBytes,
                "Full 512 mip budget mismatch");
        require(resources.uploadStats().submittedBatches == resources.uploadStats().completedBatches,
                "texture publication precedes completion");
        if (context.device.capabilities().independentCopyQueue) {
            require(resources.uploadStats().acquireCompletionSamples == resources.uploadStats().completedBatches,
                    "Full upload profile lost acquire observations");
        }
        for (auto index : resources.logicalTextureIndices()) {
            require(index > 0 && index < resources.materialTextureCount(), "invalid Full logical descriptor");
        }
        const auto pixel = sampleTexture(context, resources, 4418, 0, 0);
        for (float channel : pixel) {
            require(std::isfinite(channel), "invalid last Full descriptor sample");
        }
        auto report = statsJson(resources);
        report["geometryPayloadRead"] = false;
        report["allMipTailsUploaded"] = true;
        report["lastDescriptorSample"] = pixel;
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream output(context.outputDirectory / "zorah-textures.json");
        output << report.dump(2);
        require(bool(output), "Failed to write Full texture upload evidence");
        return RhiTestResult::pass(
            "Full 4418 KTX2 tails uploaded under 2 GiB allocation budget; no geometry cook/render");
    }
};
class BcTextureUploadTest final : public RhiTest {
  public:
    BcTextureUploadTest()
    {
        type = RhiTestType::Command;
        name = "bc_texture_padded_upload";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        for (const auto format : {Format::Bc4Unorm, Format::Bc5Unorm, Format::Bc7Srgb}) {
            for (const uint32_t width : {7u, 1u}) {
                const uint32_t height = width == 7 ? 5 : 1;
                const uint32_t rowBytes = ((width + 3) / 4) * compressedBlockBytes(format);
                const uint32_t rows = (height + 3) / 4;
                std::vector<uint8_t> source(64 * rows), expected(rowBytes * rows);
                for (uint32_t y = 0; y < rows; ++y) {
                    for (uint32_t x = 0; x < rowBytes; ++x) {
                        source[y * 64 + x] = expected[y * rowBytes + x] = uint8_t(x + y * 37);
                    }
                }
                std::unique_ptr<Streamer> streamer;
                require(context.device.createStreamer(
                            {.constantBufferSize = 4096, .dynamicBufferSizePerFrame = 1024}, streamer),
                        "BC streamer");
                std::unique_ptr<Texture> texture;
                require(context.device.createTexture({.usage = TextureUsageBits::TransferDestination |
                                                               TextureUsageBits::TransferSource,
                                                      .format = format,
                                                      .width = width,
                                                      .height = height},
                                                     texture),
                        "BC texture");
                if (width == 7) {
                    require(!streamer
                                 ->streamTextureData({.data = source.data(),
                                                      .dstTexture = texture.get(),
                                                      .dstOffsetX = 1,
                                                      .width = 4,
                                                      .height = 4})
                                 .valid(),
                            "unaligned BC offset accepted");
                    require(!streamer
                                 ->streamTextureData({.data = source.data(),
                                                      .dstTexture = texture.get(),
                                                      .width = 3,
                                                      .height = 4})
                                 .valid(),
                            "partial non-edge BC block accepted");
                }
                require(streamer
                            ->streamTextureData({.data = source.data(),
                                                 .dataRowPitch = 64,
                                                 .dataSlicePitch = 64 * rows,
                                                 .dstTexture = texture.get(),
                                                 .width = width,
                                                 .height = height})
                            .valid(),
                        "padded BC upload rejected");
                std::unique_ptr<Buffer> readback;
                require(context.device.createBuffer({.size = expected.size(),
                                                     .usage = BufferUsageBits::TransferDestination,
                                                     .memoryLocation = MemoryLocation::HostReadback},
                                                    readback),
                        "BC readback");
                std::unique_ptr<CommandPool> pool;
                std::unique_ptr<CommandBuffer> commands;
                std::unique_ptr<Fence> fence;
                require(context.device.createCommandPool(context.graphicsQueue, pool), "BC pool");
                require(pool->createCommandBuffer(commands), "BC commands");
                require(context.device.createFence({}, fence), "BC fence");
                require(commands->begin(), "BC begin");
                TextureBarrierDesc barrier{.texture = texture.get(),
                                           .before = ResourceState::Undefined,
                                           .after = ResourceState::TransferDestination};
                commands->barrier({.textures = &barrier, .textureCount = 1});
                commands->copyStreamedData(*streamer);
                barrier.before = ResourceState::TransferDestination;
                barrier.after = ResourceState::TransferSource;
                commands->barrier({.textures = &barrier, .textureCount = 1});
                commands->copyTextureToBuffer(
                    {.texture = texture.get(), .buffer = readback.get(), .width = width, .height = height});
                require(commands->end(), "BC end");
                CommandBuffer* command = commands.get();
                require(
                    context.graphicsQueue.submit(
                        {.commandBuffers = &command, .commandBufferCount = 1, .signalFence = fence.get()}),
                    "BC submit");
                require(fence->wait(), "BC wait");
                readback->invalidate();
                const auto* actual = readback->map();
                require(actual && std::memcmp(actual, expected.data(), expected.size()) == 0,
                        "BC block readback differs");
                readback->unmap();
                streamer->endFrame();
            }
        }
        return RhiTestResult::pass("BC4/5/7 padded block rows, NPOT and one-texel tails round trip");
    }
};
METALLIC_REGISTER_RHI_TEST(BcTextureUploadTest);
METALLIC_REGISTER_RHI_TEST(KtxTextureResourcesTest);
METALLIC_REGISTER_RHI_TEST(ZorahTextureResourcesTest);
} // namespace
} // namespace metallic::tests
