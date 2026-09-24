#include <limits>
#include "RhiTest.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>

namespace metallic::tests {
namespace {
using namespace render;

#define CLASSIFY_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

class StreamClusterClassificationTest final : public RhiTest {
public:
    StreamClusterClassificationTest() { type = RhiTestType::Rendering; name = "stream_cluster_cull_classify_equivalence"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        struct RestoreShaderMode {
            SlangShaderDebugMode previous = slangShaderDebugMode();
            ~RestoreShaderMode() { setSlangShaderDebugMode(previous); }
        } restoreMode;
        const char* captureSymbols = std::getenv("METALLIC_CLASSIFY_CAPTURE_SYMBOLS");
        if (captureSymbols != nullptr && std::strcmp(captureSymbols, "1") == 0) {
            setSlangShaderDebugMode(SlangShaderDebugMode::CaptureSymbols);
        }
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream classification regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless heap"); }
        CLASSIFY_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RhiTestResult::skip("Requires hybrid raster capabilities");
        }
        constexpr uint32_t groupCapacity = 2305, capacity = groupCapacity * 32, instanceCount = 16;
        constexpr uint32_t candidateBase = 16 + 5 * capacity;
        constexpr uint32_t retryBase = 16 + 8 * capacity + ((capacity + 127) / 128) * 5;
        constexpr uint32_t pageBytes = 131072, hzbElements = 21845;
        std::array<VisibilityHybridRasterizer, 2> rasterizers;
        for (auto& rasterizer : rasterizers) { CLASSIFY_REQUIRE(rasterizer.initialize(*device, 128, 128, log, 1, capacity)); }
        std::unique_ptr<BindlessHeap> heap;
        CLASSIFY_REQUIRE(device->createBindlessHeap({.maxBuffers = 18}, heap));
        std::array<BindlessHandle, 18> handles;
        for (auto& handle : handles) { CLASSIFY_REQUIRE(heap->allocateBuffer(handle)); }
        enum Input { Header, Groups, Params, Pages, PageTable, Bindings, Visibility, Requests, Records, Hzb0, Hzb1, Instances, InputCount };
        const uint32_t strides[] = {sizeof(MeshletStreamGpuActiveHeader), sizeof(MeshletStreamGpuActiveGroup),
            sizeof(MeshletStreamGpuParams), 4, sizeof(StreamPageTableEntry), sizeof(MeshletStreamGpuRasterBindings),
            4, 4, sizeof(VisibleClusterRecord), 4, 4, sizeof(GPUSceneGpuInstanceRecord)};
        const uint32_t counts[] = {1, groupCapacity, 1, pageBytes / 4, 3, 1, instanceCount, 16, capacity,
            hzbElements, hzbElements, instanceCount};
        std::array<std::unique_ptr<Buffer>, InputCount> inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            CLASSIFY_REQUIRE(device->createBuffer({.size = uint64_t(strides[i]) * counts[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload}, inputs[i]));
            CLASSIFY_REQUIRE(heap->writeStorageBuffer(handles[i], *inputs[i]));
        }
        for (size_t i = 0; i < 2; ++i) {
            CLASSIFY_REQUIRE(heap->writeStorageBuffer(handles[12 + i * 2], rasterizers[i].clusterBuffer()));
            CLASSIFY_REQUIRE(heap->writeStorageBuffer(handles[13 + i * 2], rasterizers[i].candidateArguments()));
            CLASSIFY_REQUIRE(heap->writeStorageBuffer(handles[16 + i], rasterizers[i].workloadBuffer()));
        }
        const auto upload = [&](size_t index, const void* data, size_t size) -> Result {
            void* mapped = inputs[index]->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data, size);
            inputs[index]->flush(); inputs[index]->unmap();
            return {};
        };
        std::array<std::unique_ptr<ShaderModule>, 9> shaders;
        std::array<std::unique_ptr<ComputePipeline>, 9> pipelines;
        const char* entries[] = {"streamClusterPrepareMain", "streamClusterCullMain", "streamClusterBinMain", "referenceStreamClusterBinMain", "verifyStreamSoftwareLoadMain", "streamWorkloadResetMain", "streamWorkloadMain", "streamClusterBinP0Main", "streamClusterCullP0Main"};
        for (size_t i = 0; i < pipelines.size(); ++i) {
            ShaderCompileResult compiled;
            const char* additional[] = {PROJECT_SOURCE_DIR "/Shaders"};
            const auto result = compileSlangShaderToSpirv({
                .moduleName = (i == 5 || i == 6) ? "Features/GPUDriven/GPUDrivenStreamWorkload" : (i == 3 || i == 4) ? "StreamClusterClassificationProbe" : "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = entries[i],
                .searchPath = i >= 3 ? PROJECT_SOURCE_DIR "/tests/rhi/shaders" : PROJECT_SOURCE_DIR "/Shaders",
                .additionalSearchPaths = additional, .additionalSearchPathCount = 1}, compiled);
            log = compiled.diagnostics;
            CLASSIFY_REQUIRE(result);
            CLASSIFY_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4}, shaders[i]));
            CLASSIFY_REQUIRE(device->createComputePipeline({.computeShader = shaders[i].get(), .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush)}, pipelines[i]));
        }
        // Opt-in resource report for classifier and production SW entrypoints,
        // alongside the validation-enabled classification regression below.
        const char* pipelineStats = std::getenv("METALLIC_VK_PIPELINE_STATISTICS");
        if (pipelineStats != nullptr && std::strcmp(pipelineStats, "1") == 0) {
            struct RestoreShaderMode {
                SlangShaderDebugMode previous = slangShaderDebugMode();
                ~RestoreShaderMode() { setSlangShaderDebugMode(previous); }
            } restoreShaderMode;
            for (const auto mode : {SlangShaderDebugMode::Disabled, SlangShaderDebugMode::CaptureSymbols}) {
                setSlangShaderDebugMode(mode);
                for (const char* entry : {"streamClusterBinMain", "streamClusterBinP0Main", "streamClusterCullMain", "streamClusterCullP0Main", "legacyStreamClusterBinMain", "streamClusterRasterLegacyMain", "streamClusterRasterMain", "streamClusterRasterPlaneMain", "streamClusterRasterCooperativeMain", "streamClusterRasterWorkBinsMain", "streamClusterRasterWorkControlMain"}) {
                    spdlog::info("[SW Pipeline Probe] entry={} debugMode={}", entry, int(mode));
                    const char* additionalStatsPaths[] = {PROJECT_SOURCE_DIR "/Shaders"};
                    ShaderCompileResult compiled;
                    CLASSIFY_REQUIRE(compileSlangShaderToSpirv({.moduleName = std::strncmp(entry, "streamClusterRasterWork", 23) == 0 ?
                        "Features/GPUDriven/GPUDrivenStreamWorkRaster" : std::strcmp(entry, "legacyStreamClusterBinMain") == 0 ?
                        "StreamClusterClassificationProbe" : "Features/GPUDriven/GPUDrivenStreamAsset",
                        .entryPointName = entry, .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                        .additionalSearchPaths = additionalStatsPaths, .additionalSearchPathCount = 1}, compiled));
                    std::unique_ptr<ShaderModule> shader;
                    std::unique_ptr<ComputePipeline> pipeline;
                    CLASSIFY_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4}, shader));
                    CLASSIFY_REQUIRE(device->createComputePipeline({.computeShader = shader.get(), .usesBindlessHeap = true,
                        .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush)}, pipeline));
                }
            }
        }
        const uint64_t binBytes = rasterizers[0].clusterBuffer().desc().size;
        const uint64_t recordBytes = inputs[Records]->desc().size;
        std::unique_ptr<Buffer> readback;
        CLASSIFY_REQUIRE(device->createBuffer({.size = binBytes + recordBytes + 52 + 128,
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}, readback));
        std::unique_ptr<Buffer> exactQueueReadback;
        CLASSIFY_REQUIRE(device->createBuffer({.size = uint64_t(capacity) * 16,
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}, exactQueueReadback));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        CLASSIFY_REQUIRE(device->createCommandPool(*queue, pool));
        CLASSIFY_REQUIRE(pool->createCommandBuffer(commands));
        CLASSIFY_REQUIRE(device->createFence(false, fence));
        // Optional classifier-only A/B on the P1 exact queue. This isolates the
        // removed guard; use sample profiles to include changed cull/queue work.
        // Both kernels only overwrite tags, so the queue can be safely reused.
        const char* benchmarkPath = std::getenv("METALLIC_CLASSIFY_BENCHMARK");
        std::ofstream benchmark;
        std::unique_ptr<TimestampQueryPool> timing;
        constexpr uint32_t warmupPairs = 8, measuredPairs = 40;
        constexpr uint32_t timedDispatches = (warmupPairs + measuredPairs) * 2;
        if (benchmarkPath != nullptr) {
            if (!device->capabilities().timestampQueries || queue->timestampValidBits() == 0) {
                return RhiTestResult::skip("Classification benchmark requires GPU timestamps");
            }
            benchmark.open(benchmarkPath);
            if (!benchmark) { return RhiTestResult::fail("Cannot open classification benchmark CSV"); }
            benchmark << "case,phase,groups,classify_count,round,variant,gpu_us,debug_mode\n" << std::setprecision(10);
            CLASSIFY_REQUIRE(device->createTimestampQueryPool(*queue, {.queryCount = timedDispatches * 2}, timing));
        }
        bool sawFastSoftware = false, sawFastHardware = false;
        bool submitted = false, sawRetry = false, sawLate = false, sawSoftware = false, sawHardware = false, saw2D = false;
        struct Case { uint32_t groups; float maxPixels; bool ortho; bool reversed; uint32_t culling; bool dense = false; bool jitter = false; bool tessellation = false; uint32_t payloadFault = 0; bool disableMetadata = false; };
        const Case cases[] = {{2305, 8, true, true, 0, true}, {0, 8, true, true, 28},
            {1, 1, false, true, 0}, {127, 8, false, false, 12}, {129, 32, true, false, 28},
            {2305, 8, false, true, 28}, {129, 1, true, true, 28, false, true}, {1, 32, false, false, 0},
            {129, 2, false, true, 0}, {129, 4, true, false, 0},
            {2305, 0, true, true, 0, true}, {129, 0, false, true, 28}, {0, 0, true, false, 28}, {129, 8, true, true, 0, false, false, true},
            {129, 8, false, true, 0, false, false, false, 1}, {129, 8, false, true, 0, false, false, false, 2},
            {129, 8, false, true, 0, false, false, false, 3}, {129, 8, false, true, 0, false, false, false, 4},
            {129, 8, false, true, 0, false, false, false, 5}, {129, 8, false, true, 0, false, false, false, 6},
            {129, 8, false, true, 0, false, false, false, 7}, {129, 8, false, true, 0, false, true, false, 8},
            {129, 8, true, true, 0, true, false, true},
            {129, 8, false, true, 0, false, true, false, 0, true}};
        size_t caseIndex = 0;
        for (const auto test : cases) {
            std::vector<uint8_t> page(pageBytes);
            scene::MeshletStreamPayloadHeader pageHeader;
            pageHeader.clusterCount = 32; pageHeader.vertexCount = 32 * 128; pageHeader.triangleIndexCount = 32 * 128 * 3;
            pageHeader.clusterOffsetBytes = sizeof(pageHeader);
            pageHeader.positionOffsetBytes = pageHeader.clusterOffsetBytes + 32 * sizeof(scene::MeshletStreamPayloadCluster);
            pageHeader.triangleOffsetBytes = pageHeader.positionOffsetBytes + pageHeader.vertexCount * 16;
            pageHeader.payloadByteSize = pageHeader.triangleOffsetBytes + pageHeader.triangleIndexCount;
            pageHeader.attributeFlags = 1;
            std::memcpy(page.data(), &pageHeader, sizeof(pageHeader));
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t kind = test.dense ? c % 2 : c % 16;
                const float size = kind == 1 ? 1.f : kind == 2 ? .125f : .01f;
                const float z = kind == 3 ? .005f : kind == 4 ? 12.f : kind == 5 ? -1.f : 2.f;
                scene::MeshletStreamPayloadCluster cluster;
                cluster.vertexOffset = c * 128; cluster.vertexCount = 128;
                cluster.triangleOffset = c * 128 * 3; cluster.triangleCount = kind == 6 ? 0 : 128;
                cluster.boundingSphere[0] = kind == 7 ? 20.f : 0.f;
                cluster.boundingSphere[2] = z; cluster.boundingSphere[3] = kind == 11 ? 2.1f : size * 2;
                if (kind == 14) { cluster.boundingSphere[3] = std::numeric_limits<float>::quiet_NaN(); }
                if (kind == 15) { cluster.boundingSphere[3] = -1.f; }
                cluster.coneApexCutoff[2] = z; cluster.coneApexCutoff[3] = kind == 8 ? .5f : 1.f;
                cluster.coneAxisLodError[2] = 1.f;
                if (kind == 9) { cluster.vertexCount = 129; }
                if (kind == 10) { cluster.triangleOffset = pageHeader.triangleIndexCount; }
                std::memcpy(page.data() + pageHeader.clusterOffsetBytes + c * sizeof(cluster), &cluster, sizeof(cluster));
                for (uint32_t v = 0; v < 128; ++v) {
                    const float position[] = {cluster.boundingSphere[0] + (v % 3 == 1 ? size : 0.f),
                        v % 3 == 2 ? size : 0.f, z + (kind == 11 && v % 3 == 1 ? -2.f : 0.f), 1.f};
                    std::memcpy(page.data() + pageHeader.positionOffsetBytes + (c * 128 + v) * 16, position, sizeof(position));
                    // Exercise both maximum valid and malformed indices.
                    const uint8_t triangle[] = {uint8_t(v % 126), uint8_t(v % 126 + 1), uint8_t(kind == 12 ? 255 : v % 126 + 2)};
                    std::memcpy(page.data() + pageHeader.triangleOffsetBytes + c * 384 + v * 3, triangle, 3);
                }
            }
            switch (test.payloadFault) {
            case 1: pageHeader.payloadByteSize = sizeof(pageHeader) - 4; break;
            case 2: pageHeader.payloadByteSize = pageBytes + 4; break;
            case 3: pageHeader.positionOffsetBytes += 1; break;
            case 4: pageHeader.clusterOffsetBytes = pageBytes; break;
            case 5: pageHeader.positionFormat = 0xffffffffu; break;
            case 6: pageHeader.triangleOffsetBytes = pageBytes; break;
            case 8:
                // Valid packed float3 page with the same authored vertex values.
                for (uint32_t v = 0; v < pageHeader.vertexCount; ++v) {
                    std::memmove(page.data() + pageHeader.positionOffsetBytes + v * 12,
                        page.data() + pageHeader.positionOffsetBytes + v * 16, 12);
                }
                pageHeader.positionFormat = 4;
                break;
            }
            std::memcpy(page.data(), &pageHeader, sizeof(pageHeader));
            std::vector<MeshletStreamGpuActiveGroup> groups(groupCapacity);
            for (uint32_t i = 0; i < groupCapacity; ++i) {
                auto& g = groups[i];
                g.clusterCount = 32; g.clusterSelectionMask = UINT32_MAX; g.gpuSceneInstanceIndex = i % instanceCount;
                g.pageIndex = !test.dense && i % 13 == 0 ? 1 : !test.dense && i % 17 == 0 ? 2 : 0;
                g.world0[0] = 1.f; g.world1[1] = !test.dense && i % 5 == 0 ? 2.f : 1.f; g.world2[2] = 1.f; g.world3[3] = 1.f;
                if (!test.dense && i % 3 == 0) { g.world0[0] = -1.f; g.world1[0] = .3f; }
                if (!test.dense && i % 7 == 0) { g.clusterSelectionMask = 0x80010001u; }
            }
            std::array<StreamPageTableEntry, 3> pages{};
            pages[0].deviceOffsetAndState = packStreamPageTableEntry(0, MeshletStreamPageResidencyState::Resident);
            pages[1].deviceOffsetAndState = packStreamPageTableEntry(0, MeshletStreamPageResidencyState::Unloaded);
            pages[2].deviceOffsetAndState = packStreamPageTableEntry(pageBytes - 8, MeshletStreamPageResidencyState::Resident);
            MeshletStreamGpuActiveHeader header{.activeGroupCount = test.groups, .activeGroupCapacity = groupCapacity, .maxActiveGroupClusters = 32};
            MeshletStreamGpuParams params;
            params.center[2] = 1; params.upProjection[1] = 1; params.upProjection[3] = test.ortho ? 1.f : 0.f;
            params.viewport[0] = 1; params.viewport[1] = params.viewport[2] = 128; params.viewport[3] = 1.57079632679f;
            params.clipOrtho[0] = .01f; params.clipOrtho[1] = 10; params.clipOrtho[2] = 2; params.clipOrtho[3] = test.reversed ? 1.f : 0.f;
            std::memcpy(params.previousCenter, params.center, 16); std::memcpy(params.previousUpProjection, params.upProjection, 16);
            std::memcpy(params.previousViewport, params.viewport, 16); std::memcpy(params.previousClipOrtho, params.clipOrtho, 16);
            if (test.jitter) {
                std::memcpy(params.renderCenter, params.center, 16); std::memcpy(params.renderUpProjection, params.upProjection, 16);
                std::memcpy(params.renderViewport, params.viewport, 16); std::memcpy(params.renderClipOrtho, params.clipOrtho, 16);
                params.renderEye[0] = .2f; params.renderEye[3] = .35f; params.renderCenter[3] = -.2f;
            }
            params.pageBufferBytes = pageBytes; params.drawTaskCount = capacity * 2; params.scenePageCount = 3;
            MeshletStreamGpuRasterBindings bindings{.visibleClusterBuffer = handles[Records].index,
                .instanceVisibilityBuffer = handles[Visibility].index, .hzbBuffer0 = handles[Hzb0].index, .hzbBuffer1 = handles[Hzb1].index,
                .visibleRecordBase = 371, .visibleRecordCapacity = capacity, .hzbMipCount = 8, .hzbValid = 1,
                .cullingFlags = test.culling, .width = 128, .height = 128, .gpuSceneInstanceBuffer = handles[Instances].index,
                .tessellationBuffer = test.tessellation ? 0u : UINT32_MAX,
                .classificationFlags = (test.dense || test.disableMetadata) ? 1u : 0u};
            std::array<uint32_t, instanceCount> visibility;
            std::array<GPUSceneGpuInstanceRecord, instanceCount> instances;
            for (uint32_t i = 0; i < instanceCount; ++i) {
                visibility[i] = test.dense ? 1 : i % 4;
                instances[i].identity[3] = i % 3 == 1 ? 6 : i % 3 == 2 ? 8 : 0;
            }
            std::vector<float> hzb(hzbElements, test.reversed ? 0.f : 1.f);
            CLASSIFY_REQUIRE(upload(Hzb0, hzb.data(), hzb.size() * 4));
            std::fill(hzb.begin(), hzb.end(), test.reversed ? 1.f : 0.f);
            CLASSIFY_REQUIRE(upload(Hzb1, hzb.data(), hzb.size() * 4));
            if (test.payloadFault == 7) { params.pageBufferBytes = sizeof(pageHeader) - 4; }
            CLASSIFY_REQUIRE(upload(Header, &header, sizeof(header)));
            CLASSIFY_REQUIRE(upload(Groups, groups.data(), groups.size() * sizeof(groups[0])));
            CLASSIFY_REQUIRE(upload(Params, &params, sizeof(params)));
            CLASSIFY_REQUIRE(upload(Pages, page.data(), page.size()));
            CLASSIFY_REQUIRE(upload(Bindings, &bindings, sizeof(bindings)));
            CLASSIFY_REQUIRE(upload(Visibility, visibility.data(), sizeof(visibility)));
            CLASSIFY_REQUIRE(upload(Instances, instances.data(), sizeof(instances)));
            for (uint32_t phase = 0; phase < 2; ++phase) {
                std::vector<uint32_t> reference;
                uint32_t p1Exact = 0, p1Hardware = 0, p1Software = 0;
                for (uint32_t schedule = 0; schedule < 3; ++schedule) {
                    const uint32_t bufferSet = std::min(schedule, 1u);
                    auto& rasterizer = rasterizers[bufferSet];
                    const bool measure = timing && schedule == 1 && phase == 0 &&
                        (caseIndex == 0 || caseIndex == 4 || caseIndex == 5 || caseIndex == 6 || caseIndex == 21);
                    std::vector<VisibleClusterRecord> records(capacity);
                    std::array<uint32_t, 16> requests{};
                    CLASSIFY_REQUIRE(upload(Records, records.data(), recordBytes));
                    CLASSIFY_REQUIRE(upload(Requests, requests.data(), sizeof(requests)));
                    CLASSIFY_REQUIRE(upload(PageTable, pages.data(), sizeof(pages)));
                    if (submitted) { CLASSIFY_REQUIRE(fence->reset()); CLASSIFY_REQUIRE(pool->reset()); }
                    CLASSIFY_REQUIRE(commands->begin());
                    CLASSIFY_REQUIRE(rasterizer.beginClusters(*commands, test.maxPixels, test.reversed, 0, capacity, true, true));
                    commands->bindBindlessHeap(*heap);
                    MeshletStreamUserPush push{.pageBuffer = handles[Pages].index, .activeGroupBuffer = handles[Groups].index,
                        .pageTableBuffer = handles[PageTable].index, .paramsBuffer = handles[Params].index,
                        .requestBuffer = handles[Requests].index, .activeHeaderBuffer = handles[Header].index,
                        .traversalPhase = phase, .rasterBindingsBuffer = handles[Bindings].index,
                        .hybridQueueBuffer = handles[13 + bufferSet * 2].index, .hybridClusterBuffer = handles[12 + bufferSet * 2].index};
                    CLASSIFY_REQUIRE(rasterizer.prepareStreamClusterCandidates(*commands, *pipelines[0], push));
                    if (schedule != 0) {
                        commands->bindComputePipeline(*pipelines[4]);
                        commands->pushBindlessData(&push, sizeof(push));
                        CLASSIFY_REQUIRE(commands->dispatchIndirect(rasterizer.candidateArguments()));
                        BufferBarrierDesc verifyBarrier{.buffer = &rasterizer.clusterBuffer(), .before = ResourceState::General, .after = ResourceState::General};
                        commands->barrier({.buffers = &verifyBarrier, .bufferCount = 1});
                        CLASSIFY_REQUIRE(rasterizer.cullStreamClusters(*commands, *pipelines[schedule == 1 ? 1 : 8], push));
                        // Inspect the queue before stable bin scatter overwrites it.
                        BufferBarrierDesc queueCopy{.buffer = &rasterizer.clusterBuffer(),
                            .before = ResourceState::General, .after = ResourceState::TransferSource};
                        commands->barrier({.buffers = &queueCopy, .bufferCount = 1});
                        commands->copyBuffer({.source = &rasterizer.clusterBuffer(), .destination = exactQueueReadback.get(),
                            .sourceOffset = 16 * sizeof(uint32_t), .size = uint64_t(capacity) * 16});
                        std::swap(queueCopy.before, queueCopy.after);
                        commands->barrier({.buffers = &queueCopy, .bufferCount = 1});
                    }
                    if (measure) {
                        CLASSIFY_REQUIRE(commands->resetTimestampQueries(*timing, 0, timedDispatches * 2));
                        const BufferBarrierDesc tagsReady{.buffer = &rasterizer.clusterBuffer(),
                            .before = ResourceState::General, .after = ResourceState::General};
                        for (uint32_t dispatch = 0; dispatch < timedDispatches; ++dispatch) {
                            // Swap AB/BA each pair to balance cache and clock drift.
                            const bool legacy = ((dispatch / 2 + dispatch % 2) & 1u) == 0u;
                            commands->barrier({.buffers = &tagsReady, .bufferCount = 1});
                            commands->bindComputePipeline(*pipelines[legacy ? 7 : 2]);
                            commands->pushBindlessData(&push, sizeof(push));
                            CLASSIFY_REQUIRE(commands->writeTimestamp(*timing, dispatch * 2, PipelineStageBits::TopOfPipe));
                            CLASSIFY_REQUIRE(commands->dispatchIndirect(rasterizer.candidateArguments()));
                            CLASSIFY_REQUIRE(commands->writeTimestamp(*timing, dispatch * 2 + 1, PipelineStageBits::BottomOfPipe));
                        }
                        commands->barrier({.buffers = &tagsReady, .bufferCount = 1});
                    }
                    if (schedule == 0 || test.maxPixels != 0) {
                        commands->bindComputePipeline(*pipelines[schedule == 0 ? 3 : schedule == 1 ? 2 : 7]);
                        commands->pushBindlessData(&push, sizeof(push));
                        CLASSIFY_REQUIRE(commands->dispatchIndirect(rasterizer.candidateArguments()));
                    }
                    if (schedule != 0) {
                        BufferBarrierDesc counterCopy{.buffer = &rasterizer.clusterBuffer(), .before = ResourceState::General, .after = ResourceState::TransferSource};
                        commands->barrier({.buffers = &counterCopy, .bufferCount = 1});
                        commands->copyBuffer({.source = &rasterizer.clusterBuffer(), .destination = readback.get(),
                            .destinationOffset = binBytes + recordBytes + 36, .size = 16});
                        std::swap(counterCopy.before,counterCopy.after);
                        commands->barrier({.buffers = &counterCopy, .bufferCount = 1});
                    }
                    CLASSIFY_REQUIRE(rasterizer.finishClusterBins(*commands));
                    if (schedule != 0) {
                        commands->bindBindlessHeap(*heap);
                        auto diagnosticPush = push;
                        diagnosticPush.hybridQueueBuffer = handles[16 + bufferSet].index;
                        BufferBarrierDesc ready{.buffer = &rasterizer.workloadBuffer(), .before = ResourceState::General, .after = ResourceState::General};
                        commands->barrier({.buffers = &ready, .bufferCount = 1});
                        commands->bindComputePipeline(*pipelines[5]);
                        commands->pushBindlessData(&diagnosticPush, sizeof(diagnosticPush));
                        commands->dispatch(1);
                        commands->barrier({.buffers = &ready, .bufferCount = 1});
                        commands->bindComputePipeline(*pipelines[6]);
                        CLASSIFY_REQUIRE(commands->dispatchIndirect(rasterizer.clusterArguments(), 4 * 3 * sizeof(uint32_t)));
                        ready.after = ResourceState::TransferSource;
                        commands->barrier({.buffers = &ready, .bufferCount = 1});
                        commands->copyBuffer({.source = &rasterizer.workloadBuffer(), .destination = readback.get(),
                            .destinationOffset = binBytes + recordBytes + 52, .size = 128});
                        std::swap(ready.before, ready.after);
                        commands->barrier({.buffers = &ready, .bufferCount = 1});
                    }
                    const BufferBarrierDesc copies[] = {
                        {.buffer = &rasterizer.clusterBuffer(), .before = ResourceState::ShaderRead, .after = ResourceState::TransferSource},
                        {.buffer = inputs[Records].get(), .before = ResourceState::General, .after = ResourceState::TransferSource},
                        {.buffer = &rasterizer.candidateArguments(), .before = ResourceState::IndirectArgument, .after = ResourceState::TransferSource}};
                    commands->barrier({.buffers = copies, .bufferCount = 3});
                    commands->copyBuffer({.source = &rasterizer.clusterBuffer(), .destination = readback.get(), .size = binBytes});
                    commands->copyBuffer({.source = inputs[Records].get(), .destination = readback.get(), .destinationOffset = binBytes, .size = recordBytes});
                    commands->copyBuffer({.source = &rasterizer.candidateArguments(), .destination = readback.get(), .destinationOffset = binBytes + recordBytes, .size = 36});
                    std::array<BufferBarrierDesc, 3> restore{copies[0], copies[1], copies[2]};
                    for (auto& barrier : restore) { std::swap(barrier.before, barrier.after); }
                    commands->barrier({.buffers = restore.data(), .bufferCount = 3});
                    CLASSIFY_REQUIRE(commands->end());
                    CommandBuffer* list[] = {commands.get()};
                    CLASSIFY_REQUIRE(queue->submit({.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()}));
                    CLASSIFY_REQUIRE(fence->wait()); submitted = true;
                    readback->invalidate();
                    const auto* mapped = static_cast<const uint32_t*>(readback->map());
                    if (!mapped) { return RhiTestResult::fail("Cannot map classification output"); }
                    std::vector<uint32_t> actual(mapped, mapped + (binBytes + recordBytes + 52 + 128) / 4);
                    readback->unmap();
                    if (schedule == 0) { reference = std::move(actual); continue; }
                    bool equal = std::equal(reference.begin(), reference.begin() + 16, actual.begin());
                    equal &= std::equal(reference.begin() + candidateBase, reference.begin() + candidateBase + 2 * reference[12], actual.begin() + candidateBase);
                    equal &= std::equal(reference.begin() + retryBase, reference.begin() + retryBase + test.groups, actual.begin() + retryBase);
                    equal &= std::equal(reference.begin() + binBytes / 4, reference.begin() + (binBytes + recordBytes) / 4, actual.begin() + binBytes / 4);
                    uint32_t visibleCount = 0;
                    for (uint32_t bin = 0; bin < 5; ++bin) {
                        visibleCount += actual[bin];
                        equal &= std::equal(reference.begin() + 16 + bin * capacity,
                            reference.begin() + 16 + bin * capacity + reference[bin], actual.begin() + 16 + bin * capacity);
                    }
                    const uint32_t* args = actual.data() + (binBytes + recordBytes) / 4;
                    const uint32_t* counters = actual.data() + (binBytes + recordBytes + 36) / 4;
                    const uint32_t classifyCount = counters[0];
                    if (schedule == 1 && test.maxPixels != 0) {
                        p1Exact = classifyCount; p1Software = counters[1]; p1Hardware = counters[2];
                        exactQueueReadback->invalidate();
                        const auto* queued = static_cast<const uint32_t*>(exactQueueReadback->map());
                        if (!queued) { return RhiTestResult::fail("Cannot map exact classification queue"); }
                        bool validContract = classifyCount <= capacity;
                        for (uint32_t i = 0; i < std::min(classifyCount, capacity); ++i) {
                            const uint32_t candidate = queued[4 * i];
                            if (candidate >= actual[12]) { validContract = false; break; }
                            const uint32_t record = actual[candidateBase + 2 * candidate];
                            if (record / 32 >= groups.size()) { validContract = false; break; }
                            const uint32_t instance = groups[record / 32].gpuSceneInstanceIndex;
                            validContract &= (instances[instance].identity[3] & 12u) == 0 && !test.tessellation;
                        }
                        exactQueueReadback->unmap();
                        if (!validContract) { return RhiTestResult::fail("P1 queued a forced-HW cluster"); }
                    }
                    if (schedule == 2 && test.maxPixels != 0) {
                        // Nothing was dropped or moved into SW: every removed exact
                        // entry must appear in the new cull-stage HW count instead.
                        equal &= p1Software == counters[1] && p1Hardware >= counters[2] &&
                            uint64_t(classifyCount) == uint64_t(p1Exact) + p1Hardware - counters[2];
                    }
                    if (measure) {
                        std::array<TimestampQueryResult, timedDispatches * 2> timestamps{};
                        CLASSIFY_REQUIRE(timing->readResults(0, uint32_t(timestamps.size()), timestamps.data()));
                        std::array<std::vector<double>, 2> times;
                        for (uint32_t dispatch = warmupPairs * 2; dispatch < timedDispatches; ++dispatch) {
                            if (!timestamps[dispatch * 2].available || !timestamps[dispatch * 2 + 1].available) {
                                return RhiTestResult::fail("Classification GPU timestamps unavailable");
                            }
                            const bool legacy = ((dispatch / 2 + dispatch % 2) & 1u) == 0u;
                            const double us = timing->durationMilliseconds(timestamps[dispatch * 2].value,
                                timestamps[dispatch * 2 + 1].value) * 1000.0;
                            times[legacy ? 0 : 1].push_back(us);
                            benchmark << caseIndex << ',' << phase << ',' << test.groups << ',' << classifyCount << ','
                                << dispatch / 2 - warmupPairs << ',' << (legacy ? "p0" : "p1") << ',' << us << ','
                                << int(slangShaderDebugMode()) << '\n';
                        }
                        for (auto& values : times) { std::sort(values.begin(), values.end()); }
                        constexpr uint32_t middle = measuredPairs / 2;
                        const double oldUs = (times[0][middle - 1] + times[0][middle]) * 0.5;
                        const double newUs = (times[1][middle - 1] + times[1][middle]) * 0.5;
                        spdlog::info("[Classify P1 exact queue] case={} workgroups={} p0_us={:.3f} p1_us={:.3f} reduction={:.2f}%",
                            caseIndex, classifyCount, oldUs, newUs, 100.0 * (oldUs - newUs) / oldUs);
                    }
                    std::array<uint64_t, 16> workload{};
                    std::memcpy(workload.data(), reinterpret_cast<const uint8_t*>(actual.data()) + binBytes + recordBytes + 52, 128);
                    if (workload[0] + workload[15] != actual[4] || workload[1] != workload[3] + workload[8] ||
                        workload[5] != workload[6] || workload[5] > workload[4] || workload[1] > workload[0] * 128) {
                        return RhiTestResult::fail("SW workload/bin accounting mismatch");
                    }
                    if (test.maxPixels != 0) { equal &= classifyCount + counters[1] + counters[2] == visibleCount; }
                    sawFastSoftware |= counters[1] != 0; sawFastHardware |= counters[2] != 0;
                    equal &= args[0] == std::min(classifyCount, 65535u) && args[1] == std::max(1u, (classifyCount + 65534u) / 65535u) && args[2] == 1;
                    if (test.maxPixels == 0) { equal &= actual[4] == 0; }
                    if (!equal) {
                        std::string detail;
                        for (uint32_t i=0; i<16; ++i) { detail += " h"+std::to_string(i)+"="+std::to_string(reference[i])+"/"+std::to_string(actual[i]); }
                        detail += " args="+std::to_string(args[0])+","+std::to_string(args[1]);
                        return RhiTestResult::fail("Classification mismatch in schedule " + std::to_string(schedule) + " case " + std::to_string(caseIndex) + " phase " + std::to_string(phase)+detail);
                    }
                    sawHardware |= actual[0] != 0; sawSoftware |= actual[4] != 0; saw2D |= args[1] > 1;
                    sawLate |= phase == 1 && visibleCount != 0;
                    for (uint32_t i = 0; i < test.groups; ++i) { sawRetry |= actual[retryBase + i] != 0; }
                }
            }
            ++caseIndex;
        }
        if (!sawFastSoftware || !sawFastHardware || !sawRetry || !sawLate || !sawSoftware || !sawHardware || !saw2D) { return RhiTestResult::fail("Missing paths fastSW/HW,retry,late,SW,HW,2D=" + std::to_string(sawFastSoftware)+std::to_string(sawFastHardware)+std::to_string(sawRetry)+std::to_string(sawLate)+std::to_string(sawSoftware)+std::to_string(sawHardware)+std::to_string(saw2D)); }
        return RhiTestResult::pass(std::to_string(std::size(cases)) + " early/late GPU cases comparing P1 and P0 cull/classify pairs against the independent reference; forced-HW queue exclusion and exact-to-HW accounting with cooperative raster decode, malformed headers/float3, metadata fast SW/HW and cull-only full HW: stable bins/IDs, HZB retry, clip boundaries, malformed payload, 1/8/32px, jitter, 2D survivors and empty reuse");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamClusterClassificationTest);
#undef CLASSIFY_REQUIRE
} // namespace
} // namespace metallic::tests
