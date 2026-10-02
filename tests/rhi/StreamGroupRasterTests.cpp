#include "Runtime/Render/Core/StreamRasterParameters.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include <bit>
#include <cmath>
#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

namespace metallic::tests {
namespace {
using namespace render;
#define GROUP_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RHITestResult::fail(std::string(#expr) + ": " + toString(checked)); } } while (false)

class StreamGroupRasterTest final : public RHITest {
public:
    StreamGroupRasterTest() { type = RHITestType::Rendering; name = "stream_group_raster_boundaries"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<Device> device;
        GROUP_REQUIRE(createDevice({.applicationName = "Stream group raster boundaries",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableAsyncCompute = true})
            .transform([&](auto value) { device = std::move(value); }));
        const auto& caps = device->capabilities();
        if (!caps.shaderBufferInt64Atomics || caps.subgroupSize != 32 || caps.minSubgroupSize != 32 || caps.maxSubgroupSize != 32) {
            return RHITestResult::skip("Requires fixed wave32 and 64-bit buffer atomics");
        }
        constexpr uint32_t extent = 64, pixelCount = extent * extent, pageBytes = 4096;
        enum Input { Header, Groups, Params, Pages, PageTable, Bindings, Bins, Pixels, Instances, Count };
        const uint32_t strides[] = {sizeof(MeshletStreamGPUActiveHeader), sizeof(MeshletStreamGPUActiveGroup),
            sizeof(MeshletStreamGPUParams), 4, sizeof(StreamPageTableEntry), sizeof(MeshletStreamGPURasterBindings),
            4, 8, sizeof(GPUSceneGPUInstanceRecord)};
        const uint32_t counts[] = {1, 1, 1, pageBytes / 4, 1, 1, 21, pixelCount + 16, 1};
        auto registry = device->resourceRegistry();
        if (!registry) { return RHITestResult::fail("Missing group raster registry"); }
        ParameterWriter fixtureWriter(*device, **registry);
        std::array<std::unique_ptr<Buffer>, Count> buffers;
        std::array<ShaderBuffer, Count> handles;
        for (size_t i=0; i<Count; ++i) {
            GROUP_REQUIRE(device->createBuffer({.size = uint64_t(strides[i])*counts[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostUpload,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto v) { buffers[i] = std::move(v); }));
            handles[i] = fixtureWriter.buffer(buffers[i].get());
        }
        const auto upload = [&](size_t index, const void* data, size_t bytes) -> Result<> {
            void* mapped = buffers[index]->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data, bytes); buffers[index]->flush(); buffers[index]->unmap(); return {};
        };
        std::array<ComputeKernel, 9> kernels;
        const char* entries[] = {"streamClusterRasterWorkControlMain", "streamClusterRasterGroup32Main",
            "streamClusterRasterGroup64Main", "streamClusterRasterGroup128Main",
            "streamClusterRasterMain", "streamClusterRasterLegacyMain", "streamClusterRasterPlaneMain", "streamClusterRasterCooperativeMain", "streamClusterRasterWorkBinsMain"};
        for (size_t i=0; i<std::size(entries); ++i) {
            ShaderCompileResult compiled;
            auto result = compileSlangShaderToSpirv({
                .moduleName = i == 8 ? "Features/GPUDriven/GPUDrivenStreamWorkRaster" : i >= 4 ? "Features/GPUDriven/GPUDrivenStreamAsset" : i ? "Features/GPUDriven/GPUDrivenStreamGroupRaster" : "Features/GPUDriven/GPUDrivenStreamWorkRaster",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled.diagnostics)
                .transform([&](auto v) { compiled = std::move(v); });
            if (!result) { return RHITestResult::fail(compiled.diagnostics); }
            std::string log;
            GROUP_REQUIRE(kernels[i].initialize(*device, {.spirv = compiled.spirv,
                .parameters = parameterAbi<StreamRasterParameters>(kStreamRasterABI, ParameterTransport::InlinePush)}, log));
        }
        struct QueueRecording {
            Queue* queue = nullptr;
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Fence> fence;
            bool submitted = false;
        };
        std::array<QueueRecording, 2> recordings;
        recordings[0].queue = device->getQueue(QueueType::Graphics);
        Queue* compute = device->getQueue(QueueType::Compute);
        const uint32_t queueCount = compute && !compute->sameQueue(*recordings[0].queue) ? 2u : 1u;
        recordings[1].queue = compute;
        spdlog::info("[StreamGroupRaster] queue coverage: {}",
            queueCount == 2 ? "graphics and distinct compute" : "graphics only (no distinct compute queue)");
        for (uint32_t i = 0; i < queueCount; ++i) {
            auto& recording = recordings[i];
            GROUP_REQUIRE(device->createCommandPool(*recording.queue).transform([&](auto v) { recording.pool = std::move(v); }));
            GROUP_REQUIRE(recording.pool->createCommandBuffer().transform([&](auto v) { recording.commands = std::move(v); }));
            GROUP_REQUIRE(device->createFence({}).transform([&](auto v) { recording.fence = std::move(v); }));
        }
        std::unique_ptr<Buffer> readback;
        GROUP_REQUIRE(device->createBuffer({.size = buffers[Pixels]->desc().size, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto v) { readback = std::move(v); }));
        std::unique_ptr<CommandPool> handoffPool;
        std::unique_ptr<CommandBuffer> producer, consumer;
        std::unique_ptr<Semaphore> handoff;
        std::unique_ptr<Buffer> resetPixels;
        uint64_t handoffValue = 0;
        if (queueCount == 2) {
            GROUP_REQUIRE(device->createCommandPool(*recordings[0].queue).transform([&](auto v) { handoffPool = std::move(v); }));
            GROUP_REQUIRE(handoffPool->createCommandBuffer().transform([&](auto v) { producer = std::move(v); }));
            GROUP_REQUIRE(handoffPool->createCommandBuffer().transform([&](auto v) { consumer = std::move(v); }));
            GROUP_REQUIRE(device->createSemaphore({}).transform([&](auto v) { handoff = std::move(v); }));
            GROUP_REQUIRE(device->createBuffer({.size = buffers[Pixels]->desc().size,
                .usage = BufferUsageBits::TransferSource, .memoryLocation = MemoryLocation::HostUpload})
                .transform([&](auto v) { resetPixels = std::move(v); }));
            std::vector<uint64_t> reset(pixelCount + 16, 0);
            std::fill(reset.begin() + pixelCount, reset.end(), 0x1234567887654321ull);
            void* mapped = resetPixels->map();
            if (!mapped) { return RHITestResult::fail("Reset upload map failed"); }
            std::memcpy(mapped, reset.data(), reset.size() * sizeof(uint64_t));
            resetPixels->flush(); resetPixels->unmap();
        }
        struct Case { uint32_t vertices, triangles, selected; bool float3, reversed, reflected; uint32_t fault = 0; bool varyingDepth = false, perspective = false; };
        std::vector<Case> cases;
        for (uint32_t vertices : {3u,31u,32u,33u,63u,64u,65u,127u,128u}) {
            for (uint32_t triangles : {0u,1u,31u,32u,33u,63u,64u,65u,127u,128u}) {
                cases.push_back({vertices, triangles, triangles ? triangles-1 : 0, false, true, false});
                cases.push_back({vertices, triangles, triangles ? triangles-1 : 0, true, false, true});
            }
        }
        // Isolate every triangle ID, so atomic max cannot hide a missing middle iteration.
        for (uint32_t triangle=0; triangle<128; ++triangle) { cases.push_back({128,128,triangle,true,true,false}); }
        for (uint32_t fault=1; fault<=5; ++fault) { cases.push_back({128,128,127,false,true,false,fault}); }
        // Constant-Z triangles cannot expose projection/depth arithmetic differences.
        for (bool perspective : {false, true}) for (bool reversed : {false, true})
            for (bool reflected : {false, true}) for (bool float3 : {false, true}) {
                cases.push_back({128, 128, 127, float3, reversed, reflected, 0, true, perspective});
            }
        uint32_t index = 0;
        for (const auto& test : cases) {
            scene::MeshletStreamPayloadHeader header;
            header.clusterCount = 1; header.vertexCount = 128; header.triangleIndexCount = 384;
            header.clusterOffsetBytes = sizeof(header);
            header.positionOffsetBytes = header.clusterOffsetBytes + sizeof(scene::MeshletStreamPayloadCluster);
            header.triangleOffsetBytes = header.positionOffsetBytes + 128*(test.float3 ? 12 : 16);
            header.payloadByteSize = header.triangleOffsetBytes + 384; header.attributeFlags = 1;
            header.positionFormat = uint32_t(test.float3 ? scene::MeshletStreamPayloadFormat::Float32x3 : scene::MeshletStreamPayloadFormat::Float32x4);
            scene::MeshletStreamPayloadCluster cluster;
            cluster.vertexCount = test.fault == 1 ? 0 : test.fault == 2 ? 129 : test.vertices;
            cluster.triangleCount = test.fault == 3 ? 129 : test.triangles;
            std::array<uint8_t, pageBytes> page{};
            std::memcpy(page.data(), &header, sizeof(header));
            std::memcpy(page.data()+header.clusterOffsetBytes, &cluster, sizeof(cluster));
            for (uint32_t v=0; v<128; ++v) {
                float position[4] = {0,0,1,1};
                // A visible 4x4-pixel right triangle uses the final three vertices.
                if (v == test.vertices-3) { position[0]=-.0625f; position[1]=-.0625f; }
                if (v == test.vertices-2) { position[0]= .0625f; position[1]=-.0625f; }
                if (v == test.vertices-1) { position[0]=-.0625f; position[1]= .0625f; }
                if (test.varyingDepth && v >= test.vertices - 3 && v < test.vertices) {
                    const float depths[] = {0.731f, 1.137f, 2.219f};
                    position[2] = depths[v - (test.vertices - 3)];
                    // Keep a broad footprint despite different perspective W values.
                    const float scale = 8.0f * (test.perspective ? position[2] : 1.0f);
                    position[0] *= scale; position[1] *= scale;
                }
                std::memcpy(page.data()+header.positionOffsetBytes+v*(test.float3 ? 12 : 16),position,test.float3 ? 12 : 16);
            }
            if (test.triangles) {
                for (uint32_t i=0; i<3; ++i) { page[header.triangleOffsetBytes+test.selected*3+i] = uint8_t(test.vertices-3+i); }
            }
            MeshletStreamGPUActiveHeader active{.activeGroupCount = 1, .activeGroupCapacity = 1, .maxActiveGroupClusters = 1};
            MeshletStreamGPUActiveGroup group;
            group.clusterCount = 1; group.clusterSelectionMask = test.fault == 5 ? 0 : 1; group.gpuSceneInstanceIndex = 0;
            group.world0[0] = test.reflected ? -1.f : 1.f; group.world1[1] = group.world2[2] = group.world3[3] = 1;
            MeshletStreamGPUParams params;
            params.center[2]=1; params.upProjection[1]=1; params.upProjection[3]=test.perspective ? 0.f : 1.f;
            params.viewport[0]=1; params.viewport[1]=params.viewport[2]=extent; params.viewport[3]=1.5707963f;
            params.clipOrtho[0]=.01f; params.clipOrtho[1]=10; params.clipOrtho[2]=2; params.clipOrtho[3]=test.reversed ? 1.f : 0.f;
            params.pageBufferBytes=pageBytes; params.drawTaskCount=2; params.scenePageCount=1;
            StreamPageTableEntry table;
            table.deviceOffsetAndState=packStreamPageTableEntry(0,test.fault == 4 ? MeshletStreamPageResidencyState::Unloaded : MeshletStreamPageResidencyState::Resident);
            MeshletStreamGPURasterBindings bindings{.visibleRecordBase=371, .visibleRecordCapacity=1,
                .gpuSceneInstanceBuffer = handles[Instances]};
            updateStreamRasterResourceFlags(bindings);
            GPUSceneGPUInstanceRecord instance;
            instance.identity[3]=2; // two-sided; reflected winding must preserve coverage
            std::array<uint32_t,21> bins{};
            bins[4]=bins[5]=1; bins[6]=bins[7]=extent; bins[9]=test.reversed ? 1 : 0;
            bins[10]=caps.subPixelPrecisionBits;
            GROUP_REQUIRE(upload(Header,&active,sizeof(active))); GROUP_REQUIRE(upload(Groups,&group,sizeof(group)));
            GROUP_REQUIRE(upload(Params,&params,sizeof(params))); GROUP_REQUIRE(upload(Pages,page.data(),page.size()));
            GROUP_REQUIRE(upload(PageTable,&table,sizeof(table))); GROUP_REQUIRE(upload(Bindings,&bindings,sizeof(bindings)));
            GROUP_REQUIRE(upload(Instances,&instance,sizeof(instance))); GROUP_REQUIRE(upload(Bins,bins.data(),sizeof(bins)));
            std::array<std::vector<uint64_t>, 9> graphicsOutputs;
            for (uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
                auto& recording = recordings[queueIndex];
                auto* queue = recording.queue;
                auto& pool = recording.pool;
                auto& commands = recording.commands;
                auto& fence = recording.fence;
                auto& submitted = recording.submitted;
                // Every dispatch completes before host uploads or another queue uses these buffers.
                std::vector<uint64_t> reference;
                for (uint32_t variant=0; variant<std::size(entries); ++variant) {
                    std::vector<uint64_t> pixels(pixelCount+16,0);
                    std::fill(pixels.begin()+pixelCount,pixels.end(),0x1234567887654321ull);
                    if (queueIndex != 0) {
                        // A missing GPU producer must fail instead of observing a valid host reset.
                        std::fill(pixels.begin(), pixels.end(), ~uint64_t(0));
                    }
                    GROUP_REQUIRE(upload(Pixels,pixels.data(),pixels.size()*8));
                    if (submitted) { GROUP_REQUIRE(fence->reset()); GROUP_REQUIRE(pool->reset()); }
                    GROUP_REQUIRE(commands->begin());
                    {
                        auto registry = device->resourceRegistry();
                        if (!registry) { return RHITestResult::fail("Missing raster registry"); }
                        ParameterWriter writer(*device, **registry, commands->frameContext());
                        const StreamRasterParameters raster{
                            .settings = {writer.data(&params, sizeof(params), 16), 1, sizeof(params)},
                            .pages = writer.buffer(buffers[Pages].get()), .groups = writer.buffer(buffers[Groups].get()),
                            .header = writer.buffer(buffers[Header].get()), .pageTable = writer.buffer(buffers[PageTable].get()),
                            .instances = writer.buffer(buffers[Instances].get()), .bins = writer.buffer(buffers[Bins].get()),
                            .pixels = writer.buffer(buffers[Pixels].get()), .visibleRecordBase = bindings.visibleRecordBase,
                            .visibleRecordCapacity = bindings.visibleRecordCapacity, .hasInstances = 1,
                        };
                        auto encoded = writer.encode(raster, kStreamRasterABI, ParameterTransport::InlinePush);
                        if (!encoded) { return RHITestResult::fail("Cannot encode raster parameters"); }
                        GROUP_REQUIRE(kernels[variant].dispatch(*commands, *encoded, 2));
                    }
                    BufferBarrierDesc barrier{.buffer=buffers[Pixels].get(),
                        .before={PipelineStageBits::ComputeShader,AccessBits::ShaderWrite},
                        .after={PipelineStageBits::Transfer,AccessBits::TransferRead}};
                    GROUP_REQUIRE(commands->synchronize({.buffers={&barrier,1}}));
                    auto src=buffers[Pixels]->slice({0,readback->desc().size}); auto dst=readback->slice({0,readback->desc().size});
                    if (!src || !dst) { return RHITestResult::fail("Invalid readback slice"); }
                    if (queueIndex == 0) {
                        GROUP_REQUIRE(commands->copyBuffer(*src, *dst));
                        GROUP_REQUIRE(commands->end()); CommandBuffer* list[]={commands.get()};
                        GROUP_REQUIRE(queue->submit({.commandBuffers={list,1},.signalFence=fence.get()}));
                    } else {
                        GROUP_REQUIRE(commands->end());
                        GROUP_REQUIRE(handoffPool->reset());
                        GROUP_REQUIRE(producer->begin());
                        auto reset = resetPixels->slice({0, readback->desc().size});
                        if (!reset) { return RHITestResult::fail("Invalid reset slice"); }
                        GROUP_REQUIRE(producer->copyBuffer(*reset, *src));
                        GROUP_REQUIRE(producer->end());
                        GROUP_REQUIRE(consumer->begin());
                        GROUP_REQUIRE(consumer->copyBuffer(*src, *dst));
                        GROUP_REQUIRE(consumer->end());
                        const SemaphoreSubmitDesc ready{handoff.get(), ++handoffValue};
                        const SemaphoreSubmitDesc done{handoff.get(), ++handoffValue};
                        CommandBuffer* produce[] = {producer.get()};
                        CommandBuffer* raster[] = {commands.get()};
                        CommandBuffer* consume[] = {consumer.get()};
                        GROUP_REQUIRE(recordings[0].queue->submit({.commandBuffers = produce, .signalSemaphores = {&ready, 1}}));
                        GROUP_REQUIRE(queue->submit({.waitSemaphores = {&ready, 1}, .commandBuffers = raster,
                            .signalSemaphores = {&done, 1}}));
                        GROUP_REQUIRE(recordings[0].queue->submit({.waitSemaphores = {&done, 1}, .commandBuffers = consume,
                            .signalFence = fence.get()}));
                    }
                    GROUP_REQUIRE(fence->wait()); submitted=true; readback->invalidate();
                    const void* mapped=readback->map();
                    if (!mapped) { return RHITestResult::fail("Readback map failed"); }
                    std::memcpy(pixels.data(),mapped,pixels.size()*8); readback->unmap();
                    const auto covered=std::count_if(pixels.begin(),pixels.begin()+pixelCount,[](uint64_t v){return v!=0;});
                    const bool expected = test.triangles && !test.fault;
                    if ((covered>0)!=expected || !std::all_of(pixels.begin()+pixelCount,pixels.end(),[](uint64_t v){return v==0x1234567887654321ull;})) {
                        return RHITestResult::fail("Missing coverage or guard corruption case="+std::to_string(index)+" variant="+entries[variant]);
                    }
                    for (size_t i=0; i<pixelCount; ++i) {
                        if (pixels[i] && uint32_t(pixels[i]) != ((372u<<7u)|test.selected)) {
                            return RHITestResult::fail("Unexpected triangle/stable record ID case="+std::to_string(index));
                        }
                    }
                    if (queueIndex == 0) { graphicsOutputs[variant] = pixels; }
                    else if (pixels != graphicsOutputs[variant]) {
                        const auto mismatch = std::mismatch(pixels.begin(), pixels.end(), graphicsOutputs[variant].begin());
                        return RHITestResult::fail("Cross-queue packed output mismatch case=" + std::to_string(index) +
                            " variant=" + entries[variant] + " pixel=" + std::to_string(mismatch.first - pixels.begin()) +
                            " graphics=" + std::to_string(*mismatch.second) + " compute=" + std::to_string(*mismatch.first));
                    }
                    if (!variant) { reference=pixels; }
                    else if (variant == 6) {
                        for (size_t i = 0; i < pixelCount; ++i) {
                            uint32_t actualDepth = uint32_t(pixels[i] >> 32u), expectedDepth = uint32_t(reference[i] >> 32u);
                            if (!test.reversed) { actualDepth = ~actualDepth; expectedDepth = ~expectedDepth; }
                            if (uint32_t(pixels[i]) != uint32_t(reference[i]) || (pixels[i] &&
                                std::abs(std::bit_cast<float>(actualDepth) - std::bit_cast<float>(expectedDepth)) > 1e-6f)) {
                                return RHITestResult::fail("Plane coverage/depth mismatch case=" + std::to_string(index));
                            }
                        }
                    }
                    else if (pixels!=reference) { return RHITestResult::fail("Packed depth/visibility mismatch case="+std::to_string(index)+" variant="+entries[variant]); }
                }
            }
            ++index;
        }
        return RHITestResult::pass(std::string(queueCount == 2 ? "Graphics/compute packed outputs byte-equal with GPU producer/consumer handoff; " : "Distinct compute queue unavailable; graphics only; ") + std::to_string(cases.size())+" cases x nine raster entrypoints: tail vertices/triangles, each triangle ID, float3/4, normal/reversed Z, reflected transforms, varying-depth orthographic/perspective projection, rejected pages and guard pixels; eight packed outputs byte-equal; plane preserves coverage/IDs and depth within 1e-6");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamGroupRasterTest);
#undef GROUP_REQUIRE
} // namespace
} // namespace metallic::tests
