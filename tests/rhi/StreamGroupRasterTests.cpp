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
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
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
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto v) { buffers[i] = std::move(v); }));
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
        Queue* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        GROUP_REQUIRE(device->createCommandPool(*queue).transform([&](auto v) { pool = std::move(v); }));
        std::unique_ptr<CommandBuffer> commands;
        GROUP_REQUIRE(pool->createCommandBuffer().transform([&](auto v) { commands = std::move(v); }));
        std::unique_ptr<Fence> fence;
        GROUP_REQUIRE(device->createFence({}).transform([&](auto v) { fence = std::move(v); }));
        std::unique_ptr<Buffer> readback;
        GROUP_REQUIRE(device->createBuffer({.size = buffers[Pixels]->desc().size, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto v) { readback = std::move(v); }));
        struct Case { uint32_t vertices, triangles, selected; bool float3, reversed, reflected; uint32_t fault = 0; };
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
        bool submitted = false;
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
            params.center[2]=1; params.upProjection[1]=1; params.upProjection[3]=1;
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
            std::vector<uint64_t> reference;
            for (uint32_t variant=0; variant<std::size(entries); ++variant) {
                std::vector<uint64_t> pixels(pixelCount+16,0);
                std::fill(pixels.begin()+pixelCount,pixels.end(),0x1234567887654321ull);
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
                if (!src || !dst) { return RHITestResult::fail("Invalid readback slice"); } GROUP_REQUIRE(commands->copyBuffer(*src,*dst));
                GROUP_REQUIRE(commands->end()); CommandBuffer* list[]={commands.get()};
                GROUP_REQUIRE(queue->submit({.commandBuffers={list,1},.signalFence=fence.get()}));
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
            ++index;
        }
        return RHITestResult::pass(std::to_string(cases.size())+" cases x nine raster entrypoints: tail vertices/triangles, each triangle ID, float3/4, normal/reversed Z, reflected transforms, rejected pages and guard pixels; eight packed outputs byte-equal; plane preserves coverage/IDs and depth within 1e-6");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamGroupRasterTest);
#undef GROUP_REQUIRE
} // namespace
} // namespace metallic::tests
