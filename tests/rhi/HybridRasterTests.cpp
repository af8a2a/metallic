#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "RHITest.h"
#include "Runtime/Task/TaskSystem.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderSample.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

#define HYBRID_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RHITestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

class HybridRasterDepthTest final : public RHITest {
public:
    HybridRasterDepthTest() { type = RHITestType::Rendering; name = "hybrid_raster_depth_coverage_and_overflow"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Hybrid raster regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableMeshShader = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless heap"); }
        HYBRID_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RHITestResult::skip("Requires 64-bit buffer atomics and <=8 subpixel bits");
        }
        constexpr uint32_t width = 97, height = 73, pixelCount = width * height;
        std::vector<std::array<float, 4>> vertices;
        auto triangle = [&](std::array<float, 3> a, std::array<float, 3> b, std::array<float, 3> c, bool perspective = false) {
            uint32_t corner = 0;
            for (auto point : {a, b, c}) {
                const float w = perspective ? 0.7f + float(corner++) * 0.6f : 1.0f;
                vertices.push_back({(point[0] / width * 2.0f - 1.0f) * w,
                    (point[1] / height * 2.0f - 1.0f) * w, point[2] * w, w});
            }
        };
        // A large HW receiver with overlapping small foreground/background triangles.
        triangle({0, 0, .48f}, {0, 73, .48f}, {97, 0, .48f});
        triangle({97, 0, .48f}, {0, 73, .48f}, {97, 73, .48f});
        for (uint32_t i = 0; i < 180; ++i) {
            const float x = float((i * 17) % 93) + .17f;
            const float y = float((i * 13) % 69) + .31f;
            const float size = .3f + float(i % 19);
            const float z = .05f + float(i % 89) * .01f;
            auto a = std::array<float, 3>{x, y, z};
            auto b = std::array<float, 3>{x + .2f, y + size, z + .003f};
            auto c = std::array<float, 3>{x + size, y + .4f, z + .007f};
            if (i % 3 == 0) { std::swap(b, c); }
            triangle(a, b, c, i % 2 != 0);
        }
        // Adjacent triangles through exact pixel centers exercise the top-left rule.
        triangle({24.5f, 30.5f, .97f}, {24.5f, 38.5f, .97f}, {32.5f, 30.5f, .97f});
        triangle({32.5f, 30.5f, .97f}, {24.5f, 38.5f, .97f}, {32.5f, 38.5f, .97f});
        // Near/far clipping, viewport clipping and degenerate/subpixel coverage.
        triangle({5, 7, -.2f}, {5, 20, .3f}, {17, 7, .5f});
        triangle({40, 7, 1.2f}, {40, 20, .7f}, {52, 7, .5f});
        triangle({-3, 48, .9f}, {-3, 57, .91f}, {5, 48, .92f});
        triangle({2, 2, .8f}, {2, 2, .8f}, {3, 3, .8f});

        triangle({62, 7, 0.f}, {62, 13, 0.f}, {68, 7, 0.f});
        triangle({74, 7, 1.f}, {74, 13, 1.f}, {80, 7, 1.f});

        std::unique_ptr<Buffer> input;
        HYBRID_REQUIRE(device->createBuffer({.size = vertices.size() * 16, .structureStride = 16,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { input = std::move(rhiValue); }));
        void* mapped = input->map();
        if (!mapped) { return RHITestResult::fail("Vertex upload map failed"); }
        std::memcpy(mapped, vertices.data(), vertices.size() * 16); input->flush(); input->unmap();
        std::unique_ptr<BindlessHeap> heap;
        HYBRID_REQUIRE(device->createBindlessHeap({.maxBuffers = 2}).transform([&](auto rhiValue) { heap = std::move(rhiValue); }));
        BindlessHandle inputHandle, queueHandle;
        HYBRID_REQUIRE(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { inputHandle = std::move(rhiValue); })); HYBRID_REQUIRE(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { queueHandle = std::move(rhiValue); }));
        HYBRID_REQUIRE(heap->writeStorageBuffer(inputHandle, *input));
        // Compare the new shared-vertex/integer-step SW kernel to the legacy
        // kernel before testing either against HW. Include both subpixel grids.
        {
            ShaderCompileResult compiled;
            const char* paths[] = {PROJECT_SOURCE_DIR "/Shaders"};
            HYBRID_REQUIRE(compileSlangShaderToSpirv({
                .moduleName = "PreparedRasterProbe",
                .entryPointName = "compareMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .additionalSearchPaths = {paths, 1},
            }, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); }));
            log=compiled.diagnostics;
            std::unique_ptr<ShaderModule> shader;
            std::unique_ptr<ComputePipeline> compute;
            HYBRID_REQUIRE(device->createShaderModule({.spirv = compiled.spirv}).transform([&](auto rhiValue) { shader = std::move(rhiValue); }));
            HYBRID_REQUIRE(device->createComputePipeline({.computeShader = {shader.get()}, .usesBindlessHeap = true, .bindlessUserPushDataSize = 40}).transform([&](auto rhiValue) { compute = std::move(rhiValue); }));
            ShaderCompileResult workCompiled;
            HYBRID_REQUIRE(compileSlangShaderToSpirv({
                .moduleName = "PreparedRasterProbe",
                .entryPointName = "compareWorkBinsMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .additionalSearchPaths = {paths, 1},
            }, workCompiled.diagnostics).transform([&](auto value) { workCompiled = std::move(value); }));
            std::unique_ptr<ShaderModule> workShader;
            std::unique_ptr<ComputePipeline> workCompute;
            HYBRID_REQUIRE(device->createShaderModule({.spirv = workCompiled.spirv}).transform([&](auto rhiValue) { workShader = std::move(rhiValue); }));
            HYBRID_REQUIRE(device->createComputePipeline({.computeShader = {workShader.get()}, .usesBindlessHeap = true, .bindlessUserPushDataSize = 40}).transform([&](auto rhiValue) { workCompute = std::move(rhiValue); }));
            ShaderCompileResult workloadCompiled;
            const auto workloadResult = compileSlangShaderToSpirv({
                .moduleName = "PreparedRasterProbe",
                .entryPointName = "verifyWorkloadMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .additionalSearchPaths = {paths, 1},
            }, workloadCompiled.diagnostics).transform([&](auto value) { workloadCompiled = std::move(value); });
            log=workloadCompiled.diagnostics;
            HYBRID_REQUIRE(workloadResult);
            std::unique_ptr<ShaderModule> workloadShader;
            std::unique_ptr<ComputePipeline> workloadCompute;
            HYBRID_REQUIRE(device->createShaderModule({.spirv = workloadCompiled.spirv}).transform([&](auto rhiValue) { workloadShader = std::move(rhiValue); }));
            HYBRID_REQUIRE(device->createComputePipeline({.computeShader = {workloadShader.get()}, .usesBindlessHeap = true, .bindlessUserPushDataSize = 40}).transform([&](auto rhiValue) { workloadCompute = std::move(rhiValue); }));
            std::unique_ptr<BindlessHeap> compareHeap;
            HYBRID_REQUIRE(device->createBindlessHeap({.maxBuffers=3}).transform([&](auto rhiValue) { compareHeap = std::move(rhiValue); }));
            BindlessHandle vertexHandle;
            HYBRID_REQUIRE(compareHeap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { vertexHandle = std::move(rhiValue); }));
            HYBRID_REQUIRE(compareHeap->writeStorageBuffer(vertexHandle,*input));
            std::array<std::unique_ptr<Buffer>,2> pixels;
            std::array<BindlessHandle,2> handles;
            for (size_t i=0;i<2;++i) {
                HYBRID_REQUIRE(device->createBuffer({.size=pixelCount*8,.structureStride=8,.usage=BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                    .memoryLocation=MemoryLocation::HostUpload}).transform([&](auto rhiValue) { pixels[i] = std::move(rhiValue); }));
                HYBRID_REQUIRE(compareHeap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { handles[i] = std::move(rhiValue); }));
                HYBRID_REQUIRE(compareHeap->writeStorageBuffer(handles[i],*pixels[i]));
            }
            auto* queue=device->getQueue(QueueType::Graphics);
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Fence> fence;
            HYBRID_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); })); HYBRID_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
            HYBRID_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
            bool submitted=false;
            size_t written=0;
            for (uint32_t bits : {4u,8u}) for (uint32_t reversed : {0u,1u}) for (uint32_t sided : {0u,1u}) for (uint32_t plane : {0u,1u,2u,3u,4u}) for (uint32_t count : {0u,1u,127u,128u,129u,uint32_t(vertices.size()/3)}) {
                if ((plane < 2u || plane == 4u) && count != vertices.size()/3) { continue; }
                if (submitted) { HYBRID_REQUIRE(fence->reset()); HYBRID_REQUIRE(pool->reset()); }
                for (auto& output:pixels) {
                    auto* data=output->map(); if (!data) { return RHITestResult::fail("Prepared raster map failed"); }
                    std::memset(data,0,pixelCount*8); output->flush(); output->unmap();
                }
                HYBRID_REQUIRE(commands->begin());
                commands->bindBindlessHeap(*compareHeap); if (auto commandResult = commands->bindExecution((plane==4u ? *workloadCompute : plane>=2u ? *workCompute : *compute).execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
                uint32_t push[]={vertexHandle.shaderIndex,handles[0].shaderIndex,handles[1].shaderIndex,width,height,reversed,sided,bits,plane,count};
                commands->pushBindlessData(push,sizeof(push));
                const uint32_t lanes=plane>=2u ? 128u : 64u;
                commands->dispatch(std::max(1u,(push[9]+lanes-1u)/lanes));
                HYBRID_REQUIRE(commands->end()); CommandBuffer* list[]={commands.get()};
                HYBRID_REQUIRE(queue->submit({.commandBuffers = {list, 1}, .signalFence = fence.get()}));
                HYBRID_REQUIRE(fence->wait()); submitted=true;
                std::array<std::vector<uint64_t>,2> values;
                for (size_t i=0;i<2;++i) {
                    pixels[i]->invalidate(); const auto* data=static_cast<const uint64_t*>(pixels[i]->map());
                    if (!data) { return RHITestResult::fail("Prepared raster read failed"); }
                    values[i].assign(data,data+pixelCount); pixels[i]->unmap();
                }
                if (plane == 4u) {
                    if (values[1][0] != 0 || values[1][1] < 1000) { return RHITestResult::fail("SW workload coverage/atomic attempt count mismatch"); }
                    continue;
                }
                for (size_t i=0;i<pixelCount;++i) {
                    auto a=values[0][i],b=values[1][i]; written+=uint32_t(a)!=0;
                    if (plane!=1u && a!=b) { return RHITestResult::fail("Exact SW is not bit exact mode="+std::to_string(plane)+" count="+std::to_string(count)+" pixel="+std::to_string(i)+" bits="+std::to_string(bits)+" values="+std::to_string(a)+"/"+std::to_string(b)); }
                    uint32_t za=uint32_t(a>>32),zb=uint32_t(b>>32); if (!reversed) { za=~za; zb=~zb; }
                    if (plane==1u && (uint32_t(a)!=uint32_t(b) || (uint32_t(a)!=0 && std::abs(std::bit_cast<float>(za)-std::bit_cast<float>(zb))>2e-6f))) {
                        return RHITestResult::fail("Incremental plane coverage/ID/depth tolerance mismatch at "+std::to_string(i));
                    }
                }
            }
            if (written<1000) { return RHITestResult::fail("Prepared SW fixture did not cover enough pixels"); }
        }
        std::array<std::unique_ptr<ShaderModule>, 2> shaders;
        const char* entries[] = {"probeMeshMain", "probeFragmentMain"};
        const char* capabilities[] = {"spvMeshShadingEXT"};
        for (size_t i = 0; i < shaders.size(); ++i) {
            ShaderCompileResult shader;
            const auto compiled = compileSlangShaderToSpirv({
                .moduleName = "HybridRasterProbe",
                .entryPointName = entries[i],
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .capabilities = {capabilities, 1},
            }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            log = shader.diagnostics; HYBRID_REQUIRE(compiled);
            HYBRID_REQUIRE(device->createShaderModule({.spirv = shader.spirv}).transform([&](auto rhiValue) { shaders[i] = std::move(rhiValue); }));
        }
        std::array<std::unique_ptr<Texture>, 2> textures;
        std::array<std::unique_ptr<TextureView>, 2> views;
        std::array<std::unique_ptr<Buffer>, 2> readback;
        for (size_t i = 0; i < 2; ++i) {
            const auto format = i == 0 ? Format::R32Uint : Format::D32Sfloat;
            HYBRID_REQUIRE(device->createTexture({.usage = TextureUsageBits::TransferSource |
                (i == 0 ? TextureUsageBits::ColorAttachment : TextureUsageBits::DepthStencilAttachment),
                .format = format, .width = width, .height = height}).transform([&](auto rhiValue) { textures[i] = std::move(rhiValue); }));
            HYBRID_REQUIRE(device->createTextureView(*textures[i], {.format = format}).transform([&](auto rhiValue) { views[i] = std::move(rhiValue); }));
            HYBRID_REQUIRE(device->createBuffer({.size = pixelCount * 4, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback[i] = std::move(rhiValue); }));
        }
        std::unique_ptr<Buffer> queueReadback, pixelReadback;
        HYBRID_REQUIRE(device->createBuffer({.size = 32, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { queueReadback = std::move(rhiValue); }));
        HYBRID_REQUIRE(device->createBuffer({.size = pixelCount * 8, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { pixelReadback = std::move(rhiValue); }));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        HYBRID_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        HYBRID_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); })); HYBRID_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
        bool submitted = false;
        size_t softwarePixels = 0, overflows = 0, cases = 0;
        for (bool reversed : {false, true}) {
            for (bool doubleSided : {false, true}) {
                std::unique_ptr<GraphicsPipeline> pipeline;
                HYBRID_REQUIRE(device->createGraphicsPipeline({
                    .meshShader = {shaders[0].get()},
                    .fragmentShader = {shaders[1].get()},
                    .colorFormat = Format::R32Uint,
                    .depthStencilFormat = Format::D32Sfloat,
                    .rasterization = {.cullMode = doubleSided ? CullMode::None : CullMode::Back, .frontFace = FrontFace::CounterClockwise},
                    .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true,
                        .depthCompareOp = reversed ? CompareOp::GreaterEqual : CompareOp::LessEqual},
                    .usesBindlessHeap = true,
                }).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); }));
                std::vector<uint32_t> referenceIds;
                std::vector<float> referenceDepth;
                for (uint32_t configuration = 0; configuration < 5; ++configuration) {
                    VisibilityHybridRasterizer rasterizer;
                    const uint32_t capacity = configuration == 4 ? 1u : 262144u;
                    // Simulate output-size allocation followed by DLSS render-size selection.
                    HYBRID_REQUIRE(rasterizer.initialize(*device, width * 2, height * 2, log, capacity));
                    auto* pixelAllocation = &rasterizer.pixelBuffer();
                    auto* clusters = &rasterizer.clusterBuffer();
                    HYBRID_REQUIRE(rasterizer.setRenderExtent(width, height));
                    HYBRID_REQUIRE(rasterizer.setRenderExtent(width * 2, height * 2));
                    HYBRID_REQUIRE(rasterizer.setRenderExtent(width, height));
                    if (rasterizer.setRenderExtent(0, height) || rasterizer.setRenderExtent(width * 3, height * 3) ||
                        rasterizer.width() != width || rasterizer.height() != height ||
                        &rasterizer.pixelBuffer() != pixelAllocation || &rasterizer.clusterBuffer() != clusters) {
                        return RHITestResult::fail("Hybrid extent reuse changed resources or accepted an invalid extent");
                    }
                    HYBRID_REQUIRE(heap->writeStorageBuffer(queueHandle, rasterizer.queueBuffer()));
                    if (submitted) { HYBRID_REQUIRE(fence->reset()); HYBRID_REQUIRE(pool->reset()); }
                    HYBRID_REQUIRE(commands->begin());
                    const TextureBarrierDesc transitions[] = {
                        {
                            .texture = textures[0].get(),
                            .oldLayout = metallic::render::textureLayoutForResourceState(submitted ? ResourceState::TransferSource : ResourceState::Undefined),
                            .newLayout = TextureLayout::ColorAttachment,
                            .before = metallic::render::resourceSyncScope(submitted ? ResourceState::TransferSource : ResourceState::Undefined, metallic::render::PipelineStageBits::AllCommands),
                            .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
                        },
                        {
                            .texture = textures[1].get(),
                            .oldLayout = metallic::render::textureLayoutForResourceState(submitted ? ResourceState::TransferSource : ResourceState::Undefined),
                            .newLayout = TextureLayout::DepthStencilAttachment,
                            .before = metallic::render::resourceSyncScope(submitted ? ResourceState::TransferSource : ResourceState::Undefined, metallic::render::PipelineStageBits::AllCommands),
                            .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite},
                        }};
                    if (auto commandResult = commands->synchronize({.textures = {transitions, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                    const bool hybrid = configuration != 0;
                    if (hybrid) { if (auto commandResult = rasterizer.begin(*commands, configuration == 1 ? 1.f : configuration == 2 ? 8.f : 32.f, reversed); !commandResult) { return RHITestResult::fail(std::string("begin failed: ") + render::resultToString(commandResult)); } }
                    const RenderingAttachmentDesc color{.view = views[0].get(), .layout = TextureLayout::ColorAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
                    const RenderingAttachmentDesc depth{.view = views[1].get(), .layout = TextureLayout::DepthStencilAttachment,
                        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = reversed ? 0.f : 1.f};
                    if (auto commandResult = commands->beginRendering({
                        .renderArea = {.width = width, .height = height},
                        .colorAttachments = {&color, 1},
                        .depthStencilAttachment = &depth,
                    }); !commandResult) { return RHITestResult::fail(std::string("beginRendering failed: ") + render::resultToString(commandResult)); }
                    if (auto commandResult = commands->setViewport({.width = float(width), .height = float(height), .maxDepth = 1.f}); !commandResult) { return RHITestResult::fail(std::string("setViewport failed: ") + render::resultToString(commandResult)); }
                    commands->setScissor({.width = width, .height = height});
                    commands->bindBindlessHeap(*heap); if (auto commandResult = commands->bindExecution((pipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
                    const uint32_t push[] = {inputHandle.shaderIndex, hybrid ? queueHandle.shaderIndex : UINT32_MAX, doubleSided ? 1u : 0u};
                    commands->pushBindlessData(push, sizeof(push));
                    if (auto commandResult = commands->drawMeshTasks(uint32_t(vertices.size() / 3)); !commandResult) { return RHITestResult::fail(std::string("drawMeshTasks failed: ") + render::resultToString(commandResult)); } commands->endRendering();
                    if (hybrid) {
                        HYBRID_REQUIRE(rasterizer.resolve(*commands, *textures[0], *views[0], *textures[1], *views[1]));
                        const BufferBarrierDesc bufferTransitions[] = {
                            {
                                .buffer = &rasterizer.queueBuffer(),
                                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                            },
                            {
                                .buffer = &rasterizer.pixelBuffer(),
                                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                            }};
                        if (auto commandResult = commands->synchronize({.buffers = {bufferTransitions, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                        {
                            auto sourceSlice = (&rasterizer.queueBuffer())->slice({0, 32});
                            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                            auto destinationSlice = queueReadback.get()->slice({0, 32});
                            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                            if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
                        }
                        {
                            auto sourceSlice = (&rasterizer.pixelBuffer())->slice({0, pixelCount * 8});
                            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                            auto destinationSlice = pixelReadback.get()->slice({0, pixelCount * 8});
                            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                            if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
                        }
                    }
                    TextureBarrierDesc outputTransitions[2];
                    for (size_t i = 0; i < 2; ++i) {
                        outputTransitions[i] = {
                            .texture = textures[i].get(),
                            .oldLayout = transitions[i].newLayout,
                            .newLayout = TextureLayout::TransferSource,
                            .before = transitions[i].after,
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                        };
                    }
                    if (auto commandResult = commands->synchronize({.textures = {outputTransitions, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                    for (size_t i = 0; i < 2; ++i) {
                        if (auto commandResult = commands->copyTextureToBuffer({.texture = textures[i].get(), .buffer = readback[i].get(), .width = width, .height = height}); !commandResult) { return RHITestResult::fail(std::string("copyTextureToBuffer failed: ") + render::resultToString(commandResult)); }
                    }
                    HYBRID_REQUIRE(commands->end());
                    CommandBuffer* list[] = {commands.get()};
                    HYBRID_REQUIRE(queue->submit({.commandBuffers = {list, 1}, .signalFence = fence.get()}));
                    HYBRID_REQUIRE(fence->wait()); submitted = true;
                    std::vector<uint32_t> ids(pixelCount); std::vector<float> depths(pixelCount);
                    for (size_t i = 0; i < 2; ++i) {
                        readback[i]->invalidate(); const void* data = readback[i]->map();
                        if (!data) { return RHITestResult::fail("Output readback failed"); }
                        std::memcpy(i == 0 ? static_cast<void*>(ids.data()) : depths.data(), data, pixelCount * 4); readback[i]->unmap();
                    }
                    if (!hybrid) { referenceIds = ids; referenceDepth = depths; continue; }
                    queueReadback->invalidate(); const auto* header = static_cast<const uint32_t*>(queueReadback->map());
                    if (!header || header[0] == 0) { return RHITestResult::fail("GPU producer did not enqueue software triangles"); }
                    overflows += header[0] > header[1]; queueReadback->unmap();
                    pixelReadback->invalidate(); const auto* pixels = static_cast<const uint64_t*>(pixelReadback->map());
                    if (!pixels) { return RHITestResult::fail("Software pixel readback failed"); }
                    for (size_t i = 0; i < pixelCount; ++i) { softwarePixels += uint32_t(pixels[i]) != 0; }
                    pixelReadback->unmap();
                    for (size_t i = 0; i < pixelCount; ++i) {
                        if (ids[i] != referenceIds[i] || !std::isfinite(depths[i]) || std::abs(depths[i] - referenceDepth[i]) > 2e-6f) {
                            return RHITestResult::fail("HW/SW mismatch: reversed=" + std::to_string(reversed) + " doubleSided=" +
                                std::to_string(doubleSided) + " config=" + std::to_string(configuration) + " pixel=" + std::to_string(i) +
                                " id=" + std::to_string(ids[i]) + "/" + std::to_string(referenceIds[i]) +
                                " depth=" + std::to_string(depths[i]) + "/" + std::to_string(referenceDepth[i]));
                        }
                    }
                    ++cases;
                }
            }
        }
        if (softwarePixels < 1000 || overflows != 4) { return RHITestResult::fail("Software writes or bounded-queue overflow were not exercised"); }
        return RHITestResult::pass(std::to_string(cases) + " HW/SW comparisons; software pixels=" + std::to_string(softwarePixels) +
            "; clipping, shared edges, perspective depth, both windings/Z directions, thresholds and four forced overflows");
    }
};
METALLIC_REGISTER_RHI_TEST(HybridRasterDepthTest);
// Exercise the real stable compaction and argument generation on the GPU. IDs
// deliberately differ from candidate indices so a namespace remap cannot pass.
class HybridClusterBinTest final : public RHITest {
public:
    HybridClusterBinTest() { type = RHITestType::Rendering; name = "hybrid_cluster_stable_bins_and_indirect_limits"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Hybrid cluster bin regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless heap"); }
        HYBRID_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RHITestResult::skip("Requires hybrid raster capabilities");
        }
        std::unique_ptr<BindlessHeap> heap;
        HYBRID_REQUIRE(device->createBindlessHeap({.maxBuffers = 2}).transform([&](auto rhiValue) { heap = std::move(rhiValue); }));
        BindlessHandle inputHandle, binHandle;
        HYBRID_REQUIRE(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { inputHandle = std::move(rhiValue); })); HYBRID_REQUIRE(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { binHandle = std::move(rhiValue); }));
        ShaderCompileResult compiled;
        const auto compile = compileSlangShaderToSpirv({.moduleName = "HybridClusterProbe", .entryPointName = "classifyMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); });
        log = compiled.diagnostics; HYBRID_REQUIRE(compile);
        std::unique_ptr<ShaderModule> shader;
        std::unique_ptr<ComputePipeline> pipeline;
        HYBRID_REQUIRE(device->createShaderModule({.spirv = compiled.spirv}).transform([&](auto rhiValue) { shader = std::move(rhiValue); }));
        HYBRID_REQUIRE(device->createComputePipeline({
            .computeShader = {shader.get()},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = 12,
        }).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); }));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        HYBRID_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        HYBRID_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); })); HYBRID_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
        struct Case { uint32_t count; uint32_t mode; bool stream; };
        const Case cases[] = {{0, 0, false}, {513, 0, false}, {65537, 1, false},
            {65537, 2, true}, {65535u * 32u + 1u, 2, false}, {129, 3, true}};
        bool submitted = false;
        for (const auto test : cases) {
            const uint32_t capacity = std::max(1u, test.count);
            VisibilityHybridRasterizer rasterizer;
            HYBRID_REQUIRE(rasterizer.initialize(*device, 1, 1, log, 1, capacity));
            std::vector<std::array<uint32_t, 2>> candidates(capacity);
            std::array<std::vector<uint32_t>, 5> expected;
            for (uint32_t i = 0; i < test.count; ++i) {
                const uint32_t bin = test.mode == 1 ? 4u : test.mode == 2 ? 0u :
                    test.mode == 3 ? UINT32_MAX : i % 7u;
                const uint32_t id = 17u + i * 3u;
                candidates[i] = {id, bin};
                if (bin < expected.size()) { expected[bin].push_back(id); }
            }
            std::unique_ptr<Buffer> input, readback, arguments;
            HYBRID_REQUIRE(device->createBuffer({.size = capacity * 8ull, .structureStride = 8,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { input = std::move(rhiValue); }));
            void* mapped = input->map();
            if (!mapped) { return RHITestResult::fail("Cluster input map failed"); }
            std::memcpy(mapped, candidates.data(), candidates.size() * 8); input->flush(); input->unmap();
            HYBRID_REQUIRE(heap->writeStorageBuffer(inputHandle, *input));
            HYBRID_REQUIRE(heap->writeStorageBuffer(binHandle, rasterizer.clusterBuffer()));
            const uint64_t readbackBytes = (16ull + 5ull * capacity) * 4u;
            HYBRID_REQUIRE(device->createBuffer({.size = readbackBytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }));
            HYBRID_REQUIRE(device->createBuffer({.size = 60, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { arguments = std::move(rhiValue); }));
            if (submitted) { HYBRID_REQUIRE(fence->reset()); HYBRID_REQUIRE(pool->reset()); }
            HYBRID_REQUIRE(commands->begin());
            // Capacity validation must reject oversized dispatches before recording GPU work.
            if (!hasError(rasterizer.beginClusters(*commands, 8, true, 0, capacity + 1u, test.stream), Error::InvalidArgument)) {
                return RHITestResult::fail("Oversized cluster input was accepted");
            }
            HYBRID_REQUIRE(rasterizer.beginClusters(*commands, 8, true, 0, test.count, test.stream));
            commands->bindBindlessHeap(*heap); if (auto commandResult = commands->bindExecution((pipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
            const uint32_t push[] = {inputHandle.shaderIndex, binHandle.shaderIndex, test.count};
            commands->pushBindlessData(push, sizeof(push));
            if (test.count != 0) {
                commands->dispatch(std::min(test.count, 65535u), (test.count + 65534u) / 65535u);
            }
            HYBRID_REQUIRE(rasterizer.finishClusterBins(*commands));
            const BufferBarrierDesc barriers[] = {
                {
                    .buffer = &rasterizer.clusterBuffer(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                },
                {
                    .buffer = &rasterizer.clusterArguments(),
                    .before = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                }};
            if (auto commandResult = commands->synchronize({.buffers = {barriers, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            {
                auto sourceSlice = (&rasterizer.clusterBuffer())->slice({0, readbackBytes});
                if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                auto destinationSlice = readback.get()->slice({0, readbackBytes});
                if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
            }
            {
                auto sourceSlice = (&rasterizer.clusterArguments())->slice({0, 60});
                if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                auto destinationSlice = arguments.get()->slice({0, 60});
                if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
            }
            HYBRID_REQUIRE(commands->end());
            CommandBuffer* list[] = {commands.get()};
            HYBRID_REQUIRE(queue->submit({.commandBuffers = {list, 1}, .signalFence = fence.get()}));
            HYBRID_REQUIRE(fence->wait()); submitted = true;
            readback->invalidate(); arguments->invalidate();
            const auto* bins = static_cast<const uint32_t*>(readback->map());
            const auto* args = static_cast<const uint32_t*>(arguments->map());
            if (!bins || !args) { return RHITestResult::fail("Cluster result map failed"); }
            bool valid = bins[5] == capacity && bins[12] == test.count && bins[14] == 0u;
            for (size_t bin = 0; bin < 5; ++bin) {
                const uint32_t count = static_cast<uint32_t>(expected[bin].size());
                const uint32_t groups = bin == 4 ? count : test.stream ? count : (count + 31u) / 32u;
                valid = valid && bins[bin] == count && args[bin * 3u] == std::min(groups, 65535u) &&
                    args[bin * 3u + 1u] == std::max(1u, (groups + 65534u) / 65535u) && args[bin * 3u + 2u] == 1u;
                valid = valid && std::equal(expected[bin].begin(), expected[bin].end(), bins + 16u + bin * capacity);
            }
            readback->unmap(); arguments->unmap();
            if (!valid) { return RHITestResult::fail("Stable bins/indirect args mismatch for count=" + std::to_string(test.count)); }
        }
        return RHITestResult::pass("Stable IDs, five bins, empty/culled/partial blocks, capacity rejection; 2D resident HW, stream HW and SW dispatch limits");
    }
};
METALLIC_REGISTER_RHI_TEST(HybridClusterBinTest);
#undef HYBRID_REQUIRE

class HybridRasterSceneTest final : public RHITest {
public:
    HybridRasterSceneTest() { type = RHITestType::Rendering; name = "hybrid_raster_scene_equivalence"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, false);
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders"); }
        if (!result) { return RHITestResult::fail(preview.lastLog()); }
        render::RenderGraph graph;
        graph.addNode("VisibilityBufferPass", "VBuffer", {{"path", "Asset/StandfordBunny/scene.gltf"},
            {"visualization", "triangle"}, {"camera", {{"eye", {-.0168404f, .110154f, .22f}},
                {"center", {-.0168404f, .110154f, -.00153695f}}, {"znear", .001f}, {"zfar", 10.f}, {"orthoHeight", .24f}}}});
        graph.markOutput("VBuffer.visibility");
        graph.markOutput("VBuffer.depth");
        graph.markOutput("VBuffer.color");
        const auto node = graph.findNode("VBuffer")->id;
        size_t cases = 0, roundingTies = 0;
        for (bool orthographic : {false, true}) {
            for (bool reversed : {false, true}) {
                graph.setNodeRuntimeProperty(node, "camera.projection", orthographic ? "orthographic" : "perspective");
                graph.setNodeRuntimeProperty(node, "camera.reversedZ", reversed);
                graph.setNodeRuntimeProperty(node, "hybridRaster", false);
                if (!preview.render(graph, 193, 157)) { return RHITestResult::fail(preview.lastLog()); }
                const auto reference = preview.pixels();
                if (std::count_if(reference.begin(), reference.end(), [](uint32_t id) { return id != 0; }) < 1000) {
                    return RHITestResult::fail("Bunny reference did not produce meaningful visibility coverage");
                }
                if (!preview.render(graph, 193, 157, "VBuffer.depth")) { return RHITestResult::fail(preview.lastLog()); }
                const auto referenceDepth = preview.pixels();
                for (uint32_t configuration = 0; configuration < 9; ++configuration) {
                    const float threshold = std::array{1.f, 8.f, 32.f}[configuration % 3];
                    graph.setNodeRuntimeProperty(node, "clusterPrebin", configuration >= 3);
                    graph.setNodeRuntimeProperty(node, "asyncSoftwareRaster", configuration >= 6);
                    graph.setNodeRuntimeProperty(node, "hybridRaster", true);
                    graph.setNodeRuntimeProperty(node, "softwareRasterMaxPixels", threshold);
                    for (uint32_t frame = 0; frame < 3; ++frame) {
                        if (!preview.render(graph, 193, 157)) { return RHITestResult::fail(preview.lastLog()); }
                        const bool independent = preview.subsystemHost()->device()->capabilities().independentComputeQueue;
                        const uint32_t branches = preview.executionStats().asyncComputeBranches;
                        if ((configuration >= 6 && independent) ? branches < 2u : branches != 0u) {
                            return RHITestResult::fail("Async setting did not select the expected hardware/software queue topology");
                        }
                        const auto actual = preview.pixels();
                        if (actual != reference) {
                            if (!preview.render(graph, 193, 157, "VBuffer.depth")) { return RHITestResult::fail(preview.lastLog()); }
                            for (size_t pixel = 0; pixel < reference.size(); ++pixel) {
                                if (reference[pixel] == actual[pixel]) { continue; }
                                const uint32_t depth = preview.pixels()[pixel];
                                const uint32_t delta = std::max(depth, referenceDepth[pixel]) - std::min(depth, referenceDepth[pixel]);
                                // Reclustering exposes nearly coincident triangles. Fixed-function
                                // interpolation and compute arithmetic can disagree by a few ULPs.
                                // Keep coverage and cluster identity exact; only qualify depth ties.
                                if (reference[pixel] == 0 || actual[pixel] == 0 ||
                                    (reference[pixel] >> render::kVisibilityTriangleBits) != (actual[pixel] >> render::kVisibilityTriangleBits) || delta > 8u) {
                                    return RHITestResult::fail("HW/SW visibility mismatch at pixel " + std::to_string(pixel) +
                                        "; ortho=" + std::to_string(orthographic) + ", reversed=" + std::to_string(reversed) +
                                        ", configuration=" + std::to_string(configuration) + ", frame=" + std::to_string(frame) +
                                        ", depth ULP=" + std::to_string(delta));
                                }
                                ++roundingTies;
                            }
                        }
                        ++cases;
                    }
                }
            }
        }
        std::string log;
        if (!preview.render(graph, 193, 157, "VBuffer.color")) { return RHITestResult::fail(preview.lastLog()); }
        if (!saveRgba8Png(context.outputDirectory / "HybridBunny.png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
            preview.width(), preview.height(), log)) { return RHITestResult::fail(log); }
        return RHITestResult::pass(std::to_string(cases) + " HZB frames: exact coverage/cluster IDs across queue topologies, projections and thresholds; " + std::to_string(roundingTies) + " near-coincident triangle depth ties within 8 ULP");
    }
};
METALLIC_REGISTER_RHI_TEST(HybridRasterSceneTest);
class VisibilityPreparationTest final : public RHITest {
public:
    VisibilityPreparationTest() { type = RHITestType::Rendering; name = "visibility_preparation_serial_parallel_pixels"; }
    RHITestResult run(RHITestContext& context) override
    {
        auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 2) { return RHITestResult::skip("two preparation workers required"); }
        render::RenderGraphPreviewRenderer preview;
        auto initialized = preview.initialize(context.enableValidation, false, false);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("mesh shaders required"); }
        if (!initialized) { return RHITestResult::fail(preview.lastLog()); }
        render::RenderGraph graph;
        graph.addNode("VisibilityBufferPass", "VBuffer", {{"path", "Asset/StandfordBunny/scene.gltf"},
            {"visualization", "triangle"}, {"camera", {{"eye", {-.0168404f, .110154f, .22f}},
                {"center", {-.0168404f, .110154f, -.00153695f}}, {"znear", .001f}, {"zfar", 10.f}, {"orthoHeight", .24f}}}});
        graph.addNode("VisibilityBufferMaterialPass", "Material", {{"visualization", "normal"}});
        graph.addEdge("VBuffer.visibility", "Material.visibility");
        graph.addEdge("VBuffer.rasterInfo", "Material.rasterInfo");
        graph.markOutput("Material.color");
        const auto node = graph.findNode("VBuffer")->id;
        double serialCpu = 0, parallelCpu = 0;
        for (uint32_t mode = 0; mode < 16; ++mode) {
            graph.setNodeRuntimeProperty(node, "camera.projection", (mode & 1) ? "orthographic" : "perspective");
            graph.setNodeRuntimeProperty(node, "camera.reversedZ", bool(mode & 2));
            graph.setNodeRuntimeProperty(node, "freezeCullingCamera", bool(mode & 4));
            graph.setNodeRuntimeProperty(node, "hzbSpd", bool(mode & 8));
            std::vector<uint32_t> reference;
            uint32_t branches = 0;
            for (const uint32_t workers : {1u, 4u}) {
                preview.setRecordingWorkerLimit(workers);
                for (uint32_t frame = 0; frame < 3; ++frame) {
                    if (!preview.render(graph, 193, 157)) { return RHITestResult::fail(preview.lastLog()); }
                    const auto& stats = preview.executionStats();
                    if (stats.preparationTaskCount != (workers == 1 ? 0u : 2u)) {
                        return RHITestResult::fail("VisibilityBuffer did not use the requested CPU preparation policy");
                    }
                    if (frame == 2) {
                        if (workers == 1) { reference = preview.pixels(); branches = stats.asyncComputeBranches; serialCpu += stats.cpuMilliseconds; }
                        else {
                            parallelCpu += stats.cpuMilliseconds;
                            if (preview.pixels() != reference || stats.asyncComputeBranches != branches) {
                                return RHITestResult::fail("Serial/parallel VisibilityBuffer camera/material output or GPU topology differs, mode=" + std::to_string(mode));
                            }
                        }
                    }
                }
            }
            if (std::count_if(reference.begin(), reference.end(), [](uint32_t pixel) { return (pixel & 0xffffffu) != 0; }) < 1000) {
                return RHITestResult::fail("Visibility material output was empty");
            }
        }
        std::string log;
        if (!saveRgba8Png(context.outputDirectory / "VisibilityPreparedMaterial.png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
            preview.width(), preview.height(), log)) { return RHITestResult::fail(log); }
        return RHITestResult::pass("16 configurations / 96 frames, exact material pixels and GPU branch topology; validation-build sampled CPU totals (ms), serial=" +
            std::to_string(serialCpu) + ", parallel=" + std::to_string(parallelCpu));
    }
};
METALLIC_REGISTER_RHI_TEST(VisibilityPreparationTest);

} // namespace
} // namespace metallic::tests
