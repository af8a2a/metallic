#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/StreamBLASParameters.h"
#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"

#include <array>
#include <algorithm>
#include <vector>
#include <bit>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

class StreamBLASAllocationTest final : public RHITest {
public:
    StreamBLASAllocationTest() { type = RHITestType::Rendering; name = "stream_blas_selected_allocation"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto require = [](bool success, const char* message) {
            if (!success) { throw std::runtime_error(message); }
        };
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Selected BLAS allocation",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Bindless unavailable"); }
        if (!created) { return RHITestResult::fail("Device creation failed"); }
        try {
            constexpr uint32_t count = 137, groups = count, references = count * 32;
            enum Input { Active, Groups, Params, Header, Instances, Infos, References, Pages, Addresses, Destinations, InputCount };
            const uint64_t sizes[] = {sizeof(MeshletStreamGPUActiveHeader), groups * sizeof(MeshletStreamGPUActiveGroup),
                sizeof(MeshletStreamGPUParams), sizeof(MeshletStreamGPUBLASHeader) + (groups + (count + 63) / 64) * 16,
                count * sizeof(MeshletStreamGPUInstanceBLAS), count * sizeof(MeshletStreamGPUBLASBuildInfo),
                references * 8 + 16, count * sizeof(MeshletStreamCLASPageEntry), references * 8, count * 8};
            const uint32_t strides[] = {32, 112, sizeof(MeshletStreamGPUParams), sizeof(MeshletStreamGPUBLASHeader), sizeof(MeshletStreamGPUInstanceBLAS), 16, 8, sizeof(MeshletStreamCLASPageEntry), 8, 8};
            std::array<std::unique_ptr<Buffer>, InputCount> buffers;
            for (uint32_t i = 0; i < InputCount; ++i) {
                require(bool(device->createBuffer({.size = sizes[i], .structureStride = strides[i],
                    .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                    .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto v) { buffers[i] = std::move(v); })), "buffer");
            }
            ShaderCompileResult compiled;
            auto result = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
                .entryPointName = kMeshletStreamBLASInputEntryPoint, .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled.diagnostics)
                .transform([&](auto v) { compiled = std::move(v); });
            if (!result) { return RHITestResult::fail(compiled.diagnostics); }
            ComputeKernel kernel;
            require(bool(kernel.initialize(*device, {.spirv = compiled.spirv,
                .parameters = parameterAbi<StreamBLASParameters>(kStreamBLASABI, ParameterTransport::InlinePush)}, compiled.diagnostics)), "kernel");
            auto* queue = device->getQueue(QueueType::Graphics);
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> command;
            std::unique_ptr<Fence> fence;
            std::unique_ptr<Buffer> readback;
            const uint64_t totalBytes = std::accumulate(std::begin(sizes), std::end(sizes), uint64_t(0));
            require(bool(device->createCommandPool(*queue).transform([&](auto v) { pool = std::move(v); })) &&
                bool(pool->createCommandBuffer().transform([&](auto v) { command = std::move(v); })) &&
                bool(device->createFence(false).transform([&](auto v) { fence = std::move(v); })) &&
                bool(device->createBuffer({.size = totalBytes, .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto v) { readback = std::move(v); })), "commands");
            const auto upload = [&](uint32_t i, const void* data, size_t bytes) {
                void* mapped = buffers[i]->map(); require(mapped != nullptr, "map");
                std::memset(mapped, 0, size_t(sizes[i]));
                if (data) { std::memcpy(mapped, data, bytes); }
                buffers[i]->flush(); buffers[i]->unmap();
            };
            auto registry = device->resourceRegistry();
            require(bool(registry), "registry");
            ParameterWriter writer(*device, **registry);
            StreamBLASParameters push{
                .settings = writer.dataBuffer(buffers[Params].get(), sizeof(MeshletStreamGPUParams), 16),
                .activeGroupBuffer = writer.buffer(buffers[Groups].get()),
                .activeHeaderBuffer = writer.buffer(buffers[Active].get()),
                .blasBuildInfoBuffer = writer.buffer(buffers[Infos].get()),
                .blasClusterReferenceBuffer = writer.buffer(buffers[References].get()),
                .blasHeaderBuffer = writer.buffer(buffers[Header].get()),
                .clasAddressBuffer = writer.buffer(buffers[Addresses].get()),
                .clasPageTableBuffer = writer.buffer(buffers[Pages].get()),
                .dynamicBlasAddressBuffer = writer.buffer(buffers[Destinations].get()),
                .instanceBlasBuffer = writer.buffer(buffers[Instances].get()),
                .scratch = writer.buffer(buffers[Header].get()),
                .traversalPhase = 1,
            };
            bool submitted = false;
            const auto execute = [&](bool force) {
                push.traversalPhase = force ? 1u : 0u;
                if (submitted) { require(bool(fence->reset()) && bool(pool->reset()), "reset"); }
                require(bool(command->begin()), "begin"); command->hostWriteBarrier();
                std::vector<EncodedParameters> packets;
                std::array<BufferBarrierDesc, InputCount> barriers{};
                for (uint32_t i = 0; i < InputCount; ++i) {
                    barriers[i] = {.buffer = buffers[i].get(),
                        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
                }
                for (uint32_t phase : {0u, 5u, 1u, 4u, 6u, 7u, 2u, 8u, 9u, 10u, 3u, 11u}) {
                    push.activeBuildPhase = phase;
                    auto encoded = writer.encode(push, kStreamBLASABI, ParameterTransport::InlinePush);
                    require(bool(encoded), "parameters");
                    packets.push_back(std::move(*encoded));
                    require(bool(kernel.dispatch(*command, packets.back(),
                        phase == 0 || phase == 7 || phase == 9 ? 1 : (count + 63) / 64)), "dispatch");
                    require(bool(command->synchronize({.buffers = barriers})), "phase barrier");
                }
                for (auto& barrier : barriers) { barrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead}; }
                require(bool(command->synchronize({.buffers = barriers})), "readback barrier");
                uint64_t offset = 0;
                for (uint32_t i = 0; i < InputCount; ++i) {
                    auto src = buffers[i]->slice({0, sizes[i]}); auto dst = readback->slice({offset, sizes[i]});
                    require(bool(src) && bool(dst) && bool(command->copyBuffer(*src, *dst)), "copy"); offset += sizes[i];
                }
                require(bool(command->end()), "end"); CommandBuffer* commands[] = {command.get()};
                require(bool(queue->submit({.commandBuffers = commands, .signalFence = fence.get()})) && bool(fence->wait()), "submit");
                readback->invalidate(); const auto* data = static_cast<const uint8_t*>(readback->map());
                require(data != nullptr, "readback");
                std::vector<uint8_t> copied(data, data + totalBytes);
                readback->unmap(); submitted = true;
                return copied;
            };
            // Real entrypoint, multiple scan blocks, a partial last block, zero
            // initial per-instance capacity, sparse masks and reversed ordering.
            for (uint32_t test = 0; test < 6; ++test) {
                for (uint32_t i = 0; i < InputCount; ++i) { upload(i, nullptr, 0); }
                MeshletStreamGPUParams params;
                params.activeGroupCount = groups; params.sceneInstanceCount = count; params.scenePageCount = count;
                params.blasClusterReferenceCapacity = test == 1 ? 93 : references;
                params.blasBuildCapacity = test == 2 ? 3 : count;
                params.maxBlasClustersPerBuild = test == 3 ? 3 : 32;
                params.blasStorageBytes = 1u << 20;
                params.blasStorageAddressLow = 0xfff00000u;
                params.blasStorageAddressHigh = 7;
                for (uint32_t bucket = 0; bucket < 32; ++bucket) { params.blasSizeClasses[bucket] = 64u << std::min(bucket, 12u); }
                params.blasClusterReferenceAddressLow = 0xfffffff0u;
                params.blasClusterReferenceAddressHigh = 5;
                std::array<MeshletStreamGPUActiveGroup, groups> active{};
                std::array<MeshletStreamCLASPageEntry, count> pages{};
                std::array<uint64_t, references> addresses{};
                std::array<uint32_t, count> selected{};
                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t instance = test == 4 ? count - 1 - i : i;
                    auto& group = active[i]; group.instanceIndex = instance; group.pageIndex = i;
                    group.clusterCount = 32; group.clusterSelectionMask = i % 7 == 0 ? 0 : (0x55555555u >> (i % 8 * 2));
                    selected[instance] = std::popcount(group.clusterSelectionMask);
                    pages[i].addressOffsetAndState = packMeshletStreamClasPageEntry(i * 32,
                        i == 81 ? MeshletStreamCLASPageState::Retiring : MeshletStreamCLASPageState::Active);
                    for (uint32_t c = 0; c < 32; ++c) { addresses[i * 32 + c] = 0x1234567800000000ull + i * 256 + c * 8; }
                }
                if (test == 5) {
                    // Exact fit across all three blocks; missing page is excluded.
                    params.blasClusterReferenceCapacity = std::accumulate(selected.begin(), selected.end(), 0u) - selected[81];
                }
                MeshletStreamGPUActiveHeader activeHeader{.activeGroupCount = groups, .activeGroupCapacity = groups};
                upload(Active, &activeHeader, sizeof(activeHeader)); upload(Groups, active.data(), sizeof(active));
                upload(Params, &params, sizeof(params)); upload(Pages, pages.data(), sizeof(pages));
                upload(Addresses, addresses.data(), sizeof(addresses));
                auto copied = execute(true);
                const auto* data = copied.data();
                uint64_t offset = 0;
                std::array<const uint8_t*, InputCount> output{}; offset = 0;
                for (uint32_t i = 0; i < InputCount; ++i) { output[i] = data + offset; offset += sizes[i]; }
                MeshletStreamGPUBLASHeader header; std::memcpy(&header, output[Header], sizeof(header));
                uint32_t neededRefs = 0, neededBuilds = 0, acceptedRefs = 0, acceptedBuilds = 0;
                uint32_t refRejected = 0, buildRejected = 0, oversized = 0;
                std::vector<std::pair<uint32_t, uint32_t>> ranges;
                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t page = test == 4 ? count - 1 - i : i;
                    const bool eligible = page != 81 && selected[i] && selected[i] <= params.maxBlasClustersPerBuild;
                    const bool fitRefs = neededRefs + selected[i] <= params.blasClusterReferenceCapacity;
                    const bool accept = eligible && fitRefs && neededBuilds < params.blasBuildCapacity;
                    MeshletStreamGPUInstanceBLAS actual; std::memcpy(&actual, output[Instances] + i * sizeof(actual), sizeof(actual));
                    require(((actual.flags & kMeshletStreamBLASInstanceDynamic) != 0) == accept, "admission mismatch");
                    if (accept) {
                        require(actual.storageOffset <= params.blasStorageBytes && actual.storageCapacity <= params.blasStorageBytes - actual.storageOffset,
                            "persistent storage out of bounds");
                        ranges.emplace_back(actual.storageOffset, actual.storageOffset + actual.storageCapacity);
                        require(actual.clusterReferenceOffset == acceptedRefs && actual.clusterReferenceCapacity == selected[i] &&
                            actual.insertedClusterCount == selected[i] && actual.blasBuildIndex == acceptedBuilds, "overlap/hole/partial build");
                        MeshletStreamGPUBLASBuildInfo info; std::memcpy(&info, output[Infos] + actual.blasBuildIndex * sizeof(info), sizeof(info));
                        const uint64_t address = (uint64_t(info.clusterReferencesAddressHigh) << 32) | info.clusterReferencesAddressLow;
                        require(address == 0x5fffffff0ull + acceptedRefs * 8 && info.clusterReferencesCount == selected[i], "build address/count");
                        uint32_t j = 0;
                        for (uint32_t c = 0; c < 32; ++c) {
                            if (!(active[page].clusterSelectionMask & (1u << c))) { continue; }
                            uint64_t ref; std::memcpy(&ref, output[References] + (acceptedRefs + j++) * 8, 8);
                            require(ref == addresses[page * 32 + c], "reference content");
                        }
                        acceptedRefs += selected[i]; ++acceptedBuilds;
                    } else {
                        require((actual.flags & kMeshletStreamBLASInstanceFallback) != 0, "missing complete fallback");
                        if (eligible) { if (!fitRefs) { ++refRejected; } else { ++buildRejected; } }
                    }
                    oversized += page != 81 && selected[i] > params.maxBlasClustersPerBuild;
                    if (eligible) { neededRefs += selected[i]; ++neededBuilds; }
                }
                std::sort(ranges.begin(), ranges.end());
                for (size_t i = 1; i < ranges.size(); ++i) { require(ranges[i - 1].second <= ranges[i].first, "persistent storage overlaps"); }
                require(header.liveClusterReferences == acceptedRefs, "live reference count differs from selected cut");
                require(header.blasBuildCount == acceptedBuilds && header.clusterReferenceCount == acceptedRefs &&
                    header.referenceBudgetRejected == refRejected && header.buildBudgetRejected == buildRejected &&
                    header.oversizedInstances == oversized && header.missingClasInstances == 1 &&
                    header.overflowCount == refRejected + buildRejected + oversized, "header counters");
                for (uint32_t i = acceptedRefs * 8; i < sizes[References]; ++i) {
                    require(output[References][i] == 0, "reference write outside admitted range");
                }
                if (test == 0) {
                    const auto headerAt = [&](const auto& bytes) {
                        MeshletStreamGPUBLASHeader value;
                        std::memcpy(&value, bytes.data() + sizes[Active] + sizes[Groups] + sizes[Params], sizeof(value));
                        return value;
                    };
                    const auto instanceAt = [&](const auto& bytes, uint32_t id) {
                        MeshletStreamGPUInstanceBLAS value;
                        std::memcpy(&value, bytes.data() + sizes[Active] + sizes[Groups] + sizes[Params] + sizes[Header] + id * sizeof(value), sizeof(value));
                        return value;
                    };
                    auto reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0 && headerAt(reuse).reusedInstances == acceptedBuilds, "unchanged instances rebuilt");
                    // Unrelated publication and transform changes never alter object-space BLAS.
                    ++push.clasPublicationRevision; active[100].world0[3] = 123.0f;
                    upload(Groups, active.data(), sizeof(active));
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0, "unrelated publication/transform invalidated cache");
                    ++pages[100].publicationGeneration;
                    upload(Pages, pages.data(), sizeof(pages)); reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 1 && headerAt(reuse).reusedInstances + 1 == acceptedBuilds &&
                        headerAt(reuse).publicationInvalidated == 1, "page generation did not isolate dirty instance");
                    for (uint32_t i = 0; i < count; ++i) {
                        require(instanceAt(copied, i).storageOffset == instanceAt(reuse, i).storageOffset, "stable slot relocated");
                    }
                    // Remove an empty instance's group: every following cut offset shifts.
                    std::move(active.begin() + 1, active.end(), active.begin());
                    --activeHeader.activeGroupCount;
                    upload(Active, &activeHeader, sizeof(activeHeader)); upload(Groups, active.data(), sizeof(active));
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0 && headerAt(reuse).reusedInstances == acceptedBuilds, "compacted cut offsets invalidated unrelated instances");
                    active[99].clusterSelectionMask = 0xffffffffu;
                    upload(Groups, active.data(), sizeof(active)); reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 1 && headerAt(reuse).arenaRepack == 0, "local cut growth rebuilt unrelated instances");
                    uint32_t packedBytes = 0;
                    for (uint32_t i = 0; i < count; ++i) {
                        auto instance = instanceAt(reuse, i);
                        if (instance.flags & kMeshletStreamBLASInstanceDynamic) {
                            uint32_t bucket = instance.selectedClusterCount <= 1 ? 0 : std::bit_width(instance.selectedClusterCount - 1);
                            packedBytes += params.blasSizeClasses[bucket];
                        }
                    }
                    params.blasStorageBytes = packedBytes;
                    upload(Params, &params, sizeof(params)); reuse = execute(false);
                    require(headerAt(reuse).arenaRepack == 1 && headerAt(reuse).blasBuildCount == acceptedBuilds &&
                        headerAt(reuse).storageRejected == 0, "bounded arena repack failed");
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0, "repacked arena did not settle");
                    params.blasStorageBytes = 64;
                    upload(Params, &params, sizeof(params)); reuse = execute(false);
                    require(headerAt(reuse).storageRejected == acceptedBuilds && headerAt(reuse).blasBuildCount == 0, "storage overflow published invalid BLAS");
                    params.blasStorageBytes = 1u << 20;
                    upload(Params, &params, sizeof(params)); reuse = execute(true);
                    require(headerAt(reuse).blasBuildCount == acceptedBuilds && headerAt(reuse).storageRejected == 0, "cancel/reset did not restore cache");
                    pages[81].addressOffsetAndState = packMeshletStreamClasPageEntry(81 * 32, MeshletStreamCLASPageState::Active);
                    ++pages[81].publicationGeneration;
                    upload(Pages, pages.data(), sizeof(pages)); reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 1 && headerAt(reuse).reusedInstances == acceptedBuilds, "republication invalidated unrelated instances");
                }
            }
            return RHITestResult::pass("Per-instance exact reuse, local generation/cut invalidation, shifted ranges, bounded growth/repack, reset; selected-cut packing: zero initial slices, multi-block/tail scan, reordered instances, sparse masks, exact fit, reference/build/per-build limits and missing CLAS");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamBLASAllocationTest);
} // namespace
} // namespace metallic::tests
