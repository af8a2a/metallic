#include "Runtime/Render/Streamer/UploadStreamer.h"
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
            enum Input { Active, Groups, Params, Header, Instances, Infos, References, Pages, Addresses, Destinations,
                SceneInstances, FallbackAddresses, InputCount };
            const uint64_t sizes[] = {sizeof(MeshletStreamGPUActiveHeader), groups * sizeof(MeshletStreamGPUActiveGroup),
                sizeof(MeshletStreamGPUParams), sizeof(MeshletStreamGPUBLASHeader) + (groups + (count + 63) / 64) * 16,
                count * sizeof(MeshletStreamGPUInstanceBLAS), count * sizeof(MeshletStreamGPUBLASBuildInfo),
                references * 8 + 16, count * sizeof(MeshletStreamCLASPageEntry), references * 8, count * 8,
                count * sizeof(MeshletStreamGPUInstance), sizeof(uint64_t)};
            const uint32_t strides[] = {32, 112, sizeof(MeshletStreamGPUParams), sizeof(MeshletStreamGPUBLASHeader), sizeof(MeshletStreamGPUInstanceBLAS), 16, 8, sizeof(MeshletStreamCLASPageEntry), 8, 8,
                sizeof(MeshletStreamGPUInstance), sizeof(uint64_t)};
            std::unique_ptr<BindlessHeap> heap;
            require(bool(device->createBindlessHeap({.maxBuffers = InputCount}).transform([&](auto v) { heap = std::move(v); })), "heap");
            std::array<BindlessHandle, InputCount> handles;
            std::array<std::unique_ptr<Buffer>, InputCount> buffers;
            for (uint32_t i = 0; i < InputCount; ++i) {
                require(bool(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto v) { handles[i] = std::move(v); })), "handle");
                require(bool(device->createBuffer({.size = sizes[i], .structureStride = strides[i],
                    .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                    .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto v) { buffers[i] = std::move(v); })), "buffer");
                require(bool((*buffers[i]).slice().and_then([&](const auto& bufferSlice) { return heap->writeStorageBuffer(handles[i], bufferSlice); })), "binding");
            }
            ShaderCompileResult compiled;
            auto result = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
                .entryPointName = kMeshletStreamBLASInputEntryPoint, .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled.diagnostics)
                .transform([&](auto v) { compiled = std::move(v); });
            if (!result) { return RHITestResult::fail(compiled.diagnostics); }
            std::unique_ptr<ShaderModule> shader;
            std::unique_ptr<ComputePipeline> pipeline;
            require(bool(device->createShaderModule({.spirv = compiled.spirv}).transform([&](auto v) { shader = std::move(v); })), "shader");
            require(bool(device->createComputePipeline({.computeShader = {shader.get()}, .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush)}).transform([&](auto v) { pipeline = std::move(v); })), "pipeline");
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
            MeshletStreamUserPush push{.activeGroupBuffer = handles[Groups].shaderIndex,
                .paramsBuffer = handles[Params].shaderIndex, .activeHeaderBuffer = handles[Active].shaderIndex,
                .instanceBuffer = handles[SceneInstances].shaderIndex,
                .clasAddressBuffer = handles[Addresses].shaderIndex, .clasPageTableBuffer = handles[Pages].shaderIndex,
                .blasHeaderBuffer = handles[Header].shaderIndex, .instanceBlasBuffer = handles[Instances].shaderIndex,
                .blasBuildInfoBuffer = handles[Infos].shaderIndex, .blasClusterReferenceBuffer = handles[References].shaderIndex,
                .fallbackBlasAddressBuffer = handles[FallbackAddresses].shaderIndex,
                .dynamicBlasAddressBuffer = handles[Destinations].shaderIndex,
                .traversalPhase = 1};
            bool submitted = false;
            const auto execute = [&](uint32_t cacheFlags) {
                // Build records are frame outputs. Clear them so a missing write
                // cannot be concealed by a valid record from an earlier frame.
                upload(Infos, nullptr, 0);
                upload(Destinations, nullptr, 0);
                push.traversalPhase = cacheFlags;
                if (submitted) { require(bool(fence->reset()) && bool(pool->reset()), "reset"); }
                require(bool(command->begin()), "begin"); command->hostWriteBarrier(); require(bool(command->bindBindlessHeap(*heap)), "record command");
                require(bool(command->bindExecution(pipeline->execution())), "bind");
                std::array<BufferBarrierDesc, InputCount> barriers{};
                for (uint32_t i = 0; i < InputCount; ++i) {
                    barriers[i] = {.buffer = buffers[i].get(),
                        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
                }
                for (uint32_t phase : {0u, 5u, 1u, 4u, 6u, 7u, 2u, 8u, 9u, 12u, 13u, 10u, 3u, 11u}) {
                    push.activeBuildPhase = phase; require(bool(command->pushBindlessData(&push, sizeof(push))), "record command");
                    require(bool(command->dispatch(phase == 0 || phase == 7 || phase == 9 || phase == 13 ? 1 : (count + 63) / 64)), "record command");
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
                params.scenePrimitiveCount = 1;
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
                std::array<MeshletStreamGPUInstance, count> sceneInstances{};
                for (auto& instance : sceneInstances) { instance.visible = 1; }
                uint64_t fallbackAddress = 0x24680000ull;
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
                upload(SceneInstances, sceneInstances.data(), sizeof(sceneInstances));
                upload(FallbackAddresses, &fallbackAddress, sizeof(fallbackAddress));
                auto copied = execute(1u);
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
                    require(headerAt(reuse).semanticChangeCount == 0, "Unchanged published cut invalidated shading history");
                    // Unrelated publication and transform changes never alter object-space BLAS.
                    ++push.clasPublicationRevision; active[100].world0[3] = 123.0f;
                    upload(Groups, active.data(), sizeof(active));
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0, "unrelated publication/transform invalidated cache");
                    require(headerAt(reuse).semanticChangeCount == 0, "Unrelated publication invalidated shading history");
                    ++pages[100].publicationGeneration;
                    for (uint32_t c = 0; c < 32; ++c) { addresses[100 * 32 + c] += 0x100000ull; }
                    upload(Addresses, addresses.data(), sizeof(addresses));
                    upload(Pages, pages.data(), sizeof(pages)); reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 1 && headerAt(reuse).reusedInstances + 1 == acceptedBuilds &&
                        headerAt(reuse).publicationInvalidated == 1, "page generation did not isolate dirty instance");
                    require(headerAt(reuse).semanticChangeCount == 0, "CLAS relocation/generation invalidated identical geometry");
                    for (uint32_t i = 0; i < count; ++i) {
                        require(instanceAt(copied, i).storageOffset == instanceAt(reuse, i).storageOffset, "stable slot relocated");
                    }
                    // Remove an empty instance's group: every following cut offset shifts.
                    std::move(active.begin() + 1, active.end(), active.begin());
                    --activeHeader.activeGroupCount;
                    upload(Active, &activeHeader, sizeof(activeHeader)); upload(Groups, active.data(), sizeof(active));
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0 && headerAt(reuse).reusedInstances == acceptedBuilds, "compacted cut offsets invalidated unrelated instances");
                    require(headerAt(reuse).semanticChangeCount == 0, "Cut array relocation invalidated identical geometry");
                    active[99].clusterSelectionMask = 0xffffffffu;
                    upload(Groups, active.data(), sizeof(active)); reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 1 && headerAt(reuse).arenaRepack == 0, "local cut growth rebuilt unrelated instances");
                    require(headerAt(reuse).semanticChangeCount == 1, "Real cut change did not invalidate exactly one instance in its frame");
                    reuse = execute(false);
                    require(headerAt(reuse).semanticChangeCount == 0, "Real cut change invalidated another stable frame");
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
                    require(headerAt(reuse).semanticChangeCount == 0, "BLAS arena relocation invalidated identical geometry");
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0, "repacked arena did not settle");
                    // Partial overflow must settle just like a fully admitted cut:
                    // rejected instances cannot force the surviving BLAS to repack
                    // and rebuild on every unchanged frame.
                    params.blasStorageBytes = packedBytes / 2u;
                    upload(Params, &params, sizeof(params));
                    auto partial = execute(false);
                    const auto partialHeader = headerAt(partial);
                    require(partialHeader.storageRejected > 0 && partialHeader.storageRejected < acceptedBuilds,
                        "partial storage overflow did not retain a dynamic subset");
                    require(partialHeader.semanticChangeCount == partialHeader.storageRejected,
                        "partial overflow invalidated unchanged dynamic geometry or omitted fallback changes");
                    const uint32_t partialBuilds = acceptedBuilds - partialHeader.storageRejected;
                    const auto checkStorage = [&](const auto& bytes) {
                        const auto currentHeader = headerAt(bytes);
                        std::vector<std::pair<uint32_t, uint32_t>> slots;
                        uint32_t liveReferences = 0;
                        for (uint32_t i = 0; i < count; ++i) {
                            const auto instance = instanceAt(bytes, i);
                            if ((instance.flags & kMeshletStreamBLASInstanceDynamic) == 0u) { continue; }
                            require(instance.storageCapacity != 0 && instance.storageOffset <= params.blasStorageBytes &&
                                instance.storageCapacity <= params.blasStorageBytes - instance.storageOffset,
                                "partial overflow published an out-of-bounds BLAS");
                            slots.emplace_back(instance.storageOffset, instance.storageOffset + instance.storageCapacity);
                            liveReferences += instance.selectedClusterCount;
                        }
                        std::sort(slots.begin(), slots.end());
                        for (size_t i = 1; i < slots.size(); ++i) {
                            require(slots[i - 1].second <= slots[i].first, "partial overflow BLAS storage overlaps");
                        }
                        require(liveReferences == currentHeader.liveClusterReferences &&
                            slots.size() == currentHeader.reusedInstances + currentHeader.blasBuildCount,
                            "partial overflow admission counters do not match live instances");
                        require(currentHeader.blasBuildCount <= count, "partial overflow build count exceeds capacity");
                        const uint64_t infoOffset = std::accumulate(std::begin(sizes), std::begin(sizes) + Infos, uint64_t(0));
                        const uint64_t destinationOffset = std::accumulate(std::begin(sizes), std::begin(sizes) + Destinations, uint64_t(0));
                        std::vector<uint64_t> builtAddresses;
                        uint32_t buildReferences = 0;
                        for (uint32_t build = 0; build < currentHeader.blasBuildCount; ++build) {
                            MeshletStreamGPUBLASBuildInfo info;
                            std::memcpy(&info, bytes.data() + infoOffset + build * sizeof(info), sizeof(info));
                            uint64_t destination;
                            std::memcpy(&destination, bytes.data() + destinationOffset + build * sizeof(destination), sizeof(destination));
                            require(info.clusterReferencesCount != 0 && destination != 0,
                                "partial overflow left a hole in the indirect build records");
                            bool matched = false;
                            for (uint32_t i = 0; i < count; ++i) {
                                const auto instance = instanceAt(bytes, i);
                                if ((instance.flags & kMeshletStreamBLASInstanceDynamic) == 0u) { continue; }
                                const uint64_t storageAddress = (uint64_t(params.blasStorageAddressHigh) << 32) | params.blasStorageAddressLow;
                                if (destination != storageAddress + instance.storageOffset) { continue; }
                                const uint64_t referencesAddress = (uint64_t(info.clusterReferencesAddressHigh) << 32) | info.clusterReferencesAddressLow;
                                require(instance.blasBuildIndex == build && info.clusterReferencesCount == instance.selectedClusterCount &&
                                    referencesAddress == 0x5fffffff0ull + uint64_t(instance.clusterReferenceOffset) * 8 &&
                                    instance.clusterReferenceOffset <= params.blasClusterReferenceCapacity &&
                                    info.clusterReferencesCount <= params.blasClusterReferenceCapacity - instance.clusterReferenceOffset,
                                    "partial overflow build record does not match its admitted instance");
                                matched = true;
                                break;
                            }
                            require(matched, "partial overflow build destination is not an admitted BLAS");
                            builtAddresses.push_back(destination);
                            buildReferences += info.clusterReferencesCount;
                        }
                        std::sort(builtAddresses.begin(), builtAddresses.end());
                        require(std::adjacent_find(builtAddresses.begin(), builtAddresses.end()) == builtAddresses.end(),
                            "partial overflow emitted duplicate BLAS build destinations");
                        require(buildReferences == currentHeader.clusterReferenceCount,
                            "partial overflow build reference count includes rejected instances");
                    };
                    checkStorage(partial);
                    for (uint32_t stableFrame = 0; stableFrame < 2; ++stableFrame) {
                        reuse = execute(false);
                        const auto stable = headerAt(reuse);
                        require(stable.blasBuildCount == 0 && stable.arenaRepack == 0 &&
                            stable.reusedInstances == partialBuilds && stable.storageRejected == partialHeader.storageRejected,
                            "unchanged partial storage overflow repeatedly repacked or rebuilt BLAS");
                        require(stable.semanticChangeCount == 0, "stable partial storage overflow invalidated shading history");
                        for (uint32_t i = 0; i < count; ++i) {
                            const auto before = instanceAt(partial, i);
                            const auto after = instanceAt(reuse, i);
                            require((before.flags & kMeshletStreamBLASInstanceDynamic) == (after.flags & kMeshletStreamBLASInstanceDynamic) &&
                                before.storageOffset == after.storageOffset && before.storageCapacity == after.storageCapacity,
                                "unchanged partial overflow changed admission or persistent storage");
                        }
                        checkStorage(reuse);
                    }
                    // Growing a surviving cut across a size class can evict later
                    // instances, but unchanged survivors must settle immediately.
                    uint32_t growingGroup = 0;
                    while (growingGroup < activeHeader.activeGroupCount &&
                        (instanceAt(partial, active[growingGroup].instanceIndex).flags & kMeshletStreamBLASInstanceDynamic) == 0u) {
                        ++growingGroup;
                    }
                    require(growingGroup < activeHeader.activeGroupCount, "no admitted instance available to grow");
                    const uint32_t growingInstance = active[growingGroup].instanceIndex;
                    const uint32_t originalMask = active[growingGroup].clusterSelectionMask;
                    const uint32_t originalCount = std::popcount(originalMask);
                    require(originalCount > 0 && originalCount <= 16, "growth fixture does not cross a storage size class");
                    active[growingGroup].clusterSelectionMask = 0xffffffffu;
                    upload(Groups, active.data(), sizeof(active));
                    auto grown = execute(false);
                    uint32_t evicted = 0;
                    for (uint32_t i = 0; i < count; ++i) {
                        evicted += (instanceAt(partial, i).flags & kMeshletStreamBLASInstanceDynamic) != 0u &&
                            (instanceAt(grown, i).flags & kMeshletStreamBLASInstanceDynamic) == 0u;
                    }
                    require(evicted > 0 && (instanceAt(grown, growingInstance).flags & kMeshletStreamBLASInstanceDynamic) != 0u,
                        "partial-budget growth did not retain the changed cut and evict another instance");
                    require(headerAt(grown).semanticChangeCount == evicted + 1 &&
                        headerAt(grown).dirtyInstances == headerAt(grown).blasBuildCount,
                        "partial-budget growth invalidated unchanged geometry or omitted changed geometry");
                    checkStorage(grown);
                    const auto requireStableAdmission = [&](const auto& expected, const auto& actual) {
                        const auto current = headerAt(actual);
                        require(current.blasBuildCount == 0 && current.arenaRepack == 0 && current.semanticChangeCount == 0 &&
                            current.storageRejected == headerAt(expected).storageRejected,
                            "partial-budget size-class change did not settle");
                        for (uint32_t i = 0; i < count; ++i) {
                            const auto before = instanceAt(expected, i);
                            const auto after = instanceAt(actual, i);
                            require((before.flags & kMeshletStreamBLASInstanceDynamic) == (after.flags & kMeshletStreamBLASInstanceDynamic) &&
                                before.storageOffset == after.storageOffset && before.storageCapacity == after.storageCapacity,
                                "stable size-class change altered admission or persistent storage");
                        }
                        checkStorage(actual);
                    };
                    for (uint32_t stableFrame = 0; stableFrame < 2; ++stableFrame) {
                        reuse = execute(false);
                        requireStableAdmission(grown, reuse);
                    }
                    active[growingGroup].clusterSelectionMask = originalMask;
                    upload(Groups, active.data(), sizeof(active));
                    auto shrunk = execute(false);
                    uint32_t restored = 0;
                    for (uint32_t i = 0; i < count; ++i) {
                        const bool dynamic = (instanceAt(shrunk, i).flags & kMeshletStreamBLASInstanceDynamic) != 0u;
                        require(dynamic == ((instanceAt(partial, i).flags & kMeshletStreamBLASInstanceDynamic) != 0u),
                            "restoring the original cut did not restore storage admission");
                        restored += dynamic && (instanceAt(grown, i).flags & kMeshletStreamBLASInstanceDynamic) == 0u;
                    }
                    require(restored == evicted && headerAt(shrunk).semanticChangeCount == restored + 1,
                        "cut shrink omitted readmission or invalidated unchanged geometry");
                    checkStorage(shrunk);
                    for (uint32_t stableFrame = 0; stableFrame < 2; ++stableFrame) {
                        reuse = execute(false);
                        requireStableAdmission(shrunk, reuse);
                    }
                    // Releasing selected geometry must make room for a previously
                    // rejected instance, and restoring that geometry must settle.
                    uint32_t releasedGroup = 0;
                    while (releasedGroup < activeHeader.activeGroupCount &&
                        (instanceAt(partial, active[releasedGroup].instanceIndex).flags & kMeshletStreamBLASInstanceDynamic) == 0u) {
                        ++releasedGroup;
                    }
                    require(releasedGroup < activeHeader.activeGroupCount, "no admitted instance available to release");
                    const uint32_t releasedInstance = active[releasedGroup].instanceIndex;
                    const uint32_t releasedMask = active[releasedGroup].clusterSelectionMask;
                    active[releasedGroup].clusterSelectionMask = 0u;
                    upload(Groups, active.data(), sizeof(active));
                    reuse = execute(false);
                    require((instanceAt(reuse, releasedInstance).flags & kMeshletStreamBLASInstanceDynamic) == 0u,
                        "released geometry retained dynamic admission");
                    bool readmitted = false;
                    for (uint32_t i = 0; i < count; ++i) {
                        readmitted |= (instanceAt(partial, i).flags & kMeshletStreamBLASInstanceDynamic) == 0u &&
                            (instanceAt(reuse, i).flags & kMeshletStreamBLASInstanceDynamic) != 0u;
                    }
                    require(readmitted, "released storage did not admit a previously rejected instance");
                    checkStorage(reuse);
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0 && headerAt(reuse).arenaRepack == 0 &&
                        headerAt(reuse).semanticChangeCount == 0, "released partial overflow did not settle");
                    active[releasedGroup].clusterSelectionMask = releasedMask;
                    upload(Groups, active.data(), sizeof(active));
                    reuse = execute(false);
                    require((instanceAt(reuse, releasedInstance).flags & kMeshletStreamBLASInstanceDynamic) != 0u,
                        "restored geometry did not reenter dynamic storage");
                    checkStorage(reuse);
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0 && headerAt(reuse).arenaRepack == 0 &&
                        headerAt(reuse).semanticChangeCount == 0, "restored partial overflow did not settle");
                    // Raising the storage budget retries rejected instances without
                    // forcing a cache reset or a publication-generation change.
                    const uint32_t retryCount = headerAt(reuse).storageRejected;
                    params.blasStorageBytes = 1u << 20;
                    upload(Params, &params, sizeof(params));
                    reuse = execute(false);
                    require(headerAt(reuse).storageRejected == 0 && headerAt(reuse).blasBuildCount == retryCount &&
                        headerAt(reuse).reusedInstances + retryCount == acceptedBuilds,
                        "restored storage capacity did not retry only rejected instances");
                    require(headerAt(reuse).semanticChangeCount == retryCount,
                        "capacity recovery omitted fallback-to-dynamic semantic changes");
                    checkStorage(reuse);
                    reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 0 && headerAt(reuse).arenaRepack == 0 &&
                        headerAt(reuse).reusedInstances == acceptedBuilds && headerAt(reuse).semanticChangeCount == 0,
                        "restored storage capacity did not settle");
                    params.blasStorageBytes = 64;
                    upload(Params, &params, sizeof(params)); reuse = execute(false);
                    require(headerAt(reuse).storageRejected == acceptedBuilds && headerAt(reuse).blasBuildCount == 0, "storage overflow published invalid BLAS");
                    require(headerAt(reuse).semanticChangeCount == acceptedBuilds, "Dynamic-to-fallback publication omitted semantic changes");
                    params.blasStorageBytes = 1u << 20;
                    upload(Params, &params, sizeof(params)); reuse = execute(2u);
                    require(headerAt(reuse).blasBuildCount == acceptedBuilds && headerAt(reuse).storageRejected == 0, "cancel/reset did not restore cache");
                    require(headerAt(reuse).semanticChangeCount == acceptedBuilds, "Fallback-to-dynamic publication omitted semantic changes");
                    reuse = execute(2u);
                    require(headerAt(reuse).semanticChangeCount == 0, "Forced rebuild invalidated identical published geometry");
                    pages[81].addressOffsetAndState = packMeshletStreamClasPageEntry(81 * 32, MeshletStreamCLASPageState::Active);
                    ++pages[81].publicationGeneration;
                    upload(Pages, pages.data(), sizeof(pages)); reuse = execute(false);
                    require(headerAt(reuse).blasBuildCount == 1 && headerAt(reuse).reusedInstances == acceptedBuilds, "republication invalidated unrelated instances");
                    require(headerAt(reuse).semanticChangeCount == 1, "Newly usable dynamic geometry omitted semantic change");
                    sceneInstances[100].visible = 0;
                    upload(SceneInstances, sceneInstances.data(), sizeof(sceneInstances));
                    reuse = execute(false);
                    require(headerAt(reuse).semanticChangeCount == 1, "Removing ray-visible geometry omitted semantic change");
                    fallbackAddress += 0x10000ull;
                    upload(FallbackAddresses, &fallbackAddress, sizeof(fallbackAddress));
                    reuse = execute(false);
                    require(headerAt(reuse).semanticChangeCount == 0, "Stable hidden geometry/fallback address relocation invalidated shading");
                    sceneInstances[100].visible = 1;
                    upload(SceneInstances, sceneInstances.data(), sizeof(sceneInstances));
                    reuse = execute(false);
                    require(headerAt(reuse).semanticChangeCount == 1, "Restoring ray-visible geometry omitted semantic change");
                }
            }
            return RHITestResult::pass("Per-instance exact reuse, local generation/cut invalidation, shifted ranges, bounded growth/repack, stable partial overflow, release/reentry, capacity recovery, reset; selected-cut packing: zero initial slices, multi-block/tail scan, reordered instances, sparse masks, exact fit, reference/build/per-build limits and missing CLAS");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamBLASAllocationTest);
} // namespace
} // namespace metallic::tests
