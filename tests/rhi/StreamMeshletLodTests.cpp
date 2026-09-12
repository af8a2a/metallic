#include "RhiTest.h"
#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/SlangCompiler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

using namespace render;

struct StreamLodFixture {
    std::vector<MeshletLodGroupRecord> groups;
    std::vector<MeshletLodGroupRange> ranges{{0, 2}, {2, 2}, {4, 1}, {5, 2}, {7, 1}, {8, 2}};
    std::vector<uint32_t> refined{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, 0, 1, 0, 3, 4};
    // Every original surface atom must be covered exactly once by every cut.
    std::array<uint32_t, 10> coverage{1, 2, 4, 8, 16, 1, 12, 2, 13, 2};
    std::vector<uint8_t> drawable = std::vector<uint8_t>(6, 1);
    GPUSceneGpuInstanceRecord instance;
    MeshletLodView view;

    StreamLodFixture()
    {
        for (uint32_t group = 0; group < ranges.size(); ++group) {
            groups.push_back({.sphere = {0, 0, 0, 1},
                .error = group < 3 ? .5f : group < 5 ? 2.f : 4.f,
                .level = group < 3 ? 0u : group < 5 ? 1u : 2u,
                .flags = group == 2 || group == 5 ? kMeshletLodTerminalGroup : 0u});
        }
        instance.worldMatrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        instance.identity[3] = GPUSceneGpuInstanceVisible;
        view.eye = {0, 0, 10, .1f};
        view.forward = {0, 0, -1, 1};
        view.projection = {1000, .577350269f, 100, 1.5f};
    }

    StreamMeshletLodReference select(uint32_t manual = UINT32_MAX, uint32_t capacity = UINT32_MAX) const
    {
        return selectStreamMeshletLodReference(groups, ranges, refined, drawable, instance, view, manual, capacity);
    }

    bool coversExactlyOnce(const StreamMeshletLodReference& cut) const
    {
        if (!cut.valid) { return false; }
        uint32_t covered = 0;
        for (uint32_t cluster : cut.selectedClusters) {
            if (cluster >= coverage.size() || (covered & coverage[cluster]) != 0) { return false; }
            covered |= coverage[cluster];
        }
        return covered == 31;
    }
};

class StreamMeshletLodReferenceTest final : public RhiTest {
public:
    StreamMeshletLodReferenceTest() { type = RhiTestType::Validation; name = "meshlet_lod_stream_reference_frontier"; }
    RhiTestResult run(RhiTestContext&) override
    {
        StreamLodFixture fixture;
        auto cut = fixture.select();
        if (!fixture.coversExactlyOnce(cut) || cut.selectedClusters != std::vector<uint32_t>{0, 1, 2, 3, 4}) {
            return RhiTestResult::fail("fully resident stream cut did not refine");
        }
        fixture.drawable[3] = 0;
        cut = fixture.select();
        if (!fixture.coversExactlyOnce(cut) || cut.selectedClusters != std::vector<uint32_t>{4, 7, 8} ||
            cut.requestedGroups != std::vector<uint32_t>{3}) {
            return RhiTestResult::fail("missing ancestor exposed a descendant with another resident parent");
        }
        for (uint32_t residency = 0; residency < 16; ++residency) {
            for (uint32_t group : {0u, 1u, 3u, 4u}) {
                const uint32_t bit = group < 2 ? group : group - 1;
                fixture.drawable[group] = (residency >> bit) & 1;
            }
            for (uint32_t manual : {UINT32_MAX, 0u, 1u, 2u, 31u}) {
                cut = fixture.select(manual);
                if (!fixture.coversExactlyOnce(cut)) {
                    return RhiTestResult::fail("residency/LOD combination overlaps or omits a surface atom");
                }
            }
        }
        std::fill(fixture.drawable.begin(), fixture.drawable.end(), uint8_t{1});
        cut = fixture.select(UINT32_MAX, 2);
        if (!fixture.coversExactlyOnce(cut) || !cut.capacityExceeded || !cut.capacityFallback ||
            cut.selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RhiTestResult::fail("capacity did not atomically fall back to the complete terminal cut");
        }
        cut = fixture.select(UINT32_MAX, 1);
        if (cut.valid || !cut.capacityExceeded || !cut.selectedClusters.empty()) {
            return RhiTestResult::fail("an undersized terminal capacity exposed a partial cut");
        }
        fixture.drawable[5] = 0;
        cut = fixture.select();
        if (cut.valid || !cut.selectedClusters.empty() || cut.requestedGroups != std::vector<uint32_t>{5}) {
            return RhiTestResult::fail("missing terminal page exposed orphan descendants");
        }
        fixture.drawable[5] = 1;
        fixture.drawable[3] = 0;
        const std::array<uint8_t, 6> available{1, 1, 1, 1, 1, 1};
        cut = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
            fixture.drawable, fixture.instance, fixture.view, UINT32_MAX, UINT32_MAX, available);
        if (!fixture.coversExactlyOnce(cut) || !cut.requestedGroups.empty() ||
            cut.selectedClusters != std::vector<uint32_t>{4, 7, 8}) {
            return RhiTestResult::fail("pending upload replaced the fallback or generated a duplicate request");
        }
        fixture.drawable[3] = 1;
        fixture.view.projection[3] = 100;
        if (fixture.select().selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RhiTestResult::fail("orthographic pixel error did not select cross-level terminal groups");
        }
        fixture.view.forward[3] = 0;
        fixture.view.eye[2] = .1f;
        if (fixture.select().selectedClusters != std::vector<uint32_t>{0, 1, 2, 3, 4}) {
            return RhiTestResult::fail("near-plane error did not refine");
        }
        fixture.instance.identity[3] = 0;
        cut = fixture.select();
        if (!cut.valid || !cut.selectedClusters.empty() || !cut.requestedGroups.empty()) {
            return RhiTestResult::fail("hidden instance produced geometry or page requests");
        }
        fixture.refined[8] = 5;
        if (fixture.select().valid) { return RhiTestResult::fail("cyclic stream topology accepted"); }
        return RhiTestResult::pass("80 residency/manual cuts preserve exact coverage; multi-parent gating, mixed-depth terminals, complete capacity fallback and near-plane projection");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLodReferenceTest);

#define STREAM_LOD_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked)); } } while (false)

class StreamMeshletLodGpuTest final : public RhiTest {
public:
    StreamMeshletLodGpuTest() { type = RhiTestType::Rendering; name = "meshlet_lod_stream_gpu_matches_reference"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        StreamLodFixture fixture;
        constexpr uint32_t kGroupCount = 6;
        constexpr uint32_t kRequestWords = 32;
        std::array<MeshletStreamGpuGroup, kGroupCount> groups{};
        std::vector<uint32_t> topology = fixture.refined;
        for (uint32_t group = 0; group < kGroupCount; ++group) {
            auto& output = groups[group];
            output.pageIndex = group;
            output.lodLevel = fixture.groups[group].level;
            output.clusterCount = fixture.ranges[group].clusterCount;
            std::copy(fixture.groups[group].sphere.begin(), fixture.groups[group].sphere.end(), output.boundsCenterRadius);
            output.maxQuadricError = fixture.groups[group].error;
            output.clusterRefinedOffset = fixture.ranges[group].clusterOffset;
            output.flags = fixture.groups[group].flags;
            output.parentOffset = static_cast<uint32_t>(topology.size());
            for (uint32_t parent = group + 1; parent < kGroupCount; ++parent) {
                const auto& range = fixture.ranges[parent];
                if (std::find(fixture.refined.begin() + range.clusterOffset,
                    fixture.refined.begin() + range.clusterOffset + range.clusterCount, group) !=
                    fixture.refined.begin() + range.clusterOffset + range.clusterCount) {
                    topology.push_back(parent);
                    ++output.parentCount;
                }
            }
        }
        const uint32_t instanceOffsetsOffset = static_cast<uint32_t>(topology.size());
        topology.push_back(4);
        MeshletStreamGpuPrimitive primitive;
        primitive.groupCount = kGroupCount;
        primitive.pageCount = kGroupCount;
        primitive.lodLevelCount = 3;
        enum BufferIndex { Instance, Primitive, Groups, Params, Topology, State, PageTable,
            Requests, ActiveGroups, Header, Arguments, Dummy, BufferCount };
        const uint32_t strides[] = {sizeof(MeshletStreamGpuInstance), sizeof(primitive), sizeof(MeshletStreamGpuGroup),
            sizeof(MeshletStreamGpuParams), 4, 4, sizeof(StreamPageTableEntry), 4,
            sizeof(MeshletStreamGpuActiveGroup), sizeof(MeshletStreamGpuActiveHeader), sizeof(MeshletStreamGpuDrawIndirect), 4};
        const uint64_t sizes[] = {strides[Instance], sizeof(primitive), sizeof(groups), strides[Params],
            topology.size() * 4, (4 + kGroupCount * 2) * 4, kGroupCount * sizeof(StreamPageTableEntry),
            kRequestWords * 4, kGroupCount * sizeof(MeshletStreamGpuActiveGroup), strides[Header], strides[Arguments], 64};
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream LOD frontier regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless compute"); }
        STREAM_LOD_REQUIRE(created);
        std::unique_ptr<BindlessHeap> heap;
        STREAM_LOD_REQUIRE(device->createBindlessHeap({.maxBuffers = BufferCount}, heap));
        std::array<std::unique_ptr<Buffer>, BufferCount> buffers;
        std::array<BindlessHandle, BufferCount> handles;
        for (uint32_t index = 0; index < BufferCount; ++index) {
            STREAM_LOD_REQUIRE(device->createBuffer({.size = sizes[index], .structureStride = strides[index],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload}, buffers[index]));
            STREAM_LOD_REQUIRE(heap->allocateBuffer(handles[index]));
            STREAM_LOD_REQUIRE(heap->writeStorageBuffer(handles[index], *buffers[index]));
        }
        const auto upload = [&](BufferIndex index, const void* source, size_t size) {
            void* mapped = buffers[index]->map();
            if (mapped == nullptr) { return false; }
            std::memset(mapped, 0, sizes[index]);
            if (source != nullptr) { std::memcpy(mapped, source, size); }
            buffers[index]->flush();
            buffers[index]->unmap();
            return true;
        };
        for (uint32_t index = 0; index < BufferCount; ++index) {
            if (!upload(static_cast<BufferIndex>(index), nullptr, 0)) { return RhiTestResult::fail("stream test buffer map"); }
        }
        if (!upload(Primitive, &primitive, sizeof(primitive)) || !upload(Groups, groups.data(), sizeof(groups)) ||
            !upload(Topology, topology.data(), sizes[Topology])) { return RhiTestResult::fail("stream topology map"); }
        ShaderCompileResult compiled;
        const auto compile = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamActiveBuildEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, compiled);
        if (!compile) { return RhiTestResult::fail("stream frontier shader compile: " + compiled.diagnostics); }
        std::unique_ptr<ShaderModule> shader;
        STREAM_LOD_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4}, shader));
        std::unique_ptr<ComputePipeline> pipeline;
        STREAM_LOD_REQUIRE(device->createComputePipeline({.computeShader = shader.get(), .computeEntryPoint = "main",
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush)}, pipeline));
        MeshletStreamUserPush push;
        push.instanceBuffer = handles[Instance].index;
        push.primitiveBuffer = handles[Primitive].index;
        push.groupBuffer = handles[Groups].index;
        push.paramsBuffer = handles[Params].index;
        push.pageTableBuffer = handles[PageTable].index;
        push.requestBuffer = handles[Requests].index;
        push.activeGroupBuffer = handles[ActiveGroups].index;
        push.activeHeaderBuffer = handles[Header].index;
        push.drawIndirectBuffer = handles[Arguments].index;
        push.pageBuffer = handles[Dummy].index;
        push.nodeBuffer = handles[Dummy].index;
        push.lodLevelBuffer = handles[Dummy].index;
        push.traversalHeaderBuffer = handles[Dummy].index;
        push.traversalWorkBuffer = handles[Dummy].index;
        std::unique_ptr<Buffer> readback;
        constexpr uint32_t outputs[] = {Header, ActiveGroups, Requests, Arguments};
        std::array<uint64_t, 4> offsets{};
        uint64_t outputBytes = 0;
        for (uint32_t index = 0; index < 4; ++index) { offsets[index] = outputBytes; outputBytes += sizes[outputs[index]]; }
        STREAM_LOD_REQUIRE(device->createBuffer({.size = outputBytes, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}, readback));
        Queue* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        STREAM_LOD_REQUIRE(device->createCommandPool(*queue, pool));
        STREAM_LOD_REQUIRE(pool->createCommandBuffer(commands));
        STREAM_LOD_REQUIRE(device->createFence(false, fence));
        for (uint32_t test = 0; test < 37; ++test) {
            fixture = StreamLodFixture{};
            uint32_t manual = UINT32_MAX, capacity = kGroupCount;
            if (test < 16) {
                for (uint32_t group : {0u, 1u, 3u, 4u}) {
                    const uint32_t bit = group < 2 ? group : group - 1;
                    fixture.drawable[group] = (test >> bit) & 1;
                }
            } else if (test < 32) {
                fixture.view.forward[3] = test % 2 ? 1.f : 0.f;
                fixture.view.eye[2] = test % 4 ? 30.f : .1f;
                fixture.view.projection[0] = test % 3 ? 1000.f : 500.f;
                fixture.view.projection[3] = test % 5 ? 1.5f : 100.f;
                fixture.instance.worldMatrix[4] = test % 3 ? 2.f : 0.f;
                fixture.instance.worldMatrix[0] = test % 2 ? -2.f : 1.f;
                if (test >= 28) { manual = test == 31 ? 31 : test - 28; }
            } else if (test == 32 || test == 33) {
                capacity = test == 32 ? 2 : 1;
            } else if (test == 34) {
                fixture.drawable[5] = 0;
            } else if (test == 35) {
                fixture.instance.identity[3] = 0;
            } else {
                fixture.drawable[3] = 0;
            }
            std::vector<uint8_t> available = fixture.drawable;
            if (test == 36) { available[3] = 1; }
            const auto expected = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                fixture.drawable, fixture.instance, fixture.view, manual, capacity, available);
            MeshletStreamGpuInstance instance;
            instance.visible = fixture.instance.identity[3] != 0;
            instance.gpuSceneInstanceIndex = 17;
            std::copy_n(fixture.instance.worldMatrix.data(), 4, instance.world0);
            std::copy_n(fixture.instance.worldMatrix.data() + 4, 4, instance.world1);
            std::copy_n(fixture.instance.worldMatrix.data() + 8, 4, instance.world2);
            std::copy_n(fixture.instance.worldMatrix.data() + 12, 4, instance.world3);
            MeshletStreamGpuParams params;
            std::copy_n(fixture.view.eye.data(), 3, params.eye);
            for (uint32_t axis = 0; axis < 3; ++axis) { params.center[axis] = params.eye[axis] + fixture.view.forward[axis]; }
            params.upProjection[1] = 1;
            params.upProjection[3] = fixture.view.forward[3];
            params.viewport[2] = fixture.view.projection[0];
            params.viewport[3] = 2 * std::atan(fixture.view.projection[1]);
            params.clipOrtho[0] = fixture.view.eye[3];
            params.clipOrtho[2] = fixture.view.projection[2];
            params.lodPixelError = fixture.view.projection[3];
            params.lodTopologyBuffer = handles[Topology].index;
            params.lodStateBuffer = handles[State].index;
            params.lodInstanceOffsetsOffset = instanceOffsetsOffset;
            params.sceneInstanceCount = 1;
            params.scenePrimitiveCount = 1;
            params.sceneGroupCount = kGroupCount;
            params.scenePageCount = kGroupCount;
            params.maxActiveGroupClusters = 2;
            params.activeGroupCount = capacity;
            params.drawTaskCount = capacity * 4;
            params.frameIndex = test + 1;
            params.selectedLodLevel = manual;
            params.enableGpuLodSelection = manual == UINT32_MAX;
            params.maxGpuPageRequests = 8;
            std::array<StreamPageTableEntry, kGroupCount> pages{};
            for (uint32_t group = 0; group < kGroupCount; ++group) {
                pages[group].deviceOffsetAndState = packStreamPageTableEntry(
                    fixture.drawable[group] ? group * 512 : kInvalidStreamDeviceOffsetBytes,
                    fixture.drawable[group] ? MeshletStreamPageResidencyState::Resident : MeshletStreamPageResidencyState::Unloaded);
            }
            if (test == 36) {
                pages[3].deviceOffsetAndState = packStreamPageTableEntry(1536, MeshletStreamPageResidencyState::PendingUpload);
            }
            std::array<uint32_t, kRequestWords> requests{};
            requests[0] = 8;
            requests[1] = 8;
            requests[4] = params.frameIndex;
            if (!upload(Instance, &instance, sizeof(instance)) || !upload(Params, &params, sizeof(params)) ||
                !upload(PageTable, pages.data(), sizeof(pages)) || !upload(Requests, requests.data(), sizeof(requests))) {
                return RhiTestResult::fail("stream frame input map");
            }
            if (test != 0) { STREAM_LOD_REQUIRE(fence->reset()); STREAM_LOD_REQUIRE(pool->reset()); }
            STREAM_LOD_REQUIRE(commands->begin());
            std::array<BufferBarrierDesc, BufferCount> barriers{};
            for (uint32_t index = 0; index < BufferCount; ++index) {
                barriers[index] = {.buffer = buffers[index].get(),
                    .before = test == 0 ? ResourceState::Undefined : ResourceState::General, .after = ResourceState::General};
            }
            commands->barrier({.buffers = barriers.data(), .bufferCount = BufferCount});
            commands->bindBindlessHeap(*heap);
            commands->bindComputePipeline(*pipeline);
            for (uint32_t phase : {0u, 5u, 6u, 7u, 2u}) {
                push.activeBuildPhase = phase;
                commands->pushBindlessData(&push, sizeof(push));
                commands->dispatch(1, 1, 1);
                for (auto& barrier : barriers) { barrier.before = ResourceState::General; }
                commands->barrier({.buffers = barriers.data(), .bufferCount = BufferCount});
            }
            for (uint32_t index = 0; index < 4; ++index) {
                BufferBarrierDesc barrier{.buffer = buffers[outputs[index]].get(),
                    .before = ResourceState::General, .after = ResourceState::TransferSource};
                commands->barrier({.buffers = &barrier, .bufferCount = 1});
                commands->copyBuffer({.source = buffers[outputs[index]].get(), .destination = readback.get(),
                    .destinationOffset = offsets[index], .size = sizes[outputs[index]]});
                std::swap(barrier.before, barrier.after);
                commands->barrier({.buffers = &barrier, .bufferCount = 1});
            }
            STREAM_LOD_REQUIRE(commands->end());
            CommandBuffer* submitted[] = {commands.get()};
            STREAM_LOD_REQUIRE(queue->submit({.commandBuffers = submitted, .commandBufferCount = 1, .signalFence = fence.get()}));
            STREAM_LOD_REQUIRE(fence->wait());
            readback->invalidate();
            const auto* mapped = static_cast<const uint8_t*>(readback->map());
            if (mapped == nullptr) { return RhiTestResult::fail("stream result map"); }
            MeshletStreamGpuActiveHeader header;
            std::memcpy(&header, mapped, sizeof(header));
            std::array<MeshletStreamGpuActiveGroup, kGroupCount> active;
            std::memcpy(active.data(), mapped + offsets[1], sizeof(active));
            std::memcpy(requests.data(), mapped + offsets[2], sizeof(requests));
            MeshletStreamGpuDrawIndirect arguments;
            std::memcpy(&arguments, mapped + offsets[3], sizeof(arguments));
            readback->unmap();
            if (header.activeGroupCount > capacity || header.activeGroupCount > kGroupCount) {
                return RhiTestResult::fail("stream frontier overflow in case " + std::to_string(test));
            }
            std::vector<uint32_t> actual;
            for (uint32_t index = 0; index < header.activeGroupCount; ++index) {
                const auto& group = active[index];
                if (group.pageIndex >= kGroupCount || group.gpuSceneInstanceIndex != 17 || group.instanceIndex != 0) {
                    return RhiTestResult::fail("stream selected group identity");
                }
                for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                    if ((group.clusterSelectionMask & (1u << cluster)) != 0) {
                        actual.push_back(fixture.ranges[group.pageIndex].clusterOffset + cluster);
                    }
                }
            }
            if (requests[2] > 8 || requests[5] != 0 || requests[7] != 0) {
                return RhiTestResult::fail("stream frontier request overflow or invalid page");
            }
            std::vector<uint32_t> requested(requests.begin() + 16, requests.begin() + 16 + requests[2]);
            std::sort(requested.begin(), requested.end());
            if (actual != expected.selectedClusters || requested != expected.requestedGroups ||
                (expected.valid && bool(header.padding0) != expected.capacityFallback) ||
                arguments.groupCountX != header.activeGroupCount * 4 || arguments.groupCountY != 1 || arguments.groupCountZ != 1) {
                return RhiTestResult::fail("CPU/GPU stream cut, requests, capacity fallback or indirect arguments mismatch in case " +
                    std::to_string(test) + ": expected " + std::to_string(expected.selectedClusters.size()) +
                    " clusters, actual " + std::to_string(actual.size()));
            }
        }
        return RhiTestResult::pass("37 stable GPU/reference stream cuts: arbitrary residency, shared parents, projection, transforms, manual LOD, capacity fallback and missing terminals");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLodGpuTest);

#undef STREAM_LOD_REQUIRE

} // namespace
} // namespace metallic::tests
