#pragma once

#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {

constexpr float kLumenDefaultMeshSDFTraceDistance = 1.8f;
constexpr float kLumenCardTraceEndDistance = 40.0f;

struct ShaderLumenMeshSDFGridBindings {
    MeshDistanceFieldGPUBindings meshes;
    ShaderBuffer instances, gridCells, gridIndices;
    std::array<uint32_t, 4> grid;
    std::array<float, 3> cameraPosition;
    float maxMeshSDFTraceDistance, cardTraceEndDistance;
    uint32_t bricksPerAxis;
};
static_assert(sizeof(ShaderLumenMeshSDFGridBindings) == 68);

class LumenMeshSDFGrid {
public:
    struct Frame {
        ShaderLumenMeshSDFGridBindings bindings{};
        EncodedParameters culling;
        std::unique_ptr<Buffer> instances, bounds, motion;
        std::array<BufferSlice, 2> indirect;
        uint32_t cellGroups = 0, objectGroups = 0;
        bool compactViewObjects = false;
    };

    static void reflect(RenderPassReflection& reflection, const RenderGraphCompileContext& context)
    {
        reflection.addTextureInput("detailDepth", "Current reversed-Z depth for Mesh SDF froxel culling")
            .sampledRead().format = Format::D32Sfloat;
        const uint64_t tiles = uint64_t(gridTiles(context.width)) * gridTiles(context.height);
        const uint64_t cells = tiles * kGridMaxDepth;
        const uint64_t maxInstances = context.runtimeScene ?
            std::min<uint64_t>(context.runtimeScene->renderNodes().size(), uint64_t{kMaxDispatchGroups} * kCullThreads) : 0u;
        const uint64_t capacity = std::clamp(cells * std::min(maxInstances, uint64_t{kAveragePairsPerCell}),
            uint64_t{1}, uint64_t{kMaxPairs});
        reflection.addBufferOutput("detailTileDepth", "Closest and furthest reversed-Z depth per froxel tile")
            .buffer(tiles * 8, 8).storageReadWrite();
        reflection.addBufferOutput("detailGridCells", "Mesh SDF candidate start and count per froxel")
            .buffer(cells * 8, 8).storageReadWrite().bufferLayout("RG32U");
        reflection.addBufferOutput("detailGridPairs", "Scattered Mesh SDF instance and froxel indices")
            .buffer(capacity * 8, 8).storageReadWrite();
        reflection.addBufferOutput("detailGridIndices", "View candidates followed by compacted Mesh SDF froxel indices")
            .buffer((maxInstances + capacity) * 4, 4).storageReadWrite().bufferLayout("u32");
        reflection.addBufferOutput("detailGridState", "Attempted pairs, index allocator, view candidates, overflowed froxels")
            .buffer(4 * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite().bufferLayout("u32");
        reflection.addBufferOutput("detailGridIndirectArgs", "Indirect Mesh SDF scatter and compaction dispatches")
            .buffer(2 * kDispatchArgumentBytes, sizeof(uint32_t)).storageWrite()
            .stageAccess(RenderGraphResourceAccess::BufferIndirectRead);
    }

    Result<> compile(Device& device, std::string& log)
    {
        const std::pair<const char*, ComputeKernel*> kernels[]{
            {"lumenMeshSDFGridClearMain", &clear_}, {"lumenMeshSDFGridDepthMain", &depth_},
            {"lumenMeshSDFGridCullViewMain", &cullView_}, {"lumenMeshSDFGridSetupScatterMain", &setupScatter_},
            {"lumenMeshSDFGridScatterMain", &scatter_}, {"lumenMeshSDFGridAllocateMain", &allocate_},
            {"lumenMeshSDFGridCompactMain", &compact_},
        };
        for (const auto& [entry, kernel] : kernels) {
            auto result = ShaderRegistry::instance().getComputeKernel(device,
                {.moduleName = "Features/DistanceField/LumenMeshSDFCulling", .entryPointName = entry,
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<CullingParams>(kCullingABI), .debugName = entry}, *kernel, log);
            if (!result) { return result; }
        }
        return {};
    }

    Result<Frame> prepare(RenderGraphExecutionContext& context, ParameterWriter& writer, const MeshDistanceFieldGPU& meshFields,
        float meshSDFTraceDistance, float cardTraceEndDistance, bool fastLightingUpdate = false) const
    {
        const auto* view = context.viewConstants();
        if (!view || view->current.upProjection[3] > 0.5f) {
            spdlog::error("Lumen Detail Tracing requires a perspective graph camera");
            return makeError(Error::InvalidArgument);
        }
        const float depthRange = cardTraceEndDistance - view->current.clipOrtho[0];
        const uint32_t slices = depthRange > 0.0f ? uint32_t(std::max(0.0f, std::trunc(std::log2(depthRange) * 4.0f))) + 1u : 1u;
        Frame frame;
        std::optional<MeshDistanceFieldGPU::MotionOutput> motion;
        if (fastLightingUpdate) {
            const auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
            const auto* source = gpuScene ? gpuScene->sourceOverride() : nullptr;
            if (!source && context.world()) { source = context.world()->scene(); }
            if (!gpuScene || source != context.runtimeScene()) {
                spdlog::error("Lumen Detail Tracing motion requires the matching GPU scene source");
                return makeError(Error::InvalidArgument);
            }
            motion.emplace(gpuScene->scene(), frame.motion);
        }
        uint32_t instanceCount = 0, unsupportedTransforms = 0;
        auto instances = meshFields.uploadInstances(*context.runtimeScene(), instanceCount, unsupportedTransforms, &frame.bounds,
            nullptr, motion ? &*motion : nullptr);
        if (!instances) { return makeError(instances.error()); }
        frame.instances = std::move(*instances);
        const std::array grid{gridTiles(context.width()), gridTiles(context.height()), slices, instanceCount};
        const uint32_t cellCount = grid[0] * grid[1] * grid[2];
        const uint32_t capacity = uint32_t(context.outputBuffer("detailGridPairs").desc().size / 8u);
        const uint32_t viewCapacity = uint32_t(context.outputBuffer("detailGridIndices").desc().size / 4u) - capacity;
        frame.cellGroups = (cellCount + kCullThreads - 1u) / kCullThreads;
        frame.objectGroups = (instanceCount + kCullThreads - 1u) / kCullThreads;
        if (frame.cellGroups > kMaxDispatchGroups || frame.objectGroups > kMaxDispatchGroups) {
            spdlog::error("Lumen Detail Tracing dispatch exceeds its group limit for {} instances across {} froxels", instanceCount, cellCount);
            return makeError(Error::Unsupported);
        }
        frame.compactViewObjects = instanceCount <= viewCapacity;
        auto* indirect = context.outputBuffer("detailGridIndirectArgs").buffer();
        for (size_t command = 0; command < frame.indirect.size(); ++command) {
            auto slice = indirect->slice({command * kDispatchArgumentBytes, kDispatchArgumentBytes});
            if (!slice) { return makeError(slice.error()); }
            frame.indirect[command] = *slice;
        }
        const auto bindOutput = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
        const auto meshBindings = meshFields.bind(writer);
        const auto instancesHandle = writer.buffer(frame.instances.get());
        frame.bindings = {meshBindings, instancesHandle, bindOutput("detailGridCells"), bindOutput("detailGridIndices"),
            grid, {view->current.eye[0], view->current.eye[1], view->current.eye[2]}, meshSDFTraceDistance,
            cardTraceEndDistance, meshFields.bricksPerAxis()};
        auto culling = writer.encode(CullingParams{view->current,
            {view->jitter[0], view->jitter[1], meshSDFTraceDistance, cardTraceEndDistance}, grid,
            writer.sampledImageHandle(context.inputTexture("detailDepth").view()), bindOutput("detailTileDepth"),
            writer.buffer(frame.bounds.get()), instancesHandle, meshBindings, frame.bindings.gridCells, frame.bindings.gridIndices,
            bindOutput("detailGridPairs"), bindOutput("detailGridState"), writer.buffer(indirect), capacity, viewCapacity}, kCullingABI);
        if (!culling) { return makeError(culling.error()); }
        frame.culling = *culling;
        if (context.detailedDebugSnapshots()) {
            context.debugCheckpoint("Lumen Mesh SDF Grid", {}, {{"instanceCount", instanceCount},
                {"unsupportedTransforms", unsupportedTransforms}, {"cellCount", cellCount}, {"capacity", capacity}});
        }
        return frame;
    }

    static void appendTraceUses(std::vector<RenderGraphStageUse>& uses)
    {
        using Access = RenderGraphResourceAccess;
        uses.insert(uses.end(), {{"detailGridCells", Access::BufferStorageRead}, {"detailGridIndices", Access::BufferStorageRead}});
    }

    void appendStages(std::vector<RenderGraphStage>& stages, const Frame& frame) const
    {
        using Access = RenderGraphResourceAccess;
        static constexpr RenderGraphStageUse clearUses[]{{"detailGridCells", Access::BufferStorageWrite},
            {"detailGridState", Access::BufferStorageWrite}, {"detailGridIndirectArgs", Access::BufferStorageWrite}};
        static constexpr RenderGraphStageUse depthUses[]{{"detailDepth", Access::TextureSampleRead},
            {"detailTileDepth", Access::BufferStorageWrite}};
        static constexpr RenderGraphStageUse cullViewUses[]{{"detailGridIndices", Access::BufferStorageWrite},
            {"detailGridState", Access::BufferStorageReadWrite}};
        static constexpr RenderGraphStageUse setupScatterUses[]{{"detailGridState", Access::BufferStorageRead},
            {"detailGridIndirectArgs", Access::BufferStorageWrite}};
        static constexpr RenderGraphStageUse scatterUses[]{{"detailTileDepth", Access::BufferStorageRead},
            {"detailGridCells", Access::BufferStorageReadWrite}, {"detailGridPairs", Access::BufferStorageWrite},
            {"detailGridState", Access::BufferStorageReadWrite}, {"detailGridIndices", Access::BufferStorageRead},
            {"detailGridIndirectArgs", Access::BufferIndirectRead}};
        static constexpr RenderGraphStageUse allocateUses[]{{"detailGridCells", Access::BufferStorageReadWrite},
            {"detailGridState", Access::BufferStorageReadWrite}, {"detailGridIndirectArgs", Access::BufferStorageWrite}};
        static constexpr RenderGraphStageUse compactUses[]{{"detailGridCells", Access::BufferStorageReadWrite},
            {"detailGridPairs", Access::BufferStorageRead}, {"detailGridState", Access::BufferStorageRead},
            {"detailGridIndices", Access::BufferStorageWrite}, {"detailGridIndirectArgs", Access::BufferIndirectRead}};
        stages.push_back({"Clear Mesh SDF candidate ranges", clearUses, [this, &frame](CommandBuffer& cmd) {
            return clear_.dispatch(cmd, frame.culling, frame.cellGroups, 1, 1);
        }});
        if (frame.objectGroups == 0) { return; }
        stages.push_back({"Reduce closest and furthest froxel depth", depthUses, [this, &frame](CommandBuffer& cmd) {
            return depth_.dispatch(cmd, frame.culling, frame.bindings.grid[0], frame.bindings.grid[1], 1);
        }});
        if (frame.compactViewObjects) {
            stages.push_back({"Cull Mesh SDFs to view", cullViewUses, [this, &frame](CommandBuffer& cmd) {
                return cullView_.dispatch(cmd, frame.culling, frame.objectGroups, 1, 1);
            }});
        }
        stages.push_back({"Build Mesh SDF froxel scatter dispatch", setupScatterUses, [this, &frame](CommandBuffer& cmd) {
            return setupScatter_.dispatch(cmd, frame.culling, 1, 1, 1);
        }});
        stages.push_back({"Cull Mesh SDFs to view froxels", scatterUses, [this, &frame](CommandBuffer& cmd) {
            return scatter_.dispatchIndirect(cmd, frame.culling, frame.indirect[0]);
        }});
        stages.push_back({"Allocate Mesh SDF candidate ranges", allocateUses, [this, &frame](CommandBuffer& cmd) {
            return allocate_.dispatch(cmd, frame.culling, frame.cellGroups, 1, 1);
        }});
        stages.push_back({"Compact Mesh SDF candidates", compactUses, [this, &frame](CommandBuffer& cmd) {
            return compact_.dispatchIndirect(cmd, frame.culling, frame.indirect[1]);
        }});
    }

private:
    static constexpr uint64_t kCullingABI = 0x4c554d454e060002ull;
    static constexpr uint32_t kGridTileSize = 64;
    static constexpr uint32_t kGridMaxDepth = 22;
    static constexpr uint32_t kCullThreads = 64;
    static constexpr uint32_t kMaxDispatchGroups = 65535;
    static constexpr uint32_t kAveragePairsPerCell = 512;
    static constexpr uint32_t kMaxPairs = (128u * 1024 * 1024) / (2 * sizeof(uint32_t));
    static constexpr uint32_t kDispatchArgumentBytes = 3 * sizeof(uint32_t);

    struct CullingParams {
        ViewCameraConstants camera;
        std::array<float, 4> cull;
        std::array<uint32_t, 4> grid;
        ShaderSampledImage depth;
        ShaderBuffer tileDepth, bounds, instances;
        MeshDistanceFieldGPUBindings meshes;
        ShaderBuffer gridCells, gridIndices, gridPairs, gridState, gridArgs;
        uint32_t pairCapacity, viewCapacity, padding = 0;
    };
    static_assert(sizeof(CullingParams) == 176 && offsetof(CullingParams, meshes) == 128 &&
        offsetof(CullingParams, pairCapacity) == 164);

    static uint32_t gridTiles(uint32_t extent)
    {
        return (extent + kGridTileSize - 1u) / kGridTileSize;
    }

    ComputeKernel clear_, depth_, cullView_, setupScatter_, scatter_, allocate_, compact_;
};

} // namespace metallic::render::builtin_pass
