#pragma once

#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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
        std::unique_ptr<Buffer> instances, bounds;
        uint32_t cellGroups = 0, objectGroups = 0, compactGroups = 0;
    };

    static void reflect(RenderPassReflection& reflection, const RenderGraphCompileContext& context)
    {
        reflection.addTextureInput("detailDepth", "Current reversed-Z depth for Mesh SDF froxel culling")
            .sampledRead().format = Format::D32Sfloat;
        const uint64_t tiles = uint64_t(gridTiles(context.width)) * gridTiles(context.height);
        const uint64_t cells = tiles * kGridMaxDepth;
        const uint64_t maxInstances = context.runtimeScene ? context.runtimeScene->renderNodes().size() : 0u;
        const uint64_t capacity = std::clamp(cells * maxInstances, uint64_t{1}, uint64_t{kMaxPairs});
        reflection.addBufferOutput("detailTileDepth", "Closest and furthest reversed-Z depth per froxel tile")
            .buffer(tiles * 8, 8).storageReadWrite();
        reflection.addBufferOutput("detailGridCells", "Mesh SDF candidate start and count per froxel")
            .buffer(cells * 8, 8).storageReadWrite().bufferLayout("RG32U");
        reflection.addBufferOutput("detailGridPairs", "Scattered Mesh SDF instance and froxel indices")
            .buffer(capacity * 8, 8).storageReadWrite();
        reflection.addBufferOutput("detailGridIndices", "Compacted Mesh SDF instance indices")
            .buffer(capacity * 4, 4).storageReadWrite().bufferLayout("u32");
        reflection.addBufferOutput("detailGridState", "Candidate count and compacted index allocator")
            .buffer(4 * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite().bufferLayout("u32");
    }

    Result<> compile(Device& device, std::string& log)
    {
        const std::pair<const char*, ComputeKernel*> kernels[]{
            {"lumenMeshSDFGridClearMain", &clear_}, {"lumenMeshSDFGridDepthMain", &depth_},
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
        float meshSDFTraceDistance, float cardTraceEndDistance) const
    {
        const auto* view = context.viewConstants();
        if (!view || view->current.upProjection[3] > 0.5f) {
            spdlog::error("Lumen Detail Tracing requires a perspective graph camera");
            return makeError(Error::InvalidArgument);
        }
        const float depthRange = cardTraceEndDistance - view->current.clipOrtho[0];
        const uint32_t slices = depthRange > 0.0f ? uint32_t(std::max(0.0f, std::trunc(std::log2(depthRange) * 4.0f))) + 1u : 1u;
        Frame frame;
        uint32_t instanceCount = 0, unsupportedTransforms = 0;
        auto instances = meshFields.uploadInstances(*context.runtimeScene(), instanceCount, unsupportedTransforms, &frame.bounds);
        if (!instances) { return makeError(instances.error()); }
        frame.instances = std::move(*instances);
        const std::array grid{gridTiles(context.width()), gridTiles(context.height()), slices, instanceCount};
        const uint32_t cellCount = grid[0] * grid[1] * grid[2];
        const uint32_t capacity = uint32_t(context.outputBuffer("detailGridPairs").desc().size / 8u);
        frame.cellGroups = (cellCount + kCullThreads - 1u) / kCullThreads;
        frame.objectGroups = (instanceCount + kCullThreads - 1u) / kCullThreads;
        if (uint64_t(cellCount) * instanceCount > capacity || frame.cellGroups > kMaxDispatchGroups ||
            frame.objectGroups > kMaxDispatchGroups) {
            spdlog::error("Lumen Detail Tracing candidate pool cannot hold {} instances across {} froxels", instanceCount, cellCount);
            return makeError(Error::Unsupported);
        }
        const uint32_t candidateSlots = cellCount * instanceCount;
        frame.compactGroups = (candidateSlots + kCullThreads - 1u) / kCullThreads;
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
            bindOutput("detailGridPairs"), bindOutput("detailGridState")}, kCullingABI);
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
            {"detailGridState", Access::BufferStorageWrite}};
        static constexpr RenderGraphStageUse depthUses[]{{"detailDepth", Access::TextureSampleRead},
            {"detailTileDepth", Access::BufferStorageWrite}};
        static constexpr RenderGraphStageUse scatterUses[]{{"detailTileDepth", Access::BufferStorageRead},
            {"detailGridCells", Access::BufferStorageReadWrite}, {"detailGridPairs", Access::BufferStorageWrite},
            {"detailGridState", Access::BufferStorageReadWrite}};
        static constexpr RenderGraphStageUse allocateUses[]{{"detailGridCells", Access::BufferStorageReadWrite},
            {"detailGridState", Access::BufferStorageReadWrite}};
        static constexpr RenderGraphStageUse compactUses[]{{"detailGridCells", Access::BufferStorageReadWrite},
            {"detailGridPairs", Access::BufferStorageRead}, {"detailGridState", Access::BufferStorageRead},
            {"detailGridIndices", Access::BufferStorageWrite}};
        stages.push_back({"Clear Mesh SDF candidate ranges", clearUses, [this, &frame](CommandBuffer& cmd) {
            return clear_.dispatch(cmd, frame.culling, frame.cellGroups, 1, 1);
        }});
        if (frame.objectGroups == 0) { return; }
        stages.push_back({"Reduce closest and furthest froxel depth", depthUses, [this, &frame](CommandBuffer& cmd) {
            return depth_.dispatch(cmd, frame.culling, frame.bindings.grid[0], frame.bindings.grid[1], 1);
        }});
        stages.push_back({"Cull Mesh SDFs to view froxels", scatterUses, [this, &frame](CommandBuffer& cmd) {
            return scatter_.dispatch(cmd, frame.culling, frame.bindings.grid[0], frame.bindings.grid[1], frame.objectGroups);
        }});
        stages.push_back({"Allocate Mesh SDF candidate ranges", allocateUses, [this, &frame](CommandBuffer& cmd) {
            return allocate_.dispatch(cmd, frame.culling, frame.cellGroups, 1, 1);
        }});
        stages.push_back({"Compact Mesh SDF candidates", compactUses, [this, &frame](CommandBuffer& cmd) {
            return compact_.dispatch(cmd, frame.culling, std::min(frame.compactGroups, kMaxDispatchGroups),
                (frame.compactGroups + kMaxDispatchGroups - 1u) / kMaxDispatchGroups, 1);
        }});
    }

private:
    static constexpr uint64_t kCullingABI = 0x4c554d454e060001ull;
    static constexpr uint32_t kGridTileSize = 64;
    static constexpr uint32_t kGridMaxDepth = 22;
    static constexpr uint32_t kCullThreads = 64;
    static constexpr uint32_t kMaxDispatchGroups = 65535;
    static constexpr uint32_t kMaxPairs = (128u * 1024 * 1024) / (2 * sizeof(uint32_t));

    struct CullingParams {
        ViewCameraConstants camera;
        std::array<float, 4> cull;
        std::array<uint32_t, 4> grid;
        ShaderSampledImage depth;
        ShaderBuffer tileDepth, bounds, instances;
        MeshDistanceFieldGPUBindings meshes;
        ShaderBuffer gridCells, gridIndices, gridPairs, gridState;
    };
    static_assert(sizeof(CullingParams) == 160 && offsetof(CullingParams, meshes) == 128);

    static uint32_t gridTiles(uint32_t extent)
    {
        return (extent + kGridTileSize - 1u) / kGridTileSize;
    }

    ComputeKernel clear_, depth_, scatter_, allocate_, compact_;
};

} // namespace metallic::render::builtin_pass
