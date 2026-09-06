#include "RhiTest.h"

#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace metallic::tests {
namespace {

#define GRID_CHECK(condition) \
    do { \
        if (!(condition)) { \
            return RhiTestResult::fail(std::string("ClusterLightGrid: ") + #condition); \
        } \
    } while (false)

scene::PunctualLight makeGridLight(
    const char* type, const float3& position, double range)
{
    scene::PunctualLight light;
    light.properties.type = type;
    light.properties.intensity = 100.0;
    light.properties.range = range;
    light.position = position;
    light.direction = float3(0.0f, 0.0f, -1.0f);
    return light;
}

render::ClusterLightGridDesc gridDesc()
{
    render::ClusterLightGridDesc desc;
    desc.width = 128;
    desc.height = 128;
    desc.tileSize = 64;
    desc.depthSliceCount = 4;
    desc.maxLightsPerCell = 8;
    desc.fovRadians = static_cast<float>(std::acos(-1.0) * 0.5);
    desc.zNear = 1.0f;
    desc.zFar = 16.0f;
    return desc;
}

struct GridReadback {
    render::ClusterLightGridSnapshot snapshot;
    std::vector<render::ClusterLightGridCell> cells;
    std::vector<uint32_t> indices;
    std::array<std::array<uint32_t, 4>, 9> lookup{};

    std::vector<uint32_t> cellLights(uint32_t index) const
    {
        if (index >= cells.size()) { return {}; }
        const auto& cell = cells[index];
        if (uint64_t(cell.offset) + cell.count > indices.size()) { return {}; }
        std::vector<uint32_t> result(indices.begin() + cell.offset,
            indices.begin() + cell.offset + cell.count);
        std::sort(result.begin(), result.end());
        return result;
    }
};

class GridHarness {
public:
    explicit GridHarness(RhiTestContext& context)
        : enableValidation_(context.enableValidation)
    {
    }

    ~GridHarness()
    {
        if (queue_ != nullptr) { (void)queue_->waitIdle(); }
        host_.shutdown();
    }

    RhiTestResult initialize()
    {
        const auto result = render::createDevice({.applicationName = "ClusterLightGrid GPU tests",
            .enableValidation = enableValidation_, .enableBindlessDescriptorHeap = true}, device_);
        if (render::hasError(result, render::Error::Unsupported)) {
            return RhiTestResult::skip("ClusterLightGrid compute program requires bindless descriptors");
        }
        GRID_CHECK(result);
        queue_ = device_->getQueue(render::QueueType::Graphics);
        GRID_CHECK(queue_ != nullptr);
        GRID_CHECK(host_.initialize(*device_, 2, log_));
        GRID_CHECK(tracker_.initialize(*device_, *queue_));
        GRID_CHECK(device_->createCommandPool(*queue_, pool_));
        GRID_CHECK(pool_->createCommandBuffer(commands_));
        for (uint32_t slot = 0; slot < frames_.size(); ++slot) {
            frames_[slot] = std::make_unique<render::RenderFrameContext>(slot);
        }
        return RhiTestResult::pass();
    }

    RhiTestResult record(render::ClusterLightGrid& grid, render::GPUScene& scene,
        render::GPUSceneViewId view, uint32_t slot, const render::ClusterLightGridDesc& desc)
    {
        GRID_CHECK(slot < frames_.size());
        GRID_CHECK(scene.prepareView(view, slot, {.width = desc.width, .height = desc.height}));
        GRID_CHECK(frames_[slot]->begin(nextFrame_++));
        GRID_CHECK(pool_->reset());
        GRID_CHECK(commands_->begin(frames_[slot].get()));
        GRID_CHECK(host_.beginFrame(nextFrame_ - 1, slot, nullptr, log_, frames_[slot].get()));
        const auto result = grid.record(*device_, *commands_, host_, scene, view, slot, desc, log_);
        if (!result) {
            return RhiTestResult::fail("ClusterLightGrid record failed: " + log_ + " (" + toString(result) + ")");
        }
        GRID_CHECK(grid.snapshot(scene) != nullptr);
        GRID_CHECK(grid.snapshot(scene)->valid());
        return RhiTestResult::pass();
    }

    RhiTestResult run(render::ClusterLightGrid& grid, render::GPUScene& scene,
        render::GPUSceneViewId view, uint32_t slot,
        const render::ClusterLightGridDesc& desc, GridReadback& output)
    {
        auto result = record(grid, scene, view, slot, desc);
        if (!result.passed) { return result; }
        output.snapshot = *grid.snapshot(scene);
        const uint64_t cellBytes = output.snapshot.cellCount() * sizeof(render::ClusterLightGridCell);
        const uint64_t indexBytes = output.snapshot.cellCount() * desc.maxLightsPerCell * sizeof(uint32_t);
        if (!lookupProgram_.valid()) {
            render::ShaderCompileResult shader;
            const auto compiled = render::compileSlangShaderToSpirv({.moduleName = "ClusterLightGridLookupProbe",
                .entryPointName = "clusterLightGridLookupProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
            if (!compiled) { return RhiTestResult::fail("ClusterLightGrid lookup probe: " + shader.diagnostics); }
            const std::array<render::ComputeProgramBindingDesc, 6> bindings{{
                {.binding = 0}, {.binding = 1}, {.binding = 2},
                {.binding = 3}, {.binding = 4}, {.binding = 5}}};
            GRID_CHECK(lookupProgram_.initialize(*device_, {.spirv = shader.spirv.data(),
                .byteSize = shader.spirv.size() * sizeof(uint32_t), .bindings = bindings.data(),
                .bindingCount = static_cast<uint32_t>(bindings.size()), .requiresRayQuery = false}, log_));
        }
        std::unique_ptr<render::Buffer> probe;
        GRID_CHECK(device_->createBuffer({.size = sizeof(output.lookup),
            .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
            .memoryLocation = render::MemoryLocation::Device}, probe));
        const render::BufferBarrierDesc probeToWrite{.buffer = probe.get(),
            .before = render::ResourceState::Undefined, .after = render::ResourceState::General};
        commands_->barrier({.buffers = &probeToWrite, .bufferCount = 1});
        const std::array<render::ComputeDispatchBinding, 6> probeBindings{{
            {.binding = 0, .buffer = output.snapshot.parameters},
            {.binding = 1, .buffer = output.snapshot.lights},
            {.binding = 2, .buffer = output.snapshot.candidates},
            {.binding = 3, .buffer = output.snapshot.cells},
            {.binding = 4, .buffer = output.snapshot.lightIndices},
            {.binding = 5, .buffer = probe.get()},
        }};
        GRID_CHECK(lookupProgram_.dispatch({.commandBuffer = commands_.get(),
            .bindings = probeBindings.data(), .bindingCount = static_cast<uint32_t>(probeBindings.size())}));
        std::unique_ptr<render::Buffer> readback;
        GRID_CHECK(device_->createBuffer({.size = cellBytes + indexBytes + sizeof(output.lookup),
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}, readback));
        const std::array barriers{
            render::BufferBarrierDesc{.buffer = output.snapshot.cells,
                .before = render::ResourceState::ShaderRead,
                .after = render::ResourceState::TransferSource},
            render::BufferBarrierDesc{.buffer = output.snapshot.lightIndices,
                .before = render::ResourceState::ShaderRead,
                .after = render::ResourceState::TransferSource},
            render::BufferBarrierDesc{.buffer = probe.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource},
            render::BufferBarrierDesc{.buffer = readback.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::TransferDestination},
        };
        commands_->barrier({.buffers = barriers.data(), .bufferCount = static_cast<uint32_t>(barriers.size())});
        commands_->copyBuffer({.source = output.snapshot.cells, .destination = readback.get(), .size = cellBytes});
        commands_->copyBuffer({.source = output.snapshot.lightIndices, .destination = readback.get(),
            .destinationOffset = cellBytes, .size = indexBytes});
        commands_->copyBuffer({.source = probe.get(), .destination = readback.get(),
            .destinationOffset = cellBytes + indexBytes, .size = sizeof(output.lookup)});
        const std::array restore{
            render::BufferBarrierDesc{.buffer = output.snapshot.cells,
                .before = render::ResourceState::TransferSource, .after = render::ResourceState::ShaderRead},
            render::BufferBarrierDesc{.buffer = output.snapshot.lightIndices,
                .before = render::ResourceState::TransferSource, .after = render::ResourceState::ShaderRead},
        };
        commands_->barrier({.buffers = restore.data(), .bufferCount = static_cast<uint32_t>(restore.size())});
        GRID_CHECK(commands_->end());
        host_.endFrame();
        render::CommandBuffer* submissions[] = {commands_.get()};
        GRID_CHECK(tracker_.submit({.commandBuffers = submissions, .commandBufferCount = 1}, *frames_[slot]));
        GRID_CHECK(frames_[slot]->wait(10'000'000'000ull));
        readback->invalidate();
        const auto* data = static_cast<const uint8_t*>(readback->map());
        GRID_CHECK(data != nullptr);
        output.cells.resize(static_cast<size_t>(output.snapshot.cellCount()));
        output.indices.resize(static_cast<size_t>(indexBytes / sizeof(uint32_t)));
        std::memcpy(output.cells.data(), data, static_cast<size_t>(cellBytes));
        std::memcpy(output.indices.data(), data + cellBytes, static_cast<size_t>(indexBytes));
        std::memcpy(output.lookup.data(), data + cellBytes + indexBytes, sizeof(output.lookup));
        readback->unmap();
        for (size_t cell = 0; cell < output.cells.size(); ++cell) {
            GRID_CHECK(output.cells[cell].offset == cell * desc.maxLightsPerCell);
            GRID_CHECK(output.cells[cell].count <= desc.maxLightsPerCell);
            GRID_CHECK(output.cells[cell].count == std::min(output.cells[cell].totalCount, desc.maxLightsPerCell));
            GRID_CHECK((output.cells[cell].overflow != 0) == (output.cells[cell].totalCount > desc.maxLightsPerCell));
        }
        const auto* visible = scene.visibleLights(view, slot);
        GRID_CHECK(visible != nullptr);
        for (uint32_t query = 0; query < 8; ++query) {
            uint32_t pixelX = query == 5 ? desc.width : 0;
            uint32_t pixelY = query == 6 ? desc.height : 0;
            float depth = desc.zNear;
            if (query == 1) { depth = desc.zFar; }
            if (query == 2) {
                depth = desc.orthoHeight > 0.0f ? (desc.zNear + desc.zFar) * 0.5f
                    : std::sqrt(desc.zNear * desc.zFar);
            }
            if (query == 3) { depth = desc.zNear * 0.5f; }
            if (query == 4) { depth = desc.zFar * 2.0f; }
            if (query == 7) { depth = std::numeric_limits<float>::quiet_NaN(); }
            uint32_t cell = UINT32_MAX;
            const bool inside = render::clusterLightGridCellIndex(output.snapshot.params, pixelX, pixelY, depth, cell);
            const bool fallback = !inside || output.cells[cell].overflow != 0;
            uint32_t expectedMask = 0;
            uint32_t expectedCount = 0;
            if (fallback) {
                for (const auto id : visible->localLights) { expectedMask |= 1u << id.index; }
                expectedCount = static_cast<uint32_t>(visible->localLights.size());
            } else {
                for (const auto id : output.cellLights(cell)) { expectedMask |= 1u << id; }
                expectedCount = output.cells[cell].count;
            }
            GRID_CHECK(output.lookup[query][0] == (fallback ? 1u : 0u));
            GRID_CHECK(output.lookup[query][1] == expectedCount);
            GRID_CHECK(output.lookup[query][2] == expectedMask);
            GRID_CHECK(output.lookup[query][3] == UINT32_MAX);
        }
        uint32_t globalMask = 0;
        float globalIntensity = 0.0f;
        for (const auto& globals : {visible->directionalLights, visible->unboundedLocalLights}) {
            for (const auto id : globals) {
                globalMask |= 1u << id.index;
                globalIntensity += scene.light(id)->source.gpu.colorIntensity[3];
            }
        }
        GRID_CHECK(output.lookup[8][0] == visible->directionalLights.size() + visible->unboundedLocalLights.size());
        GRID_CHECK(output.lookup[8][1] == globalMask);
        GRID_CHECK(output.lookup[8][2] == UINT32_MAX);
        GRID_CHECK(output.lookup[8][3] == static_cast<uint32_t>(std::round(globalIntensity)));
        return RhiTestResult::pass();
    }

    RhiTestResult cancel(uint32_t slot)
    {
        GRID_CHECK(commands_->end());
        host_.endFrame();
        frames_[slot]->cancel();
        return RhiTestResult::pass();
    }

    RhiTestResult acceptUntracked(render::ClusterLightGrid& grid, render::GPUScene& scene,
        render::GPUSceneViewId view, const render::ClusterLightGridDesc& desc)
    {
        GRID_CHECK(scene.prepareView(view, 0, {.width = desc.width, .height = desc.height}));
        GRID_CHECK(pool_->reset());
        GRID_CHECK(commands_->begin());
        GRID_CHECK(grid.record(*device_, *commands_, host_, scene, view, 0, desc, log_));
        GRID_CHECK(grid.snapshot(scene) != nullptr && grid.snapshot(scene)->valid());
        const auto first = *grid.snapshot(scene);
        GRID_CHECK(commands_->end());
        render::CommandBuffer* firstSubmission[] = {commands_.get()};
        GRID_CHECK(queue_->submit({.commandBuffers = firstSubmission, .commandBufferCount = 1}));

        // Recording another legacy command must not overwrite the first one's
        // buffers or descriptor table, even if the first submission is in flight.
        std::unique_ptr<render::CommandBuffer> secondCommands;
        GRID_CHECK(pool_->createCommandBuffer(secondCommands));
        GRID_CHECK(scene.prepareView(view, 0, {.width = desc.width, .height = desc.height}));
        GRID_CHECK(secondCommands->begin());
        GRID_CHECK(grid.record(*device_, *secondCommands, host_, scene, view, 0, desc, log_));
        GRID_CHECK(grid.snapshot(scene) != nullptr && grid.snapshot(scene)->valid());
        GRID_CHECK(grid.snapshot(scene)->cells != first.cells);
        GRID_CHECK(grid.snapshot(scene)->parameters != first.parameters);
        GRID_CHECK(secondCommands->end());
        render::CommandBuffer* secondSubmission[] = {secondCommands.get()};
        GRID_CHECK(queue_->submit({.commandBuffers = secondSubmission, .commandBufferCount = 1}));
        GRID_CHECK(queue_->waitIdle());
        GRID_CHECK(grid.snapshot(scene) != nullptr);
        secondCommands.reset();

        GRID_CHECK(pool_->reset());
        GRID_CHECK(scene.prepareView(view, 0, {.width = desc.width, .height = desc.height}));
        GRID_CHECK(commands_->begin());
        GRID_CHECK(grid.record(*device_, *commands_, host_, scene, view, 0, desc, log_));
        GRID_CHECK(grid.snapshot(scene) != nullptr);
        GRID_CHECK(commands_->end());
        GRID_CHECK(pool_->reset());
        GRID_CHECK(grid.snapshot(scene) == nullptr);
        return RhiTestResult::pass();
    }

    RhiTestResult reload(render::ClusterLightGrid& grid, const render::GPUScene& scene)
    {
        GRID_CHECK(grid.snapshot(scene) != nullptr);
        const auto revision = grid.snapshot(scene)->buildRevision;
        std::unique_ptr<render::RenderSubsystemShaderReload> staged;
        GRID_CHECK(grid.prepareShaderReload(*device_, staged, log_));
        GRID_CHECK(staged != nullptr);
        GRID_CHECK(grid.snapshot(scene) != nullptr && grid.snapshot(scene)->buildRevision == revision);
        staged.reset();
        GRID_CHECK(grid.snapshot(scene) != nullptr);
        GRID_CHECK(grid.prepareShaderReload(*device_, staged, log_));
        GRID_CHECK(staged != nullptr);
        staged->commit();
        GRID_CHECK(grid.snapshot(scene) == nullptr);
        return RhiTestResult::pass();
    }

private:
    std::unique_ptr<render::Device> device_;
    render::Queue* queue_ = nullptr;
    bool enableValidation_ = false;
    render::RenderSubsystemHost host_;
    render::QueueSubmissionTracker tracker_;
    render::ComputeProgram lookupProgram_;
    std::unique_ptr<render::CommandPool> pool_;
    std::unique_ptr<render::CommandBuffer> commands_;
    std::array<std::unique_ptr<render::RenderFrameContext>, 2> frames_;
    std::string log_;
    uint64_t nextFrame_ = 1;
};

class ClusterLightGridLayoutTest final : public RhiTest {
public:
    ClusterLightGridLayoutTest()
    {
        type = RhiTestType::Validation;
        name = "cluster_light_grid_layout";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        auto desc = gridDesc();
        desc.width = 129;
        desc.height = 65;
        desc.aspect = 2.0f;
        render::ClusterLightGridParams params;
        std::string log;
        GRID_CHECK(render::buildClusterLightGridParams(desc, params, log));
        GRID_CHECK((params.grid == std::array<uint32_t, 4>{3, 2, 4, 64}));
        GRID_CHECK((params.viewport == std::array<uint32_t, 4>{129, 65, 8, 0}));
        GRID_CHECK(std::abs(params.upExtent[3] - 1.0f) < 0.00001f);
        GRID_CHECK(std::abs(params.forwardExtent[3] - 2.0f) < 0.00001f);
        for (uint32_t slice = 0; slice <= 4; ++slice) {
            GRID_CHECK(std::abs(render::clusterLightGridSliceDepth(params, slice) - std::exp2(float(slice))) < 0.0001f);
        }
        uint32_t cell = UINT32_MAX;
        GRID_CHECK(render::clusterLightGridCellIndex(params, 0, 0, 1.0f, cell) && cell == 0);
        GRID_CHECK(render::clusterLightGridCellIndex(params, 128, 64, 16.0f, cell) && cell == 23);
        for (uint32_t slice = 0; slice < 4; ++slice) {
            GRID_CHECK(render::clusterLightGridCellIndex(params, 64, 0, std::exp2(float(slice) + 0.5f), cell));
            GRID_CHECK(cell == slice * 6 + 1);
        }
        GRID_CHECK(!render::clusterLightGridCellIndex(params, 129, 0, 4.0f, cell));
        GRID_CHECK(!render::clusterLightGridCellIndex(params, 0, 65, 4.0f, cell));
        GRID_CHECK(!render::clusterLightGridCellIndex(params, 0, 0, 0.99f, cell));
        GRID_CHECK(!render::clusterLightGridCellIndex(params, 0, 0, 16.01f, cell));
        GRID_CHECK(!render::clusterLightGridCellIndex(params, 0, 0, std::numeric_limits<float>::quiet_NaN(), cell));
        GRID_CHECK(!render::clusterLightGridCellIndex(params, 0, 0, std::numeric_limits<float>::infinity(), cell));

        desc.orthoHeight = 4.0f;
        desc.eye = float3(1.0f, 2.0f, 3.0f);
        desc.center = float3(2.0f, 2.0f, 3.0f);
        GRID_CHECK(render::buildClusterLightGridParams(desc, params, log));
        GRID_CHECK(params.viewport[3] == 1);
        GRID_CHECK(params.upExtent[3] == 2.0f && params.forwardExtent[3] == 4.0f);
        GRID_CHECK(params.forwardExtent[0] == 1.0f && params.rightFar[2] == 1.0f);
        for (uint32_t slice = 0; slice <= 4; ++slice) {
            GRID_CHECK(std::abs(render::clusterLightGridSliceDepth(params, slice) - (1.0f + 3.75f * slice)) < 0.0001f);
        }
        for (uint32_t slice = 0; slice < 4; ++slice) {
            GRID_CHECK(render::clusterLightGridCellIndex(params, 0, 64, 1.0f + 3.75f * (slice + 0.5f), cell));
            GRID_CHECK(cell == slice * 6 + 3);
        }
        const auto validDesc = desc;
        auto invalid = [&](auto edit) {
            auto candidate = validDesc;
            edit(candidate);
            return !render::buildClusterLightGridParams(candidate, params, log);
        };
        GRID_CHECK(invalid([](auto& d) { d.width = 0; }));
        GRID_CHECK(invalid([](auto& d) { d.height = 0; }));
        GRID_CHECK(invalid([](auto& d) { d.tileSize = 0; }));
        GRID_CHECK(invalid([](auto& d) { d.depthSliceCount = 0; }));
        GRID_CHECK(invalid([](auto& d) { d.maxLightsPerCell = 0; }));
        GRID_CHECK(invalid([](auto& d) { d.zNear = 0.0f; }));
        GRID_CHECK(invalid([](auto& d) { d.zFar = d.zNear; }));
        GRID_CHECK(invalid([](auto& d) { d.aspect = std::numeric_limits<float>::quiet_NaN(); }));
        GRID_CHECK(invalid([](auto& d) { d.center = d.eye; }));
        GRID_CHECK(invalid([](auto& d) { d.up = d.center - d.eye; }));
        GRID_CHECK(invalid([](auto& d) { d.width = UINT32_MAX; d.height = UINT32_MAX; d.tileSize = 1; }));
        return RhiTestResult::pass("partial tiles, perspective logarithmic/orthographic linear Z, lookup fallback and invalid limits");
    }
};

class ClusterLightGridGpuCullTest final : public RhiTest {
public:
    ClusterLightGridGpuCullTest()
    {
        type = RhiTestType::Command;
        name = "cluster_light_grid_gpu_culling";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        GridHarness harness(context);
        auto result = harness.initialize();
        if (!result.passed) { return result; }
        render::GPUScene scene;
        scene.setDefaultFrameSlotCount(2);
        const auto view = scene.createView();
        render::ClusterLightGrid grid;
        auto desc = gridDesc();
        std::vector<scene::PunctualLight> lights{
            makeGridLight("point", float3(0.0f, 0.0f, -6.0f), 100.0),
            makeGridLight("directional", float3(0.0f), 0.0),
            makeGridLight("point", float3(1000.0f), 0.0),
            makeGridLight("point", float3(-3.0f, 3.0f, -6.0f), 0.2),
            makeGridLight("point", float3(3.0f, -3.0f, -6.0f), 0.2),
            makeGridLight("point", float3(-3.0f, 3.0f, -12.0f), 0.2),
        };
        lights[0].enabled = false;
        GRID_CHECK(scene.syncLights({}, lights));
        GridReadback data;
        result = harness.run(grid, scene, view, 0, desc, data);
        if (!result.passed) { return result; }
        GRID_CHECK((data.snapshot.params.counts == std::array<uint32_t, 4>{3, 1, 1, 6}));
        GRID_CHECK(data.cells.size() == 16);
        for (uint32_t index = 0; index < 16; ++index) {
            const std::vector<uint32_t> expected = index == 8 ? std::vector<uint32_t>{3}
                : index == 11 ? std::vector<uint32_t>{4}
                : index == 12 ? std::vector<uint32_t>{5} : std::vector<uint32_t>{};
            GRID_CHECK(data.cellLights(index) == expected);
            GRID_CHECK(data.cells[index].overflow == 0);
        }

        // Both spots' range spheres overlap the right/top cell. Only the +X
        // cone reaches it; the narrow +Y cone must fail the cone/AABB plane test.
        lights = {
            makeGridLight("point", float3(0.0f), 1.0),
            makeGridLight("spot", float3(-1.0f, 2.0f, -7.0f), 4.0),
            makeGridLight("spot", float3(-1.0f, 2.0f, -7.0f), 4.0),
            makeGridLight("point", float3(1.0f, 2.0f, -7.0f), 0.2),
        };
        lights[0].enabled = false;
        lights[1].direction = float3(1.0f, 0.0f, 0.0f);
        lights[2].direction = float3(0.0f, 1.0f, 0.0f);
        for (size_t index : {size_t(1), size_t(2)}) {
            lights[index].properties.innerConeAngle = 0.0;
            lights[index].properties.outerConeAngle = 0.1;
        }
        GRID_CHECK(scene.syncLights({}, lights));
        GRID_CHECK(grid.snapshot(scene) == nullptr);
        desc.orthoHeight = 8.0f;
        desc.zFar = 17.0f;
        result = harness.run(grid, scene, view, 0, desc, data);
        if (!result.passed) { return result; }
        GRID_CHECK((data.snapshot.params.counts == std::array<uint32_t, 4>{3, 0, 0, 4}));
        GRID_CHECK(data.cellLights(5) == std::vector<uint32_t>({1, 3}));
        GRID_CHECK(scene.prepareView(view, 0, {.width = desc.width, .height = desc.height}));
        GRID_CHECK(grid.snapshot(scene) == nullptr);
        return RhiTestResult::pass("GPU point/AABB and spot-cone intersections, source-index holes, global lights and camera invalidation");
    }
};

class ClusterLightGridGpuLifecycleTest final : public RhiTest {
public:
    ClusterLightGridGpuLifecycleTest()
    {
        type = RhiTestType::Command;
        name = "cluster_light_grid_gpu_overflow_lifecycle";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        GridHarness harness(context);
        auto result = harness.initialize();
        if (!result.passed) { return result; }
        render::GPUScene scene;
        scene.setDefaultFrameSlotCount(2);
        const auto view = scene.createView();
        const auto secondView = scene.createView();
        render::ClusterLightGrid first, otherSlot, otherView;
        auto desc = gridDesc();
        desc.width = desc.height = 64;
        desc.depthSliceCount = 1;
        desc.maxLightsPerCell = 2;
        desc.zFar = 9.0f;
        desc.orthoHeight = 8.0f;
        std::vector<scene::PunctualLight> lights(5,
            makeGridLight("point", float3(0.0f, 0.0f, -5.0f), 1.0));
        lights.push_back(makeGridLight("directional", float3(0.0f), 0.0));
        lights.push_back(makeGridLight("point", float3(1000.0f), 0.0));
        GRID_CHECK(scene.syncLights({}, lights));
        result = harness.acceptUntracked(first, scene, view, desc);
        if (!result.passed) { return result; }
        GridReadback firstData, slotData, viewData;
        result = harness.run(first, scene, view, 0, desc, firstData);
        if (!result.passed) { return result; }
        GRID_CHECK((firstData.snapshot.params.counts == std::array<uint32_t, 4>{5, 1, 1, 7}));
        GRID_CHECK(firstData.cells[0].count == 2 && firstData.cells[0].totalCount == 5 && firstData.cells[0].overflow == 1);
        const auto stored = firstData.cellLights(0);
        GRID_CHECK(stored.size() == 2 && stored[0] < stored[1] && stored[1] < 5);
        result = harness.run(otherSlot, scene, view, 1, desc, slotData);
        if (!result.passed) { return result; }
        result = harness.run(otherView, scene, secondView, 0, desc, viewData);
        if (!result.passed) { return result; }
        GRID_CHECK(first.snapshot(scene) != nullptr && otherSlot.snapshot(scene) != nullptr && otherView.snapshot(scene) != nullptr);
        GRID_CHECK(firstData.snapshot.cells != slotData.snapshot.cells && firstData.snapshot.cells != viewData.snapshot.cells);
        GRID_CHECK(slotData.snapshot.cells != viewData.snapshot.cells);
        GRID_CHECK(slotData.snapshot.sourceView == view && slotData.snapshot.frameSlot == 1);
        GRID_CHECK(viewData.snapshot.sourceView == secondView && viewData.snapshot.frameSlot == 0);

        auto largerDesc = desc;
        largerDesc.width = 128;
        result = harness.run(otherSlot, scene, view, 1, largerDesc, slotData);
        if (!result.passed) { return result; }
        GRID_CHECK(slotData.cells.size() == 2);
        const auto* largerCells = slotData.snapshot.cells;
        result = harness.run(otherSlot, scene, view, 1, desc, slotData);
        if (!result.passed) { return result; }
        GRID_CHECK(slotData.cells.size() == 1 && slotData.snapshot.cells == largerCells);
        GRID_CHECK(slotData.cells[0].count == 2 && slotData.cells[0].totalCount == 5);

        const auto* cellsBeforeReload = firstData.snapshot.cells;
        result = harness.reload(first, scene);
        if (!result.passed) { return result; }
        result = harness.run(first, scene, view, 0, desc, firstData);
        if (!result.passed) { return result; }
        GRID_CHECK(firstData.snapshot.cells == cellsBeforeReload);
        GRID_CHECK(firstData.cells[0].totalCount == 5);

        const auto previousRevision = firstData.snapshot.buildRevision;
        result = harness.record(first, scene, view, 0, desc);
        if (!result.passed) { return result; }
        GRID_CHECK(first.snapshot(scene)->buildRevision > previousRevision);
        result = harness.cancel(0);
        if (!result.passed) { return result; }
        GRID_CHECK(first.snapshot(scene) == nullptr);
        GRID_CHECK(otherSlot.snapshot(scene) != nullptr && otherView.snapshot(scene) != nullptr);
        result = harness.run(first, scene, view, 0, desc, firstData);
        if (!result.passed) { return result; }
        GRID_CHECK(firstData.cells[0].totalCount == 5);

        GRID_CHECK(scene.syncLights({}, {}));
        GRID_CHECK(first.snapshot(scene) == nullptr && otherSlot.snapshot(scene) == nullptr && otherView.snapshot(scene) == nullptr);
        result = harness.run(first, scene, view, 0, desc, firstData);
        if (!result.passed) { return result; }
        GRID_CHECK((firstData.snapshot.params.counts == std::array<uint32_t, 4>{0, 0, 0, 0}));
        GRID_CHECK(firstData.cells[0].count == 0 && firstData.cells[0].totalCount == 0 && firstData.cells[0].overflow == 0);
        GRID_CHECK(scene.destroyView(view));
        GRID_CHECK(first.snapshot(scene) == nullptr);
        first.clear();
        GRID_CHECK(first.snapshot(scene) == nullptr);
        return RhiTestResult::pass("GPU overflow flag/full-list fallback contract, empty reset, per-view/slot isolation and cancellation recovery");
    }
};

METALLIC_REGISTER_RHI_TEST(ClusterLightGridLayoutTest);
METALLIC_REGISTER_RHI_TEST(ClusterLightGridGpuCullTest);
METALLIC_REGISTER_RHI_TEST(ClusterLightGridGpuLifecycleTest);

#undef GRID_CHECK

} // namespace
} // namespace metallic::tests
