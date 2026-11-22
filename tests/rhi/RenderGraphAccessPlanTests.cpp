#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/RenderGraph/RenderGraphResourceLifecycle.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>

namespace metallic::tests {
namespace {

using render::AccessBits;
using render::PipelineStageBits;
using render::ResourceState;
using render::SyncScope;
using render::detail::GraphAccessBarrier;
using render::detail::GraphAccessPass;
using render::detail::GraphAccessPlan;
using render::detail::GraphAccessResource;
using render::detail::GraphAccessUse;
using render::detail::buildGraphAccessPlan;

#define ACCESS_CHECK(condition) do { \
    if (!(condition)) { return RHITestResult::fail(#condition); } \
} while (false)
#define ACCESS_REQUIRE(expression) do { \
    const auto checked = (expression); \
    if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + render::resultToString(checked)); } \
} while (false)

constexpr SyncScope kComputeRead{PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
constexpr SyncScope kComputeWrite{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite};
constexpr SyncScope kFragmentRead{PipelineStageBits::FragmentShader, AccessBits::ShaderRead};
constexpr SyncScope kTransferRead{PipelineStageBits::Transfer, AccessBits::TransferRead};

template<typename T>
bool containsBits(T value, T bits)
{
    return (static_cast<uint64_t>(value) & static_cast<uint64_t>(bits)) == static_cast<uint64_t>(bits);
}

GraphAccessUse bufferUse(SyncScope scope, bool writes = false, size_t resource = 0)
{
    return {.resource = resource, .state = ResourceState::General, .scope = scope, .writes = writes};
}

bool hasPredecessor(const GraphAccessPlan& plan, size_t destination, size_t source)
{
    const auto& predecessors = plan.passes[destination].predecessors;
    return std::find(predecessors.begin(), predecessors.end(), source) != predecessors.end();
}

bool hasVisibilityBarrier(const GraphAccessPlan& plan, size_t first, size_t last,
    PipelineStageBits source, AccessBits written, PipelineStageBits destination)
{
    for (size_t pass = first; pass <= last; ++pass) {
        for (const GraphAccessBarrier& barrier : plan.passes[pass].barriers) {
            if (containsBits(barrier.beforeScope.stages, source) &&
                containsBits(barrier.beforeScope.access, written) &&
                containsBits(barrier.afterScope.stages, destination)) {
                return true;
            }
        }
    }
    return false;
}

class AccessPlanAccelerationStructureTest final : public RHITest {
public:
    AccessPlanAccelerationStructureTest() { name = "render_graph_access_plan_acceleration_structure"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {"graph.plan.contract.acceleration_structure"}};
    }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        using Access = render::RenderGraphResourceAccess;
        using Kind = render::RenderGraphPassKind;
        using Type = render::RenderGraphResourceType;
        using render::detail::declaredGraphAccess;
        const std::array resources{
            GraphAccessResource{.type = Type::Buffer},
            GraphAccessResource{.type = Type::AccelerationStructure},
        };
        const std::array passes{
            GraphAccessPass{.queue = 1, .uses = {declaredGraphAccess(0, Access::BufferStorageWrite, Kind::Compute)}},
            GraphAccessPass{.queue = 1, .uses = {
                declaredGraphAccess(0, Access::BufferAccelerationStructureBuildRead, Kind::Compute),
                declaredGraphAccess(1, Access::AccelerationStructureBuildWrite, Kind::Compute)}},
            GraphAccessPass{.queue = 1, .uses = {declaredGraphAccess(1, Access::AccelerationStructureShaderRead, Kind::Compute)}},
            GraphAccessPass{.queue = 0, .uses = {declaredGraphAccess(1, Access::AccelerationStructureShaderRead, Kind::Raster)}},
            GraphAccessPass{.queue = 1, .uses = {
                declaredGraphAccess(1, Access::AccelerationStructureBuildRead, Kind::Compute),
                declaredGraphAccess(1, Access::AccelerationStructureBuildWrite, Kind::Compute)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan && plan->passes.size() == passes.size());
        ACCESS_CHECK(hasPredecessor(*plan, 1, 0));
        ACCESS_CHECK(plan->passes[1].barriers.size() == 1);
        const auto& inputBarrier = plan->passes[1].barriers.front();
        ACCESS_CHECK(inputBarrier.resource == 0 && !inputBarrier.executionOnly);
        ACCESS_CHECK(inputBarrier.beforeScope.stages == PipelineStageBits::ComputeShader);
        ACCESS_CHECK(inputBarrier.afterScope.stages == PipelineStageBits::AccelerationStructureBuild);
        ACCESS_CHECK(inputBarrier.afterScope.access == AccessBits::ShaderRead);
        ACCESS_CHECK(hasVisibilityBarrier(*plan, 2, 2, PipelineStageBits::AccelerationStructureBuild,
            AccessBits::AccelerationStructureWrite, PipelineStageBits::ComputeShader));
        // AS has no image layout. Cross-queue visibility comes from the graph
        // semaphore dependency, with no source-stage barrier on graphics.
        ACCESS_CHECK(hasPredecessor(*plan, 3, 1) && !hasPredecessor(*plan, 3, 2));
        ACCESS_CHECK(plan->passes[3].barriers.empty());
        ACCESS_CHECK(hasPredecessor(*plan, 4, 2) && hasPredecessor(*plan, 4, 3));
        ACCESS_CHECK(plan->passes[4].uses.size() == 1 && plan->passes[4].uses.front().writes);
        ACCESS_CHECK(containsBits(plan->passes[4].uses.front().scope.access,
            AccessBits::AccelerationStructureRead | AccessBits::AccelerationStructureWrite));
        for (const auto& barrier : plan->passes[4].barriers) {
            ACCESS_CHECK(!containsBits(barrier.beforeScope.stages, PipelineStageBits::FragmentShader));
        }
        render::RenderPassReflection reflection;
        auto& input = reflection.addAccelerationStructureInput("source");
        ACCESS_CHECK(input.resourceType == Type::AccelerationStructure && input.access == Access::AccelerationStructureShaderRead);
        auto& output = reflection.addAccelerationStructureOutput("result");
        ACCESS_CHECK(output.resourceType == Type::AccelerationStructure && output.access == Access::AccelerationStructureBuildWrite);
        return RHITestResult::pass();
    }
};

class AccessPlanWriterVisibilityTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanWriterVisibilityTest() { name = "render_graph_access_plan_writer_stage_visibility"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        const std::array passes{
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kFragmentRead)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan && plan->passes.size() == passes.size());
        ACCESS_CHECK(hasPredecessor(*plan, 1, 0) && hasPredecessor(*plan, 2, 0));
        // A compute read must not erase the write that a later fragment read needs.
        // Permit a future optimizer to expose both readers in the first barrier.
        ACCESS_CHECK(hasVisibilityBarrier(*plan, 1, 2, PipelineStageBits::ComputeShader,
            AccessBits::ShaderWrite, PipelineStageBits::FragmentShader));
        ACCESS_CHECK(hasVisibilityBarrier(*plan, 1, 1, PipelineStageBits::ComputeShader,
            AccessBits::ShaderWrite, PipelineStageBits::ComputeShader));
        return RHITestResult::pass();
    }
};

class AccessPlanQueueFanoutTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanQueueFanoutTest() { name = "render_graph_access_plan_queue_fanout_and_join"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        const std::array passes{
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.queue = 1, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 2, .uses = {bufferUse(kFragmentRead)}},
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeWrite, true)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan);
        ACCESS_CHECK(hasPredecessor(*plan, 1, 0) && hasPredecessor(*plan, 2, 0));
        ACCESS_CHECK(!hasPredecessor(*plan, 2, 1));
        ACCESS_CHECK(hasPredecessor(*plan, 3, 1) && hasPredecessor(*plan, 3, 2));
        // Semaphore waits provide the remote memory dependency. Unsupported source
        // stages from another queue must never leak into this queue's barrier.
        for (const GraphAccessBarrier& barrier : plan->passes[3].barriers) {
            ACCESS_CHECK(!containsBits(barrier.beforeScope.stages, PipelineStageBits::FragmentShader));
        }
        ACCESS_CHECK(plan->passes[1].barriers.empty() && plan->passes[2].barriers.empty());
        return RHITestResult::pass();
    }
};

class AccessPlanWarTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanWarTest() { name = "render_graph_access_plan_read_frontier_execution_dependency"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        const std::array passes{
            GraphAccessPass{.uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.uses = {bufferUse(kFragmentRead)}},
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan);
        ACCESS_CHECK(plan->passes[0].barriers.empty() && plan->passes[1].barriers.empty());
        ACCESS_CHECK(!hasPredecessor(*plan, 1, 0));
        ACCESS_CHECK(hasPredecessor(*plan, 2, 0) && hasPredecessor(*plan, 2, 1));
        bool coveredCompute = false, coveredFragment = false;
        for (const GraphAccessBarrier& barrier : plan->passes[2].barriers) {
            ACCESS_CHECK(barrier.executionOnly);
            ACCESS_CHECK(barrier.beforeScope.access == AccessBits::None && barrier.afterScope.access == AccessBits::None);
            coveredCompute |= containsBits(barrier.beforeScope.stages, PipelineStageBits::ComputeShader);
            coveredFragment |= containsBits(barrier.beforeScope.stages, PipelineStageBits::FragmentShader);
        }
        ACCESS_CHECK(coveredCompute && coveredFragment);
        return RHITestResult::pass();
    }
};

class AccessPlanImageLayoutTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanImageLayoutTest() { name = "render_graph_access_plan_image_layout_frontier"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array undefinedImage{GraphAccessResource{.type = render::RenderGraphResourceType::Texture2D}};
        const std::array firstReads{
            GraphAccessPass{.queue = 0, .uses = {{.resource = 0, .state = ResourceState::ShaderRead, .scope = kComputeRead}}},
            GraphAccessPass{.queue = 1, .uses = {{.resource = 0, .state = ResourceState::ShaderRead, .scope = kFragmentRead}}},
            GraphAccessPass{.queue = 0, .uses = {{.resource = 0, .state = ResourceState::ShaderRead, .scope = kFragmentRead}}},
        };
        auto initial = buildGraphAccessPlan(undefinedImage, firstReads);
        ACCESS_CHECK(initial && initial->passes[0].barriers.size() == 1);
        ACCESS_CHECK(initial->passes[0].barriers[0].before == ResourceState::Undefined);
        ACCESS_CHECK(initial->passes[0].barriers[0].after == ResourceState::ShaderRead);
        ACCESS_CHECK(initial->passes[0].barriers[0].beforeScope.stages == PipelineStageBits::None);
        ACCESS_CHECK(hasPredecessor(*initial, 1, 0) && hasPredecessor(*initial, 2, 0));
        // The first image transition is itself a write; another local reader
        // stage needs a GPU memory dependency, not only a CPU submission edge.
        ACCESS_CHECK(hasVisibilityBarrier(*initial, 1, 2, PipelineStageBits::ComputeShader,
            AccessBits::MemoryWrite, PipelineStageBits::FragmentShader));

        using Access = render::RenderGraphResourceAccess;
        using Kind = render::RenderGraphPassKind;
        const std::array remotePasses{
            GraphAccessPass{.queue = 0, .uses = {render::detail::declaredGraphAccess(0, Access::TextureColorWrite, Kind::Raster)}},
            GraphAccessPass{.queue = 1, .uses = {render::detail::declaredGraphAccess(0, Access::TextureSampleRead, Kind::Compute)}},
        };
        auto remote = buildGraphAccessPlan(undefinedImage, remotePasses);
        ACCESS_CHECK(remote && hasPredecessor(*remote, 1, 0));
        ACCESS_CHECK(remote->passes[1].barriers.size() == 1);
        const auto& acquire = remote->passes[1].barriers.front();
        ACCESS_CHECK(acquire.beforeScope.stages == PipelineStageBits::None && acquire.beforeScope.access == AccessBits::None);
        ACCESS_CHECK(acquire.afterScope.stages == PipelineStageBits::ComputeShader && acquire.afterScope.access == AccessBits::ShaderRead);

        const std::array image{GraphAccessResource{
            .type = render::RenderGraphResourceType::Texture2D, .state = ResourceState::ShaderRead, .scope = kComputeRead}};
        const std::array layoutChange{
            GraphAccessPass{.queue = 0, .uses = {{.resource = 0, .state = ResourceState::ShaderRead, .scope = kComputeRead}}},
            GraphAccessPass{.queue = 1, .uses = {{.resource = 0, .state = ResourceState::ShaderRead, .scope = kFragmentRead}}},
            GraphAccessPass{.queue = 2, .uses = {{.resource = 0, .state = ResourceState::TransferSource, .scope = kTransferRead}}},
            GraphAccessPass{.queue = 0, .uses = {{.resource = 0, .state = ResourceState::TransferSource, .scope = kTransferRead}}},
        };
        auto changed = buildGraphAccessPlan(image, layoutChange);
        ACCESS_CHECK(changed);
        ACCESS_CHECK(!hasPredecessor(*changed, 1, 0));
        ACCESS_CHECK(hasPredecessor(*changed, 2, 0) && hasPredecessor(*changed, 2, 1));
        ACCESS_CHECK(hasPredecessor(*changed, 3, 2));
        ACCESS_CHECK(std::any_of(changed->passes[2].barriers.begin(), changed->passes[2].barriers.end(),
            [](const GraphAccessBarrier& barrier) {
                return barrier.before == ResourceState::ShaderRead && barrier.after == ResourceState::TransferSource &&
                    !barrier.executionOnly;
            }));
        return RHITestResult::pass();
    }
};

class AccessPlanAliasesTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanAliasesTest() { name = "render_graph_access_plan_alias_merge_and_rejection"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array buffers{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        const std::array aliases{GraphAccessPass{.uses = {
            {.resource = 0, .state = ResourceState::ShaderRead, .scope = kComputeRead},
            bufferUse(kComputeWrite, true),
        }}};
        auto merged = buildGraphAccessPlan(buffers, aliases);
        ACCESS_CHECK(merged && merged->passes[0].uses.size() == 1);
        const auto& use = merged->passes[0].uses[0];
        ACCESS_CHECK(use.writes && use.state == ResourceState::General);
        ACCESS_CHECK(containsBits(use.scope.access, AccessBits::ShaderRead | AccessBits::ShaderWrite));
        const std::array images{GraphAccessResource{.type = render::RenderGraphResourceType::Texture2D}};
        auto incompatible = buildGraphAccessPlan(images, aliases);
        ACCESS_CHECK(render::hasError(incompatible, render::Error::InvalidArgument));
        const std::array incompatibleBuffer{GraphAccessPass{.uses = {
            {.resource = 0, .state = ResourceState::TransferSource, .scope = kTransferRead},
            bufferUse(kComputeWrite, true),
        }}};
        ACCESS_CHECK(render::hasError(buildGraphAccessPlan(buffers, incompatibleBuffer), render::Error::InvalidArgument));
        ACCESS_CHECK(aliases[0].uses.size() == 2 && aliases[0].uses[0].state == ResourceState::ShaderRead);
        return RHITestResult::pass();
    }
};

class AccessPlanInvalidInputTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanInvalidInputTest() { name = "render_graph_access_plan_invalid_resource_is_transactional"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        const std::array passes{
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.uses = {bufferUse(kComputeRead, false, 1)}},
        };
        auto invalid = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(render::hasError(invalid, render::Error::InvalidArgument));
        ACCESS_CHECK(resources[0].state == ResourceState::Undefined && resources[0].scope.stages == PipelineStageBits::None);
        ACCESS_CHECK(passes[0].uses[0].writes && passes[1].uses[0].resource == 1);
        const std::array missingScope{GraphAccessPass{.uses = {bufferUse({})}}};
        ACCESS_CHECK(render::hasError(buildGraphAccessPlan(resources, missingScope), render::Error::InvalidArgument));
        auto empty = buildGraphAccessPlan({}, {});
        ACCESS_CHECK(empty && empty->passes.empty());
        return RHITestResult::pass();
    }
};

class AccessPlanFrameBoundaryTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanFrameBoundaryTest() { name = "render_graph_access_plan_initial_scope_and_access_types"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{
            .type = render::RenderGraphResourceType::Buffer, .state = ResourceState::General, .scope = kComputeWrite}};
        const std::array passes{
            GraphAccessPass{.uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.uses = {bufferUse(kFragmentRead)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan && plan->passes[0].predecessors.empty());
        ACCESS_CHECK(hasVisibilityBarrier(*plan, 0, 0, PipelineStageBits::ComputeShader,
            AccessBits::ShaderWrite, PipelineStageBits::ComputeShader));
        ACCESS_CHECK(hasVisibilityBarrier(*plan, 0, 1, PipelineStageBits::ComputeShader,
            AccessBits::ShaderWrite, PipelineStageBits::FragmentShader));
        using render::RenderGraphResourceAccess;
        using render::RenderGraphPassKind;
        const auto read = render::detail::scopeForGraphAccess(RenderGraphResourceAccess::BufferStorageRead, RenderGraphPassKind::Compute);
        const auto write = render::detail::scopeForGraphAccess(RenderGraphResourceAccess::TextureStorageWrite, RenderGraphPassKind::Compute);
        const auto constant = render::detail::scopeForGraphAccess(RenderGraphResourceAccess::BufferConstantRead, RenderGraphPassKind::Raster);
        ACCESS_CHECK(read.stages == PipelineStageBits::ComputeShader && read.access == AccessBits::ShaderRead);
        ACCESS_CHECK(write.access == AccessBits::ShaderWrite);
        ACCESS_CHECK(constant.access == AccessBits::UniformRead && containsBits(constant.stages, PipelineStageBits::FragmentShader));
        const auto traceRead = render::detail::scopeForGraphAccess(
            RenderGraphResourceAccess::AccelerationStructureShaderRead, RenderGraphPassKind::RayTracing);
        const auto traceWrite = render::detail::scopeForGraphAccess(
            RenderGraphResourceAccess::TextureStorageWrite, RenderGraphPassKind::RayTracing);
        ACCESS_CHECK(traceRead.stages == PipelineStageBits::RayTracingShader && traceRead.access == AccessBits::AccelerationStructureRead);
        ACCESS_CHECK(traceWrite.stages == PipelineStageBits::RayTracingShader && traceWrite.access == AccessBits::ShaderWrite);
        const auto opaque = render::detail::scopeForGraphAccess(
            RenderGraphResourceAccess::BufferStorageReadWrite, RenderGraphPassKind::Unsafe);
        ACCESS_CHECK(opaque.stages == PipelineStageBits::AllCommands);
        ACCESS_CHECK(containsBits(opaque.access, AccessBits::MemoryRead | AccessBits::MemoryWrite |
            AccessBits::ShaderRead | AccessBits::ShaderWrite));
        return RHITestResult::pass();
    }
};

class AccessPlanInternalBoundaryTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanInternalBoundaryTest() { name = "render_graph_access_plan_internal_boundary_and_stage_hazard"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{
            GraphAccessResource{.type = render::RenderGraphResourceType::Buffer,
                .state = ResourceState::General, .scope = kComputeWrite, .boundarySynchronized = true},
            GraphAccessResource{.type = render::RenderGraphResourceType::Texture2D,
                .state = ResourceState::ShaderRead, .scope = kComputeWrite, .boundarySynchronized = true},
            // Private history remains an ordinary import. Its preceding frame
            // access must still be synchronized on the first internal use.
            GraphAccessResource{.type = render::RenderGraphResourceType::Buffer,
                .state = ResourceState::General, .scope = kComputeWrite},
        };
        const std::array passes{
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true),
                {.resource = 1, .state = ResourceState::ShaderRead, .scope = kComputeRead}}},
            GraphAccessPass{.uses = {bufferUse(kComputeRead), bufferUse(kComputeWrite, true, 2)}},
            GraphAccessPass{.uses = {
                {.resource = 1, .state = ResourceState::General, .scope = kComputeWrite, .writes = true}}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan && plan->passes[0].barriers.empty());
        ACCESS_CHECK(hasPredecessor(*plan, 1, 0));
        ACCESS_CHECK(plan->passes[1].barriers.size() == 2);
        ACCESS_CHECK(hasVisibilityBarrier(*plan, 1, 1, PipelineStageBits::ComputeShader,
            AccessBits::ShaderWrite, PipelineStageBits::ComputeShader));
        ACCESS_CHECK(std::any_of(plan->passes[1].barriers.begin(), plan->passes[1].barriers.end(),
            [](const GraphAccessBarrier& barrier) { return barrier.resource == 2; }));

        const auto internal = buildGraphAccessPlan(resources, passes, render::detail::GraphAccessPlanMode::InternalStages);
        ACCESS_CHECK(internal && internal->passes.size() == plan->passes.size());
        const auto useFields = [](const GraphAccessUse& use) {
            return std::tuple{use.resource, use.state, use.scope.stages, use.scope.access, use.writes};
        };
        const auto barrierFields = [](const GraphAccessBarrier& barrier) {
            return std::tuple{barrier.resource, barrier.before, barrier.after,
                barrier.beforeScope.stages, barrier.beforeScope.access, barrier.afterScope.stages, barrier.afterScope.access,
                barrier.executionOnly, barrier.memoryAliasing};
        };
        for (size_t index = 0; index < plan->passes.size(); ++index) {
            const auto& expected = plan->passes[index];
            const auto& stage = internal->passes[index];
            ACCESS_CHECK(stage.predecessors.empty());
            ACCESS_CHECK(std::ranges::equal(stage.uses, expected.uses, {}, useFields, useFields));
            ACCESS_CHECK(std::ranges::equal(stage.barriers, expected.barriers, {}, barrierFields, barrierFields));
        }

        const std::array changeLayout{GraphAccessPass{.uses = {
            {.resource = 1, .state = ResourceState::General, .scope = kComputeWrite, .writes = true}}}};
        auto changed = buildGraphAccessPlan(resources, changeLayout);
        ACCESS_CHECK(changed && changed->passes[0].barriers.size() == 1);
        const auto& barrier = changed->passes[0].barriers.front();
        ACCESS_CHECK(barrier.before == ResourceState::ShaderRead && barrier.after == ResourceState::General);
        ACCESS_CHECK(!barrier.executionOnly && barrier.beforeScope.access == AccessBits::None);
        return RHITestResult::pass();
    }
};

class AccessPlanRawVisibilityReuseTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanRawVisibilityReuseTest() { name = "render_graph_access_plan_raw_visibility_reuse"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        std::array<GraphAccessPass, 7> passes{};
        passes[0].uses = {bufferUse(kComputeWrite, true)};
        for (size_t reader = 1; reader < passes.size(); ++reader) {
            passes[reader].uses = {bufferUse(kComputeRead)};
        }
        // Coverage belongs to this plan: rebuilding for another frame must
        // establish visibility again, even with identical declarations.
        for (uint32_t rebuild = 0; rebuild < 2; ++rebuild) {
            auto plan = buildGraphAccessPlan(resources, passes);
            ACCESS_CHECK(plan && plan->passes[0].barriers.empty());
            size_t barrierCount = 0;
            for (size_t reader = 1; reader < passes.size(); ++reader) {
                const auto& planned = plan->passes[reader];
                barrierCount += planned.barriers.size();
                ACCESS_CHECK(planned.barriers.size() == (reader == 1 ? 1u : 0u));
                // Retention uses and producer edges survive barrier removal;
                // independent readers must not gain read-to-read dependencies.
                ACCESS_CHECK(planned.uses.size() == 1 && planned.uses[0].resource == 0);
                ACCESS_CHECK(planned.predecessors.size() == 1 && planned.predecessors[0] == 0);
            }
            ACCESS_CHECK(barrierCount == 1);
        }
        return RHITestResult::pass("six same-scope readers require one RAW barrier per plan");
    }
};

class AccessPlanRawVisibilityScopePairsTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanRawVisibilityScopePairsTest() { name = "render_graph_access_plan_raw_visibility_scope_pairs"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        constexpr SyncScope fragmentUniform{PipelineStageBits::FragmentShader, AccessBits::UniformRead};
        constexpr SyncScope computeUniform{PipelineStageBits::ComputeShader, AccessBits::UniformRead};
        const std::array passes{
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.uses = {bufferUse(fragmentUniform)}},
            GraphAccessPass{.uses = {bufferUse(kFragmentRead)}},
            GraphAccessPass{.uses = {bufferUse(computeUniform)}},
            GraphAccessPass{.uses = {bufferUse(kFragmentRead)}},
            GraphAccessPass{.uses = {bufferUse(computeUniform)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan);
        // Compute/ShaderRead plus Fragment/UniformRead does not establish
        // Fragment/ShaderRead or Compute/UniformRead visibility.
        for (size_t firstScope = 1; firstScope <= 4; ++firstScope) {
            ACCESS_CHECK(plan->passes[firstScope].barriers.size() == 1);
            const auto& barrier = plan->passes[firstScope].barriers.front();
            ACCESS_CHECK(!barrier.executionOnly);
            ACCESS_CHECK(containsBits(barrier.beforeScope.access, AccessBits::ShaderWrite));
            ACCESS_CHECK(containsBits(barrier.afterScope.stages, passes[firstScope].uses[0].scope.stages));
            ACCESS_CHECK(containsBits(barrier.afterScope.access, passes[firstScope].uses[0].scope.access));
        }
        ACCESS_CHECK(plan->passes[5].barriers.empty() && plan->passes[6].barriers.empty());

        constexpr SyncScope bothScopes{PipelineStageBits::ComputeShader | PipelineStageBits::FragmentShader,
            AccessBits::ShaderRead | AccessBits::UniformRead};
        const std::array broadThenSubsets{
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.uses = {bufferUse(bothScopes)}},
            GraphAccessPass{.uses = {bufferUse(kFragmentRead)}},
            GraphAccessPass{.uses = {bufferUse(computeUniform)}},
        };
        auto subsets = buildGraphAccessPlan(resources, broadThenSubsets);
        ACCESS_CHECK(subsets && subsets->passes[1].barriers.size() == 1);
        ACCESS_CHECK(subsets->passes[2].barriers.empty() && subsets->passes[3].barriers.empty());
        return RHITestResult::pass();
    }
};

class AccessPlanRawVisibilityWriterGenerationTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanRawVisibilityWriterGenerationTest() { name = "render_graph_access_plan_raw_visibility_writer_generation"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        const std::array passes{
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.uses = {bufferUse(kComputeWrite, true)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan && plan->passes[2].barriers.empty() && plan->passes[5].barriers.empty());
        ACCESS_CHECK(plan->passes[1].barriers.size() == 1 && plan->passes[4].barriers.size() == 1);
        ACCESS_CHECK(plan->passes[4].predecessors.size() == 1 && hasPredecessor(*plan, 4, 3));
        for (size_t writer : {3u, 6u}) {
            // Both readers must remain in the WAR frontier, including the
            // reader whose redundant RAW barrier was removed.
            ACCESS_CHECK(hasPredecessor(*plan, writer, writer - 3));
            ACCESS_CHECK(hasPredecessor(*plan, writer, writer - 2));
            ACCESS_CHECK(hasPredecessor(*plan, writer, writer - 1));
            ACCESS_CHECK(plan->passes[writer].barriers.size() == 1);
            ACCESS_CHECK(!plan->passes[writer].barriers.front().executionOnly);
            ACCESS_CHECK(containsBits(plan->passes[writer].barriers.front().beforeScope.access,
                AccessBits::ShaderRead | AccessBits::ShaderWrite));
        }
        // Consecutive writes still need WAW ordering even without readers.
        ACCESS_CHECK(plan->passes[7].barriers.size() == 1 && hasPredecessor(*plan, 7, 6));
        ACCESS_CHECK(!plan->passes[7].barriers.front().executionOnly);
        ACCESS_CHECK(containsBits(plan->passes[7].barriers.front().beforeScope.access, AccessBits::ShaderWrite));
        return RHITestResult::pass();
    }
};

class AccessPlanRawVisibilityLayoutGenerationTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanRawVisibilityLayoutGenerationTest() { name = "render_graph_access_plan_raw_visibility_layout_generation"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Texture2D}};
        const auto imageUse = [](ResourceState state, SyncScope scope, bool writes = false) {
            return GraphAccessUse{.resource = 0, .state = state, .scope = scope, .writes = writes};
        };
        const std::array passes{
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kComputeRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kComputeRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kFragmentRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kFragmentRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::TransferSource, kTransferRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::TransferSource, kTransferRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kComputeRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kComputeRead)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan);
        for (size_t pass = 0; pass < passes.size(); ++pass) {
            ACCESS_CHECK(plan->passes[pass].barriers.size() == (pass % 2 == 0 ? 1u : 0u));
        }
        ACCESS_CHECK(plan->passes[0].barriers.front().before == ResourceState::Undefined);
        ACCESS_CHECK(plan->passes[4].barriers.front().before == ResourceState::ShaderRead);
        ACCESS_CHECK(plan->passes[4].barriers.front().after == ResourceState::TransferSource);
        ACCESS_CHECK(plan->passes[6].barriers.front().before == ResourceState::TransferSource);
        ACCESS_CHECK(plan->passes[6].barriers.front().after == ResourceState::ShaderRead);
        for (size_t reader = 0; reader < 4; ++reader) {
            ACCESS_CHECK(hasPredecessor(*plan, 4, reader));
        }
        ACCESS_CHECK(hasPredecessor(*plan, 6, 4) && hasPredecessor(*plan, 6, 5));
        ACCESS_CHECK(hasPredecessor(*plan, 7, 6));

        const std::array writerThenLayout{
            GraphAccessPass{.uses = {imageUse(ResourceState::General, kComputeWrite, true)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kComputeRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kComputeRead)}},
            GraphAccessPass{.uses = {imageUse(ResourceState::ShaderRead, kFragmentRead)}},
        };
        auto written = buildGraphAccessPlan(resources, writerThenLayout);
        ACCESS_CHECK(written && written->passes[1].barriers.size() == 1 && written->passes[2].barriers.empty());
        ACCESS_CHECK(written->passes[3].barriers.size() == 1);
        ACCESS_CHECK(hasPredecessor(*written, 3, 0) && hasPredecessor(*written, 3, 1));
        ACCESS_CHECK(containsBits(written->passes[3].barriers.front().beforeScope.access,
            AccessBits::ShaderWrite | AccessBits::MemoryWrite));
        return RHITestResult::pass();
    }
};

class AccessPlanRawVisibilityInitialQueueTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.plan.contract.") + name}};
    }

    AccessPlanRawVisibilityInitialQueueTest() { name = "render_graph_access_plan_raw_visibility_initial_queue"; }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const std::array resources{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer,
            .state = ResourceState::General, .scope = kComputeWrite}};
        const std::array passes{
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 1, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 1, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 2, .uses = {bufferUse(kFragmentRead)}},
            GraphAccessPass{.queue = 2, .uses = {bufferUse(kFragmentRead)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        ACCESS_CHECK(plan);
        for (size_t pass = 0; pass < passes.size(); ++pass) {
            const bool firstOnQueue = pass == 0 || pass == 2 || pass == 5;
            ACCESS_CHECK(plan->passes[pass].barriers.size() == (firstOnQueue ? 1u : 0u));
            // Initial imports have no in-plan producer. Their independent
            // consumers must not inherit another queue's coverage or edges.
            ACCESS_CHECK(plan->passes[pass].predecessors.empty());
        }
        const std::array undefined{GraphAccessResource{.type = render::RenderGraphResourceType::Buffer}};
        const std::array remoteThenLocal{
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeWrite, true)}},
            GraphAccessPass{.queue = 1, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeRead)}},
            GraphAccessPass{.queue = 0, .uses = {bufferUse(kComputeRead)}},
        };
        auto fanout = buildGraphAccessPlan(undefined, remoteThenLocal);
        ACCESS_CHECK(fanout && fanout->passes[1].barriers.empty());
        ACCESS_CHECK(fanout->passes[2].barriers.size() == 1 && fanout->passes[3].barriers.empty());
        for (size_t reader = 1; reader < remoteThenLocal.size(); ++reader) {
            ACCESS_CHECK(fanout->passes[reader].predecessors.size() == 1);
            ACCESS_CHECK(hasPredecessor(*fanout, reader, 0));
        }
        return RHITestResult::pass();
    }
};

// Execute the existing copy shader on graphics so the same reflected resource
// fans out onto genuinely different queues without duplicating shader plumbing.
class AccessPlanGraphicsCopyPass final : public render::ComputePass {
public:
    render::QueueType queueType() const override { return render::QueueType::Graphics; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    render::CPURecordingPolicy cpuRecordingPolicy() const override { return render::CPURecordingPolicy::ParallelJoined; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        return pass_->reflect(context);
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        return pass_->compile(context, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        return pass_->execute(context);
    }
private:
    std::unique_ptr<render::RenderGraphPass> pass_ = render::builtin_pass::createRenderGraphBufferCopyPass();
};

class AccessPlanGPUFanoutTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graph.fanout.readback", "graph.joined.pipelined.reuse.readback"},
            bench::Layer::RenderGraph, "async", "sync", {"fanout.bin"});
    }

    AccessPlanGPUFanoutTest() { type = RHITestType::Rendering; name = "render_graph_access_plan_gpu_fanout"; }
    RHITestResult run(RHITestContext& context) override
    {
        constexpr std::array expected{0x11223344u, 0xAABBCCDDu, 0xDEADBEEFu, 0xCAFEBABEu};
        static const bool registered = render::registerRenderGraphPassType("AccessPlanGraphicsCopyPass",
            "Access-plan queue fanout regression", [] { return std::make_unique<AccessPlanGraphicsCopyPass>(); });
        (void)registered;
        auto created = bench::createTestDevice(context, {.applicationName = "Access plan fanout",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true, .enableAsyncCompute = true});
        ACCESS_CHECK(created);
        auto device = std::move(*created);
        auto* graphics = device->getQueue(render::QueueType::Graphics);
        auto* compute = device->getQueue(render::QueueType::Compute);
        ACCESS_CHECK(graphics && compute);
        render::RenderGraph graph;
        graph.addNode("RenderGraphBufferWritePass", "Writer");
        for (uint32_t branch = 0; branch < 6; ++branch) {
            const auto name = "Reader" + std::to_string(branch);
            graph.addNode(branch % 2 ? "AccessPlanGraphicsCopyPass" : "RenderGraphBufferCopyPass", name);
            graph.addEdge("Writer.data", name + ".source");
            graph.markOutput(name + ".data");
        }
        render::RenderGraphExecutor executor;
        std::string log;
        auto compiled = executor.compile(*device, graph, 1, 1, log);
        if (!compiled) { return RHITestResult::fail("compile: " + log); }
        for (bool aliasQueues : {false, true}) {
            for (uint32_t workers : {1u, 4u}) {
                for (auto mode : {render::FrameSubmissionMode::Joined, render::FrameSubmissionMode::Pipelined}) {
                    // Consecutive submissions reuse graph resources across both frame slots.
                    for (uint32_t frame = 0; frame < 4; ++frame) {
                        ACCESS_REQUIRE(executor.execute({.graphicsQueue = graphics,
                            .computeQueue = aliasQueues ? graphics : compute, .recordingWorkerLimit = workers,
                            .recordingBatchWorkload = 1, .submissionMode = mode}));
                    }
                    ACCESS_REQUIRE(executor.waitForSubmittedWork(5'000'000'000ull));
                    for (uint32_t branch = 0; branch < 6; ++branch) {
                        auto* output = executor.outputResource("Reader" + std::to_string(branch) + ".data");
                        ACCESS_CHECK(output && output->buffer && output->bufferDesc.size == sizeof(expected));
                        output->buffer->invalidate();
                        auto* mapped = output->buffer->map();
                        ACCESS_CHECK(mapped);
                        std::array<uint32_t, 4> actual{};
                        std::memcpy(actual.data(), mapped, sizeof(actual));
                        output->buffer->unmap();
                        bench::readbackEvidence(context, "fanout.bin", std::span<const uint32_t>(actual));
                        ACCESS_CHECK(actual == expected);
                    }
                }
            }
        }
        ACCESS_REQUIRE(device->waitIdle());

        return RHITestResult::pass(graphics->sameQueue(*compute)
            ? "32 frames, six consumers; device exposes aliased compute and graphics queues"
            : "32 frames, six consumers; distinct compute/graphics queues and explicit queue aliases");
    }
};

class TemporalValidationTestPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("prevHistory").sampledRead().temporalRead("accum", -1);
        reflection.addTextureOutput("currHistory").storageWrite().temporalWrite("accum");
        reflection.addTextureOutput("color").colorWrite();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext&) override { return {}; }
};

class ResourceLifecycleContractTest final : public RHITest {
public:
    ResourceLifecycleContractTest() { name = "render_graph_resource_lifecycle_contracts"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {"graph.lifecycle.contract.ids_versions_snapshot_temporal"}};
    }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        const auto idA = render::makeLogicalResourceId("render.environment", "radiance", 1, 0x1234);
        const auto idA2 = render::makeLogicalResourceId("render.environment", "radiance", 1, 0x1234);
        const auto idViewB = render::makeLogicalResourceId("render.environment", "radiance", 2, 0x1234);
        const auto idSharedMedium = render::makeLogicalResourceId("render.environment", "transmittance", 0, 0xABCD);
        ACCESS_CHECK(idA.valid() && idA == idA2);
        ACCESS_CHECK(idA.value != idViewB.value);
        ACCESS_CHECK(idSharedMedium.viewId == 0 && idSharedMedium.value != idA.value);

        const render::ResourceVersionId v1{.logicalId = idA, .version = 1, .contentRevision = 42, .allocationGeneration = 1};
        const render::ResourceVersionId v2Rebuilt{.logicalId = idA, .version = 2, .contentRevision = 42, .allocationGeneration = 2};
        ACCESS_CHECK(v1.valid() && v2Rebuilt.valid());
        ACCESS_CHECK(v1.contentRevision == v2Rebuilt.contentRevision);
        ACCESS_CHECK(v1.allocationGeneration != v2Rebuilt.allocationGeneration);
        ACCESS_CHECK(!(v1 == v2Rebuilt));

        struct DummySet {
            int value = 0;
        };
        auto alive = std::make_shared<bool>(true);
        std::weak_ptr<bool> weakAlive = alive;
        render::FrameExecutionResourceSnapshot snapshotN;
        snapshotN.beginBuild(10, 1);
        auto setN = std::make_shared<DummySet>(DummySet{.value = 77});
        ACCESS_REQUIRE(snapshotN.bindResourceSet<DummySet>(setN, alive, idA, v1));
        ACCESS_REQUIRE(snapshotN.bindLogicalResource("env.radiance", render::ResolvedLogicalBinding{
            .logicalId = idA, .version = v1, .ownership = render::RenderGraphResourceOwnership::ProviderOwned,
            .type = render::RenderGraphResourceType::Texture2D, .backingAllocationId = 9001, .owner = alive,
            .producerName = "render.environment"}));
        snapshotN.freeze();
        ACCESS_CHECK(snapshotN.frozen());
        // Post-freeze mutation must fail without altering frozen bindings.
        ACCESS_CHECK(render::hasError(snapshotN.bindResourceSet<DummySet>(
            std::make_shared<DummySet>(DummySet{.value = 99}), nullptr, idViewB, v2Rebuilt), render::Error::InvalidArgument));
        ACCESS_CHECK(render::hasError(snapshotN.retainOwner(std::make_shared<int>(1)), render::Error::InvalidArgument));
        alive.reset();
        setN.reset();
        ACCESS_CHECK(!weakAlive.expired());
        ACCESS_CHECK(snapshotN.resourceSet<DummySet>() != nullptr && snapshotN.resourceSet<DummySet>()->value == 77);
        const auto* bound = snapshotN.findLogicalResource("env.radiance");
        ACCESS_CHECK(bound != nullptr && bound->backingAllocationId == 9001 && bound->version == v1);

        // Building Frame N+1 snapshot must not disturb Frame N snapshot.
        render::FrameExecutionResourceSnapshot snapshotN1;
        snapshotN1.beginBuild(11, 1);
        ACCESS_REQUIRE(snapshotN1.bindResourceSet<DummySet>(
            std::make_shared<DummySet>(DummySet{.value = 88}), nullptr, idViewB, v2Rebuilt));
        snapshotN1.freeze();
        ACCESS_CHECK(snapshotN.resourceSet<DummySet>()->value == 77);
        ACCESS_CHECK(snapshotN1.resourceSet<DummySet>()->value == 88);

        // Temporal reflection validation contract.
        static const bool registered = render::registerRenderGraphPassType("TemporalValidationTestPass",
            "Temporal field validation contract", [] { return std::make_unique<TemporalValidationTestPass>(); });
        (void)registered;
        render::RenderGraph validTemporalGraph;
        validTemporalGraph.addNode("TemporalValidationTestPass", "Accum");
        validTemporalGraph.markOutput("Accum.color");
        std::string log;
        ACCESS_CHECK(validTemporalGraph.validate(log));

        render::RenderGraph invalidTemporalEdgeGraph;
        invalidTemporalEdgeGraph.addNode("ClearColorPass", "Clear");
        invalidTemporalEdgeGraph.addNode("TemporalValidationTestPass", "Accum");
        invalidTemporalEdgeGraph.addEdge("Clear.color", "Accum.prevHistory");
        invalidTemporalEdgeGraph.markOutput("Accum.color");
        ACCESS_CHECK(!invalidTemporalEdgeGraph.validate(log));

        return RHITestResult::pass();
    }
};

class CrossFrameTrackerAndVersionPoolTest final : public RHITest {
public:
    CrossFrameTrackerAndVersionPoolTest() { type = RHITestType::Rendering; name = "render_graph_cross_frame_tracker_and_pool"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = true, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {"graph.lifecycle.cross_frame_tracker", "graph.lifecycle.version_pool"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto created = bench::createTestDevice(context, {.applicationName = "Cross-frame tracker and version pool",
            .enableValidation = context.enableValidation, .enableAsyncCompute = true});
        ACCESS_CHECK(created);
        auto device = std::move(*created);
        auto* graphics = device->getQueue(render::QueueType::Graphics);
        auto* compute = device->getQueue(render::QueueType::Compute);
        ACCESS_CHECK(graphics && compute);

        std::unique_ptr<render::Semaphore> gateA, gateB;
        ACCESS_REQUIRE(device->createSemaphore({.initialValue = 0})
            .transform([&](auto s) { gateA = std::move(s); }));
        ACCESS_REQUIRE(device->createSemaphore({.initialValue = 0})
            .transform([&](auto s) { gateB = std::move(s); }));

        render::QueueSubmissionTracker gfxTracker, cmpTracker;
        ACCESS_REQUIRE(gfxTracker.initialize(*device, *graphics));
        ACCESS_REQUIRE(cmpTracker.initialize(*device, *compute));

        std::unique_ptr<render::CommandPool> gfxPool, cmpPool;
        ACCESS_REQUIRE(device->createCommandPool(*graphics)
            .transform([&](auto p) { gfxPool = std::move(p); }));
        ACCESS_REQUIRE(device->createCommandPool(*compute)
            .transform([&](auto p) { cmpPool = std::move(p); }));

        auto submitGated = [&](render::QueueSubmissionTracker& tracker, render::CommandPool& pool,
            render::Semaphore& gate, render::RenderFrameContext& frame) -> render::Result<render::GPUCompletionPoint> {
            std::unique_ptr<render::CommandBuffer> cmd;
            auto r = pool.createCommandBuffer().transform([&](auto c) { cmd = std::move(c); });
            if (!r) { return render::makeError(r.error()); }
            r = cmd->begin(frame.submissionContext());
            if (!r) { return render::makeError(r.error()); }
            r = cmd->end();
            if (!r) { return render::makeError(r.error()); }
            render::RecordedBatch batch;
            render::CommandBuffer* raw = cmd.get();
            r = batch.seal(frame, {&raw, 1});
            if (!r) { return render::makeError(r.error()); }
            frame.retain(std::shared_ptr<render::CommandBuffer>(std::move(cmd)));
            const render::SemaphoreSubmitDesc wait{.semaphore = &gate, .value = 1,
                .stages = PipelineStageBits::AllCommands};
            render::SubmissionReceipt receipt;
            r = tracker.submitBatch(batch, {.waitSemaphores = {&wait, 1}}, frame)
                .transform([&](auto v) { receipt = std::move(v); });
            if (!r) { return render::makeError(r.error()); }
            return receipt.completion();
        };

        // Declare frames BEFORE gateGuard so gateGuard unwinds first and unblocks
        // any in-flight GPU work before ~RenderFrameContext() waits on completion.
        render::RenderFrameContext frameA(0), frameB(1);
        struct GateReleaseGuard {
            render::Semaphore* a = nullptr;
            render::Semaphore* b = nullptr;
            ~GateReleaseGuard()
            {
                if (a) { (void)a->signal(1); }
                if (b) { (void)b->signal(1); }
            }
        } gateGuard{gateA.get(), gateB.get()};

        ACCESS_REQUIRE(frameA.begin(0));
        ACCESS_REQUIRE(frameB.begin(1));
        render::GPUCompletionPoint compA, compB;
        ACCESS_REQUIRE(submitGated(cmpTracker, *cmpPool, *gateA, frameA).transform([&](auto c) { compA = std::move(c); }));
        ACCESS_REQUIRE(frameA.sealRecording());
        ACCESS_REQUIRE(frameA.finishSubmission());
        ACCESS_REQUIRE(submitGated(cmpTracker, *cmpPool, *gateB, frameB).transform([&](auto c) { compB = std::move(c); }));
        ACCESS_REQUIRE(frameB.sealRecording());
        ACCESS_REQUIRE(frameB.finishSubmission());

        const uint64_t backingId = 777;
        const auto logicalId = render::makeLogicalResourceId("test", "sharedTarget");
        render::CrossFrameResourceTracker tracker;

        // Multi-reader WAR: two concurrent readers on distinct completions.
        tracker.commitAcceptedAccess({.backingAllocationId = backingId, .logicalId = logicalId,
            .version = {.logicalId = logicalId, .version = 1, .contentRevision = 1, .allocationGeneration = 1},
            .type = render::RenderGraphResourceType::Texture2D, .frameIndex = 0, .queue = compute,
            .state = ResourceState::ShaderRead, .scope = kComputeRead, .writes = false,
            .completion = compA, .passName = "ReaderA"});
        tracker.commitAcceptedAccess({.backingAllocationId = backingId, .logicalId = logicalId,
            .version = {.logicalId = logicalId, .version = 1, .contentRevision = 1, .allocationGeneration = 1},
            .type = render::RenderGraphResourceType::Texture2D, .frameIndex = 1, .queue = compute,
            .state = ResourceState::ShaderRead, .scope = kComputeRead, .writes = false,
            .completion = compB, .passName = "ReaderB"});
        ACCESS_CHECK(!tracker.isBackingIdle(backingId));
        ACCESS_CHECK(tracker.activeCompletions(backingId).size() == 2);

        // Write on a distinct queue must wait on BOTH readers, not just the last reader.
        const auto warPlan = tracker.planAccess(backingId, "sharedTarget",
            render::RenderGraphResourceType::Texture2D, graphics, ResourceState::ColorAttachment, true);
        ACCESS_CHECK(!warPlan.requiresConservativeFallback);
        if (!graphics->sameQueue(*compute)) {
            ACCESS_CHECK(warPlan.waits.size() == 2);
        }

        // Read-only sharing with matching layout needs no wait on other readers.
        const auto readPlan = tracker.planAccess(backingId, "sharedTarget",
            render::RenderGraphResourceType::Texture2D, graphics, ResourceState::ShaderRead, false);
        ACCESS_CHECK(!readPlan.requiresConservativeFallback);
        ACCESS_CHECK(readPlan.waits.empty());

        // #60 exclusive queue ownership cross-family access forces conservative fallback.
        if (!graphics->sameQueue(*compute)) {
            const auto exclPlan = tracker.planAccess(backingId, "sharedTarget",
                render::RenderGraphResourceType::Texture2D, graphics, ResourceState::ShaderRead, false, true);
            ACCESS_CHECK(exclPlan.requiresConservativeFallback && !exclPlan.conservativeReason.empty());
        }

        // Test RenderGraphResourceVersionPool under MinimizeMemory vs MaximizeOverlap.
        render::DeferredReleaseQueue deferredReleases;
        const render::TextureDesc texDesc{.type = render::TextureType::Texture2D,
            .usage = render::TextureUsageBits::Storage | render::TextureUsageBits::Sampled,
            .format = render::Format::RGBA8Unorm, .width = 64, .height = 64};
        const render::TextureViewDesc viewDesc{.format = render::Format::RGBA8Unorm};
        const uint64_t poolLogicalId = render::makeLogicalResourceId("test", "pooledTex").value;

        render::RenderGraphResourceVersionPool pool;
        pool.setPolicy({.mode = render::RenderGraphMemoryPolicy::MinimizeMemory, .extraVersionBudgetBytes = 0, .maxPhysicalVersionsPerResource = 2});
        render::RenderGraphResourceVersionPool::AcquiredTexture first;
        ACCESS_REQUIRE(pool.acquireTexture(*device, poolLogicalId, texDesc, viewDesc, 0, tracker, deferredReleases)
            .transform([&](auto v) { first = std::move(v); }));
        ACCESS_CHECK(first.texture != nullptr && first.physicalVersionCount == 1 && !first.extraVersion);

        // Mark first backing busy on compA.
        tracker.commitAcceptedAccess({.backingAllocationId = first.backingAllocationId,
            .type = render::RenderGraphResourceType::Texture2D, .frameIndex = 0, .queue = graphics,
            .state = ResourceState::General, .scope = kComputeWrite, .writes = true,
            .completion = compA, .passName = "PoolWriter0"});

        // MinimizeMemory must reuse the single backing (0 extra versions).
        render::RenderGraphResourceVersionPool::AcquiredTexture secondMinMem;
        ACCESS_REQUIRE(pool.acquireTexture(*device, poolLogicalId, texDesc, viewDesc, 1, tracker, deferredReleases)
            .transform([&](auto v) { secondMinMem = std::move(v); }));
        ACCESS_CHECK(secondMinMem.backingAllocationId == first.backingAllocationId);
        ACCESS_CHECK(pool.stats().extraVersionCount == 0);

        // MaximizeOverlap with zero budget still refuses extra allocation.
        pool.setPolicy({.mode = render::RenderGraphMemoryPolicy::MaximizeOverlap, .extraVersionBudgetBytes = 1, .maxPhysicalVersionsPerResource = 2});
        render::RenderGraphResourceVersionPool::AcquiredTexture secondOverBudget;
        ACCESS_REQUIRE(pool.acquireTexture(*device, poolLogicalId, texDesc, viewDesc, 2, tracker, deferredReleases)
            .transform([&](auto v) { secondOverBudget = std::move(v); }));
        ACCESS_CHECK(secondOverBudget.backingAllocationId == first.backingAllocationId);
        ACCESS_CHECK(pool.stats().extraVersionCount == 0);

        // MaximizeOverlap with adequate budget allocates a second physical version while backing 0 is busy.
        pool.setPolicy({.mode = render::RenderGraphMemoryPolicy::MaximizeOverlap,
            .extraVersionBudgetBytes = 4ull * 1024 * 1024, .maxPhysicalVersionsPerResource = 2});
        render::RenderGraphResourceVersionPool::AcquiredTexture secondOverlap;
        ACCESS_REQUIRE(pool.acquireTexture(*device, poolLogicalId, texDesc, viewDesc, 3, tracker, deferredReleases)
            .transform([&](auto v) { secondOverlap = std::move(v); }));
        ACCESS_CHECK(secondOverlap.backingAllocationId != first.backingAllocationId);
        ACCESS_CHECK(secondOverlap.extraVersion && secondOverlap.physicalVersionCount == 2);
        ACCESS_CHECK(pool.stats().extraVersionCount == 1 && pool.stats().extraVersionBytes > 0);

        // Signal Gate A only: backingId still busy via Gate B; first pooled backing becomes idle!
        ACCESS_REQUIRE(gateA->signal(1));
        gateGuard.a = nullptr;
        ACCESS_REQUIRE(compA.wait(5'000'000'000ull));
        ACCESS_CHECK(!tracker.isBackingIdle(backingId));
        ACCESS_CHECK(tracker.isBackingIdle(first.backingAllocationId));

        // Mark second backing busy on compB; acquiring again must reuse idle first backing (no 3rd alloc).
        tracker.commitAcceptedAccess({.backingAllocationId = secondOverlap.backingAllocationId,
            .type = render::RenderGraphResourceType::Texture2D, .frameIndex = 3, .queue = compute,
            .state = ResourceState::General, .scope = kComputeWrite, .writes = true,
            .completion = compB, .passName = "PoolWriter3"});
        render::RenderGraphResourceVersionPool::AcquiredTexture thirdReused;
        ACCESS_REQUIRE(pool.acquireTexture(*device, poolLogicalId, texDesc, viewDesc, 4, tracker, deferredReleases)
            .transform([&](auto v) { thirdReused = std::move(v); }));
        ACCESS_CHECK(thirdReused.backingAllocationId == first.backingAllocationId);
        ACCESS_CHECK(pool.stats().totalPhysicalVersions == 2);

        // Signal Gate B: now both readers are done and backingId is completely idle!
        ACCESS_REQUIRE(gateB->signal(1));
        gateGuard.b = nullptr;
        ACCESS_REQUIRE(compB.wait(5'000'000'000ull));
        ACCESS_CHECK(tracker.isBackingIdle(backingId));
        const auto idleWarPlan = tracker.planAccess(backingId, "sharedTarget",
            render::RenderGraphResourceType::Texture2D, graphics, ResourceState::ColorAttachment, true);
        ACCESS_CHECK(idleWarPlan.waits.empty());

        pool.retireAll(tracker, deferredReleases);
        deferredReleases.collect();
        ACCESS_REQUIRE(frameA.wait(5'000'000'000ull));
        ACCESS_REQUIRE(frameB.wait(5'000'000'000ull));
        ACCESS_REQUIRE(device->waitIdle());
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(AccessPlanWriterVisibilityTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanAccelerationStructureTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanQueueFanoutTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanWarTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanImageLayoutTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanAliasesTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanInvalidInputTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanFrameBoundaryTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanInternalBoundaryTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanRawVisibilityReuseTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanRawVisibilityScopePairsTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanRawVisibilityWriterGenerationTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanRawVisibilityLayoutGenerationTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanRawVisibilityInitialQueueTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanGPUFanoutTest);
METALLIC_REGISTER_RHI_TEST(ResourceLifecycleContractTest);
METALLIC_REGISTER_RHI_TEST(CrossFrameTrackerAndVersionPoolTest);

#undef ACCESS_REQUIRE
#undef ACCESS_CHECK

} // namespace
} // namespace metallic::tests
