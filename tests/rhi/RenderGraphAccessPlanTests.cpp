#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"

#include <algorithm>
#include <array>
#include <cstring>

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
    if (!(condition)) { return RhiTestResult::fail(#condition); } \
} while (false)
#define ACCESS_REQUIRE(expression) do { \
    const auto checked = (expression); \
    if (!checked) { return RhiTestResult::fail(std::string(#expression) + ": " + render::resultToString(checked)); } \
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

class AccessPlanWriterVisibilityTest final : public RhiTest {
public:
    AccessPlanWriterVisibilityTest() { name = "render_graph_access_plan_writer_stage_visibility"; }
    RhiTestResult run(RhiTestContext&) override
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
        return RhiTestResult::pass();
    }
};

class AccessPlanQueueFanoutTest final : public RhiTest {
public:
    AccessPlanQueueFanoutTest() { name = "render_graph_access_plan_queue_fanout_and_join"; }
    RhiTestResult run(RhiTestContext&) override
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
        return RhiTestResult::pass();
    }
};

class AccessPlanWarTest final : public RhiTest {
public:
    AccessPlanWarTest() { name = "render_graph_access_plan_read_frontier_execution_dependency"; }
    RhiTestResult run(RhiTestContext&) override
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
        return RhiTestResult::pass();
    }
};

class AccessPlanImageLayoutTest final : public RhiTest {
public:
    AccessPlanImageLayoutTest() { name = "render_graph_access_plan_image_layout_frontier"; }
    RhiTestResult run(RhiTestContext&) override
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
        ACCESS_CHECK(hasPredecessor(*initial, 1, 0) && hasPredecessor(*initial, 2, 0));
        // The first image transition is itself a write; another local reader
        // stage needs a GPU memory dependency, not only a CPU submission edge.
        ACCESS_CHECK(hasVisibilityBarrier(*initial, 1, 2, PipelineStageBits::ComputeShader,
            AccessBits::MemoryWrite, PipelineStageBits::FragmentShader));

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
        return RhiTestResult::pass();
    }
};

class AccessPlanAliasesTest final : public RhiTest {
public:
    AccessPlanAliasesTest() { name = "render_graph_access_plan_alias_merge_and_rejection"; }
    RhiTestResult run(RhiTestContext&) override
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
        return RhiTestResult::pass();
    }
};

class AccessPlanInvalidInputTest final : public RhiTest {
public:
    AccessPlanInvalidInputTest() { name = "render_graph_access_plan_invalid_resource_is_transactional"; }
    RhiTestResult run(RhiTestContext&) override
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
        return RhiTestResult::pass();
    }
};

class AccessPlanFrameBoundaryTest final : public RhiTest {
public:
    AccessPlanFrameBoundaryTest() { name = "render_graph_access_plan_initial_scope_and_access_types"; }
    RhiTestResult run(RhiTestContext&) override
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
        return RhiTestResult::pass();
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
    render::CpuRecordingPolicy cpuRecordingPolicy() const override { return render::CpuRecordingPolicy::ParallelJoined; }
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

class AccessPlanGpuFanoutTest final : public RhiTest {
public:
    AccessPlanGpuFanoutTest() { type = RhiTestType::Rendering; name = "render_graph_access_plan_gpu_fanout"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr std::array expected{0x11223344u, 0xAABBCCDDu, 0xDEADBEEFu, 0xCAFEBABEu};
        static const bool registered = render::registerRenderGraphPassType("AccessPlanGraphicsCopyPass",
            "Access-plan queue fanout regression", [] { return std::make_unique<AccessPlanGraphicsCopyPass>(); });
        (void)registered;
        std::atomic_uint validationFailures = 0;
        auto created = render::createDevice({.applicationName = "Metallic Access Plan Fanout Test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .validationSink = {.callback = [](void* data, const render::ValidationMessage& message) noexcept {
                if (message.messageIdName && (std::strstr(message.messageIdName, "VUID-") ||
                    std::strstr(message.messageIdName, "SYNC-HAZARD"))) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationFailures}, .enableAsyncCompute = true});
        if (!created) {
            return render::hasError(created, render::Error::Unsupported)
                ? RhiTestResult::skip("bindless access-plan workload unsupported")
                : RhiTestResult::fail(render::resultToString(created));
        }
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
        if (!compiled) { return RhiTestResult::fail("compile: " + log); }
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
                        ACCESS_CHECK(actual == expected);
                    }
                }
            }
        }
        ACCESS_REQUIRE(device->waitIdle());
        ACCESS_CHECK(validationFailures == 0);
        return RhiTestResult::pass(graphics->sameQueue(*compute)
            ? "32 frames, six consumers; device exposes aliased compute and graphics queues"
            : "32 frames, six consumers; distinct compute/graphics queues and explicit queue aliases");
    }
};

METALLIC_REGISTER_RHI_TEST(AccessPlanWriterVisibilityTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanQueueFanoutTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanWarTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanImageLayoutTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanAliasesTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanInvalidInputTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanFrameBoundaryTest);
METALLIC_REGISTER_RHI_TEST(AccessPlanGpuFanoutTest);

#undef ACCESS_REQUIRE
#undef ACCESS_CHECK

} // namespace
} // namespace metallic::tests
