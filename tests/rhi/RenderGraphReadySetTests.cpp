#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"

#include <algorithm>
#include <array>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metallic::tests {
namespace {

#define READY_SET_CHECK(condition) do { \
    if (!(condition)) { return RHITestResult::fail(std::string(#condition) + ": " + log); } \
} while (false)

template<render::QueueType Queue, bool Async = true>
class ReadySetProbe final : public render::RenderGraphPass {
public:
    render::QueueType queueType() const override { return Queue; }
    render::RenderGraphPassKind kind() const override
    {
        if constexpr (Queue == render::QueueType::Compute) { return render::RenderGraphPassKind::Compute; }
        if constexpr (Queue == render::QueueType::Graphics) { return render::RenderGraphPassKind::Raster; }
        return render::RenderGraphPassKind::Unsafe;
    }
    bool supportsAsyncQueue() const override { return properties().value("async", Async); }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        for (const char* name : {"in0", "in1", "in2"}) {
            auto& field = reflection.addBufferInput(name);
            if constexpr (Queue == render::QueueType::Copy) { field.transferRead(); }
            else { field.storageRead(); }
            field.optional = true;
        }
        for (const char* name : {"out0", "out1"}) {
            auto& field = reflection.addBufferOutput(name).buffer(16);
            if constexpr (Queue == render::QueueType::Copy) { field.transferWrite(); }
            else { field.storageWrite(); }
        }
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext&) override { return {}; }
};

bool registerReadySetProbes()
{
    static const bool registered = [] {
        return render::registerRenderGraphPassType("ReadySetGraphicsProbe", "CPU ready-set graphics probe",
            [] { return std::make_unique<ReadySetProbe<render::QueueType::Graphics>>(); }) &&
            render::registerRenderGraphPassType("ReadySetComputeProbe", "CPU ready-set compute probe",
                [] { return std::make_unique<ReadySetProbe<render::QueueType::Compute>>(); }) &&
            render::registerRenderGraphPassType("ReadySetCopyProbe", "CPU ready-set copy probe",
                [] { return std::make_unique<ReadySetProbe<render::QueueType::Copy>>(); }) &&
            render::registerRenderGraphPassType("ReadySetOpaqueComputeProbe", "CPU ready-set opaque compute probe",
                [] { return std::make_unique<ReadySetProbe<render::QueueType::Compute, false>>(); }) &&
            render::registerRenderGraphPassType("ReadySetOpaqueGraphicsProbe", "CPU ready-set opaque graphics probe",
                [] { return std::make_unique<ReadySetProbe<render::QueueType::Graphics, false>>(); });
    }();
    return registered;
}

bool buildCheckedOrder(const render::RenderGraph& graph, const std::vector<std::string>& extraOutputs,
    render::detail::ActiveGraph& active, std::string& log)
{
    if (!graph.validate(log)) { return false; }
    active = {};
    const auto resolveTraits = [](const render::RenderGraphNode& node) {
        auto pass = render::createRenderGraphPass(node.type);
        auto properties = node.properties;
        properties.update(node.runtimeProperties);
        pass->setProperties(std::move(properties));
        const bool async = pass->supportsAsyncQueue();
        return render::detail::ActiveGraphSchedulingTraits{
            .queue = async ? pass->queueType() : render::QueueType::Graphics, .opaque = !async};
    };
    if (!render::detail::buildActiveGraph(graph, extraOutputs, active, log, resolveTraits)) { return false; }

    // Check the public graph contract independently of the queue selection policy:
    // every active pass appears once, and every active resource edge is ordered.
    std::unordered_map<std::string, size_t> positions;
    for (size_t index = 0; index < active.executionOrder.size(); ++index) {
        const auto& name = active.executionOrder[index];
        if (!active.activePasses.contains(name) || !positions.emplace(name, index).second) {
            log = "execution order contains an inactive or duplicate pass: " + name;
            return false;
        }
    }
    if (positions.size() != active.activePasses.size()) {
        log = "execution order omitted an active pass";
        return false;
    }
    for (const auto& edge : graph.edges()) {
        if (!active.activePasses.contains(edge.srcPass) || !active.activePasses.contains(edge.dstPass)) { continue; }
        if (positions.at(edge.srcPass) >= positions.at(edge.dstPass)) {
            log = "resource dependency reversed: " + edge.srcPass + " -> " + edge.dstPass;
            return false;
        }
    }
    return true;
}

class ReadySetContractTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.ready_set.contract.") + name}};
    }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    virtual RHITestResult check() = 0;
};

class ReadySetGraphicsChainTest final : public ReadySetContractTest {
public:
    ReadySetGraphicsChainTest() { name = "render_graph_ready_set_graphics_chain"; }
    RHITestResult check() override
    {
        std::string log;
        READY_SET_CHECK(registerReadySetProbes());
        for (const bool withTail : {false, true}) {
            render::RenderGraph graph;
            // Compute has the older ID, but it must not interrupt a ready graphics chain.
            READY_SET_CHECK(graph.addNode("ReadySetComputeProbe", "ComputeRoot"));
            READY_SET_CHECK(graph.addNode("ReadySetGraphicsProbe", "GraphicsProducer"));
            READY_SET_CHECK(graph.addNode("ReadySetComputeProbe", "ComputeFinal"));
            READY_SET_CHECK(graph.addNode("ReadySetGraphicsProbe", "GraphicsConsumer"));
            READY_SET_CHECK(graph.addEdge("GraphicsProducer.out0", "GraphicsConsumer.in0"));
            READY_SET_CHECK(graph.addEdge("ComputeRoot.out0", "ComputeFinal.in0"));
            std::vector<std::string> expected{"GraphicsProducer", "GraphicsConsumer"};
            if (withTail) {
                READY_SET_CHECK(graph.addNode("ReadySetGraphicsProbe", "GraphicsTail"));
                READY_SET_CHECK(graph.addEdge("GraphicsConsumer.out0", "GraphicsTail.in0"));
                READY_SET_CHECK(graph.addEdge("GraphicsTail.out0", "ComputeFinal.in1"));
                expected.push_back("GraphicsTail");
            } else {
                READY_SET_CHECK(graph.addEdge("GraphicsConsumer.out0", "ComputeFinal.in1"));
            }
            READY_SET_CHECK(graph.markOutput("ComputeFinal.out0"));
            expected.insert(expected.end(), {"ComputeRoot", "ComputeFinal"});
            render::detail::ActiveGraph active;
            READY_SET_CHECK(buildCheckedOrder(graph, {}, active, log));
            READY_SET_CHECK(active.executionOrder == expected);
        }

        render::RenderGraph opaqueChain;
        READY_SET_CHECK(opaqueChain.addNode("ReadySetOpaqueGraphicsProbe", "GraphicsProducer"));
        READY_SET_CHECK(opaqueChain.addNode("ReadySetComputeProbe", "ComputeRoot"));
        READY_SET_CHECK(opaqueChain.addNode("ReadySetOpaqueGraphicsProbe", "GraphicsConsumer"));
        READY_SET_CHECK(opaqueChain.addNode("ReadySetComputeProbe", "ComputeFinal"));
        READY_SET_CHECK(opaqueChain.addEdge("GraphicsProducer.out0", "GraphicsConsumer.in0"));
        READY_SET_CHECK(opaqueChain.addEdge("GraphicsConsumer.out0", "ComputeFinal.in0"));
        READY_SET_CHECK(opaqueChain.addEdge("ComputeRoot.out0", "ComputeFinal.in1"));
        READY_SET_CHECK(opaqueChain.markOutput("ComputeFinal.out0"));
        render::detail::ActiveGraph active;
        READY_SET_CHECK(buildCheckedOrder(opaqueChain, {}, active, log));
        // The LookDev shape keeps both opaque graphics passes in order while
        // the independent, audited LUT moves behind their dependent main chain.
        READY_SET_CHECK((active.executionOrder == std::vector<std::string>{
            "GraphicsProducer", "GraphicsConsumer", "ComputeRoot", "ComputeFinal"}));
        return RHITestResult::pass();
    }
};

class ReadySetFanoutDeterminismTest final : public ReadySetContractTest {
public:
    ReadySetFanoutDeterminismTest() { name = "render_graph_ready_set_fanout_copy_determinism"; }
    RHITestResult check() override
    {
        std::string log;
        READY_SET_CHECK(registerReadySetProbes());
        render::RenderGraph graph;
        for (const auto& [type, name] : std::array{
                 std::pair{"ReadySetGraphicsProbe", "GraphicsRoot"},
                 std::pair{"ReadySetCopyProbe", "CopyRoot"},
                 std::pair{"ReadySetGraphicsProbe", "GraphicsReaderA"},
                 std::pair{"ReadySetComputeProbe", "ComputeRoot"},
                 std::pair{"ReadySetGraphicsProbe", "GraphicsReaderB"},
                 std::pair{"ReadySetCopyProbe", "CopyConsumer"},
                 std::pair{"ReadySetComputeProbe", "Join"}}) {
            READY_SET_CHECK(graph.addNode(type, name));
        }
        // Two distinct resources between the same pass pair must not lose or
        // double-release the destination when pass-level predecessors are deduplicated.
        READY_SET_CHECK(graph.addEdge("GraphicsRoot.out0", "GraphicsReaderA.in0"));
        READY_SET_CHECK(graph.addEdge("GraphicsRoot.out1", "GraphicsReaderA.in1"));
        READY_SET_CHECK(graph.addEdge("GraphicsRoot.out0", "GraphicsReaderB.in0"));
        READY_SET_CHECK(graph.addEdge("GraphicsReaderA.out0", "CopyConsumer.in0"));
        READY_SET_CHECK(graph.addEdge("CopyRoot.out0", "CopyConsumer.in1"));
        READY_SET_CHECK(graph.addEdge("GraphicsReaderB.out0", "Join.in0"));
        READY_SET_CHECK(graph.addEdge("ComputeRoot.out0", "Join.in1"));
        READY_SET_CHECK(graph.addEdge("CopyConsumer.out0", "Join.in2"));
        READY_SET_CHECK(graph.markOutput("Join.out0"));
        const std::vector<std::string> expected{"GraphicsRoot", "GraphicsReaderA", "GraphicsReaderB",
            "CopyRoot", "CopyConsumer", "ComputeRoot", "Join"};
        render::detail::ActiveGraph active;
        READY_SET_CHECK(buildCheckedOrder(graph, {}, active, log));
        READY_SET_CHECK(active.executionOrder == expected);

        // File storage order and unordered-container insertion order are not
        // scheduling identities. Preserve IDs while permuting both arrays.
        auto document = render::RenderGraphProperties::parse(render::serializeRenderGraphToString(graph));
        std::mt19937 random(0x52454144u);
        for (uint32_t repetition = 0; repetition < 16; ++repetition) {
            auto& nodes = document["nodes"].get_ref<render::RenderGraphProperties::array_t&>();
            auto& edges = document["edges"].get_ref<render::RenderGraphProperties::array_t&>();
            std::shuffle(nodes.begin(), nodes.end(), random);
            std::shuffle(edges.begin(), edges.end(), random);
            render::RenderGraph reordered;
            READY_SET_CHECK(render::deserializeRenderGraphFromString(document.dump(), reordered, log));
            READY_SET_CHECK(buildCheckedOrder(reordered, {}, active, log));
            READY_SET_CHECK(active.executionOrder == expected);
        }
        return RHITestResult::pass();
    }
};

class ReadySetOpaqueAndOutputsTest final : public ReadySetContractTest {
public:
    ReadySetOpaqueAndOutputsTest() { name = "render_graph_ready_set_opaque_and_extra_outputs"; }
    RHITestResult check() override
    {
        std::string log;
        READY_SET_CHECK(registerReadySetProbes());
        render::RenderGraph graph;
        READY_SET_CHECK(graph.addNode("ReadySetComputeProbe", "ComputeRoot"));
        READY_SET_CHECK(graph.addNode("ReadySetOpaqueComputeProbe", "OpaqueFirst"));
        READY_SET_CHECK(graph.addNode("ReadySetOpaqueGraphicsProbe", "OpaqueSecond"));
        READY_SET_CHECK(graph.addNode("ReadySetGraphicsProbe", "GraphicsJoin"));
        READY_SET_CHECK(graph.addNode("ReadySetComputeProbe", "ComputeFinal"));
        READY_SET_CHECK(graph.addNode("ReadySetCopyProbe", "DormantRoot"));
        READY_SET_CHECK(graph.addNode("ReadySetCopyProbe", "DormantTail"));
        READY_SET_CHECK(graph.addEdge("ComputeRoot.out0", "OpaqueFirst.in0"));
        READY_SET_CHECK(graph.addEdge("OpaqueFirst.out0", "GraphicsJoin.in0"));
        READY_SET_CHECK(graph.addEdge("OpaqueSecond.out0", "GraphicsJoin.in1"));
        READY_SET_CHECK(graph.addEdge("GraphicsJoin.out0", "ComputeFinal.in0"));
        READY_SET_CHECK(graph.addEdge("DormantRoot.out0", "DormantTail.in0"));
        READY_SET_CHECK(graph.markOutput("ComputeFinal.out0"));

        render::detail::ActiveGraph active;
        READY_SET_CHECK(buildCheckedOrder(graph, {}, active, log));
        // Graphics preference cannot move the already-ready second opaque pass
        // before the first opaque pass in the stable legal baseline.
        const std::vector<std::string> expected{"ComputeRoot", "OpaqueFirst", "OpaqueSecond",
            "GraphicsJoin", "ComputeFinal"};
        READY_SET_CHECK(active.executionOrder == expected);
        READY_SET_CHECK(!active.activePasses.contains("DormantRoot"));
        READY_SET_CHECK(!active.activePasses.contains("DormantTail"));
        READY_SET_CHECK(buildCheckedOrder(graph, {"DormantTail.out0", "DormantTail.out0"}, active, log));
        READY_SET_CHECK(active.activePasses.size() == expected.size() + 2);
        READY_SET_CHECK(active.activePasses.contains("DormantRoot"));
        READY_SET_CHECK(active.activePasses.contains("DormantTail"));
        const auto first = std::find(active.executionOrder.begin(), active.executionOrder.end(), "OpaqueFirst");
        const auto second = std::find(active.executionOrder.begin(), active.executionOrder.end(), "OpaqueSecond");
        READY_SET_CHECK(first < second);
        READY_SET_CHECK(graph.outputs().size() == 1);

        render::RenderGraph fallback;
        READY_SET_CHECK(fallback.addNode("ReadySetComputeProbe", "EligibleCompute"));
        auto* node = fallback.addNode("ReadySetComputeProbe", "FallbackCompute", {{"async", true}});
        READY_SET_CHECK(node);
        READY_SET_CHECK(fallback.setNodeRuntimeProperties(node->id, {{"async", false}}));
        READY_SET_CHECK(fallback.markOutput("EligibleCompute.out0"));
        READY_SET_CHECK(fallback.markOutput("FallbackCompute.out0"));
        READY_SET_CHECK(buildCheckedOrder(fallback, {}, active, log));
        // Runtime opt-out selects graphics affinity even though the pass declares Compute.
        READY_SET_CHECK((active.executionOrder == std::vector<std::string>{"FallbackCompute", "EligibleCompute"}));

        render::RenderGraph reversedIds;
        READY_SET_CHECK(reversedIds.addNode("ReadySetOpaqueGraphicsProbe", "OlderConsumer"));
        READY_SET_CHECK(reversedIds.addNode("ReadySetComputeProbe", "AuditedProducer"));
        READY_SET_CHECK(reversedIds.addNode("ReadySetOpaqueComputeProbe", "NewerProducer"));
        READY_SET_CHECK(reversedIds.addEdge("AuditedProducer.out0", "NewerProducer.in0"));
        READY_SET_CHECK(reversedIds.addEdge("NewerProducer.out0", "OlderConsumer.in0"));
        READY_SET_CHECK(reversedIds.markOutput("OlderConsumer.out0"));
        READY_SET_CHECK(buildCheckedOrder(reversedIds, {}, active, log));
        // Opaque order comes from a legal topology, not an unconditional ID chain.
        READY_SET_CHECK((active.executionOrder == std::vector<std::string>{
            "AuditedProducer", "NewerProducer", "OlderConsumer"}));
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(ReadySetGraphicsChainTest);
METALLIC_REGISTER_RHI_TEST(ReadySetFanoutDeterminismTest);
METALLIC_REGISTER_RHI_TEST(ReadySetOpaqueAndOutputsTest);

#undef READY_SET_CHECK

} // namespace
} // namespace metallic::tests
