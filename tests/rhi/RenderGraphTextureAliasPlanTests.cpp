#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraphTextureAliasPlan.h"

#include <algorithm>
#include <array>
#include <random>

namespace metallic::tests {
namespace {

using render::detail::GraphTextureAliasCandidate;
using render::detail::GraphTextureAliasDependency;
using render::detail::GraphTextureAliasPlan;
using render::detail::buildGraphTextureAliasPlan;
using render::detail::kNoGraphTextureAliasSlot;

#define ALIAS_CHECK(condition) do { \
    if (!(condition)) { return RHITestResult::fail(#condition); } \
} while (false)

GraphTextureAliasCandidate textureCandidate(size_t resource, std::vector<size_t> uses,
    uint64_t sizeBytes = 256, uint32_t memoryTypeBits = 7, uint64_t alignment = 64)
{
    return {.resource = resource, .name = "Texture" + std::to_string(resource),
        .uses = std::move(uses), .sizeBytes = sizeBytes, .alignment = alignment,
        .memoryTypeBits = memoryTypeBits, .eligible = true};
}

bool sameAllocationPlan(const GraphTextureAliasPlan& left, const GraphTextureAliasPlan& right)
{
    // Candidate-order mappings naturally differ after input permutation;
    // resource identities, grouping, handoffs and safety edges must not.
    if (left.slots.size() != right.slots.size() || left.handoffs.size() != right.handoffs.size() ||
        left.safetyDependencies != right.safetyDependencies) { return false; }
    for (size_t index = 0; index < left.slots.size(); ++index) {
        const auto& before = left.slots[index];
        const auto& after = right.slots[index];
        if (before.resources != after.resources || before.sizeBytes != after.sizeBytes ||
            before.alignment != after.alignment || before.memoryTypeBits != after.memoryTypeBits) { return false; }
    }
    for (size_t index = 0; index < left.handoffs.size(); ++index) {
        const auto& before = left.handoffs[index];
        const auto& after = right.handoffs[index];
        if (before.beforeResource != after.beforeResource || before.afterResource != after.afterResource ||
            before.predecessors != after.predecessors || before.activationPass != after.activationPass) { return false; }
    }
    return true;
}

class TextureAliasPlanTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::RenderGraph,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {std::string("graph.texture_alias.plan.") + name}};
    }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    virtual RHITestResult check() = 0;
};

class TextureAliasPlanFanoutTest final : public TextureAliasPlanTest {
public:
    TextureAliasPlanFanoutTest() { name = "render_graph_texture_alias_plan_fanout"; }
    RHITestResult check() override
    {
        const std::array candidates{
            textureCandidate(10, {2, 0, 1, 2}, 512),
            textureCandidate(20, {4, 3}, 128, 3, 256),
            textureCandidate(30, {5}, 64),
        };
        const std::array<GraphTextureAliasDependency, 6> edges{{
            {0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {0, 1},
        }};
        auto plan = buildGraphTextureAliasPlan(6, candidates, edges);
        ALIAS_CHECK(plan && plan->slots.size() == 2 && plan->handoffs.size() == 1);
        ALIAS_CHECK((plan->slots[0].resources == std::vector<size_t>{10, 20}));
        ALIAS_CHECK(plan->slots[0].sizeBytes == 512 && plan->slots[0].alignment == 256);
        ALIAS_CHECK(plan->slots[0].memoryTypeBits == 3);
        ALIAS_CHECK(plan->resourceSlots[0] == plan->resourceSlots[1]);
        ALIAS_CHECK(plan->resourceSlots[2] != plan->resourceSlots[0]);
        const auto& handoff = plan->handoffs.front();
        ALIAS_CHECK(handoff.beforeResource == 10 && handoff.afterResource == 20 && handoff.activationPass == 3);
        ALIAS_CHECK((handoff.predecessors == std::vector<size_t>{1, 2}));
        ALIAS_CHECK(plan->safetyDependencies.size() == 5);
        return RHITestResult::pass();
    }
};

class TextureAliasPlanConflictsTest final : public TextureAliasPlanTest {
public:
    TextureAliasPlanConflictsTest() { name = "render_graph_texture_alias_plan_conflicts"; }
    RHITestResult check() override
    {
        // Interleaved pass indices convey no cross-branch GPU order. The last
        // two resources also conflict because they are used by the same pass.
        const std::array candidates{
            textureCandidate(10, {0, 2}), textureCandidate(20, {1, 3}),
            textureCandidate(30, {4}), textureCandidate(40, {4}),
            GraphTextureAliasCandidate{.resource = 50, .name = "Persistent"},
        };
        const std::array<GraphTextureAliasDependency, 2> edges{{{0, 2}, {1, 3}}};
        auto plan = buildGraphTextureAliasPlan(5, candidates, edges);
        ALIAS_CHECK(plan && plan->slots.size() == 4 && plan->handoffs.empty());
        ALIAS_CHECK(plan->safetyDependencies.empty());
        ALIAS_CHECK(plan->resourceSlots[0] != plan->resourceSlots[1]);
        ALIAS_CHECK(plan->resourceSlots[2] != plan->resourceSlots[3]);
        ALIAS_CHECK(plan->resourceSlots[4] == kNoGraphTextureAliasSlot);
        return RHITestResult::pass();
    }
};

class TextureAliasPlanTypeIntersectionTest final : public TextureAliasPlanTest {
public:
    TextureAliasPlanTypeIntersectionTest() { name = "render_graph_texture_alias_plan_type_intersection"; }
    RHITestResult check() override
    {
        // Every pair has a compatible type, but the three-way intersection is
        // empty. A slot must intersect every member's requirements cumulatively.
        const std::array candidates{
            textureCandidate(10, {0}, 512, 3),
            textureCandidate(20, {1}, 256, 6),
            textureCandidate(30, {2}, 128, 5),
        };
        const std::array<GraphTextureAliasDependency, 2> edges{{{0, 1}, {1, 2}}};
        auto plan = buildGraphTextureAliasPlan(3, candidates, edges);
        ALIAS_CHECK(plan && plan->slots.size() == 2);
        ALIAS_CHECK((plan->slots[0].resources == std::vector<size_t>{10, 20}));
        ALIAS_CHECK(plan->slots[0].memoryTypeBits == 2);
        ALIAS_CHECK(plan->resourceSlots[2] != plan->resourceSlots[0]);
        return RHITestResult::pass();
    }
};

class TextureAliasPlanInvalidInputTest final : public TextureAliasPlanTest {
public:
    TextureAliasPlanInvalidInputTest() { name = "render_graph_texture_alias_plan_invalid_input"; }
    RHITestResult check() override
    {
        const std::array<GraphTextureAliasDependency, 2> cycle{{{0, 1}, {1, 0}}};
        const std::array<GraphTextureAliasDependency, 1> outside{{{0, 2}}};
        const std::array<GraphTextureAliasDependency, 1> self{{{0, 0}}};
        ALIAS_CHECK(render::hasError(buildGraphTextureAliasPlan(2, {}, cycle), render::Error::InvalidArgument));
        ALIAS_CHECK(render::hasError(buildGraphTextureAliasPlan(2, {}, outside), render::Error::InvalidArgument));
        ALIAS_CHECK(render::hasError(buildGraphTextureAliasPlan(1, {}, self), render::Error::InvalidArgument));
        auto empty = buildGraphTextureAliasPlan(0, {}, {});
        ALIAS_CHECK(empty && empty->slots.empty() && empty->resourceSlots.empty());
        const std::array duplicated{textureCandidate(10, {0}), textureCandidate(10, {1})};
        ALIAS_CHECK(render::hasError(buildGraphTextureAliasPlan(2, duplicated, {}), render::Error::InvalidArgument));
        for (uint32_t variant = 0; variant < 6; ++variant) {
            auto candidate = textureCandidate(10, {0});
            switch (variant) {
            case 0: candidate.sizeBytes = 0; break;
            case 1: candidate.alignment = 0; break;
            case 2: candidate.alignment = 3; break;
            case 3: candidate.memoryTypeBits = 0; break;
            case 4: candidate.uses.clear(); break;
            case 5: candidate.uses = {1}; break;
            }
            const std::array candidates{candidate};
            ALIAS_CHECK(render::hasError(buildGraphTextureAliasPlan(1, candidates, {}), render::Error::InvalidArgument));
        }
        return RHITestResult::pass();
    }
};

class TextureAliasPlanProducerTest final : public TextureAliasPlanTest {
public:
    TextureAliasPlanProducerTest() { name = "render_graph_texture_alias_plan_unique_producer"; }
    RHITestResult check() override
    {
        const std::array<GraphTextureAliasDependency, 2> edges{{{0, 1}, {0, 2}}};
        const std::array missingProducer{textureCandidate(10, {1, 2})};
        ALIAS_CHECK(render::hasError(buildGraphTextureAliasPlan(3, missingProducer, edges), render::Error::InvalidArgument));
        const std::array complete{textureCandidate(10, {2, 0, 1, 0})};
        auto valid = buildGraphTextureAliasPlan(3, complete, edges);
        ALIAS_CHECK(valid && valid->slots.size() == 1);
        // The caller's pass IDs need not themselves be a topological order.
        const std::array reverseCandidates{textureCandidate(10, {2, 1}), textureCandidate(20, {0})};
        const std::array<GraphTextureAliasDependency, 2> reverseEdges{{{2, 1}, {1, 0}}};
        auto reversed = buildGraphTextureAliasPlan(3, reverseCandidates, reverseEdges);
        ALIAS_CHECK(reversed && reversed->slots.size() == 1 && reversed->handoffs.size() == 1);
        ALIAS_CHECK(reversed->handoffs.front().activationPass == 0);
        ALIAS_CHECK((reversed->handoffs.front().predecessors == std::vector<size_t>{1}));
        return RHITestResult::pass();
    }
};

class TextureAliasPlanDeterminismTest final : public TextureAliasPlanTest {
public:
    TextureAliasPlanDeterminismTest() { name = "render_graph_texture_alias_plan_input_order"; }
    RHITestResult check() override
    {
        std::vector candidates{
            textureCandidate(10, {0}, 512), textureCandidate(20, {1}, 512),
            textureCandidate(30, {2}, 256), textureCandidate(40, {3}, 128),
            textureCandidate(50, {4}, 64),
        };
        std::vector<GraphTextureAliasDependency> edges{{0, 2}, {1, 3}, {2, 4}, {3, 4}};
        auto reference = buildGraphTextureAliasPlan(5, candidates, edges);
        ALIAS_CHECK(reference && reference->slots.size() == 2);
        std::mt19937 random(41581);
        for (uint32_t iteration = 0; iteration < 128; ++iteration) {
            std::shuffle(candidates.begin(), candidates.end(), random);
            std::shuffle(edges.begin(), edges.end(), random);
            auto plan = buildGraphTextureAliasPlan(5, candidates, edges);
            ALIAS_CHECK(plan && sameAllocationPlan(*reference, *plan));
            for (size_t index = 0; index < candidates.size(); ++index) {
                const size_t slot = plan->resourceSlots[index];
                ALIAS_CHECK(slot < plan->slots.size());
                const auto& members = plan->slots[slot].resources;
                ALIAS_CHECK(std::find(members.begin(), members.end(), candidates[index].resource) != members.end());
            }
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(TextureAliasPlanFanoutTest);
METALLIC_REGISTER_RHI_TEST(TextureAliasPlanConflictsTest);
METALLIC_REGISTER_RHI_TEST(TextureAliasPlanTypeIntersectionTest);
METALLIC_REGISTER_RHI_TEST(TextureAliasPlanInvalidInputTest);
METALLIC_REGISTER_RHI_TEST(TextureAliasPlanProducerTest);
METALLIC_REGISTER_RHI_TEST(TextureAliasPlanDeterminismTest);

#undef ALIAS_CHECK

} // namespace
} // namespace metallic::tests
