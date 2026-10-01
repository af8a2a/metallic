#pragma once
#include "Fixtures.h"
#include "TraceRecorder.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSynchronization.h"

namespace metallic::tests::bench {
inline Json graphScope(render::SyncScope scope)
{
    return {{"stages", uint64_t(scope.stages)}, {"access", uint64_t(scope.access)}};
}
// Snapshot of the first submitted frame; excludes observation timing/completion polling.
inline std::string graphEvidence(RHITestContext& context, render::RenderGraphExecutor& executor,
    const render::RenderGraphExecutionSnapshot& snapshot)
{
    if (!context.evidence) { return {}; }
    Json graph{{"schema", 1}, {"pipelined", snapshot.pipelinedSubmission},
        {"resources", Json::array()}, {"queues", Json::array()}, {"passes", Json::array()},
        {"segments", Json::array()}, {"batches", Json::array()}};
    Json links{{"resources", Json::array()}, {"queues", Json::array()}};
    for (const auto& resource : snapshot.resources) {
        graph["resources"].push_back({{"id", resource.id}, {"name", resource.name}, {"aliases", resource.aliases}, {"type", int(resource.type)}});
        if (context.trace) {
            auto* output = executor.outputResource(resource.name);
            if (output) {
                const auto id = output->buffer ? context.trace->bufferId(*output->buffer) : output->texture ? context.trace->textureId(*output->texture) : 0;
                links["resources"].push_back({{"graph", resource.id}, {"trace", id}, {"name", resource.name}});
            }
        }
    }
    for (const auto& queue : snapshot.queues) {
        graph["queues"].push_back({{"id", queue.id}, {"type", int(queue.type)}});
        if (context.trace) {
            auto* actual = context.device.getQueue(queue.type);
            if (!actual) { return "graph queue missing from device"; }
            links["queues"].push_back({{"graph", queue.id}, {"trace", context.trace->queueId(*actual)}});
        }
    }
    for (const auto& pass : snapshot.passes) {
        Json uses = Json::array(), barriers = Json::array();
        for (const auto& use : pass.uses) {
            uses.push_back({{"resource", use.resourceId}, {"state", int(use.state)}, {"scope", graphScope(use.scope)},
                {"reads", use.reads}, {"writes", use.writes}, {"exclusive", use.exclusive}});
        }
        for (const auto& barrier : pass.barriers) {
            barriers.push_back({{"resource", barrier.resourceId}, {"before", graphScope(barrier.beforeScope)},
                {"after", graphScope(barrier.afterScope)}, {"beforeState", int(barrier.before)}, {"afterState", int(barrier.after)},
                {"executionOnly", barrier.executionOnly}});
        }
        graph["passes"].push_back({{"id", pass.id}, {"name", pass.name}, {"queue", pass.actualQueueId},
            {"predecessors", pass.predecessors}, {"uses", uses}, {"barriers", barriers}});
    }
    for (const auto& segment : snapshot.segments) {
        graph["segments"].push_back({{"id", segment.id}, {"pass", segment.passId}, {"queue", segment.queueId},
            {"role", int(segment.role)}, {"predecessors", segment.predecessors}, {"accepted", segment.accepted}});
    }
    for (const auto& batch : snapshot.batches) {
        graph["batches"].push_back({{"id", batch.id}, {"queue", batch.queueId}, {"segments", batch.segmentIds},
            {"waitPredecessors", batch.waitPredecessors}, {"externalWaits", batch.externalWaitCount},
            {"explicitWaits", batch.semaphoreWaitCount}, {"waitDetailsComplete", batch.waitDetailsComplete}, {"accepted", batch.accepted}});
    }
    context.evidence->json("graph.json", graph);
    if (!context.trace) { return {}; }
    context.evidence->json("graph-trace-links.json", links);
    const auto trace = context.trace->snapshot();
    context.evidence->json("graph-trace.json", trace);
    if (trace.at("captureFailed").get<bool>()) { return "backend trace interrupted"; }
    // The copy branch has no graph predecessor and must not acquire a graphics wait
    // after Queue::submit merges opaque command/frame dependencies.
    auto* copy = context.device.getQueue(render::QueueType::Copy);
    const auto copyId = context.trace->queueId(*copy);
    bool found = false;
    for (const auto& event : trace.at("events")) {
        if (event.at("kind") != "submit" || event.at("queue").get<uint64_t>() != copyId) { continue; }
        if (event.at("result") != int(VK_SUCCESS) || !event.at("waits").empty()) { return "encoded independent copy submission acquired waits/failed"; }
        found = true; break;
    }
    if (!found) { return "copy submission missing from trace"; }
    // Correlate planned resource/scope boundaries with actual requested and encoded
    // barriers. Buffer scopes may legitimately be coalesced into global memory scopes.
    size_t covered = 0;
    for (const auto& pass : graph.at("passes")) {
        for (const auto& planned : pass.at("barriers")) {
            uint64_t resourceId = 0;
            for (const auto& link : links.at("resources")) {
                if (link.at("graph") == planned.at("resource")) { resourceId = link.at("trace").get<uint64_t>(); }
            }
            if (!resourceId) { return "planned resource lacks trace identity"; }
            bool encoded = false;
            for (const auto& event : trace.at("events")) {
                if (event.at("kind") != "barrier") { continue; }
                for (const auto& requested : event.at("requested")) {
                    if (requested.value("resource", uint64_t(0)) != resourceId || requested.at("before") != planned.at("before") ||
                        requested.at("after") != planned.at("after")) { continue; }
                    const auto encode = [](const Json& value) {
                        return render::vulkan::scopeInfo({render::PipelineStageBits(value.at("stages").get<uint64_t>()),
                            render::AccessBits(value.at("access").get<uint64_t>())});
                    };
                    const auto before = encode(requested.at("before")), after = encode(requested.at("after"));
                    for (const auto* kind : {"memory", "buffers", "images"}) {
                        for (const auto& actual : event.at(kind)) {
                            const auto covers = [](const Json& scope, const render::vulkan::VulkanSyncScope& required) {
                                return (scope.at("stages").get<uint64_t>() & required.stage) == required.stage &&
                                    (scope.at("access").get<uint64_t>() & required.access) == required.access;
                            };
                            if (actual.contains("resource") && actual.at("resource").get<uint64_t>() != resourceId) { continue; }
                            encoded |= covers(actual.at("before"), before) && covers(actual.at("after"), after);
                        }
                    }
                }
            }
            if (!encoded) { return "planned barrier missing from backend encoding"; }
            ++covered;
        }
    }
    context.evidence->json("graph-trace-checks.json", {{"independentCopyHasNoWait", true}, {"plannedBarriersCovered", covered}});
    return covered ? std::string{} : "no planned barriers checked";
}
} // namespace metallic::tests::bench
