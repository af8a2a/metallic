#include "Runtime/Debug/DebugCore.h"
#include "Runtime/Debug/DebugTransport.h"
#include "Runtime/Debug/DebugProbe.h"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <thread>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace metallic::debug;

TEST(DebugEval, TypedPredicatesAndIntegerPrecision)
{
    DebugValue root{{"records", {{{"id", 0u}}, {{"id", 9u}}, {{"id", 2u}}}}, {"limit", 3u}, {"large", UINT64_MAX}};
    auto result = evaluate("count(records, x => x.id >= limit)", root);
    ASSERT_TRUE(result); EXPECT_EQ(*result, 1u);
    result = evaluate("findFirst(records, x => x.id >= limit)", root);
    ASSERT_TRUE(result); EXPECT_EQ(*result, 1u);
    result = evaluate("large == 18446744073709551615", root);
    ASSERT_TRUE(result); EXPECT_EQ(*result, true);
    EXPECT_EQ(evaluate("large + 1", root).error().code, "Overflow");
    EXPECT_EQ(evaluate("large == 18446744073709551615.0", root).error().code, "PrecisionLoss");
    EXPECT_EQ(evaluate("records[3]", root).error().code, "OutOfRange");
    EXPECT_EQ(evaluate("system(1)", root).error().code, "Unsupported");
    EXPECT_FALSE(evaluate("records[0].id = 17", root));
    EXPECT_FALSE(evaluate("1 / 0", root));
    EXPECT_EQ(*evaluate("false && missing.field", root), false);
    EXPECT_EQ(*evaluate("-9223372036854775808 < -1", root), true);
    EXPECT_EQ(*evaluate("2 + 3 * 4", root), 14u);
    EXPECT_EQ(evaluate("count(records)", root, 1).error().code, "BudgetExceeded");
}

TEST(DebugTypes, LosslessWireAndLayouts)
{
    DebugValue value{{"unsigned", UINT64_MAX}, {"signed", INT64_MIN}, {"nan", std::numeric_limits<double>::quiet_NaN()}};
    const auto decoded = decodeLossless(DebugValue::parse(encodeLossless(value).dump()));
    EXPECT_EQ(decoded["unsigned"].get<uint64_t>(), UINT64_MAX);
    EXPECT_EQ(decoded["signed"].get<int64_t>(), INT64_MIN);
    EXPECT_TRUE(std::isnan(decoded["nan"].get<double>()));
    DebugTypeDesc layout{"Test", 8, {{"id", "u32", 0}, {"flag", "u32", 4}}};
    const uint32_t words[] = {42, 1};
    auto rows = decodeBuffer({reinterpret_cast<const uint8_t*>(words), sizeof(words)}, layout);
    ASSERT_TRUE(rows); EXPECT_EQ((*rows)[0]["id"], 42);
    const auto hash = layout.layoutHash(); layout.fields[1].offset = 8;
    EXPECT_NE(hash, layout.layoutHash());
    EXPECT_FALSE(decodeBuffer({reinterpret_cast<const uint8_t*>(words), sizeof(words)}, layout));
}

static DebugValue graph(uint64_t generation = 1)
{
    return {{"id", "graph"}, {"generation", generation}, {"passes", {{{"name", "Cull"}, {"active", true}, {"checkpoints", {"AfterPass"}}}}}};
}

static DebugValue request(DebugCore& core, std::string method, DebugValue params = DebugValue::object())
{
    return core.dispatch({{"id", "test"}, {"method", method}, {"params", params}});
}

TEST(DebugCore, CompletionAndStaleGeneration)
{
    DebugCore core; core.setGraph(graph());
    EXPECT_EQ(request(core, "frame.latest")["error"]["code"], "NotCaptured");
    DebugSnapshot snapshot; snapshot.evidence.execution = 4; snapshot.values = {{"frame", {{"id", 4}}}};
    core.publish(snapshot);
    EXPECT_EQ(request(core, "eval", {{"expression", "frame.id"}})["result"]["value"], 4);
    auto capture = request(core, "capture.batch", {{"pass", "Cull"}, {"resources", {{{"id", "ids"}, {"count", 4}}}}});
    ASSERT_EQ(capture["status"], "ok");
    core.setGraph(graph(2));
    auto job = request(core, "jobs.get", {{"job", capture["result"]["job"]}});
    EXPECT_EQ(job["result"]["state"], "Failed");
    EXPECT_EQ(job["result"]["error"]["code"], "StaleHandle");
}

TEST(DebugCore, CancellationRetainsReservationUntilGpuCompletion)
{
    DebugLimits limits; limits.capturePoolBytes = 16; limits.jobBytes = 16;
    DebugCore core(limits); core.setGraph(graph());
    const auto first = request(core, "capture.batch", {{"pass", "Cull"}, {"resources", {{{"id", "ids"}}}}})["result"]["job"].get<std::string>();
    ASSERT_EQ(core.takeRequests("graph", 1).size(), 1u);
    ASSERT_TRUE(core.reserve(first, 16));
    core.transition(first, "Submitted");
    request(core, "jobs.cancel", {{"job", first}});
    const auto second = request(core, "capture.batch", {{"pass", "Cull"}, {"resources", {{{"id", "ids"}}}}})["result"]["job"].get<std::string>();
    core.takeRequests("graph", 1);
    EXPECT_FALSE(core.reserve(second, 16));
    core.complete(first, std::make_shared<DebugCapture>());
    EXPECT_TRUE(core.reserve(second, 16));
}

TEST(DebugCore, BoundedQueueGroupPaginationAndTimeout)
{
    DebugLimits limits; limits.queueCount = 3; limits.commandsPerFrame = 2; limits.snapshotCount = 2;
    DebugCore core(limits); core.setGraph(graph());
    const DebugValue specification{{"pass", "Cull"}, {"resources", {{{"id", "ids"}}}}};
    ASSERT_EQ(request(core, "capture.batch", specification)["status"], "ok");
    auto group = request(core, "capture.batch", {{"batches", {specification, specification}}});
    ASSERT_EQ(group["status"], "ok");
    EXPECT_EQ(request(core, "capture.batch", specification)["error"]["code"], "QueueFull");
    EXPECT_EQ(core.takeRequests("graph", 1).size(), 1u); // Never split a group.
    EXPECT_EQ(core.takeRequests("graph", 1).size(), 2u);
    for (uint64_t i = 1; i <= 3; ++i) {
        DebugSnapshot snapshot; snapshot.evidence.execution = i; snapshot.evidence.sample = i;
        snapshot.values = {{"a", {10u, 20u, 30u}}, {"n", i}}; core.publish(std::move(snapshot));
    }
    EXPECT_EQ(request(core, "eval", {{"expression", "n"}, {"frame", 1}})["error"]["code"], "NotCaptured");
    const auto page = request(core, "object.get", {{"path", "a"}, {"offset", 1}, {"count", 1}})["result"]["value"];
    EXPECT_EQ(page["items"], DebugValue::array({20})); EXPECT_EQ(page["truncated"], true);
    EXPECT_EQ(request(core, "eval", {{"expression", "n"}, {"frame", 2}})["result"]["source"], "frame");
    EXPECT_EQ(request(core, "object.get", {{"path", "a"}, {"offset", -1}})["error"]["code"], "InvalidArgument");

    DebugCore timeoutCore; timeoutCore.setGraph(graph());
    auto timed = specification; timed["timeoutMs"] = 1;
    const auto job = request(timeoutCore, "capture.batch", timed)["result"]["job"];
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    EXPECT_EQ(request(timeoutCore, "jobs.get", {{"job", job}})["result"]["error"]["code"], "Timeout");
    auto invalid = specification; invalid["checkpoint"] = "Unregistered";
    EXPECT_EQ(request(timeoutCore, "capture.batch", {{"batches", {specification, invalid}}})["status"], "error");
    EXPECT_TRUE(timeoutCore.takeRequests("graph", 1).empty()); // Atomic enqueue preflight.
}

TEST(DebugTypes, PackedFieldsAndDecodeWorkBudget)
{
    DebugTypeDesc layout{"Page", 4, {{"state", "u32", 0, 1, 0, 3, 1, {{"2", "Resident"}}}, {"offset", "u32", 0, 1, 3, 29, 8}}};
    const uint32_t word = 512u | 2u;
    auto result = decodeBuffer({reinterpret_cast<const uint8_t*>(&word), 4}, layout);
    ASSERT_TRUE(result); EXPECT_EQ((*result)[0]["state"]["name"], "Resident"); EXPECT_EQ((*result)[0]["offset"], 512u);
    EXPECT_THROW(debugUnsigned(-1), std::invalid_argument);
    EXPECT_THROW(debugUnsigned(1ull << 32, UINT32_MAX), std::invalid_argument);
    EXPECT_THROW(debugUnsigned(1.5), std::invalid_argument);
    DebugTypeDesc scalar{"u32", 4, {{"value", "u32", 0}}};
    EXPECT_EQ(decodeBuffer(std::vector<uint8_t>((1048576 + 1) * 4), scalar).error().code, "BudgetExceeded");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(evaluate("min(a)", {{"a", {1.0, nan}}})->get<double>()));
    EXPECT_EQ(evaluate("x + 1.0", {{"x", UINT64_MAX}}).error().code, "PrecisionLoss");
}

TEST(DebugCapture, ScopedRelationsMissingDependenciesAndNonFiniteStatistics)
{
    DebugCapture capture;
    capture.snapshot.values = {{"gpuScene", {{"available", true}, {"stats", {{"instanceCount", 1}, {"geometryCount", 1}}},
        {"instances", {{{"index", 0u}, {"geometryIndex", 0u}}}}, {"geometries", {{{"index", 0u}}}}}}};
    const auto add = [&](std::string id, DebugTypeDesc layout, const void* data, size_t size, uint64_t offset = 0) {
        DebugArtifact artifact;
        artifact.metadata = {{"id", id}, {"pass", "P"}, {"layout", layout.name}, {"elementOffset", offset}, {"capacity", 16u}, {"completeCoverage", false}};
        artifact.layout = std::move(layout); artifact.bytes.resize(size); std::memcpy(artifact.bytes.data(), data, size);
        capture.artifacts.push_back(std::move(artifact));
    };
    const uint32_t records[] = {1, 0, 0, 0, 2, 9, 7, 1u << 28};
    add("streaming.P.visibleClusters", {"VisibleClusterRecord", 16, {{"clusterIndex", "u32", 0}, {"instanceIndex", "u32", 4}, {"dataIndex", "u32", 8}, {"flags", "u32", 12}}}, records, sizeof(records));
    const uint32_t header[] = {3, 2, 32, 1, 99};
    add("streaming.P.activeHeader", {"MeshletStreamGpuActiveHeader", 20, {{"activeGroupCount", "u32", 0}, {"activeGroupCapacity", "u32", 4}, {"maxActiveGroupClusters", "u32", 8}, {"overflowCount", "u32", 12}, {"frameIndex", "u32", 16}}}, header, sizeof(header));
    auto root = capture.evaluationRoot(); ASSERT_TRUE(root);
    const auto& links = root->at("links").at("streaming.P.visibleClusters").at("items");
    EXPECT_EQ(links[0]["source"], "Resident"); EXPECT_EQ(links[0]["geometry"]["status"], "Captured");
    EXPECT_EQ(links[0]["meshlet"]["status"], "MissingDependency");
    EXPECT_EQ(links[1]["source"], "StreamPage"); EXPECT_EQ(links[1]["activeGroup"]["status"], "MissingDependency");
    EXPECT_EQ(links[1]["instance"]["status"], "OutOfRange");
    EXPECT_EQ(root->at("diagnostics")[0]["code"], "CounterExceedsCapacity");
    const float values[] = {1, 3, INFINITY, NAN};
    add("samples", {"f32", 4, {{"value", "f32", 0}}}, values, sizeof(values), 4);
    const auto stats = capture.statistics(); ASSERT_TRUE(stats);
    EXPECT_EQ((*stats)["samples"]["fields"]["value"]["mean"], 2.0);
    EXPECT_EQ((*stats)["samples"]["fields"]["value"]["nanCount"], 1);
    EXPECT_EQ((*stats)["samples"]["fields"]["value"]["infCount"], 1);
    EXPECT_EQ((*stats)["samples"]["coverage"]["elementOffset"], 4);
    EXPECT_EQ((*stats)["samples"]["coverage"]["completeCoverage"], false);
}

#ifdef _WIN32
TEST(DebugTransport, RoundTripSessionAndShutdown)
{
    DebugCore core;
    DebugServer server(core);
    auto started = server.start(); ASSERT_TRUE(started) << started.error().message;
    auto reply = debugRequest(GetCurrentProcessId(), {{"method", "hello"}});
    ASSERT_TRUE(reply) << reply.error().message;
    ASSERT_EQ((*reply)["status"], "ok");
    EXPECT_EQ((*reply)["result"]["session"], core.session());
    reply = debugRequest(GetCurrentProcessId(), {{"method", "hello"}, {"session", "wrong"}});
    ASSERT_TRUE(reply); EXPECT_EQ((*reply)["error"]["code"], "StaleSession");
    EXPECT_EQ(debugRequest(GetCurrentProcessId(), {{"method", "eval"}, {"padding", std::string(1u << 20, 'x')}}).error().code, "ProtocolError");
    const std::wstring path = L"\\\\.\\pipe\\Metallic.Debug." + std::to_wstring(GetCurrentProcessId());
    ASSERT_TRUE(WaitNamedPipeW(path.c_str(), 1000));
    const HANDLE abandoned = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    ASSERT_NE(abandoned, INVALID_HANDLE_VALUE);
    CloseHandle(abandoned); // A disconnected client must not terminate the server.
    reply = debugRequest(GetCurrentProcessId(), {{"method", "hello"}}, 1000);
    ASSERT_TRUE(reply); EXPECT_EQ((*reply)["status"], "ok");
    server.stop();
}
#endif


TEST(DebugProbe, ProtocolAndBoundedWatchLifecycle)
{
    DebugCore core;
    core.setGraph(graph());
    DebugValue spec{{"pass", "Cull"}, {"probes", {{{"id", "indices"}, {"name", "bounds"},
        {"operation", "outOfBounds"}, {"upper", 4}, {"count", 8}}}}};
    auto bad = spec;
    bad["probes"][0]["operation"] = "arbitraryShader";
    EXPECT_EQ(request(core, "gpu.probe", bad)["error"]["code"], "Unsupported");
    EXPECT_EQ(request(core, "capture.batch", spec)["error"]["code"], "InvalidArgument");
    bad = spec; bad["probes"][0]["count"] = -1;
    EXPECT_EQ(request(core, "gpu.probe", bad)["error"]["code"], "InvalidArgument");
    bad = spec; bad["probes"].push_back(bad["probes"][0]);
    EXPECT_EQ(request(core, "gpu.probe", bad)["error"]["code"], "InvalidArgument");
    const DebugValue config{{"probe", spec}, {"everyExecutions", 3}, {"maxSamples", 3},
        {"trigger", {{"probe", "bounds"}, {"field", "matchedCount"}, {"op", "gt"}, {"value", 0}}}};
    auto response = request(core, "watch.create", config);
    ASSERT_EQ(response["status"], "ok") << response.dump();
    const auto id = response["result"]["watch"];
    auto pending = core.takeRequests("graph", 1);
    ASSERT_EQ(pending.size(), 1);
    const auto first = pending[0].id;
    EXPECT_TRUE(core.reserve(first, 32));
    auto capture = std::make_shared<DebugCapture>();
    capture->snapshot.values["probes"]["bounds"] = {{"matchedCount", 0}};
    core.complete(first, capture);
    EXPECT_EQ(request(core, "jobs.get", {{"job", first}})["result"]["reservedBytes"], 0);
    EXPECT_TRUE(core.takeRequests("graph", 1).empty());
    EXPECT_TRUE(core.takeRequests("graph", 1).empty());
    pending = core.takeRequests("graph", 1);
    ASSERT_EQ(pending.size(), 1);
    const auto second = pending[0].id;
    auto hit = std::make_shared<DebugCapture>();
    hit->snapshot.evidence.execution = 4;
    hit->snapshot.values["probes"]["bounds"] = {{"matchedCount", 2}};
    EXPECT_TRUE(core.reserve(second, 32)); core.complete(second, hit);
    response = request(core, "watch.get", {{"watch", id}});
    EXPECT_EQ(response["result"]["state"], "Triggered");
    EXPECT_EQ(response["result"]["samples"], 2);
    EXPECT_EQ(response["result"]["result"]["evidence"]["execution"], 4);
    EXPECT_EQ(response["result"]["job"], second);
    EXPECT_TRUE(core.takeRequests("graph", 1).empty());
    EXPECT_EQ(request(core, "watch.delete", {{"watch", id}})["status"], "ok");
    EXPECT_TRUE(request(core, "watch.list")["result"].empty());
    response = request(core, "watch.create", config);
    const auto stale = response["result"]["watch"];
    core.setGraph(graph(2));
    EXPECT_EQ(request(core, "watch.get", {{"watch", stale}})["result"]["result"]["reason"], "StaleHandle");
    response = request(core, "watch.create", config);
    const auto cancelId = response["result"]["watch"];
    pending = core.takeRequests("graph", 2); ASSERT_EQ(pending.size(), 1);
    EXPECT_TRUE(core.reserve(pending[0].id, 64));
    request(core, "watch.cancel", {{"watch", cancelId}});
    EXPECT_EQ(request(core, "jobs.get", {{"job", pending[0].id}})["result"]["reservedBytes"], 64);
    core.complete(pending[0].id, nullptr);
    EXPECT_EQ(request(core, "jobs.get", {{"job", pending[0].id}})["result"]["reservedBytes"], 0);
    auto one = config; one["maxSamples"] = 1;
    response = request(core, "watch.create", one);
    const auto limited = response["result"]["watch"];
    pending = core.takeRequests("graph", 2); ASSERT_EQ(pending.size(), 1);
    core.complete(pending[0].id, capture);
    EXPECT_EQ(request(core, "watch.get", {{"watch", limited}})["result"]["state"], "Completed");
    EXPECT_TRUE(core.takeRequests("graph", 2).empty());
    one["timeoutMs"] = 1;
    response = request(core, "watch.create", one);
    const auto timed = response["result"]["watch"];
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    EXPECT_EQ(request(core, "watch.get", {{"watch", timed}})["result"]["state"], "Expired");
}

TEST(DebugProbe, PartialMergeAndOfflineEvaluation)
{
    const uint32_t words[] = {2, 1, 1, 0, 0xc0200000, 0x40a00000, 2, 0x7fc00000,
                             1, 2, 0, 2, 0, 0, 3, 0x7f800000};
    DebugArtifact artifact;
    artifact.layout = probePartialLayout();
    artifact.bytes.assign(reinterpret_cast<const uint8_t*>(words), reinterpret_cast<const uint8_t*>(words) + sizeof(words));
    artifact.metadata = {{"id", "probe.finite"}, {"name", "finite"}, {"kind", "gpuProbe"}, {"source", "FloatBuffer"},
        {"operation", "nonFinite"}, {"scalarType", "f32"}, {"field", "value"}, {"elementCount", 6}, {"elementOffset", 10},
        {"configuration", DebugValue::object()}, {"coverage", {{"completeCoverage", false}}}};
    auto summary = summarizeProbe(artifact.bytes, artifact.metadata);
    ASSERT_TRUE(summary);
    EXPECT_EQ((*summary)["matchedCount"], 3); EXPECT_EQ((*summary)["finiteCount"], 3);
    EXPECT_EQ((*summary)["min"], -2.5); EXPECT_EQ((*summary)["max"], 5.0);
    EXPECT_EQ((*summary)["firstIndex"], 12); EXPECT_TRUE(std::isnan((*summary)["firstValue"].get<double>()));
    DebugCapture capture; capture.artifacts.push_back(artifact);
    const auto root = capture.evaluationRoot(); ASSERT_TRUE(root);
    EXPECT_EQ(*evaluate("probes.finite.matchedCount", *root), 3);
    EXPECT_FALSE(root->at("buffers").contains("FloatBuffer"));
    EXPECT_EQ(decodeLossless(encodeLossless(capture.manifest()))["artifacts"][0]["kind"], "gpuProbe");
    artifact.metadata["elementCount"] = 7;
    EXPECT_EQ(summarizeProbe(artifact.bytes, artifact.metadata).error().code, "InvalidProbeResult");
}
