#pragma once

#include "Runtime/Debug/DebugTypes.h"

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace metallic::debug {

struct DebugCaptureRequest {
    std::string id;
    std::string graph;
    uint64_t generation = 0;
    DebugValue specification;
    uint32_t groupRemaining = 1;
};

struct DebugArtifact {
    DebugValue metadata;
    DebugTypeDesc layout;
    std::vector<uint8_t> bytes;
};

struct DebugCapture {
    DebugSnapshot snapshot;
    std::vector<DebugArtifact> artifacts;
    DebugValue manifest() const;
    DebugResult<DebugValue> evaluationRoot() const;
    DebugResult<DebugValue> statistics() const;
};

// Domain decoding operates exclusively on this capture's copied values/bytes.
void addCaptureRelations(DebugValue& root);
DebugValue paginateDebugValue(DebugValue value, const DebugValue& params);

struct DebugJob {
    DebugCaptureRequest request;
    std::string state = "Queued";
    DebugError error;
    uint64_t reservedBytes = 0;
    std::chrono::steady_clock::time_point deadline;
    std::shared_ptr<const DebugCapture> capture;
};

class DebugCore {
public:
    explicit DebugCore(DebugLimits limits = {});
    const DebugLimits& limits() const { return limits_; }
    const std::string& session() const { return session_; }
    void setProcess(uint32_t pid, std::string startTime);
    void setSchema(std::string name, DebugValue schema);
    void setGraph(DebugValue graph);
    void setEngineState(DebugValue state);
    void pushEvent(std::string provider, DebugValue event);
    DebugValue events(std::string_view provider) const;
    void publish(DebugSnapshot snapshot);
    DebugValue dispatch(const DebugValue& request);

    // The runtime drains these only at a graph execution boundary. No callbacks
    // into the renderer exist in DebugCore.
    std::vector<DebugCaptureRequest> takeRequests(std::string_view graph, uint64_t generation);
    bool reserve(std::string_view id, uint64_t bytes);
    void transition(std::string_view id, std::string state);
    bool cancelled(std::string_view id) const;
    void fail(std::string_view id, DebugError error);
    void complete(std::string_view id, std::shared_ptr<const DebugCapture> capture);
    void expire();

private:
    DebugValue route(std::string_view method, const DebugValue& params);
    void expireLocked();
    void pruneLocked(uint64_t needed = 0);
    DebugLimits limits_;
    std::string session_;
    uint64_t nextJob_ = 1;
    uint64_t snapshotBytes_ = 0;
    uint64_t captureBytes_ = 0;
    mutable std::mutex mutex_;
    DebugValue process_;
    DebugValue schemas_ = DebugValue::object();
    DebugValue graph_ = DebugValue::object();
    DebugValue engineState_ = {{"state", "Starting"}};
    std::deque<std::pair<std::shared_ptr<const DebugSnapshot>, uint64_t>> snapshots_;
    std::unordered_map<std::string, DebugJob> jobs_;
    std::deque<std::string> order_;
    std::unordered_map<std::string, std::deque<DebugValue>> events_;
    std::unordered_map<std::string, uint64_t> droppedEvents_;
};

DebugValue debugErrorResponse(const DebugValue& id, std::string code, std::string message);

} // namespace metallic::debug
