#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace metallic::render {

// Callbacks run on one owned worker. The caller must retain their resources
// until this writer has been destroyed, including callbacks retained after a
// failed flush. Declare the writer after the caches it accesses so that the
// writer drains and joins before those caches are destroyed.
class DeferredShaderCacheWriter {
public:
    explicit DeferredShaderCacheWriter(
        std::chrono::milliseconds quietPeriod = std::chrono::milliseconds(750),
        std::chrono::milliseconds maximumDelay = std::chrono::seconds(5));
    ~DeferredShaderCacheWriter();

    DeferredShaderCacheWriter(const DeferredShaderCacheWriter&) = delete;
    DeferredShaderCacheWriter& operator=(const DeferredShaderCacheWriter&) = delete;

    // Revisions must increase per key. Repeated or older revisions do not
    // postpone a pending save or enqueue an already saved revision again.
    void request(std::string key, uint64_t revision, std::function<bool()> save);

    // Attempt all pending work immediately and wait for callbacks to finish.
    // Failures stay queued for a later retry (at least one second apart), but
    // are attempted only once by each flush. Callbacks must not call flush()
    // on their own writer.
    void flush();

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace metallic::render
