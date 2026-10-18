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
    struct State;
public:
    class SaveDeferral {
    public:
        ~SaveDeferral() noexcept;
        SaveDeferral(const SaveDeferral&) = delete;
        SaveDeferral& operator=(const SaveDeferral&) = delete;
        SaveDeferral(SaveDeferral&& other) noexcept;
        SaveDeferral& operator=(SaveDeferral&& other) noexcept;

    private:
        friend class DeferredShaderCacheWriter;
        SaveDeferral(State* state, std::shared_ptr<void> owner);
        void reset() noexcept;
        State* state_ = nullptr;
        std::shared_ptr<void> owner_;
    };

    explicit DeferredShaderCacheWriter(
        std::chrono::milliseconds quietPeriod = std::chrono::milliseconds(750),
        std::chrono::milliseconds maximumDelay = std::chrono::seconds(5));
    ~DeferredShaderCacheWriter();

    DeferredShaderCacheWriter(const DeferredShaderCacheWriter&) = delete;
    DeferredShaderCacheWriter& operator=(const DeferredShaderCacheWriter&) = delete;

    // Revisions must increase per key. Repeated or older revisions do not
    // postpone a pending save or enqueue an already saved revision again.
    void request(std::string key, uint64_t revision, std::function<bool()> save);

    // Nestable automatic-save barrier. Entry waits for an active callback;
    // final release schedules pending keys immediately with their latest revision.
    // The writer must outlive the scope. Pass its containing owner to retain
    // that lifetime when the scope can escape the caller. Do not call from a
    // save callback. Explicit flush() overrides this barrier.
    [[nodiscard]] SaveDeferral deferSaves(std::shared_ptr<void> owner = {});

    // Attempt all pending work immediately and wait for callbacks to finish.
    // Failures stay queued for a later retry and are attempted only once by
    // each flush. Callbacks must not call flush()
    // on their own writer. Explicit flush overrides active save deferrals.
    void flush();

private:
    std::unique_ptr<State> state_;
};

} // namespace metallic::render
