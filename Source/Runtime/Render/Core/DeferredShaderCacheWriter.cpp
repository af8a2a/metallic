#include "Runtime/Render/Core/DeferredShaderCacheWriter.h"

#include <algorithm>
#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>

namespace metallic::render {

struct DeferredShaderCacheWriter::State {
    using Clock = std::chrono::steady_clock;

    struct Entry {
        uint64_t revision = 0;
        uint64_t generation = 0;
        uint64_t attemptedFlush = 0;
        std::function<bool()> save;
        Clock::time_point firstRequest;
        Clock::time_point deadline;
        bool pending = false;
    };

    State(std::chrono::milliseconds quiet, std::chrono::milliseconds maximum)
        : quietPeriod(std::max(quiet, std::chrono::milliseconds::zero()))
        , maximumDelay(std::max(maximum, std::chrono::milliseconds::zero()))
        , retryPeriod(std::max(quietPeriod, std::chrono::milliseconds(1000)))
        , worker([this] { run(); })
    {}

    void request(std::string key, uint64_t revision, std::function<bool()> save)
    {
        if (!save) { return; }
        std::lock_guard lock(mutex);
        if (closing) { return; }
        const auto existing = entries.find(key);
        if (existing != entries.end() && revision <= existing->second.revision) { return; }
        auto& entry = entries[std::move(key)];
        const auto now = Clock::now();
        if (!entry.pending) { entry.firstRequest = now; }
        entry.revision = revision;
        ++entry.generation;
        entry.attemptedFlush = 0;
        entry.save = std::move(save);
        entry.deadline = std::min(now + quietPeriod, entry.firstRequest + maximumDelay);
        entry.pending = true;
        wake.notify_one();
    }

    bool drained(uint64_t ticket) const
    {
        if (inFlight) { return false; }
        for (const auto& [key, entry] : entries) {
            if (entry.pending && entry.attemptedFlush < ticket) { return false; }
        }
        return true;
    }

    void flush()
    {
        std::unique_lock lock(mutex);
        const uint64_t ticket = ++flushTicket;
        ++activeFlushes;
        wake.notify_one();
        finished.wait(lock, [this, ticket] { return drained(ticket); });
        --activeFlushes;
    }

    void shutdown()
    {
        {
            std::lock_guard lock(mutex);
            closing = true;
        }
        flush();
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        wake.notify_one();
        worker.join();
    }

    void run()
    {
        std::unique_lock lock(mutex);
        while (!stopping) {
            const auto now = Clock::now();
            auto selected = entries.end();
            auto nextDeadline = Clock::time_point::max();
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                const auto& entry = it->second;
                if (!entry.pending) { continue; }
                const bool forced = activeFlushes != 0 && entry.attemptedFlush < flushTicket;
                if (forced || (!closing && entry.deadline <= now)) {
                    if (selected == entries.end() || entry.deadline < selected->second.deadline) {
                        selected = it;
                    }
                }
                if (!closing) { nextDeadline = std::min(nextDeadline, entry.deadline); }
            }
            if (selected == entries.end()) {
                if (nextDeadline == Clock::time_point::max()) {
                    wake.wait(lock);
                } else {
                    wake.wait_until(lock, nextDeadline);
                }
                continue;
            }

            auto& entry = selected->second;
            const uint64_t generation = entry.generation;
            auto save = std::move(entry.save);
            entry.pending = false;
            inFlight = true;
            lock.unlock();
            bool saved = false;
            try {
                saved = save();
            } catch (...) {
                // Cache persistence is optional. A bad callback must not stop
                // other saves or terminate the worker during device teardown.
            }
            lock.lock();
            inFlight = false;
            if (!saved && entry.generation == generation) {
                entry.save = std::move(save);
                entry.pending = true;
                entry.firstRequest = Clock::now();
                entry.deadline = entry.firstRequest + retryPeriod;
                // A flush arriving during this callback counts this attempt.
                entry.attemptedFlush = flushTicket;
            }
            // A newer request made during this callback remains pending with
            // its own deadline, callback, and unattempted flush generation.
            finished.notify_all();
        }
    }

    const std::chrono::milliseconds quietPeriod;
    const std::chrono::milliseconds maximumDelay;
    const std::chrono::milliseconds retryPeriod;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable finished;
    std::map<std::string, Entry> entries;
    uint64_t flushTicket = 0;
    size_t activeFlushes = 0;
    bool inFlight = false;
    bool closing = false;
    bool stopping = false;
    std::thread worker;
};

DeferredShaderCacheWriter::DeferredShaderCacheWriter(
    std::chrono::milliseconds quietPeriod, std::chrono::milliseconds maximumDelay)
    : state_(std::make_unique<State>(quietPeriod, maximumDelay))
{}

DeferredShaderCacheWriter::~DeferredShaderCacheWriter()
{
    state_->shutdown();
}

void DeferredShaderCacheWriter::request(std::string key, uint64_t revision, std::function<bool()> save)
{
    state_->request(std::move(key), revision, std::move(save));
}

void DeferredShaderCacheWriter::flush()
{
    state_->flush();
}

} // namespace metallic::render
