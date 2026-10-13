#include "Runtime/Render/Core/NativePipelineCache.h"

#include <chrono>
#include <future>
#include <list>
#include <mutex>
#include <unordered_map>

namespace metallic::render::detail {

struct NativePipelineCache::State {
    struct KeyHash {
        size_t operator()(NativePipelineKey key) const
        {
            return std::hash<uint64_t>{}(key.psoHash) ^ (static_cast<size_t>(key.type) << 1);
        }
    };

    using Owner = std::shared_ptr<void>;
    using Created = Result<Owner>;
    struct Resident {
        NativePipelineKey key;
        Owner owner;
    };
    using Residents = std::list<Resident>;
    struct Ready {
        std::weak_ptr<void> owner;
        Residents::iterator resident;
        bool retained = false;
    };
    struct Flight {
        std::promise<Created> promise;
        std::shared_future<Created> future = promise.get_future().share();
    };

    explicit State(size_t capacity) { counters.strongCapacity = capacity; }

    void pruneExpiredLocked()
    {
        std::erase_if(ready, [](const auto& item) { return item.second.owner.expired(); });
    }

    void trimLocked(Residents& retired)
    {
        while (residents.size() > counters.strongCapacity) {
            const auto victim = std::prev(residents.end());
            ready.at(victim->key).retained = false;
            retired.splice(retired.end(), residents, victim);
            ++counters.evictionCount;
        }
    }

    void touchLocked(NativePipelineKey key, Ready& entry, const Owner& owner, Residents& retired)
    {
        if (entry.retained) {
            residents.splice(residents.begin(), residents, entry.resident);
        } else if (counters.strongCapacity) {
            residents.push_front({key, owner});
            entry.resident = residents.begin();
            entry.retained = true;
        }
        trimLocked(retired);
    }

    mutable std::mutex mutex;
    NativePipelineCacheStats counters;
    std::unordered_map<NativePipelineKey, Ready, KeyHash> ready;
    std::unordered_map<NativePipelineKey, std::shared_ptr<Flight>, KeyHash> inFlight;
    Residents residents;
};

NativePipelineCache::NativePipelineCache(size_t strongCapacity)
    : state_(std::make_unique<State>(strongCapacity))
{}

NativePipelineCache::~NativePipelineCache() = default;

Result<NativePipelineAcquisition> NativePipelineCache::acquire(NativePipelineKey key,
    const std::function<Result<std::shared_ptr<void>>()>& create)
{
    if (!create) { return makeError(Error::InvalidArgument); }
    State::Residents retired;
    State::Owner owner;
    std::shared_ptr<State::Flight> flight;
    bool creator = false;
    {
        std::lock_guard lock(state_->mutex);
        ++state_->counters.lookupCount;
        const auto existing = state_->ready.find(key);
        if (existing != state_->ready.end()) { owner = existing->second.owner.lock(); }
        if (owner) {
            ++state_->counters.readyHitCount;
            // Retention is optional. Allocation failure must not turn a live
            // native object into a failed acquisition or strand a flight.
            try { state_->touchLocked(key, existing->second, owner, retired); }
            catch (const std::bad_alloc&) {}
        } else {
            const auto pending = state_->inFlight.find(key);
            if (pending != state_->inFlight.end()) {
                flight = pending->second;
                ++state_->counters.singleFlightWaitCount;
            } else {
                state_->pruneExpiredLocked();
                try {
                    flight = std::make_shared<State::Flight>();
                    state_->inFlight.emplace(key, flight);
                } catch (const std::bad_alloc&) { return makeError(Error::OutOfMemory); }
                ++state_->counters.nativeCreateCount;
                creator = true;
            }
        }
    }
    retired.clear();
    if (owner) { return NativePipelineAcquisition{.owner = std::move(owner), .nativeObjectReused = true}; }
    if (!creator) {
        const auto begin = std::chrono::steady_clock::now();
        const auto created = flight->future.get();
        const auto waitNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
        if (!created) { return makeError(created.error()); }
        return NativePipelineAcquisition{.owner = *created,
            .singleFlightWaitTimeNanoseconds = waitNs, .nativeObjectReused = true};
    }

    // Neither driver creation nor waiting holds the map mutex. Every elected
    // creator completes its promise, including factory exceptions and failures.
    State::Created created = makeError(Error::Failure);
    try {
        created = create();
        if (created && !*created) { created = makeError(Error::Failure); }
    } catch (const std::bad_alloc&) { created = makeError(Error::OutOfMemory); }
    catch (...) { created = makeError(Error::Failure); }
    {
        std::lock_guard lock(state_->mutex);
        if (created) {
            try {
                auto& entry = state_->ready[key];
                entry.owner = *created;
                state_->touchLocked(key, entry, *created, retired);
            } catch (const std::bad_alloc&) {
                // The successful native owner still reaches every waiter.
                // Skipping optional cache bookkeeping is preferable to retrying
                // the already completed driver work or abandoning the promise.
            }
        } else { ++state_->counters.failedCreateCount; }
        flight->promise.set_value(created);
        state_->inFlight.erase(key);
    }
    retired.clear();
    if (!created) { return makeError(created.error()); }
    return NativePipelineAcquisition{.owner = std::move(*created)};
}

void NativePipelineCache::setCapacity(size_t strongCapacity)
{
    State::Residents retired;
    {
        std::lock_guard lock(state_->mutex);
        state_->counters.strongCapacity = strongCapacity;
        state_->trimLocked(retired);
        state_->pruneExpiredLocked();
    }
}

NativePipelineCacheStats NativePipelineCache::stats() const
{
    std::lock_guard lock(state_->mutex);
    auto result = state_->counters;
    result.readyEntryCount = state_->ready.size();
    result.residentCount = state_->residents.size();
    result.inFlightCount = state_->inFlight.size();
    return result;
}

} // namespace metallic::render::detail
