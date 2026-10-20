// Internal source snapshots only. Ordinary requests never retain sessions or IR.
class SlangSourceEpochPool
{
public:
    struct Epoch {
        std::unique_ptr<SlangSnapshotFileSystem> snapshot;
        std::shared_ptr<CompilerProducerSeal> producer;
    };
    struct Lease {
        std::shared_ptr<Epoch> epoch;
        uint64_t acquireTimeNanoseconds = 0;
        uint64_t captureTimeNanoseconds = 0;
        bool reused = false;
    };

    static Lease acquire(const std::vector<fs::path>& orderedSearchPaths,
        const std::shared_ptr<CompilerProducerSeal>& producer, std::string& log)
    {
        const auto started = Clock::now();
        const auto elapsed = [](const auto& start) {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
        };
        Lease lease;
        if (!producer || !producer->validate(log, false)) { return lease; }
        std::vector<std::string> key{pathKey(fs::current_path()), pathKey(PROJECT_SOURCE_DIR), hex(producer->identity())};
        for (const auto& path : orderedSearchPaths) { key.push_back(pathKey(path)); }
        static std::mutex mutex;
        static std::map<std::vector<std::string>, std::shared_ptr<Epoch>> epochs;
        std::lock_guard lock(mutex);
        const auto found = epochs.find(key);
        if (found != epochs.end()) {
            std::string reason;
            if (found->second->snapshot->validateNotifications(reason) &&
                !found->second->snapshot->hasUnqualifiedLookups()) {
                lease.epoch = found->second;
                lease.reused = true;
                lease.acquireTimeNanoseconds = elapsed(started);
                return lease;
            }
            // Active requests retain immutable bytes and must still pass
            // their own source/producer boundary before returning.
            log += "[ShaderCacheV3] Source epoch discarded: " + reason + '\n';
            epochs.erase(found);
        }
        const auto captureStarted = Clock::now();
        auto snapshot = captureSourceEpoch({orderedSearchPaths}, log);
        lease.captureTimeNanoseconds = elapsed(captureStarted);
        if (snapshot) {
            lease.epoch = std::make_shared<Epoch>();
            lease.epoch->snapshot = std::move(snapshot);
            lease.epoch->producer = producer;
            constexpr size_t kMaximumRetainedEpochs = 8;
            if (epochs.size() >= kMaximumRetainedEpochs) { epochs.erase(epochs.begin()); }
            epochs.emplace(std::move(key), lease.epoch);
        }
        lease.acquireTimeNanoseconds = elapsed(started);
        return lease;
    }
};
