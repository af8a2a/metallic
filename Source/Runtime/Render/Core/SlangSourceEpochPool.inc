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
        const std::vector<std::vector<fs::path>> searchRoots{orderedSearchPaths};
        const auto plan = planSourceEpoch(searchRoots);
        // Source images share only the exact capture footprint. Search order
        // still belongs to each request/session's independent identity.
        std::vector<std::string> key{pathKey(plan.cwd), hex(producer->identity())};
        for (const auto& mapping : plan.manifest.at("mappings")) {
            key.push_back(pathKey(fs::u8path(mapping.at("logical").get<std::string>())));
            key.push_back(pathKey(fs::u8path(mapping.at("physical").get<std::string>())));
            key.push_back(mapping.at("recursive").get<bool>() ? "recursive" : "root-only");
            key.push_back(mapping.at("namespace_only").get<bool>() ? "namespace-only" : "source-bytes");
            key.push_back(mapping.at("monitor_original").get<bool>() ? "monitor-original" : "physical-only");
        }
        static std::mutex mutex;
        static std::map<std::vector<std::string>, std::shared_ptr<Epoch>> epochs;
        std::lock_guard lock(mutex);
        const auto found = epochs.find(key);
        if (found != epochs.end()) {
            std::string reason;
            if (found->second->snapshot->validateNotifications(reason) &&
                !found->second->snapshot->hasUnqualifiedLookups()) {
                // Missing paths are omitted from the plan only when a captured
                // namespace proves their lookup. Qualify every new request.
                if (!sourceEpochCoversSearchPaths(*found->second->snapshot, searchRoots, log)) { return lease; }
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
        auto snapshot = captureSourceEpoch(plan, searchRoots, log);
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
