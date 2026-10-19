// Included after the ordinary compiler entry, inside metallic::render.
// Unsupported notification/producer platforms retain fresh compilation. This
// path neither retains module IR nor consumes/publishes an unqualified cache.
void reportSlangWarmupProgress(const SlangWarmupBatchOptions& options,
    const SlangWarmupBatchProgress& progress) noexcept
{
    if (!options.onProgress) { return; }
    try { options.onProgress(progress); }
    catch (...) { /* Diagnostic callbacks must not alter compilation or publication. */ }
}

Result<SlangWarmupBatchResult> compileFreshSlangWarmupBatch(std::span<const SlangShaderDesc> requests,
    const SlangWarmupBatchOptions& options, std::string& log)
{
    const auto start = std::chrono::steady_clock::now();
    reportSlangWarmupProgress(options, {.phase = "preparing"});
    std::vector<std::unique_ptr<PreparedSlangRequest>> owned;
    owned.reserve(requests.size());
    for (const auto& request : requests) {
        auto prepared = prepareSlangRequest(request, log);
        if (!prepared) { return makeError(prepared.error()); }
        owned.push_back(std::move(*prepared));
    }
    SlangWarmupBatchResult result;
    result.shaders.resize(requests.size());
    result.sessionGroups = requests.size();
    std::vector<size_t> schedule;
    schedule.reserve(requests.size());
    for (size_t index = 0; index < requests.size(); ++index) { schedule.push_back(index); }
    std::stable_sort(schedule.begin(), schedule.end(), [&](size_t left, size_t right) {
        return estimateWarmupEntryWork(requests[left]) > estimateWarmupEntryWork(requests[right]);
    });
    std::vector<std::string> failures(requests.size());
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    reportSlangWarmupProgress(options, {.phase = "compiling"});
    {
        std::vector<std::jthread> workers;
        const auto jobs = std::min(options.jobs, requests.size());
        workers.reserve(jobs);
        for (size_t worker = 0; worker < jobs; ++worker) {
            workers.emplace_back([&] {
                struct FreshScope {
                    bool previous = gBypassOrdinarySnapshot;
                    FreshScope() { gBypassOrdinarySnapshot = true; }
                    ~FreshScope() { gBypassOrdinarySnapshot = previous; }
                } scope;
                auto cache = options.cache;
                cache.enableDiskCache = false;
                cache.outCacheHit = nullptr;
                while (true) {
                    if (failed.load(std::memory_order_relaxed)) { return; }
                    const auto scheduled = next.fetch_add(1, std::memory_order_relaxed);
                    if (scheduled >= schedule.size()) { return; }
                    const auto index = schedule[scheduled];
                    try {
                        auto shader = compileSlangShaderToSpirv(owned[index]->owned.descriptor, cache, failures[index]);
                        if (!shader || shader->metrics.cacheHit || !shader->metrics.compilerExecuted || shader->spirv.empty()) {
                            failed.store(true, std::memory_order_relaxed);
                        } else {
                            result.shaders[index] = std::move(*shader);
                            if (options.onShaderReady) { options.onShaderReady(index, result.shaders[index]); }
                            reportSlangWarmupProgress(options, {.phase = "request-complete", .requestIndex = index,
                                .cacheHit = result.shaders[index].metrics.cacheHit});
                        }
                    } catch (const std::exception& error) {
                        failures[index] = error.what();
                        failed.store(true, std::memory_order_relaxed);
                    } catch (...) {
                        failures[index] = "Unknown fresh warmup worker exception";
                        failed.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
    }
    if (failed.load(std::memory_order_relaxed)) {
        for (size_t index = 0; index < failures.size(); ++index) {
            if (!failures[index].empty()) { log += "Fresh warmup request " + std::to_string(index) + ": " + failures[index] + '\n'; }
        }
        return makeError(Error::Failure);
    }
    result.compilerExecutions = requests.size();
    result.completedAllRequests = true;
    result.totalTimeNanoseconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
    reportSlangWarmupProgress(options, {.phase = "finished"});
    return result;
}

Result<SlangWarmupBatchResult> compileSlangShaderWarmupBatch(std::span<const SlangShaderDesc> requests,
    const SlangWarmupBatchOptions& options, std::string& log)
{
    using BatchClock = std::chrono::steady_clock;
    const auto start = BatchClock::now();
    const auto elapsedBatch = [](BatchClock::time_point before) {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(BatchClock::now() - before).count());
    };
    log.clear();
    if (requests.empty() || !options.jobs || options.jobs > 256 || options.cache.outCacheHit || gWarmupSnapshotCompileContext) {
        log = "Invalid/nested warmup batch"; return makeError(Error::InvalidArgument);
    }
#ifndef _WIN32
    return compileFreshSlangWarmupBatch(requests, options, log);
#else
    try {
        reportSlangWarmupProgress(options, {.phase = "preparing"});
        struct PlannedRequest {
            std::unique_ptr<PreparedSlangRequest> prepared;
            std::unique_ptr<OwnedSlangSessionConfig> configuration;
        };
        std::vector<PlannedRequest> planned;
        std::vector<SlangShaderDesc> ownedDescriptors;
        std::vector<std::vector<fs::path>> paths;
        std::vector<WarmupGroupPlan> groups;
        std::map<std::vector<uint8_t>, size_t> groupByKey;
        planned.reserve(requests.size()); ownedDescriptors.reserve(requests.size());
        for (size_t index = 0; index < requests.size(); ++index) {
            auto preparedResult = prepareSlangRequest(requests[index], log);
            if (!preparedResult) { return makeError(preparedResult.error()); }
            auto prepared = std::move(*preparedResult);
            auto config = OwnedSlangSessionConfig::create(nullptr, prepared->owned.descriptor,
                prepared->sessionPolicy(options.cache), nullptr, log);
            if (!config) { return makeError(Error::InvalidArgument); }
            auto key = config->effectiveKey();
            // A reused session can suppress filesystem reads for loaded IR.
            // Keep its conservative provenance closure within one root module.
            const std::string root = prepared->owned.descriptor.moduleName;
            key.push_back(0xff); key.insert(key.end(), root.begin(), root.end());
            if (!options.shareSessions) {
                const auto* bytes = reinterpret_cast<const uint8_t*>(&index);
                key.insert(key.end(), bytes, bytes + sizeof(index));
            }
            const auto [entry, inserted] = groupByKey.try_emplace(std::move(key), groups.size());
            if (inserted) { groups.push_back({}); }
            groups[entry->second].requestIndices.push_back(index);
            paths.push_back(config->normalizedSearchPaths());
            ownedDescriptors.push_back(prepared->owned.descriptor);
            planned.push_back({std::move(prepared), std::move(config)});
        }
        estimateAndSortWarmupGroups(ownedDescriptors, groups);
        std::shared_ptr<CompilerProducerSeal> producer;
        const auto captureStart = BatchClock::now();
        std::unique_ptr<SlangSnapshotFileSystem> snapshot;
        reportSlangWarmupProgress(options, {.phase = "capturing"});
        try {
            producer = compilerProducerSeal();
            snapshot = captureSourceEpoch(paths, log);
        } catch (const std::exception& error) {
            log += std::string("Source snapshot reuse unavailable: ") + error.what() + '\n';
            return compileFreshSlangWarmupBatch(requests, options, log);
        }
        if (!snapshot) { return compileFreshSlangWarmupBatch(requests, options, log); }
        SlangWarmupBatchResult result;
        result.snapshotCaptureTimeNanoseconds = elapsedBatch(captureStart);
        result.sessionGroups = groups.size(); result.shaders.resize(requests.size());
        std::vector<std::optional<SlangWarmupCacheTransaction<SlangSnapshotFileSystem>::Item>> pending(requests.size());
        std::vector<std::string> failures(requests.size());
        std::atomic<size_t> nextGroup{0}; std::atomic<bool> failed{false};
        reportSlangWarmupProgress(options, {.phase = "compiling"});
        {
            std::vector<std::jthread> workers;
            const auto jobs = std::min(options.jobs, groups.size());
            workers.reserve(jobs);
            for (size_t worker = 0; worker < jobs; ++worker) {
                workers.emplace_back([&] {
                    while (!failed.load(std::memory_order_relaxed)) {
                        const auto next = nextGroup.fetch_add(1, std::memory_order_relaxed);
                        if (next >= groups.size()) { return; }
                        const auto& group = groups[next];
                        // Each worker uses the ordinary TLS IGlobalSession.
                        // No ISession is accessed concurrently by two entries.
                        SnapshotCompileContext context(*snapshot, producer, options.shareSessions, true);
                        struct ContextScope {
                            SnapshotCompileContext* previous = gWarmupSnapshotCompileContext;
                            explicit ContextScope(SnapshotCompileContext& value) { gWarmupSnapshotCompileContext = &value; }
                            ~ContextScope() { gWarmupSnapshotCompileContext = previous; }
                        } scope(context);
                        for (const auto index : group.requestIndices) {
                            if (failed.load(std::memory_order_relaxed)) { return; }
                            try {
                                context.stage = [&](uint64_t requestHash, const fs::path&, const ShaderCompileResult& shader) {
                                    if (pending[index] || requestHash != planned[index].prepared->requestHash) { return false; }
                                    pending[index] = SlangWarmupCacheTransaction<SlangSnapshotFileSystem>::Item{
                                        index, requestHash, shader.spirv, context.sourceProof()}; return true;
                                };
                                std::string diagnostics;
                                auto shader = compileSlangShaderToSpirv(ownedDescriptors[index], options.cache, diagnostics);
                                if (!shader || !pending[index]) {
                                    failures[index] = std::move(diagnostics); failed.store(true, std::memory_order_relaxed); return;
                                }
                                result.shaders[index] = std::move(*shader);
                                if (options.onShaderReady) { options.onShaderReady(index, result.shaders[index]); }
                                reportSlangWarmupProgress(options, {.phase = "request-complete", .requestIndex = index,
                                    .cacheHit = result.shaders[index].metrics.cacheHit});
                            } catch (const std::exception& error) {
                                failures[index] = error.what(); failed.store(true, std::memory_order_relaxed); return;
                            } catch (...) {
                                failures[index] = "Unknown warmup worker exception"; failed.store(true, std::memory_order_relaxed); return;
                            }
                        }
                    }
                });
            }
        } // All workers, their modules/sessions and TLS globals are destroyed.
        if (failed.load(std::memory_order_relaxed)) {
            for (size_t index = 0; index < failures.size(); ++index) {
                if (!failures[index].empty()) { log += "Warmup request " + std::to_string(index) + ": " + failures[index] + '\n'; }
            }
            return makeError(Error::Failure);
        }
        const auto validationStart = BatchClock::now();
        reportSlangWarmupProgress(options, {.phase = "validating"});
        if (!producer->validate(log, true) || !snapshot->validate(log)) { return makeError(Error::Failure); }
        const auto policiesUnchanged = [&] {
            for (size_t index = 0; index < planned.size(); ++index) {
                auto current = prepareSlangRequest(ownedDescriptors[index], log);
                if (!current || (*current)->requestHash != planned[index].prepared->requestHash) {
                    log += "Process shader policy changed during the warmup batch\n"; return false;
                }
            }
            return true;
        };
        if (!policiesUnchanged()) { return makeError(Error::Failure); }
        result.boundaryValidationTimeNanoseconds = elapsedBatch(validationStart);
        for (size_t index = 0; index < requests.size(); ++index) {
            if (!pending[index] || result.shaders[index].spirv.empty()) { log += "Incomplete warmup result set\n"; return makeError(Error::Failure); }
            result.compilerExecutions += result.shaders[index].metrics.compilerExecuted ? 1 : 0;
            result.existingCacheHits += result.shaders[index].metrics.cacheHit ? 1 : 0;
        }
        const auto publicationStart = BatchClock::now();
        reportSlangWarmupProgress(options, {.phase = "publishing"});
        if (options.cache.enableDiskCache) {
            const auto directory = shaderCachePath(options.cache, 0).parent_path();
            if (snapshot->covers(directory)) { log += "Warmup cache/source namespace overlap\n"; return makeError(Error::Failure); }
            SlangWarmupCacheTransaction<SlangSnapshotFileSystem> transaction(*snapshot, producer->identity(), directory);
            for (auto& item : pending) { if (!transaction.stage(std::move(*item), log)) { return makeError(Error::Failure); } }
            // Registry identities are registered by the normal batch wrapper,
            // only after this complete actual persistence/readback succeeds.
            if (!transaction.publish(requests.size(), writeEncodedShaderCache, [](size_t) {}, log)) { return makeError(Error::Failure); }
            const auto& phases = transaction.timings();
            result.cacheProofEncodeTimeNanoseconds = phases.proofEncodeNanoseconds;
            result.cacheAtomicWriteTimeNanoseconds = phases.atomicWriteNanoseconds;
            result.cacheFileReadDecodeTimeNanoseconds = phases.fileReadDecodeNanoseconds;
            result.cacheFullSourceValidationTimeNanoseconds = phases.fullSourceValidationNanoseconds;
        } else if (!snapshot->validate(log)) { return makeError(Error::Failure); }
        if (!producer->validate(log, true) || !snapshot->validate(log) || !policiesUnchanged()) { return makeError(Error::Failure); }
        result.cachePublicationTimeNanoseconds = elapsedBatch(publicationStart);
        result.completeSourceEpochValidated = true;
        result.completedAllRequests = true;
        result.totalTimeNanoseconds = elapsedBatch(start);
        reportSlangWarmupProgress(options, {.phase = "finished"});
        return result;
    } catch (const std::exception& error) {
        log += std::string("Warmup source-qualified batch rejected: ") + error.what() + '\n';
        return makeError(Error::Failure);
    }
#endif
}
