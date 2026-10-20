// Included in metallic::render after the original compiler's private helpers.
// The generated policy block is the original validation/resolution/hash block.
struct PreparedSlangRequest {
    explicit PreparedSlangRequest(const SlangShaderDesc& source) : owned(source) {}
    OwnedSlangRequest owned;
    SlangShaderDesc resolvedDesc{};
    bool native = false;
    SlangShaderDebugMode debugMode{};
    std::optional<SlangShaderOptimizationLevel> effectiveOptimization;
    SlangOptimizationLevel optimizationLevel{};
    std::optional<SlangShaderPostprocessMode> effectivePostprocess;
    uint64_t requestHash = 0;
    ResolvedSlangSessionPolicy sessionPolicy(const SlangShaderCacheOptions& options) const
    {
        return {debugMode, optimizationLevel, *effectivePostprocess, native,
            resolvedDesc.descriptorHeapStrides, options.reportDetailedCompilerPerf};
    }
};

Result<std::unique_ptr<PreparedSlangRequest>> prepareSlangRequest(const SlangShaderDesc& input, std::string& log)
{
    const auto& desc = input;
#include "SlangRequestPolicy.inc"
    auto result = std::make_unique<PreparedSlangRequest>(input);
    result->resolvedDesc = result->owned.descriptor;
    result->resolvedDesc.descriptorHeapMode = resolvedDesc.descriptorHeapMode;
    result->resolvedDesc.descriptorHeapStrides = resolvedDesc.descriptorHeapStrides;
    result->native = native;
    result->debugMode = debugMode; result->effectiveOptimization = effectiveOptimization;
    result->optimizationLevel = optimizationLevel; result->effectivePostprocess = effectivePostprocess;
    result->requestHash = requestHash;
    return result;
}

// Pins only producer binaries, never source files or editor writers. Actual
// Slang import-address module paths are checked, not assumed build SDK paths.
// A once-per-process tag is frozen before any request module is loaded.
class CompilerProducerSeal {
public:
    ~CompilerProducerSeal() = default;
    CompilerProducerSeal(const CompilerProducerSeal&) = delete;
    CompilerProducerSeal& operator=(const CompilerProducerSeal&) = delete;
    uint64_t identity() const noexcept { return identity_; }
    static std::shared_ptr<CompilerProducerSeal> create()
    {
#ifdef _WIN32
        auto result = std::shared_ptr<CompilerProducerSeal>(new CompilerProducerSeal);
        const auto compilerPath = modulePath(reinterpret_cast<LPCWSTR>(&slang_createGlobalSession2));
        const auto producerPath = modulePath(reinterpret_cast<LPCWSTR>(&prepareSlangRequest));
        if (compilerPath.filename() != "slang.dll" && compilerPath.filename() != "slang-compiler.dll") {
            throw std::runtime_error("Global-session API import address did not identify a loaded Slang DLL");
        }
        for (const auto& path : {producerPath, compilerPath.parent_path() / "slang.dll", compilerPath.parent_path() / "slang-compiler.dll",
                compilerPath.parent_path() / "slang-glslang.dll"}) { result->pin(path); }
        uint64_t tag = kFnvOffset;
        tag = hashText(tag, "validated-immutable-snapshot-v3-schema2-embedded-sdk-stdlib");
        tag = hashText(tag, SLANG_VERSION_NUMERIC); tag = hashText(tag, spGetBuildTagString());
        tag = hashText(tag, vulkanSpirvTarget().id); tag = hashValue(tag, vulkanSpirvTarget().revision);
        tag = hashValue(tag, kShaderCacheRequestVersion);
        // The host remains pinned/fully validated, but its unrelated main/UI
        // bytes do not prevent the same shared compiler build consuming V3.
        // The shared target's generated identity seals its actual build. An
        // unqualified toolchain/link contract falls back to per-host identity.
        // Actual SDK bytes still
        // contribute to the identity before any source module load.
        const auto build = compiledShaderCompilerBuildIdentity();
        if (build.sha256.empty()) { throw std::runtime_error("Shader compiler build identity is unavailable"); }
        const std::string buildSha(build.sha256);
        tag = hashText(tag, buildSha.c_str());
        if (!build.allowCrossHostReuse) { tag = hashValue(tag, result->files_[0]->contentHash); }
        for (size_t index = 1; index < result->files_.size(); ++index) {
            tag = hashValue(tag, result->files_[index]->contentHash);
        }
        result->identity_ = tag ? tag : 1;
        std::string reason;
        if (!result->validate(reason, true)) { throw std::runtime_error(reason); }
        return result;
#else
        throw std::runtime_error("Source snapshot reuse requires Windows producer paths/notifications");
#endif
    }
    bool validate(std::string& log, bool full) const
    {
#ifdef _WIN32
        for (const auto& ownedFile : files_) {
            const auto& file = *ownedFile;
            BY_HANDLE_FILE_INFORMATION now{};
            if (!GetFileInformationByHandle(file.handle, &now) ||
                now.dwVolumeSerialNumber != file.information.dwVolumeSerialNumber ||
                now.nFileIndexHigh != file.information.nFileIndexHigh || now.nFileIndexLow != file.information.nFileIndexLow ||
                now.nFileSizeHigh != file.information.nFileSizeHigh || now.nFileSizeLow != file.information.nFileSizeLow ||
                CompareFileTime(&now.ftLastWriteTime, &file.information.ftLastWriteTime)) {
                log += "Compiler producer binary changed or became unavailable\n"; return false;
            }
            // The pinned READ/SHARE_READ handle excludes existing writable
            // mappings and writers, and denies future writes/deletes. Reopen
            // the path to also detect ancestor replacement or redirection.
            const HANDLE current = CreateFileW(file.path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ,
                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            BY_HANDLE_FILE_INFORMATION currentInformation{};
            const bool sameFile = current != INVALID_HANDLE_VALUE && GetFileInformationByHandle(current, &currentInformation) &&
                currentInformation.dwVolumeSerialNumber == file.information.dwVolumeSerialNumber &&
                currentInformation.nFileIndexHigh == file.information.nFileIndexHigh &&
                currentInformation.nFileIndexLow == file.information.nFileIndexLow;
            if (current != INVALID_HANDLE_VALUE) { CloseHandle(current); }
            if (!sameFile) { log += "Compiler producer path no longer identifies the sealed binary\n"; return false; }
            if (full) {
                try {
                    if (!file.sealedBytes || read(file.path) != *file.sealedBytes) {
                        log += "Compiler producer exact byte seal changed\n"; return false;
                    }
                } catch (...) { log += "Compiler producer exact byte read failed\n"; return false; }
            }
            const auto name = file.path.filename().wstring();
            if (name == L"slang.dll" || name == L"slang-compiler.dll" || name == L"slang-glslang.dll") {
                const HMODULE loaded = GetModuleHandleW(name.c_str());
                if (loaded) {
                    std::array<wchar_t, 32768> path{};
                    const auto count = GetModuleFileNameW(loaded, path.data(), static_cast<DWORD>(path.size()));
                    if (!count || count >= path.size() || pathKey(fs::path(path.data())) != pathKey(file.path)) {
                        log += "Loaded downstream Slang module path differs from sealed producer\n"; return false;
                    }
                }
            }
        }
        return true;
#else
        log += "Windows SDK producer qualification unavailable\n"; return false;
#endif
    }
private:
    CompilerProducerSeal() = default;
#ifdef _WIN32
    struct File {
        fs::path path;
        HANDLE handle = INVALID_HANDLE_VALUE;
        BY_HANDLE_FILE_INFORMATION information{};
        uint64_t contentHash = 0;
        std::unique_ptr<const Bytes> sealedBytes;
        ~File() { if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); } }
    };
    std::vector<std::unique_ptr<File>> files_;
    static fs::path modulePath(LPCWSTR address)
    {
        HMODULE module{};
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                address, &module)) { throw std::runtime_error("Cannot identify the actual loaded producer module"); }
        std::array<wchar_t, 32768> path{};
        const auto count = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (!count || count >= path.size()) { throw std::runtime_error("Invalid actual producer module path"); }
        return normalizedAbsolutePath(fs::path(path.data()));
    }
    void pin(const fs::path& path)
    {
        for (auto parent = path; !parent.empty();) {
            const DWORD attributes = GetFileAttributesW(parent.c_str());
            if (attributes == INVALID_FILE_ATTRIBUTES || attributes & FILE_ATTRIBUTE_REPARSE_POINT) {
                throw std::runtime_error("Unqualified producer path/reparse ancestor");
            }
            const auto next = parent.parent_path(); if (next == parent) { break; } parent = next;
        }
        auto file = std::make_unique<File>(); file->path = normalizedAbsolutePath(path);
        file->handle = CreateFileW(file->path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file->handle == INVALID_HANDLE_VALUE) { throw std::runtime_error("Cannot pin actual producer binary"); }
        if (!GetFileInformationByHandle(file->handle, &file->information)) {
            throw std::runtime_error("Cannot record actual producer binary identity");
        }
        file->sealedBytes = std::make_unique<const Bytes>(read(file->path));
        file->contentHash = slang_snapshot::hash(file->sealedBytes->data(), file->sealedBytes->size());
        files_.push_back(std::move(file)); // Stable address and constructor-failure RAII.
    }
#endif
    uint64_t identity_ = 0;
};

std::shared_ptr<CompilerProducerSeal> compilerProducerSeal()
{
    static std::mutex mutex;
    static std::shared_ptr<CompilerProducerSeal> seal;
    std::lock_guard lock(mutex);
    if (!seal) { seal = CompilerProducerSeal::create(); }
    return seal;
}

std::unique_ptr<SlangSnapshotFileSystem> captureSourceEpoch(const std::vector<std::vector<fs::path>>& searchRoots, std::string& log)
{
    std::vector<fs::path> roots;
    const auto add = [&](const fs::path& input) {
        const auto path = normalizedAbsolutePath(input);
        std::error_code error;
        if (!fs::is_directory(path, error)) { return; } // Qualified by a captured ancestor below, else fail closed.
        if (std::ranges::any_of(roots, [&](const auto& root) { return childOf(pathKey(path), pathKey(root)); })) { return; }
        std::erase_if(roots, [&](const auto& root) { return childOf(pathKey(root), pathKey(path)); });
        roots.push_back(path);
    };
    for (const auto& paths : searchRoots) { for (const auto& path : paths) { add(path); } }
    for (const char* path : {"Shaders", "Source", ".cache/materials", "External/RTXCR-Material/shaders",
            "External/RTXCR-Geometry/shaders", "External/Rtxtf", "External/slang/include"}) { add(fs::path(PROJECT_SOURCE_DIR) / path); }
    std::sort(roots.begin(), roots.end(), [](const auto& a, const auto& b) { return pathKey(a) < pathKey(b); });
    Json manifest{{"mappings", Json::array()}};
    for (const auto& root : roots) {
        manifest["mappings"].push_back({{"logical", pathText(root)}, {"physical", pathText(root)}, {"monitor_original", false}});
    }
    const auto cwd = normalizedAbsolutePath(fs::current_path());
    const bool cwdCovered = std::ranges::any_of(roots, [&](const auto& root) { return childOf(pathKey(cwd), pathKey(root)); });
    if (!cwdCovered) {
        manifest["mappings"].push_back({{"logical", pathText(cwd)}, {"physical", pathText(cwd)},
            {"recursive", false}, {"namespace_only", true}, {"monitor_original", false}});
    }
    auto snapshot = std::make_unique<SlangSnapshotFileSystem>(manifest, 0);
    for (const auto& paths : searchRoots) {
        for (const auto& path : paths) {
            if (!snapshot->covers(path)) { log += "Effective search path has no complete captured namespace/ancestor\n"; return {}; }
        }
    }
    return snapshot;
}

bool writeEncodedShaderCache(const fs::path& path, std::span<const uint8_t> bytes)
{
    if (bytes.empty() || bytes.size() > kMaxShaderCacheFileSize) { return false; }
    std::error_code error; fs::create_directories(path.parent_path(), error); if (error) { return false; }
    const auto temporary = temporaryShaderCachePath(path);
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    const bool wrote = stream && bool(stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())));
    stream.flush(); const bool valid = wrote && bool(stream); stream.close();
    if (!valid || !replaceShaderCacheFile(temporary, path)) { fs::remove(temporary, error); return false; }
    return true;
}

#include "SlangSourceEpochPool.inc"

thread_local bool gBypassOrdinarySnapshot = false;
thread_local bool gOrdinarySnapshotUnqualified = false;
enum class SnapshotValidation { Notifications, CacheHit, Complete };
struct SnapshotCompileContext {
    SnapshotCompileContext(SlangSnapshotFileSystem& source, std::shared_ptr<CompilerProducerSeal> seal, bool share, bool defer = false)
        : snapshot(source), provenance(source.createProvenanceView()), producer(std::move(seal)),
          shareSession(share), deferPublication(defer) {}
    ~SnapshotCompileContext()
    {
        try {
            if (!deferPublication && provenance->hasUnqualifiedLookups()) { gOrdinarySnapshotUnqualified = true; }
        } catch (...) { if (!deferPublication) { gOrdinarySnapshotUnqualified = true; } }
    }
    SlangSnapshotFileSystem& snapshot;
    std::unique_ptr<SlangSnapshotFileSystem> provenance;
    std::shared_ptr<CompilerProducerSeal> producer;
    bool shareSession = false, deferPublication = false;
    // Correct destruction order: modules, ISession, then descriptor/string storage.
    std::unique_ptr<OwnedSlangSessionConfig> ownedSession;
    Slang::ComPtr<slang::ISession> session;
    std::map<std::string, Slang::ComPtr<slang::IModule>> modules;
    std::function<bool(uint64_t, const fs::path&, const ShaderCompileResult&)> stage;
    ShaderCompileMetrics* invocationMetrics = nullptr;
    ISlangFileSystem* fileSystem() const noexcept { return provenance.get(); }
    ShaderNamespaceProof sourceProof() const { return provenance->sourceProof(producer->identity()); }
    bool guard(std::string& log, SnapshotValidation validation) const
    {
        if (!producer) { return false; }
        const auto producerStarted = Clock::now();
        const bool producerValid = producer->validate(log, validation == SnapshotValidation::Complete);
        if (invocationMetrics) {
            invocationMetrics->producerValidationTimeNanoseconds += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - producerStarted).count());
        }
        if (!producerValid) { return false; }
        const auto sourceStarted = Clock::now();
        const bool sourceValid = validation == SnapshotValidation::Complete ? snapshot.validate(log) :
            validation == SnapshotValidation::CacheHit ? provenance->validateCacheHit(log) : snapshot.validateNotifications(log);
        if (invocationMetrics) {
            auto& time = validation == SnapshotValidation::Notifications ? invocationMetrics->sourceNotificationValidationTimeNanoseconds :
                invocationMetrics->sourceValidationTimeNanoseconds;
            time += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - sourceStarted).count());
        }
        return sourceValid;
    }
    bool acquireSession(slang::IGlobalSession* global, std::unique_ptr<OwnedSlangSessionConfig>& config,
        Slang::ComPtr<slang::ISession>& result, std::string& log)
    {
        if (!guard(log, SnapshotValidation::Notifications)) { return false; }
        if (!session) {
            ownedSession = std::move(config);
            if (SLANG_FAILED(global->createSession(ownedSession->desc(), session.writeRef())) || !session) { return false; }
        } else if (config->effectiveKey() != ownedSession->effectiveKey()) {
            log += "Effective session policy changed within a planned group\n"; return false;
        }
        result = session; return true;
    }
    Slang::ComPtr<slang::IModule> loadModule(slang::ISession& source, const char* name, ISlangBlob** diagnostics)
    {
        const auto found = modules.find(name); if (found != modules.end()) { return found->second; }
        Slang::ComPtr<slang::IModule> module(source.loadModule(name, diagnostics));
        if (module) { modules.emplace(name, module); }
        return module;
    }
};
thread_local SnapshotCompileContext* gWarmupSnapshotCompileContext = nullptr;

bool snapshotActualDependencies(SnapshotCompileContext& context, std::span<const fs::path> paths,
    std::vector<ShaderDependencySnapshot>& output)
{
    output.clear();
    for (const auto& path : paths) {
        ShaderSourceReadProof read; fs::file_time_type time;
        if (!context.provenance->actualDependency(path, read, time)) { output.clear(); return false; }
        output.push_back({fs::u8path(read.canonicalPath), {.exists = true, .fileSize = read.byteSize,
            .writeTime = time, .contentHash = read.contentHash}});
    }
    return !output.empty();
}

bool snapshotModuleDependencies(SnapshotCompileContext& context, slang::IModule& module,
    std::vector<ShaderDependencySnapshot>& output, std::string& log)
{
    output.clear();
    const auto count = module.getDependencyFileCount();
    if (count < 0) { log += "Negative SDK dependency count\n"; return false; }
    for (SlangInt32 index = 0; index < count; ++index) {
        const char* identity = module.getDependencyFilePath(index);
        log += "[SnapshotDependency] SDK raw identity: "; log += identity ? identity : "<null>"; log += '\n';
        ShaderSourceReadProof read; fs::file_time_type time;
        // Prefer the exact token issued by this FileSystem. A canonical path
        // is accepted only when it independently matches an actual delivered
        // image, never after interpreting an arbitrary opaque token as a path.
        const bool issued = context.provenance->actualIssuedDependency(identity, read, time);
        const bool canonical = !issued && identity && isSafeProofPath(identity, false) &&
            context.provenance->actualDependency(fs::u8path(identity), read, time) && read.canonicalPath == identity;
        if (!issued && !canonical) {
            log += "SDK dependency does not identify a delivered image in this epoch\n"; output.clear(); return false;
        }
        output.push_back({fs::u8path(read.canonicalPath), {.exists = true, .fileSize = read.byteSize,
            .writeTime = time, .contentHash = read.contentHash}});
    }
    if (!count) {
        log += "[SnapshotDependency] SDK count=0; using actual delivered read provenance\n";
        const auto proof = context.sourceProof();
        for (const auto& read : proof.actualReads) {
            fs::file_time_type time; ShaderSourceReadProof actual;
            if (!context.provenance->actualDependency(fs::u8path(read.canonicalPath), actual, time)) { return false; }
            output.push_back({fs::u8path(actual.canonicalPath), {.exists = true, .fileSize = actual.byteSize,
                .writeTime = time, .contentHash = actual.contentHash}});
        }
    }
    std::sort(output.begin(), output.end(), [](const auto& a, const auto& b) { return a.path < b.path; });
    output.erase(std::unique(output.begin(), output.end(), [](const auto& a, const auto& b) { return a.path == b.path; }), output.end());
    return !output.empty();
}

bool loadSnapshotCachedShader(SnapshotCompileContext& context, const fs::path& path, uint64_t request,
    std::vector<uint32_t>& words, std::vector<ShaderDependencySnapshot>& dependencies, std::string& log)
{
    words.clear(); dependencies.clear();
    try {
        if (!context.guard(log, SnapshotValidation::Notifications)) { return false; }
        const auto bytes = read(path); slang_shader_cache::CachedPayload output;
        if (!context.provenance->decodeAndAcceptCache(bytes, request, context.producer->identity(), output, log)) { return false; }
        std::vector<fs::path> paths;
        for (const auto& item : output.proof.actualReads) { paths.push_back(fs::u8path(item.canonicalPath)); }
        if (!snapshotActualDependencies(context, paths, dependencies)) { return false; }
        words = std::move(output.spirv); return true;
    } catch (...) { return false; }
}

bool completeSnapshotInvocation(SnapshotCompileContext& context, const SlangShaderCacheOptions& options,
    uint64_t request, const fs::path& path, const ShaderCompileResult& result, std::string& log)
{
    try {
    if (context.deferPublication) { return context.guard(log, SnapshotValidation::Notifications) && context.stage && context.stage(request, path, result); }
    // A decoded hit validates all resolver namespaces and its actual source
    // reads, while the producer's pinned handles preserve its initial byte seal.
    // Misses retain complete publication/readback guards.
    if (result.metrics.cacheHit) { return context.guard(log, SnapshotValidation::CacheHit); }
    if (!context.guard(log, SnapshotValidation::Complete)) { return false; }
    if (options.enableDiskCache && !result.metrics.cacheHit) {
        if (context.snapshot.covers(path) || context.snapshot.covers(path.parent_path())) {
            log += "[ShaderCacheV3] Cache/source namespace overlaps; persistence bypassed\n";
            return context.guard(log, SnapshotValidation::Complete);
        }
        const auto proof = context.sourceProof();
        const auto bytes = slang_shader_cache::encode(request, proof, result.spirv);
        if (bytes.empty() || !writeEncodedShaderCache(path, bytes)) {
            log += "[ShaderCacheV3] Persistence unavailable; returning validated compiled code\n";
            return context.guard(log, SnapshotValidation::Complete);
        }
        const auto real = read(path); slang_shader_cache::CachedPayload output;
        if (real != bytes || !context.provenance->decodeAndAcceptCache(real, request, context.producer->identity(), output, log) ||
            output.spirv != result.spirv) {
            log += "[ShaderCacheV3] Actual persistence readback failed; returning validated compiled code\n";
            return context.guard(log, SnapshotValidation::Complete);
        }
    }
    return context.guard(log, SnapshotValidation::Complete);
    } catch (const std::exception& error) {
        log += std::string("[ShaderCacheV3] Ordinary persistence bypassed: ") + error.what() + '\n';
        return !context.deferPublication && !context.provenance->hasUnqualifiedLookups() && context.guard(log, SnapshotValidation::Complete);
    }
}
