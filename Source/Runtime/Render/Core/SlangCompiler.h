#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/GAPI/ShaderTarget.h"

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace metallic::render {

inline constexpr const char* kDefaultSlangProfileName = "spirv_1_6";

enum class SlangShaderDebugMode : uint8_t {
    Disabled,
    CaptureSymbols, // Embedded source and NonSemantic debug information (-g2); defaults to O1.
    ShaderDebug, // Full variable debug information (-g2); defaults to O0.
};

enum class SlangShaderOptimizationLevel : uint8_t {
    // Uses METALLIC_SLANG_OPTIMIZATION when set; otherwise O1, or O0 for ShaderDebug.
    Default,
    O0,
    O1,
    O2,
    O3,
};

enum class SlangDescriptorHeapMode : uint8_t {
    // Mapped by default. METALLIC_SLANG_DESCRIPTOR_MODE=native opts into the
    // experimental native path; see Documentation/DynamicResourceUpgradeStatus.md.
    Default,
    Mapped,
    Native,
};

enum class SlangShaderPostprocessMode : uint8_t {
    // Uses METALLIC_SPIRV_POSTPROCESS when set; otherwise leaves emitted code unchanged.
    Default,
    None,
    // Opt-in mapped-descriptor experiment: dead branches, then instruction simplification.
    DeadBranches,
    // Append the two passes to Slang's downstream optimizer; validate only the final output locally.
    SlangDeadBranches,
};

struct SlangMacroDefine {
    const char* name = nullptr;
    const char* value = "1";
};

struct SlangShaderDesc {
    // Module path relative to searchPath, without .slang, e.g.
    // "Features/Environment/EnvironmentLightingPrecompute".
    const char* moduleName = nullptr;
    const char* entryPointName = nullptr;
    // Program root. Modules/ and Interop/ under this root and the project's
    // Shaders directory are also searched for named subsystem imports.
    const char* searchPath = nullptr;
    std::span<const char* const> additionalSearchPaths;
    const char* profileName = kDefaultSlangProfileName;
    std::span<const char* const> capabilities;
    std::span<const SlangMacroDefine> macroDefines;
    SlangDescriptorHeapMode descriptorHeapMode = SlangDescriptorHeapMode::Default;
    // Native compilation uses this explicit device ABI, or the active device's
    // strides when both are zero. Mapped compilation ignores this field.
    DescriptorHeapShaderStrides descriptorHeapStrides;
    // Explicit levels override the process environment and the debug-mode default.
    SlangShaderOptimizationLevel optimizationLevel = SlangShaderOptimizationLevel::Default;
    SlangShaderPostprocessMode postprocessMode = SlangShaderPostprocessMode::Default;
};

// Complete effective request identity using the compiler's ordinary policy
// resolution and exact cache-key fields. Performs no source snapshot or compile.
// Equal hashes alone are insufficient when deduplicating requests.
[[nodiscard]] Result<std::vector<uint8_t>> slangShaderRequestIdentity(const SlangShaderDesc& desc, std::string& log);

// Per-call wall-clock measurements. Compiler code generation includes Slang IR
// optimization, SPIR-V emission and any downstream compiler work; the SDK's
// elapsed-time counters below overlap those phases and must not be added to them.
// Cache hits leave all compiler phases at zero. Timings are not persisted in the
// SPIR-V cache, so loading code never reports another invocation's measurements.
struct ShaderCompileMetrics {
    uint64_t totalTimeNanoseconds = 0;
    uint64_t cacheLookupTimeNanoseconds = 0;
    // Source acquire includes capture on a new epoch; capture is its subset.
    // Complete ordinary-call boundaries retain full source and producer bytes.
    uint64_t sourceEpochAcquireTimeNanoseconds = 0;
    uint64_t sourceEpochCaptureTimeNanoseconds = 0;
    uint64_t producerAcquireTimeNanoseconds = 0;
    uint64_t producerValidationTimeNanoseconds = 0;
    uint64_t sourceValidationTimeNanoseconds = 0;
    uint64_t sourceNotificationValidationTimeNanoseconds = 0;
    uint64_t compilerSetupTimeNanoseconds = 0;
    uint64_t moduleLoadTimeNanoseconds = 0;
    uint64_t dependencyTrackingTimeNanoseconds = 0;
    uint64_t entryPointCompositionTimeNanoseconds = 0;
    uint64_t linkTimeNanoseconds = 0;
    uint64_t codeGenerationTimeNanoseconds = 0;
    uint64_t targetFinalizeTimeNanoseconds = 0;
    // SlangDeadBranches includes only final validation; its transforms remain in code generation.
    uint64_t postprocessTimeNanoseconds = 0;
    // Raw compiler bytes are unavailable on cache hits; output bytes describe the returned code.
    // The returned Slang blob already includes any selected downstream transforms.
    uint64_t rawSpirvByteSize = 0;
    uint64_t outputSpirvByteSize = 0;
    uint64_t dependencyValidationTimeNanoseconds = 0;
    uint64_t cacheWriteTimeNanoseconds = 0;
    uint64_t compilerReportedTotalTimeNanoseconds = 0;
    uint64_t compilerReportedDownstreamTimeNanoseconds = 0;
    bool cacheHit = false;
    bool sourceEpochReused = false;
    bool compilerExecuted = false;
    bool compilerReportedTimingAvailable = false;
    bool detailedCompilerPerfRequested = false;
};

struct ShaderCompileResult {
    std::vector<uint32_t> spirv;
    std::string diagnostics;
    std::vector<std::string> dependencies;
    ShaderCompileMetrics metrics;
};

struct SlangShaderCacheOptions {
    // Null uses the project-local .cache/shaders/spirv directory.
    const char* cacheDirectory = nullptr;
    bool enableDiskCache = true;
    bool* outCacheHit = nullptr;
    // Diagnostic output only. A cache hit does not invoke the compiler or
    // fabricate a per-pass report. Use a cache miss for detailed profiling.
    bool reportDetailedCompilerPerf = false;
};

struct SlangWarmupBatchProgress {
    // preparing, capturing, compiling, validating, publishing, finished, or request-complete.
    const char* phase = nullptr;
    // Only request-complete identifies a request; its index follows the input order.
    size_t requestIndex = std::numeric_limits<size_t>::max();
    bool cacheHit = false;
};

struct SlangWarmupBatchOptions {
    // Per-request outCacheHit is unsupported; the result reports all hits.
    SlangShaderCacheOptions cache;
    size_t jobs = 8;
    bool shareSessions = true;
    // May run concurrently on worker threads. Must not throw; exceptions are ignored.
    // Request completion is provisional: accept results only after the complete batch
    // returns successfully, including its source validation and cache publication.
    std::function<void(const SlangWarmupBatchProgress&)> onProgress;
};

struct SlangWarmupBatchResult {
    std::vector<ShaderCompileResult> shaders;
    size_t sessionGroups = 0;
    size_t compilerExecutions = 0;
    size_t existingCacheHits = 0;
    uint64_t totalTimeNanoseconds = 0;
    uint64_t snapshotCaptureTimeNanoseconds = 0;
    uint64_t boundaryValidationTimeNanoseconds = 0;
    uint64_t cachePublicationTimeNanoseconds = 0;
    uint64_t cacheProofEncodeTimeNanoseconds = 0;
    uint64_t cacheAtomicWriteTimeNanoseconds = 0;
    uint64_t cacheFileReadDecodeTimeNanoseconds = 0;
    uint64_t cacheFullSourceValidationTimeNanoseconds = 0;
    bool completedAllRequests = false;
    bool completeSourceEpochValidated = false;
    bool liveFSLinearizationProven = false;
};

// Explicit startup batch. Ordinary misses still create fresh Slang sessions.
// Every request composes, links and finalizes independently. Source bytes and
// resolver namespaces are validated before the complete batch is published.
[[nodiscard]] Result<SlangWarmupBatchResult> compileSlangShaderWarmupBatch(
    std::span<const SlangShaderDesc> requests, const SlangWarmupBatchOptions& options, std::string& log);

// Process-global compilation policy. Configure it before shader compilation begins.
void setSlangShaderDebugMode(SlangShaderDebugMode mode) noexcept;
SlangShaderDebugMode slangShaderDebugMode() noexcept;
// Returns the effective O0..O3, or Default for an invalid request/environment.
// Configure METALLIC_SLANG_OPTIMIZATION before shader compilation begins.
SlangShaderOptimizationLevel slangShaderOptimizationLevel(const SlangShaderDesc& desc) noexcept;
const char* slangShaderOptimizationLevelName(SlangShaderOptimizationLevel level) noexcept;
// Returns the effective mode, or Default for an invalid request/environment.
// Configure METALLIC_SPIRV_POSTPROCESS before shader compilation begins.
SlangShaderPostprocessMode slangShaderPostprocessMode(const SlangShaderDesc& desc) noexcept;
const char* slangShaderPostprocessModeName(SlangShaderPostprocessMode mode) noexcept;
// Included in enabled-mode cache identity; "unavailable" when the validator was not built.
// SlangDeadBranches identifies Slang separately from the linked final validator.
const char* slangShaderPostprocessToolVersion(
    SlangShaderPostprocessMode mode = SlangShaderPostprocessMode::DeadBranches) noexcept;
bool slangUsesNativeDescriptorHeap(const SlangShaderDesc& desc) noexcept;

// The selected Vulkan device publishes its heap ABI before runtime compilation.
// Warmup/offline compiles pass explicit strides instead of changing this state.
void setSlangDescriptorHeapShaderStrides(DescriptorHeapShaderStrides strides) noexcept;
DescriptorHeapShaderStrides slangDescriptorHeapShaderStrides() noexcept;

// Successful shader compiles automatically register their complete Slang
// dependency list. File metadata and content hashes are scanned by a background
// worker. Interactive polling only consumes stable edits and never waits for
// the worker or performs file I/O. Passing zero disables the save debounce and
// waits for a fresh background content scan, for deterministic tests only.
std::vector<std::string> pollSlangShaderChanges(
    uint32_t debounceMilliseconds = 150,
    uint32_t retryMilliseconds = 1000);
// Accept the most recently reported source snapshots after every affected
// pipeline has been committed, without file I/O. Newer background observations
// remain unaccepted. Failed reloads intentionally stay dirty and are
// reported again after retryMilliseconds, so fixing a newly-added include is
// sufficient to recover without touching an already-known file.
void acknowledgeSlangShaderChanges();
void resetSlangShaderHotReloadTracking();

// Diagnostics remain available on failure, when no compiled shader is returned.
[[nodiscard]] Result<ShaderCompileResult> compileSlangShaderToSpirv(
    const SlangShaderDesc& desc,
    std::string& diagnostics);
[[nodiscard]] Result<ShaderCompileResult> compileSlangShaderToSpirv(
    const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& cacheOptions,
    std::string& diagnostics);

} // namespace metallic::render
