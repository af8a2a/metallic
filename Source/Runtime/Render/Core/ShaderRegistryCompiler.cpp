#include "ShaderRegistry.h"
#include "ShaderRequests.h"

#include <exception>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace metallic::render {
namespace {

uint64_t hashBytes(uint64_t hash, const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) { hash = (hash ^ bytes[i]) * 1099511628211ull; }
    return hash;
}

std::string groupForSource(const char* moduleName)
{
    // Preserve existing production cache files during this migration. New
    // modules automatically receive a stable group; passes cannot omit caching.
    const std::string module = moduleName ? moduleName : "";
    if (module.starts_with("Features/PathTracing/")) {
        return "ScenePathTracePass";
    }
    if (module == "Features/Lighting/SceneRealtimeLighting") { return "RealtimeLightingPass"; }
    if (module == "Features/VisibilityBuffer/VisibilityBufferDeferred") { return "VisibilityBufferDeferredPass"; }
    if (module == "Features/VisibilityBuffer/VisibilityBufferComposite" ||
        module == "Features/GPUDriven/GPUDrivenCulling") { return "VisibilityBufferPass"; }
    if (module == "Features/GPUDriven/GPUDrivenStreamAsset") { return "GPUDrivenStreamAssetPass"; }
    if (module.starts_with("Features/Streaming/")) { return "SceneStreaming"; }
    // Include the full module path's hash so equal basenames cannot collide.
    std::string stem = std::filesystem::path(module).filename().string();
    for (char& c : stem) {
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) { c = '_'; }
    }
    std::ostringstream group;
    group << "ShaderRegistry-" << stem << '-' << std::hex << std::setw(16) << std::setfill('0')
        << hashBytes(14695981039346656037ull, module.data(), module.size());
    return group.str();
}

bool sameSourceIdentity(const PipelineRecipeShader& left, const PipelineRecipeShader& right)
{
    return left.moduleName == right.moduleName &&
        left.sourceEntryPoint == right.sourceEntryPoint &&
        left.entryPoint == right.entryPoint &&
        left.descriptorHeapMode == right.descriptorHeapMode &&
        left.optimizationLevel == right.optimizationLevel &&
        left.debugMode == right.debugMode &&
        left.spirvPostprocess == right.spirvPostprocess &&
        left.spirvPostprocessToolVersion == right.spirvPostprocessToolVersion;
}

bool validIdentityString(const std::string& value)
{
    return !value.empty() && value.find('\0') == std::string::npos;
}

Result<> rejectArtifact(std::string& log, const char* reason)
{
    log += "[ShaderArtifactOverride] Rejected: ";
    log += reason;
    log += '\n';
    return makeError(Error::InvalidArgument);
}

Result<> validateArtifact(const SlangShaderDesc& desc, ShaderArtifactOverrideResult& artifact, std::string& log)
{
    auto& identity = artifact.identity;
    const auto& words = artifact.shader.spirv;
    if (!desc.moduleName || !desc.moduleName[0] || !desc.entryPointName || !desc.entryPointName[0] ||
        !desc.searchPath || !desc.searchPath[0]) {
        return rejectArtifact(log, "the target source request is incomplete");
    }
    if (desc.descriptorHeapMode != SlangDescriptorHeapMode::Default &&
        desc.descriptorHeapMode != SlangDescriptorHeapMode::Mapped &&
        desc.descriptorHeapMode != SlangDescriptorHeapMode::Native) {
        return rejectArtifact(log, "the target descriptor mode is invalid");
    }
    if (slangUsesNativeDescriptorHeap(desc) || identity.descriptorHeapMode != "mapped") {
        return rejectArtifact(log, "artifact overrides currently require mapped descriptors");
    }
    if (!validIdentityString(identity.moduleName) || !validIdentityString(identity.sourceEntryPoint) ||
        identity.sourceEntryPoint != desc.entryPointName || !validIdentityString(identity.entryPoint) ||
        (identity.optimizationLevel != "O0" && identity.optimizationLevel != "O1" &&
            identity.optimizationLevel != "O2" && identity.optimizationLevel != "O3") ||
        (identity.debugMode != "release" && identity.debugMode != "debug" && identity.debugMode != "capture")) {
        return rejectArtifact(log, "actual artifact source, entry point or compiler provenance is missing or invalid");
    }
    if (identity.spirvPostprocess != "none" && identity.spirvPostprocess != "dead-branches" &&
        identity.spirvPostprocess != "slang-dead-branches") {
        return rejectArtifact(log, "the actual artifact postprocess policy is invalid");
    }
    if (identity.spirvPostprocessToolVersion.find('\0') != std::string::npos ||
        (identity.spirvPostprocess != "none" &&
            (!validIdentityString(identity.spirvPostprocessToolVersion) || identity.spirvPostprocessToolVersion == "unavailable"))) {
        return rejectArtifact(log, "actual artifact cleanup tool provenance is missing or invalid");
    }
    if (words.size() < 5 || words[0] != 0x07230203u ||
        words[1] < 0x00010000u || words[1] > 0x00010600u || (words[1] & 0xff0000ffu) ||
        !words[3] || words[4]) {
        return rejectArtifact(log, "the artifact SPIR-V header is invalid");
    }
    size_t entryPointMatches = 0;
    for (size_t offset = 5; offset < words.size();) {
        const size_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 0xffffu;
        if (!count || count > words.size() - offset) {
            return rejectArtifact(log, "the artifact SPIR-V instruction bounds are invalid");
        }
        if (opcode == 15u) { // OpEntryPoint: model, ID, packed name, interface IDs.
            if (count < 4) { return rejectArtifact(log, "the artifact SPIR-V entry declaration is invalid"); }
            std::string entryPoint;
            bool terminated = false;
            for (size_t index = offset + 3; index < offset + count && !terminated; ++index) {
                for (uint32_t byte = 0; byte < 4; ++byte) {
                    const char character = static_cast<char>((words[index] >> (byte * 8)) & 0xffu);
                    if (!character) { terminated = true; break; }
                    entryPoint.push_back(character);
                }
            }
            if (!terminated) { return rejectArtifact(log, "the artifact SPIR-V entry name is unterminated"); }
            if (entryPoint == identity.entryPoint) { ++entryPointMatches; }
        }
        offset += count;
    }
    if (entryPointMatches != 1) {
        return rejectArtifact(log, "the actual artifact entry point is missing or ambiguous in SPIR-V");
    }
    const auto hash = hashBytes(14695981039346656037ull, words.data(), words.size() * sizeof(uint32_t));
    if ((identity.inputSpirvHash && identity.inputSpirvHash != hash) ||
        (!identity.spirv.empty() && identity.spirv != words)) {
        return rejectArtifact(log, "artifact recipe identity does not match the returned SPIR-V");
    }
    identity.inputSpirvHash = hash;
    // Recipe capture obtains the live input words from its ShaderModule call;
    // the process source-identity map need not retain another large blob.
    std::vector<uint32_t>().swap(identity.spirv);
    return {};
}

} // namespace

ShaderRegistry& ShaderRegistry::instance()
{
    static ShaderRegistry registry;
    return registry;
}

ShaderArtifactOverrideScope::ShaderArtifactOverrideScope(ShaderRegistry* registry, uint64_t installation) noexcept
    : registry_(registry), installation_(installation)
{
}

ShaderArtifactOverrideScope::~ShaderArtifactOverrideScope()
{
    reset();
}

ShaderArtifactOverrideScope::ShaderArtifactOverrideScope(ShaderArtifactOverrideScope&& other) noexcept
    : registry_(std::exchange(other.registry_, nullptr)), installation_(std::exchange(other.installation_, 0))
{
}

ShaderArtifactOverrideScope& ShaderArtifactOverrideScope::operator=(ShaderArtifactOverrideScope&& other) noexcept
{
    if (this != &other) {
        reset();
        registry_ = std::exchange(other.registry_, nullptr);
        installation_ = std::exchange(other.installation_, 0);
    }
    return *this;
}

void ShaderArtifactOverrideScope::reset() noexcept
{
    auto* registry = std::exchange(registry_, nullptr);
    const auto installation = std::exchange(installation_, 0);
    if (registry) { registry->removeArtifactOverride(installation); }
}

Result<ShaderArtifactOverrideScope> ShaderRegistry::installArtifactOverrideForTesting(
    std::shared_ptr<ShaderArtifactOverride> artifactOverride)
{
    if (!artifactOverride) { return makeError(Error::InvalidArgument); }
    std::lock_guard lock(sourceMutex_);
    if (artifactOverride_) { return makeError(Error::InvalidArgument); }
    if (nextArtifactOverrideInstallation_ == std::numeric_limits<uint64_t>::max()) {
        return makeError(Error::Failure);
    }
    artifactOverride_ = std::move(artifactOverride);
    artifactOverrideInstallation_ = ++nextArtifactOverrideInstallation_;
    return ShaderArtifactOverrideScope(this, artifactOverrideInstallation_);
}

void ShaderRegistry::removeArtifactOverride(uint64_t installation) noexcept
{
    std::shared_ptr<ShaderArtifactOverride> removed;
    {
        std::lock_guard lock(sourceMutex_);
        if (artifactOverrideInstallation_ != installation) { return; }
        removed = std::move(artifactOverride_);
        artifactOverrideInstallation_ = 0;
    }
    // A loader destructor can release test state that reenters the registry.
    // Its final shared reference is released after unlocking, like callbacks.
}

Result<ShaderCompileResult> ShaderRegistry::getShader(const SlangShaderDesc& desc, std::string& log)
{
    return getShader(desc, {}, log);
}

Result<ShaderCompileResult> ShaderRegistry::getShader(const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& options, std::string& log)
{
    std::shared_ptr<ShaderArtifactOverride> artifactOverride;
    {
        std::lock_guard lock(sourceMutex_);
        artifactOverride = artifactOverride_;
    }
    return getShaderWithOverride(desc, options, log, artifactOverride);
}

Result<ShaderCompileResult> ShaderRegistry::getShaderWithOverride(const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& options, std::string& log,
    const std::shared_ptr<ShaderArtifactOverride>& artifactOverride)
{
    if (artifactOverride) {
        log.clear();
        std::optional<Result<ShaderArtifactOverrideResult>> resolved;
        try {
            resolved = artifactOverride->resolve(desc, log);
        } catch (const std::exception& error) {
            if (options.outCacheHit) { *options.outCacheHit = false; }
            log += std::string("[ShaderArtifactOverride] Resolve failed: ") + error.what() + '\n';
            return makeError(Error::Failure);
        } catch (...) {
            if (options.outCacheHit) { *options.outCacheHit = false; }
            log += "[ShaderArtifactOverride] Resolve failed with an unknown exception\n";
            return makeError(Error::Failure);
        }
        if (resolved) {
            if (options.outCacheHit) { *options.outCacheHit = false; }
            if (!*resolved) {
                if (log.empty()) { log = "[ShaderArtifactOverride] The artifact source rejected this request\n"; }
                return makeError(resolved->error());
            }
            auto artifact = std::move(**resolved);
            auto valid = validateArtifact(desc, artifact, log);
            if (!valid) { return makeError(valid.error()); }
            auto& shader = artifact.shader;
            shader.fromArtifactOverride = true;
            shader.metrics = {};
            const uint64_t size = shader.spirv.size() * sizeof(uint32_t);
            shader.metrics.outputSpirvByteSize = size;
            auto hash = hashBytes(14695981039346656037ull, &size, sizeof(size));
            hash = hashBytes(hash, shader.spirv.data(), size_t(size));
            {
                std::lock_guard lock(sourceMutex_);
                const std::string group = groupForSource(desc.moduleName);
                auto [groupEntry, inserted] = sourceGroups_.try_emplace(hash, group);
                if (!inserted && group < groupEntry->second) { groupEntry->second = group; }
                const auto previous = sourceIdentities_.find(hash);
                if (previous != sourceIdentities_.end()) {
                    artifact.identity.sourceIdentityAmbiguous = artifact.identity.sourceIdentityAmbiguous ||
                        previous->second.sourceIdentityAmbiguous || !sameSourceIdentity(previous->second, artifact.identity);
                }
                // An explicitly supplied artifact is the current capture
                // candidate. Retain ambiguity if these bytes had other producers.
                sourceIdentities_.insert_or_assign(hash, std::move(artifact.identity));
            }
            return std::move(shader);
        }
    }
    auto compiled = compileSlangShaderToSpirv(desc, options, log);
    if (compiled) {
        compiled->fromArtifactOverride = false;
        registerCompiledShader(desc, *compiled);
    }
    return compiled;
}

void ShaderRegistry::registerCompiledShader(const SlangShaderDesc& desc, const ShaderCompileResult& compiled)
{
    // Input byte-content identity. Module creation associates the final
    // device code after Vulkan feature finalization with the same group.
    const uint64_t size = compiled.spirv.size() * sizeof(uint32_t);
    auto hash = hashBytes(14695981039346656037ull, &size, sizeof(size));
    hash = hashBytes(hash, compiled.spirv.data(), size_t(size));
    std::lock_guard lock(sourceMutex_);
    const std::string group = groupForSource(desc.moduleName);
    auto [it, inserted] = sourceGroups_.try_emplace(hash, group);
    if (!inserted && group < it->second) { it->second = group; }
    PipelineRecipeShader identity;
    identity.moduleName = desc.moduleName ? desc.moduleName : "";
    identity.sourceEntryPoint = desc.entryPointName ? desc.entryPointName : "";
    identity.descriptorHeapMode = slangUsesNativeDescriptorHeap(desc) ? "native" : "mapped";
    identity.optimizationLevel = slangShaderOptimizationLevelName(slangShaderOptimizationLevel(desc));
    const auto postprocess = slangShaderPostprocessMode(desc);
    identity.spirvPostprocess = slangShaderPostprocessModeName(postprocess);
    if (postprocess == SlangShaderPostprocessMode::DeadBranches || postprocess == SlangShaderPostprocessMode::SlangDeadBranches) {
        identity.spirvPostprocessToolVersion = slangShaderPostprocessToolVersion(postprocess);
    }
    switch (slangShaderDebugMode()) {
    case SlangShaderDebugMode::Disabled: identity.debugMode = "release"; break;
    case SlangShaderDebugMode::ShaderDebug: identity.debugMode = "debug"; break;
    case SlangShaderDebugMode::CaptureSymbols: identity.debugMode = "capture"; break;
    }
    auto [source, sourceInserted] = sourceIdentities_.try_emplace(hash, identity);
    if (!sourceInserted) {
        const auto& previous = source->second;
        if (previous.moduleName != identity.moduleName ||
            previous.sourceEntryPoint != identity.sourceEntryPoint ||
            previous.descriptorHeapMode != identity.descriptorHeapMode ||
            previous.optimizationLevel != identity.optimizationLevel ||
            previous.debugMode != identity.debugMode ||
            previous.spirvPostprocess != identity.spirvPostprocess ||
            previous.spirvPostprocessToolVersion != identity.spirvPostprocessToolVersion) {
            source->second.sourceIdentityAmbiguous = true;
        }
    }
}

Result<SlangWarmupBatchResult> ShaderRegistry::getWarmupShaders(std::span<const SlangShaderDesc> requests,
    const SlangWarmupBatchOptions& options, std::string& log)
{
    auto result = compileSlangShaderWarmupBatch(requests, options, log);
    if (!result) { return makeError(result.error()); }
    if (!result->completedAllRequests || result->shaders.size() != requests.size()) {
        log += "Incomplete source-qualified warmup result\n"; return makeError(Error::Failure);
    }
    for (size_t index = 0; index < requests.size(); ++index) { registerCompiledShader(requests[index], result->shaders[index]); }
    return result;
}

Result<ShaderRegistryWarmupResult> ShaderRegistry::getWarmupShadersWithOverrides(
    std::span<const ShaderRequest> requests, const SlangWarmupBatchOptions& options, std::string& log)
{
    if (requests.empty() || !options.jobs || options.jobs > 256 || options.cache.outCacheHit) {
        log = "Invalid registry warmup batch";
        return makeError(Error::InvalidArgument);
    }
    std::shared_ptr<ShaderArtifactOverride> artifactOverride;
    {
        std::lock_guard lock(sourceMutex_);
        artifactOverride = artifactOverride_;
    }
    if (!artifactOverride) {
        // The cache transaction requires unique requests. Preserve the native
        // caller's order and duplicate recipes while acquiring each complete
        // effective identity once, including aliases resolved by compiler policy.
        std::vector<std::vector<uint8_t>> uniqueIdentities;
        std::vector<std::vector<uint8_t>> inputIdentities;
        std::vector<size_t> sourceIndices;
        std::vector<std::unique_ptr<ShaderRequestView>> views;
        std::vector<SlangShaderDesc> descriptors;
        for (size_t index = 0; index < requests.size(); ++index) {
            views.push_back(std::make_unique<ShaderRequestView>(requests[index]));
            auto identity = slangShaderRequestIdentity(views.back()->desc(), log);
            if (!identity) { return makeError(identity.error()); }
            inputIdentities.push_back(*identity);
            const auto found = std::find(uniqueIdentities.begin(), uniqueIdentities.end(), *identity);
            sourceIndices.push_back(static_cast<size_t>(found - uniqueIdentities.begin()));
            if (found != uniqueIdentities.end()) { continue; }
            uniqueIdentities.push_back(std::move(*identity));
            descriptors.push_back(views.back()->desc());
        }
        auto uniqueOptions = options;
        if (options.onShaderReady) {
            uniqueOptions.onShaderReady = [&](size_t uniqueIndex, const ShaderCompileResult& shader) {
                for (size_t index = 0; index < sourceIndices.size(); ++index) {
                    if (sourceIndices[index] == uniqueIndex) { options.onShaderReady(index, shader); }
                }
            };
        }
        // The original aliases form part of this acquisition boundary too.
        // Do not publish representative source identities before checking them.
        auto batch = compileSlangShaderWarmupBatch(descriptors, uniqueOptions, log);
        if (!batch) { return makeError(batch.error()); }
        if (!batch->completedAllRequests || batch->shaders.size() != descriptors.size()) {
            log += "Incomplete source-qualified warmup result\n";
            return makeError(Error::Failure);
        }
        // Include every original alias in the acquisition boundary. A process
        // policy change must not let a representative compiled under a new
        // policy masquerade as an explicit-policy request that was deduplicated.
        for (size_t index = 0; index < views.size(); ++index) {
            auto identity = slangShaderRequestIdentity(views[index]->desc(), log);
            if (!identity) { return makeError(identity.error()); }
            if (*identity != inputIdentities[index]) {
                log += "Effective shader request identity changed during warmup acquisition\n";
                return makeError(Error::Failure);
            }
        }
        for (size_t index = 0; index < descriptors.size(); ++index) {
            registerCompiledShader(descriptors[index], batch->shaders[index]);
        }
        if (descriptors.size() != requests.size()) {
            auto uniqueShaders = std::move(batch->shaders);
            batch->shaders.clear();
            batch->shaders.reserve(requests.size());
            for (const size_t index : sourceIndices) { batch->shaders.push_back(uniqueShaders[index]); }
        }
        return ShaderRegistryWarmupResult{.sources = std::move(*batch)};
    }
    // A removed/replaced override cannot split this acquisition between two
    // producers. Keep its callbacks outside the registry lock, as getShader does.
    ShaderRegistryWarmupResult result{.artifactOverrideActive = true};
    result.sources.shaders.reserve(requests.size());
    for (const auto& request : requests) {
        const ShaderRequestView view(request);
        auto shader = getShaderWithOverride(view.desc(), options.cache, log, artifactOverride);
        if (!shader) { return makeError(shader.error()); }
        result.sources.existingCacheHits += shader->metrics.cacheHit ? 1 : 0;
        result.sources.compilerExecutions += shader->metrics.compilerExecuted ? 1 : 0;
        result.sources.shaders.push_back(std::move(*shader));
        if (options.onShaderReady) {
            try { options.onShaderReady(result.sources.shaders.size() - 1, result.sources.shaders.back()); }
            catch (const std::exception& error) {
                log += std::string("Warmup shader consumer failed: ") + error.what() + '\n';
                return makeError(Error::Failure);
            } catch (...) {
                log += "Warmup shader consumer failed with an unknown exception\n";
                return makeError(Error::Failure);
            }
        }
    }
    result.sources.completedAllRequests = true;
    return result;
}

std::string ShaderRegistry::sourceCacheGroup(const char* moduleName)
{
    return groupForSource(moduleName);
}

std::string ShaderRegistry::cacheGroup(uint64_t shaderHash)
{
    std::lock_guard lock(sourceMutex_);
    const auto found = sourceGroups_.find(shaderHash);
    return found == sourceGroups_.end() ? "ShaderRegistry" : found->second;
}

void ShaderRegistry::registerShaderModule(std::span<const uint32_t> spirv, uint64_t deviceShaderHash)
{
    const uint64_t size = spirv.size_bytes();
    auto inputHash = hashBytes(14695981039346656037ull, &size, sizeof(size));
    inputHash = hashBytes(inputHash, spirv.data(), size_t(size));
    std::lock_guard lock(sourceMutex_);
    const auto source = sourceGroups_.find(inputHash);
    if (source == sourceGroups_.end()) { return; }
    const std::string group = source->second;
    auto [it, inserted] = sourceGroups_.try_emplace(deviceShaderHash, group);
    if (!inserted && group < it->second) { it->second = group; }
}

PipelineRecipeShader ShaderRegistry::recipeShader(std::span<const uint32_t> spirv)
{
    const uint64_t size = spirv.size_bytes();
    auto inputHash = hashBytes(14695981039346656037ull, &size, sizeof(size));
    inputHash = hashBytes(inputHash, spirv.data(), size_t(size));
    PipelineRecipeShader shader;
    {
        std::lock_guard lock(sourceMutex_);
        const auto identity = sourceIdentities_.find(inputHash);
        if (identity != sourceIdentities_.end()) { shader = identity->second; }
    }
    shader.spirv.assign(spirv.begin(), spirv.end());
    shader.inputSpirvHash = hashBytes(14695981039346656037ull, spirv.data(), size_t(size));
    return shader;
}

} // namespace metallic::render
