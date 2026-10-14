#pragma once

// Internal module, included in SlangCompiler.cpp after the existing
// private policy/path helpers. It does not resolve a second compiler policy.
// Callers pass the policy already resolved by the ordinary compiler.
#include <optional>
#include "SlangCompiler.h"
#include <slang.h>
#include <slang-tag-version.h>
#include <cstring>
#include <filesystem>
#include <memory>

namespace metallic::render {

struct OwnedSlangRequest {
    std::optional<std::string> module, entry, searchPath, profile;
    std::vector<std::optional<std::string>> additionalPaths, capabilities;
    std::vector<std::pair<std::optional<std::string>, std::optional<std::string>>> macros;
    std::vector<const char*> pathPointers, capabilityPointers;
    std::vector<SlangMacroDefine> macroPointers;
    SlangShaderDesc descriptor{};

    explicit OwnedSlangRequest(const SlangShaderDesc& input)
    {
        const auto copy = [](const char* text) -> std::optional<std::string> {
            return text ? std::optional<std::string>(text) : std::nullopt;
        };
        module = copy(input.moduleName); entry = copy(input.entryPointName);
        searchPath = copy(input.searchPath); profile = copy(input.profileName);
        for (const auto* path : input.additionalSearchPaths) { additionalPaths.push_back(copy(path)); }
        for (const auto* capability : input.capabilities) { capabilities.push_back(copy(capability)); }
        for (const auto& macro : input.macroDefines) { macros.emplace_back(copy(macro.name), copy(macro.value)); }
        for (const auto& path : additionalPaths) { pathPointers.push_back(path ? path->c_str() : nullptr); }
        for (const auto& capability : capabilities) { capabilityPointers.push_back(capability ? capability->c_str() : nullptr); }
        for (const auto& [name, value] : macros) { macroPointers.push_back({name ? name->c_str() : nullptr, value ? value->c_str() : nullptr}); }
        descriptor = input;
        descriptor.moduleName = module ? module->c_str() : nullptr;
        descriptor.entryPointName = entry ? entry->c_str() : nullptr;
        descriptor.searchPath = searchPath ? searchPath->c_str() : nullptr;
        descriptor.profileName = profile ? profile->c_str() : nullptr;
        descriptor.additionalSearchPaths = pathPointers;
        descriptor.capabilities = capabilityPointers;
        descriptor.macroDefines = macroPointers;
    }
    OwnedSlangRequest(const OwnedSlangRequest&) = delete;
    OwnedSlangRequest& operator=(const OwnedSlangRequest&) = delete;
    OwnedSlangRequest(OwnedSlangRequest&&) = delete;
    OwnedSlangRequest& operator=(OwnedSlangRequest&&) = delete;
};

struct ResolvedSlangSessionPolicy {
    SlangShaderDebugMode debugMode = SlangShaderDebugMode::Disabled;
    SlangOptimizationLevel optimizationLevel = SLANG_OPTIMIZATION_LEVEL_DEFAULT;
    SlangShaderPostprocessMode postprocessMode = SlangShaderPostprocessMode::None;
    bool native = false;
    DescriptorHeapShaderStrides nativeStrides{};
    bool reportDetailedCompilerPerf = false;
};

class OwnedSlangSessionConfig {
public:
    static std::unique_ptr<OwnedSlangSessionConfig> create(slang::IGlobalSession* globalSession,
        const SlangShaderDesc& request, const ResolvedSlangSessionPolicy& policy,
        ISlangFileSystem* fileSystem, std::string& log)
    {
        auto result = std::unique_ptr<OwnedSlangSessionConfig>(new OwnedSlangSessionConfig(request));
        if (!result->initialize(globalSession, policy, fileSystem, log)) { return {}; }
        return result;
    }
    const slang::SessionDesc& desc() const noexcept { return sessionDesc_; }
    const std::vector<std::filesystem::path>& normalizedSearchPaths() const noexcept { return normalizedSearchPaths_; }
    const SlangShaderDesc& source() const noexcept { return source_.descriptor; }
    OwnedSlangSessionConfig(const OwnedSlangSessionConfig&) = delete;
    OwnedSlangSessionConfig& operator=(const OwnedSlangSessionConfig&) = delete;
    OwnedSlangSessionConfig(OwnedSlangSessionConfig&&) = delete;
    OwnedSlangSessionConfig& operator=(OwnedSlangSessionConfig&&) = delete;

    // Exact value bytes, not a hash-only equality key. Profile names are
    // included rather than comparing handles from unrelated worker globals.
    std::vector<uint8_t> effectiveKey() const
    {
        std::vector<uint8_t> key;
        const auto integer = [&](auto value) {
            const auto* begin = reinterpret_cast<const uint8_t*>(&value);
            key.insert(key.end(), begin, begin + sizeof(value));
        };
        const auto text = [&](const char* value) {
            const bool present = value != nullptr; integer(present);
            const uint64_t size = value ? std::strlen(value) : 0; integer(size);
            if (size) { key.insert(key.end(), value, value + size); }
        };
        integer(targetDesc_.structureSize); integer(targetDesc_.format);
        text(source_.descriptor.profileName ? source_.descriptor.profileName : kDefaultSlangProfileName);
        integer(targetDesc_.flags); integer(targetDesc_.floatingPointMode); integer(targetDesc_.lineDirectiveMode);
        integer(targetDesc_.forceGLSLScalarBufferLayout); integer(targetDesc_.compilerOptionEntryCount);
        integer(sessionDesc_.structureSize); integer(sessionDesc_.targetCount); integer(sessionDesc_.flags);
        integer(sessionDesc_.defaultMatrixLayoutMode); integer(sessionDesc_.enableEffectAnnotations);
        integer(sessionDesc_.allowGLSLSyntax); integer(sessionDesc_.skipSPIRVValidation);
        integer(sessionDesc_.preprocessorMacroCount); integer(sessionDesc_.searchPathCount);
        for (const auto& path : searchPathStorage_) { text(path.c_str()); }
        integer(sessionDesc_.compilerOptionEntryCount);
        for (const auto& option : compilerOptions_) {
            integer(option.name); integer(option.value.kind); integer(option.value.intValue0); integer(option.value.intValue1);
            text(option.value.stringValue0); text(option.value.stringValue1);
        }
        text(SLANG_VERSION_NUMERIC);
        // fileSystem address is intentionally excluded: a warmup run shares
        // one independently qualified immutable snapshot across all workers.
        return key;
    }

private:
    explicit OwnedSlangSessionConfig(const SlangShaderDesc& request) : source_(request) {}
    bool initialize(slang::IGlobalSession* globalSession, const ResolvedSlangSessionPolicy& policy,
        ISlangFileSystem* fileSystem, std::string& log);
    // Non-moving storage keeps all SDK option/string/TargetDesc pointers valid
    // until callers destroy their ISession before destroying this object.
    OwnedSlangRequest source_;
    slang::TargetDesc targetDesc_{};
    std::vector<std::filesystem::path> normalizedSearchPaths_;
    std::vector<std::string> searchPathStorage_;
    std::vector<const char*> searchPaths_;
    std::vector<slang::CompilerOptionEntry> compilerOptions_;
    slang::SessionDesc sessionDesc_{};
};

} // namespace metallic::render
