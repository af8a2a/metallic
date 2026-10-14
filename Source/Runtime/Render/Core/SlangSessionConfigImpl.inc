// Included inside metallic::render, after the original private helpers.
bool OwnedSlangSessionConfig::initialize(slang::IGlobalSession* globalSession, const ResolvedSlangSessionPolicy& policy,
    ISlangFileSystem* fileSystem, std::string& log)
{
    const SlangShaderDesc& desc = source_.descriptor;
    SlangShaderDesc resolvedDesc = desc;
    resolvedDesc.descriptorHeapMode = policy.native ? SlangDescriptorHeapMode::Native : SlangDescriptorHeapMode::Mapped;
    resolvedDesc.descriptorHeapStrides = policy.native ? policy.nativeStrides : DescriptorHeapShaderStrides{};
    const bool native = policy.native;
    const SlangShaderDebugMode debugMode = policy.debugMode;
    const SlangOptimizationLevel optimizationLevel = policy.optimizationLevel;
    const std::optional<SlangShaderPostprocessMode> effectivePostprocess(policy.postprocessMode);
    const SlangShaderCacheOptions cacheOptions{.reportDetailedCompilerPerf = policy.reportDetailedCompilerPerf};
#include "SlangSessionDesc.inc"
    return true;
}
