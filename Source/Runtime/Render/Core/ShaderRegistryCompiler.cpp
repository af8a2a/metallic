#include "Runtime/Render/Core/ShaderRegistry.h"

#include <filesystem>
#include <iomanip>
#include <sstream>

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

} // namespace

ShaderRegistry& ShaderRegistry::instance()
{
    static ShaderRegistry registry;
    return registry;
}

Result<ShaderCompileResult> ShaderRegistry::getShader(const SlangShaderDesc& desc, std::string& log)
{
    return getShader(desc, {}, log);
}

Result<ShaderCompileResult> ShaderRegistry::getShader(const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& options, std::string& log)
{
    auto compiled = compileSlangShaderToSpirv(desc, options, log);
    if (compiled) {
        // Input byte-content identity. Module creation associates the final
        // device code after Vulkan feature finalization with the same group.
        const uint64_t size = compiled->spirv.size() * sizeof(uint32_t);
        auto hash = hashBytes(14695981039346656037ull, &size, sizeof(size));
        hash = hashBytes(hash, compiled->spirv.data(), size_t(size));
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
    return compiled;
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
