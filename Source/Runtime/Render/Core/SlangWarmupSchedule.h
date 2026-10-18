#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace metallic::render {

struct WarmupGroupPlan {
    std::vector<size_t> requestIndices;
    uint64_t estimatedWorkUnits = 0;
};

inline uint64_t estimateWarmupEntryWork(const SlangShaderDesc& desc)
{
    const std::string_view module = desc.moduleName ? desc.moduleName : "";
    // Relative conservative units, not milliseconds or a local saved report.
    // This changes dispatch order only; exact session/entry groups are fixed.
    if (module.starts_with("Features/PathTracing/") &&
        module != "Features/PathTracing/SceneSharcMaintenance" && module != "Features/PathTracing/RayMaterialQueue") {
        if (module.find("OpenPBR") != std::string_view::npos) {
            for (const auto& macro : desc.macroDefines) {
                if (macro.name && macro.value && std::string_view(macro.name) == "SHARC_UPDATE" && std::string_view(macro.value) == "1") { return 768; }
            }
            return 512;
        }
        return 256;
    }
    if (module == "Features/Lighting/SceneRealtimeLighting") { return 384; }
    if (module == "Features/VisibilityBuffer/VisibilityBufferDeferred") {
        size_t classCount = 0;
        bool backgroundOnly = false;
        for (const auto& macro : desc.macroDefines) {
            if (macro.name && std::string_view(macro.name) == "MATERIAL_CLASS") {
                ++classCount;
                backgroundOnly = macro.value && std::string_view(macro.value) == "0";
            }
        }
        return classCount == 1 && backgroundOnly ? 16 : 96;
    }
    return 1;
}

inline void estimateAndSortWarmupGroups(std::span<const SlangShaderDesc> requests, std::vector<WarmupGroupPlan>& groups)
{
    for (auto& group : groups) {
        std::vector<std::string_view> roots;
        group.estimatedWorkUnits = 0;
        for (const auto index : group.requestIndices) {
            if (index >= requests.size()) { throw std::invalid_argument("Warmup group contains an invalid request index"); }
            const auto& desc = requests[index];
            group.estimatedWorkUnits += estimateWarmupEntryWork(desc);
            const std::string_view root = desc.moduleName ? desc.moduleName : "";
            if (std::find(roots.begin(), roots.end(), root) == roots.end()) { roots.push_back(root); }
        }
        std::stable_sort(group.requestIndices.begin(), group.requestIndices.end(), [&](size_t left, size_t right) {
            return estimateWarmupEntryWork(requests[left]) > estimateWarmupEntryWork(requests[right]);
        });
        group.estimatedWorkUnits += roots.size() * 8;
    }
    std::stable_sort(groups.begin(), groups.end(), [](const auto& left, const auto& right) {
        return left.estimatedWorkUnits > right.estimatedWorkUnits;
    });
}

} // namespace metallic::render
