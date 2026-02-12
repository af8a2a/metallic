#include "pass_registry.h"
#include "render_pass.h"
#include <spdlog/spdlog.h>

PassRegistry& PassRegistry::instance() {
    static PassRegistry registry;
    return registry;
}

void PassRegistry::registerPass(const std::string& typeName, PassFactory factory, const PassTypeInfo& info) {
    if (factory) {
        m_factories[typeName] = std::move(factory);
    }
    m_typeInfos[typeName] = info;
    spdlog::debug("PassRegistry: registered '{}' ({})", typeName, info.displayName);
}

void PassRegistry::registerPass(const std::string& typeName, PassFactory factory) {
    if (m_factories.count(typeName)) {
        spdlog::warn("PassRegistry: overwriting existing factory for '{}'", typeName);
    }
    m_factories[typeName] = std::move(factory);

    // Create minimal type info if not already present
    if (m_typeInfos.find(typeName) == m_typeInfos.end()) {
        PassTypeInfo info;
        info.typeName = typeName;
        info.displayName = typeName;
        info.category = "Uncategorized";
        m_typeInfos[typeName] = info;
    }
    spdlog::debug("PassRegistry: registered '{}'", typeName);
}

bool PassRegistry::hasPass(const std::string& typeName) const {
    return m_typeInfos.count(typeName) > 0;
}

std::unique_ptr<RenderPass> PassRegistry::create(
    const std::string& typeName,
    const PassConfig& config,
    const RenderContext& ctx,
    int width, int height) const
{
    auto it = m_factories.find(typeName);
    if (it == m_factories.end() || !it->second) {
        spdlog::error("PassRegistry: no factory for pass type '{}'", typeName);
        return nullptr;
    }
    return it->second(config, ctx, width, height);
}

std::vector<std::string> PassRegistry::registeredTypes() const {
    std::vector<std::string> types;
    types.reserve(m_typeInfos.size());
    for (const auto& [name, _] : m_typeInfos) {
        types.push_back(name);
    }
    return types;
}

const PassTypeInfo* PassRegistry::getTypeInfo(const std::string& typeName) const {
    auto it = m_typeInfos.find(typeName);
    if (it != m_typeInfos.end()) {
        return &it->second;
    }
    return nullptr;
}

std::unordered_map<std::string, std::vector<const PassTypeInfo*>> PassRegistry::getTypesByCategory() const {
    std::unordered_map<std::string, std::vector<const PassTypeInfo*>> result;
    for (const auto& [name, info] : m_typeInfos) {
        result[info.category].push_back(&info);
    }
    return result;
}
