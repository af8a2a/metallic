#include "pass_registry.h"
#include "render_pass.h"
#include <spdlog/spdlog.h>

PassRegistry& PassRegistry::instance() {
    static PassRegistry registry;
    return registry;
}

void PassRegistry::registerPass(const std::string& typeName, PassFactory factory) {
    if (m_factories.count(typeName)) {
        spdlog::warn("PassRegistry: overwriting existing factory for '{}'", typeName);
    }
    m_factories[typeName] = std::move(factory);
    spdlog::debug("PassRegistry: registered '{}'", typeName);
}

bool PassRegistry::hasPass(const std::string& typeName) const {
    return m_factories.count(typeName) > 0;
}

std::unique_ptr<RenderPass> PassRegistry::create(
    const std::string& typeName,
    const PassConfig& config,
    const RenderContext& ctx,
    int width, int height) const
{
    auto it = m_factories.find(typeName);
    if (it == m_factories.end()) {
        spdlog::error("PassRegistry: unknown pass type '{}'", typeName);
        return nullptr;
    }
    return it->second(config, ctx, width, height);
}

std::vector<std::string> PassRegistry::registeredTypes() const {
    std::vector<std::string> types;
    types.reserve(m_factories.size());
    for (const auto& [name, _] : m_factories) {
        types.push_back(name);
    }
    return types;
}
