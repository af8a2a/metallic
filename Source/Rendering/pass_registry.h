#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <json.hpp>

class RenderPass;
struct RenderContext;

// Configuration passed to pass factories from JSON
struct PassConfig {
    std::string name;
    std::string type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    bool enabled = true;
    bool sideEffect = false;
    nlohmann::json config;  // pass-specific configuration
};

// Factory function signature for creating passes
using PassFactory = std::function<std::unique_ptr<RenderPass>(
    const PassConfig& config,
    const RenderContext& ctx,
    int width, int height)>;

class PassRegistry {
public:
    static PassRegistry& instance();

    void registerPass(const std::string& typeName, PassFactory factory);
    bool hasPass(const std::string& typeName) const;
    std::unique_ptr<RenderPass> create(
        const std::string& typeName,
        const PassConfig& config,
        const RenderContext& ctx,
        int width, int height) const;

    std::vector<std::string> registeredTypes() const;

private:
    PassRegistry() = default;
    std::unordered_map<std::string, PassFactory> m_factories;
};

// Helper class for auto-registration via static initialization
class PassRegistrar {
public:
    PassRegistrar(const std::string& typeName, PassFactory factory) {
        PassRegistry::instance().registerPass(typeName, std::move(factory));
    }
};

// Macro for auto-registering a pass type
// Usage: REGISTER_PASS(MyPassClass) in the .cpp file
#define REGISTER_PASS(Type) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(cfg, ctx, w, h); \
        })
