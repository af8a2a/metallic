#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <json.hpp>

class RenderPass;
struct RenderContext;

// Pass configuration from JSON
struct PassConfig {
    std::string name;
    std::string type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    bool enabled = true;
    bool sideEffect = false;
    nlohmann::json config;
};

// Metadata describing a pass type for the editor
struct PassTypeInfo {
    std::string typeName;
    std::string displayName;
    std::string category;  // e.g., "Geometry", "Lighting", "Post-Process"

    // Default inputs/outputs for new instances
    std::vector<std::string> defaultInputs;
    std::vector<std::string> defaultOutputs;

    // Config schema (JSON schema for pass-specific config)
    nlohmann::json configSchema;

    // Pass type (render, compute, blit)
    enum class Type { Render, Compute, Blit } passType = Type::Render;
};

// Factory function signature
using PassFactory = std::function<std::unique_ptr<RenderPass>(
    const PassConfig& config,
    const RenderContext& ctx,
    int width, int height)>;

class PassRegistry {
public:
    static PassRegistry& instance();

    // Register a pass type with metadata
    void registerPass(const std::string& typeName, PassFactory factory, const PassTypeInfo& info);

    // Simple registration (backward compatible)
    void registerPass(const std::string& typeName, PassFactory factory);

    bool hasPass(const std::string& typeName) const;

    std::unique_ptr<RenderPass> create(
        const std::string& typeName,
        const PassConfig& config,
        const RenderContext& ctx,
        int width, int height) const;

    // Get all registered type names
    std::vector<std::string> registeredTypes() const;

    // Get type info for a pass
    const PassTypeInfo* getTypeInfo(const std::string& typeName) const;

    // Get all type infos grouped by category
    std::unordered_map<std::string, std::vector<const PassTypeInfo*>> getTypesByCategory() const;

private:
    PassRegistry() = default;
    std::unordered_map<std::string, PassFactory> m_factories;
    std::unordered_map<std::string, PassTypeInfo> m_typeInfos;
};

// Helper class for auto-registration
class PassRegistrar {
public:
    PassRegistrar(const std::string& typeName, PassFactory factory, const PassTypeInfo& info) {
        PassRegistry::instance().registerPass(typeName, std::move(factory), info);
    }

    PassRegistrar(const std::string& typeName, PassFactory factory) {
        PassRegistry::instance().registerPass(typeName, std::move(factory));
    }
};

// ============================================================================
// Registration Macros
// ============================================================================

// Basic registration macro (minimal metadata, for backward compatibility)
#define REGISTER_PASS(Type) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(cfg, ctx, w, h); \
        })

// Full registration macro with metadata for pipeline editor
// Usage:
//   REGISTER_RENDER_PASS(VisibilityPass, "Visibility Pass", "Geometry",
//       (std::vector<std::string>{}),
//       (std::vector<std::string>{"visibility", "depth"})
//   );
#define REGISTER_RENDER_PASS(Type, DisplayName, Category, Inputs, Outputs) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(cfg, ctx, w, h); \
        }, \
        PassTypeInfo{ \
            #Type, DisplayName, Category, Inputs, Outputs, {}, PassTypeInfo::Type::Render \
        })

#define REGISTER_COMPUTE_PASS(Type, DisplayName, Category, Inputs, Outputs) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(cfg, ctx, w, h); \
        }, \
        PassTypeInfo{ \
            #Type, DisplayName, Category, Inputs, Outputs, {}, PassTypeInfo::Type::Compute \
        })

#define REGISTER_BLIT_PASS(Type, DisplayName, Category, Inputs, Outputs) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(cfg, ctx, w, h); \
        }, \
        PassTypeInfo{ \
            #Type, DisplayName, Category, Inputs, Outputs, {}, PassTypeInfo::Type::Blit \
        })

// Metadata-only registration (no factory, for editor display only)
// Use when passes have complex constructors that can't use PassConfig
#define REGISTER_PASS_INFO(Type, DisplayName, Category, Inputs, Outputs, PassType) \
    namespace { \
        static bool _info_registered_##Type = []() { \
            PassTypeInfo info; \
            info.typeName = #Type; \
            info.displayName = DisplayName; \
            info.category = Category; \
            info.defaultInputs = Inputs; \
            info.defaultOutputs = Outputs; \
            info.passType = PassType; \
            PassRegistry::instance().registerPass(#Type, nullptr, info); \
            return true; \
        }(); \
    }
