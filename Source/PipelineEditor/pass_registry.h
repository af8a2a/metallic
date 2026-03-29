#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <json.hpp>

class RenderPass;
struct RenderContext;

struct PassResourceBinding {
    std::string slotKey;
    std::string resourceId;
    std::string resourceName;
    std::string producerPassId;
    std::string producerPassType;
    std::string producerSlotKey;
};

// Pass configuration from JSON
struct PassConfig {
    std::string name;
    std::string type;
    std::vector<PassResourceBinding> inputBindings;
    std::vector<PassResourceBinding> outputBindings;
    bool enabled = true;
    bool sideEffect = false;
    nlohmann::json config;

    const PassResourceBinding* findInputBinding(const std::string& slotKey) const {
        for (const auto& binding : inputBindings) {
            if (binding.slotKey == slotKey) {
                return &binding;
            }
        }
        return nullptr;
    }

    const PassResourceBinding* findOutputBinding(const std::string& slotKey) const {
        for (const auto& binding : outputBindings) {
            if (binding.slotKey == slotKey) {
                return &binding;
            }
        }
        return nullptr;
    }
};

struct PassSlotInfo {
    std::string key;
    std::string displayName;
    bool optional = false;
    std::vector<std::string> allowedResourceKinds;
};

inline PassSlotInfo makePassSlot(std::string key,
                                 std::string displayName,
                                 bool optional,
                                 std::vector<std::string> allowedResourceKinds) {
    return PassSlotInfo{
        std::move(key),
        std::move(displayName),
        optional,
        std::move(allowedResourceKinds)
    };
}

inline PassSlotInfo makeInputSlot(std::string key,
                                  std::string displayName,
                                  bool optional = false,
                                  std::vector<std::string> allowedResourceKinds = {"transient", "imported", "backbuffer"}) {
    return makePassSlot(std::move(key),
                        std::move(displayName),
                        optional,
                        std::move(allowedResourceKinds));
}

inline PassSlotInfo makeOutputSlot(std::string key,
                                   std::string displayName,
                                   bool optional = false,
                                   std::vector<std::string> allowedResourceKinds = {"transient"}) {
    return makePassSlot(std::move(key),
                        std::move(displayName),
                        optional,
                        std::move(allowedResourceKinds));
}

inline PassSlotInfo makeTargetSlot(std::string key,
                                   std::string displayName,
                                   bool optional = false,
                                   std::vector<std::string> allowedResourceKinds = {"imported", "backbuffer"}) {
    return makePassSlot(std::move(key),
                        std::move(displayName),
                        optional,
                        std::move(allowedResourceKinds));
}

// Metadata describing a pass type for the editor
struct PassTypeInfo {
    std::string typeName;
    std::string displayName;
    std::string category;  // e.g., "Geometry", "Lighting", "Post-Process"

    // Stable slot definitions for new instances and validation
    std::vector<PassSlotInfo> inputSlots;
    std::vector<PassSlotInfo> outputSlots;

    // Config schema (JSON schema for pass-specific config)
    nlohmann::json configSchema;

    // Pass type (render, compute, blit)
    enum class PassType { Render, Compute, Blit } passType = PassType::Render;
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
            return std::make_unique<Type>(ctx, w, h); \
        })

// Full registration macro with metadata for pipeline editor
// Usage:
//   REGISTER_RENDER_PASS(VisibilityPass, "Visibility Pass", "Geometry",
//       (std::vector<PassSlotInfo>{}),
//       (std::vector<PassSlotInfo>{makeOutputSlot("visibility", "Visibility"), makeOutputSlot("depth", "Depth")})
//   );
#define REGISTER_RENDER_PASS(Type, DisplayName, Category, Inputs, Outputs) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(ctx, w, h); \
        }, \
        PassTypeInfo{ \
            #Type, DisplayName, Category, Inputs, Outputs, {}, PassTypeInfo::PassType::Render \
        })

#define REGISTER_COMPUTE_PASS(Type, DisplayName, Category, Inputs, Outputs) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(ctx, w, h); \
        }, \
        PassTypeInfo{ \
            #Type, DisplayName, Category, Inputs, Outputs, {}, PassTypeInfo::PassType::Compute \
        })

#define REGISTER_BLIT_PASS(Type, DisplayName, Category, Inputs, Outputs) \
    static PassRegistrar _registrar_##Type(#Type, \
        [](const PassConfig& cfg, const RenderContext& ctx, int w, int h) \
            -> std::unique_ptr<RenderPass> { \
            return std::make_unique<Type>(ctx, w, h); \
        }, \
        PassTypeInfo{ \
            #Type, DisplayName, Category, Inputs, Outputs, {}, PassTypeInfo::PassType::Blit \
        })

// Metadata-only registration (no factory, for editor display only)
// Use when passes have complex constructors that can't use PassConfig
#define REGISTER_PASS_INFO(Type, DisplayName, Category, Inputs, Outputs, PassTypeVal) \
    namespace { \
        static bool _info_registered_##Type = []() { \
            PassTypeInfo info; \
            info.typeName = #Type; \
            info.displayName = DisplayName; \
            info.category = Category; \
            info.inputSlots = Inputs; \
            info.outputSlots = Outputs; \
            info.passType = PassTypeVal; \
            PassRegistry::instance().registerPass(#Type, nullptr, info); \
            return true; \
        }(); \
    }
