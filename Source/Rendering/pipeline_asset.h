#pragma once

#include <string>
#include <vector>
#include <json.hpp>

// Resource declaration in pipeline JSON
struct ResourceDecl {
    std::string name;
    std::string type;    // "texture", "buffer"
    std::string format;  // "R32Uint", "Depth32Float", "RGBA16Float", etc.
    std::string size;    // "screen", "512x512", etc.

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(ResourceDecl, name, type, format, size)
};

// Pass declaration in pipeline JSON
struct PassDecl {
    std::string name;
    std::string type;    // maps to PassRegistry type name
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    bool enabled = true;
    bool sideEffect = false;
    nlohmann::json config;  // pass-specific configuration

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(PassDecl, name, type, inputs, outputs, enabled, sideEffect, config)
};

// Complete pipeline asset
struct PipelineAsset {
    std::string name;
    std::vector<ResourceDecl> resources;
    std::vector<PassDecl> passes;

    // Load pipeline from JSON file
    static PipelineAsset load(const std::string& path);

    // Save pipeline to JSON file
    void save(const std::string& path) const;

    // Validate the pipeline DAG (check for cycles, missing resources, etc.)
    bool validate(std::string& errorMsg) const;

    // Get topologically sorted pass order
    std::vector<size_t> topologicalSort() const;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(PipelineAsset, name, resources, passes)
};
