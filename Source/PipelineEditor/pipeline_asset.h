#pragma once

#include <array>
#include <string>
#include <vector>

#include <json.hpp>

inline constexpr int kPipelineAssetSchemaVersion = 2;

std::string generatePipelineAssetGuid();

// Resource declaration in pipeline JSON
struct ResourceDecl {
    std::string id;
    std::string name;
    std::string kind;      // "transient", "imported", "backbuffer"
    std::string type;      // "texture", "buffer", "token"
    std::string format;    // optional editor metadata
    std::string size;      // optional editor metadata
    std::string importKey; // runtime import key for imported resources
    std::array<float, 2> editorPos = {0.0f, 0.0f};
};

// Pass declaration in pipeline JSON
struct PassDecl {
    std::string id;
    std::string name;
    std::string type;    // maps to PassRegistry type name
    bool enabled = true;
    bool sideEffect = false;
    nlohmann::json config;  // pass-specific configuration
    std::array<float, 2> editorPos = {0.0f, 0.0f};
};

// Explicit edge declaration between pass slots and resource IDs
struct EdgeDecl {
    std::string id;
    std::string passId;
    std::string slotKey;
    std::string direction; // "input" | "output"
    std::string resourceId;
};

// Complete pipeline asset
struct PipelineAsset {
    int schemaVersion = kPipelineAssetSchemaVersion;
    std::string name;
    std::vector<ResourceDecl> resources;
    std::vector<PassDecl> passes;
    std::vector<EdgeDecl> edges;

    // Load pipeline from JSON file
    static PipelineAsset load(const std::string& path);

    // Save pipeline to JSON file
    void save(const std::string& path) const;

    // Validate the pipeline DAG (check for cycles, missing resources, etc.)
    bool validate(std::string& errorMsg) const;

    // Get topologically sorted pass order
    std::vector<size_t> topologicalSort(bool includeDisabled = true) const;

    ResourceDecl* findResourceById(const std::string& id);
    const ResourceDecl* findResourceById(const std::string& id) const;

    PassDecl* findPassById(const std::string& id);
    const PassDecl* findPassById(const std::string& id) const;

    EdgeDecl* findEdgeById(const std::string& id);
    const EdgeDecl* findEdgeById(const std::string& id) const;

    EdgeDecl* findEdge(const std::string& passId,
                       const std::string& direction,
                       const std::string& slotKey);
    const EdgeDecl* findEdge(const std::string& passId,
                             const std::string& direction,
                             const std::string& slotKey) const;

    void removeEdgesForPass(const std::string& passId);
    void removeEdgesForResource(const std::string& resourceId);
};
