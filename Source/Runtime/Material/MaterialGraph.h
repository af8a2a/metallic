#pragma once
#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Material/MaterialValueIR.h"

namespace metallic::material {

struct MaterialGraphNodeDesc
{
    std::string kind;
    std::vector<std::string> inputs;
    bool surface = false;
};

struct CompiledMaterialFrontend
{
    MaterialDefinition definition;
    nlohmann::json reflection;
    uint64_t signature = 0; // Canonical IR identity; executable keys also include target/ABI.
};

const std::vector<MaterialGraphNodeDesc>& materialGraphNodes();
nlohmann::json makeMaterialGraphNode(std::string_view kind, uint32_t id);
nlohmann::json defaultMaterialGraph(bool slab = false);
// Transactional frontends. Graph positions, labels and parameter defaults are not code identity.
bool compileMaterialGraph(const nlohmann::json& graph, CompiledMaterialFrontend& output, std::string& error);
// Read-only Slang SDK expression profile, not arbitrary Slang modules. Unsupported syntax fails closed.
bool compileSlangMaterial(std::string_view source, const nlohmann::json& parameterDefaults,
    CompiledMaterialFrontend& output, std::string& error);

} // namespace metallic::material
