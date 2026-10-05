#pragma once
#include "Runtime/Material/MaterialFeatures.h"

#include <json.hpp>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace metallic::scene { struct RenderMaterial; }

namespace metallic::material {

enum class MaterialValueType { Scalar, Color, Boolean, Enumeration, Texture };

struct MaterialPropertyDesc
{
    std::string name;
    MaterialValueType type;
    nlohmann::json defaultValue;
    double minimum = 0;
    double maximum = 1;
    std::vector<std::string> choices;
};

struct MaterialSchema
{
    std::vector<MaterialPropertyDesc> parameters;
    std::vector<MaterialPropertyDesc> resources;
    std::vector<MaterialPropertyDesc> features;
};

// Authoring definition. Its schema describes semantic values, never GPU offsets.
// The implementation selects built-in OpenPBR or the validated Slab lowering backend.
struct MaterialDefinition
{
    uint32_t version = 1;
    uint32_t definitionVersion = 1;
    std::string implementation = "OpenPBRComposite.Legacy";
    std::string surfaceProgram; // Validated Value/Closure IR authoring source, never generated Slang.
    nlohmann::json valueParameters = nlohmann::json::object(); // Sparse float4 slots 0..3.
    MaterialSchema schema;
    MaterialFeaturePolicies featurePolicies;
};

struct MaterialInstance
{
    uint32_t version = 1;
    uint32_t definitionVersion = 1;
    std::string definition;
    std::optional<std::string> parent;
    nlohmann::json parameters = nlohmann::json::object();
    nlohmann::json resources = nlohmann::json::object();
    nlohmann::json features = nlohmann::json::object();
    nlohmann::json featurePolicies = nlohmann::json::object();
    nlohmann::json valueParameters = nlohmann::json::object();
};

struct ResolvedMaterialInstance
{
    MaterialDefinition definition;
    std::string definitionUri;
    nlohmann::json parameters;
    nlohmann::json resources;
    nlohmann::json features;
    nlohmann::json valueParameters;
    MaterialFeaturePolicies featurePolicies;
    MaterialFeatureResolution featureResolution;
    std::vector<std::string> dependencies;
};

MaterialDefinition defaultOpenPBRDefinition();
bool deserializeMaterialDefinition(std::string_view text, MaterialDefinition& output, std::string& error);
std::string serializeMaterialDefinition(const MaterialDefinition& definition);
bool deserializeMaterialInstance(std::string_view text, MaterialInstance& output, std::string& error);
std::string serializeMaterialInstance(const MaterialInstance& instance);
// v0 is the documented prototype (metallic -> metalness); v1 is the first asset format.
// Unsupported directions/future versions fail without changing the input document.
bool upgradeMaterial(nlohmann::json& document, uint32_t versionFrom, uint32_t versionTo, std::string& error);

class MaterialAssetLibrary
{
public:
    explicit MaterialAssetLibrary(std::filesystem::path assetRoot);
    const std::filesystem::path& root() const { return root_; }
    bool resolve(std::string_view instanceUri, ResolvedMaterialInstance& output, std::string& error) const;
    bool resolve(const MaterialInstance& instance, ResolvedMaterialInstance& output, std::string& error) const;
    bool loadDefinition(std::string_view uri, MaterialDefinition& output, std::string& error) const;
    bool save(std::string_view uri, const MaterialInstance& instance, std::string& error) const;
    // asset:// is canonicalized inside the root, including symlinks. Readers reject missing files;
    // save destinations may not exist yet.
    std::filesystem::path pathFor(std::string_view uri) const;
private:
    std::filesystem::path root_;
};

using MaterialResourceResolver = std::function<int32_t(std::string_view uri)>;
using MaterialResourceEncoder = std::function<std::string(std::string_view slot, int32_t sourceTexture)>;

// Imported names, inactive Fiber fields and existing M2 code stay in the caller's
// RenderMaterial. Only the OpenPBR semantic fields are lowered, through the old upload path.
bool lowerMaterialInstance(const ResolvedMaterialInstance& instance, const MaterialResourceResolver& resolver,
    scene::RenderMaterial& output, std::string& error);
bool createMaterialInstance(const scene::RenderMaterial& source, std::string definitionUri,
    const MaterialDefinition& definition, const MaterialResourceEncoder& encoder,
    MaterialInstance& output, std::string& error);

} // namespace metallic::material
