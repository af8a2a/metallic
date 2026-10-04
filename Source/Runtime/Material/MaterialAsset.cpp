#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Material/MaterialAssetFields.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include <chrono>
#if defined(_WIN32)
#include <Windows.h>
#endif

namespace metallic::material {
namespace {

using Json = nlohmann::json;

void require(bool condition, std::string message)
{
    if (!condition) { throw std::runtime_error(std::move(message)); }
}

template<class Action>
bool attempt(std::string& error, Action action)
{
    error.clear();
    try { action(); return true; }
    catch (const std::exception& exception) { error = exception.what(); return false; }
}

void keys(const Json& object, std::initializer_list<std::string_view> allowed)
{
    require(object.is_object(), "Material object expected");
    for (const auto& [name, value] : object.items()) {
        require(std::find(allowed.begin(), allowed.end(), name) != allowed.end(), "Unknown material field: " + name);
    }
}

Json parse(std::string_view text)
{
    require(text.size() <= 1024 * 1024, "Material asset exceeds 1 MiB");
    return Json::parse(text, [](int depth, Json::parse_event_t, Json&) {
        require(depth <= 32, "Material JSON is too deeply nested");
        return true;
    });
}

std::string read(const std::filesystem::path& path)
{
    require(std::filesystem::is_regular_file(path), "Missing material asset: " + path.string());
    require(std::filesystem::file_size(path) <= 1024 * 1024, "Material asset exceeds 1 MiB");
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "Cannot read material asset: " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void number(const Json& value, double minimum, double maximum)
{
    require(value.is_number(), "Numeric material value expected");
    const double scalar = value.get<double>();
    require(std::isfinite(scalar) && scalar >= minimum && scalar <= maximum &&
        std::isfinite(static_cast<float>(scalar)), "Material value outside supported range");
}

bool resourceUri(std::string_view uri)
{
    return (uri.starts_with("asset://") && uri.size() > 8) ||
        (uri.starts_with("imported://") && uri.size() > 11);
}

Json texture(Json value)
{
    if (value.is_null()) { return value; }
    if (value.is_string()) { value = Json{{"uri", value}}; }
    keys(value, {"uri", "texCoord", "transform"});
    require(value.contains("uri") && value["uri"].is_string() && resourceUri(value["uri"].get<std::string>()),
        "Texture requires asset:// or imported:// URI; runtime indices are not assets");
    if (!value.contains("texCoord")) { value["texCoord"] = 0; }
    require(value["texCoord"].is_number_integer() && value["texCoord"].get<int64_t>() >= 0 &&
        value["texCoord"].get<int64_t>() <= INT32_MAX, "Texture texCoord must be a nonnegative int32");
    if (!value.contains("transform")) { value["transform"] = {1, 0, 0, 0, 1, 0}; }
    require(value["transform"].is_array() && value["transform"].size() == 6, "Texture transform requires six numbers");
    for (const auto& component : value["transform"]) { number(component, -3.4028234663852886e38, 3.4028234663852886e38); }
    return value;
}

Json validate(const MaterialPropertyDesc& descriptor, Json value)
{
    switch (descriptor.type) {
    case MaterialValueType::Scalar:
        number(value, descriptor.minimum, descriptor.maximum);
        break;
    case MaterialValueType::Color:
        require(value.is_array() && value.size() == 3, "Color requires three components: " + descriptor.name);
        for (const auto& component : value) { number(component, descriptor.minimum, descriptor.maximum); }
        break;
    case MaterialValueType::Boolean:
        require(value.is_boolean(), "Boolean expected: " + descriptor.name);
        break;
    case MaterialValueType::Enumeration:
        require(value.is_string() && std::find(descriptor.choices.begin(), descriptor.choices.end(), value.get<std::string>()) !=
            descriptor.choices.end(), "Unsupported feature value: " + descriptor.name);
        break;
    case MaterialValueType::Texture:
        return texture(std::move(value));
    }
    return value;
}

Json defaults(const std::vector<MaterialPropertyDesc>& schema)
{
    Json result = Json::object();
    for (const auto& descriptor : schema) { result[descriptor.name] = validate(descriptor, descriptor.defaultValue); }
    return result;
}

void overlay(const std::vector<MaterialPropertyDesc>& schema, const Json& values, Json& target)
{
    require(values.is_object(), "Material overrides must be objects");
    for (const auto& [name, value] : values.items()) {
        const auto found = std::find_if(schema.begin(), schema.end(), [&](const auto& entry) { return entry.name == name; });
        require(found != schema.end(), "Unknown semantic material property: " + name);
        target[name] = validate(*found, value);
    }
}

void apply(const MaterialInstance& instance, ResolvedMaterialInstance& result)
{
    require(instance.version == 1 && instance.definitionVersion == result.definition.definitionVersion,
        "Material instance/definition version mismatch; explicit migration required");
    overlay(result.definition.schema.parameters, instance.parameters, result.parameters);
    overlay(result.definition.schema.resources, instance.resources, result.resources);
    overlay(result.definition.schema.features, instance.features, result.features);
    std::string error;
    if (!overlayFeaturePolicies(instance.featurePolicies, result.featurePolicies, error)) { throw std::runtime_error(error); }
}

void validateStructure(const MaterialInstance& instance)
{
    require(instance.version == 1 && instance.definitionVersion > 0, "Unsupported material instance version");
    require(instance.definition.starts_with("asset://") && instance.definition.ends_with(".materialdef"), "Invalid material definition URI");
    require(!instance.parent || (instance.parent->starts_with("asset://") && instance.parent->ends_with(".material")), "Invalid material parent URI");
    require(instance.parameters.is_object() && instance.resources.is_object() && instance.features.is_object(), "Material overrides must be objects");
    const auto schema = defaultOpenPBRDefinition().schema;
    Json scratch = Json::object();
    overlay(schema.parameters, instance.parameters, scratch);
    overlay(schema.resources, instance.resources, scratch);
    overlay(schema.features, instance.features, scratch);
    MaterialFeaturePolicies policies;
    std::string error;
    if (!overlayFeaturePolicies(instance.featurePolicies, policies, error)) { throw std::runtime_error(error); }
}

MaterialInstance instanceFromJson(Json document)
{
    keys(document, {"type", "version", "definitionVersion", "definition", "parent", "parameters", "resources", "features", "featurePolicies"});
    require(document.at("type") == "Metallic.MaterialInstance", "Wrong material asset type");
    require(document.at("version").is_number_unsigned() || document.at("version").is_number_integer(), "Invalid instance version");
    const auto version = document.at("version").get<int64_t>();
    require(version == 0 || version == 1, "Unsupported material instance version");
    if (version == 0) {
        std::string error;
        if (!upgradeMaterial(document, 0, 1, error)) { throw std::runtime_error(error); }
    }
    MaterialInstance result;
    result.definition = document.at("definition").get<std::string>();
    if (document.contains("definitionVersion")) {
        require(document["definitionVersion"].is_number_integer(), "Invalid definition version");
        const auto value = document["definitionVersion"].get<int64_t>();
        require(value > 0 && value <= UINT32_MAX, "Invalid definition version");
        result.definitionVersion = static_cast<uint32_t>(value);
    }
    if (document.contains("parent") && !document["parent"].is_null()) { result.parent = document["parent"].get<std::string>(); }
    result.parameters = document.value("parameters", Json::object());
    result.resources = document.value("resources", Json::object());
    result.features = document.value("features", Json::object());
    result.featurePolicies = document.value("featurePolicies", Json::object());
    validateStructure(result);
    return result;
}

} // namespace

MaterialDefinition defaultOpenPBRDefinition()
{
    MaterialDefinition result;
    const scene::RenderMaterial source;
    result.schema.parameters = {
        {"baseColor", MaterialValueType::Color, {1, 1, 1}},
        {"opacity", MaterialValueType::Scalar, 1.0},
    };
    for (const auto& field : detail::kScalars) {
        result.schema.parameters.push_back({field.name, MaterialValueType::Scalar, source.*field.member, field.minimum, field.maximum});
    }
    for (const auto& field : detail::kColors) {
        const auto& color = source.*field.member;
        result.schema.parameters.push_back({field.name, MaterialValueType::Color, {color.x, color.y, color.z}, 0, field.maximum});
    }
    for (const auto& [name, member] : detail::kTextures) { result.schema.resources.push_back({name, MaterialValueType::Texture, nullptr}); }
    result.schema.features = {
        {"alphaMode", MaterialValueType::Enumeration, "opaque", 0, 1, {"opaque", "mask", "blend"}},
        {"doubleSided", MaterialValueType::Boolean, false},
        {"unlit", MaterialValueType::Boolean, false},
    };
    return result;
}

bool deserializeMaterialDefinition(std::string_view text, MaterialDefinition& output, std::string& error)
{
    return attempt(error, [&] {
        const auto document = parse(text);
        keys(document, {"type", "version", "definitionVersion", "implementation", "defaults"});
        require(document.at("type") == "Metallic.MaterialDefinition" && document.at("version").is_number_integer() && document.at("version") == 1,
            "Unsupported material definition format");
        MaterialDefinition candidate = defaultOpenPBRDefinition();
        candidate.implementation = document.at("implementation").get<std::string>();
        require(candidate.implementation == "OpenPBRComposite.Legacy", "Phase 1 only registers the existing OpenPBR implementation");
        require(document.at("definitionVersion").is_number_integer(), "Invalid definition version");
        const auto version = document.at("definitionVersion").get<int64_t>();
        require(version > 0 && version <= UINT32_MAX, "Invalid definition version");
        candidate.definitionVersion = static_cast<uint32_t>(version);
        const auto values = document.value("defaults", Json::object());
        keys(values, {"parameters", "resources", "features", "featurePolicies"});
        std::string policyError;
        if (!overlayFeaturePolicies(values.value("featurePolicies", Json::object()), candidate.featurePolicies, policyError)) {
            throw std::runtime_error(policyError);
        }
        const auto update = [&](auto& schema, const char* group) {
            auto resolved = defaults(schema);
            overlay(schema, values.value(group, Json::object()), resolved);
            for (auto& field : schema) { field.defaultValue = resolved[field.name]; }
        };
        update(candidate.schema.parameters, "parameters");
        update(candidate.schema.resources, "resources");
        update(candidate.schema.features, "features");
        output = std::move(candidate);
    });
}

std::string serializeMaterialDefinition(const MaterialDefinition& definition)
{
    require(validFeaturePolicies(definition.featurePolicies), "Invalid definition feature policies");
    return Json{{"type", "Metallic.MaterialDefinition"}, {"version", definition.version},
        {"definitionVersion", definition.definitionVersion}, {"implementation", definition.implementation},
        {"defaults", {{"parameters", defaults(definition.schema.parameters)}, {"resources", defaults(definition.schema.resources)},
            {"features", defaults(definition.schema.features)}, {"featurePolicies", serializeFeaturePolicies(definition.featurePolicies)}}}}.dump(4) + '\n';
}

bool deserializeMaterialInstance(std::string_view text, MaterialInstance& output, std::string& error)
{
    return attempt(error, [&] { auto candidate = instanceFromJson(parse(text)); output = std::move(candidate); });
}

std::string serializeMaterialInstance(const MaterialInstance& instance)
{
    validateStructure(instance);
    return Json{{"type", "Metallic.MaterialInstance"}, {"version", instance.version},
        {"definitionVersion", instance.definitionVersion}, {"definition", instance.definition},
        {"parent", instance.parent ? Json(*instance.parent) : Json(nullptr)}, {"parameters", instance.parameters},
        {"resources", instance.resources}, {"features", instance.features}, {"featurePolicies", instance.featurePolicies}}.dump(4) + '\n';
}

bool upgradeMaterial(Json& document, uint32_t versionFrom, uint32_t versionTo, std::string& error)
{
    return attempt(error, [&] {
        require(document.at("type") == "Metallic.MaterialInstance" && document.at("version") == versionFrom,
            "Migration source version/type mismatch");
        require(versionTo == 1 && versionFrom <= versionTo, "Unsupported material migration");
        auto candidate = document;
        if (versionFrom == 0 && candidate.contains("parameters") && candidate["parameters"].contains("metallic")) {
            require(!candidate["parameters"].contains("metalness"), "Ambiguous metallic/metalness migration");
            candidate["parameters"]["metalness"] = candidate["parameters"]["metallic"];
            candidate["parameters"].erase("metallic");
        }
        candidate["version"] = 1;
        if (!candidate.contains("definitionVersion")) { candidate["definitionVersion"] = 1; }
        document = std::move(candidate);
    });
}

MaterialAssetLibrary::MaterialAssetLibrary(std::filesystem::path assetRoot)
    : root_(std::filesystem::weakly_canonical(std::filesystem::absolute(assetRoot)))
{
}

std::filesystem::path MaterialAssetLibrary::pathFor(std::string_view uri) const
{
    require(uri.starts_with("asset://") && uri.size() > 8, "Expected asset:// URI");
    const auto relative = std::filesystem::path(uri.substr(8));
    require(!relative.is_absolute() && !relative.has_root_name(), "Absolute asset path is invalid");
    const auto path = std::filesystem::weakly_canonical(root_ / relative);
    const auto local = path.lexically_relative(root_);
    require(!local.empty() && *local.begin() != ".." && local != ".", "Asset URI escapes its root");
    return path;
}

bool MaterialAssetLibrary::loadDefinition(std::string_view uri, MaterialDefinition& output, std::string& error) const
{
    return attempt(error, [&] {
        require(uri.ends_with(".materialdef"), "Expected .materialdef asset");
        MaterialDefinition candidate;
        std::string diagnostic;
        if (!deserializeMaterialDefinition(read(pathFor(uri)), candidate, diagnostic)) { throw std::runtime_error(diagnostic); }
        output = std::move(candidate);
    });
}

bool MaterialAssetLibrary::resolve(std::string_view instanceUri, ResolvedMaterialInstance& output, std::string& error) const
{
    return attempt(error, [&] {
        require(instanceUri.ends_with(".material"), "Expected .material asset");
        const auto instance = instanceFromJson(parse(read(pathFor(instanceUri))));
        ResolvedMaterialInstance candidate;
        std::string diagnostic;
        if (!resolve(instance, candidate, diagnostic)) { throw std::runtime_error(diagnostic); }
        candidate.dependencies.push_back(std::string(instanceUri));
        output = std::move(candidate);
    });
}

bool MaterialAssetLibrary::resolve(const MaterialInstance& instance, ResolvedMaterialInstance& output, std::string& error) const
{
    return attempt(error, [&] {
        validateStructure(instance);
        std::vector<MaterialInstance> chain{instance};
        std::set<std::filesystem::path> visited;
        ResolvedMaterialInstance candidate;
        candidate.definitionUri = instance.definition;
        std::string diagnostic;
        if (!loadDefinition(instance.definition, candidate.definition, diagnostic)) { throw std::runtime_error(diagnostic); }
        candidate.dependencies.push_back(instance.definition);
        const auto definitionPath = pathFor(instance.definition);
        while (chain.back().parent) {
            require(chain.size() < 32, "Material inheritance exceeds 32 instances");
            const auto uri = *chain.back().parent;
            const auto path = pathFor(uri);
            require(visited.insert(path).second, "Material parent cycle: " + uri);
            auto parent = instanceFromJson(parse(read(path)));
            require(pathFor(parent.definition) == definitionPath, "Parent material uses a different definition");
            candidate.dependencies.push_back(uri);
            chain.push_back(std::move(parent));
        }
        candidate.parameters = defaults(candidate.definition.schema.parameters);
        candidate.resources = defaults(candidate.definition.schema.resources);
        candidate.features = defaults(candidate.definition.schema.features);
        candidate.featurePolicies = candidate.definition.featurePolicies;
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) { apply(*it, candidate); }
        // Analyze resource presence without loading textures or assigning GPU handles.
        scene::RenderMaterial semantic;
        if (!lowerMaterialInstance(candidate, [](std::string_view) { return 0; }, semantic, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        candidate.featureResolution = resolveMaterialFeatures(semantic);
        output = std::move(candidate);
    });
}

bool MaterialAssetLibrary::save(std::string_view uri, const MaterialInstance& instance, std::string& error) const
{
    return attempt(error, [&] {
        require(uri.ends_with(".material"), "Expected .material asset");
        ResolvedMaterialInstance resolved;
        std::string diagnostic;
        if (!resolve(instance, resolved, diagnostic)) { throw std::runtime_error(diagnostic); }
        const auto path = pathFor(uri);
        // Reject a self-reference even before the destination exists.
        for (const auto& dependency : resolved.dependencies) {
            require(pathFor(dependency) != path, "Saving this instance would create a parent cycle");
        }
        const auto text = serializeMaterialInstance(instance);
        std::filesystem::create_directories(path.parent_path());
        auto temporary = path;
        temporary += ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        require(!std::filesystem::exists(temporary), "Material temporary file already exists");
        try {
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            require(bool(stream), "Cannot write material asset: " + path.string());
            stream << text;
            stream.close();
            require(bool(stream), "Material asset write failed: " + path.string());
#if defined(_WIN32)
            require(MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,
                "Could not atomically replace material asset");
#else
            std::filesystem::rename(temporary, path);
#endif
        } catch (...) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw;
        }
    });
}

bool lowerMaterialInstance(const ResolvedMaterialInstance& instance, const MaterialResourceResolver& resolver,
    scene::RenderMaterial& output, std::string& error)
{
    return attempt(error, [&] {
        require(instance.definition.implementation == "OpenPBRComposite.Legacy", "Unsupported material implementation");
        require(!output.rtxcrHair, "OpenPBR asset cannot replace a Fiber material");
        auto candidate = output;
        candidate.featurePolicies = instance.featurePolicies;
        const auto& p = instance.parameters;
        candidate.baseColorFactor = float4(p.at("baseColor")[0].get<float>(), p.at("baseColor")[1].get<float>(),
            p.at("baseColor")[2].get<float>(), p.at("opacity").get<float>());
        for (const auto& field : detail::kScalars) { candidate.*field.member = p.at(field.name).get<float>(); }
        for (const auto& field : detail::kColors) {
            const auto& value = p.at(field.name);
            candidate.*field.member = float3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        }
        const auto mode = instance.features.at("alphaMode").get<std::string>();
        require(mode == "opaque" || mode == "mask" || mode == "blend", "Invalid alphaMode");
        candidate.alphaMode = mode == "opaque" ? "OPAQUE" : mode == "mask" ? "MASK" : "BLEND";
        candidate.doubleSided = instance.features.at("doubleSided").get<bool>();
        candidate.unlit = instance.features.at("unlit").get<bool>();
        for (const auto& [name, member] : detail::kTextures) {
            auto& info = candidate.*member;
            info = {};
            const auto value = texture(instance.resources.at(name));
            if (value.is_null()) { continue; }
            require(bool(resolver), "Material texture resolver is missing");
            info.textureIndex = resolver(value.at("uri").get<std::string>());
            require(info.textureIndex >= 0, "Unresolved texture: " + value.at("uri").get<std::string>());
            info.texCoord = value.at("texCoord").get<int32_t>();
            info.uvTransform = value.at("transform").get<std::array<float, 6>>();
        }
        require(scene::validMaterialProperties(candidate), "Resolved material is invalid");
        output = std::move(candidate);
    });
}

bool createMaterialInstance(const scene::RenderMaterial& source, std::string definitionUri,
    const MaterialDefinition& definition, const MaterialResourceEncoder& encoder,
    MaterialInstance& output, std::string& error)
{
    return attempt(error, [&] {
        require(!source.rtxcrHair && scene::validMaterialProperties(source), "Invalid OpenPBR source material");
        require(source.valueProgram.empty(), "Custom Value Program code belongs to its definition; Phase 1 export does not serialize shader code in an instance");
        MaterialInstance candidate;
        candidate.definition = std::move(definitionUri);
        candidate.definitionVersion = definition.definitionVersion;
        Json parameters = {{"baseColor", {source.baseColorFactor.x, source.baseColorFactor.y, source.baseColorFactor.z}},
            {"opacity", source.baseColorFactor.w}};
        for (const auto& field : detail::kScalars) { parameters[field.name] = source.*field.member; }
        for (const auto& field : detail::kColors) {
            const auto& color = source.*field.member;
            parameters[field.name] = {color.x, color.y, color.z};
        }
        Json resources = Json::object();
        for (const auto& [name, member] : detail::kTextures) {
            const auto& info = source.*member;
            resources[name] = nullptr;
            if (info.textureIndex < 0) { continue; }
            require(bool(encoder), "Material resource encoder is missing");
            resources[name] = texture({{"uri", encoder(name, info.textureIndex)},
                {"texCoord", info.texCoord}, {"transform", info.uvTransform}});
        }
        const Json features = {{"alphaMode", source.alphaMode == "OPAQUE" ? "opaque" : source.alphaMode == "MASK" ? "mask" : "blend"},
            {"doubleSided", source.doubleSided}, {"unlit", source.unlit}};
        const auto sparse = [](const auto& schema, const Json& values) {
            Json result = Json::object();
            for (const auto& field : schema) {
                const auto value = validate(field, values.at(field.name));
                if (value != validate(field, field.defaultValue)) { result[field.name] = value; }
            }
            return result;
        };
        candidate.parameters = sparse(definition.schema.parameters, parameters);
        candidate.resources = sparse(definition.schema.resources, resources);
        candidate.features = sparse(definition.schema.features, features);
        const auto sourcePolicies = serializeFeaturePolicies(source.featurePolicies);
        const auto defaultPolicies = serializeFeaturePolicies(definition.featurePolicies);
        for (const auto& [name, value] : sourcePolicies.items()) {
            if (value != defaultPolicies.at(name)) { candidate.featurePolicies[name] = value; }
        }
        validateStructure(candidate);
        output = std::move(candidate);
    });
}

} // namespace metallic::material
