#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Material/MaterialAssetFields.h"
#include "Runtime/Material/MaterialValueIR.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
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
    keys(value, {"uri", "texCoord", "transform", "colorSpace"});
    if (value.contains("colorSpace")) {
        render::ColorSpaceDesc source;
        require(value["colorSpace"].is_string() && render::parseColorSpace(value["colorSpace"].get<std::string>(), source),
            "Unsupported texture source colorSpace");
    }
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

Json terrainTexture(const Json& source, bool color)
{
    auto value = texture(source);
    require(!value.is_null() && value.at("texCoord") == 0, "Terrain textures require UV0 and a texture resource");
    require(color || !value.contains("colorSpace"), "Terrain weight, normal and roughness are linear data textures");
    return value;
}

constexpr std::string_view kTerrainChannels = "RGBA";
constexpr std::string_view kTerrainImplementation = "OpenPBR.TerrainLayers";

Json terrainLayers(Json value)
{
    if (value.is_null()) { return value; }
    keys(value, {"weights", "weightMaps", "defaultLayer", "layers"});
    const bool paired = value.contains("weightMaps");
    require(paired != value.contains("weights"), "Terrain material requires either weights or weightMaps");
    if (paired) {
        auto& maps = value.at("weightMaps");
        require(maps.is_array() && maps.size() == 2, "Terrain material weightMaps requires exactly two textures");
        for (auto& map : maps) { map = terrainTexture(map, false); }
        require(maps[0].at("transform") == maps[1].at("transform"), "Terrain weight maps require the same UV transform");
    } else {
        value["weights"] = terrainTexture(value.at("weights"), false);
    }
    auto& layers = value.at("layers");
    require(layers.is_array() && !layers.empty() && layers.size() <= kTerrainChannels.size() * (paired ? 2 : 1),
        paired ? "Paired terrain weight maps support one to eight layers" : "Terrain material requires one to four layers");
    std::set<std::string> names;
    std::set<size_t> slots;
    for (auto& layer : layers) {
        keys(layer, {"name", "weightMap", "channel", "baseColor", "normal", "roughness", "triplanar"});
        const auto name = layer.at("name").get<std::string>();
        require(!name.empty() && names.insert(name).second, "Terrain layer names must be nonempty and unique");
        require(paired == layer.contains("weightMap"), "A terrain layer requires weightMap only with paired weight maps");
        if (paired) {
            const auto& map = layer.at("weightMap");
            require(map.is_number_integer() && (map == 0 || map == 1), "Terrain layer weightMap must be 0 or 1");
        }
        const auto channel = layer.at("channel").get<std::string>();
        const size_t component = kTerrainChannels.find(channel);
        require(channel.size() == 1 && component != std::string_view::npos, "Terrain layers require an R, G, B or A channel");
        const size_t map = layer.value("weightMap", 0u);
        require(slots.insert(map * kTerrainChannels.size() + component).second, "Terrain weight channels must be unique within each map");
        layer["baseColor"] = terrainTexture(layer.at("baseColor"), true);
        layer["normal"] = terrainTexture(layer.at("normal"), false);
        layer["roughness"] = terrainTexture(layer.at("roughness"), false);
        if (const auto projection = layer.find("triplanar"); projection != layer.end()) {
            keys(*projection, {"worldTileSize", "normalStrength", "baseColorTextureMix", "baseColorMeanLinear"});
            const auto& tileSize = projection->at("worldTileSize");
            require(tileSize.is_array() && tileSize.size() == 2, "Terrain triplanar worldTileSize requires two dimensions in meters");
            constexpr double smallestScale = std::numeric_limits<float>::min();
            for (const auto& extent : tileSize) { number(extent, smallestScale, 1.0 / smallestScale); }
            number(projection->at("normalStrength"), 0.0, 1.0);
            number(projection->at("baseColorTextureMix"), 0.0, 1.0);
            const auto& color = projection->at("baseColorMeanLinear");
            require(color.is_array() && color.size() == 3, "Terrain triplanar baseColorMeanLinear requires linear Rec.709 RGB");
            for (const auto& component : color) { number(component, 0.0, 1.0); }
        }
    }
    require(names.contains(value.at("defaultLayer").get<std::string>()), "Terrain defaultLayer must name a declared layer");
    return value;
}

scene::RenderTextureInfo lowerTexture(const Json& value, const MaterialResourceResolver& resolver,
    render::TextureColorMetadata metadata)
{
    scene::RenderTextureInfo result{.colorMetadata = metadata};
    if (value.is_null()) { return result; }
    require(bool(resolver), "Material texture resolver is missing");
    const auto uri = value.at("uri").get<std::string>();
    result.textureIndex = resolver(uri);
    require(result.textureIndex >= 0, "Unresolved texture: " + uri);
    result.texCoord = value.at("texCoord").get<int32_t>();
    result.uvTransform = value.at("transform").get<std::array<float, 6>>();
    if (value.contains("colorSpace")) {
        require(result.colorMetadata.semantic == render::TextureSemantic::Color, "Data textures cannot declare colorSpace");
        require(render::parseColorSpace(value["colorSpace"].get<std::string>(), result.colorMetadata.source), "Unsupported colorSpace");
    }
    return result;
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
    if (!instance.terrainLayers.is_null()) { result.terrainLayers = instance.terrainLayers; }
    overlay(result.definition.schema.features, instance.features, result.features);
    for (const auto& [slot, value] : instance.valueParameters.items()) { result.valueParameters[slot] = value; }
    std::string error;
    if (!overlayFeaturePolicies(instance.featurePolicies, result.featurePolicies, error)) { throw std::runtime_error(error); }
}

void validateValueParameters(const Json& parameters)
{
    require(parameters.is_object(), "valueParameters must be a sparse object of float4 slots");
    for (const auto& [slot, value] : parameters.items()) {
        require(slot.size() == 1 && slot[0] >= '0' && slot[0] <= '3' && value.is_array() && value.size() == 4,
            "Value parameter requires slot 0..3 and four components");
        for (const auto& component : value) { number(component, -1e6, 1e6); }
    }
}

void validateStructure(const MaterialInstance& instance)
{
    require(instance.version == 1 && instance.definitionVersion > 0, "Unsupported material instance version");
    require(instance.definition.starts_with("asset://") && instance.definition.ends_with(".materialdef"), "Invalid material definition URI");
    require(!instance.parent || (instance.parent->starts_with("asset://") && instance.parent->ends_with(".material")), "Invalid material parent URI");
    require(instance.parameters.is_object() && instance.resources.is_object() && instance.features.is_object(), "Material overrides must be objects");
    validateValueParameters(instance.valueParameters);
    (void)terrainLayers(instance.terrainLayers);
    auto schema = defaultOpenPBRDefinition().schema;
    // The definition URI is resolved later. Structural parsing accepts known
    // fields from both domains; apply() enforces the resolved definition schema.
    const auto fiber = defaultRTXCRChiangDefinition();
    schema.parameters.insert(schema.parameters.end(), fiber.schema.parameters.begin(), fiber.schema.parameters.end());
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
    keys(document, {"type", "version", "definitionVersion", "definition", "parent", "parameters", "resources",
        "terrainLayers", "features", "featurePolicies", "valueParameters"});
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
    result.terrainLayers = document.value("terrainLayers", Json());
    result.features = document.value("features", Json::object());
    result.featurePolicies = document.value("featurePolicies", Json::object());
    result.valueParameters = document.value("valueParameters", Json::object());
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

MaterialDefinition defaultRTXCRChiangDefinition()
{
    MaterialDefinition result;
    result.implementation = "RTXCRChiang.DOTS";
    const scene::RenderMaterial source;
    for (const auto& field : detail::kFiberScalars) {
        result.schema.parameters.push_back({field.name, MaterialValueType::Scalar, source.*field.member, field.minimum, field.maximum});
    }
    result.schema.features = {{"doubleSided", MaterialValueType::Boolean, false}};
    return result;
}

bool deserializeMaterialDefinition(std::string_view text, MaterialDefinition& output, std::string& error)
{
    return attempt(error, [&] {
        const auto document = parse(text);
        keys(document, {"type", "version", "definitionVersion", "implementation", "defaults", "surfaceProgram"});
        require(document.at("type") == "Metallic.MaterialDefinition" && document.at("version").is_number_integer() && document.at("version") == 1,
            "Unsupported material definition format");
        const bool fiber = document.at("implementation") == "RTXCRChiang.DOTS";
        MaterialDefinition candidate = fiber ? defaultRTXCRChiangDefinition() : defaultOpenPBRDefinition();
        candidate.implementation = document.at("implementation").get<std::string>();
        // Migrate persisted definitions at the authoring boundary. Runtime
        // registration and newly saved assets use only the Surface identity.
        if (candidate.implementation == "OpenPBRComposite.Legacy") {
            candidate.implementation = "OpenPBRComposite.Surface";
        }
        require(candidate.implementation == "OpenPBRComposite.Surface" || candidate.implementation == kTerrainImplementation ||
            candidate.implementation == "Slab.Surface" ||
            candidate.implementation == "OpenPBR.Value" || fiber, "Unsupported material implementation");
        if (candidate.implementation == "OpenPBR.Value" || candidate.implementation == "Slab.Surface") {
            require(document.contains("surfaceProgram") && document["surfaceProgram"].is_object(), "Program definition requires surfaceProgram");
            candidate.surfaceProgram = document["surfaceProgram"].dump();
            require(render::MaterialValueIR::parse(candidate.surfaceProgram).closure().has_value() == (candidate.implementation == "Slab.Surface"),
                "Definition implementation does not match Closure IR");
        } else { require(!document.contains("surfaceProgram"), "This definition does not own a Surface program"); }
        require(document.at("definitionVersion").is_number_integer(), "Invalid definition version");
        const auto version = document.at("definitionVersion").get<int64_t>();
        require(version > 0 && version <= UINT32_MAX, "Invalid definition version");
        candidate.definitionVersion = static_cast<uint32_t>(version);
        const auto values = document.value("defaults", Json::object());
        keys(values, {"parameters", "resources", "features", "featurePolicies", "valueParameters"});
        candidate.valueParameters = values.value("valueParameters", Json::object());
        validateValueParameters(candidate.valueParameters);
        std::string policyError;
        if (!overlayFeaturePolicies(values.value("featurePolicies", Json::object()), candidate.featurePolicies, policyError)) {
            throw std::runtime_error(policyError);
        }
        require(!fiber || (candidate.valueParameters.empty() && candidate.featurePolicies == MaterialFeaturePolicies{}),
            "Fiber does not support Surface Value IR or Surface feature policies");
        require(candidate.implementation != kTerrainImplementation || candidate.valueParameters.empty(),
            "Terrain layer materials do not use Value parameter slots");
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
    Json document{{"type", "Metallic.MaterialDefinition"}, {"version", definition.version},
        {"definitionVersion", definition.definitionVersion}, {"implementation", definition.implementation},
        {"defaults", {{"parameters", defaults(definition.schema.parameters)}, {"resources", defaults(definition.schema.resources)},
            {"features", defaults(definition.schema.features)}, {"featurePolicies", serializeFeaturePolicies(definition.featurePolicies)},
            {"valueParameters", definition.valueParameters}}}};
    if (!definition.surfaceProgram.empty()) { document["surfaceProgram"] = Json::parse(definition.surfaceProgram); }
    MaterialDefinition checked;
    std::string error;
    if (!deserializeMaterialDefinition(document.dump(), checked, error)) { throw std::runtime_error(error); }
    document["implementation"] = checked.implementation;
    return document.dump(4) + '\n';
}

bool deserializeMaterialInstance(std::string_view text, MaterialInstance& output, std::string& error)
{
    return attempt(error, [&] { auto candidate = instanceFromJson(parse(text)); output = std::move(candidate); });
}

std::string serializeMaterialInstance(const MaterialInstance& instance)
{
    validateStructure(instance);
    Json document{{"type", "Metallic.MaterialInstance"}, {"version", instance.version},
        {"definitionVersion", instance.definitionVersion}, {"definition", instance.definition},
        {"parent", instance.parent ? Json(*instance.parent) : Json(nullptr)}, {"parameters", instance.parameters},
        {"resources", instance.resources}, {"features", instance.features}, {"featurePolicies", instance.featurePolicies},
        {"valueParameters", instance.valueParameters}};
    if (!instance.terrainLayers.is_null()) { document["terrainLayers"] = instance.terrainLayers; }
    return document.dump(4) + '\n';
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
        candidate.valueParameters = candidate.definition.valueParameters;
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
        const bool terrain = instance.definition.implementation == kTerrainImplementation;
        require(terrain == !instance.terrainLayers.is_null(),
            "Terrain layer bindings must be supplied for an OpenPBR.TerrainLayers definition");
        if (instance.definition.implementation == "RTXCRChiang.DOTS") {
            require(instance.definition.surfaceProgram.empty() && instance.valueParameters.empty() && output.valueProgram.empty(),
                "Fiber does not support Surface Value/Closure IR");
            require(instance.resources.empty() && instance.featurePolicies == MaterialFeaturePolicies{},
                "Fiber does not support Surface resources or feature policies");
            auto candidate = output;
            candidate.terrainLayers.reset();
            candidate.rtxcrHair = true;
            candidate.alphaMode = "OPAQUE";
            candidate.unlit = false;
            candidate.featurePolicies = instance.featurePolicies;
            candidate.doubleSided = instance.features.at("doubleSided").get<bool>();
            for (const auto& field : detail::kFiberScalars) {
                const auto& value = instance.parameters.at(field.name);
                number(value, field.minimum, field.maximum);
                candidate.*field.member = value.get<float>();
            }
            require(scene::validMaterialProperties(candidate), "Resolved Fiber material is invalid");
            output = std::move(candidate);
            return;
        }
        const bool slab = instance.definition.implementation == "Slab.Surface";
        const bool value = instance.definition.implementation == "OpenPBR.Value";
        require(slab || value || terrain || instance.definition.implementation == "OpenPBRComposite.Surface", "Unsupported material implementation");
        require(!output.rtxcrHair, "OpenPBR asset cannot replace a Fiber material");
        auto candidate = output;
        candidate.terrainLayers.reset();
        if (terrain) {
            candidate.valueProgram.clear();
            candidate.valueParameters = {};
        }
        if (slab || value) {
            require(render::MaterialValueIR::parse(instance.definition.surfaceProgram).closure().has_value() == slab, "Definition Closure IR mismatch");
            candidate.valueProgram = instance.definition.surfaceProgram;
            candidate.valueParameters = {};
            validateValueParameters(instance.valueParameters);
            for (const auto& [slot, values] : instance.valueParameters.items()) {
                for (uint32_t c = 0; c < 4; ++c) { candidate.valueParameters[(slot[0] - '0') * 4 + c] = values[c].get<float>(); }
            }
        }
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
        if (slab || value) {
            const auto ir = render::MaterialValueIR::parse(candidate.valueProgram);
            require(!candidate.unlit && candidate.alphaMode != "BLEND" &&
                (candidate.transmissionFactor == 0 || (!slab && ir.outputs().contains("attenuationColor"))) &&
                candidate.diffuseTransmissionFactor == 0, "IR assets require lit opaque/MASK Surface; transmission requires explicit OpenPBR attenuationColor");
            require(!ir.outputs().contains("coverage") || candidate.alphaMode == "MASK", "Coverage assets require alphaMode mask");
        }
        if (terrain) {
            require(!candidate.unlit && candidate.alphaMode == "OPAQUE" && candidate.transmissionFactor == 0 &&
                candidate.diffuseTransmissionFactor == 0 && instance.valueParameters.empty(),
                "Terrain layers require lit opaque OpenPBR");
            const auto resolved = terrainLayers(instance.terrainLayers);
            auto bindings = std::make_shared<scene::RenderTerrainLayerBindings>();
            if (resolved.contains("weightMaps")) {
                for (const auto& map : resolved.at("weightMaps")) {
                    bindings->weightMaps.push_back(lowerTexture(map, resolver, {}));
                }
            } else {
                bindings->weightMaps.push_back(lowerTexture(resolved.at("weights"), resolver, {}));
            }
            const auto defaultLayer = resolved.at("defaultLayer").get<std::string>();
            for (const auto& source : resolved.at("layers")) {
                const auto name = source.at("name").get<std::string>();
                if (name == defaultLayer) { bindings->defaultLayerIndex = static_cast<uint32_t>(bindings->layers.size()); }
                auto& layer = bindings->layers.emplace_back(scene::RenderTerrainMaterialLayer{
                    .name = name,
                    .weightMap = source.value("weightMap", 0u),
                    .channel = static_cast<uint32_t>(kTerrainChannels.find(source.at("channel").get<std::string>())),
                    .baseColor = lowerTexture(source.at("baseColor"), resolver, render::ksRGBColorTexture),
                    .normal = lowerTexture(source.at("normal"), resolver, {}),
                    .roughness = lowerTexture(source.at("roughness"), resolver, {}),
                });
                if (const auto projection = source.find("triplanar"); projection != source.end()) {
                    layer.triplanar = scene::RenderTerrainTriplanar{
                        .worldTileSize = projection->at("worldTileSize").get<std::array<float, 2>>(),
                        .normalStrength = projection->at("normalStrength").get<float>(),
                        .baseColorTextureMix = projection->at("baseColorTextureMix").get<float>(),
                        .baseColorMeanLinear = projection->at("baseColorMeanLinear").get<std::array<float, 3>>(),
                    };
                }
            }
            candidate.terrainLayers = std::move(bindings);
        }
        const scene::RenderMaterial textureDefaults;
        for (const auto& [name, member] : detail::kTextures) {
            candidate.*member = lowerTexture(texture(instance.resources.at(name)), resolver, (textureDefaults.*member).colorMetadata);
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
        if (definition.implementation == "RTXCRChiang.DOTS") {
            require(source.rtxcrHair && scene::validMaterialProperties(source), "Invalid Fiber source material");
            require(source.valueProgram.empty() && definition.surfaceProgram.empty() &&
                source.featurePolicies == MaterialFeaturePolicies{} && source.alphaMode == "OPAQUE" && !source.unlit,
                "Fiber source has unsupported Surface behavior");
            MaterialInstance candidate;
            candidate.definition = std::move(definitionUri);
            candidate.definitionVersion = definition.definitionVersion;
            const auto baseline = defaults(definition.schema.parameters);
            for (const auto& field : detail::kFiberScalars) {
                const float value = source.*field.member;
                number(value, field.minimum, field.maximum);
                if (Json(value) != baseline.at(field.name)) { candidate.parameters[field.name] = value; }
            }
            if (Json(source.doubleSided) != defaults(definition.schema.features).at("doubleSided")) {
                candidate.features["doubleSided"] = source.doubleSided;
            }
            validateStructure(candidate);
            output = std::move(candidate);
            return;
        }
        require(!source.rtxcrHair && scene::validMaterialProperties(source), "Invalid OpenPBR source material");
        require((definition.implementation == kTerrainImplementation) == bool(source.terrainLayers),
            "Terrain bindings require an OpenPBR.TerrainLayers definition");
        require(definition.surfaceProgram.empty() ? source.valueProgram.empty() : (!source.valueProgram.empty() &&
            render::MaterialValueIR::parse(source.valueProgram).canonical() == render::MaterialValueIR::parse(definition.surfaceProgram).canonical()),
            "Custom Value Program code must belong to the material definition");
        MaterialInstance candidate;
        candidate.definition = std::move(definitionUri);
        candidate.definitionVersion = definition.definitionVersion;
        if (!definition.surfaceProgram.empty()) {
            for (uint32_t slot = 0; slot < 4; ++slot) {
                const auto key = std::to_string(slot);
                Json values = Json::array();
                for (uint32_t c = 0; c < 4; ++c) { values.push_back(source.valueParameters[slot * 4 + c]); }
                if (values != definition.valueParameters.value(key, Json::array({0, 0, 0, 0}))) { candidate.valueParameters[key] = values; }
            }
        }
        Json parameters = {{"baseColor", {source.baseColorFactor.x, source.baseColorFactor.y, source.baseColorFactor.z}},
            {"opacity", source.baseColorFactor.w}};
        for (const auto& field : detail::kScalars) { parameters[field.name] = source.*field.member; }
        for (const auto& field : detail::kColors) {
            const auto& color = source.*field.member;
            parameters[field.name] = {color.x, color.y, color.z};
        }
        Json resources = Json::object();
        const auto encodeTexture = [&](std::string_view slot, const scene::RenderTextureInfo& info) -> Json {
            if (info.textureIndex < 0) { return nullptr; }
            require(bool(encoder), "Material resource encoder is missing");
            auto result = texture({{"uri", encoder(slot, info.textureIndex)},
                {"texCoord", info.texCoord}, {"transform", info.uvTransform}});
            if (info.colorMetadata.semantic == render::TextureSemantic::Color) {
                result["colorSpace"] = render::colorSpaceName(info.colorMetadata.source);
            }
            return result;
        };
        for (const auto& [name, member] : detail::kTextures) {
            resources[name] = encodeTexture(name, source.*member);
        }
        if (source.terrainLayers) {
            const auto& bindings = *source.terrainLayers;
            const bool paired = bindings.weightMaps.size() == 2;
            Json layers = Json::array();
            for (const auto& layer : bindings.layers) {
                const auto prefix = "terrainLayers." + layer.name + ".";
                layers.push_back({
                    {"name", layer.name}, {"channel", kTerrainChannels.substr(layer.channel, 1)},
                    {"baseColor", encodeTexture(prefix + "baseColor", layer.baseColor)},
                    {"normal", encodeTexture(prefix + "normal", layer.normal)},
                    {"roughness", encodeTexture(prefix + "roughness", layer.roughness)},
                });
                if (paired) { layers.back()["weightMap"] = layer.weightMap; }
                if (const auto& projection = layer.triplanar) {
                    layers.back()["triplanar"] = {
                        {"worldTileSize", projection->worldTileSize},
                        {"normalStrength", projection->normalStrength},
                        {"baseColorTextureMix", projection->baseColorTextureMix},
                        {"baseColorMeanLinear", projection->baseColorMeanLinear},
                    };
                }
            }
            candidate.terrainLayers = {
                {"defaultLayer", bindings.layers[bindings.defaultLayerIndex].name},
                {"layers", std::move(layers)},
            };
            if (paired) {
                auto& maps = candidate.terrainLayers["weightMaps"] = Json::array();
                for (size_t i = 0; i < bindings.weightMaps.size(); ++i) {
                    maps.push_back(encodeTexture("terrainLayers.weightMaps." + std::to_string(i), bindings.weightMaps[i]));
                }
            } else {
                candidate.terrainLayers["weights"] = encodeTexture("terrainLayers.weights", bindings.weightMaps.front());
            }
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
