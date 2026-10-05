#include "Runtime/Material/MaterialGraph.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

namespace metallic::material {
namespace {
using Json = nlohmann::json;
void require(bool valid, const std::string& message)
{
    if (!valid) { throw std::runtime_error(message); }
}
const MaterialGraphNodeDesc& descriptor(std::string_view kind)
{
    for (const auto& desc : materialGraphNodes()) { if (desc.kind == kind) { return desc; } }
    throw std::runtime_error("Unknown MaterialGraph node: " + std::string(kind));
}
bool finish(const Json& irSource, const Json& defaults, CompiledMaterialFrontend& output, std::string& error)
{
    try {
        auto source = irSource;
        auto ir = render::MaterialValueIR::parse(source.dump());
        // Opaque output is the default. Do not publish an unnecessary Coverage
        // program: the runtime deliberately reserves those for MASK surfaces.
        if (const auto coverage = ir.outputs().find("coverage"); coverage != ir.outputs().end()) {
            const auto& node = ir.nodes()[coverage->second];
            if (node.op == render::MaterialValueOp::Constant && node.constant[0] == 1.0f) {
                source["outputs"].erase("coverage");
                ir = render::MaterialValueIR::parse(source.dump());
            }
        }
        CompiledMaterialFrontend candidate;
        candidate.definition = defaultOpenPBRDefinition();
        candidate.definition.implementation = ir.closure() ? "Slab.Surface" : "OpenPBR.Value";
        candidate.definition.surfaceProgram = source.dump();
        candidate.definition.valueParameters = defaults;
        const auto coverageIR = ir.slice(true), surfaceIR = ir.slice(false);
        for (const auto& node : coverageIR.nodes()) {
            require(node.op <= render::MaterialValueOp::Select, "Coverage supports arithmetic/UV/parameters/BaseAlpha; general texture and geometry operations require Surface");
        }
        for (const auto& node : surfaceIR.nodes()) {
            require(node.op != render::MaterialValueOp::Alpha, "BaseAlpha is a Coverage-only input");
        }
        if (ir.outputs().contains("coverage")) {
            for (auto& feature : candidate.definition.schema.features) {
                if (feature.name == "alphaMode") { feature.defaultValue = "mask"; }
            }
        }
        // Reuse asset schema/range validation before publishing anything.
        MaterialDefinition checked;
        if (!deserializeMaterialDefinition(serializeMaterialDefinition(candidate.definition), checked, error)) { throw std::runtime_error(error); }
        candidate.signature = ir.hash();
        const auto& usage = ir.usage();
        candidate.reflection = {{"signature", ir.hash()}, {"valueNodes", ir.nodes().size()},
            {"parameterMask", usage.parameterMask}, {"textureMask", usage.textureMask},
            {"footprintMask", usage.footprintMask}, {"features", usage.features},
            {"parameterBytes", 64}, {"sideEffects", false}, {"sourceColorSpace", "lin_rec709"}};
        Json resources = Json::array();
        constexpr const char* slots[] = {"baseColor", "metallicRoughness", "normal", "occlusion", "emissive", "specular"};
        for (uint32_t i = 0; i < 6; ++i) {
            if ((usage.textureMask & (1u << i)) != 0) { resources.push_back(slots[i]); }
        }
        candidate.reflection["resources"] = std::move(resources);
        if (ir.closure()) {
            const auto lowered = render::lowerMaterialClosure(*ir.closure());
            candidate.reflection["closureFamily"] = static_cast<uint32_t>(lowered.family);
            candidate.reflection["closureRecords"] = lowered.complexity.closureRecordCount;
            candidate.reflection["payloadBytes"] = lowered.complexity.payloadBytes;
        } else { candidate.reflection["closureFamily"] = "OpenPBRComposite"; }
        output = std::move(candidate);
        error.clear();
        return true;
    } catch (const std::exception& exception) { error = exception.what(); return false; }
}
} // namespace

const std::vector<MaterialGraphNodeDesc>& materialGraphNodes()
{
    static const std::vector<MaterialGraphNodeDesc> nodes{
        {"Constant", {}}, {"Parameter", {}}, {"UV", {}}, {"Position", {}}, {"GeometryNormal", {}}, {"BaseAlpha", {}},
        {"Texture", {"uv", "lod"}}, {"Math", {"a", "b", "t"}}, {"Swizzle", {"value"}}, {"NormalMap", {"value"}},
        {"OpenPBR", {"baseColor", "metallic", "roughness", "normalTS", "coatWeight", "coatRoughness"}, true},
        {"Slab", {"reflectance", "opticalDepth"}, true}, {"Mix", {"a", "b", "weight"}, true},
        {"Layer", {"a", "b"}, true}, {"Output", {"surface", "emissive", "coverage"}, true}};
    return nodes;
}

Json makeMaterialGraphNode(std::string_view kind, uint32_t id)
{
    const auto& desc = descriptor(kind);
    Json node{{"id", id}, {"kind", kind}, {"position", {0, 0}}, {"inputs", Json::object()}};
    for (const auto& input : desc.inputs) { node["inputs"][input] = 0.0; }
    if (kind == "Constant") { node["value"] = Json::array({.18, .18, .18, 1.}); }
    if (kind == "Parameter") { node["slot"] = 0; node["default"] = Json::array({.18, .18, .18, 1.}); }
    if (kind == "Texture") { node["texture"] = "baseColor"; node["footprint"] = "RayCone"; node["inputs"]["uv"] = Json{{"op", "uv"}}; }
    if (kind == "Math") { node["operation"] = "multiply"; node["inputs"]["b"] = 1.; }
    if (kind == "NormalMap") { node["inputs"]["value"] = Json::array({.5,.5,1.,0.}); }
    if (kind == "Swizzle") { node["components"] = "xxxx"; }
    if (kind == "OpenPBR") {
        node["inputs"]["baseColor"] = Json::array({.18,.18,.18,1.});
        node["inputs"]["roughness"] = .35;
        node["inputs"]["normalTS"] = Json::array({0.,0.,1.,0.});
        node["inputs"]["coatRoughness"] = .03;
    }
    if (kind == "Slab") { node["inputs"]["reflectance"] = Json::array({.5,.5,.5,0.}); }
    if (kind == "Output") { node["inputs"]["coverage"] = 1.; }
    return node;
}

Json defaultMaterialGraph(bool slab)
{
    auto parameter = makeMaterialGraphNode("Parameter", 1);
    auto surface = makeMaterialGraphNode(slab ? "Slab" : "OpenPBR", 2);
    auto output = makeMaterialGraphNode("Output", 3);
    parameter["position"] = {20, 70}; surface["position"] = {280, 60}; output["position"] = {560, 80};
    surface["inputs"][slab ? "reflectance" : "baseColor"] = {{"node", 1}};
    output["inputs"]["surface"] = {{"node", 2}};
    return {{"type", "Metallic.MaterialGraph"}, {"version", 1}, {"nodes", Json::array({parameter, surface, output})}, {"output", 3}};
}

bool compileMaterialGraph(const Json& graph, CompiledMaterialFrontend& output, std::string& error)
{
    try {
        require(graph.is_object() && graph.value("type", "") == "Metallic.MaterialGraph" && graph.at("version") == 1,
            "Unsupported MaterialGraph format");
        require(graph.at("nodes").is_array() && graph["nodes"].size() <= 128, "MaterialGraph allows at most 128 nodes");
        std::map<uint32_t, Json> nodes;
        Json defaults = Json::object(), definitions = Json::object();
        for (const auto& node : graph["nodes"]) {
            require(node.at("id").is_number_integer(), "Graph node ID must be an integer");
            const auto id = node["id"].get<int64_t>();
            require(id > 0 && id <= 100000 && !nodes.contains(static_cast<uint32_t>(id)), "Duplicate or invalid graph node ID");
            const auto& desc = descriptor(node.at("kind").get<std::string>());
            require(node.at("inputs").is_object(), "Node inputs must be an object");
            for (const auto& [key, value] : node["inputs"].items()) {
                require(std::find(desc.inputs.begin(), desc.inputs.end(), key) != desc.inputs.end(),
                    "Node " + std::to_string(id) + ": unknown input " + key);
            }
            nodes.emplace(static_cast<uint32_t>(id), node);
        }
        std::set<uint32_t> active, done;
        std::map<uint32_t, Json> surfaces;
        std::function<Json(uint32_t)> visit;
        const auto input = [&](const Json& value, bool surface) -> Json {
            if (value.is_object() && value.contains("node")) {
                require(value.size() == 1 && value["node"].is_number_integer(), "Invalid graph link");
                const auto id = value["node"].get<int64_t>();
                require(id > 0 && id <= 100000 && nodes.contains(static_cast<uint32_t>(id)), "Link references missing node");
                const auto& source = descriptor(nodes.at(static_cast<uint32_t>(id)).at("kind").get<std::string>());
                require(source.surface == surface && source.kind != "Output", "Value / Surface pin type mismatch");
                return visit(static_cast<uint32_t>(id));
            }
            require(!surface, "Surface input must be linked");
            // Literals and read-only IR input expressions share the established validator.
            return value;
        };
        visit = [&](uint32_t id) -> Json {
            require(!active.contains(id), "Graph cycle at node " + std::to_string(id));
            const auto& node = nodes.at(id);
            const auto kind = node.at("kind").get<std::string>();
            if (done.contains(id)) { return descriptor(kind).surface ? surfaces.at(id) : Json{{"ref", "n" + std::to_string(id)}}; }
            active.insert(id);
            const auto get = [&](const char* name, bool surface = false) { return input(node.at("inputs").at(name), surface); };
            Json value;
            if (kind == "Constant") { value = node.at("value"); }
            else if (kind == "Parameter") {
                require(node.at("slot").is_number_integer(), "Parameter slot must be an integer");
                const auto slot = node["slot"].get<int64_t>();
                require(slot >= 0 && slot < 4, "Parameter slot must be 0..3");
                const auto key = std::to_string(slot);
                require(!defaults.contains(key) || defaults[key] == node.at("default"), "Conflicting defaults for shared parameter slot");
                defaults[key] = node.at("default");
                value = {{"op", "parameter"}, {"index", slot}};
            } else if (kind == "UV" || kind == "Position" || kind == "GeometryNormal" || kind == "BaseAlpha") {
                value = {{"op", kind == "UV" ? "uv" : kind == "Position" ? "position" : kind == "BaseAlpha" ? "alpha" : "geometryNormal"}};
            } else if (kind == "Texture") {
                const auto footprint = node.value("footprint", "RayCone");
                require(footprint == "RayCone" || footprint == "ExplicitLOD", "Graph texture requires RayCone or ExplicitLOD");
                value = {{"op","textureSampleLinear"},{"texture",node.at("texture")},{"footprint",footprint},{"args",Json::array({get("uv"),get("lod")})}};
            } else if (kind == "NormalMap") { value = {{"op","normalMap"},{"args",Json::array({get("value")})}}; }
            else if (kind == "Swizzle") { value = {{"op","swizzle"},{"components",node.at("components")},{"args",Json::array({get("value")})}}; }
            else if (kind == "Math") {
                const auto op = node.at("operation").get<std::string>();
                static const std::map<std::string, int> arities{{"add",2},{"multiply",2},{"dot",2},{"lerp",3},{"clamp",3},{"select",3},
                    {"sin",1},{"fract",1},{"abs",1},{"saturate",1},{"normalize",1}};
                require(arities.contains(op), "Unsupported Math operation " + op);
                Json args = Json::array({get("a")});
                if (arities.at(op) > 1) { args.push_back(get("b")); }
                if (arities.at(op) > 2) { args.push_back(get("t")); }
                value = {{"op",op == "multiply" ? "mul" : op == "lerp" ? "mix" : op},{"args",args}};
            } else if (kind == "OpenPBR") {
                value = {{"openPBR",Json::object()}};
                for (const auto& key : descriptor(kind).inputs) { value["openPBR"][key] = get(key.c_str()); }
            } else if (kind == "Slab") {
                value = {{"op","slab"},{"reflectance",get("reflectance")},{"opticalDepth",get("opticalDepth")}};
            } else if (kind == "Mix" || kind == "Layer") {
                const auto a = get("a", true), b = get("b", true);
                require(!a.contains("openPBR") && !b.contains("openPBR"), "Mix/Layer accepts Slab closures, not OpenPBR composites");
                value = {{"op",kind == "Mix" ? "mix" : "layer"},{"a",a},{"b",b}};
                if (kind == "Mix") { value["weight"] = get("weight"); }
            } else {
                value = get("surface", true);
                value["outputs"] = {{"emissive",get("emissive")},{"coverage",get("coverage")}};
            }
            active.erase(id); done.insert(id);
            if (descriptor(kind).surface) { surfaces[id] = value; return value; }
            definitions["n" + std::to_string(id)] = value;
            return Json{{"ref", "n" + std::to_string(id)}};
        };
        require(graph.at("output").is_number_integer(), "Output node ID must be an integer");
        const auto rootValue = graph["output"].get<int64_t>();
        require(rootValue > 0 && rootValue <= 100000, "Invalid output node ID");
        const auto root = static_cast<uint32_t>(rootValue);
        require(nodes.contains(root) && nodes.at(root).at("kind") == "Output", "Graph requires an Output root");
        auto surface = visit(root);
        Json ir{{"version", surface.contains("openPBR") ? 2 : 3}, {"nodes",definitions}, {"outputs",surface.at("outputs")}};
        surface.erase("outputs");
        if (surface.contains("openPBR")) {
            const std::map<std::string, std::string> absolute{{"baseColor","surfaceBaseColor"},{"metallic","surfaceMetallic"},{"roughness","surfaceRoughness"}};
            for (const auto& [key, value] : surface["openPBR"].items()) { ir["outputs"][absolute.contains(key) ? absolute.at(key) : key] = value; }
            ir["outputs"]["surfaceEmission"] = ir["outputs"]["emissive"];
            ir["outputs"].erase("emissive");
        } else { ir["closure"] = surface; }
        if (!finish(ir, defaults, output, error)) { throw std::runtime_error(error); }
        return true;
    } catch (const std::exception& exception) { error = "MaterialGraph: " + std::string(exception.what()); return false; }
}

// Implemented by the SDK expression parser; both frontends finish in the same IR validator.
bool compileMaterialFrontendIR(const Json& ir, const Json& defaults, CompiledMaterialFrontend& output, std::string& error)
{
    return finish(ir, defaults, output, error);
}
} // namespace metallic::material
