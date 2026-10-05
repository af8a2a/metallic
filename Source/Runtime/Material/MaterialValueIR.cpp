#include "Runtime/Material/MaterialValueIR.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <stdexcept>

namespace metallic::render {
namespace {
using Json = nlohmann::json;
void require(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}
Json encode(const MaterialValueNode& node)
{
    return Json::array({static_cast<uint32_t>(node.op), node.operandCount, node.operands,
        node.constant, node.index, static_cast<uint32_t>(node.footprint)});
}
float scalar(const Json& value)
{
    require(value.is_number(), "Value IR constant must be numeric");
    const float result = value.get<float>();
    require(std::isfinite(result) && std::abs(result) <= 1e6f, "Value IR constant outside finite +/-1e6 range");
    return result == 0 ? 0 : result;
}
} // namespace

struct MaterialValueIRBuilder
{
    MaterialValueIR ir;
    Json definitions = Json::object();
    std::map<std::string, uint32_t> references, common;
    std::set<std::string> active;
    uint32_t visited = 0;

    uint32_t closureExpression(const Json& value, std::vector<MaterialClosureNode>& nodes, uint32_t depth = 0)
    {
        require(depth <= 24 && nodes.size() < 256, "Closure expression exceeds safety budget");
        require(value.is_object() && value.contains("op") && value["op"].is_string(), "Closure requires an operator");
        const auto op = value["op"].get<std::string>();
        MaterialClosureNode node;
        if (op == "slab") {
            require(value.contains("reflectance") && value.size() == (value.contains("opticalDepth") ? 3 : 2),
                "Slab requires reflectance and optional opticalDepth");
        } else {
            require(op == "mix" || op == "layer", "Unsupported closure operator");
            require(value.contains("a") && value.contains("b") &&
                value.size() == (op == "mix" ? 4 : 3) && (op != "mix" || value.contains("weight")),
                "Mix/Layer requires ordered a/b operands and Mix weight");
            node.op = op == "mix" ? MaterialClosureOp::Mix : MaterialClosureOp::Layer;
            node.operands = {closureExpression(value["a"], nodes, depth + 1), closureExpression(value["b"], nodes, depth + 1)};
        }
        const auto id = static_cast<uint32_t>(nodes.size());
        const auto prefix = "closure" + std::to_string(id);
        if (node.op == MaterialClosureOp::Slab) {
            ir.outputs_[prefix + "Reflectance"] = expression(value["reflectance"]);
            ir.outputs_[prefix + "OpticalDepth"] = expression(value.value("opticalDepth", Json(0)));
        } else if (node.op == MaterialClosureOp::Mix) {
            ir.outputs_[prefix + "Weight"] = expression(value["weight"]);
        }
        nodes.push_back(node);
        return id;
    }

    uint32_t insert(MaterialValueNode node)
    {
        // Fold only fully constant arithmetic. Do not use x*0 or similar
        // identities that would change finite saturation/NaN behavior.
        bool fold = node.operandCount != 0 && node.op != MaterialValueOp::TextureSample;
        for (uint32_t i = 0; i < node.operandCount; ++i) { fold &= ir.nodes_[node.operands[i]].op == MaterialValueOp::Constant; }
        if (fold) {
            const auto a = ir.nodes_[node.operands[0]].constant;
            const auto b = node.operandCount > 1 ? ir.nodes_[node.operands[1]].constant : a;
            const auto c = node.operandCount > 2 ? ir.nodes_[node.operands[2]].constant : a;
            std::array<float, 4> value{};
            float dot = 0;
            for (uint32_t i = 0; i < 4; ++i) { dot += a[i] * b[i]; }
            for (uint32_t i = 0; i < 4; ++i) {
                switch (node.op) {
                case MaterialValueOp::Add: value[i] = a[i] + b[i]; break;
                case MaterialValueOp::Multiply: value[i] = a[i] * b[i]; break;
                case MaterialValueOp::Dot: value[i] = dot; break;
                case MaterialValueOp::Lerp: value[i] = a[i] + (b[i] - a[i]) * std::clamp(c[i], 0.f, 1.f); break;
                case MaterialValueOp::Sin: value[i] = std::sin(a[i]); break;
                case MaterialValueOp::Fract: value[i] = a[i] - std::floor(a[i]); break;
                case MaterialValueOp::Abs: value[i] = std::abs(a[i]); break;
                case MaterialValueOp::Saturate: value[i] = std::clamp(a[i], 0.f, 1.f); break;
                case MaterialValueOp::Clamp: value[i] = std::min(std::max(a[i], b[i]), c[i]); break;
                case MaterialValueOp::Select: value[i] = a[0] > 0 ? b[i] : c[i]; break;
                case MaterialValueOp::Swizzle: value[i] = a[(node.index >> (2 * i)) & 3]; break;
                default: break;
                }
            }
            if (node.op == MaterialValueOp::Normalize || node.op == MaterialValueOp::NormalMap) {
                for (uint32_t i = 0; i < 3; ++i) { value[i] = node.op == MaterialValueOp::NormalMap ? a[i] * 2 - 1 : a[i]; }
                const float length = std::sqrt(value[0]*value[0] + value[1]*value[1] + value[2]*value[2]);
                if (length > 1e-10f) { for (uint32_t i = 0; i < 3; ++i) { value[i] /= length; } }
                else { value = {0, 0, 1, 0}; }
            }
            if (node.op == MaterialValueOp::UVTransform) { value = {a[0]*b[0] + a[1]*b[1] + b[2], a[0]*c[0] + a[1]*c[1] + c[2], 0, 0}; }
            node = {};
            for (uint32_t i = 0; i < 4; ++i) { node.constant[i] = std::clamp(value[i], -1e6f, 1e6f); if (node.constant[i] == 0) { node.constant[i] = 0; } }
        }
        const auto key = encode(node).dump();
        if (const auto found = common.find(key); found != common.end()) { return found->second; }
        require(ir.nodes_.size() < 256, "Value IR exceeds 256 unique nodes");
        const auto id = static_cast<uint32_t>(ir.nodes_.size());
        ir.nodes_.push_back(node); common.emplace(key, id);
        return id;
    }

    uint32_t expression(const Json& value, uint32_t depth = 0)
    {
        require(depth <= 24 && ++visited <= 1024, "Value IR exceeds depth/visit budget (24/1024)");
        MaterialValueNode node;
        if (value.is_number()) { node.constant.fill(scalar(value)); return insert(node); }
        if (value.is_array()) {
            require(value.size() == 4, "Value IR vectors require four components");
            for (uint32_t i = 0; i < 4; ++i) { node.constant[i] = scalar(value[i]); }
            return insert(node);
        }
        require(value.is_object(), "Value IR expects an expression object");
        if (value.contains("ref")) {
            require(value.size() == 1 && value["ref"].is_string(), "Invalid Value IR reference");
            const auto name = value["ref"].get<std::string>();
            require(definitions.contains(name), "Unknown Value IR reference");
            require(!active.contains(name), "Cyclic Value IR reference");
            if (const auto found = references.find(name); found != references.end()) { return found->second; }
            active.insert(name);
            const auto id = expression(definitions[name], depth + 1);
            active.erase(name); references.emplace(name, id);
            return id;
        }
        require(value.contains("op") && value["op"].is_string(), "Value IR expression requires op");
        const auto op = value["op"].get<std::string>();
        if (op == "parameter") {
            require(value.size() == 2 && value.contains("index") && value["index"].is_number_integer(), "Invalid parameter index");
            const auto index = value["index"].get<int64_t>();
            require(index >= 0 && index < 4, "Parameter slot must be 0..3");
            node.op = MaterialValueOp::Parameter; node.index = static_cast<uint32_t>(index);
            return insert(node);
        }
        static const std::map<std::string, MaterialValueOp> inputs{
            {"uv", MaterialValueOp::UV}, {"alpha", MaterialValueOp::Alpha}, {"position", MaterialValueOp::Position},
            {"geometryNormal", MaterialValueOp::GeometryNormal}, {"baseColor", MaterialValueOp::BaseColor},
            {"metallic", MaterialValueOp::Metallic}, {"roughness", MaterialValueOp::Roughness}, {"emissive", MaterialValueOp::Emissive}};
        if (const auto found = inputs.find(op); found != inputs.end()) {
            require(value.size() == 1, "Value IR input takes no arguments"); node.op = found->second; return insert(node);
        }
        static const std::map<std::string, std::pair<MaterialValueOp, uint32_t>> operations{
            {"add", {MaterialValueOp::Add, 2}}, {"mul", {MaterialValueOp::Multiply, 2}}, {"dot", {MaterialValueOp::Dot, 2}},
            {"mix", {MaterialValueOp::Lerp, 3}}, {"sin", {MaterialValueOp::Sin, 1}}, {"fract", {MaterialValueOp::Fract, 1}},
            {"abs", {MaterialValueOp::Abs, 1}}, {"saturate", {MaterialValueOp::Saturate, 1}}, {"clamp", {MaterialValueOp::Clamp, 3}},
            {"normalize", {MaterialValueOp::Normalize, 1}}, {"normalMap", {MaterialValueOp::NormalMap, 1}},
            {"uvTransform", {MaterialValueOp::UVTransform, 3}}, {"swizzle", {MaterialValueOp::Swizzle, 1}},
            {"select", {MaterialValueOp::Select, 3}}, {"textureSample", {MaterialValueOp::TextureSample, 2}}};
        const auto found = operations.find(op);
        require(found != operations.end(), "Unsupported Value IR operation");
        node.op = found->second.first; node.operandCount = found->second.second;
        uint32_t fields = 2;
        if (node.op == MaterialValueOp::Swizzle) {
            require(value.contains("components") && value["components"].is_string(), "Swizzle requires components");
            const auto components = value["components"].get<std::string>();
            require(components.size() == 4, "Swizzle requires four xyzw components");
            for (uint32_t i = 0; i < 4; ++i) {
                const auto index = std::string_view("xyzw").find(components[i]);
                require(index != std::string_view::npos, "Invalid swizzle component"); node.index |= uint32_t(index) << (i * 2);
            }
            fields = 3;
        }
        if (node.op == MaterialValueOp::TextureSample) {
            require(value.contains("texture") && value["texture"].is_string() && value.contains("footprint") && value["footprint"].is_string(),
                "TextureSample requires a texture slot and explicit footprint");
            const std::array<std::string_view, 6> slots{"baseColor", "metallicRoughness", "normal", "occlusion", "emissive", "specular"};
            const auto slot = std::find(slots.begin(), slots.end(), value["texture"].get<std::string>());
            require(slot != slots.end(), "Unknown Value IR texture slot"); node.index = uint32_t(slot - slots.begin());
            const auto policy = value["footprint"].get<std::string>();
            require(policy == "ExplicitLOD" || policy == "SampleGrad" || policy == "RayCone", "Implicit texture footprint is forbidden");
            node.footprint = policy == "SampleGrad" ? MaterialTextureFootprint::SampleGrad : policy == "RayCone" ? MaterialTextureFootprint::RayCone : MaterialTextureFootprint::ExplicitLOD;
            node.operandCount = policy == "SampleGrad" ? 3 : 2; fields = 4;
        }
        require(value.size() == fields && value.contains("args") && value["args"].is_array() && value["args"].size() == node.operandCount,
            "Wrong Value IR operation arguments");
        node.operands[0] = expression(value["args"][0], depth + 1);
        if (node.op == MaterialValueOp::Select && ir.nodes_[node.operands[0]].op == MaterialValueOp::Constant) {
            // Branch DCE precedes resource validation: a statically unselected
            // texture need not exist on this target or declare a live footprint.
            return expression(value["args"][ir.nodes_[node.operands[0]].constant[0] > 0 ? 1 : 2], depth + 1);
        }
        for (uint32_t i = 1; i < node.operandCount; ++i) { node.operands[i] = expression(value["args"][i], depth + 1); }
        return insert(node);
    }
};

MaterialValueIR MaterialValueIR::parse(std::string_view source)
{
    require(source.size() <= 16384, "Value source exceeds 16 KiB");
    uint32_t nesting = 0; bool quoted = false, escaped = false;
    for (char c : source) {
        if (quoted) { if (escaped) { escaped = false; } else if (c == '\\') { escaped = true; } else if (c == '"') { quoted = false; } }
        else if (c == '"') { quoted = true; }
        else if (c == '{' || c == '[') { require(++nesting <= 64, "Value JSON nesting exceeds 64"); }
        else if (c == '}' || c == ']') { require(nesting != 0, "Unbalanced Value JSON"); --nesting; }
    }
    return lower(Json::parse(source));
}

MaterialValueIR MaterialValueIR::lower(const Json& root)
{
    require(root.is_object() && root.contains("version") && root["version"].is_number_integer() &&
        (root["version"] == 1 || root["version"] == 2 || root["version"] == 3), "Value program version must be 1, 2 or 3");
    MaterialValueIRBuilder builder;
    Json outputs = root;
    const bool closures = root["version"] == 3;
    if (root["version"] == 2 || closures) {
        require(root.size() == (closures ? 4 : 3) && root.contains("nodes") && root["nodes"].is_object() && root["nodes"].size() <= 256 &&
            root.contains("outputs") && root["outputs"].is_object(), "Value graph requires nodes and outputs objects");
        builder.definitions = root["nodes"]; outputs = root["outputs"];
        if (closures) {
            require(root.contains("closure"), "v3 material requires a closure");
            std::vector<MaterialClosureNode> nodes;
            const auto id = builder.closureExpression(root["closure"], nodes);
            builder.ir.closure_ = MaterialClosureIR::create(nodes, id);
            (void)lowerMaterialClosure(*builder.ir.closure_);
        }
    } else { outputs.erase("version"); }
    require(closures || !outputs.empty(), "Value program has no outputs");
    for (const auto& [name, value] : outputs.items()) {
        const bool openPBR = name == "coatWeight" || name == "coatRoughness" || name == "coatIOR" ||
            name == "fuzzWeight" || name == "fuzzColor" || name == "fuzzRoughness" ||
            name == "specularAnisotropy" || name == "anisotropyTangent" || name == "attenuationColor";
        require(name == "baseColor" || name == "metallic" || name == "roughness" || name == "emissive" || name == "coverage" || openPBR, "Unsupported Value IR output");
        require(!closures || name == "emissive" || name == "coverage", "Slab material outputs support only emissive and coverage");
        builder.ir.outputs_[name] = builder.expression(value);
    }
    builder.ir.finalize();
    return std::move(builder.ir);
}

void MaterialValueIR::finalize()
{
    // Compact live nodes in root/dependency order. Definition names, unreachable
    // nodes and source insertion order never enter executable identity.
    const auto source = std::move(nodes_);
    nodes_.clear(); std::map<uint32_t, uint32_t> remap;
    std::function<uint32_t(uint32_t)> visit = [&](uint32_t id) {
        if (const auto found = remap.find(id); found != remap.end()) { return found->second; }
        auto node = source.at(id);
        for (uint32_t i = 0; i < node.operandCount; ++i) { node.operands[i] = visit(node.operands[i]); }
        const auto target = uint32_t(nodes_.size()); nodes_.push_back(node); remap.emplace(id, target); return target;
    };
    for (auto& [name, id] : outputs_) { id = visit(id); }
    usage_ = {};
    Json encoded = Json::array();
    for (const auto& node : nodes_) {
        encoded.push_back(encode(node));
        if (node.op == MaterialValueOp::Parameter) { usage_.parameterMask |= 1u << node.index; usage_.features |= ValueParameters; }
        const std::array<MaterialValueOp, 7> inputs{MaterialValueOp::Position, MaterialValueOp::GeometryNormal, MaterialValueOp::UV,
            MaterialValueOp::BaseColor, MaterialValueOp::Metallic, MaterialValueOp::Roughness, MaterialValueOp::Emissive};
        const auto input = std::find(inputs.begin(), inputs.end(), node.op);
        if (input != inputs.end()) { usage_.inputMask |= 1u << (input - inputs.begin()); }
        if (node.op == MaterialValueOp::Position || node.op == MaterialValueOp::GeometryNormal || node.op == MaterialValueOp::UV) { usage_.features |= ValueGeometry; }
        if (node.op == MaterialValueOp::NormalMap) { usage_.features |= ValueNormalMapping; }
        if (node.op == MaterialValueOp::TextureSample || node.op == MaterialValueOp::Alpha) {
            usage_.features |= ValueTextures; usage_.textureMask |= 1u << node.index;
            usage_.footprintMask |= 1u << static_cast<uint32_t>(node.footprint);
            if (node.footprint == MaterialTextureFootprint::RayCone) { usage_.features |= ValueRayCone; }
            if (node.footprint == MaterialTextureFootprint::SampleGrad) { usage_.features |= ValueGradients; }
        }
    }
    if (outputs_.contains("coverage")) { usage_.features |= ValueCoverage; }
    Json canonical{{"ir", 1}, {"nodes", encoded}, {"outputs", outputs_}};
    if (closure_) { canonical["closure"] = closure_->canonical(); }
    canonical_ = canonical.dump();
    hash_ = 14695981039346656037ull;
    for (const unsigned char c : canonical_) { hash_ = (hash_ ^ c) * 1099511628211ull; }
}

MaterialValueIR MaterialValueIR::slice(bool coverage) const
{
    auto result = *this;
    if (coverage) { result.closure_.reset(); }
    std::erase_if(result.outputs_, [&](const auto& item) { return (item.first == "coverage") != coverage; });
    result.finalize();
    return result;
}
} // namespace metallic::render
