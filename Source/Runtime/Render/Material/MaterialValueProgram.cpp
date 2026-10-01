#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Scene/scene.h"

#include <json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>

namespace metallic::render {
namespace {
using Json = nlohmann::json;

void require(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

std::string number(const Json& value)
{
    require(value.is_number(), "Expected a numeric constant");
    const double v = value.get<double>();
    require(std::isfinite(v) && std::abs(v) <= 1e6, "Constants must be finite and within +/-1e6");
    return Json(static_cast<float>(v)).dump();
}

struct Emitter
{
    uint32_t nodes = 0;
    uint32_t temporaries = 0;
    MaterialValueManifest manifest;
    std::string statements;

    std::string expression(const Json& node, uint32_t depth = 0)
    {
        require(depth <= 24 && ++nodes <= 256, "Value expression exceeds depth/node budget (24/256)");
        if (node.is_number()) { return "float4(" + number(node) + ")"; }
        if (node.is_array()) {
            require(node.size() == 4, "Vector constants require four components");
            return "float4(" + number(node[0]) + "," + number(node[1]) + "," +
                number(node[2]) + "," + number(node[3]) + ")";
        }
        require(node.is_object() && node.contains("op") && node["op"].is_string(), "Expected an expression with op");
        const std::string op = node["op"].get<std::string>();
        if (op == "parameter") {
            require(node.size() == 2 && node.contains("index") && node["index"].is_number_integer(), "parameter requires index");
            const auto index = node["index"].get<int64_t>();
            require(index >= 0 && index < 4, "Parameter index must be 0..3");
            manifest.parameterMask |= 1u << index;
            return "instance.parameters[" + std::to_string(index) + "]";
        }
        const std::map<std::string, std::string> inputs{
            {"position", "float4(position, 0.0)"}, {"geometryNormal", "float4(geometryNormal, 0.0)"},
            {"uv", "float4(uv, 0.0, 0.0)"}, {"baseColor", "material.baseColor"},
            {"metallic", "float4(material.params.x)"}, {"roughness", "float4(material.params.y)"},
            {"emissive", "material.emissive"}};
        if (auto it = inputs.find(op); it != inputs.end()) {
            require(node.size() == 1, "Input expressions do not accept arguments");
            const std::array<std::string_view, 7> names{"position", "geometryNormal", "uv", "baseColor", "metallic", "roughness", "emissive"};
            manifest.inputMask |= 1u << std::distance(names.begin(), std::find(names.begin(), names.end(), op));
            return it->second;
        }
        const uint32_t arity = op == "mix" ? 3 :
            (op == "add" || op == "mul" || op == "dot" ? 2 :
            (op == "sin" || op == "fract" || op == "abs" || op == "saturate" ? 1 : 0));
        require(arity != 0, "Unsupported op: only read-only arithmetic is supported; textures and coverage are unavailable");
        require(node.size() == 2 && node.contains("args") && node["args"].is_array() &&
            node["args"].size() == arity, "Wrong expression arguments");
        std::vector<std::string> args;
        for (const auto& arg : node["args"]) { args.push_back(expression(arg, depth + 1)); }
        std::string result;
        if (op == "add" || op == "mul") { result = "(" + args[0] + (op == "add" ? "+" : "*") + args[1] + ")"; }
        else if (op == "dot") { result = "float4(dot(" + args[0] + "," + args[1] + "))"; }
        else if (op == "mix") { result = "lerp(" + args[0] + "," + args[1] + ",saturate(" + args[2] + "))"; }
        else { result = (op == "fract" ? "frac" : op) + "(" + args[0] + ")"; }
        const std::string name = "v" + std::to_string(temporaries++);
        // Bound intermediates as well as constants; repeated multiplication cannot overflow.
        statements += "    float4 " + name + " = clamp(" + result + ", -1e6, 1e6);\n";
        return name;
    }
};

std::string emitProgram(const Json& root, uint32_t id, MaterialValueManifest& manifest)
{
    require(root.is_object() && root.contains("version") && root["version"] == 1, "Value program version must be 1");
    Emitter emitter;
    std::string assignments;
    for (const auto& [key, value] : root.items()) {
        if (key == "version") { continue; }
        require(key == "baseColor" || key == "metallic" || key == "roughness" || key == "emissive",
            "Unsupported output: only baseColor, metallic, roughness and emissive are supported");
        const std::array<std::string_view, 4> outputs{"baseColor", "metallic", "roughness", "emissive"};
        emitter.manifest.outputMask |= 1u << std::distance(outputs.begin(), std::find(outputs.begin(), outputs.end(), key));
        const std::string expr = "(" + emitter.expression(value) + ")";
        if (key == "baseColor") { assignments += "    result.baseColor.rgb = saturate(" + expr + ".rgb);\n"; }
        if (key == "metallic") { assignments += "    result.params.x = saturate(" + expr + ".x);\n"; }
        if (key == "roughness") { assignments += "    result.params.y = saturate(" + expr + ".x);\n"; }
        if (key == "emissive") { assignments += "    result.emissive.rgb = clamp(" + expr + ".rgb, 0.0, 1e6);\n"; }
    }
    require(!assignments.empty(), "Value program must write at least one supported output");
    manifest = emitter.manifest;
    manifest.programId = id;
    manifest.expressionNodes = emitter.nodes;
    return "PathTraceMaterial materialValue" + std::to_string(id) +
        "(MaterialValueInstance instance, float3 position, float3 geometryNormal, float2 uv, PathTraceMaterial material)\n{\n"
        "    PathTraceMaterial result = material;\n" + emitter.statements + assignments + "    return result;\n}\n";
}
} // namespace

std::shared_ptr<const MaterialValueProgramSet> MaterialValueProgramSet::create(
    std::span<const scene::RenderMaterial> materials, std::string& diagnostics)
{
    diagnostics.clear();
    try {
        auto result = std::make_shared<MaterialValueProgramSet>();
        std::map<std::string, uint32_t> programs;
        std::vector<std::string> sources;
        for (const auto& material : materials) {
            require(std::all_of(material.valueParameters.begin(), material.valueParameters.end(),
                [](float v) { return std::isfinite(v) && std::abs(v) <= 1e6f; }), "Value parameters must be finite and within +/-1e6");
            if (material.valueProgram.empty()) { sources.emplace_back(); continue; }
            require(material.alphaMode == "OPAQUE" && !material.rtxcrHair && !material.unlit &&
                material.transmissionFactor == 0.0f && material.diffuseTransmissionFactor == 0.0f,
                "Value programs currently require opaque, lit, non-transmissive Surface materials");
            require(material.valueProgram.size() <= kMaxMaterialValueSourceBytes, "Value program source exceeds 16 KiB");
            // Reject excessive JSON nesting before recursive JSON parsing.
            uint32_t nesting = 0;
            bool quoted = false, escaped = false;
            for (char c : material.valueProgram) {
                if (quoted) {
                    if (escaped) { escaped = false; }
                    else if (c == '\\') { escaped = true; }
                    else if (c == '"') { quoted = false; }
                } else if (c == '"') { quoted = true; }
                else if (c == '{' || c == '[') { require(++nesting <= 64, "Value JSON nesting exceeds 64"); }
                else if (c == '}' || c == ']') { require(nesting != 0, "Unbalanced Value JSON"); --nesting; }
            }
            const std::string source = Json::parse(material.valueProgram).dump();
            programs.emplace(source, 0);
            require(programs.size() <= kMaxMaterialValuePrograms, "Scene exceeds 64 custom Value programs");
            sources.push_back(source);
        }
        result->source_ = "// Material Value ABI v1: generated, read-only, no coverage or texture sampling.\n"
            "struct MaterialValueInstance { uint programId; uint reserved0; uint reserved1; uint reserved2; float4 parameters[4]; };\n";
        for (auto& [source, id] : programs) {
            id = ++result->programCount_;
            result->manifests_.emplace_back();
            result->source_ += emitProgram(Json::parse(source), id, result->manifests_.back());
        }
        result->source_ += "PathTraceMaterial evaluateMaterialValue(uint materialIndex, float3 position, float3 geometryNormal, float2 uv, PathTraceMaterial material)\n{\n"
            "    MaterialValueInstance instance = getResource<StructuredBuffer<MaterialValueInstance>>(97)[materialIndex];\n"
            "    switch (instance.programId) {\n";
        for (const auto& [source, id] : programs) {
            const auto name = std::to_string(id);
            result->source_ += "    case " + name + ": return materialValue" + name + "(instance, position, geometryNormal, uv, material);\n";
        }
        result->source_ += "    default: return material;\n    }\n}\n";
        result->key_ = 14695981039346656037ull;
        for (unsigned char c : result->source_) { result->key_ = (result->key_ ^ c) * 1099511628211ull; }
        for (size_t i = 0; i < materials.size(); ++i) {
            MaterialValueInstance instance;
            instance.programId = sources[i].empty() ? 0 : programs.at(sources[i]);
            instance.parameters = materials[i].valueParameters;
            result->instances_.push_back(instance);
        }
        if (materials.empty()) { result->instances_.emplace_back(); }
        return result;
    } catch (const std::exception& error) {
        diagnostics = std::string("Material Value program: ") + error.what();
        return nullptr;
    }
}

bool MaterialValueProgramSet::writeInclude(const std::filesystem::path& root,
    std::filesystem::path& directory, std::string& diagnostics) const
{
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    try {
        directory = root / std::to_string(key_);
        std::filesystem::create_directories(directory);
        const auto path = directory / "MaterialValueDispatch.hlsli";
        if (std::filesystem::exists(path)) {
            std::ifstream file(path, std::ios::binary);
            const std::string existing{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
            require(existing == source_, "Generated program cache collision or corruption");
        } else {
            std::ofstream file(path, std::ios::binary);
            file.write(source_.data(), static_cast<std::streamsize>(source_.size()));
            file.close();
            require(bool(file), "Cannot write generated program include");
        }
        return true;
    } catch (const std::exception& error) {
        diagnostics = error.what();
        return false;
    }
}
} // namespace metallic::render
