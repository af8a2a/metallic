#include "Runtime/Material/MaterialGraph.h"
#include <cctype>
#include <cmath>
#include <map>
#include <stdexcept>
#include <cstdlib>

namespace metallic::material {
bool compileMaterialFrontendIR(const nlohmann::json&, const nlohmann::json&, CompiledMaterialFrontend&, std::string&);
namespace {
using Json = nlohmann::json;
struct Expression { Json value; bool material = false; };

// This grammar is deliberately closed. Source text never enters generated shader includes.
// Slang syntax outside the read-only SDK expression profile is diagnosed, not sanitized.
class Parser
{
public:
    explicit Parser(std::string_view source) : source_(source) { next(); }
    Json parse()
    {
        if (token_ == "import") { next(); take("MaterialAuthoringSDK"); take(";"); }
        take("Material"); take("evaluate"); take("("); take(")"); take("{");
        uint32_t statements = 0;
        while (token_ != "return") {
            check(++statements <= 128, "Too many SDK statements");
            const auto type = token_;
            check(type == "let" || type == "float4" || type == "Material", "Expected let/float4/Material declaration or return");
            next(); const auto name = token_;
            check(identifier(name) && !variables_.contains(name), "Invalid or duplicate local name");
            next(); take("="); auto value = expression(); take(";");
            check(type == "let" || (type == "Material") == value.material, "Local type mismatch");
            if (!value.material) {
                definitions_[name] = value.value;
                value.value = Json{{"ref", name}};
            }
            variables_[name] = std::move(value);
        }
        next(); auto result = expression(); take(";"); take("}"); take("");
        check(result.material, "evaluate must return Material");
        auto surface = result.value;
        Json outputs = surface.value("outputs", Json{{"emissive",0.0},{"coverage",1.0}});
        surface.erase("outputs");
        Json ir{{"version",surface.contains("openPBR") ? 2 : 3},{"nodes",definitions_},{"outputs",outputs}};
        if (surface.contains("openPBR")) {
            const std::map<std::string,std::string> names{{"baseColor","surfaceBaseColor"},{"metallic","surfaceMetallic"},{"roughness","surfaceRoughness"}};
            for (const auto& [key,value] : surface["openPBR"].items()) { ir["outputs"][names.contains(key) ? names.at(key) : key] = value; }
            ir["outputs"]["surfaceEmission"] = ir["outputs"]["emissive"]; ir["outputs"].erase("emissive");
        } else { ir["closure"] = surface; }
        return ir;
    }
private:
    std::string_view source_;
    size_t offset_ = 0;
    std::string token_;
    uint32_t depth_ = 0;
    std::map<std::string,Expression> variables_;
    Json definitions_ = Json::object();
    void check(bool valid, const std::string& message) const
    {
        if (!valid) { throw std::runtime_error("Slang SDK byte " + std::to_string(offset_) + ": " + message + " (near '" + token_ + "')"); }
    }
    static bool identifier(const std::string& text)
    {
        return !text.empty() && (std::isalpha(static_cast<unsigned char>(text[0])) || text[0] == '_');
    }
    void next()
    {
        for (;;) {
            while (offset_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[offset_]))) { ++offset_; }
            if (source_.substr(offset_, 2) == "//") {
                while (offset_ < source_.size() && source_[offset_] != '\n') { ++offset_; }
            } else if (source_.substr(offset_, 2) == "/*") {
                const auto end = source_.find("*/", offset_ + 2);
                check(end != std::string_view::npos, "Unterminated comment"); offset_ = end + 2;
            } else { break; }
        }
        token_.clear();
        if (offset_ == source_.size()) { return; }
        const auto begin = offset_++;
        const auto c = static_cast<unsigned char>(source_[begin]);
        if (std::isalpha(c) || c == '_') {
            while (offset_ < source_.size() && (std::isalnum(static_cast<unsigned char>(source_[offset_])) || source_[offset_] == '_')) { ++offset_; }
        } else if (std::isdigit(c) || c == '.') {
            while (offset_ < source_.size()) {
                const char next = source_[offset_];
                if (std::isdigit(static_cast<unsigned char>(next)) || next == '.' || next == 'e' || next == 'E' ||
                    ((next == '+' || next == '-') && (source_[offset_-1] == 'e' || source_[offset_-1] == 'E'))) { ++offset_; }
                else { break; }
            }
        }
        token_ = std::string(source_.substr(begin, offset_-begin));
    }
    void take(const char* token)
    {
        check(token_ == token, "Expected '" + std::string(token) + "'"); next();
    }
    Expression expression(int precedence = 0)
    {
        check(++depth_ <= 32, "Expression nesting exceeds 32");
        Expression left;
        if (token_ == "-") {
            next(); left = expression(3); check(!left.material,"Cannot negate a Material");
            if (left.value.is_number()) { left.value = -left.value.get<double>(); }
            else { left.value = Json{{"op","mul"},{"args",Json::array({left.value,-1.0})}}; }
        } else if (token_ == "(") { next(); left = expression(); take(")"); }
        else if (identifier(token_)) {
            const auto name = token_; next();
            if (token_ != "(") {
                check(variables_.contains(name), "Unknown local " + name); left = variables_.at(name);
            } else {
                next(); std::vector<Expression> args;
                if (token_ != ")") {
                    do {
                        if (!args.empty()) { take(","); }
                        check(args.size() < 8, "Too many function arguments"); args.push_back(expression());
                    } while (token_ == ",");
                }
                take(")"); left = call(name,args);
            }
        } else {
            char* end = nullptr;
            const double number = std::strtod(token_.c_str(), &end);
            check(!token_.empty() && end == token_.c_str()+token_.size() && std::isfinite(number), "Expected finite number or SDK function");
            left.value = number; next();
        }
        while ((token_ == "+" || token_ == "-" ? 1 : token_ == "*" ? 2 : 0) > precedence) {
            const auto op = token_; next();
            auto right = expression(op == "*" ? 2 : 1);
            check(!left.material && !right.material, "Arithmetic requires Value pins");
            if (op == "-") { right.value = Json{{"op","mul"},{"args",Json::array({right.value,-1.0})}}; }
            left.value = Json{{"op",op == "*" ? "mul" : "add"},{"args",Json::array({left.value,right.value})}};
        }
        --depth_; return left;
    }
    Expression call(const std::string& name, const std::vector<Expression>& args)
    {
        const auto count = [&](size_t size) { check(args.size() == size, name + " requires " + std::to_string(size) + " arguments"); };
        const auto value = [&](size_t index) -> Json { check(!args.at(index).material,"Expected Value argument"); return args[index].value; };
        const auto material = [&](size_t index) -> Json { check(args.at(index).material,"Expected Material argument"); return args[index].value; };
        if (name == "parameter") {
            count(1); const auto index = value(0); check(index.is_number() && index == std::floor(index.get<double>()),"Parameter index must be literal");
            check(index.get<double>() >= 0 && index.get<double>() < 4,"Parameter index must be 0..3");
            return {Json{{"op","parameter"},{"index",index.get<int>()}}};
        }
        if (name == "float4") {
            check(args.size() == 1 || args.size() == 4,"float4 requires one or four literal values");
            if (args.size() == 1) { return {value(0)}; }
            Json result = Json::array();
            for (size_t i=0; i<4; ++i) { auto v=value(i); check(v.is_number(),"float4 components must be literals"); result.push_back(v); }
            return {result};
        }
        if (name == "uv" || name == "position" || name == "geometryNormal" || name == "alpha") { count(0); return {Json{{"op",name}}}; }
        if (name == "x" || name == "y" || name == "z" || name == "w") {
            count(1); return {Json{{"op","swizzle"},{"components",std::string(4,name[0])},{"args",Json::array({value(0)})}}};
        }
        const std::map<std::string,std::string> textures{{"sampleBaseColor","baseColor"},{"sampleMetallicRoughness","metallicRoughness"},
            {"sampleNormal","normal"},{"sampleOcclusion","occlusion"},{"sampleEmissive","emissive"},{"sampleSpecular","specular"}};
        if (textures.contains(name)) {
            count(2); return {Json{{"op","textureSampleLinear"},{"texture",textures.at(name)},{"footprint","RayCone"},{"args",Json::array({value(0),value(1)})}}};
        }
        if (name == "openPBR") {
            count(3);
            return {Json{{"openPBR",{{"baseColor",value(0)},{"metallic",value(1)},{"roughness",value(2)},
                {"normalTS",Json::array({0.,0.,1.,0.})},{"coatWeight",0.0},{"coatRoughness",.03}}}},true};
        }
        if (name == "slab") { count(2); return {Json{{"op","slab"},{"reflectance",value(0)},{"opticalDepth",value(1)}},true}; }
        if (name == "mix" || name == "layer") {
            count(name == "mix" ? 3 : 2);
            const auto a=material(0), b=material(1);
            check(!a.contains("openPBR") && !b.contains("openPBR") && !a.contains("outputs") && !b.contains("outputs"),"Mix/Layer requires direct Slab closures");
            Json result{{"op",name},{"a",a},{"b",b}};
            if (name == "mix") { result["weight"]=value(2); }
            return {result,true};
        }
        if (name == "withNormal" || name == "withCoat" || name == "surface") {
            count(name == "withNormal" ? 2 : 3); auto result=material(0);
            if (name == "surface") { result["outputs"]={{"emissive",value(1)},{"coverage",value(2)}}; }
            else {
                check(result.contains("openPBR"),"Normal/coat modifiers require OpenPBR");
                if (name == "withNormal") { result["openPBR"]["normalTS"]=value(1); }
                else { result["openPBR"]["coatWeight"]=value(1); result["openPBR"]["coatRoughness"]=value(2); }
            }
            return {result,true};
        }
        const std::map<std::string,size_t> arities{{"add",2},{"multiply",2},{"dot",2},{"lerp",3},{"clamp",3},{"select",3},
            {"sin",1},{"fract",1},{"abs",1},{"saturate",1},{"normalize",1},{"normalMap",1}};
        check(arities.contains(name),"Unsupported SDK function/capability: " + name);
        count(arities.at(name)); Json values=Json::array();
        for (size_t i=0; i<args.size(); ++i) { values.push_back(value(i)); }
        return {Json{{"op",name == "multiply" ? "mul" : name == "lerp" ? "mix" : name},{"args",values}}};
    }
};
} // namespace

bool compileSlangMaterial(std::string_view source, const Json& defaults, CompiledMaterialFrontend& output, std::string& error)
{
    try {
        if (source.size() > 16384) { throw std::runtime_error("Slang SDK source exceeds 16 KiB"); }
        return compileMaterialFrontendIR(Parser(source).parse(), defaults, output, error);
    } catch (const std::exception& exception) { error=exception.what(); return false; }
}
} // namespace metallic::material
