#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include "Runtime/Scene/scene.h"

#include <json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <cstring>

namespace metallic::render {
namespace {
using Json = nlohmann::json;

void require(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

std::string emitProgram(const MaterialValueIR& ir, uint32_t id, MaterialValueManifest& manifest)
{
    manifest.programId = id;
    manifest.parameterMask = ir.usage().parameterMask;
    manifest.inputMask = ir.usage().inputMask;
    manifest.textureMask = ir.usage().textureMask;
    manifest.footprintMask = ir.usage().footprintMask;
    manifest.featureMask = ir.usage().features;
    manifest.irHash = ir.hash();
    manifest.expressionNodes = static_cast<uint32_t>(ir.nodes().size());
    std::string statements;
    for (size_t index = 0; index < ir.nodes().size(); ++index) {
        const auto& node = ir.nodes()[index];
        const auto operand = [&](uint32_t i) { return "v" + std::to_string(node.operands[i]); };
        const auto a = operand(0), b = operand(1), c = operand(2);
        std::string value;
        switch (node.op) {
        case MaterialValueOp::Constant:
            value = "float4(";
            for (uint32_t i = 0; i < 4; ++i) { value += (i ? "," : "") + Json(node.constant[i]).dump(); }
            value += ")"; break;
        case MaterialValueOp::Parameter: value = "instance.parameters[" + std::to_string(node.index) + "]"; break;
        case MaterialValueOp::Position: value = "float4(position,0.0)"; break;
        case MaterialValueOp::GeometryNormal: value = "float4(geometryNormal,0.0)"; break;
        case MaterialValueOp::UV: value = "float4(uv,0.0,0.0)"; break;
        case MaterialValueOp::BaseColor: value = "float4(Metallic.WorkingColor::toLinearRec709(material.baseColor.rgb),material.baseColor.a)"; break;
        case MaterialValueOp::Metallic: value = "float4(material.params.x)"; break;
        case MaterialValueOp::Roughness: value = "float4(material.params.y)"; break;
        case MaterialValueOp::Emissive: value = "float4(Metallic.WorkingColor::toLinearRec709(material.emissive.rgb),material.emissive.a)"; break;
        case MaterialValueOp::Add: value = a + "+" + b; break;
        case MaterialValueOp::Multiply: value = a + "*" + b; break;
        case MaterialValueOp::Dot: value = "float4(dot(" + a + "," + b + "))"; break;
        case MaterialValueOp::Lerp: value = "lerp(" + a + "," + b + ",saturate(" + c + "))"; break;
        case MaterialValueOp::Sin: value = "sin(" + a + ")"; break;
        case MaterialValueOp::Fract: value = "frac(" + a + ")"; break;
        case MaterialValueOp::Abs: value = "abs(" + a + ")"; break;
        case MaterialValueOp::Saturate: value = "saturate(" + a + ")"; break;
        case MaterialValueOp::Clamp: value = "min(max(" + a + "," + b + ")," + c + ")"; break;
        case MaterialValueOp::Normalize: value = "materialValueNormalize(" + a + ")"; break;
        case MaterialValueOp::NormalMap: value = "materialValueNormalize(" + a + "*2.0-1.0)"; break;
        case MaterialValueOp::UVTransform: value = "float4(dot("+a+".xy,"+b+".xy)+"+b+".z,dot("+a+".xy,"+c+".xy)+"+c+".z,0,0)"; break;
        case MaterialValueOp::Swizzle:
            value = a + ".";
            for (uint32_t i = 0; i < 4; ++i) { value += "xyzw"[(node.index >> (2*i)) & 3]; }
            break;
        case MaterialValueOp::Select: value = "(" + a + ".x>0.0?" + b + ":" + c + ")"; break;
        case MaterialValueOp::TextureSample:
            value = "sampleMaterialValueTexture(material," + std::to_string(node.index) + "u," + a + ".xy," +
                std::to_string(static_cast<uint32_t>(node.footprint)) + "u," + b + "," +
                (node.operandCount == 3 ? c : "float4(0)") + ",textureContext)"; break;
        default: throw std::runtime_error("Surface IR cannot consume Coverage alpha input");
        }
        // Preserve the legacy input semantics; bound arithmetic intermediates.
        if (node.operandCount) { value = "clamp(" + value + ",-1e6,1e6)"; }
        statements += "    float4 v" + std::to_string(index) + " = " + value + ";\n";
    }
    for (const auto& [key, root] : ir.outputs()) {
        const auto value = "v" + std::to_string(root);
        // IR arithmetic stays in its authored Rec.709 basis. Material bounds
        // apply after returning to working RGB so native AP1 colors survive.
        if (key == "baseColor") { statements += "    result.baseColor.rgb = saturate(Metallic.WorkingColor::fromLinearRec709(" + value + ".rgb));\n"; manifest.outputMask |= 1; }
        if (key == "metallic") { statements += "    result.params.x = saturate(" + value + ".x);\n"; manifest.outputMask |= 2; }
        if (key == "roughness") { statements += "    result.params.y = saturate(" + value + ".x);\n"; manifest.outputMask |= 4; }
        if (key == "emissive") { statements += "    result.emissive.rgb = clamp(Metallic.WorkingColor::fromLinearRec709(" + value + ".rgb),0.0,1e6);\n"; manifest.outputMask |= 8; }
    }
    return "PathTraceMaterial materialValue" + std::to_string(id) +
        "(MaterialValueInstance instance, float3 position, float3 geometryNormal, float2 uv, PathTraceMaterial material, MaterialValueTextureContext textureContext)\n{\n"
        "    PathTraceMaterial result = material;\n" + statements + "    return result;\n}\n";
}
} // namespace

std::shared_ptr<const MaterialValueProgramSet> MaterialValueProgramSet::create(
    std::span<const scene::RenderMaterial> materials, std::string& diagnostics)
{
    diagnostics.clear();
    try {
        require(materials.size() <= UINT32_MAX / sizeof(MaterialValueInstance), "Material input offsets exceed uint32");
        auto result = std::make_shared<MaterialValueProgramSet>();
        std::map<std::string, uint32_t> programs;
        std::map<std::string, MaterialValueIR> surfaceIR;
        std::vector<std::string> sources;
        std::vector<std::string> coverageSources;
        std::map<std::string, MaterialCoverageSlice> coveragePrograms;
        for (const auto& material : materials) {
            require(std::all_of(material.valueParameters.begin(), material.valueParameters.end(),
                [](float v) { return std::isfinite(v) && std::abs(v) <= 1e6f; }), "Value parameters must be finite and within +/-1e6");
            if (material.valueProgram.empty()) { sources.emplace_back(); coverageSources.emplace_back(); continue; }
            const auto ir = MaterialValueIR::parse(material.valueProgram);
            const auto coverage = ir.slice(true);
            const auto surface = ir.slice(false);
            std::string coverageSource;
            if (!coverage.outputs().empty()) {
                require(material.alphaMode == "MASK" && !material.rtxcrHair,
                    "Coverage programs require alphaMode MASK on a Surface material");
                coverageSource = coverage.canonical();
                if (!coveragePrograms.contains(coverageSource)) {
                    coveragePrograms.emplace(coverageSource, compileMaterialCoverageSlice(coverage));
                }
            }
            coverageSources.push_back(coverageSource);
            std::string source;
            if (!surface.outputs().empty()) {
                require((material.alphaMode == "OPAQUE" || material.alphaMode == "MASK") && !material.rtxcrHair && !material.unlit &&
                    material.transmissionFactor == 0.0f && material.diffuseTransmissionFactor == 0.0f,
                    "Surface Value programs require lit, non-transmissive Surface materials");
                source = surface.canonical();
                programs.emplace(source, 0);
                surfaceIR.emplace(source, surface);
            }
            require(programs.size() <= kMaxMaterialValuePrograms, "Scene exceeds 64 custom Value programs");
            require(coveragePrograms.size() <= kMaxMaterialValuePrograms, "Scene exceeds 64 Coverage programs");
            sources.push_back(source);
        }
        result->source_ = "// Material Value IR v1; generated Surface slice, shared 80-byte instance ABI.\n"
            "struct MaterialValueTextureContext { float normalizedRayLod; uint textureCount; uint ntcCount; };\n"
            "struct MaterialValueInstance { uint programId; uint coverageOffset; uint coverageCount; uint coverageFlags; float4 parameters[4]; };\n"
            "float4 materialValueNormalize(float4 v) { return float4(dot(v.xyz,v.xyz)>1e-20?normalize(v.xyz):float3(0,0,1),0); }\n";
        const bool usesTextures = std::any_of(surfaceIR.begin(), surfaceIR.end(), [](const auto& item) { return item.second.usage().textureMask != 0; });
        if (usesTextures) { result->source_ += "float4 sampleMaterialValueTexture(PathTraceMaterial material, uint slot, float2 uv, uint policy, float4 dxLod, float4 dy, MaterialValueTextureContext context);\n"; }
        for (auto& [source, id] : programs) {
            id = ++result->programCount_;
            result->manifests_.emplace_back();
            result->source_ += emitProgram(surfaceIR.at(source), id, result->manifests_.back());
        }
        result->source_ += "PathTraceMaterial evaluateMaterialValue(uint materialIndex, float3 position, float3 geometryNormal, float2 uv, PathTraceMaterial material, MaterialValueTextureContext textureContext = (MaterialValueTextureContext)0)\n{\n"
            "    MaterialValueInstance instance = resolveBuffer<StructuredBuffer<MaterialValueInstance>>(getResourceParameters<SceneResourceParameters>().materialValues)[materialIndex];\n"
            "    switch (instance.programId) {\n";
        for (const auto& [source, id] : programs) {
            const auto name = std::to_string(id);
            result->source_ += "    case " + name + ": return materialValue" + name + "(instance, position, geometryNormal, uv, material, textureContext);\n";
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
        result->inputBytes_.resize(result->instances_.size() * sizeof(MaterialValueInstance));
        std::map<std::string, uint32_t> coverageOffsets;
        for (const auto& [source, slice] : coveragePrograms) {
            require(result->inputBytes_.size() <= UINT32_MAX - slice.instructions.size() * sizeof(MaterialCoverageInstruction),
                "Coverage byte offsets exceed uint32");
            coverageOffsets[source] = static_cast<uint32_t>(result->inputBytes_.size());
            const auto* first = reinterpret_cast<const std::byte*>(slice.instructions.data());
            result->inputBytes_.insert(result->inputBytes_.end(), first, first + slice.instructions.size() * sizeof(MaterialCoverageInstruction));
        }
        for (size_t i = 0; i < coverageSources.size(); ++i) {
            if (coverageSources[i].empty()) { continue; }
            const auto& slice = coveragePrograms.at(coverageSources[i]);
            result->instances_[i].coverageOffset = coverageOffsets.at(coverageSources[i]);
            result->instances_[i].coverageCount = static_cast<uint32_t>(slice.instructions.size());
            result->instances_[i].coverageFlags = slice.usesBaseAlpha ? 1u : 0u;
        }
        result->coverageProgramCount_ = static_cast<uint32_t>(coveragePrograms.size());
        std::memcpy(result->inputBytes_.data(), result->instances_.data(), result->instances_.size() * sizeof(MaterialValueInstance));
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
