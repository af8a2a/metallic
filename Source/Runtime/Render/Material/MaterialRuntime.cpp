#include "Runtime/Render/Material/MaterialRuntime.h"

#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <cstring>

namespace metallic::render {
namespace {

#define MATERIAL_FIELD(name, type) MaterialParameterSchema{#name, MaterialParameterType::type, \
    offsetof(LegacyMaterialPayload, name), sizeof(LegacyMaterialPayload::name)}

constexpr std::array kParameters{
    MATERIAL_FIELD(baseColor, Float4), MATERIAL_FIELD(emissive, Float4),
    MATERIAL_FIELD(params, Float4), MATERIAL_FIELD(textureParams, Float4),
    MATERIAL_FIELD(glassParams, Float4), MATERIAL_FIELD(attenuationColor, Float4),
    MATERIAL_FIELD(diffuseTransmission, Float4), MATERIAL_FIELD(rtxcrHairBaseColor, Float4),
    MATERIAL_FIELD(rtxcrHairParams0, Float4), MATERIAL_FIELD(rtxcrHairParams1, Float4),
    MATERIAL_FIELD(rtxcrHairDiffuseTint, Float4), MATERIAL_FIELD(baseColorTexture, Texture),
    MATERIAL_FIELD(metallicRoughnessTexture, Texture), MATERIAL_FIELD(normalTexture, Texture),
    MATERIAL_FIELD(occlusionTexture, Texture), MATERIAL_FIELD(emissiveTexture, Texture),
    MATERIAL_FIELD(transmissionTexture, Texture), MATERIAL_FIELD(thicknessTexture, Texture),
    MATERIAL_FIELD(diffuseTransmissionTexture, Texture), MATERIAL_FIELD(diffuseTransmissionColorTexture, Texture),
    MATERIAL_FIELD(specular, Float4), MATERIAL_FIELD(specularTexture, Texture),
    MATERIAL_FIELD(specularColorTexture, Texture),
};
#undef MATERIAL_FIELD

constexpr MaterialSchema kSchema{kLegacyMaterialABI, sizeof(LegacyMaterialPayload),
    alignof(LegacyMaterialPayload), kParameters};
constexpr std::array kDefinitions{
    MaterialDefinition{MaterialProgramId::OpenPBRComposite, "OpenPBRComposite.Legacy", MaterialDomain::Surface,
        1, {true, true, false, true},
        "Existing glTF mapping; sheen/coat and dispersion disabled; straight transmission shadows"},
    MaterialDefinition{MaterialProgramId::RTXCRChiang, "RTXCRChiang.DOTS", MaterialDomain::Fiber,
        1, {true, false, false, false},
        "DOTS triangle interaction; existing approximate environment-light MIS PDF"},
};

constexpr MaterialProgram makeProgram(const MaterialDefinition& definition)
{
    uint64_t key = 14695981039346656037ull;
    for (uint64_t component : {uint64_t(definition.id), uint64_t(definition.domain),
             uint64_t(definition.implementationRevision), kSchema.abi,
             uint64_t(kSchema.byteSize), uint64_t(kSchema.alignment)}) {
        for (uint32_t byte = 0; byte < 8; ++byte) {
            key = (key ^ ((component >> (byte * 8)) & 255u)) * 1099511628211ull;
        }
    }
    return {&definition, &kSchema, {.definitionHash = key, .domain = definition.domain}};
}

constexpr std::array kPrograms{
    makeProgram(kDefinitions[0]),
    makeProgram(kDefinitions[1]),
};
std::atomic<uint64_t> nextGeneration{1};

} // namespace

std::span<const MaterialProgram> builtinMaterialPrograms()
{
    return kPrograms;
}

const MaterialProgram* findMaterialProgram(MaterialProgramId id)
{
    for (const auto& program : kPrograms) {
        if (program.definition->id == id) { return &program; }
    }
    return nullptr;
}

MaterialProgramId legacyMaterialProgramId(const LegacyMaterialPayload& parameters)
{
    return parameters.rtxcrHairBaseColor[3] > 0.5f
        ? MaterialProgramId::RTXCRChiang : MaterialProgramId::OpenPBRComposite;
}

const MaterialProgram* findMaterialProgram(std::string_view implementation)
{
    for (const auto& program : builtinMaterialPrograms()) {
        if (program.definition->name == implementation) { return &program; }
    }
    return nullptr;
}

bool validateMaterialSchema(const MaterialSchema& schema, std::string& diagnostics)
{
    diagnostics.clear();
    if (schema.abi == 0 || schema.byteSize == 0 || !std::has_single_bit(schema.alignment) ||
        schema.alignment < 4 || schema.byteSize % schema.alignment != 0) {
        diagnostics = "Invalid material layout size, alignment or ABI.";
        return false;
    }
    for (size_t i = 0; i < schema.parameters.size(); ++i) {
        const auto& field = schema.parameters[i];
        uint32_t size = 0;
        switch (field.type) {
        case MaterialParameterType::Float4: size = 16; break;
        case MaterialParameterType::Texture: size = sizeof(LegacyMaterialPayload::TextureInfo); break;
        case MaterialParameterType::Float:
        case MaterialParameterType::UInt: size = 4; break;
        }
        if (field.id.empty() || size == 0 || field.size != size || field.offset % 4 != 0 ||
            field.offset > schema.byteSize || field.size > schema.byteSize - field.offset) {
            diagnostics = "Invalid material field: " + std::string(field.id);
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            const auto& other = schema.parameters[j];
            if (field.id == other.id ||
                (field.offset < other.offset + other.size && other.offset < field.offset + field.size)) {
                diagnostics = "Duplicate or overlapping material field: " + std::string(field.id);
                return false;
            }
        }
    }
    return true;
}

bool migrateMaterialParameters(const MaterialSchema& source, std::span<const std::byte> values,
    const MaterialSchema& destination, std::span<const std::byte> defaults,
    std::vector<std::byte>& migrated, std::string& diagnostics)
{
    if (!validateMaterialSchema(source, diagnostics) || !validateMaterialSchema(destination, diagnostics)) { return false; }
    if (values.size() != source.byteSize || defaults.size() != destination.byteSize) {
        diagnostics = "Material parameter byte count does not match schema.";
        return false;
    }
    std::vector<std::byte> candidate(defaults.begin(), defaults.end());
    for (const auto& field : destination.parameters) {
        auto previous = std::find_if(source.parameters.begin(), source.parameters.end(),
            [&](const auto& value) { return value.id == field.id; });
        if (previous == source.parameters.end() || previous->type != field.type || previous->size != field.size) {
            diagnostics += "Defaulted material parameter '" + std::string(field.id) + "' (missing or changed type).\n";
            continue;
        }
        std::memcpy(candidate.data() + field.offset, values.data() + previous->offset, field.size);
    }
    migrated = std::move(candidate);
    return true;
}

std::shared_ptr<const MaterialGeneration> MaterialGeneration::create(
    const MaterialSchema& source, std::span<const std::byte> parameters,
    uint64_t sourceRevision, std::string& diagnostics)
{
    if (!validateMaterialSchema(source, diagnostics)) { return {}; }
    if (parameters.empty() || parameters.size() % source.byteSize != 0 ||
        parameters.size() / source.byteSize > UINT32_MAX) {
        diagnostics = "Material parameter array has an invalid stride or count.";
        return {};
    }
    const LegacyMaterialPayload defaults;
    std::vector<LegacyMaterialPayload> lowered(parameters.size() / source.byteSize);
    std::string warnings;
    for (size_t index = 0; index < lowered.size(); ++index) {
        std::vector<std::byte> migrated;
        if (!migrateMaterialParameters(source, parameters.subspan(index * source.byteSize, source.byteSize),
                kSchema, std::as_bytes(std::span(&defaults, 1)), migrated, diagnostics)) { return {}; }
        // One layout diagnostic rather than repeating it per instance.
        if (index == 0) { warnings = diagnostics; }
        std::memcpy(&lowered[index], migrated.data(), sizeof(LegacyMaterialPayload));
    }
    auto generation = create(lowered, sourceRevision, diagnostics);
    if (generation) { diagnostics = std::move(warnings); }
    return generation;
}

bool MaterialGeneration::supports(MaterialEvaluationTarget target, std::string& diagnostics) const
{
    diagnostics.clear();
    for (const auto& instance : instances_) {
        const auto& definition = *instance.program->definition;
        const bool supported = target == MaterialEvaluationTarget::VisibilityBuffer
            ? definition.capabilities.visibilityBuffer
            : definition.capabilities.rayHit && (definition.domain != MaterialDomain::Fiber ||
                target == MaterialEvaluationTarget::RayHitWithFiber);
        if (!supported) {
            diagnostics = "Material " + std::to_string(instance.parameterIndex) + " (" +
                std::string(definition.name) + ") does not support this execution target.";
            return false;
        }
    }
    return true;
}

std::shared_ptr<const MaterialGeneration> MaterialGeneration::create(
    std::span<const LegacyMaterialPayload> parameters,
    uint64_t sourceRevision,
    std::string& diagnostics)
{
    diagnostics.clear();
    if (!validateMaterialSchema(kSchema, diagnostics)) { return {}; }
    if (parameters.empty() || parameters.size() > std::numeric_limits<uint32_t>::max()) {
        diagnostics = "Material generation requires a nonempty, uint32-indexable parameter array.";
        return {};
    }
    auto candidate = std::make_shared<MaterialGeneration>();
    candidate->sourceRevision_ = sourceRevision;
    candidate->parameters_.assign(parameters.begin(), parameters.end());
    candidate->instances_.reserve(parameters.size());
    std::array<bool, kPrograms.size()> used{};
    for (uint32_t index = 0; index < parameters.size(); ++index) {
        auto& payload = candidate->parameters_[index];
        const auto id = legacyMaterialProgramId(payload);
        const float encoded = payload.textureParams[2];
        if (!std::isfinite(payload.rtxcrHairBaseColor[3]) ||
            (encoded != 0.0f && encoded != static_cast<float>(id))) {
            diagnostics = "Material " + std::to_string(index) + " has an incompatible program identity.";
            return {};
        }
        const auto* program = findMaterialProgram(id);
        payload.textureParams[2] = static_cast<float>(id);
        candidate->instances_.push_back({program, index});
        used[static_cast<uint32_t>(id) - 1] = true;
    }
    for (bool value : used) { candidate->programCount_ += value ? 1u : 0u; }
    candidate->serial_ = nextGeneration.fetch_add(1, std::memory_order_relaxed);
    return candidate;
}

} // namespace metallic::render
