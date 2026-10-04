#include "Runtime/Material/MaterialFeatures.h"
#include "Runtime/Material/MaterialValueIR.h"
#include "Runtime/Scene/Scene.h"
#include <json.hpp>
#include <array>

namespace metallic::material {
namespace {

constexpr uint32_t category(FeatureCategory value) { return static_cast<uint32_t>(value); }
constexpr std::array kFeatures{
    MaterialFeatureDesc{"baseColor", category(FeatureCategory::Dynamic), FeaturePolicy::Dynamic, false},
    MaterialFeatureDesc{"roughness", category(FeatureCategory::Dynamic), FeaturePolicy::Dynamic, false},
    MaterialFeatureDesc{"metalness", category(FeatureCategory::Closure), FeaturePolicy::Auto, true},
    MaterialFeatureDesc{"transmission", category(FeatureCategory::Closure), FeaturePolicy::Auto, true},
    MaterialFeatureDesc{"alphaMode", category(FeatureCategory::Visibility), FeaturePolicy::Auto, false},
    MaterialFeatureDesc{"doubleSided", category(FeatureCategory::PipelineState), FeaturePolicy::Auto, false},
    MaterialFeatureDesc{"unlit", category(FeatureCategory::Dynamic), FeaturePolicy::Dynamic, false},
    MaterialFeatureDesc{"BSDFImplementation", category(FeatureCategory::Specialization), FeaturePolicy::Specialization, false},
    MaterialFeatureDesc{"graphTopology", category(FeatureCategory::Structural), FeaturePolicy::Auto, false},
};

uint64_t hash(std::string_view text, uint64_t seed = 14695981039346656037ull)
{
    for (const unsigned char byte : text) { seed = (seed ^ byte) * 1099511628211ull; }
    return seed;
}

} // namespace

std::span<const MaterialFeatureDesc> materialFeatureDescriptors() { return kFeatures; }

std::string_view featurePolicyName(FeaturePolicy policy)
{
    switch (policy) {
    case FeaturePolicy::Auto: return "Auto";
    case FeaturePolicy::Dynamic: return "Dynamic";
    case FeaturePolicy::Specialization: return "Specialization";
    case FeaturePolicy::Closure: return "Closure";
    }
    return "Invalid";
}

bool validFeaturePolicies(const MaterialFeaturePolicies& policies)
{
    return policies.metalness <= FeaturePolicy::Closure && policies.transmission <= FeaturePolicy::Closure;
}

bool overlayFeaturePolicies(const nlohmann::json& value, MaterialFeaturePolicies& policies, std::string& error)
{
    error.clear();
    if (!value.is_object()) { error = "featurePolicies must be an object"; return false; }
    auto candidate = policies;
    for (const auto& [name, policy] : value.items()) {
        if ((name != "metalness" && name != "transmission") || !policy.is_string()) {
            error = "Unsupported feature policy: " + name; return false;
        }
        bool found = false;
        for (const auto choice : {FeaturePolicy::Auto, FeaturePolicy::Dynamic, FeaturePolicy::Specialization, FeaturePolicy::Closure}) {
            if (policy.get<std::string>() == featurePolicyName(choice)) {
                (name == "metalness" ? candidate.metalness : candidate.transmission) = choice;
                found = true;
                break;
            }
        }
        if (!found) { error = "Unknown policy for feature: " + name; return false; }
    }
    policies = candidate;
    return true;
}

nlohmann::json serializeFeaturePolicies(const MaterialFeaturePolicies& policies)
{
    return {{"metalness", featurePolicyName(policies.metalness)}, {"transmission", featurePolicyName(policies.transmission)}};
}

MaterialFeatureResolution resolveMaterialFeatures(const scene::RenderMaterial& material, FeatureCompileTarget target)
{
    MaterialFeatureResolution result;
    const bool transmitting = material.transmissionFactor > 0.0f;
    std::string coverageCode, surfaceCode;
    bool surfaceValues = !material.valueProgram.empty();
    if (surfaceValues) {
        try {
            const auto ir = render::MaterialValueIR::parse(material.valueProgram);
            const auto surface = ir.slice(false), coverage = ir.slice(true);
            surfaceValues = !surface.outputs().empty();
            if (surfaceValues) { surfaceCode = surface.canonical(); }
            if (!coverage.outputs().empty()) { coverageCode = coverage.canonical(); }
        } catch (const std::exception&) {
            // Diagnostics/publication reject invalid source later. Classification
            // remains conservative and cannot select an unsafe specialized lobe.
            surfaceCode = material.valueProgram;
        }
    }
    // Value programs may change metalness per hit. Ray-hit programs already use
    // the general closure. Policy hints must never remove a reachable lobe.
    const bool general = target == FeatureCompileTarget::RayHit || material.rtxcrHair ||
        surfaceValues || material.featurePolicies.transmission == FeaturePolicy::Dynamic || transmitting;
    if (!general) {
        result.surfaceProgram = SurfaceProgramClass::Opaque;
        result.transmissionDecision = FeaturePolicy::Closure;
        if (material.featurePolicies.metalness != FeaturePolicy::Dynamic) {
            result.metalnessDecision = FeaturePolicy::Closure;
            if (material.metallicFactor <= 0.0f) { result.surfaceProgram = SurfaceProgramClass::Dielectric; }
            else if (material.metallicFactor >= 1.0f && material.metallicRoughnessTexture.textureIndex < 0) {
                result.surfaceProgram = SurfaceProgramClass::Conductor;
            }
            if (result.surfaceProgram != SurfaceProgramClass::Opaque) {
                result.metalnessDecision = FeaturePolicy::Specialization;
            }
        }
    }
    // Domain/version, structural code and compiler choices only. Dynamic values,
    // resource identities, alpha mode and culling never enter this signature.
    auto signature = hash(material.rtxcrHair ? "Features/v2/RTXCRChiang" : "Features/v2/OpenPBRComposite");
    signature = hash(std::to_string(static_cast<uint32_t>(target)), signature);
    signature = hash(std::to_string(static_cast<uint32_t>(result.surfaceProgram)), signature);
    if (surfaceValues) {
        signature = hash(surfaceCode, signature);
    }
    result.programSignature = signature;
    // Coverage affects hit existence; transmission affects transport at a hit.
    result.visibilitySignature = hash("Visibility/v3/coverage/");
    result.visibilitySignature = hash(material.alphaMode, result.visibilitySignature);
    result.visibilitySignature = hash(coverageCode, result.visibilitySignature);
    result.pipelineSignature = hash(material.doubleSided ? "Pipeline/v1/two-sided" : "Pipeline/v1/back-cull");
    return result;
}

} // namespace metallic::material
