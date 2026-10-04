#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <json.hpp>

namespace metallic::scene { struct RenderMaterial; }

namespace metallic::material {

enum class FeatureCategory : uint32_t
{
    Dynamic = 1, Specialization = 2, Visibility = 4, PipelineState = 8, Closure = 16, Structural = 32
};

// Author intent, never a shader keyword or compiled variant number.
enum class FeaturePolicy : uint32_t { Auto, Dynamic, Specialization, Closure };
struct MaterialFeaturePolicies
{
    FeaturePolicy metalness = FeaturePolicy::Auto;
    FeaturePolicy transmission = FeaturePolicy::Auto;
    bool operator==(const MaterialFeaturePolicies&) const = default;
};

struct MaterialFeatureDesc
{
    std::string_view name;
    uint32_t categories;
    FeaturePolicy defaultPolicy;
    bool configurablePolicy;
};

// Compiler decisions for the existing OpenPBR deferred lighting kernels.
// These numeric values are private runtime ABI with MATERIAL_CLASS, not assets.
enum class SurfaceProgramClass : uint32_t { Dielectric = 1, Conductor = 2, Opaque = 3, General = 4 };
enum class FeatureCompileTarget : uint32_t { Deferred, RayHit };
struct MaterialFeatureResolution
{
    uint64_t programSignature = 0;
    uint64_t visibilitySignature = 0;
    uint64_t pipelineSignature = 0;
    SurfaceProgramClass surfaceProgram = SurfaceProgramClass::General;
    FeaturePolicy metalnessDecision = FeaturePolicy::Dynamic;
    FeaturePolicy transmissionDecision = FeaturePolicy::Dynamic;
};

std::span<const MaterialFeatureDesc> materialFeatureDescriptors();
std::string_view featurePolicyName(FeaturePolicy policy);
bool validFeaturePolicies(const MaterialFeaturePolicies& policies);
// Sparse objects are overlays. Explicit Auto resets an inherited policy.
bool overlayFeaturePolicies(const nlohmann::json& value, MaterialFeaturePolicies& policies, std::string& error);
nlohmann::json serializeFeaturePolicies(const MaterialFeaturePolicies& policies);
MaterialFeatureResolution resolveMaterialFeatures(const scene::RenderMaterial& material,
    FeatureCompileTarget target = FeatureCompileTarget::Deferred);

} // namespace metallic::material
