#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <json.hpp>
#include "MaterialClosureIR.h"
#include <optional>

namespace metallic::render {

// All values are finite float4s. Scalar constants broadcast; scalar consumers
// explicitly select x. IDs address topologically ordered immutable nodes.
enum class MaterialValueOp : uint32_t
{
    Constant, Parameter, UV, Alpha, Add, Multiply, Dot, Lerp, Sin, Fract, Abs, Saturate,
    Clamp, Normalize, NormalMap, UVTransform, Swizzle, Select,
    Position, GeometryNormal, BaseColor, Metallic, Roughness, Emissive, TextureSample
};
enum class MaterialTextureFootprint : uint32_t { ExplicitLOD, SampleGrad, RayCone };
enum class MaterialValueTexture : uint32_t { BaseColor, MetallicRoughness, Normal, Occlusion, Emissive, Specular };
enum MaterialValueFeature : uint32_t
{
    ValueParameters = 1, ValueGeometry = 2, ValueTextures = 4, ValueNormalMapping = 8,
    ValueCoverage = 16, ValueRayCone = 32, ValueGradients = 64
};
struct MaterialValueNode
{
    MaterialValueOp op = MaterialValueOp::Constant;
    std::array<uint32_t, 3> operands{};
    uint32_t operandCount = 0;
    std::array<float, 4> constant{};
    uint32_t index = 0; // Parameter slot, texture slot or packed swizzle.
    MaterialTextureFootprint footprint = MaterialTextureFootprint::ExplicitLOD;
};
struct MaterialValueUsage
{
    uint32_t parameterMask = 0, inputMask = 0, textureMask = 0, footprintMask = 0;
    uint32_t features = 0;
};

class MaterialValueIR final
{
public:
    // v1 nested expressions and v2 {nodes:{name:expression}, outputs:{...}}.
    // v2 references use {"ref":"name"}; unreachable definitions are removed.
    // v3 adds a Slab/Mix/Layer closure with Value expressions as its inputs.
    // Throws diagnostic exceptions. No partially validated IR is returned.
    static MaterialValueIR parse(std::string_view source);
    static MaterialValueIR lower(const nlohmann::json& root);
    MaterialValueIR slice(bool coverage) const;
    std::span<const MaterialValueNode> nodes() const { return nodes_; }
    const std::map<std::string, uint32_t>& outputs() const { return outputs_; }
    const MaterialValueUsage& usage() const { return usage_; }
    const std::string& canonical() const { return canonical_; }
    uint64_t hash() const { return hash_; }
    // v3 binds closure leaf inputs to named Value outputs. Topology is validated
    // by the same Closure IR/backend profile as the resolved Slab prototype.
    const std::optional<MaterialClosureIR>& closure() const { return closure_; }
private:
    friend struct MaterialValueIRBuilder;
    void finalize();
    std::vector<MaterialValueNode> nodes_;
    std::map<std::string, uint32_t> outputs_;
    MaterialValueUsage usage_;
    std::string canonical_;
    uint64_t hash_ = 0;
    std::optional<MaterialClosureIR> closure_;
};

} // namespace metallic::render
