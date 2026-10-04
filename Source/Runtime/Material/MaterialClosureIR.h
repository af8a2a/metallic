#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

enum class MaterialClosureOp : uint32_t { Slab, Mix, Layer };
enum class MaterialClosureFamily : uint32_t { SingleSlabClosure, DualSlabClosure, OpenPBRCompositeClosure };

// A resolved diffuse interface plus nonnegative RGB optical depth beneath it.
// Optical depth participates when this slab is the top of a Layer. The prototype
// has an opaque terminal substrate and no specular/refraction/multiple scattering.
struct MaterialSlabRecord
{
    std::array<float, 4> reflectance{0.5f, 0.5f, 0.5f, 0};
    std::array<float, 4> opticalDepth{};
};
static_assert(sizeof(MaterialSlabRecord) == 32);

struct MaterialClosureNode
{
    MaterialClosureOp op = MaterialClosureOp::Slab;
    std::array<uint32_t, 2> operands{}; // Top/bottom for Layer; A/B for Mix.
    MaterialSlabRecord slab;
    float weight = 0.5f; // Mix only.
    uint32_t normalBasis = 0; // Symbolic basis identity; prototype backend accepts context basis 0.
};

struct MaterialClosureComplexity
{
    uint32_t closureRecordCount = 0;
    uint32_t scatteringLobeCount = 0;
    uint32_t normalBasisCount = 0;
    uint32_t layerDepth = 0;
    uint32_t payloadBytes = 0; // Exact resolved family payload after lowering; 0 before lowering.
    uint32_t operatorCount = 0;
};

struct RealtimeBackendProfile
{
    uint32_t maxClosures = 2;
    uint32_t maxOperators = 1;
    uint32_t maxNormalBases = 1;
    uint32_t maxLayerDepth = 1;
    uint32_t maxPayloadBytes = 96;
};

// Immutable, reachable, topological DAG. Construction accepts larger graphs;
// the real-time restrictions are checked only by lowerMaterialClosure().
class MaterialClosureIR final
{
public:
    static MaterialClosureIR create(std::span<const MaterialClosureNode> nodes, uint32_t root);
    std::span<const MaterialClosureNode> nodes() const { return nodes_; }
    uint32_t root() const { return static_cast<uint32_t>(nodes_.size() - 1); }
    const MaterialClosureComplexity& complexity() const { return complexity_; }
    const std::string& canonical() const { return canonical_; }
    uint64_t hash() const { return hash_; }
private:
    MaterialClosureIR() = default;
    std::vector<MaterialClosureNode> nodes_;
    MaterialClosureComplexity complexity_;
    std::string canonical_;
    uint64_t hash_ = 0;
};

// Upload ABI shared with SlabClosure.slang: five float4s. Normal comes from
// SurfaceMaterialContext, so upload size differs from resolved closure size.
struct MaterialClosurePacket
{
    MaterialSlabRecord first, second;
    std::array<float, 4> control{}; // x: 0=single, 1=mix, 2=layer; y: mix weight.
};
static_assert(sizeof(MaterialClosurePacket) == 80);
struct LoweredMaterialClosure
{
    MaterialClosureFamily family;
    MaterialClosureComplexity complexity;
    MaterialClosurePacket packet;
};
LoweredMaterialClosure lowerMaterialClosure(const MaterialClosureIR& ir, const RealtimeBackendProfile& profile = {});

} // namespace metallic::render
