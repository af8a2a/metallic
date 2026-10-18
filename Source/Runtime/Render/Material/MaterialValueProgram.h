#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "Runtime/Material/MaterialClosureIR.h"

namespace metallic::scene { struct RenderMaterial; }

namespace metallic::render {

inline constexpr uint32_t kMaxMaterialValuePrograms = 64;
inline constexpr size_t kMaxMaterialValueSourceBytes = 16384;

// Separate from scene transport data. Program zero preserves its inputs.
struct alignas(16) MaterialValueInstance
{
    uint32_t programId = 0;
    uint32_t coverageOffset = 0; // Coverage program ID; historical field name preserves the 80-byte ABI.
    uint32_t coverageCount = 0; // Live IR node count; zero means no custom Coverage.
    uint32_t coverageFlags = 0; // bit 0: requires base-color texture alpha.
    std::array<float, 16> parameters{};
};
static_assert(sizeof(MaterialValueInstance) == 80);
static_assert(offsetof(MaterialValueInstance, parameters) == 16);

struct MaterialValueManifest
{
    uint32_t programId = 0;
    uint32_t parameterMask = 0;
    // Inputs: position, geometry normal, UV, baseColor, metallic, roughness, emissive.
    uint32_t inputMask = 0;
    // Outputs: baseColor, metallic, roughness, emissive, OpenPBR surface/volume inputs.
    uint32_t outputMask = 0;
    uint32_t expressionNodes = 0;
    uint32_t textureMask = 0, footprintMask = 0, featureMask = 0;
    uint64_t irHash = 0;
    MaterialClosureFamily closureFamily = MaterialClosureFamily::OpenPBRCompositeClosure;
    MaterialClosureComplexity closureComplexity;
    // Surface slice only; Coverage has independent resource requirements/code.
};

// Closed, read-only expression language. Surface outputs use static Slang
// dispatch; Coverage has an independent generated module and executable key.
class MaterialValueProgramSet final
{
public:
    static std::shared_ptr<const MaterialValueProgramSet> create(
        std::span<const scene::RenderMaterial> materials, std::string& diagnostics);
    uint32_t programCount() const { return programCount_; }
    uint32_t coverageProgramCount() const { return coverageProgramCount_; }
    std::span<const std::byte> inputBytes() const { return inputBytes_; }
    uint64_t key() const { return key_; }
    const std::string& source() const { return source_; }
    uint64_t coverageKey() const { return coverageKey_; }
    const std::string& coverageSource() const { return coverageSource_; }
    std::span<const MaterialValueInstance> instances() const { return instances_; }
    std::span<const MaterialValueManifest> manifests() const { return manifests_; }
    // Materialize only validated generated code. Detect key collisions/corruption.
    bool writeInclude(const std::filesystem::path& root, std::filesystem::path& directory,
        std::string& diagnostics) const;
    bool writeCoverageInclude(const std::filesystem::path& root, std::filesystem::path& directory,
        std::string& diagnostics) const;
private:
    uint32_t programCount_ = 0;
    uint32_t coverageProgramCount_ = 0;
    std::vector<std::byte> inputBytes_;
    uint64_t key_ = 0;
    std::string source_;
    uint64_t coverageKey_ = 0;
    std::string coverageSource_;
    std::vector<MaterialValueInstance> instances_;
    std::vector<MaterialValueManifest> manifests_;
};

} // namespace metallic::render
