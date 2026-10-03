#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::scene { struct RenderMaterial; }

namespace metallic::render {

inline constexpr uint32_t kMaterialValueBinding = 97;
inline constexpr uint32_t kMaxMaterialValuePrograms = 64;
inline constexpr size_t kMaxMaterialValueSourceBytes = 16384;

// Separate from the legacy model payload. Program zero preserves its inputs.
struct alignas(16) MaterialValueInstance
{
    uint32_t programId = 0;
    std::array<uint32_t, 3> reserved{};
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
    // Outputs: baseColor, metallic, roughness, emissive.
    uint32_t outputMask = 0;
    uint32_t expressionNodes = 0;
    // This version has no external resources, writes, coverage or texture gradients.
};

// Closed, read-only expression language lowered to static Slang dispatch.
// No texture/coverage/normal outputs are accepted by this first M2 slice.
class MaterialValueProgramSet final
{
public:
    static std::shared_ptr<const MaterialValueProgramSet> create(
        std::span<const scene::RenderMaterial> materials, std::string& diagnostics);
    uint32_t programCount() const { return programCount_; }
    uint64_t key() const { return key_; }
    const std::string& source() const { return source_; }
    std::span<const MaterialValueInstance> instances() const { return instances_; }
    std::span<const MaterialValueManifest> manifests() const { return manifests_; }
    // Materialize only validated generated code. Detect key collisions/corruption.
    bool writeInclude(const std::filesystem::path& root, std::filesystem::path& directory,
        std::string& diagnostics) const;
private:
    uint32_t programCount_ = 0;
    uint64_t key_ = 0;
    std::string source_;
    std::vector<MaterialValueInstance> instances_;
    std::vector<MaterialValueManifest> manifests_;
};

} // namespace metallic::render
