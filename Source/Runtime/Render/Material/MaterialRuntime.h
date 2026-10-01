#pragma once

#include "Runtime/Render/Material/LegacyMaterialPayload.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/GAPI/RHI.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metallic::render {

// Optional allocation policy for material edits (e.g. budget refusal). Null uses
// Device::createBuffer. Failures leave the published generation untouched.
using MaterialBufferAllocator = Result<std::unique_ptr<Buffer>> (*)(Device&, const BufferDesc&);

enum class MaterialDomain : uint32_t { Surface, Fiber };
enum class MaterialEvaluationTarget : uint32_t { SurfaceRayHit, RayHitWithFiber, VisibilityBuffer };
// Stable shader ABI; zero remains reserved for old uploads/test fixtures.
enum class MaterialProgramId : uint32_t { OpenPBRComposite = 1, RTXCRChiang = 2 };
enum class MaterialParameterType : uint32_t { Float4, Texture, Float, UInt };

struct MaterialParameterSchema
{
    std::string_view id;
    MaterialParameterType type;
    uint32_t offset;
    uint32_t size;
};

struct MaterialSchema
{
    uint64_t abi;
    uint32_t byteSize;
    uint32_t alignment;
    std::span<const MaterialParameterSchema> parameters;
};

struct MaterialCapabilities
{
    bool rayHit;
    bool visibilityBuffer;
    bool externalLayer;
    bool exactStandalonePdf;
};

struct MaterialDefinition
{
    MaterialProgramId id;
    std::string_view name;
    MaterialDomain domain;
    uint32_t implementationRevision;
    MaterialCapabilities capabilities;
    std::string_view approximations;
};

struct MaterialProgram
{
    const MaterialDefinition* definition;
    const MaterialSchema* schema;
    // Built-in semantic key, not the Slang kernel/cache key. Parameters and
    // texture handles never participate. Kernel keys remain in SlangCompiler.
    uint64_t key;
};

struct MaterialInstance
{
    const MaterialProgram* program;
    uint32_t parameterIndex;
};

inline constexpr uint64_t kLegacyMaterialABI = 0x4d41544c00000001ull;

std::span<const MaterialProgram> builtinMaterialPrograms();
const MaterialProgram* findMaterialProgram(MaterialProgramId id);
MaterialProgramId legacyMaterialProgramId(const LegacyMaterialPayload& parameters);

// Strict layout validation precedes allocation/copy. Destination defaults are
// always supplied by its definition; incompatible or absent values keep them.
bool validateMaterialSchema(const MaterialSchema& schema, std::string& diagnostics);
bool migrateMaterialParameters(const MaterialSchema& source, std::span<const std::byte> values,
    const MaterialSchema& destination, std::span<const std::byte> defaults,
    std::vector<std::byte>& migrated, std::string& diagnostics);

// Immutable, source-index-preserving CPU snapshot paired with one GPU upload.
// Publication is performed by ScenePathTraceResources only after upload succeeds.
// Recording and scene updates follow the existing frame-boundary coordinator.
class MaterialGeneration final
{
public:
    static std::shared_ptr<const MaterialGeneration> create(
        std::span<const LegacyMaterialPayload> parameters,
        uint64_t sourceRevision,
        std::string& diagnostics);
    // Lower an authored/versioned parameter layout to the built-in execution ABI.
    // Layout changes preserve semantic IDs/types; defaults belong to the model.
    static std::shared_ptr<const MaterialGeneration> create(
        const MaterialSchema& source, std::span<const std::byte> parameters,
        uint64_t sourceRevision, std::string& diagnostics);

    uint64_t serial() const { return serial_; }
    uint64_t sourceRevision() const { return sourceRevision_; }
    std::span<const MaterialInstance> instances() const { return instances_; }
    std::span<const LegacyMaterialPayload> parameters() const { return parameters_; }
    uint32_t programCount() const { return programCount_; }
    bool supports(MaterialEvaluationTarget target, std::string& diagnostics) const;

private:
    uint64_t serial_ = 0;
    uint64_t sourceRevision_ = 0;
    uint32_t programCount_ = 0;
    std::vector<MaterialInstance> instances_;
    std::vector<LegacyMaterialPayload> parameters_;
};

// One publication owns both the immutable CPU identity and its GPU parameters.
// Retain this object through the existing frame completion, never just its raw
// buffer pointer. Textures retain their own immutable descriptor snapshots.
class MaterialBindingGeneration final
{
public:
    MaterialBindingGeneration(std::shared_ptr<const MaterialGeneration> generation, std::unique_ptr<Buffer> buffer,
        std::shared_ptr<const MaterialValueProgramSet> values = {}, std::unique_ptr<Buffer> valueBuffer = {})
        : generation_(std::move(generation)), buffer_(std::move(buffer)),
          values_(std::move(values)), valueBuffer_(std::move(valueBuffer)) {}
    const std::shared_ptr<const MaterialGeneration>& generation() const { return generation_; }
    Buffer* buffer() const { return buffer_.get(); }
    const std::shared_ptr<const MaterialValueProgramSet>& values() const { return values_; }
    Buffer* valueBuffer() const { return valueBuffer_.get(); }
private:
    std::shared_ptr<const MaterialGeneration> generation_;
    std::unique_ptr<Buffer> buffer_;
    std::shared_ptr<const MaterialValueProgramSet> values_;
    std::unique_ptr<Buffer> valueBuffer_;
};

} // namespace metallic::render
