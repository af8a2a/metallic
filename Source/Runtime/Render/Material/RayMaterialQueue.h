#pragma once
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include <array>

namespace metallic::render {
// Executable slots are dense, generation-local IDs for FULL program keys.
// Parameter offsets/values and closure family never participate in grouping.
struct RayMaterialProgramTable
{
    std::vector<MaterialProgramKey> programs;
    std::vector<uint32_t> materialPrograms;
};
RayMaterialProgramTable buildRayMaterialProgramTable(std::span<const MaterialProgramKey> keys);

struct RayMaterialHitKey
{
    uint32_t materialIndex, pathIndex, generationLo, generationHi;
};
struct RayMaterialBin
{
    uint32_t offset, count, attempted, overflow;
};
// status: misses, invalid material/program, stale generation, dropped capacity.
// indices point back to original hit records: scattering/RNG state is not copied
// into slot order. A consumer must reject/retry work if any error count is nonzero.
struct RayMaterialQueueParameters
{
    GPUBufferSpan hits, materialPrograms, bins, indices, cursors, status;
    uint32_t hitCount, materialCount, programCount, capacity, generationLo, generationHi;
};
inline constexpr uint64_t kRayMaterialQueueABI = 0x5241594d41540001ull;
static_assert(sizeof(RayMaterialHitKey) == 16 && sizeof(RayMaterialBin) == 16);
static_assert(sizeof(RayMaterialQueueParameters) == 96);

// Bounded classifier primitive; no default renderer scheduling policy is changed.
// Caller owns all graph/frame buffers and retains the matching material generation
// until completion. Encode/classify only after hit producers have finished.
class RayMaterialQueue
{
public:
    Result<> initialize(Device& device, std::string& log);
    Result<> classify(CommandBuffer& commands, ParameterWriter& writer, Buffer& hits, Buffer& materialPrograms,
                      Buffer& bins, Buffer& indices, Buffer& cursors, Buffer& status, uint32_t hitCount,
                      uint32_t materialCount, uint32_t programCount, uint32_t capacity, uint64_t generation);

private:
    std::array<ComputeKernel, 4> kernels_;
};
} // namespace metallic::render
