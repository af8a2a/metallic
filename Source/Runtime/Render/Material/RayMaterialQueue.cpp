#include "RayMaterialQueue.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include <algorithm>

namespace metallic::render {
RayMaterialProgramTable buildRayMaterialProgramTable(std::span<const MaterialProgramKey> keys)
{
    RayMaterialProgramTable result;
    for (const auto& key : keys) {
        const auto found = std::ranges::find(result.programs, key);
        const auto index = uint32_t(found - result.programs.begin());
        if (found == result.programs.end()) {
            result.programs.push_back(key);
        }
        result.materialPrograms.push_back(index);
    }
    return result;
}

Result<> RayMaterialQueue::initialize(Device& device, std::string& log)
{
    const char* entries[]{"rayMaterialResetMain", "rayMaterialCountMain", "rayMaterialPrefixMain",
                          "rayMaterialScatterMain"};
    for (size_t i = 0; i < kernels_.size(); ++i) {
        auto result = ShaderRegistry::instance().getComputeKernel(
            device,
            {.moduleName = "Features/PathTracing/RayMaterialQueue",
             .entryPointName = entries[i],
             .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<RayMaterialQueueParameters>(kRayMaterialQueueABI), .debugName = entries[i]},
            kernels_[i], log);
        if (!result) {
            return result;
        }
    }
    return {};
}

Result<> RayMaterialQueue::classify(CommandBuffer& commands, ParameterWriter& writer, Buffer& hits,
                                    Buffer& materialPrograms, Buffer& bins, Buffer& indices, Buffer& cursors,
                                    Buffer& status, uint32_t hitCount, uint32_t materialCount, uint32_t programCount,
                                    uint32_t capacity, uint64_t generation)
{
    if (programCount == 0 || programCount > 4096 || hitCount > 65535u * 64 || capacity > 65535u * 64 ||
        hits.desc().size < uint64_t(hitCount) * 16 || materialPrograms.desc().size < uint64_t(materialCount) * 4 ||
        bins.desc().size < uint64_t(programCount) * 16 || cursors.desc().size < uint64_t(programCount) * 4 ||
        indices.desc().size < uint64_t(capacity) * 4 || status.desc().size < 16) {
        return makeError(Error::InvalidArgument);
    }
    RayMaterialQueueParameters params{writer.bufferSpan<RayMaterialHitKey>(&hits),
                                      writer.bufferSpan<uint32_t>(&materialPrograms),
                                      writer.bufferSpan<RayMaterialBin>(&bins),
                                      writer.bufferSpan<uint32_t>(&indices),
                                      writer.bufferSpan<uint32_t>(&cursors),
                                      writer.bufferSpan<uint32_t>(&status),
                                      hitCount,
                                      materialCount,
                                      programCount,
                                      capacity,
                                      uint32_t(generation),
                                      uint32_t(generation >> 32)};
    auto encoded = writer.encode(params, kRayMaterialQueueABI);
    if (!encoded) {
        return makeError(encoded.error());
    }
    const auto barrier = [&]() {
        std::array<BufferBarrierDesc, 4> barriers;
        std::array buffers{&bins, &indices, &cursors, &status};
        for (size_t i = 0; i < buffers.size(); ++i) {
            barriers[i] = {
                .buffer = buffers[i],
                .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
        }
        return commands.synchronize({.buffers = barriers});
    };
    const uint32_t groups[]{(std::max({programCount, capacity, 4u}) + 63) / 64, std::max(1u, (hitCount + 63) / 64), 1,
                            std::max(1u, (hitCount + 63) / 64)};
    for (size_t i = 0; i < kernels_.size(); ++i) {
        if (auto result = kernels_[i].dispatch(commands, *encoded, groups[i]); !result) {
            return result;
        }
        if (auto result = barrier(); !result) {
            return result;
        }
    }
    return {};
}
} // namespace metallic::render
