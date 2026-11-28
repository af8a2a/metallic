#include "Runtime/Render/Shadow/VirtualShadowMaps.h"

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLODInterop.h"

#include <cstring>
#include <vector>

namespace metallic::render {
namespace {

struct LODStatisticsSource
{
    uint32_t statistics, viewCount;
};
static_assert(sizeof(LODStatisticsSource) == 8);

struct LODStatisticsPush
{
    uint32_t sources, output, sourceCount;
};
static_assert(sizeof(LODStatisticsPush) == 12);

struct LODStatisticsResources
{
    std::unique_ptr<Buffer> sources;
    std::shared_ptr<Buffer> output;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::vector<std::shared_ptr<void>> sourceOwners;
    std::vector<ResourceLease> leases;
};

} // namespace

Result<std::shared_ptr<Buffer>> VirtualShadowMaps::aggregateLODStatistics(Device& device,
    CommandBuffer& commands, std::span<const VirtualShadowMapResult> faces, std::string& log)
{
    std::vector<const VirtualShadowMapResult*> sources;
    for (const auto& face : faces) {
        if (face.lodStatistics) { sources.push_back(&face); }
    }
    if (sources.empty()) { return std::shared_ptr<Buffer>{}; }

    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    auto resources = std::make_shared<LODStatisticsResources>();
    std::vector<LODStatisticsSource> records;
    std::vector<BufferBarrierDesc> barriers;
    for (const auto* face : sources) {
        auto lease = (*registry)->storageBuffer(*face->lodStatistics);
        if (!lease) { return std::unexpected(lease.error()); }
        records.push_back({lease->shaderIndex(), face->clipmapCount + 1u});
        resources->leases.push_back(*lease);
        resources->sourceOwners.push_back(face->resourceOwner);
        barriers.push_back({.buffer = face->lodStatistics,
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}});
    }
    auto table = device.createBuffer({.size = records.size() * sizeof(LODStatisticsSource),
        .structureStride = sizeof(LODStatisticsSource), .usage = BufferUsageBits::Storage,
        .memoryLocation = MemoryLocation::HostUpload});
    if (!table) { return std::unexpected(table.error()); }
    void* destination = (*table)->map();
    if (!destination) { log += "Unable to map VSM LOD statistics sources.\n"; return makeError(Error::Failure); }
    std::memcpy(destination, records.data(), records.size() * sizeof(LODStatisticsSource));
    (*table)->flush(); (*table)->unmap();
    resources->sources = std::move(*table);
    auto output = device.createBuffer({.size = kVirtualShadowMapLODStatisticsWords * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t), .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::Device});
    if (!output) { return std::unexpected(output.error()); }
    resources->output = std::move(*output);
    auto sourceLease = (*registry)->storageBuffer(*resources->sources);
    if (!sourceLease) { return std::unexpected(sourceLease.error()); }
    auto outputLease = (*registry)->storageBuffer(*resources->output);
    if (!outputLease) { return std::unexpected(outputLease.error()); }
    resources->leases.push_back(*sourceLease);
    resources->leases.push_back(*outputLease);

    constexpr const char* entry = "vsmAggregateLODStatisticsMain";
    auto shader = ShaderRegistry::instance().getShader({
        .moduleName = "Features/VirtualShadowMaps/VirtualShadowMapLODStatistics", .entryPointName = entry,
        .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, {.logCacheHitsAtInfo = false}, log);
    if (!shader) { return std::unexpected(shader.error()); }
    auto module = ShaderRegistry::instance().getShaderModule(device, {.spirv = shader->spirv, .debugName = entry});
    if (!module) { return std::unexpected(module.error()); }
    resources->shader = std::move(*module);
    auto pipeline = ShaderRegistry::instance().getComputePipeline(device, {
        .computeShader = {resources->shader.get()}, .usesBindlessHeap = true,
        .bindlessUserPushDataSize = sizeof(LODStatisticsPush)});
    if (!pipeline) { return std::unexpected(pipeline.error()); }
    resources->pipeline = std::move(*pipeline);
    if (auto r = commands.retainResource(resources); !r) { return std::unexpected(r.error()); }
    if (auto r = (*registry)->bind(commands, resources->leases); !r) { return std::unexpected(r.error()); }
    commands.hostWriteBarrier();
    if (auto r = commands.synchronize({.buffers = barriers}); !r) { return std::unexpected(r.error()); }
    const LODStatisticsPush push{sourceLease->shaderIndex(), outputLease->shaderIndex(), uint32_t(records.size())};
    if (auto r = commands.bindExecution(resources->pipeline->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
    if (auto r = commands.dispatch(1); !r) { return std::unexpected(r.error()); }
    const BufferBarrierDesc published{.buffer = resources->output.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
    if (auto r = commands.synchronize({.buffers = {&published, 1}}); !r) { return std::unexpected(r.error()); }
    return resources->output;
}

} // namespace metallic::render
