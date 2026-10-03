#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Profiling/WorkControlReplay.h"
#include "Runtime/Render/MeshletLOD.h"
#include "Runtime/Render/Profiling/CPUPhaseTrace.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"

#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

inline constexpr bool kDefaultReversedZ = true;
inline constexpr uint64_t kImmutableMetadataUploadBatchBytes = 64ull * 1024ull * 1024ull;

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

std::string resultMessage(std::string_view label, const Result<>& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

float finiteOr(float value, float fallback)
{
    return std::isfinite(value) ? value : fallback;
}

float3 transformPoint(const float matrix[16], const float3& point)
{
    return float3(
        matrix[0] * point.x + matrix[4] * point.y + matrix[8] * point.z + matrix[12],
        matrix[1] * point.x + matrix[5] * point.y + matrix[9] * point.z + matrix[13],
        matrix[2] * point.x + matrix[6] * point.y + matrix[10] * point.z + matrix[14]);
}

void includeTransformedBounds(scene::Bounds& outBounds, const scene::MeshletStreamBounds& bounds, const float matrix[16])
{
    if (bounds.valid == 0) {
        return;
    }
    const float3 minBounds(bounds.min[0], bounds.min[1], bounds.min[2]);
    const float3 maxBounds(bounds.max[0], bounds.max[1], bounds.max[2]);
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                const float3 corner(
                    x == 0 ? minBounds.x : maxBounds.x,
                    y == 0 ? minBounds.y : maxBounds.y,
                    z == 0 ? minBounds.z : maxBounds.z);
                outBounds.include(transformPoint(matrix, corner));
            }
        }
    }
}

scene::Bounds computeDrawBounds(const scene::MeshletStreamAsset& asset)
{
    scene::Bounds bounds;
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset.primitives();
    for (const scene::MeshletStreamInstanceInfo& instance : asset.instances()) {
        if (instance.visible == 0 || instance.primitiveIndex >= primitives.size()) {
            continue;
        }
        includeTransformedBounds(bounds, primitives[instance.primitiveIndex].bounds, instance.worldMatrix);
    }
    return bounds;
}

Result<> createNamedBuffer(
    Device& device,
    const BufferDesc& desc,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log,
    std::string_view label)
{
    BufferDesc shared = desc;
    if (hasFlag(shared.usage, BufferUsageBits::Storage)) {
        shared.queueAccess = shared.queueAccess | QueueAccessBits::Graphics | QueueAccessBits::Compute;
    }
    Result<> result = device.createBuffer(shared).transform([&](auto rhiValue) { outBuffer = std::move(rhiValue); });
    if (!result || outBuffer == nullptr) {
        log += resultMessage(std::string("createBuffer(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

Result<> createHostStorageBuffer(
    Device& device,
    uint64_t byteSize,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log,
    std::string_view label)
{
    return createNamedBuffer(
        device,
        BufferDesc{
            .size = byteSize,
            .structureStride = 0,
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        outBuffer,
        log,
        label);
}

Result<> updateHostBuffer(Buffer& buffer, const void* data, uint64_t byteSize)
{
    if (byteSize > buffer.desc().size || (byteSize > 0 && data == nullptr)) {
        return makeError(Error::InvalidArgument);
    }
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    if (byteSize > 0) {
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        buffer.flush({0, byteSize});
    }
    buffer.unmap();
    return {};
}

template<typename ValueType, typename Populate>
Result<> createAndPopulateHostStorageBuffer(
    Device& device,
    size_t valueCount,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log,
    std::string_view label,
    Populate&& populate)
{
    const uint64_t allocationCount = std::max<uint64_t>(valueCount, 1u);
    if (allocationCount > std::numeric_limits<uint64_t>::max() / sizeof(ValueType)) {
        log += std::string(label) + " byte size overflowed\n";
        return makeError(Error::OutOfMemory);
    }
    const uint64_t byteSize = allocationCount * sizeof(ValueType);
    Result<> result = createHostStorageBuffer(device, byteSize, outBuffer, log, label);
    if (!result) {
        return result;
    }

    auto* values = static_cast<ValueType*>(outBuffer->map());
    if (values == nullptr) {
        log += std::string("map(") + std::string(label) + ") returned null\n";
        return makeError(Error::Failure);
    }
    if (valueCount == 0) {
        values[0] = ValueType{};
    } else {
        for (size_t index = 0; index < valueCount; ++index) {
            populate(values[index], index);
        }
    }
    outBuffer->flush({0, byteSize});
    outBuffer->unmap();
    return {};
}

template<typename ValueType, typename Populate>
Result<> createAndPopulateImmutableStorageBuffer(
    Device& device, bool deviceStorage, size_t valueCount,
    std::unique_ptr<Buffer>& outBuffer, std::vector<std::byte>* uploadData,
    std::string& log, std::string_view label, Populate&& populate)
{
    if (!deviceStorage) {
        return createAndPopulateHostStorageBuffer<ValueType>(
            device, valueCount, outBuffer, log, label, std::forward<Populate>(populate));
    }
    const uint64_t allocationCount = std::max<uint64_t>(valueCount, 1u);
    if (!uploadData || allocationCount > std::numeric_limits<size_t>::max() / sizeof(ValueType)) {
        log += std::string(label) + " byte size overflowed\n";
        return makeError(Error::OutOfMemory);
    }
    const uint64_t byteSize = allocationCount * sizeof(ValueType);
    auto result = createNamedBuffer(device, BufferDesc{
        .size = byteSize,
        .structureStride = 0,
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
        .memoryLocation = MemoryLocation::Device,
        .memoryDomain = MemoryBudgetDomain::Geometry,
    }, outBuffer, log, label);
    if (!result) { return result; }
    uploadData->resize(static_cast<size_t>(byteSize));
    for (size_t index = 0; index < valueCount; ++index) {
        ValueType value{};
        populate(value, index);
        std::memcpy(uploadData->data() + index * sizeof(ValueType), &value, sizeof(ValueType));
    }
    return {};
}

Result<> allocateAndWriteBuffer(ResourceRegistry& registry, Buffer& buffer,
    ResourceLease& outHandle, std::string& log, std::string_view label)
{
    auto result = registry.storageBuffer(buffer).transform([&](auto value) { outHandle = std::move(value); });
    if (!result) { log += resultMessage(std::string("register buffer ") + std::string(label), result); }
    return result;
}

Result<> transitionBuffer(
    CommandBuffer& commandBuffer,
    Buffer& buffer,
    ResourceState& state,
    ResourceState nextState,
    bool forceBarrier = false)
{
    if (!forceBarrier && state == nextState) {
        return {};
    }
    BufferBarrierDesc barrier{
        .buffer = &buffer,
        .before = resourceSyncScope(state, PipelineStageBits::AllCommands),
        .after = resourceSyncScope(nextState, PipelineStageBits::AllCommands),
        .range = {.offset = 0, .size = buffer.desc().size},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{
        .buffers = {&barrier, 1},
    }); !commandResult) { return commandResult; }
    state = nextState;
    return {};
}

Result<> createSlangShaderModule(
    Device& device,
    const char* moduleName,
    const char* entryPoint,
    std::unique_ptr<ShaderModule>& outShader,
    std::string& log)
{
    ShaderCompileResult compileResult;
    Result<> result = compileSlangShaderToSpirv(SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPoint,
            .searchPath = kMeshletStreamShaderSearchPath,
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
    if (!result) {
        log += "compileSlangShaderToSpirv(";
        log += moduleName;
        log += ".";
        log += entryPoint;
        log += ") returned ";
        log += resultToString(result);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        log += '\n';
        return result;
    }

    const std::string shaderDebugName = std::string(moduleName) + "." + entryPoint;
    result = device.createShaderModule(ShaderModuleDesc{
        .spirv = compileResult.spirv,
        .debugName = shaderDebugName.c_str(),
    }).transform([&](auto rhiValue) { outShader = std::move(rhiValue); });
    if (!result || outShader == nullptr) {
        log += resultMessage("createShaderModule", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

} // namespace

class MeshletStreamRuntime::UpdatePass {
public:
    Result<> initialize(Device& device, ResourceRegistry& registry, uint64_t updateByteSize, uint32_t frameSlots, std::string& log,
        PipelineCache* pipelineCache)
    {
        if (updateByteSize == 0 || frameSlots == 0) {
            return makeError(Error::InvalidArgument);
        }
        Result<> result;
        updateBuffers_.resize(frameSlots);
        updateHandles_.resize(frameSlots);
        for (uint32_t slot = 0; slot < frameSlots; ++slot) {
            result = createHostStorageBuffer(device, updateByteSize, updateBuffers_[slot], log, "MeshletStreamRuntime update");
            if (!result) { return result; }
            result = allocateAndWriteBuffer(registry, *updateBuffers_[slot], updateHandles_[slot], log, "meshlet stream update");
            if (!result) { return result; }
        }

        device_ = &device;
        registry_ = &registry;
        auto initializeKernel = [&](const char* entry, ComputeKernel& kernel) -> Result<> {
            auto shader = compileSlangShaderToSpirv({
                .moduleName = kMeshletStreamShaderModuleName,
                .entryPointName = entry,
                .searchPath = kMeshletStreamShaderSearchPath,
            }, log);
            if (!shader) { return makeError(shader.error()); }
            return kernel.initialize(device, {
                .spirv = shader->spirv,
                .parameters = parameterAbi<MeshletStreamUserPush>(kPageTableABI, ParameterTransport::InlinePush),
                .debugName = entry,
                .pipelineCache = pipelineCache,
            }, log);
        };
        result = initializeKernel(kMeshletStreamPageTableInitEntryPoint, pageTableInitKernel_);
        if (!result) { return result; }
        result = initializeKernel(kMeshletStreamUpdateEntryPoint, updateKernel_);
        if (!result) { return result; }
        return {};
    }

    bool ready() const
    {
        return !updateBuffers_.empty() && updateBuffers_.front() != nullptr &&
            updateHandles_.front().valid() &&
            pageTableInitKernel_.valid() && updateKernel_.valid();
    }

    ResourceLease updateHandle() const { return updateHandles_.empty() ? ResourceLease{} : updateHandles_.front(); }

    Result<> initializePageTable(
        CommandBuffer& commandBuffer,
        MeshletStreamUserPush push,
        uint32_t pageCount,
        Buffer& pageTableBuffer,
        ResourceState& pageTableState)
    {
        if (!ready() || pageCount == 0) {
            return makeError(Error::Failure);
        }
        constexpr uint32_t kMaxDispatchGroupsPerDimension = 65535u;
        const uint32_t totalGroups = (pageCount - 1u) / 64u + 1u;
        const uint32_t groupCountX = std::min(totalGroups, kMaxDispatchGroupsPerDimension);
        const uint32_t groupCountY =
            (totalGroups + kMaxDispatchGroupsPerDimension - 1u) / kMaxDispatchGroupsPerDimension;

        push.activeBuildPhase = pageCount;
        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General); !commandResult) { return commandResult; }
        ParameterWriter writer(*device_, *registry_, RenderFrameContext::from(commandBuffer));
        auto parameters = writer.encode(push, kPageTableABI, ParameterTransport::InlinePush);
        if (!parameters) { return makeError(parameters.error()); }
        if (auto dispatched = pageTableInitKernel_.dispatch(commandBuffer, *parameters, groupCountX, groupCountY); !dispatched) { return dispatched; }
        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true); !commandResult) { return commandResult; }
        return {};
    }

    Result<> apply(
        CommandBuffer& commandBuffer,
        MeshletStreamUserPush push,
        std::span<const StreamPageTablePatch> patches,
        uint32_t maxUpdatePatches,
        uint32_t frameIndex,
        Buffer& pageTableBuffer,
        ResourceState& pageTableState)
    {
        if (patches.empty()) {
            return {};
        }
        if (!ready() || patches.size() > maxUpdatePatches) {
            return makeError(Error::Failure);
        }

        const uint32_t patchCount = static_cast<uint32_t>(patches.size());
        uint32_t unloadPatchCount = 0;
        for (const StreamPageTablePatch& patch : patches) {
            if (streamPageTablePatchState(patch) == MeshletStreamPageResidencyState::Unloaded) {
                ++unloadPatchCount;
            }
        }

        const uint32_t slot = metallic::render::RenderFrameContext::from(commandBuffer) ? metallic::render::RenderFrameContext::from(commandBuffer)->slotIndex() :
            frameIndex % static_cast<uint32_t>(updateBuffers_.size());
        if (slot >= updateBuffers_.size()) { return makeError(Error::InvalidArgument); }
        auto& updateBuffer = *updateBuffers_[slot];
        auto retained = commandBuffer.retainResource(updateBuffers_[slot]->retainAllocation());
        if (!retained) { return retained; }
        push.updateBuffer = updateHandles_[slot].shaderIndex();
        void* mapped = updateBuffer.map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        auto* header = static_cast<StreamUpdateBufferHeader*>(mapped);
        *header = StreamUpdateBufferHeader{
            .patchUnloadPageCount = unloadPatchCount,
            .patchPageCount = patchCount,
            .frameIndex = frameIndex,
        };
        auto* patchData = reinterpret_cast<StreamPageTablePatch*>(
            static_cast<uint8_t*>(mapped) + sizeof(StreamUpdateBufferHeader));
        uint32_t writeIndex = 0;
        for (const StreamPageTablePatch& patch : patches) {
            if (streamPageTablePatchState(patch) == MeshletStreamPageResidencyState::Unloaded) {
                patchData[writeIndex++] = patch;
            }
        }
        for (const StreamPageTablePatch& patch : patches) {
            if (streamPageTablePatchState(patch) != MeshletStreamPageResidencyState::Unloaded) {
                patchData[writeIndex++] = patch;
            }
        }
        updateBuffer.flush({0, sizeof(StreamUpdateBufferHeader) + static_cast<uint64_t>(patchCount) * sizeof(StreamPageTablePatch)});
        updateBuffer.unmap();

        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General); !commandResult) { return commandResult; }
        ParameterWriter writer(*device_, *registry_, RenderFrameContext::from(commandBuffer));
        if (auto used = writer.use(updateHandles_[slot]); !used) { return used; }
        auto parameters = writer.encode(push, kPageTableABI, ParameterTransport::InlinePush);
        if (!parameters) { return makeError(parameters.error()); }
        if (auto dispatched = updateKernel_.dispatch(commandBuffer, *parameters, (patchCount + 63u) / 64u, 1); !dispatched) { return dispatched; }
        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true); !commandResult) { return commandResult; }
        return {};
    }

private:
    std::vector<std::unique_ptr<Buffer>> updateBuffers_;
    // Same inline wire layout as the stream shaders; kernel packets retain executable state.
    static constexpr uint64_t kPageTableABI = 0x4d53504754420001ull;
    Device* device_ = nullptr;
    ResourceRegistry* registry_ = nullptr;
    ComputeKernel pageTableInitKernel_;
    ComputeKernel updateKernel_;
    std::vector<ResourceLease> updateHandles_;
};

class MeshletStreamRuntime::TraversalPass {
public:
    Result<> initialize(Device& device, std::string& log, PipelineCache* pipelineCache)
    {
        Result<> result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamTraversalEntryPoint,
            traversalShader_,
            log);
        if (!result) {
            return result;
        }

        result = device.createComputePipeline(ComputePipelineDesc{
            .computeShader = {traversalShader_.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            .pipelineCache = pipelineCache,
        }).transform([&](auto rhiValue) { traversalPipeline_ = std::move(rhiValue); });
        if (!result || traversalPipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime traversal)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    bool ready() const
    {
        return traversalShader_ != nullptr && traversalPipeline_ != nullptr;
    }

    Result<> dispatch(
        CommandBuffer& commandBuffer,
        BindlessHeap& bindlessHeap,
        const MeshletStreamUserPush& push,
        uint32_t threadCount,
        Buffer& pageTableBuffer,
        ResourceState& pageTableState,
        Buffer& requestBuffer,
        ResourceState& requestBufferState)
    {
        if (threadCount == 0) {
            return {};
        }
        if (!ready()) {
            return makeError(Error::Failure);
        }

        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.bindBindlessHeap(bindlessHeap); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.bindExecution((traversalPipeline_)->execution(), &push, sizeof(push)); !commandResult) { return commandResult; }
        const uint64_t groups = (uint64_t(threadCount) + 63u) / 64u;
        if (auto commandResult = commandBuffer.dispatch(static_cast<uint32_t>(std::min<uint64_t>(groups, 65535u)),
            static_cast<uint32_t>((groups + 65534u) / 65535u), 1); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        return {};
    }

private:
    std::unique_ptr<ShaderModule> traversalShader_;
    std::unique_ptr<ComputePipeline> traversalPipeline_;
};

class MeshletStreamRuntime::ActiveBuildPass {
public:
    Result<> initialize(Device& device, std::string& log, PipelineCache* pipelineCache)
    {
        profiling::CPUPhase phase("streamInit.activeShader");
        Result<> result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamActiveBuildEntryPoint,
            activeBuildShader_,
            log);
        if (!result) {
            return result;
        }

        phase.next("streamInit.activePipeline");
        result = device.createComputePipeline(ComputePipelineDesc{
            .computeShader = {activeBuildShader_.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            .pipelineCache = pipelineCache,
        }).transform([&](auto rhiValue) { activeBuildPipeline_ = std::move(rhiValue); });
        if (!result || activeBuildPipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime active build)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        phase.next("streamInit.cooperativeShader");
        result = createSlangShaderModule(device, kMeshletStreamShaderModuleName,
            kMeshletStreamCooperativeBuildEntryPoint, cooperativeShader_, log);
        if (!result) { return result; }
        phase.next("streamInit.cooperativePipeline");
        result = device.createComputePipeline({
            .computeShader = {cooperativeShader_.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            .pipelineCache = pipelineCache,
        }).transform([&](auto rhiValue) { cooperativePipeline_ = std::move(rhiValue); });
        if (!result) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime cooperative LOD)", result);
            return result;
        }
        result = createSlangShaderModule(device, kMeshletStreamShaderModuleName,
            kMeshletStreamDemandEntryPoint, demandShader_, log);
        if (!result) { return result; }
        result = device.createComputePipeline({
            .computeShader = {demandShader_.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            .pipelineCache = pipelineCache,
        }).transform([&](auto rhiValue) { demandPipeline_ = std::move(rhiValue); });
        if (!result) { return result; }
        phase.next("streamInit.lodCacheStatus");
        spdlog::info("[MeshletStreamRuntime] LOD PSO cache enabled={} activeHit={} cooperativeHit={}",
            pipelineCache != nullptr, activeBuildPipeline_->pipelineCacheHit(), cooperativePipeline_->pipelineCacheHit());
        return {};
    }

    bool ready() const
    {
        return activeBuildShader_ != nullptr && activeBuildPipeline_ != nullptr && cooperativePipeline_ != nullptr && demandPipeline_ != nullptr;
    }

    Result<> dispatch(
        CommandBuffer& commandBuffer,
        BindlessHeap& bindlessHeap,
        const MeshletStreamUserPush& push,
        uint32_t threadCount,
        Buffer& activeGroupBuffer,
        ResourceState& activeGroupBufferState,
        Buffer& activeHeaderBuffer,
        ResourceState& activeHeaderBufferState,
        Buffer& pageTableBuffer,
        ResourceState& pageTableState,
        Buffer& requestBuffer,
        ResourceState& requestBufferState,
        Buffer& drawIndirectBuffer,
        ResourceState& drawIndirectBufferState,
        Buffer& traversalHeaderBuffer,
        ResourceState& traversalHeaderBufferState,
        Buffer& traversalWorkBuffer,
        ResourceState& traversalWorkBufferState)
    {
        if (threadCount == 0) {
            return {};
        }
        if (!ready()) {
            return makeError(Error::Failure);
        }

        if (auto commandResult = transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, drawIndirectBuffer, drawIndirectBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, traversalHeaderBuffer, traversalHeaderBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, traversalWorkBuffer, traversalWorkBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.bindBindlessHeap(bindlessHeap); !commandResult) { return commandResult; }
        const bool cooperative = push.activeBuildPhase == kMeshletStreamActiveBuildFrontierPhase ||
            push.activeBuildPhase == kMeshletStreamActiveBuildEmitPhase ||
            push.activeBuildPhase == kMeshletStreamActiveBuildPrefetchPhase ||
            push.activeBuildPhase == kMeshletStreamActiveBuildClearPhase ||
            push.activeBuildPhase == kMeshletStreamActiveBuildMaskPhase;
        const bool demand = push.activeBuildPhase == kMeshletStreamActiveBuildDemandPhase ||
            push.activeBuildPhase == kMeshletStreamActiveBuildDemandResetPhase;
        if (auto commandResult = commandBuffer.bindExecution((demand ? *demandPipeline_ : cooperative ? *cooperativePipeline_ : *activeBuildPipeline_).execution(), &push, sizeof(push)); !commandResult) { return commandResult; }
        const uint32_t groups = cooperative ? threadCount : threadCount / 64u + (threadCount % 64u != 0u ? 1u : 0u);
        if (auto commandResult = commandBuffer.dispatch(std::min(groups, 65535u), (groups + 65534u) / 65535u, 1); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, drawIndirectBuffer, drawIndirectBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, traversalHeaderBuffer, traversalHeaderBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, traversalWorkBuffer, traversalWorkBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        return {};
    }

private:
    std::unique_ptr<ShaderModule> activeBuildShader_;
    std::unique_ptr<ComputePipeline> activeBuildPipeline_;
    std::unique_ptr<ShaderModule> cooperativeShader_;
    std::unique_ptr<ComputePipeline> cooperativePipeline_;
    std::unique_ptr<ShaderModule> demandShader_;
    std::unique_ptr<ComputePipeline> demandPipeline_;
};

class MeshletStreamRuntime::BLASInputPass {
public:
    Result<> initialize(Device& device, std::string& log, PipelineCache* pipelineCache)
    {
        Result<> result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamBLASInputEntryPoint,
            blasInputShader_,
            log);
        if (!result) {
            return result;
        }

        result = device.createComputePipeline(ComputePipelineDesc{
            .computeShader = {blasInputShader_.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            .pipelineCache = pipelineCache,
        }).transform([&](auto rhiValue) { blasInputPipeline_ = std::move(rhiValue); });
        if (!result || blasInputPipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime BLAS input)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    bool ready() const
    {
        return blasInputShader_ != nullptr && blasInputPipeline_ != nullptr;
    }

    Result<> dispatch(
        CommandBuffer& commandBuffer,
        BindlessHeap& bindlessHeap,
        const MeshletStreamUserPush& push,
        uint32_t threadCount,
        Buffer& activeGroupBuffer,
        ResourceState& activeGroupBufferState,
        Buffer& activeHeaderBuffer,
        ResourceState& activeHeaderBufferState,
        Buffer& blasHeaderBuffer,
        ResourceState& blasHeaderBufferState,
        Buffer& instanceBlasBuffer,
        ResourceState& instanceBlasBufferState,
        Buffer& blasBuildInfoBuffer,
        ResourceState& blasBuildInfoBufferState,
        Buffer& blasClusterReferenceBuffer,
        ResourceState& blasClusterReferenceBufferState)
    {
        if (threadCount == 0) {
            return {};
        }
        if (!ready()) {
            return makeError(Error::Failure);
        }

        if (auto commandResult = transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, blasHeaderBuffer, blasHeaderBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, instanceBlasBuffer, instanceBlasBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, blasBuildInfoBuffer, blasBuildInfoBufferState, ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(
            commandBuffer,
            blasClusterReferenceBuffer,
            blasClusterReferenceBufferState,
            ResourceState::General); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.bindBindlessHeap(bindlessHeap); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.bindExecution((blasInputPipeline_)->execution(), &push, sizeof(push)); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.dispatch((threadCount + 63u) / 64u, 1, 1); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, blasHeaderBuffer, blasHeaderBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, instanceBlasBuffer, instanceBlasBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(commandBuffer, blasBuildInfoBuffer, blasBuildInfoBufferState, ResourceState::General, true); !commandResult) { return commandResult; }
        if (auto commandResult = transitionBuffer(
            commandBuffer,
            blasClusterReferenceBuffer,
            blasClusterReferenceBufferState,
            ResourceState::General,
            true); !commandResult) { return commandResult; }
        return {};
    }

private:
    std::unique_ptr<ShaderModule> blasInputShader_;
    std::unique_ptr<ComputePipeline> blasInputPipeline_;
};

class MeshletStreamRuntime::TLASInputPass {
public:
    Result<> initialize(Device& device, std::string& log, PipelineCache* pipelineCache)
    {
        Result<> result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamTLASInputEntryPoint,
            tlasInputShader_,
            log);
        if (!result) {
            return result;
        }
        result = device.createComputePipeline(ComputePipelineDesc{
            .computeShader = {tlasInputShader_.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            .pipelineCache = pipelineCache,
        }).transform([&](auto rhiValue) { tlasInputPipeline_ = std::move(rhiValue); });
        if (!result || tlasInputPipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime TLAS input)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    bool ready() const
    {
        return tlasInputShader_ != nullptr && tlasInputPipeline_ != nullptr;
    }

    Result<> dispatch(
        CommandBuffer& commandBuffer,
        BindlessHeap& bindlessHeap,
        const MeshletStreamUserPush& push,
        uint32_t threadCount,
        Buffer& instanceBlasBuffer,
        ResourceState& instanceBlasBufferState,
        Buffer& tlasInstanceBuffer,
        ResourceState& tlasInstanceBufferState)
    {
        if (threadCount == 0) {
            return {};
        }
        if (!ready()) {
            return makeError(Error::Failure);
        }
        if (auto commandResult = commandBuffer.bindBindlessHeap(bindlessHeap); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.bindExecution((tlasInputPipeline_)->execution(), &push, sizeof(push)); !commandResult) { return commandResult; }
        if (auto commandResult = commandBuffer.dispatch((threadCount + 63u) / 64u, 1, 1); !commandResult) { return commandResult; }
        return {};
    }

private:
    std::unique_ptr<ShaderModule> tlasInputShader_;
    std::unique_ptr<ComputePipeline> tlasInputPipeline_;
};

MeshletStreamRuntime::MeshletStreamRuntime() = default;
MeshletStreamRuntime::~MeshletStreamRuntime()
{
    reset();
}

Result<> MeshletStreamRuntime::initialize(Device& device, const MeshletStreamRuntimeDesc& desc, std::string& log,
    PipelineCache* pipelineCache)
{
    profiling::CPUPhase phase("streamInit.reset");
    reset();
    log.clear();

    phase.next("streamInit.openAsset");
    std::string reason;
    scene::MeshletStreamAsset openedAsset;
    if (!openedAsset.open(desc.streamAssetPath, reason) ||
        !openedAsset.isRuntimeCompatibleForSource(desc.sourcePath, reason)) {
        log = "MeshletStreamRuntime cannot use streamasset '" + desc.streamAssetPath.string() +
            "' for source '" + desc.sourcePath.string() + "': " + reason +
            "; run MetallicMeshletCook --source <source> --output <streamasset> to rebuild";
        if (desc.autoBuildStreamAsset) {
            log += "; runtime auto-build is no longer supported";
        }
        return makeError(Error::Failure);
    }

    phase.next("streamInit.boundsAndBudget");
    asset_ = std::move(openedAsset);
    if (desc.compactShadingAttributes && !asset_.compactShadingForDevice(reason)) {
        log = "MeshletStreamRuntime compact shading: " + reason;
        return makeError(Error::InvalidArgument);
    }
    drawBounds_ = computeDrawBounds(asset_);
    if (!drawBounds_.valid) {
        log = "MeshletStreamRuntime streamasset bounds are unavailable";
        return makeError(Error::Failure);
    }

    coldPageRetentionFrames_ = desc.coldPageRetentionFrames;
    deviceImmutableMetadata_ = desc.deviceImmutableMetadata;
    if (deviceImmutableMetadata_) {
        immutableMetadataUpload_ = std::make_shared<ImmutableMetadataUpload>();
    }
    const bool enableClas = desc.enableClas && device.capabilities().clusterAccelerationStructure;
    clusterRtxEnabled_ = desc.enableClusterRtx;
    maxResidentPages_ = desc.maxResidentPages;
    maxPageUploadsPerFrame_ = desc.maxPageUploadsPerFrame;
    maxUploadBytesPerFrame_ = desc.maxUploadBytesPerFrame;
    maxGpuPageRequests_ = std::max(desc.maxGpuPageRequests, 1u);
    screenSpacePagePriority_ = desc.screenSpacePagePriority;
    viewDrivenPageDemand_ = desc.viewDrivenPageDemand;
    distributedPageDemand_ = desc.distributedPageDemand;
    distributedDemandMinGroups_ = desc.distributedDemandMinGroups;
    prefetchPages_ = desc.prefetchPages && desc.viewDrivenPageDemand && desc.screenSpacePagePriority &&
        asset_.pageCount() < kStreamPrefetchPageTag;
    predictivePrefetch_ = desc.predictivePrefetch;
    lodTransitionTelemetry_ = desc.enableLodTransitionTelemetry;
    maxGpuPageUnloadRequests_ = std::max(desc.maxGpuPageUnloadRequests, 1u);
    const uint64_t pageStride = alignUp(asset_.maxPagePayloadBytes(), 256);
    maxResidentBytes_ = desc.maxResidentBytes;
    if (maxResidentBytes_ == 0) {
        if (maxResidentPages_ == 0) {
            log = "MeshletStreamRuntime requires maxResidentBytes or maxResidentPages to be greater than zero";
            return makeError(Error::Failure);
        }
        if (pageStride == 0 ||
            pageStride > std::numeric_limits<uint64_t>::max() / maxResidentPages_) {
            log = "MeshletStreamRuntime resident byte budget overflowed";
            return makeError(Error::Failure);
        }
        maxResidentBytes_ = pageStride * maxResidentPages_;
    }
    if (maxResidentBytes_ == 0 || maxResidentBytes_ > std::numeric_limits<uint32_t>::max()) {
        log = "MeshletStreamRuntime resident page buffer size overflowed";
        return makeError(Error::Failure);
    }
    if (asset_.maxPagePayloadBytes() > maxResidentBytes_) {
        log = "MeshletStreamRuntime resident budget cannot hold the largest page payload";
        return makeError(Error::Failure);
    }
    maxResidentBytes_ = alignUp(maxResidentBytes_, kMeshletStreamStorageAlignment);
    if (maxResidentBytes_ > std::numeric_limits<uint32_t>::max()) {
        log = "MeshletStreamRuntime resident page buffer aligned size overflowed";
        return makeError(Error::Failure);
    }

    const bool gpuDecompression = desc.enableGpuDecompression && desc.completionDrivenUploads &&
        device.capabilities().memoryDecompression;
    BufferUsageBits pageBufferUsage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination;
    if (gpuDecompression) { pageBufferUsage = pageBufferUsage | BufferUsageBits::MemoryDecompression; }
    if (desc.enableClusterRtx || enableClas) {
        if (!device.capabilities().clusterAccelerationStructure) {
            log = "MeshletStreamRuntime cluster RTX requires cluster acceleration structure support";
            return makeError(Error::Unsupported);
        }
        pageBufferUsage = pageBufferUsage |
            BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::AccelerationStructureBuildInput;
    }

    phase.next("streamInit.pagePool", maxResidentBytes_);
    Result<> result = device.createBuffer(BufferDesc{
            .size = maxResidentBytes_,
            .structureStride = 0,
            .usage = pageBufferUsage,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
            .memoryDomain = MemoryBudgetDomain::Geometry,
        }).transform([&](auto rhiValue) { pageBuffer_ = std::move(rhiValue); });
    if (!result || pageBuffer_ == nullptr) {
        log += resultMessage("createBuffer(MeshletStreamRuntime pages)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    pageBufferState_ = ResourceState::Undefined;

    phase.next("streamInit.residency");
    for (const auto& page : asset_.pages()) {
        maxDevicePageBytes_ = std::max(maxDevicePageBytes_, uint64_t(scene::meshletStreamDevicePayloadSize(page)));
    }
    adaptivePageRetention_ = desc.adaptivePageRetention && coldPageRetentionFrames_ != 0;
    if (adaptivePageRetention_) {
        // Eight upload batches provide bounded turnover space, instead of
        // discarding 30% of a useful cache. Half remains reserved for demand;
        // speculative pages may use only the surplus after that reserve.
        const uint64_t uploadBudget = std::min(maxResidentBytes_,
            maxUploadBytesPerFrame_ != 0 ? maxUploadBytesPerFrame_ : 8ull * 1024ull * 1024ull);
        geometryReclaimReserveBytes_ = std::min(maxResidentBytes_ / 8u,
            std::max(uploadBudget * 8u, maxDevicePageBytes_ * 2u));
        geometryReclaimReserveBytes_ -= geometryReclaimReserveBytes_ % kMeshletStreamStorageAlignment;
        geometryDemandReserveBytes_ = geometryReclaimReserveBytes_ / 2u;
    }
    if (!residency_.initialize(
            MeshletStreamResidencyDesc{
                .asset = &asset_,
                .maxResidentBytes = maxResidentBytes_,
                .maxResidentPages = maxResidentPages_,
                .queuedFrameCount = std::max(desc.queuedFrameCount, 1u),
                .pageStride = pageStride,
                .pageLoadConcurrency = desc.pageLoadConcurrency,
                .maxPageLoadsInFlight = desc.maxPageLoadsInFlight,
                .measurePageLatency = desc.measurePageLatency,
                .immediateGpuRequests = desc.lowLatencyRequests,
                .completionDrivenUploads = desc.completionDrivenUploads,
                .gpuDecompression = gpuDecompression,
                .gpuDecompressionMinBatchBytes = desc.gpuDecompressionMinBatchBytes,
                .prefetchReserveBytes = geometryDemandReserveBytes_,
                .maxPageEvictionsPerFrame = adaptivePageRetention_ ? 1024u : 256u,
                .maxEvictionBytesPerFrame = adaptivePageRetention_
                    ? std::max(geometryReclaimReserveBytes_, alignUp(maxDevicePageBytes_, kMeshletStreamStorageAlignment)) : 0u,
            },
            reason)) {
        log = "MeshletStreamRuntime residency initialization failed: " + reason;
        return makeError(Error::Failure);
    }

    phase.next("streamInit.terminalCutAndCapacity");
    const MeshletStreamStorage& residencyStorage = residency_.storage();
    // Every terminal branch belongs to the base cut, including branches
    // that stopped simplifying before the primitive's coarsest level.
    // Reserve hidden instances too, so later visibility changes remain safe.
    std::vector<uint32_t> fallbackPages;
    std::vector<uint32_t> lockedFallbackPrimitives;
    std::unordered_set<uint32_t> consideredFallbackPrimitives;
    uint64_t terminalInstanceCount = 0;
    for (const auto& instance : asset_.instances()) {
        if (instance.primitiveIndex >= asset_.primitiveCount()) {
            continue;
        }
        terminalInstanceCount += asset_.primitiveTerminalGroups(instance.primitiveIndex).size();
        if (!consideredFallbackPrimitives.insert(instance.primitiveIndex).second) {
            continue;
        }
        lockedFallbackPrimitives.push_back(instance.primitiveIndex);
        for (uint32_t groupIndex : asset_.primitiveTerminalGroups(instance.primitiveIndex)) {
            fallbackPages.push_back(asset_.groups()[groupIndex].pageIndex);
        }
    }
    std::sort(fallbackPages.begin(), fallbackPages.end());
    fallbackPages.erase(std::unique(fallbackPages.begin(), fallbackPages.end()), fallbackPages.end());
    uint64_t lockedFallbackBytes = 0;
    for (uint32_t pageIndex : fallbackPages) {
        lockedFallbackBytes += residencyStorage.allocationSize(scene::meshletStreamDevicePayloadSize(asset_.pages()[pageIndex]));
    }
    const bool needsStreamingReserve = fallbackPages.size() < asset_.pageCount();
    const uint64_t streamedPageReserveBytes = needsStreamingReserve
        ? residencyStorage.allocationSize(asset_.maxPagePayloadBytes()) : 0;
    if (fallbackPages.size() > desc.maxLockedFallbackPages ||
        (maxResidentPages_ != 0 && fallbackPages.size() + (needsStreamingReserve ? 1u : 0u) > maxResidentPages_) ||
        lockedFallbackBytes + streamedPageReserveBytes > residencyStorage.capacityBytes() ||
        terminalInstanceCount > desc.maxActiveGroups) {
        log = "MeshletStreamRuntime budget cannot hold the complete terminal LOD cut (" +
            std::to_string(fallbackPages.size()) + " pages, " +
            std::to_string(lockedFallbackBytes) + " bytes, " +
            std::to_string(terminalInstanceCount) + " instance groups) plus a streaming page";
        return makeError(Error::InvalidArgument);
    }
    if (!residency_.lockFallbackPages(fallbackPages, reason)) {
        log = "MeshletStreamRuntime fallback residency initialization failed: " + reason;
        return makeError(Error::Failure);
    }
    lockedFallbackPages_ = std::move(fallbackPages);

    maxActiveGroups_ = computeMaxActiveGroups(desc.maxActiveGroups);
    maxActiveGroupClusters_ = asset_.maxPageClusters();
    maxPrimitiveGroupCount_ = computeMaxPrimitiveGroups();
    const uint32_t requestedTraversalWorkers = std::min(
        std::max(desc.maxTraversalWorkers, 1u),
        kMeshletStreamMaxTraversalWorkers);
    const uint32_t activeTraversalWorkers = std::min(asset_.instanceCount(), requestedTraversalWorkers);
    traversalWorkerCount_ = ((activeTraversalWorkers + 63u) / 64u) * 64u;
    traversalWorkCapacity_ = std::min(
        std::max(desc.maxTraversalWorkItems, 1u),
        kMeshletStreamMaxTraversalWorkItems);
    if (maxActiveGroupClusters_ > kMeshletStreamMaxActiveGroupClusters) {
        log = "MeshletStreamRuntime group exceeds the 32-cluster selection mask capacity";
        return makeError(Error::Failure);
    }

    if (desc.enableClusterRtx || enableClas) {
        const uint64_t defaultBuildClusters =
            static_cast<uint64_t>(std::max(maxPageUploadsPerFrame_, 1u)) * asset_.maxPageClusters();
        const uint64_t buildClusters = desc.maxClasBuildClusters != 0
            ? desc.maxClasBuildClusters
            : defaultBuildClusters;
        if (desc.maxClasBytes == 0 ||
            buildClusters < asset_.maxPageClusters() ||
            buildClusters > std::numeric_limits<uint32_t>::max()) {
            log = "MeshletStreamRuntime cluster RTX capacities are invalid";
            return makeError(Error::InvalidArgument);
        }
        maxClasBuildClusters_ = static_cast<uint32_t>(buildClusters);
        clasPool_ = std::make_unique<MeshletStreamCLASPool>();
        result = clasPool_->initialize(
            device,
            MeshletStreamCLASPoolDesc{
                .asset = &asset_,
                .maxStorageBytes = desc.maxClasBytes,
                .maxBuildClusters = static_cast<uint32_t>(buildClusters),
                .queuedFrameCount = std::max(desc.queuedFrameCount, 1u),
                .compactStorage = desc.compactClas,
                .startStorageBytes = desc.startClasBytes,
                .growStorageBytes = desc.growClasBytes,
                .emptyChunkRetentionFrames = desc.clasEmptyChunkRetentionFrames,
                .persistentGrowStorageBytes = desc.persistentClasGrowBytes,
                .persistentPages = lockedFallbackPages_,
            },
            log);
        if (!result) {
            log = "MeshletStreamRuntime CLAS pool initialization failed: " + log;
            return result;
        }
        if (clusterRtxEnabled_) {
            clasPool_->setInvalidationObserver([cache = sceneReadinessCache_, roots = lockedFallbackPages_](
                std::span<const uint32_t> pages) {
                if (pages.empty() || std::any_of(pages.begin(), pages.end(), [&](uint32_t page) {
                        return std::binary_search(roots.begin(), roots.end(), page);
                    })) {
                    // Root references in fallback BLAS cannot survive a destructive
                    // pool edit. Require a runtime rebuild, even if CLAS reappears.
                    cache->rootsInvalidated = true;
                    cache->value.ready = false;
                    cache->value.completedPages = 0;
                    cache->valid = true;
                }
            });
        }
    }
    if (maxActiveGroups_ == 0 ||
        maxActiveGroupClusters_ == 0 ||
        maxPrimitiveGroupCount_ == 0 ||
        traversalWorkerCount_ == 0) {
        log = "MeshletStreamRuntime streamasset has no drawable active groups";
        return makeError(Error::Failure);
    }
    const uint64_t visibleRecordCapacity =
        static_cast<uint64_t>(maxActiveGroups_) * maxActiveGroupClusters_;
    if (!visibilityRecordCapacityFitsId(visibleRecordCapacity)) {
        log = "MeshletStreamRuntime visible cluster capacity exceeds the common visibility ID record limit";
        return makeError(Error::Failure);
    }
    if (visibleRecordCapacity * kMeshletStreamTriangleChunkCount >
        std::numeric_limits<uint32_t>::max()) {
        log = "MeshletStreamRuntime active group draw task count overflowed";
        return makeError(Error::Failure);
    }
    rasterCandidateCapacity_ = desc.maxRasterCandidates == 0 ? visibleClusterCapacity() :
        std::min(visibleClusterCapacity(), std::max(maxActiveGroups_, desc.maxRasterCandidates));
    if (desc.enableClusterRtx) {
        const uint32_t activeClusterCapacity = visibleClusterCapacity();
        blasClusterReferenceCapacity_ = desc.maxBlasClusterReferences == 0
            ? activeClusterCapacity
            : std::min(desc.maxBlasClusterReferences, activeClusterCapacity);
        if (blasClusterReferenceCapacity_ == 0) {
            log = "MeshletStreamRuntime cluster RTX requires a non-zero BLAS cluster reference capacity";
            return makeError(Error::InvalidArgument);
        }
    }

    uint64_t residentPageCapacity = std::min<uint64_t>(
        asset_.pageCount(),
        maxResidentBytes_ / kMeshletStreamStorageAlignment);
    if (maxResidentPages_ != 0) {
        residentPageCapacity = std::min<uint64_t>(residentPageCapacity, maxResidentPages_);
    }
    residentPageCapacity_ = static_cast<uint32_t>(residentPageCapacity);
    const uint64_t maxUpdatePatches64 = std::max(residentPageCapacity * 2ull, 1ull);
    if (maxUpdatePatches64 > std::numeric_limits<uint32_t>::max()) {
        log = "MeshletStreamRuntime update patch capacity overflowed";
        return makeError(Error::Failure);
    }
    maxUpdatePatches_ = static_cast<uint32_t>(maxUpdatePatches64);

    const uint64_t pageTableByteSize =
        static_cast<uint64_t>(asset_.pageCount()) * sizeof(StreamPageTableEntry);
    const uint64_t requestReadbackByteSize =
        sizeof(StreamRequestBufferHeader) +
        (static_cast<uint64_t>(maxGpuPageRequests_) * (screenSpacePagePriority_ ? 2u : 1u) +
            maxGpuPageUnloadRequests_) * sizeof(uint32_t);
    const uint64_t requestByteSize = requestReadbackByteSize +
        (screenSpacePagePriority_ ? uint64_t(asset_.pageCount()) * sizeof(uint32_t) : 0u);
    if (requestByteSize / sizeof(uint32_t) > UINT32_MAX) {
        log += "Stream page priority buffer exceeds 32-bit word addressing\n";
        return makeError(Error::InvalidArgument);
    }
    const uint64_t updateByteSize =
        sizeof(StreamUpdateBufferHeader) + static_cast<uint64_t>(maxUpdatePatches_) * sizeof(StreamPageTablePatch);

    phase.next("streamInit.sceneMetadata");
    result = initializeSceneMetadataBuffers(device, log);
    if (!result) {
        return result;
    }

    phase.next("streamInit.buffersAndDescriptors");
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = static_cast<uint64_t>(maxActiveGroups_) * sizeof(MeshletStreamGPUActiveGroup),
            .structureStride = sizeof(MeshletStreamGPUActiveGroup),
            .usage = debugReadbackEnabled_ ? BufferUsageBits::Storage | BufferUsageBits::TransferSource : BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        },
        activeGroupBuffer_,
        log,
        "MeshletStreamRuntime active groups");
    if (!result) {
        return result;
    }
    activeGroupBufferState_ = ResourceState::Undefined;
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = sizeof(MeshletStreamGPUActiveHeader),
            .structureStride = sizeof(MeshletStreamGPUActiveHeader),
            .usage = debugReadbackEnabled_ ? BufferUsageBits::Storage | BufferUsageBits::TransferSource : BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        },
        activeHeaderBuffer_,
        log,
        "MeshletStreamRuntime active header");
    if (!result) {
        return result;
    }
    activeHeaderBufferState_ = ResourceState::Undefined;
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = kMeshletStreamDrawIndirectCommandCount * sizeof(MeshletStreamGPUDrawIndirect),
            .structureStride = sizeof(MeshletStreamGPUDrawIndirect),
            .usage = BufferUsageBits::Storage | BufferUsageBits::Indirect,
            .memoryLocation = MemoryLocation::Device,
        },
        drawIndirectBuffer_,
        log,
        "MeshletStreamRuntime draw indirect");
    if (!result) {
        return result;
    }
    drawIndirectBufferState_ = ResourceState::Undefined;
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = sizeof(MeshletStreamGPUTraversalHeader),
            .structureStride = sizeof(MeshletStreamGPUTraversalHeader),
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        },
        traversalHeaderBuffer_,
        log,
        "MeshletStreamRuntime traversal header");
    if (!result) {
        return result;
    }
    traversalHeaderBufferState_ = ResourceState::Undefined;
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = static_cast<uint64_t>(traversalWorkCapacity_) * sizeof(MeshletStreamGPUTraversalWorkItem),
            .structureStride = sizeof(MeshletStreamGPUTraversalWorkItem),
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        },
        traversalWorkBuffer_,
        log,
        "MeshletStreamRuntime traversal work");
    if (!result) {
        return result;
    }
    traversalWorkBufferState_ = ResourceState::Undefined;
    if (desc.enableClusterRtx) {
        ClusterAccelerationStructureProperties clusterProperties;
        result = device.queryClusterAccelerationStructureProperties().transform([&](auto rhiValue) { clusterProperties = std::move(rhiValue); });
        if (!result) {
            log = std::string("queryClusterAccelerationStructureProperties(stream BLAS) returned ") +
                resultToString(result);
            return result;
        }
        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = sizeof(MeshletStreamGPUBLASHeader) +
                    (uint64_t(maxActiveGroups_) + (uint64_t(asset_.instanceCount()) + 63u) / 64u) * 16u,
                .structureStride = sizeof(MeshletStreamGPUBLASHeader),
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                    BufferUsageBits::Indirect |
                    BufferUsageBits::AccelerationStructureBuildInput |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            blasHeaderBuffer_,
            log,
            "MeshletStreamRuntime BLAS header");
        if (!result) {
            return result;
        }
        blasHeaderBufferState_ = ResourceState::Undefined;

        // The per-build upper bound is conservative; references themselves are
        // packed from the selected cut on the GPU, never reserved per instance.
        uint32_t maximumPrimitiveClusters = 0;
        for (const auto& primitive : asset_.primitives()) {
            uint64_t count = 0;
            for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                count += asset_.groups()[primitive.groupOffset + local].clusterCount;
            }
            maximumPrimitiveClusters = std::max(maximumPrimitiveClusters,
                static_cast<uint32_t>(std::min<uint64_t>(count, blasClusterReferenceCapacity_)));
        }
        const uint32_t visibleInstances = static_cast<uint32_t>(std::count_if(
            asset_.instances().begin(), asset_.instances().end(), [this](const auto& instance) {
                return instance.visible != 0 && instance.primitiveIndex < asset_.primitiveCount();
            }));
        if (desc.maxBlasBytes == 0 || desc.maxBlasBuilds == 0 ||
            visibleInstances == 0 || maximumPrimitiveClusters == 0) {
            log = "MeshletStreamRuntime cluster RTX requires non-zero BLAS budgets and geometry";
            return makeError(Error::InvalidArgument);
        }
        uint32_t referenceLimit = blasClusterReferenceCapacity_;
        const uint32_t buildLimit = std::min(desc.maxBlasBuilds, visibleInstances);
        ClusterAccelerationStructureBuildSizes blasSizes;
        for (;;) {
            // Every admitted build has at least one reference. Preserve instance
            // coverage while reducing reference storage to fit the byte budget.
            const uint32_t buildCount = std::min(buildLimit, referenceLimit);
            const uint32_t perBuild = std::min(maximumPrimitiveClusters, referenceLimit);
            result = device.queryClusterAccelerationStructureBottomLevelBuildSizes(
                ClusterAccelerationStructureBottomLevelBuildSizesDesc{
                    .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                    .maxClusterCountPerAccelerationStructure = perBuild,
                    .maxTotalClusterCount = referenceLimit,
                    .maxAccelerationStructureCount = buildCount,
                }).transform([&](auto rhiValue) { blasSizes = std::move(rhiValue); });
            if (!result || blasSizes.accelerationStructureSize == 0 || blasSizes.buildScratchSize == 0) {
                log = std::string("queryClusterAccelerationStructureBottomLevelBuildSizes(stream BLAS) returned ") +
                    resultToString(result);
                return result ? makeError(Error::Failure) : result;
            }
            if (blasSizes.accelerationStructureSize <= desc.maxBlasBytes) {
                blasClusterReferenceCapacity_ = referenceLimit;
                blasBuildCapacity_ = buildCount;
                maxBlasClustersPerBuild_ = perBuild;
                break;
            }
            if (referenceLimit == 1) {
                log = "MeshletStreamRuntime maxBlasBytes cannot hold one dynamic cluster BLAS";
                return makeError(Error::OutOfMemory);
            }
            referenceLimit = std::max(referenceLimit / 2u, 1u);
        }
        // Explicit destinations reserve a queried, aligned size class per
        // instance. Growth appends; exhaustion triggers a bounded GPU repack.
        for (uint32_t bucket = 0; bucket < blasSizeClasses_.size(); ++bucket) {
            const uint32_t clusters = static_cast<uint32_t>(std::min<uint64_t>(
                uint64_t(1) << bucket, maxBlasClustersPerBuild_));
            ClusterAccelerationStructureBuildSizes single;
            result = device.queryClusterAccelerationStructureBottomLevelBuildSizes({
                .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                .maxClusterCountPerAccelerationStructure = clusters,
                .maxTotalClusterCount = clusters,
                .maxAccelerationStructureCount = 1,
            }).transform([&](auto value) { single = value; });
            if (!result) { return result; }
            const uint64_t size = alignUp(single.accelerationStructureSize, clusterProperties.bottomLevelStorageAlignment);
            if (size == 0 || size > UINT32_MAX) { return makeError(Error::OutOfMemory); }
            blasSizeClasses_[bucket] = static_cast<uint32_t>(size);
        }
        blasInstanceReuse_ = true;
        if (const char* value = std::getenv("METALLIC_BLAS_INSTANCE_REUSE")) {
            blasInstanceReuse_ = std::strcmp(value, "0") != 0;
        }
        if (blasSizes.accelerationStructureSize > UINT32_MAX) { return makeError(Error::OutOfMemory); }
        spdlog::info("[MeshletStreamRuntime] BLAS instance reuse={} storageBytes={} referenceCapacity={} buildCapacity={}",
            blasInstanceReuse_, blasSizes.accelerationStructureSize, blasClusterReferenceCapacity_, blasBuildCapacity_);
        result = createAndPopulateHostStorageBuffer<MeshletStreamGPUInstanceBLAS>(
            device, asset_.instanceCount(), instanceBlasBuffer_, log,
            "MeshletStreamRuntime instance BLAS inputs",
            [](MeshletStreamGPUInstanceBLAS& instanceBlas, size_t) {
                instanceBlas = MeshletStreamGPUInstanceBLAS{};
            });
        if (!result) { return result; }
        instanceBlasBufferState_ = ResourceState::Undefined;

        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = std::max<uint64_t>(
                    static_cast<uint64_t>(blasBuildCapacity_) * sizeof(MeshletStreamGPUBLASBuildInfo),
                    sizeof(MeshletStreamGPUBLASBuildInfo)),
                .structureStride = sizeof(MeshletStreamGPUBLASBuildInfo),
                .usage = BufferUsageBits::Storage |
                    BufferUsageBits::AccelerationStructureBuildInput |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            blasBuildInfoBuffer_,
            log,
            "MeshletStreamRuntime BLAS build infos");
        if (!result) {
            return result;
        }
        blasBuildInfoBufferState_ = ResourceState::Undefined;

        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = static_cast<uint64_t>(blasClusterReferenceCapacity_) * sizeof(uint64_t),
                .structureStride = sizeof(uint64_t),
                .usage = BufferUsageBits::Storage |
                    BufferUsageBits::AccelerationStructureBuildInput |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            blasClusterReferenceBuffer_,
            log,
            "MeshletStreamRuntime BLAS cluster references");
        if (!result) {
            return result;
        }
        blasClusterReferenceBufferState_ = ResourceState::Undefined;
        blasClusterReferenceAddress_ = blasClusterReferenceBuffer_->deviceAddress();
        if (blasClusterReferenceAddress_ == 0) {
            log = "MeshletStreamRuntime BLAS cluster reference buffer has no device address";
            return makeError(Error::Failure);
        }

        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = blasSizes.accelerationStructureSize,
                .usage = BufferUsageBits::AccelerationStructureStorage |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            blasStorageBuffer_,
            log,
            "MeshletStreamRuntime dynamic BLAS storage");
        if (!result) {
            return result;
        }
        const uint64_t scratchAlignment = clusterProperties.scratchAlignment;
        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = blasSizes.buildScratchSize + scratchAlignment - 1u,
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            blasScratchBuffer_,
            log,
            "MeshletStreamRuntime dynamic BLAS scratch");
        if (!result) {
            return result;
        }
        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = static_cast<uint64_t>(blasBuildCapacity_) * sizeof(uint64_t),
                .structureStride = sizeof(uint64_t),
                .usage = BufferUsageBits::Storage |
                    BufferUsageBits::AccelerationStructureStorage |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            blasAddressBuffer_,
            log,
            "MeshletStreamRuntime dynamic BLAS addresses");
        if (!result) {
            return result;
        }
        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = static_cast<uint64_t>(blasBuildCapacity_) * sizeof(uint32_t),
                .structureStride = sizeof(uint32_t),
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            blasSizeBuffer_,
            log,
            "MeshletStreamRuntime dynamic BLAS sizes");
        if (!result) {
            return result;
        }

        if (blasStorageBuffer_->deviceAddress() == 0 ||
            blasScratchBuffer_->deviceAddress() == 0 ||
            blasAddressBuffer_->deviceAddress() == 0 ||
            blasSizeBuffer_->deviceAddress() == 0) {
            log = "MeshletStreamRuntime dynamic BLAS buffers have no device addresses";
            return makeError(Error::Failure);
        }

        if (desc.maxFallbackBlasBytes == 0) {
            log = "MeshletStreamRuntime fallback BLAS budget must be non-zero";
            return makeError(Error::InvalidArgument);
        }
        if (lockedFallbackPrimitives.empty()) {
            log = "MeshletStreamRuntime cluster RTX requires at least one locked fallback primitive";
            return makeError(Error::OutOfMemory);
        }

        const uint64_t bottomLevelAlignment = clusterProperties.bottomLevelStorageAlignment;
        if (bottomLevelAlignment == 0) {
            log = "MeshletStreamRuntime fallback BLAS alignment is unavailable";
            return makeError(Error::Failure);
        }
        std::unordered_map<uint32_t, ClusterAccelerationStructureBuildSizes> sizeCache;
        uint64_t totalFallbackReferences = 0;
        uint64_t fallbackStorageBytes = 0;
        uint64_t fallbackScratchBytes = 0;
        fallbackBlasPrimitives_.reserve(lockedFallbackPrimitives.size());
        for (uint32_t primitiveIndex : lockedFallbackPrimitives) {
            uint64_t primitiveReferences = 0;
            for (uint32_t groupIndex : asset_.primitiveTerminalGroups(primitiveIndex)) {
                const uint32_t pageIndex = asset_.groups()[groupIndex].pageIndex;
                const uint32_t clusterCount = asset_.pages()[pageIndex].clusterCount;
                if (clusterCount > std::numeric_limits<uint64_t>::max() - primitiveReferences) {
                    log = "MeshletStreamRuntime primitive fallback CLAS reference count overflowed";
                    return makeError(Error::InvalidArgument);
                }
                primitiveReferences += clusterCount;
            }
            if (primitiveReferences == 0 || primitiveReferences > std::numeric_limits<uint32_t>::max()) {
                log = "MeshletStreamRuntime primitive fallback CLAS reference count overflowed";
                return makeError(Error::InvalidArgument);
            }
            const uint32_t clusterCount = static_cast<uint32_t>(primitiveReferences);
            auto [iter, inserted] = sizeCache.try_emplace(clusterCount);
            if (inserted) {
                result = device.queryClusterAccelerationStructureBottomLevelBuildSizes(ClusterAccelerationStructureBottomLevelBuildSizesDesc{
                        .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                        .maxClusterCountPerAccelerationStructure = clusterCount,
                        .maxTotalClusterCount = clusterCount,
                        .maxAccelerationStructureCount = 1,
                    }).transform([&](auto rhiValue) { iter->second = std::move(rhiValue); });
                if (!result ||
                    iter->second.accelerationStructureSize == 0 ||
                    iter->second.buildScratchSize == 0) {
                    log = std::string("queryClusterAccelerationStructureBottomLevelBuildSizes(fallback BLAS) returned ") +
                        resultToString(result);
                    return result ? makeError(Error::Failure) : result;
                }
            }
            if (fallbackStorageBytes >
                std::numeric_limits<uint64_t>::max() - (bottomLevelAlignment - 1u)) {
                log = "MeshletStreamRuntime fallback BLAS storage alignment overflowed";
                return makeError(Error::InvalidArgument);
            }
            const uint64_t storageOffset = alignUp(fallbackStorageBytes, bottomLevelAlignment);
            if (storageOffset > desc.maxFallbackBlasBytes ||
                iter->second.accelerationStructureSize > desc.maxFallbackBlasBytes - storageOffset) {
                log = "MeshletStreamRuntime maxFallbackBlasBytes cannot hold all terminal fallback BLASes";
                return makeError(Error::OutOfMemory);
            }
            if (primitiveReferences > std::numeric_limits<uint64_t>::max() - totalFallbackReferences) {
                log = "MeshletStreamRuntime total fallback CLAS reference count overflowed";
                return makeError(Error::InvalidArgument);
            }
            fallbackBlasPrimitives_.push_back(FallbackBLASPrimitive{
                .primitiveIndex = primitiveIndex,
                .referenceCount = clusterCount,
                .referenceOffset = totalFallbackReferences,
                .storageOffset = storageOffset,
            });
            totalFallbackReferences += primitiveReferences;
            if (iter->second.accelerationStructureSize >
                std::numeric_limits<uint64_t>::max() - storageOffset) {
                log = "MeshletStreamRuntime fallback BLAS storage size overflowed";
                return makeError(Error::InvalidArgument);
            }
            fallbackStorageBytes = storageOffset + iter->second.accelerationStructureSize;
            fallbackScratchBytes = std::max(fallbackScratchBytes, iter->second.buildScratchSize);
        }
        if (fallbackBlasPrimitives_.empty()) {
            log = "MeshletStreamRuntime maxFallbackBlasBytes cannot hold one locked fallback primitive";
            return makeError(Error::OutOfMemory);
        }
        if (totalFallbackReferences > std::numeric_limits<uint64_t>::max() / sizeof(uint64_t)) {
            log = "MeshletStreamRuntime total fallback CLAS reference bytes overflowed";
            return makeError(Error::InvalidArgument);
        }

        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = fallbackStorageBytes,
                .usage = BufferUsageBits::AccelerationStructureStorage |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            fallbackBlasStorageBuffer_,
            log,
            "MeshletStreamRuntime fallback BLAS storage");
        if (!result) {
            return result;
        }
        const uint64_t fallbackStorageAddress =
            fallbackBlasStorageBuffer_->deviceAddress();

        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = fallbackScratchBytes + scratchAlignment - 1u,
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            fallbackBlasScratchBuffer_,
            log,
            "MeshletStreamRuntime fallback BLAS scratch");
        if (!result) {
            return result;
        }
        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = totalFallbackReferences * sizeof(uint64_t),
                .structureStride = sizeof(uint64_t),
                .usage = BufferUsageBits::Storage |
                    BufferUsageBits::AccelerationStructureBuildInput |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::HostUpload,
            },
            fallbackBlasReferenceBuffer_,
            log,
            "MeshletStreamRuntime fallback BLAS references");
        if (!result) {
            return result;
        }
        const uint64_t fallbackReferenceAddress =
            fallbackBlasReferenceBuffer_->deviceAddress();

        auto createFallbackHostBuffer = [&device, &log](
                                            uint64_t size,
                                            BufferUsageBits usage,
                                            std::unique_ptr<Buffer>& buffer,
                                            std::string_view label) {
            return createNamedBuffer(
                device,
                BufferDesc{
                    .size = size,
                    .usage = usage,
                    .memoryLocation = MemoryLocation::HostUpload,
                },
                buffer,
                log,
                label);
        };
        const uint64_t fallbackPrimitiveCount = fallbackBlasPrimitives_.size();
        const uint64_t fallbackBuildInfoBytes =
            fallbackPrimitiveCount * sizeof(MeshletStreamGPUBLASBuildInfo);
        const uint64_t fallbackDestinationBytes = fallbackPrimitiveCount * sizeof(uint64_t);
        const uint64_t primitiveAddressBytes =
            static_cast<uint64_t>(asset_.primitiveCount()) * sizeof(uint64_t);
        result = createFallbackHostBuffer(
            fallbackBuildInfoBytes,
            BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
            fallbackBlasBuildInfoBuffer_,
            "MeshletStreamRuntime fallback BLAS build infos");
        if (!result) {
            return result;
        }
        auto* fallbackBuildInfos = static_cast<MeshletStreamGPUBLASBuildInfo*>(
            fallbackBlasBuildInfoBuffer_->map());
        if (fallbackBuildInfos == nullptr) {
            log = "MeshletStreamRuntime fallback BLAS build info buffer map failed";
            return makeError(Error::Failure);
        }
        for (size_t fallbackIndex = 0;
             fallbackIndex < fallbackBlasPrimitives_.size();
             ++fallbackIndex) {
            const FallbackBLASPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
            const uint64_t referenceAddress = fallbackReferenceAddress +
                fallback.referenceOffset * sizeof(uint64_t);
            fallbackBuildInfos[fallbackIndex] = MeshletStreamGPUBLASBuildInfo{
                .clusterReferencesCount = fallback.referenceCount,
                .clusterReferencesStride = sizeof(uint64_t),
                .clusterReferencesAddressLow = static_cast<uint32_t>(referenceAddress),
                .clusterReferencesAddressHigh = static_cast<uint32_t>(referenceAddress >> 32u),
            };
        }
        fallbackBlasBuildInfoBuffer_->flush({0, fallbackBuildInfoBytes});
        fallbackBlasBuildInfoBuffer_->unmap();

        result = createFallbackHostBuffer(
            fallbackDestinationBytes,
            BufferUsageBits::Storage |
                BufferUsageBits::AccelerationStructureStorage |
                BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            fallbackBlasDestinationBuffer_,
            "MeshletStreamRuntime fallback BLAS destinations");
        if (!result) {
            return result;
        }
        auto* fallbackDestinations = static_cast<uint64_t*>(fallbackBlasDestinationBuffer_->map());
        if (fallbackDestinations == nullptr) {
            log = "MeshletStreamRuntime fallback BLAS destination buffer map failed";
            return makeError(Error::Failure);
        }
        for (size_t fallbackIndex = 0;
             fallbackIndex < fallbackBlasPrimitives_.size();
             ++fallbackIndex) {
            fallbackDestinations[fallbackIndex] =
                fallbackStorageAddress + fallbackBlasPrimitives_[fallbackIndex].storageOffset;
        }
        fallbackBlasDestinationBuffer_->flush({0, fallbackDestinationBytes});
        fallbackBlasDestinationBuffer_->unmap();

        result = createFallbackHostBuffer(
            primitiveAddressBytes,
            BufferUsageBits::Storage,
            fallbackBlasAddressBuffer_,
            "MeshletStreamRuntime fallback BLAS address table");
        if (!result) {
            return result;
        }
        auto* fallbackAddresses = static_cast<uint64_t*>(fallbackBlasAddressBuffer_->map());
        if (fallbackAddresses == nullptr) {
            log = "MeshletStreamRuntime fallback BLAS address table map failed";
            return makeError(Error::Failure);
        }
        std::fill_n(fallbackAddresses, asset_.primitiveCount(), uint64_t{0});
        fallbackBlasAddressBuffer_->flush({0, primitiveAddressBytes});
        fallbackBlasAddressBuffer_->unmap();

        if (fallbackStorageAddress == 0 ||
            fallbackBlasScratchBuffer_->deviceAddress() == 0 ||
            fallbackReferenceAddress == 0 ||
            fallbackBlasBuildInfoBuffer_->deviceAddress() == 0 ||
            fallbackBlasDestinationBuffer_->deviceAddress() == 0) {
            log = "MeshletStreamRuntime fallback BLAS buffers have no device addresses";
            return makeError(Error::Failure);
        }

        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = static_cast<uint64_t>(asset_.instanceCount()) *
                    sizeof(RayTracingGPUInstance),
                .structureStride = sizeof(RayTracingGPUInstance),
                .usage = BufferUsageBits::Storage |
                    BufferUsageBits::AccelerationStructureBuildInput |
                    BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            tlasInstanceBuffer_,
            log,
            "MeshletStreamRuntime TLAS instances");
        if (!result) {
            return result;
        }
        tlasInstanceBufferState_ = ResourceState::Undefined;
        const RayTracingAccelerationStructureBuildInputs tlasInputs{
            .type = RayTracingAccelerationStructureType::TopLevel,
            .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
            .instanceCount = asset_.instanceCount(),
        };
        RayTracingAccelerationStructureBuildSizes tlasSizes;
        result = device.queryRayTracingAccelerationStructureBuildSizes(tlasInputs).transform([&](auto rhiValue) { tlasSizes = std::move(rhiValue); });
        if (!result) {
            log = resultMessage(
                "queryRayTracingAccelerationStructureBuildSizes(MeshletStreamRuntime TLAS)",
                result);
            return result;
        }

        RayTracingAccelerationStructureProperties rtasProperties;
        result = device.queryRayTracingAccelerationStructureProperties().transform([&](auto rhiValue) { rtasProperties = std::move(rhiValue); });
        if (!result) {
            log = resultMessage(
                "queryRayTracingAccelerationStructureProperties(MeshletStreamRuntime)",
                result);
            return result;
        }
        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = tlasSizes.buildScratchSize + rtasProperties.scratchAlignment - 1u,
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::Device,
            },
            tlasScratchBuffer_,
            log,
            "MeshletStreamRuntime TLAS scratch");
        if (!result) {
            return result;
        }
        result = device.createRayTracingAccelerationStructure(RayTracingAccelerationStructureDesc{
                .type = RayTracingAccelerationStructureType::TopLevel,
                .buildFlags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                .size = tlasSizes.accelerationStructureSize,
            }).transform([&](auto rhiValue) { tlas_ = std::move(rhiValue); });
        if (!result) {
            log = resultMessage(
                "createRayTracingAccelerationStructure(MeshletStreamRuntime TLAS)",
                result);
            return result;
        }
    }
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = pageTableByteSize,
            .structureStride = sizeof(StreamPageTableEntry),
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination |
                (debugReadbackEnabled_ ? BufferUsageBits::TransferSource : BufferUsageBits::None),
            .memoryLocation = MemoryLocation::Device,
        },
        pageTableBuffer_,
        log,
        "MeshletStreamRuntime page table");
    if (!result) {
        return result;
    }
    pageTableState_ = ResourceState::Undefined;
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = requestByteSize,
            .structureStride = sizeof(uint32_t),
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::Device,
        },
        requestBuffer_,
        log,
        "MeshletStreamRuntime request");
    if (!result) {
        return result;
    }
    requestBufferState_ = ResourceState::Undefined;
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = requestReadbackByteSize + kMeshletStreamDemandStatsWords * sizeof(uint32_t) + sizeof(MeshletStreamGPUBLASHeader),
            .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback,
        },
        requestReadbackBuffer_,
        log,
        "MeshletStreamRuntime request readback");
    if (!result) {
        return result;
    }
    requestReadbacks_.resize(std::max(desc.queuedFrameCount, 1u) + 1u);
    for (auto& readback : requestReadbacks_) {
        result = createNamedBuffer(device, requestReadbackBuffer_->desc(), readback.buffer,
            log, "MeshletStreamRuntime completed feedback");
        if (!result) { return result; }
    }
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = sizeof(StreamRequestBufferHeader),
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        requestClearBuffer_,
        log,
        "MeshletStreamRuntime request clear");
    if (!result) {
        return result;
    }
    StreamRequestBufferHeader clearHeader{
        .maxLoadRequests = maxGpuPageRequests_,
        .maxUnloadRequests = maxGpuPageUnloadRequests_,
    };
    result = updateHostBuffer(*requestClearBuffer_, &clearHeader, sizeof(clearHeader));
    if (!result) {
        return result;
    }
    result = createHostStorageBuffer(
        device,
        sizeof(MeshletStreamGPUParams),
        paramsBuffer_,
        log,
        "MeshletStreamRuntime params");
    if (!result) {
        return result;
    }
    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = static_cast<uint64_t>(visibleClusterCapacity()) *
                sizeof(CompactStreamVisibleRecord),
            .structureStride = sizeof(CompactStreamVisibleRecord),
            .usage = debugReadbackEnabled_ ? BufferUsageBits::Storage | BufferUsageBits::TransferSource : BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        },
        visibleClusterBuffer_,
        log,
        "MeshletStreamRuntime visible clusters");
    if (!result) {
        return result;
    }
    spdlog::info("[MeshletStreamRuntime] Visible records capacity={} stride={} bytes={}",
        visibleClusterCapacity(), sizeof(CompactStreamVisibleRecord), visibleClusterBuffer_->desc().size);
    visibleClusterBufferState_ = ResourceState::Undefined;
    result = createHostStorageBuffer(
        device,
        sizeof(MeshletStreamGPURasterBindings),
        rasterBindingsBuffer_,
        log,
        "MeshletStreamRuntime raster bindings");
    if (!result) {
        return result;
    }

    residentPageFrames_.resize(std::max(desc.queuedFrameCount, 1u));
    for (ResidentPageFrame& frame : residentPageFrames_) {
        result = createHostStorageBuffer(
            device,
            std::max<uint64_t>(
                static_cast<uint64_t>(residentPageCapacity_) * sizeof(uint32_t),
                sizeof(uint32_t)),
            frame.buffer,
            log,
            "MeshletStreamRuntime resident pages");
        if (!result) {
            return result;
        }
    }

    if (uint64_t(desc.rasterMaterialTextureCapacity) + 4u > device.capabilities().maxBindlessSampledImages) {
        log = "Stream raster material textures exceed the device descriptor capacity";
        return makeError(Error::Unsupported);
    }
    result = metallic::render::ResourceRegistry::forDevice(device).transform([&](auto rhiValue) { registry_ = std::move(rhiValue); });
    if (!result) { return result; }
    result = allocateAndWriteBuffer(*registry_, *pageBuffer_, pageHandle_, log, "meshlet stream pages");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *activeGroupBuffer_, activeGroupHandle_, log, "meshlet stream active groups");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *activeHeaderBuffer_, activeHeaderHandle_, log, "meshlet stream active header");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *pageTableBuffer_, pageTableHandle_, log, "meshlet stream page table");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *paramsBuffer_, paramsHandle_, log, "meshlet stream params");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *registry_,
        *visibleClusterBuffer_,
        visibleClusterHandle_,
        log,
        "meshlet stream visible clusters");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *registry_,
        *rasterBindingsBuffer_,
        rasterBindingsHandle_,
        log,
        "meshlet stream raster bindings");
    if (!result) {
        return result;
    }
    result = updateRasterBindings(MeshletStreamGPURasterBindings{});
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *requestBuffer_, requestHandle_, log, "meshlet stream request");
    if (!result) {
        return result;
    }
    for (ResidentPageFrame& frame : residentPageFrames_) {
        result = allocateAndWriteBuffer(
            *registry_,
            *frame.buffer,
            frame.handle,
            log,
            "meshlet stream resident pages");
        if (!result) {
            return result;
        }
    }
    result = allocateAndWriteBuffer(*registry_, *instanceBuffer_, instanceHandle_, log, "meshlet stream instances");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *primitiveBuffer_, primitiveHandle_, log, "meshlet stream primitives");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *lodLevelBuffer_, lodLevelHandle_, log, "meshlet stream LOD levels");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *groupBuffer_, groupHandle_, log, "meshlet stream groups");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *lodTopologyBuffer_, lodTopologyHandle_, log, "meshlet stream LOD topology");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *lodStateBuffer_, lodStateHandle_, log, "meshlet stream LOD frontier");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*registry_, *demandBuffer_, demandHandle_, log, "meshlet stream demand");
    if (!result) { return result; }
    result = allocateAndWriteBuffer(*registry_, *nodeBuffer_, nodeHandle_, log, "meshlet stream hierarchy nodes");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *registry_,
        *drawIndirectBuffer_,
        drawIndirectHandle_,
        log,
        "meshlet stream draw indirect");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *registry_,
        *traversalHeaderBuffer_,
        traversalHeaderHandle_,
        log,
        "meshlet stream traversal header");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *registry_,
        *traversalWorkBuffer_,
        traversalWorkHandle_,
        log,
        "meshlet stream traversal work");
    if (!result) {
        return result;
    }
    if (clasPool_ != nullptr) {
        result = allocateAndWriteBuffer(
            *registry_,
            *clasPool_->clusterAddressBuffer(),
            clasAddressHandle_,
            log,
            "meshlet stream CLAS addresses");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *registry_,
            *clasPool_->pageTableBuffer(),
            clasPageTableHandle_,
            log,
            "meshlet stream CLAS page table");
        if (!result) {
            return result;
        }
    }
    if (desc.enableClusterRtx) {
        result = allocateAndWriteBuffer(
            *registry_,
            *blasHeaderBuffer_,
            blasHeaderHandle_,
            log,
            "meshlet stream BLAS header");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *registry_,
            *instanceBlasBuffer_,
            instanceBlasHandle_,
            log,
            "meshlet stream instance BLAS inputs");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *registry_,
            *blasBuildInfoBuffer_,
            blasBuildInfoHandle_,
            log,
            "meshlet stream BLAS build infos");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *registry_,
            *blasClusterReferenceBuffer_,
            blasClusterReferenceHandle_,
            log,
            "meshlet stream BLAS cluster references");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *registry_,
            *fallbackBlasAddressBuffer_,
            fallbackBlasAddressHandle_,
            log,
            "meshlet stream fallback BLAS addresses");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *registry_,
            *blasAddressBuffer_,
            dynamicBlasAddressHandle_,
            log,
            "meshlet stream dynamic BLAS addresses");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *registry_,
            *tlasInstanceBuffer_,
            tlasInstanceHandle_,
            log,
            "meshlet stream TLAS instances");
        if (!result) {
            return result;
        }
    }

    phase.next("streamInit.updatePass");
    updatePass_ = std::make_unique<UpdatePass>();
    result = updatePass_->initialize(device, *registry_, updateByteSize, desc.queuedFrameCount, log, pipelineCache);
    if (!result) {
        return result;
    }
    phase.next("streamInit.traversalPass");
    traversalPass_ = std::make_unique<TraversalPass>();
    result = traversalPass_->initialize(device, log, pipelineCache);
    if (!result) {
        return result;
    }
    phase.next("streamInit.activePass");
    activeBuildPass_ = std::make_unique<ActiveBuildPass>();
    result = activeBuildPass_->initialize(device, log, pipelineCache);
    if (!result) {
        return result;
    }
    if (desc.enableClusterRtx) {
        blasInputPass_ = std::make_unique<BLASInputPass>();
        result = blasInputPass_->initialize(device, log, pipelineCache);
        if (!result) {
            return result;
        }
        tlasInputPass_ = std::make_unique<TLASInputPass>();
        result = tlasInputPass_->initialize(device, log, pipelineCache);
        if (!result) {
            return result;
        }
    }

    frameUploads_.resize(std::max(desc.queuedFrameCount, 1u));
    for (size_t i = 1; i < frameUploads_.size(); ++i) {
        auto& slot = frameUploads_[i];
        if (!(result = device.createBuffer(paramsBuffer_->desc()).transform([&](auto rhiValue) { slot.params = std::move(rhiValue); })) ||
            !(result = device.createBuffer(rasterBindingsBuffer_->desc()).transform([&](auto rhiValue) { slot.raster = std::move(rhiValue); })) ||
            !(result = device.createBuffer(requestClearBuffer_->desc()).transform([&](auto rhiValue) { slot.clear = std::move(rhiValue); })) ||
            !(result = allocateAndWriteBuffer(*registry_, *slot.params, slot.paramsHandle, log, "stream frame params")) ||
            !(result = allocateAndWriteBuffer(*registry_, *slot.raster, slot.rasterHandle, log, "stream frame raster"))) {
            return result;
        }
    }
    return {};
}

void MeshletStreamRuntime::reset()
{
    sceneReadinessCache_ = std::make_shared<SceneReadinessCache>();
    rasterSnapshotFrozen_ = false;
    ++debugGeneration_;
    debugRequestSourceKnown_ = false;
    tlasBuilt_ = false;
    topLevelBuildPending_ = false;
    residency_.reset();
    asset_.close();
    drawBounds_.reset();
    sceneTransformRevision_ = 0;
    sceneVisibilityRevision_ = 0;
    sceneResourceIdentity_ = 0;
    runtimeRenderNodeIndices_.clear();
    gpuSceneInstanceMapping_.clear();
    pageBuffer_.reset();
    activeGroupBuffer_.reset();
    activeHeaderBuffer_.reset();
    pageTableBuffer_.reset();
    requestBuffer_.reset();
    requestReadbackBuffer_.reset();
    requestReadbacks_.clear();
    consumedRequestFrame_ = 0;
    maintenancePrepared_ = false;
    requestClearBuffer_.reset();
    frameUploads_.clear();
    currentUploadSlot_ = 0;
    paramsBuffer_.reset();
    visibleClusterBuffer_.reset();
    rasterBindingsBuffer_.reset();
    residentPageFrames_.clear();
    instanceBuffer_.reset();
    primitiveBuffer_.reset();
    lodLevelBuffer_.reset();
    groupBuffer_.reset();
    lodTopologyBuffer_.reset();
    immutableMetadataUpload_.reset();
    deviceImmutableMetadata_ = false;
    lodStateBuffer_.reset();
    demandBuffer_.reset();
    demandHandle_ = {};
    demandBufferState_ = ResourceState::Undefined;
    demandTaskOffset_ = 0;
    demandTaskCount_ = 0;
    demandInstanceOffsetsOffset_ = 0;
    distributedPageDemand_ = false;
    currentFrameDistributedDemand_ = true;
    distributedDemandMinGroups_ = 65536;
    recentDemandGroupTests_ = UINT32_MAX;
    recentDemandStats_ = {};
    lodTransitionHistoryBytes_ = 0;
    recentPrefetchGpuRequests_ = 0;
    recentPrefetchGpuDropped_ = 0;
    lodInstanceOffsetsOffset_ = 0;
    lodStateBufferState_ = ResourceState::Undefined;
    nodeBuffer_.reset();
    drawIndirectBuffer_.reset();
    traversalHeaderBuffer_.reset();
    traversalWorkBuffer_.reset();
    blasHeaderBuffer_.reset();
    blasCacheInitialized_ = std::make_shared<bool>(false);
    instanceBlasBuffer_.reset();
    blasBuildInfoBuffer_.reset();
    blasClusterReferenceBuffer_.reset();
    blasStorageBuffer_.reset();
    blasScratchBuffer_.reset();
    blasAddressBuffer_.reset();
    blasSizeBuffer_.reset();
    fallbackBlasStorageBuffer_.reset();
    fallbackBlasScratchBuffer_.reset();
    fallbackBlasReferenceBuffer_.reset();
    fallbackBlasBuildInfoBuffer_.reset();
    fallbackBlasDestinationBuffer_.reset();
    fallbackBlasAddressBuffer_.reset();
    tlasInstanceBuffer_.reset();
    tlasScratchBuffer_.reset();
    tlas_.reset();
    registry_.reset();
    updatePass_.reset();
    traversalPass_.reset();
    activeBuildPass_.reset();
    blasInputPass_.reset();
    tlasInputPass_.reset();
    clasPool_.reset();
    pendingClasPlans_.clear();
    pendingClasPages_.clear();
    queuedClasPages_.clear();
    maxClasBuildClusters_ = 0;
    coldPageRetentionFrames_ = 0;
    adaptivePageRetention_ = false;
    geometryReclaimReserveBytes_ = 0;
    geometryDemandReserveBytes_ = 0;
    maxDevicePageBytes_ = 0;
    clusterRtxEnabled_ = false;
    pageHandle_ = {};
    activeGroupHandle_ = {};
    activeHeaderHandle_ = {};
    pageTableHandle_ = {};
    paramsHandle_ = {};
    visibleClusterHandle_ = {};
    rasterBindingsHandle_ = {};
    requestHandle_ = {};
    instanceHandle_ = {};
    primitiveHandle_ = {};
    lodLevelHandle_ = {};
    groupHandle_ = {};
    lodTopologyHandle_ = {};
    lodStateHandle_ = {};
    nodeHandle_ = {};
    drawIndirectHandle_ = {};
    traversalHeaderHandle_ = {};
    traversalWorkHandle_ = {};
    clasAddressHandle_ = {};
    clasPageTableHandle_ = {};
    blasHeaderHandle_ = {};
    instanceBlasHandle_ = {};
    blasBuildInfoHandle_ = {};
    blasClusterReferenceHandle_ = {};
    fallbackBlasAddressHandle_ = {};
    dynamicBlasAddressHandle_ = {};
    tlasInstanceHandle_ = {};
    pageBufferState_ = ResourceState::Undefined;
    activeGroupBufferState_ = ResourceState::Undefined;
    activeHeaderBufferState_ = ResourceState::Undefined;
    pageTableState_ = ResourceState::Undefined;
    requestBufferState_ = ResourceState::Undefined;
    visibleClusterBufferState_ = ResourceState::Undefined;
    drawIndirectBufferState_ = ResourceState::Undefined;
    traversalHeaderBufferState_ = ResourceState::Undefined;
    traversalWorkBufferState_ = ResourceState::Undefined;
    blasHeaderBufferState_ = ResourceState::Undefined;
    instanceBlasBufferState_ = ResourceState::Undefined;
    blasBuildInfoBufferState_ = ResourceState::Undefined;
    blasClusterReferenceBufferState_ = ResourceState::Undefined;
    tlasInstanceBufferState_ = ResourceState::Undefined;
    pageTableInitialized_ = std::make_shared<bool>(false);
    currentFrameOrderedUploadCount_ = 0;
    recentBlasHeader_ = {};
    requestReadbackValid_ = false;
    frameIndex_ = 0;
    maxResidentPages_ = 0;
    maxPageUploadsPerFrame_ = 0;
    maxUploadBytesPerFrame_ = 0;
    maxGpuPageRequests_ = 0;
    screenSpacePagePriority_ = false;
    viewDrivenPageDemand_ = false;
    prefetchPages_ = false;
    predictivePrefetch_ = true;
    lodTransitionTelemetry_ = false;
    prefetchPredictor_.reset();
    prefetchForecast_ = {};
    previousPrefetchTime_ = 0;
    currentFramePrefetch_ = false;
    recentGpuRequestCount_ = 0;
    maxGpuPageUnloadRequests_ = 0;
    maxUpdatePatches_ = 0;
    residentPageCapacity_ = 0;
    currentResidentPageCount_ = 0;
    maxResidentBytes_ = 0;
    lockedFallbackPages_.clear();
    maxActiveGroups_ = 0;
    rasterCandidateCapacity_ = 0;
    maxActiveGroupClusters_ = 0;
    maxPrimitiveGroupCount_ = 0;
    traversalWorkerCount_ = 0;
    traversalWorkCapacity_ = 0;
    blasClusterReferenceCapacity_ = 0;
    blasBuildCapacity_ = 0;
    maxBlasClustersPerBuild_ = 0;
    blasSizeClasses_ = {};
    blasPublicationEpoch_ = 0;
    blasAddressBufferState_ = ResourceState::Undefined;
    blasClusterReferenceAddress_ = 0;
    fallbackBlasPrimitives_.clear();
    currentFrameUploadCount_ = 0;
    previousFrameParams_ = {};
    previousFrameParamsValid_ = false;
}

bool MeshletStreamRuntime::immutableMetadataReady() const
{
    if (!deviceImmutableMetadata_) { return true; }
    const auto& upload = immutableMetadataUpload_;
    if (upload && upload->completed) { return true; }
    const bool completed = upload && upload->submittedBytes == upload->totalBytes && upload->submission &&
        upload->submission->resolved() && !upload->submission->cancelled() &&
        upload->completion.isSubmitted() && upload->completion.isComplete();
    if (completed) { upload->completed = true; }
    return completed;
}

uint64_t MeshletStreamRuntime::immutableMetadataUploadedBytes() const
{
    return immutableMetadataUpload_ ? immutableMetadataUpload_->submittedBytes : 0;
}

Result<> MeshletStreamRuntime::prepareImmutableMetadataRead(CommandBuffer& commandBuffer) const
{
    if (!immutableMetadataReady()) { return makeError(Error::InvalidArgument); }
    if (!immutableMetadataUpload_) { return {}; }
    // Retain and wait the timeline even after the loader's CPU wait/reset: a
    // consumer may use a different queue from the initial upload queue.
    return addCommandDependency(commandBuffer, immutableMetadataUpload_->completion);
}

Result<> MeshletStreamRuntime::uploadImmutableMetadata(CommandBuffer& commandBuffer)
{
    const auto upload = immutableMetadataUpload_;
    auto* frame = metallic::render::RenderFrameContext::from(commandBuffer);
    if (!upload || !upload->staging || !frame || !frame->recording() || !commandBuffer.recording()) {
        return makeError(Error::InvalidArgument);
    }
    // Do not reuse mapped staging while a previous accepted copy still reads it.
    // A cancelled copy never advances the offset, including a cancelled tail
    // whose frame aggregate covers some other accepted commands.
    if (upload->submission && (!upload->submission->resolved() ||
        (!upload->submission->cancelled() && !upload->completion.isComplete()))) {
        return makeError(Error::InvalidArgument);
    }
    const uint64_t offset = upload->submittedBytes;
    if (offset >= upload->totalBytes) { return makeError(Error::InvalidArgument); }
    const uint64_t bytes = std::min(upload->totalBytes - offset, upload->staging->desc().size);
    const uint64_t groupBytes = groupBuffer_->desc().size;
    const uint64_t groupCopyBytes = offset < groupBytes ? std::min(bytes, groupBytes - offset) : 0;
    const uint64_t topologyOffset = offset > groupBytes ? offset - groupBytes : 0;
    const uint64_t topologyCopyBytes = bytes - groupCopyBytes;
    auto* mapped = static_cast<std::byte*>(upload->staging->map());
    if (!mapped) { return makeError(Error::Failure); }
    if (groupCopyBytes) {
        std::memcpy(mapped, upload->groups.data() + offset, static_cast<size_t>(groupCopyBytes));
    }
    if (topologyCopyBytes) {
        std::memcpy(mapped + groupCopyBytes, upload->topology.data() + topologyOffset,
            static_cast<size_t>(topologyCopyBytes));
    }
    upload->staging->flush({0, bytes});
    upload->staging->unmap();

    auto result = commandBuffer.retainResource(upload);
    if (result) { result = commandBuffer.retainResource(upload->staging->retainAllocation()); }
    for (const auto& lease : {groupHandle_, lodTopologyHandle_}) {
        if (result) { result = registry_->retain(commandBuffer, lease); }
    }
    if (!result) { return result; }
    const BufferBarrierDesc stagingReady{
        .buffer = upload->staging.get(),
        .before = {PipelineStageBits::Host, AccessBits::HostWrite},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
        .range = {0, bytes},
    };
    result = commandBuffer.synchronize({.buffers = {&stagingReady, 1}});
    if (!result) { return result; }
    const auto copy = [&](Buffer& destination, uint64_t destinationOffset,
                          uint64_t sourceOffset, uint64_t copyBytes) -> Result<> {
        if (copyBytes == 0) { return {}; }
        const BufferBarrierDesc transferReady{
            .buffer = &destination, .before = {},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .range = {destinationOffset, copyBytes},
        };
        auto copied = commandBuffer.synchronize({.buffers = {&transferReady, 1}});
        if (!copied) { return copied; }
        auto source = upload->staging->slice({sourceOffset, copyBytes});
        if (!source) { return std::unexpected(source.error()); }
        auto target = destination.slice({destinationOffset, copyBytes});
        if (!target) { return std::unexpected(target.error()); }
        copied = commandBuffer.copyBuffer(*source, *target);
        if (!copied) { return copied; }
        const BufferBarrierDesc shaderReady{
            .buffer = &destination,
            .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
            .range = {destinationOffset, copyBytes},
        };
        return commandBuffer.synchronize({.buffers = {&shaderReady, 1}});
    };
    result = copy(*groupBuffer_, offset, 0, groupCopyBytes);
    if (result) { result = copy(*lodTopologyBuffer_, topologyOffset, groupCopyBytes, topologyCopyBytes); }
    if (!result) { return result; }
    const auto weak = std::weak_ptr<ImmutableMetadataUpload>(upload);
    auto transaction = std::make_shared<SubmissionTransaction>(
        [weak, bytes, cache = sceneReadinessCache_] {
            if (auto accepted = weak.lock()) {
                accepted->submittedBytes += bytes;
                ++accepted->uploadBatches;
                if (accepted->submittedBytes == accepted->totalBytes) {
                    std::vector<std::byte>().swap(accepted->groups);
                    std::vector<std::byte>().swap(accepted->topology);
                }
            }
            cache->valid = false;
        },
        [cache = sceneReadinessCache_] { cache->valid = false; });
    result = commandBuffer.addSubmissionTransaction(transaction);
    if (!result) { return result; }
    upload->submission = std::move(transaction);
    upload->completion = frame->completion();
    sceneReadinessCache_->valid = false;
    return {};
}

bool MeshletStreamRuntime::ready() const
{
    return asset_.valid() &&
        registry_ != nullptr &&
        updatePass_ != nullptr &&
        updatePass_->ready() &&
        traversalPass_ != nullptr &&
        traversalPass_->ready() &&
        activeBuildPass_ != nullptr &&
        activeBuildPass_->ready() &&
        pageBuffer_ != nullptr &&
        activeGroupBuffer_ != nullptr &&
        activeHeaderBuffer_ != nullptr &&
        pageTableBuffer_ != nullptr &&
        requestBuffer_ != nullptr &&
        requestReadbackBuffer_ != nullptr &&
        requestClearBuffer_ != nullptr &&
        paramsBuffer_ != nullptr &&
        visibleClusterBuffer_ != nullptr &&
        rasterBindingsBuffer_ != nullptr &&
        instanceBuffer_ != nullptr &&
        primitiveBuffer_ != nullptr &&
        lodLevelBuffer_ != nullptr &&
        groupBuffer_ != nullptr &&
        lodTopologyBuffer_ != nullptr &&
        lodStateBuffer_ != nullptr &&
        nodeBuffer_ != nullptr &&
        drawIndirectBuffer_ != nullptr &&
        traversalHeaderBuffer_ != nullptr &&
        traversalWorkBuffer_ != nullptr &&
        (clasPool_ == nullptr || clasPool_->ready()) &&
        (!clusterRtxEnabled_ ||
            (blasInputPass_ != nullptr &&
                blasInputPass_->ready() &&
                tlasInputPass_ != nullptr &&
                tlasInputPass_->ready() &&
                blasHeaderBuffer_ != nullptr &&
                instanceBlasBuffer_ != nullptr &&
                blasBuildInfoBuffer_ != nullptr &&
                blasClusterReferenceBuffer_ != nullptr &&
                blasStorageBuffer_ != nullptr &&
                blasScratchBuffer_ != nullptr &&
                blasAddressBuffer_ != nullptr &&
                blasSizeBuffer_ != nullptr &&
                fallbackBlasStorageBuffer_ != nullptr &&
                fallbackBlasScratchBuffer_ != nullptr &&
                fallbackBlasReferenceBuffer_ != nullptr &&
                fallbackBlasBuildInfoBuffer_ != nullptr &&
                fallbackBlasDestinationBuffer_ != nullptr &&
                fallbackBlasAddressBuffer_ != nullptr &&
                tlasInstanceBuffer_ != nullptr &&
                tlasScratchBuffer_ != nullptr &&
                tlas_ != nullptr &&
                tlas_->valid()));
}

StreamSceneReadiness MeshletStreamRuntime::sceneReadiness() const
{
    auto& cache = *sceneReadinessCache_;
    const bool metadataReady = immutableMetadataReady();
    if (!metadataReady) { cache.valid = false; }
    if (cache.valid || cache.rootsInvalidated) {
        auto value = cache.value;
        value.ready = value.ready && metadataReady;
        return value;
    }
    ++cache.scans;
    StreamSceneReadiness result;
    const bool needsClas = clusterRtxEnabled_ && clasPool_;
    result.requiredPages = static_cast<uint32_t>(lockedFallbackPages_.size()) * (needsClas ? 2u : 1u);
    for (uint32_t page : lockedFallbackPages_) {
        result.completedPages += residency_.pageResident(page) ? 1u : 0u;
        if (needsClas) { result.completedPages += clasPool_->pageHasClas(page) ? 1u : 0u; }
    }
    result.ready = ready() && metadataReady && *pageTableInitialized_ && result.requiredPages != 0 &&
        result.completedPages == result.requiredPages &&
        (!clusterRtxEnabled_ || std::all_of(fallbackBlasPrimitives_.begin(), fallbackBlasPrimitives_.end(),
            [](const auto& fallback) { return fallback.submitted(); }));
    cache.value = result;
    cache.valid = true;
    return result;
}

RayTracingAccelerationStructure* MeshletStreamRuntime::accelerationStructure() const
{
    return tlasBuilt_ ? tlas_.get() : nullptr;
}

void MeshletStreamRuntime::prepareMaintenance(CPUProfileRecorder* profiler, bool allowLegacyReadback)
{
    if (immutableMetadataReady() && immutableMetadataUpload_) { immutableMetadataUpload_->staging.reset(); }
    if (maintenancePrepared_ || rasterSnapshotFrozen_ || !ready()) { return; }
    maintenancePrepared_ = true;
    if (!sceneReadinessCache_->value.ready) { sceneReadinessCache_->valid = false; }
    CPUProfileScope profile(profiler, "Residency completion");
    residency_.beginFrame(profiler);
    profile.next("GPU request feedback");
    consumeGpuRequestReadback(profiler, allowLegacyReadback);
    profile.next("Joint cold page reclaim");
    if (coldPageRetentionFrames_ != 0) {
        const auto clas = clasPool_ ? clasPool_->stats() : MeshletStreamCLASPoolStats{};
        residency_.reclaimColdPages({.clasUsedBytes = clas.usedStorageBytes, .clasCapacityBytes = clas.storageBudgetBytes,
            .clasRetiringBytes = clas.retiringStorageBytes, .retentionFrames = coldPageRetentionFrames_,
            .maxPages = adaptivePageRetention_ ? 1024u : 256u,
            .geometryReserveBytes = geometryReclaimReserveBytes_,
            .geometryPressureReserveBytes = geometryDemandReserveBytes_,
            .retainDemandCache = adaptivePageRetention_,
            .clasPageBytes = [this](uint32_t page) { return clasPool_ ? clasPool_->pageStorageBytes(page) : 0; }}, profiler);
    }
    profile.next("Discard obsolete CLAS plans");
    if (clasPool_) {
        std::erase_if(pendingClasPlans_, [this](const auto& entry) { return !residency_.pageAllocated(entry.first); });
    }
}

Result<> MeshletStreamRuntime::cmdBeginFrame(
    CommandBuffer& commandBuffer,
    Streamer& streamer,
    const MeshletStreamFrameDesc& frame,
    const std::function<Result<>()>& flushUploads)
{
    return beginUploadBatch(commandBuffer, streamer, frame, flushUploads, false);
}

Result<> MeshletStreamRuntime::cmdLoadInitialResources(
    CommandBuffer& commandBuffer, Streamer& streamer, const std::function<Result<>()>& flushUploads)
{
    if (!ready() || !metallic::render::RenderFrameContext::from(commandBuffer) || !metallic::render::RenderFrameContext::from(commandBuffer)->recording()) {
        return makeError(Error::InvalidArgument);
    }
    if (!immutableMetadataReady()) { return uploadImmutableMetadata(commandBuffer); }
    if (immutableMetadataUpload_) { immutableMetadataUpload_->staging.reset(); }
    auto result = beginUploadBatch(commandBuffer, streamer, {}, flushUploads, true);
    if (!result) { return result; }
    if (residency_.stats().framePageLoadFailureCount != 0) { return makeError(Error::Failure); }
    if (!(result = initializePageTableIfNeeded(commandBuffer))) { return result; }
    if (!(result = applyPageTablePatches(commandBuffer))) { return result; }
    if (!(result = transitionPageBufferForTraversal(commandBuffer))) { return result; }
    return cmdBuildPendingClas(commandBuffer);
}

Result<> MeshletStreamRuntime::beginUploadBatch(
    CommandBuffer& commandBuffer,
    Streamer& streamer,
    const MeshletStreamFrameDesc& frame,
    const std::function<Result<>()>& flushUploads,
    bool initialLoad)
{
    if (!registry_) { return makeError(Error::InvalidArgument); }
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    for (const auto& lease : {
        pageHandle_, activeGroupHandle_, activeHeaderHandle_, pageTableHandle_,
        paramsHandle_, visibleClusterHandle_, rasterBindingsHandle_, requestHandle_,
        instanceHandle_, primitiveHandle_, lodLevelHandle_, groupHandle_,
        lodTopologyHandle_, lodStateHandle_, demandHandle_, nodeHandle_,
        drawIndirectHandle_, traversalHeaderHandle_, traversalWorkHandle_, clasAddressHandle_,
        clasPageTableHandle_, blasHeaderHandle_, instanceBlasHandle_, blasBuildInfoHandle_,
        blasClusterReferenceHandle_, fallbackBlasAddressHandle_, dynamicBlasAddressHandle_, tlasInstanceHandle_}) {
        if (lease.valid()) {
            auto retained = registry_->retain(commandBuffer, lease);
            if (!retained) { return retained; }
        }
    }
    for (const auto& slot : frameUploads_) {
        for (const auto& lease : {slot.paramsHandle, slot.rasterHandle}) {
            if (!lease.valid()) { continue; }
            auto retained = registry_->retain(commandBuffer, lease);
            if (!retained) { return retained; }
        }
    }
    for (const auto& slot : residentPageFrames_) {
        if (!slot.handle.valid()) { continue; }
        auto retained = registry_->retain(commandBuffer, slot.handle);
        if (!retained) { return retained; }
    }

    // Geometry roots are locked; completed root CLAS / accepted fallback BLAS
    // remain valid until reset or an explicit root invalidation. While loading,
    // completion polling below can advance progress, so refresh once per frame.
    if (!sceneReadinessCache_->value.ready) { sceneReadinessCache_->valid = false; }
    rasterSnapshotFrozen_ = frame.freezeRasterSnapshot;
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }

    beginFrameCpuProfile_.reset();
    auto* profiler = &beginFrameCpuProfile_;
    CPUProfileScope profile(profiler, "Residency completion");
    ++frameIndex_;
    const uint32_t uploadSlot = frameIndex_ % uint32_t(frameUploads_.size());
    auto& nextUpload = frameUploads_[uploadSlot];
    if (nextUpload.completion.isSubmitted() && !nextUpload.completion.isComplete()) {
        const auto result = nextUpload.completion.wait(UINT64_MAX);
        if (!result) { return result; }
    }
    const auto swapUpload = [&](FrameUploads& slot) {
        slot.params.swap(paramsBuffer_);
        slot.raster.swap(rasterBindingsBuffer_);
        slot.clear.swap(requestClearBuffer_);
        std::swap(slot.paramsHandle, paramsHandle_);
        std::swap(slot.rasterHandle, rasterBindingsHandle_);
    };
    if (uploadSlot != currentUploadSlot_) {
        swapUpload(frameUploads_[currentUploadSlot_]);
        swapUpload(nextUpload);
        currentUploadSlot_ = uploadSlot;
    }
    nextUpload.completion = metallic::render::RenderFrameContext::from(commandBuffer) ? metallic::render::RenderFrameContext::from(commandBuffer)->completion() : GPUCompletionPoint{};
    if (rasterSnapshotFrozen_) {
        // Rotate host-write frame slots, but do not publish completions, consume
        // requests, reclaim pages or enqueue new geometry/CLAS work.
        currentFrameUploadCount_ = 0;
        return {};
    }
    prepareMaintenance(profiler, true);
    maintenancePrepared_ = false;
    // CLAS publication writes GPU-visible metadata and stays after graph dependencies.
    if (clasPool_ != nullptr) {
        profile.next("CLAS completion / expiry");
        clasPool_->beginFrame(profiler);
        profile.next("Retire unloaded CLAS");
        clasPool_->retirePages(residency_.newlyUnloadedPages());
    }
    MeshletStreamResidencyManager::UploadObserver prepareClas;
    MeshletStreamResidencyManager::GPUUploadObserver prepareGpuClas;
    std::string planError;
    if (clasPool_) {
        prepareClas = [&](uint32_t page, std::span<const uint8_t> payload) {
            MeshletStreamCLASPagePlan plan;
            std::string reason;
            if (!buildMeshletStreamClasPagePlan(asset_.pages()[page], payload, page,
                    page * asset_.maxPageClusters(), plan, reason)) {
                planError = std::move(reason);
                return;
            }
            pendingClasPlans_.insert_or_assign(page, std::move(plan));
        };
    }
    if (clasPool_) {
        prepareGpuClas = [&](uint32_t page, const scene::MeshletStreamGPUPage& payload) {
            MeshletStreamCLASPagePlan plan;
            std::string reason;
            if (!buildMeshletStreamClasGpuPagePlan(payload, page,
                    page * asset_.maxPageClusters(), plan, reason)) {
                planError = std::move(reason);
                return;
            }
            pendingClasPlans_.insert_or_assign(page, std::move(plan));
        };
    }
    profile.next("Prepare page uploads");
    // Initialization advances by submission completion rather than Present. Keep
    // staging bounded independently of the steady-state per-frame controls.
    const uint32_t uploadPages = initialLoad ? std::min(1024u, maxUpdatePatches_) : maxPageUploadsPerFrame_;
    const uint64_t uploadBytes = initialLoad ? 64ull * 1024ull * 1024ull : maxUploadBytesPerFrame_;
    currentFrameUploadCount_ = residency_.processUploads(streamer, *pageBuffer_, uploadPages,
        prepareClas, profiler, uploadBytes, prepareGpuClas);
    if (!planError.empty()) {
        spdlog::error("[MeshletStreamRuntime] CLAS upload plan failed: {}", planError);
        return makeError(Error::Failure);
    }
    if (currentFrameUploadCount_ != 0) {
        profile.next("Record page copies");
        if (auto commandResult = transitionBuffer(commandBuffer, *pageBuffer_, pageBufferState_, ResourceState::TransferDestination); !commandResult) { return commandResult; }
        if (flushUploads) { if (auto commandResult = flushUploads(); !commandResult) { return commandResult; } }
        else { if (auto commandResult = streamer.copyStreamedData(commandBuffer); !commandResult) { return commandResult; } }
        if (auto commandResult = transitionBuffer(commandBuffer, *pageBuffer_, pageBufferState_, ResourceState::ShaderRead); !commandResult) { return commandResult; }
    }
    profile.next("Queue resident CLAS");
    if (clasPool_) {
        for (uint32_t page : residency_.newlyResidentPages()) {
            if (queuedClasPages_.insert(page).second) { pendingClasPages_.push_back(page); }
        }
    }
    profile.next("Publish resident pages");
    const std::span<const uint32_t> residentPages = residency_.residentPages();
    if (residentPages.size() > residentPageCapacity_ || residentPageFrames_.empty()) {
        return makeError(Error::Failure);
    }
    currentResidentPageCount_ = static_cast<uint32_t>(residentPages.size());
    if (!residentPages.empty()) {
        ResidentPageFrame& residentFrame = residentPageFrames_[frameIndex_ % residentPageFrames_.size()];
        const Result<> residentUpdate = updateHostBuffer(
            *residentFrame.buffer,
            residentPages.data(),
            static_cast<uint64_t>(residentPages.size()) * sizeof(uint32_t));
        if (!residentUpdate) {
            return residentUpdate;
        }
    }
    return {};
}

Result<> MeshletStreamRuntime::cmdPreTraversal(CommandBuffer& commandBuffer, const MeshletStreamFrameDesc& frame,
    const TraversalCheckpoint& checkpoint, bool deferTopLevelBuild)
{
    topLevelBuildPending_ = false;
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }

    if (rasterSnapshotFrozen_) {
        Result<> result = clearRequestBuffer(commandBuffer);
        if (result) { result = updateParamsBuffer(frame); }
        if (result) { result = transitionPageBufferForTraversal(commandBuffer); }
        return result;
    }
    if (checkpoint) { checkpoint("BeforeStreamUpdates"); }
    Result<> result = initializePageTableIfNeeded(commandBuffer);
    if (!result) {
        return result;
    }
    result = applyPageTablePatches(commandBuffer);
    if (!result) {
        return result;
    }
    result = clearRequestBuffer(commandBuffer);
    if (!result) {
        return result;
    }

    result = updateParamsBuffer(frame);
    if (!result) {
        return result;
    }
    if (checkpoint) { checkpoint("AfterStreamUpdates"); }
    if (screenSpacePagePriority_) {
        result = dispatchTraversal(commandBuffer, asset_.pageCount(), 2u);
        if (!result) { return result; }
    }
    if (checkpoint) { checkpoint("AfterStreamPriorityClear"); }
    result = buildActiveTable(commandBuffer, checkpoint);
    if (!result) {
        return result;
    }
    result = dispatchTraversal(
        commandBuffer,
        currentResidentPageCount_,
        kMeshletStreamTraversalUnloadPhase);
    if (!result) {
        return result;
    }
    result = transitionPageBufferForTraversal(commandBuffer);
    if (!result || clasPool_ == nullptr) {
        return result;
    }

    result = cmdBuildPendingClas(commandBuffer, checkpoint);
    if (!result || !clusterRtxEnabled_) { return result; }
    result = buildBlasInputs(commandBuffer, checkpoint);
    if (!result) {
        return result;
    }
    if (checkpoint) { checkpoint("BeforeBlasBuild"); }
    result = cmdBuildBlas(commandBuffer);
    if (!result) {
        return result;
    }
    topLevelBuildPending_ = true;
    return deferTopLevelBuild ? Result<>{} : cmdBuildTopLevelAccelerationStructure(commandBuffer, checkpoint);
}

Result<> MeshletStreamRuntime::cmdBuildTopLevelAccelerationStructure(CommandBuffer& commands,
    const TraversalCheckpoint& checkpoint)
{
    if (!topLevelBuildPending_) { return {}; }
    if (!instanceBlasBuffer_ || !tlasInstanceBuffer_ || !tlasScratchBuffer_ || !tlas_) {
        return makeError(Error::InvalidArgument);
    }
    using namespace detail;
    using Access = RenderGraphResourceAccess;
    constexpr auto compute = RenderGraphPassKind::Compute;
    // Buffer allocations have no layouts. Declaring the producer frontier makes
    // input generation -> AS build and scratch/AS reuse ordinary graph hazards.
    const std::array resources{
        GraphAccessResource{RenderGraphResourceType::Buffer, ResourceState::General,
            resourceSyncScope(instanceBlasBufferState_, PipelineStageBits::AllCommands)},
        GraphAccessResource{RenderGraphResourceType::Buffer, ResourceState::General,
            resourceSyncScope(tlasInstanceBufferState_, PipelineStageBits::AllCommands)},
        GraphAccessResource{RenderGraphResourceType::Buffer, ResourceState::General,
            scopeForGraphAccess(Access::BufferAccelerationStructureScratchReadWrite, compute)},
        GraphAccessResource{RenderGraphResourceType::AccelerationStructure,
            tlasBuilt_ ? ResourceState::ShaderRead : ResourceState::Undefined,
            scopeForGraphAccess(Access::AccelerationStructureShaderRead, RenderGraphPassKind::Unsafe)},
    };
    const std::array passes{
        GraphAccessPass{0, {declaredGraphAccess(0, Access::BufferShaderRead, compute),
            declaredGraphAccess(1, Access::BufferStorageWrite, compute)}},
        GraphAccessPass{0, {declaredGraphAccess(1, Access::BufferAccelerationStructureBuildRead, compute),
            declaredGraphAccess(2, Access::BufferAccelerationStructureScratchReadWrite, compute),
            declaredGraphAccess(3, Access::AccelerationStructureBuildWrite, compute)}},
        GraphAccessPass{0, {declaredGraphAccess(3, Access::AccelerationStructureShaderRead,
            RenderGraphPassKind::Unsafe)}},
    };
    auto plan = buildGraphAccessPlan(resources, passes);
    if (!plan) { return makeError(plan.error()); }
    std::array<RenderGraphResource, 4> allocations;
    allocations[0].type = allocations[1].type = allocations[2].type = RenderGraphResourceType::Buffer;
    allocations[0].buffer = instanceBlasBuffer_.get();
    allocations[1].buffer = tlasInstanceBuffer_.get();
    allocations[2].buffer = tlasScratchBuffer_.get();
    for (size_t i = 0; i < 3; ++i) { allocations[i].bufferDesc = allocations[i].buffer->desc(); }
    allocations[3].type = RenderGraphResourceType::AccelerationStructure;
    allocations[3].accelerationStructure = tlas_.get();
    std::array<GraphAccessBinding, 4> bindings;
    for (size_t i = 0; i < bindings.size(); ++i) {
        auto binding = bindGraphAccessResource(allocations[i]);
        if (!binding) { return makeError(binding.error()); }
        bindings[i] = std::move(*binding);
    }
    if (checkpoint) { checkpoint("BeforeTlasInput"); }
    auto result = recordGraphAccessBarriers(commands, plan->passes[0], bindings);
    if (result) { result = buildTlasInstances(commands); }
    if (!result) { return result; }
    if (checkpoint) { checkpoint("BeforeTlasBuild"); }
    result = recordGraphAccessBarriers(commands, plan->passes[1], bindings);
    if (result) { result = cmdBuildTlas(commands); }
    if (result) { result = recordGraphAccessBarriers(commands, plan->passes[2], bindings); }
    if (!result) { return result; }
    instanceBlasBufferState_ = tlasInstanceBufferState_ = ResourceState::General;
    topLevelBuildPending_ = false;
    if (checkpoint) { checkpoint("AfterTlasBuild"); }
    return {};
}

Result<> MeshletStreamRuntime::cmdBuildPendingClas(CommandBuffer& commandBuffer,
    const TraversalCheckpoint& checkpoint)
{
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    if (!clasPool_) { return {}; }
    Result<> result;
    if (checkpoint) { checkpoint("BeforeStreamClasBuild"); }
    std::vector<MeshletStreamCLASPageBuild> clasBuilds;
    uint32_t clusterCount = 0;
    const size_t pendingCount = pendingClasPages_.size();
    // Visit only queued pages, with a bounded batch. No resident-set scan or I/O.
    for (size_t i = 0; i < pendingCount; ++i) {
        const uint32_t page = pendingClasPages_.front();
        if (residency_.pageResident(page)) {
            const uint32_t count = clasPool_->pageHasClas(page) || clasPool_->pageBuildPending(page)
                ? 0 : asset_.pages()[page].clusterCount;
            if (count > maxClasBuildClusters_ - clusterCount) { break; }
            const auto plan = pendingClasPlans_.find(page);
            // A retired CLAS may have expired between upload admission and completion.
            // Always retain an upload plan until the resident page is built.
            if (count && plan == pendingClasPlans_.end()) {
                spdlog::error("[MeshletStreamRuntime] Missing CLAS upload plan: page={} state={} frame={} pending={} geometryOffset={}",
                    page, static_cast<uint32_t>(residency_.pageState(page)), frameIndex_, pendingClasPages_.size(),
                    residency_.deviceOffsetForPage(page));
                return makeError(Error::Failure);
            }
            clasBuilds.push_back({.pageIndex = page, .deviceOffsetBytes = residency_.deviceOffsetForPage(page),
                .plan = plan != pendingClasPlans_.end() ? &plan->second : nullptr});
            clusterCount += count;
        } else {
            // This queue entry belongs to an old residency. A new upload may
            // already own a plan while waiting for completion. Drop only the
            // queue entry; beginFrame releases plans with their allocations.
            queuedClasPages_.erase(page);
        }
        pendingClasPages_.pop_front();
    }
    { // Pending size/move batches also need progress on frames without new pages.
        std::string clasLog;
        result = clasPool_->cmdBuildPages(commandBuffer, *pageBuffer_, clasBuilds, clasLog);
        if (!result) {
            spdlog::error("[MeshletStreamRuntime] CLAS build failed: {}", clasLog);
            return result;
        }
        for (const auto& build : clasBuilds) {
            if (clasPool_->pageHasClas(build.pageIndex)) {
                pendingClasPlans_.erase(build.pageIndex);
                queuedClasPages_.erase(build.pageIndex);
            } else {
                pendingClasPages_.push_back(build.pageIndex);
            }
        }
    }
    if (checkpoint) { checkpoint("AfterStreamClasBuild"); }
    if (!clusterRtxEnabled_) { return {}; }
    if (checkpoint) { checkpoint("BeforeFallbackBlas"); }
    result = cmdBuildFallbackBlas(commandBuffer);
    if (!result) {
        return result;
    }
    return result;
}

Result<> MeshletStreamRuntime::cmdPostTraversal(CommandBuffer& commandBuffer)
{
    return ready() ? prepareImmutableMetadataRead(commandBuffer) : makeError(Error::InvalidArgument);
}

Result<> MeshletStreamRuntime::cmdEndFrame(CommandBuffer& commandBuffer)
{
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    if (rasterSnapshotFrozen_) { return ready() ? Result<>{} : makeError(Error::InvalidArgument); }
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }

    Result<> result;
    if (screenSpacePagePriority_) {
        result = dispatchTraversal(commandBuffer, maxGpuPageRequests_, 3u);
        if (!result) { return result; }
    }
    result = copyRequestBufferForReadback(commandBuffer);
    if (!result) {
        return result;
    }

    if (currentFrameUploadCount_ > 0 && pageBufferState_ != ResourceState::TransferDestination) {
        if (auto commandResult = transitionBuffer(commandBuffer, *pageBuffer_, pageBufferState_, ResourceState::TransferDestination); !commandResult) { return commandResult; }
    }
    return {};
}

MeshletStreamUserPush MeshletStreamRuntime::userPush() const
{
    return MeshletStreamUserPush{
        .pageBuffer = pageHandle_.shaderIndex(),
        .activeGroupBuffer = activeGroupHandle_.shaderIndex(),
        .pageTableBuffer = pageTableHandle_.shaderIndex(),
        .paramsBuffer = paramsHandle_.shaderIndex(),
        .requestBuffer = requestHandle_.shaderIndex(),
        .residentPageBuffer = !residentPageFrames_.empty()
            ? residentPageFrames_[frameIndex_ % residentPageFrames_.size()].handle.shaderIndex()
            : 0u,
        .updateBuffer = updatePass_ != nullptr ? updatePass_->updateHandle().shaderIndex() : 0u,
        .activeHeaderBuffer = activeHeaderHandle_.shaderIndex(),
        .instanceBuffer = instanceHandle_.shaderIndex(),
        .primitiveBuffer = primitiveHandle_.shaderIndex(),
        .lodLevelBuffer = lodLevelHandle_.shaderIndex(),
        .groupBuffer = groupHandle_.shaderIndex(),
        .nodeBuffer = nodeHandle_.shaderIndex(),
        .drawIndirectBuffer = drawIndirectHandle_.shaderIndex(),
        .traversalHeaderBuffer = traversalHeaderHandle_.shaderIndex(),
        .traversalWorkBuffer = traversalWorkHandle_.shaderIndex(),
        .clasAddressBuffer = clasAddressHandle_.valid() ? clasAddressHandle_.shaderIndex() : 0u,
        .clasPageTableBuffer = clasPageTableHandle_.valid() ? clasPageTableHandle_.shaderIndex() : 0u,
        .blasHeaderBuffer = blasHeaderHandle_.valid() ? blasHeaderHandle_.shaderIndex() : 0u,
        .instanceBlasBuffer = instanceBlasHandle_.valid() ? instanceBlasHandle_.shaderIndex() : 0u,
        .blasBuildInfoBuffer = blasBuildInfoHandle_.valid() ? blasBuildInfoHandle_.shaderIndex() : 0u,
        .blasClusterReferenceBuffer = blasClusterReferenceHandle_.valid()
            ? blasClusterReferenceHandle_.shaderIndex()
            : 0u,
        .fallbackBlasAddressBuffer = fallbackBlasAddressHandle_.valid()
            ? fallbackBlasAddressHandle_.shaderIndex()
            : 0u,
        .dynamicBlasAddressBuffer = dynamicBlasAddressHandle_.valid()
            ? dynamicBlasAddressHandle_.shaderIndex()
            : 0u,
        .tlasInstanceBuffer = tlasInstanceHandle_.valid() ? tlasInstanceHandle_.shaderIndex() : 0u,
        .rasterBindingsBuffer = rasterBindingsHandle_.shaderIndex(),
        .clasPublicationRevision = clasPool_ ? uint32_t(clasPool_->stats().publicationRevision) : 0u,
    };
}

Result<> MeshletStreamRuntime::updateRasterBindings(
    const MeshletStreamGPURasterBindings& bindings)
{
    if (rasterBindingsBuffer_ == nullptr || !visibleClusterHandle_.valid()) {
        return makeError(Error::InvalidArgument);
    }
    MeshletStreamGPURasterBindings resolved = bindings;
    resolved.visibleClusterBuffer = visibleClusterHandle_.shaderIndex();
    return updateHostBuffer(*rasterBindingsBuffer_, &resolved, sizeof(resolved));
}

Result<> MeshletStreamRuntime::cmdPrepareVisibility(CommandBuffer& commandBuffer)
{
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    if (auto commandResult = transitionBuffer(
        commandBuffer,
        *drawIndirectBuffer_,
        drawIndirectBufferState_,
        ResourceState::IndirectArgument); !commandResult) { return commandResult; }
    if (auto commandResult = transitionBuffer(
        commandBuffer,
        *visibleClusterBuffer_,
        visibleClusterBufferState_,
        ResourceState::General,
        visibleClusterBufferState_ == ResourceState::General); !commandResult) { return commandResult; }
    return {};
}

Result<> MeshletStreamRuntime::cmdPrepareDeferred(CommandBuffer& commandBuffer)
{
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    if (auto commandResult = transitionBuffer(
        commandBuffer,
        *visibleClusterBuffer_,
        visibleClusterBufferState_,
        ResourceState::ShaderRead,
        true); !commandResult) { return commandResult; }
    return {};
}

MeshletStreamDeferredGPUResourcesView MeshletStreamRuntime::deferredGpuResources() const
{
    return MeshletStreamDeferredGPUResourcesView{
        .instanceBuffer = instanceBuffer_.get(),
        .pageBuffer = pageBuffer_.get(),
        .activeGroupBuffer = activeGroupBuffer_.get(),
        .pageTableBuffer = pageTableBuffer_.get(),
        .activeHeaderBuffer = activeHeaderBuffer_.get(),
        .paramsBuffer = paramsBuffer_.get(),
        .visibleClusterBuffer = visibleClusterBuffer_.get(),
        .visibleRecordCapacity = visibleClusterCapacity(),
        .accelerationStructure = accelerationStructure(),
    };
}

uint32_t MeshletStreamRuntime::visibleClusterCapacity() const
{
    if (maxActiveGroups_ == 0 || maxActiveGroupClusters_ == 0) {
        return 0;
    }
    const uint64_t count = static_cast<uint64_t>(maxActiveGroups_) * maxActiveGroupClusters_;
    return visibilityRecordCapacityFitsId(count) ? static_cast<uint32_t>(count) : 0u;
}

uint32_t MeshletStreamRuntime::drawTaskCount() const
{
    const uint32_t clusterCapacity = visibleClusterCapacity();
    if (clusterCapacity == 0) {
        return 0;
    }
    const uint64_t count = static_cast<uint64_t>(clusterCapacity) *
        kMeshletStreamTriangleChunkCount;
    return count > std::numeric_limits<uint32_t>::max() ? 0u : static_cast<uint32_t>(count);
}

Result<> MeshletStreamRuntime::syncRuntimeScene(const scene::Scene& scene, std::string& log)
{
    const std::span<const scene::MeshletStreamInstanceInfo> instances = asset_.instances();
    std::vector<uint32_t> runtimeRenderNodeIndices(instances.size());
    for (size_t index = 0; index < instances.size(); ++index) {
        runtimeRenderNodeIndices[index] = instances[index].renderNodeIndex;
    }
    return syncRuntimeScene(scene, runtimeRenderNodeIndices, log);
}

Result<> MeshletStreamRuntime::syncRuntimeScene(
    const scene::Scene& scene,
    std::span<const uint32_t> runtimeRenderNodeIndices,
    std::string& log)
{
    log.clear();
    if (!ready() || !scene.valid() || instanceBuffer_ == nullptr) {
        log = "MeshletStreamRuntime is not ready for a scene transform update.";
        return makeError(Error::InvalidArgument);
    }
    const std::span<const scene::MeshletStreamInstanceInfo> instances = asset_.instances();
    if (runtimeRenderNodeIndices.size() != instances.size()) {
        log = "MeshletStreamRuntime runtime render-node mapping does not match the stream instances.";
        return makeError(Error::InvalidArgument);
    }
    if (sceneResourceIdentity_ == scene.resourceIdentity() &&
        sceneTransformRevision_ == scene.transformRevision() &&
        sceneVisibilityRevision_ == scene.visibilityRevision() &&
        runtimeRenderNodeIndices_.size() == runtimeRenderNodeIndices.size() &&
        std::equal(
            runtimeRenderNodeIndices.begin(),
            runtimeRenderNodeIndices.end(),
            runtimeRenderNodeIndices_.begin())) {
        return {};
    }

    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
    if (instances.size() * sizeof(MeshletStreamGPUInstance) != instanceBuffer_->desc().size) {
        log = "MeshletStreamRuntime instance layout changed.";
        return makeError(Error::InvalidArgument);
    }

    std::vector<MeshletStreamGPUInstance> gpuInstances(instances.size());
    scene::Bounds updatedBounds;
    for (size_t index = 0; index < instances.size(); ++index) {
        const scene::MeshletStreamInstanceInfo& instance = instances[index];
        const uint32_t runtimeRenderNodeIndex = runtimeRenderNodeIndices[index];
        if (runtimeRenderNodeIndex >= scene.renderNodes().size() ||
            instance.primitiveIndex >= primitives.size()) {
            log = "MeshletStreamRuntime instance mapping does not match the runtime scene.";
            return makeError(Error::InvalidArgument);
        }
        const scene::RenderNode& renderNode = scene.renderNodes()[runtimeRenderNodeIndex];
        MeshletStreamGPUInstance& gpuInstance = gpuInstances[index];
        gpuInstance = MeshletStreamGPUInstance{};
        gpuInstance.primitiveIndex = instance.primitiveIndex;
        // The cook owns shared geometry; the runtime node owns the binding.
        // Composed scenes rebase material indices independently of the cache.
        gpuInstance.materialIndex = static_cast<uint32_t>(std::max(renderNode.materialIndex, 0));
        gpuInstance.visible = renderNode.visible ? 1u : 0u;
        gpuInstance.gpuSceneInstanceIndex = index < gpuSceneInstanceMapping_.size()
            ? gpuSceneInstanceMapping_[index]
            : kMeshletStreamInvalidClusterIndex;
        for (uint32_t row = 0; row < 4; ++row) {
            gpuInstance.world0[row] = renderNode.worldMatrix.a[0 + row];
            gpuInstance.world1[row] = renderNode.worldMatrix.a[4 + row];
            gpuInstance.world2[row] = renderNode.worldMatrix.a[8 + row];
            gpuInstance.world3[row] = renderNode.worldMatrix.a[12 + row];
        }
        scene::Bounds worldBounds;
        includeTransformedBounds(
            worldBounds,
            primitives[instance.primitiveIndex].bounds,
            renderNode.worldMatrix.a);
        if (worldBounds.valid) {
            updatedBounds.include(worldBounds.min);
            updatedBounds.include(worldBounds.max);
        }
        const float3 center = worldBounds.valid ? worldBounds.center() : drawBounds_.center();
        gpuInstance.boundsCenterRadius[0] = center.x;
        gpuInstance.boundsCenterRadius[1] = center.y;
        gpuInstance.boundsCenterRadius[2] = center.z;
        gpuInstance.boundsCenterRadius[3] = std::max(
            worldBounds.valid ? worldBounds.radius() : drawBounds_.radius(),
            0.001f);
    }

    void* mapped = instanceBuffer_->map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    std::memcpy(mapped, gpuInstances.data(), static_cast<size_t>(instanceBuffer_->desc().size));
    instanceBuffer_->flush({0, instanceBuffer_->desc().size});
    instanceBuffer_->unmap();
    if (updatedBounds.valid) {
        drawBounds_ = updatedBounds;
    }
    sceneTransformRevision_ = scene.transformRevision();
    sceneVisibilityRevision_ = scene.visibilityRevision();
    sceneResourceIdentity_ = scene.resourceIdentity();
    runtimeRenderNodeIndices_.assign(
        runtimeRenderNodeIndices.begin(),
        runtimeRenderNodeIndices.end());
    return {};
}

Result<> MeshletStreamRuntime::syncGPUSceneInstanceMapping(std::span<const uint32_t> mapping)
{
    if (!ready() || instanceBuffer_ == nullptr || mapping.size() != asset_.instances().size()) {
        return makeError(Error::InvalidArgument);
    }
    if (gpuSceneInstanceMapping_.size() == mapping.size() &&
        std::equal(mapping.begin(), mapping.end(), gpuSceneInstanceMapping_.begin())) {
        return {};
    }

    void* mapped = instanceBuffer_->map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    auto* gpuInstances = static_cast<MeshletStreamGPUInstance*>(mapped);
    for (size_t index = 0; index < mapping.size(); ++index) {
        gpuInstances[index].gpuSceneInstanceIndex = mapping[index];
    }
    instanceBuffer_->flush({0, instanceBuffer_->desc().size});
    instanceBuffer_->unmap();
    gpuSceneInstanceMapping_.assign(mapping.begin(), mapping.end());
    return {};
}

Result<> MeshletStreamRuntime::cmdDrawMeshTasks(CommandBuffer& commandBuffer, bool tessellation) const
{
    if (!ready() || drawTaskCount() == 0) { return {}; }
    auto result = prepareImmutableMetadataRead(commandBuffer);
    return result ? (*drawIndirectBuffer_).slice({tessellation ? sizeof(MeshletStreamGPUDrawIndirect) : 0u, 12}).and_then([&](const auto& bufferSlice) { return commandBuffer.drawMeshTasksIndirect(bufferSlice); })
                  : result;
}

uint32_t MeshletStreamRuntime::computeMaxActiveGroups(uint32_t capacity) const
{
    if (capacity == 0) {
        return 0;
    }
    uint64_t total = 0;
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
    for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
        if (instance.primitiveIndex >= primitives.size()) {
            continue;
        }
        const scene::MeshletStreamPrimitiveInfo& primitive = primitives[instance.primitiveIndex];
        total += primitive.groupCount;
        if (total >= capacity) {
            return capacity;
        }
    }
    return static_cast<uint32_t>(total);
}

uint32_t MeshletStreamRuntime::computeMaxPrimitiveGroups() const
{
    uint32_t maxGroups = 0;
    for (const scene::MeshletStreamPrimitiveInfo& primitive : asset_.primitives()) {
        maxGroups = std::max(maxGroups, primitive.groupCount);
    }
    return maxGroups;
}

Result<> MeshletStreamRuntime::initializeSceneMetadataBuffers(Device& device, std::string& log)
{
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
    const std::span<const scene::MeshletStreamInstanceInfo> instances = asset_.instances();
    gpuSceneInstanceMapping_.assign(instances.size(), kMeshletStreamInvalidClusterIndex);
    Result<> result = createAndPopulateHostStorageBuffer<MeshletStreamGPUInstance>(
        device,
        instances.size(),
        instanceBuffer_,
        log,
        "MeshletStreamRuntime instances",
        [this, primitives, instances](MeshletStreamGPUInstance& gpuInstance, size_t index) {
            const scene::MeshletStreamInstanceInfo& instance = instances[index];
            gpuInstance = MeshletStreamGPUInstance{};
            gpuInstance.primitiveIndex = instance.primitiveIndex;
            gpuInstance.materialIndex = instance.materialIndex;
            gpuInstance.visible = instance.visible;
            gpuInstance.gpuSceneInstanceIndex = gpuSceneInstanceMapping_[index];
            for (uint32_t row = 0; row < 4; ++row) {
                gpuInstance.world0[row] = instance.worldMatrix[0 + row];
                gpuInstance.world1[row] = instance.worldMatrix[4 + row];
                gpuInstance.world2[row] = instance.worldMatrix[8 + row];
                gpuInstance.world3[row] = instance.worldMatrix[12 + row];
            }
            scene::Bounds worldBounds;
            if (instance.primitiveIndex < primitives.size()) {
                includeTransformedBounds(
                    worldBounds,
                    primitives[instance.primitiveIndex].bounds,
                    instance.worldMatrix);
            }
            const float3 center = worldBounds.valid ? worldBounds.center() : drawBounds_.center();
            const float radius = std::max(
                worldBounds.valid ? worldBounds.radius() : drawBounds_.radius(),
                0.001f);
            gpuInstance.boundsCenterRadius[0] = center.x;
            gpuInstance.boundsCenterRadius[1] = center.y;
            gpuInstance.boundsCenterRadius[2] = center.z;
            gpuInstance.boundsCenterRadius[3] = radius;
        });
    if (!result) {
        return result;
    }

    const std::span<const scene::MeshletStreamLODLevelInfo> lodLevels = asset_.lodLevels();
    result = createAndPopulateHostStorageBuffer<MeshletStreamGPULODLevel>(
        device,
        lodLevels.size(),
        lodLevelBuffer_,
        log,
        "MeshletStreamRuntime LOD levels",
        [lodLevels](MeshletStreamGPULODLevel& gpuLod, size_t index) {
            const scene::MeshletStreamLODLevelInfo& lod = lodLevels[index];
            gpuLod = MeshletStreamGPULODLevel{
                .pageOffset = lod.pageOffset,
                .pageCount = lod.pageCount,
                .lodLevel = lod.lodLevel,
                .clusterCount = lod.clusterCount,
                .minBoundingSphereRadius = lod.minBoundingSphereRadius,
                .minMaxQuadricError = lod.minMaxQuadricError,
            };
        });
    if (!result) {
        return result;
    }

    const auto refinedGroups = asset_.refinedGroups();
    std::vector<uint32_t> topology(refinedGroups.begin(), refinedGroups.end());
    std::vector<std::vector<uint32_t>> parents(asset_.groupCount());
    for (uint32_t owner = 0; owner < asset_.groupCount(); ++owner) {
        const auto& group = asset_.groups()[owner];
        for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
            const uint32_t refined = refinedGroups[group.clusterRefinedOffset + cluster];
            if (refined != UINT32_MAX) {
                parents[refined].push_back(owner);
            }
        }
    }
    std::vector<uint32_t> parentOffsets(parents.size());
    for (size_t index = 0; index < parents.size(); ++index) {
        auto& list = parents[index];
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        if (topology.size() + list.size() > UINT32_MAX) {
            log = "MeshletStreamRuntime LOD topology exceeds 32-bit addressing";
            return makeError(Error::InvalidArgument);
        }
        parentOffsets[index] = static_cast<uint32_t>(topology.size());
        topology.insert(topology.end(), list.begin(), list.end());
    }
    // The BVH is derived from resident v8/v9 metadata once; geometry payloads
    // and the on-disk asset format stay independent of selection acceleration.
    std::vector<uint32_t> bvhOffsets(primitives.size()), bvhCounts(primitives.size()), tileOffsets(primitives.size());
    std::vector<MeshletLODGroupRecord> lodGroups;
    std::vector<MeshletLODBVHNode> bvh;
    std::vector<std::vector<uint32_t>> demandRoots(primitives.size());
    for (size_t index = 0; index < primitives.size(); ++index) {
        const auto& primitive = primitives[index];
        lodGroups.resize(primitive.groupCount);
        for (uint32_t local = 0; local < primitive.groupCount; ++local) {
            const auto& group = asset_.groups()[primitive.groupOffset + local];
            auto& metric = lodGroups[local];
            std::copy_n(group.boundsCenterRadius, 4, metric.sphere.begin());
            metric.error = group.maxQuadricError;
            metric.level = group.lodLevel;
            metric.flags = group.flags;
        }
        if (!buildMeshletLodBvh(lodGroups, bvh, log)) {
            log = "MeshletStreamRuntime LOD BVH: " + log;
            return makeError(Error::InvalidArgument);
        }
        constexpr size_t kNodeWords = sizeof(MeshletLODBVHNode) / sizeof(uint32_t);
        const uint64_t wordCount = bvh.size() * uint64_t(kNodeWords);
        if (topology.size() + wordCount > UINT32_MAX) {
            log = "MeshletStreamRuntime LOD BVH exceeds 32-bit addressing";
            return makeError(Error::InvalidArgument);
        }
        bvhOffsets[index] = static_cast<uint32_t>(topology.size());
        bvhCounts[index] = static_cast<uint32_t>(bvh.size());
        topology.resize(topology.size() + static_cast<size_t>(wordCount));
        if (!bvh.empty()) {
            std::memcpy(topology.data() + bvhOffsets[index], bvh.data(), bvh.size() * sizeof(MeshletLODBVHNode));
        }
        if (!buildMeshletLodTiles(lodGroups, bvh, log)) { return makeError(Error::InvalidArgument); }
        demandRoots[index] = buildMeshletLodDemandRoots(bvh);
        const uint64_t tileWords = bvh.size() * uint64_t(kNodeWords);
        if (topology.size() + 1u + tileWords + bvh.size() > UINT32_MAX) {
            log = "MeshletStreamRuntime cooperative tiles exceed 32-bit addressing";
            return makeError(Error::InvalidArgument);
        }
        tileOffsets[index] = static_cast<uint32_t>(topology.size());
        topology.push_back(static_cast<uint32_t>(bvh.size()));
        topology.resize(topology.size() + static_cast<size_t>(tileWords));
        if (!bvh.empty()) { std::memcpy(topology.data() + tileOffsets[index] + 1, bvh.data(), tileWords * sizeof(uint32_t)); }
        const auto tileParents = buildMeshletLodTileParents(bvh);
        topology.insert(topology.end(), tileParents.begin(), tileParents.end());
    }
    result = createAndPopulateHostStorageBuffer<MeshletStreamGPUPrimitive>(
        device, primitives.size(), primitiveBuffer_, log, "MeshletStreamRuntime primitives",
        [primitives, &bvhOffsets, &bvhCounts, &tileOffsets](MeshletStreamGPUPrimitive& gpuPrimitive, size_t index) {
            const auto& primitive = primitives[index];
            gpuPrimitive = MeshletStreamGPUPrimitive{
                .lodLevelOffset = primitive.lodLevelOffset,
                .lodLevelCount = primitive.lodLevelCount,
                .pageOffset = primitive.pageOffset,
                .pageCount = primitive.pageCount,
                .fallbackPageOffset = primitive.fallbackPageOffset,
                .fallbackPageCount = primitive.fallbackPageCount,
                .groupOffset = primitive.groupOffset,
                .groupCount = primitive.groupCount,
                .fallbackGroupOffset = primitive.fallbackGroupOffset,
                .fallbackGroupCount = primitive.fallbackGroupCount,
                .materialIndex = primitive.materialIndex,
                .nodeOffset = primitive.nodeOffset,
                .nodeCount = primitive.nodeCount,
                .lodBvhOffset = bvhOffsets[index],
                .lodBvhNodeCount = bvhCounts[index],
                .lodTileOffset = tileOffsets[index],
            };
        });
    if (!result) { return result; }

    lodInstanceOffsetsOffset_ = static_cast<uint32_t>(topology.size());
    uint64_t stateWords = instances.size() * 4ull;
    lodTransitionHistoryBytes_ = 0;
    for (const auto& instance : instances) {
        topology.push_back(static_cast<uint32_t>(stateWords));
        if (instance.primitiveIndex < primitives.size()) {
            // Active/mask pairs, sparse header and the previous/current active
            // IDs. Only previous active entries need clearing each frame.
            const uint64_t groupCount = primitives[instance.primitiveIndex].groupCount;
            stateWords += 4ull + groupCount * 3ull;
            if (lodTransitionTelemetry_) {
                const uint64_t historyWords = 1ull + groupCount * 2ull;
                stateWords += historyWords;
                lodTransitionHistoryBytes_ += historyWords * sizeof(uint32_t);
            }
        }
        if (stateWords > UINT32_MAX || topology.size() > UINT32_MAX) {
            log = "MeshletStreamRuntime LOD frontier exceeds 32-bit addressing";
            return makeError(Error::InvalidArgument);
        }
    }
    demandInstanceOffsetsOffset_ = static_cast<uint32_t>(topology.size());
    uint64_t demandBits = 0, demandTasks = 0;
    for (const auto& instance : instances) {
        topology.push_back(static_cast<uint32_t>(demandBits));
        if (instance.primitiveIndex < primitives.size()) { demandBits += primitives[instance.primitiveIndex].groupCount; }
        topology.push_back(static_cast<uint32_t>(demandBits));
        if (instance.primitiveIndex < primitives.size()) { demandBits += topology[tileOffsets[instance.primitiveIndex]]; }
        topology.push_back(static_cast<uint32_t>(demandTasks));
        const uint32_t count = instance.primitiveIndex < demandRoots.size()
            ? static_cast<uint32_t>(demandRoots[instance.primitiveIndex].size()) : 0u;
        topology.push_back(count);
        demandTasks += count;
        if (demandBits > UINT32_MAX || topology.size() > UINT32_MAX) {
            log = "MeshletStreamRuntime demand topology exceeds 32-bit addressing";
            return makeError(Error::InvalidArgument);
        }
    }
    // Store immutable task templates; a separate GPU dispatch culls and
    // publishes them each frame before workers start claiming queue ranges.
    if (distributedPageDemand_ && (demandTasks > traversalWorkCapacity_ || topology.size() + demandTasks > UINT32_MAX)) {
        spdlog::warn("[MeshletStreamRuntime] Demand task budget exceeded ({} > {}); using ordered demand", demandTasks, traversalWorkCapacity_);
        distributedPageDemand_ = false;
    }
    demandTaskOffset_ = static_cast<uint32_t>(topology.size());
    if (distributedPageDemand_) {
        for (uint32_t instance = 0; instance < instances.size(); ++instance) {
            if (instances[instance].primitiveIndex >= demandRoots.size()) { continue; }
            for (uint32_t root : demandRoots[instances[instance].primitiveIndex]) {
                topology.push_back(root);
            }
        }
        demandTaskCount_ = static_cast<uint32_t>(demandTasks);
    }
    result = createNamedBuffer(device, BufferDesc{
        .size = (kMeshletStreamDemandStatsWords + (distributedPageDemand_ ? (demandBits + 31u) / 32u : 0u)) * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::Device,
    }, demandBuffer_, log, "MeshletStreamRuntime demand bits and statistics");
    if (!result) { return result; }
    result = createAndPopulateImmutableStorageBuffer<uint32_t>(device, deviceImmutableMetadata_, topology.size(),
        lodTopologyBuffer_, immutableMetadataUpload_ ? &immutableMetadataUpload_->topology : nullptr,
        log, "MeshletStreamRuntime LOD topology",
        [&topology](uint32_t& word, size_t index) { word = topology[index]; });
    if (!result) {
        return result;
    }
    result = createNamedBuffer(device, BufferDesc{
        .size = std::max<uint64_t>(stateWords, 1u) * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::Device,
    }, lodStateBuffer_, log, "MeshletStreamRuntime LOD frontier");
    if (!result) {
        return result;
    }

    const std::span<const scene::MeshletStreamGroupInfo> groups = asset_.groups();
    std::vector<MeshletLODRefinementBounds> refinementBounds;
    if (viewDrivenPageDemand_) {
        std::vector<MeshletLODGroupRecord> metrics(groups.size());
        std::vector<MeshletLODGroupRange> ranges(groups.size());
        for (size_t i = 0; i < groups.size(); ++i) {
            std::copy_n(groups[i].boundsCenterRadius, 4, metrics[i].sphere.begin());
            metrics[i].error = groups[i].maxQuadricError;
            metrics[i].flags = groups[i].flags;
            ranges[i] = {groups[i].clusterRefinedOffset, groups[i].clusterCount};
        }
        if (!buildMeshletLodRefinementBounds(metrics, ranges, refinedGroups, refinementBounds, log)) {
            return makeError(Error::InvalidArgument);
        }
    }
    result = createAndPopulateImmutableStorageBuffer<MeshletStreamGPUGroup>(
        device,
        deviceImmutableMetadata_,
        groups.size(),
        groupBuffer_,
        immutableMetadataUpload_ ? &immutableMetadataUpload_->groups : nullptr,
        log,
        "MeshletStreamRuntime groups",
        [groups, &parents, &parentOffsets, &refinementBounds](MeshletStreamGPUGroup& gpuGroup, size_t index) {
            const scene::MeshletStreamGroupInfo& group = groups[index];
            gpuGroup = MeshletStreamGPUGroup{
                .primitiveIndex = group.primitiveIndex,
                .pageIndex = group.pageIndex,
                .lodLevel = group.lodLevel,
                .clusterCount = group.clusterCount,
                .maxQuadricError = group.maxQuadricError,
                .clusterRefinedOffset = group.clusterRefinedOffset,
                .flags = group.flags,
                .parentOffset = parentOffsets[index],
                .parentCount = static_cast<uint32_t>(parents[index].size()),
                .refinementBounds = refinementBounds.empty() ? MeshletLODRefinementBounds{} : refinementBounds[index],
            };
            std::copy(
                std::begin(group.boundsCenterRadius),
                std::end(group.boundsCenterRadius),
                std::begin(gpuGroup.boundsCenterRadius));
        });
    if (!result) {
        return result;
    }

    const std::span<const scene::MeshletStreamNodeInfo> nodes = asset_.nodes();
    result = createAndPopulateHostStorageBuffer<MeshletStreamGPUNode>(
        device,
        nodes.size(),
        nodeBuffer_,
        log,
        "MeshletStreamRuntime hierarchy nodes",
        [nodes](MeshletStreamGPUNode& gpuNode, size_t index) {
            const scene::MeshletStreamNodeInfo& node = nodes[index];
            gpuNode = MeshletStreamGPUNode{
                .primitiveIndex = node.primitiveIndex,
                .childOffset = node.childOffset,
                .childCount = node.childCount,
                .groupIndex = node.groupIndex,
                .maxQuadricError = node.maxQuadricError,
                .lodLevel = node.lodLevel,
            };
            std::copy(
                std::begin(node.boundsCenterRadius),
                std::end(node.boundsCenterRadius),
                std::begin(gpuNode.boundsCenterRadius));
        });
    if (!result || !immutableMetadataUpload_) { return result; }
    auto& upload = *immutableMetadataUpload_;
    upload.totalBytes = groupBuffer_->desc().size + lodTopologyBuffer_->desc().size;
    std::unique_ptr<Buffer> staging;
    result = createNamedBuffer(device, BufferDesc{
        .size = std::min(upload.totalBytes, kImmutableMetadataUploadBatchBytes),
        .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload,
        .memoryDomain = MemoryBudgetDomain::Upload,
    }, staging, log, "MeshletStreamRuntime immutable metadata staging");
    if (!result) { return result; }
    upload.stagingPeakBytes = staging->desc().size;
    upload.stagingAllocatedBytes = staging->memoryInfo().sizeBytes;
    upload.staging = std::move(staging);
    return {};
}

Result<> MeshletStreamRuntime::initializePageTableIfNeeded(CommandBuffer& commandBuffer)
{
    if (*pageTableInitialized_) {
        return {};
    }

    Result<> result = updatePass_->initializePageTable(
        commandBuffer,
        userPush(),
        asset_.pageCount(),
        *pageTableBuffer_,
        pageTableState_);
    if (!result) {
        return result;
    }
    // Reconstruct CPU-confirmed state after an abandoned recording as well as
    // on first use. Capture only the shared validity flag, never a runtime pointer.
    result = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(nullptr,
        [valid = pageTableInitialized_, cache = sceneReadinessCache_] {
            *valid = false;
            cache->valid = false;
            cache->value.ready = false;
        }));
    if (!result) { return result; }
    residency_.rebuildPendingPatches();
    *pageTableInitialized_ = true;
    sceneReadinessCache_->valid = false;
    return {};
}

Result<> MeshletStreamRuntime::applyPageTablePatches(CommandBuffer& commandBuffer)
{
    std::vector<StreamPageTablePatch> patches;
    currentFrameOrderedUploadCount_ = residency_.buildOrderedUploadPatches(commandBuffer, patches);
    if (!patches.empty()) {
        const auto tracked = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(nullptr,
            [valid = pageTableInitialized_, cache = sceneReadinessCache_] {
                *valid = false;
                cache->valid = false;
                cache->value.ready = false;
            }));
        if (!tracked) { return tracked; }
    }
    Result<> result = updatePass_->apply(
        commandBuffer,
        userPush(),
        patches,
        maxUpdatePatches_,
        frameIndex_,
        *pageTableBuffer_,
        pageTableState_);
    if (!result) {
        return result;
    }
    residency_.clearPendingPatches();
    return {};
}

Result<> MeshletStreamRuntime::clearRequestBuffer(CommandBuffer& commandBuffer)
{
    const StreamRequestBufferHeader clearHeader{
        .maxLoadRequests = maxGpuPageRequests_,
        .maxUnloadRequests = maxGpuPageUnloadRequests_,
        .frameIndex = frameIndex_,
        .loadPriorityOffset = screenSpacePagePriority_
            ? kStreamRequestHeaderWordCount + maxGpuPageRequests_ + maxGpuPageUnloadRequests_ : 0u,
        .priorityTableOffset = screenSpacePagePriority_
            ? kStreamRequestHeaderWordCount + 2u * maxGpuPageRequests_ + maxGpuPageUnloadRequests_ : 0u,
        .prefetchRequestLimit = prefetchPages_ ? std::min(maxGpuPageRequests_ / 4u, residency_.availablePrefetchRequests()) : 0u,
    };
    Result<> result = updateHostBuffer(*requestClearBuffer_, &clearHeader, sizeof(clearHeader));
    if (!result) {
        return result;
    }

    if (auto commandResult = transitionBuffer(commandBuffer, *requestBuffer_, requestBufferState_, ResourceState::TransferDestination); !commandResult) { return commandResult; }
    {
        auto sourceSlice = requestClearBuffer_.get()->slice({0, sizeof(StreamRequestBufferHeader)});
        if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
        auto destinationSlice = requestBuffer_.get()->slice({0, sizeof(StreamRequestBufferHeader)});
        if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
        if (auto commandResult = commandBuffer.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
    }
    if (auto commandResult = transitionBuffer(commandBuffer, *requestBuffer_, requestBufferState_, ResourceState::General); !commandResult) { return commandResult; }
    return {};
}

Result<> MeshletStreamRuntime::copyRequestBufferForReadback(CommandBuffer& commandBuffer)
{
    Buffer* readback = requestReadbackBuffer_.get();
    if (auto* frame = metallic::render::RenderFrameContext::from(commandBuffer)) {
        RequestReadback* slot = nullptr;
        for (auto& candidate : requestReadbacks_) {
            if (!candidate.submission || candidate.submission->cancelled() ||
                (candidate.submission->resolved() && candidate.completion.isSubmitted() && candidate.completion.isComplete())) {
                if (!slot || candidate.frame < slot->frame) { slot = &candidate; }
            }
        }
        // Bounded backpressure: retain prior feedback rather than wait or race.
        if (!slot) { return {}; }
        auto transaction = std::make_shared<SubmissionTransaction>(nullptr, nullptr);
        const auto result = commandBuffer.addSubmissionTransaction(transaction);
        if (!result) { return result; }
        slot->submission = std::move(transaction);
        slot->completion = frame->completion();
        slot->frame = frameIndex_;
        readback = slot->buffer.get();
    }
    if (auto commandResult = transitionBuffer(commandBuffer, *requestBuffer_, requestBufferState_, ResourceState::TransferSource); !commandResult) { return commandResult; }
    {
        auto sourceSlice = requestBuffer_.get()->slice({0, readback->desc().size - kMeshletStreamDemandStatsWords * sizeof(uint32_t) - sizeof(MeshletStreamGPUBLASHeader)});
        if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
        auto destinationSlice = readback->slice({0, readback->desc().size - kMeshletStreamDemandStatsWords * sizeof(uint32_t) - sizeof(MeshletStreamGPUBLASHeader)});
        if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
        if (auto commandResult = commandBuffer.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
    }
    if (distributedPageDemand_ || lodTransitionTelemetry_) {
        // Piggyback the compact statistics header on completed request feedback;
        // no additional CPU wait or GPU-to-CPU submission is introduced.
        if (auto commandResult = transitionBuffer(commandBuffer, *demandBuffer_, demandBufferState_, ResourceState::TransferSource); !commandResult) { return commandResult; }
        {
            auto sourceSlice = demandBuffer_.get()->slice({0, kMeshletStreamDemandStatsWords * sizeof(uint32_t)});
            if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
            auto destinationSlice = readback->slice({readback->desc().size - kMeshletStreamDemandStatsWords * sizeof(uint32_t) - sizeof(MeshletStreamGPUBLASHeader), kMeshletStreamDemandStatsWords * sizeof(uint32_t)});
            if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
            if (auto commandResult = commandBuffer.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
        }
    }
    if (clusterRtxEnabled_ && blasHeaderBuffer_) {
        if (auto commandResult = transitionBuffer(commandBuffer, *blasHeaderBuffer_, blasHeaderBufferState_, ResourceState::TransferSource); !commandResult) { return commandResult; }
        {
            auto sourceSlice = blasHeaderBuffer_.get()->slice({0, sizeof(MeshletStreamGPUBLASHeader)});
            if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
            auto destinationSlice = readback->slice({readback->desc().size - sizeof(MeshletStreamGPUBLASHeader), sizeof(MeshletStreamGPUBLASHeader)});
            if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
            if (auto commandResult = commandBuffer.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
        }
    }
    requestReadbackValid_ = metallic::render::RenderFrameContext::from(commandBuffer) == nullptr;
    return {};
}

Result<> MeshletStreamRuntime::updateParamsBuffer(const MeshletStreamFrameDesc& frame)
{
    MeshletStreamGPUParams params;
    const uint32_t width = std::max(frame.width, 1u);
    const uint32_t height = std::max(frame.height, 1u);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    params.eye[0] = finiteOr(frame.camera.eye.x, 0.0f);
    params.eye[1] = finiteOr(frame.camera.eye.y, 0.0f);
    params.eye[2] = finiteOr(frame.camera.eye.z, 0.0f);
    params.eye[3] = 1.0f;
    params.center[0] = finiteOr(frame.camera.center.x, 0.0f);
    params.center[1] = finiteOr(frame.camera.center.y, 0.0f);
    params.center[2] = finiteOr(frame.camera.center.z, 0.0f);
    params.center[3] = 1.0f;
    params.upProjection[0] = finiteOr(frame.camera.up.x, 0.0f);
    params.upProjection[1] = finiteOr(frame.camera.up.y, 1.0f);
    params.upProjection[2] = finiteOr(frame.camera.up.z, 0.0f);
    params.upProjection[3] = frame.camera.orthographic ? 1.0f : 0.0f;
    params.viewport[0] = aspect;
    params.viewport[1] = static_cast<float>(width);
    params.viewport[2] = static_cast<float>(height);
    params.viewport[3] = finiteOr(frame.camera.fovDegrees, 60.0f) * 0.017453292519943295f;
    params.clipOrtho[0] = finiteOr(frame.camera.znear, 0.1f);
    params.clipOrtho[1] = finiteOr(frame.camera.zfar, 1000.0f);
    params.clipOrtho[2] = std::max(finiteOr(frame.camera.orthoHeight, 10.0f), 0.0001f);
    params.clipOrtho[3] = frame.camera.reversedZ ? 1.0f : 0.0f;
    params.clearColor[0] = 0.015f;
    params.clearColor[1] = 0.018f;
    params.clearColor[2] = 0.024f;
    params.clearColor[3] = 1.0f;
    params.debugColorMode = frame.debugColorMode;
    params.pageBufferBytes = static_cast<uint32_t>(residency_.maxResidentBytes());
    params.drawTaskCount = drawTaskCount();
    params.frameIndex = frameIndex_ == 0 ? 1u : frameIndex_;
    params.maxGpuPageRequests = maxGpuPageRequests_;
    params.maxGpuPageUnloadRequests = maxGpuPageUnloadRequests_;
    params.activeGroupCount = maxActiveGroups_;
    params.maxActiveGroupClusters = maxActiveGroupClusters_;
    params.sceneInstanceCount = asset_.instanceCount();
    params.scenePrimitiveCount = asset_.primitiveCount();
    params.sceneLodLevelCount = asset_.lodLevelCount();
    params.scenePageCount = asset_.pageCount();
    params.selectedLodLevel = frame.enableGpuLodSelection
        ? kMeshletStreamNoDebugLODOverride
        : frame.selectedLodLevel;
    params.enableGpuLodSelection = frame.enableGpuLodSelection ? 1u : 0u;
    params.lodPixelError = meshletLodRenderPixelThreshold(
        std::clamp(finiteOr(frame.lodPixelError, 1.5f), 0.05f, 16.0f) *
            std::exp2(std::clamp(finiteOr(frame.lodBias, 0.0f), -4.0f, 4.0f)),
        height, frame.displayHeight);
    params.lodTopologyBuffer = lodTopologyHandle_.shaderIndex();
    params.lodStateBuffer = lodStateHandle_.shaderIndex();
    const uint64_t threshold = uint64_t(distributedDemandMinGroups_) +
        (currentFrameDistributedDemand_ ? 0u : distributedDemandMinGroups_ / 8u);
    currentFrameDistributedDemand_ = distributedPageDemand_ &&
        (distributedDemandMinGroups_ == 0 || recentDemandGroupTests_ == UINT32_MAX || recentDemandGroupTests_ >= threshold);
    params.demandBuffer = currentFrameDistributedDemand_ ? demandHandle_.shaderIndex() : UINT32_MAX;
    params.demandStatsBuffer = distributedPageDemand_ || lodTransitionTelemetry_ ? demandHandle_.shaderIndex() : UINT32_MAX;
    params.demandTaskOffset = demandTaskOffset_;
    params.demandTaskCount = demandTaskCount_;
    params.demandInstanceOffsetsOffset = demandInstanceOffsetsOffset_;
    params.splitFrontier = currentFrameDistributedDemand_ ? 1u : 0u;
    params.lodInstanceOffsetsOffset = lodInstanceOffsetsOffset_;
    params.enableGpuUnloadRequests = 1u;
    params.sceneGroupCount = asset_.groupCount();
    params.maxPrimitiveGroupCount = maxPrimitiveGroupCount_;
    params.sceneNodeCount = asset_.nodeCount();
    params.traversalWorkerCount = traversalWorkerCount_;
    params.traversalWorkCapacity = traversalWorkCapacity_;
    params.blasClusterReferenceAddressLow = static_cast<uint32_t>(blasClusterReferenceAddress_);
    params.blasClusterReferenceAddressHigh = static_cast<uint32_t>(blasClusterReferenceAddress_ >> 32u);
    params.blasClusterReferenceCapacity = blasClusterReferenceCapacity_;
    params.blasBuildCapacity = blasBuildCapacity_;
    params.maxBlasClustersPerBuild = maxBlasClustersPerBuild_;
    params.blasStorageBytes = blasStorageBuffer_ ? static_cast<uint32_t>(blasStorageBuffer_->desc().size) : 0u;
    const uint64_t blasStorageAddress = blasStorageBuffer_ ? blasStorageBuffer_->deviceAddress() : 0u;
    params.blasStorageAddressLow = static_cast<uint32_t>(blasStorageAddress);
    params.blasStorageAddressHigh = static_cast<uint32_t>(blasStorageAddress >> 32u);
    params.enableBlasInstanceReuse = blasInstanceReuse_ ? 1u : 0u;
    std::copy(blasSizeClasses_.begin(), blasSizeClasses_.end(), params.blasSizeClasses);
    const auto& renderCamera = frame.useSeparateRenderCamera ? frame.renderCamera : frame.camera;
    params.renderEye[0] = finiteOr(renderCamera.eye.x, 0.0f);
    params.renderEye[1] = finiteOr(renderCamera.eye.y, 0.0f);
    params.renderEye[2] = finiteOr(renderCamera.eye.z, 0.0f);
    params.renderEye[3] = finiteOr(frame.jitterX, 0.0f);
    params.renderCenter[0] = finiteOr(renderCamera.center.x, 0.0f);
    params.renderCenter[1] = finiteOr(renderCamera.center.y, 0.0f);
    params.renderCenter[2] = finiteOr(renderCamera.center.z, 0.0f);
    params.renderCenter[3] = finiteOr(frame.jitterY, 0.0f);
    params.renderUpProjection[0] = finiteOr(renderCamera.up.x, 0.0f);
    params.renderUpProjection[1] = finiteOr(renderCamera.up.y, 1.0f);
    params.renderUpProjection[2] = finiteOr(renderCamera.up.z, 0.0f);
    params.renderUpProjection[3] = renderCamera.orthographic ? 1.0f : 0.0f;
    std::copy_n(params.viewport, 4, params.renderViewport);
    params.renderViewport[3] = finiteOr(renderCamera.fovDegrees, 60.0f) * 0.017453292519943295f;
    params.renderClipOrtho[0] = finiteOr(renderCamera.znear, 0.1f);
    params.renderClipOrtho[1] = finiteOr(renderCamera.zfar, 1000.0f);
    params.renderClipOrtho[2] = std::max(finiteOr(renderCamera.orthoHeight, 10.0f), 0.0001f);
    params.renderClipOrtho[3] = renderCamera.reversedZ ? 1.0f : 0.0f;
    const MeshletStreamGPUParams& previous = previousFrameParamsValid_
        ? previousFrameParams_
        : params;
    std::copy_n(previous.eye, 4u, params.previousEye);
    std::copy_n(previous.center, 4u, params.previousCenter);
    std::copy_n(previous.upProjection, 4u, params.previousUpProjection);
    std::copy_n(previous.viewport, 4u, params.previousViewport);
    std::copy_n(previous.clipOrtho, 4u, params.previousClipOrtho);
    params.previousLodPixelError = previous.lodPixelError;
    const bool moved = !previousFrameParamsValid_ || std::memcmp(params.eye, previous.eye, sizeof(params.eye)) != 0 ||
        std::memcmp(params.center, previous.center, sizeof(params.center)) != 0 ||
        std::memcmp(params.upProjection, previous.upProjection, sizeof(params.upProjection)) != 0 ||
        std::memcmp(params.viewport, previous.viewport, sizeof(params.viewport)) != 0 ||
        std::memcmp(params.clipOrtho, previous.clipOrtho, sizeof(params.clipOrtho)) != 0;
    const uint64_t now = meshletStreamTimeMicroseconds();
    const double deltaSeconds = previousPrefetchTime_ != 0 && now >= previousPrefetchTime_
        ? double(now - previousPrefetchTime_) * 1e-6 : 0;
    previousPrefetchTime_ = now;
    MeshletStreamPrefetchConfig forecastConfig;
    const double sceneRadius = std::max(double(finiteOr(drawBounds_.radius(), 1.f)), 0.001);
    forecastConfig.maxTranslationDistance = sceneRadius * 0.02;
    forecastConfig.teleportDistance = sceneRadius * 0.1;
    const MeshletStreamPrefetchCamera forecastCamera{
        .eye = {params.eye[0], params.eye[1], params.eye[2]},
        .center = {params.center[0], params.center[1], params.center[2]},
        .up = {params.upProjection[0], params.upProjection[1], params.upProjection[2]},
        .fovDegrees = finiteOr(frame.camera.fovDegrees, 60.f),
        .orthoHeight = params.clipOrtho[2], .orthographic = frame.camera.orthographic};
    prefetchForecast_ = prefetchPredictor_.update(forecastCamera, deltaSeconds,
        residency_.recentDemandLatency(now), !frame.enableGpuLodSelection, forecastConfig);
    currentFramePrefetch_ = prefetchPages_ && frame.enableGpuLodSelection && residency_.availablePrefetchRequests() != 0 &&
        (moved || recentGpuRequestCount_ != 0) &&
        residency_.canPrefetchPage(maxDevicePageBytes_);
    params.prefetchParams[0] = 1.0625f;
    params.prefetchParams[1] = .95f;
    params.prefetchParams[2] = currentFramePrefetch_ ? 1.f : 0.f;
    params.prefetchParams[3] = predictivePrefetch_ && prefetchForecast_.active ? 1.f : 0.f;
    std::copy_n(prefetchForecast_.camera.eye.data(), 3u, params.prefetchEye);
    std::copy_n(prefetchForecast_.camera.center.data(), 3u, params.prefetchCenter);
    std::copy_n(prefetchForecast_.camera.up.data(), 3u, params.prefetchUp);
    params.enableLodTransitionTelemetry = lodTransitionTelemetry_ && !frame.freezeRasterSnapshot ? 1u : 0u;
    Result<> result = updateHostBuffer(*paramsBuffer_, &params, sizeof(params));
    if (result) {
        previousFrameParams_ = params;
        previousFrameParamsValid_ = true;
    }
    return result;
}

Result<> MeshletStreamRuntime::dispatchTraversal(
    CommandBuffer& commandBuffer,
    uint32_t threadCount,
    uint32_t traversalPhase)
{
    if (traversalPass_ == nullptr || registry_ == nullptr || !traversalPass_->ready()) {
        return makeError(Error::InvalidArgument);
    }
    MeshletStreamUserPush push = userPush();
    push.traversalPhase = traversalPhase;
    push.activeBuildPhase = threadCount;
    return traversalPass_->dispatch(
        commandBuffer,
        *registry_->heap(),
        push,
        threadCount,
        *pageTableBuffer_,
        pageTableState_,
        *requestBuffer_,
        requestBufferState_);
}

Result<> MeshletStreamRuntime::buildActiveTable(CommandBuffer& commandBuffer, const TraversalCheckpoint& checkpoint)
{
    if (activeBuildPass_ == nullptr || registry_ == nullptr || !activeBuildPass_->ready()) {
        return makeError(Error::InvalidArgument);
    }

    MeshletStreamUserPush push = userPush();
    // The diagnostic observation packs a 26-bit frame tag. Reset at its wrap,
    // once per 67 million frames, so an old unvisited group cannot alias history.
    const bool initializeState = lodStateBufferState_ == ResourceState::Undefined ||
        (lodTransitionTelemetry_ && (frameIndex_ & 0x03ffffffu) == 0u);
    if (auto commandResult = transitionBuffer(commandBuffer, *lodStateBuffer_, lodStateBufferState_, ResourceState::General, true); !commandResult) { return commandResult; }
    if (distributedPageDemand_ || lodTransitionTelemetry_) { if (auto commandResult = transitionBuffer(commandBuffer, *demandBuffer_, demandBufferState_, ResourceState::General, true); !commandResult) { return commandResult; } }
    const auto dispatchPhase = [&](uint32_t phase, uint32_t threadCount) {
        push.activeBuildPhase = phase;
        Result<> result = activeBuildPass_->dispatch(
            commandBuffer, *registry_->heap(), push, threadCount,
            *activeGroupBuffer_, activeGroupBufferState_,
            *activeHeaderBuffer_, activeHeaderBufferState_,
            *pageTableBuffer_, pageTableState_, *requestBuffer_, requestBufferState_,
            *drawIndirectBuffer_, drawIndirectBufferState_,
            *traversalHeaderBuffer_, traversalHeaderBufferState_,
            *traversalWorkBuffer_, traversalWorkBufferState_);
        if (result) {
            if (auto commandResult = transitionBuffer(commandBuffer, *lodStateBuffer_, lodStateBufferState_, ResourceState::General, true); !commandResult) { return commandResult; }
            if (distributedPageDemand_ || lodTransitionTelemetry_) { if (auto commandResult = transitionBuffer(commandBuffer, *demandBuffer_, demandBufferState_, ResourceState::General, true); !commandResult) { return commandResult; } }
        }
        return result;
    };
    if (initializeState) {
        // Device allocations have undefined contents. Initialize once before
        // sparse clearing; subsequent frames touch only formerly active IDs.
        const Result<> result = dispatchPhase(kMeshletStreamActiveBuildInitializeLODStatePhase,
            static_cast<uint32_t>(lodStateBuffer_->desc().size / sizeof(uint32_t)));
        if (!result) { return result; }
    }
    auto result = dispatchPhase(kMeshletStreamActiveBuildResetPhase, 1u);
    if (!result) { return result; }
    if (currentFrameDistributedDemand_) {
        result = dispatchPhase(kMeshletStreamActiveBuildDemandResetPhase,
            static_cast<uint32_t>(demandBuffer_->desc().size / sizeof(uint32_t)));
        if (!result) { return result; }
        result = dispatchPhase(kMeshletStreamActiveBuildClearPhase, asset_.instanceCount());
        if (!result) { return result; }
    }
    if (checkpoint) { checkpoint("AfterStreamStateClear"); }
    if (currentFrameDistributedDemand_) {
        result = dispatchPhase(kMeshletStreamActiveBuildDemandPhase, traversalWorkerCount_);
        if (!result) { return result; }
    }
    if (checkpoint) { checkpoint("AfterStreamDemand"); }
    // Prefix budgets complete per-instance cuts before any records are emitted.
    // Each stage sees the complete result of its predecessor.
    for (uint32_t phase : {kMeshletStreamActiveBuildFrontierPhase, kMeshletStreamActiveBuildMaskPhase,
             kMeshletStreamActiveBuildPrefixPhase, kMeshletStreamActiveBuildEmitPhase, kMeshletStreamActiveBuildFinalizePhase}) {
        const bool perInstance = phase == kMeshletStreamActiveBuildFrontierPhase ||
            phase == kMeshletStreamActiveBuildMaskPhase || phase == kMeshletStreamActiveBuildEmitPhase;
        // Keep the original combined frontier for ordered demand and capacity
        // fallback; splitting its clear/mask without distributed work adds cost.
        if (phase != kMeshletStreamActiveBuildMaskPhase || currentFrameDistributedDemand_) {
            result = dispatchPhase(phase, perInstance ? asset_.instanceCount() : 1u);
            if (!result) { return result; }
        }
        if (checkpoint) {
            if (phase == kMeshletStreamActiveBuildFrontierPhase) { checkpoint("AfterStreamFrontier"); }
            if (phase == kMeshletStreamActiveBuildMaskPhase) { checkpoint("AfterStreamMask"); }
            if (phase == kMeshletStreamActiveBuildPrefixPhase) { checkpoint("AfterStreamPrefix"); }
            if (phase == kMeshletStreamActiveBuildEmitPhase) { checkpoint("AfterStreamEmit"); }
        }
    }
    if (currentFramePrefetch_) {
        const auto result = dispatchPhase(kMeshletStreamActiveBuildPrefetchPhase, asset_.instanceCount());
        if (!result) { return result; }
    }
    if (checkpoint) { checkpoint("AfterStreamPrefetch"); }
    return {};
}

Result<> MeshletStreamRuntime::buildBlasInputs(CommandBuffer& commandBuffer, const TraversalCheckpoint& checkpoint)
{
    if (blasInputPass_ == nullptr ||
        !blasInputPass_->ready() ||
        registry_ == nullptr ||
        clasPool_ == nullptr ||
        blasHeaderBuffer_ == nullptr ||
        instanceBlasBuffer_ == nullptr ||
        blasBuildInfoBuffer_ == nullptr ||
        blasClusterReferenceBuffer_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    // The GPU page generation is 32-bit. A global wrap conservatively resets
    // all instance caches, so an old generation can never compare equal again.
    const auto publicationEpoch = static_cast<uint32_t>(clasPool_->stats().publicationRevision >> 32u);
    if (publicationEpoch != blasPublicationEpoch_) {
        *blasCacheInitialized_ = false;
        blasPublicationEpoch_ = publicationEpoch;
    }
    commandBuffer.hostWriteBarrier();
    if (auto r = transitionBuffer(commandBuffer, *blasAddressBuffer_, blasAddressBufferState_, ResourceState::General, true); !r) { return r; }

    auto dispatchPhase = [this, &commandBuffer](uint32_t phase, uint32_t threadCount) {
        MeshletStreamUserPush push = userPush();
        push.activeBuildPhase = phase;
        push.traversalPhase = *blasCacheInitialized_ ? 0u : 1u;
        return blasInputPass_->dispatch(
            commandBuffer,
            *registry_->heap(),
            push,
            threadCount,
            *activeGroupBuffer_,
            activeGroupBufferState_,
            *activeHeaderBuffer_,
            activeHeaderBufferState_,
            *blasHeaderBuffer_,
            blasHeaderBufferState_,
            *instanceBlasBuffer_,
            instanceBlasBufferState_,
            *blasBuildInfoBuffer_,
            blasBuildInfoBufferState_,
            *blasClusterReferenceBuffer_,
            blasClusterReferenceBufferState_);
    };

    if (checkpoint) { checkpoint("BeforeBlasReset"); }
    Result<> result = dispatchPhase(
        kMeshletStreamBLASInputResetPhase,
        std::max(asset_.instanceCount(), 1u));
    if (!result) {
        return result;
    }
    result = dispatchPhase(5u, std::max(asset_.instanceCount(), 1u));
    if (!result) { return result; }
    result = commandBuffer.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(nullptr,
        [valid = blasCacheInitialized_] { *valid = false; }));
    if (!result) { return result; }
    *blasCacheInitialized_ = true;
    if (checkpoint) { checkpoint("BeforeBlasCount"); }
    result = dispatchPhase(kMeshletStreamBLASInputCountPhase, maxActiveGroups_);
    if (!result) {
        return result;
    }
    // Count current per-instance ranges before comparing against the previous
    // compact cut. Changes in other instances' offsets do not invalidate reuse.
    if (checkpoint) { checkpoint("BeforeBlasCompare"); }
    result = dispatchPhase(4u, maxActiveGroups_);
    if (!result) { return result; }
    if (checkpoint) { checkpoint("BeforeBlasPrefix"); }
    result = dispatchPhase(kMeshletStreamBLASInputPrefixPhase, std::max(asset_.instanceCount(), 1u));
    if (!result) { return result; }
    result = dispatchPhase(kMeshletStreamBLASInputBlockPrefixPhase, 1u);
    if (!result) { return result; }
    if (checkpoint) { checkpoint("BeforeBlasSetup"); }
    result = dispatchPhase(
        kMeshletStreamBLASInputSetupPhase,
        std::max(asset_.instanceCount(), 1u));
    if (!result) {
        return result;
    }
    if (checkpoint) { checkpoint("BeforeBlasAllocate"); }
    result = dispatchPhase(8u, std::max(asset_.instanceCount(), 1u));
    if (!result) { return result; }
    result = dispatchPhase(9u, 1u);
    if (!result) { return result; }
    result = dispatchPhase(10u, std::max(asset_.instanceCount(), 1u));
    if (!result) { return result; }
    if (checkpoint) { checkpoint("BeforeBlasInsert"); }
    result = dispatchPhase(kMeshletStreamBLASInputInsertPhase, maxActiveGroups_);
    if (!result) { return result; }
    result = dispatchPhase(11u, maxActiveGroups_);
    if (!result) { return result; }
    return transitionBuffer(commandBuffer, *blasAddressBuffer_, blasAddressBufferState_, ResourceState::General, true);
}

Result<> MeshletStreamRuntime::cmdBuildBlas(CommandBuffer& commandBuffer)
{
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    if (clasPool_ == nullptr ||
        blasBuildCapacity_ == 0 ||
        maxBlasClustersPerBuild_ == 0 ||
        blasClusterReferenceCapacity_ == 0 ||
        blasHeaderBuffer_ == nullptr ||
        blasBuildInfoBuffer_ == nullptr ||
        blasStorageBuffer_ == nullptr ||
        blasScratchBuffer_ == nullptr ||
        blasAddressBuffer_ == nullptr ||
        blasSizeBuffer_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return commandBuffer.buildClusterAccelerationStructureBottomLevels(
        ClusterAccelerationStructureBottomLevelBuildDesc{
            .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
            .destinationMode = ClusterAccelerationStructureDestinationMode::Explicit,
            .maxClusterCountPerAccelerationStructure = maxBlasClustersPerBuild_,
            .maxTotalClusterCount = blasClusterReferenceCapacity_,
            .maxAccelerationStructureCount = blasBuildCapacity_,
            .buildInfoBuffer = blasBuildInfoBuffer_.get(),
            .buildInfoStride = sizeof(MeshletStreamGPUBLASBuildInfo),
            .buildInfoSize = blasBuildInfoBuffer_->desc().size,
            .buildInfoCountBuffer = blasHeaderBuffer_.get(),
            .buildInfoCountBufferOffset = offsetof(
                MeshletStreamGPUBLASHeader,
                blasBuildCount),
            .destinationStorageBuffer = blasStorageBuffer_.get(),
            .destinationAddressBuffer = blasAddressBuffer_.get(),
            .destinationAddressSize = blasAddressBuffer_->desc().size,
            .destinationSizeBuffer = blasSizeBuffer_.get(),
            .destinationSizeSize = blasSizeBuffer_->desc().size,
            .scratchBuffer = blasScratchBuffer_.get(),
        });
}

Result<> MeshletStreamRuntime::cmdBuildFallbackBlas(CommandBuffer& commandBuffer)
{
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    if (clasPool_ == nullptr ||
        fallbackBlasStorageBuffer_ == nullptr ||
        fallbackBlasScratchBuffer_ == nullptr ||
        fallbackBlasReferenceBuffer_ == nullptr ||
        fallbackBlasBuildInfoBuffer_ == nullptr ||
        fallbackBlasDestinationBuffer_ == nullptr ||
        fallbackBlasAddressBuffer_ == nullptr ||
        fallbackBlasPrimitives_.empty()) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<uint32_t> readyFallbackIndices;
    for (uint32_t fallbackIndex = 0;
         fallbackIndex < fallbackBlasPrimitives_.size();
         ++fallbackIndex) {
        const FallbackBLASPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
        if (fallback.recorded()) {
            continue;
        }
        const uint32_t primitiveIndex = fallback.primitiveIndex;
        bool ready = true;
        for (uint32_t groupIndex : asset_.primitiveTerminalGroups(primitiveIndex)) {
            if (!clasPool_->pageHasClas(asset_.groups()[groupIndex].pageIndex)) {
                ready = false;
                break;
            }
        }
        if (ready) {
            readyFallbackIndices.push_back(fallbackIndex);
        }
    }
    if (readyFallbackIndices.empty()) {
        return {};
    }

    auto* referenceData = static_cast<uint64_t*>(fallbackBlasReferenceBuffer_->map());
    auto* addressData = static_cast<uint64_t*>(fallbackBlasAddressBuffer_->map());
    if (referenceData == nullptr || addressData == nullptr) {
        if (referenceData != nullptr) {
            fallbackBlasReferenceBuffer_->unmap();
        }
        if (addressData != nullptr) {
            fallbackBlasAddressBuffer_->unmap();
        }
        return makeError(Error::Failure);
    }

    const uint64_t fallbackStorageAddress =
        fallbackBlasStorageBuffer_->deviceAddress();
    for (uint32_t fallbackIndex : readyFallbackIndices) {
        const FallbackBLASPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
        const uint32_t primitiveIndex = fallback.primitiveIndex;
        const uint64_t referenceOffset = fallback.referenceOffset;
        uint64_t writeOffset = referenceOffset;
        for (uint32_t groupIndex : asset_.primitiveTerminalGroups(primitiveIndex)) {
            const uint32_t pageIndex = asset_.groups()[groupIndex].pageIndex;
            const uint32_t clusterCount = asset_.pages()[pageIndex].clusterCount;
            for (uint32_t clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex) {
                referenceData[writeOffset++] = clasPool_->clusterAddress(pageIndex, clusterIndex);
            }
        }
        if (writeOffset - referenceOffset != fallback.referenceCount) {
            fallbackBlasReferenceBuffer_->unmap();
            fallbackBlasAddressBuffer_->unmap();
            return makeError(Error::Failure);
        }
        addressData[primitiveIndex] =
            fallbackStorageAddress + fallback.storageOffset;
        fallbackBlasReferenceBuffer_->flush({referenceOffset * sizeof(uint64_t), static_cast<uint64_t>(fallback.referenceCount) * sizeof(uint64_t)});
        fallbackBlasAddressBuffer_->flush({static_cast<uint64_t>(primitiveIndex) * sizeof(uint64_t), sizeof(uint64_t)});
    }
    fallbackBlasReferenceBuffer_->unmap();
    fallbackBlasAddressBuffer_->unmap();

    if (fallbackBlasBuildInfoBuffer_->deviceAddress() == 0 ||
        fallbackBlasDestinationBuffer_->deviceAddress() == 0 ||
        fallbackStorageAddress == 0 ||
        fallbackBlasScratchBuffer_->deviceAddress() == 0) {
        return makeError(Error::InvalidArgument);
    }

    auto publication = std::make_shared<SubmissionTransaction>(
        [cache = sceneReadinessCache_] { cache->valid = false; },
        [cache = sceneReadinessCache_] { cache->valid = false; cache->value.ready = false; });
    const auto tracked = commandBuffer.addSubmissionTransaction(publication);
    if (!tracked) { return tracked; }
    sceneReadinessCache_->valid = false;
    for (uint32_t fallbackIndex : readyFallbackIndices) {
        FallbackBLASPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
        const uint32_t clusterCount = fallback.referenceCount;
        const Result<> result = commandBuffer.buildClusterAccelerationStructureBottomLevels(
            ClusterAccelerationStructureBottomLevelBuildDesc{
                .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                .destinationMode = ClusterAccelerationStructureDestinationMode::Explicit,
                .maxClusterCountPerAccelerationStructure = clusterCount,
                .maxTotalClusterCount = clusterCount,
                .maxAccelerationStructureCount = 1,
                .buildInfoBuffer = fallbackBlasBuildInfoBuffer_.get(),
                .buildInfoBufferOffset =
                    static_cast<uint64_t>(fallbackIndex) *
                    sizeof(MeshletStreamGPUBLASBuildInfo),
                .buildInfoStride = sizeof(MeshletStreamGPUBLASBuildInfo),
                .buildInfoSize = sizeof(MeshletStreamGPUBLASBuildInfo),
                .destinationAddressBuffer = fallbackBlasDestinationBuffer_.get(),
                .destinationAddressBufferOffset =
                    static_cast<uint64_t>(fallbackIndex) * sizeof(uint64_t),
                .destinationAddressSize = sizeof(uint64_t),
                .scratchBuffer = fallbackBlasScratchBuffer_.get(),
            });
        if (!result) {
            return result;
        }
        fallback.built = true;
        fallback.buildTransaction = publication;
    }
    return {};
}

Result<> MeshletStreamRuntime::buildTlasInstances(CommandBuffer& commandBuffer)
{
    if (tlasInputPass_ == nullptr ||
        !tlasInputPass_->ready() ||
        registry_ == nullptr ||
        instanceBlasBuffer_ == nullptr ||
        tlasInstanceBuffer_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    commandBuffer.hostWriteBarrier();
    return tlasInputPass_->dispatch(
        commandBuffer,
        *registry_->heap(),
        userPush(),
        asset_.instanceCount(),
        *instanceBlasBuffer_,
        instanceBlasBufferState_,
        *tlasInstanceBuffer_,
        tlasInstanceBufferState_);
}

Result<> MeshletStreamRuntime::cmdBuildTlas(CommandBuffer& commandBuffer)
{
    if (auto result = prepareImmutableMetadataRead(commandBuffer); !result) { return result; }
    if (tlas_ == nullptr ||
        !tlas_->valid() ||
        tlasScratchBuffer_ == nullptr ||
        tlasInstanceBuffer_ == nullptr ||
        asset_.instanceCount() == 0) {
        return makeError(Error::InvalidArgument);
    }
    const Result<> result = commandBuffer.buildRayTracingAccelerationStructure(
        RayTracingAccelerationStructureBuildDesc{
            .destination = tlas_.get(),
            .mode = RayTracingAccelerationStructureBuildMode::Build,
            .instanceBuffer = tlasInstanceBuffer_.get(),
            .instanceCount = asset_.instanceCount(),
            .scratchBuffer = tlasScratchBuffer_.get(),
            .graphManagedSynchronization = true,
        });
    if (!result) {
        return result;
    }
    tlasBuilt_ = true;
    return {};
}

Result<> MeshletStreamRuntime::transitionPageBufferForTraversal(CommandBuffer& commandBuffer)
{
    if (drawTaskCount() == 0) {
        return {};
    }
    if (auto commandResult = transitionBuffer(commandBuffer, *pageBuffer_, pageBufferState_, ResourceState::ShaderRead); !commandResult) { return commandResult; }
    return {};
}

void MeshletStreamRuntime::consumeGpuRequestReadback(CPUProfileRecorder* profiler, bool allowLegacyReadback)
{
    RequestReadback* latest = nullptr;
    for (auto& candidate : requestReadbacks_) {
        if (candidate.submission && candidate.submission->resolved() && !candidate.submission->cancelled() &&
            candidate.completion.isSubmitted() && candidate.completion.isComplete() && candidate.frame > consumedRequestFrame_ &&
            (!latest || candidate.frame > latest->frame)) { latest = &candidate; }
    }
    Buffer* readback = latest ? latest->buffer.get() :
        (allowLegacyReadback && requestReadbackValid_ ? requestReadbackBuffer_.get() : nullptr);
    if (!readback) { return; }
    CPUProfileScope profile(profiler, "Map feedback");
    readback->invalidate();
    const void* mapped = readback->map();
    if (mapped == nullptr) {
        requestReadbackValid_ = false;
        return;
    }

    if (latest) { consumedRequestFrame_ = latest->frame; }

    const auto* header = static_cast<const StreamRequestBufferHeader*>(mapped);
    if (distributedPageDemand_ || lodTransitionTelemetry_) {
        std::memcpy(recentDemandStats_.data(), static_cast<const uint8_t*>(mapped) +
            readback->desc().size - sizeof(recentDemandStats_) - sizeof(MeshletStreamGPUBLASHeader), sizeof(recentDemandStats_));
        recentDemandGroupTests_ = recentDemandStats_[18];
    }
    if (clusterRtxEnabled_) {
        std::memcpy(&recentBlasHeader_, static_cast<const uint8_t*>(mapped) +
            readback->desc().size - sizeof(MeshletStreamGPUBLASHeader), sizeof(recentBlasHeader_));
    }
    recentGpuRequestCount_ = header->loadCounter;
    recentPrefetchGpuRequests_ = header->prefetchRequestCounter;
    recentPrefetchGpuDropped_ = header->prefetchDroppedCounter;
    // This header is already consumed for residency even when debug capture is off.
    debugRequestSourceFrame_ = header->frameIndex;
    debugRequestSourceKnown_ = true;
    const uint32_t loadCapacity = std::min(header->maxLoadRequests, maxGpuPageRequests_);
    const uint32_t unloadCapacity = std::min(header->maxUnloadRequests, maxGpuPageUnloadRequests_);
    const uint32_t loadCount = std::min(header->loadCounter, loadCapacity);
    const uint32_t unloadCount = std::min(header->unloadCounter, unloadCapacity);
    const auto* pageIds = reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(mapped) + sizeof(StreamRequestBufferHeader));
    const uint32_t* loadPageIds = pageIds;
    const uint32_t* unloadPageIds = pageIds + maxGpuPageRequests_;
    std::span<const float> loadPriorities;
    const uint32_t expectedOffset = kStreamRequestHeaderWordCount + maxGpuPageRequests_ + maxGpuPageUnloadRequests_;
    if (screenSpacePagePriority_ && header->loadPriorityOffset == expectedOffset) {
        loadPriorities = {reinterpret_cast<const float*>(mapped) + expectedOffset, loadCount};
    }
    profile.next("Consume requests");
    // Empty feedback is meaningful: every enumerated resident page was used.
    {
        (void)residency_.consumeGpuRequests(StreamGPURequestBatch{
            .loadPageIds = std::span<const uint32_t>(loadPageIds, loadCount),
            .unloadPageIds = std::span<const uint32_t>(unloadPageIds, unloadCount),
            .loadRequestCounter = header->loadCounter,
            .unloadRequestCounter = header->unloadCounter,
            .loadOverflowCounter = header->loadOverflowCounter,
            .unloadOverflowCounter = header->unloadOverflowCounter,
            .invalidPageCounter = header->invalidPageCounter,
            .frameIndex = header->frameIndex,
            .residentDemandFeedback = true,
            .loadPriorities = loadPriorities,
            .taggedPrefetchRequests = prefetchPages_ && header->prefetchRequestLimit != 0,
        }, profiler);
    }

    profile.next("Unmap feedback");
    readback->unmap();
    requestReadbackValid_ = false;
}

void MeshletStreamRuntime::appendReplayBindings(std::vector<profiling::WorkControlReplayBinding>& bindings) const
{
    bindings.insert(bindings.end(), {
        {"pages", pageBuffer_.get(), pageHandle_.shaderIndex()},
        {"groups", activeGroupBuffer_.get(), activeGroupHandle_.shaderIndex()},
        {"header", activeHeaderBuffer_.get(), activeHeaderHandle_.shaderIndex()},
        {"pageTable", pageTableBuffer_.get(), pageTableHandle_.shaderIndex()},
        {"params", paramsBuffer_.get(), paramsHandle_.shaderIndex()},
        {"rasterBindings", rasterBindingsBuffer_.get(), rasterBindingsHandle_.shaderIndex()},
        {"requests", requestBuffer_.get(), UINT32_MAX},
        {"visibleRecords", visibleClusterBuffer_.get(), UINT32_MAX},
        {"lodState", lodStateBuffer_.get(), UINT32_MAX},
    });
}

void MeshletStreamRuntime::appendDebugBindings(std::vector<DebugResourceBinding>& bindings, const std::string& prefix) const
{
    if (!debugReadbackEnabled_ || !ready()) { return; }
    const auto add = [&](std::string name, Buffer* buffer, ResourceState state, std::string layout,
                         uint64_t offset = 0, uint64_t size = 0) {
        if (!buffer) { return; }
        bindings.push_back({.id = prefix + name, .buffer = buffer, .state = state, .offset = offset, .size = size,
            .layout = std::move(layout), .allocation = debugGeneration_,
            .metadata = {{"streamFrame", frameIndex_}, {"streamGeneration", debugGeneration_},
                {"validity", "Only header-defined live ranges contain records"}}});
    };
    add("requestHeader", requestBuffer_.get(), requestBufferState_, "StreamRequestBufferHeader", 0, sizeof(StreamRequestBufferHeader));
    add("loadRequests", requestBuffer_.get(), requestBufferState_, "u32", sizeof(StreamRequestBufferHeader), uint64_t(maxGpuPageRequests_) * 4);
    add("unloadRequests", requestBuffer_.get(), requestBufferState_, "u32", sizeof(StreamRequestBufferHeader) + uint64_t(maxGpuPageRequests_) * 4, uint64_t(maxGpuPageUnloadRequests_) * 4);
    if (screenSpacePagePriority_) {
        add("loadPriorities", requestBuffer_.get(), requestBufferState_, "f32",
            sizeof(StreamRequestBufferHeader) + (uint64_t(maxGpuPageRequests_) + maxGpuPageUnloadRequests_) * 4,
            uint64_t(maxGpuPageRequests_) * 4);
    }
    add("pageTable", pageTableBuffer_.get(), pageTableState_, "StreamPageTableEntry");
    add("activeHeader", activeHeaderBuffer_.get(), activeHeaderBufferState_, "MeshletStreamGPUActiveHeader");
    add("blasHeader", blasHeaderBuffer_.get(), blasHeaderBufferState_, "MeshletStreamGPUBLASHeader", 0,
        sizeof(MeshletStreamGPUBLASHeader));
    add("activeGroups", activeGroupBuffer_.get(), activeGroupBufferState_, "MeshletStreamGPUActiveGroup");
    add("lodState", lodStateBuffer_.get(), lodStateBufferState_, "u32");
    if (distributedPageDemand_ || lodTransitionTelemetry_) { add("demandStats", demandBuffer_.get(), demandBufferState_, "u32", 0, kMeshletStreamDemandStatsWords * sizeof(uint32_t)); }
    add("visibleClusters", visibleClusterBuffer_.get(), visibleClusterBufferState_, "CompactStreamVisibleRecord");
}

SceneStreamingProfile MeshletStreamRuntime::profilingStats() const
{
    const auto stats = residency_.stats(false);
    SceneStreamingProfile result;
    result.assetPath = asset_.path().generic_string();
    result.generation = debugGeneration_;
    result.frameIndex = frameIndex_;
    result.feedbackFrame = debugRequestSourceKnown_ ? debugRequestSourceFrame_ : UINT64_MAX;
    result.deviceImmutableMetadata = deviceImmutableMetadata_;
    result.immutableMetadataReady = immutableMetadataReady();
    result.immutableGroupBytes = groupBuffer_ ? groupBuffer_->desc().size : 0;
    result.immutableTopologyBytes = lodTopologyBuffer_ ? lodTopologyBuffer_->desc().size : 0;
    result.immutableMetadataBytes = result.immutableGroupBytes + result.immutableTopologyBytes;
    result.immutableMetadataAllocatedBytes = (groupBuffer_ ? groupBuffer_->memoryInfo().sizeBytes : 0) +
        (lodTopologyBuffer_ ? lodTopologyBuffer_->memoryInfo().sizeBytes : 0);
    if (immutableMetadataUpload_) {
        result.immutableMetadataSubmittedBytes = immutableMetadataUpload_->submittedBytes;
        result.immutableMetadataStagingBytes = immutableMetadataUpload_->staging ?
            immutableMetadataUpload_->staging->desc().size : 0;
        result.immutableMetadataUploadBatches = immutableMetadataUpload_->uploadBatches;
    }
    result.lodTransitionTelemetryEnabled = lodTransitionTelemetry_;
    result.lodTransitionHistoryBytes = lodTransitionHistoryBytes_;
    result.lodDemandedGroups = recentDemandStats_[19];
    result.lodOwnPageBlockedGroups = recentDemandStats_[20];
    result.lodDependencyBlockedGroups = recentDemandStats_[21];
    result.lodCatchupActivatedGroups = recentDemandStats_[22];
    result.lodCatchupSelectedGroups = recentDemandStats_[23];
    result.lodCatchupSelectedClusters = recentDemandStats_[24];
    result.lodThresholdSelectedGroups = recentDemandStats_[25];
    result.lodThresholdSelectedClusters = recentDemandStats_[26];
    result.lodUnclassifiedSelectedGroups = recentDemandStats_[27];
    result.lodUnclassifiedSelectedClusters = recentDemandStats_[28];
    result.predictivePrefetchEnabled = predictivePrefetch_;
    result.prefetchEnabled = prefetchPages_;
    result.prefetchMemoryWatermarkBlocked = !residency_.canPrefetchPage(maxDevicePageBytes_);
    result.prefetchQueueBlocked = residency_.availablePrefetchRequests() == 0;
    result.prefetchForecastActive = currentFramePrefetch_ && predictivePrefetch_ && prefetchForecast_.active;
    result.prefetchMeasuredLatency = prefetchForecast_.measuredLatency;
    result.prefetchHorizonMilliseconds = prefetchForecast_.horizonSeconds * 1000.0;
    result.prefetchTranslationDistance = prefetchForecast_.translationDistance;
    result.prefetchRotationDegrees = prefetchForecast_.rotationDegrees;
    const auto demandLatency = residency_.recentDemandLatency();
    result.prefetchLatencySamples = demandLatency.count;
    result.prefetchDemandLatencyP95Milliseconds = demandLatency.p95;
    result.totalPrefetchAdmitted = stats.totalPrefetchAdmitted;
    result.totalPrefetchUsed = stats.totalPrefetchUsed;
    result.totalPrefetchDeferred = stats.totalPrefetchDeferred;
    result.prefetchGpuRequests = recentPrefetchGpuRequests_;
    result.prefetchGpuDropped = recentPrefetchGpuDropped_;
    result.adaptivePageRetentionEnabled = adaptivePageRetention_;
    result.geometryReclaimReserveBytes = geometryReclaimReserveBytes_;
    result.geometryDemandReserveBytes = geometryDemandReserveBytes_;
    result.coldResidentBytes = stats.frameColdResidentBytes;
    result.pendingFreeBytes = stats.framePendingFreeBytes;
    result.evictedGeometryBytes = stats.frameEvictedGeometryBytes;
    result.evictedPrefetchPages = stats.frameEvictedPrefetchPages;
    result.blasFeedbackAvailable = clusterRtxEnabled_ && debugRequestSourceKnown_;
    result.blasFeedbackFrame = recentBlasHeader_.frameIndex;
    result.blasBuildCount = recentBlasHeader_.blasBuildCount;
    result.blasClusterReferences = recentBlasHeader_.clusterReferenceCount;
    result.blasOverflowCount = recentBlasHeader_.overflowCount;
    result.blasRequestedClusterReferences = recentBlasHeader_.requestedClusterReferences;
    result.blasRequestedInstances = recentBlasHeader_.requestedInstances;
    result.blasReferenceBudgetRejected = recentBlasHeader_.referenceBudgetRejected;
    result.blasBuildBudgetRejected = recentBlasHeader_.buildBudgetRejected;
    result.blasOversizedInstances = recentBlasHeader_.oversizedInstances;
    result.blasMissingClasInstances = recentBlasHeader_.missingClasInstances;
    result.blasInvalidGroups = recentBlasHeader_.invalidGroups;
    result.blasLiveClusterReferences = recentBlasHeader_.liveClusterReferences;
    result.blasAdmittedInstances = recentBlasHeader_.admittedInstances;
    result.blasInstanceReuseEnabled = blasInstanceReuse_;
    result.blasDirtyInstances = recentBlasHeader_.dirtyInstances;
    result.blasReusedInstances = recentBlasHeader_.reusedInstances;
    result.blasStorageRejected = recentBlasHeader_.storageRejected;
    result.blasArenaUsedBytes = recentBlasHeader_.arenaUsedBytes;
    result.blasArenaRepack = recentBlasHeader_.arenaRepack;
    result.blasPublicationInvalidated = recentBlasHeader_.publicationInvalidated;


    result.geometryUsedBytes = stats.usedResidentBytes;
    result.geometryBudgetBytes = stats.maxResidentBytes;
    result.totalPages = stats.pageCount;
    result.residentPages = stats.residentPageCount;
    result.pendingPages = stats.pendingPageCount;
    result.ioQueued = stats.pendingPageLoadCount;
    result.ioActive = stats.activePageLoadCount;
    result.uploadQueued = stats.queuedUploadCount;
    result.requests = stats.frameGpuRequestCount;
    result.uploads = stats.frameCompletedUploadCount;
    result.evictions = stats.frameEvictedPageCount;
    result.requestOverflows = stats.frameGpuRequestOverflowCount;
    result.allocationFailures = stats.frameAllocationFailureCount;
    result.uploadBytes = stats.frameUploadBytes;
    result.totalUploadBytes = stats.totalUploadBytes;
    result.storedUploadBytes = stats.frameStoredUploadBytes;
    result.totalStoredUploadBytes = stats.totalStoredUploadBytes;
    result.gpuDecompressedPages = stats.frameGpuDecompressedPages;
    result.totalGpuDecompressedPages = stats.totalGpuDecompressedPages;
    result.loadFailures = stats.totalPageLoadFailureCount;
    result.throughput = residency_.throughputSnapshot();
    result.cpuWork = stats.cpuWork;
    result.clasEnabled = clasPool_ && clasPool_->ready();
    if (result.clasEnabled) {
        const auto clas = clasPool_->stats();
        result.clasUsedBytes = clas.usedStorageBytes;
        result.clasCapacityBytes = clas.storageBudgetBytes;
        result.clasAllocatedBytes = clas.storageBytes;
        result.clasStorageChunks = clas.storageChunkCount;
        result.clasStartBytes = clas.startStorageBytes;
        result.clasGrowBytes = clas.growStorageBytes;
        result.clasEmptyBytes = clas.emptyStorageBytes;
        result.clasGrowthCount = clas.totalStorageGrowthCount;
        result.clasReleasedBytes = clas.totalStorageReleasedBytes;
        result.clasPersistentAllocatedBytes = clas.persistentStorageBytes;
        result.clasPersistentGrowBytes = clas.persistentGrowStorageBytes;
        result.clasPersistentUsedBytes = clas.persistentUsedBytes;
        result.clasTransientAllocatedBytes = clas.transientStorageBytes;
        result.clasTransientUsedBytes = clas.transientUsedBytes;
        result.clasFragmentedFreeBytes = clas.fragmentedFreeBytes;
        result.clasEncodedBytes = clas.encodedStorageBytes;
        result.clasWorstCaseBytes = clas.worstCaseStorageBytes;
        result.clasScratchBytes = clas.scratchBytes;
        result.clasMovedClusters = clas.frameMovedClusterCount;
        result.clasResidentPages = clas.builtPageCount;
        result.clasResidentClusters = clas.builtClusterCount;
        result.clasRetiringPages = clas.retiringPageCount;
        result.clasPendingPages = static_cast<uint32_t>(queuedClasPages_.size());
        result.clasBuiltPages = clas.frameBuiltPageCount;
        result.clasBuiltClusters = clas.frameBuiltClusterCount;
        result.clasRejectedPages = clas.frameRejectedPageCount;
        result.clasTotalBuiltPages = clas.totalBuiltPageCount;
        result.clasTotalBuiltClusters = clas.totalBuiltClusterCount;
    }
    return result;
}

nlohmann::json MeshletStreamRuntime::debugSnapshot(bool includePages) const
{
    const auto recentDemandLatency = residency_.recentDemandLatency();
    using debug::DebugValue;
    const auto stats = residency_.stats();
    const auto latency = residency_.latencySnapshot();
    const auto speed = residency_.throughputSnapshot();
    const DebugValue throughputJson{
        {"windowSeconds", speed.windowSeconds}, {"loadedPagesPerSecond", speed.loadedPagesPerSecond},
        {"loadedStoredMiBPerSecond", speed.loadedStoredMiBPerSecond}, {"preparedMiBPerSecond", speed.preparedMiBPerSecond},
        {"transferMiBPerSecond", speed.transferMiBPerSecond}, {"geometryReadyPagesPerSecond", speed.geometryReadyPagesPerSecond},
        {"geometryReadyMiBPerSecond", speed.geometryReadyMiBPerSecond},
        {"loadedPages", speed.totals.loadedPages}, {"loadedStoredBytes", speed.totals.loadedStoredBytes},
        {"preparedDeviceBytes", speed.totals.preparedDeviceBytes}, {"transferPayloadBytes", speed.totals.transferPayloadBytes},
        {"geometryReadyPages", speed.totals.geometryReadyPages}, {"geometryReadyBytes", speed.totals.geometryReadyBytes},
        {"smallBatchCpuPages", speed.totals.smallBatchCpuPages}};
    const auto distribution = [](const MeshletStreamLatencySummary& sample) -> DebugValue {
        return {{"count", sample.count}, {"p50", sample.p50}, {"p95", sample.p95},
            {"p99", sample.p99}, {"max", sample.maximum}, {"mean", sample.mean}};
    };
    DebugValue latencyJson{{"enabled", latency.enabled}, {"pendingDemand", latency.pendingDemand},
        {"pendingPrefetch", latency.pendingPrefetch}, {"abandonedDemand", latency.abandonedDemand},
        {"abandonedPrefetch", latency.abandonedPrefetch}, {"oldestPendingDemandMilliseconds", latency.oldestPendingDemandMilliseconds},
        {"demandFrames", distribution(latency.demandFrames)}};
    constexpr std::array names{"feedback", "admission", "ioQueue", "decode", "readyToUpload", "uploadToDrawable", "demandToDrawable", "prefetchToDrawable"};
    for (size_t i = 0; i < names.size(); ++i) { latencyJson["milliseconds"][names[i]] = distribution(latency.milliseconds[i]); }
    const auto terminalResidentPages = std::count_if(lockedFallbackPages_.begin(), lockedFallbackPages_.end(),
        [this](uint32_t page) { return residency_.pageResident(page); });
    DebugValue pages = DebugValue::array();
    const uint32_t count = includePages ? std::min(residency_.trackedPageCount(), 4096u) : 0u;
    for (uint32_t i = 0; i < count; ++i) {
        pages.push_back({{"index", i}, {"state", static_cast<uint32_t>(residency_.pageState(i))},
            {"deviceOffset", residency_.deviceOffsetForPage(i)}, {"deviceSize", residency_.deviceSizeForPage(i)}, {"age", residency_.pageAge(i)}});
    }
    const auto readiness = sceneReadiness();
    return {{"generation", debugGeneration_}, {"frame", frameIndex_},
        {"orderedUploadPages", currentFrameOrderedUploadCount_},
        {"activeGroupCapacity", maxActiveGroups_},
        {"visibleRecordCapacity", visibleClusterCapacity()},
        {"blasBuildCapacity", blasBuildCapacity_},
        {"blasClusterReferenceCapacity", blasClusterReferenceCapacity_},
        {"maxBlasClustersPerBuild", maxBlasClustersPerBuild_},
        {"visibleRecordStride", sizeof(CompactStreamVisibleRecord)},
        {"visibleRecordBytes", visibleClusterBuffer_ ? visibleClusterBuffer_->desc().size : 0ull},
        {"rasterCandidateCapacity", rasterCandidateCapacity_},
        {"sceneReady", readiness.ready}, {"scenePreparationFraction", readiness.fraction()},
        {"sceneReadinessScans", sceneReadinessCache_->scans},
        {"sceneRootsInvalidated", sceneReadinessCache_->rootsInvalidated},
        {"fallbackBlasRecorded", std::count_if(fallbackBlasPrimitives_.begin(), fallbackBlasPrimitives_.end(),
            [](const auto& fallback) { return fallback.recorded(); })},
        {"fallbackBlasSubmitted", std::count_if(fallbackBlasPrimitives_.begin(), fallbackBlasPrimitives_.end(),
            [](const auto& fallback) { return fallback.submitted(); })},
        {"primitiveCount", asset_.primitiveCount()}, {"instanceCount", asset_.instanceCount()},
        {"terminalPageCount", lockedFallbackPages_.size()}, {"terminalResidentPageCount", terminalResidentPages},
        {"terminalReady", !lockedFallbackPages_.empty() && terminalResidentPages == lockedFallbackPages_.size()},
        {"pageBufferBytes", maxResidentBytes_}, {"clusterRtxEnabled", clusterRtxEnabled_}, {"clasEnabled", clasPool_ != nullptr},
        {"maxUploadBytesPerFrame", maxUploadBytesPerFrame_},
        {"gpuDecompressionEnabled", pageBuffer_ && hasFlag(pageBuffer_->desc().usage, BufferUsageBits::MemoryDecompression)},
        {"gpuDecompressionMinBatchBytes", residency_.gpuDecompressionMinBatchBytes()},
        {"throughput", throughputJson},
        {"screenSpacePagePriority", screenSpacePagePriority_},
        {"viewDrivenPageDemand", viewDrivenPageDemand_},
        {"distributedPageDemand", distributedPageDemand_}, {"demandTaskCount", demandTaskCount_},
        {"distributedDemandActive", currentFrameDistributedDemand_},
        {"distributedDemandMinGroups", distributedDemandMinGroups_}, {"recentDemandGroupTests", recentDemandGroupTests_},
        {"demandTaskCapacity", traversalWorkCapacity_}, {"demandWorkers", traversalWorkerCount_},
        {"demandBufferBytes", demandBuffer_ ? demandBuffer_->desc().size : 0},
        {"prefetchPages", prefetchPages_}, {"prefetchActive", currentFramePrefetch_},
        {"retention", {{"adaptive", adaptivePageRetention_},
            {"reclaimReserveBytes", geometryReclaimReserveBytes_}, {"demandReserveBytes", geometryDemandReserveBytes_},
            {"coldResidentBytes", stats.frameColdResidentBytes}, {"pendingFreeBytes", stats.framePendingFreeBytes},
            {"evictedGeometryBytes", stats.frameEvictedGeometryBytes}, {"evictedPrefetchPages", stats.frameEvictedPrefetchPages},
            {"pressureEvictions", stats.cpuWork.coldPressureScheduled}, {"retentionEvictions", stats.cpuWork.coldRetentionScheduled}}},
        {"prefetchMemoryWatermarkBlocked", !residency_.canPrefetchPage(maxDevicePageBytes_)},
        {"prefetchQueueBlocked", residency_.availablePrefetchRequests() == 0},
        {"predictivePrefetch", predictivePrefetch_},
        {"prefetchForecast", {{"active", currentFramePrefetch_ && predictivePrefetch_ && prefetchForecast_.active},
            {"horizonMilliseconds", prefetchForecast_.horizonSeconds * 1000.0},
            {"translationDistance", prefetchForecast_.translationDistance}, {"rotationDegrees", prefetchForecast_.rotationDegrees},
            {"measuredLatency", prefetchForecast_.measuredLatency},
            {"latencySamples", recentDemandLatency.count}, {"demandLatencyP95Milliseconds", recentDemandLatency.p95},
            {"gpuRequests", recentPrefetchGpuRequests_}, {"gpuDropped", recentPrefetchGpuDropped_}}},
        {"lodTransitions", {{"enabled", lodTransitionTelemetry_}, {"historyBytes", lodTransitionHistoryBytes_},
            {"feedbackFrame", debugRequestSourceKnown_ ? DebugValue(debugRequestSourceFrame_) : DebugValue(nullptr)},
            {"demandedGroups", recentDemandStats_[19]}, {"ownPageBlockedGroups", recentDemandStats_[20]},
            {"dependencyBlockedGroups", recentDemandStats_[21]}, {"catchupActivatedGroups", recentDemandStats_[22]},
            {"catchupSelectedGroups", recentDemandStats_[23]}, {"catchupSelectedClusters", recentDemandStats_[24]},
            {"thresholdSelectedGroups", recentDemandStats_[25]}, {"thresholdSelectedClusters", recentDemandStats_[26]},
            {"unclassifiedSelectedGroups", recentDemandStats_[27]}, {"unclassifiedSelectedClusters", recentDemandStats_[28]}}},
        {"latency", std::move(latencyJson)},
        {"requestBufferBytes", requestBuffer_ ? requestBuffer_->desc().size : 0},
        {"requestReadbackBytes", requestReadbackBuffer_ ? requestReadbackBuffer_->desc().size * (requestReadbacks_.size() + 1u) : 0},
        {"consumedRequestFrame", consumedRequestFrame_}, {"maintenancePrepared", maintenancePrepared_},
        {"lodTopologyBytes", lodTopologyBuffer_ ? lodTopologyBuffer_->desc().size : 0},
        {"groupsBytes", groupBuffer_ ? groupBuffer_->desc().size : 0},
        {"immutableMetadataDeviceRequested", deviceImmutableMetadata_},
        {"immutableMetadataReady", immutableMetadataReady()},
        {"immutableMetadataBytes", (groupBuffer_ ? groupBuffer_->desc().size : 0) +
            (lodTopologyBuffer_ ? lodTopologyBuffer_->desc().size : 0)},
        {"immutableMetadataAllocatedBytes", (groupBuffer_ ? groupBuffer_->memoryInfo().sizeBytes : 0) +
            (lodTopologyBuffer_ ? lodTopologyBuffer_->memoryInfo().sizeBytes : 0)},
        {"immutableMetadataSubmittedBytes", immutableMetadataUploadedBytes()},
        {"immutableMetadataCopyPending", immutableMetadataUpload_ && immutableMetadataUpload_->submission &&
            (!immutableMetadataUpload_->submission->resolved() ||
                (!immutableMetadataUpload_->submission->cancelled() && !immutableMetadataUpload_->completion.isComplete()))},
        {"immutableMetadataStagingBytes", immutableMetadataUpload_ && immutableMetadataUpload_->staging ?
            immutableMetadataUpload_->staging->desc().size : 0},
        {"immutableMetadataStagingPeakBytes", immutableMetadataUpload_ ? immutableMetadataUpload_->stagingPeakBytes : 0},
        {"immutableMetadataStagingAllocatedBytes", immutableMetadataUpload_ ? immutableMetadataUpload_->stagingAllocatedBytes : 0},
        {"immutableMetadataUploadBatches", immutableMetadataUpload_ ? immutableMetadataUpload_->uploadBatches : 0},
        {"lodStateBytes", lodStateBuffer_ ? lodStateBuffer_->desc().size : 0},
        {"requestSourceFrame", debugRequestSourceKnown_ ? DebugValue(debugRequestSourceFrame_) : DebugValue(nullptr)},
        {"pageCount", residency_.trackedPageCount()}, {"pages", std::move(pages)}, {"pagesTruncated", count < residency_.trackedPageCount()},
        {"stats", {{"residentPageCount", stats.residentPageCount}, {"pendingPageCount", stats.pendingPageCount},
            {"queuedUploadCount", stats.queuedUploadCount}, {"usedResidentBytes", stats.usedResidentBytes}, {"freeResidentBytes", stats.freeResidentBytes},
            {"pendingPageLoadCount", stats.pendingPageLoadCount}, {"activePageLoadCount", stats.activePageLoadCount},
            {"pendingPatchCount", stats.pendingPatchCount}, {"frameGpuRequestCount", stats.frameGpuRequestCount},
            {"frameGpuRequestOverflowCount", stats.frameGpuRequestOverflowCount}, {"frameGpuInvalidRequestCount", stats.frameGpuInvalidRequestCount},
            {"frameEvictedPageCount", stats.frameEvictedPageCount}, {"frameAllocationFailureCount", stats.frameAllocationFailureCount},
            {"frameEvictionScanCount", stats.frameEvictionScanCount}, {"frameEvictionCandidateTests", stats.frameEvictionCandidateTests},
            {"frameAllocationDeferredCount", stats.frameAllocationDeferredCount}, {"frameAdmissionDeferredCount", stats.frameAdmissionDeferredCount},
            {"frameCachedUnusedPageCount", stats.frameCachedUnusedPageCount}, {"frameResidentDemandTransitionCount", stats.frameResidentDemandTransitionCount},
            {"frameUploadBytes", stats.frameUploadBytes}, {"totalUploadBytes", stats.totalUploadBytes},
            {"frameStoredUploadBytes", stats.frameStoredUploadBytes}, {"totalStoredUploadBytes", stats.totalStoredUploadBytes},
            {"frameGpuDecompressedPages", stats.frameGpuDecompressedPages}, {"totalGpuDecompressedPages", stats.totalGpuDecompressedPages},
            {"totalEvictedPageCount", stats.totalEvictedPageCount}, {"totalCompletedUnloadCount", stats.totalCompletedUnloadCount},
            {"totalCancelledQueuedLoadCount", stats.totalCancelledQueuedLoadCount},
            {"totalPrefetchAdmitted", stats.totalPrefetchAdmitted}, {"totalPrefetchUsed", stats.totalPrefetchUsed},
            {"totalPrefetchDeferred", stats.totalPrefetchDeferred},
            {"totalCancelledUploads", stats.totalCancelledUploads},
            {"totalCompletionDrivenUploads", stats.totalCompletionDrivenUploads},
            {"totalCompletedPageLoadCount", stats.totalCompletedPageLoadCount},
            {"totalCompletedUploadCount", stats.totalCompletedUploadCount},
            {"totalPageLoadFailureCount", stats.totalPageLoadFailureCount},
            {"totalGpuInvalidRequestCount", stats.totalGpuInvalidRequestCount}}}};
}

} // namespace metallic::render
