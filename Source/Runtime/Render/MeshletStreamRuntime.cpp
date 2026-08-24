#include "Runtime/Render/MeshletStreamRuntime.h"

#include "Runtime/Render/MeshletStreamClas.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace metallic::render {
namespace {

inline constexpr bool kDefaultReversedZ = true;

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

std::string resultMessage(std::string_view label, const Result& result)
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

Result createNamedBuffer(
    Device& device,
    const BufferDesc& desc,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log,
    std::string_view label)
{
    Result result = device.createBuffer(desc, outBuffer);
    if (!result || outBuffer == nullptr) {
        log += resultMessage(std::string("createBuffer(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

Result createHostStorageBuffer(
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

Result updateHostBuffer(Buffer& buffer, const void* data, uint64_t byteSize)
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
        buffer.flush(0, byteSize);
    }
    buffer.unmap();
    return {};
}

template<typename ValueType, typename Populate>
Result createAndPopulateHostStorageBuffer(
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
    Result result = createHostStorageBuffer(device, byteSize, outBuffer, log, label);
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
    outBuffer->flush(0, byteSize);
    outBuffer->unmap();
    return {};
}

Result allocateAndWriteBuffer(
    BindlessHeap& heap,
    Buffer& buffer,
    BindlessHandle& outHandle,
    std::string& log,
    std::string_view label)
{
    Result result = heap.allocateBuffer(outHandle);
    if (!result || !outHandle.valid()) {
        log += resultMessage(std::string("allocateBuffer(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    result = heap.writeStorageBuffer(outHandle, buffer);
    if (!result) {
        log += resultMessage(std::string("writeStorageBuffer(") + std::string(label) + ")", result);
        log += '\n';
    }
    return result;
}

void transitionBuffer(
    CommandBuffer& commandBuffer,
    Buffer& buffer,
    ResourceState& state,
    ResourceState nextState,
    bool forceBarrier = false)
{
    if (!forceBarrier && state == nextState) {
        return;
    }
    BufferBarrierDesc barrier{
        .buffer = &buffer,
        .before = state,
        .after = nextState,
        .offset = 0,
        .size = buffer.desc().size,
    };
    commandBuffer.barrier(BarrierDesc{
        .buffers = &barrier,
        .bufferCount = 1,
    });
    state = nextState;
}

Result createSlangShaderModule(
    Device& device,
    const char* moduleName,
    const char* entryPoint,
    std::unique_ptr<ShaderModule>& outShader,
    std::string& log)
{
    ShaderCompileResult compileResult;
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPoint,
            .searchPath = kMeshletStreamShaderSearchPath,
        },
        compileResult);
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
    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            .debugName = shaderDebugName.c_str(),
        },
        outShader);
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
    Result initialize(Device& device, BindlessHeap& bindlessHeap, uint64_t updateByteSize, std::string& log)
    {
        if (updateByteSize == 0) {
            return makeError(Error::InvalidArgument);
        }
        Result result = createHostStorageBuffer(
            device,
            updateByteSize,
            updateBuffer_,
            log,
            "MeshletStreamRuntime update");
        if (!result) {
            return result;
        }

        result = allocateAndWriteBuffer(bindlessHeap, *updateBuffer_, updateHandle_, log, "meshlet stream update");
        if (!result) {
            return result;
        }

        result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamPageTableInitEntryPoint,
            pageTableInitShader_,
            log);
        if (!result) {
            return result;
        }
        result = device.createComputePipeline(
            ComputePipelineDesc{
                .computeShader = pageTableInitShader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            },
            pageTableInitPipeline_);
        if (!result || pageTableInitPipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime page table init)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamUpdateEntryPoint,
            updateShader_,
            log);
        if (!result) {
            return result;
        }

        result = device.createComputePipeline(
            ComputePipelineDesc{
                .computeShader = updateShader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            },
            updatePipeline_);
        if (!result || updatePipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime update)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    bool ready() const
    {
        return updateBuffer_ != nullptr &&
            updateHandle_.valid() &&
            pageTableInitShader_ != nullptr &&
            pageTableInitPipeline_ != nullptr &&
            updatePipeline_ != nullptr;
    }

    BindlessHandle updateHandle() const { return updateHandle_; }

    Result initializePageTable(
        CommandBuffer& commandBuffer,
        BindlessHeap& bindlessHeap,
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
        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General);
        commandBuffer.bindBindlessHeap(bindlessHeap);
        commandBuffer.bindComputePipeline(*pageTableInitPipeline_);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.dispatch(groupCountX, groupCountY, 1);
        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true);
        return {};
    }

    Result apply(
        CommandBuffer& commandBuffer,
        BindlessHeap& bindlessHeap,
        const MeshletStreamUserPush& push,
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

        void* mapped = updateBuffer_->map();
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
        updateBuffer_->flush(
            0,
            sizeof(StreamUpdateBufferHeader) + static_cast<uint64_t>(patchCount) * sizeof(StreamPageTablePatch));
        updateBuffer_->unmap();

        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General);
        commandBuffer.bindBindlessHeap(bindlessHeap);
        commandBuffer.bindComputePipeline(*updatePipeline_);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.dispatch((patchCount + 63u) / 64u, 1, 1);
        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true);
        return {};
    }

private:
    std::unique_ptr<Buffer> updateBuffer_;
    std::unique_ptr<ShaderModule> pageTableInitShader_;
    std::unique_ptr<ComputePipeline> pageTableInitPipeline_;
    std::unique_ptr<ShaderModule> updateShader_;
    std::unique_ptr<ComputePipeline> updatePipeline_;
    BindlessHandle updateHandle_;
};

class MeshletStreamRuntime::TraversalPass {
public:
    Result initialize(Device& device, std::string& log)
    {
        Result result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamTraversalEntryPoint,
            traversalShader_,
            log);
        if (!result) {
            return result;
        }

        result = device.createComputePipeline(
            ComputePipelineDesc{
                .computeShader = traversalShader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            },
            traversalPipeline_);
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

    Result dispatch(
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

        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General);
        transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General);
        commandBuffer.bindBindlessHeap(bindlessHeap);
        commandBuffer.bindComputePipeline(*traversalPipeline_);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.dispatch((threadCount + 63u) / 64u, 1, 1);
        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true);
        transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General, true);
        return {};
    }

private:
    std::unique_ptr<ShaderModule> traversalShader_;
    std::unique_ptr<ComputePipeline> traversalPipeline_;
};

class MeshletStreamRuntime::ActiveBuildPass {
public:
    Result initialize(Device& device, std::string& log)
    {
        Result result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamActiveBuildEntryPoint,
            activeBuildShader_,
            log);
        if (!result) {
            return result;
        }

        result = device.createComputePipeline(
            ComputePipelineDesc{
                .computeShader = activeBuildShader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            },
            activeBuildPipeline_);
        if (!result || activeBuildPipeline_ == nullptr) {
            log += resultMessage("createComputePipeline(MeshletStreamRuntime active build)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    bool ready() const
    {
        return activeBuildShader_ != nullptr && activeBuildPipeline_ != nullptr;
    }

    Result dispatch(
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

        transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General);
        transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, drawIndirectBuffer, drawIndirectBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, traversalHeaderBuffer, traversalHeaderBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, traversalWorkBuffer, traversalWorkBufferState, ResourceState::General);
        commandBuffer.bindBindlessHeap(bindlessHeap);
        commandBuffer.bindComputePipeline(*activeBuildPipeline_);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.dispatch((threadCount + 63u) / 64u, 1, 1);
        transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, pageTableBuffer, pageTableState, ResourceState::General, true);
        transitionBuffer(commandBuffer, requestBuffer, requestBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, drawIndirectBuffer, drawIndirectBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, traversalHeaderBuffer, traversalHeaderBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, traversalWorkBuffer, traversalWorkBufferState, ResourceState::General, true);
        return {};
    }

private:
    std::unique_ptr<ShaderModule> activeBuildShader_;
    std::unique_ptr<ComputePipeline> activeBuildPipeline_;
};

class MeshletStreamRuntime::BlasInputPass {
public:
    Result initialize(Device& device, std::string& log)
    {
        Result result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamBlasInputEntryPoint,
            blasInputShader_,
            log);
        if (!result) {
            return result;
        }

        result = device.createComputePipeline(
            ComputePipelineDesc{
                .computeShader = blasInputShader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            },
            blasInputPipeline_);
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

    Result dispatch(
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

        transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, blasHeaderBuffer, blasHeaderBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, instanceBlasBuffer, instanceBlasBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, blasBuildInfoBuffer, blasBuildInfoBufferState, ResourceState::General);
        transitionBuffer(
            commandBuffer,
            blasClusterReferenceBuffer,
            blasClusterReferenceBufferState,
            ResourceState::General);
        commandBuffer.bindBindlessHeap(bindlessHeap);
        commandBuffer.bindComputePipeline(*blasInputPipeline_);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.dispatch((threadCount + 63u) / 64u, 1, 1);
        transitionBuffer(commandBuffer, activeGroupBuffer, activeGroupBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, activeHeaderBuffer, activeHeaderBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, blasHeaderBuffer, blasHeaderBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, instanceBlasBuffer, instanceBlasBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, blasBuildInfoBuffer, blasBuildInfoBufferState, ResourceState::General, true);
        transitionBuffer(
            commandBuffer,
            blasClusterReferenceBuffer,
            blasClusterReferenceBufferState,
            ResourceState::General,
            true);
        return {};
    }

private:
    std::unique_ptr<ShaderModule> blasInputShader_;
    std::unique_ptr<ComputePipeline> blasInputPipeline_;
};

class MeshletStreamRuntime::TlasInputPass {
public:
    Result initialize(Device& device, std::string& log)
    {
        Result result = createSlangShaderModule(
            device,
            kMeshletStreamShaderModuleName,
            kMeshletStreamTlasInputEntryPoint,
            tlasInputShader_,
            log);
        if (!result) {
            return result;
        }
        result = device.createComputePipeline(
            ComputePipelineDesc{
                .computeShader = tlasInputShader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
            },
            tlasInputPipeline_);
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

    Result dispatch(
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
        transitionBuffer(commandBuffer, instanceBlasBuffer, instanceBlasBufferState, ResourceState::General);
        transitionBuffer(commandBuffer, tlasInstanceBuffer, tlasInstanceBufferState, ResourceState::General);
        commandBuffer.bindBindlessHeap(bindlessHeap);
        commandBuffer.bindComputePipeline(*tlasInputPipeline_);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.dispatch((threadCount + 63u) / 64u, 1, 1);
        transitionBuffer(commandBuffer, instanceBlasBuffer, instanceBlasBufferState, ResourceState::General, true);
        transitionBuffer(commandBuffer, tlasInstanceBuffer, tlasInstanceBufferState, ResourceState::General, true);
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

Result MeshletStreamRuntime::initialize(Device& device, const MeshletStreamRuntimeDesc& desc, std::string& log)
{
    reset();
    log.clear();

    std::string reason;
    scene::MeshletStreamAsset openedAsset;
    if (!openedAsset.open(desc.streamAssetPath, reason) || !openedAsset.isCurrentForSource(desc.sourcePath)) {
        log = "MeshletStreamRuntime failed to open current streamasset: " + reason;
        if (desc.autoBuildStreamAsset) {
            log += "; runtime auto-build is no longer supported, run Metallic --build-meshstream first";
        }
        return makeError(Error::Failure);
    }

    asset_ = std::move(openedAsset);
    drawBounds_ = computeDrawBounds(asset_);
    if (!drawBounds_.valid) {
        log = "MeshletStreamRuntime streamasset bounds are unavailable";
        return makeError(Error::Failure);
    }

    maxResidentPages_ = desc.maxResidentPages;
    maxPageUploadsPerFrame_ = desc.maxPageUploadsPerFrame;
    maxGpuPageRequests_ = std::max(desc.maxGpuPageRequests, 1u);
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

    BufferUsageBits pageBufferUsage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination;
    if (desc.enableClusterRtx) {
        if (!device.capabilities().clusterAccelerationStructure) {
            log = "MeshletStreamRuntime cluster RTX requires cluster acceleration structure support";
            return makeError(Error::Unsupported);
        }
        pageBufferUsage = pageBufferUsage |
            BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::AccelerationStructureBuildInput;
    }

    Result result = device.createBuffer(
        BufferDesc{
            .size = maxResidentBytes_,
            .structureStride = 0,
            .usage = pageBufferUsage,
            .memoryLocation = MemoryLocation::Device,
        },
        pageBuffer_);
    if (!result || pageBuffer_ == nullptr) {
        log += resultMessage("createBuffer(MeshletStreamRuntime pages)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    pageBufferState_ = ResourceState::Undefined;

    if (!residency_.initialize(
            MeshletStreamResidencyDesc{
                .asset = &asset_,
                .maxResidentBytes = maxResidentBytes_,
                .maxResidentPages = maxResidentPages_,
                .queuedFrameCount = std::max(desc.queuedFrameCount, 1u),
                .pageStride = pageStride,
                .pageLoadConcurrency = desc.pageLoadConcurrency,
                .maxPageLoadsInFlight = desc.maxPageLoadsInFlight,
            },
            reason)) {
        log = "MeshletStreamRuntime residency initialization failed: " + reason;
        return makeError(Error::Failure);
    }

    const MeshletStreamStorage& residencyStorage = residency_.storage();
    const uint64_t streamedPageReserveBytes =
        residencyStorage.allocationSize(asset_.maxPagePayloadBytes());
    const uint64_t lockedFallbackByteBudget = residencyStorage.capacityBytes() > streamedPageReserveBytes
        ? residencyStorage.capacityBytes() - streamedPageReserveBytes
        : 0;
    uint64_t lockedFallbackPageBudget = desc.maxLockedFallbackPages;
    if (maxResidentPages_ != 0) {
        const uint32_t pageBudgetWithStreamingReserve = maxResidentPages_ > 1
            ? maxResidentPages_ - 1
            : 0;
        lockedFallbackPageBudget = std::min<uint64_t>(
            lockedFallbackPageBudget,
            pageBudgetWithStreamingReserve);
    }
    lockedFallbackPageBudget = std::min<uint64_t>(
        lockedFallbackPageBudget,
        lockedFallbackByteBudget / residencyStorage.alignmentBytes());

    std::vector<uint32_t> fallbackPages;
    std::vector<uint32_t> lockedFallbackPrimitives;
    fallbackPages.reserve(static_cast<size_t>(std::min<uint64_t>(
        lockedFallbackPageBudget,
        asset_.pageCount())));
    lockedFallbackPrimitives.reserve(static_cast<size_t>(std::min<uint64_t>(
        lockedFallbackPageBudget,
        asset_.primitiveCount())));
    const uint64_t fallbackPrimitiveCandidateBudget = std::min<uint64_t>(
        asset_.primitiveCount(),
        std::max<uint64_t>(lockedFallbackPageBudget, 1u) * 4u);
    std::unordered_set<uint32_t> consideredFallbackPrimitives;
    consideredFallbackPrimitives.reserve(static_cast<size_t>(fallbackPrimitiveCandidateBudget));
    uint64_t lockedFallbackBytes = 0;
    for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
        if (fallbackPages.size() == lockedFallbackPageBudget ||
            lockedFallbackByteBudget - lockedFallbackBytes < residencyStorage.alignmentBytes() ||
            consideredFallbackPrimitives.size() == fallbackPrimitiveCandidateBudget) {
            break;
        }
        if (instance.visible == 0 ||
            instance.primitiveIndex >= asset_.primitiveCount() ||
            !consideredFallbackPrimitives.insert(instance.primitiveIndex).second) {
            continue;
        }
        const uint32_t primitiveIndex = instance.primitiveIndex;
        const scene::MeshletStreamPrimitiveInfo& primitive = asset_.primitives()[primitiveIndex];
        if (primitive.fallbackPageOffset > asset_.pageCount() ||
            primitive.fallbackPageCount > asset_.pageCount() - primitive.fallbackPageOffset) {
            log = "MeshletStreamRuntime primitive fallback page range exceeds the asset page count";
            return makeError(Error::Failure);
        }

        uint64_t primitiveFallbackBytes = 0;
        for (uint32_t localPage = 0; localPage < primitive.fallbackPageCount; ++localPage) {
            const uint32_t pageIndex = primitive.fallbackPageOffset + localPage;
            const uint64_t pageBytes = residencyStorage.allocationSize(
                asset_.pages()[pageIndex].uncompressedSize);
            if (pageBytes == 0 ||
                pageBytes > std::numeric_limits<uint64_t>::max() - primitiveFallbackBytes) {
                log = "MeshletStreamRuntime primitive fallback page byte budget overflowed";
                return makeError(Error::Failure);
            }
            primitiveFallbackBytes += pageBytes;
        }
        if (primitive.fallbackPageCount > lockedFallbackPageBudget - fallbackPages.size() ||
            primitiveFallbackBytes > lockedFallbackByteBudget - lockedFallbackBytes) {
            continue;
        }

        lockedFallbackPrimitives.push_back(primitiveIndex);
        lockedFallbackBytes += primitiveFallbackBytes;
        for (uint32_t localPage = 0; localPage < primitive.fallbackPageCount; ++localPage) {
            fallbackPages.push_back(primitive.fallbackPageOffset + localPage);
        }
    }
    if (!residency_.lockFallbackPages(fallbackPages, reason)) {
        log = "MeshletStreamRuntime fallback residency initialization failed: " + reason;
        return makeError(Error::Failure);
    }

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

    if (desc.enableClusterRtx) {
        const uint64_t defaultBuildClusters =
            static_cast<uint64_t>(std::max(maxPageUploadsPerFrame_, 1u)) * asset_.maxPageClusters();
        const uint64_t buildClusters = desc.maxClasBuildClusters != 0
            ? desc.maxClasBuildClusters
            : defaultBuildClusters;
        if (desc.maxClasBytes == 0 ||
            buildClusters == 0 ||
            buildClusters > std::numeric_limits<uint32_t>::max()) {
            log = "MeshletStreamRuntime cluster RTX capacities are invalid";
            return makeError(Error::InvalidArgument);
        }
        clasPool_ = std::make_unique<MeshletStreamClasPool>();
        result = clasPool_->initialize(
            device,
            MeshletStreamClasPoolDesc{
                .asset = &asset_,
                .maxStorageBytes = desc.maxClasBytes,
                .maxBuildClusters = static_cast<uint32_t>(buildClusters),
                .queuedFrameCount = std::max(desc.queuedFrameCount, 1u),
            },
            log);
        if (!result) {
            log = "MeshletStreamRuntime CLAS pool initialization failed: " + log;
            return result;
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
    if (clasPool_ != nullptr) {
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
    const uint64_t requestByteSize =
        sizeof(StreamRequestBufferHeader) +
        (static_cast<uint64_t>(maxGpuPageRequests_) + maxGpuPageUnloadRequests_) * sizeof(uint32_t);
    const uint64_t updateByteSize =
        sizeof(StreamUpdateBufferHeader) + static_cast<uint64_t>(maxUpdatePatches_) * sizeof(StreamPageTablePatch);

    result = initializeSceneMetadataBuffers(device, log);
    if (!result) {
        return result;
    }

    result = createNamedBuffer(
        device,
        BufferDesc{
            .size = static_cast<uint64_t>(maxActiveGroups_) * sizeof(MeshletStreamGpuActiveGroup),
            .structureStride = sizeof(MeshletStreamGpuActiveGroup),
            .usage = BufferUsageBits::Storage,
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
            .size = sizeof(MeshletStreamGpuActiveHeader),
            .structureStride = sizeof(MeshletStreamGpuActiveHeader),
            .usage = BufferUsageBits::Storage,
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
            .size = sizeof(MeshletStreamGpuDrawIndirect),
            .structureStride = sizeof(MeshletStreamGpuDrawIndirect),
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
            .size = sizeof(MeshletStreamGpuTraversalHeader),
            .structureStride = sizeof(MeshletStreamGpuTraversalHeader),
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
            .size = static_cast<uint64_t>(traversalWorkCapacity_) * sizeof(MeshletStreamGpuTraversalWorkItem),
            .structureStride = sizeof(MeshletStreamGpuTraversalWorkItem),
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
    if (clasPool_ != nullptr) {
        ClusterAccelerationStructureProperties clusterProperties;
        result = device.queryClusterAccelerationStructureProperties(clusterProperties);
        if (!result) {
            log = std::string("queryClusterAccelerationStructureProperties(stream BLAS) returned ") +
                resultToString(result);
            return result;
        }
        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = sizeof(MeshletStreamGpuBlasHeader),
                .structureStride = sizeof(MeshletStreamGpuBlasHeader),
                .usage = BufferUsageBits::Storage |
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

        std::vector<uint32_t> primitiveClusterCapacities(asset_.primitiveCount(), 0);
        const std::span<const scene::MeshletStreamGroupInfo> groups = asset_.groups();
        for (uint32_t primitiveIndex = 0; primitiveIndex < asset_.primitiveCount(); ++primitiveIndex) {
            const scene::MeshletStreamPrimitiveInfo& primitive = asset_.primitives()[primitiveIndex];
            uint32_t capacity = 0;
            for (uint32_t localGroup = 0;
                 localGroup < primitive.groupCount && capacity < blasClusterReferenceCapacity_;
                 ++localGroup) {
                const uint32_t remaining = blasClusterReferenceCapacity_ - capacity;
                capacity += std::min(
                    groups[primitive.groupOffset + localGroup].clusterCount,
                    remaining);
            }
            primitiveClusterCapacities[primitiveIndex] = capacity;
        }
        if (desc.maxBlasBytes == 0 || desc.maxBlasBuilds == 0) {
            log = "MeshletStreamRuntime cluster RTX BLAS budgets must be non-zero";
            return makeError(Error::InvalidArgument);
        }

        uint32_t referenceLimit = blasClusterReferenceCapacity_;
        uint32_t buildLimit = std::min(desc.maxBlasBuilds, asset_.instanceCount());
        ClusterAccelerationStructureBuildSizes blasSizes;
        struct BlasCapacityPlan {
            uint32_t referenceCount = 0;
            uint32_t buildCount = 0;
            uint32_t maxClustersPerBuild = 0;
        };
        auto planBlasCapacities = [this, &primitiveClusterCapacities](
                                      uint32_t plannedReferenceLimit,
                                      uint32_t plannedBuildLimit) {
            BlasCapacityPlan plan;
            for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
                if (plan.buildCount == plannedBuildLimit ||
                    plan.referenceCount == plannedReferenceLimit) {
                    break;
                }
                if (instance.visible == 0 ||
                    instance.primitiveIndex >= primitiveClusterCapacities.size()) {
                    continue;
                }
                const uint32_t remaining = plannedReferenceLimit - plan.referenceCount;
                const uint32_t capacity = std::min(
                    primitiveClusterCapacities[instance.primitiveIndex],
                    remaining);
                if (capacity == 0) {
                    continue;
                }
                plan.referenceCount += capacity;
                plan.maxClustersPerBuild = std::max(plan.maxClustersPerBuild, capacity);
                ++plan.buildCount;
            }
            return plan;
        };
        BlasCapacityPlan blasPlan;
        for (;;) {
            blasPlan = planBlasCapacities(referenceLimit, buildLimit);
            if (blasPlan.referenceCount == 0 ||
                blasPlan.buildCount == 0 ||
                blasPlan.maxClustersPerBuild == 0) {
                log = "MeshletStreamRuntime cluster RTX BLAS budget cannot cover one visible instance";
                return makeError(Error::OutOfMemory);
            }

            result = device.queryClusterAccelerationStructureBottomLevelBuildSizes(
                ClusterAccelerationStructureBottomLevelBuildSizesDesc{
                    .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                    .maxClusterCountPerAccelerationStructure = blasPlan.maxClustersPerBuild,
                    .maxTotalClusterCount = blasPlan.referenceCount,
                    .maxAccelerationStructureCount = blasPlan.buildCount,
                },
                blasSizes);
            if (!result || blasSizes.accelerationStructureSize == 0 || blasSizes.buildScratchSize == 0) {
                log = std::string("queryClusterAccelerationStructureBottomLevelBuildSizes(stream BLAS) returned ") +
                    resultToString(result);
                return result ? makeError(Error::Failure) : result;
            }
            if (blasSizes.accelerationStructureSize <= desc.maxBlasBytes) {
                blasClusterReferenceCapacity_ = blasPlan.referenceCount;
                blasBuildCapacity_ = blasPlan.buildCount;
                maxBlasClustersPerBuild_ = blasPlan.maxClustersPerBuild;
                break;
            }
            if (referenceLimit == 1 && buildLimit == 1) {
                log = "MeshletStreamRuntime maxBlasBytes cannot hold one dynamic cluster BLAS";
                return makeError(Error::OutOfMemory);
            }
            referenceLimit = std::max(referenceLimit / 2u, 1u);
            buildLimit = std::max(buildLimit / 2u, 1u);
        }

        uint32_t instanceReferenceOffset = 0;
        uint32_t instanceBuildCount = 0;
        const std::span<const scene::MeshletStreamInstanceInfo> instances = asset_.instances();
        result = createAndPopulateHostStorageBuffer<MeshletStreamGpuInstanceBlas>(
            device,
            instances.size(),
            instanceBlasBuffer_,
            log,
            "MeshletStreamRuntime instance BLAS inputs",
            [instances,
             &primitiveClusterCapacities,
             &instanceReferenceOffset,
             &instanceBuildCount,
             this](MeshletStreamGpuInstanceBlas& instanceBlas, size_t index) {
                instanceBlas = MeshletStreamGpuInstanceBlas{};
                const scene::MeshletStreamInstanceInfo& instance = instances[index];
                if (instance.visible == 0 ||
                    instance.primitiveIndex >= primitiveClusterCapacities.size() ||
                    instanceBuildCount == blasBuildCapacity_ ||
                    instanceReferenceOffset == blasClusterReferenceCapacity_) {
                    return;
                }
                const uint32_t remaining =
                    blasClusterReferenceCapacity_ - instanceReferenceOffset;
                const uint32_t capacity = std::min(
                    primitiveClusterCapacities[instance.primitiveIndex],
                    remaining);
                if (capacity == 0) {
                    return;
                }
                instanceBlas.clusterReferenceOffset = instanceReferenceOffset;
                instanceBlas.clusterReferenceCapacity = capacity;
                instanceReferenceOffset += capacity;
                ++instanceBuildCount;
            });
        if (!result) {
            return result;
        }
        if (instanceReferenceOffset != blasClusterReferenceCapacity_ ||
            instanceBuildCount != blasBuildCapacity_) {
            log = "MeshletStreamRuntime instance BLAS capacity plan changed while populating inputs";
            return makeError(Error::Failure);
        }
        instanceBlasBufferState_ = ResourceState::Undefined;

        result = createNamedBuffer(
            device,
            BufferDesc{
                .size = std::max<uint64_t>(
                    static_cast<uint64_t>(blasBuildCapacity_) * sizeof(MeshletStreamGpuBlasBuildInfo),
                    sizeof(MeshletStreamGpuBlasBuildInfo)),
                .structureStride = sizeof(MeshletStreamGpuBlasBuildInfo),
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
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
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
            const scene::MeshletStreamPrimitiveInfo& primitive = asset_.primitives()[primitiveIndex];
            uint64_t primitiveReferences = 0;
            for (uint32_t localPage = 0; localPage < primitive.fallbackPageCount; ++localPage) {
                const uint32_t clusterCount =
                    asset_.pages()[primitive.fallbackPageOffset + localPage].clusterCount;
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
                result = device.queryClusterAccelerationStructureBottomLevelBuildSizes(
                    ClusterAccelerationStructureBottomLevelBuildSizesDesc{
                        .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                        .maxClusterCountPerAccelerationStructure = clusterCount,
                        .maxTotalClusterCount = clusterCount,
                        .maxAccelerationStructureCount = 1,
                    },
                    iter->second);
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
                continue;
            }
            if (primitiveReferences > std::numeric_limits<uint64_t>::max() - totalFallbackReferences) {
                log = "MeshletStreamRuntime total fallback CLAS reference count overflowed";
                return makeError(Error::InvalidArgument);
            }
            fallbackBlasPrimitives_.push_back(FallbackBlasPrimitive{
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
            fallbackPrimitiveCount * sizeof(MeshletStreamGpuBlasBuildInfo);
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
        auto* fallbackBuildInfos = static_cast<MeshletStreamGpuBlasBuildInfo*>(
            fallbackBlasBuildInfoBuffer_->map());
        if (fallbackBuildInfos == nullptr) {
            log = "MeshletStreamRuntime fallback BLAS build info buffer map failed";
            return makeError(Error::Failure);
        }
        for (size_t fallbackIndex = 0;
             fallbackIndex < fallbackBlasPrimitives_.size();
             ++fallbackIndex) {
            const FallbackBlasPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
            const uint64_t referenceAddress = fallbackReferenceAddress +
                fallback.referenceOffset * sizeof(uint64_t);
            fallbackBuildInfos[fallbackIndex] = MeshletStreamGpuBlasBuildInfo{
                .clusterReferencesCount = fallback.referenceCount,
                .clusterReferencesStride = sizeof(uint64_t),
                .clusterReferencesAddressLow = static_cast<uint32_t>(referenceAddress),
                .clusterReferencesAddressHigh = static_cast<uint32_t>(referenceAddress >> 32u),
            };
        }
        fallbackBlasBuildInfoBuffer_->flush(0, fallbackBuildInfoBytes);
        fallbackBlasBuildInfoBuffer_->unmap();

        result = createFallbackHostBuffer(
            fallbackDestinationBytes,
            BufferUsageBits::Storage |
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
        fallbackBlasDestinationBuffer_->flush(0, fallbackDestinationBytes);
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
        fallbackBlasAddressBuffer_->flush(0, primitiveAddressBytes);
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
                    sizeof(RayTracingGpuInstance),
                .structureStride = sizeof(RayTracingGpuInstance),
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
        result = device.queryRayTracingAccelerationStructureBuildSizes(tlasInputs, tlasSizes);
        if (!result) {
            log = resultMessage(
                "queryRayTracingAccelerationStructureBuildSizes(MeshletStreamRuntime TLAS)",
                result);
            return result;
        }

        RayTracingAccelerationStructureProperties rtasProperties;
        result = device.queryRayTracingAccelerationStructureProperties(rtasProperties);
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
        result = device.createRayTracingAccelerationStructure(
            RayTracingAccelerationStructureDesc{
                .type = RayTracingAccelerationStructureType::TopLevel,
                .buildFlags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                .size = tlasSizes.accelerationStructureSize,
            },
            tlas_);
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
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
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
            .size = requestByteSize,
            .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback,
        },
        requestReadbackBuffer_,
        log,
        "MeshletStreamRuntime request readback");
    if (!result) {
        return result;
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
        sizeof(MeshletStreamGpuParams),
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
                sizeof(VisibleClusterRecord),
            .structureStride = sizeof(VisibleClusterRecord),
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        },
        visibleClusterBuffer_,
        log,
        "MeshletStreamRuntime visible clusters");
    if (!result) {
        return result;
    }
    visibleClusterBufferState_ = ResourceState::Undefined;
    result = createHostStorageBuffer(
        device,
        sizeof(MeshletStreamGpuRasterBindings),
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

    result = device.createBindlessHeap(
        BindlessHeapDesc{
            .maxSamplers = 0,
            .maxSampledImages = 4,
            .maxBuffers = 64u +
                static_cast<uint32_t>(residentPageFrames_.size()),
        },
        bindlessHeap_);
    if (!result || bindlessHeap_ == nullptr) {
        log += resultMessage("createBindlessHeap(MeshletStreamRuntime)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *pageBuffer_, pageHandle_, log, "meshlet stream pages");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *activeGroupBuffer_, activeGroupHandle_, log, "meshlet stream active groups");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *activeHeaderBuffer_, activeHeaderHandle_, log, "meshlet stream active header");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *pageTableBuffer_, pageTableHandle_, log, "meshlet stream page table");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *paramsBuffer_, paramsHandle_, log, "meshlet stream params");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *bindlessHeap_,
        *visibleClusterBuffer_,
        visibleClusterHandle_,
        log,
        "meshlet stream visible clusters");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *bindlessHeap_,
        *rasterBindingsBuffer_,
        rasterBindingsHandle_,
        log,
        "meshlet stream raster bindings");
    if (!result) {
        return result;
    }
    result = updateRasterBindings(MeshletStreamGpuRasterBindings{});
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *requestBuffer_, requestHandle_, log, "meshlet stream request");
    if (!result) {
        return result;
    }
    for (ResidentPageFrame& frame : residentPageFrames_) {
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *frame.buffer,
            frame.handle,
            log,
            "meshlet stream resident pages");
        if (!result) {
            return result;
        }
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *instanceBuffer_, instanceHandle_, log, "meshlet stream instances");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *primitiveBuffer_, primitiveHandle_, log, "meshlet stream primitives");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *lodLevelBuffer_, lodLevelHandle_, log, "meshlet stream LOD levels");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *groupBuffer_, groupHandle_, log, "meshlet stream groups");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *nodeBuffer_, nodeHandle_, log, "meshlet stream hierarchy nodes");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *bindlessHeap_,
        *drawIndirectBuffer_,
        drawIndirectHandle_,
        log,
        "meshlet stream draw indirect");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *bindlessHeap_,
        *traversalHeaderBuffer_,
        traversalHeaderHandle_,
        log,
        "meshlet stream traversal header");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(
        *bindlessHeap_,
        *traversalWorkBuffer_,
        traversalWorkHandle_,
        log,
        "meshlet stream traversal work");
    if (!result) {
        return result;
    }
    if (clasPool_ != nullptr) {
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *clasPool_->clusterAddressBuffer(),
            clasAddressHandle_,
            log,
            "meshlet stream CLAS addresses");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *clasPool_->pageTableBuffer(),
            clasPageTableHandle_,
            log,
            "meshlet stream CLAS page table");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *blasHeaderBuffer_,
            blasHeaderHandle_,
            log,
            "meshlet stream BLAS header");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *instanceBlasBuffer_,
            instanceBlasHandle_,
            log,
            "meshlet stream instance BLAS inputs");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *blasBuildInfoBuffer_,
            blasBuildInfoHandle_,
            log,
            "meshlet stream BLAS build infos");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *blasClusterReferenceBuffer_,
            blasClusterReferenceHandle_,
            log,
            "meshlet stream BLAS cluster references");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *fallbackBlasAddressBuffer_,
            fallbackBlasAddressHandle_,
            log,
            "meshlet stream fallback BLAS addresses");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *blasAddressBuffer_,
            dynamicBlasAddressHandle_,
            log,
            "meshlet stream dynamic BLAS addresses");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *tlasInstanceBuffer_,
            tlasInstanceHandle_,
            log,
            "meshlet stream TLAS instances");
        if (!result) {
            return result;
        }
    }

    updatePass_ = std::make_unique<UpdatePass>();
    result = updatePass_->initialize(device, *bindlessHeap_, updateByteSize, log);
    if (!result) {
        return result;
    }
    traversalPass_ = std::make_unique<TraversalPass>();
    result = traversalPass_->initialize(device, log);
    if (!result) {
        return result;
    }
    activeBuildPass_ = std::make_unique<ActiveBuildPass>();
    result = activeBuildPass_->initialize(device, log);
    if (!result) {
        return result;
    }
    if (clasPool_ != nullptr) {
        blasInputPass_ = std::make_unique<BlasInputPass>();
        result = blasInputPass_->initialize(device, log);
        if (!result) {
            return result;
        }
        tlasInputPass_ = std::make_unique<TlasInputPass>();
        result = tlasInputPass_->initialize(device, log);
        if (!result) {
            return result;
        }
    }

    return {};
}

void MeshletStreamRuntime::reset()
{
    tlasBuilt_ = false;
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
    requestClearBuffer_.reset();
    paramsBuffer_.reset();
    visibleClusterBuffer_.reset();
    rasterBindingsBuffer_.reset();
    residentPageFrames_.clear();
    instanceBuffer_.reset();
    primitiveBuffer_.reset();
    lodLevelBuffer_.reset();
    groupBuffer_.reset();
    nodeBuffer_.reset();
    drawIndirectBuffer_.reset();
    traversalHeaderBuffer_.reset();
    traversalWorkBuffer_.reset();
    blasHeaderBuffer_.reset();
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
    bindlessHeap_.reset();
    updatePass_.reset();
    traversalPass_.reset();
    activeBuildPass_.reset();
    blasInputPass_.reset();
    tlasInputPass_.reset();
    clasPool_.reset();
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
    pageTableInitialized_ = false;
    requestReadbackValid_ = false;
    frameIndex_ = 0;
    maxResidentPages_ = 0;
    maxPageUploadsPerFrame_ = 0;
    maxGpuPageRequests_ = 0;
    maxGpuPageUnloadRequests_ = 0;
    maxUpdatePatches_ = 0;
    residentPageCapacity_ = 0;
    currentResidentPageCount_ = 0;
    maxResidentBytes_ = 0;
    maxActiveGroups_ = 0;
    maxActiveGroupClusters_ = 0;
    maxPrimitiveGroupCount_ = 0;
    traversalWorkerCount_ = 0;
    traversalWorkCapacity_ = 0;
    blasClusterReferenceCapacity_ = 0;
    blasBuildCapacity_ = 0;
    maxBlasClustersPerBuild_ = 0;
    blasClusterReferenceAddress_ = 0;
    fallbackBlasPrimitives_.clear();
    currentFrameUploadCount_ = 0;
    previousFrameParams_ = {};
    previousFrameParamsValid_ = false;
}

bool MeshletStreamRuntime::ready() const
{
    return asset_.valid() &&
        bindlessHeap_ != nullptr &&
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
        nodeBuffer_ != nullptr &&
        drawIndirectBuffer_ != nullptr &&
        traversalHeaderBuffer_ != nullptr &&
        traversalWorkBuffer_ != nullptr &&
        (clasPool_ == nullptr ||
            (clasPool_->ready() &&
                blasInputPass_ != nullptr &&
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

RayTracingAccelerationStructure* MeshletStreamRuntime::accelerationStructure() const
{
    return tlasBuilt_ ? tlas_.get() : nullptr;
}

Result MeshletStreamRuntime::cmdBeginFrame(
    CommandBuffer& commandBuffer,
    Streamer& streamer,
    const MeshletStreamFrameDesc& frame)
{
    (void)commandBuffer;
    (void)frame;
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }

    ++frameIndex_;
    residency_.beginFrame();
    if (clasPool_ != nullptr) {
        clasPool_->beginFrame();
        clasPool_->retirePages(residency_.newlyUnloadedPages());
    }
    consumeGpuRequestReadback();
    currentFrameUploadCount_ = residency_.processUploads(streamer, *pageBuffer_, maxPageUploadsPerFrame_);
    const std::span<const uint32_t> residentPages = residency_.residentPages();
    if (residentPages.size() > residentPageCapacity_ || residentPageFrames_.empty()) {
        return makeError(Error::Failure);
    }
    currentResidentPageCount_ = static_cast<uint32_t>(residentPages.size());
    if (!residentPages.empty()) {
        ResidentPageFrame& residentFrame = residentPageFrames_[frameIndex_ % residentPageFrames_.size()];
        const Result residentUpdate = updateHostBuffer(
            *residentFrame.buffer,
            residentPages.data(),
            static_cast<uint64_t>(residentPages.size()) * sizeof(uint32_t));
        if (!residentUpdate) {
            return residentUpdate;
        }
    }
    return {};
}

Result MeshletStreamRuntime::cmdPreTraversal(CommandBuffer& commandBuffer, const MeshletStreamFrameDesc& frame)
{
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }

    Result result = initializePageTableIfNeeded(commandBuffer);
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
    result = buildActiveTable(commandBuffer);
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

    std::vector<MeshletStreamClasPageBuild> clasBuilds;
    clasBuilds.reserve(
        residency_.newlyResidentPages().size() + residency_.residentPages().size());
    for (uint32_t pageIndex : residency_.newlyResidentPages()) {
        const uint64_t deviceOffset = residency_.deviceOffsetForPage(pageIndex);
        if (deviceOffset != UINT64_MAX) {
            clasBuilds.push_back(MeshletStreamClasPageBuild{
                .pageIndex = pageIndex,
                .deviceOffsetBytes = deviceOffset,
            });
        }
    }
    for (uint32_t pageIndex : residency_.residentPages()) {
        if (clasPool_->pageHasClas(pageIndex)) {
            continue;
        }
        const uint64_t deviceOffset = residency_.deviceOffsetForPage(pageIndex);
        if (deviceOffset != UINT64_MAX) {
            clasBuilds.push_back(MeshletStreamClasPageBuild{
                .pageIndex = pageIndex,
                .deviceOffsetBytes = deviceOffset,
            });
        }
    }
    if (!clasBuilds.empty()) {
        std::string clasLog;
        result = clasPool_->cmdBuildPages(commandBuffer, *pageBuffer_, clasBuilds, clasLog);
        if (!result) {
            return result;
        }
    }
    result = cmdBuildFallbackBlas(commandBuffer);
    if (!result) {
        return result;
    }
    result = buildBlasInputs(commandBuffer);
    if (!result) {
        return result;
    }
    result = cmdBuildBlas(commandBuffer);
    if (!result) {
        return result;
    }
    result = buildTlasInstances(commandBuffer);
    if (!result) {
        return result;
    }
    return cmdBuildTlas(commandBuffer);
}

Result MeshletStreamRuntime::cmdPostTraversal(CommandBuffer& commandBuffer)
{
    (void)commandBuffer;
    return ready() ? Result{} : makeError(Error::InvalidArgument);
}

Result MeshletStreamRuntime::cmdEndFrame(CommandBuffer& commandBuffer)
{
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }

    Result result = copyRequestBufferForReadback(commandBuffer);
    if (!result) {
        return result;
    }

    if (currentFrameUploadCount_ > 0 && pageBufferState_ != ResourceState::TransferDestination) {
        transitionBuffer(commandBuffer, *pageBuffer_, pageBufferState_, ResourceState::TransferDestination);
    }
    return {};
}

MeshletStreamUserPush MeshletStreamRuntime::userPush() const
{
    return MeshletStreamUserPush{
        .pageBuffer = pageHandle_.index,
        .activeGroupBuffer = activeGroupHandle_.index,
        .pageTableBuffer = pageTableHandle_.index,
        .paramsBuffer = paramsHandle_.index,
        .requestBuffer = requestHandle_.index,
        .residentPageBuffer = !residentPageFrames_.empty()
            ? residentPageFrames_[frameIndex_ % residentPageFrames_.size()].handle.index
            : 0u,
        .updateBuffer = updatePass_ != nullptr ? updatePass_->updateHandle().index : 0u,
        .activeHeaderBuffer = activeHeaderHandle_.index,
        .instanceBuffer = instanceHandle_.index,
        .primitiveBuffer = primitiveHandle_.index,
        .lodLevelBuffer = lodLevelHandle_.index,
        .groupBuffer = groupHandle_.index,
        .nodeBuffer = nodeHandle_.index,
        .drawIndirectBuffer = drawIndirectHandle_.index,
        .traversalHeaderBuffer = traversalHeaderHandle_.index,
        .traversalWorkBuffer = traversalWorkHandle_.index,
        .clasAddressBuffer = clasAddressHandle_.valid() ? clasAddressHandle_.index : 0u,
        .clasPageTableBuffer = clasPageTableHandle_.valid() ? clasPageTableHandle_.index : 0u,
        .blasHeaderBuffer = blasHeaderHandle_.valid() ? blasHeaderHandle_.index : 0u,
        .instanceBlasBuffer = instanceBlasHandle_.valid() ? instanceBlasHandle_.index : 0u,
        .blasBuildInfoBuffer = blasBuildInfoHandle_.valid() ? blasBuildInfoHandle_.index : 0u,
        .blasClusterReferenceBuffer = blasClusterReferenceHandle_.valid()
            ? blasClusterReferenceHandle_.index
            : 0u,
        .fallbackBlasAddressBuffer = fallbackBlasAddressHandle_.valid()
            ? fallbackBlasAddressHandle_.index
            : 0u,
        .dynamicBlasAddressBuffer = dynamicBlasAddressHandle_.valid()
            ? dynamicBlasAddressHandle_.index
            : 0u,
        .tlasInstanceBuffer = tlasInstanceHandle_.valid() ? tlasInstanceHandle_.index : 0u,
        .rasterBindingsBuffer = rasterBindingsHandle_.index,
    };
}

Result MeshletStreamRuntime::updateRasterBindings(
    const MeshletStreamGpuRasterBindings& bindings)
{
    if (rasterBindingsBuffer_ == nullptr || !visibleClusterHandle_.valid()) {
        return makeError(Error::InvalidArgument);
    }
    MeshletStreamGpuRasterBindings resolved = bindings;
    resolved.visibleClusterBuffer = visibleClusterHandle_.index;
    return updateHostBuffer(*rasterBindingsBuffer_, &resolved, sizeof(resolved));
}

Result MeshletStreamRuntime::cmdPrepareVisibility(CommandBuffer& commandBuffer)
{
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }
    transitionBuffer(
        commandBuffer,
        *drawIndirectBuffer_,
        drawIndirectBufferState_,
        ResourceState::IndirectArgument);
    transitionBuffer(
        commandBuffer,
        *visibleClusterBuffer_,
        visibleClusterBufferState_,
        ResourceState::General,
        visibleClusterBufferState_ == ResourceState::General);
    return {};
}

Result MeshletStreamRuntime::cmdPrepareDeferred(CommandBuffer& commandBuffer)
{
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }
    transitionBuffer(
        commandBuffer,
        *visibleClusterBuffer_,
        visibleClusterBufferState_,
        ResourceState::ShaderRead,
        true);
    return {};
}

MeshletStreamDeferredGpuResourcesView MeshletStreamRuntime::deferredGpuResources() const
{
    return MeshletStreamDeferredGpuResourcesView{
        .pageBuffer = pageBuffer_.get(),
        .activeGroupBuffer = activeGroupBuffer_.get(),
        .pageTableBuffer = pageTableBuffer_.get(),
        .activeHeaderBuffer = activeHeaderBuffer_.get(),
        .paramsBuffer = paramsBuffer_.get(),
        .visibleClusterBuffer = visibleClusterBuffer_.get(),
        .visibleRecordCapacity = visibleClusterCapacity(),
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

Result MeshletStreamRuntime::syncRuntimeScene(const scene::Scene& scene, std::string& log)
{
    const std::span<const scene::MeshletStreamInstanceInfo> instances = asset_.instances();
    std::vector<uint32_t> runtimeRenderNodeIndices(instances.size());
    for (size_t index = 0; index < instances.size(); ++index) {
        runtimeRenderNodeIndices[index] = instances[index].renderNodeIndex;
    }
    return syncRuntimeScene(scene, runtimeRenderNodeIndices, log);
}

Result MeshletStreamRuntime::syncRuntimeScene(
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
    if (instances.size() * sizeof(MeshletStreamGpuInstance) != instanceBuffer_->desc().size) {
        log = "MeshletStreamRuntime instance layout changed.";
        return makeError(Error::InvalidArgument);
    }

    std::vector<MeshletStreamGpuInstance> gpuInstances(instances.size());
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
        MeshletStreamGpuInstance& gpuInstance = gpuInstances[index];
        gpuInstance = MeshletStreamGpuInstance{};
        gpuInstance.primitiveIndex = instance.primitiveIndex;
        gpuInstance.materialIndex = instance.materialIndex;
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
    instanceBuffer_->flush(0, instanceBuffer_->desc().size);
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

Result MeshletStreamRuntime::syncGPUSceneInstanceMapping(std::span<const uint32_t> mapping)
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
    auto* gpuInstances = static_cast<MeshletStreamGpuInstance*>(mapped);
    for (size_t index = 0; index < mapping.size(); ++index) {
        gpuInstances[index].gpuSceneInstanceIndex = mapping[index];
    }
    instanceBuffer_->flush(0, instanceBuffer_->desc().size);
    instanceBuffer_->unmap();
    gpuSceneInstanceMapping_.assign(mapping.begin(), mapping.end());
    return {};
}

void MeshletStreamRuntime::cmdDrawMeshTasks(CommandBuffer& commandBuffer) const
{
    if (ready() && drawTaskCount() > 0) {
        commandBuffer.drawMeshTasksIndirect(*drawIndirectBuffer_);
    }
}

uint32_t MeshletStreamRuntime::computeMaxActiveGroups(uint32_t capacity) const
{
    if (capacity == 0) {
        return 0;
    }
    uint64_t total = 0;
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
    for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
        if (instance.visible == 0 || instance.primitiveIndex >= primitives.size()) {
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

Result MeshletStreamRuntime::initializeSceneMetadataBuffers(Device& device, std::string& log)
{
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
    const std::span<const scene::MeshletStreamInstanceInfo> instances = asset_.instances();
    gpuSceneInstanceMapping_.assign(instances.size(), kMeshletStreamInvalidClusterIndex);
    Result result = createAndPopulateHostStorageBuffer<MeshletStreamGpuInstance>(
        device,
        instances.size(),
        instanceBuffer_,
        log,
        "MeshletStreamRuntime instances",
        [this, primitives, instances](MeshletStreamGpuInstance& gpuInstance, size_t index) {
            const scene::MeshletStreamInstanceInfo& instance = instances[index];
            gpuInstance = MeshletStreamGpuInstance{};
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

    result = createAndPopulateHostStorageBuffer<MeshletStreamGpuPrimitive>(
        device,
        primitives.size(),
        primitiveBuffer_,
        log,
        "MeshletStreamRuntime primitives",
        [primitives](MeshletStreamGpuPrimitive& gpuPrimitive, size_t index) {
            const scene::MeshletStreamPrimitiveInfo& primitive = primitives[index];
            gpuPrimitive = MeshletStreamGpuPrimitive{
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
            };
        });
    if (!result) {
        return result;
    }

    const std::span<const scene::MeshletStreamLodLevelInfo> lodLevels = asset_.lodLevels();
    result = createAndPopulateHostStorageBuffer<MeshletStreamGpuLodLevel>(
        device,
        lodLevels.size(),
        lodLevelBuffer_,
        log,
        "MeshletStreamRuntime LOD levels",
        [lodLevels](MeshletStreamGpuLodLevel& gpuLod, size_t index) {
            const scene::MeshletStreamLodLevelInfo& lod = lodLevels[index];
            gpuLod = MeshletStreamGpuLodLevel{
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

    const std::span<const scene::MeshletStreamGroupInfo> groups = asset_.groups();
    result = createAndPopulateHostStorageBuffer<MeshletStreamGpuGroup>(
        device,
        groups.size(),
        groupBuffer_,
        log,
        "MeshletStreamRuntime groups",
        [groups](MeshletStreamGpuGroup& gpuGroup, size_t index) {
            const scene::MeshletStreamGroupInfo& group = groups[index];
            gpuGroup = MeshletStreamGpuGroup{
                .primitiveIndex = group.primitiveIndex,
                .pageIndex = group.pageIndex,
                .lodLevel = group.lodLevel,
                .clusterCount = group.clusterCount,
                .maxQuadricError = group.maxQuadricError,
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
    return createAndPopulateHostStorageBuffer<MeshletStreamGpuNode>(
        device,
        nodes.size(),
        nodeBuffer_,
        log,
        "MeshletStreamRuntime hierarchy nodes",
        [nodes](MeshletStreamGpuNode& gpuNode, size_t index) {
            const scene::MeshletStreamNodeInfo& node = nodes[index];
            gpuNode = MeshletStreamGpuNode{
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
}

Result MeshletStreamRuntime::initializePageTableIfNeeded(CommandBuffer& commandBuffer)
{
    if (pageTableInitialized_) {
        return {};
    }

    Result result = updatePass_->initializePageTable(
        commandBuffer,
        *bindlessHeap_,
        userPush(),
        asset_.pageCount(),
        *pageTableBuffer_,
        pageTableState_);
    if (!result) {
        return result;
    }
    pageTableInitialized_ = true;
    return {};
}

Result MeshletStreamRuntime::applyPageTablePatches(CommandBuffer& commandBuffer)
{
    const std::span<const StreamPageTablePatch> patches = residency_.pendingPatches();
    Result result = updatePass_->apply(
        commandBuffer,
        *bindlessHeap_,
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

Result MeshletStreamRuntime::clearRequestBuffer(CommandBuffer& commandBuffer)
{
    const StreamRequestBufferHeader clearHeader{
        .maxLoadRequests = maxGpuPageRequests_,
        .maxUnloadRequests = maxGpuPageUnloadRequests_,
        .frameIndex = frameIndex_,
    };
    Result result = updateHostBuffer(*requestClearBuffer_, &clearHeader, sizeof(clearHeader));
    if (!result) {
        return result;
    }

    transitionBuffer(commandBuffer, *requestBuffer_, requestBufferState_, ResourceState::TransferDestination);
    commandBuffer.copyBuffer(BufferCopyDesc{
        .source = requestClearBuffer_.get(),
        .destination = requestBuffer_.get(),
        .sourceOffset = 0,
        .destinationOffset = 0,
        .size = sizeof(StreamRequestBufferHeader),
    });
    transitionBuffer(commandBuffer, *requestBuffer_, requestBufferState_, ResourceState::General);
    return {};
}

Result MeshletStreamRuntime::copyRequestBufferForReadback(CommandBuffer& commandBuffer)
{
    transitionBuffer(commandBuffer, *requestBuffer_, requestBufferState_, ResourceState::TransferSource);
    commandBuffer.copyBuffer(BufferCopyDesc{
        .source = requestBuffer_.get(),
        .destination = requestReadbackBuffer_.get(),
        .sourceOffset = 0,
        .destinationOffset = 0,
        .size = requestBuffer_->desc().size,
    });
    requestReadbackValid_ = true;
    return {};
}

Result MeshletStreamRuntime::updateParamsBuffer(const MeshletStreamFrameDesc& frame)
{
    MeshletStreamGpuParams params;
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
    params.upProjection[3] = 0.0f;
    params.viewport[0] = aspect;
    params.viewport[1] = static_cast<float>(width);
    params.viewport[2] = static_cast<float>(height);
    params.viewport[3] = finiteOr(frame.camera.fovDegrees, 60.0f) * 0.017453292519943295f;
    params.clipOrtho[0] = finiteOr(frame.camera.znear, 0.1f);
    params.clipOrtho[1] = finiteOr(frame.camera.zfar, 1000.0f);
    params.clipOrtho[2] = std::max(drawBounds_.radius(), 1.0f) * 2.0f;
    params.clipOrtho[3] = kDefaultReversedZ ? 1.0f : 0.0f;
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
        ? kMeshletStreamNoDebugLodOverride
        : frame.selectedLodLevel;
    params.enableGpuLodSelection = frame.enableGpuLodSelection ? 1u : 0u;
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
    const MeshletStreamGpuParams& previous = previousFrameParamsValid_
        ? previousFrameParams_
        : params;
    std::copy_n(previous.eye, 4u, params.previousEye);
    std::copy_n(previous.center, 4u, params.previousCenter);
    std::copy_n(previous.upProjection, 4u, params.previousUpProjection);
    std::copy_n(previous.viewport, 4u, params.previousViewport);
    std::copy_n(previous.clipOrtho, 4u, params.previousClipOrtho);
    Result result = updateHostBuffer(*paramsBuffer_, &params, sizeof(params));
    if (result) {
        previousFrameParams_ = params;
        previousFrameParamsValid_ = true;
    }
    return result;
}

Result MeshletStreamRuntime::dispatchTraversal(
    CommandBuffer& commandBuffer,
    uint32_t threadCount,
    uint32_t traversalPhase)
{
    if (traversalPass_ == nullptr || bindlessHeap_ == nullptr || !traversalPass_->ready()) {
        return makeError(Error::InvalidArgument);
    }
    MeshletStreamUserPush push = userPush();
    push.traversalPhase = traversalPhase;
    push.activeBuildPhase = threadCount;
    return traversalPass_->dispatch(
        commandBuffer,
        *bindlessHeap_,
        push,
        threadCount,
        *pageTableBuffer_,
        pageTableState_,
        *requestBuffer_,
        requestBufferState_);
}

Result MeshletStreamRuntime::buildActiveTable(CommandBuffer& commandBuffer)
{
    if (activeBuildPass_ == nullptr || bindlessHeap_ == nullptr || !activeBuildPass_->ready()) {
        return makeError(Error::InvalidArgument);
    }

    MeshletStreamUserPush push = userPush();
    push.activeBuildPhase = kMeshletStreamActiveBuildResetPhase;
    Result result = activeBuildPass_->dispatch(
        commandBuffer,
        *bindlessHeap_,
        push,
        1,
        *activeGroupBuffer_,
        activeGroupBufferState_,
        *activeHeaderBuffer_,
        activeHeaderBufferState_,
        *pageTableBuffer_,
        pageTableState_,
        *requestBuffer_,
        requestBufferState_,
        *drawIndirectBuffer_,
        drawIndirectBufferState_,
        *traversalHeaderBuffer_,
        traversalHeaderBufferState_,
        *traversalWorkBuffer_,
        traversalWorkBufferState_);
    if (!result) {
        return result;
    }

    push.activeBuildPhase = kMeshletStreamActiveBuildSeedPhase;
    result = activeBuildPass_->dispatch(
        commandBuffer,
        *bindlessHeap_,
        push,
        traversalWorkerCount_,
        *activeGroupBuffer_,
        activeGroupBufferState_,
        *activeHeaderBuffer_,
        activeHeaderBufferState_,
        *pageTableBuffer_,
        pageTableState_,
        *requestBuffer_,
        requestBufferState_,
        *drawIndirectBuffer_,
        drawIndirectBufferState_,
        *traversalHeaderBuffer_,
        traversalHeaderBufferState_,
        *traversalWorkBuffer_,
        traversalWorkBufferState_);
    if (!result) {
        return result;
    }

    push.activeBuildPhase = kMeshletStreamActiveBuildRunPhase;
    result = activeBuildPass_->dispatch(
        commandBuffer,
        *bindlessHeap_,
        push,
        traversalWorkerCount_,
        *activeGroupBuffer_,
        activeGroupBufferState_,
        *activeHeaderBuffer_,
        activeHeaderBufferState_,
        *pageTableBuffer_,
        pageTableState_,
        *requestBuffer_,
        requestBufferState_,
        *drawIndirectBuffer_,
        drawIndirectBufferState_,
        *traversalHeaderBuffer_,
        traversalHeaderBufferState_,
        *traversalWorkBuffer_,
        traversalWorkBufferState_);
    if (!result) {
        return result;
    }

    push.activeBuildPhase = kMeshletStreamActiveBuildFinalizePhase;
    return activeBuildPass_->dispatch(
        commandBuffer,
        *bindlessHeap_,
        push,
        1,
        *activeGroupBuffer_,
        activeGroupBufferState_,
        *activeHeaderBuffer_,
        activeHeaderBufferState_,
        *pageTableBuffer_,
        pageTableState_,
        *requestBuffer_,
        requestBufferState_,
        *drawIndirectBuffer_,
        drawIndirectBufferState_,
        *traversalHeaderBuffer_,
        traversalHeaderBufferState_,
        *traversalWorkBuffer_,
        traversalWorkBufferState_);
}

Result MeshletStreamRuntime::buildBlasInputs(CommandBuffer& commandBuffer)
{
    if (blasInputPass_ == nullptr ||
        !blasInputPass_->ready() ||
        bindlessHeap_ == nullptr ||
        clasPool_ == nullptr ||
        blasHeaderBuffer_ == nullptr ||
        instanceBlasBuffer_ == nullptr ||
        blasBuildInfoBuffer_ == nullptr ||
        blasClusterReferenceBuffer_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    commandBuffer.hostWriteBarrier();

    auto dispatchPhase = [this, &commandBuffer](uint32_t phase, uint32_t threadCount) {
        MeshletStreamUserPush push = userPush();
        push.activeBuildPhase = phase;
        return blasInputPass_->dispatch(
            commandBuffer,
            *bindlessHeap_,
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

    Result result = dispatchPhase(
        kMeshletStreamBlasInputResetPhase,
        std::max(asset_.instanceCount(), 1u));
    if (!result) {
        return result;
    }
    result = dispatchPhase(kMeshletStreamBlasInputCountPhase, maxActiveGroups_);
    if (!result) {
        return result;
    }
    result = dispatchPhase(
        kMeshletStreamBlasInputSetupPhase,
        std::max(asset_.instanceCount(), 1u));
    if (!result) {
        return result;
    }
    return dispatchPhase(kMeshletStreamBlasInputInsertPhase, maxActiveGroups_);
}

Result MeshletStreamRuntime::cmdBuildBlas(CommandBuffer& commandBuffer)
{
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
            .destinationMode = ClusterAccelerationStructureDestinationMode::Implicit,
            .maxClusterCountPerAccelerationStructure = maxBlasClustersPerBuild_,
            .maxTotalClusterCount = blasClusterReferenceCapacity_,
            .maxAccelerationStructureCount = blasBuildCapacity_,
            .buildInfoBuffer = blasBuildInfoBuffer_.get(),
            .buildInfoStride = sizeof(MeshletStreamGpuBlasBuildInfo),
            .buildInfoSize = blasBuildInfoBuffer_->desc().size,
            .buildInfoCountBuffer = blasHeaderBuffer_.get(),
            .buildInfoCountBufferOffset = offsetof(
                MeshletStreamGpuBlasHeader,
                blasBuildCount),
            .destinationStorageBuffer = blasStorageBuffer_.get(),
            .destinationAddressBuffer = blasAddressBuffer_.get(),
            .destinationAddressSize = blasAddressBuffer_->desc().size,
            .destinationSizeBuffer = blasSizeBuffer_.get(),
            .destinationSizeSize = blasSizeBuffer_->desc().size,
            .scratchBuffer = blasScratchBuffer_.get(),
        });
}

Result MeshletStreamRuntime::cmdBuildFallbackBlas(CommandBuffer& commandBuffer)
{
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
        const FallbackBlasPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
        if (fallback.built) {
            continue;
        }
        const uint32_t primitiveIndex = fallback.primitiveIndex;
        const scene::MeshletStreamPrimitiveInfo& primitive = asset_.primitives()[primitiveIndex];
        bool ready = true;
        for (uint32_t localPage = 0; localPage < primitive.fallbackPageCount; ++localPage) {
            if (!clasPool_->pageHasClas(primitive.fallbackPageOffset + localPage)) {
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
        const FallbackBlasPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
        const uint32_t primitiveIndex = fallback.primitiveIndex;
        const scene::MeshletStreamPrimitiveInfo& primitive = asset_.primitives()[primitiveIndex];
        const uint64_t referenceOffset = fallback.referenceOffset;
        uint64_t writeOffset = referenceOffset;
        for (uint32_t localPage = 0; localPage < primitive.fallbackPageCount; ++localPage) {
            const uint32_t pageIndex = primitive.fallbackPageOffset + localPage;
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
        fallbackBlasReferenceBuffer_->flush(
            referenceOffset * sizeof(uint64_t),
            static_cast<uint64_t>(fallback.referenceCount) * sizeof(uint64_t));
        fallbackBlasAddressBuffer_->flush(
            static_cast<uint64_t>(primitiveIndex) * sizeof(uint64_t),
            sizeof(uint64_t));
    }
    fallbackBlasReferenceBuffer_->unmap();
    fallbackBlasAddressBuffer_->unmap();

    if (fallbackBlasBuildInfoBuffer_->deviceAddress() == 0 ||
        fallbackBlasDestinationBuffer_->deviceAddress() == 0 ||
        fallbackStorageAddress == 0 ||
        fallbackBlasScratchBuffer_->deviceAddress() == 0) {
        return makeError(Error::InvalidArgument);
    }

    for (uint32_t fallbackIndex : readyFallbackIndices) {
        FallbackBlasPrimitive& fallback = fallbackBlasPrimitives_[fallbackIndex];
        const uint32_t clusterCount = fallback.referenceCount;
        const Result result = commandBuffer.buildClusterAccelerationStructureBottomLevels(
            ClusterAccelerationStructureBottomLevelBuildDesc{
                .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                .destinationMode = ClusterAccelerationStructureDestinationMode::Explicit,
                .maxClusterCountPerAccelerationStructure = clusterCount,
                .maxTotalClusterCount = clusterCount,
                .maxAccelerationStructureCount = 1,
                .buildInfoBuffer = fallbackBlasBuildInfoBuffer_.get(),
                .buildInfoBufferOffset =
                    static_cast<uint64_t>(fallbackIndex) *
                    sizeof(MeshletStreamGpuBlasBuildInfo),
                .buildInfoStride = sizeof(MeshletStreamGpuBlasBuildInfo),
                .buildInfoSize = sizeof(MeshletStreamGpuBlasBuildInfo),
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
    }
    return {};
}

Result MeshletStreamRuntime::buildTlasInstances(CommandBuffer& commandBuffer)
{
    if (tlasInputPass_ == nullptr ||
        !tlasInputPass_->ready() ||
        bindlessHeap_ == nullptr ||
        instanceBlasBuffer_ == nullptr ||
        tlasInstanceBuffer_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    commandBuffer.hostWriteBarrier();
    return tlasInputPass_->dispatch(
        commandBuffer,
        *bindlessHeap_,
        userPush(),
        asset_.instanceCount(),
        *instanceBlasBuffer_,
        instanceBlasBufferState_,
        *tlasInstanceBuffer_,
        tlasInstanceBufferState_);
}

Result MeshletStreamRuntime::cmdBuildTlas(CommandBuffer& commandBuffer)
{
    if (tlas_ == nullptr ||
        !tlas_->valid() ||
        tlasScratchBuffer_ == nullptr ||
        tlasInstanceBuffer_ == nullptr ||
        asset_.instanceCount() == 0) {
        return makeError(Error::InvalidArgument);
    }
    const Result result = commandBuffer.buildRayTracingAccelerationStructure(
        RayTracingAccelerationStructureBuildDesc{
            .destination = tlas_.get(),
            .mode = RayTracingAccelerationStructureBuildMode::Build,
            .instanceBuffer = tlasInstanceBuffer_.get(),
            .instanceCount = asset_.instanceCount(),
            .scratchBuffer = tlasScratchBuffer_.get(),
        });
    if (!result) {
        return result;
    }
    tlasBuilt_ = true;
    return {};
}

Result MeshletStreamRuntime::transitionPageBufferForTraversal(CommandBuffer& commandBuffer)
{
    if (drawTaskCount() == 0) {
        return {};
    }
    transitionBuffer(commandBuffer, *pageBuffer_, pageBufferState_, ResourceState::ShaderRead);
    return {};
}

void MeshletStreamRuntime::consumeGpuRequestReadback()
{
    if (!requestReadbackValid_ || requestReadbackBuffer_ == nullptr) {
        return;
    }

    requestReadbackBuffer_->invalidate();
    const void* mapped = requestReadbackBuffer_->map();
    if (mapped == nullptr) {
        requestReadbackValid_ = false;
        return;
    }

    const auto* header = static_cast<const StreamRequestBufferHeader*>(mapped);
    const uint32_t loadCapacity = std::min(header->maxLoadRequests, maxGpuPageRequests_);
    const uint32_t unloadCapacity = std::min(header->maxUnloadRequests, maxGpuPageUnloadRequests_);
    const uint32_t loadCount = std::min(header->loadCounter, loadCapacity);
    const uint32_t unloadCount = std::min(header->unloadCounter, unloadCapacity);
    const auto* pageIds = reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(mapped) + sizeof(StreamRequestBufferHeader));
    const uint32_t* loadPageIds = pageIds;
    const uint32_t* unloadPageIds = pageIds + maxGpuPageRequests_;
    if (header->loadCounter > 0 || header->unloadCounter > 0 ||
        header->loadOverflowCounter > 0 || header->unloadOverflowCounter > 0 ||
        header->invalidPageCounter > 0) {
        (void)residency_.consumeGpuRequests(StreamGpuRequestBatch{
            .loadPageIds = std::span<const uint32_t>(loadPageIds, loadCount),
            .unloadPageIds = std::span<const uint32_t>(unloadPageIds, unloadCount),
            .loadRequestCounter = header->loadCounter,
            .unloadRequestCounter = header->unloadCounter,
            .loadOverflowCounter = header->loadOverflowCounter,
            .unloadOverflowCounter = header->unloadOverflowCounter,
            .invalidPageCounter = header->invalidPageCounter,
            .frameIndex = header->frameIndex,
        });
    }

    requestReadbackBuffer_->unmap();
    requestReadbackValid_ = false;
}

} // namespace metallic::render
