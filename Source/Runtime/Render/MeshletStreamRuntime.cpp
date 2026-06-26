#include "Runtime/Render/MeshletStreamRuntime.h"

#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>

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
            .profileName = "glsl_460",
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

    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
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
            updatePipeline_ != nullptr;
    }

    BindlessHandle updateHandle() const { return updateHandle_; }

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
            if (patch.state == static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded)) {
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
            if (patch.state == static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded)) {
                patchData[writeIndex++] = patch;
            }
        }
        for (const StreamPageTablePatch& patch : patches) {
            if (patch.state != static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded)) {
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
    std::unique_ptr<ShaderModule> updateShader_;
    std::unique_ptr<ComputePipeline> updatePipeline_;
    BindlessHandle updateHandle_;
};

MeshletStreamRuntime::MeshletStreamRuntime() = default;
MeshletStreamRuntime::~MeshletStreamRuntime() = default;

Result MeshletStreamRuntime::initialize(Device& device, const MeshletStreamRuntimeDesc& desc, std::string& log)
{
    reset();
    log.clear();

    std::string reason;
    scene::MeshletStreamAsset openedAsset;
    if (!openedAsset.open(desc.streamAssetPath, reason) || !openedAsset.isCurrentForSource(desc.sourcePath)) {
        if (!desc.autoBuildStreamAsset) {
            log = "MeshletStreamRuntime failed to open current streamasset: " + reason;
            return makeError(Error::Failure);
        }

        scene::Scene loadedScene;
        if (!loadedScene.load(desc.sourcePath)) {
            log = "MeshletStreamRuntime failed to load source scene: " + loadedScene.lastLoadResult().error;
            return makeError(Error::Failure);
        }
        if (!scene::buildMeshletStreamAsset(
                scene::MeshletStreamAssetBuildDesc{
                    .scene = &loadedScene,
                    .sourcePath = desc.sourcePath,
                    .outputPath = desc.streamAssetPath,
                },
                reason)) {
            log = "MeshletStreamRuntime failed to build streamasset: " + reason;
            return makeError(Error::Failure);
        }
        if (!openedAsset.open(desc.streamAssetPath, reason)) {
            log = "MeshletStreamRuntime failed to open built streamasset: " + reason;
            return makeError(Error::Failure);
        }
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
    if (maxResidentPages_ == 0) {
        log = "MeshletStreamRuntime requires maxResidentPages to be greater than zero";
        return makeError(Error::Failure);
    }
    if (pageStride == 0 ||
        pageStride > std::numeric_limits<uint64_t>::max() / maxResidentPages_) {
        log = "MeshletStreamRuntime resident page buffer size overflowed";
        return makeError(Error::Failure);
    }

    Result result = device.createBuffer(
        BufferDesc{
            .size = pageStride * maxResidentPages_,
            .structureStride = 0,
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::Device,
        },
        pageBuffer_);
    if (!result || pageBuffer_ == nullptr) {
        log += resultMessage("createBuffer(MeshletStreamRuntime pages)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    pageBufferState_ = ResourceState::Undefined;

    std::vector<uint32_t> fallbackPages;
    for (const scene::MeshletStreamPrimitiveInfo& primitive : asset_.primitives()) {
        for (uint32_t page = 0; page < primitive.fallbackPageCount; ++page) {
            fallbackPages.push_back(primitive.fallbackPageOffset + page);
        }
    }

    if (!residency_.initialize(
            MeshletStreamResidencyDesc{
                .asset = &asset_,
                .maxResidentPages = maxResidentPages_,
                .queuedFrameCount = std::max(desc.queuedFrameCount, 1u),
                .pageStride = pageStride,
            },
            reason) ||
        !residency_.lockFallbackPages(fallbackPages, reason)) {
        log = "MeshletStreamRuntime residency initialization failed: " + reason;
        return makeError(Error::Failure);
    }

    maxActiveGroups_ = computeMaxActiveGroups();
    maxActiveGroupClusters_ = computeMaxPageClusters();
    if (maxActiveGroups_ == 0 || maxActiveGroupClusters_ == 0) {
        log = "MeshletStreamRuntime streamasset has no drawable active groups";
        return makeError(Error::Failure);
    }
    if (static_cast<uint64_t>(maxActiveGroups_) * maxActiveGroupClusters_ >
        std::numeric_limits<uint32_t>::max()) {
        log = "MeshletStreamRuntime active group draw task count overflowed";
        return makeError(Error::Failure);
    }

    const uint64_t maxUpdatePatches64 = std::max<uint64_t>(
        static_cast<uint64_t>(asset_.pageCount()) * 2ull,
        1ull);
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

    result = createHostStorageBuffer(
        device,
        static_cast<uint64_t>(maxActiveGroups_) * sizeof(MeshletStreamGpuActiveGroup),
        activeGroupBuffer_,
        log,
        "MeshletStreamRuntime active groups");
    if (!result) {
        return result;
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
            .size = pageTableByteSize,
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
        },
        pageTableUploadBuffer_,
        log,
        "MeshletStreamRuntime page table upload");
    if (!result) {
        return result;
    }
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

    result = device.createBindlessHeap(
        BindlessHeapDesc{
            .maxSamplers = 0,
            .maxSampledImages = 0,
            .maxBuffers = 6,
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
    result = allocateAndWriteBuffer(*bindlessHeap_, *pageTableBuffer_, pageTableHandle_, log, "meshlet stream page table");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *paramsBuffer_, paramsHandle_, log, "meshlet stream params");
    if (!result) {
        return result;
    }
    result = allocateAndWriteBuffer(*bindlessHeap_, *requestBuffer_, requestHandle_, log, "meshlet stream request");
    if (!result) {
        return result;
    }

    updatePass_ = std::make_unique<UpdatePass>();
    result = updatePass_->initialize(device, *bindlessHeap_, updateByteSize, log);
    if (!result) {
        return result;
    }

    return {};
}

void MeshletStreamRuntime::reset()
{
    asset_.close();
    residency_.reset();
    drawBounds_.reset();
    activeGroups_.clear();
    pageTable_.clear();
    pageBuffer_.reset();
    activeGroupBuffer_.reset();
    pageTableBuffer_.reset();
    pageTableUploadBuffer_.reset();
    requestBuffer_.reset();
    requestReadbackBuffer_.reset();
    requestClearBuffer_.reset();
    paramsBuffer_.reset();
    bindlessHeap_.reset();
    updatePass_.reset();
    pageHandle_ = {};
    activeGroupHandle_ = {};
    pageTableHandle_ = {};
    paramsHandle_ = {};
    requestHandle_ = {};
    pageBufferState_ = ResourceState::Undefined;
    pageTableState_ = ResourceState::Undefined;
    requestBufferState_ = ResourceState::Undefined;
    pageTableInitialized_ = false;
    requestReadbackValid_ = false;
    frameIndex_ = 0;
    maxResidentPages_ = 0;
    maxPageUploadsPerFrame_ = 0;
    maxGpuPageRequests_ = 0;
    maxGpuPageUnloadRequests_ = 0;
    maxUpdatePatches_ = 0;
    maxActiveGroups_ = 0;
    maxActiveGroupClusters_ = 0;
    currentFrameUploadCount_ = 0;
}

bool MeshletStreamRuntime::ready() const
{
    return asset_.valid() &&
        bindlessHeap_ != nullptr &&
        updatePass_ != nullptr &&
        updatePass_->ready() &&
        pageBuffer_ != nullptr &&
        activeGroupBuffer_ != nullptr &&
        pageTableBuffer_ != nullptr &&
        pageTableUploadBuffer_ != nullptr &&
        requestBuffer_ != nullptr &&
        requestReadbackBuffer_ != nullptr &&
        requestClearBuffer_ != nullptr &&
        paramsBuffer_ != nullptr;
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
    consumeGpuRequestReadback();
    currentFrameUploadCount_ = residency_.processUploads(streamer, *pageBuffer_, maxPageUploadsPerFrame_);
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

    buildFrameActiveGroups(frame.selectedLodLevel);
    result = updateActiveGroupBuffer();
    if (!result) {
        return result;
    }
    result = updateParamsBuffer(frame);
    if (!result) {
        return result;
    }
    return transitionPageBufferForTraversal(commandBuffer);
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
        .updateBuffer = updatePass_ != nullptr ? updatePass_->updateHandle().index : 0u,
    };
}

uint32_t MeshletStreamRuntime::drawTaskCount() const
{
    if (activeGroups_.empty() || maxActiveGroupClusters_ == 0) {
        return 0;
    }
    const uint64_t count = static_cast<uint64_t>(activeGroups_.size()) * maxActiveGroupClusters_;
    return count > std::numeric_limits<uint32_t>::max() ? 0u : static_cast<uint32_t>(count);
}

uint32_t MeshletStreamRuntime::computeMaxActiveGroups() const
{
    uint64_t total = 0;
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
    const std::span<const scene::MeshletStreamLodLevelInfo> lodLevels = asset_.lodLevels();
    for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
        if (instance.visible == 0 || instance.primitiveIndex >= primitives.size()) {
            continue;
        }
        const scene::MeshletStreamPrimitiveInfo& primitive = primitives[instance.primitiveIndex];
        uint32_t maxGroups = primitive.fallbackPageCount;
        for (uint32_t lod = 0; lod < primitive.lodLevelCount; ++lod) {
            const scene::MeshletStreamLodLevelInfo& lodInfo = lodLevels[primitive.lodLevelOffset + lod];
            maxGroups = std::max(maxGroups, lodInfo.pageCount);
            maxGroups = std::max(maxGroups, lodInfo.pageCount + primitive.fallbackPageCount);
        }
        total += maxGroups;
        if (total > std::numeric_limits<uint32_t>::max()) {
            return 0;
        }
    }
    return static_cast<uint32_t>(total);
}

uint32_t MeshletStreamRuntime::computeMaxPageClusters() const
{
    uint32_t maxClusters = 0;
    for (const scene::MeshletStreamPageInfo& page : asset_.pages()) {
        maxClusters = std::max(maxClusters, page.clusterCount);
    }
    return maxClusters;
}

void MeshletStreamRuntime::appendResidentPageGroup(
    const scene::MeshletStreamInstanceInfo& instance,
    const scene::MeshletStreamPageInfo& page,
    uint32_t pageIndex)
{
    const uint32_t slot = residency_.slotForPage(pageIndex);
    if (slot == UINT32_MAX || activeGroups_.size() >= maxActiveGroups_) {
        return;
    }

    MeshletStreamGpuActiveGroup group;
    group.pageSlot = slot;
    group.pageIndex = pageIndex;
    group.clusterCount = page.clusterCount;
    group.primitiveIndex = page.primitiveIndex;
    group.lodLevel = page.lodLevel;
    group.materialIndex = instance.materialIndex;
    group.colorSeed = pageIndex * 131u;
    group.flags = kMeshletStreamActiveGroupResident;
    for (uint32_t row = 0; row < 4; ++row) {
        group.world0[row] = instance.worldMatrix[0 + row];
        group.world1[row] = instance.worldMatrix[4 + row];
        group.world2[row] = instance.worldMatrix[8 + row];
        group.world3[row] = instance.worldMatrix[12 + row];
    }
    activeGroups_.push_back(group);
}

void MeshletStreamRuntime::appendLoadRequestGroup(
    const scene::MeshletStreamInstanceInfo& instance,
    const scene::MeshletStreamPageInfo& page,
    uint32_t pageIndex)
{
    if (activeGroups_.size() >= maxActiveGroups_) {
        return;
    }

    MeshletStreamGpuActiveGroup group;
    group.pageSlot = UINT32_MAX;
    group.pageIndex = pageIndex;
    group.clusterCount = 0;
    group.primitiveIndex = page.primitiveIndex;
    group.lodLevel = page.lodLevel;
    group.materialIndex = instance.materialIndex;
    group.colorSeed = pageIndex * 131u;
    group.flags = kMeshletStreamActiveGroupLoadRequest;
    for (uint32_t row = 0; row < 4; ++row) {
        group.world0[row] = instance.worldMatrix[0 + row];
        group.world1[row] = instance.worldMatrix[4 + row];
        group.world2[row] = instance.worldMatrix[8 + row];
        group.world3[row] = instance.worldMatrix[12 + row];
    }
    activeGroups_.push_back(group);
}

bool MeshletStreamRuntime::appendResidentPageRange(
    const scene::MeshletStreamInstanceInfo& instance,
    uint32_t pageOffset,
    uint32_t pageCount)
{
    const std::span<const scene::MeshletStreamPageInfo> pages = asset_.pages();
    bool allResident = true;
    for (uint32_t page = 0; page < pageCount; ++page) {
        const uint32_t pageIndex = pageOffset + page;
        if (!residency_.pageResident(pageIndex)) {
            allResident = false;
        }
    }
    if (!allResident) {
        return false;
    }
    for (uint32_t page = 0; page < pageCount; ++page) {
        const uint32_t pageIndex = pageOffset + page;
        appendResidentPageGroup(instance, pages[pageIndex], pageIndex);
    }
    return true;
}

void MeshletStreamRuntime::appendRequestPageRange(
    const scene::MeshletStreamInstanceInfo& instance,
    uint32_t pageOffset,
    uint32_t pageCount)
{
    const std::span<const scene::MeshletStreamPageInfo> pages = asset_.pages();
    for (uint32_t page = 0; page < pageCount; ++page) {
        const uint32_t pageIndex = pageOffset + page;
        if (!residency_.pageResident(pageIndex)) {
            appendLoadRequestGroup(instance, pages[pageIndex], pageIndex);
        }
    }
}

void MeshletStreamRuntime::buildFrameActiveGroups(uint32_t selectedLodLevel)
{
    activeGroups_.clear();
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
    const std::span<const scene::MeshletStreamLodLevelInfo> lodLevels = asset_.lodLevels();

    for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
        if (instance.visible == 0 || instance.primitiveIndex >= primitives.size()) {
            continue;
        }
        const scene::MeshletStreamPrimitiveInfo& primitive = primitives[instance.primitiveIndex];
        if (primitive.lodLevelCount == 0 || primitive.fallbackPageCount == 0) {
            continue;
        }
        const uint32_t localLod = std::min(selectedLodLevel, primitive.lodLevelCount - 1u);
        const scene::MeshletStreamLodLevelInfo& lodInfo = lodLevels[primitive.lodLevelOffset + localLod];
        if (!appendResidentPageRange(instance, lodInfo.pageOffset, lodInfo.pageCount)) {
            appendRequestPageRange(instance, lodInfo.pageOffset, lodInfo.pageCount);
            appendResidentPageRange(instance, primitive.fallbackPageOffset, primitive.fallbackPageCount);
        }
    }
}

Result MeshletStreamRuntime::initializePageTableIfNeeded(CommandBuffer& commandBuffer)
{
    if (pageTableInitialized_) {
        return {};
    }

    pageTable_.resize(asset_.pageCount());
    residency_.buildInitialPageTable(pageTable_);
    const uint64_t byteSize = static_cast<uint64_t>(pageTable_.size()) * sizeof(StreamPageTableEntry);
    Result result = updateHostBuffer(*pageTableUploadBuffer_, pageTable_.data(), byteSize);
    if (!result) {
        return result;
    }

    transitionBuffer(commandBuffer, *pageTableBuffer_, pageTableState_, ResourceState::TransferDestination);
    commandBuffer.copyBuffer(BufferCopyDesc{
        .source = pageTableUploadBuffer_.get(),
        .destination = pageTableBuffer_.get(),
        .sourceOffset = 0,
        .destinationOffset = 0,
        .size = byteSize,
    });
    transitionBuffer(commandBuffer, *pageTableBuffer_, pageTableState_, ResourceState::General);
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

Result MeshletStreamRuntime::updateActiveGroupBuffer()
{
    return updateHostBuffer(
        *activeGroupBuffer_,
        activeGroups_.data(),
        static_cast<uint64_t>(activeGroups_.size() * sizeof(MeshletStreamGpuActiveGroup)));
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
    params.pageStrideWords = static_cast<uint32_t>(residency_.pageStride() / sizeof(uint32_t));
    params.drawTaskCount = drawTaskCount();
    params.frameIndex = frameIndex_ == 0 ? 1u : frameIndex_;
    params.maxGpuPageRequests = maxGpuPageRequests_;
    params.maxGpuPageUnloadRequests = maxGpuPageUnloadRequests_;
    params.activeGroupCount = static_cast<uint32_t>(activeGroups_.size());
    params.maxActiveGroupClusters = maxActiveGroupClusters_;
    return updateHostBuffer(*paramsBuffer_, &params, sizeof(params));
}

Result MeshletStreamRuntime::transitionPageBufferForTraversal(CommandBuffer& commandBuffer)
{
    if (activeGroups_.empty()) {
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
