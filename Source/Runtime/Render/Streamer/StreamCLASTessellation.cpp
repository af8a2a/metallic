#include "Runtime/Render/Streamer/StreamCLASTessellation.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Core/ComputeResourceEncoder.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GPUDrivenTessellation.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"
#include "Runtime/Debug/DebugHash.h"
#include "Shaders/Interop/StreamCLASTessellation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

namespace metallic::render {
namespace {

constexpr auto kDataUsage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
    BufferUsageBits::TransferSource | BufferUsageBits::AccelerationStructureBuildInput;
constexpr auto kASUsage = kDataUsage | BufferUsageBits::AccelerationStructureStorage;
constexpr uint32_t kTemplateCount = 8u * 8u * 8u;

Result<> synchronizeTessellation(CommandBuffer& commands)
{
    const MemoryBarrierDesc barrier{
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
    return commands.synchronize({.memory = {&barrier, 1}});
}

Result<> upload(Buffer& buffer, const void* data, size_t bytes)
{
    void* mapped = buffer.map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, data, bytes);
    buffer.flush({0, bytes});
    buffer.unmap();
    return {};
}

} // namespace

struct StreamCLASTessellation::Impl {
    struct Materials {
        std::unique_ptr<Buffer> data;
        ResourceLease dataLease;
        std::vector<ResourceLease> textures;
        std::shared_ptr<const ComputeSampledImageSnapshot> snapshot;
        uint64_t sceneIdentity = 0, materialRevision = 0;
    };
    struct Templates {
        std::unique_ptr<Buffer> vertices, indices, infos, storage, addresses, scratch;
        BufferSlice storageRange;
        ClusterAccelerationStructureTriangleBuildSizesDesc limits;
        bool initialized = false;
    };
    Device* device = nullptr;
    std::shared_ptr<ResourceRegistry> registry;
    std::shared_ptr<Materials> materials;
    std::shared_ptr<Templates> templates;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::unique_ptr<Buffer> leaves, vertices, triangles, instanceStates, counts;
    std::shared_ptr<Buffer> history;
    std::unique_ptr<Buffer> clasInfos, clasAddresses, clasStorage, clasScratch, clasSizes;
    std::unique_ptr<Buffer> blasInfos, blasAddresses, blasStorage, blasScratch;
    std::unique_ptr<Buffer> parameters;
    std::shared_ptr<bool> historyInitialized = std::make_shared<bool>(false);
    ClusterAccelerationStructureTriangleBuildSizesDesc clasLimits;
    StreamCLASTessellationBuildMode buildMode = StreamCLASTessellationBuildMode::Triangles;
    StreamCLASTessellationTemplateShape templateShape = StreamCLASTessellationTemplateShape::Right;
    StreamCLASTessellationInstantiationFlags instantiationFlags = StreamCLASTessellationInstantiationFlags::FastTrace;
    uint32_t clasInfoStride = sizeof(ClusterAccelerationStructureTriangleIndirectBuildInfo);
    uint32_t capacity = 0, instanceCount = 0, baseClusterId = 0, materialRevision = 0;
    std::string shaderHash;
    bool enabled = false;
    bool diagnosticBuildSizes = false;

    Result<> buffer(uint64_t bytes, MemoryLocation memory, std::unique_ptr<Buffer>& output,
        BufferUsageBits usage = kDataUsage)
    {
        return device->createBuffer({.size = std::max(bytes, uint64_t(16)), .usage = usage,
            .memoryLocation = memory, .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
            .memoryDomain = MemoryBudgetDomain::CLASScratch})
            .transform([&](auto value) { output = std::move(value); });
    }

    Result<> dispatch(CommandBuffer& commands, uint32_t phase, uint32_t groupCount)
    {
        const StreamCLASTessellationPush push{parameters->deviceAddress(), phase, 0};
        auto result = registry->bind(commands, materials->textures);
        if (result) { result = registry->retain(commands, materials->dataLease); }
        if (result) { result = commands.bindExecution(pipeline->execution(), &push, sizeof(push)); }
        if (result) { result = commands.dispatch(std::min(std::max(groupCount, 1u), 65535u),
            (std::max(groupCount, 1u) + 65534u) / 65535u, 1); }
        if (result) { result = synchronizeTessellation(commands); }
        return result;
    }

    Result<> prepareTemplates(CommandBuffer& commands)
    {
        if (!templates) {
            auto pool = std::make_shared<Templates>();
            pool->limits = {.maxClusterTriangleCount = METALLIC_TESS_TRIANGLES_PER_PATCH,
                .maxClusterVertexCount = METALLIC_TESS_VERTICES_PER_PATCH,
                .maxTotalTriangleCount = kTemplateCount * METALLIC_TESS_TRIANGLES_PER_PATCH,
                .maxTotalVertexCount = kTemplateCount * METALLIC_TESS_VERTICES_PER_PATCH,
                .maxAccelerationStructureCount = kTemplateCount,
                .operation = ClusterAccelerationStructureTriangleOperation::BuildTemplate};
            const auto sizes = device->queryClusterAccelerationStructureTriangleBuildSizes(pool->limits,
                ClusterAccelerationStructureDestinationMode::Implicit);
            const auto properties = device->queryClusterAccelerationStructureProperties();
            if (!sizes || !properties) { return makeError(!sizes ? sizes.error() : properties.error()); }
            Result<> result;
            if (!(result = buffer(uint64_t(kTemplateCount) * METALLIC_TESS_VERTICES_PER_PATCH * 12u,
                    MemoryLocation::HostUpload, pool->vertices)) ||
                !(result = buffer(uint64_t(kTemplateCount) * METALLIC_TESS_TRIANGLES_PER_PATCH * 12u,
                    MemoryLocation::HostUpload, pool->indices)) ||
                !(result = buffer(uint64_t(kTemplateCount) * sizeof(ClusterAccelerationStructureTriangleTemplateIndirectBuildInfo),
                    MemoryLocation::HostUpload, pool->infos)) ||
                !(result = buffer(sizes->accelerationStructureSize + properties->templateStorageAlignment - 1u,
                    MemoryLocation::Device, pool->storage, kASUsage)) ||
                !(result = buffer(uint64_t(kTemplateCount) * 8u, MemoryLocation::Device, pool->addresses, kASUsage)) ||
                !(result = buffer(sizes->buildScratchSize + properties->scratchAlignment,
                    MemoryLocation::Device, pool->scratch, kASUsage))) { return result; }
            const auto alignment = properties->templateStorageAlignment;
            const auto offset = (alignment - pool->storage->deviceAddress() % alignment) % alignment;
            pool->storageRange = *pool->storage->slice({offset, sizes->accelerationStructureSize});
            std::vector<std::array<float, 3>> positions(kTemplateCount * METALLIC_TESS_VERTICES_PER_PATCH);
            std::vector<std::array<uint32_t, 3>> indices(kTemplateCount * METALLIC_TESS_TRIANGLES_PER_PATCH);
            std::vector<ClusterAccelerationStructureTriangleTemplateIndirectBuildInfo> infos(kTemplateCount);
            for (uint32_t index = 0; index < kTemplateCount; ++index) {
                const auto pattern = makeTessellationPattern({(index & 7u) + 1u,
                    ((index >> 3u) & 7u) + 1u, (index >> 6u) + 1u});
                for (size_t vertex = 0; vertex < pattern.vertices.size(); ++vertex) {
                    const float b1 = float(pattern.vertices[vertex][1]) / 65535.0f;
                    const float b2 = float(pattern.vertices[vertex][2]) / 65535.0f;
                    positions[index * METALLIC_TESS_VERTICES_PER_PATCH + vertex] =
                        templateShape == StreamCLASTessellationTemplateShape::Equilateral
                        ? std::array<float, 3>{b1 + 0.5f * b2, std::sqrt(3.0f) * 0.5f * b2, 0.0f}
                        : std::array<float, 3>{b1, b2, 0.0f};
                }
                std::copy(pattern.triangles.begin(), pattern.triangles.end(),
                    indices.begin() + index * METALLIC_TESS_TRIANGLES_PER_PATCH);
                auto& info = infos[index].triangle;
                info.countsAndIndexType = METALLIC_CLAS_PACK_COUNTS(uint32_t(pattern.triangles.size()),
                    uint32_t(pattern.vertices.size()), 0u, METALLIC_CLAS_INDEX_UINT32);
                info.indexAndVertexStrides = METALLIC_CLAS_PACK_STRIDES(4u, 12u);
                info.indexBufferAddress = pool->indices->deviceAddress() + uint64_t(index) * METALLIC_TESS_TRIANGLES_PER_PATCH * 12u;
                info.vertexBufferAddress = pool->vertices->deviceAddress() + uint64_t(index) * METALLIC_TESS_VERTICES_PER_PATCH * 12u;
                // ID/geometry base zero; instantiation supplies the generated ID.
                // An absent bounding limit permits arbitrary source transforms
                // and world-space displacement without a guessed height envelope.
            }
            if (!(result = upload(*pool->vertices, positions.data(), positions.size() * sizeof(positions[0]))) ||
                !(result = upload(*pool->indices, indices.data(), indices.size() * sizeof(indices[0]))) ||
                !(result = upload(*pool->infos, infos.data(), infos.size() * sizeof(infos[0])))) { return result; }
            templates = std::move(pool);
        }
        auto result = commands.retainResource(templates);
        if (!result || templates->initialized) { return result; }
        commands.hostWriteBarrier();
        const std::array geometry{*templates->vertices->slice(), *templates->indices->slice()};
        result = commands.buildClusterAccelerationStructureTrianglesIndirect({.limits = templates->limits,
            .buildInfoBuffer = *templates->infos->slice(),
            .buildInfoStride = sizeof(ClusterAccelerationStructureTriangleTemplateIndirectBuildInfo),
            .destinationStorageBuffer = templates->storageRange,
            .destinationAddressBuffer = *templates->addresses->slice(),
            .scratchBuffer = *templates->scratch->slice(), .geometryBuffers = geometry});
        if (result) { result = synchronizeTessellation(commands); }
        if (result) {
            result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
                [pool = templates] { pool->initialized = true; }, nullptr));
        }
        return result;
    }
};

StreamCLASTessellation::StreamCLASTessellation() : impl_(std::make_unique<Impl>())
{
}
StreamCLASTessellation::~StreamCLASTessellation() = default;

Result<> StreamCLASTessellation::initialize(Device& device, std::shared_ptr<ResourceRegistry> registry,
    const scene::MeshletStreamAsset& asset, const StreamCLASTessellationSettings& settings, std::string& log)
{
    const auto maxPatches = settings.maxPatches;
    const auto buildMode = settings.buildMode;
    if (maxPatches == 0 || maxPatches > 1048576u || asset.instanceCount() == 0 ||
        uint64_t(asset.pageCount()) * METALLIC_STREAM_CLAS_CLUSTER_STRIDE + maxPatches > UINT32_MAX) {
        log = "CLAS tessellation patch capacity or generated cluster namespace is invalid.";
        return makeError(Error::InvalidArgument);
    }
    auto& p = *impl_;
    p.device = &device;
    p.registry = std::move(registry);
    p.capacity = maxPatches;
    p.buildMode = buildMode;
    p.templateShape = buildMode == StreamCLASTessellationBuildMode::Templates
        ? settings.templateShape : StreamCLASTessellationTemplateShape::Right;
    p.instantiationFlags = buildMode == StreamCLASTessellationBuildMode::Templates
        ? settings.instantiationFlags : StreamCLASTessellationInstantiationFlags::FastTrace;
    p.clasInfoStride = buildMode == StreamCLASTessellationBuildMode::Templates
        ? sizeof(ClusterAccelerationStructureTriangleTemplateInstantiationInfo)
        : sizeof(ClusterAccelerationStructureTriangleIndirectBuildInfo);
    p.instanceCount = asset.instanceCount();
    p.baseClusterId = asset.pageCount() * METALLIC_STREAM_CLAS_CLUSTER_STRIDE;
    p.clasLimits = {
        .maxClusterTriangleCount = METALLIC_TESS_TRIANGLES_PER_PATCH,
        .maxClusterVertexCount = METALLIC_TESS_VERTICES_PER_PATCH,
        .maxTotalTriangleCount = maxPatches * METALLIC_TESS_TRIANGLES_PER_PATCH,
        .maxTotalVertexCount = maxPatches * METALLIC_TESS_VERTICES_PER_PATCH,
        .maxAccelerationStructureCount = maxPatches,
        .operation = buildMode == StreamCLASTessellationBuildMode::Templates
            ? ClusterAccelerationStructureTriangleOperation::InstantiateTemplate
            : ClusterAccelerationStructureTriangleOperation::Build,
        .flags = p.instantiationFlags == StreamCLASTessellationInstantiationFlags::None
            ? RayTracingAccelerationStructureBuildFlags::None
            : RayTracingAccelerationStructureBuildFlags::PreferFastTrace};
    auto properties = device.queryClusterAccelerationStructureProperties();
    auto clasSize = device.queryClusterAccelerationStructureTriangleBuildSizes(p.clasLimits,
        ClusterAccelerationStructureDestinationMode::Implicit);
    auto blasSize = device.queryClusterAccelerationStructureBottomLevelBuildSizes({
        .maxClusterCountPerAccelerationStructure = maxPatches,
        .maxTotalClusterCount = maxPatches, .maxAccelerationStructureCount = p.instanceCount});
    if (!properties || !clasSize || !blasSize) {
        log = "Cannot query dynamic CLAS/BLAS tessellation allocation sizes.";
        return makeError(!properties ? properties.error() : !clasSize ? clasSize.error() : blasSize.error());
    }
    Result<> result;
    std::unique_ptr<Buffer> history;
    if (!(result = p.buffer(uint64_t(maxPatches) * sizeof(StreamCLASTessellationLeaf), MemoryLocation::Device, p.leaves)) ||
        !(result = p.buffer(sizeof(StreamCLASTessellationParameters), MemoryLocation::Device, p.parameters,
            kDataUsage | BufferUsageBits::TransferDestination)) ||
        !(result = p.buffer(uint64_t(maxPatches) * METALLIC_TESS_VERTICES_PER_PATCH * sizeof(StreamCLASTessellationVertex), MemoryLocation::Device, p.vertices)) ||
        !(result = p.buffer(uint64_t(maxPatches) * METALLIC_TESS_TRIANGLES_PER_PATCH * 3u * sizeof(uint32_t), MemoryLocation::Device, p.triangles)) ||
        !(result = p.buffer(uint64_t(p.instanceCount) * sizeof(StreamCLASTessellationInstance), MemoryLocation::Device, p.instanceStates)) ||
        !(result = p.buffer(16, MemoryLocation::Device, p.counts)) ||
        !(result = p.buffer(uint64_t(p.instanceCount) * 16u, MemoryLocation::Device, history)) ||
        !(result = p.buffer(uint64_t(maxPatches) * p.clasInfoStride, MemoryLocation::Device, p.clasInfos)) ||
        !(result = p.buffer(uint64_t(maxPatches) * sizeof(uint64_t), MemoryLocation::Device, p.clasAddresses, kASUsage)) ||
        !(result = p.buffer(clasSize->accelerationStructureSize, MemoryLocation::Device, p.clasStorage, kASUsage)) ||
        !(result = p.buffer(clasSize->buildScratchSize + properties->scratchAlignment, MemoryLocation::Device, p.clasScratch, kASUsage)) ||
        !(result = p.buffer(uint64_t(p.instanceCount) * sizeof(ClusterAccelerationStructureBottomLevelBuildInfo), MemoryLocation::Device, p.blasInfos)) ||
        !(result = p.buffer(uint64_t(p.instanceCount) * sizeof(uint64_t), MemoryLocation::Device, p.blasAddresses, kASUsage)) ||
        !(result = p.buffer(blasSize->accelerationStructureSize, MemoryLocation::Device, p.blasStorage, kASUsage)) ||
        !(result = p.buffer(blasSize->buildScratchSize + properties->scratchAlignment, MemoryLocation::Device, p.blasScratch, kASUsage))) {
        log = "Cannot allocate bounded CLAS tessellation resources.";
        return result;
    }
    p.history = std::move(history);
    ShaderCompileResult compiled;
    result = ShaderRegistry::instance().getShader({.moduleName = "Features/GPUDriven/StreamCLASTessellation",
        .entryPointName = "tessellationMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled.diagnostics)
        .transform([&](auto value) { compiled = std::move(value); });
    if (!result) { log = compiled.diagnostics; return result; }
    p.shaderHash = debug::debugSha256(std::span(reinterpret_cast<const uint8_t*>(compiled.spirv.data()),
        compiled.spirv.size() * sizeof(compiled.spirv[0])));
    result = ShaderRegistry::instance().getShaderModule(device, {.spirv = compiled.spirv,
        .debugName = "StreamCLASTessellation.tessellationMain"})
        .transform([&](auto value) { p.shader = std::move(value); });
    if (result) {
        result = ShaderRegistry::instance().getComputePipeline(device,
            {.computeShader = {p.shader.get(), "main"}, .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(StreamCLASTessellationPush)})
            .transform([&](auto value) { p.pipeline = std::move(value); });
    }
    return result;
}

void StreamCLASTessellation::reusePersistentState(const StreamCLASTessellation& previous)
{
    auto& p = *impl_;
    const auto& prior = *previous.impl_;
    assert(p.instanceCount == prior.instanceCount && p.baseClusterId == prior.baseClusterId);
    p.history = prior.history;
    p.historyInitialized = prior.historyInitialized;
    p.materials = prior.materials;
    p.materialRevision = prior.materialRevision;
    if (p.templateShape == prior.templateShape) { p.templates = prior.templates; }
}

Result<> StreamCLASTessellation::updateMaterials(const scene::Scene& scene,
    const ScenePathTraceResources& resources, std::string& log)
{
    auto& p = *impl_;
    auto snapshot = resources.materialTextureSnapshot();
    if (p.materials && p.materials->sceneIdentity == scene.resourceIdentity() &&
        p.materials->materialRevision == scene.materialRevision() &&
        p.materials->snapshot == snapshot) { return {}; }
    auto replacement = std::make_shared<Impl::Materials>();
    replacement->sceneIdentity = scene.resourceIdentity();
    replacement->materialRevision = scene.materialRevision();
    replacement->snapshot = std::move(snapshot);
    const auto indices = resources.logicalTextureIndices();
    const auto& views = resources.materialTextureViews();
    std::vector<uint32_t> descriptors(indices.size(), UINT32_MAX);
    for (const auto& material : scene.materials()) {
        if (material.displacementMagnitude == 0.0f || material.displacementTexture.textureIndex < 0) { continue; }
        const auto logical = uint32_t(material.displacementTexture.textureIndex);
        if (material.displacementTexture.texCoord != 0 || logical >= indices.size() ||
            indices[logical] == 0u || indices[logical] >= views.size() || !views[indices[logical]]) {
            log = "CLAS displacement requires a decoded height texture and TEXCOORD_0.";
            return makeError(Error::InvalidArgument);
        }
        if (descriptors[logical] != UINT32_MAX) { continue; }
        auto lease = p.registry->sampledImage(*views[indices[logical]], TextureLayout::ShaderRead);
        if (!lease) { return makeError(lease.error()); }
        descriptors[logical] = lease->shaderIndex();
        replacement->textures.push_back(std::move(*lease));
    }
    const auto data = buildTessellationData(scene.materials(), descriptors);
    auto result = p.buffer(data.size() * sizeof(uint32_t), MemoryLocation::HostUpload, replacement->data);
    if (result) { result = upload(*replacement->data, data.data(), data.size() * sizeof(uint32_t)); }
    if (result) {
        result = p.registry->storageBuffer(*replacement->data)
            .transform([&](auto value) { replacement->dataLease = std::move(value); });
    }
    if (!result) { return result; }
    p.materials = std::move(replacement);
    ++p.materialRevision;
    return {};
}

Result<> StreamCLASTessellation::record(CommandBuffer& commands, const StreamCLASTessellationInputs& inputs,
    const MeshletStreamFrameDesc& frame, std::string& log,
    const std::function<void(std::string_view)>& checkpoint)
{
    auto& p = *impl_;
    const auto& settings = frame.rtTessellation;
    p.enabled = settings.enabled;
    p.diagnosticBuildSizes = settings.diagnosticBuildSizes;
    assert(p.buildMode == settings.buildMode || !settings.enabled);
    assert(!settings.enabled || p.buildMode != StreamCLASTessellationBuildMode::Templates ||
        (p.templateShape == settings.templateShape && p.instantiationFlags == settings.instantiationFlags));
    if (!p.materials || !std::isfinite(settings.edgePixels) || settings.edgePixels < 1.0f ||
        settings.edgePixels > 256.0f || settings.maxFactor < 1u || settings.maxFactor > 8u || settings.maxSplitDepth > 3u) {
        log = "Invalid CLAS tessellation quality settings or unpublished height data.";
        return makeError(Error::InvalidArgument);
    }
    if (p.diagnosticBuildSizes && !p.clasSizes) {
        log = "Allocate diagnostic CLAS build sizes";
        if (auto result = p.buffer(uint64_t(p.capacity) * sizeof(uint32_t), MemoryLocation::Device,
                p.clasSizes, kASUsage); !result) { return result; }
    }
    if (p.buildMode == StreamCLASTessellationBuildMode::Templates) {
        log = "Prepare persistent CLAS tessellation templates";
        if (auto result = p.prepareTemplates(commands); !result) { return result; }
    }
    StreamCLASTessellationParameters parameters{
        .geometry = {p.leaves->deviceAddress(), p.vertices->deviceAddress(), p.triangles->deviceAddress(), p.baseClusterId, p.capacity},
        .pages = inputs.pages->deviceAddress(), .pageTable = inputs.pageTable->deviceAddress(),
        .instances = inputs.instances->deviceAddress(), .primitives = inputs.primitives->deviceAddress(),
        .groups = inputs.groups->deviceAddress(), .representations = inputs.representations->deviceAddress(),
        .representationGroups = inputs.representationGroups->deviceAddress(),
        .representationState = inputs.representationState->deviceAddress(), .streamParams = inputs.params->deviceAddress(),
        .instanceStates = p.instanceStates->deviceAddress(), .counts = p.counts->deviceAddress(),
        .clasBuildInfos = p.clasInfos->deviceAddress(), .clasAddresses = p.clasAddresses->deviceAddress(),
        .blasBuildInfos = p.blasInfos->deviceAddress(), .blasAddresses = p.blasAddresses->deviceAddress(),
        .tlasInstances = inputs.tlasInstances->deviceAddress(), .blasHeader = inputs.blasHeader->deviceAddress(),
        .history = p.history->deviceAddress(), .materialData = p.materials->dataLease.shaderIndex(),
        .instanceCount = p.instanceCount, .enabled = settings.enabled ? 1u : 0u,
        .initializeHistory = *p.historyInitialized ? 0u : 1u, .edgePixels = settings.edgePixels,
        .maxFactor = settings.maxFactor, .maxSplitDepth = settings.maxSplitDepth, .revision = p.materialRevision,
        .pageDescriptor = inputs.pageDescriptor, .pageTableDescriptor = inputs.pageTableDescriptor,
        .templateAddresses = p.templates ? p.templates->addresses->deviceAddress() : 0,
        .buildMode = uint32_t(p.buildMode)};
    std::unique_ptr<Buffer> uploadBuffer;
    log = "Upload CLAS tessellation parameters";
    auto result = p.buffer(sizeof(parameters), MemoryLocation::HostUpload, uploadBuffer);
    if (result) { result = upload(*uploadBuffer, &parameters, sizeof(parameters)); }
    if (!result) { return result; }
    result = commands.retainResource(uploadBuffer->retainAllocation());
    if (result) { result = commands.retainResource(p.materials); }
    if (!result) { return result; }
    for (Buffer* resource : {p.parameters.get(), p.leaves.get(), p.vertices.get(), p.triangles.get(), p.instanceStates.get(), p.counts.get(),
            p.history.get(), p.clasInfos.get(), p.clasAddresses.get(), p.clasStorage.get(), p.clasScratch.get(),
            p.blasInfos.get(), p.blasAddresses.get(), p.blasStorage.get(), p.blasScratch.get(),
            inputs.pages, inputs.pageTable, inputs.instances, inputs.primitives, inputs.groups, inputs.params,
            inputs.representations, inputs.representationGroups, inputs.representationState, inputs.blasHeader, inputs.tlasInstances}) {
        result = commands.retainResource(resource->retainAllocation());
        if (!result) { return result; }
    }
    commands.hostWriteBarrier();
    log = "Copy CLAS tessellation parameters";
    result = synchronizeTessellation(commands);
    if (result) { result = commands.copyBuffer(*uploadBuffer->slice(), *p.parameters->slice()); }
    if (result) { result = synchronizeTessellation(commands); }
    if (result) { log = "Count CLAS tessellation patches"; result = p.dispatch(commands, 0, p.instanceCount); }
    if (result) { log = "Admit CLAS tessellation instances"; result = p.dispatch(commands, 1, 1); }
    if (result) { log = "Emit CLAS tessellation patches"; result = p.dispatch(commands, 2, p.instanceCount); }
    if (result) { log = "Dice CLAS tessellation patches"; result = p.dispatch(commands, 3, p.capacity); }
    if (!result) { return result; }
    log = "Build tessellated triangle CLAS";
    const std::array geometry{*p.vertices->slice(), *p.triangles->slice()};
    if (checkpoint) { checkpoint("BeforeTessellationCLAS"); }
    result = commands.buildClusterAccelerationStructureTrianglesIndirect({.limits = p.clasLimits,
        .buildInfoBuffer = *p.clasInfos->slice(), .buildInfoStride = p.clasInfoStride,
        .buildInfoCountBuffer = *p.counts->slice({0, 4}),
        .destinationStorageBuffer = *p.clasStorage->slice(), .destinationAddressBuffer = *p.clasAddresses->slice(),
        .destinationSizeBuffer = p.diagnosticBuildSizes ? *p.clasSizes->slice() : BufferSlice{},
        .scratchBuffer = *p.clasScratch->slice(), .geometryBuffers = geometry,
        .templateStorageBuffer = p.buildMode == StreamCLASTessellationBuildMode::Templates
            ? p.templates->storageRange : BufferSlice{}});
    if (checkpoint) { checkpoint("AfterTessellationCLAS"); }
    if (result) { result = synchronizeTessellation(commands); }
    if (result) { log = "Prepare tessellated BLAS inputs"; result = p.dispatch(commands, 4, (p.instanceCount + 63u) / 64u); }
    if (result) {
        log = "Build tessellated instance BLAS";
        result = commands.buildClusterAccelerationStructureBottomLevels({
            .maxClusterCountPerAccelerationStructure = p.capacity, .maxTotalClusterCount = p.capacity,
            .maxAccelerationStructureCount = p.instanceCount, .buildInfoBuffer = *p.blasInfos->slice(),
            .buildInfoCountBuffer = *p.counts->slice({4, 4}), .destinationStorageBuffer = *p.blasStorage->slice(),
            .destinationAddressBuffer = *p.blasAddresses->slice(), .scratchBuffer = *p.blasScratch->slice()});
    }
    if (result) { result = synchronizeTessellation(commands); }
    if (result) { log.clear(); }
    return result;
}

Result<> StreamCLASTessellation::publish(CommandBuffer& commands)
{
    auto& p = *impl_;
    if (!p.parameters) { return {}; }
    auto result = commands.retainResource(p.parameters->retainAllocation());
    if (result) { result = commands.retainResource(p.materials); }
    if (result) { result = synchronizeTessellation(commands); }
    if (result) { result = p.dispatch(commands, 5, (p.instanceCount + 63u) / 64u); }
    if (result) {
        result = commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
            [initialized = p.historyInitialized] { *initialized = true; }, nullptr));
    }
    return result;
}

uint32_t StreamCLASTessellation::patchCapacity() const
{
    return impl_->capacity;
}

StreamCLASTessellationBuildMode StreamCLASTessellation::buildMode() const
{
    return impl_->buildMode;
}

StreamCLASTessellationTemplateShape StreamCLASTessellation::templateShape() const
{
    return impl_->templateShape;
}

StreamCLASTessellationInstantiationFlags StreamCLASTessellation::instantiationFlags() const
{
    return impl_->instantiationFlags;
}

std::string StreamCLASTessellation::diagnosticIdentity() const
{
    const auto& p = *impl_;
    return nlohmann::json{{"mode", p.buildMode == StreamCLASTessellationBuildMode::Templates ? "templates" : "triangles"},
        {"templateShape", p.templateShape == StreamCLASTessellationTemplateShape::Equilateral ? "equilateral" : "right"},
        {"instantiationFlags", p.instantiationFlags == StreamCLASTessellationInstantiationFlags::None ? "none" : "fastTrace"},
        {"enabled", p.enabled}, {"diagnosticBuildSizes", p.diagnosticBuildSizes},
        {"module", "Features/GPUDriven/StreamCLASTessellation"},
        {"entry", "tessellationMain"}, {"spirvSha256", p.shaderHash}, {"maxPatches", p.capacity},
        {"patternCount", p.templates ? kTemplateCount : 0u},
        {"templatesReady", p.templates && p.templates->initialized},
        {"templateBytes", p.templates ? p.templates->storage->slice()->size() : 0u},
        {"templateAuxiliaryBytes", p.templates ? p.templates->vertices->desc().size +
            p.templates->indices->desc().size + p.templates->infos->desc().size +
            p.templates->addresses->desc().size + p.templates->scratch->desc().size : 0u},
        {"clasBytes", p.clasStorage->slice()->size()},
        {"scratchBytes", p.clasScratch->slice()->size()}, {"buildInfoStride", p.clasInfoStride}}.dump();
}

Buffer* StreamCLASTessellation::statisticsBuffer() const
{
    return impl_->counts.get();
}

Buffer* StreamCLASTessellation::buildSizesBuffer() const
{
    return impl_->diagnosticBuildSizes ? impl_->clasSizes.get() : nullptr;
}

Buffer* StreamCLASTessellation::metadataBuffer() const
{
    return impl_->leaves.get();
}

Buffer* StreamCLASTessellation::vertexBuffer() const
{
    return impl_->vertices.get();
}

} // namespace metallic::render
