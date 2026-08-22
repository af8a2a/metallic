#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace metallic::render {
namespace {

using BuildClock = std::chrono::steady_clock;

struct RayTracingVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct PrimitiveInput {
    uint32_t renderPrimitiveIndex = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t triangleCount = 0;
    bool opaque = true;
};

double elapsedMilliseconds(BuildClock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(BuildClock::now() - begin).count();
}

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    return alignment > 1 ? (value + alignment - 1) & ~(alignment - 1) : value;
}

std::string resultMessage(const char* action, Result result)
{
    return std::string(action) + " returned " + resultToString(result);
}

bool primitiveUsesAlphaMask(
    const scene::Scene& scene,
    const scene::RenderPrimitive& primitive)
{
    if (primitive.materialIndex < 0 ||
        static_cast<size_t>(primitive.materialIndex) >= scene.materials().size()) {
        return false;
    }
    return scene.materials()[static_cast<size_t>(primitive.materialIndex)].alphaMode == "MASK";
}

void copyTransform(float (&destination)[3][4], const float4x4& source)
{
    destination[0][0] = source.a00;
    destination[0][1] = source.a01;
    destination[0][2] = source.a02;
    destination[0][3] = source.a03;
    destination[1][0] = source.a10;
    destination[1][1] = source.a11;
    destination[1][2] = source.a12;
    destination[1][3] = source.a13;
    destination[2][0] = source.a20;
    destination[2][1] = source.a21;
    destination[2][2] = source.a22;
    destination[2][3] = source.a23;
}

Result createBuffer(
    Device& device,
    uint64_t size,
    BufferUsageBits usage,
    MemoryLocation memoryLocation,
    std::unique_ptr<Buffer>& outBuffer,
    const char* label,
    std::string& log)
{
    const Result result = device.createBuffer(
        BufferDesc{
            .size = size,
            .usage = usage,
            .memoryLocation = memoryLocation,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        },
        outBuffer);
    if (!result) {
        log = resultMessage(label, result);
    }
    return result;
}

template <typename T>
Result uploadVector(Buffer& buffer, const std::vector<T>& values, const char* label, std::string& log)
{
    if (values.empty()) {
        return {};
    }
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        log = std::string(label) + " map failed";
        return makeError(Error::Failure);
    }
    const uint64_t byteSize = static_cast<uint64_t>(values.size()) * sizeof(T);
    std::memcpy(mapped, values.data(), static_cast<size_t>(byteSize));
    buffer.flush(0, byteSize);
    buffer.unmap();
    return {};
}

} // namespace

struct SceneAccelerationStructureBuilder::Impl {
    SceneAccelerationStructureStats stats;
    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> instanceBuffer;
    std::unique_ptr<Buffer> scratchBuffer;
    std::vector<std::unique_ptr<RayTracingAccelerationStructure>> blases;
    std::unique_ptr<RayTracingAccelerationStructure> tlas;
    std::vector<int32_t> primitiveToBlas;
    uint64_t sourceTransformRevision = 0;
    uint64_t scratchOffset = 0;
    std::unique_ptr<CommandPool> buildCommandPool;
    std::unique_ptr<CommandBuffer> buildCommandBuffer;
    std::unique_ptr<Fence> buildFence;
    SceneAccelerationStructureBuildState buildState =
        SceneAccelerationStructureBuildState::Idle;

    void destroy()
    {
        if (buildFence != nullptr && !buildFence->isSignaled()) {
            (void)buildFence->wait();
        }
        buildCommandBuffer.reset();
        buildCommandPool.reset();
        buildFence.reset();
        tlas.reset();
        blases.clear();
        scratchBuffer.reset();
        instanceBuffer.reset();
        indexBuffer.reset();
        vertexBuffer.reset();
        primitiveToBlas.clear();
        stats = {};
        sourceTransformRevision = 0;
        scratchOffset = 0;
        buildState = SceneAccelerationStructureBuildState::Idle;
    }

    ~Impl()
    {
        destroy();
    }
};

SceneAccelerationStructureBuilder::SceneAccelerationStructureBuilder()
    : impl_(std::make_unique<Impl>())
{
}

SceneAccelerationStructureBuilder::~SceneAccelerationStructureBuilder() = default;
SceneAccelerationStructureBuilder::SceneAccelerationStructureBuilder(
    SceneAccelerationStructureBuilder&&) noexcept = default;
SceneAccelerationStructureBuilder& SceneAccelerationStructureBuilder::operator=(
    SceneAccelerationStructureBuilder&&) noexcept = default;

Result SceneAccelerationStructureBuilder::build(
    Device& device,
    Queue& queue,
    const scene::Scene& scene,
    std::string& log)
{
    return buildInternal(device, queue, scene, true, log);
}

Result SceneAccelerationStructureBuilder::beginBuild(
    Device& device,
    Queue& queue,
    const scene::Scene& scene,
    std::string& log)
{
    return buildInternal(device, queue, scene, false, log);
}

Result SceneAccelerationStructureBuilder::buildInternal(
    Device& device,
    Queue& queue,
    const scene::Scene& scene,
    bool waitForCompletion,
    std::string& log)
{
    log.clear();
    if (!scene.valid()) {
        log = "Scene is not loaded.";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().rayTracingAccelerationStructure) {
        log = "Ray tracing acceleration structure capability is unavailable.";
        return makeError(Error::Unsupported);
    }

    clear();
    const BuildClock::time_point begin = BuildClock::now();
    const std::vector<scene::RenderPrimitive>& renderPrimitives = scene.renderPrimitives();
    std::vector<PrimitiveInput> primitiveInputs;
    std::vector<int32_t> primitiveToBlas(renderPrimitives.size(), -1);
    std::vector<RayTracingVertex> vertices;
    std::vector<uint32_t> indices;

    for (uint32_t primitiveIndex = 0; primitiveIndex < renderPrimitives.size(); ++primitiveIndex) {
        const scene::RenderPrimitive& primitive = renderPrimitives[primitiveIndex];
        if (primitive.mode != 4 || primitive.positions.size() < 3) {
            continue;
        }
        const uint64_t sourceIndexCount = primitive.indices.empty()
            ? (primitive.positions.size() / 3) * 3
            : (primitive.indices.size() / 3) * 3;
        if (sourceIndexCount < 3 ||
            sourceIndexCount > std::numeric_limits<uint32_t>::max() ||
            primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        const PrimitiveInput input{
            .renderPrimitiveIndex = primitiveIndex,
            .firstVertex = static_cast<uint32_t>(vertices.size()),
            .vertexCount = static_cast<uint32_t>(primitive.positions.size()),
            .firstIndex = static_cast<uint32_t>(indices.size()),
            .indexCount = static_cast<uint32_t>(sourceIndexCount),
            .triangleCount = static_cast<uint32_t>(sourceIndexCount / 3),
            .opaque = !primitiveUsesAlphaMask(scene, primitive),
        };
        for (const float3& position : primitive.positions) {
            vertices.push_back(RayTracingVertex{position.x, position.y, position.z});
        }
        bool validIndices = true;
        if (primitive.indices.empty()) {
            for (uint32_t index = 0; index < input.indexCount; ++index) {
                indices.push_back(index);
            }
        } else {
            for (uint32_t index = 0; index < input.indexCount; ++index) {
                const uint32_t sourceIndex = primitive.indices[index];
                if (sourceIndex >= input.vertexCount) {
                    validIndices = false;
                    break;
                }
                indices.push_back(sourceIndex);
            }
        }
        if (!validIndices) {
            vertices.resize(input.firstVertex);
            indices.resize(input.firstIndex);
            continue;
        }
        primitiveToBlas[primitiveIndex] = static_cast<int32_t>(primitiveInputs.size());
        primitiveInputs.push_back(input);
    }

    if (primitiveInputs.empty()) {
        log = "Scene contains no triangle primitives suitable for acceleration structures.";
        return makeError(Error::Unsupported);
    }

    Result result = createBuffer(
        device,
        static_cast<uint64_t>(vertices.size()) * sizeof(RayTracingVertex),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->vertexBuffer,
        "createBuffer(scene RTAS vertices)",
        log);
    if (!result || !(result = uploadVector(*impl_->vertexBuffer, vertices, "scene RTAS vertices", log))) {
        clear();
        return result;
    }
    result = createBuffer(
        device,
        static_cast<uint64_t>(indices.size()) * sizeof(uint32_t),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->indexBuffer,
        "createBuffer(scene RTAS indices)",
        log);
    if (!result || !(result = uploadVector(*impl_->indexBuffer, indices, "scene RTAS indices", log))) {
        clear();
        return result;
    }

    std::vector<RayTracingTriangleGeometryDesc> geometries;
    std::vector<RayTracingAccelerationStructureBuildSizes> blasSizes;
    geometries.reserve(primitiveInputs.size());
    blasSizes.reserve(primitiveInputs.size());
    impl_->blases.reserve(primitiveInputs.size());
    uint64_t maxScratchSize = 0;
    uint64_t accelerationStructureBytes = 0;
    for (const PrimitiveInput& input : primitiveInputs) {
        geometries.push_back(RayTracingTriangleGeometryDesc{
            .vertexBuffer = impl_->vertexBuffer.get(),
            .vertexOffset = static_cast<uint64_t>(input.firstVertex) * sizeof(RayTracingVertex),
            .vertexStride = sizeof(RayTracingVertex),
            .vertexFormat = Format::Rgb32Sfloat,
            .vertexCount = input.vertexCount,
            .indexBuffer = impl_->indexBuffer.get(),
            .indexOffset = static_cast<uint64_t>(input.firstIndex) * sizeof(uint32_t),
            .indexType = RayTracingIndexType::Uint32,
            .primitiveCount = input.triangleCount,
            .flags = input.opaque ? RayTracingGeometryFlags::Opaque : RayTracingGeometryFlags::None,
        });
        RayTracingAccelerationStructureBuildSizes sizes;
        result = device.queryRayTracingAccelerationStructureBuildSizes(
            RayTracingAccelerationStructureBuildInputs{
                .type = RayTracingAccelerationStructureType::BottomLevel,
                .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                .geometries = &geometries.back(),
                .geometryCount = 1,
            },
            sizes);
        if (!result) {
            log = resultMessage("queryRayTracingAccelerationStructureBuildSizes(BLAS)", result);
            clear();
            return result;
        }
        std::unique_ptr<RayTracingAccelerationStructure> blas;
        result = device.createRayTracingAccelerationStructure(
            RayTracingAccelerationStructureDesc{
                .type = RayTracingAccelerationStructureType::BottomLevel,
                .buildFlags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                .size = sizes.accelerationStructureSize,
            },
            blas);
        if (!result) {
            log = resultMessage("createRayTracingAccelerationStructure(BLAS)", result);
            clear();
            return result;
        }
        maxScratchSize = std::max(maxScratchSize, sizes.buildScratchSize);
        accelerationStructureBytes += sizes.accelerationStructureSize;
        blasSizes.push_back(sizes);
        impl_->blases.push_back(std::move(blas));
    }

    std::vector<RayTracingInstanceDesc> instances;
    instances.reserve(scene.renderNodes().size());
    for (const scene::RenderNode& renderNode : scene.renderNodes()) {
        if (!renderNode.visible || renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >= primitiveToBlas.size()) {
            continue;
        }
        const int32_t blasIndex =
            primitiveToBlas[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (blasIndex < 0 || static_cast<size_t>(blasIndex) >= impl_->blases.size()) {
            continue;
        }
        RayTracingInstanceDesc instance{
            .bottomLevel = impl_->blases[static_cast<size_t>(blasIndex)].get(),
            .customIndex = static_cast<uint32_t>(renderNode.renderPrimitiveIndex),
            .mask = 0xff,
            .flags = RayTracingInstanceFlags::TriangleFacingCullDisable,
        };
        copyTransform(instance.transform, renderNode.worldMatrix);
        instances.push_back(instance);
    }
    const uint32_t visibleInstanceCount = static_cast<uint32_t>(instances.size());
    if (instances.empty()) {
        RayTracingInstanceDesc sentinel{
            .bottomLevel = impl_->blases.front().get(),
            .mask = 0,
            .flags = RayTracingInstanceFlags::TriangleFacingCullDisable,
        };
        copyTransform(sentinel.transform, float4x4::Identity());
        instances.push_back(sentinel);
    }

    result = device.createRayTracingInstanceBuffer(
        instances.data(),
        static_cast<uint32_t>(instances.size()),
        impl_->instanceBuffer);
    if (!result) {
        log = resultMessage("createRayTracingInstanceBuffer", result);
        clear();
        return result;
    }

    constexpr RayTracingAccelerationStructureBuildFlags kTlasFlags =
        RayTracingAccelerationStructureBuildFlags::PreferFastTrace |
        RayTracingAccelerationStructureBuildFlags::AllowUpdate;
    RayTracingAccelerationStructureBuildSizes tlasSizes;
    result = device.queryRayTracingAccelerationStructureBuildSizes(
        RayTracingAccelerationStructureBuildInputs{
            .type = RayTracingAccelerationStructureType::TopLevel,
            .flags = kTlasFlags,
            .instanceCount = static_cast<uint32_t>(instances.size()),
        },
        tlasSizes);
    if (!result) {
        log = resultMessage("queryRayTracingAccelerationStructureBuildSizes(TLAS)", result);
        clear();
        return result;
    }
    result = device.createRayTracingAccelerationStructure(
        RayTracingAccelerationStructureDesc{
            .type = RayTracingAccelerationStructureType::TopLevel,
            .buildFlags = kTlasFlags,
            .size = tlasSizes.accelerationStructureSize,
        },
        impl_->tlas);
    if (!result) {
        log = resultMessage("createRayTracingAccelerationStructure(TLAS)", result);
        clear();
        return result;
    }
    accelerationStructureBytes += tlasSizes.accelerationStructureSize;
    maxScratchSize = std::max(
        maxScratchSize,
        std::max(tlasSizes.buildScratchSize, tlasSizes.updateScratchSize));

    RayTracingAccelerationStructureProperties properties;
    result = device.queryRayTracingAccelerationStructureProperties(properties);
    if (!result) {
        log = resultMessage("queryRayTracingAccelerationStructureProperties", result);
        clear();
        return result;
    }
    const uint64_t scratchBufferSize = maxScratchSize + properties.scratchAlignment - 1;
    result = createBuffer(
        device,
        scratchBufferSize,
        BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->scratchBuffer,
        "createBuffer(scene RTAS scratch)",
        log);
    if (!result) {
        clear();
        return result;
    }
    const uint64_t scratchAddress = impl_->scratchBuffer->deviceAddress();
    impl_->scratchOffset = alignUp(scratchAddress, properties.scratchAlignment) - scratchAddress;

    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
    if (!(result = device.createCommandPool(queue, commandPool)) ||
        !(result = commandPool->createCommandBuffer(commandBuffer)) ||
        !(result = device.createFence(false, fence)) ||
        !(result = commandBuffer->begin())) {
        log = resultMessage("create scene RTAS build submission", result);
        clear();
        return result;
    }
    for (size_t index = 0; index < geometries.size(); ++index) {
        result = commandBuffer->buildRayTracingAccelerationStructure(
            RayTracingAccelerationStructureBuildDesc{
                .destination = impl_->blases[index].get(),
                .geometries = &geometries[index],
                .geometryCount = 1,
                .scratchBuffer = impl_->scratchBuffer.get(),
                .scratchBufferOffset = impl_->scratchOffset,
            });
        if (!result) {
            log = resultMessage("buildRayTracingAccelerationStructure(BLAS)", result);
            clear();
            return result;
        }
    }
    result = commandBuffer->buildRayTracingAccelerationStructure(
        RayTracingAccelerationStructureBuildDesc{
            .destination = impl_->tlas.get(),
            .instanceBuffer = impl_->instanceBuffer.get(),
            .instanceCount = static_cast<uint32_t>(instances.size()),
            .scratchBuffer = impl_->scratchBuffer.get(),
            .scratchBufferOffset = impl_->scratchOffset,
        });
    if (!result || !(result = commandBuffer->end())) {
        log = resultMessage("record scene TLAS build", result);
        clear();
        return result;
    }
    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    result = queue.submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (!result) {
        log = resultMessage("Queue::submit(scene RTAS build)", result);
        clear();
        return result;
    }

    if (waitForCompletion) {
        result = fence->wait();
        if (!result) {
            log = resultMessage("Fence::wait(scene RTAS build)", result);
            clear();
            return result;
        }
        impl_->buildState = SceneAccelerationStructureBuildState::Ready;
    } else {
        impl_->buildCommandPool = std::move(commandPool);
        impl_->buildCommandBuffer = std::move(commandBuffer);
        impl_->buildFence = std::move(fence);
        impl_->buildState = SceneAccelerationStructureBuildState::Building;
    }

    uint64_t triangleCount = 0;
    for (const PrimitiveInput& input : primitiveInputs) {
        triangleCount += input.triangleCount;
    }
    impl_->stats = SceneAccelerationStructureStats{
        .blasCount = static_cast<uint32_t>(impl_->blases.size()),
        .instanceCount = visibleInstanceCount,
        .triangleCount = triangleCount,
        .vertexCount = vertices.size(),
        .indexCount = indices.size(),
        .geometryBytes = static_cast<uint64_t>(vertices.size()) * sizeof(RayTracingVertex) +
            static_cast<uint64_t>(indices.size()) * sizeof(uint32_t) +
            static_cast<uint64_t>(instances.size()) * properties.instanceRecordSize,
        .accelerationStructureBytes = accelerationStructureBytes,
        .scratchBytes = scratchBufferSize,
    };
    impl_->primitiveToBlas = std::move(primitiveToBlas);
    impl_->sourceTransformRevision = scene.transformRevision();
    log = "Built scene acceleration structures: " +
        std::to_string(impl_->stats.blasCount) + " BLAS, " +
        std::to_string(impl_->stats.instanceCount) + " visible TLAS instances.";
    spdlog::info(
        "[RTAS] {} triangles={} asBytes={} scratchBytes={} timeMs={:.2f}",
        log,
        impl_->stats.triangleCount,
        impl_->stats.accelerationStructureBytes,
        impl_->stats.scratchBytes,
        elapsedMilliseconds(begin));
    return {};
}

bool SceneAccelerationStructureBuilder::pollBuild()
{
    if (impl_ == nullptr ||
        impl_->buildState != SceneAccelerationStructureBuildState::Building) {
        return valid();
    }
    if (impl_->buildFence == nullptr || !impl_->buildFence->isSignaled()) {
        return false;
    }
    impl_->buildCommandBuffer.reset();
    impl_->buildCommandPool.reset();
    impl_->buildFence.reset();
    impl_->buildState = SceneAccelerationStructureBuildState::Ready;
    return true;
}

SceneAccelerationStructureBuildState SceneAccelerationStructureBuilder::buildState() const
{
    if (impl_ == nullptr) {
        return SceneAccelerationStructureBuildState::Idle;
    }
    if (impl_->buildState == SceneAccelerationStructureBuildState::Building &&
        impl_->buildFence != nullptr && impl_->buildFence->isSignaled()) {
        return SceneAccelerationStructureBuildState::Ready;
    }
    return impl_->buildState;
}

Result SceneAccelerationStructureBuilder::updateInstanceTransforms(
    Device& device,
    Queue& queue,
    const scene::Scene& scene,
    std::string& log)
{
    log.clear();
    if (!valid() || !scene.valid() || impl_->instanceBuffer == nullptr ||
        impl_->scratchBuffer == nullptr || impl_->primitiveToBlas.empty()) {
        log = "Scene acceleration structures are not ready for an instance update.";
        return makeError(Error::InvalidArgument);
    }
    if (impl_->sourceTransformRevision == scene.transformRevision()) {
        return {};
    }

    std::vector<RayTracingInstanceDesc> instances;
    instances.reserve(scene.renderNodes().size());
    for (const scene::RenderNode& renderNode : scene.renderNodes()) {
        if (!renderNode.visible || renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >=
                impl_->primitiveToBlas.size()) {
            continue;
        }
        const int32_t blasIndex = impl_->primitiveToBlas[
            static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (blasIndex < 0 || static_cast<size_t>(blasIndex) >= impl_->blases.size()) {
            continue;
        }
        RayTracingInstanceDesc instance{
            .bottomLevel = impl_->blases[static_cast<size_t>(blasIndex)].get(),
            .customIndex = static_cast<uint32_t>(renderNode.renderPrimitiveIndex),
            .mask = 0xff,
            .flags = RayTracingInstanceFlags::TriangleFacingCullDisable,
        };
        copyTransform(instance.transform, renderNode.worldMatrix);
        instances.push_back(instance);
    }
    if (instances.size() != impl_->stats.instanceCount) {
        log = "Scene instance topology changed; a full acceleration-structure rebuild is required.";
        return makeError(Error::InvalidArgument);
    }
    if (instances.empty()) {
        impl_->sourceTransformRevision = scene.transformRevision();
        return {};
    }

    Result result = device.writeRayTracingInstances(
        *impl_->instanceBuffer,
        instances.data(),
        static_cast<uint32_t>(instances.size()));
    if (!result) {
        log = resultMessage("writeRayTracingInstances", result);
        return result;
    }
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
    if (!(result = device.createCommandPool(queue, commandPool)) ||
        !(result = commandPool->createCommandBuffer(commandBuffer)) ||
        !(result = device.createFence(false, fence)) ||
        !(result = commandBuffer->begin())) {
        log = resultMessage("create scene TLAS update submission", result);
        return result;
    }
    result = commandBuffer->buildRayTracingAccelerationStructure(
        RayTracingAccelerationStructureBuildDesc{
            .destination = impl_->tlas.get(),
            .source = impl_->tlas.get(),
            .mode = RayTracingAccelerationStructureBuildMode::Update,
            .instanceBuffer = impl_->instanceBuffer.get(),
            .instanceCount = static_cast<uint32_t>(instances.size()),
            .scratchBuffer = impl_->scratchBuffer.get(),
            .scratchBufferOffset = impl_->scratchOffset,
        });
    if (!result || !(result = commandBuffer->end())) {
        log = resultMessage("record scene TLAS update", result);
        return result;
    }
    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    result = queue.submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (!result || !(result = fence->wait())) {
        log = resultMessage("submit scene TLAS update", result);
        return result;
    }
    impl_->sourceTransformRevision = scene.transformRevision();
    log = "Updated scene acceleration-structure instance transforms.";
    return {};
}

void SceneAccelerationStructureBuilder::clear()
{
    if (impl_ != nullptr) {
        impl_->destroy();
    }
}

bool SceneAccelerationStructureBuilder::valid() const
{
    return impl_ != nullptr && impl_->tlas != nullptr && impl_->tlas->valid() &&
        buildState() == SceneAccelerationStructureBuildState::Ready;
}

RayTracingAccelerationStructure* SceneAccelerationStructureBuilder::accelerationStructure() const
{
    return impl_ != nullptr ? impl_->tlas.get() : nullptr;
}

const SceneAccelerationStructureStats& SceneAccelerationStructureBuilder::stats() const
{
    static const SceneAccelerationStructureStats emptyStats;
    return impl_ != nullptr ? impl_->stats : emptyStats;
}

} // namespace metallic::render
