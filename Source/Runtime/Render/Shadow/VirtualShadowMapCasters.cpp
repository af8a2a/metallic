#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"

#include "Runtime/Render/Shadow/VirtualShadowMapPrototype.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>

namespace metallic::render {
namespace {

constexpr uint32_t kStreamTrianglesPerCluster = 128;
// VirtualShadowMapGeometry.slang borrows this exact stream instance ABI.
static_assert(sizeof(MeshletStreamGPUInstance) == 96);
static_assert(offsetof(MeshletStreamGPUInstance, world0) == 16);
static_assert(offsetof(MeshletStreamGPUInstance, world3) == 64);
static_assert(offsetof(MeshletStreamGPUInstance, boundsCenterRadius) == 80);

struct GeometryParameters
{
    uint32_t streamed = 0, triangles = 0, candidates = 0, instances = 0;
    uint32_t pages = 0, pageTable = 0, pageWordCount = 0, triangleCapacity = 0;
};
static_assert(sizeof(GeometryParameters) == 32);

struct CasterUploads
{
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::shared_ptr<MeshletStreamRuntime> stream;
};

Result<uint32_t> upload(Device& device, ResourceRegistry& registry, CasterUploads& owner,
    std::vector<ResourceLease>& leases, const void* data, uint64_t bytes, uint32_t stride)
{
    auto buffer = device.createBuffer({.size = std::max<uint64_t>(bytes, stride), .structureStride = stride,
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
    if (!buffer) { return std::unexpected(buffer.error()); }
    if (bytes != 0) {
        void* destination = (*buffer)->map();
        if (!destination) { return makeError(Error::Failure); }
        std::memcpy(destination, data, static_cast<size_t>(bytes));
        (*buffer)->flush();
        (*buffer)->unmap();
    }
    auto lease = registry.storageBuffer(**buffer);
    if (!lease) { return std::unexpected(lease.error()); }
    const uint32_t index = lease->shaderIndex();
    leases.push_back(*lease);
    owner.buffers.push_back(std::move(*buffer));
    return index;
}

Result<> gatherResidentTriangles(const scene::Scene& scene,
    std::vector<VirtualShadowMapPrototypeTriangle>& triangles, scene::Bounds& bounds, std::string& log)
{
    for (const auto& node : scene.renderNodes()) {
        if (!node.visible || node.renderPrimitiveIndex == scene::kInvalidSceneIndex) { continue; }
        if (node.renderPrimitiveIndex < 0 || size_t(node.renderPrimitiveIndex) >= scene.renderPrimitives().size()) {
            log = "VSM resident caster references an invalid scene primitive";
            return makeError(Error::InvalidArgument);
        }
        const auto& primitive = scene.renderPrimitives()[node.renderPrimitiveIndex];
        if (primitive.mode != 4) { continue; }
        if (primitive.positions.empty() && primitive.triangleCount != 0) {
            log = "VSM requires the published StreamAsset geometry for a metadata-only scene";
            return makeError(Error::InvalidArgument);
        }
        const bool indexed = !primitive.indices.empty();
        const size_t count = indexed ? primitive.indices.size() : primitive.positions.size();
        if (count % 3 != 0 || triangles.size() + count / 3 > UINT32_MAX) {
            log = "VSM resident triangle list has an invalid or unsupported size";
            return makeError(Error::InvalidArgument);
        }
        for (size_t triangleIndex = 0; triangleIndex < count; triangleIndex += 3) {
            VirtualShadowMapPrototypeTriangle triangle;
            triangle.materialIndex = static_cast<uint32_t>(std::max(node.materialIndex, 0));
            for (uint32_t vertex = 0; vertex < 3; ++vertex) {
                const size_t index = indexed ? primitive.indices[triangleIndex + vertex] : triangleIndex + vertex;
                if (index >= primitive.positions.size()) {
                    log = "VSM resident triangle references an invalid vertex";
                    return makeError(Error::InvalidArgument);
                }
                const float3 position = node.worldMatrix * primitive.positions[index];
                triangle.vertices[vertex].position = {position.x, position.y, position.z, 1};
                if (index < primitive.texcoords0.size()) {
                    const float2 uv = primitive.texcoords0[index];
                    triangle.vertices[vertex].uv = {uv.x, uv.y, 0, 0};
                }
                bounds.include(position);
            }
            triangles.push_back(triangle);
        }
    }
    return {};
}

} // namespace

Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapCasters(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, std::string& log)
{
    for (const auto& node : scene.renderNodes()) {
        if (!node.visible || node.renderPrimitiveIndex == scene::kInvalidSceneIndex) { continue; }
        // -1 is the renderer's existing material-zero binding. Other indices
        // enter a GPU material array and must be valid before uploading casters.
        if (node.materialIndex < scene::kInvalidSceneIndex ||
            (node.materialIndex >= 0 && size_t(node.materialIndex) >= std::max<size_t>(scene.materials().size(), 1))) {
            log = "VSM caster references an invalid scene material";
            return makeError(Error::InvalidArgument);
        }
    }
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    auto owner = std::make_shared<CasterUploads>();
    VirtualShadowMapCasterResources resources;
    resources.owner = owner;
    GeometryParameters parameters;
    if (stream) {
        if (!stream->valid()) { return makeError(Error::InvalidArgument); }
        // The coarse projection must cover every terminal branch. Lazy loading
        // cannot make an absent caster page equivalent to empty shadow depth.
        if (!stream->owner->sceneReady()) {
            log = "VSM requires completed initial terminal-page load (initialLoad=true)";
            return makeError(Error::InvalidArgument);
        }
        const auto& asset = *stream->asset;
        if (asset.maxClusterTriangles() > kStreamTrianglesPerCluster) {
            log = "VSM StreamAsset exceeds the supported 128 triangles per cluster";
            return makeError(Error::Unsupported);
        }
        if (auto result = stream->owner->cmdPrepareShadowGeometry(commands); !result) {
            return std::unexpected(result.error());
        }
        owner->stream = stream->owner;
        std::vector<std::array<uint32_t, 4>> candidates;
        for (uint32_t instanceIndex = 0; instanceIndex < asset.instances().size(); ++instanceIndex) {
            const auto& instance = asset.instances()[instanceIndex];
            // All terminal branches are mandatory. The coarsest LOD's fallback
            // range alone omits branches that terminate at a finer LOD.
            for (uint32_t groupIndex : asset.primitiveTerminalGroups(instance.primitiveIndex)) {
                const auto& group = asset.groups()[groupIndex];
                if (candidates.size() + group.clusterCount > UINT32_MAX / kStreamTrianglesPerCluster) {
                    log = "VSM terminal caster triangle capacity exceeds the GPU index range";
                    return makeError(Error::Unsupported);
                }
                for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                    candidates.push_back({instanceIndex, group.pageIndex, cluster, 0});
                }
            }
        }
        parameters.streamed = 1;
        parameters.triangleCapacity = static_cast<uint32_t>(candidates.size()) * kStreamTrianglesPerCluster;
        parameters.pageWordCount = stream->pageBufferBytes / sizeof(uint32_t);
        auto candidateIndex = upload(device, **registry, *owner, resources.leases,
            candidates.data(), candidates.size() * sizeof(candidates[0]), sizeof(candidates[0]));
        if (!candidateIndex) { return std::unexpected(candidateIndex.error()); }
        parameters.candidates = *candidateIndex;
        Buffer* buffers[] = {stream->instanceBuffer, stream->pageBuffer, stream->pageTableBuffer};
        uint32_t* indices[] = {&parameters.instances, &parameters.pages, &parameters.pageTable};
        for (size_t i = 0; i < std::size(buffers); ++i) {
            auto lease = (*registry)->storageBuffer(*buffers[i]);
            if (!lease) { return std::unexpected(lease.error()); }
            *indices[i] = lease->shaderIndex();
            resources.leases.push_back(*lease);
        }
        // Scene::refreshTransforms maintains complete bounds after node/mount
        // edits, independently of the producer's camera-visible geometry cut.
        resources.bounds = scene.bounds();
        resources.streamed = true;
    } else {
        std::vector<VirtualShadowMapPrototypeTriangle> triangles;
        if (auto result = gatherResidentTriangles(scene, triangles, resources.bounds, log); !result) {
            return std::unexpected(result.error());
        }
        auto triangleIndex = upload(device, **registry, *owner, resources.leases,
            triangles.data(), triangles.size() * sizeof(triangles[0]), sizeof(triangles[0]));
        if (!triangleIndex) { return std::unexpected(triangleIndex.error()); }
        parameters.triangles = *triangleIndex;
        parameters.triangleCapacity = static_cast<uint32_t>(triangles.size());
    }
    auto parameterIndex = upload(device, **registry, *owner, resources.leases,
        &parameters, sizeof(parameters), sizeof(parameters));
    if (!parameterIndex) { return std::unexpected(parameterIndex.error()); }
    resources.parameters = owner->buffers.back().get();
    resources.triangleCapacity = parameters.triangleCapacity;
    if (auto result = commands.retainResource(owner); !result) { return std::unexpected(result.error()); }
    for (const auto& lease : resources.leases) {
        if (auto result = (*registry)->retain(commands, lease); !result) { return std::unexpected(result.error()); }
    }
    return resources;
}

} // namespace metallic::render
