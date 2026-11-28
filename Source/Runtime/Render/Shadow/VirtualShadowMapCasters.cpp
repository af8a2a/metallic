#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"

#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/Shadow/VirtualShadowMapPrototype.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include "Runtime/Render/GPUDrivenTessellation.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <type_traits>

namespace metallic::render {

struct VirtualShadowMapLODMetadataCache
{
    uint64_t sceneIdentity = 0;
    uint64_t structuralRevision = 0;
    uint64_t geometryRevision = 0;
    bool streamed = false;
    std::vector<std::unique_ptr<VirtualShadowMapLODHierarchy>> primitives;
};

namespace {

constexpr uint32_t kStreamTrianglesPerCluster = 1u << kVSMLODStreamClusterShift;
// VirtualShadowMapGeometry.slang borrows this exact stream instance ABI.
static_assert(sizeof(MeshletStreamGPUInstance) == 96);
static_assert(offsetof(MeshletStreamGPUInstance, world0) == 16);
static_assert(offsetof(MeshletStreamGPUInstance, world3) == 64);
static_assert(offsetof(MeshletStreamGPUInstance, boundsCenterRadius) == 80);

struct GeometryParameters
{
    uint32_t streamed = 0, triangles = 0, candidates = 0, instances = 0;
    uint32_t pages = 0, pageTable = 0, pageWordCount = 0, triangleCapacity = 0;
    uint32_t normals = UINT32_MAX, tessellation = UINT32_MAX, tessellationFactor = 1, subdivisionDepth = 0;
    uint32_t residentClusterBounds = UINT32_MAX, casterLayerMask = UINT32_MAX, casterLayer = 0, reserved2 = 0;
};
static_assert(sizeof(GeometryParameters) == 64);
static_assert(offsetof(GeometryParameters, subdivisionDepth) == 44);
static_assert(offsetof(GeometryParameters, residentClusterBounds) == 48);

struct CasterClusterBounds
{
    std::array<float, 4> minimum, maximum;
};
static_assert(sizeof(CasterClusterBounds) == 32);

struct CasterHierarchyNode
{
    std::array<float, 3> minimum;
    uint32_t escapeIndex;
    std::array<float, 3> maximum;
    uint32_t instanceIndex;
};
static_assert(sizeof(CasterHierarchyNode) == 32);
static_assert(offsetof(CasterHierarchyNode, maximum) == 16);

struct CasterHierarchyLeaf
{
    scene::Bounds bounds;
    std::array<float, 3> center;
    uint32_t instanceIndex;
};

void splitCasterHierarchy(std::span<CasterHierarchyLeaf> leaves)
{
    scene::Bounds centers;
    for (const auto& leaf : leaves) {
        centers.include(float3{leaf.center[0], leaf.center[1], leaf.center[2]});
    }
    const auto extent = centers.max - centers.min;
    size_t axis = extent.x < extent.y ? 1u : 0u;
    if (extent.z > std::max(extent.x, extent.y)) { axis = 2u; }
    std::nth_element(leaves.begin(), leaves.begin() + leaves.size() / 2u, leaves.end(),
        [axis](const CasterHierarchyLeaf& a, const CasterHierarchyLeaf& b) {
            if (a.center[axis] != b.center[axis]) { return a.center[axis] < b.center[axis]; }
            return a.instanceIndex < b.instanceIndex;
        });
}

void appendCasterHierarchyTree(std::span<CasterHierarchyLeaf> leaves,
    std::span<CasterHierarchyNode> nodes, uint32_t& next)
{
    auto& node = nodes[next++];
    scene::Bounds bounds;
    for (const auto& leaf : leaves) {
        if (!leaf.bounds.valid) {
            // An unknown child must also prevent its ancestors from hiding the range.
            bounds.reset();
            break;
        }
        bounds.include(leaf.bounds);
    }
    node.minimum = {bounds.valid ? bounds.min.x : 1.0f, bounds.min.y, bounds.min.z};
    node.maximum = {bounds.valid ? bounds.max.x : 0.0f, bounds.max.y, bounds.max.z};
    node.instanceIndex = UINT32_MAX;
    if (leaves.size() == 1u) {
        node.instanceIndex = leaves.front().instanceIndex;
    } else {
        splitCasterHierarchy(leaves);
        const size_t firstCount = leaves.size() / 2u;
        appendCasterHierarchyTree(leaves.first(firstCount), nodes, next);
        appendCasterHierarchyTree(leaves.subspan(firstCount), nodes, next);
    }
    node.escapeIndex = next;
}

std::vector<CasterHierarchyNode> buildCasterHierarchy(std::span<const VirtualShadowMapCasterState> states,
    std::span<const scene::Bounds> streamedBounds)
{
    assert(streamedBounds.empty() || streamedBounds.size() == states.size());
    std::vector<CasterHierarchyLeaf> leaves;
    leaves.reserve(states.size());
    for (uint32_t index = 0; index < states.size(); ++index) {
        const auto& state = states[index];
        if (state.triangleCount == 0) { continue; }
        auto bounds = streamedBounds.empty() ? state.bounds : streamedBounds[index];
        bounds.valid = bounds.valid && std::isfinite(bounds.min.x) && std::isfinite(bounds.min.y) &&
            std::isfinite(bounds.min.z) && std::isfinite(bounds.max.x) && std::isfinite(bounds.max.y) &&
            std::isfinite(bounds.max.z);
        const auto center = bounds.valid ? bounds.min * 0.5f + bounds.max * 0.5f : float3(0.0f);
        leaves.push_back({bounds, {center.x, center.y, center.z}, index});
    }
    std::vector<CasterHierarchyNode> nodes;
    const auto append = [&](auto&& self, std::span<CasterHierarchyLeaf> remaining) -> void {
        constexpr uint32_t leavesPerTree = (kVSMHierarchyHZBTreeStride + 1u) / 2u;
        if (remaining.size() > leavesPerTree) {
            splitCasterHierarchy(remaining);
            const size_t firstCount = remaining.size() / 2u;
            self(self, remaining.first(firstCount));
            self(self, remaining.subspan(firstCount));
            return;
        }
        uint32_t next = uint32_t(nodes.size());
        nodes.resize(nodes.size() + kVSMHierarchyHZBTreeStride);
        appendCasterHierarchyTree(remaining, nodes, next);
    };
    if (!leaves.empty()) { append(append, leaves); }
    return nodes;
}

struct CasterUploads
{
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::shared_ptr<MeshletStreamRuntime> stream;
    std::shared_ptr<void> sourceOwner;
    GeometryParameters parameters;
};

struct ShadowCasterSelection
{
    bool replaceResident = false;
    bool replaceStream = false;
    bool includeBase = false;
    std::vector<uint32_t> residentClusters;
    std::vector<std::array<uint32_t, 2>> streamClusters; // Global group, local cluster.
    uint64_t lodKey = 0;
};

std::shared_ptr<VirtualShadowMapLODMetadataCache> makeLODMetadataCache(
    const scene::Scene& scene, const MeshletStreamShadowGeometryView* stream)
{
    auto cache = std::make_shared<VirtualShadowMapLODMetadataCache>();
    cache->sceneIdentity = scene.resourceIdentity();
    cache->structuralRevision = scene.sceneGraph().structuralRevision();
    cache->geometryRevision = stream ? stream->geometryRevision : 0;
    cache->streamed = stream != nullptr;
    return cache;
}

struct ContentHash
{
    uint64_t value = 14695981039346656037ull;

    void append(const void* data, size_t bytes)
    {
        const auto* source = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < bytes; ++i) { value = (value ^ source[i]) * 1099511628211ull; }
    }

    template<class T> void append(const T& data)
    {
        append(&data, sizeof(data));
    }
};

struct ShadowMaterialKeys
{
    uint64_t material = 0, texture = 0, tessellation = 0, displacementMaterial = 0;
};

Result<std::vector<ShadowMaterialKeys>> gatherMaterialKeys(const scene::Scene& scene,
    const VirtualShadowMapCasterSettings& settings, std::string& log)
{
    std::vector<ShadowMaterialKeys> keys;
    const scene::RenderMaterial defaultMaterial;
    const bool scoped = !settings.materialTextureRevisions.empty();
    try {
        for (size_t i = 0; i < std::max<size_t>(scene.materials().size(), 1); ++i) {
            const auto& material = scene.materials().empty() ? defaultMaterial : scene.materials()[i];
            ShadowMaterialKeys key;
            ContentHash hash;
            hash.append(material.doubleSided);
            const bool masked = material.alphaMode == "MASK";
            hash.append(masked);
            if (masked) {
                hash.append(material.alphaCutoff);
                bool usesBaseAlpha = true;
                if (!material.valueProgram.empty()) {
                    const auto coverage = MaterialValueIR::parse(material.valueProgram).slice(true);
                    if (!coverage.outputs().empty()) {
                        const auto slice = compileMaterialCoverageSlice(coverage);
                        hash.append(coverage.canonical().data(), coverage.canonical().size());
                        usesBaseAlpha = slice.usesBaseAlpha;
                        for (uint32_t parameter = 0; parameter < 4; ++parameter) {
                            if ((slice.parameterMask & (1u << parameter)) != 0) {
                                hash.append(material.valueParameters.data() + parameter * 4, sizeof(float) * 4);
                            }
                        }
                    }
                }
                if (usesBaseAlpha) {
                    hash.append(material.baseColorFactor.w);
                    hash.append(material.baseColorTexture.textureIndex);
                    hash.append(material.baseColorTexture.uvTransform);
                    if (scoped && material.baseColorTexture.textureIndex >= 0) {
                        key.texture = settings.materialTextureRevisions[i].alpha;
                    }
                }
            }
            const bool displaced = material.displacementMagnitude != 0 && material.displacementTexture.textureIndex >= 0;
            if (!scoped) { hash.append(displaced); }
            if (displaced) {
                ContentHash heightMaterial;
                auto& target = scoped ? heightMaterial : hash;
                target.append(material.displacementMagnitude);
                target.append(material.displacementCenter);
                target.append(material.displacementTexture.textureIndex);
                target.append(material.displacementTexture.uvTransform);
                if (scoped) {
                    key.displacementMaterial = heightMaterial.value;
                    if (settings.tessellationEnabled) {
                        ContentHash tessellation;
                        tessellation.append(settings.tessellationFactor);
                        tessellation.append(settings.subdivisionDepth);
                        key.tessellation = tessellation.value;
                        key.texture = settings.materialTextureRevisions[i].height;
                    }
                }
            }
            key.material = hash.value;
            keys.push_back(key);
        }
    } catch (const std::exception& error) {
        log = std::string("VSM caster Coverage: ") + error.what();
        return makeError(Error::InvalidArgument);
    }
    return keys;
}

float displacementBound(const scene::RenderMaterial& material)
{
    return material.displacementTexture.textureIndex >= 0 ? std::abs(material.displacementMagnitude) *
        std::max(material.displacementCenter, 1.0f - material.displacementCenter) : 0.0f;
}

void expandDisplacedBounds(scene::Bounds& bounds, float displacement)
{
    if (bounds.valid && displacement != 0) {
        bounds.min -= float3(displacement);
        bounds.max += float3(displacement);
    }
}

scene::Bounds transformedBounds(const scene::MeshletStreamBounds& local, const float4x4& world)
{
    scene::Bounds result;
    if (local.valid == 0) { return result; }
    for (uint32_t corner = 0; corner < 8; ++corner) {
        result.include(world * float3((corner & 1u) ? local.max[0] : local.min[0],
            (corner & 2u) ? local.max[1] : local.min[1], (corner & 4u) ? local.max[2] : local.min[2]));
    }
    return result;
}

scene::Bounds streamedCasterHierarchyBounds(scene::Bounds bounds, const scene::MeshletStreamBounds& local,
    const float4x4& world)
{
    if (!bounds.valid) { return bounds; }
    // The shader adds translation after the vertex products. MathLib includes
    // it in the first FMA. Bound both evaluations by the untransformed operand
    // magnitudes so that cancellation cannot underestimate an occlusion bound.
    const float x = std::max(std::abs(local.min[0]), std::abs(local.max[0]));
    const float y = std::max(std::abs(local.min[1]), std::abs(local.max[1]));
    const float z = std::max(std::abs(local.min[2]), std::abs(local.max[2]));
    float3 padding;
    for (uint32_t row = 0; row < 3; ++row) {
        const float magnitude = std::abs(world.a[row]) * x + std::abs(world.a[4u + row]) * y +
            std::abs(world.a[8u + row]) * z + std::abs(world.a[12u + row]);
        padding.a[row] = std::max(magnitude, 1.0f) * 1e-6f;
    }
    bounds.min -= padding;
    bounds.max += padding;
    return bounds;
}

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
    std::span<const ShadowMaterialKeys> materialKeys, std::vector<VirtualShadowMapPrototypeTriangle>& triangles,
    std::vector<std::array<float, 4>>& normals, std::vector<std::array<uint32_t, 2>>& candidates,
    std::vector<scene::Bounds>& clusterBounds, uint32_t patchTriangleCount,
    VirtualShadowMapCasterResources& resources, std::string& log,
    std::span<const ShadowCasterSelection> selections = {})
{
    for (size_t nodeIndex = 0; nodeIndex < scene.renderNodes().size(); ++nodeIndex) {
        const auto& node = scene.renderNodes()[nodeIndex];
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
        const ShadowCasterSelection* selection = selections.empty() ? nullptr : &selections[nodeIndex];
        const bool replace = selection && selection->replaceResident;
        const bool indexed = !primitive.indices.empty();
        std::vector<uint32_t> vertices;
        if (replace) {
            if (selection->includeBase) {
                if (indexed) { vertices = primitive.indices; }
                else {
                    vertices.resize(primitive.positions.size());
                    for (uint32_t vertex = 0; vertex < vertices.size(); ++vertex) { vertices[vertex] = vertex; }
                }
            }
            for (const uint32_t clusterIndex : selection->residentClusters) {
                const auto& cluster = primitive.meshletLodClusters[clusterIndex];
                for (uint32_t vertex = 0; vertex < cluster.triangleCount * 3u; ++vertex) {
                    const uint32_t local = primitive.meshletLodTriangles[cluster.triangleOffset + vertex];
                    vertices.push_back(primitive.meshletLodVertices[cluster.vertexOffset + local]);
                }
            }
        }
        const size_t count = replace ? vertices.size() :
            (indexed ? primitive.indices.size() : primitive.positions.size());
        if (count % 3 != 0 || triangles.size() + count / 3 > UINT32_MAX) {
            log = "VSM resident triangle list has an invalid or unsupported size";
            return makeError(Error::InvalidArgument);
        }
        if (count == 0) { continue; }
        const scene::RenderMaterial defaultMaterial;
        const auto& material = scene.materials().empty() ? defaultMaterial : scene.materials()[std::max(node.materialIndex, 0)];
        const bool displaced = resources.displacedGeometry && displacementBound(material) != 0;
        if (displaced && (primitive.normals.size() != primitive.positions.size() ||
            primitive.texcoords0.size() != primitive.positions.size())) {
            log = "VSM displacement requires complete NORMAL and TEXCOORD_0 vertex streams";
            return makeError(Error::Unsupported);
        }
        const uint32_t expandedCount = displaced ? patchTriangleCount : 1;
        if (count / 3 > (UINT32_MAX - candidates.size()) / expandedCount) {
            log = "VSM displaced resident triangle capacity exceeds the GPU index range";
            return makeError(Error::Unsupported);
        }
        const float3 column0(node.worldMatrix.a[0], node.worldMatrix.a[1], node.worldMatrix.a[2]);
        const float3 column1(node.worldMatrix.a[4], node.worldMatrix.a[5], node.worldMatrix.a[6]);
        const float3 column2(node.worldMatrix.a[8], node.worldMatrix.a[9], node.worldMatrix.a[10]);
        const std::array cofactors{cross(column1, column2), cross(column2, column0), cross(column0, column1)};
        const bool reflected = dot(column0, cofactors[0]) < 0;
        VirtualShadowMapCasterState state{.id = nodeIndex};
        ContentHash transform;
        transform.append(node.worldMatrix.a, sizeof(node.worldMatrix.a));
        state.transformKey = transform.value;
        const auto& keys = materialKeys[static_cast<size_t>(std::max(node.materialIndex, 0))];
        state.materialKey = keys.material;
        state.textureKey = keys.texture;
        state.tessellationKey = keys.tessellation;
        ContentHash heightMaterial{keys.displacementMaterial};
        state.triangleFirst = static_cast<uint32_t>(resources.displacedGeometry ? candidates.size() : triangles.size());
        ContentHash hash;
        hash.append(node.worldMatrix.a, sizeof(node.worldMatrix.a));
        hash.append(state.materialKey);
        hash.append(count);
        for (size_t triangleIndex = 0; triangleIndex < count; triangleIndex += 3) {
            VirtualShadowMapPrototypeTriangle triangle;
            scene::Bounds triangleBounds;
            triangle.materialIndex = static_cast<uint32_t>(std::max(node.materialIndex, 0));
            triangle.reserved[0] = reflected ? 1u : 0u;
            for (uint32_t vertex = 0; vertex < 3; ++vertex) {
                const size_t index = replace ? vertices[triangleIndex + vertex] :
                    (indexed ? primitive.indices[triangleIndex + vertex] : triangleIndex + vertex);
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
                state.bounds.include(position);
                triangleBounds.include(position);
                hash.append(triangle.vertices[vertex].position);
                hash.append(triangle.vertices[vertex].uv);
                if (keys.displacementMaterial != 0 && index < primitive.normals.size()) {
                    const auto local = primitive.normals[index];
                    heightMaterial.append(local.x); heightMaterial.append(local.y); heightMaterial.append(local.z);
                }
                if (resources.displacedGeometry) {
                    float3 normal(0, 1, 0);
                    if (index < primitive.normals.size()) {
                        const auto local = primitive.normals[index];
                        normal = cofactors[0] * local.x + cofactors[1] * local.y + cofactors[2] * local.z;
                        const float normSquared = dot(normal, normal);
                        normal = normSquared > 1e-20f ? normal / std::sqrt(normSquared) : float3(0, 1, 0);
                        normal *= reflected ? -1.0f : 1.0f;
                    }
                    normals.push_back({normal.x, normal.y, normal.z, 0});
                    if (displaced && keys.displacementMaterial == 0) { hash.append(normals.back()); }
                }
            }
            // Microtriangles stay inside the base triangle plus the maximum
            // height excursion. Accumulate once per touched 64-slot group,
            // without evaluating or storing displaced vertices on the CPU.
            expandDisplacedBounds(triangleBounds, displaced ? displacementBound(material) : 0.0f);
            const size_t firstSlot = resources.displacedGeometry ? candidates.size() : triangles.size();
            const size_t firstCluster = firstSlot / kVirtualShadowMapCasterClusterSize;
            const size_t lastCluster = (firstSlot + expandedCount - 1) / kVirtualShadowMapCasterClusterSize;
            clusterBounds.resize(lastCluster + 1);
            for (size_t cluster = firstCluster; cluster <= lastCluster; ++cluster) { clusterBounds[cluster].include(triangleBounds); }
            if (resources.displacedGeometry) {
                for (uint32_t patchTriangle = 0; patchTriangle < expandedCount; ++patchTriangle) {
                    candidates.push_back({static_cast<uint32_t>(triangles.size()), patchTriangle});
                }
            }
            triangles.push_back(triangle);
        }
        expandDisplacedBounds(state.bounds, displaced ? displacementBound(material) : 0.0f);
        state.contentKey = hash.value;
        state.displacementMaterialKey = heightMaterial.value;
        state.triangleCount = static_cast<uint32_t>(resources.displacedGeometry ? candidates.size() : triangles.size()) - state.triangleFirst;
        resources.bounds.include(state.bounds);
        resources.states.push_back(state);
    }
    return {};
}

} // namespace

static Result<VirtualShadowMapCasterResources> prepareCasterGeometry(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, std::string& log,
    const VirtualShadowMapCasterSettings& settings, std::span<const ShadowCasterSelection> selections = {},
    std::shared_ptr<VirtualShadowMapLODMetadataCache> lodMetadata = {})
{
    if (settings.tessellationFactor == 0 || settings.tessellationFactor > kTessellationMaxFactor || settings.subdivisionDepth > 2 ||
        (!settings.displacementTextures.empty() && settings.displacementTextures.size() != scene.materials().size()) ||
        (!settings.materialTextureRevisions.empty() && settings.materialTextureRevisions.size() != scene.materials().size())) {
        log = "VSM caster settings require tessellationFactor 1..8, subdivisionDepth 0..2 and matching material slot counts";
        return makeError(Error::InvalidArgument);
    }
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
    std::vector<scene::Bounds> hierarchyStreamBounds;
    resources.owner = owner;
    resources.sceneIdentity = scene.resourceIdentity();
    resources.structuralRevision = scene.sceneGraph().structuralRevision();
    const bool scoped = !settings.materialTextureRevisions.empty();
    resources.texturePublicationRevision = scoped ? 0 : settings.texturePublicationRevision;
    auto materialKeys = gatherMaterialKeys(scene, settings, log);
    if (!materialKeys) { return std::unexpected(materialKeys.error()); }
    GeometryParameters parameters;
    uint32_t patchTriangleCount = 1;
    std::vector<uint32_t> textureDescriptors;
    for (size_t materialIndex = 0; materialIndex < scene.materials().size(); ++materialIndex) {
        const auto& material = scene.materials()[materialIndex];
        const auto& texture = material.displacementTexture;
        if (!settings.tessellationEnabled || material.displacementMagnitude == 0 || texture.textureIndex < 0) { continue; }
        if (material.alphaMode != "OPAQUE" || texture.texCoord != 0) {
            log = "VSM displacement requires OPAQUE materials and TEXCOORD_0, matching streamed raster displacement";
            return makeError(Error::Unsupported);
        }
        if (!std::isfinite(material.displacementMagnitude) || !std::isfinite(material.displacementCenter) ||
            material.displacementCenter < 0 || material.displacementCenter > 1 ||
            settings.displacementTextures.empty() || !settings.displacementTextures[materialIndex] ||
            settings.texturePublicationRevision == 0) {
            log = "VSM displacement requires finite parameters and a published height texture generation";
            return makeError(Error::InvalidArgument);
        }
        auto lease = (*registry)->sampledImage(*settings.displacementTextures[materialIndex]);
        if (!lease) { return std::unexpected(lease.error()); }
        resources.leases.push_back(*lease);
        if (textureDescriptors.size() <= size_t(texture.textureIndex)) {
            textureDescriptors.resize(size_t(texture.textureIndex) + 1, UINT32_MAX);
        }
        textureDescriptors[texture.textureIndex] = lease->shaderIndex();
        resources.displacedGeometry = true;
    }
    if (resources.displacedGeometry) {
        auto data = buildTessellationData(scene.materials(), textureDescriptors);
        auto tessellationIndex = upload(device, **registry, *owner, resources.leases,
            data.data(), data.size() * sizeof(uint32_t), sizeof(uint32_t));
        if (!tessellationIndex) { return std::unexpected(tessellationIndex.error()); }
        parameters.tessellation = *tessellationIndex;
        parameters.tessellationFactor = settings.tessellationFactor;
        parameters.subdivisionDepth = settings.subdivisionDepth;
        patchTriangleCount = static_cast<uint32_t>(makeTessellationPattern(
            {settings.tessellationFactor, settings.tessellationFactor, settings.tessellationFactor}).triangles.size()) *
            (1u << (settings.subdivisionDepth * 2));
    }
    if (stream) {
        if (!stream->valid()) { return makeError(Error::InvalidArgument); }
        // The coarse projection must cover every terminal branch. Lazy loading
        // cannot make an absent caster page equivalent to empty shadow depth.
        if (!stream->owner->sceneReady()) {
            log = "VSM requires completed initial terminal-page load (initialLoad=true)";
            return makeError(Error::InvalidArgument);
        }
        const auto& asset = *stream->asset;
        if (stream->runtimeRenderNodeIndices.size() != asset.instances().size()) {
            log = "VSM requires the published runtime scene instance mapping";
            return makeError(Error::InvalidArgument);
        }
        if (asset.maxClusterTriangles() > kStreamTrianglesPerCluster) {
            log = "VSM StreamAsset exceeds the supported 128 triangles per cluster";
            return makeError(Error::Unsupported);
        }
        owner->stream = stream->owner;
        std::vector<std::array<uint32_t, 4>> candidates;
        for (uint32_t instanceIndex = 0; instanceIndex < asset.instances().size(); ++instanceIndex) {
            if (!stream->publishedRegionInstances.empty() && !stream->publishedRegionInstances[instanceIndex]) {
                continue;
            }
            const auto& instance = asset.instances()[instanceIndex];
            const uint32_t nodeIndex = stream->runtimeRenderNodeIndices[instanceIndex];
            const auto& node = scene.renderNodes()[nodeIndex];
            const scene::RenderMaterial defaultMaterial;
            const auto& material = scene.materials().empty() ? defaultMaterial : scene.materials()[std::max(node.materialIndex, 0)];
            const float displacement = resources.displacedGeometry ? displacementBound(material) : 0.0f;
            const uint32_t firstTriangle = static_cast<uint32_t>(candidates.size()) * kStreamTrianglesPerCluster;
            if (node.visible) {
                ContentHash transform;
                transform.append(node.worldMatrix.a, sizeof(node.worldMatrix.a));
                ContentHash hash;
                hash.append(instance.primitiveIndex);
                hash.append(node.worldMatrix.a, sizeof(node.worldMatrix.a));
                const auto& keys = (*materialKeys)[static_cast<size_t>(std::max(node.materialIndex, 0))];
                hash.append(keys.material);
                const auto& localBounds = asset.primitives()[instance.primitiveIndex].bounds;
                resources.states.push_back({.id = nodeIndex,
                    .bounds = transformedBounds(localBounds, node.worldMatrix),
                    .contentKey = hash.value,
                    .transformKey = transform.value,
                    .materialKey = keys.material,
                    .textureKey = keys.texture, .tessellationKey = keys.tessellation,
                    .displacementMaterialKey = keys.displacementMaterial,
                    .triangleFirst = firstTriangle});
                expandDisplacedBounds(resources.states.back().bounds, displacement);
                resources.bounds.include(resources.states.back().bounds);
                if (settings.hierarchyHZB && !resources.displacedGeometry) {
                    hierarchyStreamBounds.push_back(streamedCasterHierarchyBounds(resources.states.back().bounds,
                        localBounds, node.worldMatrix));
                }
            }
            // Each replacement cut is complete, including branches that
            // terminate early. The base path retains every terminal cluster.
            std::vector<std::array<uint32_t, 2>> terminalClusters;
            const ShadowCasterSelection* selection = selections.empty() ? nullptr : &selections[nodeIndex];
            if (!selection || !selection->replaceStream) {
                for (uint32_t groupIndex : asset.primitiveTerminalGroups(instance.primitiveIndex)) {
                    for (uint32_t cluster = 0; cluster < asset.groups()[groupIndex].clusterCount; ++cluster) {
                        terminalClusters.push_back({groupIndex, cluster});
                    }
                }
            }
            const auto& selected = selection && selection->replaceStream ? selection->streamClusters : terminalClusters;
            for (const auto& cluster : selected) {
                const auto& group = asset.groups()[cluster[0]];
                constexpr uint32_t requiredAttributes = scene::kMeshletStreamPayloadAttributeNormal |
                    scene::kMeshletStreamPayloadAttributeTexcoord0;
                if (displacement != 0 && (asset.pages()[group.pageIndex].attributeFlags & requiredAttributes) != requiredAttributes) {
                    log = "VSM displacement requires terminal pages with NORMAL and TEXCOORD_0 streams";
                    return makeError(Error::Unsupported);
                }
                const uint32_t expandedCount = displacement != 0 ? patchTriangleCount : 1;
                if (uint64_t(candidates.size()) + expandedCount > UINT32_MAX / kStreamTrianglesPerCluster) {
                    log = "VSM caster triangle capacity exceeds the GPU index range";
                    return makeError(Error::Unsupported);
                }
                for (uint32_t patchTriangle = 0; patchTriangle < expandedCount; ++patchTriangle) {
                    candidates.push_back({instanceIndex, group.pageIndex, cluster[1], patchTriangle});
                }
            }
            if (node.visible) {
                resources.states.back().triangleCount = static_cast<uint32_t>(candidates.size()) * kStreamTrianglesPerCluster - firstTriangle;
            }
        }
        if (auto result = stream->owner->cmdPrepareShadowGeometry(commands); !result) {
            return std::unexpected(result.error());
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
        resources.streamed = true;
        resources.dependencyKey = stream->geometryRevision;
        resources.geometryRevision = stream->geometryRevision;
        std::ranges::sort(resources.states, {}, &VirtualShadowMapCasterState::id);
    } else {
        std::vector<VirtualShadowMapPrototypeTriangle> triangles;
        std::vector<std::array<float, 4>> normals;
        std::vector<std::array<uint32_t, 2>> candidates;
        std::vector<scene::Bounds> clusterBounds;
        if (auto result = gatherResidentTriangles(scene, *materialKeys, triangles, normals, candidates,
            clusterBounds, patchTriangleCount, resources, log, selections); !result) {
            return std::unexpected(result.error());
        }
        auto triangleIndex = upload(device, **registry, *owner, resources.leases,
            triangles.data(), triangles.size() * sizeof(triangles[0]), sizeof(triangles[0]));
        if (!triangleIndex) { return std::unexpected(triangleIndex.error()); }
        parameters.triangles = *triangleIndex;
        parameters.triangleCapacity = static_cast<uint32_t>(triangles.size());
        if (resources.displacedGeometry) {
            auto normalIndex = upload(device, **registry, *owner, resources.leases,
                normals.data(), normals.size() * sizeof(normals[0]), sizeof(normals[0]));
            if (!normalIndex) { return std::unexpected(normalIndex.error()); }
            parameters.normals = *normalIndex;
            auto candidateIndex = upload(device, **registry, *owner, resources.leases,
                candidates.data(), candidates.size() * sizeof(candidates[0]), sizeof(candidates[0]));
            if (!candidateIndex) { return std::unexpected(candidateIndex.error()); }
            parameters.candidates = *candidateIndex;
            parameters.triangleCapacity = static_cast<uint32_t>(candidates.size());
        }
        std::vector<CasterClusterBounds> packedBounds;
        packedBounds.reserve(clusterBounds.size());
        for (const auto& bounds : clusterBounds) {
            packedBounds.push_back({{bounds.min.x, bounds.min.y, bounds.min.z, 0},
                {bounds.max.x, bounds.max.y, bounds.max.z, 0}});
        }
        auto boundsIndex = upload(device, **registry, *owner, resources.leases,
            packedBounds.data(), packedBounds.size() * sizeof(packedBounds[0]), sizeof(packedBounds[0]));
        if (!boundsIndex) { return std::unexpected(boundsIndex.error()); }
        parameters.residentClusterBounds = *boundsIndex;
    }
    ContentHash tessellation;
    // Match the existing dependency key: textured publications include the
    // tessellation mode even when no material currently displaces geometry.
    const bool tessellationDependency = settings.tessellationEnabled &&
        (settings.texturePublicationRevision != 0 || resources.displacedGeometry);
    tessellation.append(tessellationDependency);
    if (resources.displacedGeometry) {
        tessellation.append(settings.tessellationFactor);
        tessellation.append(settings.subdivisionDepth);
    }
    resources.tessellationKey = scoped ? 0 : tessellation.value;
    if (!scoped && (settings.texturePublicationRevision != 0 || resources.displacedGeometry)) {
        ContentHash dependency;
        dependency.append(resources.dependencyKey);
        dependency.append(settings.texturePublicationRevision);
        dependency.append(settings.tessellationEnabled);
        if (resources.displacedGeometry) {
            dependency.append(settings.tessellationFactor);
            dependency.append(settings.subdivisionDepth);
        }
        resources.dependencyKey = dependency.value;
    }
    if ((settings.instanceBuild || settings.hierarchyHZB) && !resources.states.empty()) {
        struct InstanceRange {
            std::array<float, 3> minimum;
            uint32_t first;
            std::array<float, 3> maximum;
            uint32_t count;
        };
        static_assert(sizeof(InstanceRange) == 32);
        std::vector<InstanceRange> instances;
        instances.reserve(resources.states.size());
        for (const auto& state : resources.states) {
            const auto& b = state.bounds;
            instances.push_back({{b.valid ? b.min.x : 1.f, b.min.y, b.min.z}, state.triangleFirst,
                {b.valid ? b.max.x : 0.f, b.max.y, b.max.z}, state.triangleCount});
        }
        auto index = upload(device, **registry, *owner, resources.leases, instances.data(),
            instances.size() * sizeof(InstanceRange), sizeof(InstanceRange));
        if (!index) { return std::unexpected(index.error()); }
        resources.instanceTable = *index;
    }
    if (settings.hierarchyHZB && !resources.displacedGeometry) {
        const auto nodes = buildCasterHierarchy(resources.states, hierarchyStreamBounds);
        if (!nodes.empty()) {
            auto index = upload(device, **registry, *owner, resources.leases,
                nodes.data(), nodes.size() * sizeof(nodes[0]), sizeof(nodes[0]));
            if (!index) { return std::unexpected(index.error()); }
            resources.hierarchyNodes = *index;
            resources.hierarchyRootCount = uint32_t(nodes.size()) / kVSMHierarchyHZBTreeStride;
        }
    }
    owner->parameters = parameters;
    auto parameterIndex = upload(device, **registry, *owner, resources.leases,
        &parameters, sizeof(parameters), sizeof(parameters));
    if (!parameterIndex) { return std::unexpected(parameterIndex.error()); }
    resources.parameters = owner->buffers.back().get();
    resources.triangleCapacity = parameters.triangleCapacity;
    resources.lodMetadata = lodMetadata ? std::move(lodMetadata) : makeLODMetadataCache(scene, stream);
    if (auto result = commands.retainResource(owner); !result) { return std::unexpected(result.error()); }
    for (const auto& lease : resources.leases) {
        if (auto result = (*registry)->retain(commands, lease); !result) { return std::unexpected(result.error()); }
    }
    return resources;
}

Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapCasters(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, std::string& log,
    const VirtualShadowMapCasterSettings& settings)
{
    return prepareCasterGeometry(device, commands, scene, stream, log, settings);
}

static Result<VirtualShadowMapCasterResources> selectShadowLOD(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, const VirtualShadowMapCasterResources& base,
    const VirtualShadowMapCasterSettings& settings,
    std::span<const VirtualShadowMapLODProjection> projections, float maxErrorTexels, std::string& log,
    float maxWorldError, const VirtualShadowMapCasterResources* previous)
{
    if (!std::isfinite(maxErrorTexels) || maxErrorTexels < 0 || !std::isfinite(maxWorldError) || maxWorldError < 0) {
        log = "VSM shadow LOD errors must be finite and nonnegative";
        return makeError(Error::InvalidArgument);
    }
    if (stream) {
        // Register this recording before requests can schedule any eviction;
        // the selected pool ranges remain alive through GPU completion.
        if (auto prepared = stream->owner->cmdPrepareShadowGeometry(commands); !prepared) {
            return std::unexpected(prepared.error());
        }
    }
    // Base preparation owns the scene/stream validation and metadata identity.
    // Allocate primitive slots only when this snapshot first consumes LOD.
    if (base.lodMetadata->primitives.empty()) {
        base.lodMetadata->primitives.resize(stream ? stream->asset->primitives().size() : scene.renderPrimitives().size());
    }
    std::vector<ShadowCasterSelection> selections(scene.renderNodes().size());
    VirtualShadowMapShadowLODDiagnostics diagnostics;
    std::vector<uint32_t> pages;
    bool replacement = false;
    const auto addSelected = [&](uint32_t node, uint32_t group, uint32_t cluster, uint32_t level) {
        diagnostics.selectedClusterIDs.push_back({node, group, cluster});
        ++diagnostics.selectedClusters;
        diagnostics.selectedLevels.resize(std::max<size_t>(diagnostics.selectedLevels.size(), size_t(level) + 1));
        ++diagnostics.selectedLevels[level];
    };
    const auto select = [&](const VirtualShadowMapLODHierarchy& hierarchy,
        const std::vector<uint8_t>& drawable, const auto& node) {
        GPUSceneGPUInstanceRecord instance;
        std::copy_n(node.worldMatrix.a, 16, instance.worldMatrix.begin());
        instance.identity[3] = GPUSceneGPUInstanceVisible;
        auto selected = selectVirtualShadowMapLOD(hierarchy, drawable, instance, projections, maxErrorTexels, maxWorldError);
        diagnostics.metricGroups += selected.metricGroups;
        diagnostics.hierarchyNodes += selected.hierarchyNodes;
        diagnostics.prunedGroups += selected.prunedGroups;
        diagnostics.topologyGroups += selected.topologyGroups;
        return std::move(selected.cut);
    };
    if (stream) {
        const auto& asset = *stream->asset;
        for (uint32_t instanceIndex = 0; instanceIndex < asset.instances().size(); ++instanceIndex) {
            if (!stream->publishedRegionInstances.empty() && !stream->publishedRegionInstances[instanceIndex]) {
                continue;
            }
            const auto& sourceInstance = asset.instances()[instanceIndex];
            const auto& primitive = asset.primitives()[sourceInstance.primitiveIndex];
            const uint32_t nodeIndex = stream->runtimeRenderNodeIndices[instanceIndex];
            const auto& node = scene.renderNodes()[nodeIndex];
            if (!node.visible) { continue; }
            auto& selection = selections[nodeIndex];
            const scene::RenderMaterial defaultMaterial;
            const auto& material = scene.materials().empty() ? defaultMaterial : scene.materials()[std::max(node.materialIndex, 0)];
            if (material.alphaMode == "MASK" || displacementBound(material) != 0) {
                ++diagnostics.protectedCasters;
                for (const uint32_t globalGroup : asset.primitiveTerminalGroups(sourceInstance.primitiveIndex)) {
                    const auto& group = asset.groups()[globalGroup];
                    pages.push_back(group.pageIndex);
                    for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                        addSelected(nodeIndex, globalGroup - primitive.groupOffset, cluster, group.lodLevel);
                    }
                }
                continue;
            }
            auto& metadata = base.lodMetadata->primitives[sourceInstance.primitiveIndex];
            if (!metadata) {
                std::vector<MeshletLODGroupRecord> groups;
                std::vector<MeshletLODGroupRange> ranges;
                std::vector<uint32_t> refined;
                for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                    const auto& source = asset.groups()[primitive.groupOffset + local];
                    MeshletLODGroupRecord group;
                    std::copy_n(source.boundsCenterRadius, 4, group.sphere.begin());
                    group.error = source.maxQuadricError;
                    group.level = source.lodLevel;
                    group.flags = source.flags;
                    groups.push_back(group);
                    ranges.push_back({static_cast<uint32_t>(refined.size()), source.clusterCount});
                    for (uint32_t cluster = 0; cluster < source.clusterCount; ++cluster) {
                        const uint32_t child = asset.refinedGroups()[source.clusterRefinedOffset + cluster];
                        refined.push_back(child == UINT32_MAX ? child : child - primitive.groupOffset);
                    }
                }
                auto built = std::make_unique<VirtualShadowMapLODHierarchy>();
                if (!buildVirtualShadowMapLODHierarchy(groups, ranges, refined, *built, log)) {
                    log = "VSM shadow LOD stream metadata: " + log;
                    return makeError(Error::InvalidArgument);
                }
                metadata = std::move(built);
                ++diagnostics.metadataBuilds;
            } else {
                ++diagnostics.metadataCacheHits;
            }
            std::vector<uint8_t> drawable;
            drawable.reserve(primitive.groupCount);
            for (uint32_t group = 0; group < primitive.groupCount; ++group) {
                drawable.push_back(stream->owner->residency().pageResident(asset.groups()[primitive.groupOffset + group].pageIndex) ? 1u : 0u);
            }
            const auto cut = select(*metadata, drawable, node);
            if (!cut.valid) {
                log = "VSM shadow LOD stream cut: " + cut.reason;
                return makeError(Error::InvalidArgument);
            }
            diagnostics.testedGroups += cut.testedGroups;
            diagnostics.fallbackCasters += cut.requestedGroups.empty() ? 0u : 1u;
            for (uint32_t group : cut.requestedGroups) { pages.push_back(asset.groups()[primitive.groupOffset + group].pageIndex); }
            ContentHash hash;
            bool terminalOnly = true;
            // Selector output is sorted in immutable cluster order, matching
            // the original group-major packing and persistent cut identity.
            for (const uint32_t clusterIndex : cut.selectedClusters) {
                const uint32_t group = metadata->clusterGroups[clusterIndex];
                const uint32_t cluster = clusterIndex - metadata->ranges[group].clusterOffset;
                selection.streamClusters.push_back({primitive.groupOffset + group, cluster});
                pages.push_back(asset.groups()[primitive.groupOffset + group].pageIndex);
                addSelected(nodeIndex, group, cluster, metadata->groups[group].level);
                hash.append(group);
                hash.append(cluster);
                terminalOnly &= (metadata->groups[group].flags & kMeshletLODTerminalGroup) != 0;
            }
            // An unrefined complete terminal cut is the original producer ABI.
            selection.replaceStream = !terminalOnly;
            selection.lodKey = terminalOnly ? 0 : hash.value;
            replacement |= selection.replaceStream;
        }
        std::ranges::sort(pages);
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        const auto requests = stream->owner->requestShadowPages(pages);
        diagnostics.requestedPages = requests.requestedPages;
        diagnostics.blockedPages = requests.blockedPages;
    } else {
        for (const auto& state : base.states) {
            const uint32_t nodeIndex = static_cast<uint32_t>(state.id);
            const auto& node = scene.renderNodes()[nodeIndex];
            const auto& primitive = scene.renderPrimitives()[node.renderPrimitiveIndex];
            const scene::RenderMaterial defaultMaterial;
            const auto& material = scene.materials().empty() ? defaultMaterial : scene.materials()[std::max(node.materialIndex, 0)];
            if (material.alphaMode == "MASK" || displacementBound(material) != 0) {
                ++diagnostics.protectedCasters;
                continue;
            }
            if (primitive.meshletLodGroups.empty() || primitive.meshletLodClusters.empty()) {
                ++diagnostics.fallbackCasters;
                continue;
            }
            auto& metadata = base.lodMetadata->primitives[node.renderPrimitiveIndex];
            if (!metadata) {
                std::vector<MeshletLODGroupRecord> groups;
                if (!buildMeshletLodMetadata(primitive, groups, log)) {
                    log = "VSM shadow LOD resident metadata: " + log;
                    return makeError(Error::InvalidArgument);
                }
                std::vector<MeshletLODGroupRange> ranges;
                std::vector<uint32_t> refined;
                for (const auto& group : primitive.meshletLodGroups) { ranges.push_back({group.clusterOffset, group.clusterCount}); }
                for (const auto& cluster : primitive.meshletLodClusters) {
                    refined.push_back(cluster.refinedGroupIndex < 0 ? UINT32_MAX : uint32_t(cluster.refinedGroupIndex));
                }
                auto built = std::make_unique<VirtualShadowMapLODHierarchy>();
                if (!buildVirtualShadowMapLODHierarchy(groups, ranges, refined, *built, log)) {
                    log = "VSM shadow LOD resident hierarchy: " + log;
                    return makeError(Error::InvalidArgument);
                }
                metadata = std::move(built);
                ++diagnostics.metadataBuilds;
            } else {
                ++diagnostics.metadataCacheHits;
            }
            const std::vector<uint8_t> drawable(metadata->groups.size(), 1);
            auto cut = select(*metadata, drawable, node);
            if (!cut.valid) {
                log = "VSM shadow LOD resident cut: " + cut.reason;
                return makeError(Error::InvalidArgument);
            }
            diagnostics.testedGroups += cut.testedGroups;
            auto& selection = selections[nodeIndex];
            ContentHash hash;
            bool finest = true;
            for (const uint32_t clusterIndex : cut.selectedClusters) {
                const auto& cluster = primitive.meshletLodClusters[clusterIndex];
                addSelected(nodeIndex, uint32_t(cluster.lodGroupIndex), cluster.lodGroupChildIndex, cluster.lodLevel);
                finest &= cluster.lodLevel == 0;
                hash.append(clusterIndex);
            }
            // Preserve authored triangle ordering for exact finest/base A/Bs.
            selection.replaceResident = !finest && maxErrorTexels != 0;
            if (selection.replaceResident) {
                selection.residentClusters = std::move(cut.selectedClusters);
            }
            selection.lodKey = selection.replaceResident ? hash.value : 0;
            replacement |= selection.replaceResident;
        }
    }
    // The cache supplies a previous geometry only for the exact same immutable
    // resident base. Ordered node/group/cluster identities uniquely name the cut.
    const bool sameGeometry = replacement && previous && previous->owner != base.owner &&
        diagnostics.selectedClusterIDs == previous->shadowLOD.selectedClusterIDs;
    auto result = sameGeometry ? Result<VirtualShadowMapCasterResources>(*previous) :
        replacement ? prepareCasterGeometry(device, commands, scene, stream, log, settings, selections, base.lodMetadata) :
        Result<VirtualShadowMapCasterResources>(base);
    if (!result) { return std::unexpected(result.error()); }
    if (!replacement || sameGeometry) {
        auto registry = ResourceRegistry::forDevice(device);
        if (!registry) { return std::unexpected(registry.error()); }
        if (auto retained = commands.retainResource(result->owner); !retained) { return std::unexpected(retained.error()); }
        for (const auto& lease : result->leases) {
            if (auto retained = (*registry)->retain(commands, lease); !retained) { return std::unexpected(retained.error()); }
        }
    }
    // These identities and bounds describe the full authored scene, never the
    // selected approximation. Only the contiguous slot ranges follow the cut.
    result->lodMetadata = base.lodMetadata;
    result->bounds = base.bounds;
    result->dependencyKey = base.dependencyKey;
    result->geometryRevision = base.geometryRevision;
    result->texturePublicationRevision = base.texturePublicationRevision;
    result->tessellationKey = base.tessellationKey;
    for (auto& state : result->states) {
        const auto source = std::ranges::lower_bound(base.states, state.id, {}, &VirtualShadowMapCasterState::id);
        const uint32_t first = state.triangleFirst, count = state.triangleCount;
        state = *source;
        state.triangleFirst = first;
        state.triangleCount = count;
        state.lodKey = selections[state.id].lodKey;
    }
    diagnostics.selectedTriangleSlots = result->triangleCapacity;
    result->shadowLOD = std::move(diagnostics);
    return result;
}

Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapShadowLOD(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, const VirtualShadowMapCasterResources& base,
    const VirtualShadowMapCasterSettings& settings,
    std::span<const VirtualShadowMapLODProjection> projections, float maxErrorTexels, std::string& log,
    float maxWorldError)
{
    return selectShadowLOD(device, commands, scene, stream, base, settings, projections, maxErrorTexels, log,
        maxWorldError, nullptr);
}

Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapGPUShadowLOD(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, const VirtualShadowMapCasterResources& base,
    const VirtualShadowMapCasterSettings& settings, std::string& log)
{
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    auto& metadata = base.lodMetadata->primitives;
    if (metadata.empty()) {
        metadata.resize(stream ? stream->asset->primitives().size() : scene.renderPrimitives().size());
    }
    std::vector<uint32_t> streamInstances(scene.renderNodes().size(), UINT32_MAX);
    if (stream) {
        for (uint32_t instance = 0; instance < stream->asset->instances().size(); ++instance) {
            streamInstances[stream->runtimeRenderNodeIndices[instance]] = instance;
        }
    }
    std::vector<ShadowCasterSelection> selections(scene.renderNodes().size());
    std::vector<const VirtualShadowMapLODHierarchy*> hierarchies;
    std::vector<bool> protectedCasters;
    VirtualShadowMapShadowLODDiagnostics diagnostics;
    for (const auto& state : base.states) {
        const auto& node = scene.renderNodes()[state.id];
        const scene::RenderMaterial defaultMaterial;
        const auto& material = scene.materials().empty() ? defaultMaterial : scene.materials()[std::max(node.materialIndex, 0)];
        const bool protectedCaster = material.alphaMode == "MASK" || displacementBound(material) != 0;
        protectedCasters.push_back(protectedCaster);
        if (protectedCaster) {
            ++diagnostics.protectedCasters;
            hierarchies.push_back(nullptr);
            continue;
        }
        const uint32_t primitiveIndex = stream ? stream->asset->instances()[streamInstances[state.id]].primitiveIndex :
            uint32_t(node.renderPrimitiveIndex);
        if (!stream && (scene.renderPrimitives()[primitiveIndex].meshletLodGroups.empty() ||
            scene.renderPrimitives()[primitiveIndex].meshletLodClusters.empty())) {
            ++diagnostics.fallbackCasters;
            hierarchies.push_back(nullptr);
            continue;
        }
        if (!metadata[primitiveIndex]) {
            std::vector<MeshletLODGroupRecord> groups;
            std::vector<MeshletLODGroupRange> ranges;
            std::vector<uint32_t> refined;
            if (stream) {
                const auto& asset = *stream->asset;
                const auto& primitive = asset.primitives()[primitiveIndex];
                for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                    const auto& source = asset.groups()[primitive.groupOffset + local];
                    MeshletLODGroupRecord group;
                    std::copy_n(source.boundsCenterRadius, 4, group.sphere.begin());
                    group.error = source.maxQuadricError; group.level = source.lodLevel; group.flags = source.flags;
                    groups.push_back(group);
                    ranges.push_back({uint32_t(refined.size()), source.clusterCount});
                    for (uint32_t cluster = 0; cluster < source.clusterCount; ++cluster) {
                        const uint32_t child = asset.refinedGroups()[source.clusterRefinedOffset + cluster];
                        refined.push_back(child == UINT32_MAX ? child : child - primitive.groupOffset);
                    }
                }
            } else {
                const auto& primitive = scene.renderPrimitives()[primitiveIndex];
                if (!buildMeshletLodMetadata(primitive, groups, log)) { return makeError(Error::InvalidArgument); }
                for (const auto& group : primitive.meshletLodGroups) { ranges.push_back({group.clusterOffset, group.clusterCount}); }
                for (const auto& cluster : primitive.meshletLodClusters) {
                    refined.push_back(cluster.refinedGroupIndex < 0 ? UINT32_MAX : uint32_t(cluster.refinedGroupIndex));
                }
            }
            auto hierarchy = std::make_unique<VirtualShadowMapLODHierarchy>();
            if (!buildVirtualShadowMapLODHierarchy(groups, ranges, refined, *hierarchy, log)) {
                return makeError(Error::InvalidArgument);
            }
            metadata[primitiveIndex] = std::move(hierarchy);
            ++diagnostics.metadataBuilds;
        } else { ++diagnostics.metadataCacheHits; }
        const auto& hierarchy = *metadata[primitiveIndex];
        hierarchies.push_back(&hierarchy);
        auto& selection = selections[state.id];
        if (stream) {
            selection.replaceStream = true;
            const auto& primitive = stream->asset->primitives()[primitiveIndex];
            for (uint32_t group = 0; group < hierarchy.groups.size(); ++group) {
                for (uint32_t cluster = 0; cluster < hierarchy.ranges[group].clusterCount; ++cluster) {
                    selection.streamClusters.push_back({primitive.groupOffset + group, cluster});
                }
            }
        } else {
            selection.replaceResident = selection.includeBase = true;
            selection.residentClusters.resize(hierarchy.clusterGroups.size());
            for (uint32_t cluster = 0; cluster < selection.residentClusters.size(); ++cluster) {
                selection.residentClusters[cluster] = cluster;
            }
        }
    }
    auto result = prepareCasterGeometry(device, commands, scene, stream, log, settings, selections, base.lodMetadata);
    if (!result) { return std::unexpected(result.error()); }
    result->bounds = base.bounds;
    result->dependencyKey = base.dependencyKey;
    result->geometryRevision = base.geometryRevision;
    result->texturePublicationRevision = base.texturePublicationRevision;
    result->tessellationKey = base.tessellationKey;
    auto owner = std::make_shared<CasterUploads>();
    owner->sourceOwner = result->owner;
    auto gpu = std::make_shared<VirtualShadowMapGPUHierarchy>();
    gpu->owner = owner;
    auto& context = gpu->context;
    context.pageTable = context.pageRequests = context.groupStates = context.selected = context.previous = UINT32_MAX;
    context.changed = context.statistics = UINT32_MAX;
    const uint32_t triangleClusterShift = stream ? kVSMLODStreamClusterShift : 0u;
    context.triangleClusterShift = triangleClusterShift;
    std::vector<VirtualShadowMapLODGroupGPU> groups;
    std::vector<VirtualShadowMapLODNodeGPU> nodes;
    std::vector<VirtualShadowMapLODClusterGPU> clusters;
    std::vector<VirtualShadowMapLODInstanceGPU> instances;
    std::vector<uint32_t> parents, metrics;
    // Every streamed candidate owns 128 padded slots, including protected
    // casters and unused slots belonging to hidden instances.
    assert(!stream || result->triangleCapacity % kStreamTrianglesPerCluster == 0);
    std::vector<uint32_t> triangleClusters(result->triangleCapacity >> triangleClusterShift, UINT32_MAX);
    ContentHash layout;
    layout.append(triangleClusterShift);
    for (uint32_t instanceIndex = 0; instanceIndex < result->states.size(); ++instanceIndex) {
        auto& state = result->states[instanceIndex];
        assert(state.id == base.states[instanceIndex].id);
        const uint32_t triangleFirst = state.triangleFirst, triangleCount = state.triangleCount;
        state = base.states[instanceIndex];
        state.triangleFirst = triangleFirst; state.triangleCount = triangleCount;
        const auto& node = scene.renderNodes()[state.id];
        VirtualShadowMapLODInstanceGPU instance{};
        std::copy_n(node.worldMatrix.a, 4, instance.world0.begin());
        std::copy_n(node.worldMatrix.a + 4, 4, instance.world1.begin());
        std::copy_n(node.worldMatrix.a + 8, 4, instance.world2.begin());
        std::copy_n(node.worldMatrix.a + 12, 4, instance.world3.begin());
        instance.boundsMin = {state.bounds.min.x, state.bounds.min.y, state.bounds.min.z, 0};
        instance.boundsMax = {state.bounds.max.x, state.bounds.max.y, state.bounds.max.z, 0};
        instance.groupOffset = uint32_t(groups.size()); instance.nodeOffset = uint32_t(nodes.size());
        instance.metricOffset = uint32_t(metrics.size()); instance.clusterOffset = uint32_t(clusters.size());
        instance.stateOffset = context.groupStateCount;
        instance.baseTriangleCount = base.states[instanceIndex].triangleCount;
        instance.flags = (stream ? 0u : kVirtualShadowMapLODInstanceResident) |
            (protectedCasters[instanceIndex] ? kVirtualShadowMapLODInstanceProtected : 0u);
        uint32_t nextTriangle = state.triangleFirst;
        const auto addCluster = [&](uint32_t group, uint32_t refined, uint32_t count, uint32_t level, uint32_t flags) {
            const uint32_t id = uint32_t(clusters.size());
            clusters.push_back({group, refined, nextTriangle, count, level, instanceIndex, flags, 0});
            assert(!stream || (nextTriangle % kStreamTrianglesPerCluster == 0 && count % kStreamTrianglesPerCluster == 0));
            std::fill_n(triangleClusters.begin() + (nextTriangle >> triangleClusterShift), count >> triangleClusterShift, id);
            nextTriangle += count;
        };
        const auto* hierarchy = hierarchies[instanceIndex];
        if (!stream || !hierarchy) {
            const uint32_t baseCount = hierarchy ? base.states[instanceIndex].triangleCount : state.triangleCount;
            addCluster(UINT32_MAX, UINT32_MAX, baseCount, 0, kVirtualShadowMapLODClusterBase);
            instance.baseClusterCount = 1;
        }
        if (hierarchy) {
            instance.groupCount = uint32_t(hierarchy->groups.size());
            instance.nodeCount = uint32_t(hierarchy->bvh.size());
            instance.metricGroupCount = uint32_t(hierarchy->metricGroupIDs.size());
            context.groupStateCount += instance.groupCount;
            for (uint32_t groupIndex = 0; groupIndex < hierarchy->groups.size(); ++groupIndex) {
                const auto& group = hierarchy->groups[groupIndex];
                const auto& range = hierarchy->ranges[groupIndex];
                uint32_t page = UINT32_MAX;
                if (stream) {
                    const auto& primitive = stream->asset->primitives()[stream->asset->instances()[streamInstances[state.id]].primitiveIndex];
                    page = stream->asset->groups()[primitive.groupOffset + groupIndex].pageIndex;
                }
                groups.push_back({group.sphere, group.error, group.level, group.flags, uint32_t(parents.size()),
                    uint32_t(hierarchy->parents[groupIndex].size()), instance.clusterOffset + instance.baseClusterCount + range.clusterOffset,
                    range.clusterCount, page});
                parents.insert(parents.end(), hierarchy->parents[groupIndex].begin(), hierarchy->parents[groupIndex].end());
            }
            for (const auto& source : hierarchy->bvh) {
                nodes.push_back({source.sphere, source.maxError, source.escapeIndex, source.groupOffset, source.groupCount});
            }
            metrics.insert(metrics.end(), hierarchy->metricGroupIDs.begin(), hierarchy->metricGroupIDs.end());
            for (uint32_t cluster = 0; cluster < hierarchy->clusterGroups.size(); ++cluster) {
                const uint32_t group = hierarchy->clusterGroups[cluster];
                const uint32_t count = stream ? kStreamTrianglesPerCluster :
                    scene.renderPrimitives()[node.renderPrimitiveIndex].meshletLodClusters[cluster].triangleCount;
                addCluster(group, hierarchy->refinedGroups[cluster], count, hierarchy->groups[group].level, 0);
            }
        }
        assert(nextTriangle == state.triangleFirst + state.triangleCount);
        instance.clusterCount = uint32_t(clusters.size()) - instance.clusterOffset;
        layout.append(state.id); layout.append(node.renderPrimitiveIndex);
        layout.append(&instance.groupOffset, 12 * sizeof(uint32_t));
        instances.push_back(instance);
    }
    context.instanceCount = uint32_t(instances.size()); context.clusterCount = uint32_t(clusters.size());
    auto uploadRecords = [&](const auto& records, uint32_t& handle, size_t bufferIndex) -> Result<> {
        using Record = typename std::decay_t<decltype(records)>::value_type;
        layout.append(records.data(), records.size() * sizeof(Record));
        auto uploaded = upload(device, **registry, *owner, gpu->leases, records.data(), records.size() * sizeof(Record), sizeof(Record));
        if (!uploaded) { return std::unexpected(uploaded.error()); }
        handle = *uploaded;
        gpu->metadataBuffers[bufferIndex] = owner->buffers.back().get();
        return {};
    };
    if (auto r = uploadRecords(groups, context.groups, 0); !r) { return std::unexpected(r.error()); }
    if (auto r = uploadRecords(parents, context.parents, 1); !r) { return std::unexpected(r.error()); }
    if (auto r = uploadRecords(nodes, context.nodes, 2); !r) { return std::unexpected(r.error()); }
    if (auto r = uploadRecords(metrics, context.metrics, 3); !r) { return std::unexpected(r.error()); }
    if (auto r = uploadRecords(clusters, context.clusters, 5); !r) { return std::unexpected(r.error()); }
    if (auto r = uploadRecords(triangleClusters, context.triangleClusters, 6); !r) { return std::unexpected(r.error()); }
    // Transforms and bounds affect selection, but do not rename previous slots.
    auto instanceHandle = upload(device, **registry, *owner, gpu->leases, instances.data(),
        instances.size() * sizeof(instances[0]), sizeof(instances[0]));
    if (!instanceHandle) { return std::unexpected(instanceHandle.error()); }
    context.instances = *instanceHandle;
    gpu->metadataBuffers[4] = owner->buffers.back().get();
    gpu->layoutKey = layout.value;
    if (stream) {
        gpu->stream = stream->owner;
        gpu->pageCount = uint32_t(stream->asset->pages().size());
        context.requestsEnabled = stream->owner->shadowPageRequestsEnabled();
        auto pageTable = (*registry)->storageBuffer(*stream->pageTableBuffer);
        if (!pageTable) { return std::unexpected(pageTable.error()); }
        context.pageTable = pageTable->shaderIndex(); gpu->leases.push_back(*pageTable);
        auto requests = device.createBuffer({.size = std::max<uint64_t>(gpu->pageCount, 1u) * sizeof(uint32_t),
            .structureStride = sizeof(uint32_t), .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload});
        if (!requests) { return std::unexpected(requests.error()); }
        void* destination = (*requests)->map();
        if (!destination) { log += "Unable to map VSM LOD page requests.\n"; return makeError(Error::Failure); }
        std::memset(destination, 0, size_t((*requests)->desc().size));
        (*requests)->flush(); (*requests)->unmap();
        gpu->pageRequests = requests->get();
        auto lease = (*registry)->storageBuffer(**requests);
        if (!lease) { return std::unexpected(lease.error()); }
        context.pageRequests = lease->shaderIndex(); gpu->leases.push_back(*lease);
        owner->buffers.push_back(std::move(*requests));
        const BufferBarrierDesc ready{.buffer = gpu->pageRequests,
            .before = {PipelineStageBits::Host, AccessBits::HostWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
        if (auto r = commands.synchronize({.buffers = {&ready, 1}}); !r) { return std::unexpected(r.error()); }
    }
    if (auto r = commands.retainResource(owner); !r) { return std::unexpected(r.error()); }
    for (const auto& lease : gpu->leases) {
        if (auto r = (*registry)->retain(commands, lease); !r) { return std::unexpected(r.error()); }
    }
    result->leases.insert(result->leases.end(), gpu->leases.begin(), gpu->leases.end());
    result->gpuLOD = std::move(gpu);
    result->shadowLOD = std::move(diagnostics);
    commands.hostWriteBarrier();
    return result;
}

Result<std::shared_ptr<const VirtualShadowMapCasterLayerBindings>> VirtualShadowMapCasterLayerCache::prepare(
    ResourceRegistry& registry, const VirtualShadowMapCasterResources& source,
    std::span<const uint64_t> dynamicCasterIds, bool merge, const AcquireBuffer& acquireBuffer)
{
    std::vector<std::array<uint32_t, 2>> ranges;
    if (!dynamicCasterIds.empty()) {
        ranges.reserve(dynamicCasterIds.size());
        for (const auto& caster : source.states) {
            if (caster.triangleCount != 0 && std::ranges::binary_search(dynamicCasterIds, caster.id)) {
                ranges.push_back({caster.triangleFirst, caster.triangleCount});
            }
        }
    }
    const bool reuseMask = bindings_ && triangleCapacity_ == source.triangleCapacity && ranges_ == ranges;
    if (reuseMask && bindings_->sourceOwner == source.owner) { return bindings_; }

    auto bindings = std::make_shared<VirtualShadowMapCasterLayerBindings>();
    bindings->sourceOwner = source.owner;
    const auto uploadLayer = [&](size_t index, const void* data, uint64_t bytes, uint32_t stride) -> Result<> {
        auto buffer = acquireBuffer(index, {.size = std::max<uint64_t>(bytes, stride), .structureStride = stride,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
        if (!buffer) { return std::unexpected(buffer.error()); }
        if (bytes != 0) {
            auto* destination = (*buffer)->map();
            if (!destination) { return makeError(Error::Failure); }
            std::memcpy(destination, data, static_cast<size_t>(bytes));
            (*buffer)->flush({0, bytes});
            (*buffer)->unmap();
        }
        auto lease = registry.storageBuffer(**buffer);
        if (!lease) { return std::unexpected(lease.error()); }
        bindings->buffers[index] = std::move(*buffer);
        bindings->leases[index] = std::move(*lease);
        return {};
    };
    if (reuseMask) {
        bindings->buffers[0] = bindings_->buffers[0];
        bindings->leases[0] = bindings_->leases[0];
    } else {
        std::vector<uint32_t> dynamicMask((uint64_t(source.triangleCapacity) + 31u) / 32u, 0u);
        for (const auto& range : ranges) {
            uint32_t first = range[0];
            const uint32_t end = first + range[1];
            while (first < end) {
                const uint32_t bit = first & 31u;
                const uint32_t count = std::min(32u - bit, end - first);
                dynamicMask[first / 32u] |= (UINT32_MAX >> (32u - count)) << bit;
                first += count;
            }
        }
        if (auto r = uploadLayer(0, dynamicMask.data(), dynamicMask.size() * sizeof(uint32_t), sizeof(uint32_t)); !r) {
            return std::unexpected(r.error());
        }
    }
    auto parameters = std::static_pointer_cast<CasterUploads>(source.owner)->parameters;
    parameters.casterLayerMask = bindings->leases[0].shaderIndex();
    parameters.casterLayer = merge ? 2u : 1u;
    if (auto r = uploadLayer(1, &parameters, sizeof(parameters), sizeof(GeometryParameters)); !r) {
        return std::unexpected(r.error());
    }
    if (!merge) {
        parameters.casterLayer = 2u;
        if (auto r = uploadLayer(2, &parameters, sizeof(parameters), sizeof(GeometryParameters)); !r) {
            return std::unexpected(r.error());
        }
    }
    triangleCapacity_ = source.triangleCapacity;
    ranges_ = std::move(ranges);
    bindings_ = std::move(bindings);
    return bindings_;
}

struct VirtualShadowMapCasterCache::Resident
{
    struct Key
    {
        const Device* device = nullptr;
        uint64_t sceneIdentity = 0;
        uint64_t structuralRevision = 0;
        uint64_t geometryTransformRevision = 0;
        uint64_t visibilityRevision = 0;
        uint64_t materialRevision = 0;
        uint64_t texturePublicationRevision = 0;
        bool tessellationEnabled = false;
        bool instanceBuild = false;
        bool hierarchyHZB = false;
        uint32_t tessellationFactor = 0;
        uint32_t subdivisionDepth = 0;

        bool operator==(const Key&) const = default;
    };

    Key key;
    std::vector<TextureView*> displacementTextures;
    std::vector<VirtualShadowMapMaterialTextureRevisions> materialTextureRevisions;
    VirtualShadowMapCasterResources resources;
    struct Cut
    {
        std::vector<VirtualShadowMapLODProjection> projections;
        float maxErrorTexels, maxWorldError;
        VirtualShadowMapCasterResources geometry;
    };
    std::optional<Cut> cut;
    struct GPUHierarchy
    {
        std::shared_ptr<void> baseOwner;
        VirtualShadowMapCasterResources geometry;
    };
    std::optional<GPUHierarchy> gpuHierarchy;
};

Result<VirtualShadowMapCasterResources> VirtualShadowMapCasterCache::prepareGPUShadowLOD(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, const VirtualShadowMapCasterResources& base,
    const VirtualShadowMapCasterSettings& settings, std::string& log)
{
    if (stream) { return prepareVirtualShadowMapGPUShadowLOD(device, commands, scene, stream, base, settings, log); }
    // prepare() owns the complete resident settings/texture-publication key.
    // GPU metadata can be reused only with that exact immutable base upload.
    auto& cached = resident_->gpuHierarchy;
    if (cached && cached->baseOwner == base.owner) {
        const auto& geometry = cached->geometry;
        auto registry = ResourceRegistry::forDevice(device);
        if (!registry) { return std::unexpected(registry.error()); }
        if (auto r = commands.retainResource(geometry.owner); !r) { return std::unexpected(r.error()); }
        if (auto r = commands.retainResource(geometry.gpuLOD->owner); !r) { return std::unexpected(r.error()); }
        for (const auto& lease : geometry.leases) {
            if (auto r = (*registry)->retain(commands, lease); !r) { return std::unexpected(r.error()); }
        }
        auto result = geometry;
        result.shadowLOD.metadataCacheHits += result.shadowLOD.metadataBuilds;
        result.shadowLOD.metadataBuilds = 0;
        return result;
    }
    auto geometry = prepareVirtualShadowMapGPUShadowLOD(device, commands, scene, nullptr, base, settings, log);
    if (!geometry) { return std::unexpected(geometry.error()); }
    cached.emplace(Resident::GPUHierarchy{base.owner, *geometry});
    return geometry;
}

Result<VirtualShadowMapCasterResources> VirtualShadowMapCasterCache::prepareShadowLOD(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, const VirtualShadowMapCasterResources& base,
    const VirtualShadowMapCasterSettings& settings, std::span<const VirtualShadowMapLODProjection> projections,
    float maxErrorTexels, std::string& log, float maxWorldError)
{
    // Stream cuts must re-evaluate residency and submit page requests on every recording.
    if (stream) {
        return prepareVirtualShadowMapShadowLOD(device, commands, scene, stream, base, settings, projections,
            maxErrorTexels, log, maxWorldError);
    }
    auto& cut = resident_->cut;
    if (cut && cut->maxErrorTexels == maxErrorTexels && cut->maxWorldError == maxWorldError &&
        std::ranges::equal(cut->projections, projections)) {
        const auto& geometry = cut->geometry;
        auto registry = ResourceRegistry::forDevice(device);
        if (!registry) { return std::unexpected(registry.error()); }
        if (auto retained = commands.retainResource(geometry.owner); !retained) { return std::unexpected(retained.error()); }
        for (const auto& lease : geometry.leases) {
            if (auto retained = (*registry)->retain(commands, lease); !retained) { return std::unexpected(retained.error()); }
        }
        return geometry;
    }
    const auto* previous = cut ? &cut->geometry : nullptr;
    auto selected = selectShadowLOD(device, commands, scene, nullptr, base, settings, projections,
        maxErrorTexels, log, maxWorldError, previous);
    if (!selected) { return std::unexpected(selected.error()); }
    // The immutable host uploads are ready before submission. Replacing this entry
    // (or cancelling its first recording) does not affect per-command retention.
    cut.emplace(Resident::Cut{{projections.begin(), projections.end()}, maxErrorTexels, maxWorldError, *selected});
    auto& diagnostics = cut->geometry.shadowLOD;
    diagnostics.reusedResidentCut = true;
    diagnostics.testedGroups = diagnostics.metricGroups = diagnostics.hierarchyNodes = diagnostics.prunedGroups = 0;
    diagnostics.topologyGroups = diagnostics.metadataBuilds = diagnostics.metadataCacheHits = 0;
    return selected;
}

Result<VirtualShadowMapCasterResources> VirtualShadowMapCasterCache::prepare(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, std::string& log,
    const VirtualShadowMapCasterSettings& settings)
{
    // SceneLoader finalizes deferred meshlet jobs before takeResult publishes
    // the scene. Primitive geometry then stays read-only. Scene load and
    // composition replace resourceIdentity; graph edits change structuralRevision.
    // Transform/material changes do not alter these object-space hierarchies.
    const uint64_t geometryRevision = stream ? stream->geometryRevision : 0;
    if (!lodMetadata_ || lodMetadata_->sceneIdentity != scene.resourceIdentity() ||
        lodMetadata_->structuralRevision != scene.sceneGraph().structuralRevision() ||
        lodMetadata_->geometryRevision != geometryRevision || lodMetadata_->streamed != (stream != nullptr)) {
        lodMetadata_ = makeLODMetadataCache(scene, stream);
    }
    if (stream) {
        resident_.reset();
        auto resources = prepareCasterGeometry(device, commands, scene, stream, log, settings, {}, lodMetadata_);
        return resources;
    }

    // Scene exposes resident geometry/materials read-only. Its mutation APIs
    // publish these revisions; camera/light-only edits leave geometry unchanged.
    const Resident::Key key{
        .device = &device,
        .sceneIdentity = scene.resourceIdentity(),
        .structuralRevision = scene.sceneGraph().structuralRevision(),
        .geometryTransformRevision = scene.geometryTransformRevision(),
        .visibilityRevision = scene.visibilityRevision(),
        .materialRevision = scene.materialRevision(),
        .texturePublicationRevision = settings.materialTextureRevisions.empty() ? settings.texturePublicationRevision :
            uint64_t(settings.texturePublicationRevision != 0),
        .tessellationEnabled = settings.tessellationEnabled,
        .instanceBuild = settings.instanceBuild,
        .hierarchyHZB = settings.hierarchyHZB,
        .tessellationFactor = settings.tessellationFactor,
        .subdivisionDepth = settings.subdivisionDepth};
    if (resident_ && resident_->key == key &&
        std::ranges::equal(resident_->materialTextureRevisions, settings.materialTextureRevisions) &&
        std::ranges::equal(resident_->displacementTextures, settings.displacementTextures)) {
        auto registry = ResourceRegistry::forDevice(device);
        if (!registry) { return std::unexpected(registry.error()); }
        const auto& resources = resident_->resources;
        if (auto result = commands.retainResource(resources.owner); !result) { return std::unexpected(result.error()); }
        for (const auto& lease : resources.leases) {
            if (auto result = (*registry)->retain(commands, lease); !result) { return std::unexpected(result.error()); }
        }
        return resources;
    }

    auto resources = prepareCasterGeometry(device, commands, scene, nullptr, log, settings, {}, lodMetadata_);
    if (!resources) { return std::unexpected(resources.error()); }
    // Keep the original shadow-specific hashes. A material RGB edit rebuilds
    // this upload snapshot without invalidating otherwise unchanged shadow pages.
    resident_ = std::make_shared<Resident>(Resident{
        .key = key,
        .displacementTextures = {settings.displacementTextures.begin(), settings.displacementTextures.end()},
        .materialTextureRevisions = {settings.materialTextureRevisions.begin(), settings.materialTextureRevisions.end()},
        .resources = *resources});
    return resources;
}

} // namespace metallic::render
