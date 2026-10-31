#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"

#include "Runtime/Render/Shadow/VirtualShadowMapPrototype.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include "Runtime/Render/GPUDrivenTessellation.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <cmath>

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
    std::vector<uint32_t> vertices;
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

Result<std::vector<uint64_t>> gatherMaterialKeys(const scene::Scene& scene, std::string& log)
{
    std::vector<uint64_t> keys;
    const scene::RenderMaterial defaultMaterial;
    try {
        for (size_t i = 0; i < std::max<size_t>(scene.materials().size(), 1); ++i) {
            const auto& material = scene.materials().empty() ? defaultMaterial : scene.materials()[i];
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
                }
            }
            const bool displaced = material.displacementMagnitude != 0 && material.displacementTexture.textureIndex >= 0;
            hash.append(displaced);
            if (displaced) {
                hash.append(material.displacementMagnitude);
                hash.append(material.displacementCenter);
                hash.append(material.displacementTexture.textureIndex);
                hash.append(material.displacementTexture.uvTransform);
            }
            keys.push_back(hash.value);
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
    std::span<const uint64_t> materialKeys, std::vector<VirtualShadowMapPrototypeTriangle>& triangles,
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
        const size_t count = replace ? selection->vertices.size() :
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
        state.materialKey = materialKeys[static_cast<size_t>(std::max(node.materialIndex, 0))];
        state.triangleFirst = static_cast<uint32_t>(resources.displacedGeometry ? candidates.size() : triangles.size());
        ContentHash hash;
        hash.append(node.worldMatrix.a, sizeof(node.worldMatrix.a));
        hash.append(materialKeys[static_cast<size_t>(std::max(node.materialIndex, 0))]);
        hash.append(count);
        for (size_t triangleIndex = 0; triangleIndex < count; triangleIndex += 3) {
            VirtualShadowMapPrototypeTriangle triangle;
            scene::Bounds triangleBounds;
            triangle.materialIndex = static_cast<uint32_t>(std::max(node.materialIndex, 0));
            triangle.reserved[0] = reflected ? 1u : 0u;
            for (uint32_t vertex = 0; vertex < 3; ++vertex) {
                const size_t index = replace ? selection->vertices[triangleIndex + vertex] :
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
                    if (displaced) { hash.append(normals.back()); }
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
        (!settings.displacementTextures.empty() && settings.displacementTextures.size() != scene.materials().size())) {
        log = "VSM caster settings require tessellationFactor 1..8, subdivisionDepth 0..2 and one height slot per material";
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
    resources.owner = owner;
    resources.sceneIdentity = scene.resourceIdentity();
    resources.structuralRevision = scene.sceneGraph().structuralRevision();
    resources.texturePublicationRevision = settings.texturePublicationRevision;
    auto materialKeys = gatherMaterialKeys(scene, log);
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
                hash.append((*materialKeys)[static_cast<size_t>(std::max(node.materialIndex, 0))]);
                resources.states.push_back({.id = nodeIndex,
                    .bounds = transformedBounds(asset.primitives()[instance.primitiveIndex].bounds, node.worldMatrix),
                    .contentKey = hash.value,
                    .transformKey = transform.value,
                    .materialKey = (*materialKeys)[static_cast<size_t>(std::max(node.materialIndex, 0))],
                    .triangleFirst = firstTriangle});
                expandDisplacedBounds(resources.states.back().bounds, displacement);
                resources.bounds.include(resources.states.back().bounds);
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
    resources.tessellationKey = tessellation.value;
    if (settings.texturePublicationRevision != 0 || resources.displacedGeometry) {
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

Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapShadowLOD(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, const VirtualShadowMapCasterResources& base,
    const VirtualShadowMapCasterSettings& settings,
    std::span<const VirtualShadowMapLODProjection> projections, float maxErrorTexels, std::string& log,
    bool hierarchical)
{
    if (!std::isfinite(maxErrorTexels) || maxErrorTexels < 0) {
        log = "VSM shadow LOD error must be finite and nonnegative";
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
        auto selected = selectVirtualShadowMapLOD(hierarchy, drawable, instance, projections, maxErrorTexels, hierarchical);
        diagnostics.metricGroups += selected.metricGroups;
        diagnostics.hierarchyNodes += selected.hierarchyNodes;
        diagnostics.prunedGroups += selected.prunedGroups;
        diagnostics.topologyGroups += selected.topologyGroups;
        return std::move(selected.cut);
    };
    if (stream) {
        const auto& asset = *stream->asset;
        for (uint32_t instanceIndex = 0; instanceIndex < asset.instances().size(); ++instanceIndex) {
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
            const auto cut = select(*metadata, drawable, node);
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
                for (const uint32_t clusterIndex : cut.selectedClusters) {
                    const auto& cluster = primitive.meshletLodClusters[clusterIndex];
                    for (uint32_t vertex = 0; vertex < cluster.triangleCount * 3u; ++vertex) {
                        const uint32_t local = primitive.meshletLodTriangles[cluster.triangleOffset + vertex];
                        selection.vertices.push_back(primitive.meshletLodVertices[cluster.vertexOffset + local]);
                    }
                }
            }
            selection.lodKey = selection.replaceResident ? hash.value : 0;
            replacement |= selection.replaceResident;
        }
    }
    auto result = replacement ? prepareCasterGeometry(device, commands, scene, stream, log, settings, selections, base.lodMetadata) :
        Result<VirtualShadowMapCasterResources>(base);
    if (!result) { return std::unexpected(result.error()); }
    if (!replacement) {
        auto registry = ResourceRegistry::forDevice(device);
        if (!registry) { return std::unexpected(registry.error()); }
        if (auto retained = commands.retainResource(base.owner); !retained) { return std::unexpected(retained.error()); }
        for (const auto& lease : base.leases) {
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

Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapCasterLayer(
    Device& device, CommandBuffer& commands, const VirtualShadowMapCasterResources& source,
    std::span<const uint64_t> dynamicCasterIds, bool staticLayer, std::string& log)
{
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    const auto sourceOwner = std::static_pointer_cast<CasterUploads>(source.owner);
    auto owner = std::make_shared<CasterUploads>();
    owner->sourceOwner = source.owner;
    owner->parameters = sourceOwner->parameters;
    VirtualShadowMapCasterResources resources = source;
    resources.owner = owner;
    resources.states.clear();
    std::vector<uint32_t> dynamicMask((uint64_t(source.triangleCapacity) + 31u) / 32u, 0u);
    for (const auto& caster : source.states) {
        const bool dynamic = std::ranges::binary_search(dynamicCasterIds, caster.id);
        if (dynamic != staticLayer) { resources.states.push_back(caster); }
        if (!dynamic || caster.triangleCount == 0) { continue; }
        uint32_t first = caster.triangleFirst;
        const uint32_t end = first + caster.triangleCount;
        while (first < end) {
            const uint32_t bit = first & 31u;
            const uint32_t count = std::min(32u - bit, end - first);
            dynamicMask[first / 32u] |= (UINT32_MAX >> (32u - count)) << bit;
            first += count;
        }
    }
    auto mask = upload(device, **registry, *owner, resources.leases,
        dynamicMask.data(), dynamicMask.size() * sizeof(uint32_t), sizeof(uint32_t));
    if (!mask) { return std::unexpected(mask.error()); }
    owner->parameters.casterLayerMask = *mask;
    owner->parameters.casterLayer = staticLayer ? 1u : 2u;
    auto parameters = upload(device, **registry, *owner, resources.leases,
        &owner->parameters, sizeof(GeometryParameters), sizeof(GeometryParameters));
    if (!parameters) { return std::unexpected(parameters.error()); }
    resources.parameters = owner->buffers.back().get();
    resources.layerBufferBytes = 0;
    for (const auto& buffer : owner->buffers) { resources.layerBufferBytes += buffer->desc().size; }
    if (auto result = commands.retainResource(owner); !result) { return std::unexpected(result.error()); }
    for (const auto& lease : resources.leases) {
        if (auto result = (*registry)->retain(commands, lease); !result) { return std::unexpected(result.error()); }
    }
    return resources;
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
        uint32_t tessellationFactor = 0;
        uint32_t subdivisionDepth = 0;

        bool operator==(const Key&) const = default;
    };

    Key key;
    std::vector<TextureView*> displacementTextures;
    VirtualShadowMapCasterResources resources;
};

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
        .texturePublicationRevision = settings.texturePublicationRevision,
        .tessellationEnabled = settings.tessellationEnabled,
        .tessellationFactor = settings.tessellationFactor,
        .subdivisionDepth = settings.subdivisionDepth};
    if (resident_ && resident_->key == key &&
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
        .resources = *resources});
    return resources;
}

} // namespace metallic::render
