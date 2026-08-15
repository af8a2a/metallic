#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <vector>

namespace metallic::render {
namespace {

uint32_t gpuUint(int32_t value)
{
    return std::bit_cast<uint32_t>(value);
}

uint32_t gpuCount(uint64_t value)
{
    return static_cast<uint32_t>(
        std::min<uint64_t>(value, std::numeric_limits<uint32_t>::max()));
}

std::array<const scene::RenderTextureInfo*, kGPUSceneMaterialTextureSlotCount>
materialTextures(const scene::RenderMaterial& material)
{
    return {
        &material.baseColorTexture,
        &material.metallicRoughnessTexture,
        &material.normalTexture,
        &material.occlusionTexture,
        &material.emissiveTexture,
        &material.transmissionTexture,
        &material.thicknessTexture,
        &material.diffuseTransmissionTexture,
        &material.diffuseTransmissionColorTexture,
    };
}

struct GPUSceneCpuUploadData {
    std::vector<GPUSceneGpuGeometryRecord> geometries;
    std::vector<GPUSceneGpuMaterialRecord> materials;
    std::vector<GPUSceneGpuInstanceRecord> instances;
    std::vector<GPUSceneGpuDrawKeyRecord> drawKeys;
    std::vector<uint32_t> drawInstanceIds;
    std::vector<GPUSceneGpuVertexRecord> vertices;
    std::vector<uint32_t> indices;
    std::vector<GPUSceneGpuMeshletRecord> meshlets;
    std::vector<GPUSceneGpuMeshletDrawRecord> meshletDraws;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint32_t> meshletTriangleWords;
    std::vector<GPUSceneGpuDescriptorRemapRecord> descriptorRemap;
    GPUSceneRasterDrawLayout rasterDrawLayout;
};

struct GPUSceneGeometryRasterRanges {
    GPUSceneRasterDrawRange baseRange;
    std::vector<GPUSceneRasterDrawRange> lodRanges;
};

std::vector<GPUSceneGpuInstanceRecord> buildGpuInstanceRecords(
    const GPUScene& gpuScene)
{
    std::vector<GPUSceneGpuInstanceRecord> records;
    records.reserve(gpuScene.instances().size());
    for (const GPUSceneInstanceRecord& instance : gpuScene.instances()) {
        GPUSceneGpuInstanceRecord gpu;
        std::copy_n(instance.worldMatrix.a, 16, gpu.worldMatrix.begin());
        std::copy_n(instance.previousWorldMatrix.a, 16, gpu.previousWorldMatrix.begin());
        gpu.localBoundingSphere = {
            instance.localBoundingSphere.x,
            instance.localBoundingSphere.y,
            instance.localBoundingSphere.z,
            instance.localBoundingSphere.w,
        };
        uint32_t flags = instance.visible ? GPUSceneGpuInstanceVisible : 0u;
        const GPUSceneDrawBucket bucket = instance.drawKey.bucket;
        if (bucket == GPUSceneDrawBucket::OpaqueDoubleSided ||
            bucket == GPUSceneDrawBucket::MaskedDoubleSided ||
            bucket == GPUSceneDrawBucket::Blend) {
            flags |= GPUSceneGpuInstanceDoubleSided;
        }
        if (bucket == GPUSceneDrawBucket::MaskedSingleSided ||
            bucket == GPUSceneDrawBucket::MaskedDoubleSided) {
            flags |= GPUSceneGpuInstanceMasked;
        } else if (bucket == GPUSceneDrawBucket::Blend) {
            flags |= GPUSceneGpuInstanceBlend;
        }
        gpu.identity = {
            instance.geometry.index,
            instance.material.index,
            gpuUint(instance.sourceRenderNodeIndex),
            flags,
        };
        records.push_back(gpu);
    }
    return records;
}

uint32_t appendMeshletTriangleWords(
    std::span<const uint8_t> bytes,
    std::vector<uint32_t>& words)
{
    const uint32_t firstWord = gpuCount(words.size());
    for (size_t offset = 0; offset < bytes.size(); offset += 4) {
        uint32_t word = 0;
        const size_t byteCount = std::min<size_t>(4, bytes.size() - offset);
        std::memcpy(&word, bytes.data() + offset, byteCount);
        words.push_back(word);
    }
    return firstWord;
}

GPUSceneRasterDrawRange appendMeshletRange(
    const scene::RenderPrimitive& primitive,
    std::span<const scene::MeshletCluster> clusters,
    std::span<const uint32_t> meshletVertices,
    std::span<const uint8_t> meshletTriangles,
    uint32_t firstCluster,
    uint32_t clusterCount,
    GPUSceneCpuUploadData& data)
{
    GPUSceneRasterDrawRange range{
        .offset = gpuCount(data.meshlets.size()),
    };
    if (firstCluster > clusters.size() ||
        clusterCount > clusters.size() - firstCluster) {
        return range;
    }

    const std::span<const scene::MeshletCluster> selected =
        clusters.subspan(firstCluster, clusterCount);
    for (const scene::MeshletCluster& cluster : selected) {
        const uint64_t triangleByteCount =
            static_cast<uint64_t>(cluster.triangleCount) * 3u;
        if (cluster.vertexCount == 0 || cluster.triangleCount == 0 ||
            cluster.vertexOffset > meshletVertices.size() ||
            cluster.vertexCount > meshletVertices.size() - cluster.vertexOffset ||
            cluster.triangleOffset > meshletTriangles.size() ||
            triangleByteCount > meshletTriangles.size() - cluster.triangleOffset) {
            continue;
        }

        const std::span<const uint32_t> localVertices = meshletVertices.subspan(
            cluster.vertexOffset,
            cluster.vertexCount);
        const std::span<const uint8_t> localTriangles = meshletTriangles.subspan(
            cluster.triangleOffset,
            static_cast<size_t>(triangleByteCount));
        const bool validVertices = std::ranges::all_of(
            localVertices,
            [&](uint32_t vertexIndex) {
                return vertexIndex < primitive.positions.size();
            });
        const bool validTriangles = std::ranges::all_of(
            localTriangles,
            [&](uint8_t vertexIndex) {
                return vertexIndex < cluster.vertexCount;
            });
        if (!validVertices || !validTriangles) {
            continue;
        }

        const uint32_t vertexOffset = gpuCount(data.meshletVertices.size());
        data.meshletVertices.insert(
            data.meshletVertices.end(),
            localVertices.begin(),
            localVertices.end());
        const uint32_t triangleWordOffset =
            appendMeshletTriangleWords(localTriangles, data.meshletTriangleWords);

        GPUSceneGpuMeshletRecord meshlet;
        meshlet.ranges = {
            vertexOffset,
            cluster.vertexCount,
            triangleWordOffset,
            cluster.triangleCount,
        };
        meshlet.lod = {
            cluster.lodLevel,
            static_cast<uint32_t>(std::max(cluster.lodGroupIndex, 0)),
            0,
            0,
        };
        meshlet.boundingSphere = {
            cluster.boundingSphereCenter.x,
            cluster.boundingSphereCenter.y,
            cluster.boundingSphereCenter.z,
            std::max(cluster.boundingSphereRadius, 0.0f),
        };
        meshlet.coneApexCutoff = {
            cluster.coneApex.x,
            cluster.coneApex.y,
            cluster.coneApex.z,
            cluster.coneCutoff,
        };
        meshlet.coneAxisLodError = {
            cluster.coneAxis.x,
            cluster.coneAxis.y,
            cluster.coneAxis.z,
            cluster.lodError,
        };
        data.meshlets.push_back(meshlet);
    }
    range.count = gpuCount(data.meshlets.size()) - range.offset;
    return range;
}

GPUSceneGpuMaterialTextureInfo buildGpuMaterialTextureInfo(
    const scene::RenderTextureInfo& source,
    uint32_t materialIndex,
    uint32_t textureSlot,
    GPUSceneCpuUploadData& data)
{
    GPUSceneGpuMaterialTextureInfo texture;
    texture.textureIndex = gpuCount(data.descriptorRemap.size());
    texture.texCoord = static_cast<uint32_t>(std::max(source.texCoord, 0));
    texture.transform0 = {
        source.uvTransform[0],
        source.uvTransform[1],
        source.uvTransform[2],
        0.0f,
    };
    texture.transform1 = {
        source.uvTransform[3],
        source.uvTransform[4],
        source.uvTransform[5],
        0.0f,
    };
    data.descriptorRemap.push_back(GPUSceneGpuDescriptorRemapRecord{
        .logicalTextureId = source.textureIndex,
        .descriptorIndex = std::numeric_limits<uint32_t>::max(),
        .materialIndex = materialIndex,
        .textureSlot = textureSlot,
    });
    return texture;
}

GPUSceneCpuUploadData buildGpuUploadData(
    const GPUScene& gpuScene)
{
    GPUSceneCpuUploadData data;
    data.geometries.reserve(gpuScene.geometries().size());
    data.materials.reserve(gpuScene.materials().size());
    data.instances.reserve(gpuScene.instances().size());
    data.drawInstanceIds.reserve(gpuScene.drawSet().instances.size());
    std::vector<GPUSceneGeometryRasterRanges> geometryRanges;
    geometryRanges.resize(gpuScene.geometries().size());

    for (const GPUSceneGeometryRecord& geometry : gpuScene.geometries()) {
        GPUSceneGpuGeometryRecord gpu;
        gpu.source = {
            gpuUint(geometry.sourceRenderPrimitiveIndex),
            gpuUint(geometry.meshIndex),
            gpuUint(geometry.primitiveIndex),
            gpuUint(geometry.mode),
        };
        gpu.counts = {
            gpuCount(geometry.vertexCount),
            gpuCount(geometry.indexCount),
            gpuCount(geometry.triangleCount),
            0,
        };
        const float3 center = geometry.localBounds.valid
            ? geometry.localBounds.center()
            : float3(0.0f, 0.0f, 0.0f);
        gpu.localBoundingSphere = {
            center.x,
            center.y,
            center.z,
            geometry.localBounds.valid ? std::max(geometry.localBounds.radius(), 0.0f) : 0.0f,
        };
        gpu.identity = {
            geometry.id.index,
            geometry.id.generation,
            static_cast<uint32_t>(geometry.payloadFingerprint),
            static_cast<uint32_t>(geometry.payloadFingerprint >> 32u),
        };

        gpu.payload = {
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            0,
        };
        gpu.meshletPayload = {
            std::numeric_limits<uint32_t>::max(),
            0,
            std::numeric_limits<uint32_t>::max(),
            0,
        };

        const scene::RenderPrimitive* sourcePrimitive =
            gpuScene.geometrySourcePrimitive(geometry.id);
        if (sourcePrimitive != nullptr) {
            const scene::RenderPrimitive& primitive = *sourcePrimitive;
            const uint32_t vertexOffset = gpuCount(data.vertices.size());
            for (size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
                const float3 position = primitive.positions[vertexIndex];
                const float3 normal = vertexIndex < primitive.normals.size()
                    ? primitive.normals[vertexIndex]
                    : float3(0.0f, 0.0f, 1.0f);
                const float4 tangent = vertexIndex < primitive.tangents.size()
                    ? primitive.tangents[vertexIndex]
                    : float4(1.0f, 0.0f, 0.0f, 1.0f);
                const float2 texcoord = vertexIndex < primitive.texcoords0.size()
                    ? primitive.texcoords0[vertexIndex]
                    : float2(0.0f, 0.0f);
                GPUSceneGpuVertexRecord vertex;
                vertex.position = {position.x, position.y, position.z, 1.0f};
                vertex.normal = {normal.x, normal.y, normal.z, 0.0f};
                vertex.tangent = {tangent.x, tangent.y, tangent.z, tangent.w};
                vertex.texcoord = {texcoord.x, texcoord.y};
                vertex.flags = (primitive.hasAuthoredNormals ? 1u : 0u) |
                    (primitive.hasAuthoredTangents ? 2u : 0u);
                data.vertices.push_back(vertex);
            }

            const uint32_t indexOffset = gpuCount(data.indices.size());
            if (primitive.indices.empty()) {
                for (uint32_t index = 0; index < primitive.positions.size(); ++index) {
                    data.indices.push_back(index);
                }
            } else {
                data.indices.insert(
                    data.indices.end(),
                    primitive.indices.begin(),
                    primitive.indices.end());
            }

            const uint32_t meshletVertexOffset = gpuCount(data.meshletVertices.size());
            const uint32_t meshletTriangleWordOffset = gpuCount(data.meshletTriangleWords.size());
            GPUSceneGeometryRasterRanges& ranges = geometryRanges[geometry.id.index];
            ranges.baseRange = appendMeshletRange(
                primitive,
                primitive.meshletClusters,
                primitive.meshletVertices,
                primitive.meshletTriangles,
                0,
                gpuCount(primitive.meshletClusters.size()),
                data);
            ranges.lodRanges.resize(primitive.meshletLodLevels.size());
            for (uint32_t lodLevel = 0;
                 lodLevel < primitive.meshletLodLevels.size();
                 ++lodLevel) {
                const scene::MeshletLodLevel& level =
                    primitive.meshletLodLevels[lodLevel];
                ranges.lodRanges[lodLevel] = appendMeshletRange(
                    primitive,
                    primitive.meshletLodClusters,
                    primitive.meshletLodVertices,
                    primitive.meshletLodTriangles,
                    level.clusterOffset,
                    level.clusterCount,
                    data);
            }
            gpu.payload = {
                vertexOffset,
                indexOffset,
                ranges.baseRange.offset,
                ranges.baseRange.count,
            };
            gpu.meshletPayload = {
                meshletVertexOffset,
                gpuCount(data.meshletVertices.size()) - meshletVertexOffset,
                meshletTriangleWordOffset,
                gpuCount(data.meshletTriangleWords.size()) - meshletTriangleWordOffset,
            };
            gpu.counts[3] = gpuCount(ranges.lodRanges.size());
        }
        data.geometries.push_back(gpu);
    }

    for (const GPUSceneMaterialRecord& material : gpuScene.materials()) {
        const scene::RenderMaterial& source = material.material;
        GPUSceneGpuMaterialRecord gpu;
        gpu.baseColor = {
            source.baseColorFactor.x,
            source.baseColorFactor.y,
            source.baseColorFactor.z,
            source.baseColorFactor.w,
        };
        gpu.emissive = {
            source.emissiveFactor.x,
            source.emissiveFactor.y,
            source.emissiveFactor.z,
            0.0f,
        };
        gpu.params = {
            source.metallicFactor,
            source.roughnessFactor,
            source.alphaCutoff,
            source.doubleSided ? 1.0f : 0.0f,
        };
        gpu.textureParams = {
            source.normalTextureScale,
            source.occlusionTextureStrength,
            0.0f,
            0.0f,
        };
        gpu.glassParams = {
            source.transmissionFactor,
            source.ior,
            source.thicknessFactor,
            source.attenuationDistance,
        };
        gpu.attenuationColor = {
            source.attenuationColor.x,
            source.attenuationColor.y,
            source.attenuationColor.z,
            0.0f,
        };
        gpu.diffuseTransmission = {
            source.diffuseTransmissionColor.x,
            source.diffuseTransmissionColor.y,
            source.diffuseTransmissionColor.z,
            source.diffuseTransmissionFactor,
        };
        const auto textures = materialTextures(source);
        std::array<GPUSceneGpuMaterialTextureInfo*, kGPUSceneMaterialTextureSlotCount>
            gpuTextures{
                &gpu.baseColorTexture,
                &gpu.metallicRoughnessTexture,
                &gpu.normalTexture,
                &gpu.occlusionTexture,
                &gpu.emissiveTexture,
                &gpu.transmissionTexture,
                &gpu.thicknessTexture,
                &gpu.diffuseTransmissionTexture,
                &gpu.diffuseTransmissionColorTexture,
            };
        for (uint32_t textureSlot = 0; textureSlot < textures.size(); ++textureSlot) {
            *gpuTextures[textureSlot] = buildGpuMaterialTextureInfo(
                *textures[textureSlot],
                material.id.index,
                textureSlot,
                data);
        }
        uint32_t flags = material.fallback ? 1u << 4 : 0u;
        if (source.doubleSided) {
            flags |= GPUSceneGpuInstanceDoubleSided;
        }
        if (material.bucket == GPUSceneDrawBucket::MaskedSingleSided ||
            material.bucket == GPUSceneDrawBucket::MaskedDoubleSided) {
            flags |= GPUSceneGpuInstanceMasked;
        } else if (material.bucket == GPUSceneDrawBucket::Blend) {
            flags |= GPUSceneGpuInstanceBlend;
        }
        gpu.identity = {
            material.id.index,
            material.id.generation,
            gpuUint(material.sourceMaterialIndex),
            flags,
        };
        data.materials.push_back(gpu);
    }

    data.instances = buildGpuInstanceRecords(gpuScene);

    const GPUSceneDrawSet& drawSet = gpuScene.drawSet();
    for (GPUSceneInstanceId id : drawSet.instances) {
        data.drawInstanceIds.push_back(id.index);
        const GPUSceneInstanceRecord* instance = gpuScene.instance(id);
        if (instance == nullptr) {
            continue;
        }
        if (!data.drawKeys.empty()) {
            GPUSceneGpuDrawKeyRecord& last = data.drawKeys.back();
            if (last.key[0] == static_cast<uint32_t>(instance->drawKey.bucket) &&
                last.key[1] == instance->material.index &&
                last.key[2] == instance->geometry.index) {
                ++last.range[0];
                continue;
            }
        }
        GPUSceneGpuDrawKeyRecord key;
        key.key = {
            static_cast<uint32_t>(instance->drawKey.bucket),
            instance->material.index,
            instance->geometry.index,
            gpuCount(data.drawInstanceIds.size() - 1),
        };
        key.range = {
            1,
            instance->material.generation,
            instance->geometry.generation,
            0,
        };
        data.drawKeys.push_back(key);
    }

    const auto appendDrawRange = [&](uint32_t lodLevel, bool baseRange) {
        GPUSceneRasterDrawRange range{
            .offset = gpuCount(data.meshletDraws.size()),
        };
        for (GPUSceneInstanceId id : drawSet.instances) {
            const GPUSceneInstanceRecord* instance = gpuScene.instance(id);
            if (instance == nullptr || instance->geometry.index >= geometryRanges.size()) {
                continue;
            }
            const GPUSceneGeometryRasterRanges& geometry =
                geometryRanges[instance->geometry.index];
            const GPUSceneRasterDrawRange* geometryRange = nullptr;
            if (baseRange) {
                geometryRange = &geometry.baseRange;
            } else if (lodLevel < geometry.lodRanges.size()) {
                geometryRange = &geometry.lodRanges[lodLevel];
            }
            if (geometryRange == nullptr) {
                continue;
            }
            for (uint32_t meshletOffset = 0;
                 meshletOffset < geometryRange->count;
                 ++meshletOffset) {
                data.meshletDraws.push_back(VisibleClusterRecord{
                    .clusterIndex = geometryRange->offset + meshletOffset,
                    .instanceIndex = id.index,
                    .dataIndex = instance->geometry.index,
                    .flags = visibleClusterFlags(
                        VisibleClusterSource::Resident,
                        static_cast<uint32_t>(instance->drawKey.bucket)),
                });
            }
        }
        range.count = gpuCount(data.meshletDraws.size()) - range.offset;
        data.rasterDrawLayout.maxRangeCount =
            std::max(data.rasterDrawLayout.maxRangeCount, range.count);
        return range;
    };

    data.rasterDrawLayout.baseRange = appendDrawRange(0, true);
    size_t maxLodLevelCount = 0;
    for (const GPUSceneGeometryRasterRanges& geometry : geometryRanges) {
        maxLodLevelCount = std::max(maxLodLevelCount, geometry.lodRanges.size());
    }
    data.rasterDrawLayout.lodRanges.resize(maxLodLevelCount);
    for (uint32_t lodLevel = 0; lodLevel < maxLodLevelCount; ++lodLevel) {
        data.rasterDrawLayout.lodRanges[lodLevel] = appendDrawRange(lodLevel, false);
    }
    data.rasterDrawLayout.drawSetGeneration = drawSet.generation;
    data.rasterDrawLayout.drawSetRevision = drawSet.revision;
    return data;
}

} // namespace

struct GPUSceneSubsystem::GpuBufferResource {
    std::unique_ptr<Buffer> buffer;
    std::unique_ptr<BufferView> view;
    uint64_t byteSize = 0;
    uint32_t structureStride = 0;

    GPUSceneBufferView sceneView(uint32_t generation, uint64_t revision) const
    {
        if (buffer == nullptr || view == nullptr) {
            return {};
        }
        return GPUSceneBufferView{
            .buffer = buffer.get(),
            .view = view.get(),
            .offset = 0,
            .size = byteSize,
            .structureStride = structureStride,
            .generation = generation,
            .revision = revision,
        };
    }
};

struct GPUSceneSubsystem::GpuResources {
    std::array<GpuBufferResource, kGPUSceneGlobalBufferKindCount> buffers;
    uint32_t generation = 0;
    uint64_t revision = 0;

    GpuBufferResource& resource(GPUSceneGlobalBufferKind kind)
    {
        return buffers[static_cast<size_t>(kind)];
    }

    const GpuBufferResource& resource(GPUSceneGlobalBufferKind kind) const
    {
        return buffers[static_cast<size_t>(kind)];
    }

    GPUSceneGlobalBufferViews views() const
    {
        GPUSceneGlobalBufferViews result;
        result.geometries = resource(GPUSceneGlobalBufferKind::Geometries).sceneView(generation, revision);
        result.materials = resource(GPUSceneGlobalBufferKind::Materials).sceneView(generation, revision);
        result.instances = resource(GPUSceneGlobalBufferKind::Instances).sceneView(generation, revision);
        result.drawKeys = resource(GPUSceneGlobalBufferKind::DrawKeys).sceneView(generation, revision);
        result.drawInstanceIds = resource(GPUSceneGlobalBufferKind::DrawInstanceIds).sceneView(generation, revision);
        result.vertices = resource(GPUSceneGlobalBufferKind::Vertices).sceneView(generation, revision);
        result.indices = resource(GPUSceneGlobalBufferKind::Indices).sceneView(generation, revision);
        result.meshlets = resource(GPUSceneGlobalBufferKind::Meshlets).sceneView(generation, revision);
        result.meshletDraws = resource(GPUSceneGlobalBufferKind::MeshletDraws).sceneView(generation, revision);
        result.meshletVertices = resource(GPUSceneGlobalBufferKind::MeshletVertices).sceneView(generation, revision);
        result.meshletTriangleWords = resource(GPUSceneGlobalBufferKind::MeshletTriangleWords).sceneView(generation, revision);
        result.descriptorRemap = resource(GPUSceneGlobalBufferKind::DescriptorRemap).sceneView(generation, revision);
        result.drawSetGeneration = generation;
        result.drawSetRevision = revision;
        return result;
    }
};

struct GPUSceneSubsystem::ViewGpuResources {
    struct FrameSlotResources {
        GpuBufferResource instanceVisibilityStates;
        GpuBufferResource visibleInstanceIds;
        GpuBufferResource visibleInstanceCounter;
        std::array<GpuBufferResource, kGPUSceneCullPhaseCount> visibleMeshletIds;
        std::array<GpuBufferResource, kGPUSceneCullPhaseCount> indirectArguments;
        bool initialized = false;
    };

    GPUSceneViewId sourceView;
    GPUSceneViewDesc desc;
    std::vector<FrameSlotResources> frameSlots;
    std::array<GpuBufferResource, 2> hzbHistory;
    uint64_t allocationId = 0;
    bool hzbInitialized = false;

    GPUSceneViewGpuResourcesView view(
        uint32_t frameSlot,
        uint32_t generation,
        uint64_t revision) const
    {
        GPUSceneViewGpuResourcesView result;
        if (frameSlot >= frameSlots.size()) {
            return result;
        }
        result.sourceView = sourceView;
        result.desc = desc;
        result.frameSlot = frameSlot;
        result.allocationId = allocationId;
        result.frameSlotInitialized = frameSlots[frameSlot].initialized;
        result.hzbInitialized = hzbInitialized;
        const FrameSlotResources& slot = frameSlots[frameSlot];
        result.instanceVisibilityStates =
            slot.instanceVisibilityStates.sceneView(generation, revision);
        result.visibleInstanceIds =
            slot.visibleInstanceIds.sceneView(generation, revision);
        result.visibleInstanceCounter =
            slot.visibleInstanceCounter.sceneView(generation, revision);
        uint32_t visibleMeshletOffset = 0;
        for (size_t phaseIndex = 0;
             phaseIndex < kGPUSceneCullPhaseCount;
             ++phaseIndex) {
            GPUSceneCullPhaseGpuView& phase = result.phases[phaseIndex];
            phase.visibleMeshletIds =
                slot.visibleMeshletIds[phaseIndex].sceneView(generation, revision);
            const GPUSceneBufferView indirect =
                slot.indirectArguments[phaseIndex].sceneView(generation, revision);
            visibleMeshletOffset = 0;
            for (size_t bucketIndex = 0;
                 bucketIndex < kGPUSceneRasterDrawBucketCount;
                 ++bucketIndex) {
                GPUSceneBucketGpuView& bucket = phase.buckets[bucketIndex];
                bucket.indirectArguments = indirect;
                bucket.indirectArguments.offset = bucketIndex * 4u * sizeof(uint32_t);
                bucket.indirectArguments.size = 3u * sizeof(uint32_t);
                bucket.overflow = indirect;
                bucket.overflow.offset =
                    bucketIndex * 4u * sizeof(uint32_t) + 3u * sizeof(uint32_t);
                bucket.overflow.size = sizeof(uint32_t);
                bucket.visibleMeshletOffset = visibleMeshletOffset;
                bucket.visibleMeshletCapacity = desc.visibleMeshletCapacity[bucketIndex];
                visibleMeshletOffset += bucket.visibleMeshletCapacity;
            }
        }
        for (size_t historyIndex = 0;
             historyIndex < hzbHistory.size();
             ++historyIndex) {
            result.hzbHistory[historyIndex] =
                hzbHistory[historyIndex].sceneView(generation, revision);
        }
        return result;
    }
};

struct GPUSceneSubsystem::UploadResources {
    std::vector<std::unique_ptr<Buffer>> stagingBuffers;
};

Result GPUSceneSubsystem::initialize(
    const RenderSubsystemInitContext& context,
    std::string&)
{
    device_ = &context.device;
    host_ = &context.host;
    frameSlotCount_ = context.host.frameSlotCount();
    scene_.setDefaultFrameSlotCount(frameSlotCount_);
    pendingUpload_ = PendingUpload::Full;
    return {};
}

uint64_t GPUSceneSubsystem::viewResourceKey(GPUSceneViewId view)
{
    return (static_cast<uint64_t>(view.generation) << 32u) | view.index;
}

void GPUSceneSubsystem::retireViewGpuResources(
    std::shared_ptr<ViewGpuResources> resources)
{
    if (resources != nullptr && host_ != nullptr) {
        host_->retire(std::static_pointer_cast<void>(std::move(resources)));
    }
}

GPUSceneViewId GPUSceneSubsystem::createView(const GPUSceneViewDesc& desc)
{
    GPUSceneViewId view;
    std::string ignoredLog;
    return createView(desc, view, ignoredLog) ? view : GPUSceneViewId{};
}

Result GPUSceneSubsystem::createView(
    const GPUSceneViewDesc& desc,
    GPUSceneViewId& view,
    std::string& log)
{
    view = scene_.createView(desc);
    if (!view) {
        log = "GPUSceneSubsystem failed to allocate a CPU View slot";
        return makeError(Error::Failure);
    }
    const bool requestsGpuResources = desc.instanceCapacity != 0 ||
        std::ranges::any_of(
            desc.visibleMeshletCapacity,
            [](uint32_t capacity) { return capacity != 0; }) ||
        desc.hzbWidth != 0 || desc.hzbHeight != 0 ||
        desc.hzbMipCount != 0 || desc.hzbElementCount != 0;
    if (!requestsGpuResources) {
        return {};
    }

    Result result = ensureViewGpuResources(view, desc, log);
    if (!result) {
        scene_.destroyView(view);
        view = {};
    }
    return result;
}

bool GPUSceneSubsystem::destroyView(GPUSceneViewId view)
{
    if (!scene_.destroyView(view)) {
        return false;
    }
    const auto resources = viewGpuResources_.find(viewResourceKey(view));
    if (resources != viewGpuResources_.end()) {
        std::shared_ptr<ViewGpuResources> retired = std::move(resources->second);
        viewGpuResources_.erase(resources);
        retireViewGpuResources(std::move(retired));
    }
    return true;
}

Result GPUSceneSubsystem::ensureViewGpuResources(
    GPUSceneViewId view,
    const GPUSceneViewDesc& requestedDesc,
    std::string& log)
{
    if (device_ == nullptr || host_ == nullptr) {
        log = "GPUSceneSubsystem View GPU resources require an initialized subsystem";
        return makeError(Error::InvalidArgument);
    }
    const uint32_t viewFrameSlotCount = scene_.viewFrameSlotCount(view);
    if (viewFrameSlotCount == 0) {
        log = "GPUSceneSubsystem View GPU resources require a live View";
        return makeError(Error::InvalidArgument);
    }

    GPUSceneViewDesc desc = requestedDesc;
    if (desc.frameSlotCount == 0) {
        desc.frameSlotCount = viewFrameSlotCount;
    }
    if (desc.frameSlotCount != viewFrameSlotCount ||
        desc.instanceCapacity == 0 ||
        desc.hzbWidth == 0 || desc.hzbHeight == 0 ||
        desc.hzbMipCount == 0 || desc.hzbMipCount > 32 ||
        desc.hzbElementCount == 0 ||
        std::ranges::any_of(
            desc.visibleMeshletCapacity,
            [](uint32_t capacity) { return capacity == 0; })) {
        log = "GPUSceneSubsystem View GPU resource description is incomplete or does not match the View frame-slot count";
        return makeError(Error::InvalidArgument);
    }

    uint32_t mipWidth = desc.hzbWidth;
    uint32_t mipHeight = desc.hzbHeight;
    uint64_t minimumHzbElementCount = 0;
    for (uint32_t mipLevel = 0; mipLevel < desc.hzbMipCount; ++mipLevel) {
        const uint64_t mipElementCount =
            static_cast<uint64_t>(mipWidth) * mipHeight;
        if (mipElementCount >
            std::numeric_limits<uint64_t>::max() - minimumHzbElementCount) {
            log = "GPUSceneSubsystem HZB mip-chain element count overflow";
            return makeError(Error::InvalidArgument);
        }
        minimumHzbElementCount += mipElementCount;
        mipWidth = std::max(1u, mipWidth / 2u + mipWidth % 2u);
        mipHeight = std::max(1u, mipHeight / 2u + mipHeight % 2u);
    }
    if (desc.hzbElementCount < minimumHzbElementCount) {
        log = "GPUSceneSubsystem HZB element capacity is smaller than its mip chain";
        return makeError(Error::InvalidArgument);
    }

    const uint64_t key = viewResourceKey(view);
    const auto current = viewGpuResources_.find(key);
    if (current != viewGpuResources_.end()) {
        const GPUSceneViewDesc& currentDesc = current->second->desc;
        desc.instanceCapacity = std::max(
            desc.instanceCapacity,
            currentDesc.instanceCapacity);
        for (size_t bucketIndex = 0;
             bucketIndex < kGPUSceneRasterDrawBucketCount;
             ++bucketIndex) {
            desc.visibleMeshletCapacity[bucketIndex] = std::max(
                desc.visibleMeshletCapacity[bucketIndex],
                currentDesc.visibleMeshletCapacity[bucketIndex]);
        }
        if (desc.hzbWidth == currentDesc.hzbWidth &&
            desc.hzbHeight == currentDesc.hzbHeight &&
            desc.hzbMipCount == currentDesc.hzbMipCount) {
            desc.hzbElementCount = std::max(
                desc.hzbElementCount,
                currentDesc.hzbElementCount);
        }
        const bool sameAllocation =
            desc.frameSlotCount == currentDesc.frameSlotCount &&
            desc.instanceCapacity == currentDesc.instanceCapacity &&
            desc.visibleMeshletCapacity == currentDesc.visibleMeshletCapacity &&
            desc.hzbWidth == currentDesc.hzbWidth &&
            desc.hzbHeight == currentDesc.hzbHeight &&
            desc.hzbMipCount == currentDesc.hzbMipCount &&
            desc.hzbElementCount == currentDesc.hzbElementCount;
        if (sameAllocation) {
            return {};
        }
    }

    uint64_t totalVisibleMeshletCapacity = 0;
    for (uint32_t capacity : desc.visibleMeshletCapacity) {
        totalVisibleMeshletCapacity += capacity;
    }
    if (totalVisibleMeshletCapacity > std::numeric_limits<uint32_t>::max() ||
        desc.instanceCapacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ||
        totalVisibleMeshletCapacity >
            std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ||
        desc.hzbElementCount >
            std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        log = "GPUSceneSubsystem View GPU resource size overflow";
        return makeError(Error::InvalidArgument);
    }

    auto next = std::make_shared<ViewGpuResources>();
    next->sourceView = view;
    next->desc = desc;
    next->frameSlots.resize(desc.frameSlotCount);
    next->allocationId = nextViewGpuResourceAllocationId_++;
    if (nextViewGpuResourceAllocationId_ == 0) {
        nextViewGpuResourceAllocationId_ = 1;
    }

    auto createBufferResource = [this, &log](
                                    uint64_t byteSize,
                                    uint32_t structureStride,
                                    BufferUsageBits usage,
                                    GpuBufferResource& resource,
                                    const char* label) -> Result {
        Result result = device_->createBuffer(
            BufferDesc{
                .size = byteSize,
                .structureStride = structureStride,
                .usage = usage,
                .memoryLocation = MemoryLocation::Device,
                .queueAccess = QueueAccessBits::Graphics |
                    QueueAccessBits::Compute,
            },
            resource.buffer);
        if (!result || resource.buffer == nullptr) {
            log = std::string("GPUSceneSubsystem failed to create View ") +
                label + " buffer: " + resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        resource.byteSize = byteSize;
        resource.structureStride = structureStride;
        result = device_->createBufferView(
            *resource.buffer,
            BufferViewDesc{
                .type = BufferViewType::ReadWriteStructured,
                .offset = 0,
                .size = byteSize,
                .structureStride = structureStride,
            },
            resource.view);
        if (!result || resource.view == nullptr) {
            log = std::string("GPUSceneSubsystem failed to create View ") +
                label + " buffer view: " + resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    };

    const uint64_t instanceByteSize =
        static_cast<uint64_t>(desc.instanceCapacity) * sizeof(uint32_t);
    const uint64_t visibleMeshletByteSize =
        totalVisibleMeshletCapacity * sizeof(uint32_t);
    constexpr uint64_t kIndirectByteSize =
        kGPUSceneRasterDrawBucketCount * 4u * sizeof(uint32_t);
    const uint64_t hzbByteSize = desc.hzbElementCount * sizeof(float);
    Result result;
    for (uint32_t frameSlot = 0;
         frameSlot < desc.frameSlotCount && result;
         ++frameSlot) {
        ViewGpuResources::FrameSlotResources& slot = next->frameSlots[frameSlot];
        result = createBufferResource(
            instanceByteSize,
            sizeof(uint32_t),
            BufferUsageBits::Storage,
            slot.instanceVisibilityStates,
            "instance visibility");
        if (result) {
            result = createBufferResource(
                instanceByteSize,
                sizeof(uint32_t),
                BufferUsageBits::Storage,
                slot.visibleInstanceIds,
                "visible instance IDs");
        }
        if (result) {
            result = createBufferResource(
                sizeof(uint32_t),
                sizeof(uint32_t),
                BufferUsageBits::Storage,
                slot.visibleInstanceCounter,
                "visible instance counter");
        }
        for (size_t phaseIndex = 0;
             phaseIndex < kGPUSceneCullPhaseCount && result;
             ++phaseIndex) {
            result = createBufferResource(
                visibleMeshletByteSize,
                sizeof(uint32_t),
                BufferUsageBits::Storage,
                slot.visibleMeshletIds[phaseIndex],
                "visible meshlet worklist");
            if (result) {
                result = createBufferResource(
                    kIndirectByteSize,
                    sizeof(uint32_t),
                    BufferUsageBits::Storage | BufferUsageBits::Indirect,
                    slot.indirectArguments[phaseIndex],
                    "mesh task indirect arguments");
            }
        }
    }
    for (size_t historyIndex = 0;
         historyIndex < next->hzbHistory.size() && result;
         ++historyIndex) {
        result = createBufferResource(
            hzbByteSize,
            sizeof(float),
            BufferUsageBits::Storage,
            next->hzbHistory[historyIndex],
            "HZB history");
    }
    if (!result) {
        return result;
    }

    // The new HZB allocation has no temporal contents. Clear all previously
    // published raw views before the old bundle enters deferred retirement.
    scene_.invalidateViewGpuResources(view, true);
    if (current != viewGpuResources_.end()) {
        std::shared_ptr<ViewGpuResources> retired = std::move(current->second);
        current->second = next;
        retireViewGpuResources(std::move(retired));
    } else {
        viewGpuResources_.emplace(key, next);
    }
    return {};
}

bool GPUSceneSubsystem::viewGpuResources(
    GPUSceneViewId view,
    uint32_t frameSlot,
    GPUSceneViewGpuResourcesView& resources) const
{
    resources = {};
    const auto found = viewGpuResources_.find(viewResourceKey(view));
    if (found == viewGpuResources_.end() || found->second == nullptr ||
        found->second->sourceView != view ||
        frameSlot >= found->second->frameSlots.size()) {
        return false;
    }
    resources = found->second->view(
        frameSlot,
        scene_.drawSet().generation,
        scene_.drawSet().revision);
    return resources.sourceView == view;
}

Result GPUSceneSubsystem::recordInitialize(
    CommandBuffer& commandBuffer,
    GPUSceneViewId view,
    uint32_t frameSlot,
    std::string& log)
{
    const auto found = viewGpuResources_.find(viewResourceKey(view));
    if (found == viewGpuResources_.end() || found->second == nullptr ||
        found->second->sourceView != view ||
        frameSlot >= found->second->frameSlots.size()) {
        log = "GPUSceneSubsystem recordInitialize requires a live View GPU bundle and frame slot";
        return makeError(Error::InvalidArgument);
    }
    ViewGpuResources& resources = *found->second;
    ViewGpuResources::FrameSlotResources& slot = resources.frameSlots[frameSlot];
    std::vector<BufferBarrierDesc> barriers;
    barriers.reserve(9);
    auto initializeResource = [&barriers](const GpuBufferResource& resource) {
        barriers.push_back(BufferBarrierDesc{
            .buffer = resource.buffer.get(),
            .before = ResourceState::Undefined,
            .after = ResourceState::General,
            .offset = 0,
            .size = resource.byteSize,
        });
    };
    if (!slot.initialized) {
        initializeResource(slot.instanceVisibilityStates);
        initializeResource(slot.visibleInstanceIds);
        initializeResource(slot.visibleInstanceCounter);
        for (size_t phaseIndex = 0;
             phaseIndex < kGPUSceneCullPhaseCount;
             ++phaseIndex) {
            initializeResource(slot.visibleMeshletIds[phaseIndex]);
            initializeResource(slot.indirectArguments[phaseIndex]);
        }
    }
    if (!resources.hzbInitialized) {
        for (const GpuBufferResource& history : resources.hzbHistory) {
            initializeResource(history);
        }
    }
    if (!barriers.empty()) {
        commandBuffer.barrier(BarrierDesc{
            .buffers = barriers.data(),
            .bufferCount = gpuCount(barriers.size()),
        });
    }
    slot.initialized = true;
    resources.hzbInitialized = true;
    return {};
}

Result GPUSceneSubsystem::publishViewGpuResources(
    GPUSceneViewId view,
    uint32_t frameSlot,
    uint32_t hzbWriteIndex,
    std::string& log)
{
    if (hzbWriteIndex >= 2) {
        log = "GPUSceneSubsystem publishViewGpuResources received an invalid HZB write index";
        return makeError(Error::InvalidArgument);
    }
    const GPUSceneVisibleDrawSet* visible = scene_.visibleDrawSet(view, frameSlot);
    GPUSceneViewGpuResourcesView owned;
    if (visible == nullptr || !viewGpuResources(view, frameSlot, owned)) {
        log = "GPUSceneSubsystem publishViewGpuResources requires a prepared live View frame slot";
        return makeError(Error::InvalidArgument);
    }

    GPUSceneVisibleGpuResources published;
    published.sourceView = view;
    published.frameSlot = frameSlot;
    published.sourceDrawSetGeneration = scene_.drawSet().generation;
    published.sourceDrawSetRevision = scene_.drawSet().revision;
    published.instanceVisibilityStates = owned.instanceVisibilityStates;
    published.visibleInstanceIds = owned.visibleInstanceIds;
    published.visibleInstanceCounter = owned.visibleInstanceCounter;
    published.phases = owned.phases;
    published.hzb.history = owned.hzbHistory;
    published.hzb.width = owned.desc.hzbWidth;
    published.hzb.height = owned.desc.hzbHeight;
    published.hzb.mipCount = owned.desc.hzbMipCount;
    published.hzb.writeIndex = hzbWriteIndex;
    published.hzb.historyEpoch = visible->stats.hzbHistoryEpoch;
    published.hzb.valid = visible->stats.hzbValid;
    if (!scene_.setVisibleGpuResources(view, frameSlot, std::move(published))) {
        log = "GPUSceneSubsystem failed to publish its View GPU bundle";
        return makeError(Error::Failure);
    }
    return {};
}

void GPUSceneSubsystem::requestUpload(PendingUpload upload)
{
    if (static_cast<uint8_t>(upload) > static_cast<uint8_t>(pendingUpload_)) {
        pendingUpload_ = upload;
    }
}

void GPUSceneSubsystem::onWorldChanged(RenderWorld* world)
{
    world_ = world;
    sourceDirty_ = true;
}

const scene::Scene* GPUSceneSubsystem::effectiveSourceOverride() const
{
    return leasedSourceOverride_ != nullptr ? leasedSourceOverride_ : sourceOverride_;
}

const scene::Scene* GPUSceneSubsystem::sourceOverride() const
{
    return effectiveSourceOverride();
}

void GPUSceneSubsystem::sourceOverrideChanged(const scene::Scene* previousOverride)
{
    if (effectiveSourceOverride() == previousOverride) {
        return;
    }
    ++sourceOverrideRevision_;
    if (sourceOverrideRevision_ == 0) {
        sourceOverrideRevision_ = 1;
    }
    sourceDirty_ = true;
}

Result GPUSceneSubsystem::acquireSourceOverride(
    const scene::Scene* scene,
    GPUSceneSourceOverrideToken& token,
    std::string& log)
{
    token = {};
    if (scene == nullptr) {
        log = "GPUScene source override lease requires a non-null Scene";
        return makeError(Error::InvalidArgument);
    }
    if (leasedSourceOverride_ != nullptr && leasedSourceOverride_ != scene) {
        log = "GPUScene source override lease conflicts with a different Scene";
        return makeError(Error::InvalidArgument);
    }

    const scene::Scene* previousOverride = effectiveSourceOverride();
    uint64_t tokenValue = nextSourceOverrideToken_++;
    if (nextSourceOverrideToken_ == 0) {
        nextSourceOverrideToken_ = 1;
    }
    while (tokenValue == 0 || sourceOverrideLeases_.contains(tokenValue)) {
        tokenValue = nextSourceOverrideToken_++;
        if (nextSourceOverrideToken_ == 0) {
            nextSourceOverrideToken_ = 1;
        }
    }

    sourceOverrideLeases_.emplace(tokenValue, scene);
    leasedSourceOverride_ = scene;
    token.value = tokenValue;
    sourceOverrideChanged(previousOverride);
    return {};
}

bool GPUSceneSubsystem::releaseSourceOverride(GPUSceneSourceOverrideToken token)
{
    const auto lease = sourceOverrideLeases_.find(token.value);
    if (!token || lease == sourceOverrideLeases_.end()) {
        return false;
    }

    const scene::Scene* previousOverride = effectiveSourceOverride();
    sourceOverrideLeases_.erase(lease);
    if (sourceOverrideLeases_.empty()) {
        leasedSourceOverride_ = nullptr;
    }
    sourceOverrideChanged(previousOverride);
    return true;
}

void GPUSceneSubsystem::setSourceOverride(const scene::Scene* scene)
{
    if (sourceOverride_ == scene) {
        return;
    }
    const scene::Scene* previousOverride = effectiveSourceOverride();
    sourceOverride_ = scene;
    sourceOverrideChanged(previousOverride);
}

bool GPUSceneSubsystem::clearSourceOverride(const scene::Scene* scene)
{
    if (sourceOverride_ != scene) {
        return false;
    }
    setSourceOverride(nullptr);
    return true;
}

Result GPUSceneSubsystem::beginFrame(
    const RenderSubsystemFrameContext& context,
    RenderChangeBits& changes,
    std::string& log)
{
    world_ = context.world;
    currentFrameIndex_ = context.frameIndex;
    currentFrameSlot_ = context.frameSlot;
    const scene::Scene* overrideScene = effectiveSourceOverride();
    const scene::Scene* currentScene = overrideScene != nullptr
        ? overrideScene
        : (world_ != nullptr ? world_->scene() : nullptr);
    const uint64_t externalRevision = overrideScene != nullptr
        ? sourceOverrideRevision_
        : (world_ != nullptr ? world_->sceneRevision() : 0);

    if (sourceDirty_ || currentScene != sourceScene_) {
        sourceDirty_ = false;
        sourceScene_ = currentScene;
        if (sourceScene_ == nullptr) {
            const bool hadSource = scene_.stats().geometryCount != 0 ||
                scene_.stats().materialCount != 0 || scene_.stats().instanceCount != 0;
            scene_.clearSource();
            if (hadSource) {
                requestUpload(PendingUpload::Full);
                changes |= RenderChangeBits::Geometry |
                    RenderChangeBits::Material |
                    RenderChangeBits::InvalidateTemporalHistory;
            }
            return {};
        }

        Result result = scene_.rebuild(
            GPUSceneSourceView::fromScene(*sourceScene_, externalRevision),
            log);
        if (!result) {
            return result;
        }
        requestUpload(PendingUpload::Full);
        changes |= RenderChangeBits::Geometry |
            RenderChangeBits::Material |
            RenderChangeBits::InvalidateTemporalHistory;
        return {};
    }

    if (sourceScene_ == nullptr) {
        return {};
    }

    const GPUSceneSourceView source =
        GPUSceneSourceView::fromScene(*sourceScene_, externalRevision);
    const GPUSceneSyncResult syncResult = scene_.sync(source);
    if (syncResult == GPUSceneSyncResult::RebuildRequired) {
        Result result = scene_.rebuild(source, log);
        if (!result) {
            return result;
        }
        requestUpload(PendingUpload::Full);
        changes |= RenderChangeBits::Geometry |
            RenderChangeBits::Material |
            RenderChangeBits::InvalidateTemporalHistory;
    } else if (syncResult == GPUSceneSyncResult::Updated) {
        requestUpload(PendingUpload::Instances);
        changes |= RenderChangeBits::Geometry |
            RenderChangeBits::InvalidateTemporalHistory;
    } else if (syncResult == GPUSceneSyncResult::HistoryUpdated) {
        requestUpload(PendingUpload::Instances);
    }
    return {};
}

Result GPUSceneSubsystem::recordPreGraph(
    const RenderSubsystemFrameContext& context,
    std::string& log)
{
    if (context.commandBuffer == nullptr || device_ == nullptr || host_ == nullptr) {
        log = "GPUSceneSubsystem recordPreGraph requires an initialized subsystem and command buffer";
        return makeError(Error::InvalidArgument);
    }
    const GPUSceneDrawSet& drawSet = scene_.drawSet();
    if (drawSet.generation == 0 || drawSet.revision == 0) {
        scene_.invalidateGpuResources();
        if (gpuResources_ != nullptr) {
            context.host.retire(std::static_pointer_cast<void>(gpuResources_));
            gpuResources_.reset();
        }
        gpuUploadStats_.drawSetGeneration = 0;
        gpuUploadStats_.drawSetRevision = 0;
        rasterDrawLayout_ = {};
        return {};
    }
    PendingUpload upload = pendingUpload_;
    if (gpuResources_ == nullptr || gpuResources_->generation != drawSet.generation) {
        upload = PendingUpload::Full;
    } else if (gpuResources_->revision != drawSet.revision &&
        upload == PendingUpload::None) {
        upload = PendingUpload::Instances;
    }

    Result result;
    if (upload == PendingUpload::Full) {
        result = uploadFullScene(context, log);
    } else if (upload == PendingUpload::Instances) {
        result = uploadInstances(context, log);
    } else if (!scene_.globalBufferViews().validFor(
                   drawSet.generation,
                   drawSet.revision)) {
        if (gpuResources_ == nullptr ||
            !scene_.setGlobalBufferViews(gpuResources_->views())) {
            log = "GPUSceneSubsystem could not republish its current global buffer views";
            return makeError(Error::Failure);
        }
    }
    if (result) {
        pendingUpload_ = PendingUpload::None;
    }
    return result;
}

Result GPUSceneSubsystem::uploadFullScene(
    const RenderSubsystemFrameContext& context,
    std::string& log)
{
    GPUSceneCpuUploadData data = buildGpuUploadData(scene_);
    if (!visibilityRecordCapacityFitsId(data.meshletDraws.size())) {
        log = "GPUScene resident meshlet draw count exceeds the common visibility ID record limit";
        return makeError(Error::InvalidArgument);
    }
    // Mandatory tables always retain one sentinel element so consumers can
    // bind a stable non-zero structured buffer for an empty DrawSet.
    if (data.geometries.empty()) {
        data.geometries.emplace_back();
    }
    if (data.materials.empty()) {
        data.materials.emplace_back();
    }
    if (data.instances.empty()) {
        data.instances.emplace_back();
    }
    if (data.drawKeys.empty()) {
        data.drawKeys.emplace_back();
    }

    auto next = std::make_shared<GpuResources>();
    next->generation = scene_.drawSet().generation;
    next->revision = scene_.drawSet().revision;
    auto uploads = std::make_shared<UploadResources>();
    struct PendingCopy {
        Buffer* source = nullptr;
        Buffer* destination = nullptr;
        uint64_t byteSize = 0;
    };
    std::vector<PendingCopy> copies;
    copies.reserve(kGPUSceneGlobalBufferKindCount);

    auto createResource = [&]<typename T>(
                              GPUSceneGlobalBufferKind kind,
                              const std::vector<T>& records,
                              const char* label) -> Result {
        if (records.empty()) {
            return {};
        }
        if (records.size() > std::numeric_limits<uint64_t>::max() / sizeof(T)) {
            log = std::string("GPUScene ") + label + " buffer size overflow";
            return makeError(Error::InvalidArgument);
        }
        const uint64_t byteSize = static_cast<uint64_t>(records.size()) * sizeof(T);
        GpuBufferResource& resource = next->resource(kind);
        Result result = device_->createBuffer(
            BufferDesc{
                .size = byteSize,
                .structureStride = sizeof(T),
                .usage = BufferUsageBits::Storage |
                    BufferUsageBits::TransferSource |
                    BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::Device,
                .queueAccess = QueueAccessBits::Graphics |
                    QueueAccessBits::Compute,
            },
            resource.buffer);
        if (!result || resource.buffer == nullptr) {
            log = std::string("GPUScene failed to create device-local ") + label +
                " buffer: " + resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        resource.byteSize = byteSize;
        resource.structureStride = sizeof(T);
        result = device_->createBufferView(
            *resource.buffer,
            BufferViewDesc{
                .type = BufferViewType::Structured,
                .offset = 0,
                .size = byteSize,
                .structureStride = sizeof(T),
            },
            resource.view);
        if (!result || resource.view == nullptr) {
            log = std::string("GPUScene failed to create ") + label +
                " buffer view: " + resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }

        std::unique_ptr<Buffer> staging;
        result = device_->createBuffer(
            BufferDesc{
                .size = byteSize,
                .structureStride = sizeof(T),
                .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload,
                .queueAccess = QueueAccessBits::Graphics,
            },
            staging);
        if (!result || staging == nullptr) {
            log = std::string("GPUScene failed to create ") + label +
                " staging buffer: " + resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        void* mapped = staging->map();
        if (mapped == nullptr) {
            log = std::string("GPUScene failed to map ") + label + " staging buffer";
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, records.data(), static_cast<size_t>(byteSize));
        staging->flush(0, byteSize);
        staging->unmap();
        copies.push_back(PendingCopy{
            .source = staging.get(),
            .destination = resource.buffer.get(),
            .byteSize = byteSize,
        });
        uploads->stagingBuffers.push_back(std::move(staging));
        return {};
    };

    Result result = createResource(
        GPUSceneGlobalBufferKind::Geometries, data.geometries, "geometry");
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::Materials, data.materials, "material");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::Instances, data.instances, "instance");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::DrawKeys, data.drawKeys, "draw-key");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::DrawInstanceIds,
            data.drawInstanceIds,
            "draw-instance ID");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::Vertices, data.vertices, "vertex payload");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::Indices, data.indices, "index payload");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::Meshlets, data.meshlets, "meshlet payload");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::MeshletDraws,
            data.meshletDraws,
            "meshlet draw payload");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::MeshletVertices,
            data.meshletVertices,
            "meshlet vertex payload");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::MeshletTriangleWords,
            data.meshletTriangleWords,
            "meshlet triangle payload");
    }
    if (result) {
        result = createResource(
            GPUSceneGlobalBufferKind::DescriptorRemap,
            data.descriptorRemap,
            "descriptor remap");
    }
    if (!result) {
        return result;
    }

    const GPUSceneGlobalBufferViews views = next->views();
    if (!views.validFor(next->generation, next->revision)) {
        log = "GPUScene generated invalid global buffer views";
        return makeError(Error::Failure);
    }

    std::vector<BufferBarrierDesc> toTransfer;
    std::vector<BufferBarrierDesc> toRead;
    toTransfer.reserve(copies.size());
    toRead.reserve(copies.size());
    uint64_t uploadedBytes = 0;
    for (const PendingCopy& copy : copies) {
        toTransfer.push_back(BufferBarrierDesc{
            .buffer = copy.destination,
            .before = ResourceState::Undefined,
            .after = ResourceState::TransferDestination,
            .offset = 0,
            .size = copy.byteSize,
        });
        toRead.push_back(BufferBarrierDesc{
            .buffer = copy.destination,
            .before = ResourceState::TransferDestination,
            .after = ResourceState::ShaderRead,
            .offset = 0,
            .size = copy.byteSize,
        });
        uploadedBytes += copy.byteSize;
    }
    if (!toTransfer.empty()) {
        context.commandBuffer->barrier(BarrierDesc{
            .buffers = toTransfer.data(),
            .bufferCount = gpuCount(toTransfer.size()),
        });
    }
    for (const PendingCopy& copy : copies) {
        context.commandBuffer->copyBuffer(BufferCopyDesc{
            .source = copy.source,
            .destination = copy.destination,
            .sourceOffset = 0,
            .destinationOffset = 0,
            .size = copy.byteSize,
        });
    }
    if (!toRead.empty()) {
        context.commandBuffer->barrier(BarrierDesc{
            .buffers = toRead.data(),
            .bufferCount = gpuCount(toRead.size()),
        });
    }

    if (!scene_.setGlobalBufferViews(views)) {
        // Commands already reference these resources. Retire them with the
        // staging bundle even on this defensive publication failure path.
        context.host.retire(std::static_pointer_cast<void>(uploads));
        context.host.retire(std::static_pointer_cast<void>(next));
        log = "GPUScene failed to publish uploaded global buffer views";
        return makeError(Error::Failure);
    }

    context.host.retire(std::static_pointer_cast<void>(uploads));
    if (gpuResources_ != nullptr) {
        context.host.retire(std::static_pointer_cast<void>(gpuResources_));
    }
    gpuResources_ = std::move(next);
    rasterDrawLayout_ = std::move(data.rasterDrawLayout);
    ++gpuUploadStats_.fullUploadCount;
    gpuUploadStats_.uploadedByteCount += uploadedBytes;
    gpuUploadStats_.drawSetGeneration = gpuResources_->generation;
    gpuUploadStats_.drawSetRevision = gpuResources_->revision;
    return {};
}

Result GPUSceneSubsystem::uploadInstances(
    const RenderSubsystemFrameContext& context,
    std::string& log)
{
    if (gpuResources_ == nullptr ||
        gpuResources_->generation != scene_.drawSet().generation) {
        return uploadFullScene(context, log);
    }
    std::vector<GPUSceneGpuInstanceRecord> instances = buildGpuInstanceRecords(scene_);
    if (instances.empty()) {
        instances.emplace_back();
    }
    const uint64_t byteSize =
        static_cast<uint64_t>(instances.size()) * sizeof(GPUSceneGpuInstanceRecord);
    GpuBufferResource& resource =
        gpuResources_->resource(GPUSceneGlobalBufferKind::Instances);
    if (resource.buffer == nullptr || resource.view == nullptr ||
        resource.byteSize != byteSize ||
        resource.structureStride != sizeof(GPUSceneGpuInstanceRecord)) {
        return uploadFullScene(context, log);
    }

    auto uploads = std::make_shared<UploadResources>();
    std::unique_ptr<Buffer> staging;
    Result result = device_->createBuffer(
        BufferDesc{
            .size = byteSize,
            .structureStride = sizeof(GPUSceneGpuInstanceRecord),
            .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics,
        },
        staging);
    if (!result || staging == nullptr) {
        log = "GPUScene failed to create instance staging buffer: ";
        log += resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }
    void* mapped = staging->map();
    if (mapped == nullptr) {
        log = "GPUScene failed to map instance staging buffer";
        return makeError(Error::Failure);
    }
    std::memcpy(mapped, instances.data(), static_cast<size_t>(byteSize));
    staging->flush(0, byteSize);
    staging->unmap();

    BufferBarrierDesc toTransfer{
        .buffer = resource.buffer.get(),
        .before = ResourceState::ShaderRead,
        .after = ResourceState::TransferDestination,
        .offset = 0,
        .size = byteSize,
    };
    context.commandBuffer->barrier(BarrierDesc{
        .buffers = &toTransfer,
        .bufferCount = 1,
    });
    context.commandBuffer->copyBuffer(BufferCopyDesc{
        .source = staging.get(),
        .destination = resource.buffer.get(),
        .sourceOffset = 0,
        .destinationOffset = 0,
        .size = byteSize,
    });
    BufferBarrierDesc toRead{
        .buffer = resource.buffer.get(),
        .before = ResourceState::TransferDestination,
        .after = ResourceState::ShaderRead,
        .offset = 0,
        .size = byteSize,
    };
    context.commandBuffer->barrier(BarrierDesc{
        .buffers = &toRead,
        .bufferCount = 1,
    });

    uploads->stagingBuffers.push_back(std::move(staging));
    context.host.retire(std::static_pointer_cast<void>(uploads));
    gpuResources_->revision = scene_.drawSet().revision;
    rasterDrawLayout_.drawSetRevision = scene_.drawSet().revision;
    if (!scene_.setGlobalBufferViews(gpuResources_->views())) {
        log = "GPUScene failed to publish incrementally updated instance buffer views";
        return makeError(Error::Failure);
    }
    ++gpuUploadStats_.instanceUploadCount;
    gpuUploadStats_.uploadedByteCount += byteSize;
    gpuUploadStats_.drawSetGeneration = gpuResources_->generation;
    gpuUploadStats_.drawSetRevision = gpuResources_->revision;
    return {};
}

Result GPUSceneSubsystem::createBindings(
    BindlessHeap& heap,
    GPUSceneConsumerBindings& bindings,
    std::string& log) const
{
    releaseBindings(heap, bindings);
    const GPUSceneGlobalBufferViews& views = scene_.globalBufferViews();
    const uint32_t generation = scene_.drawSet().generation;
    const uint64_t revision = scene_.drawSet().revision;
    if (!views.validFor(generation, revision)) {
        log = "GPUScene createBindings requires current global GPU buffer views";
        return makeError(Error::InvalidArgument);
    }
    const std::array<const GPUSceneBufferView*, kGPUSceneGlobalBufferKindCount> bufferViews{
        &views.geometries,
        &views.materials,
        &views.instances,
        &views.drawKeys,
        &views.drawInstanceIds,
        &views.vertices,
        &views.indices,
        &views.meshlets,
        &views.meshletDraws,
        &views.meshletVertices,
        &views.meshletTriangleWords,
        &views.descriptorRemap,
    };
    for (size_t index = 0; index < bufferViews.size(); ++index) {
        const GPUSceneBufferView& view = *bufferViews[index];
        if (view.buffer == nullptr && view.view == nullptr && view.size == 0) {
            continue;
        }
        if (view.view == nullptr) {
            log = "GPUScene createBindings encountered a buffer without a BufferView";
            releaseBindings(heap, bindings);
            return makeError(Error::Failure);
        }
        Result result = heap.allocateBuffer(bindings.buffers[index]);
        if (!result) {
            log = "GPUScene createBindings failed to allocate a bindless buffer handle: ";
            log += resultToString(result);
            releaseBindings(heap, bindings);
            return result;
        }
        result = heap.writeBufferView(bindings.buffers[index], *view.view);
        if (!result) {
            log = "GPUScene createBindings failed to write a bindless buffer view: ";
            log += resultToString(result);
            releaseBindings(heap, bindings);
            return result;
        }
    }
    bindings.drawSetGeneration = generation;
    bindings.drawSetRevision = revision;
    if (!bindings.validFor(views)) {
        log = "GPUScene createBindings produced an incomplete binding set";
        releaseBindings(heap, bindings);
        return makeError(Error::Failure);
    }
    return {};
}

void GPUSceneSubsystem::releaseBindings(
    BindlessHeap& heap,
    GPUSceneConsumerBindings& bindings) const
{
    for (BindlessHandle handle : bindings.buffers) {
        if (handle.valid()) {
            heap.release(handle);
        }
    }
    bindings = {};
}

void GPUSceneSubsystem::shutdown()
{
    scene_.invalidateGpuResources();
    viewGpuResources_.clear();
    gpuResources_.reset();
    rasterDrawLayout_ = {};
    scene_.shutdown();
    device_ = nullptr;
    host_ = nullptr;
    world_ = nullptr;
    sourceScene_ = nullptr;
    sourceOverride_ = nullptr;
    leasedSourceOverride_ = nullptr;
    sourceOverrideLeases_.clear();
    currentFrameIndex_ = 0;
    currentFrameSlot_ = 0;
    frameSlotCount_ = 1;
    sourceOverrideRevision_ = 0;
    nextSourceOverrideToken_ = 1;
    nextViewGpuResourceAllocationId_ = 1;
    gpuUploadStats_ = {};
    pendingUpload_ = PendingUpload::Full;
    sourceDirty_ = true;
}

Result GPUSceneSubsystem::recordCull(
    CommandBuffer& commandBuffer,
    GPUSceneViewId view,
    uint32_t frameSlot,
    const GPUSceneCullRecordDesc& desc,
    std::string& log)
{
    const GPUSceneVisibleDrawSet* visible = scene_.visibleDrawSet(view, frameSlot);
    const uint32_t generation = scene_.drawSet().generation;
    const uint64_t revision = scene_.drawSet().revision;
    if (visible == nullptr ||
        visible->gpu.sourceView != view ||
        visible->gpu.frameSlot != frameSlot ||
        !visible->gpu.validFor(generation, revision)) {
        log = "GPUScene recordCull requires a live View with a prepared current-revision frame slot";
        return makeError(Error::InvalidArgument);
    }
    const size_t phaseIndex = static_cast<size_t>(desc.phase);
    if (phaseIndex >= kGPUSceneCullPhaseCount ||
        desc.bindlessHeap == nullptr ||
        desc.resetPipeline == nullptr ||
        desc.instanceCullPipeline == nullptr ||
        desc.compactPipeline == nullptr ||
        desc.pushData == nullptr ||
        desc.pushDataSize == 0) {
        log = "GPUScene recordCull received an incomplete culling description";
        return makeError(Error::InvalidArgument);
    }

    const GPUSceneCullPhaseGpuView& phase = visible->gpu.phases[phaseIndex];
    Buffer* indirectBuffer = phase.buckets.front().indirectArguments.buffer;
    Buffer* instanceVisibilityBuffer = visible->gpu.instanceVisibilityStates.buffer;
    Buffer* visibleMeshletBuffer = phase.visibleMeshletIds.buffer;
    if (indirectBuffer == nullptr ||
        instanceVisibilityBuffer == nullptr ||
        visibleMeshletBuffer == nullptr) {
        log = "GPUScene recordCull found an incomplete VisibleDrawSet GPU bundle";
        return makeError(Error::InvalidArgument);
    }
    for (const GPUSceneBucketGpuView& bucket : phase.buckets) {
        if (bucket.indirectArguments.buffer != indirectBuffer ||
            bucket.overflow.buffer != indirectBuffer) {
            log = "GPUScene recordCull requires all bucket arguments and overflow counters in one buffer";
            return makeError(Error::InvalidArgument);
        }
    }

    commandBuffer.bindBindlessHeap(*desc.bindlessHeap);
    commandBuffer.pushBindlessData(desc.pushData, desc.pushDataSize);
    commandBuffer.bindComputePipeline(*desc.resetPipeline);
    commandBuffer.dispatch(1, 1, 1);

    std::vector<BufferBarrierDesc> resetBarriers;
    resetBarriers.reserve(2);
    resetBarriers.push_back(BufferBarrierDesc{
        .buffer = indirectBuffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    });
    if (visible->gpu.visibleInstanceCounter.buffer != nullptr) {
        resetBarriers.push_back(BufferBarrierDesc{
            .buffer = visible->gpu.visibleInstanceCounter.buffer,
            .before = ResourceState::General,
            .after = ResourceState::General,
        });
    }
    commandBuffer.barrier(BarrierDesc{
        .buffers = resetBarriers.data(),
        .bufferCount = gpuCount(resetBarriers.size()),
    });

    commandBuffer.pushBindlessData(desc.pushData, desc.pushDataSize);
    commandBuffer.bindComputePipeline(*desc.instanceCullPipeline);
    commandBuffer.dispatch(desc.instanceGroupCountX, 1, 1);

    std::vector<BufferBarrierDesc> cullBarriers;
    cullBarriers.reserve(4);
    cullBarriers.push_back(BufferBarrierDesc{
        .buffer = indirectBuffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    });
    cullBarriers.push_back(BufferBarrierDesc{
        .buffer = instanceVisibilityBuffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    });
    if (visible->gpu.visibleInstanceIds.buffer != nullptr) {
        cullBarriers.push_back(BufferBarrierDesc{
            .buffer = visible->gpu.visibleInstanceIds.buffer,
            .before = ResourceState::General,
            .after = ResourceState::General,
        });
    }
    if (visible->gpu.visibleInstanceCounter.buffer != nullptr) {
        cullBarriers.push_back(BufferBarrierDesc{
            .buffer = visible->gpu.visibleInstanceCounter.buffer,
            .before = ResourceState::General,
            .after = ResourceState::General,
        });
    }
    commandBuffer.barrier(BarrierDesc{
        .buffers = cullBarriers.data(),
        .bufferCount = gpuCount(cullBarriers.size()),
    });

    commandBuffer.pushBindlessData(desc.pushData, desc.pushDataSize);
    commandBuffer.bindComputePipeline(*desc.compactPipeline);
    commandBuffer.dispatch(desc.meshletGroupCountX, 1, 1);

    std::vector<BufferBarrierDesc> compactBarriers;
    compactBarriers.reserve(4);
    compactBarriers.push_back(BufferBarrierDesc{
        .buffer = visibleMeshletBuffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    });
    compactBarriers.push_back(BufferBarrierDesc{
        .buffer = indirectBuffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    });
    if (visible->gpu.visibleInstanceIds.buffer != nullptr) {
        compactBarriers.push_back(BufferBarrierDesc{
            .buffer = visible->gpu.visibleInstanceIds.buffer,
            .before = ResourceState::General,
            .after = ResourceState::General,
        });
    }
    if (visible->gpu.visibleInstanceCounter.buffer != nullptr) {
        compactBarriers.push_back(BufferBarrierDesc{
            .buffer = visible->gpu.visibleInstanceCounter.buffer,
            .before = ResourceState::General,
            .after = ResourceState::General,
        });
    }
    commandBuffer.barrier(BarrierDesc{
        .buffers = compactBarriers.data(),
        .bufferCount = gpuCount(compactBarriers.size()),
    });
    return {};
}

Result GPUSceneSubsystem::recordInstanceCull(
    CommandBuffer& commandBuffer,
    GPUSceneViewId view,
    uint32_t frameSlot,
    const GPUSceneInstanceCullRecordDesc& desc,
    std::string& log)
{
    const GPUSceneVisibleDrawSet* visible = scene_.visibleDrawSet(view, frameSlot);
    const uint32_t generation = scene_.drawSet().generation;
    const uint64_t revision = scene_.drawSet().revision;
    if (visible == nullptr ||
        visible->gpu.sourceView != view ||
        visible->gpu.frameSlot != frameSlot ||
        !visible->gpu.validFor(generation, revision)) {
        log = "GPUScene recordInstanceCull requires a live View with a prepared current-revision frame slot";
        return makeError(Error::InvalidArgument);
    }
    const size_t phaseIndex = static_cast<size_t>(desc.phase);
    if (phaseIndex >= kGPUSceneCullPhaseCount ||
        desc.bindlessHeap == nullptr ||
        desc.resetPipeline == nullptr ||
        desc.instanceCullPipeline == nullptr ||
        desc.pushData == nullptr ||
        desc.pushDataSize == 0 ||
        desc.instanceGroupCountX == 0 ||
        visible->gpu.instanceVisibilityStates.buffer == nullptr ||
        visible->gpu.visibleInstanceCounter.buffer == nullptr) {
        log = "GPUScene recordInstanceCull received an incomplete culling description";
        return makeError(Error::InvalidArgument);
    }

    commandBuffer.bindBindlessHeap(*desc.bindlessHeap);
    commandBuffer.pushBindlessData(desc.pushData, desc.pushDataSize);
    commandBuffer.bindComputePipeline(*desc.resetPipeline);
    commandBuffer.dispatch(1u, 1u, 1u);

    BufferBarrierDesc resetBarrier{
        .buffer = visible->gpu.visibleInstanceCounter.buffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    };
    commandBuffer.barrier(BarrierDesc{
        .buffers = &resetBarrier,
        .bufferCount = 1,
    });

    commandBuffer.pushBindlessData(desc.pushData, desc.pushDataSize);
    commandBuffer.bindComputePipeline(*desc.instanceCullPipeline);
    commandBuffer.dispatch(desc.instanceGroupCountX, 1u, 1u);

    std::array<BufferBarrierDesc, 3> barriers{};
    uint32_t barrierCount = 0;
    barriers[barrierCount++] = BufferBarrierDesc{
        .buffer = visible->gpu.instanceVisibilityStates.buffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    };
    if (visible->gpu.visibleInstanceIds.buffer != nullptr) {
        barriers[barrierCount++] = BufferBarrierDesc{
            .buffer = visible->gpu.visibleInstanceIds.buffer,
            .before = ResourceState::General,
            .after = ResourceState::General,
        };
    }
    barriers[barrierCount++] = BufferBarrierDesc{
        .buffer = visible->gpu.visibleInstanceCounter.buffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    };
    commandBuffer.barrier(BarrierDesc{
        .buffers = barriers.data(),
        .bufferCount = barrierCount,
    });
    return {};
}

Result GPUSceneSubsystem::recordBuildHzb(
    CommandBuffer& commandBuffer,
    GPUSceneViewId view,
    uint32_t frameSlot,
    const GPUSceneHzbRecordDesc& desc,
    std::string& log)
{
    const GPUSceneVisibleDrawSet* visible = scene_.visibleDrawSet(view, frameSlot);
    const uint32_t generation = scene_.drawSet().generation;
    const uint64_t revision = scene_.drawSet().revision;
    if (visible == nullptr ||
        visible->gpu.sourceView != view ||
        visible->gpu.frameSlot != frameSlot ||
        !visible->gpu.validFor(generation, revision)) {
        log = "GPUScene recordBuildHzb requires a live View with a prepared current-revision frame slot";
        return makeError(Error::InvalidArgument);
    }
    if (desc.bindlessHeap == nullptr ||
        desc.pipeline == nullptr ||
        desc.dispatches.empty()) {
        log = "GPUScene recordBuildHzb received an incomplete HZB description";
        return makeError(Error::InvalidArgument);
    }

    const GPUSceneHzbGpuView& hzb = visible->gpu.hzb;
    Buffer* writeBuffer = hzb.history[hzb.writeIndex].buffer;
    if (writeBuffer == nullptr || desc.dispatches.size() != hzb.mipCount) {
        log = "GPUScene recordBuildHzb dispatch count does not match the View HZB mip chain";
        return makeError(Error::InvalidArgument);
    }
    for (const GPUSceneComputeDispatchDesc& dispatch : desc.dispatches) {
        if (dispatch.pushData == nullptr ||
            dispatch.pushDataSize == 0 ||
            dispatch.groupCountX == 0 ||
            dispatch.groupCountY == 0 ||
            dispatch.groupCountZ == 0) {
            log = "GPUScene recordBuildHzb received an invalid mip dispatch";
            return makeError(Error::InvalidArgument);
        }
    }

    commandBuffer.bindBindlessHeap(*desc.bindlessHeap);
    const BufferBarrierDesc writeBarrier{
        .buffer = writeBuffer,
        .before = ResourceState::General,
        .after = ResourceState::General,
    };
    commandBuffer.barrier(BarrierDesc{
        .buffers = &writeBarrier,
        .bufferCount = 1,
    });
    for (const GPUSceneComputeDispatchDesc& dispatch : desc.dispatches) {
        commandBuffer.pushBindlessData(dispatch.pushData, dispatch.pushDataSize);
        commandBuffer.bindComputePipeline(*desc.pipeline);
        commandBuffer.dispatch(
            dispatch.groupCountX,
            dispatch.groupCountY,
            dispatch.groupCountZ);
        commandBuffer.barrier(BarrierDesc{
            .buffers = &writeBarrier,
            .bufferCount = 1,
        });
    }
    return {};
}

} // namespace metallic::render
