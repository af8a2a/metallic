#include "Runtime/Render/Subsystem/GPUScene.h"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace metallic::render {
namespace {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

size_t bucketIndex(GPUSceneDrawBucket bucket)
{
    return static_cast<size_t>(bucket);
}

void hashBytes(uint64_t& hash, const void* data, size_t byteSize)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < byteSize; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

template <typename T>
void hashValue(uint64_t& hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    hashBytes(hash, &value, sizeof(value));
}

void hashValue(uint64_t& hash, const float2& value)
{
    hashValue(hash, value.x);
    hashValue(hash, value.y);
}

void hashValue(uint64_t& hash, const float3& value)
{
    hashValue(hash, value.x);
    hashValue(hash, value.y);
    hashValue(hash, value.z);
}

void hashValue(uint64_t& hash, const float4& value)
{
    hashValue(hash, value.x);
    hashValue(hash, value.y);
    hashValue(hash, value.z);
    hashValue(hash, value.w);
}

void hashString(uint64_t& hash, const std::string& value)
{
    hashValue(hash, value.size());
    hashBytes(hash, value.data(), value.size());
}

template <typename T>
void hashVector(uint64_t& hash, const std::vector<T>& values)
{
    hashValue(hash, values.size());
    if constexpr (
        std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>) {
        for (const T& value : values) {
            hashValue(hash, value);
        }
    } else {
        hashBytes(hash, values.data(), values.size() * sizeof(T));
    }
}

void hashBounds(uint64_t& hash, const scene::Bounds& bounds)
{
    hashValue(hash, bounds.min);
    hashValue(hash, bounds.max);
    hashValue(hash, bounds.valid);
}

void hashTextureInfo(uint64_t& hash, const scene::RenderTextureInfo& texture)
{
    hashValue(hash, texture.textureIndex);
    hashValue(hash, texture.texCoord);
    hashValue(hash, texture.uvTransform);
}

uint64_t geometryFingerprint(const scene::RenderPrimitive& primitive)
{
    uint64_t hash = kFnvOffsetBasis;
    hashValue(hash, primitive.mode);
    hashValue(hash, primitive.vertexCount);
    hashValue(hash, primitive.indexCount);
    hashValue(hash, primitive.triangleCount);
    hashBounds(hash, primitive.localBounds);
    hashVector(hash, primitive.positions);
    hashVector(hash, primitive.normals);
    hashVector(hash, primitive.tangents);
    hashVector(hash, primitive.texcoords0);
    hashValue(hash, primitive.hasAuthoredNormals);
    hashValue(hash, primitive.hasAuthoredTangents);
    hashVector(hash, primitive.indices);
    hashVector(hash, primitive.meshletClusters);
    hashVector(hash, primitive.meshletVertices);
    hashVector(hash, primitive.meshletTriangles);
    hashVector(hash, primitive.meshletLodLevels);
    hashVector(hash, primitive.meshletLodGroups);
    hashVector(hash, primitive.meshletLodClusters);
    hashVector(hash, primitive.meshletLodVertices);
    hashVector(hash, primitive.meshletLodTriangles);
    return hash;
}

uint64_t sourcePrimitiveFingerprint(const scene::RenderPrimitive& primitive)
{
    uint64_t hash = geometryFingerprint(primitive);
    hashValue(hash, primitive.meshIndex);
    hashValue(hash, primitive.primitiveIndex);
    hashValue(hash, primitive.materialIndex);
    return hash;
}

template <typename T>
bool sameVectorBytes(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return lhs.size() == rhs.size() &&
        (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0);
}

bool sameBounds(const scene::Bounds& lhs, const scene::Bounds& rhs)
{
    return std::memcmp(&lhs.min, &rhs.min, sizeof(lhs.min)) == 0 &&
        std::memcmp(&lhs.max, &rhs.max, sizeof(lhs.max)) == 0 &&
        lhs.valid == rhs.valid;
}

bool sameGeometryPayload(
    const scene::RenderPrimitive& lhs,
    const scene::RenderPrimitive& rhs)
{
    return lhs.mode == rhs.mode &&
        lhs.vertexCount == rhs.vertexCount &&
        lhs.indexCount == rhs.indexCount &&
        lhs.triangleCount == rhs.triangleCount &&
        sameBounds(lhs.localBounds, rhs.localBounds) &&
        sameVectorBytes(lhs.positions, rhs.positions) &&
        sameVectorBytes(lhs.normals, rhs.normals) &&
        sameVectorBytes(lhs.tangents, rhs.tangents) &&
        sameVectorBytes(lhs.texcoords0, rhs.texcoords0) &&
        lhs.hasAuthoredNormals == rhs.hasAuthoredNormals &&
        lhs.hasAuthoredTangents == rhs.hasAuthoredTangents &&
        sameVectorBytes(lhs.indices, rhs.indices) &&
        sameVectorBytes(lhs.meshletClusters, rhs.meshletClusters) &&
        sameVectorBytes(lhs.meshletVertices, rhs.meshletVertices) &&
        sameVectorBytes(lhs.meshletTriangles, rhs.meshletTriangles) &&
        sameVectorBytes(lhs.meshletLodLevels, rhs.meshletLodLevels) &&
        sameVectorBytes(lhs.meshletLodGroups, rhs.meshletLodGroups) &&
        sameVectorBytes(lhs.meshletLodClusters, rhs.meshletLodClusters) &&
        sameVectorBytes(lhs.meshletLodVertices, rhs.meshletLodVertices) &&
        sameVectorBytes(lhs.meshletLodTriangles, rhs.meshletLodTriangles);
}

uint64_t materialFingerprint(const scene::RenderMaterial& material)
{
    uint64_t hash = kFnvOffsetBasis;
    hashString(hash, material.name);
    hashValue(hash, material.baseColorFactor);
    hashValue(hash, material.metallicFactor);
    hashValue(hash, material.roughnessFactor);
    hashValue(hash, material.emissiveFactor);
    hashValue(hash, material.alphaCutoff);
    hashString(hash, material.alphaMode);
    hashValue(hash, material.doubleSided);
    hashValue(hash, material.normalTextureScale);
    hashValue(hash, material.occlusionTextureStrength);
    hashValue(hash, material.transmissionFactor);
    hashValue(hash, material.ior);
    hashValue(hash, material.thicknessFactor);
    hashValue(hash, material.attenuationDistance);
    hashValue(hash, material.attenuationColor);
    hashValue(hash, material.diffuseTransmissionFactor);
    hashValue(hash, material.diffuseTransmissionColor);
    hashValue(hash, material.rtxcrHair);
    hashValue(hash, material.rtxcrHairBaseColor);
    hashValue(hash, material.rtxcrHairMelanin);
    hashValue(hash, material.rtxcrHairMelaninRedness);
    hashValue(hash, material.rtxcrHairLongitudinalRoughness);
    hashValue(hash, material.rtxcrHairAzimuthalRoughness);
    hashValue(hash, material.rtxcrHairIor);
    hashValue(hash, material.rtxcrHairCuticleAngleDegrees);
    hashValue(hash, material.rtxcrHairDiffuseReflectionWeight);
    hashValue(hash, material.rtxcrHairDiffuseReflectionTint);
    hashTextureInfo(hash, material.baseColorTexture);
    hashTextureInfo(hash, material.metallicRoughnessTexture);
    hashTextureInfo(hash, material.normalTexture);
    hashTextureInfo(hash, material.occlusionTexture);
    hashTextureInfo(hash, material.emissiveTexture);
    hashTextureInfo(hash, material.transmissionTexture);
    hashTextureInfo(hash, material.thicknessTexture);
    hashTextureInfo(hash, material.diffuseTransmissionTexture);
    hashTextureInfo(hash, material.diffuseTransmissionColorTexture);
    return hash;
}

uint64_t renderNodeTopologyFingerprint(const scene::RenderNode& node)
{
    uint64_t hash = kFnvOffsetBasis;
    const uint32_t object = static_cast<uint32_t>(node.object);
    hashValue(hash, object);
    hashValue(hash, node.nodeIndex);
    hashValue(hash, node.renderPrimitiveIndex);
    hashValue(hash, node.materialIndex);
    return hash;
}

bool validateDrawablePrimitive(
    const scene::RenderPrimitive& primitive,
    uint32_t sourceRenderPrimitiveIndex,
    GPUSceneInvalidPrimitiveDiagnostic& diagnostic)
{
    diagnostic = GPUSceneInvalidPrimitiveDiagnostic{
        .sourceRenderPrimitiveIndex = sourceRenderPrimitiveIndex,
    };
    if (primitive.mode != 4) {
        diagnostic.reason = GPUSceneInvalidPrimitiveReason::UnsupportedMode;
        return false;
    }
    if (primitive.positions.size() < 3) {
        diagnostic.reason = GPUSceneInvalidPrimitiveReason::InsufficientVertices;
        return false;
    }
    if (primitive.indexCount % 3 != 0 || primitive.indices.size() % 3 != 0) {
        diagnostic.reason = GPUSceneInvalidPrimitiveReason::IndexCountNotMultipleOfThree;
        return false;
    }
    for (uint64_t indexOffset = 0; indexOffset < primitive.indices.size(); ++indexOffset) {
        const uint32_t vertexIndex = primitive.indices[static_cast<size_t>(indexOffset)];
        if (vertexIndex >= primitive.positions.size()) {
            diagnostic.reason = GPUSceneInvalidPrimitiveReason::IndexOutOfRange;
            diagnostic.indexOffset = indexOffset;
            diagnostic.vertexIndex = vertexIndex;
            return false;
        }
    }
    return primitive.indices.empty() || primitive.indices.size() >= 3;
}

float4 localBoundingSphere(const scene::Bounds& bounds)
{
    if (!bounds.valid) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float3 center = bounds.center();
    return float4(center.x, center.y, center.z, std::max(bounds.radius(), 0.0f));
}

uint64_t geometrySourceKey(const scene::RenderPrimitive& primitive)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(primitive.meshIndex)) << 32u) |
        static_cast<uint32_t>(primitive.primitiveIndex);
}

bool hasDeduplicationKey(const scene::RenderPrimitive& primitive)
{
    return primitive.meshIndex >= 0 && primitive.primitiveIndex >= 0;
}

template <typename Id>
Id mappedId(const std::vector<Id>& mapping, uint32_t index)
{
    return index < mapping.size() ? mapping[index] : Id{};
}

bool normalizeBufferView(
    GPUSceneBufferView& view,
    uint32_t generation,
    uint64_t revision)
{
    if (view.generation == 0) {
        view.generation = generation;
    }
    if (view.revision == 0) {
        view.revision = revision;
    }
    return view.validFor(generation, revision);
}

bool emptyBufferView(const GPUSceneBufferView& view)
{
    return view.buffer == nullptr && view.view == nullptr && view.size == 0;
}

bool normalizeOptionalBufferView(
    GPUSceneBufferView& view,
    uint32_t generation,
    uint64_t revision)
{
    return emptyBufferView(view) || normalizeBufferView(view, generation, revision);
}

bool optionalBufferViewValidFor(
    const GPUSceneBufferView& view,
    uint32_t generation,
    uint64_t revision)
{
    return emptyBufferView(view) || view.validFor(generation, revision);
}

} // namespace

const char* gpuSceneDrawBucketName(GPUSceneDrawBucket bucket)
{
    switch (bucket) {
    case GPUSceneDrawBucket::OpaqueSingleSided:
        return "opaque-single-sided";
    case GPUSceneDrawBucket::OpaqueDoubleSided:
        return "opaque-double-sided";
    case GPUSceneDrawBucket::MaskedSingleSided:
        return "masked-single-sided";
    case GPUSceneDrawBucket::MaskedDoubleSided:
        return "masked-double-sided";
    case GPUSceneDrawBucket::Blend:
        return "blend";
    case GPUSceneDrawBucket::Count:
        break;
    }
    return "invalid";
}

GPUSceneDrawBucket classifyGPUSceneMaterial(const scene::RenderMaterial& material)
{
    if (material.alphaMode == "BLEND") {
        return GPUSceneDrawBucket::Blend;
    }
    if (material.alphaMode == "MASK") {
        return material.doubleSided
            ? GPUSceneDrawBucket::MaskedDoubleSided
            : GPUSceneDrawBucket::MaskedSingleSided;
    }
    return material.doubleSided
        ? GPUSceneDrawBucket::OpaqueDoubleSided
        : GPUSceneDrawBucket::OpaqueSingleSided;
}

GPUSceneSourceView GPUSceneSourceView::fromScene(
    const scene::Scene& scene,
    uint64_t externalRevision)
{
    const scene::SceneGraph& graph = scene.sceneGraph();
    return GPUSceneSourceView{
        .renderPrimitives = scene.renderPrimitives(),
        .renderNodes = scene.renderNodes(),
        .materials = scene.materials(),
        .lifetimeRevision = graph.lifetimeRevision(),
        .structuralRevision = graph.structuralRevision(),
        .contentRevision = scene.contentRevision(),
        .transformRevision = scene.transformRevision(),
        .visibilityRevision = scene.visibilityRevision(),
        .externalRevision = externalRevision,
    };
}

std::span<const GPUSceneInstanceId> GPUSceneDrawSet::instancesForBucket(
    GPUSceneDrawBucket bucket) const
{
    const size_t index = bucketIndex(bucket);
    return index < buckets.size() ? std::span<const GPUSceneInstanceId>(buckets[index])
                                  : std::span<const GPUSceneInstanceId>();
}

bool GPUSceneBufferView::valid() const
{
    if (buffer == nullptr || size == 0 || structureStride == 0 ||
        generation == 0 || revision == 0) {
        return false;
    }
    const BufferDesc& bufferDesc = buffer->desc();
    if (offset > bufferDesc.size || size > bufferDesc.size - offset ||
        size % structureStride != 0 ||
        (bufferDesc.structureStride != 0 &&
            bufferDesc.structureStride != structureStride)) {
        return false;
    }
    if (view != nullptr) {
        const BufferViewDesc& viewDesc = view->desc();
        if (offset < viewDesc.offset ||
            size > viewDesc.size ||
            offset - viewDesc.offset > viewDesc.size - size ||
            (viewDesc.structureStride != 0 &&
                viewDesc.structureStride != structureStride)) {
            return false;
        }
    }
    return true;
}

bool GPUSceneBufferView::validFor(
    uint32_t expectedGeneration,
    uint64_t expectedRevision) const
{
    return generation == expectedGeneration && revision == expectedRevision && valid();
}

bool GPUSceneGlobalBufferViews::validFor(
    uint32_t generation,
    uint64_t revision) const
{
    return drawSetGeneration == generation && drawSetRevision == revision &&
        geometries.validFor(generation, revision) &&
        materials.validFor(generation, revision) &&
        instances.validFor(generation, revision) &&
        drawKeys.validFor(generation, revision) &&
        optionalBufferViewValidFor(drawInstanceIds, generation, revision) &&
        optionalBufferViewValidFor(vertices, generation, revision) &&
        optionalBufferViewValidFor(indices, generation, revision) &&
        optionalBufferViewValidFor(meshlets, generation, revision) &&
        optionalBufferViewValidFor(meshletDraws, generation, revision) &&
        optionalBufferViewValidFor(meshletVertices, generation, revision) &&
        optionalBufferViewValidFor(meshletTriangleWords, generation, revision) &&
        optionalBufferViewValidFor(descriptorRemap, generation, revision);
}

bool GPUSceneConsumerBindings::validFor(const GPUSceneGlobalBufferViews& views) const
{
    if (drawSetGeneration == 0 ||
        drawSetGeneration != views.drawSetGeneration ||
        !views.validFor(views.drawSetGeneration, views.drawSetRevision)) {
        return false;
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
        const bool required = !emptyBufferView(*bufferViews[index]);
        if (required != buffers[index].valid()) {
            return false;
        }
    }
    return true;
}

bool GPUSceneRasterDrawLayout::validFor(
    uint32_t generation,
    uint64_t revision) const
{
    return generation != 0 && revision != 0 &&
        drawSetGeneration == generation && drawSetRevision == revision;
}

bool GPUSceneVisibleGpuResources::validFor(
    uint32_t generation,
    uint64_t revision) const
{
    if (sourceDrawSetGeneration != generation ||
        sourceDrawSetRevision != revision ||
        !sourceView.valid() ||
        !instanceVisibilityStates.validFor(generation, revision) ||
        !optionalBufferViewValidFor(visibleInstanceIds, generation, revision) ||
        !optionalBufferViewValidFor(visibleInstanceCounter, generation, revision) ||
        hzb.width == 0 || hzb.height == 0 || hzb.mipCount == 0 ||
        hzb.writeIndex >= hzb.history.size()) {
        return false;
    }
    for (const GPUSceneCullPhaseGpuView& phase : phases) {
        if (!phase.visibleMeshletIds.validFor(generation, revision)) {
            return false;
        }
        const uint64_t visibleMeshletElementCount =
            phase.visibleMeshletIds.size / phase.visibleMeshletIds.structureStride;
        for (const GPUSceneBucketGpuView& bucket : phase.buckets) {
            if (bucket.visibleMeshletCapacity == 0 ||
                bucket.visibleMeshletOffset > visibleMeshletElementCount ||
                bucket.visibleMeshletCapacity >
                    visibleMeshletElementCount - bucket.visibleMeshletOffset ||
                !bucket.indirectArguments.validFor(generation, revision) ||
                !bucket.overflow.validFor(generation, revision)) {
                return false;
            }
        }
    }
    return std::ranges::all_of(
        hzb.history,
        [generation, revision](const GPUSceneBufferView& history) {
            return history.validFor(generation, revision);
        });
}

std::span<const GPUSceneInstanceId> GPUSceneVisibleDrawSet::instancesForBucket(
    GPUSceneDrawBucket bucket) const
{
    const size_t index = bucketIndex(bucket);
    return index < buckets.size() ? std::span<const GPUSceneInstanceId>(buckets[index])
                                  : std::span<const GPUSceneInstanceId>();
}

void GPUScene::setDefaultFrameSlotCount(uint32_t frameSlotCount)
{
    defaultFrameSlotCount_ = std::max(frameSlotCount, 1u);
}

uint32_t GPUScene::advanceGeneration(uint32_t generation)
{
    ++generation;
    return generation == 0 ? 1 : generation;
}

void GPUScene::invalidateSourceIds()
{
    geometryGeneration_ = advanceGeneration(geometryGeneration_);
    materialGeneration_ = advanceGeneration(materialGeneration_);
    instanceGeneration_ = advanceGeneration(instanceGeneration_);
    drawSetGeneration_ = advanceGeneration(drawSetGeneration_);
    geometries_.clear();
    geometrySourcePrimitives_.clear();
    materials_.clear();
    instances_.clear();
    geometryForRenderPrimitive_.clear();
    materialForSourceMaterial_.clear();
    instanceForRenderNode_.clear();
    instancesForObject_.clear();
    invalidPrimitiveDiagnostics_.clear();
    fallbackMaterial_ = {};
}

Result GPUScene::rebuild(const GPUSceneSourceView& source, std::string& log)
{
    if (source.renderPrimitives.size() > std::numeric_limits<uint32_t>::max() ||
        source.renderNodes.size() > std::numeric_limits<uint32_t>::max() ||
        source.materials.size() > std::numeric_limits<uint32_t>::max()) {
        log = "GPUScene source exceeds 32-bit generational ID capacity";
        return makeError(Error::OutOfMemory);
    }

    const uint64_t fullRebuildCount = stats_.fullRebuildCount + 1;
    const uint64_t incrementalSyncCount = stats_.incrementalSyncCount;
    const uint64_t unchangedSyncCount = stats_.unchangedSyncCount;
    const uint32_t viewCount = stats_.viewCount;
    invalidateSourceIds();
    stats_ = {};
    stats_.fullRebuildCount = fullRebuildCount;
    stats_.incrementalSyncCount = incrementalSyncCount;
    stats_.unchangedSyncCount = unchangedSyncCount;
    stats_.viewCount = viewCount;

    geometryForRenderPrimitive_.resize(source.renderPrimitives.size());
    materialForSourceMaterial_.resize(source.materials.size());
    instanceForRenderNode_.resize(source.renderNodes.size());
    primitiveFingerprints_.resize(source.renderPrimitives.size());
    materialFingerprints_.resize(source.materials.size());
    renderNodeTopologyFingerprints_.resize(source.renderNodes.size());

    std::vector<bool> drawablePrimitives(source.renderPrimitives.size(), false);
    for (uint32_t index = 0; index < source.renderPrimitives.size(); ++index) {
        GPUSceneInvalidPrimitiveDiagnostic diagnostic;
        drawablePrimitives[index] = validateDrawablePrimitive(
            source.renderPrimitives[index],
            index,
            diagnostic);
        if (drawablePrimitives[index]) {
            continue;
        }
        ++stats_.invalidPrimitiveCount;
        if (diagnostic.reason ==
            GPUSceneInvalidPrimitiveReason::IndexCountNotMultipleOfThree) {
            ++stats_.invalidIndexCountPrimitiveCount;
        } else if (diagnostic.reason == GPUSceneInvalidPrimitiveReason::IndexOutOfRange) {
            ++stats_.outOfRangeIndexPrimitiveCount;
        }
        invalidPrimitiveDiagnostics_.push_back(diagnostic);
    }

    materials_.reserve(source.materials.size() + 1);
    for (uint32_t index = 0; index < source.materials.size(); ++index) {
        const scene::RenderMaterial& material = source.materials[index];
        const GPUSceneMaterialId id{
            .index = static_cast<uint32_t>(materials_.size()),
            .generation = materialGeneration_,
        };
        const uint64_t fingerprint = materialFingerprint(material);
        materials_.push_back(GPUSceneMaterialRecord{
            .id = id,
            .sourceMaterialIndex = static_cast<int32_t>(index),
            .material = material,
            .bucket = classifyGPUSceneMaterial(material),
            .payloadFingerprint = fingerprint,
        });
        materialForSourceMaterial_[index] = id;
        materialFingerprints_[index] = fingerprint;
    }

    auto ensureFallbackMaterial = [&]() {
        if (fallbackMaterial_) {
            return fallbackMaterial_;
        }
        scene::RenderMaterial fallback;
        fallback.name = "GPUScene fallback material";
        const GPUSceneMaterialId id{
            .index = static_cast<uint32_t>(materials_.size()),
            .generation = materialGeneration_,
        };
        materials_.push_back(GPUSceneMaterialRecord{
            .id = id,
            .material = fallback,
            .bucket = classifyGPUSceneMaterial(fallback),
            .payloadFingerprint = materialFingerprint(fallback),
            .fallback = true,
        });
        fallbackMaterial_ = id;
        return id;
    };

    for (uint32_t index = 0; index < source.renderPrimitives.size(); ++index) {
        primitiveFingerprints_[index] = sourcePrimitiveFingerprint(source.renderPrimitives[index]);
    }
    for (uint32_t index = 0; index < source.renderNodes.size(); ++index) {
        renderNodeTopologyFingerprints_[index] =
            renderNodeTopologyFingerprint(source.renderNodes[index]);
    }

    std::unordered_map<uint64_t, std::vector<GPUSceneGeometryId>> geometryCandidates;
    geometries_.reserve(source.renderPrimitives.size());
    geometrySourcePrimitives_.reserve(source.renderPrimitives.size());
    instances_.reserve(source.renderNodes.size());

    auto geometryForPrimitive = [&](uint32_t sourceIndex) {
        GPUSceneGeometryId& mapped = geometryForRenderPrimitive_[sourceIndex];
        if (mapped) {
            return mapped;
        }
        const scene::RenderPrimitive& primitive = source.renderPrimitives[sourceIndex];
        const uint64_t fingerprint = geometryFingerprint(primitive);
        std::vector<GPUSceneGeometryId>* candidates = nullptr;
        if (hasDeduplicationKey(primitive)) {
            candidates = &geometryCandidates[geometrySourceKey(primitive)];
            for (GPUSceneGeometryId candidateId : *candidates) {
                const GPUSceneGeometryRecord& candidate = geometries_[candidateId.index];
                const scene::RenderPrimitive& canonical =
                    source.renderPrimitives[static_cast<uint32_t>(candidate.sourceRenderPrimitiveIndex)];
                if (candidate.payloadFingerprint == fingerprint &&
                    sameGeometryPayload(canonical, primitive)) {
                    mapped = candidateId;
                    ++stats_.deduplicatedGeometryCount;
                    return mapped;
                }
            }
            if (!candidates->empty()) {
                ++stats_.geometryPayloadConflictCount;
            }
        }

        mapped = GPUSceneGeometryId{
            .index = static_cast<uint32_t>(geometries_.size()),
            .generation = geometryGeneration_,
        };
        geometries_.push_back(GPUSceneGeometryRecord{
            .id = mapped,
            .sourceRenderPrimitiveIndex = static_cast<int32_t>(sourceIndex),
            .meshIndex = primitive.meshIndex,
            .primitiveIndex = primitive.primitiveIndex,
            .mode = primitive.mode,
            .vertexCount = primitive.vertexCount,
            .indexCount = primitive.indexCount,
            .triangleCount = primitive.triangleCount,
            .localBounds = primitive.localBounds,
            .payloadFingerprint = fingerprint,
        });
        geometrySourcePrimitives_.push_back(primitive);
        if (candidates != nullptr) {
            candidates->push_back(mapped);
        }
        return mapped;
    };

    for (uint32_t renderNodeIndex = 0; renderNodeIndex < source.renderNodes.size(); ++renderNodeIndex) {
        const scene::RenderNode& node = source.renderNodes[renderNodeIndex];
        if (node.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(node.renderPrimitiveIndex) >= source.renderPrimitives.size()) {
            ++stats_.skippedRenderNodeCount;
            continue;
        }
        const uint32_t primitiveIndex = static_cast<uint32_t>(node.renderPrimitiveIndex);
        const scene::RenderPrimitive& primitive = source.renderPrimitives[primitiveIndex];
        if (!drawablePrimitives[primitiveIndex]) {
            ++stats_.skippedRenderNodeCount;
            continue;
        }

        GPUSceneMaterialId materialId;
        int32_t sourceMaterialIndex = node.materialIndex;
        if (sourceMaterialIndex < 0) {
            sourceMaterialIndex = primitive.materialIndex;
        }
        if (sourceMaterialIndex >= 0 &&
            static_cast<size_t>(sourceMaterialIndex) < materialForSourceMaterial_.size()) {
            materialId = materialForSourceMaterial_[static_cast<size_t>(sourceMaterialIndex)];
        } else {
            sourceMaterialIndex = scene::kInvalidSceneIndex;
            materialId = ensureFallbackMaterial();
        }

        const GPUSceneGeometryId geometryId = geometryForPrimitive(primitiveIndex);
        const GPUSceneDrawBucket bucket = materials_[materialId.index].bucket;
        const GPUSceneInstanceId id{
            .index = static_cast<uint32_t>(instances_.size()),
            .generation = instanceGeneration_,
        };
        instances_.push_back(GPUSceneInstanceRecord{
            .id = id,
            .sourceRenderNodeIndex = static_cast<int32_t>(renderNodeIndex),
            .sourceObject = node.object,
            .sourceNodeIndex = node.nodeIndex,
            .sourceRenderPrimitiveIndex = node.renderPrimitiveIndex,
            .sourceMaterialIndex = sourceMaterialIndex,
            .geometry = geometryId,
            .material = materialId,
            .drawKey = GPUSceneDrawKey{
                .bucket = bucket,
                .material = materialId,
                .geometry = geometryId,
            },
            .worldMatrix = node.worldMatrix,
            .previousWorldMatrix = node.worldMatrix,
            .localBoundingSphere = localBoundingSphere(
                geometries_[geometryId.index].localBounds),
            .transformRevision = node.transformRevision,
            .visible = node.visible,
        });
        instanceForRenderNode_[renderNodeIndex] = id;
        instancesForObject_[static_cast<uint32_t>(node.object)].push_back(id);
    }

    sourceLifetimeRevision_ = source.lifetimeRevision;
    sourceStructuralRevision_ = source.structuralRevision;
    sourceContentRevision_ = source.contentRevision;
    sourceTransformRevision_ = source.transformRevision;
    sourceVisibilityRevision_ = source.visibilityRevision;
    sourceExternalRevision_ = source.externalRevision;
    hasSource_ = true;
    rebuildDrawSet();
    invalidateGpuResources();
    if (!invalidPrimitiveDiagnostics_.empty()) {
        if (!log.empty() && log.back() != '\n') {
            log.push_back('\n');
        }
        log += "GPUScene skipped " +
            std::to_string(invalidPrimitiveDiagnostics_.size()) +
            " invalid triangle primitive payload(s)";
    }
    return {};
}

GPUSceneSyncResult GPUScene::sync(const GPUSceneSourceView& source)
{
    if (!hasSource_ || source.renderPrimitives.size() != primitiveFingerprints_.size() ||
        source.materials.size() != materialFingerprints_.size() ||
        source.renderNodes.size() != renderNodeTopologyFingerprints_.size() ||
        source.lifetimeRevision != sourceLifetimeRevision_) {
        return GPUSceneSyncResult::RebuildRequired;
    }

    const bool validateContent =
        source.structuralRevision != sourceStructuralRevision_ ||
        source.contentRevision != sourceContentRevision_ ||
        source.externalRevision != sourceExternalRevision_;
    if (validateContent) {
        for (uint32_t index = 0; index < source.renderPrimitives.size(); ++index) {
            if (sourcePrimitiveFingerprint(source.renderPrimitives[index]) !=
                primitiveFingerprints_[index]) {
                return GPUSceneSyncResult::RebuildRequired;
            }
        }
        for (uint32_t index = 0; index < source.materials.size(); ++index) {
            if (materialFingerprint(source.materials[index]) != materialFingerprints_[index]) {
                return GPUSceneSyncResult::RebuildRequired;
            }
        }
        for (uint32_t index = 0; index < source.renderNodes.size(); ++index) {
            if (renderNodeTopologyFingerprint(source.renderNodes[index]) !=
                renderNodeTopologyFingerprints_[index]) {
                return GPUSceneSyncResult::RebuildRequired;
            }
        }
    }

    bool updated = false;
    bool historyUpdated = false;
    for (uint32_t renderNodeIndex = 0; renderNodeIndex < source.renderNodes.size(); ++renderNodeIndex) {
        const GPUSceneInstanceId id = instanceForRenderNode_[renderNodeIndex];
        if (!id) {
            continue;
        }
        GPUSceneInstanceRecord& instance = instances_[id.index];
        const scene::RenderNode& node = source.renderNodes[renderNodeIndex];
        if (instance.transformRevision != node.transformRevision ||
            std::memcmp(&instance.worldMatrix, &node.worldMatrix, sizeof(float4x4)) != 0) {
            instance.previousWorldMatrix = instance.worldMatrix;
            instance.worldMatrix = node.worldMatrix;
            instance.transformRevision = node.transformRevision;
            updated = true;
        } else if (std::memcmp(
                       &instance.previousWorldMatrix,
                       &instance.worldMatrix,
                       sizeof(float4x4)) != 0) {
            instance.previousWorldMatrix = instance.worldMatrix;
            historyUpdated = true;
        }
        if (instance.visible != node.visible) {
            instance.visible = node.visible;
            updated = true;
        }
    }

    sourceLifetimeRevision_ = source.lifetimeRevision;
    sourceStructuralRevision_ = source.structuralRevision;
    sourceContentRevision_ = source.contentRevision;
    sourceTransformRevision_ = source.transformRevision;
    sourceVisibilityRevision_ = source.visibilityRevision;
    sourceExternalRevision_ = source.externalRevision;
    if (!updated && !historyUpdated) {
        ++stats_.unchangedSyncCount;
        return GPUSceneSyncResult::Unchanged;
    }

    drawSet_.revision = nextDrawSetRevision_++;
    if (nextDrawSetRevision_ == 0) {
        nextDrawSetRevision_ = 1;
    }
    stats_.drawSetRevision = drawSet_.revision;
    ++stats_.incrementalSyncCount;
    if (updated) {
        ++temporalHistoryEpoch_;
        if (temporalHistoryEpoch_ == 0) {
            temporalHistoryEpoch_ = 1;
        }
    }
    invalidateVisibleDrawSets();
    invalidateGpuResources();
    return updated ? GPUSceneSyncResult::Updated : GPUSceneSyncResult::HistoryUpdated;
}

void GPUScene::rebuildDrawSet()
{
    ++temporalHistoryEpoch_;
    if (temporalHistoryEpoch_ == 0) {
        temporalHistoryEpoch_ = 1;
    }
    drawSet_.generation = drawSetGeneration_;
    drawSet_.instances.clear();
    for (auto& bucket : drawSet_.buckets) {
        bucket.clear();
    }
    drawSet_.instances.reserve(instances_.size());
    for (const GPUSceneInstanceRecord& instance : instances_) {
        drawSet_.instances.push_back(instance.id);
    }
    std::ranges::sort(drawSet_.instances, [&](GPUSceneInstanceId lhs, GPUSceneInstanceId rhs) {
        const GPUSceneInstanceRecord& lhsInstance = instances_[lhs.index];
        const GPUSceneInstanceRecord& rhsInstance = instances_[rhs.index];
        if (lhsInstance.drawKey != rhsInstance.drawKey) {
            return lhsInstance.drawKey < rhsInstance.drawKey;
        }
        return lhs < rhs;
    });
    for (GPUSceneInstanceId id : drawSet_.instances) {
        const size_t index = bucketIndex(instances_[id.index].drawKey.bucket);
        drawSet_.buckets[index].push_back(id);
    }
    drawSet_.revision = nextDrawSetRevision_++;
    if (nextDrawSetRevision_ == 0) {
        nextDrawSetRevision_ = 1;
    }

    stats_.geometryCount = static_cast<uint32_t>(geometries_.size());
    stats_.materialCount = static_cast<uint32_t>(materials_.size());
    stats_.instanceCount = static_cast<uint32_t>(instances_.size());
    stats_.drawSetGeneration = drawSet_.generation;
    stats_.drawSetRevision = drawSet_.revision;
    for (size_t index = 0; index < drawSet_.buckets.size(); ++index) {
        stats_.bucketInstanceCounts[index] =
            static_cast<uint32_t>(drawSet_.buckets[index].size());
    }
    invalidateVisibleDrawSets();
}

void GPUScene::clearSource()
{
    if (!hasSource_ && geometries_.empty() && materials_.empty() && instances_.empty()) {
        return;
    }
    const uint64_t fullRebuildCount = stats_.fullRebuildCount;
    const uint64_t incrementalSyncCount = stats_.incrementalSyncCount;
    const uint64_t unchangedSyncCount = stats_.unchangedSyncCount;
    const uint32_t viewCount = stats_.viewCount;
    invalidateSourceIds();
    primitiveFingerprints_.clear();
    materialFingerprints_.clear();
    renderNodeTopologyFingerprints_.clear();
    sourceLifetimeRevision_ = 0;
    sourceStructuralRevision_ = 0;
    sourceContentRevision_ = 0;
    sourceTransformRevision_ = 0;
    sourceVisibilityRevision_ = 0;
    sourceExternalRevision_ = 0;
    hasSource_ = false;
    stats_ = {};
    stats_.fullRebuildCount = fullRebuildCount;
    stats_.incrementalSyncCount = incrementalSyncCount;
    stats_.unchangedSyncCount = unchangedSyncCount;
    stats_.viewCount = viewCount;
    rebuildDrawSet();
    invalidateGpuResources();
}

void GPUScene::shutdown()
{
    clearSource();
    views_.clear();
    freeViews_.clear();
    stats_ = {};
    globalBufferViews_ = {};
}

const GPUSceneGeometryRecord* GPUScene::geometry(GPUSceneGeometryId id) const
{
    return id.generation == geometryGeneration_ && id.index < geometries_.size() &&
            geometries_[id.index].id == id
        ? &geometries_[id.index]
        : nullptr;
}

const scene::RenderPrimitive* GPUScene::geometrySourcePrimitive(
    GPUSceneGeometryId id) const
{
    return geometry(id) != nullptr && id.index < geometrySourcePrimitives_.size()
        ? &geometrySourcePrimitives_[id.index]
        : nullptr;
}

const GPUSceneMaterialRecord* GPUScene::material(GPUSceneMaterialId id) const
{
    return id.generation == materialGeneration_ && id.index < materials_.size() &&
            materials_[id.index].id == id
        ? &materials_[id.index]
        : nullptr;
}

const GPUSceneInstanceRecord* GPUScene::instance(GPUSceneInstanceId id) const
{
    return id.generation == instanceGeneration_ && id.index < instances_.size() &&
            instances_[id.index].id == id
        ? &instances_[id.index]
        : nullptr;
}

GPUSceneGeometryId GPUScene::geometryForRenderPrimitive(uint32_t renderPrimitiveIndex) const
{
    return mappedId(geometryForRenderPrimitive_, renderPrimitiveIndex);
}

GPUSceneMaterialId GPUScene::materialForSourceMaterial(uint32_t materialIndex) const
{
    return mappedId(materialForSourceMaterial_, materialIndex);
}

GPUSceneInstanceId GPUScene::instanceForRenderNode(uint32_t renderNodeIndex) const
{
    return mappedId(instanceForRenderNode_, renderNodeIndex);
}

std::span<const GPUSceneInstanceId> GPUScene::instancesForObject(scene::SceneEntity object) const
{
    const auto iter = instancesForObject_.find(static_cast<uint32_t>(object));
    return iter == instancesForObject_.end()
        ? std::span<const GPUSceneInstanceId>()
        : std::span<const GPUSceneInstanceId>(iter->second);
}

GPUSceneViewId GPUScene::createView(const GPUSceneViewDesc& desc)
{
    uint32_t index = 0;
    if (freeViews_.empty()) {
        index = static_cast<uint32_t>(views_.size());
        views_.emplace_back();
    } else {
        index = freeViews_.back();
        freeViews_.pop_back();
    }
    ViewSlot& slot = views_[index];
    slot.occupied = true;
    slot.desc = desc;
    slot.desc.frameSlotCount = desc.frameSlotCount == 0
        ? defaultFrameSlotCount_
        : desc.frameSlotCount;
    slot.frameSlots.clear();
    slot.frameSlots.resize(slot.desc.frameSlotCount);
    slot.width = 0;
    slot.height = 0;
    slot.temporalHistoryEpoch = 0;
    slot.hzbHistoryEpoch = 1;
    slot.freezeCullingCamera = false;
    slot.freezeStateValid = false;
    slot.hzbValid = false;
    ++stats_.viewCount;
    return GPUSceneViewId{.index = index, .generation = slot.generation};
}

bool GPUScene::validView(GPUSceneViewId view) const
{
    return view.generation != 0 && view.index < views_.size() &&
        views_[view.index].occupied && views_[view.index].generation == view.generation;
}

bool GPUScene::destroyView(GPUSceneViewId view)
{
    if (!validView(view)) {
        return false;
    }
    ViewSlot& slot = views_[view.index];
    slot.occupied = false;
    slot.desc = {};
    slot.frameSlots.clear();
    slot.width = 0;
    slot.height = 0;
    slot.temporalHistoryEpoch = 0;
    slot.hzbHistoryEpoch = 1;
    slot.freezeCullingCamera = false;
    slot.freezeStateValid = false;
    slot.hzbValid = false;
    slot.generation = advanceGeneration(slot.generation);
    freeViews_.push_back(view.index);
    --stats_.viewCount;
    return true;
}

uint32_t GPUScene::viewFrameSlotCount(GPUSceneViewId view) const
{
    return validView(view)
        ? static_cast<uint32_t>(views_[view.index].frameSlots.size())
        : 0;
}

bool GPUScene::invalidateViewGpuResources(
    GPUSceneViewId view,
    bool invalidateHzbHistory)
{
    if (!validView(view)) {
        return false;
    }
    ViewSlot& viewSlot = views_[view.index];
    if (invalidateHzbHistory) {
        viewSlot.hzbValid = false;
        ++viewSlot.hzbHistoryEpoch;
        if (viewSlot.hzbHistoryEpoch == 0) {
            viewSlot.hzbHistoryEpoch = 1;
        }
    }
    for (GPUSceneVisibleDrawSet& visible : viewSlot.frameSlots) {
        visible.gpu = {};
        if (invalidateHzbHistory &&
            visible.stats.sourceDrawSetGeneration == drawSet_.generation &&
            visible.stats.sourceDrawSetRevision == drawSet_.revision) {
            visible.stats.hzbHistoryEpoch = viewSlot.hzbHistoryEpoch;
            visible.stats.hzbValid = false;
        }
    }
    return true;
}

bool GPUScene::prepareView(
    GPUSceneViewId view,
    uint32_t frameSlot,
    const VisibilityPredicate& predicate)
{
    return prepareView(view, frameSlot, GPUSceneViewPrepareInfo{}, predicate);
}

bool GPUScene::prepareView(
    GPUSceneViewId view,
    uint32_t frameSlot,
    const GPUSceneViewPrepareInfo& info,
    const VisibilityPredicate& predicate)
{
    if (!validView(view) || frameSlot >= views_[view.index].frameSlots.size()) {
        return false;
    }
    ViewSlot& viewSlot = views_[view.index];
    bool invalidateHistory = viewSlot.temporalHistoryEpoch != temporalHistoryEpoch_ ||
        info.cameraCut;
    if (info.width != 0 && info.height != 0 &&
        (viewSlot.width != info.width || viewSlot.height != info.height)) {
        viewSlot.width = info.width;
        viewSlot.height = info.height;
        invalidateHistory = true;
    }
    if (viewSlot.freezeStateValid &&
        viewSlot.freezeCullingCamera != info.freezeCullingCamera) {
        invalidateHistory = true;
    }
    viewSlot.freezeCullingCamera = info.freezeCullingCamera;
    viewSlot.freezeStateValid = true;
    viewSlot.temporalHistoryEpoch = temporalHistoryEpoch_;
    if (invalidateHistory) {
        viewSlot.hzbValid = false;
        ++viewSlot.hzbHistoryEpoch;
        if (viewSlot.hzbHistoryEpoch == 0) {
            viewSlot.hzbHistoryEpoch = 1;
        }
    }

    GPUSceneVisibleDrawSet& visible = viewSlot.frameSlots[frameSlot];
    const uint64_t prepareCount = visible.stats.prepareCount + 1;
    visible = {};
    visible.instances.reserve(drawSet_.instances.size());
    for (GPUSceneInstanceId id : drawSet_.instances) {
        const GPUSceneInstanceRecord& record = instances_[id.index];
        if (!record.visible || (predicate && !predicate(record))) {
            continue;
        }
        visible.instances.push_back(id);
        visible.buckets[bucketIndex(record.drawKey.bucket)].push_back(id);
    }
    visible.stats.sourceInstanceCount = static_cast<uint32_t>(drawSet_.instances.size());
    visible.stats.visibleInstanceCount = static_cast<uint32_t>(visible.instances.size());
    for (size_t index = 0; index < visible.buckets.size(); ++index) {
        visible.stats.bucketInstanceCounts[index] =
            static_cast<uint32_t>(visible.buckets[index].size());
    }
    visible.stats.sourceDrawSetGeneration = drawSet_.generation;
    visible.stats.sourceDrawSetRevision = drawSet_.revision;
    visible.stats.prepareCount = prepareCount;
    visible.stats.hzbHistoryEpoch = viewSlot.hzbHistoryEpoch;
    visible.stats.hzbValid = viewSlot.hzbValid;
    return true;
}

bool GPUScene::markViewHzbValid(
    GPUSceneViewId view,
    uint32_t frameSlot,
    bool valid)
{
    if (!validView(view) || frameSlot >= views_[view.index].frameSlots.size()) {
        return false;
    }
    ViewSlot& viewSlot = views_[view.index];
    GPUSceneVisibleDrawSet& visible = viewSlot.frameSlots[frameSlot];
    if (visible.stats.sourceDrawSetGeneration != drawSet_.generation ||
        visible.stats.sourceDrawSetRevision != drawSet_.revision) {
        return false;
    }
    viewSlot.hzbValid = valid;
    visible.stats.hzbValid = valid;
    return true;
}

const GPUSceneVisibleDrawSet* GPUScene::visibleDrawSet(
    GPUSceneViewId view,
    uint32_t frameSlot) const
{
    if (!validView(view) || frameSlot >= views_[view.index].frameSlots.size()) {
        return nullptr;
    }
    const GPUSceneVisibleDrawSet& visible = views_[view.index].frameSlots[frameSlot];
    return visible.stats.sourceDrawSetGeneration == drawSet_.generation &&
            visible.stats.sourceDrawSetRevision == drawSet_.revision
        ? &visible
        : nullptr;
}

GPUSceneVisibleDrawSet* GPUScene::visibleDrawSetForUpdate(
    GPUSceneViewId view,
    uint32_t frameSlot)
{
    return const_cast<GPUSceneVisibleDrawSet*>(
        std::as_const(*this).visibleDrawSet(view, frameSlot));
}

bool GPUScene::setVisibleGpuResources(
    GPUSceneViewId view,
    uint32_t frameSlot,
    GPUSceneVisibleGpuResources resources)
{
    GPUSceneVisibleDrawSet* visible = visibleDrawSetForUpdate(view, frameSlot);
    if (visible == nullptr) {
        return false;
    }
    if (resources.sourceDrawSetRevision == 0) {
        resources.sourceDrawSetRevision = drawSet_.revision;
    }
    if (resources.sourceDrawSetGeneration == 0) {
        resources.sourceDrawSetGeneration = drawSet_.generation;
    }
    if (!resources.sourceView.valid()) {
        resources.sourceView = view;
        resources.frameSlot = frameSlot;
    }
    if (resources.sourceDrawSetGeneration != drawSet_.generation ||
        resources.sourceDrawSetRevision != drawSet_.revision ||
        resources.sourceView != view || resources.frameSlot != frameSlot ||
        !normalizeBufferView(
            resources.instanceVisibilityStates,
            drawSet_.generation,
            drawSet_.revision) ||
        !normalizeOptionalBufferView(
            resources.visibleInstanceIds,
            drawSet_.generation,
            drawSet_.revision) ||
        !normalizeOptionalBufferView(
            resources.visibleInstanceCounter,
            drawSet_.generation,
            drawSet_.revision)) {
        return false;
    }
    for (GPUSceneCullPhaseGpuView& phase : resources.phases) {
        if (!normalizeBufferView(
                phase.visibleMeshletIds,
                drawSet_.generation,
                drawSet_.revision)) {
            return false;
        }
        const uint64_t visibleMeshletElementCount =
            phase.visibleMeshletIds.size / phase.visibleMeshletIds.structureStride;
        for (GPUSceneBucketGpuView& bucket : phase.buckets) {
            if (bucket.visibleMeshletCapacity == 0 ||
                bucket.visibleMeshletOffset > visibleMeshletElementCount ||
                bucket.visibleMeshletCapacity >
                    visibleMeshletElementCount - bucket.visibleMeshletOffset ||
                !normalizeBufferView(
                    bucket.indirectArguments,
                    drawSet_.generation,
                    drawSet_.revision) ||
                !normalizeBufferView(
                    bucket.overflow,
                    drawSet_.generation,
                    drawSet_.revision)) {
                return false;
            }
        }
    }
    if (resources.hzb.width == 0 || resources.hzb.height == 0 ||
        resources.hzb.mipCount == 0 ||
        resources.hzb.writeIndex >= resources.hzb.history.size() ||
        resources.hzb.historyEpoch != visible->stats.hzbHistoryEpoch ||
        resources.hzb.valid != visible->stats.hzbValid) {
        return false;
    }
    for (GPUSceneBufferView& history : resources.hzb.history) {
        if (!normalizeBufferView(history, drawSet_.generation, drawSet_.revision)) {
            return false;
        }
    }
    visible->gpu = std::move(resources);
    return true;
}

bool GPUScene::setGlobalBufferViews(GPUSceneGlobalBufferViews views)
{
    if (views.drawSetRevision == 0) {
        views.drawSetRevision = drawSet_.revision;
    }
    if (views.drawSetGeneration == 0) {
        views.drawSetGeneration = drawSet_.generation;
    }
    if (views.drawSetGeneration != drawSet_.generation ||
        views.drawSetRevision != drawSet_.revision ||
        !normalizeBufferView(views.geometries, drawSet_.generation, drawSet_.revision) ||
        !normalizeBufferView(views.materials, drawSet_.generation, drawSet_.revision) ||
        !normalizeBufferView(views.instances, drawSet_.generation, drawSet_.revision) ||
        !normalizeBufferView(views.drawKeys, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.drawInstanceIds, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.vertices, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.indices, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.meshlets, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.meshletDraws, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.meshletVertices, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.meshletTriangleWords, drawSet_.generation, drawSet_.revision) ||
        !normalizeOptionalBufferView(views.descriptorRemap, drawSet_.generation, drawSet_.revision)) {
        return false;
    }
    globalBufferViews_ = std::move(views);
    return true;
}

void GPUScene::invalidateVisibleDrawSets()
{
    for (ViewSlot& view : views_) {
        if (!view.occupied) {
            continue;
        }
        for (GPUSceneVisibleDrawSet& visible : view.frameSlots) {
            visible.stats.sourceDrawSetGeneration = 0;
            visible.stats.sourceDrawSetRevision = 0;
            visible.gpu.sourceDrawSetGeneration = 0;
            visible.gpu.sourceDrawSetRevision = 0;
        }
    }
}

void GPUScene::invalidateGpuResources()
{
    globalBufferViews_.drawSetGeneration = 0;
    globalBufferViews_.drawSetRevision = 0;
    for (ViewSlot& view : views_) {
        if (!view.occupied) {
            continue;
        }
        for (GPUSceneVisibleDrawSet& visible : view.frameSlots) {
            visible.gpu.sourceDrawSetGeneration = 0;
            visible.gpu.sourceDrawSetRevision = 0;
        }
    }
}

} // namespace metallic::render
