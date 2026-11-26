#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/StaticTextureUpload.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

constexpr uint32_t kBrickSize = scene::kMeshDistanceFieldBrickSize;
constexpr uint32_t kBrickBytes = kBrickSize * kBrickSize * kBrickSize;
constexpr uint32_t kObjectGridInstanceIndexMask = 0x00ffffffu;

struct alignas(16) MeshDistanceFieldGPUMip {
    std::array<uint32_t, 4> dimensionsAndIndirectionOffset;
    std::array<float, 4> volumeToVirtualUVScaleAndDistanceScale;
    std::array<float, 4> volumeToVirtualUVAddAndDistanceBias;
};

struct alignas(16) MeshDistanceFieldGPUInstance {
    std::array<std::array<float, 4>, 3> worldToVolume;
    std::array<float, 4> volumeExtentAndWorldScale;
    std::array<uint32_t, 4> indices;
};

using MeshDistanceFieldGPUMotion = std::array<std::array<float, 4>, 3>;

struct alignas(16) MeshDistanceFieldGPUObject {
    std::array<float, 3> center;
    uint32_t object;
    std::array<float, 3> extent;
    uint32_t flags;
};

static_assert(sizeof(MeshDistanceFieldGPUMip) == 48);
static_assert(sizeof(MeshDistanceFieldGPUInstance) == 80);
static_assert(sizeof(MeshDistanceFieldGPUMotion) == 48);
static_assert(sizeof(MeshDistanceFieldGPUObject) == 32 && offsetof(MeshDistanceFieldGPUObject, extent) == 16);

Result<std::unique_ptr<Buffer>> makeUploadBuffer(Device& device, std::span<const std::byte> bytes,
    BufferUsageBits usage, uint32_t stride = 0, QueueAccessBits queueAccess = QueueAccessBits::Graphics)
{
    auto buffer = device.createBuffer({.size = bytes.size(), .structureStride = stride,
        .usage = usage, .memoryLocation = MemoryLocation::HostUpload, .queueAccess = queueAccess});
    if (!buffer) {
        return makeError(buffer.error());
    }
    void* mapped = (*buffer)->map();
    if (!mapped) {
        return makeError(Error::Failure);
    }
    std::memcpy(mapped, bytes.data(), bytes.size());
    (*buffer)->flush();
    (*buffer)->unmap();
    return buffer;
}

bool makeMeshDistanceFieldInstance(const float4x4& world, float3 localCenter, float3 normalizedExtent,
    float volumeToLocal, uint32_t mipBase, uint32_t debugId, MeshDistanceFieldGPUInstance& instance)
{
    const std::array axes{float3(world.a[0], world.a[1], world.a[2]),
        float3(world.a[4], world.a[5], world.a[6]), float3(world.a[8], world.a[9], world.a[10])};
    const float3 origin(world.a[12], world.a[13], world.a[14]);
    const float scaleSquared = dot(axes[0], axes[0]);
    const float tolerance = scaleSquared * 0.0001f;
    // The sign of the determinant is unrestricted. Orthogonal columns of equal
    // length allow the inverse and the distance factor to share one world scale.
    if (!(scaleSquared > 0.0f) || !std::isfinite(scaleSquared) ||
        !(std::abs(dot(axes[1], axes[1]) - scaleSquared) <= tolerance) ||
        !(std::abs(dot(axes[2], axes[2]) - scaleSquared) <= tolerance) ||
        !(std::abs(dot(axes[0], axes[1])) <= tolerance) ||
        !(std::abs(dot(axes[0], axes[2])) <= tolerance) ||
        !(std::abs(dot(axes[1], axes[2])) <= tolerance)) {
        return false;
    }
    const float factor = 1.0f / (scaleSquared * volumeToLocal);
    const float worldScale = std::sqrt(scaleSquared) * volumeToLocal;
    if (!(factor > 0.0f) || !std::isfinite(factor) || !(worldScale > 0.0f) || !std::isfinite(worldScale)) {
        return false;
    }
    const std::array center{localCenter.x, localCenter.y, localCenter.z};
    for (size_t axis = 0; axis < 3; ++axis) {
        const float3 row = axes[axis] * factor;
        const float offset = -dot(row, origin) - center[axis] / volumeToLocal;
        if (!std::isfinite(offset)) { return false; }
        instance.worldToVolume[axis] = {row.x, row.y, row.z, offset};
    }
    instance.volumeExtentAndWorldScale = {normalizedExtent.x, normalizedExtent.y, normalizedExtent.z, worldScale};
    instance.indices = {mipBase, debugId, 0u, 0u};
    return true;
}

MeshDistanceFieldGPUMotion makeMeshDistanceFieldMotion(const float4x4& previousWorld,
    const std::array<float, 3>& localCenter, float volumeToLocal)
{
    MeshDistanceFieldGPUMotion rows;
    for (size_t axis = 0; axis < rows.size(); ++axis) {
        rows[axis] = {
            previousWorld.a[axis] * volumeToLocal,
            previousWorld.a[axis + 4] * volumeToLocal,
            previousWorld.a[axis + 8] * volumeToLocal,
            previousWorld.a[axis] * localCenter[0] + previousWorld.a[axis + 4] * localCenter[1] +
                previousWorld.a[axis + 8] * localCenter[2] + previousWorld.a[axis + 12],
        };
    }
    return rows;
}

MeshDistanceFieldGPUObject makeMeshDistanceFieldGPUObject(const float4x4& world,
    const std::array<float, 3>& center, const std::array<float, 3>& extent, scene::SceneEntity object)
{
    MeshDistanceFieldGPUObject result{};
    for (size_t axis = 0; axis < 3; ++axis) {
        result.center[axis] = world.a[axis] * center[0] + world.a[4 + axis] * center[1] +
            world.a[8 + axis] * center[2] + world.a[12 + axis];
        result.extent[axis] = std::abs(world.a[axis]) * extent[0] +
            std::abs(world.a[4 + axis]) * extent[1] + std::abs(world.a[8 + axis]) * extent[2];
    }
    result.object = static_cast<uint32_t>(object);
    return result;
}

MeshDistanceFieldGPU::UpdateInstance makeUpdateInstance(uint32_t renderNode, const float4x4& world,
    const MeshDistanceFieldGPUInstance& instance, const MeshDistanceFieldGPUObject& geometry)
{
    // Invert the packed GPU rows. The nearly uniform transforms accepted by
    // makeMeshDistanceFieldInstance can differ from the exact scene transform.
    const auto& rows = instance.worldToVolume;
    std::array<std::array<double, 3>, 3> inverseColumns;
    for (size_t column = 0; column < 3; ++column) {
        const auto& left = rows[(column + 1) % 3];
        const auto& right = rows[(column + 2) % 3];
        inverseColumns[column] = {
            double(left[1]) * right[2] - double(left[2]) * right[1],
            double(left[2]) * right[0] - double(left[0]) * right[2],
            double(left[0]) * right[1] - double(left[1]) * right[0],
        };
    }
    const double determinant = rows[0][0] * inverseColumns[0][0] + rows[0][1] * inverseColumns[0][1] +
        rows[0][2] * inverseColumns[0][2];
    std::array<double, 3> volumeMinimum{}, volumeMaximum{};
    double worldMagnitude = 0.0;
    for (size_t axis = 0; axis < 3; ++axis) {
        double center = 0.0, extent = 0.0;
        for (size_t column = 0; column < 3; ++column) {
            const double inverse = inverseColumns[column][axis] / determinant;
            center -= inverse * rows[column][3];
            extent += std::abs(inverse) * instance.volumeExtentAndWorldScale[column];
        }
        volumeMinimum[axis] = center - extent;
        volumeMaximum[axis] = center + extent;
        worldMagnitude = std::max(worldMagnitude, std::abs(center) + extent);
    }
    // Bound roundoff from the GPU world-to-volume dot products, including
    // cancellation of the world position and translation at large offsets.
    constexpr double kDotProductError = 16.0 * std::numeric_limits<float>::epsilon();
    constexpr float kInfinity = std::numeric_limits<float>::infinity();
    const double padding = kDotProductError * worldMagnitude;
    MeshDistanceFieldGPU::UpdateInstance bounds{.renderNode = renderNode, .object = geometry.object,
        .mipBase = instance.indices[0], .gridFlags = geometry.flags};
    std::copy_n(world.a, bounds.worldMatrix.size(), bounds.worldMatrix.begin());
    for (size_t axis = 0; axis < 3; ++axis) {
        bounds.minimum[axis] = std::nextafter(float(std::min(volumeMinimum[axis] - padding,
            double(geometry.center[axis]) - geometry.extent[axis])), -kInfinity);
        bounds.maximum[axis] = std::nextafter(float(std::max(volumeMaximum[axis] + padding,
            double(geometry.center[axis]) + geometry.extent[axis])), kInfinity);
    }
    return bounds;
}

} // namespace

Result<> MeshDistanceFieldGPU::initialize(Device& device, Queue& queue, const scene::Scene& scene,
    std::span<const std::filesystem::path> paths, std::string& log, Queue* computeConsumer)
{
    std::vector<Asset> assets;
    std::vector<scene::MeshDistanceField> fields;
    uint64_t brickCount = 0;
    uint64_t indirectionCount = 0;
    std::unordered_set<int32_t> meshIndices;
    for (const auto& path : paths) {
        scene::MeshDistanceField field;
        if (!scene::loadMeshDistanceField(path, field, log)) { return makeError(Error::InvalidArgument); }
        if (!meshIndices.insert(field.sourceMeshIndex).second) {
            log = "Multiple mesh distance fields reference source mesh " + std::to_string(field.sourceMeshIndex);
            return makeError(Error::InvalidArgument);
        }
        scene::MeshDistanceFieldGeometry geometry;
        if (!scene::getMeshDistanceFieldGeometry(scene, field.sourceMeshIndex, geometry, log)) {
            return makeError(Error::InvalidArgument);
        }
        if (geometry.fingerprint != field.sourceFingerprint) {
            log = "Mesh distance field source geometry changed; rebake " + path.string();
            return makeError(Error::InvalidArgument);
        }

        Asset asset{};
        asset.meshIndex = field.sourceMeshIndex;
        asset.mipBase = static_cast<uint32_t>(assets.size() * scene::kMeshDistanceFieldMipCount);
        scene::Bounds geometryBounds;
        for (const auto& position : geometry.positions) {
            geometryBounds.include(float3(position[0], position[1], position[2]));
        }
        const float3 geometryCenter = geometryBounds.center();
        const float3 geometryExtent = (geometryBounds.max - geometryBounds.min) * 0.5f;
        asset.geometryCenter = {geometryCenter.x, geometryCenter.y, geometryCenter.z};
        asset.geometryExtent = {geometryExtent.x, geometryExtent.y, geometryExtent.z};
        for (size_t axis = 0; axis < 3; ++axis) {
            asset.localCenter[axis] = (field.localBoundsMin[axis] + field.localBoundsMax[axis]) * 0.5f;
            asset.normalizedExtent[axis] = (field.localBoundsMax[axis] - field.localBoundsMin[axis]) * 0.5f;
        }
        asset.volumeToLocal = std::ranges::max(asset.normalizedExtent);
        for (float& axisExtent : asset.normalizedExtent) { axisExtent /= asset.volumeToLocal; }
        assets.push_back(asset);
        for (const auto& mip : field.mips) {
            brickCount += mip.bricks.size() / kBrickBytes;
            indirectionCount += mip.indirection.size();
        }
        fields.push_back(std::move(field));
    }
    if (brickCount >= scene::kMeshDistanceFieldInvalidBrick || indirectionCount > UINT32_MAX) {
        log = "Mesh distance field atlas exceeds its 32-bit index range";
        return makeError(Error::Unsupported);
    }

    uint32_t bricksPerAxis = 1;
    while (uint64_t(bricksPerAxis) * bricksPerAxis * bricksPerAxis < brickCount) { ++bricksPerAxis; }
    const uint32_t size = bricksPerAxis * kBrickSize;
    const auto queueAccess = computeConsumer ? QueueAccessBits::Graphics | QueueAccessBits::Compute : QueueAccessBits::Graphics;
    auto atlas = device.createTexture({.type = TextureType::Texture3D,
        .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination | TextureUsageBits::TransferSource,
        .format = Format::R8Unorm, .width = size, .height = size, .depth = size, .queueAccess = queueAccess});
    if (!atlas) { return makeError(atlas.error()); }
    auto view = device.createTextureView(**atlas, {});
    if (!view) { return makeError(view.error()); }

    std::vector<MeshDistanceFieldGPUMip> mips;
    std::vector<uint32_t> indirection;
    mips.reserve(fields.size() * scene::kMeshDistanceFieldMipCount);
    indirection.reserve(static_cast<size_t>(indirectionCount));
    std::vector<uint8_t> voxels(size_t(size) * size * size);
    uint32_t brickBase = 0;
    for (const auto& field : fields) {
        for (const auto& mip : field.mips) {
            const auto& dimensions = mip.indirectionDimensions;
            const auto& scale = mip.volumeToVirtualUVScale;
            const auto& bias = mip.volumeToVirtualUVAdd;
            mips.push_back({{dimensions[0], dimensions[1], dimensions[2], static_cast<uint32_t>(indirection.size())},
                {scale[0], scale[1], scale[2], mip.distanceFieldToVolumeScaleBias[0]},
                {bias[0], bias[1], bias[2], mip.distanceFieldToVolumeScaleBias[1]}});
            for (uint32_t brick : mip.indirection) {
                indirection.push_back(brick == scene::kMeshDistanceFieldInvalidBrick ? brick : brick + brickBase);
            }
            const auto mipBricks = static_cast<uint32_t>(mip.bricks.size() / kBrickBytes);
            for (uint32_t brick = 0; brick < mipBricks; ++brick) {
                const uint32_t index = brickBase + brick;
                const uint32_t x = (index % bricksPerAxis) * kBrickSize;
                const uint32_t y = (index / bricksPerAxis % bricksPerAxis) * kBrickSize;
                const uint32_t z = (index / (bricksPerAxis * bricksPerAxis)) * kBrickSize;
                const uint8_t* next = mip.bricks.data() + size_t(brick) * kBrickBytes;
                for (uint32_t slice = 0; slice < kBrickSize; ++slice) {
                    for (uint32_t row = 0; row < kBrickSize; ++row) {
                        const size_t destination = (size_t(z + slice) * size + y + row) * size + x;
                        std::copy_n(next, kBrickSize, voxels.data() + destination);
                        next += kBrickSize;
                    }
                }
            }
            brickBase += mipBricks;
        }
    }
    // Keep descriptors valid for the zero-instance compose dispatch.
    if (mips.empty()) {
        mips.emplace_back();
        indirection.push_back(scene::kMeshDistanceFieldInvalidBrick);
    }
    auto mipBuffer = makeUploadBuffer(device, std::as_bytes(std::span(mips)),
        BufferUsageBits::Storage, sizeof(mips[0]), queueAccess);
    if (!mipBuffer) { return makeError(mipBuffer.error()); }
    auto indirectionBuffer = makeUploadBuffer(device, std::as_bytes(std::span(indirection)),
        BufferUsageBits::Storage, sizeof(uint32_t), queueAccess);
    if (!indirectionBuffer) { return makeError(indirectionBuffer.error()); }
    auto staging = makeUploadBuffer(device, std::as_bytes(std::span(voxels)), BufferUsageBits::TransferSource);
    if (!staging) { return makeError(staging.error()); }
    auto result = uploadStaticComputeTexture(device, queue, **atlas, **staging, computeConsumer);
    if (!result) { return result; }

    device_ = &device;
    queueAccess_ = queueAccess;
    assets_ = std::move(assets);
    atlas_ = std::move(*atlas);
    atlasView_ = std::move(*view);
    mips_ = std::move(*mipBuffer);
    indirection_ = std::move(*indirectionBuffer);
    return {};
}

Result<std::unique_ptr<Buffer>> MeshDistanceFieldGPU::uploadInstances(const scene::Scene& scene,
    uint32_t& count, uint32_t& unsupportedTransforms, std::unique_ptr<Buffer>* objectGridObjects,
    std::unique_ptr<Buffer>* validationBounds, const MotionOutput* motion,
    std::vector<UpdateInstance>* updateInstances) const
{
    std::vector<MeshDistanceFieldGPUInstance> instances;
    std::vector<MeshDistanceFieldGPUObject> gridObjects;
    std::vector<std::array<float, 8>> localBounds;
    std::vector<MeshDistanceFieldGPUMotion> previousTransforms;
    unsupportedTransforms = 0;
    if (updateInstances) { updateInstances->clear(); }
    const auto& renderNodes = scene.renderNodes();
    for (uint32_t next = 0; next < renderNodes.size();) {
        const uint32_t renderNodeIndex = next++;
        const auto& node = renderNodes[renderNodeIndex];
        while (next < renderNodes.size() && renderNodes[next].object == node.object) { ++next; }
        if (!node.visible) { continue; }
        const auto& primitive = scene.renderPrimitives()[node.renderPrimitiveIndex];
        const auto asset = std::ranges::find(assets_, primitive.meshIndex, &Asset::meshIndex);
        if (asset == assets_.end()) {
            if (validationBounds) {
                spdlog::error("Mesh SDF verification is missing a bake for visible mesh {}", primitive.meshIndex);
                return makeError(Error::InvalidArgument);
            }
            continue;
        }
        const auto otherSections = std::span(renderNodes).subspan(renderNodeIndex + 1, next - renderNodeIndex - 1);
        // The bake describes the whole mesh, so it only represents a rigidly placed set of sections.
        if (std::ranges::any_of(otherSections, [&](const auto& section) {
                return !std::ranges::equal(node.worldMatrix.a, section.worldMatrix.a);
            })) {
            ++unsupportedTransforms;
            if (validationBounds) {
                spdlog::error("Mesh SDF verification requires a shared transform for all sections of object {}",
                    static_cast<uint32_t>(node.object));
                return makeError(Error::Unsupported);
            }
            continue;
        }

        const auto& center = asset->localCenter;
        const auto& extent = asset->normalizedExtent;
        MeshDistanceFieldGPUInstance instance;
        if (makeMeshDistanceFieldInstance(node.worldMatrix, float3(center[0], center[1], center[2]),
                float3(extent[0], extent[1], extent[2]), asset->volumeToLocal,
                asset->mipBase, static_cast<uint32_t>(node.nodeIndex) + 1u, instance)) {
            instance.indices[2] = static_cast<uint32_t>(node.object);
            instances.push_back(instance);
            if (motion) {
                const auto* history = motion->scene.instance(motion->scene.instanceForRenderNode(renderNodeIndex));
                if (!history) {
                    spdlog::error("Mesh SDF motion requires a GPU scene instance for object {}", static_cast<uint32_t>(node.object));
                    return makeError(Error::InvalidArgument);
                }
                previousTransforms.push_back(makeMeshDistanceFieldMotion(history->previousWorldMatrix, center, asset->volumeToLocal));
            }
            if (validationBounds) {
                std::array<float, 8> bounds{};
                for (size_t axis = 0; axis < 3; ++axis) {
                    const float center = asset->geometryCenter[axis] - asset->localCenter[axis];
                    bounds[axis] = (center - asset->geometryExtent[axis]) / asset->volumeToLocal;
                    bounds[axis + 4] = (center + asset->geometryExtent[axis]) / asset->volumeToLocal;
                }
                localBounds.push_back(bounds);
            }
            if (objectGridObjects || updateInstances) {
                auto bounds = makeMeshDistanceFieldGPUObject(node.worldMatrix,
                    asset->geometryCenter, asset->geometryExtent, node.object);
                if (primitive.materialIndex >= 0) {
                    const auto& emissive = scene.materials()[primitive.materialIndex].emissiveFactor;
                    bounds.flags = emissive.x > 0.0f || emissive.y > 0.0f || emissive.z > 0.0f;
                }
                if (objectGridObjects) { gridObjects.push_back(bounds); }
                if (updateInstances) {
                    updateInstances->push_back(makeUpdateInstance(renderNodeIndex, node.worldMatrix, instance, bounds));
                }
            }
        } else {
            ++unsupportedTransforms;
            if (validationBounds) {
                spdlog::error("Mesh SDF verification requires a uniform transform for object {}",
                    static_cast<uint32_t>(node.object));
                return makeError(Error::Unsupported);
            }
        }
    }
    // At maximum rank, a packed 0xffffff index would collide with the invalid entry.
    if (objectGridObjects && instances.size() > kObjectGridInstanceIndexMask) {
        spdlog::error("Global Distance Field object grid exceeds its 24-bit instance index range");
        return makeError(Error::Unsupported);
    }
    count = static_cast<uint32_t>(instances.size());
    if (instances.empty()) { instances.emplace_back(); }
    auto instanceBuffer = makeUploadBuffer(*device_, std::as_bytes(std::span(instances)),
        BufferUsageBits::Storage, sizeof(instances[0]), queueAccess_);
    if (instanceBuffer && motion) {
        if (previousTransforms.empty()) { previousTransforms.emplace_back(); }
        auto motionBuffer = makeUploadBuffer(*device_, std::as_bytes(std::span(previousTransforms)),
            BufferUsageBits::Storage, sizeof(previousTransforms[0]), queueAccess_);
        if (!motionBuffer) { return makeError(motionBuffer.error()); }
        motion->buffer = std::move(*motionBuffer);
    }
    if (instanceBuffer && validationBounds) {
        if (localBounds.empty()) { localBounds.emplace_back(); }
        auto boundsBuffer = makeUploadBuffer(*device_, std::as_bytes(std::span(localBounds)),
            BufferUsageBits::Storage, sizeof(localBounds[0]), queueAccess_);
        if (!boundsBuffer) { return makeError(boundsBuffer.error()); }
        *validationBounds = std::move(*boundsBuffer);
    }
    if (!instanceBuffer || !objectGridObjects) { return instanceBuffer; }
    if (gridObjects.empty()) { gridObjects.emplace_back(); }
    auto objectBuffer = makeUploadBuffer(*device_, std::as_bytes(std::span(gridObjects)),
        BufferUsageBits::Storage, sizeof(gridObjects[0]), queueAccess_);
    if (!objectBuffer) { return makeError(objectBuffer.error()); }
    *objectGridObjects = std::move(*objectBuffer);
    return instanceBuffer;
}

MeshDistanceFieldGPUBindings MeshDistanceFieldGPU::bind(ParameterWriter& writer) const
{
    return {writer.buffer(mips_.get()), writer.buffer(indirection_.get()),
        writer.sampledImageHandle(atlasView_.get()), writer.samplerHandle({})};
}

} // namespace metallic::render
