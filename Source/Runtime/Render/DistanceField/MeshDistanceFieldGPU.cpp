#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace metallic::render {
namespace {

constexpr uint32_t kBrickSize = scene::kMeshDistanceFieldBrickSize;
constexpr uint32_t kBrickBytes = kBrickSize * kBrickSize * kBrickSize;

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

static_assert(sizeof(MeshDistanceFieldGPUMip) == 48);
static_assert(sizeof(MeshDistanceFieldGPUInstance) == 80);

Result<std::unique_ptr<Buffer>> makeUploadBuffer(Device& device, std::span<const std::byte> bytes,
    BufferUsageBits usage, uint32_t stride = 0)
{
    auto buffer = device.createBuffer({.size = bytes.size(), .structureStride = stride,
        .usage = usage, .memoryLocation = MemoryLocation::HostUpload});
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

Result<> uploadMeshDistanceFieldAtlas(Device& device, Queue& queue, Texture& atlas,
    std::span<const uint8_t> bytes, uint32_t size)
{
    auto source = makeUploadBuffer(device, std::as_bytes(bytes), BufferUsageBits::TransferSource);
    if (!source) { return makeError(source.error()); }
    auto sourceSlice = (*source)->slice();
    if (!sourceSlice) { return makeError(sourceSlice.error()); }
    auto pool = device.createCommandPool(queue);
    if (!pool) { return makeError(pool.error()); }
    auto commands = (*pool)->createCommandBuffer();
    if (!commands) { return makeError(commands.error()); }
    auto fence = device.createFence(false);
    if (!fence) { return makeError(fence.error()); }
    auto result = (*commands)->begin();
    if (!result) { return result; }

    TextureBarrierDesc barrier{.texture = &atlas, .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::TransferDestination,
        .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
    result = (*commands)->synchronize({.textures = {&barrier, 1}});
    if (!result) { return result; }
    result = (*commands)->copyBufferToTexture({.texture = &atlas, .buffer = *sourceSlice,
        .bufferRowPitch = size, .bufferSlicePitch = size * size,
        .width = size, .height = size, .depth = size});
    if (!result) { return result; }
    barrier.oldLayout = TextureLayout::TransferDestination;
    barrier.newLayout = TextureLayout::ShaderRead;
    barrier.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite};
    barrier.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
    result = (*commands)->synchronize({.textures = {&barrier, 1}});
    if (!result) { return result; }
    result = (*commands)->end();
    if (!result) { return result; }
    CommandBuffer* command = commands->get();
    result = queue.submit({.commandBuffers = {&command, 1}, .signalFence = fence->get()});
    return result ? (*fence)->wait() : result;
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
    const std::array center{localCenter.x, localCenter.y, localCenter.z};
    for (size_t axis = 0; axis < 3; ++axis) {
        const float3 row = axes[axis] * factor;
        instance.worldToVolume[axis] = {row.x, row.y, row.z, -dot(row, origin) - center[axis] / volumeToLocal};
    }
    instance.volumeExtentAndWorldScale = {normalizedExtent.x, normalizedExtent.y, normalizedExtent.z,
        std::sqrt(scaleSquared) * volumeToLocal};
    instance.indices = {mipBase, debugId, 0u, 0u};
    return true;
}

} // namespace

Result<> MeshDistanceFieldGPU::initialize(Device& device, Queue& queue, const scene::Scene& scene,
    std::span<const std::filesystem::path> paths, std::string& log)
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
    auto atlas = device.createTexture({.type = TextureType::Texture3D,
        .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination | TextureUsageBits::TransferSource,
        .format = Format::R8Unorm, .width = size, .height = size, .depth = size});
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
        BufferUsageBits::Storage, sizeof(mips[0]));
    if (!mipBuffer) { return makeError(mipBuffer.error()); }
    auto indirectionBuffer = makeUploadBuffer(device, std::as_bytes(std::span(indirection)),
        BufferUsageBits::Storage, sizeof(uint32_t));
    if (!indirectionBuffer) { return makeError(indirectionBuffer.error()); }
    auto result = uploadMeshDistanceFieldAtlas(device, queue, **atlas, voxels, size);
    if (!result) { return result; }

    device_ = &device;
    assets_ = std::move(assets);
    atlas_ = std::move(*atlas);
    atlasView_ = std::move(*view);
    mips_ = std::move(*mipBuffer);
    indirection_ = std::move(*indirectionBuffer);
    return {};
}

Result<std::unique_ptr<Buffer>> MeshDistanceFieldGPU::uploadInstances(const scene::Scene& scene,
    uint32_t& count, uint32_t& unsupportedTransforms) const
{
    std::vector<MeshDistanceFieldGPUInstance> instances;
    std::unordered_set<scene::SceneEntity> objects;
    unsupportedTransforms = 0;
    for (const auto& node : scene.renderNodes()) {
        if (!node.visible) { continue; }
        const int32_t meshIndex = scene.renderPrimitives()[node.renderPrimitiveIndex].meshIndex;
        const auto asset = std::ranges::find(assets_, meshIndex, &Asset::meshIndex);
        if (asset == assets_.end() || !objects.insert(node.object).second) { continue; }

        const auto& center = asset->localCenter;
        const auto& extent = asset->normalizedExtent;
        MeshDistanceFieldGPUInstance instance;
        if (makeMeshDistanceFieldInstance(node.worldMatrix, float3(center[0], center[1], center[2]),
                float3(extent[0], extent[1], extent[2]), asset->volumeToLocal,
                asset->mipBase, static_cast<uint32_t>(node.nodeIndex) + 1u, instance)) {
            instances.push_back(instance);
        } else {
            ++unsupportedTransforms;
        }
    }
    count = static_cast<uint32_t>(instances.size());
    if (instances.empty()) { instances.emplace_back(); }
    return makeUploadBuffer(*device_, std::as_bytes(std::span(instances)), BufferUsageBits::Storage, sizeof(instances[0]));
}

MeshDistanceFieldGPUBindings MeshDistanceFieldGPU::bind(ParameterWriter& writer) const
{
    return {writer.buffer(mips_.get()), writer.buffer(indirection_.get()),
        writer.sampledImageHandle(atlasView_.get()), writer.samplerHandle({})};
}

} // namespace metallic::render
