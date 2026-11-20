#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kMeshDistanceFieldDebugABI = 0x4d44464400000001ull;
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

struct alignas(16) MeshDistanceFieldDebugParams {
    ShaderBuffer instances, mips, indirection;
    ShaderSampledImage atlas;
    ShaderSampler atlasSampler;
    ShaderStorageImage color, trace;
    uint32_t instanceCount;
    std::array<std::array<float, 4>, 4> clipToWorld;
    std::array<float, 4> cameraEyeAndOrthographic;
    uint32_t bricksPerAxis;
    float minTraceDistance, maxTraceDistance, nearClipZ, farClipZ;
    uint32_t debugMode, padding0, padding1;
};

static_assert(sizeof(MeshDistanceFieldGPUMip) == 48);
static_assert(sizeof(MeshDistanceFieldGPUInstance) == 80);
static_assert(sizeof(MeshDistanceFieldDebugParams) == 144);
static_assert(offsetof(MeshDistanceFieldDebugParams, clipToWorld) == 32);
static_assert(offsetof(MeshDistanceFieldDebugParams, farClipZ) == 128);

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
    float volumeToLocal, uint32_t debugId, MeshDistanceFieldGPUInstance& instance)
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
    instance.indices = {0u, debugId, 0u, 0u};
    return true;
}

void setCamera(const ViewCameraConstants& camera, MeshDistanceFieldDebugParams& params)
{
    const float3 eye(camera.eye[0], camera.eye[1], camera.eye[2]);
    const float3 center(camera.center[0], camera.center[1], camera.center[2]);
    const float3 up(camera.upProjection[0], camera.upProjection[1], camera.upProjection[2]);
    const float3 forward = (center - eye) / std::sqrt(dot(center - eye, center - eye));
    const float3 cameraRight = cross(forward, up);
    const float3 right = cameraRight / std::sqrt(dot(cameraRight, cameraRight));
    const bool orthographic = camera.upProjection[3] > 0.5f;
    const float nearPlane = camera.clipOrtho[0];
    const float farPlane = camera.clipOrtho[1];
    const float halfHeight = orthographic ? camera.clipOrtho[2] * 0.5f : nearPlane * std::tan(camera.viewport[3] * 0.5f);
    const float3 horizontal = right * (halfHeight * camera.viewport[0]);
    const float3 vertical = cross(right, forward) * halfHeight;
    // Perspective unprojects only the near plane. Orthographic rays use both
    // affine endpoints, with depth set to 0 and 1 by the debug pass.
    const float3 depth = forward * (orthographic ? farPlane - nearPlane : 0.0f);
    const float3 nearCenter = eye + forward * nearPlane;
    params.clipToWorld = {{{horizontal.x, vertical.x, depth.x, nearCenter.x},
        {horizontal.y, vertical.y, depth.y, nearCenter.y},
        {horizontal.z, vertical.z, depth.z, nearCenter.z}, {0.0f, 0.0f, 0.0f, 1.0f}}};
    params.cameraEyeAndOrthographic = {eye.x, eye.y, eye.z, orthographic ? 1.0f : 0.0f};
    params.nearClipZ = 0.0f;
    params.farClipZ = 1.0f;
    params.minTraceDistance = orthographic ? 0.0f : nearPlane;
    params.maxTraceDistance = orthographic ? farPlane - nearPlane : farPlane;
}

class MeshDistanceFieldDebugPass final : public ComputePass {
public:
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Mesh distance field diagnostic color")
            .storageWrite().format = Format::RGBA16Sfloat;
        reflection.addTextureOutput("trace", "Hit distance in meters, step count, instance ID, hit flag")
            .storageWrite().format = Format::RGBA32Sfloat;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {runtimeEnumSetting("debugMode", "Display", "normals",
            {{"Normals", "normals"}, {"Trace Steps", "steps"}, {"Instances", "instances"}})};
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.device || !context.graphicsQueue || !context.runtimeScene) {
            log = "Mesh Distance Field Debug requires a world scene and a graphics queue";
            return makeError(Error::InvalidArgument);
        }
        scene::MeshDistanceField field;
        const std::filesystem::path path = std::filesystem::path(PROJECT_SOURCE_DIR) /
            properties().value("distanceFieldPath", std::string{});
        if (!scene::loadMeshDistanceField(path, field, log)) {
            return makeError(Error::InvalidArgument);
        }
        scene::MeshDistanceFieldGeometry geometry;
        if (!scene::getMeshDistanceFieldGeometry(*context.runtimeScene, field.sourceMeshIndex, geometry, log)) {
            return makeError(Error::InvalidArgument);
        }
        if (geometry.fingerprint != field.sourceFingerprint) {
            log = "Mesh distance field source geometry changed; rebake the selected mesh";
            return makeError(Error::InvalidArgument);
        }
        device_ = context.device;
        sourceMeshIndex_ = field.sourceMeshIndex;
        const float3 boundsMin(field.localBoundsMin[0], field.localBoundsMin[1], field.localBoundsMin[2]);
        const float3 boundsMax(field.localBoundsMax[0], field.localBoundsMax[1], field.localBoundsMax[2]);
        localCenter_ = (boundsMin + boundsMax) * 0.5f;
        const float3 extent = (boundsMax - boundsMin) * 0.5f;
        volumeToLocal_ = std::max({extent.x, extent.y, extent.z});
        normalizedExtent_ = extent / volumeToLocal_;

        auto result = ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/MeshDistanceFieldDebug",
                .entryPointName = "meshDistanceFieldDebugMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<MeshDistanceFieldDebugParams>(kMeshDistanceFieldDebugABI),
                .debugName = "MeshDistanceFieldDebug"}, program_, log);
        if (result) {
            result = initializeAtlas(field, *context.graphicsQueue);
            if (!result) {
                log = "Mesh distance field GPU upload failed: " + std::string(resultToString(result));
            }
        }
        return result;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        std::vector<MeshDistanceFieldGPUInstance> instances;
        std::unordered_set<scene::SceneEntity> objects;
        uint32_t unsupportedInstances = 0;
        const scene::Scene& scene = *context.runtimeScene();
        for (const auto& node : scene.renderNodes()) {
            if (!node.visible || scene.renderPrimitives()[node.renderPrimitiveIndex].meshIndex != sourceMeshIndex_ ||
                !objects.insert(node.object).second) {
                continue;
            }
            MeshDistanceFieldGPUInstance instance;
            if (makeMeshDistanceFieldInstance(node.worldMatrix, localCenter_, normalizedExtent_, volumeToLocal_,
                    static_cast<uint32_t>(node.nodeIndex) + 1u, instance)) {
                instances.push_back(instance);
            } else {
                ++unsupportedInstances;
            }
        }
        MeshDistanceFieldDebugParams params{};
        params.instanceCount = static_cast<uint32_t>(instances.size());
        if (instances.empty()) { instances.emplace_back(); }
        auto instanceBuffer = makeUploadBuffer(*device_, std::as_bytes(std::span(instances)),
            BufferUsageBits::Storage, sizeof(instances[0]));
        if (!instanceBuffer) { return makeError(instanceBuffer.error()); }
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        params.instances = writer.buffer(instanceBuffer->get());
        params.mips = writer.buffer(mips_.get());
        params.indirection = writer.buffer(indirection_.get());
        params.atlas = writer.sampledImageHandle(atlasView_.get());
        params.atlasSampler = writer.samplerHandle({});
        params.color = writer.storageImageHandle(context.outputTexture("color").view());
        params.trace = writer.storageImageHandle(context.outputTexture("trace").view());
        params.bricksPerAxis = bricksPerAxis_;
        const auto mode = context.properties().value("debugMode", std::string("normals"));
        params.debugMode = mode == "instances" ? 2u : (mode == "steps" ? 1u : 0u);
        if (const auto* view = context.viewConstants()) {
            setCamera(view->current, params);
        } else {
            setCamera(RenderView{}.constants(0, context.width(), context.height(), context.width(), context.height()).current,
                params);
        }

        auto encoded = writer.encode(params, kMeshDistanceFieldDebugABI);
        if (!encoded) { return makeError(encoded.error()); }
        auto result = program_.dispatch(commands, *encoded,
            (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
        if (result) {
            const DebugResourceBinding atlas{.id = "atlas", .texture = atlas_.get(), .state = ResourceState::ShaderRead};
            context.debugCheckpoint("MeshDistanceField", {&atlas, 1}, {
                {"assetCount", 1}, {"instanceCount", params.instanceCount},
                {"unsupportedTransforms", unsupportedInstances}, {"atlasBricksPerAxis", bricksPerAxis_}});
        }
        return result;
    }

    std::vector<std::string> debugCheckpoints() const override { return {"AfterPass", "MeshDistanceField"}; }

private:
    Result<> initializeAtlas(const scene::MeshDistanceField& field, Queue& queue)
    {
        uint32_t bricks = 0;
        for (const auto& mip : field.mips) {
            bricks += static_cast<uint32_t>(mip.bricks.size() / kBrickBytes);
        }
        while (uint64_t(bricksPerAxis_) * bricksPerAxis_ * bricksPerAxis_ < bricks) {
            ++bricksPerAxis_;
        }
        const uint32_t size = bricksPerAxis_ * kBrickSize;
        auto atlas = device_->createTexture({.type = TextureType::Texture3D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination | TextureUsageBits::TransferSource,
            .format = Format::R8Unorm, .width = size, .height = size, .depth = size});
        if (!atlas) { return makeError(atlas.error()); }
        auto view = device_->createTextureView(**atlas, {});
        if (!view) { return makeError(view.error()); }

        std::array<MeshDistanceFieldGPUMip, scene::kMeshDistanceFieldMipCount> mips;
        std::vector<uint32_t> indirection;
        std::vector<uint8_t> voxels(size_t(size) * size * size);
        uint32_t brickBase = 0;
        for (size_t i = 0; i < field.mips.size(); ++i) {
            const auto& mip = field.mips[i];
            const auto& dimensions = mip.indirectionDimensions;
            const auto& scale = mip.volumeToVirtualUVScale;
            const auto& bias = mip.volumeToVirtualUVAdd;
            mips[i] = {{dimensions[0], dimensions[1], dimensions[2], static_cast<uint32_t>(indirection.size())},
                {scale[0], scale[1], scale[2], mip.distanceFieldToVolumeScaleBias[0]},
                {bias[0], bias[1], bias[2], mip.distanceFieldToVolumeScaleBias[1]}};
            for (uint32_t brick : mip.indirection) {
                indirection.push_back(brick == scene::kMeshDistanceFieldInvalidBrick ? brick : brick + brickBase);
            }
            const auto mipBricks = static_cast<uint32_t>(mip.bricks.size() / kBrickBytes);
            for (uint32_t brick = 0; brick < mipBricks; ++brick) {
                const uint32_t index = brickBase + brick;
                const uint32_t x = (index % bricksPerAxis_) * kBrickSize;
                const uint32_t y = (index / bricksPerAxis_ % bricksPerAxis_) * kBrickSize;
                const uint32_t z = (index / (bricksPerAxis_ * bricksPerAxis_)) * kBrickSize;
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
        auto mipBuffer = makeUploadBuffer(*device_, std::as_bytes(std::span(mips)),
            BufferUsageBits::Storage, sizeof(mips[0]));
        if (!mipBuffer) { return makeError(mipBuffer.error()); }
        auto indirectionBuffer = makeUploadBuffer(*device_, std::as_bytes(std::span(indirection)),
            BufferUsageBits::Storage, sizeof(uint32_t));
        if (!indirectionBuffer) { return makeError(indirectionBuffer.error()); }
        auto result = uploadMeshDistanceFieldAtlas(*device_, queue, **atlas, voxels, size);
        if (!result) { return result; }
        atlas_ = std::move(*atlas);
        atlasView_ = std::move(*view);
        mips_ = std::move(*mipBuffer);
        indirection_ = std::move(*indirectionBuffer);
        return {};
    }

    Device* device_ = nullptr;
    ComputeKernel program_;
    std::unique_ptr<Texture> atlas_;
    std::unique_ptr<TextureView> atlasView_;
    std::unique_ptr<Buffer> mips_, indirection_;
    int32_t sourceMeshIndex_ = scene::kInvalidSceneIndex;
    uint32_t bricksPerAxis_ = 1;
    float volumeToLocal_ = 0.0f;
    float3 localCenter_, normalizedExtent_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createMeshDistanceFieldDebugPass()
{
    return std::make_unique<MeshDistanceFieldDebugPass>();
}

} // namespace metallic::render::builtin_pass
