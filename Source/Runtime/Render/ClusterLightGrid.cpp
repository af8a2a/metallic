#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace metallic::render {
namespace {

constexpr uint64_t kMaxGridBytes = 256ull * 1024 * 1024;

bool finiteVector(const float3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

float3 safeNormalize(const float3& value, const float3& fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-8f ? value / std::sqrt(lengthSquared) : fallback;
}

Result compileClusterLightGridProgram(std::vector<uint32_t>& spirv, std::string& log)
{
    ShaderCompileResult shader;
    Result result = compileSlangShaderToSpirv({.moduleName = "ClusterLightGrid",
        .entryPointName = "clusterLightGridMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
    if (!result) { log = shader.diagnostics; return result; }
    spirv = std::move(shader.spirv);
    return {};
}

Result initializeClusterLightGridProgram(Device& device, ComputeProgram& program,
    const std::vector<uint32_t>& spirv, std::string& log)
{
    const std::array<ComputeProgramBindingDesc, 5> bindings{{
        {.binding = 0}, {.binding = 1}, {.binding = 2}, {.binding = 3}, {.binding = 4}}};
    return program.initialize(device, {.spirv = spirv.data(),
        .byteSize = spirv.size() * sizeof(uint32_t), .bindings = bindings.data(),
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .debugName = "Cluster light grid", .requiresRayQuery = false}, log);
}

} // namespace

Result buildClusterLightGridParams(const ClusterLightGridDesc& desc,
    ClusterLightGridParams& params, std::string& log)
{
    params = {};
    if (desc.width == 0 || desc.height == 0 || desc.tileSize == 0 || desc.tileSize > 4096 ||
        desc.depthSliceCount == 0 || desc.depthSliceCount > 256 ||
        desc.maxLightsPerCell == 0 || desc.maxLightsPerCell > 1024 ||
        !finiteVector(desc.eye) || !finiteVector(desc.center) || !finiteVector(desc.up) ||
        !std::isfinite(desc.aspect) || desc.aspect <= 0.0f ||
        !std::isfinite(desc.fovRadians) || desc.fovRadians <= 0.0f || desc.fovRadians >= 3.141592654f ||
        !std::isfinite(desc.zNear) || !std::isfinite(desc.zFar) ||
        desc.zNear <= 0.0f || desc.zFar <= desc.zNear ||
        !std::isfinite(desc.orthoHeight) || desc.orthoHeight < 0.0f) {
        log = "ClusterLightGrid requires a finite camera, positive viewport and bounded grid configuration";
        return makeError(Error::InvalidArgument);
    }
    const uint64_t gridX = (uint64_t(desc.width) + desc.tileSize - 1) / desc.tileSize;
    const uint64_t gridY = (uint64_t(desc.height) + desc.tileSize - 1) / desc.tileSize;
    // Each dispatch dimension fits Vulkan's portable minimum workgroup limit.
    if (gridX > 65535 || gridY > 65535) {
        log = "ClusterLightGrid dispatch dimensions exceed the portable workgroup limit";
        return makeError(Error::InvalidArgument);
    }
    const uint64_t cells = gridX * gridY * desc.depthSliceCount;
    const uint64_t bytes = cells * (sizeof(ClusterLightGridCell) +
        uint64_t(desc.maxLightsPerCell) * sizeof(uint32_t));
    if (bytes > kMaxGridBytes || cells * desc.maxLightsPerCell > UINT32_MAX) {
        log = "ClusterLightGrid exceeds the 256 MiB per-view/frame-slot grid budget";
        return makeError(Error::InvalidArgument);
    }
    const float3 forwardDelta = desc.center - desc.eye;
    if (!finiteVector(forwardDelta) || !std::isfinite(dot(forwardDelta, forwardDelta)) ||
        dot(forwardDelta, forwardDelta) <= 1e-8f || !std::isfinite(dot(desc.up, desc.up)) ||
        dot(desc.up, desc.up) <= 1e-8f) {
        log = "ClusterLightGrid camera direction and up vector must be non-degenerate";
        return makeError(Error::InvalidArgument);
    }
    const float3 forward = safeNormalize(forwardDelta, float3(0.0f, 0.0f, -1.0f));
    const float3 sourceUp = safeNormalize(desc.up, float3(0.0f, 1.0f, 0.0f));
    if (dot(cross(forward, sourceUp), cross(forward, sourceUp)) <= 1e-8f) {
        log = "ClusterLightGrid camera up vector must not be parallel to its direction";
        return makeError(Error::InvalidArgument);
    }
    const float3 right = safeNormalize(cross(forward, sourceUp), float3(1.0f, 0.0f, 0.0f));
    const float3 up = safeNormalize(cross(right, forward), float3(0.0f, 1.0f, 0.0f));
    if (!finiteVector(forward) || !finiteVector(right) || !finiteVector(up) ||
        dot(forward, forward) < 0.5f || dot(right, right) < 0.5f || dot(up, up) < 0.5f) {
        log = "ClusterLightGrid camera basis is not representable";
        return makeError(Error::InvalidArgument);
    }
    const bool orthographic = desc.orthoHeight > 0.0f;
    const float extentY = orthographic ? std::max(desc.orthoHeight * 0.5f, 0.0001f)
        : std::tan(std::clamp(desc.fovRadians, 0.017453292f, 3.12413936f) * 0.5f);
    const float extentX = extentY * std::max(desc.aspect, 0.001f);
    const float zScale = orthographic ? desc.depthSliceCount / (desc.zFar - desc.zNear)
        : static_cast<float>(double(desc.depthSliceCount) /
            std::log2(double(desc.zFar) / desc.zNear));
    const float zB = orthographic ? 0.0f : 1.0f / desc.zNear;
    if (!std::isfinite(extentX) || !std::isfinite(extentY) ||
        !std::isfinite(zScale) || zScale <= 0.0f || !std::isfinite(zB) ||
        !std::isfinite(desc.zFar * extentX) || !std::isfinite(desc.zFar * extentY) ||
        (!orthographic && !std::isfinite(desc.zFar * zB))) {
        log = "ClusterLightGrid projection or depth distribution exceeds float precision";
        return makeError(Error::InvalidArgument);
    }
    params.grid = {static_cast<uint32_t>(gridX), static_cast<uint32_t>(gridY),
        desc.depthSliceCount, desc.tileSize};
    params.viewport = {desc.width, desc.height, desc.maxLightsPerCell, orthographic ? 1u : 0u};
    params.eyeNear = {desc.eye.x, desc.eye.y, desc.eye.z, desc.zNear};
    params.rightFar = {right.x, right.y, right.z, desc.zFar};
    params.upExtent = {up.x, up.y, up.z, extentY};
    params.forwardExtent = {forward.x, forward.y, forward.z, extentX};
    params.zParams = {zB, 0.0f, zScale, 0.0f};
    return {};
}

float clusterLightGridSliceDepth(const ClusterLightGridParams& params, uint32_t slice)
{
    if (slice == 0) { return params.eyeNear[3]; }
    if (slice >= params.grid[2]) { return params.rightFar[3]; }
    if (params.viewport[3] != 0) {
        return params.eyeNear[3] + slice / params.zParams[2];
    }
    return (std::exp2(slice / params.zParams[2]) - params.zParams[1]) / params.zParams[0];
}

bool clusterLightGridCellIndex(const ClusterLightGridParams& params,
    uint32_t pixelX, uint32_t pixelY, float viewDepth, uint32_t& cellIndex)
{
    cellIndex = UINT32_MAX;
    if (params.grid[0] == 0 || params.grid[1] == 0 || params.grid[2] == 0 || params.grid[3] == 0 ||
        pixelX >= params.viewport[0] || pixelY >= params.viewport[1] || !std::isfinite(viewDepth) ||
        viewDepth < params.eyeNear[3] || viewDepth > params.rightFar[3]) {
        return false;
    }
    const float z = params.viewport[3] != 0
        ? (viewDepth - params.eyeNear[3]) * params.zParams[2]
        : std::log2(viewDepth * params.zParams[0] + params.zParams[1]) * params.zParams[2];
    if (!std::isfinite(z)) { return false; }
    const uint32_t slice = static_cast<uint32_t>(std::clamp(z, 0.0f, float(params.grid[2] - 1)));
    cellIndex = (slice * params.grid[1] + pixelY / params.grid[3]) * params.grid[0] +
        pixelX / params.grid[3];
    return true;
}

uint64_t ClusterLightGridSnapshot::cellCount() const
{
    return uint64_t(params.grid[0]) * params.grid[1] * params.grid[2];
}

bool ClusterLightGridSnapshot::valid() const
{
    return parameters != nullptr && lights != nullptr && candidates != nullptr &&
        cells != nullptr && lightIndices != nullptr && sourceView.valid() &&
        sourceLightGeneration != 0 && sourceLightRevision != 0 && sourcePrepareCount != 0 &&
        buildRevision != 0 && cellCount() != 0;
}

struct ClusterLightGrid::Resources {
    std::array<std::shared_ptr<Buffer>, 5> buffers;
    // A legacy command without a frame must not mutate a shared descriptor table.
    std::shared_ptr<ComputeProgram> untrackedProgram;
    GpuCompletionPoint completion;
    bool cancelled = false;
};

class ClusterLightGrid::Publication final : public SubmissionTransaction {
public:
    explicit Publication(std::shared_ptr<Resources> resources)
        : SubmissionTransaction([]() {}, [resources]() { resources->cancelled = true; })
        , resources_(std::move(resources))
    {
    }

private:
    // Unlike callback captures (released at submit), this ownership survives
    // submission until CommandBuffer resets its submission state. Construct as
    // shared_ptr<Publication> so the control block destroys the derived object.
    std::shared_ptr<Resources> resources_;
};

class ClusterLightGrid::ShaderReload final : public RenderSubsystemShaderReload {
public:
    ShaderReload(ClusterLightGrid& owner, ComputeProgram program, std::vector<uint32_t> spirv)
        : owner_(owner), program_(std::move(program)), spirv_(std::move(spirv))
    {
    }

    void commit() noexcept override
    {
        if (committed_) { return; }
        // Dispatch retains the old ComputeProgram implementation in its tracked
        // frame, so replacement needs no device-wide idle or buffer retirement.
        owner_.program_ = std::move(program_);
        owner_.programSpirv_ = std::move(spirv_);
        owner_.snapshot_ = {};
        committed_ = true;
    }

private:
    ClusterLightGrid& owner_;
    ComputeProgram program_;
    std::vector<uint32_t> spirv_;
    bool committed_ = false;
};

Result ClusterLightGrid::record(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
    const GPUScene& scene, GPUSceneViewId view, uint32_t frameSlot,
    const ClusterLightGridDesc& desc, std::string& log)
{
    RenderFrameContext* frame = commands.frameContext();
    if (frame != nullptr && !frame->recording()) {
        log = "ClusterLightGrid received an inactive RenderFrameContext";
        return makeError(Error::InvalidArgument);
    }
    if (publication_ != nullptr && !publication_->resolved()) {
        log = "ClusterLightGrid previous recording must be submitted or cancelled before rebuilding";
        return makeError(Error::InvalidArgument);
    }
    snapshot_ = {};
    const auto* visible = scene.visibleDrawSet(view, frameSlot);
    const auto* visibleLights = scene.visibleLights(view, frameSlot);
    if (visible == nullptr || visibleLights == nullptr) {
        log = "ClusterLightGrid requires current prepared light candidates for the View/frame slot";
        return makeError(Error::InvalidArgument);
    }
    ClusterLightGridParams params;
    Result result = buildClusterLightGridParams(desc, params, log);
    if (!result) { return result; }
    if (scene.lights().size() > UINT32_MAX ||
        scene.lights().size() * sizeof(GpuPunctualLight) > kMaxGridBytes) {
        log = "ClusterLightGrid source light upload exceeds its buffer budget";
        return makeError(Error::InvalidArgument);
    }
    params.counts = {static_cast<uint32_t>(visibleLights->localLights.size()),
        static_cast<uint32_t>(visibleLights->directionalLights.size()),
        static_cast<uint32_t>(visibleLights->unboundedLocalLights.size()),
        static_cast<uint32_t>(scene.lights().size())};
    std::vector<GpuPunctualLight> lightData;
    lightData.reserve(std::max(scene.lights().size(), size_t(1)));
    for (const auto& record : scene.lights()) { lightData.push_back(record.source.gpu); }
    if (lightData.empty()) { lightData.emplace_back(); }
    std::vector<uint32_t> candidates;
    const auto append = [&](std::span<const GPUSceneLightId> ids) {
        for (auto id : ids) { candidates.push_back(id.index); }
    };
    append(visibleLights->localLights);
    append(visibleLights->directionalLights);
    append(visibleLights->unboundedLocalLights);
    if (candidates.empty()) { candidates.push_back(0); }

    if (programSpirv_.empty()) {
        result = compileClusterLightGridProgram(programSpirv_, log);
        if (!result) { return result; }
    }
    const auto untrackedProgram = frame == nullptr ? std::make_shared<ComputeProgram>() : nullptr;
    ComputeProgram& dispatchProgram = untrackedProgram != nullptr ? *untrackedProgram : program_;
    if (!dispatchProgram.valid()) {
        result = initializeClusterLightGridProgram(device, dispatchProgram, programSpirv_, log);
        if (!result) { return result; }
    }
    const uint64_t cellCount = uint64_t(params.grid[0]) * params.grid[1] * params.grid[2];
    const std::array<uint64_t, 5> sizes{sizeof(params), lightData.size() * sizeof(GpuPunctualLight),
        candidates.size() * sizeof(uint32_t), cellCount * sizeof(ClusterLightGridCell),
        cellCount * desc.maxLightsPerCell * sizeof(uint32_t)};
    // Reuse only after the full submission (including every consumer) completed.
    bool reuse = frame != nullptr && resources_ != nullptr &&
        resources_->completion.valid() && resources_->completion.isComplete();
    if (reuse) {
        for (size_t index = 0; index < sizes.size(); ++index) {
            reuse &= resources_->buffers[index]->desc().size >= sizes[index];
        }
    }
    auto next = reuse ? resources_ : std::make_shared<Resources>();
    for (size_t index = 0; index < sizes.size(); ++index) {
        if (!reuse) {
            std::unique_ptr<Buffer> buffer;
            result = device.createBuffer({.size = sizes[index],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = index < 3 ? MemoryLocation::HostUpload : MemoryLocation::Device}, buffer);
            if (!result) { log = "ClusterLightGrid buffer allocation failed"; return result; }
            next->buffers[index] = std::move(buffer);
        }
        if (index < 3) {
            void* mapped = next->buffers[index]->map();
            if (mapped == nullptr) { log = "ClusterLightGrid upload mapping failed"; return makeError(Error::Failure); }
            const void* data = index == 0 ? static_cast<const void*>(&params)
                : index == 1 ? static_cast<const void*>(lightData.data()) : static_cast<const void*>(candidates.data());
            std::memcpy(mapped, data, static_cast<size_t>(sizes[index]));
            next->buffers[index]->flush(0, sizes[index]);
            next->buffers[index]->unmap();
        }
    }
    next->untrackedProgram = untrackedProgram;
    if (frame != nullptr) {
        next->completion = frame->completion();
        frame->retain(next);
    }
    next->cancelled = false;
    auto publication = std::make_shared<Publication>(next);
    result = commands.addSubmissionTransaction(publication);
    if (!result) { return result; }
    commands.hostWriteBarrier();
    const ResourceState previousState = reuse ? ResourceState::ShaderRead : ResourceState::Undefined;
    const std::array<BufferBarrierDesc, 2> toWrite{{
        {.buffer = next->buffers[3].get(), .before = previousState, .after = ResourceState::General},
        {.buffer = next->buffers[4].get(), .before = previousState, .after = ResourceState::General}}};
    commands.barrier({.buffers = toWrite.data(), .bufferCount = static_cast<uint32_t>(toWrite.size())});
    std::array<ComputeDispatchBinding, 5> bindings;
    for (uint32_t index = 0; index < bindings.size(); ++index) {
        // Bind the entire grow-only allocation. Logical counts in params bound
        // all shader accesses, including after a viewport or light-count shrink.
        bindings[index] = {.binding = index, .buffer = next->buffers[index].get()};
    }
    result = dispatchProgram.dispatch({.commandBuffer = &commands, .bindings = bindings.data(),
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .groupCountX = params.grid[0], .groupCountY = params.grid[1], .groupCountZ = params.grid[2]});
    if (!result) {
        publication->cancel();
        log = "ClusterLightGrid compute dispatch failed";
        return result;
    }
    const std::array<BufferBarrierDesc, 2> toRead{{
        {.buffer = next->buffers[3].get(), .before = ResourceState::General, .after = ResourceState::ShaderRead},
        {.buffer = next->buffers[4].get(), .before = ResourceState::General, .after = ResourceState::ShaderRead}}};
    commands.barrier({.buffers = toRead.data(), .bufferCount = static_cast<uint32_t>(toRead.size())});
    if (resources_ != next) { host.retire(resources_); }
    resources_ = std::move(next);
    publication_ = std::move(publication);
    snapshot_ = {.parameters = resources_->buffers[0].get(), .lights = resources_->buffers[1].get(),
        .candidates = resources_->buffers[2].get(), .cells = resources_->buffers[3].get(),
        .lightIndices = resources_->buffers[4].get(), .params = params, .sourceView = view, .frameSlot = frameSlot,
        .sourceLightGeneration = scene.drawSet().lightGeneration,
        .sourceLightRevision = scene.drawSet().lightRevision,
        .sourcePrepareCount = visible->stats.prepareCount, .buildRevision = nextBuildRevision_++};
    if (nextBuildRevision_ == 0) { nextBuildRevision_ = 1; }
    return {};
}

const ClusterLightGridSnapshot* ClusterLightGrid::snapshot(const GPUScene& scene) const
{
    if (!snapshot_.valid() || resources_ == nullptr || resources_->cancelled) { return nullptr; }
    const auto* visible = scene.visibleDrawSet(snapshot_.sourceView, snapshot_.frameSlot);
    return visible != nullptr && visible->stats.prepareCount == snapshot_.sourcePrepareCount &&
        visible->lights.validFor(snapshot_.sourceLightGeneration, snapshot_.sourceLightRevision) &&
        scene.drawSet().lightGeneration == snapshot_.sourceLightGeneration &&
        scene.drawSet().lightRevision == snapshot_.sourceLightRevision ? &snapshot_ : nullptr;
}

Result ClusterLightGrid::prepareShaderReload(Device& device,
    std::unique_ptr<RenderSubsystemShaderReload>& outReload, std::string& log)
{
    outReload.reset();
    std::vector<uint32_t> nextSpirv;
    Result result = compileClusterLightGridProgram(nextSpirv, log);
    if (!result) { return result; }
    ComputeProgram nextProgram;
    result = initializeClusterLightGridProgram(device, nextProgram, nextSpirv, log);
    if (!result) { return result; }
    outReload = std::make_unique<ShaderReload>(*this, std::move(nextProgram), std::move(nextSpirv));
    return {};
}

void ClusterLightGrid::clear(RenderSubsystemHost* host)
{
    if (host != nullptr) { host->retire(resources_); }
    resources_.reset();
    publication_.reset();
    snapshot_ = {};
    program_.clear();
    programSpirv_.clear();
}

} // namespace metallic::render
