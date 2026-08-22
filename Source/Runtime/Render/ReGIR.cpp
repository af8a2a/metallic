#include "Runtime/Render/ReGIR.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

inline constexpr const char* kBuildReGIRShaderModuleName = "BuildReGIR";
inline constexpr const char* kBuildReGIREntryPoint = "buildReGIRMain";
inline constexpr uint32_t kReGIRBuildGroupSize = 256;

struct BuildReGIRPush {
    uint32_t lightCount = 0;
    uint32_t gridSize = 0;
    uint32_t lightsPerCell = 0;
    uint32_t buildSamples = 0;
    uint32_t frameIndex = 0;
    uint32_t animateLights = 0;
    uint32_t lightSlotCount = 0;
    uint32_t padding0 = 0;
    float sceneCenterRadius[4] = {};
    float lightIntensity = 1.0f;
    float samplingJitter = 1.0f;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

static_assert(sizeof(BuildReGIRPush) == 64);

std::string resultMessage(std::string_view label, Result result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

} // namespace

bool ReGIRGridLayout::valid() const
{
    return gridSize != 0 &&
        lightsPerCell != 0 &&
        cellCount != 0 &&
        lightSlotCount != 0 &&
        bufferByteSize >= static_cast<uint64_t>(kReGIRHeaderRecordCount) * kReGIRRecordByteSize;
}

ReGIRGridLayout computeReGIRGridLayout(uint32_t gridSize, uint32_t lightsPerCell)
{
    ReGIRGridLayout layout;
    if (gridSize == 0 || lightsPerCell == 0) {
        return {};
    }
    const uint64_t gridPlane = static_cast<uint64_t>(gridSize) * gridSize;
    if (gridPlane > std::numeric_limits<uint32_t>::max() / gridSize) {
        return {};
    }
    const uint64_t cellCount = gridPlane * gridSize;
    if (cellCount > std::numeric_limits<uint32_t>::max() / lightsPerCell) {
        return {};
    }
    const uint64_t lightSlotCount = cellCount * lightsPerCell;
    const uint64_t recordCount = lightSlotCount + kReGIRHeaderRecordCount;

    layout.gridSize = gridSize;
    layout.lightsPerCell = lightsPerCell;
    layout.cellCount = static_cast<uint32_t>(cellCount);
    layout.lightSlotCount = static_cast<uint32_t>(lightSlotCount);
    layout.bufferByteSize = recordCount * kReGIRRecordByteSize;
    return layout;
}

struct ReGIRLightSelector::Impl {
    ComputeProgram program;
    ReGIRGridLayout layout;
    std::unique_ptr<Buffer> buffer;
    std::vector<std::unique_ptr<Buffer>> retiredBuffers;
    ResourceState state = ResourceState::Undefined;

    void clearGrid()
    {
        layout = {};
        buffer.reset();
        retiredBuffers.clear();
        state = ResourceState::Undefined;
    }
};

ReGIRLightSelector::ReGIRLightSelector()
    : impl_(std::make_unique<Impl>())
{
}

ReGIRLightSelector::~ReGIRLightSelector() = default;
ReGIRLightSelector::ReGIRLightSelector(ReGIRLightSelector&&) noexcept = default;
ReGIRLightSelector& ReGIRLightSelector::operator=(ReGIRLightSelector&&) noexcept = default;

Result ReGIRLightSelector::initialize(Device& device, std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->program.valid()) {
        return {};
    }

    ShaderCompileResult compileResult;
    const Result compile = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = kBuildReGIRShaderModuleName,
            .entryPointName = kBuildReGIREntryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
        },
        compileResult);
    if (!compile) {
        log = resultMessage("compileSlangShaderToSpirv(BuildReGIR)", compile);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        return compile;
    }

    const ComputeProgramBindingDesc bindings[] = {
        {.binding = 0, .kind = ComputeResourceBindingKind::SampledImage},
        {.binding = 1, .kind = ComputeResourceBindingKind::StorageBuffer},
    };
    return impl_->program.initialize(
        device,
        ComputeProgramDesc{
            .spirv = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            .pushConstantSize = sizeof(BuildReGIRPush),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .debugName = "BuildReGIR",
        },
        log);
}

Result ReGIRLightSelector::ensureGrid(
    Device& device,
    uint32_t gridSize,
    uint32_t lightsPerCell,
    std::string& log)
{
    if (impl_ == nullptr || !impl_->program.valid()) {
        log = "ReGIR compute program is not initialized";
        return makeError(Error::InvalidArgument);
    }

    const ReGIRGridLayout nextLayout = computeReGIRGridLayout(gridSize, lightsPerCell);
    if (!nextLayout.valid()) {
        log = "ReGIR grid layout is invalid or exceeds uint32_t addressing";
        return makeError(Error::InvalidArgument);
    }
    if (impl_->buffer != nullptr &&
        impl_->layout.gridSize == nextLayout.gridSize &&
        impl_->layout.lightsPerCell == nextLayout.lightsPerCell) {
        return {};
    }

    std::unique_ptr<Buffer> nextBuffer;
    const Result result = device.createBuffer(
        BufferDesc{
            .size = nextLayout.bufferByteSize,
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        },
        nextBuffer);
    if (!result || nextBuffer == nullptr) {
        log = resultMessage("createBuffer(ReGIR light selector)", result);
        return result ? makeError(Error::Failure) : result;
    }

    if (impl_->buffer != nullptr) {
        impl_->retiredBuffers.push_back(std::move(impl_->buffer));
    }
    impl_->buffer = std::move(nextBuffer);
    impl_->layout = nextLayout;
    impl_->state = ResourceState::Undefined;
    return {};
}

Result ReGIRLightSelector::build(
    CommandBuffer& commandBuffer,
    TextureView& localLightPdf,
    const ReGIRBuildParameters& parameters)
{
    if (!valid() || parameters.lightCount == 0 || parameters.buildSamples == 0) {
        return makeError(Error::InvalidArgument);
    }

    BufferBarrierDesc toGeneral{
        .buffer = impl_->buffer.get(),
        .before = impl_->state,
        .after = ResourceState::General,
        .offset = 0,
        .size = impl_->layout.bufferByteSize,
    };
    commandBuffer.barrier(BarrierDesc{.buffers = &toGeneral, .bufferCount = 1});
    impl_->state = ResourceState::General;

    TextureView* const pdfViews[] = {&localLightPdf};
    const ComputeDispatchBinding bindings[] = {
        {
            .binding = 0,
            .textureViews = pdfViews,
            .textureViewCount = static_cast<uint32_t>(std::size(pdfViews)),
        },
        {.binding = 1, .buffer = impl_->buffer.get()},
    };
    BuildReGIRPush push;
    push.lightCount = parameters.lightCount;
    push.gridSize = impl_->layout.gridSize;
    push.lightsPerCell = impl_->layout.lightsPerCell;
    push.buildSamples = parameters.buildSamples;
    push.frameIndex = parameters.frameIndex;
    push.animateLights = parameters.animateLights ? 1u : 0u;
    push.lightSlotCount = impl_->layout.lightSlotCount;
    push.sceneCenterRadius[0] = parameters.sceneCenter[0];
    push.sceneCenterRadius[1] = parameters.sceneCenter[1];
    push.sceneCenterRadius[2] = parameters.sceneCenter[2];
    push.sceneCenterRadius[3] = parameters.sceneRadius;
    push.lightIntensity = parameters.lightIntensity;
    push.samplingJitter = parameters.samplingJitter;

    Result result = impl_->program.dispatch(ComputeDispatchDesc{
        .commandBuffer = &commandBuffer,
        .bindings = bindings,
        .bindingCount = static_cast<uint32_t>(std::size(bindings)),
        .pushData = &push,
        .pushDataSize = sizeof(push),
        .groupCountX = (impl_->layout.lightSlotCount + kReGIRBuildGroupSize - 1u) /
            kReGIRBuildGroupSize,
        .groupCountY = 1,
        .groupCountZ = 1,
    });

    BufferBarrierDesc toShaderRead{
        .buffer = impl_->buffer.get(),
        .before = impl_->state,
        .after = ResourceState::ShaderRead,
        .offset = 0,
        .size = impl_->layout.bufferByteSize,
    };
    commandBuffer.barrier(BarrierDesc{.buffers = &toShaderRead, .bufferCount = 1});
    impl_->state = ResourceState::ShaderRead;
    return result;
}

void ReGIRLightSelector::clear()
{
    if (impl_ != nullptr) {
        impl_->program.clear();
        impl_->clearGrid();
    }
}

bool ReGIRLightSelector::valid() const
{
    return impl_ != nullptr &&
        impl_->program.valid() &&
        impl_->layout.valid() &&
        impl_->buffer != nullptr;
}

Buffer* ReGIRLightSelector::buffer() const
{
    return impl_ != nullptr ? impl_->buffer.get() : nullptr;
}

const ReGIRGridLayout& ReGIRLightSelector::layout() const
{
    static const ReGIRGridLayout kEmptyLayout;
    return impl_ != nullptr ? impl_->layout : kEmptyLayout;
}

} // namespace metallic::render
