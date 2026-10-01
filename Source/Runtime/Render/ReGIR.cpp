#include "Runtime/Render/Core/LightingKernelParameters.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/ReGIR.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <array>
#include <cmath>
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

inline constexpr const char* kBuildReGIRShaderModuleName = "Features/Lighting/BuildReGIR";
inline constexpr const char* kBuildReGIREntryPoint = "buildReGIRMain";
inline constexpr uint32_t kReGIRBuildGroupSize = 256;

std::string resultMessage(std::string_view label, Result<> result)
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
    ComputeKernel program;
    Device* device = nullptr;
    ReGIRGridLayout layout;
    std::shared_ptr<Buffer> buffer;
    ResourceState state = ResourceState::Undefined;

    void clearGrid()
    {
        layout = {};
        buffer.reset();
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

Result<> ReGIRLightSelector::initialize(Device& device, std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->program.valid()) {
        return {};
    }

    ShaderCompileResult compileResult;
    const Result<> compile = compileSlangShaderToSpirv(SlangShaderDesc{
            .moduleName = kBuildReGIRShaderModuleName,
            .entryPointName = kBuildReGIREntryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
    if (!compile) {
        log = resultMessage("compileSlangShaderToSpirv(BuildReGIR)", compile);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        return compile;
    }

    impl_->device = &device;
    return impl_->program.initialize(
        device,
        ComputeKernelDesc{
            .spirv = compileResult.spirv,
            .parameters = parameterAbi<BuildReGIRParams>(kBuildReGIRABI, ParameterTransport::InlinePush),
            .debugName = "BuildReGIR",
        },
        log);
}

Result<> ReGIRLightSelector::ensureGrid(
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
    const Result<> result = device.createBuffer(BufferDesc{
            .size = nextLayout.bufferByteSize,
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
        }).transform([&](auto rhiValue) { nextBuffer = std::move(rhiValue); });
    if (!result || nextBuffer == nullptr) {
        log = resultMessage("createBuffer(ReGIR light selector)", result);
        return result ? makeError(Error::Failure) : result;
    }

    impl_->buffer = std::move(nextBuffer);
    impl_->layout = nextLayout;
    impl_->state = ResourceState::Undefined;
    return {};
}

Result<> ReGIRLightSelector::build(
    CommandBuffer& commandBuffer,
    TextureView& localLightPdf,
    Buffer& punctualLights,
    const ReGIRBuildParameters& parameters)
{
    constexpr uint64_t kPunctualLightByteSize = 64;
    if (!valid() || parameters.buildSamples == 0 ||
        !std::isfinite(parameters.sceneRadius) || parameters.sceneRadius <= 0.0f ||
        !std::isfinite(parameters.samplingJitter) || parameters.samplingJitter < 0.0f ||
        !std::isfinite(parameters.sceneCenter[0]) || !std::isfinite(parameters.sceneCenter[1]) ||
        !std::isfinite(parameters.sceneCenter[2]) ||
        !hasFlag(punctualLights.desc().usage, BufferUsageBits::Storage) ||
        punctualLights.desc().size < (uint64_t(parameters.lightCount) + 1u) * kPunctualLightByteSize) {
        return makeError(Error::InvalidArgument);
    }
    if (auto* frame = commandBuffer.frameContext()) {
        if (!frame->recording()) { return makeError(Error::InvalidArgument); }

    }
    commandBuffer.hostWriteBarrier();

    BufferBarrierDesc toGeneral{
        .buffer = impl_->buffer.get(),
        .before = resourceSyncScope(impl_->state, PipelineStageBits::AllCommands),
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .range = {.offset = 0, .size = impl_->layout.bufferByteSize},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{.buffers = {&toGeneral, 1}}); !commandResult) { return commandResult; }
    impl_->state = ResourceState::General;

    BuildReGIRPush push{};
    push.lightCount = parameters.lightCount;
    push.gridSize = impl_->layout.gridSize;
    push.lightsPerCell = impl_->layout.lightsPerCell;
    push.buildSamples = parameters.buildSamples;
    push.frameIndex = parameters.frameIndex;
    push.lightSlotCount = impl_->layout.lightSlotCount;
    push.sceneCenterRadius[0] = parameters.sceneCenter[0];
    push.sceneCenterRadius[1] = parameters.sceneCenter[1];
    push.sceneCenterRadius[2] = parameters.sceneCenter[2];
    push.sceneCenterRadius[3] = parameters.sceneRadius;
    push.samplingJitter = parameters.samplingJitter;

    auto registry = impl_->device->resourceRegistry();
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(*impl_->device, **registry, commandBuffer.frameContext());
    const BuildReGIRParams params{
        .localLightPdf = writer.sampledImage(&localLightPdf),
        .output = writer.dataBuffer(impl_->buffer.get(), 16, 16),
        .lights = writer.dataBuffer(&punctualLights, 64, 16),
        .settings = push,
    };
    auto encoded = writer.encode(params, kBuildReGIRABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    Result<> result = impl_->program.dispatch(commandBuffer, *encoded,
        static_cast<uint32_t>((uint64_t(impl_->layout.lightSlotCount) + kReGIRBuildGroupSize - 1u) / kReGIRBuildGroupSize));

    BufferBarrierDesc toShaderRead{
        .buffer = impl_->buffer.get(),
        .before = resourceSyncScope(impl_->state, PipelineStageBits::AllCommands),
        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
        .range = {.offset = 0, .size = impl_->layout.bufferByteSize},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{.buffers = {&toShaderRead, 1}}); !commandResult) { return commandResult; }
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
