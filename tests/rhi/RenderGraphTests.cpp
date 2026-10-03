#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/NamedResourceLayouts.h"
#include "RHITest.h"
#include "Editor/StreamSceneOpen.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Core/HistoryResources.h"

#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/ReGIR.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::tests {
namespace {

constexpr const char* kShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kBindlessSmokeShaderModuleName = "Features/SmokeTests/BindlessSmoke";
constexpr const char* kBindlessSmokeVertexEntryPoint = "bindlessSmokeVertexMain";
constexpr const char* kBindlessSmokeFragmentEntryPoint = "bindlessSmokeFragmentMain";
constexpr uint32_t kSPIRVMagic = 0x07230203u;
constexpr uint32_t kSPIRVVersion16 = 0x00010600u;
constexpr uint16_t kSPIRVOpExtension = 10u;
constexpr uint16_t kSPIRVOpExtInstImport = 11u;
constexpr uint16_t kSPIRVOpCapability = 17u;
constexpr uint16_t kSPIRVOpRayQueryGetIntersectionTriangleVertexPositionsKhr = 5340u;
constexpr uint32_t kSPIRVRayQueryPositionFetchKhr = 5391u;
constexpr uint16_t kSPIRVOpRayQueryGetIntersectionClusterIdNv = 5345u;
constexpr uint32_t kSPIRVRayTracingClusterAccelerationStructureNv = 5437u;

// Native DescriptorHandle shaders expose only Slang's unbounded heap arrays.
// A scalar/fixed-array descriptor here would silently restore a per-pass Vulkan
// binding, even if its binding number happened to match a heap's number.
bool hasNativeComputeResourceInterface(const std::vector<uint32_t>& words)
{
    if (words.size() < 5 || words[0] != kSPIRVMagic) { return false; }
    bool heapCapability = false, heapBuiltin = false, deviceAddresses = false;
    uint32_t pushBlocks = 0;
    for (size_t offset = 5; offset < words.size();) {
        const uint32_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 0xffff;
        if (count == 0 || count > words.size() - offset) { return false; }
        const uint32_t* instruction = words.data() + offset;
        if (opcode == 17 && count == 2) {
            heapCapability |= instruction[1] == 5128; // DescriptorHeapEXT
            deviceAddresses |= instruction[1] == 5347; // PhysicalStorageBufferAddresses
        }
        if (opcode == 71 && count >= 4) {
            if (instruction[2] == 33 || instruction[2] == 34) { return false; }
            if (instruction[2] == 11 && (instruction[3] == 5122 || instruction[3] == 5123)) {
                heapBuiltin = true; // SamplerHeapEXT / ResourceHeapEXT
            }
        }
        if (opcode == 59 && count >= 4 && instruction[3] == 9) { ++pushBlocks; }
        offset += count;
    }
    return heapCapability && heapBuiltin && deviceAddresses && pushBlocks == 1;
}

render::EnvironmentSettings sampleEnvironmentSettings(const render::RenderSampleDesc& desc)
{
    if (!desc.environment.has_value()) {
        return {};
    }
    const render::RenderSampleEnvironmentDesc& sampleEnvironment = *desc.environment;
    render::EnvironmentSettings environment{
        .enabled = sampleEnvironment.enabled,
        .path = sampleEnvironment.path,
        .intensity = sampleEnvironment.intensity,
        .rotationDegrees = sampleEnvironment.rotationDegrees,
        .visible = sampleEnvironment.visible,
    };
    if (!environment.path.empty() && environment.path.is_relative()) {
        environment.path = std::filesystem::path(PROJECT_SOURCE_DIR) / environment.path;
    }
    return environment;
}

class ConfigurableRenderSubsystemProbe final : public render::IRenderSubsystem {
public:
    struct Desc {
        uint32_t value = 0;
    };

    static constexpr render::RenderSubsystemId kSubsystemId = "test.configurable";

    render::Result<> initialize(
        const render::RenderSubsystemInitContext& context,
        std::string&) override
    {
        const Desc* desc = context.host.configuration<ConfigurableRenderSubsystemProbe>();
        observedValue = desc != nullptr ? desc->value : 0;
        return {};
    }

    uint32_t observedValue = 0;
};

bool spirvContainsOpcode(const std::vector<uint32_t>& spirv, uint16_t expectedOpcode)
{
    for (size_t wordIndex = 5; wordIndex < spirv.size();) {
        const uint32_t instruction = spirv[wordIndex];
        const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16u);
        const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);
        if (wordCount == 0 || wordIndex + wordCount > spirv.size()) {
            return false;
        }
        if (opcode == expectedOpcode) {
            return true;
        }
        wordIndex += wordCount;
    }
    return false;
}

bool spirvContainsCapability(const std::vector<uint32_t>& spirv, uint32_t expectedCapability)
{
    for (size_t wordIndex = 5; wordIndex < spirv.size();) {
        const uint32_t instruction = spirv[wordIndex];
        const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16u);
        const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);
        if (wordCount == 0 || wordIndex + wordCount > spirv.size()) {
            return false;
        }
        if (opcode == kSPIRVOpCapability && wordCount >= 2 && spirv[wordIndex + 1] == expectedCapability) {
            return true;
        }
        wordIndex += wordCount;
    }
    return false;
}

bool spirvContainsExtension(const std::vector<uint32_t>& spirv, std::string_view expectedExtension)
{
    for (size_t wordIndex = 5; wordIndex < spirv.size();) {
        const uint32_t instruction = spirv[wordIndex];
        const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16u);
        const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);
        if (wordCount == 0 || wordIndex + wordCount > spirv.size()) {
            return false;
        }
        if (opcode == kSPIRVOpExtension && wordCount >= 2) {
            const char* begin = reinterpret_cast<const char*>(spirv.data() + wordIndex + 1);
            const char* limit = begin + static_cast<size_t>(wordCount - 1) * sizeof(uint32_t);
            const char* end = std::find(begin, limit, '\0');
            if (end != limit && std::string_view(begin, static_cast<size_t>(end - begin)) == expectedExtension) {
                return true;
            }
        }
        wordIndex += wordCount;
    }
    return false;
}

bool spirvContainsExtendedInstructionSet(
    const std::vector<uint32_t>& spirv,
    std::string_view expectedSet)
{
    for (size_t wordIndex = 5; wordIndex < spirv.size();) {
        const uint32_t instruction = spirv[wordIndex];
        const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16u);
        const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);
        if (wordCount == 0 || wordIndex + wordCount > spirv.size()) {
            return false;
        }
        if (opcode == kSPIRVOpExtInstImport && wordCount >= 3) {
            const char* begin = reinterpret_cast<const char*>(spirv.data() + wordIndex + 2);
            const char* limit = begin + static_cast<size_t>(wordCount - 2) * sizeof(uint32_t);
            const char* end = std::find(begin, limit, '\0');
            if (end != limit && std::string_view(begin, static_cast<size_t>(end - begin)) == expectedSet) {
                return true;
            }
        }
        wordIndex += wordCount;
    }
    return false;
}

bool spirvContainsCaptureDebugInfo(
    const std::vector<uint32_t>& spirv,
    std::string_view sourceName,
    std::string_view entryPointName)
{
    std::unordered_map<uint32_t, std::string_view> strings;
    std::unordered_set<uint32_t> sourceIds;
    uint32_t debugSet = 0;
    bool hasFunction = false;
    bool hasLine = false;
    for (size_t wordIndex = 5; wordIndex < spirv.size();) {
        const uint32_t wordCount = spirv[wordIndex] >> 16u;
        const uint32_t opcode = spirv[wordIndex] & 0xffffu;
        if (wordCount == 0 || wordIndex + wordCount > spirv.size()) {
            return false;
        }
        if ((opcode == 7u || opcode == 11u) && wordCount >= 3u) {
            const char* begin = reinterpret_cast<const char*>(spirv.data() + wordIndex + 2);
            const char* limit = begin + (wordCount - 2u) * sizeof(uint32_t);
            const char* end = std::find(begin, limit, '\0');
            if (end != limit) {
                const std::string_view text(begin, end - begin);
                if (opcode == 7u) {
                    strings.emplace(spirv[wordIndex + 1], text);
                } else if (text == "NonSemantic.Shader.DebugInfo.100") {
                    debugSet = spirv[wordIndex + 1];
                }
            }
        }
        // OpExtInst: DebugSource must embed text, not just a filesystem path.
        if (opcode == 12u && wordCount >= 7u && debugSet != 0 &&
            spirv[wordIndex + 3] == debugSet && spirv[wordIndex + 4] == 35u &&
            strings[spirv[wordIndex + 5]].ends_with(sourceName) &&
            strings[spirv[wordIndex + 6]].find(entryPointName) != std::string_view::npos) {
            sourceIds.insert(spirv[wordIndex + 2]);
        }
        wordIndex += wordCount;
    }
    for (size_t wordIndex = 5; wordIndex < spirv.size();) {
        const uint32_t wordCount = spirv[wordIndex] >> 16u;
        const uint32_t opcode = spirv[wordIndex] & 0xffffu;
        if (opcode == 12u && wordCount >= 5u && spirv[wordIndex + 3] == debugSet) {
            // DebugFunction names the entry and ties it to the embedded source.
            hasFunction |= wordCount >= 8u && spirv[wordIndex + 4] == 20u &&
                strings[spirv[wordIndex + 5]] == entryPointName &&
                sourceIds.contains(spirv[wordIndex + 7]);
            hasLine |= wordCount >= 10u && spirv[wordIndex + 4] == 103u &&
                sourceIds.contains(spirv[wordIndex + 5]);
        }
        wordIndex += wordCount;
    }
    return hasFunction && hasLine;
}

render::Result<> createSlangShaderModule(
    render::Device& device,
    const char* moduleName,
    const char* entryPointName,
    std::unique_ptr<render::ShaderModule>& outShaderModule,
    std::string& log)
{
    render::ShaderCompileResult compileResult;
    render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kShaderSearchPath,
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
    if (!result) {
        log += std::string("compileSlangShaderToSpirv(") + moduleName + "." + entryPointName + ") returned ";
        log += toString(result);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        log += '\n';
        return result;
    }

    return device.createShaderModule(render::ShaderModuleDesc{
        .spirv = compileResult.spirv,
    }).transform([&](auto rhiValue) { outShaderModule = std::move(rhiValue); });
}

render::Result<> writeHostBuffer(render::Buffer& buffer, const void* data, uint64_t byteSize)
{
    if (byteSize > buffer.desc().size || (byteSize > 0 && data == nullptr)) {
        return render::makeError(render::Error::InvalidArgument);
    }
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return render::makeError(render::Error::Failure);
    }
    if (byteSize > 0) {
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        buffer.flush({0, byteSize});
    }
    buffer.unmap();
    return {};
}

bool readHostBuffer(render::Buffer& buffer, void* outData, uint64_t byteSize)
{
    if (byteSize > buffer.desc().size || (byteSize > 0 && outData == nullptr)) {
        return false;
    }
    buffer.invalidate({0, byteSize});
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return false;
    }
    if (byteSize > 0) {
        std::memcpy(outData, mapped, static_cast<size_t>(byteSize));
    }
    buffer.unmap();
    return true;
}

class TestInputOutputPass final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addInput("input", "Required input");
        reflection.addOutput("color", "Output color");
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
};

class TestBufferOutputPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Buffer output")
            .buffer(16)
            .storageReadWrite();
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
};

class TestBufferInputPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferInput("data", "Buffer input")
            .buffer(16)
            .shaderRead();
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
};

class TestBindlessSamplePass final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBindlessSampledInput("source", "Source bindless sampled texture");
        reflection.addOutput("color", "Bindless sampled output")
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return render::makeError(render::Error::InvalidArgument);
        }

        render::Result<> result = createSlangShaderModule(
            *context.device,
            kBindlessSmokeShaderModuleName,
            kBindlessSmokeVertexEntryPoint,
            vertexShader_,
            log);
        if (!result) {
            return result;
        }
        result = createSlangShaderModule(
            *context.device,
            kBindlessSmokeShaderModuleName,
            kBindlessSmokeFragmentEntryPoint,
            fragmentShader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(render::GraphicsPipelineDesc{
            .vertexShader = {vertexShader_.get()},
            .fragmentShader = {fragmentShader_.get()},
            .colorFormat = render::Format::RGBA8Unorm,
            .topology = render::PrimitiveTopology::TriangleList,
            .usesBindlessHeap = true,
        }).transform([&](auto rhiValue) { pipeline_ = std::move(rhiValue); });
        if (!result) {
            log += std::string("createGraphicsPipeline(bindless graph pass) returned ") + toString(result) + '\n';
        }
        return result;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::BindlessHandle* sourceHandle = context.bindlessInput("source");
        render::TextureHandle color = context.outputTexture("color");
        if (sourceHandle == nullptr ||
            sourceHandle->kind != render::BindlessHandleKind::SampledImage ||
            !color.valid() ||
            pipeline_ == nullptr) {
            return render::makeError(render::Error::InvalidArgument);
        }

        const render::Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        render::RenderingAttachmentDesc attachment{
            .view = color.view(),
            .layout = render::TextureLayout::ColorAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearColor = render::ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        if (auto commandResult = context.commandBuffer().beginRendering(render::RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = {&attachment, 1},
        }); !commandResult) { return commandResult; }
        context.commandBuffer().setViewport(render::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        if (auto commandResult = context.commandBuffer().bindExecution((pipeline_)->execution()); !commandResult) { return commandResult; }
        context.commandBuffer().pushBindlessData(&sourceHandle->shaderIndex, sizeof(sourceHandle->shaderIndex));
        context.commandBuffer().draw(3);
        context.commandBuffer().endRendering();
        return {};
    }

private:
    std::unique_ptr<render::ShaderModule> vertexShader_;
    std::unique_ptr<render::ShaderModule> fragmentShader_;
    std::unique_ptr<render::GraphicsPipeline> pipeline_;
};

uint32_t& testResizeCompileCount()
{
    static uint32_t count = 0;
    return count;
}

class TestResizeCompilePass final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addOutput("color", "Resize output color");
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext&, std::string&) override
    {
        ++testResizeCompileCount();
        return {};
    }

    render::Result<> execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
};

// Models a stream owner shared by a pass and submitted frame slots.
std::weak_ptr<render::Buffer> testRetainedSceneBuffer;
class TestRetainedScenePass final : public render::RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addOutput("color", "Retirement test output");
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        if (!testRetainedSceneBuffer.expired()) {
            log = "Previous scene buffer is still retained before replacement allocation";
            return render::makeError(render::Error::OutOfMemory);
        }
        std::unique_ptr<render::Buffer> buffer;
        auto result = context.device->createBuffer(render::BufferDesc{
            .size = 16ull * 1024 * 1024,
            .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::Device,
        }).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
        if (!result) { return result; }
        buffer_ = std::move(buffer);
        testRetainedSceneBuffer = buffer_;
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto* frame = metallic::render::RenderFrameContext::from(context.commandBuffer());
        if (!frame) { return render::makeError(render::Error::InvalidArgument); }
        frame->retain(buffer_);
        return {};
    }
private:
    std::shared_ptr<render::Buffer> buffer_;
};

struct TestTextureExtentExecutionState {
    uint32_t producerContextWidth = 0;
    uint32_t producerContextHeight = 0;
    uint32_t producerDisplayWidth = 0;
    uint32_t producerDisplayHeight = 0;
    uint32_t producerOutputWidth = 0;
    uint32_t producerOutputHeight = 0;
    uint32_t relayContextWidth = 0;
    uint32_t relayContextHeight = 0;
    uint32_t relayDisplayWidth = 0;
    uint32_t relayDisplayHeight = 0;
    uint32_t relayInputWidth = 0;
    uint32_t relayInputHeight = 0;
    uint32_t relayOutputWidth = 0;
    uint32_t relayOutputHeight = 0;
    uint32_t consumerContextWidth = 0;
    uint32_t consumerContextHeight = 0;
    uint32_t consumerDisplayWidth = 0;
    uint32_t consumerDisplayHeight = 0;
    uint32_t consumerInputWidth = 0;
    uint32_t consumerInputHeight = 0;
    uint32_t consumerOutputWidth = 0;
    uint32_t consumerOutputHeight = 0;
};

TestTextureExtentExecutionState& testTextureExtentExecutionState()
{
    static TestTextureExtentExecutionState state;
    return state;
}

class TestTextureExtentProducerPass final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        const render::RenderGraphProperties& passProperties = properties();
        const uint32_t outputWidth = passProperties.is_object()
            ? passProperties.value("outputWidth", 0u)
            : 0u;
        const uint32_t outputHeight = passProperties.is_object()
            ? passProperties.value("outputHeight", 0u)
            : 0u;
        const bool outputRgba16 = passProperties.is_object()
            ? passProperties.value("outputRgba16", false)
            : false;

        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Texture extent producer")
            .texture2D(outputWidth, outputHeight)
            .storageReadWrite()
            .format = outputRgba16
                ? render::Format::RGBA16Sfloat
                : render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::TextureHandle color = context.outputTexture("color");
        if (!color.valid()) {
            return render::makeError(render::Error::InvalidArgument);
        }

        TestTextureExtentExecutionState& state = testTextureExtentExecutionState();
        state.producerContextWidth = context.width();
        state.producerContextHeight = context.height();
        state.producerDisplayWidth = context.displayWidth();
        state.producerDisplayHeight = context.displayHeight();
        state.producerOutputWidth = color.desc().width;
        state.producerOutputHeight = color.desc().height;
        return {};
    }
};

class TestTextureExtentRelayPass final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        const render::RenderGraphProperties& passProperties = properties();
        const uint32_t outputWidth = passProperties.is_object()
            ? passProperties.value("outputWidth", 0u)
            : 0u;
        const uint32_t outputHeight = passProperties.is_object()
            ? passProperties.value("outputHeight", 0u)
            : 0u;

        render::RenderPassReflection reflection;
        reflection.addTextureInput("input", "Implicit-sized texture extent relay input")
            .storageReadWrite()
            .format = render::Format::RGBA8Unorm;
        reflection.addTextureOutput("color", "Texture extent relay output")
            .texture2D(outputWidth, outputHeight)
            .storageReadWrite()
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::TextureHandle input = context.inputTexture("input");
        const render::TextureHandle color = context.outputTexture("color");
        if (!input.valid() || !color.valid()) {
            return render::makeError(render::Error::InvalidArgument);
        }

        TestTextureExtentExecutionState& state = testTextureExtentExecutionState();
        state.relayContextWidth = context.width();
        state.relayContextHeight = context.height();
        state.relayDisplayWidth = context.displayWidth();
        state.relayDisplayHeight = context.displayHeight();
        state.relayInputWidth = input.desc().width;
        state.relayInputHeight = input.desc().height;
        state.relayOutputWidth = color.desc().width;
        state.relayOutputHeight = color.desc().height;
        return {};
    }
};

class TestTextureExtentConsumerPass final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        const render::RenderGraphProperties& passProperties = properties();
        const uint32_t inputWidth = passProperties.is_object()
            ? passProperties.value("inputWidth", 0u)
            : 0u;
        const uint32_t inputHeight = passProperties.is_object()
            ? passProperties.value("inputHeight", 0u)
            : 0u;

        render::RenderPassReflection reflection;
        reflection.addTextureInput("input", "Explicit-sized texture extent consumer input")
            .texture2D(inputWidth, inputHeight)
            .storageReadWrite()
            .format = render::Format::RGBA8Unorm;
        reflection.addTextureOutput("color", "Default-sized texture extent consumer output")
            .storageReadWrite()
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::TextureHandle input = context.inputTexture("input");
        const render::TextureHandle color = context.outputTexture("color");
        if (!input.valid() || !color.valid()) {
            return render::makeError(render::Error::InvalidArgument);
        }

        TestTextureExtentExecutionState& state = testTextureExtentExecutionState();
        state.consumerContextWidth = context.width();
        state.consumerContextHeight = context.height();
        state.consumerDisplayWidth = context.displayWidth();
        state.consumerDisplayHeight = context.displayHeight();
        state.consumerInputWidth = input.desc().width;
        state.consumerInputHeight = input.desc().height;
        state.consumerOutputWidth = color.desc().width;
        state.consumerOutputHeight = color.desc().height;
        return {};
    }
};

struct TestShaderReloadState {
    bool failCompile = false;
    bool changeReflection = false;
    uint32_t nextInstanceId = 0;
    uint32_t compileCount = 0;
    uint32_t lastSuccessfulCompileInstanceId = 0;
    std::vector<uint32_t> destroyedInstanceIds;
};

TestShaderReloadState& testShaderReloadState()
{
    static TestShaderReloadState state;
    return state;
}

class TestShaderReloadPass final : public render::RasterPass {
public:
    TestShaderReloadPass()
        : instanceId_(++testShaderReloadState().nextInstanceId)
    {
    }

    ~TestShaderReloadPass() override
    {
        testShaderReloadState().destroyedInstanceIds.push_back(instanceId_);
    }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addOutput("color", "Shader reload output color");
        if (testShaderReloadState().changeReflection) {
            reflection.addOutput("changedContract", "Intentionally changed shader reload contract");
        }
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext&, std::string& log) override
    {
        TestShaderReloadState& state = testShaderReloadState();
        ++state.compileCount;
        if (state.failCompile) {
            log = "intentional shader reload compile failure";
            return render::makeError(render::Error::Failure);
        }
        state.lastSuccessfulCompileInstanceId = instanceId_;
        return {};
    }

    render::Result<> execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }

private:
    uint32_t instanceId_ = 0;
};

class TestMissingSubsystemPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addOutput("color", "Missing-subsystem diagnostic output");
        return reflection;
    }

    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{
            render::RenderSubsystemId{"test.missing-required-subsystem"},
        };
        return required;
    }

    render::Result<> execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
};

struct TestTextureFeedbackState {
    std::weak_ptr<render::ScenePathTraceResources> resources;
    render::Buffer* feedback = nullptr;
    bool sharedFeedback = false;
};

TestTextureFeedbackState& testTextureFeedbackState()
{
    static TestTextureFeedbackState state;
    return state;
}

class TestTextureFeedbackPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        if (properties().value("after", false)) {
            reflection.addBufferInput("previous", "Earlier texture feedback consumer").buffer(16).shaderRead();
        }
        reflection.addBufferOutput("token", "Texture feedback consumer ordering").buffer(16).storageReadWrite();
        return reflection;
    }

    render::SceneStreamingRequirements sceneResourcesRequired(const render::RenderGraphCompileContext&) const override
    {
        return {.features = render::SceneResourceFeatureBits::Materials | render::SceneResourceFeatureBits::MaterialTextures,
            .textureFeedback = true};
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({
            .moduleName = "Features/Debug/TextureResidencyProbe", .entryPointName = "main",
            .searchPath = kShaderSearchPath}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc binding{.binding = 0};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv, .pushConstantSize = 16, .bindings = {&binding, 1},
            .requiresRayQuery = false, .resourceParameters = render::kTextureFeedbackResourceLayout}, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const auto* prepared = context.preparedScene();
        if (!prepared || !prepared->ready || !prepared->snapshot || !prepared->snapshot->pathTraceResources ||
            !prepared->textureFeedback || prepared->textureFeedback->desc().memoryLocation != render::MemoryLocation::Device) {
            return render::makeError(render::Error::InvalidArgument);
        }
        auto resources = prepared->snapshot->pathTraceResources;
        if (resources->logicalTextureIndices().empty()) { return render::makeError(render::Error::InvalidArgument); }
        auto& state = testTextureFeedbackState();
        if (context.properties().value("after", false)) {
            state.sharedFeedback = state.resources.lock() == resources && state.feedback == prepared->textureFeedback;
            if (!state.sharedFeedback) { return render::makeError(render::Error::InvalidArgument); }
        } else {
            state.resources = resources;
            state.feedback = prepared->textureFeedback;
        }
        const render::ComputeDispatchBinding binding{.binding = 0, .buffer = prepared->textureFeedback};
        const uint32_t push[]{resources->logicalTextureIndices()[0], context.properties().value("wantedMip", 0u), 1u, 0u};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {&binding, 1},
            .pushData = push, .pushDataSize = sizeof(push)});
    }

private:
    render::ComputeProgram program_;
};

class TestEnvironmentConsumerPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addOutput("color", "Environment consumer output");
        return reflection;
    }

    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{
            render::EnvironmentLightingSubsystem::kSubsystemId,
        };
        return required;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::EnvironmentLightingSubsystem* environment =
            context.subsystem<render::EnvironmentLightingSubsystem>();
        return environment != nullptr && environment->snapshot().valid()
            ? render::Result<>{}
            : render::makeError(render::Error::InvalidArgument);
    }
};

void registerTestPass()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;
    render::registerRenderGraphPassType(
        "TestInputOutputPass",
        "Test-only pass with one required input and one output",
        []() { return std::make_unique<TestInputOutputPass>(); });
    render::registerRenderGraphPassType(
        "TestBufferOutputPass",
        "Test-only pass with one buffer output",
        []() { return std::make_unique<TestBufferOutputPass>(); });
    render::registerRenderGraphPassType(
        "TestBufferInputPass",
        "Test-only pass with one buffer input",
        []() { return std::make_unique<TestBufferInputPass>(); });
    render::registerRenderGraphPassType(
        "TestBindlessSamplePass",
        "Test-only pass that samples a RenderGraph input through bindless",
        []() { return std::make_unique<TestBindlessSamplePass>(); });
    render::registerRenderGraphPassType(
        "TestRetainedScenePass", "Test submitted scene owner retirement",
        []() { return std::make_unique<TestRetainedScenePass>(); });
    render::registerRenderGraphPassType(
        "TestResizeCompilePass",
        "Test-only pass that counts RenderGraph compile calls",
        []() { return std::make_unique<TestResizeCompilePass>(); });
    render::registerRenderGraphPassType(
        "TestTextureExtentProducerPass",
        "Test-only pass with a configurable texture output extent",
        []() { return std::make_unique<TestTextureExtentProducerPass>(); });
    render::registerRenderGraphPassType(
        "TestTextureExtentRelayPass",
        "Test-only pass with an implicit input and configurable output extent",
        []() { return std::make_unique<TestTextureExtentRelayPass>(); });
    render::registerRenderGraphPassType(
        "TestTextureExtentConsumerPass",
        "Test-only pass with an explicit-sized texture input and default-sized output",
        []() { return std::make_unique<TestTextureExtentConsumerPass>(); });
    render::registerRenderGraphPassType(
        "TestShaderReloadPass",
        "Test-only pass that validates transactional shader reload",
        []() { return std::make_unique<TestShaderReloadPass>(); });
    render::registerRenderGraphPassType(
        "TestMissingSubsystemPass",
        "Test-only pass with an intentionally missing subsystem",
        []() { return std::make_unique<TestMissingSubsystemPass>(); });
    render::registerRenderGraphPassType(
        "TestTextureFeedbackPass", "Test shared texture feedback and graph epilogue submission",
        []() { return std::make_unique<TestTextureFeedbackPass>(); });
    render::registerRenderGraphPassType(
        "TestEnvironmentConsumerPass",
        "Test-only environment subsystem consumer",
        []() { return std::make_unique<TestEnvironmentConsumerPass>(); });
}

uint32_t countBrightPixels(const std::vector<uint32_t>& pixels)
{
    uint32_t brightPixelCount = 0;
    for (uint32_t pixel : pixels) {
        const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
        const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
        const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
        if (r > 120 || g > 120 || b > 120) {
            ++brightPixelCount;
        }
    }
    return brightPixelCount;
}

uint32_t countVisiblePixels(const std::vector<uint32_t>& pixels)
{
    uint32_t visiblePixelCount = 0;
    for (uint32_t pixel : pixels) {
        const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
        const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
        const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
        if (r > 8 || g > 8 || b > 8) {
            ++visiblePixelCount;
        }
    }
    return visiblePixelCount;
}
uint32_t countDistinctVisibleColorBins(const std::vector<uint32_t>& pixels)
{
    std::unordered_set<uint32_t> bins;
    for (uint32_t pixel : pixels) {
        const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
        const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
        const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
        if (r > 8 || g > 8 || b > 8) {
            bins.insert(
                (static_cast<uint32_t>(r >> 5u) << 6u) |
                (static_cast<uint32_t>(g >> 5u) << 3u) |
                static_cast<uint32_t>(b >> 5u));
        }
    }
    return static_cast<uint32_t>(bins.size());
}

uint64_t sumAbsoluteRgbDifference(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b)
{
    const size_t count = std::min(a.size(), b.size());
    uint64_t totalDifference = 0;
    for (size_t index = 0; index < count; ++index) {
        const uint32_t left = a[index];
        const uint32_t right = b[index];
        for (uint32_t channel = 0; channel < 3; ++channel) {
            const int32_t leftValue = static_cast<int32_t>((left >> (channel * 8u)) & 0xffu);
            const int32_t rightValue = static_cast<int32_t>((right >> (channel * 8u)) & 0xffu);
            totalDifference += static_cast<uint64_t>(std::abs(leftValue - rightValue));
        }
    }
    return totalDifference;
}

uint32_t packRgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    return static_cast<uint32_t>(r) |
        (static_cast<uint32_t>(g) << 8u) |
        (static_cast<uint32_t>(b) << 16u) |
        (static_cast<uint32_t>(a) << 24u);
}

template <typename T, size_t N>
bool writeBinaryArray(std::ofstream& stream, const std::array<T, N>& values)
{
    stream.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    return stream.good();
}

bool writeAlphaMaskScene(
    const std::filesystem::path& directory,
    std::filesystem::path& outPath,
    std::string& outMessage,
    bool maskedDoubleSided = true)
{
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        outMessage = "failed to create alpha mask scene directory: " + error.message();
        return false;
    }

    const std::filesystem::path imagePath = directory / "alpha_mask.png";
    std::array<uint32_t, 16> alphaPixels{};
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            const uint8_t alpha = x < 2 ? 0 : 255;
            alphaPixels[y * 4 + x] = packRgba8(255, 255, 255, alpha);
        }
    }
    const auto* alphaBytes = reinterpret_cast<const uint8_t*>(alphaPixels.data());
    if (!saveRgba8Png(imagePath, alphaBytes, 4, 4, outMessage)) {
        return false;
    }

    const std::filesystem::path binPath = directory / "alpha_mask.bin";
    std::ofstream bin(binPath, std::ios::binary);
    if (!bin) {
        outMessage = "failed to open alpha mask scene binary";
        return false;
    }

    const std::array<float, 12> frontPositions{
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
    };
    const std::array<float, 12> backPositions{
        -1.0f, -1.0f, -0.1f,
        1.0f, -1.0f, -0.1f,
        1.0f, 1.0f, -0.1f,
        -1.0f, 1.0f, -0.1f,
    };
    const std::array<float, 12> normals{
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    const std::array<float, 8> texcoords{
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
    };
    const std::array<uint32_t, 6> indices{0, 1, 2, 0, 2, 3};
    if (!writeBinaryArray(bin, frontPositions) ||
        !writeBinaryArray(bin, normals) ||
        !writeBinaryArray(bin, texcoords) ||
        !writeBinaryArray(bin, indices) ||
        !writeBinaryArray(bin, backPositions) ||
        !writeBinaryArray(bin, normals) ||
        !writeBinaryArray(bin, texcoords) ||
        !writeBinaryArray(bin, indices)) {
        outMessage = "failed to write alpha mask scene binary";
        return false;
    }
    bin.close();

    const std::filesystem::path gltfPath = directory / "alpha_mask.gltf";
    std::ofstream gltf(gltfPath);
    if (!gltf) {
        outMessage = "failed to open alpha mask glTF";
        return false;
    }

    gltf << R"json({
  "asset": { "version": "2.0", "generator": "MetallicRHITests" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0, "name": "Alpha Mask Stack" }],
  "buffers": [{ "uri": "alpha_mask.bin", "byteLength": 304 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 48, "target": 34962 },
    { "buffer": 0, "byteOffset": 96, "byteLength": 32, "target": 34962 },
    { "buffer": 0, "byteOffset": 128, "byteLength": 24, "target": 34963 },
    { "buffer": 0, "byteOffset": 152, "byteLength": 48, "target": 34962 },
    { "buffer": 0, "byteOffset": 200, "byteLength": 48, "target": 34962 },
    { "buffer": 0, "byteOffset": 248, "byteLength": 32, "target": 34962 },
    { "buffer": 0, "byteOffset": 280, "byteLength": 24, "target": 34963 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0] },
    { "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3" },
    { "bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2" },
    { "bufferView": 3, "componentType": 5125, "count": 6, "type": "SCALAR" },
    { "bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, -0.1], "max": [1, 1, -0.1] },
    { "bufferView": 5, "componentType": 5126, "count": 4, "type": "VEC3" },
    { "bufferView": 6, "componentType": 5126, "count": 4, "type": "VEC2" },
    { "bufferView": 7, "componentType": 5125, "count": 6, "type": "SCALAR" }
  ],
  "samplers": [{ "magFilter": 9728, "minFilter": 9728, "wrapS": 10497, "wrapT": 10497 }],
  "images": [{ "uri": "alpha_mask.png", "name": "Alpha Mask" }],
  "textures": [{ "source": 0, "sampler": 0, "name": "Alpha Mask Texture" }],
  "materials": [
    {
      "name": "Masked Red",
      "alphaMode": "MASK",
      "alphaCutoff": 0.5,
      "doubleSided": )json"
         << (maskedDoubleSided ? "true" : "false")
         << R"json(,
      "emissiveFactor": [1.0, 0.0, 0.0],
      "pbrMetallicRoughness": {
        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
        "baseColorTexture": { "index": 0 },
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      }
    },
    {
      "name": "Blend Blue Downgrade",
      "alphaMode": "BLEND",
      "emissiveFactor": [0.0, 0.0, 1.0],
      "pbrMetallicRoughness": {
        "baseColorFactor": [0.0, 0.0, 1.0, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      }
    },
    {
      "name": "Blend Downgrade",
      "alphaMode": "BLEND",
      "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 0.5] }
    }
  ],
  "meshes": [
    {
      "name": "Alpha Mask Mesh",
      "primitives": [
        { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3, "material": 0 },
        { "attributes": { "POSITION": 4, "NORMAL": 5, "TEXCOORD_0": 6 }, "indices": 7, "material": 1 }
      ]
    }
  ]
})json";
    gltf.close();

    outPath = gltfPath;
    outMessage.clear();
    return true;
}

bool writeTransmissionTextureScene(
    const std::filesystem::path& directory,
    std::filesystem::path& outPath,
    std::string& outMessage)
{
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        outMessage = "failed to create transmission texture scene directory: " + error.message();
        return false;
    }

    const std::array<std::pair<const char*, uint32_t>, 4> textures{
        std::pair<const char*, uint32_t>{"transmission_zero.png", packRgba8(0, 255, 255, 255)},
        std::pair<const char*, uint32_t>{"thickness_half.png", packRgba8(255, 128, 255, 255)},
        std::pair<const char*, uint32_t>{"diffuse_transmission_zero.png", packRgba8(255, 255, 255, 0)},
        std::pair<const char*, uint32_t>{"diffuse_transmission_color.png", packRgba8(64, 128, 255, 255)},
    };
    for (const auto& texture : textures) {
        std::array<uint32_t, 4> pixels{};
        for (uint32_t& pixel : pixels) {
            pixel = texture.second;
        }
        const auto* bytes = reinterpret_cast<const uint8_t*>(pixels.data());
        if (!saveRgba8Png(directory / texture.first, bytes, 2, 2, outMessage)) {
            return false;
        }
    }

    const std::filesystem::path binPath = directory / "transmission_textures.bin";
    std::ofstream bin(binPath, std::ios::binary);
    if (!bin) {
        outMessage = "failed to open transmission texture scene binary";
        return false;
    }

    const std::array<float, 12> positions{
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,
    };
    const std::array<float, 12> normals{
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    const std::array<float, 8> texcoords{
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
    };
    const std::array<uint32_t, 6> indices{0, 1, 2, 0, 2, 3};
    if (!writeBinaryArray(bin, positions) ||
        !writeBinaryArray(bin, normals) ||
        !writeBinaryArray(bin, texcoords) ||
        !writeBinaryArray(bin, indices)) {
        outMessage = "failed to write transmission texture scene binary";
        return false;
    }
    bin.close();

    const std::filesystem::path gltfPath = directory / "transmission_textures.gltf";
    std::ofstream gltf(gltfPath);
    if (!gltf) {
        outMessage = "failed to open transmission texture glTF";
        return false;
    }

    gltf << R"json({
  "asset": { "version": "2.0", "generator": "MetallicRHITests" },
  "extensionsUsed": [
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_diffuse_transmission"
  ],
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0, "name": "Transmission Texture Quad" }],
  "buffers": [{ "uri": "transmission_textures.bin", "byteLength": 152 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 48, "target": 34962 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 48, "target": 34962 },
    { "buffer": 0, "byteOffset": 96, "byteLength": 32, "target": 34962 },
    { "buffer": 0, "byteOffset": 128, "byteLength": 24, "target": 34963 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0] },
    { "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3" },
    { "bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2" },
    { "bufferView": 3, "componentType": 5125, "count": 6, "type": "SCALAR" }
  ],
  "samplers": [{ "magFilter": 9728, "minFilter": 9728, "wrapS": 10497, "wrapT": 10497 }],
  "images": [
    { "uri": "transmission_zero.png", "name": "Transmission Zero" },
    { "uri": "thickness_half.png", "name": "Thickness Half" },
    { "uri": "diffuse_transmission_zero.png", "name": "Diffuse Transmission Zero" },
    { "uri": "diffuse_transmission_color.png", "name": "Diffuse Transmission Color" }
  ],
  "textures": [
    { "source": 0, "sampler": 0, "name": "Transmission Zero Texture" },
    { "source": 1, "sampler": 0, "name": "Thickness Half Texture" },
    { "source": 2, "sampler": 0, "name": "Diffuse Transmission Zero Texture" },
    { "source": 3, "sampler": 0, "name": "Diffuse Transmission Color Texture" }
  ],
  "materials": [
    {
      "name": "Texture Gated Red",
      "doubleSided": true,
      "pbrMetallicRoughness": {
        "baseColorFactor": [1.0, 0.0, 0.0, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      },
      "extensions": {
        "KHR_materials_transmission": {
          "transmissionFactor": 1.0,
          "transmissionTexture": { "index": 0 }
        },
        "KHR_materials_volume": {
          "thicknessFactor": 0.8,
          "attenuationDistance": 4.0,
          "attenuationColor": [0.8, 0.9, 1.0],
          "thicknessTexture": { "index": 1 }
        },
        "KHR_materials_diffuse_transmission": {
          "diffuseTransmissionFactor": 1.0,
          "diffuseTransmissionColor": [1.0, 1.0, 1.0],
          "diffuseTransmissionTexture": { "index": 2 },
          "diffuseTransmissionColorTexture": { "index": 3 }
        }
      }
    }
  ],
  "meshes": [
    {
      "name": "Transmission Texture Mesh",
      "primitives": [
        { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3, "material": 0 }
      ]
    }
  ]
})json";
    gltf.close();

    outPath = gltfPath;
    outMessage.clear();
    return true;
}

class RenderGraphReflectionAPITest : public RHITest {
public:
    RenderGraphReflectionAPITest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_reflection_api";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderPassReflection reflection;
        render::RenderGraphField& texture = reflection.addTextureInput("source", "Texture source")
            .texture2D(32, 16)
            .sampledRead()
            .bindlessSampledImage()
            .setOptional();
        texture.format = render::Format::RGBA8Unorm;

        render::RenderGraphField& buffer = reflection.addBufferOutput("data", "Buffer output")
            .buffer(64, 8)
            .storageReadWrite()
            .bindlessBuffer()
            .hostReadback();

        reflection.addTextureOutput("depth", "Depth output")
            .depthStencilWrite();

        const render::RenderGraphField* foundTexture =
            reflection.findField("source", render::RenderGraphFieldVisibility::Input);
        const render::RenderGraphField* foundBuffer =
            reflection.findField("data", render::RenderGraphFieldVisibility::Output);
        const render::RenderGraphField* foundDepth =
            reflection.findField("depth", render::RenderGraphFieldVisibility::Output);
        if (foundTexture == nullptr || foundBuffer == nullptr || foundDepth == nullptr) {
            return RHITestResult::fail("reflection did not preserve fields");
        }
        if (foundTexture->resourceType != render::RenderGraphResourceType::Texture2D ||
            foundTexture->access != render::RenderGraphResourceAccess::TextureSampleRead ||
            foundTexture->bindlessAccess != render::RenderGraphBindlessAccess::SampledImage ||
            foundTexture->width != 32 ||
            foundTexture->height != 16 ||
            !foundTexture->optional) {
            return RHITestResult::fail("texture field metadata was not preserved");
        }
        if (foundBuffer->resourceType != render::RenderGraphResourceType::Buffer ||
            foundBuffer->access != render::RenderGraphResourceAccess::BufferStorageReadWrite ||
            foundBuffer->bindlessAccess != render::RenderGraphBindlessAccess::Buffer ||
            foundBuffer->size != 64 ||
            foundBuffer->structureStride != 8 ||
            foundBuffer->memoryLocation != render::MemoryLocation::HostReadback) {
            return RHITestResult::fail("buffer field metadata was not preserved");
        }
        if (foundDepth->resourceType != render::RenderGraphResourceType::Texture2D ||
            foundDepth->access != render::RenderGraphResourceAccess::TextureDepthStencilWrite ||
            foundDepth->format != render::Format::D32Sfloat ||
            foundDepth->usage != render::TextureUsageBits::DepthStencilAttachment ||
            foundDepth->state != render::ResourceState::DepthStencilAttachment) {
            return RHITestResult::fail("depth field metadata was not preserved");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphPassKindTest : public RHITest {
public:
    RenderGraphPassKindTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_pass_kind";
    }

    RHITestResult run(RHITestContext&) override
    {
        const std::unique_ptr<render::RenderGraphPass> triangle =
            render::createRenderGraphPass("TriangleRasterPass");
        const std::unique_ptr<render::RenderGraphPass> copy =
            render::createRenderGraphPass("CopyColorPass");
        const std::unique_ptr<render::RenderGraphPass> bufferWrite =
            render::createRenderGraphPass("RenderGraphBufferWritePass");
        const std::unique_ptr<render::RenderGraphPass> pathTrace =
            render::createRenderGraphPass("ScenePathTracePass");
        const std::unique_ptr<render::RenderGraphPass> rtxdi =
            render::createRenderGraphPass("SceneRTXDIPass");
        const std::unique_ptr<render::RenderGraphPass> rtxdiConfidence =
            render::createRenderGraphPass("RTXDIConfidencePass");
        const std::unique_ptr<render::RenderGraphPass> rtxdiComposite =
            render::createRenderGraphPass("RTXDICompositePass");
        const std::unique_ptr<render::RenderGraphPass> materialVisualization =
            render::createRenderGraphPass("SceneMaterialVisualizationPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenPreview =
            render::createRenderGraphPass("VisibilityBufferPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenStreamAsset =
            render::createRenderGraphPass("GPUDrivenStreamAssetPass");
        const std::unique_ptr<render::RenderGraphPass> nrdDenoise =
            render::createRenderGraphPass("NRDDenoisePass");
        const std::unique_ptr<render::RenderGraphPass> streamlineDlssSr =
            render::createRenderGraphPass("StreamlineDLSSSRPass");
        const std::unique_ptr<render::RenderGraphPass> streamlineDlssRr =
            render::createRenderGraphPass("StreamlineDLSSRRPass");

        if (triangle == nullptr ||
            copy == nullptr ||
            bufferWrite == nullptr ||
            pathTrace == nullptr ||
            rtxdi == nullptr ||
            rtxdiConfidence == nullptr ||
            rtxdiComposite == nullptr ||
            materialVisualization == nullptr ||
            gpuDrivenPreview == nullptr ||
            gpuDrivenStreamAsset == nullptr ||
            nrdDenoise == nullptr ||
            streamlineDlssSr == nullptr ||
            streamlineDlssRr == nullptr) {
            return RHITestResult::fail("failed to create built-in render graph passes");
        }
        if (triangle->kind() != render::RenderGraphPassKind::Raster ||
            triangle->queueType() != render::QueueType::Graphics) {
            return RHITestResult::fail("TriangleRasterPass is not classified as Raster/Graphics");
        }
        if (copy->kind() != render::RenderGraphPassKind::Unsafe ||
            copy->queueType() != render::QueueType::Copy) {
            return RHITestResult::fail("CopyColorPass is not classified as Unsafe/Copy");
        }
        if (bufferWrite->kind() != render::RenderGraphPassKind::Compute ||
            bufferWrite->queueType() != render::QueueType::Compute) {
            return RHITestResult::fail("RenderGraphBufferWritePass is not classified as Compute/Compute");
        }
        if (pathTrace->kind() != render::RenderGraphPassKind::Compute ||
            pathTrace->queueType() != render::QueueType::Compute) {
            return RHITestResult::fail("ScenePathTracePass is not classified as Compute/Compute");
        }
        if (rtxdi->kind() != render::RenderGraphPassKind::Compute ||
            rtxdi->queueType() != render::QueueType::Compute) {
            return RHITestResult::fail("SceneRTXDIPass is not classified as Compute/Compute");
        }
        if (rtxdiConfidence->kind() != render::RenderGraphPassKind::Compute ||
            rtxdiConfidence->queueType() != render::QueueType::Compute) {
            return RHITestResult::fail("RTXDIConfidencePass is not classified as Compute/Compute");
        }
        if (rtxdiComposite->kind() != render::RenderGraphPassKind::Compute ||
            rtxdiComposite->queueType() != render::QueueType::Compute) {
            return RHITestResult::fail("RTXDICompositePass is not classified as Compute/Compute");
        }
        if (materialVisualization->kind() != render::RenderGraphPassKind::Compute ||
            materialVisualization->queueType() != render::QueueType::Compute) {
            return RHITestResult::fail("SceneMaterialVisualizationPass is not classified as Compute/Compute");
        }
        if (gpuDrivenPreview->kind() != render::RenderGraphPassKind::Unsafe ||
            gpuDrivenPreview->queueType() != render::QueueType::Graphics) {
            return RHITestResult::fail("VisibilityBufferPass is not classified as Unsafe/Graphics");
        }
        if (gpuDrivenStreamAsset->kind() != render::RenderGraphPassKind::Unsafe ||
            gpuDrivenStreamAsset->queueType() != render::QueueType::Graphics) {
            return RHITestResult::fail("GPUDrivenStreamAssetPass is not classified as Unsafe/Graphics");
        }
        if (nrdDenoise->kind() != render::RenderGraphPassKind::Compute ||
            nrdDenoise->queueType() != render::QueueType::Compute) {
            return RHITestResult::fail("NRDDenoisePass is not classified as Compute/Compute");
        }
        if (streamlineDlssRr->kind() != render::RenderGraphPassKind::Unsafe ||
            streamlineDlssRr->queueType() != render::QueueType::Graphics) {
            return RHITestResult::fail("StreamlineDLSSRRPass is not classified as Unsafe/Graphics");
        }
        if (streamlineDlssSr->kind() != render::RenderGraphPassKind::Unsafe ||
            streamlineDlssSr->queueType() != render::QueueType::Graphics) {
            return RHITestResult::fail("StreamlineDLSSSRPass is not classified as Unsafe/Graphics");
        }

        bool foundTriangle = false;
        bool foundCopy = false;
        bool foundBufferWrite = false;
        bool foundPathTrace = false;
        bool foundRtxdi = false;
        bool foundRtxdiConfidence = false;
        bool foundRtxdiComposite = false;
        bool foundMaterialVisualization = false;
        bool foundGPUDrivenPreview = false;
        bool foundGPUDrivenStreamAsset = false;
        bool foundNrdDenoise = false;
        bool foundStreamlineDlssSr = false;
        bool foundStreamlineDlssRr = false;
        for (const render::RenderGraphPassInfo& passInfo : render::listRenderGraphPassTypes()) {
            if (passInfo.type == "TriangleRasterPass") {
                foundTriangle = passInfo.kind == render::RenderGraphPassKind::Raster &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "CopyColorPass") {
                foundCopy = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Copy;
            } else if (passInfo.type == "RenderGraphBufferWritePass") {
                foundBufferWrite = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "ScenePathTracePass") {
                foundPathTrace = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "SceneRTXDIPass") {
                foundRtxdi = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "RTXDIConfidencePass") {
                foundRtxdiConfidence = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "RTXDICompositePass") {
                foundRtxdiComposite = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "SceneMaterialVisualizationPass") {
                foundMaterialVisualization = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "VisibilityBufferPass") {
                foundGPUDrivenPreview = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "GPUDrivenStreamAssetPass") {
                foundGPUDrivenStreamAsset = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "NRDDenoisePass") {
                foundNrdDenoise = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "StreamlineDLSSRRPass") {
                foundStreamlineDlssRr = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "StreamlineDLSSSRPass") {
                foundStreamlineDlssSr = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Graphics;
            }
        }
        if (!foundTriangle ||
            !foundCopy ||
            !foundBufferWrite ||
            !foundPathTrace ||
            !foundRtxdi ||
            !foundRtxdiConfidence ||
            !foundRtxdiComposite ||
            !foundMaterialVisualization ||
            !foundGPUDrivenPreview ||
            !foundGPUDrivenStreamAsset ||
            !foundNrdDenoise ||
            !foundStreamlineDlssSr ||
            !foundStreamlineDlssRr) {
            return RHITestResult::fail("RenderGraphPassInfo did not preserve pass kind metadata");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphDLSSRRMotionVectorContractTest : public RHITest {
public:
    RenderGraphDLSSRRMotionVectorContractTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_dlss_rr_motion_vector_contract";
    }

    RHITestResult run(RHITestContext&) override
    {
        std::unique_ptr<render::RenderGraphPass> pathTrace =
            render::createRenderGraphPass("ScenePathTracePass");
        std::unique_ptr<render::RenderGraphPass> streamlineDlssSr =
            render::createRenderGraphPass("StreamlineDLSSSRPass");
        std::unique_ptr<render::RenderGraphPass> streamlineDlssRr =
            render::createRenderGraphPass("StreamlineDLSSRRPass");
        if (pathTrace == nullptr || streamlineDlssSr == nullptr || streamlineDlssRr == nullptr) {
            return RHITestResult::fail("failed to create Streamline DLSS motion-vector passes");
        }

        render::RenderGraphProperties pathTraceProperties = render::RenderGraphProperties::object();
        pathTraceProperties["exportDenoiserGuides"] = true;
        pathTrace->setProperties(std::move(pathTraceProperties));

        const render::RenderGraphCompileContext reflectContext{};
        const render::RenderPassReflection pathTraceReflection = pathTrace->reflect(reflectContext);
        const render::RenderPassReflection streamlineSrReflection = streamlineDlssSr->reflect(reflectContext);
        const render::RenderPassReflection streamlineReflection = streamlineDlssRr->reflect(reflectContext);
        for (const auto* reflection : {&streamlineSrReflection, &streamlineReflection}) {
            for (const auto& field : reflection->fields()) {
                if (field.visibility == render::RenderGraphFieldVisibility::Input &&
                    !render::hasFlag(field.usage, render::TextureUsageBits::Sampled)) {
                    return RHITestResult::fail("NGX inputs must allow sampling and shader-read layouts");
                }
            }
        }
        const render::RenderGraphField* pathTraceMotionVectors = pathTraceReflection.findField(
            "motionVectors",
            render::RenderGraphFieldVisibility::Output);
        const render::RenderGraphField* streamlineMotionVectors = streamlineReflection.findField(
            "motionVectors",
            render::RenderGraphFieldVisibility::Input);
        const render::RenderGraphField* streamlineSrMotionVectors = streamlineSrReflection.findField(
            "motionVectors",
            render::RenderGraphFieldVisibility::Input);
        const render::RenderGraphField* pathTraceDepth = pathTraceReflection.findField(
            "depth",
            render::RenderGraphFieldVisibility::Output);
        const render::RenderGraphField* streamlineSrDepth = streamlineSrReflection.findField(
            "depth",
            render::RenderGraphFieldVisibility::Input);
        if (pathTraceMotionVectors == nullptr ||
            pathTraceMotionVectors->format != render::Format::RG16Sfloat) {
            return RHITestResult::fail(
                "ScenePathTracePass motionVectors output must use RG16Sfloat");
        }
        if (streamlineMotionVectors == nullptr ||
            streamlineMotionVectors->format != render::Format::RG16Sfloat) {
            return RHITestResult::fail(
                "StreamlineDLSSRRPass motionVectors input must use RG16Sfloat");
        }
        if (streamlineSrMotionVectors == nullptr ||
            streamlineSrMotionVectors->format != render::Format::RG16Sfloat) {
            return RHITestResult::fail(
                "StreamlineDLSSSRPass motionVectors input must use RG16Sfloat");
        }
        if (pathTraceDepth == nullptr ||
            pathTraceDepth->format != render::Format::R32Sfloat ||
            streamlineSrDepth == nullptr ||
            streamlineSrDepth->format != render::Format::R32Sfloat ||
            !render::hasFlag(streamlineSrDepth->usage, render::TextureUsageBits::Sampled)) {
            return RHITestResult::fail(
                "ScenePathTracePass and StreamlineDLSSSRPass must share sampled R32Sfloat depth staging data");
        }

        return RHITestResult::pass();
    }
};


bool hasRuntimeSetting(
    const render::RenderGraphPass& pass,
    const std::string& key,
    render::RenderGraphRuntimeSettingType type,
    bool requireHistoryInvalidation = false,
    bool requireGraphRebuild = false)
{
    const std::vector<render::RenderGraphRuntimeSetting> settings = pass.runtimeSettings();
    for (const render::RenderGraphRuntimeSetting& setting : settings) {
        if (setting.key == key &&
            setting.type == type &&
            (!requireHistoryInvalidation || setting.invalidateHistory) &&
            (!requireGraphRebuild || setting.rebuildGraph)) {
            return true;
        }
    }
    return false;
}

bool hasBoolRuntimeSetting(
    const render::RenderGraphPass& pass,
    const std::string& key,
    bool requireHistoryInvalidation = false)
{
    return hasRuntimeSetting(
        pass,
        key,
        render::RenderGraphRuntimeSettingType::Bool,
        requireHistoryInvalidation);
}

class RenderGraphRuntimeSettingsDeclarationTest : public RHITest {
public:
    RenderGraphRuntimeSettingsDeclarationTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_runtime_settings_declarations";
    }

    RHITestResult run(RHITestContext&) override
    {
        const std::unique_ptr<render::RenderGraphPass> pathTrace =
            render::createRenderGraphPass("ScenePathTracePass");
        const std::unique_ptr<render::RenderGraphPass> rtxdi =
            render::createRenderGraphPass("SceneRTXDIPass");
        const std::unique_ptr<render::RenderGraphPass> nrdDenoise =
            render::createRenderGraphPass("NRDDenoisePass");
        const std::unique_ptr<render::RenderGraphPass> materialVisualization =
            render::createRenderGraphPass("SceneMaterialVisualizationPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenPreview =
            render::createRenderGraphPass("VisibilityBufferPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenStreamAsset =
            render::createRenderGraphPass("GPUDrivenStreamAssetPass");
        const std::unique_ptr<render::RenderGraphPass> streamlineDlssSr =
            render::createRenderGraphPass("StreamlineDLSSSRPass");
        const std::unique_ptr<render::RenderGraphPass> streamlineDlssRr =
            render::createRenderGraphPass("StreamlineDLSSRRPass");
        if (pathTrace == nullptr ||
            rtxdi == nullptr ||
            nrdDenoise == nullptr ||
            materialVisualization == nullptr ||
            gpuDrivenPreview == nullptr ||
            gpuDrivenStreamAsset == nullptr ||
            streamlineDlssSr == nullptr ||
            streamlineDlssRr == nullptr) {
            return RHITestResult::fail("failed to create passes for runtime settings declaration test");
        }
        if (!hasBoolRuntimeSetting(*pathTrace, "flipBitangent")) {
            return RHITestResult::fail("ScenePathTracePass missing Bool runtime setting flipBitangent");
        }
        if (!hasRuntimeSetting(
                *pathTrace,
                "debugView",
                render::RenderGraphRuntimeSettingType::Enum,
                true)) {
            return RHITestResult::fail("ScenePathTracePass missing history-invalidating debugView enum");
        }
        for (const char* key : {
                 "debugDisableNormalMap",
                 "debugForceGeometryNormal",
                 "debugDisableMaterialTextures",
                 "debugDisableDirectLighting",
                 "debugUseOpaqueShadows",
                 "debugDisableShadows",
                 "debugDisableVolumeAttenuation",
                 "debugDisableTransmission",
             }) {
            if (!hasBoolRuntimeSetting(*pathTrace, key, true)) {
                return RHITestResult::fail(
                    std::string("ScenePathTracePass missing history-invalidating debug Bool setting ") + key);
            }
        }
        if (!hasBoolRuntimeSetting(*rtxdi, "temporalReuse") ||
            !hasBoolRuntimeSetting(*rtxdi, "spatialReuse") ||
            !hasBoolRuntimeSetting(*rtxdi, "initialVisibility") ||
            !hasBoolRuntimeSetting(*rtxdi, "animateLights")) {
            return RHITestResult::fail("SceneRTXDIPass missing ReSTIR DI Bool runtime settings");
        }
        if (!hasBoolRuntimeSetting(*nrdDenoise, "relaxAntiFirefly")) {
            return RHITestResult::fail("NRDDenoisePass missing RELAX runtime settings");
        }
        if (!hasBoolRuntimeSetting(*nrdDenoise, "relaxConfidenceInputs")) {
            return RHITestResult::fail("NRDDenoisePass missing RELAX confidence setting");
        }
        if (!hasBoolRuntimeSetting(*materialVisualization, "flipBitangent")) {
            return RHITestResult::fail("SceneMaterialVisualizationPass missing Bool runtime setting flipBitangent");
        }
        if (!hasRuntimeSetting(*gpuDrivenPreview, "visualization",
                render::RenderGraphRuntimeSettingType::Enum, false, false)) {
            return RHITestResult::fail("VisibilityBufferPass must expose runtime-only visualization");
        }
        const auto visibilitySubsystems = gpuDrivenPreview->requiredSubsystems();
        if (std::find(visibilitySubsystems.begin(), visibilitySubsystems.end(),
                render::EnvironmentLightingSubsystem::kSubsystemId) != visibilitySubsystems.end()) {
            return RHITestResult::fail("VisibilityBufferPass must not require environment lighting");
        }
        if (!hasBoolRuntimeSetting(*gpuDrivenPreview, "instanceFrustumCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "instanceHzbCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "meshletFrustumCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "meshletNormalConeCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "freezeCullingCamera")) {
            return RHITestResult::fail("VisibilityBufferPass missing visibility culling runtime settings");
        }
        for (const auto* pass : {gpuDrivenPreview.get(), gpuDrivenStreamAsset.get()}) {
            if (!hasBoolRuntimeSetting(*pass, "autoLod") ||
                !hasRuntimeSetting(*pass, "lodPixelError", render::RenderGraphRuntimeSettingType::Float, false, false) ||
                !hasRuntimeSetting(*pass, "lodBias", render::RenderGraphRuntimeSettingType::Float, false, false) ||
                !hasRuntimeSetting(*pass, "lodLevel", render::RenderGraphRuntimeSettingType::Int, false, false)) {
                return RHITestResult::fail("Resident and stream passes must expose the shared runtime-only LOD controls");
            }
        }
        if (!hasRuntimeSetting(
                *streamlineDlssSr,
                "mode",
                render::RenderGraphRuntimeSettingType::Enum,
                true,
                true)) {
            return RHITestResult::fail(
                "StreamlineDLSSSRPass mode must invalidate history and rebuild the graph");
        }
        if (!hasRuntimeSetting(
                *streamlineDlssRr,
                "mode",
                render::RenderGraphRuntimeSettingType::Enum,
                true,
                true)) {
            return RHITestResult::fail(
                "StreamlineDLSSRRPass mode must invalidate history and rebuild the graph");
        }
        return RHITestResult::pass();
    }
};

class RenderGraphRuntimeRebuildDirtyTest : public RHITest {
public:
    RenderGraphRuntimeRebuildDirtyTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_runtime_rebuild_dirty";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderGraphProperties staticProperties = render::RenderGraphProperties::object();
        staticProperties["mode"] = "Balanced";

        render::RenderGraph graph;
        render::RenderGraphNode* node = graph.addNode(
            "StreamlineDLSSRRPass",
            "DLSSRR",
            std::move(staticProperties));
        if (node == nullptr) {
            return RHITestResult::fail("failed to create Streamline DLSS-RR runtime dirty test node");
        }

        graph.clearDirty();
        if (!graph.setNodeRuntimeProperty(node->id, "camera.fovDegrees", 55.0f)) {
            return RHITestResult::fail("failed to set ordinary DLSS-RR runtime property");
        }
        if (graph.dirty()) {
            return RHITestResult::fail("ordinary DLSS-RR runtime property unexpectedly dirtied graph");
        }

        if (!graph.setNodeRuntimeProperty(node->id, "mode", "Balanced")) {
            return RHITestResult::fail("failed to set same-effective DLSS-RR runtime mode");
        }
        if (graph.dirty()) {
            return RHITestResult::fail("same-effective DLSS-RR runtime mode unexpectedly dirtied graph");
        }

        if (!graph.setNodeRuntimeProperty(node->id, "mode", "Quality")) {
            return RHITestResult::fail("failed to set DLSS-RR runtime mode through single-property API");
        }
        if (!graph.dirty()) {
            return RHITestResult::fail("single-property DLSS-RR mode change did not dirty graph");
        }

        graph.clearDirty();
        if (!graph.setNodeRuntimeProperty(node->id, "mode", "Quality")) {
            return RHITestResult::fail("failed to repeat DLSS-RR runtime mode through single-property API");
        }
        if (graph.dirty()) {
            return RHITestResult::fail("repeating the same DLSS-RR runtime mode dirtied graph");
        }

        render::RenderGraphProperties runtimeProperties = node->runtimeProperties;
        runtimeProperties["mode"] = "Performance";
        if (!graph.setNodeRuntimeProperties(node->id, runtimeProperties)) {
            return RHITestResult::fail("failed to set DLSS-RR runtime mode through bulk API");
        }
        if (!graph.dirty()) {
            return RHITestResult::fail("bulk DLSS-RR mode change did not dirty graph");
        }

        graph.clearDirty();
        if (!graph.setNodeRuntimeProperties(node->id, runtimeProperties)) {
            return RHITestResult::fail("failed to repeat DLSS-RR runtime properties through bulk API");
        }
        if (graph.dirty()) {
            return RHITestResult::fail("repeating same-effective bulk runtime properties dirtied graph");
        }

        runtimeProperties.erase("mode");
        if (!graph.setNodeRuntimeProperties(node->id, std::move(runtimeProperties))) {
            return RHITestResult::fail("failed to remove DLSS-RR runtime mode overlay");
        }
        if (!graph.dirty()) {
            return RHITestResult::fail("removing DLSS-RR mode overlay did not dirty graph");
        }

        graph.clearDirty();
        render::RenderGraphProperties overlayWithoutMode = node->runtimeProperties;
        if (!graph.setNodeRuntimeProperties(node->id, std::move(overlayWithoutMode))) {
            return RHITestResult::fail("failed to repeat DLSS-RR runtime properties without mode overlay");
        }
        if (graph.dirty()) {
            return RHITestResult::fail("same-effective removed DLSS-RR mode overlay dirtied graph");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphSerializationTest : public RHITest {
public:
    RenderGraphSerializationTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_json_roundtrip";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraph graph = render::RenderGraph::createDefaultTriangleGraph();
        render::RenderGraphNode* node = graph.findNode("Triangle");
        if (node == nullptr) {
            return RHITestResult::fail("default graph did not create Triangle node");
        }
        graph.setNodePosition(node->id, 123.0f, 456.0f);
        graph.clearDirty();
        if (!graph.setNodeRuntimeProperty(node->id, "runtimeOnlySentinel", 42) ||
            !graph.setNodeRuntimeProperty(node->id, "camera.eye", {1.0f, 2.0f, 3.0f})) {
            return RHITestResult::fail("setNodeRuntimeProperty failed");
        }
        if (graph.dirty()) {
            return RHITestResult::fail("runtime property update unexpectedly marked graph dirty");
        }
        if (!node->runtimeProperties.is_object() ||
            !node->runtimeProperties.contains("camera") ||
            !node->runtimeProperties["camera"].contains("eye")) {
            return RHITestResult::fail("nested runtime property was not stored as an overlay object");
        }

        const std::string json = render::serializeRenderGraphToString(graph);
        if (json.find("runtimeOnlySentinel") != std::string::npos ||
            json.find("runtimeProperties") != std::string::npos) {
            return RHITestResult::fail("runtime properties leaked into serialized graph JSON");
        }
        render::RenderGraph loaded;
        std::string message;
        if (!render::deserializeRenderGraphFromString(json, loaded, message)) {
            return RHITestResult::fail(message);
        }

        if (loaded.nodes().size() != 1 || loaded.edges().size() != 0 || loaded.outputs().size() != 1) {
            return RHITestResult::fail("round-trip changed graph topology");
        }
        const render::RenderGraphNode* loadedNode = loaded.findNode("Triangle");
        if (loadedNode == nullptr ||
            loadedNode->type != "TriangleRasterPass" ||
            loadedNode->uiX != 123.0f ||
            loadedNode->uiY != 456.0f) {
            return RHITestResult::fail("round-trip changed node data");
        }
        if (loaded.firstOutputName() != "Triangle.color") {
            return RHITestResult::fail("round-trip changed marked output");
        }

        if (!loadedNode->runtimeProperties.empty()) {
            return RHITestResult::fail("round-trip restored runtime overlay from JSON");
        }
        if (!graph.setNodeProperties(node->id, node->properties) || !graph.dirty()) {
            return RHITestResult::fail("static property update did not mark graph dirty");
        }

        const std::string legacyJson = R"json({
            "version": 1,
            "name": "LegacyMissingEdgeIds",
            "nodes": [
                {
                    "id": 1,
                    "name": "PathTrace",
                    "type": "ScenePathTracePass",
                    "properties": {
                        "exportDenoiserGuides": true
                    }
                },
                {
                    "id": 2,
                    "name": "DLSSRR",
                    "type": "StreamlineDLSSRRPass",
                    "properties": {}
                }
            ],
            "edges": [
                {"src": "PathTrace.color", "dst": "DLSSRR.inputColor"},
                {"src": "PathTrace.albedo", "dst": "DLSSRR.albedo"},
                {"src": "PathTrace.specularAlbedo", "dst": "DLSSRR.specularAlbedo"},
                {"src": "PathTrace.normalRoughness", "dst": "DLSSRR.normalRoughness"},
                {"src": "PathTrace.motionVectors", "dst": "DLSSRR.motionVectors"},
                {"src": "PathTrace.linearDepth", "dst": "DLSSRR.linearDepth"},
                {"src": "PathTrace.specularHitDistance", "dst": "DLSSRR.specularHitDistance"}
            ],
            "outputs": [
                "DLSSRR.color"
            ]
        })json";
        render::RenderGraph legacyLoaded;
        if (!render::deserializeRenderGraphFromString(legacyJson, legacyLoaded, message)) {
            return RHITestResult::fail(message);
        }
        std::unordered_set<uint32_t> legacyEdgeIds;
        for (const render::RenderGraphEdge& edge : legacyLoaded.edges()) {
            if (edge.id == 0u || !legacyEdgeIds.insert(edge.id).second) {
                return RHITestResult::fail("legacy graph edges did not receive unique ids");
            }
        }
        if (legacyEdgeIds.size() != 7u) {
            return RHITestResult::fail("legacy graph changed edge count");
        }
        return RHITestResult::pass();
    }
};

class VisibilityBufferPassLegacyGraphTest : public RHITest {
public:
    VisibilityBufferPassLegacyGraphTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_visibility_buffer_pass_legacy_json";
    }

    RHITestResult run(RHITestContext&) override
    {
        const std::string legacyJson = R"json({
            "version": 1,
            "name": "SavedVisibilityGraph",
            "nodes": [{
                "id": 7,
                "name": "GPUDriven",
                "type": "GPUDrivenPreviewPass",
                "position": {"x": 12.0, "y": 34.0},
                "properties": {"mode": "baseColor", "freezeCullingCamera": true}
            }],
            "edges": [],
            "outputs": ["GPUDriven.color", "GPUDriven.visibility", "GPUDriven.depth"]
        })json";
        render::RenderGraph graph;
        std::string message;
        if (!render::deserializeRenderGraphFromString(legacyJson, graph, message)) {
            return RHITestResult::fail(message);
        }
        const render::RenderGraphNode* node = graph.findNode("GPUDriven");
        if (node == nullptr || node->id != 7u ||
            node->type != "VisibilityBufferPass" ||
            node->uiX != 12.0f || node->uiY != 34.0f ||
            node->properties.value("mode", "") != "baseColor" ||
            !node->properties.value("freezeCullingCamera", false) ||
            graph.outputs().size() != 3u ||
            graph.firstOutputName() != "GPUDriven.color") {
            return RHITestResult::fail("legacy visibility graph migration changed node data or output connections");
        }
        const std::string serialized = render::serializeRenderGraphToString(graph);
        if (serialized.find("GPUDrivenPreviewPass") != std::string::npos ||
            serialized.find("VisibilityBufferPass") == std::string::npos) {
            return RHITestResult::fail("migrated visibility graph did not serialize the canonical pass name");
        }
        render::RenderGraph reloaded;
        if (!render::deserializeRenderGraphFromString(serialized, reloaded, message) ||
            reloaded.findNode("GPUDriven") == nullptr ||
            reloaded.findNode("GPUDriven")->type != "VisibilityBufferPass") {
            return RHITestResult::fail("canonical visibility graph did not round-trip: " + message);
        }
        bool foundCanonical = false;
        for (const render::RenderGraphPassInfo& info : render::listRenderGraphPassTypes()) {
            if (info.type == "GPUDrivenPreviewPass") {
                return RHITestResult::fail("legacy visibility pass is still exposed in the editor registry");
            }
            foundCanonical = foundCanonical || info.type == "VisibilityBufferPass";
        }
        if (!foundCanonical || render::createRenderGraphPass("VisibilityBufferPass") == nullptr) {
            return RHITestResult::fail("VisibilityBufferPass is not registered");
        }
        return RHITestResult::pass();
    }
};

class RenderGraphLegacyAcronymPassNamesTest : public RHITest {
public:
    RenderGraphLegacyAcronymPassNamesTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_legacy_acronym_pass_names_json";
    }

    RHITestResult run(RHITestContext&) override
    {
        constexpr std::array aliases{
            std::pair{"SceneRtxdiPass", "SceneRTXDIPass"},
            std::pair{"RtxdiConfidencePass", "RTXDIConfidencePass"},
            std::pair{"RtxdiCompositePass", "RTXDICompositePass"},
            std::pair{"RtxcrMaterialSamplePass", "RTXCRMaterialSamplePass"},
            std::pair{"NrdDenoisePass", "NRDDenoisePass"},
            std::pair{"StreamlineDlssSrPass", "StreamlineDLSSSRPass"},
            std::pair{"StreamlineDlssRrPass", "StreamlineDLSSRRPass"},
            std::pair{"DlssNrPass", "DLSSNRPass"},
        };
        constexpr std::array sampleGraphs{
            "rtxdi_meet_mat.metallic_graph.json",
            "pathtracing_abeautiful_game_openpbr_dlss_sr.metallic_graph.json",
            "pathtracing_abeautiful_game_openpbr_dlss_rr.metallic_graph.json",
            "pathtracing_abeautiful_game_openpbr_dlss_nr.metallic_graph.json",
        };
        std::array<bool, aliases.size()> covered{};
        for (size_t sampleIndex = 0; sampleIndex < sampleGraphs.size(); ++sampleIndex) {
            render::RenderGraph canonical;
            std::string message;
            const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) /
                "Pipelines/Samples" / sampleGraphs[sampleIndex];
            if (!render::loadRenderGraphFromFile(path, canonical, message)) {
                return RHITestResult::fail("canonical sample graph did not load: " + message);
            }
            if (sampleIndex == 0) {
                // A node name may contain the old type verbatim. Only its type
                // should migrate; marked outputs still refer to its saved name.
                if (canonical.addNode("RTXCRMaterialSamplePass", "RtxcrMaterialSamplePass",
                        render::RenderGraphProperties{{"view", "chiang"}}, 12.0f, 34.0f) == nullptr ||
                    !canonical.markOutput("RtxcrMaterialSamplePass.color")) {
                    return RHITestResult::fail("failed to add the RTXCR legacy-name fixture");
                }
            }
            nlohmann::json expected = nlohmann::json::parse(render::serializeRenderGraphToString(canonical));
            expected["view"]["savedTypeSentinel"] = "SceneRtxdiPass";
            nlohmann::json legacy = expected;
            for (size_t nodeIndex = 0; nodeIndex < expected["nodes"].size(); ++nodeIndex) {
                auto& node = expected["nodes"][nodeIndex];
                for (size_t aliasIndex = 0; aliasIndex < aliases.size(); ++aliasIndex) {
                    const auto& [oldType, currentType] = aliases[aliasIndex];
                    if (node["type"] != currentType) {
                        continue;
                    }
                    covered[aliasIndex] = true;
                    node["properties"]["savedTypeSentinel"] = {
                        {"type", oldType}, {"values", {1, 2, 3}},
                    };
                    legacy["nodes"][nodeIndex] = node;
                    legacy["nodes"][nodeIndex]["type"] = oldType;
                    break;
                }
            }
            render::RenderGraph migrated;
            if (!render::deserializeRenderGraphFromString(legacy.dump(), migrated, message)) {
                return RHITestResult::fail("legacy acronym pass graph did not load: " + message);
            }
            const std::string serialized = render::serializeRenderGraphToString(migrated);
            if (nlohmann::json::parse(serialized) != expected) {
                return RHITestResult::fail(
                    "legacy pass migration changed graph data beyond the saved node types");
            }
            render::RenderGraph reloaded;
            if (!render::deserializeRenderGraphFromString(serialized, reloaded, message) ||
                nlohmann::json::parse(render::serializeRenderGraphToString(reloaded)) != expected) {
                return RHITestResult::fail("canonical acronym pass graph did not round-trip: " + message);
            }
        }
        if (std::find(covered.begin(), covered.end(), false) != covered.end()) {
            return RHITestResult::fail("legacy graph fixtures did not cover every renamed pass type");
        }
        const auto registeredTypes = render::listRenderGraphPassTypes();
        for (const auto& [oldType, currentType] : aliases) {
            if (std::any_of(registeredTypes.begin(), registeredTypes.end(), [&](const auto& info) {
                    return info.type == oldType;
                }) || render::createRenderGraphPass(oldType) != nullptr) {
                return RHITestResult::fail("legacy acronym pass type is exposed in the editor registry");
            }
            if (render::createRenderGraphPass(currentType) == nullptr) {
                return RHITestResult::fail("canonical acronym pass type is not registered");
            }
        }
        return RHITestResult::pass();
    }
};

class TestPathTraceSample final : public render::RenderSample {
public:
    TestPathTraceSample(std::string id, std::string scenePath, std::string previewOutput) :
        id_(std::move(id)),
        scenePath_(std::move(scenePath)),
        previewOutput_(std::move(previewOutput))
    {
    }

    std::string_view id() const override { return id_; }
    std::string_view name() const override { return "Test Path Trace Sample"; }
    std::string_view category() const override { return "PathTracing"; }
    std::string scenePath() const override { return scenePath_; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/pathtracing_meet_mat.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::string previewOutput() const override { return previewOutput_; }

private:
    std::string id_;
    std::string scenePath_;
    std::string previewOutput_;
};

class GPUDrivenSceneCatalogTest : public RHITest {
public:
    GPUDrivenSceneCatalogTest()
    {
        type = RHITestType::Validation;
        name = "gpu_driven_scene_catalog";
    }

    RHITestResult run(RHITestContext&) override
    {
        const auto scenes = render::listGPUDrivenSceneSamples();
        if (scenes.size() != 2 || scenes[0].name != "MiniZorah" || scenes[1].name != "ZorahFull") {
            return RHITestResult::fail("GPUDriven scene selector must expose MiniZorah and ZorahFull only");
        }
        for (const auto& scene : scenes) {
            for (const auto& path : {std::filesystem::path(scene.scenePath),
                     std::filesystem::path(PROJECT_SOURCE_DIR) / scene.scenePath}) {
                const char* id = render::gpuDrivenSceneSampleIdForPath(path);
                if (!id || scene.id != id) {
                    return RHITestResult::fail("Relative/absolute File Open path selected the wrong scene preset");
                }
            }
#if defined(_WIN32)
            auto caseVariant = std::filesystem::path(PROJECT_SOURCE_DIR) / "aSSET" /
                std::filesystem::path(scene.scenePath).lexically_relative("Asset");
            const char* caseId = render::gpuDrivenSceneSampleIdForPath(caseVariant);
            if (!caseId || scene.id != caseId) {
                return RHITestResult::fail("Windows scene path matching must ignore filename case");
            }
#endif
            render::RenderSampleLoadResult loaded;
            std::string message;
            if (!render::loadBuiltInRenderSample(scene.id, loaded, message)) {
                return RHITestResult::fail(message);
            }
            const auto* vbuffer = loaded.graph.findNode("VBuffer");
            if (!vbuffer || loaded.desc.loadSceneInEditor ||
                vbuffer->properties.value("path", "") != scene.scenePath ||
                !vbuffer->properties.value("streamAssetOnly", false) ||
                vbuffer->properties.value("autoBuildStreamAsset", true) ||
                !loaded.graph.viewProperties().contains("camera")) {
                return RHITestResult::fail("Scene preset must select metadata streaming, its own source and camera");
            }
            const bool full = scene.id == render::kGPUDrivenZorahFullSampleId;
            if (vbuffer->properties.value("maxResidentBytes", uint64_t(0)) !=
                    (full ? 3758096384ull : 536870912ull) ||
                vbuffer->properties.value("maxClasBytes", uint64_t(0)) !=
                    (full ? 2147483648ull : 268435456ull) ||
                vbuffer->properties.value("compactShadingAttributes", false) != full ||
                vbuffer->properties.value("streamAssetPath", "") !=
                    (full ? "Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin" :
                            "Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin")) {
                return RHITestResult::fail("Scene switch must load the matching cook, attribute layout and budgets");
            }
        }
        if (render::gpuDrivenSceneSampleIdForPath({}) ||
            render::gpuDrivenSceneSampleIdForPath("Asset/Other/zorah_main_public.v2.gltf") ||
            render::gpuDrivenSceneSampleIdForPath("Asset/meet_mat.glb") ||
            render::isGPUDrivenSceneSample("gpu-driven-visibility-buffer")) {
            return RHITestResult::fail("Unsupported scenes and matching basenames must not enter GPUDrivenSample");
        }
        const auto allSamples = render::listBuiltInRenderSamples();
        if (std::none_of(allSamples.begin(), allSamples.end(), [](const auto& sample) {
                return sample.id == render::kGPUDrivenVisibilitySampleId;
            })) {
            return RHITestResult::fail("The generic editor must retain diagnostic samples");
        }
        return RHITestResult::pass();
    }
};

class RenderSampleLoadTest : public RHITest {
public:
    RenderSampleLoadTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_sample_load";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-meet-mat", sample, message)) {
            return RHITestResult::fail(message);
        }

        if (sample.desc.id != "pathtracing-meet-mat" ||
            sample.desc.name != "Path Tracing / meet_mat" ||
            sample.desc.category != "PathTracing" ||
            sample.desc.scenePath != "Asset/meet_mat.glb" ||
            sample.desc.graphPath != "Pipelines/Samples/pathtracing_meet_mat.metallic_graph.json" ||
            sample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("built-in Sample metadata did not load as expected");
        }

        const render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr ||
            !pathTrace->properties.is_object() ||
            pathTrace->properties.value("path", "") != sample.desc.scenePath) {
            return RHITestResult::fail("Sample did not apply scene path to target node");
        }

        std::string validationLog;
        if (!sample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (sample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("Sample graph first output changed");
        }

        render::RenderSampleLoadResult pathTracingSample;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", pathTracingSample, message)) {
            return RHITestResult::fail(message);
        }
        if (pathTracingSample.desc.id != "pathtracing-sample" ||
            pathTracingSample.desc.name != "PathTracingSample" ||
            pathTracingSample.desc.category != "PathTracing" ||
            pathTracingSample.desc.scenePath != "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf" ||
            pathTracingSample.desc.graphPath != "Pipelines/Samples/pathtracing_abeautiful_game_openpbr.metallic_graph.json" ||
            !pathTracingSample.desc.environment.has_value() ||
            pathTracingSample.desc.environment->path != "Asset/ABeautifulGame/environment.hdr" ||
            pathTracingSample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("OpenPBR PathTracingSample metadata did not load as expected");
        }
        const render::RenderGraphNode* openPBRPathTrace = pathTracingSample.graph.findNode("PathTrace");
        if (openPBRPathTrace == nullptr ||
            !openPBRPathTrace->properties.is_object() ||
            openPBRPathTrace->properties.value("path", "") != pathTracingSample.desc.scenePath ||
            openPBRPathTrace->properties.value("bsdf", "") != "openpbr") {
            return RHITestResult::fail("OpenPBR PathTracingSample did not apply pass defaults");
        }
        if (!pathTracingSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (pathTracingSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("OpenPBR PathTracingSample graph first output changed");
        }

        render::RenderSampleLoadResult rtxdiSample;
        if (!render::loadBuiltInRenderSample("rtxdi-sample", rtxdiSample, message)) {
            return RHITestResult::fail(message);
        }
        if (rtxdiSample.desc.id != "rtxdi-sample" ||
            rtxdiSample.desc.name != "RTXDI / ReSTIR DI" ||
            rtxdiSample.desc.category != "RTXDI" ||
            rtxdiSample.desc.scenePath != "Asset/meet_mat.glb" ||
            rtxdiSample.desc.graphPath != "Pipelines/Samples/rtxdi_meet_mat.metallic_graph.json" ||
            !rtxdiSample.desc.environment.has_value() ||
            rtxdiSample.desc.environment->path != "Asset/ABeautifulGame/environment.hdr" ||
            rtxdiSample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("RTXDI Sample metadata did not load as expected");
        }
        const render::RenderGraphNode* rtxdi = rtxdiSample.graph.findNode("RTXDI");
        const render::RenderGraphNode* confidence = rtxdiSample.graph.findNode("Confidence");
        const render::RenderGraphNode* relax = rtxdiSample.graph.findNode("Relax");
        const render::RenderGraphNode* composite = rtxdiSample.graph.findNode("Composite");
        if (rtxdi == nullptr ||
            confidence == nullptr ||
            relax == nullptr ||
            composite == nullptr ||
            rtxdi->type != "SceneRTXDIPass" ||
            confidence->type != "RTXDIConfidencePass" ||
            relax->type != "NRDDenoisePass" ||
            composite->type != "RTXDICompositePass" ||
            !rtxdi->properties.is_object() ||
            rtxdi->properties.value("path", "") != rtxdiSample.desc.scenePath ||
            rtxdi->properties.value("lightCount", 0) != 256 ||
            rtxdi->properties.value("initialSamples", 0) != 8 ||
            rtxdi->properties.value("environmentSamples", 0) != 4 ||
            rtxdi->properties.value("spatialSamples", 0) != 1 ||
            !rtxdi->properties.value("localLightImportanceSampling", false) ||
            !rtxdi->properties.value("environmentImportanceSampling", false) ||
            !rtxdi->properties.value("temporalReuse", false) ||
            !rtxdi->properties.value("spatialReuse", false) ||
            !rtxdi->properties.value("initialVisibility", false) ||
            !confidence->properties.is_object() ||
            confidence->properties.value("gradientFilterPasses", 0) != 4 ||
            confidence->properties.value("gradientSensitivity", 0.0f) != 8.0f ||
            !relax->properties.is_object() ||
            relax->properties.value("denoiser", "") != "RELAX" ||
            !relax->properties.value("relaxConfidenceInputs", false) ||
            !relax->properties.value("relaxAntiFirefly", false)) {
            return RHITestResult::fail("RTXDI Sample did not apply ReSTIR DI and RELAX defaults");
        }
        if (!rtxdiSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (rtxdiSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("RTXDI Sample graph first output changed");
        }

        render::RenderSampleLoadResult rtxcrSample;
        if (!render::loadBuiltInRenderSample("rtxcr-material-sample", rtxcrSample, message)) {
            return RHITestResult::fail(message);
        }
        if (rtxcrSample.desc.id != "rtxcr-material-sample" ||
            rtxcrSample.desc.name != "RTXCR Claire Ponytail" ||
            rtxcrSample.desc.category != "RTXCR" ||
            !rtxcrSample.desc.loadSceneInEditor ||
            rtxcrSample.desc.graphPath !=
                "Pipelines/Samples/rtxcr_material_showcase.metallic_graph.json" ||
            rtxcrSample.desc.scenePath.find("ponyTail_15vtx.gltf") == std::string::npos ||
            rtxcrSample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("RTXCR Sample metadata did not load as expected");
        }
        const render::RenderGraphNode* rtxcr = rtxcrSample.graph.findNode("PathTrace");
        if (rtxcr == nullptr ||
            rtxcr->type != "ScenePathTracePass" ||
            !rtxcr->properties.is_object() ||
            rtxcr->properties.value("samples", 0) != 4 ||
            rtxcr->properties.value("maxDepth", 0) != 4 ||
            rtxcr->properties.value("path", "").find("ponyTail_15vtx.gltf") ==
                std::string::npos) {
            return RHITestResult::fail("RTXCR Sample did not preserve Claire groom defaults");
        }
        if (!rtxcrSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (rtxcrSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("RTXCR Sample graph first output changed");
        }

        render::RenderSampleLoadResult dlssSrSample;
        if (!render::loadBuiltInRenderSample("pathtracing-sample-dlss-sr", dlssSrSample, message)) {
            return RHITestResult::fail(message);
        }
        if (dlssSrSample.desc.id != "pathtracing-sample-dlss-sr" ||
            dlssSrSample.desc.name != "PathTracingSample / DLSS-SR" ||
            dlssSrSample.desc.category != "PathTracing" ||
            dlssSrSample.desc.scenePath != "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf" ||
            dlssSrSample.desc.graphPath != "Pipelines/Samples/pathtracing_abeautiful_game_openpbr_dlss_sr.metallic_graph.json" ||
            dlssSrSample.desc.previewOutput != "FinalBlit.color" ||
            !dlssSrSample.desc.requiresStreamline) {
            return RHITestResult::fail("DLSS-SR PathTracingSample metadata did not load as expected");
        }
        const render::RenderGraphNode* dlssSrPathTrace = dlssSrSample.graph.findNode("PathTrace");
        const render::RenderGraphNode* dlssSrPass = dlssSrSample.graph.findNode("DLSSSR");
        if (dlssSrPathTrace == nullptr ||
            dlssSrPass == nullptr ||
            !dlssSrPathTrace->properties.is_object() ||
            dlssSrPathTrace->properties.value("path", "") != dlssSrSample.desc.scenePath ||
            !dlssSrPathTrace->properties.value("exportDenoiserGuides", false) ||
            dlssSrPass->type != "StreamlineDLSSSRPass") {
            return RHITestResult::fail("DLSS-SR PathTracingSample did not apply expected graph defaults");
        }
        if (!dlssSrSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (dlssSrSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("DLSS-SR PathTracingSample graph first output changed");
        }

        render::RenderSampleLoadResult dlssRrSample;
        if (!render::loadBuiltInRenderSample("pathtracing-sample-dlss-rr", dlssRrSample, message)) {
            return RHITestResult::fail(message);
        }
        if (dlssRrSample.desc.id != "pathtracing-sample-dlss-rr" ||
            dlssRrSample.desc.name != "PathTracingSample / DLSS-RR" ||
            dlssRrSample.desc.category != "PathTracing" ||
            dlssRrSample.desc.scenePath != "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf" ||
            dlssRrSample.desc.graphPath != "Pipelines/Samples/pathtracing_abeautiful_game_openpbr_dlss_rr.metallic_graph.json" ||
            dlssRrSample.desc.previewOutput != "FinalBlit.color" ||
            !dlssRrSample.desc.requiresStreamline) {
            return RHITestResult::fail("DLSS-RR PathTracingSample metadata did not load as expected");
        }
        const render::RenderGraphNode* dlssRrPathTrace = dlssRrSample.graph.findNode("PathTrace");
        const render::RenderGraphNode* dlssRrPass = dlssRrSample.graph.findNode("DLSSRR");
        if (dlssRrPathTrace == nullptr ||
            dlssRrPass == nullptr ||
            !dlssRrPathTrace->properties.is_object() ||
            dlssRrPathTrace->properties.value("path", "") != dlssRrSample.desc.scenePath ||
            !dlssRrPathTrace->properties.value("exportDenoiserGuides", false) ||
            dlssRrPass->type != "StreamlineDLSSRRPass") {
            return RHITestResult::fail("DLSS-RR PathTracingSample did not apply expected graph defaults");
        }
        if (!dlssRrSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (dlssRrSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("DLSS-RR PathTracingSample graph first output changed");
        }

        render::RenderSampleLoadResult materialSample;
        if (!render::loadBuiltInRenderSample("material-visualization-abeautiful-game", materialSample, message)) {
            return RHITestResult::fail(message);
        }
        if (materialSample.desc.id != "material-visualization-abeautiful-game" ||
            materialSample.desc.name != "Material Visualization / ABeautifulGame" ||
            materialSample.desc.category != "Material" ||
            materialSample.desc.scenePath != "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf" ||
            materialSample.desc.graphPath != "Pipelines/Samples/material_visualization_abeautiful_game.metallic_graph.json" ||
            materialSample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("material visualization Sample metadata did not load as expected");
        }
        const render::RenderGraphNode* materialViz = materialSample.graph.findNode("MaterialViz");
        if (materialViz == nullptr ||
            !materialViz->properties.is_object() ||
            materialViz->properties.value("path", "") != materialSample.desc.scenePath ||
            materialViz->properties.value("mode", "") != "material") {
            return RHITestResult::fail("material visualization Sample did not apply scene path and defaults");
        }
        if (!materialSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (materialSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("material visualization Sample graph first output changed");
        }

        render::RenderSampleLoadResult gpuDrivenSample;
        if (!render::loadBuiltInRenderSample(render::kDefaultGPUDrivenSampleId, gpuDrivenSample, message)) {
            return RHITestResult::fail(message);
        }
        if (gpuDrivenSample.desc.id != "gpu-driven-sample" ||
            gpuDrivenSample.desc.name != "GPUDrivenSample" ||
            gpuDrivenSample.desc.category != "GPUDriven" ||
            gpuDrivenSample.desc.scenePath != "Asset/MiniZorah/zorah_main_public.v2.gltf" ||
            gpuDrivenSample.desc.loadSceneInEditor ||
            gpuDrivenSample.desc.graphPath != "Pipelines/Samples/gpu_driven_realtime.metallic_graph.json" ||
            !gpuDrivenSample.desc.environment.has_value() ||
            gpuDrivenSample.desc.previewOutput != "FinalBlit.color" ||
            !gpuDrivenSample.desc.requiresStreamline) {
            return RHITestResult::fail("GPUDrivenSample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDriven = gpuDrivenSample.graph.findNode("VBuffer");
        const render::RenderGraphNode* gpuDrivenDeferred = gpuDrivenSample.graph.findNode("Deferred");
        if (gpuDrivenSample.graph.nodes().size() != 7u ||
            gpuDriven == nullptr ||
            gpuDriven->type != "VisibilityBufferPass" ||
            !gpuDriven->properties.is_object() ||
            gpuDriven->properties.value("path", "") != gpuDrivenSample.desc.scenePath ||
            gpuDriven->properties.value("visualization", "") != "none" ||
            !gpuDriven->properties.value("streamAssetOnly", false) ||
            !gpuDriven->properties.value("enableMeshletStreaming", false) ||
            gpuDriven->properties.value("autoBuildStreamAsset", true) ||
            gpuDrivenDeferred == nullptr ||
            gpuDrivenDeferred->type != "VisibilityBufferDeferredPass" ||
            gpuDrivenDeferred->properties.value("path", "") != gpuDrivenSample.desc.scenePath ||
            gpuDrivenDeferred->properties.value("lightingMode", "") != "realtime" ||
            gpuDrivenSample.graph.findNode("Shadows") == nullptr ||
            gpuDrivenSample.graph.findNode("Shadows")->type != "RayTracedShadowPass" ||
            !gpuDrivenSample.graph.findNode("Shadows")->properties.value("sigmaDenoise", false) ||
            !gpuDrivenSample.graph.viewProperties().contains("camera")) {
            return RHITestResult::fail("GPUDrivenSample did not apply pass defaults");
        }
        for (const char* type : {"RayTracedShadowPass", "ScreenSpaceShadowPass"}) {
            auto shadow = render::createRenderGraphPass(type);
            if (shadow == nullptr) { return RHITestResult::fail("Ray-traced shadow node or legacy alias unavailable"); }
            const auto controls = shadow->runtimeSettings();
            bool enabledControl = false;
            for (const auto& control : controls) {
                enabledControl |= control.key == "rayTracedShadows";
                if (control.key == "shadowSteps" || control.key == "shadowThickness" ||
                    control.key == "shadowDistance" || control.key == "preserveGeometryShadows") {
                    return RHITestResult::fail("Obsolete screen-space shadow control still exposed");
                }
            }
            if (!enabledControl) { return RHITestResult::fail("Ray-traced shadow enable control missing"); }
        }
        if (!gpuDrivenSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (gpuDrivenSample.graph.firstOutputName() != "FinalBlit.color" ||
            !gpuDrivenSample.graph.outputs().empty()) {
            return RHITestResult::fail("GPUDrivenSample graph first output changed");
        }
        if (!render::setRenderSampleScenePath(gpuDrivenSample, "Asset/meet_mat.glb", message) ||
            gpuDrivenSample.graph.findNode("VBuffer")->properties.value("path", "") != "Asset/meet_mat.glb" ||
            gpuDrivenSample.graph.findNode("Deferred")->properties.value("path", "") != "Asset/meet_mat.glb") {
            return RHITestResult::fail("GPUDrivenSample scene override did not reach visibility and lighting");
        }
        render::RenderSampleLoadResult gpuDrivenVisibilitySample;
        if (!render::loadBuiltInRenderSample(render::kGPUDrivenVisibilitySampleId, gpuDrivenVisibilitySample, message) ||
            gpuDrivenVisibilitySample.desc.scenePath != "Asset/Sponza/glTF/Sponza.gltf" ||
            gpuDrivenVisibilitySample.desc.requiresStreamline ||
            gpuDrivenVisibilitySample.graph.nodes().size() != 2u ||
            gpuDrivenVisibilitySample.graph.findNode("GPUDriven") == nullptr ||
            gpuDrivenVisibilitySample.graph.findNode("GPUDriven")->type != "VisibilityBufferPass" ||
            !gpuDrivenVisibilitySample.graph.validate(validationLog)) {
            return RHITestResult::fail("GPUDrivenSample visibility diagnostics did not retain their standalone graph");
        }
        bool requiresStreamline = true;
        if (!render::queryBuiltInRenderSampleStreamlineRequirement(
                "gpu-driven-sample",
                requiresStreamline) ||
            !requiresStreamline ||
            !render::queryBuiltInRenderSampleStreamlineRequirement(
                render::kGPUDrivenVisibilitySampleId,
                requiresStreamline) ||
            requiresStreamline ||
            !render::queryBuiltInRenderSampleStreamlineRequirement(
                "pathtracing-sample-dlss-sr",
                requiresStreamline) ||
            !requiresStreamline ||
            !render::queryBuiltInRenderSampleStreamlineRequirement(
                "pathtracing-sample-dlss-rr",
                requiresStreamline) ||
            !requiresStreamline ||
            render::queryBuiltInRenderSampleStreamlineRequirement(
                "unknown-sample",
                requiresStreamline)) {
            return RHITestResult::fail("built-in Sample Streamline requirements are inconsistent");
        }

        render::RenderSampleLoadResult gpuDrivenStreamAssetSample;
        if (!render::loadBuiltInRenderSample("gpu-driven-streamasset", gpuDrivenStreamAssetSample, message)) {
            return RHITestResult::fail(message);
        }
        if (gpuDrivenStreamAssetSample.desc.id != "gpu-driven-streamasset" ||
            gpuDrivenStreamAssetSample.desc.name != "GPUDrivenSample / StreamAsset" ||
            gpuDrivenStreamAssetSample.desc.category != "GPUDriven" ||
            gpuDrivenStreamAssetSample.desc.scenePath != "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf" ||
            gpuDrivenStreamAssetSample.desc.loadSceneInEditor ||
            gpuDrivenStreamAssetSample.desc.graphPath !=
                "Pipelines/Samples/gpu_driven_sponza_streamasset.metallic_graph.json" ||
            gpuDrivenStreamAssetSample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("GPUDriven StreamAsset sample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDrivenStreamAsset =
            gpuDrivenStreamAssetSample.graph.findNode("GPUDriven");
        if (gpuDrivenStreamAsset == nullptr ||
            gpuDrivenStreamAsset->type != "GPUDrivenStreamAssetPass" ||
            !gpuDrivenStreamAsset->properties.is_object() ||
            gpuDrivenStreamAsset->properties.value("path", "") != gpuDrivenStreamAssetSample.desc.scenePath ||
            gpuDrivenStreamAsset->properties.value("autoBuildStreamAsset", true) ||
            !gpuDrivenStreamAsset->properties.value("enableGpuLodSelection", false) ||
            gpuDrivenStreamAsset->properties.value("debugColorMode", "") != "page" ||
            gpuDrivenStreamAsset->properties.value("selectedLodLevel", -1) != 0) {
            return RHITestResult::fail("GPUDriven StreamAsset sample did not preserve streamasset defaults");
        }
        if (!gpuDrivenStreamAssetSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (gpuDrivenStreamAssetSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("GPUDriven StreamAsset graph first output changed");
        }
        if (!render::setRenderSampleScenePath(
                gpuDrivenStreamAssetSample,
                "Asset/Zorah/zorah_main_public.v2.gltf",
                message) ||
            gpuDrivenStreamAssetSample.desc.scenePath != "Asset/Zorah/zorah_main_public.v2.gltf" ||
            gpuDrivenStreamAsset->properties.value("path", "") !=
                "Asset/Zorah/zorah_main_public.v2.gltf" ||
            gpuDrivenStreamAssetSample.graph.dirty()) {
            return RHITestResult::fail("GPUDriven StreamAsset scene override failed");
        }

        render::RenderSampleLoadResult gpuDrivenTerrainP0Sample;
        if (!render::loadBuiltInRenderSample("gpu-driven-terrain-p0", gpuDrivenTerrainP0Sample, message)) {
            return RHITestResult::fail(message);
        }
        if (gpuDrivenTerrainP0Sample.desc.id != "gpu-driven-terrain-p0" ||
            gpuDrivenTerrainP0Sample.desc.name != "GPUDrivenSample / Terrain P0" ||
            gpuDrivenTerrainP0Sample.desc.category != "GPUDriven" ||
            gpuDrivenTerrainP0Sample.desc.scenePath !=
                "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf" ||
            gpuDrivenTerrainP0Sample.desc.loadSceneInEditor ||
            gpuDrivenTerrainP0Sample.desc.graphPath !=
                "Pipelines/Samples/gpu_driven_terrain_p0_streamasset.metallic_graph.json" ||
            gpuDrivenTerrainP0Sample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("GPUDriven Terrain P0 sample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDrivenTerrainP0 =
            gpuDrivenTerrainP0Sample.graph.findNode("GPUDriven");
        if (gpuDrivenTerrainP0 == nullptr ||
            gpuDrivenTerrainP0->type != "GPUDrivenStreamAssetPass" ||
            !gpuDrivenTerrainP0->properties.is_object() ||
            gpuDrivenTerrainP0->properties.value("path", "") != gpuDrivenTerrainP0Sample.desc.scenePath ||
            gpuDrivenTerrainP0->properties.value("streamAssetPath", "") !=
                "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf.meshstream.bin" ||
            gpuDrivenTerrainP0->properties.value("autoBuildStreamAsset", true) ||
            !gpuDrivenTerrainP0->properties.value("enableGpuLodSelection", false) ||
            gpuDrivenTerrainP0->properties.value("debugColorMode", "") != "lod" ||
            !gpuDrivenTerrainP0->properties.contains("camera") ||
            !gpuDrivenTerrainP0->properties["camera"].is_object()) {
            return RHITestResult::fail("GPUDriven Terrain P0 sample did not preserve terrain defaults");
        }
        if (!gpuDrivenTerrainP0Sample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (gpuDrivenTerrainP0Sample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("GPUDriven Terrain P0 graph first output changed");
        }

        render::RenderSampleLoadResult gpuDrivenTerrainP1Sample;
        if (!render::loadBuiltInRenderSample(
                "gpu-driven-terrain-p1-unified",
                gpuDrivenTerrainP1Sample,
                message)) {
            return RHITestResult::fail(message);
        }
        if (gpuDrivenTerrainP1Sample.desc.id != "gpu-driven-terrain-p1-unified" ||
            gpuDrivenTerrainP1Sample.desc.name != "GPUDrivenSample / Terrain P1 Unified" ||
            gpuDrivenTerrainP1Sample.desc.category != "GPUDriven" ||
            gpuDrivenTerrainP1Sample.desc.scenePath !=
                "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf" ||
            !gpuDrivenTerrainP1Sample.desc.loadSceneInEditor ||
            gpuDrivenTerrainP1Sample.desc.graphPath !=
                "Pipelines/Samples/gpu_driven_terrain_p1_unified.metallic_graph.json" ||
            gpuDrivenTerrainP1Sample.desc.environment.has_value() ||
            gpuDrivenTerrainP1Sample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail(
                "GPUDriven Terrain P1 unified sample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDrivenTerrainP1 =
            gpuDrivenTerrainP1Sample.graph.findNode("GPUDriven");
        if (gpuDrivenTerrainP1 == nullptr ||
            gpuDrivenTerrainP1->type != "VisibilityBufferPass" ||
            !gpuDrivenTerrainP1->properties.is_object() ||
            gpuDrivenTerrainP1->properties.value("path", "") !=
                gpuDrivenTerrainP1Sample.desc.scenePath ||
            gpuDrivenTerrainP1->properties.value("streamAssetPath", "") !=
                "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf.meshstream.bin" ||
            !gpuDrivenTerrainP1->properties.value("enableMeshletStreaming", false) ||
            !gpuDrivenTerrainP1->properties.value("instanceHzbCull", false) ||
            !gpuDrivenTerrainP1->properties.value("meshletFrustumCull", false) ||
            gpuDrivenTerrainP1->properties.value("visualization", "") != "meshlet" ||
            !gpuDrivenTerrainP1->properties.contains("camera") ||
            !gpuDrivenTerrainP1->properties["camera"].is_object()) {
            return RHITestResult::fail(
                "GPUDriven Terrain P1 unified sample did not preserve unified raster defaults");
        }
        if (!gpuDrivenTerrainP1Sample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (gpuDrivenTerrainP1Sample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("GPUDriven Terrain P1 unified graph first output changed");
        }

        render::RenderSampleLoadResult gpuDrivenRtasSample;
        if (!render::loadBuiltInRenderSample("gpu-driven-rtas-visualization", gpuDrivenRtasSample, message)) {
            return RHITestResult::fail(message);
        }
        if (gpuDrivenRtasSample.desc.id != "gpu-driven-rtas-visualization" ||
            gpuDrivenRtasSample.desc.name != "GPUDrivenSample / RTAS Visualization" ||
            gpuDrivenRtasSample.desc.category != "GPUDriven" ||
            gpuDrivenRtasSample.desc.scenePath != "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf" ||
            gpuDrivenRtasSample.desc.loadSceneInEditor ||
            gpuDrivenRtasSample.desc.graphPath !=
                "Pipelines/Samples/gpu_driven_sponza_rtas_visualization.metallic_graph.json" ||
            !gpuDrivenRtasSample.desc.environment.has_value() ||
            gpuDrivenRtasSample.desc.environment->path != "Asset/ABeautifulGame/environment.hdr" ||
            gpuDrivenRtasSample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("GPUDriven RTAS visualization sample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDrivenRtas = gpuDrivenRtasSample.graph.findNode("GPUDriven");
        if (gpuDrivenRtas == nullptr ||
            gpuDrivenRtas->type != "GPUDrivenStreamAssetPass" ||
            !gpuDrivenRtas->properties.is_object() ||
            gpuDrivenRtas->properties.value("path", "") != gpuDrivenRtasSample.desc.scenePath ||
            !gpuDrivenRtas->properties.value("enableClusterRtx", false) ||
            !gpuDrivenRtas->properties.value("rtasVisualization", false) ||
            gpuDrivenRtas->properties.value("rtasGranularity", "") != "cluster-id") {
            return RHITestResult::fail("GPUDriven RTAS visualization sample did not apply defaults");
        }
        if (!gpuDrivenRtasSample.graph.validate(validationLog)) {
            return RHITestResult::fail(validationLog);
        }
        if (gpuDrivenRtasSample.graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("GPUDriven RTAS visualization graph first output changed");
        }

        bool listedPathTrace = false;
        bool listedOpenPBRPathTrace = false;
        bool listedDlssSrPathTrace = false;
        bool listedDlssRrPathTrace = false;
        bool listedMaterialVisualization = false;
        bool listedGPUDriven = false;
        bool listedGPUDrivenStreamAsset = false;
        bool listedGPUDrivenTerrainP0 = false;
        bool listedGPUDrivenTerrainP1 = false;
        bool listedGPUDrivenRtasVisualization = false;
        bool listedRtxcr = false;
        for (const render::RenderSampleDesc& desc : render::listBuiltInRenderSamples()) {
            listedPathTrace = listedPathTrace || desc.id == "pathtracing-meet-mat";
            listedOpenPBRPathTrace = listedOpenPBRPathTrace || desc.id == "pathtracing-sample";
            listedDlssSrPathTrace = listedDlssSrPathTrace || desc.id == "pathtracing-sample-dlss-sr";
            listedDlssRrPathTrace = listedDlssRrPathTrace || desc.id == "pathtracing-sample-dlss-rr";
            listedMaterialVisualization = listedMaterialVisualization ||
                desc.id == "material-visualization-abeautiful-game";
            listedGPUDriven = listedGPUDriven || desc.id == "gpu-driven-sample";
            listedGPUDrivenStreamAsset = listedGPUDrivenStreamAsset || desc.id == "gpu-driven-streamasset";
            listedGPUDrivenTerrainP0 = listedGPUDrivenTerrainP0 || desc.id == "gpu-driven-terrain-p0";
            listedGPUDrivenTerrainP1 = listedGPUDrivenTerrainP1 ||
                desc.id == "gpu-driven-terrain-p1-unified";
            listedGPUDrivenRtasVisualization = listedGPUDrivenRtasVisualization ||
                desc.id == "gpu-driven-rtas-visualization";
            listedRtxcr = listedRtxcr || desc.id == "rtxcr-material-sample";
        }
        if (!listedPathTrace ||
            !listedOpenPBRPathTrace ||
            !listedDlssSrPathTrace ||
            !listedDlssRrPathTrace ||
            !listedMaterialVisualization ||
            !listedGPUDriven ||
            !listedGPUDrivenStreamAsset ||
            !listedGPUDrivenTerrainP0 ||
            !listedGPUDrivenTerrainP1 ||
            !listedGPUDrivenRtasVisualization ||
            !listedRtxcr) {
            return RHITestResult::fail("built-in Sample list did not contain expected samples");
        }
        return RHITestResult::pass();
    }
};

class RenderSampleFallbackAndValidationTest : public RHITest {
public:
    RenderSampleFallbackAndValidationTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_sample_fallback_and_validation";
    }

    RHITestResult run(RHITestContext&) override
    {
        const TestPathTraceSample fallback(
            "test-fallback-preview",
            "Asset/StandfordBunny/scene.gltf",
            "");

        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadRenderSample(fallback, sample, message)) {
            return RHITestResult::fail(message);
        }
        if (sample.desc.previewOutput != "FinalBlit.color") {
            return RHITestResult::fail("Sample loader did not fallback to first graph output");
        }
        const render::RenderGraphNode* node = sample.graph.findNode("PathTrace");
        if (node == nullptr || node->properties.value("path", "") != "Asset/StandfordBunny/scene.gltf") {
            return RHITestResult::fail("Sample loader did not override target scene path");
        }

        for (const char* output : {"FinalBlit.color", "PathTrace.color"}) {
            const TestPathTraceSample explicitPreview(
                "test-explicit-preview", "Asset/meet_mat.glb", output);
            if (!render::loadRenderSample(explicitPreview, sample, message) ||
                sample.desc.previewOutput != output || !sample.graph.outputs().empty()) {
                return RHITestResult::fail("Sample loader rejected an unmarked texture preview: " + message);
            }
        }
        for (const char* output : {"Missing.color", "PathTrace.missing", "FinalBlit.source"}) {
            const TestPathTraceSample invalid("test-invalid-preview", "Asset/meet_mat.glb", output);
            if (render::loadRenderSample(invalid, sample, message)) {
                return RHITestResult::fail("Sample loader accepted invalid previewOutput");
            }
            if (message.find("previewOutput") == std::string::npos) {
                return RHITestResult::fail("Sample loader did not report previewOutput failure");
            }
        }
        return RHITestResult::pass();
    }
};
class RenderGraphValidationTest : public RHITest {
public:
    RenderGraphValidationTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_validation";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        std::string log;
        render::RenderGraph missingOutput;
        missingOutput.addNode("TriangleRasterPass", "Triangle");
        if (missingOutput.validate(log)) {
            return RHITestResult::fail("graph without outputs validated successfully");
        }

        render::RenderGraph badEndpoint = render::RenderGraph::createDefaultTriangleGraph();
        badEndpoint.addEdge("Triangle.color", "Triangle.missing");
        if (badEndpoint.validate(log)) {
            return RHITestResult::fail("graph with invalid edge endpoint validated successfully");
        }

        render::RenderGraph cyclic;
        cyclic.addNode("TestInputOutputPass", "A");
        cyclic.addNode("TestInputOutputPass", "B");
        cyclic.addEdge("A.color", "B.input");
        cyclic.addEdge("B.color", "A.input");
        cyclic.markOutput("A.color");
        if (cyclic.validate(log)) {
            return RHITestResult::fail("cyclic graph validated successfully");
        }

        render::RenderGraph textureToBuffer;
        textureToBuffer.addNode("TriangleRasterPass", "Triangle");
        textureToBuffer.addNode("TestBufferInputPass", "BufferRead");
        textureToBuffer.addEdge("Triangle.color", "BufferRead.data");
        textureToBuffer.markOutput("Triangle.color");
        if (textureToBuffer.validate(log)) {
            return RHITestResult::fail("texture-to-buffer edge validated successfully");
        }

        render::RenderGraph bufferToTexture;
        bufferToTexture.addNode("TestBufferOutputPass", "BufferWrite");
        bufferToTexture.addNode("TestInputOutputPass", "TextureRead");
        bufferToTexture.addEdge("BufferWrite.data", "TextureRead.input");
        bufferToTexture.markOutput("TextureRead.color");
        if (bufferToTexture.validate(log)) {
            return RHITestResult::fail("buffer-to-texture edge validated successfully");
        }

        render::RenderGraph missingBufferInput;
        missingBufferInput.addNode("RenderGraphBufferCopyPass", "Copy");
        missingBufferInput.markOutput("Copy.data");
        if (missingBufferInput.validate(log)) {
            return RHITestResult::fail("graph with missing required buffer input validated successfully");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphPreviewTest : public RHITest {
public:
    RenderGraphPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_triangle_preview";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph = render::RenderGraph::createDefaultTriangleGraph();
        result = preview.render(graph, 128, 96);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result));
        }
        if (countBrightPixels(preview.pixels()) < 128) {
            return RHITestResult::fail("default triangle graph produced too few bright pixels");
        }

        graph.markDirty();
        result = preview.render(graph, 64, 64);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphPreviewRenderer::render resize returned ") + toString(result));
        }
        if (preview.width() != 64 || preview.height() != 64) {
            return RHITestResult::fail("preview resize did not update output dimensions");
        }
        if (countBrightPixels(preview.pixels()) < 64) {
            return RHITestResult::fail("resized default triangle graph produced too few bright pixels");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphBunnyWireframePreviewTest : public RHITest {
public:
    RenderGraphBunnyWireframePreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_bunny_wireframe_preview";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph = render::RenderGraph::createDefaultBunnyGraph();
        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("Bunny wireframe preview is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("Bunny wireframe preview render returned ") + toString(result) + ": " + preview.lastLog());
        }

        const uint32_t brightPixels = countBrightPixels(preview.pixels());
        if (brightPixels < 512) {
            return RHITestResult::fail(
                std::string("Bunny wireframe preview produced too few bright pixels: ") +
                std::to_string(brightPixels));
        }

        return RHITestResult::pass();
    }
};

class RenderGraphBunnyCameraSyncTest : public RHITest {
public:
    RenderGraphBunnyCameraSyncTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_bunny_camera_sync";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph = render::RenderGraph::createDefaultBunnyGraph();
        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("Bunny wireframe preview is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("initial Bunny wireframe preview returned ") + toString(result) + ": " + preview.lastLog());
        }
        if (graph.dirty()) {
            return RHITestResult::fail("preview render did not clear graph dirty state");
        }

        render::RenderGraphNode* bunnyNode = graph.findNode("Bunny");
        if (bunnyNode == nullptr) {
            return RHITestResult::fail("default Bunny graph did not create Bunny node");
        }

        if (!graph.setNodeRuntimeProperty(bunnyNode->id, "camera.fovDegrees", 35.0f) ||
            !graph.setNodeRuntimeProperty(bunnyNode->id, "camera.eye", {-0.0168404f, 0.110154f, 0.34f})) {
            return RHITestResult::fail("runtime camera property update failed");
        }
        if (graph.dirty()) {
            return RHITestResult::fail("runtime camera property update unexpectedly marked graph dirty");
        }
        result = preview.render(graph, 256, 256);
        if (!result) {
            return RHITestResult::fail(
                std::string("camera-synced Bunny preview returned ") + toString(result) + ": " + preview.lastLog());
        }
        const uint32_t brightPixels = countBrightPixels(preview.pixels());
        if (brightPixels < 512) {
            return RHITestResult::fail(
                std::string("camera-synced Bunny preview produced too few bright pixels: ") +
                std::to_string(brightPixels));
        }

        return RHITestResult::pass();
    }
};

class RenderGraphSceneRayQueryVisualizationPreviewTest : public RHITest {
public:
    RenderGraphSceneRayQueryVisualizationPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_scene_rayquery_visualization_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraphProperties properties{
            {"path", "Asset/StandfordBunny/scene.gltf"},
            {"granularity", "instance"},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 60.0f},
                {"znear", 0.1f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {-0.0168404f, 0.110154f, 0.22f}},
                {"center", {-0.0168404f, 0.110154f, -0.00153695f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
        render::RenderGraph graph;
        graph.setName("SceneRayQueryVisualization");
        graph.addNode("SceneRayQueryVisualizationPass", "RayQuery", properties);
        graph.markOutput("RayQuery.color");

        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("SceneRayQueryVisualizationPass is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("SceneRayQueryVisualizationPass render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("RayQuery instance visualization produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        properties["granularity"] = "primitive";
        render::RenderGraphNode* node = graph.findNode("RayQuery");
        if (node == nullptr || !graph.setNodeProperties(node->id, properties)) {
            return RHITestResult::fail("failed to switch RayQuery visualization to primitive granularity");
        }

        result = preview.render(graph, 256, 256);
        if (!result) {
            return RHITestResult::fail(
                std::string("RayQuery primitive visualization render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("RayQuery primitive visualization produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string resultMessage;
        properties["granularity"] = "cluster-id";
        node = graph.findNode("RayQuery");
        if (node == nullptr || !graph.setNodeProperties(node->id, properties)) {
            return RHITestResult::fail("failed to switch RayQuery visualization to cluster-id granularity");
        }

        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                resultMessage = std::string("cluster-id visualization unsupported: ") + preview.lastLog() + "; ";
            } else {
                return RHITestResult::fail(
                    std::string("RayQuery cluster-id visualization render returned ") +
                    toString(result) +
                    ": " +
                    preview.lastLog());
            }
        } else {
            visiblePixelCount = countVisiblePixels(preview.pixels());
            if (visiblePixelCount < 512) {
                return RHITestResult::fail(
                    std::string("RayQuery cluster-id visualization produced too few visible pixels: ") +
                    std::to_string(visiblePixelCount));
            }
        }

        std::string outputMessage;
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_rayquery_visualization_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RHITestResult::fail(outputMessage);
        }

        return RHITestResult::pass(resultMessage + "wrote " + outputPath.string());
    }
};

class RenderGraphSceneMaterialVisualizationPreviewTest : public RHITest {
public:
    RenderGraphSceneMaterialVisualizationPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_scene_material_visualization_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("material-visualization-abeautiful-game", sample, message)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        const std::array<const char*, 13> modes{
            "material",
            "baseColor",
            "normal",
            "roughness",
            "metallic",
            "ao",
            "geometryNormal",
            "vertexNormal",
            "normalTexture",
            "tangent",
            "bitangent",
            "nrdNormalRoughness",
            "normalDeviation",
        };
        const std::string graphOutputBefore = sample.graph.firstOutputName();
        const size_t graphOutputCountBefore = sample.graph.outputs().size();
        sample.graph.clearDirty();
        for (const char* mode : modes) {
            render::RenderGraphNode* materialViz = sample.graph.findNode("MaterialViz");
            if (materialViz == nullptr || !materialViz->properties.is_object()) {
                return RHITestResult::fail("material visualization Sample graph is missing MaterialViz properties");
            }
            if (!sample.graph.setNodeRuntimeProperty(materialViz->id, "mode", mode)) {
                return RHITestResult::fail(std::string("failed to set runtime material visualization mode ") + mode);
            }
            if (sample.graph.dirty()) {
                return RHITestResult::fail(std::string("runtime material visualization mode dirtied graph: ") + mode);
            }

            result = preview.render(sample.graph, 160, 160, sample.desc.previewOutput);
            if (sample.graph.firstOutputName() != graphOutputBefore ||
                sample.graph.outputs().size() != graphOutputCountBefore) {
                return RHITestResult::fail("preview output render modified graph outputs");
            }
            if (!result) {
                if (render::hasError(result, render::Error::Unsupported)) {
                    return RHITestResult::skip(
                        std::string("SceneMaterialVisualizationPass is unsupported on this device: ") +
                        preview.lastLog());
                }
                return RHITestResult::fail(
                    std::string("SceneMaterialVisualizationPass render returned ") +
                    toString(result) +
                    " for mode " +
                    mode +
                    ": " +
                    preview.lastLog());
            }

            const std::string modeName(mode);
            const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
            if (modeName != "metallic" && visiblePixelCount < 512) {
                return RHITestResult::fail(
                    std::string("material visualization mode produced too few visible pixels: ") +
                    mode +
                    " visible=" +
                    std::to_string(visiblePixelCount));
            }
            if (modeName == "material") {
                const uint32_t distinctColorBins = countDistinctVisibleColorBins(preview.pixels());
                if (distinctColorBins < 4) {
                    return RHITestResult::fail(
                        std::string("material visualization expected multiple material colors, got bins=") +
                        std::to_string(distinctColorBins));
                }
            }

            const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
            const std::filesystem::path outputPath =
                context.outputDirectory /
                (std::string("render_graph_scene_material_visualization_") + mode + ".png");
            if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
                return RHITestResult::fail(message);
            }
        }

        return RHITestResult::pass("wrote scene material visualization previews");
    }
};
class RenderGraphScenePathTracePreviewTest : public RHITest {
public:
    RenderGraphScenePathTracePreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_scene_path_trace_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraphProperties properties{
            {"path", "Asset/meet_mat.glb"},
            {"maxDepth", 2},
            {"samples", 1},
            {"accumulate", true},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 50.0f},
                {"znear", 0.001f},
                {"zfar", 10000.0f},
                {"reversedZ", true},
                {"eye", {0.0f, 0.25f, 3.0f}},
                {"center", {0.0f, 0.15f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
        render::RenderGraph graph;
        graph.setName("ScenePathTracePreview");
        graph.addNode("ScenePathTracePass", "PathTrace", properties);
        graph.markOutput("PathTrace.color");

        result = preview.render(graph, 192, 192);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("ScenePathTracePass render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("ScenePathTracePass produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        result = preview.render(graph, 192, 192);
        if (!result) {
            return RHITestResult::fail(
                std::string("ScenePathTracePass accumulated render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("ScenePathTracePass accumulated frame produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string outputMessage;
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RHITestResult::fail(outputMessage);
        }

        return RHITestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class PathTraceCacheReadbackPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("color").transferRead().format = render::Format::RGBA8Unorm;
        reflection.addBufferOutput("data").buffer(uint64_t(context.width) * context.height * 4)
            .transferWrite().hostReadback();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        context.commandBuffer().copyTextureToBuffer({.texture = context.inputTexture("color").texture(),
            .buffer = context.outputBuffer("data").buffer(), .width = context.width(), .height = context.height()});
        return {};
    }
};

RHITestResult runPathTraceCacheStages(RHITestContext& context, bool nrc)
{
    constexpr uint32_t kWidth = 64, kHeight = 48;
    const char* mode = nrc ? "nrc" : "sharc";
    std::atomic_uint validationErrors{0};
    std::unique_ptr<render::Device> device;
    auto result = render::createDevice({.applicationName = "Path trace cache stages",
        .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
        .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
        .validationSink = {[](void* target, const render::ValidationMessage& message) noexcept {
            if (message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                ++*static_cast<std::atomic_uint*>(target);
            }
        }, &validationErrors}, .backendExtensions = metallic::render::vulkan::VulkanDeviceExtensions{.preferUnifiedImageLayouts = false}})
        .transform([&](auto value) { device = std::move(value); });
    if (render::hasError(result, render::Error::Unsupported)) {
        return RHITestResult::skip("cache stages require ray query and bindless resources");
    }
    if (!result) { return RHITestResult::fail("cache-stage device creation failed"); }
    const auto outcome = [&]() -> RHITestResult {
        auto* queue = device->getQueue(render::QueueType::Graphics);
        if (!queue) { return RHITestResult::fail("cache-stage graphics queue missing"); }
        render::registerRenderGraphPassType("PathTraceCacheReadbackPass", "Cache-stage pixel consumer",
            [] { return std::make_unique<PathTraceCacheReadbackPass>(); });
        render::RenderGraph graph;
        graph.addNode("ScenePathTracePass", "PathTrace", {
            {"path", "Asset/meet_mat.glb"}, {"bsdf", "standard"}, {"cacheMode", mode},
            {"maxDepth", 2}, {"samples", 1}, {"accumulate", true},
            {"sharc.entriesLog2", 16}, {"sharc.updateStride", 1},
            {"camera", {{"eye", {0.0, 0.25, 3.0}}, {"center", {0.0, 0.15, 0.0}}, {"fovDegrees", 50.0}}},
        });
        graph.addNode("PathTraceCacheReadbackPass", "Readback");
        graph.addEdge("PathTrace.color", "Readback.color");
        graph.markOutput("Readback.data");
        render::RenderWorld world;
        world.setEnvironment({.enabled = false, .visible = false});
        scene::LightingSettings lighting;
        lighting.exposureEV100 = 8;
        auto& light = lighting.lights.emplace_back();
        light.properties.type = "directional";
        light.properties.intensityUnit = scene::LightUnit::Lux;
        light.properties.intensity = 1000;
        light.direction = float3(0.0f, -0.2f, -1.0f);
        if (!world.setLighting(lighting)) { return RHITestResult::fail("cache-stage lighting setup failed"); }
        render::HistoryResourceManager history;
        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        std::string log;
        if (!history.initialize(*device) || !executor.compile(*device, graph, kWidth, kHeight, log)) {
            return RHITestResult::fail(std::string(mode) + " cache graph compile: " + log);
        }
        if (log.find("cache disabled") != std::string::npos) {
            return RHITestResult::fail(std::string(mode) + " test silently disabled its cache: " + log);
        }
        const std::string historyName = std::string("ScenePathTracePass.PathTrace.accumulation") + (nrc ? ".hdr" : "");
        const auto hasStage = [&](std::string_view stage) {
            for (const auto& node : executor.executionStats().nodes) {
                if (node.name != "PathTrace") { continue; }
                for (const auto& section : node.sections) {
                    if (section.name == stage) { return true; }
                }
            }
            return false;
        };
        const auto checkFrame = [&]() -> std::string {
            for (const char* stage : nrc ? std::initializer_list<const char*>{"NRC begin frame", "NRC update", "NRC query",
                     "NRC train", "NRC resolve", "NRC tonemap"}
                     : std::initializer_list<const char*>{"SHaRC update", "SHaRC resolve", "SHaRC query"}) {
                if (!hasStage(stage)) { return std::string("cache did not execute stage ") + stage; }
            }
            const auto current = history.texture(historyName, render::HistorySlot::Current);
            if (!current.valid || current.state != render::ResourceState::General) {
                return "accepted cache frame did not publish General history";
            }
            auto* output = executor.outputResource("Readback.data");
            if (!output || !output->buffer) { return "cache readback output missing"; }
            std::array<uint32_t, kWidth * kHeight> pixels{};
            if (!readHostBuffer(*output->buffer, pixels.data(), sizeof(pixels))) { return "cache readback map failed"; }
            bool visible = false;
            for (uint32_t pixel : pixels) {
                if ((pixel >> 24u) != 255u) { return "cache stages left unwritten output pixels"; }
                visible = visible || (pixel & 0x00ffffffu) != 0;
            }
            return visible ? std::string{} : "cache output contains no visible light";
        };
        for (uint32_t frame = 0; frame < 2; ++frame) {
            result = executor.execute({.graphicsQueue = queue, .historyResources = &history, .recordingWorkerLimit = 1,
                .submissionMode = render::FrameSubmissionMode::Joined});
            if (!result) {
                if (nrc && render::hasError(result, render::Error::Unsupported)) {
                    return RHITestResult::skip("NRC SDK/runtime unsupported on this device");
                }
                return RHITestResult::fail(std::string(mode) + " cache frame " + std::to_string(frame) + ": " + toString(result));
            }
            if (!executor.waitForSubmittedWork(5'000'000'000ull)) { return RHITestResult::fail("cache frame did not complete"); }
            const auto failure = checkFrame();
            if (!failure.empty()) { return RHITestResult::fail(std::string(mode) + ": " + failure); }
            if (!nrc && frame == 0 && !hasStage("SHaRC clear")) { return RHITestResult::fail("first cache frame omitted SHaRC clear"); }
        }
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        if (!device->createCommandPool(*queue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); })) {
            return RHITestResult::fail("cache cancellation command setup failed");
        }
        history.beginFrame(2);
        const auto beforeCurrent = history.texture(historyName, render::HistorySlot::Current).state;
        const auto beforePrevious = history.texture(historyName, render::HistorySlot::Previous).state;
        if (!commands->begin() || !executor.execute(*commands, &history) || !commands->end() || !pool->reset()) {
            return RHITestResult::fail(std::string(mode) + " could not cancel recorded cache stages");
        }
        const auto cancelledCurrent = history.texture(historyName, render::HistorySlot::Current);
        const auto cancelledPrevious = history.texture(historyName, render::HistorySlot::Previous);
        if (cancelledCurrent.valid || cancelledPrevious.valid || cancelledCurrent.state != beforeCurrent ||
            cancelledPrevious.state != beforePrevious) {
            return RHITestResult::fail("cancelled cache frame retained contents or changed accepted history layouts");
        }
        if (!executor.execute({.graphicsQueue = queue, .historyResources = &history, .recordingWorkerLimit = 1,
                .submissionMode = render::FrameSubmissionMode::Joined}) || !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail(std::string(mode) + " cache retry after cancellation failed");
        }
        const auto failure = checkFrame();
        if (!failure.empty()) { return RHITestResult::fail(std::string(mode) + " cancelled retry: " + failure); }
        if (!nrc && !hasStage("SHaRC clear")) { return RHITestResult::fail("cancelled SHaRC cache was not reset on retry"); }
        return RHITestResult::pass(std::string(mode) + ": 64x48, two accepted frames, cancel and retry with history/pixel checks");
    }();
    // Include executor, SDK context and device destruction in validation.
    device.reset();
    if (validationErrors != 0) {
        return RHITestResult::fail(std::string(mode) + " cache lifecycle emitted " +
            std::to_string(validationErrors.load()) + " Vulkan validation errors: " + outcome.message);
    }
    return outcome;
}

class RenderGraphSharcStagesTest final : public RHITest {
public:
    RenderGraphSharcStagesTest() { type = RHITestType::Rendering; name = "render_graph_sharc_stages_history_and_cancel"; }
    RHITestResult run(RHITestContext& context) override { return runPathTraceCacheStages(context, false); }
};

class RenderGraphNRCStagesTest final : public RHITest {
public:
    RenderGraphNRCStagesTest() { type = RHITestType::Rendering; name = "render_graph_nrc_stages_history_and_cancel"; }
    RHITestResult run(RHITestContext& context) override
    {
#if METALLIC_HAS_NRC
        const char* enabled = std::getenv("METALLIC_TEST_NRC_CACHE");
        if (!enabled || std::string_view(enabled) != "1") { return RHITestResult::skip("set METALLIC_TEST_NRC_CACHE=1 for NRC SDK cache stages"); }
        return runPathTraceCacheStages(context, true);
#else
        (void)context;
        return RHITestResult::skip("built without the NRC SDK");
#endif
    }
};

class SlangShaderDiskCacheTest : public RHITest {
public:
    SlangShaderDiskCacheTest()
    {
        type = RHITestType::Resource;
        name = "slang_shader_disk_cache_and_source_invalidation";
    }

    RHITestResult run(RHITestContext& context) override
    {
        struct ShaderDebugModeGuard {
            render::SlangShaderDebugMode previousMode = render::slangShaderDebugMode();

            ~ShaderDebugModeGuard()
            {
                render::setSlangShaderDebugMode(previousMode);
            }
        } shaderDebugModeGuard;
        render::setSlangShaderDebugMode(render::SlangShaderDebugMode::Disabled);
        struct ShaderHotReloadTrackingGuard {
            ShaderHotReloadTrackingGuard()
            {
                render::resetSlangShaderHotReloadTracking();
            }

            ~ShaderHotReloadTrackingGuard()
            {
                render::resetSlangShaderHotReloadTracking();
            }
        } shaderHotReloadTrackingGuard;

        const std::filesystem::path testRoot =
            context.outputDirectory / "slang_shader_disk_cache";
        std::error_code fileError;
        std::filesystem::remove_all(testRoot, fileError);
        if (fileError) {
            return RHITestResult::fail(
                "failed to clear shader cache test directory: " + fileError.message());
        }
        const std::filesystem::path sourceDirectory = testRoot / "source";
        const std::filesystem::path cacheDirectory = testRoot / "cache";
        const std::filesystem::path sourcePath =
            sourceDirectory / "Features/Cache/ShaderCacheTest.slang";
        const std::filesystem::path dependencyPath =
            sourceDirectory / "Libraries/Math/ShaderCacheValue.slang";
        std::filesystem::create_directories(sourcePath.parent_path(), fileError);
        if (fileError) {
            return RHITestResult::fail(
                "failed to create shader cache test directory: " + fileError.message());
        }
        std::filesystem::create_directories(dependencyPath.parent_path(), fileError);
        if (fileError) {
            return RHITestResult::fail(
                "failed to create shader library test directory: " + fileError.message());
        }
        const std::string normalizedSourcePath =
            std::filesystem::absolute(sourcePath).lexically_normal().generic_string();
        const std::string normalizedDependencyPath =
            std::filesystem::absolute(dependencyPath).lexically_normal().generic_string();
        const auto writeShader = [&]() {
            std::ofstream stream(sourcePath, std::ios::binary | std::ios::trunc);
            stream << "#include \"../../Libraries/Math/ShaderCacheValue.slang\"\n"
                   << "RWStructuredBuffer<uint> outputBuffer;\n"
                   << "[shader(\"compute\")]\n"
                   << "[numthreads(1, 1, 1)]\n"
                   << "void shaderCacheMain(uint3 dispatchId : SV_DispatchThreadID)\n"
                   << "{\n"
                   << "    outputBuffer[dispatchId.x] = kShaderCacheValue;\n"
                   << "}\n";
            return static_cast<bool>(stream);
        };
        const auto writeDependency = [&](uint32_t value) {
            std::ofstream stream(dependencyPath, std::ios::binary | std::ios::trunc);
            stream << "static const uint kShaderCacheValue = " << value << "u;\n";
            return static_cast<bool>(stream);
        };
        if (!writeShader() || !writeDependency(1u)) {
            return RHITestResult::fail("failed to write initial shader cache test source");
        }

        const std::string sourceDirectoryString = sourceDirectory.string();
        const std::string cacheDirectoryString = cacheDirectory.string();
        const render::SlangShaderDesc shaderDesc{
            .moduleName = "Features/Cache/ShaderCacheTest",
            .entryPointName = "shaderCacheMain",
            .searchPath = sourceDirectoryString.c_str(),
        };
        bool cacheHit = false;
        const render::SlangShaderCacheOptions cacheOptions{
            .cacheDirectory = cacheDirectoryString.c_str(),
            .outCacheHit = &cacheHit,
        };

        render::ShaderCompileResult firstCompile;
        render::Result<> result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, firstCompile.diagnostics).transform([&](auto value) { firstCompile = std::move(value); });
        if (!result || firstCompile.spirv.empty() || cacheHit) {
            return RHITestResult::fail(
                std::string("initial shader cache compile returned ") +
                toString(result) +
                ": " +
                firstCompile.diagnostics);
        }
        if (std::find(
                firstCompile.dependencies.begin(),
                firstCompile.dependencies.end(),
                normalizedSourcePath) == firstCompile.dependencies.end() ||
            std::find(
                firstCompile.dependencies.begin(),
                firstCompile.dependencies.end(),
                normalizedDependencyPath) == firstCompile.dependencies.end()) {
            return RHITestResult::fail(
                "initial shader compile did not publish its module and include dependencies");
        }

        render::resetSlangShaderHotReloadTracking();
        render::ShaderCompileResult cachedCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, cachedCompile.diagnostics).transform([&](auto value) { cachedCompile = std::move(value); });
        if (!result || !cacheHit ||
            cachedCompile.spirv != firstCompile.spirv) {
            return RHITestResult::fail("unchanged shader source did not hit the SPIR-V disk cache");
        }
        if (cachedCompile.dependencies != firstCompile.dependencies) {
            return RHITestResult::fail(
                "cached shader compile did not restore the complete dependency list");
        }
        if (!render::pollSlangShaderChanges(0).empty()) {
            return RHITestResult::fail(
                "cached shader dependency registration reported an unchanged file");
        }

        if (!writeDependency(123456u)) {
            return RHITestResult::fail("failed to update shader cache dependency source");
        }
        const std::vector<std::string> changedDependencies =
            render::pollSlangShaderChanges(0);
        if (changedDependencies.size() != 1 ||
            changedDependencies.front() != normalizedDependencyPath) {
            return RHITestResult::fail(
                "included shader edit was not reported as the only hot-reload dependency change");
        }
        if (!render::pollSlangShaderChanges(0).empty()) {
            return RHITestResult::fail(
                "included shader edit ignored the default retry interval");
        }
        const std::vector<std::string> retriedDependencies =
            render::pollSlangShaderChanges(0, 0);
        if (retriedDependencies != changedDependencies) {
            return RHITestResult::fail(
                "unacknowledged shader edit was not reported again after its retry interval");
        }
        render::acknowledgeSlangShaderChanges();
        if (!render::pollSlangShaderChanges(0, 0).empty()) {
            return RHITestResult::fail(
                "acknowledged shader edit remained pending");
        }
        render::ShaderCompileResult changedCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, changedCompile.diagnostics).transform([&](auto value) { changedCompile = std::move(value); });
        if (!result || changedCompile.spirv.empty() || cacheHit ||
            changedCompile.spirv == firstCompile.spirv) {
            return RHITestResult::fail("changed shader dependency did not invalidate the SPIR-V cache");
        }

        render::ShaderCompileResult changedCachedCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, changedCachedCompile.diagnostics).transform([&](auto value) { changedCachedCompile = std::move(value); });
        if (!result || !cacheHit ||
            changedCachedCompile.spirv != changedCompile.spirv) {
            return RHITestResult::fail("rebuilt shader did not become the new disk cache entry");
        }

        render::setSlangShaderDebugMode(render::SlangShaderDebugMode::CaptureSymbols);
        render::ShaderCompileResult symbolCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, symbolCompile.diagnostics).transform([&](auto value) { symbolCompile = std::move(value); });
        if (!result || symbolCompile.spirv.empty() || cacheHit ||
            !spirvContainsCaptureDebugInfo(
                symbolCompile.spirv, "ShaderCacheTest.slang", "shaderCacheMain")) {
            return RHITestResult::fail(
                "capture-symbol shader must embed source, function and line debug information");
        }
        render::ShaderCompileResult cachedSymbolCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, cachedSymbolCompile.diagnostics).transform([&](auto value) { cachedSymbolCompile = std::move(value); });
        if (!result || !cacheHit || cachedSymbolCompile.spirv != symbolCompile.spirv) {
            return RHITestResult::fail("capture-symbol shader did not use its isolated cache entry");
        }

        render::setSlangShaderDebugMode(render::SlangShaderDebugMode::ShaderDebug);
        render::ShaderCompileResult unoptimizedDebugCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, unoptimizedDebugCompile.diagnostics).transform([&](auto value) { unoptimizedDebugCompile = std::move(value); });
        if (!result || unoptimizedDebugCompile.spirv.empty() || cacheHit ||
            !spirvContainsExtendedInstructionSet(
                unoptimizedDebugCompile.spirv,
                "NonSemantic.Shader.DebugInfo.100")) {
            return RHITestResult::fail(
                "unoptimized shader-debug compile did not emit NonSemantic debug information");
        }

        size_t cacheFileCount = 0;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(cacheDirectory)) {
            cacheFileCount += entry.is_regular_file() && entry.path().extension() == ".spv"
                ? 1u
                : 0u;
        }
        if (cacheFileCount != 3) {
            return RHITestResult::fail(
                "shader cache did not isolate normal, capture-symbol, and shader-debug modes");
        }

        std::error_code stampError;
        const uintmax_t originalDependencySize =
            std::filesystem::file_size(dependencyPath, stampError);
        if (stampError) {
            return RHITestResult::fail(
                "failed to read shader dependency size before content-only edit");
        }
        const std::filesystem::file_time_type originalDependencyWriteTime =
            std::filesystem::last_write_time(dependencyPath, stampError);
        if (stampError) {
            return RHITestResult::fail(
                "failed to read shader dependency timestamp before content-only edit");
        }
        if (!writeDependency(654321u)) {
            return RHITestResult::fail(
                "failed to rewrite shader dependency with same-size content");
        }
        const uintmax_t rewrittenDependencySize =
            std::filesystem::file_size(dependencyPath, stampError);
        if (stampError || rewrittenDependencySize != originalDependencySize) {
            return RHITestResult::fail(
                "shader dependency content-only edit did not preserve its file size");
        }
        std::filesystem::last_write_time(
            dependencyPath,
            originalDependencyWriteTime,
            stampError);
        if (stampError) {
            return RHITestResult::fail(
                "failed to restore shader dependency timestamp after content-only edit");
        }
        const std::vector<std::string> contentOnlyChanges =
            render::pollSlangShaderChanges(0);
        if (contentOnlyChanges.size() != 1 ||
            contentOnlyChanges.front() != normalizedDependencyPath) {
            return RHITestResult::fail(
                "same-size shader edit with restored timestamp was not detected");
        }
        render::acknowledgeSlangShaderChanges();

        render::resetSlangShaderHotReloadTracking();
        if (!writeDependency(765432u)) {
            return RHITestResult::fail("failed to update dependency after resetting hot-reload tracking");
        }
        if (!render::pollSlangShaderChanges(0).empty()) {
            return RHITestResult::fail(
                "resetSlangShaderHotReloadTracking did not clear registered dependencies");
        }
        return RHITestResult::pass(
            "validated dependency polling, SPIR-V cache invalidation, and isolated source-line/full debug modes");
    }
};

class RenderGraphOpenPBRPathTracingShaderCompileTest : public RHITest {
public:
    RenderGraphOpenPBRPathTracingShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_openpbr_pathtracing_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::ShaderCompileResult compileResult;
        const char* capabilities[] = {"spvRayQueryKHR"};
        render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/PathTracing/OpenPBRRayQueryPathTrace",
            .entryPointName = "openPbrRayQueryPathTraceMain",
            .searchPath = kShaderSearchPath,
            .capabilities = {capabilities, static_cast<uint32_t>(std::size(capabilities))},
            .descriptorHeapMode = render::SlangDescriptorHeapMode::Native,
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("OpenPBR RayQuery path tracing shader compile returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        if (!hasNativeComputeResourceInterface(compileResult.spirv)) {
            return RHITestResult::fail("OpenPBR shader must use native heap arrays and address-based compute resources");
        }
        return RHITestResult::pass(
            std::string("compiled OpenPBR RayQuery path tracing shader, words=") +
            std::to_string(compileResult.spirv.size()));
    }
};

class GPUDrivenPreviewGeometryDedupPlanTest : public RHITest {
public:
    GPUDrivenPreviewGeometryDedupPlanTest()
    {
        type = RHITestType::Validation;
        name = "gpu_driven_preview_geometry_dedup_plan";
    }

    RHITestResult run(RHITestContext&) override
    {
        scene::RenderPrimitive shared;
        shared.meshIndex = 7;
        shared.primitiveIndex = 3;
        shared.mode = 4;
        shared.vertexCount = 3;
        shared.indexCount = 3;
        shared.triangleCount = 1;
        shared.positions = {
            float3(-1.0f, 0.0f, 0.0f),
            float3(1.0f, 0.0f, 0.0f),
            float3(0.0f, 1.0f, 0.0f),
        };
        shared.indices = {0u, 1u, 2u};
        shared.meshletClusters.resize(1);
        shared.meshletClusters[0].vertexCount = 3;
        shared.meshletClusters[0].triangleCount = 1;
        shared.meshletVertices = {0u, 1u, 2u};
        shared.meshletTriangles = {0u, 1u, 2u};

        scene::RenderPrimitive conflicting = shared;
        conflicting.positions[0].x = -2.0f;
        scene::RenderPrimitive distinct = shared;
        distinct.primitiveIndex = 4;

        const std::array<render::builtin_pass::GPUDrivenPreviewGeometrySource, 4> sources{{
            {.primitive = &shared, .renderPrimitiveIndex = 11u},
            {.primitive = &shared, .renderPrimitiveIndex = 11u},
            {.primitive = &conflicting, .renderPrimitiveIndex = 12u},
            {.primitive = &distinct, .renderPrimitiveIndex = 13u},
        }};
        const render::builtin_pass::GPUDrivenPreviewGeometryDedupPlan plan =
            render::builtin_pass::buildGPUDrivenPreviewGeometryDedupPlan(sources);
        const std::array<uint32_t, 4> expected{0u, 0u, 1u, 2u};
        if (plan.geometryCount != 3u ||
            plan.conflictingPayloadCount != 1u ||
            !std::equal(plan.geometryIndices.begin(), plan.geometryIndices.end(), expected.begin())) {
            return RHITestResult::fail(
                "shared geometry was not deduplicated or conflicting payload fallback was lost");
        }
        return RHITestResult::pass(
            "two instances share one geometry payload; conflicting and distinct keys remain separate");
    }
};

class RenderGraphGPUDrivenPreviewShaderCompileTest : public RHITest {
public:
    RenderGraphGPUDrivenPreviewShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_driven_preview_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::ShaderCompileResult amplificationCompile;
        const char* capabilities[] = {
            "spvMeshShadingEXT",
            "spvGroupNonUniformBallot",
        };
        render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/VisibilityBuffer/VisibilityBuffer",
            .entryPointName = "visibilityBufferAmplificationMain",
            .searchPath = kShaderSearchPath,
            .capabilities = {capabilities, static_cast<uint32_t>(std::size(capabilities))},
        }, amplificationCompile.diagnostics).transform([&](auto value) { amplificationCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBuffer amplification shader compile returned ") +
                toString(result) +
                ": " +
                amplificationCompile.diagnostics);
        }
        if (amplificationCompile.spirv.empty()) {
            return RHITestResult::fail(
                "VisibilityBuffer amplification shader produced empty SPIR-V");
        }

        const render::SlangMacroDefine atomicFallbackDefine{
            "GPU_DRIVEN_AMPLIFICATION_WAVE_OPS",
            "0",
        };
        const char* atomicFallbackCapabilities[] = {
            "spvMeshShadingEXT",
        };
        render::ShaderCompileResult atomicFallbackCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/VisibilityBuffer/VisibilityBuffer",
            .entryPointName = "visibilityBufferAmplificationMain",
            .searchPath = kShaderSearchPath,
            .capabilities = {atomicFallbackCapabilities, static_cast<uint32_t>(
                    std::size(atomicFallbackCapabilities))},
            .macroDefines = {&atomicFallbackDefine, 1u},
        }, atomicFallbackCompile.diagnostics).transform([&](auto value) { atomicFallbackCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBuffer atomic amplification fallback compile returned ") +
                toString(result) +
                ": " +
                atomicFallbackCompile.diagnostics);
        }
        if (atomicFallbackCompile.spirv.empty()) {
            return RHITestResult::fail(
                "VisibilityBuffer atomic amplification fallback produced empty SPIR-V");
        }

        render::ShaderCompileResult meshCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/VisibilityBuffer/VisibilityBuffer",
            .entryPointName = "visibilityBufferMeshMain",
            .searchPath = kShaderSearchPath,
            .capabilities = {capabilities, static_cast<uint32_t>(std::size(capabilities))},
        }, meshCompile.diagnostics).transform([&](auto value) { meshCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBuffer mesh shader compile returned ") +
                toString(result) +
                ": " +
                meshCompile.diagnostics);
        }
        if (meshCompile.spirv.empty()) {
            return RHITestResult::fail("VisibilityBuffer mesh shader produced empty SPIR-V");
        }

        const render::SlangMacroDefine maskedMeshDefine{
            "VISIBILITY_BUFFER_ALPHA_MASKED", "1",
        };
        render::ShaderCompileResult maskedMeshCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/VisibilityBuffer/VisibilityBuffer",
            .entryPointName = "visibilityBufferMeshMain",
            .searchPath = kShaderSearchPath,
            .capabilities = {atomicFallbackCapabilities, static_cast<uint32_t>(
                    std::size(atomicFallbackCapabilities))},
            .macroDefines = {&maskedMeshDefine, 1u},
        }, maskedMeshCompile.diagnostics).transform([&](auto value) { maskedMeshCompile = std::move(value); });
        if (!result || maskedMeshCompile.spirv.empty()) {
            return RHITestResult::fail(
                "VisibilityBuffer masked mesh compile failed: " +
                maskedMeshCompile.diagnostics);
        }

        // No vertex attributes in opaque visibility rasterization. Only the
        // masked variant may export UV and material index as user locations.
        auto countLocations = [](const std::vector<uint32_t>& spirv) -> uint32_t {
            constexpr uint16_t kOpDecorate = 71;
            constexpr uint16_t kOpMemberDecorate = 72;
            constexpr uint32_t kDecorationLocation = 30;
            uint32_t count = 0;
            for (size_t offset = 5; offset < spirv.size();) {
                const uint32_t words = spirv[offset] >> 16u;
                const uint16_t opcode = static_cast<uint16_t>(spirv[offset]);
                if (words == 0 || offset + words > spirv.size()) {
                    return UINT32_MAX;
                }
                if ((opcode == kOpDecorate && words >= 4 &&
                        spirv[offset + 2] == kDecorationLocation) ||
                    (opcode == kOpMemberDecorate && words >= 5 &&
                        spirv[offset + 3] == kDecorationLocation)) {
                    ++count;
                }
                offset += words;
            }
            return count;
        };
        if (countLocations(meshCompile.spirv) != 0u ||
            countLocations(maskedMeshCompile.spirv) != 2u) {
            return RHITestResult::fail(
                "VisibilityBuffer mesh interfaces must be position-only for opaque and UV/material for masked");
        }

        render::ShaderCompileResult fragmentCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = "Features/VisibilityBuffer/VisibilityBuffer",
                .entryPointName = "visibilityBufferFragmentMain",
                .searchPath = kShaderSearchPath,
            }, fragmentCompile.diagnostics).transform([&](auto value) { fragmentCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBuffer fragment shader compile returned ") +
                toString(result) +
                ": " +
                fragmentCompile.diagnostics);
        }
        if (fragmentCompile.spirv.empty()) {
            return RHITestResult::fail("VisibilityBuffer fragment shader produced empty SPIR-V");
        }

        struct ShaderEntry {
            const char* module;
            const char* entry;
        };
        constexpr std::array<ShaderEntry, 6> additionalEntryPoints{
            ShaderEntry{"Features/VisibilityBuffer/VisibilityBuffer", "visibilityBufferMaskedFragmentMain"},
            ShaderEntry{"Features/GPUDriven/GPUDrivenCulling", "gpuDrivenPreviewResetMain"},
            ShaderEntry{"Features/GPUDriven/GPUDrivenCulling", "gpuDrivenPreviewInstanceCullMain"},
            ShaderEntry{"Features/GPUDriven/GPUDrivenCulling", "gpuDrivenPreviewHzbMain"},
            ShaderEntry{"Features/VisibilityBuffer/VisibilityBufferComposite", "visibilityBufferCompositeVertexMain"},
            ShaderEntry{"Features/VisibilityBuffer/VisibilityBufferComposite", "visibilityBufferCompositeFragmentMain"},
        };
        size_t additionalWordCount = maskedMeshCompile.spirv.size();
        for (const ShaderEntry& shaderEntry : additionalEntryPoints) {
            const char* entryPoint = shaderEntry.entry;
            render::ShaderCompileResult compile;
            result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                    .moduleName = shaderEntry.module,
                    .entryPointName = entryPoint,
                    .searchPath = kShaderSearchPath,
                }, compile.diagnostics).transform([&](auto value) { compile = std::move(value); });
            if (!result) {
                return RHITestResult::fail(
                    std::string("VisibilityBuffer shader compile returned ") +
                    toString(result) +
                    " for " +
                    entryPoint +
                    ": " +
                    compile.diagnostics);
            }
            if (compile.spirv.empty()) {
                return RHITestResult::fail(
                    std::string("VisibilityBuffer shader produced empty SPIR-V for ") + entryPoint);
            }
            additionalWordCount += compile.spirv.size();
        }

        return RHITestResult::pass(
            std::string("compiled VisibilityBuffer shaders, amplification words=") +
            std::to_string(amplificationCompile.spirv.size()) +
            ", mesh words=" +
            std::to_string(meshCompile.spirv.size()) +
            ", fragment words=" +
            std::to_string(fragmentCompile.spirv.size()) +
            ", additional words=" +
            std::to_string(additionalWordCount));
    }
};

class RenderGraphGPUDrivenStreamAssetShaderCompileTest : public RHITest {
public:
    RenderGraphGPUDrivenStreamAssetShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_driven_streamasset_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::ShaderCompileResult meshCompile;
        const char* capabilities[] = {"spvMeshShadingEXT"};
        render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
            .entryPointName = "gpuDrivenStreamAssetMeshMain",
            .searchPath = kShaderSearchPath,
            .capabilities = {capabilities, static_cast<uint32_t>(std::size(capabilities))},
        }, meshCompile.diagnostics).transform([&](auto value) { meshCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("GPUDrivenStreamAsset mesh shader compile returned ") +
                toString(result) +
                ": " +
                meshCompile.diagnostics);
        }
        if (meshCompile.spirv.empty()) {
            return RHITestResult::fail("GPUDrivenStreamAsset mesh shader produced empty SPIR-V");
        }

        render::ShaderCompileResult fragmentCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetFragmentMain",
                .searchPath = kShaderSearchPath,
            }, fragmentCompile.diagnostics).transform([&](auto value) { fragmentCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("GPUDrivenStreamAsset fragment shader compile returned ") +
                toString(result) +
                ": " +
                fragmentCompile.diagnostics);
        }
        if (fragmentCompile.spirv.empty()) {
            return RHITestResult::fail("GPUDrivenStreamAsset fragment shader produced empty SPIR-V");
        }

        constexpr std::array<const char*, 6> kRasterEntryPoints{
            render::kMeshletStreamDeferredEntryPoint,
            render::kMeshletStreamCompositeVertexEntryPoint,
            render::kMeshletStreamCompositeFragmentEntryPoint,
            render::kMeshletStreamCullResetEntryPoint,
            render::kMeshletStreamInstanceCullEntryPoint,
            render::kMeshletStreamHZBEntryPoint,
        };
        for (const char* entryPoint : kRasterEntryPoints) {
            render::ShaderCompileResult rasterCompile;
            result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                    .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                    .entryPointName = entryPoint,
                    .searchPath = kShaderSearchPath,
                }, rasterCompile.diagnostics).transform([&](auto value) { rasterCompile = std::move(value); });
            if (!result) {
                return RHITestResult::fail(
                    std::string("GPUDrivenStreamAsset shader compile returned ") +
                    toString(result) +
                    " for " +
                    entryPoint +
                    ": " +
                    rasterCompile.diagnostics);
            }
            if (rasterCompile.spirv.empty()) {
                return RHITestResult::fail(
                    std::string("GPUDrivenStreamAsset shader produced empty SPIR-V for ") +
                    entryPoint);
            }
        }

        render::ShaderCompileResult updateCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetApplyUpdatesMain",
                .searchPath = kShaderSearchPath,
            }, updateCompile.diagnostics).transform([&](auto value) { updateCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("GPUDrivenStreamAsset update shader compile returned ") +
                toString(result) +
                ": " +
                updateCompile.diagnostics);
        }
        if (updateCompile.spirv.empty()) {
            return RHITestResult::fail("GPUDrivenStreamAsset update shader produced empty SPIR-V");
        }

        render::ShaderCompileResult traversalCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetTraversalMain",
                .searchPath = kShaderSearchPath,
            }, traversalCompile.diagnostics).transform([&](auto value) { traversalCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("GPUDrivenStreamAsset traversal shader compile returned ") +
                toString(result) +
                ": " +
                traversalCompile.diagnostics);
        }
        if (traversalCompile.spirv.empty()) {
            return RHITestResult::fail("GPUDrivenStreamAsset traversal shader produced empty SPIR-V");
        }

        render::ShaderCompileResult activeBuildCompile;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetBuildActiveMain",
                .searchPath = kShaderSearchPath,
            }, activeBuildCompile.diagnostics).transform([&](auto value) { activeBuildCompile = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("GPUDrivenStreamAsset active build shader compile returned ") +
                toString(result) +
                ": " +
                activeBuildCompile.diagnostics);
        }
        if (activeBuildCompile.spirv.empty()) {
            return RHITestResult::fail("GPUDrivenStreamAsset active build shader produced empty SPIR-V");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphGPUDrivenStreamAssetTraversalDemandTest : public RHITest {
public:
    RenderGraphGPUDrivenStreamAssetTraversalDemandTest()
    {
        type = RHITestType::Command;
        name = "render_graph_gpu_driven_streamasset_traversal_demand";
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint32_t kMaxLoadRequests = 8;
        constexpr uint32_t kMaxUnloadRequests = 8;
        constexpr uint32_t kTraversalWorkCapacity = 32;
        constexpr uint32_t kFrameIndex = 7;
        constexpr uint64_t kRequestByteSize =
            sizeof(render::StreamRequestBufferHeader) +
            (static_cast<uint64_t>(kMaxLoadRequests) + kMaxUnloadRequests) * sizeof(uint32_t);

        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic GPUDrivenStreamAsset traversal demand test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RHITestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::Queue* queue = device->getQueue(render::QueueType::Compute);
        if (queue == nullptr) {
            queue = device->getQueue(render::QueueType::Graphics);
        }
        if (queue == nullptr) {
            return RHITestResult::skip("traversal demand test device has no compute-capable queue");
        }

        auto createBuffer = [&device](
            const render::BufferDesc& desc,
            const char* label,
            std::unique_ptr<render::Buffer>& outBuffer) -> RHITestResult {
            render::Result<> bufferResult = device->createBuffer(desc).transform([&](auto rhiValue) { outBuffer = std::move(rhiValue); });
            if (!bufferResult || outBuffer == nullptr) {
                return RHITestResult::fail(std::string("createBuffer(") + label + ") returned " + toString(bufferResult));
            }
            return RHITestResult::pass();
        };

        constexpr uint32_t kActiveGroupCapacity = 8;
        const std::array<render::MeshletStreamGPUInstance, 2> instances = [] {
            std::array<render::MeshletStreamGPUInstance, 2> values{};
            for (uint32_t index = 0; index < values.size(); ++index) {
                values[index].primitiveIndex = index;
                values[index].materialIndex = index + 10u;
                values[index].visible = 1;
                values[index].world0[0] = 1.0f;
                values[index].world1[1] = 1.0f;
                values[index].world2[2] = 1.0f;
                values[index].world3[3] = 1.0f;
                values[index].boundsCenterRadius[2] = 5.0f + static_cast<float>(index);
                values[index].boundsCenterRadius[3] = 1.0f;
            }
            return values;
        }();
        const std::array<render::MeshletStreamGPUPrimitive, 2> primitives = {{
            render::MeshletStreamGPUPrimitive{
                .lodLevelOffset = 0,
                .lodLevelCount = 1,
                .pageOffset = 0,
                .pageCount = 2,
                .fallbackPageOffset = 2,
                .fallbackPageCount = 1,
                .groupOffset = 0,
                .groupCount = 3,
                .fallbackGroupOffset = 2,
                .fallbackGroupCount = 1,
                .nodeOffset = 0,
                .nodeCount = 5,
            },
            render::MeshletStreamGPUPrimitive{
                .lodLevelOffset = 1,
                .lodLevelCount = 1,
                .pageOffset = 3,
                .pageCount = 1,
                .fallbackPageOffset = 4,
                .fallbackPageCount = 1,
                .groupOffset = 3,
                .groupCount = 2,
                .fallbackGroupOffset = 4,
                .fallbackGroupCount = 1,
                .nodeOffset = 5,
                .nodeCount = 3,
            },
        }};
        const std::array<render::MeshletStreamGPULODLevel, 2> lodLevels = {{
            render::MeshletStreamGPULODLevel{
                .pageOffset = 0,
                .pageCount = 2,
                .lodLevel = 0,
                .clusterCount = 2,
                .minBoundingSphereRadius = 1.0f,
                .minMaxQuadricError = 0.0f,
            },
            render::MeshletStreamGPULODLevel{
                .pageOffset = 3,
                .pageCount = 1,
                .lodLevel = 0,
                .clusterCount = 1,
                .minBoundingSphereRadius = 1.0f,
                .minMaxQuadricError = 0.0f,
            },
        }};
        constexpr uint32_t kScenePageCount = 6;
        std::array<render::MeshletStreamGPUGroup, 5> groups{};
        const std::array<uint32_t, 5> groupPages{0, 1, 2, 3, 4};
        const std::array<uint32_t, 5> groupPrimitives{0, 0, 0, 1, 1};
        const std::array<uint32_t, 5> groupLods{0, 0, 1, 0, 1};
        const std::array<uint32_t, 5> groupClusterCounts{3, 5, 2, 11, 1};
        for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            groups[groupIndex].primitiveIndex = groupPrimitives[groupIndex];
            groups[groupIndex].pageIndex = groupPages[groupIndex];
            groups[groupIndex].lodLevel = groupLods[groupIndex];
            groups[groupIndex].clusterCount = groupClusterCounts[groupIndex];
            groups[groupIndex].boundsCenterRadius[2] = 5.0f + static_cast<float>(groupPrimitives[groupIndex]);
            groups[groupIndex].boundsCenterRadius[3] = 1.0f;
            groups[groupIndex].maxQuadricError = groupLods[groupIndex] == 0
                ? 1.0f
                : std::numeric_limits<float>::max();
        }
        std::array<render::MeshletStreamGPUNode, 8> nodes{};
        nodes[0].primitiveIndex = 0;
        nodes[0].childOffset = 1;
        nodes[0].childCount = 2;
        nodes[0].lodLevel = render::kMeshletStreamInvalidClusterIndex;
        nodes[0].maxQuadricError = std::numeric_limits<float>::max();
        nodes[1].primitiveIndex = 0;
        nodes[1].childOffset = 3;
        nodes[1].childCount = 2;
        nodes[1].lodLevel = 0;
        nodes[1].maxQuadricError = 1.0f;
        nodes[2].primitiveIndex = 0;
        nodes[2].groupIndex = 2;
        nodes[2].lodLevel = 1;
        nodes[2].maxQuadricError = std::numeric_limits<float>::max();
        nodes[3].primitiveIndex = 0;
        nodes[3].groupIndex = 0;
        nodes[3].lodLevel = 0;
        nodes[3].maxQuadricError = 1.0f;
        nodes[4].primitiveIndex = 0;
        nodes[4].groupIndex = 1;
        nodes[4].lodLevel = 0;
        nodes[4].maxQuadricError = 1.0f;
        nodes[5].primitiveIndex = 1;
        nodes[5].childOffset = 6;
        nodes[5].childCount = 2;
        nodes[5].lodLevel = render::kMeshletStreamInvalidClusterIndex;
        nodes[5].maxQuadricError = std::numeric_limits<float>::max();
        nodes[6].primitiveIndex = 1;
        nodes[6].groupIndex = 3;
        nodes[6].lodLevel = 0;
        nodes[6].maxQuadricError = 1.0f;
        nodes[7].primitiveIndex = 1;
        nodes[7].groupIndex = 4;
        nodes[7].lodLevel = 1;
        nodes[7].maxQuadricError = std::numeric_limits<float>::max();
        for (render::MeshletStreamGPUNode& node : nodes) {
            const uint32_t groupIndex = node.groupIndex;
            node.boundsCenterRadius[2] = groupIndex < groups.size()
                ? groups[groupIndex].boundsCenterRadius[2]
                : 5.5f;
            node.boundsCenterRadius[3] = 1.0f;
        }
        render::MeshletStreamGPUParams params;
        params.viewport[2] = 96.0f;
        params.viewport[3] = 1.0471975512f;
        params.frameIndex = kFrameIndex;
        params.maxGpuPageRequests = kMaxLoadRequests;
        params.maxGpuPageUnloadRequests = kMaxUnloadRequests;
        params.sceneInstanceCount = static_cast<uint32_t>(instances.size());
        params.scenePrimitiveCount = static_cast<uint32_t>(primitives.size());
        params.sceneLodLevelCount = static_cast<uint32_t>(lodLevels.size());
        params.scenePageCount = kScenePageCount;
        params.selectedLodLevel = render::kMeshletStreamNoDebugLODOverride;
        params.enableGpuLodSelection = 1;
        params.enableGpuUnloadRequests = 1;
        params.sceneGroupCount = static_cast<uint32_t>(groups.size());
        params.maxPrimitiveGroupCount = 3;
        params.sceneNodeCount = static_cast<uint32_t>(nodes.size());
        params.traversalWorkerCount = 64;
        params.traversalWorkCapacity = kTraversalWorkCapacity;
        params.activeGroupCount = kActiveGroupCapacity;
        params.maxActiveGroupClusters = 11;
        params.drawTaskCount = kActiveGroupCapacity *
            params.maxActiveGroupClusters *
            render::kMeshletStreamTriangleChunkCount;

        std::array<render::StreamPageTableEntry, kScenePageCount> pageTable{};
        pageTable[0].deviceOffsetAndState = render::packStreamPageTableEntry(
            render::kInvalidStreamDeviceOffsetBytes,
            render::MeshletStreamPageResidencyState::Unloaded);
        pageTable[1].deviceOffsetAndState = render::packStreamPageTableEntry(
            512,
            render::MeshletStreamPageResidencyState::Resident);
        pageTable[1].lastRequestFrame = 3;
        pageTable[2].deviceOffsetAndState = render::packStreamPageTableEntry(
            1536,
            render::MeshletStreamPageResidencyState::LockedFallback);
        pageTable[2].lastRequestFrame = 3;
        pageTable[3].deviceOffsetAndState = render::packStreamPageTableEntry(
            2048,
            render::MeshletStreamPageResidencyState::Resident);
        pageTable[3].lastRequestFrame = 3;
        pageTable[4].deviceOffsetAndState = render::packStreamPageTableEntry(
            4096,
            render::MeshletStreamPageResidencyState::LockedFallback);
        pageTable[4].lastRequestFrame = 3;
        pageTable[5].deviceOffsetAndState = render::packStreamPageTableEntry(
            8192,
            render::MeshletStreamPageResidencyState::Resident);
        pageTable[5].lastRequestFrame = 3;
        const std::array<uint32_t, 5> residentPageIds = {1, 2, 3, 4, 5};

        constexpr uint32_t kPageBufferBytes = 16u * 1024u;
        std::vector<uint32_t> pageWords(kPageBufferBytes / sizeof(uint32_t), 0u);
        static_assert(sizeof(scene::MeshletStreamPayloadHeader) == 112u);
        static_assert(sizeof(scene::MeshletStreamPayloadCluster) == 96u);
        constexpr uint32_t kClusterOffsetBytes =
            sizeof(scene::MeshletStreamPayloadHeader);
        constexpr uint32_t kClusterStrideWords =
            sizeof(scene::MeshletStreamPayloadCluster) / sizeof(uint32_t);
        for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            const uint32_t pageIndex = groups[groupIndex].pageIndex;
            const render::StreamPageTableEntry& entry = pageTable[pageIndex];
            const uint32_t deviceOffsetBytes = render::streamPageTableDeviceOffset(entry);
            if (deviceOffsetBytes == render::kInvalidStreamDeviceOffsetBytes) {
                continue;
            }
            const uint32_t pageWord = deviceOffsetBytes / sizeof(uint32_t);
            const uint32_t payloadBytes =
                kClusterOffsetBytes + groups[groupIndex].clusterCount *
                    sizeof(scene::MeshletStreamPayloadCluster);
            pageWords[pageWord + 2u] = groups[groupIndex].clusterCount;
            pageWords[pageWord + 9u] = kClusterOffsetBytes;
            pageWords[pageWord + 12u] = payloadBytes;
            const uint32_t clusterWord = pageWord + kClusterOffsetBytes / sizeof(uint32_t);
            for (uint32_t clusterIndex = 0; clusterIndex < groups[groupIndex].clusterCount; ++clusterIndex) {
                pageWords[clusterWord + clusterIndex * kClusterStrideWords + 8u] = UINT32_MAX;
            }
        }
        const uint32_t group2ClusterWord =
            render::streamPageTableDeviceOffset(pageTable[groups[2].pageIndex]) /
                sizeof(uint32_t) + kClusterOffsetBytes / sizeof(uint32_t);
        pageWords[group2ClusterWord + 8u] = 0u;
        pageWords[group2ClusterWord + kClusterStrideWords + 8u] = 1u;
        const uint32_t group4ClusterWord =
            render::streamPageTableDeviceOffset(pageTable[groups[4].pageIndex]) /
                sizeof(uint32_t) + kClusterOffsetBytes / sizeof(uint32_t);
        pageWords[group4ClusterWord + 8u] = 3u;
        params.pageBufferBytes = kPageBufferBytes;

        std::vector<uint8_t> requestInit(static_cast<size_t>(kRequestByteSize), 0);
        auto* requestHeader = reinterpret_cast<render::StreamRequestBufferHeader*>(requestInit.data());
        requestHeader->maxLoadRequests = kMaxLoadRequests;
        requestHeader->maxUnloadRequests = kMaxUnloadRequests;
        requestHeader->frameIndex = kFrameIndex;

        std::unique_ptr<render::Buffer> instanceBuffer;
        RHITestResult testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(instances),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "instances",
            instanceBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> primitiveBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(primitives),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "primitives",
            primitiveBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> lodLevelBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(lodLevels),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "lod levels",
            lodLevelBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> groupBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(groups),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "groups",
            groupBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> residentPageBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(residentPageIds),
                .structureStride = sizeof(uint32_t),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "resident pages",
            residentPageBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> pageBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = static_cast<uint64_t>(pageWords.size()) * sizeof(uint32_t),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "stream pages",
            pageBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> nodeBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(nodes),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "hierarchy nodes",
            nodeBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> paramsBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(params),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "params",
            paramsBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> pageTableUploadBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(pageTable),
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "page table upload",
            pageTableUploadBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> requestUploadBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = kRequestByteSize,
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "request upload",
            requestUploadBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> pageTableBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(pageTable),
                .structureStride = sizeof(render::StreamPageTableEntry),
                .usage = render::BufferUsageBits::Storage |
                    render::BufferUsageBits::TransferDestination |
                    render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::Device,
            },
            "page table",
            pageTableBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> requestBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = kRequestByteSize,
                .structureStride = sizeof(uint32_t),
                .usage = render::BufferUsageBits::Storage |
                    render::BufferUsageBits::TransferDestination |
                    render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::Device,
            },
            "request",
            requestBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> activeGroupBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = static_cast<uint64_t>(kActiveGroupCapacity) * sizeof(render::MeshletStreamGPUActiveGroup),
                .structureStride = sizeof(render::MeshletStreamGPUActiveGroup),
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::Device,
            },
            "active groups",
            activeGroupBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> activeHeaderBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(render::MeshletStreamGPUActiveHeader),
                .structureStride = sizeof(render::MeshletStreamGPUActiveHeader),
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::Device,
            },
            "active header",
            activeHeaderBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> drawIndirectBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = render::kMeshletStreamDrawIndirectCommandCount * sizeof(render::MeshletStreamGPUDrawIndirect),
                .structureStride = sizeof(render::MeshletStreamGPUDrawIndirect),
                .usage = render::BufferUsageBits::Storage |
                    render::BufferUsageBits::Indirect |
                    render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::Device,
            },
            "draw indirect",
            drawIndirectBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> traversalHeaderBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(render::MeshletStreamGPUTraversalHeader),
                .structureStride = sizeof(render::MeshletStreamGPUTraversalHeader),
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::Device,
            },
            "traversal header",
            traversalHeaderBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> traversalWorkBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = static_cast<uint64_t>(kTraversalWorkCapacity) *
                    sizeof(render::MeshletStreamGPUTraversalWorkItem),
                .structureStride = sizeof(render::MeshletStreamGPUTraversalWorkItem),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::Device,
            },
            "traversal work",
            traversalWorkBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> activeGroupReadbackBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = activeGroupBuffer->desc().size,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "active groups readback",
            activeGroupReadbackBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> activeHeaderReadbackBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = activeHeaderBuffer->desc().size,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "active header readback",
            activeHeaderReadbackBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> drawIndirectReadbackBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = render::kMeshletStreamDrawIndirectCommandCount * sizeof(render::MeshletStreamGPUDrawIndirect),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "draw indirect readback",
            drawIndirectReadbackBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> traversalHeaderReadbackBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(render::MeshletStreamGPUTraversalHeader),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "traversal header readback",
            traversalHeaderReadbackBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> pageTableReadbackBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = sizeof(pageTable),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "page table readback",
            pageTableReadbackBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> requestReadbackBuffer;
        testResult = createBuffer(
            render::BufferDesc{
                .size = kRequestByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "request readback",
            requestReadbackBuffer);
        if (!testResult.passed) {
            return testResult;
        }

        result = writeHostBuffer(*instanceBuffer, instances.data(), sizeof(instances));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(instances) returned ") + toString(result));
        }
        result = writeHostBuffer(*primitiveBuffer, primitives.data(), sizeof(primitives));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(primitives) returned ") + toString(result));
        }
        result = writeHostBuffer(*lodLevelBuffer, lodLevels.data(), sizeof(lodLevels));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(lod levels) returned ") + toString(result));
        }
        result = writeHostBuffer(*groupBuffer, groups.data(), sizeof(groups));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(groups) returned ") + toString(result));
        }
        result = writeHostBuffer(
            *pageBuffer,
            pageWords.data(),
            static_cast<uint64_t>(pageWords.size()) * sizeof(uint32_t));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(stream pages) returned ") + toString(result));
        }
        result = writeHostBuffer(*nodeBuffer, nodes.data(), sizeof(nodes));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(hierarchy nodes) returned ") + toString(result));
        }
        result = writeHostBuffer(*paramsBuffer, &params, sizeof(params));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(params) returned ") + toString(result));
        }
        result = writeHostBuffer(*pageTableUploadBuffer, pageTable.data(), sizeof(pageTable));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(page table upload) returned ") + toString(result));
        }
        result = writeHostBuffer(*requestUploadBuffer, requestInit.data(), kRequestByteSize);
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(request upload) returned ") + toString(result));
        }
        result = writeHostBuffer(*residentPageBuffer, residentPageIds.data(), sizeof(residentPageIds));
        if (!result) {
            return RHITestResult::fail(std::string("writeHostBuffer(resident pages) returned ") + toString(result));
        }

        std::unique_ptr<render::BindlessHeap> bindlessHeap;
        result = device->createBindlessHeap(render::BindlessHeapDesc{
                .maxSamplers = 0,
                .maxSampledImages = 0,
                .maxBuffers = 15,
            }).transform([&](auto rhiValue) { bindlessHeap = std::move(rhiValue); });
        if (!result || bindlessHeap == nullptr) {
            return RHITestResult::fail(std::string("createBindlessHeap returned ") + toString(result));
        }

        auto allocateStorageBuffer = [&bindlessHeap](
            render::Buffer& buffer,
            const char* label,
            render::BindlessHandle& outHandle) -> RHITestResult {
            render::Result<> bindlessResult = bindlessHeap->allocateBuffer().transform([&](auto rhiValue) { outHandle = std::move(rhiValue); });
            if (!bindlessResult || !outHandle.valid()) {
                return RHITestResult::fail(std::string("allocateBuffer(") + label + ") returned " + toString(bindlessResult));
            }
            bindlessResult = bindlessHeap->writeStorageBuffer(outHandle, buffer);
            if (!bindlessResult) {
                return RHITestResult::fail(std::string("writeStorageBuffer(") + label + ") returned " + toString(bindlessResult));
            }
            return RHITestResult::pass();
        };

        render::BindlessHandle instanceHandle;
        testResult = allocateStorageBuffer(*instanceBuffer, "instances", instanceHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle primitiveHandle;
        testResult = allocateStorageBuffer(*primitiveBuffer, "primitives", primitiveHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle lodLevelHandle;
        testResult = allocateStorageBuffer(*lodLevelBuffer, "lod levels", lodLevelHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle groupHandle;
        testResult = allocateStorageBuffer(*groupBuffer, "groups", groupHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle residentPageHandle;
        testResult = allocateStorageBuffer(*residentPageBuffer, "resident pages", residentPageHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle pageHandle;
        testResult = allocateStorageBuffer(*pageBuffer, "stream pages", pageHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle nodeHandle;
        testResult = allocateStorageBuffer(*nodeBuffer, "hierarchy nodes", nodeHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle pageTableHandle;
        testResult = allocateStorageBuffer(*pageTableBuffer, "page table", pageTableHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle paramsHandle;
        testResult = allocateStorageBuffer(*paramsBuffer, "params", paramsHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle requestHandle;
        testResult = allocateStorageBuffer(*requestBuffer, "request", requestHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle activeGroupHandle;
        testResult = allocateStorageBuffer(*activeGroupBuffer, "active groups", activeGroupHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle activeHeaderHandle;
        testResult = allocateStorageBuffer(*activeHeaderBuffer, "active header", activeHeaderHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle drawIndirectHandle;
        testResult = allocateStorageBuffer(*drawIndirectBuffer, "draw indirect", drawIndirectHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle traversalHeaderHandle;
        testResult = allocateStorageBuffer(*traversalHeaderBuffer, "traversal header", traversalHeaderHandle);
        if (!testResult.passed) {
            return testResult;
        }
        render::BindlessHandle traversalWorkHandle;
        testResult = allocateStorageBuffer(*traversalWorkBuffer, "traversal work", traversalWorkHandle);
        if (!testResult.passed) {
            return testResult;
        }

        render::ShaderCompileResult compileResult;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetTraversalMain",
                .searchPath = kShaderSearchPath,
            }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("compileSlangShaderToSpirv(traversal) returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        std::unique_ptr<render::ShaderModule> traversalShader;
        result = device->createShaderModule(render::ShaderModuleDesc{
            .spirv = compileResult.spirv,
        }).transform([&](auto rhiValue) { traversalShader = std::move(rhiValue); });
        if (!result || traversalShader == nullptr) {
            return RHITestResult::fail(std::string("createShaderModule(traversal) returned ") + toString(result));
        }

        std::unique_ptr<render::ComputePipeline> pipeline;
        result = device->createComputePipeline(render::ComputePipelineDesc{
            .computeShader = {traversalShader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(render::MeshletStreamUserPush),
        }).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); });
        if (!result || pipeline == nullptr) {
            return RHITestResult::fail(std::string("createComputePipeline(traversal) returned ") + toString(result));
        }

        render::ShaderCompileResult activeBuildCompileResult;
        result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = "Features/GPUDriven/GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetBuildActiveMain",
                .searchPath = kShaderSearchPath,
            }, activeBuildCompileResult.diagnostics).transform([&](auto value) { activeBuildCompileResult = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("compileSlangShaderToSpirv(active build) returned ") +
                toString(result) +
                ": " +
                activeBuildCompileResult.diagnostics);
        }
        std::unique_ptr<render::ShaderModule> activeBuildShader;
        result = device->createShaderModule(render::ShaderModuleDesc{
            .spirv = activeBuildCompileResult.spirv,
        }).transform([&](auto rhiValue) { activeBuildShader = std::move(rhiValue); });
        if (!result || activeBuildShader == nullptr) {
            return RHITestResult::fail(std::string("createShaderModule(active build) returned ") + toString(result));
        }

        std::unique_ptr<render::ComputePipeline> activeBuildPipeline;
        result = device->createComputePipeline(render::ComputePipelineDesc{
            .computeShader = {activeBuildShader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(render::MeshletStreamUserPush),
        }).transform([&](auto rhiValue) { activeBuildPipeline = std::move(rhiValue); });
        if (!result || activeBuildPipeline == nullptr) {
            return RHITestResult::fail(std::string("createComputePipeline(active build) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*queue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }
        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        std::array<render::BufferBarrierDesc, 2> uploadBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = {},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
                .range = {.offset = 0, .size = pageTableBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = {},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
                .range = {.offset = 0, .size = requestBuffer->desc().size},
            },
        }};
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = uploadBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        {
            auto sourceSlice = pageTableUploadBuffer.get()->slice({0, pageTableBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = pageTableBuffer.get()->slice({0, pageTableBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        {
            auto sourceSlice = requestUploadBuffer.get()->slice({0, requestBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = requestBuffer.get()->slice({0, requestBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        std::array<render::BufferBarrierDesc, 2> generalBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = pageTableBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = requestBuffer->desc().size},
            },
        }};
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = generalBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }

        commandBuffer->bindBindlessHeap(*bindlessHeap);
        render::MeshletStreamUserPush push{
            .pageBuffer = pageHandle.shaderIndex,
            .activeGroupBuffer = activeGroupHandle.shaderIndex,
            .pageTableBuffer = pageTableHandle.shaderIndex,
            .paramsBuffer = paramsHandle.shaderIndex,
            .requestBuffer = requestHandle.shaderIndex,
            .residentPageBuffer = residentPageHandle.shaderIndex,
            .activeHeaderBuffer = activeHeaderHandle.shaderIndex,
            .instanceBuffer = instanceHandle.shaderIndex,
            .primitiveBuffer = primitiveHandle.shaderIndex,
            .lodLevelBuffer = lodLevelHandle.shaderIndex,
            .groupBuffer = groupHandle.shaderIndex,
            .nodeBuffer = nodeHandle.shaderIndex,
            .drawIndirectBuffer = drawIndirectHandle.shaderIndex,
            .traversalHeaderBuffer = traversalHeaderHandle.shaderIndex,
            .traversalWorkBuffer = traversalWorkHandle.shaderIndex,
            .traversalPhase = render::kMeshletStreamTraversalLoadPhase,
        };

        std::array<render::BufferBarrierDesc, 7> activeBuildBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = pageTableBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = requestBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = activeGroupBuffer.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = activeGroupBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = activeHeaderBuffer.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = activeHeaderBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = drawIndirectBuffer.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = drawIndirectBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = traversalHeaderBuffer.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = traversalHeaderBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = traversalWorkBuffer.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = traversalWorkBuffer->desc().size},
            },
        }};
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = activeBuildBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }

        if (auto commandResult = commandBuffer->bindExecution((activeBuildPipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
        push.activeBuildPhase = render::kMeshletStreamActiveBuildResetPhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        std::array<render::BufferBarrierDesc, 7> activePhaseBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = pageTableBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = requestBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = activeGroupBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = activeGroupBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = activeHeaderBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = activeHeaderBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = drawIndirectBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = drawIndirectBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = traversalHeaderBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = traversalHeaderBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = traversalWorkBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .range = {.offset = 0, .size = traversalWorkBuffer->desc().size},
            },
        }};
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = activePhaseBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }

        push.activeBuildPhase = render::kMeshletStreamActiveBuildSeedPhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = activePhaseBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        push.activeBuildPhase = render::kMeshletStreamActiveBuildRunPhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = activePhaseBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        push.activeBuildPhase = render::kMeshletStreamActiveBuildFinalizePhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = activePhaseBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        if (auto commandResult = commandBuffer->bindExecution((pipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
        push.traversalPhase = render::kMeshletStreamTraversalUnloadPhase;
        push.activeBuildPhase = static_cast<uint32_t>(residentPageIds.size());
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        std::array<render::BufferBarrierDesc, 6> readbackBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                .range = {.offset = 0, .size = pageTableBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                .range = {.offset = 0, .size = requestBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = activeGroupBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                .range = {.offset = 0, .size = activeGroupBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = activeHeaderBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                .range = {.offset = 0, .size = activeHeaderBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = drawIndirectBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                .range = {.offset = 0, .size = drawIndirectBuffer->desc().size},
            },
            render::BufferBarrierDesc{
                .buffer = traversalHeaderBuffer.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                .range = {.offset = 0, .size = traversalHeaderBuffer->desc().size},
            },
        }};
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = readbackBarriers,
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        {
            auto sourceSlice = pageTableBuffer.get()->slice({0, pageTableBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = pageTableReadbackBuffer.get()->slice({0, pageTableBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        {
            auto sourceSlice = requestBuffer.get()->slice({0, requestBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = requestReadbackBuffer.get()->slice({0, requestBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        {
            auto sourceSlice = activeGroupBuffer.get()->slice({0, activeGroupBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = activeGroupReadbackBuffer.get()->slice({0, activeGroupBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        {
            auto sourceSlice = activeHeaderBuffer.get()->slice({0, activeHeaderBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = activeHeaderReadbackBuffer.get()->slice({0, activeHeaderBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        {
            auto sourceSlice = drawIndirectBuffer.get()->slice({0, drawIndirectBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = drawIndirectReadbackBuffer.get()->slice({0, drawIndirectBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        {
            auto sourceSlice = traversalHeaderBuffer.get()->slice({0, traversalHeaderBuffer->desc().size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = traversalHeaderReadbackBuffer.get()->slice({0, traversalHeaderBuffer->desc().size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = queue->submit(render::QueueSubmitDesc{
            .commandBuffers = {commandBuffers, 1},
            .signalFence = fence.get(),
        });
        if (!result) {
            return RHITestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }
        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        std::array<render::StreamPageTableEntry, 6> pageTableResult{};
        if (!readHostBuffer(*pageTableReadbackBuffer, pageTableResult.data(), sizeof(pageTableResult))) {
            return RHITestResult::fail("page table readback buffer did not map");
        }
        std::vector<uint8_t> requestResult(static_cast<size_t>(kRequestByteSize), 0);
        if (!readHostBuffer(*requestReadbackBuffer, requestResult.data(), kRequestByteSize)) {
            return RHITestResult::fail("request readback buffer did not map");
        }
        const auto* actualHeader =
            reinterpret_cast<const render::StreamRequestBufferHeader*>(requestResult.data());
        const auto* actualPageIds = reinterpret_cast<const uint32_t*>(
            requestResult.data() + sizeof(render::StreamRequestBufferHeader));
        if (actualHeader->loadCounter != 1 ||
            actualHeader->unloadCounter != 1 ||
            actualHeader->loadOverflowCounter != 0 ||
            actualHeader->unloadOverflowCounter != 0 ||
            actualHeader->invalidPageCounter != 0 ||
            actualPageIds[0] != 0 ||
            actualPageIds[kMaxLoadRequests] != 5) {
            return RHITestResult::fail("traversal demand shader did not emit expected load/unload requests");
        }
        if (pageTableResult[0].lastRequestFrame != kFrameIndex ||
            pageTableResult[1].lastRequestFrame != kFrameIndex ||
            pageTableResult[2].lastRequestFrame != kFrameIndex ||
            pageTableResult[3].lastRequestFrame != kFrameIndex ||
            pageTableResult[4].lastRequestFrame != kFrameIndex ||
            pageTableResult[5].lastRequestFrame == kFrameIndex) {
            return RHITestResult::fail("traversal demand shader did not mark selected pages conservatively");
        }
        render::MeshletStreamGPUTraversalHeader traversalHeaderResult;
        if (!readHostBuffer(
                *traversalHeaderReadbackBuffer,
                &traversalHeaderResult,
                sizeof(traversalHeaderResult))) {
            return RHITestResult::fail("traversal header readback buffer did not map");
        }
        if (traversalHeaderResult.writeCounter != nodes.size() ||
            traversalHeaderResult.readCounter < traversalHeaderResult.writeCounter ||
            traversalHeaderResult.taskCounter != 0 ||
            traversalHeaderResult.overflowCount != 0 ||
            traversalHeaderResult.frameIndex != kFrameIndex) {
            return RHITestResult::fail("persistent traversal queue did not drain as expected");
        }
        render::MeshletStreamGPUActiveHeader activeHeaderResult;
        if (!readHostBuffer(*activeHeaderReadbackBuffer, &activeHeaderResult, sizeof(activeHeaderResult))) {
            return RHITestResult::fail("active header readback buffer did not map");
        }
        std::array<render::MeshletStreamGPUActiveGroup, kActiveGroupCapacity> activeGroupsResult{};
        if (!readHostBuffer(*activeGroupReadbackBuffer, activeGroupsResult.data(), sizeof(activeGroupsResult))) {
            return RHITestResult::fail("active group readback buffer did not map");
        }
        if (activeHeaderResult.activeGroupCount != 3 ||
            activeHeaderResult.activeGroupCapacity != kActiveGroupCapacity ||
            activeHeaderResult.maxActiveGroupClusters != params.maxActiveGroupClusters ||
            activeHeaderResult.overflowCount != 0 ||
            activeHeaderResult.frameIndex != kFrameIndex) {
            return RHITestResult::fail("active table header was not built as expected");
        }
        render::MeshletStreamGPUDrawIndirect drawIndirectResult;
        if (!readHostBuffer(*drawIndirectReadbackBuffer, &drawIndirectResult, sizeof(drawIndirectResult))) {
            return RHITestResult::fail("draw indirect readback buffer did not map");
        }
        if (drawIndirectResult.groupCountX !=
                activeHeaderResult.activeGroupCount *
                    activeHeaderResult.maxActiveGroupClusters *
                    render::kMeshletStreamTriangleChunkCount ||
            drawIndirectResult.groupCountY != 1 ||
            drawIndirectResult.groupCountZ != 1) {
            return RHITestResult::fail("active table did not generate the expected indirect mesh task command");
        }

        bool foundResidentFinePage0 = false;
        bool foundFallbackPage = false;
        bool foundResidentFinePage = false;
        for (uint32_t index = 0; index < activeHeaderResult.activeGroupCount; ++index) {
            const render::MeshletStreamGPUActiveGroup& group = activeGroupsResult[index];
            if (group.pageIndex == 1 &&
                group.clusterCount == groupClusterCounts[1] &&
                group.materialIndex == instances[0].materialIndex &&
                group.instanceIndex == 0u &&
                group.clusterSelectionMask == 0x1fu &&
                group.flags == render::kMeshletStreamActiveGroupResident) {
                foundResidentFinePage0 = true;
            }
            if (group.pageIndex == 2 &&
                group.clusterCount == groupClusterCounts[2] &&
                group.materialIndex == instances[0].materialIndex &&
                group.instanceIndex == 0u &&
                group.clusterSelectionMask == 0x1u &&
                group.flags == render::kMeshletStreamActiveGroupResident) {
                foundFallbackPage = true;
            }
            if (group.pageIndex == 3 &&
                group.clusterCount == groupClusterCounts[3] &&
                group.materialIndex == instances[1].materialIndex &&
                group.instanceIndex == 1u &&
                group.clusterSelectionMask == 0x7ffu &&
                group.flags == render::kMeshletStreamActiveGroupResident) {
                foundResidentFinePage = true;
            }
        }
        if (!foundResidentFinePage0 || !foundFallbackPage || !foundResidentFinePage) {
            return RHITestResult::fail("active table did not compact group-level fine and fallback selections");
        }

        (void)device->waitIdle();
        return RHITestResult::pass();
    }
};

class RenderGraphRTXDIPreviewTest : public RHITest {
public:
    RenderGraphRTXDIPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_rtxdi_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(true, true);
        if (!result) {
            return RHITestResult::skip(
                std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("rtxdi-sample", sample, message)) {
            return RHITestResult::fail(message);
        }
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        constexpr uint32_t kRelaxFrameCount = 8;
        for (uint32_t frame = 0; frame < kRelaxFrameCount; ++frame) {
            result = preview.render(sample.graph, 256, 256, sample.desc.previewOutput);
            if (!result) {
                if (frame == 0 && render::hasError(result, render::Error::Unsupported)) {
                    return RHITestResult::skip(
                        std::string("RTXDI/RELAX graph is unsupported on this device: ") + preview.lastLog());
                }
                return RHITestResult::fail(
                    std::string("RTXDI/RELAX frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result) +
                    ": " +
                    preview.lastLog());
            }
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 128) {
            return RHITestResult::fail(
                std::string("RTXDI/RELAX graph produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_rtxdi_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RHITestResult::fail(message);
        }

        result = preview.render(sample.graph, 256, 256, "Confidence.diffuseConfidence");
        if (!result) {
            return RHITestResult::fail(
                std::string("RTXDI diffuse confidence readback returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const auto* confidenceBytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const size_t confidencePixelCount =
            static_cast<size_t>(preview.width()) * static_cast<size_t>(preview.height());
        uint8_t minimumConfidence = std::numeric_limits<uint8_t>::max();
        uint8_t maximumConfidence = 0;
        for (size_t pixelIndex = 0; pixelIndex < confidencePixelCount; ++pixelIndex) {
            minimumConfidence = std::min(minimumConfidence, confidenceBytes[pixelIndex]);
            maximumConfidence = std::max(maximumConfidence, confidenceBytes[pixelIndex]);
        }
        if (maximumConfidence == 0 || minimumConfidence == maximumConfidence) {
            return RHITestResult::fail(
                "RTXDI diffuse confidence output is empty or constant");
        }
        return RHITestResult::pass("wrote RTXDI RELAX preview");
    }
};

#if defined(METALLIC_HAS_RTXCR) && METALLIC_HAS_RTXCR
class RenderGraphRTXCRMaterialShaderCompileTest : public RHITest {
public:
    RenderGraphRTXCRMaterialShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_rtxcr_material_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        const char* additionalSearchPaths[] = {METALLIC_RTXCR_SHADER_INCLUDE_DIR};
        render::ShaderCompileResult compileResult;
        render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/Samples/RTXCRMaterialSample",
            .entryPointName = "rtxcrMaterialSampleMain",
            .searchPath = kShaderSearchPath,
            .additionalSearchPaths = {additionalSearchPaths, static_cast<uint32_t>(std::size(additionalSearchPaths))},
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result || compileResult.spirv.empty()) {
            return RHITestResult::fail(
                std::string("RTXCR material shader compile returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        return RHITestResult::pass(
            std::string("compiled RTXCR material shader, words=") +
            std::to_string(compileResult.spirv.size()));
    }
};

#if defined(METALLIC_HAS_RTXCR_GEOMETRY) && METALLIC_HAS_RTXCR_GEOMETRY && \
    defined(METALLIC_HAS_RTXCR_ASSETS) && METALLIC_HAS_RTXCR_ASSETS
class RenderGraphRTXCRMaterialPreviewTest : public RHITest {
public:
    RenderGraphRTXCRMaterialPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_rtxcr_material_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("rtxcr-material-sample", sample, message)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(
                std::string("RenderGraphPreviewRenderer::initialize returned ") +
                toString(result));
        }
        result = preview.render(sample.graph, 768, 432, sample.desc.previewOutput);
        if (!result) {
            return RHITestResult::fail(
                std::string("RTXCR material preview returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        // AutoExposure meters the HDRI-dominated frame to middle gray, so the
        // old >120 bright-pixel check no longer matches the sample graph.
        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 1024) {
            return RHITestResult::fail(
                std::string("RTXCR material preview produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_rtxcr_material_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RHITestResult::fail(message);
        }
        return RHITestResult::pass("wrote " + outputPath.string());
    }
};
#endif
#endif

class RenderGraphRTXDIShaderCompileTest : public RHITest {
public:
    RenderGraphRTXDIShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_rtxdi_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        const char* capabilities[] = {"spvRayQueryKHR"};
        const struct ShaderEntry {
            const char* moduleName;
            const char* entryPointName;
            bool rayQuery;
        } entries[] = {
            {"Features/Lighting/BuildReGIR", "buildReGIRMain", false},
            {"Features/Lighting/PrepareLightsPdf", "prepareLightsPdfMain", false},
            {"Features/ReSTIR/SceneRTXDI", "sceneRtxdiMain", true},
            {"Features/ReSTIR/RTXDIConfidence", "rtxdiConfidenceMain", false},
            {"Features/ReSTIR/RTXDIComposite", "rtxdiCompositeMain", false},
        };
        for (const ShaderEntry& entry : entries) {
            render::ShaderCompileResult compileResult;
            render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                .moduleName = entry.moduleName,
                .entryPointName = entry.entryPointName,
                .searchPath = kShaderSearchPath,
                .capabilities = {entry.rayQuery ? capabilities : nullptr, entry.rayQuery
                        ? static_cast<uint32_t>(std::size(capabilities))
                        : 0u},
                .descriptorHeapMode = render::SlangDescriptorHeapMode::Native,
            }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
            if (!result) {
                return RHITestResult::fail(
                    std::string("RTXDI shader compile returned ") +
                    toString(result) +
                    ": " +
                    compileResult.diagnostics);
            }
            if (!hasNativeComputeResourceInterface(compileResult.spirv)) {
                return RHITestResult::fail(std::string(entry.moduleName) + " retained a fixed descriptor binding or invalid compute resource ABI");
            }
        }
        return RHITestResult::pass("compiled RTXDI ReSTIR DI and RELAX composite shaders");
    }
};

class RenderGraphPathTracingGuidesShaderCompileTest : public RHITest {
public:
    RenderGraphPathTracingGuidesShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_pathtracing_guides_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        const char* capabilities[] = {"spvRayQueryKHR", "spvRayQueryPositionFetchKHR"};
        const struct ShaderEntry {
            const char* moduleName;
            const char* entryPointName;
        } entries[] = {
            {"Features/PathTracing/ScenePathTraceGuides", "scenePathTraceGuidesMain"},
            {"Features/PathTracing/OpenPBRRayQueryPathTraceGuides", "openPbrRayQueryPathTraceGuidesMain"},
            {"Features/Debug/SceneMaterialVisualize", "sceneMaterialVisualizeMain"},
            {"Features/ReSTIR/SceneRTXDI", "sceneRtxdiMain"},
        };

        for (uint32_t positionFetch : {0u, 1u}) {
            const render::SlangMacroDefine defines[] = {
                {"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", positionFetch != 0 ? "1" : "0"},
            };
            for (const ShaderEntry& entry : entries) {
                render::ShaderCompileResult compileResult;
                render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                    .moduleName = entry.moduleName,
                    .entryPointName = entry.entryPointName,
                    .searchPath = kShaderSearchPath,
                    .capabilities = {capabilities, 1u + positionFetch},
                    .macroDefines = {defines, static_cast<uint32_t>(std::size(defines))},
                    .descriptorHeapMode = render::SlangDescriptorHeapMode::Native,
                }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
                if (!result || compileResult.spirv.empty()) {
                    return RHITestResult::fail(
                        std::string("Path tracing guide shader compile failed for ") +
                        entry.moduleName + "." + entry.entryPointName + ": " +
                        toString(result) + " " + compileResult.diagnostics);
                }
                if (!hasNativeComputeResourceInterface(compileResult.spirv)) {
                    return RHITestResult::fail(std::string(entry.moduleName) + " must use native compute resources in both position-fetch variants");
                }
                if (spirvContainsOpcode(compileResult.spirv,
                        kSPIRVOpRayQueryGetIntersectionTriangleVertexPositionsKhr) != (positionFetch != 0) ||
                    spirvContainsCapability(compileResult.spirv,
                        kSPIRVRayQueryPositionFetchKhr) != (positionFetch != 0) ||
                    spirvContainsExtension(compileResult.spirv,
                        "SPV_KHR_ray_tracing_position_fetch") != (positionFetch != 0)) {
                    return RHITestResult::fail("guide shader position-fetch instruction/capability mismatch");
                }
            }
        }

        return RHITestResult::pass("compiled Standard/OpenPBR guides, material visualization and RTXDI with position fetch enabled and disabled");
    }
};

class RenderGraphStreamlineDLSSSupportShaderCompileTest : public RHITest {
public:
    RenderGraphStreamlineDLSSSupportShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_streamline_dlss_support_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        const char* entryPoints[] = {
            "streamlineDlssDepthVertexMain",
            "streamlineDlssDepthFragmentMain",
            "streamlineDlssAlphaMain",
        };
        for (const char* entryPoint : entryPoints) {
            render::ShaderCompileResult compileResult;
            render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
                    .moduleName = "Features/PostProcess/StreamlineDLSSSupport",
                    .entryPointName = entryPoint,
                    .searchPath = kShaderSearchPath,
                }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
            if (!result) {
                return RHITestResult::fail(
                    std::string("Streamline DLSS support shader compile returned ") +
                    toString(result) +
                    " for " +
                    entryPoint +
                    ": " +
                    compileResult.diagnostics);
            }
            if (compileResult.spirv.empty()) {
                return RHITestResult::fail(
                    std::string("Streamline DLSS support shader produced empty SPIR-V for ") +
                    entryPoint);
            }
        }
        return RHITestResult::pass("compiled DLSS depth export and alpha resolve shaders");
    }
};

class RenderGraphSceneRayQueryClusterShaderCompileTest : public RHITest {
public:
    RenderGraphSceneRayQueryClusterShaderCompileTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_scene_rayquery_cluster_shader_compile";
    }

    RHITestResult run(RHITestContext&) override
    {
        const char* capabilities[] = {
            "spvRayQueryKHR",
            "SPV_NV_cluster_acceleration_structure",
            "spvRayTracingClusterAccelerationStructureNV",
        };
        const render::SlangMacroDefine macros[] = {
            render::SlangMacroDefine{
                .name = "SCENE_RAYQUERY_ENABLE_CLUSTER_ID",
                .value = "1",
            },
        };
        render::ShaderCompileResult compileResult;
        render::Result<> result = render::compileSlangShaderToSpirv(render::SlangShaderDesc{
            .moduleName = "Features/Debug/SceneRayQueryVisualize",
            .entryPointName = "sceneRayQueryVisualizeMain",
            .searchPath = kShaderSearchPath,
            .capabilities = {capabilities, static_cast<uint32_t>(std::size(capabilities))},
            .macroDefines = {macros, static_cast<uint32_t>(std::size(macros))},
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result) {
            return RHITestResult::fail(
                std::string("Cluster ray-query shader compile returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        if (compileResult.spirv.size() < 5 ||
            compileResult.spirv[0] != kSPIRVMagic ||
            compileResult.spirv[1] != kSPIRVVersion16) {
            return RHITestResult::fail("Cluster ray-query shader did not produce a SPIR-V 1.6 module");
        }
        if (!spirvContainsCapability(
                compileResult.spirv,
                kSPIRVRayTracingClusterAccelerationStructureNv)) {
            return RHITestResult::fail(
                "Cluster ray-query shader omitted RayTracingClusterAccelerationStructureNV");
        }
        if (!spirvContainsExtension(
                compileResult.spirv,
                "SPV_NV_cluster_acceleration_structure")) {
            return RHITestResult::fail(
                "Cluster ray-query shader omitted SPV_NV_cluster_acceleration_structure");
        }
        if (!spirvContainsOpcode(
                compileResult.spirv,
                kSPIRVOpRayQueryGetIntersectionClusterIdNv)) {
            return RHITestResult::fail(
                "Cluster ray-query shader omitted OpRayQueryGetIntersectionClusterIdNV");
        }
        return RHITestResult::pass("compiled SPIR-V 1.6 cluster ray-query shader");
    }
};

class RenderGraphOpenPBRPathTracingSamplePreviewTest : public RHITest {
public:
    RenderGraphOpenPBRPathTracingSamplePreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_openpbr_pathtracing_sample_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", sample, message)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr) {
            return RHITestResult::fail("OpenPBR PathTracingSample is missing PathTrace node");
        }
        // User-reported close view whose glass sphere exposes a horizontal band.
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "maxDepth", 12) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "samples", 2) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", false) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.eye", {-0.008599f, 0.073623f, 0.058931f}) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.center", {-1.384997f, -0.182991f, 2.025709f})) {
            return RHITestResult::fail("failed to set OpenPBR PathTracingSample preview runtime properties");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("OpenPBR PathTracingSample is unsupported on this device: ") +
                    preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("OpenPBR PathTracingSample render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        if (preview.lastLog().find("environment map does not exist") != std::string::npos ||
            preview.lastLog().find("failed to decode environment map") != std::string::npos ||
            preview.lastLog().find("decoded environment map is too large") != std::string::npos) {
            return RHITestResult::fail(
                std::string("OpenPBR PathTracingSample did not load the HDRI environment: ") +
                preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 64) {
            return RHITestResult::fail(
                std::string("OpenPBR PathTracingSample produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_openpbr_pathtracing_sample_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RHITestResult::fail(message);
        }

        return RHITestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphOpenPBRPathTracingDebugViewsTest : public RHITest {
public:
    RenderGraphOpenPBRPathTracingDebugViewsTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_openpbr_pathtracing_debug_views";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", sample, message)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr) {
            return RHITestResult::fail("OpenPBR PathTracingSample is missing PathTrace node");
        }
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "maxDepth", 12) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "samples", 2) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", false) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.eye", {-0.001590f, 0.072671f, 0.069807f}) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.center", {-2.046089f, 0.350581f, 1.323329f})) {
            return RHITestResult::fail("failed to set OpenPBR debug-view camera properties");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }
        // Debug views are display-referred. Histogram auto-exposure remeters
        // each view and makes otherwise-identical diagnostics incomparable.
        render::RenderGraphNode* autoExposure = sample.graph.findNode("AutoExposure");
        if (autoExposure == nullptr || !sample.graph.removeNode(autoExposure->id)) {
            return RHITestResult::fail("failed to remove AutoExposure from OpenPBR debug views");
        }
        if (sample.graph.addEdge("PathTrace.color", "FinalBlit.source") == nullptr) {
            return RHITestResult::fail("failed to connect PathTrace.color to FinalBlit for OpenPBR debug views");
        }

        struct DebugCase {
            const char* name;
            const char* view;
            const char* enabledFlag;
        };
        const std::array debugFlags{
            "debugDisableNormalMap",
            "debugForceGeometryNormal",
            "debugDisableMaterialTextures",
            "debugDisableDirectLighting",
            "debugUseOpaqueShadows",
            "debugDisableShadows",
            "debugDisableVolumeAttenuation",
            "debugDisableTransmission",
        };
        const std::array cases{
            DebugCase{"final", "final", nullptr},
            DebugCase{"geometry_normal", "geometryNormal", nullptr},
            DebugCase{"shading_normal", "shadingNormal", nullptr},
            DebugCase{"mapped_normal", "mappedNormal", nullptr},
            DebugCase{"tangent", "tangent", nullptr},
            DebugCase{"bitangent", "bitangent", nullptr},
            DebugCase{"tangent_handedness", "tangentHandedness", nullptr},
            DebugCase{"texcoord", "texcoord", nullptr},
            DebugCase{"front_face", "frontFace", nullptr},
            DebugCase{"shading_side", "shadingSide", nullptr},
            DebugCase{"triangle", "triangle", nullptr},
            DebugCase{"base_color", "baseColor", nullptr},
            DebugCase{"normal_texture", "normalTexture", nullptr},
            DebugCase{"shadow_transmittance", "shadowTransmittance", nullptr},
            DebugCase{"mapped_no_normal_map", "mappedNormal", "debugDisableNormalMap"},
            DebugCase{"mapped_force_geometry", "mappedNormal", "debugForceGeometryNormal"},
            DebugCase{"final_no_material_textures", "final", "debugDisableMaterialTextures"},
            DebugCase{"final_no_direct_lighting", "final", "debugDisableDirectLighting"},
            DebugCase{"final_opaque_shadows", "final", "debugUseOpaqueShadows"},
            DebugCase{"final_unoccluded", "final", "debugDisableShadows"},
            DebugCase{"final_no_volume_attenuation", "final", "debugDisableVolumeAttenuation"},
            DebugCase{"final_no_transmission", "final", "debugDisableTransmission"},
            DebugCase{"shadow_opaque", "shadowTransmittance", "debugUseOpaqueShadows"},
            DebugCase{"shadow_unoccluded", "shadowTransmittance", "debugDisableShadows"},
        };

        std::vector<uint32_t> geometryNormalPixels;
        std::vector<uint32_t> shadingNormalPixels;
        std::vector<uint32_t> frontFacePixels;
        std::vector<uint32_t> mappedNoNormalMapPixels;
        std::vector<uint32_t> mappedForceGeometryPixels;
        std::vector<uint32_t> shadowTransmittancePixels;
        std::vector<uint32_t> shadowOpaquePixels;
        std::vector<uint32_t> shadowUnoccludedPixels;
        sample.graph.clearDirty();
        for (const DebugCase& debugCase : cases) {
            for (const char* flag : debugFlags) {
                if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, flag, false)) {
                    return RHITestResult::fail(std::string("failed to clear OpenPBR debug flag ") + flag);
                }
            }
            if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "debugView", debugCase.view) ||
                (debugCase.enabledFlag != nullptr &&
                 !sample.graph.setNodeRuntimeProperty(pathTrace->id, debugCase.enabledFlag, true))) {
                return RHITestResult::fail(std::string("failed to set OpenPBR debug case ") + debugCase.name);
            }
            if (sample.graph.dirty()) {
                return RHITestResult::fail(std::string("OpenPBR debug case dirtied graph: ") + debugCase.name);
            }

            result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
            if (!result) {
                if (render::hasError(result, render::Error::Unsupported)) {
                    return RHITestResult::skip(
                        std::string("OpenPBR debug views are unsupported on this device: ") +
                        preview.lastLog());
                }
                return RHITestResult::fail(
                    std::string("OpenPBR debug case render returned ") +
                    toString(result) +
                    " for " +
                    debugCase.name +
                    ": " +
                    preview.lastLog());
            }
            if (countVisiblePixels(preview.pixels()) < 512) {
                return RHITestResult::fail(
                    std::string("OpenPBR debug case produced too few visible pixels: ") +
                    debugCase.name);
            }

            const std::string caseName(debugCase.name);
            if (caseName == "geometry_normal") {
                geometryNormalPixels = preview.pixels();
            }
            if (caseName == "shading_normal") {
                shadingNormalPixels = preview.pixels();
            }
            if (caseName == "front_face") {
                frontFacePixels = preview.pixels();
            }
            if (caseName == "mapped_no_normal_map") {
                mappedNoNormalMapPixels = preview.pixels();
            }
            if (caseName == "mapped_force_geometry") {
                mappedForceGeometryPixels = preview.pixels();
            }
            if (caseName == "shadow_transmittance") {
                shadowTransmittancePixels = preview.pixels();
            }
            if (caseName == "shadow_opaque") {
                shadowOpaquePixels = preview.pixels();
            }
            if (caseName == "shadow_unoccluded") {
                shadowUnoccludedPixels = preview.pixels();
            }

            const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
            const std::filesystem::path outputPath =
                context.outputDirectory /
                (std::string("render_graph_openpbr_debug_") + debugCase.name + ".png");
            if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
                return RHITestResult::fail(message);
            }
        }

        for (const char* flag : debugFlags) {
            if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, flag, false)) {
                return RHITestResult::fail(std::string("failed to clear OpenPBR stability flag ") + flag);
            }
        }
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", true) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "debugView", "shadowTransmittance")) {
            return RHITestResult::fail("failed to configure accumulated OpenPBR debug stability check");
        }
        result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
        if (!result) {
            return RHITestResult::fail(
                std::string("first accumulated OpenPBR debug render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const std::vector<uint32_t> firstStableDebugPixels = preview.pixels();
        result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
        if (!result) {
            return RHITestResult::fail(
                std::string("second accumulated OpenPBR debug render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint64_t accumulatedDebugDifference =
            sumAbsoluteRgbDifference(firstStableDebugPixels, preview.pixels());
        if (accumulatedDebugDifference != 0) {
            return RHITestResult::fail(
                "OpenPBR debug view changed while accumulation was enabled: difference=" +
                std::to_string(accumulatedDebugDifference));
        }

        constexpr uint32_t kDebugWidth = 576;
        constexpr uint32_t kDebugHeight = 300;
        const size_t debugPixelCount = static_cast<size_t>(kDebugWidth) * kDebugHeight;
        if (geometryNormalPixels.size() != debugPixelCount ||
            shadingNormalPixels.size() != debugPixelCount ||
            frontFacePixels.size() != debugPixelCount ||
            mappedNoNormalMapPixels.size() != debugPixelCount ||
            mappedForceGeometryPixels.size() != debugPixelCount ||
            shadowTransmittancePixels.size() != debugPixelCount ||
            shadowOpaquePixels.size() != debugPixelCount ||
            shadowUnoccludedPixels.size() != debugPixelCount ||
            firstStableDebugPixels.size() != debugPixelCount ||
            preview.pixels().size() != debugPixelCount) {
            return RHITestResult::fail("OpenPBR captured debug views have unexpected dimensions");
        }

        constexpr uint64_t kNormalDebugDifferenceTolerance = 1024;
        const uint64_t normalBypassDifference =
            sumAbsoluteRgbDifference(shadingNormalPixels, mappedNoNormalMapPixels);
        if (normalBypassDifference > kNormalDebugDifferenceTolerance) {
            return RHITestResult::fail(
                "OpenPBR disable-normal-map view did not match shading normals: difference=" +
                std::to_string(normalBypassDifference));
        }
        const uint64_t geometryOverrideDifference =
            sumAbsoluteRgbDifference(geometryNormalPixels, mappedForceGeometryPixels);
        if (geometryOverrideDifference > kNormalDebugDifferenceTolerance) {
            return RHITestResult::fail(
                "OpenPBR force-geometry-normal view did not match geometry normals: difference=" +
                std::to_string(geometryOverrideDifference));
        }
        if (sumAbsoluteRgbDifference(shadowTransmittancePixels, shadowOpaquePixels) < 1024 ||
            sumAbsoluteRgbDifference(shadowTransmittancePixels, shadowUnoccludedPixels) < 1024) {
            return RHITestResult::fail("OpenPBR shadow debug modes did not produce distinct visibility results");
        }

        std::unordered_set<uint32_t> geometryNormalBins;
        uint32_t surfacePixelCount = 0;
        uint32_t backFacePixelCount = 0;
        for (uint32_t y = 120; y < 240; ++y) {
            for (uint32_t x = 170; x < 300; ++x) {
                const size_t pixelIndex = static_cast<size_t>(y) * kDebugWidth + x;
                const uint32_t frontFacePixel = frontFacePixels[pixelIndex];
                const uint32_t frontFaceR = frontFacePixel & 0xffu;
                const uint32_t frontFaceG = (frontFacePixel >> 8u) & 0xffu;
                const uint32_t frontFaceB = (frontFacePixel >> 16u) & 0xffu;
                const bool isFrontFace = frontFaceG > 200u && frontFaceR < 64u && frontFaceB < 80u;
                const bool isBackFace = frontFaceR > 200u && frontFaceG < 64u && frontFaceB < 80u;
                if (!isFrontFace && !isBackFace) {
                    continue;
                }

                ++surfacePixelCount;
                const uint32_t geometryPixel = geometryNormalPixels[pixelIndex];
                const uint32_t r = geometryPixel & 0xffu;
                const uint32_t g = (geometryPixel >> 8u) & 0xffu;
                const uint32_t b = (geometryPixel >> 16u) & 0xffu;
                geometryNormalBins.insert((r >> 4u) | ((g >> 4u) << 4u) | ((b >> 4u) << 8u));

                if (isBackFace) {
                    ++backFacePixelCount;
                }
            }
        }
        if (surfacePixelCount < 1024) {
            return RHITestResult::fail(
                "OpenPBR primary glass sphere debug ROI contains too few surface pixels: pixels=" +
                std::to_string(surfacePixelCount));
        }
        if (geometryNormalBins.size() < 8) {
            return RHITestResult::fail(
                "OpenPBR geometry normals collapsed across the glass sphere: bins=" +
                std::to_string(geometryNormalBins.size()));
        }
        if (backFacePixelCount != 0) {
            return RHITestResult::fail(
                "OpenPBR primary glass sphere contains false back faces: pixels=" +
                std::to_string(backFacePixelCount));
        }

        return RHITestResult::pass("wrote OpenPBR path-tracing debug views");
    }
};

class RenderGraphOpenPBRPathTracingEnvironmentRotationTest : public RHITest {
public:
    RenderGraphOpenPBRPathTracingEnvironmentRotationTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_openpbr_pathtracing_environment_rotation";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", sample, message)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr) {
            return RHITestResult::fail("OpenPBR PathTracingSample is missing PathTrace node");
        }
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "maxDepth", 4) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "samples", 1) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", false)) {
            return RHITestResult::fail("failed to set OpenPBR environment rotation test runtime properties");
        }

        render::RenderGraphPreviewRenderer preview;
        render::EnvironmentSettings environment = sampleEnvironmentSettings(sample.desc);
        environment.rotationDegrees = 0.0f;
        preview.setEnvironment(environment);
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        result = preview.render(sample.graph, 96, 96, sample.desc.previewOutput);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("OpenPBR PathTracingSample is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("initial OpenPBR rotation render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const std::vector<uint32_t> rotation0Pixels = preview.pixels();

        environment.rotationDegrees = 90.0f;
        preview.setEnvironment(environment);
        result = preview.render(sample.graph, 96, 96, sample.desc.previewOutput);
        if (!result) {
            return RHITestResult::fail(
                std::string("rotated OpenPBR environment render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint64_t difference = sumAbsoluteRgbDifference(rotation0Pixels, preview.pixels());
        if (difference < 4096) {
            return RHITestResult::fail(
                std::string("environment rotation did not materially affect path tracing output, diff=") +
                std::to_string(difference));
        }

        return RHITestResult::pass(std::string("environment rotation diff=") + std::to_string(difference));
    }
};

class RenderGraphScenePathTraceMaterialTexturesPreviewTest : public RHITest {
public:
    RenderGraphScenePathTraceMaterialTexturesPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_scene_path_trace_material_textures_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraphProperties properties{
            {"path", PROJECT_SOURCE_DIR "/Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"},
            {"maxDepth", 2},
            {"samples", 1},
            {"accumulate", false},
        };
        render::RenderGraph graph;
        graph.setName("ScenePathTraceMaterialTexturesPreview");
        graph.addNode("ScenePathTracePass", "PathTrace", properties);
        graph.markOutput("PathTrace.color");

        result = preview.render(graph, 128, 128);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("ScenePathTracePass textured material render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("ScenePathTracePass textured material preview produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string outputMessage;
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_material_textures_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RHITestResult::fail(outputMessage);
        }

        return RHITestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphScenePathTraceTransmissionTexturesPreviewTest : public RHITest {
public:
    RenderGraphScenePathTraceTransmissionTexturesPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_scene_path_trace_transmission_textures_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::filesystem::path scenePath;
        std::string message;
        if (!writeTransmissionTextureScene(context.outputDirectory / "transmission-texture-scene", scenePath, message)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraphProperties properties{
            {"path", scenePath.string()},
            {"maxDepth", 1},
            {"samples", 1},
            {"accumulate", false},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 45.0f},
                {"znear", 0.001f},
                {"zfar", 10.0f},
                {"eye", {0.0f, 0.0f, 2.0f}},
                {"center", {0.0f, 0.0f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
        render::RenderGraph graph;
        graph.setName("ScenePathTraceTransmissionTexturesPreview");
        graph.addNode("ScenePathTracePass", "PathTrace", properties);
        graph.markOutput("PathTrace.color");

        result = preview.render(graph, 96, 96);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("ScenePathTracePass transmission texture render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        uint32_t redPixelCount = 0;
        for (uint32_t pixel : preview.pixels()) {
            const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
            const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
            const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
            if (r > 48 && r > g + 24 && r > b + 24) {
                ++redPixelCount;
            }
        }
        if (redPixelCount < 1024) {
            return RHITestResult::fail(
                std::string("transmission texture preview expected visible red diffuse pixels, got red=") +
                std::to_string(redPixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_transmission_textures_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RHITestResult::fail(message);
        }

        return RHITestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphScenePathTraceAlphaMaskPreviewTest : public RHITest {
public:
    RenderGraphScenePathTraceAlphaMaskPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_scene_path_trace_alpha_mask_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::filesystem::path scenePath;
        std::string message;
        if (!writeAlphaMaskScene(context.outputDirectory / "alpha-mask-scene", scenePath, message)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false, true);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraphProperties properties{
            {"path", scenePath.string()},
            {"maxDepth", 1},
            {"samples", 1},
            {"accumulate", false},
            {"camera", {
                {"projection", "perspective"},
                {"fovDegrees", 45.0f},
                {"znear", 0.001f},
                {"zfar", 10.0f},
                {"eye", {0.0f, 0.0f, 2.0f}},
                {"center", {0.0f, 0.0f, 0.0f}},
                {"up", {0.0f, 1.0f, 0.0f}},
            }},
        };
        render::RenderGraph graph;
        graph.setName("ScenePathTraceAlphaMaskPreview");
        graph.addNode("ScenePathTracePass", "PathTrace", properties);
        graph.markOutput("PathTrace.color");

        result = preview.render(graph, 96, 96);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("ScenePathTracePass alpha mask render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        uint32_t redPixelCount = 0;
        uint32_t bluePixelCount = 0;
        for (uint32_t pixel : preview.pixels()) {
            const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
            const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
            const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
            if (r > 24 && r > g + 24 && r > b + 24) {
                ++redPixelCount;
            }
            if (b > 24 && b > r + 24 && b > g + 24) {
                ++bluePixelCount;
            }
        }
        if (redPixelCount < 1024 || bluePixelCount < 1024) {
            return RHITestResult::fail(
                std::string("alpha mask preview expected red masked pixels and blue revealed pixels, got red=") +
                std::to_string(redPixelCount) +
                " blue=" +
                std::to_string(bluePixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_alpha_mask_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RHITestResult::fail(message);
        }

        return RHITestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphSceneSwitchRetirementTest : public RHITest {
public:
    RenderGraphSceneSwitchRetirementTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_scene_switch_retirement";
    }
    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();
        render::RenderGraph graph;
        auto* node = graph.addNode("TestRetainedScenePass", "Scene");
        graph.markOutput("Scene.color");
        {
            render::RenderGraphExecutor executor;
            for (uint32_t scene = 0; scene < 3; ++scene) {
                graph.setNodeProperties(node->id, {{"scene", scene}});
                std::string log;
                auto result = executor.compile(context.device, graph, 32, 32, log);
                if (!result) { return RHITestResult::fail(log); }
                // Leave both queued slots populated. Compile must wait and release
                // these owners before the next scene's compile allocates anything.
                for (uint32_t frame = 0; frame < 2; ++frame) {
                    result = executor.execute(render::RenderGraphSubmitDesc{
                        .graphicsQueue = context.device.getQueue(render::QueueType::Graphics),
                    });
                    if (!result) { return RHITestResult::fail(toString(result)); }
                }
            }
        }
        if (!testRetainedSceneBuffer.expired()) {
            return RHITestResult::fail("Submitted scene owner survived executor destruction");
        }
        return RHITestResult::pass();
    }
};

class RenderGraphResizeReusesCompiledPassesTest : public RHITest {
public:
    RenderGraphResizeReusesCompiledPassesTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_resize_reuses_compiled_passes";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        const auto initialBudget = context.device.memoryBudget();
        struct RestoreBudget {
            render::Device& device;
            render::MemoryBudgetPolicy policy;
            ~RestoreBudget() { device.setMemoryBudgetPolicy(policy); }
        } restoreBudget{context.device, initialBudget.policy};
        auto policy = initialBudget.policy;
        policy.enabled = true;
        policy.safetyBytes = 8ull * 1024 * 1024;
        policy.graphReserveBytes = 8ull * 1024 * 1024;
        context.device.setMemoryBudgetPolicy(policy);

        render::RenderGraph graph;
        graph.setName("ResizeReuse");
        render::RenderGraphNode* node = graph.addNode("TestResizeCompilePass", "Resize");
        if (node == nullptr) {
            return RHITestResult::fail("failed to add resize test pass node");
        }
        graph.markOutput("Resize.color");

        uint32_t& compileCount = testResizeCompileCount();
        compileCount = 0;

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result<> result = executor.compile(context.device, graph, 64, 48, log);
        if (!result) {
            return RHITestResult::fail(std::string("initial RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }
        if (compileCount != 1) {
            return RHITestResult::fail(
                std::string("expected one pass compile after initial compile, got ") +
                std::to_string(compileCount));
        }

        const render::RenderGraphResource* output = executor.outputResource("Resize.color");
        if (output == nullptr || output->desc.width != 64 || output->desc.height != 48) {
            return RHITestResult::fail("initial resize test output dimensions are invalid");
        }

        result = executor.compile(context.device, graph, 128, 96, log);
        if (!result) {
            return RHITestResult::fail(std::string("resize RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }
        if (compileCount != 1) {
            return RHITestResult::fail(
                std::string("resize recompiled pass PSO path; compile count is ") +
                std::to_string(compileCount));
        }

        output = executor.outputResource("Resize.color");
        if (output == nullptr || output->desc.width != 128 || output->desc.height != 96) {
            return RHITestResult::fail("resized graph output dimensions were not rebuilt");
        }

        render::RenderGraphProperties properties = render::RenderGraphProperties::object();
        properties["variant"] = 1;
        if (!graph.setNodeProperties(node->id, std::move(properties))) {
            return RHITestResult::fail("failed to update resize test pass static properties");
        }

        result = executor.compile(context.device, graph, 128, 96, log);
        if (!result) {
            return RHITestResult::fail(std::string("static property RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }
        if (compileCount != 2) {
            return RHITestResult::fail(
                std::string("static property change did not force full pass compile; compile count is ") +
                std::to_string(compileCount));
        }

        if (context.device.memoryBudget().reservedBytes != initialBudget.reservedBytes) {
            return RHITestResult::fail("Compile/resize leaked graph budget reservations");
        }
        policy.graphReserveBytes = UINT64_MAX;
        context.device.setMemoryBudgetPolicy(policy);
        result = executor.compile(context.device, graph, 128, 96, log);
        if (!render::hasError(result, render::Error::OutOfMemory) ||
                context.device.memoryBudget().reservedBytes != initialBudget.reservedBytes) {
            return RHITestResult::fail("Failed graph budget preflight did not release reservations");
        }
        policy.graphReserveBytes = 8ull * 1024 * 1024;
        context.device.setMemoryBudgetPolicy(policy);
        result = executor.compile(context.device, graph, 128, 96, log);
        if (!result || context.device.memoryBudget().reservedBytes != initialBudget.reservedBytes) {
            return RHITestResult::fail("Graph could not recover after budget preflight failure: " + log);
        }

        return RHITestResult::pass();
    }
};

class RenderGraphPreviewActualOutputExtentTest : public RHITest {
public:
    RenderGraphPreviewActualOutputExtentTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_preview_actual_output_extent";
    }

    RHITestResult run(RHITestContext&) override
    {
        registerTestPass();

        constexpr uint32_t kRequestedWidth = 320;
        constexpr uint32_t kRequestedHeight = 180;
        constexpr uint32_t kOutputWidth = 80;
        constexpr uint32_t kOutputHeight = 45;

        render::RenderGraphProperties properties = render::RenderGraphProperties::object();
        properties["outputWidth"] = kOutputWidth;
        properties["outputHeight"] = kOutputHeight;

        render::RenderGraph graph;
        graph.setName("PreviewActualOutputExtent");
        render::RenderGraphNode* producer = graph.addNode(
            "TestTextureExtentProducerPass",
            "Producer",
            properties);
        if (producer == nullptr ||
            !graph.markOutput("Producer.color")) {
            return RHITestResult::fail("failed to construct actual-output-extent preview graph");
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false);
        if (!result) {
            return RHITestResult::skip(
                std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        result = preview.render(
            graph,
            kRequestedWidth,
            kRequestedHeight,
            "Producer.color");
        if (!result) {
            return RHITestResult::fail(
                std::string("actual-output-extent preview render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        if (preview.width() != kOutputWidth || preview.height() != kOutputHeight) {
            return RHITestResult::fail(
                std::string("preview reported ") +
                std::to_string(preview.width()) +
                "x" +
                std::to_string(preview.height()) +
                " instead of the actual 80x45 output extent");
        }

        constexpr size_t kExpectedPixelCount =
            static_cast<size_t>(kOutputWidth) * static_cast<size_t>(kOutputHeight);
        if (preview.pixels().size() != kExpectedPixelCount) {
            return RHITestResult::fail(
                std::string("preview pixel count was ") +
                std::to_string(preview.pixels().size()) +
                " instead of " +
                std::to_string(kExpectedPixelCount));
        }

        properties["outputRgba16"] = true;
        if (!graph.setNodeProperties(producer->id, std::move(properties))) {
            return RHITestResult::fail("failed to switch preview output to RGBA16F");
        }
        result = preview.render(
            graph,
            kRequestedWidth,
            kRequestedHeight,
            "Producer.color");
        if (!result) {
            return RHITestResult::fail(
                std::string("RGBA16F actual-output-extent preview returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        if (preview.width() != kOutputWidth ||
            preview.height() != kOutputHeight ||
            preview.pixels().size() != kExpectedPixelCount) {
            return RHITestResult::fail("RGBA16F preview did not preserve the actual output extent");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphTextureExtentConstraintPropagationTest : public RHITest {
public:
    RenderGraphTextureExtentConstraintPropagationTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_texture_extent_constraint_propagation";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        constexpr uint32_t kGraphWidth = 320;
        constexpr uint32_t kGraphHeight = 180;
        constexpr uint32_t kProducerWidth = 80;
        constexpr uint32_t kProducerHeight = 45;
        constexpr uint32_t kUpdatedProducerWidth = 64;
        constexpr uint32_t kUpdatedProducerHeight = 36;

        render::RenderGraphProperties consumerProperties = render::RenderGraphProperties::object();
        consumerProperties["inputWidth"] = kProducerWidth;
        consumerProperties["inputHeight"] = kProducerHeight;

        render::RenderGraph graph;
        graph.setName("TextureExtentConstraintPropagation");
        render::RenderGraphNode* producer = graph.addNode("TestTextureExtentProducerPass", "Producer");
        render::RenderGraphNode* consumer = graph.addNode("TestTextureExtentConsumerPass", "Consumer");
        if (producer == nullptr ||
            consumer == nullptr ||
            !graph.addEdge("Producer.color", "Consumer.input") ||
            !graph.markOutput("Consumer.color")) {
            return RHITestResult::fail("failed to construct texture extent propagation graph");
        }
        if (!graph.setNodeRuntimeProperties(consumer->id, consumerProperties)) {
            return RHITestResult::fail("failed to set runtime texture extent constraints");
        }

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result<> result = executor.compile(
            context.device,
            graph,
            kGraphWidth,
            kGraphHeight,
            log);
        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::compile returned ") +
                toString(result) + ": " + log);
        }

        const render::RenderGraphResource* producerOutput = executor.outputResource("Producer.color");
        const render::RenderGraphResource* consumerOutput = executor.outputResource("Consumer.color");
        if (producerOutput == nullptr ||
            producerOutput->desc.width != kProducerWidth ||
            producerOutput->desc.height != kProducerHeight) {
            return RHITestResult::fail(
                "explicit consumer input extent did not propagate to the producer resource");
        }
        if (consumerOutput == nullptr ||
            consumerOutput->desc.width != kGraphWidth ||
            consumerOutput->desc.height != kGraphHeight) {
            return RHITestResult::fail(
                "default consumer output did not preserve the global graph extent");
        }

        testTextureExtentExecutionState() = {};
        result = executor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = &context.graphicsQueue,
        });
        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }
        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::waitForSubmittedWork returned ") +
                toString(result));
        }

        const TestTextureExtentExecutionState& state = testTextureExtentExecutionState();
        if (state.producerContextWidth != kProducerWidth ||
            state.producerContextHeight != kProducerHeight ||
            state.producerOutputWidth != kProducerWidth ||
            state.producerOutputHeight != kProducerHeight) {
            return RHITestResult::fail(
                "producer execution context did not use the propagated texture extent");
        }
        if (state.consumerContextWidth != kGraphWidth ||
            state.consumerContextHeight != kGraphHeight ||
            state.consumerInputWidth != kProducerWidth ||
            state.consumerInputHeight != kProducerHeight ||
            state.consumerOutputWidth != kGraphWidth ||
            state.consumerOutputHeight != kGraphHeight) {
            return RHITestResult::fail(
                "consumer execution context or resources did not preserve split input/output extents");
        }

        consumerProperties["inputWidth"] = kUpdatedProducerWidth;
        consumerProperties["inputHeight"] = kUpdatedProducerHeight;
        if (!graph.setNodeRuntimeProperties(consumer->id, std::move(consumerProperties))) {
            return RHITestResult::fail("failed to update runtime texture extent constraints");
        }
        result = executor.compile(context.device, graph, kGraphWidth, kGraphHeight, log);
        if (!result) {
            return RHITestResult::fail(
                std::string("runtime extent RenderGraphExecutor::compile returned ") +
                toString(result) + ": " + log);
        }
        producerOutput = executor.outputResource("Producer.color");
        if (producerOutput == nullptr ||
            producerOutput->desc.width != kUpdatedProducerWidth ||
            producerOutput->desc.height != kUpdatedProducerHeight) {
            return RHITestResult::fail(
                "updated runtime input extent did not rebuild the producer resource");
        }

        testTextureExtentExecutionState() = {};
        result = executor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = &context.graphicsQueue,
        });
        if (!result) {
            return RHITestResult::fail(
                std::string("updated extent RenderGraphExecutor::execute returned ") + toString(result));
        }
        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(
                std::string("updated extent waitForSubmittedWork returned ") + toString(result));
        }
        const TestTextureExtentExecutionState& updatedState = testTextureExtentExecutionState();
        if (updatedState.producerContextWidth != kUpdatedProducerWidth ||
            updatedState.producerContextHeight != kUpdatedProducerHeight ||
            updatedState.consumerContextWidth != kGraphWidth ||
            updatedState.consumerContextHeight != kGraphHeight ||
            updatedState.consumerInputWidth != kUpdatedProducerWidth ||
            updatedState.consumerInputHeight != kUpdatedProducerHeight) {
            return RHITestResult::fail(
                "runtime extent rebuild did not update producer/consumer execution dimensions");
        }

        return RHITestResult::pass(
            "propagated and rebuilt runtime input extents while preserving 320x180 output");
    }
};

class RenderGraphTextureExtentConstraintConflictTest : public RHITest {
public:
    RenderGraphTextureExtentConstraintConflictTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_texture_extent_constraint_conflict";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        render::RenderGraphProperties firstConsumerProperties =
            render::RenderGraphProperties::object();
        firstConsumerProperties["inputWidth"] = 80u;
        firstConsumerProperties["inputHeight"] = 45u;
        render::RenderGraphProperties secondConsumerProperties =
            render::RenderGraphProperties::object();
        secondConsumerProperties["inputWidth"] = 64u;
        secondConsumerProperties["inputHeight"] = 36u;

        render::RenderGraph graph;
        graph.setName("TextureExtentConstraintConflict");
        if (graph.addNode("TestTextureExtentProducerPass", "Producer") == nullptr ||
            graph.addNode(
                "TestTextureExtentConsumerPass",
                "FirstConsumer",
                firstConsumerProperties) == nullptr ||
            graph.addNode(
                "TestTextureExtentConsumerPass",
                "SecondConsumer",
                secondConsumerProperties) == nullptr ||
            !graph.addEdge("Producer.color", "FirstConsumer.input") ||
            !graph.addEdge("Producer.color", "SecondConsumer.input") ||
            !graph.markOutput("FirstConsumer.color") ||
            !graph.markOutput("SecondConsumer.color")) {
            return RHITestResult::fail("failed to construct conflicting texture extent graph");
        }

        render::RenderGraphExecutor executor;
        std::string log;
        const render::Result<> result = executor.compile(context.device, graph, 320, 180, log);
        if (result ||
            !render::hasError(result, render::Error::InvalidArgument) ||
            executor.compiled()) {
            return RHITestResult::fail(
                "RenderGraph compile accepted conflicting explicit consumer input extents: " + log);
        }

        return RHITestResult::pass(
            "rejected conflicting 80x45 and 64x36 constraints on one producer output");
    }
};

class RenderGraphTextureExtentConstraintMultihopPropagationTest : public RHITest {
public:
    RenderGraphTextureExtentConstraintMultihopPropagationTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_texture_extent_constraint_multihop_propagation";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        constexpr uint32_t kGraphWidth = 320;
        constexpr uint32_t kGraphHeight = 180;
        constexpr uint32_t kConstrainedWidth = 80;
        constexpr uint32_t kConstrainedHeight = 45;

        render::RenderGraphProperties consumerProperties =
            render::RenderGraphProperties::object();
        consumerProperties["inputWidth"] = kConstrainedWidth;
        consumerProperties["inputHeight"] = kConstrainedHeight;

        render::RenderGraph graph;
        graph.setName("TextureExtentConstraintMultihopPropagation");
        if (graph.addNode("TestTextureExtentProducerPass", "Producer", {{"viewBinding", "local"}}) == nullptr ||
            graph.addNode("TestTextureExtentRelayPass", "Relay") == nullptr ||
            graph.addNode(
                "TestTextureExtentConsumerPass",
                "Consumer",
                consumerProperties) == nullptr ||
            !graph.addEdge("Producer.color", "Relay.input") ||
            !graph.addEdge("Relay.color", "Consumer.input") ||
            !graph.markOutput("Consumer.color")) {
            return RHITestResult::fail("failed to construct multihop texture extent graph");
        }

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result<> result = executor.compile(
            context.device,
            graph,
            kGraphWidth,
            kGraphHeight,
            log);
        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::compile returned ") +
                toString(result) + ": " + log);
        }

        const render::RenderGraphResource* producerOutput =
            executor.outputResource("Producer.color");
        const render::RenderGraphResource* relayOutput = executor.outputResource("Relay.color");
        const render::RenderGraphResource* consumerOutput =
            executor.outputResource("Consumer.color");
        if (producerOutput == nullptr ||
            producerOutput->desc.width != kConstrainedWidth ||
            producerOutput->desc.height != kConstrainedHeight ||
            relayOutput == nullptr ||
            relayOutput->desc.width != kConstrainedWidth ||
            relayOutput->desc.height != kConstrainedHeight) {
            return RHITestResult::fail(
                "explicit downstream input extent did not propagate through the relay resources");
        }
        if (consumerOutput == nullptr ||
            consumerOutput->desc.width != kGraphWidth ||
            consumerOutput->desc.height != kGraphHeight) {
            return RHITestResult::fail(
                "multihop consumer output did not preserve the global graph extent");
        }

        testTextureExtentExecutionState() = {};
        result = executor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = &context.graphicsQueue,
        });
        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }
        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::waitForSubmittedWork returned ") +
                toString(result));
        }

        const TestTextureExtentExecutionState& state = testTextureExtentExecutionState();
        if (state.producerContextWidth != kConstrainedWidth ||
            state.producerContextHeight != kConstrainedHeight ||
            state.producerOutputWidth != kConstrainedWidth ||
            state.producerOutputHeight != kConstrainedHeight) {
            return RHITestResult::fail(
                "multihop producer resource or execution context has the wrong extent");
        }
        if (state.relayContextWidth != kConstrainedWidth ||
            state.relayContextHeight != kConstrainedHeight ||
            state.relayInputWidth != kConstrainedWidth ||
            state.relayInputHeight != kConstrainedHeight ||
            state.relayOutputWidth != kConstrainedWidth ||
            state.relayOutputHeight != kConstrainedHeight) {
            return RHITestResult::fail(
                "relay input, output, or execution context did not inherit the multihop constraint");
        }
        if (state.consumerContextWidth != kGraphWidth ||
            state.consumerContextHeight != kGraphHeight ||
            state.consumerInputWidth != kConstrainedWidth ||
            state.consumerInputHeight != kConstrainedHeight ||
            state.consumerOutputWidth != kGraphWidth ||
            state.consumerOutputHeight != kGraphHeight) {
            return RHITestResult::fail(
                "multihop consumer did not preserve split input/output execution extents");
        }
        if (state.producerDisplayWidth != kGraphWidth || state.producerDisplayHeight != kGraphHeight ||
            state.relayDisplayWidth != kGraphWidth || state.relayDisplayHeight != kGraphHeight ||
            state.consumerDisplayWidth != kGraphWidth || state.consumerDisplayHeight != kGraphHeight) {
            return RHITestResult::fail(
                "Local and shared-view passes did not retain the graph display extent independently of internal texture sizes");
        }

        return RHITestResult::pass(
            "propagated 80x45 through an implicit relay while preserving 320x180 output and display extent across local/shared views");
    }
};

class RenderGraphTextureExtentConstraintRelayConflictTest : public RHITest {
public:
    RenderGraphTextureExtentConstraintRelayConflictTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_texture_extent_constraint_relay_conflict";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        render::RenderGraphProperties producerProperties =
            render::RenderGraphProperties::object();
        producerProperties["outputWidth"] = 80u;
        producerProperties["outputHeight"] = 45u;
        render::RenderGraphProperties relayProperties =
            render::RenderGraphProperties::object();
        relayProperties["outputWidth"] = 320u;
        relayProperties["outputHeight"] = 180u;

        render::RenderGraph graph;
        graph.setName("TextureExtentConstraintRelayConflict");
        if (graph.addNode(
                "TestTextureExtentProducerPass",
                "Producer",
                producerProperties) == nullptr ||
            graph.addNode("TestTextureExtentRelayPass", "Relay", relayProperties) == nullptr ||
            !graph.addEdge("Producer.color", "Relay.input") ||
            !graph.markOutput("Relay.color")) {
            return RHITestResult::fail("failed to construct conflicting relay extent graph");
        }

        render::RenderGraphExecutor executor;
        std::string log;
        const render::Result<> result = executor.compile(context.device, graph, 320, 180, log);
        if (result ||
            !render::hasError(result, render::Error::InvalidArgument) ||
            executor.compiled()) {
            return RHITestResult::fail(
                "RenderGraph compile accepted an implicit relay input that conflicts with its output: " +
                log);
        }

        return RHITestResult::pass(
            "rejected explicit 80x45 producer feeding implicit input of a 320x180 relay");
    }
};

class RenderGraphShaderReloadTransactionTest : public RHITest {
public:
    RenderGraphShaderReloadTransactionTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_shader_reload_transaction";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        TestShaderReloadState& state = testShaderReloadState();
        state = {};
        struct StateGuard {
            TestShaderReloadState& state;

            ~StateGuard()
            {
                state.failCompile = false;
                state.changeReflection = false;
            }
        } stateGuard{state};
        const auto wasDestroyed = [&state](uint32_t instanceId) {
            return std::find(
                       state.destroyedInstanceIds.begin(),
                       state.destroyedInstanceIds.end(),
                       instanceId) != state.destroyedInstanceIds.end();
        };

        render::RenderGraph graph;
        graph.setName("ShaderReloadTransaction");
        graph.addNode("TestShaderReloadPass", "Reload");
        if (!graph.markOutput("Reload.color")) {
            return RHITestResult::fail("failed to mark shader reload test output");
        }

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result<> result = executor.compile(context.device, graph, 32, 24, log);
        if (!result || !executor.compiled()) {
            return RHITestResult::fail(
                std::string("initial shader reload graph compile returned ") +
                toString(result) + ": " + log);
        }
        const render::RenderGraphResource* output = executor.outputResource("Reload.color");
        if (output == nullptr || output->desc.width != 32 || output->desc.height != 24) {
            return RHITestResult::fail("initial shader reload output resource is invalid");
        }
        const uint32_t initialInstanceId = state.lastSuccessfulCompileInstanceId;
        if (initialInstanceId == 0) {
            return RHITestResult::fail("initial shader reload pass did not compile");
        }

        result = executor.reloadShaders(log);
        const uint32_t reloadedInstanceId = state.lastSuccessfulCompileInstanceId;
        if (!result || !executor.compiled() || reloadedInstanceId == initialInstanceId ||
            !wasDestroyed(initialInstanceId)) {
            return RHITestResult::fail(
                std::string("successful shader reload did not transactionally replace the pass: ") +
                toString(result) + ": " + log);
        }
        if (executor.outputResource("Reload.color") != output) {
            return RHITestResult::fail("successful shader reload replaced graph resources");
        }

        state.failCompile = true;
        const uint32_t compileCountBeforeFailure = state.compileCount;
        result = executor.reloadShaders(log);
        if (result || !render::hasError(result, render::Error::Failure) ||
            state.compileCount != compileCountBeforeFailure + 1 ||
            state.lastSuccessfulCompileInstanceId != reloadedInstanceId) {
            return RHITestResult::fail(
                "failed shader reload did not preserve the last successful pass state: " + log);
        }
        if (!executor.compiled() || executor.outputResource("Reload.color") != output ||
            wasDestroyed(reloadedInstanceId)) {
            return RHITestResult::fail(
                "failed shader reload invalidated the compiled graph or its output resource");
        }

        state.failCompile = false;
        state.changeReflection = true;
        const uint32_t compileCountBeforeContractChange = state.compileCount;
        result = executor.reloadShaders(log);
        if (result || !render::hasError(result, render::Error::InvalidArgument) ||
            state.compileCount != compileCountBeforeContractChange ||
            log.find("contract changed") == std::string::npos) {
            return RHITestResult::fail(
                "shader reload did not reject a changed render-graph reflection contract: " + log);
        }
        if (!executor.compiled() || executor.outputResource("Reload.color") != output ||
            state.lastSuccessfulCompileInstanceId != reloadedInstanceId ||
            wasDestroyed(reloadedInstanceId)) {
            return RHITestResult::fail(
                "reflection rejection invalidated the last successful shader pass");
        }

        return RHITestResult::pass(
            "validated successful replacement and last-good preservation across shader reload failures");
    }
};

class RenderGraphCopyColorWorkflowTest : public RHITest {
public:
    RenderGraphCopyColorWorkflowTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_copy_color_workflow";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("CopyColorWorkflow");
        graph.addNode("TriangleRasterPass", "Triangle");
        graph.addNode("CopyColorPass", "Copy");
        graph.addEdge("Triangle.color", "Copy.source");
        graph.markOutput("Copy.color");

        result = preview.render(graph, 128, 96);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result));
        }
        if (countBrightPixels(preview.pixels()) < 128) {
            return RHITestResult::fail("copy color graph produced too few bright pixels");
        }

        graph.markDirty();
        result = preview.render(graph, 80, 80);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphPreviewRenderer::render resize returned ") + toString(result));
        }
        if (preview.width() != 80 || preview.height() != 80) {
            return RHITestResult::fail("copy color graph resize did not update output dimensions");
        }
        if (countBrightPixels(preview.pixels()) < 80) {
            return RHITestResult::fail("resized copy color graph produced too few bright pixels");
        }

        return RHITestResult::pass();
    }
};

class RenderGraphBindlessTextureWorkflowTest : public RHITest {
public:
    RenderGraphBindlessTextureWorkflowTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_bindless_texture_workflow";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();

        constexpr uint32_t kWidth = 16;
        constexpr uint32_t kHeight = 16;
        constexpr uint64_t kReadbackByteSize = static_cast<uint64_t>(kWidth) * kHeight * 4ull;

        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Bindless Texture Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RHITestResult::fail("bindless test device has no graphics queue");
        }

        render::RenderGraphProperties sourceProperties = render::RenderGraphProperties::object();
        sourceProperties["color"] = {0.25f, 0.50f, 0.75f, 1.0f};

        render::RenderGraph graph;
        graph.setName("BindlessTextureWorkflow");
        graph.addNode("ClearColorPass", "Source", sourceProperties);
        graph.addNode("TestBindlessSamplePass", "Sample");
        graph.addEdge("Source.color", "Sample.source");
        graph.markOutput("Sample.color");

        render::RenderGraphExecutor executor;
        std::string log;
        result = executor.compile(*device, graph, kWidth, kHeight, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(log);
            }
            return RHITestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = device->createBuffer(render::BufferDesc{
                .size = kReadbackByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            }).transform([&](auto rhiValue) { readbackBuffer = std::move(rhiValue); });
        if (!result || readbackBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        render::RenderFrameContext frame;
        render::QueueSubmissionTracker submissions;
        result = submissions.initialize(*device, *graphicsQueue);
        if (!result) {
            return RHITestResult::fail(std::string("QueueSubmissionTracker::initialize returned ") + toString(result));
        }
        result = frame.begin(0);
        if (!result) {
            return RHITestResult::fail(std::string("RenderFrameContext::begin returned ") + toString(result));
        }
        result = commandBuffer->begin(frame.submissionContext());
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        result = executor.execute(*commandBuffer);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("Sample.color");
        if (output == nullptr || output->texture == nullptr) {
            return RHITestResult::fail("bindless graph output resource is missing");
        }

        result = executor.transitionOutput(*commandBuffer, "Sample.color", render::ResourceState::TransferSource);
        if (!result) {
            return RHITestResult::fail(std::string("transitionOutput returned ") + toString(result));
        }
        commandBuffer->copyTextureToBuffer(render::TextureBufferCopyDesc{
            .texture = output->texture,
            .buffer = readbackBuffer.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
            .mipLevel = 0,
            .baseLayer = 0,
        });

        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        render::RecordedBatch batch;
        result = batch.seal(frame, commandBuffers);
        if (!result) {
            return RHITestResult::fail(std::string("RecordedBatch::seal returned ") + toString(result));
        }
        render::SubmissionReceipt receipt;
        result = submissions.submitBatch(batch, {}, frame).transform([&](auto value) { receipt = std::move(value); });
        if (!result || !receipt.accepted()) {
            return RHITestResult::fail(std::string("QueueSubmissionTracker::submitBatch returned ") + toString(result));
        }
        result = frame.finishSubmission();
        if (!result) {
            return RHITestResult::fail(std::string("RenderFrameContext::finishSubmission returned ") + toString(result));
        }
        result = frame.wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("RenderFrameContext::wait returned ") + toString(result));
        }

        if (device->capabilities().timestampQueries) {
            std::vector<render::RenderGraphExecutionStats> completedGpuStats;
            result = executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completedGpuStats = std::move(value); });
            if (!result) {
                return RHITestResult::fail(
                    std::string("collectCompletedGpuExecutionStats returned ") + toString(result));
            }
            if (completedGpuStats.size() != 1 ||
                !completedGpuStats[0].gpuTimingAvailable ||
                completedGpuStats[0].nodes.size() != 2 ||
                !std::all_of(
                    completedGpuStats[0].nodes.begin(),
                    completedGpuStats[0].nodes.end(),
                    [](const render::RenderGraphNodeExecutionStat& stat) {
                        return stat.gpuTimingAvailable;
                    })) {
                return RHITestResult::fail("RenderGraph pass GPU timestamps were incomplete");
            }
        }

        readbackBuffer->invalidate();
        void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("readback buffer did not map");
        }

        std::vector<uint8_t> pixels(static_cast<size_t>(kReadbackByteSize));
        std::memcpy(pixels.data(), mapped, pixels.size());
        readbackBuffer->unmap();

        uint32_t matchedPixelCount = 0;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index) {
            const uint8_t r = pixels[index * 4 + 0];
            const uint8_t g = pixels[index * 4 + 1];
            const uint8_t b = pixels[index * 4 + 2];
            const uint8_t a = pixels[index * 4 + 3];
            if (r >= 48 && r <= 80 && g >= 112 && g <= 144 && b >= 176 && b <= 208 && a >= 240) {
                ++matchedPixelCount;
            }
        }

        if (matchedPixelCount < (kWidth * kHeight) / 2) {
            return RHITestResult::fail(
                std::string("bindless graph sampled too few source pixels: ") +
                std::to_string(matchedPixelCount));
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_bindless_texture_workflow.png";
        if (!saveRgba8Png(outputPath, pixels.data(), kWidth, kHeight, outputMessage)) {
            return RHITestResult::fail(outputMessage);
        }

        (void)device->waitIdle();
        return RHITestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphBufferWorkflowTest : public RHITest {
public:
    RenderGraphBufferWorkflowTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_buffer_workflow";
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint64_t kByteSize = 16;
        constexpr std::array<uint32_t, 4> kExpectedWords = {
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };

        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Buffer Workflow Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RHITestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::Queue* computeQueue = device->getQueue(render::QueueType::Compute);
        if (computeQueue == nullptr) {
            return RHITestResult::skip("buffer workflow device has no compute queue");
        }

        render::RenderGraph graph;
        graph.setName("BufferWorkflow");
        graph.addNode("RenderGraphBufferWritePass", "Write");
        graph.addNode("RenderGraphBufferCopyPass", "Copy");
        graph.addEdge("Write.data", "Copy.source");
        graph.markOutput("Copy.data");

        render::RenderGraphExecutor executor;
        std::string log;
        result = executor.compile(*device, graph, 1, 1, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(log);
            }
            return RHITestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        result = executor.execute(render::RenderGraphSubmitDesc{
            .computeQueue = computeQueue,
        });
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }

        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::waitForSubmittedWork returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("Copy.data");
        if (output == nullptr ||
            output->type != render::RenderGraphResourceType::Buffer ||
            output->buffer == nullptr ||
            output->bufferDesc.memoryLocation != render::MemoryLocation::HostReadback ||
            output->bufferDesc.size != kByteSize) {
            return RHITestResult::fail("buffer graph output resource is invalid");
        }

        output->buffer->invalidate();
        void* mapped = output->buffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("buffer graph output did not map");
        }

        std::array<uint32_t, 4> actualWords{};
        std::memcpy(actualWords.data(), mapped, actualWords.size() * sizeof(uint32_t));
        output->buffer->unmap();

        if (actualWords != kExpectedWords) {
            return RHITestResult::fail("buffer graph output bytes did not match expected pattern");
        }

        (void)device->waitIdle();
        return RHITestResult::pass();
    }
};

class RenderGraphMultiQueueSubmitTest : public RHITest {
public:
    RenderGraphMultiQueueSubmitTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_multi_queue_submit";
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint64_t kByteSize = 16;
        constexpr std::array<uint32_t, 4> kExpectedWords = {
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };

        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Multi Queue Submit Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RHITestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        render::Queue* computeQueue = device->getQueue(render::QueueType::Compute);
        if (graphicsQueue == nullptr) {
            return RHITestResult::fail("multi queue submit device has no graphics queue");
        }
        if (computeQueue == nullptr) {
            return RHITestResult::skip("multi queue submit device has no compute queue");
        }

        render::RenderGraph graph;
        graph.setName("MultiQueueSubmit");
        graph.addNode("TriangleRasterPass", "Triangle");
        graph.addNode("RenderGraphBufferWritePass", "Write");
        graph.markOutput("Triangle.color");
        graph.markOutput("Write.data");

        render::RenderGraphExecutor executor;
        std::string log;
        result = executor.compile(*device, graph, 32, 32, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(log);
            }
            return RHITestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        result = executor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = graphicsQueue,
            .computeQueue = computeQueue,
        });
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::execute(RenderGraphSubmitDesc) returned ") + toString(result));
        }

        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::waitForSubmittedWork returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("Write.data");
        if (output == nullptr ||
            output->type != render::RenderGraphResourceType::Buffer ||
            output->buffer == nullptr ||
            output->bufferDesc.memoryLocation != render::MemoryLocation::HostReadback ||
            output->bufferDesc.size != kByteSize) {
            return RHITestResult::fail("multi queue buffer output resource is invalid");
        }

        output->buffer->invalidate();
        void* mapped = output->buffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("multi queue buffer graph output did not map");
        }

        std::array<uint32_t, 4> actualWords{};
        std::memcpy(actualWords.data(), mapped, actualWords.size() * sizeof(uint32_t));
        output->buffer->unmap();

        if (actualWords != kExpectedWords) {
            return RHITestResult::fail("multi queue buffer graph output bytes did not match expected pattern");
        }

        (void)device->waitIdle();
        return RHITestResult::pass();
    }
};

class RenderGraphTextureFeedbackEpilogueTest final : public RHITest {
public:
    RenderGraphTextureFeedbackEpilogueTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_texture_feedback_epilogue";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        if (!context.device.capabilities().bindlessDescriptorHeap) {
            return RHITestResult::skip("Requires --rhi-bindless, --rhi-realtime or --rhi-streamline");
        }
        registerTestPass();
        const auto directory = std::filesystem::absolute(context.outputDirectory / "graph-texture-feedback");
        std::filesystem::create_directories(directory);
        // An uncompressed BC4 mip chain keeps this fixture independent of asset
        // caches. Its coarse tail starts at mip 2; only the later pass asks for 0.
        constexpr uint32_t kMipCount = 7;
        constexpr uint32_t kDfdOffset = 80 + kMipCount * 24;
        const std::array<uint8_t, 12> magic{0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a};
        std::vector<uint8_t> texture(kDfdOffset + 28);
        std::copy(magic.begin(), magic.end(), texture.begin());
        const auto put = [&]<typename T>(size_t offset, T value) {
            std::memcpy(texture.data() + offset, &value, sizeof(value));
        };
        put(12, 139u); put(16, 1u); put(20, 64u); put(24, 64u);
        put(36, 1u); put(40, kMipCount); put(48, kDfdOffset); put(52, 28u); put(kDfdOffset, 28u);
        for (uint32_t mip = 0; mip < kMipCount; ++mip) {
            const uint32_t dimension = std::max(64u >> mip, 1u);
            const uint64_t bytes = uint64_t((dimension + 3) / 4) * ((dimension + 3) / 4) * 8;
            const size_t offset = texture.size();
            put(80 + mip * 24, uint64_t(offset));
            put(88 + mip * 24, bytes); put(96 + mip * 24, bytes);
            texture.resize(offset + bytes);
            for (size_t block = offset; block < texture.size(); block += 8) {
                texture[block] = texture[block + 1] = 77;
            }
        }
        {
            std::ofstream output(directory / "texture.ktx2", std::ios::binary);
            output.write(reinterpret_cast<const char*>(texture.data()), texture.size());
        }
        const auto scenePath = directory / "Scene.gltf";
        {
            std::ofstream output(scenePath);
            output << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
                "buffers":[{"uri":"unused.bin","byteLength":36}],"bufferViews":[{"buffer":0,"byteLength":36}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
                "images":[{"uri":"texture.ktx2"}],"textures":[{"source":0}],
                "materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}]})";
        }

        for (bool externalCommands : {false, true}) {
            testTextureFeedbackState() = {};
            RenderGraph graph;
            const RenderGraphProperties properties{{"path", scenePath.generic_string()}, {"streamAssetOnly", true},
                {"materialTextureStreaming", true}, {"materialTextureMaxDimension", 16},
                {"materialTextureRefineDimension", 64}, {"materialTextureBudgetMiB", 8}, {"wantedMip", 2}};
            graph.addNode("TestTextureFeedbackPass", "Coarse", properties);
            auto fineProperties = properties;
            fineProperties["after"] = true;
            fineProperties["wantedMip"] = 0;
            graph.addNode("TestTextureFeedbackPass", "Fine", fineProperties);
            if (!graph.addEdge("Coarse.token", "Fine.previous") || !graph.markOutput("Fine.token")) {
                return RHITestResult::fail("Could not construct shared-feedback graph");
            }
            RenderGraphExecutor executor;
            std::string log;
            auto result = executor.compile(context.device, graph, 16, 16, log);
            if (!result) { return RHITestResult::fail("Texture feedback graph compile: " + log); }
            if (executor.subsystemHost()->get<GPUSceneSubsystem>() != nullptr) {
                return RHITestResult::fail("Feedback regression must exercise a streamer-only graph");
            }
            QueueSubmissionTracker submissions;
            RenderFrameContext frame;
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> commands;
            if (externalCommands) {
                result = submissions.initialize(context.device, context.graphicsQueue);
                if (result) { result = context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); }); }
                if (result) { result = pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }); }
                if (!result) { return RHITestResult::fail("Texture feedback command resources: " + std::string(resultToString(result))); }
            }
            struct Drain {
                Queue& queue;
                RenderFrameContext& frame;
                ~Drain() { frame.cancel(); (void)queue.waitIdle(); }
            } drain{context.graphicsQueue, frame};
            std::shared_ptr<ScenePathTraceResources> resources;
            bool refined = false;
            for (uint32_t iteration = 0; iteration < 300 && !refined; ++iteration) {
                if (externalCommands) {
                    result = frame.wait(5'000'000'000ull);
                    if (result) { result = pool->reset(); }
                    if (result) { result = frame.begin(iteration + 1); }
                    if (result) { result = commands->begin(frame.submissionContext()); }
                    if (result) { result = executor.execute(*commands); }
                    if (result) { result = commands->end(); }
                    CommandBuffer* list[]{commands.get()};
                    if (result) { result = submissions.submit({.commandBuffers = {list, 1}}, frame); }
                    if (result) { result = frame.wait(5'000'000'000ull); }
                } else {
                    result = executor.execute(RenderGraphSubmitDesc{.graphicsQueue = &context.graphicsQueue});
                    if (result) { result = executor.waitForSubmittedWork(5'000'000'000ull); }
                }
                if (!result) { return RHITestResult::fail("Texture feedback graph execute: " + std::string(resultToString(result))); }
                resources = testTextureFeedbackState().resources.lock();
                if (!resources || !testTextureFeedbackState().sharedFeedback || resources->materialTextureFirstMips().size() != 1) {
                    return RHITestResult::fail("Consumers did not share one streamed texture feedback generation");
                }
                if (iteration == 0 && resources->materialTextureFirstMips()[0] != 2) {
                    return RHITestResult::fail("Texture fixture did not start at its coarse tail");
                }
                refined = resources->materialTextureFirstMips()[0] == 0;
                if (!refined) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            }
            // Streaming publishes one mip at a time, so mip 2 -> 0 requires
            // two accepted physical tail replacements.
            if (!refined || resources->textureStats().feedbackFrames == 0 || resources->textureStats().upgrades != 2) {
                return RHITestResult::fail(std::string(externalCommands ? "External-command" : "Graph-submitted") +
                    " epilogue did not consume the later pass's fine-mip demand: mip=" +
                    std::to_string(resources->materialTextureFirstMips()[0]) + " feedbackFrames=" +
                    std::to_string(resources->textureStats().feedbackFrames) + " upgrades=" +
                    std::to_string(resources->textureStats().upgrades));
            }
        }
        testTextureFeedbackState() = {};
        return RHITestResult::pass("Streamer-only graph joins shared Device feedback consumers and asynchronously refines through both executor entry points");
    }
};

class RenderGraphImageSamplePassPreviewTest : public RHITest {
public:
    RenderGraphImageSamplePassPreviewTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_image_sample_pass_preview";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(false);
        if (!result) {
            return RHITestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("ImageSamplePreview");
        graph.addNode("ImageSamplePass", "Image");
        graph.markOutput("Image.color");

        result = preview.render(graph, 160, 120);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(preview.lastLog());
            }
            return RHITestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result) + ": " + preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 160 * 120 / 2) {
            return RHITestResult::fail(
                std::string("image sample pass produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_image_sample_pass_preview.png";
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RHITestResult::fail(outputMessage);
        }

        return RHITestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphMaterialShaderObjectPassSmokeTest : public RHITest {
public:
    RenderGraphMaterialShaderObjectPassSmokeTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_material_shader_object_pass_smoke";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Shader Object Smoke Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("MaterialShaderObjectSmoke");
        graph.addNode(
            "SceneMaterialShaderObjectPass",
            "MaterialScene",
            render::RenderGraphProperties{
                {"path", "Asset/StandfordBunny/scene.gltf"},
                {"debugAlternateShaders", true},
            });
        graph.markOutput("MaterialScene.color");

        render::RenderGraphExecutor executor;
        std::string log;
        result = executor.compile(*device, graph, 128, 96, log);
        const bool hasRequiredCapabilities =
            device->capabilities().shaderObject &&
            device->capabilities().bindlessDescriptorHeap;
        if (!hasRequiredCapabilities) {
            if (!render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::fail(
                    std::string("expected Unsupported without shader-object capabilities, got ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RHITestResult::pass("SceneMaterialShaderObjectPass reported Unsupported without required capabilities");
        }

        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::compile returned ") +
                toString(result) +
                ": " +
                log);
        }

        return RHITestResult::pass();
    }
};

class RenderGraphVisibilityBufferPassSmokeTest : public RHITest {
public:
    RenderGraphVisibilityBufferPassSmokeTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_driven_preview_pass_smoke";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic VisibilityBufferPass Smoke Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableMeshShader = true,
                .enableTaskShader = true,
                .enableTaskShaderSubgroupBallot = true,
                .enableGeometryShader = true,
                .enableSubgroupSizeControl = true,
                .enableComputeFullSubgroups = true,
                .preferredTaskSubgroupSize = 32,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenPreviewSmoke");
        graph.addNode(
            "VisibilityBufferPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", "Asset/StandfordBunny/scene.gltf"},
                {"mode", "meshlet"},
            });
        graph.markOutput("GPUDriven.color");

        render::RenderGraphExecutor executor;
        std::string log;
        result = executor.compile(*device, graph, 128, 96, log);
        const bool hasRequiredCapabilities =
            device->capabilities().meshShader &&
            device->capabilities().taskShader &&
            device->capabilities().geometryShader &&
            device->capabilities().bindlessDescriptorHeap;
        if (!hasRequiredCapabilities) {
            if (!render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::fail(
                    std::string("expected Unsupported without task/mesh/geometry shader capabilities, got ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RHITestResult::pass("VisibilityBufferPass reported Unsupported without required capabilities");
        }

        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::compile returned ") +
                toString(result) +
                ": " +
                log);
        }

        return RHITestResult::pass();
    }
};

class RenderGraphGPUDrivenStreamAssetPassSmokeTest : public RHITest {
public:
    RenderGraphGPUDrivenStreamAssetPassSmokeTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_driven_streamasset_pass_smoke";
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint32_t kWidth = 128;
        constexpr uint32_t kHeight = 96;
        constexpr uint64_t kReadbackByteSize = static_cast<uint64_t>(kWidth) * kHeight * 4u;

        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic GPUDrivenStreamAssetPass Smoke Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableMeshShader = true,
                .enableGeometryShader = true,
                .enableRayQuery = true,
                .enableClusterAccelerationStructure = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RHITestResult::fail("GPUDrivenStreamAssetPass smoke device has no graphics queue");
        }

        const std::filesystem::path streamAssetPath =
            context.outputDirectory / "gpu_driven_streamasset_smoke.meshstream.bin";
        const std::filesystem::path sourcePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        scene::Scene runtimeScene;
        if (!runtimeScene.load(sourcePath)) {
            return RHITestResult::fail(
                "GPUDrivenStreamAssetPass smoke scene load failed: " +
                runtimeScene.lastLoadResult().error);
        }
        std::string buildReason;
        if (!scene::buildMeshletStreamAssetOffline(
                scene::MeshletStreamAssetOfflineBuildDesc{
                    .sourcePath = sourcePath,
                    .outputPath = streamAssetPath,
                },
                buildReason)) {
            return RHITestResult::fail("buildMeshletStreamAssetOffline failed: " + buildReason);
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenStreamAssetSmoke");
        graph.addNode(
            "GPUDrivenStreamAssetPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", sourcePath.string()},
                {"streamAssetPath", streamAssetPath.string()},
                {"enableClusterRtx", true},
                {"compactClas", true},
                {"rtasVisualization", true},
                {"rtasGranularity", "cluster-id"},
                {"maxClasBytes", 64ull * 1024ull * 1024ull},
                {"maxClasBuildClusters", 32},
                {"maxBlasClusterReferences", 4096},
                {"maxBlasBytes", 64ull * 1024ull * 1024ull},
                {"maxBlasBuilds", 4},
                {"maxFallbackBlasBytes", 64ull * 1024ull * 1024ull},
                {"maxLockedFallbackPages", 1},
                {"maxResidentPages", 64},
                {"maxPageUploadsPerFrame", 1},
            });
        graph.markOutput("GPUDriven.color");

        render::RenderGraphExecutor executor;
        executor.bindRuntimeScene(&runtimeScene);
        std::string log;
        result = executor.compile(*device, graph, kWidth, kHeight, log);
        const bool hasRequiredCapabilities =
            device->capabilities().meshShader &&
            device->capabilities().bindlessDescriptorHeap &&
            device->capabilities().rayQuery &&
            device->capabilities().clusterAccelerationStructure;
        if (!hasRequiredCapabilities) {
            if (!render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::fail(
                    std::string("expected Unsupported without mesh shader capabilities, got ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RHITestResult::pass("GPUDrivenStreamAssetPass reported Unsupported without required capabilities");
        }

        if (!result) {
            return RHITestResult::fail(
                std::string("RenderGraphExecutor::compile returned ") +
                toString(result) +
                ": " +
                log);
        }

        constexpr uint32_t kStreamingWarmupFrameCount = 16;
        for (uint32_t frame = 0; frame < kStreamingWarmupFrameCount; ++frame) {
            result = executor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
            });
            if (!result) {
                return RHITestResult::fail(
                    std::string("RenderGraphExecutor::execute frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
            result = executor.waitForSubmittedWork(5'000'000'000ull);
            if (!result) {
                return RHITestResult::fail(
                    std::string("RenderGraphExecutor::waitForSubmittedWork frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = device->createBuffer(render::BufferDesc{
                .size = kReadbackByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            }).transform([&](auto rhiValue) { readbackBuffer = std::move(rhiValue); });
        if (!result || readbackBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        render::RenderFrameContext readbackFrame;
        render::QueueSubmissionTracker readbackTracker;
        result = readbackTracker.initialize(*device, *graphicsQueue);
        if (result) { result = readbackFrame.begin(kStreamingWarmupFrameCount); }
        if (result) { result = commandBuffer->begin(readbackFrame.submissionContext()); }
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        result = executor.execute(*commandBuffer);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::execute(readback) returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("GPUDriven.color");
        if (output == nullptr || output->texture == nullptr) {
            return RHITestResult::fail("GPUDrivenStreamAssetPass smoke output resource is missing");
        }

        result = executor.transitionOutput(*commandBuffer, "GPUDriven.color", render::ResourceState::TransferSource);
        if (!result) {
            return RHITestResult::fail(std::string("transitionOutput returned ") + toString(result));
        }
        commandBuffer->copyTextureToBuffer(render::TextureBufferCopyDesc{
            .texture = output->texture,
            .buffer = readbackBuffer.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
            .mipLevel = 0,
            .baseLayer = 0,
        });

        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = readbackTracker.submit(render::QueueSubmitDesc{
            .commandBuffers = {commandBuffers, 1},
            .signalFence = fence.get(),
        }, readbackFrame);
        if (!result) {
            return RHITestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }
        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        readbackBuffer->invalidate();
        void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("readback buffer did not map");
        }

        std::vector<uint8_t> pixels(static_cast<size_t>(kReadbackByteSize));
        std::memcpy(pixels.data(), mapped, pixels.size());
        readbackBuffer->unmap();

        uint32_t nonClearPixelCount = 0;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index) {
            const uint8_t r = pixels[index * 4 + 0];
            const uint8_t g = pixels[index * 4 + 1];
            const uint8_t b = pixels[index * 4 + 2];
            if (r > 24 || g > 24 || b > 24) {
                ++nonClearPixelCount;
            }
        }
        if (nonClearPixelCount == 0) {
            return RHITestResult::fail("GPUDrivenStreamAssetPass smoke produced only clear pixels");
        }

        render::RenderGraph streamedFallbackGraph;
        streamedFallbackGraph.setName("GPUDrivenStreamAssetStreamedFallbackSmoke");
        streamedFallbackGraph.addNode(
            "GPUDrivenStreamAssetPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", sourcePath.string()},
                {"streamAssetPath", streamAssetPath.string()},
                {"maxLockedFallbackPages", 1},
                {"maxResidentPages", 2},
                {"maxPageUploadsPerFrame", 1},
            });
        streamedFallbackGraph.markOutput("GPUDriven.color");
        streamedFallbackGraph.markOutput("GPUDriven.visibility");

        render::RenderGraphExecutor streamedFallbackExecutor;
        streamedFallbackExecutor.bindRuntimeScene(&runtimeScene);
        log.clear();
        result = streamedFallbackExecutor.compile(
            *device,
            streamedFallbackGraph,
            kWidth,
            kHeight,
            log);
        if (!result) {
            return RHITestResult::fail(
                std::string("streamed-fallback RenderGraphExecutor::compile returned ") +
                toString(result) +
                ": " +
                log);
        }
        for (uint32_t frame = 0; frame < kStreamingWarmupFrameCount; ++frame) {
            result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
            });
            if (!result) {
                return RHITestResult::fail(
                    std::string("streamed-fallback RenderGraphExecutor::execute frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
            result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
            if (!result) {
                return RHITestResult::fail(
                    std::string("streamed-fallback RenderGraphExecutor::waitForSubmittedWork frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
        }

        std::unique_ptr<render::CommandBuffer> rasterCommandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { rasterCommandBuffer = std::move(rhiValue); });
        if (!result || rasterCommandBuffer == nullptr) {
            return RHITestResult::fail(
                std::string("createCommandBuffer(raster readback) returned ") +
                toString(result));
        }
        std::unique_ptr<render::Fence> rasterFence;
        result = device->createFence(false).transform([&](auto rhiValue) { rasterFence = std::move(rhiValue); });
        if (!result || rasterFence == nullptr) {
            return RHITestResult::fail(
                std::string("createFence(raster readback) returned ") +
                toString(result));
        }
        std::unique_ptr<render::Buffer> rasterColorReadback;
        std::unique_ptr<render::Buffer> rasterVisibilityReadback;
        auto createRasterReadback = [&](std::unique_ptr<render::Buffer>& buffer) {
            return device->createBuffer(render::BufferDesc{
                    .size = kReadbackByteSize,
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback,
                }).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
        };
        result = createRasterReadback(rasterColorReadback);
        if (!result || rasterColorReadback == nullptr) {
            return RHITestResult::fail(
                std::string("createBuffer(raster color readback) returned ") +
                toString(result));
        }
        result = createRasterReadback(rasterVisibilityReadback);
        if (!result || rasterVisibilityReadback == nullptr) {
            return RHITestResult::fail(
                std::string("createBuffer(raster visibility readback) returned ") +
                toString(result));
        }

        result = rasterCommandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(
                std::string("raster readback CommandBuffer::begin returned ") +
                toString(result));
        }
        result = streamedFallbackExecutor.execute(*rasterCommandBuffer);
        if (!result) {
            return RHITestResult::fail(
                std::string("streamed-fallback execute(readback) returned ") +
                toString(result));
        }
        render::RenderGraphResource* rasterColor =
            streamedFallbackExecutor.outputResource("GPUDriven.color");
        render::RenderGraphResource* rasterVisibility =
            streamedFallbackExecutor.outputResource("GPUDriven.visibility");
        if (rasterColor == nullptr || rasterColor->texture == nullptr ||
            rasterVisibility == nullptr || rasterVisibility->texture == nullptr) {
            return RHITestResult::fail(
                "streamed-fallback raster outputs are missing");
        }
        result = streamedFallbackExecutor.transitionOutput(
            *rasterCommandBuffer,
            "GPUDriven.color",
            render::ResourceState::TransferSource);
        if (result) {
            result = streamedFallbackExecutor.transitionOutput(
                *rasterCommandBuffer,
                "GPUDriven.visibility",
                render::ResourceState::TransferSource);
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("transitionOutput(raster readback) returned ") +
                toString(result));
        }
        const render::TextureBufferCopyDesc colorCopy{
            .texture = rasterColor->texture,
            .buffer = rasterColorReadback.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
            .mipLevel = 0,
            .baseLayer = 0,
        };
        render::TextureBufferCopyDesc visibilityCopy = colorCopy;
        visibilityCopy.texture = rasterVisibility->texture;
        visibilityCopy.buffer = rasterVisibilityReadback.get();
        rasterCommandBuffer->copyTextureToBuffer(colorCopy);
        rasterCommandBuffer->copyTextureToBuffer(visibilityCopy);
        result = rasterCommandBuffer->end();
        if (!result) {
            return RHITestResult::fail(
                std::string("raster readback CommandBuffer::end returned ") +
                toString(result));
        }
        render::CommandBuffer* rasterCommandBuffers[] = {rasterCommandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = {rasterCommandBuffers, 1},
            .signalFence = rasterFence.get(),
        });
        if (!result) {
            return RHITestResult::fail(
                std::string("raster readback Queue::submit returned ") +
                toString(result));
        }
        result = rasterFence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(
                std::string("raster readback Fence::wait returned ") +
                toString(result));
        }

        rasterColorReadback->invalidate();
        const auto* rasterColorPixels =
            static_cast<const uint8_t*>(rasterColorReadback->map());
        if (rasterColorPixels == nullptr) {
            return RHITestResult::fail("raster color readback did not map");
        }
        uint32_t rasterColorPixelCount = 0;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index) {
            const uint8_t r = rasterColorPixels[index * 4u + 0u];
            const uint8_t g = rasterColorPixels[index * 4u + 1u];
            const uint8_t b = rasterColorPixels[index * 4u + 2u];
            rasterColorPixelCount += r > 24u || g > 24u || b > 24u ? 1u : 0u;
        }
        rasterColorReadback->unmap();

        rasterVisibilityReadback->invalidate();
        const auto* visibilityIds =
            static_cast<const uint32_t*>(rasterVisibilityReadback->map());
        if (visibilityIds == nullptr) {
            return RHITestResult::fail("raster visibility readback did not map");
        }
        uint32_t rasterVisibilityPixelCount = 0;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index) {
            const uint32_t visibilityId = visibilityIds[index];
            if ((visibilityId >> 7u) != 0u) {
                ++rasterVisibilityPixelCount;
            }
        }
        rasterVisibilityReadback->unmap();
        if (rasterColorPixelCount == 0 || rasterVisibilityPixelCount == 0) {
            return RHITestResult::fail(
                std::string("streamed-fallback raster path produced colorPixels=") +
                std::to_string(rasterColorPixelCount) +
                " visibilityPixels=" +
                std::to_string(rasterVisibilityPixelCount));
        }

        auto captureVisibilityPixelCount = [&](
                                               uint32_t width,
                                               uint32_t height,
                                               uint32_t& pixelCount,
                                               std::string& error) -> bool {
            pixelCount = 0;
            error.clear();
            render::RenderGraphResource* captureOutput =
                streamedFallbackExecutor.outputResource("GPUDriven.visibility");
            if (captureOutput == nullptr ||
                captureOutput->texture == nullptr ||
                captureOutput->desc.width != width ||
                captureOutput->desc.height != height) {
                error = "output dimensions do not match the capture";
                return false;
            }

            std::unique_ptr<render::Buffer> captureReadback;
            result = device->createBuffer(render::BufferDesc{
                    .size = static_cast<uint64_t>(width) * height * sizeof(uint32_t),
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback,
                }).transform([&](auto rhiValue) { captureReadback = std::move(rhiValue); });
            if (!result || captureReadback == nullptr) {
                error = std::string("createBuffer(visibility capture) returned ") +
                    toString(result);
                return false;
            }

            std::unique_ptr<render::CommandBuffer> captureCommandBuffer;
            result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { captureCommandBuffer = std::move(rhiValue); });
            if (!result || captureCommandBuffer == nullptr) {
                error = std::string("createCommandBuffer(visibility capture) returned ") +
                    toString(result);
                return false;
            }
            std::unique_ptr<render::Fence> captureFence;
            result = device->createFence(false).transform([&](auto rhiValue) { captureFence = std::move(rhiValue); });
            if (!result || captureFence == nullptr) {
                error = std::string("createFence(visibility capture) returned ") +
                    toString(result);
                return false;
            }

            result = captureCommandBuffer->begin();
            if (result) {
                result = streamedFallbackExecutor.transitionOutput(
                    *captureCommandBuffer,
                    "GPUDriven.visibility",
                    render::ResourceState::TransferSource);
            }
            if (result) {
                captureCommandBuffer->copyTextureToBuffer(render::TextureBufferCopyDesc{
                    .texture = captureOutput->texture,
                    .buffer = captureReadback.get(),
                    .width = width,
                    .height = height,
                    .depth = 1,
                    .mipLevel = 0,
                    .baseLayer = 0,
                });
                result = captureCommandBuffer->end();
            }
            if (!result) {
                error = std::string("record visibility capture returned ") + toString(result);
                return false;
            }

            render::CommandBuffer* captureCommandBuffers[] = {captureCommandBuffer.get()};
            result = graphicsQueue->submit(render::QueueSubmitDesc{
                .commandBuffers = {captureCommandBuffers, 1},
                .signalFence = captureFence.get(),
            });
            if (result) {
                result = captureFence->wait(5'000'000'000ull);
            }
            if (!result) {
                error = std::string("submit visibility capture returned ") + toString(result);
                return false;
            }

            captureReadback->invalidate();
            const auto* visibilityPixels =
                static_cast<const uint32_t*>(captureReadback->map());
            if (visibilityPixels == nullptr) {
                error = "visibility capture buffer did not map";
                return false;
            }
            for (uint32_t index = 0; index < width * height; ++index) {
                pixelCount += (visibilityPixels[index] >> 7u) != 0u ? 1u : 0u;
            }
            captureReadback->unmap();
            return true;
        };

        constexpr uint32_t kResizedWidth = 96;
        constexpr uint32_t kResizedHeight = 72;
        log.clear();
        result = streamedFallbackExecutor.compile(
            *device,
            streamedFallbackGraph,
            kResizedWidth,
            kResizedHeight,
            log);
        if (!result) {
            return RHITestResult::fail(
                std::string("resized streamed-fallback compile returned ") +
                toString(result) +
                ": " +
                log);
        }
        for (uint32_t frame = 0; frame < 3 && result; ++frame) {
            result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
            });
            if (result) {
                result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
            }
        }
        uint32_t resizedVisibilityPixelCount = 0;
        std::string captureError;
        if (!result ||
            !captureVisibilityPixelCount(
                kResizedWidth,
                kResizedHeight,
                resizedVisibilityPixelCount,
                captureError)) {
            return RHITestResult::fail(
                std::string("resized streamed-fallback capture failed: ") +
                captureError);
        }
        if (resizedVisibilityPixelCount == 0) {
            return RHITestResult::fail("resized stream frame produced no visibility pixels");
        }

        std::vector<scene::SceneEntity> visibleObjects;
        for (const scene::RenderNode& renderNode : runtimeScene.renderNodes()) {
            if (renderNode.visible &&
                std::find(visibleObjects.begin(), visibleObjects.end(), renderNode.object) ==
                    visibleObjects.end()) {
                visibleObjects.push_back(renderNode.object);
            }
        }
        if (visibleObjects.empty()) {
            return RHITestResult::fail("stream visibility sync test found no visible objects");
        }
        const uint64_t transformRevisionBeforeHide = runtimeScene.transformRevision();
        const uint64_t visibilityRevisionBeforeHide = runtimeScene.visibilityRevision();
        for (scene::SceneEntity object : visibleObjects) {
            if (!runtimeScene.setObjectVisible(object, false)) {
                return RHITestResult::fail("stream visibility sync test could not hide an object");
            }
        }
        if (runtimeScene.transformRevision() != transformRevisionBeforeHide ||
            runtimeScene.visibilityRevision() == visibilityRevisionBeforeHide) {
            return RHITestResult::fail(
                "stream visibility sync test did not isolate visibility revision changes");
        }

        result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = graphicsQueue,
        });
        if (result) {
            result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("hidden stream frame returned ") + toString(result));
        }
        uint32_t hiddenVisibilityPixelCount = 0;
        if (!captureVisibilityPixelCount(
                kResizedWidth,
                kResizedHeight,
                hiddenVisibilityPixelCount,
                captureError)) {
            return RHITestResult::fail(captureError);
        }
        if (hiddenVisibilityPixelCount != 0) {
            return RHITestResult::fail(
                "visibility-only hide left " +
                std::to_string(hiddenVisibilityPixelCount) +
                " raster pixels");
        }

        for (scene::SceneEntity object : visibleObjects) {
            if (!runtimeScene.setObjectVisible(object, true)) {
                return RHITestResult::fail("stream visibility sync test could not restore an object");
            }
        }
        result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = graphicsQueue,
        });
        if (result) {
            result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("restored stream frame returned ") + toString(result));
        }
        uint32_t restoredVisibilityPixelCount = 0;
        if (!captureVisibilityPixelCount(
                kResizedWidth,
                kResizedHeight,
                restoredVisibilityPixelCount,
                captureError)) {
            return RHITestResult::fail(captureError);
        }
        if (restoredVisibilityPixelCount == 0) {
            return RHITestResult::fail("visibility-only show did not restore raster pixels");
        }

        for (uint32_t frame = 0; frame < 3; ++frame) {
            result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
            });
            if (result) {
                result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
            }
            if (!result) {
                return RHITestResult::fail(
                    "post-show stabilization frame " +
                    std::to_string(frame) +
                    " failed");
            }
            uint32_t stableVisibilityPixelCount = 0;
            if (!captureVisibilityPixelCount(
                    kResizedWidth,
                    kResizedHeight,
                    stableVisibilityPixelCount,
                    captureError)) {
                return RHITestResult::fail(
                    "post-show stabilization capture failed: " +
                    captureError);
            }
            if (stableVisibilityPixelCount == 0) {
                return RHITestResult::fail(
                    "post-show stabilization frame " +
                    std::to_string(frame) +
                    " produced no visibility pixels");
            }
        }

        constexpr uint32_t kSecondResizeWidth = 80;
        constexpr uint32_t kSecondResizeHeight = 60;
        log.clear();
        result = streamedFallbackExecutor.compile(
            *device,
            streamedFallbackGraph,
            kSecondResizeWidth,
            kSecondResizeHeight,
            log);
        if (!result) {
            return RHITestResult::fail(
                std::string("post-show resize compile returned ") +
                toString(result) +
                ": " +
                log);
        }
        for (uint32_t frame = 0; frame < 3; ++frame) {
            result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
            });
            if (result) {
                result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
            }
            if (!result) {
                return RHITestResult::fail(
                    "post-show resized frame " +
                    std::to_string(frame) +
                    " failed");
            }
            uint32_t postShowResizedPixels = 0;
            if (!captureVisibilityPixelCount(
                    kSecondResizeWidth,
                    kSecondResizeHeight,
                    postShowResizedPixels,
                    captureError)) {
                return RHITestResult::fail(
                    "post-show resize capture failed: " +
                    captureError);
            }
            if (postShowResizedPixels == 0) {
                return RHITestResult::fail(
                    "post-show resized frame " +
                    std::to_string(frame) +
                    " produced no visibility pixels");
            }
        }

        (void)device->waitIdle();
        return RHITestResult::pass();
    }
};

class RenderGraphGPUDrivenMixedProducerRenderTest : public RHITest {
    int rasterMode_ = -1;
public:
    explicit RenderGraphGPUDrivenMixedProducerRenderTest(int rasterMode = -1) : rasterMode_(rasterMode)
    {
        type = RHITestType::Rendering;
        static constexpr const char* names[] = {
            "mixed_producer_raster_prepared", "mixed_producer_raster_legacy", "mixed_producer_raster_plane",
            "mixed_producer_raster_cooperative", "mixed_producer_raster_work_bins"};
        name = rasterMode < 0 ? "render_graph_gpu_driven_mixed_producer_render" : names[rasterMode];
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint32_t kWidth = 256;
        constexpr uint32_t kHeight = 192;
        constexpr uint32_t kMaxActiveGroups = 64;
        // Reach the finest cut through all dependency levels before asserting
        // that both producers contain clusters eligible for software raster.
        constexpr uint32_t kWarmupFrameCount = 128;
        constexpr uint64_t kPixelByteSize =
            static_cast<uint64_t>(kWidth) * kHeight * sizeof(uint32_t);

        const std::filesystem::path sourcePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) /
            "Asset/StandfordBunny/scene.gltf";
        const std::filesystem::path streamAssetPath =
            context.outputDirectory / "gpu_driven_mixed_producer.meshstream.bin";
        std::string reason;
        if (!scene::buildMeshletStreamAssetOffline(
                scene::MeshletStreamAssetOfflineBuildDesc{
                    .sourcePath = sourcePath,
                    .outputPath = streamAssetPath,
                },
                reason)) {
            return RHITestResult::fail(
                "mixed-producer streamasset build failed: " + reason);
        }

        scene::MeshletStreamAsset streamAsset;
        if (!streamAsset.open(streamAssetPath, reason) ||
            !streamAsset.isCurrentForSource(sourcePath)) {
            return RHITestResult::fail(
                "mixed-producer streamasset open failed: " + reason);
        }

        float4x4 streamMount = float4x4::Identity();
        float4x4 residentMount = float4x4::Identity();
        streamMount.SetupByTranslation(float3(-0.14f, 0.0f, 0.0f));
        residentMount.SetupByTranslation(float3(0.14f, 0.0f, 0.0f));
        scene::Scene runtimeScene;
        if (!runtimeScene.compose(
                {
                    scene::SceneSourceDesc{
                        .id = "resident",
                        .path = sourcePath,
                        .mountMatrix = residentMount,
                    },
                    scene::SceneSourceDesc{
                        .id = "stream",
                        .path = sourcePath,
                        .mountMatrix = streamMount,
                    },
                },
                reason,
                sourcePath)) {
            return RHITestResult::fail(
                "mixed-producer scene composition failed: " + reason);
        }
        if (runtimeScene.renderNodes().size() != 2 ||
            streamAsset.instances().size() != 1 ||
            streamAsset.instances()[0].renderNodeIndex != 0 ||
            runtimeScene.renderNodeIndexForSource("resident", 0) != 0 ||
            runtimeScene.renderNodeIndexForSource("stream", 0) != 1) {
            return RHITestResult::fail(
                "mixed-producer fixture no longer has one stream owner and one ordinary instance");
        }

        uint64_t requestedStreamRecordCapacity = 0;
        for (const scene::MeshletStreamInstanceInfo& instance :
             streamAsset.instances()) {
            if (instance.visible == 0 ||
                instance.primitiveIndex >= streamAsset.primitives().size()) {
                continue;
            }
            requestedStreamRecordCapacity +=
                streamAsset.primitives()[instance.primitiveIndex].groupCount;
        }
        requestedStreamRecordCapacity =
            std::min<uint64_t>(requestedStreamRecordCapacity, kMaxActiveGroups) *
            streamAsset.maxPageClusters();
        if (requestedStreamRecordCapacity == 0 ||
            !render::visibilityRecordCapacityFitsId(
                requestedStreamRecordCapacity)) {
            return RHITestResult::fail(
                "mixed-producer fixture has an invalid stream record capacity");
        }

        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic GPUDriven mixed-producer render test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableMeshShader = true,
                .enableTaskShader = true,
                .enableTaskShaderSubgroupBallot = true,
                .enableGeometryShader = true,
                .enableSubgroupSizeControl = true,
                .enableComputeFullSubgroups = true,
                .preferredTaskSubgroupSize = 32,
                .enableAsyncCompute = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(
                std::string("createDevice returned ") + toString(result));
        }
        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RHITestResult::fail(
                "mixed-producer device has no graphics queue");
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenMixedProducer");
        render::RenderDebugRuntime debugRuntime;
        debug::DebugValue debugJobs = debug::DebugValue::array();
        debug::DebugValue clusterJobs = debug::DebugValue::array();
        graph.addNode(
            "VisibilityBufferPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", sourcePath.string()},
                {"streamAssetPath", streamAssetPath.string()},
                {"streamSourceId", "stream"},
                {"enableMeshletStreaming", true},
                {"maxLockedFallbackPages", 4},
                {"maxResidentPages", 64},
                {"maxPageUploadsPerFrame", 4},
                {"maxActiveGroups", kMaxActiveGroups},
                {"mode", "meshlet"},
                {"autoLod", false}, {"lodLevel", 0},
                {"instanceFrustumCull", false},
                {"instanceHzbCull", false},
                {"meshletFrustumCull", false},
                {"meshletNormalConeCull", false},
                {"camera", {
                    {"projection", "perspective"},
                    {"fovDegrees", 60.0f},
                    {"znear", 0.01f},
                    {"zfar", 100.0f},
                    {"reversedZ", true},
                    {"eye", {-0.0168404f, 0.110154f, 0.55f}},
                    {"center", {-0.0168404f, 0.110154f, 0.0f}},
                    {"up", {0.0f, 1.0f, 0.0f}},
                }},
            });
        if (rasterMode_ >= 0) {
            const auto node = graph.findNode("GPUDriven")->id;
            graph.setNodeRuntimeProperty(node, "softwareRasterPreparedVertices", rasterMode_ == 0 || rasterMode_ == 2);
            graph.setNodeRuntimeProperty(node, "softwareRasterIncrementalDepth", rasterMode_ == 2);
            graph.setNodeRuntimeProperty(node, "softwareRasterCooperativeLoad", rasterMode_ != 1);
            graph.setNodeRuntimeProperty(node, "softwareRasterSharedScreenVertices", false);
            graph.setNodeRuntimeProperty(node, "softwareRasterWorkBins", rasterMode_ == 4);
        }
        graph.markOutput("GPUDriven.color");
        graph.markOutput("GPUDriven.visibility");
        graph.markOutput("GPUDriven.depth");

        render::RenderGraphExecutor executor;
        executor.bindRuntimeScene(&runtimeScene);
        executor.setDebugObserver(&debugRuntime);
        std::string log;
        result = executor.compile(*device, graph, kWidth, kHeight, log);
        const bool hasRequiredCapabilities =
            device->capabilities().meshShader &&
            device->capabilities().taskShader &&
            device->capabilities().geometryShader &&
            device->capabilities().bindlessDescriptorHeap;
        if (!hasRequiredCapabilities) {
            if (!render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::fail(
                    std::string("expected Unsupported without mixed raster capabilities, got ") +
                    toString(result) + ": " + log);
            }
            return RHITestResult::skip(
                "VisibilityBufferPass mixed producer mode requires task/mesh/geometry shaders and bindless descriptors");
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("mixed-producer graph compile returned ") +
                toString(result) + ": " + log);
        }

        for (uint32_t frame = 0; frame < kWarmupFrameCount; ++frame) {
            if (frame + 1 == kWarmupFrameCount) {
                debug::DebugValue batches = debug::DebugValue::array();
                for (const char* point : {"AfterTraversal", "AfterEarlyCull", "AfterLateCull", "AfterPass"}) {
                    debug::DebugValue resources = debug::DebugValue::array({{{"id", "streaming.GPUDriven.activeHeader"}, {"count", 1}}});
                    if (std::string_view(point) == "AfterPass") {
                        for (const char* id : {"gpuScene.GPUDriven.meshletDraws", "gpuScene.GPUDriven.geometries", "streaming.GPUDriven.visibleClusters"}) {
                            resources.push_back({{"id", id}, {"count", 1}});
                        }
                    }
                    batches.push_back({{"pass", "GPUDriven"}, {"checkpoint", point}, {"resources", std::move(resources)},
                        {"probes", {{{"id", "streaming.GPUDriven.activeHeader"}, {"name", "active"},
                            {"operation", "minMax"}, {"field", "activeGroupCount"}, {"count", 1}}}}});
                }
                const auto queued = debugRuntime.core().dispatch({{"method", "gpu.probe"}, {"params", {{"batches", batches}}}});
                if (queued["status"] != "ok") { return RHITestResult::fail("Mixed debug capture enqueue: " + queued.dump()); }
                debugJobs = queued["result"]["jobs"];
                if (device->capabilities().shaderBufferInt64Atomics) {
                    debug::DebugValue clusterBatches = debug::DebugValue::array();
                    for (const char* point : {"AfterResidentEarlyBins", "AfterStreamEarlyBins", "AfterResidentLateBins", "AfterStreamLateBins"}) {
                        clusterBatches.push_back({{"pass", "GPUDriven"}, {"checkpoint", point}, {"resources", {
                            {{"id", "hybrid.GPUDriven.clusters"}, {"count", 16}},
                            {{"id", "hybrid.GPUDriven.arguments"}, {"count", 15}}}}});
                    }
                    const auto clusters = debugRuntime.core().dispatch({{"method", "capture.batch"}, {"params", {{"batches", clusterBatches}}}});
                    if (clusters["status"] != "ok") { return RHITestResult::fail("Cluster capture enqueue: " + clusters.dump()); }
                    clusterJobs = clusters["result"]["jobs"];
                }
            }
            result = executor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
                .computeQueue = device->getQueue(render::QueueType::Compute),
            });
            if (result) {
                result = executor.waitForSubmittedWork(5'000'000'000ull);
            }
            if (!result) {
                return RHITestResult::fail(
                    "mixed-producer warmup frame " +
                    std::to_string(frame) + " returned " + toString(result));
            }
            if (device->capabilities().independentComputeQueue && executor.executionStats().asyncComputeBranches != 3) {
                return RHITestResult::fail("Mixed rendering must fork resident early/late and stream early under the default policy");
            }
            debugRuntime.poll();
        }

        uint64_t debugExecution = UINT64_MAX;
        for (const auto& job : debugJobs) {
            const auto completed = debugRuntime.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", job.at("job")}}}});
            if (completed["status"] != "ok" || completed["result"]["state"] != "Ready") {
                return RHITestResult::fail("Mixed checkpoint capture: " + completed.dump());
            }
            const auto execution = completed["result"]["evidence"]["execution"].get<uint64_t>();
            if (debugExecution != UINT64_MAX && execution != debugExecution) { return RHITestResult::fail("Mixed checkpoints span executions"); }
            debugExecution = execution;
            const auto comparison = debugRuntime.core().dispatch({{"method", "eval"}, {"params", {
                {"job", job.at("job")}, {"expression", "probes.active.min == buffers[\"streaming.GPUDriven.activeHeader\"][0].activeGroupCount"}}}});
            if (comparison["status"] != "ok" || comparison["result"]["value"] != true) {
                return RHITestResult::fail("GPU probe differs from checkpoint readback: " + comparison.dump());
            }
        }
        std::array<uint32_t, 2> softwareClusters{};
        for (size_t jobIndex = 0; jobIndex < clusterJobs.size(); ++jobIndex) {
            const auto job = clusterJobs[jobIndex].at("job");
            std::array<uint32_t, 16> header{};
            for (size_t i = 0; i < header.size(); ++i) {
                const auto value = debugRuntime.core().dispatch({{"method", "eval"}, {"params", {
                    {"job", job}, {"expression", "buffers[\"hybrid.GPUDriven.clusters\"][" + std::to_string(i) + "]"}}}});
                if (value["status"] != "ok") { return RHITestResult::fail("Cluster header capture: " + value.dump()); }
                header[i] = value["result"]["value"].get<uint32_t>();
            }
            uint32_t total = 0;
            for (size_t bin = 0; bin < 5; ++bin) { total += header[bin]; }
            if (header[14] != 0 || total > header[12] || header[12] > header[5]) {
                return RHITestResult::fail("Real cluster bins overflowed their candidate capacity");
            }
            softwareClusters[jobIndex % 2] += header[4];
        }
        if (!clusterJobs.empty() && (softwareClusters[0] == 0 || softwareClusters[1] == 0)) {
            return RHITestResult::fail("Real resident/stream producers did not both use software cluster bins: " +
                std::to_string(softwareClusters[0]) + "/" + std::to_string(softwareClusters[1]));
        }
        const auto residentRecord = debugRuntime.core().dispatch({{"method", "eval"}, {"params", {
            {"job", debugJobs.back().at("job")}, {"expression", "buffers[\"gpuScene.GPUDriven.meshletDraws\"][0].source.name"}}}});
        if (residentRecord["status"] != "ok" || residentRecord["result"]["value"] != "Resident") {
            return RHITestResult::fail("Resident record lost typed source: " + residentRecord.dump());
        }

        render::RenderSubsystemHost* subsystemHost = executor.subsystemHost();
        render::GPUSceneSubsystem* gpuScene = subsystemHost != nullptr
            ? subsystemHost->get<render::GPUSceneSubsystem>()
            : nullptr;
        if (gpuScene == nullptr) {
            return RHITestResult::fail(
                "mixed-producer graph did not publish GPUScene");
        }
        const render::GPUSceneGlobalBufferViews& globalViews =
            gpuScene->globalBufferViews();
        if (!globalViews.meshletDraws.valid() ||
            globalViews.meshletDraws.structureStride !=
                sizeof(render::VisibleClusterRecord) ||
            globalViews.meshletDraws.size == 0 ||
            (globalViews.meshletDraws.size %
                sizeof(render::VisibleClusterRecord)) != 0) {
            return RHITestResult::fail(
                "mixed-producer GPUScene resident record namespace is invalid");
        }
        const uint32_t streamRecordBase = static_cast<uint32_t>(
            globalViews.meshletDraws.size /
            sizeof(render::VisibleClusterRecord));
        if (residentRecord["result"]["coverage"]["streaming.GPUDriven.visibleClusters"]["visibleRecordBase"] != streamRecordBase) {
            return RHITestResult::fail("Stream capture lost the mixed visibility namespace offset");
        }
        if (!render::visibilityRecordCapacityFitsId(
                static_cast<uint64_t>(streamRecordBase) +
                requestedStreamRecordCapacity)) {
            return RHITestResult::fail(
                "mixed-producer combined record namespace exceeds visibility IDs");
        }

        const render::GPUSceneInstanceId streamInstance =
            gpuScene->instanceForRenderNode(1);
        const render::GPUSceneInstanceId residentInstance =
            gpuScene->instanceForRenderNode(0);
        if (!streamInstance.valid() || !residentInstance.valid() ||
            streamInstance == residentInstance) {
            return RHITestResult::fail(
                "mixed-producer fixture did not map two dense GPUScene instances");
        }

        render::RenderGraphResource* color =
            executor.outputResource("GPUDriven.color");
        render::RenderGraphResource* visibility =
            executor.outputResource("GPUDriven.visibility");
        render::RenderGraphResource* depth =
            executor.outputResource("GPUDriven.depth");
        if (color == nullptr || color->texture == nullptr ||
            visibility == nullptr || visibility->texture == nullptr ||
            depth == nullptr || depth->texture == nullptr ||
            color->desc.format != render::Format::RGBA8Unorm ||
            visibility->desc.format != render::Format::R32Uint ||
            depth->desc.format != render::Format::D32Sfloat) {
            return RHITestResult::fail(
                "mixed-producer shared color/visibility/depth surfaces are missing");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(
                std::string("createCommandPool(mixed producer) returned ") +
                toString(result));
        }
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(
                std::string("createCommandBuffer(mixed producer) returned ") +
                toString(result));
        }
        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(
                std::string("createFence(mixed producer) returned ") +
                toString(result));
        }

        auto makeReadback = [&](uint64_t size,
                                std::unique_ptr<render::Buffer>& buffer) {
            return device->createBuffer(render::BufferDesc{
                    .size = size,
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback,
                    .queueAccess = render::QueueAccessBits::Graphics,
                }).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
        };
        std::unique_ptr<render::Buffer> colorReadback;
        std::unique_ptr<render::Buffer> visibilityReadback;
        std::unique_ptr<render::Buffer> depthReadback;
        std::unique_ptr<render::Buffer> residentRecordReadback;
        result = makeReadback(kPixelByteSize, colorReadback);
        if (result) {
            result = makeReadback(kPixelByteSize, visibilityReadback);
        }
        if (result) {
            result = makeReadback(kPixelByteSize, depthReadback);
        }
        if (result) {
            result = makeReadback(
                globalViews.meshletDraws.size,
                residentRecordReadback);
        }
        if (!result || colorReadback == nullptr ||
            visibilityReadback == nullptr || depthReadback == nullptr ||
            residentRecordReadback == nullptr) {
            return RHITestResult::fail(
                std::string("createBuffer(mixed producer readback) returned ") +
                toString(result));
        }

        result = commandBuffer->begin();
        if (result) {
            result = executor.execute(*commandBuffer);
        }
        for (const char* outputName : {
                 "GPUDriven.color",
                 "GPUDriven.visibility",
                 "GPUDriven.depth",
             }) {
            if (result) {
                result = executor.transitionOutput(
                    *commandBuffer,
                    outputName,
                    render::ResourceState::TransferSource);
            }
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("record mixed-producer outputs returned ") +
                toString(result));
        }

        const render::TextureBufferCopyDesc colorCopy{
            .texture = color->texture,
            .buffer = colorReadback.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
            .mipLevel = 0,
            .baseLayer = 0,
        };
        render::TextureBufferCopyDesc visibilityCopy = colorCopy;
        visibilityCopy.texture = visibility->texture;
        visibilityCopy.buffer = visibilityReadback.get();
        render::TextureBufferCopyDesc depthCopy = colorCopy;
        depthCopy.texture = depth->texture;
        depthCopy.buffer = depthReadback.get();
        commandBuffer->copyTextureToBuffer(colorCopy);
        commandBuffer->copyTextureToBuffer(visibilityCopy);
        commandBuffer->copyTextureToBuffer(depthCopy);

        const render::BufferBarrierDesc residentRecordsToCopy{
            .buffer = globalViews.meshletDraws.buffer,
            .before = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead},
            .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
            .range = {.offset = globalViews.meshletDraws.offset, .size = globalViews.meshletDraws.size},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = {&residentRecordsToCopy, 1},
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        {
            auto sourceSlice = globalViews.meshletDraws.buffer->slice({globalViews.meshletDraws.offset, globalViews.meshletDraws.size});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = residentRecordReadback.get()->slice({0, globalViews.meshletDraws.size});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commandBuffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        const render::BufferBarrierDesc residentRecordsToRead{
            .buffer = globalViews.meshletDraws.buffer,
            .before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead},
            .range = {.offset = globalViews.meshletDraws.offset, .size = globalViews.meshletDraws.size},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = {&residentRecordsToRead, 1},
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(
                std::string("end mixed-producer capture returned ") +
                toString(result));
        }
        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = {commandBuffers, 1},
            .signalFence = fence.get(),
        });
        if (result) {
            result = fence->wait(5'000'000'000ull);
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("submit mixed-producer capture returned ") +
                toString(result));
        }

        auto copyReadback = [](
                                render::Buffer& buffer,
                                void* destination,
                                uint64_t size) -> bool {
            buffer.invalidate({0, size});
            const void* mapped = buffer.map();
            if (mapped == nullptr) {
                return false;
            }
            std::memcpy(destination, mapped, static_cast<size_t>(size));
            buffer.unmap();
            return true;
        };
        std::vector<uint32_t> colorPixels(kWidth * kHeight);
        std::vector<uint32_t> visibilityPixels(kWidth * kHeight);
        std::vector<float> depthPixels(kWidth * kHeight);
        std::vector<render::VisibleClusterRecord> residentRecords(
            streamRecordBase);
        if (!copyReadback(
                *colorReadback,
                colorPixels.data(),
                kPixelByteSize) ||
            !copyReadback(
                *visibilityReadback,
                visibilityPixels.data(),
                kPixelByteSize) ||
            !copyReadback(
                *depthReadback,
                depthPixels.data(),
                kPixelByteSize) ||
            !copyReadback(
                *residentRecordReadback,
                residentRecords.data(),
                globalViews.meshletDraws.size)) {
            return RHITestResult::fail(
                "mixed-producer capture did not map all shared surfaces and records");
        }

        uint32_t residentPixelCount = 0;
        uint32_t streamPixelCount = 0;
        uint32_t residentMinX = kWidth;
        uint32_t residentMaxX = 0;
        uint32_t streamMinX = kWidth;
        uint32_t streamMaxX = 0;
        std::unordered_set<uint32_t> residentRecordIds;
        std::unordered_set<uint32_t> streamRecordIds;
        for (uint32_t pixelIndex = 0;
             pixelIndex < kWidth * kHeight;
             ++pixelIndex) {
            const uint32_t packedVisibility = visibilityPixels[pixelIndex];
            const uint32_t encodedRecord =
                packedVisibility >> render::kVisibilityTriangleBits;
            if (encodedRecord == 0) {
                continue;
            }
            const uint32_t recordIndex = encodedRecord - 1u;
            const uint32_t x = pixelIndex % kWidth;
            const uint32_t colorPixel = colorPixels[pixelIndex];
            const uint8_t red = static_cast<uint8_t>(colorPixel & 0xffu);
            const uint8_t green =
                static_cast<uint8_t>((colorPixel >> 8u) & 0xffu);
            const uint8_t blue =
                static_cast<uint8_t>((colorPixel >> 16u) & 0xffu);
            if ((!std::isfinite(depthPixels[pixelIndex])) ||
                depthPixels[pixelIndex] <= 0.0f ||
                depthPixels[pixelIndex] > 1.0f ||
                (red <= 8u && green <= 8u && blue <= 8u)) {
                return RHITestResult::fail(
                    "mixed-producer visibility does not match the shared depth/debug surfaces");
            }

            if (recordIndex < streamRecordBase) {
                const render::VisibleClusterRecord& record =
                    residentRecords[recordIndex];
                if (render::visibleClusterSource(record.flags) !=
                        render::VisibleClusterSource::Resident ||
                    record.instanceIndex != residentInstance.index ||
                    record.instanceIndex == streamInstance.index) {
                    return RHITestResult::fail(
                        "stream-owned geometry leaked into the resident producer namespace");
                }
                ++residentPixelCount;
                residentMinX = std::min(residentMinX, x);
                residentMaxX = std::max(residentMaxX, x);
                residentRecordIds.insert(recordIndex);
                continue;
            }

            const uint64_t localStreamRecord =
                static_cast<uint64_t>(recordIndex) - streamRecordBase;
            if (localStreamRecord >= requestedStreamRecordCapacity) {
                return RHITestResult::fail(
                    "visibility referenced a stream record outside its logical namespace");
            }
            ++streamPixelCount;
            streamMinX = std::min(streamMinX, x);
            streamMaxX = std::max(streamMaxX, x);
            streamRecordIds.insert(recordIndex);
        }

        if (residentPixelCount < 32 || streamPixelCount < 32 ||
            residentRecordIds.empty() || streamRecordIds.empty()) {
            return RHITestResult::fail(
                "mixed-producer frame did not contain both resident and stream visibility IDs: resident=" +
                std::to_string(residentPixelCount) +
                " stream=" + std::to_string(streamPixelCount));
        }
        if (streamMinX > streamMaxX || residentMinX > residentMaxX ||
            streamMaxX >= residentMinX) {
            return RHITestResult::fail(
                "mixed-producer mounted fixtures overlap or were assigned to the wrong producer");
        }

        uint32_t unifiedPassNodeCount = 0;
        for (const render::RenderGraphNodeExecutionStat& node :
             executor.executionStats().nodes) {
            unifiedPassNodeCount +=
                node.type == "VisibilityBufferPass" ? 1u : 0u;
        }
        if (unifiedPassNodeCount != 1) {
            return RHITestResult::fail(
                "mixed producers were not rasterized by one visibility-buffer graph node");
        }

        const auto* visualizationNode = graph.findNode("GPUDriven");
        if (visualizationNode == nullptr) {
            return RHITestResult::fail("mixed-producer visualization node is missing");
        }
        const auto visualizationNodeId = visualizationNode->id;
        const std::array visualizationModes{"triangle", "depth", "coverage", "none"};
        for (size_t configuration = 0; configuration < visualizationModes.size() * 2; ++configuration) {
            const char* mode = visualizationModes[configuration % visualizationModes.size()];
            const bool freezeCullingCamera = configuration >= visualizationModes.size();
            if (!graph.setNodeRuntimeProperty(visualizationNodeId, "visualization", mode) ||
                !graph.setNodeRuntimeProperty(
                    visualizationNodeId, "freezeCullingCamera", freezeCullingCamera) ||
                !graph.setNodeRuntimeProperty(visualizationNodeId, "asyncSoftwareRaster", configuration % 2 == 0) ||
                !executor.syncRuntimeProperties(graph)) {
                return RHITestResult::fail("could not switch mixed-producer visualization");
            }
            result = fence->reset();
            if (result) {
                result = commandPool->reset();
            }
            if (result) {
                result = commandBuffer->begin();
            }
            if (result) {
                result = executor.execute(render::RenderGraphSubmitDesc{.graphicsQueue = graphicsQueue,
                    .computeQueue = device->getQueue(render::QueueType::Compute)});
            }
            for (const char* outputName : {
                     "GPUDriven.color", "GPUDriven.visibility", "GPUDriven.depth"}) {
                if (result) {
                    result = executor.transitionOutput(
                        *commandBuffer, outputName, render::ResourceState::TransferSource);
                }
            }
            if (result) {
                commandBuffer->copyTextureToBuffer(colorCopy);
                commandBuffer->copyTextureToBuffer(visibilityCopy);
                commandBuffer->copyTextureToBuffer(depthCopy);
                result = commandBuffer->end();
            }
            if (result) {
                result = graphicsQueue->submit(render::QueueSubmitDesc{
                    .commandBuffers = {commandBuffers, 1},
                    .signalFence = fence.get(),
                });
            }
            if (result) {
                result = fence->wait(5'000'000'000ull);
            }
            if (!result) {
                return RHITestResult::fail(
                    std::string("capture visualization returned ") + toString(result));
            }
            std::vector<uint32_t> displayPixels(kWidth * kHeight);
            std::vector<uint32_t> currentVisibility(kWidth * kHeight);
            std::vector<float> currentDepth(kWidth * kHeight);
            if (!copyReadback(*colorReadback, displayPixels.data(), kPixelByteSize) ||
                !copyReadback(*visibilityReadback, currentVisibility.data(), kPixelByteSize) ||
                !copyReadback(*depthReadback, currentDepth.data(), kPixelByteSize)) {
                return RHITestResult::fail("could not read visualization surfaces");
            }
            if (currentVisibility != visibilityPixels || currentDepth != depthPixels) {
                size_t idDiff = 0, depthDiff = 0, first = currentDepth.size();
                float maxDepthDiff = 0;
                for (size_t i = 0; i < currentDepth.size(); ++i) {
                    idDiff += currentVisibility[i] != visibilityPixels[i];
                    depthDiff += currentDepth[i] != depthPixels[i];
                    maxDepthDiff = std::max(maxDepthDiff, std::abs(currentDepth[i] - depthPixels[i]));
                    if (first == currentDepth.size() && currentVisibility[i] != visibilityPixels[i]) { first = i; }
                }
                return RHITestResult::fail(
                    std::string("visualization changed raw visibility/depth: ") + mode +
                    " ids=" + std::to_string(idDiff) + " depths=" + std::to_string(depthDiff) +
                    " maxDepthDiff=" + std::to_string(maxDepthDiff) +
                    (first < currentDepth.size() ? " first=" + std::to_string(first) + " id=" +
                        std::to_string(currentVisibility[first]) + "/" + std::to_string(visibilityPixels[first]) : ""));
            }
            for (size_t pixelIndex = 0; pixelIndex < displayPixels.size(); ++pixelIndex) {
                if (std::string_view(mode) == "coverage" &&
                    ((displayPixels[pixelIndex] == 0xffffffffu) !=
                     (visibilityPixels[pixelIndex] != 0u))) {
                    return RHITestResult::fail("coverage visualization does not match raw IDs");
                }
                if (std::string_view(mode) == "none" &&
                    displayPixels[pixelIndex] != displayPixels.front()) {
                    return RHITestResult::fail("disabled visualization still draws color");
                }
            }
        }

        (void)device->waitIdle();
        return RHITestResult::pass(
            "software clusters resident/stream=" + std::to_string(softwareClusters[0]) + "/" + std::to_string(softwareClusters[1]) +
            "; visualization preserves shared visibility/depth; residentPixels=" +
            std::to_string(residentPixelCount) +
            " streamPixels=" + std::to_string(streamPixelCount) +
            " residentRecords=" + std::to_string(residentRecordIds.size()) +
            " streamRecords=" + std::to_string(streamRecordIds.size()));
    }
};

class RenderGraphVisibilityBufferPassRenderTest : public RHITest {
public:
    RenderGraphVisibilityBufferPassRenderTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_driven_preview_pass_render";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(render::EnvironmentSettings{
            .enabled = true,
            .path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/environment.hdr",
            .intensity = 3.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        });
        render::Result<> result = preview.initialize(context.enableValidation, false);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip("RenderGraphPreviewRenderer is unsupported");
            }
            return RHITestResult::fail(
                std::string("RenderGraphPreviewRenderer::initialize returned ") +
                toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenPreviewRender");
        graph.addNode(
            "VisibilityBufferPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", "Asset/StandfordBunny/scene.gltf"},
                {"mode", "meshlet"},
                {"camera", {
                    {"projection", "perspective"},
                    {"fovDegrees", 60.0f},
                    {"znear", 0.1f},
                    {"zfar", 10000.0f},
                    {"reversedZ", true},
                    {"eye", {-0.0168404f, 0.110154f, 0.22f}},
                    {"center", {-0.0168404f, 0.110154f, -0.00153695f}},
                    {"up", {0.0f, 1.0f, 0.0f}},
                }},
            });
        graph.markOutput("GPUDriven.color");

        result = preview.render(graph, 192, 192);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("VisibilityBufferPass is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("VisibilityBufferPass render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }
        const std::vector<uint32_t> firstFramePixels = preview.pixels();

        result = preview.render(graph, 192, 192);
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass second-frame HZB render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t hzbVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (hzbVisiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass second-frame HZB render produced too few visible pixels: ") +
                std::to_string(hzbVisiblePixelCount));
        }
        if (preview.pixels() != firstFramePixels) {
            size_t mismatchCount = 0;
            for (size_t pixelIndex = 0; pixelIndex < firstFramePixels.size(); ++pixelIndex) {
                mismatchCount += preview.pixels()[pixelIndex] != firstFramePixels[pixelIndex] ? 1u : 0u;
            }
            return RHITestResult::fail(
                std::string("VisibilityBufferPass stationary HZB frame changed ") +
                std::to_string(mismatchCount) +
                " pixels");
        }

        render::RenderGraphNode* gpuDrivenNode = graph.findNode("GPUDriven");
        if (gpuDrivenNode == nullptr ||
            !graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.fovDegrees", 20.0f)) {
            return RHITestResult::fail("failed to configure the GPUDriven culling test camera");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass narrow culling-camera render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const std::vector<uint32_t> capturedCullingPixels = preview.pixels();

        if (
            !graph.setNodeRuntimeProperty(gpuDrivenNode->id, "freezeCullingCamera", true)) {
            return RHITestResult::fail("failed to freeze the GPUDriven culling camera");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass frozen-camera capture render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        if (preview.pixels() != capturedCullingPixels) {
            return RHITestResult::fail(
                "freezing the GPUDriven culling camera changed the captured view");
        }

        const render::RenderGraphProperties oppositeEye =
            render::RenderGraphProperties::array({0.22f, 0.110154f, -0.00153695f});
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.eye", oppositeEye)) {
            return RHITestResult::fail("failed to move the GPUDriven observation camera");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass frozen-culling observation render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t frozenVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (frozenVisiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass frozen culling produced too few visible pixels: ") +
                std::to_string(frozenVisiblePixelCount));
        }
        const std::vector<uint32_t> frozenObservationPixels = preview.pixels();

        result = preview.render(graph, 192, 192);
        if (!result || preview.pixels() != frozenObservationPixels) {
            return RHITestResult::fail(
                "VisibilityBufferPass frozen culling camera was not stable while observing from another view");
        }

        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "freezeCullingCamera", false)) {
            return RHITestResult::fail("failed to restore live GPUDriven camera culling");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass restored live-camera render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t liveVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (liveVisiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass restored live culling produced too few visible pixels: ") +
                std::to_string(liveVisiblePixelCount));
        }
        size_t cullingCameraMismatchCount = 0;
        for (size_t pixelIndex = 0; pixelIndex < frozenObservationPixels.size(); ++pixelIndex) {
            cullingCameraMismatchCount +=
                preview.pixels()[pixelIndex] != frozenObservationPixels[pixelIndex] ? 1u : 0u;
        }
        if (cullingCameraMismatchCount < 64) {
            return RHITestResult::fail(
                "disabling the frozen culling camera did not restore view-dependent culling");
        }

        const render::RenderGraphProperties originalEye =
            render::RenderGraphProperties::array({-0.0168404f, 0.110154f, 0.22f});
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.eye", originalEye)) {
            return RHITestResult::fail("failed to restore the GPUDriven observation camera");
        }
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.fovDegrees", 60.0f)) {
            return RHITestResult::fail("failed to restore the GPUDriven camera FOV");
        }

        result = preview.render(graph, 128, 96);
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass resize-down render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t resizedDownVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (resizedDownVisiblePixelCount < 128) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass resize-down render produced too few visible pixels: ") +
                std::to_string(resizedDownVisiblePixelCount));
        }

        result = preview.render(graph, 256, 144);
        if (!result) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass resize-up render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t resizedUpVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (resizedUpVisiblePixelCount < 256) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass resize-up render produced too few visible pixels: ") +
                std::to_string(resizedUpVisiblePixelCount));
        }

        render::RenderGraph lodGraph;
        lodGraph.setName("GPUDrivenPreviewLODRender");
        lodGraph.addNode(
            "VisibilityBufferPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", "Asset/StandfordBunny/scene.gltf"},
                {"mode", "lod"},
                {"lodLevel", 0},
                {"camera", {
                    {"projection", "perspective"},
                    {"fovDegrees", 60.0f},
                    {"znear", 0.1f},
                    {"zfar", 10000.0f},
                    {"reversedZ", true},
                    {"eye", {-0.0168404f, 0.110154f, 0.22f}},
                    {"center", {-0.0168404f, 0.110154f, -0.00153695f}},
                    {"up", {0.0f, 1.0f, 0.0f}},
                }},
            });
        lodGraph.markOutput("GPUDriven.color");

        result = preview.render(lodGraph, 192, 192);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("VisibilityBufferPass LOD mode is unsupported on this device: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("VisibilityBufferPass LOD render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        const uint32_t lodVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (lodVisiblePixelCount < 512) {
            return RHITestResult::fail(
                std::string("VisibilityBufferPass LOD mode produced too few visible pixels: ") +
                std::to_string(lodVisiblePixelCount));
        }

        return RHITestResult::pass();
    }
};

class RenderGraphGPUDrivenAlphaMaskRenderTest : public RHITest {
public:
    RenderGraphGPUDrivenAlphaMaskRenderTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_driven_alpha_mask_render";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::filesystem::path scenePath;
        std::string message;
        if (!writeAlphaMaskScene(
                context.outputDirectory / "gpu-driven-alpha-mask-scene",
                scenePath,
                message)) {
            return RHITestResult::fail(message);
        }
        std::filesystem::path singleSidedScenePath;
        if (!writeAlphaMaskScene(
                context.outputDirectory / "gpu-driven-alpha-mask-single-sided-scene",
                singleSidedScenePath,
                message,
                false)) {
            return RHITestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(render::EnvironmentSettings{
            .enabled = false,
            .intensity = 0.0f,
            .visible = false,
        });
        render::Result<> result = preview.initialize(context.enableValidation, false);
        if (!result) {
            return render::hasError(result, render::Error::Unsupported)
                ? RHITestResult::skip("GPUDriven alpha-mask preview is unsupported")
                : RHITestResult::fail(
                      std::string("RenderGraphPreviewRenderer::initialize returned ") +
                      toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenAlphaMaskRender");
        graph.addNode(
            "VisibilityBufferPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", scenePath.string()},
                {"visualization", "coverage"},
                {"instanceHzbCull", false},
                {"meshletNormalConeCull", false},
                {"camera", {
                    {"projection", "perspective"},
                    {"fovDegrees", 60.0f},
                    {"znear", 0.01f},
                    {"zfar", 10.0f},
                    {"reversedZ", true},
                    {"eye", {0.0f, 0.0f, 2.0f}},
                    {"center", {0.0f, 0.0f, 0.0f}},
                    {"up", {0.0f, 1.0f, 0.0f}},
                }},
            });
        graph.markOutput("GPUDriven.color");

        auto classifyPixels = [&preview]() {
            std::array<uint32_t, 3> counts{};
            for (uint32_t pixel : preview.pixels()) {
                const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
                const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
                const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
                if (r == 255 && g == 255 && b == 255) {
                    ++counts[0];
                }
                if (r != g && (r >= 32 || g >= 32 || b >= 32)) {
                    ++counts[1];
                }
                if (r < 32 && g < 32 && b < 32) {
                    ++counts[2];
                }
            }
            return counts;
        };

        result = preview.render(graph, 128, 128);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("VisibilityBufferPass is unsupported: ") +
                    preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("GPUDriven alpha-mask front render returned ") +
                toString(result) + ": " + preview.lastLog());
        }
        const std::array<uint32_t, 3> front = classifyPixels();
        if (front[0] < 1024 || front[0] > 7500 || front[2] < 1024 || front[1] > 64) {
            return RHITestResult::fail(
                "GPUDriven MASK/BLEND classification is incorrect on the front face: coverage=" +
                std::to_string(front[0]) + " unexpected-color=" + std::to_string(front[1]) +
                " dark=" + std::to_string(front[2]));
        }

        render::RenderGraphNode* node = graph.findNode("GPUDriven");
        if (node == nullptr ||
            !graph.setNodeRuntimeProperty(
                node->id,
                "camera.eye",
                render::RenderGraphProperties::array({0.0f, 0.0f, -2.0f}))) {
            return RHITestResult::fail("failed to move the GPUDriven camera behind the double-sided MASK quad");
        }
        result = preview.render(graph, 128, 128);
        if (!result) {
            return RHITestResult::fail(
                std::string("GPUDriven alpha-mask back render returned ") +
                toString(result) + ": " + preview.lastLog());
        }
        const std::array<uint32_t, 3> back = classifyPixels();
        if (back[0] < 1024 || back[0] > 7500 || back[2] < 1024 || back[1] > 64) {
            return RHITestResult::fail(
                "GPUDriven double-sided MASK did not survive back-face rendering: coverage=" +
                std::to_string(back[0]) + " unexpected-color=" + std::to_string(back[1]) +
                " dark=" + std::to_string(back[2]));
        }

        render::RenderGraph singleSidedGraph;
        singleSidedGraph.setName("GPUDrivenSingleSidedMaskRender");
        singleSidedGraph.addNode(
            "VisibilityBufferPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", singleSidedScenePath.string()},
                {"visualization", "coverage"},
                {"instanceHzbCull", false},
                {"meshletNormalConeCull", false},
                {"camera", {
                    {"projection", "perspective"},
                    {"fovDegrees", 60.0f},
                    {"znear", 0.01f},
                    {"zfar", 10.0f},
                    {"reversedZ", true},
                    {"eye", {0.0f, 0.0f, -2.0f}},
                    {"center", {0.0f, 0.0f, 0.0f}},
                    {"up", {0.0f, 1.0f, 0.0f}},
                }},
            });
        singleSidedGraph.markOutput("GPUDriven.color");
        result = preview.render(singleSidedGraph, 128, 128);
        if (!result) {
            return RHITestResult::fail(
                std::string("GPUDriven single-sided MASK back render returned ") +
                toString(result) + ": " + preview.lastLog());
        }
        const std::array<uint32_t, 3> singleSidedBack = classifyPixels();
        if (singleSidedBack[0] > 64 || singleSidedBack[1] > 64 ||
            singleSidedBack[2] < 4096) {
            return RHITestResult::fail(
                "GPUDriven single-sided MASK was not back-face culled: coverage=" +
                std::to_string(singleSidedBack[0]) + " unexpected-color=" +
                std::to_string(singleSidedBack[1]) + " dark=" +
                std::to_string(singleSidedBack[2]));
        }
        return RHITestResult::pass();
    }
};

class RenderGraphGPUDrivenSponzaVisibilityRenderTest : public RHITest {
public:
    RenderGraphGPUDrivenSponzaVisibilityRenderTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_driven_sponza_visibility_render";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const std::filesystem::path sponzaPath =
            std::filesystem::path(PROJECT_SOURCE_DIR) /
            "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf";
        if (!std::filesystem::is_regular_file(sponzaPath)) {
            return RHITestResult::skip("SuperSponza glTF is not present");
        }

        render::RenderGraphPreviewRenderer preview;
        render::EnvironmentSettings environment{
            .enabled = true,
            .path = std::filesystem::path(PROJECT_SOURCE_DIR) /
                "Asset/ABeautifulGame/environment.hdr",
            .intensity = 3.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
        preview.setEnvironment(environment);
        render::Result<> result = preview.initialize(context.enableValidation, false);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip("RenderGraphPreviewRenderer is unsupported");
            }
            return RHITestResult::fail(
                std::string("RenderGraphPreviewRenderer::initialize returned ") +
                toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenSponzaVisibilityRender");
        graph.addNode(
            "VisibilityBufferPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"},
                {"materialTextureBudgetMiB", 8192},
                {"visualization", "meshlet"},
                {"instanceFrustumCull", true},
                {"instanceHzbCull", true},
                {"meshletFrustumCull", true},
                {"meshletNormalConeCull", true},
                {"camera", {
                    {"eye", {5.433790f, 5.599402f, 1.739370f}},
                    {"center", {5.630164f, 5.576646f, 1.765344f}},
                    {"up", {0.0f, 1.0f, 0.0f}},
                    {"projection", "perspective"},
                    {"fovDegrees", 60.0f},
                    {"znear", 0.1f},
                    {"zfar", 10000.0f},
                    {"reversedZ", true},
                }},
            });
        graph.markOutput("GPUDriven.color");

        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(
                    std::string("SuperSponza visibility rendering is unsupported: ") + preview.lastLog());
            }
            return RHITestResult::fail(
                std::string("SuperSponza visibility render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }


        const std::vector<uint32_t> meshletPixels = preview.pixels();
        if (countVisiblePixels(meshletPixels) < 2048) {
            return RHITestResult::fail("Sponza visibility contains too few covered pixels");
        }
        const auto memory = preview.subsystemHost()->device()->memoryBudget();
        const auto& textures = memory.domains[size_t(render::MemoryBudgetDomain::MaterialTextures)];
        const auto& heap = memory.heaps[memory.primaryDeviceLocalHeap];
        std::ofstream(context.outputDirectory / "SuperSponzaResidentAllocation.json") <<
            nlohmann::json{{"materialAllocationBytes", textures.allocationBytes},
                {"materialPeakAllocationBytes", textures.peakAllocationBytes}, {"materialImages", textures.allocationCount},
                {"localHeapUsageBytes", heap.usageBytes}, {"localHeapBlockBytes", heap.blockBytes},
                {"localHeapAllocationBytes", heap.allocationBytes}, {"coveredPixels", countVisiblePixels(meshletPixels)}}.dump(2) << '\n';
        std::string imageLog;
        if (!saveRgba8Png(context.outputDirectory / "SuperSponzaResident-meshlet.png",
                reinterpret_cast<const uint8_t*>(meshletPixels.data()), 256, 256, imageLog)) {
            return RHITestResult::fail(imageLog);
        }
        result = preview.render(graph, 256, 256);
        if (!result || preview.pixels() != meshletPixels) {
            return RHITestResult::fail("stationary HZB changed meshlet-ID visualization");
        }
        // Lighting is intentionally irrelevant to this pass, even with an
        // invalid environment path; no environment subsystem should activate.
        environment.path = "Asset/does-not-exist.hdr";
        environment.rotationDegrees = 90.0f;
        environment.intensity = 100.0f;
        preview.setEnvironment(environment);
        result = preview.render(graph, 256, 256);
        if (!result || preview.pixels() != meshletPixels ||
            preview.subsystemHost()->get<render::EnvironmentLightingSubsystem>() != nullptr) {
            return RHITestResult::fail("visibility rasterization still depends on environment shading");
        }
        result = preview.render(graph, 256, 256, "GPUDriven.visibility");
        if (!result || preview.pixels().size() != meshletPixels.size()) {
            return RHITestResult::fail("Sponza visibility-ID readback failed");
        }
        const std::vector<uint32_t> visibilityPixels = preview.pixels();
        std::ofstream visibilityFile(context.outputDirectory / "SuperSponzaResident-visibility.bin", std::ios::binary);
        visibilityFile.write(reinterpret_cast<const char*>(visibilityPixels.data()),
            visibilityPixels.size() * sizeof(uint32_t));
        if (!visibilityFile) {
            return RHITestResult::fail("failed to save Sponza visibility-ID readback");
        }
        size_t legacyCoverageMismatch = 0;
        size_t lowRedCoveredPixels = 0;
        for (size_t i = 0; i < visibilityPixels.size(); ++i) {
            // Match GPUDrivenRasterCommon::unpackVisibilityId. Shaded ID colors
            // can have a red channel below 32 even when the visibility is valid.
            const bool covered = (visibilityPixels[i] >> render::kVisibilityTriangleBits) != 0u;
            const bool legacyCovered = (meshletPixels[i] & 0xffu) >= 32u;
            legacyCoverageMismatch += covered != legacyCovered ? 1u : 0u;
            lowRedCoveredPixels += covered && !legacyCovered ? 1u : 0u;
        }
        nlohmann::json comparison{{"pixels", visibilityPixels.size()},
            {"legacyRedCoverageMismatch", legacyCoverageMismatch}, {"lowRedCoveredPixels", lowRedCoveredPixels},
            {"minimumCoveredPixels", 2048}, {"minimumTriangleDifferences", 1024}, {"modes", nlohmann::json::array()}};
        render::RenderGraphNode* node = graph.findNode("GPUDriven");
        if (node == nullptr) {
            return RHITestResult::fail("Sponza visibility node is missing");
        }
        std::string message;
        for (const char* mode : {"meshlet", "triangle", "depth", "coverage", "none"}) {
            if (!graph.setNodeRuntimeProperty(node->id, "visualization", mode)) {
                return RHITestResult::fail("failed to select visibility visualization");
            }
            result = preview.render(graph, 256, 256);
            if (!result) {
                return RHITestResult::fail(std::string("Sponza visualization ") + mode +
                    " returned " + toString(result) + ": " + preview.lastLog());
            }
            const auto& pixels = preview.pixels();
            size_t different = 0;
            size_t covered = 0;
            size_t coverageMismatch = 0;
            for (size_t i = 0; i < pixels.size(); ++i) {
                const bool wasCovered = (visibilityPixels[i] >> render::kVisibilityTriangleBits) != 0u;
                const uint32_t rgb = pixels[i] & 0x00ffffffu;
                covered += wasCovered ? 1u : 0u;
                different += pixels[i] != meshletPixels[i] ? 1u : 0u;
                if (std::string_view(mode) == "coverage" &&
                    ((rgb == 0x00ffffffu) != wasCovered)) {
                    ++coverageMismatch;
                }
                if (std::string_view(mode) == "depth" && wasCovered &&
                    (((rgb >> 8u) & 0xffu) != (rgb & 0xffu) ||
                        ((rgb >> 16u) & 0xffu) != (rgb & 0xffu))) {
                    return RHITestResult::fail("device-depth visualization is not grayscale");
                }
                if (std::string_view(mode) == "none" && pixels[i] != pixels.front()) {
                    return RHITestResult::fail("disabled visualization did not clear the debug output");
                }
            }
            if (covered < 2048 || (std::string_view(mode) == "triangle" && different < 1024)) {
                return RHITestResult::fail("triangle-ID visualization did not distinguish triangles");
            }
            if (!saveRgba8Png(
                    context.outputDirectory / (std::string("visibility_buffer_sponza_") + mode + ".png"),
                    reinterpret_cast<const uint8_t*>(pixels.data()), 256, 256, message)) {
                return RHITestResult::fail(message);
            }
            comparison["modes"].push_back({{"mode", mode}, {"coveredPixels", covered},
                {"differentPixels", different}, {"coverageMismatch", coverageMismatch}});
            std::ofstream(context.outputDirectory / "SuperSponzaResidentVisibilityComparison.json") << comparison.dump(2) << '\n';
            if (coverageMismatch != 0) {
                return RHITestResult::fail("coverage display differs from visibility coverage");
            }
        }
        graph.setNodeRuntimeProperty(node->id, "visualization", "meshlet");
        for (uint32_t frame = 0; frame < 2; ++frame) {
            result = preview.render(graph, 256, 133);
            if (!result || preview.pixels().size() != 256u * 133u ||
                countVisiblePixels(preview.pixels()) < 1024) {
                return RHITestResult::fail("Sponza non-square visibility/HZB regression");
            }
        }
        return RHITestResult::pass("validated unshaded ID/depth/coverage/off visualizations");
    }
};

class RenderGraphStreamedAsyncAccelerationStructureTest : public RHITest {
public:
    RenderGraphStreamedAsyncAccelerationStructureTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_streamed_async_acceleration_structure";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        constexpr uint32_t kWidth = 160, kHeight = 96;
        const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        const auto cache = std::filesystem::absolute(context.outputDirectory / "AsyncRtasBunny.meshstream.bin");
        std::string log;
        if (!scene::buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = cache,
                .meshletOptions = {.maxWorkers = 1}}, log)) {
            return RHITestResult::fail(log);
        }
        RenderSampleLoadResult sample;
        if (!loadBuiltInRenderSample(kDefaultGPUDrivenSampleId, sample, log) ||
            !setRenderSampleScenePath(sample, source.generic_string(), log)) {
            return RHITestResult::fail(log);
        }
        auto& graph = sample.graph;
        auto* visibility = graph.findNode("VBuffer");
        if (!visibility) { return RHITestResult::fail("streamed sample has no visibility producer"); }
        const auto visibilityId = visibility->id;
        auto& props = visibility->properties;
        props["streamAssetPath"] = cache.generic_string();
        props["enableClusterRtx"] = true;
        props["maxResidentPages"] = 64; props["maxResidentBytes"] = 16777216;
        props["maxLockedFallbackPages"] = 64; props["maxActiveGroups"] = 1024;
        props["maxClasBytes"] = 16777216; props["maxClasBuildClusters"] = 2048;
        props["maxBlasBytes"] = 16777216; props["maxFallbackBlasBytes"] = 16777216;
        props["maxGpuPageRequests"] = 1024; props["maxGpuPageUnloadRequests"] = 1024;
        props["maxTraversalWorkers"] = 32; props["maxTraversalWorkItems"] = 2048;
        props["autoLod"] = false; props["lodLevel"] = 0;
        props["instanceFrustumCull"] = false; props["instanceHzbCull"] = false;
        props["meshletFrustumCull"] = false; props["hybridRaster"] = false;
        graph.findNode("Shadows")->properties["sigmaDenoise"] = false;
        graph.findNode("Shadows")->properties["shadowAngularRadius"] = 0.0;
        for (const char* name : {"DLSSSR", "DLSSNR", "AutoExposure"}) {
            if (auto* node = graph.findNode(name)) { graph.removeNode(node->id); }
        }
        graph.addEdge("Deferred.color", "FinalBlit.source");
        graph.addEdge("VBuffer.accelerationStructure", "Deferred.accelerationStructure");
        graph.addEdge("VBuffer.accelerationStructure", "Shadows.accelerationStructure");
        graph.markOutput("FinalBlit.color");
        scene::Scene fixture;
        if (!fixture.loadStreamMetadata(source)) { return RHITestResult::fail(fixture.lastLoadResult().error); }
        const auto center = fixture.bounds().center();
        const auto radius = fixture.bounds().radius();
        graph.setViewProperties({{"camera", {{"eye", {center.x, center.y + radius * .3f, center.z + radius * 3}},
            {"center", {center.x, center.y, center.z}}, {"znear", .01}, {"zfar", 100},
            {"fovDegrees", 50}, {"reversedZ", true}}}, {"temporalJitter", false}});
        scene::LightingSettings lighting;
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 2;
        scene::PunctualLight sun;
        sun.properties.type = "directional"; sun.properties.intensity = 10;
        sun.direction = float3(.6f, -1, -.3f);
        lighting.lights.push_back(sun);
        RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, true, false);
        if (!result) {
            return hasError(result, Error::Unsupported) ? RHITestResult::skip("streamed RTAS features unavailable") :
                RHITestResult::fail(std::string("preview initialize returned ") + toString(result));
        }
        preview.setEnvironment({.enabled = false});
        preview.setLighting(lighting);
        preview.setExecutionCaptureEnabled(true);
        for (uint32_t frame = 0; frame < 24; ++frame) {
            result = preview.render(graph, kWidth, kHeight, "FinalBlit.color");
            if (!result) {
                return hasError(result, Error::Unsupported) ? RHITestResult::skip(preview.lastLog()) :
                    RHITestResult::fail("async RTAS render failed: " + preview.lastLog());
            }
        }
        const auto* streamer = preview.subsystemHost()->get<StreamerSubsystem>();
        if (!streamer || !streamer->sceneReadiness().ready) {
            return RHITestResult::fail("streamed RTAS fixture did not finish fallback publication");
        }
        // Keep timing informational: a small correctness fixture does not
        // establish a production speedup. Both modes use the same warmed graph.
        RenderGraphProperties timings = RenderGraphProperties::array();
        const auto measure = [&](bool preferred) -> Result<> {
            for (uint32_t frame = 0; frame < 32; ++frame) {
                const auto started = std::chrono::steady_clock::now();
                auto measured = preview.render(graph, kWidth, kHeight, "FinalBlit.color");
                const double wallMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                if (!measured) { return measured; }
                const auto& stats = preview.executionStats();
                RenderGraphProperties sampleTiming{{"AsyncComputePreferred", preferred},
                    {"sample", frame}, {"renderAndReadbackWallMilliseconds", wallMs}};
                if (stats.gpuTimingAvailable) {
                    sampleTiming["graphGpuMilliseconds"] = stats.gpuMilliseconds;
                }
                timings.push_back(std::move(sampleTiming));
            }
            return {};
        };
        if (!measure(true)) { return RHITestResult::fail("async RTAS timing frames failed: " + preview.lastLog()); }
        const auto asynchronous = preview.pixels();
        if (asynchronous.empty() || std::count_if(asynchronous.begin(), asynchronous.end(),
                [&](uint32_t pixel) { return pixel != asynchronous.front(); }) < 64) {
            return RHITestResult::fail("async RTAS deferred image contains no useful geometry");
        }
        auto snapshot = preview.executionSnapshot();
        if (!snapshot || !snapshot->success) { return RHITestResult::fail("async RTAS capture is missing"); }
        const RenderGraphExecutionResourceSnapshot* accelerationStructure = nullptr;
        const RenderGraphExecutionPassSnapshot* producer = nullptr;
        for (const auto& resource : snapshot->resources) {
            if (resource.type == RenderGraphResourceType::AccelerationStructure &&
                resource.name == "VBuffer.accelerationStructure") { accelerationStructure = &resource; }
        }
        for (const auto& pass : snapshot->passes) { if (pass.name == "VBuffer") { producer = &pass; } }
        if (!accelerationStructure || !producer || !accelerationStructure->memory.allocationId) {
            return RHITestResult::fail("stream TLAS is missing from first-class graph allocation capture");
        }
        bool buildWrite = false, consumerRead = false, consumerBarrier = false;
        for (const auto& use : producer->uses) {
            buildWrite |= use.resourceId == accelerationStructure->id && use.writes &&
                (uint64_t(use.scope.access) & uint64_t(AccessBits::AccelerationStructureWrite));
        }
        for (const auto& pass : snapshot->passes) {
            if (pass.name != "Shadows" && pass.name != "Deferred") { continue; }
            const bool producerDependency = std::find(pass.predecessors.begin(), pass.predecessors.end(), producer->id) != pass.predecessors.end();
            for (const auto& use : pass.uses) {
                consumerRead |= producerDependency && use.resourceId == accelerationStructure->id && use.reads &&
                    (uint64_t(use.scope.access) & uint64_t(AccessBits::AccelerationStructureRead));
            }
            for (const auto& barrier : pass.barriers) {
                consumerBarrier |= barrier.resourceId == accelerationStructure->id &&
                    (uint64_t(barrier.beforeScope.access) & uint64_t(AccessBits::AccelerationStructureWrite)) &&
                    (uint64_t(barrier.afterScope.access) & uint64_t(AccessBits::AccelerationStructureRead));
            }
        }
        if (!buildWrite || !consumerRead || !consumerBarrier) {
            return RHITestResult::fail("AS build-to-ray-query graph dependency or inferred barrier is missing");
        }
        const auto hasRtasBranch = [&](const RenderGraphExecutionSnapshot& capture) {
            return std::any_of(capture.segments.begin(), capture.segments.end(), [&](const auto& segment) {
                return segment.passId == producer->id && segment.role == RenderGraphSegmentRole::ComputeBranch;
            });
        };
        const bool independentCompute = std::any_of(snapshot->queues.begin(), snapshot->queues.end(),
            [](const auto& queue) { return queue.type == QueueType::Compute; });
        if (independentCompute && !hasRtasBranch(*snapshot)) {
            return RHITestResult::fail("default RTAS preference did not fork the compute build");
        }
        if (!graph.setNodeRuntimeProperty(visibilityId, "AsyncComputePreferred", false)) {
            return RHITestResult::fail("failed to disable async RTAS preference");
        }
        for (uint32_t frame = 0; frame < 24; ++frame) {
            if (!preview.render(graph, kWidth, kHeight, "FinalBlit.color")) {
                return RHITestResult::fail("graphics RTAS fallback render failed: " + preview.lastLog());
            }
        }
        if (!measure(false)) { return RHITestResult::fail("graphics RTAS timing frames failed: " + preview.lastLog()); }
        std::ofstream(context.outputDirectory / "StreamedAsyncRtasTiming.json") <<
            RenderGraphProperties{{"width", kWidth}, {"height", kHeight},
                {"warmupFramesPerMode", 24}, {"measuredScope", "Headless render and readback wall time; optional graph GPU envelope"},
                {"samples", std::move(timings)}}.dump(2) << '\n';
        const auto fallback = preview.executionSnapshot();
        if (!fallback || !fallback->success || hasRtasBranch(*fallback)) {
            return RHITestResult::fail("disabled RTAS preference still forked the compute build");
        }
        if (preview.pixels() != asynchronous) {
            return RHITestResult::fail("async and graphics RTAS builds produced different deferred/shadow output");
        }
        if (!saveRgba8Png(context.outputDirectory / "StreamedAsyncRtas.png",
                reinterpret_cast<const uint8_t*>(asynchronous.data()), kWidth, kHeight, log)) {
            return RHITestResult::fail(log);
        }
        return RHITestResult::pass(independentCompute ?
            "Async stream TLAS, first-class AS edges/barriers and flag-disabled output equality" :
            "Compute queue unavailable: validated AS graph barriers and graphics fallback output equality");
    }
};

class ImportancePdfSizeTest : public RHITest {
public:
    ImportancePdfSizeTest()
    {
        type = RHITestType::Resource;
        name = "importance_pdf_size";
    }

    RHITestResult run(RHITestContext&) override
    {
        const render::ImportancePdfSize lightPdfSize = render::computeImportancePdfTextureSize(257);
        if (lightPdfSize.width != 32 || lightPdfSize.height != 16 || lightPdfSize.mipCount != 6) {
            return RHITestResult::fail("RTXDI local-light PDF sizing does not match a power-of-two rectangle");
        }
        return RHITestResult::pass("validated GPU PDF texture sizing");
    }
};

class ReGIRGridLayoutTest : public RHITest {
public:
    ReGIRGridLayoutTest()
    {
        type = RHITestType::Resource;
        name = "regir_grid_layout";
    }

    RHITestResult run(RHITestContext&) override
    {
        const render::ReGIRGridLayout layout = render::computeReGIRGridLayout(12, 64);
        if (!layout.valid() ||
            layout.cellCount != 1728 ||
            layout.lightSlotCount != 110592 ||
            layout.bufferByteSize !=
                static_cast<uint64_t>(110592 + render::kReGIRHeaderRecordCount) *
                    render::kReGIRRecordByteSize) {
            return RHITestResult::fail("ReGIR grid layout or buffer sizing is incorrect");
        }

        if (render::computeReGIRGridLayout(0, 64).valid() ||
            render::computeReGIRGridLayout(12, 0).valid() ||
            render::computeReGIRGridLayout(UINT32_MAX, UINT32_MAX).valid()) {
            return RHITestResult::fail("invalid ReGIR grid parameters were accepted");
        }
        return RHITestResult::pass("validated ReGIR cell, slot, and buffer layout");
    }
};

class EnvironmentSubsystemAsyncSnapshotTest : public RHITest {
public:
    EnvironmentSubsystemAsyncSnapshotTest()
    {
        type = RHITestType::Rendering;
        name = "environment_subsystem_async_snapshot";
    }

    RHITestResult run(RHITestContext&) override
    {
        registerTestPass();
        render::RenderGraph graph;
        graph.addNode("TestEnvironmentConsumerPass", "EnvironmentConsumerA");
        graph.addNode("TestEnvironmentConsumerPass", "EnvironmentConsumerB");
        if (!graph.markOutput("EnvironmentConsumerA.color") ||
            !graph.markOutput("EnvironmentConsumerB.color")) {
            return RHITestResult::fail("failed to construct the shared environment graph");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(render::EnvironmentSettings{});
        render::Result<> result = preview.initialize(false, false);
        if (!result) {
            return RHITestResult::skip(
                std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }
        result = preview.render(graph, 16, 16, "EnvironmentConsumerA.color");
        if (!result) {
            return RHITestResult::fail("initial environment render failed: " + preview.lastLog());
        }

        render::EnvironmentLightingSubsystem* subsystem =
            preview.subsystemHost()->get<render::EnvironmentLightingSubsystem>();
        if (subsystem == nullptr ||
            !subsystem->snapshot().valid() ||
            subsystem->snapshot().pdfView == nullptr) {
            return RHITestResult::fail("environment subsystem did not publish its black fallback");
        }
        const render::TextureView* fallbackView = subsystem->snapshot().radianceView;
        const uint64_t fallbackRevision = subsystem->snapshot().resourceRevision;

        result = preview.render(graph, 24, 16, "EnvironmentConsumerA.color");
        if (!result ||
            subsystem->snapshot().radianceView != fallbackView ||
            subsystem->snapshot().resourceRevision != fallbackRevision) {
            return RHITestResult::fail("RenderGraph resize recreated the active environment resource");
        }

        render::EnvironmentSettings environment;
        environment.path = std::filesystem::path(PROJECT_SOURCE_DIR) /
            "Asset/ABeautifulGame/environment.hdr";
        preview.setEnvironment(environment);
        result = preview.render(graph, 24, 16, "EnvironmentConsumerA.color");
        if (!result ||
            subsystem->snapshot().radianceView != fallbackView ||
            subsystem->snapshot().resourceRevision != fallbackRevision) {
            return RHITestResult::fail("environment resource changed before asynchronous decode completed");
        }

        bool switched = false;
        for (uint32_t attempt = 0; attempt < 5000 && !switched; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            result = preview.render(graph, 24, 16, "EnvironmentConsumerA.color");
            if (!result) {
                return RHITestResult::fail("environment switch render failed: " + preview.lastLog());
            }
            const render::EnvironmentLightingSnapshot& snapshot = subsystem->snapshot();
            switched = snapshot.status == render::EnvironmentLightingStatus::Ready &&
                snapshot.mapAvailable &&
                snapshot.pdfView != nullptr &&
                snapshot.resourceRevision > fallbackRevision;
        }
        if (!switched || subsystem->decodeCount() != 1u) {
            return RHITestResult::fail(
                "shared environment did not complete exactly one HDR decode: status=" +
                std::to_string(static_cast<uint32_t>(subsystem->snapshot().status)) +
                " decodeCount=" + std::to_string(subsystem->decodeCount()) +
                " revision=" + std::to_string(subsystem->snapshot().resourceRevision) +
                " error=" + subsystem->snapshot().error);
        }

        const render::TextureView* readyView = subsystem->snapshot().radianceView;
        const uint64_t readyRevision = subsystem->snapshot().resourceRevision;
        environment.path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/does-not-exist.hdr";
        preview.setEnvironment(environment);
        bool degraded = false;
        for (uint32_t attempt = 0; attempt < 100 && !degraded; ++attempt) {
            result = preview.render(graph, 24, 16, "EnvironmentConsumerA.color");
            if (!result) {
                return RHITestResult::fail("degraded environment render failed: " + preview.lastLog());
            }
            const render::EnvironmentLightingSnapshot& snapshot = subsystem->snapshot();
            degraded = snapshot.status == render::EnvironmentLightingStatus::Degraded;
            if (!degraded) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (!degraded ||
            subsystem->snapshot().radianceView != readyView ||
            subsystem->snapshot().resourceRevision != readyRevision ||
            subsystem->decodeCount() != 2u) {
            return RHITestResult::fail("failed environment switch did not preserve the last ready snapshot");
        }
        return RHITestResult::pass();
    }
};

class RenderGraphMissingSubsystemDiagnosticTest : public RHITest {
public:
    RenderGraphMissingSubsystemDiagnosticTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_missing_subsystem_diagnostic";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTestPass();
        render::RenderGraph graph;
        graph.addNode("TestMissingSubsystemPass", "MissingSubsystemUser");
        if (!graph.markOutput("MissingSubsystemUser.color")) {
            return RHITestResult::fail("failed to construct missing-subsystem graph");
        }

        render::RenderGraphExecutor executor;
        std::string log;
        const render::Result<> result = executor.compile(context.device, graph, 16, 16, log);
        if (result ||
            log.find("MissingSubsystemUser") == std::string::npos ||
            log.find("test.missing-required-subsystem") == std::string::npos) {
            return RHITestResult::fail("missing subsystem diagnostic did not name the pass and subsystem: " + log);
        }
        return RHITestResult::pass();
    }
};

class RenderSubsystemHostLifecycleTest : public RHITest {
public:
    RenderSubsystemHostLifecycleTest()
    {
        type = RHITestType::Resource;
        name = "render_subsystem_host_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        struct Probe final : render::IRenderSubsystem {
            Probe(std::string name, std::vector<std::string>& events, bool failBegin = false)
                : name(std::move(name)), events(events), failBegin(failBegin)
            {
            }

            render::Result<> initialize(const render::RenderSubsystemInitContext&, std::string&) override
            {
                events.push_back("init:" + name);
                return {};
            }

            render::Result<> beginFrame(
                const render::RenderSubsystemFrameContext&,
                render::RenderChangeBits&,
                std::string& log) override
            {
                events.push_back("begin:" + name);
                if (failBegin) {
                    log = "probe failure";
                    return render::makeError(render::Error::Failure);
                }
                return {};
            }

            void endFrame(const render::RenderSubsystemFrameContext&) override
            {
                events.push_back("end:" + name);
            }

            void shutdown() override
            {
                events.push_back("shutdown:" + name);
            }

            std::string name;
            std::vector<std::string>& events;
            bool failBegin = false;
        };

        auto registerProbe = [](
                                 render::RenderSubsystemHost& host,
                                 std::string id,
                                 std::vector<std::string> dependencies,
                                 std::vector<std::string>& events,
                                 std::string& log,
                                 bool failBegin = false) {
            const std::string probeName = id;
            return host.registerSubsystem(
                render::RenderSubsystemRegistration{
                    .id = std::move(id),
                    .dependencies = std::move(dependencies),
                    .factory = [probeName, &events, failBegin]() {
                        return std::make_unique<Probe>(probeName, events, failBegin);
                    },
                },
                log);
        };

        std::vector<std::string> events;
        std::string log;
        render::RenderSubsystemHost host;
        if (!registerProbe(host, "test.base", {}, events, log) ||
            !registerProbe(host, "test.consumer", {"test.base"}, events, log)) {
            return RHITestResult::fail(log);
        }
        if (registerProbe(host, "test.base", {}, events, log)) {
            return RHITestResult::fail("duplicate subsystem id was accepted");
        }
        log.clear();
        render::Result<> result = host.initialize(context.device, 3, log);
        if (!result) {
            return RHITestResult::fail(log);
        }
        result = host.activate("test.consumer", log);
        if (!result || events != std::vector<std::string>{"init:test.base", "init:test.consumer"}) {
            return RHITestResult::fail("dependency initialization order is incorrect: " + log);
        }
        result = host.activate("test.consumer", log);
        if (!result || events.size() != 2u) {
            return RHITestResult::fail("active subsystem was not kept warm");
        }

        if (!host.registerSubsystem<ConfigurableRenderSubsystemProbe>(log) ||
            !host.configure<ConfigurableRenderSubsystemProbe>({.value = 42}, log) ||
            !host.activate(ConfigurableRenderSubsystemProbe::kSubsystemId, log)) {
            return RHITestResult::fail("subsystem configuration failed: " + log);
        }
        const ConfigurableRenderSubsystemProbe* configured =
            host.get<ConfigurableRenderSubsystemProbe>();
        if (configured == nullptr || configured->observedValue != 42 ||
            host.configure<ConfigurableRenderSubsystemProbe>({.value = 7}, log)) {
            return RHITestResult::fail("subsystem configuration was not applied before activation");
        }

        render::RenderWorld world;
        host.setWorld(&world);
        render::EnvironmentSettings environment;
        environment.intensity = 2.0f;
        world.setEnvironment(environment);
        result = host.beginFrame(7, 1, nullptr, log);
        if (!result ||
            !render::hasRenderChange(host.lastChanges(), render::RenderChangeBits::Lighting) ||
            !render::hasRenderChange(
                host.lastChanges(),
                render::RenderChangeBits::InvalidateTemporalHistory)) {
            return RHITestResult::fail("world change bits were not aggregated");
        }
        host.endFrame();
        result = host.beginFrame(8, 2, nullptr, log);
        if (!result || host.lastChanges() != render::RenderChangeBits::None) {
            return RHITestResult::fail("world change bits were not consumed exactly once");
        }
        host.endFrame();
        host.shutdown();
        const std::vector<std::string> expectedTail{
            "shutdown:test.consumer",
            "shutdown:test.base",
        };
        if (events.size() < expectedTail.size() ||
            !std::equal(expectedTail.begin(), expectedTail.end(), events.end() - expectedTail.size())) {
            return RHITestResult::fail("subsystems did not shut down in reverse dependency order");
        }

        render::RenderSubsystemHost missingHost;
        if (!registerProbe(missingHost, "test.missing-user", {"test.not-registered"}, events, log)) {
            return RHITestResult::fail(log);
        }
        result = missingHost.initialize(context.device, 1, log);
        if (!result) {
            return RHITestResult::fail(log);
        }
        result = missingHost.activate("test.missing-user", log);
        if (result || log.find("test.not-registered") == std::string::npos) {
            return RHITestResult::fail("missing dependency did not produce a named error");
        }

        render::RenderSubsystemHost cycleHost;
        log.clear();
        registerProbe(cycleHost, "test.cycle-a", {"test.cycle-b"}, events, log);
        registerProbe(cycleHost, "test.cycle-b", {"test.cycle-a"}, events, log);
        result = cycleHost.initialize(context.device, 1, log);
        if (!result) {
            return RHITestResult::fail(log);
        }
        result = cycleHost.activate("test.cycle-a", log);
        if (result || log.find("cycle") == std::string::npos) {
            return RHITestResult::fail("dependency cycle was not detected");
        }

        render::RenderSubsystemHost failingHost;
        log.clear();
        registerProbe(failingHost, "test.failing", {}, events, log, true);
        result = failingHost.initialize(context.device, 1, log);
        if (!result || !failingHost.activate("test.failing", log)) {
            return RHITestResult::fail(log);
        }
        result = failingHost.beginFrame(0, 0, nullptr, log);
        if (result || log.find("test.failing") == std::string::npos) {
            return RHITestResult::fail("hook error did not propagate with subsystem id");
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphSerializationTest);
METALLIC_REGISTER_RHI_TEST(VisibilityBufferPassLegacyGraphTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphLegacyAcronymPassNamesTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphReflectionAPITest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPassKindTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphDLSSRRMotionVectorContractTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphRuntimeSettingsDeclarationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphRuntimeRebuildDirtyTest);
METALLIC_REGISTER_RHI_TEST(RenderSampleLoadTest);
METALLIC_REGISTER_RHI_TEST(GPUDrivenSceneCatalogTest);
METALLIC_REGISTER_RHI_TEST(RenderSampleFallbackAndValidationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphValidationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBunnyWireframePreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBunnyCameraSyncTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneRayQueryVisualizationPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneMaterialVisualizationPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTracePreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSharcStagesTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphNRCStagesTest);
METALLIC_REGISTER_RHI_TEST(SlangShaderDiskCacheTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingShaderCompileTest);
#if defined(METALLIC_HAS_RTXCR) && METALLIC_HAS_RTXCR
METALLIC_REGISTER_RHI_TEST(RenderGraphRTXCRMaterialShaderCompileTest);
#if defined(METALLIC_HAS_RTXCR_GEOMETRY) && METALLIC_HAS_RTXCR_GEOMETRY && \
    defined(METALLIC_HAS_RTXCR_ASSETS) && METALLIC_HAS_RTXCR_ASSETS
METALLIC_REGISTER_RHI_TEST(RenderGraphRTXCRMaterialPreviewTest);
#endif
#endif
METALLIC_REGISTER_RHI_TEST(GPUDrivenPreviewGeometryDedupPlanTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenPreviewShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenStreamAssetShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenStreamAssetTraversalDemandTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphRTXDIPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphRTXDIShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPathTracingGuidesShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphStreamlineDLSSSupportShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneRayQueryClusterShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingSamplePreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingDebugViewsTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingEnvironmentRotationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceMaterialTexturesPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceTransmissionTexturesPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceAlphaMaskPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneSwitchRetirementTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphResizeReusesCompiledPassesTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPreviewActualOutputExtentTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureExtentConstraintPropagationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureExtentConstraintConflictTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureExtentConstraintMultihopPropagationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureExtentConstraintRelayConflictTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphShaderReloadTransactionTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphCopyColorWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBindlessTextureWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBufferWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMultiQueueSubmitTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureFeedbackEpilogueTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphImageSamplePassPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMaterialShaderObjectPassSmokeTest);
class StreamSceneOpenRoutingTest final : public RHITest {
public:
    StreamSceneOpenRoutingTest() { name = "stream_scene_open_routing"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto directory = std::filesystem::absolute(context.outputDirectory / "stream-open-routing");
        std::filesystem::create_directories(directory);
        const auto first = directory / "first.gltf";
        const auto second = directory / "second.gltf";
        std::ofstream(first) << "{}";
        std::ofstream(second) << "{}";
        RenderGraph graph;
        const auto id = graph.addNode("VisibilityBufferPass", "VBuffer", {
            {"path", "first.gltf"}, {"streamAssetPath", "custom.meshstream.bin"},
            {"sceneBinding", "asset"}, {"enableMeshletStreaming", true}, {"streamAssetOnly", true}})->id;
        const auto same = editor::streamSceneLoadOptions(graph, first, directory);
        const auto switched = editor::streamSceneLoadOptions(graph, second, directory);
        if (same.streamAssetPath != directory / "custom.meshstream.bin" ||
            switched.streamAssetPath != scene::meshletStreamAssetPathFor(second) ||
            !graph.findNode(id)->runtimeProperties.empty()) {
            return RHITestResult::fail("Stream open planning reused another scene's cache or mutated the live graph");
        }
        if (!editor::applyStreamSceneOpen(graph, second, switched.streamAssetPath)) {
            return RHITestResult::fail("Stream scene switch was not committed");
        }
        const auto& properties = graph.findNode(id)->runtimeProperties;
        if (properties.at("path") != second.generic_string() || properties.at("sceneBinding") != "world" ||
            properties.at("streamAssetPath") != switched.streamAssetPath.generic_string() ||
            editor::streamSceneLoadOptions(graph, second, directory).streamAssetPath != switched.streamAssetPath) {
            return RHITestResult::fail("Source, world binding and cache did not switch together");
        }
        RenderGraph resident;
        resident.addNode("VisibilityBufferPass", "Resident", {});
        if (!editor::streamSceneLoadOptions(resident, second, directory).streamAssetPath.empty()) {
            return RHITestResult::fail("Resident graph unexpectedly selected metadata-only loading");
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(StreamSceneOpenRoutingTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphVisibilityBufferPassSmokeTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenStreamAssetPassSmokeTest);
template<int Mode>
class MixedProducerRasterVariantTest final : public RenderGraphGPUDrivenMixedProducerRenderTest {
public:
    MixedProducerRasterVariantTest() : RenderGraphGPUDrivenMixedProducerRenderTest(Mode) {}
};
using MixedProducerPreparedTest = MixedProducerRasterVariantTest<0>;
using MixedProducerLegacyTest = MixedProducerRasterVariantTest<1>;
using MixedProducerPlaneTest = MixedProducerRasterVariantTest<2>;
using MixedProducerCooperativeTest = MixedProducerRasterVariantTest<3>;
using MixedProducerWorkBinsTest = MixedProducerRasterVariantTest<4>;
METALLIC_REGISTER_RHI_TEST(MixedProducerPreparedTest);
METALLIC_REGISTER_RHI_TEST(MixedProducerLegacyTest);
METALLIC_REGISTER_RHI_TEST(MixedProducerPlaneTest);
METALLIC_REGISTER_RHI_TEST(MixedProducerCooperativeTest);
METALLIC_REGISTER_RHI_TEST(MixedProducerWorkBinsTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenMixedProducerRenderTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphVisibilityBufferPassRenderTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenAlphaMaskRenderTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenSponzaVisibilityRenderTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphStreamedAsyncAccelerationStructureTest);
METALLIC_REGISTER_RHI_TEST(ImportancePdfSizeTest);
METALLIC_REGISTER_RHI_TEST(ReGIRGridLayoutTest);
METALLIC_REGISTER_RHI_TEST(EnvironmentSubsystemAsyncSnapshotTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMissingSubsystemDiagnosticTest);
METALLIC_REGISTER_RHI_TEST(RenderSubsystemHostLifecycleTest);

} // namespace
} // namespace metallic::tests
