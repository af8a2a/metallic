#include "RhiTest.h"

#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/ReGIR.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
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
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::tests {
namespace {

constexpr const char* kShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kBindlessSmokeShaderModuleName = "BindlessSmoke";
constexpr const char* kBindlessSmokeVertexEntryPoint = "bindlessSmokeVertexMain";
constexpr const char* kBindlessSmokeFragmentEntryPoint = "bindlessSmokeFragmentMain";
constexpr uint32_t kSpirvMagic = 0x07230203u;
constexpr uint32_t kSpirvVersion16 = 0x00010600u;
constexpr uint16_t kSpirvOpExtension = 10u;
constexpr uint16_t kSpirvOpExtInstImport = 11u;
constexpr uint16_t kSpirvOpCapability = 17u;
constexpr uint16_t kSpirvOpRayQueryGetIntersectionClusterIdNv = 5345u;
constexpr uint32_t kSpirvRayTracingClusterAccelerationStructureNv = 5437u;

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

    render::Result initialize(
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
        if (opcode == kSpirvOpCapability && wordCount >= 2 && spirv[wordIndex + 1] == expectedCapability) {
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
        if (opcode == kSpirvOpExtension && wordCount >= 2) {
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
        if (opcode == kSpirvOpExtInstImport && wordCount >= 3) {
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

render::Result createSlangShaderModule(
    render::Device& device,
    const char* moduleName,
    const char* entryPointName,
    std::unique_ptr<render::ShaderModule>& outShaderModule,
    std::string& log)
{
    render::ShaderCompileResult compileResult;
    render::Result result = render::compileSlangShaderToSpirv(
        render::SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kShaderSearchPath,
        },
        compileResult);
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

    return device.createShaderModule(
        render::ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
        },
        outShaderModule);
}

render::Result writeHostBuffer(render::Buffer& buffer, const void* data, uint64_t byteSize)
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
        buffer.flush(0, byteSize);
    }
    buffer.unmap();
    return {};
}

bool readHostBuffer(render::Buffer& buffer, void* outData, uint64_t byteSize)
{
    if (byteSize > buffer.desc().size || (byteSize > 0 && outData == nullptr)) {
        return false;
    }
    buffer.invalidate(0, byteSize);
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

    render::Result execute(render::RenderGraphExecutionContext&) override
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

    render::Result execute(render::RenderGraphExecutionContext&) override
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

    render::Result execute(render::RenderGraphExecutionContext&) override
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
            .format = render::Format::Rgba8Unorm;
        return reflection;
    }

    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return render::makeError(render::Error::InvalidArgument);
        }

        render::Result result = createSlangShaderModule(
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

        result = context.device->createGraphicsPipeline(
            render::GraphicsPipelineDesc{
                .vertexShader = vertexShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = render::Format::Rgba8Unorm,
                .topology = render::PrimitiveTopology::TriangleList,
                .usesBindlessHeap = true,
            },
            pipeline_);
        if (!result) {
            log += std::string("createGraphicsPipeline(bindless graph pass) returned ") + toString(result) + '\n';
        }
        return result;
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        const render::BindlessHandle* sourceHandle = context.bindlessInput("source");
        render::TextureHandle color = context.outputTexture("color");
        if (sourceHandle == nullptr ||
            sourceHandle->kind != render::BindlessHandleKind::SampledImage ||
            sourceHandle->index != 0 ||
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
            .state = render::ResourceState::ColorAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearColor = render::ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        context.commandBuffer().beginRendering(render::RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(render::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        context.commandBuffer().bindGraphicsPipeline(*pipeline_);
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

    render::Result compile(const render::RenderGraphCompileContext&, std::string&) override
    {
        ++testResizeCompileCount();
        return {};
    }

    render::Result execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
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

    render::Result execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
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

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        const render::EnvironmentLightingSubsystem* environment =
            context.subsystem<render::EnvironmentLightingSubsystem>();
        return environment != nullptr && environment->snapshot().valid()
            ? render::Result{}
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
        "TestResizeCompilePass",
        "Test-only pass that counts RenderGraph compile calls",
        []() { return std::make_unique<TestResizeCompilePass>(); });
    render::registerRenderGraphPassType(
        "TestMissingSubsystemPass",
        "Test-only pass with an intentionally missing subsystem",
        []() { return std::make_unique<TestMissingSubsystemPass>(); });
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
  "asset": { "version": "2.0", "generator": "MetallicRhiTests" },
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
  "asset": { "version": "2.0", "generator": "MetallicRhiTests" },
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

class RenderGraphReflectionApiTest : public RhiTest {
public:
    RenderGraphReflectionApiTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_reflection_api";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderPassReflection reflection;
        render::RenderGraphField& texture = reflection.addTextureInput("source", "Texture source")
            .texture2D(32, 16)
            .sampledRead()
            .bindlessSampledImage()
            .setOptional();
        texture.format = render::Format::Rgba8Unorm;

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
            return RhiTestResult::fail("reflection did not preserve fields");
        }
        if (foundTexture->resourceType != render::RenderGraphResourceType::Texture2D ||
            foundTexture->access != render::RenderGraphResourceAccess::TextureSampleRead ||
            foundTexture->bindlessAccess != render::RenderGraphBindlessAccess::SampledImage ||
            foundTexture->width != 32 ||
            foundTexture->height != 16 ||
            !foundTexture->optional) {
            return RhiTestResult::fail("texture field metadata was not preserved");
        }
        if (foundBuffer->resourceType != render::RenderGraphResourceType::Buffer ||
            foundBuffer->access != render::RenderGraphResourceAccess::BufferStorageReadWrite ||
            foundBuffer->bindlessAccess != render::RenderGraphBindlessAccess::Buffer ||
            foundBuffer->size != 64 ||
            foundBuffer->structureStride != 8 ||
            foundBuffer->memoryLocation != render::MemoryLocation::HostReadback) {
            return RhiTestResult::fail("buffer field metadata was not preserved");
        }
        if (foundDepth->resourceType != render::RenderGraphResourceType::Texture2D ||
            foundDepth->access != render::RenderGraphResourceAccess::TextureDepthStencilWrite ||
            foundDepth->format != render::Format::D32Sfloat ||
            foundDepth->usage != render::TextureUsageBits::DepthStencilAttachment ||
            foundDepth->state != render::ResourceState::DepthStencilAttachment) {
            return RhiTestResult::fail("depth field metadata was not preserved");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphPassKindTest : public RhiTest {
public:
    RenderGraphPassKindTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_pass_kind";
    }

    RhiTestResult run(RhiTestContext&) override
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
            render::createRenderGraphPass("SceneRtxdiPass");
        const std::unique_ptr<render::RenderGraphPass> rtxdiConfidence =
            render::createRenderGraphPass("RtxdiConfidencePass");
        const std::unique_ptr<render::RenderGraphPass> rtxdiComposite =
            render::createRenderGraphPass("RtxdiCompositePass");
        const std::unique_ptr<render::RenderGraphPass> materialVisualization =
            render::createRenderGraphPass("SceneMaterialVisualizationPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenPreview =
            render::createRenderGraphPass("GPUDrivenPreviewPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenStreamAsset =
            render::createRenderGraphPass("GPUDrivenStreamAssetPass");
        const std::unique_ptr<render::RenderGraphPass> nrdDenoise =
            render::createRenderGraphPass("NrdDenoisePass");
        const std::unique_ptr<render::RenderGraphPass> streamlineDlssRr =
            render::createRenderGraphPass("StreamlineDlssRrPass");

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
            streamlineDlssRr == nullptr) {
            return RhiTestResult::fail("failed to create built-in render graph passes");
        }
        if (triangle->kind() != render::RenderGraphPassKind::Raster ||
            triangle->queueType() != render::QueueType::Graphics) {
            return RhiTestResult::fail("TriangleRasterPass is not classified as Raster/Graphics");
        }
        if (copy->kind() != render::RenderGraphPassKind::Unsafe ||
            copy->queueType() != render::QueueType::Graphics) {
            return RhiTestResult::fail("CopyColorPass is not classified as Unsafe/Graphics");
        }
        if (bufferWrite->kind() != render::RenderGraphPassKind::Compute ||
            bufferWrite->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("RenderGraphBufferWritePass is not classified as Compute/Compute");
        }
        if (pathTrace->kind() != render::RenderGraphPassKind::Compute ||
            pathTrace->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("ScenePathTracePass is not classified as Compute/Compute");
        }
        if (rtxdi->kind() != render::RenderGraphPassKind::Compute ||
            rtxdi->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("SceneRtxdiPass is not classified as Compute/Compute");
        }
        if (rtxdiConfidence->kind() != render::RenderGraphPassKind::Compute ||
            rtxdiConfidence->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("RtxdiConfidencePass is not classified as Compute/Compute");
        }
        if (rtxdiComposite->kind() != render::RenderGraphPassKind::Compute ||
            rtxdiComposite->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("RtxdiCompositePass is not classified as Compute/Compute");
        }
        if (materialVisualization->kind() != render::RenderGraphPassKind::Compute ||
            materialVisualization->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("SceneMaterialVisualizationPass is not classified as Compute/Compute");
        }
        if (gpuDrivenPreview->kind() != render::RenderGraphPassKind::Unsafe ||
            gpuDrivenPreview->queueType() != render::QueueType::Graphics) {
            return RhiTestResult::fail("GPUDrivenPreviewPass is not classified as Unsafe/Graphics");
        }
        if (gpuDrivenStreamAsset->kind() != render::RenderGraphPassKind::Unsafe ||
            gpuDrivenStreamAsset->queueType() != render::QueueType::Graphics) {
            return RhiTestResult::fail("GPUDrivenStreamAssetPass is not classified as Unsafe/Graphics");
        }
        if (nrdDenoise->kind() != render::RenderGraphPassKind::Compute ||
            nrdDenoise->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("NrdDenoisePass is not classified as Compute/Compute");
        }
        if (streamlineDlssRr->kind() != render::RenderGraphPassKind::Unsafe ||
            streamlineDlssRr->queueType() != render::QueueType::Graphics) {
            return RhiTestResult::fail("StreamlineDlssRrPass is not classified as Unsafe/Graphics");
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
        bool foundStreamlineDlssRr = false;
        for (const render::RenderGraphPassInfo& passInfo : render::listRenderGraphPassTypes()) {
            if (passInfo.type == "TriangleRasterPass") {
                foundTriangle = passInfo.kind == render::RenderGraphPassKind::Raster &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "CopyColorPass") {
                foundCopy = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "RenderGraphBufferWritePass") {
                foundBufferWrite = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "ScenePathTracePass") {
                foundPathTrace = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "SceneRtxdiPass") {
                foundRtxdi = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "RtxdiConfidencePass") {
                foundRtxdiConfidence = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "RtxdiCompositePass") {
                foundRtxdiComposite = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "SceneMaterialVisualizationPass") {
                foundMaterialVisualization = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "GPUDrivenPreviewPass") {
                foundGPUDrivenPreview = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "GPUDrivenStreamAssetPass") {
                foundGPUDrivenStreamAsset = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
                    passInfo.queueType == render::QueueType::Graphics;
            } else if (passInfo.type == "NrdDenoisePass") {
                foundNrdDenoise = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "StreamlineDlssRrPass") {
                foundStreamlineDlssRr = passInfo.kind == render::RenderGraphPassKind::Unsafe &&
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
            !foundStreamlineDlssRr) {
            return RhiTestResult::fail("RenderGraphPassInfo did not preserve pass kind metadata");
        }

        return RhiTestResult::pass();
    }
};


bool hasRuntimeSetting(
    const render::RenderGraphPass& pass,
    const std::string& key,
    render::RenderGraphRuntimeSettingType type,
    bool requireHistoryInvalidation = false)
{
    const std::vector<render::RenderGraphRuntimeSetting> settings = pass.runtimeSettings();
    for (const render::RenderGraphRuntimeSetting& setting : settings) {
        if (setting.key == key &&
            setting.type == type &&
            (!requireHistoryInvalidation || setting.invalidateHistory)) {
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

class RenderGraphRuntimeSettingsDeclarationTest : public RhiTest {
public:
    RenderGraphRuntimeSettingsDeclarationTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_runtime_settings_declarations";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        const std::unique_ptr<render::RenderGraphPass> pathTrace =
            render::createRenderGraphPass("ScenePathTracePass");
        const std::unique_ptr<render::RenderGraphPass> rtxdi =
            render::createRenderGraphPass("SceneRtxdiPass");
        const std::unique_ptr<render::RenderGraphPass> nrdDenoise =
            render::createRenderGraphPass("NrdDenoisePass");
        const std::unique_ptr<render::RenderGraphPass> materialVisualization =
            render::createRenderGraphPass("SceneMaterialVisualizationPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenPreview =
            render::createRenderGraphPass("GPUDrivenPreviewPass");
        const std::unique_ptr<render::RenderGraphPass> gpuDrivenStreamAsset =
            render::createRenderGraphPass("GPUDrivenStreamAssetPass");
        if (pathTrace == nullptr ||
            rtxdi == nullptr ||
            nrdDenoise == nullptr ||
            materialVisualization == nullptr ||
            gpuDrivenPreview == nullptr ||
            gpuDrivenStreamAsset == nullptr) {
            return RhiTestResult::fail("failed to create passes for runtime settings declaration test");
        }
        if (!hasBoolRuntimeSetting(*pathTrace, "flipBitangent")) {
            return RhiTestResult::fail("ScenePathTracePass missing Bool runtime setting flipBitangent");
        }
        if (!hasRuntimeSetting(
                *pathTrace,
                "debugView",
                render::RenderGraphRuntimeSettingType::Enum,
                true)) {
            return RhiTestResult::fail("ScenePathTracePass missing history-invalidating debugView enum");
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
                return RhiTestResult::fail(
                    std::string("ScenePathTracePass missing history-invalidating debug Bool setting ") + key);
            }
        }
        if (!hasBoolRuntimeSetting(*rtxdi, "temporalReuse") ||
            !hasBoolRuntimeSetting(*rtxdi, "spatialReuse") ||
            !hasBoolRuntimeSetting(*rtxdi, "initialVisibility") ||
            !hasBoolRuntimeSetting(*rtxdi, "animateLights")) {
            return RhiTestResult::fail("SceneRtxdiPass missing ReSTIR DI Bool runtime settings");
        }
        if (!hasBoolRuntimeSetting(*nrdDenoise, "relaxAntiFirefly")) {
            return RhiTestResult::fail("NrdDenoisePass missing RELAX runtime settings");
        }
        if (!hasBoolRuntimeSetting(*nrdDenoise, "relaxConfidenceInputs")) {
            return RhiTestResult::fail("NrdDenoisePass missing RELAX confidence setting");
        }
        if (!hasBoolRuntimeSetting(*materialVisualization, "flipBitangent")) {
            return RhiTestResult::fail("SceneMaterialVisualizationPass missing Bool runtime setting flipBitangent");
        }
        if (!hasBoolRuntimeSetting(*gpuDrivenPreview, "instanceFrustumCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "instanceHzbCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "meshletFrustumCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "meshletNormalConeCull") ||
            !hasBoolRuntimeSetting(*gpuDrivenPreview, "freezeCullingCamera")) {
            return RhiTestResult::fail("GPUDrivenPreviewPass missing visibility culling runtime settings");
        }
        if (!hasBoolRuntimeSetting(*gpuDrivenStreamAsset, "enableGpuLodSelection")) {
            return RhiTestResult::fail("GPUDrivenStreamAssetPass missing Bool runtime setting enableGpuLodSelection");
        }
        return RhiTestResult::pass();
    }
};
class RenderGraphSerializationTest : public RhiTest {
public:
    RenderGraphSerializationTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_json_roundtrip";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraph graph = render::RenderGraph::createDefaultTriangleGraph();
        render::RenderGraphNode* node = graph.findNode("Triangle");
        if (node == nullptr) {
            return RhiTestResult::fail("default graph did not create Triangle node");
        }
        graph.setNodePosition(node->id, 123.0f, 456.0f);
        graph.clearDirty();
        if (!graph.setNodeRuntimeProperty(node->id, "runtimeOnlySentinel", 42) ||
            !graph.setNodeRuntimeProperty(node->id, "camera.eye", {1.0f, 2.0f, 3.0f})) {
            return RhiTestResult::fail("setNodeRuntimeProperty failed");
        }
        if (graph.dirty()) {
            return RhiTestResult::fail("runtime property update unexpectedly marked graph dirty");
        }
        if (!node->runtimeProperties.is_object() ||
            !node->runtimeProperties.contains("camera") ||
            !node->runtimeProperties["camera"].contains("eye")) {
            return RhiTestResult::fail("nested runtime property was not stored as an overlay object");
        }

        const std::string json = render::serializeRenderGraphToString(graph);
        if (json.find("runtimeOnlySentinel") != std::string::npos ||
            json.find("runtimeProperties") != std::string::npos) {
            return RhiTestResult::fail("runtime properties leaked into serialized graph JSON");
        }
        render::RenderGraph loaded;
        std::string message;
        if (!render::deserializeRenderGraphFromString(json, loaded, message)) {
            return RhiTestResult::fail(message);
        }

        if (loaded.nodes().size() != 1 || loaded.edges().size() != 0 || loaded.outputs().size() != 1) {
            return RhiTestResult::fail("round-trip changed graph topology");
        }
        const render::RenderGraphNode* loadedNode = loaded.findNode("Triangle");
        if (loadedNode == nullptr ||
            loadedNode->type != "TriangleRasterPass" ||
            loadedNode->uiX != 123.0f ||
            loadedNode->uiY != 456.0f) {
            return RhiTestResult::fail("round-trip changed node data");
        }
        if (loaded.firstOutputName() != "Triangle.color") {
            return RhiTestResult::fail("round-trip changed marked output");
        }

        if (!loadedNode->runtimeProperties.empty()) {
            return RhiTestResult::fail("round-trip restored runtime overlay from JSON");
        }
        if (!graph.setNodeProperties(node->id, node->properties) || !graph.dirty()) {
            return RhiTestResult::fail("static property update did not mark graph dirty");
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
                    "name": "DlssRr",
                    "type": "StreamlineDlssRrPass",
                    "properties": {}
                }
            ],
            "edges": [
                {"src": "PathTrace.color", "dst": "DlssRr.inputColor"},
                {"src": "PathTrace.albedo", "dst": "DlssRr.albedo"},
                {"src": "PathTrace.specularAlbedo", "dst": "DlssRr.specularAlbedo"},
                {"src": "PathTrace.normalRoughness", "dst": "DlssRr.normalRoughness"},
                {"src": "PathTrace.motionVectors", "dst": "DlssRr.motionVectors"},
                {"src": "PathTrace.linearDepth", "dst": "DlssRr.linearDepth"},
                {"src": "PathTrace.specularHitDistance", "dst": "DlssRr.specularHitDistance"}
            ],
            "outputs": [
                "DlssRr.color"
            ]
        })json";
        render::RenderGraph legacyLoaded;
        if (!render::deserializeRenderGraphFromString(legacyJson, legacyLoaded, message)) {
            return RhiTestResult::fail(message);
        }
        std::unordered_set<uint32_t> legacyEdgeIds;
        for (const render::RenderGraphEdge& edge : legacyLoaded.edges()) {
            if (edge.id == 0u || !legacyEdgeIds.insert(edge.id).second) {
                return RhiTestResult::fail("legacy graph edges did not receive unique ids");
            }
        }
        if (legacyEdgeIds.size() != 7u) {
            return RhiTestResult::fail("legacy graph changed edge count");
        }
        return RhiTestResult::pass();
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

class RenderSampleLoadTest : public RhiTest {
public:
    RenderSampleLoadTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_sample_load";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-meet-mat", sample, message)) {
            return RhiTestResult::fail(message);
        }

        if (sample.desc.id != "pathtracing-meet-mat" ||
            sample.desc.name != "Path Tracing / meet_mat" ||
            sample.desc.category != "PathTracing" ||
            sample.desc.scenePath != "Asset/meet_mat.glb" ||
            sample.desc.graphPath != "Pipelines/Samples/pathtracing_meet_mat.metallic_graph.json" ||
            sample.desc.previewOutput != "PathTrace.color") {
            return RhiTestResult::fail("built-in Sample metadata did not load as expected");
        }

        const render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr ||
            !pathTrace->properties.is_object() ||
            pathTrace->properties.value("path", "") != sample.desc.scenePath) {
            return RhiTestResult::fail("Sample did not apply scene path to target node");
        }

        std::string validationLog;
        if (!sample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (sample.graph.firstOutputName() != "PathTrace.color") {
            return RhiTestResult::fail("Sample graph first output changed");
        }

        render::RenderSampleLoadResult pathTracingSample;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", pathTracingSample, message)) {
            return RhiTestResult::fail(message);
        }
        if (pathTracingSample.desc.id != "pathtracing-sample" ||
            pathTracingSample.desc.name != "PathTracingSample" ||
            pathTracingSample.desc.category != "PathTracing" ||
            pathTracingSample.desc.scenePath != "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf" ||
            pathTracingSample.desc.graphPath != "Pipelines/Samples/pathtracing_abeautiful_game_openpbr.metallic_graph.json" ||
            !pathTracingSample.desc.environment.has_value() ||
            pathTracingSample.desc.environment->path != "Asset/ABeautifulGame/environment.hdr" ||
            pathTracingSample.desc.previewOutput != "PathTrace.color") {
            return RhiTestResult::fail("OpenPBR PathTracingSample metadata did not load as expected");
        }
        const render::RenderGraphNode* openPBRPathTrace = pathTracingSample.graph.findNode("PathTrace");
        if (openPBRPathTrace == nullptr ||
            !openPBRPathTrace->properties.is_object() ||
            openPBRPathTrace->properties.value("path", "") != pathTracingSample.desc.scenePath ||
            openPBRPathTrace->properties.value("bsdf", "") != "openpbr") {
            return RhiTestResult::fail("OpenPBR PathTracingSample did not apply pass defaults");
        }
        if (!pathTracingSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (pathTracingSample.graph.firstOutputName() != "PathTrace.color") {
            return RhiTestResult::fail("OpenPBR PathTracingSample graph first output changed");
        }

        render::RenderSampleLoadResult rtxdiSample;
        if (!render::loadBuiltInRenderSample("rtxdi-sample", rtxdiSample, message)) {
            return RhiTestResult::fail(message);
        }
        if (rtxdiSample.desc.id != "rtxdi-sample" ||
            rtxdiSample.desc.name != "RTXDI / ReSTIR DI" ||
            rtxdiSample.desc.category != "RTXDI" ||
            rtxdiSample.desc.scenePath != "Asset/meet_mat.glb" ||
            rtxdiSample.desc.graphPath != "Pipelines/Samples/rtxdi_meet_mat.metallic_graph.json" ||
            !rtxdiSample.desc.environment.has_value() ||
            rtxdiSample.desc.environment->path != "Asset/ABeautifulGame/environment.hdr" ||
            rtxdiSample.desc.previewOutput != "Composite.color") {
            return RhiTestResult::fail("RTXDI Sample metadata did not load as expected");
        }
        const render::RenderGraphNode* rtxdi = rtxdiSample.graph.findNode("Rtxdi");
        const render::RenderGraphNode* confidence = rtxdiSample.graph.findNode("Confidence");
        const render::RenderGraphNode* relax = rtxdiSample.graph.findNode("Relax");
        const render::RenderGraphNode* composite = rtxdiSample.graph.findNode("Composite");
        if (rtxdi == nullptr ||
            confidence == nullptr ||
            relax == nullptr ||
            composite == nullptr ||
            rtxdi->type != "SceneRtxdiPass" ||
            confidence->type != "RtxdiConfidencePass" ||
            relax->type != "NrdDenoisePass" ||
            composite->type != "RtxdiCompositePass" ||
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
            return RhiTestResult::fail("RTXDI Sample did not apply ReSTIR DI and RELAX defaults");
        }
        if (!rtxdiSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (rtxdiSample.graph.firstOutputName() != "Composite.color") {
            return RhiTestResult::fail("RTXDI Sample graph first output changed");
        }

        render::RenderSampleLoadResult rtxcrSample;
        if (!render::loadBuiltInRenderSample("rtxcr-material-sample", rtxcrSample, message)) {
            return RhiTestResult::fail(message);
        }
        if (rtxcrSample.desc.id != "rtxcr-material-sample" ||
            rtxcrSample.desc.name != "RTXCR Claire Ponytail" ||
            rtxcrSample.desc.category != "RTXCR" ||
            !rtxcrSample.desc.loadSceneInEditor ||
            rtxcrSample.desc.graphPath !=
                "Pipelines/Samples/rtxcr_material_showcase.metallic_graph.json" ||
            rtxcrSample.desc.scenePath.find("ponyTail_15vtx.gltf") == std::string::npos ||
            rtxcrSample.desc.previewOutput != "PathTrace.color") {
            return RhiTestResult::fail("RTXCR Sample metadata did not load as expected");
        }
        const render::RenderGraphNode* rtxcr = rtxcrSample.graph.findNode("PathTrace");
        if (rtxcr == nullptr ||
            rtxcr->type != "ScenePathTracePass" ||
            !rtxcr->properties.is_object() ||
            rtxcr->properties.value("samples", 0) != 4 ||
            rtxcr->properties.value("maxDepth", 0) != 4 ||
            rtxcr->properties.value("path", "").find("ponyTail_15vtx.gltf") ==
                std::string::npos) {
            return RhiTestResult::fail("RTXCR Sample did not preserve Claire groom defaults");
        }
        if (!rtxcrSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (rtxcrSample.graph.firstOutputName() != "PathTrace.color") {
            return RhiTestResult::fail("RTXCR Sample graph first output changed");
        }

        render::RenderSampleLoadResult dlssRrSample;
        if (!render::loadBuiltInRenderSample("pathtracing-sample-dlss-rr", dlssRrSample, message)) {
            return RhiTestResult::fail(message);
        }
        if (dlssRrSample.desc.id != "pathtracing-sample-dlss-rr" ||
            dlssRrSample.desc.name != "PathTracingSample / DLSS-RR" ||
            dlssRrSample.desc.category != "PathTracing" ||
            dlssRrSample.desc.scenePath != "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf" ||
            dlssRrSample.desc.graphPath != "Pipelines/Samples/pathtracing_abeautiful_game_openpbr_dlss_rr.metallic_graph.json" ||
            dlssRrSample.desc.previewOutput != "DlssRr.color" ||
            !dlssRrSample.desc.requiresStreamline) {
            return RhiTestResult::fail("DLSS-RR PathTracingSample metadata did not load as expected");
        }
        const render::RenderGraphNode* dlssRrPathTrace = dlssRrSample.graph.findNode("PathTrace");
        const render::RenderGraphNode* dlssRrPass = dlssRrSample.graph.findNode("DlssRr");
        if (dlssRrPathTrace == nullptr ||
            dlssRrPass == nullptr ||
            !dlssRrPathTrace->properties.is_object() ||
            dlssRrPathTrace->properties.value("path", "") != dlssRrSample.desc.scenePath ||
            !dlssRrPathTrace->properties.value("exportDenoiserGuides", false) ||
            dlssRrPass->type != "StreamlineDlssRrPass") {
            return RhiTestResult::fail("DLSS-RR PathTracingSample did not apply expected graph defaults");
        }
        if (!dlssRrSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (dlssRrSample.graph.firstOutputName() != "DlssRr.color") {
            return RhiTestResult::fail("DLSS-RR PathTracingSample graph first output changed");
        }

        render::RenderSampleLoadResult materialSample;
        if (!render::loadBuiltInRenderSample("material-visualization-abeautiful-game", materialSample, message)) {
            return RhiTestResult::fail(message);
        }
        if (materialSample.desc.id != "material-visualization-abeautiful-game" ||
            materialSample.desc.name != "Material Visualization / ABeautifulGame" ||
            materialSample.desc.category != "Material" ||
            materialSample.desc.scenePath != "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf" ||
            materialSample.desc.graphPath != "Pipelines/Samples/material_visualization_abeautiful_game.metallic_graph.json" ||
            materialSample.desc.previewOutput != "MaterialViz.color") {
            return RhiTestResult::fail("material visualization Sample metadata did not load as expected");
        }
        const render::RenderGraphNode* materialViz = materialSample.graph.findNode("MaterialViz");
        if (materialViz == nullptr ||
            !materialViz->properties.is_object() ||
            materialViz->properties.value("path", "") != materialSample.desc.scenePath ||
            materialViz->properties.value("mode", "") != "material") {
            return RhiTestResult::fail("material visualization Sample did not apply scene path and defaults");
        }
        if (!materialSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (materialSample.graph.firstOutputName() != "MaterialViz.color") {
            return RhiTestResult::fail("material visualization Sample graph first output changed");
        }

        render::RenderSampleLoadResult gpuDrivenSample;
        if (!render::loadBuiltInRenderSample("gpu-driven-sample", gpuDrivenSample, message)) {
            return RhiTestResult::fail(message);
        }
        if (gpuDrivenSample.desc.id != "gpu-driven-sample" ||
            gpuDrivenSample.desc.name != "GPUDrivenSample" ||
            gpuDrivenSample.desc.category != "GPUDriven" ||
            gpuDrivenSample.desc.scenePath != "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf" ||
            gpuDrivenSample.desc.loadSceneInEditor ||
            gpuDrivenSample.desc.graphPath != "Pipelines/Samples/gpu_driven_sponza.metallic_graph.json" ||
            !gpuDrivenSample.desc.environment.has_value() ||
            gpuDrivenSample.desc.environment->path != "Asset/ABeautifulGame/environment.hdr" ||
            gpuDrivenSample.desc.previewOutput != "GPUDriven.color" ||
            gpuDrivenSample.desc.requiresStreamline) {
            return RhiTestResult::fail("GPUDrivenSample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDriven = gpuDrivenSample.graph.findNode("GPUDriven");
        if (gpuDriven == nullptr ||
            gpuDriven->type != "GPUDrivenPreviewPass" ||
            !gpuDriven->properties.is_object() ||
            gpuDriven->properties.value("path", "") != gpuDrivenSample.desc.scenePath ||
            gpuDriven->properties.value("mode", "") != "shaded" ||
            !gpuDriven->properties.contains("camera") ||
            !gpuDriven->properties["camera"].is_object()) {
            return RhiTestResult::fail("GPUDrivenSample did not apply pass defaults");
        }
        if (!gpuDrivenSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (gpuDrivenSample.graph.firstOutputName() != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDrivenSample graph first output changed");
        }
        bool requiresStreamline = true;
        if (!render::queryBuiltInRenderSampleStreamlineRequirement(
                "gpu-driven-sample",
                requiresStreamline) ||
            requiresStreamline ||
            !render::queryBuiltInRenderSampleStreamlineRequirement(
                "pathtracing-sample-dlss-rr",
                requiresStreamline) ||
            !requiresStreamline ||
            render::queryBuiltInRenderSampleStreamlineRequirement(
                "unknown-sample",
                requiresStreamline)) {
            return RhiTestResult::fail("built-in Sample Streamline requirements are inconsistent");
        }

        render::RenderSampleLoadResult gpuDrivenStreamAssetSample;
        if (!render::loadBuiltInRenderSample("gpu-driven-streamasset", gpuDrivenStreamAssetSample, message)) {
            return RhiTestResult::fail(message);
        }
        if (gpuDrivenStreamAssetSample.desc.id != "gpu-driven-streamasset" ||
            gpuDrivenStreamAssetSample.desc.name != "GPUDrivenSample / StreamAsset" ||
            gpuDrivenStreamAssetSample.desc.category != "GPUDriven" ||
            gpuDrivenStreamAssetSample.desc.scenePath != "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf" ||
            gpuDrivenStreamAssetSample.desc.loadSceneInEditor ||
            gpuDrivenStreamAssetSample.desc.graphPath !=
                "Pipelines/Samples/gpu_driven_sponza_streamasset.metallic_graph.json" ||
            gpuDrivenStreamAssetSample.desc.previewOutput != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDriven StreamAsset sample metadata did not load as expected");
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
            return RhiTestResult::fail("GPUDriven StreamAsset sample did not preserve streamasset defaults");
        }
        if (!gpuDrivenStreamAssetSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (gpuDrivenStreamAssetSample.graph.firstOutputName() != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDriven StreamAsset graph first output changed");
        }
        if (!render::setRenderSampleScenePath(
                gpuDrivenStreamAssetSample,
                "Asset/Zorah/zorah_main_public.v2.gltf",
                message) ||
            gpuDrivenStreamAssetSample.desc.scenePath != "Asset/Zorah/zorah_main_public.v2.gltf" ||
            gpuDrivenStreamAsset->properties.value("path", "") !=
                "Asset/Zorah/zorah_main_public.v2.gltf" ||
            gpuDrivenStreamAssetSample.graph.dirty()) {
            return RhiTestResult::fail("GPUDriven StreamAsset scene override failed");
        }

        render::RenderSampleLoadResult gpuDrivenTerrainP0Sample;
        if (!render::loadBuiltInRenderSample("gpu-driven-terrain-p0", gpuDrivenTerrainP0Sample, message)) {
            return RhiTestResult::fail(message);
        }
        if (gpuDrivenTerrainP0Sample.desc.id != "gpu-driven-terrain-p0" ||
            gpuDrivenTerrainP0Sample.desc.name != "GPUDrivenSample / Terrain P0" ||
            gpuDrivenTerrainP0Sample.desc.category != "GPUDriven" ||
            gpuDrivenTerrainP0Sample.desc.scenePath !=
                "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf" ||
            gpuDrivenTerrainP0Sample.desc.loadSceneInEditor ||
            gpuDrivenTerrainP0Sample.desc.graphPath !=
                "Pipelines/Samples/gpu_driven_terrain_p0_streamasset.metallic_graph.json" ||
            gpuDrivenTerrainP0Sample.desc.previewOutput != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDriven Terrain P0 sample metadata did not load as expected");
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
            return RhiTestResult::fail("GPUDriven Terrain P0 sample did not preserve terrain defaults");
        }
        if (!gpuDrivenTerrainP0Sample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (gpuDrivenTerrainP0Sample.graph.firstOutputName() != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDriven Terrain P0 graph first output changed");
        }

        render::RenderSampleLoadResult gpuDrivenTerrainP1Sample;
        if (!render::loadBuiltInRenderSample(
                "gpu-driven-terrain-p1-unified",
                gpuDrivenTerrainP1Sample,
                message)) {
            return RhiTestResult::fail(message);
        }
        if (gpuDrivenTerrainP1Sample.desc.id != "gpu-driven-terrain-p1-unified" ||
            gpuDrivenTerrainP1Sample.desc.name != "GPUDrivenSample / Terrain P1 Unified" ||
            gpuDrivenTerrainP1Sample.desc.category != "GPUDriven" ||
            gpuDrivenTerrainP1Sample.desc.scenePath !=
                "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf" ||
            !gpuDrivenTerrainP1Sample.desc.loadSceneInEditor ||
            gpuDrivenTerrainP1Sample.desc.graphPath !=
                "Pipelines/Samples/gpu_driven_terrain_p1_unified.metallic_graph.json" ||
            !gpuDrivenTerrainP1Sample.desc.environment.has_value() ||
            gpuDrivenTerrainP1Sample.desc.previewOutput != "GPUDriven.color") {
            return RhiTestResult::fail(
                "GPUDriven Terrain P1 unified sample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDrivenTerrainP1 =
            gpuDrivenTerrainP1Sample.graph.findNode("GPUDriven");
        if (gpuDrivenTerrainP1 == nullptr ||
            gpuDrivenTerrainP1->type != "GPUDrivenPreviewPass" ||
            !gpuDrivenTerrainP1->properties.is_object() ||
            gpuDrivenTerrainP1->properties.value("path", "") !=
                gpuDrivenTerrainP1Sample.desc.scenePath ||
            gpuDrivenTerrainP1->properties.value("streamAssetPath", "") !=
                "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf.meshstream.bin" ||
            !gpuDrivenTerrainP1->properties.value("enableMeshletStreaming", false) ||
            !gpuDrivenTerrainP1->properties.value("instanceHzbCull", false) ||
            !gpuDrivenTerrainP1->properties.value("meshletFrustumCull", false) ||
            gpuDrivenTerrainP1->properties.value("mode", "") != "shaded" ||
            !gpuDrivenTerrainP1->properties.contains("camera") ||
            !gpuDrivenTerrainP1->properties["camera"].is_object()) {
            return RhiTestResult::fail(
                "GPUDriven Terrain P1 unified sample did not preserve unified raster defaults");
        }
        if (!gpuDrivenTerrainP1Sample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (gpuDrivenTerrainP1Sample.graph.firstOutputName() != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDriven Terrain P1 unified graph first output changed");
        }

        render::RenderSampleLoadResult gpuDrivenRtasSample;
        if (!render::loadBuiltInRenderSample("gpu-driven-rtas-visualization", gpuDrivenRtasSample, message)) {
            return RhiTestResult::fail(message);
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
            gpuDrivenRtasSample.desc.previewOutput != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDriven RTAS visualization sample metadata did not load as expected");
        }
        const render::RenderGraphNode* gpuDrivenRtas = gpuDrivenRtasSample.graph.findNode("GPUDriven");
        if (gpuDrivenRtas == nullptr ||
            gpuDrivenRtas->type != "GPUDrivenStreamAssetPass" ||
            !gpuDrivenRtas->properties.is_object() ||
            gpuDrivenRtas->properties.value("path", "") != gpuDrivenRtasSample.desc.scenePath ||
            !gpuDrivenRtas->properties.value("enableClusterRtx", false) ||
            !gpuDrivenRtas->properties.value("rtasVisualization", false) ||
            gpuDrivenRtas->properties.value("rtasGranularity", "") != "cluster-id") {
            return RhiTestResult::fail("GPUDriven RTAS visualization sample did not apply defaults");
        }
        if (!gpuDrivenRtasSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (gpuDrivenRtasSample.graph.firstOutputName() != "GPUDriven.color") {
            return RhiTestResult::fail("GPUDriven RTAS visualization graph first output changed");
        }

        bool listedPathTrace = false;
        bool listedOpenPBRPathTrace = false;
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
            !listedDlssRrPathTrace ||
            !listedMaterialVisualization ||
            !listedGPUDriven ||
            !listedGPUDrivenStreamAsset ||
            !listedGPUDrivenTerrainP0 ||
            !listedGPUDrivenTerrainP1 ||
            !listedGPUDrivenRtasVisualization ||
            !listedRtxcr) {
            return RhiTestResult::fail("built-in Sample list did not contain expected samples");
        }
        return RhiTestResult::pass();
    }
};

class RenderSampleFallbackAndValidationTest : public RhiTest {
public:
    RenderSampleFallbackAndValidationTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_sample_fallback_and_validation";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        const TestPathTraceSample fallback(
            "test-fallback-preview",
            "Asset/StandfordBunny/scene.gltf",
            "");

        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadRenderSample(fallback, sample, message)) {
            return RhiTestResult::fail(message);
        }
        if (sample.desc.previewOutput != "PathTrace.color") {
            return RhiTestResult::fail("Sample loader did not fallback to first graph output");
        }
        const render::RenderGraphNode* node = sample.graph.findNode("PathTrace");
        if (node == nullptr || node->properties.value("path", "") != "Asset/StandfordBunny/scene.gltf") {
            return RhiTestResult::fail("Sample loader did not override target scene path");
        }

        const TestPathTraceSample invalid(
            "test-invalid-preview",
            "Asset/meet_mat.glb",
            "Missing.color");
        if (render::loadRenderSample(invalid, sample, message)) {
            return RhiTestResult::fail("Sample loader accepted invalid previewOutput");
        }
        if (message.find("previewOutput") == std::string::npos) {
            return RhiTestResult::fail("Sample loader did not report previewOutput failure");
        }
        return RhiTestResult::pass();
    }
};
class RenderGraphValidationTest : public RhiTest {
public:
    RenderGraphValidationTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_validation";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerTestPass();

        std::string log;
        render::RenderGraph missingOutput;
        missingOutput.addNode("TriangleRasterPass", "Triangle");
        if (missingOutput.validate(log)) {
            return RhiTestResult::fail("graph without outputs validated successfully");
        }

        render::RenderGraph badEndpoint = render::RenderGraph::createDefaultTriangleGraph();
        badEndpoint.addEdge("Triangle.color", "Triangle.missing");
        if (badEndpoint.validate(log)) {
            return RhiTestResult::fail("graph with invalid edge endpoint validated successfully");
        }

        render::RenderGraph cyclic;
        cyclic.addNode("TestInputOutputPass", "A");
        cyclic.addNode("TestInputOutputPass", "B");
        cyclic.addEdge("A.color", "B.input");
        cyclic.addEdge("B.color", "A.input");
        cyclic.markOutput("A.color");
        if (cyclic.validate(log)) {
            return RhiTestResult::fail("cyclic graph validated successfully");
        }

        render::RenderGraph textureToBuffer;
        textureToBuffer.addNode("TriangleRasterPass", "Triangle");
        textureToBuffer.addNode("TestBufferInputPass", "BufferRead");
        textureToBuffer.addEdge("Triangle.color", "BufferRead.data");
        textureToBuffer.markOutput("Triangle.color");
        if (textureToBuffer.validate(log)) {
            return RhiTestResult::fail("texture-to-buffer edge validated successfully");
        }

        render::RenderGraph bufferToTexture;
        bufferToTexture.addNode("TestBufferOutputPass", "BufferWrite");
        bufferToTexture.addNode("TestInputOutputPass", "TextureRead");
        bufferToTexture.addEdge("BufferWrite.data", "TextureRead.input");
        bufferToTexture.markOutput("TextureRead.color");
        if (bufferToTexture.validate(log)) {
            return RhiTestResult::fail("buffer-to-texture edge validated successfully");
        }

        render::RenderGraph missingBufferInput;
        missingBufferInput.addNode("RenderGraphBufferCopyPass", "Copy");
        missingBufferInput.markOutput("Copy.data");
        if (missingBufferInput.validate(log)) {
            return RhiTestResult::fail("graph with missing required buffer input validated successfully");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphPreviewTest : public RhiTest {
public:
    RenderGraphPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_triangle_preview";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph = render::RenderGraph::createDefaultTriangleGraph();
        result = preview.render(graph, 128, 96);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result));
        }
        if (countBrightPixels(preview.pixels()) < 128) {
            return RhiTestResult::fail("default triangle graph produced too few bright pixels");
        }

        graph.markDirty();
        result = preview.render(graph, 64, 64);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render resize returned ") + toString(result));
        }
        if (preview.width() != 64 || preview.height() != 64) {
            return RhiTestResult::fail("preview resize did not update output dimensions");
        }
        if (countBrightPixels(preview.pixels()) < 64) {
            return RhiTestResult::fail("resized default triangle graph produced too few bright pixels");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphBunnyWireframePreviewTest : public RhiTest {
public:
    RenderGraphBunnyWireframePreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_bunny_wireframe_preview";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph = render::RenderGraph::createDefaultBunnyGraph();
        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("Bunny wireframe preview is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("Bunny wireframe preview render returned ") + toString(result) + ": " + preview.lastLog());
        }

        const uint32_t brightPixels = countBrightPixels(preview.pixels());
        if (brightPixels < 512) {
            return RhiTestResult::fail(
                std::string("Bunny wireframe preview produced too few bright pixels: ") +
                std::to_string(brightPixels));
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphBunnyCameraSyncTest : public RhiTest {
public:
    RenderGraphBunnyCameraSyncTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_bunny_camera_sync";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph = render::RenderGraph::createDefaultBunnyGraph();
        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("Bunny wireframe preview is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("initial Bunny wireframe preview returned ") + toString(result) + ": " + preview.lastLog());
        }
        if (graph.dirty()) {
            return RhiTestResult::fail("preview render did not clear graph dirty state");
        }

        render::RenderGraphNode* bunnyNode = graph.findNode("Bunny");
        if (bunnyNode == nullptr) {
            return RhiTestResult::fail("default Bunny graph did not create Bunny node");
        }

        if (!graph.setNodeRuntimeProperty(bunnyNode->id, "camera.fovDegrees", 35.0f) ||
            !graph.setNodeRuntimeProperty(bunnyNode->id, "camera.eye", {-0.0168404f, 0.110154f, 0.34f})) {
            return RhiTestResult::fail("runtime camera property update failed");
        }
        if (graph.dirty()) {
            return RhiTestResult::fail("runtime camera property update unexpectedly marked graph dirty");
        }
        result = preview.render(graph, 256, 256);
        if (!result) {
            return RhiTestResult::fail(
                std::string("camera-synced Bunny preview returned ") + toString(result) + ": " + preview.lastLog());
        }
        const uint32_t brightPixels = countBrightPixels(preview.pixels());
        if (brightPixels < 512) {
            return RhiTestResult::fail(
                std::string("camera-synced Bunny preview produced too few bright pixels: ") +
                std::to_string(brightPixels));
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphSceneRayQueryVisualizationPreviewTest : public RhiTest {
public:
    RenderGraphSceneRayQueryVisualizationPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_scene_rayquery_visualization_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
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
                return RhiTestResult::skip(
                    std::string("SceneRayQueryVisualizationPass is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("SceneRayQueryVisualizationPass render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("RayQuery instance visualization produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        properties["granularity"] = "primitive";
        render::RenderGraphNode* node = graph.findNode("RayQuery");
        if (node == nullptr || !graph.setNodeProperties(node->id, properties)) {
            return RhiTestResult::fail("failed to switch RayQuery visualization to primitive granularity");
        }

        result = preview.render(graph, 256, 256);
        if (!result) {
            return RhiTestResult::fail(
                std::string("RayQuery primitive visualization render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("RayQuery primitive visualization produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string resultMessage;
        properties["granularity"] = "cluster-id";
        node = graph.findNode("RayQuery");
        if (node == nullptr || !graph.setNodeProperties(node->id, properties)) {
            return RhiTestResult::fail("failed to switch RayQuery visualization to cluster-id granularity");
        }

        result = preview.render(graph, 256, 256);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                resultMessage = std::string("cluster-id visualization unsupported: ") + preview.lastLog() + "; ";
            } else {
                return RhiTestResult::fail(
                    std::string("RayQuery cluster-id visualization render returned ") +
                    toString(result) +
                    ": " +
                    preview.lastLog());
            }
        } else {
            visiblePixelCount = countVisiblePixels(preview.pixels());
            if (visiblePixelCount < 512) {
                return RhiTestResult::fail(
                    std::string("RayQuery cluster-id visualization produced too few visible pixels: ") +
                    std::to_string(visiblePixelCount));
            }
        }

        std::string outputMessage;
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_rayquery_visualization_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(resultMessage + "wrote " + outputPath.string());
    }
};

class RenderGraphSceneMaterialVisualizationPreviewTest : public RhiTest {
public:
    RenderGraphSceneMaterialVisualizationPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_scene_material_visualization_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("material-visualization-abeautiful-game", sample, message)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
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
                return RhiTestResult::fail("material visualization Sample graph is missing MaterialViz properties");
            }
            if (!sample.graph.setNodeRuntimeProperty(materialViz->id, "mode", mode)) {
                return RhiTestResult::fail(std::string("failed to set runtime material visualization mode ") + mode);
            }
            if (sample.graph.dirty()) {
                return RhiTestResult::fail(std::string("runtime material visualization mode dirtied graph: ") + mode);
            }

            result = preview.render(sample.graph, 160, 160, sample.desc.previewOutput);
            if (sample.graph.firstOutputName() != graphOutputBefore ||
                sample.graph.outputs().size() != graphOutputCountBefore) {
                return RhiTestResult::fail("preview output render modified graph outputs");
            }
            if (!result) {
                if (render::hasError(result, render::Error::Unsupported)) {
                    return RhiTestResult::skip(
                        std::string("SceneMaterialVisualizationPass is unsupported on this device: ") +
                        preview.lastLog());
                }
                return RhiTestResult::fail(
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
                return RhiTestResult::fail(
                    std::string("material visualization mode produced too few visible pixels: ") +
                    mode +
                    " visible=" +
                    std::to_string(visiblePixelCount));
            }
            if (modeName == "material") {
                const uint32_t distinctColorBins = countDistinctVisibleColorBins(preview.pixels());
                if (distinctColorBins < 4) {
                    return RhiTestResult::fail(
                        std::string("material visualization expected multiple material colors, got bins=") +
                        std::to_string(distinctColorBins));
                }
            }

            const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
            const std::filesystem::path outputPath =
                context.outputDirectory /
                (std::string("render_graph_scene_material_visualization_") + mode + ".png");
            if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
                return RhiTestResult::fail(message);
            }
        }

        return RhiTestResult::pass("wrote scene material visualization previews");
    }
};
class RenderGraphScenePathTracePreviewTest : public RhiTest {
public:
    RenderGraphScenePathTracePreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_scene_path_trace_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
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
                return RhiTestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("ScenePathTracePass render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("ScenePathTracePass produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        result = preview.render(graph, 192, 192);
        if (!result) {
            return RhiTestResult::fail(
                std::string("ScenePathTracePass accumulated render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("ScenePathTracePass accumulated frame produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string outputMessage;
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class SlangShaderDiskCacheTest : public RhiTest {
public:
    SlangShaderDiskCacheTest()
    {
        type = RhiTestType::Resource;
        name = "slang_shader_disk_cache_and_source_invalidation";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        struct ShaderDebugModeGuard {
            render::SlangShaderDebugMode previousMode = render::slangShaderDebugMode();

            ~ShaderDebugModeGuard()
            {
                render::setSlangShaderDebugMode(previousMode);
            }
        } shaderDebugModeGuard;
        render::setSlangShaderDebugMode(render::SlangShaderDebugMode::Disabled);

        const std::filesystem::path testRoot =
            context.outputDirectory / "slang_shader_disk_cache";
        std::error_code fileError;
        std::filesystem::remove_all(testRoot, fileError);
        if (fileError) {
            return RhiTestResult::fail(
                "failed to clear shader cache test directory: " + fileError.message());
        }
        const std::filesystem::path sourceDirectory = testRoot / "source";
        const std::filesystem::path cacheDirectory = testRoot / "cache";
        std::filesystem::create_directories(sourceDirectory, fileError);
        if (fileError) {
            return RhiTestResult::fail(
                "failed to create shader cache test directory: " + fileError.message());
        }
        const std::filesystem::path sourcePath = sourceDirectory / "ShaderCacheTest.slang";
        const std::filesystem::path dependencyPath =
            sourceDirectory / "ShaderCacheValue.slang";
        const auto writeShader = [&]() {
            std::ofstream stream(sourcePath, std::ios::binary | std::ios::trunc);
            stream << "#include \"ShaderCacheValue.slang\"\n"
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
            return RhiTestResult::fail("failed to write initial shader cache test source");
        }

        const std::string sourceDirectoryString = sourceDirectory.string();
        const std::string cacheDirectoryString = cacheDirectory.string();
        const render::SlangShaderDesc shaderDesc{
            .moduleName = "ShaderCacheTest",
            .entryPointName = "shaderCacheMain",
            .searchPath = sourceDirectoryString.c_str(),
        };
        bool cacheHit = false;
        const render::SlangShaderCacheOptions cacheOptions{
            .cacheDirectory = cacheDirectoryString.c_str(),
            .outCacheHit = &cacheHit,
        };

        render::ShaderCompileResult firstCompile;
        render::Result result = render::compileSlangShaderToSpirv(
            shaderDesc,
            cacheOptions,
            firstCompile);
        if (!result || firstCompile.spirv.empty() || cacheHit) {
            return RhiTestResult::fail(
                std::string("initial shader cache compile returned ") +
                toString(result) +
                ": " +
                firstCompile.diagnostics);
        }

        render::ShaderCompileResult cachedCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, cachedCompile);
        if (!result || !cacheHit ||
            cachedCompile.spirv != firstCompile.spirv) {
            return RhiTestResult::fail("unchanged shader source did not hit the SPIR-V disk cache");
        }

        if (!writeDependency(123456u)) {
            return RhiTestResult::fail("failed to update shader cache dependency source");
        }
        render::ShaderCompileResult changedCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, changedCompile);
        if (!result || changedCompile.spirv.empty() || cacheHit ||
            changedCompile.spirv == firstCompile.spirv) {
            return RhiTestResult::fail("changed shader dependency did not invalidate the SPIR-V cache");
        }

        render::ShaderCompileResult changedCachedCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, changedCachedCompile);
        if (!result || !cacheHit ||
            changedCachedCompile.spirv != changedCompile.spirv) {
            return RhiTestResult::fail("rebuilt shader did not become the new disk cache entry");
        }

        render::setSlangShaderDebugMode(render::SlangShaderDebugMode::CaptureSymbols);
        render::ShaderCompileResult symbolCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, symbolCompile);
        if (!result || symbolCompile.spirv.empty() || cacheHit ||
            !spirvContainsExtendedInstructionSet(
                symbolCompile.spirv,
                "NonSemantic.Shader.DebugInfo.100")) {
            return RhiTestResult::fail(
                "capture-symbol shader compile did not emit NonSemantic debug information");
        }
        render::ShaderCompileResult cachedSymbolCompile;
        result = render::compileSlangShaderToSpirv(shaderDesc, cacheOptions, cachedSymbolCompile);
        if (!result || !cacheHit || cachedSymbolCompile.spirv != symbolCompile.spirv) {
            return RhiTestResult::fail("capture-symbol shader did not use its isolated cache entry");
        }

        render::setSlangShaderDebugMode(render::SlangShaderDebugMode::ShaderDebug);
        render::ShaderCompileResult unoptimizedDebugCompile;
        result = render::compileSlangShaderToSpirv(
            shaderDesc,
            cacheOptions,
            unoptimizedDebugCompile);
        if (!result || unoptimizedDebugCompile.spirv.empty() || cacheHit ||
            !spirvContainsExtendedInstructionSet(
                unoptimizedDebugCompile.spirv,
                "NonSemantic.Shader.DebugInfo.100")) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                "shader cache did not isolate normal, capture-symbol, and shader-debug modes");
        }
        return RhiTestResult::pass(
            "validated SPIR-V cache invalidation and isolated NonSemantic debug modes");
    }
};

class RenderGraphOpenPBRPathTracingShaderCompileTest : public RhiTest {
public:
    RenderGraphOpenPBRPathTracingShaderCompileTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_openpbr_pathtracing_shader_compile";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::ShaderCompileResult compileResult;
        const char* capabilities[] = {"spvRayQueryKHR"};
        render::Result result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "OpenPBRRayQueryPathTrace",
                .entryPointName = "openPbrRayQueryPathTraceMain",
                .searchPath = kShaderSearchPath,
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            compileResult);
        if (!result) {
            return RhiTestResult::fail(
                std::string("OpenPBR RayQuery path tracing shader compile returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        if (compileResult.spirv.empty()) {
            return RhiTestResult::fail("OpenPBR RayQuery path tracing shader produced empty SPIR-V");
        }
        return RhiTestResult::pass(
            std::string("compiled OpenPBR RayQuery path tracing shader, words=") +
            std::to_string(compileResult.spirv.size()));
    }
};

class GPUDrivenPreviewGeometryDedupPlanTest : public RhiTest {
public:
    GPUDrivenPreviewGeometryDedupPlanTest()
    {
        type = RhiTestType::Validation;
        name = "gpu_driven_preview_geometry_dedup_plan";
    }

    RhiTestResult run(RhiTestContext&) override
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
            return RhiTestResult::fail(
                "shared geometry was not deduplicated or conflicting payload fallback was lost");
        }
        return RhiTestResult::pass(
            "two instances share one geometry payload; conflicting and distinct keys remain separate");
    }
};

class RenderGraphGPUDrivenPreviewShaderCompileTest : public RhiTest {
public:
    RenderGraphGPUDrivenPreviewShaderCompileTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_preview_shader_compile";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::ShaderCompileResult meshCompile;
        const char* capabilities[] = {"spvMeshShadingEXT"};
        render::Result result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenPreview",
                .entryPointName = "gpuDrivenPreviewMeshMain",
                .searchPath = kShaderSearchPath,
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            meshCompile);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreview mesh shader compile returned ") +
                toString(result) +
                ": " +
                meshCompile.diagnostics);
        }
        if (meshCompile.spirv.empty()) {
            return RhiTestResult::fail("GPUDrivenPreview mesh shader produced empty SPIR-V");
        }

        render::ShaderCompileResult fragmentCompile;
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenPreview",
                .entryPointName = "gpuDrivenPreviewFragmentMain",
                .searchPath = kShaderSearchPath,
            },
            fragmentCompile);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreview fragment shader compile returned ") +
                toString(result) +
                ": " +
                fragmentCompile.diagnostics);
        }
        if (fragmentCompile.spirv.empty()) {
            return RhiTestResult::fail("GPUDrivenPreview fragment shader produced empty SPIR-V");
        }

        constexpr std::array<const char*, 7> additionalEntryPoints{
            "gpuDrivenPreviewMaskedFragmentMain",
            "gpuDrivenPreviewResetMain",
            "gpuDrivenPreviewInstanceCullMain",
            "gpuDrivenPreviewCompactMain",
            "gpuDrivenPreviewHzbMain",
            "gpuDrivenPreviewCompositeVertexMain",
            "gpuDrivenPreviewCompositeFragmentMain",
        };
        size_t additionalWordCount = 0;
        for (const char* entryPoint : additionalEntryPoints) {
            render::ShaderCompileResult compile;
            result = render::compileSlangShaderToSpirv(
                render::SlangShaderDesc{
                    .moduleName = "GPUDrivenPreview",
                    .entryPointName = entryPoint,
                    .searchPath = kShaderSearchPath,
                },
                compile);
            if (!result) {
                return RhiTestResult::fail(
                    std::string("GPUDrivenPreview shader compile returned ") +
                    toString(result) +
                    " for " +
                    entryPoint +
                    ": " +
                    compile.diagnostics);
            }
            if (compile.spirv.empty()) {
                return RhiTestResult::fail(
                    std::string("GPUDrivenPreview shader produced empty SPIR-V for ") + entryPoint);
            }
            additionalWordCount += compile.spirv.size();
        }

        render::ShaderCompileResult deferredCompile;
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenDeferred",
                .entryPointName = "gpuDrivenPreviewDeferredMain",
                .searchPath = kShaderSearchPath,
            },
            deferredCompile);
        if (!result || deferredCompile.spirv.empty()) {
            return RhiTestResult::fail(
                std::string("GPUDriven OpenPBR deferred shader compile returned ") +
                toString(result) +
                ": " +
                deferredCompile.diagnostics);
        }
        additionalWordCount += deferredCompile.spirv.size();

        return RhiTestResult::pass(
            std::string("compiled GPUDrivenPreview shaders, mesh words=") +
            std::to_string(meshCompile.spirv.size()) +
            ", fragment words=" +
            std::to_string(fragmentCompile.spirv.size()) +
            ", additional words=" +
            std::to_string(additionalWordCount));
    }
};

class RenderGraphGPUDrivenStreamAssetShaderCompileTest : public RhiTest {
public:
    RenderGraphGPUDrivenStreamAssetShaderCompileTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_streamasset_shader_compile";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::ShaderCompileResult meshCompile;
        const char* capabilities[] = {"spvMeshShadingEXT"};
        render::Result result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetMeshMain",
                .searchPath = kShaderSearchPath,
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            meshCompile);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenStreamAsset mesh shader compile returned ") +
                toString(result) +
                ": " +
                meshCompile.diagnostics);
        }
        if (meshCompile.spirv.empty()) {
            return RhiTestResult::fail("GPUDrivenStreamAsset mesh shader produced empty SPIR-V");
        }

        render::ShaderCompileResult fragmentCompile;
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetFragmentMain",
                .searchPath = kShaderSearchPath,
            },
            fragmentCompile);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenStreamAsset fragment shader compile returned ") +
                toString(result) +
                ": " +
                fragmentCompile.diagnostics);
        }
        if (fragmentCompile.spirv.empty()) {
            return RhiTestResult::fail("GPUDrivenStreamAsset fragment shader produced empty SPIR-V");
        }

        constexpr std::array<const char*, 6> kRasterEntryPoints{
            render::kMeshletStreamDeferredEntryPoint,
            render::kMeshletStreamCompositeVertexEntryPoint,
            render::kMeshletStreamCompositeFragmentEntryPoint,
            render::kMeshletStreamCullResetEntryPoint,
            render::kMeshletStreamInstanceCullEntryPoint,
            render::kMeshletStreamHzbEntryPoint,
        };
        for (const char* entryPoint : kRasterEntryPoints) {
            render::ShaderCompileResult rasterCompile;
            result = render::compileSlangShaderToSpirv(
                render::SlangShaderDesc{
                    .moduleName = "GPUDrivenStreamAsset",
                    .entryPointName = entryPoint,
                    .searchPath = kShaderSearchPath,
                },
                rasterCompile);
            if (!result) {
                return RhiTestResult::fail(
                    std::string("GPUDrivenStreamAsset shader compile returned ") +
                    toString(result) +
                    " for " +
                    entryPoint +
                    ": " +
                    rasterCompile.diagnostics);
            }
            if (rasterCompile.spirv.empty()) {
                return RhiTestResult::fail(
                    std::string("GPUDrivenStreamAsset shader produced empty SPIR-V for ") +
                    entryPoint);
            }
        }

        render::ShaderCompileResult updateCompile;
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetApplyUpdatesMain",
                .searchPath = kShaderSearchPath,
            },
            updateCompile);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenStreamAsset update shader compile returned ") +
                toString(result) +
                ": " +
                updateCompile.diagnostics);
        }
        if (updateCompile.spirv.empty()) {
            return RhiTestResult::fail("GPUDrivenStreamAsset update shader produced empty SPIR-V");
        }

        render::ShaderCompileResult traversalCompile;
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetTraversalMain",
                .searchPath = kShaderSearchPath,
            },
            traversalCompile);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenStreamAsset traversal shader compile returned ") +
                toString(result) +
                ": " +
                traversalCompile.diagnostics);
        }
        if (traversalCompile.spirv.empty()) {
            return RhiTestResult::fail("GPUDrivenStreamAsset traversal shader produced empty SPIR-V");
        }

        render::ShaderCompileResult activeBuildCompile;
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetBuildActiveMain",
                .searchPath = kShaderSearchPath,
            },
            activeBuildCompile);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenStreamAsset active build shader compile returned ") +
                toString(result) +
                ": " +
                activeBuildCompile.diagnostics);
        }
        if (activeBuildCompile.spirv.empty()) {
            return RhiTestResult::fail("GPUDrivenStreamAsset active build shader produced empty SPIR-V");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphGPUDrivenStreamAssetTraversalDemandTest : public RhiTest {
public:
    RenderGraphGPUDrivenStreamAssetTraversalDemandTest()
    {
        type = RhiTestType::Command;
        name = "render_graph_gpu_driven_streamasset_traversal_demand";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint32_t kMaxLoadRequests = 8;
        constexpr uint32_t kMaxUnloadRequests = 8;
        constexpr uint32_t kTraversalWorkCapacity = 32;
        constexpr uint32_t kFrameIndex = 7;
        constexpr uint64_t kRequestByteSize =
            sizeof(render::StreamRequestBufferHeader) +
            (static_cast<uint64_t>(kMaxLoadRequests) + kMaxUnloadRequests) * sizeof(uint32_t);

        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic GPUDrivenStreamAsset traversal demand test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::Queue* queue = device->getQueue(render::QueueType::Compute);
        if (queue == nullptr) {
            queue = device->getQueue(render::QueueType::Graphics);
        }
        if (queue == nullptr) {
            return RhiTestResult::skip("traversal demand test device has no compute-capable queue");
        }

        auto createBuffer = [&device](
            const render::BufferDesc& desc,
            const char* label,
            std::unique_ptr<render::Buffer>& outBuffer) -> RhiTestResult {
            render::Result bufferResult = device->createBuffer(desc, outBuffer);
            if (!bufferResult || outBuffer == nullptr) {
                return RhiTestResult::fail(std::string("createBuffer(") + label + ") returned " + toString(bufferResult));
            }
            return RhiTestResult::pass();
        };

        constexpr uint32_t kActiveGroupCapacity = 8;
        const std::array<render::MeshletStreamGpuInstance, 2> instances = [] {
            std::array<render::MeshletStreamGpuInstance, 2> values{};
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
        const std::array<render::MeshletStreamGpuPrimitive, 2> primitives = {{
            render::MeshletStreamGpuPrimitive{
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
            render::MeshletStreamGpuPrimitive{
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
        const std::array<render::MeshletStreamGpuLodLevel, 2> lodLevels = {{
            render::MeshletStreamGpuLodLevel{
                .pageOffset = 0,
                .pageCount = 2,
                .lodLevel = 0,
                .clusterCount = 2,
                .minBoundingSphereRadius = 1.0f,
                .minMaxQuadricError = 0.0f,
            },
            render::MeshletStreamGpuLodLevel{
                .pageOffset = 3,
                .pageCount = 1,
                .lodLevel = 0,
                .clusterCount = 1,
                .minBoundingSphereRadius = 1.0f,
                .minMaxQuadricError = 0.0f,
            },
        }};
        constexpr uint32_t kScenePageCount = 6;
        std::array<render::MeshletStreamGpuGroup, 5> groups{};
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
        std::array<render::MeshletStreamGpuNode, 8> nodes{};
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
        for (render::MeshletStreamGpuNode& node : nodes) {
            const uint32_t groupIndex = node.groupIndex;
            node.boundsCenterRadius[2] = groupIndex < groups.size()
                ? groups[groupIndex].boundsCenterRadius[2]
                : 5.5f;
            node.boundsCenterRadius[3] = 1.0f;
        }
        render::MeshletStreamGpuParams params;
        params.viewport[2] = 96.0f;
        params.viewport[3] = 1.0471975512f;
        params.frameIndex = kFrameIndex;
        params.maxGpuPageRequests = kMaxLoadRequests;
        params.maxGpuPageUnloadRequests = kMaxUnloadRequests;
        params.sceneInstanceCount = static_cast<uint32_t>(instances.size());
        params.scenePrimitiveCount = static_cast<uint32_t>(primitives.size());
        params.sceneLodLevelCount = static_cast<uint32_t>(lodLevels.size());
        params.scenePageCount = kScenePageCount;
        params.selectedLodLevel = render::kMeshletStreamNoDebugLodOverride;
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
        RhiTestResult testResult = createBuffer(
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
                .size = static_cast<uint64_t>(kActiveGroupCapacity) * sizeof(render::MeshletStreamGpuActiveGroup),
                .structureStride = sizeof(render::MeshletStreamGpuActiveGroup),
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
                .size = sizeof(render::MeshletStreamGpuActiveHeader),
                .structureStride = sizeof(render::MeshletStreamGpuActiveHeader),
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
                .size = sizeof(render::MeshletStreamGpuDrawIndirect),
                .structureStride = sizeof(render::MeshletStreamGpuDrawIndirect),
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
                .size = sizeof(render::MeshletStreamGpuTraversalHeader),
                .structureStride = sizeof(render::MeshletStreamGpuTraversalHeader),
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
                    sizeof(render::MeshletStreamGpuTraversalWorkItem),
                .structureStride = sizeof(render::MeshletStreamGpuTraversalWorkItem),
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
                .size = sizeof(render::MeshletStreamGpuDrawIndirect),
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
                .size = sizeof(render::MeshletStreamGpuTraversalHeader),
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
            return RhiTestResult::fail(std::string("writeHostBuffer(instances) returned ") + toString(result));
        }
        result = writeHostBuffer(*primitiveBuffer, primitives.data(), sizeof(primitives));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(primitives) returned ") + toString(result));
        }
        result = writeHostBuffer(*lodLevelBuffer, lodLevels.data(), sizeof(lodLevels));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(lod levels) returned ") + toString(result));
        }
        result = writeHostBuffer(*groupBuffer, groups.data(), sizeof(groups));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(groups) returned ") + toString(result));
        }
        result = writeHostBuffer(
            *pageBuffer,
            pageWords.data(),
            static_cast<uint64_t>(pageWords.size()) * sizeof(uint32_t));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(stream pages) returned ") + toString(result));
        }
        result = writeHostBuffer(*nodeBuffer, nodes.data(), sizeof(nodes));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(hierarchy nodes) returned ") + toString(result));
        }
        result = writeHostBuffer(*paramsBuffer, &params, sizeof(params));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(params) returned ") + toString(result));
        }
        result = writeHostBuffer(*pageTableUploadBuffer, pageTable.data(), sizeof(pageTable));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(page table upload) returned ") + toString(result));
        }
        result = writeHostBuffer(*requestUploadBuffer, requestInit.data(), kRequestByteSize);
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(request upload) returned ") + toString(result));
        }
        result = writeHostBuffer(*residentPageBuffer, residentPageIds.data(), sizeof(residentPageIds));
        if (!result) {
            return RhiTestResult::fail(std::string("writeHostBuffer(resident pages) returned ") + toString(result));
        }

        std::unique_ptr<render::BindlessHeap> bindlessHeap;
        result = device->createBindlessHeap(
            render::BindlessHeapDesc{
                .maxSamplers = 0,
                .maxSampledImages = 0,
                .maxBuffers = 15,
            },
            bindlessHeap);
        if (!result || bindlessHeap == nullptr) {
            return RhiTestResult::fail(std::string("createBindlessHeap returned ") + toString(result));
        }

        auto allocateStorageBuffer = [&bindlessHeap](
            render::Buffer& buffer,
            const char* label,
            render::BindlessHandle& outHandle) -> RhiTestResult {
            render::Result bindlessResult = bindlessHeap->allocateBuffer(outHandle);
            if (!bindlessResult || !outHandle.valid()) {
                return RhiTestResult::fail(std::string("allocateBuffer(") + label + ") returned " + toString(bindlessResult));
            }
            bindlessResult = bindlessHeap->writeStorageBuffer(outHandle, buffer);
            if (!bindlessResult) {
                return RhiTestResult::fail(std::string("writeStorageBuffer(") + label + ") returned " + toString(bindlessResult));
            }
            return RhiTestResult::pass();
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
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetTraversalMain",
                .searchPath = kShaderSearchPath,
            },
            compileResult);
        if (!result) {
            return RhiTestResult::fail(
                std::string("compileSlangShaderToSpirv(traversal) returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        std::unique_ptr<render::ShaderModule> traversalShader;
        result = device->createShaderModule(
            render::ShaderModuleDesc{
                .code = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            },
            traversalShader);
        if (!result || traversalShader == nullptr) {
            return RhiTestResult::fail(std::string("createShaderModule(traversal) returned ") + toString(result));
        }

        std::unique_ptr<render::ComputePipeline> pipeline;
        result = device->createComputePipeline(
            render::ComputePipelineDesc{
                .computeShader = traversalShader.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(render::MeshletStreamUserPush),
            },
            pipeline);
        if (!result || pipeline == nullptr) {
            return RhiTestResult::fail(std::string("createComputePipeline(traversal) returned ") + toString(result));
        }

        render::ShaderCompileResult activeBuildCompileResult;
        result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "GPUDrivenStreamAsset",
                .entryPointName = "gpuDrivenStreamAssetBuildActiveMain",
                .searchPath = kShaderSearchPath,
            },
            activeBuildCompileResult);
        if (!result) {
            return RhiTestResult::fail(
                std::string("compileSlangShaderToSpirv(active build) returned ") +
                toString(result) +
                ": " +
                activeBuildCompileResult.diagnostics);
        }
        std::unique_ptr<render::ShaderModule> activeBuildShader;
        result = device->createShaderModule(
            render::ShaderModuleDesc{
                .code = activeBuildCompileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(activeBuildCompileResult.spirv.size() * sizeof(uint32_t)),
            },
            activeBuildShader);
        if (!result || activeBuildShader == nullptr) {
            return RhiTestResult::fail(std::string("createShaderModule(active build) returned ") + toString(result));
        }

        std::unique_ptr<render::ComputePipeline> activeBuildPipeline;
        result = device->createComputePipeline(
            render::ComputePipelineDesc{
                .computeShader = activeBuildShader.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(render::MeshletStreamUserPush),
            },
            activeBuildPipeline);
        if (!result || activeBuildPipeline == nullptr) {
            return RhiTestResult::fail(std::string("createComputePipeline(active build) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*queue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }
        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        std::array<render::BufferBarrierDesc, 2> uploadBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::TransferDestination,
                .offset = 0,
                .size = pageTableBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::TransferDestination,
                .offset = 0,
                .size = requestBuffer->desc().size,
            },
        }};
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = uploadBarriers.data(),
            .bufferCount = static_cast<uint32_t>(uploadBarriers.size()),
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = pageTableUploadBuffer.get(),
            .destination = pageTableBuffer.get(),
            .size = pageTableBuffer->desc().size,
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = requestUploadBuffer.get(),
            .destination = requestBuffer.get(),
            .size = requestBuffer->desc().size,
        });
        std::array<render::BufferBarrierDesc, 2> generalBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = render::ResourceState::TransferDestination,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = pageTableBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = render::ResourceState::TransferDestination,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = requestBuffer->desc().size,
            },
        }};
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = generalBarriers.data(),
            .bufferCount = static_cast<uint32_t>(generalBarriers.size()),
        });

        commandBuffer->bindBindlessHeap(*bindlessHeap);
        render::MeshletStreamUserPush push{
            .pageBuffer = pageHandle.index,
            .activeGroupBuffer = activeGroupHandle.index,
            .pageTableBuffer = pageTableHandle.index,
            .paramsBuffer = paramsHandle.index,
            .requestBuffer = requestHandle.index,
            .residentPageBuffer = residentPageHandle.index,
            .activeHeaderBuffer = activeHeaderHandle.index,
            .instanceBuffer = instanceHandle.index,
            .primitiveBuffer = primitiveHandle.index,
            .lodLevelBuffer = lodLevelHandle.index,
            .groupBuffer = groupHandle.index,
            .nodeBuffer = nodeHandle.index,
            .drawIndirectBuffer = drawIndirectHandle.index,
            .traversalHeaderBuffer = traversalHeaderHandle.index,
            .traversalWorkBuffer = traversalWorkHandle.index,
            .traversalPhase = render::kMeshletStreamTraversalLoadPhase,
        };

        std::array<render::BufferBarrierDesc, 7> activeBuildBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = pageTableBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = requestBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = activeGroupBuffer.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = activeGroupBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = activeHeaderBuffer.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = activeHeaderBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = drawIndirectBuffer.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = drawIndirectBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = traversalHeaderBuffer.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = traversalHeaderBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = traversalWorkBuffer.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = traversalWorkBuffer->desc().size,
            },
        }};
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = activeBuildBarriers.data(),
            .bufferCount = static_cast<uint32_t>(activeBuildBarriers.size()),
        });

        commandBuffer->bindComputePipeline(*activeBuildPipeline);
        push.activeBuildPhase = render::kMeshletStreamActiveBuildResetPhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        std::array<render::BufferBarrierDesc, 7> activePhaseBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = pageTableBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = requestBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = activeGroupBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = activeGroupBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = activeHeaderBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = activeHeaderBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = drawIndirectBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = drawIndirectBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = traversalHeaderBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = traversalHeaderBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = traversalWorkBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::General,
                .offset = 0,
                .size = traversalWorkBuffer->desc().size,
            },
        }};
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = activePhaseBarriers.data(),
            .bufferCount = static_cast<uint32_t>(activePhaseBarriers.size()),
        });

        push.activeBuildPhase = render::kMeshletStreamActiveBuildSeedPhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        commandBuffer->barrier(render::BarrierDesc{
            .buffers = activePhaseBarriers.data(),
            .bufferCount = static_cast<uint32_t>(activePhaseBarriers.size()),
        });
        push.activeBuildPhase = render::kMeshletStreamActiveBuildRunPhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        commandBuffer->barrier(render::BarrierDesc{
            .buffers = activePhaseBarriers.data(),
            .bufferCount = static_cast<uint32_t>(activePhaseBarriers.size()),
        });
        push.activeBuildPhase = render::kMeshletStreamActiveBuildFinalizePhase;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        commandBuffer->barrier(render::BarrierDesc{
            .buffers = activePhaseBarriers.data(),
            .bufferCount = static_cast<uint32_t>(activePhaseBarriers.size()),
        });
        commandBuffer->bindComputePipeline(*pipeline);
        push.traversalPhase = render::kMeshletStreamTraversalUnloadPhase;
        push.activeBuildPhase = static_cast<uint32_t>(residentPageIds.size());
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        std::array<render::BufferBarrierDesc, 6> readbackBarriers = {{
            render::BufferBarrierDesc{
                .buffer = pageTableBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource,
                .offset = 0,
                .size = pageTableBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = requestBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource,
                .offset = 0,
                .size = requestBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = activeGroupBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource,
                .offset = 0,
                .size = activeGroupBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = activeHeaderBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource,
                .offset = 0,
                .size = activeHeaderBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = drawIndirectBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource,
                .offset = 0,
                .size = drawIndirectBuffer->desc().size,
            },
            render::BufferBarrierDesc{
                .buffer = traversalHeaderBuffer.get(),
                .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource,
                .offset = 0,
                .size = traversalHeaderBuffer->desc().size,
            },
        }};
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = readbackBarriers.data(),
            .bufferCount = static_cast<uint32_t>(readbackBarriers.size()),
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = pageTableBuffer.get(),
            .destination = pageTableReadbackBuffer.get(),
            .size = pageTableBuffer->desc().size,
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = requestBuffer.get(),
            .destination = requestReadbackBuffer.get(),
            .size = requestBuffer->desc().size,
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = activeGroupBuffer.get(),
            .destination = activeGroupReadbackBuffer.get(),
            .size = activeGroupBuffer->desc().size,
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = activeHeaderBuffer.get(),
            .destination = activeHeaderReadbackBuffer.get(),
            .size = activeHeaderBuffer->desc().size,
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = drawIndirectBuffer.get(),
            .destination = drawIndirectReadbackBuffer.get(),
            .size = drawIndirectBuffer->desc().size,
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = traversalHeaderBuffer.get(),
            .destination = traversalHeaderReadbackBuffer.get(),
            .size = traversalHeaderBuffer->desc().size,
        });
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = queue->submit(render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = fence.get(),
        });
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }
        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        std::array<render::StreamPageTableEntry, 6> pageTableResult{};
        if (!readHostBuffer(*pageTableReadbackBuffer, pageTableResult.data(), sizeof(pageTableResult))) {
            return RhiTestResult::fail("page table readback buffer did not map");
        }
        std::vector<uint8_t> requestResult(static_cast<size_t>(kRequestByteSize), 0);
        if (!readHostBuffer(*requestReadbackBuffer, requestResult.data(), kRequestByteSize)) {
            return RhiTestResult::fail("request readback buffer did not map");
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
            return RhiTestResult::fail("traversal demand shader did not emit expected load/unload requests");
        }
        if (pageTableResult[0].lastRequestFrame != kFrameIndex ||
            pageTableResult[1].lastRequestFrame != kFrameIndex ||
            pageTableResult[2].lastRequestFrame != kFrameIndex ||
            pageTableResult[3].lastRequestFrame != kFrameIndex ||
            pageTableResult[4].lastRequestFrame != kFrameIndex ||
            pageTableResult[5].lastRequestFrame == kFrameIndex) {
            return RhiTestResult::fail("traversal demand shader did not mark selected pages conservatively");
        }
        render::MeshletStreamGpuTraversalHeader traversalHeaderResult;
        if (!readHostBuffer(
                *traversalHeaderReadbackBuffer,
                &traversalHeaderResult,
                sizeof(traversalHeaderResult))) {
            return RhiTestResult::fail("traversal header readback buffer did not map");
        }
        if (traversalHeaderResult.writeCounter != nodes.size() ||
            traversalHeaderResult.readCounter < traversalHeaderResult.writeCounter ||
            traversalHeaderResult.taskCounter != 0 ||
            traversalHeaderResult.overflowCount != 0 ||
            traversalHeaderResult.frameIndex != kFrameIndex) {
            return RhiTestResult::fail("persistent traversal queue did not drain as expected");
        }
        render::MeshletStreamGpuActiveHeader activeHeaderResult;
        if (!readHostBuffer(*activeHeaderReadbackBuffer, &activeHeaderResult, sizeof(activeHeaderResult))) {
            return RhiTestResult::fail("active header readback buffer did not map");
        }
        std::array<render::MeshletStreamGpuActiveGroup, kActiveGroupCapacity> activeGroupsResult{};
        if (!readHostBuffer(*activeGroupReadbackBuffer, activeGroupsResult.data(), sizeof(activeGroupsResult))) {
            return RhiTestResult::fail("active group readback buffer did not map");
        }
        if (activeHeaderResult.activeGroupCount != 3 ||
            activeHeaderResult.activeGroupCapacity != kActiveGroupCapacity ||
            activeHeaderResult.maxActiveGroupClusters != params.maxActiveGroupClusters ||
            activeHeaderResult.overflowCount != 0 ||
            activeHeaderResult.frameIndex != kFrameIndex) {
            return RhiTestResult::fail("active table header was not built as expected");
        }
        render::MeshletStreamGpuDrawIndirect drawIndirectResult;
        if (!readHostBuffer(*drawIndirectReadbackBuffer, &drawIndirectResult, sizeof(drawIndirectResult))) {
            return RhiTestResult::fail("draw indirect readback buffer did not map");
        }
        if (drawIndirectResult.groupCountX !=
                activeHeaderResult.activeGroupCount *
                    activeHeaderResult.maxActiveGroupClusters *
                    render::kMeshletStreamTriangleChunkCount ||
            drawIndirectResult.groupCountY != 1 ||
            drawIndirectResult.groupCountZ != 1) {
            return RhiTestResult::fail("active table did not generate the expected indirect mesh task command");
        }

        bool foundResidentFinePage0 = false;
        bool foundFallbackPage = false;
        bool foundResidentFinePage = false;
        for (uint32_t index = 0; index < activeHeaderResult.activeGroupCount; ++index) {
            const render::MeshletStreamGpuActiveGroup& group = activeGroupsResult[index];
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
            return RhiTestResult::fail("active table did not compact group-level fine and fallback selections");
        }

        (void)device->waitIdle();
        return RhiTestResult::pass();
    }
};

class RenderGraphRtxdiPreviewTest : public RhiTest {
public:
    RenderGraphRtxdiPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_rtxdi_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(true, true);
        if (!result) {
            return RhiTestResult::skip(
                std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("rtxdi-sample", sample, message)) {
            return RhiTestResult::fail(message);
        }
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        constexpr uint32_t kRelaxFrameCount = 8;
        for (uint32_t frame = 0; frame < kRelaxFrameCount; ++frame) {
            result = preview.render(sample.graph, 256, 256, sample.desc.previewOutput);
            if (!result) {
                if (frame == 0 && render::hasError(result, render::Error::Unsupported)) {
                    return RhiTestResult::skip(
                        std::string("RTXDI/RELAX graph is unsupported on this device: ") + preview.lastLog());
                }
                return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                std::string("RTXDI/RELAX graph produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_rtxdi_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RhiTestResult::fail(message);
        }

        result = preview.render(sample.graph, 256, 256, "Confidence.diffuseConfidence");
        if (!result) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                "RTXDI diffuse confidence output is empty or constant");
        }
        return RhiTestResult::pass("wrote RTXDI RELAX preview");
    }
};

#if defined(METALLIC_HAS_RTXCR) && METALLIC_HAS_RTXCR
class RenderGraphRtxcrMaterialShaderCompileTest : public RhiTest {
public:
    RenderGraphRtxcrMaterialShaderCompileTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_rtxcr_material_shader_compile";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        const char* additionalSearchPaths[] = {METALLIC_RTXCR_SHADER_INCLUDE_DIR};
        render::ShaderCompileResult compileResult;
        render::Result result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "RtxcrMaterialSample",
                .entryPointName = "rtxcrMaterialSampleMain",
                .searchPath = kShaderSearchPath,
                .additionalSearchPaths = additionalSearchPaths,
                .additionalSearchPathCount =
                    static_cast<uint32_t>(std::size(additionalSearchPaths)),
            },
            compileResult);
        if (!result || compileResult.spirv.empty()) {
            return RhiTestResult::fail(
                std::string("RTXCR material shader compile returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        return RhiTestResult::pass(
            std::string("compiled RTXCR material shader, words=") +
            std::to_string(compileResult.spirv.size()));
    }
};

#if defined(METALLIC_HAS_RTXCR_GEOMETRY) && METALLIC_HAS_RTXCR_GEOMETRY && \
    defined(METALLIC_HAS_RTXCR_ASSETS) && METALLIC_HAS_RTXCR_ASSETS
class RenderGraphRtxcrMaterialPreviewTest : public RhiTest {
public:
    RenderGraphRtxcrMaterialPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_rtxcr_material_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("rtxcr-material-sample", sample, message)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(
                std::string("RenderGraphPreviewRenderer::initialize returned ") +
                toString(result));
        }
        result = preview.render(sample.graph, 768, 432, sample.desc.previewOutput);
        if (!result) {
            return RhiTestResult::fail(
                std::string("RTXCR material preview returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t brightPixelCount = countBrightPixels(preview.pixels());
        if (brightPixelCount < 1024) {
            return RhiTestResult::fail(
                std::string("RTXCR material preview produced too few bright pixels: ") +
                std::to_string(brightPixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_rtxcr_material_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RhiTestResult::fail(message);
        }
        return RhiTestResult::pass("wrote " + outputPath.string());
    }
};
#endif
#endif

class RenderGraphRtxdiShaderCompileTest : public RhiTest {
public:
    RenderGraphRtxdiShaderCompileTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_rtxdi_shader_compile";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        const char* capabilities[] = {"spvRayQueryKHR"};
        const struct ShaderEntry {
            const char* moduleName;
            const char* entryPointName;
            bool rayQuery;
        } entries[] = {
            {"BuildReGIR", "buildReGIRMain", false},
            {"PrepareLightsPdf", "prepareLightsPdfMain", false},
            {"SceneRtxdi", "sceneRtxdiMain", true},
            {"RtxdiConfidence", "rtxdiConfidenceMain", false},
            {"RtxdiComposite", "rtxdiCompositeMain", false},
        };
        for (const ShaderEntry& entry : entries) {
            render::ShaderCompileResult compileResult;
            render::Result result = render::compileSlangShaderToSpirv(
                render::SlangShaderDesc{
                    .moduleName = entry.moduleName,
                    .entryPointName = entry.entryPointName,
                    .searchPath = kShaderSearchPath,
                    .capabilities = entry.rayQuery ? capabilities : nullptr,
                    .capabilityCount = entry.rayQuery
                        ? static_cast<uint32_t>(std::size(capabilities))
                        : 0u,
                },
                compileResult);
            if (!result) {
                return RhiTestResult::fail(
                    std::string("RTXDI shader compile returned ") +
                    toString(result) +
                    ": " +
                    compileResult.diagnostics);
            }
            if (compileResult.spirv.empty()) {
                return RhiTestResult::fail("RTXDI shader produced empty SPIR-V");
            }
        }
        return RhiTestResult::pass("compiled RTXDI ReSTIR DI and RELAX composite shaders");
    }
};

class RenderGraphPathTracingGuidesShaderCompileTest : public RhiTest {
public:
    RenderGraphPathTracingGuidesShaderCompileTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_pathtracing_guides_shader_compile";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        const char* capabilities[] = {"spvRayQueryKHR"};
        const struct ShaderEntry {
            const char* moduleName;
            const char* entryPointName;
        } entries[] = {
            {"ScenePathTraceGuides", "scenePathTraceGuidesMain"},
            {"OpenPBRRayQueryPathTraceGuides", "openPbrRayQueryPathTraceGuidesMain"},
        };

        for (const ShaderEntry& entry : entries) {
            render::ShaderCompileResult compileResult;
            render::Result result = render::compileSlangShaderToSpirv(
                render::SlangShaderDesc{
                    .moduleName = entry.moduleName,
                    .entryPointName = entry.entryPointName,
                    .searchPath = kShaderSearchPath,
                    .capabilities = capabilities,
                    .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
                },
                compileResult);
            if (!result) {
                return RhiTestResult::fail(
                    std::string("Path tracing guide shader compile returned ") +
                    toString(result) +
                    " for " +
                    entry.moduleName +
                    "." +
                    entry.entryPointName +
                    ": " +
                    compileResult.diagnostics);
            }
            if (compileResult.spirv.empty()) {
                return RhiTestResult::fail(
                    std::string("Path tracing guide shader produced empty SPIR-V for ") +
                    entry.moduleName +
                    "." +
                    entry.entryPointName);
            }
        }

        return RhiTestResult::pass("compiled path tracing guide shaders");
    }
};

class RenderGraphSceneRayQueryClusterShaderCompileTest : public RhiTest {
public:
    RenderGraphSceneRayQueryClusterShaderCompileTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_scene_rayquery_cluster_shader_compile";
    }

    RhiTestResult run(RhiTestContext&) override
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
        render::Result result = render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "SceneRayQueryVisualize",
                .entryPointName = "sceneRayQueryVisualizeMain",
                .searchPath = kShaderSearchPath,
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
                .macroDefines = macros,
                .macroDefineCount = static_cast<uint32_t>(std::size(macros)),
            },
            compileResult);
        if (!result) {
            return RhiTestResult::fail(
                std::string("Cluster ray-query shader compile returned ") +
                toString(result) +
                ": " +
                compileResult.diagnostics);
        }
        if (compileResult.spirv.size() < 5 ||
            compileResult.spirv[0] != kSpirvMagic ||
            compileResult.spirv[1] != kSpirvVersion16) {
            return RhiTestResult::fail("Cluster ray-query shader did not produce a SPIR-V 1.6 module");
        }
        if (!spirvContainsCapability(
                compileResult.spirv,
                kSpirvRayTracingClusterAccelerationStructureNv)) {
            return RhiTestResult::fail(
                "Cluster ray-query shader omitted RayTracingClusterAccelerationStructureNV");
        }
        if (!spirvContainsExtension(
                compileResult.spirv,
                "SPV_NV_cluster_acceleration_structure")) {
            return RhiTestResult::fail(
                "Cluster ray-query shader omitted SPV_NV_cluster_acceleration_structure");
        }
        if (!spirvContainsOpcode(
                compileResult.spirv,
                kSpirvOpRayQueryGetIntersectionClusterIdNv)) {
            return RhiTestResult::fail(
                "Cluster ray-query shader omitted OpRayQueryGetIntersectionClusterIdNV");
        }
        return RhiTestResult::pass("compiled SPIR-V 1.6 cluster ray-query shader");
    }
};

class RenderGraphOpenPBRPathTracingSamplePreviewTest : public RhiTest {
public:
    RenderGraphOpenPBRPathTracingSamplePreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_openpbr_pathtracing_sample_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", sample, message)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr) {
            return RhiTestResult::fail("OpenPBR PathTracingSample is missing PathTrace node");
        }
        // User-reported close view whose glass sphere exposes a horizontal band.
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "maxDepth", 12) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "samples", 2) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", false) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.eye", {-0.008599f, 0.073623f, 0.058931f}) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.center", {-1.384997f, -0.182991f, 2.025709f})) {
            return RhiTestResult::fail("failed to set OpenPBR PathTracingSample preview runtime properties");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("OpenPBR PathTracingSample is unsupported on this device: ") +
                    preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("OpenPBR PathTracingSample render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        if (preview.lastLog().find("environment map does not exist") != std::string::npos ||
            preview.lastLog().find("failed to decode environment map") != std::string::npos ||
            preview.lastLog().find("decoded environment map is too large") != std::string::npos) {
            return RhiTestResult::fail(
                std::string("OpenPBR PathTracingSample did not load the HDRI environment: ") +
                preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 64) {
            return RhiTestResult::fail(
                std::string("OpenPBR PathTracingSample produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_openpbr_pathtracing_sample_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RhiTestResult::fail(message);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphOpenPBRPathTracingDebugViewsTest : public RhiTest {
public:
    RenderGraphOpenPBRPathTracingDebugViewsTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_openpbr_pathtracing_debug_views";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", sample, message)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr) {
            return RhiTestResult::fail("OpenPBR PathTracingSample is missing PathTrace node");
        }
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "maxDepth", 12) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "samples", 2) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", false) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.eye", {-0.001590f, 0.072671f, 0.069807f}) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "camera.center", {-2.046089f, 0.350581f, 1.323329f})) {
            return RhiTestResult::fail("failed to set OpenPBR debug-view camera properties");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(sampleEnvironmentSettings(sample.desc));
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
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
                    return RhiTestResult::fail(std::string("failed to clear OpenPBR debug flag ") + flag);
                }
            }
            if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "debugView", debugCase.view) ||
                (debugCase.enabledFlag != nullptr &&
                 !sample.graph.setNodeRuntimeProperty(pathTrace->id, debugCase.enabledFlag, true))) {
                return RhiTestResult::fail(std::string("failed to set OpenPBR debug case ") + debugCase.name);
            }
            if (sample.graph.dirty()) {
                return RhiTestResult::fail(std::string("OpenPBR debug case dirtied graph: ") + debugCase.name);
            }

            result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
            if (!result) {
                if (render::hasError(result, render::Error::Unsupported)) {
                    return RhiTestResult::skip(
                        std::string("OpenPBR debug views are unsupported on this device: ") +
                        preview.lastLog());
                }
                return RhiTestResult::fail(
                    std::string("OpenPBR debug case render returned ") +
                    toString(result) +
                    " for " +
                    debugCase.name +
                    ": " +
                    preview.lastLog());
            }
            if (countVisiblePixels(preview.pixels()) < 512) {
                return RhiTestResult::fail(
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
                return RhiTestResult::fail(message);
            }
        }

        for (const char* flag : debugFlags) {
            if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, flag, false)) {
                return RhiTestResult::fail(std::string("failed to clear OpenPBR stability flag ") + flag);
            }
        }
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", true) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "debugView", "shadowTransmittance")) {
            return RhiTestResult::fail("failed to configure accumulated OpenPBR debug stability check");
        }
        result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
        if (!result) {
            return RhiTestResult::fail(
                std::string("first accumulated OpenPBR debug render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const std::vector<uint32_t> firstStableDebugPixels = preview.pixels();
        result = preview.render(sample.graph, 576, 300, sample.desc.previewOutput);
        if (!result) {
            return RhiTestResult::fail(
                std::string("second accumulated OpenPBR debug render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint64_t accumulatedDebugDifference =
            sumAbsoluteRgbDifference(firstStableDebugPixels, preview.pixels());
        if (accumulatedDebugDifference != 0) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail("OpenPBR captured debug views have unexpected dimensions");
        }

        constexpr uint64_t kNormalDebugDifferenceTolerance = 1024;
        const uint64_t normalBypassDifference =
            sumAbsoluteRgbDifference(shadingNormalPixels, mappedNoNormalMapPixels);
        if (normalBypassDifference > kNormalDebugDifferenceTolerance) {
            return RhiTestResult::fail(
                "OpenPBR disable-normal-map view did not match shading normals: difference=" +
                std::to_string(normalBypassDifference));
        }
        const uint64_t geometryOverrideDifference =
            sumAbsoluteRgbDifference(geometryNormalPixels, mappedForceGeometryPixels);
        if (geometryOverrideDifference > kNormalDebugDifferenceTolerance) {
            return RhiTestResult::fail(
                "OpenPBR force-geometry-normal view did not match geometry normals: difference=" +
                std::to_string(geometryOverrideDifference));
        }
        if (sumAbsoluteRgbDifference(shadowTransmittancePixels, shadowOpaquePixels) < 1024 ||
            sumAbsoluteRgbDifference(shadowTransmittancePixels, shadowUnoccludedPixels) < 1024) {
            return RhiTestResult::fail("OpenPBR shadow debug modes did not produce distinct visibility results");
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
            return RhiTestResult::fail(
                "OpenPBR primary glass sphere debug ROI contains too few surface pixels: pixels=" +
                std::to_string(surfacePixelCount));
        }
        if (geometryNormalBins.size() < 8) {
            return RhiTestResult::fail(
                "OpenPBR geometry normals collapsed across the glass sphere: bins=" +
                std::to_string(geometryNormalBins.size()));
        }
        if (backFacePixelCount != 0) {
            return RhiTestResult::fail(
                "OpenPBR primary glass sphere contains false back faces: pixels=" +
                std::to_string(backFacePixelCount));
        }

        return RhiTestResult::pass("wrote OpenPBR path-tracing debug views");
    }
};

class RenderGraphOpenPBRPathTracingEnvironmentRotationTest : public RhiTest {
public:
    RenderGraphOpenPBRPathTracingEnvironmentRotationTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_openpbr_pathtracing_environment_rotation";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("pathtracing-sample", sample, message)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphNode* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr) {
            return RhiTestResult::fail("OpenPBR PathTracingSample is missing PathTrace node");
        }
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "maxDepth", 4) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "samples", 1) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", false)) {
            return RhiTestResult::fail("failed to set OpenPBR environment rotation test runtime properties");
        }

        render::RenderGraphPreviewRenderer preview;
        render::EnvironmentSettings environment = sampleEnvironmentSettings(sample.desc);
        environment.rotationDegrees = 0.0f;
        preview.setEnvironment(environment);
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        result = preview.render(sample.graph, 96, 96, sample.desc.previewOutput);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("OpenPBR PathTracingSample is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                std::string("rotated OpenPBR environment render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint64_t difference = sumAbsoluteRgbDifference(rotation0Pixels, preview.pixels());
        if (difference < 4096) {
            return RhiTestResult::fail(
                std::string("environment rotation did not materially affect path tracing output, diff=") +
                std::to_string(difference));
        }

        return RhiTestResult::pass(std::string("environment rotation diff=") + std::to_string(difference));
    }
};

class RenderGraphScenePathTraceMaterialTexturesPreviewTest : public RhiTest {
public:
    RenderGraphScenePathTraceMaterialTexturesPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_scene_path_trace_material_textures_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
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
                return RhiTestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("ScenePathTracePass textured material render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("ScenePathTracePass textured material preview produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string outputMessage;
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_material_textures_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphScenePathTraceTransmissionTexturesPreviewTest : public RhiTest {
public:
    RenderGraphScenePathTraceTransmissionTexturesPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_scene_path_trace_transmission_textures_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::filesystem::path scenePath;
        std::string message;
        if (!writeTransmissionTextureScene(context.outputDirectory / "transmission-texture-scene", scenePath, message)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
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
                return RhiTestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                std::string("transmission texture preview expected visible red diffuse pixels, got red=") +
                std::to_string(redPixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_transmission_textures_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RhiTestResult::fail(message);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphScenePathTraceAlphaMaskPreviewTest : public RhiTest {
public:
    RenderGraphScenePathTraceAlphaMaskPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_scene_path_trace_alpha_mask_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::filesystem::path scenePath;
        std::string message;
        if (!writeAlphaMaskScene(context.outputDirectory / "alpha-mask-scene", scenePath, message)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
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
                return RhiTestResult::skip(
                    std::string("ScenePathTracePass is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                std::string("alpha mask preview expected red masked pixels and blue revealed pixels, got red=") +
                std::to_string(redPixelCount) +
                " blue=" +
                std::to_string(bluePixelCount));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_path_trace_alpha_mask_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), message)) {
            return RhiTestResult::fail(message);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphResizeReusesCompiledPassesTest : public RhiTest {
public:
    RenderGraphResizeReusesCompiledPassesTest()
    {
        type = RhiTestType::Resource;
        name = "render_graph_resize_reuses_compiled_passes";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerTestPass();

        render::RenderGraph graph;
        graph.setName("ResizeReuse");
        render::RenderGraphNode* node = graph.addNode("TestResizeCompilePass", "Resize");
        if (node == nullptr) {
            return RhiTestResult::fail("failed to add resize test pass node");
        }
        graph.markOutput("Resize.color");

        uint32_t& compileCount = testResizeCompileCount();
        compileCount = 0;

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result result = executor.compile(context.device, graph, 64, 48, log);
        if (!result) {
            return RhiTestResult::fail(std::string("initial RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }
        if (compileCount != 1) {
            return RhiTestResult::fail(
                std::string("expected one pass compile after initial compile, got ") +
                std::to_string(compileCount));
        }

        const render::RenderGraphResource* output = executor.outputResource("Resize.color");
        if (output == nullptr || output->desc.width != 64 || output->desc.height != 48) {
            return RhiTestResult::fail("initial resize test output dimensions are invalid");
        }

        result = executor.compile(context.device, graph, 128, 96, log);
        if (!result) {
            return RhiTestResult::fail(std::string("resize RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }
        if (compileCount != 1) {
            return RhiTestResult::fail(
                std::string("resize recompiled pass PSO path; compile count is ") +
                std::to_string(compileCount));
        }

        output = executor.outputResource("Resize.color");
        if (output == nullptr || output->desc.width != 128 || output->desc.height != 96) {
            return RhiTestResult::fail("resized graph output dimensions were not rebuilt");
        }

        render::RenderGraphProperties properties = render::RenderGraphProperties::object();
        properties["variant"] = 1;
        if (!graph.setNodeProperties(node->id, std::move(properties))) {
            return RhiTestResult::fail("failed to update resize test pass static properties");
        }

        result = executor.compile(context.device, graph, 128, 96, log);
        if (!result) {
            return RhiTestResult::fail(std::string("static property RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }
        if (compileCount != 2) {
            return RhiTestResult::fail(
                std::string("static property change did not force full pass compile; compile count is ") +
                std::to_string(compileCount));
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphCopyColorWorkflowTest : public RhiTest {
public:
    RenderGraphCopyColorWorkflowTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_copy_color_workflow";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("CopyColorWorkflow");
        graph.addNode("TriangleRasterPass", "Triangle");
        graph.addNode("CopyColorPass", "Copy");
        graph.addEdge("Triangle.color", "Copy.source");
        graph.markOutput("Copy.color");

        result = preview.render(graph, 128, 96);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result));
        }
        if (countBrightPixels(preview.pixels()) < 128) {
            return RhiTestResult::fail("copy color graph produced too few bright pixels");
        }

        graph.markDirty();
        result = preview.render(graph, 80, 80);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render resize returned ") + toString(result));
        }
        if (preview.width() != 80 || preview.height() != 80) {
            return RhiTestResult::fail("copy color graph resize did not update output dimensions");
        }
        if (countBrightPixels(preview.pixels()) < 80) {
            return RhiTestResult::fail("resized copy color graph produced too few bright pixels");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphBindlessTextureWorkflowTest : public RhiTest {
public:
    RenderGraphBindlessTextureWorkflowTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_bindless_texture_workflow";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerTestPass();

        constexpr uint32_t kWidth = 16;
        constexpr uint32_t kHeight = 16;
        constexpr uint64_t kReadbackByteSize = static_cast<uint64_t>(kWidth) * kHeight * 4ull;

        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Bindless Texture Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("bindless test device has no graphics queue");
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
                return RhiTestResult::skip(log);
            }
            return RhiTestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = device->createBuffer(
            render::BufferDesc{
                .size = kReadbackByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        result = executor.execute(*commandBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("Sample.color");
        if (output == nullptr || output->texture == nullptr) {
            return RhiTestResult::fail("bindless graph output resource is missing");
        }

        result = executor.transitionOutput(*commandBuffer, "Sample.color", render::ResourceState::TransferSource);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionOutput returned ") + toString(result));
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
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = fence.get(),
        });
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        if (device->capabilities().timestampQueries) {
            std::vector<render::RenderGraphExecutionStats> completedGpuStats;
            result = executor.collectCompletedGpuExecutionStats(completedGpuStats);
            if (!result) {
                return RhiTestResult::fail(
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
                return RhiTestResult::fail("RenderGraph pass GPU timestamps were incomplete");
            }
        }

        readbackBuffer->invalidate();
        void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("readback buffer did not map");
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
            return RhiTestResult::fail(
                std::string("bindless graph sampled too few source pixels: ") +
                std::to_string(matchedPixelCount));
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_bindless_texture_workflow.png";
        if (!saveRgba8Png(outputPath, pixels.data(), kWidth, kHeight, outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        (void)device->waitIdle();
        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphBufferWorkflowTest : public RhiTest {
public:
    RenderGraphBufferWorkflowTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_buffer_workflow";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint64_t kByteSize = 16;
        constexpr std::array<uint32_t, 4> kExpectedWords = {
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };

        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Buffer Workflow Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::Queue* computeQueue = device->getQueue(render::QueueType::Compute);
        if (computeQueue == nullptr) {
            return RhiTestResult::skip("buffer workflow device has no compute queue");
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
                return RhiTestResult::skip(log);
            }
            return RhiTestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        result = executor.execute(render::RenderGraphSubmitDesc{
            .computeQueue = computeQueue,
        });
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }

        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::waitForSubmittedWork returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("Copy.data");
        if (output == nullptr ||
            output->type != render::RenderGraphResourceType::Buffer ||
            output->buffer == nullptr ||
            output->bufferDesc.memoryLocation != render::MemoryLocation::HostReadback ||
            output->bufferDesc.size != kByteSize) {
            return RhiTestResult::fail("buffer graph output resource is invalid");
        }

        output->buffer->invalidate();
        void* mapped = output->buffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("buffer graph output did not map");
        }

        std::array<uint32_t, 4> actualWords{};
        std::memcpy(actualWords.data(), mapped, actualWords.size() * sizeof(uint32_t));
        output->buffer->unmap();

        if (actualWords != kExpectedWords) {
            return RhiTestResult::fail("buffer graph output bytes did not match expected pattern");
        }

        (void)device->waitIdle();
        return RhiTestResult::pass();
    }
};

class RenderGraphMultiQueueSubmitTest : public RhiTest {
public:
    RenderGraphMultiQueueSubmitTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_multi_queue_submit";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint64_t kByteSize = 16;
        constexpr std::array<uint32_t, 4> kExpectedWords = {
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };

        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Multi Queue Submit Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        render::Queue* computeQueue = device->getQueue(render::QueueType::Compute);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("multi queue submit device has no graphics queue");
        }
        if (computeQueue == nullptr) {
            return RhiTestResult::skip("multi queue submit device has no compute queue");
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
                return RhiTestResult::skip(log);
            }
            return RhiTestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        result = executor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = graphicsQueue,
            .computeQueue = computeQueue,
        });
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::execute(RenderGraphSubmitDesc) returned ") + toString(result));
        }

        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::waitForSubmittedWork returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("Write.data");
        if (output == nullptr ||
            output->type != render::RenderGraphResourceType::Buffer ||
            output->buffer == nullptr ||
            output->bufferDesc.memoryLocation != render::MemoryLocation::HostReadback ||
            output->bufferDesc.size != kByteSize) {
            return RhiTestResult::fail("multi queue buffer output resource is invalid");
        }

        output->buffer->invalidate();
        void* mapped = output->buffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("multi queue buffer graph output did not map");
        }

        std::array<uint32_t, 4> actualWords{};
        std::memcpy(actualWords.data(), mapped, actualWords.size() * sizeof(uint32_t));
        output->buffer->unmap();

        if (actualWords != kExpectedWords) {
            return RhiTestResult::fail("multi queue buffer graph output bytes did not match expected pattern");
        }

        (void)device->waitIdle();
        return RhiTestResult::pass();
    }
};

class RenderGraphImageSamplePassPreviewTest : public RhiTest {
public:
    RenderGraphImageSamplePassPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_image_sample_pass_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("ImageSamplePreview");
        graph.addNode("ImageSamplePass", "Image");
        graph.markOutput("Image.color");

        result = preview.render(graph, 160, 120);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(preview.lastLog());
            }
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result) + ": " + preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 160 * 120 / 2) {
            return RhiTestResult::fail(
                std::string("image sample pass produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_image_sample_pass_preview.png";
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphMaterialShaderObjectPassSmokeTest : public RhiTest {
public:
    RenderGraphMaterialShaderObjectPassSmokeTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_material_shader_object_pass_smoke";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Shader Object Smoke Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
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
                return RhiTestResult::fail(
                    std::string("expected Unsupported without shader-object capabilities, got ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RhiTestResult::pass("SceneMaterialShaderObjectPass reported Unsupported without required capabilities");
        }

        if (!result) {
            return RhiTestResult::fail(
                std::string("RenderGraphExecutor::compile returned ") +
                toString(result) +
                ": " +
                log);
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphGPUDrivenPreviewPassSmokeTest : public RhiTest {
public:
    RenderGraphGPUDrivenPreviewPassSmokeTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_preview_pass_smoke";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic GPUDrivenPreviewPass Smoke Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableMeshShader = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenPreviewSmoke");
        graph.addNode(
            "GPUDrivenPreviewPass",
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
            device->capabilities().bindlessDescriptorHeap;
        if (!hasRequiredCapabilities) {
            if (!render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::fail(
                    std::string("expected Unsupported without mesh shader capabilities, got ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RhiTestResult::pass("GPUDrivenPreviewPass reported Unsupported without required capabilities");
        }

        if (!result) {
            return RhiTestResult::fail(
                std::string("RenderGraphExecutor::compile returned ") +
                toString(result) +
                ": " +
                log);
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphGPUDrivenStreamAssetPassSmokeTest : public RhiTest {
public:
    RenderGraphGPUDrivenStreamAssetPassSmokeTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_streamasset_pass_smoke";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint32_t kWidth = 128;
        constexpr uint32_t kHeight = 96;
        constexpr uint64_t kReadbackByteSize = static_cast<uint64_t>(kWidth) * kHeight * 4u;

        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic GPUDrivenStreamAssetPass Smoke Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableMeshShader = true,
                .enableRayQuery = true,
                .enableClusterAccelerationStructure = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("GPUDrivenStreamAssetPass smoke device has no graphics queue");
        }

        const std::filesystem::path streamAssetPath =
            context.outputDirectory / "gpu_driven_streamasset_smoke.meshstream.bin";
        const std::filesystem::path sourcePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        scene::Scene runtimeScene;
        if (!runtimeScene.load(sourcePath)) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail("buildMeshletStreamAssetOffline failed: " + buildReason);
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
                return RhiTestResult::fail(
                    std::string("expected Unsupported without mesh shader capabilities, got ") +
                    toString(result) +
                    ": " +
                    log);
            }
            return RhiTestResult::pass("GPUDrivenStreamAssetPass reported Unsupported without required capabilities");
        }

        if (!result) {
            return RhiTestResult::fail(
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
                return RhiTestResult::fail(
                    std::string("RenderGraphExecutor::execute frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
            result = executor.waitForSubmittedWork(5'000'000'000ull);
            if (!result) {
                return RhiTestResult::fail(
                    std::string("RenderGraphExecutor::waitForSubmittedWork frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = device->createBuffer(
            render::BufferDesc{
                .size = kReadbackByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        result = executor.execute(*commandBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::execute(readback) returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("GPUDriven.color");
        if (output == nullptr || output->texture == nullptr) {
            return RhiTestResult::fail("GPUDrivenStreamAssetPass smoke output resource is missing");
        }

        result = executor.transitionOutput(*commandBuffer, "GPUDriven.color", render::ResourceState::TransferSource);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionOutput returned ") + toString(result));
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
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = fence.get(),
        });
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }
        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        readbackBuffer->invalidate();
        void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("readback buffer did not map");
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
            return RhiTestResult::fail("GPUDrivenStreamAssetPass smoke produced only clear pixels");
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
            return RhiTestResult::fail(
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
                return RhiTestResult::fail(
                    std::string("streamed-fallback RenderGraphExecutor::execute frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
            result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
            if (!result) {
                return RhiTestResult::fail(
                    std::string("streamed-fallback RenderGraphExecutor::waitForSubmittedWork frame ") +
                    std::to_string(frame) +
                    " returned " +
                    toString(result));
            }
        }

        std::unique_ptr<render::CommandBuffer> rasterCommandBuffer;
        result = commandPool->createCommandBuffer(rasterCommandBuffer);
        if (!result || rasterCommandBuffer == nullptr) {
            return RhiTestResult::fail(
                std::string("createCommandBuffer(raster readback) returned ") +
                toString(result));
        }
        std::unique_ptr<render::Fence> rasterFence;
        result = device->createFence(false, rasterFence);
        if (!result || rasterFence == nullptr) {
            return RhiTestResult::fail(
                std::string("createFence(raster readback) returned ") +
                toString(result));
        }
        std::unique_ptr<render::Buffer> rasterColorReadback;
        std::unique_ptr<render::Buffer> rasterVisibilityReadback;
        auto createRasterReadback = [&](std::unique_ptr<render::Buffer>& buffer) {
            return device->createBuffer(
                render::BufferDesc{
                    .size = kReadbackByteSize,
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback,
                },
                buffer);
        };
        result = createRasterReadback(rasterColorReadback);
        if (!result || rasterColorReadback == nullptr) {
            return RhiTestResult::fail(
                std::string("createBuffer(raster color readback) returned ") +
                toString(result));
        }
        result = createRasterReadback(rasterVisibilityReadback);
        if (!result || rasterVisibilityReadback == nullptr) {
            return RhiTestResult::fail(
                std::string("createBuffer(raster visibility readback) returned ") +
                toString(result));
        }

        result = rasterCommandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(
                std::string("raster readback CommandBuffer::begin returned ") +
                toString(result));
        }
        result = streamedFallbackExecutor.execute(*rasterCommandBuffer);
        if (!result) {
            return RhiTestResult::fail(
                std::string("streamed-fallback execute(readback) returned ") +
                toString(result));
        }
        render::RenderGraphResource* rasterColor =
            streamedFallbackExecutor.outputResource("GPUDriven.color");
        render::RenderGraphResource* rasterVisibility =
            streamedFallbackExecutor.outputResource("GPUDriven.visibility");
        if (rasterColor == nullptr || rasterColor->texture == nullptr ||
            rasterVisibility == nullptr || rasterVisibility->texture == nullptr) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                std::string("raster readback CommandBuffer::end returned ") +
                toString(result));
        }
        render::CommandBuffer* rasterCommandBuffers[] = {rasterCommandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = rasterCommandBuffers,
            .commandBufferCount = 1,
            .signalFence = rasterFence.get(),
        });
        if (!result) {
            return RhiTestResult::fail(
                std::string("raster readback Queue::submit returned ") +
                toString(result));
        }
        result = rasterFence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(
                std::string("raster readback Fence::wait returned ") +
                toString(result));
        }

        rasterColorReadback->invalidate();
        const auto* rasterColorPixels =
            static_cast<const uint8_t*>(rasterColorReadback->map());
        if (rasterColorPixels == nullptr) {
            return RhiTestResult::fail("raster color readback did not map");
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
            return RhiTestResult::fail("raster visibility readback did not map");
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
            return RhiTestResult::fail(
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
            result = device->createBuffer(
                render::BufferDesc{
                    .size = static_cast<uint64_t>(width) * height * sizeof(uint32_t),
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback,
                },
                captureReadback);
            if (!result || captureReadback == nullptr) {
                error = std::string("createBuffer(visibility capture) returned ") +
                    toString(result);
                return false;
            }

            std::unique_ptr<render::CommandBuffer> captureCommandBuffer;
            result = commandPool->createCommandBuffer(captureCommandBuffer);
            if (!result || captureCommandBuffer == nullptr) {
                error = std::string("createCommandBuffer(visibility capture) returned ") +
                    toString(result);
                return false;
            }
            std::unique_ptr<render::Fence> captureFence;
            result = device->createFence(false, captureFence);
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
                .commandBuffers = captureCommandBuffers,
                .commandBufferCount = 1,
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
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                std::string("resized streamed-fallback capture failed: ") +
                captureError);
        }
        if (resizedVisibilityPixelCount == 0) {
            return RhiTestResult::fail("resized stream frame produced no visibility pixels");
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
            return RhiTestResult::fail("stream visibility sync test found no visible objects");
        }
        const uint64_t transformRevisionBeforeHide = runtimeScene.transformRevision();
        const uint64_t visibilityRevisionBeforeHide = runtimeScene.visibilityRevision();
        for (scene::SceneEntity object : visibleObjects) {
            if (!runtimeScene.setObjectVisible(object, false)) {
                return RhiTestResult::fail("stream visibility sync test could not hide an object");
            }
        }
        if (runtimeScene.transformRevision() != transformRevisionBeforeHide ||
            runtimeScene.visibilityRevision() == visibilityRevisionBeforeHide) {
            return RhiTestResult::fail(
                "stream visibility sync test did not isolate visibility revision changes");
        }

        result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = graphicsQueue,
        });
        if (result) {
            result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
        }
        if (!result) {
            return RhiTestResult::fail(
                std::string("hidden stream frame returned ") + toString(result));
        }
        uint32_t hiddenVisibilityPixelCount = 0;
        if (!captureVisibilityPixelCount(
                kResizedWidth,
                kResizedHeight,
                hiddenVisibilityPixelCount,
                captureError)) {
            return RhiTestResult::fail(captureError);
        }
        if (hiddenVisibilityPixelCount != 0) {
            return RhiTestResult::fail(
                "visibility-only hide left " +
                std::to_string(hiddenVisibilityPixelCount) +
                " raster pixels");
        }

        for (scene::SceneEntity object : visibleObjects) {
            if (!runtimeScene.setObjectVisible(object, true)) {
                return RhiTestResult::fail("stream visibility sync test could not restore an object");
            }
        }
        result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = graphicsQueue,
        });
        if (result) {
            result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
        }
        if (!result) {
            return RhiTestResult::fail(
                std::string("restored stream frame returned ") + toString(result));
        }
        uint32_t restoredVisibilityPixelCount = 0;
        if (!captureVisibilityPixelCount(
                kResizedWidth,
                kResizedHeight,
                restoredVisibilityPixelCount,
                captureError)) {
            return RhiTestResult::fail(captureError);
        }
        if (restoredVisibilityPixelCount == 0) {
            return RhiTestResult::fail("visibility-only show did not restore raster pixels");
        }

        for (uint32_t frame = 0; frame < 3; ++frame) {
            result = streamedFallbackExecutor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
            });
            if (result) {
                result = streamedFallbackExecutor.waitForSubmittedWork(5'000'000'000ull);
            }
            if (!result) {
                return RhiTestResult::fail(
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
                return RhiTestResult::fail(
                    "post-show stabilization capture failed: " +
                    captureError);
            }
            if (stableVisibilityPixelCount == 0) {
                return RhiTestResult::fail(
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
            return RhiTestResult::fail(
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
                return RhiTestResult::fail(
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
                return RhiTestResult::fail(
                    "post-show resize capture failed: " +
                    captureError);
            }
            if (postShowResizedPixels == 0) {
                return RhiTestResult::fail(
                    "post-show resized frame " +
                    std::to_string(frame) +
                    " produced no visibility pixels");
            }
        }

        (void)device->waitIdle();
        return RhiTestResult::pass();
    }
};

class RenderGraphGPUDrivenMixedProducerRenderTest : public RhiTest {
public:
    RenderGraphGPUDrivenMixedProducerRenderTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_mixed_producer_render";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint32_t kWidth = 256;
        constexpr uint32_t kHeight = 192;
        constexpr uint32_t kMaxActiveGroups = 64;
        constexpr uint32_t kWarmupFrameCount = 16;
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
            return RhiTestResult::fail(
                "mixed-producer streamasset build failed: " + reason);
        }

        scene::MeshletStreamAsset streamAsset;
        if (!streamAsset.open(streamAssetPath, reason) ||
            !streamAsset.isCurrentForSource(sourcePath)) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                "mixed-producer scene composition failed: " + reason);
        }
        if (runtimeScene.renderNodes().size() != 2 ||
            streamAsset.instances().size() != 1 ||
            streamAsset.instances()[0].renderNodeIndex != 0 ||
            runtimeScene.renderNodeIndexForSource("resident", 0) != 0 ||
            runtimeScene.renderNodeIndexForSource("stream", 0) != 1) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                "mixed-producer fixture has an invalid stream record capacity");
        }

        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic GPUDriven mixed-producer render test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableMeshShader = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(
                    std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(
                std::string("createDevice returned ") + toString(result));
        }
        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail(
                "mixed-producer device has no graphics queue");
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenMixedProducer");
        graph.addNode(
            "GPUDrivenPreviewPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", sourcePath.string()},
                {"streamAssetPath", streamAssetPath.string()},
                {"streamSourceId", "stream"},
                {"enableMeshletStreaming", true},
                {"maxLockedFallbackPages", 4},
                {"maxResidentPages", 16},
                {"maxPageUploadsPerFrame", 4},
                {"maxActiveGroups", kMaxActiveGroups},
                {"mode", "meshlet"},
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
        graph.markOutput("GPUDriven.color");
        graph.markOutput("GPUDriven.visibility");
        graph.markOutput("GPUDriven.depth");

        render::RenderGraphExecutor executor;
        executor.bindRuntimeScene(&runtimeScene);
        std::string log;
        result = executor.compile(*device, graph, kWidth, kHeight, log);
        const bool hasRequiredCapabilities =
            device->capabilities().meshShader &&
            device->capabilities().bindlessDescriptorHeap;
        if (!hasRequiredCapabilities) {
            if (!render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::fail(
                    std::string("expected Unsupported without mixed raster capabilities, got ") +
                    toString(result) + ": " + log);
            }
            return RhiTestResult::skip(
                "GPUDrivenPreviewPass mixed producer mode requires mesh shaders and bindless descriptors");
        }
        if (!result) {
            return RhiTestResult::fail(
                std::string("mixed-producer graph compile returned ") +
                toString(result) + ": " + log);
        }

        for (uint32_t frame = 0; frame < kWarmupFrameCount; ++frame) {
            result = executor.execute(render::RenderGraphSubmitDesc{
                .graphicsQueue = graphicsQueue,
            });
            if (result) {
                result = executor.waitForSubmittedWork(5'000'000'000ull);
            }
            if (!result) {
                return RhiTestResult::fail(
                    "mixed-producer warmup frame " +
                    std::to_string(frame) + " returned " + toString(result));
            }
        }

        render::RenderSubsystemHost* subsystemHost = executor.subsystemHost();
        render::GPUSceneSubsystem* gpuScene = subsystemHost != nullptr
            ? subsystemHost->get<render::GPUSceneSubsystem>()
            : nullptr;
        if (gpuScene == nullptr) {
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
                "mixed-producer GPUScene resident record namespace is invalid");
        }
        const uint32_t streamRecordBase = static_cast<uint32_t>(
            globalViews.meshletDraws.size /
            sizeof(render::VisibleClusterRecord));
        if (!render::visibilityRecordCapacityFitsId(
                static_cast<uint64_t>(streamRecordBase) +
                requestedStreamRecordCapacity)) {
            return RhiTestResult::fail(
                "mixed-producer combined record namespace exceeds visibility IDs");
        }

        const render::GPUSceneInstanceId streamInstance =
            gpuScene->instanceForRenderNode(1);
        const render::GPUSceneInstanceId residentInstance =
            gpuScene->instanceForRenderNode(0);
        if (!streamInstance.valid() || !residentInstance.valid() ||
            streamInstance == residentInstance) {
            return RhiTestResult::fail(
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
            color->desc.format != render::Format::Rgba8Unorm ||
            visibility->desc.format != render::Format::R32Uint ||
            depth->desc.format != render::Format::D32Sfloat) {
            return RhiTestResult::fail(
                "mixed-producer shared color/visibility/depth surfaces are missing");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(
                std::string("createCommandPool(mixed producer) returned ") +
                toString(result));
        }
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(
                std::string("createCommandBuffer(mixed producer) returned ") +
                toString(result));
        }
        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(
                std::string("createFence(mixed producer) returned ") +
                toString(result));
        }

        auto makeReadback = [&](uint64_t size,
                                std::unique_ptr<render::Buffer>& buffer) {
            return device->createBuffer(
                render::BufferDesc{
                    .size = size,
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback,
                    .queueAccess = render::QueueAccessBits::Graphics,
                },
                buffer);
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
            return RhiTestResult::fail(
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
            return RhiTestResult::fail(
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
            .before = render::ResourceState::ShaderRead,
            .after = render::ResourceState::TransferSource,
            .offset = globalViews.meshletDraws.offset,
            .size = globalViews.meshletDraws.size,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = &residentRecordsToCopy,
            .bufferCount = 1,
        });
        commandBuffer->copyBuffer(render::BufferCopyDesc{
            .source = globalViews.meshletDraws.buffer,
            .destination = residentRecordReadback.get(),
            .sourceOffset = globalViews.meshletDraws.offset,
            .destinationOffset = 0,
            .size = globalViews.meshletDraws.size,
        });
        const render::BufferBarrierDesc residentRecordsToRead{
            .buffer = globalViews.meshletDraws.buffer,
            .before = render::ResourceState::TransferSource,
            .after = render::ResourceState::ShaderRead,
            .offset = globalViews.meshletDraws.offset,
            .size = globalViews.meshletDraws.size,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = &residentRecordsToRead,
            .bufferCount = 1,
        });
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(
                std::string("end mixed-producer capture returned ") +
                toString(result));
        }
        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = fence.get(),
        });
        if (result) {
            result = fence->wait(5'000'000'000ull);
        }
        if (!result) {
            return RhiTestResult::fail(
                std::string("submit mixed-producer capture returned ") +
                toString(result));
        }

        auto copyReadback = [](
                                render::Buffer& buffer,
                                void* destination,
                                uint64_t size) -> bool {
            buffer.invalidate(0, size);
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
            return RhiTestResult::fail(
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
                return RhiTestResult::fail(
                    "mixed-producer visibility was not resolved by the shared depth/deferred surfaces");
            }

            if (recordIndex < streamRecordBase) {
                const render::VisibleClusterRecord& record =
                    residentRecords[recordIndex];
                if (render::visibleClusterSource(record.flags) !=
                        render::VisibleClusterSource::Resident ||
                    record.instanceIndex != residentInstance.index ||
                    record.instanceIndex == streamInstance.index) {
                    return RhiTestResult::fail(
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
                return RhiTestResult::fail(
                    "visibility referenced a stream record outside its logical namespace");
            }
            ++streamPixelCount;
            streamMinX = std::min(streamMinX, x);
            streamMaxX = std::max(streamMaxX, x);
            streamRecordIds.insert(recordIndex);
        }

        if (residentPixelCount < 32 || streamPixelCount < 32 ||
            residentRecordIds.empty() || streamRecordIds.empty()) {
            return RhiTestResult::fail(
                "mixed-producer frame did not contain both resident and stream visibility IDs: resident=" +
                std::to_string(residentPixelCount) +
                " stream=" + std::to_string(streamPixelCount));
        }
        if (streamMinX > streamMaxX || residentMinX > residentMaxX ||
            streamMaxX >= residentMinX) {
            return RhiTestResult::fail(
                "mixed-producer mounted fixtures overlap or were assigned to the wrong producer");
        }

        uint32_t unifiedPassNodeCount = 0;
        for (const render::RenderGraphNodeExecutionStat& node :
             executor.executionStats().nodes) {
            unifiedPassNodeCount +=
                node.type == "GPUDrivenPreviewPass" ? 1u : 0u;
        }
        if (unifiedPassNodeCount != 1) {
            return RhiTestResult::fail(
                "mixed producers were not resolved by one unified raster/deferred graph node");
        }

        (void)device->waitIdle();
        return RhiTestResult::pass(
            "shared visibility/depth/color resolved residentPixels=" +
            std::to_string(residentPixelCount) +
            " streamPixels=" + std::to_string(streamPixelCount) +
            " residentRecords=" + std::to_string(residentRecordIds.size()) +
            " streamRecords=" + std::to_string(streamRecordIds.size()));
    }
};

class RenderGraphGPUDrivenPreviewPassRenderTest : public RhiTest {
public:
    RenderGraphGPUDrivenPreviewPassRenderTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_preview_pass_render";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(render::EnvironmentSettings{
            .enabled = true,
            .path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/environment.hdr",
            .intensity = 3.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        });
        render::Result result = preview.initialize(context.enableValidation, false);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip("RenderGraphPreviewRenderer is unsupported");
            }
            return RhiTestResult::fail(
                std::string("RenderGraphPreviewRenderer::initialize returned ") +
                toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenPreviewRender");
        graph.addNode(
            "GPUDrivenPreviewPass",
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
                return RhiTestResult::skip(
                    std::string("GPUDrivenPreviewPass is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }
        const std::vector<uint32_t> firstFramePixels = preview.pixels();

        result = preview.render(graph, 192, 192);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass second-frame HZB render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t hzbVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (hzbVisiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass second-frame HZB render produced too few visible pixels: ") +
                std::to_string(hzbVisiblePixelCount));
        }
        if (preview.pixels() != firstFramePixels) {
            size_t mismatchCount = 0;
            for (size_t pixelIndex = 0; pixelIndex < firstFramePixels.size(); ++pixelIndex) {
                mismatchCount += preview.pixels()[pixelIndex] != firstFramePixels[pixelIndex] ? 1u : 0u;
            }
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass stationary HZB frame changed ") +
                std::to_string(mismatchCount) +
                " pixels");
        }

        render::RenderGraphNode* gpuDrivenNode = graph.findNode("GPUDriven");
        if (gpuDrivenNode == nullptr ||
            !graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.fovDegrees", 20.0f)) {
            return RhiTestResult::fail("failed to configure the GPUDriven culling test camera");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass narrow culling-camera render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const std::vector<uint32_t> capturedCullingPixels = preview.pixels();

        if (
            !graph.setNodeRuntimeProperty(gpuDrivenNode->id, "freezeCullingCamera", true)) {
            return RhiTestResult::fail("failed to freeze the GPUDriven culling camera");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass frozen-camera capture render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        if (preview.pixels() != capturedCullingPixels) {
            return RhiTestResult::fail(
                "freezing the GPUDriven culling camera changed the captured view");
        }

        const render::RenderGraphProperties oppositeEye =
            render::RenderGraphProperties::array({0.22f, 0.110154f, -0.00153695f});
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.eye", oppositeEye)) {
            return RhiTestResult::fail("failed to move the GPUDriven observation camera");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass frozen-culling observation render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t frozenVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (frozenVisiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass frozen culling produced too few visible pixels: ") +
                std::to_string(frozenVisiblePixelCount));
        }
        const std::vector<uint32_t> frozenObservationPixels = preview.pixels();

        result = preview.render(graph, 192, 192);
        if (!result || preview.pixels() != frozenObservationPixels) {
            return RhiTestResult::fail(
                "GPUDrivenPreviewPass frozen culling camera was not stable while observing from another view");
        }

        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "freezeCullingCamera", false)) {
            return RhiTestResult::fail("failed to restore live GPUDriven camera culling");
        }
        result = preview.render(graph, 192, 192);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass restored live-camera render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t liveVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (liveVisiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass restored live culling produced too few visible pixels: ") +
                std::to_string(liveVisiblePixelCount));
        }
        size_t cullingCameraMismatchCount = 0;
        for (size_t pixelIndex = 0; pixelIndex < frozenObservationPixels.size(); ++pixelIndex) {
            cullingCameraMismatchCount +=
                preview.pixels()[pixelIndex] != frozenObservationPixels[pixelIndex] ? 1u : 0u;
        }
        if (cullingCameraMismatchCount < 64) {
            return RhiTestResult::fail(
                "disabling the frozen culling camera did not restore view-dependent culling");
        }

        const render::RenderGraphProperties originalEye =
            render::RenderGraphProperties::array({-0.0168404f, 0.110154f, 0.22f});
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.eye", originalEye)) {
            return RhiTestResult::fail("failed to restore the GPUDriven observation camera");
        }
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "camera.fovDegrees", 60.0f)) {
            return RhiTestResult::fail("failed to restore the GPUDriven camera FOV");
        }

        result = preview.render(graph, 128, 96);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass resize-down render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t resizedDownVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (resizedDownVisiblePixelCount < 128) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass resize-down render produced too few visible pixels: ") +
                std::to_string(resizedDownVisiblePixelCount));
        }

        result = preview.render(graph, 256, 144);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass resize-up render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        const uint32_t resizedUpVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (resizedUpVisiblePixelCount < 256) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass resize-up render produced too few visible pixels: ") +
                std::to_string(resizedUpVisiblePixelCount));
        }

        render::RenderGraph lodGraph;
        lodGraph.setName("GPUDrivenPreviewLodRender");
        lodGraph.addNode(
            "GPUDrivenPreviewPass",
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
                return RhiTestResult::skip(
                    std::string("GPUDrivenPreviewPass LOD mode is unsupported on this device: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass LOD render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        const uint32_t lodVisiblePixelCount = countVisiblePixels(preview.pixels());
        if (lodVisiblePixelCount < 512) {
            return RhiTestResult::fail(
                std::string("GPUDrivenPreviewPass LOD mode produced too few visible pixels: ") +
                std::to_string(lodVisiblePixelCount));
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphGPUDrivenAlphaMaskRenderTest : public RhiTest {
public:
    RenderGraphGPUDrivenAlphaMaskRenderTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_alpha_mask_render";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::filesystem::path scenePath;
        std::string message;
        if (!writeAlphaMaskScene(
                context.outputDirectory / "gpu-driven-alpha-mask-scene",
                scenePath,
                message)) {
            return RhiTestResult::fail(message);
        }
        std::filesystem::path singleSidedScenePath;
        if (!writeAlphaMaskScene(
                context.outputDirectory / "gpu-driven-alpha-mask-single-sided-scene",
                singleSidedScenePath,
                message,
                false)) {
            return RhiTestResult::fail(message);
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(render::EnvironmentSettings{
            .enabled = false,
            .intensity = 0.0f,
            .visible = false,
        });
        render::Result result = preview.initialize(context.enableValidation, false);
        if (!result) {
            return render::hasError(result, render::Error::Unsupported)
                ? RhiTestResult::skip("GPUDriven alpha-mask preview is unsupported")
                : RhiTestResult::fail(
                      std::string("RenderGraphPreviewRenderer::initialize returned ") +
                      toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenAlphaMaskRender");
        graph.addNode(
            "GPUDrivenPreviewPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", scenePath.string()},
                {"mode", "shaded"},
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
                if (r > 48 && r > g + 32 && r > b + 32) {
                    ++counts[0];
                }
                if (b > 48 && b > r + 32 && b > g + 32) {
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
                return RhiTestResult::skip(
                    std::string("GPUDrivenPreviewPass is unsupported: ") +
                    preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("GPUDriven alpha-mask front render returned ") +
                toString(result) + ": " + preview.lastLog());
        }
        const std::array<uint32_t, 3> front = classifyPixels();
        if (front[0] < 1024 || front[2] < 1024 || front[1] > 64) {
            return RhiTestResult::fail(
                "GPUDriven MASK/BLEND classification is incorrect on the front face: red=" +
                std::to_string(front[0]) + " blue=" + std::to_string(front[1]) +
                " dark=" + std::to_string(front[2]));
        }

        render::RenderGraphNode* node = graph.findNode("GPUDriven");
        if (node == nullptr ||
            !graph.setNodeRuntimeProperty(
                node->id,
                "camera.eye",
                render::RenderGraphProperties::array({0.0f, 0.0f, -2.0f}))) {
            return RhiTestResult::fail("failed to move the GPUDriven camera behind the double-sided MASK quad");
        }
        result = preview.render(graph, 128, 128);
        if (!result) {
            return RhiTestResult::fail(
                std::string("GPUDriven alpha-mask back render returned ") +
                toString(result) + ": " + preview.lastLog());
        }
        const std::array<uint32_t, 3> back = classifyPixels();
        if (back[0] < 1024 || back[2] < 1024 || back[1] > 64) {
            return RhiTestResult::fail(
                "GPUDriven double-sided MASK did not survive back-face rendering: red=" +
                std::to_string(back[0]) + " blue=" + std::to_string(back[1]) +
                " dark=" + std::to_string(back[2]));
        }

        render::RenderGraph singleSidedGraph;
        singleSidedGraph.setName("GPUDrivenSingleSidedMaskRender");
        singleSidedGraph.addNode(
            "GPUDrivenPreviewPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", singleSidedScenePath.string()},
                {"mode", "shaded"},
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
            return RhiTestResult::fail(
                std::string("GPUDriven single-sided MASK back render returned ") +
                toString(result) + ": " + preview.lastLog());
        }
        const std::array<uint32_t, 3> singleSidedBack = classifyPixels();
        if (singleSidedBack[0] > 64 || singleSidedBack[1] > 64 ||
            singleSidedBack[2] < 4096) {
            return RhiTestResult::fail(
                "GPUDriven single-sided MASK was not back-face culled: red=" +
                std::to_string(singleSidedBack[0]) + " blue=" +
                std::to_string(singleSidedBack[1]) + " dark=" +
                std::to_string(singleSidedBack[2]));
        }
        return RhiTestResult::pass();
    }
};

class RenderGraphGPUDrivenSponzaVisibilityRenderTest : public RhiTest {
public:
    RenderGraphGPUDrivenSponzaVisibilityRenderTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_driven_sponza_visibility_render";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
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
        render::Result result = preview.initialize(context.enableValidation, false);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip("RenderGraphPreviewRenderer is unsupported");
            }
            return RhiTestResult::fail(
                std::string("RenderGraphPreviewRenderer::initialize returned ") +
                toString(result));
        }

        render::RenderGraph graph;
        graph.setName("GPUDrivenSponzaVisibilityRender");
        graph.addNode(
            "GPUDrivenPreviewPass",
            "GPUDriven",
            render::RenderGraphProperties{
                {"path", "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"},
                {"mode", "shaded"},
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
                return RhiTestResult::skip(
                    std::string("SuperSponza visibility rendering is unsupported: ") + preview.lastLog());
            }
            return RhiTestResult::fail(
                std::string("SuperSponza visibility render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }

        render::EnvironmentLightingSubsystem* environmentSubsystem =
            preview.subsystemHost()->get<render::EnvironmentLightingSubsystem>();
        if (environmentSubsystem == nullptr) {
            return RhiTestResult::fail("SuperSponza environment subsystem was not activated");
        }
        bool environmentReady = false;
        for (uint32_t attempt = 0; attempt < 5000 && !environmentReady; ++attempt) {
            const render::EnvironmentLightingSnapshot& snapshot = environmentSubsystem->snapshot();
            environmentReady = snapshot.status == render::EnvironmentLightingStatus::Ready &&
                snapshot.mapAvailable;
            if (!environmentReady) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                result = preview.render(graph, 256, 256);
                if (!result) {
                    return RhiTestResult::fail(
                        "SuperSponza failed while waiting for the environment snapshot");
                }
            }
        }
        if (!environmentReady) {
            return RhiTestResult::fail("SuperSponza environment snapshot did not become ready");
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 2048) {
            return RhiTestResult::fail(
                std::string("SuperSponza visibility render produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }
        const std::vector<uint32_t> firstFramePixels = preview.pixels();

        result = preview.render(graph, 256, 256);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SuperSponza stationary HZB render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        if (preview.pixels() != firstFramePixels) {
            size_t mismatchCount = 0;
            for (size_t pixelIndex = 0; pixelIndex < firstFramePixels.size(); ++pixelIndex) {
                mismatchCount += preview.pixels()[pixelIndex] != firstFramePixels[pixelIndex] ? 1u : 0u;
            }
            return RhiTestResult::fail(
                std::string("SuperSponza stationary meshlet visualization changed ") +
                std::to_string(mismatchCount) +
                " pixels");
        }
        render::RenderGraphNode* gpuDrivenNode = graph.findNode("GPUDriven");
        if (gpuDrivenNode == nullptr) {
            return RhiTestResult::fail("failed to find the SuperSponza GPUDriven node");
        }
        environment.rotationDegrees = 90.0f;
        preview.setEnvironment(environment);
        result = preview.render(graph, 256, 256);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SuperSponza rotated-environment render returned ") +
                toString(result) +
                ": " +
                preview.lastLog());
        }
        size_t environmentMismatchCount = 0;
        for (size_t pixelIndex = 0; pixelIndex < firstFramePixels.size(); ++pixelIndex) {
            environmentMismatchCount +=
                preview.pixels()[pixelIndex] != firstFramePixels[pixelIndex] ? 1u : 0u;
        }
        if (environmentMismatchCount < 4096) {
            return RhiTestResult::fail(
                "rotating the HDR environment did not materially change the OpenPBR resolve");
        }
        environment.rotationDegrees = 0.0f;
        preview.setEnvironment(environment);
        result = preview.render(graph, 256, 256);
        if (!result || preview.pixels() != firstFramePixels) {
            return RhiTestResult::fail(
                "restoring the HDR environment rotation did not restore the deterministic OpenPBR image");
        }
        if (
            !graph.setNodeRuntimeProperty(gpuDrivenNode->id, "freezeCullingCamera", true)) {
            return RhiTestResult::fail("failed to freeze the SuperSponza culling camera");
        }
        result = preview.render(graph, 256, 256);
        if (!result || preview.pixels() != firstFramePixels) {
            return RhiTestResult::fail(
                "SuperSponza changed when switching to the captured culling camera");
        }
        result = preview.render(graph, 256, 256);
        if (!result || preview.pixels() != firstFramePixels) {
            return RhiTestResult::fail(
                "SuperSponza frozen-camera HZB result was not stable");
        }
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "freezeCullingCamera", false)) {
            return RhiTestResult::fail("failed to restore live SuperSponza camera culling");
        }
        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "mode", "meshlet")) {
            return RhiTestResult::fail("failed to select the SuperSponza meshlet debug resolve");
        }
        result = preview.render(graph, 256, 256);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SuperSponza meshlet debug comparison returned ") +
                toString(result));
        }
        size_t debugMismatchCount = 0;
        for (size_t pixelIndex = 0; pixelIndex < firstFramePixels.size(); ++pixelIndex) {
            debugMismatchCount +=
                preview.pixels()[pixelIndex] != firstFramePixels[pixelIndex] ? 1u : 0u;
        }
        if (debugMismatchCount < 4096) {
            return RhiTestResult::fail(
                "SuperSponza OpenPBR shading was not distinguishable from meshlet debug colors");
        }

        const auto* debugBytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path debugOutputPath =
            context.outputDirectory / "render_graph_gpu_driven_sponza_meshlets.png";
        std::string outputMessage;
        if (!saveRgba8Png(
                debugOutputPath,
                debugBytes,
                256,
                256,
                outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        if (!graph.setNodeRuntimeProperty(gpuDrivenNode->id, "mode", "baseColor")) {
            return RhiTestResult::fail("failed to select the SuperSponza base-color resolve");
        }
        result = preview.render(graph, 256, 256);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SuperSponza base-color resolve returned ") + toString(result));
        }
        const auto* baseColorBytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path baseColorOutputPath =
            context.outputDirectory / "render_graph_gpu_driven_sponza_base_color.png";
        if (!saveRgba8Png(
                baseColorOutputPath,
                baseColorBytes,
                256,
                256,
                outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(firstFramePixels.data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_gpu_driven_sponza_openpbr.png";
        if (!saveRgba8Png(
                outputPath,
                bytes,
                256,
                256,
                outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }
        return RhiTestResult::pass(
            std::string("SuperSponza OpenPBR pixels=") + std::to_string(visiblePixelCount) +
            ", environment mismatches=" + std::to_string(environmentMismatchCount) +
            ", wrote " + outputPath.string());
    }
};

class ImportancePdfSizeTest : public RhiTest {
public:
    ImportancePdfSizeTest()
    {
        type = RhiTestType::Resource;
        name = "importance_pdf_size";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        const render::ImportancePdfSize lightPdfSize = render::computeImportancePdfTextureSize(257);
        if (lightPdfSize.width != 32 || lightPdfSize.height != 16 || lightPdfSize.mipCount != 6) {
            return RhiTestResult::fail("RTXDI local-light PDF sizing does not match a power-of-two rectangle");
        }
        return RhiTestResult::pass("validated GPU PDF texture sizing");
    }
};

class ReGIRGridLayoutTest : public RhiTest {
public:
    ReGIRGridLayoutTest()
    {
        type = RhiTestType::Resource;
        name = "regir_grid_layout";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        const render::ReGIRGridLayout layout = render::computeReGIRGridLayout(12, 64);
        if (!layout.valid() ||
            layout.cellCount != 1728 ||
            layout.lightSlotCount != 110592 ||
            layout.bufferByteSize !=
                static_cast<uint64_t>(110592 + render::kReGIRHeaderRecordCount) *
                    render::kReGIRRecordByteSize) {
            return RhiTestResult::fail("ReGIR grid layout or buffer sizing is incorrect");
        }

        if (render::computeReGIRGridLayout(0, 64).valid() ||
            render::computeReGIRGridLayout(12, 0).valid() ||
            render::computeReGIRGridLayout(UINT32_MAX, UINT32_MAX).valid()) {
            return RhiTestResult::fail("invalid ReGIR grid parameters were accepted");
        }
        return RhiTestResult::pass("validated ReGIR cell, slot, and buffer layout");
    }
};

class EnvironmentSubsystemAsyncSnapshotTest : public RhiTest {
public:
    EnvironmentSubsystemAsyncSnapshotTest()
    {
        type = RhiTestType::Rendering;
        name = "environment_subsystem_async_snapshot";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        registerTestPass();
        render::RenderGraph graph;
        graph.addNode("TestEnvironmentConsumerPass", "EnvironmentConsumerA");
        graph.addNode("TestEnvironmentConsumerPass", "EnvironmentConsumerB");
        if (!graph.markOutput("EnvironmentConsumerA.color") ||
            !graph.markOutput("EnvironmentConsumerB.color")) {
            return RhiTestResult::fail("failed to construct the shared environment graph");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.setEnvironment(render::EnvironmentSettings{});
        render::Result result = preview.initialize(false, false);
        if (!result) {
            return RhiTestResult::skip(
                std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }
        result = preview.render(graph, 16, 16, "EnvironmentConsumerA.color");
        if (!result) {
            return RhiTestResult::fail("initial environment render failed: " + preview.lastLog());
        }

        render::EnvironmentLightingSubsystem* subsystem =
            preview.subsystemHost()->get<render::EnvironmentLightingSubsystem>();
        if (subsystem == nullptr ||
            !subsystem->snapshot().valid() ||
            subsystem->snapshot().pdfView == nullptr) {
            return RhiTestResult::fail("environment subsystem did not publish its black fallback");
        }
        const render::TextureView* fallbackView = subsystem->snapshot().radianceView;
        const uint64_t fallbackRevision = subsystem->snapshot().resourceRevision;

        result = preview.render(graph, 24, 16, "EnvironmentConsumerA.color");
        if (!result ||
            subsystem->snapshot().radianceView != fallbackView ||
            subsystem->snapshot().resourceRevision != fallbackRevision) {
            return RhiTestResult::fail("RenderGraph resize recreated the active environment resource");
        }

        render::EnvironmentSettings environment;
        environment.path = std::filesystem::path(PROJECT_SOURCE_DIR) /
            "Asset/ABeautifulGame/environment.hdr";
        preview.setEnvironment(environment);
        result = preview.render(graph, 24, 16, "EnvironmentConsumerA.color");
        if (!result ||
            subsystem->snapshot().radianceView != fallbackView ||
            subsystem->snapshot().resourceRevision != fallbackRevision) {
            return RhiTestResult::fail("environment resource changed before asynchronous decode completed");
        }

        bool switched = false;
        for (uint32_t attempt = 0; attempt < 5000 && !switched; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            result = preview.render(graph, 24, 16, "EnvironmentConsumerA.color");
            if (!result) {
                return RhiTestResult::fail("environment switch render failed: " + preview.lastLog());
            }
            const render::EnvironmentLightingSnapshot& snapshot = subsystem->snapshot();
            switched = snapshot.status == render::EnvironmentLightingStatus::Ready &&
                snapshot.mapAvailable &&
                snapshot.pdfView != nullptr &&
                snapshot.resourceRevision > fallbackRevision;
        }
        if (!switched || subsystem->decodeCount() != 1u) {
            return RhiTestResult::fail(
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
                return RhiTestResult::fail("degraded environment render failed: " + preview.lastLog());
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
            return RhiTestResult::fail("failed environment switch did not preserve the last ready snapshot");
        }
        return RhiTestResult::pass();
    }
};

class RenderGraphMissingSubsystemDiagnosticTest : public RhiTest {
public:
    RenderGraphMissingSubsystemDiagnosticTest()
    {
        type = RhiTestType::Resource;
        name = "render_graph_missing_subsystem_diagnostic";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerTestPass();
        render::RenderGraph graph;
        graph.addNode("TestMissingSubsystemPass", "MissingSubsystemUser");
        if (!graph.markOutput("MissingSubsystemUser.color")) {
            return RhiTestResult::fail("failed to construct missing-subsystem graph");
        }

        render::RenderGraphExecutor executor;
        std::string log;
        const render::Result result = executor.compile(context.device, graph, 16, 16, log);
        if (result ||
            log.find("MissingSubsystemUser") == std::string::npos ||
            log.find("test.missing-required-subsystem") == std::string::npos) {
            return RhiTestResult::fail("missing subsystem diagnostic did not name the pass and subsystem: " + log);
        }
        return RhiTestResult::pass();
    }
};

class RenderSubsystemHostLifecycleTest : public RhiTest {
public:
    RenderSubsystemHostLifecycleTest()
    {
        type = RhiTestType::Resource;
        name = "render_subsystem_host_lifecycle";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        struct Probe final : render::IRenderSubsystem {
            Probe(std::string name, std::vector<std::string>& events, bool failBegin = false)
                : name(std::move(name)), events(events), failBegin(failBegin)
            {
            }

            render::Result initialize(const render::RenderSubsystemInitContext&, std::string&) override
            {
                events.push_back("init:" + name);
                return {};
            }

            render::Result beginFrame(
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
            return RhiTestResult::fail(log);
        }
        if (registerProbe(host, "test.base", {}, events, log)) {
            return RhiTestResult::fail("duplicate subsystem id was accepted");
        }
        log.clear();
        render::Result result = host.initialize(context.device, 3, log);
        if (!result) {
            return RhiTestResult::fail(log);
        }
        result = host.activate("test.consumer", log);
        if (!result || events != std::vector<std::string>{"init:test.base", "init:test.consumer"}) {
            return RhiTestResult::fail("dependency initialization order is incorrect: " + log);
        }
        result = host.activate("test.consumer", log);
        if (!result || events.size() != 2u) {
            return RhiTestResult::fail("active subsystem was not kept warm");
        }

        if (!host.registerSubsystem<ConfigurableRenderSubsystemProbe>(log) ||
            !host.configure<ConfigurableRenderSubsystemProbe>({.value = 42}, log) ||
            !host.activate(ConfigurableRenderSubsystemProbe::kSubsystemId, log)) {
            return RhiTestResult::fail("subsystem configuration failed: " + log);
        }
        const ConfigurableRenderSubsystemProbe* configured =
            host.get<ConfigurableRenderSubsystemProbe>();
        if (configured == nullptr || configured->observedValue != 42 ||
            host.configure<ConfigurableRenderSubsystemProbe>({.value = 7}, log)) {
            return RhiTestResult::fail("subsystem configuration was not applied before activation");
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
            return RhiTestResult::fail("world change bits were not aggregated");
        }
        host.endFrame();
        result = host.beginFrame(8, 2, nullptr, log);
        if (!result || host.lastChanges() != render::RenderChangeBits::None) {
            return RhiTestResult::fail("world change bits were not consumed exactly once");
        }
        host.endFrame();
        host.shutdown();
        const std::vector<std::string> expectedTail{
            "shutdown:test.consumer",
            "shutdown:test.base",
        };
        if (events.size() < expectedTail.size() ||
            !std::equal(expectedTail.begin(), expectedTail.end(), events.end() - expectedTail.size())) {
            return RhiTestResult::fail("subsystems did not shut down in reverse dependency order");
        }

        render::RenderSubsystemHost missingHost;
        if (!registerProbe(missingHost, "test.missing-user", {"test.not-registered"}, events, log)) {
            return RhiTestResult::fail(log);
        }
        result = missingHost.initialize(context.device, 1, log);
        if (!result) {
            return RhiTestResult::fail(log);
        }
        result = missingHost.activate("test.missing-user", log);
        if (result || log.find("test.not-registered") == std::string::npos) {
            return RhiTestResult::fail("missing dependency did not produce a named error");
        }

        render::RenderSubsystemHost cycleHost;
        log.clear();
        registerProbe(cycleHost, "test.cycle-a", {"test.cycle-b"}, events, log);
        registerProbe(cycleHost, "test.cycle-b", {"test.cycle-a"}, events, log);
        result = cycleHost.initialize(context.device, 1, log);
        if (!result) {
            return RhiTestResult::fail(log);
        }
        result = cycleHost.activate("test.cycle-a", log);
        if (result || log.find("cycle") == std::string::npos) {
            return RhiTestResult::fail("dependency cycle was not detected");
        }

        render::RenderSubsystemHost failingHost;
        log.clear();
        registerProbe(failingHost, "test.failing", {}, events, log, true);
        result = failingHost.initialize(context.device, 1, log);
        if (!result || !failingHost.activate("test.failing", log)) {
            return RhiTestResult::fail(log);
        }
        result = failingHost.beginFrame(0, 0, nullptr, log);
        if (result || log.find("test.failing") == std::string::npos) {
            return RhiTestResult::fail("hook error did not propagate with subsystem id");
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphSerializationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphReflectionApiTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPassKindTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphRuntimeSettingsDeclarationTest);
METALLIC_REGISTER_RHI_TEST(RenderSampleLoadTest);
METALLIC_REGISTER_RHI_TEST(RenderSampleFallbackAndValidationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphValidationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBunnyWireframePreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBunnyCameraSyncTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneRayQueryVisualizationPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneMaterialVisualizationPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTracePreviewTest);
METALLIC_REGISTER_RHI_TEST(SlangShaderDiskCacheTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingShaderCompileTest);
#if defined(METALLIC_HAS_RTXCR) && METALLIC_HAS_RTXCR
METALLIC_REGISTER_RHI_TEST(RenderGraphRtxcrMaterialShaderCompileTest);
#if defined(METALLIC_HAS_RTXCR_GEOMETRY) && METALLIC_HAS_RTXCR_GEOMETRY && \
    defined(METALLIC_HAS_RTXCR_ASSETS) && METALLIC_HAS_RTXCR_ASSETS
METALLIC_REGISTER_RHI_TEST(RenderGraphRtxcrMaterialPreviewTest);
#endif
#endif
METALLIC_REGISTER_RHI_TEST(GPUDrivenPreviewGeometryDedupPlanTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenPreviewShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenStreamAssetShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenStreamAssetTraversalDemandTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphRtxdiPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphRtxdiShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPathTracingGuidesShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneRayQueryClusterShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingSamplePreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingDebugViewsTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingEnvironmentRotationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceMaterialTexturesPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceTransmissionTexturesPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceAlphaMaskPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphResizeReusesCompiledPassesTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphCopyColorWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBindlessTextureWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBufferWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMultiQueueSubmitTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphImageSamplePassPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMaterialShaderObjectPassSmokeTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenPreviewPassSmokeTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenStreamAssetPassSmokeTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenMixedProducerRenderTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenPreviewPassRenderTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenAlphaMaskRenderTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUDrivenSponzaVisibilityRenderTest);
METALLIC_REGISTER_RHI_TEST(ImportancePdfSizeTest);
METALLIC_REGISTER_RHI_TEST(ReGIRGridLayoutTest);
METALLIC_REGISTER_RHI_TEST(EnvironmentSubsystemAsyncSnapshotTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMissingSubsystemDiagnosticTest);
METALLIC_REGISTER_RHI_TEST(RenderSubsystemHostLifecycleTest);

} // namespace
} // namespace metallic::tests
