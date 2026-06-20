#include "RhiTest.h"

#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::tests {
namespace {

constexpr const char* kShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kBindlessSmokeShaderModuleName = "bindless_smoke";
constexpr const char* kBindlessSmokeVertexEntryPoint = "bindlessSmokeVertexMain";
constexpr const char* kBindlessSmokeFragmentEntryPoint = "bindlessSmokeFragmentMain";

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
    std::string& outMessage)
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
      "doubleSided": true,
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
        const std::unique_ptr<render::RenderGraphPass> materialVisualization =
            render::createRenderGraphPass("SceneMaterialVisualizationPass");
        const std::unique_ptr<render::RenderGraphPass> nrdDenoise =
            render::createRenderGraphPass("NrdDenoisePass");

        if (triangle == nullptr ||
            copy == nullptr ||
            bufferWrite == nullptr ||
            pathTrace == nullptr ||
            materialVisualization == nullptr ||
            nrdDenoise == nullptr) {
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
        if (materialVisualization->kind() != render::RenderGraphPassKind::Compute ||
            materialVisualization->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("SceneMaterialVisualizationPass is not classified as Compute/Compute");
        }
        if (nrdDenoise->kind() != render::RenderGraphPassKind::Compute ||
            nrdDenoise->queueType() != render::QueueType::Compute) {
            return RhiTestResult::fail("NrdDenoisePass is not classified as Compute/Compute");
        }

        bool foundTriangle = false;
        bool foundCopy = false;
        bool foundBufferWrite = false;
        bool foundPathTrace = false;
        bool foundMaterialVisualization = false;
        bool foundNrdDenoise = false;
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
            } else if (passInfo.type == "SceneMaterialVisualizationPass") {
                foundMaterialVisualization = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            } else if (passInfo.type == "NrdDenoisePass") {
                foundNrdDenoise = passInfo.kind == render::RenderGraphPassKind::Compute &&
                    passInfo.queueType == render::QueueType::Compute;
            }
        }
        if (!foundTriangle ||
            !foundCopy ||
            !foundBufferWrite ||
            !foundPathTrace ||
            !foundMaterialVisualization ||
            !foundNrdDenoise) {
            return RhiTestResult::fail("RenderGraphPassInfo did not preserve pass kind metadata");
        }

        return RhiTestResult::pass();
    }
};


bool hasBoolRuntimeSetting(const render::RenderGraphPass& pass, const std::string& key)
{
    const std::vector<render::RenderGraphRuntimeSetting> settings = pass.runtimeSettings();
    for (const render::RenderGraphRuntimeSetting& setting : settings) {
        if (setting.key == key && setting.type == render::RenderGraphRuntimeSettingType::Bool) {
            return true;
        }
    }
    return false;
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
        const std::unique_ptr<render::RenderGraphPass> materialVisualization =
            render::createRenderGraphPass("SceneMaterialVisualizationPass");
        if (pathTrace == nullptr || materialVisualization == nullptr) {
            return RhiTestResult::fail("failed to create passes for runtime settings declaration test");
        }
        if (!hasBoolRuntimeSetting(*pathTrace, "flipBitangent")) {
            return RhiTestResult::fail("ScenePathTracePass missing Bool runtime setting flipBitangent");
        }
        if (!hasBoolRuntimeSetting(*materialVisualization, "flipBitangent")) {
            return RhiTestResult::fail("SceneMaterialVisualizationPass missing Bool runtime setting flipBitangent");
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
            pathTracingSample.desc.environment.path != "Asset/ABeautifulGame/environment.hdr" ||
            pathTracingSample.desc.previewOutput != "PathTrace.color") {
            return RhiTestResult::fail("OpenPBR PathTracingSample metadata did not load as expected");
        }
        const render::RenderGraphNode* openPBRPathTrace = pathTracingSample.graph.findNode("PathTrace");
        if (openPBRPathTrace == nullptr ||
            !openPBRPathTrace->properties.is_object() ||
            openPBRPathTrace->properties.value("path", "") != pathTracingSample.desc.scenePath ||
            openPBRPathTrace->properties.value("bsdf", "") != "openpbr" ||
            !openPBRPathTrace->properties.contains("environment") ||
            !openPBRPathTrace->properties["environment"].is_object() ||
            openPBRPathTrace->properties["environment"].value("path", "") != "Asset/ABeautifulGame/environment.hdr") {
            return RhiTestResult::fail("OpenPBR PathTracingSample did not apply scene, BSDF, and environment properties");
        }
        if (!pathTracingSample.graph.validate(validationLog)) {
            return RhiTestResult::fail(validationLog);
        }
        if (pathTracingSample.graph.firstOutputName() != "PathTrace.color") {
            return RhiTestResult::fail("OpenPBR PathTracingSample graph first output changed");
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

        bool listedPathTrace = false;
        bool listedOpenPBRPathTrace = false;
        bool listedMaterialVisualization = false;
        for (const render::RenderSampleDesc& desc : render::listBuiltInRenderSamples()) {
            listedPathTrace = listedPathTrace || desc.id == "pathtracing-meet-mat";
            listedOpenPBRPathTrace = listedOpenPBRPathTrace || desc.id == "pathtracing-sample";
            listedMaterialVisualization = listedMaterialVisualization ||
                desc.id == "material-visualization-abeautiful-game";
        }
        if (!listedPathTrace || !listedOpenPBRPathTrace || !listedMaterialVisualization) {
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

        std::string outputMessage;
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        const std::filesystem::path outputPath =
            context.outputDirectory / "render_graph_scene_rayquery_visualization_preview.png";
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
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
                .moduleName = "openpbr_rayquery_path_trace",
                .entryPointName = "openPbrRayQueryPathTraceMain",
                .searchPath = kShaderSearchPath,
                .profileName = "glsl_460",
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
        if (!sample.graph.setNodeRuntimeProperty(pathTrace->id, "maxDepth", 1) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "samples", 1) ||
            !sample.graph.setNodeRuntimeProperty(pathTrace->id, "accumulate", false)) {
            return RhiTestResult::fail("failed to set OpenPBR PathTracingSample preview runtime properties");
        }

        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false, true);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        result = preview.render(sample.graph, 64, 64, sample.desc.previewOutput);
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
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingShaderCompileTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphOpenPBRPathTracingSamplePreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceMaterialTexturesPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceTransmissionTexturesPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphScenePathTraceAlphaMaskPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphCopyColorWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBindlessTextureWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBufferWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMultiQueueSubmitTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphImageSamplePassPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphMaterialShaderObjectPassSmokeTest);

} // namespace
} // namespace metallic::tests
