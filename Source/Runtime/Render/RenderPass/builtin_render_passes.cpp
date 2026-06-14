#include "Runtime/Render/RenderGraph/render_graph.h"

#include "Runtime/Render/slang_compiler.h"
#include "Runtime/Scene/scene.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kTriangleShaderModuleName = "triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";
constexpr const char* kImageSampleShaderModuleName = "image_sample";
constexpr const char* kImageSampleVertexEntryPoint = "imageSampleVertexMain";
constexpr const char* kImageSampleFragmentEntryPoint = "imageSampleFragmentMain";
constexpr const char* kBunnyWireframeShaderModuleName = "bunny_wireframe";
constexpr const char* kBunnyWireframeVertexEntryPoint = "bunnyWireframeVertexMain";
constexpr const char* kBunnyWireframeFragmentEntryPoint = "bunnyWireframeFragmentMain";
constexpr const char* kRenderGraphBufferShaderModuleName = "render_graph_buffer";
constexpr const char* kRenderGraphBufferWriteEntryPoint = "renderGraphBufferWriteMain";
constexpr const char* kRenderGraphBufferCopyEntryPoint = "renderGraphBufferCopyMain";
constexpr const char* kDefaultImageSamplePath = PROJECT_SOURCE_DIR "/Asset/statue-1275469_1280.jpg";
constexpr const char* kDefaultBunnyScenePath = PROJECT_SOURCE_DIR "/Asset/StandfordBunny/scene.gltf";
constexpr uint64_t kRenderGraphBufferByteSize = 16;
constexpr int32_t kGltfTriangleListMode = 4;

struct RenderGraphBufferUserPush {
    uint32_t inputBuffer = 0;
    uint32_t outputBuffer = 0;
    uint32_t passIndex = 0;
    uint32_t padding = 0;
};

struct BunnyWireframeGpuPosition {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct BunnyWireframeGpuParams {
    float centerScale[4] = {};
    float viewport[4] = {};
    float clearColor[4] = {};
    float wireColor[4] = {};
    float settings[4] = {};
};

std::string resultMessage(std::string_view label, const Result& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

Result createSlangShaderModule(
    Device& device,
    const char* moduleName,
    const char* entryPointName,
    std::unique_ptr<ShaderModule>& outShaderModule,
    std::string& log)
{
    ShaderCompileResult compileResult;
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        },
        compileResult);
    if (!result) {
        log += "compileSlangShaderToSpirv(";
        log += moduleName;
        log += ".";
        log += entryPointName;
        log += ") returned ";
        log += resultToString(result);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        log += '\n';
        return result;
    }

    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
        },
        outShaderModule);
    if (!result) {
        log += resultMessage("createShaderModule", result);
        log += '\n';
    }
    return result;
}

class ClearColorPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Cleared color target")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid()) {
            return makeError(Error::InvalidArgument);
        }

        ColorValue clear{0.04f, 0.06f, 0.09f, 1.0f};
        const RenderGraphProperties& props = context.properties();
        if (props.contains("color") && props["color"].is_array() && props["color"].size() >= 4) {
            clear.r = props["color"][0].get<float>();
            clear.g = props["color"][1].get<float>();
            clear.b = props["color"][2].get<float>();
            clear.a = props["color"][3].get<float>();
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = clear,
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().endRendering();
        return {};
    }
};

class CopyColorPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("source", "Source color texture")
            .transferRead();
        reflection.addTextureOutput("color", "Copied color texture")
            .transferWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle source = context.inputTexture("source");
        TextureHandle color = context.outputTexture("color");
        if (!source.valid() || !color.valid()) {
            return makeError(Error::InvalidArgument);
        }

        context.commandBuffer().copyTexture(TextureCopyDesc{
            .source = source.texture(),
            .destination = color.texture(),
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .sourceMipLevel = 0,
            .sourceBaseLayer = 0,
            .destinationMipLevel = 0,
            .destinationBaseLayer = 0,
        });
        return {};
    }
};

class TriangleRasterPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Rasterized triangle color")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result result = createShaderModule(*context.device, kTriangleVertexEntryPoint, vertexShader_, log);
        if (!result) {
            return result;
        }
        result = createShaderModule(*context.device, kTriangleFragmentEntryPoint, fragmentShader_, log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = vertexShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
                .topology = PrimitiveTopology::TriangleList,
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createGraphicsPipeline", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() || pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.04f, 0.06f, 0.09f, 1.0f},
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(Viewport{
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
    static Result createShaderModule(
        Device& device,
        const char* entryPointName,
        std::unique_ptr<ShaderModule>& outShaderModule,
        std::string& log)
    {
        ShaderCompileResult compileResult;
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kTriangleShaderModuleName,
                .entryPointName = entryPointName,
                .searchPath = kTriangleShaderSearchPath,
            },
            compileResult);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += entryPointName;
            log += ") returned ";
            log += resultToString(result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            log += '\n';
            return result;
        }

        result = device.createShaderModule(
            ShaderModuleDesc{
                .code = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            },
            outShaderModule);
        if (!result) {
            log += resultMessage("createShaderModule", result);
            log += '\n';
        }
        return result;
    }

    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
};

class ImageSamplePass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Fullscreen sampled image")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "ImageSamplePass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        const std::string imagePath = imagePathFromProperties();
        int imageWidth = 0;
        int imageHeight = 0;
        int channelCount = 0;
        stbi_uc* pixels = stbi_load(imagePath.c_str(), &imageWidth, &imageHeight, &channelCount, 4);
        if (pixels == nullptr || imageWidth <= 0 || imageHeight <= 0) {
            log = std::string("ImageSamplePass failed to load image '") + imagePath + "'";
            if (const char* reason = stbi_failure_reason()) {
                log += ": ";
                log += reason;
            }
            return makeError(Error::Failure);
        }

        imageWidth_ = static_cast<uint32_t>(imageWidth);
        imageHeight_ = static_cast<uint32_t>(imageHeight);
        const uint64_t imageByteSize =
            static_cast<uint64_t>(imageWidth_) * static_cast<uint64_t>(imageHeight_) * 4ull;

        Result result = context.device->createBuffer(
            BufferDesc{
                .size = imageByteSize,
                .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload,
            },
            uploadBuffer_);
        if (!result || uploadBuffer_ == nullptr) {
            stbi_image_free(pixels);
            log += resultMessage("createBuffer(ImageSamplePass upload)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        void* mapped = uploadBuffer_->map();
        if (mapped == nullptr) {
            stbi_image_free(pixels);
            log = "ImageSamplePass failed to map upload buffer";
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, pixels, static_cast<size_t>(imageByteSize));
        uploadBuffer_->flush(0, imageByteSize);
        uploadBuffer_->unmap();
        stbi_image_free(pixels);

        result = context.device->createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::Rgba8Unorm,
                .width = imageWidth_,
                .height = imageHeight_,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            },
            imageTexture_);
        if (!result || imageTexture_ == nullptr) {
            log += resultMessage("createTexture(ImageSamplePass image)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = context.device->createTextureView(
            *imageTexture_,
            TextureViewDesc{
                .format = Format::Rgba8Unorm,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            imageView_);
        if (!result || imageView_ == nullptr) {
            log += resultMessage("createTextureView(ImageSamplePass image)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = context.device->createBindlessHeap(
            BindlessHeapDesc{
                .maxSampledImages = 1,
            },
            bindlessHeap_);
        if (!result || bindlessHeap_ == nullptr) {
            log += resultMessage("createBindlessHeap(ImageSamplePass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = bindlessHeap_->allocateSampledImage(imageHandle_);
        if (!result || !imageHandle_.valid()) {
            log += resultMessage("allocateSampledImage(ImageSamplePass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = bindlessHeap_->writeSampledImage(
            imageHandle_,
            *imageView_,
            ResourceState::ShaderRead);
        if (!result) {
            log += resultMessage("writeSampledImage(ImageSamplePass)", result);
            log += '\n';
            return result;
        }

        result = createShaderModule(*context.device, kImageSampleVertexEntryPoint, vertexShader_, log);
        if (!result) {
            return result;
        }
        result = createShaderModule(*context.device, kImageSampleFragmentEntryPoint, fragmentShader_, log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = vertexShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
                .topology = PrimitiveTopology::TriangleList,
                .usesBindlessHeap = true,
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createGraphicsPipeline(ImageSamplePass)", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            uploadBuffer_ == nullptr ||
            imageTexture_ == nullptr ||
            bindlessHeap_ == nullptr ||
            pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        if (!uploaded_) {
            TextureBarrierDesc toTransfer{
                .texture = imageTexture_.get(),
                .before = imageState_,
                .after = ResourceState::TransferDestination,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            };
            context.commandBuffer().barrier(BarrierDesc{
                .textures = &toTransfer,
                .textureCount = 1,
            });
            imageState_ = ResourceState::TransferDestination;

            context.commandBuffer().copyBufferToTexture(BufferTextureCopyDesc{
                .buffer = uploadBuffer_.get(),
                .texture = imageTexture_.get(),
                .width = imageWidth_,
                .height = imageHeight_,
                .depth = 1,
                .mipLevel = 0,
                .baseLayer = 0,
            });

            TextureBarrierDesc toShaderRead{
                .texture = imageTexture_.get(),
                .before = imageState_,
                .after = ResourceState::ShaderRead,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            };
            context.commandBuffer().barrier(BarrierDesc{
                .textures = &toShaderRead,
                .textureCount = 1,
            });
            imageState_ = ResourceState::ShaderRead;
            uploaded_ = true;
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        context.commandBuffer().bindBindlessHeap(*bindlessHeap_);
        context.commandBuffer().bindGraphicsPipeline(*pipeline_);
        context.commandBuffer().draw(3);
        context.commandBuffer().endRendering();
        return {};
    }

private:
    static Result createShaderModule(
        Device& device,
        const char* entryPointName,
        std::unique_ptr<ShaderModule>& outShaderModule,
        std::string& log)
    {
        ShaderCompileResult compileResult;
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kImageSampleShaderModuleName,
                .entryPointName = entryPointName,
                .searchPath = kTriangleShaderSearchPath,
            },
            compileResult);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += entryPointName;
            log += ") returned ";
            log += resultToString(result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            log += '\n';
            return result;
        }

        result = device.createShaderModule(
            ShaderModuleDesc{
                .code = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            },
            outShaderModule);
        if (!result) {
            log += resultMessage("createShaderModule(ImageSamplePass)", result);
            log += '\n';
        }
        return result;
    }

    std::string imagePathFromProperties() const
    {
        const RenderGraphProperties& props = properties();
        if (props.contains("path") && props["path"].is_string()) {
            std::filesystem::path path = props["path"].get<std::string>();
            if (path.is_relative()) {
                path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
            }
            return path.string();
        }
        return kDefaultImageSamplePath;
    }

    std::unique_ptr<Buffer> uploadBuffer_;
    std::unique_ptr<Texture> imageTexture_;
    std::unique_ptr<TextureView> imageView_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle imageHandle_;
    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
    uint32_t imageWidth_ = 0;
    uint32_t imageHeight_ = 0;
    ResourceState imageState_ = ResourceState::Undefined;
    bool uploaded_ = false;
};

class BunnyWireframePass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Stanford Bunny barycentric wireframe")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "BunnyWireframePass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        std::vector<BunnyWireframeGpuPosition> positions;
        BunnyWireframeGpuParams params;
        if (!loadBunnyGeometry(context.width, context.height, properties(), positions, params, log)) {
            return makeError(Error::Failure);
        }
        if (positions.size() > std::numeric_limits<uint32_t>::max()) {
            log = "BunnyWireframePass geometry is too large to draw";
            return makeError(Error::Unsupported);
        }

        Result result = uploadStorageBuffer(
            *context.device,
            positions.data(),
            static_cast<uint64_t>(positions.size() * sizeof(BunnyWireframeGpuPosition)),
            positionBuffer_,
            log,
            "BunnyWireframePass positions");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            &params,
            sizeof(params),
            paramsBuffer_,
            log,
            "BunnyWireframePass params");
        if (!result) {
            return result;
        }

        result = context.device->createBindlessHeap(
            BindlessHeapDesc{
                .maxBuffers = 2,
            },
            bindlessHeap_);
        if (!result || bindlessHeap_ == nullptr) {
            log += resultMessage("createBindlessHeap(BunnyWireframePass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = bindlessHeap_->allocateBuffer(paramsHandle_);
        if (!result || !paramsHandle_.valid()) {
            log += resultMessage("allocateBuffer(BunnyWireframePass params)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = bindlessHeap_->allocateBuffer(positionHandle_);
        if (!result || !positionHandle_.valid()) {
            log += resultMessage("allocateBuffer(BunnyWireframePass positions)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        if (paramsHandle_.index != 0 || positionHandle_.index != 1) {
            log = "BunnyWireframePass expected fresh bindless buffer handles 0 and 1";
            return makeError(Error::Failure);
        }

        result = bindlessHeap_->writeStorageBuffer(paramsHandle_, *paramsBuffer_);
        if (!result) {
            log += resultMessage("writeStorageBuffer(BunnyWireframePass params)", result);
            log += '\n';
            return result;
        }
        result = bindlessHeap_->writeStorageBuffer(positionHandle_, *positionBuffer_);
        if (!result) {
            log += resultMessage("writeStorageBuffer(BunnyWireframePass positions)", result);
            log += '\n';
            return result;
        }

        result = createSlangShaderModule(
            *context.device,
            kBunnyWireframeShaderModuleName,
            kBunnyWireframeVertexEntryPoint,
            vertexShader_,
            log);
        if (!result) {
            return result;
        }
        result = createSlangShaderModule(
            *context.device,
            kBunnyWireframeShaderModuleName,
            kBunnyWireframeFragmentEntryPoint,
            fragmentShader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = vertexShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
                .topology = PrimitiveTopology::TriangleList,
                .usesBindlessHeap = true,
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createGraphicsPipeline(BunnyWireframePass)", result);
            log += '\n';
            return result;
        }

        drawVertexCount_ = static_cast<uint32_t>(positions.size());
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            bindlessHeap_ == nullptr ||
            pipeline_ == nullptr ||
            drawVertexCount_ == 0) {
            return makeError(Error::InvalidArgument);
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.015f, 0.018f, 0.024f, 1.0f},
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        context.commandBuffer().bindBindlessHeap(*bindlessHeap_);
        context.commandBuffer().bindGraphicsPipeline(*pipeline_);
        context.commandBuffer().draw(drawVertexCount_);
        context.commandBuffer().endRendering();
        return {};
    }

private:
    static Result uploadStorageBuffer(
        Device& device,
        const void* data,
        uint64_t byteSize,
        std::unique_ptr<Buffer>& outBuffer,
        std::string& log,
        std::string_view label)
    {
        if (data == nullptr || byteSize == 0) {
            log = std::string(label) + " upload data is empty";
            return makeError(Error::InvalidArgument);
        }

        Result result = device.createBuffer(
            BufferDesc{
                .size = byteSize,
                .structureStride = 0,
                .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostUpload,
            },
            outBuffer);
        if (!result || outBuffer == nullptr) {
            log += resultMessage(std::string("createBuffer(") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        void* mapped = outBuffer->map();
        if (mapped == nullptr) {
            log = std::string(label) + " failed to map upload buffer";
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        outBuffer->flush(0, byteSize);
        outBuffer->unmap();
        return {};
    }

    static std::filesystem::path scenePathFromProperties(const RenderGraphProperties& props)
    {
        if (props.contains("path") && props["path"].is_string()) {
            std::filesystem::path path = props["path"].get<std::string>();
            if (path.is_relative()) {
                path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
            }
            return path;
        }
        return kDefaultBunnyScenePath;
    }

    static bool loadBunnyGeometry(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        std::vector<BunnyWireframeGpuPosition>& outPositions,
        BunnyWireframeGpuParams& outParams,
        std::string& log)
    {
        scene::Scene bunnyScene;
        const std::filesystem::path path = scenePathFromProperties(properties);
        if (!bunnyScene.load(path)) {
            log = "BunnyWireframePass failed to load glTF: " + bunnyScene.lastLoadResult().error;
            return false;
        }

        scene::Bounds drawBounds;
        for (const scene::RenderNode& renderNode : bunnyScene.renderNodes()) {
            if (renderNode.renderPrimitiveIndex < 0 ||
                static_cast<size_t>(renderNode.renderPrimitiveIndex) >= bunnyScene.renderPrimitives().size()) {
                continue;
            }

            const scene::RenderPrimitive& primitive =
                bunnyScene.renderPrimitives()[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
            if (primitive.mode != kGltfTriangleListMode || primitive.positions.empty()) {
                continue;
            }

            const auto appendVertex = [&](uint32_t localIndex) {
                if (static_cast<size_t>(localIndex) >= primitive.positions.size()) {
                    return;
                }
                const float3 world = renderNode.worldMatrix * primitive.positions[static_cast<size_t>(localIndex)];
                outPositions.push_back(BunnyWireframeGpuPosition{
                    .x = world.x,
                    .y = world.y,
                    .z = world.z,
                    .w = 1.0f,
                });
                drawBounds.include(world);
            };

            const std::vector<uint32_t>& indices = primitive.indices;
            if (!indices.empty()) {
                const size_t triangleIndexCount = indices.size() - (indices.size() % 3);
                for (size_t index = 0; index < triangleIndexCount; ++index) {
                    appendVertex(indices[index]);
                }
            } else {
                const size_t triangleVertexCount = primitive.positions.size() - (primitive.positions.size() % 3);
                for (size_t index = 0; index < triangleVertexCount; ++index) {
                    appendVertex(static_cast<uint32_t>(index));
                }
            }
        }

        if (outPositions.empty() || !drawBounds.valid) {
            log = "BunnyWireframePass found no drawable triangle geometry in " + path.string();
            return false;
        }

        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const float scale = frameHalfHeight > 0.000001f ? 0.82f / frameHalfHeight : 1.0f;

        outParams.centerScale[0] = center.x;
        outParams.centerScale[1] = center.y;
        outParams.centerScale[2] = center.z;
        outParams.centerScale[3] = scale;
        outParams.viewport[0] = aspect;
        outParams.viewport[1] = static_cast<float>(width);
        outParams.viewport[2] = static_cast<float>(height);
        outParams.clearColor[0] = 0.015f;
        outParams.clearColor[1] = 0.018f;
        outParams.clearColor[2] = 0.024f;
        outParams.clearColor[3] = 1.0f;
        outParams.wireColor[0] = 0.82f;
        outParams.wireColor[1] = 0.92f;
        outParams.wireColor[2] = 1.0f;
        outParams.wireColor[3] = 1.0f;
        outParams.settings[0] = 0.75f;
        outParams.settings[1] = 0.75f;
        outParams.settings[2] = 0.0f;
        outParams.settings[3] = 0.0f;
        return true;
    }

    std::unique_ptr<Buffer> positionBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle positionHandle_;
    BindlessHandle paramsHandle_;
    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
    uint32_t drawVertexCount_ = 0;
};

class RenderGraphBufferWritePass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Known test byte pattern")
            .buffer(kRenderGraphBufferByteSize)
            .storageReadWrite()
            .bindlessBuffer();
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "RenderGraphBufferWritePass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result result = createSlangShaderModule(
            *context.device,
            kRenderGraphBufferShaderModuleName,
            kRenderGraphBufferWriteEntryPoint,
            shader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createComputePipeline(
            ComputePipelineDesc{
                .computeShader = shader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(RenderGraphBufferUserPush),
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createComputePipeline(RenderGraphBufferWritePass)", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        BufferHandle data = context.outputBuffer("data");
        if (!data.valid() || !data.bindlessHandle().valid() || pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphBufferUserPush push{
            .inputBuffer = 0,
            .outputBuffer = data.bindlessHandle().index,
            .passIndex = 0,
            .padding = 0,
        };
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().bindComputePipeline(*pipeline_);
        context.commandBuffer().dispatch(1, 1, 1);
        return {};
    }

private:
    std::unique_ptr<ShaderModule> shader_;
    std::unique_ptr<ComputePipeline> pipeline_;
};

class RenderGraphBufferCopyPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferInput("source", "Source byte buffer")
            .buffer(kRenderGraphBufferByteSize)
            .storageReadWrite()
            .bindlessBuffer();
        reflection.addBufferOutput("data", "Copied byte buffer")
            .buffer(kRenderGraphBufferByteSize)
            .storageReadWrite()
            .bindlessBuffer();
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "RenderGraphBufferCopyPass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result result = createSlangShaderModule(
            *context.device,
            kRenderGraphBufferShaderModuleName,
            kRenderGraphBufferCopyEntryPoint,
            shader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createComputePipeline(
            ComputePipelineDesc{
                .computeShader = shader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(RenderGraphBufferUserPush),
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createComputePipeline(RenderGraphBufferCopyPass)", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        BufferHandle source = context.inputBuffer("source");
        BufferHandle data = context.outputBuffer("data");
        if (!source.valid() ||
            !source.bindlessHandle().valid() ||
            !data.valid() ||
            !data.bindlessHandle().valid() ||
            pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphBufferUserPush push{
            .inputBuffer = source.bindlessHandle().index,
            .outputBuffer = data.bindlessHandle().index,
            .passIndex = 0,
            .padding = 0,
        };
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().bindComputePipeline(*pipeline_);
        context.commandBuffer().dispatch(1, 1, 1);
        return {};
    }

private:
    std::unique_ptr<ShaderModule> shader_;
    std::unique_ptr<ComputePipeline> pipeline_;
};

} // namespace

void registerBuiltInRenderGraphPasses()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;

    registerRenderGraphPassType(
        "ClearColorPass",
        "Clear a color texture",
        []() { return std::make_unique<ClearColorPass>(); });
    registerRenderGraphPassType(
        "CopyColorPass",
        "Copy a color texture",
        []() { return std::make_unique<CopyColorPass>(); });
    registerRenderGraphPassType(
        "TriangleRasterPass",
        "Rasterize the built-in triangle shader",
        []() { return std::make_unique<TriangleRasterPass>(); });
    registerRenderGraphPassType(
        "ImageSamplePass",
        "Draw a fullscreen sampled image",
        []() { return std::make_unique<ImageSamplePass>(); });
    registerRenderGraphPassType(
        "BunnyWireframePass",
        "Draw the Stanford Bunny glTF as a barycentric wireframe",
        []() { return std::make_unique<BunnyWireframePass>(); });
    registerRenderGraphPassType(
        "RenderGraphBufferWritePass",
        "Write a known byte pattern into a graph buffer",
        []() { return std::make_unique<RenderGraphBufferWritePass>(); });
    registerRenderGraphPassType(
        "RenderGraphBufferCopyPass",
        "Copy a graph buffer through bindless compute",
        []() { return std::make_unique<RenderGraphBufferCopyPass>(); });
}

} // namespace metallic::render
