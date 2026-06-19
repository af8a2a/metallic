#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace metallic::render::builtin_pass {
namespace {

class ImageSamplePass final : public UnsafePass {
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

} // namespace

std::unique_ptr<RenderGraphPass> createImageSamplePass()
{
    return std::make_unique<ImageSamplePass>();
}

} // namespace metallic::render::builtin_pass
