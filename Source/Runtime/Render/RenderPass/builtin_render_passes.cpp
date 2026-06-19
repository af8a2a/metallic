#include "Runtime/Render/RenderGraph/render_graph.h"

#include "Runtime/Render/GAPI/scene_rtx.h"
#include "Runtime/Render/GAPI/Vulkan/vulkan_nrd_wrapper.h"
#include "Runtime/Render/RenderPass/scene_path_trace_resources.h"
#include "Runtime/Render/slang_compiler.h"
#include "Runtime/Render/history_resources.h"
#include "Runtime/Scene/scene.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cmath>
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
constexpr const char* kMaterialShaderObjectShaderModuleName = "material_shader_object";
constexpr const char* kMaterialShaderObjectVertexEntryPoint = "materialShaderObjectVertexMain";
constexpr const char* kMaterialShaderObjectFragmentEntryPoint = "materialShaderObjectFragmentMain";
constexpr const char* kMaterialShaderObjectAlternateFragmentEntryPoint =
    "materialShaderObjectAlternateFragmentMain";
constexpr const char* kSceneRayQueryVisualizationShaderModuleName = "scene_rayquery_visualize";
constexpr const char* kSceneRayQueryVisualizationEntryPoint = "sceneRayQueryVisualizeMain";
constexpr const char* kScenePathTraceShaderModuleName = "scene_path_trace";
constexpr const char* kScenePathTraceEntryPoint = "scenePathTraceMain";
constexpr const char* kRenderGraphBufferShaderModuleName = "render_graph_buffer";
constexpr const char* kRenderGraphBufferWriteEntryPoint = "renderGraphBufferWriteMain";
constexpr const char* kRenderGraphBufferCopyEntryPoint = "renderGraphBufferCopyMain";
constexpr const char* kDefaultImageSamplePath = PROJECT_SOURCE_DIR "/Asset/statue-1275469_1280.jpg";
constexpr const char* kDefaultBunnyScenePath = PROJECT_SOURCE_DIR "/Asset/StandfordBunny/scene.gltf";
constexpr const char* kDefaultMaterialScenePath = PROJECT_SOURCE_DIR "/Asset/StandfordBunny/scene.gltf";
constexpr uint64_t kRenderGraphBufferByteSize = 16;
constexpr int32_t kGltfTriangleListMode = 4;
constexpr uint32_t kRayQueryVisualizationGranularityInstance = 0;
constexpr uint32_t kRayQueryVisualizationGranularityPrimitive = 1;
constexpr uint32_t kDefaultPathTraceMaxDepth = 3;
constexpr uint32_t kDefaultPathTraceSamples = 2;
constexpr uint32_t kNrdDenoiserModeReblur = 0;
constexpr uint32_t kNrdDenoiserModeRelax = 1;
constexpr uint32_t kNrdDenoiserModeReference = 2;
constexpr const char* kScenePathTraceHistoryPrefix = "ScenePathTracePass.";
constexpr bool kDefaultReversedZ = true;

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
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    float clearColor[4] = {};
    float wireColor[4] = {};
    float settings[4] = {};
};

struct BunnyWireframeUserPush {
    uint32_t paramsBuffer = 0;
    uint32_t positionBuffer = 0;
};

struct MaterialShaderObjectGpuPosition {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct MaterialShaderObjectGpuMaterial {
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

struct MaterialShaderObjectGpuParams {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
};

struct MaterialShaderObjectUserPush {
    uint32_t positionBuffer = 0;
    uint32_t materialIndexBuffer = 0;
    uint32_t materialBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t vertexOffset = 0;
    uint32_t materialVariant = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct MaterialShaderObjectBatch {
    uint32_t materialIndex = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
};

struct SceneRayQueryVisualizationPush {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    uint32_t mode = kRayQueryVisualizationGranularityInstance;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t padding = 0;
};

struct ScenePathTracePush {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t maxDepth = kDefaultPathTraceMaxDepth;
    uint32_t samples = kDefaultPathTraceSamples;
    uint32_t accumulationFrame = 0;
    uint32_t hasHistory = 0;
    uint32_t enableAccumulation = 1;
    uint32_t materialTextureCount = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

std::string resultMessage(std::string_view label, const Result& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

bool boolProperty(const RenderGraphProperties* properties, const char* key, bool fallback)
{
    if (properties == nullptr) {
        return fallback;
    }
    auto iter = properties->find(key);
    return iter != properties->end() && iter->is_boolean() ? iter->get<bool>() : fallback;
}

bool cameraUsesReversedZ(const RenderGraphProperties* camera)
{
    return boolProperty(camera, "reversedZ", kDefaultReversedZ);
}

float depthClearValue(bool reversedZ)
{
    return reversedZ ? 0.0f : 1.0f;
}

CompareOp depthCompareOp(bool reversedZ)
{
    return reversedZ ? CompareOp::GreaterEqual : CompareOp::LessEqual;
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

Result compileSlangShader(
    const char* moduleName,
    const char* entryPointName,
    ShaderCompileResult& outCompileResult,
    std::string& log)
{
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        },
        outCompileResult);
    if (!result) {
        log += "compileSlangShaderToSpirv(";
        log += moduleName;
        log += ".";
        log += entryPointName;
        log += ") returned ";
        log += resultToString(result);
        if (!outCompileResult.diagnostics.empty()) {
            log += ": ";
            log += outCompileResult.diagnostics;
        }
        log += '\n';
    }
    return result;
}

class ClearColorPass final : public RasterPass {
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

class CopyColorPass final : public UnsafePass {
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

class TriangleRasterPass final : public RasterPass {
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

class BunnyWireframePass final : public RasterPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Stanford Bunny barycentric wireframe")
            .format = Format::Rgba8Unorm;
        reflection.addTextureOutput("depth", "Stanford Bunny depth")
            .depthStencilWrite();
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().shaderObject ||
            !context.device->capabilities().bindlessDescriptorHeap) {
            log = "BunnyWireframePass requires shaderObject and bindlessDescriptorHeap capabilities";
            return makeError(Error::Unsupported);
        }
        if (program_ != nullptr) {
            return {};
        }

        std::vector<BunnyWireframeGpuPosition> positions;
        BunnyWireframeGpuParams params;
        if (!loadBunnyGeometry(properties(), positions, drawBounds_, log)) {
            return makeError(Error::Failure);
        }
        if (positions.size() > std::numeric_limits<uint32_t>::max()) {
            log = "BunnyWireframePass geometry is too large to draw";
            return makeError(Error::Unsupported);
        }
        buildBunnyParams(context.width, context.height, properties(), drawBounds_, params);

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

        ShaderCompileResult vertexCompile;
        result = compileSlangShader(
            kBunnyWireframeShaderModuleName,
            kBunnyWireframeVertexEntryPoint,
            vertexCompile,
            log);
        if (!result) {
            return result;
        }
        ShaderCompileResult fragmentCompile;
        result = compileSlangShader(
            kBunnyWireframeShaderModuleName,
            kBunnyWireframeFragmentEntryPoint,
            fragmentCompile,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsShaderObjectProgram(
            GraphicsShaderObjectProgramDesc{
                .vertexCode = vertexCompile.spirv.data(),
                .vertexByteSize = static_cast<uint64_t>(vertexCompile.spirv.size() * sizeof(uint32_t)),
                .fragmentCode = fragmentCompile.spirv.data(),
                .fragmentByteSize = static_cast<uint64_t>(fragmentCompile.spirv.size() * sizeof(uint32_t)),
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(BunnyWireframeUserPush),
            },
            program_);
        if (!result || program_ == nullptr) {
            log += resultMessage("createGraphicsShaderObjectProgram(BunnyWireframePass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        drawVertexCount_ = static_cast<uint32_t>(positions.size());
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !depth.valid() ||
            bindlessHeap_ == nullptr ||
            program_ == nullptr ||
            drawVertexCount_ == 0) {
            return makeError(Error::InvalidArgument);
        }

        Result result = updateParamsBuffer(context.width(), context.height(), context.properties());
        if (!result) {
            return result;
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
        const bool reversedZ = cameraUsesReversedZ(cameraPropertiesFrom(context.properties()));
        RenderingAttachmentDesc depthAttachment{
            .view = depth.view(),
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearDepth = depthClearValue(reversedZ),
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &depthAttachment,
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
        context.commandBuffer().bindGraphicsShaderObjectProgram(*program_);
        context.commandBuffer().setGraphicsShaderObjectState();
        context.commandBuffer().setDepthStencilState(DepthStencilState{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = depthCompareOp(reversedZ),
        });
        const BunnyWireframeUserPush push{
            .paramsBuffer = paramsHandle_.index,
            .positionBuffer = positionHandle_.index,
        };
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
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

    Result updateParamsBuffer(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties)
    {
        if (paramsBuffer_ == nullptr || !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        BunnyWireframeGpuParams params;
        buildBunnyParams(width, height, properties, drawBounds_, params);

        void* mapped = paramsBuffer_->map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, &params, sizeof(params));
        paramsBuffer_->flush(0, sizeof(params));
        paramsBuffer_->unmap();
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

    static float finiteOr(float value, float fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    static const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return nullptr;
        }
        auto iter = properties.find("camera");
        if (iter == properties.end() || !iter->is_object()) {
            return nullptr;
        }
        return &(*iter);
    }

    static float cameraFloat(
        const RenderGraphProperties* camera,
        const char* key,
        float fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_number()) {
            return fallback;
        }
        return finiteOr(iter->get<float>(), fallback);
    }

    static float3 cameraVec3(
        const RenderGraphProperties* camera,
        const char* key,
        const float3& fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_array() || iter->size() < 3) {
            return fallback;
        }

        float values[3] = {fallback.x, fallback.y, fallback.z};
        for (size_t index = 0; index < 3; ++index) {
            const RenderGraphProperties& component = (*iter)[index];
            if (component.is_number()) {
                values[index] = finiteOr(component.get<float>(), values[index]);
            }
        }
        return float3(values[0], values[1], values[2]);
    }

    static bool cameraIsOrthographic(const RenderGraphProperties* camera)
    {
        if (camera == nullptr) {
            return false;
        }
        auto iter = camera->find("projection");
        if (iter == camera->end() || !iter->is_string()) {
            return false;
        }
        const std::string projection = iter->get<std::string>();
        return projection == "orthographic" || projection == "ortho";
    }

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    static bool loadBunnyGeometry(
        const RenderGraphProperties& properties,
        std::vector<BunnyWireframeGpuPosition>& outPositions,
        scene::Bounds& outBounds,
        std::string& log)
    {
        scene::Scene bunnyScene;
        const std::filesystem::path path = scenePathFromProperties(properties);
        if (!bunnyScene.load(path)) {
            log = "BunnyWireframePass failed to load glTF: " + bunnyScene.lastLoadResult().error;
            return false;
        }

        outBounds.reset();
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
                outBounds.include(world);
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

        if (outPositions.empty() || !outBounds.valid) {
            log = "BunnyWireframePass found no drawable triangle geometry in " + path.string();
            return false;
        }

        return true;
    }

    static void buildBunnyParams(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const scene::Bounds& drawBounds,
        BunnyWireframeGpuParams& outParams)
    {
        outParams = BunnyWireframeGpuParams{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 60.0f),
            1.0f,
            179.0f);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float halfDepth = std::max(halfExtent.z, 0.001f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + halfDepth
                : 1.0f,
            0.05f);
        const float3 defaultEye(center.x, center.y, center.z + defaultDistance);
        const float3 eye = cameraVec3(cameraProperties, "eye", defaultEye);
        const float3 target = cameraVec3(cameraProperties, "center", center);
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.1f), 0.0001f);
        const float zFar = std::max(cameraFloat(cameraProperties, "zfar", 10000.0f), zNear + 0.001f);
        const bool reversedZ = cameraUsesReversedZ(cameraProperties);
        const float cameraDistance = std::max(length(eye - target), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(
            cameraFloat(cameraProperties, "orthoHeight", defaultOrthoHeight),
            0.0001f);

        writeParamVec3(eye, outParams.eye, 0.0f);
        writeParamVec3(target, outParams.center, 0.0f);
        writeParamVec3(up, outParams.upProjection, cameraIsOrthographic(cameraProperties) ? 1.0f : 0.0f);
        outParams.viewport[0] = aspect;
        outParams.viewport[1] = static_cast<float>(width);
        outParams.viewport[2] = static_cast<float>(height);
        outParams.viewport[3] = fovRadians;
        outParams.clipOrtho[0] = zNear;
        outParams.clipOrtho[1] = zFar;
        outParams.clipOrtho[2] = orthoHeight;
        outParams.clipOrtho[3] = reversedZ ? 1.0f : 0.0f;
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
    }

    std::unique_ptr<Buffer> positionBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle positionHandle_;
    BindlessHandle paramsHandle_;
    std::unique_ptr<GraphicsShaderObjectProgram> program_;
    scene::Bounds drawBounds_;
    uint32_t drawVertexCount_ = 0;
};

class SceneMaterialShaderObjectPass final : public RasterPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "glTF material color via VK_EXT_shader_object")
            .format = Format::Rgba8Unorm;
        reflection.addTextureOutput("depth", "glTF material depth")
            .depthStencilWrite();
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().shaderObject ||
            !context.device->capabilities().bindlessDescriptorHeap) {
            log = "SceneMaterialShaderObjectPass requires shaderObject and bindlessDescriptorHeap capabilities";
            return makeError(Error::Unsupported);
        }
        if (defaultProgram_ != nullptr && !batches_.empty()) {
            return {};
        }

        std::vector<MaterialShaderObjectGpuPosition> positions;
        std::vector<uint32_t> materialIndices;
        std::vector<MaterialShaderObjectGpuMaterial> materials;
        if (!loadSceneGeometry(properties(), positions, materialIndices, materials, batches_, drawBounds_, log)) {
            return makeError(Error::Failure);
        }

        MaterialShaderObjectGpuParams params;
        buildParams(context.width, context.height, drawBounds_, params);

        Result result = uploadStorageBuffer(
            *context.device,
            positions.data(),
            static_cast<uint64_t>(positions.size() * sizeof(MaterialShaderObjectGpuPosition)),
            positionBuffer_,
            log,
            "SceneMaterialShaderObjectPass positions");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            materialIndices.data(),
            static_cast<uint64_t>(materialIndices.size() * sizeof(uint32_t)),
            materialIndexBuffer_,
            log,
            "SceneMaterialShaderObjectPass material indices");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            materials.data(),
            static_cast<uint64_t>(materials.size() * sizeof(MaterialShaderObjectGpuMaterial)),
            materialBuffer_,
            log,
            "SceneMaterialShaderObjectPass materials");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            &params,
            sizeof(params),
            paramsBuffer_,
            log,
            "SceneMaterialShaderObjectPass params");
        if (!result) {
            return result;
        }

        result = context.device->createBindlessHeap(
            BindlessHeapDesc{
                .maxBuffers = 4,
            },
            bindlessHeap_);
        if (!result || bindlessHeap_ == nullptr) {
            log += resultMessage("createBindlessHeap(SceneMaterialShaderObjectPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = allocateAndWriteBuffer(*bindlessHeap_, *positionBuffer_, positionHandle_, log, "positions");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *materialIndexBuffer_, materialIndexHandle_, log, "material indices");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *materialBuffer_, materialHandle_, log, "materials");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *paramsBuffer_, paramsHandle_, log, "params");
        if (!result) {
            return result;
        }

        ShaderCompileResult vertexCompile;
        result = compileSlangShader(
            kMaterialShaderObjectShaderModuleName,
            kMaterialShaderObjectVertexEntryPoint,
            vertexCompile,
            log);
        if (!result) {
            return result;
        }
        ShaderCompileResult fragmentCompile;
        result = compileSlangShader(
            kMaterialShaderObjectShaderModuleName,
            kMaterialShaderObjectFragmentEntryPoint,
            fragmentCompile,
            log);
        if (!result) {
            return result;
        }
        ShaderCompileResult alternateFragmentCompile;
        result = compileSlangShader(
            kMaterialShaderObjectShaderModuleName,
            kMaterialShaderObjectAlternateFragmentEntryPoint,
            alternateFragmentCompile,
            log);
        if (!result) {
            return result;
        }

        result = createProgram(*context.device, vertexCompile, fragmentCompile, defaultProgram_, log, "default");
        if (!result) {
            return result;
        }
        result = createProgram(*context.device, vertexCompile, alternateFragmentCompile, alternateProgram_, log, "alternate");
        if (!result) {
            return result;
        }

        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !depth.valid() ||
            bindlessHeap_ == nullptr ||
            defaultProgram_ == nullptr ||
            alternateProgram_ == nullptr ||
            batches_.empty()) {
            return makeError(Error::InvalidArgument);
        }

        Result result = updateParamsBuffer(context.width(), context.height());
        if (!result) {
            return result;
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
        constexpr bool kMaterialReversedZ = kDefaultReversedZ;
        RenderingAttachmentDesc depthAttachment{
            .view = depth.view(),
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearDepth = depthClearValue(kMaterialReversedZ),
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &depthAttachment,
        });
        context.commandBuffer().bindBindlessHeap(*bindlessHeap_);
        context.commandBuffer().bindGraphicsShaderObjectProgram(*defaultProgram_);
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        context.commandBuffer().setGraphicsShaderObjectState();
        context.commandBuffer().setDepthStencilState(DepthStencilState{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = depthCompareOp(kMaterialReversedZ),
        });

        const bool debugAlternateShaders =
            context.properties().value("debugAlternateShaders", false);
        GraphicsShaderObjectProgram* currentProgram = defaultProgram_.get();
        for (const MaterialShaderObjectBatch& batch : batches_) {
            GraphicsShaderObjectProgram* desiredProgram =
                debugAlternateShaders && ((batch.materialIndex & 1u) != 0)
                ? alternateProgram_.get()
                : defaultProgram_.get();
            if (desiredProgram != currentProgram) {
                context.commandBuffer().bindGraphicsShaderObjectProgram(*desiredProgram);
                currentProgram = desiredProgram;
            }

            const MaterialShaderObjectUserPush push{
                .positionBuffer = positionHandle_.index,
                .materialIndexBuffer = materialIndexHandle_.index,
                .materialBuffer = materialHandle_.index,
                .paramsBuffer = paramsHandle_.index,
                .vertexOffset = batch.firstVertex,
                .materialVariant = desiredProgram == alternateProgram_.get() ? 1u : 0u,
            };
            context.commandBuffer().pushBindlessData(&push, sizeof(push));
            context.commandBuffer().draw(batch.vertexCount);
        }

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

    static Result allocateAndWriteBuffer(
        BindlessHeap& heap,
        Buffer& buffer,
        BindlessHandle& outHandle,
        std::string& log,
        std::string_view label)
    {
        Result result = heap.allocateBuffer(outHandle);
        if (!result || !outHandle.valid()) {
            log += resultMessage(std::string("allocateBuffer(SceneMaterialShaderObjectPass ") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = heap.writeStorageBuffer(outHandle, buffer);
        if (!result) {
            log += resultMessage(std::string("writeStorageBuffer(SceneMaterialShaderObjectPass ") + std::string(label) + ")", result);
            log += '\n';
        }
        return result;
    }

    static Result createProgram(
        Device& device,
        const ShaderCompileResult& vertexCompile,
        const ShaderCompileResult& fragmentCompile,
        std::unique_ptr<GraphicsShaderObjectProgram>& outProgram,
        std::string& log,
        std::string_view label)
    {
        Result result = device.createGraphicsShaderObjectProgram(
            GraphicsShaderObjectProgramDesc{
                .vertexCode = vertexCompile.spirv.data(),
                .vertexByteSize = static_cast<uint64_t>(vertexCompile.spirv.size() * sizeof(uint32_t)),
                .fragmentCode = fragmentCompile.spirv.data(),
                .fragmentByteSize = static_cast<uint64_t>(fragmentCompile.spirv.size() * sizeof(uint32_t)),
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MaterialShaderObjectUserPush),
            },
            outProgram);
        if (!result || outProgram == nullptr) {
            log += resultMessage(std::string("createGraphicsShaderObjectProgram(SceneMaterialShaderObjectPass ") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
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
        return kDefaultMaterialScenePath;
    }

    static uint32_t materialIndexOrDefault(int32_t materialIndex, uint32_t materialCount)
    {
        if (materialCount == 0 ||
            materialIndex < 0 ||
            static_cast<uint32_t>(materialIndex) >= materialCount) {
            return 0;
        }
        return static_cast<uint32_t>(materialIndex);
    }

    static void appendTriangleVertex(
        const scene::RenderNode& renderNode,
        const scene::RenderPrimitive& primitive,
        uint32_t localIndex,
        std::vector<MaterialShaderObjectGpuPosition>& outPositions,
        scene::Bounds& outBounds)
    {
        if (static_cast<size_t>(localIndex) >= primitive.positions.size()) {
            return;
        }
        const float3 world = renderNode.worldMatrix * primitive.positions[static_cast<size_t>(localIndex)];
        outPositions.push_back(MaterialShaderObjectGpuPosition{
            .x = world.x,
            .y = world.y,
            .z = world.z,
            .w = 1.0f,
        });
        outBounds.include(world);
    }

    static bool loadSceneGeometry(
        const RenderGraphProperties& properties,
        std::vector<MaterialShaderObjectGpuPosition>& outPositions,
        std::vector<uint32_t>& outMaterialIndices,
        std::vector<MaterialShaderObjectGpuMaterial>& outMaterials,
        std::vector<MaterialShaderObjectBatch>& outBatches,
        scene::Bounds& outBounds,
        std::string& log)
    {
        scene::Scene loadedScene;
        const std::filesystem::path path = scenePathFromProperties(properties);
        if (!loadedScene.load(path)) {
            log = "SceneMaterialShaderObjectPass failed to load glTF: " + loadedScene.lastLoadResult().error;
            return false;
        }

        outMaterials.clear();
        if (loadedScene.materials().empty()) {
            outMaterials.push_back(MaterialShaderObjectGpuMaterial{});
        } else {
            outMaterials.reserve(loadedScene.materials().size());
            for (const scene::RenderMaterial& material : loadedScene.materials()) {
                outMaterials.push_back(MaterialShaderObjectGpuMaterial{
                    .baseColor = {
                        material.baseColorFactor.x,
                        material.baseColorFactor.y,
                        material.baseColorFactor.z,
                        material.baseColorFactor.w,
                    },
                });
            }
        }

        std::vector<std::vector<MaterialShaderObjectGpuPosition>> positionsByMaterial(outMaterials.size());
        outBounds.reset();
        for (const scene::RenderNode& renderNode : loadedScene.renderNodes()) {
            if (!renderNode.visible ||
                renderNode.renderPrimitiveIndex < 0 ||
                static_cast<size_t>(renderNode.renderPrimitiveIndex) >= loadedScene.renderPrimitives().size()) {
                continue;
            }

            const scene::RenderPrimitive& primitive =
                loadedScene.renderPrimitives()[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
            if (primitive.mode != kGltfTriangleListMode || primitive.positions.empty()) {
                continue;
            }

            const uint32_t materialIndex = materialIndexOrDefault(
                renderNode.materialIndex,
                static_cast<uint32_t>(outMaterials.size()));
            std::vector<MaterialShaderObjectGpuPosition>& materialPositions = positionsByMaterial[materialIndex];
            const std::vector<uint32_t>& indices = primitive.indices;
            if (!indices.empty()) {
                const size_t triangleIndexCount = indices.size() - (indices.size() % 3);
                for (size_t index = 0; index < triangleIndexCount; ++index) {
                    appendTriangleVertex(renderNode, primitive, indices[index], materialPositions, outBounds);
                }
            } else {
                const size_t triangleVertexCount = primitive.positions.size() - (primitive.positions.size() % 3);
                for (size_t index = 0; index < triangleVertexCount; ++index) {
                    appendTriangleVertex(
                        renderNode,
                        primitive,
                        static_cast<uint32_t>(index),
                        materialPositions,
                        outBounds);
                }
            }
        }

        outPositions.clear();
        outMaterialIndices.clear();
        outBatches.clear();
        for (uint32_t materialIndex = 0; materialIndex < positionsByMaterial.size(); ++materialIndex) {
            const std::vector<MaterialShaderObjectGpuPosition>& materialPositions = positionsByMaterial[materialIndex];
            if (materialPositions.empty()) {
                continue;
            }

            const uint32_t firstVertex = static_cast<uint32_t>(outPositions.size());
            outPositions.insert(outPositions.end(), materialPositions.begin(), materialPositions.end());
            outMaterialIndices.insert(outMaterialIndices.end(), materialPositions.size(), materialIndex);
            outBatches.push_back(MaterialShaderObjectBatch{
                .materialIndex = materialIndex,
                .firstVertex = firstVertex,
                .vertexCount = static_cast<uint32_t>(materialPositions.size()),
            });
        }

        if (outPositions.empty() || outMaterialIndices.size() != outPositions.size() || !outBounds.valid) {
            log = "SceneMaterialShaderObjectPass found no drawable triangle geometry in " + path.string();
            return false;
        }
        if (outPositions.size() > std::numeric_limits<uint32_t>::max()) {
            log = "SceneMaterialShaderObjectPass geometry is too large to draw";
            return false;
        }
        return true;
    }

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    static void buildParams(
        uint32_t width,
        uint32_t height,
        const scene::Bounds& drawBounds,
        MaterialShaderObjectGpuParams& outParams)
    {
        outParams = MaterialShaderObjectGpuParams{};
        const float3 center = drawBounds.center();
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovRadians = 60.0f * (kPi / 180.0f);
        const float distance = std::max(radius / std::tan(fovRadians * 0.5f), 0.1f) + radius;
        const float3 eye(center.x, center.y, center.z + distance);

        writeParamVec3(eye, outParams.eye, 0.0f);
        writeParamVec3(center, outParams.center, 0.0f);
        writeParamVec3(float3(0.0f, 1.0f, 0.0f), outParams.upProjection, 0.0f);
        outParams.viewport[0] = aspect;
        outParams.viewport[1] = static_cast<float>(width);
        outParams.viewport[2] = static_cast<float>(height);
        outParams.viewport[3] = fovRadians;
        outParams.clipOrtho[0] = 0.001f;
        outParams.clipOrtho[1] = std::max(distance + radius * 3.0f, 1.0f);
        outParams.clipOrtho[2] = radius * 2.0f;
        outParams.clipOrtho[3] = kDefaultReversedZ ? 1.0f : 0.0f;
    }

    Result updateParamsBuffer(uint32_t width, uint32_t height)
    {
        if (paramsBuffer_ == nullptr || !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        MaterialShaderObjectGpuParams params;
        buildParams(width, height, drawBounds_, params);

        void* mapped = paramsBuffer_->map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, &params, sizeof(params));
        paramsBuffer_->flush(0, sizeof(params));
        paramsBuffer_->unmap();
        return {};
    }

    std::unique_ptr<Buffer> positionBuffer_;
    std::unique_ptr<Buffer> materialIndexBuffer_;
    std::unique_ptr<Buffer> materialBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle positionHandle_;
    BindlessHandle materialIndexHandle_;
    BindlessHandle materialHandle_;
    BindlessHandle paramsHandle_;
    std::unique_ptr<GraphicsShaderObjectProgram> defaultProgram_;
    std::unique_ptr<GraphicsShaderObjectProgram> alternateProgram_;
    std::vector<MaterialShaderObjectBatch> batches_;
    scene::Bounds drawBounds_;
};

class SceneRayQueryVisualizationPass final : public ComputePass {
public:
    ~SceneRayQueryVisualizationPass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "RayQuery acceleration-structure visualization")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "SceneRayQueryVisualizationPass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().rayTracingAccelerationStructure ||
            !context.device->capabilities().rayQuery) {
            log = "SceneRayQueryVisualizationPass requires rayTracingAccelerationStructure and rayQuery capabilities";
            return makeError(Error::Unsupported);
        }
        if (rayQueryProgram_.valid() && rtxBuilder_.valid()) {
            return {};
        }

        scene::Scene loadedScene;
        const std::filesystem::path path = scenePathFromProperties(properties());
        if (!loadedScene.load(path)) {
            log = "SceneRayQueryVisualizationPass failed to load glTF: " + loadedScene.lastLoadResult().error;
            return makeError(Error::Failure);
        }
        if (!loadedScene.bounds().valid) {
            log = "SceneRayQueryVisualizationPass scene bounds are unavailable";
            return makeError(Error::Failure);
        }

        drawBounds_ = loadedScene.bounds();
        Result result = rtxBuilder_.build(*context.device, *context.graphicsQueue, loadedScene, log);
        if (!result) {
            return result;
        }

        ShaderCompileResult computeCompile;
        const char* capabilities[] = {"spvRayQueryKHR"};
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kSceneRayQueryVisualizationShaderModuleName,
                .entryPointName = kSceneRayQueryVisualizationEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .profileName = "glsl_460",
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            computeCompile);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += kSceneRayQueryVisualizationShaderModuleName;
            log += ".";
            log += kSceneRayQueryVisualizationEntryPoint;
            log += ") returned ";
            log += resultToString(result);
            if (!computeCompile.diagnostics.empty()) {
                log += ": ";
                log += computeCompile.diagnostics;
            }
            log += '\n';
            return result;
        }

        const SceneRayQueryBindingDesc bindings[] = {
            SceneRayQueryBindingDesc{
                .binding = 0,
                .kind = SceneRayQueryBindingKind::AccelerationStructure,
            },
            SceneRayQueryBindingDesc{
                .binding = 1,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
        };
        result = rayQueryProgram_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = computeCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(computeCompile.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(SceneRayQueryVisualizationPush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "SceneRayQueryVisualizationPass",
            },
            log);
        if (!result) {
            rayQueryProgram_.clear();
            return result;
        }

        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            color.view() == nullptr ||
            !rayQueryProgram_.valid() ||
            !rtxBuilder_.valid() ||
            !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        SceneRayQueryVisualizationPush push;
        buildPush(context.width(), context.height(), context.properties(), drawBounds_, push);

        const SceneRayQueryDispatchBinding bindings[] = {
            SceneRayQueryDispatchBinding{
                .binding = 0,
                .accelerationStructure = &rtxBuilder_,
            },
            SceneRayQueryDispatchBinding{
                .binding = 1,
                .textureView = color.view(),
            },
        };
        return rayQueryProgram_.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
            .groupCountZ = 1,
        });
    }

private:
    static std::filesystem::path scenePathFromProperties(const RenderGraphProperties& props)
    {
        if (props.contains("path") && props["path"].is_string()) {
            std::filesystem::path path = props["path"].get<std::string>();
            if (path.is_relative()) {
                path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
            }
            return path;
        }
        return kDefaultMaterialScenePath;
    }

    static float finiteOr(float value, float fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    static const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return nullptr;
        }
        auto iter = properties.find("camera");
        if (iter == properties.end() || !iter->is_object()) {
            return nullptr;
        }
        return &(*iter);
    }

    static float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_number()) {
            return fallback;
        }
        return finiteOr(iter->get<float>(), fallback);
    }

    static float3 cameraVec3(
        const RenderGraphProperties* camera,
        const char* key,
        const float3& fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_array() || iter->size() < 3) {
            return fallback;
        }

        float values[3] = {fallback.x, fallback.y, fallback.z};
        for (size_t index = 0; index < 3; ++index) {
            const RenderGraphProperties& component = (*iter)[index];
            if (component.is_number()) {
                values[index] = finiteOr(component.get<float>(), values[index]);
            }
        }
        return float3(values[0], values[1], values[2]);
    }

    static bool cameraIsOrthographic(const RenderGraphProperties* camera)
    {
        if (camera == nullptr) {
            return false;
        }
        auto iter = camera->find("projection");
        if (iter == camera->end() || !iter->is_string()) {
            return false;
        }
        const std::string projection = iter->get<std::string>();
        return projection == "orthographic" || projection == "ortho";
    }

    static uint32_t visualizationModeFromProperties(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return kRayQueryVisualizationGranularityInstance;
        }
        auto iter = properties.find("granularity");
        if (iter == properties.end() || !iter->is_string()) {
            return kRayQueryVisualizationGranularityInstance;
        }
        const std::string value = iter->get<std::string>();
        return value == "primitive" || value == "per primitive" || value == "perPrimitive"
            ? kRayQueryVisualizationGranularityPrimitive
            : kRayQueryVisualizationGranularityInstance;
    }

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    static void buildPush(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const scene::Bounds& drawBounds,
        SceneRayQueryVisualizationPush& outPush)
    {
        outPush = SceneRayQueryVisualizationPush{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 60.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y, center.z + defaultDistance);
        const float3 eye = cameraVec3(cameraProperties, "eye", defaultEye);
        const float3 target = cameraVec3(cameraProperties, "center", center);
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.001f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(cameraProperties, "zfar", defaultDistance + radius * 3.0f),
            zNear + 0.001f);
        const float cameraDistance = std::max(length(eye - target), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(
            cameraFloat(cameraProperties, "orthoHeight", defaultOrthoHeight),
            0.0001f);

        writeParamVec3(eye, outPush.eye, 0.0f);
        writeParamVec3(target, outPush.center, 0.0f);
        writeParamVec3(up, outPush.upProjection, cameraIsOrthographic(cameraProperties) ? 1.0f : 0.0f);
        outPush.viewport[0] = aspect;
        outPush.viewport[1] = static_cast<float>(width);
        outPush.viewport[2] = static_cast<float>(height);
        outPush.viewport[3] = fovRadians;
        outPush.clipOrtho[0] = zNear;
        outPush.clipOrtho[1] = zFar;
        outPush.clipOrtho[2] = orthoHeight;
        outPush.clipOrtho[3] = 0.0f;
        outPush.mode = visualizationModeFromProperties(properties);
        outPush.width = width;
        outPush.height = height;
    }

    SceneRtxBuilder rtxBuilder_;
    SceneRayQueryProgram rayQueryProgram_;
    scene::Bounds drawBounds_;
};

class ScenePathTracePass final : public ComputePass {
public:
    ~ScenePathTracePass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Path-traced glTF scene")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "ScenePathTracePass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().rayTracingAccelerationStructure ||
            !context.device->capabilities().rayQuery) {
            log = "ScenePathTracePass requires rayTracingAccelerationStructure and rayQuery capabilities";
            return makeError(Error::Unsupported);
        }

        Result result = sceneResources_.prepare(*context.device, *context.graphicsQueue, properties(), log);
        if (!result) {
            return result;
        }
        const uint64_t resourceRevision = sceneResources_.revision();
        if (resourceRevision != sceneResourceRevision_) {
            sceneResourceRevision_ = resourceRevision;
            resetAccumulation_ = true;
        }
        if (rayQueryProgram_.valid()) {
            return {};
        }

        ShaderCompileResult computeCompile;
        const char* capabilities[] = {"spvRayQueryKHR"};
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kScenePathTraceShaderModuleName,
                .entryPointName = kScenePathTraceEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .profileName = "glsl_460",
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            computeCompile);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += kScenePathTraceShaderModuleName;
            log += ".";
            log += kScenePathTraceEntryPoint;
            log += ") returned ";
            log += resultToString(result);
            if (!computeCompile.diagnostics.empty()) {
                log += ": ";
                log += computeCompile.diagnostics;
            }
            log += '\n';
            rayQueryProgram_.clear();
            return result;
        }

        const SceneRayQueryBindingDesc bindings[] = {
            SceneRayQueryBindingDesc{
                .binding = 0,
                .kind = SceneRayQueryBindingKind::AccelerationStructure,
            },
            SceneRayQueryBindingDesc{
                .binding = 1,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 2,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 3,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 4,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 5,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 6,
                .kind = SceneRayQueryBindingKind::StorageBuffer,
            },
            SceneRayQueryBindingDesc{
                .binding = 7,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 8,
                .kind = SceneRayQueryBindingKind::StorageImage,
            },
            SceneRayQueryBindingDesc{
                .binding = 9,
                .kind = SceneRayQueryBindingKind::SampledImage,
                .descriptorCount = kScenePathTraceMaxMaterialTextures,
            },
        };
        std::string programLog;
        result = rayQueryProgram_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = computeCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(computeCompile.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(ScenePathTracePush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "ScenePathTracePass",
            },
            programLog);
        if (!programLog.empty()) {
            if (!log.empty() && log.back() != '\n') {
                log += '\n';
            }
            log += programLog;
        }
        if (!result) {
            rayQueryProgram_.clear();
            return result;
        }
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        const auto& materialTextureViews = sceneResources_.materialTextureViews();
        if (!color.valid() ||
            color.view() == nullptr ||
            !rayQueryProgram_.valid() ||
            !sceneResources_.valid() ||
            materialTextureViews[0] == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        ScenePathTracePush push;
        buildPush(context.width(), context.height(), context.properties(), sceneResources_.bounds(), push);
        push.materialTextureCount = sceneResources_.materialTextureCount();

        TextureView* historyCurrentView = color.view();
        TextureView* historyPreviousView = color.view();
        Result result = prepareHistoryTextures(
            context,
            *color.view(),
            push,
            historyCurrentView,
            historyPreviousView);
        if (!result) {
            return result;
        }
        if (historyCurrentView == nullptr || historyPreviousView == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        result = sceneResources_.uploadMaterialTextures(context.commandBuffer());
        if (!result) {
            return result;
        }

        const SceneRayQueryDispatchBinding bindings[] = {
            SceneRayQueryDispatchBinding{
                .binding = 0,
                .accelerationStructure = &sceneResources_.accelerationStructure(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 1,
                .textureView = color.view(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 2,
                .buffer = sceneResources_.vertexBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 3,
                .buffer = sceneResources_.indexBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 4,
                .buffer = sceneResources_.primitiveBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 5,
                .buffer = sceneResources_.instanceBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 6,
                .buffer = sceneResources_.materialBuffer(),
            },
            SceneRayQueryDispatchBinding{
                .binding = 7,
                .textureView = historyCurrentView,
            },
            SceneRayQueryDispatchBinding{
                .binding = 8,
                .textureView = historyPreviousView,
            },
            SceneRayQueryDispatchBinding{
                .binding = 9,
                .textureViews = materialTextureViews.data(),
                .textureViewCount = static_cast<uint32_t>(materialTextureViews.size()),
            },
        };
        result = rayQueryProgram_.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
            .groupCountZ = 1,
        });
        if (!result) {
            return result;
        }
        if (push.enableAccumulation != 0 && context.historyResources() != nullptr) {
            context.historyResources()->markWritten(historyNameForContext(context));
        }
        return {};
    }

private:
    Result prepareHistoryTextures(
        RenderGraphExecutionContext& context,
        TextureView& fallbackView,
        ScenePathTracePush& push,
        TextureView*& outCurrentView,
        TextureView*& outPreviousView)
    {
        HistoryResourceManager* history = context.historyResources();
        const bool accumulationEnabled = boolProperty(context.properties(), "accumulate", true);
        push.enableAccumulation = accumulationEnabled && history != nullptr ? 1u : 0u;
        push.hasHistory = 0;
        push.accumulationFrame = 0;
        outCurrentView = &fallbackView;
        outPreviousView = &fallbackView;
        if (push.enableAccumulation == 0) {
            accumulationFrame_ = 0;
            resetAccumulation_ = true;
            return {};
        }

        const TextureDesc historyDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled |
                TextureUsageBits::Storage |
                TextureUsageBits::TransferSource,
            .format = Format::Rgba8Unorm,
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        const std::string historyName = historyNameForContext(context);
        Result result = history->ensureTexture(
            historyName,
            historyDesc,
            TextureViewDesc{.format = Format::Rgba8Unorm});
        if (!result) {
            return result;
        }

        HistoryTextureRef current = history->texture(historyName, HistorySlot::Current);
        HistoryTextureRef previous = history->texture(historyName, HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr || previous.texture == nullptr || previous.view == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        result = history->transitionTexture(
            context.commandBuffer(),
            historyName,
            HistorySlot::Current,
            ResourceState::General);
        if (!result) {
            return result;
        }
        result = history->transitionTexture(
            context.commandBuffer(),
            historyName,
            HistorySlot::Previous,
            ResourceState::General);
        if (!result) {
            return result;
        }

        outCurrentView = current.view;
        outPreviousView = previous.view;

        if (previous.valid && !resetAccumulation_) {
            ++accumulationFrame_;
            push.hasHistory = 1;
        } else {
            accumulationFrame_ = 0;
            push.hasHistory = 0;
        }
        resetAccumulation_ = false;
        push.accumulationFrame = accumulationFrame_;
        return {};
    }

    static uint32_t uintProperty(
        const RenderGraphProperties& properties,
        const char* key,
        uint32_t fallback,
        uint32_t minimum,
        uint32_t maximum)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        uint32_t value = fallback;
        if (iter->is_number_unsigned()) {
            value = iter->get<uint32_t>();
        } else if (iter->is_number_integer()) {
            const int64_t signedValue = iter->get<int64_t>();
            value = signedValue > 0 ? static_cast<uint32_t>(signedValue) : minimum;
        } else {
            value = static_cast<uint32_t>(std::max(iter->get<float>(), static_cast<float>(minimum)));
        }
        return std::clamp(value, minimum, maximum);
    }

    static bool boolProperty(const RenderGraphProperties& properties, const char* key, bool fallback)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_boolean()) {
            return fallback;
        }
        return iter->get<bool>();
    }

    static std::string historyNameForContext(const RenderGraphExecutionContext& context)
    {
        std::string name(kScenePathTraceHistoryPrefix);
        name += context.passName();
        name += ".accumulation";
        return name;
    }

    static float finiteOr(float value, float fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    static const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return nullptr;
        }
        auto iter = properties.find("camera");
        if (iter == properties.end() || !iter->is_object()) {
            return nullptr;
        }
        return &(*iter);
    }

    static float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_number()) {
            return fallback;
        }
        return finiteOr(iter->get<float>(), fallback);
    }

    static float3 cameraVec3(
        const RenderGraphProperties* camera,
        const char* key,
        const float3& fallback)
    {
        if (camera == nullptr) {
            return fallback;
        }
        auto iter = camera->find(key);
        if (iter == camera->end() || !iter->is_array() || iter->size() < 3) {
            return fallback;
        }

        float values[3] = {fallback.x, fallback.y, fallback.z};
        for (size_t index = 0; index < 3; ++index) {
            const RenderGraphProperties& component = (*iter)[index];
            if (component.is_number()) {
                values[index] = finiteOr(component.get<float>(), values[index]);
            }
        }
        return float3(values[0], values[1], values[2]);
    }

    static bool cameraIsOrthographic(const RenderGraphProperties* camera)
    {
        if (camera == nullptr) {
            return false;
        }
        auto iter = camera->find("projection");
        if (iter == camera->end() || !iter->is_string()) {
            return false;
        }
        const std::string projection = iter->get<std::string>();
        return projection == "orthographic" || projection == "ortho";
    }

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    static void buildPush(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const scene::Bounds& drawBounds,
        ScenePathTracePush& outPush)
    {
        outPush = ScenePathTracePush{};
        const float3 center = drawBounds.center();
        const float3 halfExtent = (drawBounds.max - drawBounds.min) * 0.5f;
        const float radius = std::max(drawBounds.radius(), 0.01f);
        const float aspect = height == 0 ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        const float frameHalfHeight = std::max(halfExtent.y, halfExtent.x / std::max(aspect, 0.001f));
        const RenderGraphProperties* cameraProperties = cameraPropertiesFrom(properties);
        constexpr float kPi = 3.14159265358979323846f;
        const float fovDegrees = std::clamp(
            cameraFloat(cameraProperties, "fovDegrees", 50.0f),
            1.0f,
            179.0f);
        const float fovRadians = fovDegrees * (kPi / 180.0f);
        const float defaultDistance = std::max(
            frameHalfHeight > 0.000001f
                ? frameHalfHeight / (0.72f * std::tan(fovRadians * 0.5f)) + radius
                : radius * 2.5f,
            0.05f);
        const float3 defaultEye(center.x, center.y + radius * 0.12f, center.z + defaultDistance);
        const float3 eye = cameraVec3(cameraProperties, "eye", defaultEye);
        const float3 target = cameraVec3(cameraProperties, "center", center);
        const float3 up = cameraVec3(cameraProperties, "up", float3(0.0f, 1.0f, 0.0f));
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.001f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(cameraProperties, "zfar", defaultDistance + radius * 4.0f),
            zNear + 0.001f);
        const float cameraDistance = std::max(length(eye - target), 0.001f);
        const float defaultOrthoHeight = std::max(2.0f * cameraDistance * std::tan(fovRadians * 0.5f), 0.0001f);
        const float orthoHeight = std::max(
            cameraFloat(cameraProperties, "orthoHeight", defaultOrthoHeight),
            0.0001f);

        writeParamVec3(eye, outPush.eye, 0.0f);
        writeParamVec3(target, outPush.center, 0.0f);
        writeParamVec3(up, outPush.upProjection, cameraIsOrthographic(cameraProperties) ? 1.0f : 0.0f);
        outPush.viewport[0] = aspect;
        outPush.viewport[1] = static_cast<float>(width);
        outPush.viewport[2] = static_cast<float>(height);
        outPush.viewport[3] = fovRadians;
        outPush.clipOrtho[0] = zNear;
        outPush.clipOrtho[1] = zFar;
        outPush.clipOrtho[2] = orthoHeight;
        outPush.clipOrtho[3] = 0.0f;
        outPush.width = width;
        outPush.height = height;
        outPush.maxDepth = uintProperty(properties, "maxDepth", kDefaultPathTraceMaxDepth, 1, 12);
        outPush.samples = uintProperty(properties, "samples", kDefaultPathTraceSamples, 1, 16);
    }

    ScenePathTraceResources sceneResources_;
    SceneRayQueryProgram rayQueryProgram_;
    uint64_t sceneResourceRevision_ = 0;
    uint32_t accumulationFrame_ = 0;
    bool resetAccumulation_ = false;
};

class NrdDenoisePass final : public ComputePass {
public:
    ~NrdDenoisePass() override = default;

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("noisyDiffuse", "Noisy diffuse radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("noisySpecular", "Noisy specular radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("normalRoughness", "NRD packed normal and roughness")
            .storageReadWrite()
            .format = vulkan::nrdNormalRoughnessFormat();
        reflection.addTextureInput("motionVectors", "NRD motion vectors")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("viewZ", "NRD linear view depth")
            .storageReadWrite()
            .format = Format::R16Sfloat;
        reflection.addTextureInput("baseColorMetalness", "Base color and metalness")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        reflection.addTextureOutput("denoisedDiffuse", "NRD denoised diffuse radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("denoisedSpecular", "NRD denoised specular radiance and hit distance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("validation", "NRD validation/debug output")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
#if !METALLIC_HAS_NRD
        log = "NrdDenoisePass requires the NRD SDK target";
        return makeError(Error::Unsupported);
#else
        if (context.device == nullptr || context.graphicsQueue == nullptr) {
            log = "NrdDenoisePass requires a device and graphics queue";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().pushDescriptor) {
            log = "NrdDenoisePass requires VK_KHR_push_descriptor";
            return makeError(Error::Unsupported);
        }

        device_ = context.device;
        queue_ = context.graphicsQueue;
        return {};
#endif
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
#if !METALLIC_HAS_NRD
        (void)context;
        return makeError(Error::Unsupported);
#else
        TextureHandle noisyDiffuse = context.inputTexture("noisyDiffuse");
        TextureHandle noisySpecular = context.inputTexture("noisySpecular");
        TextureHandle normalRoughness = context.inputTexture("normalRoughness");
        TextureHandle motionVectors = context.inputTexture("motionVectors");
        TextureHandle viewZ = context.inputTexture("viewZ");
        TextureHandle baseColorMetalness = context.inputTexture("baseColorMetalness");
        TextureHandle denoisedDiffuse = context.outputTexture("denoisedDiffuse");
        TextureHandle denoisedSpecular = context.outputTexture("denoisedSpecular");
        TextureHandle validation = context.outputTexture("validation");

        if (!validTexture(noisyDiffuse) ||
            !validTexture(noisySpecular) ||
            !validTexture(normalRoughness) ||
            !validTexture(motionVectors) ||
            !validTexture(viewZ) ||
            !validTexture(baseColorMetalness) ||
            !validTexture(denoisedDiffuse) ||
            !validTexture(denoisedSpecular) ||
            !validTexture(validation) ||
            device_ == nullptr ||
            queue_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const uint32_t denoiserMode = denoiserModeFromProperties(context.properties());
        const uint32_t resetSerial = uintProperty(
            context.properties(),
            "resetSerial",
            0,
            0,
            std::numeric_limits<uint32_t>::max());
        if (lastDenoiserMode_ != denoiserMode || lastResetSerial_ != resetSerial) {
            frameIndex_ = 0;
            lastDenoiserMode_ = denoiserMode;
            lastResetSerial_ = resetSerial;
        }

        Result result = ensureNrd(
            context,
            noisyDiffuse,
            noisySpecular,
            normalRoughness,
            motionVectors,
            viewZ,
            baseColorMetalness,
            denoisedDiffuse,
            denoisedSpecular,
            validation);
        if (!result) {
            return result;
        }

        result = runNrd(context, denoiserMode, noisyDiffuse, noisySpecular, denoisedDiffuse, denoisedSpecular);
        if (result) {
            ++frameIndex_;
        }
        return result;
#endif
    }

private:
    static bool validTexture(TextureHandle texture)
    {
        return texture.valid() && texture.texture() != nullptr && texture.view() != nullptr;
    }

    static uint32_t uintProperty(
        const RenderGraphProperties& properties,
        const char* key,
        uint32_t fallback,
        uint32_t minimum,
        uint32_t maximum)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        uint32_t value = fallback;
        if (iter->is_number_unsigned()) {
            value = iter->get<uint32_t>();
        } else if (iter->is_number_integer()) {
            const int64_t signedValue = iter->get<int64_t>();
            value = signedValue > 0 ? static_cast<uint32_t>(signedValue) : minimum;
        } else {
            value = static_cast<uint32_t>(std::max(iter->get<float>(), static_cast<float>(minimum)));
        }
        return std::clamp(value, minimum, maximum);
    }

    static float floatProperty(
        const RenderGraphProperties& properties,
        const char* key,
        float fallback,
        float minimum,
        float maximum)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        const float value = iter->get<float>();
        if (!std::isfinite(value)) {
            return fallback;
        }
        return std::clamp(value, minimum, maximum);
    }

    static uint32_t denoiserModeFromProperties(const RenderGraphProperties& properties)
    {
        if (properties.is_object()) {
            auto iter = properties.find("denoiser");
            if (iter != properties.end() && iter->is_string()) {
                const std::string value = iter->get<std::string>();
                if (value == "RELAX" || value == "Relax" || value == "relax") {
                    return kNrdDenoiserModeRelax;
                }
                if (value == "REFERENCE" || value == "Reference" || value == "reference") {
                    return kNrdDenoiserModeReference;
                }
            }
        }
        return kNrdDenoiserModeReblur;
    }

#if METALLIC_HAS_NRD
    static void setIdentity(float matrix[16])
    {
        for (uint32_t index = 0; index < 16; ++index) {
            matrix[index] = 0.0f;
        }
        matrix[0] = 1.0f;
        matrix[5] = 1.0f;
        matrix[10] = 1.0f;
        matrix[15] = 1.0f;
    }

    static vulkan::NrdDenoiserMode wrapperMode(uint32_t denoiserMode)
    {
        if (denoiserMode == kNrdDenoiserModeRelax) {
            return vulkan::NrdDenoiserMode::Relax;
        }
        if (denoiserMode == kNrdDenoiserModeReference) {
            return vulkan::NrdDenoiserMode::Reference;
        }
        return vulkan::NrdDenoiserMode::Reblur;
    }

    Result ensureNrd(
        RenderGraphExecutionContext& context,
        TextureHandle noisyDiffuse,
        TextureHandle noisySpecular,
        TextureHandle normalRoughness,
        TextureHandle motionVectors,
        TextureHandle viewZ,
        TextureHandle baseColorMetalness,
        TextureHandle denoisedDiffuse,
        TextureHandle denoisedSpecular,
        TextureHandle validation)
    {
        if (context.width() > std::numeric_limits<uint16_t>::max() ||
            context.height() > std::numeric_limits<uint16_t>::max()) {
            return makeError(Error::InvalidArgument);
        }

        vulkan::NrdUserTexturePool pool{};
        auto put = [&pool](nrd::ResourceType resource, TextureHandle texture) {
            pool[static_cast<size_t>(resource)] = vulkan::NrdTextureRef{
                .texture = texture.texture(),
                .view = texture.view(),
            };
        };
        put(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, noisyDiffuse);
        put(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, noisySpecular);
        put(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, denoisedDiffuse);
        put(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, denoisedSpecular);
        put(nrd::ResourceType::IN_NORMAL_ROUGHNESS, normalRoughness);
        put(nrd::ResourceType::IN_MV, motionVectors);
        put(nrd::ResourceType::IN_VIEWZ, viewZ);
        put(nrd::ResourceType::IN_BASECOLOR_METALNESS, baseColorMetalness);
        put(nrd::ResourceType::OUT_VALIDATION, validation);
        put(nrd::ResourceType::IN_SIGNAL, noisyDiffuse);
        put(nrd::ResourceType::OUT_SIGNAL, denoisedDiffuse);

        const uint16_t width = static_cast<uint16_t>(context.width());
        const uint16_t height = static_cast<uint16_t>(context.height());
        const bool sizeChanged = nrd_ == nullptr ||
            !nrd_->valid() ||
            nrd_->width() != width ||
            nrd_->height() != height;
        if (sizeChanged) {
            nrd_ = std::make_unique<vulkan::NrdDenoiser>();
            std::string log;
            Result result = nrd_->initialize(*device_, *queue_, width, height, pool, log);
            if (!result) {
                nrd_.reset();
                return result;
            }
            frameIndex_ = 0;
            return {};
        }

        nrd_->setUserPoolTexture(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, *noisyDiffuse.texture(), *noisyDiffuse.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, *noisySpecular.texture(), *noisySpecular.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, *denoisedDiffuse.texture(), *denoisedDiffuse.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, *denoisedSpecular.texture(), *denoisedSpecular.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_NORMAL_ROUGHNESS, *normalRoughness.texture(), *normalRoughness.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_MV, *motionVectors.texture(), *motionVectors.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_VIEWZ, *viewZ.texture(), *viewZ.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::IN_BASECOLOR_METALNESS, *baseColorMetalness.texture(), *baseColorMetalness.view());
        nrd_->setUserPoolTexture(nrd::ResourceType::OUT_VALIDATION, *validation.texture(), *validation.view());
        return {};
    }

    Result runNrd(
        RenderGraphExecutionContext& context,
        uint32_t denoiserMode,
        TextureHandle noisyDiffuse,
        TextureHandle noisySpecular,
        TextureHandle denoisedDiffuse,
        TextureHandle denoisedSpecular)
    {
        if (nrd_ == nullptr || !nrd_->valid()) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphProperties& properties = context.properties();
        nrd::CommonSettings commonSettings;
        setIdentity(commonSettings.viewToClipMatrix);
        setIdentity(commonSettings.viewToClipMatrixPrev);
        setIdentity(commonSettings.worldToViewMatrix);
        setIdentity(commonSettings.worldToViewMatrixPrev);
        commonSettings.motionVectorScale[0] = floatProperty(properties, "motionVectorScaleX", 1.0f, -65504.0f, 65504.0f);
        commonSettings.motionVectorScale[1] = floatProperty(properties, "motionVectorScaleY", 1.0f, -65504.0f, 65504.0f);
        commonSettings.motionVectorScale[2] = 0.0f;
        commonSettings.resourceSize[0] = static_cast<uint16_t>(context.width());
        commonSettings.resourceSize[1] = static_cast<uint16_t>(context.height());
        commonSettings.resourceSizePrev[0] = static_cast<uint16_t>(context.width());
        commonSettings.resourceSizePrev[1] = static_cast<uint16_t>(context.height());
        commonSettings.rectSize[0] = static_cast<uint16_t>(context.width());
        commonSettings.rectSize[1] = static_cast<uint16_t>(context.height());
        commonSettings.rectSizePrev[0] = static_cast<uint16_t>(context.width());
        commonSettings.rectSizePrev[1] = static_cast<uint16_t>(context.height());
        commonSettings.frameIndex = frameIndex_;
        commonSettings.timeDeltaBetweenFrames = floatProperty(properties, "timeDeltaSeconds", 1.0f / 60.0f, 0.0f, 1.0f);
        commonSettings.denoisingRange = floatProperty(properties, "denoisingRange", 10000.0f, 0.0f, 1000000.0f);
        commonSettings.accumulationMode = frameIndex_ == 0
            ? nrd::AccumulationMode::CLEAR_AND_RESTART
            : nrd::AccumulationMode::CONTINUE;
        commonSettings.isMotionVectorInWorldSpace = boolProperty(&properties, "motionVectorInWorldSpace", false);
        commonSettings.isBaseColorMetalnessAvailable = true;
        commonSettings.enableValidation = boolProperty(&properties, "enableValidation", true);

        Result result = nrd_->setCommonSettings(commonSettings);
        if (!result) {
            return result;
        }

        if (denoiserMode == kNrdDenoiserModeReference) {
            nrd_->setUserPoolTexture(nrd::ResourceType::IN_SIGNAL, *noisyDiffuse.texture(), *noisyDiffuse.view());
            nrd_->setUserPoolTexture(nrd::ResourceType::OUT_SIGNAL, *denoisedDiffuse.texture(), *denoisedDiffuse.view());
            nrd::Identifier referenceDiffuse = static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE);
            result = nrd_->denoiseIdentifiers(&referenceDiffuse, 1, context.commandBuffer());
            if (!result) {
                return result;
            }

            nrd_->setUserPoolTexture(nrd::ResourceType::IN_SIGNAL, *noisySpecular.texture(), *noisySpecular.view());
            nrd_->setUserPoolTexture(nrd::ResourceType::OUT_SIGNAL, *denoisedSpecular.texture(), *denoisedSpecular.view());
            nrd::Identifier referenceSpecular = static_cast<nrd::Identifier>(nrd::Denoiser::REFERENCE) + 1;
            return nrd_->denoiseIdentifiers(&referenceSpecular, 1, context.commandBuffer());
        }

        if (denoiserMode == kNrdDenoiserModeRelax) {
            nrd::RelaxSettings relaxSettings;
            result = nrd_->setRelaxSettings(relaxSettings);
            if (!result) {
                return result;
            }
        } else {
            nrd::ReblurSettings reblurSettings;
            result = nrd_->setReblurSettings(reblurSettings);
            if (!result) {
                return result;
            }
        }
        return nrd_->denoise(wrapperMode(denoiserMode), context.commandBuffer());
    }
#endif

    Device* device_ = nullptr;
    Queue* queue_ = nullptr;
    uint32_t frameIndex_ = 0;
    uint32_t lastDenoiserMode_ = std::numeric_limits<uint32_t>::max();
    uint32_t lastResetSerial_ = 0;
#if METALLIC_HAS_NRD
    std::unique_ptr<vulkan::NrdDenoiser> nrd_;
#endif
};

class RenderGraphBufferWritePass final : public ComputePass {
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

class RenderGraphBufferCopyPass final : public ComputePass {
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
        "SceneMaterialShaderObjectPass",
        "Draw glTF material colors with VK_EXT_shader_object",
        []() { return std::make_unique<SceneMaterialShaderObjectPass>(); });
    registerRenderGraphPassType(
        "SceneRayQueryVisualizationPass",
        "Visualize a glTF acceleration structure with RayQuery",
        []() { return std::make_unique<SceneRayQueryVisualizationPass>(); });
    registerRenderGraphPassType(
        "ScenePathTracePass",
        "Path trace a glTF scene with RayQuery",
        []() { return std::make_unique<ScenePathTracePass>(); });
    registerRenderGraphPassType(
        "NrdDenoisePass",
        "Denoise connected NRD radiance resources",
        []() { return std::make_unique<NrdDenoisePass>(); });
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
