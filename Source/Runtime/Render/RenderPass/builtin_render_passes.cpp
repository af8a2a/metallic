#include "Runtime/Render/RenderGraph/render_graph.h"

#include "Runtime/Render/GAPI/Vulkan/vulkan_native.h"
#include "Runtime/Render/GAPI/Vulkan/vulkan_scene_rtx.h"
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
constexpr const char* kDefaultPathTraceScenePath = PROJECT_SOURCE_DIR "/Asset/meet_mat.glb";
constexpr uint64_t kRenderGraphBufferByteSize = 16;
constexpr int32_t kGltfTriangleListMode = 4;
constexpr uint32_t kRayQueryVisualizationGranularityInstance = 0;
constexpr uint32_t kRayQueryVisualizationGranularityPrimitive = 1;
constexpr uint32_t kDefaultPathTraceMaxDepth = 3;
constexpr uint32_t kDefaultPathTraceSamples = 2;
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

struct ScenePathTraceGpuVertex {
    float position[4] = {};
    float normal[4] = {};
    float texcoord[4] = {};
};

struct ScenePathTraceGpuPrimitive {
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
};

struct ScenePathTraceGpuInstance {
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t flags = 0;
    uint32_t padding = 0;
};

struct ScenePathTraceGpuMaterial {
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float emissive[4] = {};
    float params[4] = {};
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
    uint32_t padding = 0;
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
    ~SceneRayQueryVisualizationPass() override
    {
        destroyNative();
    }

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
        if (pipeline_ != VK_NULL_HANDLE && rtxBuilder_.valid()) {
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

        nativeDevice_ = vulkan::nativeDevice(*context.device).device;
        if (nativeDevice_ == VK_NULL_HANDLE) {
            log = "SceneRayQueryVisualizationPass Vulkan device is unavailable";
            return makeError(Error::InvalidArgument);
        }
        volkLoadDevice(nativeDevice_);

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

        result = createNativePipeline(computeCompile, log);
        if (!result) {
            destroyNative();
            return result;
        }

        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            nativeDevice_ == VK_NULL_HANDLE ||
            descriptorSet_ == VK_NULL_HANDLE ||
            pipeline_ == VK_NULL_HANDLE ||
            pipelineLayout_ == VK_NULL_HANDLE ||
            !rtxBuilder_.valid() ||
            !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        VkImageView outputView = vulkan::nativeImageView(*color.view());
        if (outputView == VK_NULL_HANDLE) {
            return makeError(Error::InvalidArgument);
        }
        updateDescriptorSet(outputView);

        SceneRayQueryVisualizationPush push;
        buildPush(context.width(), context.height(), context.properties(), drawBounds_, push);

        VkCommandBuffer commandBuffer = vulkan::nativeCommandBuffer(context.commandBuffer());
        if (commandBuffer == VK_NULL_HANDLE) {
            return makeError(Error::InvalidArgument);
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout_,
            0,
            1,
            &descriptorSet_,
            0,
            nullptr);
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(push),
            &push);
        vkCmdDispatch(commandBuffer, (context.width() + 7) / 8, (context.height() + 7) / 8, 1);
        return {};
    }

private:
    void destroyNative()
    {
        if (nativeDevice_ == VK_NULL_HANDLE) {
            return;
        }
        volkLoadDevice(nativeDevice_);
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(nativeDevice_, pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(nativeDevice_, pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (shaderModule_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(nativeDevice_, shaderModule_, nullptr);
            shaderModule_ = VK_NULL_HANDLE;
        }
        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(nativeDevice_, descriptorPool_, nullptr);
            descriptorPool_ = VK_NULL_HANDLE;
            descriptorSet_ = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(nativeDevice_, descriptorSetLayout_, nullptr);
            descriptorSetLayout_ = VK_NULL_HANDLE;
        }
        nativeDevice_ = VK_NULL_HANDLE;
    }

    Result createNativePipeline(const ShaderCompileResult& computeCompile, std::string& log)
    {
        destroyNativePipelineObjects();

        VkDescriptorSetLayoutBinding accelerationBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        VkDescriptorSetLayoutBinding outputBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        const VkDescriptorSetLayoutBinding bindings[] = {accelerationBinding, outputBinding};
        VkDescriptorSetLayoutCreateInfo setLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pBindings = bindings,
        };
        VkResult vkResult = vkCreateDescriptorSetLayout(
            nativeDevice_,
            &setLayoutInfo,
            nullptr,
            &descriptorSetLayout_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateDescriptorSetLayout(SceneRayQueryVisualizationPass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkDescriptorPoolSize poolSizes[] = {
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                .descriptorCount = 1,
            },
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
            },
        };
        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = static_cast<uint32_t>(std::size(poolSizes)),
            .pPoolSizes = poolSizes,
        };
        vkResult = vkCreateDescriptorPool(nativeDevice_, &poolInfo, nullptr, &descriptorPool_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateDescriptorPool(SceneRayQueryVisualizationPass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkDescriptorSetAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptorSetLayout_,
        };
        vkResult = vkAllocateDescriptorSets(nativeDevice_, &allocateInfo, &descriptorSet_);
        if (vkResult != VK_SUCCESS) {
            log = "vkAllocateDescriptorSets(SceneRayQueryVisualizationPass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkPushConstantRange pushRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(SceneRayQueryVisualizationPush),
        };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushRange,
        };
        vkResult = vkCreatePipelineLayout(nativeDevice_, &pipelineLayoutInfo, nullptr, &pipelineLayout_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreatePipelineLayout(SceneRayQueryVisualizationPass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkShaderModuleCreateInfo shaderInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = computeCompile.spirv.size() * sizeof(uint32_t),
            .pCode = computeCompile.spirv.data(),
        };
        vkResult = vkCreateShaderModule(nativeDevice_, &shaderInfo, nullptr, &shaderModule_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateShaderModule(SceneRayQueryVisualizationPass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkPipelineShaderStageCreateInfo stageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderModule_,
            .pName = "main",
        };
        VkComputePipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stageInfo,
            .layout = pipelineLayout_,
        };
        vkResult = vkCreateComputePipelines(
            nativeDevice_,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &pipeline_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateComputePipelines(SceneRayQueryVisualizationPass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        return {};
    }

    void destroyNativePipelineObjects()
    {
        if (nativeDevice_ == VK_NULL_HANDLE) {
            return;
        }
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(nativeDevice_, pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(nativeDevice_, pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (shaderModule_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(nativeDevice_, shaderModule_, nullptr);
            shaderModule_ = VK_NULL_HANDLE;
        }
        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(nativeDevice_, descriptorPool_, nullptr);
            descriptorPool_ = VK_NULL_HANDLE;
            descriptorSet_ = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(nativeDevice_, descriptorSetLayout_, nullptr);
            descriptorSetLayout_ = VK_NULL_HANDLE;
        }
    }

    void updateDescriptorSet(VkImageView outputView)
    {
        VkAccelerationStructureKHR tlas = rtxBuilder_.tlas();
        VkWriteDescriptorSetAccelerationStructureKHR accelerationInfo{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &tlas,
        };
        VkWriteDescriptorSet accelerationWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &accelerationInfo,
            .dstSet = descriptorSet_,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        };

        VkDescriptorImageInfo outputInfo{
            .imageView = outputView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkWriteDescriptorSet outputWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet_,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &outputInfo,
        };

        const VkWriteDescriptorSet writes[] = {accelerationWrite, outputWrite};
        vkUpdateDescriptorSets(nativeDevice_, static_cast<uint32_t>(std::size(writes)), writes, 0, nullptr);
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

    vulkan::SceneRtxBuilder rtxBuilder_;
    scene::Bounds drawBounds_;
    VkDevice nativeDevice_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
};

class ScenePathTracePass final : public ComputePass {
public:
    ~ScenePathTracePass() override
    {
        destroyNative();
    }

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
        if (pipeline_ != VK_NULL_HANDLE && rtxBuilder_.valid() && vertexBuffer_ != nullptr) {
            return {};
        }

        scene::Scene loadedScene;
        const std::filesystem::path path = scenePathFromProperties(properties());
        if (!loadedScene.load(path)) {
            log = "ScenePathTracePass failed to load glTF: " + loadedScene.lastLoadResult().error;
            return makeError(Error::Failure);
        }
        if (!loadedScene.bounds().valid) {
            log = "ScenePathTracePass scene bounds are unavailable";
            return makeError(Error::Failure);
        }

        ScenePathTraceGpuScene gpuScene;
        if (!buildGpuScene(loadedScene, gpuScene, log)) {
            return makeError(Error::Failure);
        }

        Result result = rtxBuilder_.build(*context.device, *context.graphicsQueue, loadedScene, log);
        if (!result) {
            return result;
        }

        result = uploadStorageBuffer(
            *context.device,
            gpuScene.vertices.data(),
            static_cast<uint64_t>(gpuScene.vertices.size() * sizeof(ScenePathTraceGpuVertex)),
            sizeof(ScenePathTraceGpuVertex),
            vertexBuffer_,
            log,
            "ScenePathTracePass vertices");
        if (!result) {
            resetGpuBuffers();
            rtxBuilder_.clear();
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            gpuScene.indices.data(),
            static_cast<uint64_t>(gpuScene.indices.size() * sizeof(uint32_t)),
            sizeof(uint32_t),
            indexBuffer_,
            log,
            "ScenePathTracePass indices");
        if (!result) {
            resetGpuBuffers();
            rtxBuilder_.clear();
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            gpuScene.primitives.data(),
            static_cast<uint64_t>(gpuScene.primitives.size() * sizeof(ScenePathTraceGpuPrimitive)),
            sizeof(ScenePathTraceGpuPrimitive),
            primitiveBuffer_,
            log,
            "ScenePathTracePass primitives");
        if (!result) {
            resetGpuBuffers();
            rtxBuilder_.clear();
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            gpuScene.instances.data(),
            static_cast<uint64_t>(gpuScene.instances.size() * sizeof(ScenePathTraceGpuInstance)),
            sizeof(ScenePathTraceGpuInstance),
            instanceBuffer_,
            log,
            "ScenePathTracePass instances");
        if (!result) {
            resetGpuBuffers();
            rtxBuilder_.clear();
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            gpuScene.materials.data(),
            static_cast<uint64_t>(gpuScene.materials.size() * sizeof(ScenePathTraceGpuMaterial)),
            sizeof(ScenePathTraceGpuMaterial),
            materialBuffer_,
            log,
            "ScenePathTracePass materials");
        if (!result) {
            resetGpuBuffers();
            rtxBuilder_.clear();
            return result;
        }

        nativeDevice_ = vulkan::nativeDevice(*context.device).device;
        if (nativeDevice_ == VK_NULL_HANDLE) {
            log = "ScenePathTracePass Vulkan device is unavailable";
            resetGpuBuffers();
            rtxBuilder_.clear();
            return makeError(Error::InvalidArgument);
        }
        volkLoadDevice(nativeDevice_);

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
            destroyNative();
            resetGpuBuffers();
            rtxBuilder_.clear();
            return result;
        }

        result = createNativePipeline(computeCompile, log);
        if (!result) {
            destroyNative();
            resetGpuBuffers();
            rtxBuilder_.clear();
            return result;
        }

        drawBounds_ = loadedScene.bounds();
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            nativeDevice_ == VK_NULL_HANDLE ||
            descriptorSet_ == VK_NULL_HANDLE ||
            pipeline_ == VK_NULL_HANDLE ||
            pipelineLayout_ == VK_NULL_HANDLE ||
            vertexBuffer_ == nullptr ||
            indexBuffer_ == nullptr ||
            primitiveBuffer_ == nullptr ||
            instanceBuffer_ == nullptr ||
            materialBuffer_ == nullptr ||
            !rtxBuilder_.valid() ||
            !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        VkImageView outputView = vulkan::nativeImageView(*color.view());
        if (outputView == VK_NULL_HANDLE) {
            return makeError(Error::InvalidArgument);
        }

        ScenePathTracePush push;
        buildPush(context.width(), context.height(), context.properties(), drawBounds_, push);

        VkImageView historyCurrentView = outputView;
        VkImageView historyPreviousView = outputView;
        Result result = prepareHistoryTextures(
            context,
            outputView,
            push,
            historyCurrentView,
            historyPreviousView);
        if (!result) {
            return result;
        }
        updateDescriptorSet(outputView, historyCurrentView, historyPreviousView);

        VkCommandBuffer commandBuffer = vulkan::nativeCommandBuffer(context.commandBuffer());
        if (commandBuffer == VK_NULL_HANDLE) {
            return makeError(Error::InvalidArgument);
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout_,
            0,
            1,
            &descriptorSet_,
            0,
            nullptr);
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(push),
            &push);
        vkCmdDispatch(commandBuffer, (context.width() + 7) / 8, (context.height() + 7) / 8, 1);
        if (push.enableAccumulation != 0 && context.historyResources() != nullptr) {
            context.historyResources()->markWritten(historyNameForContext(context));
        }
        return {};
    }

private:
    struct ScenePathTraceGpuScene {
        std::vector<ScenePathTraceGpuVertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<ScenePathTraceGpuPrimitive> primitives;
        std::vector<ScenePathTraceGpuInstance> instances;
        std::vector<ScenePathTraceGpuMaterial> materials;
    };

    Result prepareHistoryTextures(
        RenderGraphExecutionContext& context,
        VkImageView fallbackView,
        ScenePathTracePush& push,
        VkImageView& outCurrentView,
        VkImageView& outPreviousView)
    {
        HistoryResourceManager* history = context.historyResources();
        const bool accumulationEnabled = boolProperty(context.properties(), "accumulate", true);
        push.enableAccumulation = accumulationEnabled && history != nullptr ? 1u : 0u;
        push.hasHistory = 0;
        push.accumulationFrame = 0;
        outCurrentView = fallbackView;
        outPreviousView = fallbackView;
        if (push.enableAccumulation == 0) {
            accumulationFrame_ = 0;
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

        outCurrentView = vulkan::nativeImageView(*current.view);
        outPreviousView = vulkan::nativeImageView(*previous.view);
        if (outCurrentView == VK_NULL_HANDLE || outPreviousView == VK_NULL_HANDLE) {
            return makeError(Error::InvalidArgument);
        }

        if (previous.valid) {
            ++accumulationFrame_;
            push.hasHistory = 1;
        } else {
            accumulationFrame_ = 0;
            push.hasHistory = 0;
        }
        push.accumulationFrame = accumulationFrame_;
        return {};
    }

    void destroyNative()
    {
        if (nativeDevice_ == VK_NULL_HANDLE) {
            return;
        }
        volkLoadDevice(nativeDevice_);
        destroyNativePipelineObjects();
        nativeDevice_ = VK_NULL_HANDLE;
    }

    Result createNativePipeline(const ShaderCompileResult& computeCompile, std::string& log)
    {
        destroyNativePipelineObjects();

        const VkDescriptorSetLayoutBinding bindings[] = {
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 4,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 5,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 6,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 7,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding = 8,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
        };
        VkDescriptorSetLayoutCreateInfo setLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pBindings = bindings,
        };
        VkResult vkResult = vkCreateDescriptorSetLayout(
            nativeDevice_,
            &setLayoutInfo,
            nullptr,
            &descriptorSetLayout_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateDescriptorSetLayout(ScenePathTracePass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkDescriptorPoolSize poolSizes[] = {
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                .descriptorCount = 1,
            },
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 3,
            },
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 5,
            },
        };
        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = static_cast<uint32_t>(std::size(poolSizes)),
            .pPoolSizes = poolSizes,
        };
        vkResult = vkCreateDescriptorPool(nativeDevice_, &poolInfo, nullptr, &descriptorPool_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateDescriptorPool(ScenePathTracePass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkDescriptorSetAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptorSetLayout_,
        };
        vkResult = vkAllocateDescriptorSets(nativeDevice_, &allocateInfo, &descriptorSet_);
        if (vkResult != VK_SUCCESS) {
            log = "vkAllocateDescriptorSets(ScenePathTracePass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkPushConstantRange pushRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(ScenePathTracePush),
        };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout_,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushRange,
        };
        vkResult = vkCreatePipelineLayout(nativeDevice_, &pipelineLayoutInfo, nullptr, &pipelineLayout_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreatePipelineLayout(ScenePathTracePass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkShaderModuleCreateInfo shaderInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = computeCompile.spirv.size() * sizeof(uint32_t),
            .pCode = computeCompile.spirv.data(),
        };
        vkResult = vkCreateShaderModule(nativeDevice_, &shaderInfo, nullptr, &shaderModule_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateShaderModule(ScenePathTracePass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        VkPipelineShaderStageCreateInfo stageInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderModule_,
            .pName = "main",
        };
        VkComputePipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stageInfo,
            .layout = pipelineLayout_,
        };
        vkResult = vkCreateComputePipelines(
            nativeDevice_,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &pipeline_);
        if (vkResult != VK_SUCCESS) {
            log = "vkCreateComputePipelines(ScenePathTracePass) returned " +
                std::to_string(static_cast<int>(vkResult));
            return makeError(Error::Failure);
        }

        return {};
    }

    void destroyNativePipelineObjects()
    {
        if (nativeDevice_ == VK_NULL_HANDLE) {
            return;
        }
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(nativeDevice_, pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(nativeDevice_, pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (shaderModule_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(nativeDevice_, shaderModule_, nullptr);
            shaderModule_ = VK_NULL_HANDLE;
        }
        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(nativeDevice_, descriptorPool_, nullptr);
            descriptorPool_ = VK_NULL_HANDLE;
            descriptorSet_ = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(nativeDevice_, descriptorSetLayout_, nullptr);
            descriptorSetLayout_ = VK_NULL_HANDLE;
        }
    }

    void updateDescriptorSet(
        VkImageView outputView,
        VkImageView historyCurrentView,
        VkImageView historyPreviousView)
    {
        VkAccelerationStructureKHR tlas = rtxBuilder_.tlas();
        VkWriteDescriptorSetAccelerationStructureKHR accelerationInfo{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &tlas,
        };

        VkDescriptorImageInfo outputInfo{
            .imageView = outputView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkDescriptorImageInfo historyCurrentInfo{
            .imageView = historyCurrentView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkDescriptorImageInfo historyPreviousInfo{
            .imageView = historyPreviousView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        const vulkan::NativeBuffer vertexBuffer = vulkan::nativeBuffer(*vertexBuffer_);
        const vulkan::NativeBuffer indexBuffer = vulkan::nativeBuffer(*indexBuffer_);
        const vulkan::NativeBuffer primitiveBuffer = vulkan::nativeBuffer(*primitiveBuffer_);
        const vulkan::NativeBuffer instanceBuffer = vulkan::nativeBuffer(*instanceBuffer_);
        const vulkan::NativeBuffer materialBuffer = vulkan::nativeBuffer(*materialBuffer_);
        const VkDescriptorBufferInfo bufferInfos[] = {
            VkDescriptorBufferInfo{.buffer = vertexBuffer.buffer, .offset = 0, .range = vertexBuffer.size},
            VkDescriptorBufferInfo{.buffer = indexBuffer.buffer, .offset = 0, .range = indexBuffer.size},
            VkDescriptorBufferInfo{.buffer = primitiveBuffer.buffer, .offset = 0, .range = primitiveBuffer.size},
            VkDescriptorBufferInfo{.buffer = instanceBuffer.buffer, .offset = 0, .range = instanceBuffer.size},
            VkDescriptorBufferInfo{.buffer = materialBuffer.buffer, .offset = 0, .range = materialBuffer.size},
        };

        const VkWriteDescriptorSet writes[] = {
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = &accelerationInfo,
                .dstSet = descriptorSet_,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &outputInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bufferInfos[0],
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bufferInfos[1],
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 4,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bufferInfos[2],
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 5,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bufferInfos[3],
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 6,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bufferInfos[4],
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 7,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &historyCurrentInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet_,
                .dstBinding = 8,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &historyPreviousInfo,
            },
        };
        vkUpdateDescriptorSets(nativeDevice_, static_cast<uint32_t>(std::size(writes)), writes, 0, nullptr);
    }

    static Result uploadStorageBuffer(
        Device& device,
        const void* data,
        uint64_t byteSize,
        uint32_t structureStride,
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
                .structureStride = structureStride,
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
        return kDefaultPathTraceScenePath;
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

    static ScenePathTraceGpuMaterial makeMaterial(const scene::RenderMaterial& material)
    {
        ScenePathTraceGpuMaterial gpuMaterial;
        gpuMaterial.baseColor[0] = material.baseColorFactor.x;
        gpuMaterial.baseColor[1] = material.baseColorFactor.y;
        gpuMaterial.baseColor[2] = material.baseColorFactor.z;
        gpuMaterial.baseColor[3] = material.baseColorFactor.w;
        gpuMaterial.emissive[0] = material.emissiveFactor.x;
        gpuMaterial.emissive[1] = material.emissiveFactor.y;
        gpuMaterial.emissive[2] = material.emissiveFactor.z;
        gpuMaterial.emissive[3] = 0.0f;
        gpuMaterial.params[0] = material.metallicFactor;
        gpuMaterial.params[1] = material.roughnessFactor;
        gpuMaterial.params[2] = material.alphaCutoff;
        gpuMaterial.params[3] = material.doubleSided ? 1.0f : 0.0f;
        return gpuMaterial;
    }

    static uint32_t materialIndexForNode(const scene::RenderNode& renderNode, uint32_t materialCount)
    {
        if (renderNode.materialIndex >= 0 &&
            static_cast<uint32_t>(renderNode.materialIndex) < materialCount) {
            return static_cast<uint32_t>(renderNode.materialIndex);
        }
        return 0;
    }

    static bool appendPrimitiveGeometry(
        const scene::RenderPrimitive& primitive,
        ScenePathTraceGpuScene& outScene,
        ScenePathTraceGpuPrimitive& outPrimitive)
    {
        const uint64_t sourceIndexCount = primitive.indices.empty()
            ? (primitive.positions.size() / 3) * 3
            : (primitive.indices.size() / 3) * 3;
        if (primitive.mode != kGltfTriangleListMode ||
            primitive.positions.size() < 3 ||
            sourceIndexCount < 3 ||
            sourceIndexCount > std::numeric_limits<uint32_t>::max() ||
            primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        outPrimitive = ScenePathTraceGpuPrimitive{
            .firstVertex = static_cast<uint32_t>(outScene.vertices.size()),
            .vertexCount = static_cast<uint32_t>(primitive.positions.size()),
            .firstIndex = static_cast<uint32_t>(outScene.indices.size()),
            .indexCount = static_cast<uint32_t>(sourceIndexCount),
        };

        for (size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
            const float3 position = primitive.positions[vertexIndex];
            const float3 normal = vertexIndex < primitive.normals.size()
                ? primitive.normals[vertexIndex]
                : float3(0.0f, 0.0f, 0.0f);
            const float2 texcoord = vertexIndex < primitive.texcoords0.size()
                ? primitive.texcoords0[vertexIndex]
                : float2(0.0f, 0.0f);
            ScenePathTraceGpuVertex vertex;
            vertex.position[0] = position.x;
            vertex.position[1] = position.y;
            vertex.position[2] = position.z;
            vertex.position[3] = 1.0f;
            vertex.normal[0] = normal.x;
            vertex.normal[1] = normal.y;
            vertex.normal[2] = normal.z;
            vertex.normal[3] = 0.0f;
            vertex.texcoord[0] = texcoord.x;
            vertex.texcoord[1] = texcoord.y;
            outScene.vertices.push_back(vertex);
        }

        if (primitive.indices.empty()) {
            for (uint32_t index = 0; index < outPrimitive.indexCount; ++index) {
                outScene.indices.push_back(index);
            }
            return true;
        }

        for (uint32_t index = 0; index < outPrimitive.indexCount; ++index) {
            const uint32_t sourceIndex = primitive.indices[index];
            if (sourceIndex >= outPrimitive.vertexCount) {
                outScene.vertices.resize(outPrimitive.firstVertex);
                outScene.indices.resize(outPrimitive.firstIndex);
                return false;
            }
            outScene.indices.push_back(sourceIndex);
        }
        return true;
    }

    static bool buildGpuScene(
        const scene::Scene& scene,
        ScenePathTraceGpuScene& outScene,
        std::string& log)
    {
        outScene = ScenePathTraceGpuScene{};
        outScene.materials.reserve(std::max<size_t>(scene.materials().size(), 1));
        if (scene.materials().empty()) {
            outScene.materials.push_back(ScenePathTraceGpuMaterial{});
        } else {
            for (const scene::RenderMaterial& material : scene.materials()) {
                outScene.materials.push_back(makeMaterial(material));
            }
        }

        constexpr uint32_t kInvalidPrimitiveIndex = std::numeric_limits<uint32_t>::max();
        std::vector<uint32_t> primitiveToGpuPrimitive(
            scene.renderPrimitives().size(),
            kInvalidPrimitiveIndex);
        for (uint32_t primitiveIndex = 0; primitiveIndex < scene.renderPrimitives().size(); ++primitiveIndex) {
            ScenePathTraceGpuPrimitive gpuPrimitive;
            if (!appendPrimitiveGeometry(scene.renderPrimitives()[primitiveIndex], outScene, gpuPrimitive)) {
                continue;
            }
            primitiveToGpuPrimitive[primitiveIndex] = static_cast<uint32_t>(outScene.primitives.size());
            outScene.primitives.push_back(gpuPrimitive);
        }

        for (const scene::RenderNode& renderNode : scene.renderNodes()) {
            if (!renderNode.visible ||
                renderNode.renderPrimitiveIndex < 0 ||
                static_cast<size_t>(renderNode.renderPrimitiveIndex) >= primitiveToGpuPrimitive.size()) {
                continue;
            }
            const uint32_t primitiveIndex =
                primitiveToGpuPrimitive[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
            if (primitiveIndex == kInvalidPrimitiveIndex) {
                continue;
            }

            outScene.instances.push_back(ScenePathTraceGpuInstance{
                .primitiveIndex = primitiveIndex,
                .materialIndex = materialIndexForNode(
                    renderNode,
                    static_cast<uint32_t>(outScene.materials.size())),
            });
        }

        if (outScene.vertices.empty() ||
            outScene.indices.empty() ||
            outScene.primitives.empty() ||
            outScene.instances.empty() ||
            outScene.materials.empty()) {
            log = "ScenePathTracePass found no visible triangle geometry for path tracing";
            return false;
        }
        return true;
    }

    void resetGpuBuffers()
    {
        vertexBuffer_.reset();
        indexBuffer_.reset();
        primitiveBuffer_.reset();
        instanceBuffer_.reset();
        materialBuffer_.reset();
    }

    vulkan::SceneRtxBuilder rtxBuilder_;
    scene::Bounds drawBounds_;
    std::unique_ptr<Buffer> vertexBuffer_;
    std::unique_ptr<Buffer> indexBuffer_;
    std::unique_ptr<Buffer> primitiveBuffer_;
    std::unique_ptr<Buffer> instanceBuffer_;
    std::unique_ptr<Buffer> materialBuffer_;
    VkDevice nativeDevice_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    uint32_t accumulationFrame_ = 0;
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
        "Visualize a glTF Vulkan acceleration structure with RayQuery",
        []() { return std::make_unique<SceneRayQueryVisualizationPass>(); });
    registerRenderGraphPassType(
        "ScenePathTracePass",
        "Path trace a glTF scene with RayQuery",
        []() { return std::make_unique<ScenePathTracePass>(); });
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
