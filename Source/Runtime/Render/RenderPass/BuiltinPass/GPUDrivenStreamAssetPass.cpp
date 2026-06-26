#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

namespace metallic::render::builtin_pass {
namespace {

std::filesystem::path pathFromProperties(
    const RenderGraphProperties& props,
    const char* key,
    const std::filesystem::path& fallback)
{
    if (props.contains(key) && props[key].is_string()) {
        std::filesystem::path path = props[key].get<std::string>();
        if (path.is_relative()) {
            path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
        }
        return path;
    }
    return fallback;
}

std::filesystem::path scenePathFromProperties(const RenderGraphProperties& props)
{
    return pathFromProperties(props, "path", kDefaultGPUDrivenScenePath);
}

std::filesystem::path streamAssetPathFromProperties(
    const RenderGraphProperties& props,
    const std::filesystem::path& scenePath)
{
    return pathFromProperties(
        props,
        "streamAssetPath",
        scene::meshletStreamAssetPathFor(scenePath));
}

bool boolProperty(const RenderGraphProperties& props, const char* key, bool fallback)
{
    auto iter = props.find(key);
    return iter != props.end() && iter->is_boolean() ? iter->get<bool>() : fallback;
}

uint32_t uintProperty(const RenderGraphProperties& props, const char* key, uint32_t fallback)
{
    auto iter = props.find(key);
    if (iter == props.end() || !iter->is_number_integer()) {
        return fallback;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 || value > std::numeric_limits<uint32_t>::max()
        ? fallback
        : static_cast<uint32_t>(value);
}

uint32_t selectedLodProperty(const RenderGraphProperties& props)
{
    auto iter = props.find("selectedLodLevel");
    if (iter == props.end() || !iter->is_number_integer()) {
        return 0;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 ? 0u : static_cast<uint32_t>(std::min<int64_t>(value, std::numeric_limits<uint32_t>::max()));
}

uint32_t debugColorModeFromProperties(const RenderGraphProperties& props)
{
    auto iter = props.find("debugColorMode");
    if (iter == props.end() || !iter->is_string()) {
        return kMeshletStreamDebugPage;
    }
    const std::string mode = iter->get<std::string>();
    if (mode == "lod") {
        return kMeshletStreamDebugLod;
    }
    if (mode == "primitive") {
        return kMeshletStreamDebugPrimitive;
    }
    return kMeshletStreamDebugPage;
}

const RenderGraphProperties* cameraPropertiesFrom(const RenderGraphProperties& properties)
{
    auto iter = properties.find("camera");
    return iter != properties.end() && iter->is_object() ? &(*iter) : nullptr;
}

float finiteOr(float value, float fallback)
{
    return std::isfinite(value) ? value : fallback;
}

float cameraFloat(const RenderGraphProperties* camera, const char* key, float fallback)
{
    if (camera == nullptr) {
        return fallback;
    }
    auto iter = camera->find(key);
    return iter != camera->end() && iter->is_number()
        ? finiteOr(iter->get<float>(), fallback)
        : fallback;
}

float3 cameraVec3(const RenderGraphProperties* camera, const char* key, const float3& fallback)
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
        if ((*iter)[index].is_number()) {
            values[index] = finiteOr((*iter)[index].get<float>(), values[index]);
        }
    }
    return float3(values[0], values[1], values[2]);
}

MeshletStreamRuntimeDesc runtimeDescFromProperties(const RenderGraphProperties& properties)
{
    const std::filesystem::path scenePath = scenePathFromProperties(properties);
    return MeshletStreamRuntimeDesc{
        .sourcePath = scenePath,
        .streamAssetPath = streamAssetPathFromProperties(properties, scenePath),
        .autoBuildStreamAsset = boolProperty(properties, "autoBuildStreamAsset", true),
        .maxResidentPages = uintProperty(properties, "maxResidentPages", 4096),
        .maxPageUploadsPerFrame = uintProperty(properties, "maxPageUploadsPerFrame", 64),
        .maxGpuPageRequests = std::max<uint32_t>(
            uintProperty(properties, "maxGpuPageRequests", kMeshletStreamDefaultMaxGpuPageRequests),
            1u),
        .queuedFrameCount = 3,
    };
}

Result createMeshShader(Device& device, std::unique_ptr<ShaderModule>& outShader, std::string& log)
{
    ShaderCompileResult meshCompile;
    const char* capabilities[] = {"spvMeshShadingEXT"};
    Result result = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamMeshEntryPoint,
            .searchPath = kMeshletStreamShaderSearchPath,
            .profileName = "glsl_460",
            .capabilities = capabilities,
            .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
        },
        meshCompile);
    if (!result) {
        log += "compileSlangShaderToSpirv(gpu_driven_streamasset.mesh) returned ";
        log += resultToString(result);
        if (!meshCompile.diagnostics.empty()) {
            log += ": ";
            log += meshCompile.diagnostics;
        }
        log += '\n';
        return result;
    }

    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = meshCompile.spirv.data(),
            .byteSize = static_cast<uint64_t>(meshCompile.spirv.size() * sizeof(uint32_t)),
        },
        outShader);
    if (!result || outShader == nullptr) {
        log += resultMessage("createShaderModule(GPUDrivenStreamAsset mesh)", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

} // namespace

class GPUDrivenStreamAssetPass final : public UnsafePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Meshlet streamasset debug color")
            .texture2D()
            .colorWrite();
        reflection.addTextureOutput("depth", "Meshlet streamasset debug depth")
            .texture2D()
            .depthStencilWrite();
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeIntSetting("selectedLodLevel", "LOD", 0, 0, 31),
            runtimeEnumSetting(
                "debugColorMode",
                "Color",
                "page",
                {{"Page", "page"}, {"LOD", "lod"}, {"Primitive", "primitive"}}),
        };
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        log.clear();
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().meshShader ||
            !context.device->capabilities().bindlessDescriptorHeap) {
            log = "GPUDrivenStreamAssetPass requires meshShader and bindlessDescriptorHeap capabilities";
            return makeError(Error::Unsupported);
        }

        Result result = streamRuntime_.initialize(
            *context.device,
            runtimeDescFromProperties(properties()),
            log);
        if (!result) {
            return result;
        }

        result = createMeshShader(*context.device, meshShader_, log);
        if (!result) {
            return result;
        }

        ShaderCompileResult fragmentCompile;
        result = compileSlangShader(
            kMeshletStreamShaderModuleName,
            kMeshletStreamFragmentEntryPoint,
            fragmentCompile,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createShaderModule(
            ShaderModuleDesc{
                .code = fragmentCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(fragmentCompile.spirv.size() * sizeof(uint32_t)),
            },
            fragmentShader_);
        if (!result || fragmentShader_ == nullptr) {
            log += resultMessage("createShaderModule(GPUDrivenStreamAsset fragment)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .meshShader = meshShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = context.defaultFormat,
                .depthStencilFormat = Format::D32Sfloat,
                .depthStencil = DepthStencilState{
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = depthCompareOp(kDefaultReversedZ),
                },
                .usesBindlessHeap = true,
            },
            pipeline_);
        if (!result || pipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenStreamAssetPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !depth.valid() ||
            context.streamer() == nullptr ||
            !streamRuntime_.ready() ||
            streamRuntime_.bindlessHeap() == nullptr ||
            pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const MeshletStreamFrameDesc frame = frameDescFromContext(context);
        Result result = streamRuntime_.cmdBeginFrame(context.commandBuffer(), *context.streamer(), frame);
        if (!result) {
            return result;
        }
        result = streamRuntime_.cmdPreTraversal(context.commandBuffer(), frame);
        if (!result) {
            return result;
        }
        result = draw(context, color, depth);
        if (!result) {
            return result;
        }
        result = streamRuntime_.cmdPostTraversal(context.commandBuffer());
        if (!result) {
            return result;
        }
        return streamRuntime_.cmdEndFrame(context.commandBuffer());
    }

private:
    MeshletStreamFrameDesc frameDescFromContext(const RenderGraphExecutionContext& context) const
    {
        const scene::Bounds& bounds = streamRuntime_.bounds();
        const float3 center = bounds.center();
        const float radius = std::max(bounds.radius(), 1.0f);
        const RenderGraphProperties* camera = cameraPropertiesFrom(context.properties());
        const float3 defaultEye(center.x, center.y + radius * 0.35f, center.z + radius * 2.5f);
        const float3 eye = cameraVec3(camera, "eye", defaultEye);
        const float3 lookAt = cameraVec3(camera, "center", center);
        const float3 up = cameraVec3(camera, "up", float3(0.0f, 1.0f, 0.0f));
        const float fovDegrees = cameraFloat(camera, "fovDegrees", 60.0f);
        const float znear = cameraFloat(camera, "znear", 0.1f);
        const float zfar = cameraFloat(camera, "zfar", std::max(radius * 8.0f, znear + 100.0f));

        return MeshletStreamFrameDesc{
            .width = context.width(),
            .height = context.height(),
            .selectedLodLevel = selectedLodProperty(context.properties()),
            .debugColorMode = debugColorModeFromProperties(context.properties()),
            .camera = MeshletStreamCameraDesc{
                .eye = eye,
                .center = lookAt,
                .up = up,
                .fovDegrees = fovDegrees,
                .znear = znear,
                .zfar = zfar,
            },
        };
    }

    Result draw(RenderGraphExecutionContext& context, TextureHandle color, TextureHandle depth)
    {
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
        RenderingAttachmentDesc depthAttachment{
            .view = depth.view(),
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearDepth = depthClearValue(kDefaultReversedZ),
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
        if (streamRuntime_.drawTaskCount() > 0) {
            context.commandBuffer().bindBindlessHeap(*streamRuntime_.bindlessHeap());
            context.commandBuffer().bindGraphicsPipeline(*pipeline_);
            const MeshletStreamUserPush push = streamRuntime_.userPush();
            context.commandBuffer().pushBindlessData(&push, sizeof(push));
            context.commandBuffer().drawMeshTasks(streamRuntime_.drawTaskCount());
        }
        context.commandBuffer().endRendering();
        return {};
    }

    MeshletStreamRuntime streamRuntime_;
    std::unique_ptr<ShaderModule> meshShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
};

std::unique_ptr<RenderGraphPass> createGPUDrivenStreamAssetPass()
{
    return std::make_unique<GPUDrivenStreamAssetPass>();
}

} // namespace metallic::render::builtin_pass
