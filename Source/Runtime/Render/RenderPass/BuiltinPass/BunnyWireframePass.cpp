#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

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
        const scene::Scene* runtimeScene = runtimeSceneForPath(
            context.runtimeScene,
            scenePathFromProperties(properties()));
        const uint64_t runtimeRevision = runtimeScene != nullptr ? runtimeScene->transformRevision() : 0;
        if (program_ != nullptr && sceneRevision_ == runtimeRevision) {
            return {};
        }

        std::vector<BunnyWireframeGpuPosition> positions;
        std::vector<SceneGpuTransform> transforms;
        BunnyWireframeGpuParams params;
        if (!loadBunnyGeometry(properties(), runtimeScene, positions, transforms, drawBounds_, log)) {
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
            transforms.data(),
            static_cast<uint64_t>(transforms.size() * sizeof(SceneGpuTransform)),
            transformBuffer_,
            log,
            "BunnyWireframePass transforms");
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
                .maxBuffers = 3,
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
        result = bindlessHeap_->allocateBuffer(transformHandle_);
        if (!result || !transformHandle_.valid()) {
            log += resultMessage("allocateBuffer(BunnyWireframePass transforms)", result);
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
        result = bindlessHeap_->writeStorageBuffer(transformHandle_, *transformBuffer_);
        if (!result) {
            log += resultMessage("writeStorageBuffer(BunnyWireframePass transforms)", result);
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
        sceneRevision_ = runtimeRevision;
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        Result syncResult = syncRuntimeGeometry(context.runtimeScene());
        if (!syncResult) {
            return syncResult;
        }
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
            .transformBuffer = transformHandle_.index,
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

    Result syncRuntimeGeometry(const scene::Scene* runtimeScene)
    {
        runtimeScene = runtimeSceneForPath(runtimeScene, scenePathFromProperties(properties()));
        if (runtimeScene == nullptr || runtimeScene->transformRevision() == sceneRevision_) {
            return {};
        }
        const std::vector<SceneGpuTransform> transforms = buildSceneGpuTransforms(*runtimeScene);
        if (transformBuffer_ == nullptr ||
            transforms.size() * sizeof(SceneGpuTransform) != transformBuffer_->desc().size) {
            spdlog::warn("[BunnyWireframePass] Runtime scene transform layout changed");
            return makeError(Error::Failure);
        }
        void* mapped = transformBuffer_->map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, transforms.data(), static_cast<size_t>(transformBuffer_->desc().size));
        transformBuffer_->flush(0, transformBuffer_->desc().size);
        transformBuffer_->unmap();
        drawBounds_ = runtimeScene->bounds();
        sceneRevision_ = runtimeScene->transformRevision();
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
        const scene::Scene* runtimeScene,
        std::vector<BunnyWireframeGpuPosition>& outPositions,
        std::vector<SceneGpuTransform>& outTransforms,
        scene::Bounds& outBounds,
        std::string& log)
    {
        const std::filesystem::path path = scenePathFromProperties(properties);
        scene::Scene fallbackScene;
        if (runtimeScene == nullptr) {
            if (!fallbackScene.load(path)) {
                log = "BunnyWireframePass failed to load glTF: " + fallbackScene.lastLoadResult().error;
                return false;
            }
            runtimeScene = &fallbackScene;
        }
        const scene::Scene& bunnyScene = *runtimeScene;
        outTransforms = buildSceneGpuTransforms(bunnyScene);

        outBounds.reset();
        for (size_t renderNodeIndex = 0; renderNodeIndex < bunnyScene.renderNodes().size(); ++renderNodeIndex) {
            const scene::RenderNode& renderNode = bunnyScene.renderNodes()[renderNodeIndex];
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
                const float3 local = primitive.positions[static_cast<size_t>(localIndex)];
                outPositions.push_back(BunnyWireframeGpuPosition{
                    .x = local.x,
                    .y = local.y,
                    .z = local.z,
                    .w = static_cast<float>(renderNodeIndex),
                });
                const float3 world = renderNode.worldMatrix * local;
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
    std::unique_ptr<Buffer> transformBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle positionHandle_;
    BindlessHandle transformHandle_;
    BindlessHandle paramsHandle_;
    std::unique_ptr<GraphicsShaderObjectProgram> program_;
    scene::Bounds drawBounds_;
    uint64_t sceneRevision_ = 0;
    uint32_t drawVertexCount_ = 0;
};

} // namespace

std::unique_ptr<RenderGraphPass> createBunnyWireframePass()
{
    return std::make_unique<BunnyWireframePass>();
}

} // namespace metallic::render::builtin_pass
