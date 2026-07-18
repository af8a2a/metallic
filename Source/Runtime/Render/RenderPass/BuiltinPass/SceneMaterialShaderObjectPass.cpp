#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

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
        const scene::Scene* runtimeScene = runtimeSceneForPath(
            context.runtimeScene,
            scenePathFromProperties(properties()));
        const uint64_t runtimeRevision = runtimeScene != nullptr ? runtimeScene->transformRevision() : 0;
        if (defaultProgram_ != nullptr && !batches_.empty() && sceneRevision_ == runtimeRevision) {
            return {};
        }

        std::vector<MaterialShaderObjectGpuPosition> positions;
        std::vector<uint32_t> materialIndices;
        std::vector<MaterialShaderObjectGpuMaterial> materials;
        std::vector<SceneGpuTransform> transforms;
        if (!loadSceneGeometry(
                properties(),
                runtimeScene,
                positions,
                materialIndices,
                materials,
                transforms,
                batches_,
                drawBounds_,
                log)) {
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
            transforms.data(),
            static_cast<uint64_t>(transforms.size() * sizeof(SceneGpuTransform)),
            transformBuffer_,
            log,
            "SceneMaterialShaderObjectPass transforms");
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
                .maxBuffers = 5,
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
        result = allocateAndWriteBuffer(*bindlessHeap_, *transformBuffer_, transformHandle_, log, "transforms");
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
                .transformBuffer = transformHandle_.index,
            };
            context.commandBuffer().pushBindlessData(&push, sizeof(push));
            context.commandBuffer().draw(batch.vertexCount);
        }

        context.commandBuffer().endRendering();
        return {};
    }

private:
    Result syncRuntimeGeometry(const scene::Scene* runtimeScene)
    {
        runtimeScene = runtimeSceneForPath(runtimeScene, scenePathFromProperties(properties()));
        if (runtimeScene == nullptr || runtimeScene->transformRevision() == sceneRevision_) {
            return {};
        }
        const std::vector<SceneGpuTransform> transforms = buildSceneGpuTransforms(*runtimeScene);
        if (transformBuffer_ == nullptr ||
            transforms.size() * sizeof(SceneGpuTransform) != transformBuffer_->desc().size) {
            spdlog::warn("[SceneMaterialShaderObjectPass] Runtime scene transform layout changed");
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
        uint32_t renderNodeIndex,
        const scene::RenderPrimitive& primitive,
        uint32_t localIndex,
        std::vector<MaterialShaderObjectGpuPosition>& outPositions,
        scene::Bounds& outBounds)
    {
        if (static_cast<size_t>(localIndex) >= primitive.positions.size()) {
            return;
        }
        const float3 local = primitive.positions[static_cast<size_t>(localIndex)];
        outPositions.push_back(MaterialShaderObjectGpuPosition{
            .x = local.x,
            .y = local.y,
            .z = local.z,
            .w = static_cast<float>(renderNodeIndex),
        });
        const float3 world = renderNode.worldMatrix * local;
        outBounds.include(world);
    }

    static bool loadSceneGeometry(
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        std::vector<MaterialShaderObjectGpuPosition>& outPositions,
        std::vector<uint32_t>& outMaterialIndices,
        std::vector<MaterialShaderObjectGpuMaterial>& outMaterials,
        std::vector<SceneGpuTransform>& outTransforms,
        std::vector<MaterialShaderObjectBatch>& outBatches,
        scene::Bounds& outBounds,
        std::string& log)
    {
        const std::filesystem::path path = scenePathFromProperties(properties);
        scene::Scene fallbackScene;
        if (runtimeScene == nullptr) {
            if (!fallbackScene.load(path)) {
                log = "SceneMaterialShaderObjectPass failed to load glTF: " + fallbackScene.lastLoadResult().error;
                return false;
            }
            runtimeScene = &fallbackScene;
        }
        const scene::Scene& loadedScene = *runtimeScene;
        outTransforms = buildSceneGpuTransforms(loadedScene);

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
        for (size_t renderNodeIndex = 0; renderNodeIndex < loadedScene.renderNodes().size(); ++renderNodeIndex) {
            const scene::RenderNode& renderNode = loadedScene.renderNodes()[renderNodeIndex];
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
                    appendTriangleVertex(
                        renderNode,
                        static_cast<uint32_t>(renderNodeIndex),
                        primitive,
                        indices[index],
                        materialPositions,
                        outBounds);
                }
            } else {
                const size_t triangleVertexCount = primitive.positions.size() - (primitive.positions.size() % 3);
                for (size_t index = 0; index < triangleVertexCount; ++index) {
                    appendTriangleVertex(
                        renderNode,
                        static_cast<uint32_t>(renderNodeIndex),
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
    std::unique_ptr<Buffer> transformBuffer_;
    std::unique_ptr<Buffer> materialIndexBuffer_;
    std::unique_ptr<Buffer> materialBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle positionHandle_;
    BindlessHandle transformHandle_;
    BindlessHandle materialIndexHandle_;
    BindlessHandle materialHandle_;
    BindlessHandle paramsHandle_;
    std::unique_ptr<GraphicsShaderObjectProgram> defaultProgram_;
    std::unique_ptr<GraphicsShaderObjectProgram> alternateProgram_;
    std::vector<MaterialShaderObjectBatch> batches_;
    scene::Bounds drawBounds_;
    uint64_t sceneRevision_ = 0;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSceneMaterialShaderObjectPass()
{
    return std::make_unique<SceneMaterialShaderObjectPass>();
}

} // namespace metallic::render::builtin_pass
