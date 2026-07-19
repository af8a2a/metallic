#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/SceneResourceManager.h"

namespace metallic::render::builtin_pass {
namespace {

struct GPUDrivenPreviewMeshletRange {
    uint32_t offset = 0;
    uint32_t count = 0;
};

class GPUDrivenPreviewPass final : public RasterPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Mesh shader meshlet scene preview")
            .format = Format::Rgba8Unorm;
        reflection.addTextureOutput("depth", "Mesh shader meshlet scene depth")
            .depthStencilWrite();
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeEnumSetting(
                "mode",
                "Mode",
                "meshlet",
                {{"Meshlet", "meshlet"}, {"Primitive", "primitive"}, {"LOD Group", "lod"}}),
            runtimeIntSetting("lodLevel", "LOD Level", 0, 0, 31),
        };
        appendCameraRuntimeSettings(
            settings,
            std::array<float, 3>{0.0f, 2.0f, 8.0f},
            std::array<float, 3>{0.0f, 1.0f, 0.0f},
            60.0f);
        return settings;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().meshShader ||
            !context.device->capabilities().bindlessDescriptorHeap) {
            log = "GPUDrivenPreviewPass requires meshShader and bindlessDescriptorHeap capabilities";
            return makeError(Error::Unsupported);
        }
        const scene::Scene* runtimeScene = runtimeSceneForPath(
            context.runtimeScene,
            scenePathFromProperties(properties()));
        if (runtimeScene == nullptr && context.sceneResourceManager != nullptr) {
            Result sceneResult = context.sceneResourceManager->resolveScene(
                properties(), context.runtimeScene, runtimeScene, log);
            if (!sceneResult) {
                return sceneResult;
            }
        }
        const uint64_t runtimeRevision = runtimeScene != nullptr ? runtimeScene->transformRevision() : 0;
        if (pipeline_ != nullptr && drawTaskCount_ > 0 && sceneRevision_ == runtimeRevision) {
            return {};
        }

        std::vector<GPUDrivenPreviewGpuPosition> positions;
        std::vector<GPUDrivenPreviewGpuMeshlet> meshlets;
        std::vector<uint32_t> meshletVertices;
        std::vector<uint32_t> meshletTriangles;
        std::vector<SceneGpuTransform> transforms;
        std::vector<GPUDrivenPreviewMeshletRange> lodLevelRanges;
        GPUDrivenPreviewMeshletRange baseMeshletRange;
        if (!loadMeshletScene(
                properties(),
                runtimeScene,
                positions,
                meshlets,
                meshletVertices,
                meshletTriangles,
                transforms,
                drawBounds_,
                baseMeshletRange,
                lodLevelRanges,
                log)) {
            return makeError(Error::Failure);
        }

        baseMeshletRange_ = baseMeshletRange;
        lodLevelRanges_ = std::move(lodLevelRanges);
        drawTaskCount_ = maxMeshletRangeCount(baseMeshletRange_, lodLevelRanges_);

        GPUDrivenPreviewGpuParams params;
        buildParams(
            context.width,
            context.height,
            properties(),
            drawBounds_,
            baseMeshletRange_,
            lodLevelRanges_,
            params);

        Result result = uploadStorageBuffer(
            *context.device,
            positions.data(),
            static_cast<uint64_t>(positions.size() * sizeof(GPUDrivenPreviewGpuPosition)),
            positionBuffer_,
            log,
            "GPUDrivenPreviewPass positions");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            transforms.data(),
            static_cast<uint64_t>(transforms.size() * sizeof(SceneGpuTransform)),
            transformBuffer_,
            log,
            "GPUDrivenPreviewPass transforms");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            meshlets.data(),
            static_cast<uint64_t>(meshlets.size() * sizeof(GPUDrivenPreviewGpuMeshlet)),
            meshletBuffer_,
            log,
            "GPUDrivenPreviewPass meshlets");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            meshletVertices.data(),
            static_cast<uint64_t>(meshletVertices.size() * sizeof(uint32_t)),
            meshletVertexBuffer_,
            log,
            "GPUDrivenPreviewPass meshlet vertices");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            meshletTriangles.data(),
            static_cast<uint64_t>(meshletTriangles.size() * sizeof(uint32_t)),
            meshletTriangleBuffer_,
            log,
            "GPUDrivenPreviewPass meshlet triangles");
        if (!result) {
            return result;
        }
        result = uploadStorageBuffer(
            *context.device,
            &params,
            sizeof(params),
            paramsBuffer_,
            log,
            "GPUDrivenPreviewPass params");
        if (!result) {
            return result;
        }

        result = context.device->createBindlessHeap(
            BindlessHeapDesc{
                .maxBuffers = 6,
            },
            bindlessHeap_);
        if (!result || bindlessHeap_ == nullptr) {
            log += resultMessage("createBindlessHeap(GPUDrivenPreviewPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = allocateAndWriteBuffer(*bindlessHeap_, *positionBuffer_, positionHandle_, log, "positions");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *meshletBuffer_, meshletHandle_, log, "meshlets");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *meshletVertexBuffer_, meshletVertexHandle_, log, "meshlet vertices");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *meshletTriangleBuffer_, meshletTriangleHandle_, log, "meshlet triangles");
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

        ShaderCompileResult meshCompile;
        const char* capabilities[] = {"spvMeshShadingEXT"};
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kGPUDrivenPreviewShaderModuleName,
                .entryPointName = kGPUDrivenPreviewMeshEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .capabilities = capabilities,
                .capabilityCount = static_cast<uint32_t>(std::size(capabilities)),
            },
            meshCompile);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += kGPUDrivenPreviewShaderModuleName;
            log += ".";
            log += kGPUDrivenPreviewMeshEntryPoint;
            log += ") returned ";
            log += resultToString(result);
            if (!meshCompile.diagnostics.empty()) {
                log += ": ";
                log += meshCompile.diagnostics;
            }
            log += '\n';
            return result;
        }

        ShaderCompileResult fragmentCompile;
        result = compileSlangShader(
            kGPUDrivenPreviewShaderModuleName,
            kGPUDrivenPreviewFragmentEntryPoint,
            fragmentCompile,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createShaderModule(
            ShaderModuleDesc{
                .code = meshCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(meshCompile.spirv.size() * sizeof(uint32_t)),
            },
            meshShader_);
        if (!result) {
            log += resultMessage("createShaderModule(GPUDrivenPreviewPass mesh)", result);
            log += '\n';
            return result;
        }
        result = context.device->createShaderModule(
            ShaderModuleDesc{
                .code = fragmentCompile.spirv.data(),
                .byteSize = static_cast<uint64_t>(fragmentCompile.spirv.size() * sizeof(uint32_t)),
            },
            fragmentShader_);
        if (!result) {
            log += resultMessage("createShaderModule(GPUDrivenPreviewPass fragment)", result);
            log += '\n';
            return result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .meshShader = meshShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
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
            log += resultMessage("createGraphicsPipeline(GPUDrivenPreviewPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
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
            pipeline_ == nullptr ||
            drawTaskCount_ == 0) {
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
        context.commandBuffer().bindBindlessHeap(*bindlessHeap_);
        context.commandBuffer().bindGraphicsPipeline(*pipeline_);
        const GPUDrivenPreviewUserPush push{
            .positionBuffer = positionHandle_.index,
            .meshletBuffer = meshletHandle_.index,
            .meshletVertexBuffer = meshletVertexHandle_.index,
            .meshletTriangleBuffer = meshletTriangleHandle_.index,
            .paramsBuffer = paramsHandle_.index,
            .transformBuffer = transformHandle_.index,
        };
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().drawMeshTasks(drawTaskCount_);
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
            spdlog::warn("[GPUDrivenPreviewPass] Runtime scene transform layout changed");
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
            log += resultMessage(std::string("allocateBuffer(GPUDrivenPreviewPass ") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = heap.writeStorageBuffer(outHandle, buffer);
        if (!result) {
            log += resultMessage(std::string("writeStorageBuffer(GPUDrivenPreviewPass ") + std::string(label) + ")", result);
            log += '\n';
        }
        return result;
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
        return kDefaultGPUDrivenScenePath;
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

    static uint32_t previewModeFromProperties(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return kGPUDrivenPreviewModeMeshlet;
        }
        auto iter = properties.find("mode");
        if (iter == properties.end() || !iter->is_string()) {
            return kGPUDrivenPreviewModeMeshlet;
        }
        const std::string value = iter->get<std::string>();
        if (value == "primitive" || value == "perPrimitive" || value == "per primitive") {
            return kGPUDrivenPreviewModePrimitive;
        }
        if (value == "lod" || value == "lodLevel" || value == "lod level" || value == "LOD") {
            return kGPUDrivenPreviewModeLod;
        }
        return kGPUDrivenPreviewModeMeshlet;
    }

    static uint32_t lodLevelFromProperties(const RenderGraphProperties& properties)
    {
        if (!properties.is_object()) {
            return 0;
        }
        auto iter = properties.find("lodLevel");
        if (iter == properties.end() || !iter->is_number_integer()) {
            return 0;
        }
        return static_cast<uint32_t>(std::clamp(iter->get<int32_t>(), 0, 31));
    }

    static GPUDrivenPreviewMeshletRange selectedMeshletRange(
        uint32_t mode,
        uint32_t requestedLodLevel,
        const GPUDrivenPreviewMeshletRange& baseRange,
        const std::vector<GPUDrivenPreviewMeshletRange>& lodLevelRanges,
        uint32_t& outSelectedLodLevel)
    {
        outSelectedLodLevel = 0;
        if (mode != kGPUDrivenPreviewModeLod || lodLevelRanges.empty()) {
            return baseRange;
        }

        uint32_t lodLevel = std::min<uint32_t>(
            requestedLodLevel,
            static_cast<uint32_t>(lodLevelRanges.size() - 1u));
        if (lodLevelRanges[lodLevel].count == 0) {
            uint32_t fallback = lodLevel;
            while (fallback > 0 && lodLevelRanges[fallback].count == 0) {
                --fallback;
            }
            if (lodLevelRanges[fallback].count == 0) {
                for (uint32_t index = lodLevel + 1u; index < lodLevelRanges.size(); ++index) {
                    if (lodLevelRanges[index].count != 0) {
                        fallback = index;
                        break;
                    }
                }
            }
            lodLevel = fallback;
        }

        if (lodLevelRanges[lodLevel].count == 0) {
            return baseRange;
        }
        outSelectedLodLevel = lodLevel;
        return lodLevelRanges[lodLevel];
    }

    static uint32_t maxMeshletRangeCount(
        const GPUDrivenPreviewMeshletRange& baseRange,
        const std::vector<GPUDrivenPreviewMeshletRange>& lodLevelRanges)
    {
        uint32_t result = baseRange.count;
        for (const GPUDrivenPreviewMeshletRange& range : lodLevelRanges) {
            result = std::max(result, range.count);
        }
        return result;
    }

    static void writeParamVec3(const float3& value, float out[4], float w)
    {
        out[0] = value.x;
        out[1] = value.y;
        out[2] = value.z;
        out[3] = w;
    }

    struct PrimitiveInstanceRef {
        const scene::RenderPrimitive* primitive = nullptr;
        uint32_t positionBase = 0;
        uint32_t primitiveIndex = 0;
        uint32_t materialIndex = 0;
        uint32_t transformIndex = 0;
    };

    static bool appendPrimitivePositions(
        const scene::RenderPrimitive& primitive,
        std::vector<GPUDrivenPreviewGpuPosition>& outPositions,
        uint32_t& outPositionBase,
        std::string& log)
    {
        if (outPositions.size() + primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
            log = "GPUDrivenPreviewPass scene is too large to address with uint32 vertex indices";
            return false;
        }

        outPositionBase = static_cast<uint32_t>(outPositions.size());
        outPositions.reserve(outPositions.size() + primitive.positions.size());
        for (const float3& localPosition : primitive.positions) {
            outPositions.push_back(GPUDrivenPreviewGpuPosition{
                .x = localPosition.x,
                .y = localPosition.y,
                .z = localPosition.z,
                .w = 1.0f,
            });
        }
        return true;
    }

    static bool appendPrimitiveClusters(
        const scene::RenderPrimitive& primitive,
        const std::vector<scene::MeshletCluster>& clusters,
        const std::vector<uint32_t>& clusterVertices,
        const std::vector<uint8_t>& clusterTriangles,
        uint32_t firstCluster,
        uint32_t clusterCount,
        uint32_t positionBase,
        uint32_t primitiveIndex,
        uint32_t materialIndex,
        uint32_t transformIndex,
        std::vector<GPUDrivenPreviewGpuMeshlet>& outMeshlets,
        std::vector<uint32_t>& outMeshletVertices,
        std::vector<uint32_t>& outMeshletTriangles,
        std::string& log)
    {
        if (static_cast<size_t>(firstCluster) + clusterCount > clusters.size()) {
            log = "GPUDrivenPreviewPass found invalid meshlet cluster range";
            return false;
        }

        for (uint32_t clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex) {
            const scene::MeshletCluster& cluster = clusters[static_cast<size_t>(firstCluster) + clusterIndex];
            if (cluster.vertexCount == 0 ||
                cluster.triangleCount == 0 ||
                cluster.vertexCount > 128 ||
                cluster.triangleCount > 128 ||
                static_cast<size_t>(cluster.vertexOffset) + cluster.vertexCount > clusterVertices.size() ||
                static_cast<size_t>(cluster.triangleOffset) + static_cast<size_t>(cluster.triangleCount) * 3u >
                    clusterTriangles.size()) {
                log = "GPUDrivenPreviewPass found invalid meshlet cluster data";
                return false;
            }

            const uint32_t meshletVertexOffset = static_cast<uint32_t>(outMeshletVertices.size());
            const uint32_t meshletTriangleOffset = static_cast<uint32_t>(outMeshletTriangles.size());

            for (uint32_t vertexIndex = 0; vertexIndex < cluster.vertexCount; ++vertexIndex) {
                const uint32_t localVertex =
                    clusterVertices[static_cast<size_t>(cluster.vertexOffset) + vertexIndex];
                if (localVertex >= primitive.positions.size()) {
                    log = "GPUDrivenPreviewPass found out-of-range meshlet vertex reference";
                    return false;
                }
                outMeshletVertices.push_back(positionBase + localVertex);
            }

            for (uint32_t triangleIndex = 0; triangleIndex < cluster.triangleCount * 3u; ++triangleIndex) {
                const uint32_t localVertex =
                    clusterTriangles[static_cast<size_t>(cluster.triangleOffset) + triangleIndex];
                if (localVertex >= cluster.vertexCount) {
                    log = "GPUDrivenPreviewPass found out-of-range meshlet triangle index";
                    return false;
                }
                outMeshletTriangles.push_back(localVertex);
            }

            outMeshlets.push_back(GPUDrivenPreviewGpuMeshlet{
                .vertexOffset = meshletVertexOffset,
                .vertexCount = cluster.vertexCount,
                .triangleOffset = meshletTriangleOffset,
                .triangleCount = cluster.triangleCount,
                .primitiveIndex = primitiveIndex,
                .materialIndex = materialIndex,
                .lodLevel = cluster.lodLevel,
                .lodGroupIndex = static_cast<uint32_t>(std::max(cluster.lodGroupIndex, 0)),
                .transformIndex = transformIndex,
            });
        }

        return true;
    }

    static bool loadMeshletScene(
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        std::vector<GPUDrivenPreviewGpuPosition>& outPositions,
        std::vector<GPUDrivenPreviewGpuMeshlet>& outMeshlets,
        std::vector<uint32_t>& outMeshletVertices,
        std::vector<uint32_t>& outMeshletTriangles,
        std::vector<SceneGpuTransform>& outTransforms,
        scene::Bounds& outBounds,
        GPUDrivenPreviewMeshletRange& outBaseMeshletRange,
        std::vector<GPUDrivenPreviewMeshletRange>& outLodLevelRanges,
        std::string& log)
    {
        const std::filesystem::path path = scenePathFromProperties(properties);
        if (runtimeScene == nullptr) {
            log = "GPUDrivenPreviewPass requires a runtime scene resource provider";
            return false;
        }
        const scene::Scene& loadedScene = *runtimeScene;
        if (!loadedScene.bounds().valid) {
            log = "GPUDrivenPreviewPass scene bounds are unavailable";
            return false;
        }

        outPositions.clear();
        outMeshlets.clear();
        outMeshletVertices.clear();
        outMeshletTriangles.clear();
        outBaseMeshletRange = GPUDrivenPreviewMeshletRange{};
        outLodLevelRanges.clear();
        outBounds = loadedScene.bounds();
        outTransforms = buildSceneGpuTransforms(loadedScene);

        std::vector<PrimitiveInstanceRef> primitiveInstances;
        primitiveInstances.reserve(loadedScene.renderNodes().size());
        size_t maxLodLevelCount = 0;
        for (size_t renderNodeIndex = 0; renderNodeIndex < loadedScene.renderNodes().size(); ++renderNodeIndex) {
            const scene::RenderNode& renderNode = loadedScene.renderNodes()[renderNodeIndex];
            if (!renderNode.visible ||
                renderNode.renderPrimitiveIndex < 0 ||
                static_cast<size_t>(renderNode.renderPrimitiveIndex) >= loadedScene.renderPrimitives().size()) {
                continue;
            }

            const scene::RenderPrimitive& primitive =
                loadedScene.renderPrimitives()[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
            if (primitive.mode != kGltfTriangleListMode ||
                primitive.positions.empty() ||
                primitive.meshletClusters.empty()) {
                continue;
            }

            uint32_t positionBase = 0;
            if (!appendPrimitivePositions(primitive, outPositions, positionBase, log)) {
                return false;
            }
            primitiveInstances.push_back(PrimitiveInstanceRef{
                .primitive = &primitive,
                .positionBase = positionBase,
                .primitiveIndex = static_cast<uint32_t>(std::max(renderNode.renderPrimitiveIndex, 0)),
                .materialIndex = static_cast<uint32_t>(std::max(renderNode.materialIndex, 0)),
                .transformIndex = static_cast<uint32_t>(renderNodeIndex),
            });
            maxLodLevelCount = std::max(maxLodLevelCount, primitive.meshletLodLevels.size());
        }

        outBaseMeshletRange.offset = static_cast<uint32_t>(outMeshlets.size());
        for (const PrimitiveInstanceRef& instance : primitiveInstances) {
            const scene::RenderPrimitive& primitive = *instance.primitive;
            if (!appendPrimitiveClusters(
                    primitive,
                    primitive.meshletClusters,
                    primitive.meshletVertices,
                    primitive.meshletTriangles,
                    0,
                    static_cast<uint32_t>(primitive.meshletClusters.size()),
                    instance.positionBase,
                    instance.primitiveIndex,
                    instance.materialIndex,
                    instance.transformIndex,
                    outMeshlets,
                    outMeshletVertices,
                    outMeshletTriangles,
                    log)) {
                return false;
            }
        }
        outBaseMeshletRange.count = static_cast<uint32_t>(outMeshlets.size()) - outBaseMeshletRange.offset;

        outLodLevelRanges.resize(maxLodLevelCount);
        for (uint32_t lodLevel = 0; lodLevel < maxLodLevelCount; ++lodLevel) {
            GPUDrivenPreviewMeshletRange range;
            range.offset = static_cast<uint32_t>(outMeshlets.size());

            for (const PrimitiveInstanceRef& instance : primitiveInstances) {
                const scene::RenderPrimitive& primitive = *instance.primitive;
                if (lodLevel >= primitive.meshletLodLevels.size()) {
                    continue;
                }
                const scene::MeshletLodLevel& level = primitive.meshletLodLevels[lodLevel];
                if (!appendPrimitiveClusters(
                        primitive,
                        primitive.meshletLodClusters,
                        primitive.meshletLodVertices,
                        primitive.meshletLodTriangles,
                        level.clusterOffset,
                        level.clusterCount,
                        instance.positionBase,
                        instance.primitiveIndex,
                        instance.materialIndex,
                        instance.transformIndex,
                        outMeshlets,
                        outMeshletVertices,
                        outMeshletTriangles,
                        log)) {
                    return false;
                }
            }

            range.count = static_cast<uint32_t>(outMeshlets.size()) - range.offset;
            outLodLevelRanges[lodLevel] = range;
        }

        if (outPositions.empty() ||
            outMeshlets.empty() ||
            outMeshletVertices.empty() ||
            outMeshletTriangles.empty()) {
            log = "GPUDrivenPreviewPass found no drawable meshlet geometry in " + path.string();
            return false;
        }
        if (outMeshlets.size() > std::numeric_limits<uint32_t>::max()) {
            log = "GPUDrivenPreviewPass scene has too many meshlets";
            return false;
        }
        if (outBaseMeshletRange.count == 0) {
            log = "GPUDrivenPreviewPass found no base meshlet geometry in " + path.string();
            return false;
        }
        return true;
    }

    static void buildParams(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties,
        const scene::Bounds& drawBounds,
        const GPUDrivenPreviewMeshletRange& baseMeshletRange,
        const std::vector<GPUDrivenPreviewMeshletRange>& lodLevelRanges,
        GPUDrivenPreviewGpuParams& outParams)
    {
        outParams = GPUDrivenPreviewGpuParams{};
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
        const float zNear = std::max(cameraFloat(cameraProperties, "znear", 0.1f), 0.0001f);
        const float zFar = std::max(
            cameraFloat(cameraProperties, "zfar", defaultDistance + radius * 3.0f),
            zNear + 0.001f);
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
        outParams.clipOrtho[3] = kDefaultReversedZ ? 1.0f : 0.0f;
        outParams.clearColor[0] = 0.015f;
        outParams.clearColor[1] = 0.018f;
        outParams.clearColor[2] = 0.024f;
        outParams.clearColor[3] = 1.0f;
        const uint32_t mode = previewModeFromProperties(properties);
        uint32_t selectedLodLevel = 0;
        const GPUDrivenPreviewMeshletRange meshletRange = selectedMeshletRange(
            mode,
            lodLevelFromProperties(properties),
            baseMeshletRange,
            lodLevelRanges,
            selectedLodLevel);
        outParams.mode = mode;
        outParams.meshletOffset = meshletRange.offset;
        outParams.meshletCount = meshletRange.count;
        outParams.selectedLodLevel = selectedLodLevel;
    }

    Result updateParamsBuffer(
        uint32_t width,
        uint32_t height,
        const RenderGraphProperties& properties)
    {
        if (paramsBuffer_ == nullptr || !drawBounds_.valid) {
            return makeError(Error::InvalidArgument);
        }

        GPUDrivenPreviewGpuParams params;
        buildParams(width, height, properties, drawBounds_, baseMeshletRange_, lodLevelRanges_, params);

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
    std::unique_ptr<Buffer> meshletBuffer_;
    std::unique_ptr<Buffer> meshletVertexBuffer_;
    std::unique_ptr<Buffer> meshletTriangleBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<Buffer> transformBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle positionHandle_;
    BindlessHandle meshletHandle_;
    BindlessHandle meshletVertexHandle_;
    BindlessHandle meshletTriangleHandle_;
    BindlessHandle paramsHandle_;
    BindlessHandle transformHandle_;
    std::unique_ptr<ShaderModule> meshShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
    scene::Bounds drawBounds_;
    uint64_t sceneRevision_ = 0;
    GPUDrivenPreviewMeshletRange baseMeshletRange_;
    std::vector<GPUDrivenPreviewMeshletRange> lodLevelRanges_;
    uint32_t drawTaskCount_ = 0;
};

} // namespace

std::unique_ptr<RenderGraphPass> createGPUDrivenPreviewPass()
{
    return std::make_unique<GPUDrivenPreviewPass>();
}

} // namespace metallic::render::builtin_pass
