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
        reflection.addTextureOutput("color", "Deferred meshlet visualization")
            .format = Format::Rgba8Unorm;
        RenderGraphField& visibility = reflection.addTextureOutput(
            "visibility",
            "Mesh shader visibility buffer");
        visibility.colorWrite();
        visibility.format = Format::R32Uint;
        visibility.usage = visibility.usage | TextureUsageBits::Sampled;
        RenderGraphField& depth = reflection.addTextureOutput(
            "depth",
            "Visibility pass depth and HZB source");
        depth.depthStencilWrite();
        depth.usage = depth.usage | TextureUsageBits::Sampled;
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
            runtimeBoolSetting("instanceFrustumCull", "Instance Frustum Cull", true),
            runtimeBoolSetting("instanceHzbCull", "Instance HZB Cull", true),
            runtimeBoolSetting("meshletFrustumCull", "Meshlet Sphere / Frustum Cull", true),
            runtimeBoolSetting("meshletNormalConeCull", "Meshlet Normal Cone Cull", true),
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
        device_ = context.device;
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
        if (visibilityPipeline_ != nullptr &&
            drawTaskCount_ > 0 &&
            sceneRevision_ == runtimeRevision &&
            frameWidth_ == context.width &&
            frameHeight_ == context.height) {
            return {};
        }

        std::vector<GPUDrivenPreviewGpuPosition> positions;
        std::vector<GPUDrivenPreviewGpuMeshlet> meshlets;
        std::vector<uint32_t> meshletVertices;
        std::vector<uint32_t> meshletTriangles;
        std::vector<SceneGpuTransform> transforms;
        std::vector<GPUDrivenPreviewGpuInstance> instances;
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
                instances,
                drawBounds_,
                baseMeshletRange,
                lodLevelRanges,
                log)) {
            return makeError(Error::Failure);
        }

        baseMeshletRange_ = baseMeshletRange;
        lodLevelRanges_ = std::move(lodLevelRanges);
        drawTaskCount_ = maxMeshletRangeCount(baseMeshletRange_, lodLevelRanges_);
        instanceCount_ = static_cast<uint32_t>(instances.size());
        if (drawTaskCount_ == 0 || instanceCount_ == 0) {
            log = "GPUDrivenPreviewPass found no drawable meshlet instances";
            return makeError(Error::Failure);
        }

        frameWidth_ = std::max(context.width, 1u);
        frameHeight_ = std::max(context.height, 1u);
        hzbMipCount_ = computeHzbMipCount(frameWidth_, frameHeight_);
        hzbElementCount_ = computeHzbElementCount(frameWidth_, frameHeight_, hzbMipCount_);
        frameIndex_ = 0;
        hzbValid_ = false;
        previousCameraValid_ = false;
        internalBuffersInitialized_ = false;
        frameBuffersInitialized_ = false;

        GPUDrivenPreviewGpuParams params;
        buildParams(
            context.width,
            context.height,
            properties(),
            drawBounds_,
            baseMeshletRange_,
            lodLevelRanges_,
            instanceCount_,
            hzbMipCount_,
            frameIndex_,
            false,
            nullptr,
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
        result = uploadStorageBuffer(
            *context.device,
            instances.data(),
            static_cast<uint64_t>(instances.size() * sizeof(GPUDrivenPreviewGpuInstance)),
            instanceBuffer_,
            log,
            "GPUDrivenPreviewPass instances");
        if (!result) {
            return result;
        }

        result = createDeviceStorageBuffer(
            *context.device,
            static_cast<uint64_t>(instanceCount_) * sizeof(uint32_t),
            BufferUsageBits::Storage,
            instanceVisibilityBuffer_,
            log,
            "instance visibility");
        if (!result) {
            return result;
        }
        for (uint32_t passIndex = 0; passIndex < 2; ++passIndex) {
            result = createDeviceStorageBuffer(
                *context.device,
                static_cast<uint64_t>(drawTaskCount_) * sizeof(uint32_t),
                BufferUsageBits::Storage,
                visibleMeshletBuffers_[passIndex],
                log,
                "visible meshlets");
            if (!result) {
                return result;
            }
            result = createDeviceStorageBuffer(
                *context.device,
                3u * sizeof(uint32_t),
                BufferUsageBits::Storage | BufferUsageBits::Indirect,
                indirectBuffers_[passIndex],
                log,
                "mesh task indirect arguments");
            if (!result) {
                return result;
            }
            result = createDeviceStorageBuffer(
                *context.device,
                hzbElementCount_ * sizeof(float),
                BufferUsageBits::Storage,
                hzbBuffers_[passIndex],
                log,
                "HZB");
            if (!result) {
                return result;
            }
        }
        result = createDeviceStorageBuffer(
            *context.device,
            static_cast<uint64_t>(frameWidth_) * frameHeight_ * sizeof(uint32_t),
            BufferUsageBits::Storage,
            deferredColorBuffer_,
            log,
            "deferred color");
        if (!result) {
            return result;
        }

        result = context.device->createBindlessHeap(
            BindlessHeapDesc{
                .maxSampledImages = 2,
                .maxBuffers = 15,
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
        result = allocateAndWriteBuffer(*bindlessHeap_, *instanceBuffer_, instanceHandle_, log, "instances");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *instanceVisibilityBuffer_,
            instanceVisibilityHandle_,
            log,
            "instance visibility");
        if (!result) {
            return result;
        }
        for (uint32_t passIndex = 0; passIndex < 2; ++passIndex) {
            result = allocateAndWriteBuffer(
                *bindlessHeap_,
                *visibleMeshletBuffers_[passIndex],
                visibleMeshletHandles_[passIndex],
                log,
                "visible meshlets");
            if (!result) {
                return result;
            }
            result = allocateAndWriteBuffer(
                *bindlessHeap_,
                *indirectBuffers_[passIndex],
                indirectHandles_[passIndex],
                log,
                "indirect arguments");
            if (!result) {
                return result;
            }
            result = allocateAndWriteBuffer(
                *bindlessHeap_,
                *hzbBuffers_[passIndex],
                hzbHandles_[passIndex],
                log,
                "HZB");
            if (!result) {
                return result;
            }
        }
        result = allocateAndWriteBuffer(
            *bindlessHeap_,
            *deferredColorBuffer_,
            deferredColorHandle_,
            log,
            "deferred color");
        if (!result) {
            return result;
        }
        result = bindlessHeap_->allocateSampledImage(depthImageHandle_);
        if (!result || !depthImageHandle_.valid()) {
            log += resultMessage("allocateSampledImage(GPUDrivenPreviewPass depth)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = bindlessHeap_->allocateSampledImage(visibilityImageHandle_);
        if (!result || !visibilityImageHandle_.valid()) {
            log += resultMessage("allocateSampledImage(GPUDrivenPreviewPass visibility)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = createPipelines(*context.device, log);
        if (!result) {
            return result;
        }
        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .meshShader = meshShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::R32Uint,
                .depthStencilFormat = Format::D32Sfloat,
                .depthStencil = DepthStencilState{
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = depthCompareOp(kDefaultReversedZ),
                },
                .usesBindlessHeap = true,
            },
            visibilityPipeline_);
        if (!result || visibilityPipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenPreviewPass visibility)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = compositeVertexShader_.get(),
                .fragmentShader = compositeFragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
                .topology = PrimitiveTopology::TriangleList,
                .usesBindlessHeap = true,
            },
            compositePipeline_);
        if (!result || compositePipeline_ == nullptr) {
            log += resultMessage("createGraphicsPipeline(GPUDrivenPreviewPass composite)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        sceneRevision_ = runtimeRevision;
        previousParams_ = params;
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        Result syncResult = syncRuntimeGeometry(context.runtimeScene());
        if (!syncResult) {
            return syncResult;
        }
        TextureHandle color = context.outputTexture("color");
        TextureHandle visibility = context.outputTexture("visibility");
        TextureHandle depth = context.outputTexture("depth");
        if (!color.valid() ||
            !visibility.valid() ||
            !depth.valid() ||
            bindlessHeap_ == nullptr ||
            visibilityPipeline_ == nullptr ||
            compositePipeline_ == nullptr ||
            drawTaskCount_ == 0) {
            return makeError(Error::InvalidArgument);
        }

        Result result = ensureFrameResources(context.width(), context.height());
        if (!result) {
            return result;
        }
        result = updateParamsBuffer(context.width(), context.height(), context.properties());
        if (!result) {
            return result;
        }

        result = bindlessHeap_->writeSampledImage(
            depthImageHandle_,
            *depth.view(),
            ResourceState::ShaderRead);
        if (!result) {
            return result;
        }
        result = bindlessHeap_->writeSampledImage(
            visibilityImageHandle_,
            *visibility.view(),
            ResourceState::ShaderRead);
        if (!result) {
            return result;
        }

        CommandBuffer& commandBuffer = context.commandBuffer();
        commandBuffer.bindBindlessHeap(*bindlessHeap_);
        initializeInternalBuffers(commandBuffer);

        dispatchCulling(commandBuffer, 0);
        drawVisibility(commandBuffer, visibility, depth, 0, LoadOp::Clear);

        transitionTexture(commandBuffer, *depth.texture(), ResourceState::DepthStencilAttachment, ResourceState::ShaderRead);
        buildHzb(commandBuffer);
        transitionTexture(commandBuffer, *depth.texture(), ResourceState::ShaderRead, ResourceState::DepthStencilAttachment);

        dispatchCulling(commandBuffer, 1);
        drawVisibility(commandBuffer, visibility, depth, 1, LoadOp::Load);

        transitionTexture(commandBuffer, *visibility.texture(), ResourceState::ColorAttachment, ResourceState::ShaderRead);
        transitionTexture(commandBuffer, *depth.texture(), ResourceState::DepthStencilAttachment, ResourceState::ShaderRead);
        buildHzb(commandBuffer);
        dispatchDeferred(commandBuffer);
        barrierBuffer(commandBuffer, *deferredColorBuffer_, ResourceState::General, ResourceState::General);
        transitionTexture(commandBuffer, *visibility.texture(), ResourceState::ShaderRead, ResourceState::ColorAttachment);
        transitionTexture(commandBuffer, *depth.texture(), ResourceState::ShaderRead, ResourceState::DepthStencilAttachment);
        drawComposite(commandBuffer, color);

        hzbValid_ = true;
        ++frameIndex_;
        return {};
    }

private:
    static uint32_t divideRoundUp(uint32_t value, uint32_t divisor)
    {
        return (value + divisor - 1u) / divisor;
    }

    static uint32_t computeHzbMipCount(uint32_t width, uint32_t height)
    {
        uint32_t mipCount = 1;
        while (width > 1 || height > 1) {
            width = std::max(1u, (width + 1u) / 2u);
            height = std::max(1u, (height + 1u) / 2u);
            ++mipCount;
        }
        return mipCount;
    }

    static uint64_t computeHzbElementCount(uint32_t width, uint32_t height, uint32_t mipCount)
    {
        uint64_t elementCount = 0;
        for (uint32_t mipLevel = 0; mipLevel < mipCount; ++mipLevel) {
            elementCount += static_cast<uint64_t>(width) * height;
            width = std::max(1u, (width + 1u) / 2u);
            height = std::max(1u, (height + 1u) / 2u);
        }
        return elementCount;
    }

    static Result createDeviceStorageBuffer(
        Device& device,
        uint64_t byteSize,
        BufferUsageBits usage,
        std::unique_ptr<Buffer>& outBuffer,
        std::string& log,
        std::string_view label)
    {
        if (byteSize == 0) {
            log = std::string("GPUDrivenPreviewPass ") + std::string(label) + " buffer size is zero";
            return makeError(Error::InvalidArgument);
        }
        Result result = device.createBuffer(
            BufferDesc{
                .size = byteSize,
                .structureStride = 0,
                .usage = usage,
                .memoryLocation = MemoryLocation::Device,
            },
            outBuffer);
        if (!result || outBuffer == nullptr) {
            log += resultMessage(std::string("createBuffer(GPUDrivenPreviewPass ") + std::string(label) + ")", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        return {};
    }

    static Result createShader(
        Device& device,
        const char* entryPoint,
        bool meshShader,
        std::unique_ptr<ShaderModule>& outShader,
        std::string& log)
    {
        ShaderCompileResult compileResult;
        const char* capabilities[] = {"spvMeshShadingEXT"};
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kGPUDrivenPreviewShaderModuleName,
                .entryPointName = entryPoint,
                .searchPath = kTriangleShaderSearchPath,
                .capabilities = meshShader ? capabilities : nullptr,
                .capabilityCount = meshShader ? static_cast<uint32_t>(std::size(capabilities)) : 0u,
            },
            compileResult);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += kGPUDrivenPreviewShaderModuleName;
            log += ".";
            log += entryPoint;
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
            outShader);
        if (!result) {
            log += resultMessage(
                std::string("createShaderModule(GPUDrivenPreviewPass ") + entryPoint + ")",
                result);
            log += '\n';
        }
        return result;
    }

    Result createPipelines(Device& device, std::string& log)
    {
        struct ShaderRequest {
            const char* entryPoint = nullptr;
            bool meshShader = false;
            std::unique_ptr<ShaderModule>* shader = nullptr;
        };
        const std::array<ShaderRequest, 9> requests{
            ShaderRequest{kGPUDrivenPreviewMeshEntryPoint, true, &meshShader_},
            ShaderRequest{kGPUDrivenPreviewFragmentEntryPoint, false, &fragmentShader_},
            ShaderRequest{kGPUDrivenPreviewResetEntryPoint, false, &resetShader_},
            ShaderRequest{kGPUDrivenPreviewInstanceCullEntryPoint, false, &instanceCullShader_},
            ShaderRequest{kGPUDrivenPreviewCompactEntryPoint, false, &compactShader_},
            ShaderRequest{kGPUDrivenPreviewHzbEntryPoint, false, &hzbShader_},
            ShaderRequest{kGPUDrivenPreviewDeferredEntryPoint, false, &deferredShader_},
            ShaderRequest{kGPUDrivenPreviewCompositeVertexEntryPoint, false, &compositeVertexShader_},
            ShaderRequest{kGPUDrivenPreviewCompositeFragmentEntryPoint, false, &compositeFragmentShader_},
        };
        for (const ShaderRequest& request : requests) {
            Result result = createShader(
                device,
                request.entryPoint,
                request.meshShader,
                *request.shader,
                log);
            if (!result) {
                return result;
            }
        }

        auto createCompute = [&](ShaderModule& shader, std::unique_ptr<ComputePipeline>& pipeline, const char* label) {
            Result result = device.createComputePipeline(
                ComputePipelineDesc{
                    .computeShader = &shader,
                    .computeEntryPoint = "main",
                    .usesBindlessHeap = true,
                    .bindlessUserPushDataSize = sizeof(GPUDrivenPreviewUserPush),
                },
                pipeline);
            if (!result || pipeline == nullptr) {
                log += resultMessage(std::string("createComputePipeline(GPUDrivenPreviewPass ") + label + ")", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
            return result;
        };

        Result result = createCompute(*resetShader_, resetPipeline_, "reset");
        if (!result) {
            return result;
        }
        result = createCompute(*instanceCullShader_, instanceCullPipeline_, "instance cull");
        if (!result) {
            return result;
        }
        result = createCompute(*compactShader_, compactPipeline_, "compact");
        if (!result) {
            return result;
        }
        result = createCompute(*hzbShader_, hzbPipeline_, "HZB");
        if (!result) {
            return result;
        }
        return createCompute(*deferredShader_, deferredPipeline_, "deferred");
    }

    GPUDrivenPreviewUserPush makePush(uint32_t passIndex = 0, uint32_t mipLevel = 0) const
    {
        return GPUDrivenPreviewUserPush{
            .positionBuffer = positionHandle_.index,
            .meshletBuffer = meshletHandle_.index,
            .meshletVertexBuffer = meshletVertexHandle_.index,
            .meshletTriangleBuffer = meshletTriangleHandle_.index,
            .paramsBuffer = paramsHandle_.index,
            .transformBuffer = transformHandle_.index,
            .instanceBuffer = instanceHandle_.index,
            .instanceVisibilityBuffer = instanceVisibilityHandle_.index,
            .visibleMeshletBuffer0 = visibleMeshletHandles_[0].index,
            .visibleMeshletBuffer1 = visibleMeshletHandles_[1].index,
            .indirectBuffer0 = indirectHandles_[0].index,
            .indirectBuffer1 = indirectHandles_[1].index,
            .hzbBuffer0 = hzbHandles_[0].index,
            .hzbBuffer1 = hzbHandles_[1].index,
            .deferredColorBuffer = deferredColorHandle_.index,
            .depthImage = depthImageHandle_.index,
            .visibilityImage = visibilityImageHandle_.index,
            .passIndex = passIndex,
            .mipLevel = mipLevel,
        };
    }

    static void transitionTexture(
        CommandBuffer& commandBuffer,
        Texture& texture,
        ResourceState before,
        ResourceState after)
    {
        const TextureBarrierDesc barrier{
            .texture = &texture,
            .before = before,
            .after = after,
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = &barrier,
            .textureCount = 1,
        });
    }

    static void barrierBuffer(
        CommandBuffer& commandBuffer,
        Buffer& buffer,
        ResourceState before,
        ResourceState after)
    {
        const BufferBarrierDesc barrier{
            .buffer = &buffer,
            .before = before,
            .after = after,
        };
        commandBuffer.barrier(BarrierDesc{
            .buffers = &barrier,
            .bufferCount = 1,
        });
    }

    void initializeInternalBuffers(CommandBuffer& commandBuffer)
    {
        if (!internalBuffersInitialized_) {
            const std::array<BufferBarrierDesc, 5> barriers{
                BufferBarrierDesc{.buffer = instanceVisibilityBuffer_.get(), .before = ResourceState::Undefined, .after = ResourceState::General},
                BufferBarrierDesc{.buffer = visibleMeshletBuffers_[0].get(), .before = ResourceState::Undefined, .after = ResourceState::General},
                BufferBarrierDesc{.buffer = visibleMeshletBuffers_[1].get(), .before = ResourceState::Undefined, .after = ResourceState::General},
                BufferBarrierDesc{.buffer = indirectBuffers_[0].get(), .before = ResourceState::Undefined, .after = ResourceState::General},
                BufferBarrierDesc{.buffer = indirectBuffers_[1].get(), .before = ResourceState::Undefined, .after = ResourceState::General},
            };
            commandBuffer.barrier(BarrierDesc{
                .buffers = barriers.data(),
                .bufferCount = static_cast<uint32_t>(barriers.size()),
            });
            internalBuffersInitialized_ = true;
        }

        if (!frameBuffersInitialized_) {
            const std::array<BufferBarrierDesc, 3> barriers{
                BufferBarrierDesc{.buffer = hzbBuffers_[0].get(), .before = ResourceState::Undefined, .after = ResourceState::General},
                BufferBarrierDesc{.buffer = hzbBuffers_[1].get(), .before = ResourceState::Undefined, .after = ResourceState::General},
                BufferBarrierDesc{.buffer = deferredColorBuffer_.get(), .before = ResourceState::Undefined, .after = ResourceState::General},
            };
            commandBuffer.barrier(BarrierDesc{
                .buffers = barriers.data(),
                .bufferCount = static_cast<uint32_t>(barriers.size()),
            });
            frameBuffersInitialized_ = true;
        }
    }

    void dispatchCulling(CommandBuffer& commandBuffer, uint32_t passIndex)
    {
        GPUDrivenPreviewUserPush push = makePush(passIndex);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.bindComputePipeline(*resetPipeline_);
        commandBuffer.dispatch(1, 1, 1);

        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.bindComputePipeline(*instanceCullPipeline_);
        commandBuffer.dispatch(divideRoundUp(instanceCount_, 64u), 1, 1);

        const std::array<BufferBarrierDesc, 2> cullBarriers{
            BufferBarrierDesc{.buffer = indirectBuffers_[passIndex].get(), .before = ResourceState::General, .after = ResourceState::General},
            BufferBarrierDesc{.buffer = instanceVisibilityBuffer_.get(), .before = ResourceState::General, .after = ResourceState::General},
        };
        commandBuffer.barrier(BarrierDesc{
            .buffers = cullBarriers.data(),
            .bufferCount = static_cast<uint32_t>(cullBarriers.size()),
        });

        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.bindComputePipeline(*compactPipeline_);
        commandBuffer.dispatch(divideRoundUp(activeMeshletCount_, 64u), 1, 1);

        const std::array<BufferBarrierDesc, 2> compactBarriers{
            BufferBarrierDesc{.buffer = visibleMeshletBuffers_[passIndex].get(), .before = ResourceState::General, .after = ResourceState::General},
            BufferBarrierDesc{.buffer = indirectBuffers_[passIndex].get(), .before = ResourceState::General, .after = ResourceState::General},
        };
        commandBuffer.barrier(BarrierDesc{
            .buffers = compactBarriers.data(),
            .bufferCount = static_cast<uint32_t>(compactBarriers.size()),
        });
    }

    void drawVisibility(
        CommandBuffer& commandBuffer,
        TextureHandle visibility,
        TextureHandle depth,
        uint32_t passIndex,
        LoadOp loadOp)
    {
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = frameWidth_,
            .height = frameHeight_,
        };
        const RenderingAttachmentDesc visibilityAttachment{
            .view = visibility.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 0.0f},
        };
        const RenderingAttachmentDesc depthAttachment{
            .view = depth.view(),
            .state = ResourceState::DepthStencilAttachment,
            .loadOp = loadOp,
            .storeOp = StoreOp::Store,
            .clearDepth = depthClearValue(kDefaultReversedZ),
        };
        commandBuffer.beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &visibilityAttachment,
            .colorAttachmentCount = 1,
            .depthStencilAttachment = &depthAttachment,
        });
        commandBuffer.setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frameWidth_),
            .height = static_cast<float>(frameHeight_),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        commandBuffer.setScissor(renderArea);
        commandBuffer.bindGraphicsPipeline(*visibilityPipeline_);
        const GPUDrivenPreviewUserPush push = makePush(passIndex);
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.drawMeshTasksIndirect(*indirectBuffers_[passIndex]);
        commandBuffer.endRendering();
    }

    void buildHzb(CommandBuffer& commandBuffer)
    {
        Buffer& currentHzb = *hzbBuffers_[frameIndex_ & 1u];
        barrierBuffer(commandBuffer, currentHzb, ResourceState::General, ResourceState::General);
        uint32_t mipWidth = frameWidth_;
        uint32_t mipHeight = frameHeight_;
        for (uint32_t mipLevel = 0; mipLevel < hzbMipCount_; ++mipLevel) {
            const GPUDrivenPreviewUserPush push = makePush(0, mipLevel);
            commandBuffer.pushBindlessData(&push, sizeof(push));
            commandBuffer.bindComputePipeline(*hzbPipeline_);
            commandBuffer.dispatch(divideRoundUp(mipWidth, 8u), divideRoundUp(mipHeight, 8u), 1);
            barrierBuffer(commandBuffer, currentHzb, ResourceState::General, ResourceState::General);
            mipWidth = std::max(1u, (mipWidth + 1u) / 2u);
            mipHeight = std::max(1u, (mipHeight + 1u) / 2u);
        }
    }

    void dispatchDeferred(CommandBuffer& commandBuffer)
    {
        const GPUDrivenPreviewUserPush push = makePush();
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.bindComputePipeline(*deferredPipeline_);
        commandBuffer.dispatch(divideRoundUp(frameWidth_, 8u), divideRoundUp(frameHeight_, 8u), 1);
    }

    void drawComposite(CommandBuffer& commandBuffer, TextureHandle color)
    {
        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = frameWidth_,
            .height = frameHeight_,
        };
        const RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.015f, 0.018f, 0.024f, 1.0f},
        };
        commandBuffer.beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        commandBuffer.setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(frameWidth_),
            .height = static_cast<float>(frameHeight_),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        commandBuffer.setScissor(renderArea);
        commandBuffer.bindGraphicsPipeline(*compositePipeline_);
        const GPUDrivenPreviewUserPush push = makePush();
        commandBuffer.pushBindlessData(&push, sizeof(push));
        commandBuffer.draw(3);
        commandBuffer.endRendering();
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
        hzbValid_ = false;
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

    Result ensureFrameResources(uint32_t width, uint32_t height)
    {
        width = std::max(width, 1u);
        height = std::max(height, 1u);
        if (frameWidth_ == width && frameHeight_ == height) {
            return {};
        }
        if (device_ == nullptr || bindlessHeap_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const uint32_t mipCount = computeHzbMipCount(width, height);
        const uint64_t elementCount = computeHzbElementCount(width, height, mipCount);
        std::array<std::unique_ptr<Buffer>, 2> resizedHzbBuffers;
        std::unique_ptr<Buffer> resizedDeferredColorBuffer;
        std::string log;
        for (uint32_t bufferIndex = 0; bufferIndex < resizedHzbBuffers.size(); ++bufferIndex) {
            Result result = createDeviceStorageBuffer(
                *device_,
                elementCount * sizeof(float),
                BufferUsageBits::Storage,
                resizedHzbBuffers[bufferIndex],
                log,
                "resized HZB");
            if (!result) {
                spdlog::error("[GPUDrivenPreviewPass] {}", log);
                return result;
            }
        }
        Result result = createDeviceStorageBuffer(
            *device_,
            static_cast<uint64_t>(width) * height * sizeof(uint32_t),
            BufferUsageBits::Storage,
            resizedDeferredColorBuffer,
            log,
            "resized deferred color");
        if (!result) {
            spdlog::error("[GPUDrivenPreviewPass] {}", log);
            return result;
        }

        hzbBuffers_ = std::move(resizedHzbBuffers);
        deferredColorBuffer_ = std::move(resizedDeferredColorBuffer);
        for (uint32_t bufferIndex = 0; bufferIndex < hzbBuffers_.size(); ++bufferIndex) {
            result = bindlessHeap_->writeStorageBuffer(hzbHandles_[bufferIndex], *hzbBuffers_[bufferIndex]);
            if (!result) {
                return result;
            }
        }
        result = bindlessHeap_->writeStorageBuffer(deferredColorHandle_, *deferredColorBuffer_);
        if (!result) {
            return result;
        }

        spdlog::info(
            "[GPUDrivenPreviewPass] Resized frame resources {}x{} -> {}x{}",
            frameWidth_,
            frameHeight_,
            width,
            height);
        frameWidth_ = width;
        frameHeight_ = height;
        hzbMipCount_ = mipCount;
        hzbElementCount_ = elementCount;
        frameIndex_ = 0;
        hzbValid_ = false;
        previousCameraValid_ = false;
        frameBuffersInitialized_ = false;
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
        uint32_t instanceIndex = 0;
        uint32_t meshletFlags = 0;
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
        uint32_t instanceIndex,
        uint32_t meshletFlags,
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
                .instanceIndex = instanceIndex,
                .flags = meshletFlags,
                .boundingSphere = {
                    cluster.boundingSphereCenter.x,
                    cluster.boundingSphereCenter.y,
                    cluster.boundingSphereCenter.z,
                    std::max(cluster.boundingSphereRadius, 0.0f),
                },
                .coneApexCutoff = {
                    cluster.coneApex.x,
                    cluster.coneApex.y,
                    cluster.coneApex.z,
                    cluster.coneCutoff,
                },
                .coneAxis = {
                    cluster.coneAxis.x,
                    cluster.coneAxis.y,
                    cluster.coneAxis.z,
                    0.0f,
                },
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
        std::vector<GPUDrivenPreviewGpuInstance>& outInstances,
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
        outInstances.clear();
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
            if (!primitive.localBounds.valid ||
                outInstances.size() >= std::numeric_limits<uint32_t>::max()) {
                log = "GPUDrivenPreviewPass found invalid or unaddressable instance bounds";
                return false;
            }
            const uint32_t materialIndex = static_cast<uint32_t>(std::max(renderNode.materialIndex, 0));
            const uint32_t instanceIndex = static_cast<uint32_t>(outInstances.size());
            const float3 instanceCenter = primitive.localBounds.center();
            outInstances.push_back(GPUDrivenPreviewGpuInstance{
                .boundingSphere = {
                    instanceCenter.x,
                    instanceCenter.y,
                    instanceCenter.z,
                    std::max(primitive.localBounds.radius(), 0.000001f),
                },
                .transformIndex = static_cast<uint32_t>(renderNodeIndex),
                .primitiveIndex = static_cast<uint32_t>(std::max(renderNode.renderPrimitiveIndex, 0)),
            });
            const bool doubleSided =
                renderNode.materialIndex >= 0 &&
                static_cast<size_t>(renderNode.materialIndex) < loadedScene.materials().size() &&
                loadedScene.materials()[static_cast<size_t>(renderNode.materialIndex)].doubleSided;
            primitiveInstances.push_back(PrimitiveInstanceRef{
                .primitive = &primitive,
                .positionBase = positionBase,
                .primitiveIndex = static_cast<uint32_t>(std::max(renderNode.renderPrimitiveIndex, 0)),
                .materialIndex = materialIndex,
                .transformIndex = static_cast<uint32_t>(renderNodeIndex),
                .instanceIndex = instanceIndex,
                .meshletFlags = doubleSided ? 1u : 0u,
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
                    instance.instanceIndex,
                    instance.meshletFlags,
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
                        instance.instanceIndex,
                        instance.meshletFlags,
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
            outMeshletTriangles.empty() ||
            outInstances.empty()) {
            log = "GPUDrivenPreviewPass found no drawable meshlet geometry in " + path.string();
            return false;
        }
        if (outMeshlets.size() >= std::numeric_limits<uint32_t>::max()) {
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
        uint32_t instanceCount,
        uint32_t hzbMipCount,
        uint32_t frameIndex,
        bool hzbValid,
        const GPUDrivenPreviewGpuParams* previousParams,
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
        outParams.instanceCount = instanceCount;
        outParams.width = std::max(width, 1u);
        outParams.height = std::max(height, 1u);
        outParams.hzbMipCount = std::max(hzbMipCount, 1u);
        outParams.frameIndex = frameIndex;
        outParams.hzbValid = hzbValid ? 1u : 0u;
        outParams.cullingFlags = 0;
        if (boolProperty(&properties, "instanceFrustumCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullInstanceFrustum;
        }
        if (boolProperty(&properties, "instanceHzbCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullInstanceHzb;
        }
        if (boolProperty(&properties, "meshletFrustumCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullMeshletFrustum;
        }
        if (boolProperty(&properties, "meshletNormalConeCull", true)) {
            outParams.cullingFlags |= kGPUDrivenPreviewCullMeshletNormalCone;
        }

        const GPUDrivenPreviewGpuParams& previous = previousParams != nullptr ? *previousParams : outParams;
        std::memcpy(outParams.previousEye, previous.eye, sizeof(outParams.previousEye));
        std::memcpy(outParams.previousCenter, previous.center, sizeof(outParams.previousCenter));
        std::memcpy(outParams.previousUpProjection, previous.upProjection, sizeof(outParams.previousUpProjection));
        std::memcpy(outParams.previousViewport, previous.viewport, sizeof(outParams.previousViewport));
        std::memcpy(outParams.previousClipOrtho, previous.clipOrtho, sizeof(outParams.previousClipOrtho));
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
        buildParams(
            width,
            height,
            properties,
            drawBounds_,
            baseMeshletRange_,
            lodLevelRanges_,
            instanceCount_,
            hzbMipCount_,
            frameIndex_,
            hzbValid_,
            previousCameraValid_ ? &previousParams_ : nullptr,
            params);

        void* mapped = paramsBuffer_->map();
        if (mapped == nullptr) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, &params, sizeof(params));
        paramsBuffer_->flush(0, sizeof(params));
        paramsBuffer_->unmap();
        activeMeshletCount_ = params.meshletCount;
        previousParams_ = params;
        previousCameraValid_ = true;
        return {};
    }

    std::unique_ptr<Buffer> positionBuffer_;
    std::unique_ptr<Buffer> meshletBuffer_;
    std::unique_ptr<Buffer> meshletVertexBuffer_;
    std::unique_ptr<Buffer> meshletTriangleBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<Buffer> transformBuffer_;
    std::unique_ptr<Buffer> instanceBuffer_;
    std::unique_ptr<Buffer> instanceVisibilityBuffer_;
    std::array<std::unique_ptr<Buffer>, 2> visibleMeshletBuffers_;
    std::array<std::unique_ptr<Buffer>, 2> indirectBuffers_;
    std::array<std::unique_ptr<Buffer>, 2> hzbBuffers_;
    std::unique_ptr<Buffer> deferredColorBuffer_;
    Device* device_ = nullptr;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle positionHandle_;
    BindlessHandle meshletHandle_;
    BindlessHandle meshletVertexHandle_;
    BindlessHandle meshletTriangleHandle_;
    BindlessHandle paramsHandle_;
    BindlessHandle transformHandle_;
    BindlessHandle instanceHandle_;
    BindlessHandle instanceVisibilityHandle_;
    std::array<BindlessHandle, 2> visibleMeshletHandles_;
    std::array<BindlessHandle, 2> indirectHandles_;
    std::array<BindlessHandle, 2> hzbHandles_;
    BindlessHandle deferredColorHandle_;
    BindlessHandle depthImageHandle_;
    BindlessHandle visibilityImageHandle_;
    std::unique_ptr<ShaderModule> meshShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<ShaderModule> resetShader_;
    std::unique_ptr<ShaderModule> instanceCullShader_;
    std::unique_ptr<ShaderModule> compactShader_;
    std::unique_ptr<ShaderModule> hzbShader_;
    std::unique_ptr<ShaderModule> deferredShader_;
    std::unique_ptr<ShaderModule> compositeVertexShader_;
    std::unique_ptr<ShaderModule> compositeFragmentShader_;
    std::unique_ptr<GraphicsPipeline> visibilityPipeline_;
    std::unique_ptr<GraphicsPipeline> compositePipeline_;
    std::unique_ptr<ComputePipeline> resetPipeline_;
    std::unique_ptr<ComputePipeline> instanceCullPipeline_;
    std::unique_ptr<ComputePipeline> compactPipeline_;
    std::unique_ptr<ComputePipeline> hzbPipeline_;
    std::unique_ptr<ComputePipeline> deferredPipeline_;
    scene::Bounds drawBounds_;
    uint64_t sceneRevision_ = 0;
    GPUDrivenPreviewMeshletRange baseMeshletRange_;
    std::vector<GPUDrivenPreviewMeshletRange> lodLevelRanges_;
    GPUDrivenPreviewGpuParams previousParams_;
    uint32_t drawTaskCount_ = 0;
    uint32_t activeMeshletCount_ = 0;
    uint32_t instanceCount_ = 0;
    uint32_t frameWidth_ = 1;
    uint32_t frameHeight_ = 1;
    uint32_t hzbMipCount_ = 1;
    uint64_t hzbElementCount_ = 1;
    uint32_t frameIndex_ = 0;
    bool hzbValid_ = false;
    bool previousCameraValid_ = false;
    bool internalBuffersInitialized_ = false;
    bool frameBuffersInitialized_ = false;
};

} // namespace

std::unique_ptr<RenderGraphPass> createGPUDrivenPreviewPass()
{
    return std::make_unique<GPUDrivenPreviewPass>();
}

} // namespace metallic::render::builtin_pass
