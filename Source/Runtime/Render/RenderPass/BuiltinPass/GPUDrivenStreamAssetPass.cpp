#include "Runtime/Render/MeshletStreamResidency.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>

namespace metallic::render::builtin_pass {
namespace {

struct GPUDrivenStreamAssetGpuDrawItem {
    uint32_t pageSlot = 0;
    uint32_t clusterIndex = 0;
    uint32_t pageIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t materialIndex = 0;
    uint32_t colorSeed = 0;
    uint32_t padding0 = 0;
    float world0[4] = {};
    float world1[4] = {};
    float world2[4] = {};
    float world3[4] = {};
};

struct GPUDrivenStreamAssetGpuPageTableEntry {
    uint32_t slot = UINT32_MAX;
    uint32_t state = 0;
    uint32_t payloadBytes = 0;
    uint32_t lodLevel = 0;
};

struct GPUDrivenStreamAssetGpuParams {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    float clearColor[4] = {};
    uint32_t debugColorMode = kGPUDrivenStreamAssetDebugPage;
    uint32_t pageStrideWords = 0;
    uint32_t drawItemCount = 0;
    uint32_t padding0 = 0;
};

struct GPUDrivenStreamAssetUserPush {
    uint32_t pageBuffer = 0;
    uint32_t drawItemBuffer = 0;
    uint32_t pageTableBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
    uint32_t padding3 = 0;
};

static_assert(sizeof(GPUDrivenStreamAssetGpuDrawItem) == 96);
static_assert(sizeof(GPUDrivenStreamAssetGpuParams) == 112);

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

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
        return kGPUDrivenStreamAssetDebugPage;
    }
    const std::string mode = iter->get<std::string>();
    if (mode == "lod") {
        return kGPUDrivenStreamAssetDebugLod;
    }
    if (mode == "primitive") {
        return kGPUDrivenStreamAssetDebugPrimitive;
    }
    return kGPUDrivenStreamAssetDebugPage;
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

float3 transformPoint(const float matrix[16], const float3& point)
{
    return float3(
        matrix[0] * point.x + matrix[4] * point.y + matrix[8] * point.z + matrix[12],
        matrix[1] * point.x + matrix[5] * point.y + matrix[9] * point.z + matrix[13],
        matrix[2] * point.x + matrix[6] * point.y + matrix[10] * point.z + matrix[14]);
}

void includeTransformedBounds(scene::Bounds& outBounds, const scene::MeshletStreamBounds& bounds, const float matrix[16])
{
    if (bounds.valid == 0) {
        return;
    }
    const float3 minBounds(bounds.min[0], bounds.min[1], bounds.min[2]);
    const float3 maxBounds(bounds.max[0], bounds.max[1], bounds.max[2]);
    for (uint32_t z = 0; z < 2; ++z) {
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                const float3 corner(
                    x == 0 ? minBounds.x : maxBounds.x,
                    y == 0 ? minBounds.y : maxBounds.y,
                    z == 0 ? minBounds.z : maxBounds.z);
                outBounds.include(transformPoint(matrix, corner));
            }
        }
    }
}

scene::Bounds computeDrawBounds(const scene::MeshletStreamAsset& asset)
{
    scene::Bounds bounds;
    const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset.primitives();
    for (const scene::MeshletStreamInstanceInfo& instance : asset.instances()) {
        if (instance.visible == 0 || instance.primitiveIndex >= primitives.size()) {
            continue;
        }
        includeTransformedBounds(bounds, primitives[instance.primitiveIndex].bounds, instance.worldMatrix);
    }
    return bounds;
}

Result createHostStorageBuffer(
    Device& device,
    uint64_t byteSize,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log,
    std::string_view label)
{
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
    return {};
}

Result updateHostBuffer(Buffer& buffer, const void* data, uint64_t byteSize)
{
    if (byteSize > buffer.desc().size || (byteSize > 0 && data == nullptr)) {
        return makeError(Error::InvalidArgument);
    }
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    if (byteSize > 0) {
        std::memcpy(mapped, data, static_cast<size_t>(byteSize));
        buffer.flush(0, byteSize);
    }
    buffer.unmap();
    return {};
}

Result allocateAndWriteBuffer(
    BindlessHeap& heap,
    Buffer& buffer,
    BindlessHandle& outHandle,
    std::string& log,
    std::string_view label)
{
    Result result = heap.allocateBuffer(outHandle);
    if (!result || !outHandle.valid()) {
        log += resultMessage(std::string("allocateBuffer(") + std::string(label) + ")", result);
        log += '\n';
        return result ? makeError(Error::Failure) : result;
    }
    result = heap.writeStorageBuffer(outHandle, buffer);
    if (!result) {
        log += resultMessage(std::string("writeStorageBuffer(") + std::string(label) + ")", result);
        log += '\n';
    }
    return result;
}

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

        const std::filesystem::path scenePath = scenePathFromProperties(properties());
        const std::filesystem::path streamAssetPath = streamAssetPathFromProperties(properties(), scenePath);
        const bool autoBuild = boolProperty(properties(), "autoBuildStreamAsset", true);

        std::string reason;
        scene::MeshletStreamAsset openedAsset;
        if (!openedAsset.open(streamAssetPath, reason) || !openedAsset.isCurrentForSource(scenePath)) {
            if (!autoBuild) {
                log = "GPUDrivenStreamAssetPass failed to open current streamasset: " + reason;
                return makeError(Error::Failure);
            }

            scene::Scene loadedScene;
            if (!loadedScene.load(scenePath)) {
                log = "GPUDrivenStreamAssetPass failed to load source scene: " + loadedScene.lastLoadResult().error;
                return makeError(Error::Failure);
            }
            if (!scene::buildMeshletStreamAsset(
                    scene::MeshletStreamAssetBuildDesc{
                        .scene = &loadedScene,
                        .sourcePath = scenePath,
                        .outputPath = streamAssetPath,
                    },
                    reason)) {
                log = "GPUDrivenStreamAssetPass failed to build streamasset: " + reason;
                return makeError(Error::Failure);
            }
            if (!openedAsset.open(streamAssetPath, reason)) {
                log = "GPUDrivenStreamAssetPass failed to open built streamasset: " + reason;
                return makeError(Error::Failure);
            }
        }

        asset_ = std::move(openedAsset);
        drawBounds_ = computeDrawBounds(asset_);
        if (!drawBounds_.valid) {
            log = "GPUDrivenStreamAssetPass streamasset bounds are unavailable";
            return makeError(Error::Failure);
        }

        maxResidentPages_ = uintProperty(properties(), "maxResidentPages", 4096);
        maxPageUploadsPerFrame_ = uintProperty(properties(), "maxPageUploadsPerFrame", 64);
        const uint64_t pageStride = alignUp(asset_.maxPagePayloadBytes(), 256);

        Result result = context.device->createBuffer(
            BufferDesc{
                .size = pageStride * maxResidentPages_,
                .structureStride = 0,
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::Device,
            },
            pageBuffer_);
        if (!result || pageBuffer_ == nullptr) {
            log += resultMessage("createBuffer(GPUDrivenStreamAsset pages)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        pageBufferState_ = ResourceState::Undefined;

        std::vector<uint32_t> fallbackPages;
        for (const scene::MeshletStreamPrimitiveInfo& primitive : asset_.primitives()) {
            for (uint32_t page = 0; page < primitive.fallbackPageCount; ++page) {
                fallbackPages.push_back(primitive.fallbackPageOffset + page);
            }
        }

        if (!residency_.initialize(
                MeshletStreamResidencyDesc{
                    .asset = &asset_,
                    .maxResidentPages = maxResidentPages_,
                    .queuedFrameCount = 3,
                    .pageStride = pageStride,
                },
                reason) ||
            !residency_.lockFallbackPages(fallbackPages, reason)) {
            log = "GPUDrivenStreamAssetPass residency initialization failed: " + reason;
            return makeError(Error::Failure);
        }

        maxDrawItems_ = computeMaxDrawItems();
        if (maxDrawItems_ == 0) {
            log = "GPUDrivenStreamAssetPass streamasset has no drawable clusters";
            return makeError(Error::Failure);
        }

        result = createHostStorageBuffer(
            *context.device,
            static_cast<uint64_t>(maxDrawItems_) * sizeof(GPUDrivenStreamAssetGpuDrawItem),
            drawItemBuffer_,
            log,
            "GPUDrivenStreamAsset draw items");
        if (!result) {
            return result;
        }
        result = createHostStorageBuffer(
            *context.device,
            static_cast<uint64_t>(asset_.pageCount()) * sizeof(GPUDrivenStreamAssetGpuPageTableEntry),
            pageTableBuffer_,
            log,
            "GPUDrivenStreamAsset page table");
        if (!result) {
            return result;
        }
        result = createHostStorageBuffer(
            *context.device,
            sizeof(GPUDrivenStreamAssetGpuParams),
            paramsBuffer_,
            log,
            "GPUDrivenStreamAsset params");
        if (!result) {
            return result;
        }

        result = context.device->createBindlessHeap(
            BindlessHeapDesc{
                .maxSamplers = 0,
                .maxSampledImages = 0,
                .maxBuffers = 4,
            },
            bindlessHeap_);
        if (!result || bindlessHeap_ == nullptr) {
            log += resultMessage("createBindlessHeap(GPUDrivenStreamAssetPass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *pageBuffer_, pageHandle_, log, "streamasset pages");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *drawItemBuffer_, drawItemHandle_, log, "streamasset draw items");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *pageTableBuffer_, pageTableHandle_, log, "streamasset page table");
        if (!result) {
            return result;
        }
        result = allocateAndWriteBuffer(*bindlessHeap_, *paramsBuffer_, paramsHandle_, log, "streamasset params");
        if (!result) {
            return result;
        }

        ShaderCompileResult meshCompile;
        const char* capabilities[] = {"spvMeshShadingEXT"};
        result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kGPUDrivenStreamAssetShaderModuleName,
                .entryPointName = kGPUDrivenStreamAssetMeshEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
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

        ShaderCompileResult fragmentCompile;
        result = compileSlangShader(
            kGPUDrivenStreamAssetShaderModuleName,
            kGPUDrivenStreamAssetFragmentEntryPoint,
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
        if (!result || meshShader_ == nullptr) {
            log += resultMessage("createShaderModule(GPUDrivenStreamAsset mesh)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
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
            bindlessHeap_ == nullptr ||
            pipeline_ == nullptr ||
            pageBuffer_ == nullptr ||
            drawItemBuffer_ == nullptr ||
            pageTableBuffer_ == nullptr ||
            paramsBuffer_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        residency_.beginFrame();
        buildFrameDrawItems(context.properties());
        const uint32_t uploadCount =
            residency_.processUploads(*context.streamer(), *pageBuffer_, maxPageUploadsPerFrame_);
        Result result = updatePageTableBuffer();
        if (!result) {
            return result;
        }
        result = updateDrawItemBuffer();
        if (!result) {
            return result;
        }
        result = updateParamsBuffer(context.width(), context.height(), context.properties());
        if (!result) {
            return result;
        }

        const bool needsShaderRead = !drawItems_.empty();
        if (needsShaderRead && pageBufferState_ != ResourceState::ShaderRead) {
            BufferBarrierDesc barrier{
                .buffer = pageBuffer_.get(),
                .before = pageBufferState_,
                .after = ResourceState::ShaderRead,
                .offset = 0,
                .size = pageBuffer_->desc().size,
            };
            context.commandBuffer().barrier(BarrierDesc{
                .buffers = &barrier,
                .bufferCount = 1,
            });
            pageBufferState_ = ResourceState::ShaderRead;
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
        if (!drawItems_.empty()) {
            context.commandBuffer().bindBindlessHeap(*bindlessHeap_);
            context.commandBuffer().bindGraphicsPipeline(*pipeline_);
            const GPUDrivenStreamAssetUserPush push{
                .pageBuffer = pageHandle_.index,
                .drawItemBuffer = drawItemHandle_.index,
                .pageTableBuffer = pageTableHandle_.index,
                .paramsBuffer = paramsHandle_.index,
            };
            context.commandBuffer().pushBindlessData(&push, sizeof(push));
            context.commandBuffer().drawMeshTasks(static_cast<uint32_t>(drawItems_.size()));
        }
        context.commandBuffer().endRendering();

        if (uploadCount > 0 && pageBufferState_ != ResourceState::TransferDestination) {
            BufferBarrierDesc barrier{
                .buffer = pageBuffer_.get(),
                .before = pageBufferState_,
                .after = ResourceState::TransferDestination,
                .offset = 0,
                .size = pageBuffer_->desc().size,
            };
            context.commandBuffer().barrier(BarrierDesc{
                .buffers = &barrier,
                .bufferCount = 1,
            });
            pageBufferState_ = ResourceState::TransferDestination;
        }
        return {};
    }

private:
    uint32_t computeMaxDrawItems() const
    {
        uint64_t total = 0;
        const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
        const std::span<const scene::MeshletStreamLodLevelInfo> lodLevels = asset_.lodLevels();
        for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
            if (instance.visible == 0 || instance.primitiveIndex >= primitives.size()) {
                continue;
            }
            const scene::MeshletStreamPrimitiveInfo& primitive = primitives[instance.primitiveIndex];
            uint32_t maxClusters = 0;
            for (uint32_t lod = 0; lod < primitive.lodLevelCount; ++lod) {
                const scene::MeshletStreamLodLevelInfo& lodInfo = lodLevels[primitive.lodLevelOffset + lod];
                maxClusters = std::max(maxClusters, lodInfo.clusterCount);
            }
            total += maxClusters;
            if (total > std::numeric_limits<uint32_t>::max()) {
                return 0;
            }
        }
        return static_cast<uint32_t>(total);
    }

    void appendPageDrawItems(
        const scene::MeshletStreamInstanceInfo& instance,
        const scene::MeshletStreamPageInfo& page,
        uint32_t pageIndex)
    {
        const uint32_t slot = residency_.slotForPage(pageIndex);
        if (slot == UINT32_MAX || drawItems_.size() >= maxDrawItems_) {
            return;
        }
        for (uint32_t cluster = 0; cluster < page.clusterCount && drawItems_.size() < maxDrawItems_; ++cluster) {
            GPUDrivenStreamAssetGpuDrawItem item;
            item.pageSlot = slot;
            item.clusterIndex = cluster;
            item.pageIndex = pageIndex;
            item.primitiveIndex = page.primitiveIndex;
            item.lodLevel = page.lodLevel;
            item.materialIndex = instance.materialIndex;
            item.colorSeed = pageIndex * 131u + cluster;
            for (uint32_t row = 0; row < 4; ++row) {
                item.world0[row] = instance.worldMatrix[0 + row];
                item.world1[row] = instance.worldMatrix[4 + row];
                item.world2[row] = instance.worldMatrix[8 + row];
                item.world3[row] = instance.worldMatrix[12 + row];
            }
            drawItems_.push_back(item);
        }
    }

    bool appendResidentPageRange(
        const scene::MeshletStreamInstanceInfo& instance,
        uint32_t pageOffset,
        uint32_t pageCount)
    {
        const std::span<const scene::MeshletStreamPageInfo> pages = asset_.pages();
        bool allResident = true;
        for (uint32_t page = 0; page < pageCount; ++page) {
            const uint32_t pageIndex = pageOffset + page;
            if (!residency_.pageResident(pageIndex)) {
                allResident = false;
                residency_.requestPage(pageIndex);
            }
        }
        if (!allResident) {
            return false;
        }
        for (uint32_t page = 0; page < pageCount; ++page) {
            const uint32_t pageIndex = pageOffset + page;
            appendPageDrawItems(instance, pages[pageIndex], pageIndex);
        }
        return true;
    }

    void buildFrameDrawItems(const RenderGraphProperties& properties)
    {
        drawItems_.clear();
        const uint32_t selectedLod = selectedLodProperty(properties);
        const std::span<const scene::MeshletStreamPrimitiveInfo> primitives = asset_.primitives();
        const std::span<const scene::MeshletStreamLodLevelInfo> lodLevels = asset_.lodLevels();

        for (const scene::MeshletStreamInstanceInfo& instance : asset_.instances()) {
            if (instance.visible == 0 || instance.primitiveIndex >= primitives.size()) {
                continue;
            }
            const scene::MeshletStreamPrimitiveInfo& primitive = primitives[instance.primitiveIndex];
            if (primitive.lodLevelCount == 0 || primitive.fallbackPageCount == 0) {
                continue;
            }
            const uint32_t localLod = std::min(selectedLod, primitive.lodLevelCount - 1u);
            const scene::MeshletStreamLodLevelInfo& lodInfo = lodLevels[primitive.lodLevelOffset + localLod];
            if (!appendResidentPageRange(instance, lodInfo.pageOffset, lodInfo.pageCount)) {
                appendResidentPageRange(instance, primitive.fallbackPageOffset, primitive.fallbackPageCount);
            }
        }
    }

    Result updatePageTableBuffer()
    {
        pageTable_.resize(asset_.pageCount());
        const std::span<const scene::MeshletStreamPageInfo> pages = asset_.pages();
        for (uint32_t pageIndex = 0; pageIndex < asset_.pageCount(); ++pageIndex) {
            const MeshletStreamPageResidencyState state = residency_.pageState(pageIndex);
            pageTable_[pageIndex] = GPUDrivenStreamAssetGpuPageTableEntry{
                .slot = residency_.slotForPage(pageIndex),
                .state = static_cast<uint32_t>(state),
                .payloadBytes = static_cast<uint32_t>(pages[pageIndex].payloadSize),
                .lodLevel = pages[pageIndex].lodLevel,
            };
        }
        return updateHostBuffer(
            *pageTableBuffer_,
            pageTable_.data(),
            static_cast<uint64_t>(pageTable_.size() * sizeof(GPUDrivenStreamAssetGpuPageTableEntry)));
    }

    Result updateDrawItemBuffer()
    {
        return updateHostBuffer(
            *drawItemBuffer_,
            drawItems_.data(),
            static_cast<uint64_t>(drawItems_.size() * sizeof(GPUDrivenStreamAssetGpuDrawItem)));
    }

    Result updateParamsBuffer(uint32_t width, uint32_t height, const RenderGraphProperties& properties)
    {
        GPUDrivenStreamAssetGpuParams params;
        const float3 center = drawBounds_.center();
        const float radius = std::max(drawBounds_.radius(), 1.0f);
        const RenderGraphProperties* camera = cameraPropertiesFrom(properties);
        const float3 defaultEye(center.x, center.y + radius * 0.35f, center.z + radius * 2.5f);
        const float3 eye = cameraVec3(camera, "eye", defaultEye);
        const float3 lookAt = cameraVec3(camera, "center", center);
        const float3 up = cameraVec3(camera, "up", float3(0.0f, 1.0f, 0.0f));
        const float fovDegrees = cameraFloat(camera, "fovDegrees", 60.0f);
        const float znear = cameraFloat(camera, "znear", 0.1f);
        const float zfar = cameraFloat(camera, "zfar", std::max(radius * 8.0f, znear + 100.0f));
        const float aspect = height != 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;

        params.eye[0] = eye.x;
        params.eye[1] = eye.y;
        params.eye[2] = eye.z;
        params.eye[3] = 1.0f;
        params.center[0] = lookAt.x;
        params.center[1] = lookAt.y;
        params.center[2] = lookAt.z;
        params.center[3] = 1.0f;
        params.upProjection[0] = up.x;
        params.upProjection[1] = up.y;
        params.upProjection[2] = up.z;
        params.upProjection[3] = 0.0f;
        params.viewport[0] = aspect;
        params.viewport[1] = static_cast<float>(width);
        params.viewport[2] = static_cast<float>(height);
        params.viewport[3] = fovDegrees * 0.017453292519943295f;
        params.clipOrtho[0] = znear;
        params.clipOrtho[1] = zfar;
        params.clipOrtho[2] = radius * 2.0f;
        params.clipOrtho[3] = kDefaultReversedZ ? 1.0f : 0.0f;
        params.clearColor[0] = 0.015f;
        params.clearColor[1] = 0.018f;
        params.clearColor[2] = 0.024f;
        params.clearColor[3] = 1.0f;
        params.debugColorMode = debugColorModeFromProperties(properties);
        params.pageStrideWords = static_cast<uint32_t>(residency_.pageStride() / sizeof(uint32_t));
        params.drawItemCount = static_cast<uint32_t>(drawItems_.size());
        return updateHostBuffer(*paramsBuffer_, &params, sizeof(params));
    }

    scene::MeshletStreamAsset asset_;
    MeshletStreamResidencyManager residency_;
    scene::Bounds drawBounds_;
    std::vector<GPUDrivenStreamAssetGpuDrawItem> drawItems_;
    std::vector<GPUDrivenStreamAssetGpuPageTableEntry> pageTable_;
    std::unique_ptr<Buffer> pageBuffer_;
    std::unique_ptr<Buffer> drawItemBuffer_;
    std::unique_ptr<Buffer> pageTableBuffer_;
    std::unique_ptr<Buffer> paramsBuffer_;
    std::unique_ptr<BindlessHeap> bindlessHeap_;
    std::unique_ptr<ShaderModule> meshShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
    BindlessHandle pageHandle_;
    BindlessHandle drawItemHandle_;
    BindlessHandle pageTableHandle_;
    BindlessHandle paramsHandle_;
    ResourceState pageBufferState_ = ResourceState::Undefined;
    uint32_t maxResidentPages_ = 0;
    uint32_t maxPageUploadsPerFrame_ = 0;
    uint32_t maxDrawItems_ = 0;
};

} // namespace

std::unique_ptr<RenderGraphPass> createGPUDrivenStreamAssetPass()
{
    return std::make_unique<GPUDrivenStreamAssetPass>();
}

} // namespace metallic::render::builtin_pass
