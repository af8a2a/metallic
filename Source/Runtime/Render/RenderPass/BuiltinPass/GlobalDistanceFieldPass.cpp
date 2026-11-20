#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kGlobalDistanceFieldABI = 0x4744460000000001ull;
constexpr uint64_t kGlobalDistanceFieldDebugABI = 0x4744464400000001ull;
constexpr uint64_t kGlobalDistanceFieldPagesABI = 0x4744465000000001ull;
constexpr uint64_t kGlobalDistanceFieldPagesDebugABI = 0x4744504400000001ull;
constexpr uint64_t kGlobalDistanceFieldMipABI = 0x4744464d00000001ull;
constexpr uint64_t kQuantizedGlobalDistanceFieldDebugABI = 0x4744514400000001ull;
constexpr uint32_t kDefaultResolution = 64;
constexpr float kInfluenceRangeInVoxels = 4.0f;
constexpr uint32_t kCellsPerPage = 7;
constexpr uint32_t kSamplesPerPage = kCellsPerPage + 1;
constexpr uint32_t kDefaultPhysicalPages = 216;
constexpr uint32_t kPageStatisticsCount = 5;
constexpr uint32_t kMipFactor = 4;
constexpr SlangMacroDefine kQuantizedPageAtlasDefines[]{{"METALLIC_GDF_PAGE_ATLAS_R8", "1"}};

struct alignas(16) GlobalDistanceFieldDomain {
    std::array<float, 4> minAndVoxelSize;
    std::array<float, 4> maxAndInfluenceRadius;
};

struct alignas(16) GlobalDistanceFieldParams {
    ShaderBuffer instances, mips, indirection;
    ShaderSampledImage atlas;
    ShaderSampler atlasSampler;
    ShaderStorageImage distanceField;
    ShaderBuffer domainOutput;
    uint32_t instanceCount;
    GlobalDistanceFieldDomain domain;
};

struct alignas(16) GlobalDistanceFieldPagesParams {
    ShaderBuffer instances, mips, indirection;
    ShaderSampledImage atlas;
    ShaderSampler atlasSampler;
    ShaderStorageImage pageTable, pageAtlas;
    ShaderBuffer domainOutput, statistics;
    uint32_t instanceCount, logicalResolution, maxPhysicalPages;
    GlobalDistanceFieldDomain domain;
};

struct alignas(16) GlobalDistanceFieldMipParams {
    ShaderSampledImage pageTable, pageAtlas;
    ShaderSampler sampler;
    ShaderStorageImage previousMip, outputMip;
    uint32_t logicalResolution, coarseResolution, padding;
};

struct alignas(16) GlobalDistanceFieldDebugParams {
    ShaderSampledImage distanceField;
    ShaderSampler sampler;
    ShaderBuffer domain;
    ShaderStorageImage color, trace;
    uint32_t debugMode;
    uint32_t padding0, padding1;
    ViewCameraConstants current;
};

struct alignas(16) GlobalDistanceFieldPagesDebugParams {
    ShaderSampledImage pageTable, pageAtlas;
    ShaderSampler sampler;
    ShaderBuffer domain;
    ShaderStorageImage color, trace;
    uint32_t debugMode, padding;
    ViewCameraConstants current;
};

struct alignas(16) QuantizedGlobalDistanceFieldDebugParams {
    ShaderSampledImage pageTable, pageAtlas, coarseMip;
    ShaderSampler sampler;
    ShaderBuffer domain;
    ShaderStorageImage color, trace;
    uint32_t debugMode;
    ViewCameraConstants current;
};

static_assert(sizeof(GlobalDistanceFieldDomain) == 32);
static_assert(sizeof(GlobalDistanceFieldParams) == 64 && offsetof(GlobalDistanceFieldParams, domain) == 32);
static_assert(sizeof(GlobalDistanceFieldDebugParams) == 112 && offsetof(GlobalDistanceFieldDebugParams, current) == 32);
static_assert(sizeof(GlobalDistanceFieldPagesParams) == 80 && offsetof(GlobalDistanceFieldPagesParams, domain) == 48);
static_assert(sizeof(GlobalDistanceFieldPagesDebugParams) == 112 &&
    offsetof(GlobalDistanceFieldPagesDebugParams, current) == 32);
static_assert(sizeof(GlobalDistanceFieldMipParams) == 32);
static_assert(sizeof(QuantizedGlobalDistanceFieldDebugParams) == 112 &&
    offsetof(QuantizedGlobalDistanceFieldDebugParams, current) == 32);

uint32_t coarseMipResolution(uint32_t resolution)
{
    return (resolution + kMipFactor - 1u) / kMipFactor;
}

uint32_t logicalResolution(const RenderGraphProperties& properties)
{
    const auto value = properties.find("resolution");
    if (value == properties.end()) { return kDefaultResolution; }
    if (!value->is_number_integer()) { return 0; }
    const uint64_t resolution = value->get<uint64_t>();
    return resolution >= 2 && resolution <= UINT32_MAX - 4 ? static_cast<uint32_t>(resolution) : 0;
}

uint32_t pagesPerAxis(uint32_t resolution)
{
    return resolution / kCellsPerPage + (resolution % kCellsPerPage != 0);
}

std::optional<uint32_t> physicalPageBudget(const RenderGraphProperties& properties)
{
    const auto value = properties.find("maxPhysicalPages");
    if (value == properties.end()) { return kDefaultPhysicalPages; }
    if (!value->is_number_integer()) { return std::nullopt; }
    const uint64_t pages = value->get<uint64_t>();
    // The final three values are reserved for positive, negative, and unavailable pages.
    if (pages > UINT32_MAX - 2) { return std::nullopt; }
    return static_cast<uint32_t>(pages);
}

uint32_t pageAtlasSize(uint32_t maxPhysicalPages)
{
    uint32_t side = 1;
    while (uint64_t(side) * side * side < maxPhysicalPages) { ++side; }
    return side * kSamplesPerPage;
}

class GlobalDistanceFieldPass final : public ComputePass {
public:
    explicit GlobalDistanceFieldPass(bool usePages) : usePages_(usePages) {}

    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        const uint32_t resolution = logicalResolution(properties());
        if (usePages_) {
            const uint32_t tableSize = std::max(1u, pagesPerAxis(resolution));
            const uint32_t atlasSize = pageAtlasSize(physicalPageBudget(properties()).value_or(0));
            const bool quantized = quantizedPages();
            auto& pageTable = reflection.addTextureOutput("pageTable", "Virtual page to atlas mapping or constant page state")
                .texture3D(tableSize, tableSize, tableSize).storageWrite();
            pageTable.format = Format::R32Uint;
            if (quantized) { pageTable.stageAccess(RenderGraphResourceAccess::TextureSampleRead); }
            auto& pageAtlas = reflection.addTextureOutput("pageAtlas", "Physical pages of clipped signed distances")
                .texture3D(atlasSize, atlasSize, atlasSize).storageWrite();
            pageAtlas.format = quantized ? Format::R8Unorm : Format::R32Sfloat;
            if (quantized) {
                pageAtlas.stageAccess(RenderGraphResourceAccess::TextureSampleRead);
                const uint32_t mipSize = std::max(1u, coarseMipResolution(resolution));
                reflection.addTextureOutput("coarseMip", "Propagated coarse distances encoded into R8")
                    .texture3D(mipSize, mipSize, mipSize).storageReadWrite().format = Format::R8Unorm;
                reflection.addTextureOutput("mipScratch", "Ping-pong storage for coarse distance propagation")
                    .texture3D(mipSize, mipSize, mipSize).storageReadWrite().format = Format::R8Unorm;
            }
            reflection.addBufferOutput("statistics", "Required, resident, positive, negative and unavailable pages")
                .buffer(kPageStatisticsCount * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").storageReadWrite();
        } else {
            // The physical lattice has one additional sample at the positive edge.
            const uint32_t samples = resolution + 1u;
            reflection.addTextureOutput("distanceField", "Clipped signed world distance in meters")
                .texture3D(samples, samples, samples).storageWrite().format = Format::R32Sfloat;
        }
        reflection.addBufferOutput("domain", "World bounds, voxel size and influence radius")
            .buffer(sizeof(GlobalDistanceFieldDomain), sizeof(GlobalDistanceFieldDomain)).storageWrite();
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.device || !context.graphicsQueue || !context.runtimeScene) {
            log = "Global Distance Field requires a world scene and a graphics queue";
            return makeError(Error::InvalidArgument);
        }
        const bool quantized = quantizedPages();
        if (quantized && !context.device->capabilities().shaderStorageImageExtendedFormats) {
            log = "Quantized Global Distance Field requires R8 storage image support";
            return makeError(Error::Unsupported);
        }
        const uint32_t resolution = logicalResolution(properties());
        if (resolution == 0) {
            log = "Global Distance Field resolution must be an integer of at least 2";
            return makeError(Error::InvalidArgument);
        }
        resolution_ = resolution;
        if (usePages_) {
            const auto budget = physicalPageBudget(properties());
            const uint64_t pages = pagesPerAxis(resolution);
            if (!budget || pages * pages > UINT32_MAX / pages) {
                log = "Global Distance Field page budget or virtual page count exceeds its 32-bit range";
                return makeError(Error::InvalidArgument);
            }
            maxPhysicalPages_ = *budget;
        }
        std::vector<std::filesystem::path> paths;
        try {
            const float extent = properties().value("extent", 3.0f);
            const auto center = properties().value("center", std::array<float, 3>{});
            for (size_t axis = 0; axis < 3; ++axis) {
                domain_.minAndVoxelSize[axis] = center[axis] - extent;
                domain_.maxAndInfluenceRadius[axis] = center[axis] + extent;
                if (!std::isfinite(domain_.minAndVoxelSize[axis]) || !std::isfinite(domain_.maxAndInfluenceRadius[axis]) ||
                    domain_.minAndVoxelSize[axis] >= domain_.maxAndInfluenceRadius[axis]) {
                    log = "Global Distance Field requires finite bounds with a positive extent";
                    return makeError(Error::InvalidArgument);
                }
            }
            const float voxelSize = (extent * 2.0f) / static_cast<float>(resolution);
            domain_.minAndVoxelSize[3] = voxelSize;
            domain_.maxAndInfluenceRadius[3] = voxelSize * kInfluenceRangeInVoxels;
            if (voxelSize <= 0.0f || !std::isfinite(domain_.maxAndInfluenceRadius[3])) {
                log = "Global Distance Field voxel size or influence radius is out of range";
                return makeError(Error::InvalidArgument);
            }
            const auto fields = properties().value("distanceFieldPaths", std::vector<std::string>{});
            for (const auto& path : fields) { paths.push_back(std::filesystem::path(PROJECT_SOURCE_DIR) / path); }
        } catch (const nlohmann::json::exception& error) {
            log = "Invalid Global Distance Field properties: " + std::string(error.what());
            return makeError(Error::InvalidArgument);
        }

        device_ = context.device;
        auto result = meshFields_.initialize(*device_, *context.graphicsQueue, *context.runtimeScene, paths, log);
        if (!result) {
            if (log.empty()) { log = "Global Distance Field mesh atlas upload failed: " + std::string(resultToString(result)); }
            return result;
        }
        if (usePages_) {
            constexpr auto abi = parameterAbi<GlobalDistanceFieldPagesParams>(kGlobalDistanceFieldPagesABI);
            auto& shaders = ShaderRegistry::instance();
            result = shaders.getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/GlobalDistanceFieldPageBuild",
                    .entryPointName = "globalDistanceFieldClearPageStatsMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = abi, .debugName = "GlobalDistanceFieldClearPageStats"}, clearPageStats_, log);
            if (!result) { return result; }
            const std::span<const SlangMacroDefine> defines = quantized ?
                std::span<const SlangMacroDefine>(kQuantizedPageAtlasDefines) : std::span<const SlangMacroDefine>{};
            result = shaders.getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/GlobalDistanceFieldPageBuild",
                    .entryPointName = "globalDistanceFieldBuildPagesMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                    .macroDefines = defines},
                {.parameters = abi, .debugName = quantized ? "GlobalDistanceFieldBuildR8Pages" : "GlobalDistanceFieldBuildPages"},
                program_, log);
            if (!result || !quantized) { return result; }
            constexpr auto mipAbi = parameterAbi<GlobalDistanceFieldMipParams>(kGlobalDistanceFieldMipABI);
            result = shaders.getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/GlobalDistanceFieldMip",
                    .entryPointName = "globalDistanceFieldSeedMipMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = mipAbi, .debugName = "GlobalDistanceFieldSeedMip"}, seedMip_, log);
            if (!result) { return result; }
            return shaders.getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/GlobalDistanceFieldMip",
                    .entryPointName = "globalDistanceFieldPropagateMipMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = mipAbi, .debugName = "GlobalDistanceFieldPropagateMip"}, propagateMip_, log);
        }
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/GlobalDistanceFieldCompose",
                .entryPointName = "globalDistanceFieldMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<GlobalDistanceFieldParams>(kGlobalDistanceFieldABI),
                .debugName = "GlobalDistanceField"}, program_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        uint32_t instanceCount = 0;
        uint32_t unsupportedTransforms = 0;
        auto instances = meshFields_.uploadInstances(*context.runtimeScene(), instanceCount, unsupportedTransforms);
        if (!instances) { return makeError(instances.error()); }
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        const auto meshBindings = meshFields_.bind(writer);
        const auto bindMeshes = [&](auto& params) {
            params.instances = writer.buffer(instances->get());
            params.mips = meshBindings.mips;
            params.indirection = meshBindings.indirection;
            params.atlas = meshBindings.atlas;
            params.atlasSampler = meshBindings.atlasSampler;
            params.domainOutput = writer.buffer(context.outputBuffer("domain").buffer());
            params.instanceCount = instanceCount;
            params.domain = domain_;
        };
        Result<> result;
        if (usePages_) {
            GlobalDistanceFieldPagesParams params{};
            bindMeshes(params);
            params.pageTable = writer.storageImageHandle(context.outputTexture("pageTable").view());
            params.pageAtlas = writer.storageImageHandle(context.outputTexture("pageAtlas").view());
            params.statistics = writer.buffer(context.outputBuffer("statistics").buffer());
            params.logicalResolution = resolution_;
            params.maxPhysicalPages = maxPhysicalPages_;
            auto encoded = writer.encode(params, kGlobalDistanceFieldPagesABI);
            if (!encoded) { return makeError(encoded.error()); }
            result = executePages(context, *encoded, writer);
        } else {
            GlobalDistanceFieldParams params{};
            bindMeshes(params);
            const auto distanceField = context.outputTexture("distanceField");
            params.distanceField = writer.storageImageHandle(distanceField.view());
            auto encoded = writer.encode(params, kGlobalDistanceFieldABI);
            if (!encoded) { return makeError(encoded.error()); }
            const uint32_t groups = (distanceField.desc().width + 3u) / 4u;
            result = program_.dispatch(commands, *encoded, groups, groups, groups);
        }
        if (result && context.detailedDebugSnapshots()) {
            context.debugCheckpoint("GlobalDistanceField", {}, {
                {"assetCount", meshFields_.assetCount()}, {"instanceCount", instanceCount},
                {"unsupportedTransforms", unsupportedTransforms}, {"sampleCountPerAxis", resolution_ + 1u},
                {"virtualPagesPerAxis", usePages_ ? pagesPerAxis(resolution_) : 0u},
                {"maxPhysicalPages", maxPhysicalPages_},
                {"minAndVoxelSize", domain_.minAndVoxelSize}, {"maxAndInfluenceRadius", domain_.maxAndInfluenceRadius}});
        }
        return result;
    }

    std::vector<std::string> debugCheckpoints() const override { return {"AfterPass", "GlobalDistanceField"}; }

private:
    bool quantizedPages() const { return usePages_ && properties().value("quantized", false); }

    Result<> executePages(RenderGraphExecutionContext& context, const EncodedParameters& encoded, ParameterWriter& writer)
    {
        using Access = RenderGraphResourceAccess;
        const RenderGraphStageUse clearUses[] = {{"statistics", Access::BufferStorageWrite}};
        const RenderGraphStageUse buildUses[] = {
            {"pageTable", Access::TextureStorageWrite}, {"pageAtlas", Access::TextureStorageWrite},
            {"domain", Access::BufferStorageWrite}, {"statistics", Access::BufferStorageReadWrite},
        };
        const uint32_t groups = pagesPerAxis(resolution_);
        const RenderGraphStage pageStages[] = {
            {"Clear page statistics", clearUses, [&](CommandBuffer& commands) {
                return clearPageStats_.dispatch(commands, encoded, 1, 1, 1);
            }},
            {"Compose sparse pages", buildUses, [&](CommandBuffer& commands) {
                return program_.dispatch(commands, encoded, groups, groups, groups);
            }},
        };
        if (!quantizedPages()) {
            return context.executeComputeStages(pageStages);
        }
        return executeMipStages(context, writer, pageStages);
    }

    Result<> executeMipStages(RenderGraphExecutionContext& context, ParameterWriter& writer,
        std::span<const RenderGraphStage, 2> pageStages)
    {
        const auto coarse = writer.storageImageHandle(context.outputTexture("coarseMip").view());
        const auto scratch = writer.storageImageHandle(context.outputTexture("mipScratch").view());
        GlobalDistanceFieldMipParams params{};
        params.pageTable = writer.sampledImageHandle(context.outputTexture("pageTable").view());
        params.pageAtlas = writer.sampledImageHandle(context.outputTexture("pageAtlas").view());
        params.sampler = writer.samplerHandle({});
        params.logicalResolution = resolution_;
        params.coarseResolution = coarseMipResolution(resolution_);
        params.previousMip = scratch;
        params.outputMip = coarse;
        const auto toCoarse = writer.encode(params, kGlobalDistanceFieldMipABI);
        if (!toCoarse) { return makeError(toCoarse.error()); }
        params.previousMip = coarse;
        params.outputMip = scratch;
        const auto toScratch = writer.encode(params, kGlobalDistanceFieldMipABI);
        if (!toScratch) { return makeError(toScratch.error()); }

        using Access = RenderGraphResourceAccess;
        const RenderGraphStageUse seedUses[] = {
            {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead},
            {"coarseMip", Access::TextureStorageWrite},
        };
        const RenderGraphStageUse scratchUses[] = {
            {"coarseMip", Access::TextureStorageRead}, {"mipScratch", Access::TextureStorageWrite},
        };
        const RenderGraphStageUse coarseUses[] = {
            {"mipScratch", Access::TextureStorageRead}, {"coarseMip", Access::TextureStorageWrite},
        };
        const uint32_t groups = (params.coarseResolution + 3u) / 4u;
        const auto toScratchStage = [&](CommandBuffer& commands) {
            return propagateMip_.dispatch(commands, *toScratch, groups, groups, groups);
        };
        const auto toCoarseStage = [&](CommandBuffer& commands) {
            return propagateMip_.dispatch(commands, *toCoarse, groups, groups, groups);
        };
        // UE 5.7.4 GlobalDistanceField.cpp uses five R8 writes, including the
        // first Eikonal solve from fine pages. The fifth write lands in coarseMip.
        const RenderGraphStage stages[] = {
            pageStages[0], pageStages[1],
            {"Seed coarse mip from fine pages", seedUses, [&](CommandBuffer& commands) {
                return seedMip_.dispatch(commands, *toCoarse, groups, groups, groups);
            }},
            {"Propagate coarse mip 1", scratchUses, toScratchStage},
            {"Propagate coarse mip 2", coarseUses, toCoarseStage},
            {"Propagate coarse mip 3", scratchUses, toScratchStage},
            {"Propagate coarse mip 4", coarseUses, toCoarseStage},
        };
        return context.executeStages(stages);
    }

    bool usePages_;
    Device* device_ = nullptr;
    uint32_t resolution_ = 0;
    uint32_t maxPhysicalPages_ = 0;
    GlobalDistanceFieldDomain domain_{};
    ComputeKernel program_;
    ComputeKernel clearPageStats_;
    ComputeKernel seedMip_, propagateMip_;
    MeshDistanceFieldGPU meshFields_;
};

class GlobalDistanceFieldDebugPass final : public ComputePass {
public:
    explicit GlobalDistanceFieldDebugPass(bool usePages) : usePages_(usePages) {}

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        if (usePages_) {
            reflection.addTextureInput("pageTable", "Virtual page to atlas mapping or constant page state")
                .texture3D(0, 0, 0).sampledRead().format = Format::R32Uint;
            reflection.addTextureInput("pageAtlas", "Physical pages of signed world distances in meters")
                .texture3D(0, 0, 0).sampledRead().format = quantizedPages() ? Format::R8Unorm : Format::R32Sfloat;
            if (quantizedPages()) {
                reflection.addTextureInput("coarseMip", "Propagated coarse distances encoded into R8")
                    .texture3D(0, 0, 0).sampledRead().format = Format::R8Unorm;
            }
        } else {
            reflection.addTextureInput("distanceField", "Clipped signed world distance in meters")
                .texture3D(0, 0, 0).sampledRead().format = Format::R32Sfloat;
        }
        reflection.addBufferInput("domain", "World bounds, voxel size and influence radius")
            .buffer(sizeof(GlobalDistanceFieldDomain), sizeof(GlobalDistanceFieldDomain)).storageRead();
        reflection.addTextureOutput("color", "Global distance field trace visualization")
            .storageWrite().format = Format::RGBA16Sfloat;
        reflection.addTextureOutput("trace", "World hit distance, steps, coverage state, surface expansion")
            .storageWrite().format = Format::RGBA32Sfloat;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {runtimeEnumSetting("debugMode", "Display", "normals",
            {{"Normals", "normals"}, {"Trace Steps", "steps"}, {"Coverage", "coverage"}})};
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        if (quantizedPages()) {
            return ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = "Features/DistanceField/GlobalDistanceFieldDebug",
                    .entryPointName = "globalDistanceFieldQuantizedDebugMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<QuantizedGlobalDistanceFieldDebugParams>(kQuantizedGlobalDistanceFieldDebugABI),
                    .debugName = "QuantizedGlobalDistanceFieldDebug"}, program_, log);
        }
        const auto abi = usePages_ ? parameterAbi<GlobalDistanceFieldPagesDebugParams>(kGlobalDistanceFieldPagesDebugABI) :
            parameterAbi<GlobalDistanceFieldDebugParams>(kGlobalDistanceFieldDebugABI);
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/GlobalDistanceFieldDebug",
                .entryPointName = usePages_ ? "globalDistanceFieldPagesDebugMain" : "globalDistanceFieldDebugMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = abi, .debugName = usePages_ ? "GlobalDistanceFieldPagesDebug" : "GlobalDistanceFieldDebug"},
            program_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        const auto encode = [&](auto& params, uint64_t abi) {
            params.sampler = writer.samplerHandle({});
            params.domain = writer.buffer(context.inputBuffer("domain").buffer());
            params.color = writer.storageImageHandle(context.outputTexture("color").view());
            params.trace = writer.storageImageHandle(context.outputTexture("trace").view());
            const auto mode = context.properties().value("debugMode", std::string("normals"));
            params.debugMode = mode == "coverage" ? 2u : (mode == "steps" ? 1u : 0u);
            params.current = context.viewConstants() ? context.viewConstants()->current :
                RenderView{}.constants(0, context.width(), context.height(), context.width(), context.height()).current;
            return writer.encode(params, abi);
        };
        const auto encoded = [&] {
            if (quantizedPages()) {
                QuantizedGlobalDistanceFieldDebugParams params{};
                params.pageTable = writer.sampledImageHandle(context.inputTexture("pageTable").view());
                params.pageAtlas = writer.sampledImageHandle(context.inputTexture("pageAtlas").view());
                params.coarseMip = writer.sampledImageHandle(context.inputTexture("coarseMip").view());
                return encode(params, kQuantizedGlobalDistanceFieldDebugABI);
            }
            if (usePages_) {
                GlobalDistanceFieldPagesDebugParams params{};
                params.pageTable = writer.sampledImageHandle(context.inputTexture("pageTable").view());
                params.pageAtlas = writer.sampledImageHandle(context.inputTexture("pageAtlas").view());
                return encode(params, kGlobalDistanceFieldPagesDebugABI);
            }
            GlobalDistanceFieldDebugParams params{};
            params.distanceField = writer.sampledImageHandle(context.inputTexture("distanceField").view());
            return encode(params, kGlobalDistanceFieldDebugABI);
        }();
        if (!encoded) { return makeError(encoded.error()); }
        return program_.dispatch(commands, *encoded, (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
    }

private:
    bool quantizedPages() const { return usePages_ && properties().value("quantized", false); }

    bool usePages_;
    Device* device_ = nullptr;
    ComputeKernel program_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createGlobalDistanceFieldPass()
{
    return std::make_unique<GlobalDistanceFieldPass>(false);
}

std::unique_ptr<RenderGraphPass> createGlobalDistanceFieldDebugPass()
{
    return std::make_unique<GlobalDistanceFieldDebugPass>(false);
}

std::unique_ptr<RenderGraphPass> createGlobalDistanceFieldPagesPass()
{
    return std::make_unique<GlobalDistanceFieldPass>(true);
}

std::unique_ptr<RenderGraphPass> createGlobalDistanceFieldPagesDebugPass()
{
    return std::make_unique<GlobalDistanceFieldDebugPass>(true);
}

} // namespace metallic::render::builtin_pass
