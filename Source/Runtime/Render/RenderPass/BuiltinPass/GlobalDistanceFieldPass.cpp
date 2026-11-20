#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"

#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kGlobalDistanceFieldABI = 0x4744460000000001ull;
constexpr uint64_t kGlobalDistanceFieldDebugABI = 0x4744464400000001ull;
constexpr uint32_t kDefaultResolution = 64;
constexpr float kInfluenceRangeInVoxels = 4.0f;

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

struct alignas(16) GlobalDistanceFieldDebugParams {
    ShaderSampledImage distanceField;
    ShaderSampler sampler;
    ShaderBuffer domain;
    ShaderStorageImage color, trace;
    uint32_t debugMode;
    uint32_t padding0, padding1;
    ViewCameraConstants current;
};

static_assert(sizeof(GlobalDistanceFieldDomain) == 32);
static_assert(sizeof(GlobalDistanceFieldParams) == 64 && offsetof(GlobalDistanceFieldParams, domain) == 32);
static_assert(sizeof(GlobalDistanceFieldDebugParams) == 112 && offsetof(GlobalDistanceFieldDebugParams, current) == 32);

uint32_t logicalResolution(const RenderGraphProperties& properties)
{
    const auto value = properties.find("resolution");
    if (value == properties.end()) { return kDefaultResolution; }
    if (!value->is_number_integer()) { return 0; }
    const uint64_t resolution = value->get<uint64_t>();
    return resolution >= 2 && resolution <= UINT32_MAX - 4 ? static_cast<uint32_t>(resolution) : 0;
}

class GlobalDistanceFieldPass final : public ComputePass {
public:
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        // The physical lattice has one additional sample at the positive edge.
        const uint32_t samples = logicalResolution(properties()) + 1u;
        RenderPassReflection reflection;
        reflection.addTextureOutput("distanceField", "Clipped signed world distance in meters")
            .texture3D(samples, samples, samples).storageWrite().format = Format::R32Sfloat;
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
        const uint32_t resolution = logicalResolution(properties());
        if (resolution == 0) {
            log = "Global Distance Field resolution must be an integer of at least 2";
            return makeError(Error::InvalidArgument);
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
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/GlobalDistanceFieldCompose",
                .entryPointName = "globalDistanceFieldMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<GlobalDistanceFieldParams>(kGlobalDistanceFieldABI),
                .debugName = "GlobalDistanceField"}, program_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        GlobalDistanceFieldParams params{};
        params.domain = domain_;
        uint32_t unsupportedTransforms = 0;
        auto instances = meshFields_.uploadInstances(*context.runtimeScene(), params.instanceCount, unsupportedTransforms);
        if (!instances) { return makeError(instances.error()); }
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        const auto meshBindings = meshFields_.bind(writer);
        params.instances = writer.buffer(instances->get());
        params.mips = meshBindings.mips;
        params.indirection = meshBindings.indirection;
        params.atlas = meshBindings.atlas;
        params.atlasSampler = meshBindings.atlasSampler;
        const auto distanceField = context.outputTexture("distanceField");
        params.distanceField = writer.storageImageHandle(distanceField.view());
        params.domainOutput = writer.buffer(context.outputBuffer("domain").buffer());
        auto encoded = writer.encode(params, kGlobalDistanceFieldABI);
        if (!encoded) { return makeError(encoded.error()); }
        const uint32_t groups = (distanceField.desc().width + 3u) / 4u;
        auto result = program_.dispatch(commands, *encoded, groups, groups, groups);
        if (result && context.detailedDebugSnapshots()) {
            context.debugCheckpoint("GlobalDistanceField", {}, {
                {"assetCount", meshFields_.assetCount()}, {"instanceCount", params.instanceCount},
                {"unsupportedTransforms", unsupportedTransforms}, {"sampleCountPerAxis", distanceField.desc().width},
                {"minAndVoxelSize", domain_.minAndVoxelSize}, {"maxAndInfluenceRadius", domain_.maxAndInfluenceRadius}});
        }
        return result;
    }

    std::vector<std::string> debugCheckpoints() const override { return {"AfterPass", "GlobalDistanceField"}; }

private:
    Device* device_ = nullptr;
    GlobalDistanceFieldDomain domain_{};
    ComputeKernel program_;
    MeshDistanceFieldGPU meshFields_;
};

class GlobalDistanceFieldDebugPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("distanceField", "Clipped signed world distance in meters")
            .texture3D(0, 0, 0).sampledRead().format = Format::R32Sfloat;
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
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/GlobalDistanceFieldDebug",
                .entryPointName = "globalDistanceFieldDebugMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<GlobalDistanceFieldDebugParams>(kGlobalDistanceFieldDebugABI),
                .debugName = "GlobalDistanceFieldDebug"}, program_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        GlobalDistanceFieldDebugParams params{};
        params.distanceField = writer.sampledImageHandle(context.inputTexture("distanceField").view());
        params.sampler = writer.samplerHandle({});
        params.domain = writer.buffer(context.inputBuffer("domain").buffer());
        params.color = writer.storageImageHandle(context.outputTexture("color").view());
        params.trace = writer.storageImageHandle(context.outputTexture("trace").view());
        const auto mode = context.properties().value("debugMode", std::string("normals"));
        params.debugMode = mode == "coverage" ? 2u : (mode == "steps" ? 1u : 0u);
        params.current = context.viewConstants() ? context.viewConstants()->current :
            RenderView{}.constants(0, context.width(), context.height(), context.width(), context.height()).current;
        auto encoded = writer.encode(params, kGlobalDistanceFieldDebugABI);
        if (!encoded) { return makeError(encoded.error()); }
        return program_.dispatch(commands, *encoded, (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
    }

private:
    Device* device_ = nullptr;
    ComputeKernel program_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createGlobalDistanceFieldPass()
{
    return std::make_unique<GlobalDistanceFieldPass>();
}

std::unique_ptr<RenderGraphPass> createGlobalDistanceFieldDebugPass()
{
    return std::make_unique<GlobalDistanceFieldDebugPass>();
}

} // namespace metallic::render::builtin_pass
