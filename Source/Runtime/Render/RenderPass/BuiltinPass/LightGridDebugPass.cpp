#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/Subsystem/GPUSceneLightFrustum.h"

namespace metallic::render::builtin_pass {
namespace {

struct LightGridDebugPush {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mode = 0;
    uint32_t sliceIndex = 16;
    float viewDepth = 10.0f;
    float heatmapMaxLights = 32.0f;
    uint32_t flags = 0;
    uint32_t reserved = 0;
};
static_assert(sizeof(LightGridDebugPush) == 32);

float numberProperty(const RenderGraphProperties& properties, const char* key,
    float fallback, float minimum, float maximum)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_number()) { return fallback; }
    const double value = iter->get<double>();
    return std::isfinite(value) ? static_cast<float>(std::clamp(value,
        static_cast<double>(minimum), static_cast<double>(maximum))) : fallback;
}

uint32_t integerProperty(const RenderGraphProperties& properties, const char* key,
    uint32_t fallback, uint32_t minimum, uint32_t maximum)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_number()) { return fallback; }
    const double value = iter->get<double>();
    return std::isfinite(value) ? static_cast<uint32_t>(std::clamp(value,
        static_cast<double>(minimum), static_cast<double>(maximum))) : fallback;
}

std::string stringProperty(const RenderGraphProperties& properties, const char* key,
    const char* fallback)
{
    const auto iter = properties.find(key);
    return iter != properties.end() && iter->is_string() ? iter->get<std::string>() : fallback;
}

float3 vectorProperty(const RenderGraphProperties& properties, const char* key,
    const float3& fallback)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_array() || iter->size() != 3) { return fallback; }
    float values[3];
    for (size_t i = 0; i < 3; ++i) {
        if (!(*iter)[i].is_number()) { return fallback; }
        values[i] = (*iter)[i].get<float>();
        if (!std::isfinite(values[i])) { return fallback; }
    }
    return float3(values[0], values[1], values[2]);
}

// Keep descriptors alive on the legacy, untracked command path as well. The
// command owns this transaction until reset, not merely until queue submission.
class DebugProgramLifetime final : public SubmissionTransaction {
public:
    explicit DebugProgramLifetime(std::shared_ptr<ComputeProgram> program)
        : SubmissionTransaction([]() {}, []() {}), program_(std::move(program))
    {
    }

private:
    std::shared_ptr<ComputeProgram> program_;
};

class LightGridDebugPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    // Grid-owned buffers are private dependencies, so retain graphics ordering.
    bool supportsAsyncQueue() const override { return false; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Cluster light coverage heatmap")
            .storageReadWrite().format = Format::Rgba8Unorm;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        std::vector<RenderGraphRuntimeSetting> settings{
            runtimeEnumSetting("source", "Light Source", "world", {{"World", "world"}, {"Test Bench", "bench"}}),
            runtimeEnumSetting("visualization", "Coverage View", "peak",
                {{"Projected Peak", "peak"}, {"Z Slice", "slice"}, {"View Depth", "depth"}}),
            runtimeIntSetting("sliceIndex", "Z Slice", 16, 0, 255),
            runtimeFloatSetting("viewDepth", "View Depth (m)", 10.0f, 0.0f, 10000.0f),
            runtimeFloatSetting("heatmapMaxLights", "Heatmap Maximum", 32.0f, 1.0f, 4096.0f),
            runtimeBoolSetting("showGrid", "Tile Boundaries", true),
            runtimeBoolSetting("showLegend", "Color Legend", true),
            runtimeBoolSetting("showCounts", "Tile Counts", false),
            runtimeBoolSetting("includeGlobalLights", "Include Global Lights", false),
            runtimeIntSetting("tileSize", "Tile Size (px)", 32, 8, 256),
            runtimeIntSetting("depthSliceCount", "Z Slices", 32, 1, 256),
            runtimeIntSetting("maxLightsPerCell", "Cell List Capacity", 64, 1, 1024),
            runtimeIntSetting("lightCount", "Bench Light Count", 512, 0, 4096),
            runtimeFloatSetting("lightRange", "Bench Range (m)", 3.0f, 0.1f, 100.0f),
            runtimeIntSetting("seed", "Bench Seed", 1, 0, 1000000000),
            runtimeFloatSetting("spotFraction", "Bench Spot Fraction", 0.25f, 0.0f, 1.0f),
            runtimeEnumSetting("layout", "Bench Layout", "volume", {{"Volume", "volume"}, {"Overlap Stress", "overlap"}}),
            runtimeBoolSetting("animate", "Animate Bench", false),
        };
        appendCameraRuntimeSettings(settings, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, 60.0f);
        settings.push_back(runtimeFloatSetting("camera.znear", "Near Plane (m)", 0.1f, 0.001f, 100.0f));
        settings.push_back(runtimeFloatSetting("camera.zfar", "Far Plane (m)", 60.0f, 0.01f, 10000.0f));
        settings.push_back(runtimeEnumSetting("camera.projection", "Projection", "perspective",
            {{"Perspective", "perspective"}, {"Orthographic", "orthographic"}}));
        settings.push_back(runtimeFloatSetting("camera.orthoHeight", "Ortho Height (m)", 20.0f, 0.01f, 10000.0f));
        return settings;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.subsystems() == nullptr) {
            log = "LightGridDebugPass requires a device and render subsystem host";
            return makeError(Error::InvalidArgument);
        }
        device_ = context.device;
        if (!view_.valid()) {
            const uint32_t slots = std::max(context.subsystems()->frameSlotCount(), 1u);
            gpuScene_.setDefaultFrameSlotCount(slots);
            view_ = gpuScene_.createView();
            for (uint32_t slot = 0; slot < slots; ++slot) {
                grids_.push_back(std::make_unique<ClusterLightGrid>());
            }
        }
        if (program_.valid()) { return {}; }
        ShaderCompileResult shader;
        Result result = compileSlangShaderToSpirv({.moduleName = "LightGridDebug",
            .entryPointName = "lightGridDebugMain", .searchPath = kTriangleShaderSearchPath}, shader);
        if (!result) {
            log = "LightGridDebug shader compilation failed: " + shader.diagnostics;
            return result;
        }
        spirv_ = std::move(shader.spirv);
        result = initializeProgram(program_, log);
        if (!result) { return result; }
        // Graph shader reload stages a replacement pass. Validate its producer
        // here too, so a broken culling shader cannot replace a working graph
        // and only fail later during execute().
        for (auto& grid : grids_) {
            std::unique_ptr<RenderSubsystemShaderReload> reload;
            result = grid->prepareShaderReload(*device_, reload, log);
            if (!result) {
                program_.clear();
                return result;
            }
            reload->commit();
        }
        return {};
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() || color.view() == nullptr || !program_.valid() ||
            context.subsystems() == nullptr || device_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const auto& props = context.properties();
        const auto cameraIter = props.find("camera");
        const RenderGraphProperties camera = cameraIter != props.end() && cameraIter->is_object()
            ? *cameraIter : RenderGraphProperties::object();
        const ClusterLightGridDesc desc{
            .width = context.width(), .height = context.height(),
            .tileSize = integerProperty(props, "tileSize", 32, 8, 256),
            .depthSliceCount = integerProperty(props, "depthSliceCount", 32, 1, 256),
            .maxLightsPerCell = integerProperty(props, "maxLightsPerCell", 64, 1, 1024),
            .eye = vectorProperty(camera, "eye", float3(0.0f)),
            .center = vectorProperty(camera, "center", float3(0.0f, 0.0f, -1.0f)),
            .up = vectorProperty(camera, "up", float3(0.0f, 1.0f, 0.0f)),
            .aspect = float(context.width()) / std::max(context.height(), 1u),
            .fovRadians = numberProperty(camera, "fovDegrees", 60.0f, 1.0f, 179.0f) * 0.01745329252f,
            .zNear = numberProperty(camera, "znear", 0.1f, 0.001f, 100.0f),
            .zFar = numberProperty(camera, "zfar", 60.0f, 0.01f, 10000.0f),
            .orthoHeight = stringProperty(camera, "projection", "perspective") == "orthographic"
                ? numberProperty(camera, "orthoHeight", 20.0f, 0.01f, 10000.0f) : 0.0f,
        };
        std::string log;
        ClusterLightGridParams checkedParams;
        Result result = buildClusterLightGridParams(desc, checkedParams, log);
        if (!result) {
            spdlog::error("[LightGridDebugPass] {}", log);
            return result;
        }

        // A light-only view runs the production DrawSet coarse collection and
        // ClusterLightGrid GPU builder without replacing the renderer's view or
        // mutating RenderWorld. Bench fixtures cannot leak into scene lighting.
        if (stringProperty(props, "source", "world") == "bench") {
            buildBenchLights(props, context.frameIndex());
            gpuScene_.syncLights({}, benchLights_);
        } else {
            const scene::Scene* scene = context.world() != nullptr
                ? context.world()->scene() : context.runtimeScene();
            const auto resolvedLighting = resolveSceneLighting(scene, context.world());
            gpuScene_.syncLights(scene != nullptr ? std::span<const scene::RenderLight>(scene->lights())
                : std::span<const scene::RenderLight>{}, resolvedLighting.lights);
        }
        RenderFrameContext* frame = context.commandBuffer().frameContext();
        const uint32_t slot = frame != nullptr ? frame->slotIndex() : 0u;
        if (slot >= grids_.size() || !gpuScene_.prepareView(view_, slot, {
            .width = desc.width, .height = desc.height,
            .lightFrustumPlanes = gpuSceneLightFrustumPlanes(desc.eye, desc.center, desc.up,
                desc.aspect, desc.fovRadians, desc.zNear, desc.zFar, desc.orthoHeight)})) {
            return makeError(Error::InvalidArgument);
        }
        result = grids_[slot]->record(*device_, context.commandBuffer(), *context.subsystems(),
            gpuScene_, view_, slot, desc, log);
        if (!result) {
            spdlog::error("[LightGridDebugPass] {}", log);
            return result;
        }
        const auto* grid = grids_[slot]->snapshot(gpuScene_);
        if (grid == nullptr) { return makeError(Error::Failure); }

        ComputeProgram* program = &program_;
        if (frame == nullptr) {
            auto isolated = std::make_shared<ComputeProgram>();
            result = initializeProgram(*isolated, log);
            if (!result) { return result; }
            result = context.commandBuffer().addSubmissionTransaction(
                std::make_shared<DebugProgramLifetime>(isolated));
            if (!result) { return result; }
            program = isolated.get();
        }
        const std::string mode = stringProperty(props, "visualization", "peak");
        const LightGridDebugPush push{
            .width = desc.width, .height = desc.height,
            .mode = mode == "slice" ? 1u : mode == "depth" ? 2u : 0u,
            .sliceIndex = integerProperty(props, "sliceIndex", 16, 0, 255),
            .viewDepth = numberProperty(props, "viewDepth", 10.0f, 0.0f, 10000.0f),
            .heatmapMaxLights = numberProperty(props, "heatmapMaxLights", 32.0f, 1.0f, 4096.0f),
            .flags = (boolProperty(&props, "showGrid", true) ? 1u : 0u) |
                (boolProperty(&props, "showLegend", true) ? 2u : 0u) |
                (boolProperty(&props, "includeGlobalLights", false) ? 4u : 0u) |
                (boolProperty(&props, "showCounts", false) ? 8u : 0u),
        };
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .buffer = grid->parameters},
            {.binding = 1, .buffer = grid->cells},
            {.binding = 2, .textureView = color.view()},
        };
        return program->dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)), .pushData = &push,
            .pushDataSize = sizeof(push), .groupCountX = (desc.width + 7u) / 8u,
            .groupCountY = (desc.height + 7u) / 8u});
    }

private:
    Result initializeProgram(ComputeProgram& program, std::string& log)
    {
        const ComputeProgramBindingDesc bindings[] = {
            {.binding = 0}, {.binding = 1},
            {.binding = 2, .kind = ComputeResourceBindingKind::StorageImage},
        };
        return program.initialize(*device_, {.spirv = spirv_.data(),
            .byteSize = spirv_.size() * sizeof(uint32_t), .pushConstantSize = sizeof(LightGridDebugPush),
            .bindings = bindings, .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .debugName = "LightGridDebugPass", .requiresRayQuery = false}, log);
    }

    void buildBenchLights(const RenderGraphProperties& props, uint64_t frameIndex)
    {
        const uint32_t count = integerProperty(props, "lightCount", 512, 0, 4096);
        uint32_t state = integerProperty(props, "seed", 1, 0, 1000000000);
        const auto random = [&state]() {
            state = state * 1664525u + 1013904223u;
            return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
        };
        const float range = numberProperty(props, "lightRange", 3.0f, 0.1f, 100.0f);
        const float spotFraction = numberProperty(props, "spotFraction", 0.25f, 0.0f, 1.0f);
        const bool overlap = stringProperty(props, "layout", "volume") == "overlap";
        const bool animate = boolProperty(&props, "animate", false);
        const float time = static_cast<float>(frameIndex % 216000u) / 60.0f;
        benchLights_.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& light = benchLights_[i];
            light.properties = {.type = random() < spotFraction ? "spot" : "point",
                .intensity = 100.0, .range = range, .innerConeAngle = 0.25,
                .outerConeAngle = 0.6};
            const float x = (random() * 2.0f - 1.0f) * 16.0f;
            const float y = (random() * 2.0f - 1.0f) * 8.0f;
            const float z = -(2.0f + random() * 38.0f);
            light.position = overlap ? float3(0.0f, 0.0f, -10.0f) : float3(x, y, z);
            if (animate) {
                light.position.x += std::sin(time + i * 0.37f) * 2.0f;
                light.position.y += std::cos(time * 0.7f + i * 0.23f);
            }
            light.direction = float3(0.0f, 0.0f, -1.0f);
            light.enabled = true;
        }
    }

    Device* device_ = nullptr;
    ComputeProgram program_;
    std::vector<uint32_t> spirv_;
    GPUScene gpuScene_;
    GPUSceneViewId view_;
    std::vector<std::unique_ptr<ClusterLightGrid>> grids_;
    std::vector<scene::PunctualLight> benchLights_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createLightGridDebugPass()
{
    return std::make_unique<LightGridDebugPass>();
}

} // namespace metallic::render::builtin_pass
