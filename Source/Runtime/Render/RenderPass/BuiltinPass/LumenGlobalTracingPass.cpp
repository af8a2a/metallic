#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kTraceABI = 0x4c554d454e010004ull;
constexpr uint32_t kMaxFixedRays = 4096;

struct LumenGlobalTraceRay {
    std::array<float, 4> originAndMinDistance;
    std::array<float, 4> directionAndMaxDistance;
};

struct LumenGlobalTraceResult {
    std::array<float, 4> metrics, normal, radiance;
    std::array<uint32_t, 4> state, candidates, card;
};

struct LumenGlobalTraceParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderBuffer rays, results;
    ShaderStorageImage radiance;
    uint32_t rayCount, clipmapCount;
};

static_assert(sizeof(LumenGlobalTraceRay) == 32);
static_assert(sizeof(LumenGlobalTraceResult) == 96);
static_assert(sizeof(LumenGlobalTraceParams) == 64);

uint32_t fixedRayCount(const RenderGraphProperties& properties)
{
    const auto rays = properties.find("rays");
    return rays != properties.end() && rays->is_array() && rays->size() <= kMaxFixedRays ?
        static_cast<uint32_t>(rays->size()) : 0u;
}

std::optional<LumenGlobalTraceRay> readRay(const RenderGraphProperties& properties)
{
    try {
        const auto& originSource = properties.at("origin");
        const auto& directionSource = properties.at("direction");
        if (!originSource.is_array() || originSource.size() != 3 ||
            !directionSource.is_array() || directionSource.size() != 3) {
            return std::nullopt;
        }
        const auto origin = originSource.get<std::array<float, 3>>();
        const auto direction = directionSource.get<std::array<double, 3>>();
        const float minDistance = properties.at("minDistance").get<float>();
        const float maxDistance = properties.at("maxDistance").get<float>();
        const double length = std::hypot(direction[0], direction[1], direction[2]);
        if (!std::ranges::all_of(origin, [](float value) { return std::isfinite(value); }) ||
            !std::isfinite(length) || length <= 0.0 || !std::isfinite(maxDistance) ||
            !(minDistance >= 0.0f && maxDistance > minDistance)) {
            return std::nullopt;
        }
        return LumenGlobalTraceRay{
            {origin[0], origin[1], origin[2], minDistance},
            {static_cast<float>(direction[0] / length), static_cast<float>(direction[1] / length),
                static_cast<float>(direction[2] / length), maxDistance},
        };
    } catch (const RenderGraphProperties::exception&) {
        return std::nullopt;
    }
}

class LumenGlobalTracingPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    Device* device_ = nullptr;
    std::unique_ptr<Buffer> rays_;
    uint32_t rayCount_ = 0;
    ComputeKernel program_;
};

RenderPassReflection LumenGlobalTracingPass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    reflectLumenGlobalTraceInputs(reflection);
    const uint32_t count = std::max(1u, fixedRayCount(properties()));
    reflection.addBufferOutput("results", "Trace metrics, normal, radiance, state, candidate objects and dominant card")
        .buffer(count * sizeof(LumenGlobalTraceResult), sizeof(LumenGlobalTraceResult)).storageWrite();
    auto& radiance = reflection.addTextureOutput("radiance", "Scene-linear outgoing radiance in ray order")
        .texture2D(count, 1).storageWrite();
    radiance.format = Format::RGBA16Sfloat;
    radiance.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenGlobalTracingPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    rayCount_ = fixedRayCount(properties());
    if (rayCount_ == 0) {
        log += "Lumen Global Tracing requires 1 to 4096 fixed rays";
        return makeError(Error::InvalidArgument);
    }
    std::vector<LumenGlobalTraceRay> rays;
    rays.reserve(rayCount_);
    for (const auto& source : properties().at("rays")) {
        const auto ray = readRay(source);
        if (!ray) {
            log += "Invalid origin, direction or distance interval for Lumen fixed ray " + std::to_string(rays.size());
            return makeError(Error::InvalidArgument);
        }
        rays.push_back(*ray);
    }

    const auto bytes = std::as_bytes(std::span(rays));
    auto buffer = device_->createBuffer({.size = bytes.size(), .structureStride = sizeof(LumenGlobalTraceRay),
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
    if (!buffer) { return makeError(buffer.error()); }
    void* mapped = (*buffer)->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, bytes.data(), bytes.size());
    (*buffer)->flush();
    (*buffer)->unmap();
    rays_ = std::move(*buffer);

    return ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = "Features/DistanceField/LumenGlobalTracing", .entryPointName = "lumenGlobalTraceMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenGlobalTraceParams>(kTraceABI), .debugName = "LumenGlobalTracing"}, program_, log);
}

Result<> LumenGlobalTracingPass::execute(RenderGraphExecutionContext& context)
{
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    LumenGlobalTraceParams params{};
    params.scene = bindLumenGlobalTrace(context, writer);
    params.rays = writer.buffer(rays_.get());
    params.results = writer.buffer(context.outputBuffer("results").buffer());
    params.radiance = writer.storageImageHandle(context.outputTexture("radiance").view());
    params.rayCount = rayCount_;
    params.clipmapCount = *clipmapCount;
    const auto encoded = writer.encode(params, kTraceABI);
    if (!encoded) { return makeError(encoded.error()); }
    return program_.dispatch(commands, *encoded, (rayCount_ + 63u) / 64u, 1);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenGlobalTracingPass()
{
    return std::make_unique<LumenGlobalTracingPass>();
}

} // namespace metallic::render::builtin_pass
