#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenScreenProbePlacement.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kCompactProbeAtlasPlacementABI = 0x4c554d454e010020ull;
constexpr uint32_t kAdaptiveProbesPerTile = kLumenScreenProbeSlotsPerTile - 1;

struct CompactProbeAtlasPlacementParams {
    ShaderSampledImage position, normal;
    ShaderBuffer positions, origins, masks, adaptiveCount, tileCounts, tileIndices;
    uint32_t width, height, probeWidth, probeHeight, maxAdaptiveProbes, frameIndex;
};
static_assert(sizeof(CompactProbeAtlasPlacementParams) == 56);

class LumenScreenProbePlacementPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        // A buffer-only pass must not constrain Deferred's receivers to the presentation extent.
        auto& position = reflection.addTextureInput("position", "Screen receiver world position and linear depth").sampledRead();
        position.format = Format::RGBA32Sfloat;
        position.matchOutputExtent = false;
        auto& normal = reflection.addTextureInput("normal", "Screen receiver world normal").sampledRead();
        normal.format = Format::RGBA16Sfloat;
        normal.matchOutputExtent = false;
        const uint32_t uniformCount = lumenScreenProbeTiles(context.width) * lumenScreenProbeTiles(context.height);
        LumenCompactProbePlacement::reflect(reflection, uniformCount,
            uniformCount + lumenAdaptiveProbeCapacity(properties(), uniformCount), false);
        return reflection;
    }

    Result<> prepare(const RenderGraphCompileContext&, std::string& log) override
    {
        if (!lumenAdaptiveProbeFraction(properties())) {
            log = "Lumen screen probe placement requires adaptiveProbeAllocationFraction from 0 to 8";
            return makeError(Error::InvalidArgument);
        }
        return {};
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (auto result = prepare(context, log); !result) { return result; }
        device_ = context.device;
        return placement_.compile(*device_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto& position = context.inputTexture("position").desc();
        const auto& normal = context.inputTexture("normal").desc();
        if (position.width != context.width() || position.height != context.height() ||
                normal.width != context.width() || normal.height != context.height()) {
            spdlog::error("Lumen screen probe placement guides must match the graph render extent");
            return makeError(Error::InvalidArgument);
        }
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(context.commandBuffer()));
        const uint32_t uniformCount = lumenScreenProbeTiles(context.width()) * lumenScreenProbeTiles(context.height());
        std::vector<RenderGraphStage> stages;
        if (auto result = placement_.appendStages(context, writer, lumenAdaptiveProbeCapacity(properties(), uniformCount), stages);
            !result) {
            return result;
        }
        return context.executeStages(stages);
    }

private:
    Device* device_ = nullptr;
    LumenCompactProbePlacement placement_;
};

} // namespace

std::optional<double> lumenAdaptiveProbeFraction(const RenderGraphProperties& properties)
{
    const auto fraction = properties.find("adaptiveProbeAllocationFraction");
    if (fraction == properties.end()) { return kLumenAdaptiveProbeAllocationFraction; }
    if (!fraction->is_number()) { return std::nullopt; }
    const double value = fraction->get<double>();
    return value >= 0.0 && value <= kAdaptiveProbesPerTile ? std::optional<double>{value} : std::nullopt;
}

uint32_t lumenAdaptiveProbeCapacity(const RenderGraphProperties& properties, uint32_t uniformCount)
{
    // Reflection runs before compile validates properties.
    return uint32_t(float(uniformCount) * float(lumenAdaptiveProbeFraction(properties).value_or(0.0)));
}

void LumenCompactProbePlacement::reflect(RenderPassReflection& reflection, uint32_t uniformCount, uint64_t capacity, bool input)
{
    const auto field = [&](const char* name, const char* description, uint64_t size, uint32_t stride) -> RenderGraphField& {
        if (input) { return reflection.addBufferInput(name, description).buffer(size, stride).storageRead(); }
        return reflection.addBufferOutput(name, description).buffer(size, stride).storageReadWrite();
    };
    constexpr uint32_t word = sizeof(uint32_t);
    field("probePositions", "Uniform and adaptive receiver pixels in compact atlas order", capacity * 2 * word, 2 * word);
    field("probeOrigins", "Original tile and candidate slot per allocated atlas probe", capacity * word, word).bufferLayout("u32");
    auto& count = field("probeAdaptiveCount", "Attempted adaptive allocations before the global cap", word, word).bufferLayout("u32");
    if (!input) { count.transient(RenderGraphInitialization::FullOverwrite); }
    auto& tileCounts = field("probeTileAdaptiveCounts", "Admitted adaptive probes per screen tile", uint64_t(uniformCount) * word, word);
    if (!input) {
        tileCounts.transient(RenderGraphInitialization::FullOverwrite);
        reflection.addBufferOutput("probePlacementMasks", "Marked adaptive probe candidates per screen tile")
            .buffer(uint64_t(uniformCount) * word, word).storageReadWrite().transient(RenderGraphInitialization::FullOverwrite);
    }
    field("probeTileAdaptiveIndices", "Per-tile list of admitted adaptive suffix indices",
        uint64_t(uniformCount) * kAdaptiveProbesPerTile * word, word).bufferLayout("u32");
}

Result<> LumenCompactProbePlacement::compile(Device& device, std::string& log)
{
    const std::pair<const char*, ComputeKernel*> kernels[]{
        {"lumenResetCompactProbeAtlasMain", &reset_}, {"lumenMarkCompactProbeAtlasMain", &mark_},
        {"lumenSpawnCompactProbeAtlasMain", &spawn_},
    };
    for (const auto& [entry, kernel] : kernels) {
        if (auto result = ShaderRegistry::instance().getComputeKernel(device,
                {.moduleName = "Features/DistanceField/LumenScreenProbeAtlasPlacement", .entryPointName = entry,
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<CompactProbeAtlasPlacementParams>(kCompactProbeAtlasPlacementABI),
                    .debugName = entry}, *kernel, log); !result) {
            return result;
        }
    }
    return {};
}

Result<> LumenCompactProbePlacement::appendStages(RenderGraphExecutionContext& context, ParameterWriter& writer,
    uint32_t adaptiveCapacity, std::vector<RenderGraphStage>& stages)
{
    const auto bind = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
    const uint32_t probeWidth = lumenScreenProbeTiles(context.width());
    const uint32_t probeHeight = lumenScreenProbeTiles(context.height());
    const auto encoded = writer.encode(CompactProbeAtlasPlacementParams{
        writer.sampledImageHandle(context.inputTexture("position").view()),
        writer.sampledImageHandle(context.inputTexture("normal").view()),
        bind("probePositions"), bind("probeOrigins"), bind("probePlacementMasks"), bind("probeAdaptiveCount"),
        bind("probeTileAdaptiveCounts"), bind("probeTileAdaptiveIndices"), context.width(), context.height(),
        probeWidth, probeHeight, adaptiveCapacity, uint32_t(context.frameIndex())}, kCompactProbeAtlasPlacementABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    static constexpr RenderGraphStageUse resetUses[]{{"probeAdaptiveCount", Access::BufferStorageWrite}};
    static constexpr RenderGraphStageUse markUses[]{{"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"probePlacementMasks", Access::BufferStorageWrite}, {"probeTileAdaptiveCounts", Access::BufferStorageWrite},
        {"probePositions", Access::BufferStorageWrite}, {"probeOrigins", Access::BufferStorageWrite}};
    static constexpr RenderGraphStageUse spawnUses[]{{"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"probePlacementMasks", Access::BufferStorageRead}, {"probeAdaptiveCount", Access::BufferStorageReadWrite},
        {"probeTileAdaptiveCounts", Access::BufferStorageReadWrite}, {"probeTileAdaptiveIndices", Access::BufferStorageWrite},
        {"probePositions", Access::BufferStorageWrite}, {"probeOrigins", Access::BufferStorageWrite}};
    // UE's 8x8 placement group covers 2x4 screen tiles with eight samples per tile.
    const uint32_t groupsX = (probeWidth + 1u) / 2u;
    const uint32_t groupsY = (probeHeight + 3u) / 4u;
    stages.push_back({"Reset adaptive probe counter", resetUses, [this, params = *encoded](CommandBuffer& cmd) {
        return reset_.dispatch(cmd, params, 1, 1, 1);
    }});
    stages.push_back({"Mark adaptive probe candidates", markUses, [this, params = *encoded, groupsX, groupsY](CommandBuffer& cmd) {
        return mark_.dispatch(cmd, params, groupsX, groupsY, 1);
    }});
    if (adaptiveCapacity != 0u) {
        stages.push_back({"Allocate adaptive probes in the global atlas", spawnUses,
            [this, params = *encoded, groupsX, groupsY](CommandBuffer& cmd) {
                return spawn_.dispatch(cmd, params, groupsX, groupsY, 1);
            }});
    }
    return {};
}

std::unique_ptr<RenderGraphPass> createLumenScreenProbePlacementPass()
{
    return std::make_unique<LumenScreenProbePlacementPass>();
}

} // namespace metallic::render::builtin_pass
