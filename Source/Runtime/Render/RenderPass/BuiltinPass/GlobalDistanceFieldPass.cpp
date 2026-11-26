#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/DistanceField/MeshDistanceFieldGPU.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kGlobalDistanceFieldABI = 0x4744460000000001ull;
constexpr uint64_t kGlobalDistanceFieldDebugABI = 0x4744464400000001ull;
constexpr uint64_t kGlobalDistanceFieldPagesABI = 0x4744465000000003ull;
constexpr uint64_t kPersistentGlobalDistanceFieldPagesABI = 0x4744465000000005ull;
constexpr uint64_t kGlobalDistanceFieldUpdateStatsABI = 0x4744465500000001ull;
constexpr uint64_t kGlobalDistanceFieldPagesDebugABI = 0x4744504400000001ull;
constexpr uint64_t kGlobalDistanceFieldMipABI = 0x4744464d00000003ull;
constexpr uint64_t kQuantizedGlobalDistanceFieldDebugABI = 0x4744514400000001ull;
constexpr uint32_t kDefaultResolution = 64;
constexpr float kInfluenceRangeInVoxels = 4.0f;
constexpr uint32_t kCellsPerPage = 7;
constexpr uint32_t kCullGridPagesPerAxis = 4;
constexpr uint32_t kCullGridFullScan = UINT32_MAX;
constexpr uint32_t kCullGridWordBudget = 64u * 1024u * 1024u / sizeof(uint32_t);
constexpr uint32_t kSamplesPerPage = kCellsPerPage + 1;
constexpr uint32_t kDefaultPhysicalPages = 216;
constexpr uint32_t kFreePageInitRowSize = 65536;
constexpr uint32_t kPageStatisticsCount = 5;
constexpr uint32_t kUpdateStatisticsCount = 4;
constexpr uint32_t kObjectGridCellsPerPage = 64;
constexpr uint32_t kObjectGridCellStride = 4 * sizeof(uint32_t);
constexpr uint32_t kMipFactor = 4;
constexpr uint32_t kMaxClipmaps = 6;
constexpr uint32_t kClipmapResolutionAlignment = kMipFactor * kCellsPerPage;

struct alignas(16) GlobalDistanceFieldDomain {
    std::array<float, 4> minAndVoxelSize;
    std::array<float, 4> maxAndInfluenceRadius;
    bool operator==(const GlobalDistanceFieldDomain&) const = default;
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
    uint32_t instanceCount, logicalResolution, maxPhysicalPages, clipmapCount;
    ShaderBuffer objectGridObjects, objectGrid, objectGridScene;
    std::array<uint32_t, 2> sceneIdentity;
    ShaderBuffer cullGrid;
    uint32_t cullGridSide;
    std::array<GlobalDistanceFieldDomain, kMaxClipmaps> domains;
};

using ClipmapPageOrigins = std::array<std::array<double, 3>, kMaxClipmaps>;
using ClipmapPageMovements = std::array<std::array<int32_t, 4>, kMaxClipmaps>;

struct alignas(16) PersistentGlobalDistanceFieldPagesParams {
    GlobalDistanceFieldPagesParams pages;
    ShaderStorageImage pageHistory;
    ShaderBuffer freeCount, freePages;
    uint32_t fullUpdate;
    ClipmapPageMovements pageMovements;
    GPUBufferSpan dirtyPages;
};

struct alignas(16) GlobalDistanceFieldUpdateStatsParams {
    ShaderBuffer updateStatistics;
    uint32_t fullUpdate, updatedPages, preservedPages, updatedLevels;
};

struct alignas(16) GlobalDistanceFieldMipParams {
    ShaderSampledImage pageTable, pageAtlas;
    ShaderSampler sampler;
    ShaderStorageImage previousMip, outputMip;
    uint32_t logicalResolution, coarseResolution, clipmapCount, updateMask;
};

struct PersistentPageState {
    uint64_t atlasAllocation, sceneIdentity, structuralRevision;
    std::array<uint64_t, 3> instanceRevisions;
    ClipmapPageOrigins origins;
    std::shared_ptr<const std::vector<MeshDistanceFieldGPU::UpdateInstance>> instances;
};

struct PersistentPageUpdate {
    bool full;
    uint32_t levelMask = 0;
    uint32_t updatedPages = 0;
    uint32_t totalPages = 0;
    ClipmapPageMovements movements{};
    std::vector<uint32_t> dirtyPages;
};

struct MeshDistanceFieldBakeRevision {
    std::filesystem::file_time_type writeTime;
    uintmax_t bytes;
    bool operator==(const MeshDistanceFieldBakeRevision&) const = default;
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
static_assert(sizeof(GlobalDistanceFieldPagesParams) == 272 &&
    offsetof(GlobalDistanceFieldPagesParams, clipmapCount) == 48 &&
    offsetof(GlobalDistanceFieldPagesParams, objectGridObjects) == 52 &&
    offsetof(GlobalDistanceFieldPagesParams, sceneIdentity) == 64 &&
    offsetof(GlobalDistanceFieldPagesParams, cullGrid) == 72 &&
    offsetof(GlobalDistanceFieldPagesParams, cullGridSide) == 76 &&
    offsetof(GlobalDistanceFieldPagesParams, domains) == 80);
static_assert(sizeof(GlobalDistanceFieldPagesDebugParams) == 112 &&
    offsetof(GlobalDistanceFieldPagesDebugParams, current) == 32);
static_assert(sizeof(PersistentGlobalDistanceFieldPagesParams) == 400 &&
    offsetof(PersistentGlobalDistanceFieldPagesParams, pageHistory) == 272 &&
    offsetof(PersistentGlobalDistanceFieldPagesParams, fullUpdate) == 284 &&
    offsetof(PersistentGlobalDistanceFieldPagesParams, pageMovements) == 288 &&
    offsetof(PersistentGlobalDistanceFieldPagesParams, dirtyPages) == 384);
static_assert(sizeof(GlobalDistanceFieldUpdateStatsParams) == 32 &&
    offsetof(GlobalDistanceFieldUpdateStatsParams, updatedLevels) == 16);
static_assert(sizeof(GlobalDistanceFieldMipParams) == 48 && offsetof(GlobalDistanceFieldMipParams, updateMask) == 32);
static_assert(sizeof(QuantizedGlobalDistanceFieldDebugParams) == 112 &&
    offsetof(QuantizedGlobalDistanceFieldDebugParams, current) == 32);

uint32_t coarseMipResolution(uint32_t resolution)
{
    return (resolution + kMipFactor - 1u) / kMipFactor;
}

uint32_t readClipmapCount(const RenderGraphProperties& properties)
{
    const auto value = properties.find("clipmapCount");
    if (value == properties.end()) { return 1; }
    if (!value->is_number_integer()) { return 0; }
    const uint64_t count = value->get<uint64_t>();
    return count >= 1 && count <= kMaxClipmaps ? static_cast<uint32_t>(count) : 0;
}

uint32_t logicalResolution(const RenderGraphProperties& properties, uint32_t clipmapCount)
{
    const auto value = properties.find("resolution");
    if (value != properties.end() && !value->is_number_integer()) { return 0; }
    const uint64_t resolution = value == properties.end() ? kDefaultResolution : value->get<uint64_t>();
    if (resolution < 2 || resolution > UINT32_MAX - 4) { return 0; }
    if (clipmapCount <= 1) { return static_cast<uint32_t>(resolution); }
    // UE 5.7.4 aligns clipmaps to both the page and coarse mip lattices.
    const uint64_t aligned = (resolution + kClipmapResolutionAlignment - 1) /
        kClipmapResolutionAlignment * kClipmapResolutionAlignment;
    const uint64_t pages = aligned / kCellsPerPage;
    return pages * pages <= UINT32_MAX / clipmapCount / pages ? static_cast<uint32_t>(aligned) : 0;
}

uint32_t pagesPerAxis(uint32_t resolution)
{
    return resolution / kCellsPerPage + (resolution % kCellsPerPage != 0);
}

struct GlobalDistanceFieldCullGrid {
    uint32_t side = 0;
    uint32_t fullScanCells = 0;
    uint32_t references = 0;
    std::vector<uint32_t> words;
};

GlobalDistanceFieldCullGrid buildGlobalDistanceFieldCullGrid(std::span<const GlobalDistanceFieldDomain> domains,
    uint32_t pageSide, uint32_t levelMask, std::span<const MeshDistanceFieldGPU::UpdateInstance> instances)
{
    const uint32_t side = (pageSide + kCullGridPagesPerAxis - 1u) / kCullGridPagesPerAxis;
    const uint32_t cellCount = side * side * side * static_cast<uint32_t>(domains.size());
    if (cellCount > kCullGridWordBudget / 2u) { return {}; }

    // UE 5.7.4 CullObjectsToGridCS uses a variable object list per 4x4x4 pages.
    // Scatter each instance's expanded bounds on the CPU instead of gathering all objects per GPU cell.
    const auto visitOverlappedCells = [&](auto&& visit) {
        for (uint32_t instance = 0; instance < instances.size(); ++instance) {
            const auto& bounds = instances[instance];
            for (uint32_t level = 0; level < domains.size(); ++level) {
                if ((levelMask & (1u << level)) == 0u) { continue; }
                const auto& domain = domains[level];
                const double cellSize = double(domain.minAndVoxelSize[3]) * (kCellsPerPage * kCullGridPagesPerAxis);
                // Cover the 1e-4 tolerance on Mesh SDF scale and orthogonality tests.
                const double radius = domain.maxAndInfluenceRadius[3] * 1.001;
                std::array<uint32_t, 3> first, end;
                for (size_t axis = 0; axis < first.size(); ++axis) {
                    const double origin = domain.minAndVoxelSize[axis];
                    // The GPU constructs the sample positions in float. Retain both cells at a shared boundary.
                    const double rounding = 16.0 * std::numeric_limits<float>::epsilon() *
                        std::max(std::abs(origin), std::abs(origin + side * cellSize));
                    const double lo = (bounds.minimum[axis] - radius - rounding - origin) / cellSize;
                    const double hi = (bounds.maximum[axis] + radius + rounding - origin) / cellSize;
                    first[axis] = static_cast<uint32_t>(std::clamp(std::ceil(lo) - 1.0, 0.0, double(side)));
                    end[axis] = static_cast<uint32_t>(std::clamp(std::floor(hi) + 1.0, 0.0, double(side)));
                }
                for (uint32_t z = first[2]; z < end[2]; ++z) {
                    for (uint32_t y = first[1]; y < end[1]; ++y) {
                        const uint32_t row = ((level * side + z) * side + y) * side;
                        for (uint32_t x = first[0]; x < end[0]; ++x) { visit(row + x, instance); }
                    }
                }
            }
        }
    };

    // Headers are (absolute word offset, candidate count), followed by dense instance indices.
    GlobalDistanceFieldCullGrid grid{.side = side, .words = std::vector<uint32_t>(2u * cellCount)};
    visitOverlappedCells([&](uint32_t cell, uint32_t) { ++grid.words[2u * cell + 1u]; });
    uint32_t next = static_cast<uint32_t>(grid.words.size());
    for (uint32_t cell = 0; cell < cellCount; ++cell) {
        auto& count = grid.words[2u * cell + 1u];
        if (count == 0u) { continue; }
        if (count == instances.size() || count > kCullGridWordBudget - next) {
            ++grid.fullScanCells;
            count = kCullGridFullScan;
        } else {
            grid.words[2u * cell] = next;
            next += count;
        }
    }
    grid.references = next - static_cast<uint32_t>(grid.words.size());
    grid.words.resize(next);
    if (grid.references != 0u) {
        visitOverlappedCells([&](uint32_t cell, uint32_t instance) {
            if (grid.words[2u * cell + 1u] != kCullGridFullScan) {
                grid.words[grid.words[2u * cell]++] = instance;
            }
        });
        for (uint32_t cell = 0; cell < cellCount; ++cell) {
            if (grid.words[2u * cell + 1u] != kCullGridFullScan) {
                grid.words[2u * cell] -= grid.words[2u * cell + 1u];
            }
        }
    }
    return grid;
}

Result<std::unique_ptr<Buffer>> uploadPageBuildBuffer(Device& device, std::span<const uint32_t> words)
{
    auto buffer = device.createBuffer({.size = words.size_bytes(), .structureStride = sizeof(uint32_t),
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
    if (!buffer) { return makeError(buffer.error()); }
    void* mapped = (*buffer)->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, words.data(), words.size_bytes());
    (*buffer)->flush();
    (*buffer)->unmap();
    return buffer;
}

PersistentPageUpdate planPersistentPageUpdate(const PersistentPageState& next, const PersistentPageState* previous,
    uint32_t pageSide, uint32_t clipmapCount)
{
    const uint32_t pagesPerLevel = pageSide * pageSide * pageSide;
    PersistentPageUpdate update{
        .full = !previous || previous->sceneIdentity != next.sceneIdentity ||
            previous->structuralRevision != next.structuralRevision,
        .totalPages = pagesPerLevel * clipmapCount,
    };
    if (!update.full) {
        for (uint32_t level = 0; level < clipmapCount; ++level) {
            auto& movement = update.movements[level];
            uint32_t overlap = 1;
            for (size_t axis = 0; axis < 3; ++axis) {
                const double delta = next.origins[level][axis] - previous->origins[level][axis];
                if (std::abs(delta) >= pageSide) {
                    update.full = true;
                    break;
                }
                movement[axis] = static_cast<int32_t>(delta);
                overlap *= pageSide - static_cast<uint32_t>(std::abs(delta));
            }
            if (update.full) { break; }
            if (overlap != pagesPerLevel) {
                movement[3] = 1;
                update.levelMask |= 1u << level;
                update.updatedPages += pagesPerLevel - overlap;
            }
        }
    }
    if (update.full) {
        update.levelMask = (1u << clipmapCount) - 1u;
        update.updatedPages = update.totalPages;
    }
    return update;
}

void markDirtyInstancePages(PersistentPageUpdate& update, const MeshDistanceFieldGPU::UpdateInstance& instance,
    const std::array<GlobalDistanceFieldDomain, kMaxClipmaps>& domains, uint32_t pageSide, uint32_t clipmapCount)
{
    for (uint32_t level = 0; level < clipmapCount; ++level) {
        const auto& domain = domains[level];
        auto& delta = update.movements[level];
        const double pageSize = double(domain.minAndVoxelSize[3]) * kCellsPerPage;
        // The accepted Mesh SDF transform tolerates small scale and orthogonality errors.
        const double pageRadius = double(domain.maxAndInfluenceRadius[3]) * 1.001 / pageSize;
        const double pageRadiusSquared = pageRadius * pageRadius;
        std::array<double, 3> minimum{}, maximum{};
        std::array<int32_t, 3> first{}, last{};
        for (size_t axis = 0; axis < minimum.size(); ++axis) {
            const double origin = domain.minAndVoxelSize[axis];
            minimum[axis] = (instance.minimum[axis] - origin) / pageSize;
            maximum[axis] = (instance.maximum[axis] - origin) / pageSize;
            // Newly exposed scroll slabs are already counted and recomposed.
            first[axis] = static_cast<int32_t>(std::clamp(std::ceil(minimum[axis] - pageRadius) - 1.0,
                double(std::max(0, -delta[axis])), double(pageSide)));
            last[axis] = static_cast<int32_t>(std::clamp(std::floor(maximum[axis] + pageRadius),
                -1.0, double(int32_t(pageSide) - 1 - std::max(0, delta[axis]))));
        }
        for (int32_t z = first[2]; z <= last[2]; ++z) {
            const double dz = std::max({minimum[2] - (z + 1), z - maximum[2], 0.0});
            for (int32_t y = first[1]; y <= last[1]; ++y) {
                const double dy = std::max({minimum[1] - (y + 1), y - maximum[1], 0.0});
                for (int32_t x = first[0]; x <= last[0]; ++x) {
                    const double dx = std::max({minimum[0] - (x + 1), x - maximum[0], 0.0});
                    // UE 5.7.4 BuildGridTilesCS uses squared distance between the tile and modified bounds.
                    if (dx * dx + dy * dy + dz * dz > pageRadiusSquared) { continue; }
                    const uint32_t page = x + pageSide * (y + pageSide * (z + pageSide * level));
                    if (update.dirtyPages.empty()) { update.dirtyPages.resize((update.totalPages + 31u) / 32u); }
                    const uint32_t bit = 1u << (page % 32u);
                    auto& word = update.dirtyPages[page / 32u];
                    if ((word & bit) == 0) {
                        word |= bit;
                        ++update.updatedPages;
                        delta[3] = 1;
                        update.levelMask |= 1u << level;
                    }
                }
            }
        }
    }
}

void markChangedInstancePages(PersistentPageUpdate& update, const PersistentPageState& previous,
    const PersistentPageState& next, const std::array<GlobalDistanceFieldDomain, kMaxClipmaps>& domains,
    uint32_t pageSide, uint32_t clipmapCount)
{
    // Both snapshots follow scene render-node order. Structural changes already force a full update.
    const auto& before = *previous.instances;
    const auto& after = *next.instances;
    const auto mark = [&](const MeshDistanceFieldGPU::UpdateInstance& instance) {
        markDirtyInstancePages(update, instance, domains, pageSide, clipmapCount);
    };
    size_t oldIndex = 0, newIndex = 0;
    while (oldIndex < before.size() && newIndex < after.size()) {
        const auto& oldInstance = before[oldIndex];
        const auto& newInstance = after[newIndex];
        if (oldInstance.renderNode < newInstance.renderNode) {
            mark(oldInstance);
            ++oldIndex;
        } else if (newInstance.renderNode < oldInstance.renderNode) {
            mark(newInstance);
            ++newIndex;
        } else {
            if (oldInstance != newInstance) {
                mark(oldInstance);
                mark(newInstance);
            }
            ++oldIndex;
            ++newIndex;
        }
    }
    for (; oldIndex < before.size(); ++oldIndex) { mark(before[oldIndex]); }
    for (; newIndex < after.size(); ++newIndex) { mark(after[newIndex]); }
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

bool setDomainBounds(GlobalDistanceFieldDomain& domain, const std::array<double, 3>& center, float extent)
{
    for (size_t axis = 0; axis < center.size(); ++axis) {
        const float minimum = static_cast<float>(center[axis] - extent);
        const float maximum = static_cast<float>(center[axis] + extent);
        if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum >= maximum) { return false; }
        domain.minAndVoxelSize[axis] = minimum;
        domain.maxAndInfluenceRadius[axis] = maximum;
    }
    return true;
}

class GlobalDistanceFieldPass final : public ComputePass {
public:
    explicit GlobalDistanceFieldPass(bool usePages) : usePages_(usePages) {}

    bool supportsFrameOverlap() const override { return usePages_; }
    bool supportsExternalFrameOverlap() const override { return false; }
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {.source = RenderGraphSceneSource::World, .recompileOnTransform = false};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        const uint32_t clipmapCount = std::max(1u, readClipmapCount(properties()));
        const uint32_t resolution = logicalResolution(properties(), clipmapCount);
        if (usePages_) {
            const uint32_t tableSize = std::max(1u, pagesPerAxis(resolution));
            const uint32_t pageBudget = physicalPageBudget(properties()).value_or(0);
            const uint32_t atlasSize = pageAtlasSize(pageBudget);
            const bool quantized = quantizedPages();
            auto& pageTable = reflection.addTextureOutput("pageTable", "Virtual page to atlas mapping or constant page state")
                .texture3D(tableSize, tableSize, tableSize * clipmapCount).storageWrite();
            pageTable.format = Format::R32Uint;
            if (quantized) { pageTable.stageAccess(RenderGraphResourceAccess::TextureSampleRead); }
            if (persistentPagesRequested()) {
                pageTable.storageReadWrite();
                reflection.addTextureOutput("pageHistory", "Logical page table before scrolling")
                    .texture3D(tableSize, tableSize, tableSize * clipmapCount).storageReadWrite().format = Format::R32Uint;
                reflection.addBufferOutput("freeCount", "Number of unallocated physical pages")
                    .buffer(sizeof(int32_t), sizeof(int32_t)).storageReadWrite();
                reflection.addBufferOutput("freePages", "Physical atlas slots available for allocation")
                    .buffer(uint64_t(std::max(1u, pageBudget)) * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite();
                reflection.addBufferOutput("updateStatistics", "Full refresh, updated pages, preserved pages and updated levels")
                    .buffer(kUpdateStatisticsCount * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").storageWrite();
            }
            auto& pageAtlas = reflection.addTextureOutput("pageAtlas", "Physical pages of clipped signed distances")
                .texture3D(atlasSize, atlasSize, atlasSize).storageWrite();
            pageAtlas.format = quantized ? Format::R8Unorm : Format::R32Sfloat;
            if (quantized) {
                pageAtlas.stageAccess(RenderGraphResourceAccess::TextureSampleRead);
                const uint32_t mipSize = std::max(1u, coarseMipResolution(resolution));
                reflection.addTextureOutput("coarseMip", "Propagated coarse distances encoded into R8")
                    .texture3D(mipSize, mipSize, mipSize * clipmapCount).storageReadWrite().format = Format::R8Unorm;
                reflection.addTextureOutput("mipScratch", "Ping-pong storage for coarse distance propagation")
                    .texture3D(mipSize, mipSize, mipSize * clipmapCount).storageReadWrite().format = Format::R8Unorm;
                if (objectGridRequested()) {
                    const uint64_t gridSize = uint64_t(std::max(1u, pageBudget)) * kObjectGridCellsPerPage * kObjectGridCellStride;
                    reflection.addBufferOutput("objectGrid", "Ranked scene object candidates per physical page cell")
                        .buffer(gridSize, kObjectGridCellStride).storageWrite();
                    reflection.addBufferOutput("objectGridScene", "Scene resource identity (low/high)")
                        .buffer(2 * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").storageWrite();
                }
            }
            reflection.addBufferOutput("statistics", "Required, resident, positive, negative and unavailable pages")
                .buffer(kPageStatisticsCount * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").storageReadWrite();
        } else {
            // The physical lattice has one additional sample at the positive edge.
            const uint32_t samples = resolution + 1u;
            reflection.addTextureOutput("distanceField", "Clipped signed world distance in meters")
                .texture3D(samples, samples, samples).storageWrite().format = Format::R32Sfloat;
        }
        reflection.addBufferOutput("domain", "World bounds, voxel size and influence radius per level")
            .buffer(sizeof(GlobalDistanceFieldDomain) * clipmapCount, sizeof(GlobalDistanceFieldDomain)).storageWrite();
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.device || !context.graphicsQueue || !context.runtimeScene) {
            log = "Global Distance Field requires a world scene and a graphics queue";
            return makeError(Error::InvalidArgument);
        }
        const bool quantized = quantizedPages();
        const auto persistentPages = properties().find("persistentPages");
        if (persistentPages != properties().end() &&
            (!persistentPages->is_boolean() || (persistentPagesRequested() && !quantized))) {
            log = "Global Distance Field persistentPages must be a boolean and requires quantized pages when enabled";
            return makeError(Error::InvalidArgument);
        }
        const bool persistent = persistentPagesRequested();
        const auto objectGrid = properties().find("buildObjectGrid");
        if (objectGrid != properties().end() && (!objectGrid->is_boolean() || (objectGridRequested() && !quantized))) {
            log = "Global Distance Field buildObjectGrid must be a boolean and requires quantized pages when enabled";
            return makeError(Error::InvalidArgument);
        }
        const bool buildObjectGrid = objectGridRequested();
        const auto cullGrid = properties().find("buildCullGrid");
        if (cullGrid != properties().end() && !cullGrid->is_boolean()) {
            log = "Global Distance Field buildCullGrid must be a boolean";
            return makeError(Error::InvalidArgument);
        }
        const uint32_t clipmapCount = readClipmapCount(properties());
        if (clipmapCount == 0 || (clipmapCount > 1 && !quantized)) {
            log = "Global Distance Field clipmapCount must be an integer from 1 to 6. Multiple levels require quantized pages.";
            return makeError(Error::InvalidArgument);
        }
        if (quantized && !context.device->capabilities().shaderStorageImageExtendedFormats) {
            log = "Quantized Global Distance Field requires R8 storage image support";
            return makeError(Error::Unsupported);
        }
        const uint32_t resolution = logicalResolution(properties(), clipmapCount);
        if (resolution == 0) {
            log = "Global Distance Field resolution must be an integer of at least 2 and fit the clipmap layout";
            return makeError(Error::InvalidArgument);
        }
        uint32_t maxPhysicalPages = 0;
        if (usePages_) {
            const auto budget = physicalPageBudget(properties());
            const uint64_t pages = pagesPerAxis(resolution);
            if (!budget || pages * pages > UINT32_MAX / pages) {
                log = "Global Distance Field page budget or virtual page count exceeds its 32-bit range";
                return makeError(Error::InvalidArgument);
            }
            maxPhysicalPages = *budget;
            if (persistent && (maxPhysicalPages > INT32_MAX ||
                pages * pages * pages * clipmapCount > INT32_MAX)) {
                log = "Persistent Global Distance Field page counts exceed the signed allocator range";
                return makeError(Error::InvalidArgument);
            }
            if (buildObjectGrid && maxPhysicalPages > UINT32_MAX / kObjectGridCellsPerPage) {
                log = "Global Distance Field object grid exceeds its 32-bit cell index range";
                return makeError(Error::InvalidArgument);
            }
        }
        std::vector<std::filesystem::path> paths;
        std::array<float, kMaxClipmaps> extents{};
        std::array<GlobalDistanceFieldDomain, kMaxClipmaps> domains{};
        try {
            const float extent = properties().value("extent", 3.0f);
            const auto center = clipmapCount == 1 ? properties().value("center", std::array<double, 3>{}) :
                std::array<double, 3>{};
            for (uint32_t level = 0; level < clipmapCount; ++level) {
                extents[level] = std::ldexp(extent, static_cast<int>(level));
                auto& domain = domains[level];
                if (!setDomainBounds(domain, center, extents[level])) {
                    log = "Global Distance Field requires finite bounds with a positive extent";
                    return makeError(Error::InvalidArgument);
                }
                const float voxelSize = (extents[level] * 2.0f) / static_cast<float>(resolution);
                domain.minAndVoxelSize[3] = voxelSize;
                domain.maxAndInfluenceRadius[3] = voxelSize * kInfluenceRangeInVoxels;
                if (voxelSize <= 0.0f || !std::isfinite(domain.maxAndInfluenceRadius[3])) {
                    log = "Global Distance Field voxel size or influence radius is out of range";
                    return makeError(Error::InvalidArgument);
                }
            }
            const auto fields = properties().value("distanceFieldPaths", std::vector<std::string>{});
            for (const auto& path : fields) { paths.push_back(std::filesystem::path(PROJECT_SOURCE_DIR) / path); }
        } catch (const nlohmann::json::exception& error) {
            log = "Invalid Global Distance Field properties: " + std::string(error.what());
            return makeError(Error::InvalidArgument);
        }

        const auto& scene = *context.runtimeScene;
        const std::array meshSource{scene.resourceIdentity(), scene.sceneGraph().lifetimeRevision(),
            scene.sceneGraph().structuralRevision()};
        std::vector<MeshDistanceFieldBakeRevision> bakeRevisions;
        bakeRevisions.reserve(paths.size());
        for (const auto& path : paths) {
            std::error_code error;
            const auto writeTime = std::filesystem::last_write_time(path, error);
            if (error) { break; }
            const auto bytes = std::filesystem::file_size(path, error);
            if (error) { break; }
            bakeRevisions.push_back({writeTime, bytes});
        }
        const bool reloadMeshes = device_ != context.device || meshSource != meshSource_ || paths != meshPaths_ ||
            bakeRevisions.size() != paths.size() || bakeRevisions != meshBakeRevisions_;
        if (reloadMeshes) {
            const auto result = meshFields_.initialize(*context.device, *context.graphicsQueue, scene, paths, log);
            if (!result) {
                if (log.empty()) { log = "Global Distance Field mesh atlas upload failed: " + std::string(resultToString(result)); }
                return result;
            }
            meshSource_ = meshSource;
            meshPaths_ = std::move(paths);
            meshBakeRevisions_ = std::move(bakeRevisions);
        }
        // The executor recompiles world-bound passes on scene edits while
        // retaining their graph allocations. Keep their submitted page state.
        if (persistent) {
            if (!persistentPageState_ || reloadMeshes || buildObjectGrid_ != buildObjectGrid || domains_ != domains ||
                resolution_ != resolution || clipmapCount_ != clipmapCount || maxPhysicalPages_ != maxPhysicalPages) {
                persistentPageState_ = std::make_shared<std::optional<PersistentPageState>>();
            }
        } else {
            persistentPageState_.reset();
        }
        device_ = context.device;
        resolution_ = resolution;
        maxPhysicalPages_ = maxPhysicalPages;
        clipmapCount_ = clipmapCount;
        extents_ = extents;
        domains_ = domains;
        buildObjectGrid_ = buildObjectGrid;
        buildCullGrid_ = usePages_ && (cullGrid == properties().end() || cullGrid->get<bool>());
        Result<> result;
        if (usePages_) {
            auto& shaders = ShaderRegistry::instance();
            const auto pageABI = persistentPageState_ ?
                parameterAbi<PersistentGlobalDistanceFieldPagesParams>(kPersistentGlobalDistanceFieldPagesABI) :
                parameterAbi<GlobalDistanceFieldPagesParams>(kGlobalDistanceFieldPagesABI);
            std::vector<SlangMacroDefine> defines;
            if (quantized) { defines.push_back({"METALLIC_GDF_PAGE_ATLAS_R8", "1"}); }
            if (buildObjectGrid_) { defines.push_back({"METALLIC_GDF_BUILD_OBJECT_GRID", "1"}); }
            if (persistentPageState_) { defines.push_back({"METALLIC_GLOBAL_DISTANCE_FIELD_PERSISTENT_PAGES", "1"}); }
            const auto compilePages = [&](const char* entry, const char* name, ComputeKernel& kernel) {
                return shaders.getComputeKernel(*device_,
                    {.moduleName = "Features/DistanceField/GlobalDistanceFieldPageBuild", .entryPointName = entry,
                        .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = defines},
                    {.parameters = pageABI, .debugName = name}, kernel, log);
            };
            result = compilePages("globalDistanceFieldClearPageStatsMain", "GlobalDistanceFieldClearPageStats", clearPageStats_);
            if (!result) { return result; }
            result = compilePages("globalDistanceFieldBuildPagesMain",
                quantized ? "GlobalDistanceFieldBuildR8Pages" : "GlobalDistanceFieldBuildPages", program_);
            if (!result || !quantized) { return result; }
            if (persistentPageState_) {
                result = compilePages("globalDistanceFieldInitFreePagesMain", "GlobalDistanceFieldInitFreePages", initFreePages_);
                if (!result) { return result; }
                result = compilePages("globalDistanceFieldPreparePageScrollMain", "GlobalDistanceFieldPreparePageScroll", preparePageScroll_);
                if (!result) { return result; }
                result = compilePages("globalDistanceFieldFinalizeFreePagesMain", "GlobalDistanceFieldFinalizeFreePages", finalizeFreePages_);
                if (!result) { return result; }
                result = shaders.getComputeKernel(*device_,
                    {.moduleName = "Features/DistanceField/GlobalDistanceFieldPageBuild",
                        .entryPointName = "globalDistanceFieldPublishUpdateStatsMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                    {.parameters = parameterAbi<GlobalDistanceFieldUpdateStatsParams>(kGlobalDistanceFieldUpdateStatsABI),
                        .debugName = "GlobalDistanceFieldPublishUpdateStats"}, publishUpdateStats_, log);
                if (!result) { return result; }
            }
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
        auto domains = domains_;
        ClipmapPageOrigins origins{};
        if (clipmapCount_ > 1) {
            const auto* view = context.viewConstants();
            if (!view) {
                spdlog::error("Global Distance Field clipmaps require the graph camera");
                return makeError(Error::InvalidArgument);
            }
            for (uint32_t level = 0; level < clipmapCount_; ++level) {
                const double pageSize = domains[level].minAndVoxelSize[3] * float(kCellsPerPage);
                std::array<double, 3> center;
                for (size_t axis = 0; axis < center.size(); ++axis) {
                    // UE FMath::RoundToInt rounds halfway toward positive infinity.
                    origins[level][axis] = std::floor(view->current.eye[axis] / pageSize + 0.5);
                    center[axis] = origins[level][axis] * pageSize;
                }
                if (!setDomainBounds(domains[level], center, extents_[level])) {
                    spdlog::error("Global Distance Field clipmap {} bounds exceed world-space float precision", level);
                    return makeError(Error::InvalidArgument);
                }
            }
        }
        auto& commands = context.commandBuffer();
        auto* frame = RenderFrameContext::from(commands);
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, frame);
        const auto& scene = *context.runtimeScene();
        PersistentPageState nextState{};
        const PersistentPageState* previousState = nullptr;
        std::optional<PersistentPageUpdate> update;
        std::optional<EncodedParameters> encodedUpdate;
        bool instanceContentChanged = false;
        if (persistentPageState_) {
            if (!frame) {
                spdlog::error("Persistent Global Distance Field requires a tracked render frame");
                return makeError(Error::Unsupported);
            }
            nextState = {
                .atlasAllocation = context.outputTexture("pageAtlas").texture()->memoryInfo().allocationId,
                .sceneIdentity = scene.resourceIdentity(),
                .structuralRevision = scene.sceneGraph().structuralRevision(),
                .instanceRevisions = {scene.geometryTransformRevision(), scene.visibilityRevision(),
                    buildObjectGrid_ ? scene.materialRevision() : 0},
                .origins = origins,
            };
            const auto& previous = *persistentPageState_;
            if (previous && previous->atlasAllocation == nextState.atlasAllocation) { previousState = &*previous; }
            update = planPersistentPageUpdate(nextState, previousState, pagesPerAxis(resolution_), clipmapCount_);
            if (!update->full) {
                instanceContentChanged = previousState->instanceRevisions != nextState.instanceRevisions;
                nextState.instances = previousState->instances;
            }
        }
        uint32_t instanceCount = 0;
        uint32_t unsupportedTransforms = 0;
        std::unique_ptr<Buffer> instances, objectGridObjects;
        if (!update || update->levelMask != 0 || instanceContentChanged) {
            const bool captureInstances = update ? update->full || instanceContentChanged : buildCullGrid_;
            auto updateInstances = captureInstances ? std::make_shared<std::vector<MeshDistanceFieldGPU::UpdateInstance>>() :
                nullptr;
            auto uploaded = meshFields_.uploadInstances(scene, instanceCount, unsupportedTransforms,
                buildObjectGrid_ ? &objectGridObjects : nullptr, nullptr, nullptr, updateInstances.get());
            if (!uploaded) { return makeError(uploaded.error()); }
            instances = std::move(*uploaded);
            if (updateInstances) { nextState.instances = std::move(updateInstances); }
            if (instanceContentChanged) {
                markChangedInstancePages(*update, *previousState, nextState, domains, pagesPerAxis(resolution_),
                    clipmapCount_);
            }
        }
        const auto publishPageState = [&]() {
            return commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
                [state = persistentPageState_, nextState]() { *state = nextState; }, [] {}));
        };
        if (update) {
            const GlobalDistanceFieldUpdateStatsParams statistics{
                .updateStatistics = writer.buffer(context.outputBuffer("updateStatistics").buffer()),
                .fullUpdate = update->full,
                .updatedPages = update->updatedPages,
                .preservedPages = update->totalPages - update->updatedPages,
                .updatedLevels = static_cast<uint32_t>(std::popcount(update->levelMask)),
            };
            auto encoded = writer.encode(statistics, kGlobalDistanceFieldUpdateStatsABI);
            if (!encoded) { return makeError(encoded.error()); }
            encodedUpdate = std::move(*encoded);
            if (update->levelMask == 0) {
                const RenderGraphStageUse uses[] = {{"updateStatistics", RenderGraphResourceAccess::BufferStorageWrite}};
                const RenderGraphStage reuse[] = {{"Publish page update statistics", uses, [&](CommandBuffer& cmd) {
                    return publishUpdateStats_.dispatch(cmd, *encodedUpdate, 1);
                }}};
                auto result = context.executeComputeStages(reuse);
                if (result && instanceContentChanged) { return publishPageState(); }
                return result;
            }
        }
        const auto meshBindings = meshFields_.bind(writer);
        const auto bindMeshes = [&](auto& params) {
            params.instances = writer.buffer(instances.get());
            params.mips = meshBindings.mips;
            params.indirection = meshBindings.indirection;
            params.atlas = meshBindings.atlas;
            params.atlasSampler = meshBindings.atlasSampler;
            params.domainOutput = writer.buffer(context.outputBuffer("domain").buffer());
            params.instanceCount = instanceCount;
        };
        Result<> result;
        GlobalDistanceFieldCullGrid cullGrid;
        if (usePages_) {
            PersistentGlobalDistanceFieldPagesParams pageParams{};
            auto& params = pageParams.pages;
            bindMeshes(params);
            params.pageTable = writer.storageImageHandle(context.outputTexture("pageTable").view());
            params.pageAtlas = writer.storageImageHandle(context.outputTexture("pageAtlas").view());
            params.statistics = writer.buffer(context.outputBuffer("statistics").buffer());
            params.logicalResolution = resolution_;
            params.maxPhysicalPages = maxPhysicalPages_;
            params.clipmapCount = clipmapCount_;
            if (buildObjectGrid_) {
                params.objectGridObjects = writer.buffer(objectGridObjects.get());
                params.objectGrid = writer.buffer(context.outputBuffer("objectGrid").buffer());
                params.objectGridScene = writer.buffer(context.outputBuffer("objectGridScene").buffer());
                const uint64_t identity = scene.resourceIdentity();
                params.sceneIdentity = {uint32_t(identity), uint32_t(identity >> 32)};
            }
            params.domains = domains;
            std::unique_ptr<Buffer> cullGridBuffer;
            if (buildCullGrid_ && instanceCount != 0u) {
                cullGrid = buildGlobalDistanceFieldCullGrid(std::span(domains).first(clipmapCount_),
                    pagesPerAxis(resolution_), update ? update->levelMask : (1u << clipmapCount_) - 1u, *nextState.instances);
                if (!cullGrid.words.empty()) {
                    auto buffer = uploadPageBuildBuffer(*device_, cullGrid.words);
                    if (!buffer) { return makeError(buffer.error()); }
                    cullGridBuffer = std::move(*buffer);
                    params.cullGrid = writer.buffer(cullGridBuffer.get());
                    params.cullGridSide = cullGrid.side;
                }
            }
            if (update) {
                pageParams.pageHistory = writer.storageImageHandle(context.outputTexture("pageHistory").view());
                pageParams.freeCount = writer.buffer(context.outputBuffer("freeCount").buffer());
                pageParams.freePages = writer.buffer(context.outputBuffer("freePages").buffer());
                pageParams.fullUpdate = update->full;
                pageParams.pageMovements = update->movements;
                if (!update->dirtyPages.empty()) {
                    pageParams.dirtyPages = writer.dataSpan(update->dirtyPages.data(),
                        update->dirtyPages.size() * sizeof(uint32_t));
                }
            }
            auto encoded = update ? writer.encode(pageParams, kPersistentGlobalDistanceFieldPagesABI) :
                writer.encode(params, kGlobalDistanceFieldPagesABI);
            if (!encoded) { return makeError(encoded.error()); }
            result = executePages(context, *encoded, writer, update ? &*update : nullptr,
                encodedUpdate ? &*encodedUpdate : nullptr);
            if (result && update) { result = publishPageState(); }
        } else {
            GlobalDistanceFieldParams params{};
            bindMeshes(params);
            params.domain = domains[0];
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
                {"maxPhysicalPages", maxPhysicalPages_}, {"clipmapCount", clipmapCount_},
                {"cullGridCellsPerAxis", cullGrid.side}, {"cullGridReferences", cullGrid.references},
                {"cullGridFullScanCells", cullGrid.fullScanCells},
                {"minAndVoxelSize", domains[0].minAndVoxelSize}, {"maxAndInfluenceRadius", domains[0].maxAndInfluenceRadius}});
        }
        return result;
    }

    std::vector<std::string> debugCheckpoints() const override { return {"AfterPass", "GlobalDistanceField"}; }

private:
    bool quantizedPages() const { return usePages_ && properties().value("quantized", false); }
    bool persistentPagesRequested() const
    {
        const auto value = properties().find("persistentPages");
        return value != properties().end() && value->is_boolean() && value->get<bool>();
    }

    bool objectGridRequested() const
    {
        const auto value = properties().find("buildObjectGrid");
        return value != properties().end() && value->is_boolean() && value->get<bool>();
    }

    Result<> executePages(RenderGraphExecutionContext& context, const EncodedParameters& encoded, ParameterWriter& writer,
        const PersistentPageUpdate* update, const EncodedParameters* encodedUpdate)
    {
        using Access = RenderGraphResourceAccess;
        const RenderGraphStageUse clearUses[]{{"statistics", Access::BufferStorageWrite}};
        const RenderGraphStageUse initUses[]{{"freeCount", Access::BufferStorageWrite}, {"freePages", Access::BufferStorageWrite}};
        const RenderGraphStageUse scrollUses[]{{"pageTable", Access::TextureStorageRead},
            {"pageHistory", Access::TextureStorageWrite}, {"freeCount", Access::BufferStorageReadWrite},
            {"freePages", Access::BufferStorageWrite}};
        const RenderGraphStageUse finalizeUses[]{{"freeCount", Access::BufferStorageReadWrite}};
        const RenderGraphStageUse statsUses[]{{"updateStatistics", Access::BufferStorageWrite}};
        RenderGraphStageUse buildUses[9] = {
            {"pageTable", update && !update->full ? Access::TextureStorageReadWrite : Access::TextureStorageWrite},
            {"pageAtlas", Access::TextureStorageWrite},
            {"domain", Access::BufferStorageWrite}, {"statistics", Access::BufferStorageReadWrite},
        };
        size_t buildUseCount = 4;
        if (buildObjectGrid_) {
            buildUses[buildUseCount++] = {"objectGrid", Access::BufferStorageWrite};
            buildUses[buildUseCount++] = {"objectGridScene", Access::BufferStorageWrite};
        }
        if (update) {
            buildUses[buildUseCount++] = {"freeCount", Access::BufferStorageReadWrite};
            buildUses[buildUseCount++] = {"freePages", Access::BufferStorageRead};
            if (!update->full) { buildUses[buildUseCount++] = {"pageHistory", Access::TextureStorageRead}; }
        }
        const uint32_t groups = pagesPerAxis(resolution_);
        const uint32_t depth = groups * clipmapCount_;
        std::vector<RenderGraphStage> pageStages;
        pageStages.reserve(10);
        pageStages.push_back({"Clear page statistics", clearUses, [&](CommandBuffer& commands) {
            return clearPageStats_.dispatch(commands, encoded, 1);
        }});
        if (update) {
            if (update->full) {
                pageStages.push_back({"Initialize physical page pool", initUses, [&](CommandBuffer& commands) {
                    const uint32_t rowSize = std::min(maxPhysicalPages_, kFreePageInitRowSize);
                    return initFreePages_.dispatch(commands, encoded, std::max(1u, (rowSize + 63u) / 64u),
                        std::max(1u, (maxPhysicalPages_ + kFreePageInitRowSize - 1u) / kFreePageInitRowSize));
                }});
            } else {
                pageStages.push_back({"Reclaim dirty and leaving pages", scrollUses, [&](CommandBuffer& commands) {
                    const uint32_t groupsXY = (groups + 3u) / 4u;
                    return preparePageScroll_.dispatch(commands, encoded, groupsXY, groupsXY, (depth + 3u) / 4u);
                }});
            }
        }
        pageStages.push_back({"Compose sparse pages", {buildUses, buildUseCount}, [&](CommandBuffer& commands) {
            return program_.dispatch(commands, encoded, groups, groups, depth);
        }});
        if (update) {
            pageStages.push_back({"Finalize physical page pool", finalizeUses, [&](CommandBuffer& commands) {
                return finalizeFreePages_.dispatch(commands, encoded, 1);
            }});
            pageStages.push_back({"Publish page update statistics", statsUses, [&](CommandBuffer& commands) {
                return publishUpdateStats_.dispatch(commands, *encodedUpdate, 1);
            }});
        }
        if (!quantizedPages()) {
            return context.executeComputeStages(pageStages);
        }
        const uint32_t updateMask = update ? update->levelMask : (1u << clipmapCount_) - 1u;
        return executeMipStages(context, writer, pageStages, updateMask);
    }

    Result<> executeMipStages(RenderGraphExecutionContext& context, ParameterWriter& writer,
        std::vector<RenderGraphStage>& stages, uint32_t updateMask)
    {
        const auto coarse = writer.storageImageHandle(context.outputTexture("coarseMip").view());
        const auto scratch = writer.storageImageHandle(context.outputTexture("mipScratch").view());
        GlobalDistanceFieldMipParams params{};
        params.pageTable = writer.sampledImageHandle(context.outputTexture("pageTable").view());
        params.pageAtlas = writer.sampledImageHandle(context.outputTexture("pageAtlas").view());
        params.sampler = writer.samplerHandle({});
        params.logicalResolution = resolution_;
        params.coarseResolution = coarseMipResolution(resolution_);
        params.clipmapCount = clipmapCount_;
        params.updateMask = updateMask;
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
        const uint32_t depthGroups = (params.coarseResolution * clipmapCount_ + 3u) / 4u;
        const auto toScratchStage = [&](CommandBuffer& commands) {
            return propagateMip_.dispatch(commands, *toScratch, groups, groups, depthGroups);
        };
        const auto toCoarseStage = [&](CommandBuffer& commands) {
            return propagateMip_.dispatch(commands, *toCoarse, groups, groups, depthGroups);
        };
        // UE 5.7.4 GlobalDistanceField.cpp uses five R8 writes, including the
        // first Eikonal solve from fine pages. The fifth write lands in coarseMip.
        const RenderGraphStage mipStages[] = {
            {"Seed coarse mip from fine pages", seedUses, [&](CommandBuffer& commands) {
                return seedMip_.dispatch(commands, *toCoarse, groups, groups, depthGroups);
            }},
            {"Propagate coarse mip 1", scratchUses, toScratchStage},
            {"Propagate coarse mip 2", coarseUses, toCoarseStage},
            {"Propagate coarse mip 3", scratchUses, toScratchStage},
            {"Propagate coarse mip 4", coarseUses, toCoarseStage},
        };
        stages.insert(stages.end(), std::begin(mipStages), std::end(mipStages));
        return context.executeStages(stages);
    }

    bool usePages_;
    bool buildObjectGrid_ = false;
    bool buildCullGrid_ = false;
    Device* device_ = nullptr;
    uint32_t resolution_ = 0;
    uint32_t maxPhysicalPages_ = 0;
    uint32_t clipmapCount_ = 1;
    std::array<float, kMaxClipmaps> extents_{};
    std::array<GlobalDistanceFieldDomain, kMaxClipmaps> domains_{};
    ComputeKernel program_;
    ComputeKernel clearPageStats_;
    ComputeKernel initFreePages_, preparePageScroll_, finalizeFreePages_, publishUpdateStats_;
    ComputeKernel seedMip_, propagateMip_;
    MeshDistanceFieldGPU meshFields_;
    std::array<uint64_t, 3> meshSource_{};
    std::vector<std::filesystem::path> meshPaths_;
    std::vector<MeshDistanceFieldBakeRevision> meshBakeRevisions_;
    std::shared_ptr<std::optional<PersistentPageState>> persistentPageState_;
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
        reflection.addBufferInput("domain", "World bounds, voxel size and influence radius per level")
            .buffer(0, sizeof(GlobalDistanceFieldDomain)).storageRead();
        reflection.addTextureOutput("color", "Global distance field trace visualization")
            .storageWrite().format = Format::RGBA16Sfloat;
        reflection.addTextureOutput("trace", "World hit distance, steps, coverage state, surface expansion or clipmap index")
            .storageWrite().format = Format::RGBA32Sfloat;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        if (quantizedPages()) {
            return {runtimeEnumSetting("debugMode", "Display", "normals",
                {{"Normals", "normals"}, {"Trace Steps", "steps"}, {"Coverage", "coverage"}, {"Clipmap Levels", "clipmaps"}})};
        }
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
            params.debugMode = mode == "clipmaps" ? 3u : (mode == "coverage" ? 2u : (mode == "steps" ? 1u : 0u));
            params.current = context.viewConstants() ? context.viewConstants()->current :
                RenderView{}.constants(0, context.width(), context.height(), context.width(), context.height()).current;
            return writer.encode(params, abi);
        };
        const auto encoded = [&]() -> Result<EncodedParameters> {
            if (quantizedPages()) {
                const auto pageTable = context.inputTexture("pageTable");
                const auto coarseMip = context.inputTexture("coarseMip");
                const auto& table = pageTable.desc();
                const auto& coarse = coarseMip.desc();
                const uint32_t levels = coarse.depth / coarse.width;
                const auto& domain = context.inputBuffer("domain").desc();
                if (levels < 1 || levels > kMaxClipmaps || coarse.width != coarse.height ||
                    coarse.depth != coarse.width * levels || table.width != table.height ||
                    table.depth != table.width * levels || domain.size != levels * sizeof(GlobalDistanceFieldDomain)) {
                    spdlog::error("Global Distance Field page table, coarse mip and domain buffer must describe the same clipmap levels");
                    return makeError(Error::InvalidArgument);
                }
                QuantizedGlobalDistanceFieldDebugParams params{};
                params.pageTable = writer.sampledImageHandle(pageTable.view());
                params.pageAtlas = writer.sampledImageHandle(context.inputTexture("pageAtlas").view());
                params.coarseMip = writer.sampledImageHandle(coarseMip.view());
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
