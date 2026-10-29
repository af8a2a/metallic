#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Page = std::array<uint32_t, 4>;
using Mapping = std::array<uint32_t, 2>;
using Rectangle = std::array<uint32_t, 4>;
using Json = nlohmann::json;
constexpr uint32_t kInvalid = UINT32_MAX;
constexpr uint32_t kPool = 32;

void requireRemap(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}
template<typename T>
void requireRemap(const Result<T>& result, const std::string& message)
{
    requireRemap(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct RemapPush
{
    std::array<uint32_t, 52> words{};
};
static_assert(sizeof(RemapPush) == 208);

struct LevelChange
{
    int64_t dx = 0, dy = 0;
    std::vector<Rectangle> rectangles;
};
struct RemapCase
{
    std::string name;
    uint32_t axis = 7, lightType = 0;
    std::vector<LevelChange> levels = std::vector<LevelChange>(3);
    uint32_t dirtyBounds = 0;
    bool reuse = true, force = false, enabled = true;
};

uint32_t levelAxis(const RemapCase& input, uint32_t level)
{
    return input.lightType == 0 ? input.axis : std::max(1u, input.axis >> level);
}

// Retain the previous implementation's dense, per-cell policy as the oracle.
// It uses the original 64-bit shift, before the compact producer clamps it.
std::vector<Mapping> denseReference(const RemapCase& input)
{
    const uint32_t stride = input.axis * input.axis;
    std::vector<Mapping> result(input.levels.size() * stride + 1, {kInvalid, 1});
    if (!input.reuse) { return result; }
    for (uint32_t level = 0; level < input.levels.size(); ++level) {
        const int64_t axis = levelAxis(input, level);
        const auto& change = input.levels[level];
        for (int64_t y = 0; y < axis; ++y) {
            for (int64_t x = 0; x < axis; ++x) {
                const int64_t oldX = x + change.dx, oldY = y + change.dy;
                result[level * stride + y * axis + x] = {
                    oldX >= 0 && oldX < axis && oldY >= 0 && oldY < axis ?
                        uint32_t(level * stride + oldY * axis + oldX) : kInvalid, input.force ? 1u : 0u};
            }
        }
        for (const auto& rectangle : change.rectangles) {
            for (uint32_t y = rectangle[1]; y <= rectangle[3]; ++y) {
                for (uint32_t x = rectangle[0]; x <= rectangle[2]; ++x) {
                    result[level * stride + y * axis + x][1] = 1;
                }
            }
        }
    }
    result.back() = {uint32_t(result.size() - 1), input.dirtyBounds != 0 || input.force ? 1u : 0u};
    return result;
}

std::vector<Page> compactInput(const RemapCase& input)
{
    std::vector<Page> records(input.levels.size());
    for (uint32_t level = 0; level < input.levels.size(); ++level) {
        const int64_t axis = levelAxis(input, level);
        const auto& change = input.levels[level];
        records[level] = {uint32_t(int32_t(std::clamp(change.dx, -axis, axis))),
            uint32_t(int32_t(std::clamp(change.dy, -axis, axis))), uint32_t(records.size()),
            uint32_t(change.rectangles.size())};
        records.insert(records.end(), change.rectangles.begin(), change.rectangles.end());
    }
    return records;
}

std::vector<RemapCase> remapCases()
{
    std::vector<RemapCase> result;
    const auto append = [&](const char* name, uint32_t axis, uint32_t levels, uint32_t lightType) -> RemapCase& {
        result.push_back({.name = name, .axis = axis, .lightType = lightType,
            .levels = std::vector<LevelChange>(levels)});
        return result.back();
    };
    append("ColdNoReuse", 3, 2, 0).reuse = false;
    append("ColdDisabled", 1, 1, 0).reuse = false;
    result.back().enabled = false;
    append("SingleCell", 1, 1, 0);
    auto& scroll = append("SignedScrollNonPowerOfTwo", 7, 3, 0);
    scroll.levels[0].dx = 1; scroll.levels[0].dy = -2;
    scroll.levels[1].dx = -3; scroll.levels[1].dy = 1;
    auto& empty = append("ExactlyNoOverlap", 7, 3, 0);
    empty.levels[0].dx = 7; empty.levels[1].dy = -7; empty.levels[2].dx = -8;
    auto& huge = append("HugePositiveNegativeScroll", 127, 2, 0);
    huge.levels[0].dx = int64_t(1) << 50; huge.levels[1].dy = -(int64_t(1) << 50);
    append("ProductionDirectionalSize", 128, 6, 0);
    auto& edges = append("InclusiveDirtyEdgesAndOverlap", 7, 3, 0);
    edges.dirtyBounds = 4;
    edges.levels[0].rectangles = {{0, 0, 0, 0}, {6, 6, 6, 6}, {0, 3, 6, 3}, {2, 2, 4, 4}};
    edges.levels[1].rectangles = {{0, 0, 6, 6}, {0, 0, 6, 6}};
    edges.levels[2].rectangles = {{0, 0, 0, 6}};
    append("CoarseDirtyWithNoValidRectangle", 7, 3, 0).dirtyBounds = 1;
    append("ForceRedraw", 127, 3, 0).force = true;
    for (uint32_t type : {1u, 2u}) {
        auto& local = append(type == 1 ? "SpotPackedMipHoles" : "PointPackedMipHoles", 7, 5, type);
        local.dirtyBounds = 2;
        for (uint32_t level = 0; level < local.levels.size(); ++level) {
            const auto last = levelAxis(local, level) - 1;
            local.levels[level].rectangles = {{0, 0, 0, last}, {last, last, last, last}};
        }
    }
    append("PointForceRedrawHoles", 16, 6, 2).force = true;
    append("DisabledCached", 3, 2, 0).enabled = false;
    auto& many = append("ManyOverlappingDirtyRectangles", 128, 6, 0);
    many.dirtyBounds = 128;
    std::mt19937 random(0x72656d61u);
    for (auto& level : many.levels) {
        for (uint32_t index = 0; index < 128; ++index) {
            const uint32_t x = random() % 128, y = random() % 128;
            level.rectangles.push_back({x, y, std::min(127u, x + uint32_t(random() % 16)), std::min(127u, y + uint32_t(random() % 16))});
        }
    }
    return result;
}

class VirtualShadowMapRemapTest final : public RHITest {
public:
    VirtualShadowMapRemapTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_remap"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapRemap";
        Json report{{"passed", false}, {"cases", Json::array()}};
        bench::ValidationRecorder validation;
        try {
            std::filesystem::create_directories(directory);
            auto created = createDevice({.applicationName = "VSM compact page changes",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless mesh-capable Vulkan"); }
            requireRemap(created, "Create remap device");
            auto device = std::move(*created);
            auto queue = device->getQueue(QueueType::Graphics);
            requireRemap(queue != nullptr, "Remap graphics queue");
            auto registryResult = ResourceRegistry::forDevice(*device);
            requireRemap(registryResult, "Remap registry");
            auto registry = *registryResult;
            std::string log;
            const std::array materials{scene::RenderMaterial{}};
            const auto coverage = MaterialValueProgramSet::create(materials, log);
            requireRemap(bool(coverage), "Default coverage: " + log);
            std::filesystem::path generatedDirectory;
            requireRemap(coverage->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", generatedDirectory, log),
                "Remap coverage module: " + log);
            const auto generated = generatedDirectory.string();
            const std::array paths{generated.c_str()};
            const std::array capabilities{"spvMeshShadingEXT"};
            const std::array macros{SlangMacroDefine{"METALLIC_CUSTOM_COVERAGE", "0"}};
            const std::array entries{"vsmResetMain", "vsmCountMain"};
            std::array<std::unique_ptr<ShaderModule>, 2> modules;
            std::array<std::unique_ptr<ComputePipeline>, 2> pipelines;
            for (size_t index = 0; index < entries.size(); ++index) {
                auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                    .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders", .additionalSearchPaths = paths,
                    .capabilities = capabilities, .macroDefines = macros}, log);
                requireRemap(shader, "Compile remap entry: " + log);
                auto module = ShaderRegistry::instance().getShaderModule(*device, {.spirv = shader->spirv, .debugName = entries[index]});
                requireRemap(module, "Remap shader module"); modules[index] = std::move(*module);
                auto pipeline = ShaderRegistry::instance().getComputePipeline(*device, {.computeShader = {modules[index].get()},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(RemapPush)});
                requireRemap(pipeline, "Remap pipeline"); pipelines[index] = std::move(*pipeline);
            }
            for (const auto& input : remapCases()) {
                validation.phase(input.name);
                const auto dense = denseReference(input);
                const auto compact = compactInput(input);
                const auto count = uint32_t(dense.size()), levels = uint32_t(input.levels.size());
                const uint32_t blocks = (count + 255) / 256;
                std::vector<Page> previous(count), expected(count), initial(count, {0xcdcdcdcdu, 0xcdcdcdcdu, 0xcdcdcdcdu, 0xcdcdcdcdu});
                std::vector<uint32_t> expectedFlags(kPool + count, 0);
                for (uint32_t page = 0; page < count; ++page) {
                    previous[page] = {page % 11 == 0 ? kInvalid : page % (kPool - 1) + 1, 0xffffffffu, 123,
                        page % 5 == 0 ? 0u : page % 3 == 0 ? 2u : 1u};
                }
                previous.back() = {0, 3, levels, 1};
                uint32_t invalidated = 0, remapped = 0;
                for (uint32_t page = 0; page < count; ++page) {
                    const bool coarse = page + 1 == count;
                    auto& entry = expected[page];
                    entry = {coarse ? 0u : kInvalid, coarse && input.enabled ? 3u : 0u,
                        coarse ? levels : page / (input.axis * input.axis), 2};
                    if (input.reuse && dense[page][0] != kInvalid) {
                        const auto& old = previous[dense[page][0]];
                        if (old[0] != kInvalid) {
                            entry[0] = old[0];
                            entry[3] = dense[page][1] || (old[3] & 1) == 0 || input.force ? 2u : 1u;
                        }
                        if (entry[0] != kInvalid) {
                            expectedFlags[kPool + page] = (dense[page][1] ? 1u : 0u) | (dense[page][0] != page ? 2u : 0u);
                            invalidated += dense[page][1] != 0;
                            remapped += dense[page][0] != page;
                        }
                    }
                }
                const uint32_t mode = (input.reuse ? 1u : 8u) | 2u | (input.force ? 4u : 0u);
                std::array<uint32_t, 32> statistics;
                statistics.fill(0xcdcdcdcdu);
                std::array<uint32_t, 9> arguments;
                arguments.fill(0xcdcdcdcdu);
                std::vector<uint32_t> owners(kPool * 2 + count + blocks, 0xcdcdcdcdu);
                const std::array<uint64_t, 6> bytes{initial.size() * sizeof(Page), previous.size() * sizeof(Page),
                    compact.size() * sizeof(Page), sizeof(statistics), sizeof(arguments), owners.size() * sizeof(uint32_t)};
                const std::array<const void*, 6> data{initial.data(), previous.data(), compact.data(), statistics.data(), arguments.data(), owners.data()};
                const std::array<uint32_t, 6> stride{16,16,16,4,4,4};
                std::array<std::unique_ptr<Buffer>, 6> buffers;
                std::array<ResourceLease, 6> leases;
                for (size_t index = 0; index < buffers.size(); ++index) {
                    auto buffer = device->createBuffer({.size = bytes[index], .structureStride = stride[index],
                        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource, .memoryLocation = MemoryLocation::HostUpload});
                    requireRemap(buffer, "Remap fixture buffer"); buffers[index] = std::move(*buffer);
                    void* mapped = buffers[index]->map(); requireRemap(mapped != nullptr, "Remap upload map");
                    std::memcpy(mapped, data[index], bytes[index]); buffers[index]->flush(); buffers[index]->unmap();
                    auto lease = registry->storageBuffer(*buffers[index]); requireRemap(lease, "Remap descriptor"); leases[index] = *lease;
                }
                auto flagReadback = device->createBuffer({.size = expectedFlags.size() * sizeof(uint32_t),
                    .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback});
                requireRemap(flagReadback, "Remap flag readback");
                RemapPush push;
                push.words[2] = leases[0].shaderIndex(); push.words[5] = leases[3].shaderIndex();
                push.words[12] = leases[4].shaderIndex(); push.words[13] = 111;
                push.words[14] = levels; push.words[15] = input.axis; push.words[18] = kPool;
                push.words[21] = 2; push.words[25] = input.enabled; push.words[26] = count; push.words[27] = input.lightType;
                push.words[28] = leases[1].shaderIndex(); push.words[29] = leases[2].shaderIndex(); push.words[30] = leases[5].shaderIndex();
                push.words[31] = mode; push.words[32] = 1; push.words[33] = 3; push.words[34] = 7; push.words[35] = input.dirtyBounds;
                bench::GPUCommands gpu(*queue);
                requireRemap(gpu.initialize(*device), "Begin remap commands");
                requireRemap(registry->bind(*gpu.commands, leases), "Bind remap resources");
                gpu.commands->hostWriteBarrier();
                requireRemap(gpu.commands->bindExecution(pipelines[0]->execution(), &push, sizeof(push)), "Bind production reset");
                requireRemap(gpu.commands->dispatch((std::max(count, kPool) + 63) / 64), "Dispatch production reset");
                std::array<BufferBarrierDesc, 4> ready{
                    BufferBarrierDesc{.buffer = buffers[0].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}},
                    BufferBarrierDesc{.buffer = buffers[3].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}},
                    BufferBarrierDesc{.buffer = buffers[4].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}},
                    BufferBarrierDesc{.buffer = buffers[5].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
                requireRemap(gpu.commands->synchronize({.buffers = ready}), "Reset output visibility");
                auto flagsSource = buffers[5]->slice({0, expectedFlags.size() * sizeof(uint32_t)});
                auto flagsDestination = (*flagReadback)->slice();
                requireRemap(flagsSource, "Flags source slice"); requireRemap(flagsDestination, "Flags destination slice");
                requireRemap(gpu.commands->copyBuffer(*flagsSource, *flagsDestination), "Capture reset remap flags");
                const BufferBarrierDesc rankReady{.buffer = buffers[5].get(),
                    .before = {PipelineStageBits::ComputeShader | PipelineStageBits::Transfer, AccessBits::ShaderWrite | AccessBits::TransferRead},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
                requireRemap(gpu.commands->synchronize({.buffers = {&rankReady, 1}}), "Reset flags to count rank dependency");
                requireRemap(gpu.commands->bindExecution(pipelines[1]->execution(), &push, sizeof(push)), "Bind production count");
                requireRemap(gpu.commands->dispatch(blocks), "Dispatch production count");
                const BufferBarrierDesc readbackReady[]{
                    {.buffer = buffers[0].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite | AccessBits::ShaderRead},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}},
                    {.buffer = buffers[3].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}},
                    {.buffer = flagReadback->get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}}};
                requireRemap(gpu.commands->synchronize({.buffers = readbackReady}), "Remap host readback visibility");
                requireRemap(gpu.submitAndWait(), "Complete remap commands");
                std::array<uint32_t, 32> expectedStatistics{};
                expectedStatistics[10] = 111; expectedStatistics[11] = levels; expectedStatistics[12] = kPool;
                expectedStatistics[14] = !input.enabled; expectedStatistics[18] = invalidated; expectedStatistics[19] = remapped;
                expectedStatistics[22] = !input.reuse; expectedStatistics[23] = 7; expectedStatistics[25] = input.dirtyBounds;
                expectedStatistics[26] = input.force; expectedStatistics[27] = 1; expectedStatistics[28] = 3;
                const std::array<uint32_t, 9> expectedArguments{0,1,1,0,1,1,0,1,1};
                const std::array<Buffer*, 4> actual{buffers[0].get(), buffers[3].get(), buffers[4].get(), flagReadback->get()};
                const std::array<const void*, 4> expectedData{expected.data(), expectedStatistics.data(), expectedArguments.data(), expectedFlags.data()};
                const std::array<uint64_t, 4> expectedBytes{bytes[0], sizeof(expectedStatistics), sizeof(expectedArguments), expectedFlags.size() * sizeof(uint32_t)};
                for (size_t index = 0; index < actual.size(); ++index) {
                    actual[index]->invalidate(); const void* mapped = actual[index]->map();
                    requireRemap(mapped != nullptr, "Remap result map");
                    const bool equal = std::memcmp(mapped, expectedData[index], expectedBytes[index]) == 0;
                    actual[index]->unmap();
                    requireRemap(equal, input.name + ": compact reset/count differs from independent dense oracle, output " + std::to_string(index));
                }
                report["cases"].push_back({{"name", input.name}, {"virtualPages", count}, {"axis", input.axis},
                    {"levels", levels}, {"lightType", input.lightType}, {"denseBytes", dense.size() * sizeof(Mapping)},
                    {"compactBytes", compact.size() * sizeof(Page)}, {"allPageWordsEqual", true}, {"resetFlagsEqual", true},
                    {"statisticsEqual", true}, {"indirectResetEqual", true}, {"invalidated", invalidated}, {"remapped", remapped}});
            }
            report["validation"] = validation.snapshot();
            requireRemap(!validation.failed(), "Compact remap Vulkan validation error");
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapRemap.json") << report.dump(2) << '\n';
            return RHITestResult::pass("Production compact reset/count pages, remap flags and statistics equal independent dense mapping oracle");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot();
            std::ofstream(directory / "VirtualShadowMapRemap.json") << report.dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapRemapTest);
} // namespace
} // namespace metallic::tests
