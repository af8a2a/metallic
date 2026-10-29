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
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Page = std::array<uint32_t, 4>;
using Mapping = std::array<uint32_t, 2>;
using Json = nlohmann::json;
constexpr uint32_t kInvalid = UINT32_MAX;

void requireAllocator(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}
template<typename T>
void requireAllocator(const Result<T>& result, const std::string& message)
{
    requireAllocator(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

// The production inline ABI is 48 words. This probe binds only allocation
// fields; image, material and geometry indices are intentionally unavailable.
struct AllocatorPush
{
    std::array<uint32_t, 48> words{};
};
static_assert(sizeof(AllocatorPush) == 192);

struct AllocationCase
{
    std::vector<Page> pages;
    std::vector<Mapping> reuse;
    std::vector<uint32_t> owners;
    std::array<uint32_t, 32> statistics{};
    uint32_t pool = 1;
    uint32_t cacheMode = 0;
};

AllocationCase makeCase(uint32_t count, uint32_t pool, uint32_t mode)
{
    AllocationCase input;
    input.pool = pool; input.cacheMode = mode == 0 ? 0 : 1;
    input.pages.resize(count); input.reuse.resize(count); input.owners.resize(pool);
    for (uint32_t page = 0; page < count; ++page) {
        input.pages[page] = {kInvalid, 0, page / 16384, 2};
        input.reuse[page] = {kInvalid, 1};
    }
    input.pages.back() = {0, 3, 12, mode == 0 ? 2u : 1u};
    input.reuse.back() = {count - 1, 0};
    if (mode == 5 || mode == 6) {
        // Three levels with four virtual pages each. Both coarse levels must
        // win over the two requested level-zero pages when only four fit.
        for (uint32_t page = 0; page + 1 < count; ++page) { input.pages[page][2] = page / 4; }
        for (uint32_t page : {0u, 1u, 4u, 5u, 8u, 9u}) { input.pages[page][1] = 1; }
        const auto resident = [&](uint32_t page, uint32_t physical) {
            input.pages[page][0] = physical; input.pages[page][3] = 1;
            input.reuse[page] = {page, 0};
        };
        resident(4, 2);
        if (mode == 5) {
            resident(1, 1); resident(8, 3); resident(3, 4);
        } else {
            // Slot 3 is free while slot 1 has an unrequested owner: a global
            // ascending candidate list would incorrectly evict before reuse.
            resident(3, 1); resident(8, 4);
        }
        return input;
    }
    std::vector<uint32_t> order(count - 1);
    std::iota(order.begin(), order.end(), 0u);
    std::mt19937 random(0x70616765u + mode);
    std::shuffle(order.begin(), order.end(), random);
    uint32_t resident = mode == 0 ? 0 : std::min(count - 1, pool - 1);
    if (mode == 2) { resident = resident * 3 / 4; }
    for (uint32_t index = 0; index < resident; ++index) {
        const uint32_t page = order[index];
        input.pages[page][0] = index + 1;
        input.pages[page][3] = index % 7 == 0 ? 2 : 1;
        input.reuse[page] = {(page + 17) % (count - 1), index % 7 == 0 ? 1u : 0u};
    }
    for (uint32_t page = 0; page + 1 < count; ++page) {
        const bool requested = mode == 0 ? page % 3 == 0 : mode == 1 ? input.pages[page][0] != kInvalid :
            mode == 2 ? page < pool : mode == 3;
        input.pages[page][1] = requested ? 1u : 0u;
    }
    return input;
}

// Independent serial policy reference: admit requested pages from coarser
// levels first (reverse virtual order), then assign missing pages in ascending
// virtual order. Admitted residents stay pinned; ascending free physical slots
// precede ascending unrequested-owner eviction.
AllocationCase serialReference(AllocationCase value)
{
    uint32_t requested = 0, allocated = 0, invalidated = 0, remapped = 0;
    for (uint32_t page = 0; page < value.pages.size(); ++page) {
        auto& entry = value.pages[page];
        if (value.cacheMode && entry[0] != kInvalid && value.reuse[page][0] != kInvalid) {
            invalidated += value.reuse[page][1] != 0;
            remapped += value.reuse[page][0] != page;
        }
    }
    for (size_t page = value.pages.size() - 1; page-- > 0;) {
        auto& entry = value.pages[page];
        if (entry[1] & 1) {
            ++requested;
            if (allocated + 1 < value.pool) { ++allocated; entry[1] |= 2; }
            else { entry[0] = kInvalid; entry[3] = 2; }
        }
    }
    for (uint32_t page = 0; page < value.pages.size(); ++page) {
        const auto& entry = value.pages[page];
        if (entry[0] != kInvalid) { value.owners[entry[0]] = page + 1; }
    }
    uint32_t free = 1, evict = 1, added = 0, dirty = 0, clean = 0;
    for (uint32_t page = 0; page + 1 < value.pages.size(); ++page) {
        auto& entry = value.pages[page];
        if ((entry[1] & 2) == 0) { continue; }
        if (entry[0] == kInvalid) {
            while (free < value.pool && value.owners[free] != 0) { ++free; }
            uint32_t physical = free;
            if (physical == value.pool) {
                while (evict < value.pool && (value.pages[value.owners[evict] - 1][1] & 2)) { ++evict; }
                requireAllocator(evict < value.pool, "Reference allocation lacks a candidate");
                physical = evict++;
                auto& old = value.pages[value.owners[physical] - 1];
                old[0] = kInvalid; old[3] = 2;
            } else { ++free; }
            entry[0] = physical; entry[3] = 2;
            value.owners[physical] = page + 1; ++added;
        }
        dirty += (entry[3] & 2) != 0;
        clean += (entry[3] & 3) == 1;
    }
    value.statistics[0] = requested; value.statistics[1] = allocated; value.statistics[2] = requested - allocated;
    value.statistics[16] = dirty; value.statistics[17] = clean;
    value.statistics[18] = invalidated; value.statistics[19] = remapped;
    value.statistics[20] = added; value.statistics[21] = (value.pages.back()[3] & 2) != 0;
    return value;
}

class VirtualShadowMapAllocatorTest final : public RHITest {
public:
    VirtualShadowMapAllocatorTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_allocator"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapAllocator";
        Json report{{"passed", false}, {"cases", Json::array()}};
        bench::ValidationRecorder validation;
        try {
            std::filesystem::create_directories(directory);
            auto created = createDevice({.applicationName = "VSM deterministic parallel allocator",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless mesh-capable Vulkan"); }
            requireAllocator(created, "Create allocation device");
            auto device = std::move(*created);
            auto queue = device->getQueue(QueueType::Graphics);
            requireAllocator(queue != nullptr, "Allocation graphics queue");
            auto registryResult = ResourceRegistry::forDevice(*device);
            requireAllocator(registryResult, "Allocation resource registry");
            auto registry = *registryResult;
            std::string log;
            const std::array materials{scene::RenderMaterial{}};
            const auto coverage = MaterialValueProgramSet::create(materials, log);
            requireAllocator(bool(coverage), "Allocation default coverage: " + log);
            std::filesystem::path generatedDirectory;
            requireAllocator(coverage->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", generatedDirectory, log),
                "Allocation coverage module: " + log);
            const auto generated = generatedDirectory.string();
            const std::array paths{generated.c_str()};
            const std::array capabilities{"spvMeshShadingEXT"};
            const std::array macros{SlangMacroDefine{"METALLIC_CUSTOM_COVERAGE", "0"}};
            const std::array entries{"vsmCountMain", "vsmAllocateMain", "vsmAdmitMain", "vsmPhysicalMain", "vsmAssignMain"};
            std::array<std::unique_ptr<ShaderModule>, 5> modules;
            std::array<std::unique_ptr<ComputePipeline>, 5> pipelines;
            for (size_t index = 0; index < entries.size(); ++index) {
                auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                    .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders", .additionalSearchPaths = paths,
                    .capabilities = capabilities, .macroDefines = macros}, log);
                requireAllocator(shader, "Compile allocator entry: " + log);
                auto module = ShaderRegistry::instance().getShaderModule(*device, {.spirv = shader->spirv, .debugName = entries[index]});
                requireAllocator(module, "Allocation shader module"); modules[index] = std::move(*module);
                auto pipeline = ShaderRegistry::instance().getComputePipeline(*device, {.computeShader = {modules[index].get()},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(AllocatorPush)});
                requireAllocator(pipeline, "Allocation pipeline"); pipelines[index] = std::move(*pipeline);
            }
            const std::array cases{
                std::array{1u,1u,0u}, std::array{2u,1u,0u}, std::array{255u,17u,1u},
                std::array{256u,17u,2u}, std::array{257u,17u,3u}, std::array{1025u,256u,1u},
                std::array{1025u,256u,2u}, std::array{1025u,256u,3u}, std::array{65537u,256u,1u},
                std::array{65537u,4096u,2u}, std::array{196609u,4096u,0u}, std::array{196609u,4096u,1u},
                std::array{196609u,4096u,2u}, std::array{196609u,4096u,3u}, std::array{196609u,4096u,4u},
                std::array{196609u,1u,3u}, std::array{13u,5u,5u}, std::array{13u,5u,6u}};
            for (const auto& config : cases) {
                const auto input = makeCase(config[0], config[1], config[2]);
                const auto expected = serialReference(input);
                if (config[2] == 5 || config[2] == 6) {
                    requireAllocator(expected.statistics[0] == 6 && expected.statistics[1] == 4 &&
                        expected.statistics[2] == 2, "Directed budget pressure was not exercised");
                    for (uint32_t page : {0u, 1u}) {
                        requireAllocator((expected.pages[page][1] & 2) == 0 && expected.pages[page][0] == kInvalid,
                            "Fine requested pages displaced the coarser receiver coverage");
                    }
                    for (uint32_t page : {4u, 5u, 8u, 9u}) {
                        requireAllocator((expected.pages[page][1] & 2) != 0,
                            "Coarse-first admission did not preserve both fallback levels");
                    }
                    requireAllocator(expected.pages[4][0] == 2 && expected.pages[8][0] == input.pages[8][0],
                        "Admitted resident physical slots were not pinned");
                    const uint32_t freeSlot = config[2] == 5 ? 1 : 3;
                    const uint32_t evictionSlot = config[2] == 5 ? 4 : 1;
                    requireAllocator(expected.pages[5][0] == freeSlot && expected.pages[9][0] == evictionSlot &&
                        expected.pages[3][0] == kInvalid, "Missing pages did not consume free slots before eviction");
                }
                const uint32_t blocks = (config[0] + 255u) / 256u;
                for (uint32_t repetition = 0; repetition < 2; ++repetition) {
                    validation.phase(std::to_string(config[0]) + "-" + std::to_string(config[1]) + "-" + std::to_string(config[2]));
                    auto owners = input.owners;
                    owners.resize(uint64_t(config[1]) * 2 + config[0] + blocks, 0xcdcdcdcdu);
                    // Reset now leaves compact-remap diagnostic flags in the
                    // rank slots. Seed them from this fixture's independent
                    // dense mappings before testing the allocator in isolation.
                    for (uint32_t page = 0; page < config[0]; ++page) {
                        const auto mapping = input.reuse[page];
                        owners[config[1] + page] = mapping[0] == kInvalid ? 0u :
                            (mapping[1] != 0 ? 1u : 0u) | (mapping[0] != page ? 2u : 0u);
                    }
                    const std::array<uint64_t, 3> bytes{input.pages.size() * sizeof(Page),
                        sizeof(input.statistics), owners.size() * sizeof(uint32_t)};
                    const std::array<const void*, 3> data{input.pages.data(), input.statistics.data(), owners.data()};
                    const std::array<uint32_t, 3> stride{16,4,4};
                    std::array<std::unique_ptr<Buffer>, 3> buffers;
                    std::array<ResourceLease, 3> leases;
                    for (size_t index = 0; index < buffers.size(); ++index) {
                        auto buffer = device->createBuffer({.size = bytes[index], .structureStride = stride[index],
                            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
                        requireAllocator(buffer, "Allocation fixture buffer"); buffers[index] = std::move(*buffer);
                        void* mapped = buffers[index]->map(); requireAllocator(mapped != nullptr, "Allocation upload map");
                        std::memcpy(mapped, data[index], bytes[index]); buffers[index]->flush(); buffers[index]->unmap();
                        auto lease = registry->storageBuffer(*buffers[index]); requireAllocator(lease, "Allocation descriptor");
                        leases[index] = *lease;
                    }
                    AllocatorPush push;
                    push.words[2] = leases[0].shaderIndex(); // pages
                    push.words[5] = leases[1].shaderIndex(); // statistics
                    push.words[18] = config[1]; // physicalPageCount
                    push.words[25] = 1; // enabled
                    push.words[26] = config[0]; // totalPages
                    push.words[30] = leases[2].shaderIndex(); // owners and scratch
                    push.words[31] = input.cacheMode;
                    bench::GPUCommands gpu(*queue);
                    requireAllocator(gpu.initialize(*device), "Begin allocator commands");
                    requireAllocator(registry->bind(*gpu.commands, leases), "Bind allocator resources");
                    std::array<BufferBarrierDesc, 3> barriers;
                    for (size_t index = 0; index < barriers.size(); ++index) {
                        barriers[index] = {.buffer = buffers[index].get(),
                            .before = {PipelineStageBits::Host, AccessBits::HostWrite},
                            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
                    }
                    requireAllocator(gpu.commands->synchronize({.buffers = barriers}), "Allocator upload visibility");
                    for (size_t index = 0; index < pipelines.size(); ++index) {
                        requireAllocator(gpu.commands->bindExecution(pipelines[index]->execution(), &push, sizeof(push)), "Bind allocator stage");
                        requireAllocator(gpu.commands->dispatch(index == 1 || index == 3 ? 1 : blocks), "Dispatch allocator stage");
                        for (auto& barrier : barriers) {
                            barrier.before = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite};
                            barrier.after = index + 1 == pipelines.size() ? SyncScope{PipelineStageBits::Host, AccessBits::HostRead} :
                                SyncScope{PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite};
                        }
                        requireAllocator(gpu.commands->synchronize({.buffers = barriers}), "Allocator stage dependency");
                    }
                    requireAllocator(gpu.submitAndWait(), "Complete allocator commands");
                    const std::array<const void*, 3> expectedData{expected.pages.data(), expected.statistics.data(), expected.owners.data()};
                    const std::array<size_t, 3> outputIndices{0,1,2};
                    const std::array<uint64_t, 3> outputBytes{bytes[0],bytes[1],uint64_t(config[1])*4};
                    for (size_t index = 0; index < outputIndices.size(); ++index) {
                        auto& buffer = *buffers[outputIndices[index]]; buffer.invalidate();
                        const void* mapped = buffer.map(); requireAllocator(mapped != nullptr, "Allocator result map");
                        const bool equal = std::memcmp(mapped, expectedData[index], outputBytes[index]) == 0;
                        buffer.unmap(); requireAllocator(equal, "GPU allocator differs from serial reference: output " +
                            std::to_string(index) + ", pages " + std::to_string(config[0]) + ", pool " + std::to_string(config[1]) +
                            ", mode " + std::to_string(config[2]));
                    }
                }
                report["cases"].push_back({{"virtualPages",config[0]}, {"physicalPages",config[1]}, {"mode",config[2]},
                    {"repetitions",2}, {"allPageWordsEqual",true}, {"allOwnersEqual",true}, {"statistics",expected.statistics}});
            }
            report["validation"] = validation.snapshot();
            requireAllocator(!validation.failed(), "Allocator Vulkan validation error");
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapAllocator.json") << report.dump(2) << '\n';
            return RHITestResult::pass("36 deterministic parallel allocations match coarse-first admission, resident pinning and free-before-evict reference");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot();
            std::ofstream(directory / "VirtualShadowMapAllocator.json") << report.dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapAllocatorTest);

} // namespace
} // namespace metallic::tests
