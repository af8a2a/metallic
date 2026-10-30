#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

void requireCompact(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}
template<typename T>
void requireCompact(const Result<T>& result, const std::string& message)
{
    requireCompact(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

// Production inline/resource ABIs. This probe invokes the production entrypoints,
// rather than copying the append, batch split or indirect consumer into a probe shader.
struct CompactProbePush { std::array<uint32_t, 52> words{}; };
struct CompactProbeProjection
{
    std::array<float, 4> rowX{1,0,0,0}, rowY{0,1,0,0}, rowZ{0,0,1,0}, rowW{0,0,0,1}, metric{};
    std::array<uint32_t, 4> page{};
    std::array<float, 4> originNear{};
};
struct CompactProbeTriangle
{
    std::array<std::array<float, 8>, 3> vertices{};
    std::array<uint32_t, 4> material{};
};
static_assert(sizeof(CompactProbePush) == 208 && sizeof(CompactProbeProjection) == 112 && sizeof(CompactProbeTriangle) == 112);

class VirtualShadowMapCompactBuildArgumentsTest final : public RHITest
{
public:
    VirtualShadowMapCompactBuildArgumentsTest()
    {
        type = RHITestType::Rendering;
        name = "virtual_shadow_map_compact_build_arguments";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapCompactBuildArguments";
        Json report{{"passed",false},{"cases",Json::array()}};
        bench::ValidationRecorder validation;
        try {
            std::filesystem::create_directories(directory);
            auto created = createDevice({.applicationName = "VSM compact build argument probe",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless mesh-capable Vulkan"); }
            requireCompact(created, "Create compact probe device");
            auto device = std::move(*created);
            auto* queue = device->getQueue(QueueType::Graphics);
            requireCompact(queue != nullptr, "Compact probe graphics queue");
            auto registered = ResourceRegistry::forDevice(*device);
            requireCompact(registered, "Compact probe resource registry");
            auto registry = *registered;
            std::string log;
            const std::array materials{scene::RenderMaterial{}};
            auto coverage = MaterialValueProgramSet::create(materials, log);
            requireCompact(bool(coverage), "Default coverage: " + log);
            std::filesystem::path generatedDirectory;
            requireCompact(coverage->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", generatedDirectory, log), "Coverage module: " + log);
            const auto generated = generatedDirectory.string();
            const std::array paths{generated.c_str()};
            const std::array capabilities{"spvMeshShadingEXT"};
            const std::array entries{"vsmResetMain", "vsmCompactMain", "vsmBuildMain"};
            for (bool bitset : {false, true}) {
                const std::array macros{SlangMacroDefine{"METALLIC_CUSTOM_COVERAGE","0"},
                    SlangMacroDefine{"VSM_WORKLOAD_DIAGNOSTICS","1"}, SlangMacroDefine{"VSM_CLUSTER_CULL","1"},
                    SlangMacroDefine{"VSM_CLUSTER_PAGE_BITSET",bitset ? "1" : "0"},
                    SlangMacroDefine{"VSM_TRIANGLE_PAGE_BITSET",bitset ? "1" : "0"},
                    SlangMacroDefine{"VSM_DIRECTIONAL_BUILD","1"}, SlangMacroDefine{"VSM_COMPACT_BUILD","1"}};
                std::array<std::unique_ptr<ShaderModule>, 3> modules;
                std::array<std::unique_ptr<ComputePipeline>, 3> pipelines;
                for (size_t index = 0; index < entries.size(); ++index) {
                    auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                        .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders", .additionalSearchPaths = paths,
                        .capabilities = capabilities, .macroDefines = macros}, log);
                    requireCompact(shader, "Compile compact probe: " + log);
                    auto module = ShaderRegistry::instance().getShaderModule(*device, {.spirv = shader->spirv, .debugName = entries[index]});
                    requireCompact(module, "Compact probe shader module"); modules[index] = std::move(*module);
                    auto pipeline = ShaderRegistry::instance().getComputePipeline(*device, {.computeShader = {modules[index].get()},
                        .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(CompactProbePush)});
                    requireCompact(pipeline, "Compact probe pipeline"); pipelines[index] = std::move(*pipeline);
                }
                for (uint32_t mode = 0; mode < 5; ++mode) {
                    const bool sparse = mode == 1, outside = mode == 2, warm = mode == 3, boundary = mode == 4;
                    const std::string label = std::string(bitset ? "Bitset-" : "Pages-") +
                        std::array{"AllTail", "Sparse", "NoCandidates", "Warm", "Batch65535"}[mode];
                    validation.phase(label);
                    constexpr uint32_t totalPages = 9, physicalPages = 2, sentinel = 0xcdcdcdcdu;
                    const uint32_t groups = boundary ? 2 : 8;
                    const uint32_t triangleCount = boundary ? 65536u * 64u : groups * 64u - (mode == 0 ? 7u : 0u);
                    const uint32_t capacity = (triangleCount + 63u) / 64u;
                    const uint32_t batches = (capacity + 65534u) / 65535u;
                    constexpr uint32_t rasterBatches = 2, argumentOffset = rasterBatches * 3;
                    const uint32_t bitsetOffset = physicalPages * 2 + totalPages + 1;
                    const uint32_t candidateOffset = bitsetOffset + (bitset ? 1u : 0u);
                    std::vector<std::array<float, 8>> bounds(groups);
                    for (uint32_t group = 0; group < groups; ++group) {
                        const float center = outside || (sparse && (group & 1u)) ? 10.f : 0.f;
                        bounds[group] = {center-.3f,center-.3f,.4f,0,center+.3f,center+.3f,.6f,0};
                    }
                    std::vector<CompactProbeTriangle> triangles(boundary ? groups * 64u : triangleCount);
                    for (auto& triangle : triangles) {
                        triangle.vertices = {{{-.25f,-.25f,.5f,1,0,0,0,0},{.25f,-.25f,.5f,1,0,0,0,0},{0,.25f,.5f,1,0,0,0,0}}};
                    }
                    std::array<uint32_t, 16> geometry{};
                    geometry[7] = triangleCount; geometry[9] = UINT32_MAX;
                    std::array<std::array<uint32_t, 4>, totalPages> pages{};
                    std::vector<uint32_t> statistics((128 + kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize) / 4, sentinel);
                    std::array<CompactProbeProjection, 2> projections;
                    projections[0].page = {0,2,0,0}; projections[1].page = {4,2,1,0};
                    std::vector<uint32_t> arguments(argumentOffset + batches * 3 + 4, sentinel);
                    std::vector<uint32_t> owners(candidateOffset + 1 + capacity * 2 + 4, sentinel);
                    std::vector<std::array<uint32_t, 2>> work(4096, {sentinel,sentinel});
                    const std::array<uint64_t, 9> sizes{bounds.size()*32,triangles.size()*112,sizeof(geometry),sizeof(pages),
                        statistics.size()*4,sizeof(projections),arguments.size()*4,owners.size()*4,work.size()*8};
                    const std::array<uint32_t, 9> strides{32,112,64,16,4,112,4,4,8};
                    const std::array<const void*, 9> data{bounds.data(),triangles.data(),geometry.data(),pages.data(),statistics.data(),
                        projections.data(),arguments.data(),owners.data(),work.data()};
                    std::array<std::unique_ptr<Buffer>, 9> buffers;
                    std::array<ResourceLease, 9> leases;
                    const auto upload = [&](uint32_t index, const void* value, size_t bytes) {
                        void* mapped = buffers[index]->map(); requireCompact(mapped != nullptr, "Map compact probe input");
                        std::memcpy(mapped,value,bytes); buffers[index]->flush(); buffers[index]->unmap();
                    };
                    const auto read = [&](uint32_t index, void* value, size_t bytes) {
                        buffers[index]->invalidate(); const void* mapped = buffers[index]->map();
                        requireCompact(mapped != nullptr, "Map compact probe output");
                        std::memcpy(value,mapped,bytes); buffers[index]->unmap();
                    };
                    for (uint32_t index = 0; index < buffers.size(); ++index) {
                        auto buffer = device->createBuffer({.size = sizes[index], .structureStride = strides[index],
                            .usage = BufferUsageBits::Storage | (index == 6 ? BufferUsageBits::Indirect : BufferUsageBits::None),
                            .memoryLocation = MemoryLocation::HostUpload});
                        requireCompact(buffer, "Compact probe buffer"); buffers[index] = std::move(*buffer);
                        upload(index,data[index],sizes[index]);
                        auto lease = registry->storageBuffer(*buffers[index]); requireCompact(lease, "Compact probe descriptor"); leases[index] = *lease;
                    }
                    geometry[1] = leases[1].shaderIndex(); geometry[12] = leases[0].shaderIndex();
                    upload(2,geometry.data(),sizeof(geometry));
                    CompactProbePush push;
                    push.words[0] = leases[2].shaderIndex(); push.words[2] = leases[3].shaderIndex();
                    push.words[4] = leases[8].shaderIndex(); push.words[5] = leases[4].shaderIndex();
                    push.words[6] = leases[5].shaderIndex(); push.words[12] = leases[6].shaderIndex();
                    push.words[13] = triangleCount; push.words[14] = 2; push.words[15] = 2;
                    push.words[18] = physicalPages; push.words[20] = uint32_t(work.size()); push.words[21] = 1;
                    push.words[25] = 1; push.words[26] = totalPages; push.words[30] = leases[7].shaderIndex(); push.words[32] = 1;
                    const auto synchronize = [&](CommandBuffer& commands, SyncScope before, SyncScope after) {
                        std::array<BufferBarrierDesc, 9> barriers;
                        for (size_t index = 0; index < barriers.size(); ++index) {
                            barriers[index] = {.buffer=buffers[index].get(), .before=before, .after=after};
                        }
                        requireCompact(commands.synchronize({.buffers=barriers}), "Compact probe buffer dependency");
                    };
                    const auto reset = [&] {
                        bench::GPUCommands gpu(*queue); requireCompact(gpu.initialize(*device), "Begin compact reset");
                        requireCompact(registry->bind(*gpu.commands,leases), "Bind compact reset resources");
                        synchronize(*gpu.commands,{PipelineStageBits::AllCommands | PipelineStageBits::Host,AccessBits::MemoryRead | AccessBits::MemoryWrite | AccessBits::HostWrite},
                            {PipelineStageBits::ComputeShader,AccessBits::ShaderRead | AccessBits::ShaderWrite});
                        requireCompact(gpu.commands->bindExecution(pipelines[0]->execution(),&push,sizeof(push)), "Bind production reset");
                        requireCompact(gpu.commands->dispatch(2), "Dispatch production reset");
                        synchronize(*gpu.commands,{PipelineStageBits::ComputeShader,AccessBits::ShaderWrite},{PipelineStageBits::Host,AccessBits::HostRead});
                        requireCompact(gpu.submitAndWait(), "Complete production reset");
                        read(6,arguments.data(),sizes[6]); read(7,owners.data(),sizes[7]); read(4,statistics.data(),sizes[4]);
                        requireCompact(owners[candidateOffset] == 0, label + ": reset retained candidate count");
                        for (uint32_t batch = 0; batch < rasterBatches + batches; ++batch) {
                            requireCompact(arguments[batch*3] == 0 && arguments[batch*3+1] == 1 && arguments[batch*3+2] == 1,
                                label + ": reset retained or malformed indirect arguments");
                        }
                    };
                    reset();
                    const uint32_t seed = boundary ? 65534u : 0u;
                    owners[candidateOffset] = seed;
                    if (bitset && !warm) { owners[bitsetOffset] = 0xffu; }
                    arguments[argumentOffset] = seed;
                    statistics[16] = warm ? 0u : 8u;
                    for (uint32_t page = 0; page < totalPages; ++page) { pages[page] = {0,2,page/4,2}; }
                    upload(3,pages.data(),sizeof(pages)); upload(4,statistics.data(),sizes[4]);
                    upload(6,arguments.data(),sizes[6]); upload(7,owners.data(),sizes[7]);
                    {
                        bench::GPUCommands gpu(*queue); requireCompact(gpu.initialize(*device), "Begin compact/build probe");
                        requireCompact(registry->bind(*gpu.commands,leases), "Bind compact/build resources");
                        synchronize(*gpu.commands,{PipelineStageBits::Host,AccessBits::HostWrite},
                            {PipelineStageBits::ComputeShader,AccessBits::ShaderRead | AccessBits::ShaderWrite});
                        requireCompact(gpu.commands->bindExecution(pipelines[1]->execution(),&push,sizeof(push)), "Bind production compact");
                        requireCompact(gpu.commands->dispatch(groups), "Dispatch production compact");
                        if (!boundary) {
                            synchronize(*gpu.commands,{PipelineStageBits::ComputeShader,AccessBits::ShaderWrite},
                                {PipelineStageBits::ComputeShader | PipelineStageBits::DrawIndirect,AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::IndirectRead});
                            requireCompact(gpu.commands->bindExecution(pipelines[2]->execution(),&push,sizeof(push)), "Bind production indirect build");
                            auto slice = buffers[6]->slice({argumentOffset*4u,12}); requireCompact(slice, "Indirect argument slice");
                            requireCompact(gpu.commands->dispatchIndirect(*slice), "Execute production indirect build");
                        }
                        synchronize(*gpu.commands,{PipelineStageBits::AllCommands,AccessBits::MemoryRead | AccessBits::MemoryWrite},
                            {PipelineStageBits::Host,AccessBits::HostRead});
                        requireCompact(gpu.submitAndWait(), "Complete compact/build probe");
                    }
                    read(6,arguments.data(),sizes[6]); read(7,owners.data(),sizes[7]); read(4,statistics.data(),sizes[4]);
                    const uint32_t survivors = warm || outside ? 0u : sparse ? groups/2u : groups;
                    const uint32_t total = seed + survivors;
                    requireCompact(owners[candidateOffset] == total, label + ": candidate count mismatch");
                    uint32_t argumentTotal = 0;
                    for (uint32_t batch = 0; batch < batches; ++batch) {
                        const uint32_t expected = std::min(total - std::min(total,batch*65535u),65535u);
                        const auto offset = argumentOffset + batch*3u;
                        requireCompact(arguments[offset] == expected && arguments[offset+1] == 1 && arguments[offset+2] == 1,
                            label + ": indirect groups differ from compact candidate batches");
                        argumentTotal += arguments[offset];
                    }
                    requireCompact(argumentTotal == total, label + ": indirect groups did not partition candidates");
                    std::vector<uint32_t> emitted;
                    for (uint32_t index = seed; index < total; ++index) {
                        emitted.push_back(owners[candidateOffset+1+index*2]);
                        requireCompact(owners[candidateOffset+2+index*2] == 3, label + ": candidate lost exact level mask");
                    }
                    std::ranges::sort(emitted);
                    std::vector<uint32_t> expected;
                    for (uint32_t group = 0; group < groups; ++group) {
                        if (!warm && !outside && (!sparse || (group & 1u) == 0)) { expected.push_back(group*64u); }
                    }
                    requireCompact(emitted == expected, label + ": candidate list duplicated or lost a cluster");
                    const auto counter = [&](uint32_t index) { return uint64_t(statistics[index*2]) | (uint64_t(statistics[index*2+1]) << 32u); };
                    requireCompact(counter(32+kVSMBuildDiagnosticClusterGroups) == (warm ? 0u : groups) &&
                        counter(32+kVSMBuildDiagnosticClusterSurvivingGroups) == survivors, label + ": compact diagnostics mismatch");
                    if (!boundary) {
                        const uint32_t active = mode == 0 ? triangleCount : survivors*64u;
                        requireCompact(counter(16+kVSMWorkBuildActiveTriangleSlots) == active &&
                            counter(16+kVSMWorkBuildDecodedTriangles) == active && statistics[3] == active*8u,
                            label + ": indirect build did not consume exactly the compact candidates and tail lanes");
                    }
                    requireCompact(std::ranges::all_of(std::span(owners).last(4), [](uint32_t value) { return value == sentinel; }) &&
                        std::ranges::all_of(std::span(arguments).last(4), [](uint32_t value) { return value == sentinel; }),
                        label + ": candidate or argument allocation overrun");
                    report["cases"].push_back({{"name",label},{"seededPrefix",seed},{"newCandidates",survivors},{"totalCandidates",total},
                        {"indirectGroups",argumentTotal},{"emittedFirstTriangles",emitted},{"indirectBuildExecuted",!boundary}});
                    // Reset the same buffers after nonzero dispatches; a new frame must
                    // never inherit candidate counts or either indirect batch's X count.
                    reset();
                }
            }
            report["validation"] = validation.snapshot();
            requireCompact(!validation.failed(), "Compact argument probe Vulkan validation error");
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapCompactBuildArguments.json") << report.dump(2) << '\n';
            return RHITestResult::pass("Production reset/compact/indirect build preserve exact candidate lists, tails, empty work and the 65535 batch frontier");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot();
            std::ofstream(directory / "VirtualShadowMapCompactBuildArguments.json") << report.dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapCompactBuildArgumentsTest);

} // namespace
} // namespace metallic::tests
