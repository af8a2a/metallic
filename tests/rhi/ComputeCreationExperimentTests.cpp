#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ComputeResourceEncoder.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Material/MaterialExecutable.h"
#include "TestResourceParameters.h"
#include <cstring>
#include <set>
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <chrono>
#include <fstream>

namespace metallic::tests {
namespace {

using namespace render;

RHITestResult exerciseCreationBackend(RHITestContext& context, ComputePipelineBackend backend)
{
    auto& registry = ShaderRegistry::instance();
    const auto root = std::filesystem::absolute(context.outputDirectory / "compute-creation" /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    {
        std::ofstream source(root / "ComputeCreationProbe.slang");
        source << "[shader(\"compute\")] [numthreads(1,1,1)] void computeMain(uint3 id : SV_DispatchThreadID) {}\n";
        if (!source) { return RHITestResult::fail("could not write compute creation fixture"); }
    }
    const auto search = root.string();
    const auto binaryDirectory = (root / "native").string();
    std::string log;
    auto code = registry.getShader({.moduleName = "ComputeCreationProbe", .entryPointName = "computeMain",
        .searchPath = search.c_str()}, {.enableDiskCache = false}, log);
    if (!code) { return RHITestResult::fail("could not compile compute creation fixture: " + log); }
    DeviceDesc deviceDesc{.applicationName = "Compute native binary experiment",
        .enableValidation = context.enableValidation,
        .enableBindlessDescriptorHeap = context.device.capabilities().bindlessDescriptorHeap,
        .enablePipelineBinary = backend == ComputePipelineBackend::PipelineBinary};
    auto firstDevice = bench::createTestDevice(context, deviceDesc, true);
    if (!firstDevice) {
        return firstDevice.error() == Error::Unsupported ? RHITestResult::skip("native binary backend unsupported by device")
            : RHITestResult::fail("could not create native binary experiment device");
    }
    auto module = registry.getShaderModule(**firstDevice, {.spirv = code->spirv});
    if (!module) { return RHITestResult::fail("could not create compute experiment module"); }
    ComputePipelineDesc desc{.computeShader = {module->get()}, .backend = backend,
        .nativeBinaryCacheDirectory = binaryDirectory.c_str()};
    PipelineCreationStats firstStats;
    auto first = (*firstDevice)->createComputePipeline(desc, &firstStats);
    if (!first || !firstStats.nativeBinaryPersisted || firstStats.nativeBinaryCacheHit ||
        firstStats.nativeBinaryDataSize == 0 || firstStats.computeBackend != backend ||
        firstStats.nativeObjectReused || firstStats.nativeCacheHit) {
        return RHITestResult::fail("source native creation did not export a distinct native binary cache");
    }
    auto cache = (*firstDevice)->createPipelineCache({.saveOnDestroy = false});
    if (!cache) { return RHITestResult::fail("could not create explicit cache fixture"); }
    auto incompatible = desc;
    incompatible.pipelineCache = cache->get();
    if ((*firstDevice)->createComputePipeline(incompatible)) {
        return RHITestResult::fail("native binary backend silently accepted an unfilled explicit VkPipelineCache");
    }
    auto baseline = desc;
    baseline.backend = ComputePipelineBackend::Pipeline; baseline.nativeBinaryCacheDirectory = nullptr;
    auto optimized = (*firstDevice)->createComputePipeline(baseline);
    baseline.disableOptimization = true;
    auto unoptimized = (*firstDevice)->createComputePipeline(baseline);
    if (!optimized || !unoptimized || (*optimized)->psoHash() == (*unoptimized)->psoHash() ||
        (*optimized)->psoHash() == (*first)->psoHash()) {
        return RHITestResult::fail("creation policy/backend aliases the default native object identity");
    }
    if (backend == ComputePipelineBackend::ShaderObject) {
        incompatible = desc; incompatible.disableOptimization = true;
        if ((*firstDevice)->createComputePipeline(incompatible) || vulkan::nativePipeline(**first).pipeline != VK_NULL_HANDLE) {
            return RHITestResult::fail("shader object accepted a pipeline-only optimization policy or exposed a false VkPipeline");
        }
    }
    // A second logical device imports the exact persisted binary rather than
    // benefiting from a previously retained device-local native object.
    auto secondDevice = bench::createTestDevice(context, deviceDesc, true);
    if (!secondDevice) { return RHITestResult::fail("could not create independent native binary import device"); }
    auto secondModule = registry.getShaderModule(**secondDevice, {.spirv = code->spirv});
    if (!secondModule) { return RHITestResult::fail("could not create import module"); }
    desc.computeShader.module = secondModule->get();
    PipelineCreationStats importStats;
    auto imported = (*secondDevice)->createComputePipeline(desc, &importStats);
    if (!imported || !importStats.nativeBinaryCacheHit || importStats.nativeBinaryDriverRejected ||
        importStats.nativeCacheHit || importStats.nativeObjectReused || importStats.computeBackend != backend ||
        (*imported)->psoHash() != (*first)->psoHash()) {
        return RHITestResult::fail("independent device failed native binary import or reported false application/object hits");
    }
    auto explicitRegistry = registry.getComputePipeline(**secondDevice, desc);
    if (!explicitRegistry || (*explicitRegistry)->creationStats().nativeObjectReused ||
        !(*explicitRegistry)->creationStats().nativeBinaryCacheHit) {
        return RHITestResult::fail("explicit native binary request was intercepted by the registry native object cache");
    }
    desc.nativeBinaryCacheDirectory = nullptr;
    auto retained = registry.getComputePipeline(**secondDevice, desc);
    auto reused = registry.getComputePipeline(**secondDevice, desc);
    if (!retained || !reused || !(*reused)->creationStats().nativeObjectReused ||
        (*reused)->creationStats().computeBackend != backend || (*reused)->creationStats().nativeCreateTimeNanoseconds != 0) {
        return RHITestResult::fail("implicit native object reuse lost its compute backend identity");
    }
    auto* queue = (*secondDevice)->getQueue(QueueType::Graphics);
    if (!queue) { return RHITestResult::fail("could not acquire compute fixture queue"); }
    auto execution = (*imported)->execution();
    imported->reset(); secondModule->reset();
    bench::GPUCommands commands(*queue);
    if (!commands.initialize(**secondDevice) || !commands.commands->bindExecution(execution) ||
        !commands.commands->recordIsolatedCompute([&]() -> Result<> {
            if (auto bound = commands.commands->bindExecution((*reused)->execution()); !bound) { return bound; }
            return commands.commands->dispatch(1, 1, 1);
        }) || !commands.commands->dispatch(1, 1, 1) || !commands.submitAndWait()) {
        return RHITestResult::fail("prepared native binary execution or isolated compute restoration failed");
    }
    return RHITestResult::pass("source/export, independent device binary import, policy hashes, explicit cache bypass, prepared ownership and compute dispatch validated; " + root.string());
}

RHITestResult exerciseScalarSpecialization(RHITestContext& context, ComputePipelineBackend backend)
{
    const auto directory = std::filesystem::absolute(context.outputDirectory / ("scalar-specialization-" +
        std::to_string(static_cast<uint32_t>(backend))) /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto nativeDirectory = (directory / "native").string();
    const auto codeCache = (directory / "spirv").string();
    auto& registry = ShaderRegistry::instance();
    std::string log;
    const SlangShaderDesc source{.moduleName = "ShaderSpecializationProbe", .entryPointName = "specializationProbeMain",
        .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .optimizationLevel = SlangShaderOptimizationLevel::O1,
        .postprocessMode = SlangShaderPostprocessMode::None};
    auto code = registry.getShader(source, {.cacheDirectory = codeCache.c_str()}, log);
    if (!code) { return RHITestResult::fail("specialization runtime compiler/link failed: " + log); }
    // Do not assume Slang's automatically assigned IDs or infer them from names.
    // The two actual declarations must uniquely validate as Int32 and Bool32.
    std::optional<uint32_t> intId, boolId;
    size_t declarationCount = 0;
    for (size_t offset = 5; offset < code->spirv.size();) {
        const uint32_t count = code->spirv[offset] >> 16, op = code->spirv[offset] & 0xffffu;
        if (!count || count > code->spirv.size() - offset) { return RHITestResult::fail("malformed specialization probe SPIR-V"); }
        if (op == 71 && count == 4 && code->spirv[offset + 2] == 1) {
            ++declarationCount;
            ShaderSpecializationConstant declaration{code->spirv[offset + 3], ShaderSpecializationType::Int32, 0};
            std::string reason;
            if (detail::validateShaderSpecializationModule(code->spirv, {&declaration, 1}, reason)) {
                if (intId) { return RHITestResult::fail("probe has multiple Int32 SpecIds"); }
                intId = declaration.id;
            } else {
                declaration.type = ShaderSpecializationType::Bool32;
                if (!detail::validateShaderSpecializationModule(code->spirv, {&declaration, 1}, reason) || boolId) {
                    return RHITestResult::fail("probe SpecId type is unsupported or ambiguous");
                }
                boolId = declaration.id;
            }
        }
        offset += count;
    }
    if (declarationCount != 2 || !intId || !boolId) { return RHITestResult::fail("probe must retain exactly Int32 and Bool32 SpecIds"); }
    auto device = bench::createTestDevice(context, {.applicationName = "scalar32 specialization tuples",
        .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
        .enablePipelineBinary = backend == ComputePipelineBackend::PipelineBinary}, true);
    if (!device) { return device.error() == Error::Unsupported ? RHITestResult::skip("specialization backend unsupported")
        : RHITestResult::fail("specialization device creation failed"); }
    auto module = registry.getShaderModule(**device, {.spirv = code->spirv});
    if (!module) { return RHITestResult::fail("specialization module creation failed"); }
    const ComputeResourceBindingDesc binding{.binding = METALLIC_RESOURCE_MEMBER(BatchBarrierProbeResources, output)};
    const ResourceComputeKernelDesc layout{.bindings = {&binding, 1}, .requiresRayQuery = false,
        .resourceParameterSize = sizeof(BatchBarrierProbeResources)};
    ComputeResourceEncoder encoder;
    if (!encoder.initialize(**device, layout, log)) { return RHITestResult::fail("specialization encoder failed: " + log); }
    auto output = (*device)->createBuffer({.size = 4 * sizeof(uint32_t), .structureStride = sizeof(uint32_t),
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback});
    auto* queue = (*device)->getQueue(QueueType::Graphics);
    if (!output || !queue) { return RHITestResult::fail("specialization output/queue failed"); }
    const std::array<std::array<uint32_t, 2>, 4> tuples{{{0, 1}, {1, 1}, {0xffffffffu, 1}, {0xffffffffu, 0}}};
    std::set<uint64_t> identities;
    std::array<std::shared_ptr<const MaterialExecutableArtifact>, 4> artifacts;
    for (size_t tuple = 0; tuple < tuples.size(); ++tuple) {
        // Caller order does not define identity. Mutating these
        // borrowed inputs after creation must not mutate the native executable.
        std::array values{ShaderSpecializationConstant{*boolId, ShaderSpecializationType::Bool32, tuples[tuple][1]},
            ShaderSpecializationConstant{*intId, ShaderSpecializationType::Int32, tuples[tuple][0]}};
        const auto originalValues = detail::canonicalShaderSpecializationConstants(values);
        ComputePipelineDesc desc{.computeShader = {module->get(), "main", values}, .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(GPUBufferSpan), .backend = backend,
            .nativeBinaryCacheDirectory = backend == ComputePipelineBackend::Pipeline ? nullptr : nativeDirectory.c_str()};
        PipelineCreationStats stats;
        auto pipeline = (*device)->createComputePipeline(desc, &stats);
        if (!pipeline || !identities.insert((*pipeline)->psoHash()).second) {
            return RHITestResult::fail("specialization tuple creation/hash alias");
        }
        if (backend != ComputePipelineBackend::Pipeline && (stats.nativeBinaryCacheHit || !stats.nativeBinaryPersisted)) {
            return RHITestResult::fail("new tuple imported another tuple binary or failed export");
        }
        if (backend == ComputePipelineBackend::Pipeline) {
            auto materialLayout = layout; materialLayout.specializationConstants = values;
            ComputeKernel kernel;
            if (!compileMaterialExecutable(**device, source, materialLayout, kernel, artifacts[tuple], log)) {
                return RHITestResult::fail("specialization material artifact failed: " + log);
            }
            const auto expected = detail::canonicalShaderSpecializationConstants(values);
            if (artifacts[tuple]->specializationConstants != expected) { return RHITestResult::fail("artifact did not own canonical tuple"); }
            ComputeKernel duplicate;
            std::shared_ptr<const MaterialExecutableArtifact> cached;
            materialLayout.specializationConstants = expected;
            if (!compileMaterialExecutable(**device, source, materialLayout, duplicate, cached, log) || cached != artifacts[tuple]) {
                return RHITestResult::fail("same tuple lost material cache reuse");
            }
            for (size_t previous = 0; previous < tuple; ++previous) {
                if (artifacts[previous] == artifacts[tuple] || artifacts[previous]->programKey == artifacts[tuple]->programKey) {
                    return RHITestResult::fail("material executable identity aliased different tuples");
                }
            }
        }
        values[0].bits ^= 1u; values[1].bits ^= 0x12345678u;
        if (artifacts[tuple] && artifacts[tuple]->specializationConstants != originalValues) {
            return RHITestResult::fail("caller mutation changed published material tuple");
        }
        auto* mapped = static_cast<uint32_t*>((*output)->map());
        if (!mapped) { return RHITestResult::fail("specialization poison mapping failed"); }
        std::fill(mapped, mapped + 4, 0xdeadbeefu); (*output)->flush(); (*output)->unmap();
        const ComputeDispatchBinding dispatchBinding{.binding = binding.binding, .buffer = output->get()};
        auto packets = encoder.encode(nullptr, {.bindings = {&dispatchBinding, 1}});
        if (!packets || packets->size() != 1) { return RHITestResult::fail("specialization packet encode failed"); }
        const auto root = (*packets)[0].root();
        bench::GPUCommands commands(*queue); // drains before packet/output destruction
        if (!commands.initialize(**device) || !(*packets)[0].bindResources(*commands.commands) ||
            !commands.commands->bindExecution((*pipeline)->execution(), &root, sizeof(root)) ||
            !commands.commands->dispatch(1, 1, 1) || !commands.submitAndWait()) {
            return RHITestResult::fail("specialization tuple dispatch failed");
        }
        (*output)->invalidate(); mapped = static_cast<uint32_t*>((*output)->map());
        if (!mapped) { return RHITestResult::fail("specialization readback mapping failed"); }
        std::array<uint32_t, 4> actual{}; std::memcpy(actual.data(), mapped, sizeof(actual)); (*output)->unmap();
        const std::array<uint32_t, 4> expected{tuples[tuple][0], tuples[tuple][1],
            tuples[tuple][0] ^ 0x6D3A9F27u, tuples[tuple][1] ^ 0x6D3A9F27u};
        { std::ofstream raw(directory / ("tuple-" + std::to_string(tuple) + ".bin"), std::ios::binary);
            raw.write(reinterpret_cast<const char*>(actual.data()), sizeof(actual));
            if (!raw) { return RHITestResult::fail("specialization raw readback write failed"); } }
        if (actual != expected) { return RHITestResult::fail("specialization tuple uint readback mismatch"); }
        if (backend != ComputePipelineBackend::Pipeline) {
            // Recreate with a fresh, correct caller vector; binary identity must
            // admit this tuple only, not any of the three neighboring tuples.
            values = {ShaderSpecializationConstant{*intId, ShaderSpecializationType::Int32, tuples[tuple][0]},
                ShaderSpecializationConstant{*boolId, ShaderSpecializationType::Bool32, tuples[tuple][1]}};
            PipelineCreationStats importedStats;
            auto imported = (*device)->createComputePipeline(desc, &importedStats);
            if (!imported || !importedStats.nativeBinaryCacheHit || importedStats.nativeBinaryDriverRejected ||
                (*imported)->psoHash() != (*pipeline)->psoHash()) {
                return RHITestResult::fail("same tuple native binary import lost identity");
            }
            mapped = static_cast<uint32_t*>((*output)->map());
            if (!mapped) { return RHITestResult::fail("import poison mapping failed"); }
            std::fill(mapped, mapped + 4, 0xdeadbeefu); (*output)->flush(); (*output)->unmap();
            bench::GPUCommands importedCommands(*queue);
            if (!importedCommands.initialize(**device) || !(*packets)[0].bindResources(*importedCommands.commands) ||
                !importedCommands.commands->bindExecution((*imported)->execution(), &root, sizeof(root)) ||
                !importedCommands.commands->dispatch(1, 1, 1) || !importedCommands.submitAndWait()) {
                return RHITestResult::fail("specialization imported tuple dispatch failed");
            }
            (*output)->invalidate(); mapped = static_cast<uint32_t*>((*output)->map());
            if (!mapped) { return RHITestResult::fail("import readback mapping failed"); }
            std::memcpy(actual.data(), mapped, sizeof(actual)); (*output)->unmap();
            { std::ofstream raw(directory / ("tuple-import-" + std::to_string(tuple) + ".bin"), std::ios::binary);
                raw.write(reinterpret_cast<const char*>(actual.data()), sizeof(actual));
                if (!raw) { return RHITestResult::fail("import raw readback write failed"); } }
            if (actual != expected) { return RHITestResult::fail("imported specialization tuple readback mismatch"); }
        }
    }
    return RHITestResult::pass("four scalar32 tuples, runtime compiler, owned material identity and actual uint readbacks; " + directory.string());
}

class ScalarSpecializationPipelineTest final : public RHITest {
public:
    ScalarSpecializationPipelineTest() { type = RHITestType::Command; name = "specialization_scalar32_pipeline_readback"; }
    RHITestResult run(RHITestContext& context) override { return exerciseScalarSpecialization(context, ComputePipelineBackend::Pipeline); }
};
METALLIC_REGISTER_RHI_TEST(ScalarSpecializationPipelineTest);
class ScalarSpecializationShaderObjectTest final : public RHITest {
public:
    ScalarSpecializationShaderObjectTest() { type = RHITestType::Command; name = "specialization_scalar32_shader_object_readback"; }
    RHITestResult run(RHITestContext& context) override { return exerciseScalarSpecialization(context, ComputePipelineBackend::ShaderObject); }
};
METALLIC_REGISTER_RHI_TEST(ScalarSpecializationShaderObjectTest);
class ScalarSpecializationPipelineBinaryTest final : public RHITest {
public:
    ScalarSpecializationPipelineBinaryTest() { type = RHITestType::Command; name = "specialization_scalar32_pipeline_binary_readback"; }
    RHITestResult run(RHITestContext& context) override { return exerciseScalarSpecialization(context, ComputePipelineBackend::PipelineBinary); }
};
METALLIC_REGISTER_RHI_TEST(ScalarSpecializationPipelineBinaryTest);

class ComputeShaderObjectBinaryTest final : public RHITest {
public:
    ComputeShaderObjectBinaryTest() { type = RHITestType::Command; name = "compute_shader_object_binary_creation_dispatch"; }
    RHITestResult run(RHITestContext& context) override { return exerciseCreationBackend(context, ComputePipelineBackend::ShaderObject); }
};
METALLIC_REGISTER_RHI_TEST(ComputeShaderObjectBinaryTest);

class ComputePipelineBinaryTest final : public RHITest {
public:
    ComputePipelineBinaryTest() { type = RHITestType::Command; name = "compute_pipeline_binary_creation_dispatch"; }
    RHITestResult run(RHITestContext& context) override { return exerciseCreationBackend(context, ComputePipelineBackend::PipelineBinary); }
};
METALLIC_REGISTER_RHI_TEST(ComputePipelineBinaryTest);

} // namespace
} // namespace metallic::tests
