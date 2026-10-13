#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
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
