#include "RHITest.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>

namespace metallic::tests {
namespace {

class ShaderRegistryTest final : public RHITest {
public:
    ShaderRegistryTest() { type = RHITestType::Resource; name = "shader_registry_source_pso_and_device_lifetime"; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        auto& registry = ShaderRegistry::instance();
        const auto root = std::filesystem::absolute(context.outputDirectory / "shader-registry" /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        {
            std::ofstream file(root / "RegistrySource.slang");
            file << "#ifndef REGISTRY_THREADS\n#define REGISTRY_THREADS 1\n#endif\n"
                    "[shader(\"compute\")] [numthreads(REGISTRY_THREADS, 1, 1)]\n"
                    "void registryMain(uint3 id : SV_DispatchThreadID) {}\n";
            if (!file) { return RHITestResult::fail("could not write registry source fixture"); }
        }
        const std::string sourceRoot = root.string();
        const std::string cacheRoot = (root / "spirv").string();
        const SlangShaderDesc source{.moduleName = "RegistrySource", .entryPointName = "registryMain",
            .searchPath = sourceRoot.c_str()};
        bool cacheHit = false;
        const SlangShaderCacheOptions options{.cacheDirectory = cacheRoot.c_str(), .outCacheHit = &cacheHit};
        std::string log;
        auto first = registry.getShader(source, options, log);
        if (!first || cacheHit) { return RHITestResult::fail("missing source cache did not compile: " + log); }
        auto cached = registry.getShader(source, options, log);
        if (!cached || !cacheHit || cached->spirv != first->spirv) {
            return RHITestResult::fail("precompiled source cache was not reused: " + log);
        }
        auto shader = registry.getShaderModule(context.device, {.spirv = first->spirv});
        if (!shader) { return RHITestResult::fail("registry shader module creation failed"); }
        const ComputePipelineDesc pipelineDesc{.computeShader = {shader->get()}};
        auto pipeline = registry.getComputePipeline(context.device, pipelineDesc);
        auto reused = registry.getComputePipeline(context.device, pipelineDesc);
        if (!pipeline || !reused || !reused->get()->pipelineCacheHit() ||
            pipeline->get()->psoHash() != reused->get()->psoHash()) {
            return RHITestResult::fail("default registry PSO cache was omitted or not reused");
        }
        const uint64_t originalHash = pipeline->get()->psoHash();
        const SlangMacroDefine threads{"REGISTRY_THREADS", "2"};
        auto changedSource = source;
        changedSource.macroDefines = {&threads, 1};
        auto changed = registry.getShader(changedSource, options, log);
        if (!changed || changed->spirv == first->spirv) { return RHITestResult::fail("registry aliased shader variants"); }
        auto changedShader = registry.getShaderModule(context.device, {.spirv = changed->spirv});
        if (!changedShader) { return RHITestResult::fail("changed shader creation failed"); }
        auto changedPipeline = registry.getComputePipeline(context.device, {.computeShader = {changedShader->get()}});
        if (!changedPipeline || changedPipeline->get()->psoHash() == originalHash) {
            return RHITestResult::fail("changed shader reused the old PSO identity");
        }
        if (!context.device.capabilities().bindlessDescriptorHeap) {
            return RHITestResult::skip("requires --rhi-bindless for typed kernel/state tests");
        }
        ComputeKernel kernel;
        constexpr auto abi = parameterAbi<uint32_t>(0x5352454749535452ull, ParameterTransport::InlinePush);
        if (!registry.getComputeKernel(context.device, source, {.parameters = abi}, kernel, log) || !kernel.valid()) {
            return RHITestResult::fail("source-to-kernel acquisition failed: " + log);
        }
        auto badSource = source;
        badSource.moduleName = "MissingRegistryModule";
        if (registry.getComputeKernel(context.device, badSource, {.parameters = abi}, kernel, log) || !kernel.valid()) {
            return RHITestResult::fail("failed source acquisition replaced the published kernel");
        }
        if (registry.getComputeKernel(context.device, source, {}, kernel, log) || !kernel.valid()) {
            return RHITestResult::fail("failed pipeline creation replaced the published kernel");
        }
        ComputeProgram program;
        const ComputeProgramBindingDesc binding{.binding = 0, .kind = ComputeResourceBindingKind::StorageBuffer};
        const ComputeResourceField field{.binding = 0, .kind = ComputeResourceBindingKind::StorageBuffer};
        const ComputeProgramDesc programLayout{.bindings = {&binding, 1}, .requiresRayQuery = false,
            .resourceParameters = {.size = 4, .fields = {&field, 1}}};
        if (!registry.getComputeProgram(context.device, source, programLayout, program, log) || !program.valid() ||
            registry.getComputeProgram(context.device, badSource, programLayout, program, log) || !program.valid() ||
            registry.getComputeProgram(context.device, source, {}, program, log) || !program.valid()) {
            return RHITestResult::fail("source-to-program acquisition or failed generation publication violated its contract: " + log);
        }
        const ComputePipelineDesc typedDesc{.computeShader = {shader->get()},
            .usesBindlessHeap = true, .bindlessUserPushDataSize = 4};
        auto typed = registry.getComputePipeline(context.device, typedDesc);
        auto otherLayout = typedDesc;
        otherLayout.bindlessUserPushDataSize = 8;
        auto different = registry.getComputePipeline(context.device, otherLayout);
        if (!typed || !different || typed->get()->psoHash() == different->get()->psoHash()) {
            return RHITestResult::fail("registry aliased pipeline layouts");
        }
        std::vector<std::future<Result<std::unique_ptr<ComputePipeline>>>> jobs;
        for (uint32_t i = 0; i < 4; ++i) {
            jobs.push_back(std::async(std::launch::async, [&] { return registry.getComputePipeline(context.device, pipelineDesc); }));
        }
        for (auto& job : jobs) {
            auto value = job.get();
            if (!value || !value->get()->pipelineCacheHit() || value->get()->psoHash() != originalHash) {
                return RHITestResult::fail("parallel registry request failed or aliased code");
            }
        }
        const auto triangleSource = [](const char* entry) {
            return SlangShaderDesc{.moduleName = "Features/Samples/Triangle", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"};
        };
        auto vertexCode = registry.getShader(triangleSource("triangleVertexMain"), log);
        auto fragmentCode = registry.getShader(triangleSource("triangleFragmentMain"), log);
        auto greenCode = registry.getShader(triangleSource("solidGreenFragmentMain"), log);
        if (!vertexCode || !fragmentCode || !greenCode) { return RHITestResult::fail("graphics source acquisition failed: " + log); }
        auto vertex = registry.getShaderModule(context.device, {.spirv = vertexCode->spirv});
        auto fragment = registry.getShaderModule(context.device, {.spirv = fragmentCode->spirv});
        auto green = registry.getShaderModule(context.device, {.spirv = greenCode->spirv});
        if (!vertex || !fragment || !green) { return RHITestResult::fail("graphics module acquisition failed"); }
        const GraphicsPipelineDesc graphicsDesc{.vertexShader = {vertex->get()}, .fragmentShader = {fragment->get()},
            .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1};
        auto graphics = registry.getGraphicsPipeline(context.device, graphicsDesc);
        auto graphicsCached = registry.getGraphicsPipeline(context.device, graphicsDesc);
        auto changedGraphicsDesc = graphicsDesc;
        changedGraphicsDesc.fragmentShader.module = green->get();
        auto greenGraphics = registry.getGraphicsPipeline(context.device, changedGraphicsDesc);
        changedGraphicsDesc = graphicsDesc;
        changedGraphicsDesc.colorFormats[0] = Format::RGBA16Sfloat;
        auto otherFormat = registry.getGraphicsPipeline(context.device, changedGraphicsDesc);
        if (!graphics || !graphicsCached || !greenGraphics || !otherFormat || !graphicsCached->get()->pipelineCacheHit() ||
            graphics->get()->psoHash() != graphicsCached->get()->psoHash() ||
            graphics->get()->psoHash() == greenGraphics->get()->psoHash() ||
            graphics->get()->psoHash() == otherFormat->get()->psoHash()) {
            return RHITestResult::fail("default graphics cache aliased shader/state variants or omitted caching");
        }
        auto stats = registry.pipelineCacheStats(context.device);
        if (!stats || std::none_of(stats->begin(), stats->end(), [](const auto& group) {
                return group.group.starts_with("ShaderRegistry-RegistrySource-") &&
                    group.cache.storedPsoCount >= 4 && group.cache.backendDataSize > 0;
            })) { return RHITestResult::fail("registry did not retain and persist all pipeline states"); }
        // Separate native devices must load disk state, never borrow handles.
        auto secondDevice = createDevice({.applicationName = "ShaderRegistry lifecycle",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        if (!secondDevice) { return RHITestResult::fail("second registry device creation failed"); }
        auto secondShader = registry.getShaderModule(**secondDevice, {.spirv = first->spirv});
        if (!secondShader) { return RHITestResult::fail("second device shader creation failed"); }
        auto secondPipeline = registry.getComputePipeline(**secondDevice, {.computeShader = {secondShader->get()}});
        if (!secondPipeline || !secondPipeline->get()->pipelineCacheHit() ||
            secondPipeline->get()->execution().deviceIdentity() != (*secondDevice)->identity() ||
            secondPipeline->get()->execution().deviceIdentity() == context.device.identity()) {
            return RHITestResult::fail("registry leaked device state or did not reload persistent PSOs");
        }
        auto secondVertex = registry.getShaderModule(**secondDevice, {.spirv = vertexCode->spirv});
        auto secondFragment = registry.getShaderModule(**secondDevice, {.spirv = fragmentCode->spirv});
        if (!secondVertex || !secondFragment) { return RHITestResult::fail("second device graphics module creation failed"); }
        auto secondGraphicsDesc = graphicsDesc;
        secondGraphicsDesc.vertexShader.module = secondVertex->get();
        secondGraphicsDesc.fragmentShader.module = secondFragment->get();
        auto secondGraphics = registry.getGraphicsPipeline(**secondDevice, secondGraphicsDesc);
        if (!secondGraphics || !secondGraphics->get()->pipelineCacheHit() ||
            secondGraphics->get()->execution().deviceIdentity() != (*secondDevice)->identity()) {
            return RHITestResult::fail("independent device did not reload persistent graphics PSOs");
        }
        return RHITestResult::pass("source fallback/cache, compute/graphics variants, failure publication, concurrency and independent devices");
    }
};
METALLIC_REGISTER_RHI_TEST(ShaderRegistryTest);

} // namespace
} // namespace metallic::tests
