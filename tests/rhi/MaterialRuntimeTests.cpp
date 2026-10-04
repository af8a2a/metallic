#include "TestResourceLayouts.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "RHITest.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Material/MaterialExecutable.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SceneColorConversion.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Render/RenderSample.h"

#include <array>
#include <cstring>
#include <chrono>
#include <limits>
#include <fstream>
#include <cmath>

namespace metallic::tests {
namespace {

class MaterialRuntimeTest final : public RHITest
{
public:
    MaterialRuntimeTest() { name = "material_runtime_generations"; type = RHITestType::Resource; }

    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        std::vector<LegacyMaterialPayload> input(1000);
        for (uint32_t index = 0; index < input.size(); ++index) {
            input[index].baseColor[0] = float(index) / 1000.0f;
            input[index].baseColorTexture.textureIndex = index;
        }
        std::string log;
        auto first = MaterialGeneration::create(input, 7, log);
        if (!first || first->programCount() != 1 || first->instances().size() != 1000) {
            return RHITestResult::fail("Instances were not deduplicated by program: " + log);
        }
        const auto* program = first->instances()[0].program;
        uint32_t end = 0;
        for (const auto& field : program->schema->parameters) {
            if (field.offset != end || field.id.empty()) {
                return RHITestResult::fail("Schema does not cover the exact legacy upload layout");
            }
            end += field.size;
        }
        if (end != 720 || program->schema->abi != kLegacyMaterialABI) {
            return RHITestResult::fail("Material schema size/version mismatch");
        }
        for (uint32_t index = 0; index < input.size(); ++index) {
            if (first->instances()[index].program != program ||
                first->instances()[index].parameterIndex != index ||
                first->parameters()[index].baseColor[0] != input[index].baseColor[0]) {
                return RHITestResult::fail("Instance identity or dynamic parameters changed");
            }
        }
        input[0].baseColor[0] = 0.75f;
        input.back().rtxcrHairBaseColor[3] = 1.0f;
        auto second = MaterialGeneration::create(input, 8, log);
        if (!second || second->programCount() != 2 || second->serial() == first->serial() ||
            second->sourceRevision() != 8 || first->sourceRevision() != 7 ||
            first->parameters()[0].baseColor[0] != 0.0f ||
            second->instances()[0].program != program ||
            second->instances().back().program->definition->domain != MaterialDomain::Fiber ||
            second->instances().back().program->definition->capabilities.visibilityBuffer) {
            return RHITestResult::fail("Generation immutability, stable programs or Fiber capabilities failed");
        }
        const auto old = second;
        if (second->supports(MaterialEvaluationTarget::VisibilityBuffer, log) || log.empty() ||
            second->supports(MaterialEvaluationTarget::SurfaceRayHit, log) ||
            !second->supports(MaterialEvaluationTarget::RayHitWithFiber, log)) {
            return RHITestResult::fail("Unsupported Fiber execution target was silently accepted");
        }
        input[0].textureParams[2] = 99.0f;
        auto rejected = MaterialGeneration::create(input, 9, log);
        if (rejected || log.empty() || old->sourceRevision() != 8 ||
            MaterialGeneration::create({}, 9, log)) {
            return RHITestResult::fail("Invalid generation was accepted");
        }
        input[0].textureParams[2] = std::numeric_limits<float>::quiet_NaN();
        if (MaterialGeneration::create(input, 9, log)) {
            return RHITestResult::fail("NaN program ID was accepted");
        }
        input[0].textureParams[2] = 0.0f;
        auto recovered = MaterialGeneration::create(input, 10, log);
        if (!recovered || recovered->instances()[0].program->key != program->key || !log.empty()) {
            return RHITestResult::fail("Recovery recompiled instance parameters or retained an error");
        }
        input[0].params[0] = 1;
        input[0].metallicRoughnessTexture.ntcTextureSetIndex = 0;
        auto neural = MaterialGeneration::create(input, 11, log);
        if (!neural || neural->features()[0].surfaceProgram != material::SurfaceProgramClass::Opaque) {
            return RHITestResult::fail("NTC metalness was incorrectly specialized as a constant conductor");
        }
        const scene::RenderMaterial single;
        if (MaterialGeneration::create(input, 12, log, {&single, 1}) || log.empty()) {
            return RHITestResult::fail("Mismatched feature snapshot was accepted");
        }
        return RHITestResult::pass("1000 instances, two shared programs, complete schema and immutable generations");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialRuntimeTest);

class MaterialMigrationTest final : public RHITest
{
public:
    MaterialMigrationTest() { name = "material_runtime_schema_migration"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        const MaterialParameterSchema oldFields[] = {
            {"baseColor", MaterialParameterType::Float4, 16, 16},
            {"roughness", MaterialParameterType::Float, 0, 4}};
        const MaterialParameterSchema newFields[] = {
            {"roughness", MaterialParameterType::UInt, 16, 4},
            {"baseColor", MaterialParameterType::Float4, 0, 16},
            {"added", MaterialParameterType::Float, 20, 4}};
        const MaterialSchema oldSchema{1, 32, 16, oldFields}, newSchema{2, 32, 16, newFields};
        std::array<float, 8> values{0.3f, 0, 0, 0, 0.25f, 0.5f, 0.75f, 1};
        std::array<float, 8> defaults{0, 0, 0, 0, 0, 7, 0, 0};
        std::vector<std::byte> migrated;
        std::string log;
        if (!migrateMaterialParameters(oldSchema, std::as_bytes(std::span(values)), newSchema,
                std::as_bytes(std::span(defaults)), migrated, log) ||
            std::memcmp(migrated.data(), values.data() + 4, 16) != 0 ||
            std::memcmp(migrated.data() + 16, defaults.data() + 4, 16) != 0 ||
            log.find("roughness") == std::string::npos || log.find("added") == std::string::npos) {
            return RHITestResult::fail("Semantic migration lost reordered values, defaults or diagnostics: " + log);
        }
        auto malformed = newSchema;
        malformed.byteSize = 16;
        const auto previous = migrated;
        if (migrateMaterialParameters(oldSchema, std::as_bytes(std::span(values)), malformed,
                std::as_bytes(std::span(defaults)), migrated, log) || previous != migrated) {
            return RHITestResult::fail("Malformed layout mutated the last valid payload");
        }
        auto generation = MaterialGeneration::create(oldSchema, std::as_bytes(std::span(values)), 42, log);
        if (!generation || generation->parameters()[0].baseColor[0] != 0.25f ||
            generation->parameters()[0].glassParams[1] != 1.5f || log.empty()) {
            return RHITestResult::fail("Versioned layout was not lowered to the actual built-in upload ABI");
        }
        return RHITestResult::pass("Reorder, removed/added/type-changed defaults, malformed rejection and executable ABI lowering");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialMigrationTest);

std::array<render::LegacyMaterialPayload, 3> probeMaterials()
{
    std::array<render::LegacyMaterialPayload, 3> values;
    for (uint32_t index = 0; index < values.size(); ++index) {
        // Every word is distinct; the shader reads each named field into a flat
        // uint stream so this tests Slang offsets and stride against CPU offsets.
        std::array<uint32_t, 180> words;
        for (uint32_t word = 0; word < words.size(); ++word) {
            words[word] = 0x3f000000u + index * 256u + word;
        }
        std::memcpy(&values[index], words.data(), sizeof(values[index]));
        values[index].textureParams[2] = index == 2 ? 0.0f : float(index + 1);
        values[index].rtxcrHairBaseColor[3] = index == 0 ? 0.0f : 1.0f;
    }
    return values;
}

class MaterialRuntimeProbePass final : public render::ComputePass
{
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        auto& result = reflection.addBufferOutput("result").buffer(3 * 181 * 4, 4).storageWrite();
        result.memoryLocation = render::MemoryLocation::HostReadback;
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        const auto values = probeMaterials();
        auto result = context.device->createBuffer({.size = sizeof(values),
            .structureStride = sizeof(LegacyMaterialPayload), .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload})
            .transform([&](auto buffer) { input_ = std::move(buffer); });
        if (!result) { return result; }
        void* mapped = input_->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, values.data(), sizeof(values));
        input_->flush();
        input_->unmap();
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "MaterialRuntimeProbe",
            .entryPointName = "materialRuntimeProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
            log).transform([&](auto value) { shader = std::move(value); });
        if (!result) { return result; }
        const std::array bindings{
            ComputeProgramBindingDesc{.binding = 0, .kind = ComputeResourceBindingKind::StorageBuffer},
            ComputeProgramBindingDesc{.binding = 1, .kind = ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {.spirv = shader.spirv,
            .bindings = bindings, .requiresRayQuery = false, .resourceParameters = metallic::tests::kMaterialRuntimeProbeLayout}, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const std::array bindings{
            render::ComputeDispatchBinding{.binding = 0, .buffer = input_.get()},
            render::ComputeDispatchBinding{.binding = 1, .buffer = context.outputBuffer("result").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings});
    }
private:
    std::unique_ptr<render::Buffer> input_;
    render::ComputeProgram program_;
};

class MaterialRuntimeABITest final : public RHITest
{
public:
    MaterialRuntimeABITest() { name = "material_runtime_gpu_abi"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        registerRenderGraphPassType("MaterialRuntimeProbe", "Material runtime ABI",
            [] { return std::make_unique<MaterialRuntimeProbePass>(); });
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Material ABI", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
        if (!result) { return RHITestResult::fail("Device initialization failed"); }
        RenderGraph graph;
        graph.addNode("MaterialRuntimeProbe", "Probe");
        graph.markOutput("Probe.result");
        RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(*device, graph, 1, 1, log)) { return RHITestResult::fail(log); }
        if (!executor.execute({.graphicsQueue = device->getQueue(QueueType::Graphics)}) ||
            !executor.waitForSubmittedWork()) { return RHITestResult::fail("ABI probe execution failed"); }
        auto* buffer = executor.outputResource("Probe.result")->buffer;
        buffer->invalidate();
        const void* mapped = buffer->map();
        if (!mapped) { return RHITestResult::fail("ABI probe readback failed"); }
        std::array<uint32_t, 3 * 181> output{};
        std::memcpy(output.data(), mapped, sizeof(output));
        buffer->unmap();
        const auto expected = probeMaterials();
        for (uint32_t index = 0; index < expected.size(); ++index) {
            if (std::memcmp(output.data() + index * 181, &expected[index], 720) != 0 ||
                output[index * 181 + 180] != (index == 0 ? 1u : 2u)) {
                return RHITestResult::fail("CPU/Slang field offsets, stride or program identity mismatch");
            }
        }
        return RHITestResult::pass("All 720 bytes, array stride, explicit IDs and legacy hair identity agree on GPU");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialRuntimeABITest);

class MaterialRuntimeSceneTest final : public RHITest
{
public:
    MaterialRuntimeSceneTest() { name = "material_runtime_scene_publication"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto initialized = createDevice({.applicationName = "Material scene publication",
            .enableValidation = context.enableValidation, .enableRayTracingAccelerationStructure = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(initialized, Error::Unsupported)) { return RHITestResult::skip("Requires ray tracing"); }
        if (!initialized) { return RHITestResult::fail("Material scene device initialization failed"); }
        auto& queue = *device->getQueue(QueueType::Graphics);
        const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) /
            "Asset/LookDev/OpenPBRDefault/OpenPbrDefault.gltf";
        scene::SceneDocument document;
        if (!document.load(path)) { return RHITestResult::fail("LookDev scene load failed"); }
        std::string log;
        // The same materials-only preparation is used by StreamAsset consumers;
        // test both upload routes without requiring a second scene fixture.
        for (bool materialsOnly : {false, true}) {
            ScenePathTraceResources resources;
            auto result = resources.beginPrepareAsync(*device, queue,
                {{"path", path.string()}}, document, log, materialsOnly);
            if (hasError(result, Error::Unsupported)) { return RHITestResult::skip(log); }
            if (!result) { return RHITestResult::fail(log); }
            bool complete = false;
            scene::SceneLoadProgress progress;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!complete && std::chrono::steady_clock::now() < deadline) {
                result = resources.pumpPrepareAsync(8.0, progress, log)
                    .transform([&](auto value) { complete = value; });
                if (!result || !queue.waitIdle()) { return RHITestResult::fail(log); }
            }
            auto first = resources.materialGeneration();
            if (!complete || !resources.valid() || !first ||
                first->instances().size() != document.materials().size() ||
                resources.materialBuffer()->desc().structureStride != first->instances()[0].program->schema->byteSize) {
                return RHITestResult::fail("Scene upload did not publish matching GPU/instance generations");
            }
            const auto* program = first->instances()[0].program;
            const auto original = document.materials()[0];
            const auto originalWorking = resolveWorkingMaterial(original);
            auto edited = original;
            edited.featurePolicies.transmission = material::FeaturePolicy::Dynamic;
            edited.baseColorFactor.x = original.baseColorFactor.x == 0.25f ? 0.75f : 0.25f;
            const auto editedWorking = resolveWorkingMaterial(edited);
            if (!document.setMaterialProperties(0, edited)) { return RHITestResult::fail("Edit rejected"); }
            const auto oldBinding = resources.materialBinding();
            auto refuseAllocation = +[](Device&, const BufferDesc&) -> Result<std::unique_ptr<Buffer>> {
                return makeError(Error::Failure);
            };
            if (resources.syncRuntimeScene(&document, log, refuseAllocation) || log.empty() ||
                resources.materialGeneration() != first || resources.materialBinding() != oldBinding ||
                resources.materialBuffer() != oldBinding->buffer()) {
                return RHITestResult::fail("Failed upload published partial material state");
            }
            auto refuseMapping = +[](Device&, const BufferDesc&) -> Result<std::unique_ptr<Buffer>> {
                return std::make_unique<Buffer>(); // Valid wrapper, deliberately no mappable allocation.
            };
            if (resources.syncRuntimeScene(&document, log, refuseMapping) ||
                log.find("failed to map") == std::string::npos || resources.materialBinding() != oldBinding) {
                return RHITestResult::fail("Mapping failure replaced the prior material generation");
            }
            if (!resources.syncRuntimeScene(&document, log)) { return RHITestResult::fail(log); }
            auto second = resources.materialGeneration();
            const auto expectedFeatures = material::resolveMaterialFeatures(edited);
            if (second->features()[0].programSignature != expectedFeatures.programSignature ||
                second->features()[0].surfaceProgram != material::SurfaceProgramClass::General ||
                first->features()[0].programSignature != material::resolveMaterialFeatures(original).programSignature) {
                return RHITestResult::fail("Feature policy was not atomically published with its material revision");
            }
            if (!second || first == second || second->instances()[0].program != program ||
                second->sourceRevision() != document.materialRevision() ||
                second->parameters()[0].baseColor[0] != editedWorking.baseColorFactor.x ||
                first->parameters()[0].baseColor[0] != originalWorking.baseColorFactor.x) {
                return RHITestResult::fail("Material edit changed program identity or mutated a prior snapshot");
            }
            const void* mapped = resources.materialBuffer()->map();
            if (!mapped) { return RHITestResult::fail("Updated material buffer is not readable"); }
            const bool equal = std::memcmp(mapped, second->parameters().data(), second->parameters().size_bytes()) == 0;
            resources.materialBuffer()->unmap();
            if (!equal) { return RHITestResult::fail("Published GPU parameters differ from the CPU generation"); }
            if (!resources.syncRuntimeScene(&document, log) || resources.materialGeneration() != second) {
                return RHITestResult::fail("Unchanged scene republished material programs");
            }
            if (!document.setMaterialProperties(0, original) ||
                !resources.syncRuntimeScene(&document, log)) { return RHITestResult::fail(log); }
            if (resources.materialGeneration()->parameters()[0].baseColor[0] != originalWorking.baseColorFactor.x) {
                return RHITestResult::fail("Undo did not restore material parameters");
            }
            resources.clear();
            if (resources.materialGeneration() || resources.materialBuffer() ||
                first->parameters()[0].baseColor[0] != originalWorking.baseColorFactor.x) {
                return RHITestResult::fail("Clear retained publication or destroyed a held CPU generation");
            }
        }
        return RHITestResult::pass("Resident and material-only uploads, edits, no-op sync, undo and snapshot retention");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialRuntimeSceneTest);

class MaterialRetirementTest final : public RHITest
{
public:
    MaterialRetirementTest() { name = "material_runtime_inflight_reload"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Material retirement",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!result) { return RHITestResult::fail("Device creation failed"); }
        auto& queue = *device->getQueue(QueueType::Graphics);
        std::unique_ptr<Semaphore> gate;
        if (!device->createSemaphore({})
                .transform([&](auto value) { gate = std::move(value); })) { return RHITestResult::fail("Gate failed"); }
        RenderFrameContext frame;
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Buffer> output;
        QueueSubmissionTracker tracker;
        if (!tracker.initialize(*device, queue)) { return RHITestResult::fail("Tracker failed"); }
        // Declared last: always release the gate before any waiting destructor.
        struct Drain {
            Queue& queue; Semaphore& gate;
            ~Drain() { if (gate.currentValue() < 1) { (void)gate.signal(1); } (void)queue.waitIdle(); }
        } drain{queue, *gate};
        if (!device->createCommandPool(queue).transform([&](auto v) { pool = std::move(v); }) ||
            !pool->createCommandBuffer().transform([&](auto v) { commands = std::move(v); }) ||
            !device->createBuffer({.size = 3 * 181 * 4, .structureStride = 4,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
                .transform([&](auto v) { output = std::move(v); })) { return RHITestResult::fail("Resources failed"); }
        std::string log;
        std::array<LegacyMaterialPayload, 3> values;
        values[0].baseColor[0] = 0.25f;
        const auto makeBinding = [&](uint64_t revision) -> std::shared_ptr<MaterialBindingGeneration> {
            auto generation = MaterialGeneration::create(values, revision, log);
            std::unique_ptr<Buffer> buffer;
            if (!generation || !device->createBuffer({.size = sizeof(values), .structureStride = sizeof(LegacyMaterialPayload),
                    .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload})
                    .transform([&](auto v) { buffer = std::move(v); })) { return {}; }
            auto* mapped = buffer->map();
            if (!mapped) { return {}; }
            std::memcpy(mapped, generation->parameters().data(), sizeof(values));
            buffer->flush(); buffer->unmap();
            return std::make_shared<MaterialBindingGeneration>(std::move(generation), std::move(buffer));
        };
        auto published = makeBinding(1);
        if (!published) { return RHITestResult::fail(log); }
        const auto originalParameters = published->generation();
        std::weak_ptr<MaterialBindingGeneration> oldBinding = published;
        ComputeProgram program;
        std::shared_ptr<const MaterialExecutableArtifact> artifact;
        const ComputeProgramBindingDesc layout[] = {{.binding = 0}, {.binding = 1}};
        SlangShaderDesc source{.moduleName = "MaterialRuntimeProbe", .entryPointName = "materialRuntimeProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"};
        scene::RenderMaterial semantic;
        semantic.metallicFactor = 0;
        const auto features = material::resolveMaterialFeatures(semantic);
        const auto classification = std::to_string(static_cast<uint32_t>(features.surfaceProgram));
        const SlangMacroDefine featureDefine{"MATERIAL_CLASS", classification.c_str()};
        source.macroDefines = {&featureDefine, 1};
        const ComputeProgramDesc description{.bindings = layout, .requiresRayQuery = false,
            .resourceParameters = kMaterialRuntimeProbeLayout};
        if (!compileMaterialExecutable(*device, source, description, program, artifact, log)) {
            return RHITestResult::fail(log);
        }
        auto originalArtifact = artifact;
        std::weak_ptr<const MaterialExecutableArtifact> retiredArtifact = artifact;
        const auto cacheBefore = materialProgramCacheStats();
        ComputeProgram sharedProgram;
        std::shared_ptr<const MaterialExecutableArtifact> sharedArtifact;
        semantic.roughnessFactor = 0.17f;
        semantic.baseColorFactor.x = 0.42f;
        semantic.baseColorTexture.textureIndex = 77;
        semantic.alphaMode = "MASK";
        semantic.doubleSided = true;
        semantic.featurePolicies.metalness = material::FeaturePolicy::Specialization;
        const auto otherFeatures = material::resolveMaterialFeatures(semantic);
        if (otherFeatures.programSignature != features.programSignature ||
            otherFeatures.visibilitySignature == features.visibilitySignature ||
            otherFeatures.pipelineSignature == features.pipelineSignature) {
            return RHITestResult::fail("Dynamic, visibility or pipeline features leaked into ProgramSignature");
        }
        const auto otherClass = std::to_string(static_cast<uint32_t>(otherFeatures.surfaceProgram));
        const SlangMacroDefine otherDefine{"MATERIAL_CLASS", otherClass.c_str()};
        source.macroDefines = {&otherDefine, 1};
        if (!compileMaterialExecutable(*device, source, description, sharedProgram, sharedArtifact, log) ||
            sharedArtifact != artifact || sharedArtifact->programKey != artifact->programKey ||
            materialProgramCacheStats().hits != cacheBefore.hits + 1 ||
            materialProgramCacheStats().pipelineBuilds != cacheBefore.pipelineBuilds) {
            return RHITestResult::fail("Identical ProgramKey did not reuse the executable generation");
        }
        sharedProgram.clear(); sharedArtifact.reset();
        if (!program.valid()) { return RHITestResult::fail("Clearing a cache alias invalidated its peer"); }
        if (!frame.begin(0) || !commands->begin(frame.submissionContext())) { return RHITestResult::fail("Begin failed"); }
        frame.retain(published);
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .buffer = published->buffer()}, {.binding = 1, .buffer = output.get()}};
        if (!program.dispatch({.commandBuffer = commands.get(), .bindings = bindings}) || !commands->end()) {
            return RHITestResult::fail("Material dispatch failed");
        }
        CommandBuffer* recorded[] = {commands.get()};
        const SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        if (!tracker.submit({.waitSemaphores = {&wait, 1}, .commandBuffers = recorded}, frame) ||
            frame.completion().isComplete()) { return RHITestResult::fail("Submission was not held by the gate"); }

        source.entryPointName = "missingMaterialEntry";
        if (compileMaterialExecutable(*device, source, description, program, artifact, log) ||
            artifact != originalArtifact || !program.valid() || log.empty()) {
            return RHITestResult::fail("Compile failure replaced last successful executable");
        }
        source.entryPointName = "materialRuntimeProbeMain";
        const ComputeProgramBindingDesc invalidManifest[] = {{.binding = 0}, {.binding = 0}};
        if (compileMaterialExecutable(*device, source,
                {.bindings = invalidManifest, .requiresRayQuery = false,
                 .resourceParameters = kMaterialRuntimeProbeLayout}, program, artifact, log) ||
            artifact != originalArtifact) { return RHITestResult::fail("Invalid manifest was published"); }
        const SlangMacroDefine revision{"MATERIAL_PROBE_REVISION", "1"};
        source.macroDefines = {&revision, 1};
        if (!compileMaterialExecutable(*device, source, description, program, artifact, log) ||
            artifact->key == originalArtifact->key || artifact->generation == originalArtifact->generation ||
            artifact->programKey.irHash == originalArtifact->programKey.irHash ||
            artifact->programKey.definitionHash != originalArtifact->programKey.definitionHash) {
            return RHITestResult::fail("Executable recovery failed: " + log);
        }
        // The weak cache must not pin the old artifact. Its GPU kernel must
        // survive solely through the already-recorded, still-blocked dispatch.
        originalArtifact.reset();
        if (!retiredArtifact.expired()) { return RHITestResult::fail("Cache pinned an obsolete generation"); }
        values[0].baseColor[0] = 0.75f;
        published = makeBinding(2);
        if (!published || oldBinding.expired() || frame.completion().isComplete()) {
            return RHITestResult::fail("Pending material resources retired before completion");
        }
        if (!gate->signal(1) || !frame.wait(5'000'000'000ull)) { return RHITestResult::fail("Gate completion failed"); }
        output->invalidate();
        auto* mapped = static_cast<const uint32_t*>(output->map());
        bool correct = mapped != nullptr;
        for (size_t index = 0; correct && index < 3; ++index) {
            correct = std::memcmp(mapped + index * 181, &originalParameters->parameters()[index], 720) == 0 &&
                mapped[index * 181 + 180] == 1;
        }
        if (mapped) { output->unmap(); }
        if (!correct) { return RHITestResult::fail("Old submission observed mixed code/parameters/descriptors"); }
        if (!pool->reset() || !frame.reset() || !oldBinding.expired()) {
            return RHITestResult::fail("Completed material generation did not retire");
        }
        // Submit the recovered executable and new parameters, verifying that the
        // previous output was not merely a permanently stale dispatch.
        if (!frame.begin(1) || !commands->begin(frame.submissionContext())) { return RHITestResult::fail("Recovery begin failed"); }
        const ComputeDispatchBinding next[] = {
            {.binding = 0, .buffer = published->buffer()}, {.binding = 1, .buffer = output.get()}};
        if (!program.dispatch({.commandBuffer = commands.get(), .bindings = next}) || !commands->end() ||
            !tracker.submit({.commandBuffers = recorded}, frame) || !frame.wait(5'000'000'000ull)) {
            return RHITestResult::fail("Recovered dispatch failed");
        }
        output->invalidate(); mapped = static_cast<const uint32_t*>(output->map());
        correct = mapped && std::memcmp(mapped, &published->generation()->parameters()[0], 720) == 0 && mapped[180] == 11;
        if (mapped) { output->unmap(); }
        (void)pool->reset(); (void)frame.reset();
        return correct ? RHITestResult::pass("Gated old submission retained code/data/descriptors; failed compile/manifest preserved state; recovery and retirement verified")
            : RHITestResult::fail("Recovered generation was not executed");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialRetirementTest);

class MaterialErrorPass final : public render::ComputePass
{
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").storageWrite().format = render::Format::RGBA32Sfloat;
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        std::shared_ptr<const render::MaterialExecutableArtifact> artifact;
        const auto failed = render::compileMaterialExecutable(*context.device,
            {.moduleName = "MaterialRuntimeProbe", .entryPointName = "missingMaterialEntry",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
            {.requiresRayQuery = false}, program_, artifact, log);
        if (failed || program_.valid() || artifact || log.empty()) { return render::makeError(render::Error::Failure); }
        std::string fallbackLog;
        return render::initializeMaterialErrorProgram(*context.device, program_, fallbackLog);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const uint32_t color = 1;
        const render::ComputeDispatchBinding binding{.binding = 0, .textureView = context.outputTexture("color").view()};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {&binding, 1},
            .pushData = &color, .pushDataSize = sizeof(color), .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8});
    }
private:
    render::ComputeProgram program_;
};

class MaterialErrorTest final : public RHITest
{
public:
    MaterialErrorTest() { name = "material_runtime_error_material"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        registerRenderGraphPassType("MaterialErrorProbe", "Material error", [] { return std::make_unique<MaterialErrorPass>(); });
        RenderGraph graph;
        graph.addNode("MaterialErrorProbe", "Error"); graph.markOutput("Error.color");
        RenderGraphPreviewRenderer preview;
        preview.setRawReadbackEnabled(true);
        if (!preview.initialize(context.enableValidation) || !preview.render(graph, 17, 9, "Error.color")) {
            return RHITestResult::fail(preview.lastLog());
        }
        if (preview.readbackFormat() != Format::RGBA32Sfloat || preview.readbackBytes().size() != 17 * 9 * 16) {
            return RHITestResult::fail("Error material HDR readback has wrong layout");
        }
        std::array<float, 17 * 9 * 4> pixels;
        std::memcpy(pixels.data(), preview.readbackBytes().data(), sizeof(pixels));
        for (uint32_t y = 0; y < 9; ++y) {
            for (uint32_t x = 0; x < 17; ++x) {
                const auto offset = (y * 17 + x) * 4;
                const float expected = ((x / 8 + y / 8) & 1) ? 1.0f : 0.08f;
                if (pixels[offset] != expected || pixels[offset + 1] != 0 ||
                    pixels[offset + 2] != expected || pixels[offset + 3] != 1) {
                    return RHITestResult::fail("First failure did not render the explicit error material");
                }
            }
        }
        return RHITestResult::pass("First compile failure produced a deterministic magenta checker in floating point");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialErrorTest);

class MaterialFeatureRenderingTest final : public RHITest
{
public:
    MaterialFeatureRenderingTest() { name = "material_feature_policy_rendering"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::vector<float> reference;
        double squaredError = 0, squaredReference = 0, maximumError = 0;
        for (bool dynamic : {false, true}) {
            RenderSampleLoadResult sample;
            std::string log;
            if (!loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
            scene::SceneDocument document;
            if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
                return RHITestResult::fail("Feature fixture load failed");
            }
            if (dynamic) {
                for (int32_t i = 0; i < static_cast<int32_t>(document.materials().size()); ++i) {
                    auto edited = document.materials()[i];
                    edited.featurePolicies = {material::FeaturePolicy::Dynamic, material::FeaturePolicy::Dynamic};
                    if (!document.setMaterialProperties(i, edited)) { return RHITestResult::fail("Feature policy edit rejected"); }
                }
            }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&document);
            preview.setEnvironment(document.environment()); preview.setLighting(document.lighting());
            preview.setRawReadbackEnabled(true);
            if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Feature device failed"); }
            uint64_t builds = 0;
            for (uint32_t frame = 0; frame < 8; ++frame) {
                if (!preview.render(sample.graph, 256, 256, "Deferred.color")) { return RHITestResult::fail(preview.lastLog()); }
                if (frame == 0) { builds = materialProgramCacheStats().pipelineBuilds; }
                if (materialProgramCacheStats().pipelineBuilds != builds) {
                    return RHITestResult::fail("Instances sharing a Feature ProgramSignature rebuilt shader pipelines");
                }
            }
            const auto& bytes = preview.readbackBytes();
            if (preview.readbackFormat() != Format::RGBA32Sfloat || bytes.size() != 256 * 256 * 16) {
                return RHITestResult::fail("Unexpected Feature HDR readback format");
            }
            std::ofstream evidence(context.outputDirectory / (dynamic ? "Dynamic.rgba32f" : "Auto.rgba32f"), std::ios::binary);
            evidence.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            std::vector<float> pixels(bytes.size() / sizeof(float));
            std::memcpy(pixels.data(), bytes.data(), bytes.size());
            for (float value : pixels) { if (!std::isfinite(value)) { return RHITestResult::fail("Nonfinite Feature output"); } }
            if (!dynamic) { reference = std::move(pixels); }
            else {
                for (size_t i = 0; i < pixels.size(); ++i) {
                    const double difference = double(pixels[i]) - reference[i];
                    squaredError += difference * difference;
                    squaredReference += double(reference[i]) * reference[i];
                    maximumError = std::max(maximumError, std::abs(difference));
                }
            }
        }
        const double relativeRMSE = std::sqrt(squaredError / std::max(squaredReference, 1e-20));
        std::ofstream(context.outputDirectory / "FeatureComparison.txt") << "relativeRMSE=" << relativeRMSE << " maxAbsolute=" << maximumError << '\n';
        if (relativeRMSE > 1e-5) { return RHITestResult::fail("Auto and Dynamic closure policies changed shading: " + std::to_string(relativeRMSE)); }
        return RHITestResult::pass("Production deferred Auto/Dynamic closures agree; shared signatures allocate no per-frame pipelines");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialFeatureRenderingTest);

class MaterialHDRCaptureTest final : public RHITest
{
public:
    MaterialHDRCaptureTest() { name = "material_runtime_hdr_capture"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        // Each output starts with a fresh history. Baseline/current source runs
        // use this exact workload; a raw dump never goes through preview UNORM.
        for (const char* outputName : {"Reference.color", "Deferred.color", "PathTrace.color"}) {
            const bool fiber = std::string_view(outputName) == "PathTrace.color";
            RenderSampleLoadResult sample;
            std::string log;
            if (!loadBuiltInRenderSample(fiber ? "rtxcr-material-sample" : "lookdev-vbuffer", sample, log)) {
                return RHITestResult::fail(log);
            }
            scene::SceneDocument document;
            if (!fiber && !document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
                return RHITestResult::fail("LookDev load failed");
            }
            RenderGraphPreviewRenderer preview;
            if (!fiber) {
                preview.bindRuntimeScene(&document);
                preview.setEnvironment(document.environment()); preview.setLighting(document.lighting());
            } else {
                const auto& environment = sample.desc.environment.value();
                preview.setEnvironment({.enabled = environment.enabled, .path = environment.path,
                    .intensity = environment.intensity, .rotationDegrees = environment.rotationDegrees,
                    .visible = environment.visible});
            }
            if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("HDR device failed"); }
            for (uint32_t frame = 0; frame < 256; ++frame) {
                preview.setRawReadbackEnabled(frame == 255);
                if (!preview.render(sample.graph, 768, fiber ? 432 : 768, outputName)) { return RHITestResult::fail(preview.lastLog()); }
            }
            const auto format = preview.readbackFormat();
            if (format != Format::RGBA16Sfloat && format != Format::RGBA32Sfloat) {
                return RHITestResult::fail("Material output is not linear floating point");
            }
            const auto& bytes = preview.readbackBytes();
            const auto path = context.outputDirectory / (std::string(outputName) +
                (format == Format::RGBA16Sfloat ? ".rgba16f" : ".rgba32f"));
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (!stream) { return RHITestResult::fail("HDR evidence write failed"); }
            // NaN and infinity must not be hidden by UNORM conversion.
            if (format == Format::RGBA16Sfloat) {
                for (size_t i = 0; i < bytes.size(); i += 2) {
                    uint16_t bits; std::memcpy(&bits, bytes.data() + i, 2);
                    if ((bits & 0x7c00u) == 0x7c00u) { return RHITestResult::fail("Nonfinite HDR output"); }
                }
            } else {
                for (size_t i = 0; i < bytes.size(); i += 4) {
                    float value; std::memcpy(&value, bytes.data() + i, 4);
                    if (!std::isfinite(value)) { return RHITestResult::fail("Nonfinite HDR output"); }
                }
            }
        }
        return RHITestResult::pass("Frozen LookDev PT/deferred and Claire groom, 256 frames each, finite raw HDR evidence");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialHDRCaptureTest);

} // namespace
} // namespace metallic::tests
