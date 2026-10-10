#include "RHITest.h"
#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "TestResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"

#include <chrono>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

class MaterialAssetProbePass final : public render::ComputePass
{
public:
    explicit MaterialAssetProbePass(std::shared_ptr<render::MaterialBindingGeneration> binding) : binding_(std::move(binding)) {}
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        auto& result = reflection.addBufferOutput("result").buffer(binding_->buffer()->desc().size, 4).storageWrite();
        result.memoryLocation = render::MemoryLocation::HostReadback;
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        ShaderCompileResult shader;
        const auto result = compileSlangShaderToSpirv({.moduleName = "MaterialAssetProbe", .entryPointName = "materialAssetCopyMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log).transform([&](auto value) { shader = std::move(value); });
        if (!result) { return result; }
        const std::array bindings{ComputeResourceBindingDesc{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialRuntimeProbeResources, materials), .kind = ComputeResourceBindingKind::StorageBuffer},
            ComputeResourceBindingDesc{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialRuntimeProbeResources, output), .kind = ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {.spirv = shader.spirv, .bindings = bindings,
            .requiresRayQuery = false, .resourceParameterSize = sizeof(metallic::tests::MaterialRuntimeProbeResources)}, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const std::array bindings{render::ComputeDispatchBinding{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialRuntimeProbeResources, materials), .buffer = binding_->buffer()},
            render::ComputeDispatchBinding{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialRuntimeProbeResources, output), .buffer = context.outputBuffer("result").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .groupCountX = static_cast<uint32_t>((binding_->buffer()->desc().size / 4 + 63) / 64)});
    }
private:
    std::shared_ptr<render::MaterialBindingGeneration> binding_;
    render::ComputeProgram program_;
};

bool checkUpload(render::Device& device, render::Queue& queue, const std::shared_ptr<render::MaterialBindingGeneration>& binding,
    std::span<const render::LegacyMaterialPayload> expected, std::string& log)
{
    using namespace render;
    registerRenderGraphPassType("MaterialAssetProbe", "Material asset readback",
        [weak = std::weak_ptr(binding)] { return std::make_unique<MaterialAssetProbePass>(weak.lock()); });
    RenderGraph graph;
    graph.addNode("MaterialAssetProbe", "Probe");
    graph.markOutput("Probe.result");
    RenderGraphExecutor executor;
    if (!executor.compile(device, graph, 1, 1, log) || !executor.execute({.graphicsQueue = &queue}) ||
        !executor.waitForSubmittedWork()) { return false; }
    auto* output = executor.outputResource("Probe.result")->buffer;
    output->invalidate();
    const auto* mapped = output->map();
    if (!mapped) { return false; }
    const bool same = std::memcmp(mapped, expected.data(), expected.size_bytes()) == 0;
    output->unmap();
    return same;
}

class MaterialAssetUploadTest final : public RHITest
{
public:
    MaterialAssetUploadTest()
    {
        name = "material_asset_upload_equivalence";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto root = context.outputDirectory / "assets";
        std::filesystem::create_directories(root);
        const auto definition = material::defaultOpenPBRDefinition();
        std::ofstream(root / "OpenPBR.materialdef") << material::serializeMaterialDefinition(definition);
        material::MaterialAssetLibrary library(root);
        std::string log;
        const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/LookDev/OpenPBRDefault/OpenPbrDefault.gltf";
        std::unique_ptr<Device> device;
        if (!createDevice({.applicationName = "Material asset upload equivalence", .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableRayTracingAccelerationStructure = true})
                .transform([&](auto value) { device = std::move(value); })) {
            return RHITestResult::fail("Material asset device failed");
        }
        auto& queue = *device->getQueue(QueueType::Graphics);
        for (bool materialsOnly : {false, true}) {
            scene::SceneDocument document;
            if (!document.load(path)) { return RHITestResult::fail("Material fixture failed"); }
            ScenePathTraceResources resources;
            if (!resources.beginPrepareAsync(*device, queue, {{"path", path.string()}}, document, log, materialsOnly)) {
                return RHITestResult::fail(log);
            }
            bool complete = false;
            scene::SceneLoadProgress progress;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!complete && std::chrono::steady_clock::now() < deadline) {
                if (!resources.pumpPrepareAsync(8.0, progress, log).transform([&](auto value) { complete = value; })) {
                    return RHITestResult::fail(log);
                }
                if (!queue.waitIdle()) { return RHITestResult::fail("Upload queue wait failed"); }
            }
            if (!complete || !resources.valid()) { return RHITestResult::fail("Asset upload timed out: " + log); }
            const auto original = resources.materialGeneration();
            if (!original) { return RHITestResult::fail("Missing initial material generation"); }
            const auto identity = document.resourceIdentity();
            for (size_t index = 0; index < document.materials().size(); ++index) {
                material::MaterialInstance instance;
                if (!material::createMaterialInstance(document.materials()[index], "asset://OpenPBR.materialdef", definition,
                    [](auto slot, auto) { return "imported://" + std::string(slot); }, instance, log)) { return RHITestResult::fail(log); }
                const auto uri = "asset://Material" + std::to_string(index) + ".material";
                if (!library.save(uri, instance, log) || !document.setMaterialAsset(static_cast<int32_t>(index), uri, root, log)) {
                    return RHITestResult::fail(log);
                }
            }
            if (document.resourceIdentity() != identity || !resources.syncRuntimeScene(&document, log)) {
                return RHITestResult::fail("Asset round trip changed geometry/resource identity: " + log);
            }
            const auto current = resources.materialGeneration();
            if (current->parameters().size_bytes() != original->parameters().size_bytes() ||
                std::memcmp(current->parameters().data(), original->parameters().data(), original->parameters().size_bytes()) != 0) {
                return RHITestResult::fail("Asset round trip changed the legacy GPU payload");
            }
            const auto* sharedProgram = findMaterialProgram(definition.implementation);
            if (!sharedProgram) { return RHITestResult::fail("Definition has no existing MaterialProgram"); }
            for (const auto& instance : current->instances()) {
                if (instance.program != sharedProgram) { return RHITestResult::fail("Asset created a per-instance program"); }
            }
            if (!checkUpload(*device, queue, resources.materialBinding(), original->parameters(), log)) {
                return RHITestResult::fail("GPU upload differs after asset round trip: " + log);
            }
            auto edited = document.materials()[0];
            edited.roughnessFactor = 0.67f;
            if (!document.setMaterialProperties(0, edited) || !resources.syncRuntimeScene(&document, log) ||
                resources.materialGeneration()->instances()[0].program != sharedProgram ||
                resources.materialGeneration()->parameters()[0].params[1] != 0.67f ||
                !checkUpload(*device, queue, resources.materialBinding(), resources.materialGeneration()->parameters(), log)) {
                return RHITestResult::fail("Asset parameter edit changed program or failed to upload");
            }
            const uint32_t pixels[] = {0xff2040ffu, 0xff2040ffu, 0xff2040ffu, 0xff2040ffu};
            if (!saveRgba8Png(root / "Color.png", reinterpret_cast<const uint8_t*>(pixels), 2, 2, log)) { return RHITestResult::fail(log); }
            material::MaterialInstance texturedAsset;
            texturedAsset.definition = "asset://OpenPBR.materialdef";
            texturedAsset.resources["baseColorTexture"] = "asset://Color.png";
            if (!library.save("asset://Textured.material", texturedAsset, log) ||
                !document.setMaterialAsset(0, "asset://Textured.material", root, log)) { return RHITestResult::fail(log); }
            ScenePathTraceResources textured;
            if (!textured.beginPrepareAsync(*device, queue, {{"path", path.string()}}, document, log, materialsOnly)) { return RHITestResult::fail(log); }
            complete = false;
            const auto textureDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!complete && std::chrono::steady_clock::now() < textureDeadline) {
                if (!textured.pumpPrepareAsync(8.0, progress, log).transform([&](auto value) { complete = value; }) || !queue.waitIdle()) {
                    return RHITestResult::fail(log);
                }
            }
            if (!complete || textured.materialTextureCount() < 2 ||
                textured.materialGeneration()->parameters()[0].baseColorTexture.textureIndex == UINT32_MAX ||
                !checkUpload(*device, queue, textured.materialBinding(), textured.materialGeneration()->parameters(), log)) {
                return RHITestResult::fail("Semantic texture URI did not reach the production GPU upload: " + log);
            }
        }
        return RHITestResult::pass("All OpenPBR asset fields preserved the 720-byte payload; resident/material-only uploads share existing programs");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialAssetUploadTest);

} // namespace
} // namespace metallic::tests
