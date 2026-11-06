#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeResourceParameters.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Material/MaterialExecutable.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string_view>

namespace metallic::tests {
namespace {

using namespace render;

class UnmarkedSceneArtifactOverride final : public ShaderArtifactOverride
{
public:
    explicit UnmarkedSceneArtifactOverride(const ShaderCompileResult& shader) : shader_(shader) {}

    void setShader(const ShaderCompileResult& shader)
    {
        shader_ = shader;
    }

    std::optional<Result<ShaderArtifactOverrideResult>> resolve(const SlangShaderDesc& desc, std::string&) override
    {
        if (!desc.moduleName || std::string_view(desc.moduleName) != "TrustedSceneResourceProbe") { return std::nullopt; }
        ShaderArtifactOverrideResult artifact;
        artifact.shader = shader_;
        // The provider must not be able to impersonate an ordinary source result.
        artifact.shader.fromArtifactOverride = false;
        artifact.identity.moduleName = "Artifacts/TrustedSceneResourceProbe";
        artifact.identity.sourceEntryPoint = "main";
        artifact.identity.entryPoint = "main";
        artifact.identity.descriptorHeapMode = "mapped";
        artifact.identity.optimizationLevel = "O1";
        artifact.identity.debugMode = "release";
        artifact.identity.spirvPostprocess = "none";
        return std::optional<Result<ShaderArtifactOverrideResult>>{std::move(artifact)};
    }

private:
    ShaderCompileResult shader_;
};

class TrustedSceneResourceTest final : public RHITest
{
public:
    TrustedSceneResourceTest()
    {
        type = RHITestType::Resource;
        name = "trusted_scene_resource_contract_and_bounds";
    }

    RHITestResult run(RHITestContext& context) override
    {
        static_assert(sizeof(SceneResourceParameters) == 408);
        const auto fail = [](const std::string& text) { return RHITestResult::fail(text); };
        std::string log;
        const SlangMacroDefine validDefines[] = {
            {"METALLIC_INLINE_COMPUTE_ROOT", "1"}, {"METALLIC_TRUSTED_SCENE_RESOURCE_BYTES", "408"}};
        SlangShaderDesc source{.moduleName = "TrustedSceneResourceProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = validDefines};
        ResourceComputeKernelDesc layout{.requiresRayQuery = false,
            .resourceParameterSize = sizeof(SceneResourceParameters), .rootTransport = ParameterTransport::InlinePush};
        if (!validateTrustedSceneResourceContract(source, layout, log)) { return fail("Valid trusted contract: " + log); }
        auto wrongLayout = layout;
        wrongLayout.resourceParameterSize -= 4;
        if (validateTrustedSceneResourceContract(source, wrongLayout, log)) { return fail("Short host layout accepted"); }
        wrongLayout = layout;
        wrongLayout.rootTransport = ParameterTransport::DescriptorBuffer;
        if (validateTrustedSceneResourceContract(source, wrongLayout, log)) { return fail("Buffered trusted root accepted"); }
        for (const char* value : {"388", "396", "400", "-1", "392junk", ""}) {
            const SlangMacroDefine badDefines[] = {
                {"METALLIC_INLINE_COMPUTE_ROOT", "1"}, {"METALLIC_TRUSTED_SCENE_RESOURCE_BYTES", value}};
            source.macroDefines = badDefines;
            if (validateTrustedSceneResourceContract(source, layout, log)) { return fail("Invalid trusted byte declaration accepted"); }
        }
        const SlangMacroDefine duplicate[] = {validDefines[0], validDefines[1], validDefines[1]};
        source.macroDefines = duplicate;
        if (validateTrustedSceneResourceContract(source, layout, log)) { return fail("Duplicate trusted declaration accepted"); }
        source.macroDefines = std::span<const SlangMacroDefine>(validDefines + 1, 1);
        if (validateTrustedSceneResourceContract(source, layout, log)) { return fail("Missing inline shader root accepted"); }

        auto fixture = bench::createTestDevice(context, {.applicationName = "Trusted scene resources",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        if (hasError(fixture, Error::Unsupported)) { return RHITestResult::skip("Bindless descriptors unavailable"); }
        if (!fixture) { return fail("Device creation failed"); }
        auto& device = **fixture;
        {
            source.macroDefines = validDefines;
            source.descriptorHeapMode = SlangDescriptorHeapMode::Mapped;
            auto materialLayout = layout;
            materialLayout.pushConstantSize = 16;
            ComputeKernel sourceProgram;
            std::shared_ptr<const MaterialExecutableArtifact> sourceArtifact;
            if (!compileMaterialExecutable(device, source, materialLayout, sourceProgram, sourceArtifact, log) ||
                !sourceArtifact || sourceArtifact->shader.fromArtifactOverride) {
                return fail("Ordinary trusted material compilation failed: " + log);
            }
            auto artifactOverride = std::make_shared<UnmarkedSceneArtifactOverride>(sourceArtifact->shader);
            auto installed = ShaderRegistry::instance().installArtifactOverrideForTesting(artifactOverride);
            if (!installed) { return fail("Artifact override installation failed"); }
            auto replacement = ShaderRegistry::instance().getShader(source, log);
            if (!replacement || !replacement->fromArtifactOverride) {
                return fail("Registry failed to stamp artifact provenance: " + log);
            }
            ComputeKernel rejectedProgram;
            std::shared_ptr<const MaterialExecutableArtifact> rejectedArtifact;
            auto rejected = compileMaterialExecutable(device, source, materialLayout, rejectedProgram, rejectedArtifact, log);
            if (!hasError(rejected, Error::InvalidArgument) || rejectedProgram.valid() || rejectedArtifact) {
                return fail("Trusted artifact override bypassed rejection or reused the source cache");
            }
            const SlangMacroDefine checkedDefines[] = {
                {"METALLIC_INLINE_COMPUTE_ROOT", "1"}, {"METALLIC_TRUSTED_SCENE_RESOURCE_BYTES", "0"}};
            source.macroDefines = checkedDefines;
            auto checkedShader = compileSlangShaderToSpirv(source, log);
            if (!checkedShader) { return fail("Checked artifact preparation failed: " + log); }
            artifactOverride->setShader(*checkedShader);
            ComputeKernel checkedProgram;
            std::shared_ptr<const MaterialExecutableArtifact> checkedArtifact;
            if (!compileMaterialExecutable(device, source, materialLayout, checkedProgram, checkedArtifact, log) ||
                !checkedProgram.valid() || !checkedArtifact || !checkedArtifact->shader.fromArtifactOverride) {
                return fail("Checked material rejected an artifact override: " + log);
            }
        }
        auto* queue = device.getQueue(QueueType::Graphics);
        auto registry = ResourceRegistry::forDevice(device);
        if (!queue || !registry) { return fail("Queue/registry unavailable"); }
        using Readback = std::array<uint32_t, 11>;
        auto output = device.createBuffer({.size = sizeof(Readback), .structureStride = 4,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback,
            .queueAccess = QueueAccessBits::Graphics});
        if (!output) { return fail("Readback allocation failed"); }
        constexpr uint64_t kABI = 0x545255535453434eull;
        constexpr uint32_t kMaterialsWord = offsetof(SceneResourceParameters, materials) / 4;
        std::vector<uint32_t> evidence;
        for (auto mode : {SlangDescriptorHeapMode::Mapped, SlangDescriptorHeapMode::Native}) {
            for (bool trusted : {false, true}) {
                const SlangMacroDefine defines[] = {
                    {"METALLIC_INLINE_COMPUTE_ROOT", "1"},
                    {"METALLIC_TRUSTED_SCENE_RESOURCE_BYTES", trusted ? "408" : "0"}};
                source.macroDefines = defines;
                source.descriptorHeapMode = mode;
                auto shader = compileSlangShaderToSpirv(source, log);
                if (!shader) { return fail("Probe compilation failed: " + log); }
                ComputeKernel kernel;
                if (!kernel.initialize(device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<ComputeResourceParameters>(kABI, ParameterTransport::InlinePush)}, log)) {
                    return fail("Probe pipeline failed: " + log);
                }
                for (bool shortPacket : {false, true}) {
                    for (bool sentinel : {false, true}) {
                        // Both packet and constants start beyond a guard word.
                        constexpr uint32_t kPacketWords = sizeof(SceneResourceParameters) / sizeof(uint32_t);
                        std::array<uint32_t, kPacketWords + 2> words;
                        for (uint32_t i = 0; i < words.size(); ++i) { words[i] = sentinel ? UINT32_MAX : 0x12340000u + i; }
                        ParameterWriter writer(device, **registry);
                        const auto handle = writer.buffer(output->get());
                        const std::array<uint32_t, 5> constants{0xdeadbeefu, handle.index, shortPacket ? 1u : 0u, sizeof(SceneResourceParameters) - sizeof(uint32_t), 0u};
                        ComputeResourceParameters root;
                        root.resources = writer.dataSpan(words.data(), sizeof(words), 4, 4);
                        root.resources.byteOffset += 4;
                        root.resources.count = shortPacket ? kPacketWords - 1 : kPacketWords;
                        root.constants = writer.dataSpan(constants.data(), sizeof(constants), 4, 4);
                        root.constants.byteOffset += 4;
                        root.constants.count = 4;
                        auto encoded = writer.encode(root, kABI, ParameterTransport::InlinePush);
                        if (!encoded) { return fail("Parameter upload failed"); }
                        bench::GPUCommands gpu(*queue);
                        if (!gpu.initialize(device)) { return fail("Commands unavailable"); }
                        gpu.commands->hostWriteBarrier();
                        if (!kernel.dispatch(*gpu.commands, *encoded, 1) || !gpu.submitAndWait()) { return fail("Dispatch failed"); }
                        (*output)->invalidate();
                        const auto* mapped = static_cast<const uint32_t*>((*output)->map());
                        if (!mapped) { return fail("Readback mapping failed"); }
                        Readback actual;
                        std::copy_n(mapped, actual.size(), actual.begin());
                        (*output)->unmap();
                        const uint32_t material = shortPacket ? 0 : words[kMaterialsWord + 1];
                        const uint32_t tail = shortPacket ? 0 : words[kPacketWords];
                        const Readback expected{material, material, tail, material, material, tail, tail, 0, 0, 0, 0};
                        const size_t compared = shortPacket ? 3 : expected.size();
                        if (!std::equal(expected.begin(), expected.begin() + compared, actual.begin())) {
                            return fail("Checked/trusted readback mismatch: trusted=" + std::to_string(trusted) +
                                " short=" + std::to_string(shortPacket) + " sentinel=" + std::to_string(sentinel));
                        }
                        evidence.insert(evidence.end(), actual.begin(), actual.begin() + compared);
                    }
                }
            }
        }
        bench::readbackEvidence(context, "trusted-scene-resource-readback.bin", std::span<const uint32_t>(evidence));
        return RHITestResult::pass("Host contract rejects mismatch; checked short packets remain zero; trusted fields preserve offsets and sentinels on mapped/native heaps");
    }
};

METALLIC_REGISTER_RHI_TEST(TrustedSceneResourceTest);

} // namespace
} // namespace metallic::tests
