#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <array>
#include <bit>
#include <cstring>

namespace metallic::tests {
namespace {

// Independent CPU expectations for Slang scalar-layout records. Context is a
// transient shader value in production; this test deliberately transports it.
struct ProbeContext { std::array<uint32_t, 32> words; };
struct ProbeInstance { uint32_t parameterOffset, resourceOffset; };
struct ProbeResult { std::array<uint32_t, 16> words; };
struct ProbeParams { render::GPUBufferSpan parameters, resources, instances, contexts, contextOutput, resultOutput; };
static_assert(sizeof(ProbeContext) == 128 && sizeof(ProbeInstance) == 8 && sizeof(ProbeResult) == 64);
static_assert(sizeof(ProbeParams) == 72 && offsetof(ProbeParams, resultOutput) == 60);
constexpr uint64_t kProbeABI = 0x4d50524f47000001ull;

#define PROGRAM_REQUIRE(expression) do { \
    const render::Result<> result = (expression); \
    if (!result) { return RHITestResult::fail(std::string(#expression) + ": " + toString(result) + " " + log); } \
} while (false)

class MaterialProgramContractTest final : public RHITest
{
public:
    MaterialProgramContractTest()
    {
        name = "material_program_shader_contract";
        type = RHITestType::Resource;
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"material.program.context", "material.program.instance_offsets", "material.program.shared_code"},
            bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::string log;
        ShaderCompileResult shader;
        PROGRAM_REQUIRE(compileSlangShaderToSpirv({.moduleName = "MaterialProgramProbe",
            .entryPointName = "materialProgramProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
            {.enableDiskCache = false}, log).transform([&](auto value) { shader = std::move(value); }));
        bool hasContext = false;
        for (const auto& dependency : shader.dependencies) {
            if (dependency.find("SurfaceMaterialContext.slang") != std::string::npos) { hasContext = true; }
            if (dependency.find("OpenPBR") != std::string::npos || dependency.find("SceneMaterial.slang") != std::string::npos ||
                dependency.find("Lighting") != std::string::npos || dependency.find("SceneSurface") != std::string::npos) {
                return RHITestResult::fail("Model-dependent shader contract: " + dependency);
            }
        }
        if (!hasContext) { return RHITestResult::fail("Standalone context dependency was not compiled"); }
        std::atomic_uint validationErrors{0};
        bench::TestDevice device;
        PROGRAM_REQUIRE(bench::createTestDevice(context, {.applicationName = "Material program contract",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                if ((message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
                    (message.type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) {
                    ++*static_cast<std::atomic_uint*>(target);
                }
            }, &validationErrors}}).transform([&](auto value) { device = std::move(value); }));
        ResourceRegistry registry;
        PROGRAM_REQUIRE(registry.initialize(*device));
        ComputeKernel kernel;
        PROGRAM_REQUIRE(kernel.initialize(*device, {.spirv = shader.spirv,
            .parameters = parameterAbi<ProbeParams>(kProbeABI, ParameterTransport::InlinePush)}, log));
        std::array<float, 4> parameters{0.25f, 0.5f, 0.75f, 1.0f};
        const std::array<float, 4> resources{2.0f, 4.0f, 8.0f, 16.0f};
        // Independent resource/parameter offsets, repeated and nonmonotonic refs.
        const std::array<ProbeInstance, 8> instances{{{12, 4}, {0, 12}, {8, 0}, {4, 8}, {12, 4}, {8, 12}, {0, 8}, {4, 0}}};
        std::array<ProbeContext, 2> contexts{};
        for (uint32_t index = 0; index < contexts.size(); ++index) {
            for (uint32_t word = 0; word < 29; ++word) {
                contexts[index].words[word] = std::bit_cast<uint32_t>(float(word + 1) * (index == 0 ? 0.125f : -0.25f));
            }
            // A back face keeps the same authored TBN signs. One producer has
            // UV derivatives, the other an already projected ray-cone footprint.
            contexts[index].words[29] = index == 0 ? 3u : 12u;
            contexts[index].words[30] = 3u;
            contexts[index].words[31] = index;
        }
        std::unique_ptr<Buffer> parameterBuffer, resourceBuffer, instanceBuffer, contextBuffer, contextOutput, resultOutput;
        const auto buffer = [&](size_t size, MemoryLocation location, std::unique_ptr<Buffer>& output) {
            return device->createBuffer({.size = size, .usage = BufferUsageBits::Storage, .memoryLocation = location})
                .transform([&](auto value) { output = std::move(value); });
        };
        PROGRAM_REQUIRE(buffer(sizeof(parameters), MemoryLocation::HostUpload, parameterBuffer));
        PROGRAM_REQUIRE(buffer(sizeof(resources), MemoryLocation::HostUpload, resourceBuffer));
        PROGRAM_REQUIRE(buffer(sizeof(instances), MemoryLocation::HostUpload, instanceBuffer));
        PROGRAM_REQUIRE(buffer(sizeof(contexts), MemoryLocation::HostUpload, contextBuffer));
        PROGRAM_REQUIRE(buffer(8 * sizeof(ProbeContext), MemoryLocation::HostReadback, contextOutput));
        PROGRAM_REQUIRE(buffer(8 * sizeof(ProbeResult), MemoryLocation::HostReadback, resultOutput));
        const auto upload = [](Buffer& buffer, const auto& values) {
            auto* mapped = buffer.map();
            if (!mapped) { return false; }
            std::memcpy(mapped, &values, sizeof(values));
            buffer.flush(); buffer.unmap();
            return true;
        };
        if (!upload(*resourceBuffer, resources) || !upload(*instanceBuffer, instances) || !upload(*contextBuffer, contexts)) {
            return RHITestResult::fail("Probe input upload failed");
        }
        // One compiled kernel, two dispatches. Only instance data changes.
        for (uint32_t iteration = 0; iteration < 2; ++iteration) {
            if (iteration == 1) { parameters[3] = 0.125f; }
            if (!upload(*parameterBuffer, parameters)) { return RHITestResult::fail("Parameter update failed"); }
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            PROGRAM_REQUIRE(gpu.initialize(*device));
            ParameterWriter writer(*device, registry);
            ProbeParams params{writer.bufferSpan<float>(parameterBuffer.get()), writer.bufferSpan<float>(resourceBuffer.get()),
                writer.bufferSpan<ProbeInstance>(instanceBuffer.get()), writer.bufferSpan<ProbeContext>(contextBuffer.get()),
                writer.bufferSpan<ProbeContext>(contextOutput.get()), writer.bufferSpan<ProbeResult>(resultOutput.get())};
            auto encoded = writer.encode(params, kProbeABI, ParameterTransport::InlinePush);
            if (!encoded) { return RHITestResult::fail("Probe parameters failed"); }
            PROGRAM_REQUIRE(kernel.dispatch(*gpu.commands, *encoded, 1));
            PROGRAM_REQUIRE(gpu.submitAndWait());
            contextOutput->invalidate(); resultOutput->invalidate();
            const auto* actualContexts = static_cast<const ProbeContext*>(contextOutput->map());
            const auto* actualResults = static_cast<const ProbeResult*>(resultOutput->map());
            if (!actualContexts || !actualResults) { return RHITestResult::fail("Probe readback failed"); }
            bool equal = true;
            for (uint32_t lane = 0; lane < 8; ++lane) {
                const auto& source = contexts[lane & 1u].words;
                const auto& instance = instances[lane];
                const auto p = std::bit_cast<uint32_t>(parameters[instance.parameterOffset / 4]);
                const auto r = std::bit_cast<uint32_t>(resources[instance.resourceOffset / 4]);
                const auto c = std::bit_cast<uint32_t>(std::bit_cast<float>(source[0]) + std::bit_cast<float>(source[17]));
                const auto quarter = std::bit_cast<uint32_t>(0.25f);
                const std::array<uint32_t, 16> expected{p, r, c, quarter, 5u, source[6], source[7], source[8],
                    p, r, c, quarter, std::bit_cast<uint32_t>(1.0f), 5u, instance.parameterOffset, instance.resourceOffset};
                equal &= actualContexts[lane].words == source && actualResults[lane].words == expected;
            }
            bench::readbackEvidence(context, "readback.bin", std::span(actualResults, 8));
            contextOutput->unmap(); resultOutput->unmap();
            if (!equal) { return RHITestResult::fail("Context, offset, BSDF ABI or shared-program data update mismatch"); }
        }
        if (validationErrors != 0) { return RHITestResult::fail("Material contract caused Vulkan validation errors"); }
        return RHITestResult::pass("Independent model-free module; 8 refs and 2 data updates share one statically specialized GPU kernel");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialProgramContractTest);

class MaterialProgramGuideCompileTest final : public RHITest
{
public:
    MaterialProgramGuideCompileTest()
    {
        name = "material_program_guide_compile";
        type = RHITestType::Resource;
    }
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        std::string log;
        for (const bool positionFetch : {false, true}) {
            const SceneShaderOptions options{.hasRTXCR = METALLIC_HAS_RTXCR != 0, .positionFetch = positionFetch,
                .rtxcrInclude = METALLIC_RTXCR_SHADER_INCLUDE_DIR};
            for (const auto program : {SceneShaderProgram::PathTraceGuides, SceneShaderProgram::OpenPBRPathTraceGuides}) {
                const auto request = makeSceneShaderRequest(program, options);
                const ShaderRequestView view(request);
                PROGRAM_REQUIRE(compileSlangShaderToSpirv(view.desc(), log).transform([](auto) {}));
            }
        }
        return RHITestResult::pass("Standard/OpenPBR guide entry points compile with hardware and fallback position fetch; compile-only");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialProgramGuideCompileTest);

#undef PROGRAM_REQUIRE
} // namespace
} // namespace metallic::tests
