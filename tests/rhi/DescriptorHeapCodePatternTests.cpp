#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>

namespace metallic::tests {
namespace {

constexpr uint32_t kThreadCount = 128;
constexpr uint64_t kWaitTimeout = 10'000'000'000ull;
// Production failures also changed after editing only the SPIR-V generator
// word or inserting OpNop. Aggregate copying is therefore an experimental
// variable, not an established necessary cause; keep context controls separate.
constexpr std::array kPatternNames{
    "push_fields", "entry_value", "push_copy", "buffer_copy", "buffer_fields",
    "environment_sh", "environment_sh_procedural"};

struct PatternValues {
    uint32_t a, b, c, d, e, f, g, h;
};
static_assert(sizeof(PatternValues) == 32);

uint32_t weightedSum(const PatternValues& value)
{
    return value.a + 2 * value.b + 3 * value.c + 4 * value.d +
        5 * value.e + 6 * value.f + 7 * value.g + 8 * value.h;
}

bool parseNumber(std::string_view text, uint32_t& value, int base = 10)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

RhiTestResult verifyEnvironmentPartials(render::Buffer& buffer, render::Buffer& coefficients, uint32_t groups,
    bool procedural, bool integrateOnly, const std::string& description)
{
    buffer.invalidate();
    const auto* actual = static_cast<const std::array<float, 4>*>(buffer.map());
    if (actual == nullptr) { return RhiTestResult::fail("SH partials mapping failed"); }
    // The single texel covers 4 pi steradians and samples direction (1, 0, 0).
    constexpr std::array<float, 9> kBasis{
        0.2820947918f, 0, 0, 0.4886025119f, 0, 0, -0.3153915653f, 0, 0.5462742153f};
    constexpr float kFourPi = 12.566370614359172f;
    std::array<std::array<double, 4>, 9> sums{};
    double dcSum = 0;
    for (uint32_t group = 0; group < groups; ++group) {
        for (uint32_t coefficient = 0; coefficient < 9; ++coefficient) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                const float value = actual[group * 9 + coefficient][channel];
                const float expected = channel == 3 ? 0.0f :
                    float(channel + 1) * kFourPi * kBasis[coefficient];
                if (!std::isfinite(value) || (channel == 3 && value != 0.0f) ||
                    (!procedural && std::abs(value - expected) > 0.0001f)) {
                    const std::string mismatch = description + ": partial[" + std::to_string(group) +
                        "][" + std::to_string(coefficient) + "][" + std::to_string(channel) +
                        "]=" + std::to_string(value) + (procedural ? ", expected finite RGB / zero alpha" :
                            ", expected=" + std::to_string(expected));
                    buffer.unmap();
                    return RhiTestResult::fail(mismatch);
                }
                if (coefficient == 0 && channel < 3) { dcSum += value; }
                sums[coefficient][channel] += value;
            }
        }
    }
    buffer.unmap();
    if (procedural && !(dcSum > 0.0)) { return RhiTestResult::fail(description + ": nonpositive SH DC sum"); }
    if (integrateOnly) {
        return RhiTestResult::pass(description + ", verified " + std::to_string(groups * 9) +
            " SH partial float4 values");
    }
    coefficients.invalidate();
    const auto* finalized = static_cast<const std::array<float, 4>*>(coefficients.map());
    if (finalized == nullptr) { return RhiTestResult::fail("SH coefficients mapping failed"); }
    constexpr double kPi = 3.14159265358979323846;
    for (uint32_t coefficient = 0; coefficient < 9; ++coefficient) {
        const double convolution = coefficient == 0 ? kPi : (coefficient < 4 ? 2.0 * kPi / 3.0 : kPi / 4.0);
        for (uint32_t channel = 0; channel < 4; ++channel) {
            const float value = finalized[coefficient][channel];
            const double expected = sums[coefficient][channel] * convolution;
            if (!std::isfinite(value) || std::abs(double(value) - expected) > 0.0001 * (1.0 + std::abs(expected))) {
                const std::string mismatch = description + ": coefficient[" + std::to_string(coefficient) +
                    "][" + std::to_string(channel) + "]=" + std::to_string(value) +
                    ", expected=" + std::to_string(expected);
                coefficients.unmap();
                return RhiTestResult::fail(mismatch);
            }
        }
    }
    coefficients.unmap();
    return RhiTestResult::pass(description + ", verified " + std::to_string(groups * 9) +
        " SH partial float4 values and 9 finalized coefficients");
}

// Construct after resources so both successful and failed submissions are
// drained before the allocations and ComputeProgram can be destroyed.
struct PatternCommands {
    render::RenderFrameContext frame;
    render::QueueSubmissionTracker tracker;
    std::unique_ptr<render::CommandPool> pool;
    std::unique_ptr<render::CommandBuffer> buffer;

    ~PatternCommands()
    {
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        if (pool != nullptr) { (void)pool->reset(); }
        (void)frame.reset();
    }
};

#define PATTERN_REQUIRE(expression) do { \
    const render::Result patternResult = (expression); \
    if (!patternResult) { return RhiTestResult::fail(std::string(#expression) + ": " + toString(patternResult)); } \
} while (false)

class DescriptorHeapCodePatternTest final : public RhiTest {
public:
    DescriptorHeapCodePatternTest()
    {
        type = RhiTestType::Command;
        name = "descriptor_heap_code_pattern";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        // Shader objects are mandatory even for this ordinary compute pipeline.
        // GPU execution using an existing driver cache requires explicit opt-in.
        uint32_t pattern = 0;
        uint32_t groups = 1;
        if (const char* requested = std::getenv("METALLIC_GPU_PATTERN")) {
            const std::string_view text(requested);
            bool found = false;
            for (uint32_t i = 0; i < kPatternNames.size(); ++i) {
                if (text == kPatternNames[i]) { pattern = i; found = true; break; }
            }
            if (!found && (!parseNumber(text, pattern) || pattern >= kPatternNames.size())) {
                return RhiTestResult::fail("METALLIC_GPU_PATTERN: use 0..6 or push_fields, entry_value, push_copy, buffer_copy, buffer_fields, environment_sh, environment_sh_procedural");
            }
        }
        if (const char* requested = std::getenv("METALLIC_GPU_PATTERN_GROUPS")) {
            if (!parseNumber(requested, groups) || groups == 0 || groups > 65535) {
                return RhiTestResult::fail("METALLIC_GPU_PATTERN_GROUPS must be in [1, 65535]");
            }
        }
        const bool environment = pattern >= 5;
        const bool procedural = pattern == 6;
        // Production SH fixtures deliberately use fixed image/partial counts.
        if (environment) { groups = procedural ? 256u : 1u; }
        std::string_view tableMode = "shared";
        if (environment) {
            if (const char* requested = std::getenv("METALLIC_GPU_PATTERN_TABLES")) { tableMode = requested; }
            if (tableMode != "shared" && tableMode != "separate" && tableMode != "constant") {
                return RhiTestResult::fail("METALLIC_GPU_PATTERN_TABLES must be shared, separate or constant");
            }
        }
        const uint32_t descriptorSetCount = tableMode == "constant" ? 1u : 2u;
        const uint32_t finalizeTable = tableMode == "shared" ? 1u : 0u;
        const char* pdfValue = std::getenv("METALLIC_GPU_PATTERN_PREFIX_PDF");
        const bool prefixPdf = environment && pdfValue != nullptr && std::strcmp(pdfValue, "1") == 0;
        const char* integrateOnlyValue = std::getenv("METALLIC_GPU_PATTERN_INTEGRATE_ONLY");
        const bool integrateOnly = environment && integrateOnlyValue != nullptr && std::strcmp(integrateOnlyValue, "1") == 0;
        const char* compileOnlyValue = std::getenv("METALLIC_GPU_PATTERN_COMPILE_ONLY");
        const bool compileOnly = compileOnlyValue != nullptr && std::strcmp(compileOnlyValue, "1") == 0;
        uint32_t generatorOverride = 0;
        const char* generatorValue = std::getenv("METALLIC_GPU_PATTERN_SPIRV_GENERATOR");
        if (generatorValue != nullptr) {
            std::string_view text(generatorValue);
            const bool hexadecimal = text.starts_with("0x") || text.starts_with("0X");
            if (hexadecimal) { text.remove_prefix(2); }
            if (!parseNumber(text, generatorOverride, hexadecimal ? 16 : 10)) {
                return RhiTestResult::fail("METALLIC_GPU_PATTERN_SPIRV_GENERATOR must be a uint32 decimal or 0x hexadecimal integer");
            }
        }
        const char* aftermathValue = std::getenv("METALLIC_TEST_AFTERMATH");
        const bool aftermath = aftermathValue != nullptr && std::strcmp(aftermathValue, "1") == 0;
        const char* previewValue = std::getenv("METALLIC_GPU_PATTERN_PREVIEW_DEVICE");
        const bool previewDevice = previewValue != nullptr && std::strcmp(previewValue, "1") == 0;
        // Accept old explicit-on commands, but never silently reinterpret an
        // old feature-off experiment as a required-feature device.
        if (const char* legacyShaderObject = std::getenv("METALLIC_GPU_PATTERN_SHADER_OBJECT");
            legacyShaderObject != nullptr && std::strcmp(legacyShaderObject, "1") != 0) {
            return RhiTestResult::fail("ShaderObject is required; METALLIC_GPU_PATTERN_SHADER_OBJECT may only be 1. "
                "Use the standalone Vulkan reproducer for feature-off cache experiments");
        }
        const char* internalCacheValue = std::getenv("METALLIC_VK_INTERNAL_PIPELINE_CACHE");
        const bool internalCacheDisabled = internalCacheValue != nullptr && std::strcmp(internalCacheValue, "disabled") == 0;
        const char* allowDeviceLostValue = std::getenv("METALLIC_GPU_PATTERN_ALLOW_DEVICE_LOST");
        const bool allowDeviceLost = allowDeviceLostValue != nullptr && std::strcmp(allowDeviceLostValue, "1") == 0;
        const std::string description = std::string(kPatternNames[pattern]) +
            ", groups=" + std::to_string(groups) + ", Aftermath=" + (aftermath ? "on" : "off") +
            ", device=" + (previewDevice ? "preview" : "required shader object") +
            ", compile only=" + (compileOnly ? "on" : "off") +
            (environment ? ", tables=" + std::string(tableMode) + ", prefix PDF=" + (prefixPdf ? "on" : "off") +
                ", SH=" + (integrateOnly ? "integrate only" : "integrate + finalize") : "");
        std::cout << "Descriptor heap pattern: " << description << std::endl;
        // Old shaderObject=false cache entries can produce DeviceLost when reused
        // by a shaderObject=true device. Keep those experiments outside ordinary
        // regression runs; feature-off controls now live in the standalone repro.
        if (!compileOnly && !internalCacheDisabled && !allowDeviceLost) {
            return RhiTestResult::skip("GPU cache experiment requires METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled "
                "or explicit METALLIC_GPU_PATTERN_ALLOW_DEVICE_LOST=1; compile-only needs neither");
        }
        if (!compileOnly && !internalCacheDisabled) {
            std::cout << "[expected-driver-reset experiment] Explicit reuse of a potentially incompatible driver cache; "
                "this may cause DeviceLost and temporarily hang or reset the desktop driver."
                << std::endl;
        }

        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "DescriptorHeap Code Pattern",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            // Preview mode adds its other features to the mandatory baseline.
            .enableShaderObject = true,
            .enableMeshShader = previewDevice,
            .enableTaskShader = previewDevice,
            .enableTaskShaderSubgroupBallot = previewDevice,
            .enableGeometryShader = previewDevice,
            .enableSubgroupSizeControl = previewDevice,
            .enableComputeFullSubgroups = previewDevice,
            .preferredTaskSubgroupSize = previewDevice ? 32u : 0u,
            .enableRayTracingAccelerationStructure = previewDevice,
            .enableRayQuery = previewDevice,
            .enablePushDescriptor = previewDevice,
            .enableClusterAccelerationStructure = previewDevice,
            .enableAftermath = aftermath}, device);
        if (render::hasError(result, render::Error::Unsupported)) {
            return RhiTestResult::skip("Requested device capabilities unavailable: " + description);
        }
        if (!result) { return RhiTestResult::fail("Device initialization failed: " + std::string(toString(result))); }
        auto* queue = device->getQueue(render::QueueType::Graphics);
        if (queue == nullptr) { return RhiTestResult::fail("Missing graphics queue"); }

        const std::string patternMacro = std::to_string(pattern);
        const render::SlangMacroDefine macro{"METALLIC_GPU_PATTERN", patternMacro.c_str()};
        render::ShaderCompileResult shader;
        result = render::compileSlangShaderToSpirv({
            .moduleName = environment ? "EnvironmentLightingPrecompute" : "DescriptorHeapCodePattern",
            .entryPointName = environment ? "environmentLightingPrecomputeMain" : "descriptorHeapCodePatternMain",
            .searchPath = environment ? PROJECT_SOURCE_DIR "/Shaders" : PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .macroDefines = environment ? nullptr : &macro, .macroDefineCount = environment ? 0u : 1u}, shader);
        if (!result) { return RhiTestResult::fail(shader.diagnostics); }
        if (shader.spirv.size() < 5 || shader.spirv[0] != 0x07230203u) {
            return RhiTestResult::fail("Pattern shader has an invalid SPIR-V header");
        }
        // Match the compiler cache's FNV-1a byte hash without changing its
        // contents. The experimental generator edit applies only to this copy.
        uint64_t originalHash = 14695981039346656037ull;
        const auto* originalBytes = reinterpret_cast<const unsigned char*>(shader.spirv.data());
        for (size_t i = 0; i < shader.spirv.size() * sizeof(uint32_t); ++i) {
            originalHash ^= originalBytes[i];
            originalHash *= 1099511628211ull;
        }
        const uint32_t originalGenerator = shader.spirv[2];
        if (generatorValue != nullptr) { shader.spirv[2] = generatorOverride; }
        std::cout << "Pattern SPIR-V: original generator=0x" << std::hex << originalGenerator <<
            ", effective generator=0x" << shader.spirv[2] << ", original FNV-1a64=0x" << originalHash <<
            std::dec << ", words=" << shader.spirv.size() <<
            ", generator override=" << (generatorValue != nullptr ? "on (memory only)" : "off") << std::endl;
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer},
            {.binding = 2, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
        render::ComputeProgram program;
        std::string log;
        result = program.initialize(*device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = sizeof(PatternValues),
            .bindings = bindings, .bindingCount = 3, .debugName = "DescriptorHeap Code Pattern",
            .descriptorSetCount = descriptorSetCount, .requiresRayQuery = false}, log);
        if (!result) { return RhiTestResult::fail(log); }
        if (compileOnly) {
            std::cout << "Pattern compile only: ComputeProgram initialized; no fixture resources, commands or submissions; "
                "PDF prefix skipped." << std::endl;
            return RhiTestResult::pass(description + ", pipeline initialization completed without GPU submission");
        }

        const PatternValues push = environment ?
            PatternValues{0, procedural ? 256u : 1u, procedural ? 128u : 1u, groups, groups, procedural ? 1u : 0u, 0, 0} :
            PatternValues{2, 3, 5, 7, 11, 13, 17, 19};
        constexpr PatternValues kInput{23, 29, 31, 37, 41, 43, 47, 53};
        constexpr std::array<float, 4> kTexel{1, 2, 3, 4};
        const uint32_t elementCount = groups * kThreadCount;
        const uint64_t inputBytes = environment ? uint64_t(groups) * 9 * sizeof(kTexel) : sizeof(kInput);
        const uint64_t outputBytes = environment ? 9 * sizeof(kTexel) : uint64_t(elementCount) * sizeof(uint32_t);
        std::unique_ptr<render::Buffer> input, output, upload;
        std::unique_ptr<render::Texture> texture;
        std::unique_ptr<render::TextureView> view;
        PATTERN_REQUIRE(device->createBuffer({.size = inputBytes,
            .structureStride = environment ? uint32_t(sizeof(kTexel)) : uint32_t(sizeof(kInput)),
            .usage = render::BufferUsageBits::Storage,
            .memoryLocation = environment ? render::MemoryLocation::HostReadback : render::MemoryLocation::HostUpload}, input));
        PATTERN_REQUIRE(device->createBuffer({.size = outputBytes,
            .structureStride = environment ? uint32_t(sizeof(kTexel)) : uint32_t(sizeof(uint32_t)),
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}, output));
        PATTERN_REQUIRE(device->createBuffer({.size = sizeof(kTexel),
            .usage = render::BufferUsageBits::TransferSource, .memoryLocation = render::MemoryLocation::HostUpload}, upload));
        void* mapped = input->map();
        if (mapped == nullptr) { return RhiTestResult::fail("Input mapping failed"); }
        if (environment) { std::memset(mapped, 0, static_cast<size_t>(inputBytes)); }
        else { std::memcpy(mapped, &kInput, sizeof(kInput)); }
        input->flush();
        input->unmap();
        mapped = upload->map();
        if (mapped == nullptr) { return RhiTestResult::fail("Texture upload mapping failed"); }
        std::memcpy(mapped, kTexel.data(), sizeof(kTexel));
        upload->flush();
        upload->unmap();
        PATTERN_REQUIRE(device->createTexture({
            .usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferDestination,
            .format = render::Format::Rgba32Sfloat, .width = 1, .height = 1}, texture));
        PATTERN_REQUIRE(device->createTextureView(*texture, {.format = render::Format::Rgba32Sfloat}, view));

        // These objects outlive PatternCommands; production PDF code also
        // retains its descriptor tables and allocations through the frame.
        render::ImportancePdfCompute pdfCompute;
        render::ImportancePdfTexture pdfTexture;
        if (prefixPdf) {
            result = pdfCompute.initialize(*device, log);
            if (result) { result = pdfTexture.initialize(*device, 1, 1, "DescriptorHeap Pattern PDF", log); }
            if (!result) { return RhiTestResult::fail("PDF prefix initialization: " + log); }
        }

        PatternCommands commands;
        PATTERN_REQUIRE(commands.tracker.initialize(*device, *queue));
        PATTERN_REQUIRE(device->createCommandPool(*queue, commands.pool));
        PATTERN_REQUIRE(commands.pool->createCommandBuffer(commands.buffer));
        PATTERN_REQUIRE(commands.frame.begin(0));
        PATTERN_REQUIRE(commands.buffer->begin(&commands.frame));
        const render::TextureBarrierDesc toTransfer{.texture = texture.get(),
            .before = render::ResourceState::Undefined, .after = render::ResourceState::TransferDestination};
        const render::BufferBarrierDesc toGeneral[] = {
            {.buffer = input.get(), .before = render::ResourceState::Undefined, .after = render::ResourceState::General},
            {.buffer = output.get(), .before = render::ResourceState::Undefined, .after = render::ResourceState::General}};
        commands.buffer->barrier({.textures = &toTransfer, .textureCount = 1, .buffers = toGeneral, .bufferCount = 2});
        commands.buffer->copyBufferToTexture({.buffer = upload.get(), .texture = texture.get(), .width = 1, .height = 1});
        const render::TextureBarrierDesc toRead{.texture = texture.get(),
            .before = render::ResourceState::TransferDestination, .after = render::ResourceState::ShaderRead};
        commands.buffer->barrier({.textures = &toRead, .textureCount = 1});
        if (prefixPdf) {
            PATTERN_REQUIRE(pdfCompute.buildEnvironment(*commands.buffer, *view, pdfTexture));
        }
        auto* sampledView = view.get();
        const render::ComputeDispatchBinding resources[] = {
            {.binding = 0, .textureViews = &sampledView, .textureViewCount = 1},
            {.binding = 1, .buffer = input.get()},
            {.binding = 2, .buffer = output.get()}};
        PATTERN_REQUIRE(program.dispatch({.commandBuffer = commands.buffer.get(),
            .bindings = resources, .bindingCount = 3, .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = groups}));
        if (environment && !integrateOnly) {
            const render::BufferBarrierDesc partialsBarrier{.buffer = input.get(),
                .before = render::ResourceState::General, .after = render::ResourceState::General,
                .size = inputBytes};
            commands.buffer->barrier({.buffers = &partialsBarrier, .bufferCount = 1});
            PatternValues finalizePush = push;
            finalizePush.a = 1;
            PATTERN_REQUIRE(program.dispatch({.commandBuffer = commands.buffer.get(),
                .bindings = resources, .bindingCount = 3,
                .pushData = &finalizePush, .pushDataSize = sizeof(finalizePush),
                .groupCountX = 9, .descriptorSetIndex = finalizeTable}));
        }
        PATTERN_REQUIRE(commands.buffer->end());
        render::CommandBuffer* submitted[] = {commands.buffer.get()};
        PATTERN_REQUIRE(commands.tracker.submit({.commandBuffers = submitted, .commandBufferCount = 1}, commands.frame));
        PATTERN_REQUIRE(commands.frame.wait(kWaitTimeout));

        if (environment) { return verifyEnvironmentPartials(*input, *output, groups, procedural, integrateOnly, description); }

        output->invalidate();
        const auto* actual = static_cast<const uint32_t*>(output->map());
        if (actual == nullptr) { return RhiTestResult::fail("Output mapping failed"); }
        const uint32_t base = weightedSum(push) + weightedSum(kInput) + 30;
        for (uint32_t i = 0; i < elementCount; ++i) {
            if (actual[i] != base + i) {
                const std::string mismatch = description + ": output[" + std::to_string(i) +
                    "]=" + std::to_string(actual[i]) + ", expected=" + std::to_string(base + i);
                output->unmap();
                return RhiTestResult::fail(mismatch);
            }
        }
        output->unmap();
        return RhiTestResult::pass(description + ", verified " + std::to_string(elementCount) + " words");
    }
};

METALLIC_REGISTER_RHI_TEST(DescriptorHeapCodePatternTest);

#undef PATTERN_REQUIRE

} // namespace
} // namespace metallic::tests
