#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/Hash.h"
#include "Runtime/Render/GAPI/PipelineCacheFile.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <array>
#include <chrono>
#include <fstream>
#include <string_view>

namespace metallic::tests {
namespace {

using namespace render;
using ShaderCodes = std::array<ShaderCompileResult, 3>;
using ShaderModules = std::array<std::unique_ptr<ShaderModule>, 3>;

Result<ShaderModules> createModules(Device& device, const ShaderCodes& codes)
{
    ShaderModules modules;
    for (size_t index = 0; index < modules.size(); ++index) {
        auto module = ShaderRegistry::instance().getShaderModule(device, {.spirv = codes[index].spirv});
        if (!module) { return std::unexpected(module.error()); }
        modules[index] = std::move(*module);
    }
    return modules;
}

Result<std::vector<uint8_t>> exportEmptyNativeCache(Device& device)
{
    const auto native = vulkan::nativeDevice(device);
    VkPipelineCache cache = VK_NULL_HANDLE;
    const VkPipelineCacheCreateInfo info{.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (native.functions->vkCreatePipelineCache(native.device, &info, nullptr, &cache) != VK_SUCCESS) {
        return std::unexpected(Error::Failure);
    }
    size_t size = 0;
    VkResult result = native.functions->vkGetPipelineCacheData(native.device, cache, &size, nullptr);
    std::vector<uint8_t> bytes(size);
    if (result == VK_SUCCESS) {
        result = native.functions->vkGetPipelineCacheData(native.device, cache, &size, bytes.data());
    }
    native.functions->vkDestroyPipelineCache(native.device, cache, nullptr);
    if (result != VK_SUCCESS || size == 0) { return std::unexpected(Error::Failure); }
    bytes.resize(size);
    return bytes;
}

bool readFixtureIdentity(const std::filesystem::path& path, detail::PipelineCacheFileIdentity& identity)
{
    // Read the established .pso container identity instead of duplicating the
    // backend's device-feature compatibility policy in this fixture.
    std::ifstream stream(path, std::ios::binary);
    stream.seekg(12);
    stream.read(reinterpret_cast<char*>(&identity.backendTag), sizeof(identity.backendTag));
    stream.seekg(48);
    stream.read(reinterpret_cast<char*>(identity.compatibilityKey.data()), identity.compatibilityKey.size());
    return static_cast<bool>(stream);
}

Result<> createExternalGraphics(Device& device, PipelineCache& cache, const ShaderModules& modules,
    VkPipelineLayout layout, bool bindless, uint64_t hash)
{
    const auto native = vulkan::nativeDevice(device);
    const VkPipelineShaderStageCreateInfo stages[]{
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vulkan::nativeShaderModule(*modules[1]), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = vulkan::nativeShaderModule(*modules[2]), .pName = "main"},
    };
    const VkPipelineVertexInputStateCreateInfo vertexInput{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    const VkPipelineInputAssemblyStateCreateInfo assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    const VkPipelineViewportStateCreateInfo viewport{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    const VkPipelineColorBlendAttachmentState attachment{.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &attachment};
    constexpr VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dynamicStates};
    constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    const VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
    VkPipelineCreationFeedback callerFeedback{.duration = UINT64_MAX};
    std::array<VkPipelineCreationFeedback, 2> callerStageFeedback{};
    const VkPipelineCreationFeedbackCreateInfo feedbackInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO, .pNext = &rendering,
        .pPipelineCreationFeedback = &callerFeedback,
        .pipelineStageCreationFeedbackCount = static_cast<uint32_t>(callerStageFeedback.size()),
        .pPipelineStageCreationFeedbacks = callerStageFeedback.data()};
    const VkPipelineCreateFlags2CreateInfo flags{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO, .pNext = &feedbackInfo,
        .flags = bindless ? VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT : 0};
    const VkGraphicsPipelineCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = bindless ? static_cast<const void*>(&flags) : static_cast<const void*>(&feedbackInfo),
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vertexInput, .pInputAssemblyState = &assembly,
        .pViewportState = &viewport, .pRasterizationState = &rasterization, .pMultisampleState = &multisample,
        .pColorBlendState = &blend, .pDynamicState = &dynamic, .layout = layout};
    const auto before = cache.stats();
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto result = vulkan::createCachedGraphicsPipeline(cache, info, hash, pipeline);
    if (pipeline != VK_NULL_HANDLE) { native.functions->vkDestroyPipeline(native.device, pipeline, nullptr); }
    if (!result) { return result; }
    const auto after = cache.stats();
    const bool valid = (callerFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) != 0;
    const bool hit = valid && (callerFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0;
    // The RHI must observe this existing output structure. Adding a duplicate
    // feedback node, discarding this chain or reading an owned empty output
    // would violate these counters and the caller's exact duration.
    if (after.nativeHitCount != before.nativeHitCount + uint64_t(hit) ||
        after.nativeMissCount != before.nativeMissCount + uint64_t(valid && !hit) ||
        after.unknownFeedbackCount != before.unknownFeedbackCount + uint64_t(!valid) ||
        after.lastDriverDurationNanoseconds != (valid ? callerFeedback.duration : 0)) {
        return makeError(Error::Failure);
    }
    return {};
}

std::string checkFeedback(const PipelineCacheStats& before, const PipelineCacheStats& after, bool firstSessionUse)
{
    if (after.hitCount != before.hitCount + 1 || after.missCount != before.missCount) {
        return "known application PSO identity was not reported independently as a hash hit";
    }
    const auto hits = after.nativeHitCount - before.nativeHitCount;
    const auto misses = after.nativeMissCount - before.nativeMissCount;
    const auto unknown = after.unknownFeedbackCount - before.unknownFeedbackCount;
    if (hits + misses + unknown != 1 || after.lastNativeCreateTimeNanoseconds == 0) {
        return "native pipeline feedback or host creation time was not recorded exactly once";
    }
    const bool changed = misses != 0 || (unknown != 0 && firstSessionUse);
    if (after.dirtyRevision != before.dirtyRevision + uint64_t(changed)) {
        return "native cache miss/first unknown did not refresh a known hash, or unchanged feedback dirtied it";
    }
    return {};
}

void writeFeedback(std::ofstream& stream, std::string_view phase, const PipelineCacheStats& stats)
{
    stream << "{\"phase\":\"" << phase << "\",\"hash_hits\":" << stats.hitCount
           << ",\"hash_misses\":" << stats.missCount << ",\"native_hits\":" << stats.nativeHitCount
           << ",\"native_misses\":" << stats.nativeMissCount << ",\"unknown\":" << stats.unknownFeedbackCount
           << ",\"create_ns\":" << stats.lastNativeCreateTimeNanoseconds
           << ",\"driver_ns\":" << stats.lastDriverDurationNanoseconds
           << ",\"lock_wait_ns\":" << stats.lastCacheLockWaitTimeNanoseconds
           << ",\"backend_bytes\":" << stats.backendDataSize << ",\"save_count\":" << stats.saveCount
           << ",\"dirty_revision\":" << stats.dirtyRevision << ",\"saved_revision\":" << stats.persistedRevision << "}\n";
}

Result<> checkGreenReadback(Device& device, Queue& queue, const GraphicsPipeline& pipeline)
{
#define FEEDBACK_REQUIRE(expression) do { const auto& checked = (expression); \
    if (!checked) { return std::unexpected(checked.error()); } } while (false)
    constexpr uint32_t kWidth = 16;
    constexpr uint32_t kHeight = 16;
    auto texture = device.createTexture({.usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
        .format = Format::RGBA8Unorm, .width = kWidth, .height = kHeight});
    FEEDBACK_REQUIRE(texture);
    auto view = device.createTextureView(**texture, {});
    auto readback = device.createBuffer({.size = kWidth * kHeight * 4, .usage = BufferUsageBits::TransferDestination,
        .memoryLocation = MemoryLocation::HostReadback});
    FEEDBACK_REQUIRE(view);
    FEEDBACK_REQUIRE(readback);
    bench::GPUCommands recording(queue);
    FEEDBACK_REQUIRE(recording.initialize(device));
    auto& commands = *recording.commands;
    TextureBarrierDesc barrier{.texture = texture->get(), .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::ColorAttachment, .before = {},
        .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorWrite}, .range = {0, 1, 0, 1}};
    FEEDBACK_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
    RenderingAttachmentDesc attachment{.view = view->get(), .layout = TextureLayout::ColorAttachment,
        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearColor = {0, 0, 0, 1}};
    FEEDBACK_REQUIRE(commands.beginRendering({.renderArea = {0, 0, kWidth, kHeight}, .colorAttachments = {&attachment, 1}}));
    FEEDBACK_REQUIRE(commands.bindExecution(pipeline.execution()));
    FEEDBACK_REQUIRE(commands.setViewport({0, 0, float(kWidth), float(kHeight), 0, 1}));
    commands.setScissor({0, 0, kWidth, kHeight});
    FEEDBACK_REQUIRE(commands.draw(3));
    commands.endRendering();
    barrier.oldLayout = TextureLayout::ColorAttachment;
    barrier.newLayout = TextureLayout::TransferSource;
    barrier.before = barrier.after;
    barrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead};
    FEEDBACK_REQUIRE(commands.synchronize({.textures = {&barrier, 1}}));
    auto slice = (*readback)->slice();
    FEEDBACK_REQUIRE(slice);
    FEEDBACK_REQUIRE(commands.copyTextureToBuffer({.texture = texture->get(), .buffer = *slice, .width = kWidth, .height = kHeight}));
    FEEDBACK_REQUIRE(recording.submitAndWait());
    const auto* mapped = static_cast<const uint8_t*>((*readback)->map());
    if (!mapped) { return std::unexpected(Error::Failure); }
    (*readback)->invalidate();
    bool correct = true;
    for (size_t index = 0; index < kWidth * kHeight * 4; ++index) {
        if (mapped[index] != uint8_t(index % 4 == 1 || index % 4 == 3 ? 255 : 0)) { correct = false; break; }
    }
    (*readback)->unmap();
    return correct ? Result<>{} : makeError(Error::Failure);
#undef FEEDBACK_REQUIRE
}

class PipelineCacheNativeFeedbackTest final : public RHITest {
public:
    PipelineCacheNativeFeedbackTest()
    {
        type = RHITestType::Rendering;
        name = "pipeline_cache_native_feedback_and_refresh";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto root = std::filesystem::absolute(context.outputDirectory / "pipeline-cache-feedback" /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        {
            std::ofstream source(root / "NativeFeedbackProbe.slang");
            source << "[shader(\"compute\")] [numthreads(37, 1, 1)] void computeMain(uint3 id : SV_DispatchThreadID) {}\n"
                      "[shader(\"vertex\")] float4 vertexMain(uint id : SV_VertexID) : SV_Position {\n"
                      "return float4(float(id & 1u) * 4.0 - 1.0, float((id >> 1u) & 1u) * 4.0 - 1.0, 0.0, 1.0); }\n"
                      "[shader(\"fragment\")] float4 fragmentMain() : SV_Target { return float4(0, 1, 0, 1); }\n";
            if (!source) { return RHITestResult::fail("could not write native-feedback shader fixture"); }
        }
        ShaderCodes codes;
        const char* entries[]{"computeMain", "vertexMain", "fragmentMain"};
        const std::string sourceRoot = root.string();
        const std::string spirvRoot = (root / "spirv").string();
        std::string log;
        for (size_t index = 0; index < codes.size(); ++index) {
            auto code = ShaderRegistry::instance().getShader({.moduleName = "NativeFeedbackProbe",
                .entryPointName = entries[index], .searchPath = sourceRoot.c_str()}, {.cacheDirectory = spirvRoot.c_str()}, log);
            if (!code) { return RHITestResult::fail("native-feedback source compilation failed: " + log); }
            codes[index] = std::move(*code);
        }
        auto modules = createModules(context.device, codes);
        auto emptyBackend = exportEmptyNativeCache(context.device);
        if (!modules || !emptyBackend) { return RHITestResult::fail("native-feedback modules/empty backend creation failed"); }
        const bool bindless = context.device.capabilities().bindlessDescriptorHeap;
        const auto createCompute = [&](Device& device, const ShaderModules& shaders, PipelineCache& cache) {
            return device.createComputePipeline({.computeShader = {shaders[0].get()}, .usesBindlessHeap = bindless, .pipelineCache = &cache});
        };
        const auto createGraphics = [&](Device& device, const ShaderModules& shaders, PipelineCache& cache) {
            return device.createGraphicsPipeline({.vertexShader = {shaders[1].get()}, .fragmentShader = {shaders[2].get()},
                .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1, .usesBindlessHeap = bindless, .pipelineCache = &cache});
        };

        const std::string seedPath = (root / "seed.pso").string();
        auto seed = context.device.createPipelineCache({.filePath = seedPath.c_str(), .saveOnDestroy = false});
        if (!seed) { return RHITestResult::fail("could not create native-feedback seed cache"); }
        auto seedCompute = createCompute(context.device, *modules, **seed);
        auto seedGraphics = createGraphics(context.device, *modules, **seed);
        if (!seedCompute || !seedGraphics) { return RHITestResult::fail("native-feedback seed pipeline creation failed"); }
        const auto layout = vulkan::nativePipeline(**seedGraphics).layout;
        const auto externalHash = detail::hashValue((*seedGraphics)->psoHash(), uint64_t{0x464545444241434b});
        if (!createExternalGraphics(context.device, **seed, *modules, layout, bindless, externalHash) || !(*seed)->save()) {
            return RHITestResult::fail("native-feedback seed external pipeline/save failed");
        }
        detail::PipelineCacheFileIdentity identity;
        if (!readFixtureIdentity(seedPath, identity)) { return RHITestResult::fail("could not inspect seed .pso identity"); }
        const std::array knownHashes{(*seedCompute)->psoHash(), (*seedGraphics)->psoHash(), externalHash};
        const auto refreshPath = root / "hashes-with-empty-backend.pso";
        std::string reason;
        if (!detail::savePipelineCacheFile(refreshPath, identity, knownHashes, *emptyBackend, reason)) {
            return RHITestResult::fail("could not create known-hash/empty-native fixture: " + reason);
        }
        const std::string refreshPathString = refreshPath.string();
        auto cache = context.device.createPipelineCache({.filePath = refreshPathString.c_str(), .saveOnDestroy = false});
        if (!cache || (*cache)->stats().loadStatus != PipelineCacheLoadStatus::Loaded ||
            (*cache)->stats().storedPsoCount != knownHashes.size() || (*cache)->stats().dirtyRevision != 0) {
            return RHITestResult::fail("known-hash/empty-native fixture did not load as a clean compatible cache");
        }
        std::ofstream metrics(root / "feedback.jsonl");
        writeFeedback(metrics, "loaded-empty-native", (*cache)->stats());
        auto before = (*cache)->stats();
        auto compute = createCompute(context.device, *modules, **cache);
        auto after = (*cache)->stats();
        if (!compute || !(*compute)->pipelineCacheHit() || (*compute)->psoHash() != knownHashes[0]) {
            return RHITestResult::fail("empty native cache lost the known compute hash identity");
        }
        if (const auto error = checkFeedback(before, after, true); !error.empty()) { return RHITestResult::fail("compute: " + error); }
        writeFeedback(metrics, "empty-compute", after);
        before = after;
        auto graphics = createGraphics(context.device, *modules, **cache);
        after = (*cache)->stats();
        if (!graphics || !(*graphics)->pipelineCacheHit() || (*graphics)->psoHash() != knownHashes[1]) {
            return RHITestResult::fail("empty native cache lost the known graphics hash identity");
        }
        if (const auto error = checkFeedback(before, after, true); !error.empty()) { return RHITestResult::fail("graphics: " + error); }
        writeFeedback(metrics, "empty-graphics", after);
        before = after;
        if (!createExternalGraphics(context.device, **cache, *modules, layout, bindless, externalHash)) {
            return RHITestResult::fail("external graphics discarded its caller feedback/rendering/flags2 pNext chain or did not consume the caller's output");
        }
        after = (*cache)->stats();
        if (const auto error = checkFeedback(before, after, true); !error.empty()) { return RHITestResult::fail("external graphics: " + error); }
        writeFeedback(metrics, "empty-external-graphics", after);
        if (!checkGreenReadback(context.device, context.graphicsQueue, **graphics)) {
            return RHITestResult::fail("graphics feedback pipeline did not produce the expected green readback");
        }
        const bool observedNativeMiss = after.nativeMissCount != 0;
        const auto pendingRevision = after.dirtyRevision;
        if (!(*cache)->save() || (*cache)->stats().persistedRevision != pendingRevision ||
            (*cache)->stats().saveCount != uint64_t(pendingRevision != 0)) {
            return RHITestResult::fail("known metadata hash suppressed persistence of refreshed native cache data");
        }
        writeFeedback(metrics, "saved-refresh", (*cache)->stats());
        before = (*cache)->stats();
        auto repeated = createCompute(context.device, *modules, **cache);
        after = (*cache)->stats();
        if (!repeated) { return RHITestResult::fail("repeated compute creation failed"); }
        if (const auto error = checkFeedback(before, after, false); !error.empty()) { return RHITestResult::fail("repeated compute: " + error); }
        writeFeedback(metrics, "repeat-compute", after);
        if (!(*cache)->save()) { return RHITestResult::fail("could not persist repeated native miss if present"); }
        detail::PipelineCacheFileData savedData;
        if (detail::loadPipelineCacheFile(refreshPath, identity, savedData, reason) != detail::PipelineCacheFileLoadStatus::Loaded ||
            savedData.psoHashes.size() != knownHashes.size() || savedData.backendData.empty()) {
            return RHITestResult::fail("refreshed cache did not retain valid native bytes and all known metadata identities");
        }

        auto otherDevice = bench::createTestDevice(context, {.applicationName = "Native feedback reload",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = bindless}, true);
        if (!otherDevice) { return RHITestResult::fail("could not create independent native-feedback reload device"); }
        auto otherModules = createModules(**otherDevice, codes);
        auto loaded = (*otherDevice)->createPipelineCache({.filePath = refreshPathString.c_str(), .saveOnDestroy = false});
        if (!otherModules || !loaded || (*loaded)->stats().loadStatus != PipelineCacheLoadStatus::Loaded ||
            (*loaded)->stats().storedPsoCount != knownHashes.size()) {
            return RHITestResult::fail("independent device did not reload the refreshed native cache and metadata identities");
        }
        before = (*loaded)->stats();
        auto loadedCompute = createCompute(**otherDevice, *otherModules, **loaded);
        after = (*loaded)->stats();
        if (!loadedCompute || !(*loadedCompute)->pipelineCacheHit() || (*loadedCompute)->psoHash() != knownHashes[0]) {
            return RHITestResult::fail("independent device changed the persisted compute pipeline identity");
        }
        if (const auto error = checkFeedback(before, after, true); !error.empty()) { return RHITestResult::fail("reload compute: " + error); }
        writeFeedback(metrics, "reload-compute", after);
        before = after;
        auto loadedGraphics = createGraphics(**otherDevice, *otherModules, **loaded);
        after = (*loaded)->stats();
        if (!loadedGraphics || !(*loadedGraphics)->pipelineCacheHit() || (*loadedGraphics)->psoHash() != knownHashes[1]) {
            return RHITestResult::fail("independent device changed the persisted graphics pipeline identity");
        }
        if (const auto error = checkFeedback(before, after, true); !error.empty()) { return RHITestResult::fail("reload graphics: " + error); }
        writeFeedback(metrics, "reload-graphics", after);
        if (!metrics) { return RHITestResult::fail("could not write native-feedback timing evidence"); }
        if (!observedNativeMiss) {
            return RHITestResult::skip("driver reported no valid native miss for the empty backend; hash accounting, feedback fallback, pNext chain and readback checked; " + root.string());
        }
        return RHITestResult::pass("known app hashes with an empty native cache refreshed on driver misses; compute, graphics, external pNext chain, green readback and independent reload checked; " + root.string());
    }
};
METALLIC_REGISTER_RHI_TEST(PipelineCacheNativeFeedbackTest);

} // namespace
} // namespace metallic::tests
