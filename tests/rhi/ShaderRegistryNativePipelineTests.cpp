#include "RHITest.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>

namespace metallic::tests {
namespace {

using namespace render;
using namespace std::chrono_literals;

class NativePipelineCalls {
public:
    explicit NativePipelineCalls(Device& device)
        : functions_(*const_cast<VolkDeviceTable*>(vulkan::nativeDevice(device).functions)),
          compute_(functions_.vkCreateComputePipelines), graphics_(functions_.vkCreateGraphicsPipelines)
    {
        active_ = this;
        functions_.vkCreateComputePipelines = createCompute;
        functions_.vkCreateGraphicsPipelines = createGraphics;
    }

    ~NativePipelineCalls()
    {
        functions_.vkCreateComputePipelines = compute_;
        functions_.vkCreateGraphicsPipelines = graphics_;
        active_ = nullptr;
    }

    std::atomic<uint32_t> computeCalls{0};
    std::atomic<uint32_t> graphicsCalls{0};
    std::atomic<bool> failCompute{false};

    void blockCompute()
    {
        std::lock_guard lock(gateMutex_);
        blocked_ = true;
        entered_ = 0;
    }

    bool waitForEntries(uint32_t count)
    {
        std::unique_lock lock(gateMutex_);
        return gateCondition_.wait_for(lock, 2s, [&] { return entered_ >= count; });
    }

    void releaseCompute()
    {
        std::lock_guard lock(gateMutex_);
        blocked_ = false;
        gateCondition_.notify_all();
    }

private:
    static VKAPI_ATTR VkResult VKAPI_CALL createCompute(VkDevice device, VkPipelineCache cache, uint32_t count,
        const VkComputePipelineCreateInfo* infos, const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
    {
        auto& self = *active_;
        self.computeCalls.fetch_add(count);
        {
            std::unique_lock lock(self.gateMutex_);
            ++self.entered_;
            self.gateCondition_.notify_all();
            self.gateCondition_.wait(lock, [&] { return !self.blocked_; });
        }
        if (self.failCompute.load()) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
        return self.compute_(device, cache, count, infos, allocator, pipelines);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createGraphics(VkDevice device, VkPipelineCache cache, uint32_t count,
        const VkGraphicsPipelineCreateInfo* infos, const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
    {
        auto& self = *active_;
        self.graphicsCalls.fetch_add(count);
        return self.graphics_(device, cache, count, infos, allocator, pipelines);
    }

    inline static NativePipelineCalls* active_ = nullptr;
    VolkDeviceTable& functions_;
    PFN_vkCreateComputePipelines compute_;
    PFN_vkCreateGraphicsPipelines graphics_;
    std::mutex gateMutex_;
    std::condition_variable gateCondition_;
    uint32_t entered_ = 0;
    bool blocked_ = false;
};

template <typename Predicate>
bool waitUntil(Predicate&& predicate)
{
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::yield();
    }
    return true;
}

class ShaderRegistryNativePipelineTest final : public RHITest {
public:
    ShaderRegistryNativePipelineTest()
    {
        type = RHITestType::Resource;
        name = "shader_registry_native_objects_single_flight_and_bypass";
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto& registry = ShaderRegistry::instance();
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto root = std::filesystem::absolute(context.outputDirectory / "native-pipeline-registry" / std::to_string(nonce));
        std::filesystem::create_directories(root);
        {
            std::ofstream source(root / "NativeCacheProbe.slang");
            source << "[shader(\"compute\")] [numthreads(PROBE_THREADS, 1, 1)]\n"
                      "void computeMain(uint3 id : SV_DispatchThreadID) {}\n";
            if (!source) { return RHITestResult::fail("could not write native cache fixture"); }
        }
        {
            std::ofstream source(root / "NativeCacheGraphicsProbe.slang");
            source << "[shader(\"vertex\")] float4 vertexMain(uint id : SV_VertexID) : SV_Position {\n"
                   << "return float4(float(id & 1u), asfloat(" << (0x3f000000u | uint32_t(nonce & 0x007fffffu))
                   << "u), 0.0, 1.0); }\n"
                      "[shader(\"fragment\")] float4 fragmentMain() : SV_Target0 { return float4(1,0,0,1); }\n"
                      "[shader(\"fragment\")] float4 greenMain() : SV_Target0 { return float4(0,1,0,1); }\n";
            if (!source) { return RHITestResult::fail("could not write native graphics fixture"); }
        }
        const std::string sourceRoot = root.string();
        const std::string spirvRoot = (root / "spirv").string();
        std::array<ShaderCompileResult, 6> code;
        std::array<std::unique_ptr<ShaderModule>, 6> modules;
        for (uint32_t variant = 0; variant < modules.size(); ++variant) {
            const std::string threads = std::to_string(57 + variant);
            const SlangMacroDefine macro{"PROBE_THREADS", threads.c_str()};
            std::string diagnostics;
            auto compiled = registry.getShader({.moduleName = "NativeCacheProbe", .entryPointName = "computeMain",
                .searchPath = sourceRoot.c_str(), .macroDefines = {&macro, 1}},
                {.cacheDirectory = spirvRoot.c_str()}, diagnostics);
            if (!compiled) { return RHITestResult::fail("could not compile native cache fixture: " + diagnostics); }
            code[variant] = std::move(*compiled);
            auto module = registry.getShaderModule(context.device, {.spirv = code[variant].spirv});
            if (!module) { return RHITestResult::fail("could not create native cache fixture module"); }
            modules[variant] = std::move(*module);
        }
        NativePipelineCalls native(context.device);
        const ComputePipelineDesc desc{.computeShader = {modules[0].get()}};
        auto first = registry.getComputePipeline(context.device, desc);
        auto second = registry.getComputePipeline(context.device, desc);
        if (!first || !second || native.computeCalls != 1 ||
            first->get() == second->get() ||
            vulkan::nativePipeline(**first).pipeline != vulkan::nativePipeline(**second).pipeline ||
            vulkan::nativePipeline(**first).layout != vulkan::nativePipeline(**second).layout) {
            return RHITestResult::fail("compute registry did not share a single native object through independent wrappers");
        }
        const auto reusedMetrics = (*second)->creationStats();
        if (!reusedMetrics.nativeObjectReused || reusedMetrics.nativeCreateTimeNanoseconds != 0 ||
            reusedMetrics.driverDurationNanoseconds != 0 || reusedMetrics.cacheLockWaitTimeNanoseconds != 0 ||
            reusedMetrics.feedbackValid || reusedMetrics.nativeCacheHit || reusedMetrics.singleFlightWaitTimeNanoseconds != 0 ||
            (*first)->creationStats().nativeObjectReused) {
            return RHITestResult::fail("ready object reuse reported origin native work as work done by this acquisition");
        }
        auto nullEntry = desc;
        nullEntry.computeShader.entryPoint = nullptr;
        if (registry.getComputePipeline(context.device, nullEntry) || native.computeCalls != 1) {
            return RHITestResult::fail("warm native cache bypassed compute entry point validation");
        }
        // A new module wrapper with identical finalized code must use the same
        // object; source path and wrapper address are not native cache keys.
        auto equivalent = registry.getShaderModule(context.device, {.spirv = code[0].spirv});
        if (!equivalent) { return RHITestResult::fail("could not create equivalent shader module"); }
        auto equivalentPipeline = registry.getComputePipeline(context.device, {.computeShader = {equivalent->get()}});
        auto changed = registry.getComputePipeline(context.device, {.computeShader = {modules[1].get()}});
        if (!equivalentPipeline || !changed || native.computeCalls != 2 ||
            vulkan::nativePipeline(**equivalentPipeline).pipeline != vulkan::nativePipeline(**first).pipeline ||
            changed->get()->psoHash() == first->get()->psoHash() ||
            vulkan::nativePipeline(**changed).pipeline == vulkan::nativePipeline(**first).pipeline) {
            return RHITestResult::fail("native compute identity aliased changed SPIR-V or failed to share identical device code");
        }
        const uint32_t beforeParallel = native.computeCalls;
        const auto beforeParallelStats = registry.nativePipelineCacheStats(context.device);
        if (!beforeParallelStats) { return RHITestResult::fail("could not read native object cache stats"); }
        native.blockCompute();
        std::barrier start(9);
        std::array<std::future<Result<std::unique_ptr<ComputePipeline>>>, 8> workers;
        for (auto& worker : workers) {
            worker = std::async(std::launch::async, [&] {
                start.arrive_and_wait();
                return registry.getComputePipeline(context.device, {.computeShader = {modules[2].get()}});
            });
        }
        start.arrive_and_wait();
        const bool entered = native.waitForEntries(1);
        const bool waited = waitUntil([&] {
            const auto stats = registry.nativePipelineCacheStats(context.device);
            return stats && stats->singleFlightWaitCount == beforeParallelStats->singleFlightWaitCount + 7;
        });
        native.releaseCompute();
        std::array<std::unique_ptr<ComputePipeline>, 8> parallelPipelines;
        for (uint32_t worker = 0; worker < workers.size(); ++worker) {
            auto created = workers[worker].get();
            if (!created) { return RHITestResult::fail("parallel native cache creation failed"); }
            parallelPipelines[worker] = std::move(*created);
        }
        const VkPipeline parallelHandle = vulkan::nativePipeline(*parallelPipelines[0]).pipeline;
        if (!entered || !waited || native.computeCalls != beforeParallel + 1) {
            return RHITestResult::fail("same-key parallel registry requests performed more than one native creation");
        }
        uint32_t waiterCount = 0;
        uint64_t waiterTime = 0;
        for (const auto& pipeline : parallelPipelines) {
            if (vulkan::nativePipeline(*pipeline).pipeline != parallelHandle) {
                return RHITestResult::fail("same-key waiters received different native pipelines");
            }
            const auto metrics = pipeline->creationStats();
            if (metrics.nativeObjectReused) {
                ++waiterCount;
                waiterTime += metrics.singleFlightWaitTimeNanoseconds;
                if (metrics.nativeCreateTimeNanoseconds != 0 ||
                    metrics.feedbackValid) { return RHITestResult::fail("single-flight waiter reported native compilation work"); }
            }
        }
        if (waiterCount != 7 || waiterTime == 0) {
            return RHITestResult::fail("single-flight acquisition metrics did not identify all seven waiters");
        }
        const auto beforeFailure = registry.nativePipelineCacheStats(context.device);
        if (!beforeFailure) { return RHITestResult::fail("could not read native failure baseline stats"); }
        native.failCompute = true;
        native.blockCompute();
        std::barrier failureStart(9);
        for (auto& worker : workers) {
            worker = std::async(std::launch::async, [&] {
                failureStart.arrive_and_wait();
                return registry.getComputePipeline(context.device, {.computeShader = {modules[3].get()}});
            });
        }
        failureStart.arrive_and_wait();
        const bool failureEntered = native.waitForEntries(1);
        const bool failureWaited = waitUntil([&] {
            const auto stats = registry.nativePipelineCacheStats(context.device);
            return stats && stats->singleFlightWaitCount == beforeFailure->singleFlightWaitCount + 7;
        });
        native.releaseCompute();
        bool errorsShared = true;
        for (auto& worker : workers) { errorsShared &= hasError(worker.get(), Error::OutOfMemory); }
        native.failCompute = false;
        auto retry = registry.getComputePipeline(context.device, {.computeShader = {modules[3].get()}});
        if (!failureEntered || !failureWaited || !errorsShared || !retry || native.computeCalls != beforeParallel + 3) {
            return RHITestResult::fail("native failure remained cached or retry did not create a fresh native object");
        }
        // Keep both callbacks inside Vulkan open at once. This proves distinct
        // keys do not hold a global object-cache mutex across driver creation.
        native.blockCompute();
        auto distinctFirst = std::async(std::launch::async, [&] {
            return registry.getComputePipeline(context.device, {.computeShader = {modules[4].get()}});
        });
        auto distinctSecond = std::async(std::launch::async, [&] {
            return registry.getComputePipeline(context.device, {.computeShader = {modules[5].get()}});
        });
        const bool distinctOverlapped = native.waitForEntries(2);
        native.releaseCompute();
        auto distinctFirstResult = distinctFirst.get();
        auto distinctSecondResult = distinctSecond.get();
        if (!distinctOverlapped || !distinctFirstResult || !distinctSecondResult ||
            native.computeCalls != beforeParallel + 5) {
            return RHITestResult::fail("distinct native object keys could not reach Vulkan concurrently");
        }
        // Explicit caches must be filled even when the registry already holds
        // this exact native pipeline. Direct RHI factories must create too.
        auto explicitCache = context.device.createPipelineCache({.saveOnDestroy = false});
        if (!explicitCache) { return RHITestResult::fail("could not create explicit bypass cache"); }
        auto explicitDesc = desc;
        explicitDesc.pipelineCache = explicitCache->get();
        const uint32_t beforeBypass = native.computeCalls;
        auto explicitFirst = registry.getComputePipeline(context.device, explicitDesc);
        auto explicitSecond = registry.getComputePipeline(context.device, explicitDesc);
        auto directFirst = context.device.createComputePipeline(desc);
        auto directSecond = context.device.createComputePipeline(desc);
        if (!explicitFirst || !explicitSecond || !directFirst || !directSecond ||
            native.computeCalls != beforeBypass + 4 || (*explicitCache)->stats().missCount != 1 ||
            (*explicitCache)->stats().hitCount != 1 || (*explicitCache)->stats().sessionPsoCount != 1 ||
            vulkan::nativePipeline(**explicitFirst).pipeline == vulkan::nativePipeline(**first).pipeline ||
            vulkan::nativePipeline(**explicitFirst).pipeline == vulkan::nativePipeline(**explicitSecond).pipeline ||
            vulkan::nativePipeline(**directFirst).pipeline == vulkan::nativePipeline(**directSecond).pipeline) {
            return RHITestResult::fail("explicit cache/direct RHI bypass skipped creation or did not populate the caller's cache");
        }
        std::string diagnostics;
        const auto graphicsSource = [&](const char* entry) {
            return SlangShaderDesc{.moduleName = "NativeCacheGraphicsProbe", .entryPointName = entry,
                .searchPath = sourceRoot.c_str()};
        };
        auto vertexCode = registry.getShader(graphicsSource("vertexMain"), {.cacheDirectory = spirvRoot.c_str()}, diagnostics);
        auto fragmentCode = registry.getShader(graphicsSource("fragmentMain"), {.cacheDirectory = spirvRoot.c_str()}, diagnostics);
        auto greenCode = registry.getShader(graphicsSource("greenMain"), {.cacheDirectory = spirvRoot.c_str()}, diagnostics);
        if (!vertexCode || !fragmentCode || !greenCode) {
            return RHITestResult::fail("could not compile native graphics fixtures: " + diagnostics);
        }
        auto vertex = registry.getShaderModule(context.device, {.spirv = vertexCode->spirv});
        auto fragment = registry.getShaderModule(context.device, {.spirv = fragmentCode->spirv});
        auto green = registry.getShaderModule(context.device, {.spirv = greenCode->spirv});
        if (!vertex || !fragment || !green) { return RHITestResult::fail("could not create native graphics modules"); }
        const GraphicsPipelineDesc graphicsDesc{.vertexShader = {vertex->get()}, .fragmentShader = {fragment->get()},
            .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1};
        auto graphics = registry.getGraphicsPipeline(context.device, graphicsDesc);
        auto graphicsReused = registry.getGraphicsPipeline(context.device, graphicsDesc);
        auto formatDesc = graphicsDesc;
        formatDesc.colorFormats[0] = Format::RGBA16Sfloat;
        auto format = registry.getGraphicsPipeline(context.device, formatDesc);
        auto rasterDesc = graphicsDesc;
        rasterDesc.rasterization.cullMode = CullMode::Front;
        auto raster = registry.getGraphicsPipeline(context.device, rasterDesc);
        auto shaderDesc = graphicsDesc;
        shaderDesc.fragmentShader.module = green->get();
        auto shader = registry.getGraphicsPipeline(context.device, shaderDesc);
        if (!graphics || !graphicsReused || !format || !raster || !shader || native.graphicsCalls != 4 ||
            vulkan::nativePipeline(**graphics).pipeline != vulkan::nativePipeline(**graphicsReused).pipeline ||
            graphics->get()->psoHash() == format->get()->psoHash() ||
            graphics->get()->psoHash() == raster->get()->psoHash() ||
            graphics->get()->psoHash() == shader->get()->psoHash()) {
            return RHITestResult::fail("graphics native cache did not share exact PSOs or aliased code/format/raster variants");
        }
        auto invalidTask = graphicsDesc;
        invalidTask.taskRequiredSubgroupSize = 32;
        invalidTask.taskRequireFullSubgroups = true;
        auto invalidEntry = graphicsDesc;
        invalidEntry.vertexShader.entryPoint = nullptr;
        if (registry.getGraphicsPipeline(context.device, invalidTask) ||
            registry.getGraphicsPipeline(context.device, invalidEntry) || native.graphicsCalls != 4) {
            return RHITestResult::fail("warm graphics native cache bypassed ignored-field or entry point validation");
        }
        auto graphicsExplicitDesc = graphicsDesc;
        graphicsExplicitDesc.pipelineCache = explicitCache->get();
        auto graphicsExplicitFirst = registry.getGraphicsPipeline(context.device, graphicsExplicitDesc);
        auto graphicsExplicitSecond = registry.getGraphicsPipeline(context.device, graphicsExplicitDesc);
        auto graphicsDirectFirst = context.device.createGraphicsPipeline(graphicsDesc);
        auto graphicsDirectSecond = context.device.createGraphicsPipeline(graphicsDesc);
        if (!graphicsExplicitFirst || !graphicsExplicitSecond || !graphicsDirectFirst || !graphicsDirectSecond ||
            native.graphicsCalls != 8 || (*explicitCache)->stats().missCount != 2 ||
            (*explicitCache)->stats().hitCount != 2 || (*explicitCache)->stats().sessionPsoCount != 2) {
            return RHITestResult::fail("graphics bypass did not issue native creates and populate the explicit binary cache");
        }
        auto otherDevice = createDevice({.applicationName = "Native pipeline isolation",
            .enableValidation = context.enableValidation});
        if (!otherDevice) { return RHITestResult::fail("could not create independent native cache test device"); }
        auto otherModule = registry.getShaderModule(**otherDevice, {.spirv = code[0].spirv});
        if (!otherModule) { return RHITestResult::fail("could not create independent device shader module"); }
        const uint32_t beforeForeign = native.computeCalls;
        if (registry.getComputePipeline(context.device, {.computeShader = {otherModule->get()}}) ||
            native.computeCalls != beforeForeign) {
            return RHITestResult::fail("warm native cache bypassed shader module device ownership validation");
        }
        auto otherPipeline = registry.getComputePipeline(**otherDevice, {.computeShader = {otherModule->get()}});
        if (!otherPipeline || (*otherPipeline)->execution().deviceIdentity() != (*otherDevice)->identity() ||
            (*otherPipeline)->execution().deviceIdentity() == context.device.identity() ||
            vulkan::nativePipeline(**otherPipeline).device == vulkan::nativePipeline(**first).device) {
            return RHITestResult::fail("native cache reused an object owned by a different device");
        }
        const VkPipeline retainedHandle = vulkan::nativePipeline(**first).pipeline;
        PreparedExecution retained = (*first)->execution();
        if (!registry.setNativePipelineCacheCapacity(context.device, 0)) {
            return RHITestResult::fail("could not switch native cache to weak-only retention");
        }
        first->reset();
        second->reset();
        equivalentPipeline->reset();
        const uint32_t beforeWeakReuse = native.computeCalls;
        auto weakReused = registry.getComputePipeline(context.device, desc);
        if (!weakReused || native.computeCalls != beforeWeakReuse ||
            vulkan::nativePipeline(**weakReused).pipeline != retainedHandle) {
            return RHITestResult::fail("PreparedExecution did not preserve weak reuse after wrapper and LRU release");
        }
        weakReused->reset();
        retained = {};
        auto expired = registry.getComputePipeline(context.device, desc);
        if (!expired || native.computeCalls != beforeWeakReuse + 1 || (*expired)->creationStats().nativeObjectReused) {
            return RHITestResult::fail("weak-only cache retained an expired native object or did not recreate it");
        }
        if (!registry.setNativePipelineCacheCapacity(context.device, 128)) {
            return RHITestResult::fail("could not restore native object cache capacity");
        }
        if (!registry.flushPipelineCaches(context.device) || !registry.flushPipelineCaches(**otherDevice)) {
            return RHITestResult::fail("binary cache persistence failed after native object reuse");
        }
        return RHITestResult::pass("native compute/graphics reuse, gated single-flight/failure fanout, distinct-key overlap, acquisition metrics, warm-hit validation, explicit/direct bypass, state/device isolation and PreparedExecution weak lifetime");
    }
};
METALLIC_REGISTER_RHI_TEST(ShaderRegistryNativePipelineTest);

} // namespace
} // namespace metallic::tests
