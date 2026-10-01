#include "Fixtures.h"
#include "TraceRecorder.h"
#include "BufferSequence.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSynchronization.h"
#include <thread>

namespace metallic::tests {
namespace {
using namespace render;
class SynchronizationContractTest final : public RHITest {
public:
    SynchronizationContractTest() { type = RHITestType::Validation; name = "synchronization_encoding_cpu"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .requirements = {.requiresDevice = false,
            .validation = bench::Validation::Off, .queues = {}},
            .coverage = {"sync.productionEncoding", "trace.bounded.protocol", "sequence.legal.contract"}, .artifacts = {"encoding.json"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        bench::Evidence evidence(context.outputDirectory / "encoding"); return runCpu(evidence);
    }
    RHITestResult runCpu(bench::Evidence& evidence) override
    {
        bench::Json checks = bench::Json::array();
        bool valid = true;
        const auto check = [&](bool pass, const std::string& label) {
            checks.push_back({{"check", label}, {"pass", pass}}); valid &= pass;
        };
        const std::pair<PipelineStageBits, VkPipelineStageFlags2> stages[]{
            {PipelineStageBits::TopOfPipe, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT},
            {PipelineStageBits::DrawIndirect, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT},
            {PipelineStageBits::VertexShader, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT},
            {PipelineStageBits::FragmentShader, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT},
            {PipelineStageBits::ComputeShader, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT},
            {PipelineStageBits::ColorAttachment, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
            {PipelineStageBits::Transfer, VK_PIPELINE_STAGE_2_TRANSFER_BIT},
            {PipelineStageBits::BottomOfPipe, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT},
            {PipelineStageBits::AllCommands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
            {PipelineStageBits::DepthStencil, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT},
            {PipelineStageBits::PreRasterization, VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT},
            {PipelineStageBits::AccelerationStructureBuild, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR},
            {PipelineStageBits::RayTracingShader, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR},
            {PipelineStageBits::MemoryDecompression, VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT},
            {PipelineStageBits::Host, VK_PIPELINE_STAGE_2_HOST_BIT}};
        for (const auto [stage, flags] : stages) { check(vulkan::toVkPipelineStages(stage) == flags, "stage/" + std::to_string(uint64_t(stage))); }
        const std::pair<AccessBits, VkAccessFlags2> accesses[]{
            {AccessBits::ShaderRead, VK_ACCESS_2_SHADER_READ_BIT}, {AccessBits::ShaderWrite, VK_ACCESS_2_SHADER_WRITE_BIT},
            {AccessBits::UniformRead, VK_ACCESS_2_UNIFORM_READ_BIT}, {AccessBits::IndirectRead, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT},
            {AccessBits::TransferRead, VK_ACCESS_2_TRANSFER_READ_BIT}, {AccessBits::TransferWrite, VK_ACCESS_2_TRANSFER_WRITE_BIT},
            {AccessBits::ColorRead, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT}, {AccessBits::ColorWrite, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT},
            {AccessBits::DepthStencilRead, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT}, {AccessBits::DepthStencilWrite, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
            {AccessBits::AccelerationStructureRead, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR},
            {AccessBits::AccelerationStructureWrite, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR},
            {AccessBits::DecompressionRead, VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT}, {AccessBits::DecompressionWrite, VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT},
            {AccessBits::HostRead, VK_ACCESS_2_HOST_READ_BIT}, {AccessBits::HostWrite, VK_ACCESS_2_HOST_WRITE_BIT},
            {AccessBits::MemoryRead, VK_ACCESS_2_MEMORY_READ_BIT}, {AccessBits::MemoryWrite, VK_ACCESS_2_MEMORY_WRITE_BIT},
            {AccessBits::DescriptorRead, VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT | VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT}};
        for (const auto [access, flags] : accesses) { check(vulkan::accessFlags(access) == flags, "access/" + std::to_string(uint64_t(access))); }
        const auto combined = vulkan::scopeInfo({PipelineStageBits::Transfer | PipelineStageBits::Host, AccessBits::TransferWrite | AccessBits::HostRead});
        check(combined.stage == (VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT) &&
            combined.access == (VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_READ_BIT), "combined scope");
        check(vulkan::scopeInfo({}).stage == 0 && vulkan::scopeInfo({}).access == 0, "empty scope");
        const vulkan::SyncSupport copy{VK_QUEUE_TRANSFER_BIT};
        const vulkan::SyncSupport full{VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, true, true, true, true};
        check(vulkan::validScope({}, copy), "empty valid");
        check(vulkan::validScope({PipelineStageBits::Transfer, AccessBits::None}, copy), "execution-only");
        check(vulkan::validScope({PipelineStageBits::Transfer, AccessBits::TransferWrite}, copy), "copy transfer");
        check(!vulkan::validScope({PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, copy), "queue support");
        check(!vulkan::validScope({PipelineStageBits::Transfer, AccessBits::ShaderWrite}, full), "stage access mismatch");
        check(!vulkan::validScope({PipelineStageBits::AllCommands, AccessBits::HostRead}, full), "host stage required");
        check(!vulkan::validScope({PipelineStageBits::None, AccessBits::TransferRead}, full), "access without stages");
        check(!vulkan::validScope({PipelineStageBits(uint64_t(1) << 63), AccessBits::None}, full), "unknown stages");
        check(!vulkan::validScope({PipelineStageBits::AllCommands, AccessBits(uint64_t(1) << 63)}, full), "unknown access");
        check(vulkan::validScope({PipelineStageBits::AccelerationStructureBuild, AccessBits::AccelerationStructureWrite}, full) &&
            !vulkan::validScope({PipelineStageBits::AccelerationStructureBuild, AccessBits::AccelerationStructureWrite}, copy), "AS gating");
        check(!vulkan::validScope({PipelineStageBits::AllCommands, AccessBits::DescriptorRead}, copy), "descriptor gating");
        const std::pair<TextureLayout, VkImageLayout> layouts[]{
            {TextureLayout::Undefined, VK_IMAGE_LAYOUT_UNDEFINED}, {TextureLayout::General, VK_IMAGE_LAYOUT_GENERAL},
            {TextureLayout::Present, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR}, {TextureLayout::ShaderRead, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {TextureLayout::TransferSource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}, {TextureLayout::TransferDestination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
            {TextureLayout::ColorAttachment, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, {TextureLayout::DepthStencilAttachment, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
        for (const auto [layout, native] : layouts) {
            check(vulkan::imageLayout(layout, false) == native, "layout/" + std::to_string(int(layout)));
            check(vulkan::imageLayout(layout, true) == (layout == TextureLayout::Undefined || layout == TextureLayout::Present ? native : VK_IMAGE_LAYOUT_GENERAL), "unified/" + std::to_string(int(layout)));
        }
        bench::TraceRecorder trace(4);
        const auto command = reinterpret_cast<VkCommandBuffer>(uintptr_t(0x123456));
        trace.capture({.kind = vulkan::TraceKind::CommandBegin, .command = command});
        VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
        trace.capture({.kind = vulkan::TraceKind::Barrier, .command = command, .dependency = &dependency});
        barrier.srcAccessMask = 0;
        trace.capture({.kind = vulkan::TraceKind::Retire, .objectType = VK_OBJECT_TYPE_COMMAND_BUFFER, .object = uint64_t(command)});
        trace.capture({.kind = vulkan::TraceKind::CommandBegin, .command = command});
        const auto events = trace.snapshot().at("events");
        check(events[1]["memory"][0]["before"]["access"].get<uint64_t>() == VK_ACCESS_2_TRANSFER_WRITE_BIT, "trace owns data");
        check(events[0]["command"] != events[3]["command"] && events[0]["recording"] != events[3]["recording"], "trace generation reuse");
        check(events.dump().find("1193046") == std::string::npos, "trace hides raw handle");
        trace.capture({.kind = vulkan::TraceKind::CommandBegin, .command = command});
        check(trace.failed(), "trace overflow is failure");
        bench::TraceRecorder oversized;
        VkDependencyInfo tooMany{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1025};
        oversized.capture({.kind = vulkan::TraceKind::Barrier, .command = command, .dependency = &tooMany});
        check(oversized.failed(), "trace per-event budget before dereference");
        bench::TraceRecorder submissions;
        submissions.capture({.kind = vulkan::TraceKind::CommandBegin, .command = command});
        VkCommandBufferSubmitInfo commands{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = command};
        VkSemaphoreSubmitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = reinterpret_cast<VkSemaphore>(uintptr_t(0x567890)), .value = 7, .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT};
        VkSemaphoreSubmitInfo signal = wait; signal.value = 8;
        VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait,
            .commandBufferInfoCount = 1, .pCommandBufferInfos = &commands, .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal};
        submissions.capture({.kind = vulkan::TraceKind::Submit, .queue = reinterpret_cast<VkQueue>(uintptr_t(0x987654)), .submit = &submit});
        wait.value = 100; signal.stageMask = 0;
        const auto submitted = submissions.snapshot().at("events")[1];
        check(submitted.at("waits")[0].at("value") == 7 && submitted.at("signals")[0].at("value") == 8 &&
            submitted.at("signals")[0].at("stages").get<uint64_t>() == VK_PIPELINE_STAGE_2_TRANSFER_BIT &&
            submitted.at("commands")[0].at("recording") == 1 &&
            submitted.at("waits")[0].at("id") == submitted.at("signals")[0].at("id"), "trace owns final submission");
        uint32_t callbacks = 0;
        vulkan::TraceSink sink{reinterpret_cast<VkDevice>(uintptr_t(0xabcdef)),
            [](void* counter, const vulkan::TraceEvent&) noexcept { ++*static_cast<uint32_t*>(counter); }, &callbacks};
        const bool installed = vulkan::installTraceSink(&sink);
        check(installed == vulkan::traceCompiled(), "trace build switch");
        if (installed) {
            check(!vulkan::installTraceSink(&sink), "single trace session");
            vulkan::emitTrace({.kind = vulkan::TraceKind::CommandBegin, .device = sink.device});
            vulkan::emitTrace({.kind = vulkan::TraceKind::CommandBegin, .device = VK_NULL_HANDLE});
            vulkan::removeTraceSink(&sink);
            vulkan::emitTrace({.kind = vulkan::TraceKind::CommandBegin, .device = sink.device});
            check(callbacks == 1, "trace device filter and detach");
        }
        bench::TraceRecorder concurrent;
        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) { threads.emplace_back([&] {
            for (int n = 0; n < 32; ++n) { concurrent.capture({.kind = vulkan::TraceKind::CommandBegin, .command = command}); }
        }); }
        threads.clear(); check(concurrent.snapshot().at("events").size() == 128 && !concurrent.failed(), "trace concurrent capture");
        for (uint32_t i = 0; i < 32; ++i) {
            auto program = bench::generateBufferSequence(i, 0, i % 2 == 0);
            bench::validateBufferSequence(program);
            program["commands"].erase(program["commands"].begin()); bench::validateBufferSequence(program);
        }
        check(bench::generateBufferSequence(42, 3) == bench::generateBufferSequence(42, 3) &&
            bench::generateBufferSequence(42, 3) != bench::generateBufferSequence(42, 4), "deterministic seed and iteration");
        auto invalid = bench::generateBufferSequence(1, 0); invalid["commands"][0]["offset"] = -1;
        bool rejected = false; try { bench::validateBufferSequence(invalid); } catch (...) { rejected = true; }
        check(rejected, "invalid sequence rejected before GPU");
        invalid = bench::generateBufferSequence(1, 0); invalid["schema"] = 1.0;
        rejected = false; try { bench::validateBufferSequence(invalid); } catch (...) { rejected = true; }
        check(rejected, "sequence schema rejects floating point");
        invalid = bench::generateBufferSequence(1, 0); invalid["seed"] = -1;
        rejected = false; try { bench::validateBufferSequence(invalid); } catch (...) { rejected = true; }
        check(rejected, "sequence rejects negative seed");
        evidence.json("encoding.json", checks);
        return valid ? RHITestResult::pass() : RHITestResult::fail("encoding/trace/sequence contract failed; see encoding.json");
    }
};
METALLIC_REGISTER_RHI_TEST(SynchronizationContractTest);
} // namespace
} // namespace metallic::tests
