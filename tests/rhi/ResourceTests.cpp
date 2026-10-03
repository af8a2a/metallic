#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <type_traits>
#include <array>
#include <barrier>
#include <future>
#include <stdexcept>

namespace metallic::tests {
namespace {

// Deliberately disable legacy device dispatch. Device creation must not reload
// it, and normal work must keep using its owner's table. Restore for other tests.
struct DisabledGlobalDeviceDispatch {
    VolkDeviceTable saved{};
    DisabledGlobalDeviceDispatch()
    {
#define DISABLE_DEVICE_CALL(name) saved.name = std::exchange(name, nullptr)
        DISABLE_DEVICE_CALL(vkCreateFence);
        DISABLE_DEVICE_CALL(vkDestroyFence);
        DISABLE_DEVICE_CALL(vkCreateSemaphore);
        DISABLE_DEVICE_CALL(vkGetSemaphoreCounterValue);
        DISABLE_DEVICE_CALL(vkCreateCommandPool);
        DISABLE_DEVICE_CALL(vkAllocateCommandBuffers);
        DISABLE_DEVICE_CALL(vkBeginCommandBuffer);
        DISABLE_DEVICE_CALL(vkCmdCopyMemoryKHR);
        DISABLE_DEVICE_CALL(vkQueueSubmit2);
        DISABLE_DEVICE_CALL(vkDeviceWaitIdle);
#undef DISABLE_DEVICE_CALL
    }
    ~DisabledGlobalDeviceDispatch()
    {
#define RESTORE_DEVICE_CALL(name) name = saved.name
        RESTORE_DEVICE_CALL(vkCreateFence);
        RESTORE_DEVICE_CALL(vkDestroyFence);
        RESTORE_DEVICE_CALL(vkCreateSemaphore);
        RESTORE_DEVICE_CALL(vkGetSemaphoreCounterValue);
        RESTORE_DEVICE_CALL(vkCreateCommandPool);
        RESTORE_DEVICE_CALL(vkAllocateCommandBuffers);
        RESTORE_DEVICE_CALL(vkBeginCommandBuffer);
        RESTORE_DEVICE_CALL(vkCmdCopyMemoryKHR);
        RESTORE_DEVICE_CALL(vkQueueSubmit2);
        RESTORE_DEVICE_CALL(vkDeviceWaitIdle);
#undef RESTORE_DEVICE_CALL
    }
    bool unchanged() const
    {
        return !vkCreateFence && !vkDestroyFence && !vkCreateSemaphore && !vkGetSemaphoreCounterValue &&
            !vkCreateCommandPool && !vkAllocateCommandBuffers && !vkBeginCommandBuffer &&
            !vkCmdCopyMemoryKHR && !vkQueueSubmit2 && !vkDeviceWaitIdle;
    }
};

class ConcurrentDeviceDispatchTest final : public RHITest {
public:
    ConcurrentDeviceDispatchTest() { type = RHITestType::Resource; name = "concurrent_device_dispatch_tables"; }
    RHITestResult run(RHITestContext& context) override
    {
        DisabledGlobalDeviceDispatch globals;
        std::array<bench::ValidationRecorder, 2> validation;
        std::array<const VolkDeviceTable*, 2> tables{};
        std::barrier ready(2);
        auto worker = [&](uint32_t index) -> std::string {
            auto created = render::createDevice({.applicationName = "Concurrent Vulkan dispatch",
                .enableValidation = true, .validationSink = validation[index].sink()});
            // Both devices must stay alive during the concurrent workload.
            if (created) { tables[index] = render::vulkan::nativeDevice(**created).functions; }
            ready.arrive_and_wait();
            if (!created) { return "concurrent device creation failed"; }
            auto device = std::move(*created);
            const auto native = render::vulkan::nativeDevice(*device);
            if (!native.validationEnabled || !native.validationMessengerActive) { return "validation unavailable"; }
            auto take = [](auto result) {
                if (!result) { throw std::runtime_error("resource creation failed"); }
                return std::move(*result);
            };
            auto check = [](render::Result<> result) {
                if (!result) { throw std::runtime_error("concurrent device operation failed"); }
            };
            try {
                auto& queue = *device->getQueue(render::QueueType::Graphics);
                auto semaphore = take(device->createSemaphore());
                auto upload = take(device->createBuffer({.size = 256,
                    .usage = render::BufferUsageBits::TransferSource,
                    .memoryLocation = render::MemoryLocation::HostUpload}));
                auto readback = take(device->createBuffer({.size = 256,
                    .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback}));
                for (uint32_t iteration = 1; iteration <= 64; ++iteration) {
                    const uint32_t expected = (index + 1) * 1000 + iteration;
                    auto* source = static_cast<uint32_t*>(upload->map());
                    if (!source) { throw std::runtime_error("upload map failed"); }
                    for (uint32_t word = 0; word < 64; ++word) { source[word] = expected + word; }
                    upload->unmap();
                    bench::GPUCommands commands(queue);
                    check(commands.initialize(*device));
                    check(commands.commands->copyBuffer(take(upload->slice()), take(readback->slice())));
                    check(commands.submitAndWait());
                    check(semaphore->signal(iteration));
                    if (semaphore->currentValue() != iteration) { throw std::runtime_error("timeline crossed devices"); }
                    auto* output = static_cast<const uint32_t*>(readback->map());
                    if (!output) { throw std::runtime_error("readback map failed"); }
                    bool equal = true;
                    for (uint32_t word = 0; word < 64; ++word) { equal &= output[word] == expected + word; }
                    readback->unmap();
                    if (!equal) { throw std::runtime_error("readback crossed devices"); }
                }
                check(device->waitIdle());
            } catch (const std::exception& error) {
                return error.what();
            }
            return {};
        };
        auto first = std::async(std::launch::async, worker, 0);
        auto second = std::async(std::launch::async, worker, 1);
        const auto firstError = first.get();
        const auto secondError = second.get();
        for (size_t i = 0; i < validation.size(); ++i) {
            const auto evidence = validation[i].snapshot();
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream(context.outputDirectory / (std::string(name) + std::to_string(i) + ".json")) << evidence.dump(2);
            if (evidence.at("captureFailed").get<bool>()) { return RHITestResult::fail("validation capture failed"); }
            for (const auto& message : evidence.at("messages")) {
                if ((message.at("severity").get<uint32_t>() & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
                    (message.at("type").get<uint32_t>() & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) {
                    return RHITestResult::fail(message.at("text").get<std::string>());
                }
            }
        }
        if (!globals.unchanged()) { return RHITestResult::fail("device creation rewrote global device dispatch"); }
        if (firstError == "validation unavailable" || secondError == "validation unavailable") {
            return RHITestResult::skip("validation required for concurrent device dispatch test");
        }
        if (!firstError.empty() || !secondError.empty()) { return RHITestResult::fail(firstError + " " + secondError); }
        if (!tables[0] || !tables[1] || tables[0] == tables[1]) { return RHITestResult::fail("devices share dispatch table storage"); }
        return RHITestResult::pass("two concurrent devices, 128 validated copy submissions with global dispatch disabled");
    }
};
METALLIC_REGISTER_RHI_TEST(ConcurrentDeviceDispatchTest);

enum class MoveResource { Fence, Semaphore, SwapchainSemaphore, ShaderModule, CommandPool, Timestamp, Compaction };

template<MoveResource Resource>
class ResourceMoveAssignmentTest final : public RHITest {
public:
    ResourceMoveAssignmentTest()
    {
        type = RHITestType::Resource;
        constexpr const char* names[] = {
            "resource_move_assignment_fence", "resource_move_assignment_semaphore",
            "resource_move_assignment_swapchain_semaphore", "resource_move_assignment_shader_module",
            "resource_move_assignment_command_pool", "resource_move_assignment_timestamp_query_pool",
            "resource_move_assignment_compaction_query_pool"};
        name = names[static_cast<size_t>(Resource)];
    }

    RHITestResult run(RHITestContext& context) override
    {
        bench::ValidationRecorder validation;
        // This must be an owned device: vkDestroyDevice diagnoses leaked children
        // after every wrapper (including moved-from wrappers) has been destroyed.
        auto created = render::createDevice({.applicationName = name, .enableValidation = true,
            .enableRayTracingAccelerationStructure = Resource == MoveResource::Compaction,
            .validationSink = validation.sink()});
        if (render::hasError(created, render::Error::Unsupported)) {
            return RHITestResult::skip("required device capabilities unavailable");
        }
        if (!created) { return RHITestResult::fail("cannot create lifecycle test device"); }
        auto device = std::move(*created);
        const auto native = render::vulkan::nativeDevice(*device);
        if (!native.validationEnabled || !native.validationMessengerActive) {
            return RHITestResult::skip("Khronos validation and debug messenger required to detect handle leaks");
        }
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        render::ShaderCompileResult shader;
        if constexpr (Resource == MoveResource::ShaderModule) {
            auto compiled = render::compileSlangShaderToSpirv({.moduleName = "Features/Samples/Triangle",
                .entryPointName = "triangleVertexMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics);
            if (!compiled) { return RHITestResult::fail(shader.diagnostics); }
            shader = std::move(*compiled);
        }
        auto create = [&]() {
            if constexpr (Resource == MoveResource::Fence) { return device->createFence(true); }
            else if constexpr (Resource == MoveResource::Semaphore) { return device->createSemaphore({.initialValue = 9}); }
            else if constexpr (Resource == MoveResource::SwapchainSemaphore) { return device->createSwapchainSemaphore(); }
            else if constexpr (Resource == MoveResource::ShaderModule) { return device->createShaderModule({.spirv = shader.spirv}); }
            else if constexpr (Resource == MoveResource::CommandPool) { return device->createCommandPool(queue); }
            else if constexpr (Resource == MoveResource::Timestamp) { return device->createTimestampQueryPool(queue, {.queryCount = 2}); }
            else { return device->createRayTracingAccelerationStructureCompactionQueryPool({.queryCount = 2}); }
        };
        auto usable = [](auto& object) {
            if constexpr (Resource == MoveResource::Fence) { return object.isSignaled() && bool(object.wait(1'000'000)); }
            else if constexpr (Resource == MoveResource::Semaphore) { return object.currentValue() == 9 && bool(object.wait(9, 1'000'000)); }
            else if constexpr (Resource == MoveResource::ShaderModule) { return object.contentHash() != 0; }
            else if constexpr (Resource == MoveResource::CommandPool) {
                if (!object.reset()) { return false; }
                auto command = object.createCommandBuffer();
                return command && (*command)->begin() && (*command)->end();
            }
            else if constexpr (Resource == MoveResource::Timestamp) { return object.desc().queryCount == 2 && bool(object.reset(0, 2)); }
            else if constexpr (Resource == MoveResource::Compaction) { return object.desc().queryCount == 2; }
            else { return true; } // Binary semaphore ownership is checked by validation at teardown.
        };
        validation.phase("move ownership");
        {
            auto destination = create();
            auto source = create();
            if (render::hasError(destination, render::Error::Unsupported) ||
                render::hasError(source, render::Error::Unsupported)) {
                return RHITestResult::skip("resource capability unavailable");
            }
            if (!destination || !source) { return RHITestResult::fail("resource creation failed"); }
            using Object = std::remove_reference_t<decltype(**source)>;
            Object relocated(std::move(**source));
            if (!usable(relocated)) { return RHITestResult::fail("move construction lost resource"); }
            **destination = std::move(relocated); // Replace a live Vulkan object.
            if (!usable(**destination)) { return RHITestResult::fail("move assignment lost resource"); }
            auto& self = **destination;
            **destination = std::move(self);
            if (!usable(**destination)) { return RHITestResult::fail("self move lost resource"); }
            **source = std::move(**destination); // Reuse a moved-from wrapper.
            if (!usable(**source)) { return RHITestResult::fail("moved-from wrapper could not be reused"); }
            **source = std::move(**destination); // Empty source must release a live destination.
        }
        validation.phase("device destruction");
        device.reset();
        const auto evidence = validation.snapshot();
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream(context.outputDirectory / (std::string(name) + ".json")) << evidence.dump(2);
        if (evidence.at("captureFailed").get<bool>()) { return RHITestResult::fail("validation capture failed"); }
        for (const auto& message : evidence.at("messages")) {
            // Retain loader GENERAL diagnostics in evidence, but only API validation
            // errors (including leaked children and double destruction) fail this test.
            if ((message.at("severity").get<uint32_t>() & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
                (message.at("type").get<uint32_t>() & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) {
                return RHITestResult::fail(message.at("text").get<std::string>());
            }
        }
        return RHITestResult::pass();
    }
};

using FenceMoveAssignmentTest = ResourceMoveAssignmentTest<MoveResource::Fence>;
using SemaphoreMoveAssignmentTest = ResourceMoveAssignmentTest<MoveResource::Semaphore>;
using SwapchainSemaphoreMoveAssignmentTest = ResourceMoveAssignmentTest<MoveResource::SwapchainSemaphore>;
using ShaderModuleMoveAssignmentTest = ResourceMoveAssignmentTest<MoveResource::ShaderModule>;
using CommandPoolMoveAssignmentTest = ResourceMoveAssignmentTest<MoveResource::CommandPool>;
using TimestampMoveAssignmentTest = ResourceMoveAssignmentTest<MoveResource::Timestamp>;
using CompactionMoveAssignmentTest = ResourceMoveAssignmentTest<MoveResource::Compaction>;

METALLIC_REGISTER_RHI_TEST(FenceMoveAssignmentTest);
METALLIC_REGISTER_RHI_TEST(SemaphoreMoveAssignmentTest);
METALLIC_REGISTER_RHI_TEST(SwapchainSemaphoreMoveAssignmentTest);
METALLIC_REGISTER_RHI_TEST(ShaderModuleMoveAssignmentTest);
METALLIC_REGISTER_RHI_TEST(CommandPoolMoveAssignmentTest);
METALLIC_REGISTER_RHI_TEST(TimestampMoveAssignmentTest);
METALLIC_REGISTER_RHI_TEST(CompactionMoveAssignmentTest);

class ResourceLifecycleTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"resource.lifecycle.contract"}, bench::Layer::RHI, "core", "core");
    }

    ResourceLifecycleTest()
    {
        type = RHITestType::Resource;
        name = "resource_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Buffer> invalidBuffer;
        render::Result<> result = context.device.createBuffer(render::BufferDesc{
                .size = 0,
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            }).transform([&](auto rhiValue) { invalidBuffer = std::move(rhiValue); });
        if (!render::hasError(result, render::Error::InvalidArgument) || invalidBuffer != nullptr) {
            return RHITestResult::fail("zero-sized buffer was not rejected");
        }

        std::unique_ptr<render::Buffer> uploadBuffer;
        result = context.device.createBuffer(render::BufferDesc{
                .size = 256,
                .structureStride = 16,
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            }).transform([&](auto rhiValue) { uploadBuffer = std::move(rhiValue); });
        if (!result || uploadBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer returned ") + toString(result));
        }
        if (uploadBuffer->desc().size != 256 || uploadBuffer->desc().structureStride != 16) {
            return RHITestResult::fail("created buffer descriptor does not match request");
        }
        if (uploadBuffer->deviceAddress() == 0) {
            return RHITestResult::fail("transfer buffer requires an implicit device address");
        }
        // These usages must work without Storage or an explicit ShaderDeviceAddress request.
        for (auto usage : {render::BufferUsageBits::Indirect, render::BufferUsageBits::TransferDestination}) {
            std::unique_ptr<render::Buffer> addressBuffer;
            result = context.device.createBuffer({.size = 16, .usage = usage, .memoryLocation = render::MemoryLocation::Device}).transform([&](auto rhiValue) { addressBuffer = std::move(rhiValue); });
            if (!result || addressBuffer == nullptr || addressBuffer->deviceAddress() == 0) {
                return RHITestResult::fail("command buffer usage requires an implicit device address");
            }
        }

        void* mapped = uploadBuffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("host upload buffer did not map");
        }
        std::memset(mapped, 0x5a, static_cast<size_t>(uploadBuffer->desc().size));
        uploadBuffer->unmap();

        std::unique_ptr<render::Texture> invalidTexture;
        result = context.device.createTexture(render::TextureDesc{
                .format = render::Format::Unknown,
            }).transform([&](auto rhiValue) { invalidTexture = std::move(rhiValue); });
        if (!render::hasError(result, render::Error::InvalidArgument) || invalidTexture != nullptr) {
            return RHITestResult::fail("texture with unknown format was not rejected");
        }

        std::unique_ptr<render::Texture> texture;
        result = context.device.createTexture(render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                .format = render::Format::RGBA8Unorm,
                .width = 16,
                .height = 16,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            }).transform([&](auto rhiValue) { texture = std::move(rhiValue); });
        if (!result || texture == nullptr) {
            return RHITestResult::fail(std::string("createTexture returned ") + toString(result));
        }
        if (texture->desc().format != render::Format::RGBA8Unorm ||
            texture->desc().width != 16 ||
            texture->desc().height != 16) {
            return RHITestResult::fail("created texture descriptor does not match request");
        }

        std::unique_ptr<render::TextureView> textureView;
        result = context.device.createTextureView(*texture,
            render::TextureViewDesc{
                .format = render::Format::RGBA8Unorm,
                .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
            }).transform([&](auto rhiValue) { textureView = std::move(rhiValue); });
        if (!result || textureView == nullptr) {
            return RHITestResult::fail(std::string("createTextureView returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(true).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
        }
        if (!fence->isSignaled()) {
            return RHITestResult::fail("signaled fence reported unsignaled");
        }
        result = fence->wait(1'000'000);
        if (!result) {
            return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }
        result = fence->reset();
        if (!result) {
            return RHITestResult::fail(std::string("Fence::reset returned ") + toString(result));
        }
        if (fence->isSignaled()) {
            return RHITestResult::fail("reset fence reported signaled");
        }

        std::unique_ptr<render::Semaphore> semaphore;
        result = context.device.createSemaphore(render::SemaphoreDesc{.initialValue = 2}).transform([&](auto rhiValue) { semaphore = std::move(rhiValue); });
        if (!result || semaphore == nullptr) {
            return RHITestResult::fail(std::string("createSemaphore returned ") + toString(result));
        }
        if (semaphore->currentValue() != 2) {
            return RHITestResult::fail("timeline semaphore initial value does not match request");
        }
        result = semaphore->wait(2, 1'000'000);
        if (!result) {
            return RHITestResult::fail(std::string("Semaphore::wait returned ") + toString(result));
        }
        result = semaphore->signal(3);
        if (!result) {
            return RHITestResult::fail(std::string("Semaphore::signal returned ") + toString(result));
        }
        if (semaphore->currentValue() != 3) {
            return RHITestResult::fail("timeline semaphore host signal did not update value");
        }

        std::unique_ptr<render::SwapchainSemaphore> swapchainSemaphore;
        result = context.device.createSwapchainSemaphore().transform([&](auto rhiValue) { swapchainSemaphore = std::move(rhiValue); });
        if (!result || swapchainSemaphore == nullptr) {
            return RHITestResult::fail(std::string("createSwapchainSemaphore returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

class IndirectArgumentBarrierTest : public RHITest {
public:
    IndirectArgumentBarrierTest()
    {
        type = RHITestType::Command;
        name = "indirect_argument_barrier";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Buffer> indirectBuffer;
        render::Result<> result = context.device.createBuffer(render::BufferDesc{
                .size = 16,
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Indirect,
                .memoryLocation = render::MemoryLocation::Device,
            }).transform([&](auto rhiValue) { indirectBuffer = std::move(rhiValue); });
        if (!result || indirectBuffer == nullptr) {
            return RHITestResult::fail(
                std::string("createBuffer(indirect) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(
                std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(
                std::string("createCommandBuffer returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(
                std::string("CommandBuffer::begin returned ") + toString(result));
        }

        const render::BufferBarrierDesc toIndirect{
            .buffer = indirectBuffer.get(),
            .before = {},
            .after = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{.buffers = {&toIndirect, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }

        const render::BufferBarrierDesc toGeneral{
            .buffer = indirectBuffer.get(),
            .before = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{.buffers = {&toGeneral, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }

        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(
                std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(
                std::string("createFence returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
                .commandBuffers = {commandBuffers, 1},
                .signalFence = fence.get(),
            });
        if (!result) {
            return RHITestResult::fail(
                std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(
                std::string("Fence::wait returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(ResourceLifecycleTest);
METALLIC_REGISTER_RHI_TEST(IndirectArgumentBarrierTest);

} // namespace
} // namespace metallic::tests
