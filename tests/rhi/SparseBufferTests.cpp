#include "RHITest.h"
#include "Runtime/Render/GAPI/RHI.h"
#include <cstring>
#include <stdexcept>

namespace metallic::tests {
namespace {
class SparseBufferLifetimeTest final : public RHITest {
public:
    SparseBufferLifetimeTest()
    {
        type = RHITestType::Resource;
        name = "sparse_buffer_append_cancel_lifetime";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Sparse buffer regression", .enableValidation = context.enableValidation})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Required device features unavailable"); }
        const auto require = [](bool success, const char* message) {
            if (!success) { throw std::runtime_error(message); }
        };
        try {
            require(bool(result), "Device creation failed");
            if (!device->capabilities().sparseBuffers) { return RHITestResult::skip("Sparse buffers unavailable on graphics queue"); }
            const auto committed = [&] { return device->memoryBudget().domains[size_t(MemoryBudgetDomain::CLAS)].allocationBytes; };
            const uint64_t baseline = committed();
            constexpr uint64_t virtualBytes = 256ull * 1024 * 1024;
            constexpr auto usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination;
            std::unique_ptr<Buffer> arena;
            require(bool(device->createSparseBuffer({.size = virtualBytes, .usage = usage, .memoryDomain = MemoryBudgetDomain::CLAS})
                .transform([&](auto value) { arena = std::move(value); })), "Sparse reserve failed");
            require(committed() == baseline && arena->sparseCommittedBytes() == 0, "Virtual reservation charged physical budget");
            const uint64_t address = arena->deviceAddress();
            const uint64_t block = arena->sparseBlockAlignment();
            require(address && block >= 8 && block * 3 <= virtualBytes, "Invalid sparse properties");
            // Virtual capacity must already be aligned. Unlike a commitment
            // request, it must not silently grow and expose bytes past its budget.
            require(hasError(device->createSparseBuffer({.size = block - 1, .usage = usage,
                .memoryDomain = MemoryBudgetDomain::CLAS}), Error::InvalidArgument),
                "Unaligned sparse virtual capacity accepted");
            require(committed() == baseline, "Rejected sparse reservation charged physical budget");
            auto* queue = device->getQueue(QueueType::Graphics);
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Fence> fence;
            require(bool(device->createCommandPool(*queue).transform([&](auto value) { pool = std::move(value); })) &&
                bool(pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); })) &&
                bool(device->createFence(false).transform([&](auto value) { fence = std::move(value); })), "Command resources failed");
            const auto makeBuffer = [&](MemoryLocation location) {
                std::unique_ptr<Buffer> buffer;
                require(bool(device->createBuffer({.size = 16, .usage = usage, .memoryLocation = location})
                    .transform([&](auto value) { buffer = std::move(value); })), "Transfer buffer failed");
                return buffer;
            };
            auto upload = makeBuffer(MemoryLocation::HostUpload);
            auto readback = makeBuffer(MemoryLocation::HostReadback);
            const uint32_t pattern[4] = {0x12345678, 0xabcdef01, 0x99887766, 0x10203040};
            std::memcpy(upload->map(), pattern, sizeof(pattern));
            upload->flush();
            upload->unmap();
            const auto slice = [&](Buffer& buffer, uint64_t offset, uint64_t bytes) {
                auto value = buffer.slice({offset, bytes});
                require(bool(value), "Buffer slice failed");
                return *value;
            };
            const auto submit = [&] {
                require(bool(commands->end()), "End failed");
                CommandBuffer* submitted[] = {commands.get()};
                require(bool(queue->submit({.commandBuffers = submitted, .signalFence = fence.get()})) &&
                    bool(fence->wait(5000000000ull)), "Submit failed");
            };
            const MemoryBarrierDesc copies{{PipelineStageBits::Transfer, AccessBits::TransferWrite},
                {PipelineStageBits::Transfer, AccessBits::TransferRead}};
            const MemoryBarrierDesc host{{PipelineStageBits::Transfer, AccessBits::TransferWrite},
                {PipelineStageBits::Host, AccessBits::HostRead}};
            require(bool(commands->begin()), "Begin failed");
            require(bool(commands->commitSparseBuffer(*arena, block - 1)), "First commit failed");
            require(arena->sparseCommittedBytes() == block && committed() == baseline + block, "Commit was not aligned/accounted");
            commands->hostWriteBarrier();
            require(bool(commands->copyBuffer(slice(*upload, 0, 8), slice(*arena, block - 8, 8))), "First copy failed");
            submit();

            // Bind acceptance is independent of the cancelled command recording.
            require(bool(pool->reset()) && bool(commands->begin()), "Cancel recording failed");
            require(bool(commands->commitSparseBuffer(*arena, 2 * block)), "Cancelled-frame growth failed");
            require(bool(commands->end()) && bool(pool->reset()) && bool(commands->begin()), "Cancel/reset failed");
            require(arena->sparseCommittedBytes() == 2 * block && arena->deviceAddress() == address &&
                committed() == baseline + 2 * block, "Cancellation lost accepted backing or changed addresses");
            require(hasError(commands->commitSparseBuffer(*arena, virtualBytes + 1), Error::InvalidArgument), "Oversized commit accepted");
            require(bool(commands->commitSparseBuffer(*arena, block)), "Reusing accepted backing failed");
            require(arena->sparseCommittedBytes() == 2 * block, "Append-only reuse shrank backing");
            commands->hostWriteBarrier();
            require(bool(commands->copyBuffer(slice(*upload, 8, 8), slice(*arena, block, 8))), "Appended-range copy failed");
            require(bool(commands->synchronize({.memory = {&copies, 1}})) &&
                bool(commands->copyBuffer(slice(*arena, block - 8, 16), slice(*readback, 0, 16))) &&
                bool(commands->synchronize({.memory = {&host, 1}})), "Cross-binding copy failed");
            require(bool(fence->reset()), "Fence reset failed");
            submit();
            readback->invalidate();
            const bool matched = std::memcmp(readback->map(), pattern, sizeof(pattern)) == 0;
            readback->unmap();
            require(matched, "Growth changed resident bytes or cross-binding copy failed");

            // An unsignalled host gate makes the last GPU use deterministically in flight.
            std::unique_ptr<Semaphore> gate;
            require(bool(device->createSemaphore().transform([&](auto value) { gate = std::move(value); })), "Gate failed");
            struct GateRelease {
                Semaphore& gate;
                Fence& fence;
                bool submitted = false;
                bool signalled = false;
                ~GateRelease() { if (!signalled) { (void)gate.signal(1); } if (submitted) { (void)fence.wait(); } }
            } release{*gate, *fence};
            require(bool(pool->reset()) && bool(commands->begin()) && bool(fence->reset()), "In-flight recording failed");
            require(bool(commands->commitSparseBuffer(*arena, 2 * block)), "In-flight retention failed");
            require(bool(commands->copyBuffer(slice(*arena, block - 8, 16), slice(*readback, 0, 16))) &&
                bool(commands->synchronize({.memory = {&host, 1}})) && bool(commands->end()), "In-flight read failed");
            CommandBuffer* submitted[] = {commands.get()};
            const SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
            require(bool(queue->submit({.waitSemaphores = {&wait, 1}, .commandBuffers = submitted, .signalFence = fence.get()})), "Gated submit failed");
            release.submitted = true;
            std::weak_ptr<void> lifetime = arena->retainAllocation();
            arena.reset();
            require(!lifetime.expired() && committed() == baseline + 2 * block, "In-flight sparse memory freed early");
            const auto signalled = gate->signal(1);
            release.signalled = bool(signalled);
            require(bool(signalled) && bool(fence->wait(5000000000ull)), "Gated completion failed");
            release.submitted = false;
            require(bool(pool->reset()) && bool(commands->begin()) && bool(commands->end()), "Final retirement failed");
            // A queue operation also collects completed sparse-bind ownership.
            require(bool(queue->submit({})), "Bind collection failed");
            require(lifetime.expired() && committed() == baseline, "Completed sparse allocation leaked");
            return RHITestResult::pass("Sparse virtual budget/alignment rejection, append, cross-binding reads, cancelled recording and in-flight backing lifetime");
        } catch (const std::exception& error) {
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(SparseBufferLifetimeTest);
} // namespace
} // namespace metallic::tests
