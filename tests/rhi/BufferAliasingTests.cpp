#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

using namespace render;

BufferDesc aliasBufferDesc(uint64_t bytes, BufferUsageBits extra = BufferUsageBits::None)
{
    return {.size = bytes,
        .usage = BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination |
            BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress | extra,
        .memoryDomain = MemoryBudgetDomain::FrameResources};
}

Result<> fillAliasBuffer(CommandBuffer& commands, Buffer& buffer, uint32_t value)
{
    if (auto retained = commands.retainResource(buffer.retainAllocation()); !retained) { return retained; }
    vulkan::nativeCommandBufferFunctions(commands).vkCmdFillBuffer(vulkan::nativeCommandBuffer(commands), vulkan::nativeBuffer(buffer).buffer,
        0, buffer.desc().size, value);
    return {};
}

class BufferAliasingNativeReadbackTest final : public RHITest {
public:
    BufferAliasingNativeReadbackTest()
    {
        type = RHITestType::Rendering;
        name = "buffer_aliasing_native_buffers_address_and_readback";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RHI,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"buffer.aliasing.nativeBuffers", "buffer.aliasing.deviceAddress",
                "buffer.aliasing.addressCommandFlagsMixedUsage", "buffer.aliasing.physicalCopyOverlap",
                "buffer.aliasing.handover.readback"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto secondDesc = aliasBufferDesc(4096, BufferUsageBits::Indirect);
        // Address commands must consider both overlapping native buffers even
        // when their Storage usage differs and one member is inactive.
        secondDesc.usage = BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination |
            BufferUsageBits::ShaderDeviceAddress | BufferUsageBits::Indirect;
        const std::array descriptions{aliasBufferDesc(8192), secondDesc};
        std::array<BufferAllocationRequirements, 2> requirements;
        for (size_t index = 0; index < requirements.size(); ++index) {
            const auto result = context.device.bufferAliasAllocationRequirements(descriptions[index])
                .transform([&](auto value) { requirements[index] = value; });
            if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Device buffer aliasing unsupported"); }
            if (!result || !requirements[index].sizeBytes || !requirements[index].alignmentBytes ||
                !requirements[index].memoryTypeBits) {
                return RHITestResult::fail("Buffer alias native requirements are missing or invalid");
            }
            const auto standalone = context.device.bufferAllocationSize(descriptions[index]);
            if (!standalone || *standalone != requirements[index].sizeBytes) {
                return RHITestResult::fail("Standalone and alias buffer requirement sizes disagree");
            }
        }
        if (!(requirements[0].memoryTypeBits & requirements[1].memoryTypeBits) ||
            requirements[0].requiresDedicatedAllocation || requirements[1].requiresDedicatedAllocation) {
            return RHITestResult::skip("Native buffer requirements cannot share non-dedicated backing");
        }
        std::vector<std::unique_ptr<Buffer>> buffers;
        if (!context.device.createAliasedBuffers(descriptions)
                .transform([&](auto value) { buffers = std::move(value); }) || buffers.size() != 2) {
            return RHITestResult::fail("Cannot create mixed size/usage independent alias buffers");
        }
        const auto first = buffers[0]->memoryInfo();
        const auto second = buffers[1]->memoryInfo();
        const auto firstNative = vulkan::nativeBuffer(*buffers[0]);
        const auto secondNative = vulkan::nativeBuffer(*buffers[1]);
        if (!first.known || !second.known || !first.allocationId || !second.allocationId ||
            first.allocationId == second.allocationId || !first.backingAllocationId ||
            first.backingAllocationId != second.backingAllocationId ||
            first.memoryBlockId != second.memoryBlockId || first.offsetBytes != second.offsetBytes ||
            first.backingSizeBytes != second.backingSizeBytes ||
            first.backingSizeBytes < std::max(requirements[0].sizeBytes, requirements[1].sizeBytes) ||
            first.sizeBytes != requirements[0].sizeBytes || second.sizeBytes != requirements[1].sizeBytes ||
            first.offsetBytes % requirements[0].alignmentBytes || second.offsetBytes % requirements[1].alignmentBytes ||
            first.memoryTypeIndex >= 32 || second.memoryTypeIndex >= 32 ||
            !(requirements[0].memoryTypeBits & (1u << first.memoryTypeIndex)) ||
            !(requirements[1].memoryTypeBits & (1u << second.memoryTypeIndex)) ||
            firstNative.buffer == secondNative.buffer || !firstNative.address || !secondNative.address ||
            firstNative.address != buffers[0]->deviceAddress() || secondNative.address != buffers[1]->deviceAddress()) {
            return RHITestResult::fail("Aliased buffers lost independent native handles, compatible backing metadata or BDA");
        }
        std::array<std::unique_ptr<Buffer>, 2> readbacks;
        for (size_t index = 0; index < readbacks.size(); ++index) {
            if (!context.device.createBuffer({.size = descriptions[index].size,
                    .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
                    .transform([&](auto value) { readbacks[index] = std::move(value); })) {
                return RHITestResult::fail("Cannot create native buffer alias readbacks");
            }
        }
        std::unique_ptr<CommandPool> pool;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); })) {
            return RHITestResult::fail("Cannot create native buffer alias command pool");
        }
        constexpr SyncScope allMemory{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
        for (uint32_t frame = 0; frame < 4; ++frame) {
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Fence> fence;
            if (!pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
                !context.device.createFence(false).transform([&](auto value) { fence = std::move(value); }) ||
                !commands->begin()) {
                return RHITestResult::fail("Cannot begin native buffer alias commands");
            }
            const auto overlapSource = buffers[0]->slice({.offset = 0, .size = 256});
            const auto overlapDestination = buffers[1]->slice({.offset = 128, .size = 256});
            constexpr auto domain = size_t(MemoryBudgetDomain::FrameResources);
            const auto beforeRejectedCopy = context.device.memoryBudget().domains[domain];
            if (!overlapSource || !overlapDestination ||
                !hasError(commands->copyBuffer(*overlapSource, *overlapDestination), Error::InvalidArgument)) {
                return RHITestResult::fail("Overlapping slices of distinct alias members were accepted for memory copy");
            }
            const auto afterRejectedCopy = context.device.memoryBudget().domains[domain];
            if (beforeRejectedCopy.allocationBytes != afterRejectedCopy.allocationBytes ||
                beforeRejectedCopy.allocationCount != afterRejectedCopy.allocationCount ||
                beforeRejectedCopy.deviceLocalBytes != afterRejectedCopy.deviceLocalBytes) {
                return RHITestResult::fail("Rejected physical overlap changed the backing memory budget");
            }
            for (size_t index = 0; index < buffers.size(); ++index) {
                const MemoryBarrierDesc handover{.before = allMemory, .after = allMemory};
                if (!commands->synchronize({.memory = {&handover, 1}}) ||
                    !fillAliasBuffer(*commands, *buffers[index], 0xa5010000u + frame * 16u + uint32_t(index))) {
                    return RHITestResult::fail("Cannot activate and initialize alias buffer");
                }
                const BufferBarrierDesc ready{.buffer = buffers[index].get(),
                    .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                const auto source = buffers[index]->slice();
                const auto destination = readbacks[index]->slice();
                if (!source || !destination || !commands->synchronize({.buffers = {&ready, 1}})) {
                    return RHITestResult::fail("Cannot prepare native alias buffer before handover");
                }
                if (index == 0) {
                    const auto disjointSource = buffers[0]->slice({.offset = 0, .size = 256});
                    const auto disjointDestination = buffers[1]->slice({.offset = 512, .size = 256});
                    const MemoryBarrierDesc physicalOrder{.before = allMemory, .after = allMemory};
                    if (!disjointSource || !disjointDestination ||
                        !commands->synchronize({.memory = {&physicalOrder, 1}})) {
                        return RHITestResult::fail("Cannot order the sentinel write after the full alias initialization");
                    }
                    // Corrupt the destination bytes first: the full-buffer
                    // oracle must fail if the following copy is not executed.
                    vulkan::nativeCommandBufferFunctions(*commands).vkCmdFillBuffer(vulkan::nativeCommandBuffer(*commands), secondNative.buffer, 512, 256, 0);
                    if (!commands->synchronize({.memory = {&physicalOrder, 1}}) ||
                        !commands->copyBuffer(*disjointSource, *disjointDestination) ||
                        !commands->synchronize({.memory = {&physicalOrder, 1}})) {
                        return RHITestResult::fail("Disjoint physical ranges in the same alias backing cannot copy");
                    }
                }
                if (!commands->copyBuffer(*source, *destination)) {
                    return RHITestResult::fail("Cannot copy native alias buffer before handover");
                }
            }
            const std::array<BufferBarrierDesc, 2> host{{
                {.buffer = readbacks[0].get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead}},
                {.buffer = readbacks[1].get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead}},
            }};
            if (!commands->synchronize({.buffers = host}) || !commands->end()) {
                return RHITestResult::fail("Cannot finish native buffer alias readback commands");
            }
            CommandBuffer* submitted[] = {commands.get()};
            if (!context.graphicsQueue.submit({.commandBuffers = submitted, .signalFence = fence.get()}) ||
                !fence->wait(5'000'000'000ull)) {
                (void)context.device.waitIdle();
                return RHITestResult::fail("Native buffer alias readback submission failed");
            }
            for (size_t index = 0; index < readbacks.size(); ++index) {
                readbacks[index]->invalidate();
                const auto* words = static_cast<const uint32_t*>(readbacks[index]->map());
                if (!words) { return RHITestResult::fail("Native buffer alias readback does not map"); }
                const uint32_t expected = 0xa5010000u + frame * 16u + uint32_t(index);
                const bool correct = std::all_of(words, words + descriptions[index].size / 4,
                    [expected](uint32_t value) { return value == expected; });
                readbacks[index]->unmap();
                if (!correct) { return RHITestResult::fail("Native buffer alias handover corrupted a readback word"); }
            }
            commands.reset();
            if (!pool->reset()) { return RHITestResult::fail("Cannot reset native buffer alias command pool"); }
        }
        return RHITestResult::pass("Distinct native buffers and valid BDA share compatible backing; four full-buffer readbacks survive alias handovers");
    }
};

class BufferAliasingOwnershipTest final : public RHITest {
public:
    BufferAliasingOwnershipTest()
    {
        type = RHITestType::Resource;
        name = "buffer_aliasing_backing_budget_and_retention";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RHI,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"buffer.aliasing.backing.budget", "buffer.aliasing.sliceAndCommandRetention",
                "buffer.aliasing.failureAtomicity"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr size_t domain = size_t(MemoryBudgetDomain::FrameResources);
        const auto initial = context.device.memoryBudget().domains[domain];
        std::vector<std::unique_ptr<Buffer>> rejected;
        auto failure = context.device.createAliasedBuffers({})
            .transform([&](auto value) { rejected = std::move(value); });
        if (!hasError(failure, Error::InvalidArgument) || !rejected.empty()) {
            return RHITestResult::fail("An empty native buffer alias group was accepted");
        }
        const std::array invalidDescriptions{aliasBufferDesc(8192), aliasBufferDesc(0)};
        failure = context.device.createAliasedBuffers(invalidDescriptions)
            .transform([&](auto value) { rejected = std::move(value); });
        auto host = aliasBufferDesc(4096);
        host.memoryLocation = MemoryLocation::HostUpload;
        const std::array hostDescriptions{aliasBufferDesc(8192), host};
        const auto hostFailure = context.device.createAliasedBuffers(hostDescriptions);
        const auto afterFailure = context.device.memoryBudget().domains[domain];
        if (!hasError(failure, Error::InvalidArgument) || !hasError(hostFailure, Error::Unsupported) ||
            !rejected.empty() || afterFailure.allocationBytes != initial.allocationBytes ||
            afterFailure.allocationCount != initial.allocationCount) {
            return RHITestResult::fail("Invalid/host buffer alias creation leaked a backing or budget charge");
        }
        const std::array descriptions{aliasBufferDesc(8192), aliasBufferDesc(4096)};
        std::vector<std::unique_ptr<Buffer>> buffers;
        const auto created = context.device.createAliasedBuffers(descriptions)
            .transform([&](auto value) { buffers = std::move(value); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Device buffer aliasing unsupported"); }
        if (!created || buffers.size() != 2) { return RHITestResult::fail("Cannot create buffer alias ownership group"); }
        const auto info = buffers[0]->memoryInfo();
        const auto live = context.device.memoryBudget().domains[domain];
        if (!info.backingSizeBytes || live.allocationBytes != initial.allocationBytes + info.backingSizeBytes ||
            live.allocationCount != initial.allocationCount + 1) {
            return RHITestResult::fail("Native alias buffers charged their backing more than once");
        }
        std::weak_ptr<void> firstOwner = buffers[0]->retainAllocation();
        std::weak_ptr<void> secondOwner = buffers[1]->retainAllocation();
        auto source = buffers[0]->slice({.size = 4096});
        auto destination = buffers[1]->slice();
        if (!source || !destination) { return RHITestResult::fail("Cannot create retaining alias buffer slices"); }
        std::unique_ptr<BufferView> view;
        if (context.device.capabilities().bindlessDescriptorHeap &&
            !context.device.createBufferView(*buffers[0], {.type = BufferViewType::Raw})
                .transform([&](auto value) { view = std::move(value); })) {
            return RHITestResult::fail("Cannot create retaining native alias buffer view");
        }
        Buffer moved(std::move(*buffers[1]));
        if (buffers[1]->memoryInfo().allocationId || moved.memoryInfo().backingAllocationId != info.backingAllocationId) {
            return RHITestResult::fail("Moving a native alias buffer lost backing ownership metadata");
        }
        moved = Buffer{};
        buffers.clear();
        if (firstOwner.expired() || secondOwner.expired() ||
            context.device.memoryBudget().domains[domain].allocationBytes != live.allocationBytes) {
            return RHITestResult::fail("Buffer slices failed to retain both native aliases and their common backing");
        }
        if (view) {
            source = BufferSlice{};
            if (firstOwner.expired() || context.device.memoryBudget().domains[domain].allocationBytes != live.allocationBytes) {
                return RHITestResult::fail("A native alias buffer view failed to retain its buffer and common backing");
            }
        }
        // Copy to separate host memory; copying between the aliases would overlap.
        std::unique_ptr<Buffer> readback;
        if (!context.device.createBuffer({.size = 4096, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback})
                .transform([&](auto value) { readback = std::move(value); })) {
            return RHITestResult::fail("Cannot create recorded-copy alias retention readback");
        }
        const auto readbackSlice = readback->slice();
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !commands->begin() || !commands->retainResource(firstOwner.lock()) ||
            !readbackSlice || !commands->copyBuffer(*destination, *readbackSlice) || !commands->end()) {
            return RHITestResult::fail("Cannot record retaining alias buffer commands");
        }
        source = BufferSlice{};
        destination = BufferSlice{};
        view.reset();
        if (firstOwner.expired() || secondOwner.expired() ||
            context.device.memoryBudget().domains[domain].allocationBytes != live.allocationBytes) {
            return RHITestResult::fail("Recorded commands lost native alias backing after all wrappers and slices were destroyed");
        }
        commands.reset();
        pool.reset();
        const auto final = context.device.memoryBudget().domains[domain];
        if (!firstOwner.expired() || !secondOwner.expired() || final.allocationBytes != initial.allocationBytes ||
            final.allocationCount != initial.allocationCount) {
            return RHITestResult::fail("Alias buffer release leaked or incorrectly released its single budget charge");
        }
        return RHITestResult::pass("One backing charge survives sibling destruction, moves, slices and recorded commands; failed alias groups are atomic");
    }
};

METALLIC_REGISTER_RHI_TEST(BufferAliasingNativeReadbackTest);
METALLIC_REGISTER_RHI_TEST(BufferAliasingOwnershipTest);

} // namespace
} // namespace metallic::tests
