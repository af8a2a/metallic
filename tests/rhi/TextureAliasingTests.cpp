#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

using namespace render;

TextureDesc aliasTextureDesc(uint32_t extent, Format format = Format::RGBA8Unorm)
{
    return {.usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
        .format = format, .width = extent, .height = extent,
        .memoryDomain = MemoryBudgetDomain::FrameResources};
}

Result<> transitionAliasTexture(CommandBuffer& commands, Texture& texture,
    TextureLayout oldLayout, TextureLayout newLayout, SyncScope before, SyncScope after)
{
    const TextureBarrierDesc barrier{.texture = &texture, .oldLayout = oldLayout,
        .newLayout = newLayout, .before = before, .after = after,
        .range = {.mipCount = 1, .layerCount = 1}};
    return commands.synchronize({.textures = {&barrier, 1}});
}

Result<> clearAliasTexture(CommandBuffer& commands, Texture& texture, TextureView& view, ColorValue color)
{
    const RenderingAttachmentDesc attachment{.view = &view, .loadOp = LoadOp::Clear,
        .storeOp = StoreOp::Store, .clearColor = color};
    auto result = commands.beginRendering({.renderArea = {.width = texture.desc().width,
        .height = texture.desc().height}, .colorAttachments = {&attachment, 1}});
    if (result) { commands.endRendering(); }
    return result;
}

class TextureAliasingReadbackTest final : public RHITest {
public:
    TextureAliasingReadbackTest()
    {
        type = RHITestType::Rendering;
        name = "texture_aliasing_native_images_and_readback";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RHI,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"texture.aliasing.nativeImages", "texture.aliasing.handover.readback"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        const std::array descriptions{aliasTextureDesc(32), aliasTextureDesc(16, Format::BGRA8Unorm)};
        std::array<TextureAllocationRequirements, 2> requirements;
        for (size_t index = 0; index < requirements.size(); ++index) {
            auto result = context.device.textureAliasAllocationRequirements(descriptions[index])
                .transform([&](auto value) { requirements[index] = value; });
            if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Alias-capable images unsupported"); }
            if (!result) { return RHITestResult::fail("Alias requirement query failed"); }
            if (!requirements[index].sizeBytes || !requirements[index].alignmentBytes ||
                !requirements[index].memoryTypeBits) {
                return RHITestResult::fail("Alias requirement query returned an incomplete requirement");
            }
        }
        if ((requirements[0].memoryTypeBits & requirements[1].memoryTypeBits) == 0 ||
            requirements[0].requiresDedicatedAllocation || requirements[1].requiresDedicatedAllocation) {
            return RHITestResult::skip("Formats require incompatible or dedicated backing memory");
        }
        std::vector<std::unique_ptr<Texture>> textures;
        if (!context.device.createAliasedTextures(descriptions)
                .transform([&](auto value) { textures = std::move(value); }) || textures.size() != 2) {
            return RHITestResult::fail("Cannot create differently sized and formatted alias images");
        }
        const auto first = textures[0]->memoryInfo();
        const auto second = textures[1]->memoryInfo();
        const auto firstNative = vulkan::nativeTexture(*textures[0]);
        const auto secondNative = vulkan::nativeTexture(*textures[1]);
        if (!first.known || !second.known || !first.allocationId || !second.allocationId ||
            first.allocationId == second.allocationId || !first.backingAllocationId ||
            first.backingAllocationId != second.backingAllocationId ||
            first.memoryBlockId != second.memoryBlockId || first.offsetBytes != second.offsetBytes ||
            first.backingSizeBytes != second.backingSizeBytes ||
            first.backingSizeBytes < std::max(requirements[0].sizeBytes, requirements[1].sizeBytes) ||
            first.sizeBytes != requirements[0].sizeBytes || second.sizeBytes != requirements[1].sizeBytes ||
            first.offsetBytes % requirements[0].alignmentBytes ||
            second.offsetBytes % requirements[1].alignmentBytes ||
            first.memoryTypeIndex >= 32 || second.memoryTypeIndex >= 32 ||
            !(requirements[0].memoryTypeBits & (1u << first.memoryTypeIndex)) ||
            !(requirements[1].memoryTypeBits & (1u << second.memoryTypeIndex)) ||
            firstNative.image == secondNative.image || firstNative.memory != secondNative.memory) {
            return RHITestResult::fail("Alias images do not have distinct Vulkan objects sharing one compatible backing range");
        }
        std::array<std::unique_ptr<TextureView>, 2> views;
        std::array<std::unique_ptr<Buffer>, 2> readbacks;
        for (size_t index = 0; index < textures.size(); ++index) {
            if (!context.device.createTextureView(*textures[index], {.format = descriptions[index].format,
                    .range = {.mipCount = 1, .layerCount = 1}})
                    .transform([&](auto value) { views[index] = std::move(value); }) ||
                !context.device.createBuffer({.size = uint64_t(descriptions[index].width) * descriptions[index].height * 4,
                    .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
                    .transform([&](auto value) { readbacks[index] = std::move(value); })) {
                return RHITestResult::fail("Cannot create alias image views and readbacks");
            }
        }
        std::unique_ptr<CommandPool> pool;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); })) {
            return RHITestResult::fail("Cannot create alias test command pool");
        }
        constexpr SyncScope colorWrite{PipelineStageBits::ColorAttachment, AccessBits::ColorWrite};
        constexpr SyncScope transferRead{PipelineStageBits::Transfer, AccessBits::TransferRead};
        constexpr SyncScope allMemory{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
        for (uint32_t frame = 0; frame < 4; ++frame) {
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Fence> fence;
            if (!pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
                !context.device.createFence(false).transform([&](auto value) { fence = std::move(value); }) ||
                !commands->begin()) {
                return RHITestResult::fail("Cannot begin alias readback commands");
            }
            for (size_t index = 0; index < textures.size(); ++index) {
                if (frame != 0 || index != 0) {
                    const MemoryBarrierDesc handover{.before = allMemory, .after = allMemory};
                    if (!commands->synchronize({.memory = {&handover, 1}})) {
                        return RHITestResult::fail("Cannot synchronize physical-memory handover");
                    }
                }
                if (!transitionAliasTexture(*commands, *textures[index], TextureLayout::Undefined,
                        TextureLayout::ColorAttachment, {}, colorWrite)) {
                    return RHITestResult::fail("Cannot initialize alias image layout");
                }
                const bool red = ((frame + index) % 2) == 0;
                if (!clearAliasTexture(*commands, *textures[index], *views[index],
                        red ? ColorValue{1, 0, 0, 1} : ColorValue{0, 1, 0, 1}) ||
                    !transitionAliasTexture(*commands, *textures[index], TextureLayout::ColorAttachment,
                        TextureLayout::TransferSource, colorWrite, transferRead)) {
                    return RHITestResult::fail("Cannot clear and read the current alias image");
                }
                if (auto commandResult = (readbacks[index].get())->slice().and_then([&](const auto& bufferSlice) { return commands->copyTextureToBuffer({.texture = textures[index].get(), .buffer = bufferSlice,
                    .width = descriptions[index].width, .height = descriptions[index].height}); }); !commandResult) { return RHITestResult::fail(render::resultToString(commandResult)); }
            }
            const std::array<BufferBarrierDesc, 2> hostBarriers{{
                {.buffer = readbacks[0].get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead}},
                {.buffer = readbacks[1].get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead}},
            }};
            if (!commands->synchronize({.buffers = hostBarriers}) || !commands->end()) {
                return RHITestResult::fail("Cannot finish alias readback commands");
            }
            CommandBuffer* submitted[] = {commands.get()};
            if (!context.graphicsQueue.submit({.commandBuffers = submitted, .signalFence = fence.get()}) ||
                !fence->wait(5'000'000'000ull)) {
                (void)context.device.waitIdle();
                return RHITestResult::fail("Alias readback submission failed");
            }
            for (size_t index = 0; index < readbacks.size(); ++index) {
                readbacks[index]->invalidate();
                const auto* bytes = static_cast<const uint8_t*>(readbacks[index]->map());
                if (!bytes) { return RHITestResult::fail("Alias readback did not map"); }
                const bool red = ((frame + index) % 2) == 0;
                std::array<uint8_t, 4> expected = red ? std::array<uint8_t, 4>{255, 0, 0, 255}
                    : std::array<uint8_t, 4>{0, 255, 0, 255};
                if (descriptions[index].format == Format::BGRA8Unorm) { std::swap(expected[0], expected[2]); }
                bool matches = true;
                for (size_t pixel = 0; pixel < size_t(descriptions[index].width) * descriptions[index].height; ++pixel) {
                    matches &= std::memcmp(bytes + pixel * 4, expected.data(), 4) == 0;
                }
                readbacks[index]->unmap();
                if (!matches) { return RHITestResult::fail("Alias handover preserved stale or corrupted pixels"); }
            }
            commands.reset();
            if (!pool->reset()) { return RHITestResult::fail("Cannot reset completed alias command pool"); }
        }
        return RHITestResult::pass("Distinct Vulkan images share one compatible range; four alternating frames preserve every pixel");
    }
};

class TextureAliasingOwnershipTest final : public RHITest {
public:
    TextureAliasingOwnershipTest()
    {
        type = RHITestType::Resource;
        name = "texture_aliasing_backing_budget_and_retention";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RHI,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"texture.aliasing.backing.budget", "texture.aliasing.retention", "texture.aliasing.failureAtomicity"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr size_t domain = size_t(MemoryBudgetDomain::FrameResources);
        const auto initial = context.device.memoryBudget().domains[domain];
        std::vector<std::unique_ptr<Texture>> rejected;
        auto invalidResult = context.device.createAliasedTextures({})
            .transform([&](auto value) { rejected = std::move(value); });
        if (!hasError(invalidResult, Error::InvalidArgument) || !rejected.empty()) {
            return RHITestResult::fail("An empty alias group was accepted");
        }
        auto invalid = aliasTextureDesc(0);
        const std::array invalidDescriptions{aliasTextureDesc(32), invalid};
        invalidResult = context.device.createAliasedTextures(invalidDescriptions)
            .transform([&](auto value) { rejected = std::move(value); });
        if (!hasError(invalidResult, Error::InvalidArgument) || !rejected.empty() ||
            context.device.memoryBudget().domains[domain].allocationBytes != initial.allocationBytes ||
            context.device.memoryBudget().domains[domain].allocationCount != initial.allocationCount) {
            return RHITestResult::fail("A failed alias group leaked an allocation or budget charge");
        }
        const std::array descriptions{aliasTextureDesc(32), aliasTextureDesc(16)};
        std::vector<std::unique_ptr<Texture>> textures;
        auto result = context.device.createAliasedTextures(descriptions)
            .transform([&](auto value) { textures = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Alias-capable images unsupported"); }
        if (!result || textures.size() != 2) { return RHITestResult::fail("Cannot create ownership test alias group"); }
        const auto info = textures[0]->memoryInfo();
        const auto live = context.device.memoryBudget().domains[domain];
        if (!info.backingSizeBytes || live.allocationBytes != initial.allocationBytes + info.backingSizeBytes ||
            live.allocationCount != initial.allocationCount + 1) {
            return RHITestResult::fail("Aliased images charged their shared backing more than once");
        }
        std::unique_ptr<TextureView> view;
        if (!context.device.createTextureView(*textures[0], {.format = descriptions[0].format,
                .range = {.mipCount = 1, .layerCount = 1}})
                .transform([&](auto value) { view = std::move(value); })) {
            return RHITestResult::fail("Cannot create retaining alias view");
        }
        std::weak_ptr<void> firstOwner = textures[0]->retainAllocation();
        std::weak_ptr<void> secondOwner = textures[1]->retainAllocation();
        Texture moved(std::move(*textures[1]));
        if (textures[1]->memoryInfo().allocationId || moved.memoryInfo().backingAllocationId != info.backingAllocationId) {
            return RHITestResult::fail("Texture move lost alias ownership metadata");
        }
        moved = Texture{};
        textures.clear();
        if (firstOwner.expired() || !secondOwner.expired() ||
            context.device.memoryBudget().domains[domain].allocationBytes != live.allocationBytes) {
            return RHITestResult::fail("An alias view did not preserve its image and shared backing independently");
        }
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !commands->begin() || !commands->useNativeTextureView(*view) || !commands->end()) {
            return RHITestResult::fail("Cannot record a retained alias view");
        }
        view.reset();
        if (firstOwner.expired() || context.device.memoryBudget().domains[domain].allocationBytes != live.allocationBytes) {
            return RHITestResult::fail("Recorded commands did not retain shared alias backing after wrappers were destroyed");
        }
        commands.reset();
        pool.reset();
        const auto final = context.device.memoryBudget().domains[domain];
        if (!firstOwner.expired() || final.allocationBytes != initial.allocationBytes ||
            final.allocationCount != initial.allocationCount) {
            return RHITestResult::fail("Shared alias backing was leaked or its budget was released incorrectly");
        }
        return RHITestResult::pass("One backing charge survives sibling destruction, moves, views and command recording; failed groups are atomic");
    }
};

METALLIC_REGISTER_RHI_TEST(TextureAliasingReadbackTest);
METALLIC_REGISTER_RHI_TEST(TextureAliasingOwnershipTest);

} // namespace
} // namespace metallic::tests
