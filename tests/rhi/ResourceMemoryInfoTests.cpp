#include "RhiTest.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <array>
#include <cstring>
#include <type_traits>

namespace metallic::tests {
namespace {

using namespace render;
static_assert(std::is_trivially_copyable_v<ResourceMemoryInfo>);

class ResourceMemoryInfoTest final : public RhiTest {
public:
    ResourceMemoryInfoTest() { type = RhiTestType::Resource; name = "resource_memory_info_identity_and_backing"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (Buffer{}.memoryInfo().known || Texture{}.memoryInfo().known || BufferSlice{}.memoryInfo().allocationId) {
            return RhiTestResult::fail("Empty resources report an allocation");
        }
        const auto heapCount = context.device.memoryBudget().heaps.size();
        const auto valid = [heapCount](const ResourceMemoryInfo& info, uint64_t minimumBytes) {
            return info.known && info.allocationId && info.memoryBlockId && info.sizeBytes >= minimumBytes &&
                info.offsetBytes <= UINT64_MAX - info.sizeBytes && info.memoryTypeIndex < 32 && info.heapIndex < heapCount;
        };
        std::unique_ptr<Buffer> first, second;
        const BufferDesc desc{.size = 8192, .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload};
        if (!context.device.createBuffer(desc).transform([&](auto value) { first = std::move(value); }) ||
            !context.device.createBuffer(desc).transform([&](auto value) { second = std::move(value); })) {
            return RhiTestResult::fail("Cannot allocate metadata buffers");
        }
        const auto firstInfo = first->memoryInfo();
        const auto secondInfo = second->memoryInfo();
        if (!valid(firstInfo, desc.size) || !valid(secondInfo, desc.size)) {
            return RhiTestResult::fail("Buffer backing metadata is incomplete");
        }
        BufferSlice slice;
        if (!first->slice(256, 1024).transform([&](auto value) { slice = std::move(value); })) {
            return RhiTestResult::fail("Cannot create metadata slice");
        }
        std::weak_ptr<void> owner = first->retainAllocation();
        Buffer moved(std::move(*first));
        const auto sliceInfo = slice.memoryInfo();
        if (first->memoryInfo().allocationId || moved.memoryInfo().allocationId != firstInfo.allocationId ||
            sliceInfo.allocationId != firstInfo.allocationId || sliceInfo.memoryBlockId != firstInfo.memoryBlockId ||
            sliceInfo.offsetBytes != firstInfo.offsetBytes || sliceInfo.sizeBytes != firstInfo.sizeBytes ||
            slice.offset() != 256 || slice.size() != 1024) {
            return RhiTestResult::fail("Move or slice changed backing identity or confused its range with the allocation");
        }
        moved = Buffer{};
        if (owner.expired()) { return RhiTestResult::fail("Slice stopped retaining the backing allocation"); }
        slice = {};
        if (!owner.expired()) { return RhiTestResult::fail("Value-only memory metadata retained an allocation"); }
        std::unique_ptr<Buffer> replacement;
        if (!context.device.createBuffer(desc).transform([&](auto value) { replacement = std::move(value); }) ||
            replacement->memoryInfo().allocationId == firstInfo.allocationId) {
            return RhiTestResult::fail("Replacement resource reused an allocation generation");
        }

        // Small material images can share a VMA block. Such live ranges must
        // remain disjoint; sharing a block is not physical memory aliasing.
        std::array<std::unique_ptr<Texture>, 2> textures;
        std::array<ResourceMemoryInfo, 3> live{secondInfo};
        const TextureDesc imageDesc{.usage = TextureUsageBits::Sampled, .format = Format::Rgba8Unorm,
            .width = 64, .height = 64, .memoryDomain = MemoryBudgetDomain::MaterialTextures};
        for (size_t i = 0; i < textures.size(); ++i) {
            if (!context.device.createTexture(imageDesc).transform([&](auto value) { textures[i] = std::move(value); })) {
                return RhiTestResult::fail("Cannot allocate metadata images");
            }
            live[i + 1] = textures[i]->memoryInfo();
            const auto native = vulkan::nativeTexture(*textures[i]);
            uint64_t nativeBlock = 0;
            static_assert(sizeof(native.memory) <= sizeof(nativeBlock));
            std::memcpy(&nativeBlock, &native.memory, sizeof(native.memory));
            if (!valid(live[i + 1], 64 * 64 * 4) || live[i + 1].sizeBytes != textures[i]->allocationSize() ||
                live[i + 1].memoryBlockId != nativeBlock) {
                return RhiTestResult::fail("Texture metadata does not describe its actual Vulkan backing");
            }
        }
        for (size_t i = 0; i < live.size(); ++i) {
            for (size_t j = i + 1; j < live.size(); ++j) {
                const auto& a = live[i];
                const auto& b = live[j];
                if (a.allocationId == b.allocationId ||
                    (a.memoryBlockId == b.memoryBlockId && a.offsetBytes < b.offsetBytes + b.sizeBytes &&
                        b.offsetBytes < a.offsetBytes + a.sizeBytes)) {
                    return RhiTestResult::fail("Distinct live allocations report aliased backing ranges");
                }
            }
        }
        std::weak_ptr<void> imageOwner = textures[0]->retainAllocation();
        Texture movedTexture(std::move(*textures[0]));
        if (textures[0]->memoryInfo().allocationId || movedTexture.memoryInfo().allocationId != live[1].allocationId) {
            return RhiTestResult::fail("Texture move changed its allocation generation");
        }
        movedTexture = Texture{};
        if (!imageOwner.expired()) { return RhiTestResult::fail("Texture memory snapshot retained GPU backing"); }
        return RhiTestResult::pass("VMA backing ranges, heap identity, moves, slices, generations and non-owning snapshots");
    }
};

METALLIC_REGISTER_RHI_TEST(ResourceMemoryInfoTest);

} // namespace
} // namespace metallic::tests
