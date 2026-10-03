#include "RHITest.h"
#include "Runtime/Render/GAPI/TextureFormat.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanResult.h"

#include <array>
#include <limits>

namespace metallic::tests {
namespace {

#define REFINE_CHECK(expression) do { if (!(expression)) { return RHITestResult::fail(#expression); } } while (false)

class TextureCopyFootprintTest : public RHITest {
public:
    TextureCopyFootprintTest()
    {
        name = "texture_copy_footprint_validation";
    }

    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        const auto packed = textureCopyFootprint(Format::BGRA4Unorm, 3, 2, 2, 8, 24);
        REFINE_CHECK(packed && packed->rowBytes == 6 && packed->rows == 2 &&
            packed->requiredBytes == 38);
        const auto compressed = textureCopyFootprint(Format::BC7Unorm, 5, 7, 3, 48, 144);
        REFINE_CHECK(compressed && compressed->rowBytes == 32 && compressed->rows == 2 &&
            compressed->requiredBytes == 368);
        // CPU rows may have arbitrary padding, independently of native copy alignment.
        const auto cpu = textureCopyFootprint(Format::BGRA4Unorm, 3, 2, 1, 7, 15);
        REFINE_CHECK(cpu && cpu->requiredBytes == 13);
        const auto large = textureCopyFootprint(Format::R8Unorm, 1, UINT32_MAX, 1, 2);
        REFINE_CHECK(large && large->slicePitch == uint64_t(UINT32_MAX) * 2);
        REFINE_CHECK(!textureCopyFootprint(Format::Unknown, 1, 1));
        REFINE_CHECK(!textureCopyFootprint(Format::R8Unorm, 0, 1));
        REFINE_CHECK(!textureCopyFootprint(Format::R8Unorm, 1, 1, 0));
        REFINE_CHECK(!textureCopyFootprint(Format::BGRA4Unorm, 3, 2, 1, 5));
        REFINE_CHECK(!textureCopyFootprint(Format::BGRA4Unorm, 3, 2, 1, 8, 15));
        REFINE_CHECK(!textureCopyFootprint(Format::R8Unorm, 1, 2, 1, UINT64_MAX));
        REFINE_CHECK(!textureCopyFootprint(Format::R8Unorm, 1, 1, 3, 1, UINT64_MAX));
        return RHITestResult::pass();
    }
};

class VulkanResultMappingTest : public RHITest {
public:
    VulkanResultMappingTest()
    {
        name = "vulkan_result_mapping";
    }

    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        using vulkan::mapVkResult;
        REFINE_CHECK(mapVkResult(VK_SUCCESS));
        for (auto code : {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY}) {
            REFINE_CHECK(hasError(mapVkResult(code), Error::OutOfMemory));
        }
        for (auto code : {VK_ERROR_EXTENSION_NOT_PRESENT, VK_ERROR_FEATURE_NOT_PRESENT,
            VK_ERROR_FORMAT_NOT_SUPPORTED, VK_ERROR_INCOMPATIBLE_DRIVER, VK_ERROR_LAYER_NOT_PRESENT}) {
            REFINE_CHECK(hasError(mapVkResult(code), Error::Unsupported));
        }
        REFINE_CHECK(hasError(mapVkResult(VK_ERROR_DEVICE_LOST), Error::DeviceLost));
        REFINE_CHECK(hasError(mapVkResult(VK_ERROR_OUT_OF_DATE_KHR), Error::OutOfDate));
        REFINE_CHECK(hasError(mapVkResult(VK_ERROR_SURFACE_LOST_KHR), Error::OutOfDate));
        REFINE_CHECK(hasError(mapVkResult(VK_ERROR_UNKNOWN), Error::Failure));
        return RHITestResult::pass();
    }
};

class CommandBindingValidationTest : public RHITest {
public:
    CommandBindingValidationTest()
    {
        name = "command_binding_validation";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.requirements = {.capabilities = {bench::Capability::Bindless}}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        auto pool = context.device.createCommandPool(context.graphicsQueue);
        REFINE_CHECK(pool);
        auto commands = (*pool)->createCommandBuffer();
        REFINE_CHECK(commands);
        auto heap = context.device.createBindlessHeap({.maxBuffers = 1});
        REFINE_CHECK(heap);
        const uint32_t value = 42;
        auto invalid = [](Result<> result) { return hasError(result, Error::InvalidArgument); };
        REFINE_CHECK(invalid((*commands)->bindBindlessHeap(**heap)));
        REFINE_CHECK(invalid((*commands)->pushBindlessData(&value, sizeof(value))));
        REFINE_CHECK(invalid((*commands)->dispatch(0)));
        REFINE_CHECK((*commands)->begin());
        BindlessHeap emptyHeap;
        REFINE_CHECK(invalid((*commands)->bindBindlessHeap(emptyHeap)));
        REFINE_CHECK((*commands)->bindBindlessHeap(**heap));
        REFINE_CHECK((*commands)->pushBindlessData(&value, sizeof(value)));
        REFINE_CHECK(invalid((*commands)->pushBindlessData(nullptr, 4)));
        REFINE_CHECK(invalid((*commands)->pushBindlessData(&value, 3)));
        REFINE_CHECK(invalid((*commands)->pushBindlessData(&value, 0xfffffffcu)));
        REFINE_CHECK((*commands)->pushBindlessData(nullptr, 0));
        REFINE_CHECK(invalid((*commands)->dispatch(UINT32_MAX)));
        REFINE_CHECK((*commands)->dispatch(0)); // No pipeline needed for the existing no-op contract.
        REFINE_CHECK((*commands)->dispatch(0, UINT32_MAX, 1));
        auto other = createDevice({.applicationName = "Foreign heap validation",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        REFINE_CHECK(other);
        auto foreignHeap = (*other)->createBindlessHeap({.maxBuffers = 1});
        REFINE_CHECK(foreignHeap);
        REFINE_CHECK(invalid((*commands)->bindBindlessHeap(**foreignHeap)));
        REFINE_CHECK((*commands)->end());
        REFINE_CHECK(invalid((*commands)->pushBindlessData(&value, sizeof(value))));
        REFINE_CHECK(invalid((*commands)->dispatch(0)));
        if (context.device.capabilities().independentCopyQueue) {
            auto copyPool = context.device.createCommandPool(*context.device.getQueue(QueueType::Copy));
            REFINE_CHECK(copyPool);
            auto copy = (*copyPool)->createCommandBuffer();
            REFINE_CHECK(copy && (*copy)->begin());
            REFINE_CHECK(invalid((*copy)->bindBindlessHeap(**heap)));
            REFINE_CHECK(invalid((*copy)->pushBindlessData(&value, sizeof(value))));
            REFINE_CHECK(invalid((*copy)->dispatch(0)));
            REFINE_CHECK((*copy)->end());
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(TextureCopyFootprintTest);
METALLIC_REGISTER_RHI_TEST(VulkanResultMappingTest);
METALLIC_REGISTER_RHI_TEST(CommandBindingValidationTest);

#undef REFINE_CHECK
} // namespace
} // namespace metallic::tests

