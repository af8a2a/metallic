#include "RhiTest.h"

namespace metallic::tests {
namespace {

class ValidateDeviceTest : public RhiTest {
public:
    ValidateDeviceTest()
    {
        type = RhiTestType::Validation;
        name = "validate_device";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::Queue* graphicsQueue = context.device.getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("graphics queue is unavailable");
        }
        if (graphicsQueue->type() != render::QueueType::Graphics) {
            return RhiTestResult::fail("graphics queue reported the wrong type");
        }

        render::Result result = context.device.waitIdle();
        if (!result) {
            return RhiTestResult::fail(std::string("Device::waitIdle returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

class OptionalFeatureSoftRequestTest : public RhiTest {
public:
    OptionalFeatureSoftRequestTest()
    {
        type = RhiTestType::Validation;
        name = "optional_feature_soft_request";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RHI Optional Feature Soft Request Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
                .enableRayTracingAccelerationStructure = true,
                .enableRayQuery = true,
                .enablePushDescriptor = true,
            },
            device);
        if (!result) {
            return RhiTestResult::fail(
                std::string("createDevice(optional features) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RhiTestResult::fail("createDevice(optional features) returned a null device");
        }

        const render::DeviceCapabilities& capabilities = device->capabilities();
        if (capabilities.rayQuery && !capabilities.rayTracingAccelerationStructure) {
            return RhiTestResult::fail("rayQuery capability was enabled without acceleration structure support");
        }
        if (capabilities.bindlessDescriptorHeap &&
            (capabilities.maxBindlessSamplers == 0 ||
                capabilities.maxBindlessSampledImages == 0 ||
                capabilities.maxBindlessBuffers == 0)) {
            return RhiTestResult::fail("bindless descriptor heap capability reported zero capacity");
        }

        result = device->waitIdle();
        if (!result) {
            return RhiTestResult::fail(
                std::string("Device::waitIdle(optional features) returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(ValidateDeviceTest);
METALLIC_REGISTER_RHI_TEST(OptionalFeatureSoftRequestTest);

} // namespace
} // namespace metallic::tests
