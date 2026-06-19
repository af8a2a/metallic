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

METALLIC_REGISTER_RHI_TEST(ValidateDeviceTest);

} // namespace
} // namespace metallic::tests
