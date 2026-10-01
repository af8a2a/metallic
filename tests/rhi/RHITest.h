#pragma once

#include <atomic>

#include "Runtime/Render/GAPI/RHI.h"
#include "harness/Requirements.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace metallic::render::profiling {
class NsightGraphicsCapture;
} // namespace metallic::render::profiling

namespace metallic::tests {

namespace bench { class Evidence; class TraceRecorder; }

enum class RHITestType {
    Validation,
    Resource,
    Command,
    Rendering,
};

struct RHITestResult {
    bool passed = false;
    bool skipped = false;
    std::string message;

    static RHITestResult pass(std::string message = {})
    {
        return RHITestResult{true, false, std::move(message)};
    }

    static RHITestResult fail(std::string message)
    {
        return RHITestResult{false, false, std::move(message)};
    }

    static RHITestResult skip(std::string message)
    {
        return RHITestResult{false, true, std::move(message)};
    }
};

struct RHITestContext {
    render::Device& device;
    render::Queue& graphicsQueue;
    std::filesystem::path outputDirectory;
    bool enableValidation = false;
    std::atomic_uint* validationMessageCount = nullptr;
    render::profiling::NsightGraphicsCapture* nsightCapture = nullptr;
    bench::Evidence* evidence = nullptr;
    const render::DeviceDesc* deviceDesc = nullptr;
    bench::TraceRecorder* trace = nullptr;
};

class RHITest {
public:
    RHITestType type = RHITestType::Validation;
    const char* name = nullptr;

    virtual ~RHITest() = default;
    virtual std::optional<bench::Metadata> metadata() const { return std::nullopt; }
    virtual RHITestResult runCpu(bench::Evidence&) { return RHITestResult::fail("CPU entry point is not implemented"); }
    virtual void cleanupCpu() {}
    virtual void init(RHITestContext&) {}
    virtual RHITestResult run(RHITestContext& context) = 0;
    virtual void cleanup(RHITestContext&) {}
};

class RHITestRegistry {
public:
    using Factory = std::function<std::unique_ptr<RHITest>()>;

    static std::vector<Factory>& factories()
    {
        static std::vector<Factory> registeredFactories;
        return registeredFactories;
    }

    static void registerTest(Factory factory)
    {
        factories().push_back(std::move(factory));
    }

    static std::vector<std::unique_ptr<RHITest>> createAll()
    {
        std::vector<std::unique_ptr<RHITest>> tests;
        for (const Factory& factory : factories()) {
            tests.push_back(factory());
        }
        return tests;
    }
};

const char* toString(render::Result<> result);
const char* toString(RHITestType type);
bool saveRgba8Png(
    const std::filesystem::path& outputPath,
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height,
    std::string& outMessage);

} // namespace metallic::tests

#define METALLIC_REGISTER_RHI_TEST(ClassName)                                      \
    static bool ClassName##Registered = []() {                                     \
        metallic::tests::RHITestRegistry::registerTest(                            \
            []() { return std::make_unique<ClassName>(); });                       \
        return true;                                                               \
    }()
