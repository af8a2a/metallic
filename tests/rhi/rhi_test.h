#pragma once

#include "Runtime/Render/GAPI/rhi.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace metallic::tests {

enum class RhiTestType {
    Validation,
    Resource,
    Command,
    Rendering,
};

struct RhiTestResult {
    bool passed = false;
    bool skipped = false;
    std::string message;

    static RhiTestResult pass(std::string message = {})
    {
        return RhiTestResult{true, false, std::move(message)};
    }

    static RhiTestResult fail(std::string message)
    {
        return RhiTestResult{false, false, std::move(message)};
    }

    static RhiTestResult skip(std::string message)
    {
        return RhiTestResult{false, true, std::move(message)};
    }
};

struct RhiTestContext {
    render::Device& device;
    render::Queue& graphicsQueue;
    std::filesystem::path outputDirectory;
    bool enableValidation = false;
};

class RhiTest {
public:
    RhiTestType type = RhiTestType::Validation;
    const char* name = nullptr;

    virtual ~RhiTest() = default;
    virtual void init(RhiTestContext&) {}
    virtual RhiTestResult run(RhiTestContext& context) = 0;
    virtual void cleanup(RhiTestContext&) {}
};

class RhiTestRegistry {
public:
    using Factory = std::function<std::unique_ptr<RhiTest>()>;

    static std::vector<Factory>& factories()
    {
        static std::vector<Factory> registeredFactories;
        return registeredFactories;
    }

    static void registerTest(Factory factory)
    {
        factories().push_back(std::move(factory));
    }

    static std::vector<std::unique_ptr<RhiTest>> createAll()
    {
        std::vector<std::unique_ptr<RhiTest>> tests;
        for (const Factory& factory : factories()) {
            tests.push_back(factory());
        }
        return tests;
    }
};

const char* toString(render::Result result);
const char* toString(RhiTestType type);
bool saveRgba8Png(
    const std::filesystem::path& outputPath,
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height,
    std::string& outMessage);

} // namespace metallic::tests

#define METALLIC_REGISTER_RHI_TEST(ClassName)                                      \
    static bool ClassName##Registered = []() {                                     \
        metallic::tests::RhiTestRegistry::registerTest(                            \
            []() { return std::make_unique<ClassName>(); });                       \
        return true;                                                               \
    }()
