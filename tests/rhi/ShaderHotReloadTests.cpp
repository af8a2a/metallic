#include "RhiTest.h"

#include "Runtime/Render/SlangCompiler.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace metallic::tests {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct ShaderHotReloadFixture {
    explicit ShaderHotReloadFixture(const std::filesystem::path& directory)
        : sourceDirectory(directory), dependencyPath(directory / "HotReloadValue.slang")
    {
        render::resetSlangShaderHotReloadTracking();
    }

    ~ShaderHotReloadFixture()
    {
        render::resetSlangShaderHotReloadTracking();
    }

    bool writeValue(uint32_t value) const
    {
        std::ofstream stream(dependencyPath, std::ios::binary | std::ios::trunc);
        stream << "static const uint kValue = " << value << "u;\n";
        stream.close();
        return static_cast<bool>(stream);
    }

    render::Result compile()
    {
        const std::string searchPath = sourceDirectory.string();
        return render::compileSlangShaderToSpirv(
            render::SlangShaderDesc{
                .moduleName = "HotReloadTest",
                .entryPointName = "main",
                .searchPath = searchPath.c_str(),
            },
            render::SlangShaderCacheOptions{.enableDiskCache = false},
            compiled);
    }

    bool initialize()
    {
        std::error_code error;
        std::filesystem::create_directories(sourceDirectory, error);
        if (error || !writeValue(100u)) {
            return false;
        }
        std::ofstream stream(sourceDirectory / "HotReloadTest.slang", std::ios::binary | std::ios::trunc);
        stream << "#include \"HotReloadValue.slang\"\n"
               << "RWStructuredBuffer<uint> outputBuffer;\n"
               << "[shader(\"compute\")] [numthreads(1, 1, 1)]\n"
               << "void main(uint3 id : SV_DispatchThreadID) { outputBuffer[id.x] = kValue; }\n";
        stream.close();
        return stream && compile().has_value();
    }

    std::vector<std::string> expectedChanges() const
    {
        return {std::filesystem::absolute(dependencyPath).lexically_normal().generic_string()};
    }

    bool waitForChange() const
    {
        const auto deadline = Clock::now() + 5s;
        while (Clock::now() < deadline) {
            const auto changes = render::pollSlangShaderChanges();
            if (!changes.empty()) {
                return changes == expectedChanges();
            }
            std::this_thread::sleep_for(10ms);
        }
        return false;
    }

    bool remainsQuiet(std::chrono::milliseconds duration) const
    {
        const auto deadline = Clock::now() + duration;
        do {
            if (!render::pollSlangShaderChanges().empty()) {
                return false;
            }
            std::this_thread::sleep_for(10ms);
        } while (Clock::now() < deadline);
        return true;
    }

    std::filesystem::path sourceDirectory;
    std::filesystem::path dependencyPath;
    render::ShaderCompileResult compiled;
};

class SlangShaderBackgroundHotReloadTest : public RhiTest {
public:
    SlangShaderBackgroundHotReloadTest()
    {
        type = RhiTestType::Resource;
        name = "slang_shader_background_hot_reload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        ShaderHotReloadFixture fixture(context.outputDirectory / name);
        if (!fixture.initialize()) {
            return RhiTestResult::fail("shader fixture initialization failed: " + fixture.compiled.diagnostics);
        }
        if (!fixture.remainsQuiet(100ms)) {
            return RhiTestResult::fail("unchanged registered sources produced a background change");
        }
        if (!fixture.writeValue(101u) || !fixture.remainsQuiet(75ms) || !fixture.waitForChange()) {
            return RhiTestResult::fail("background include detection did not preserve save debounce");
        }
        if (!fixture.remainsQuiet(100ms) || !fixture.waitForChange()) {
            return RhiTestResult::fail("unacknowledged background change did not preserve delayed retry");
        }

        // The caller is still compiling the delivered 101 snapshot while the
        // worker discovers 102. Acknowledging 101 must not swallow 102.
        if (!fixture.writeValue(102u)) {
            return RhiTestResult::fail("could not write source during a simulated reload");
        }
        std::this_thread::sleep_for(350ms);
        render::acknowledgeSlangShaderChanges();
        if (!fixture.waitForChange()) {
            return RhiTestResult::fail("acknowledgement swallowed an edit observed during reload");
        }
        render::acknowledgeSlangShaderChanges();
        if (!fixture.remainsQuiet(200ms)) {
            return RhiTestResult::fail("acknowledged source continued to produce changes");
        }

        std::error_code error;
        const auto writeTime = std::filesystem::last_write_time(fixture.dependencyPath, error);
        if (error || !fixture.writeValue(103u)) {
            return RhiTestResult::fail("could not prepare same-size content-only edit");
        }
        std::filesystem::last_write_time(fixture.dependencyPath, writeTime, error);
        if (error || !fixture.waitForChange()) {
            return RhiTestResult::fail("background hash missed a same-size edit with a restored timestamp");
        }
        render::acknowledgeSlangShaderChanges();

        if (!std::filesystem::remove(fixture.dependencyPath, error) || error || !fixture.waitForChange()) {
            return RhiTestResult::fail("deleted shader dependency was not reported");
        }
        // A missing include causes reload failure; recreating it must recover
        // without acknowledging the failed reload or editing the main module.
        if (!fixture.writeValue(104u) || !fixture.waitForChange()) {
            return RhiTestResult::fail("recreated include did not recover from an unacknowledged deletion");
        }
        render::acknowledgeSlangShaderChanges();
        return RhiTestResult::pass("validated background debounce, retry, acknowledgement, hashing and recreation");
    }
};

class SlangShaderBackgroundResetTest : public RhiTest {
public:
    SlangShaderBackgroundResetTest()
    {
        type = RhiTestType::Resource;
        name = "slang_shader_background_reset_and_reregistration";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        ShaderHotReloadFixture fixture(context.outputDirectory / name);
        if (!fixture.initialize() || !fixture.writeValue(101u)) {
            return RhiTestResult::fail("shader reset fixture initialization failed");
        }
        // Let the worker run while the frame consumer is absent, then discard
        // both queued changes and any scan still in flight.
        std::this_thread::sleep_for(350ms);
        render::resetSlangShaderHotReloadTracking();
        if (!fixture.remainsQuiet(200ms) || !fixture.compile()) {
            return RhiTestResult::fail("reset retained old changes or prevented dependency registration");
        }
        if (!fixture.remainsQuiet(200ms) || !fixture.writeValue(102u) || !fixture.waitForChange()) {
            return RhiTestResult::fail("re-registered dependency used a stale snapshot or stopped scanning");
        }
        render::acknowledgeSlangShaderChanges();

        // Returning to the accepted source before acknowledgement still needs
        // a reload after the delivered version has been committed.
        if (!fixture.writeValue(103u) || !fixture.waitForChange() || !fixture.writeValue(102u)) {
            return RhiTestResult::fail("could not prepare an edit reverted during reload");
        }
        std::this_thread::sleep_for(350ms);
        render::acknowledgeSlangShaderChanges();
        if (!fixture.waitForChange()) {
            return RhiTestResult::fail("revert during reload was lost after acknowledging the delivered version");
        }
        render::acknowledgeSlangShaderChanges();
        return RhiTestResult::pass("validated reset, re-registration and edits reverted during reload");
    }
};

METALLIC_REGISTER_RHI_TEST(SlangShaderBackgroundHotReloadTest);
METALLIC_REGISTER_RHI_TEST(SlangShaderBackgroundResetTest);

} // namespace
} // namespace metallic::tests
