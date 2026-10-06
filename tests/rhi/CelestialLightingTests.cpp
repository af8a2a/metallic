#include "RHITest.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"

#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

std::vector<float> celestialHDR(const render::RenderGraphPreviewRenderer& preview)
{
    const auto& bytes = preview.readbackBytes();
    std::vector<float> result;
    if (preview.readbackFormat() == render::Format::RGBA32Sfloat) {
        result.resize(bytes.size() / sizeof(float));
        std::memcpy(result.data(), bytes.data(), bytes.size());
    } else if (preview.readbackFormat() == render::Format::RGBA16Sfloat) {
        result.reserve(bytes.size() / sizeof(uint16_t));
        for (size_t i = 0; i < bytes.size(); i += 2) {
            uint16_t value;
            std::memcpy(&value, bytes.data() + i, sizeof(value));
            const uint32_t exponent = (value >> 10) & 31u;
            const float magnitude = exponent == 0 ? std::ldexp(float(value & 1023u), -24)
                : exponent == 31 ? INFINITY : std::ldexp(float(1024u + (value & 1023u)), int(exponent) - 25);
            result.push_back((value & 0x8000u) != 0 ? -magnitude : magnitude);
        }
    }
    return result;
}

class CelestialLightingRenderingTest final : public RHITest {
public:
    CelestialLightingRenderingTest() { name = "celestial_raster_pt_fixed_slots"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        RenderSampleLoadResult sample;
        std::string log;
        if (!loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        scene::LightingSettings local;
        local.autoExposure.enabled = false;
        for (auto path : {LookDevRenderPath::PathTraceOnly, LookDevRenderPath::DeferredOnly}) {
            const bool pt = path == LookDevRenderPath::PathTraceOnly;
            const std::string label = pt ? "PathTrace" : "Deferred";
            RenderGraph graph;
            if (!makeLookDevRenderGraph(sample.graph, path, graph, log)) { return RHITestResult::fail(log); }
            std::array<std::vector<float>, 4> images;
            for (uint32_t mode = 0; mode < 4; ++mode) {
                RenderGraphPreviewRenderer preview;
                preview.bindRuntimeScene(&document);
                preview.setEnvironment({.enabled = false});
                preview.setLighting(local);
                environment::WorldEnvironment celestial;
                celestial.sun = document.worldEnvironment().sun;
                celestial.sun.enabled = (mode & 1u) != 0;
                celestial.sun.illuminance = 3.0f;
                celestial.moon = celestial.sun;
                celestial.moon.enabled = (mode & 2u) != 0;
                preview.setWorldEnvironment(celestial);
                if (!preview.initialize(context.enableValidation, true, false)) {
                    return RHITestResult::fail("Celestial preview setup failed");
                }
                for (uint32_t frame = 0; frame < 32; ++frame) {
                    preview.setRawReadbackEnabled(frame == 31);
                    if (!preview.render(graph, 128, 128, pt ? "Reference.color" : "Deferred.color", frame == 31)) {
                        return RHITestResult::fail(preview.lastLog());
                    }
                }
                images[mode] = celestialHDR(preview);
                if (images[mode].size() != 128 * 128 * 4) { return RHITestResult::fail("Celestial HDR readback missing"); }
                for (float value : images[mode]) {
                    if (!std::isfinite(value)) { return RHITestResult::fail("Nonfinite celestial HDR"); }
                }
                std::ofstream raw(context.outputDirectory / (label + std::to_string(mode) + ".rgba32f"), std::ios::binary);
                raw.write(reinterpret_cast<const char*>(images[mode].data()), images[mode].size() * sizeof(float));
                if (!raw) { return RHITestResult::fail("Celestial HDR evidence write failed"); }
            }
            double contribution = 0, independentError = 0, sumError = 0;
            for (size_t i = 0; i < images[0].size(); ++i) {
                if ((i & 3u) == 3u) { continue; }
                const double sun = images[1][i] - images[0][i];
                const double moon = images[2][i] - images[0][i];
                contribution += std::abs(sun);
                independentError += std::abs(sun - moon);
                sumError += std::abs(double(images[3][i]) - images[0][i] - sun - moon);
            }
            if (contribution <= 1.0 || independentError / contribution > 0.01 || sumError / contribution > 0.03) {
                return RHITestResult::fail(label + " fixed-slot independence/additivity failed: contribution=" +
                    std::to_string(contribution) + ", independent=" + std::to_string(independentError / contribution) +
                    ", additive=" + std::to_string(sumError / contribution));
            }
        }
        return RHITestResult::pass("32-frame raster/PT HDR: disabled, Sun, Moon, both; finite, independent and additive");
    }
};
METALLIC_REGISTER_RHI_TEST(CelestialLightingRenderingTest);

class CelestialPublicationTest final : public RHITest {
public:
    CelestialPublicationTest() { name = "celestial_publication_override_reuse"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        if (!context.device.createCommandPool(context.graphicsQueue)
                .transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !commands->begin()) { return RHITestResult::fail("Celestial publication recording setup failed"); }
        RenderSubsystemHost host;
        EnvironmentLightingSubsystem subsystem;
        environment::WorldEnvironment world;
        world.sun.enabled = true;
        world.sun.illuminance = 10;
        CelestialLightingResources first, other, reused;
        if (!subsystem.updateCelestial(context.device, *commands, host, world.snapshot())
                .transform([&](auto value) { first = std::move(value); })) {
            return RHITestResult::fail("Initial celestial publication failed");
        }
        world.sun.illuminance = 20;
        if (!subsystem.updateCelestial(context.device, *commands, host, world.snapshot())
                .transform([&](auto value) { other = std::move(value); })) {
            return RHITestResult::fail("Override celestial publication failed");
        }
        world.sun.illuminance = 10;
        if (!subsystem.updateCelestial(context.device, *commands, host, world.snapshot())
                .transform([&](auto value) { reused = std::move(value); }) ||
            reused.buffer != first.buffer || reused.revision != first.revision || other.buffer == first.buffer) {
            return RHITestResult::fail("World/scene override changed an unchanged publication identity");
        }
        for (uint32_t i = 0; i < 12; ++i) {
            world.sun.illuminance = 30.0f + i;
            if (!subsystem.updateCelestial(context.device, *commands, host, world.snapshot())) {
                return RHITestResult::fail("Bounded celestial publication cache failed");
            }
        }
        const auto* retained = static_cast<const GPUCelestialLight*>(first.buffer->map());
        if (retained == nullptr) { return RHITestResult::fail("Retained celestial publication was destroyed"); }
        const float irradiance = retained[0].irradiance[0];
        first.buffer->unmap();
        if (std::abs(irradiance - 10.0f) > 0.001f || !commands->end()) {
            return RHITestResult::fail("Immutable celestial publication changed after cache eviction");
        }
        return RHITestResult::pass("Alternating overrides reuse identity; retained records survive bounded-cache eviction");
    }
};
METALLIC_REGISTER_RHI_TEST(CelestialPublicationTest);

} // namespace
} // namespace metallic::tests
