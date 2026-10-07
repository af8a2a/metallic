#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/RenderGraph/NRDRuntime.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Denoising/NRDPlan.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/Streamer/SceneResourceManager.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include <fstream>
#include <limits>

#include <gtest/gtest.h>
#include <SDL3/SDL.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <atomic>

namespace metallic::tests {
namespace {
namespace rd = render::denoising;

rd::CommonSettings commonSettings(uint16_t width = 63, uint16_t height = 37)
{
    rd::CommonSettings settings;
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const float projection[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.001f, 1, 0, 0, -0.1001f, 0};
    std::memcpy(settings.worldToViewMatrix, identity, sizeof(identity));
    std::memcpy(settings.worldToViewMatrixPrev, identity, sizeof(identity));
    std::memcpy(settings.viewToClipMatrix, projection, sizeof(projection));
    std::memcpy(settings.viewToClipMatrixPrev, projection, sizeof(projection));
    for (auto* size : {settings.resourceSize, settings.resourceSizePrev, settings.rectSize, settings.rectSizePrev}) {
        size[0] = width;
        size[1] = height;
    }
    settings.timeDeltaBetweenFrames = 1000.0f / 60.0f;
    return settings;
}

TEST(NRDPlan, ShadowLightUsesStableSourceSlots)
{
    scene::LightingSettings lighting;
    lighting.lights.resize(2);
    lighting.lights[0].enabled = false;
    lighting.lights[1].properties.type = "point";
    environment::WorldEnvironment world;
    world.sun.enabled = true;
    world.sun.illuminance = 1000.0f;
    world.moon.enabled = true;
    world.moon.illuminance = 1.0f;
    auto lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting, world.snapshot());
    ASSERT_EQ(lights.size(), 4u);
    EXPECT_FALSE(lights[2].enabled);
    EXPECT_TRUE(lights[0].isCelestial);
    EXPECT_TRUE(lights[1].isCelestial);
    EXPECT_EQ(lights[0].sourceIndex, 0u);
    EXPECT_EQ(lights[1].sourceIndex, 1u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), 0u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 2), 0u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 1), 1u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 3), 3u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 0), 0u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 1412), 0u);
    lighting.lights[0].enabled = true;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting, world.snapshot());
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), 0u) << "Enabling locals changed the sun slot";
    world.sun.enabled = false;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting, world.snapshot());
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), 1u) << "Disabling Sun changed the Moon slot";
    world.moon.enabled = false;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting, world.snapshot());
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), 2u);
    for (auto& light : lighting.lights) { light.enabled = false; }
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting, world.snapshot());
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), UINT32_MAX);
}

TEST(NRDPlan, CelestialGPURecordsPreserveIlluminanceAndFixedSlots)
{
    environment::WorldEnvironment world;
    world.sun = {.direction = float3(0.0f, -2.0f, 0.0f), .color = float3(0.9f, 0.5f, 0.2f),
        .illuminance = 120000.0f, .angularRadius = 0.01f, .enabled = true};
    const auto snapshot = world.snapshot();
    const auto records = render::buildCelestialLightRecords(snapshot);
    EXPECT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0].flags, render::kGPUCelestialLightEnabled | render::kGPUCelestialLightCastsShadow);
    EXPECT_EQ(records[1].flags, 0u);
    EXPECT_FLOAT_EQ(records[0].direction[1], -1.0f);
    const auto color = render::color::fromLinearRec709({0.9f, 0.5f, 0.2f});
    const double projectedSolidAngle = std::acos(-1.0) * std::pow(std::sin(0.01), 2);
    for (size_t channel = 0; channel < 3; ++channel) {
        EXPECT_FLOAT_EQ(records[0].irradiance[channel], color[channel] * 120000.0f);
        EXPECT_NEAR(records[0].diskRadiance[channel] * projectedSolidAngle,
            records[0].irradiance[channel], 0.02);
    }
    world.sun.enabled = false;
    world.moon = snapshot.celestial[0];
    const auto moon = render::buildCelestialLightRecords(world.snapshot());
    EXPECT_EQ(moon[0].flags, 0u);
    EXPECT_EQ(moon[1].flags, records[0].flags);
    EXPECT_EQ(render::buildCelestialLightRecords(snapshot)[0].flags, records[0].flags);
}

TEST(NRDPlan, LegacyDirectionalSourcesNeverEnterLocalResources)
{
    scene::RenderLight imported;
    imported.type = "directional";
    scene::PunctualLight legacy;
    legacy.properties.type = "directional";
    scene::PunctualLight local;
    const std::array importedLights{imported};
    const std::array virtualLights{legacy, local};
    const auto records = render::buildSceneLightRecords(importedLights, virtualLights);
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].sourceVirtualLightIndex, 1);
    EXPECT_EQ(records[0].gpu.directionType[3], 1.0f);
    legacy.enabled = false;
    EXPECT_TRUE(render::buildSceneLightRecords(importedLights, std::span(&legacy, 1)).empty());
}

TEST(NRDPlan, CelestialResolverPreservesSceneOwnershipAndExplicitOverrides)
{
    scene::SceneDocument first;
    scene::SceneDocument second;
    environment::WorldEnvironment firstEnvironment;
    firstEnvironment.sun = {.illuminance = 1000.0f, .enabled = true};
    environment::WorldEnvironment secondEnvironment;
    secondEnvironment.moon = {.illuminance = 2.0f, .enabled = true};
    ASSERT_TRUE(first.setWorldEnvironment(firstEnvironment));
    ASSERT_TRUE(second.setWorldEnvironment(secondEnvironment));
    const auto detached = first.environmentSnapshot();

    render::RenderWorld world;
    world.setScene(&first);
    EXPECT_FALSE(world.hasWorldEnvironmentOverride());
    EXPECT_EQ(render::resolveWorldEnvironment(&first, &world).celestial, detached.celestial);
    auto edited = firstEnvironment;
    edited.sun.illuminance = 2000.0f;
    ASSERT_TRUE(world.setWorldEnvironment(edited));
    EXPECT_EQ(render::resolveWorldEnvironment(&first, &world).celestial, edited.snapshot().celestial);
    EXPECT_EQ(render::resolveWorldEnvironment(&second, &world).celestial, second.environmentSnapshot().celestial);
    EXPECT_EQ(first.environmentSnapshot().celestial, detached.celestial);
    EXPECT_EQ(detached.celestial[0].illuminance, 1000.0f);

    render::RenderWorld unbound;
    EXPECT_FALSE(unbound.hasWorldEnvironmentOverride());
    EXPECT_EQ(render::resolveWorldEnvironment(&first, &unbound).celestial, first.environmentSnapshot().celestial);
    const auto revision = unbound.lightingRevision();
    const auto contentRevision = unbound.sceneContentRevision();
    // Even equal all-disabled data changes the resolver from inheritance to an
    // explicit override, so the next frame must discard the inherited history.
    ASSERT_TRUE(unbound.setWorldEnvironment({}));
    EXPECT_TRUE(unbound.hasWorldEnvironmentOverride());
    EXPECT_GT(unbound.lightingRevision(), revision);
    EXPECT_EQ(unbound.sceneContentRevision(), contentRevision);
    const auto changes = unbound.consumeChanges();
    EXPECT_TRUE(render::hasRenderChange(changes, render::RenderChangeBits::Lighting));
    EXPECT_TRUE(render::hasRenderChange(changes, render::RenderChangeBits::InvalidateTemporalHistory));
    const auto suppressed = render::resolveWorldEnvironment(&first, &unbound);
    EXPECT_FALSE(suppressed.celestial[0].enabled);
    EXPECT_FALSE(suppressed.celestial[1].enabled);
    EXPECT_EQ(render::buildCelestialLightRecords(suppressed)[0].flags, 0u);
    EXPECT_FALSE(unbound.setWorldEnvironment({}));
    EXPECT_EQ(unbound.consumeChanges(), render::RenderChangeBits::None);

    unbound.setScene(&second);
    EXPECT_FALSE(unbound.hasWorldEnvironmentOverride());
    EXPECT_EQ(render::resolveWorldEnvironment(&first, &unbound).celestial, first.environmentSnapshot().celestial);
    EXPECT_EQ(render::resolveWorldEnvironment(&second, &unbound).celestial, second.environmentSnapshot().celestial);
    EXPECT_EQ(render::resolveWorldEnvironment(&second, nullptr).celestial, second.environmentSnapshot().celestial);
    EXPECT_EQ(render::resolveWorldEnvironment(nullptr, &unbound).celestial, unbound.environmentSnapshot().celestial);
}

TEST(NRDPlan, HistoryCountersAndPassSchedule)
{
    rd::NRDPlan plan;
    auto settings = commonSettings();
    ASSERT_TRUE(plan.beginFrame(settings));
    auto reference = plan.schedule(2);
    ASSERT_EQ(reference.size(), 2u);
    EXPECT_FLOAT_EQ(*reinterpret_cast<const float*>(reference[0].constantBufferData), 1.0f);
    reference = plan.schedule(3);
    EXPECT_FLOAT_EQ(*reinterpret_cast<const float*>(reference[0].constantBufferData), 1.0f);
    ++settings.frameIndex;
    ASSERT_TRUE(plan.beginFrame(settings));
    reference = plan.schedule(2);
    EXPECT_FLOAT_EQ(*reinterpret_cast<const float*>(reference[0].constantBufferData), 0.5f);
    reference = plan.schedule(3);
    EXPECT_FLOAT_EQ(*reinterpret_cast<const float*>(reference[0].constantBufferData), 0.5f);
    auto reblur = plan.schedule(0);
    ASSERT_GE(reblur.size(), 6u);
    EXPECT_NE(std::string_view(reblur.front().name).find("Classify"), std::string_view::npos);
    rd::RelaxSettings relax;
    relax.atrousIterationNum = 8;
    plan.setRelaxSettings(relax);
    settings.enableValidation = true;
    settings.isHistoryConfidenceAvailable = true;
    ASSERT_TRUE(plan.beginFrame(settings));
    auto stages = plan.schedule(1);
    uint32_t atrous = 0;
    for (const auto& stage : stages) {
        EXPECT_GT(stage.gridWidth, 0);
        EXPECT_GT(stage.gridHeight, 0);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(stage.constantBufferData) % 16, 0u);
        if (plan.pipelines()[stage.pipelineIndex].shaderName.starts_with("RELAX_Atrous"))
            ++atrous;
        for (uint32_t i = 0; i < stage.resourcesNum; ++i) {
            const auto& resource = stage.resources[i];
            if (resource.type == rd::ResourceType::PERMANENT_POOL)
                EXPECT_LT(resource.indexInPool, plan.permanentPool().size());
            if (resource.type == rd::ResourceType::TRANSIENT_POOL)
                EXPECT_LT(resource.indexInPool, plan.transientPool().size());
        }
    }
    EXPECT_EQ(atrous, 8u);
    EXPECT_NE(std::string_view(stages.back().name).find("Validation"), std::string_view::npos);
    settings.accumulationMode = rd::AccumulationMode::CLEAR_AND_RESTART;
    ASSERT_TRUE(plan.beginFrame(settings));
    reference = plan.schedule(2);
    EXPECT_FLOAT_EQ(*reinterpret_cast<const float*>(reference[0].constantBufferData), 1.0f);
}

TEST(NRDShaders, EverySupportedPermutationUsesNativeHandles)
{
    rd::NRDPlan plan;
    for (const auto& pipeline : plan.pipelines()) {
        SCOPED_TRACE(pipeline.shaderName);
        std::vector<render::SlangMacroDefine> defines;
        for (const auto& define : pipeline.defines)
            defines.push_back({define.name, define.value});
        const char* includes[] = {PROJECT_SOURCE_DIR "/External/MathLib"};
        render::ShaderCompileResult compiled;
        ASSERT_TRUE(
            render::compileSlangShaderToSpirv({
                .moduleName = pipeline.shaderName.c_str(),
                .entryPointName = "main",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders/Interop/Denoising/NRD",
                .additionalSearchPaths = {includes, 1},
                .macroDefines = defines,
            }, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); }))
            << compiled.diagnostics;
        // Slang lowers DescriptorHandle to runtime heap arrays at set 0,
        // bindings 0 (samplers) and 2 (resources), which the RHI maps once.
        // A per-resource binding or a non-array descriptor is a regression.
        std::unordered_map<uint32_t, uint32_t> pointees, variables;
        std::unordered_set<uint32_t> arrays;
        std::vector<uint32_t> boundVariables;
        for (size_t i = 5; i < compiled.spirv.size();) {
            const uint32_t length = compiled.spirv[i] >> 16;
            ASSERT_GT(length, 0u);
            ASSERT_LE(i + length, compiled.spirv.size());
            const uint32_t opcode = compiled.spirv[i] & 0xffffu;
            if (opcode == 32)
                pointees[compiled.spirv[i + 1]] = compiled.spirv[i + 3];
            if (opcode == 59)
                variables[compiled.spirv[i + 2]] = compiled.spirv[i + 1];
            if (opcode == 29)
                arrays.insert(compiled.spirv[i + 1]);
            if (opcode == 71 && length >= 4) {
                if (compiled.spirv[i + 2] == 33u) {
                    EXPECT_TRUE(compiled.spirv[i + 3] == 0u || compiled.spirv[i + 3] == 2u);
                    boundVariables.push_back(compiled.spirv[i + 1]);
                }
                if (compiled.spirv[i + 2] == 34u)
                    EXPECT_EQ(compiled.spirv[i + 3], 0u);
            }
            i += length;
        }
        for (uint32_t variable : boundVariables)
            EXPECT_TRUE(arrays.contains(pointees[variables[variable]]));
    }
}

TEST(NRDPlan, RejectsInvalidImageAndProjectionData)
{
    rd::NRDPlan plan;
    auto settings = commonSettings();
    settings.rectSize[0] = settings.resourceSize[0] + 1;
    EXPECT_FALSE(plan.beginFrame(settings));
    settings = commonSettings();
    settings.resourceSize[0] = 0;
    EXPECT_FALSE(plan.beginFrame(settings));
    settings = commonSettings();
    settings.viewToClipMatrix[0] = 0;
    EXPECT_FALSE(plan.beginFrame(settings));
    settings = commonSettings();
    EXPECT_TRUE(plan.beginFrame(settings));
}

void require(render::Result<> result)
{
    if (!result)
        throw std::runtime_error(render::resultToString(result));
}

class NRDGPU : public ::testing::Test {
protected:
    virtual bool requiresRayQueries() const { return false; }

    void SetUp() override
    {
        ASSERT_TRUE(video.ready) << SDL_GetError();
        const auto result = render::createDevice({.applicationName = "Metallic native NRD tests",
             .enableValidation = true,
             .enableBindlessDescriptorHeap = true,
             .enableRayTracingAccelerationStructure = requiresRayQueries(),
             .enableRayQuery = requiresRayQueries(),
             .validationSink = {.callback =
                                    [](void* context, const render::ValidationMessage& message) noexcept {
                                        // GENERAL loader registration errors are logged separately from API validation.
                                        if ((message.severity == render::ValidationSeverity::Warning || message.severity == render::ValidationSeverity::Error) && render::hasFlag(message.type, render::ValidationCategory::Validation))
                                            static_cast<std::atomic<uint32_t>*>(context)->fetch_add(1);
                                    },
                                .context = &validationErrors}}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(result, render::Error::Unsupported))
            GTEST_SKIP() << "Bindless descriptor heaps unavailable";
        require(result);
        queue = device->getQueue(render::QueueType::Graphics);
        require(device->createCommandPool(*queue).transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        require(commands->createCommandBuffer().transform([&](auto rhiValue) { command = std::move(rhiValue); }));
        require(createStreamer(*device, {.constantBufferSize = 1024 * 1024}).transform([&](auto rhiValue) { streamer = std::move(rhiValue); }));
        require(device->createBuffer({.size = 63 * 37 * 16,
                                      .usage = render::BufferUsageBits::TransferDestination,
                                      .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }));
        createTextures(63, 37);
    }

    void createTextures(uint16_t width, uint16_t height)
    {
        w = width;
        h = height;
        runtime.clear();
        textures.clear();
        views.clear();
        pool = {};
        for (uint32_t i = 0; i < static_cast<uint32_t>(rd::ResourceType::TRANSIENT_POOL); ++i) {
            const auto resource = static_cast<rd::ResourceType>(i);
            render::Format format = render::Format::RGBA32Sfloat;
            if (resource == rd::ResourceType::IN_NORMAL_ROUGHNESS)
                format = render::nrdNormalRoughnessFormat();
            if (resource == rd::ResourceType::IN_VIEWZ || resource == rd::ResourceType::IN_DIFF_CONFIDENCE ||
                resource == rd::ResourceType::IN_SPEC_CONFIDENCE ||
                resource == rd::ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX ||
                resource == rd::ResourceType::IN_PENUMBRA || resource == rd::ResourceType::OUT_SHADOW_TRANSLUCENCY)
                format = render::Format::R32Sfloat;
            std::unique_ptr<render::Texture> texture;
            require(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage |
                          render::TextureUsageBits::TransferDestination | render::TextureUsageBits::TransferSource,
                 .format = format,
                 .width = width,
                 .height = height}).transform([&](auto rhiValue) { texture = std::move(rhiValue); }));
            std::unique_ptr<render::TextureView> view;
            require(device->createTextureView(*texture, {.format = format}).transform([&](auto rhiValue) { view = std::move(rhiValue); }));
            pool[i] = {texture.get(), view.get()};
            textures.push_back(std::move(texture));
            views.push_back(std::move(view));
        }
        std::string log;
        require(runtime.initialize(*device, width, height, pool, log));
        initialized = false;
        frameIndex = 0;
    }

    void TearDown() override
    {
        EXPECT_EQ(validationErrors.load(), 0u) << "Vulkan validation reported warnings or errors";
    }

    // Constant, positive radiance over a flat surface is invariant under the
    // spatial filter. Distinct diffuse/specular values catch descriptor aliasing.
    std::array<float, 2> frame(render::NRDDenoiserMode mode, float diffuse, float specular, bool reset = false,
                               bool confidence = true, bool discard = false, bool retireRuntime = false)
    {
        require(recording.begin(frameIndex));
        require(command->begin(recording.submissionContext()));
        for (size_t i = 0; i < textures.size(); ++i) {
            render::TextureBarrierDesc barrier{
                .texture = textures[i].get(),
                .oldLayout = metallic::render::textureLayoutForResourceState(initialized ? render::ResourceState::General
                                                                     : render::ResourceState::Undefined),
                .newLayout = render::TextureLayout::TransferDestination,
                .before = metallic::render::resourceSyncScope(initialized ? render::ResourceState::General
                                                                     : render::ResourceState::Undefined, metallic::render::PipelineStageBits::AllCommands),
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
            };
            if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
            render::ColorValue value{0, 0, 0, 0};
            const auto resource = static_cast<rd::ResourceType>(i);
            if (resource == rd::ResourceType::IN_DIFF_RADIANCE_HITDIST)
                value = {diffuse, diffuse, diffuse, 0.5f};
            if (resource == rd::ResourceType::IN_SPEC_RADIANCE_HITDIST)
                value = {specular, specular, specular, 0.5f};
            if (resource == rd::ResourceType::IN_NORMAL_ROUGHNESS)
                value = {0.5f, 0.5f, 0.25f, 0.0f};
            if (resource == rd::ResourceType::IN_VIEWZ)
                value = {2, 0, 0, 0};
            if (resource == rd::ResourceType::IN_DIFF_CONFIDENCE || resource == rd::ResourceType::IN_SPEC_CONFIDENCE)
                value = {1, 0, 0, 0};
            if (resource == rd::ResourceType::IN_PENUMBRA)
                value = {diffuse, 0, 0, 0};
            if (auto commandResult = command->clearColorTexture(*textures[i], render::TextureLayout::TransferDestination, value); !commandResult) { throw std::runtime_error(std::string("clearColorTexture failed: ") + metallic::render::resultToString(commandResult)); }
            barrier.oldLayout = render::TextureLayout::TransferDestination; barrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite};
            barrier.newLayout = render::TextureLayout::General; barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite};
            if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
        }
        auto settings = commonSettings(w, h);
        settings.frameIndex = frameIndex++;
        settings.isHistoryConfidenceAvailable = confidence;
        settings.isBaseColorMetalnessAvailable = true;
        settings.enableValidation = true;
        if (reset)
            settings.accumulationMode = rd::AccumulationMode::CLEAR_AND_RESTART;
        require(runtime.setCommonSettings(settings));
        if (mode == render::NRDDenoiserMode::Reference) {
            for (uint32_t i = 0; i < 2; ++i) {
                auto input = pool[static_cast<size_t>(i ? rd::ResourceType::IN_SPEC_RADIANCE_HITDIST
                                                        : rd::ResourceType::IN_DIFF_RADIANCE_HITDIST)];
                auto output = pool[static_cast<size_t>(i ? rd::ResourceType::OUT_SPEC_RADIANCE_HITDIST
                                                         : rd::ResourceType::OUT_DIFF_RADIANCE_HITDIST)];
                runtime.setUserPoolTexture(rd::ResourceType::IN_SIGNAL, *input.texture, *input.view);
                runtime.setUserPoolTexture(rd::ResourceType::OUT_SIGNAL, *output.texture, *output.view);
                require(runtime.denoiseReference(i != 0, *command));
            }
        } else {
            require(runtime.denoise(mode, *command));
        }
        std::array<float, 2> values{};
        for (uint32_t i = 0; i < 2; ++i) {
            auto output = pool[static_cast<size_t>(mode == render::NRDDenoiserMode::Sigma
                ? rd::ResourceType::OUT_SHADOW_TRANSLUCENCY : (i ? rd::ResourceType::OUT_SPEC_RADIANCE_HITDIST
                : rd::ResourceType::OUT_DIFF_RADIANCE_HITDIST))];
            render::TextureBarrierDesc barrier{
                .texture = output.texture,
                .oldLayout = render::TextureLayout::General,
                .newLayout = render::TextureLayout::TransferSource,
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
            };
            if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
            if (auto commandResult = (readback.get())->slice({i * 16}).and_then([&](const auto& bufferSlice) { return command->copyTextureToBuffer({.texture = output.texture,
                                          .buffer = bufferSlice,
                                          .textureOffsetX = w / 2,
                                          .textureOffsetY = h / 2,
                                          .width = 1,
                                          .height = 1,
                                          .depth = 1}); }); !commandResult) { throw std::runtime_error(std::string("copyTextureToBuffer failed: ") + metallic::render::resultToString(commandResult)); }
            barrier.oldLayout = render::TextureLayout::TransferSource; barrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead};
            barrier.newLayout = render::TextureLayout::General; barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite};
            if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
        }
        require(command->end());
        if (retireRuntime) { runtime.clear(); }
        if (discard) {
            recording.cancel();
            command.reset();
            require(commands->createCommandBuffer().transform([&](auto rhiValue) { command = std::move(rhiValue); }));
        } else {
            render::CommandBuffer* list[] = {command.get()};
            render::QueueSubmissionTracker tracker;
            require(tracker.initialize(*device, *queue));
            require(tracker.submit({.commandBuffers = {list, 1}}, recording));
            require(queue->waitIdle());
            initialized = true;
            readback->invalidate();
            const auto* pixels = static_cast<const float*>(readback->map());
            values = {pixels[0], pixels[4]};
            readback->unmap();
        }
        streamer->endFrame();
        return values;
    }

    struct Video {
        bool ready = SDL_Init(SDL_INIT_VIDEO);
        ~Video()
        {
            if (ready)
                SDL_Quit();
        }
    } video;
    std::atomic<uint32_t> validationErrors{0};
    std::unique_ptr<render::Device> device;
    render::Queue* queue = nullptr;
    std::unique_ptr<render::CommandPool> commands;
    std::unique_ptr<render::CommandBuffer> command;
    std::unique_ptr<render::Streamer> streamer;
    std::unique_ptr<render::Buffer> readback;
    std::vector<std::unique_ptr<render::Texture>> textures;
    std::vector<std::unique_ptr<render::TextureView>> views;
    render::NRDUserTexturePool pool;
    render::NRDRuntime runtime;
    uint16_t w = 0, h = 0;
    uint32_t frameIndex = 0;
    bool initialized = false;
    render::RenderFrameContext recording;
};

TEST_F(NRDGPU, ReferenceIndependentSignalsResetResizeAndDiscard)
{
    auto values = frame(render::NRDDenoiserMode::Reference, 2, 10);
    EXPECT_FLOAT_EQ(values[0], 2);
    EXPECT_FLOAT_EQ(values[1], 10);
    values = frame(render::NRDDenoiserMode::Reference, 4, 20);
    EXPECT_FLOAT_EQ(values[0], 3);
    EXPECT_FLOAT_EQ(values[1], 15);
    values = frame(render::NRDDenoiserMode::Reference, 8, 30, true);
    EXPECT_FLOAT_EQ(values[0], 8);
    EXPECT_FLOAT_EQ(values[1], 30);
    frame(render::NRDDenoiserMode::Reference, 100, 100, false, true, true);
    values = frame(render::NRDDenoiserMode::Reference, 6, 12);
    EXPECT_FLOAT_EQ(values[0], 6);
    EXPECT_FLOAT_EQ(values[1], 12);
    createTextures(31, 19);
    values = frame(render::NRDDenoiserMode::Reference, 7, 14);
    EXPECT_FLOAT_EQ(values[0], 7);
    EXPECT_FLOAT_EQ(values[1], 14);
}

TEST_F(NRDGPU, ReferencePreflightFailureRestartsHistory)
{
    const auto first = frame(render::NRDDenoiserMode::Reference, 2, 10);
    ASSERT_FLOAT_EQ(first[0], 2);
    ASSERT_FLOAT_EQ(first[1], 10);

    require(recording.begin(frameIndex));
    require(command->begin(recording.submissionContext()));
    auto settings = commonSettings(w, h);
    settings.frameIndex = frameIndex++;
    require(runtime.setCommonSettings(settings));
    const auto input = pool[static_cast<size_t>(rd::ResourceType::IN_DIFF_RADIANCE_HITDIST)];
    const auto otherInput = pool[static_cast<size_t>(rd::ResourceType::IN_SPEC_RADIANCE_HITDIST)];
    const auto output = pool[static_cast<size_t>(rd::ResourceType::OUT_DIFF_RADIANCE_HITDIST)];
    // Both handles are valid, but the view belongs to another allocation.
    // Validation fails after scheduling advances Reference's history, before
    // any commands or submission cancellation callbacks have been registered.
    runtime.setUserPoolTexture(rd::ResourceType::IN_SIGNAL, *input.texture, *otherInput.view);
    runtime.setUserPoolTexture(rd::ResourceType::OUT_SIGNAL, *output.texture, *output.view);
    const auto failed = runtime.denoiseReference(false, *command);
    EXPECT_TRUE(render::hasError(failed, render::Error::InvalidArgument));
    require(command->end());
    recording.cancel();
    command.reset();
    require(commands->createCommandBuffer().transform([&](auto rhiValue) { command = std::move(rhiValue); }));
    streamer->endFrame();

    // Changing both signals makes stale accumulation observable in the GPU
    // output. Recovery must match a separately requested clean reset without
    // the caller explicitly resetting the failed frame's history.
    const auto recovered = frame(render::NRDDenoiserMode::Reference, 8, 30);
    const auto accumulated = frame(render::NRDDenoiserMode::Reference, 16, 50);
    EXPECT_FLOAT_EQ(accumulated[0], 12);
    EXPECT_FLOAT_EQ(accumulated[1], 40);
    const auto restarted = frame(render::NRDDenoiserMode::Reference, 8, 30, true);
    EXPECT_FLOAT_EQ(restarted[0], 8);
    EXPECT_FLOAT_EQ(restarted[1], 30);
    EXPECT_FLOAT_EQ(recovered[0], restarted[0]);
    EXPECT_FLOAT_EQ(recovered[1], restarted[1]);
}

TEST_F(NRDGPU, SharedRegistryAndRetiredRuntimeSubmission)
{
    const auto first = frame(render::NRDDenoiserMode::Reference, 2, 10);
    EXPECT_FLOAT_EQ(first[0], 2);
    EXPECT_FLOAT_EQ(first[1], 10);
    std::shared_ptr<render::ResourceRegistry> registry;
    require(metallic::render::ResourceRegistry::forDevice(*device).transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
    const auto before = registry->stats().descriptorWrites;
    render::ResourceLease output;
    require(registry->storageImage(*pool[static_cast<size_t>(rd::ResourceType::OUT_DIFF_RADIANCE_HITDIST)].view).transform([&](auto value) { output = std::move(value); }));
    EXPECT_EQ(registry->stats().descriptorWrites, before);
    // The frame, rather than the SDK wrapper, owns the old pipelines, internal
    // images and parameter data until this queued recording completes.
    const auto second = frame(render::NRDDenoiserMode::Reference, 4, 20, false, true, false, true);
    EXPECT_FALSE(runtime.valid());
    EXPECT_FLOAT_EQ(second[0], 3);
    EXPECT_FLOAT_EQ(second[1], 15);
}

TEST_F(NRDGPU, ReblurAndRelaxPreserveFlatRadiance)
{
    if (!device->capabilities().shaderImageGatherExtended)
        GTEST_SKIP() << "REBLUR needs extended image gather";
    for (const auto mode : {render::NRDDenoiserMode::Reblur, render::NRDDenoiserMode::Relax}) {
        for (uint32_t i = 0; i < 4; ++i) {
            SCOPED_TRACE(static_cast<uint32_t>(mode));
            SCOPED_TRACE(i);
            if (mode == render::NRDDenoiserMode::Relax) {
                rd::RelaxSettings settings;
                settings.atrousIterationNum = i % 2 == 0 ? 2 : 8;
                settings.enableAntiFirefly = i % 2 == 0;
                settings.hitDistanceReconstructionMode = rd::HitDistanceReconstructionMode::AREA_5X5;
                require(runtime.setRelaxSettings(settings));
            } else {
                rd::ReblurSettings settings;
                settings.hitDistanceReconstructionMode = rd::HitDistanceReconstructionMode::AREA_5X5;
                if (i >= 2)
                    settings.maxStabilizedFrameNum = 0;
                require(runtime.setReblurSettings(settings));
            }
            // Restart when switching the stabilization topology, as a graph
            // mode change does; consecutive frames still exercise each history.
            const auto values = frame(mode, 2, 5, i == 0 || i == 2, i % 2 == 0);
            EXPECT_NEAR(values[0], 2, 0.12f);
            EXPECT_NEAR(values[1], 5, 0.25f);
        }
    }
}

TEST(NRDPlan, SigmaScheduleAndIsolatedAllocation)
{
    rd::NRDPlan plan(true);
    auto common = commonSettings();
    ASSERT_TRUE(plan.beginFrame(common));
    auto stages = plan.schedule(4);
    ASSERT_EQ(stages.size(), 6u);
    EXPECT_EQ(plan.permanentPool().size(), 1u);
    EXPECT_EQ(plan.transientPool().size(), 8u);
    EXPECT_TRUE(plan.schedule(0).empty());
    rd::SigmaSettings sigma;
    sigma.maxStabilizedFrameNum = 0;
    plan.setSigmaSettings(sigma);
    ASSERT_TRUE(plan.beginFrame(common));
    stages = plan.schedule(4);
    EXPECT_EQ(stages.size(), 4u);
    common.splitScreen = 1.0f;
    ASSERT_TRUE(plan.beginFrame(common));
    stages = plan.schedule(4);
    ASSERT_EQ(stages.size(), 1u);
    EXPECT_EQ(plan.pipelines()[stages[0].pipelineIndex].shaderName, "SIGMA_SplitScreen");
}

TEST_F(NRDGPU, SigmaLitOccludedResetResizeAndDiscard)
{
    for (uint32_t history : {0u, 5u}) {
        rd::SigmaSettings sigma;
        sigma.maxStabilizedFrameNum = history;
        require(runtime.setSigmaSettings(sigma));
        for (float penumbra : {65504.0f, 0.0f, 0.05f}) {
            auto value = frame(render::NRDDenoiserMode::Sigma, penumbra, 0, true, false);
            EXPECT_NEAR(value[0], penumbra == 65504.0f ? 1.0f : 0.0f, 0.005f);
        }
    }
    frame(render::NRDDenoiserMode::Sigma, 65504, 0, true, false, true);
    EXPECT_NEAR(frame(render::NRDDenoiserMode::Sigma, 0, 0, true, false)[0], 0, 0.005f);
    createTextures(31, 19);
    EXPECT_NEAR(frame(render::NRDDenoiserMode::Sigma, 65504, 0, false, false)[0], 1, 0.005f);
}

class NRDRayTracingGPU : public NRDGPU {
protected:
    bool requiresRayQueries() const override { return true; }
};

TEST(NRDWorkingColor, RTXDIRelaxSceneChromaticity)
{
    ASSERT_TRUE(SDL_Init(SDL_INIT_VIDEO)) << SDL_GetError();
    struct VideoLifetime { ~VideoLifetime() { SDL_Quit(); } } video;
    render::RenderSampleLoadResult sample;
    std::string log;
    ASSERT_TRUE(render::loadBuiltInRenderSample("rtxdi-sample", sample, log)) << log;
    sample.graph.findNode("RTXDI")->properties["lightSource"] = "scene";
    sample.graph.findNode("RTXDI")->properties["environmentSamples"] = 0;
    render::RenderGraphPreviewRenderer preview;
    const auto initialized = preview.initialize(true, true);
    if (render::hasError(initialized, render::Error::Unsupported)) { GTEST_SKIP() << "Ray query unavailable"; }
    ASSERT_TRUE(initialized);
    preview.setRawReadbackEnabled(true);
    preview.setEnvironment({.enabled = false, .visible = false});
    environment::WorldEnvironment world;
    world.sun = {.direction = float3(0, -0.2f, -1), .color = float3(0.9f, 0.5f, 0.2f),
        .illuminance = 1000.0f, .enabled = true};
    ASSERT_TRUE(preview.setWorldEnvironment(world));
    std::array<render::color::RGB, 2> chromaticities{};
    for (size_t mode = 0; mode < 2; ++mode) {
        const char* output = mode == 0 ? "RTXDI.color" : "Composite.color";
        for (uint32_t frame = 0; frame < 16; ++frame) {
            ASSERT_TRUE(preview.render(sample.graph, 64, 64, output, frame == 15)) << preview.lastLog();
        }
        ASSERT_EQ(preview.readbackFormat(), render::Format::RGBA32Sfloat);
        ASSERT_EQ(preview.readbackBytes().size(), 64 * 64 * 16);
        const auto* values = reinterpret_cast<const float*>(preview.readbackBytes().data());
        render::color::RGB energy{};
        for (size_t pixel = 0; pixel < 64 * 64; ++pixel) {
            for (size_t channel = 0; channel < 3; ++channel) {
                ASSERT_TRUE(std::isfinite(values[pixel * 4 + channel]));
                energy[channel] += values[pixel * 4 + channel];
            }
        }
        energy = render::color::toLinearRec709(energy);
        const float total = energy[0] + energy[1] + energy[2];
        ASSERT_GT(total, 1.0f);
        for (size_t channel = 0; channel < 3; ++channel) { chromaticities[mode][channel] = energy[channel] / total; }
    }
    for (size_t channel = 0; channel < 3; ++channel) {
        EXPECT_NEAR(chromaticities[0][channel], chromaticities[1][channel], 0.015f)
            << "NRD changed scene chromaticity in channel " << channel;
    }
}

std::vector<float> physicalNRDReadback(const render::RenderGraphPreviewRenderer& preview)
{
    const auto& bytes = preview.readbackBytes();
    std::vector<float> result;
    if (preview.readbackFormat() == render::Format::RGBA32Sfloat) {
        result.resize(bytes.size() / sizeof(float));
        std::memcpy(result.data(), bytes.data(), bytes.size());
    } else if (preview.readbackFormat() == render::Format::RGBA16Sfloat ||
        preview.readbackFormat() == render::Format::R16Sfloat) {
        result.reserve(bytes.size() / sizeof(uint16_t));
        for (size_t offset = 0; offset < bytes.size(); offset += sizeof(uint16_t)) {
            uint16_t half;
            std::memcpy(&half, bytes.data() + offset, sizeof(half));
            const uint32_t exponent = (half >> 10) & 31u;
            const uint32_t mantissa = half & 1023u;
            const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -24) :
                exponent == 31 ? (mantissa == 0 ? std::numeric_limits<float>::infinity() :
                    std::numeric_limits<float>::quiet_NaN()) :
                std::ldexp(float(1024u + mantissa), int(exponent) - 25);
            result.push_back((half & 0x8000u) != 0 ? -magnitude : magnitude);
        }
    } else {
        throw std::runtime_error("Unexpected physical NRD readback format");
    }
    for (const float value : result) {
        if (!std::isfinite(value)) { throw std::runtime_error("Nonfinite physical NRD color/guide"); }
    }
    return result;
}

TEST(NRDWorkingColor, PhysicalAtmosphereHDRAndAerial)
{
    ASSERT_TRUE(SDL_Init(SDL_INIT_VIDEO)) << SDL_GetError();
    struct VideoLifetime { ~VideoLifetime() { SDL_Quit(); } } video;
    render::RenderSampleLoadResult sample;
    render::RenderSampleLoadResult physicalSample;
    std::string log;
    ASSERT_TRUE(render::loadBuiltInRenderSample("rtxdi-sample", sample, log)) << log;
    ASSERT_TRUE(render::loadBuiltInRenderSample("physical-atmosphere-lookdev", physicalSample, log)) << log;
    scene::SceneDocument document;
    ASSERT_TRUE(document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / physicalSample.desc.scenePath))
        << document.documentWarning() << document.lastLoadResult().error;
    // Non-emissive surfaces make emissive-guide energy an observation of aerial
    // in-scattering, rather than authored material emission.
    for (size_t index = 0; index < document.materials().size(); ++index) {
        auto material = document.materials()[index];
        material.emissiveFactor = float3(0.0f);
        (void)document.setMaterialProperties(static_cast<int32_t>(index), material);
    }
    sample.graph.findNode("RTXDI")->properties["path"] = physicalSample.desc.scenePath;
    sample.graph.findNode("RTXDI")->properties["lightSource"] = "scene";
    sample.graph.findNode("RTXDI")->properties["environmentSamples"] = 4;
    sample.graph.findNode("RTXDI")->properties["outputLinear"] = true;
    sample.graph.findNode("Composite")->properties["outputLinear"] = true;
    auto camera = physicalSample.graph.findNode("Reference")->properties.at("camera");
    camera["zfar"] = 100.0;
    auto lighting = document.lighting();
    lighting.autoExposure.enabled = false;
    lighting.exposureEV100 = 14.0f;
    render::RenderGraphPreviewRenderer preview;
    preview.bindRuntimeScene(&document);
    preview.setLighting(lighting);
    preview.setEnvironment(document.environment());
    const auto initialized = preview.initialize(true, true, false);
    if (render::hasError(initialized, render::Error::Unsupported)) { GTEST_SKIP() << "Ray query unavailable"; }
    ASSERT_TRUE(initialized) << preview.lastLog();
    preview.setRawReadbackEnabled(true);
    constexpr uint32_t width = 96, height = 96;
    const size_t pixels = size_t(width) * height;
    const auto evidenceDirectory = std::filesystem::path(PROJECT_SOURCE_DIR) / "build/physical-environment-nrd";
    std::filesystem::create_directories(evidenceDirectory);
    auto measurements = render::RenderGraphProperties::array();
    std::array<double, 2> aerialSurfaceEnergy{};
    try {
        const auto renderOutput = [&](const char* output, uint32_t frames) {
            for (uint32_t frame = 0; frame < frames; ++frame) {
                preview.setRawReadbackEnabled(frame + 1 == frames);
                require(preview.render(sample.graph, width, height, output, frame + 1 == frames));
                bool sawRTXDI = false, sawRelax = false, sawComposite = false;
                for (const auto& node : preview.executionStats().nodes) {
                    sawRTXDI |= node.name == "RTXDI";
                    sawRelax |= node.name == "Relax";
                    sawComposite |= node.name == "Composite";
                }
                if (!sawRTXDI || !sawRelax || !sawComposite) {
                    throw std::runtime_error("Physical NRD case did not execute RTXDI -> RELAX -> Composite");
                }
            }
            return physicalNRDReadback(preview);
        };
        const auto writeHDR = [&](const std::string& label, const std::vector<float>& values) {
            std::ofstream raw(evidenceDirectory / (label + ".float32"), std::ios::binary);
            raw.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            if (!raw) { throw std::runtime_error("Physical NRD evidence write failed"); }
        };
        const auto setCamera = [&](const render::RenderGraphProperties& selectedCamera) {
            for (const char* name : {"RTXDI", "Relax"}) {
                auto* node = sample.graph.findNode(name);
                auto properties = node->properties;
                properties["camera"] = selectedCamera;
                if (node->properties == properties) { continue; }
                if (!sample.graph.setNodeProperties(node->id, std::move(properties))) {
                    throw std::runtime_error("Physical NRD camera update failed");
                }
            }
        };
        auto world = document.worldEnvironment();
        for (uint32_t mode = 0; mode < 4; ++mode) {
            const std::string label = mode == 0 ? "Physical" : mode == 1 ? "AerialMieHeavy" :
                mode == 2 ? "HDRIReturn" : "SolarDisk";
            auto state = world;
            auto selectedCamera = camera;
            if (mode == 1) {
                state.atmosphere.mieScattering *= 10.0f;
                state.atmosphere.mieExtinction *= 10.0f;
            } else if (mode == 2) {
                state.source = environment::EnvironmentSource::HDRI;
                state.sun.illuminance = 1000.0f;
                preview.setEnvironment({.enabled = true,
                    .path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/LookDev/OpenPbrDefault/san_giuseppe_bridge_split.hdr",
                    .intensity = 1.0f, .visible = true});
            } else if (mode == 3) {
                const auto& eye = selectedCamera["eye"];
                const float3 sourceDirection = -normalize(state.sun.direction);
                selectedCamera["center"] = {eye[0].get<float>() + sourceDirection.x,
                    eye[1].get<float>() + sourceDirection.y, eye[2].get<float>() + sourceDirection.z};
                selectedCamera["fovDegrees"] = 8.0;
                preview.setEnvironment(document.environment());
            }
            setCamera(selectedCamera);
            if (!preview.setWorldEnvironment(state) && mode != 0) {
                throw std::runtime_error("Physical NRD world update was rejected");
            }
            const auto color = renderOutput("Composite.color", 32);
            if (preview.readbackFormat() != render::Format::RGBA32Sfloat || color.size() != pixels * 4) {
                throw std::runtime_error("Physical NRD composite must retain RGBA32F");
            }
            const auto emissive = renderOutput("RTXDI.emissive", 1);
            if (preview.readbackFormat() != render::Format::RGBA32Sfloat || emissive.size() != pixels * 4) {
                throw std::runtime_error("Physical NRD emissive/sky guide must retain RGBA32F");
            }
            const auto diffuse = renderOutput("RTXDI.noisyDiffuse", 1);
            if (preview.readbackFormat() != render::Format::RGBA16Sfloat || diffuse.size() != pixels * 4) {
                throw std::runtime_error("Physical NRD diffuse adapter must produce finite RGBA16F");
            }
            const auto specular = renderOutput("RTXDI.noisySpecular", 1);
            if (preview.readbackFormat() != render::Format::RGBA16Sfloat || specular.size() != pixels * 4) {
                throw std::runtime_error("Physical NRD specular adapter must produce finite RGBA16F");
            }
            const auto depth = renderOutput("RTXDI.viewZ", 1);
            if (depth.size() != pixels) { throw std::runtime_error("Physical NRD depth readback missing"); }
            double energy = 0.0, maxColor = 0.0, surfaceAerial = 0.0;
            const float inverseScale = emissive[3];
            size_t surfacePixels = 0;
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                for (size_t channel = 0; channel < 3; ++channel) {
                    const float value = color[pixel * 4 + channel];
                    if (value < -0.0001f) { throw std::runtime_error("Negative physical NRD HDR"); }
                    energy += value;
                    maxColor = std::max(maxColor, double(value));
                    if (mode < 2 && depth[pixel] < 90.0f) { surfaceAerial += emissive[pixel * 4 + channel]; }
                }
                if (mode < 2 && depth[pixel] < 90.0f) { ++surfacePixels; }
                if (emissive[pixel * 4 + 3] != inverseScale) {
                    throw std::runtime_error("Physical NRD inverse exposure scale varies across one scene");
                }
            }
            if (mode == 2 ? inverseScale != 1.0f : inverseScale <= 1.0f) {
                throw std::runtime_error("Physical NRD pre-exposure failed to switch physical/HDRI domains");
            }
            if (mode < 2) {
                if (surfacePixels < 100 || surfaceAerial <= 0.00001) {
                    throw std::runtime_error("Non-emissive physical surfaces lack aerial in-scattering");
                }
                aerialSurfaceEnergy[mode] = surfaceAerial;
            }
            if (mode == 3 && maxColor <= 65504.0) {
                throw std::runtime_error("NRD composite clamped the visible physical Sun to FP16 range");
            }
            writeHDR(label + "Composite", color);
            writeHDR(label + "Emissive", emissive);
            writeHDR(label + "NoisyDiffuse", diffuse);
            writeHDR(label + "NoisySpecular", specular);
            measurements.push_back({{"case", label}, {"meanRGBEnergy", energy / pixels},
                {"maximumRGB", maxColor}, {"inversePreExposure", inverseScale},
                {"surfacePixels", surfacePixels}, {"surfaceAerialEnergy", surfaceAerial}});
        }
        std::ofstream summary(evidenceDirectory / "PhysicalNRDIntegration.json");
        summary << render::RenderGraphProperties{{"width", width}, {"height", height},
            {"framesPerCase", 32}, {"cases", measurements}}.dump(2) << '\n';
        summary.close();
        if (!summary) { throw std::runtime_error("Physical NRD summary write failed"); }
        EXPECT_GT(std::abs(aerialSurfaceEnergy[1] - aerialSurfaceEnergy[0]), aerialSurfaceEnergy[0] * 0.001)
            << "Atmospheric edits did not update the aerial guide";
    } catch (const std::exception& error) {
        FAIL() << error.what() << ": " << preview.lastLog();
    }
}

TEST_F(NRDRayTracingGPU, RayTracedShadowOcclusionAndHistory)
{
    render::ScreenSpaceShadows shadows;
    render::ScreenSpaceShadowSettings settings;
    settings.maxDistance = 100000;
    if (!device->capabilities().rayQuery || !device->capabilities().rayTracingAccelerationStructure) {
        GTEST_SKIP() << "Ray queries unavailable";
    }
    // The blocker is behind the camera (z=-1), so it never appears in the depth
    // buffer. It casts onto the visible z=3 plane along the celestial-light ray.
    const auto fixtureDirectory = std::filesystem::path(PROJECT_SOURCE_DIR) / ".tmp/NRDRayTracedShadowGeometry";
    std::filesystem::create_directories(fixtureDirectory);
    const float vertices[] = {
        -10, -10, 3, 10, -10, 3, 10, 10, 3, -10, 10, 3,
        4, -10, -1, 7, -10, -1, 7, 10, -1, 4, 10, -1,
    };
    const uint32_t indices[] = {0, 1, 2, 0, 2, 3, 0, 1, 2, 0, 2, 3};
    {
        std::ofstream mesh(fixtureDirectory / "Geometry.bin", std::ios::binary);
        mesh.write(reinterpret_cast<const char*>(vertices), sizeof(vertices));
        mesh.write(reinterpret_cast<const char*>(indices), sizeof(indices));
        ASSERT_TRUE(mesh.good());
    }
    std::array<scene::SceneDocument, 3> scenes;
    std::array<render::ScenePathTraceResources, 3> geometry;
    render::SceneResourceManager sceneResources;
    for (size_t i = 0; i < geometry.size(); ++i) {
        auto document = render::RenderGraphProperties::parse(R"json({
            "asset":{"version":"2.0"}, "scene":0, "scenes":[{"nodes":[0,1]}],
            "nodes":[{"mesh":0},{"mesh":1}],
            "buffers":[{"uri":"Geometry.bin","byteLength":144}],
            "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":96},
                           {"buffer":0,"byteOffset":96,"byteLength":48}],
            "accessors":[
                {"bufferView":0,"byteOffset":0,"componentType":5126,"count":4,"type":"VEC3","min":[-10,-10,3],"max":[10,10,3]},
                {"bufferView":0,"byteOffset":48,"componentType":5126,"count":4,"type":"VEC3","min":[4,-10,-1],"max":[7,10,-1]},
                {"bufferView":1,"byteOffset":0,"componentType":5125,"count":6,"type":"SCALAR"},
                {"bufferView":1,"byteOffset":24,"componentType":5125,"count":6,"type":"SCALAR"}],
            "materials":[{"doubleSided":true},{"doubleSided":true}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":2,"material":0}]},
                      {"primitives":[{"attributes":{"POSITION":1},"indices":3,"material":1}]}]
        })json");
        if (i == 0) { document["scenes"][0]["nodes"] = {0}; }
        if (i == 2) {
            document["materials"][1]["alphaMode"] = "MASK";
            document["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] = {1, 1, 1, 0};
        }
        const auto path = fixtureDirectory / ("Scene" + std::to_string(i) + ".gltf");
        { std::ofstream file(path); file << document.dump(); ASSERT_TRUE(file.good()); }
        ASSERT_TRUE(scenes[i].load(path)) << scenes[i].lastLoadResult().error;
        std::string log;
        std::shared_ptr<render::SceneResourceSnapshot> snapshot;
        ASSERT_TRUE(sceneResources.acquire(*device, *queue, {{"path", path.generic_string()}}, &scenes[i], render::SceneResourceFeatureBits::Geometry | render::SceneResourceFeatureBits::Materials |
            render::SceneResourceFeatureBits::MaterialTextures | render::SceneResourceFeatureBits::StandardAccelerationStructure, log).transform([&](auto value) { snapshot = std::move(value); })) << log;
        ASSERT_NE(snapshot, nullptr);
        ASSERT_NE(snapshot->pathTraceResources, nullptr);
        geometry[i] = *snapshot->pathTraceResources;
        ASSERT_TRUE(geometry[i].valid());
    }
    bool alphaCutout = false;
    render::RenderView camera;
    render::ViewCamera pose;
    pose.eye = {0, 0, 0};
    pose.center = {0, 0, 1};
    pose.orthographic = true;
    pose.orthoHeight = 4;
    pose.nearPlane = 0.1f;
    pose.farPlane = 10;
    ASSERT_TRUE(camera.setCamera(pose));
    environment::WorldEnvironment world;
    world.sun = {.direction = float3(-0.70710678f, 0, 0.70710678f), .illuminance = 10.0f,
        .angularRadius = 0.5f * 0.01745329252f, .enabled = true};
    auto lights = render::buildScreenSpaceShadowLightRecords(nullptr, {}, world.snapshot());
    std::unique_ptr<render::Texture> depth;
    std::unique_ptr<render::TextureView> depthView;
    std::unique_ptr<render::Buffer> upload;
    require(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferDestination,
        .format = render::Format::R32Sfloat, .width = w, .height = h}).transform([&](auto rhiValue) { depth = std::move(rhiValue); }));
    require(device->createTextureView(*depth, {.format = render::Format::R32Sfloat}).transform([&](auto rhiValue) { depthView = std::move(rhiValue); }));
    require(device->createBuffer({.size = uint64_t(w) * h * 4, .usage = render::BufferUsageBits::TransferSource,
        .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto rhiValue) { upload = std::move(rhiValue); }));
    bool depthReady = false;
    uint32_t shadowFrame = 0;
    render::ViewConstants previous{};
    auto renderShadow = [&](bool blocker, bool discard = false, bool falseDepthBlocker = false) {
        auto view = camera.constants(shadowFrame++, w, h, w, h, depthReady ? &previous : nullptr);
        auto* data = static_cast<float*>(upload->map());
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                const float z = falseDepthBlocker && x < 20 ? 2.0f : 3.0f;
                float d = pose.orthographic ? (z - pose.nearPlane) / (pose.farPlane - pose.nearPlane)
                    : pose.farPlane / (pose.farPlane - pose.nearPlane) * (1.0f - pose.nearPlane / z);
                data[y * w + x] = 1.0f - d;
            }
        }
        upload->flush({0, uint64_t(w) * h * 4});
        upload->unmap();
        require(recording.begin(frameIndex));
        require(command->begin(recording.submissionContext()));
        command->hostWriteBarrier();
        render::TextureBarrierDesc barrier{
            .texture = depth.get(),
            .oldLayout = metallic::render::textureLayoutForResourceState(depthReady ? render::ResourceState::ShaderRead : render::ResourceState::Undefined),
            .newLayout = render::TextureLayout::TransferDestination,
            .before = metallic::render::resourceSyncScope(depthReady ? render::ResourceState::ShaderRead : render::ResourceState::Undefined, metallic::render::PipelineStageBits::AllCommands),
            .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
        };
        if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
        if (auto commandResult = (upload.get())->slice().and_then([&](const auto& bufferSlice) { return command->copyBufferToTexture({.texture = depth.get(), .buffer = bufferSlice, .width = w, .height = h}); }); !commandResult) { throw std::runtime_error(std::string("copyBufferToTexture failed: ") + metallic::render::resultToString(commandResult)); }
        barrier.oldLayout = render::TextureLayout::TransferDestination; barrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite};
        barrier.newLayout = render::TextureLayout::ShaderRead; barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead};
        if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
        render::ScreenSpaceShadowResult output;
        std::string log;
        auto result = shadows.record(*device, *command, *streamer, *depthView, view, lights, blocker ? (alphaCutout ? 3 : 1) : 2, 0, settings, log, &geometry[blocker ? (alphaCutout ? 2 : 1) : 0]).transform([&](auto value) { output = std::move(value); });
        if (!result) { throw std::runtime_error(log + render::resultToString(result)); }
        barrier.texture = output.texture;
        barrier.oldLayout = render::TextureLayout::ShaderRead; barrier.before = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead};
        barrier.newLayout = render::TextureLayout::TransferSource; barrier.after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead};
        if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
        if (auto commandResult = (readback.get())->slice().and_then([&](const auto& bufferSlice) { return command->copyTextureToBuffer({.texture = output.texture, .buffer = bufferSlice, .width = w, .height = h}); }); !commandResult) { throw std::runtime_error(std::string("copyTextureToBuffer failed: ") + metallic::render::resultToString(commandResult)); }
        barrier.oldLayout = render::TextureLayout::TransferSource; barrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead};
        barrier.newLayout = render::TextureLayout::ShaderRead; barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead};
        if (auto commandResult = command->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
        require(command->end());
        if (discard) {
            recording.cancel();
            command.reset();
            require(commands->createCommandBuffer().transform([&](auto rhiValue) { command = std::move(rhiValue); }));
            streamer->endFrame();
            return std::vector<uint8_t>{};
        }
        render::CommandBuffer* list[] = {command.get()};
        render::QueueSubmissionTracker tracker;
        require(tracker.initialize(*device, *queue));
        require(tracker.submit({.commandBuffers = {list, 1}}, recording));
        require(queue->waitIdle());
        depthReady = true;
        previous = view;
        readback->invalidate();
        auto* pixels = static_cast<const uint8_t*>(readback->map());
        std::vector<uint8_t> image(pixels, pixels + w * h);
        readback->unmap();
        streamer->endFrame();
        return image;
    };
    for (bool perspective : {false, true}) {
        pose.orthographic = !perspective;
        ASSERT_TRUE(camera.setCamera(pose));
        camera.cameraCut();
        for (bool denoise : {false, true}) {
            settings.denoise = denoise;
            auto flat = renderShadow(false);
            EXPECT_EQ(*std::min_element(flat.begin(), flat.end()), 255) << "Flat surface self-shadowed";
            auto falseOcclusion = renderShadow(false, false, true);
            EXPECT_EQ(*std::min_element(falseOcclusion.begin(), falseOcclusion.end()), 255)
                << "A depth-only occluder affected full ray-traced shadows";
            auto blocked = renderShadow(true);
            size_t dark = 0;
            for (uint32_t y = 5; y < h - 5; ++y) {
                for (uint32_t x = 21; x < w - 5; ++x) { dark += blocked[y * w + x] < 128; }
            }
            EXPECT_GT(dark, 30u) << "Offscreen geometry did not cast a shadow";
            for (uint32_t i = 0; i < 3; ++i) { blocked = renderShadow(true); }
            flat = renderShadow(false);
            EXPECT_EQ(*std::min_element(flat.begin(), flat.end()), 255) << "Removed blocker retained history";
        }
    }
    // Moon keeps slot 1 and traces its own visibility when Sun is disabled.
    world.moon = world.sun;
    world.sun.enabled = false;
    world.moon.angularRadius = 0.0f;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, {}, world.snapshot());
    ASSERT_EQ(render::selectScreenSpaceShadowLight(lights, -1), 1u);
    for (bool denoise : {false, true}) {
        settings.denoise = denoise;
        const auto flat = renderShadow(false);
        EXPECT_EQ(*std::min_element(flat.begin(), flat.end()), 255) << "Moon-only flat surface self-shadowed";
        const auto blocked = renderShadow(true);
        size_t dark = 0;
        for (uint32_t y = 5; y < h - 5; ++y) {
            for (uint32_t x = 21; x < w - 5; ++x) { dark += blocked[y * w + x] < 128; }
        }
        EXPECT_GT(dark, 30u) << "Moon-only offscreen geometry did not cast a shadow";
    }
    world.sun.enabled = true;
    world.moon.enabled = false;
    world.sun.angularRadius = 3.0f * 0.01745329252f;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, {}, world.snapshot());
    auto temporalVariation = [&](bool denoise) {
        settings.denoise = denoise;
        std::vector<uint8_t> last;
        double variation = 0.0;
        for (uint32_t i = 0; i < 16; ++i) {
            auto image = renderShadow(true);
            if (i >= 8) {
                for (size_t pixel = 0; pixel < image.size(); ++pixel) {
                    variation += std::abs(int(image[pixel]) - int(last[pixel]));
                }
            }
            last = std::move(image);
        }
        return variation;
    };
    const double rawVariation = temporalVariation(false);
    const double filteredVariation = temporalVariation(true);
    EXPECT_GT(rawVariation, 0.0);
    EXPECT_LT(filteredVariation, rawVariation) << "SIGMA did not reduce temporal shadow noise";
    // Local-light penumbra packing and source switching use the same history owner.
    scene::LightingSettings localLighting;
    auto& localLight = localLighting.lights.emplace_back();
    localLight.properties.intensity = 10.0;
    localLight.properties.intensityUnit = scene::LightUnit::Candela;
    localLight.position = float3(6.0f, 0.0f, -4.0f);
    world.sun.enabled = false;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, localLighting, world.snapshot());
    settings.lightIndex = 2;
    for (bool denoise : {false, true}) {
        settings.denoise = denoise;
        auto local = renderShadow(true);
        EXPECT_LT(*std::min_element(local.begin(), local.end()), 128);
        auto flat = renderShadow(false);
        EXPECT_EQ(*std::min_element(flat.begin(), flat.end()), 255);
    }
    alphaCutout = true;
    auto cutout = renderShadow(true);
    EXPECT_EQ(*std::min_element(cutout.begin(), cutout.end()), 255) << "Alpha-cutout geometry cast an opaque shadow";
    alphaCutout = false;
    settings.maxDistance = 1.0f;
    auto truncated = renderShadow(true);
    EXPECT_EQ(*std::min_element(truncated.begin(), truncated.end()), 255) << "Shadow ray ignored its maximum length";
    settings.maxDistance = 100000.0f;
    camera.setTemporalJitter(true);
    renderShadow(true, true);
    auto afterDiscard = renderShadow(false);
    EXPECT_EQ(*std::min_element(afterDiscard.begin(), afterDiscard.end()), 255);
    settings.enabled = false;
    auto disabled = renderShadow(true);
    EXPECT_EQ(*std::min_element(disabled.begin(), disabled.end()), 255);
    settings.enabled = true;
    settings.lightIndex = -1;
    world.sun.enabled = false;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, {}, world.snapshot());
    auto noLight = renderShadow(true);
    EXPECT_EQ(*std::min_element(noLight.begin(), noLight.end()), 255);
    w = 31;
    h = 19;
    depthView.reset();
    depth.reset();
    require(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferDestination,
        .format = render::Format::R32Sfloat, .width = w, .height = h}).transform([&](auto rhiValue) { depth = std::move(rhiValue); }));
    require(device->createTextureView(*depth, {.format = render::Format::R32Sfloat}).transform([&](auto rhiValue) { depthView = std::move(rhiValue); }));
    depthReady = false;
    world.sun.enabled = true;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, {}, world.snapshot());
    auto resized = renderShadow(false);
    EXPECT_EQ(resized.size(), size_t(w) * h);
    EXPECT_EQ(*std::min_element(resized.begin(), resized.end()), 255);
}
} // namespace
} // namespace metallic::tests
