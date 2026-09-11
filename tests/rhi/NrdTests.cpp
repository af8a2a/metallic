#include "Runtime/Render/RenderGraph/NrdRuntime.h"
#include "Runtime/Render/Denoising/NrdPlan.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Scene/SceneDocument.h"
#include <fstream>

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

TEST(NrdPlan, ShadowLightUsesStableSourceSlots)
{
    scene::LightingSettings lighting;
    lighting.lights.resize(3);
    lighting.lights[0].enabled = false;
    lighting.lights[1].properties.type = "point";
    lighting.lights[2].properties.type = "directional";
    auto lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting);
    ASSERT_EQ(lights.size(), 4u);
    EXPECT_EQ(lights[1].colorIntensity[3], 0);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), 2u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 2), 2u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 1), 1u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 0), 2u);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 1412), 2u);
    lighting.lights[0].enabled = true;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), 2u) << "Enabling earlier lights changed the sun slot";
    lighting.lights[2].enabled = false;
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, 2), 0u);
    for (auto& light : lighting.lights) { light.enabled = false; }
    lights = render::buildScreenSpaceShadowLightRecords(nullptr, lighting);
    EXPECT_EQ(render::selectScreenSpaceShadowLight(lights, -1), UINT32_MAX);
}

TEST(NrdPlan, HistoryCountersAndPassSchedule)
{
    rd::NrdPlan plan;
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

TEST(NrdShaders, EverySupportedPermutationUsesNativeHandles)
{
    rd::NrdPlan plan;
    for (const auto& pipeline : plan.pipelines()) {
        SCOPED_TRACE(pipeline.shaderName);
        std::vector<render::SlangMacroDefine> defines;
        for (const auto& define : pipeline.defines)
            defines.push_back({define.name, define.value});
        const char* includes[] = {PROJECT_SOURCE_DIR "/External/MathLib"};
        render::ShaderCompileResult compiled;
        ASSERT_TRUE(
            render::compileSlangShaderToSpirv({.moduleName = pipeline.shaderName.c_str(),
                                               .entryPointName = "main",
                                               .searchPath = PROJECT_SOURCE_DIR "/Shaders/Libraries/Denoising/NRD",
                                               .additionalSearchPaths = includes,
                                               .additionalSearchPathCount = 1,
                                               .macroDefines = defines.data(),
                                               .macroDefineCount = static_cast<uint32_t>(defines.size())},
                                              compiled))
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

TEST(NrdPlan, RejectsInvalidImageAndProjectionData)
{
    rd::NrdPlan plan;
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

void require(render::Result result)
{
    if (!result)
        throw std::runtime_error(render::resultToString(result));
}

class NrdGpu : public ::testing::Test {
protected:
    virtual bool requiresRayQueries() const { return false; }

    void SetUp() override
    {
        ASSERT_TRUE(video.ready) << SDL_GetError();
        const auto result = render::createDevice(
            {.applicationName = "Metallic native NRD tests",
             .enableValidation = true,
             .enableBindlessDescriptorHeap = true,
             .enableRayTracingAccelerationStructure = requiresRayQueries(),
             .enableRayQuery = requiresRayQueries(),
             .validationSink = {.callback =
                                    [](void* context, const render::ValidationMessage& message) noexcept {
                                        if ((message.severity & 0x1100u) != 0)
                                            static_cast<std::atomic<uint32_t>*>(context)->fetch_add(1);
                                    },
                                .context = &validationErrors}},
            device);
        if (render::hasError(result, render::Error::Unsupported))
            GTEST_SKIP() << "Bindless descriptor heaps unavailable";
        require(result);
        queue = device->getQueue(render::QueueType::Graphics);
        require(device->createCommandPool(*queue, commands));
        require(commands->createCommandBuffer(command));
        require(device->createStreamer({.constantBufferSize = 1024 * 1024}, streamer));
        require(device->createBuffer({.size = 63 * 37 * 16,
                                      .usage = render::BufferUsageBits::TransferDestination,
                                      .memoryLocation = render::MemoryLocation::HostReadback},
                                     readback));
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
            render::Format format = render::Format::Rgba32Sfloat;
            if (resource == rd::ResourceType::IN_NORMAL_ROUGHNESS)
                format = render::nrdNormalRoughnessFormat();
            if (resource == rd::ResourceType::IN_VIEWZ || resource == rd::ResourceType::IN_DIFF_CONFIDENCE ||
                resource == rd::ResourceType::IN_SPEC_CONFIDENCE ||
                resource == rd::ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX ||
                resource == rd::ResourceType::IN_PENUMBRA || resource == rd::ResourceType::OUT_SHADOW_TRANSLUCENCY)
                format = render::Format::R32Sfloat;
            std::unique_ptr<render::Texture> texture;
            require(device->createTexture(
                {.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage |
                          render::TextureUsageBits::TransferDestination | render::TextureUsageBits::TransferSource,
                 .format = format,
                 .width = width,
                 .height = height},
                texture));
            std::unique_ptr<render::TextureView> view;
            require(device->createTextureView(*texture, {.format = format}, view));
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
    std::array<float, 2> frame(render::NrdDenoiserMode mode, float diffuse, float specular, bool reset = false,
                               bool confidence = true, bool discard = false)
    {
        require(command->begin());
        for (size_t i = 0; i < textures.size(); ++i) {
            render::TextureBarrierDesc barrier{.texture = textures[i].get(),
                                               .before = initialized ? render::ResourceState::General
                                                                     : render::ResourceState::Undefined,
                                               .after = render::ResourceState::TransferDestination};
            command->barrier({.textures = &barrier, .textureCount = 1});
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
            command->clearColorTexture(*textures[i], render::ResourceState::TransferDestination, value);
            barrier.before = render::ResourceState::TransferDestination;
            barrier.after = render::ResourceState::General;
            command->barrier({.textures = &barrier, .textureCount = 1});
        }
        auto settings = commonSettings(w, h);
        settings.frameIndex = frameIndex++;
        settings.isHistoryConfidenceAvailable = confidence;
        settings.isBaseColorMetalnessAvailable = true;
        settings.enableValidation = true;
        if (reset)
            settings.accumulationMode = rd::AccumulationMode::CLEAR_AND_RESTART;
        require(runtime.setCommonSettings(settings));
        if (mode == render::NrdDenoiserMode::Reference) {
            for (uint32_t i = 0; i < 2; ++i) {
                auto input = pool[static_cast<size_t>(i ? rd::ResourceType::IN_SPEC_RADIANCE_HITDIST
                                                        : rd::ResourceType::IN_DIFF_RADIANCE_HITDIST)];
                auto output = pool[static_cast<size_t>(i ? rd::ResourceType::OUT_SPEC_RADIANCE_HITDIST
                                                         : rd::ResourceType::OUT_DIFF_RADIANCE_HITDIST)];
                runtime.setUserPoolTexture(rd::ResourceType::IN_SIGNAL, *input.texture, *input.view);
                runtime.setUserPoolTexture(rd::ResourceType::OUT_SIGNAL, *output.texture, *output.view);
                require(runtime.denoiseReference(i != 0, *command, *streamer));
            }
        } else {
            require(runtime.denoise(mode, *command, *streamer));
        }
        std::array<float, 2> values{};
        for (uint32_t i = 0; i < 2; ++i) {
            auto output = pool[static_cast<size_t>(mode == render::NrdDenoiserMode::Sigma
                ? rd::ResourceType::OUT_SHADOW_TRANSLUCENCY : (i ? rd::ResourceType::OUT_SPEC_RADIANCE_HITDIST
                : rd::ResourceType::OUT_DIFF_RADIANCE_HITDIST))];
            render::TextureBarrierDesc barrier{.texture = output.texture,
                                               .before = render::ResourceState::General,
                                               .after = render::ResourceState::TransferSource};
            command->barrier({.textures = &barrier, .textureCount = 1});
            command->copyTextureToBuffer({.texture = output.texture,
                                          .buffer = readback.get(),
                                          .bufferOffset = i * 16,
                                          .textureOffsetX = w / 2,
                                          .textureOffsetY = h / 2,
                                          .width = 1,
                                          .height = 1,
                                          .depth = 1});
            barrier.before = render::ResourceState::TransferSource;
            barrier.after = render::ResourceState::General;
            command->barrier({.textures = &barrier, .textureCount = 1});
        }
        require(command->end());
        if (discard) {
            command.reset();
            require(commands->createCommandBuffer(command));
        } else {
            render::CommandBuffer* list[] = {command.get()};
            require(queue->submit({.commandBuffers = list, .commandBufferCount = 1}));
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
    render::NrdUserTexturePool pool;
    render::NrdRuntime runtime;
    uint16_t w = 0, h = 0;
    uint32_t frameIndex = 0;
    bool initialized = false;
};

TEST_F(NrdGpu, ReferenceIndependentSignalsResetResizeAndDiscard)
{
    auto values = frame(render::NrdDenoiserMode::Reference, 2, 10);
    EXPECT_FLOAT_EQ(values[0], 2);
    EXPECT_FLOAT_EQ(values[1], 10);
    values = frame(render::NrdDenoiserMode::Reference, 4, 20);
    EXPECT_FLOAT_EQ(values[0], 3);
    EXPECT_FLOAT_EQ(values[1], 15);
    values = frame(render::NrdDenoiserMode::Reference, 8, 30, true);
    EXPECT_FLOAT_EQ(values[0], 8);
    EXPECT_FLOAT_EQ(values[1], 30);
    frame(render::NrdDenoiserMode::Reference, 100, 100, false, true, true);
    values = frame(render::NrdDenoiserMode::Reference, 6, 12);
    EXPECT_FLOAT_EQ(values[0], 6);
    EXPECT_FLOAT_EQ(values[1], 12);
    createTextures(31, 19);
    values = frame(render::NrdDenoiserMode::Reference, 7, 14);
    EXPECT_FLOAT_EQ(values[0], 7);
    EXPECT_FLOAT_EQ(values[1], 14);
}

TEST_F(NrdGpu, ReblurAndRelaxPreserveFlatRadiance)
{
    if (!device->capabilities().shaderImageGatherExtended)
        GTEST_SKIP() << "REBLUR needs extended image gather";
    for (const auto mode : {render::NrdDenoiserMode::Reblur, render::NrdDenoiserMode::Relax}) {
        for (uint32_t i = 0; i < 4; ++i) {
            SCOPED_TRACE(static_cast<uint32_t>(mode));
            SCOPED_TRACE(i);
            if (mode == render::NrdDenoiserMode::Relax) {
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

TEST(NrdPlan, SigmaScheduleAndIsolatedAllocation)
{
    rd::NrdPlan plan(true);
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

TEST_F(NrdGpu, SigmaLitOccludedResetResizeAndDiscard)
{
    for (uint32_t history : {0u, 5u}) {
        rd::SigmaSettings sigma;
        sigma.maxStabilizedFrameNum = history;
        require(runtime.setSigmaSettings(sigma));
        for (float penumbra : {65504.0f, 0.0f, 0.05f}) {
            auto value = frame(render::NrdDenoiserMode::Sigma, penumbra, 0, true, false);
            EXPECT_NEAR(value[0], penumbra == 65504.0f ? 1.0f : 0.0f, 0.005f);
        }
    }
    frame(render::NrdDenoiserMode::Sigma, 65504, 0, true, false, true);
    EXPECT_NEAR(frame(render::NrdDenoiserMode::Sigma, 0, 0, true, false)[0], 0, 0.005f);
    createTextures(31, 19);
    EXPECT_NEAR(frame(render::NrdDenoiserMode::Sigma, 65504, 0, false, false)[0], 1, 0.005f);
}

class NrdRayTracingGpu : public NrdGpu {
protected:
    bool requiresRayQueries() const override { return true; }
};

TEST_F(NrdRayTracingGpu, RayTracedShadowOcclusionAndHistory)
{
    render::ScreenSpaceShadows shadows;
    render::ScreenSpaceShadowSettings settings;
    settings.angularRadiusDegrees = 0.5f;
    settings.maxDistance = 100000;
    if (!device->capabilities().rayQuery || !device->capabilities().rayTracingAccelerationStructure) {
        GTEST_SKIP() << "Ray queries unavailable";
    }
    // The blocker is behind the camera (z=-1), so it never appears in the depth
    // buffer. It casts onto the visible z=3 plane along the directional-light ray.
    const auto fixtureDirectory = std::filesystem::path(PROJECT_SOURCE_DIR) / ".tmp/NrdRayTracedShadowGeometry";
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
        ASSERT_TRUE(sceneResources.acquire(*device, *queue, {{"path", path.generic_string()}}, &scenes[i],
            render::SceneResourceFeatureBits::Geometry | render::SceneResourceFeatureBits::Materials |
            render::SceneResourceFeatureBits::MaterialTextures | render::SceneResourceFeatureBits::StandardAccelerationStructure,
            snapshot, log)) << log;
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
    std::array<render::GpuPunctualLight, 2> lights{};
    lights[0].positionRange[0] = 1;
    lights[1].directionType[0] = -0.70710678f;
    lights[1].directionType[2] = 0.70710678f;
    lights[1].colorIntensity[0] = lights[1].colorIntensity[1] = lights[1].colorIntensity[2] = 1;
    lights[1].colorIntensity[3] = 10;
    std::unique_ptr<render::Texture> depth;
    std::unique_ptr<render::TextureView> depthView;
    std::unique_ptr<render::Buffer> upload;
    require(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferDestination,
        .format = render::Format::R32Sfloat, .width = w, .height = h}, depth));
    require(device->createTextureView(*depth, {.format = render::Format::R32Sfloat}, depthView));
    require(device->createBuffer({.size = uint64_t(w) * h * 4, .usage = render::BufferUsageBits::TransferSource,
        .memoryLocation = render::MemoryLocation::HostUpload}, upload));
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
                data[y * w + x] = pose.reversedZ ? 1.0f - d : d;
            }
        }
        upload->flush(0, uint64_t(w) * h * 4);
        upload->unmap();
        require(command->begin());
        command->hostWriteBarrier();
        render::TextureBarrierDesc barrier{.texture = depth.get(),
            .before = depthReady ? render::ResourceState::ShaderRead : render::ResourceState::Undefined,
            .after = render::ResourceState::TransferDestination};
        command->barrier({.textures = &barrier, .textureCount = 1});
        command->copyBufferToTexture({.buffer = upload.get(), .texture = depth.get(), .width = w, .height = h});
        barrier.before = render::ResourceState::TransferDestination;
        barrier.after = render::ResourceState::ShaderRead;
        command->barrier({.textures = &barrier, .textureCount = 1});
        render::ScreenSpaceShadowResult output;
        std::string log;
        auto result = shadows.record(*device, *command, *streamer, *depthView, view, lights,
            blocker ? (alphaCutout ? 3 : 1) : 2, 0, settings, output, log,
            &geometry[blocker ? (alphaCutout ? 2 : 1) : 0]);
        if (!result) { throw std::runtime_error(log + render::resultToString(result)); }
        barrier.texture = output.texture;
        barrier.before = render::ResourceState::ShaderRead;
        barrier.after = render::ResourceState::TransferSource;
        command->barrier({.textures = &barrier, .textureCount = 1});
        command->copyTextureToBuffer({.texture = output.texture, .buffer = readback.get(), .width = w, .height = h});
        barrier.before = render::ResourceState::TransferSource;
        barrier.after = render::ResourceState::ShaderRead;
        command->barrier({.textures = &barrier, .textureCount = 1});
        require(command->end());
        if (discard) {
            command.reset();
            require(commands->createCommandBuffer(command));
            streamer->endFrame();
            return std::vector<uint8_t>{};
        }
        render::CommandBuffer* list[] = {command.get()};
        require(queue->submit({.commandBuffers = list, .commandBufferCount = 1}));
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
        for (bool reversed : {false, true}) {
            pose.reversedZ = reversed;
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
    }
    settings.angularRadiusDegrees = 3.0f;
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
    lights[1].directionType[3] = 1.0f;
    lights[1].positionRange[0] = 6.0f;
    lights[1].positionRange[2] = -4.0f;
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
    lights[1].colorIntensity[3] = 0;
    auto noLight = renderShadow(true);
    EXPECT_EQ(*std::min_element(noLight.begin(), noLight.end()), 255);
    w = 31;
    h = 19;
    depthView.reset();
    depth.reset();
    require(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferDestination,
        .format = render::Format::R32Sfloat, .width = w, .height = h}, depth));
    require(device->createTextureView(*depth, {.format = render::Format::R32Sfloat}, depthView));
    depthReady = false;
    lights[1].colorIntensity[3] = 10;
    auto resized = renderShadow(false);
    EXPECT_EQ(resized.size(), size_t(w) * h);
    EXPECT_EQ(*std::min_element(resized.begin(), resized.end()), 255);
}
} // namespace
} // namespace metallic::tests
