#include "Runtime/Render/RenderGraph/NrdRuntime.h"
#include "Runtime/Render/Denoising/NrdPlan.h"
#include "Runtime/Render/SlangCompiler.h"

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
    void SetUp() override
    {
        ASSERT_TRUE(video.ready) << SDL_GetError();
        const auto result = render::createDevice(
            {.applicationName = "Metallic native NRD tests",
             .enableValidation = true,
             .enableBindlessDescriptorHeap = true,
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
                resource == rd::ResourceType::IN_DISOCCLUSION_THRESHOLD_MIX)
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
            auto output = pool[static_cast<size_t>(i ? rd::ResourceType::OUT_SPEC_RADIANCE_HITDIST
                                                     : rd::ResourceType::OUT_DIFF_RADIANCE_HITDIST)];
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
} // namespace
} // namespace metallic::tests
