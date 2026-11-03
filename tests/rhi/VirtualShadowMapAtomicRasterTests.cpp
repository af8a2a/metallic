#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <slang.h>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

template<typename T>
T checked(Result<T> result, std::string_view operation)
{
    if (!result) { throw std::runtime_error(std::string(operation) + ": " + resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

class VirtualShadowMapAtomicRasterTest final : public RHITest
{
public:
    VirtualShadowMapAtomicRasterTest()
    {
        name = "virtual_shadow_map_atomic_micro_raster";
        type = RHITestType::Rendering;
    }
    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint32_t page = 16, width = 2 * page, height = page;
        constexpr uint32_t trianglesPerPage = 9, pixelCount = width * height;
        bench::ValidationRecorder validation;
        try {
            auto created = createDevice({.applicationName = "VSM opaque atomic raster probe",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires Vulkan bindless storage images"); }
            auto device = checked(std::move(created), "create device");
            const auto bits = device->capabilities().subPixelPrecisionBits;
            if (bits > 8) { return RHITestResult::skip("Integer edge setup covers devices with up to 8 subpixel bits"); }
            auto registry = checked(ResourceRegistry::forDevice(*device), "device registry");
            std::string log;
            const auto format = std::to_string(SLANG_IMAGE_FORMAT_r32ui);
            const std::array macros{SlangMacroDefine{"VSM_R32UI_FORMAT", format.c_str()}};
            const auto compile = [&](const char* entry) {
                auto compiled = compileSlangShaderToSpirv({.moduleName = "VirtualShadowMapAtomicRasterProbe",
                    .entryPointName = entry, .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = macros}, log);
                if (!compiled) { throw std::runtime_error(std::string(entry) + ": " + log); }
                return checked(device->createShaderModule({.spirv = compiled->spirv}), entry);
            };
            auto vs = compile("depthVertex"), fs = compile("depthFragment");
            auto init = compile("initializeDepth"), raster = compile("rasterDepth");
            auto hw = checked(device->createGraphicsPipeline({.vertexShader = {vs.get()}, .fragmentShader = {fs.get()},
                .depthStencilFormat = Format::D32Sfloat, .rasterization = {.cullMode = CullMode::None},
                .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::GreaterEqual}}),
                "D32 hardware pipeline");
            struct Params { uint32_t atlas, subpixelBits; };
            const auto compute = [&](const auto& shader) {
                return checked(device->createComputePipeline({.computeShader = {shader.get()}, .usesBindlessHeap = true,
                    .bindlessUserPushDataSize = sizeof(Params)}), "atomic image pipeline");
            };
            auto clear = compute(init), sw = compute(raster);
            std::array<std::unique_ptr<Texture>, 2> images;
            std::array<std::unique_ptr<TextureView>, 2> views;
            std::array<std::unique_ptr<Buffer>, 2> buffers;
            for (uint32_t i = 0; i < images.size(); ++i) {
                const auto fmt = i ? Format::R32Uint : Format::D32Sfloat;
                images[i] = checked(device->createTexture({.usage = TextureUsageBits::TransferSource |
                    (i ? TextureUsageBits::Storage : TextureUsageBits::DepthStencilAttachment),
                    .format = fmt, .width = width, .height = height}), "atlas");
                views[i] = checked(device->createTextureView(*images[i], {.format = fmt}), "atlas view");
                buffers[i] = checked(device->createBuffer({.size = pixelCount * sizeof(uint32_t),
                    .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}), "atlas readback");
            }
            auto storage = checked(registry->storageImage(*views[1]), "register atomic atlas");
            const Params push{storage.shaderIndex(), bits};
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            checked(gpu.initialize(*device), "begin");
            auto& commands = *gpu.commands;
            TextureBarrierDesc hwReady{.texture = images[0].get(), .newLayout = TextureLayout::DepthStencilAttachment,
                .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite}};
            TextureBarrierDesc swReady{.texture = images[1].get(), .newLayout = TextureLayout::General,
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
            const std::array ready{hwReady, swReady};
            checked(commands.synchronize({.textures = ready}), "prepare atlases");
            const RenderingAttachmentDesc depth{.view = views[0].get(), .layout = TextureLayout::DepthStencilAttachment,
                .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = .125f};
            checked(commands.beginRendering({.renderArea = {.width = width, .height = height}, .depthStencilAttachment = &depth}), "begin D32");
            checked(commands.setViewport({.width = float(width), .height = float(height), .maxDepth = 1}), "viewport");
            checked(commands.bindExecution(hw->execution()), "bind D32");
            for (uint32_t slot = 0; slot < 2; ++slot) {
                commands.setScissor({.x = int32_t(slot * page), .width = page, .height = page});
                checked(commands.draw(3 * trianglesPerPage, 1, slot * 3 * trianglesPerPage), "scissored triangles");
            }
            commands.endRendering();
            checked(registry->bind(commands, {&storage, 1}), "bind R32_UINT");
            checked(commands.bindExecution(clear->execution(), &push, sizeof(push)), "bind clear");
            checked(commands.dispatch(width / 8, height / 8, 1), "initialize retained depth");
            swReady.oldLayout = swReady.newLayout;
            swReady.before = swReady.after;
            checked(commands.synchronize({.textures = {&swReady, 1}}), "publish retained depth");
            checked(commands.bindExecution(sw->execution(), &push, sizeof(push)), "bind raster");
            checked(commands.dispatch(trianglesPerPage, 1, 1), "raster opaque microtriangles");
            for (uint32_t i = 0; i < images.size(); ++i) {
                auto src = ready[i];
                src.oldLayout = src.newLayout; src.before = src.after;
                src.newLayout = TextureLayout::TransferSource; src.after = {PipelineStageBits::Transfer, AccessBits::TransferRead};
                checked(commands.synchronize({.textures = {&src, 1}}), "publish depth image");
                checked(commands.copyTextureToBuffer({.texture = images[i].get(), .buffer = checked(buffers[i]->slice(), "slice"),
                    .width = width, .height = height}), "read image");
                const BufferBarrierDesc host{.buffer = buffers[i].get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead}};
                checked(commands.synchronize({.buffers = {&host, 1}}), "publish readback");
            }
            checked(gpu.submitAndWait(), "submit");
            std::array<std::array<uint32_t, pixelCount>, 2> values;
            for (uint32_t i = 0; i < values.size(); ++i) {
                buffers[i]->invalidate();
                auto* data = buffers[i]->map();
                if (!data) { throw std::runtime_error("Failed to map readback"); }
                std::memcpy(values[i].data(), data, sizeof(values[i]));
                buffers[i]->unmap();
            }
            uint32_t touched = 0, different = 0;
            for (size_t i = 0; i < pixelCount; ++i) {
                touched += values[0][i] != std::bit_cast<uint32_t>(.125f);
                different += values[0][i] != values[1][i];
            }
            const auto dir = context.outputDirectory / "VirtualShadowMapAtomicRaster";
            std::filesystem::create_directories(dir);
            std::ofstream(dir / "Comparison.json") << nlohmann::json{{"triangles", 2 * trianglesPerPage},
                {"pixels", pixelCount}, {"touched", touched}, {"differentDepthBits", different},
                {"subpixelBits", bits}, {"validation", validation.snapshot()}}.dump(2);
            if (validation.failed() || !touched || different) { return RHITestResult::fail("Atomic image mismatch; see Comparison.json"); }
            return RHITestResult::pass("Opaque page-local R32_UINT atomic depth matches D32 hardware depth bitwise");
        } catch (const std::exception& e) { return RHITestResult::fail(e.what()); }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapAtomicRasterTest);
} // namespace
} // namespace metallic::tests
