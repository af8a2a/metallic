#include "RHITest.h"
#include "Runtime/Material/MaterialValueIR.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include "Runtime/Scene/Scene.h"
#include "harness/Fixtures.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
void check(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}
class MaterialValueIRTest final : public RHITest
{
public:
    MaterialValueIRTest() { name = "material_value_ir"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext&) override
    {
        try {
            auto graph = Json::parse(R"({"version":2,"nodes":{
                "shared":{"op":"mul","args":[{"op":"parameter","index":0},0.5]},
                "unused":{"op":"textureSample","texture":"invalid","footprint":"Implicit","args":[]}},
                "outputs":{"baseColor":{"ref":"shared"},"roughness":{"ref":"shared"},"coverage":{"ref":"shared"}}})");
            const auto ir = MaterialValueIR::lower(graph);
            check(ir.nodes().size() == 3 && ir.usage().parameterMask == 1 && ir.usage().textureMask == 0, "CSE/DCE/resource analysis failed");
            check(ir.slice(true).nodes().size() == 3 && ir.slice(false).outputs().size() == 2, "Automatic output slicing failed");
            check(compileMaterialCoverageSlice(ir.slice(true)).instructions.size() == 3, "Coverage backend did not use IR");
            graph["nodes"]["renamed"] = graph["nodes"]["shared"]; graph["nodes"].erase("shared");
            for (auto& value : graph["outputs"]) { value["ref"] = "renamed"; }
            check(MaterialValueIR::lower(graph).canonical() == ir.canonical(), "Author node names affected stable identity");
            const auto folded = MaterialValueIR::parse(R"({"version":1,"roughness":{"op":"add","args":[0.25,0.25]}})");
            const auto literal = MaterialValueIR::parse(R"({"version":1,"roughness":0.5})");
            check(folded.nodes().size() == 1 && folded.hash() == literal.hash(), "Constant folding/hash failed");
            check(MaterialValueIR::parse(R"({"version":1,"roughness":{"op":"select","args":[0,{"op":"textureSample"},0.5]}})").hash() == literal.hash(),
                "Dead invalid texture branch survived DCE");
            scene::RenderMaterial material;
            material.valueProgram = R"({"version":1,"roughness":{"op":"add","args":[0.25,0.25]}})";
            std::string log;
            const auto first = MaterialValueProgramSet::create({&material, 1}, log);
            material.valueProgram = R"({"version":1,"roughness":0.5})";
            const auto second = MaterialValueProgramSet::create({&material, 1}, log);
            check(first && second && first->key() == second->key(), "Production code did not deduplicate optimized IR");
            for (const char* policy : {"ExplicitLOD", "SampleGrad", "RayCone"}) {
                Json sample{{"op", "textureSample"}, {"texture", "normal"}, {"footprint", policy},
                    {"args", Json::array({Json{{"op", "uv"}}, 0})}};
                if (std::string_view(policy) == "SampleGrad") { sample["args"].push_back(0); }
                const auto texture = MaterialValueIR::lower({{"version", 1}, {"baseColor", sample}});
                check(texture.usage().textureMask == 4 && texture.usage().footprintMask != 0, "Explicit resource/footprint manifest missing");
                if (std::string_view(policy) == "RayCone") { check((texture.usage().features & ValueRayCone) != 0, "Ray-cone feature missing"); }
            }
            for (const char* invalid : {
                R"({"version":2,"nodes":{"a":{"ref":"b"},"b":{"ref":"a"}},"outputs":{"baseColor":{"ref":"a"}}})",
                R"({"version":2,"nodes":{},"outputs":{"baseColor":{"ref":"missing"}}})",
                R"({"version":1,"baseColor":{"op":"textureSample","texture":"baseColor","args":[0,0]}})",
                R"({"version":1,"baseColor":{"op":"swizzle","components":"xqzw","args":[0]}})",
                R"({"version":1,"baseColor":{"op":"clamp","args":[0,1]}})"}) {
                bool rejected = false;
                try { MaterialValueIR::parse(invalid); } catch (const std::exception&) { rejected = true; }
                check(rejected, "Invalid live IR accepted");
            }
            return RHITestResult::pass("Validation, constant folding, DCE, CSE, manifests, slicing and stable production hash");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialValueIRTest);

class MaterialValueClosureIRTest final : public RHITest
{
public:
    MaterialValueClosureIRTest() { name = "material_value_closure_ir"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext&) override
    {
        try {
            const Json slab{{"op", "slab"}, {"reflectance", {{"op", "parameter"}, {"index", 0}}}};
            Json graph{{"version", 3}, {"nodes", Json::object()}, {"outputs", {{"coverage", 1}}}, {"closure", slab}};
            auto single = MaterialValueIR::lower(graph);
            check(single.closure().has_value() && single.slice(false).closure().has_value(), "Surface lost Closure topology");
            check(!single.slice(true).closure() && single.slice(true).outputs().size() == 1, "Coverage retained Closure topology");
            graph["closure"] = {{"op", "mix"}, {"a", slab}, {"b", slab}, {"weight", {{"op", "parameter"}, {"index", 1}}}};
            const auto dual = MaterialValueIR::lower(graph);
            check(dual.hash() != single.hash() && dual.usage().parameterMask == 3, "Closure topology or dynamic inputs lost in identity");
            scene::RenderMaterial material; material.alphaMode = "MASK"; material.valueProgram = graph.dump();
            std::string log;
            auto set = MaterialValueProgramSet::create({&material, 1}, log);
            check(set && set->manifests()[0].closureFamily == MaterialClosureFamily::DualSlabClosure &&
                set->manifests()[0].closureComplexity.payloadBytes == 96, "Production manifest lost family/budget");
            material.valueParameters[4] = 0.8f;
            auto updated = MaterialValueProgramSet::create({&material, 1}, log);
            check(updated && updated->key() == set->key(), "Dynamic Slab input changed program identity");
            graph["closure"]["a"] = graph["closure"];
            bool rejected = false;
            try { MaterialValueIR::lower(graph); } catch (const std::exception&) { rejected = true; }
            check(rejected, "Nested Closure exceeded realtime budget silently");
            return RHITestResult::pass("Value to Closure topology, dynamic inputs, coverage slicing, family manifest and budget rejection");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialValueClosureIRTest);

class MaterialValueIRTextureTest final : public RHITest
{
public:
    MaterialValueIRTextureTest() { name = "material_value_ir_texture_footprints"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            check(bool(bench::createTestDevice(context, {.applicationName = "Value IR footprints",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if ((message.severity == render::ValidationSeverity::Error) &&
                        (render::hasFlag(message.type, render::ValidationCategory::Validation))) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}})
                .transform([&](auto value) { device = std::move(value); })), "IR device failed");
            std::vector<scene::RenderMaterial> materials(14);
            for (uint32_t i = 0; i < 5; ++i) {
                Json sample{{"op", "textureSample"}, {"texture", "baseColor"}, {"footprint", "ExplicitLOD"},
                    {"args", Json::array({Json{{"op", "uv"}}, Json{{"op", "parameter"}, {"index", 0}}})}};
                materials[i].valueParameters[0] = i == 1 ? 1 : i == 2 ? 0.5f : 0;
                if (i == 3) {
                    sample["footprint"] = "SampleGrad";
                    sample["args"] = Json::array({Json{{"op", "uv"}}, Json::array({1,0,0,0}), Json::array({0,1,0,0})});
                }
                if (i == 4) { sample["footprint"] = "RayCone"; }
                materials[i].valueProgram = Json{{"version", 1}, {"baseColor", sample}}.dump();
            }
            materials[5].valueProgram = R"({"version":1,"baseColor":{"op":"clamp","args":[{"op":"parameter","index":0},0,1]}})";
            materials[5].valueParameters = {-1, 0.5f, 2, 0};
            materials[6].valueProgram = R"({"version":1,"baseColor":{"op":"normalize","args":[{"op":"parameter","index":0}]}})";
            materials[6].valueParameters = {3, 0, 4, 0};
            materials[7].valueProgram = R"({"version":1,"baseColor":{"op":"normalMap","args":[{"op":"parameter","index":0}]}})";
            materials[7].valueParameters = {0.5f, 0.5f, 1, 0};
            materials[8].valueProgram = R"({"version":1,"baseColor":{"op":"uvTransform","args":[{"op":"uv"},[2,0,0.25,0],[0,1,0.5,0]]}})";
            materials[9].valueProgram = R"({"version":1,"baseColor":{"op":"select","args":[{"op":"parameter","index":0},{"op":"swizzle","components":"zyxx","args":[{"op":"parameter","index":1}]},0]}})";
            materials[9].valueParameters = {1, 0, 0, 0, 0.2f, 0.3f, 0.4f, 0};
            // These programs exercise the production wide-source -> legacy IR
            // -> working output path, including a value above display white.
            materials[10].valueProgram = R"({"version":1,"baseColor":{"op":"textureSample","texture":"baseColor","footprint":"ExplicitLOD","args":[{"op":"uv"},0]}})";
            materials[11].valueProgram = R"({"version":1,"emissive":{"op":"textureSample","texture":"emissive","footprint":"ExplicitLOD","args":[{"op":"uv"},0]}})";
            materials[12].valueProgram = R"({"version":1,"emissive":{"op":"mul","args":[{"op":"textureSample","texture":"emissive","footprint":"ExplicitLOD","args":[{"op":"uv"},0]},4]}})";
            materials[13].valueProgram = R"({"version":1,"baseColor":{"op":"textureSample","texture":"baseColor","footprint":"ExplicitLOD","args":[{"op":"uv"},1]}})";
            for (uint32_t i = 10; i < materials.size(); ++i) {
                materials[i].baseColorTexture.colorMetadata = {TextureSemantic::Color, kACEScg};
                materials[i].emissiveTexture.colorMetadata = {TextureSemantic::Color, kACEScg};
                // The probe reads the same CPU-derived texture flags as upload.
                materials[i].valueParameters[12] = float(textureColorFlags(materials[i].baseColorTexture.colorMetadata));
            }
            std::string log;
            const auto values = MaterialValueProgramSet::create(materials, log);
            check(bool(values), log.c_str());
            using Float4 = std::array<float, 4>;
            const std::array<Float4, 5> texels{{{1,0,0,1}, {1,0,0,1}, {1,0,0,1}, {1,0,0,1}, {0,1,0,1}}};
            std::unique_ptr<Buffer> input, upload, output;
            auto allocate = [&](size_t bytes, MemoryLocation memory, BufferUsageBits usage, auto& target) {
                check(bool(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = memory})
                    .transform([&](auto value) { target = std::move(value); })), "IR buffer failed");
            };
            allocate(values->inputBytes().size_bytes(), MemoryLocation::HostUpload, BufferUsageBits::Storage, input);
            allocate(sizeof(texels), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
            allocate(materials.size() * sizeof(Float4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
            auto write = [&](Buffer& buffer, const void* data, size_t bytes) {
                void* mapped = buffer.map(); check(mapped != nullptr, "IR upload mapping failed");
                std::memcpy(mapped, data, bytes); buffer.flush(); buffer.unmap();
            };
            write(*input, values->inputBytes().data(), values->inputBytes().size_bytes()); write(*upload, texels.data(), sizeof(texels));
            std::unique_ptr<Texture> texture;
            std::unique_ptr<TextureView> view;
            check(bool(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::RGBA32Sfloat, .width = 2, .height = 2, .mipCount = 2})
                .transform([&](auto value) { texture = std::move(value); })), "IR texture failed");
            check(bool(device->createTextureView(*texture, {.format = Format::RGBA32Sfloat, .range = {.mipCount = 2}})
                .transform([&](auto value) { view = std::move(value); })), "IR texture view failed");
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            check(bool(gpu.initialize(*device)), "IR commands failed");
            const TextureBarrierDesc before{.texture = texture.get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}, .range = {.mipCount = 2}};
            check(bool(gpu.commands->synchronize({.textures = {&before, 1}})), "IR upload barrier failed");
            for (uint32_t mip = 0; mip < 2; ++mip) {
                check(bool(upload->slice({mip ? 4 * sizeof(Float4) : 0}).and_then([&](const auto& slice) {
                    return gpu.commands->copyBufferToTexture({.texture = texture.get(), .buffer = slice,
                        .width = mip ? 1u : 2u, .height = mip ? 1u : 2u, .mipLevel = mip});
                })), "IR mip upload failed");
            }
            const TextureBarrierDesc after{.texture = texture.get(), .oldLayout = TextureLayout::TransferDestination,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}, .range = {.mipCount = 2}};
            check(bool(gpu.commands->synchronize({.textures = {&after, 1}})), "IR sample barrier failed");
            std::filesystem::path directory;
            const bool written = values->writeInclude(PROJECT_SOURCE_DIR "/.cache/materials", directory, log);
            check(written, log.c_str());
            const auto search = directory.string(); const char* paths[] = {search.c_str()};
            const SlangMacroDefine macros[] = {{"METALLIC_CUSTOM_MATERIALS", "1"}};
            auto shader = compileSlangShaderToSpirv({.moduleName = "MaterialValueIRTextureProbe", .entryPointName = "materialValueIRTextureProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .additionalSearchPaths = paths, .macroDefines = macros}, log);
            check(bool(shader), log.c_str());
            const ComputeResourceField fields[] = {
                {0, ComputeResourceBindingKind::StorageBuffer, offsetof(SceneResourceParameters, probeOutput), ComputeResourceFieldFormat::Handle},
                {9, ComputeResourceBindingKind::SampledImage, offsetof(SceneResourceParameters, materialTextures), ComputeResourceFieldFormat::IndexSpan},
                {97, ComputeResourceBindingKind::StorageBuffer, offsetof(SceneResourceParameters, materialValues), ComputeResourceFieldFormat::Handle}};
            const ComputeResourceBindingDesc bindings[] = {{0}, {9, ComputeResourceBindingKind::SampledImage, 1}, {97}};
            ComputeProgram program;
            const auto initialized = program.initialize(*device, {.spirv = shader->spirv, .bindings = bindings, .requiresRayQuery = false,
                .resourceParameters = {sizeof(SceneResourceParameters), fields}}, log);
            check(bool(initialized), log.c_str());
            TextureView* views[] = {view.get()};
            const ComputeDispatchBinding dispatch[] = {{.binding = 0, .buffer = output.get()},
                {.binding = 9, .textureViews = views}, {.binding = 97, .buffer = input.get()}};
            check(bool(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = dispatch, .groupCountX = uint32_t(materials.size())})), "IR dispatch failed");
            check(bool(gpu.submitAndWait()), "IR GPU execution failed");
            output->invalidate(); const auto* data = static_cast<const Float4*>(output->map());
            check(data != nullptr, "IR readback failed");
            std::array<Float4, 14> actual; std::memcpy(actual.data(), data, sizeof(actual)); output->unmap();
            const std::array<color::RGB, 10> authoredExpected{{{1,0,0}, {0,1,0}, {0.5f,0.5f,0}, {0,1,0}, {0,1,0},
                {0,0.5f,1}, {0.6f,0,0.8f}, {0,0,1}, {0.75f,0.75f,0}, {0.4f,0.3f,0.2f}}};
            std::array<color::RGB, 14> expected;
            for (size_t i = 0; i < authoredExpected.size(); ++i) { expected[i] = color::fromLinearRec709(authoredExpected[i]); }
            expected[10] = color::fromSource({1,0,0}, kACEScg);
            expected[11] = color::fromSource({1,0,0}, kACEScg);
            expected[12] = color::fromSource({4,0,0}, kACEScg);
            expected[13] = color::fromSource({0,1,0}, kACEScg);
            // Compatibility mode still bounds its own Rec.709 material basis;
            // signed out-of-gamut coordinates are not valid reflectance there.
            for (size_t i = 10; i < expected.size(); ++i) {
                for (float& channel : expected[i]) { channel = std::clamp(channel, 0.0f, i == 11 || i == 12 ? 1e6f : 1.0f); }
            }
            for (uint32_t i = 0; i < actual.size(); ++i) { for (uint32_t c = 0; c < 3; ++c) {
                check(std::isfinite(actual[i][c]) && std::abs(actual[i][c] - expected[i][c]) < 1e-5f,
                    ("IR footprint/arithmetic mismatch case=" + std::to_string(i) + " actual=" + Json(actual).dump()).c_str());
            }}
            check(validationErrors == 0, "Value IR GPU probe emitted Vulkan validation errors");
            return RHITestResult::pass("Production IR: five mip/footprint cases, five dynamic arithmetic cases and four native AP1 reflectance/emission/HDR cases");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialValueIRTextureTest);
} // namespace
} // namespace metallic::tests
