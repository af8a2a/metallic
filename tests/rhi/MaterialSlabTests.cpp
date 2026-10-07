#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Material/MaterialClosureIR.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;
void check(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}
template<typename F> void rejects(F action, const char* diagnostic)
{
    try { action(); }
    catch (const std::invalid_argument& error) {
        check(std::string(error.what()).find(diagnostic) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("Expected Closure IR rejection");
}
MaterialClosureNode slab(float r, float g, float b, float tau = 0)
{
    MaterialClosureNode node;
    node.slab.reflectance = {r, g, b, 0};
    node.slab.opticalDepth = {tau, tau * 0.5f, tau * 0.25f, 0};
    return node;
}
MaterialClosureNode combine(MaterialClosureOp op, uint32_t a, uint32_t b, float weight = 0.5f)
{
    MaterialClosureNode node;
    node.op = op; node.operands = {a, b}; node.weight = weight;
    return node;
}

class MaterialClosureIRTest final : public RHITest
{
public:
    MaterialClosureIRTest() { name = "material_closure_ir_lowering"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext&) override
    {
        try {
            std::vector nodes{slab(0.2f, 0.3f, 0.4f), slab(0.8f, 0.6f, 0.4f), combine(MaterialClosureOp::Mix, 0, 1)};
            const auto ir = MaterialClosureIR::create(nodes, 2);
            auto reordered = nodes;
            std::swap(reordered[0], reordered[1]); reordered[2].operands = {1, 0};
            check(MaterialClosureIR::create(reordered, 2).hash() == ir.hash(), "Authoring order changed canonical hash");
            const auto dual = lowerMaterialClosure(ir);
            const auto single = lowerMaterialClosure(MaterialClosureIR::create(nodes, 0));
            check(single.family == MaterialClosureFamily::SingleSlabClosure && single.complexity.payloadBytes == 48,
                "Single family/payload mismatch");
            check(dual.family == MaterialClosureFamily::DualSlabClosure && dual.complexity.closureRecordCount == 2 &&
                dual.complexity.scatteringLobeCount == 2 && dual.complexity.normalBasisCount == 1 &&
                dual.complexity.layerDepth == 0 && dual.complexity.operatorCount == 1 && dual.complexity.payloadBytes == 96,
                "Independent complexity metrics mismatch");
            nodes[2].op = MaterialClosureOp::Layer;
            const auto layer = lowerMaterialClosure(MaterialClosureIR::create(nodes, 2));
            check(layer.family == dual.family && layer.complexity.layerDepth == 1 && layer.packet.control[0] == 2,
                "Different programs did not lower to the same dual family");
            nodes.push_back(combine(MaterialClosureOp::Layer, 0, 2));
            const auto larger = MaterialClosureIR::create(nodes, 3);
            check(larger.complexity().closureRecordCount == 3 && larger.complexity().layerDepth == 2,
                "IR inherited backend limits or counted DAG sharing as one scattering occurrence");
            rejects([&] { lowerMaterialClosure(larger); }, "maxClosures");
            auto profile = RealtimeBackendProfile{};
            profile.maxClosures = 4; profile.maxOperators = 3; profile.maxLayerDepth = 3;
            rejects([&] { lowerMaterialClosure(larger, profile); }, "no executable family");
            profile = {}; profile.maxOperators = 0;
            rejects([&] { lowerMaterialClosure(ir, profile); }, "maxOperators");
            profile = {}; profile.maxPayloadBytes = 48;
            rejects([&] { lowerMaterialClosure(ir, profile); }, "maxPayloadBytes");
            profile = {}; profile.maxLayerDepth = 0;
            rejects([&] { lowerMaterialClosure(MaterialClosureIR::create(nodes, 2), profile); }, "maxLayerDepth");
            nodes[1].normalBasis = 1;
            rejects([&] { lowerMaterialClosure(MaterialClosureIR::create(nodes, 2)); }, "maxNormalBases");
            profile = {}; profile.maxNormalBases = 2;
            rejects([&] { lowerMaterialClosure(MaterialClosureIR::create(nodes, 2), profile); }, "shared context");
            nodes[1].normalBasis = 0; nodes.resize(3); nodes[2].op = MaterialClosureOp::Mix;
            nodes.push_back(slab(-1, -1, -1));
            check(MaterialClosureIR::create(nodes, 2).canonical() == ir.canonical(), "Unreachable data changed canonical IR");
            nodes[0].slab.reflectance[0] = std::numeric_limits<float>::quiet_NaN();
            rejects([&] { MaterialClosureIR::create(nodes, 2); }, "reflectance");
            nodes[0] = slab(0.2f, 0.3f, 0.4f, -1);
            rejects([&] { MaterialClosureIR::create(nodes, 2); }, "optical depth");
            nodes[0] = slab(0.2f, 0.3f, 0.4f); nodes[2].weight = 1.1f;
            rejects([&] { MaterialClosureIR::create(nodes, 2); }, "Mix weight");
            nodes[2].weight = 0.5f; nodes[2].operands[0] = 2;
            rejects([&] { MaterialClosureIR::create(nodes, 2); }, "precede");
            rejects([&] { MaterialClosureIR::create({}, 0); }, "size/root");
            return RHITestResult::pass("Closure DAG validation, reachability, metrics, profile budgets and canonical family lowering");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialClosureIRTest);

struct SlabProbeParams
{
    GPUBufferSpan packets, output;
    uint32_t caseIndex;
    float cosineOut;
    uint32_t width, height;
};
static_assert(sizeof(SlabProbeParams) == 40 && offsetof(SlabProbeParams, caseIndex) == 24);
constexpr uint64_t kSlabProbeABI = 0x534c414250520001ull;

class MaterialSlabTest final : public RHITest
{
public:
    MaterialSlabTest() { name = "material_slab_furnace_sample_render"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::vector<LoweredMaterialClosure> cases;
            const auto addCase = [&](MaterialClosureNode a, MaterialClosureNode b, MaterialClosureOp op, float weight) {
                const std::array nodes{a, b, combine(op, 0, 1, weight)};
                cases.push_back(lowerMaterialClosure(MaterialClosureIR::create(nodes, op == MaterialClosureOp::Slab ? 0 : 2)));
            };
            const auto top = slab(0.12f, 0.25f, 0.5f, 0.2f), bottom = slab(0.8f, 0.45f, 0.12f);
            addCase(top, bottom, MaterialClosureOp::Slab, 0);
            addCase(top, bottom, MaterialClosureOp::Mix, 0.35f);
            addCase(top, bottom, MaterialClosureOp::Layer, 0);
            addCase(slab(1, 1, 1), bottom, MaterialClosureOp::Slab, 0);
            addCase(slab(0, 0, 0), bottom, MaterialClosureOp::Slab, 0);
            for (float weight : {0.0f, 1.0f, 0.5f}) {
                addCase(top, bottom, MaterialClosureOp::Mix, weight);
                addCase(slab(1, 1, 1), slab(1, 1, 1), MaterialClosureOp::Mix, weight);
            }
            for (float reflectance : {0.0f, 0.2f, 0.8f, 1.0f}) {
                for (float depth : {0.0f, 0.25f, 100.0f}) {
                    addCase(slab(reflectance, reflectance, reflectance, depth), slab(1, 1, 1), MaterialClosureOp::Layer, 0);
                }
            }
            std::vector<MaterialClosurePacket> packets;
            for (const auto& value : cases) { packets.push_back(value.packet); }
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            check(bool(bench::createTestDevice(context, {.applicationName = "Slab Closure IR",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if ((message.severity == render::ValidationSeverity::Error) &&
                        (render::hasFlag(message.type, render::ValidationCategory::Validation))) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}}).transform([&](auto value) { device = std::move(value); })), "Device failed");
            ResourceRegistry registry;
            check(bool(registry.initialize(*device)), "Registry failed");
            std::unique_ptr<Buffer> input, output;
            check(bool(device->createBuffer({.size = packets.size() * sizeof(MaterialClosurePacket), .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto value) { input = std::move(value); })), "Input allocation failed");
            check(bool(device->createBuffer({.size = 256 * 192 * sizeof(Float4), .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { output = std::move(value); })), "Output allocation failed");
            auto* mapped = input->map(); check(mapped != nullptr, "Input map failed");
            std::memcpy(mapped, packets.data(), packets.size() * sizeof(MaterialClosurePacket)); input->flush(); input->unmap();
            std::array<ComputeKernel, 4> kernels;
            const std::array entries{"singleContractMain", "dualContractMain", "singleRenderMain", "dualRenderMain"};
            std::string log;
            for (size_t i = 0; i < kernels.size(); ++i) {
                auto compiled = compileSlangShaderToSpirv({.moduleName = "MaterialSlabProbe", .entryPointName = entries[i],
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, {.enableDiskCache = false}, log);
                check(bool(compiled), "Slab compile failed: " + log);
                for (const auto& path : compiled->dependencies) {
                    check(path.find("OpenPBR") == std::string::npos && path.find("SceneSurface") == std::string::npos,
                        "Slab prototype depends on the production reference BSDF");
                }
                auto initialized = kernels[i].initialize(*device, {.spirv = compiled->spirv,
                    .parameters = parameterAbi<SlabProbeParams>(kSlabProbeABI, ParameterTransport::InlinePush)}, log);
                check(bool(initialized), "Slab pipeline failed: " + log);
            }
            const auto dispatch = [&](uint32_t caseIndex, float mu, bool render) {
                const bool single = cases[caseIndex].family == MaterialClosureFamily::SingleSlabClosure;
                auto& kernel = kernels[(render ? 2 : 0) + (single ? 0 : 1)];
                bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                check(bool(gpu.initialize(*device)), "Commands failed");
                ParameterWriter writer(*device, registry);
                const SlabProbeParams params{writer.bufferSpan<Float4>(input.get()), writer.bufferSpan<Float4>(output.get()), caseIndex, mu, 256, 192};
                auto encoded = writer.encode(params, kSlabProbeABI, ParameterTransport::InlinePush);
                check(bool(encoded), "Encode failed");
                check(bool(kernel.dispatch(*gpu.commands, *encoded, render ? 32 : 64, render ? 24 : 1)), "Dispatch failed");
                check(bool(gpu.submitAndWait()), "GPU submit failed");
                output->invalidate();
                const auto* pixels = static_cast<const Float4*>(output->map());
                check(pixels != nullptr, "Readback failed");
                std::vector<Float4> data(pixels, pixels + (render ? 256 * 192 : 4096 * 9)); output->unmap();
                for (const auto& pixel : data) { for (float value : pixel) { check(std::isfinite(value), "Nonfinite Slab output"); } }
                return data;
            };
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream report(context.outputDirectory / "SlabEnergy.txt");
            report << "4096 samples/case/view; RGB energy: cosine sampling vs uniform hemisphere integration\n";
            const double pi = std::numbers::pi;
            const auto near = [](double a, double b) { return std::abs(a - b) < 5e-5; };
            for (uint32_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
                const auto& value = cases[caseIndex];
                const auto& packet = value.packet;
                for (float muOut : {1.0f, 0.5f, 0.01f}) {
                    const auto data = dispatch(caseIndex, muOut, false);
                    std::array<double, 3> sampled{}, integrated{};
                    double pdfIntegral = 0, meanCosine = 0;
                    for (size_t lane = 0; lane < 4096; ++lane) {
                        const auto base = lane * 9;
                        const auto& wi = data[base];
                        const double mu = (caseIndex & 1u) ? wi[1] * 0.6 + wi[2] * 0.8 : wi[2];
                        const double quadratureMu = (lane + 0.5) / 4096.0;
                        meanCosine += mu;
                        check(mu > 0 && near(wi[0] * wi[0] + wi[1] * wi[1] + wi[2] * wi[2], 1), "Invalid sampled direction");
                        check(near(wi[3], mu / pi) && near(wi[3], data[base + 2][3]) && near(wi[3], data[base + 7][3]), "PDF mismatch");
                        check(data[base + 1][3] == 5 && data[base + 4][3] == 1 && data[base + 8][3] > 0, "Flags/eta/endpoint mismatch");
                        check(data[base + 5][3] == value.complexity.payloadBytes, "CPU metrics differ from Slang sizeof(Closure)");
                        for (float component : data[base + 6]) { check(component == 0, "Backface/outside support is nonzero"); }
                        pdfIntegral += data[base + 3][3] * 2 * pi / 4096;
                        for (size_t c = 0; c < 3; ++c) {
                            const double a = packet.first.reflectance[c], b = packet.second.reflectance[c];
                            const auto expected = [&](double cosine) {
                                if (packet.control[0] == 0) { return a / pi; }
                                if (packet.control[0] == 1) { return ((1 - packet.control[1]) * a + packet.control[1] * b) / pi; }
                                return (a + (1-a) * (1-a) * b * std::exp(-packet.first.opticalDepth[c] * (1/cosine + 1/muOut))) / pi;
                            };
                            check(near(data[base + 1][c], expected(mu)) && near(data[base + 2][c], expected(mu)), "Sample/eval differs from CPU scattering oracle");
                            check(near(data[base + 3][c], expected(quadratureMu)), "Uniform eval differs from CPU scattering oracle");
                            check(near(data[base + 3][c], data[base + 4][c]) && near(data[base + 3][c], data[base + 5][c]), "Reciprocity/transport mismatch");
                            const double weight = data[base + 1][c] * mu / wi[3];
                            check(near(weight, data[base + 7][c]) && weight >= 0 && weight <= 1.00001, "Weight/energy mismatch");
                            sampled[c] += weight / 4096;
                            integrated[c] += data[base + 3][c] * quadratureMu * 2 * pi / 4096;
                        }
                    }
                    check(near(pdfIntegral, 1) && std::abs(meanCosine / 4096 - 2.0/3.0) < 0.0001, "Sampling/PDF normalization failed");
                    report << "case=" << caseIndex << " op=" << packet.control[0] << " muO=" << muOut;
                    for (size_t c = 0; c < 3; ++c) {
                        check(integrated[c] >= 0 && integrated[c] <= 1.00001 && std::abs(sampled[c] - integrated[c]) < 0.0002,
                            "White furnace energy or independent quadrature mismatch");
                        report << " " << sampled[c] << "/" << integrated[c];
                    }
                    report << '\n';
                }
            }
            const std::array labels{"SingleSlab", "DualSlabMix", "DualSlabLayer"};
            std::array<std::vector<Float4>, 3> renders;
            for (uint32_t index = 0; index < 3; ++index) {
                auto& pixels = renders[index]; pixels = dispatch(index, 1, true);
                std::ofstream raw(context.outputDirectory / (std::string(labels[index]) + ".rgba32f"), std::ios::binary);
                raw.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(Float4));
                check(bool(raw), "HDR artifact write failed");
                std::vector<uint8_t> rgba(pixels.size() * 4, 255);
                double sum = 0;
                for (size_t pixel = 0; pixel < pixels.size(); ++pixel) {
                    for (size_t c = 0; c < 3; ++c) {
                        check(pixels[pixel][c] >= 0, "Negative radiance"); sum += pixels[pixel][c];
                        const float linear = std::clamp(pixels[pixel][c], 0.0f, 1.0f);
                        const float srgb = linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1/2.4f) - 0.055f;
                        rgba[pixel * 4 + c] = static_cast<uint8_t>(std::lround(srgb * 255));
                    }
                }
                check(sum > 100, "Rendered image is empty");
                const auto saved = saveRgba8Png(context.outputDirectory / (std::string(labels[index]) + ".png"), rgba.data(), 256, 192, log);
                check(saved, log);
            }
            check(renders[0] != renders[1] && renders[1] != renders[2], "Canonical families/operators produced identical images");
            check(bool(report), "Energy report failed");
            check(validationErrors == 0, "Slab workload caused Vulkan validation errors");
            return RHITestResult::pass("23 IR-lowered materials x 3 views x 4096 samples: energy, oracle, reciprocity, sample/eval/pdf, payload ABI; 3 rendered families/operators");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialSlabTest);
} // namespace
} // namespace metallic::tests
