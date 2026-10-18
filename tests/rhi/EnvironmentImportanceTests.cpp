#include "RHITest.h"
#include "TestComputeProgram.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

void require(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

struct ImportanceFixture
{
    const char* name;
    uint32_t width;
    uint32_t height;
    uint32_t mipCount;
    uint32_t pattern;
    uint32_t baseMip = 0;
};

uint32_t importanceHash(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

uint32_t importanceSeed(uint32_t lane)
{
    constexpr uint32_t specialSeeds[] = {0u, 0x7f92df0au, 0u - 0xd1b54a35u,
        0xffffffffu, 0u - 0x9e3779b9u, 0u - 0x9e3779b9u - 1u,
        0u - 0x9e3779b9u - 4u, 0u - 0x9e3779b9u - 8u};
    return lane < std::size(specialSeeds) ? specialSeeds[lane] : lane * 747796405u + 2891336453u;
}

class EnvironmentImportanceTest final : public RHITest
{
public:
    EnvironmentImportanceTest()
    {
        name = "environment_importance_mip_dimensions";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        try {
            const char* requestedRngMode = std::getenv("METALLIC_ENVIRONMENT_IMPORTANCE_RNG_MODE");
            const std::string rngMode = requestedRngMode ? requestedRngMode : "serial";
            require(rngMode == "serial" || rngMode == "hash" || rngMode == "warp4", "Unknown environment importance RNG mode");
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            require(bool(bench::createTestDevice(context, {.applicationName = "Environment importance mip dimensions",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if (message.severity == ValidationSeverity::Error && hasFlag(message.type, ValidationCategory::Validation)) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}}).transform([&](auto value) { device = std::move(value); })), "Device creation failed");
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "EnvironmentImportanceProbe",
                .entryPointName = "environmentImportanceProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
            require(bool(shader), log);
            const ComputeResourceBindingDesc bindings[] = {
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput)},
                {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, environmentPdf), ComputeResourceBindingKind::SampledImage}};
            ComputeProgram program;
            require(bool(program.initialize(*device, {.spirv = shader->spirv, .bindings = bindings,
                .requiresRayQuery = false, .resourceParameterSize = sizeof(SceneResourceParameters)}, log)), log);
            constexpr ImportanceFixture fixtures[] = {
                {"square", 8, 8, 4, 0}, {"two-x-uniform", 2, 1, 2, 5}, {"two-y-uniform", 1, 2, 2, 5},
                {"CDF-boundary", 2, 2, 2, 7}, {"root-rounding", 2, 2, 2, 8},
                {"HDR", 16, 8, 5, 6}, {"single", 1, 1, 1, 0}, {"single-zero", 1, 1, 1, 1},
                {"rectangular", 8, 4, 4, 0}, {"thin-x", 8, 1, 4, 0}, {"thin-y", 1, 8, 4, 0},
                {"odd", 7, 5, 3, 0}, {"odd-thin", 5, 1, 3, 0}, {"partial-chain", 8, 4, 2, 0},
                {"zero", 8, 4, 4, 1}, {"sparse", 8, 4, 4, 2}, {"negative", 7, 5, 3, 3},
                {"zero-intermediate", 8, 4, 4, 4}, {"view-base-mip", 7, 5, 3, 0, 1},
                {"deep-thin-x", 1024, 1, 11, 0}, {"deep-thin-y", 1, 1024, 11, 6},
                {"deep-square", 128, 128, 8, 0}, {"deep-HDR", 128, 128, 8, 6},
                {"deep-sparse", 1024, 1, 11, 9}};
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream report(context.outputDirectory / "EnvironmentImportance.txt");
            require(bool(report), "Could not write environment importance evidence");
            report << "65536 deterministic seeds per fixture. Frozen original and both cached variants remain bit-exact.\n"
                << "rngMode=" << rngMode << ". Production may change selections and its RNG stream; the selected continuation contract is checked exactly. Legacy differences are informational.\n"
                << "Serial production must match the frozen raw-CDF/root-PDF branched traversal exactly: validity, texel, RNG and PDF bits.\n"
                << "Direct child selection checks cover 132 quartet/target cases including zero, exact CDF ties, adjacent floats, repeated CDFs, empty slots, HDR rounding and nextDown(sum).\n"
                << "PDF relative tolerance=2e-5 versus uploaded CPU leaf/root and hierarchy oracles. Zero probabilities require exact zero.\n"
                << "Successful multi-mip samples must match the evaluator PDF bit-for-bit at the selected texel, using the actual uploaded root.\n"
                << "Histogram uses CPU hierarchy probabilities; bins with expected count <16 are pooled. Each bin must be within 8 binomial standard deviations +8 counts (engineering regression threshold, not an RNG quality p-value).\n"
                << "Additional spatial 8x8 bins and x/y marginals prevent rare-texel pooling from hiding deep-chain bias. Special seeds cover zero, continuation hash zero, uint wrap, and sample hash zero at depths 0/1/4/8.\n"
                << "NPOT/partial chains sample only the subtree of top mip texel (0,0). Single-mip zero keeps legacy valid/PDF=1 behavior. Corrupt intermediate chains test failure/RNG only.\n";
            for (const auto& fixture : fixtures) {
                std::vector<float> texels;
                std::vector<size_t> offsets;
                for (uint32_t mip = 0; mip < fixture.mipCount; ++mip) {
                    const uint32_t width = std::max(fixture.width >> mip, 1u);
                    const uint32_t height = std::max(fixture.height >> mip, 1u);
                    offsets.push_back(texels.size());
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            float value = float(1 + (x * 3 + y * 5) % 11);
                            if (mip > 0) {
                                value = 0;
                                const uint32_t previousWidth = std::max(fixture.width >> (mip - 1), 1u);
                                const uint32_t previousHeight = std::max(fixture.height >> (mip - 1), 1u);
                                for (uint32_t dy = 0; dy < 2; ++dy) {
                                    for (uint32_t dx = 0; dx < 2; ++dx) {
                                        if (x * 2 + dx < previousWidth && y * 2 + dy < previousHeight) {
                                            value += std::max(texels[offsets[mip - 1] + (y * 2 + dy) * previousWidth + x * 2 + dx], 0.0f);
                                        }
                                    }
                                }
                                value *= 0.25f;
                            } else if (fixture.pattern == 9) {
                                value = x % 17u == 0u ? std::ldexp(float(1u + x % 3u), int(x % 13u) - 6) : 0.0f;
                            } else if (fixture.pattern == 8) {
                                // CPU mip construction sums 00,10,01,11; the sampler visits
                                // 00,01,10,11. Recomputing the top sum can round differently
                                // from loading the stored root, despite an equivalent real sum.
                                constexpr float roundingWeights[] = {16777216.0f, 1.0f, 2.0f, 0.0f};
                                value = roundingWeights[y * 2 + x];
                            } else if (fixture.pattern == 7) {
                                constexpr float boundaryWeights[] = {909.0f, 255.0f, 221.0f, 185.0f};
                                value = boundaryWeights[y * 2 + x];
                            } else if (fixture.pattern == 5) {
                                value = 1.0f;
                            } else if (fixture.pattern == 6) {
                                value = std::ldexp(float(1 + (x + y) % 3), int((x * 7 + y * 3) % 25) - 12);
                            } else if (fixture.pattern == 2) {
                                value = x == 0 && y == 0 ? 8.0f : 0.0f;
                            } else if (fixture.pattern == 3 && (x + y) % 3 == 0) {
                                value = -value;
                            }
                            if (fixture.pattern == 1) { value = 0; }
                            texels.push_back(value);
                        }
                    }
                }
                // Fail after a successful top-level selection to check partial RNG advancement.
                if (fixture.pattern == 4) {
                    std::fill(texels.begin() + offsets[1], texels.begin() + offsets[2], 0.0f);
                }
                using UInt4 = std::array<uint32_t, 4>;
                constexpr uint32_t kLaneCount = 65536;
                constexpr uint32_t kRowsPerLane = 12;
                std::unique_ptr<Buffer> upload, output;
                auto allocate = [&](size_t bytes, MemoryLocation location, BufferUsageBits usage, auto& target) {
                    require(bool(device->createBuffer({.size = bytes, .usage = usage, .memoryLocation = location})
                        .transform([&](auto value) { target = std::move(value); })), "Buffer creation failed");
                };
                allocate(texels.size() * sizeof(float), MemoryLocation::HostUpload, BufferUsageBits::TransferSource, upload);
                allocate(kLaneCount * kRowsPerLane * sizeof(UInt4), MemoryLocation::HostReadback, BufferUsageBits::Storage, output);
                void* mapped = upload->map();
                require(mapped != nullptr, "Upload mapping failed");
                std::memcpy(mapped, texels.data(), texels.size() * sizeof(float));
                upload->flush();
                upload->unmap();
                std::unique_ptr<Texture> texture;
                std::unique_ptr<TextureView> view;
                require(bool(device->createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                    .format = Format::R32Sfloat, .width = fixture.width << fixture.baseMip,
                    .height = fixture.height << fixture.baseMip, .mipCount = fixture.mipCount + fixture.baseMip})
                    .transform([&](auto value) { texture = std::move(value); })), "Texture creation failed");
                require(bool(device->createTextureView(*texture, {.format = Format::R32Sfloat,
                    .range = {.baseMip = fixture.baseMip, .mipCount = fixture.mipCount}})
                    .transform([&](auto value) { view = std::move(value); })), "Texture view creation failed");
                bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                require(bool(gpu.initialize(*device)), "GPU commands failed");
                const TextureBarrierDesc before{.texture = texture.get(), .oldLayout = TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .range = {.baseMip = fixture.baseMip, .mipCount = fixture.mipCount}};
                require(bool(gpu.commands->synchronize({.textures = {&before, 1}})), "Upload barrier failed");
                for (uint32_t mip = 0; mip < fixture.mipCount; ++mip) {
                    require(bool(upload->slice({offsets[mip] * sizeof(float)}).and_then([&](const auto& slice) {
                        return gpu.commands->copyBufferToTexture({.texture = texture.get(), .buffer = slice,
                            .width = std::max(fixture.width >> mip, 1u), .height = std::max(fixture.height >> mip, 1u),
                            .mipLevel = mip + fixture.baseMip});
                    })), "Mip upload failed");
                }
                const TextureBarrierDesc after{.texture = texture.get(), .oldLayout = TextureLayout::TransferDestination,
                    .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead},
                    .range = {.baseMip = fixture.baseMip, .mipCount = fixture.mipCount}};
                require(bool(gpu.commands->synchronize({.textures = {&after, 1}})), "Sampling barrier failed");
                TextureView* views[] = {view.get()};
                const ComputeDispatchBinding dispatch[] = {
                    {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput), .buffer = output.get()},
                    {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, environmentPdf), .textureViews = views}};
                require(bool(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = dispatch, .groupCountX = kLaneCount / 64})), "Dispatch failed");
                require(bool(gpu.submitAndWait()), "GPU execution failed");
                output->invalidate();
                const auto* data = static_cast<const UInt4*>(output->map());
                require(data != nullptr, "Readback failed");
                std::vector<UInt4> actual(kLaneCount * kRowsPerLane);
                std::memcpy(actual.data(), data, actual.size() * sizeof(UInt4));
                output->unmap();
                uint32_t successful = 0, outOfBounds = 0, validMismatches = 0, texelMismatches = 0;
                uint32_t probabilityBitMismatches = 0, evaluatorBitMismatches = 0, rngMismatches = 0, legacyRngDifferences = 0;
                uint32_t sampleEvaluatorBitChecks = 0;
                uint32_t rawBranchedBitChecks = 0, childBoundaryChecks = 0;
                std::array<bool, 132> childBoundaryCases{};
                double maxPdfRelativeError = 0, maxHierarchyRelativeError = 0, maxOriginalRelativeError = 0;
                const uint32_t representedWidth = std::min(fixture.width, 1u << (fixture.mipCount - 1));
                const uint32_t representedHeight = std::min(fixture.height, 1u << (fixture.mipCount - 1));
                const double total = std::ldexp(double(std::max(texels[offsets.back()], 0.0f)), 2 * int(fixture.mipCount - 1));
                auto leafProbability = [&](uint32_t x, uint32_t y) {
                    return x < representedWidth && y < representedHeight && total > 0
                        ? double(std::max(texels[y * fixture.width + x], 0.0f)) / total : 0.0;
                };
                // Independent product of conditional probabilities from the actual uploaded
                // hierarchy, evaluated in double precision (including thin-mip missing children).
                auto hierarchyProbability = [&](uint32_t x, uint32_t y) {
                    if (x >= representedWidth || y >= representedHeight) { return 0.0; }
                    double probability = 1;
                    for (int mip = int(fixture.mipCount) - 2; mip >= 0; --mip) {
                        const uint32_t width = std::max(fixture.width >> mip, 1u);
                        const uint32_t height = std::max(fixture.height >> mip, 1u);
                        const uint32_t childX = x >> mip, childY = y >> mip;
                        const uint32_t parentX = childX & ~1u, parentY = childY & ~1u;
                        double sum = 0;
                        for (uint32_t dy = 0; dy < 2; ++dy) {
                            for (uint32_t dx = 0; dx < 2; ++dx) {
                                if (parentX + dx < width && parentY + dy < height) {
                                    sum += std::max(texels[offsets[mip] + (parentY + dy) * width + parentX + dx], 0.0f);
                                }
                            }
                        }
                        if (sum <= 0) { return 0.0; }
                        probability *= double(std::max(texels[offsets[mip] + childY * width + childX], 0.0f)) / sum;
                    }
                    return probability;
                };
                auto checkProbability = [&](float observed, double expected, double& maximum, const std::string& where) {
                    require(std::isfinite(observed) && observed >= 0, where + " nonfinite/negative PDF");
                    if (expected == 0) { require(observed == 0, where + " expected zero PDF"); return; }
                    const double error = std::abs(double(observed) - expected) / expected;
                    maximum = std::max(maximum, error);
                    require(error <= 2e-5, where + " relative PDF error=" + std::to_string(error));
                };
                std::vector<uint32_t> histogram(fixture.width * fixture.height, 0);
                for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
                    const auto* row = &actual[lane * kRowsPerLane];
                    const std::string where = std::string(fixture.name) + " lane=" + std::to_string(lane);
                    if (rngMode == "serial") {
                        require(row[0] == row[8], where + " frozen raw-CDF branched validity/texel/RNG mismatch");
                        require(row[2][0] == row[9][0], where + " frozen raw-CDF/root probability bits mismatch");
                        ++rawBranchedBitChecks;
                    }
                    require(row[9][1] == row[9][2] && row[9][1] < 4u, where + " child CDF boundary index mismatch");
                    require(row[10][0] == row[10][1], where + " child CDF boundary weight bits mismatch");
                    require(row[10][0] == row[11][row[9][1]], where + " child index/selected weight mismatch");
                    require(row[9][3] == lane % childBoundaryCases.size(), where + " child boundary case ID mismatch");
                    childBoundaryCases[row[9][3]] = true;
                    ++childBoundaryChecks;
                    require(row[5] == row[1], where + " frozen cached-entry-derived sample/texel/RNG mismatch");
                    require(row[6] == row[1], where + " frozen cached-per-level sample/texel/RNG mismatch");
                    require(row[7][0] == row[2][1] && row[7][1] == row[2][1], where + " frozen probability bits mismatch");
                    validMismatches += row[0][0] != row[1][0];
                    texelMismatches += row[0][1] != row[1][1] || row[0][2] != row[1][2];
                    probabilityBitMismatches += row[2][0] != row[2][1];
                    evaluatorBitMismatches += row[2][2] != row[2][3];
                    const uint32_t initialSeed = importanceSeed(lane);
                    uint32_t expectedSeed = initialSeed;
                    if (rngMode == "serial") { expectedSeed = row[1][3]; }
                    else if (fixture.mipCount > 1) {
                        expectedSeed = importanceHash(initialSeed + 0xd1b54a35u);
                        if (expectedSeed == 0u) { expectedSeed = 0xd1b54a35u; }
                    }
                    legacyRngDifferences += row[0][3] != row[1][3];
                    rngMismatches += row[0][3] != expectedSeed;
                    require(row[0][3] == expectedSeed, where + " production RNG continuation contract mismatch");
                    require(row[3][0] == row[3][1] && row[3][0] == row[3][2], where + " weight bits mismatch");
                    require(row[7][2] == row[3][2], where + " cached texture weight bits mismatch");
                    const uint32_t mip = row[3][3];
                    const uint32_t width = std::max(fixture.width >> mip, 1u), height = std::max(fixture.height >> mip, 1u);
                    const bool inBounds = row[4][0] < width && row[4][1] < height;
                    const float expectedWeight = inBounds ? std::max(texels[offsets[mip] + row[4][1] * width + row[4][0]], 0.0f) : 0.0f;
                    require(row[3][0] == std::bit_cast<uint32_t>(expectedWeight), where + " uploaded weight/OOB mismatch");
                    checkProbability(std::bit_cast<float>(row[2][2]), leafProbability(row[4][2], row[4][3]),
                        maxPdfRelativeError, where + " evaluator");
                    const float observedPdf = std::bit_cast<float>(row[2][0]);
                    if (row[0][0]) {
                        require(row[0][1] < representedWidth && row[0][2] < representedHeight, where + " selected outside represented subtree");
                        ++histogram[row[0][2] * fixture.width + row[0][1]];
                        if (fixture.mipCount > 1) {
                            checkProbability(observedPdf, leafProbability(row[0][1], row[0][2]), maxPdfRelativeError, where + " sampled leaf");
                            checkProbability(observedPdf, hierarchyProbability(row[0][1], row[0][2]), maxHierarchyRelativeError, where + " sampled hierarchy");
                            checkProbability(std::bit_cast<float>(row[7][3]), double(observedPdf), maxPdfRelativeError, where + " sample/evaluator agreement");
                            require(row[2][0] == row[7][3], where + " sample/evaluator PDF bits mismatch");
                            ++sampleEvaluatorBitChecks;
                        } else { require(observedPdf == 1, where + " single-mip legacy PDF changed"); }
                        if (row[0][1] == row[1][1] && row[0][2] == row[1][2]) {
                            checkProbability(observedPdf, std::bit_cast<float>(row[2][1]), maxOriginalRelativeError, where + " same selected texel legacy PDF");
                        }
                    } else { require(observedPdf == 0, where + " failed sample PDF"); }
                    successful += row[0][0];
                    outOfBounds += !inBounds;
                }
                const bool shouldSucceed = fixture.mipCount == 1 || (fixture.pattern != 1 && fixture.pattern != 4);
                require(successful == (shouldSucceed ? kLaneCount : 0), std::string(fixture.name) + " unexpected validity count");
                uint32_t distributionBins = 0;
                double maxBinStandardDeviations = 0;
                if (shouldSucceed) {
                    double smallProbability = 0;
                    uint32_t smallObserved = 0;
                    // Each multinomial marginal has variance N*p*(1-p). Eight
                    // sigma is a generous multi-fixture regression threshold;
                    // +8 protects rare bins, quantization and dedicated seeds.
                    // Deterministic seeds do not establish RNG statistical quality.
                    auto checkBin = [&](uint32_t observed, double probability) {
                        const double expected = kLaneCount * probability;
                        const double sigma = std::sqrt(std::max(expected * (1 - probability), 0.0));
                        const double deviation = std::abs(double(observed) - expected);
                        maxBinStandardDeviations = std::max(maxBinStandardDeviations, deviation / std::max(sigma, 1.0));
                        require(deviation <= 8 * sigma + 8, std::string(fixture.name) + " histogram outside 8 sigma +8 counts");
                        ++distributionBins;
                    };
                    for (uint32_t y = 0; y < fixture.height; ++y) {
                        for (uint32_t x = 0; x < fixture.width; ++x) {
                            const double probability = hierarchyProbability(x, y);
                            const uint32_t observed = histogram[y * fixture.width + x];
                            if (probability == 0) { require(observed == 0, std::string(fixture.name) + " sampled zero-probability texel"); }
                            else if (kLaneCount * probability < 16) { smallProbability += probability; smallObserved += observed; }
                            else { checkBin(observed, probability); }
                        }
                    }
                    if (smallProbability > 0) { checkBin(smallObserved, smallProbability); }
                    std::array<uint32_t, 64> coarseObserved{};
                    std::array<double, 64> coarseProbability{};
                    for (uint32_t y = 0; y < fixture.height; ++y) {
                        for (uint32_t x = 0; x < fixture.width; ++x) {
                            const uint32_t bin = std::min(y * 8u / fixture.height, 7u) * 8u + std::min(x * 8u / fixture.width, 7u);
                            coarseObserved[bin] += histogram[y * fixture.width + x];
                            coarseProbability[bin] += hierarchyProbability(x, y);
                        }
                    }
                    for (uint32_t index = 0; index < 64; ++index) { checkBin(coarseObserved[index], coarseProbability[index]); }
                    for (uint32_t index = 0; index < 8; ++index) {
                        uint32_t xObserved = 0, yObserved = 0;
                        double xProbability = 0, yProbability = 0;
                        for (uint32_t other = 0; other < 8; ++other) {
                            xObserved += coarseObserved[other * 8 + index];
                            xProbability += coarseProbability[other * 8 + index];
                            yObserved += coarseObserved[index * 8 + other];
                            yProbability += coarseProbability[index * 8 + other];
                        }
                        checkBin(xObserved, xProbability);
                        checkBin(yObserved, yProbability);
                    }
                }
                require(outOfBounds > 0, "Fixture did not exercise out-of-bounds weights");
                require(std::all_of(childBoundaryCases.begin(), childBoundaryCases.end(), [](bool covered) { return covered; }),
                    "Missing direct child selection boundary coverage");
                report << fixture.name << " dimensions=" << fixture.width << 'x' << fixture.height
                    << " mips=" << fixture.mipCount << " baseMip=" << fixture.baseMip << " seeds=" << kLaneCount
                    << " success=" << successful << " outOfBounds=" << outOfBounds
                    << " validityMismatches=" << validMismatches << " texelMismatches=" << texelMismatches
                    << " probabilityBitMismatches=" << probabilityBitMismatches << " evaluatorBitMismatches=" << evaluatorBitMismatches
                    << " sampleEvaluatorBitChecks=" << sampleEvaluatorBitChecks
                    << " rawBranchedBitChecks=" << rawBranchedBitChecks << " childBoundaryChecks=" << childBoundaryChecks
                    << " childBoundaryUniqueCases=" << childBoundaryCases.size()
                    << " rngMismatches=" << rngMismatches << " legacyRngDifferences=" << legacyRngDifferences
                    << " maxPdfRelativeError=" << maxPdfRelativeError
                    << " maxHierarchyRelativeError=" << maxHierarchyRelativeError << " maxOriginalRelativeError=" << maxOriginalRelativeError
                    << " distributionBins=" << distributionBins << " maxBinSigma=" << maxBinStandardDeviations << '\n';
                report.flush();
            }
            require(validationErrors == 0, "Vulkan validation errors");
            return RHITestResult::pass(std::to_string(std::size(fixtures) * 65536u)
                + " GPU cases: frozen bit-exact variants, raw-CDF branched exact comparison, child CDF boundary selection, " + rngMode + " RNG continuation contract, exact sample/evaluator PDF, CPU hierarchy distribution and spatial marginals, deep/thin/NPOT/partial/HDR chains");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(EnvironmentImportanceTest);
} // namespace
} // namespace metallic::tests
