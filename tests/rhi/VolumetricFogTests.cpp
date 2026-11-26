#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Environment/VolumetricFogParameters.h"
#include "Runtime/Render/Environment/LocalFogVolumePacking.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

constexpr uint32_t kWidth = 128, kHeight = 96, kTile = 16, kX = 8, kY = 6, kZ = 64;
constexpr double kNearCm = 5.0, kMaxCm = 6000.0, kDistribution = 32.0, kExtinction = 0.45 / 1000.0 * 0.5;
constexpr double kFirstCm = kNearCm + 9.5;
const double kO = (kMaxCm - kFirstCm * std::exp2(kZ / kDistribution)) / (kMaxCm - kFirstCm);
const double kB = (1.0 - kO) / kFirstCm;

template<typename T>
T checked(Result<T> result, std::string_view label)
{
    if (!result) { throw std::runtime_error(std::string(label) + ": " + resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

struct FogTestImage {
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    ResourceState state = ResourceState::Undefined;

    void transition(CommandBuffer& commands, ResourceState next)
    {
        const TextureBarrierDesc barrier{.texture = texture.get(),
            .oldLayout = textureLayoutForResourceState(state), .newLayout = textureLayoutForResourceState(next),
            .before = resourceSyncScope(state, PipelineStageBits::ComputeShader),
            .after = resourceSyncScope(next, PipelineStageBits::ComputeShader)};
        checked(commands.synchronize({.textures = {&barrier, 1}}), "transition fog test image");
        state = next;
    }
};

// Independent homogeneous Beer solution from the front boundary to each slice
// center. The oblique camera ray is longer than its axial view depth.
double transmittanceAt(int x, int y, int z)
{
    const double ndcX = 2.0 * (x + 0.5) / kX - 1.0;
    const double ndcY = (1.0 - 2.0 * (y + 0.5) / kY) * kHeight / kWidth;
    const double sliceCm = (std::exp2((z + 0.5) / kDistribution) - kO) / kB;
    return std::exp(-kExtinction * (sliceCm - kFirstCm) * std::sqrt(1 + ndcX * ndcX + ndcY * ndcY));
}

double compositeReference(int x, int y, double planeCm)
{
    const double cx = std::min(x + 0.5, double(kWidth - (kTile / 2 + 1))) / kTile - 0.5;
    const double cy = std::min(y + 0.5, double(kHeight - (kTile / 2 + 1))) / kTile - 0.5;
    const double cz = std::log2(planeCm * kB + kO) * kDistribution - 0.5;
    const int x0 = int(std::floor(cx)), y0 = int(std::floor(cy)), z0 = int(std::floor(cz));
    double value = 0;
    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                value += transmittanceAt(std::clamp(x0 + dx, 0, int(kX - 1)),
                    std::clamp(y0 + dy, 0, int(kY - 1)), std::clamp(z0 + dz, 0, int(kZ - 1))) *
                    (dx ? cx - x0 : 1 - (cx - x0)) * (dy ? cy - y0 : 1 - (cy - y0)) *
                    (dz ? cz - z0 : 1 - (cz - z0));
            }
        }
    }
    return value;
}

float decodeHalf(uint16_t value)
{
    const int exponent = (value >> 10) & 31;
    const int mantissa = value & 1023;
    if (exponent == 31) { return std::numeric_limits<float>::infinity(); }
    const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -24) :
        std::ldexp(float(1024 + mantissa), exponent - 25);
    return (value & 0x8000u) ? -magnitude : magnitude;
}

class VolumetricFogTransportTest final : public RHITest
{
public:
    VolumetricFogTransportTest()
    {
        name = "volumetric_fog_homogeneous_transport";
        type = RHITestType::Rendering;
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.volumetric_fog.transport"}, bench::Layer::Core,
            "binding", "binding", {"transmittance.json"});
    }

    RHITestResult run(RHITestContext& context) override
    {
        bench::ValidationRecorder validation;
        try {
            auto created = bench::createTestDevice(context, {.applicationName = "Volumetric fog homogeneous transport",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless Vulkan compute"); }
            auto device = checked(std::move(created), "create device");
            auto registry = checked(ResourceRegistry::forDevice(*device), "resource registry");
            std::array<ComputeKernel, 4> kernels;
            const std::array entries{"materialSetup", "lightScattering", "finalIntegration", "compositeFog"};
            for (size_t i = 0; i < entries.size(); ++i) {
                std::string log;
                auto compiled = ShaderRegistry::instance().getComputeKernel(*device, {
                    .moduleName = "Features/Environment/VolumetricFog", .entryPointName = entries[i],
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, {.parameters = parameterAbi<VolumetricFogParams>(kVolumetricFogABI),
                    .debugName = entries[i]}, kernels[i], log);
                if (!compiled) { return RHITestResult::fail(std::string(entries[i]) + ": " + log + " " + resultToString(compiled)); }
            }
            const auto makeImage = [&](Format format, bool volume, TextureUsageBits usage) {
                FogTestImage image;
                image.texture = checked(device->createTexture({.type = volume ? TextureType::Texture3D : TextureType::Texture2D,
                    .usage = usage, .format = format, .width = volume ? kX : kWidth, .height = volume ? kY : kHeight,
                    .depth = volume ? kZ : 1}), "create fog image");
                image.view = checked(device->createTextureView(*image.texture, {.format = format}), "create fog view");
                return image;
            };
            const auto volumeUsage = TextureUsageBits::Sampled | TextureUsageBits::Storage;
            auto material = makeImage(Format::RGBA16Sfloat, true, volumeUsage);
            auto scattering = makeImage(Format::RGBA16Sfloat, true, volumeUsage | TextureUsageBits::TransferSource);
            auto integrated = makeImage(Format::RGBA16Sfloat, true, volumeUsage | TextureUsageBits::TransferSource);
            auto depth = makeImage(Format::R32Sfloat, false, TextureUsageBits::Sampled | TextureUsageBits::TransferDestination);
            auto source = makeImage(Format::RGBA32Sfloat, false, TextureUsageBits::Sampled | TextureUsageBits::TransferDestination);
            auto output = makeImage(Format::RGBA32Sfloat, false, TextureUsageBits::Storage | TextureUsageBits::TransferSource);
            const auto giUsage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination;
            auto giAmbient = makeImage(Format::RGBA16Sfloat, true, giUsage);
            auto giDirectional = makeImage(Format::RGBA16Sfloat, true, giUsage);
            constexpr std::array<float, 3> ambient{2.0f, 1.0f, 0.5f}, directional{1.25f, -0.75f, 0.5f};
            constexpr std::array<float, 2> phaseG{-0.7f, 0.7f};
            constexpr uint64_t planeBytes = kWidth * kHeight * 4 * sizeof(float);
            constexpr uint64_t volumeBytes = kX * kY * kZ * 4 * sizeof(uint16_t);
            constexpr uint64_t sliceBytes = volumeBytes / kZ;
            const std::array<float, 3> planes{10.0f, 20.0f, 40.0f};
            const uint64_t giOffset = planes.size() * planeBytes + volumeBytes;
            auto readback = checked(device->createBuffer({.size = giOffset + phaseG.size() * sliceBytes,
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}), "readback");
            ParameterWriter writer(*device, *registry);
            VolumetricFogParams p{};
            p.current.center[2] = -1.0f;
            p.current.upProjection[1] = 1.0f;
            p.current.viewport[0] = float(kWidth) / kHeight;
            p.current.viewport[1] = float(kWidth);
            p.current.viewport[2] = float(kHeight);
            p.current.viewport[3] = 2.0f * std::atan(float(kHeight) / kWidth); // 90 degree horizontal FOV
            p.current.clipOrtho[0] = float(kNearCm / 100.0);
            p.current.clipOrtho[1] = 1000.0f;
            p.previous = p.current;
            p.gridZ = {float(kB), float(kO), float(kDistribution), 0};
            p.grid = {kX, kY, kZ, kTile};
            p.fogLayer0[0] = 0.45f / 1000.0f;
            p.albedoExtinction[3] = 1.0f;
            p.lightDirectionPhase[2] = -1.0f;
            p.distances[1] = 1.0e8f;
            p.materialOut = writer.storageImageHandle(material.view.get());
            p.scatteringOut = writer.storageImageHandle(scattering.view.get());
            p.integratedOut = writer.storageImageHandle(integrated.view.get());
            p.colorOut = writer.storageImageHandle(output.view.get());
            p.material = writer.sampledImageHandle(material.view.get());
            p.scattering = writer.sampledImageHandle(scattering.view.get());
            p.integrated = writer.sampledImageHandle(integrated.view.get());
            p.depth = writer.sampledImageHandle(depth.view.get());
            p.sceneColor = writer.sampledImageHandle(source.view.get());
            p.sampler = writer.samplerHandle({});
            auto unlit = checked(writer.encode(p, kVolumetricFogABI), "encode unlit parameters");
            p.albedoExtinction = {1.0f, 0.5f, 0.0f, 1.0f};
            p.lightColorIntensity = {4.0f * std::numbers::pi_v<float>, 4.0f * std::numbers::pi_v<float>,
                4.0f * std::numbers::pi_v<float>, 1.0f};
            auto lit = checked(writer.encode(p, kVolumetricFogABI), "encode lit parameters");
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            checked(gpu.initialize(*device), "begin");
            auto& commands = *gpu.commands;
            const auto recordVolume = [&](const EncodedParameters& params) {
                material.transition(commands, ResourceState::General);
                checked(kernels[0].dispatch(commands, params, 2, 2, 16), "material");
                material.transition(commands, ResourceState::ShaderRead);
                scattering.transition(commands, ResourceState::General);
                checked(kernels[1].dispatch(commands, params, 2, 2, 16), "scattering");
                scattering.transition(commands, ResourceState::ShaderRead);
                integrated.transition(commands, ResourceState::General);
                checked(kernels[2].dispatch(commands, params, 1, 1, 1), "integration");
            };
            recordVolume(unlit);
            integrated.transition(commands, ResourceState::ShaderRead);
            source.transition(commands, ResourceState::TransferDestination);
            checked(commands.clearColorTexture(*source.texture, TextureLayout::TransferDestination, {1, 1, 1, 1}), "clear source");
            source.transition(commands, ResourceState::ShaderRead);
            for (size_t i = 0; i < planes.size(); ++i) {
                const float nearZ = p.current.clipOrtho[0], farZ = p.current.clipOrtho[1];
                const float deviceZ = nearZ * (farZ - planes[i]) / (planes[i] * (farZ - nearZ));
                depth.transition(commands, ResourceState::TransferDestination);
                checked(commands.clearColorTexture(*depth.texture, TextureLayout::TransferDestination, {deviceZ, 0, 0, 0}), "clear plane depth");
                depth.transition(commands, ResourceState::ShaderRead);
                output.transition(commands, ResourceState::General);
                checked(kernels[3].dispatch(commands, unlit, kWidth / 8, kHeight / 8, 1), "composite");
                output.transition(commands, ResourceState::TransferSource);
                checked(commands.copyTextureToBuffer({.texture = output.texture.get(),
                    .buffer = checked(readback->slice({i * planeBytes, planeBytes}), "plane range"), .width = kWidth,
                    .height = kHeight}), "read plane");
            }
            recordVolume(lit);
            integrated.transition(commands, ResourceState::TransferSource);
            checked(commands.copyTextureToBuffer({.texture = integrated.texture.get(),
                .buffer = checked(readback->slice({planes.size() * planeBytes, volumeBytes}), "volume range"),
                .width = kX, .height = kY, .depth = kZ}), "read integrated volume");
            const auto clearSH = [&](FogTestImage& image, const std::array<float, 3>& rgb) {
                image.transition(commands, ResourceState::TransferDestination);
                checked(commands.clearColorTexture(*image.texture, TextureLayout::TransferDestination,
                    {rgb[0], rgb[1], rgb[2], 0}), "initialize Lumen coefficients");
                image.transition(commands, ResourceState::ShaderRead);
            };
            clearSH(giAmbient, ambient);
            clearSH(giDirectional, directional);
            p.lightColorIntensity = {};
            p.indirectLight.ambient = writer.sampledImageHandle(giAmbient.view.get());
            p.indirectLight.directional = writer.sampledImageHandle(giDirectional.view.get());
            p.indirectLight.gridZ = {0.01f, 1.0f, 4.0f, float(kZ)};
            p.indirectLight.textureDepth = float(kZ);
            for (size_t i = 0; i < phaseG.size(); ++i) {
                p.lightDirectionPhase[3] = phaseG[i];
                const auto params = checked(writer.encode(p, kVolumetricFogABI), "encode directional fog GI");
                scattering.transition(commands, ResourceState::General);
                checked(kernels[1].dispatch(commands, params, 2, 2, 1), "fog GI scattering");
                scattering.transition(commands, ResourceState::TransferSource);
                checked(commands.copyTextureToBuffer({.texture = scattering.texture.get(),
                    .buffer = checked(readback->slice({giOffset + i * sliceBytes, sliceBytes}), "GI slice range"),
                    .width = kX, .height = kY, .depth = 1}), "read directional scattering slice");
            }
            const BufferBarrierDesc host{.buffer = readback.get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}};
            checked(commands.synchronize({.buffers = {&host, 1}}), "make readback visible");
            checked(gpu.submitAndWait(), "submit");
            readback->invalidate();
            const auto* pixels = static_cast<const float*>(readback->map());
            if (!pixels) { return RHITestResult::fail("Could not map fog readback"); }
            bench::Json observations = bench::Json::array();
            double maxError = 0;
            const auto compare = [&](double actual, double expected) {
                maxError = std::isfinite(actual) ? std::max(maxError, std::abs(actual - expected)) :
                    std::numeric_limits<double>::infinity();
            };
            constexpr std::array<std::array<int, 2>, 3> samples{{{64, 48}, {30, 24}, {6, 8}}};
            for (size_t i = 0; i < planes.size(); ++i) {
                for (const auto& [x, y] : samples) {
                    const float* pixel = pixels + 4 * (i * kWidth * kHeight + y * kWidth + x);
                    const double expected = compositeReference(x, y, planes[i] * 100.0);
                    for (int c = 0; c < 3; ++c) { compare(pixel[c], expected); }
                    compare(pixel[3], 1.0);
                    observations.push_back({{"planeMeters", planes[i]}, {"x", x}, {"y", y},
                        {"transmittance", pixel[0]}, {"reference", expected}});
                }
            }
            const auto* volume = reinterpret_cast<const uint16_t*>(pixels + planes.size() * kWidth * kHeight * 4);
            for (int z = 0; z < int(kZ); ++z) {
                for (int y = 0; y < int(kY); ++y) {
                    for (int x = 0; x < int(kX); ++x) {
                        const uint16_t* voxel = volume + 4 * ((z * kY + y) * kX + x);
                        const double t = transmittanceAt(x, y, z);
                        compare(decodeHalf(voxel[0]), 1.0 - t);
                        compare(decodeHalf(voxel[1]), 0.5 * (1.0 - t));
                        compare(decodeHalf(voxel[2]), 0.0);
                        compare(decodeHalf(voxel[3]), t);
                    }
                }
            }
            // Camera looks down Metallic -Z (UE +X). The encoded first-order
            // coefficients multiply UE Y, Z, X, equal here to ray.x, ray.y, -ray.z.
            double maxPhaseError = 0;
            const double luminance = color::luminance(ambient);
            const auto* gi = volume + volumeBytes / sizeof(uint16_t);
            for (size_t i = 0; i < phaseG.size(); ++i) {
                for (uint32_t y = 0; y < kY; ++y) {
                    for (uint32_t x = 0; x < kX; ++x) {
                        const double dx = 2.0 * (x + 0.5) / kX - 1.0;
                        const double dy = (1.0 - 2.0 * (y + 0.5) / kY) * kHeight / kWidth;
                        const double moment = (directional[0] * dx + directional[1] * dy + directional[2]) /
                            std::sqrt(1.0 + dx * dx + dy * dy);
                        const auto* voxel = gi + 4 * ((i * kY + y) * kX + x);
                        for (size_t c = 0; c < ambient.size(); ++c) {
                            const double expected = p.albedoExtinction[c] * std::max(0.0,
                                ambient[c] + phaseG[i] * moment * ambient[c] / (luminance + 0.00001));
                            const double actual = decodeHalf(voxel[c]) / kExtinction;
                            maxPhaseError = std::max(maxPhaseError, std::abs(actual - expected));
                        }
                    }
                }
            }
            readback->unmap();
            if (maxPhaseError > 0.003) {
                return RHITestResult::fail("Fog GI rotated HG SH mismatch, max error " + std::to_string(maxPhaseError));
            }
            std::ofstream(context.outputDirectory / "transmittance.json") << observations.dump(2);
            if (context.evidence) { context.evidence->json("transmittance.json", observations); }
            if (validation.failed() || maxError > 0.0015) {
                return RHITestResult::fail("Fog volume/scene-color transport max absolute error " + std::to_string(maxError) +
                    ", validation errors " + std::to_string(validation.failed()));
            }
            return RHITestResult::pass("Three depth planes and 3,072 lit froxels match homogeneous transport, max error " +
                std::to_string(maxError) + "; 96 signed-phase SH samples, max error " + std::to_string(maxPhaseError));
        } catch (const std::exception& e) {
            return RHITestResult::fail(e.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VolumetricFogTransportTest);

class LocalFogVolumePackingTest final : public RHITest
{
public:
    LocalFogVolumePackingTest()
    {
        name = "local_fog_volume_view_packing";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext&) override
    {
        std::vector<environment::LocalFogVolume> volumes(37);
        for (size_t i = 0; i < volumes.size(); ++i) { volumes[i].centerMeters[0] = float(i); }
        volumes[0].sortPriority = -127;
        volumes[34].sortPriority = 127;
        volumes[35].enabled = false;
        volumes[36].radialExtinction = volumes[36].heightExtinction = 0.0f;
        volumes[34].rotationDegrees = {90, 90, 90};
        auto packed = packLocalFogVolumes(volumes, {});
        if (packed.size() != 32) { return RHITestResult::fail("Expected 32 local spheres"); }
        for (size_t i = 0; i < 32; ++i) {
            const float x = std::bit_cast<float>(packed[i][0][0]) * 0.01f;
            if (x != (i == 31 ? 34 : 31 - i)) { return RHITestResult::fail("Incorrect priority/distance cap"); }
        }
        // Applying X,Y,Z quarter turns maps local +X to world -Z, +Y to
        // +Y and +Z to +X. In the packed Z-up basis these rows are (0,-1,0)
        // and (1,0,0), whose cross gives the shader's third row.
        const auto& rot = packed.back()[1];
        const std::array<double, 6> rows{decodeHalf(uint16_t(rot[0])), decodeHalf(uint16_t(rot[0] >> 16)),
            decodeHalf(uint16_t(rot[1])), decodeHalf(uint16_t(rot[1] >> 16)),
            decodeHalf(uint16_t(rot[2])), decodeHalf(uint16_t(rot[2] >> 16))};
        constexpr std::array<double, 6> expected{0, -1, 0, 1, 0, 0};
        for (size_t i = 0; i < rows.size(); ++i) {
            if (std::abs(rows[i] - expected[i]) > 0.00001) { return RHITestResult::fail("Incorrect packed XYZ rotation"); }
        }
        return RHITestResult::pass("Verified per-view sorting/cap, empty volumes and inverse basis packing");
    }
};

METALLIC_REGISTER_RHI_TEST(LocalFogVolumePackingTest);

class VolumetricFogToggleTest final : public RHITest
{
public:
    VolumetricFogToggleTest()
    {
        name = "volumetric_fog_graph_toggle";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        try {
            const auto& caps = context.device.capabilities();
            if (!caps.meshShader || !caps.taskShader || !caps.geometryShader || !caps.bindlessDescriptorHeap) {
                return RHITestResult::skip("The Cornell visibility-buffer graph requires --rhi-realtime");
            }
            scene::SceneDocument document;
            const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Lumen/CornellBox.metallic_scene.json";
            if (!document.load(path)) { return RHITestResult::fail("Load Cornell box: " + document.documentWarning() + document.lastLoadResult().error); }
            RenderWorld world;
            world.setScene(&document);
            auto environment = document.worldEnvironment();
            environment.volumetricFog.enabled = true;
            environment.volumetricFog.density = 0.45f;
            environment.volumetricFog.emissive = float3(1.0f, 0.25f, 0.0f);
            RenderGraph graph;
            graph.addNode("VisibilityBufferPass", "VBuffer", {{"initialLoad", true}, {"enableClusterRtx", false}});
            graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"outputLinear", true}});
            graph.addNode("VolumetricFogPass", "Fog", {{"localFogVolumes", true}, {"localFogStartDistanceMeters", 0.1},
                {"conservativeDepth", false}, {"jitter", false}});
            graph.addNode("VolumetricFogCompositePass", "Composite");
            for (auto field : {"visibility", "rasterInfo", "domain", "depth"}) {
                graph.addEdge(std::string("VBuffer.") + field, std::string("Deferred.") + field);
            }
            graph.addEdge("VBuffer.depth", "Fog.depth");
            graph.addEdge("VBuffer.rasterInfo", "Fog.rasterInfo");
            graph.addEdge("Deferred.color", "Composite.sceneColor");
            graph.addEdge("VBuffer.depth", "Composite.depth");
            graph.addEdge("Fog.integrated", "Composite.integrated");
            graph.addEdge("Fog.view", "Composite.view");
            const std::array outputs{"Deferred.color", "Composite.color", "Fog.integrated", "Fog.localFogTiles"};
            for (auto name : outputs) { graph.markOutput(name); }
            constexpr uint32_t width = 289, height = 65;
            constexpr uint64_t imageBytes = width * height * 16, volumeBytes = 19 * 5 * 64 * 8, tileBytes = 3 * 33;
            const std::array<uint64_t, 4> sizes{imageBytes, imageBytes, volumeBytes, tileBytes};
            auto readback = checked(context.device.createBuffer({.size = 2 * imageBytes + volumeBytes + tileBytes,
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}), "allocate readback");
            RenderView view;
            view.setRenderResolution(width, height);
            view.setCamera({.eye = {0, 1, 5}, .center = {0, 1, 4}, .orthoHeight = 2.0f, .orthographic = true});
            RenderGraphExecutor executor;
            executor.bindRenderView(&view);
            executor.bindRenderWorld(&world);
            std::string log;
            const auto errors = context.validationMessageCount ? context.validationMessageCount->load() : 0;
            if (!executor.compile(context.device, graph, width, height, log)) { return RHITestResult::fail(log); }
            std::vector<std::byte> baseline;
            for (int frame = 0; frame < 4; ++frame) {
                const bool enabled = frame != 3;
                const bool hasSphere = frame == 1;
                environment.volumetricFog.enabled = enabled;
                environment.volumetricFog.localVolumes.clear();
                if (hasSphere) {
                    // Pixel x=128 separates the first two tiles; the last tile has only 33 pixels.
                    environment.volumetricFog.localVolumes.push_back({.centerMeters = {-0.508f, 1, 0}, .radiusMeters = 1,
                        .heightFalloff = 1, .emissive = {1, 0.25f, 0}});
                }
                if (!world.setWorldEnvironment(environment)) { return RHITestResult::fail("Toggle failed"); }
                checked(executor.execute({.graphicsQueue = &context.graphicsQueue}), "execute fog graph");
                checked(executor.waitForSubmittedWork(), "wait for fog graph");
                bench::GPUCommands capture(context.graphicsQueue);
                checked(capture.initialize(context.device), "begin fog capture");
                uint64_t offset = 0;
                for (size_t i = 0; i < outputs.size(); ++i) {
                    auto& resource = *executor.outputResource(outputs[i]);
                    checked(executor.transitionOutput(*capture.commands, outputs[i], ResourceState::TransferSource), "prepare graph readback");
                    checked(capture.commands->copyTextureToBuffer({.texture = resource.texture,
                        .buffer = checked(readback->slice({offset, sizes[i]}), "readback range"),
                        .width = resource.desc.width, .height = resource.desc.height, .depth = resource.desc.depth,
                        .layerCount = resource.desc.layerCount}), "read graph output");
                    offset += sizes[i];
                }
                const BufferBarrierDesc host{.buffer = readback.get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead}};
                checked(capture.commands->synchronize({.buffers = {&host, 1}}), "publish readback");
                checked(capture.submitAndWait(), "submit fog capture");
                readback->invalidate();
                const auto* bytes = static_cast<const std::byte*>(readback->map());
                if (!bytes) { return RHITestResult::fail("Cannot map fog graph readback"); }
                const bool sameColor = std::memcmp(bytes, bytes + imageBytes, imageBytes) == 0;
                const uint16_t* volume = reinterpret_cast<const uint16_t*>(bytes + 2 * imageBytes);
                const auto* integrated = bytes + 2 * imageBytes;
                const auto* tiles = reinterpret_cast<const uint8_t*>(integrated + volumeBytes);
                const bool tilesMatch = hasSphere ? tiles[0] == 1 && tiles[1] == 1 && tiles[2] == 0 && tiles[3] == 0 && tiles[4] == 0 :
                    std::all_of(tiles, tiles + tileBytes, [](uint8_t value) { return value == 0; });
                if (!frame) { baseline.assign(integrated, integrated + volumeBytes); }
                const bool sameMedium = std::memcmp(integrated, baseline.data(), volumeBytes) == 0;
                constexpr std::array<uint16_t, 4> identity{0, 0, 0, 0x3c00};
                size_t mediumCount = 0;
                for (size_t i = 0; i < volumeBytes / 8; ++i) {
                    mediumCount += !std::equal(identity.begin(), identity.end(), volume + 4 * i);
                }
                readback->unmap();
                if (!tilesMatch || (frame == 1 && sameMedium) || (frame == 2 && !sameMedium)) {
                    return RHITestResult::fail("Unexpected tile culling or stale local fog history on frame " + std::to_string(frame));
                }
                if (enabled == sameColor || (mediumCount != 0) != enabled) {
                    return RHITestResult::fail("Fog toggle color/volume mismatch, enabled=" + std::to_string(enabled) +
                        " equalColor=" + std::to_string(sameColor) + " nonIdentityVoxels=" + std::to_string(mediumCount));
                }
            }
            if (context.validationMessageCount && context.validationMessageCount->load() != errors) {
                return RHITestResult::fail("Vulkan validation reported a fog graph error");
            }
            return RHITestResult::pass("Verified tile boundary, empty tile lists, exact local history removal and disabled HDR passthrough");
        } catch (const std::exception& e) { return RHITestResult::fail(e.what()); }
    }
};

METALLIC_REGISTER_RHI_TEST(VolumetricFogToggleTest);
} // namespace
} // namespace metallic::tests
