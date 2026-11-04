#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
void check(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T> void check(const Result<T>& result, const std::string& message)
{
    check(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct MappingCommands
{
    RenderFrameContext frame;
    QueueSubmissionTracker tracker;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    uint64_t index = 0;

    explicit MappingCommands(Device& device)
    {
        auto& queue = *device.getQueue(QueueType::Graphics);
        check(tracker.initialize(device, queue), "Content submission tracker");
        check(device.createCommandPool(queue).transform([&](auto p) { pool = std::move(p); }), "Content pool");
        check(pool->createCommandBuffer().transform([&](auto c) { commands = std::move(c); }), "Content commands");
    }
    ~MappingCommands()
    {
        frame.cancel();
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        (void)pool->reset();
        (void)frame.reset();
    }
    void begin()
    {
        check(frame.begin(index++), "Content frame begin");
        check(pool->reset(), "Content pool reset");
        check(commands->begin(frame.submissionContext()), "Content command begin");
    }
    void submit(Semaphore* gate = nullptr)
    {
        check(commands->end(), "Content command end");
        CommandBuffer* list[]{commands.get()};
        const SemaphoreSubmitDesc wait{.semaphore = gate, .value = 1};
        check(tracker.submit({.waitSemaphores = {gate ? &wait : nullptr, gate ? 1u : 0u},
            .commandBuffers = list}, frame), "Content submit");
    }
    void finish(bool cancel = false)
    {
        if (cancel) {
            check(commands->end(), "Content cancelled command end");
            frame.cancel();
            return;
        }
        submit();
        check(frame.wait(), "Content wait");
    }
};

struct HeightImage
{
    uint32_t width = 0;
    std::vector<uint8_t> pixels;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
};

HeightImage makeHeight(Device& device, MappingCommands& gpu, uint32_t width, bool inverted)
{
    HeightImage image;
    image.width = width;
    image.pixels.resize(width * width * 4);
    for (uint32_t y = 0; y < width; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            auto* p = image.pixels.data() + (y * width + x) * 4;
            // Every atlas texel has a distinct value, including page 0.
            p[0] = uint8_t(inverted ? 255u - (y * width + x) : y * width + x);
            p[1] = p[2] = p[0];
            p[3] = inverted ? 0 : (x < width / 2 ? 0 : 255);
        }
    }
    check(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
        .format = Format::RGBA8Unorm, .width = width, .height = width})
        .transform([&](auto t) { image.texture = std::move(t); }), "Height image");
    check(device.createTextureView(*image.texture, {.format = Format::RGBA8Unorm})
        .transform([&](auto v) { image.view = std::move(v); }), "Height view");
    auto upload = device.createBuffer({.size = image.pixels.size(), .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload});
    check(upload, "Height staging");
    auto* data = (*upload)->map(); check(data != nullptr, "Height map");
    std::memcpy(data, image.pixels.data(), image.pixels.size()); (*upload)->flush(); (*upload)->unmap();
    gpu.begin();
    const TextureBarrierDesc before{.texture = image.texture.get(), .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
    check(gpu.commands->synchronize({.textures = {&before, 1}}), "Height upload barrier");
    check((*upload)->slice().and_then([&](const auto& slice) {
        return gpu.commands->copyBufferToTexture({.texture = image.texture.get(), .buffer = slice, .width = width, .height = width});
    }), "Height upload");
    const TextureBarrierDesc after{.texture = image.texture.get(), .oldLayout = TextureLayout::TransferDestination,
        .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}};
    check(gpu.commands->synchronize({.textures = {&after, 1}}), "Height ready");
    gpu.finish();
    return image;
}

class VirtualShadowMapPropagatedMappingTest final : public RHITest
{
public:
    VirtualShadowMapPropagatedMappingTest() { name = "virtual_shadow_map_propagated_mapping"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapPropagatedMapping");
        std::filesystem::create_directories(directory);
        Json report{{"passed", false}, {"cases", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapPropagatedMapping.json") << report.dump(2); };
        try {
            auto created = createDevice({.applicationName = "VSM page fallback",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless and mesh shader module support"); }
            check(created, "Fallback device");
            auto& device = **created;
            MappingCommands gpu(device);
            auto atlas = makeHeight(device, gpu, 16, false);
            auto registry = ResourceRegistry::forDevice(device); check(registry, "Fallback registry");
            struct ProbePush
            {
                uint32_t pages, projections, statistics, atlas;
                uint32_t positions, output, count, axis;
                uint32_t levels, pageSize, physicalPagesX, reserved;
            };
            struct Projection
            {
                std::array<float, 4> rowX, rowY, rowZ, rowW, metric;
                std::array<uint32_t, 4> page;
                std::array<float, 4> originNear;
            };
            static_assert(sizeof(Projection) == 112);
            constexpr uint64_t abi = 0x56534d50524f5031ull;
            constexpr uint32_t levels = 12, pageSize = 4, count = 6, maxPages = levels * 64 + 1;
            std::array<ComputeKernel, 2> kernels;
            std::string log;
            const std::array capabilities{"spvMeshShadingEXT"};
            for (uint32_t index = 0; index < kernels.size(); ++index) {
                const std::array macros{SlangMacroDefine{"VSM_PROPAGATED_MAPPING_PROBE", "1"}};
                auto shader = compileSlangShaderToSpirv({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                    .entryPointName = index == 0 ? "propagatedMappingBuildProbeMain" : "propagatedMappingLookupProbeMain",
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders", .capabilities = capabilities,
                    .macroDefines = macros}, log);
                check(shader, "Fallback probe compile: " + log);
                check(kernels[index].initialize(device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<ProbePush>(abi, ParameterTransport::InlinePush)}, log), "Fallback probe pipeline: " + log);
            }
            const auto buffer = [&](uint64_t bytes, uint32_t stride, MemoryLocation memory) {
                auto result = device.createBuffer({.size = bytes, .structureStride = stride,
                    .usage = BufferUsageBits::Storage, .memoryLocation = memory});
                check(result, "Fallback buffer");
                return std::move(*result);
            };
            auto pages = buffer(maxPages * 2u * 16u, 16, MemoryLocation::HostUpload);
            auto projections = buffer((levels + 1u) * sizeof(Projection), sizeof(Projection), MemoryLocation::HostUpload);
            auto statistics = buffer(32u * 4u, 4, MemoryLocation::HostUpload);
            auto positions = buffer(count * 16u, 16, MemoryLocation::HostUpload);
            auto output = buffer(count * 128u, 16, MemoryLocation::HostReadback);
            const auto upload = [&](Buffer& destination, const void* data, size_t bytes) {
                void* mapped = destination.map(); check(mapped != nullptr, "Fallback upload map");
                std::memcpy(mapped, data, bytes); destination.flush(); destination.unmap();
            };
            std::vector<std::array<uint32_t, 4>> initial(maxPages * 2, {0xdeadbeefu, 0u, 0u, 0u});
            upload(*pages, initial.data(), initial.size() * sizeof(initial[0]));
            struct Case
            {
                const char* name;
                uint32_t axis;
                int x, y;
                uint32_t mapped, overflow;
                bool border, expectedDirect;
                uint32_t physical = UINT32_MAX;
                bool coarseValid = true;
                uint32_t staleFlags = UINT32_MAX;
                uint32_t lightType = 0;
                bool differentZ = false;
            };
            // Reuse the same GPU allocation: remapping and all-missing must replace stale metadata.
            const std::array cases{
                Case{"two-level-hit", 8, 0, 0, 2, 0, false, true},
                Case{"rebuild-next-level", 8, 0, 0, 3, 0, false, true},
                Case{"rebuild-all-missing", 8, 0, 0, levels, 0, false, true},
                Case{"negative-origin", 8, -5, -3, 2, 0, false, true},
                Case{"even-non-power-of-two", 6, -2, -2, 2, 0, false, false},
                Case{"odd-axis", 7, 0, 0, 2, 0, false, false},
                Case{"parent-border-inset", 8, 0, 0, 1, 0, true, false},
                Case{"overflow-coarse", 8, 0, 0, 2, 1, false, false},
                Case{"high-delta-boundary", 8, 0, 0, 11, 0, false, true},
                Case{"high-delta-resident-clamp", 8, 0, 0, 11, 0, false, true},
                Case{.name="unsupported-shared-z", .axis=8, .x=0, .y=0, .mapped=2, .overflow=0,
                    .border=false, .expectedDirect=false, .differentZ=true},
                Case{"own-level-hit", 8, 0, 0, 0, 0, false, true},
                Case{"physical-address-remap", 8, 0, 0, 2, 0, false, true, 15},
                Case{"target-invalid-repropagate", 8, 0, 0, levels, 0, false, true, UINT32_MAX, true, 2},
                Case{"target-unadmitted", 8, 0, 0, levels, 0, false, true, UINT32_MAX, true, 1},
                Case{"coarse-invalid", 8, 0, 0, levels, 0, false, true, UINT32_MAX, false},
                Case{"spot-dyadic-edges", 8, 0, 0, 2, 0, false, true, UINT32_MAX, true, UINT32_MAX, 1},
                Case{"spot-coarse-edges", 8, 0, 0, levels, 0, false, true, UINT32_MAX, true, UINT32_MAX, 1},
                Case{"spot-non-dyadic", 6, 0, 0, 2, 0, false, false, UINT32_MAX, true, UINT32_MAX, 1}};
            for (const auto& fixture : cases) {
                const bool highDelta = fixture.mapped == 11u;
                const bool splitHighDelta = highDelta && std::string_view(fixture.name) != "high-delta-resident-clamp";
                const uint32_t stride = fixture.axis * fixture.axis, totalPages = levels * stride + 1;
                std::vector<std::array<uint32_t, 4>> entries(totalPages, {UINT32_MAX, 0u, 0u, 0u});
                for (uint32_t level = 0; level < levels; ++level) {
                    for (uint32_t page = 0; page < stride; ++page) {
                        const bool mapped = level == fixture.mapped && (!splitHighDelta || page % fixture.axis >= 4u);
                        const uint32_t physical = fixture.physical == UINT32_MAX ? level + 1u : fixture.physical;
                        entries[level * stride + page] = {mapped ? physical : UINT32_MAX,
                            mapped ? 2u : 0u, level, mapped ? 1u : 0u};
                        if (fixture.staleFlags != UINT32_MAX && level == 2u) {
                            entries[level * stride + page] = {15u, fixture.staleFlags & 2u, level, fixture.staleFlags & 1u};
                        }
                    }
                }
                entries.back() = {0u, 2u, levels, fixture.coarseValid ? 1u : 0u};
                std::array<Projection, levels + 1> transforms{};
                for (uint32_t level = 0; level <= levels; ++level) {
                    const bool coarse = level == levels;
                    const float scale = std::ldexp(1.f, -int(level));
                    const float originX = coarse ? 0.f : std::floor(float(fixture.x) * scale);
                    const float originY = coarse ? 0.f : std::floor(float(fixture.y) * scale);
                    const float x = fixture.border && level == 1 ? 4.f : originX;
                    const float inverseExtent = coarse ? 1.f / 64.f : scale;
                    transforms[level] = {{inverseExtent, 0, 0, coarse ? 0.f : -x * (2.f / fixture.axis)},
                        {0, inverseExtent, 0, coarse ? 0.f : -originY * (2.f / fixture.axis)},
                        {0, 0, .5f, .5f}, {0, 0, 0, 1}, {1.f / inverseExtent, 0, 0, 0},
                        {coarse ? totalPages - 1u : level * stride, coarse ? 1u : fixture.axis, level, coarse ? 1u : 0u},
                        {x, originY, 0, 0}};
                    if (fixture.lightType != 0u) {
                        transforms[level] = {{1,0,0,0}, {0,1,0,0}, {0,0,.5f,.5f}, {0,0,.25f,1}, {1,0,0,1},
                            {coarse ? totalPages - 1u : level * stride, coarse ? 1u : std::max(1u,fixture.axis >> level),
                                level, coarse ? 1u : 0u}, {0,0,0,0}};
                    }
                    if (fixture.differentZ && level == 2u) { transforms[level].rowZ = {0,0,.5f,.625f}; }
                }
                const float centerX = float(fixture.x) * (2.f / fixture.axis);
                const float centerY = float(fixture.y) * (2.f / fixture.axis);
                const float boundary = centerX + 2.f / fixture.axis;
                const float infinity = std::numeric_limits<float>::infinity();
                std::array<std::array<float, 4>, count> samples{{
                    {centerX + 1.f / fixture.axis, centerY + 1.f / fixture.axis, 0, 0},
                    {std::nextafter(boundary, -infinity), centerY + 1.f / fixture.axis, 0, 0},
                    {boundary, centerY + 1.f / fixture.axis, 0, 0},
                    {std::nextafter(boundary, infinity), centerY + 1.f / fixture.axis, 0, 0},
                    {centerX + 1.25f, centerY + 1.f / fixture.axis, 0, 0},
                    {centerX + 1.f / fixture.axis, centerY + 1.f / fixture.axis, 2, 0}}};
                if (fixture.border) {
                    // Exact lower inset at parent level 1, plus its adjacent representable positions.
                    samples[1][0] = std::nextafter(.0625f, -infinity);
                    samples[2][0] = .0625f;
                    samples[3][0] = std::nextafter(.0625f, infinity);
                }
                if (highDelta) {
                    // Integer ancestry selects x=3 although the independent old projection
                    // rounds to x=4. The resident variant must clamp the texel to page 3.
                    constexpr float x = -1.562500074e-5f;
                    samples[1][0] = std::nextafter(x, -infinity);
                    samples[2][0] = x;
                    samples[3][0] = std::nextafter(x, infinity);
                }
                if (fixture.lightType != 0u) {
                    // Exact cube/spot edges are inclusive and clamp to the final texel.
                    samples[1] = {-1.f, .125f, 0.f, 0.f};
                    samples[2] = {1.f, .125f, 0.f, 0.f};
                    samples[3] = {std::nextafter(0.f, -infinity), .125f, 0.f, 0.f};
                    samples[4] = {std::nextafter(1.f, infinity), .125f, 0.f, 0.f};
                    samples[5] = {.125f, .125f, 3.f, 0.f};
                }
                std::array<uint32_t, 32> stats{}; stats[5] = fixture.overflow;
                upload(*pages, entries.data(), entries.size() * sizeof(entries[0]));
                upload(*projections, transforms.data(), sizeof(transforms));
                upload(*statistics, stats.data(), sizeof(stats)); upload(*positions, samples.data(), sizeof(samples));
                gpu.begin();
                gpu.commands->hostWriteBarrier();
                const BufferBarrierDesc recycled{.buffer = pages.get(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
                check(gpu.commands->synchronize({.buffers = {&recycled, 1}}), "Fallback metadata recycle");
                ParameterWriter writer(device, **registry);
                const ProbePush push{writer.buffer(pages.get()).index, writer.buffer(projections.get()).index,
                    writer.buffer(statistics.get()).index, writer.sampledImage(atlas.view.get()).index,
                    writer.buffer(positions.get()).index, writer.buffer(output.get()).index,
                    count, fixture.axis, levels, pageSize, 4, fixture.lightType};
                auto encoded = writer.encode(push, abi, ParameterTransport::InlinePush);
                check(encoded, "Fallback parameters");
                check(kernels[0].dispatch(*gpu.commands, *encoded, (totalPages + 63u) / 64u), "Fallback metadata build");
                const BufferBarrierDesc ready{.buffer = pages.get(),
                    .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
                check(gpu.commands->synchronize({.buffers = {&ready, 1}}), "Fallback metadata ready");
                check(kernels[1].dispatch(*gpu.commands, *encoded, 1), "Fallback lookup");
                const std::array hostReady{
                    BufferBarrierDesc{.buffer = output.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}},
                    BufferBarrierDesc{.buffer = pages.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}}};
                check(gpu.commands->synchronize({.buffers = hostReady}), "Fallback readback ready");
                gpu.finish();
                pages->invalidate();
                const void* mappingData = pages->map(); check(mappingData != nullptr, "Mapping prefix readback");
                const bool prefixUnchanged = std::memcmp(mappingData, entries.data(), entries.size() * sizeof(entries[0])) == 0;
                std::vector<std::array<uint32_t,4>> actualMappings(totalPages);
                std::memcpy(actualMappings.data(), static_cast<const std::byte*>(mappingData) + entries.size() * sizeof(entries[0]),
                    actualMappings.size() * sizeof(actualMappings[0]));
                pages->unmap();
                check(prefixUnchanged, std::string(fixture.name) + ": propagation changed residency/cache prefix");
                std::array<std::array<uint32_t, 32>, count> results{};
                output->invalidate(); const void* mapped = output->map();
                check(mapped != nullptr, "Fallback readback map");
                std::memcpy(results.data(), mapped, sizeof(results)); output->unmap();
                const auto resident = [](const auto& entry) {
                    return entry[0] != UINT32_MAX && (entry[1] & 2u) != 0u && (entry[3] & 1u) != 0u;
                };
                // Independent CPU routing oracle: integer world-page coordinates are
                // divided with mathematical floor, not the shader's signed bit shifts.
                const auto expectedMapping = [&](uint32_t id) -> std::array<uint32_t,4> {
                    const std::array<uint32_t,4> unsupported{UINT32_MAX,levels,0,0};
                    if (id + 1u == totalPages) { return unsupported; }
                    const uint32_t level = id / stride, local = id % stride;
                    if (resident(entries[id])) { return {entries[id][0],level,0,1}; }
                    const auto& source = transforms[level];
                    const uint32_t axis = source.page[1];
                    if (!std::has_single_bit(axis) || (fixture.lightType == 0u && axis < 2u) || local >= axis * axis) {
                        return unsupported;
                    }
                    const std::array<uint32_t,2> sourcePage{local % axis,local / axis};
                    for (uint32_t target = level + 1u; target < levels; ++target) {
                        const auto& parent = transforms[target];
                        const uint32_t delta = target - level;
                        std::array<int64_t,2> parentPage{};
                        if (fixture.lightType != 0u) {
                            if (parent.page[1] != std::max(1u,axis >> delta) || parent.rowX != source.rowX ||
                                parent.rowY != source.rowY || parent.rowZ != source.rowZ || parent.rowW != source.rowW) {
                                return unsupported;
                            }
                            const uint32_t divisor = 1u << std::min(delta,std::bit_width(axis)-1u);
                            parentPage = {sourcePage[0]/divisor,sourcePage[1]/divisor};
                        } else {
                            if (parent.rowZ != source.rowZ) { return unsupported; }
                            for (uint32_t component = 0; component < 3; ++component) {
                                if (parent.rowX[component] != std::ldexp(source.rowX[component],-int(delta)) ||
                                    parent.rowY[component] != std::ldexp(source.rowY[component],-int(delta))) { return unsupported; }
                            }
                            for (uint32_t axisIndex = 0; axisIndex < 2; ++axisIndex) {
                                const int64_t worldPage = int64_t(sourcePage[axisIndex]) + int64_t(source.originNear[axisIndex]) - axis/2;
                                parentPage[axisIndex] = int64_t(std::floor(std::ldexp(double(worldPage),-int(delta)))) -
                                    int64_t(parent.originNear[axisIndex]) + axis/2;
                            }
                            if (parentPage[0] < 0 || parentPage[1] < 0 || parentPage[0] >= axis || parentPage[1] >= axis) { continue; }
                            if (parentPage[0] == 0 || parentPage[1] == 0 || parentPage[0] == axis-1 || parentPage[1] == axis-1) {
                                return unsupported;
                            }
                        }
                        const auto& entry = entries[parent.page[0] + uint32_t(parentPage[1]) * parent.page[1] + uint32_t(parentPage[0])];
                        if (resident(entry)) {
                            return {entry[0],target,fixture.lightType == 0u ? uint32_t(parentPage[0]) | (uint32_t(parentPage[1]) << 16u) : 0u,
                                fixture.lightType == 0u ? 2u : 3u};
                        }
                    }
                    return {UINT32_MAX,levels,0,fixture.lightType == 0u ? 2u : 3u};
                };
                std::vector<std::array<uint32_t,4>> expectedMappings(totalPages);
                for (uint32_t page = 0; page < totalPages; ++page) {
                    expectedMappings[page] = expectedMapping(page);
                    check(actualMappings[page] == expectedMappings[page], std::string(fixture.name) +
                        ": integer mapping oracle mismatch at page " + std::to_string(page));
                }
                const auto project = [&](uint32_t level, const auto& position) {
                    const auto& transform = transforms[level];
                    const auto dot = [&](const auto& row) {
                        return float(double(row[0])*position[0] + double(row[1])*position[1] +
                            double(row[2])*position[2] + row[3]);
                    };
                    std::array<float,3> value{dot(transform.rowX),dot(transform.rowY),dot(transform.rowZ)};
                    if (transform.metric[3] > .5f) {
                        const float w = dot(transform.rowW);
                        for (auto& component : value) { component /= w; }
                    }
                    return value;
                };
                const auto contains = [&](const auto& p, uint32_t level, bool inset) {
                    const auto& transform = transforms[level];
                    if (p[2] < 0 || p[2] > 1) { return false; }
                    if (transform.metric[3] > .5f) { return p[0] >= -1 && p[0] <= 1 && p[1] >= -1 && p[1] <= 1; }
                    const float margin = inset ? 1.f/float(transform.page[1]*pageSize) : 0.f;
                    return p[0] >= -1.f+margin && p[0] < 1.f-margin && p[1] >= -1.f+margin && p[1] < 1.f-margin;
                };
                const auto virtualTexel = [&](const auto& p, uint32_t level) {
                    std::array<uint32_t,2> address{};
                    const uint32_t resolution = transforms[level].page[1]*pageSize;
                    for (uint32_t component = 0; component < 2; ++component) {
                        address[component] = uint32_t(std::floor((p[component]*.5f+.5f)*float(resolution)));
                        if (fixture.lightType != 0u) { address[component] = std::min(address[component],resolution-1u); }
                    }
                    return address;
                };
                struct LookupOracle
                {
                    bool valid = false;
                    uint32_t level = UINT32_MAX, physical = UINT32_MAX, coarseMiss = 0;
                    std::array<float,3> projected{};
                    std::array<uint32_t,2> texel{};
                    bool clamped = false;
                };
                const auto lookupOracle = [&](const auto& position) {
                    LookupOracle expected;
                    expected.level = levels;
                    if (!fixture.overflow) {
                        for (uint32_t level = 0; level < levels; ++level) {
                            auto projected = project(level,position);
                            if (!contains(projected,level,true)) { continue; }
                            auto texel = virtualTexel(projected,level);
                            const auto& source = transforms[level];
                            const uint32_t page = source.page[0] + texel[1]/pageSize*source.page[1] + texel[0]/pageSize;
                            const auto& mapping = expectedMappings[page];
                            if (mapping[3] == 0u) { continue; }
                            const uint32_t target = mapping[1];
                            if (target == levels) { break; }
                            if (mapping[3] == 2u) {
                                const auto& parent = transforms[target];
                                const double scale = std::ldexp(1.0,-int(target-level));
                                for (uint32_t component = 0; component < 2; ++component) {
                                    const float sourceW = component == 0 ? source.rowX[3] : source.rowY[3];
                                    const float parentW = component == 0 ? parent.rowX[3] : parent.rowY[3];
                                    projected[component] = float(double(projected[component])*scale + double(parentW)-double(sourceW)*scale);
                                    const uint32_t parentPage = component == 0 ? mapping[2] & 0xffffu : mapping[2] >> 16u;
                                    const uint32_t lower = parentPage*pageSize, upper = lower+pageSize-1u;
                                    const float continuous = (projected[component]*.5f+.5f)*float(parent.page[1]*pageSize);
                                    expected.clamped |= continuous < float(lower) || continuous > float(upper);
                                    texel[component] = uint32_t(std::clamp(continuous,float(lower),float(upper)));
                                }
                            } else if (mapping[3] == 3u) {
                                projected = project(target,position);
                                texel = virtualTexel(projected,target);
                            }
                            expected.valid = true; expected.level = target; expected.physical = mapping[0];
                            expected.projected = projected; expected.texel = texel;
                            return expected;
                        }
                    }
                    expected.projected = project(levels,position);
                    if (!contains(expected.projected,levels,false)) { return expected; }
                    if (!resident(entries.back())) { expected.coarseMiss = 1; return expected; }
                    expected.valid = true; expected.physical = entries.back()[0];
                    expected.texel = virtualTexel(expected.projected,levels);
                    return expected;
                };
                const auto closeFloat = [](float actual, float expected) {
                    if (actual == expected) { return true; }
                    const float spacing = std::abs(std::nextafter(expected,std::numeric_limits<float>::infinity())-expected);
                    return std::isfinite(actual) && std::abs(actual-expected) <= 4.f*spacing;
                };
                Json oracleObservations = Json::array();
                uint32_t clampedSamples = 0;
                for (uint32_t sample = 0; sample < count; ++sample) {
                    const auto& actual = results[sample];
                    const auto expected = lookupOracle(samples[sample]);
                    const std::string label = std::string(fixture.name) + ": sample " + std::to_string(sample);
                    check(actual[0] == uint32_t(expected.valid) && actual[1] == expected.level, label + ": validity/level oracle mismatch");
                    check(actual[3] == 1u && actual[6] == uint32_t(expected.valid && expected.level == levels) &&
                        actual[7] == expected.coarseMiss, label + ": lookup counter oracle mismatch");
                    if (expected.valid) {
                        check(actual[4] == expected.texel[0] && actual[5] == expected.texel[1], label + ": texel ownership oracle mismatch");
                        for (uint32_t component = 0; component < 3; ++component) {
                            check(closeFloat(std::bit_cast<float>(actual[8+component]),expected.projected[component]), label + ": relative projection oracle mismatch");
                        }
                        const uint32_t x = expected.physical%4u*pageSize+expected.texel[0]%pageSize;
                        const uint32_t y = expected.physical/4u*pageSize+expected.texel[1]%pageSize;
                        const float depth = float(atlas.pixels[(y*atlas.width+x)*4u])/255.f;
                        check(closeFloat(std::bit_cast<float>(actual[2]),depth), label + ": physical atlas texel oracle mismatch");
                    }
                    if (actual[12] && actual[15] == 2u && actual[14] < levels) {
                        check(actual[26] == actual[18], label + ": shared Z changed");
                        check(actual[24] == actual[8] && actual[25] == actual[9] && actual[28] == actual[4] && actual[29] == actual[5],
                            label + ": direct helper and lookup disagree");
                    }
                    clampedSamples += expected.clamped;
                    oracleObservations.push_back({{"valid",expected.valid},{"level",expected.level},{"physical",expected.physical},
                        {"projected",expected.projected},{"texel",expected.texel},{"clamped",expected.clamped}});
                }
                const uint32_t expectedLevel = fixture.overflow ? levels : fixture.mapped;
                check(results[0][0] == (fixture.coarseValid ? 1u : 0u) && results[0][1] == expectedLevel,
                    std::string(fixture.name) + ": wrong mapped level/validity");
                check(results[0][12] == uint32_t(fixture.expectedDirect), std::string(fixture.name) + ": expected direct/slow path not exercised");
                if (fixture.expectedDirect) {
                    const uint32_t expectedPhysical = fixture.mapped == levels ? UINT32_MAX :
                        fixture.physical == UINT32_MAX ? fixture.mapped + 1u : fixture.physical;
                    check(results[0][13] == expectedPhysical && results[0][14] == fixture.mapped,
                        std::string(fixture.name) + ": stale/wrong propagated physical address or target");
                    check(results[0][15] == (fixture.mapped == 0u ? 1u : fixture.lightType != 0u ? 3u : 2u),
                        "Wrong own/ancestor mapping tag");
                }
                if (fixture.lightType != 0u && fixture.expectedDirect) {
                    for (uint32_t sample = 1; sample <= 3; ++sample) {
                        check(results[sample][12] == 1u, "Dyadic local boundary failed direct mapping");
                    }
                    check(results[4][0] == 0u && results[5][0] == 0u, "Local out-of-domain lookup became valid");
                }
                if (fixture.lightType == 0u && !fixture.border && !fixture.overflow && fixture.mapped != 0u && fixture.axis == 8 && fixture.x == 0) {
                    for (uint32_t sample = 1; sample <= 3; ++sample) {
                        if (!fixture.differentZ) { check(results[sample][12] == 1u, "Integer page boundary did not use direct mapping"); }
                        if (splitHighDelta) {
                            check(results[sample][1] == levels,
                                "High-delta integer ancestry unexpectedly selected the independent world-projection page");
                        }
                    }
                }
                if (std::string_view(fixture.name) == "high-delta-resident-clamp") {
                    check(clampedSamples >= 3u, "Target-page texel clamp was not exercised");
                }
                report["cases"].push_back({{"name", fixture.name}, {"axis", fixture.axis}, {"lightType", fixture.lightType}, {"samples", samples},
                    {"lookup", results}, {"expectedLevel", expectedLevel},
                    {"expectedDirect", fixture.expectedDirect},{"integerMappingOraclePages",totalPages},
                    {"lookupOracle",oracleObservations},{"clampedSamples",clampedSamples}});
            }
            report["validation"] = validation.snapshot(); check(!validation.failed(), "Fallback Vulkan validation error");
            report["passed"] = true; save();
            report["contract"] = "Integer ancestor routing, relative directional XY, shared Z and mapped-page texel clamp; boundary ownership is checked by the CPU oracle";
            save();
            return RHITestResult::pass("Propagated mappings match the integer/relative projection oracle, including boundary texel ownership");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapPropagatedMappingTest);

class VirtualShadowMapProjectedRayStateTest final : public RHITest
{
public:
    VirtualShadowMapProjectedRayStateTest() { name = "virtual_shadow_map_projected_ray_state"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapProjectedRayState");
        std::filesystem::create_directories(directory);
        Json report{{"passed", false}, {"cases", Json::array()},
            {"scope", "Dyadic/shared-Z directional ray coordinates, integer page ancestry and constrained GPU SMRT samples inside Mark intervals; finite fixture coverage, not a universal floating-point certificate"}};
        bench::ValidationRecorder validation;
        const auto save = [&] { std::ofstream(directory / "VirtualShadowMapProjectedRayState.json") << report.dump(2) << '\n'; };
        try {
            auto created = createDevice({.applicationName = "VSM projected ray state",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless and mesh shader module support"); }
            check(created, "Projected ray device");
            auto& device = **created;
            MappingCommands gpu(device);
            auto atlas = makeHeight(device, gpu, 16, false);
            auto registry = ResourceRegistry::forDevice(device); check(registry, "Projected ray registry");
            struct Push
            {
                uint32_t pages, projections, statistics, atlas, positions, output, count, axis;
                uint32_t levels, pageSize, physicalPagesX, reserved;
            };
            struct Projection
            {
                std::array<float, 4> rowX, rowY, rowZ, rowW, metric;
                std::array<uint32_t, 4> page;
                std::array<float, 4> originNear;
            };
            struct Input
            {
                std::array<float, 4> position, endpoint, dither;
                std::array<uint32_t, 4> control;
            };
            static_assert(sizeof(Projection) == 112 && sizeof(Input) == 64 && sizeof(Push) == 48);
            constexpr uint64_t abi = 0x56534d5052415931ull;
            constexpr uint32_t levels = 12, axis = 8, pageSize = 4, stride = axis * axis;
            constexpr uint32_t totalPages = levels * stride + 1, rayCount = 6, samples = 11;
            using Packet = std::array<std::array<uint32_t, 4>, 10>;
            std::array<ComputeKernel, 2> kernels;
            std::string log;
            const std::array capabilities{"spvMeshShadingEXT"};
            const std::array macros{SlangMacroDefine{"VSM_PROPAGATED_MAPPING_PROBE", "1"},
                SlangMacroDefine{"VSM_PROJECTED_RAY_PROBE", "1"}, SlangMacroDefine{"VSM_DIRECTIONAL_PROJECTION", "1"},
                SlangMacroDefine{"VSM_DIRECTIONAL_MARK", "1"}};
            for (uint32_t index = 0; index < kernels.size(); ++index) {
                auto shader = compileSlangShaderToSpirv({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                    .entryPointName = index == 0 ? "propagatedMappingBuildProbeMain" : "projectedRayStateProbeMain",
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders", .capabilities = capabilities, .macroDefines = macros}, log);
                check(shader, "Projected ray compile: " + log);
                check(kernels[index].initialize(device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<Push>(abi, ParameterTransport::InlinePush)}, log), "Projected ray pipeline: " + log);
            }
            const auto buffer = [&](uint64_t bytes, uint32_t elementSize, MemoryLocation location) {
                auto result = device.createBuffer({.size = bytes, .structureStride = elementSize,
                    .usage = BufferUsageBits::Storage, .memoryLocation = location});
                check(result, "Projected ray buffer"); return std::move(*result);
            };
            auto pages = buffer(totalPages * 32u, 16, MemoryLocation::HostUpload);
            auto projections = buffer((levels + 1u) * sizeof(Projection), sizeof(Projection), MemoryLocation::HostUpload);
            auto statistics = buffer(32u * 4u, 4, MemoryLocation::HostUpload);
            auto inputs = buffer(rayCount * sizeof(Input), sizeof(Input), MemoryLocation::HostUpload);
            auto output = buffer(rayCount * samples * sizeof(Packet), 16, MemoryLocation::HostReadback);
            const auto upload = [&](Buffer& destination, const void* data, size_t bytes) {
                void* mapped = destination.map(); check(mapped != nullptr, "Projected ray upload");
                std::memcpy(mapped, data, bytes); destination.flush(); destination.unmap();
            };
            struct Case { const char* name; uint32_t resident, first; int x, y; bool overflow, coarseValid; float offset; };
            const std::array cases{
                Case{"own",0,0,0,0,false,true,0}, Case{"nonzero-receiver-level",2,2,0,0,false,true,0},
                Case{"ancestor",2,0,0,0,false,true,0}, Case{"negative-origin",2,0,-5,-3,false,true,0},
                Case{"level-delta-eleven",11,0,0,0,false,true,0}, Case{"coarse",levels,0,0,0,false,true,0},
                Case{"overflow",0,0,0,0,true,true,0}, Case{"coarse-invalid",levels,0,0,0,false,false,0},
                Case{"outside-coarse",levels,0,0,0,true,true,300}};
            uint64_t tested = 0, coverage = 0;
            for (const auto& fixture : cases) {
                validation.phase(fixture.name);
                std::array<Projection, levels + 1> transforms{};
                for (uint32_t level = 0; level <= levels; ++level) {
                    const bool coarse = level == levels;
                    const float scale = coarse ? 1.f / 64.f : std::ldexp(1.f, -int(level));
                    const float x = coarse ? 0.f : std::floor(float(fixture.x) * std::ldexp(1.f, -int(level)));
                    const float y = coarse ? 0.f : std::floor(float(fixture.y) * std::ldexp(1.f, -int(level)));
                    const uint32_t pageAxis = coarse ? 1u : axis;
                    transforms[level] = {{scale,0,0,coarse ? 0.f : -x * (2.f / axis)},
                        {0,scale,0,coarse ? 0.f : -y * (2.f / axis)}, {0,0,1.f/32.f,.5f}, {0,0,0,1},
                        {1.f/scale,2.f/(scale*pageAxis*pageSize),.001f,0},
                        {coarse ? totalPages-1u : level*stride,pageAxis,level,coarse ? 1u : 0u}, {x,y,0,0}};
                }
                std::vector<std::array<uint32_t, 4>> table(totalPages * 2u, {UINT32_MAX,0,0,0});
                for (uint32_t level = 0; level < levels; ++level) {
                    for (uint32_t page = 0; page < stride; ++page) {
                        if (level == fixture.resident) { table[level*stride+page] = {5u,2u,level,1u}; }
                    }
                }
                table[totalPages-1u] = {0u,2u,levels,fixture.coarseValid ? 1u : 0u};
                std::array<uint32_t, 32> stats{}; stats[5] = fixture.overflow;
                std::array<Input, rayCount> rays{};
                for (uint32_t ray = 0; ray < rayCount; ++ray) {
                    const float x = fixture.x * (2.f/axis) + fixture.offset + (ray & 1u ? -.3125f : .28125f);
                    const float y = fixture.y * (2.f/axis) + (ray & 2u ? -.21875f : .15625f);
                    const float z = ray < 2 ? -2.f + ray : 0.f;
                    rays[ray] = {{x,y,z,0}, {x+.375f,y-.25f,z+1.5f,0}, {.015625f,-.03125f,.0078125f,0},
                        {fixture.first,ray < 2 ? 0u : ray-2u,ray*3u,ray < 2 ? 0u : 1u}};
                }
                upload(*pages, table.data(), table.size()*sizeof(table[0]));
                upload(*projections, transforms.data(), sizeof(transforms)); upload(*statistics, stats.data(), sizeof(stats));
                upload(*inputs, rays.data(), sizeof(rays));
                gpu.begin(); gpu.commands->hostWriteBarrier();
                const BufferBarrierDesc recycled{.buffer=pages.get(),
                    .before={PipelineStageBits::AllCommands,AccessBits::MemoryRead|AccessBits::MemoryWrite},
                    .after={PipelineStageBits::ComputeShader,AccessBits::ShaderRead|AccessBits::ShaderWrite}};
                check(gpu.commands->synchronize({.buffers={&recycled,1}}), "Projected ray metadata recycle");
                ParameterWriter writer(device, **registry);
                const Push push{writer.buffer(pages.get()).index,writer.buffer(projections.get()).index,
                    writer.buffer(statistics.get()).index,writer.sampledImage(atlas.view.get()).index,
                    writer.buffer(inputs.get()).index,writer.buffer(output.get()).index,rayCount,axis,levels,pageSize,4,0};
                auto parameters = writer.encode(push, abi, ParameterTransport::InlinePush); check(parameters, "Projected ray parameters");
                check(kernels[0].dispatch(*gpu.commands, *parameters, (totalPages+63u)/64u), "Projected ray mapping");
                const BufferBarrierDesc ready{.buffer=pages.get(), .before={PipelineStageBits::ComputeShader,AccessBits::ShaderWrite},
                    .after={PipelineStageBits::ComputeShader,AccessBits::ShaderRead}};
                check(gpu.commands->synchronize({.buffers={&ready,1}}), "Projected ray mapping ready");
                check(kernels[1].dispatch(*gpu.commands, *parameters, 1), "Projected ray dispatch");
                const BufferBarrierDesc host{.buffer=output.get(), .before={PipelineStageBits::ComputeShader,AccessBits::ShaderWrite},
                    .after={PipelineStageBits::Host,AccessBits::HostRead}};
                check(gpu.commands->synchronize({.buffers={&host,1}}), "Projected ray readback ready"); gpu.finish();
                output->invalidate(); const void* mapped = output->map(); check(mapped != nullptr, "Projected ray map");
                std::array<Packet, rayCount*samples> results{}; std::memcpy(results.data(), mapped, sizeof(results)); output->unmap();
                double maximumError = 0;
                uint64_t caseCoverage = 0;
                const auto value = [](uint32_t bits) { return std::bit_cast<float>(bits); };
                const auto close = [&](float actual, double expected, const std::string& label) {
                    const double error = std::abs(double(actual)-expected);
                    maximumError = std::max(maximumError,error);
                    check(std::isfinite(actual) && error <= 16.0*std::numeric_limits<float>::epsilon()*std::max(1.0,std::abs(expected)),
                        std::string(fixture.name)+": "+label+" exceeds the float coordinate bound");
                };
                for (uint32_t ray = 0; ray < rayCount; ++ray) {
                    for (uint32_t step = 0; step < samples; ++step) {
                        const auto& result = results[ray*samples+step];
                        const auto& input = rays[ray];
                        const double time = value(result[3][3]);
                        check(time >= 0 && time <= 1 && (step != 0 || time == 0) && (step != 1 || time == 1) &&
                            (step != samples-1 || time == 0), "Projected ray endpoints or sample time changed");
                        std::array<double,3> endpoint{}, dither{}, point{}, lookup{};
                        for (uint32_t component = 0; component < 3; ++component) {
                            endpoint[component] = value(result[3][component]); dither[component] = value(result[4][component]);
                            if (input.control[3] == 0) {
                                check(endpoint[component] == input.endpoint[component] && dither[component] == input.dither[component],
                                    "Synthetic projected ray input layout changed");
                            }
                            point[component] = double(input.position[component]) + time*(endpoint[component]-input.position[component]);
                            lookup[component] = point[component]+dither[component];
                        }
                        const auto project = [&](const auto& position, uint32_t level) {
                            std::array<double,3> p{};
                            const auto& transform = transforms[level];
                            const std::array rows{transform.rowX,transform.rowY,transform.rowZ};
                            for (uint32_t component = 0; component < 3; ++component) {
                                p[component] = rows[component][3];
                                for (uint32_t axisIndex = 0; axisIndex < 3; ++axisIndex) { p[component] += double(rows[component][axisIndex])*position[axisIndex]; }
                            }
                            return p;
                        };
                        const auto contains = [&](const auto& p, uint32_t level, bool inset) {
                            const double margin = inset ? 1.0/(transforms[level].page[1]*pageSize) : 0.0;
                            return p[0] >= -1+margin && p[0] < 1-margin && p[1] >= -1+margin && p[1] < 1-margin && p[2] >= 0 && p[2] <= 1;
                        };
                        const auto baseLookup = project(lookup,fixture.first), baseCompare = project(point,fixture.first);
                        for (uint32_t component = 0; component < 3; ++component) {
                            close(value(result[5][component]),baseLookup[component],"lookup origin/step");
                            close(value(result[6][component]),baseCompare[component],"undithered origin/step");
                        }
                        std::array<double,3> receiver{input.position[0],input.position[1],input.position[2]};
                        const double receiverDepth = project(receiver,fixture.first)[2];
                        close(value(result[4][3]),receiverDepth,"receiver depth");
                        close(value(result[5][3]),1.0/32.0,"depth scale");
                        close(value(result[2][3]),(.75-receiverDepth)*32.0,"linear ray depth");
                        uint32_t first = levels;
                        for (uint32_t level = fixture.first; level < levels; ++level) {
                            if (contains(project(lookup,level),level,true)) { first=level; break; }
                        }
                        check(result[9][3] == first, std::string(fixture.name)+": first containing clipmap differs from the double oracle");
                        uint32_t expectedLevel = levels;
                        std::array<int64_t,2> parentPage{};
                        if (!fixture.overflow && first < levels && fixture.resident >= first && fixture.resident < levels) {
                            const auto& source = transforms[first];
                            const auto& target = transforms[fixture.resident];
                            const uint32_t delta = fixture.resident-first;
                            for (uint32_t component = 0; component < 2; ++component) {
                                const float projected = value(result[9][component]);
                                const auto expected = project(lookup,first);
                                close(projected,expected[component],"first clipmap position");
                                const uint32_t texel = uint32_t(std::floor((projected*.5f+.5f)*float(axis*pageSize)));
                                const int64_t worldPage = int64_t(texel/pageSize)+int64_t(source.originNear[component])-int64_t(axis/2u);
                                parentPage[component] = int64_t(std::floor(double(worldPage)/std::ldexp(1.0,int(delta))))-
                                    int64_t(target.originNear[component])+int64_t(axis/2u);
                            }
                            if (parentPage[0] >= 0 && parentPage[0] < axis && parentPage[1] >= 0 && parentPage[1] < axis) { expectedLevel=fixture.resident; }
                        }
                        const auto expectedLookup = project(lookup,expectedLevel), expectedCompare = project(point,expectedLevel);
                        const bool expectedValid = expectedLevel < levels || (fixture.coarseValid && contains(expectedLookup,levels,false));
                        check(result[0][0] == uint32_t(expectedValid) && result[0][1] == expectedLevel,
                            std::string(fixture.name)+": lookup validity/level differs from integer ancestry oracle");
                        for (uint32_t component = 0; component < 3; ++component) {
                            close(value(result[2][component]),expectedCompare[component],"target undithered comparison");
                            if (expectedValid) { close(value(result[1][component]),expectedLookup[component],"target lookup"); }
                        }
                        if (expectedValid) {
                            const uint32_t resolution = transforms[expectedLevel].page[1]*pageSize;
                            for (uint32_t component = 0; component < 2; ++component) {
                                const double bound = 16.0*std::numeric_limits<float>::epsilon()*std::max(1.0,std::abs(expectedLookup[component]));
                                const double lower = expectedLevel < levels ? double(parentPage[component]*pageSize) : 0.0;
                                const double upper = expectedLevel < levels ? lower+pageSize-1u : double(resolution-1u);
                                const uint32_t minimum = uint32_t(std::floor(std::clamp((expectedLookup[component]-bound)*.5*resolution+.5*resolution,lower,upper)));
                                const uint32_t maximum = uint32_t(std::floor(std::clamp((expectedLookup[component]+bound)*.5*resolution+.5*resolution,lower,upper)));
                                check(result[0][component+2u] >= minimum && result[0][component+2u] <= maximum,
                                    std::string(fixture.name)+": texel is outside its rounded integer parent-page interval");
                            }
                            const uint32_t physical = expectedLevel < levels ? 5u : 0u;
                            const uint32_t x = (physical%4u)*pageSize+result[0][2]%pageSize;
                            const uint32_t y = (physical/4u)*pageSize+result[0][3]%pageSize;
                            close(value(result[1][3]),double(y*16u+x)/255.0,"physical atlas address");
                        }
                        check(result[7][1] == 0, std::string(fixture.name)+": a generated projected sample lies outside the original Mark interval");
                        check(result[7][2] == 1, "Projected lookup did not retain its filtered-query count");
                        if (input.control[3] != 0) {
                            double lengthSquared = 0;
                            for (uint32_t component=0; component<3; ++component) { lengthSquared += std::pow(endpoint[component]-input.position[component],2); }
                            const double length = std::sqrt(lengthSquared);
                            check(std::abs(length-3.0) < 1e-4 && (endpoint[2]-input.position[2])/length >= std::cos(.08)-1e-5,
                                "Mark coverage ray is outside the authored light cone");
                            const double texel = transforms[fixture.first].metric[1];
                            check(std::abs(dither[0]) <= texel*.5+1e-6 && std::abs(dither[1]) <= texel*.5+1e-6 && dither[2] == 0,
                                "Mark coverage dither exceeds the receiver-selected texel");
                            caseCoverage += result[7][0];
                        }
                        ++tested;
                    }
                }
                coverage += caseCoverage;
                report["cases"].push_back({{"name",fixture.name},{"samples",rayCount*samples},{"receiverLevel",fixture.first},
                    {"residentLevel",fixture.resident},{"overflow",fixture.overflow},{"maximumCoordinateError",maximumError},
                    {"markCoverageSamples",caseCoverage},{"zeroMarkCoverageMisses",true}}); save();
            }
            check(coverage > 0, "Projected ray Mark coverage fixture was empty");
            report["testedRaySamples"] = tested; report["markCoverageSamples"] = coverage;
            report["zeroMarkCoverageMisses"] = true; report["validation"] = validation.snapshot();
            check(!validation.failed(), "Projected ray validation error"); report["passed"] = true; save();
            return RHITestResult::pass("Projected directional rays match double coordinate/integer ancestry oracles and constrained SMRT samples remain inside Mark coverage");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapProjectedRayStateTest);
} // namespace
} // namespace metallic::tests
