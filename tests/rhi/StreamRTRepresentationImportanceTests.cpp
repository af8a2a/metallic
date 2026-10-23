#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <json.hpp>

namespace metallic::tests {
namespace {
using namespace render;

template<typename T>
T importanceChecked(Result<T> result)
{
    if (!result) { throw std::runtime_error(resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

void importanceRequire(bool condition, const char* reason)
{
    if (!condition) { throw std::runtime_error(reason); }
}

constexpr uint32_t kInstanceCount = 137;
constexpr uint32_t kPrimitiveCount = 46;
constexpr uint32_t kLevelCount = kPrimitiveCount * 4;
constexpr uint32_t kOwnerCount = kPrimitiveCount * 3;
constexpr uint32_t kPrimitiveBase = kLevelCount * 4;
constexpr uint32_t kInstanceBase = kPrimitiveBase + kPrimitiveCount * 12;
constexpr uint32_t kImportanceBase = kInstanceBase + kInstanceCount * 4;
constexpr uint32_t kStateWords = kImportanceBase + kOwnerCount * 12 + 2;
using ImportanceState = std::array<uint32_t, kStateWords>;
using Importance = std::array<uint32_t, 6>;
using ImportanceOwners = std::array<Importance, kOwnerCount>;

struct ImportanceInput {
    std::array<MeshletStreamGPUInstance, kInstanceCount> instances{};
    std::array<uint32_t, kInstanceCount> minimumLevels{};
    bool orthographic = false;
    bool allowSaturatedError = false;

    ImportanceInput() { minimumLevels.fill(1); }
};

MeshletStreamGPUInstance importanceInstance(float x, float z, uint32_t primitive = 0)
{
    MeshletStreamGPUInstance result;
    result.primitiveIndex = primitive;
    result.visible = 1;
    result.world0[0] = result.world1[1] = result.world2[2] = result.world3[3] = 1.f;
    result.world3[0] = x; result.world3[2] = z;
    result.boundsCenterRadius[0] = x; result.boundsCenterRadius[2] = z;
    result.boundsCenterRadius[3] = .5f;
    return result;
}

float importanceError(const Importance& value)
{
    importanceRequire(value[0] != 0, "An active consumer has no importance key");
    return std::bit_cast<float>(value[0] - 1);
}

bool importanceClose(float a, float b)
{
    return std::abs(a - b) <= 1e-5f * std::max({1.f, std::abs(a), std::abs(b)});
}

// Runs the actual phase 20 reset and phase 23 owner classification/aggregation.
// The intervening CPU upload seeds already-classified LOD bounds, so this test
// isolates importance from DAG traversal, storage admission and native builds.
struct ImportanceFixture {
    enum Input { Params, Instances, Primitives, State, Cache, Active, Header, Metadata, InputCount };
    Device& device;
    Queue& queue;
    std::unique_ptr<BindlessHeap> heap;
    std::array<std::unique_ptr<Buffer>, InputCount> buffers;
    std::array<BindlessHandle, InputCount> handles;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    std::unique_ptr<Fence> fence;
    MeshletStreamGPUParams params;
    MeshletStreamUserPush push;
    bool submitted = false;
    uint32_t frameNumber = 0;

    explicit ImportanceFixture(Device& value) : device(value), queue(*value.getQueue(QueueType::Graphics))
    {
        const uint64_t sizes[] = {sizeof(params), kInstanceCount * sizeof(MeshletStreamGPUInstance),
            kPrimitiveCount * sizeof(MeshletStreamGPUPrimitive), sizeof(ImportanceState), kPrimitiveCount * 16 * 4,
            sizeof(MeshletStreamGPUActiveHeader), sizeof(MeshletStreamGPUBLASHeader),
            (kLevelCount + kPrimitiveCount) * sizeof(MeshletStreamGPURTRepresentationLODMetadata)};
        const uint32_t strides[] = {sizeof(params), sizeof(MeshletStreamGPUInstance), sizeof(MeshletStreamGPUPrimitive),
            4, 4, sizeof(MeshletStreamGPUActiveHeader), sizeof(MeshletStreamGPUBLASHeader), 16};
        heap = importanceChecked(device.createBindlessHeap({.maxBuffers = InputCount}));
        for (uint32_t input = 0; input < InputCount; ++input) {
            buffers[input] = importanceChecked(device.createBuffer({.size = sizes[input], .structureStride = strides[input],
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}));
            handles[input] = importanceChecked(heap->allocate(BindlessHandleKind::Buffer));
            const auto slice = importanceChecked(buffers[input]->slice());
            importanceChecked(heap->writeStorageBuffer(handles[input], slice));
            void* mapped = buffers[input]->map();
            importanceRequire(mapped != nullptr, "Importance buffer map failed");
            std::memset(mapped, 0, size_t(sizes[input])); buffers[input]->flush(); buffers[input]->unmap();
        }
        std::string diagnostics;
        auto compiled = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamBLASInputEntryPoint, .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, diagnostics);
        if (!compiled) { throw std::runtime_error(diagnostics); }
        shader = importanceChecked(device.createShaderModule({.spirv = compiled->spirv}));
        pipeline = importanceChecked(device.createComputePipeline({.computeShader = {shader.get()},
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(push)}));
        pool = importanceChecked(device.createCommandPool(queue));
        commands = importanceChecked(pool->createCommandBuffer());
        fence = importanceChecked(device.createFence(false));

        std::array<MeshletStreamGPUPrimitive, kPrimitiveCount> primitives{};
        for (uint32_t primitive = 0; primitive < kPrimitiveCount; ++primitive) {
            primitives[primitive].lodLevelOffset = primitive * 4;
            primitives[primitive].lodLevelCount = 4;
        }
        write(Primitives, primitives);
        std::array<MeshletStreamGPURTRepresentationLODMetadata, kLevelCount + kPrimitiveCount> metadata{};
        for (auto& entry : metadata) { entry.boundsCenterRadius[3] = .5f; entry.errorAndPadding[0] = .25f; }
        metadata[kLevelCount + 2].errorAndPadding[0] = 0.f;
        write(Metadata, metadata);
        params.sceneInstanceCount = kInstanceCount; params.scenePrimitiveCount = kPrimitiveCount;
        params.sceneLodLevelCount = kLevelCount;
        params.rtRepresentationCount = kOwnerCount;
        params.enableBlasRepresentations = 1;
        params.rtRepresentationStateBuffer = handles[State].shaderIndex;
        params.rtRepresentationCacheBuffer = handles[Cache].shaderIndex;
        params.rtRepresentationHeaderBuffer = handles[Active].shaderIndex;
        params.rtRepresentationMetadataBuffer = handles[Metadata].shaderIndex;
        params.lodTopologyBuffer = params.lodStateBuffer = handles[State].shaderIndex;
        params.center[2] = -1.f; params.upProjection[1] = 1.f;
        params.viewport[0] = 1.f; params.viewport[1] = params.viewport[2] = 512.f;
        params.viewport[3] = 1.57079632679f;
        params.clipOrtho[0] = .1f; params.clipOrtho[1] = 1000.f; params.clipOrtho[2] = 10.f;
        params.rtLodFlags = 3;
        push.paramsBuffer = handles[Params].shaderIndex;
        push.instanceBuffer = handles[Instances].shaderIndex;
        push.primitiveBuffer = push.groupBuffer = handles[Primitives].shaderIndex;
        push.pageTableBuffer = handles[State].shaderIndex;
        push.blasHeaderBuffer = handles[Header].shaderIndex;
    }

    ~ImportanceFixture() { (void)queue.waitIdle(); }

    template<typename T>
    void write(Input input, const T& value)
    {
        void* mapped = buffers[input]->map(); importanceRequire(mapped != nullptr, "Importance upload map failed");
        std::memcpy(mapped, &value, sizeof(value)); buffers[input]->flush(); buffers[input]->unmap();
    }

    ImportanceState readState()
    {
        buffers[State]->invalidate(); const void* mapped = buffers[State]->map();
        importanceRequire(mapped != nullptr, "Importance state map failed");
        ImportanceState value; std::memcpy(value.data(), mapped, sizeof(value)); buffers[State]->unmap(); return value;
    }

    void dispatch(uint32_t phase, uint32_t traversal = 0, bool cancel = false)
    {
        if (submitted) { importanceChecked(fence->reset()); importanceChecked(pool->reset()); }
        importanceChecked(commands->begin()); commands->hostWriteBarrier();
        importanceChecked(commands->bindBindlessHeap(*heap));
        push.activeBuildPhase = phase;
        push.traversalPhase = traversal;
        importanceChecked(commands->bindExecution(pipeline->execution(), &push, sizeof(push)));
        importanceChecked(commands->dispatch((kInstanceCount + 63) / 64));
        std::array<BufferBarrierDesc, InputCount> barriers;
        for (uint32_t input = 0; input < InputCount; ++input) {
            barriers[input] = {.buffer = buffers[input].get(),
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}};
        }
        importanceChecked(commands->synchronize({.buffers = barriers}));
        importanceChecked(commands->end()); CommandBuffer* list[]{commands.get()};
        if (cancel) { return; }
        importanceChecked(queue.submit({.commandBuffers = list, .signalFence = fence.get()}));
        importanceChecked(fence->wait()); submitted = true;
    }

    ImportanceOwners frame(const ImportanceInput& input)
    {
        params.frameIndex = ++frameNumber;
        params.upProjection[3] = input.orthographic ? 1.f : 0.f;
        write(Params, params); write(Instances, input.instances);
        dispatch(20);
        auto state = readState();
        for (uint32_t owner = 0; owner < kOwnerCount; ++owner) {
            const uint32_t base = kImportanceBase + owner * 6;
            importanceRequire(state[base] == 0 && state[base + 1] == 0x7f7fffffu &&
                state[base + 2] == 0 && state[base + 3] == 0 &&
                state[base + 4] == 0 && state[base + 5] == 0,
                "Production reset retained importance or consumers from a previous frame");
        }
        for (uint32_t primitive = 0; primitive < kPrimitiveCount; ++primitive) {
            const uint32_t base = kPrimitiveBase + primitive * 12;
            state[base] = 0; state[base + 1] = 1; state[base + 2] = 2;
        }
        for (uint32_t instance = 0; instance < kInstanceCount; ++instance) {
            const uint32_t base = kInstanceBase + instance * 4;
            state[base] = input.minimumLevels[instance];
            state[base + 1] = 1; state[base + 2] = UINT32_MAX; state[base + 3] = 0;
        }
        write(State, state); dispatch(23); state = readState();
        ImportanceOwners output;
        for (uint32_t owner = 0; owner < kOwnerCount; ++owner) {
            std::copy_n(state.begin() + kImportanceBase + owner * 6, 6, output[owner].begin());
            const auto& value = output[owner];
            importanceRequire(value[3] == value[4] + value[5], "Visible/offscreen counts do not partition consumers");
            if (value[3]) {
                const float error = importanceError(value), distance = std::bit_cast<float>(value[1]);
                const float coverage = std::bit_cast<float>(value[2]);
                importanceRequire(std::isfinite(error) && error >= 0 && (input.allowSaturatedError || error < 1e20f) &&
                    std::isfinite(distance) && distance >= 0 && std::isfinite(coverage) && coverage >= 0 && coverage <= 1,
                    "Ordinary finite bounds produced an invalid or sentinel importance metric");
            } else {
                importanceRequire(value[0] == 0 && value[2] == 0, "An owner with no consumers retained importance");
            }
        }
        return output;
    }

    std::vector<uint32_t> sort(const ImportanceOwners& values, uint32_t count = kOwnerCount,
        bool cache = false, bool initialize = true, bool expectedReuse = false, bool cancelSort = false)
    {
        params.rtRepresentationCount = count;
        params.enableBlasPriority = 1;
        params.enableBlasPriorityCache = cache;
        write(Params, params);
        // Production owns exactly three representations per primitive. Smaller
        // counts below exercise the sorter alone, so seed only its reset words
        // rather than aliasing that artificial tail with full-scene importance.
        if (count == kOwnerCount) { dispatch(20, initialize ? 1u : 0u); }
        auto state = readState();
        if (count != kOwnerCount) {
            state[kImportanceBase + count * 12] = !cache || initialize ? 1u : 0u;
            state[kImportanceBase + count * 12 + 1] = 0u;
        }
        for (uint32_t owner = 0; owner < count; ++owner) {
            std::copy(values[owner].begin(), values[owner].end(), state.begin() + kImportanceBase + owner * 6);
        }
        write(State, state);
        dispatch(41, initialize ? 1u : 0u);
        if (cancelSort) {
            const auto before = readState();
            dispatch(35, 0, true);
            importanceRequire(before == readState(), "Canceled sort recording mutated submitted GPU state");
            return {};
        }
        dispatch(35);
        uint32_t parity = 0;
        for (uint32_t runLog2 = 6; (1u << runLog2) < count; ++runLog2) {
            dispatch(36, runLog2 | (parity << 8));
            parity ^= 1;
        }
        dispatch(37, parity << 8);
        state = readState();
        std::vector<uint32_t> expected(count);
        std::iota(expected.begin(), expected.end(), 0);
        std::sort(expected.begin(), expected.end(), [&](uint32_t a, uint32_t b) {
            if (values[a][0] != values[b][0]) { return values[a][0] > values[b][0]; }
            if (values[a][1] != values[b][1]) { return values[a][1] < values[b][1]; }
            if (values[a][2] != values[b][2]) { return values[a][2] > values[b][2]; }
            return a < b;
        });
        const uint32_t orderBase = kImportanceBase + count * 6;
        importanceRequire(state[orderBase + count * 6 + 1] == uint32_t(expectedReuse),
            "Exact sort cache reuse diagnostic disagrees with expected invalidation");
        for (uint32_t rank = 0; rank < count; ++rank) {
            importanceRequire(state[orderBase + rank] == expected[rank] &&
                state[orderBase + count * 2 + expected[rank]] == rank,
                "GPU tile sort/merge/normalization disagrees with the stable importance oracle or inverse ranks");
        }
        params.rtRepresentationCount = kOwnerCount;
        params.enableBlasPriority = 0;
        write(Params, params);
        return expected;
    }
};

class StreamRTRepresentationImportanceTest final : public RHITest {
public:
    StreamRTRepresentationImportanceTest()
    {
        type = RHITestType::Rendering;
        name = "stream_rt_representation_importance";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::atomic_uint validationErrors = 0;
        auto created = createDevice({.applicationName = "RT representation importance",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}});
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("Bindless unavailable")
                : RHITestResult::fail(resultToString(created));
        }
        auto device = std::move(*created);
        std::string phase = "shared owner aggregation";
        try {
            ImportanceFixture fixture(*device);
            ImportanceInput farInput;
            farInput.instances[0] = importanceInstance(0.f, -40.f);
            const auto far = fixture.frame(farInput)[1];
            ImportanceInput nearInput;
            nearInput.instances[136] = importanceInstance(0.f, -4.f);
            const auto near = fixture.frame(nearInput)[1];
            importanceRequire(near[0] > far[0] && near[1] < far[1] && near[2] > far[2],
                "Near consumer does not outrank the far representative");
            ImportanceInput shared;
            for (auto& instance : shared.instances) { instance = farInput.instances[0]; }
            shared.instances[136] = nearInput.instances[136];
            shared.instances[1] = importanceInstance(0.f, 0.f);
            shared.instances[1].visible = 0;
            auto aggregate = fixture.frame(shared);
            importanceRequire(aggregate[1][0] == near[0] && aggregate[1][1] == near[1] &&
                aggregate[1][2] == near[2] && aggregate[1][3] == 136 && aggregate[1][4] == 136 && aggregate[1][5] == 0,
                "Shared importance did not aggregate all workgroups or excluded hidden consumers incorrectly");
            auto permuted = shared;
            std::reverse(permuted.instances.begin(), permuted.instances.end());
            importanceRequire(fixture.frame(permuted) == aggregate,
                "Permuting instance indices changed owner importance");

            phase = "offscreen and behind-camera ray demand";
            ImportanceInput offscreen;
            offscreen.instances[5] = importanceInstance(40.f, -4.f);
            const auto side = fixture.frame(offscreen)[1];
            offscreen.instances[5] = importanceInstance(0.f, 40.f);
            const auto behind = fixture.frame(offscreen)[1];
            for (const auto& value : {side, behind}) {
                importanceRequire(value[0] > 1 && value[2] == 0 && value[3] == 1 && value[4] == 0 && value[5] == 1,
                    "Offscreen or behind-camera consumer lost finite ray importance or acquired raster coverage");
            }

            phase = "near-plane and inside-sphere saturation";
            ImportanceInput nearPlaneInput;
            nearPlaneInput.allowSaturatedError = true;
            nearPlaneInput.instances[5] = importanceInstance(0.f, -.55f);
            const auto nearPlane = fixture.frame(nearPlaneInput)[1];
            nearPlaneInput.instances[5] = importanceInstance(0.f, 0.f);
            const auto insideSphere = fixture.frame(nearPlaneInput)[1];
            for (const auto& value : {nearPlane, insideSphere}) {
                importanceRequire(value[0] == 0x7f800000u && std::isfinite(importanceError(value)) &&
                    value[2] == std::bit_cast<uint32_t>(1.f) && value[3] == 1 && value[4] == 1 && value[5] == 0,
                    "Near-plane/inside-sphere saturation lost finite FLT_MAX key encoding or conservative full coverage");
            }
            importanceRequire(nearPlane[1] > 0 && insideSphere[1] == 0,
                "Nearest surface distance did not distinguish near-plane intersection from an inside-sphere camera");

            phase = "orthographic distance invariance";
            farInput.orthographic = nearInput.orthographic = true;
            const auto farOrtho = fixture.frame(farInput)[1], nearOrtho = fixture.frame(nearInput)[1];
            importanceRequire(importanceClose(importanceError(farOrtho), importanceError(nearOrtho)) &&
                importanceClose(std::bit_cast<float>(farOrtho[2]), std::bit_cast<float>(nearOrtho[2])) &&
                nearOrtho[1] < farOrtho[1], "Orthographic error or coverage changes with depth");

            phase = "scale and shear";
            nearInput.orthographic = false;
            auto scaled = nearInput;
            scaled.instances[136].world0[0] = scaled.instances[136].world1[1] = scaled.instances[136].world2[2] = 2.f;
            const auto scale = fixture.frame(scaled)[1];
            auto sheared = nearInput;
            sheared.instances[136].world1[0] = 2.f;
            const auto shear = fixture.frame(sheared)[1];
            importanceRequire(scale[0] > near[0] && scale[1] < near[1] && scale[2] >= near[2] &&
                shear[0] > near[0] && shear[1] < near[1] && shear[2] >= near[2],
                "Scale/shear failed to conservatively increase fallback importance");
            auto nonuniform = nearInput;
            nonuniform.instances[136].world0[0] = 2.f;
            nonuniform.instances[136].world1[1] = .5f;
            nonuniform.instances[136].world2[2] = 1.5f;
            const auto positiveNonuniform = fixture.frame(nonuniform)[1];
            nonuniform.instances[136].world0[0] = -2.f;
            nonuniform.instances[136].world2[2] = -1.5f;
            const auto negativeNonuniform = fixture.frame(nonuniform)[1];
            importanceRequire(negativeNonuniform == positiveNonuniform && negativeNonuniform[0] > near[0] &&
                negativeNonuniform[1] < near[1], "Negative nonuniform scale changed conservative sphere importance");

            phase = "active zero error versus unused owners";
            auto zeroError = nearInput;
            zeroError.instances[136].primitiveIndex = 2;
            const auto zero = fixture.frame(zeroError);
            importanceRequire(zero[7][0] == 1 && zero[7][3] == 1 && zero[1][0] == 0 && zero[1][3] == 0,
                "Zero-error active owner collided with unused-owner sentinel");

            phase = "continuous removal and owner migration";
            nlohmann::json frames = nlohmann::json::array();
            for (uint32_t frame = 0; frame < 32; ++frame) {
                auto input = shared;
                const uint32_t mode = frame % 3;
                if (frame % 4 == 1) { input.instances[136].visible = 0; }
                if (frame % 4 >= 2) {
                    input.instances[136].primitiveIndex = 1;
                    input.minimumLevels[136] = mode;
                }
                const auto output = fixture.frame(input);
                const bool nearOriginal = frame % 4 == 0;
                importanceRequire(output[1][0] == (nearOriginal ? near[0] : far[0]) &&
                    output[1][1] == (nearOriginal ? near[1] : far[1]) &&
                    output[1][2] == (nearOriginal ? near[2] : far[2]) && output[1][3] == (nearOriginal ? 136u : 135u),
                    "Removing or migrating the near consumer left stale shared maxima/minima/counts");
                for (uint32_t owner = 3; owner < kOwnerCount; ++owner) {
                    const bool current = frame % 4 >= 2 && owner == 3 + mode;
                    importanceRequire(output[owner][0] == (current ? near[0] : 0u) &&
                        output[owner][3] == (current ? 1u : 0u), "Owner migration retained importance in the previous owner");
                }
                frames.push_back({{"frame", frame}, {"owners", output}});
            }
            const auto empty = fixture.frame(ImportanceInput{});
            for (const auto& owner : empty) {
                importanceRequire(owner[0] == 0 && owner[2] == 0 && owner[3] == 0,
                    "Removing all consumers failed to reset owner importance");
            }
            phase = "137-owner partial-tail priority sort";
            ImportanceOwners keys{};
            for (uint32_t owner = 0; owner < kOwnerCount; ++owner) {
                keys[owner] = {std::bit_cast<uint32_t>(float(owner % 7)) + 1,
                    std::bit_cast<uint32_t>(float(owner % 5)),
                    std::bit_cast<uint32_t>(float(owner % 3) * .25f), 1, 1, 0};
            }
            keys[0] = far;
            keys[136] = near;
            // Consumer counts are telemetry, not an extra ranking factor.
            keys[11] = keys[12] = aggregate[1];
            keys[11][3] = keys[11][4] = 1;
            keys[127] = side; keys[128] = behind;
            keys[129] = zero[7]; keys[130] = empty[0];
            auto sortedOwners = fixture.sort(keys, 137);
            importanceRequire(std::find(sortedOwners.begin(), sortedOwners.end(), 136) <
                std::find(sortedOwners.begin(), sortedOwners.end(), 0), "High-index near owner lost to low-index far owner");
            for (uint32_t iteration = 0; iteration < 8; ++iteration) {
                std::rotate(keys.begin(), keys.begin() + 17, keys.begin() + 137);
                sortedOwners = fixture.sort(keys, 137);
            }
            // 65 owners exercises final normalization from B; 137 ends in A.
            fixture.sort(keys, 65);
            fixture.sort(keys, 64);
            phase = "exact priority cache consecutive frames and cancellation";
            fixture.sort(keys, kOwnerCount, true, true, false);
            for (uint32_t frame = 0; frame < 32; ++frame) {
                fixture.sort(keys, kOwnerCount, true, false, true);
            }
            // Every comparator word is exact, including one-bit changes which
            // can leave the total order unchanged. Consumer telemetry alone is
            // not an input to sorting (admission validates its own demand).
            for (uint32_t word = 0; word < 3; ++word) {
                ++keys[136][word];
                fixture.sort(keys, kOwnerCount, true, false, false);
                fixture.sort(keys, kOwnerCount, true, false, true);
            }
            ++keys[136][3]; ++keys[136][5];
            fixture.sort(keys, kOwnerCount, true, false, true);
            keys[128] = empty[0]; // Offscreen membership loss must invalidate.
            fixture.sort(keys, kOwnerCount, true, false, false);
            ++keys[127][0];
            fixture.sort(keys, kOwnerCount, true, false, false, true);
            fixture.sort(keys, kOwnerCount, true, false, false);
            fixture.sort(keys, kOwnerCount, true, false, true);
            // Baseline always sorts and publishes history, even while caching
            // is disabled. Toggling caching back on can reuse that exact order.
            ++keys[127][1];
            fixture.sort(keys, kOwnerCount, false, false, false);
            fixture.sort(keys, kOwnerCount, true, false, true);
            fixture.sort(keys, 65, true, true, false);
            fixture.sort(keys, 65, true, false, true);
            fixture.sort(keys, 64, true, true, false);
            fixture.sort(keys, 64, true, false, true);
            importanceRequire(validationErrors.load() == 0, "Vulkan validation errors in importance regression");
            const auto directory = context.outputDirectory / "RTRepresentationImportance";
            std::filesystem::create_directories(directory);
            std::ofstream report(directory / "Frames.json");
            report << nlohmann::json{{"frames", frames}, {"far", far}, {"near", near},
                {"shared", aggregate}, {"side", side}, {"behind", behind}, {"farOrtho", farOrtho},
                {"nearPlane", nearPlane}, {"insideSphere", insideSphere}, {"negativeNonuniform", negativeNonuniform},
                {"priorityOwnerCount", 137}, {"priorityOrder", sortedOwners},
                {"exactPriorityCacheStableFrames", 32}, {"priorityCacheCancellationVerified", true},
                {"nearOrtho", nearOrtho}, {"scale", scale}, {"shear", shear}, {"zeroError", zero},
                {"validationErrors", validationErrors.load()},
                {"scope", "Production phase20/23; classified LOD bounds seeded; no admission or native BLAS build"}}.dump(2);
            return RHITestResult::pass("137 consumers across three workgroups; shared max/min aggregation; hidden exclusion; permutation invariance; finite offscreen ray demand; ortho/scale/shear; zero-error sentinel; 32 reset/removal/owner-migration frames");
        } catch (const std::exception& error) { return RHITestResult::fail(phase + ": " + error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamRTRepresentationImportanceTest);
} // namespace
} // namespace metallic::tests
