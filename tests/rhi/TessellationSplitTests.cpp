#include "RhiTest.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <numeric>
#include <set>

namespace metallic::tests {
namespace {

#define TESS_REQUIRE(expr) do { const auto result = (expr); if (!result) { \
    return RhiTestResult::fail(std::string(#expr) + ": " + toString(result)); } } while (false)

class TessellationSplitTest final : public RhiTest {
public:
    TessellationSplitTest() { type = RhiTestType::Rendering; name = "tessellation_recursive_topology"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Recursive tessellation topology",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless compute"); }
        TESS_REQUIRE(created);
        constexpr uint32_t count = 512, poison = 0xa5a5a5a5u;
        using Vec = std::array<float, 4>;
        struct Input { Vec p[3], eyeNear, forwardScale, options; };
        static_assert(sizeof(Input) == 96);
        using Record = std::array<uint32_t, 4>;
        std::vector<Input> input(count);
        std::vector<Record> output(count * 65, Record{poison,poison,poison,poison});
        // Pairs share the quad diagonal but have a different third vertex.
        // Rotate both source domains independently to exercise all edge masks.
        for (uint32_t i = 0; i < count; i += 2) {
            const uint32_t pair = i / 2;
            const float x = std::array<float, 4>{.25f, 6.f, 12.f, 80.f}[pair % 4];
            const float y = std::array<float, 4>{.5f, 6.f, 18.f, 160.f}[(pair / 4) % 4];
            const bool ortho = (pair & 16) != 0;
            const float z = (pair & 32) != 0 ? 2.99f : 0.f;
            const Vec a{0,0,z,0}, b{x,0,z,0}, c{x,y,z + (ortho ? 0.f : .1f),0}, d{0,y,z,0};
            for (uint32_t side = 0; side < 2; ++side) {
                auto& item = input[i + side];
                const std::array<Vec, 3> p = side == 0 ? std::array<Vec,3>{a,b,c} : std::array<Vec,3>{a,c,d};
                for (uint32_t v = 0; v < 3; ++v) { item.p[v] = p[(v + pair / 16 + side) % 3]; }
                item.eyeNear = {0,0,3,.02f}; item.forwardScale = {0,0,-1,1};
                item.options = {1, 8, float(pair / 64), ortho ? 1.f : 0.f};
            }
        }
        std::unique_ptr<BindlessHeap> heap;
        TESS_REQUIRE(device->createBindlessHeap({.maxBuffers = 2}, heap));
        std::array<std::unique_ptr<Buffer>, 2> buffers;
        std::array<BindlessHandle, 2> handles;
        const uint64_t sizes[] = {input.size() * sizeof(Input), output.size() * sizeof(Record)};
        const void* initial[] = {input.data(), output.data()};
        for (uint32_t i = 0; i < 2; ++i) {
            TESS_REQUIRE(device->createBuffer({.size = sizes[i], .structureStride = i == 0 ? 96u : 16u,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, buffers[i]));
            TESS_REQUIRE(heap->allocateBuffer(handles[i]));
            TESS_REQUIRE(heap->writeStorageBuffer(handles[i], *buffers[i]));
            void* p = buffers[i]->map();
            if (!p) { return RhiTestResult::fail("Cannot map split probe buffer"); }
            std::memcpy(p, initial[i], sizes[i]); buffers[i]->flush(); buffers[i]->unmap();
        }
        ShaderCompileResult compiled;
        const auto compilation = compileSlangShaderToSpirv({.moduleName = "TessellationSplitProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, compiled);
        if (!compilation) { return RhiTestResult::fail(compiled.diagnostics); }
        std::unique_ptr<ShaderModule> shader;
        TESS_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4}, shader));
        std::unique_ptr<ComputePipeline> pipeline;
        TESS_REQUIRE(device->createComputePipeline({.computeShader = shader.get(), .usesBindlessHeap = true,
            .bindlessUserPushDataSize = 8}, pipeline));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        TESS_REQUIRE(device->createCommandPool(*queue, pool));
        TESS_REQUIRE(pool->createCommandBuffer(commands)); TESS_REQUIRE(device->createFence(false, fence));
        TESS_REQUIRE(commands->begin()); commands->hostWriteBarrier();
        const BufferBarrierDesc barriers[] = {
            {.buffer = buffers[0].get(), .before = ResourceState::Undefined, .after = ResourceState::General},
            {.buffer = buffers[1].get(), .before = ResourceState::Undefined, .after = ResourceState::General}};
        commands->barrier({.buffers = barriers, .bufferCount = 2});
        commands->bindBindlessHeap(*heap); commands->bindComputePipeline(*pipeline);
        const uint32_t push[] = {handles[0].shaderIndex, handles[1].shaderIndex};
        commands->pushBindlessData(push, sizeof(push)); commands->dispatch(count / 16);
        TESS_REQUIRE(commands->end());
        CommandBuffer* list[] = {commands.get()};
        TESS_REQUIRE(queue->submit({.commandBuffers = list, .commandBufferCount = 1, .signalFence = fence.get()}));
        TESS_REQUIRE(fence->wait()); buffers[1]->invalidate();
        const void* p = buffers[1]->map();
        if (!p) { return RhiTestResult::fail("Cannot read split probe"); }
        std::memcpy(output.data(), p, sizes[1]); buffers[1]->unmap();

        const auto component = [](uint32_t v, uint32_t axis) -> int64_t {
            return axis == 0 ? 4096 - int64_t(v & 65535) - int64_t(v >> 16) : (axis == 1 ? v & 65535 : v >> 16);
        };
        using Fraction = std::pair<int64_t, int64_t>;
        std::set<Fraction> previousBoundary;
        uint32_t masks = 0, peak = 0, total = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const auto header = output[i * 65];
            const uint32_t leaves = header[0], depth = uint32_t(input[i].options[2]);
            if (leaves == 0 || leaves > (1u << (2 * depth)) || header[1] > 7) { return RhiTestResult::fail("Invalid leaf count or split mask"); }
            masks |= 1u << header[1]; peak = std::max(peak, leaves); total += leaves;
            int64_t area = 0;
            // Edge use and rate checks are independent of the implementation's
            // traversal order. Every internal boundary must have two owners.
            std::map<std::pair<uint32_t,uint32_t>, std::pair<uint32_t,uint32_t>> edges;
            for (uint32_t j = 0; j < 64; ++j) {
                const auto q = output[i * 65 + j + 1];
                if (j >= leaves) {
                    if (q != Record{poison,poison,poison,poison}) { return RhiTestResult::fail("Out-of-bounds leaf write"); }
                    continue;
                }
                for (uint32_t v = 0; v < 3; ++v) {
                    for (uint32_t a = 0; a < 3; ++a) {
                        if (component(q[v], a) < 0 || component(q[v], a) > 4096) { return RhiTestResult::fail("Split escaped source domain"); }
                    }
                }
                const auto x = component(q[0],1), y = component(q[0],2);
                const auto signedArea = (component(q[1],1)-x)*(component(q[2],2)-y) - (component(q[2],1)-x)*(component(q[1],2)-y);
                if (signedArea <= 0) { return RhiTestResult::fail("Degenerate or reversed leaf"); }
                area += signedArea;
                for (uint32_t e = 0; e < 3; ++e) {
                    const auto rate = (q[3] >> (8 * e)) & 255;
                    if (rate < 1 || rate > 8) { return RhiTestResult::fail("Unbounded leaf dice rate"); }
                    auto& edge = edges[std::minmax(q[e],q[(e+1)%3])];
                    if (edge.first++ != 0 && edge.second != rate) { return RhiTestResult::fail("Interior dice rates disagree"); }
                    edge.second = rate;
                }
            }
            if (area != 4096ll * 4096ll) { return RhiTestResult::fail("Budget termination lost or overlapped coverage"); }
            std::array<uint32_t, 2> common{}; uint32_t commonCount = 0;
            for (uint32_t a = 0; a < 3; ++a) {
                for (uint32_t b = 0; b < 3; ++b) {
                    if (input[i].p[a] == input[i ^ 1].p[b]) { common[commonCount++] = a; }
                }
            }
            if (commonCount != 2) { return RhiTestResult::fail("Invalid shared-edge fixture"); }
            if (input[i].p[common[0]] > input[i].p[common[1]]) { std::swap(common[0],common[1]); }
            const uint32_t zero = 3 - common[0] - common[1], end = common[1];
            std::set<Fraction> boundary;
            for (const auto& [key, usesRate] : edges) {
                const auto [a,b] = key; const auto [uses,rate] = usesRate;
                bool outer = false;
                for (uint32_t axis = 0; axis < 3; ++axis) { outer |= component(a,axis) == 0 && component(b,axis) == 0; }
                if (uses != (outer ? 1u : 2u)) { return RhiTestResult::fail("Non-manifold split edge / T-junction"); }
                if (component(a,zero) != 0 || component(b,zero) != 0) { continue; }
                for (uint32_t step = 0; step <= rate; ++step) {
                    const int64_t n = component(a,end) * (rate-step) + component(b,end) * step;
                    const int64_t d = 4096ll * rate, gcd = std::gcd(n,d);
                    boundary.emplace(n/gcd,d/gcd);
                }
            }
            if ((i & 1) != 0 && boundary != previousBoundary) { return RhiTestResult::fail("Neighbor source triangles produce different shared-edge samples"); }
            previousBoundary = std::move(boundary);
        }
        if (masks != 255 || peak != 64) { return RhiTestResult::fail("Probe failed to exercise all split masks and full depth budget"); }
        return RhiTestResult::pass(std::to_string(count) + " GPU roots / " + std::to_string(total) +
            " leaves: all split masks, depths 0..3, near-plane crossing, exact coverage, shared-edge samples and guarded writes");
    }
};
METALLIC_REGISTER_RHI_TEST(TessellationSplitTest);
#undef TESS_REQUIRE
} // namespace
} // namespace metallic::tests
