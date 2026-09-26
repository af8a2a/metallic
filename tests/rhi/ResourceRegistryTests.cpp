#include "RhiTest.h"
#include "Runtime/Render/ComputeKernel.h"
#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

#define REG_REQUIRE(expression) do { \
    const render::Result result = (expression); \
    if (!result) { return RhiTestResult::fail(std::string(#expression) + ": " + toString(result)); } \
} while (false)
#define REG_CHECK(expression) do { \
    if (!(expression)) { return RhiTestResult::fail(#expression); } \
} while (false)

constexpr uint64_t kAbi = 0x5245475000000001ull;
struct ProbeParams {
    render::ShaderBuffer source, output;
    uint32_t add, index;
};
static_assert(sizeof(ProbeParams) == 24 && offsetof(ProbeParams, add) == 16);

render::Result makeBuffer(render::Device& device, std::unique_ptr<render::Buffer>& buffer, uint32_t value = 0)
{
    auto result = device.createBuffer({.size = 64, .structureStride = 4,
        .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Indirect,
        .memoryLocation = render::MemoryLocation::HostReadback,
        .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute}, buffer);
    if (!result) { return result; }
    void* mapped = buffer->map();
    if (!mapped) { return render::makeError(render::Error::Failure); }
    std::memset(mapped, 0, 64);
    std::memcpy(mapped, &value, 4);
    buffer->flush();
    buffer->unmap();
    return {};
}

struct Commands {
    render::RenderFrameContext frame;
    std::unique_ptr<render::CommandPool> pool;
    std::unique_ptr<render::CommandBuffer> commands;
    ~Commands()
    {
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        if (pool) { (void)pool->reset(); }
        (void)frame.reset();
    }
    render::Result initialize(render::Device& device, render::Queue& queue)
    {
        auto result = device.createCommandPool(queue, pool);
        return result ? pool->createCommandBuffer(commands) : result;
    }
    render::Result begin(uint64_t index)
    {
        auto result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? commands->begin(&frame) : result;
    }
    render::Result submit(render::QueueSubmissionTracker& tracker, render::Semaphore& gate)
    {
        auto result = commands->end();
        if (!result) { return result; }
        render::CommandBuffer* buffers[] = {commands.get()};
        render::SemaphoreSubmitDesc wait{.semaphore = &gate, .value = 1};
        return tracker.submit({.waitSemaphores = &wait, .waitSemaphoreCount = 1,
            .commandBuffers = buffers, .commandBufferCount = 1}, frame);
    }
};

struct Drain {
    render::Queue& queue;
    render::Semaphore& gate;
    ~Drain()
    {
        if (gate.currentValue() < 1) { (void)gate.signal(1); }
        (void)queue.waitIdle();
    }
};

render::Result makeKernel(render::Device& device, render::ComputeKernel& kernel, std::string& log)
{
    render::ShaderCompileResult shader;
    auto result = render::compileSlangShaderToSpirv({.moduleName = "RegistryProbe",
        .entryPointName = "registryProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
    if (!result) { log = shader.diagnostics; return result; }
    return kernel.initialize(device, {.spirv = shader.spirv, .parameters = render::parameterAbi<ProbeParams>(kAbi)}, log);
}

class RegistryIdentityTest final : public RhiTest {
public:
    RegistryIdentityTest() { type = RhiTestType::Resource; name = "registry_identity_capacity_and_views"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        REG_REQUIRE(render::createDevice({.applicationName = "Registry identity", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}, device));
        render::ResourceRegistry registry;
        REG_REQUIRE(registry.initialize(*device, {.maxSamplers = 2, .maxSampledImages = 2,
            .maxStorageImages = 1, .maxBuffers = 2}));
        std::unique_ptr<render::Buffer> a, b, c;
        REG_REQUIRE(makeBuffer(*device, a));
        REG_REQUIRE(makeBuffer(*device, b));
        REG_REQUIRE(makeBuffer(*device, c));
        render::ResourceLease aLease, bLease, duplicate, cLease;
        REG_REQUIRE(registry.storageBuffer(*a, aLease));
        REG_REQUIRE(registry.storageBuffer(*b, bLease));
        REG_REQUIRE(registry.storageBuffer(*b, duplicate));
        REG_CHECK(bLease.shaderValue() == duplicate.shaderValue());
        REG_CHECK(registry.stats().descriptorWrites == 2 && registry.stats().cacheHits == 1);
        std::weak_ptr<void> oldAllocation = a->retainAllocation();
        a.reset();
        REG_CHECK(!oldAllocation.expired());
        REG_CHECK(render::hasError(registry.storageBuffer(*c, cLease), render::Error::OutOfMemory));
        REG_CHECK(!cLease.valid());
        const auto releasedIndex = aLease.shaderValue();
        aLease = {};
        REG_CHECK(oldAllocation.expired());
        REG_REQUIRE(registry.storageBuffer(*c, cLease));
        REG_CHECK(cLease.shaderValue() == releasedIndex);
        REG_CHECK(registry.stats().liveDescriptors == 2);

        std::unique_ptr<render::Texture> texture;
        std::unique_ptr<render::TextureView> first, second;
        REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage,
            .format = render::Format::Rgba8Unorm}, texture));
        REG_REQUIRE(device->createTextureView(*texture, {}, first));
        REG_REQUIRE(device->createTextureView(*texture, {.format = render::Format::Rgba8Unorm}, second));
        render::ResourceLease imageA, imageB, generalImage, storageImage;
        REG_REQUIRE(registry.sampledImage(*first, imageA));
        REG_REQUIRE(registry.sampledImage(*second, imageB));
        REG_CHECK(imageA.shaderValue() == imageB.shaderValue());
        REG_REQUIRE(registry.sampledImage(*second, generalImage, render::ResourceState::General));
        REG_CHECK(generalImage.shaderValue() != imageA.shaderValue());
        REG_REQUIRE(registry.storageImage(*second, storageImage));
        REG_CHECK(storageImage.kind() == render::ShaderResourceKind::StorageImage);
        std::weak_ptr<void> textureAllocation = first->retainTexture();
        texture.reset(); first.reset(); second.reset();
        REG_CHECK(!textureAllocation.expired());
        imageA = {}; imageB = {}; generalImage = {}; storageImage = {};
        REG_CHECK(textureAllocation.expired());
        registry.collect();
        REG_CHECK(registry.stats().liveDescriptors == 2);
        render::ResourceLease samplerA, samplerB;
        REG_REQUIRE(registry.sampler({}, samplerA));
        REG_REQUIRE(registry.sampler({}, samplerB));
        REG_CHECK(samplerA.shaderValue() == samplerB.shaderValue());

        render::ResourceRegistry other;
        REG_REQUIRE(other.initialize(*device, {.maxSamplers = 1, .maxSampledImages = 1,
            .maxStorageImages = 1, .maxBuffers = 1}));
        render::RenderFrameContext frame;
        REG_REQUIRE(frame.begin(0));
        render::ParameterWriter writer(*device, frame, other);
        REG_CHECK(render::hasError(writer.use(bLease), render::Error::InvalidArgument));
        render::EncodedParameters invalid;
        REG_CHECK(!writer.encode(ProbeParams{}, kAbi, invalid) && !invalid.valid());
        render::ParameterWriter staleWriter(*device, frame, registry);
        REG_REQUIRE(staleWriter.encode(ProbeParams{}, kAbi, invalid));
        frame.cancel();
        REG_REQUIRE(frame.begin(1));
        REG_CHECK(render::hasError(staleWriter.encode(ProbeParams{}, kAbi, invalid), render::Error::InvalidArgument));
        REG_CHECK(!invalid.valid());
        frame.cancel();
        return RhiTestResult::pass();
    }
};

class RegistrySubmissionTest final : public RhiTest {
public:
    RegistrySubmissionTest() { type = RhiTestType::Command; name = "registry_typed_submission_lifetime"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        REG_REQUIRE(render::createDevice({.applicationName = "Registry lifetime", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}, device));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry, sameRegistry;
        REG_REQUIRE(device->resourceRegistry(registry));
        REG_REQUIRE(device->resourceRegistry(sameRegistry));
        REG_CHECK(registry == sameRegistry);
        render::ComputeKernel firstKernel, secondKernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, firstKernel, log));
        REG_REQUIRE(makeKernel(*device, secondKernel, log));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 11));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> oldAllocation = source->retainAllocation();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands first, second;
        REG_REQUIRE(first.initialize(*device, queue));
        REG_REQUIRE(second.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore(gate));
        Drain drain{queue, *gate};
        REG_REQUIRE(first.begin(0));
        render::EncodedParameters stale;
        {
            render::ParameterWriter writer(*device, first.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 100, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kAbi, encoded));
            stale = encoded;
            REG_REQUIRE(firstKernel.dispatch(*first.commands, encoded, 1));
            params.add = 200; params.index = 1;
            REG_REQUIRE(writer.encode(params, kAbi, encoded));
            render::BufferBarrierDesc barrier{.buffer = output.get(),
                .before = render::ResourceState::General, .after = render::ResourceState::General};
            first.commands->barrier({.buffers = &barrier, .bufferCount = 1});
            REG_REQUIRE(secondKernel.dispatch(*first.commands, encoded, 1));
            // Force the parameter arena to grow without moving already encoded roots.
            std::array<uint32_t, 17000> burst{};
            render::EncodedParameters oversized;
            REG_REQUIRE(writer.encode(burst, kAbi + 2, oversized));
            REG_CHECK(oversized.address() != stale.address());
            params.add = 999; // Encoded packets must not reference this mutable CPU struct.
            render::EncodedParameters wrong;
            REG_REQUIRE(writer.encode(params, kAbi + 1, wrong));
            REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, wrong, 1), render::Error::InvalidArgument));
        }
        source.reset();
        REG_REQUIRE(makeBuffer(*device, source, 22));
        REG_REQUIRE(first.submit(tracker, *gate));
        REG_CHECK(!first.frame.completion().isComplete());
        REG_CHECK(!oldAllocation.expired());
        REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, stale, 1), render::Error::InvalidArgument));
        REG_REQUIRE(second.begin(1));
        REG_CHECK(render::hasError(secondKernel.dispatch(*second.commands, stale, 1), render::Error::InvalidArgument));
        {
            render::ParameterWriter writer(*device, second.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 300, 2};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kAbi, encoded));
            REG_CHECK(encoded.address() != stale.address());
            REG_REQUIRE(secondKernel.dispatch(*second.commands, encoded, 1));
        }
        REG_REQUIRE(second.submit(tracker, *gate));
        firstKernel.clear(); secondKernel.clear(); stale = {};
        REG_CHECK(!oldAllocation.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(second.frame.wait(5'000'000'000ull));
        output->invalidate();
        std::array<uint32_t, 3> values{};
        void* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::memcpy(values.data(), mapped, sizeof(values));
        output->unmap();
        REG_CHECK((values == std::array<uint32_t, 3>{111, 211, 322}));
        REG_REQUIRE(first.pool->reset());
        REG_REQUIRE(first.frame.reset());
        REG_CHECK(oldAllocation.expired());
        REG_REQUIRE(second.pool->reset());
        REG_REQUIRE(second.frame.reset());
        registry->collect();
        REG_CHECK(registry->stats().liveDescriptors == 2);
        const uint64_t capacity = registry->stats().parameterCapacity;

        REG_REQUIRE(makeKernel(*device, firstKernel, log));
        REG_REQUIRE(first.begin(2));
        std::weak_ptr<void> cancelledAllocation = source->retainAllocation();
        render::EncodedParameters cancelled;
        {
            render::ParameterWriter writer(*device, first.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 1, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kAbi, encoded));
            REG_REQUIRE(firstKernel.dispatch(*first.commands, encoded, 1));
            cancelled = encoded;
        }
        source.reset();
        REG_CHECK(!cancelledAllocation.expired());
        REG_REQUIRE(first.commands->end());
        // A frame can still be recording while this command buffer has ended.
        REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, cancelled, 1), render::Error::InvalidArgument));
        cancelled = {};
        REG_REQUIRE(first.pool->reset());
        first.frame.cancel();
        REG_CHECK(cancelledAllocation.expired());
        REG_CHECK(registry->stats().parameterCapacity == capacity);
        registry->collect();
        REG_CHECK(registry->stats().liveDescriptors == 1);
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(RegistryIdentityTest);
METALLIC_REGISTER_RHI_TEST(RegistrySubmissionTest);

class RegistryPartialSubmissionTest final : public RhiTest {
public:
    RegistryPartialSubmissionTest() { type = RhiTestType::Command; name = "registry_partial_multi_queue_retention"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        REG_REQUIRE(render::createDevice({.applicationName = "Registry partial submission",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device));
        auto& graphics = *device->getQueue(render::QueueType::Graphics);
        auto* copy = device->getQueue(render::QueueType::Copy);
        if (!copy) { return RhiTestResult::skip("Requires a copy queue"); }
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry(registry));
        render::ComputeKernel kernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, kernel, log));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 17));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> allocation = source->retainAllocation();
        auto owner = std::make_shared<uint32_t>(19);
        std::weak_ptr<void> transitiveOwner = owner;
        render::QueueSubmissionTracker graphicsTracker, copyTracker;
        REG_REQUIRE(graphicsTracker.initialize(*device, graphics));
        REG_REQUIRE(copyTracker.initialize(*device, *copy));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, graphics));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore(gate));
        Drain drain{*copy, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::ParameterWriter writer(*device, recording.frame, *registry);
            writer.retain(owner);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 1, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kAbi, encoded));
            REG_REQUIRE(kernel.dispatch(*recording.commands, encoded, 1));
        }
        REG_REQUIRE(recording.commands->end());
        source.reset(); owner.reset(); kernel.clear();
        render::CommandBuffer* buffers[] = {recording.commands.get()};
        render::GpuCompletionPoint graphicsDone, copyDone, rejected;
        REG_REQUIRE(graphicsTracker.submitSegment({.commandBuffers = buffers, .commandBufferCount = 1},
            recording.frame, graphicsDone));
        render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        REG_REQUIRE(copyTracker.submitSegment({.waitSemaphores = &wait, .waitSemaphoreCount = 1},
            recording.frame, copyDone));
        REG_CHECK(!copyTracker.submitSegment({.commandBufferCount = 1}, recording.frame, rejected));
        recording.frame.cancel(); // Must seal accepted segments, not release their packets.
        REG_REQUIRE(graphicsDone.wait(5'000'000'000ull));
        registry->collect();
        REG_CHECK(!recording.frame.completion().isComplete() && !copyDone.isComplete());
        REG_CHECK(!allocation.expired() && !transitiveOwner.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(allocation.expired() && transitiveOwner.expired());
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryPartialSubmissionTest);

class RegistryTextureSubmissionTest final : public RhiTest {
public:
    RegistryTextureSubmissionTest() { type = RhiTestType::Rendering; name = "registry_texture_array_submission_lifetime"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        REG_REQUIRE(render::createDevice({.applicationName = "Registry texture array",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry(registry));
        struct Params { render::ShaderStorageImage image; uint64_t samples; render::ShaderBuffer output; };
        static_assert(sizeof(Params) == 24);
        const char* entries[] = {"registryTextureWriteMain", "registryTextureReadMain"};
        std::array<render::ComputeKernel, 2> kernels;
        std::string log;
        for (size_t i = 0; i < kernels.size(); ++i) {
            render::ShaderCompileResult shader;
            REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "RegistryTextureProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader));
            REG_REQUIRE(kernels[i].initialize(*device, {.spirv = shader.spirv,
                .parameters = render::parameterAbi<Params>(kAbi + 3)}, log));
        }
        std::unique_ptr<render::Texture> image;
        std::unique_ptr<render::TextureView> view;
        std::unique_ptr<render::Buffer> output;
        REG_REQUIRE(makeBuffer(*device, output));
        REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage,
            .format = render::Format::R32Uint}, image));
        REG_REQUIRE(device->createTextureView(*image, {}, view));
        std::weak_ptr<void> allocation = view->retainTexture();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore(gate));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::ParameterWriter writer(*device, recording.frame, *registry);
            const std::array<render::TextureView*, 3> views{view.get(), view.get(), view.get()};
            Params params{writer.storageImage(view.get()), writer.sampledImages(views), writer.buffer(output.get())};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kAbi + 3, encoded));
            REG_CHECK(registry->stats().descriptorWrites == 3); // storage image, sampled image, output
            render::TextureBarrierDesc barrier{.texture = image.get(),
                .before = render::ResourceState::Undefined, .after = render::ResourceState::General};
            recording.commands->barrier({.textures = &barrier, .textureCount = 1});
            REG_REQUIRE(kernels[0].dispatch(*recording.commands, encoded, 1));
            barrier.before = render::ResourceState::General; barrier.after = render::ResourceState::ShaderRead;
            recording.commands->barrier({.textures = &barrier, .textureCount = 1});
            REG_REQUIRE(kernels[1].dispatch(*recording.commands, encoded, 1));
        }
        image.reset(); view.reset();
        REG_CHECK(!allocation.expired());
        REG_REQUIRE(recording.submit(tracker, *gate));
        kernels = {};
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        void* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::array<uint32_t, 3> values;
        std::memcpy(values.data(), mapped, sizeof(values));
        output->unmap();
        REG_CHECK((values == std::array<uint32_t, 3>{1001, 1001, 1001}));
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(allocation.expired());
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryTextureSubmissionTest);
#undef REG_REQUIRE
#undef REG_CHECK
} // namespace
} // namespace metallic::tests
