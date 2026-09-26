#include "RhiTest.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/ComputeKernel.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <cstring>
#include <thread>

namespace metallic::tests {
namespace {

#define REG_REQUIRE(expression) do { \
    const render::Result<> result = (expression); \
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

render::Result<> makeBuffer(render::Device& device, std::unique_ptr<render::Buffer>& buffer, uint32_t value = 0)
{
    auto result = device.createBuffer({.size = 64, .structureStride = 4,
        .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Indirect,
        .memoryLocation = render::MemoryLocation::HostReadback,
        .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
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
    render::Result<> initialize(render::Device& device, render::Queue& queue)
    {
        auto result = device.createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); });
        return result ? pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) : result;
    }
    render::Result<> begin(uint64_t index)
    {
        auto result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? commands->begin(&frame) : result;
    }
    render::Result<> submit(render::QueueSubmissionTracker& tracker, render::Semaphore& gate)
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

render::Result<> makeKernel(render::Device& device, render::ComputeKernel& kernel, std::string& log,
    render::SlangDescriptorHeapMode mode = render::SlangDescriptorHeapMode::Default)
{
    render::ShaderCompileResult shader;
    auto result = render::compileSlangShaderToSpirv({.moduleName = "RegistryProbe",
        .entryPointName = "registryProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
        .descriptorHeapMode = mode}, shader);
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
            .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
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
            .format = render::Format::Rgba8Unorm}).transform([&](auto rhiValue) { texture = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*texture, {}).transform([&](auto rhiValue) { first = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*texture, {.format = render::Format::Rgba8Unorm}).transform([&](auto rhiValue) { second = std::move(rhiValue); }));
        render::ResourceLease imageA, imageB, generalImage, storageImage;
        REG_REQUIRE(registry.sampledImage(*first, imageA));
        REG_REQUIRE(registry.sampledImage(*second, imageB));
        REG_CHECK(imageA.shaderValue() == imageB.shaderValue());
        REG_REQUIRE(registry.sampledImage(*second, generalImage, render::ResourceState::General));
        REG_CHECK(generalImage.shaderValue() != imageA.shaderValue());
        REG_REQUIRE(registry.storageImage(*second, storageImage));
        REG_CHECK(storageImage.kind() == render::ShaderResourceKind::StorageImage);
        REG_CHECK(!first->hasNativeView() && !second->hasNativeView());
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
            .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry, sameRegistry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { sameRegistry = std::move(rhiValue); }));
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
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
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
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& graphics = *device->getQueue(render::QueueType::Graphics);
        auto* copy = device->getQueue(render::QueueType::Copy);
        if (!copy) { return RhiTestResult::skip("Requires a copy queue"); }
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
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
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
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
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
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
            .format = render::Format::R32Uint}).transform([&](auto rhiValue) { image = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*image, {}).transform([&](auto rhiValue) { view = std::move(rhiValue); }));
        std::weak_ptr<void> allocation = view->retainTexture();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
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
// Native provenance and narrowing are checked without creating descriptors.
class BufferSliceValidationTest final : public RhiTest {
public:
    BufferSliceValidationTest() { type = RhiTestType::Resource; name = "buffer_slice_range_and_provenance"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device, other;
        REG_REQUIRE(render::createDevice({.applicationName = "Buffer slice ranges",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        REG_REQUIRE(render::createDevice({.applicationName = "Buffer slice foreign source",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { other = std::move(rhiValue); }));
        std::unique_ptr<render::Buffer> buffer;
        REG_REQUIRE(makeBuffer(*device, buffer));
        render::BufferSlice parent, child, invalid, empty;
        REG_REQUIRE(buffer->slice(16, 32).transform([&](auto rhiValue) { parent = std::move(rhiValue); }));
        REG_REQUIRE(parent.subslice(8, 8).transform([&](auto rhiValue) { child = std::move(rhiValue); }));
        REG_CHECK(child.offset() == 24 && child.size() == 8);
        REG_CHECK(child.deviceAddress() == buffer->deviceAddress() + 24);
        REG_CHECK(child.deviceIdentity() == device->identity());
        REG_CHECK(child.allocationIdentity() == buffer->retainAllocation().get());
        REG_REQUIRE(child.validateData(device->identity(), 4, 4));
        REG_CHECK(render::hasError(child.validateData(other->identity(), 4, 4), render::Error::InvalidArgument));
        REG_REQUIRE(parent.subslice(32).transform([&](auto value) { empty = std::move(value); }));
        REG_CHECK(empty.valid() && empty.size() == 0);
        REG_CHECK(!empty.validateData(device->identity(), 4, 4));
        for (uint64_t offset : {uint64_t(33), UINT64_MAX}) {
            const auto rejected = parent.subslice(offset);
            REG_CHECK(render::hasError(rejected, render::Error::InvalidArgument));
            REG_CHECK(parent.offset() == 16 && parent.size() == 32);
        }
        REG_CHECK(render::hasError(parent.subslice(0, 33), render::Error::InvalidArgument));
        REG_CHECK(render::hasError(parent.subslice(31, UINT64_MAX - 1), render::Error::InvalidArgument));
        REG_REQUIRE(parent.subslice(1, 8).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
        REG_CHECK(!invalid.validateData(device->identity(), 4, 4));
        REG_REQUIRE(parent.subslice(0, 12).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
        REG_CHECK(!invalid.validateData(device->identity(), 8, 4));
        REG_CHECK(!child.validateData(device->identity(), 0, 4));
        REG_CHECK(!child.validateData(device->identity(), 4, 0));
        REG_CHECK(!child.validateData(device->identity(), 4, 3));
        REG_CHECK(!child.validateData(device->identity(), 3, 4));
        REG_CHECK(!child.validate(device->identity(), render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource));
        REG_REQUIRE(parent.subslice(8, 8).transform([&](auto rhiValue) { parent = std::move(rhiValue); }));
        REG_CHECK(parent.offset() == child.offset() && parent.size() == child.size());

        std::weak_ptr<void> allocation = buffer->retainAllocation();
        const auto address = child.deviceAddress();
        render::Buffer moved = std::move(*buffer);
        REG_CHECK(!buffer->retainAllocation());
        REG_REQUIRE(makeBuffer(*device, buffer, 99));
        moved = {};
        REG_CHECK(!allocation.expired() && child.deviceAddress() == address);
        parent = {}; invalid = {}; empty = {};
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        render::RenderFrameContext frame;
        REG_REQUIRE(frame.begin(0));
        render::EncodedParameters packet;
        {
            render::ParameterWriter invalidWriter(*other, frame, *registry);
            invalidWriter.dataBuffer<uint32_t>(child);
            REG_CHECK(!invalidWriter.status());
            REG_CHECK(!invalidWriter.encode(render::ShaderDataSpan{}, kAbi + 4, packet) && !packet.valid());
            render::ParameterWriter writer(*device, frame, *registry);
            const auto data = writer.dataBuffer<uint32_t>(child);
            REG_CHECK(data.address == address && data.count == 2 && data.stride == 4);
            REG_REQUIRE(writer.encode(data, kAbi + 4, packet));
        }
        child = {};
        REG_CHECK(!allocation.expired());
        packet = {};
        frame.cancel();
        REG_REQUIRE(frame.reset());
        REG_CHECK(allocation.expired());
        REG_CHECK(registry->stats().descriptorWrites == 0);
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(BufferSliceValidationTest);

class BufferSliceSubmissionTest final : public RhiTest {
public:
    BufferSliceSubmissionTest() { type = RhiTestType::Rendering; name = "buffer_slice_bda_copy_compute_indirect_lifetime"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        REG_REQUIRE(render::createDevice({.applicationName = "Buffer slice data chain",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        struct Params { render::ShaderDataSpan source, output, arguments; uint32_t add; };
        static_assert(sizeof(Params) == 56 && offsetof(Params, add) == 48);
        std::array<render::ComputeKernel, 2> kernels;
        const char* entries[] = {"dataProduceMain", "dataIndirectMain"};
        std::string log;
        for (size_t i = 0; i < kernels.size(); ++i) {
            render::ShaderCompileResult shader;
            auto result = render::compileSlangShaderToSpirv({.moduleName = "DataSliceProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
            if (!result) { return RhiTestResult::fail(shader.diagnostics); }
            REG_REQUIRE(kernels[i].initialize(*device, {.spirv = shader.spirv,
                .parameters = render::parameterAbi<Params>(kAbi + 5)}, log));
        }
        render::ShaderCompileResult shader;
        REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "DataSliceProbe",
            .entryPointName = "dataAdapterMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader));
        render::ComputeProgram adapter;
        const render::ComputeProgramBindingDesc layout{.binding = 0,
            .kind = render::ComputeResourceBindingKind::DataBuffer, .dataStride = 4, .dataAlignment = 4};
        REG_REQUIRE(adapter.initialize(*device, {.spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
            .bindings = &layout, .bindingCount = 1, .requiresRayQuery = false}, log));
        std::unique_ptr<render::Buffer> source, work, output;
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::TransferSource,
            .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto rhiValue) { source = std::move(rhiValue); }));
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::ShaderDeviceAddress |
            render::BufferUsageBits::TransferDestination | render::BufferUsageBits::Indirect}).transform([&](auto rhiValue) { work = std::move(rhiValue); }));
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::ShaderDeviceAddress |
            render::BufferUsageBits::TransferSource | render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        auto* sourceWords = static_cast<uint32_t*>(source->map());
        REG_CHECK(sourceWords);
        for (uint32_t i = 0; i < 16; ++i) { sourceWords[i] = 100 + i; }
        source->flush(); source->unmap();
        auto* outputWords = static_cast<uint32_t*>(output->map());
        REG_CHECK(outputWords);
        for (uint32_t i = 0; i < 16; ++i) { outputWords[i] = 0xdeadbeef; }
        output->flush(); output->unmap();
        std::weak_ptr<void> sourceAllocation = source->retainAllocation(), workAllocation = work->retainAllocation();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::BufferSlice from, data, to, arguments, invalid;
            REG_REQUIRE(source->slice(8, 16).transform([&](auto rhiValue) { from = std::move(rhiValue); }));
            REG_REQUIRE(work->slice(16, 16).transform([&](auto rhiValue) { data = std::move(rhiValue); }));
            REG_REQUIRE(work->slice(48, 12).transform([&](auto rhiValue) { arguments = std::move(rhiValue); }));
            REG_REQUIRE(output->slice(20, 16).transform([&](auto rhiValue) { to = std::move(rhiValue); }));
            // Transfer-only memory is not a shader data buffer.
            REG_CHECK(!from.validateData(device->identity(), 4, 4));
            REG_REQUIRE(to.subslice(0, 12).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
            REG_CHECK(!recording.commands->copyBuffer(from, invalid));
            REG_CHECK(!recording.commands->copyBuffer(to, to));
            REG_CHECK(!recording.commands->dispatchIndirect(to));
            REG_REQUIRE(arguments.subslice(1, 8).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
            REG_CHECK(!recording.commands->dispatchIndirect(invalid));
            render::BufferBarrierDesc workBarrier{.buffer = work.get(),
                .before = render::ResourceState::Undefined, .after = render::ResourceState::TransferDestination};
            recording.commands->barrier({.buffers = &workBarrier, .bufferCount = 1});
            REG_REQUIRE(recording.commands->copyBuffer(from, data));
            workBarrier.before = render::ResourceState::TransferDestination; workBarrier.after = render::ResourceState::General;
            recording.commands->barrier({.buffers = &workBarrier, .bufferCount = 1});
            render::BufferBarrierDesc outputBarrier{.buffer = output.get(),
                .before = render::ResourceState::Undefined, .after = render::ResourceState::General};
            recording.commands->barrier({.buffers = &outputBarrier, .bufferCount = 1});
            render::ParameterWriter writer(*device, recording.frame, *registry);
            const Params params{writer.dataBuffer<uint32_t>(data), writer.dataBuffer<uint32_t>(to),
                writer.dataBuffer<uint32_t>(arguments), 7};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kAbi + 5, encoded));
            REG_REQUIRE(kernels[0].dispatch(*recording.commands, encoded, 1));
            outputBarrier.before = render::ResourceState::General;
            workBarrier.before = render::ResourceState::General; workBarrier.after = render::ResourceState::IndirectArgument;
            const render::BufferBarrierDesc barriers[] = {outputBarrier, workBarrier};
            recording.commands->barrier({.buffers = barriers, .bufferCount = 2});
            REG_REQUIRE(kernels[1].dispatchIndirect(*recording.commands, encoded, arguments));
            recording.commands->barrier({.buffers = &outputBarrier, .bufferCount = 1});
            render::ComputeDispatchBinding binding{.binding = 0, .data = to};
            render::ComputeDispatchDesc dispatch{.commandBuffer = recording.commands.get(), .bindings = &binding, .bindingCount = 1};
            binding.offset = 4;
            REG_CHECK(render::hasError(adapter.dispatch(dispatch), render::Error::InvalidArgument));
            binding.offset = 0;
            REG_REQUIRE(adapter.dispatch(dispatch));
        }
        source.reset(); work.reset();
        REG_CHECK(!sourceAllocation.expired() && !workAllocation.expired());
        REG_CHECK(registry->stats().descriptorWrites == 0 && registry->stats().liveDescriptors == 0);
        REG_REQUIRE(recording.submit(tracker, *gate));
        kernels = {}; adapter.clear();
        REG_CHECK(!recording.frame.completion().isComplete());
        REG_CHECK(!sourceAllocation.expired() && !workAllocation.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        outputWords = static_cast<uint32_t*>(output->map());
        REG_CHECK(outputWords);
        std::array<uint32_t, 16> values;
        std::memcpy(values.data(), outputWords, sizeof(values));
        output->unmap();
        for (uint32_t i = 0; i < values.size(); ++i) {
            const auto expected = i >= 5 && i < 9 ? 221 + (i - 5) * 2 : 0xdeadbeef;
            REG_CHECK(values[i] == expected);
        }
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(sourceAllocation.expired() && workAllocation.expired());
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(BufferSliceSubmissionTest);


class SynchronizationScopesTest final : public RhiTest {
public:
    SynchronizationScopesTest() { type = RhiTestType::Command; name = "synchronization_scopes_batch_and_validation"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using S = render::PipelineStageBits;
        using A = render::AccessBits;
        auto& device = context.device;
        Commands recording;
        REG_REQUIRE(recording.initialize(device, context.graphicsQueue));
        REG_REQUIRE(recording.begin(0));
        auto& command = *recording.commands;
        std::array<std::unique_ptr<render::Buffer>, 3> buffers;
        std::array<render::BufferBarrierDesc, 3> barriers;
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            REG_REQUIRE(makeBuffer(device, buffers[i]));
            barriers[i] = {.buffer = buffers[i].get(), .before = render::ResourceState::General,
                .after = render::ResourceState::ShaderRead,
                .beforeScope = {S::ComputeShader, A::ShaderWrite}, .afterScope = {S::ComputeShader, A::ShaderRead}};
        }
        REG_REQUIRE(command.synchronize({.buffers = barriers.data(), .bufferCount = 3}));
        auto stats = command.synchronizationStats();
        REG_CHECK(stats.calls == 1 && stats.memoryBarriers == 1 && stats.coalescedResources == 3 && stats.imageTransitions == 0);
        for (auto& barrier : barriers) { barrier.beforeScope.access = A::ShaderRead; }
        REG_REQUIRE(command.synchronize({.buffers = barriers.data(), .bufferCount = 3}));
        REG_CHECK(command.synchronizationStats().calls == 1); // Read/read does not order execution.
        std::array<render::MemoryBarrierDesc, 2> memory{{
            {{S::ComputeShader, A::ShaderWrite}, {S::DrawIndirect, A::IndirectRead}},
            {{S::Transfer, A::TransferWrite}, {S::ComputeShader, A::ShaderRead}},
        }};
        REG_REQUIRE(command.synchronize({.memory = memory.data(), .memoryCount = 2}));
        REG_CHECK(command.synchronizationStats().memoryBarriers == 3); // Keep distinct stage pairs.
        const std::array<render::SyncScope, 5> invalid{{
            {S::Transfer, A::ShaderWrite}, {S::ComputeShader, A::IndirectRead},
            {S::None, A::MemoryRead}, {static_cast<S>(1ull << 63), A::None}, {S::Transfer, static_cast<A>(1ull << 63)},
        }};
        for (const auto scope : invalid) {
            memory[1].after = scope;
            REG_CHECK(render::hasError(command.synchronize({.memory = memory.data(), .memoryCount = 2}), render::Error::InvalidArgument));
            REG_CHECK(command.synchronizationStats().calls == 2); // Validation is atomic.
        }
        barriers[0].offset = 64;
        REG_CHECK(render::hasError(command.synchronize({.buffers = barriers.data(), .bufferCount = 3}), render::Error::InvalidArgument));
        REG_REQUIRE(command.end());
        REG_CHECK(render::hasError(command.synchronize({}), render::Error::InvalidArgument));
        recording.frame.cancel();
        REG_REQUIRE(recording.begin(1));
        REG_CHECK(command.synchronizationStats().calls == 0);
        recording.frame.cancel();
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(SynchronizationScopesTest);

class PreparedExecutionViewsTest final : public RhiTest {
public:
    PreparedExecutionViewsTest() { type = RhiTestType::Rendering; name = "prepared_execution_lazy_views_layout_policy"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint32_t extent = 32, bytes = extent * extent * 4;
        std::array<uint8_t, bytes> reference{};
        bool unifiedTested = false;
        for (bool preferUnified : {false, true}) {
            std::atomic_uint errors{0};
            std::unique_ptr<render::Device> device;
            REG_REQUIRE(render::createDevice({.applicationName = "Prepared execution lifetime", .enableValidation = context.enableValidation,
                .validationSink = {[](void* target, const render::ValidationMessage& message) noexcept {
                    if (message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &errors}, .preferUnifiedImageLayouts = preferUnified}).transform([&](auto value) { device = std::move(value); }));
            const bool unified = device->capabilities().unifiedImageLayouts;
            REG_CHECK(preferUnified || !unified);
            unifiedTested |= unified;
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            std::array<render::ShaderCompileResult, 2> compiled;
            std::array<std::unique_ptr<render::ShaderModule>, 2> modules;
            const char* entries[] = {"triangleVertexMain", "triangleFragmentMain"};
            for (uint32_t i = 0; i < modules.size(); ++i) {
                REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "Features/Samples/Triangle",
                    .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled[i]));
                REG_REQUIRE(device->createShaderModule({.code = compiled[i].spirv.data(),
                    .byteSize = compiled[i].spirv.size() * sizeof(uint32_t)}).transform([&](auto value) { modules[i] = std::move(value); }));
            }
            std::unique_ptr<render::GraphicsPipeline> pipeline;
            REG_REQUIRE(device->createGraphicsPipeline({.vertexShader = modules[0].get(), .fragmentShader = modules[1].get(),
                .colorFormat = render::Format::Rgba8Unorm}).transform([&](auto value) { pipeline = std::move(value); }));
            std::unique_ptr<render::GraphicsShaderObjectProgram> program;
            REG_REQUIRE(device->createGraphicsShaderObjectProgram({.vertexCode = compiled[0].spirv.data(),
                .vertexByteSize = compiled[0].spirv.size() * sizeof(uint32_t), .fragmentCode = compiled[1].spirv.data(),
                .fragmentByteSize = compiled[1].spirv.size() * sizeof(uint32_t)}).transform([&](auto value) { program = std::move(value); }));
            std::array<render::PreparedExecution, 3> executions{pipeline->execution(), program->execution(), pipeline->execution()};
            auto invalidState = program->execution({.colorAttachmentCount = 9});
            // Snapshots survive hot replacement of all source objects before recording.
            pipeline.reset(); program.reset(); modules = {};
            render::QueueSubmissionTracker tracker;
            REG_REQUIRE(tracker.initialize(*device, queue));
            Commands recording;
            REG_REQUIRE(recording.initialize(*device, queue));
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(recording.begin(0));
            auto& command = *recording.commands;
            REG_CHECK(render::hasError(command.bindExecution({}), render::Error::InvalidArgument));
            REG_CHECK(render::hasError(command.bindExecution(invalidState), render::Error::InvalidArgument));
            invalidState = {};
            std::array<std::unique_ptr<render::Buffer>, 3> readbacks;
            std::array<std::weak_ptr<void>, 3> allocations;
            for (uint32_t i = 0; i < executions.size(); ++i) {
                std::unique_ptr<render::Texture> texture;
                REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                    .format = render::Format::Rgba8Unorm, .width = extent, .height = extent}).transform([&](auto value) { texture = std::move(value); }));
                std::unique_ptr<render::TextureView> view;
                REG_REQUIRE(device->createTextureView(*texture, {}).transform([&](auto value) { view = std::move(value); }));
                REG_CHECK(!view->hasNativeView());
                REG_CHECK(render::hasError(device->createTextureView(*texture, {.baseMip = 1}).transform([](auto) {}), render::Error::InvalidArgument));
                REG_CHECK(render::vulkan::nativeImageLayout(*view, render::ResourceState::ColorAttachment) ==
                    (unified ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
                REG_CHECK(!view->hasNativeView());
                allocations[i] = view->retainTexture();
                REG_REQUIRE(device->createBuffer({.size = bytes, .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto value) { readbacks[i] = std::move(value); }));
                render::TextureBarrierDesc barrier{.texture = texture.get(), .after = render::ResourceState::ColorAttachment};
                REG_REQUIRE(command.synchronize({.textures = &barrier, .textureCount = 1}));
                render::RenderingAttachmentDesc attachment{.view = view.get(), .state = render::ResourceState::ColorAttachment,
                    .loadOp = render::LoadOp::Clear, .clearColor = {0, 0, 0, 1}};
                REG_REQUIRE(command.beginRendering({.renderArea = {0, 0, extent, extent}, .colorAttachments = &attachment, .colorAttachmentCount = 1}));
                REG_CHECK(view->hasNativeView());
                const auto native = render::vulkan::nativeImageView(*view);
                REG_CHECK(native != VK_NULL_HANDLE && native == render::vulkan::nativeImageView(*view));
                // Simulate the SDK/DGC boundary, then explicitly establish the new state.
                if (i == 1) { render::vulkan::notifyExternalDescriptorSetBinding(command); }
                REG_REQUIRE(command.bindExecution(executions[i]));
                command.setViewport({0, 0, float(extent), float(extent), 0, 1});
                command.setScissor({0, 0, extent, extent});
                command.draw(3);
                command.endRendering();
                barrier.before = render::ResourceState::ColorAttachment;
                barrier.after = render::ResourceState::TransferSource;
                REG_REQUIRE(command.synchronize({.textures = &barrier, .textureCount = 1}));
                command.copyTextureToBuffer({.texture = texture.get(), .buffer = readbacks[i].get(), .width = extent, .height = extent});
                view.reset(); texture.reset();
                REG_CHECK(!allocations[i].expired());
            }
            const auto stats = command.synchronizationStats();
            REG_CHECK(stats.imageTransitions == (unified ? 3 : 6));
            REG_CHECK(stats.memoryBarriers == (unified ? 3 : 0));
            executions = {}; // Only recorded commands now own the native programs.
            REG_REQUIRE(recording.submit(tracker, *gate));
            REG_CHECK(!recording.frame.completion().isComplete());
            REG_REQUIRE(gate->signal(1));
            REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
            for (uint32_t i = 0; i < readbacks.size(); ++i) {
                readbacks[i]->invalidate();
                const auto* pixels = static_cast<const uint8_t*>(readbacks[i]->map());
                REG_CHECK(pixels && pixels[(extent / 2 * extent + extent / 2) * 4] > 0);
                if (!preferUnified && i == 0) { std::memcpy(reference.data(), pixels, bytes); }
                const bool same = std::memcmp(reference.data(), pixels, bytes) == 0;
                readbacks[i]->unmap();
                REG_CHECK(same);
            }
            REG_REQUIRE(recording.pool->reset());
            REG_REQUIRE(recording.frame.reset());
            for (const auto& allocation : allocations) { REG_CHECK(allocation.expired()); }
            // Cancellation also releases an exported native view without submitting it.
            REG_REQUIRE(recording.begin(1));
            std::weak_ptr<void> cancelled;
            {
                auto texture = device->createTexture({.usage = render::TextureUsageBits::ColorAttachment, .format = render::Format::Rgba8Unorm});
                REG_CHECK(texture);
                auto view = device->createTextureView(**texture, {});
                REG_CHECK(view);
                cancelled = (*view)->retainTexture();
                REG_REQUIRE(command.useNativeTextureView(**view));
            }
            REG_CHECK(!cancelled.expired());
            recording.frame.cancel();
            REG_REQUIRE(recording.pool->reset());
            REG_REQUIRE(recording.frame.reset());
            REG_CHECK(cancelled.expired());
            REG_CHECK(errors.load() == 0);
        }
        return RhiTestResult::pass(unifiedTested ? "GENERAL and optimal layouts produced identical PSO/shader-object readback" :
            "Optimal-layout fallback passed; unified image layouts unavailable on this device");
    }
};
METALLIC_REGISTER_RHI_TEST(PreparedExecutionViewsTest);

class ParallelRegistryTest final : public RhiTest {
public:
    ParallelRegistryTest() { type = RhiTestType::Command; name = "parallel_registry_packets"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            auto result = runMode(context, mode);
            if (!result.passed) { return result; }
        }
        return RhiTestResult::pass("Four concurrent writers sharing a registry and kernel in mapped/native modes");
    }
private:
    static RhiTestResult runMode(RhiTestContext& context, render::SlangDescriptorHeapMode mode)
    {
        std::unique_ptr<render::Device> device;
        REG_REQUIRE(render::createDevice({.applicationName = "Parallel registry", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto value) { registry = std::move(value); }));
        render::ComputeKernel kernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, kernel, log, mode));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 73));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> allocation = source->retainAllocation();
        render::RenderFrameContext frame;
        std::array<render::CommandRecordingContext, 4> contexts;
        std::array<render::CommandBuffer*, 4> commands{};
        std::array<render::Result<>, 4> results;
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        REG_REQUIRE(frame.begin(0));
        for (uint32_t i = 0; i < contexts.size(); ++i) {
            REG_REQUIRE(contexts[i].initialize(*device, queue));
            REG_REQUIRE(contexts[i].prepare(frame).transform([&](auto value) { commands[i] = value; }));
        }
        std::vector<std::jthread> workers;
        for (uint32_t i = 0; i < contexts.size(); ++i) {
            workers.emplace_back([&, i] {
                results[i] = contexts[i].record([&]() -> render::Result<> {
                    render::ParameterWriter writer(*device, frame, *registry);
                    ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), i, i};
                    render::EncodedParameters encoded;
                    auto result = writer.encode(params, kAbi, encoded);
                    if (result) { result = kernel.dispatch(*commands[i], encoded, 1); }
                    return result ? commands[i]->end() : result;
                });
            });
        }
        workers.clear(); // jthread joins every local resource/parameter writer.
        for (const auto& recorded : results) { REG_REQUIRE(recorded); }
        source.reset();
        kernel.clear();
        REG_CHECK(!allocation.expired());
        REG_REQUIRE(tracker.submit({.commandBuffers = commands.data(), .commandBufferCount = uint32_t(commands.size())}, frame));
        REG_REQUIRE(frame.wait(5'000'000'000ull));
        output->invalidate();
        auto* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::array<uint32_t, 4> actual{};
        std::memcpy(actual.data(), mapped, sizeof(actual));
        output->unmap();
        REG_CHECK((actual == std::array<uint32_t, 4>{73, 74, 75, 76}));
        for (auto& recording : contexts) { REG_REQUIRE(recording.reset()); }
        REG_REQUIRE(frame.reset());
        registry->collect();
        REG_CHECK(allocation.expired());
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(ParallelRegistryTest);

#undef REG_REQUIRE
#undef REG_CHECK
} // namespace
} // namespace metallic::tests
