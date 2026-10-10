#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Streamer/StreamUploadCompletion.h"

#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace metallic::tests {
namespace {

#define STREAM_REQUIRE(expression) do { \
    const auto result = (expression); \
    if (!result) { return RHITestResult::fail(std::string(#expression) + ": " + render::errorToString(result.error())); } \
} while (false)

// This benchmark also builds against the pre-batch implementation. Baseline
// transactions need an external lock around staging AND recording to preserve
// request ownership. GPU execution, setup and thread creation are not timed.
class StreamerThroughputTest final : public RHITest {
public:
    StreamerThroughputTest()
    {
        type = RHITestType::Resource;
        name = "streamer_cpu_throughput";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const char* mode = std::getenv("METALLIC_STREAMER_BENCHMARK");
        if (!mode) { return RHITestResult::skip("set METALLIC_STREAMER_BENCHMARK=baseline or batches"); }
        const bool batches = std::strcmp(mode, "batches") == 0;
        if (!batches && std::strcmp(mode, "baseline")) { return RHITestResult::fail("unknown benchmark mode"); }
#ifdef METALLIC_STREAMER_BENCHMARK_BASELINE
        if (batches) { return RHITestResult::skip("baseline build has no explicit batches"); }
#endif
        std::filesystem::create_directories(context.outputDirectory / name);
        std::ofstream csv(context.outputDirectory / name / "samples.csv");
        csv << "workload,bytes_per_request,threads,mode,sample,requests,elapsed_ms,requests_per_second\n";
        auto* queue = context.device.getQueue(render::QueueType::Graphics);
        if (!queue) { return RHITestResult::fail("missing graphics queue"); }
        std::atomic_bool failed = false;
        std::mutex externalMutex;
        for (bool record : {false, true}) {
            for (uint32_t bytes : {64u, 4096u}) {
                for (uint32_t count : {1u, 2u, 4u, 8u}) {
                    const uint32_t requestCount = bytes == 64 ? 32768 : 8192;
                    std::unique_ptr<render::Streamer> streamer;
                    STREAM_REQUIRE(render::createStreamer(context.device, {
                        .dynamicBufferSizePerFrame = uint64_t(requestCount) * count * bytes,
                        .queuedFrameCount = 1}).transform([&](auto value) { streamer = std::move(value); }));
                    std::unique_ptr<render::Buffer> destination;
                    STREAM_REQUIRE(context.device.createBuffer({.size = uint64_t(requestCount) * count * bytes,
                        .usage = render::BufferUsageBits::TransferDestination})
                        .transform([&](auto value) { destination = std::move(value); }));
                    std::vector<uint8_t> data(bytes, 0x5a);
                    const render::StreamDataChunk chunk{data.data(), bytes};
                    for (int sample = -3; sample < 7; ++sample) {
                        std::vector<std::unique_ptr<bench::GPUCommands>> commands;
                        for (uint32_t worker = 0; worker < count; ++worker) {
                            commands.push_back(std::make_unique<bench::GPUCommands>(*queue));
                            STREAM_REQUIRE(commands.back()->initialize(context.device));
                        }
                        std::barrier phase(count + 1);
                        std::vector<std::jthread> workers;
                        for (uint32_t worker = 0; worker < count; ++worker) {
                            workers.emplace_back([&, worker] {
                                phase.arrive_and_wait();
                                phase.arrive_and_wait();
                                for (uint32_t first = 0; first < requestCount; first += 16) {
                                    std::unique_lock lock(externalMutex, std::defer_lock);
                                    if (record && !batches) { lock.lock(); }
#ifndef METALLIC_STREAMER_BENCHMARK_BASELINE
                                    render::StreamerCopyBatch batch;
                                    if (record && batches) {
                                        auto created = streamer->beginCopyBatch();
                                        if (!created) { failed = true; continue; }
                                        batch = *created;
                                    }
#endif
                                    for (uint32_t item = first; item < first + 16; ++item) {
                                        if (!streamer->streamBufferData({.dataChunks = {&chunk, 1},
                                                .placementAlignment = 16,
                                                .dstBuffer = record ? destination.get() : nullptr,
                                                .dstOffset = (uint64_t(worker) * requestCount + item) * bytes,
#ifndef METALLIC_STREAMER_BENCHMARK_BASELINE
                                                .copyBatch = batch,
#endif
                                            }).valid()) { failed = true; }
                                    }
#ifdef METALLIC_STREAMER_BENCHMARK_BASELINE
                                    if (record && !streamer->copyStreamedData(*commands[worker]->commands)) { failed = true; }
#else
                                    if (record && !streamer->copyStreamedData(*commands[worker]->commands, batch)) { failed = true; }
#endif
                                }
                                phase.arrive_and_wait();
                            });
                        }
                        phase.arrive_and_wait();
                        const auto begin = std::chrono::steady_clock::now();
                        phase.arrive_and_wait();
                        phase.arrive_and_wait();
                        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                        workers.clear();
                        for (auto& command : commands) { STREAM_REQUIRE(command->commands->end()); }
                        streamer->endFrame();
                        if (sample >= 0) {
                            csv << (record ? "stage_record_16" : "stage_only") << ',' << bytes << ',' << count << ','
                                << mode << ',' << sample << ',' << requestCount * count << ',' << seconds * 1000 << ','
                                << requestCount * count / seconds << '\n';
                        }
                    }
                }
            }
        }
        csv.flush();
        return failed || !csv ? RHITestResult::fail("benchmark upload/record/write failed") :
            RHITestResult::pass("CPU API samples saved; three warmups, seven measured samples per case");
    }
};

METALLIC_REGISTER_RHI_TEST(StreamerThroughputTest);

#ifndef METALLIC_STREAMER_BENCHMARK_BASELINE
class StreamerConcurrentBatchesTest final : public RHITest {
public:
    StreamerConcurrentBatchesTest()
    {
        type = RHITestType::Command;
        name = "streamer_concurrent_batches_readback";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"streamer.concurrent.batches", "streamer.concurrent.growth"},
            bench::Layer::RHI, "core", "core", {"readback.bin"});
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        constexpr uint32_t kWorkers = 8, kBytes = 128 * 1024, kTextureBytes = 64;
        std::unique_ptr<Streamer> streamer;
        STREAM_REQUIRE(createStreamer(context.device, {.constantBufferSize = 4096,
            .dynamicBufferSizePerFrame = 65536, .queuedFrameCount = 1})
            .transform([&](auto value) { streamer = std::move(value); }));
        std::array<std::unique_ptr<Buffer>, kWorkers> outputs;
        std::array<std::unique_ptr<Texture>, kWorkers> textures;
        for (uint32_t i = 0; i < kWorkers; ++i) {
            STREAM_REQUIRE(context.device.createBuffer({.size = kBytes + kTextureBytes,
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
                .transform([&](auto value) { outputs[i] = std::move(value); }));
            STREAM_REQUIRE(context.device.createTexture({.usage = TextureUsageBits::TransferSource | TextureUsageBits::TransferDestination,
                .format = Format::RGBA8Unorm, .width = 4, .height = 4})
                .transform([&](auto value) { textures[i] = std::move(value); }));
        }
        RenderFrameContext frame;
        STREAM_REQUIRE(frame.begin(0));
        STREAM_REQUIRE(streamer->beginFrame(frame));
        std::array<CommandRecordingContext, kWorkers> lanes;
        std::array<CommandBuffer*, kWorkers> commands{};
        std::array<std::shared_ptr<StreamUploadCompletion>, kWorkers> completions;
        std::array<uint64_t, kWorkers> constants{};
        for (uint32_t i = 0; i < kWorkers; ++i) {
            STREAM_REQUIRE(lanes[i].initialize(context.device, context.graphicsQueue));
            STREAM_REQUIRE(lanes[i].prepare(frame).transform([&](auto value) { commands[i] = value; }));
        }
        QueueSubmissionTracker tracker;
        STREAM_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        std::unique_ptr<Semaphore> gate;
        STREAM_REQUIRE(context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        struct Drain {
            Queue& queue;
            Semaphore& gate;
            ~Drain() { if (gate.currentValue() < 1) { (void)gate.signal(1); } (void)queue.waitIdle(); }
        } drain{context.graphicsQueue, *gate};
        std::atomic_bool failed = false;
        std::barrier staged(kWorkers);
        std::mutex rendezvousMutex;
        std::condition_variable rendezvous;
        uint32_t recordingCount = 0;
        std::vector<std::jthread> workers;
        for (uint32_t worker = 0; worker < kWorkers; ++worker) {
            workers.emplace_back([&, worker] {
                std::vector<uint8_t> data(kBytes, uint8_t(worker + 1));
                auto batch = streamer->beginCopyBatch();
                if (!batch) { failed = true; }
                if (batch) {
                    for (uint32_t offset = 0; offset < kBytes; offset += 8192) {
                        StreamDataChunk chunk{data.data() + offset, 8192};
                        if (!streamer->streamBufferData({.dataChunks = {&chunk, 1}, .placementAlignment = 16,
                                .dstBuffer = outputs[worker].get(), .dstOffset = offset, .copyBatch = *batch}).valid()) { failed = true; }
                    }
                    if (!streamer->streamTextureData({.data = data.data(), .dstTexture = textures[worker].get(),
                            .width = 4, .height = 4, .copyBatch = *batch}).valid()) { failed = true; }
                    const uint32_t constant = worker + 17;
                    constants[worker] = streamer->streamConstantData(&constant, sizeof(constant));
                    completions[worker] = streamer->pendingCopyCompletion(*batch);
                    if (!completions[worker] || streamer->pendingCopyStats(*batch).copyCount() != 17) { failed = true; }
                }
                staged.arrive_and_wait();
                if (!batch) { return; }
                const auto result = lanes[worker].record([&]() -> Result<> {
                    auto& command = *commands[worker];
                    TextureBarrierDesc textureBarrier{.texture = textures[worker].get(),
                        .oldLayout = TextureLayout::Undefined, .newLayout = TextureLayout::TransferDestination,
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
                    if (auto result = command.synchronize({.textures = {&textureBarrier, 1}}); !result) { return result; }
                    auto result = streamer->copyStreamedData(command, *batch, [&](const char*) {
                        // A timeout reports accidental global recording serialization
                        // without hanging the test. Querying/staging here also tests reentry.
                        (void)streamer->stats();
                        if (worker == 0) {
                            std::vector<uint8_t> growth(2 * 1024 * 1024, 0xab);
                            StreamDataChunk chunk{growth.data(), growth.size()};
                            if (!streamer->streamBufferData({.dataChunks = {&chunk, 1}}).valid()) { failed = true; }
                        }
                        std::unique_lock lock(rendezvousMutex);
                        ++recordingCount;
                        rendezvous.notify_all();
                        if (!rendezvous.wait_for(lock, std::chrono::seconds(5), [&] { return recordingCount == kWorkers; })) { failed = true; }
                    });
                    if (!result) { return result; }
                    if (!completions[worker]->isRecordedBefore(command)) { failed = true; }
                    textureBarrier.oldLayout = TextureLayout::TransferDestination;
                    textureBarrier.newLayout = TextureLayout::TransferSource;
                    textureBarrier.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite};
                    textureBarrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead};
                    if (auto result = command.synchronize({.textures = {&textureBarrier, 1}}); !result) { return result; }
                    auto target = outputs[worker]->slice({kBytes, kTextureBytes});
                    if (!target) { return std::unexpected(target.error()); }
                    if (auto result = command.copyTextureToBuffer({.texture = textures[worker].get(), .buffer = *target,
                            .width = 4, .height = 4, .depth = 1}); !result) { return result; }
                    BufferBarrierDesc host{.buffer = outputs[worker].get(),
                        .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
                    if (auto result = command.synchronize({.buffers = {&host, 1}}); !result) { return result; }
                    return command.end();
                });
                if (!result) { failed = true; }
            });
        }
        workers.clear();
        if (failed || streamer->stats().pendingCopies.copyCount()) { return RHITestResult::fail("concurrent staging/recording isolation failed"); }
        auto* constantData = static_cast<const uint8_t*>(streamer->constantBuffer()->map());
        if (!constantData) { return RHITestResult::fail("constant map failed"); }
        for (uint32_t i = 0; i < kWorkers; ++i) {
            uint32_t value = 0;
            if (constants[i] <= 4096 - sizeof(value)) { std::memcpy(&value, constantData + constants[i], sizeof(value)); }
            if (value != i + 17) { failed = true; }
        }
        streamer->constantBuffer()->unmap();
        streamer->endFrame();
        SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        STREAM_REQUIRE(tracker.submit({.waitSemaphores = {&wait, 1}, .commandBuffers = commands}, frame));
        for (const auto& completion : completions) { if (completion->isComplete()) { failed = true; } }
        RenderFrameContext conflicting;
        STREAM_REQUIRE(conflicting.begin(1));
        if (streamer->beginFrame(conflicting)) { failed = true; }
        conflicting.cancel();
        streamer.reset(); // Growth arenas must survive in-flight GPU reads.
        STREAM_REQUIRE(gate->signal(1));
        STREAM_REQUIRE(frame.wait(5'000'000'000ull));
        for (uint32_t i = 0; i < kWorkers; ++i) {
            if (!completions[i]->isComplete() || completions[i]->isCancelled()) { failed = true; }
            const auto* data = static_cast<const uint8_t*>(outputs[i]->map());
            if (!data) { return RHITestResult::fail("readback map failed"); }
            outputs[i]->invalidate();
            bench::readbackEvidence(context, "readback.bin", std::span<const uint8_t>(data, kBytes + kTextureBytes));
            for (uint32_t offset = 0; offset < kBytes + kTextureBytes; ++offset) { if (data[offset] != i + 1) { failed = true; break; } }
            outputs[i]->unmap();
        }
        return failed ? RHITestResult::fail("constant uniqueness, receipt, lifetime or GPU bytes mismatch") :
            RHITestResult::pass("8 concurrent buffer/texture batches; callback overlap; growth, constants and gated GPU lifetime verified");
    }
};

class StreamerBatchIsolationTest final : public RHITest {
public:
    StreamerBatchIsolationTest()
    {
        type = RHITestType::Command;
        name = "streamer_batch_isolation_cancellation";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Streamer> streamer, foreign;
        STREAM_REQUIRE(createStreamer(context.device, {}).transform([&](auto value) { streamer = std::move(value); }));
        STREAM_REQUIRE(createStreamer(context.device, {}).transform([&](auto value) { foreign = std::move(value); }));
        std::unique_ptr<Buffer> destination;
        STREAM_REQUIRE(context.device.createBuffer({.size = 256, .usage = BufferUsageBits::TransferDestination})
            .transform([&](auto value) { destination = std::move(value); }));
        RenderFrameContext frame, otherFrame;
        STREAM_REQUIRE(frame.begin(0));
        STREAM_REQUIRE(otherFrame.begin(1));
        STREAM_REQUIRE(streamer->beginFrame(frame));
        CommandRecordingContext lane, otherLane;
        STREAM_REQUIRE(lane.initialize(context.device, context.graphicsQueue));
        STREAM_REQUIRE(otherLane.initialize(context.device, context.graphicsQueue));
        auto command = lane.prepare(frame), wrongCommand = otherLane.prepare(otherFrame);
        STREAM_REQUIRE(command);
        STREAM_REQUIRE(wrongCommand);
        const uint32_t data = 42;
        const StreamDataChunk chunk{&data, sizeof(data)};
        auto stage = [&](StreamerCopyBatch batch, uint64_t offset) {
            return streamer->streamBufferData({.dataChunks = {&chunk, 1}, .dstBuffer = destination.get(), .dstOffset = offset, .copyBatch = batch}).valid();
        };
        auto first = streamer->beginCopyBatch(), cancelled = streamer->beginCopyBatch(), wrong = streamer->beginCopyBatch();
        STREAM_REQUIRE(first); STREAM_REQUIRE(cancelled); STREAM_REQUIRE(wrong);
        if (!stage(*first, 0) || !stage(*cancelled, 4) || !stage(*wrong, 8) || !stage({}, 12)) { return RHITestResult::fail("stage failed"); }
        auto firstReceipt = streamer->pendingCopyCompletion(*first);
        auto cancelledReceipt = streamer->pendingCopyCompletion(*cancelled);
        auto wrongReceipt = streamer->pendingCopyCompletion(*wrong);
        auto defaultReceipt = streamer->pendingCopyCompletion();
        if (!firstReceipt || !cancelledReceipt || !wrongReceipt || !defaultReceipt) { return RHITestResult::fail("missing receipts"); }
        if (foreign->cancelCopyBatch(*first) || foreign->copyStreamedData(**command, *first)) { return RHITestResult::fail("foreign batch accepted"); }
        STREAM_REQUIRE(streamer->cancelCopyBatch(*cancelled));
        if (!cancelledReceipt->isCancelled() || firstReceipt->isCancelled() || stage(*cancelled, 16) ||
            streamer->cancelCopyBatch(*cancelled) || streamer->copyStreamedData(**command, *cancelled)) { return RHITestResult::fail("cancel isolation failed"); }
        if (streamer->copyStreamedData(**wrongCommand, *wrong) || !wrongReceipt->isCancelled()) { return RHITestResult::fail("wrong-frame receipt accepted"); }
        STREAM_REQUIRE(streamer->copyStreamedData(**command));
        if (!defaultReceipt->isRecordedBefore(**command) || firstReceipt->isRecordedBefore(**command) ||
            streamer->pendingCopyStats(*first).copyCount() != 1 || streamer->stats().pendingCopies.copyCount() != 1) {
            return RHITestResult::fail("default flush consumed explicit batch");
        }
        STREAM_REQUIRE(streamer->copyStreamedData(**command, *first));
        if (!firstReceipt->isRecordedBefore(**command) || streamer->copyStreamedData(**command, *first) || stage(*first, 20)) {
            return RHITestResult::fail("batch consumed more than once");
        }
        const StreamDataChunk overflow[]{{&data, UINT64_MAX - 3}, {&data, 8}};
        if (streamer->streamBufferData({.dataChunks = overflow}).valid()) { return RHITestResult::fail("chunk size overflow accepted"); }
        auto throwing = streamer->beginCopyBatch();
        STREAM_REQUIRE(throwing);
        if (!stage(*throwing, 36)) { return RHITestResult::fail("exception stage failed"); }
        auto throwingReceipt = streamer->pendingCopyCompletion(*throwing);
        auto failedCommand = lane.prepare(frame);
        STREAM_REQUIRE(failedCommand);
        try {
            (void)streamer->copyStreamedData(**failedCommand, *throwing, [](const char*) { throw std::runtime_error("injected callback failure"); });
            return RHITestResult::fail("callback exception was lost");
        } catch (const std::runtime_error&) {
            if (!throwingReceipt || !throwingReceipt->isCancelled() || firstReceipt->isCancelled()) {
                return RHITestResult::fail("exception cancelled wrong batch or left receipt pending");
            }
        }
        STREAM_REQUIRE((*failedCommand)->end());
        auto abandoned = streamer->beginCopyBatch();
        STREAM_REQUIRE(abandoned);
        if (!stage(*abandoned, 24)) { return RHITestResult::fail("abandoned stage failed"); }
        auto abandonedReceipt = streamer->pendingCopyCompletion(*abandoned);
        streamer->endFrame();
        if (!abandonedReceipt || !abandonedReceipt->isCancelled() || streamer->beginCopyBatch()) { return RHITestResult::fail("endFrame did not cancel/close batches"); }
        STREAM_REQUIRE(streamer->beginFrame(frame));
        if (stage(*abandoned, 28)) { return RHITestResult::fail("stale batch survived frame boundary"); }
        auto destroyed = streamer->beginCopyBatch();
        STREAM_REQUIRE(destroyed);
        if (!stage(*destroyed, 32)) { return RHITestResult::fail("destruction stage failed"); }
        auto destroyedReceipt = streamer->pendingCopyCompletion(*destroyed);
        streamer.reset();
        if (!destroyedReceipt || !destroyedReceipt->isCancelled() || firstReceipt->isCancelled()) { return RHITestResult::fail("destruction cancelled wrong transaction"); }
        STREAM_REQUIRE((*command)->end());
        STREAM_REQUIRE((*wrongCommand)->end());
        frame.cancel();
        otherFrame.cancel();
        if (!firstReceipt->isCancelled() || firstReceipt->isComplete()) { return RHITestResult::fail("frame cancellation publication failed"); }
        return RHITestResult::pass("default, explicit, foreign, consumed, cancelled, wrong-frame and abandoned batches isolated");
    }
};

METALLIC_REGISTER_RHI_TEST(StreamerConcurrentBatchesTest);
METALLIC_REGISTER_RHI_TEST(StreamerBatchIsolationTest);
#endif

#undef STREAM_REQUIRE
} // namespace
} // namespace metallic::tests
