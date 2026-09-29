#include "Runtime/Render/Streamer/MeshletStreamInitialLoader.h"

#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <chrono>
#include <cmath>

namespace metallic::render {
namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kGpuWaitTimeoutNanoseconds = 30'000'000'000ull;

double elapsedMilliseconds(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct AccumulatedTimer {
    double& milliseconds;
    Clock::time_point start = Clock::now();
    ~AccumulatedTimer() { milliseconds += elapsedMilliseconds(start); }
};

} // namespace

MeshletStreamInitialLoader::~MeshletStreamInitialLoader()
{
    (void)reset();
}

Result<> MeshletStreamInitialLoader::initialize(Device& device, std::string& log)
{
    log.clear();
    if (device_ != nullptr) {
        if (device_ == &device && commands_ != nullptr) { return {}; }
        log = "Initial geometry loader is already initialized for another device";
        return makeError(Error::InvalidArgument);
    }
    queue_ = device.getQueue(QueueType::Graphics);
    if (queue_ == nullptr) {
        log = "Initial geometry loader requires a graphics queue";
        return makeError(Error::Unsupported);
    }
    auto result = submissions_.initialize(device, *queue_);
    if (result) {
        result = device.createCommandPool(*queue_).transform([&](auto pool) {
            commandPool_ = std::move(pool);
        });
    }
    if (result) {
        result = commandPool_->createCommandBuffer().transform([&](auto commands) {
            commands_ = std::move(commands);
        });
    }
    if (result) { result = uploads_.initialize(device, log, 1); }
    if (!result) {
        if (log.empty()) {
            log = std::string("Initial geometry loader initialization failed: ") + resultToString(result);
        }
        (void)reset();
        return result;
    }
    device_ = &device;
    return {};
}

Result<> MeshletStreamInitialLoader::pump(MeshletStreamRuntime& runtime,
    double budgetMilliseconds, bool& complete, std::string& log)
{
    log.clear();
    complete = false;
    if (!device_ || !queue_ || !commands_ || !runtime.ready() ||
        !std::isfinite(budgetMilliseconds) || budgetMilliseconds < 0.0) {
        log = "Initial geometry loader requires initialized resources and a finite non-negative budget";
        return makeError(Error::InvalidArgument);
    }
    ++stats_.pumpCalls;
    const AccumulatedTimer pumpTimer{stats_.pumpMilliseconds};
    auto readiness = runtime.sceneReadiness();
    if (readiness.ready) {
        const auto result = frame_.wait(kGpuWaitTimeoutNanoseconds);
        if (!result) {
            log = std::string("Initial geometry loader handoff wait failed (30 s timeout): ") + resultToString(result);
            return result;
        }
        complete = true;
        return {};
    }

    const auto fail = [&](Result<> result, const char* phase) {
        uploads_.endFrame();
        frame_.cancel();
        log = std::string("Initial geometry loading ") + phase + " failed: " + resultToString(result);
        return result;
    };
    do {
        auto result = frame_.begin(++frameIndex_, kGpuWaitTimeoutNanoseconds);
        if (!result) { return fail(result, "frame begin (30 s timeout)"); }
        result = commandPool_->reset();
        if (!result) { return fail(result, "command pool reset"); }
        result = commands_->begin(&frame_);
        if (!result) { return fail(result, "command buffer begin"); }
        result = uploads_.beginFrame(frame_);
        if (!result) { return fail(result, "upload frame begin"); }

        result = runtime.cmdLoadInitialResources(*commands_, *uploads_.streamer(), [&] {
            return uploads_.flush(*commands_);
        });
        if (!result) { return fail(result, "root resource recording"); }
        const auto residency = runtime.residency().stats();
        const auto clas = runtime.clasPool() ? runtime.clasPool()->stats() : MeshletStreamClasPoolStats{};
        uploads_.endFrame();
        result = commands_->end();
        if (!result) { return fail(result, "command buffer end"); }

        CommandBuffer* batch[] = {commands_.get()};
        // submit closes this frame's submission window. CLAS size readback and
        // relocation depend on its aggregate completion being published here.
        result = submissions_.submit({.commandBuffers = batch}, frame_);
        if (!result) { return fail(result, "queue submission"); }
        ++stats_.batches;
        stats_.uploadBytes += residency.frameUploadBytes;
        {
            const AccumulatedTimer waitTimer{stats_.gpuWaitMilliseconds};
            result = frame_.wait(kGpuWaitTimeoutNanoseconds);
        }
        if (!result) { return fail(result, "GPU completion wait (30 s timeout)"); }

        const auto nextReadiness = runtime.sceneReadiness();
        complete = nextReadiness.ready;
        const bool progressed = residency.frameUploadBytes != 0 ||
            clas.frameBuiltClusterCount != 0 || clas.frameMovedClusterCount != 0 ||
            nextReadiness.completedPages != readiness.completedPages;
        readiness = nextReadiness;
        if (complete || !progressed) { break; }
        // The next batch polls completed uploads/CLAS before admitting more.
        // If only asynchronous page I/O remains, yield to the caller instead of
        // issuing empty GPU submissions for the rest of the wall-time budget.
    } while (elapsedMilliseconds(pumpTimer.start) < budgetMilliseconds);
    return {};
}

Result<> MeshletStreamInitialLoader::reset()
{
    uploads_.endFrame();
    // Accepted commands survive cancellation and must finish before their pool,
    // streamer staging allocations, or runtime resources may be destroyed.
    auto result = frame_.reset();
    if (!result && !hasError(result, Error::DeviceLost)) { return result; }
    const auto submitted = submissions_.reset();
    if (!submitted && !hasError(submitted, Error::DeviceLost)) { return submitted; }
    commands_.reset();
    commandPool_.reset();
    uploads_.reset();
    queue_ = nullptr;
    device_ = nullptr;
    frameIndex_ = 0;
    stats_ = {};
    return !result ? result : submitted;
}

} // namespace metallic::render
