#include "Runtime/Scene/SceneLoader.h"

#include "Runtime/Task/TaskSystem.h"

#include <spdlog/spdlog.h>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metallic::scene {
namespace {

using SceneLoadClock = std::chrono::steady_clock;

bool isTerminal(SceneLoadStatus status)
{
    return status == SceneLoadStatus::Succeeded ||
        status == SceneLoadStatus::Failed ||
        status == SceneLoadStatus::Cancelled;
}

uint64_t rgba8ByteSize(uint32_t width, uint32_t height)
{
    return static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull;
}

std::filesystem::path imagePathForUri(
    const SceneDocument& scene,
    std::string uri)
{
#ifndef _WIN32
    std::replace(uri.begin(), uri.end(), '\\', '/');
#endif
    std::filesystem::path imagePath = std::move(uri);
    if (imagePath.is_relative()) {
        imagePath = scene.filename().parent_path() / imagePath;
    }
    return imagePath;
}

template <typename ImageSourceT>
bool queryImageInfo(
    const SceneDocument& scene,
    const ImageSourceT& source,
    int& width,
    int& height,
    int& channels)
{
    if (!source.encodedData.empty() &&
        source.encodedData.size() <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        return stbi_info_from_memory(
                   source.encodedData.data(),
                   static_cast<int>(source.encodedData.size()),
                   &width,
                   &height,
                   &channels) != 0;
    }
    if (!source.uri.empty() && source.uri.rfind("data:", 0) != 0) {
        const std::filesystem::path imagePath = imagePathForUri(scene, source.uri);
        return stbi_info(imagePath.string().c_str(), &width, &height, &channels) != 0;
    }
    return false;
}

uint64_t saturatedAdd(uint64_t left, uint64_t right)
{
    return left + std::min(right, std::numeric_limits<uint64_t>::max() - left);
}

uint64_t estimatedImageWorkingBytes(const SceneDocument& scene, const RenderImage& image)
{
    const auto sourceBytes = [&scene](const auto& source) {
        int width = 0;
        int height = 0;
        int channels = 0;
        return queryImageInfo(scene, source, width, height, channels) && width > 0 && height > 0
            ? rgba8ByteSize(static_cast<uint32_t>(width), static_cast<uint32_t>(height))
            : std::max<uint64_t>(source.encodedData.size(), 1u);
    };
    if (!image.channelComposition.has_value()) {
        const uint64_t baseBytes = sourceBytes(image);
        // STB output and its vector copy coexist during decode. A complete mip
        // chain (including a 1D chain) fits within this same two-base-level peak.
        return saturatedAdd(baseBytes, baseBytes);
    }

    uint64_t retainedSourceBytes = 0;
    uint64_t largestSourceBytes = 4;
    uint64_t peakBytes = 0;
    for (const auto& source : image.channelComposition->sources) {
        const uint64_t bytes = sourceBytes(source);
        peakBytes = std::max(peakBytes, saturatedAdd(retainedSourceBytes, saturatedAdd(bytes, bytes)));
        retainedSourceBytes = saturatedAdd(retainedSourceBytes, bytes);
        largestSourceBytes = std::max(largestSourceBytes, bytes);
    }
    // Composition retains all sources and one target; the sources are released
    // before mip generation. Use each source's dimensions, which may differ.
    peakBytes = std::max(peakBytes, saturatedAdd(retainedSourceBytes, largestSourceBytes));
    return std::max(peakBytes, saturatedAdd(largestSourceBytes, largestSourceBytes));
}

std::vector<uint8_t> buildNextMip(
    const uint8_t* source,
    uint32_t sourceWidth,
    uint32_t sourceHeight)
{
    const uint32_t width = std::max(sourceWidth / 2u, 1u);
    const uint32_t height = std::max(sourceHeight / 2u, 1u);
    std::vector<uint8_t> pixels(static_cast<size_t>(rgba8ByteSize(width, height)));
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t sums[4]{};
            uint32_t sampleCount = 0;
            for (uint32_t offsetY = 0; offsetY < 2; ++offsetY) {
                const uint32_t sourceY = std::min(y * 2u + offsetY, sourceHeight - 1u);
                for (uint32_t offsetX = 0; offsetX < 2; ++offsetX) {
                    const uint32_t sourceX = std::min(x * 2u + offsetX, sourceWidth - 1u);
                    const size_t sourceOffset = static_cast<size_t>(sourceY * sourceWidth + sourceX) * 4u;
                    for (uint32_t component = 0; component < 4; ++component) {
                        sums[component] += source[sourceOffset + component];
                    }
                    ++sampleCount;
                }
            }
            const size_t targetOffset = static_cast<size_t>(y * width + x) * 4u;
            for (uint32_t component = 0; component < 4; ++component) {
                pixels[targetOffset + component] =
                    static_cast<uint8_t>((sums[component] + sampleCount / 2u) / sampleCount);
            }
        }
    }
    return pixels;
}

struct DecodedImageResult {
    std::vector<RenderImage::Mip> mips;
    std::string warning;
};

void appendDecodeWarning(std::string& warning, std::string message)
{
    if (message.empty()) {
        return;
    }
    if (!warning.empty() && warning.back() != '\n') {
        warning += '\n';
    }
    warning += std::move(message);
}

struct LoadedImageSource {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;
};

template <typename ImageSourceT>
LoadedImageSource loadImageSource(
    const SceneDocument& scene,
    const ImageSourceT& source,
    std::string_view label,
    std::string& warning)
{
    LoadedImageSource result;
    int width = 0;
    int height = 0;
    int channelCount = 0;
    stbi_uc* decoded = nullptr;
    if (!source.encodedData.empty()) {
        if (source.encodedData.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            appendDecodeWarning(warning, std::string(label) + " is too large to decode");
            return result;
        }
        decoded = stbi_load_from_memory(
            source.encodedData.data(),
            static_cast<int>(source.encodedData.size()),
            &width,
            &height,
            &channelCount,
            4);
    } else if (!source.uri.empty()) {
        if (source.uri.rfind("data:", 0) == 0) {
            appendDecodeWarning(
                warning,
                "data URI material textures are not supported yet");
            return result;
        }
        const std::filesystem::path imagePath = imagePathForUri(scene, source.uri);
        decoded = stbi_load(imagePath.string().c_str(), &width, &height, &channelCount, 4);
    }

    if (decoded == nullptr || width <= 0 || height <= 0) {
        std::string message = "failed to decode image '" + std::string(label) + "'";
        if (const char* reason = stbi_failure_reason()) {
            message += ": ";
            message += reason;
        }
        appendDecodeWarning(warning, std::move(message));
        if (decoded != nullptr) {
            stbi_image_free(decoded);
        }
        return result;
    }

    const uint64_t byteSize = rgba8ByteSize(
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height));
    if (byteSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        stbi_image_free(decoded);
        appendDecodeWarning(warning, std::string(label) + " is too large");
        return result;
    }

    result.width = static_cast<uint32_t>(width);
    result.height = static_cast<uint32_t>(height);
    result.pixels.assign(decoded, decoded + static_cast<size_t>(byteSize));
    stbi_image_free(decoded);
    return result;
}

void appendMipChain(
    DecodedImageResult& result,
    const std::atomic_bool& cancelled,
    task::TaskContext& context)
{
    while (!result.mips.empty() &&
        (result.mips.back().width > 1 || result.mips.back().height > 1) &&
        !cancelled.load(std::memory_order_acquire) && !context.stopRequested()) {
        const RenderImage::Mip& source = result.mips.back();
        RenderImage::Mip mip;
        mip.width = std::max(source.width / 2u, 1u);
        mip.height = std::max(source.height / 2u, 1u);
        mip.pixels = buildNextMip(source.pixels.data(), source.width, source.height);
        result.mips.push_back(std::move(mip));
    }
}

DecodedImageResult decodeImageBase(const SceneDocument& scene, size_t imageIndex)
{
    DecodedImageResult result;
    if (imageIndex >= scene.images().size()) {
        result.warning = "image index is out of range";
        return result;
    }

    const RenderImage& image = scene.images()[imageIndex];
    if (image.channelComposition.has_value()) {
        const RenderImage::ChannelComposition& composition = *image.channelComposition;
        std::vector<LoadedImageSource> sources;
        sources.reserve(composition.sources.size());
        uint32_t targetWidth = 0;
        uint32_t targetHeight = 0;
        for (const RenderImage::ChannelSource& source : composition.sources) {
            const std::string label = source.uri.empty() ? image.name : source.uri;
            LoadedImageSource loaded = loadImageSource(scene, source, label, result.warning);
            if (targetWidth == 0 && !loaded.pixels.empty()) {
                targetWidth = loaded.width;
                targetHeight = loaded.height;
            }
            sources.push_back(std::move(loaded));
        }
        if (targetWidth == 0 || targetHeight == 0) {
            targetWidth = 1;
            targetHeight = 1;
        }

        RenderImage::Mip baseMip;
        baseMip.width = targetWidth;
        baseMip.height = targetHeight;
        baseMip.pixels.resize(static_cast<size_t>(rgba8ByteSize(targetWidth, targetHeight)));
        for (uint32_t y = 0; y < targetHeight; ++y) {
            for (uint32_t x = 0; x < targetWidth; ++x) {
                const size_t targetOffset =
                    (static_cast<size_t>(y) * targetWidth + x) * 4u;
                for (size_t channel = 0; channel < 4; ++channel) {
                    uint8_t value = composition.constants[channel];
                    const int32_t sourceIndex = composition.sourceIndices[channel];
                    if (sourceIndex >= 0 && static_cast<size_t>(sourceIndex) < sources.size()) {
                        const LoadedImageSource& source = sources[static_cast<size_t>(sourceIndex)];
                        if (!source.pixels.empty()) {
                            const uint32_t sourceX = std::min(
                                static_cast<uint32_t>(
                                    static_cast<uint64_t>(x) * source.width / targetWidth),
                                source.width - 1u);
                            const uint32_t sourceY = std::min(
                                static_cast<uint32_t>(
                                    static_cast<uint64_t>(y) * source.height / targetHeight),
                                source.height - 1u);
                            const uint8_t sourceChannel = std::min<uint8_t>(
                                composition.sourceChannels[channel],
                                3u);
                            const size_t sourceOffset =
                                (static_cast<size_t>(sourceY) * source.width + sourceX) * 4u;
                            value = source.pixels[sourceOffset + sourceChannel];
                        }
                    }
                    baseMip.pixels[targetOffset + channel] = value;
                }
            }
        }
        result.mips.push_back(std::move(baseMip));
        return result;
    }

    const std::string label = image.name.empty() ? image.uri : image.name;
    LoadedImageSource loaded = loadImageSource(scene, image, label, result.warning);
    if (loaded.pixels.empty()) {
        return result;
    }

    RenderImage::Mip baseMip;
    baseMip.width = loaded.width;
    baseMip.height = loaded.height;
    baseMip.pixels = std::move(loaded.pixels);
    result.mips.push_back(std::move(baseMip));
    return result;
}

// Encode memory admission in the graph instead of blocking TaskSystem workers.
// Reserve the larger stage peak from decode through mip publication, so queued
// base levels remain covered by the budget even when mip workers are busy.
struct ImageTaskBudget {
    struct Reservation {
        task::TaskNodeHandle finish;
        uint64_t bytes;
    };

    std::expected<void, task::TaskError> reserve(
        task::TaskGraph& graph,
        task::TaskNodeHandle start,
        task::TaskNodeHandle finish,
        uint64_t estimatedBytes)
    {
        if (byteLimit == 0) {
            return {};
        }
        // Oversized images consume the entire budget and therefore run alone.
        const uint64_t bytes = std::min(std::max(estimatedBytes, uint64_t{1}), byteLimit);
        if (reservedBytes > byteLimit - bytes) {
            const auto nextBarrier = graph.addTask(
                {.name = "ImageMemoryAvailable", .category = "SceneLoad"}, []() {});
            if (barrier.valid()) {
                if (auto dependency = graph.addDependency(barrier, nextBarrier); !dependency) {
                    return dependency;
                }
            }
            while (reservedBytes > byteLimit - bytes) {
                const auto reservation = reservations.front();
                if (auto dependency = graph.addDependency(reservation.finish, nextBarrier); !dependency) {
                    return dependency;
                }
                reservedBytes -= reservation.bytes;
                reservations.pop_front();
            }
            barrier = nextBarrier;
        }
        // Carry the entire release frontier forward. Depending only on the most
        // recently retired image is unsafe when differently sized jobs finish
        // out of order and reuse the same reserved bytes.
        if (barrier.valid()) {
            if (auto dependency = graph.addDependency(barrier, start); !dependency) {
                return dependency;
            }
        }
        reservedBytes += bytes;
        reservations.push_back({finish, bytes});
        return {};
    }

    uint64_t byteLimit = 0;
    uint64_t reservedBytes = 0;
    task::TaskNodeHandle barrier;
    std::deque<Reservation> reservations;
};

struct ImageStageMetrics {
    std::atomic_uint32_t active{0};
    std::atomic_uint32_t peak{0};
    std::atomic_int64_t workNanoseconds{0};
};

struct ImageStageScope {
    explicit ImageStageScope(ImageStageMetrics& metrics)
        : metrics(metrics), begin(SceneLoadClock::now())
    {
        const uint32_t active = metrics.active.fetch_add(1, std::memory_order_relaxed) + 1u;
        uint32_t peak = metrics.peak.load(std::memory_order_relaxed);
        while (peak < active &&
            !metrics.peak.compare_exchange_weak(peak, active, std::memory_order_relaxed)) {
        }
    }

    ~ImageStageScope()
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(SceneLoadClock::now() - begin);
        metrics.workNanoseconds.fetch_add(elapsed.count(), std::memory_order_relaxed);
        metrics.active.fetch_sub(1, std::memory_order_relaxed);
    }

    ImageStageMetrics& metrics;
    SceneLoadClock::time_point begin;
};

struct ImagePipelineMetrics {
    ImageStageMetrics decode;
    ImageStageMetrics mips;
    SceneLoadClock::time_point begin;
};

} // namespace

struct SceneLoadHandle::State {
    mutable std::mutex mutex;
    SceneLoadProgress progress;
    std::unique_ptr<SceneDocument> result;
    std::shared_ptr<SceneDocument> candidate;
    std::vector<std::shared_ptr<task::TaskGraphRun>> runs;
    std::atomic_bool cancelRequested{false};
    SceneLoadClock::time_point begin = SceneLoadClock::now();
    bool resultTaken = false;
};

void SceneLoadHandle::refreshTerminalState() const
{
    if (state_ == nullptr) {
        return;
    }
    std::vector<std::shared_ptr<task::TaskGraphRun>> runs;
    {
        std::lock_guard lock(state_->mutex);
        if (isTerminal(state_->progress.status)) {
            return;
        }
        runs = state_->runs;
    }
    if (runs.empty() || std::any_of(runs.begin(), runs.end(), [](const auto& run) {
            return run == nullptr || !run->isComplete();
        })) {
        return;
    }

    std::vector<task::TaskGraphSnapshot> snapshots;
    snapshots.reserve(runs.size());
    for (const std::shared_ptr<task::TaskGraphRun>& run : runs) {
        snapshots.push_back(run->snapshot());
    }
    std::lock_guard lock(state_->mutex);
    if (isTerminal(state_->progress.status)) {
        return;
    }
    const bool graphCancelled = std::any_of(snapshots.begin(), snapshots.end(), [](const auto& snapshot) {
        return snapshot.status == task::TaskGraphStatus::Cancelled;
    });
    if (graphCancelled || state_->cancelRequested.load(std::memory_order_acquire)) {
        state_->progress.status = SceneLoadStatus::Cancelled;
        state_->progress.phase = SceneLoadPhase::Cancelled;
        state_->progress.currentItem.clear();
        state_->candidate.reset();
        state_->runs.clear();
        return;
    }
    const auto failed = std::find_if(snapshots.begin(), snapshots.end(), [](const auto& snapshot) {
        return snapshot.status == task::TaskGraphStatus::Failed;
    });
    if (failed != snapshots.end()) {
        state_->progress.status = SceneLoadStatus::Failed;
        state_->progress.phase = SceneLoadPhase::Failed;
        state_->candidate.reset();
        state_->runs.clear();
        for (const task::TaskNodeSnapshot& node : failed->nodes) {
            if (!node.error.empty()) {
                state_->progress.error = node.error;
                break;
            }
        }
    }
}

SceneLoadHandle::SceneLoadHandle(std::shared_ptr<State> state)
    : state_(std::move(state))
{
}

bool SceneLoadHandle::valid() const
{
    return state_ != nullptr;
}

bool SceneLoadHandle::complete() const
{
    if (state_ == nullptr) {
        return false;
    }
    refreshTerminalState();
    std::lock_guard lock(state_->mutex);
    return isTerminal(state_->progress.status);
}

SceneLoadProgress SceneLoadHandle::progress() const
{
    if (state_ == nullptr) {
        return {};
    }
    refreshTerminalState();
    std::lock_guard lock(state_->mutex);
    SceneLoadProgress progress = state_->progress;
    progress.elapsed = SceneLoadClock::now() - state_->begin;
    return progress;
}

bool SceneLoadHandle::cancel()
{
    if (state_ == nullptr) {
        return false;
    }
    std::vector<std::shared_ptr<task::TaskGraphRun>> runs;
    {
        std::lock_guard lock(state_->mutex);
        if (isTerminal(state_->progress.status)) {
            return false;
        }
        state_->cancelRequested.store(true, std::memory_order_release);
        state_->progress.status = SceneLoadStatus::Cancelled;
        state_->progress.phase = SceneLoadPhase::Cancelled;
        state_->progress.currentItem.clear();
        // Skipped task callbacks retain their captures. Break the state/run
        // cycle so intermediate base levels are freed when the graph drains.
        runs = std::move(state_->runs);
        state_->candidate.reset();
    }
    for (const std::shared_ptr<task::TaskGraphRun>& run : runs) {
        if (run != nullptr) {
            (void)run->requestStop();
        }
    }
    return true;
}

std::unique_ptr<SceneDocument> SceneLoadHandle::takeResult()
{
    if (state_ == nullptr) {
        return nullptr;
    }
    std::lock_guard lock(state_->mutex);
    if (state_->progress.status != SceneLoadStatus::Succeeded || state_->resultTaken) {
        return nullptr;
    }
    state_->resultTaken = true;
    return std::move(state_->result);
}

SceneLoadHandle SceneLoader::request(
    const std::filesystem::path& path,
    const SceneLoadOptions& options) const
{
    auto state = std::make_shared<SceneLoadHandle::State>();
    state->progress.status = SceneLoadStatus::Running;
    state->progress.phase = SceneLoadPhase::Queued;
    state->progress.currentItem = path.string();

    const std::shared_ptr<task::TaskSystem> system = task::detail::tryAcquireTaskSystem();
    if (system == nullptr || !system->acceptingTasks()) {
        std::lock_guard lock(state->mutex);
        state->progress.status = SceneLoadStatus::Failed;
        state->progress.phase = SceneLoadPhase::Failed;
        state->progress.error = "TaskSystem is not initialized";
        return SceneLoadHandle(std::move(state));
    }

    const uint32_t workers = std::max(system->workerCount(), 1u);
    const uint32_t automaticConcurrency = std::min(8u, std::max(workers - 1u, 1u));
    const uint32_t decodeConcurrency = options.decodeConcurrency != 0
        ? std::min(options.decodeConcurrency, workers) : automaticConcurrency;
    const uint32_t mipConcurrency = options.mipConcurrency != 0
        ? std::min(options.mipConcurrency, workers) : automaticConcurrency;

    task::TaskGraph graph("SceneLoad");
    graph.addTask(
        task::TaskDesc{
            .name = "LoadSceneDocument",
            .category = "SceneLoad",
        },
        [state, path, system, options, decodeConcurrency, mipConcurrency](task::TaskContext& context) -> task::TaskOutcome {
            auto candidate = std::make_shared<SceneDocument>();
            const SceneLoadProgressCallback callback =
                [state, &context](const SceneLoadProgress& update) {
                    if (state->cancelRequested.load(std::memory_order_acquire) || context.stopRequested()) {
                        return false;
                    }
                    std::lock_guard lock(state->mutex);
                    if (isTerminal(state->progress.status)) {
                        return false;
                    }
                    const float previousFraction = state->progress.fraction;
                    state->progress = update;
                    state->progress.status = SceneLoadStatus::Running;
                    float mappedFraction = previousFraction;
                    if (update.phase == SceneLoadPhase::Parsing) {
                        mappedFraction = std::min(update.fraction, 0.10f);
                    } else if (update.phase == SceneLoadPhase::Geometry) {
                        mappedFraction = update.currentItem == "Meshlet cache"
                            ? 0.40f
                            : 0.20f;
                    }
                    state->progress.fraction = std::max(previousFraction, mappedFraction);
                    state->progress.elapsed = SceneLoadClock::now() - state->begin;
                    return true;
                };

            const bool loaded = candidate->loadDeferredMeshlets(path, callback);
            const bool cancelled = state->cancelRequested.load(std::memory_order_acquire) || context.stopRequested();
            if (cancelled) {
                std::lock_guard lock(state->mutex);
                state->progress.status = SceneLoadStatus::Cancelled;
                state->progress.phase = SceneLoadPhase::Cancelled;
                state->progress.currentItem.clear();
                return {};
            }
            if (!loaded) {
                std::lock_guard lock(state->mutex);
                state->progress.status = SceneLoadStatus::Failed;
                state->progress.phase = SceneLoadPhase::Failed;
                state->progress.error = !candidate->lastLoadResult().error.empty()
                    ? candidate->lastLoadResult().error
                    : candidate->documentWarning();
                return std::unexpected(state->progress.error.empty()
                    ? std::string("Scene load failed")
                    : state->progress.error);
            }

            {
                std::lock_guard lock(state->mutex);
                if (isTerminal(state->progress.status)) {
                    return {};
                }
                state->candidate = candidate;
                state->progress.phase = candidate->hasDeferredMeshlets()
                    ? SceneLoadPhase::Geometry
                    : SceneLoadPhase::Images;
                state->progress.fraction = std::max(state->progress.fraction, 0.40f);
                state->progress.completedUnits = 0;
                state->progress.totalUnits = candidate->hasDeferredMeshlets()
                    ? candidate->renderPrimitives().size()
                    : candidate->images().size() * 2u;
                state->progress.currentItem.clear();
            }

            task::TaskGraph decodeGraph("SceneCpuPayload");
            auto builtPrimitiveCount = std::make_shared<std::atomic_size_t>(0);
            std::vector<task::TaskNodeHandle> geometryTasks;
            if (candidate->hasDeferredMeshlets()) {
                geometryTasks.reserve(candidate->renderPrimitives().size());
                for (size_t primitiveIndex = 0;
                     primitiveIndex < candidate->renderPrimitives().size();
                     ++primitiveIndex) {
                    geometryTasks.push_back(decodeGraph.addTask(
                        task::TaskDesc{
                            .name = "BuildPrimitiveMeshlets",
                            .category = "SceneLoad",
                            .userTag = primitiveIndex,
                        },
                        [state, candidate, builtPrimitiveCount, primitiveIndex]() {
                            if (!state->cancelRequested.load(std::memory_order_acquire)) {
                                (void)candidate->buildDeferredMeshlet(primitiveIndex);
                                const size_t completed = builtPrimitiveCount->fetch_add(
                                    1,
                                    std::memory_order_acq_rel) + 1u;
                                std::lock_guard lock(state->mutex);
                                if (!isTerminal(state->progress.status)) {
                                    state->progress.phase = SceneLoadPhase::Geometry;
                                    state->progress.fraction = std::max(
                                        state->progress.fraction,
                                        0.20f + 0.20f * static_cast<float>(completed) /
                                            static_cast<float>(candidate->renderPrimitives().size()));
                                    state->progress.completedUnits = completed;
                                    state->progress.totalUnits = candidate->renderPrimitives().size();
                                    state->progress.currentItem =
                                        candidate->renderPrimitives()[primitiveIndex].name;
                                }
                            }
                        }));
                }
            }
            const task::TaskNodeHandle geometryFinalize = decodeGraph.addTask(
                task::TaskDesc{
                    .name = "FinalizeSceneMeshlets",
                    .category = "SceneLoad",
                },
                [state, candidate]() {
                    if (state->cancelRequested.load(std::memory_order_acquire)) {
                        return;
                    }
                    (void)candidate->finalizeDeferredMeshlets();
                    std::lock_guard lock(state->mutex);
                    if (!isTerminal(state->progress.status)) {
                        state->progress.phase = SceneLoadPhase::Images;
                        state->progress.fraction = std::max(state->progress.fraction, 0.40f);
                        state->progress.completedUnits = 0;
                        state->progress.totalUnits = candidate->images().size() * 2u;
                    }
                });
            for (const task::TaskNodeHandle geometryTask : geometryTasks) {
                const auto dependency = decodeGraph.addDependency(geometryTask, geometryFinalize);
                if (!dependency) {
                    return std::unexpected(dependency.error().message);
                }
            }

            auto metrics = std::make_shared<ImagePipelineMetrics>();
            const auto imagesBegin = decodeGraph.addTask(
                {.name = "BeginSceneImages", .category = "SceneLoad"},
                [metrics]() { metrics->begin = SceneLoadClock::now(); });
            if (auto dependency = decodeGraph.addDependency(geometryFinalize, imagesBegin); !dependency) {
                return std::unexpected(dependency.error().message);
            }
            const auto completeImageStage = [state, candidate](size_t imageIndex, std::string_view stage) {
                std::lock_guard lock(state->mutex);
                if (!isTerminal(state->progress.status)) {
                    state->progress.phase = SceneLoadPhase::Images;
                    ++state->progress.completedUnits;
                    state->progress.totalUnits = candidate->images().size() * 2u;
                    state->progress.fraction = std::max(
                        state->progress.fraction,
                        0.40f + 0.25f * static_cast<float>(state->progress.completedUnits) /
                            static_cast<float>(state->progress.totalUnits));
                    state->progress.currentItem = std::string(stage) + candidate->images()[imageIndex].name;
                    state->progress.elapsed = SceneLoadClock::now() - state->begin;
                }
            };

            ImageTaskBudget budget{.byteLimit = options.maxDecodedBytesInFlight};
            std::vector<task::TaskNodeHandle> decodeTasks;
            std::vector<task::TaskNodeHandle> mipTasks;
            decodeTasks.reserve(candidate->images().size());
            mipTasks.reserve(candidate->images().size());
            for (size_t imageIndex = 0; imageIndex < candidate->images().size(); ++imageIndex) {
                const uint64_t estimatedBytes = estimatedImageWorkingBytes(*candidate, candidate->images()[imageIndex]);
                auto decoded = std::make_shared<DecodedImageResult>();
                const auto decodeTask = decodeGraph.addTask(
                    {.name = "DecodeImage", .category = "SceneLoad", .userTag = imageIndex},
                    [state, candidate, decoded, metrics, completeImageStage, imageIndex](task::TaskContext& context) {
                        if (state->cancelRequested.load(std::memory_order_acquire) || context.stopRequested()) {
                            return;
                        }
                        {
                            ImageStageScope scope(metrics->decode);
                            *decoded = decodeImageBase(*candidate, imageIndex);
                        }
                        completeImageStage(imageIndex, "Decode: ");
                    });
                const auto mipTask = decodeGraph.addTask(
                    {.name = "BuildImageMips", .category = "SceneLoad", .userTag = imageIndex},
                    [state, candidate, decoded, metrics, completeImageStage, imageIndex](task::TaskContext& context) {
                        if (state->cancelRequested.load(std::memory_order_acquire) || context.stopRequested()) {
                            return;
                        }
                        {
                            ImageStageScope scope(metrics->mips);
                            appendMipChain(*decoded, state->cancelRequested, context);
                        }
                        if (state->cancelRequested.load(std::memory_order_acquire) || context.stopRequested()) {
                            return;
                        }
                        // Publish only a complete chain; decoding never exposes a partial image.
                        (void)candidate->setImageDecodeResult(
                            imageIndex, std::move(decoded->mips), std::move(decoded->warning));
                        completeImageStage(imageIndex, "Mips: ");
                    });
                if (auto dependency = decodeGraph.addDependency(imagesBegin, decodeTask); !dependency) {
                    return std::unexpected(dependency.error().message);
                }
                if (auto dependency = decodeGraph.addDependency(decodeTask, mipTask); !dependency) {
                    return std::unexpected(dependency.error().message);
                }
                // Separate dependency lanes bound each stage without occupying
                // a worker while waiting for a slot in the other stage.
                if (imageIndex >= decodeConcurrency) {
                    if (auto dependency = decodeGraph.addDependency(decodeTasks[imageIndex - decodeConcurrency], decodeTask);
                        !dependency) {
                        return std::unexpected(dependency.error().message);
                    }
                }
                if (imageIndex >= mipConcurrency) {
                    if (auto dependency = decodeGraph.addDependency(mipTasks[imageIndex - mipConcurrency], mipTask);
                        !dependency) {
                        return std::unexpected(dependency.error().message);
                    }
                }
                if (auto reservation = budget.reserve(decodeGraph, decodeTask, mipTask, estimatedBytes); !reservation) {
                    return std::unexpected(reservation.error().message);
                }
                decodeTasks.push_back(decodeTask);
                mipTasks.push_back(mipTask);
            }

            const task::TaskNodeHandle finalizeTask = decodeGraph.addTask(
                task::TaskDesc{
                    .name = "FinalizeSceneImages",
                    .category = "SceneLoad",
                },
                [state, candidate, metrics, decodeConcurrency, mipConcurrency, options, system]() {
                    std::lock_guard lock(state->mutex);
                    if (state->cancelRequested.load(std::memory_order_acquire) ||
                        isTerminal(state->progress.status)) {
                        return;
                    }
                    spdlog::info(
                        "[SceneLoad] Images: count={}, elapsed={:.2f} ms, decode task sum={:.2f} ms, mip task sum={:.2f} ms, "
                        "decode peak/limit={}/{}, mip peak/limit={}/{}, workers={}, working budget={} bytes (0=unlimited)",
                        candidate->images().size(),
                        std::chrono::duration<double, std::milli>(SceneLoadClock::now() - metrics->begin).count(),
                        metrics->decode.workNanoseconds.load(std::memory_order_relaxed) / 1.0e6,
                        metrics->mips.workNanoseconds.load(std::memory_order_relaxed) / 1.0e6,
                        metrics->decode.peak.load(std::memory_order_relaxed), decodeConcurrency,
                        metrics->mips.peak.load(std::memory_order_relaxed), mipConcurrency,
                        system->workerCount(), options.maxDecodedBytesInFlight);
                    state->result = std::make_unique<SceneDocument>(std::move(*candidate));
                    state->candidate.reset();
                    state->progress.status = SceneLoadStatus::Succeeded;
                    state->progress.phase = SceneLoadPhase::Completed;
                    state->progress.fraction = 1.0f;
                    state->progress.completedUnits = 1;
                    state->progress.totalUnits = 1;
                    state->progress.currentItem.clear();
                    state->progress.elapsed = SceneLoadClock::now() - state->begin;
                });
            for (const task::TaskNodeHandle mipTask : mipTasks) {
                const auto dependency = decodeGraph.addDependency(mipTask, finalizeTask);
                if (!dependency) {
                    std::lock_guard lock(state->mutex);
                    state->progress.status = SceneLoadStatus::Failed;
                    state->progress.phase = SceneLoadPhase::Failed;
                    state->progress.error = dependency.error().message;
                    return std::unexpected(dependency.error().message);
                }
            }
            if (mipTasks.empty()) {
                const auto dependency = decodeGraph.addDependency(imagesBegin, finalizeTask);
                if (!dependency) {
                    return std::unexpected(dependency.error().message);
                }
            }

            auto submittedDecode = system->submit(std::move(decodeGraph));
            if (!submittedDecode) {
                std::lock_guard lock(state->mutex);
                state->progress.status = SceneLoadStatus::Failed;
                state->progress.phase = SceneLoadPhase::Failed;
                state->progress.error = submittedDecode.error().message;
                return std::unexpected(submittedDecode.error().message);
            }
            {
                std::lock_guard lock(state->mutex);
                // Cancellation can race submission of the second graph.
                if (state->cancelRequested.load(std::memory_order_acquire)) {
                    (void)submittedDecode->requestStop();
                } else {
                    state->runs.push_back(
                        std::make_shared<task::TaskGraphRun>(std::move(*submittedDecode)));
                }
            }
            return {};
        });

    auto submitted = system->submit(std::move(graph));
    if (!submitted) {
        std::lock_guard lock(state->mutex);
        state->progress.status = SceneLoadStatus::Failed;
        state->progress.phase = SceneLoadPhase::Failed;
        state->progress.error = submitted.error().message;
        return SceneLoadHandle(std::move(state));
    }
    {
        std::lock_guard lock(state->mutex);
        state->runs.push_back(std::make_shared<task::TaskGraphRun>(std::move(*submitted)));
    }
    return SceneLoadHandle(std::move(state));
}

const char* sceneLoadPhaseName(SceneLoadPhase phase)
{
    switch (phase) {
    case SceneLoadPhase::Idle: return "Idle";
    case SceneLoadPhase::Queued: return "Queued";
    case SceneLoadPhase::Parsing: return "Parsing";
    case SceneLoadPhase::Geometry: return "Geometry";
    case SceneLoadPhase::Images: return "Images";
    case SceneLoadPhase::GpuUpload: return "GPU Upload";
    case SceneLoadPhase::AccelerationStructures: return "Acceleration Structures";
    case SceneLoadPhase::Finalizing: return "Finalizing";
    case SceneLoadPhase::Completed: return "Completed";
    case SceneLoadPhase::Failed: return "Failed";
    case SceneLoadPhase::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

const char* sceneLoadStatusName(SceneLoadStatus status)
{
    switch (status) {
    case SceneLoadStatus::Idle: return "Idle";
    case SceneLoadStatus::Running: return "Running";
    case SceneLoadStatus::Succeeded: return "Succeeded";
    case SceneLoadStatus::Failed: return "Failed";
    case SceneLoadStatus::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

} // namespace metallic::scene
