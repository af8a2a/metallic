#include "Runtime/Scene/SceneLoader.h"

#include "Runtime/Task/TaskSystem.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
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

uint64_t estimatedDecodedByteSize(const SceneDocument& scene, size_t imageIndex)
{
    if (imageIndex >= scene.images().size()) {
        return 1;
    }
    const RenderImage& image = scene.images()[imageIndex];
    int width = 0;
    int height = 0;
    int channels = 0;
    size_t sourceCount = 1;
    bool valid = false;
    if (image.channelComposition.has_value()) {
        sourceCount = std::max<size_t>(image.channelComposition->sources.size(), 1u);
        for (const RenderImage::ChannelSource& source : image.channelComposition->sources) {
            if (queryImageInfo(scene, source, width, height, channels)) {
                valid = true;
                break;
            }
        }
    } else {
        valid = queryImageInfo(scene, image, width, height, channels);
    }
    if (!valid || width <= 0 || height <= 0) {
        return std::max<uint64_t>(image.encodedData.size(), 1u);
    }
    const uint64_t baseBytes = rgba8ByteSize(
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height));
    const uint64_t workingBytes = baseBytes * static_cast<uint64_t>(sourceCount + 1u);
    return workingBytes + baseBytes / 3u + 4u;
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

void appendMipChain(DecodedImageResult& result, RenderImage::Mip baseMip)
{
    result.mips.push_back(std::move(baseMip));
    while (result.mips.back().width > 1 || result.mips.back().height > 1) {
        const RenderImage::Mip& source = result.mips.back();
        RenderImage::Mip mip;
        mip.width = std::max(source.width / 2u, 1u);
        mip.height = std::max(source.height / 2u, 1u);
        mip.pixels = buildNextMip(source.pixels.data(), source.width, source.height);
        result.mips.push_back(std::move(mip));
    }
}

DecodedImageResult decodeImage(const SceneDocument& scene, size_t imageIndex)
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
        appendMipChain(result, std::move(baseMip));
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
    appendMipChain(result, std::move(baseMip));
    return result;
}

struct DecodeThrottle {
    explicit DecodeThrottle(uint32_t concurrency)
        : concurrency(std::max(concurrency, 1u))
    {
    }

    bool acquire(const std::atomic_bool& cancelled)
    {
        std::unique_lock lock(mutex);
        while (active >= concurrency && !cancelled.load(std::memory_order_acquire)) {
            condition.wait_for(lock, std::chrono::milliseconds(5));
        }
        if (cancelled.load(std::memory_order_acquire)) {
            return false;
        }
        ++active;
        return true;
    }

    void release()
    {
        {
            std::lock_guard lock(mutex);
            --active;
        }
        condition.notify_one();
    }

    std::mutex mutex;
    std::condition_variable condition;
    uint32_t concurrency = 1;
    uint32_t active = 0;
};

struct DecodeByteThrottle {
    explicit DecodeByteThrottle(uint64_t byteLimit)
        : byteLimit(byteLimit)
    {
    }

    bool acquire(uint64_t byteCount, const std::atomic_bool& cancelled)
    {
        if (byteLimit == 0) {
            return !cancelled.load(std::memory_order_acquire);
        }
        std::unique_lock lock(mutex);
        const uint64_t reservation = std::max<uint64_t>(byteCount, 1u);
        while (!cancelled.load(std::memory_order_acquire)) {
            const bool oversizedExclusive = reservation > byteLimit && bytesInUse == 0;
            const bool fits = reservation <= byteLimit && bytesInUse <= byteLimit - reservation;
            if (oversizedExclusive || fits) {
                bytesInUse += reservation;
                return true;
            }
            condition.wait_for(lock, std::chrono::milliseconds(5));
        }
        return false;
    }

    void release(uint64_t byteCount)
    {
        if (byteLimit == 0) {
            return;
        }
        {
            std::lock_guard lock(mutex);
            bytesInUse -= std::min(bytesInUse, std::max<uint64_t>(byteCount, 1u));
        }
        condition.notify_all();
    }

    std::mutex mutex;
    std::condition_variable condition;
    uint64_t byteLimit = 0;
    uint64_t bytesInUse = 0;
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
        return;
    }
    const auto failed = std::find_if(snapshots.begin(), snapshots.end(), [](const auto& snapshot) {
        return snapshot.status == task::TaskGraphStatus::Failed;
    });
    if (failed != snapshots.end()) {
        state_->progress.status = SceneLoadStatus::Failed;
        state_->progress.phase = SceneLoadPhase::Failed;
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
    state_->cancelRequested.store(true, std::memory_order_release);
    std::vector<std::shared_ptr<task::TaskGraphRun>> runs;
    {
        std::lock_guard lock(state_->mutex);
        if (isTerminal(state_->progress.status)) {
            return false;
        }
        state_->progress.status = SceneLoadStatus::Cancelled;
        state_->progress.phase = SceneLoadPhase::Cancelled;
        state_->progress.currentItem.clear();
        runs = state_->runs;
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

    const uint32_t decodeConcurrency = options.decodeConcurrency != 0
        ? options.decodeConcurrency
        : std::min(8u, std::max(1u, system->workerCount() > 1 ? system->workerCount() - 1u : 1u));
    auto decodeThrottle = std::make_shared<DecodeThrottle>(decodeConcurrency);
    auto decodeByteThrottle = std::make_shared<DecodeByteThrottle>(options.maxDecodedBytesInFlight);
    auto decodedImageCount = std::make_shared<std::atomic_size_t>(0);

    task::TaskGraph graph("SceneLoad");
    graph.addTask(
        task::TaskDesc{
            .name = "LoadSceneDocument",
            .category = "SceneLoad",
        },
        [state, path, system, decodeThrottle, decodeByteThrottle, decodedImageCount](task::TaskContext& context) -> task::TaskOutcome {
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
                    : candidate->images().size();
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
                        state->progress.totalUnits = candidate->images().size();
                    }
                });
            for (const task::TaskNodeHandle geometryTask : geometryTasks) {
                const auto dependency = decodeGraph.addDependency(geometryTask, geometryFinalize);
                if (!dependency) {
                    return std::unexpected(dependency.error().message);
                }
            }

            std::vector<task::TaskNodeHandle> decodeTasks;
            decodeTasks.reserve(candidate->images().size());
            for (size_t imageIndex = 0; imageIndex < candidate->images().size(); ++imageIndex) {
                const uint64_t estimatedBytes = estimatedDecodedByteSize(*candidate, imageIndex);
                const task::TaskNodeHandle decodeTask = decodeGraph.addTask(
                    task::TaskDesc{
                        .name = "DecodeImage",
                        .category = "SceneLoad",
                        .userTag = imageIndex,
                    },
                    [state,
                     candidate,
                     decodeThrottle,
                     decodeByteThrottle,
                     decodedImageCount,
                     imageIndex,
                     estimatedBytes]() {
                        if (!decodeThrottle->acquire(state->cancelRequested)) {
                            return;
                        }
                        struct ReleaseGuard {
                            std::shared_ptr<DecodeThrottle> throttle;
                            ~ReleaseGuard() { throttle->release(); }
                        } releaseGuard{decodeThrottle};

                        if (!decodeByteThrottle->acquire(estimatedBytes, state->cancelRequested)) {
                            return;
                        }
                        struct ByteReleaseGuard {
                            std::shared_ptr<DecodeByteThrottle> throttle;
                            uint64_t byteCount = 0;
                            ~ByteReleaseGuard() { throttle->release(byteCount); }
                        } byteReleaseGuard{decodeByteThrottle, estimatedBytes};

                        if (state->cancelRequested.load(std::memory_order_acquire)) {
                            return;
                        }
                        DecodedImageResult decoded = decodeImage(*candidate, imageIndex);
                        (void)candidate->setImageDecodeResult(
                            imageIndex,
                            std::move(decoded.mips),
                            std::move(decoded.warning));
                        const size_t completed = decodedImageCount->fetch_add(1, std::memory_order_acq_rel) + 1u;
                        std::lock_guard lock(state->mutex);
                        if (!isTerminal(state->progress.status)) {
                            state->progress.status = SceneLoadStatus::Running;
                            state->progress.phase = SceneLoadPhase::Images;
                            state->progress.fraction = std::max(
                                state->progress.fraction,
                                0.40f + 0.25f * static_cast<float>(completed) /
                                    static_cast<float>(candidate->images().size()));
                            state->progress.completedUnits = completed;
                            state->progress.totalUnits = candidate->images().size();
                            state->progress.currentItem = candidate->images()[imageIndex].name;
                            state->progress.elapsed = SceneLoadClock::now() - state->begin;
                        }
                    });
                decodeTasks.push_back(decodeTask);
                const auto dependency = decodeGraph.addDependency(geometryFinalize, decodeTask);
                if (!dependency) {
                    return std::unexpected(dependency.error().message);
                }
            }

            const task::TaskNodeHandle finalizeTask = decodeGraph.addTask(
                task::TaskDesc{
                    .name = "FinalizeSceneImages",
                    .category = "SceneLoad",
                },
                [state, candidate]() {
                    std::lock_guard lock(state->mutex);
                    if (state->cancelRequested.load(std::memory_order_acquire) ||
                        isTerminal(state->progress.status)) {
                        return;
                    }
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
            for (const task::TaskNodeHandle decodeTask : decodeTasks) {
                const auto dependency = decodeGraph.addDependency(decodeTask, finalizeTask);
                if (!dependency) {
                    std::lock_guard lock(state->mutex);
                    state->progress.status = SceneLoadStatus::Failed;
                    state->progress.phase = SceneLoadPhase::Failed;
                    state->progress.error = dependency.error().message;
                    return std::unexpected(dependency.error().message);
                }
            }
            if (decodeTasks.empty()) {
                const auto dependency = decodeGraph.addDependency(geometryFinalize, finalizeTask);
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
                state->runs.push_back(
                    std::make_shared<task::TaskGraphRun>(std::move(*submittedDecode)));
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
