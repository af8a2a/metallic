#include "Runtime/Render/MeshletStreamPageLoader.h"
#include "Runtime/Task/TaskSystem.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <span>
#include <utility>

namespace metallic::render {

struct MeshletStreamPageLoader::Impl : std::enable_shared_from_this<MeshletStreamPageLoader::Impl> {
    task::TaskOutcome loadPage(uint32_t pageIndex) noexcept
    {
        MeshletStreamPageLoadResult result;
        result.pageIndex = pageIndex;
        try {
            if (asset == nullptr || pageIndex >= asset->pageCount()) {
                result.failureReason = "stream page index is out of range";
            } else {
                const scene::MeshletStreamPageInfo& page = asset->pages()[pageIndex];
                const std::span<const uint8_t> storedPayload = asset->pagePayload(pageIndex);
                std::vector<uint8_t> decodeStorage;
                std::span<const uint8_t> devicePayload;
                if (!scene::decodeMeshletStreamPayloadForDevice(
                        page,
                        storedPayload,
                        decodeStorage,
                        devicePayload,
                        result.failureReason) ||
                    devicePayload.empty() ||
                    devicePayload.size() != page.uncompressedSize) {
                    if (result.failureReason.empty()) {
                        result.failureReason = "stream page payload decode produced an invalid size";
                    }
                } else {
                    result.payload.assign(devicePayload.begin(), devicePayload.end());
                }
            }
        } catch (const std::exception& exception) {
            result.failureReason = exception.what();
        } catch (...) {
            result.failureReason = "stream page decode threw an unknown exception";
        }

        std::string taskFailure = result.failureReason;
        bool replenish = false;
        try {
            std::lock_guard lock(mutex);
            if (activeLoads != 0) {
                --activeLoads;
            }
            try {
                completedLoads.push_back(std::move(result));
            } catch (const std::exception& exception) {
                if (taskFailure.empty()) {
                    taskFailure = exception.what();
                }
            } catch (...) {
                if (taskFailure.empty()) {
                    taskFailure = "failed to store the completed stream page load";
                }
            }
            replenish = !stopping && asset != nullptr && !pendingPages.empty();
        } catch (...) {
            if (taskFailure.empty()) {
                taskFailure = "failed to finalize the stream page load";
            }
        }
        condition.notify_all();
        if (replenish) {
            scheduleAvailable();
        }

        if (!taskFailure.empty()) {
            return std::unexpected(std::move(taskFailure));
        }
        return {};
    }

    void completeRejectedSubmission(uint32_t pageIndex, std::string reason)
    {
        {
            std::lock_guard lock(mutex);
            if (activeLoads != 0) {
                --activeLoads;
            }
            completedLoads.push_back(MeshletStreamPageLoadResult{
                .pageIndex = pageIndex,
                .failureReason = std::move(reason),
            });
        }
        condition.notify_all();
    }

    void scheduleAvailable()
    {
        for (;;) {
            uint32_t pageIndex = UINT32_MAX;
            {
                std::lock_guard lock(mutex);
                if (stopping || asset == nullptr ||
                    activeLoads >= pageLoadConcurrency || pendingPages.empty()) {
                    return;
                }
                pageIndex = pendingPages.front();
                pendingPages.pop_front();
                ++activeLoads;
            }

            task::TaskGraph graph("MeshletStreamPageLoad");
            const std::shared_ptr<Impl> self = shared_from_this();
            graph.addTask(
                task::TaskDesc{
                    .name = "MeshletStreamPageLoad",
                    .category = "Streaming",
                    .userTag = pageIndex,
                },
                [self, pageIndex]() -> task::TaskOutcome {
                    return self->loadPage(pageIndex);
                });

            const std::shared_ptr<task::TaskSystem> system = task::detail::tryAcquireTaskSystem();
            if (system == nullptr) {
                completeRejectedSubmission(pageIndex, "TaskSystem is not initialized");
                continue;
            }
            auto submitted = system->submit(std::move(graph));
            if (!submitted) {
                completeRejectedSubmission(pageIndex, submitted.error().message);
            }
        }
    }

    const scene::MeshletStreamAsset* asset = nullptr;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<uint32_t> pendingPages;
    std::deque<MeshletStreamPageLoadResult> completedLoads;
    uint32_t pageLoadConcurrency = 0;
    uint32_t activeLoads = 0;
    bool stopping = false;
};

MeshletStreamPageLoader::MeshletStreamPageLoader()
    : impl_(std::make_shared<Impl>())
{
}

MeshletStreamPageLoader::~MeshletStreamPageLoader()
{
    reset();
}

bool MeshletStreamPageLoader::initialize(
    const scene::MeshletStreamAsset& asset,
    uint32_t concurrency,
    std::string& reason)
{
    reset();
    reason.clear();
    if (!asset.valid()) {
        reason = "MeshletStreamPageLoader requires a valid streamasset";
        return false;
    }
    if (concurrency == 0) {
        reason = "MeshletStreamPageLoader requires non-zero concurrency";
        return false;
    }
    if (concurrency > kMeshletStreamMaxPageLoadConcurrency) {
        reason = "MeshletStreamPageLoader concurrency exceeds the supported limit";
        return false;
    }
    const std::shared_ptr<task::TaskSystem> system = task::detail::tryAcquireTaskSystem();
    if (system == nullptr || !system->acceptingTasks()) {
        reason = "MeshletStreamPageLoader requires an initialized TaskSystem";
        return false;
    }

    impl_->asset = &asset;
    impl_->stopping = false;
    impl_->pageLoadConcurrency = concurrency;
    return true;
}

void MeshletStreamPageLoader::reset()
{
    if (impl_ == nullptr) {
        return;
    }
    {
        std::lock_guard lock(impl_->mutex);
        impl_->stopping = true;
        impl_->pendingPages.clear();
    }
    {
        std::unique_lock lock(impl_->mutex);
        impl_->condition.wait(lock, [this] {
            return impl_->activeLoads == 0;
        });
    }
    {
        std::lock_guard lock(impl_->mutex);
        impl_->pendingPages.clear();
        impl_->completedLoads.clear();
        impl_->activeLoads = 0;
        impl_->pageLoadConcurrency = 0;
        impl_->asset = nullptr;
        impl_->stopping = false;
    }
}

bool MeshletStreamPageLoader::enqueue(uint32_t pageIndex)
{
    if (impl_ == nullptr) {
        return false;
    }
    bool schedule = false;
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->pageLoadConcurrency == 0 || impl_->stopping || impl_->asset == nullptr) {
            return false;
        }
        impl_->pendingPages.push_back(pageIndex);
        schedule = impl_->activeLoads < impl_->pageLoadConcurrency;
    }
    if (schedule) {
        impl_->scheduleAvailable();
    }
    return true;
}

bool MeshletStreamPageLoader::tryPop(MeshletStreamPageLoadResult& outResult)
{
    if (impl_ == nullptr) {
        return false;
    }
    std::lock_guard lock(impl_->mutex);
    if (impl_->completedLoads.empty()) {
        return false;
    }
    outResult = std::move(impl_->completedLoads.front());
    impl_->completedLoads.pop_front();
    return true;
}

bool MeshletStreamPageLoader::ready() const
{
    return concurrency() != 0;
}

uint32_t MeshletStreamPageLoader::concurrency() const
{
    if (impl_ == nullptr) {
        return 0;
    }
    std::lock_guard lock(impl_->mutex);
    return impl_->pageLoadConcurrency;
}

uint32_t MeshletStreamPageLoader::pendingCount() const
{
    if (impl_ == nullptr) {
        return 0;
    }
    std::lock_guard lock(impl_->mutex);
    return static_cast<uint32_t>(impl_->pendingPages.size());
}

uint32_t MeshletStreamPageLoader::activeCount() const
{
    if (impl_ == nullptr) {
        return 0;
    }
    std::lock_guard lock(impl_->mutex);
    return impl_->activeLoads;
}

uint32_t MeshletStreamPageLoader::completedCount() const
{
    if (impl_ == nullptr) {
        return 0;
    }
    std::lock_guard lock(impl_->mutex);
    return static_cast<uint32_t>(impl_->completedLoads.size());
}

uint32_t MeshletStreamPageLoader::outstandingCount() const
{
    if (impl_ == nullptr) {
        return 0;
    }
    std::lock_guard lock(impl_->mutex);
    return static_cast<uint32_t>(impl_->pendingPages.size()) +
        impl_->activeLoads +
        static_cast<uint32_t>(impl_->completedLoads.size());
}

} // namespace metallic::render
