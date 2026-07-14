#include "Runtime/Render/MeshletStreamPageLoader.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <span>
#include <thread>
#include <utility>

namespace metallic::render {

struct MeshletStreamPageLoader::Impl {
    void runWorker()
    {
        for (;;) {
            uint32_t pageIndex = UINT32_MAX;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [this] {
                    return stopping || !pendingPages.empty();
                });
                if (stopping && pendingPages.empty()) {
                    return;
                }
                pageIndex = pendingPages.front();
                pendingPages.pop_front();
                ++activeLoads;
            }

            MeshletStreamPageLoadResult result;
            result.pageIndex = pageIndex;
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

            {
                std::lock_guard lock(mutex);
                --activeLoads;
                completedLoads.push_back(std::move(result));
            }
        }
    }

    const scene::MeshletStreamAsset* asset = nullptr;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<uint32_t> pendingPages;
    std::deque<MeshletStreamPageLoadResult> completedLoads;
    std::vector<std::thread> workers;
    uint32_t activeLoads = 0;
    bool stopping = false;
};

MeshletStreamPageLoader::MeshletStreamPageLoader()
    : impl_(std::make_unique<Impl>())
{
}

MeshletStreamPageLoader::~MeshletStreamPageLoader()
{
    reset();
}

bool MeshletStreamPageLoader::initialize(
    const scene::MeshletStreamAsset& asset,
    uint32_t workerCount,
    std::string& reason)
{
    reset();
    reason.clear();
    if (!asset.valid()) {
        reason = "MeshletStreamPageLoader requires a valid streamasset";
        return false;
    }
    if (workerCount == 0) {
        reason = "MeshletStreamPageLoader requires at least one worker";
        return false;
    }
    if (workerCount > kMeshletStreamMaxPageLoadWorkers) {
        reason = "MeshletStreamPageLoader worker count exceeds the supported limit";
        return false;
    }

    impl_->asset = &asset;
    impl_->stopping = false;
    impl_->workers.reserve(workerCount);
    for (uint32_t worker = 0; worker < workerCount; ++worker) {
        impl_->workers.emplace_back([this] {
            impl_->runWorker();
        });
    }
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
    impl_->condition.notify_all();
    for (std::thread& worker : impl_->workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    impl_->workers.clear();
    {
        std::lock_guard lock(impl_->mutex);
        impl_->pendingPages.clear();
        impl_->completedLoads.clear();
        impl_->activeLoads = 0;
        impl_->asset = nullptr;
        impl_->stopping = false;
    }
}

bool MeshletStreamPageLoader::enqueue(uint32_t pageIndex)
{
    if (impl_ == nullptr) {
        return false;
    }
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->workers.empty() || impl_->stopping || impl_->asset == nullptr) {
            return false;
        }
        impl_->pendingPages.push_back(pageIndex);
    }
    impl_->condition.notify_one();
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
    return workerCount() != 0;
}

uint32_t MeshletStreamPageLoader::workerCount() const
{
    if (impl_ == nullptr) {
        return 0;
    }
    std::lock_guard lock(impl_->mutex);
    return static_cast<uint32_t>(impl_->workers.size());
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
