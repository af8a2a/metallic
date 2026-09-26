#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"

#include <memory>

namespace metallic::editor {

class RenderGraphExecutionViewer {
public:
    enum class Tab { Resources, Queues, Memory };

    void update(std::shared_ptr<const render::RenderGraphExecutionSnapshot> snapshot);
    void draw(float scale = 1.0f);
    void setTab(Tab tab) { tab_ = tab; selectTab_ = true; }
    void setLive(bool live) { live_ = live; }
    void requestCapture() { live_ = false; captureNext_ = true; }
    bool wantsCapture() const { return live_ || captureNext_; }
    const std::shared_ptr<const render::RenderGraphExecutionSnapshot>& snapshot() const { return snapshot_; }
    uint32_t selectedPassId() const { return selectedPassId_; }
    uint64_t selectedResourceId() const { return selectedResourceId_; }

private:
    void drawResources(float scale);
    void drawQueues(float scale);
    void drawMemory(float scale);
    void drawInspector();

    std::shared_ptr<const render::RenderGraphExecutionSnapshot> snapshot_;
    bool live_ = true;
    bool captureNext_ = false;
    bool selectTab_ = false;
    bool showBarriers_ = true;
    bool onlyUsed_ = false;
    bool fitQueues_ = true;
    int resourceType_ = 0;
    Tab tab_ = Tab::Resources;
    float columnWidth_ = 64.0f;
    float queueZoom_ = 1.0f;
    uint32_t selectedPassId_ = UINT32_MAX;
    uint64_t selectedResourceId_ = 0;
    char resourceFilter_[160]{};
    char passFilter_[160]{};
};

} // namespace metallic::editor
