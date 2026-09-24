#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cassert>
#include <string>
#include <string_view>
#include <vector>

namespace metallic::render::detail {

// Logical profiling scopes can span a fork/join. Emit balanced native labels
// per recording to preserve the active scope ancestry on each queue.
// Commands is parameterized so tests can inspect exactly what Nsight receives.
template <typename Commands>
class RenderGraphGpuLabels {
public:
    RenderGraphGpuLabels(std::string_view name, ColorValue color) : root_{std::string(name), color} {}

    void resume(Commands& commands)
    {
        assert(commands_ == nullptr);
        commands_ = &commands;
        emit(root_);
        for (const auto& scope : scopes_) { emit(scope); }
    }

    void suspend()
    {
        if (!commands_) { return; }
        for (size_t i = 0; i < scopes_.size(); ++i) { commands_->endDebugLabel(); }
        commands_->endDebugLabel();
        commands_ = nullptr;
    }

    void begin(std::string_view name, ColorValue color)
    {
        assert(commands_ != nullptr);
        scopes_.push_back({std::string(name), color});
        emit(scopes_.back());
    }

    void end()
    {
        assert(!scopes_.empty());
        // Failure unwinds logical RAII scopes after the producer has ended.
        if (commands_) { commands_->endDebugLabel(); }
        scopes_.pop_back();
    }

private:
    struct Label {
        std::string name;
        ColorValue color;
    };

    void emit(const Label& label)
    {
        commands_->beginDebugLabel({.name = label.name.c_str(), .color = label.color});
    }

    Label root_;
    std::vector<Label> scopes_;
    Commands* commands_ = nullptr;
};

} // namespace metallic::render::detail
