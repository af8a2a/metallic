#pragma once

#include <memory>
#include <vector>

namespace metallic::render {
class TextureView;

// Publish through shared_ptr<const ...> and never mutate afterwards. The owner
// retains the underlying images, while views supply stable ownership identities.
struct SampledImageSnapshot {
    std::shared_ptr<const void> owner;
    std::vector<std::shared_ptr<TextureView>> views;
};

} // namespace metallic::render
