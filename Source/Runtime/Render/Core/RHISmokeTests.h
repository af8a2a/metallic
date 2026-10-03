#pragma once

#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render {
namespace detail { struct TrianglePreviewRendererImpl; }

int runRhiSmokeTest(bool enableValidation);
int runRhiTrianglePreviewTest(bool enableValidation);
int runRhiBindlessDescriptorHeapSmokeTest(bool enableValidation);

class TrianglePreviewRenderer {
public:
    TrianglePreviewRenderer();
    ~TrianglePreviewRenderer();

    TrianglePreviewRenderer(TrianglePreviewRenderer&&) noexcept;
    TrianglePreviewRenderer& operator=(TrianglePreviewRenderer&&) noexcept;

    TrianglePreviewRenderer(const TrianglePreviewRenderer&) = delete;
    TrianglePreviewRenderer& operator=(const TrianglePreviewRenderer&) = delete;

    Result<> initialize(bool enableValidation = false);
    Result<> render(uint32_t width, uint32_t height);
    const std::vector<uint32_t>& pixels() const;
    uint32_t width() const;
    uint32_t height() const;

private:
    std::unique_ptr<detail::TrianglePreviewRendererImpl> impl_;
};

} // namespace metallic::render
