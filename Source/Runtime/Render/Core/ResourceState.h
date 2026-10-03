#pragma once

#include <cstdint>

namespace metallic::render {

// Coarse renderer usage for graph/history tracking, not an RHI image layout.
// Lower synchronization to SyncScope and select TextureLayout in Core.
enum class ResourceState : uint8_t {
    Undefined,
    Present,
    ColorAttachment,
    DepthStencilAttachment,
    ShaderRead,
    IndirectArgument,
    TransferSource,
    TransferDestination,
    General,
    DecompressionSource,
    DecompressionDestination,
};

} // namespace metallic::render
