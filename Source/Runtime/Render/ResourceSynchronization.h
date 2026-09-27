#pragma once

#include "Runtime/Render/GAPI/rhi.h"

namespace metallic::render {

// Lower the renderer's coarse tracked usage at its synchronization boundary.
// Callers choose shader stages explicitly; the RHI never infers a scope.
constexpr SyncScope resourceSyncScope(ResourceState state, PipelineStageBits shaderStages)
{
    switch (state) {
    case ResourceState::Undefined:
    case ResourceState::Present: return {};
    case ResourceState::ColorAttachment:
        return {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite};
    case ResourceState::DepthStencilAttachment:
        return {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite};
    case ResourceState::ShaderRead: return {shaderStages, AccessBits::ShaderRead};
    case ResourceState::IndirectArgument: return {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead};
    case ResourceState::TransferSource: return {PipelineStageBits::Transfer, AccessBits::TransferRead};
    case ResourceState::TransferDestination: return {PipelineStageBits::Transfer, AccessBits::TransferWrite};
    case ResourceState::General: return {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
    case ResourceState::DecompressionSource: return {PipelineStageBits::MemoryDecompression, AccessBits::DecompressionRead};
    case ResourceState::DecompressionDestination: return {PipelineStageBits::MemoryDecompression, AccessBits::DecompressionWrite};
    }
    // Preserve invalid input for synchronize() to reject before recording.
    return {static_cast<PipelineStageBits>(UINT64_MAX), AccessBits::None};
}

constexpr TextureLayout textureLayoutForResourceState(ResourceState state)
{
    switch (state) {
    case ResourceState::Undefined: return TextureLayout::Undefined;
    case ResourceState::Present: return TextureLayout::Present;
    case ResourceState::ColorAttachment: return TextureLayout::ColorAttachment;
    case ResourceState::DepthStencilAttachment: return TextureLayout::DepthStencilAttachment;
    case ResourceState::ShaderRead: return TextureLayout::ShaderRead;
    case ResourceState::TransferSource: return TextureLayout::TransferSource;
    case ResourceState::TransferDestination: return TextureLayout::TransferDestination;
    case ResourceState::General: return TextureLayout::General;
    default: return static_cast<TextureLayout>(UINT8_MAX);
    }
}

} // namespace metallic::render
