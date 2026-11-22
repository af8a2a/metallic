#pragma once

#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render {

// Initialize a single-mip, tightly packed lookup texture before its first compute dispatch.
inline Result<> uploadStaticComputeTexture(Device& device, Queue& queue, Texture& texture, Buffer& staging)
{
    auto source = staging.slice();
    if (!source) { return makeError(source.error()); }
    auto pool = device.createCommandPool(queue);
    if (!pool) { return makeError(pool.error()); }
    auto commands = (*pool)->createCommandBuffer();
    if (!commands) { return makeError(commands.error()); }
    auto fence = device.createFence(false);
    if (!fence) { return makeError(fence.error()); }
    auto result = (*commands)->begin();
    if (!result) { return result; }

    TextureBarrierDesc barrier{.texture = &texture, .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::TransferDestination,
        .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
    result = (*commands)->synchronize({.textures = {&barrier, 1}});
    if (!result) { return result; }
    const auto& extent = texture.desc();
    result = (*commands)->copyBufferToTexture({.texture = &texture, .buffer = *source,
        .width = extent.width, .height = extent.height, .depth = extent.depth});
    if (!result) { return result; }
    barrier.oldLayout = TextureLayout::TransferDestination;
    barrier.newLayout = TextureLayout::ShaderRead;
    barrier.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite};
    barrier.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
    result = (*commands)->synchronize({.textures = {&barrier, 1}});
    if (!result) { return result; }
    result = (*commands)->end();
    if (!result) { return result; }
    CommandBuffer* command = commands->get();
    result = queue.submit({.commandBuffers = {&command, 1}, .signalFence = fence->get()});
    return result ? (*fence)->wait() : result;
}

} // namespace metallic::render
