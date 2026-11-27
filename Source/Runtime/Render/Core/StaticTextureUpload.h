#pragma once

#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render {

// Publish an immutable upload to both queues before discarding its staging resources.
inline Result<> submitStaticComputeUpload(Device& device, Queue& queue, CommandBuffer& commands,
    Queue* computeConsumer = nullptr)
{
    auto fence = device.createFence(false);
    if (!fence) { return makeError(fence.error()); }
    CommandBuffer* command = &commands;
    if (!computeConsumer || computeConsumer->sameQueue(queue)) {
        auto result = queue.submit({.commandBuffers = {&command, 1}, .signalFence = fence->get()});
        return result ? (*fence)->wait() : result;
    }

    auto ready = device.createSemaphore();
    if (!ready) { return makeError(ready.error()); }
    auto consumerFence = device.createFence(false);
    if (!consumerFence) { return makeError(consumerFence.error()); }
    const SemaphoreSubmitDesc publication{.semaphore = ready->get(), .value = 1};
    auto result = queue.submit({.commandBuffers = {&command, 1}, .signalSemaphores = {&publication, 1},
        .signalFence = fence->get()});
    if (!result) { return result; }
    auto visibility = computeConsumer->submit({.waitSemaphores = {&publication, 1}, .signalFence = consumerFence->get()});
    auto uploaded = (*fence)->wait();
    if (!visibility) { return visibility; }
    auto published = (*consumerFence)->wait();
    return uploaded ? published : uploaded;
}

// Initialize a single-mip, tightly packed lookup texture before its first compute dispatch.
inline Result<> uploadStaticComputeTexture(Device& device, Queue& queue, Texture& texture, Buffer& staging,
    Queue* computeConsumer = nullptr, PipelineStageBits shaderStages = PipelineStageBits::ComputeShader)
{
    auto source = staging.slice();
    if (!source) { return makeError(source.error()); }
    auto pool = device.createCommandPool(queue);
    if (!pool) { return makeError(pool.error()); }
    auto commands = (*pool)->createCommandBuffer();
    if (!commands) { return makeError(commands.error()); }
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
    barrier.after = {shaderStages, AccessBits::ShaderRead};
    result = (*commands)->synchronize({.textures = {&barrier, 1}});
    if (!result) { return result; }
    result = (*commands)->end();
    if (!result) { return result; }
    return submitStaticComputeUpload(device, queue, **commands, computeConsumer);
}

} // namespace metallic::render
