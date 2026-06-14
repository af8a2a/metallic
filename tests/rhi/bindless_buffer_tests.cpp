#include "rhi_test.h"

#include "Runtime/Render/slang_compiler.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::tests {
namespace {

constexpr const char* kShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kBindlessBufferShaderModuleName = "bindless_buffer";

struct BindlessBufferUserPush {
    uint32_t inputBuffer = 0;
    uint32_t outputBuffer = 0;
    uint32_t passIndex = 0;
    uint32_t padding = 0;
};

struct BindlessDeviceSetup {
    std::unique_ptr<render::Device> device;
    render::Queue* computeQueue = nullptr;
};

RhiTestResult setupBindlessDevice(bool enableValidation, BindlessDeviceSetup& setup)
{
    setup = {};

    render::Result result = render::createDevice(
        render::DeviceDesc{
            .applicationName = "Metallic RHI Bindless Buffer Test",
            .enableValidation = enableValidation,
            .enableBindlessDescriptorHeap = true,
        },
        setup.device);
    if (!result) {
        if (render::hasError(result, render::Error::Unsupported)) {
            return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
        }
        return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
    }

    if (!setup.device->capabilities().bindlessDescriptorHeap) {
        return RhiTestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
    }

    setup.computeQueue = setup.device->getQueue(render::QueueType::Compute);
    if (setup.computeQueue == nullptr) {
        return RhiTestResult::skip("bindless test device has no compute queue");
    }

    return RhiTestResult::pass();
}

RhiTestResult createShaderModule(
    render::Device& device,
    const char* entryPointName,
    std::unique_ptr<render::ShaderModule>& outShaderModule)
{
    render::ShaderCompileResult compileResult;
    render::Result result = render::compileSlangShaderToSpirv(
        render::SlangShaderDesc{
            .moduleName = kBindlessBufferShaderModuleName,
            .entryPointName = entryPointName,
            .searchPath = kShaderSearchPath,
        },
        compileResult);
    if (!result) {
        std::string message = std::string("compileSlangShaderToSpirv(") + entryPointName + ") returned ";
        message += toString(result);
        if (!compileResult.diagnostics.empty()) {
            message += ": ";
            message += compileResult.diagnostics;
        }
        return RhiTestResult::fail(std::move(message));
    }

    result = device.createShaderModule(
        render::ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
        },
        outShaderModule);
    if (!result || outShaderModule == nullptr) {
        return RhiTestResult::fail(std::string("createShaderModule returned ") + toString(result));
    }

    return RhiTestResult::pass();
}

RhiTestResult createBuffer(
    render::Device& device,
    const render::BufferDesc& desc,
    const char* label,
    std::unique_ptr<render::Buffer>& outBuffer)
{
    render::Result result = device.createBuffer(desc, outBuffer);
    if (!result || outBuffer == nullptr) {
        return RhiTestResult::fail(std::string("createBuffer(") + label + ") returned " + toString(result));
    }
    return RhiTestResult::pass();
}

RhiTestResult createCommandObjects(
    render::Device& device,
    render::Queue& queue,
    std::unique_ptr<render::CommandPool>& outCommandPool,
    std::unique_ptr<render::CommandBuffer>& outCommandBuffer,
    std::unique_ptr<render::Fence>& outFence)
{
    render::Result result = device.createCommandPool(queue, outCommandPool);
    if (!result || outCommandPool == nullptr) {
        return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
    }

    result = outCommandPool->createCommandBuffer(outCommandBuffer);
    if (!result || outCommandBuffer == nullptr) {
        return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
    }

    result = device.createFence(false, outFence);
    if (!result || outFence == nullptr) {
        return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
    }

    return RhiTestResult::pass();
}

RhiTestResult submitAndWait(render::Queue& queue, render::CommandBuffer& commandBuffer, render::Fence& fence)
{
    render::CommandBuffer* commandBuffers[] = {&commandBuffer};
    render::Result result = queue.submit(
        render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = &fence,
        });
    if (!result) {
        return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
    }

    result = fence.wait(5'000'000'000ull);
    if (!result) {
        return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
    }

    return RhiTestResult::pass();
}

RhiTestResult readBufferBytes(render::Buffer& buffer, uint64_t byteSize, std::vector<uint8_t>& outBytes)
{
    buffer.invalidate(0, byteSize);
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return RhiTestResult::fail("readback buffer did not map");
    }

    outBytes.resize(static_cast<size_t>(byteSize));
    std::memcpy(outBytes.data(), mapped, outBytes.size());
    buffer.unmap();
    return RhiTestResult::pass();
}

bool equalBytes(const std::vector<uint8_t>& actual, const uint8_t* expected, size_t expectedSize)
{
    return actual.size() >= expectedSize &&
        std::memcmp(actual.data(), expected, expectedSize) == 0;
}

class BindlessBufferConstantReadTest : public RhiTest {
public:
    BindlessBufferConstantReadTest()
    {
        type = RhiTestType::Command;
        name = "bindless_buffer_constant_read";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        BindlessDeviceSetup setup;
        RhiTestResult testResult = setupBindlessDevice(context.enableValidation, setup);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::Buffer> constantBuffer;
        testResult = createBuffer(
            *setup.device,
            render::BufferDesc{
                .size = 256,
                .structureStride = 16,
                .usage = render::BufferUsageBits::Constant,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            "constant",
            constantBuffer);
        if (!testResult.passed) {
            return testResult;
        }

        constexpr std::array<uint32_t, 4> kInputWords = {
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };
        void* mappedConstant = constantBuffer->map();
        if (mappedConstant == nullptr) {
            return RhiTestResult::fail("constant buffer did not map");
        }
        std::memcpy(mappedConstant, kInputWords.data(), kInputWords.size() * sizeof(uint32_t));
        constantBuffer->flush(0, kInputWords.size() * sizeof(uint32_t));
        constantBuffer->unmap();

        std::unique_ptr<render::Buffer> outputBuffer;
        testResult = createBuffer(
            *setup.device,
            render::BufferDesc{
                .size = kInputWords.size() * sizeof(uint32_t),
                .structureStride = sizeof(uint32_t),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "output",
            outputBuffer);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::BindlessHeap> bindlessHeap;
        render::Result result = setup.device->createBindlessHeap(
            render::BindlessHeapDesc{
                .maxSamplers = 0,
                .maxSampledImages = 0,
                .maxBuffers = 2,
            },
            bindlessHeap);
        if (!result || bindlessHeap == nullptr) {
            return RhiTestResult::fail(std::string("createBindlessHeap returned ") + toString(result));
        }

        render::BindlessHandle constantHandle;
        result = bindlessHeap->allocateBuffer(constantHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(constant) returned ") + toString(result));
        }
        render::BindlessHandle outputHandle;
        result = bindlessHeap->allocateBuffer(outputHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(output) returned ") + toString(result));
        }

        result = bindlessHeap->writeConstantBuffer(constantHandle, *constantBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeConstantBuffer returned ") + toString(result));
        }
        result = bindlessHeap->writeStorageBuffer(outputHandle, *outputBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeStorageBuffer(output) returned ") + toString(result));
        }

        std::unique_ptr<render::ShaderModule> shader;
        testResult = createShaderModule(*setup.device, "bindlessBufferConstantMain", shader);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::ComputePipeline> pipeline;
        result = setup.device->createComputePipeline(
            render::ComputePipelineDesc{
                .computeShader = shader.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(BindlessBufferUserPush),
            },
            pipeline);
        if (!result || pipeline == nullptr) {
            return RhiTestResult::fail(std::string("createComputePipeline returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        testResult = createCommandObjects(*setup.device, *setup.computeQueue, commandPool, commandBuffer, fence);
        if (!testResult.passed) {
            return testResult;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        const BindlessBufferUserPush push{
            .inputBuffer = constantHandle.index,
            .outputBuffer = outputHandle.index,
            .passIndex = 0,
        };
        commandBuffer->bindBindlessHeap(*bindlessHeap);
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->bindComputePipeline(*pipeline);
        commandBuffer->dispatch(1, 1, 1);

        render::BufferBarrierDesc outputBarrier{
            .buffer = outputBuffer.get(),
            .before = render::ResourceState::General,
            .after = render::ResourceState::General,
            .offset = 0,
            .size = outputBuffer->desc().size,
        };
        commandBuffer->barrier(render::BarrierDesc{.buffers = &outputBarrier, .bufferCount = 1});

        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        testResult = submitAndWait(*setup.computeQueue, *commandBuffer, *fence);
        if (!testResult.passed) {
            return testResult;
        }

        std::vector<uint8_t> readback;
        testResult = readBufferBytes(*outputBuffer, outputBuffer->desc().size, readback);
        if (!testResult.passed) {
            return testResult;
        }

        if (!equalBytes(readback, reinterpret_cast<const uint8_t*>(kInputWords.data()), kInputWords.size() * sizeof(uint32_t))) {
            return RhiTestResult::fail("bindless constant buffer readback mismatch");
        }

        (void)setup.device->waitIdle();
        return RhiTestResult::pass();
    }
};

class BindlessBufferRwByteAddressTest : public RhiTest {
public:
    BindlessBufferRwByteAddressTest()
    {
        type = RhiTestType::Command;
        name = "bindless_buffer_rwbyteaddress_write";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        BindlessDeviceSetup setup;
        RhiTestResult testResult = setupBindlessDevice(context.enableValidation, setup);
        if (!testResult.passed) {
            return testResult;
        }

        constexpr uint64_t kByteSize = 28;
        std::unique_ptr<render::Buffer> rwBuffer;
        testResult = createBuffer(
            *setup.device,
            render::BufferDesc{
                .size = kByteSize,
                .structureStride = sizeof(uint32_t),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "rw",
            rwBuffer);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::Buffer> outputBuffer;
        testResult = createBuffer(
            *setup.device,
            render::BufferDesc{
                .size = kByteSize,
                .structureStride = sizeof(uint32_t),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            "output",
            outputBuffer);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::BindlessHeap> bindlessHeap;
        render::Result result = setup.device->createBindlessHeap(
            render::BindlessHeapDesc{
                .maxSamplers = 0,
                .maxSampledImages = 0,
                .maxBuffers = 2,
            },
            bindlessHeap);
        if (!result || bindlessHeap == nullptr) {
            return RhiTestResult::fail(std::string("createBindlessHeap returned ") + toString(result));
        }

        render::BindlessHandle rwHandle;
        result = bindlessHeap->allocateBuffer(rwHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(rw) returned ") + toString(result));
        }
        render::BindlessHandle outputHandle;
        result = bindlessHeap->allocateBuffer(outputHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(output) returned ") + toString(result));
        }

        result = bindlessHeap->writeStorageBuffer(rwHandle, *rwBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeStorageBuffer(rw) returned ") + toString(result));
        }
        result = bindlessHeap->writeStorageBuffer(outputHandle, *outputBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeStorageBuffer(output) returned ") + toString(result));
        }

        std::unique_ptr<render::ShaderModule> shader;
        testResult = createShaderModule(*setup.device, "bindlessBufferRwByteAddressMain", shader);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::ComputePipeline> pipeline;
        result = setup.device->createComputePipeline(
            render::ComputePipelineDesc{
                .computeShader = shader.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(BindlessBufferUserPush),
            },
            pipeline);
        if (!result || pipeline == nullptr) {
            return RhiTestResult::fail(std::string("createComputePipeline returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        testResult = createCommandObjects(*setup.device, *setup.computeQueue, commandPool, commandBuffer, fence);
        if (!testResult.passed) {
            return testResult;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        commandBuffer->bindComputePipeline(*pipeline);
        commandBuffer->bindBindlessHeap(*bindlessHeap);

        BindlessBufferUserPush push{
            .inputBuffer = rwHandle.index,
            .outputBuffer = outputHandle.index,
            .passIndex = 0,
        };
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        render::BufferBarrierDesc rwBarrier{
            .buffer = rwBuffer.get(),
            .before = render::ResourceState::General,
            .after = render::ResourceState::General,
            .offset = 0,
            .size = rwBuffer->desc().size,
        };
        commandBuffer->barrier(render::BarrierDesc{.buffers = &rwBarrier, .bufferCount = 1});

        push.passIndex = 1;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->dispatch(1, 1, 1);

        render::BufferBarrierDesc outputBarrier{
            .buffer = outputBuffer.get(),
            .before = render::ResourceState::General,
            .after = render::ResourceState::General,
            .offset = 0,
            .size = outputBuffer->desc().size,
        };
        commandBuffer->barrier(render::BarrierDesc{.buffers = &outputBarrier, .bufferCount = 1});

        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        testResult = submitAndWait(*setup.computeQueue, *commandBuffer, *fence);
        if (!testResult.passed) {
            return testResult;
        }

        const std::array<uint32_t, 7> expectedWords = {
            0xDEADBEEFu,
            0x11223344u,
            0xAABBCCDDu,
            1u,
            2u,
            3u,
            4u,
        };
        std::vector<uint8_t> readback;
        testResult = readBufferBytes(*outputBuffer, outputBuffer->desc().size, readback);
        if (!testResult.passed) {
            return testResult;
        }

        if (!equalBytes(readback, reinterpret_cast<const uint8_t*>(expectedWords.data()), expectedWords.size() * sizeof(uint32_t))) {
            return RhiTestResult::fail("bindless RWByteAddressBuffer readback mismatch");
        }

        (void)setup.device->waitIdle();
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(BindlessBufferConstantReadTest);
METALLIC_REGISTER_RHI_TEST(BindlessBufferRwByteAddressTest);

} // namespace
} // namespace metallic::tests
