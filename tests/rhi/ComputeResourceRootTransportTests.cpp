#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeResourceEncoder.h"
#include "Runtime/Render/Core/ComputeResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Material/MaterialExecutable.h"

#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

using namespace render;

struct RootTransportResources {
    ShaderBuffer source;
};

static_assert(sizeof(RootTransportResources) == 4);
static_assert(ResourceComputeKernelDesc{}.rootTransport == ParameterTransport::DescriptorBuffer);

#define ROOT_CHECK(expression) do { \
    if (!(expression)) { return RHITestResult::fail(#expression); } \
} while (false)

class ComputeResourceRootTransportTest final : public RHITest {
public:
    ComputeResourceRootTransportTest()
    {
        type = RHITestType::Resource;
        name = "compute_resource_root_transport_encoding_and_abi";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.named.root.transport.abi", "compute.named.root.transport.encoding"},
            bench::Layer::Core, "binding", "binding");
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto fixture = bench::createTestDevice(context, {.applicationName = "Compute resource root transport",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        ROOT_CHECK(fixture);
        auto& device = **fixture;
        const ComputeResourceBindingDesc binding{
            .binding = METALLIC_RESOURCE_MEMBER(RootTransportResources, source)};
        ResourceComputeKernelDesc desc{.pushConstantSize = sizeof(uint32_t), .bindings = {&binding, 1},
            .requiresRayQuery = false, .resourceParameterSize = sizeof(RootTransportResources)};
        ComputeResourceEncoder buffered, direct, equivalent, invalid;
        std::string log;
        ROOT_CHECK(buffered.initialize(device, desc, log));
        ROOT_CHECK(buffered.parameterABI().transport == ParameterTransport::DescriptorBuffer);
        desc.rootTransport = ParameterTransport::InlinePush;
        ROOT_CHECK(direct.initialize(device, desc, log));
        ROOT_CHECK(equivalent.initialize(device, desc, log));
        ROOT_CHECK(direct.parameterABI().transport == ParameterTransport::InlinePush);
        ROOT_CHECK(direct.parameterABI().size == sizeof(ComputeResourceParameters));
        ROOT_CHECK(direct.parameterABI().id != buffered.parameterABI().id);
        ROOT_CHECK(direct.parameterABI() != buffered.parameterABI());
        ROOT_CHECK(!direct.compatible(buffered) && !buffered.compatible(direct));
        ROOT_CHECK(direct.compatible(equivalent) && direct.parameterABI() == equivalent.parameterABI());
        desc.rootTransport = static_cast<ParameterTransport>(0xffu);
        ROOT_CHECK(hasError(invalid.initialize(device, desc, log), Error::InvalidArgument) && !invalid.valid());

        auto source = device.createBuffer({.size = 16, .structureStride = 4,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
        ROOT_CHECK(source);
        auto arguments = device.createBuffer({.size = 24, .structureStride = 4,
            .usage = BufferUsageBits::Indirect, .memoryLocation = MemoryLocation::HostUpload});
        ROOT_CHECK(arguments);
        auto registry = ResourceRegistry::forDevice(device);
        ROOT_CHECK(registry);
        const ComputeDispatchBinding input{.binding = binding.binding, .buffer = source->get()};
        const uint32_t constants = 7;
        const ComputeDispatchDesc dispatch{.bindings = {&input, 1}, .pushData = &constants,
            .pushDataSize = sizeof(constants)};
        const auto beforeBuffered = (*registry)->stats().parameterBytes;
        auto bufferedPackets = buffered.encode(nullptr, dispatch);
        ROOT_CHECK(bufferedPackets && bufferedPackets->size() == 1);
        const auto bufferedBytes = (*registry)->stats().parameterBytes - beforeBuffered;
        ROOT_CHECK(bufferedPackets->front().inlineData().empty());
        ROOT_CHECK(bufferedPackets->front().root().count == sizeof(ComputeResourceParameters) / 4);
        ROOT_CHECK(bufferedPackets->front().abi() == buffered.parameterABI());
        const auto beforeDirect = (*registry)->stats().parameterBytes;
        auto directPackets = direct.encode(nullptr, dispatch);
        ROOT_CHECK(directPackets && directPackets->size() == 1);
        const auto directBytes = (*registry)->stats().parameterBytes - beforeDirect;
        ROOT_CHECK(bufferedBytes == directBytes + sizeof(ComputeResourceParameters));
        ROOT_CHECK(directPackets->front().abi() == direct.parameterABI());
        ROOT_CHECK(directPackets->front().root() == GPUBufferSpan{});
        ROOT_CHECK(directPackets->front().inlineData().size() == sizeof(ComputeResourceParameters));
        ComputeResourceParameters root;
        std::memcpy(&root, directPackets->front().inlineData().data(), sizeof(root));
        ROOT_CHECK(root.resources.count == sizeof(RootTransportResources) / 4);
        ROOT_CHECK(root.constants.count == sizeof(constants) / 4);
        ROOT_CHECK(root.resources.resource.index != UINT32_MAX && root.constants.resource.index != UINT32_MAX);

        const std::array<uint32_t, 2> batchConstants{11, 13};
        const ComputeIndirectDispatch items[] = {
            {.pushData = &batchConstants[0], .argumentOffset = 0},
            {.pushData = &batchConstants[1], .argumentOffset = 12}};
        auto indirect = dispatch;
        indirect.indirectArguments = arguments->get();
        auto batch = direct.encode(nullptr, indirect, items);
        ROOT_CHECK(batch && batch->size() == 2);
        std::array<ComputeResourceParameters, 2> roots;
        for (uint32_t i = 0; i < roots.size(); ++i) {
            ROOT_CHECK((*batch)[i].abi() == direct.parameterABI());
            ROOT_CHECK((*batch)[i].inlineData().size() == sizeof(roots[i]));
            std::memcpy(&roots[i], (*batch)[i].inlineData().data(), sizeof(roots[i]));
        }
        ROOT_CHECK(roots[0].resources == roots[1].resources);
        ROOT_CHECK(roots[0].constants.resource == roots[1].constants.resource);
        ROOT_CHECK(roots[0].constants.count == 1 && roots[1].constants.count == 1);
        ROOT_CHECK(roots[1].constants.byteOffset == roots[0].constants.byteOffset + 16);

        // The direct root still owns the transitive resource allocation. The
        // shared prepared-dispatch test covers actual direct/indirect execution.
        std::weak_ptr<void> sourceLife = (*source)->retainAllocation();
        source->reset();
        bufferedPackets->clear();
        directPackets->clear();
        ROOT_CHECK(!sourceLife.expired());
        batch->clear();
        ROOT_CHECK(sourceLife.expired());
        return RHITestResult::pass("Root transport ABI separation, 24-byte inline encoding, indirect constant slices and resource retention");
    }
};

METALLIC_REGISTER_RHI_TEST(ComputeResourceRootTransportTest);

class ComputeResourceRootTransportCacheTest final : public RHITest {
public:
    ComputeResourceRootTransportCacheTest()
    {
        type = RHITestType::Resource;
        name = "compute_resource_root_transport_material_cache";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.named.root.transport.materialCache"},
            bench::Layer::Core, "binding", "binding");
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto fixture = bench::createTestDevice(context, {.applicationName = "Compute root transport cache",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        ROOT_CHECK(fixture);
        auto& device = **fixture;
        // The empty entry has no root dependency, so identical SPIR-V is valid
        // with either transport. The cache must distinguish the CPU contract.
        const SlangShaderDesc source{.moduleName = "ComputeRootTransportCacheProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"};
        ResourceComputeKernelDesc layout{.requiresRayQuery = false, .resourceParameterSize = sizeof(uint32_t)};
        std::array<ComputeKernel, 2> kernels;
        std::array<std::shared_ptr<const MaterialExecutableArtifact>, 2> artifacts;
        std::string log;
        for (uint32_t i = 0; i < kernels.size(); ++i) {
            layout.rootTransport = i == 0 ? ParameterTransport::DescriptorBuffer : ParameterTransport::InlinePush;
            ROOT_CHECK(compileMaterialExecutable(device, source, layout, kernels[i], artifacts[i], log));
            ROOT_CHECK(artifacts[i]->rootTransport == layout.rootTransport);
            ROOT_CHECK(artifacts[i]->encoder.parameterABI() == kernels[i].parameterABI());
            ComputeKernel reused;
            std::shared_ptr<const MaterialExecutableArtifact> cached;
            ROOT_CHECK(compileMaterialExecutable(device, source, layout, reused, cached, log));
            ROOT_CHECK(cached == artifacts[i] && reused.parameterABI() == kernels[i].parameterABI());
        }
        ROOT_CHECK(artifacts[0]->shader.spirv == artifacts[1]->shader.spirv);
        ROOT_CHECK(artifacts[0] != artifacts[1]);
        ROOT_CHECK(artifacts[0]->programKey.specializationSignature != artifacts[1]->programKey.specializationSignature);
        ROOT_CHECK(kernels[0].parameterABI() != kernels[1].parameterABI());
        ROOT_CHECK(!artifacts[0]->encoder.compatible(artifacts[1]->encoder));
        auto arguments = device.createBuffer({.size = 12, .structureStride = 4,
            .usage = BufferUsageBits::Indirect, .memoryLocation = MemoryLocation::HostUpload});
        ROOT_CHECK(arguments);
        auto argumentSlice = (*arguments)->slice();
        ROOT_CHECK(argumentSlice);
        for (uint32_t i = 0; i < kernels.size(); ++i) {
            auto packets = artifacts[i]->encoder.encode(nullptr, {});
            ROOT_CHECK(packets && packets->size() == 1);
            ROOT_CHECK(kernels[i].prepareDispatch(packets->front(), 1));
            ROOT_CHECK(hasError(kernels[i ^ 1u].prepareDispatch(packets->front(), 1), Error::InvalidArgument));
            const ComputeIndirectParameters item{packets->front(), *argumentSlice};
            ROOT_CHECK(kernels[i].prepareIndirectBatch({&item, 1}));
            ROOT_CHECK(hasError(kernels[i ^ 1u].prepareIndirectBatch({&item, 1}), Error::InvalidArgument));
        }
        return RHITestResult::pass("Identical code retains separate root transport artifacts and reuses each matching contract");
    }
};

METALLIC_REGISTER_RHI_TEST(ComputeResourceRootTransportCacheTest);

#undef ROOT_CHECK

} // namespace
} // namespace metallic::tests
