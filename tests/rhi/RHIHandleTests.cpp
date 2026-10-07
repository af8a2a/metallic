// Only public interfaces: no complete backend Impl or Vulkan headers.
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"

#include <gtest/gtest.h>
#include <type_traits>

namespace metallic::tests {
namespace {

using namespace render;

template <typename T>
class RHIEmptyHandle : public testing::Test {};

using PublicHandles = testing::Types<Queue, Fence, Semaphore, SwapchainSemaphore,
    Buffer, BufferView, TimestampQueryPool, RayTracingAccelerationStructureCompactionQueryPool,
    RayTracingBottomLevelBuildPlan, RayTracingAccelerationStructure, Texture, TextureView, ShaderModule, PipelineCache,
    GraphicsPipeline, ComputePipeline, GraphicsShaderObjectProgram, BindlessHeap,
    CommandBuffer, CommandPool, Swapchain, Device, Streamer>;
TYPED_TEST_SUITE(RHIEmptyHandle, PublicHandles);

TYPED_TEST(RHIEmptyHandle, SupportsOpaqueConstructionAndMoveOnlyLifetime)
{
    static_assert(std::is_nothrow_default_constructible_v<TypeParam>);
    static_assert(std::is_nothrow_move_constructible_v<TypeParam>);
    static_assert(std::is_nothrow_move_assignable_v<TypeParam>);
    static_assert(std::is_nothrow_destructible_v<TypeParam>);
    static_assert(!std::is_copy_constructible_v<TypeParam>);
    static_assert(!std::is_copy_assignable_v<TypeParam>);
    static_assert(!std::is_polymorphic_v<TypeParam>);
    TypeParam source;
    TypeParam target(std::move(source));
    TypeParam destination;
    destination = std::move(target);
    auto& self = destination;
    destination = std::move(self);
}

} // namespace

// Use counted Impl owners to verify native-like destruction deterministically,
// without relying on the driver to report leaked objects during device teardown.
namespace handle_test {
namespace detail {
struct UniqueProbeImpl {
    int& destroyed;
    ~UniqueProbeImpl() { ++destroyed; }
};
struct SharedProbeImpl {
    int& destroyed;
    ~SharedProbeImpl() { ++destroyed; }
};
} // namespace detail

struct Factory;

class UniqueProbe {
    METALLIC_RHI_HANDLE(UniqueProbe, unique_ptr, friend struct Factory;)
    bool valid() const { return impl_ != nullptr; }
};

class SharedProbe {
    METALLIC_RHI_HANDLE(SharedProbe, shared_ptr, friend struct Factory;)
    std::shared_ptr<void> retain() const { return impl_; }
    bool valid() const { return impl_ != nullptr; }
};

METALLIC_RHI_HANDLE_DEFINITIONS(UniqueProbe)
METALLIC_RHI_HANDLE_DEFINITIONS(SharedProbe)

struct Factory {
    static UniqueProbe unique(int& destroyed)
    {
        return UniqueProbe(std::make_unique<detail::UniqueProbeImpl>(destroyed));
    }
    static SharedProbe shared(int& destroyed)
    {
        return SharedProbe(std::make_unique<detail::SharedProbeImpl>(destroyed));
    }
};

static_assert(sizeof(UniqueProbe) == sizeof(std::unique_ptr<detail::UniqueProbeImpl>));
static_assert(sizeof(SharedProbe) == sizeof(std::shared_ptr<detail::SharedProbeImpl>));
static_assert(!std::is_constructible_v<UniqueProbe, std::unique_ptr<detail::UniqueProbeImpl>>);
static_assert(!std::is_constructible_v<SharedProbe, std::unique_ptr<detail::SharedProbeImpl>>);

TEST(RHIHandleOwnership, LiveUniqueMoveDestroysOldImplExactlyOnce)
{
    int oldDestroyed = 0, newDestroyed = 0;
    {
        auto destination = Factory::unique(oldDestroyed);
        auto source = Factory::unique(newDestroyed);
        destination = std::move(source);
        EXPECT_EQ(oldDestroyed, 1);
        EXPECT_EQ(newDestroyed, 0);
        EXPECT_FALSE(source.valid());
        EXPECT_TRUE(destination.valid());
        auto& self = destination;
        destination = std::move(self);
        EXPECT_EQ(newDestroyed, 0);
        EXPECT_TRUE(destination.valid());
        source = std::move(destination);
        EXPECT_FALSE(destination.valid());
        EXPECT_TRUE(source.valid());
    }
    EXPECT_EQ(oldDestroyed, 1);
    EXPECT_EQ(newDestroyed, 1);
}

TEST(RHIHandleOwnership, SharedMovePreservesRetainedAllocationAfterWrapperDies)
{
    int oldDestroyed = 0, newDestroyed = 0;
    std::shared_ptr<void> retainedOld, retainedNew;
    {
        auto destination = Factory::shared(oldDestroyed);
        auto source = Factory::shared(newDestroyed);
        retainedOld = destination.retain();
        retainedNew = source.retain();
        destination = std::move(source);
        EXPECT_FALSE(source.valid());
        EXPECT_EQ(destination.retain(), retainedNew);
        EXPECT_EQ(oldDestroyed, 0);
        EXPECT_EQ(newDestroyed, 0);
        auto& self = destination;
        destination = std::move(self);
        EXPECT_EQ(destination.retain(), retainedNew);
        retainedOld.reset();
        EXPECT_EQ(oldDestroyed, 1);
    }
    EXPECT_EQ(newDestroyed, 0);
    retainedNew.reset();
    EXPECT_EQ(oldDestroyed, 1);
    EXPECT_EQ(newDestroyed, 1);
}

} // namespace handle_test
} // namespace metallic::tests
