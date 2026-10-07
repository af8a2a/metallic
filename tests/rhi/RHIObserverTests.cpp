#include "Runtime/Render/GAPI/RHIEvents.h"
#include "Runtime/Render/Profiling/SchedulingDiagnostics.h"
#include <gtest/gtest.h>
#include <thread>

namespace metallic::tests {
using namespace render;
using namespace render::profiling;

TEST(RHIObservers, NestedAndDisabledCapturesRestoreTheirOwner)
{
    SchedulingMetrics outer, inner;
    int command;
    const auto initial = rhiCommandObserver;
    {
        SchedulingCapture first(&outer);
        observeRHICommand(RHICommandEvent::BeginRendering, &command);
        observeRHICommand(RHICommandEvent::Draw, &command);
        {
            SchedulingCapture disabled(nullptr);
            observeRHICommand(RHICommandEvent::Draw, &command);
            EXPECT_EQ(rhiCommandObserver.notify, nullptr);
        }
        {
            SchedulingCapture second(&inner);
            observeRHICommand(RHICommandEvent::Dispatch, &command);
            observeRHICommand(RHICommandEvent::SubmitBegin);
            observeRHICommand(RHICommandEvent::SubmitEnd);
        }
        observeRHICommand(RHICommandEvent::Draw, &command);
        observeRHICommand(RHICommandEvent::EndRendering, &command);
    }
    EXPECT_EQ(rhiCommandObserver.context, initial.context);
    EXPECT_EQ(rhiCommandObserver.notify, initial.notify);
    EXPECT_EQ(outer.drawCalls, 2u);
    EXPECT_EQ(outer.renderingScopes, 1u);
    EXPECT_EQ(outer.maxScopeDrawCalls, 2u);
    EXPECT_EQ(outer.invalidScopes, 0u);
    EXPECT_EQ(outer.dispatchCalls, 0u);
    EXPECT_EQ(inner.dispatchCalls, 1u);
    EXPECT_EQ(inner.nativeSubmits, 1u);
}

TEST(RHIObservers, WorkerCaptureIsThreadLocal)
{
    SchedulingMetrics main, worker;
    SchedulingCapture capture(&main);
    std::jthread thread([&] {
        EXPECT_EQ(rhiCommandObserver.notify, nullptr);
        SchedulingCapture capture(&worker);
        observeRHICommand(RHICommandEvent::Dispatch);
    });
    observeRHICommand(RHICommandEvent::Dispatch);
    thread.join();
    EXPECT_EQ(main.dispatchCalls, 1u);
    EXPECT_EQ(worker.dispatchCalls, 1u);
}

TEST(RHIObservers, InvalidAndUnclosedRenderingScopesRemainVisible)
{
    SchedulingMetrics metrics;
    int command, other;
    {
        SchedulingCapture capture(&metrics);
        observeRHICommand(RHICommandEvent::Draw, &command);
        observeRHICommand(RHICommandEvent::BeginRendering, &command);
        observeRHICommand(RHICommandEvent::EndRendering, &other);
    }
    EXPECT_EQ(metrics.invalidScopes, 3u);
    EXPECT_EQ(metrics.renderingScopes, 0u);
}

namespace {
thread_local int begins = 0, ends = 0;
void beginOperation(RHIOperation, uint64_t value, void* storage)
{
    ++begins;
    std::construct_at(static_cast<uint64_t*>(storage), value);
}
void endOperation(RHIOperation, void* storage)
{
    EXPECT_EQ(*static_cast<uint64_t*>(storage), 17u);
    ++ends;
}
}
TEST(RHIObservers, OperationScopeRetainsItsOriginalObserver)
{
    static const RHIOperationObserver observer{beginOperation, endOperation};
    const auto* previous = rhiOperationObserver.exchange(&observer);
    begins = ends = 0;
    {
        RHIOperationScope scope(RHIOperation::Submit, 17);
        rhiOperationObserver.store(nullptr);
    }
    EXPECT_EQ(begins, 1);
    EXPECT_EQ(ends, 1);
    {
        RHIOperationScope disabled(RHIOperation::Submit, 17);
    }
    EXPECT_EQ(begins, 1);
    rhiOperationObserver.store(previous);
}
} // namespace metallic::tests
