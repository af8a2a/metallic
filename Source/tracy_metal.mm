// Tracy Metal GPU profiling bridge — ObjC++ implementation.
// This file must be compiled as ObjC++ (.mm) with ARC enabled.

#include "tracy_metal.h"

#ifdef TRACY_ENABLE

#include <tracy/Tracy.hpp>
#include <tracy/TracyMetal.hmm>

// metal-cpp types are toll-free bridged with ObjC Metal types.
// We cast between them since they share the same binary layout.
// TracyMetalSrcLoc is layout-compatible with tracy::SourceLocationData.
static_assert(sizeof(TracyMetalSrcLoc) == sizeof(tracy::SourceLocationData));

TracyMetalCtxHandle tracyMetalCreate(MTL::Device* device) {
    auto* ctx = TracyMetalContext((__bridge id<MTLDevice>)device);
    return static_cast<TracyMetalCtxHandle>(ctx);
}

void tracyMetalDestroy(TracyMetalCtxHandle ctx) {
    TracyMetalDestroy(static_cast<TracyMetalCtx*>(ctx));
}

void tracyMetalCollect(TracyMetalCtxHandle ctx) {
    TracyMetalCollect(static_cast<TracyMetalCtx*>(ctx));
}

TracyMetalGpuZone tracyMetalZoneBeginRender(TracyMetalCtxHandle ctx,
                                             MTL::RenderPassDescriptor* desc,
                                             const TracyMetalSrcLoc* srcloc) {
    auto* metalCtx = static_cast<tracy::MetalCtx*>(ctx);
    auto* objcDesc = (__bridge MTLRenderPassDescriptor*)desc;
    auto* zone = new tracy::MetalZoneScope(metalCtx, objcDesc,
        reinterpret_cast<const tracy::SourceLocationData*>(srcloc), true);
    return static_cast<TracyMetalGpuZone>(zone);
}

TracyMetalGpuZone tracyMetalZoneBeginCompute(TracyMetalCtxHandle ctx,
                                              MTL::ComputePassDescriptor* desc,
                                              const TracyMetalSrcLoc* srcloc) {
    auto* metalCtx = static_cast<tracy::MetalCtx*>(ctx);
    auto* objcDesc = (__bridge MTLComputePassDescriptor*)desc;
    auto* zone = new tracy::MetalZoneScope(metalCtx, objcDesc,
        reinterpret_cast<const tracy::SourceLocationData*>(srcloc), true);
    return static_cast<TracyMetalGpuZone>(zone);
}

TracyMetalGpuZone tracyMetalZoneBeginBlit(TracyMetalCtxHandle ctx,
                                           MTL::BlitPassDescriptor* desc,
                                           const TracyMetalSrcLoc* srcloc) {
    auto* metalCtx = static_cast<tracy::MetalCtx*>(ctx);
    auto* objcDesc = (__bridge MTLBlitPassDescriptor*)desc;
    auto* zone = new tracy::MetalZoneScope(metalCtx, objcDesc,
        reinterpret_cast<const tracy::SourceLocationData*>(srcloc), true);
    return static_cast<TracyMetalGpuZone>(zone);
}

void tracyMetalZoneEnd(TracyMetalGpuZone zone) {
    auto* scope = static_cast<tracy::MetalZoneScope*>(zone);
    delete scope;
}

#else // !TRACY_ENABLE

TracyMetalCtxHandle tracyMetalCreate(MTL::Device*) { return nullptr; }
void tracyMetalDestroy(TracyMetalCtxHandle) {}
void tracyMetalCollect(TracyMetalCtxHandle) {}

TracyMetalGpuZone tracyMetalZoneBeginRender(TracyMetalCtxHandle, MTL::RenderPassDescriptor*,
                                             const TracyMetalSrcLoc*) { return nullptr; }
TracyMetalGpuZone tracyMetalZoneBeginCompute(TracyMetalCtxHandle, MTL::ComputePassDescriptor*,
                                              const TracyMetalSrcLoc*) { return nullptr; }
TracyMetalGpuZone tracyMetalZoneBeginBlit(TracyMetalCtxHandle, MTL::BlitPassDescriptor*,
                                           const TracyMetalSrcLoc*) { return nullptr; }
void tracyMetalZoneEnd(TracyMetalGpuZone) {}

#endif // TRACY_ENABLE
