#pragma once

#include <cstdint>

// Tracy Metal GPU profiling bridge.
// TracyMetal.hmm requires ObjC++ and ARC, so we wrap it behind a C++ interface.
// metal-cpp pointers are binary-compatible with ObjC id<MTL...> types.

namespace MTL {
class Device;
class RenderPassDescriptor;
class ComputePassDescriptor;
class BlitPassDescriptor;
}

// Opaque handle to tracy::MetalCtx
using TracyMetalCtxHandle = void*;

// Opaque handle to a GPU zone scope (tracy::MetalZoneScope allocated on heap)
using TracyMetalGpuZone = void*;

// Source location data matching tracy::SourceLocationData layout.
// Used to create static source locations at call sites.
struct TracyMetalSrcLoc {
    const char* name;
    const char* function;
    const char* file;
    uint32_t line;
    uint32_t color;
};

TracyMetalCtxHandle tracyMetalCreate(MTL::Device* device);
void tracyMetalDestroy(TracyMetalCtxHandle ctx);
void tracyMetalCollect(TracyMetalCtxHandle ctx);

// GPU zone begin/end — srcloc must point to static storage
TracyMetalGpuZone tracyMetalZoneBeginRender(TracyMetalCtxHandle ctx,
                                             MTL::RenderPassDescriptor* desc,
                                             const TracyMetalSrcLoc* srcloc);
TracyMetalGpuZone tracyMetalZoneBeginCompute(TracyMetalCtxHandle ctx,
                                              MTL::ComputePassDescriptor* desc,
                                              const TracyMetalSrcLoc* srcloc);
TracyMetalGpuZone tracyMetalZoneBeginBlit(TracyMetalCtxHandle ctx,
                                           MTL::BlitPassDescriptor* desc,
                                           const TracyMetalSrcLoc* srcloc);
void tracyMetalZoneEnd(TracyMetalGpuZone zone);

// Convenience macros that create a static source location at the call site
#define TracyMetalRenderZone(ctx, desc, name) \
    [&]() -> TracyMetalGpuZone { \
        static constexpr TracyMetalSrcLoc srcloc { name, __FUNCTION__, __FILE__, __LINE__, 0 }; \
        return tracyMetalZoneBeginRender(ctx, desc, &srcloc); \
    }()

#define TracyMetalComputeZone(ctx, desc, name) \
    [&]() -> TracyMetalGpuZone { \
        static constexpr TracyMetalSrcLoc srcloc { name, __FUNCTION__, __FILE__, __LINE__, 0 }; \
        return tracyMetalZoneBeginCompute(ctx, desc, &srcloc); \
    }()

#define TracyMetalBlitZone(ctx, desc, name) \
    [&]() -> TracyMetalGpuZone { \
        static constexpr TracyMetalSrcLoc srcloc { name, __FUNCTION__, __FILE__, __LINE__, 0 }; \
        return tracyMetalZoneBeginBlit(ctx, desc, &srcloc); \
    }()
